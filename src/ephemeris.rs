//! Ephemeris binding: the parsed SP3 precise product, arbitrary-epoch
//! position/clock interpolation, per-record state access, and serialization.
//!
//! Marshals SP3 bytes (or a file path) into [`sidereon_core::ephemeris::Sp3`] and
//! exposes its query surface Pythonically: the node epoch axis as J2000 seconds,
//! batched interpolation to numpy `(n, 3)` / `(n,)` arrays, the exact per-record
//! state, the retained clock-only records and header descriptors, and the SP3
//! text writer. No modeling lives here: the interpolation is the engine's
//! `position_at_j2000_seconds` recipe and the writer is `to_sp3_string`, so the
//! numbers and bytes are exactly what `sidereon-core` produces, and a product
//! the writer refuses raises `Sp3WriteError` with the core refusal as its
//! typed `detail`. The per-query loop runs inside Rust, one FFI crossing per
//! call.

use std::path::PathBuf;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyList, PyModule};

use sidereon_core::astro::time::civil::seconds_between_splits;
use sidereon_core::astro::time::{Instant, InstantRepr, JulianDateSplit};
use sidereon_core::constants::J2000_JD;
use sidereon_core::data::ProductDate;
use sidereon_core::ephemeris::{
    align_clock_reference as core_align_clock_reference,
    clock_reference_offset as core_clock_reference_offset,
    observable_states_at_j2000_s as core_observable_states_at_j2000_s,
    observable_states_at_shared_j2000_s as core_observable_states_at_shared_j2000_s,
    parse_exact_sp3 as core_parse_exact_sp3, sample as core_sample,
    validate_exact_sp3 as core_validate_exact_sp3, ClockReferenceOffset, EphemerisSampleRow,
    EphemerisSampleStatus, ExactSp3Coverage, ExactSp3Request, ExactSp3ValidationError,
    MmapPreciseEphemerisInterpolant, ObservableEphemerisSource, ObservableStateBatch,
    ObservableStateElementStatus, ObservablesError, PreciseEphemerisAccuracySample,
    PreciseEphemerisInterpolant, PreciseEphemerisSample, PreciseEphemerisSamples,
    PreciseInterpolantError, PreciseInterpolantStoreError,
    PreciseSamplesError as CorePreciseSamplesError, Sp3, Sp3AccuracyCodeGroup, Sp3AccuracyValue,
    Sp3ClockRecord, Sp3DataType, Sp3Header, Sp3PositionClockAccuracy, Sp3RawRecordAccuracy,
    Sp3RecordAccuracy, Sp3VelocityAccuracy, Sp3Version, Sp3WriteError,
    OBSERVABLE_STATE_MISSING_POSITION_ECEF_M,
};
use sidereon_core::ephemeris::{
    check_continuity, CellSelection, ContinuityDefect, ContinuityOptions, ContinuityOptionsError,
    EpochWindow, MergeCombine, MergeContinuityViolation, OrbitClass, Sp3InterpolationOptions,
    SpeedBound, StencilExtent, UnusableSampleReason, WindowContinuityDecision,
    WindowContinuityVerdict,
};
use sidereon_core::positioning::{ClockRelativity as CoreClockRelativity, EphemerisSource};
use sidereon_core::DigestProvenance;
use sidereon_core::Error as CoreError;
use sidereon_core::GnssSatelliteId;

use crate::core_error_detail::{
    attach_core_error_detail, attach_observables_error_detail, observables_error_detail,
};
use crate::exact_time::PyExactEpochQuery;
use crate::frames::PyTimeScale;
use crate::marshal::{rows3_to_array, PyGnssSystem};
use crate::products::{continuity_cell_role, PySp3Coverage};
use crate::rinex::PyBroadcastEphemeris;
use crate::rinex_clock::PyClockInstant;
use crate::{
    np_array, parse_claimed_checksum64, sp3_write_error_type, to_solve_err, to_sp3_err,
    AccuracySamplesMismatchError, InvalidAccuracyValueError,
    PreciseInterpolantArtifactCorruptError, PreciseInterpolantArtifactError,
    PreciseInterpolantArtifactTruncatedError, PreciseSamplesError,
};

/// Seconds in one day, for the J2000-second <-> split-Julian-date reconstruction.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// A parsed SP3 precise-ephemeris product.
///
/// Construct with [`load_sp3`]. Query satellite states by epoch
/// ([`Sp3.interpolate`] for arbitrary epochs, [`Sp3.state`] for the exact parsed
/// records, [`Sp3.clock_record`] for a clock kept beside a missing orbit), read
/// the node epoch grid with [`Sp3.epochs_j2000_seconds`] and the header with
/// [`Sp3.header`], and serialize back to SP3 text with [`Sp3.to_sp3_string`].
/// Also passed to the solve functions as the ephemeris source. Wraps
/// [`sidereon_core::ephemeris::Sp3`] unchanged.
#[pyclass(module = "sidereon._sidereon", name = "Sp3")]
pub struct PySp3 {
    pub(crate) inner: Sp3,
}

#[pyclass(module = "sidereon._sidereon", name = "EphemerisQueryState")]
#[derive(Clone, Copy)]
pub struct PyEphemerisQueryState {
    position_m: [f64; 3],
    clock_s: f64,
    group_delay_s: Option<f64>,
    degraded_reason: Option<&'static str>,
}

#[pymethods]
impl PyEphemerisQueryState {
    #[getter]
    fn position_ecef_m(&self) -> [f64; 3] {
        self.position_m
    }

    #[getter]
    fn clock_s(&self) -> f64 {
        self.clock_s
    }

    #[getter]
    fn group_delay_s(&self) -> Option<f64> {
        self.group_delay_s
    }

    #[getter]
    fn degraded_reason(&self) -> Option<&'static str> {
        self.degraded_reason
    }
}

#[pyclass(module = "sidereon._sidereon", name = "ClockRelativity")]
#[derive(Clone, Copy)]
pub struct PyClockRelativity {
    inner: CoreClockRelativity,
}

#[pymethods]
impl PyClockRelativity {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreClockRelativity::NotApplicable => "not_applicable",
            CoreClockRelativity::Term(_) => "term",
            CoreClockRelativity::Unavailable => "unavailable",
        }
    }

    #[getter]
    fn term_s(&self) -> Option<f64> {
        self.inner.term()
    }
}

pub(crate) fn source_state_at_epoch_query<Source: EphemerisSource + ?Sized>(
    source: &Source,
    satellite_id: &str,
    epoch: &PyExactEpochQuery,
    selection_epoch: &PyExactEpochQuery,
) -> PyResult<Option<PyEphemerisQueryState>> {
    let satellite = parse_sat(satellite_id)?;
    source
        .try_position_clock_group_delay_selected_at_epoch_query(
            satellite,
            &epoch.inner,
            &selection_epoch.inner,
        )
        .map(|state| {
            state.map(|state| PyEphemerisQueryState {
                position_m: state.value.0,
                clock_s: state.value.1,
                group_delay_s: state.value.2,
                degraded_reason: state.degraded.map(crate::degrade_reason_label),
            })
        })
        .map_err(ephemeris_source_query_error)
}

pub(crate) fn source_transmit_clock_at_epoch_query<Source: EphemerisSource + ?Sized>(
    source: &Source,
    satellite_id: &str,
    epoch: &PyExactEpochQuery,
    selection_epoch: &PyExactEpochQuery,
) -> PyResult<Option<(f64, Option<&'static str>)>> {
    let satellite = parse_sat(satellite_id)?;
    source
        .try_transmit_epoch_clock_at_epoch_query(satellite, &epoch.inner, &selection_epoch.inner)
        .map(|clock| {
            clock.map(|clock| (clock.value, clock.degraded.map(crate::degrade_reason_label)))
        })
        .map_err(ephemeris_source_query_error)
}

pub(crate) fn source_variance_at_epoch_query<Source: EphemerisSource + ?Sized>(
    source: &Source,
    satellite_id: &str,
    state_epoch: &PyExactEpochQuery,
    selection_epoch: &PyExactEpochQuery,
) -> PyResult<f64> {
    let satellite = parse_sat(satellite_id)?;
    Ok(source.ephemeris_variance_at_epoch_query(
        satellite,
        &state_epoch.inner,
        &selection_epoch.inner,
    ))
}

pub(crate) fn source_clock_relativity_at_epoch_query<Source: EphemerisSource + ?Sized>(
    source: &Source,
    satellite_id: &str,
    epoch: &PyExactEpochQuery,
    position_ecef_m: [f64; 3],
) -> PyResult<PyClockRelativity> {
    let satellite = parse_sat(satellite_id)?;
    Ok(PyClockRelativity {
        inner: source.clock_relativity_for_state_at_epoch_query(
            satellite,
            &epoch.inner,
            position_ecef_m,
        ),
    })
}

fn ephemeris_source_query_error(error: sidereon_core::Error) -> PyErr {
    let python_error = match &error {
        sidereon_core::Error::Ut1OutsideCoverage(reason) => {
            crate::ut1_outside_coverage_err("ephemeris source query", *reason)
        }
        _ => to_solve_err(error.clone()),
    };
    attach_core_error_detail(python_error, &error)
}

#[pyclass(module = "sidereon._sidereon", name = "Sp3AccuracyValue")]
#[derive(Clone)]
pub struct PySp3AccuracyValue {
    kind: String,
    value: Option<f64>,
}

impl From<Sp3AccuracyValue> for PySp3AccuracyValue {
    fn from(value: Sp3AccuracyValue) -> Self {
        match value {
            Sp3AccuracyValue::Known(number) => Self {
                kind: "known".to_owned(),
                value: Some(number),
            },
            Sp3AccuracyValue::Unknown => Self {
                kind: "unknown".to_owned(),
                value: None,
            },
            Sp3AccuracyValue::TooLarge => Self {
                kind: "too_large".to_owned(),
                value: None,
            },
            Sp3AccuracyValue::InvalidBase => Self {
                kind: "invalid_base".to_owned(),
                value: None,
            },
            Sp3AccuracyValue::Overflow => Self {
                kind: "overflow".to_owned(),
                value: None,
            },
            other => Self {
                kind: format!("other:{other:?}"),
                value: None,
            },
        }
    }
}

impl PySp3AccuracyValue {
    fn to_core(&self) -> PyResult<Sp3AccuracyValue> {
        match (self.kind.as_str(), self.value) {
            ("known", Some(value)) if value.is_finite() && value >= 0.0 => {
                Ok(Sp3AccuracyValue::Known(value))
            }
            ("known", _) => Err(PyValueError::new_err(
                "known accuracy must be finite and non-negative",
            )),
            ("unknown", None) => Ok(Sp3AccuracyValue::Unknown),
            ("too_large", None) => Ok(Sp3AccuracyValue::TooLarge),
            ("invalid_base", None) => Ok(Sp3AccuracyValue::InvalidBase),
            ("overflow", None) => Ok(Sp3AccuracyValue::Overflow),
            _ => Err(PyValueError::new_err("invalid accuracy outcome")),
        }
    }
}

#[pymethods]
impl PySp3AccuracyValue {
    #[staticmethod]
    fn known(value: f64) -> PyResult<Self> {
        if !value.is_finite() || value < 0.0 {
            return Err(PyValueError::new_err(
                "known accuracy must be finite and non-negative",
            ));
        }
        Ok(Self {
            kind: "known".to_owned(),
            value: Some(value),
        })
    }

    #[staticmethod]
    fn unknown() -> Self {
        Self {
            kind: "unknown".to_owned(),
            value: None,
        }
    }

    #[staticmethod]
    fn too_large() -> Self {
        Self {
            kind: "too_large".to_owned(),
            value: None,
        }
    }

    #[staticmethod]
    fn invalid_base() -> Self {
        Self {
            kind: "invalid_base".to_owned(),
            value: None,
        }
    }

    #[staticmethod]
    fn overflow() -> Self {
        Self {
            kind: "overflow".to_owned(),
            value: None,
        }
    }

    #[getter]
    fn kind(&self) -> String {
        self.kind.clone()
    }

    #[getter]
    fn value(&self) -> Option<f64> {
        self.value
    }

    fn variance(&self) -> Self {
        self.to_core()
            .map(Sp3AccuracyValue::variance)
            .unwrap_or(Sp3AccuracyValue::Overflow)
            .into()
    }

    fn __repr__(&self) -> String {
        match self.value {
            Some(value) => format!("Sp3AccuracyValue(kind={:?}, value={value})", self.kind),
            None => format!("Sp3AccuracyValue(kind={:?})", self.kind),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Sp3AccuracyCodeGroup")]
#[derive(Clone, Copy)]
pub struct PySp3AccuracyCodeGroup {
    inner: Sp3AccuracyCodeGroup,
}

impl From<Sp3AccuracyCodeGroup> for PySp3AccuracyCodeGroup {
    fn from(inner: Sp3AccuracyCodeGroup) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySp3AccuracyCodeGroup {
    #[getter]
    fn axis_exponents(&self) -> [Option<i16>; 3] {
        self.inner.axis_exponents
    }
    #[getter]
    fn clock_exponent(&self) -> Option<i16> {
        self.inner.clock_exponent
    }
    #[getter]
    fn position_velocity_base(&self) -> Option<f64> {
        self.inner.position_velocity_base
    }
    #[getter]
    fn clock_rate_base(&self) -> Option<f64> {
        self.inner.clock_rate_base
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Sp3RawRecordAccuracy")]
#[derive(Clone, Copy)]
pub struct PySp3RawRecordAccuracy {
    inner: Sp3RawRecordAccuracy,
}

impl From<Sp3RawRecordAccuracy> for PySp3RawRecordAccuracy {
    fn from(inner: Sp3RawRecordAccuracy) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySp3RawRecordAccuracy {
    #[getter]
    fn p(&self) -> Option<PySp3AccuracyCodeGroup> {
        self.inner.p.map(Into::into)
    }
    #[getter]
    fn v(&self) -> Option<PySp3AccuracyCodeGroup> {
        self.inner.v.map(Into::into)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Sp3PositionClockAccuracy")]
#[derive(Clone, Copy)]
pub struct PySp3PositionClockAccuracy {
    inner: Sp3PositionClockAccuracy,
}

impl From<Sp3PositionClockAccuracy> for PySp3PositionClockAccuracy {
    fn from(inner: Sp3PositionClockAccuracy) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySp3PositionClockAccuracy {
    #[getter]
    fn position_sigma_m(&self) -> [PySp3AccuracyValue; 3] {
        self.inner.position_sigma_m.map(Into::into)
    }
    #[getter]
    fn clock_sigma_m(&self) -> PySp3AccuracyValue {
        self.inner.clock_sigma_m.into()
    }
    fn position_variance_m2(&self) -> [PySp3AccuracyValue; 3] {
        self.inner.position_variance_m2().map(Into::into)
    }
    fn clock_variance_m2(&self) -> PySp3AccuracyValue {
        self.inner.clock_variance_m2().into()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Sp3VelocityAccuracy")]
#[derive(Clone, Copy)]
pub struct PySp3VelocityAccuracy {
    inner: Sp3VelocityAccuracy,
}

impl From<Sp3VelocityAccuracy> for PySp3VelocityAccuracy {
    fn from(inner: Sp3VelocityAccuracy) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySp3VelocityAccuracy {
    #[getter]
    fn velocity_sigma_m_s(&self) -> [PySp3AccuracyValue; 3] {
        self.inner.velocity_sigma_m_s.map(Into::into)
    }
    #[getter]
    fn clock_rate_sigma_m_s(&self) -> PySp3AccuracyValue {
        self.inner.clock_rate_sigma_m_s.into()
    }
    fn velocity_variance_m2_s2(&self) -> [PySp3AccuracyValue; 3] {
        self.inner.velocity_variance_m2_s2().map(Into::into)
    }
    fn clock_rate_variance_m2_s2(&self) -> PySp3AccuracyValue {
        self.inner.clock_rate_variance_m2_s2().into()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Sp3RecordAccuracy")]
#[derive(Clone, Copy)]
pub struct PySp3RecordAccuracy {
    inner: Sp3RecordAccuracy,
}

impl From<Sp3RecordAccuracy> for PySp3RecordAccuracy {
    fn from(inner: Sp3RecordAccuracy) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySp3RecordAccuracy {
    #[getter]
    fn p(&self) -> Option<PySp3PositionClockAccuracy> {
        self.inner.p.map(Into::into)
    }
    #[getter]
    fn v(&self) -> Option<PySp3VelocityAccuracy> {
        self.inner.v.map(Into::into)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PreciseEphemerisAccuracySample")]
#[derive(Clone, Copy)]
pub struct PyPreciseEphemerisAccuracySample {
    inner: PreciseEphemerisAccuracySample,
}

impl From<PreciseEphemerisAccuracySample> for PyPreciseEphemerisAccuracySample {
    fn from(inner: PreciseEphemerisAccuracySample) -> Self {
        Self { inner }
    }
}

impl PyPreciseEphemerisAccuracySample {
    fn to_core(self) -> PreciseEphemerisAccuracySample {
        self.inner
    }
}

#[pymethods]
impl PyPreciseEphemerisAccuracySample {
    #[staticmethod]
    #[pyo3(signature = (satellite, epoch_j2000_seconds, position_variance_m2, clock_variance_m2, *, time_scale=PyTimeScale::GPST))]
    fn new(
        py: Python<'_>,
        satellite: &str,
        epoch_j2000_seconds: f64,
        position_variance_m2: Vec<Py<PySp3AccuracyValue>>,
        clock_variance_m2: Py<PySp3AccuracyValue>,
        time_scale: PyTimeScale,
    ) -> PyResult<Self> {
        if position_variance_m2.len() != 3 {
            return Err(PyValueError::new_err(
                "position_variance_m2 must contain three components",
            ));
        }
        let mut variances = [Sp3AccuracyValue::Unknown; 3];
        for (axis_index, variance) in position_variance_m2.iter().enumerate() {
            variances[axis_index] = variance.borrow(py).to_core()?;
        }
        let inner = PreciseEphemerisAccuracySample::new(
            parse_sat(satellite)?,
            instant_from_j2000_seconds(epoch_j2000_seconds, time_scale.into())?,
            variances,
            clock_variance_m2.borrow(py).to_core()?,
        );
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_instant(
        py: Python<'_>,
        satellite: &str,
        epoch: &PyClockInstant,
        position_variance_m2: Vec<Py<PySp3AccuracyValue>>,
        clock_variance_m2: Py<PySp3AccuracyValue>,
    ) -> PyResult<Self> {
        if position_variance_m2.len() != 3 {
            return Err(PyValueError::new_err(
                "position_variance_m2 must contain three components",
            ));
        }
        let mut variances = [Sp3AccuracyValue::Unknown; 3];
        for (axis_index, variance) in position_variance_m2.iter().enumerate() {
            variances[axis_index] = variance.borrow(py).to_core()?;
        }
        let inner = PreciseEphemerisAccuracySample::new(
            parse_sat(satellite)?,
            epoch.to_core(),
            variances,
            clock_variance_m2.borrow(py).to_core()?,
        );
        Ok(Self { inner })
    }

    #[getter]
    fn satellite(&self) -> String {
        self.inner.sat.to_string()
    }
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        instant_to_j2000_seconds(&self.inner.epoch).unwrap_or(f64::NAN)
    }
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.epoch)
    }
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.epoch.scale.into()
    }
    #[getter]
    fn position_variance_m2(&self) -> [PySp3AccuracyValue; 3] {
        self.inner.position_variance_m2.map(Into::into)
    }
    #[getter]
    fn clock_variance_m2(&self) -> PySp3AccuracyValue {
        self.inner.clock_variance_m2.into()
    }
}

/// Prediction status aggregated over every satellite record at one SP3 epoch.
#[pyclass(module = "sidereon._sidereon", name = "Sp3EpochPrediction")]
#[derive(Clone)]
pub struct PySp3EpochPrediction {
    epoch_j2000_seconds: f64,
    orbit_predicted_satellites: Vec<String>,
    clock_predicted_satellites: Vec<String>,
}

#[pymethods]
impl PySp3EpochPrediction {
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        self.epoch_j2000_seconds
    }

    #[getter]
    fn observed(&self) -> bool {
        self.orbit_predicted_satellites.is_empty() && self.clock_predicted_satellites.is_empty()
    }

    #[getter]
    fn orbit_predicted_satellites(&self) -> Vec<String> {
        self.orbit_predicted_satellites.clone()
    }

    #[getter]
    fn clock_predicted_satellites(&self) -> Vec<String> {
        self.clock_predicted_satellites.clone()
    }
}

/// Product-wide prediction metadata derived from SP3 record flags.
#[pyclass(module = "sidereon._sidereon", name = "Sp3PredictionSummary")]
#[derive(Clone)]
pub struct PySp3PredictionSummary {
    epochs: Vec<PySp3EpochPrediction>,
    observed_through_j2000_seconds: Option<f64>,
}

#[pymethods]
impl PySp3PredictionSummary {
    #[getter]
    fn epochs(&self) -> Vec<PySp3EpochPrediction> {
        self.epochs.clone()
    }

    #[getter]
    fn observed_through_j2000_seconds(&self) -> Option<f64> {
        self.observed_through_j2000_seconds
    }
}

impl PySp3 {
    /// Wrap an owned core product, for the staleness selection layer which hands
    /// back the selected (present or nearest-prior) product.
    pub(crate) fn from_sp3(inner: Sp3) -> Self {
        Self { inner }
    }
}

/// Parse a satellite token (e.g. `"G01"`) into a typed id, raising `ValueError`
/// on a malformed token (bad input, never a domain error).
pub(crate) fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|e| PyValueError::new_err(format!("invalid satellite token {token:?}: {e}")))
}

fn precise_samples_error(py: Python<'_>, error: CorePreciseSamplesError) -> PyErr {
    let (error_type, kind, satellite) = match &error {
        CorePreciseSamplesError::Empty => (py.get_type::<PreciseSamplesError>(), "empty", None),
        CorePreciseSamplesError::SingleSampleSatellite(satellite) => (
            py.get_type::<PreciseSamplesError>(),
            "single_sample_satellite",
            Some(satellite.to_string()),
        ),
        CorePreciseSamplesError::NonMonotonicEpochs(satellite) => (
            py.get_type::<PreciseSamplesError>(),
            "non_monotonic_epochs",
            Some(satellite.to_string()),
        ),
        CorePreciseSamplesError::MixedTimeScales => (
            py.get_type::<PreciseSamplesError>(),
            "mixed_time_scales",
            None,
        ),
        CorePreciseSamplesError::EpochNotRepresentable(satellite) => (
            py.get_type::<PreciseSamplesError>(),
            "epoch_not_representable",
            Some(satellite.to_string()),
        ),
        CorePreciseSamplesError::NonFiniteSample(satellite) => (
            py.get_type::<PreciseSamplesError>(),
            "non_finite_sample",
            Some(satellite.to_string()),
        ),
        CorePreciseSamplesError::AccuracySamplesMismatch => (
            py.get_type::<AccuracySamplesMismatchError>(),
            "accuracy_samples_mismatch",
            None,
        ),
        CorePreciseSamplesError::InvalidAccuracyValue(satellite) => (
            py.get_type::<InvalidAccuracyValueError>(),
            "invalid_accuracy_value",
            Some(satellite.to_string()),
        ),
        _ => (py.get_type::<PreciseSamplesError>(), "unknown", None),
    };
    let python_error = PyErr::from_type(error_type, error.to_string());
    let value = python_error.value(py);
    if let Err(set_error) = value.setattr("kind", kind) {
        return set_error;
    }
    if let Err(set_error) = value.setattr("satellite", satellite) {
        return set_error;
    }
    python_error
}

pub(crate) fn continuity_options(
    py: Python<'_>,
    orbit_class: Option<&str>,
    residual_tolerance_m: Option<f64>,
    gap_threshold_factor: Option<f64>,
) -> PyResult<ContinuityOptions> {
    let speed_bound = match orbit_class {
        None => None,
        Some("meo_gnss") => Some(SpeedBound::OrbitClass(OrbitClass::MeoGnss)),
        Some("geosynchronous") => Some(SpeedBound::OrbitClass(OrbitClass::Geosynchronous)),
        Some("leo") => Some(SpeedBound::OrbitClass(OrbitClass::Leo)),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "unknown orbit class: {other}"
            )))
        }
    };
    let mut options = ContinuityOptions::new(speed_bound, residual_tolerance_m)
        .map_err(|error| continuity_options_error(py, error))?;
    options.speed_bound = speed_bound;
    options.residual_tolerance_m = residual_tolerance_m;
    if let Some(factor) = gap_threshold_factor {
        let interpolation = Sp3InterpolationOptions::new(factor)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        options = options.with_interpolation_options(interpolation);
    }
    Ok(options)
}

fn continuity_options_error(py: Python<'_>, error: ContinuityOptionsError) -> PyErr {
    let python_error = PyValueError::new_err(error.to_string());
    let value = python_error.value(py);
    let _ = value.setattr("kind", "continuity_options");
    let _ = value.setattr("field", error.field);
    // A string retains NaN/infinity without introducing non-JSON numeric values.
    let _ = value.setattr("value", format!("{:?}", error.value));
    let _ = value.setattr("reason", format!("{:?}", error.reason));
    python_error
}

fn continuity_defect_to_dict<'py>(
    py: Python<'py>,
    defect: &ContinuityDefect,
) -> PyResult<Bound<'py, PyDict>> {
    let entry = PyDict::new(py);
    let (kind, from_s, to_s, magnitude, bound) = match defect {
        ContinuityDefect::DuplicateEpoch {
            epoch_j2000_s,
            occurrences,
            ..
        } => (
            "duplicate_epoch",
            Some(*epoch_j2000_s),
            Some(*epoch_j2000_s),
            Some(*occurrences as f64),
            None,
        ),
        ContinuityDefect::SingleSampleSeries { .. } => {
            ("single_sample_series", None, None, None, None)
        }
        ContinuityDefect::UnusableSample { epoch_j2000_s, .. } => (
            "unusable_sample",
            *epoch_j2000_s,
            *epoch_j2000_s,
            None,
            None,
        ),
        ContinuityDefect::SpeedBound {
            from_j2000_s,
            to_j2000_s,
            implied_speed_m_s,
            bound_m_s,
            ..
        } => (
            "speed_bound",
            Some(*from_j2000_s),
            Some(*to_j2000_s),
            Some(*implied_speed_m_s),
            Some(*bound_m_s),
        ),
        ContinuityDefect::HoldOutResidual {
            preceding_j2000_s,
            epoch_j2000_s,
            residual_m,
            tolerance_m,
            ..
        } => (
            "hold_out_residual",
            Some(*preceding_j2000_s),
            Some(*epoch_j2000_s),
            Some(*residual_m),
            Some(*tolerance_m),
        ),
    };
    entry.set_item("kind", kind)?;
    entry.set_item("satellite", defect.satellite().to_string())?;
    entry.set_item("from_j2000_s", from_s)?;
    entry.set_item("to_j2000_s", to_s)?;
    entry.set_item("magnitude", magnitude)?;
    entry.set_item("bound", bound)?;
    // Every field of the variant under its core name, beside the flattened
    // summary above, so the dict loses nothing the defect states.
    match defect {
        ContinuityDefect::DuplicateEpoch {
            epoch_j2000_s,
            occurrences,
            ..
        } => {
            entry.set_item("epoch_j2000_s", *epoch_j2000_s)?;
            entry.set_item("occurrences", *occurrences)?;
        }
        ContinuityDefect::SingleSampleSeries { .. } => {}
        ContinuityDefect::UnusableSample {
            sample_index,
            epoch_j2000_s,
            reason,
            ..
        } => {
            entry.set_item("sample_index", *sample_index)?;
            entry.set_item("epoch_j2000_s", *epoch_j2000_s)?;
            entry.set_item(
                "reason",
                match reason {
                    UnusableSampleReason::EpochNotPlaced => "epoch_not_placed",
                    UnusableSampleReason::NonFinitePosition => "non_finite_position",
                    _ => "unknown",
                },
            )?;
        }
        ContinuityDefect::SpeedBound {
            interval_s,
            displacement_m,
            implied_speed_m_s,
            bound_m_s,
            ..
        } => {
            entry.set_item("interval_s", *interval_s)?;
            entry.set_item("displacement_m", *displacement_m)?;
            entry.set_item("implied_speed_m_s", *implied_speed_m_s)?;
            entry.set_item("bound_m_s", *bound_m_s)?;
        }
        ContinuityDefect::HoldOutResidual {
            epoch_j2000_s,
            preceding_j2000_s,
            residual_m,
            tolerance_m,
            node_epochs_j2000_s,
            ..
        } => {
            entry.set_item("epoch_j2000_s", *epoch_j2000_s)?;
            entry.set_item("preceding_j2000_s", *preceding_j2000_s)?;
            entry.set_item("residual_m", *residual_m)?;
            entry.set_item("tolerance_m", *tolerance_m)?;
            entry.set_item("node_epochs_j2000_s", node_epochs_j2000_s.clone())?;
        }
    }
    Ok(entry)
}

fn cell_selection_to_dict<'py>(
    py: Python<'py>,
    selection: &CellSelection,
) -> PyResult<Bound<'py, PyDict>> {
    let entry = PyDict::new(py);
    let (kind, rule) = match selection {
        CellSelection::SingleSource { .. } => ("single_source", None),
        CellSelection::Precedence { .. } => ("precedence", None),
        CellSelection::Combined { rule, .. } => (
            "combined",
            Some(match rule {
                MergeCombine::Mean => "mean",
                MergeCombine::Median => "median",
                MergeCombine::Precedence => "precedence",
            }),
        ),
    };
    entry.set_item("kind", kind)?;
    entry.set_item("selected_source", selection.selected_source())?;
    entry.set_item("members", selection.members())?;
    entry.set_item("rule", rule)?;
    Ok(entry)
}

fn continuity_defects_to_list<'py, 'a>(
    py: Python<'py>,
    defects: impl IntoIterator<Item = &'a ContinuityDefect>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for defect in defects {
        out.append(continuity_defect_to_dict(py, defect)?)?;
    }
    Ok(out)
}

fn continuity_violation_to_dict<'py>(
    py: Python<'py>,
    violation: &MergeContinuityViolation,
) -> PyResult<Bound<'py, PyDict>> {
    let entry = PyDict::new(py);
    entry.set_item("defect", continuity_defect_to_dict(py, &violation.defect)?)?;
    entry.set_item("from_sources", &violation.from_sources)?;
    entry.set_item("to_sources", &violation.to_sources)?;
    entry.set_item("crosses_contributors", violation.crosses_contributors)?;
    let cells = PyList::empty(py);
    for cell in &violation.cells {
        let cell_entry = PyDict::new(py);
        cell_entry.set_item("epoch_j2000_s", cell.epoch_j2000_s)?;
        cell_entry.set_item("role", continuity_cell_role(cell.role))?;
        match &cell.selection {
            Some(selection) => {
                cell_entry.set_item("selection", cell_selection_to_dict(py, selection)?)?
            }
            None => cell_entry.set_item("selection", py.None())?,
        }
        cells.append(cell_entry)?;
    }
    entry.set_item("cells", cells)?;
    entry.set_item("sources", &violation.sources)?;
    Ok(entry)
}

fn continuity_violations_to_list<'py, 'a>(
    py: Python<'py>,
    violations: impl IntoIterator<Item = &'a MergeContinuityViolation>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for violation in violations {
        out.append(continuity_violation_to_dict(py, violation)?)?;
    }
    Ok(out)
}

pub(crate) fn continuity_verdict_to_py(
    py: Python<'_>,
    verdict: WindowContinuityVerdict<'_>,
) -> PyResult<PyObject> {
    let out = PyDict::new(py);
    out.set_item(
        "decision",
        match verdict.decision {
            WindowContinuityDecision::Accept => "accept",
            WindowContinuityDecision::Refuse => "refuse",
        },
    )?;
    out.set_item("accepted", verdict.accepted())?;
    out.set_item(
        "influencing_defects",
        continuity_defects_to_list(py, verdict.influencing_defects)?,
    )?;
    out.set_item(
        "influencing_splices",
        continuity_violations_to_list(py, verdict.influencing_splices)?,
    )?;
    out.set_item(
        "all_defects",
        continuity_defects_to_list(py, verdict.all_defects)?,
    )?;
    out.set_item(
        "all_splices",
        continuity_violations_to_list(py, verdict.all_splices)?,
    )?;
    Ok(out.into())
}

#[pymethods]
impl PySp3 {
    /// Number of epochs in the product.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.inner.epoch_count()
    }

    /// Epoch count declared on SP3 header line 1.
    #[getter]
    fn declared_epoch_count(&self) -> u64 {
        self.inner.declared_epoch_count()
    }

    /// Start epoch declared on SP3 header line 1, as J2000 seconds.
    #[getter]
    fn declared_start_j2000_s(&self) -> Option<f64> {
        self.inner.declared_start_j2000_s()
    }

    /// The satellite tokens (e.g. `"G01"`) present in the product, ascending.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner
            .satellites()
            .iter()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// SP3 interpolation gap threshold factor carried by this product.
    #[getter]
    fn gap_threshold_factor(&self) -> f64 {
        self.inner.interpolation_options().gap_threshold_factor()
    }

    /// Return a copy of this product with an explicit gap threshold factor.
    fn with_interpolation_options(&self, gap_threshold_factor: f64) -> PyResult<Self> {
        let options = Sp3InterpolationOptions::new(gap_threshold_factor)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: self.inner.clone().with_interpolation_options(options),
        })
    }

    /// The product's parsed epochs as seconds since J2000, in the file's own time
    /// scale, ascending, as a numpy `(n,)` `float64` array.
    ///
    /// This is the exact query axis [`Sp3.interpolate`] consumes; read it, form
    /// query times on it (e.g. midpoints, a finer grid), and pass them straight
    /// back without a Julian-date round-trip.
    /// Attest that this product is physically continuous, or report each
    /// violation.
    ///
    /// A merged product is assembled per satellite and epoch from several
    /// analysis centers, which is exactly the operation that can splice two
    /// physically inconsistent arcs together while every input stays
    /// individually well-formed. Two checks run, with different jobs: a
    /// physical earth-fixed speed gate whose bound is a true upper bound for
    /// the orbit class, so it cannot false-positive and catches gross
    /// corruption; and a hold-out interpolation residual, which supplies the
    /// sensitivity a speed gate structurally cannot (adjacent GNSS MEO epochs
    /// are hundreds of kilometres apart, so a metre-scale splice moves the
    /// implied speed by a fraction of a percent).
    ///
    /// `orbit_class` is one of `"meo_gnss"` (default), `"geosynchronous"`,
    /// `"leo"`, or `None` to disable the speed gate.
    /// `residual_tolerance_m` enables the residual check; `None` disables it.
    /// `gap_threshold_factor` configures the hold-out interpolation policy;
    /// `None` leaves the core default (1.5).
    ///
    /// Returns a dict with `defects`, `attested`, and the counts of what was
    /// examined, so "checked and clean" stays distinguishable from "not
    /// checked". This reports rather than refuses: whether a product with
    /// defects is acceptable is the caller's decision.
    #[pyo3(signature = (
        orbit_class = "meo_gnss",
        residual_tolerance_m = Some(1.0),
        gap_threshold_factor = None,
    ))]
    fn check_continuity(
        &self,
        py: Python<'_>,
        orbit_class: Option<&str>,
        residual_tolerance_m: Option<f64>,
        gap_threshold_factor: Option<f64>,
    ) -> PyResult<PyObject> {
        let options =
            continuity_options(py, orbit_class, residual_tolerance_m, gap_threshold_factor)?;
        let report = check_continuity(&self.inner.precise_ephemeris_samples(), &options)
            .map_err(|error| continuity_options_error(py, error))?;

        let defects = continuity_defects_to_list(py, &report.defects)?;

        let out = pyo3::types::PyDict::new(py);
        out.set_item("attested", report.attested())?;
        out.set_item("defects", defects)?;
        out.set_item("pairs_checked", report.pairs_checked)?;
        out.set_item("residuals_checked", report.residuals_checked)?;
        out.set_item("residuals_skipped", report.residuals_skipped)?;
        Ok(out.into())
    }

    /// Time reach of the SP3 position interpolator before and after a query.
    ///
    /// The values are derived by the core from this product's declared epoch
    /// interval and interpolation-node count; callers do not supply a stencil
    /// duration.
    fn stencil_extent(&self) -> PyResult<(f64, f64)> {
        let stencil = StencilExtent::for_sp3(&self.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok((stencil.before_s(), stencil.after_s()))
    }

    /// Decide whether product-wide continuity findings can influence an
    /// inclusive evaluation window through this product's interpolation
    /// stencil.
    #[pyo3(signature = (
        from_j2000_s,
        through_j2000_s,
        orbit_class = "meo_gnss",
        residual_tolerance_m = Some(1.0),
        gap_threshold_factor = None,
    ))]
    fn continuity_verdict(
        &self,
        py: Python<'_>,
        from_j2000_s: f64,
        through_j2000_s: f64,
        orbit_class: Option<&str>,
        residual_tolerance_m: Option<f64>,
        gap_threshold_factor: Option<f64>,
    ) -> PyResult<PyObject> {
        let window = EpochWindow::new(from_j2000_s, through_j2000_s)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let stencil = StencilExtent::for_sp3(&self.inner)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        let options =
            continuity_options(py, orbit_class, residual_tolerance_m, gap_threshold_factor)?;
        let report = check_continuity(&self.inner.precise_ephemeris_samples(), &options)
            .map_err(|error| continuity_options_error(py, error))?;
        continuity_verdict_to_py(py, report.verdict_for_window(window, stencil))
    }

    #[getter]
    fn epochs_j2000_seconds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.epochs_j2000_seconds())
    }

    /// The product's parsed epochs in file order, each scale-tagged in the
    /// core's own representation, so two products' epochs compare exactly.
    #[getter]
    fn epoch_instants(&self) -> Vec<PyClockInstant> {
        self.inner
            .epochs
            .iter()
            .copied()
            .map(PyClockInstant::from_core)
            .collect()
    }

    /// Position and clock coverage of every satellite, with the grid the
    /// product's epochs lie on.
    ///
    /// Every satellite the header declares is listed, including one with no
    /// record, and so is every satellite with a record the header does not
    /// declare. A clock-only record counts as a clock and not a position; a
    /// position record with the missing-clock sentinel counts as a position
    /// and not a clock. Coverage states which records exist, not where an
    /// interpolation is served.
    fn satellite_coverage(&self) -> PySp3Coverage {
        PySp3Coverage::from(self.inner.satellite_coverage())
    }

    /// Per-epoch observed/predicted status and the contiguous observed-through
    /// boundary, derived from actual SP3 record flags.
    fn prediction_summary(&self) -> PySp3PredictionSummary {
        let summary = self.inner.prediction_summary();
        PySp3PredictionSummary {
            epochs: summary
                .epochs
                .into_iter()
                .map(|epoch| PySp3EpochPrediction {
                    epoch_j2000_seconds: instant_to_j2000_seconds(&epoch.epoch).unwrap_or(f64::NAN),
                    orbit_predicted_satellites: epoch
                        .orbit_predicted_satellites
                        .into_iter()
                        .map(|satellite| satellite.to_string())
                        .collect(),
                    clock_predicted_satellites: epoch
                        .clock_predicted_satellites
                        .into_iter()
                        .map(|satellite| satellite.to_string())
                        .collect(),
                })
                .collect(),
            observed_through_j2000_seconds: summary
                .observed_through
                .as_ref()
                .and_then(instant_to_j2000_seconds),
        }
    }

    /// Interpolate `satellite`'s position and clock at each query epoch.
    ///
    /// `j2000_seconds` is a 1-D `float64` array of query times in seconds since
    /// J2000, in the product's own time scale (see
    /// [`epochs_j2000_seconds`](Self::epochs_j2000_seconds)). Returns a
    /// [`Sp3Interpolation`] whose `position_m` is a numpy `(n, 3)` ECEF array in
    /// metres and `clock_s` is a numpy `(n,)` array in seconds (NaN where the
    /// satellite has no clock estimate at that epoch).
    ///
    /// Raises `ValueError` if `satellite` is not present in the product or the
    /// query array is empty, and `SolveError` if a query lies outside the
    /// satellite's coverage (the engine refuses to interpolate across a gap rather
    /// than returning a diverging extrapolation).
    fn interpolate(
        &self,
        satellite: &str,
        j2000_seconds: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PySp3Interpolation> {
        let sat = parse_sat(satellite)?;
        let queries = j2000_seconds
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if queries.is_empty() {
            return Err(PyValueError::new_err("j2000_seconds array is empty"));
        }

        let mut positions = Vec::with_capacity(queries.len());
        let mut clocks = Vec::with_capacity(queries.len());
        for &q in queries {
            let state = self
                .inner
                .position_at_j2000_seconds(sat, q)
                .map_err(|error| {
                    let python_error = match &error {
                        // The satellite simply is not in the product: bad input.
                        CoreError::UnknownSatellite(id) => {
                            PyValueError::new_err(format!("satellite {id} is not in the product"))
                        }
                        // Out of coverage / too few nodes: a solve condition.
                        other => {
                            to_solve_err(format!("interpolation at j2000 second {q}: {other}"))
                        }
                    };
                    attach_core_error_detail(python_error, &error)
                })?;
            positions.push(state.position.as_array());
            clocks.push(state.clock_s.unwrap_or(f64::NAN));
        }
        Ok(PySp3Interpolation { positions, clocks })
    }

    fn position_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
    ) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        self.inner
            .position_at_epoch_query(sat, &epoch.inner)
            .map(PySp3State::from_state)
            .map_err(|error| attach_core_error_detail(to_solve_err(error.to_string()), &error))
    }

    fn selected_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<PyEphemerisQueryState>> {
        source_state_at_epoch_query(&self.inner, satellite, epoch, selection_epoch)
    }

    fn transmit_epoch_clock_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<(f64, Option<&'static str>)>> {
        source_transmit_clock_at_epoch_query(&self.inner, satellite, epoch, selection_epoch)
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        satellite: &str,
        state_epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<f64> {
        source_variance_at_epoch_query(&self.inner, satellite, state_epoch, selection_epoch)
    }

    fn clock_relativity_for_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        position_ecef_m: [f64; 3],
    ) -> PyResult<PyClockRelativity> {
        source_clock_relativity_at_epoch_query(&self.inner, satellite, epoch, position_ecef_m)
    }

    /// The exact parsed state of `satellite` at the record with index
    /// `epoch_index` (no interpolation).
    ///
    /// Returns an [`Sp3State`]. Raises `IndexError` if `epoch_index` is past the
    /// last epoch and `KeyError` if the satellite has no state at that epoch. A
    /// record whose orbit is the missing-orbit sentinel beside a valid clock is
    /// not a state: read it with [`Sp3.clock_record`].
    fn state(&self, satellite: &str, epoch_index: usize) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        let state = self
            .inner
            .state(sat, epoch_index)
            .map_err(|e| record_lookup_error(e, epoch_index, "record"))?;
        Ok(PySp3State::from_state(state))
    }

    /// The retained clock-only record of `satellite` at the record with index
    /// `epoch_index`: a record whose orbit is the missing-orbit sentinel
    /// (`0.0 0.0 0.0`) beside a valid clock estimate.
    ///
    /// Returns an [`Sp3ClockRecord`]. Raises `IndexError` if `epoch_index` is
    /// past the last epoch and `KeyError` if the satellite has no clock-only
    /// record at that epoch. A satellite with an orbit at that epoch has a state
    /// ([`Sp3.state`]) and no clock-only record.
    fn clock_record(&self, satellite: &str, epoch_index: usize) -> PyResult<PySp3ClockRecord> {
        let sat = parse_sat(satellite)?;
        let record = self
            .inner
            .clock_record(sat, epoch_index)
            .map_err(|e| record_lookup_error(e, epoch_index, "clock-only record"))?;
        Ok(PySp3ClockRecord::from(record))
    }

    fn record_accuracy_codes(
        &self,
        satellite: &str,
        epoch_index: usize,
    ) -> PyResult<PySp3RawRecordAccuracy> {
        self.inner
            .record_accuracy_codes(parse_sat(satellite)?, epoch_index)
            .map(Into::into)
            .map_err(|error| record_lookup_error(error, epoch_index, "accuracy record"))
    }

    fn record_accuracy(
        &self,
        satellite: &str,
        epoch_index: usize,
    ) -> PyResult<PySp3RecordAccuracy> {
        self.inner
            .record_accuracy(parse_sat(satellite)?, epoch_index)
            .map(Into::into)
            .map_err(|error| record_lookup_error(error, epoch_index, "accuracy record"))
    }

    /// Every clock-only record at the epoch with index `epoch_index`, as a dict
    /// from satellite token to [`Sp3ClockRecord`] in ascending satellite order.
    /// An epoch with none returns an empty dict. Raises `IndexError` if
    /// `epoch_index` is past the last epoch.
    fn clock_records_at<'py>(
        &self,
        py: Python<'py>,
        epoch_index: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let records = self
            .inner
            .clock_records_at(epoch_index)
            .map_err(|e| record_lookup_error(e, epoch_index, "clock-only record"))?;
        let out = PyDict::new(py);
        for (sat, record) in records {
            out.set_item(sat.to_string(), PySp3ClockRecord::from(*record))?;
        }
        Ok(out)
    }

    /// The parsed header: format version and record type, the line-1
    /// descriptors, the line-2 timing fields, the first `%c` line's file type
    /// and time system, the first `%f` line's bases, and the satellite list
    /// with its accuracy codes.
    #[getter]
    fn header(&self) -> PySp3Header {
        PySp3Header {
            inner: self.inner.header.clone(),
        }
    }

    /// The text of every `/*` comment record that carries text, in file order,
    /// with trailing blanks removed as the reader removes them.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Entries the parser skipped rather than stored: records and `+` header
    /// declarations whose satellite token names no representable satellite,
    /// and `EP`/`EV` correlation records, which the product does not model.
    #[getter]
    fn skipped_records(&self) -> usize {
        self.inner.skipped_records
    }

    /// Serialize this product to standard SP3 text (the format named by its
    /// header version). Pure and deterministic: the same product always
    /// produces byte-identical text, and re-parsing it yields every value the
    /// product holds, bit for bit.
    ///
    /// Raises `Sp3WriteError`, a subclass of `Sp3ParseError` and `ValueError`,
    /// when a value cannot be stated in its fixed columns without changing it:
    /// a record value finer than the `F14.6` column (the usual case for a mean
    /// or median merge), a value that would read back as an absence sentinel,
    /// a header base finer than its field, text wider than its columns, or an
    /// epoch no record restates exactly. Nothing is rounded, shifted or dropped
    /// to make the write succeed. The exception's `detail` is an
    /// [`Sp3WriteErrorDetail`] naming the core refusal and its fields.
    fn to_sp3_string(&self, py: Python<'_>) -> PyResult<String> {
        self.inner
            .to_sp3_string()
            .map_err(|err| to_sp3_write_err(py, err))
    }

    /// Extract this product as the canonical precise-ephemeris samples, in SI
    /// units, one per parsed position record in ascending epoch order.
    ///
    /// Round-tripping the result through
    /// [`PreciseEphemerisSamples.from_samples`] rebuilds the same interpolatable
    /// source (byte-identical for samples whose metres are the faithful image of
    /// the fitted kilometres, sub-micron otherwise; see
    /// [`PreciseEphemerisSamples`]).
    fn precise_ephemeris_samples(&self) -> Vec<PyPreciseEphemerisSample> {
        self.inner
            .precise_ephemeris_samples()
            .into_iter()
            .map(PyPreciseEphemerisSample::from)
            .collect()
    }

    fn precise_ephemeris_accuracy_samples(&self) -> Vec<PyPreciseEphemerisAccuracySample> {
        self.inner
            .precise_ephemeris_accuracy_samples()
            .into_iter()
            .map(Into::into)
            .collect()
    }

    /// Build deterministic memory-mappable precise-interpolant artifact bytes.
    #[pyo3(signature = (gap_threshold_factor = None))]
    fn precise_interpolant_artifact_bytes<'py>(
        &self,
        py: Python<'py>,
        gap_threshold_factor: Option<f64>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let mut product = self.inner.clone();
        if let Some(factor) = gap_threshold_factor {
            let options = Sp3InterpolationOptions::new(factor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            product = product.with_interpolation_options(options);
        }
        let bytes = product
            .precise_interpolant_store_bytes()
            .map_err(precise_artifact_error_without_bytes)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3(epoch_count={}, satellites={})",
            self.inner.epoch_count(),
            self.inner.satellites().len()
        )
    }
}

/// A batch of interpolated SP3 states: `position_m` as a numpy `(n, 3)` ECEF
/// array in metres and `clock_s` as a numpy `(n,)` array in seconds (NaN where no
/// clock estimate exists). Returned by [`Sp3.interpolate`].
#[pyclass(module = "sidereon._sidereon", name = "Sp3Interpolation")]
pub struct PySp3Interpolation {
    positions: Vec<[f64; 3]>,
    clocks: Vec<f64>,
}

#[pymethods]
impl PySp3Interpolation {
    /// Interpolated ECEF positions as a numpy `(n, 3)` array, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.positions)
    }

    /// Interpolated clock offsets as a numpy `(n,)` array, seconds (NaN where the
    /// satellite has no clock estimate at that epoch).
    #[getter]
    fn clock_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.clocks)
    }

    /// Number of query epochs in the batch.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.positions.len()
    }

    fn __len__(&self) -> usize {
        self.positions.len()
    }

    fn __repr__(&self) -> String {
        format!("Sp3Interpolation(epoch_count={})", self.positions.len())
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "EphemerisSampleStatus",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyEphemerisSampleStatus {
    VALID,
    GAP,
}

impl From<EphemerisSampleStatus> for PyEphemerisSampleStatus {
    fn from(value: EphemerisSampleStatus) -> Self {
        match value {
            EphemerisSampleStatus::Valid => Self::VALID,
            EphemerisSampleStatus::Gap => Self::GAP,
        }
    }
}

#[pymethods]
impl PyEphemerisSampleStatus {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::VALID => "valid",
            Self::GAP => "gap",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::VALID => "EphemerisSampleStatus.VALID",
            Self::GAP => "EphemerisSampleStatus.GAP",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "EphemerisSampleRow")]
#[derive(Clone)]
pub struct PyEphemerisSampleRow {
    inner: EphemerisSampleRow,
}

impl From<EphemerisSampleRow> for PyEphemerisSampleRow {
    fn from(inner: EphemerisSampleRow) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEphemerisSampleRow {
    #[getter]
    fn satellite(&self) -> String {
        self.inner.sat.to_string()
    }

    #[getter]
    fn epoch_j2000_s(&self) -> f64 {
        self.inner.epoch_j2000_s
    }

    #[getter]
    fn status(&self) -> PyEphemerisSampleStatus {
        self.inner.status.into()
    }

    #[getter]
    fn position_ecef_m<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .position_ecef_m
            .map(|position| np_array(py, &position))
    }

    #[getter]
    fn clock_s(&self) -> Option<f64> {
        self.inner.clock_s
    }

    #[getter]
    fn is_gap(&self) -> bool {
        self.inner.is_gap()
    }

    fn __repr__(&self) -> String {
        format!(
            "EphemerisSampleRow(satellite={:?}, epoch_j2000_s={}, status={})",
            self.inner.sat.to_string(),
            self.inner.epoch_j2000_s,
            PyEphemerisSampleStatus::from(self.inner.status).label()
        )
    }
}

/// Status for one element of an [`ObservableStateBatch`].
#[pyclass(
    module = "sidereon._sidereon",
    name = "ObservableStateElementStatus",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyObservableStateElementStatus {
    /// The element contains a usable ECEF state.
    VALID,
    /// The source has no usable state for this satellite and epoch.
    GAP,
    /// The scalar evaluator returned a non-gap error.
    ERROR,
}

impl From<ObservableStateElementStatus> for PyObservableStateElementStatus {
    fn from(value: ObservableStateElementStatus) -> Self {
        match value {
            ObservableStateElementStatus::Valid => Self::VALID,
            ObservableStateElementStatus::Gap => Self::GAP,
            ObservableStateElementStatus::Error => Self::ERROR,
        }
    }
}

#[pymethods]
impl PyObservableStateElementStatus {
    /// Stable lowercase status label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::VALID => "valid",
            Self::GAP => "gap",
            Self::ERROR => "error",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::VALID => "ObservableStateElementStatus.VALID",
            Self::GAP => "ObservableStateElementStatus.GAP",
            Self::ERROR => "ObservableStateElementStatus.ERROR",
        }
    }
}

/// Contiguous output arrays for a batched satellite-state query.
///
/// Entry `i` belongs to input satellite `i` and epoch `i` for
/// [`observable_states_at_j2000_s`], or to input satellite `i` at the shared
/// epoch for [`observable_states_at_shared_j2000_s`]. `positions_ecef_m` is
/// numpy `(n, 3)` in ECEF metres. `clocks_s` is numpy `(n,)` in seconds, with
/// NaN when the core result has no clock. Failed elements use the public missing
/// position sentinel and carry their error text in `element_results`, with
/// structured core failures available through `element_error_details`.
#[pyclass(module = "sidereon._sidereon", name = "ObservableStateBatch")]
#[derive(Clone)]
pub struct PyObservableStateBatch {
    inner: ObservableStateBatch,
}

impl From<ObservableStateBatch> for PyObservableStateBatch {
    fn from(inner: ObservableStateBatch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObservableStateBatch {
    /// Satellite ECEF positions as numpy `(n, 3)`, metres.
    ///
    /// Failed elements are filled with
    /// `OBSERVABLE_STATE_MISSING_POSITION_ECEF_M`.
    #[getter]
    fn positions_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.inner.positions_ecef_m)
    }

    /// Satellite clock offsets as numpy `(n,)`, seconds.
    ///
    /// Entries are NaN when the source has no clock or the element failed.
    #[getter]
    fn clocks_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let clocks: Vec<_> = self
            .inner
            .clocks_s
            .iter()
            .map(|clock| clock.unwrap_or(f64::NAN))
            .collect();
        np_array(py, &clocks)
    }

    /// Per-element status categories.
    #[getter]
    fn statuses(&self) -> Vec<PyObservableStateElementStatus> {
        (0..self.inner.len())
            .filter_map(|index| self.inner.element_status(index))
            .map(Into::into)
            .collect()
    }

    /// Per-element result text: `None` for success, error text for failure.
    #[getter]
    fn element_results(&self) -> Vec<Option<String>> {
        self.inner
            .element_results
            .iter()
            .map(|result| result.as_ref().err().map(ToString::to_string))
            .collect()
    }

    /// Fresh structured details for each failed element; successful elements
    /// contain `None`. Mutating a returned dictionary does not change this batch.
    #[getter]
    fn element_error_details(&self, py: Python<'_>) -> PyResult<Vec<Option<Py<PyDict>>>> {
        self.inner
            .element_results
            .iter()
            .map(|result| {
                result
                    .as_ref()
                    .err()
                    .map(|error| observables_error_detail(py, error))
                    .transpose()
            })
            .collect()
    }

    /// Number of batch elements.
    #[getter]
    fn element_count(&self) -> usize {
        self.inner.len()
    }

    /// Whether this batch has no elements.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the status category for one element.
    fn element_status(&self, index: usize) -> PyResult<PyObservableStateElementStatus> {
        self.inner
            .element_status(index)
            .map(Into::into)
            .ok_or_else(|| PyIndexError::new_err(format!("element index {index} out of range")))
    }

    /// Return `None` for a successful element or the core error text on failure.
    fn element_result(&self, index: usize) -> PyResult<Option<String>> {
        self.inner
            .element_results
            .get(index)
            .map(|result| result.as_ref().err().map(ToString::to_string))
            .ok_or_else(|| PyIndexError::new_err(format!("element index {index} out of range")))
    }

    /// Return a fresh structured detail dictionary for one failed element, or
    /// `None` when that element succeeded.
    fn element_error_detail(&self, py: Python<'_>, index: usize) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .element_results
            .get(index)
            .map(|result| {
                result
                    .as_ref()
                    .err()
                    .map(|error| observables_error_detail(py, error))
                    .transpose()
            })
            .ok_or_else(|| PyIndexError::new_err(format!("element index {index} out of range")))?
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("ObservableStateBatch(element_count={})", self.inner.len())
    }
}

/// One epoch's clock-reference datum offset between two SP3 products.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ClockReferenceOffset")]
#[derive(Clone)]
pub struct PySp3ClockReferenceOffset {
    epoch_j2000_seconds: f64,
    offset_s: f64,
    satellites: usize,
}

#[pymethods]
impl PySp3ClockReferenceOffset {
    /// Matched epoch as seconds since J2000 in the product time scale.
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        self.epoch_j2000_seconds
    }

    /// Clock datum offset, seconds, computed as `other - reference`.
    #[getter]
    fn offset_s(&self) -> f64 {
        self.offset_s
    }

    /// Number of common clocked satellites used in the median estimate.
    #[getter]
    fn satellites(&self) -> usize {
        self.satellites
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ClockReferenceOffset(epoch_j2000_seconds={}, offset_s={}, satellites={})",
            self.epoch_j2000_seconds, self.offset_s, self.satellites
        )
    }
}

impl From<ClockReferenceOffset> for PySp3ClockReferenceOffset {
    fn from(value: ClockReferenceOffset) -> Self {
        Self {
            epoch_j2000_seconds: instant_to_j2000_seconds(&value.epoch).unwrap_or(f64::NAN),
            offset_s: value.offset_s,
            satellites: value.satellites,
        }
    }
}

/// The exact parsed state of one satellite at one SP3 epoch.
///
/// `position_m` is the ECEF position (metres); `clock_s` is the clock offset
/// (seconds) or `None` for the bad-clock sentinel; `velocity_m_s` is the ECEF
/// velocity (metres per second) or `None` for a position-only product;
/// `clock_rate_s_s` is the clock rate (seconds per second) from the paired `V`
/// record, or `None` where there is none or it holds the bad-rate sentinel. The
/// four status flags are surfaced verbatim from the record.
#[pyclass(module = "sidereon._sidereon", name = "Sp3State")]
pub struct PySp3State {
    position: [f64; 3],
    clock_s: Option<f64>,
    velocity: Option<[f64; 3]>,
    clock_rate_s_s: Option<f64>,
    clock_event: bool,
    clock_predicted: bool,
    maneuver: bool,
    orbit_predicted: bool,
}

impl PySp3State {
    /// Build from a core parsed or interpolated state. [`Sp3.state`], the
    /// interpolant queries and the staleness selection layer all construct
    /// through here, so every field reaches Python by one path.
    pub(crate) fn from_state(state: sidereon_core::ephemeris::Sp3State) -> Self {
        Self {
            position: state.position.as_array(),
            clock_s: state.clock_s,
            velocity: state.velocity.map(|v| v.as_array()),
            clock_rate_s_s: state.clock_rate_s_s,
            clock_event: state.flags.clock_event,
            clock_predicted: state.flags.clock_predicted,
            maneuver: state.flags.maneuver,
            orbit_predicted: state.flags.orbit_predicted,
        }
    }
}

#[pymethods]
impl PySp3State {
    /// ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.position)
    }

    /// Clock offset in seconds, or `None` for the bad-clock sentinel.
    #[getter]
    fn clock_s(&self) -> Option<f64> {
        self.clock_s
    }

    /// ECEF velocity as a numpy `(3,)` array in metres per second, or `None` for a
    /// position-only product.
    #[getter]
    fn velocity_m_s<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.velocity.map(|v| np_array(py, &v))
    }

    /// Clock rate in seconds per second from the paired `V` record, or `None`
    /// for a position-only product or the bad-rate sentinel.
    #[getter]
    fn clock_rate_s_s(&self) -> Option<f64> {
        self.clock_rate_s_s
    }

    /// Clock discontinuity (`E`) flagged at this epoch (clock interpolation across
    /// it is unsafe).
    #[getter]
    fn clock_event(&self) -> bool {
        self.clock_event
    }

    /// The clock is predicted, not fitted.
    #[getter]
    fn clock_predicted(&self) -> bool {
        self.clock_predicted
    }

    /// The satellite was being maneuvered at this epoch.
    #[getter]
    fn maneuver(&self) -> bool {
        self.maneuver
    }

    /// The orbit is predicted, not fitted.
    #[getter]
    fn orbit_predicted(&self) -> bool {
        self.orbit_predicted
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3State(position_m=[{}, {}, {}], clock_s={:?})",
            self.position[0], self.position[1], self.position[2], self.clock_s
        )
    }
}

/// A clock-only SP3 record: a satellite whose orbit is the missing-orbit
/// sentinel (`0.0 0.0 0.0`) at one epoch beside a valid clock estimate.
///
/// Kept apart from [`Sp3State`] so a consumer of positions never meets a
/// fabricated geocentre, and never interpolated as an orbit node. `clock_s` is
/// the clock in seconds and `clock_us` the same value in the file's own
/// microseconds, exactly as read. `velocity_m_s`, `clock_rate_s_s` and
/// `clock_rate_raw` (the rate in the file's 1e-4 microseconds per second) come
/// from the paired `V` record and are `None` where there is none or it holds
/// the format's absence sentinel. The four status flags are surfaced verbatim.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ClockRecord")]
#[derive(Clone, Copy)]
pub struct PySp3ClockRecord {
    inner: Sp3ClockRecord,
}

impl From<Sp3ClockRecord> for PySp3ClockRecord {
    fn from(inner: Sp3ClockRecord) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySp3ClockRecord {
    /// Clock offset in seconds.
    #[getter]
    fn clock_s(&self) -> f64 {
        self.inner.clock_s
    }

    /// Clock offset in the file's microseconds, exactly as read.
    #[getter]
    fn clock_us(&self) -> f64 {
        self.inner.clock_us
    }

    /// ECEF velocity as a numpy `(3,)` array in metres per second, or `None`.
    #[getter]
    fn velocity_m_s<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.velocity.map(|v| np_array(py, &v.as_array()))
    }

    /// Clock rate in seconds per second, or `None`.
    #[getter]
    fn clock_rate_s_s(&self) -> Option<f64> {
        self.inner.clock_rate_s_s
    }

    /// Clock rate in the file's 1e-4 microseconds per second, exactly as read,
    /// or `None`.
    #[getter]
    fn clock_rate_raw(&self) -> Option<f64> {
        self.inner.clock_rate_raw
    }

    /// Clock discontinuity (`E`) flagged at this epoch.
    #[getter]
    fn clock_event(&self) -> bool {
        self.inner.flags.clock_event
    }

    /// The clock is predicted, not fitted.
    #[getter]
    fn clock_predicted(&self) -> bool {
        self.inner.flags.clock_predicted
    }

    /// The satellite was being maneuvered at this epoch.
    #[getter]
    fn maneuver(&self) -> bool {
        self.inner.flags.maneuver
    }

    /// The orbit is predicted, not fitted.
    #[getter]
    fn orbit_predicted(&self) -> bool {
        self.inner.flags.orbit_predicted
    }

    fn __eq__(&self, other: &PySp3ClockRecord) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ClockRecord(clock_s={:?}, clock_us={:?}, clock_rate_s_s={:?})",
            self.inner.clock_s, self.inner.clock_us, self.inner.clock_rate_s_s
        )
    }
}

/// The parsed SP3 header, as the core holds it.
///
/// `version` is the format letter (`"a"` to `"d"`) and `data_type` the record
/// type letter (`"P"` or `"V"`). `data_used` (line 1, columns 41-45) and
/// `file_type` (first `%c` line, columns 4-5) are `None` when their columns are
/// blank. `pos_vel_base` and `clock_rate_base` are the first `%f` line's bases,
/// `None` when blank; an explicit zero is kept with its sign. `time_system` is
/// the exact SP3 label (`"GPS"`, `"GLO"`, `"GAL"`, `"TAI"`, `"UTC"`, `"QZS"`,
/// `"BDT"`, `"IRN"`) and `time_scale` the core scale the epochs are tagged
/// with. `satellite_accuracy_codes` is index-aligned with `satellites`.
/// `num_epochs` is the number of epoch records parsed, which the writer states
/// on line 1; the count line 1 declared is `Sp3.declared_epoch_count`.
#[pyclass(module = "sidereon._sidereon", name = "Sp3Header")]
#[derive(Clone)]
pub struct PySp3Header {
    inner: Sp3Header,
}

#[pymethods]
impl PySp3Header {
    /// SP3 format version letter: `"a"`, `"b"`, `"c"` or `"d"`.
    #[getter]
    fn version(&self) -> &'static str {
        match self.inner.version {
            Sp3Version::A => "a",
            Sp3Version::B => "b",
            Sp3Version::C => "c",
            Sp3Version::D => "d",
        }
    }

    /// Record type letter: `"P"` (position) or `"V"` (position and velocity).
    #[getter]
    fn data_type(&self) -> &'static str {
        match self.inner.data_type {
            Sp3DataType::Position => "P",
            Sp3DataType::Velocity => "V",
        }
    }

    /// Number of epoch records parsed.
    #[getter]
    fn num_epochs(&self) -> u64 {
        self.inner.num_epochs
    }

    /// Data-used descriptor from line 1 (for example `"ORBIT"`), or `None`.
    #[getter]
    fn data_used(&self) -> Option<String> {
        self.inner.data_used.clone()
    }

    /// Coordinate-system label (for example `"IGS20"`).
    #[getter]
    fn coordinate_system(&self) -> String {
        self.inner.coordinate_system.clone()
    }

    /// Orbit-type label (for example `"FIT"`).
    #[getter]
    fn orbit_type(&self) -> String {
        self.inner.orbit_type.clone()
    }

    /// Producing agency.
    #[getter]
    fn agency(&self) -> String {
        self.inner.agency.clone()
    }

    /// GNSS week of the first epoch, in the file's time system.
    #[getter]
    fn gnss_week(&self) -> u32 {
        self.inner.gnss_week
    }

    /// Seconds of week of the first epoch.
    #[getter]
    fn seconds_of_week(&self) -> f64 {
        self.inner.seconds_of_week
    }

    /// Nominal epoch spacing in seconds.
    #[getter]
    fn epoch_interval_s(&self) -> f64 {
        self.inner.epoch_interval_s
    }

    /// Modified Julian Day of the first epoch (integer part).
    #[getter]
    fn mjd(&self) -> u32 {
        self.inner.mjd
    }

    /// Fractional day of the first epoch.
    #[getter]
    fn mjd_fraction(&self) -> f64 {
        self.inner.mjd_fraction
    }

    /// File-type descriptor from the first `%c` line (for example `"G"`,
    /// `"M"`), or `None`.
    #[getter]
    fn file_type(&self) -> Option<String> {
        self.inner.file_type.clone()
    }

    /// The exact SP3 time-system label from the first `%c` line.
    #[getter]
    fn time_system(&self) -> &'static str {
        self.inner.time_system.label()
    }

    /// The core time scale the parsed epochs are tagged with.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        PyTimeScale::from(self.inner.time_scale)
    }

    /// Position/velocity standard-deviation base from the first `%f` line, or
    /// `None` when blank.
    #[getter]
    fn pos_vel_base(&self) -> Option<f64> {
        self.inner.pos_vel_base
    }

    /// Clock/clock-rate standard-deviation base from the first `%f` line, or
    /// `None` when blank.
    #[getter]
    fn clock_rate_base(&self) -> Option<f64> {
        self.inner.clock_rate_base
    }

    /// The satellite tokens declared in the `+` lines, in declaration order.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner
            .satellites
            .iter()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// Accuracy exponent codes from the `++` lines, index-aligned with
    /// `satellites`.
    #[getter]
    fn satellite_accuracy_codes(&self) -> Vec<u16> {
        self.inner.satellite_accuracy_codes.clone()
    }

    fn __eq__(&self, other: &PySp3Header) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3Header(version={:?}, data_type={:?}, agency={:?}, time_system={:?}, satellites={})",
            self.version(),
            self.data_type(),
            self.inner.agency,
            self.time_system(),
            self.inner.satellites.len()
        )
    }
}

/// The typed payload an `Sp3WriteError` carries on its `detail` attribute.
///
/// `kind` is the core `Sp3WriteError` variant name and `details()` that
/// variant's own fields, so a caller reads the native values rather than
/// parsing `message`. Keys are the core field names, except that a satellite
/// is the token under `"satellite"`, a time scale is a `TimeScale`, and the
/// SP3 time system is its label. `EpochNotRestatable`'s `residual_s` is None
/// where the core holds NaN, its spelling of "no statement could be read back".
/// `field`, `epoch_index`, `satellite`, `columns` and `decimals` are
/// conveniences that return None for a variant that names no such field.
///
/// A variant this build does not name - one a later core adds - still maps:
/// `kind` is its core name and `details()` holds its complete core `Debug`
/// rendering under `"debug"`.
#[pyclass(module = "sidereon._sidereon", name = "Sp3WriteErrorDetail")]
#[derive(Clone, Debug, PartialEq)]
pub struct PySp3WriteErrorDetail {
    inner: Sp3WriteError,
}

impl From<Sp3WriteError> for PySp3WriteErrorDetail {
    fn from(inner: Sp3WriteError) -> Self {
        Self { inner }
    }
}

/// The variant name at the head of the core's derived `Debug` rendering.
///
/// `Sp3WriteError` is `#[non_exhaustive]`, so a core newer than this binding
/// can hand back a variant no arm here names. Its derived `Debug` output starts
/// with the variant name, which is the `kind` it reports.
fn debug_variant_name(err: &Sp3WriteError) -> String {
    format!("{err:?}")
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

#[pymethods]
impl PySp3WriteErrorDetail {
    /// Error kind string matching the core `Sp3WriteError` variant name.
    #[getter]
    fn kind(&self) -> String {
        let name = match &self.inner {
            Sp3WriteError::TextNotColumnSafe { .. } => "TextNotColumnSafe",
            Sp3WriteError::TextNotColumnStable { .. } => "TextNotColumnStable",
            Sp3WriteError::BlankDescriptor { .. } => "BlankDescriptor",
            Sp3WriteError::EmptyComment { .. } => "EmptyComment",
            Sp3WriteError::TextTooWide { .. } => "TextTooWide",
            Sp3WriteError::IntegerTooWide { .. } => "IntegerTooWide",
            Sp3WriteError::NonFinite { .. } => "NonFinite",
            Sp3WriteError::NumberTooWide { .. } => "NumberTooWide",
            Sp3WriteError::PrecisionNotRepresentable { .. } => "PrecisionNotRepresentable",
            Sp3WriteError::YearNotRepresentable { .. } => "YearNotRepresentable",
            Sp3WriteError::AccuracyNotRepresentable { .. } => "AccuracyNotRepresentable",
            Sp3WriteError::AccuracyRecordMismatch { .. } => "AccuracyRecordMismatch",
            Sp3WriteError::AccuracyBasisMissing { .. } => "AccuracyBasisMissing",
            Sp3WriteError::EpochNotRestatable { .. } => "EpochNotRestatable",
            Sp3WriteError::EpochTimeScaleMismatch { .. } => "EpochTimeScaleMismatch",
            Sp3WriteError::HeaderTimeScaleMismatch { .. } => "HeaderTimeScaleMismatch",
            Sp3WriteError::EpochCountMismatch { .. } => "EpochCountMismatch",
            Sp3WriteError::AccuracyCodeCountMismatch { .. } => "AccuracyCodeCountMismatch",
            Sp3WriteError::DuplicateSatellite { .. } => "DuplicateSatellite",
            Sp3WriteError::SatelliteNotRepresentable { .. } => "SatelliteNotRepresentable",
            Sp3WriteError::EpochArrayLengthMismatch { .. } => "EpochArrayLengthMismatch",
            Sp3WriteError::UndeclaredSatelliteRecord { .. } => "UndeclaredSatelliteRecord",
            Sp3WriteError::ConflictingRecords { .. } => "ConflictingRecords",
            Sp3WriteError::VelocityStateInPositionProduct { .. } => {
                "VelocityStateInPositionProduct"
            }
            Sp3WriteError::RecordValueNonFinite { .. } => "RecordValueNonFinite",
            Sp3WriteError::RecordValueTooWide { .. } => "RecordValueTooWide",
            Sp3WriteError::RecordValueNotRepresentable { .. } => "RecordValueNotRepresentable",
            Sp3WriteError::RecordReadsAsAbsent { .. } => "RecordReadsAsAbsent",
            Sp3WriteError::RecordFieldsDisagree { .. } => "RecordFieldsDisagree",
            other => return debug_variant_name(other),
        };
        name.to_string()
    }

    /// Formatted core error message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// The field the refusal names, for the variants that name one.
    #[getter]
    fn field(&self) -> Option<&'static str> {
        match &self.inner {
            Sp3WriteError::TextNotColumnSafe { field, .. }
            | Sp3WriteError::TextNotColumnStable { field, .. }
            | Sp3WriteError::BlankDescriptor { field, .. }
            | Sp3WriteError::TextTooWide { field, .. }
            | Sp3WriteError::IntegerTooWide { field, .. }
            | Sp3WriteError::NonFinite { field }
            | Sp3WriteError::NumberTooWide { field, .. }
            | Sp3WriteError::PrecisionNotRepresentable { field, .. }
            | Sp3WriteError::EpochArrayLengthMismatch { field, .. }
            | Sp3WriteError::VelocityStateInPositionProduct { field, .. }
            | Sp3WriteError::RecordValueNonFinite { field, .. }
            | Sp3WriteError::RecordValueTooWide { field, .. }
            | Sp3WriteError::RecordValueNotRepresentable { field, .. }
            | Sp3WriteError::RecordReadsAsAbsent { field, .. }
            | Sp3WriteError::RecordFieldsDisagree { field, .. } => Some(*field),
            _ => None,
        }
    }

    /// Index into `Sp3.epochs_j2000_seconds` of the epoch the refusal names.
    #[getter]
    fn epoch_index(&self) -> Option<usize> {
        match &self.inner {
            Sp3WriteError::YearNotRepresentable { epoch_index, .. }
            | Sp3WriteError::EpochNotRestatable { epoch_index, .. }
            | Sp3WriteError::EpochTimeScaleMismatch { epoch_index, .. }
            | Sp3WriteError::AccuracyNotRepresentable { epoch_index, .. }
            | Sp3WriteError::AccuracyRecordMismatch { epoch_index, .. }
            | Sp3WriteError::AccuracyBasisMissing { epoch_index, .. }
            | Sp3WriteError::UndeclaredSatelliteRecord { epoch_index, .. }
            | Sp3WriteError::ConflictingRecords { epoch_index, .. }
            | Sp3WriteError::VelocityStateInPositionProduct { epoch_index, .. }
            | Sp3WriteError::RecordValueNonFinite { epoch_index, .. }
            | Sp3WriteError::RecordValueTooWide { epoch_index, .. }
            | Sp3WriteError::RecordValueNotRepresentable { epoch_index, .. }
            | Sp3WriteError::RecordReadsAsAbsent { epoch_index, .. }
            | Sp3WriteError::RecordFieldsDisagree { epoch_index, .. } => Some(*epoch_index),
            _ => None,
        }
    }

    /// The satellite token the refusal names.
    #[getter]
    fn satellite(&self) -> Option<String> {
        match &self.inner {
            Sp3WriteError::DuplicateSatellite { sat }
            | Sp3WriteError::SatelliteNotRepresentable { sat }
            | Sp3WriteError::AccuracyNotRepresentable { sat, .. }
            | Sp3WriteError::AccuracyRecordMismatch { sat, .. }
            | Sp3WriteError::AccuracyBasisMissing { sat, .. }
            | Sp3WriteError::UndeclaredSatelliteRecord { sat, .. }
            | Sp3WriteError::ConflictingRecords { sat, .. }
            | Sp3WriteError::VelocityStateInPositionProduct { sat, .. }
            | Sp3WriteError::RecordValueNonFinite { sat, .. }
            | Sp3WriteError::RecordValueTooWide { sat, .. }
            | Sp3WriteError::RecordValueNotRepresentable { sat, .. }
            | Sp3WriteError::RecordReadsAsAbsent { sat, .. }
            | Sp3WriteError::RecordFieldsDisagree { sat, .. } => Some(sat.to_string()),
            _ => None,
        }
    }

    /// Columns the refused field occupies.
    #[getter]
    fn columns(&self) -> Option<usize> {
        match &self.inner {
            Sp3WriteError::TextTooWide { columns, .. }
            | Sp3WriteError::IntegerTooWide { columns, .. }
            | Sp3WriteError::NumberTooWide { columns, .. }
            | Sp3WriteError::PrecisionNotRepresentable { columns, .. }
            | Sp3WriteError::RecordValueTooWide { columns, .. }
            | Sp3WriteError::RecordValueNotRepresentable { columns, .. } => Some(*columns),
            _ => None,
        }
    }

    /// Decimal places the refused `F` field carries.
    #[getter]
    fn decimals(&self) -> Option<usize> {
        match &self.inner {
            Sp3WriteError::NumberTooWide { decimals, .. }
            | Sp3WriteError::PrecisionNotRepresentable { decimals, .. }
            | Sp3WriteError::RecordValueTooWide { decimals, .. }
            | Sp3WriteError::RecordValueNotRepresentable { decimals, .. } => Some(*decimals),
            _ => None,
        }
    }

    /// Dictionary containing every field of this refusal variant.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            Sp3WriteError::TextNotColumnSafe { field, value }
            | Sp3WriteError::TextNotColumnStable { field, value }
            | Sp3WriteError::BlankDescriptor { field, value } => {
                dict.set_item("field", *field)?;
                dict.set_item("value", value.as_str())?;
            }
            Sp3WriteError::EmptyComment { index, value } => {
                dict.set_item("index", *index)?;
                dict.set_item("value", value.as_str())?;
            }
            Sp3WriteError::TextTooWide {
                field,
                columns,
                value,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("columns", *columns)?;
                dict.set_item("value", value.as_str())?;
            }
            Sp3WriteError::IntegerTooWide {
                field,
                columns,
                value,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("columns", *columns)?;
                dict.set_item("value", *value)?;
            }
            Sp3WriteError::NonFinite { field } => {
                dict.set_item("field", *field)?;
            }
            Sp3WriteError::NumberTooWide {
                field,
                columns,
                decimals,
                value,
            }
            | Sp3WriteError::PrecisionNotRepresentable {
                field,
                columns,
                decimals,
                value,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("columns", *columns)?;
                dict.set_item("decimals", *decimals)?;
                dict.set_item("value", *value)?;
            }
            Sp3WriteError::YearNotRepresentable { epoch_index, year } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("year", *year)?;
            }
            Sp3WriteError::EpochNotRestatable {
                epoch_index,
                field_seconds,
                residual_s,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("field_seconds", *field_seconds)?;
                dict.set_item("residual_s", (!residual_s.is_nan()).then_some(*residual_s))?;
            }
            Sp3WriteError::EpochTimeScaleMismatch {
                epoch_index,
                epoch_scale,
                header_scale,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("epoch_scale", PyTimeScale::from(*epoch_scale))?;
                dict.set_item("header_scale", PyTimeScale::from(*header_scale))?;
            }
            Sp3WriteError::HeaderTimeScaleMismatch {
                time_system,
                time_scale,
            } => {
                dict.set_item("time_system", time_system.label())?;
                dict.set_item("time_scale", PyTimeScale::from(*time_scale))?;
            }
            Sp3WriteError::EpochCountMismatch { declared, epochs } => {
                dict.set_item("declared", *declared)?;
                dict.set_item("epochs", *epochs)?;
            }
            Sp3WriteError::AccuracyCodeCountMismatch { satellites, codes } => {
                dict.set_item("satellites", *satellites)?;
                dict.set_item("codes", *codes)?;
            }
            Sp3WriteError::AccuracyNotRepresentable {
                sat,
                epoch_index,
                component,
                exponent,
            } => {
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("component", *component)?;
                dict.set_item("exponent", *exponent)?;
            }
            Sp3WriteError::AccuracyRecordMismatch { sat, epoch_index }
            | Sp3WriteError::AccuracyBasisMissing { sat, epoch_index } => {
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
            }
            Sp3WriteError::DuplicateSatellite { sat } => {
                dict.set_item("satellite", sat.to_string())?;
            }
            Sp3WriteError::SatelliteNotRepresentable { sat } => {
                // The satellite has no `01`..`99` token, so its system and
                // number are given apart from the rendered text.
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("system", PyGnssSystem::from(sat.system))?;
                dict.set_item("prn", sat.prn)?;
            }
            Sp3WriteError::EpochArrayLengthMismatch {
                field,
                epochs,
                entries,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("epochs", *epochs)?;
                dict.set_item("entries", *entries)?;
            }
            Sp3WriteError::UndeclaredSatelliteRecord { sat, epoch_index }
            | Sp3WriteError::ConflictingRecords { sat, epoch_index } => {
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
            }
            Sp3WriteError::VelocityStateInPositionProduct {
                field,
                sat,
                epoch_index,
            }
            | Sp3WriteError::RecordValueNonFinite {
                field,
                sat,
                epoch_index,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
            }
            Sp3WriteError::RecordValueTooWide {
                field,
                sat,
                epoch_index,
                columns,
                decimals,
                column_value,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("columns", *columns)?;
                dict.set_item("decimals", *decimals)?;
                dict.set_item("column_value", *column_value)?;
            }
            Sp3WriteError::RecordValueNotRepresentable {
                field,
                sat,
                epoch_index,
                columns,
                decimals,
                stored,
                column_value,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("columns", *columns)?;
                dict.set_item("decimals", *decimals)?;
                dict.set_item("stored", *stored)?;
                dict.set_item("column_value", *column_value)?;
            }
            Sp3WriteError::RecordReadsAsAbsent {
                field,
                sat,
                epoch_index,
                column_value,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("column_value", *column_value)?;
            }
            Sp3WriteError::RecordFieldsDisagree {
                field,
                sat,
                epoch_index,
                stored,
                native,
            } => {
                dict.set_item("field", *field)?;
                dict.set_item("satellite", sat.to_string())?;
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("stored", *stored)?;
                dict.set_item("native", *native)?;
            }
            other => {
                dict.set_item("debug", format!("{other:?}"))?;
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3WriteErrorDetail(kind=\"{}\", message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &PySp3WriteErrorDetail) -> bool {
        self.inner == other.inner
    }
}

/// Map an SP3 write refusal into `Sp3WriteError`, carrying its typed payload
/// as `detail`.
pub(crate) fn to_sp3_write_err(py: Python<'_>, err: Sp3WriteError) -> PyErr {
    let ty = match sp3_write_error_type(py) {
        Ok(ty) => ty,
        Err(e) => return e,
    };
    let py_err = PyErr::from_type(ty, err.to_string());
    let detail = match PySp3WriteErrorDetail::from(err).into_pyobject(py) {
        Ok(detail) => detail,
        Err(e) => return e,
    };
    if let Err(e) = py_err.value(py).setattr("detail", detail) {
        return e;
    }
    py_err
}

/// Map a per-record lookup failure: an epoch past the end is `IndexError`, a
/// satellite with no such record at that epoch is `KeyError`.
fn record_lookup_error(err: CoreError, epoch_index: usize, what: &str) -> PyErr {
    let python_error = match &err {
        CoreError::EpochOutOfRange => {
            PyIndexError::new_err(format!("epoch index {epoch_index} out of range"))
        }
        CoreError::UnknownSatellite(id) => PyKeyError::new_err(format!(
            "satellite {id} has no {what} at epoch {epoch_index}"
        )),
        other => to_solve_err(other.to_string()),
    };
    let python_error = attach_core_error_detail(python_error, &err);
    Python::with_gil(|py| {
        let value = python_error.value(py);
        if let Ok(detail) = value.getattr("detail") {
            if let Ok(detail) = detail.downcast::<PyDict>() {
                if let Err(error) = detail.set_item("epoch_index", epoch_index) {
                    return error;
                }
            }
        }
        python_error
    })
}

/// One precise-ephemeris sample: a satellite's ECEF position (and optional
/// clock) at one epoch, in SI units.
///
/// This is the canonical serialization-independent element behind
/// [`PreciseEphemerisSamples`]. `position_ecef_m` is the ITRF/IGS ECEF position
/// in metres; `clock_s` is the satellite clock offset in seconds, `None` when
/// the source carried no clock estimate. `clock_event` mirrors the SP3 `E`
/// clock-event flag: `True` marks a clock discontinuity, splitting the
/// interpolated clock arc there. The epoch is carried as seconds since J2000 in
/// the sample's own `time_scale`.
#[pyclass(module = "sidereon._sidereon", name = "PreciseEphemerisSample")]
#[derive(Clone)]
pub struct PyPreciseEphemerisSample {
    inner: PreciseEphemerisSample,
}

impl From<PreciseEphemerisSample> for PyPreciseEphemerisSample {
    fn from(inner: PreciseEphemerisSample) -> Self {
        Self { inner }
    }
}

impl PyPreciseEphemerisSample {
    pub(crate) fn to_core(&self) -> PreciseEphemerisSample {
        self.inner
    }
}

#[pymethods]
impl PyPreciseEphemerisSample {
    /// Build one precise-ephemeris sample.
    ///
    /// `satellite` is a canonical token such as `"G01"`, `epoch_j2000_seconds`
    /// is the sample epoch in `time_scale`, `position_ecef_m` is a length-3 ECEF
    /// position in metres, and `clock_s` is the optional clock offset in seconds.
    /// Set `clock_event=True` to reconstruct an epoch that carries the SP3 `E`
    /// clock reset.
    #[new]
    #[pyo3(signature = (
        satellite,
        epoch_j2000_seconds,
        position_ecef_m,
        clock_s=None,
        *,
        time_scale=PyTimeScale::GPST,
        clock_event=false,
    ))]
    fn new(
        satellite: &str,
        epoch_j2000_seconds: f64,
        position_ecef_m: [f64; 3],
        clock_s: Option<f64>,
        time_scale: PyTimeScale,
        clock_event: bool,
    ) -> PyResult<Self> {
        let sat = parse_sat(satellite)?;
        let epoch = instant_from_j2000_seconds(epoch_j2000_seconds, time_scale.into())?;
        let mut inner = PreciseEphemerisSample::new(sat, epoch, position_ecef_m, clock_s);
        inner.clock_event = clock_event;
        Ok(Self { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (satellite, epoch, position_ecef_m, clock_s=None, *, clock_event=false))]
    fn from_instant(
        satellite: &str,
        epoch: &PyClockInstant,
        position_ecef_m: [f64; 3],
        clock_s: Option<f64>,
        clock_event: bool,
    ) -> PyResult<Self> {
        let mut inner = PreciseEphemerisSample::new(
            parse_sat(satellite)?,
            epoch.to_core(),
            position_ecef_m,
            clock_s,
        );
        inner.clock_event = clock_event;
        Ok(Self { inner })
    }

    /// Satellite token, e.g. `"G01"`.
    #[getter]
    fn satellite(&self) -> String {
        self.inner.sat.to_string()
    }

    /// Sample epoch as seconds since J2000, in `time_scale`.
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        instant_to_j2000_seconds(&self.inner.epoch).unwrap_or(f64::NAN)
    }

    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.epoch)
    }

    /// Time scale the epoch is expressed in.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.epoch.scale.into()
    }

    /// ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn position_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_ecef_m)
    }

    /// Clock offset in seconds, or `None` when the source carried no estimate.
    #[getter]
    fn clock_s(&self) -> Option<f64> {
        self.inner.clock_s
    }

    /// Whether this epoch carries the SP3 `E` clock-event flag.
    #[getter]
    fn clock_event(&self) -> bool {
        self.inner.clock_event
    }

    fn __repr__(&self) -> String {
        format!(
            "PreciseEphemerisSample(satellite={:?}, epoch_j2000_seconds={}, clock_s={:?}, clock_event={})",
            self.inner.sat.to_string(),
            self.epoch_j2000_seconds(),
            self.inner.clock_s,
            self.inner.clock_event
        )
    }
}

/// A precise-ephemeris source built from samples rather than parsed SP3 text.
///
/// Construct with [`PreciseEphemerisSamples.from_samples`]. It drives the same
/// interpolation substrate the SP3-parsed product uses, so it can be passed to
/// [`predict_ranges`](crate) as an ephemeris source and yields interpolated
/// states / predicted ranges that match the SP3 path byte-for-byte for samples
/// that are the faithful image of the fitted nodes (the round-trip case), and to
/// within sub-micron precision otherwise.
#[pyclass(module = "sidereon._sidereon", name = "PreciseEphemerisSamples")]
pub struct PyPreciseEphemerisSamples {
    pub(crate) inner: PreciseEphemerisSamples,
}

#[pymethods]
impl PyPreciseEphemerisSamples {
    /// Build a source from a sequence of [`PreciseEphemerisSample`].
    ///
    /// Samples are grouped by satellite in supplied order and validated. Raises
    /// Raises `PreciseSamplesError` for core validation failures; `kind` names
    /// the core variant and `satellite` is populated for satellite-specific
    /// failures.
    #[staticmethod]
    #[pyo3(signature = (samples, gap_threshold_factor = None))]
    fn from_samples(
        py: Python<'_>,
        samples: Vec<Py<PyPreciseEphemerisSample>>,
        gap_threshold_factor: Option<f64>,
    ) -> PyResult<Self> {
        let samples = samples.iter().map(|s| s.borrow(py).to_core());
        let mut inner = PreciseEphemerisSamples::from_samples(samples)
            .map_err(|error| precise_samples_error(py, error))?;
        if let Some(factor) = gap_threshold_factor {
            let options = Sp3InterpolationOptions::new(factor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            inner = inner.with_interpolation_options(options);
        }
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_samples_with_accuracy(
        py: Python<'_>,
        samples: Vec<Py<PyPreciseEphemerisSample>>,
        accuracy: Vec<Py<PyPreciseEphemerisAccuracySample>>,
    ) -> PyResult<Self> {
        let samples = samples.iter().map(|sample| sample.borrow(py).to_core());
        let accuracy = accuracy.iter().map(|item| item.borrow(py).to_core());
        let inner = PreciseEphemerisSamples::from_samples_with_accuracy(samples, accuracy)
            .map_err(|error| precise_samples_error(py, error))?;
        Ok(Self { inner })
    }

    /// SP3 interpolation gap threshold factor carried by this source.
    #[getter]
    fn gap_threshold_factor(&self) -> f64 {
        self.inner.interpolation_options().gap_threshold_factor()
    }

    /// Return a copy of these samples with an explicit gap threshold factor.
    fn with_interpolation_options(&self, gap_threshold_factor: f64) -> PyResult<Self> {
        let options = Sp3InterpolationOptions::new(gap_threshold_factor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: self.inner.clone().with_interpolation_options(options),
        })
    }

    /// Time scale every sample epoch is expressed in.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale().into()
    }

    /// Satellite tokens this source can interpolate, ascending.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner.satellites().map(|sat| sat.to_string()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PreciseEphemerisSamples(satellites={})",
            self.inner.satellites().count()
        )
    }
}

/// A reusable precise-ephemeris interpolant with cached per-satellite nodes.
///
/// Build it once from an [`Sp3`] product, from raw [`PreciseEphemerisSample`]
/// rows, or from an existing [`PreciseEphemerisSamples`] source. State queries
/// use seconds since J2000 in the source time scale and return ECEF metres plus
/// optional clock seconds, matching the scalar precise-ephemeris evaluator.
#[pyclass(module = "sidereon._sidereon", name = "PreciseEphemerisInterpolant")]
#[derive(Clone)]
pub struct PyPreciseEphemerisInterpolant {
    pub(crate) inner: PreciseEphemerisInterpolant,
}

#[pymethods]
impl PyPreciseEphemerisInterpolant {
    /// Build a cached interpolant from a parsed SP3 product.
    #[staticmethod]
    #[pyo3(signature = (source, gap_threshold_factor = None))]
    fn from_sp3(source: &PySp3, gap_threshold_factor: Option<f64>) -> PyResult<Self> {
        let mut inner = PreciseEphemerisInterpolant::from_sp3(&source.inner);
        if let Some(factor) = gap_threshold_factor {
            let options = Sp3InterpolationOptions::new(factor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            inner = inner.with_interpolation_options(options);
        }
        Ok(Self { inner })
    }

    /// Build a cached interpolant directly from precise-ephemeris samples.
    #[staticmethod]
    #[pyo3(signature = (samples, gap_threshold_factor = None))]
    fn from_samples(
        py: Python<'_>,
        samples: Vec<Py<PyPreciseEphemerisSample>>,
        gap_threshold_factor: Option<f64>,
    ) -> PyResult<Self> {
        let samples = samples.iter().map(|s| s.borrow(py).to_core());
        let mut inner =
            PreciseEphemerisInterpolant::from_samples(samples).map_err(|error| match error {
                PreciseInterpolantError::Samples(sample_error) => {
                    precise_samples_error(py, sample_error)
                }
            })?;
        if let Some(factor) = gap_threshold_factor {
            let options = Sp3InterpolationOptions::new(factor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            inner = inner.with_interpolation_options(options);
        }
        Ok(Self { inner })
    }

    #[staticmethod]
    fn from_samples_with_accuracy(
        py: Python<'_>,
        samples: Vec<Py<PyPreciseEphemerisSample>>,
        accuracy: Vec<Py<PyPreciseEphemerisAccuracySample>>,
    ) -> PyResult<Self> {
        let samples = samples.iter().map(|sample| sample.borrow(py).to_core());
        let accuracy = accuracy.iter().map(|item| item.borrow(py).to_core());
        let inner = PreciseEphemerisInterpolant::from_samples_with_accuracy(samples, accuracy)
            .map_err(|error| match error {
                PreciseInterpolantError::Samples(sample_error) => {
                    precise_samples_error(py, sample_error)
                }
            })?;
        Ok(Self { inner })
    }

    /// Build a cached interpolant from an existing sample-backed source.
    #[staticmethod]
    #[pyo3(signature = (source, gap_threshold_factor = None))]
    fn from_precise_ephemeris_samples(
        source: &PyPreciseEphemerisSamples,
        gap_threshold_factor: Option<f64>,
    ) -> PyResult<Self> {
        let mut inner = PreciseEphemerisInterpolant::from_precise_ephemeris_samples(&source.inner);
        if let Some(factor) = gap_threshold_factor {
            let options = Sp3InterpolationOptions::new(factor)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            inner = inner.with_interpolation_options(options);
        }
        Ok(Self { inner })
    }

    /// SP3 interpolation gap threshold factor carried by this interpolant.
    #[getter]
    fn gap_threshold_factor(&self) -> f64 {
        self.inner.interpolation_options().gap_threshold_factor()
    }

    /// Return a copy of this interpolant with an explicit gap threshold factor.
    fn with_interpolation_options(&self, gap_threshold_factor: f64) -> PyResult<Self> {
        let options = Sp3InterpolationOptions::new(gap_threshold_factor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: self.inner.clone().with_interpolation_options(options),
        })
    }

    /// Time scale of the source epochs used to build this handle.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale().into()
    }

    /// Satellite tokens this handle can interpolate, ascending.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner.satellites().map(|sat| sat.to_string()).collect()
    }

    /// Interpolate one satellite state at seconds since J2000.
    ///
    /// Returns an [`Sp3State`] whose position is ECEF metres and whose clock is
    /// seconds or `None`. Raises `ValueError` for a malformed satellite token and
    /// `SolveError` for out-of-coverage or missing-source cases.
    fn position_at_j2000_seconds(
        &self,
        satellite: &str,
        epoch_j2000_s: f64,
    ) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        let state = self
            .inner
            .position_at_j2000_seconds(sat, epoch_j2000_s)
            .map_err(|error| attach_core_error_detail(to_solve_err(error.to_string()), &error))?;
        Ok(PySp3State::from_state(state))
    }

    fn position_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
    ) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        self.inner
            .position_at_epoch_query(sat, &epoch.inner)
            .map(PySp3State::from_state)
            .map_err(|error| attach_core_error_detail(to_solve_err(error.to_string()), &error))
    }

    fn selected_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<PyEphemerisQueryState>> {
        source_state_at_epoch_query(&self.inner, satellite, epoch, selection_epoch)
    }

    fn transmit_epoch_clock_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<(f64, Option<&'static str>)>> {
        source_transmit_clock_at_epoch_query(&self.inner, satellite, epoch, selection_epoch)
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        satellite: &str,
        state_epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<f64> {
        source_variance_at_epoch_query(&self.inner, satellite, state_epoch, selection_epoch)
    }

    fn clock_relativity_for_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        position_ecef_m: [f64; 3],
    ) -> PyResult<PyClockRelativity> {
        source_clock_relativity_at_epoch_query(&self.inner, satellite, epoch, position_ecef_m)
    }

    /// Evaluate ECEF states for parallel satellite and epoch arrays.
    ///
    /// `satellites[i]` is evaluated at `epochs_j2000_s[i]`. The result keeps
    /// contiguous position and clock arrays plus per-element status and error
    /// text and structured details.
    fn observable_states_at_j2000_s(
        &self,
        satellites: Vec<String>,
        epochs_j2000_s: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyObservableStateBatch> {
        let satellites = parse_satellites(&satellites)?;
        let epochs = epochs_j2000_s
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .observable_states_at_j2000_s(&satellites, epochs)
            .map(PyObservableStateBatch::from)
            .map_err(observable_state_batch_error)
    }

    /// Evaluate ECEF states for many satellites at one shared J2000-second epoch.
    fn observable_states_at_shared_j2000_s(
        &self,
        satellites: Vec<String>,
        epoch_j2000_s: f64,
    ) -> PyResult<PyObservableStateBatch> {
        let satellites = parse_satellites(&satellites)?;
        Ok(self
            .inner
            .observable_states_at_shared_j2000_s(&satellites, epoch_j2000_s)
            .into())
    }

    fn __repr__(&self) -> String {
        format!(
            "PreciseEphemerisInterpolant(satellites={})",
            self.inner.satellites().count()
        )
    }
}

/// Precise-interpolant artifact opened from canonical store bytes.
#[pyclass(module = "sidereon._sidereon", name = "PreciseInterpolantArtifact")]
pub struct PyPreciseInterpolantArtifact {
    inner: MmapPreciseEphemerisInterpolant<'static>,
}

impl PyPreciseInterpolantArtifact {
    fn from_vec(bytes: Vec<u8>) -> PyResult<Self> {
        let truncated = artifact_looks_truncated(&bytes);
        MmapPreciseEphemerisInterpolant::from_vec(bytes)
            .map(|inner| Self { inner })
            .map_err(|err| precise_artifact_error(err, truncated))
    }
}

#[pymethods]
impl PyPreciseInterpolantArtifact {
    /// Open an owned artifact from Python bytes.
    #[staticmethod]
    fn from_bytes(source: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(bytes) = source.downcast::<PyBytes>() {
            return Self::from_vec(bytes.as_bytes().to_vec());
        }
        if let Ok(buf) = source.downcast::<PyByteArray>() {
            // SAFETY: the bytes are copied before control returns to Python.
            return Self::from_vec(unsafe { buf.as_bytes() }.to_vec());
        }
        Err(PyTypeError::new_err(
            "PreciseInterpolantArtifact.from_bytes expects bytes or bytearray",
        ))
    }

    /// Read and open an artifact from a filesystem path.
    #[staticmethod]
    fn from_path(path: PathBuf) -> PyResult<Self> {
        let bytes = std::fs::read(&path)?;
        Self::from_vec(bytes)
    }

    /// Open an artifact from disk using a caller-attested checksum.
    ///
    /// The claim must fit an unsigned 64-bit integer and match the checksum in
    /// the artifact header. The payload is not hashed; call [`Self.verify`] to
    /// escalate the handle to verified provenance.
    #[staticmethod]
    fn from_path_attested(path: PathBuf, claimed_checksum64: &Bound<'_, PyAny>) -> PyResult<Self> {
        let claimed_checksum64 = parse_claimed_checksum64(claimed_checksum64)?;
        MmapPreciseEphemerisInterpolant::from_path_attested(path, claimed_checksum64)
            .map(|inner| Self { inner })
            .map_err(precise_artifact_path_error)
    }

    /// Artifact byte length.
    #[getter]
    fn byte_len(&self) -> usize {
        self.inner.as_bytes().len()
    }

    /// File-level checksum stored by the artifact format.
    #[getter]
    fn checksum64(&self) -> u64 {
        self.inner.checksum64()
    }

    /// Whether the checksum was measured by the library or attested by the
    /// caller.
    #[getter]
    fn digest_provenance(&self) -> &'static str {
        match self.inner.digest_provenance() {
            DigestProvenance::Verified => "verified",
            DigestProvenance::Attested => "attested",
        }
    }

    /// Verify file-level and per-satellite payload checksums for this handle.
    ///
    /// Success changes [`Self.digest_provenance`] to `"verified"`.
    fn verify(&mut self) -> PyResult<()> {
        self.inner
            .verify()
            .map_err(precise_artifact_error_without_bytes)
    }

    /// Time scale of the stored epoch axis.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale().into()
    }

    /// SP3 interpolation gap threshold factor read from the artifact header.
    #[getter]
    fn gap_threshold_factor(&self) -> f64 {
        self.inner.interpolation_options().gap_threshold_factor()
    }

    /// Satellite tokens present in the artifact.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner
            .satellites()
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Copy the backing artifact bytes into a Python `bytes` object.
    fn as_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.as_bytes())
    }

    /// Interpolate one satellite state at seconds since J2000.
    fn position_at_j2000_seconds(
        &self,
        satellite: &str,
        epoch_j2000_s: f64,
    ) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        let state = self
            .inner
            .position_at_j2000_seconds(sat, epoch_j2000_s)
            .map_err(|error| attach_core_error_detail(to_solve_err(error.to_string()), &error))?;
        Ok(PySp3State::from_state(state))
    }

    fn position_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
    ) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        self.inner
            .position_at_epoch_query(sat, &epoch.inner)
            .map(PySp3State::from_state)
            .map_err(|error| attach_core_error_detail(to_solve_err(error.to_string()), &error))
    }

    fn selected_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<PyEphemerisQueryState>> {
        source_state_at_epoch_query(&self.inner, satellite, epoch, selection_epoch)
    }

    fn transmit_epoch_clock_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<(f64, Option<&'static str>)>> {
        source_transmit_clock_at_epoch_query(&self.inner, satellite, epoch, selection_epoch)
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        satellite: &str,
        state_epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<f64> {
        source_variance_at_epoch_query(&self.inner, satellite, state_epoch, selection_epoch)
    }

    fn clock_relativity_for_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        position_ecef_m: [f64; 3],
    ) -> PyResult<PyClockRelativity> {
        source_clock_relativity_at_epoch_query(&self.inner, satellite, epoch, position_ecef_m)
    }

    /// Evaluate ECEF states for parallel satellite and epoch arrays.
    fn observable_states_at_j2000_s(
        &self,
        satellites: Vec<String>,
        epochs_j2000_s: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyObservableStateBatch> {
        let satellites = parse_satellites(&satellites)?;
        let epochs = epochs_j2000_s
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.inner
            .observable_states_at_j2000_s(&satellites, epochs)
            .map(PyObservableStateBatch::from)
            .map_err(observable_state_batch_error)
    }

    /// Evaluate ECEF states for many satellites at one shared J2000-second epoch.
    fn observable_states_at_shared_j2000_s(
        &self,
        satellites: Vec<String>,
        epoch_j2000_s: f64,
    ) -> PyResult<PyObservableStateBatch> {
        let satellites = parse_satellites(&satellites)?;
        Ok(self
            .inner
            .observable_states_at_shared_j2000_s(&satellites, epoch_j2000_s)
            .into())
    }

    fn __repr__(&self) -> String {
        format!(
            "PreciseInterpolantArtifact(byte_len={}, satellites={})",
            self.inner.as_bytes().len(),
            self.inner.satellites().len()
        )
    }
}

/// Build a split-Julian-date [`Instant`] from J2000 seconds in a time scale.
///
/// The residual day fraction is split off `jd_whole` so it stays within one day,
/// matching the SP3 parser's node axis after the shared floor.
fn instant_from_j2000_seconds(
    seconds: f64,
    scale: sidereon_core::astro::time::TimeScale,
) -> PyResult<Instant> {
    if !seconds.is_finite() {
        return Err(PyValueError::new_err("epoch_j2000_seconds must be finite"));
    }
    let days = seconds / SECONDS_PER_DAY;
    let whole_days = days.floor();
    let split = JulianDateSplit::new(J2000_JD + whole_days, days - whole_days)
        .map_err(|e| PyValueError::new_err(format!("invalid epoch: {e}")))?;
    Ok(Instant {
        scale,
        repr: InstantRepr::JulianDate(split),
    })
}

fn artifact_looks_truncated(bytes: &[u8]) -> bool {
    const HEADER_LEN: usize = 64;
    const MAGIC: &[u8; 8] = b"PEMAP001";
    const TOTAL_LEN_OFFSET: usize = 32;

    if bytes.len() < HEADER_LEN {
        return true;
    }
    if &bytes[..MAGIC.len()] == MAGIC {
        let mut raw = [0_u8; 8];
        raw.copy_from_slice(&bytes[TOTAL_LEN_OFFSET..TOTAL_LEN_OFFSET + 8]);
        let total_len = u64::from_le_bytes(raw) as usize;
        return total_len > bytes.len();
    }
    false
}

fn precise_artifact_error_without_bytes(err: PreciseInterpolantStoreError) -> PyErr {
    precise_artifact_error(err, false)
}

fn precise_artifact_path_error(err: PreciseInterpolantStoreError) -> PyErr {
    match err {
        PreciseInterpolantStoreError::Io { .. } => PyOSError::new_err(err.to_string()),
        other => precise_artifact_error_without_bytes(other),
    }
}

fn precise_artifact_error(err: PreciseInterpolantStoreError, truncated: bool) -> PyErr {
    if truncated {
        return PreciseInterpolantArtifactTruncatedError::new_err(err.to_string());
    }
    match err {
        PreciseInterpolantStoreError::Checksum { .. }
        | PreciseInterpolantStoreError::SatelliteChecksum { .. } => {
            PreciseInterpolantArtifactCorruptError::new_err(err.to_string())
        }
        PreciseInterpolantStoreError::Parse { ref reason }
            if reason.contains("past store length") || reason.contains("out of bounds") =>
        {
            PreciseInterpolantArtifactTruncatedError::new_err(err.to_string())
        }
        other => PreciseInterpolantArtifactError::new_err(other.to_string()),
    }
}

/// Parse an SP3-c or SP3-d product from in-memory bytes or a file path.
///
/// `source` may be:
/// - `bytes` / `bytearray`: the full, already-decompressed file content, parsed
///   directly; or
/// - a path (`str` or `os.PathLike`): the file is read and parsed.
///
/// `gap_threshold_factor` optionally configures the product-carried SP3
/// coverage-gap interpolation policy; `None` leaves the core default (1.5).
///
/// Raises [`Sp3ParseError`](crate::Sp3ParseError) on malformed content, `OSError`
/// if the path cannot be read, `ValueError` if `gap_threshold_factor` is not
/// finite or <= 1.0, and `TypeError` if `source` is neither bytes nor a path.
#[pyfunction]
#[pyo3(signature = (source, gap_threshold_factor = None))]
fn load_sp3(source: &Bound<'_, PyAny>, gap_threshold_factor: Option<f64>) -> PyResult<PySp3> {
    // bytes-like first, so a `bytes` argument keeps the prior "content" meaning.
    let mut inner = if let Ok(bytes) = source.downcast::<PyBytes>() {
        sidereon::load_sp3(bytes.as_bytes()).map_err(to_sp3_err)?
    } else if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        sidereon::load_sp3(unsafe { buf.as_bytes() }).map_err(to_sp3_err)?
    } else {
        // Otherwise treat it as a path (str / os.PathLike via PyO3's fspath support).
        let path: PathBuf = source.extract().map_err(|_| {
            PyValueError::new_err("load_sp3 expects bytes, bytearray, or a path (str/os.PathLike)")
        })?;
        let data = std::fs::read(&path)?;
        sidereon::load_sp3(&data).map_err(to_sp3_err)?
    };
    if let Some(factor) = gap_threshold_factor {
        let options = Sp3InterpolationOptions::new(factor)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        inner = inner.with_interpolation_options(options);
    }
    Ok(PySp3 { inner })
}

fn exact_sp3_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn exact_validation_error(py: Python<'_>, error: ExactSp3ValidationError) -> PyErr {
    let python_error = PyValueError::new_err(error.to_string());
    let object = python_error.value(py);
    let _ = object.setattr("kind", "exact_sp3_validation");
    if let ExactSp3ValidationError::DeclaredStartMismatch {
        requested_j2000_s,
        declared_j2000_s,
        requested_tick,
        declared_tick,
    } = error
    {
        let detail = PyDict::new(py);
        let _ = detail.set_item("kind", "declared_start_mismatch");
        if requested_j2000_s.is_finite() {
            let _ = detail.set_item("requested_j2000_s", requested_j2000_s);
        } else {
            let _ = detail.set_item("requested_j2000_s", format!("{requested_j2000_s:?}"));
        }
        if declared_j2000_s.is_finite() {
            let _ = detail.set_item("declared_j2000_s", declared_j2000_s);
        } else {
            let _ = detail.set_item("declared_j2000_s", format!("{declared_j2000_s:?}"));
        }
        let _ = detail.set_item("requested_tick", requested_tick.to_string());
        let _ = detail.set_item("declared_tick", declared_tick.map(|tick| tick.to_string()));
        let _ = object.setattr("detail", detail);
        let _ = object.setattr("kind", "declared_start_mismatch");
    }
    python_error
}

#[allow(clippy::too_many_arguments)]
fn exact_sp3_request(
    year: i32,
    month: u8,
    day: u8,
    issue: Option<&str>,
    span: &str,
    sample: &str,
    expected_agency: Option<&str>,
    identity_json: Option<&str>,
) -> PyResult<ExactSp3Request> {
    let mut request = if let Some(identity_json) = identity_json {
        ExactSp3Request::from_identity(&crate::exact_cache::identity(identity_json)?)
            .map_err(exact_sp3_error)?
    } else {
        ExactSp3Request::new(
            ProductDate::new(year, month, day).map_err(exact_sp3_error)?,
            issue,
            span,
            sample,
        )
        .map_err(exact_sp3_error)?
    };
    if let Some(agency) = expected_agency {
        request = request
            .with_expected_agency(agency)
            .map_err(exact_sp3_error)?;
    }
    Ok(request)
}

fn exact_sp3_coverage(coverage: ExactSp3Coverage) -> &'static str {
    match coverage {
        ExactSp3Coverage::HalfOpen => "half_open",
        ExactSp3Coverage::Inclusive => "inclusive",
    }
}

type ExactSp3RequestFields = (
    i32,
    u8,
    u8,
    Option<String>,
    String,
    String,
    Option<String>,
    Option<String>,
);

#[pyfunction]
fn _exact_sp3_request_from_identity(identity_json: &str) -> PyResult<ExactSp3RequestFields> {
    let request = ExactSp3Request::from_identity(&crate::exact_cache::identity(identity_json)?)
        .map_err(exact_sp3_error)?;
    let date = request.date();
    Ok((
        date.year,
        date.month,
        date.day,
        request.issue().map(ToOwned::to_owned),
        request.span().to_owned(),
        request.sample().to_owned(),
        request.format_version().map(ToOwned::to_owned),
        request.expected_agency().map(ToOwned::to_owned),
    ))
}

#[pyfunction]
#[pyo3(signature = (year, month, day, issue, span, sample, expected_agency=None, identity_json=None))]
#[allow(clippy::too_many_arguments)]
fn _validate_exact_sp3_request(
    year: i32,
    month: u8,
    day: u8,
    issue: Option<&str>,
    span: &str,
    sample: &str,
    expected_agency: Option<&str>,
    identity_json: Option<&str>,
) -> PyResult<()> {
    exact_sp3_request(
        year,
        month,
        day,
        issue,
        span,
        sample,
        expected_agency,
        identity_json,
    )
    .map(|_| ())
}

#[pyfunction]
#[pyo3(signature = (content, year, month, day, issue, span, sample, expected_agency=None, identity_json=None))]
#[allow(clippy::too_many_arguments)]
fn _parse_exact_sp3(
    py: Python<'_>,
    content: &[u8],
    year: i32,
    month: u8,
    day: u8,
    issue: Option<&str>,
    span: &str,
    sample: &str,
    expected_agency: Option<&str>,
    identity_json: Option<&str>,
) -> PyResult<(PySp3, &'static str)> {
    let request = exact_sp3_request(
        year,
        month,
        day,
        issue,
        span,
        sample,
        expected_agency,
        identity_json,
    )?;
    let (sp3, coverage) = core_parse_exact_sp3(content, &request)
        .map_err(|error| exact_validation_error(py, error))?;
    Ok((PySp3 { inner: sp3 }, exact_sp3_coverage(coverage)))
}

#[pyfunction]
#[pyo3(signature = (sp3, year, month, day, issue, span, sample, expected_agency=None, identity_json=None))]
#[allow(clippy::too_many_arguments)]
fn _validate_exact_sp3(
    py: Python<'_>,
    sp3: &PySp3,
    year: i32,
    month: u8,
    day: u8,
    issue: Option<&str>,
    span: &str,
    sample: &str,
    expected_agency: Option<&str>,
    identity_json: Option<&str>,
) -> PyResult<&'static str> {
    let request = exact_sp3_request(
        year,
        month,
        day,
        issue,
        span,
        sample,
        expected_agency,
        identity_json,
    )?;
    core_validate_exact_sp3(&sp3.inner, &request)
        .map(exact_sp3_coverage)
        .map_err(|error| exact_validation_error(py, error))
}

/// Build deterministic precise-interpolant artifact bytes from an SP3 product.
#[pyfunction]
#[pyo3(signature = (sp3, gap_threshold_factor = None))]
fn build_precise_interpolant_artifact_bytes<'py>(
    py: Python<'py>,
    sp3: &PySp3,
    gap_threshold_factor: Option<f64>,
) -> PyResult<Bound<'py, PyBytes>> {
    sp3.precise_interpolant_artifact_bytes(py, gap_threshold_factor)
}

pub(crate) fn parse_satellites(tokens: &[String]) -> PyResult<Vec<GnssSatelliteId>> {
    tokens.iter().map(|token| parse_sat(token)).collect()
}

pub(crate) fn observable_state_batch_error(err: ObservablesError) -> PyErr {
    let python_error = match &err {
        ObservablesError::InvalidInput { .. } | ObservablesError::Media(_) => {
            PyValueError::new_err(err.to_string())
        }
        ObservablesError::NoEphemeris | ObservablesError::Ephemeris(_) => to_solve_err(err.clone()),
    };
    attach_observables_error_detail(python_error, &err)
}

pub(crate) fn with_observable_source<R>(
    source: &Bound<'_, PyAny>,
    f: impl FnOnce(&dyn ObservableEphemerisSource) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(sp3) = source.extract::<PyRef<'_, PySp3>>() {
        f(&sp3.inner)
    } else if let Ok(samples) = source.extract::<PyRef<'_, PyPreciseEphemerisSamples>>() {
        f(&samples.inner)
    } else if let Ok(interpolant) = source.extract::<PyRef<'_, PyPreciseEphemerisInterpolant>>() {
        f(&interpolant.inner)
    } else if let Ok(artifact) = source.extract::<PyRef<'_, PyPreciseInterpolantArtifact>>() {
        f(&artifact.inner)
    } else if let Ok(broadcast) = source.extract::<PyRef<'_, PyBroadcastEphemeris>>() {
        f(&broadcast.inner)
    } else {
        Err(PyTypeError::new_err(
            "source must be Sp3, PreciseEphemerisSamples, PreciseEphemerisInterpolant, PreciseInterpolantArtifact, or BroadcastEphemeris",
        ))
    }
}

/// Evaluate ECEF states for parallel satellite and epoch arrays.
///
/// `source` may be [`Sp3`], [`PreciseEphemerisSamples`],
/// [`PreciseEphemerisInterpolant`], [`PreciseInterpolantArtifact`], or
/// [`BroadcastEphemeris`](crate). The input arrays are parallel:
/// `satellites[i]` is evaluated at `epochs_j2000_s[i]`. The returned
/// [`ObservableStateBatch`] keeps ECEF position metres, clock seconds,
/// per-element status, and per-element error text.
#[pyfunction]
fn observable_states_at_j2000_s(
    source: &Bound<'_, PyAny>,
    satellites: Vec<String>,
    epochs_j2000_s: PyReadonlyArray1<'_, f64>,
) -> PyResult<PyObservableStateBatch> {
    let satellites = parse_satellites(&satellites)?;
    let epochs = epochs_j2000_s
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    with_observable_source(source, |source| {
        core_observable_states_at_j2000_s(source, &satellites, epochs)
            .map(PyObservableStateBatch::from)
            .map_err(observable_state_batch_error)
    })
}

/// Evaluate ECEF states for many satellites at one shared J2000-second epoch.
///
/// `source` may be [`Sp3`], [`PreciseEphemerisSamples`],
/// [`PreciseEphemerisInterpolant`], [`PreciseInterpolantArtifact`], or
/// [`BroadcastEphemeris`](crate). The result is index-aligned with `satellites`.
#[pyfunction]
fn observable_states_at_shared_j2000_s(
    source: &Bound<'_, PyAny>,
    satellites: Vec<String>,
    epoch_j2000_s: f64,
) -> PyResult<PyObservableStateBatch> {
    let satellites = parse_satellites(&satellites)?;
    with_observable_source(source, |source| {
        Ok(core_observable_states_at_shared_j2000_s(source, &satellites, epoch_j2000_s).into())
    })
}

#[pyfunction]
fn ephemeris_sample(
    source: &Bound<'_, PyAny>,
    satellites: Vec<String>,
    start_j2000_s: f64,
    stop_j2000_s: f64,
    step_s: f64,
) -> PyResult<Vec<PyEphemerisSampleRow>> {
    let satellites = satellites
        .iter()
        .map(|token| parse_sat(token))
        .collect::<PyResult<Vec<_>>>()?;

    let rows = if let Ok(sp3) = source.extract::<PyRef<'_, PySp3>>() {
        core_sample(&sp3.inner, &satellites, start_j2000_s, stop_j2000_s, step_s)
    } else if let Ok(samples) = source.extract::<PyRef<'_, PyPreciseEphemerisSamples>>() {
        core_sample(
            &samples.inner,
            &satellites,
            start_j2000_s,
            stop_j2000_s,
            step_s,
        )
    } else if let Ok(interpolant) = source.extract::<PyRef<'_, PyPreciseEphemerisInterpolant>>() {
        core_sample(
            &interpolant.inner,
            &satellites,
            start_j2000_s,
            stop_j2000_s,
            step_s,
        )
    } else if let Ok(artifact) = source.extract::<PyRef<'_, PyPreciseInterpolantArtifact>>() {
        core_sample(
            &artifact.inner,
            &satellites,
            start_j2000_s,
            stop_j2000_s,
            step_s,
        )
    } else if let Ok(broadcast) = source.extract::<PyRef<'_, PyBroadcastEphemeris>>() {
        core_sample(
            &broadcast.inner,
            &satellites,
            start_j2000_s,
            stop_j2000_s,
            step_s,
        )
    } else {
        return Err(PyValueError::new_err(
            "source must be Sp3, PreciseEphemerisSamples, PreciseEphemerisInterpolant, PreciseInterpolantArtifact, or BroadcastEphemeris",
        ));
    }
    .map_err(|error| {
        attach_observables_error_detail(to_solve_err(error.to_string()), &error)
    })?;

    Ok(rows.into_iter().map(PyEphemerisSampleRow::from).collect())
}

/// Estimate per-epoch clock-reference offsets of `other` relative to
/// `reference`.
#[pyfunction]
#[pyo3(signature = (reference, other, min_common=3))]
fn sp3_clock_reference_offset(
    reference: &PySp3,
    other: &PySp3,
    min_common: usize,
) -> Vec<PySp3ClockReferenceOffset> {
    core_clock_reference_offset(&reference.inner, &other.inner, min_common)
        .into_iter()
        .map(PySp3ClockReferenceOffset::from)
        .collect()
}

/// Return a copy of `other` with clocks aligned to `reference` where possible.
#[pyfunction]
#[pyo3(signature = (reference, other, min_common=3))]
fn align_sp3_clock_reference(reference: &PySp3, other: &PySp3, min_common: usize) -> PySp3 {
    PySp3 {
        inner: core_align_clock_reference(&reference.inner, &other.inner, min_common),
    }
}

fn instant_to_j2000_seconds(epoch: &Instant) -> Option<f64> {
    match epoch.repr {
        InstantRepr::JulianDate(jd) => Some(seconds_between_splits(
            jd.jd_whole,
            jd.fraction,
            J2000_JD,
            0.0,
        )),
        InstantRepr::Nanos(_) => None,
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySp3>()?;
    m.add_class::<PyEphemerisQueryState>()?;
    m.add_class::<PyClockRelativity>()?;
    m.add_class::<PySp3AccuracyValue>()?;
    m.add_class::<PySp3AccuracyCodeGroup>()?;
    m.add_class::<PySp3RawRecordAccuracy>()?;
    m.add_class::<PySp3PositionClockAccuracy>()?;
    m.add_class::<PySp3VelocityAccuracy>()?;
    m.add_class::<PySp3RecordAccuracy>()?;
    m.add_class::<PyPreciseEphemerisAccuracySample>()?;
    m.add_class::<PySp3EpochPrediction>()?;
    m.add_class::<PySp3PredictionSummary>()?;
    m.add_class::<PySp3Interpolation>()?;
    m.add_class::<PyEphemerisSampleStatus>()?;
    m.add_class::<PyEphemerisSampleRow>()?;
    m.add_class::<PyObservableStateElementStatus>()?;
    m.add_class::<PyObservableStateBatch>()?;
    m.add_class::<PySp3ClockReferenceOffset>()?;
    m.add_class::<PySp3State>()?;
    m.add_class::<PySp3ClockRecord>()?;
    m.add_class::<PySp3Header>()?;
    m.add_class::<PySp3WriteErrorDetail>()?;
    m.add_class::<PyPreciseEphemerisSample>()?;
    m.add_class::<PyPreciseEphemerisSamples>()?;
    m.add_class::<PyPreciseEphemerisInterpolant>()?;
    m.add_class::<PyPreciseInterpolantArtifact>()?;
    m.add(
        "OBSERVABLE_STATE_MISSING_POSITION_ECEF_M",
        OBSERVABLE_STATE_MISSING_POSITION_ECEF_M,
    )?;
    m.add_function(wrap_pyfunction!(load_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(_exact_sp3_request_from_identity, m)?)?;
    m.add_function(wrap_pyfunction!(_validate_exact_sp3_request, m)?)?;
    m.add_function(wrap_pyfunction!(_parse_exact_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(_validate_exact_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(
        build_precise_interpolant_artifact_bytes,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(observable_states_at_j2000_s, m)?)?;
    m.add_function(wrap_pyfunction!(observable_states_at_shared_j2000_s, m)?)?;
    m.add_function(wrap_pyfunction!(ephemeris_sample, m)?)?;
    m.add_function(wrap_pyfunction!(sp3_clock_reference_offset, m)?)?;
    m.add_function(wrap_pyfunction!(align_sp3_clock_reference, m)?)?;
    Ok(())
}

#[cfg(test)]
mod precise_samples_error_tests {
    use super::*;

    #[test]
    fn invalid_accuracy_from_valid_samples_keeps_typed_satellite_payload() {
        let satellite = "G07".parse::<GnssSatelliteId>().unwrap();
        let epochs = [
            instant_from_j2000_seconds(0.0, sidereon_core::astro::time::TimeScale::Gpst).unwrap(),
            instant_from_j2000_seconds(1.0, sidereon_core::astro::time::TimeScale::Gpst).unwrap(),
        ];
        let samples = epochs
            .map(|epoch| PreciseEphemerisSample::new(satellite, epoch, [1.0, 2.0, 3.0], None));
        let invalid_accuracy = samples.map(|sample| {
            PreciseEphemerisAccuracySample::new(
                sample.sat,
                sample.epoch,
                [
                    Sp3AccuracyValue::Known(f64::NAN),
                    Sp3AccuracyValue::Unknown,
                    Sp3AccuracyValue::Unknown,
                ],
                Sp3AccuracyValue::Unknown,
            )
        });
        let valid_accuracy = samples.map(|sample| {
            PreciseEphemerisAccuracySample::new(
                sample.sat,
                sample.epoch,
                [Sp3AccuracyValue::Known(1.0); 3],
                Sp3AccuracyValue::Known(1.0),
            )
        });

        assert!(
            PreciseEphemerisSamples::from_samples_with_accuracy(samples, valid_accuracy).is_ok()
        );

        let error = PreciseEphemerisSamples::from_samples_with_accuracy(samples, invalid_accuracy)
            .unwrap_err();
        Python::with_gil(|py| {
            let python_error = precise_samples_error(py, error);
            assert!(python_error.is_instance_of::<InvalidAccuracyValueError>(py));
            let value = python_error.value(py);
            assert_eq!(
                value.getattr("kind").unwrap().extract::<String>().unwrap(),
                "invalid_accuracy_value"
            );
            assert_eq!(
                value
                    .getattr("satellite")
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "G07"
            );
        });
    }
}
