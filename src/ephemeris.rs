//! Ephemeris binding: the parsed SP3 precise product, arbitrary-epoch
//! position/clock interpolation, per-record state access, and serialization.
//!
//! Marshals SP3 bytes (or a file path) into [`sidereon_core::ephemeris::Sp3`] and
//! exposes its query surface Pythonically: the node epoch axis as J2000 seconds,
//! batched interpolation to numpy `(n, 3)` / `(n,)` arrays, the exact per-record
//! state, and the deterministic SP3 text writer. No modeling lives here: the
//! interpolation is the engine's `position_at_j2000_seconds` recipe and the writer
//! is `to_sp3_string`, so the numbers and bytes are exactly what `sidereon-core`
//! produces. The per-query loop runs inside Rust, one FFI crossing per call.

use std::path::PathBuf;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyModule};

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
    EphemerisSampleStatus, ExactSp3Coverage, ExactSp3Request, MmapPreciseEphemerisInterpolant,
    ObservableEphemerisSource, ObservableStateBatch, ObservableStateElementStatus,
    ObservablesError, PreciseEphemerisInterpolant, PreciseEphemerisSample, PreciseEphemerisSamples,
    PreciseInterpolantStoreError, Sp3, OBSERVABLE_STATE_MISSING_POSITION_ECEF_M,
};
use sidereon_core::Error as CoreError;
use sidereon_core::GnssSatelliteId;

use crate::frames::PyTimeScale;
use crate::marshal::rows3_to_array;
use crate::rinex::PyBroadcastEphemeris;
use crate::{
    np_array, to_solve_err, to_sp3_err, PreciseInterpolantArtifactCorruptError,
    PreciseInterpolantArtifactError, PreciseInterpolantArtifactTruncatedError,
};

/// Seconds in one day, for the J2000-second <-> split-Julian-date reconstruction.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// A parsed SP3 precise-ephemeris product.
///
/// Construct with [`load_sp3`]. Query satellite states by epoch
/// ([`Sp3.interpolate`] for arbitrary epochs, [`Sp3.state`] for the exact parsed
/// records), read the node epoch grid with [`Sp3.epochs_j2000_seconds`], and
/// serialize back to SP3 text with [`Sp3.to_sp3_string`]. Also passed to the
/// solve functions as the ephemeris source. Wraps
/// [`sidereon_core::ephemeris::Sp3`] unchanged.
#[pyclass(module = "sidereon._sidereon", name = "Sp3")]
pub struct PySp3 {
    pub(crate) inner: Sp3,
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
fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|e| PyValueError::new_err(format!("invalid satellite token {token:?}: {e}")))
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

    /// The product's parsed epochs as seconds since J2000, in the file's own time
    /// scale, ascending, as a numpy `(n,)` `float64` array.
    ///
    /// This is the exact query axis [`Sp3.interpolate`] consumes; read it, form
    /// query times on it (e.g. midpoints, a finer grid), and pass them straight
    /// back without a Julian-date round-trip.
    #[getter]
    fn epochs_j2000_seconds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.epochs_j2000_seconds())
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
            let state = self.inner.position_at_j2000_seconds(sat, q).map_err(|e| {
                match e {
                    // The satellite simply is not in the product: bad input.
                    CoreError::UnknownSatellite(id) => {
                        PyValueError::new_err(format!("satellite {id} is not in the product"))
                    }
                    // Out of coverage / too few nodes: a solve condition.
                    other => to_solve_err(format!("interpolation at j2000 second {q}: {other}")),
                }
            })?;
            positions.push(state.position.as_array());
            clocks.push(state.clock_s.unwrap_or(f64::NAN));
        }
        Ok(PySp3Interpolation { positions, clocks })
    }

    /// The exact parsed state of `satellite` at the record with index
    /// `epoch_index` (no interpolation).
    ///
    /// Returns an [`Sp3State`]. Raises `IndexError` if `epoch_index` is past the
    /// last epoch and `KeyError` if the satellite has no record at that epoch.
    fn state(&self, satellite: &str, epoch_index: usize) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        let state = self.inner.state(sat, epoch_index).map_err(|e| match e {
            CoreError::EpochOutOfRange => {
                PyIndexError::new_err(format!("epoch index {epoch_index} out of range"))
            }
            CoreError::UnknownSatellite(id) => PyKeyError::new_err(format!(
                "satellite {id} has no record at epoch {epoch_index}"
            )),
            other => to_solve_err(other.to_string()),
        })?;
        Ok(PySp3State {
            position: state.position.as_array(),
            clock_s: state.clock_s,
            velocity: state.velocity.map(|v| v.as_array()),
            clock_event: state.flags.clock_event,
            clock_predicted: state.flags.clock_predicted,
            maneuver: state.flags.maneuver,
            orbit_predicted: state.flags.orbit_predicted,
        })
    }

    /// Serialize this product to standard SP3 text (the format named by its header
    /// version, `c` or `d`). Pure and deterministic: the same product always
    /// produces byte-identical text, and re-parsing the output round-trips the
    /// epochs, satellites, positions, and clocks.
    fn to_sp3_string(&self) -> String {
        self.inner.to_sp3_string()
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

    /// Build deterministic memory-mappable precise-interpolant artifact bytes.
    fn precise_interpolant_artifact_bytes<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self
            .inner
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
/// position sentinel and carry their error text in `element_results`.
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
/// velocity (metres per second) or `None` for a position-only product. The four
/// status flags are surfaced verbatim from the record.
#[pyclass(module = "sidereon._sidereon", name = "Sp3State")]
pub struct PySp3State {
    position: [f64; 3],
    clock_s: Option<f64>,
    velocity: Option<[f64; 3]>,
    clock_event: bool,
    clock_predicted: bool,
    maneuver: bool,
    orbit_predicted: bool,
}

impl PySp3State {
    /// Build from a core interpolated/parsed state, for the staleness selection
    /// layer's `position_at_j2000_seconds` query.
    pub(crate) fn from_state(state: sidereon_core::ephemeris::Sp3State) -> Self {
        Self {
            position: state.position.as_array(),
            clock_s: state.clock_s,
            velocity: state.velocity.map(|v| v.as_array()),
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
    /// `ValueError` if the set is empty, a satellite has a single sample, a
    /// satellite's epochs are not strictly increasing, the samples mix time
    /// scales, a sample is non-finite, or an epoch is not representable as J2000
    /// seconds.
    #[staticmethod]
    fn from_samples(py: Python<'_>, samples: Vec<Py<PyPreciseEphemerisSample>>) -> PyResult<Self> {
        let samples = samples.iter().map(|s| s.borrow(py).to_core());
        let inner = PreciseEphemerisSamples::from_samples(samples)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
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
    inner: PreciseEphemerisInterpolant,
}

#[pymethods]
impl PyPreciseEphemerisInterpolant {
    /// Build a cached interpolant from a parsed SP3 product.
    #[staticmethod]
    fn from_sp3(source: &PySp3) -> Self {
        Self {
            inner: PreciseEphemerisInterpolant::from_sp3(&source.inner),
        }
    }

    /// Build a cached interpolant directly from precise-ephemeris samples.
    #[staticmethod]
    fn from_samples(py: Python<'_>, samples: Vec<Py<PyPreciseEphemerisSample>>) -> PyResult<Self> {
        let samples = samples.iter().map(|s| s.borrow(py).to_core());
        let inner = PreciseEphemerisInterpolant::from_samples(samples)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Build a cached interpolant from an existing sample-backed source.
    #[staticmethod]
    fn from_precise_ephemeris_samples(source: &PyPreciseEphemerisSamples) -> Self {
        Self {
            inner: PreciseEphemerisInterpolant::from_precise_ephemeris_samples(&source.inner),
        }
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
            .map_err(to_solve_err)?;
        Ok(PySp3State::from_state(state))
    }

    /// Evaluate ECEF states for parallel satellite and epoch arrays.
    ///
    /// `satellites[i]` is evaluated at `epochs_j2000_s[i]`. The result keeps
    /// contiguous position and clock arrays plus per-element status and error
    /// text.
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

    /// Time scale of the stored epoch axis.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale().into()
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
            .map_err(to_solve_err)?;
        Ok(PySp3State::from_state(state))
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
/// Raises [`Sp3ParseError`](crate::Sp3ParseError) on malformed content, `OSError`
/// if the path cannot be read, and `TypeError` if `source` is neither bytes nor a
/// path.
#[pyfunction]
fn load_sp3(source: &Bound<'_, PyAny>) -> PyResult<PySp3> {
    // bytes-like first, so a `bytes` argument keeps the prior "content" meaning.
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        let inner = sidereon::load_sp3(bytes.as_bytes()).map_err(to_sp3_err)?;
        return Ok(PySp3 { inner });
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        let inner = sidereon::load_sp3(unsafe { buf.as_bytes() }).map_err(to_sp3_err)?;
        return Ok(PySp3 { inner });
    }
    // Otherwise treat it as a path (str / os.PathLike via PyO3's fspath support).
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_sp3 expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    let inner = sidereon::load_sp3(&data).map_err(to_sp3_err)?;
    Ok(PySp3 { inner })
}

fn exact_sp3_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
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
    let (sp3, coverage) = core_parse_exact_sp3(content, &request).map_err(exact_sp3_error)?;
    Ok((PySp3 { inner: sp3 }, exact_sp3_coverage(coverage)))
}

#[pyfunction]
#[pyo3(signature = (sp3, year, month, day, issue, span, sample, expected_agency=None, identity_json=None))]
#[allow(clippy::too_many_arguments)]
fn _validate_exact_sp3(
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
        .map_err(exact_sp3_error)
}

/// Build deterministic precise-interpolant artifact bytes from an SP3 product.
#[pyfunction]
fn build_precise_interpolant_artifact_bytes<'py>(
    py: Python<'py>,
    sp3: &PySp3,
) -> PyResult<Bound<'py, PyBytes>> {
    sp3.precise_interpolant_artifact_bytes(py)
}

pub(crate) fn parse_satellites(tokens: &[String]) -> PyResult<Vec<GnssSatelliteId>> {
    tokens.iter().map(|token| parse_sat(token)).collect()
}

pub(crate) fn observable_state_batch_error(err: ObservablesError) -> PyErr {
    match err {
        ObservablesError::InvalidInput { .. } | ObservablesError::Media(_) => {
            PyValueError::new_err(err.to_string())
        }
        ObservablesError::NoEphemeris | ObservablesError::Ephemeris(_) => to_solve_err(err),
    }
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
    .map_err(to_solve_err)?;

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
    m.add_class::<PySp3EpochPrediction>()?;
    m.add_class::<PySp3PredictionSummary>()?;
    m.add_class::<PySp3Interpolation>()?;
    m.add_class::<PyEphemerisSampleStatus>()?;
    m.add_class::<PyEphemerisSampleRow>()?;
    m.add_class::<PyObservableStateElementStatus>()?;
    m.add_class::<PyObservableStateBatch>()?;
    m.add_class::<PySp3ClockReferenceOffset>()?;
    m.add_class::<PySp3State>()?;
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
