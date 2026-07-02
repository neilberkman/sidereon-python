//! Observation-domain math binding.
//!
//! This module exposes GNSS observable helpers without reimplementing any
//! numeric policy. Every lookup delegates to `sidereon-core`.

use std::collections::BTreeMap;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::carrier_phase::{
    self, ArcEpoch, CarrierPhaseError, CycleSlipOptions, IonoFreeSmoothResult, SlipReason,
    SlipResult, SmoothCodeResult, DEFAULT_HATCH_WINDOW_CAP,
};
use sidereon_core::combinations::{self, IonosphereFreeError, PseudorangeDropReason};
use sidereon_core::ephemeris::{PreciseEphemerisSamples, Sp3};
use sidereon_core::frequencies::{
    default_iono_free_pair, frequency_hz, rinex_band_frequency_hz, rinex_band_wavelength_m,
    rinex_observation_frequency_hz as core_rinex_observation_frequency_hz,
    rinex_observation_wavelength_m as core_rinex_observation_wavelength_m, wavelength_m,
    CarrierBand, CarrierPair,
};
use sidereon_core::observables::{
    predict, predict_batch, predict_batch_parallel, predict_ranges as core_predict_ranges,
    ObservableEphemerisSource, ObservablesError, PredictOptions, PredictRequest,
    PredictedObservables, RangePrediction, RangePredictionRequest,
};
use sidereon_core::quality::{
    self, PseudorangeVarianceModel, PseudorangeVarianceOptions, QualityError, RaimWeights,
    WeightEntry,
};
use sidereon_core::signal::{
    self, AcquisitionGrid, AcquisitionOptions, AcquisitionResult, CorrelateOptions,
    CorrelationResult, IqSample, ReplicaOptions, SignalError,
};
use sidereon_core::velocity::{
    self, VelocityObservable, VelocityObservation, VelocitySolution, VelocitySolveOptions,
};
use sidereon_core::GnssSatelliteId;

use crate::marshal::{
    fixed_array, option_py_or_default, rows3_from_array, EmptyPolicy, FinitePolicy, PyGnssSystem,
};
use crate::rinex::PyBroadcastEphemeris;
use crate::{np_array, to_solve_err, PyPreciseEphemerisSamples, PySp3};

type PyPseudorangeObservation = (String, f64);
type PyDroppedPseudorange = (String, PyPseudorangeDropReason);
type PyPseudorangeCombinationResult = (Vec<PyPseudorangeObservation>, Vec<PyDroppedPseudorange>);

/// A canonical GNSS carrier band.
#[pyclass(module = "sidereon._sidereon", name = "CarrierBand", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyCarrierBand {
    /// GPS/QZSS L1.
    L1,
    /// GPS/QZSS L2.
    L2,
    /// GPS/QZSS L5.
    L5,
    /// Galileo E1.
    E1,
    /// Galileo E5a.
    E5A,
    /// Galileo E5b.
    E5B,
    /// Galileo E5 AltBOC.
    E5,
    /// Galileo E6.
    E6,
    /// BeiDou B1C.
    B1C,
    /// BeiDou B1I.
    B1I,
    /// BeiDou B2a.
    B2A,
    /// BeiDou B2b.
    B2B,
    /// BeiDou B2.
    B2,
    /// BeiDou B3I.
    B3I,
    /// GLONASS G1 FDMA.
    G1,
    /// GLONASS G2 FDMA.
    G2,
}

impl From<PyCarrierBand> for CarrierBand {
    fn from(band: PyCarrierBand) -> Self {
        match band {
            PyCarrierBand::L1 => CarrierBand::L1,
            PyCarrierBand::L2 => CarrierBand::L2,
            PyCarrierBand::L5 => CarrierBand::L5,
            PyCarrierBand::E1 => CarrierBand::E1,
            PyCarrierBand::E5A => CarrierBand::E5a,
            PyCarrierBand::E5B => CarrierBand::E5b,
            PyCarrierBand::E5 => CarrierBand::E5,
            PyCarrierBand::E6 => CarrierBand::E6,
            PyCarrierBand::B1C => CarrierBand::B1c,
            PyCarrierBand::B1I => CarrierBand::B1i,
            PyCarrierBand::B2A => CarrierBand::B2a,
            PyCarrierBand::B2B => CarrierBand::B2b,
            PyCarrierBand::B2 => CarrierBand::B2,
            PyCarrierBand::B3I => CarrierBand::B3i,
            PyCarrierBand::G1 => CarrierBand::G1,
            PyCarrierBand::G2 => CarrierBand::G2,
        }
    }
}

impl From<CarrierBand> for PyCarrierBand {
    fn from(band: CarrierBand) -> Self {
        match band {
            CarrierBand::L1 => PyCarrierBand::L1,
            CarrierBand::L2 => PyCarrierBand::L2,
            CarrierBand::L5 => PyCarrierBand::L5,
            CarrierBand::E1 => PyCarrierBand::E1,
            CarrierBand::E5a => PyCarrierBand::E5A,
            CarrierBand::E5b => PyCarrierBand::E5B,
            CarrierBand::E5 => PyCarrierBand::E5,
            CarrierBand::E6 => PyCarrierBand::E6,
            CarrierBand::B1c => PyCarrierBand::B1C,
            CarrierBand::B1i => PyCarrierBand::B1I,
            CarrierBand::B2a => PyCarrierBand::B2A,
            CarrierBand::B2b => PyCarrierBand::B2B,
            CarrierBand::B2 => PyCarrierBand::B2,
            CarrierBand::B3i => PyCarrierBand::B3I,
            CarrierBand::G1 => PyCarrierBand::G1,
            CarrierBand::G2 => PyCarrierBand::G2,
        }
    }
}

#[pymethods]
impl PyCarrierBand {
    /// Canonical lower-case carrier-band token.
    #[getter]
    fn name(&self) -> &'static str {
        CarrierBand::from(*self).name()
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyCarrierBand::L1 => "CarrierBand.L1",
            PyCarrierBand::L2 => "CarrierBand.L2",
            PyCarrierBand::L5 => "CarrierBand.L5",
            PyCarrierBand::E1 => "CarrierBand.E1",
            PyCarrierBand::E5A => "CarrierBand.E5A",
            PyCarrierBand::E5B => "CarrierBand.E5B",
            PyCarrierBand::E5 => "CarrierBand.E5",
            PyCarrierBand::E6 => "CarrierBand.E6",
            PyCarrierBand::B1C => "CarrierBand.B1C",
            PyCarrierBand::B1I => "CarrierBand.B1I",
            PyCarrierBand::B2A => "CarrierBand.B2A",
            PyCarrierBand::B2B => "CarrierBand.B2B",
            PyCarrierBand::B2 => "CarrierBand.B2",
            PyCarrierBand::B3I => "CarrierBand.B3I",
            PyCarrierBand::G1 => "CarrierBand.G1",
            PyCarrierBand::G2 => "CarrierBand.G2",
        }
    }
}

/// Standard two-carrier ionosphere-free pair.
#[pyclass(module = "sidereon._sidereon", name = "CarrierPair")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyCarrierPair {
    inner: CarrierPair,
}

impl From<CarrierPair> for PyCarrierPair {
    fn from(inner: CarrierPair) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCarrierPair {
    /// Create a two-carrier pair.
    #[new]
    fn new(band1: PyCarrierBand, band2: PyCarrierBand) -> Self {
        Self {
            inner: CarrierPair::new(band1.into(), band2.into()),
        }
    }

    /// First carrier band in the affine combination.
    #[getter]
    fn band1(&self) -> PyCarrierBand {
        self.inner.band1.into()
    }

    /// Second carrier band in the affine combination.
    #[getter]
    fn band2(&self) -> PyCarrierBand {
        self.inner.band2.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "CarrierPair(band1=CarrierBand.{}, band2=CarrierBand.{})",
            band_variant_name(self.inner.band1),
            band_variant_name(self.inner.band2)
        )
    }

    fn __eq__(&self, other: &PyCarrierPair) -> bool {
        self.inner == other.inner
    }
}

/// Reason a satellite was dropped from paired pseudorange combination.
#[pyclass(
    module = "sidereon._sidereon",
    name = "PseudorangeDropReason",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyPseudorangeDropReason {
    /// Present in band 2 only.
    MISSING_BAND1,
    /// Present in band 1 only.
    MISSING_BAND2,
    /// Repeated within at least one input band.
    DUPLICATE_OBSERVATION,
    /// Unsupported constellation or override band.
    UNKNOWN_SYSTEM,
}

impl From<PseudorangeDropReason> for PyPseudorangeDropReason {
    fn from(reason: PseudorangeDropReason) -> Self {
        match reason {
            PseudorangeDropReason::MissingBand1 => Self::MISSING_BAND1,
            PseudorangeDropReason::MissingBand2 => Self::MISSING_BAND2,
            PseudorangeDropReason::DuplicateObservation => Self::DUPLICATE_OBSERVATION,
            PseudorangeDropReason::UnknownSystem => Self::UNKNOWN_SYSTEM,
        }
    }
}

#[pymethods]
impl PyPseudorangeDropReason {
    /// Stable lowercase reason token.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::MISSING_BAND1 => "missing_band1",
            Self::MISSING_BAND2 => "missing_band2",
            Self::DUPLICATE_OBSERVATION => "duplicate_observation",
            Self::UNKNOWN_SYSTEM => "unknown_system",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::MISSING_BAND1 => "PseudorangeDropReason.MISSING_BAND1",
            Self::MISSING_BAND2 => "PseudorangeDropReason.MISSING_BAND2",
            Self::DUPLICATE_OBSERVATION => "PseudorangeDropReason.DUPLICATE_OBSERVATION",
            Self::UNKNOWN_SYSTEM => "PseudorangeDropReason.UNKNOWN_SYSTEM",
        }
    }
}

/// Pseudorange measurement variance model.
#[pyclass(
    module = "sidereon._sidereon",
    name = "PseudorangeVarianceModel",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum PyPseudorangeVarianceModel {
    /// Elevation-only `a^2 + b^2 / sin(el)^2`.
    ELEVATION,
    /// Elevation plus a C/N0 variance contribution.
    ELEVATION_CN0,
}

impl From<PyPseudorangeVarianceModel> for PseudorangeVarianceModel {
    fn from(model: PyPseudorangeVarianceModel) -> Self {
        match model {
            PyPseudorangeVarianceModel::ELEVATION => PseudorangeVarianceModel::Elevation,
            PyPseudorangeVarianceModel::ELEVATION_CN0 => PseudorangeVarianceModel::ElevationCn0,
        }
    }
}

impl From<PseudorangeVarianceModel> for PyPseudorangeVarianceModel {
    fn from(model: PseudorangeVarianceModel) -> Self {
        match model {
            PseudorangeVarianceModel::Elevation => PyPseudorangeVarianceModel::ELEVATION,
            PseudorangeVarianceModel::ElevationCn0 => PyPseudorangeVarianceModel::ELEVATION_CN0,
        }
    }
}

#[pymethods]
impl PyPseudorangeVarianceModel {
    /// Stable lowercase model token.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::ELEVATION => "elevation",
            Self::ELEVATION_CN0 => "elevation_cn0",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::ELEVATION => "PseudorangeVarianceModel.ELEVATION",
            Self::ELEVATION_CN0 => "PseudorangeVarianceModel.ELEVATION_CN0",
        }
    }
}

/// Options for pseudorange variance weighting.
#[pyclass(module = "sidereon._sidereon", name = "PseudorangeVarianceOptions")]
#[derive(Clone, Copy)]
pub struct PyPseudorangeVarianceOptions {
    inner: PseudorangeVarianceOptions,
}

#[pymethods]
impl PyPseudorangeVarianceOptions {
    /// Create pseudorange variance options.
    ///
    /// `a_m` and `b_m` are metres, `cn0_dbhz` is dB-Hz, and `cn0_scale_m2` is
    /// square metres.
    #[new]
    #[pyo3(signature = (
        a_m=0.3,
        b_m=0.3,
        model=PyPseudorangeVarianceModel::ELEVATION,
        cn0_dbhz=None,
        cn0_scale_m2=1.0,
    ))]
    fn new(
        a_m: f64,
        b_m: f64,
        model: PyPseudorangeVarianceModel,
        cn0_dbhz: Option<f64>,
        cn0_scale_m2: f64,
    ) -> Self {
        Self {
            inner: PseudorangeVarianceOptions {
                a_m,
                b_m,
                model: model.into(),
                cn0_dbhz,
                cn0_scale_m2,
            },
        }
    }

    /// Zenith-floor term, metres.
    #[getter]
    fn a_m(&self) -> f64 {
        self.inner.a_m
    }

    /// Elevation-scaled term, metres.
    #[getter]
    fn b_m(&self) -> f64 {
        self.inner.b_m
    }

    /// Selected variance model.
    #[getter]
    fn model(&self) -> PyPseudorangeVarianceModel {
        self.inner.model.into()
    }

    /// Carrier-to-noise density, dB-Hz.
    #[getter]
    fn cn0_dbhz(&self) -> Option<f64> {
        self.inner.cn0_dbhz
    }

    /// C/N0 variance scale, square metres.
    #[getter]
    fn cn0_scale_m2(&self) -> f64 {
        self.inner.cn0_scale_m2
    }

    fn __repr__(&self) -> String {
        format!(
            "PseudorangeVarianceOptions(a_m={}, b_m={}, model={}, cn0_dbhz={:?}, cn0_scale_m2={})",
            self.inner.a_m,
            self.inner.b_m,
            self.model().__repr__(),
            self.inner.cn0_dbhz,
            self.inner.cn0_scale_m2
        )
    }
}

/// One satellite/elevation row for sigma or weight construction.
#[pyclass(module = "sidereon._sidereon", name = "WeightEntry")]
#[derive(Clone)]
pub struct PyWeightEntry {
    inner: WeightEntry,
}

#[pymethods]
impl PyWeightEntry {
    /// Create one weighting row.
    ///
    /// `satellite_id` is a token such as `"G01"`, elevation is degrees, and
    /// `cn0_dbhz` is an optional carrier-to-noise density in dB-Hz.
    #[new]
    #[pyo3(signature = (satellite_id, elevation_deg, cn0_dbhz=None))]
    fn new(satellite_id: String, elevation_deg: f64, cn0_dbhz: Option<f64>) -> Self {
        Self {
            inner: WeightEntry {
                satellite_id,
                elevation_deg,
                cn0_dbhz,
            },
        }
    }

    /// Satellite token, e.g. `"G01"`.
    #[getter]
    fn satellite_id(&self) -> String {
        self.inner.satellite_id.clone()
    }

    /// Topocentric elevation, degrees.
    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.inner.elevation_deg
    }

    /// Optional carrier-to-noise density, dB-Hz.
    #[getter]
    fn cn0_dbhz(&self) -> Option<f64> {
        self.inner.cn0_dbhz
    }

    fn __repr__(&self) -> String {
        format!(
            "WeightEntry(satellite_id={:?}, elevation_deg={}, cn0_dbhz={:?})",
            self.inner.satellite_id, self.inner.elevation_deg, self.inner.cn0_dbhz
        )
    }
}

impl PyWeightEntry {
    fn to_core(&self) -> WeightEntry {
        self.inner.clone()
    }
}

/// RAIM weighting mode.
#[pyclass(module = "sidereon._sidereon", name = "RaimWeights")]
#[derive(Clone)]
pub struct PyRaimWeights {
    inner: RaimWeights,
}

#[pymethods]
impl PyRaimWeights {
    /// Create unit RAIM weights.
    #[new]
    fn new() -> Self {
        Self {
            inner: RaimWeights::Unit,
        }
    }

    /// Unit weights, equivalent to sigma = 1 m for every satellite.
    #[staticmethod]
    fn unit() -> Self {
        Self {
            inner: RaimWeights::Unit,
        }
    }

    /// Per-satellite inverse-variance weights.
    #[staticmethod]
    fn by_satellite(
        satellite_ids: Vec<String>,
        weights: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        let weights = weights
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if satellite_ids.len() != weights.len() {
            return Err(PyValueError::new_err(
                "satellite_ids and weights must have the same length",
            ));
        }

        let mut map = BTreeMap::new();
        for (satellite_id, weight) in satellite_ids.into_iter().zip(weights.iter().copied()) {
            if !weight.is_finite() || weight <= 0.0 {
                return Err(PyValueError::new_err(
                    "RAIM weights must be positive finite values",
                ));
            }
            map.insert(satellite_id, weight);
        }

        Ok(Self {
            inner: RaimWeights::BySatellite(map),
        })
    }

    /// True when all satellites use unit weight.
    #[getter]
    fn is_unit(&self) -> bool {
        matches!(self.inner, RaimWeights::Unit)
    }

    /// Satellite tokens for per-satellite weights, sorted by token.
    #[getter]
    fn satellite_ids(&self) -> Vec<String> {
        match &self.inner {
            RaimWeights::Unit => Vec::new(),
            RaimWeights::BySatellite(weights) => weights.keys().cloned().collect(),
        }
    }

    /// Inverse-variance weights as numpy `(n,)`, sorted by satellite token.
    #[getter]
    fn weights<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        match &self.inner {
            RaimWeights::Unit => PyArray1::from_slice(py, &[]),
            RaimWeights::BySatellite(weights) => {
                let values: Vec<_> = weights.values().copied().collect();
                np_array(py, &values)
            }
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            RaimWeights::Unit => "RaimWeights.unit()".to_string(),
            RaimWeights::BySatellite(weights) => {
                format!("RaimWeights.by_satellite(n={})", weights.len())
            }
        }
    }
}

/// Reason a carrier-phase arc split was flagged.
#[pyclass(module = "sidereon._sidereon", name = "SlipReason", eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum PySlipReason {
    /// Loss-of-lock indicator bit 0 was set on either band.
    LLI,
    /// Gap to the previous usable sample exceeded the configured threshold.
    DATA_GAP,
    /// Geometry-free phase step exceeded the configured threshold.
    GEOMETRY_FREE,
    /// Melbourne-Wubbena step exceeded the configured threshold.
    MELBOURNE_WUBBENA,
}

impl From<SlipReason> for PySlipReason {
    fn from(reason: SlipReason) -> Self {
        match reason {
            SlipReason::Lli => Self::LLI,
            SlipReason::DataGap => Self::DATA_GAP,
            SlipReason::GeometryFree => Self::GEOMETRY_FREE,
            SlipReason::MelbourneWubbena => Self::MELBOURNE_WUBBENA,
        }
    }
}

impl From<PySlipReason> for SlipReason {
    fn from(reason: PySlipReason) -> Self {
        match reason {
            PySlipReason::LLI => SlipReason::Lli,
            PySlipReason::DATA_GAP => SlipReason::DataGap,
            PySlipReason::GEOMETRY_FREE => SlipReason::GeometryFree,
            PySlipReason::MELBOURNE_WUBBENA => SlipReason::MelbourneWubbena,
        }
    }
}

#[pymethods]
impl PySlipReason {
    /// Stable lowercase reason token.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::LLI => "lli",
            Self::DATA_GAP => "data_gap",
            Self::GEOMETRY_FREE => "geometry_free",
            Self::MELBOURNE_WUBBENA => "melbourne_wubbena",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::LLI => "SlipReason.LLI",
            Self::DATA_GAP => "SlipReason.DATA_GAP",
            Self::GEOMETRY_FREE => "SlipReason.GEOMETRY_FREE",
            Self::MELBOURNE_WUBBENA => "SlipReason.MELBOURNE_WUBBENA",
        }
    }
}

/// One epoch in a single-satellite carrier-phase arc.
#[pyclass(module = "sidereon._sidereon", name = "ArcEpoch")]
#[derive(Clone, Copy)]
pub struct PyArcEpoch {
    inner: ArcEpoch,
}

#[pymethods]
impl PyArcEpoch {
    /// Create one carrier-phase arc epoch.
    ///
    /// Phases are cycles, code values are metres, carrier frequencies are hertz,
    /// and `gap_time_s` is any comparable epoch coordinate in seconds.
    #[new]
    #[pyo3(signature = (
        phi1_cycles=None,
        phi2_cycles=None,
        p1_m=None,
        p2_m=None,
        lli1=None,
        lli2=None,
        f1_hz=None,
        f2_hz=None,
        gap_time_s=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        phi1_cycles: Option<f64>,
        phi2_cycles: Option<f64>,
        p1_m: Option<f64>,
        p2_m: Option<f64>,
        lli1: Option<i64>,
        lli2: Option<i64>,
        f1_hz: Option<f64>,
        f2_hz: Option<f64>,
        gap_time_s: Option<f64>,
    ) -> Self {
        Self {
            inner: ArcEpoch {
                phi1_cycles,
                phi2_cycles,
                p1_m,
                p2_m,
                lli1,
                lli2,
                f1_hz,
                f2_hz,
                gap_time_s,
            },
        }
    }

    /// Band-1 carrier phase, cycles.
    #[getter]
    fn phi1_cycles(&self) -> Option<f64> {
        self.inner.phi1_cycles
    }

    /// Band-2 carrier phase, cycles.
    #[getter]
    fn phi2_cycles(&self) -> Option<f64> {
        self.inner.phi2_cycles
    }

    /// Band-1 code pseudorange, metres.
    #[getter]
    fn p1_m(&self) -> Option<f64> {
        self.inner.p1_m
    }

    /// Band-2 code pseudorange, metres.
    #[getter]
    fn p2_m(&self) -> Option<f64> {
        self.inner.p2_m
    }

    /// Band-1 loss-of-lock indicator.
    #[getter]
    fn lli1(&self) -> Option<i64> {
        self.inner.lli1
    }

    /// Band-2 loss-of-lock indicator.
    #[getter]
    fn lli2(&self) -> Option<i64> {
        self.inner.lli2
    }

    /// Band-1 carrier frequency, hertz.
    #[getter]
    fn f1_hz(&self) -> Option<f64> {
        self.inner.f1_hz
    }

    /// Band-2 carrier frequency, hertz.
    #[getter]
    fn f2_hz(&self) -> Option<f64> {
        self.inner.f2_hz
    }

    /// Comparable epoch coordinate in seconds.
    #[getter]
    fn gap_time_s(&self) -> Option<f64> {
        self.inner.gap_time_s
    }

    fn __repr__(&self) -> String {
        format!(
            "ArcEpoch(phi1_cycles={:?}, phi2_cycles={:?}, p1_m={:?}, p2_m={:?}, lli1={:?}, lli2={:?}, f1_hz={:?}, f2_hz={:?}, gap_time_s={:?})",
            self.inner.phi1_cycles,
            self.inner.phi2_cycles,
            self.inner.p1_m,
            self.inner.p2_m,
            self.inner.lli1,
            self.inner.lli2,
            self.inner.f1_hz,
            self.inner.f2_hz,
            self.inner.gap_time_s
        )
    }
}

impl PyArcEpoch {
    #[allow(clippy::wrong_self_convention)]
    fn to_core(&self) -> ArcEpoch {
        self.inner
    }
}

/// Options controlling carrier-phase cycle-slip classification.
#[pyclass(module = "sidereon._sidereon", name = "CycleSlipOptions")]
#[derive(Clone, Copy)]
pub struct PyCycleSlipOptions {
    inner: CycleSlipOptions,
}

#[pymethods]
impl PyCycleSlipOptions {
    /// Create cycle-slip detection options.
    #[new]
    #[pyo3(signature = (
        gf_threshold_m=0.05,
        mw_threshold_cycles=4.0,
        min_arc_gap_s=300.0,
    ))]
    fn new(gf_threshold_m: f64, mw_threshold_cycles: f64, min_arc_gap_s: f64) -> Self {
        Self {
            inner: CycleSlipOptions {
                gf_threshold_m,
                mw_threshold_cycles,
                min_arc_gap_s,
            },
        }
    }

    /// Geometry-free step threshold, metres.
    #[getter]
    fn gf_threshold_m(&self) -> f64 {
        self.inner.gf_threshold_m
    }

    /// Melbourne-Wubbena step threshold, wide-lane cycles.
    #[getter]
    fn mw_threshold_cycles(&self) -> f64 {
        self.inner.mw_threshold_cycles
    }

    /// Data-gap threshold, seconds.
    #[getter]
    fn min_arc_gap_s(&self) -> f64 {
        self.inner.min_arc_gap_s
    }

    fn __repr__(&self) -> String {
        format!(
            "CycleSlipOptions(gf_threshold_m={}, mw_threshold_cycles={}, min_arc_gap_s={})",
            self.inner.gf_threshold_m, self.inner.mw_threshold_cycles, self.inner.min_arc_gap_s
        )
    }
}

/// Cycle-slip classification for one input epoch.
#[pyclass(module = "sidereon._sidereon", name = "SlipResult")]
#[derive(Clone)]
pub struct PySlipResult {
    inner: SlipResult,
}

impl From<SlipResult> for PySlipResult {
    fn from(inner: SlipResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySlipResult {
    /// Whether any slip reason was flagged.
    #[getter]
    fn slip(&self) -> bool {
        self.inner.slip
    }

    /// Slip reasons in deterministic API order.
    #[getter]
    fn reasons(&self) -> Vec<PySlipReason> {
        self.inner.reasons.iter().copied().map(Into::into).collect()
    }

    /// Current geometry-free phase, metres.
    #[getter]
    fn gf_m(&self) -> Option<f64> {
        self.inner.gf_m
    }

    /// Current Melbourne-Wubbena combination, metres.
    #[getter]
    fn mw_m(&self) -> Option<f64> {
        self.inner.mw_m
    }

    /// Whether the epoch was skipped because a frequency was unavailable.
    #[getter]
    fn skipped(&self) -> bool {
        self.inner.skipped
    }

    fn __repr__(&self) -> String {
        format!(
            "SlipResult(slip={}, reasons={:?}, gf_m={:?}, mw_m={:?}, skipped={})",
            self.inner.slip,
            self.reasons(),
            self.inner.gf_m,
            self.inner.mw_m,
            self.inner.skipped
        )
    }
}

/// Hatch-smoothed single-frequency code output for one epoch.
#[pyclass(module = "sidereon._sidereon", name = "SmoothCodeResult")]
#[derive(Clone, Copy)]
pub struct PySmoothCodeResult {
    inner: SmoothCodeResult,
}

impl From<SmoothCodeResult> for PySmoothCodeResult {
    fn from(inner: SmoothCodeResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySmoothCodeResult {
    /// Smoothed code pseudorange, metres.
    #[getter]
    fn p_smooth_m(&self) -> Option<f64> {
        self.inner.p_smooth_m
    }

    /// Hatch window length used at this epoch.
    #[getter]
    fn window(&self) -> usize {
        self.inner.window
    }

    /// True when a prior running window was reset by a slip.
    #[getter]
    fn reset(&self) -> bool {
        self.inner.reset
    }

    fn __repr__(&self) -> String {
        format!(
            "SmoothCodeResult(p_smooth_m={:?}, window={}, reset={})",
            self.inner.p_smooth_m, self.inner.window, self.inner.reset
        )
    }
}

/// Hatch-smoothed ionosphere-free code output for one epoch.
#[pyclass(module = "sidereon._sidereon", name = "IonoFreeSmoothResult")]
#[derive(Clone, Copy)]
pub struct PyIonoFreeSmoothResult {
    inner: IonoFreeSmoothResult,
}

impl From<IonoFreeSmoothResult> for PyIonoFreeSmoothResult {
    fn from(inner: IonoFreeSmoothResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIonoFreeSmoothResult {
    /// Smoothed ionosphere-free code pseudorange, metres.
    #[getter]
    fn p_smooth_m(&self) -> Option<f64> {
        self.inner.p_smooth_m
    }

    /// Instantaneous ionosphere-free code, metres.
    #[getter]
    fn p_if_m(&self) -> Option<f64> {
        self.inner.p_if_m
    }

    /// Instantaneous ionosphere-free carrier phase, metres.
    #[getter]
    fn l_if_m(&self) -> Option<f64> {
        self.inner.l_if_m
    }

    /// Hatch window length used at this epoch.
    #[getter]
    fn window(&self) -> usize {
        self.inner.window
    }

    /// True when a prior running window was reset by a slip.
    #[getter]
    fn reset(&self) -> bool {
        self.inner.reset
    }

    fn __repr__(&self) -> String {
        format!(
            "IonoFreeSmoothResult(p_smooth_m={:?}, p_if_m={:?}, l_if_m={:?}, window={}, reset={})",
            self.inner.p_smooth_m,
            self.inner.p_if_m,
            self.inner.l_if_m,
            self.inner.window,
            self.inner.reset
        )
    }
}

/// Observation value convention for receiver velocity solving.
#[pyclass(module = "sidereon._sidereon", name = "VelocityObservable", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum PyVelocityObservable {
    /// Observation values are pseudorange rates in metres per second.
    RANGE_RATE,
    /// Observation values are Doppler shifts in hertz.
    DOPPLER,
}

impl From<PyVelocityObservable> for VelocityObservable {
    fn from(observable: PyVelocityObservable) -> Self {
        match observable {
            PyVelocityObservable::RANGE_RATE => VelocityObservable::RangeRate,
            PyVelocityObservable::DOPPLER => VelocityObservable::Doppler,
        }
    }
}

impl From<VelocityObservable> for PyVelocityObservable {
    fn from(observable: VelocityObservable) -> Self {
        match observable {
            VelocityObservable::RangeRate => PyVelocityObservable::RANGE_RATE,
            VelocityObservable::Doppler => PyVelocityObservable::DOPPLER,
        }
    }
}

#[pymethods]
impl PyVelocityObservable {
    /// Stable lowercase observation convention.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::RANGE_RATE => "range_rate",
            Self::DOPPLER => "doppler",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::RANGE_RATE => "VelocityObservable.RANGE_RATE",
            Self::DOPPLER => "VelocityObservable.DOPPLER",
        }
    }
}

/// One satellite observation for a receiver velocity solve.
#[pyclass(module = "sidereon._sidereon", name = "VelocityObservation")]
#[derive(Clone, Copy)]
pub struct PyVelocityObservation {
    inner: VelocityObservation,
}

#[pymethods]
impl PyVelocityObservation {
    /// Create one velocity observation.
    ///
    /// `value` is pseudorange rate in m/s for `RANGE_RATE` solves or Doppler in
    /// Hz for `DOPPLER` solves. `carrier_hz` is used only for Doppler conversion.
    #[new]
    #[pyo3(signature = (satellite_id, value, carrier_hz, sat_clock_drift_s_s=0.0))]
    fn new(
        satellite_id: &str,
        value: f64,
        carrier_hz: f64,
        sat_clock_drift_s_s: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: VelocityObservation {
                satellite_id: parse_sat(satellite_id)?,
                value,
                carrier_hz,
                sat_clock_drift_s_s,
            },
        })
    }

    /// Satellite token, e.g. `"G07"`.
    #[getter]
    fn satellite_id(&self) -> String {
        self.inner.satellite_id.to_string()
    }

    /// Pseudorange rate in m/s or Doppler in Hz, depending on solve options.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    /// Carrier frequency in hertz, used for Doppler observations.
    #[getter]
    fn carrier_hz(&self) -> f64 {
        self.inner.carrier_hz
    }

    /// Satellite clock drift in seconds per second.
    #[getter]
    fn sat_clock_drift_s_s(&self) -> f64 {
        self.inner.sat_clock_drift_s_s
    }

    fn __repr__(&self) -> String {
        format!(
            "VelocityObservation(satellite_id={:?}, value={}, carrier_hz={}, sat_clock_drift_s_s={})",
            self.inner.satellite_id.to_string(),
            self.inner.value,
            self.inner.carrier_hz,
            self.inner.sat_clock_drift_s_s
        )
    }
}

impl PyVelocityObservation {
    #[allow(clippy::wrong_self_convention)]
    fn to_core(&self) -> VelocityObservation {
        self.inner
    }
}

/// Options controlling receiver velocity solving.
#[pyclass(module = "sidereon._sidereon", name = "VelocitySolveOptions")]
#[derive(Clone, Copy)]
pub struct PyVelocitySolveOptions {
    inner: VelocitySolveOptions,
}

#[pymethods]
impl PyVelocitySolveOptions {
    /// Create velocity solve options.
    #[new]
    #[pyo3(signature = (
        observable=PyVelocityObservable::RANGE_RATE,
        light_time=true,
        sagnac=true,
    ))]
    fn new(observable: PyVelocityObservable, light_time: bool, sagnac: bool) -> Self {
        Self {
            inner: VelocitySolveOptions {
                observable: observable.into(),
                light_time,
                sagnac,
            },
        }
    }

    /// Observation value convention.
    #[getter]
    fn observable(&self) -> PyVelocityObservable {
        self.inner.observable.into()
    }

    /// Apply fixed-point light-time correction in geometry prediction.
    #[getter]
    fn light_time(&self) -> bool {
        self.inner.light_time
    }

    /// Apply Earth-rotation Sagnac correction in geometry prediction.
    #[getter]
    fn sagnac(&self) -> bool {
        self.inner.sagnac
    }

    fn __repr__(&self) -> String {
        format!(
            "VelocitySolveOptions(observable={}, light_time={}, sagnac={})",
            self.observable().__repr__(),
            self.inner.light_time,
            self.inner.sagnac
        )
    }
}

/// Receiver velocity solve result.
#[pyclass(module = "sidereon._sidereon", name = "VelocitySolution")]
pub struct PyVelocitySolution {
    inner: VelocitySolution,
}

#[pymethods]
impl PyVelocitySolution {
    /// Receiver ECEF velocity as numpy `(3,)`, metres per second.
    #[getter]
    fn velocity_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_m_s)
    }

    /// Receiver speed in metres per second.
    #[getter]
    fn speed_m_s(&self) -> f64 {
        self.inner.speed_m_s
    }

    /// Receiver clock drift in seconds per second.
    #[getter]
    fn clock_drift_s_s(&self) -> f64 {
        self.inner.clock_drift_s_s
    }

    /// Satellite tokens contributing rows, in residual order.
    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner
            .used_sats
            .iter()
            .map(ToString::to_string)
            .collect()
    }

    /// Post-fit range-rate residuals as numpy `(n,)`, metres per second.
    #[getter]
    fn residuals_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let residuals: Vec<_> = self
            .inner
            .residuals_m_s
            .iter()
            .map(|(_, residual)| *residual)
            .collect();
        np_array(py, &residuals)
    }

    fn __repr__(&self) -> String {
        format!(
            "VelocitySolution(velocity_m_s=[{:.3}, {:.3}, {:.3}], clock_drift_s_s={:.6e}, used_sats={})",
            self.inner.velocity_m_s[0],
            self.inner.velocity_m_s[1],
            self.inner.velocity_m_s[2],
            self.inner.clock_drift_s_s,
            self.inner.used_sats.len()
        )
    }
}

/// GPS C/A replica generation options.
#[pyclass(module = "sidereon._sidereon", name = "ReplicaOptions")]
#[derive(Clone, Copy)]
pub struct PyReplicaOptions {
    inner: ReplicaOptions,
}

#[pymethods]
impl PyReplicaOptions {
    /// Create sampled GPS C/A replica options.
    ///
    /// `sample_rate_hz` and `code_doppler_hz` are hertz, `num_samples` is a
    /// count, and `code_phase_chips` is in C/A chips.
    #[new]
    #[pyo3(signature = (
        sample_rate_hz=2_046_000.0,
        num_samples=2046,
        code_phase_chips=0.0,
        code_doppler_hz=0.0,
    ))]
    fn new(
        sample_rate_hz: f64,
        num_samples: usize,
        code_phase_chips: f64,
        code_doppler_hz: f64,
    ) -> Self {
        Self {
            inner: ReplicaOptions {
                sample_rate_hz,
                num_samples,
                code_phase_chips,
                code_doppler_hz,
            },
        }
    }

    /// One C/A code period at 2.046 MHz.
    #[staticmethod]
    fn one_code_period() -> Self {
        Self {
            inner: ReplicaOptions::one_code_period(),
        }
    }

    /// Sampling rate, hertz.
    #[getter]
    fn sample_rate_hz(&self) -> f64 {
        self.inner.sample_rate_hz
    }

    /// Output sample count.
    #[getter]
    fn num_samples(&self) -> usize {
        self.inner.num_samples
    }

    /// Initial C/A code phase, chips.
    #[getter]
    fn code_phase_chips(&self) -> f64 {
        self.inner.code_phase_chips
    }

    /// Code-rate Doppler, hertz.
    #[getter]
    fn code_doppler_hz(&self) -> f64 {
        self.inner.code_doppler_hz
    }

    fn __repr__(&self) -> String {
        format!(
            "ReplicaOptions(sample_rate_hz={}, num_samples={}, code_phase_chips={}, code_doppler_hz={})",
            self.inner.sample_rate_hz,
            self.inner.num_samples,
            self.inner.code_phase_chips,
            self.inner.code_doppler_hz
        )
    }
}

/// Coherent GPS C/A correlation options.
#[pyclass(module = "sidereon._sidereon", name = "CorrelateOptions")]
#[derive(Clone, Copy)]
pub struct PyCorrelateOptions {
    inner: CorrelateOptions,
}

#[pymethods]
impl PyCorrelateOptions {
    /// Create coherent correlator options.
    ///
    /// `sample_rate_hz`, `doppler_hz`, and `code_doppler_hz` are hertz.
    /// `code_phase_chips` is in C/A chips.
    #[new]
    #[pyo3(signature = (
        sample_rate_hz=2_046_000.0,
        doppler_hz=0.0,
        code_phase_chips=0.0,
        code_doppler_hz=0.0,
    ))]
    fn new(
        sample_rate_hz: f64,
        doppler_hz: f64,
        code_phase_chips: f64,
        code_doppler_hz: f64,
    ) -> Self {
        Self {
            inner: CorrelateOptions {
                sample_rate_hz,
                doppler_hz,
                code_phase_chips,
                code_doppler_hz,
            },
        }
    }

    /// Sampling rate, hertz.
    #[getter]
    fn sample_rate_hz(&self) -> f64 {
        self.inner.sample_rate_hz
    }

    /// Residual carrier Doppler wiped off, hertz.
    #[getter]
    fn doppler_hz(&self) -> f64 {
        self.inner.doppler_hz
    }

    /// Replica C/A code phase, chips.
    #[getter]
    fn code_phase_chips(&self) -> f64 {
        self.inner.code_phase_chips
    }

    /// Replica code-rate Doppler, hertz.
    #[getter]
    fn code_doppler_hz(&self) -> f64 {
        self.inner.code_doppler_hz
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrelateOptions(sample_rate_hz={}, doppler_hz={}, code_phase_chips={}, code_doppler_hz={})",
            self.inner.sample_rate_hz,
            self.inner.doppler_hz,
            self.inner.code_phase_chips,
            self.inner.code_doppler_hz
        )
    }
}

/// Coherent GPS C/A correlation result.
#[pyclass(module = "sidereon._sidereon", name = "CorrelationResult")]
#[derive(Clone, Copy)]
pub struct PyCorrelationResult {
    inner: CorrelationResult,
}

impl From<CorrelationResult> for PyCorrelationResult {
    fn from(inner: CorrelationResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCorrelationResult {
    /// In-phase coherent sum.
    #[getter]
    fn i(&self) -> f64 {
        self.inner.i
    }

    /// Quadrature coherent sum.
    #[getter]
    fn q(&self) -> f64 {
        self.inner.q
    }

    /// Squared magnitude, `i*i + q*q`.
    #[getter]
    fn power(&self) -> f64 {
        self.inner.power
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrelationResult(i={}, q={}, power={})",
            self.inner.i, self.inner.q, self.inner.power
        )
    }
}

/// GPS C/A acquisition search options.
#[pyclass(module = "sidereon._sidereon", name = "AcquisitionOptions")]
#[derive(Clone, Copy)]
pub struct PyAcquisitionOptions {
    inner: AcquisitionOptions,
}

#[pymethods]
impl PyAcquisitionOptions {
    /// Create direct C/A acquisition search options.
    ///
    /// All rates and Doppler bins are in hertz.
    #[new]
    #[pyo3(signature = (
        sample_rate_hz=2_046_000.0,
        doppler_min_hz=-2500.0,
        doppler_max_hz=2500.0,
        doppler_step_hz=500.0,
    ))]
    fn new(
        sample_rate_hz: f64,
        doppler_min_hz: f64,
        doppler_max_hz: f64,
        doppler_step_hz: f64,
    ) -> Self {
        Self {
            inner: AcquisitionOptions {
                sample_rate_hz,
                doppler_min_hz,
                doppler_max_hz,
                doppler_step_hz,
            },
        }
    }

    /// Sampling rate, hertz.
    #[getter]
    fn sample_rate_hz(&self) -> f64 {
        self.inner.sample_rate_hz
    }

    /// Minimum Doppler bin, hertz.
    #[getter]
    fn doppler_min_hz(&self) -> f64 {
        self.inner.doppler_min_hz
    }

    /// Maximum Doppler bin, hertz.
    #[getter]
    fn doppler_max_hz(&self) -> f64 {
        self.inner.doppler_max_hz
    }

    /// Doppler bin step, hertz.
    #[getter]
    fn doppler_step_hz(&self) -> f64 {
        self.inner.doppler_step_hz
    }

    fn __repr__(&self) -> String {
        format!(
            "AcquisitionOptions(sample_rate_hz={}, doppler_min_hz={}, doppler_max_hz={}, doppler_step_hz={})",
            self.inner.sample_rate_hz,
            self.inner.doppler_min_hz,
            self.inner.doppler_max_hz,
            self.inner.doppler_step_hz
        )
    }
}

/// Acquisition grid metadata.
#[pyclass(module = "sidereon._sidereon", name = "AcquisitionGrid")]
#[derive(Clone)]
pub struct PyAcquisitionGrid {
    inner: AcquisitionGrid,
}

impl From<AcquisitionGrid> for PyAcquisitionGrid {
    fn from(inner: AcquisitionGrid) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAcquisitionGrid {
    /// Doppler bins searched as numpy `(n,)`, hertz.
    #[getter]
    fn doppler_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_hz)
    }

    /// Number of code-phase bins searched.
    #[getter]
    fn code_phase_bins(&self) -> usize {
        self.inner.code_phase_bins
    }

    /// Doppler step, hertz.
    #[getter]
    fn doppler_step_hz(&self) -> f64 {
        self.inner.doppler_step_hz
    }

    /// Samples per C/A chip.
    #[getter]
    fn samples_per_chip(&self) -> f64 {
        self.inner.samples_per_chip
    }

    fn __repr__(&self) -> String {
        format!(
            "AcquisitionGrid(doppler_bins={}, code_phase_bins={}, doppler_step_hz={}, samples_per_chip={})",
            self.inner.doppler_hz.len(),
            self.inner.code_phase_bins,
            self.inner.doppler_step_hz,
            self.inner.samples_per_chip
        )
    }
}

/// Result of a 2D C/A code-phase and Doppler acquisition search.
#[pyclass(module = "sidereon._sidereon", name = "AcquisitionResult")]
#[derive(Clone)]
pub struct PyAcquisitionResult {
    inner: AcquisitionResult,
}

impl From<AcquisitionResult> for PyAcquisitionResult {
    fn from(inner: AcquisitionResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAcquisitionResult {
    /// Recovered C/A code phase, chips.
    #[getter]
    fn code_phase_chips(&self) -> f64 {
        self.inner.code_phase_chips
    }

    /// Recovered Doppler bin, hertz.
    #[getter]
    fn doppler_hz(&self) -> f64 {
        self.inner.doppler_hz
    }

    /// Peak-to-mean-off-peak acquisition metric.
    #[getter]
    fn peak_metric(&self) -> f64 {
        self.inner.peak_metric
    }

    /// Alias of `peak_metric`.
    #[getter]
    fn metric(&self) -> f64 {
        self.inner.metric
    }

    /// Peak correlator power.
    #[getter]
    fn peak_power(&self) -> f64 {
        self.inner.peak_power
    }

    /// Search-grid metadata.
    #[getter]
    fn grid(&self) -> PyAcquisitionGrid {
        self.inner.grid.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "AcquisitionResult(code_phase_chips={}, doppler_hz={}, peak_metric={}, peak_power={})",
            self.inner.code_phase_chips,
            self.inner.doppler_hz,
            self.inner.peak_metric,
            self.inner.peak_power
        )
    }
}

/// Carrier frequency in hertz for a constellation and canonical carrier band.
#[pyfunction]
fn carrier_frequency_hz(system: PyGnssSystem, band: PyCarrierBand) -> Option<f64> {
    frequency_hz(system.into(), band.into())
}

/// Carrier wavelength in metres for a constellation and canonical carrier band.
#[pyfunction(name = "wavelength_m")]
fn wavelength_m_py(system: PyGnssSystem, band: PyCarrierBand) -> Option<f64> {
    wavelength_m(system.into(), band.into())
}

/// RINEX observation band frequency in hertz for a system and band digit.
#[pyfunction(name = "rinex_band_frequency_hz")]
#[pyo3(signature = (system, band, glonass_channel=None))]
fn rinex_band_frequency_hz_py(
    system: PyGnssSystem,
    band: &str,
    glonass_channel: Option<i8>,
) -> PyResult<Option<f64>> {
    Ok(rinex_band_frequency_hz(
        system.into(),
        rinex_band_char(band)?,
        glonass_channel,
    ))
}

/// RINEX observation band wavelength in metres for a system and band digit.
#[pyfunction(name = "rinex_band_wavelength_m")]
#[pyo3(signature = (system, band, glonass_channel=None))]
fn rinex_band_wavelength_m_py(
    system: PyGnssSystem,
    band: &str,
    glonass_channel: Option<i8>,
) -> PyResult<Option<f64>> {
    Ok(rinex_band_wavelength_m(
        system.into(),
        rinex_band_char(band)?,
        glonass_channel,
    ))
}

#[pyfunction]
#[pyo3(signature = (system, code, rinex_version, glonass_channel=None))]
fn rinex_observation_frequency_hz(
    system: PyGnssSystem,
    code: &str,
    rinex_version: f64,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    core_rinex_observation_frequency_hz(system.into(), code, rinex_version, glonass_channel)
}

#[pyfunction]
#[pyo3(signature = (system, code, rinex_version, glonass_channel=None))]
fn rinex_observation_wavelength_m(
    system: PyGnssSystem,
    code: &str,
    rinex_version: f64,
    glonass_channel: Option<i8>,
) -> Option<f64> {
    core_rinex_observation_wavelength_m(system.into(), code, rinex_version, glonass_channel)
}

/// Standard dual-frequency ionosphere-free carrier pair for a constellation.
#[pyfunction]
fn default_pair(system: PyGnssSystem) -> Option<PyCarrierPair> {
    default_iono_free_pair(system.into()).map(Into::into)
}

/// Ionosphere-free coefficient `gamma = f1^2 / (f1^2 - f2^2)`.
#[pyfunction]
fn gamma(f1_hz: f64, f2_hz: f64) -> PyResult<f64> {
    combinations::gamma(f1_hz, f2_hz).map_err(ionosphere_free_error)
}

/// Equal-variance noise amplification of the ionosphere-free combination.
#[pyfunction]
fn noise_amplification(f1_hz: f64, f2_hz: f64) -> PyResult<f64> {
    combinations::noise_amplification(f1_hz, f2_hz).map_err(ionosphere_free_error)
}

/// Ionosphere-free code or meter-valued phase combination, metres.
#[pyfunction]
fn ionosphere_free(obs1_m: f64, obs2_m: f64, f1_hz: f64, f2_hz: f64) -> PyResult<f64> {
    combinations::ionosphere_free(obs1_m, obs2_m, f1_hz, f2_hz).map_err(ionosphere_free_error)
}

/// Ionosphere-free carrier-phase combination from meter-valued phase inputs.
#[pyfunction(name = "ionosphere_free_phase_m")]
fn ionosphere_free_phase_m_py(
    phase1_m: f64,
    phase2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> PyResult<f64> {
    combinations::ionosphere_free_phase_m(phase1_m, phase2_m, f1_hz, f2_hz)
        .map_err(ionosphere_free_error)
}

/// Ionosphere-free carrier-phase combination from cycle-valued phase inputs.
#[pyfunction]
fn ionosphere_free_phase_cycles(
    phi1_cycles: f64,
    phi2_cycles: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> PyResult<f64> {
    combinations::ionosphere_free_phase_cycles(phi1_cycles, phi2_cycles, f1_hz, f2_hz)
        .map_err(ionosphere_free_error)
}

/// Combine two satellite-keyed pseudorange bands into ionosphere-free ranges.
#[pyfunction(name = "ionosphere_free_pseudoranges")]
#[pyo3(signature = (band1, band2, overrides=None))]
fn ionosphere_free_pseudoranges_py(
    band1: Vec<PyPseudorangeObservation>,
    band2: Vec<PyPseudorangeObservation>,
    overrides: Option<Vec<(String, String, String)>>,
) -> PyResult<PyPseudorangeCombinationResult> {
    let overrides = overrides
        .unwrap_or_default()
        .into_iter()
        .map(|(system, band1, band2)| Ok((system_char(&system)?, band1, band2)))
        .collect::<PyResult<Vec<_>>>()?;
    let (combined, dropped) =
        combinations::ionosphere_free_pseudoranges(&band1, &band2, &overrides)
            .map_err(ionosphere_free_error)?;
    let dropped: Vec<(String, PyPseudorangeDropReason)> = dropped
        .into_iter()
        .map(|(sat, reason)| (sat, reason.into()))
        .collect();
    Ok((combined, dropped))
}

/// Pseudorange measurement variance, square metres.
#[pyfunction]
#[pyo3(signature = (elevation_deg, options=None))]
fn pseudorange_variance(
    py: Python<'_>,
    elevation_deg: f64,
    options: Option<Py<PyPseudorangeVarianceOptions>>,
) -> PyResult<f64> {
    let options = pseudorange_variance_options(py, options.as_ref());
    quality::pseudorange_variance(elevation_deg, options).map_err(quality_error)
}

/// Build satellite-keyed pseudorange sigmas in metres.
///
/// Returns `(satellite_ids, sigmas)`; invalid entries are dropped by the core API.
#[pyfunction]
#[pyo3(signature = (entries, options=None))]
fn sigmas<'py>(
    py: Python<'py>,
    entries: Vec<Py<PyWeightEntry>>,
    options: Option<Py<PyPseudorangeVarianceOptions>>,
) -> (Vec<String>, Bound<'py, PyArray1<f64>>) {
    let entries = weight_entries_to_core(py, &entries);
    let options = pseudorange_variance_options(py, options.as_ref());
    map_to_vector(py, quality::sigmas(&entries, options))
}

/// Build satellite-keyed inverse-variance pseudorange weights.
///
/// Returns `(satellite_ids, weights)`; invalid entries are dropped by the core API.
#[pyfunction]
#[pyo3(signature = (entries, options=None))]
fn weight_vector<'py>(
    py: Python<'py>,
    entries: Vec<Py<PyWeightEntry>>,
    options: Option<Py<PyPseudorangeVarianceOptions>>,
) -> (Vec<String>, Bound<'py, PyArray1<f64>>) {
    let entries = weight_entries_to_core(py, &entries);
    let options = pseudorange_variance_options(py, options.as_ref());
    map_to_vector(py, quality::weight_vector(&entries, options))
}

/// Carrier phase converted to metres, `L = c / f * phi`.
#[pyfunction]
fn phase_meters(phi_cycles: f64, f_hz: f64) -> PyResult<f64> {
    carrier_phase::phase_meters(phi_cycles, f_hz).map_err(carrier_phase_error)
}

#[pyfunction]
fn code_minus_carrier(p_m: f64, phi_cycles: f64, f_hz: f64) -> PyResult<f64> {
    carrier_phase::code_minus_carrier(p_m, phi_cycles, f_hz).map_err(carrier_phase_error)
}

/// Geometry-free phase combination `L_GF = L1 - L2`, metres.
#[pyfunction]
fn geometry_free(l1_m: f64, l2_m: f64) -> PyResult<f64> {
    carrier_phase::geometry_free(l1_m, l2_m).map_err(carrier_phase_error)
}

/// Wide-lane wavelength `c / (f1 - f2)`, metres.
#[pyfunction]
fn wide_lane_wavelength(f1_hz: f64, f2_hz: f64) -> PyResult<f64> {
    carrier_phase::wide_lane_wavelength(f1_hz, f2_hz).map_err(carrier_phase_error)
}

/// Narrow-lane code combination, metres.
#[pyfunction]
fn narrow_lane_code(p1_m: f64, p2_m: f64, f1_hz: f64, f2_hz: f64) -> PyResult<f64> {
    carrier_phase::narrow_lane_code(p1_m, p2_m, f1_hz, f2_hz).map_err(carrier_phase_error)
}

/// Melbourne-Wubbena combination, metres.
#[pyfunction]
fn melbourne_wubbena(
    phi1_cycles: f64,
    phi2_cycles: f64,
    p1_m: f64,
    p2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> PyResult<f64> {
    carrier_phase::melbourne_wubbena(phi1_cycles, phi2_cycles, p1_m, p2_m, f1_hz, f2_hz)
        .map_err(carrier_phase_error)
}

/// Melbourne-Wubbena wide-lane ambiguity estimate, wide-lane cycles.
#[pyfunction]
fn wide_lane_cycles(
    phi1_cycles: f64,
    phi2_cycles: f64,
    p1_m: f64,
    p2_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> PyResult<f64> {
    carrier_phase::wide_lane_cycles(phi1_cycles, phi2_cycles, p1_m, p2_m, f1_hz, f2_hz)
        .map_err(carrier_phase_error)
}

/// Detect cycle slips on a time-ordered single-satellite carrier-phase arc.
#[pyfunction]
#[pyo3(signature = (arc, options=None))]
fn detect_cycle_slips(
    py: Python<'_>,
    arc: Vec<Py<PyArcEpoch>>,
    options: Option<Py<PyCycleSlipOptions>>,
) -> PyResult<Vec<PySlipResult>> {
    let arc = arc_to_core(py, &arc);
    let options = cycle_slip_options(py, options.as_ref());
    Ok(carrier_phase::detect_cycle_slips(&arc, options)
        .map_err(carrier_phase_error)?
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Single-frequency Hatch carrier-smoothed code on band 1.
#[pyfunction]
#[pyo3(signature = (arc, options=None, hatch_window_cap=DEFAULT_HATCH_WINDOW_CAP))]
fn smooth_code(
    py: Python<'_>,
    arc: Vec<Py<PyArcEpoch>>,
    options: Option<Py<PyCycleSlipOptions>>,
    hatch_window_cap: usize,
) -> PyResult<Vec<PySmoothCodeResult>> {
    let arc = arc_to_core(py, &arc);
    let options = cycle_slip_options(py, options.as_ref());
    Ok(carrier_phase::smooth_code(&arc, options, hatch_window_cap)
        .map_err(carrier_phase_error)?
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Dual-frequency ionosphere-free Hatch carrier-smoothed code.
#[pyfunction]
#[pyo3(signature = (arc, options=None, hatch_window_cap=DEFAULT_HATCH_WINDOW_CAP))]
fn smooth_iono_free_code(
    py: Python<'_>,
    arc: Vec<Py<PyArcEpoch>>,
    options: Option<Py<PyCycleSlipOptions>>,
    hatch_window_cap: usize,
) -> PyResult<Vec<PyIonoFreeSmoothResult>> {
    let arc = arc_to_core(py, &arc);
    let options = cycle_slip_options(py, options.as_ref());
    Ok(
        carrier_phase::smooth_iono_free_code(&arc, options, hatch_window_cap)
            .map_err(carrier_phase_error)?
            .into_iter()
            .map(Into::into)
            .collect(),
    )
}

/// Convert a Doppler shift in hertz to pseudorange rate in metres per second.
#[pyfunction]
fn doppler_to_range_rate(doppler_hz: f64, carrier_hz: f64) -> PyResult<f64> {
    velocity::doppler_to_range_rate(doppler_hz, carrier_hz).map_err(to_solve_err)
}

/// Convert a pseudorange rate in metres per second to Doppler shift in hertz.
#[pyfunction]
fn range_rate_to_doppler(range_rate_m_s: f64, carrier_hz: f64) -> PyResult<f64> {
    velocity::range_rate_to_doppler(range_rate_m_s, carrier_hz).map_err(to_solve_err)
}

/// Solve receiver ECEF velocity and clock drift from one epoch of observations.
#[pyfunction]
#[pyo3(signature = (sp3, observations, receiver_ecef_m, t_rx_j2000_s, options=None))]
fn solve_velocity(
    py: Python<'_>,
    sp3: &PySp3,
    observations: Vec<Py<PyVelocityObservation>>,
    receiver_ecef_m: PyReadonlyArray1<'_, f64>,
    t_rx_j2000_s: f64,
    options: Option<Py<PyVelocitySolveOptions>>,
) -> PyResult<PyVelocitySolution> {
    let observations: Vec<_> = observations
        .iter()
        .map(|obs| obs.borrow(py).to_core())
        .collect();
    let receiver_ecef_m = fixed_array::<3>(
        "receiver_ecef_m",
        &receiver_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |options| options.inner,
        VelocitySolveOptions::default,
    );
    let inner = velocity::solve(
        &sp3.inner,
        &observations,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
    )
    .map_err(to_solve_err)?;
    Ok(PyVelocitySolution { inner })
}

/// Solve receiver ECEF velocity and clock drift from a broadcast NAV source.
///
/// Identical to [`solve_velocity`] but resolves satellite states from a parsed
/// broadcast ephemeris store rather than an SP3 precise product.
#[pyfunction]
#[pyo3(signature = (broadcast, observations, receiver_ecef_m, t_rx_j2000_s, options=None))]
fn solve_velocity_broadcast(
    py: Python<'_>,
    broadcast: &PyBroadcastEphemeris,
    observations: Vec<Py<PyVelocityObservation>>,
    receiver_ecef_m: PyReadonlyArray1<'_, f64>,
    t_rx_j2000_s: f64,
    options: Option<Py<PyVelocitySolveOptions>>,
) -> PyResult<PyVelocitySolution> {
    let observations: Vec<_> = observations
        .iter()
        .map(|obs| obs.borrow(py).to_core())
        .collect();
    let receiver_ecef_m = fixed_array::<3>(
        "receiver_ecef_m",
        &receiver_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |options| options.inner,
        VelocitySolveOptions::default,
    );
    let inner = velocity::solve(
        &broadcast.inner,
        &observations,
        receiver_ecef_m,
        t_rx_j2000_s,
        options,
    )
    .map_err(to_solve_err)?;
    Ok(PyVelocitySolution { inner })
}

/// Predicted single-satellite geometric observables at a receive epoch.
///
/// Range, range-rate, Doppler, satellite clock, look angles, transmit time, and
/// the line-of-sight and satellite state, all produced by the core predictor.
#[pyclass(module = "sidereon._sidereon", name = "PredictedObservables")]
#[derive(Clone)]
pub struct PyPredictedObservables {
    inner: PredictedObservables,
}

#[pymethods]
impl PyPredictedObservables {
    /// Geometric range after optional Sagnac rotation, meters.
    #[getter]
    fn geometric_range_m(&self) -> f64 {
        self.inner.geometric_range_m
    }

    /// Range-rate LOS projection, meters per second.
    #[getter]
    fn range_rate_m_s(&self) -> f64 {
        self.inner.range_rate_m_s
    }

    /// Doppler shift at the requested carrier, hertz.
    #[getter]
    fn doppler_hz(&self) -> f64 {
        self.inner.doppler_hz
    }

    /// Satellite clock offset at transmit time, seconds (or `None`).
    #[getter]
    fn sat_clock_s(&self) -> Option<f64> {
        self.inner.sat_clock_s
    }

    /// Topocentric elevation, degrees.
    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.inner.elevation_deg
    }

    /// Topocentric azimuth in `[0, 360)`, degrees.
    #[getter]
    fn azimuth_deg(&self) -> f64 {
        self.inner.azimuth_deg
    }

    /// Transmit-time offset from receive time, microseconds.
    #[getter]
    fn transmit_offset_us(&self) -> i64 {
        self.inner.transmit_offset_us
    }

    /// Transmit time as seconds since J2000.
    #[getter]
    fn transmit_time_j2000_s(&self) -> f64 {
        self.inner.transmit_time_j2000_s
    }

    /// Receiver-to-satellite line-of-sight unit vector in ECEF, `(3,)`.
    #[getter]
    fn los_unit<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.los_unit)
    }

    /// Sagnac-rotated satellite ECEF position, meters, `(3,)`.
    #[getter]
    fn sat_pos_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.sat_pos_ecef_m)
    }

    /// Sagnac-rotated satellite ECEF velocity, meters per second, `(3,)`.
    #[getter]
    fn sat_velocity_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.sat_velocity_m_s)
    }

    fn __repr__(&self) -> String {
        format!(
            "PredictedObservables(geometric_range_m={:.3}, elevation_deg={:.3}, \
             azimuth_deg={:.3}, doppler_hz={:.3})",
            self.inner.geometric_range_m,
            self.inner.elevation_deg,
            self.inner.azimuth_deg,
            self.inner.doppler_hz
        )
    }
}

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite token {token:?}: {err}")))
}

/// Map a core observable failure to a Python exception, splitting a malformed
/// input ([`ObservablesError::InvalidInput`] -> `ValueError`) from a no-data gap
/// or structured ephemeris/solve failure ([`ObservablesError::NoEphemeris`] /
/// [`ObservablesError::Ephemeris`] -> `SolveError`). The expected no-ephemeris
/// gap is folded to a `None` batch entry at the call site before this runs, so on
/// the batch path it only ever classifies a genuine failure.
fn observables_error(err: ObservablesError) -> PyErr {
    match err {
        ObservablesError::InvalidInput { .. } => PyValueError::new_err(err.to_string()),
        ObservablesError::NoEphemeris | ObservablesError::Ephemeris(_) => to_solve_err(err),
    }
}

/// Predict geometric observables for one satellite from an SP3 precise product.
#[pyfunction]
#[pyo3(signature = (sp3, satellite_id, receiver_ecef_m, t_rx_j2000_s, carrier_hz, light_time=false, sagnac=true))]
fn observe(
    sp3: &PySp3,
    satellite_id: &str,
    receiver_ecef_m: PyReadonlyArray1<'_, f64>,
    t_rx_j2000_s: f64,
    carrier_hz: f64,
    light_time: bool,
    sagnac: bool,
) -> PyResult<PyPredictedObservables> {
    let sat = parse_satellite(satellite_id)?;
    let receiver_ecef_m = fixed_array::<3>(
        "receiver_ecef_m",
        &receiver_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let inner = predict(
        &sp3.inner,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        PredictOptions {
            carrier_hz,
            light_time,
            sagnac,
        },
    )
    .map_err(observables_error)?;
    Ok(PyPredictedObservables { inner })
}

/// Predict geometric observables for one satellite from a broadcast NAV source.
#[pyfunction]
#[pyo3(signature = (broadcast, satellite_id, receiver_ecef_m, t_rx_j2000_s, carrier_hz, light_time=false, sagnac=true))]
fn observe_broadcast(
    broadcast: &PyBroadcastEphemeris,
    satellite_id: &str,
    receiver_ecef_m: PyReadonlyArray1<'_, f64>,
    t_rx_j2000_s: f64,
    carrier_hz: f64,
    light_time: bool,
    sagnac: bool,
) -> PyResult<PyPredictedObservables> {
    let sat = parse_satellite(satellite_id)?;
    let receiver_ecef_m = fixed_array::<3>(
        "receiver_ecef_m",
        &receiver_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let inner = predict(
        &broadcast.inner,
        sat,
        receiver_ecef_m,
        t_rx_j2000_s,
        PredictOptions {
            carrier_hz,
            light_time,
            sagnac,
        },
    )
    .map_err(observables_error)?;
    Ok(PyPredictedObservables { inner })
}

/// One batch range-prediction request: the satellite, the static receiver ECEF
/// position in metres, and the receive epoch in seconds since J2000.
#[pyclass(module = "sidereon._sidereon", name = "RangePredictionRequest")]
#[derive(Clone, Copy)]
pub struct PyRangePredictionRequest {
    inner: RangePredictionRequest,
}

#[pymethods]
impl PyRangePredictionRequest {
    /// Create one range-prediction request.
    ///
    /// `satellite` is a canonical token such as `"G01"`, `receiver_ecef_m` is a
    /// length-3 static ECEF position in metres, and `t_rx_j2000_s` is the receive
    /// epoch in seconds since J2000.
    #[new]
    fn new(satellite: &str, receiver_ecef_m: [f64; 3], t_rx_j2000_s: f64) -> PyResult<Self> {
        Ok(Self {
            inner: RangePredictionRequest {
                sat: parse_satellite(satellite)?,
                receiver_ecef_m,
                t_rx_j2000_s,
            },
        })
    }

    /// Satellite token, e.g. `"G01"`.
    #[getter]
    fn satellite(&self) -> String {
        self.inner.sat.to_string()
    }

    /// Static receiver ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn receiver_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.receiver_ecef_m)
    }

    /// Receive epoch, seconds since J2000.
    #[getter]
    fn t_rx_j2000_s(&self) -> f64 {
        self.inner.t_rx_j2000_s
    }

    fn __repr__(&self) -> String {
        format!(
            "RangePredictionRequest(satellite={:?}, t_rx_j2000_s={})",
            self.inner.sat.to_string(),
            self.inner.t_rx_j2000_s
        )
    }
}

/// The geometry-only result of one [`predict_ranges`] request.
#[pyclass(module = "sidereon._sidereon", name = "RangePrediction")]
#[derive(Clone, Copy)]
pub struct PyRangePrediction {
    inner: RangePrediction,
}

impl From<RangePrediction> for PyRangePrediction {
    fn from(inner: RangePrediction) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRangePrediction {
    /// Geometric range after optional Sagnac transport, metres.
    #[getter]
    fn geometric_range_m(&self) -> f64 {
        self.inner.geometric_range_m
    }

    /// Satellite clock offset at transmit time, seconds (or `None`).
    #[getter]
    fn sat_clock_s(&self) -> Option<f64> {
        self.inner.sat_clock_s
    }

    /// Transmit time as seconds since J2000.
    #[getter]
    fn transmit_time_j2000_s(&self) -> f64 {
        self.inner.transmit_time_j2000_s
    }

    /// Sagnac-transported satellite ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn sat_pos_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.sat_pos_ecef_m)
    }

    fn __repr__(&self) -> String {
        format!(
            "RangePrediction(geometric_range_m={:.3}, transmit_time_j2000_s={}, sat_clock_s={:?})",
            self.inner.geometric_range_m, self.inner.transmit_time_j2000_s, self.inner.sat_clock_s
        )
    }
}

/// An owned precise-ephemeris source for [`predict_ranges_py`], decoded under
/// the GIL so the batch compute can run with the GIL released without holding a
/// Python borrow across the `allow_threads` boundary.
enum OwnedRangeSource {
    // `Sp3` is far larger than `Samples`; box it so the enum is not sized to the
    // largest variant (clippy::large_enum_variant).
    Sp3(Box<Sp3>),
    Samples(PreciseEphemerisSamples),
}

impl OwnedRangeSource {
    fn as_source(&self) -> &dyn ObservableEphemerisSource {
        match self {
            OwnedRangeSource::Sp3(sp3) => sp3.as_ref(),
            OwnedRangeSource::Samples(samples) => samples,
        }
    }
}

/// Predict geometric ranges for many `(satellite, receiver, epoch)` requests in
/// one call.
///
/// `source` is an [`Sp3`](crate) precise product or a
/// [`PreciseEphemerisSamples`](crate) source. `requests` is a list of
/// [`RangePredictionRequest`]; the whole batch is evaluated inside one call so no
/// per-request Python dispatch is paid. Returns a list of [`RangePrediction`]
/// whose entry `i` corresponds to request `i`, using the same light-time
/// iteration and Sagnac transport as the scalar predictor.
///
/// Raises `ValueError` on a malformed request input and `SolveError` on a
/// structured ephemeris failure (a missing or out-of-coverage satellite/epoch),
/// which aborts the batch on the first failing request.
#[pyfunction(name = "predict_ranges")]
#[pyo3(signature = (source, requests, *, light_time=true, sagnac=true))]
fn predict_ranges_py(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    requests: Vec<Py<PyRangePredictionRequest>>,
    light_time: bool,
    sagnac: bool,
) -> PyResult<Vec<PyRangePrediction>> {
    let requests: Vec<RangePredictionRequest> =
        requests.iter().map(|r| r.borrow(py).inner).collect();
    let options = PredictOptions {
        light_time,
        sagnac,
        ..PredictOptions::default()
    };
    let mut out = vec![
        RangePrediction {
            geometric_range_m: 0.0,
            sat_clock_s: None,
            transmit_time_j2000_s: 0.0,
            sat_pos_ecef_m: [0.0; 3],
        };
        requests.len()
    ];

    // Decode and own the selected source under the GIL: the borrow from the
    // PyRef must not cross the `allow_threads` boundary, so clone the underlying
    // core source into an owned value the compute can hold with the GIL
    // released. The source-polymorphism (Sp3 | PreciseEphemerisSamples,
    // anything else -> TypeError) is unchanged.
    let source: OwnedRangeSource = if let Ok(sp3) = source.downcast::<PySp3>() {
        OwnedRangeSource::Sp3(Box::new(sp3.borrow().inner.clone()))
    } else if let Ok(samples) = source.downcast::<PyPreciseEphemerisSamples>() {
        OwnedRangeSource::Samples(samples.borrow().inner.clone())
    } else {
        return Err(PyTypeError::new_err(
            "source must be an Sp3 or a PreciseEphemerisSamples",
        ));
    };

    // The source, requests, and output buffer are all fully owned, so the
    // pure-Rust batch runs with the GIL released and reacquires it only to build
    // the Python result list. Releasing the GIL does not change results.
    py.allow_threads(|| core_predict_ranges(source.as_source(), &requests, options, &mut out))
        .map_err(observables_error)?;

    Ok(out.into_iter().map(PyRangePrediction::from).collect())
}

/// Predict geometric observables for many `(satellite, receiver, epoch)` requests
/// from an SP3 precise product, in one call.
///
/// `satellite_ids` is a length-`n` list of canonical tokens, `receivers_ecef_m`
/// an `(n, 3)` numpy array of static receiver ECEF positions in metres, and
/// `t_rx_j2000_s` a length-`n` numpy array of receive epochs in seconds since
/// J2000. Each request is independent, so one batch can mix satellites,
/// receivers, and epochs freely. Returns a length-`n` list whose entry `i` is the
/// [`PredictedObservables`] for request `i`, or `None` only when that request has
/// no ephemeris at the epoch (the expected no-data gap). An invalid input raises
/// `ValueError` and a structured ephemeris/solve failure raises `SolveError`
/// rather than being masked as a missing entry. With `parallel=True` the
/// independent requests fan across a thread pool; each value is bit-identical to
/// the serial path.
#[pyfunction]
#[pyo3(signature = (sp3, satellite_ids, receivers_ecef_m, t_rx_j2000_s, carrier_hz, *, light_time=false, sagnac=true, parallel=true))]
#[allow(clippy::too_many_arguments)]
fn observe_batch(
    sp3: &PySp3,
    satellite_ids: Vec<String>,
    receivers_ecef_m: PyReadonlyArray2<'_, f64>,
    t_rx_j2000_s: PyReadonlyArray1<'_, f64>,
    carrier_hz: f64,
    light_time: bool,
    sagnac: bool,
    parallel: bool,
) -> PyResult<Vec<Option<PyPredictedObservables>>> {
    let receivers = rows3_from_array(
        "receivers_ecef_m",
        &receivers_ecef_m,
        EmptyPolicy::Allow,
        FinitePolicy::RequireFinite,
    )?;
    let epochs = t_rx_j2000_s
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if satellite_ids.len() != receivers.len() || satellite_ids.len() != epochs.len() {
        return Err(PyValueError::new_err(format!(
            "satellite_ids ({}), receivers_ecef_m ({}), and t_rx_j2000_s ({}) must have the same length",
            satellite_ids.len(),
            receivers.len(),
            epochs.len()
        )));
    }
    let mut requests: Vec<PredictRequest> = Vec::with_capacity(satellite_ids.len());
    for (index, token) in satellite_ids.iter().enumerate() {
        requests.push((parse_satellite(token)?, receivers[index], epochs[index]));
    }
    let options = PredictOptions {
        carrier_hz,
        light_time,
        sagnac,
    };
    let results = if parallel {
        predict_batch_parallel(&sp3.inner, &requests, options)
    } else {
        predict_batch(&sp3.inner, &requests, options)
    };
    results
        .into_iter()
        .map(|result| match result {
            Ok(inner) => Ok(Some(PyPredictedObservables { inner })),
            // The expected no-data gap is the only failure that maps to a missing
            // entry; an invalid input or a structured ephemeris/solve failure
            // surfaces as a Python exception rather than a silent `None`.
            Err(ObservablesError::NoEphemeris) => Ok(None),
            Err(err) => Err(observables_error(err)),
        })
        .collect()
}

/// GPS C/A code chips for a PRN as numpy `int8` `(1023,)`, with chips `+1` or `-1`.
#[pyfunction(name = "ca_code")]
fn ca_code_py<'py>(py: Python<'py>, prn: i64) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let chips = signal::ca_code(prn).map_err(signal_error)?;
    Ok(PyArray1::from_vec(py, chips))
}

/// One wrapping GPS C/A chip at a zero-based index.
#[pyfunction(name = "ca_chip")]
fn ca_chip_py(prn: i64, index: i64) -> PyResult<i8> {
    signal::ca_chip(prn, index).map_err(signal_error)
}

#[pyfunction]
fn autocorrelation<'py>(py: Python<'py>, code: Vec<i8>) -> Bound<'py, PyArray1<i32>> {
    PyArray1::from_vec(py, signal::autocorrelation(&code))
}

#[pyfunction]
fn cross_correlation<'py>(
    py: Python<'py>,
    code_a: Vec<i8>,
    code_b: Vec<i8>,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    signal::cross_correlation(&code_a, &code_b)
        .map(|values| PyArray1::from_vec(py, values))
        .map_err(signal_error)
}

#[pyfunction]
fn correlation_at(code_a: Vec<i8>, code_b: Vec<i8>, lag: i64) -> PyResult<i32> {
    signal::correlation_at(&code_a, &code_b, lag).map_err(signal_error)
}

#[pyfunction]
fn correlate_against(
    iq: PyReadonlyArray2<'_, f64>,
    code: Vec<i8>,
    sample_rate_hz: f64,
    doppler_hz: f64,
) -> PyResult<PyCorrelationResult> {
    let iq = iq_samples_from_array("iq", &iq)?;
    let (i, q) =
        signal::correlate_against(&iq, &code, sample_rate_hz, doppler_hz).map_err(signal_error)?;
    Ok(CorrelationResult {
        i,
        q,
        power: i * i + q * q,
    }
    .into())
}

/// Build a sampled GPS C/A code replica as numpy `int8` `(n,)`.
#[pyfunction(name = "replica")]
#[pyo3(signature = (prn, options=None))]
fn replica_py<'py>(
    py: Python<'py>,
    prn: i64,
    options: Option<Py<PyReplicaOptions>>,
) -> PyResult<Bound<'py, PyArray1<i8>>> {
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |options| options.inner,
        ReplicaOptions::one_code_period,
    );
    let samples = signal::replica(prn, options).map_err(signal_error)?;
    Ok(PyArray1::from_vec(py, samples))
}

/// Coherently correlate IQ samples against a GPS C/A PRN replica.
///
/// `iq` must have shape `(n, 2)` with in-phase and quadrature columns.
#[pyfunction(name = "correlate")]
#[pyo3(signature = (iq, prn, options=None))]
fn correlate_py(
    py: Python<'_>,
    iq: PyReadonlyArray2<'_, f64>,
    prn: i64,
    options: Option<Py<PyCorrelateOptions>>,
) -> PyResult<PyCorrelationResult> {
    let iq = iq_samples_from_array("iq", &iq)?;
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |options| options.inner,
        CorrelateOptions::default,
    );
    signal::correlate(&iq, prn, options)
        .map(Into::into)
        .map_err(signal_error)
}

/// Acquire a GPS C/A PRN by direct code-phase and Doppler search.
///
/// `samples` must have shape `(n, 2)` with in-phase and quadrature columns.
#[pyfunction(name = "acquire")]
#[pyo3(signature = (samples, prn, options=None))]
fn acquire_py(
    py: Python<'_>,
    samples: PyReadonlyArray2<'_, f64>,
    prn: i64,
    options: Option<Py<PyAcquisitionOptions>>,
) -> PyResult<PyAcquisitionResult> {
    let samples = iq_samples_from_array("samples", &samples)?;
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |options| options.inner,
        AcquisitionOptions::default,
    );
    signal::acquire(&samples, prn, options)
        .map(Into::into)
        .map_err(signal_error)
}

/// Coherent integration loss from residual frequency error.
#[pyfunction]
fn coherent_loss(freq_error_hz: f64, integration_time_s: f64) -> PyResult<f64> {
    signal::coherent_loss(freq_error_hz, integration_time_s).map_err(signal_error)
}

/// Coherent integration loss in decibels.
#[pyfunction]
fn coherent_loss_db(freq_error_hz: f64, integration_time_s: f64) -> PyResult<f64> {
    signal::coherent_loss_db(freq_error_hz, integration_time_s).map_err(signal_error)
}

/// Post-correlation predetection SNR in decibels.
#[pyfunction]
fn snr_post_db(cn0_dbhz: f64, integration_time_s: f64) -> PyResult<f64> {
    signal::snr_post_db(cn0_dbhz, integration_time_s).map_err(signal_error)
}

fn rinex_band_char(value: &str) -> PyResult<char> {
    one_char(
        value,
        "band must be a single RINEX observation band character",
    )
}

fn system_char(value: &str) -> PyResult<char> {
    one_char(
        value,
        "override system must be a single RINEX system character",
    )
}

fn one_char(value: &str, message: &'static str) -> PyResult<char> {
    let mut chars = value.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => Ok(ch),
        _ => Err(PyValueError::new_err(message)),
    }
}

fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|e| PyValueError::new_err(format!("invalid satellite token {token:?}: {e}")))
}

fn arc_to_core(py: Python<'_>, arc: &[Py<PyArcEpoch>]) -> Vec<ArcEpoch> {
    arc.iter().map(|epoch| epoch.borrow(py).to_core()).collect()
}

fn cycle_slip_options(
    py: Python<'_>,
    options: Option<&Py<PyCycleSlipOptions>>,
) -> CycleSlipOptions {
    option_py_or_default(
        py,
        options,
        |options| options.inner,
        CycleSlipOptions::default,
    )
}

fn pseudorange_variance_options(
    py: Python<'_>,
    options: Option<&Py<PyPseudorangeVarianceOptions>>,
) -> PseudorangeVarianceOptions {
    option_py_or_default(
        py,
        options,
        |options| options.inner,
        PseudorangeVarianceOptions::default,
    )
}

fn weight_entries_to_core(py: Python<'_>, entries: &[Py<PyWeightEntry>]) -> Vec<WeightEntry> {
    entries
        .iter()
        .map(|entry| entry.borrow(py).to_core())
        .collect()
}

fn map_to_vector<'py>(
    py: Python<'py>,
    map: BTreeMap<String, f64>,
) -> (Vec<String>, Bound<'py, PyArray1<f64>>) {
    let (satellite_ids, values): (Vec<_>, Vec<_>) = map.into_iter().unzip();
    (satellite_ids, np_array(py, &values))
}

fn iq_samples_from_array(
    name: &str,
    samples: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<IqSample>> {
    let view = samples.as_array();
    if view.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 2), got (_, {})",
            view.ncols()
        )));
    }
    Ok(view
        .outer_iter()
        .map(|row| IqSample::new(row[0], row[1]))
        .collect())
}

fn band_variant_name(band: CarrierBand) -> &'static str {
    band.as_str()
}

fn ionosphere_free_error(err: IonosphereFreeError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn carrier_phase_error(err: CarrierPhaseError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn quality_error(err: QualityError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn signal_error(err: SignalError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCarrierBand>()?;
    m.add_class::<PyCarrierPair>()?;
    m.add_class::<PyPseudorangeDropReason>()?;
    m.add_class::<PyPseudorangeVarianceModel>()?;
    m.add_class::<PyPseudorangeVarianceOptions>()?;
    m.add_class::<PyWeightEntry>()?;
    m.add_class::<PyRaimWeights>()?;
    m.add_class::<PySlipReason>()?;
    m.add_class::<PyArcEpoch>()?;
    m.add_class::<PyCycleSlipOptions>()?;
    m.add_class::<PySlipResult>()?;
    m.add_class::<PySmoothCodeResult>()?;
    m.add_class::<PyIonoFreeSmoothResult>()?;
    m.add_class::<PyVelocityObservable>()?;
    m.add_class::<PyVelocityObservation>()?;
    m.add_class::<PyVelocitySolveOptions>()?;
    m.add_class::<PyVelocitySolution>()?;
    m.add_class::<PyReplicaOptions>()?;
    m.add_class::<PyCorrelateOptions>()?;
    m.add_class::<PyCorrelationResult>()?;
    m.add_class::<PyAcquisitionOptions>()?;
    m.add_class::<PyAcquisitionGrid>()?;
    m.add_class::<PyAcquisitionResult>()?;
    m.add_function(wrap_pyfunction!(carrier_frequency_hz, m)?)?;
    m.add_function(wrap_pyfunction!(wavelength_m_py, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_band_frequency_hz_py, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_band_wavelength_m_py, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_observation_frequency_hz, m)?)?;
    m.add_function(wrap_pyfunction!(rinex_observation_wavelength_m, m)?)?;
    m.add_function(wrap_pyfunction!(default_pair, m)?)?;
    m.add_function(wrap_pyfunction!(gamma, m)?)?;
    m.add_function(wrap_pyfunction!(noise_amplification, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_free, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_free_phase_m_py, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_free_phase_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_free_pseudoranges_py, m)?)?;
    m.add_function(wrap_pyfunction!(pseudorange_variance, m)?)?;
    m.add_function(wrap_pyfunction!(sigmas, m)?)?;
    m.add_function(wrap_pyfunction!(weight_vector, m)?)?;
    m.add_function(wrap_pyfunction!(phase_meters, m)?)?;
    m.add_function(wrap_pyfunction!(code_minus_carrier, m)?)?;
    m.add_function(wrap_pyfunction!(geometry_free, m)?)?;
    m.add_function(wrap_pyfunction!(wide_lane_wavelength, m)?)?;
    m.add_function(wrap_pyfunction!(narrow_lane_code, m)?)?;
    m.add_function(wrap_pyfunction!(melbourne_wubbena, m)?)?;
    m.add_function(wrap_pyfunction!(wide_lane_cycles, m)?)?;
    m.add_function(wrap_pyfunction!(detect_cycle_slips, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_code, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_iono_free_code, m)?)?;
    m.add_function(wrap_pyfunction!(doppler_to_range_rate, m)?)?;
    m.add_function(wrap_pyfunction!(range_rate_to_doppler, m)?)?;
    m.add_function(wrap_pyfunction!(solve_velocity, m)?)?;
    m.add_function(wrap_pyfunction!(solve_velocity_broadcast, m)?)?;
    m.add_class::<PyPredictedObservables>()?;
    m.add_function(wrap_pyfunction!(observe, m)?)?;
    m.add_function(wrap_pyfunction!(observe_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(observe_batch, m)?)?;
    m.add_class::<PyRangePredictionRequest>()?;
    m.add_class::<PyRangePrediction>()?;
    m.add_function(wrap_pyfunction!(predict_ranges_py, m)?)?;
    m.add_function(wrap_pyfunction!(ca_code_py, m)?)?;
    m.add_function(wrap_pyfunction!(ca_chip_py, m)?)?;
    m.add_function(wrap_pyfunction!(autocorrelation, m)?)?;
    m.add_function(wrap_pyfunction!(cross_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(correlation_at, m)?)?;
    m.add_function(wrap_pyfunction!(correlate_against, m)?)?;
    m.add_function(wrap_pyfunction!(replica_py, m)?)?;
    m.add_function(wrap_pyfunction!(correlate_py, m)?)?;
    m.add_function(wrap_pyfunction!(acquire_py, m)?)?;
    m.add_function(wrap_pyfunction!(coherent_loss, m)?)?;
    m.add_function(wrap_pyfunction!(coherent_loss_db, m)?)?;
    m.add_function(wrap_pyfunction!(snr_post_db, m)?)?;
    Ok(())
}
