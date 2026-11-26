//! Prediction-horizon scoring against precise orbit and clock products.

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::time::model::Instant;
use sidereon_core::ephemeris::{Sp3FrameLabelSet, Sp3FrameReconciliationOptions};
use sidereon_core::frame_catalog::FrameCatalogError;
use sidereon_core::orbit_scoring::{self as scoring, OrbitSource};
use sidereon_core::GnssSatelliteId;

use crate::ephemeris::{parse_sat, parse_satellites, PySp3};
use crate::frame_catalog::PyTerrestrialFrame;
use crate::frames::PyTimeScale;
use crate::marshal::PyGnssSystem;
use crate::rinex::PyBroadcastEphemeris;
use crate::rinex_clock::PyClockInstant;
use crate::{binding_error, binding_error_detail, SolveError};

create_exception!(
    _sidereon,
    OrbitScoreError,
    SolveError,
    "An orbit scoring input, coverage, or evaluation failure."
);

pub(crate) fn orbit_score_error(error: scoring::OrbitScoreError) -> PyErr {
    let exception = OrbitScoreError::new_err(error.to_string());
    match error {
        scoring::OrbitScoreError::InvalidInput { field, reason } => {
            binding_error(exception, "invalid_input", |detail| {
                detail.set_item("field", field)?;
                detail.set_item("reason", reason)
            })
        }
        scoring::OrbitScoreError::TimeSystemMismatch {
            orbit_source,
            reference,
        } => binding_error(exception, "time_system_mismatch", |detail| {
            detail.set_item("orbit_source", PyTimeScale::from(orbit_source))?;
            detail.set_item("reference", PyTimeScale::from(reference))
        }),
        scoring::OrbitScoreError::TimeSystemLabelMismatch {
            orbit_source,
            reference,
        } => binding_error(exception, "time_system_label_mismatch", |detail| {
            detail.set_item("orbit_source", orbit_source.label())?;
            detail.set_item("reference", reference.label())
        }),
        scoring::OrbitScoreError::InstantTimeSystemMismatch {
            field,
            index,
            found,
            expected,
        } => binding_error(exception, "instant_time_system_mismatch", |detail| {
            detail.set_item("field", field)?;
            detail.set_item("index", index)?;
            detail.set_item("found", PyTimeScale::from(found))?;
            detail.set_item("expected", PyTimeScale::from(expected))
        }),
        scoring::OrbitScoreError::FrameMismatch {
            orbit_source,
            reference,
        } => binding_error(exception, "frame_mismatch", |detail| {
            detail.set_item("orbit_source", orbit_source.to_string())?;
            detail.set_item("reference", reference.to_string())
        }),
        scoring::OrbitScoreError::MissingSatellite { product, satellite } => {
            binding_error(exception, "missing_satellite", |detail| {
                detail.set_item("product", PyScoredProduct::from(product))?;
                detail.set_item("satellite", satellite.to_string())
            })
        }
        scoring::OrbitScoreError::MissingEpoch {
            product,
            satellite,
            epoch_index,
            error,
        } => binding_error(exception, "missing_epoch", |detail| {
            detail.set_item("product", PyScoredProduct::from(product))?;
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("epoch_index", epoch_index)?;
            detail.set_item(
                "error",
                crate::core_error_detail::core_error_detail(detail.py(), &error)?,
            )
        }),
        scoring::OrbitScoreError::NoPredictionOrigin { satellite } => {
            binding_error(exception, "no_prediction_origin", |detail| {
                detail.set_item("satellite", satellite.to_string())
            })
        }
        scoring::OrbitScoreError::NoSisreWeights { satellite, regime } => {
            binding_error(exception, "no_sisre_weights", |detail| {
                detail.set_item("satellite", satellite.to_string())?;
                detail.set_item("regime", PyOrbitRegime::from(regime))
            })
        }
        scoring::OrbitScoreError::UndeterminedOrbitRegime { satellite } => {
            binding_error(exception, "undetermined_orbit_regime", |detail| {
                detail.set_item("satellite", satellite.to_string())
            })
        }
        scoring::OrbitScoreError::NonFiniteSourceState {
            satellite,
            epoch_index,
        } => binding_error(exception, "non_finite_source_state", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("epoch_index", epoch_index)
        }),
        scoring::OrbitScoreError::DegenerateReferenceState {
            satellite,
            epoch_index,
        } => binding_error(exception, "degenerate_reference_state", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("epoch_index", epoch_index)
        }),
        scoring::OrbitScoreError::NonFiniteResult { satellite } => {
            binding_error(exception, "non_finite_result", |detail| {
                detail.set_item("satellite", satellite.to_string())
            })
        }
        scoring::OrbitScoreError::FrameTransform(error) => {
            binding_error(exception, "frame_transform", |detail| {
                let py = detail.py();
                let source = match error {
                    FrameCatalogError::InvalidInput { field, reason } => {
                        binding_error_detail(py, "invalid_input", |source| {
                            source.set_item("field", field)?;
                            source.set_item("reason", reason)
                        })
                    }
                    FrameCatalogError::NoCatalogPath { from, to } => {
                        binding_error_detail(py, "no_catalog_path", |source| {
                            source.set_item("from", PyTerrestrialFrame::from(from))?;
                            source.set_item("to", PyTerrestrialFrame::from(to))
                        })
                    }
                    FrameCatalogError::SingularTransform {
                        from,
                        to,
                        epoch_year,
                    } => binding_error_detail(py, "singular_transform", |source| {
                        source.set_item("from", PyTerrestrialFrame::from(from))?;
                        source.set_item("to", PyTerrestrialFrame::from(to))?;
                        source.set_item("epoch_year", epoch_year)
                    }),
                }?;
                detail.set_item("error", source)
            })
        }
        _ => exception,
    }
}

/// Reference velocity used to build the orbital frame.
#[pyclass(module = "sidereon._sidereon", name = "TriadVelocity", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyTriadVelocity {
    /// Use the reference orbit's inertial velocity.
    INERTIAL,
    /// Use the reference orbit's Earth-fixed velocity.
    EARTH_FIXED,
}

impl From<PyTriadVelocity> for scoring::TriadVelocity {
    fn from(value: PyTriadVelocity) -> Self {
        match value {
            PyTriadVelocity::INERTIAL => Self::Inertial,
            PyTriadVelocity::EARTH_FIXED => Self::EarthFixed,
        }
    }
}

impl From<scoring::TriadVelocity> for PyTriadVelocity {
    fn from(value: scoring::TriadVelocity) -> Self {
        match value {
            scoring::TriadVelocity::Inertial => Self::INERTIAL,
            scoring::TriadVelocity::EarthFixed => Self::EARTH_FIXED,
        }
    }
}

#[pymethods]
impl PyTriadVelocity {
    /// Core default.
    #[staticmethod]
    fn default() -> Self {
        scoring::TriadVelocity::default().into()
    }
}

/// Clock comparison datum.
#[pyclass(module = "sidereon._sidereon", name = "ClockDatum", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyClockDatum {
    /// Remove the per-epoch median clock difference for each constellation.
    EPOCH_SYSTEM_MEDIAN,
    /// Use the original clock differences.
    UNADJUSTED,
}

impl From<PyClockDatum> for scoring::ClockDatum {
    fn from(value: PyClockDatum) -> Self {
        match value {
            PyClockDatum::EPOCH_SYSTEM_MEDIAN => Self::EpochSystemMedian,
            PyClockDatum::UNADJUSTED => Self::Unadjusted,
        }
    }
}

impl From<scoring::ClockDatum> for PyClockDatum {
    fn from(value: scoring::ClockDatum) -> Self {
        match value {
            scoring::ClockDatum::EpochSystemMedian => Self::EPOCH_SYSTEM_MEDIAN,
            scoring::ClockDatum::Unadjusted => Self::UNADJUSTED,
        }
    }
}

#[pymethods]
impl PyClockDatum {
    /// Core default.
    #[staticmethod]
    fn default() -> Self {
        scoring::ClockDatum::default().into()
    }
}

/// Treatment of missing orbit samples.
#[pyclass(
    module = "sidereon._sidereon",
    name = "OrbitScoreGapPolicy",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyOrbitScoreGapPolicy {
    /// Count gaps in the coverage report.
    COUNT,
    /// Return an error at the first missing satellite, epoch, or prediction origin.
    REFUSE,
}

impl From<PyOrbitScoreGapPolicy> for scoring::GapPolicy {
    fn from(value: PyOrbitScoreGapPolicy) -> Self {
        match value {
            PyOrbitScoreGapPolicy::COUNT => Self::Count,
            PyOrbitScoreGapPolicy::REFUSE => Self::Refuse,
        }
    }
}

impl From<scoring::GapPolicy> for PyOrbitScoreGapPolicy {
    fn from(value: scoring::GapPolicy) -> Self {
        match value {
            scoring::GapPolicy::Count => Self::COUNT,
            scoring::GapPolicy::Refuse => Self::REFUSE,
        }
    }
}

#[pymethods]
impl PyOrbitScoreGapPolicy {
    /// Core default.
    #[staticmethod]
    fn default() -> Self {
        scoring::GapPolicy::default().into()
    }
}

/// Satellite point represented by a position.
#[pyclass(
    module = "sidereon._sidereon",
    name = "OrbitScoreReferencePoint",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyOrbitScoreReferencePoint {
    /// The centre of mass, as in an SP3 product.
    CENTER_OF_MASS,
    /// The antenna phase centre, as in a broadcast ephemeris.
    ANTENNA_PHASE_CENTER,
}

impl From<PyOrbitScoreReferencePoint> for scoring::OrbitReferencePoint {
    fn from(value: PyOrbitScoreReferencePoint) -> Self {
        match value {
            PyOrbitScoreReferencePoint::CENTER_OF_MASS => Self::CenterOfMass,
            PyOrbitScoreReferencePoint::ANTENNA_PHASE_CENTER => Self::AntennaPhaseCenter,
        }
    }
}

impl From<scoring::OrbitReferencePoint> for PyOrbitScoreReferencePoint {
    fn from(value: scoring::OrbitReferencePoint) -> Self {
        match value {
            scoring::OrbitReferencePoint::CenterOfMass => Self::CENTER_OF_MASS,
            scoring::OrbitReferencePoint::AntennaPhaseCenter => Self::ANTENNA_PHASE_CENTER,
        }
    }
}

/// Navigation satellite orbit regime.
#[pyclass(module = "sidereon._sidereon", name = "OrbitRegime", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyOrbitRegime {
    /// Medium Earth orbit.
    MEO,
    /// Inclined geosynchronous orbit.
    IGSO,
    /// Geostationary orbit.
    GEO,
}

impl From<PyOrbitRegime> for scoring::OrbitRegime {
    fn from(value: PyOrbitRegime) -> Self {
        match value {
            PyOrbitRegime::MEO => Self::Meo,
            PyOrbitRegime::IGSO => Self::Igso,
            PyOrbitRegime::GEO => Self::Geo,
        }
    }
}

impl From<scoring::OrbitRegime> for PyOrbitRegime {
    fn from(value: scoring::OrbitRegime) -> Self {
        match value {
            scoring::OrbitRegime::Meo => Self::MEO,
            scoring::OrbitRegime::Igso => Self::IGSO,
            scoring::OrbitRegime::Geo => Self::GEO,
        }
    }
}

#[pymethods]
impl PyOrbitRegime {
    /// Core display name.
    fn __str__(&self) -> String {
        scoring::OrbitRegime::from(*self).to_string()
    }

    /// Classify an inertial state by navigation orbit regime.
    #[staticmethod]
    fn from_inertial_state(position_m: [f64; 3], inertial_velocity_m_s: [f64; 3]) -> Option<Self> {
        scoring::OrbitRegime::from_inertial_state(position_m, inertial_velocity_m_s).map(Into::into)
    }
}

/// Source or reference product in an orbit comparison.
#[pyclass(module = "sidereon._sidereon", name = "ScoredProduct", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyScoredProduct {
    /// The orbit source being scored.
    SOURCE,
    /// The reference SP3 product.
    REFERENCE,
}

impl From<PyScoredProduct> for scoring::ScoredProduct {
    fn from(value: PyScoredProduct) -> Self {
        match value {
            PyScoredProduct::SOURCE => Self::Source,
            PyScoredProduct::REFERENCE => Self::Reference,
        }
    }
}

impl From<scoring::ScoredProduct> for PyScoredProduct {
    fn from(value: scoring::ScoredProduct) -> Self {
        match value {
            scoring::ScoredProduct::Source => Self::SOURCE,
            scoring::ScoredProduct::Reference => Self::REFERENCE,
        }
    }
}

#[pymethods]
impl PyScoredProduct {
    /// Core display name.
    fn __str__(&self) -> String {
        scoring::ScoredProduct::from(*self).to_string()
    }
}

/// Position difference in the reference radial, transverse and normal frame.
#[pyclass(module = "sidereon._sidereon", name = "RtnComponents", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyRtnComponents(pub(crate) scoring::RtnComponents);

#[pymethods]
impl PyRtnComponents {
    /// Build an orbit-frame position difference in metres.
    #[new]
    fn new(radial_m: f64, transverse_m: f64, normal_m: f64) -> Self {
        Self(scoring::RtnComponents {
            radial_m,
            transverse_m,
            normal_m,
        })
    }

    /// Radial difference in metres.
    #[getter]
    fn radial_m(&self) -> f64 {
        self.0.radial_m
    }

    /// Along-track difference in metres.
    #[getter]
    fn transverse_m(&self) -> f64 {
        self.0.transverse_m
    }

    /// Cross-track difference in metres.
    #[getter]
    fn normal_m(&self) -> f64 {
        self.0.normal_m
    }
}

/// Earth-fixed orbit source state.
#[pyclass(module = "sidereon._sidereon", name = "OrbitSourceState", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitSourceState(scoring::OrbitSourceState);

#[pymethods]
impl PyOrbitSourceState {
    /// Earth-fixed position in metres.
    #[getter]
    fn position_ecef_m(&self) -> [f64; 3] {
        self.0.position_ecef_m
    }

    /// Clock offset in seconds, when present.
    #[getter]
    fn clock_s(&self) -> Option<f64> {
        self.0.clock_s
    }
}

/// Statistics of a signed error series.
#[pyclass(module = "sidereon._sidereon", name = "ErrorStats", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyErrorStats(scoring::ErrorStats);

#[pymethods]
impl PyErrorStats {
    /// Root mean square error in metres.
    #[getter]
    fn rms_m(&self) -> f64 {
        self.0.rms_m
    }

    /// Mean error in metres.
    #[getter]
    fn mean_m(&self) -> f64 {
        self.0.mean_m
    }

    /// Largest absolute error in metres.
    #[getter]
    fn max_abs_m(&self) -> f64 {
        self.0.max_abs_m
    }
}

/// Orbit and clock error statistics in one prediction-horizon bin.
#[pyclass(module = "sidereon._sidereon", name = "HorizonBinStats", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyHorizonBinStats(scoring::HorizonBinStats);

#[pymethods]
impl PyHorizonBinStats {
    /// Exclusive lower horizon in seconds.
    #[getter]
    fn horizon_start_s(&self) -> f64 {
        self.0.horizon_start_s
    }

    /// Inclusive upper horizon in seconds.
    #[getter]
    fn horizon_end_s(&self) -> f64 {
        self.0.horizon_end_s
    }

    /// Number of scored epochs.
    #[getter]
    fn count(&self) -> usize {
        self.0.count
    }

    /// Radial error.
    #[getter]
    fn radial(&self) -> Option<PyErrorStats> {
        self.0.radial.map(PyErrorStats)
    }

    /// Transverse error.
    #[getter]
    fn transverse(&self) -> Option<PyErrorStats> {
        self.0.transverse.map(PyErrorStats)
    }

    /// Normal error.
    #[getter]
    fn normal(&self) -> Option<PyErrorStats> {
        self.0.normal.map(PyErrorStats)
    }

    /// Three-dimensional orbit error.
    #[getter]
    fn orbit_3d(&self) -> Option<PyErrorStats> {
        self.0.orbit_3d.map(PyErrorStats)
    }

    /// Orbit-only SISRE.
    #[getter]
    fn sisre_orbit(&self) -> Option<PyErrorStats> {
        self.0.sisre_orbit.map(PyErrorStats)
    }

    /// Number of scored clock epochs.
    #[getter]
    fn clock_count(&self) -> usize {
        self.0.clock_count
    }

    /// Clock error.
    #[getter]
    fn clock(&self) -> Option<PyErrorStats> {
        self.0.clock.map(PyErrorStats)
    }

    /// SISRE with clock.
    #[getter]
    fn sisre(&self) -> Option<PyErrorStats> {
        self.0.sisre.map(PyErrorStats)
    }
}

/// Counts of evaluated and missing satellite epochs.
#[pyclass(module = "sidereon._sidereon", name = "ScoreCoverage", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyScoreCoverage(scoring::ScoreCoverage);

#[pymethods]
impl PyScoreCoverage {
    /// Build empty coverage counts.
    #[new]
    fn new() -> Self {
        Self(scoring::ScoreCoverage::default())
    }

    /// Requested satellite and epoch pairs.
    #[getter]
    fn epochs(&self) -> usize {
        self.0.epochs
    }

    /// Pairs scored.
    #[getter]
    fn scored(&self) -> usize {
        self.0.scored
    }

    /// Scored pairs with clock errors.
    #[getter]
    fn clocked(&self) -> usize {
        self.0.clocked
    }

    /// Pairs outside the horizon bins.
    #[getter]
    fn outside_horizon_bins(&self) -> usize {
        self.0.outside_horizon_bins
    }

    /// Pairs missing a prediction origin.
    #[getter]
    fn no_prediction_origin(&self) -> usize {
        self.0.no_prediction_origin
    }

    /// Pairs whose satellite is missing from the source.
    #[getter]
    fn source_satellite_missing(&self) -> usize {
        self.0.source_satellite_missing
    }

    /// Pairs whose epoch is missing from the source.
    #[getter]
    fn source_epoch_missing(&self) -> usize {
        self.0.source_epoch_missing
    }

    /// Pairs whose satellite is missing from the reference.
    #[getter]
    fn reference_satellite_missing(&self) -> usize {
        self.0.reference_satellite_missing
    }

    /// Pairs whose epoch is missing from the reference.
    #[getter]
    fn reference_epoch_missing(&self) -> usize {
        self.0.reference_epoch_missing
    }
}

/// Prediction-horizon scores for one satellite.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteScore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PySatelliteScore(scoring::SatelliteScore);

#[pymethods]
impl PySatelliteScore {
    /// Satellite identifier.
    #[getter]
    fn satellite(&self) -> String {
        self.0.satellite.to_string()
    }

    /// Prediction origin in seconds from J2000.
    #[getter]
    fn prediction_origin_j2000_s(&self) -> Option<f64> {
        self.0.prediction_origin_j2000_s
    }

    /// Reference orbit regime, if determined.
    #[getter]
    fn orbit_regime(&self) -> Option<PyOrbitRegime> {
        self.0.orbit_regime.map(Into::into)
    }

    /// SISRE weights applied to the satellite.
    #[getter]
    fn sisre_weights(&self) -> Option<PySisreWeights> {
        self.0.sisre_weights.map(PySisreWeights)
    }

    /// Statistics by horizon bin.
    #[getter]
    fn bins(&self) -> Vec<PyHorizonBinStats> {
        self.0.bins.iter().cloned().map(PyHorizonBinStats).collect()
    }

    /// Coverage of requested epochs.
    #[getter]
    fn coverage(&self) -> PyScoreCoverage {
        PyScoreCoverage(self.0.coverage)
    }
}

/// Prediction-horizon scores for one constellation.
#[pyclass(module = "sidereon._sidereon", name = "SystemScore", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PySystemScore(scoring::SystemScore);

#[pymethods]
impl PySystemScore {
    /// GNSS constellation.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.0.system.into()
    }

    /// Statistics by horizon bin.
    #[getter]
    fn bins(&self) -> Vec<PyHorizonBinStats> {
        self.0.bins.iter().cloned().map(PyHorizonBinStats).collect()
    }

    /// Coverage across satellites.
    #[getter]
    fn coverage(&self) -> PyScoreCoverage {
        PyScoreCoverage(self.0.coverage)
    }
}

/// Orbit and clock prediction score report.
#[pyclass(module = "sidereon._sidereon", name = "OrbitScoreReport", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitScoreReport(scoring::OrbitScoreReport);

#[pymethods]
impl PyOrbitScoreReport {
    /// Horizon bin edges in seconds.
    #[getter]
    fn horizon_bin_edges_s(&self) -> Vec<f64> {
        self.0.horizon_bin_edges_s.clone()
    }

    /// Frame agreement or reconciliation.
    #[getter]
    fn frame(&self) -> PyFrameAgreement {
        PyFrameAgreement(self.0.frame.clone())
    }

    /// Point represented by source positions.
    #[getter]
    fn source_reference_point(&self) -> PyOrbitScoreReferencePoint {
        self.0.source_reference_point.into()
    }

    /// Scores by satellite, in requested order.
    #[getter]
    fn per_satellite(&self) -> Vec<PySatelliteScore> {
        self.0
            .per_satellite
            .iter()
            .cloned()
            .map(PySatelliteScore)
            .collect()
    }

    /// Scores by constellation.
    #[getter]
    fn per_system(&self) -> Vec<PySystemScore> {
        self.0
            .per_system
            .iter()
            .cloned()
            .map(PySystemScore)
            .collect()
    }

    /// Coverage across all requested epochs and satellites.
    #[getter]
    fn coverage(&self) -> PyScoreCoverage {
        PyScoreCoverage(self.0.coverage)
    }
}

/// Earth-fixed satellite position and clock at one epoch.
#[pyclass(module = "sidereon._sidereon", name = "TrajectorySample", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyTrajectorySample(scoring::TrajectorySample);

#[pymethods]
impl PyTrajectorySample {
    /// Build a trajectory sample with an optional clock value.
    #[new]
    #[pyo3(signature = (satellite, epoch, position_ecef_m, clock_s=None))]
    fn new(
        satellite: &str,
        epoch: PyClockInstant,
        position_ecef_m: [f64; 3],
        clock_s: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self(scoring::TrajectorySample {
            satellite: parse_sat(satellite)?,
            epoch: epoch.to_core(),
            position_ecef_m,
            clock_s,
        }))
    }

    /// Satellite identifier.
    #[getter]
    fn satellite(&self) -> String {
        self.0.satellite.to_string()
    }

    /// Epoch in the trajectory time scale.
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.0.epoch)
    }

    /// Earth-fixed position in metres.
    #[getter]
    fn position_ecef_m(&self) -> [f64; 3] {
        self.0.position_ecef_m
    }

    /// Satellite clock offset in seconds, when present.
    #[getter]
    fn clock_s(&self) -> Option<f64> {
        self.0.clock_s
    }
}

/// How source coordinates were matched to the reference.
#[pyclass(module = "sidereon._sidereon", name = "FrameAgreement", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyFrameAgreement(scoring::FrameAgreement);

#[pymethods]
impl PyFrameAgreement {
    /// The two labels match.
    #[classattr]
    const SAME_LABEL: Self = Self(scoring::FrameAgreement::SameLabel);
    /// The labels were asserted equivalent.
    #[classattr]
    const ASSERTED_EQUIVALENT: Self = Self(scoring::FrameAgreement::AssertedEquivalent);
    /// The source uses the constellation broadcast datums.
    #[classattr]
    const BROADCAST_DATUMS: Self = Self(scoring::FrameAgreement::BroadcastDatums);
    /// Represent a Helmert transformation between two terrestrial realizations.
    #[staticmethod]
    fn helmert(from_frame: PyTerrestrialFrame, to_frame: PyTerrestrialFrame) -> Self {
        Self(scoring::FrameAgreement::Helmert {
            from: from_frame.into(),
            to: to_frame.into(),
        })
    }

    /// Source realization in a Helmert transformation.
    #[getter]
    #[allow(clippy::wrong_self_convention)]
    fn from_frame(&self) -> Option<PyTerrestrialFrame> {
        match self.0 {
            scoring::FrameAgreement::Helmert { from, .. } => Some(from.into()),
            _ => None,
        }
    }

    /// Reference realization in a Helmert transformation.
    #[getter]
    fn to_frame(&self) -> Option<PyTerrestrialFrame> {
        match self.0 {
            scoring::FrameAgreement::Helmert { to, .. } => Some(to.into()),
            _ => None,
        }
    }
}

/// Radial and transverse-normal weights for signal-in-space range error.
#[pyclass(module = "sidereon._sidereon", name = "SisreWeights", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PySisreWeights(scoring::SisreWeights);

#[pymethods]
impl PySisreWeights {
    /// Build SISRE weights from the radial weight and squared transverse-normal weight.
    #[new]
    fn new(radial: f64, transverse_normal_squared: f64) -> PyResult<Self> {
        scoring::SisreWeights::new(radial, transverse_normal_squared)
            .map(Self)
            .map_err(orbit_score_error)
    }

    /// Radial weight.
    fn radial(&self) -> f64 {
        self.0.radial()
    }

    /// Squared transverse-normal weight.
    fn transverse_normal_squared(&self) -> f64 {
        self.0.transverse_normal_squared()
    }

    /// Published SISRE weights for a constellation and regime, when available.
    #[staticmethod]
    fn montenbruck_2018(system: PyGnssSystem, regime: PyOrbitRegime) -> Option<Self> {
        scoring::SisreWeights::montenbruck_2018(system.into(), regime.into()).map(Self)
    }

    /// Compute the signal-in-space range error in metres.
    fn orbit_only_m(&self, components: &PyRtnComponents) -> PyResult<f64> {
        self.0.orbit_only_m(components.0).map_err(orbit_score_error)
    }

    /// Compute the signal-in-space range error in metres.
    fn with_clock_m(&self, components: &PyRtnComponents, clock_m: f64) -> PyResult<f64> {
        self.0
            .with_clock_m(components.0, clock_m)
            .map_err(orbit_score_error)
    }
}

/// Coordinate frame of an orbit source.
#[pyclass(module = "sidereon._sidereon", name = "OrbitFrame", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitFrame(scoring::OrbitFrame);

#[pymethods]
impl PyOrbitFrame {
    /// Frame as reported by the broadcast ephemerides.
    #[classattr]
    const BROADCAST_DATUMS: Self = Self(scoring::OrbitFrame::BroadcastDatums);
    /// Build the selection from label.
    #[staticmethod]
    fn label(label: String) -> Self {
        Self(scoring::OrbitFrame::Label(label))
    }

    /// Configured label, if explicitly supplied.
    #[getter]
    fn frame_label(&self) -> Option<String> {
        match &self.0 {
            scoring::OrbitFrame::Label(value) => Some(value.clone()),
            _ => None,
        }
    }
}

/// Epochs to score against the reference product.
#[pyclass(module = "sidereon._sidereon", name = "ScoreEpochs", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyScoreEpochs(scoring::ScoreEpochs);

#[pymethods]
impl PyScoreEpochs {
    /// Default selection.
    #[classattr]
    #[allow(non_snake_case)]
    fn REFERENCE_RECORDS() -> Self {
        Self(scoring::ScoreEpochs::default())
    }
    /// Build the core default selection.
    #[new]
    fn new() -> Self {
        Self(scoring::ScoreEpochs::default())
    }

    /// Build the selection from instants.
    #[staticmethod]
    fn instants(instants: Vec<PyClockInstant>) -> Self {
        Self(scoring::ScoreEpochs::Instants(
            instants.into_iter().map(PyClockInstant::to_core).collect(),
        ))
    }

    /// Configured instants, if explicitly supplied.
    #[getter]
    fn epoch_instants(&self) -> Option<Vec<PyClockInstant>> {
        match &self.0 {
            scoring::ScoreEpochs::Instants(values) => Some(
                values
                    .iter()
                    .copied()
                    .map(PyClockInstant::from_core)
                    .collect(),
            ),
            _ => None,
        }
    }
}

/// Choice of published or fixed SISRE weights.
#[pyclass(
    module = "sidereon._sidereon",
    name = "SisreWeightSelection",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PySisreWeightSelection(scoring::SisreWeightSelection);

#[pymethods]
impl PySisreWeightSelection {
    /// Default selection.
    #[classattr]
    #[allow(non_snake_case)]
    fn MONTENBRUCK_2018() -> Self {
        Self(scoring::SisreWeightSelection::default())
    }
    /// Build the core default selection.
    #[new]
    fn new() -> Self {
        Self(scoring::SisreWeightSelection::default())
    }

    /// Build the selection from weights.
    #[staticmethod]
    fn fixed(weights: &PySisreWeights) -> Self {
        Self(scoring::SisreWeightSelection::Fixed(weights.0))
    }

    /// Configured weights, if explicitly supplied.
    #[getter]
    fn fixed_weights(&self) -> Option<PySisreWeights> {
        match self.0 {
            scoring::SisreWeightSelection::Fixed(value) => Some(PySisreWeights(value)),
            _ => None,
        }
    }
}

/// Frame assertions and catalog Helmert policy for comparing orbit products.
#[pyclass(
    module = "sidereon._sidereon",
    name = "Sp3FrameReconciliationOptions",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PySp3FrameReconciliationOptions(Sp3FrameReconciliationOptions);

#[pymethods]
impl PySp3FrameReconciliationOptions {
    /// Build frame reconciliation controls.
    #[new]
    #[pyo3(signature = (asserted_equivalent_label_sets=Vec::new(), helmert=None))]
    fn new(asserted_equivalent_label_sets: Vec<Vec<String>>, helmert: Option<bool>) -> Self {
        let mut options = Sp3FrameReconciliationOptions::disabled();
        options.asserted_equivalent_label_sets = asserted_equivalent_label_sets
            .into_iter()
            .map(Sp3FrameLabelSet::new)
            .collect();
        if let Some(helmert) = helmert {
            options.helmert = helmert;
        }
        Self(options)
    }

    /// Sets of labels asserted to represent one terrestrial frame.
    #[getter]
    fn asserted_equivalent_label_sets(&self) -> Vec<Vec<String>> {
        self.0
            .asserted_equivalent_label_sets
            .iter()
            .map(|set| set.labels.iter().cloned().collect())
            .collect()
    }

    /// Whether catalog Helmert transformations are allowed.
    #[getter]
    fn helmert(&self) -> bool {
        self.0.helmert
    }
}

/// Controls for orbit prediction scoring.
#[pyclass(module = "sidereon._sidereon", name = "OrbitScoreOptions", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitScoreOptions(scoring::OrbitScoreOptions);

#[pymethods]
impl PyOrbitScoreOptions {
    /// Configure horizon bins, epoch selection, reference triad, and comparison policies.
    #[new]
    #[pyo3(signature = (horizon_bin_edges_s, *, epochs=None, triad=None, velocity_half_step_s=None, sisre_weights=None, clock_datum=None, gaps=None, frame_reconciliation=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        horizon_bin_edges_s: Vec<f64>,
        epochs: Option<&PyScoreEpochs>,
        triad: Option<PyTriadVelocity>,
        velocity_half_step_s: Option<f64>,
        sisre_weights: Option<&PySisreWeightSelection>,
        clock_datum: Option<PyClockDatum>,
        gaps: Option<PyOrbitScoreGapPolicy>,
        frame_reconciliation: Option<&PySp3FrameReconciliationOptions>,
    ) -> Self {
        let mut options = scoring::OrbitScoreOptions::new(horizon_bin_edges_s);
        if let Some(epochs) = epochs {
            options.epochs = epochs.0.clone();
        }
        if let Some(triad) = triad {
            options.triad = triad.into();
        }
        if let Some(velocity_half_step_s) = velocity_half_step_s {
            options.velocity_half_step_s = velocity_half_step_s;
        }
        if let Some(weights) = sisre_weights {
            options.sisre_weights = weights.0;
        }
        if let Some(clock_datum) = clock_datum {
            options.clock_datum = clock_datum.into();
        }
        if let Some(gaps) = gaps {
            options.gaps = gaps.into();
        }
        if let Some(frames) = frame_reconciliation {
            options.frame_reconciliation = frames.0.clone();
        }
        Self(options)
    }
    /// Epoch selection.
    #[getter]
    fn epochs(&self) -> PyScoreEpochs {
        PyScoreEpochs(self.0.epochs.clone())
    }

    /// Prediction horizon bin edges in seconds.
    #[getter]
    fn horizon_bin_edges_s(&self) -> Vec<f64> {
        self.0.horizon_bin_edges_s.clone()
    }

    /// Frame velocity selection.
    #[getter]
    fn triad(&self) -> PyTriadVelocity {
        self.0.triad.into()
    }

    /// Reference velocity half step in seconds.
    #[getter]
    fn velocity_half_step_s(&self) -> f64 {
        self.0.velocity_half_step_s
    }

    /// SISRE weight selection.
    #[getter]
    fn sisre_weights(&self) -> PySisreWeightSelection {
        PySisreWeightSelection(self.0.sisre_weights)
    }

    /// Clock comparison datum.
    #[getter]
    fn clock_datum(&self) -> PyClockDatum {
        self.0.clock_datum.into()
    }

    /// Policy for missing satellite or epoch data.
    #[getter]
    fn gaps(&self) -> PyOrbitScoreGapPolicy {
        self.0.gaps.into()
    }

    /// Frame reconciliation policy.
    #[getter]
    fn frame_reconciliation(&self) -> PySp3FrameReconciliationOptions {
        PySp3FrameReconciliationOptions(self.0.frame_reconciliation.clone())
    }
}

/// An SP3 orbit with a prediction origin from flags or a caller cutoff.
#[pyclass(module = "sidereon._sidereon", name = "PredictedSp3")]
pub struct PyPredictedSp3 {
    product: Py<PySp3>,
    cutoff: Option<Instant>,
}

impl PyPredictedSp3 {
    fn with_source<T>(
        &self,
        py: Python<'_>,
        evaluate: impl FnOnce(&dyn OrbitSource) -> PyResult<T>,
    ) -> PyResult<T> {
        let product = self.product.borrow(py);
        let source = match self.cutoff {
            Some(cutoff) => scoring::PredictedSp3::with_cutoff(&product.inner, cutoff),
            None => scoring::PredictedSp3::from_prediction_flags(&product.inner),
        }
        .map_err(orbit_score_error)?;
        evaluate(&source)
    }
}

#[pymethods]
impl PyPredictedSp3 {
    /// Take each satellite's prediction origin from the SP3 prediction flags.
    #[staticmethod]
    fn from_prediction_flags(py: Python<'_>, product: Py<PySp3>) -> PyResult<Self> {
        let result = Self {
            product,
            cutoff: None,
        };
        result.with_source(py, |_| Ok(()))?;
        Ok(result)
    }

    /// Use an explicit prediction cutoff in the product's time scale.
    #[staticmethod]
    fn with_cutoff(py: Python<'_>, product: Py<PySp3>, cutoff: PyClockInstant) -> PyResult<Self> {
        let result = Self {
            product,
            cutoff: Some(cutoff.to_core()),
        };
        result.with_source(py, |_| Ok(()))?;
        Ok(result)
    }

    /// The SP3 product used by the score.
    fn product(&self, py: Python<'_>) -> Py<PySp3> {
        self.product.clone_ref(py)
    }

    /// Time scale of the source epoch axis.
    fn time_scale(&self, py: Python<'_>) -> PyResult<PyTimeScale> {
        self.with_source(py, |source| Ok(source.time_scale().into()))
    }

    /// Explicit SP3 time-system label, when present.
    fn sp3_time_system(&self, py: Python<'_>) -> PyResult<Option<&'static str>> {
        self.with_source(py, |source| {
            Ok(source.sp3_time_system().map(|system| system.label()))
        })
    }

    /// Coordinate frame of the source positions.
    fn frame(&self, py: Python<'_>) -> PyResult<PyOrbitFrame> {
        self.with_source(py, |source| Ok(PyOrbitFrame(source.frame())))
    }

    /// Satellite point represented by source positions.
    fn reference_point(&self, py: Python<'_>) -> PyResult<PyOrbitScoreReferencePoint> {
        self.with_source(py, |source| Ok(source.reference_point().into()))
    }

    /// Whether the source clock includes periodic relativity.
    fn clock_includes_relativity(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_source(py, |source| Ok(source.clock_includes_relativity()))
    }

    /// Source state for a satellite at seconds from J2000 in the source scale.
    fn state_at_j2000_s(
        &self,
        py: Python<'_>,
        satellite: &str,
        t_j2000_s: f64,
    ) -> PyResult<PyOrbitSourceState> {
        let satellite = parse_sat(satellite)?;
        self.with_source(py, |source| {
            source
                .state_at_j2000_s(satellite, t_j2000_s)
                .map(PyOrbitSourceState)
                .map_err(crate::ephemeris::ephemeris_source_query_error)
        })
    }

    /// Prediction origin for the satellite, in seconds from J2000 in the source scale.
    fn prediction_origin_j2000_s(&self, py: Python<'_>, satellite: &str) -> PyResult<Option<f64>> {
        let satellite = parse_sat(satellite)?;
        self.with_source(py, |source| Ok(source.prediction_origin_j2000_s(satellite)))
    }
}

/// A broadcast ephemeris with a caller-supplied prediction origin.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastOrbit")]
pub struct PyBroadcastOrbit {
    ephemeris: Py<PyBroadcastEphemeris>,
    prediction_origin: Instant,
}

impl PyBroadcastOrbit {
    fn with_source<T>(
        &self,
        py: Python<'_>,
        evaluate: impl FnOnce(&dyn OrbitSource) -> PyResult<T>,
    ) -> PyResult<T> {
        let ephemeris = self.ephemeris.borrow(py);
        let source = scoring::BroadcastOrbit::new(&ephemeris.inner, self.prediction_origin)
            .map_err(orbit_score_error)?;
        evaluate(&source)
    }
}

#[pymethods]
impl PyBroadcastOrbit {
    /// Build with a GPS-time prediction origin.
    #[new]
    fn new(
        py: Python<'_>,
        ephemeris: Py<PyBroadcastEphemeris>,
        prediction_origin: PyClockInstant,
    ) -> PyResult<Self> {
        let result = Self {
            ephemeris,
            prediction_origin: prediction_origin.to_core(),
        };
        result.with_source(py, |_| Ok(()))?;
        Ok(result)
    }

    /// Time scale of the source epoch axis.
    fn time_scale(&self, py: Python<'_>) -> PyResult<PyTimeScale> {
        self.with_source(py, |source| Ok(source.time_scale().into()))
    }

    /// Explicit SP3 time-system label, when present.
    fn sp3_time_system(&self, py: Python<'_>) -> PyResult<Option<&'static str>> {
        self.with_source(py, |source| {
            Ok(source.sp3_time_system().map(|system| system.label()))
        })
    }

    /// Coordinate frame of the source positions.
    fn frame(&self, py: Python<'_>) -> PyResult<PyOrbitFrame> {
        self.with_source(py, |source| Ok(PyOrbitFrame(source.frame())))
    }

    /// Satellite point represented by source positions.
    fn reference_point(&self, py: Python<'_>) -> PyResult<PyOrbitScoreReferencePoint> {
        self.with_source(py, |source| Ok(source.reference_point().into()))
    }

    /// Whether the source clock includes periodic relativity.
    fn clock_includes_relativity(&self, py: Python<'_>) -> PyResult<bool> {
        self.with_source(py, |source| Ok(source.clock_includes_relativity()))
    }

    /// Source state for a satellite at seconds from J2000 in the source scale.
    fn state_at_j2000_s(
        &self,
        py: Python<'_>,
        satellite: &str,
        t_j2000_s: f64,
    ) -> PyResult<PyOrbitSourceState> {
        let satellite = parse_sat(satellite)?;
        self.with_source(py, |source| {
            source
                .state_at_j2000_s(satellite, t_j2000_s)
                .map(PyOrbitSourceState)
                .map_err(crate::ephemeris::ephemeris_source_query_error)
        })
    }

    /// Prediction origin for the satellite, in seconds from J2000 in the source scale.
    fn prediction_origin_j2000_s(&self, py: Python<'_>, satellite: &str) -> PyResult<Option<f64>> {
        let satellite = parse_sat(satellite)?;
        self.with_source(py, |source| Ok(source.prediction_origin_j2000_s(satellite)))
    }
}

/// Caller-supplied Earth-fixed orbit states at explicit sample epochs.
#[pyclass(module = "sidereon._sidereon", name = "OrbitTrajectory", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitTrajectory(scoring::OrbitTrajectory);

#[pymethods]
impl PyOrbitTrajectory {
    /// Build from a frame label, prediction origin, and trajectory samples.
    #[new]
    fn new(
        frame_label: String,
        prediction_origin: PyClockInstant,
        samples: Vec<PyTrajectorySample>,
    ) -> PyResult<Self> {
        scoring::OrbitTrajectory::new(
            frame_label,
            prediction_origin.to_core(),
            samples.into_iter().map(|sample| sample.0),
        )
        .map(Self)
        .map_err(orbit_score_error)
    }

    /// Set whether sample clocks include the periodic relativistic correction.
    fn with_clock_relativity_included(&self, included: bool) -> Self {
        Self(self.0.clone().with_clock_relativity_included(included))
    }

    /// Distinct sample epochs in ascending order.
    fn epochs(&self) -> Vec<PyClockInstant> {
        self.0
            .epochs()
            .into_iter()
            .map(PyClockInstant::from_core)
            .collect()
    }

    /// Satellite identifiers in ascending order.
    fn satellites(&self) -> Vec<String> {
        self.0
            .satellites()
            .map(|satellite| satellite.to_string())
            .collect()
    }

    /// Time scale of the source epoch axis.
    fn time_scale(&self) -> PyTimeScale {
        self.0.time_scale().into()
    }

    /// Explicit SP3 time-system label, when present.
    fn sp3_time_system(&self) -> Option<&'static str> {
        self.0.sp3_time_system().map(|system| system.label())
    }

    /// Coordinate frame of the source positions.
    fn frame(&self) -> PyOrbitFrame {
        PyOrbitFrame(self.0.frame())
    }

    /// Satellite point represented by source positions.
    fn reference_point(&self) -> PyOrbitScoreReferencePoint {
        self.0.reference_point().into()
    }

    /// Whether the source clock includes periodic relativity.
    fn clock_includes_relativity(&self) -> bool {
        self.0.clock_includes_relativity()
    }

    /// Source state for a satellite at seconds from J2000 in the source scale.
    fn state_at_j2000_s(&self, satellite: &str, t_j2000_s: f64) -> PyResult<PyOrbitSourceState> {
        let satellite = parse_sat(satellite)?;
        self.0
            .state_at_j2000_s(satellite, t_j2000_s)
            .map(PyOrbitSourceState)
            .map_err(crate::ephemeris::ephemeris_source_query_error)
    }

    /// Prediction origin for the satellite, in seconds from J2000 in the source scale.
    fn prediction_origin_j2000_s(&self, satellite: &str) -> PyResult<Option<f64>> {
        let satellite = parse_sat(satellite)?;
        Ok(self.0.prediction_origin_j2000_s(satellite))
    }
}

#[derive(FromPyObject)]
pub(crate) enum PyOrbitSource {
    PredictedSp3(Py<PyPredictedSp3>),
    BroadcastOrbit(Py<PyBroadcastOrbit>),
    OrbitTrajectory(Py<PyOrbitTrajectory>),
}

impl PyOrbitSource {
    pub(crate) fn with_source<T>(
        &self,
        py: Python<'_>,
        evaluate: impl FnOnce(&dyn OrbitSource) -> PyResult<T>,
    ) -> PyResult<T> {
        match self {
            Self::PredictedSp3(source) => source.borrow(py).with_source(py, evaluate),
            Self::BroadcastOrbit(source) => source.borrow(py).with_source(py, evaluate),
            Self::OrbitTrajectory(source) => evaluate(&source.borrow(py).0),
        }
    }
}

/// Compare orbit and clock predictions against a reference SP3 product.
#[pyfunction]
fn score_orbit(
    py: Python<'_>,
    source: PyOrbitSource,
    reference: &PySp3,
    satellites: Vec<String>,
    options: &PyOrbitScoreOptions,
) -> PyResult<PyOrbitScoreReport> {
    let satellites: Vec<GnssSatelliteId> = parse_satellites(&satellites)?;
    source.with_source(py, |source| {
        scoring::score_orbit(source, &reference.inner, &satellites, &options.0)
            .map(PyOrbitScoreReport)
            .map_err(orbit_score_error)
    })
}

/// Resolve a position difference on the reference orbit's radial, transverse and normal frame.
#[pyfunction]
fn rtn_components(
    difference_m: [f64; 3],
    position_m: [f64; 3],
    velocity_m_s: [f64; 3],
) -> PyResult<PyRtnComponents> {
    scoring::rtn_components(difference_m, position_m, velocity_m_s)
        .map(PyRtnComponents)
        .map_err(orbit_score_error)
}

/// Return inertial velocity expressed in Earth-fixed axes, in metres per second.
#[pyfunction]
fn inertial_velocity_from_earth_fixed(
    position_ecef_m: [f64; 3],
    velocity_ecef_m_s: [f64; 3],
) -> PyResult<[f64; 3]> {
    scoring::inertial_velocity_from_earth_fixed(position_ecef_m, velocity_ecef_m_s)
        .map_err(orbit_score_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("OrbitScoreError", m.py().get_type::<OrbitScoreError>())?;
    m.add_class::<PyTriadVelocity>()?;
    m.add_class::<PyClockDatum>()?;
    m.add_class::<PyOrbitScoreGapPolicy>()?;
    m.add_class::<PyOrbitScoreReferencePoint>()?;
    m.add_class::<PyOrbitRegime>()?;
    m.add_class::<PyScoredProduct>()?;
    m.add(
        "DEFAULT_VELOCITY_HALF_STEP_S",
        scoring::DEFAULT_VELOCITY_HALF_STEP_S,
    )?;
    m.add(
        "MIN_VELOCITY_HALF_STEP_S",
        scoring::MIN_VELOCITY_HALF_STEP_S,
    )?;
    m.add_class::<PyRtnComponents>()?;
    m.add_class::<PyOrbitSourceState>()?;
    m.add_class::<PyErrorStats>()?;
    m.add_class::<PyHorizonBinStats>()?;
    m.add_class::<PyScoreCoverage>()?;
    m.add_class::<PySatelliteScore>()?;
    m.add_class::<PySystemScore>()?;
    m.add_class::<PyOrbitScoreReport>()?;
    m.add_class::<PyTrajectorySample>()?;
    m.add_class::<PyFrameAgreement>()?;
    m.add_class::<PySisreWeights>()?;
    m.add_class::<PyOrbitFrame>()?;
    m.add_class::<PyScoreEpochs>()?;
    m.add_class::<PySisreWeightSelection>()?;
    m.add_class::<PySp3FrameReconciliationOptions>()?;
    m.add_class::<PyOrbitScoreOptions>()?;
    m.add_class::<PyOrbitTrajectory>()?;
    m.add_class::<PyPredictedSp3>()?;
    m.add_class::<PyBroadcastOrbit>()?;
    m.add_function(wrap_pyfunction!(score_orbit, m)?)?;
    m.add_function(wrap_pyfunction!(rtn_components, m)?)?;
    m.add_function(wrap_pyfunction!(inertial_velocity_from_earth_fixed, m)?)?;
    Ok(())
}
