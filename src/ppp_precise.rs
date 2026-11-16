//! Dual-frequency preparation, cycle-slip, and TEC surfaces for precise positioning.
//!
//! This module only converts Python values to the core's records and delegates
//! all validation and numerical work to `sidereon-core`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use sidereon_core::carrier_phase::SlipReason;
use sidereon_core::precise_positioning as core;

use crate::ephemeris::PySp3;
use crate::events::PyWgs84Geodetic;
use crate::exact_time::PyExactEpoch;
use crate::observables::PyCycleSlipOptions;
use crate::ppp::{
    PppUnplacedObservationRow, PyPppEpoch, PyPppFloatConfig, PyPppFloatResidual,
    PyPppFloatSolution, PyPppFloatState, PyPppMeasurementWeights, PyPppRangeCorrections,
    PyPppSsrBiasExclusion, PyPppTroposphereOptions,
};
use crate::SolveError;

fn detailed_error(
    py: Python<'_>,
    family: &str,
    kind: &str,
    message: String,
    fields: &[(&str, String)],
) -> PyErr {
    let exception = SolveError::new_err(message);
    let detail = PyDict::new(py);
    if detail.set_item("family", family).is_ok() && detail.set_item("kind", kind).is_ok() {
        for (key, value) in fields {
            if detail.set_item(*key, value).is_err() {
                return exception;
            }
        }
        let _ = exception.value(py).setattr("detail", detail);
    }
    exception
}

fn attach_detail(py: Python<'_>, message: String, detail: &Bound<'_, PyDict>) -> PyErr {
    let exception = SolveError::new_err(message);
    let _ = exception.value(py).setattr("detail", detail);
    exception
}

fn tec_error(py: Python<'_>, error: core::TecError) -> PyErr {
    use core::TecError as E;
    let kind = match error {
        E::NonFiniteObservation => "non_finite_observation",
        E::InvalidShellHeight => "invalid_shell_height",
        E::InvalidEarthRadius => "invalid_earth_radius",
        E::InvalidFrequency => "invalid_frequency",
        E::EqualFrequencies => "equal_frequencies",
        E::InvalidReceiverLatitude => "invalid_receiver_latitude",
        E::InvalidReceiverLongitude => "invalid_receiver_longitude",
        E::InvalidElevation => "invalid_elevation",
        E::InvalidAzimuth => "invalid_azimuth",
        E::NonFiniteTec => "non_finite_tec",
        E::EmptyArc => "empty_arc",
        E::NoEpochs => "no_epochs",
        E::NoObservations => "no_observations",
        E::NonFiniteEpochTime => "non_finite_epoch_time",
        E::EpochsNotOrdered => "epochs_not_ordered",
        E::InsufficientArcSamples => "insufficient_arc_samples",
    };
    detailed_error(py, "TecError", kind, error.to_string(), &[])
}

fn cycle_config_kind(error: core::CycleSlipConfigError) -> &'static str {
    use core::CycleSlipConfigError as E;
    match error {
        E::InvalidMelbourneWubbenaThreshold => "invalid_melbourne_wubbena_threshold",
        E::InvalidGeometryFreeThreshold => "invalid_geometry_free_threshold",
        E::InvalidMinimumArcLength => "invalid_minimum_arc_length",
        E::InvalidRunningStatisticKFactor => "invalid_running_statistic_k_factor",
        E::InvalidMaximumGap => "invalid_maximum_gap",
    }
}
fn cycle_config_error(py: Python<'_>, error: core::CycleSlipConfigError) -> PyErr {
    detailed_error(
        py,
        "CycleSlipConfigError",
        cycle_config_kind(error),
        error.to_string(),
        &[],
    )
}
fn cycle_error(py: Python<'_>, error: core::CycleSlipError) -> PyErr {
    let message = error.to_string();
    match error {
        core::CycleSlipError::InvalidConfig(reason) => detailed_error(
            py,
            "CycleSlipError",
            "invalid_config",
            message,
            &[("reason", cycle_config_kind(reason).to_string())],
        ),
        core::CycleSlipError::NonFiniteObservation => {
            detailed_error(py, "CycleSlipError", "non_finite_observation", message, &[])
        }
        core::CycleSlipError::NonFiniteEpochTime => {
            detailed_error(py, "CycleSlipError", "non_finite_epoch_time", message, &[])
        }
        core::CycleSlipError::EpochsNotOrdered => {
            detailed_error(py, "CycleSlipError", "epochs_not_ordered", message, &[])
        }
        core::CycleSlipError::InvalidFrequency => {
            detailed_error(py, "CycleSlipError", "invalid_frequency", message, &[])
        }
        core::CycleSlipError::EqualFrequencies => {
            detailed_error(py, "CycleSlipError", "equal_frequencies", message, &[])
        }
    }
}

fn wide_lane_error(py: Python<'_>, error: core::WideLanePrepError) -> PyErr {
    use core::WideLanePrepError as E;
    let message = format!("{error:?}");
    let detail = PyDict::new(py);
    let _ = detail.set_item("family", "WideLanePrepError");
    match error {
        E::CycleSlipDetected {
            satellite_id,
            epoch_index,
            reasons,
        } => {
            let _ = detail.set_item("kind", "cycle_slip_detected");
            let _ = detail.set_item("satellite_id", satellite_id);
            let _ = detail.set_item("epoch_index", epoch_index);
            let _ = detail.set_item(
                "reasons",
                reasons.iter().map(slip_reason).collect::<Vec<_>>(),
            );
        }
        E::WideLaneFailed {
            ambiguity_id,
            reason,
        } => {
            let _ = detail.set_item("kind", "wide_lane_failed");
            let _ = detail.set_item("ambiguity_id", ambiguity_id);
            let _ = detail.set_item("reason", reason.to_string());
        }
        E::TooFewWideLaneEpochs {
            ambiguity_id,
            count,
            minimum,
        } => {
            let _ = detail.set_item("kind", "too_few_wide_lane_epochs");
            let _ = detail.set_item("ambiguity_id", ambiguity_id);
            let _ = detail.set_item("count", count);
            let _ = detail.set_item("minimum", minimum);
        }
        E::WideLaneNotInteger {
            ambiguity_id,
            mean_cycles,
            fixed_cycles,
        } => {
            let _ = detail.set_item("kind", "wide_lane_not_integer");
            let _ = detail.set_item("ambiguity_id", ambiguity_id);
            let _ = detail.set_item("mean_cycles", mean_cycles);
            let _ = detail.set_item(
                "mean_cycles_bits",
                format!("{:016x}", mean_cycles.to_bits()),
            );
            let _ = detail.set_item("fixed_cycles", fixed_cycles);
        }
        E::MissingWideLaneAmbiguity(ambiguity_id) => {
            let _ = detail.set_item("kind", "missing_wide_lane_ambiguity");
            let _ = detail.set_item("ambiguity_id", ambiguity_id);
        }
        E::InconsistentFrequencies(ambiguity_id) => {
            let _ = detail.set_item("kind", "inconsistent_frequencies");
            let _ = detail.set_item("ambiguity_id", ambiguity_id);
        }
        E::IonosphereFreeFailed {
            satellite_id,
            reason,
        } => {
            let _ = detail.set_item("kind", "ionosphere_free_failed");
            let _ = detail.set_item("satellite_id", satellite_id);
            let _ = detail.set_item("reason", reason.to_string());
        }
    }
    attach_detail(py, message, &detail)
}

fn raim_error(py: Python<'_>, error: core::RaimError) -> PyErr {
    use core::RaimError as E;
    let message = error.to_string();
    match error {
        E::InvalidConfig { field, reason } => detailed_error(
            py,
            "RaimError",
            "invalid_config",
            message,
            &[("field", field.to_string()), ("reason", reason.to_string())],
        ),
        E::InvalidResidual { field } => detailed_error(
            py,
            "RaimError",
            "invalid_residual",
            message,
            &[("field", field.to_string())],
        ),
        E::InvalidGeometry { field, reason } => detailed_error(
            py,
            "RaimError",
            "invalid_geometry",
            message,
            &[("field", field.to_string()), ("reason", reason.to_string())],
        ),
        E::Dop(reason) => {
            let reason_text = reason.to_string();
            detailed_error(py, "RaimError", "dop", message, &[("reason", reason_text)])
        }
        E::SingularGeometry => detailed_error(py, "RaimError", "singular_geometry", message, &[]),
        E::NumericalProtectionLevel => {
            detailed_error(py, "RaimError", "numerical_protection_level", message, &[])
        }
        E::UnsupportedNoncentrality => {
            detailed_error(py, "RaimError", "unsupported_noncentrality", message, &[])
        }
        E::UnsupportedDof { dof, max } => {
            let detail = PyDict::new(py);
            let _ = detail.set_item("family", "RaimError");
            let _ = detail.set_item("kind", "unsupported_dof");
            let _ = detail.set_item("dof", dof);
            let _ = detail.set_item("max", max);
            attach_detail(py, message, &detail)
        }
    }
}

fn raim_fde_error(py: Python<'_>, error: core::RaimFdeError) -> PyErr {
    match error {
        core::RaimFdeError::Solve(error) => crate::solve_error_detail::ppp_float_error(py, error),
        core::RaimFdeError::Raim(error) => raim_error(py, error),
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppDualFrequencyObservation",
    eq
)]
#[derive(Clone, PartialEq)]
pub struct PyDualFrequencyObservation {
    inner: core::DualFrequencyObservation,
}

#[pymethods]
impl PyDualFrequencyObservation {
    #[new]
    #[pyo3(signature = (satellite_id, ambiguity_id, p1_m, p2_m, phi1_cyc, phi2_cyc, f1_hz, f2_hz, lli1=None, lli2=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        satellite_id: String,
        ambiguity_id: String,
        p1_m: f64,
        p2_m: f64,
        phi1_cyc: f64,
        phi2_cyc: f64,
        f1_hz: f64,
        f2_hz: f64,
        lli1: Option<i64>,
        lli2: Option<i64>,
    ) -> Self {
        Self {
            inner: core::DualFrequencyObservation {
                satellite_id,
                ambiguity_id,
                p1_m,
                p2_m,
                phi1_cyc,
                phi2_cyc,
                f1_hz,
                f2_hz,
                lli1,
                lli2,
            },
        }
    }
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn p1_m(&self) -> f64 {
        self.inner.p1_m
    }
    #[getter]
    fn p2_m(&self) -> f64 {
        self.inner.p2_m
    }
    #[getter]
    fn phi1_cyc(&self) -> f64 {
        self.inner.phi1_cyc
    }
    #[getter]
    fn phi2_cyc(&self) -> f64 {
        self.inner.phi2_cyc
    }
    #[getter]
    fn f1_hz(&self) -> f64 {
        self.inner.f1_hz
    }
    #[getter]
    fn f2_hz(&self) -> f64 {
        self.inner.f2_hz
    }
    #[getter]
    fn lli1(&self) -> Option<i64> {
        self.inner.lli1
    }
    #[getter]
    fn lli2(&self) -> Option<i64> {
        self.inner.lli2
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppDualFrequencyEpoch", eq)]
#[derive(Clone, PartialEq)]
pub struct PyDualFrequencyEpoch {
    inner: core::DualFrequencyEpoch,
}

#[pymethods]
impl PyDualFrequencyEpoch {
    #[new]
    #[pyo3(signature = (observations, gap_time_s=None, gap_epoch=None))]
    fn new(
        py: Python<'_>,
        observations: Vec<Py<PyDualFrequencyObservation>>,
        gap_time_s: Option<f64>,
        gap_epoch: Option<&PyExactEpoch>,
    ) -> Self {
        Self {
            inner: core::DualFrequencyEpoch {
                gap_time_s,
                gap_epoch: gap_epoch.map(|v| v.inner),
                observations: observations
                    .iter()
                    .map(|v| v.borrow(py).inner.clone())
                    .collect(),
            },
        }
    }
    #[getter]
    fn gap_time_s(&self) -> Option<f64> {
        self.inner.gap_time_s
    }
    #[getter]
    fn gap_epoch(&self) -> Option<PyExactEpoch> {
        self.inner.gap_epoch.map(|inner| PyExactEpoch { inner })
    }
    #[getter]
    fn observations(&self) -> Vec<PyDualFrequencyObservation> {
        self.inner
            .observations
            .iter()
            .cloned()
            .map(|inner| PyDualFrequencyObservation { inner })
            .collect()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppCycleSlipConfig", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyCycleSlipConfig {
    inner: core::CycleSlipConfig,
}

#[pymethods]
impl PyCycleSlipConfig {
    #[new]
    #[pyo3(signature = (melbourne_wubbena_threshold_cycles=4.0, geometry_free_threshold_m=0.05, minimum_arc_length=2, running_statistic_k_factor=4.0, maximum_gap_s=300.0))]
    fn new(
        melbourne_wubbena_threshold_cycles: f64,
        geometry_free_threshold_m: f64,
        minimum_arc_length: usize,
        running_statistic_k_factor: f64,
        maximum_gap_s: f64,
    ) -> Self {
        let mut inner = core::CycleSlipConfig::default();
        inner.melbourne_wubbena_threshold_cycles = melbourne_wubbena_threshold_cycles;
        inner.geometry_free_threshold_m = geometry_free_threshold_m;
        inner.minimum_arc_length = minimum_arc_length;
        inner.running_statistic_k_factor = running_statistic_k_factor;
        inner.maximum_gap_s = maximum_gap_s;
        Self { inner }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::CycleSlipConfig::default(),
        }
    }
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.validate().map_err(|e| cycle_config_error(py, e))
    }
    #[getter]
    fn melbourne_wubbena_threshold_cycles(&self) -> f64 {
        self.inner.melbourne_wubbena_threshold_cycles
    }
    #[getter]
    fn geometry_free_threshold_m(&self) -> f64 {
        self.inner.geometry_free_threshold_m
    }
    #[getter]
    fn minimum_arc_length(&self) -> usize {
        self.inner.minimum_arc_length
    }
    #[getter]
    fn running_statistic_k_factor(&self) -> f64 {
        self.inner.running_statistic_k_factor
    }
    #[getter]
    fn maximum_gap_s(&self) -> f64 {
        self.inner.maximum_gap_s
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppRunningMeanVariance", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyRunningMeanVariance {
    inner: core::RunningMeanVariance,
}

#[pymethods]
impl PyRunningMeanVariance {
    #[new]
    fn new() -> Self {
        Self {
            inner: core::RunningMeanVariance::new(),
        }
    }
    #[staticmethod]
    fn default() -> Self {
        Self::new()
    }
    fn push(&mut self, value: f64) {
        self.inner.push(value);
    }
    fn standard_deviation(&self) -> Option<f64> {
        self.inner.standard_deviation()
    }
    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.sample_count
    }
    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }
    #[getter]
    fn variance(&self) -> f64 {
        self.inner.variance
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppSatelliteCycleSlipState", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PySatelliteCycleSlipState {
    inner: core::SatelliteCycleSlipState,
}

#[pymethods]
impl PySatelliteCycleSlipState {
    #[new]
    fn new() -> Self {
        Self {
            inner: core::SatelliteCycleSlipState::new(),
        }
    }
    #[staticmethod]
    fn default() -> Self {
        Self::new()
    }
    fn reset_arc(&mut self) {
        self.inner.reset_arc();
    }
    #[getter]
    fn previous_epoch_time_s(&self) -> Option<f64> {
        self.inner.previous_epoch_time_s
    }
    #[getter]
    fn previous_epoch(&self) -> Option<PyExactEpoch> {
        self.inner
            .previous_epoch
            .map(|inner| PyExactEpoch { inner })
    }
    #[getter]
    fn previous_melbourne_wubbena_cycles(&self) -> Option<f64> {
        self.inner.previous_melbourne_wubbena_cycles
    }
    #[getter]
    fn melbourne_wubbena(&self) -> PyRunningMeanVariance {
        PyRunningMeanVariance {
            inner: self.inner.melbourne_wubbena,
        }
    }
    #[getter]
    fn previous_geometry_free_m(&self) -> Option<f64> {
        self.inner.previous_geometry_free_m
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppMelbourneWubbenaUpdate", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyMelbourneWubbenaUpdate {
    inner: core::MelbourneWubbenaUpdate,
}
#[pymethods]
impl PyMelbourneWubbenaUpdate {
    #[getter]
    fn melbourne_wubbena_cycles(&self) -> f64 {
        self.inner.melbourne_wubbena_cycles
    }
    #[getter]
    fn slip(&self) -> bool {
        self.inner.slip
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppGeometryFreeUpdate", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyGeometryFreeUpdate {
    inner: core::GeometryFreeUpdate,
}
#[pymethods]
impl PyGeometryFreeUpdate {
    #[getter]
    fn geometry_free_m(&self) -> f64 {
        self.inner.geometry_free_m
    }
    #[getter]
    fn slip(&self) -> bool {
        self.inner.slip
    }
    #[getter]
    fn reset(&self) -> bool {
        self.inner.reset
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppCycleSlipFlagObservation",
    eq
)]
#[derive(Clone, PartialEq)]
pub struct PyCycleSlipFlagObservation {
    inner: core::CycleSlipFlagObservation,
}
#[pymethods]
impl PyCycleSlipFlagObservation {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn slip(&self) -> bool {
        self.inner.slip
    }
    #[getter]
    fn reasons(&self) -> Vec<String> {
        self.inner
            .reasons
            .iter()
            .map(slip_reason)
            .map(str::to_owned)
            .collect()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppCycleSlipFlagEpoch", eq)]
#[derive(Clone, PartialEq)]
pub struct PyCycleSlipFlagEpoch {
    inner: core::CycleSlipFlagEpoch,
}
#[pymethods]
impl PyCycleSlipFlagEpoch {
    #[getter]
    fn gap_time_s(&self) -> Option<f64> {
        self.inner.gap_time_s
    }
    #[getter]
    fn gap_epoch(&self) -> Option<PyExactEpoch> {
        self.inner.gap_epoch.map(|inner| PyExactEpoch { inner })
    }
    #[getter]
    fn observations(&self) -> Vec<PyCycleSlipFlagObservation> {
        self.inner
            .observations
            .iter()
            .cloned()
            .map(|inner| PyCycleSlipFlagObservation { inner })
            .collect()
    }
}

fn slip_reason(reason: &SlipReason) -> &'static str {
    match reason {
        SlipReason::Lli => "lli",
        SlipReason::DataGap => "data_gap",
        SlipReason::GeometryFree => "geometry_free",
        SlipReason::MelbourneWubbena => "melbourne_wubbena",
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppWideLanePrepOptions", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyWideLanePrepOptions {
    inner: core::WideLanePrepOptions,
}
#[pymethods]
impl PyWideLanePrepOptions {
    #[new]
    fn new(min_epochs: usize, tolerance_cycles: f64) -> Self {
        Self {
            inner: core::WideLanePrepOptions::new(min_epochs, tolerance_cycles),
        }
    }
    #[getter]
    fn min_epochs(&self) -> usize {
        self.inner.min_epochs
    }
    #[getter]
    fn tolerance_cycles(&self) -> f64 {
        self.inner.tolerance_cycles
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppPreparedFloatObservation",
    eq
)]
#[derive(Clone, PartialEq)]
pub struct PyPreparedFloatObservation {
    inner: core::PreparedFloatObservation,
}
#[pymethods]
impl PyPreparedFloatObservation {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn code_m(&self) -> f64 {
        self.inner.code_m
    }
    #[getter]
    fn phase_m(&self) -> f64 {
        self.inner.phase_m
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppPreparedFloatEpoch", eq)]
#[derive(Clone, PartialEq)]
pub struct PyPreparedFloatEpoch {
    inner: core::PreparedFloatEpoch,
}
#[pymethods]
impl PyPreparedFloatEpoch {
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }
    #[getter]
    fn observations(&self) -> Vec<PyPreparedFloatObservation> {
        self.inner
            .observations
            .iter()
            .cloned()
            .map(|inner| PyPreparedFloatObservation { inner })
            .collect()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppSplitArc", eq)]
#[derive(Clone, PartialEq)]
pub struct PyPppSplitArc {
    inner: core::PppSplitArc,
}
#[pymethods]
impl PyPppSplitArc {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn start_epoch_index(&self) -> usize {
        self.inner.start_epoch_index
    }
    #[getter]
    fn end_epoch_index(&self) -> usize {
        self.inner.end_epoch_index
    }
    #[getter]
    fn n_epochs(&self) -> usize {
        self.inner.n_epochs
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppWideLanePrepResult", eq)]
#[derive(Clone, PartialEq)]
pub struct PyWideLanePrepResult {
    inner: core::WideLanePrepResult,
}
#[pymethods]
impl PyWideLanePrepResult {
    #[getter]
    fn epochs(&self) -> Vec<PyPreparedFloatEpoch> {
        self.inner
            .epochs
            .iter()
            .cloned()
            .map(|inner| PyPreparedFloatEpoch { inner })
            .collect()
    }
    #[getter]
    fn wavelengths_m(&self) -> std::collections::BTreeMap<String, f64> {
        self.inner.wavelengths_m.clone()
    }
    #[getter]
    fn offsets_m(&self) -> std::collections::BTreeMap<String, f64> {
        self.inner.offsets_m.clone()
    }
    #[getter]
    fn wide_lane_cycles(&self) -> std::collections::BTreeMap<String, i64> {
        self.inner.wide_lane_cycles.clone()
    }
    #[getter]
    fn dropped_sats(&self) -> Vec<String> {
        self.inner.dropped_sats.clone()
    }
    #[getter]
    fn split_arcs(&self) -> Vec<PyPppSplitArc> {
        self.inner
            .split_arcs
            .iter()
            .cloned()
            .map(|inner| PyPppSplitArc { inner })
            .collect()
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppFloatCycleSlipObservation",
    eq
)]
#[derive(Clone, PartialEq)]
pub struct PyFloatCycleSlipObservation {
    inner: core::FloatCycleSlipObservation,
}
#[pymethods]
impl PyFloatCycleSlipObservation {
    #[new]
    #[pyo3(signature = (satellite_id, ambiguity_id, raw=None))]
    fn new(
        satellite_id: String,
        ambiguity_id: String,
        raw: Option<&PyDualFrequencyObservation>,
    ) -> Self {
        Self {
            inner: core::FloatCycleSlipObservation {
                satellite_id,
                ambiguity_id,
                raw: raw.map(|v| v.inner.clone()),
            },
        }
    }
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn raw(&self) -> Option<PyDualFrequencyObservation> {
        self.inner
            .raw
            .clone()
            .map(|inner| PyDualFrequencyObservation { inner })
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppFloatCycleSlipEpoch", eq)]
#[derive(Clone, PartialEq)]
pub struct PyFloatCycleSlipEpoch {
    inner: core::FloatCycleSlipEpoch,
}
#[pymethods]
impl PyFloatCycleSlipEpoch {
    #[new]
    #[pyo3(signature = (observations, gap_time_s=None, gap_epoch=None))]
    fn new(
        py: Python<'_>,
        observations: Vec<Py<PyFloatCycleSlipObservation>>,
        gap_time_s: Option<f64>,
        gap_epoch: Option<&PyExactEpoch>,
    ) -> Self {
        Self {
            inner: core::FloatCycleSlipEpoch {
                gap_time_s,
                gap_epoch: gap_epoch.map(|v| v.inner),
                observations: observations
                    .iter()
                    .map(|v| v.borrow(py).inner.clone())
                    .collect(),
            },
        }
    }
    #[getter]
    fn gap_time_s(&self) -> Option<f64> {
        self.inner.gap_time_s
    }
    #[getter]
    fn gap_epoch(&self) -> Option<PyExactEpoch> {
        self.inner.gap_epoch.map(|inner| PyExactEpoch { inner })
    }
    #[getter]
    fn observations(&self) -> Vec<PyFloatCycleSlipObservation> {
        self.inner
            .observations
            .iter()
            .cloned()
            .map(|inner| PyFloatCycleSlipObservation { inner })
            .collect()
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppFloatCycleSlipTaggedObservation",
    eq
)]
#[derive(Clone, PartialEq)]
pub struct PyFloatCycleSlipTaggedObservation {
    inner: core::FloatCycleSlipTaggedObservation,
}
#[pymethods]
impl PyFloatCycleSlipTaggedObservation {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppFloatCycleSlipTaggedEpoch",
    eq
)]
#[derive(Clone, PartialEq)]
pub struct PyFloatCycleSlipTaggedEpoch {
    inner: core::FloatCycleSlipTaggedEpoch,
}
#[pymethods]
impl PyFloatCycleSlipTaggedEpoch {
    #[getter]
    fn observations(&self) -> Vec<PyFloatCycleSlipTaggedObservation> {
        self.inner
            .observations
            .iter()
            .cloned()
            .map(|inner| PyFloatCycleSlipTaggedObservation { inner })
            .collect()
    }
}

fn cycle_slip_policy(value: &str) -> PyResult<core::CycleSlipPolicy> {
    match value {
        "error" => Ok(core::CycleSlipPolicy::Error),
        "drop_satellite" => Ok(core::CycleSlipPolicy::DropSatellite),
        "split_arc" => Ok(core::CycleSlipPolicy::SplitArc),
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "cycle_slip_policy must be 'error', 'drop_satellite', or 'split_arc'",
        )),
    }
}

#[pyfunction]
fn prepare_widelane_fixed_epochs(
    py: Python<'_>,
    epochs: Vec<Py<PyDualFrequencyEpoch>>,
    wide_lane: &PyWideLanePrepOptions,
    cycle_slip_policy_label: &str,
    cycle_slip_options: &PyCycleSlipOptions,
) -> PyResult<PyWideLanePrepResult> {
    let epochs = epochs
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::prepare_widelane_fixed_epochs(
        &epochs,
        wide_lane.inner,
        cycle_slip_policy(cycle_slip_policy_label)?,
        cycle_slip_options.inner,
    )
    .map(|inner| PyWideLanePrepResult { inner })
    .map_err(|e| wide_lane_error(py, e))
}

#[pyfunction]
fn split_float_cycle_slip_epochs(
    py: Python<'_>,
    epochs: Vec<Py<PyFloatCycleSlipEpoch>>,
    cycle_slip_options: &PyCycleSlipOptions,
) -> Vec<PyFloatCycleSlipTaggedEpoch> {
    let epochs = epochs
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::split_float_cycle_slip_epochs(&epochs, cycle_slip_options.inner)
        .into_iter()
        .map(|inner| PyFloatCycleSlipTaggedEpoch { inner })
        .collect()
}

#[pyfunction]
fn melbourne_wubbena_cycles(
    py: Python<'_>,
    observation: &PyDualFrequencyObservation,
) -> PyResult<f64> {
    core::melbourne_wubbena_cycles(&observation.inner).map_err(|e| cycle_error(py, e))
}
#[pyfunction(name = "ppp_geometry_free_m")]
fn geometry_free_m(py: Python<'_>, observation: &PyDualFrequencyObservation) -> PyResult<f64> {
    core::geometry_free_m(&observation.inner).map_err(|e| cycle_error(py, e))
}
#[pyfunction]
fn update_melbourne_wubbena(
    py: Python<'_>,
    state: &mut PySatelliteCycleSlipState,
    observation: &PyDualFrequencyObservation,
    config: &PyCycleSlipConfig,
) -> PyResult<PyMelbourneWubbenaUpdate> {
    core::update_melbourne_wubbena(&mut state.inner, &observation.inner, config.inner)
        .map(|inner| PyMelbourneWubbenaUpdate { inner })
        .map_err(|e| cycle_error(py, e))
}
#[pyfunction]
#[pyo3(signature = (state, observation, config, epoch_time_s=None, epoch=None))]
fn update_geometry_free(
    py: Python<'_>,
    state: &mut PySatelliteCycleSlipState,
    observation: &PyDualFrequencyObservation,
    config: &PyCycleSlipConfig,
    epoch_time_s: Option<f64>,
    epoch: Option<&PyExactEpoch>,
) -> PyResult<PyGeometryFreeUpdate> {
    core::update_geometry_free(
        &mut state.inner,
        &observation.inner,
        epoch_time_s,
        epoch.map(|v| v.inner),
        config.inner,
    )
    .map(|inner| PyGeometryFreeUpdate { inner })
    .map_err(|e| cycle_error(py, e))
}
#[pyfunction(name = "ppp_detect_cycle_slips")]
fn detect_cycle_slips(
    py: Python<'_>,
    epochs: Vec<Py<PyDualFrequencyEpoch>>,
    config: &PyCycleSlipConfig,
) -> PyResult<Vec<PyCycleSlipFlagEpoch>> {
    let epochs = epochs
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::detect_cycle_slips(&epochs, config.inner)
        .map(|v| {
            v.into_iter()
                .map(|inner| PyCycleSlipFlagEpoch { inner })
                .collect()
        })
        .map_err(|e| cycle_error(py, e))
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecConfig", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyTecConfig {
    inner: core::TecConfig,
}
#[pymethods]
impl PyTecConfig {
    #[new]
    #[pyo3(signature = (shell_height_m=350000.0, earth_radius_m=6371008.8))]
    fn new(shell_height_m: f64, earth_radius_m: f64) -> Self {
        let mut inner = core::TecConfig::default();
        inner.shell_height_m = shell_height_m;
        inner.earth_radius_m = earth_radius_m;
        Self { inner }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::TecConfig::default(),
        }
    }
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        self.inner.validate().map_err(|e| tec_error(py, e))
    }
    #[getter]
    fn shell_height_m(&self) -> f64 {
        self.inner.shell_height_m
    }
    #[getter]
    fn earth_radius_m(&self) -> f64 {
        self.inner.earth_radius_m
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecLevelingSample", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyTecLevelingSample {
    inner: core::TecLevelingSample,
}
#[pymethods]
impl PyTecLevelingSample {
    #[new]
    fn new(code_slant_tec_tecu: f64, phase_slant_tec_tecu: f64, elevation_rad: f64) -> Self {
        Self {
            inner: core::TecLevelingSample {
                code_slant_tec_tecu,
                phase_slant_tec_tecu,
                elevation_rad,
            },
        }
    }
    #[getter]
    fn code_slant_tec_tecu(&self) -> f64 {
        self.inner.code_slant_tec_tecu
    }
    #[getter]
    fn phase_slant_tec_tecu(&self) -> f64 {
        self.inner.phase_slant_tec_tecu
    }
    #[getter]
    fn elevation_rad(&self) -> f64 {
        self.inner.elevation_rad
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppLeveledTecSample", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyLeveledTecSample {
    inner: core::LeveledTecSample,
}
#[pymethods]
impl PyLeveledTecSample {
    #[getter]
    fn code_slant_tec_tecu(&self) -> f64 {
        self.inner.code_slant_tec_tecu
    }
    #[getter]
    fn phase_slant_tec_tecu(&self) -> f64 {
        self.inner.phase_slant_tec_tecu
    }
    #[getter]
    fn leveled_slant_tec_tecu(&self) -> f64 {
        self.inner.leveled_slant_tec_tecu
    }
    #[getter]
    fn mapping_function(&self) -> f64 {
        self.inner.mapping_function
    }
    #[getter]
    fn vertical_tec_tecu(&self) -> f64 {
        self.inner.vertical_tec_tecu
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecLevelingResult", eq)]
#[derive(Clone, PartialEq)]
pub struct PyTecLevelingResult {
    inner: core::TecLevelingResult,
}
#[pymethods]
impl PyTecLevelingResult {
    #[getter]
    fn phase_bias_tecu(&self) -> f64 {
        self.inner.phase_bias_tecu
    }
    #[getter]
    fn samples(&self) -> Vec<PyLeveledTecSample> {
        self.inner
            .samples
            .iter()
            .copied()
            .map(|inner| PyLeveledTecSample { inner })
            .collect()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppIonosphericPiercePoint", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyIonosphericPiercePoint {
    inner: core::IonosphericPiercePoint,
}
#[pymethods]
impl PyIonosphericPiercePoint {
    #[getter]
    fn latitude_rad(&self) -> f64 {
        self.inner.latitude_rad
    }
    #[getter]
    fn longitude_rad(&self) -> f64 {
        self.inner.longitude_rad
    }
    #[getter]
    fn latitude_deg(&self) -> f64 {
        self.inner.latitude_deg
    }
    #[getter]
    fn longitude_deg(&self) -> f64 {
        self.inner.longitude_deg
    }
    #[getter]
    fn earth_central_angle_rad(&self) -> f64 {
        self.inner.earth_central_angle_rad
    }
    #[getter]
    fn shell_height_m(&self) -> f64 {
        self.inner.shell_height_m
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppCodeSlantTecEstimate", eq)]
#[derive(Clone, PartialEq)]
pub struct PyCodeSlantTecEstimate {
    inner: core::CodeSlantTecEstimate,
}
#[pymethods]
impl PyCodeSlantTecEstimate {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn code_geometry_free_m(&self) -> f64 {
        self.inner.code_geometry_free_m
    }
    #[getter]
    fn slant_tec_tecu(&self) -> f64 {
        self.inner.slant_tec_tecu
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppPhaseSlantTecEstimate", eq)]
#[derive(Clone, PartialEq)]
pub struct PyPhaseSlantTecEstimate {
    inner: core::PhaseSlantTecEstimate,
}
#[pymethods]
impl PyPhaseSlantTecEstimate {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn phase_geometry_free_m(&self) -> f64 {
        self.inner.phase_geometry_free_m
    }
    #[getter]
    fn slant_tec_tecu(&self) -> f64 {
        self.inner.slant_tec_tecu
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecObservation", eq)]
#[derive(Clone, PartialEq)]
pub struct PyTecObservation {
    inner: core::TecObservation,
}
#[pymethods]
impl PyTecObservation {
    #[new]
    fn new(observation: &PyDualFrequencyObservation, elevation_rad: f64, azimuth_rad: f64) -> Self {
        Self {
            inner: core::TecObservation {
                observation: observation.inner.clone(),
                elevation_rad,
                azimuth_rad,
            },
        }
    }
    #[getter]
    fn observation(&self) -> PyDualFrequencyObservation {
        PyDualFrequencyObservation {
            inner: self.inner.observation.clone(),
        }
    }
    #[getter]
    fn elevation_rad(&self) -> f64 {
        self.inner.elevation_rad
    }
    #[getter]
    fn azimuth_rad(&self) -> f64 {
        self.inner.azimuth_rad
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecEpoch", eq)]
#[derive(Clone, PartialEq)]
pub struct PyTecEpoch {
    inner: core::TecEpoch,
}
#[pymethods]
impl PyTecEpoch {
    #[new]
    fn new(
        py: Python<'_>,
        time_s: f64,
        receiver_latitude_rad: f64,
        receiver_longitude_rad: f64,
        observations: Vec<Py<PyTecObservation>>,
    ) -> Self {
        Self {
            inner: core::TecEpoch {
                time_s,
                receiver_latitude_rad,
                receiver_longitude_rad,
                observations: observations
                    .iter()
                    .map(|v| v.borrow(py).inner.clone())
                    .collect(),
            },
        }
    }
    #[getter]
    fn time_s(&self) -> f64 {
        self.inner.time_s
    }
    #[getter]
    fn receiver_latitude_rad(&self) -> f64 {
        self.inner.receiver_latitude_rad
    }
    #[getter]
    fn receiver_longitude_rad(&self) -> f64 {
        self.inner.receiver_longitude_rad
    }
    #[getter]
    fn observations(&self) -> Vec<PyTecObservation> {
        self.inner
            .observations
            .iter()
            .cloned()
            .map(|inner| PyTecObservation { inner })
            .collect()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecEstimateSample", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyTecEstimateSample {
    inner: core::TecEstimateSample,
}
#[pymethods]
impl PyTecEstimateSample {
    #[getter]
    fn time_s(&self) -> f64 {
        self.inner.time_s
    }
    #[getter]
    fn elevation_rad(&self) -> f64 {
        self.inner.elevation_rad
    }
    #[getter]
    fn azimuth_rad(&self) -> f64 {
        self.inner.azimuth_rad
    }
    #[getter]
    fn code_geometry_free_m(&self) -> f64 {
        self.inner.code_geometry_free_m
    }
    #[getter]
    fn phase_geometry_free_m(&self) -> f64 {
        self.inner.phase_geometry_free_m
    }
    #[getter]
    fn code_slant_tec_tecu(&self) -> f64 {
        self.inner.code_slant_tec_tecu
    }
    #[getter]
    fn phase_slant_tec_tecu(&self) -> f64 {
        self.inner.phase_slant_tec_tecu
    }
    #[getter]
    fn leveled_slant_tec_tecu(&self) -> f64 {
        self.inner.leveled_slant_tec_tecu
    }
    #[getter]
    fn mapping_function(&self) -> f64 {
        self.inner.mapping_function
    }
    #[getter]
    fn vertical_tec_tecu(&self) -> f64 {
        self.inner.vertical_tec_tecu
    }
    #[getter]
    fn pierce_point(&self) -> PyIonosphericPiercePoint {
        PyIonosphericPiercePoint {
            inner: self.inner.pierce_point,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecSatelliteArc", eq)]
#[derive(Clone, PartialEq)]
pub struct PyTecSatelliteArc {
    inner: core::TecSatelliteArc,
}
#[pymethods]
impl PyTecSatelliteArc {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn phase_bias_tecu(&self) -> f64 {
        self.inner.phase_bias_tecu
    }
    #[getter]
    fn samples(&self) -> Vec<PyTecEstimateSample> {
        self.inner
            .samples
            .iter()
            .copied()
            .map(|inner| PyTecEstimateSample { inner })
            .collect()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppTecEstimate", eq)]
#[derive(Clone, PartialEq)]
pub struct PyTecEstimate {
    inner: core::TecEstimate,
}
#[pymethods]
impl PyTecEstimate {
    #[getter]
    fn arcs(&self) -> Vec<PyTecSatelliteArc> {
        self.inner
            .arcs
            .iter()
            .cloned()
            .map(|inner| PyTecSatelliteArc { inner })
            .collect()
    }
}

#[pyfunction]
fn code_geometry_free_m(py: Python<'_>, observation: &PyDualFrequencyObservation) -> PyResult<f64> {
    core::code_geometry_free_m(&observation.inner).map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn phase_geometry_free_m(
    py: Python<'_>,
    observation: &PyDualFrequencyObservation,
) -> PyResult<f64> {
    core::phase_geometry_free_m(&observation.inner).map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn slant_tec_from_code_geometry_free_m(
    py: Python<'_>,
    value: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> PyResult<f64> {
    core::slant_tec_from_code_geometry_free_m(value, f1_hz, f2_hz).map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn slant_tec_from_phase_geometry_free_m(
    py: Python<'_>,
    value: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> PyResult<f64> {
    core::slant_tec_from_phase_geometry_free_m(value, f1_hz, f2_hz).map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn estimate_code_slant_tec(
    py: Python<'_>,
    observation: &PyDualFrequencyObservation,
) -> PyResult<PyCodeSlantTecEstimate> {
    core::estimate_code_slant_tec(&observation.inner)
        .map(|inner| PyCodeSlantTecEstimate { inner })
        .map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn estimate_phase_slant_tec(
    py: Python<'_>,
    observation: &PyDualFrequencyObservation,
) -> PyResult<PyPhaseSlantTecEstimate> {
    core::estimate_phase_slant_tec(&observation.inner)
        .map(|inner| PyPhaseSlantTecEstimate { inner })
        .map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn thin_shell_mapping_function(
    py: Python<'_>,
    elevation_rad: f64,
    config: &PyTecConfig,
) -> PyResult<f64> {
    core::thin_shell_mapping_function(elevation_rad, config.inner).map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn vertical_tec_from_slant_tec(
    py: Python<'_>,
    slant_tec_tecu: f64,
    elevation_rad: f64,
    config: &PyTecConfig,
) -> PyResult<f64> {
    core::vertical_tec_from_slant_tec(slant_tec_tecu, elevation_rad, config.inner)
        .map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn level_slant_tec_arc(
    py: Python<'_>,
    samples: Vec<Py<PyTecLevelingSample>>,
    config: &PyTecConfig,
) -> PyResult<PyTecLevelingResult> {
    let samples = samples
        .iter()
        .map(|v| v.borrow(py).inner)
        .collect::<Vec<_>>();
    core::level_slant_tec_arc(&samples, config.inner)
        .map(|inner| PyTecLevelingResult { inner })
        .map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn ionospheric_pierce_point(
    py: Python<'_>,
    receiver_latitude_rad: f64,
    receiver_longitude_rad: f64,
    elevation_rad: f64,
    azimuth_rad: f64,
    config: &PyTecConfig,
) -> PyResult<PyIonosphericPiercePoint> {
    core::ionospheric_pierce_point(
        receiver_latitude_rad,
        receiver_longitude_rad,
        elevation_rad,
        azimuth_rad,
        config.inner,
    )
    .map(|inner| PyIonosphericPiercePoint { inner })
    .map_err(|e| tec_error(py, e))
}
#[pyfunction]
fn estimate_tec(
    py: Python<'_>,
    epochs: Vec<Py<PyTecEpoch>>,
    config: &PyTecConfig,
) -> PyResult<PyTecEstimate> {
    let epochs = epochs
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::estimate_tec(&epochs, config.inner)
        .map(|inner| PyTecEstimate { inner })
        .map_err(|e| tec_error(py, e))
}

#[pyclass(module = "sidereon._sidereon", name = "PppRaimConfig", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyRaimConfig {
    inner: core::RaimConfig,
}
#[pymethods]
impl PyRaimConfig {
    #[new]
    #[pyo3(signature = (false_alarm_probability=1.0e-3, missed_detection_probability=1.0e-3, measurement_sigma_m=1.0, chi_square_threshold=None))]
    fn new(
        false_alarm_probability: f64,
        missed_detection_probability: f64,
        measurement_sigma_m: f64,
        chi_square_threshold: Option<f64>,
    ) -> Self {
        let mut inner = core::RaimConfig::default();
        inner.false_alarm_probability = false_alarm_probability;
        inner.missed_detection_probability = missed_detection_probability;
        inner.measurement_sigma_m = measurement_sigma_m;
        inner.chi_square_threshold = chi_square_threshold;
        Self { inner }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::RaimConfig::default(),
        }
    }
    #[getter]
    fn false_alarm_probability(&self) -> f64 {
        self.inner.false_alarm_probability
    }
    #[getter]
    fn missed_detection_probability(&self) -> f64 {
        self.inner.missed_detection_probability
    }
    #[getter]
    fn measurement_sigma_m(&self) -> f64 {
        self.inner.measurement_sigma_m
    }
    #[getter]
    fn chi_square_threshold(&self) -> Option<f64> {
        self.inner.chi_square_threshold
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppRaimStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyRaimStatus {
    PASSED,
    FAULT_DETECTED,
    NOT_ENOUGH_REDUNDANCY,
}
impl From<core::RaimStatus> for PyRaimStatus {
    fn from(value: core::RaimStatus) -> Self {
        match value {
            core::RaimStatus::Passed => Self::PASSED,
            core::RaimStatus::FaultDetected => Self::FAULT_DETECTED,
            core::RaimStatus::NotEnoughRedundancy => Self::NOT_ENOUGH_REDUNDANCY,
        }
    }
}
#[pymethods]
impl PyRaimStatus {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::PASSED => "passed",
            Self::FAULT_DETECTED => "fault_detected",
            Self::NOT_ENOUGH_REDUNDANCY => "not_enough_redundancy",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppSatelliteTestStatistic", eq)]
#[derive(Clone, PartialEq)]
pub struct PySatelliteTestStatistic {
    inner: core::SatelliteTestStatistic,
}
#[pymethods]
impl PySatelliteTestStatistic {
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn code(&self) -> f64 {
        self.inner.code
    }
    #[getter]
    fn phase(&self) -> f64 {
        self.inner.phase
    }
    #[getter]
    fn statistic(&self) -> f64 {
        self.inner.statistic
    }
    #[getter]
    fn code_testable(&self) -> bool {
        self.inner.code_testable
    }
    #[getter]
    fn phase_testable(&self) -> bool {
        self.inner.phase_testable
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppRaimResult", eq)]
#[derive(Clone, PartialEq)]
pub struct PyRaimResult {
    inner: core::RaimResult,
}
#[pymethods]
impl PyRaimResult {
    #[getter]
    fn status(&self) -> PyRaimStatus {
        self.inner.status.into()
    }
    #[getter]
    fn detected(&self) -> bool {
        self.inner.detected
    }
    #[getter]
    fn test_statistic(&self) -> f64 {
        self.inner.test_statistic
    }
    #[getter]
    fn threshold(&self) -> Option<f64> {
        self.inner.threshold
    }
    #[getter]
    fn redundancy(&self) -> isize {
        self.inner.redundancy
    }
    #[getter]
    fn satellite_statistics(&self) -> Vec<PySatelliteTestStatistic> {
        self.inner
            .satellite_statistics
            .iter()
            .cloned()
            .map(|inner| PySatelliteTestStatistic { inner })
            .collect()
    }
    #[getter]
    fn most_likely_fault(&self) -> Option<String> {
        self.inner.most_likely_fault.clone()
    }
    #[getter]
    fn hpl_m(&self) -> Option<f64> {
        self.inner.hpl_m
    }
    #[getter]
    fn vpl_m(&self) -> Option<f64> {
        self.inner.vpl_m
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppRaimGeometryRow", eq)]
#[derive(Clone, PartialEq)]
pub struct PyRaimGeometryRow {
    inner: core::RaimGeometryRow,
}
#[pymethods]
impl PyRaimGeometryRow {
    #[new]
    fn new(satellite_id: String, line_of_sight: [f64; 3]) -> Self {
        Self {
            inner: core::RaimGeometryRow {
                satellite_id,
                line_of_sight: sidereon_core::dop::LineOfSight::new(
                    line_of_sight[0],
                    line_of_sight[1],
                    line_of_sight[2],
                ),
            },
        }
    }
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn line_of_sight(&self) -> [f64; 3] {
        [
            self.inner.line_of_sight.e_x,
            self.inner.line_of_sight.e_y,
            self.inner.line_of_sight.e_z,
        ]
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppRaimIdentification", eq)]
#[derive(Clone, PartialEq)]
pub struct PyRaimIdentification {
    inner: core::RaimIdentification,
}
#[pymethods]
impl PyRaimIdentification {
    #[getter]
    fn statistics(&self) -> Vec<PySatelliteTestStatistic> {
        self.inner
            .statistics
            .iter()
            .cloned()
            .map(|inner| PySatelliteTestStatistic { inner })
            .collect()
    }
    #[getter]
    fn most_likely_fault(&self) -> Option<String> {
        self.inner.most_likely_fault.clone()
    }
}

#[pyfunction]
fn global_test(
    py: Python<'_>,
    residuals: Vec<Py<PyPppFloatResidual>>,
    n_states: usize,
    config: &PyRaimConfig,
) -> PyResult<PyRaimResult> {
    let residuals = residuals
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::raim::global_test(&residuals, n_states, config.inner)
        .map(|inner| PyRaimResult { inner })
        .map_err(|e| raim_error(py, e))
}
#[pyfunction]
fn global_test_with_geometry(
    py: Python<'_>,
    residuals: Vec<Py<PyPppFloatResidual>>,
    geometry: Vec<Py<PyRaimGeometryRow>>,
    config: &PyRaimConfig,
) -> PyResult<PyRaimResult> {
    let residuals = residuals
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    let geometry = geometry
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::raim::global_test_with_geometry(&residuals, &geometry, config.inner)
        .map(|inner| PyRaimResult { inner })
        .map_err(|e| raim_error(py, e))
}
#[pyfunction]
fn per_satellite_statistics(
    py: Python<'_>,
    residuals: Vec<Py<PyPppFloatResidual>>,
    geometry: Vec<Py<PyRaimGeometryRow>>,
    config: &PyRaimConfig,
) -> PyResult<PyRaimIdentification> {
    let residuals = residuals
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    let geometry = geometry
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::raim::per_satellite_statistics(&residuals, &geometry, config.inner)
        .map(|inner| PyRaimIdentification { inner })
        .map_err(|e| raim_error(py, e))
}

#[pyclass(module = "sidereon._sidereon", name = "PppProtectionLevels", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyProtectionLevels {
    inner: core::ProtectionLevels,
}
#[pymethods]
impl PyProtectionLevels {
    #[getter]
    fn hpl_m(&self) -> f64 {
        self.inner.hpl_m
    }
    #[getter]
    fn vpl_m(&self) -> f64 {
        self.inner.vpl_m
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppRaimFdeStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyRaimFdeStatus {
    CLEAN,
    RESTORED,
    CANNOT_EXCLUDE,
    INTEGRITY_NOT_RESTORED,
    UNTESTED,
}
impl From<core::RaimFdeStatus> for PyRaimFdeStatus {
    fn from(value: core::RaimFdeStatus) -> Self {
        match value {
            core::RaimFdeStatus::Clean => Self::CLEAN,
            core::RaimFdeStatus::Untested => Self::UNTESTED,
            core::RaimFdeStatus::Restored => Self::RESTORED,
            core::RaimFdeStatus::CannotExclude => Self::CANNOT_EXCLUDE,
            core::RaimFdeStatus::IntegrityNotRestored => Self::INTEGRITY_NOT_RESTORED,
        }
    }
}
#[pymethods]
impl PyRaimFdeStatus {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::CLEAN => "clean",
            Self::UNTESTED => "untested",
            Self::RESTORED => "restored",
            Self::CANNOT_EXCLUDE => "cannot_exclude",
            Self::INTEGRITY_NOT_RESTORED => "integrity_not_restored",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppRaimFdeResult", eq)]
#[derive(PartialEq)]
pub struct PyRaimFdeResult {
    inner: core::RaimFdeResult,
}
#[pymethods]
impl PyRaimFdeResult {
    #[getter]
    fn solution(&self) -> PyPppFloatSolution {
        PyPppFloatSolution {
            inner: self.inner.solution.clone(),
        }
    }
    #[getter]
    fn raim(&self) -> PyRaimResult {
        PyRaimResult {
            inner: self.inner.raim.clone(),
        }
    }
    #[getter]
    fn excluded_sats(&self) -> Vec<String> {
        self.inner.excluded_sats.clone()
    }
    #[getter]
    fn status(&self) -> PyRaimFdeStatus {
        self.inner.status.into()
    }
}

/// Brown snapshot-RAIM fault protection levels; requires positive redundancy.
#[pyfunction]
fn protection_levels(
    py: Python<'_>,
    geometry: Vec<Py<PyRaimGeometryRow>>,
    receiver: &PyWgs84Geodetic,
    config: &PyRaimConfig,
) -> PyResult<PyProtectionLevels> {
    let geometry = geometry
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    let receiver = sidereon_core::frame::Wgs84Geodetic::try_from(receiver)?;
    core::raim::protection_levels(&geometry, receiver, config.inner)
        .map(|inner| PyProtectionLevels { inner })
        .map_err(|e| raim_error(py, e))
}

/// Brown snapshot-RAIM fault protection levels using inverse-standard-deviation weights.
#[pyfunction]
fn protection_levels_with_weights(
    py: Python<'_>,
    geometry: Vec<Py<PyRaimGeometryRow>>,
    weights: Vec<f64>,
    receiver: &PyWgs84Geodetic,
    config: &PyRaimConfig,
) -> PyResult<PyProtectionLevels> {
    let geometry = geometry
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    let receiver = sidereon_core::frame::Wgs84Geodetic::try_from(receiver)?;
    core::raim::protection_levels_with_weights(&geometry, &weights, receiver, config.inner)
        .map(|inner| PyProtectionLevels { inner })
        .map_err(|e| raim_error(py, e))
}

/// Fault-free K * sigma * DOP accuracy bounds using inverse-standard-deviation weights.
#[pyfunction]
fn fault_free_accuracy_bounds(
    py: Python<'_>,
    geometry: Vec<Py<PyRaimGeometryRow>>,
    weights: Vec<f64>,
    receiver: &PyWgs84Geodetic,
    config: &PyRaimConfig,
) -> PyResult<PyProtectionLevels> {
    let geometry = geometry
        .iter()
        .map(|v| v.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    let receiver = sidereon_core::frame::Wgs84Geodetic::try_from(receiver)?;
    core::raim::fault_free_accuracy_bounds(&geometry, &weights, receiver, config.inner)
        .map(|inner| PyProtectionLevels { inner })
        .map_err(|e| raim_error(py, e))
}

fn run_fde(
    py: Python<'_>,
    source: &PySp3,
    epoch: &PyPppEpoch,
    initial_state: &PyPppFloatState,
    solve_config: &PyPppFloatConfig,
    raim_config: &PyRaimConfig,
    solve: bool,
) -> PyResult<PyRaimFdeResult> {
    let result = if solve {
        core::solve_float_epoch_with_raim(
            &source.inner,
            epoch.inner.clone(),
            initial_state.inner.clone(),
            solve_config.inner.clone(),
            raim_config.inner,
        )
    } else {
        core::raim::fde_float_epoch(
            &source.inner,
            epoch.inner.clone(),
            initial_state.inner.clone(),
            solve_config.inner.clone(),
            raim_config.inner,
        )
    };
    result
        .map(|inner| PyRaimFdeResult { inner })
        .map_err(|e| raim_fde_error(py, e))
}

#[pyfunction]
fn fde_float_epoch(
    py: Python<'_>,
    source: &PySp3,
    epoch: &PyPppEpoch,
    initial_state: &PyPppFloatState,
    solve_config: &PyPppFloatConfig,
    raim_config: &PyRaimConfig,
) -> PyResult<PyRaimFdeResult> {
    run_fde(
        py,
        source,
        epoch,
        initial_state,
        solve_config,
        raim_config,
        false,
    )
}

#[pyfunction]
fn solve_float_epoch_with_raim(
    py: Python<'_>,
    source: &PySp3,
    epoch: &PyPppEpoch,
    initial_state: &PyPppFloatState,
    solve_config: &PyPppFloatConfig,
    raim_config: &PyRaimConfig,
) -> PyResult<PyRaimFdeResult> {
    run_fde(
        py,
        source,
        epoch,
        initial_state,
        solve_config,
        raim_config,
        true,
    )
}

#[pyclass(module = "sidereon._sidereon", name = "PppKinematicState", eq)]
#[derive(Clone, PartialEq)]
pub struct PyKinematicState {
    inner: core::KinematicState,
}
#[pymethods]
impl PyKinematicState {
    #[new]
    #[pyo3(signature = (position_m=[0.0, 0.0, 0.0], clock_m=0.0, ztd_residual_m=0.0, ambiguities_m=None))]
    fn new(
        position_m: [f64; 3],
        clock_m: f64,
        ztd_residual_m: f64,
        ambiguities_m: Option<std::collections::BTreeMap<String, f64>>,
    ) -> Self {
        Self {
            inner: core::KinematicState {
                position_m,
                clock_m,
                ztd_residual_m,
                ambiguities_m: ambiguities_m.unwrap_or_default(),
            },
        }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::KinematicState::default(),
        }
    }
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }
    #[getter]
    fn clock_m(&self) -> f64 {
        self.inner.clock_m
    }
    #[getter]
    fn ztd_residual_m(&self) -> f64 {
        self.inner.ztd_residual_m
    }
    #[getter]
    fn ambiguities_m(&self) -> std::collections::BTreeMap<String, f64> {
        self.inner.ambiguities_m.clone()
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppKinematicPositionProcessNoise",
    eq
)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyKinematicPositionProcessNoise {
    inner: core::KinematicPositionProcessNoise,
}
#[pymethods]
impl PyKinematicPositionProcessNoise {
    #[staticmethod]
    fn random_walk(spectral_density_m2_s: f64) -> Self {
        Self {
            inner: core::KinematicPositionProcessNoise::RandomWalk {
                spectral_density_m2_s,
            },
        }
    }
    #[staticmethod]
    fn white_noise_acceleration(spectral_density_m2_s3: f64) -> Self {
        Self {
            inner: core::KinematicPositionProcessNoise::WhiteNoiseAcceleration {
                spectral_density_m2_s3,
            },
        }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::KinematicPositionProcessNoise::default(),
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            core::KinematicPositionProcessNoise::RandomWalk { .. } => "random_walk",
            core::KinematicPositionProcessNoise::WhiteNoiseAcceleration { .. } => {
                "white_noise_acceleration"
            }
        }
    }
    #[getter]
    fn spectral_density_m2_s(&self) -> Option<f64> {
        match self.inner {
            core::KinematicPositionProcessNoise::RandomWalk {
                spectral_density_m2_s,
            } => Some(spectral_density_m2_s),
            _ => None,
        }
    }
    #[getter]
    fn spectral_density_m2_s3(&self) -> Option<f64> {
        match self.inner {
            core::KinematicPositionProcessNoise::WhiteNoiseAcceleration {
                spectral_density_m2_s3,
            } => Some(spectral_density_m2_s3),
            _ => None,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppKinematicMotionModel", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyKinematicMotionModel {
    inner: core::KinematicMotionModel,
}
#[pymethods]
impl PyKinematicMotionModel {
    #[staticmethod]
    fn hold() -> Self {
        Self {
            inner: core::KinematicMotionModel::Hold,
        }
    }
    #[staticmethod]
    fn constant_velocity(velocity_m_s: [f64; 3]) -> Self {
        Self {
            inner: core::KinematicMotionModel::ConstantVelocity { velocity_m_s },
        }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::KinematicMotionModel::default(),
        }
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            core::KinematicMotionModel::Hold => "hold",
            core::KinematicMotionModel::ConstantVelocity { .. } => "constant_velocity",
        }
    }
    #[getter]
    fn velocity_m_s(&self) -> Option<[f64; 3]> {
        match self.inner {
            core::KinematicMotionModel::ConstantVelocity { velocity_m_s } => Some(velocity_m_s),
            _ => None,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppKinematicProcessNoise", eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PyKinematicProcessNoise {
    inner: core::KinematicProcessNoise,
}
#[pymethods]
impl PyKinematicProcessNoise {
    #[new]
    fn new(
        position: &PyKinematicPositionProcessNoise,
        clock_white_m2_s: f64,
        ztd_random_walk_m2_s: f64,
        ambiguity_hold_m2_s: f64,
    ) -> Self {
        Self {
            inner: core::KinematicProcessNoise {
                position: position.inner,
                clock_white_m2_s,
                ztd_random_walk_m2_s,
                ambiguity_hold_m2_s,
            },
        }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::KinematicProcessNoise::default(),
        }
    }
    #[getter]
    fn position(&self) -> PyKinematicPositionProcessNoise {
        PyKinematicPositionProcessNoise {
            inner: self.inner.position,
        }
    }
    #[getter]
    fn clock_white_m2_s(&self) -> f64 {
        self.inner.clock_white_m2_s
    }
    #[getter]
    fn ztd_random_walk_m2_s(&self) -> f64 {
        self.inner.ztd_random_walk_m2_s
    }
    #[getter]
    fn ambiguity_hold_m2_s(&self) -> f64 {
        self.inner.ambiguity_hold_m2_s
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppKinematicConfig", eq)]
#[derive(Clone, PartialEq)]
pub struct PyKinematicConfig {
    inner: core::KinematicConfig,
}
#[pymethods]
impl PyKinematicConfig {
    #[new]
    #[pyo3(signature = (initial_state=None, initial_covariance_m2=None, motion=None, process_noise=None, new_ambiguity_variance_m2=None, weights=None, tropo=None, corrections=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        initial_state: Option<&PyKinematicState>,
        initial_covariance_m2: Option<Vec<Vec<f64>>>,
        motion: Option<&PyKinematicMotionModel>,
        process_noise: Option<&PyKinematicProcessNoise>,
        new_ambiguity_variance_m2: Option<f64>,
        weights: Option<&PyPppMeasurementWeights>,
        tropo: Option<&PyPppTroposphereOptions>,
        corrections: Option<&PyPppRangeCorrections>,
    ) -> Self {
        let mut inner = core::KinematicConfig::default();
        if let Some(v) = initial_state {
            inner.initial_state = v.inner.clone();
        }
        if let Some(v) = initial_covariance_m2 {
            inner.initial_covariance_m2 = v;
        }
        if let Some(v) = motion {
            inner.motion = v.inner;
        }
        if let Some(v) = process_noise {
            inner.process_noise = v.inner;
        }
        if let Some(v) = new_ambiguity_variance_m2 {
            inner.new_ambiguity_variance_m2 = v;
        }
        if let Some(v) = weights {
            inner.weights = v.inner;
        }
        if let Some(v) = tropo {
            inner.tropo = v.inner;
        }
        if let Some(v) = corrections {
            inner.corrections = v.inner.clone();
        }
        Self { inner }
    }
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: core::KinematicConfig::default(),
        }
    }
    #[getter]
    fn initial_state(&self) -> PyKinematicState {
        PyKinematicState {
            inner: self.inner.initial_state.clone(),
        }
    }
    #[getter]
    fn initial_covariance_m2(&self) -> Vec<Vec<f64>> {
        self.inner.initial_covariance_m2.clone()
    }
    #[getter]
    fn motion(&self) -> PyKinematicMotionModel {
        PyKinematicMotionModel {
            inner: self.inner.motion,
        }
    }
    #[getter]
    fn process_noise(&self) -> PyKinematicProcessNoise {
        PyKinematicProcessNoise {
            inner: self.inner.process_noise,
        }
    }
    #[getter]
    fn new_ambiguity_variance_m2(&self) -> f64 {
        self.inner.new_ambiguity_variance_m2
    }
    #[getter]
    fn weights(&self) -> PyPppMeasurementWeights {
        PyPppMeasurementWeights {
            inner: self.inner.weights,
        }
    }
    #[getter]
    fn tropo(&self) -> PyPppTroposphereOptions {
        PyPppTroposphereOptions {
            inner: self.inner.tropo,
        }
    }
    #[getter]
    fn corrections(&self) -> PyPppRangeCorrections {
        PyPppRangeCorrections {
            inner: self.inner.corrections.clone(),
        }
    }
}

fn kinematic_error(py: Python<'_>, error: core::KinematicSolveError) -> PyErr {
    use core::KinematicSolveError as E;
    let message = error.to_string();
    match error {
        E::NoEphemeris {
            satellite_id,
            reason,
        } => {
            let (reason_kind, reason) = match reason {
                core::NoEphemerisReason::NoEphemeris => (
                    "no_ephemeris",
                    "no ephemeris product covers the epoch".to_string(),
                ),
                core::NoEphemerisReason::MissingSatelliteClock => (
                    "missing_satellite_clock",
                    "satellite clock is unavailable".to_string(),
                ),
                core::NoEphemerisReason::Reason(reason) => ("reason", reason),
            };
            detailed_error(
                py,
                "KinematicSolveError",
                "no_ephemeris",
                message,
                &[
                    ("satellite_id", satellite_id),
                    ("reason_kind", reason_kind.to_string()),
                    ("reason", reason),
                ],
            )
        }
        E::SingularGeometry => {
            detailed_error(py, "KinematicSolveError", "singular_geometry", message, &[])
        }
        E::InvalidSolveOption { field, reason } => detailed_error(
            py,
            "KinematicSolveError",
            "invalid_solve_option",
            message,
            &[("field", field.to_string()), ("reason", reason.to_string())],
        ),
        E::InvalidInput { field, reason } => detailed_error(
            py,
            "KinematicSolveError",
            "invalid_input",
            message,
            &[("field", field.to_string()), ("reason", reason.to_string())],
        ),
        E::MissingCorrection {
            satellite_id,
            correction,
        } => detailed_error(
            py,
            "KinematicSolveError",
            "missing_correction",
            message,
            &[
                ("satellite_id", satellite_id),
                ("correction", correction.to_string()),
            ],
        ),
        E::Ut1OutsideCoverage(reason) => detailed_error(
            py,
            "KinematicSolveError",
            "ut1_outside_coverage",
            message,
            &[("reason", crate::degrade_reason_label(reason).to_string())],
        ),
    }
}

#[pyfunction]
fn predict_kinematic_state(
    state: &mut PyKinematicState,
    covariance_m2: Vec<Vec<f64>>,
    dt_s: f64,
    active_ambiguity_ids: Vec<String>,
    config: &PyKinematicConfig,
    _py: Python<'_>,
) -> PyResult<Vec<Vec<f64>>> {
    let mut covariance_m2 = covariance_m2;
    core::predict_kinematic_state(
        &mut state.inner,
        &mut covariance_m2,
        dt_s,
        &active_ambiguity_ids,
        &config.inner,
    )
    .map_err(|error| kinematic_error(_py, error))?;
    Ok(covariance_m2)
}

#[pyclass(module = "sidereon._sidereon", name = "PppKinematicUpdateSummary", eq)]
#[derive(Clone, PartialEq)]
pub struct PyKinematicUpdateSummary {
    inner: core::KinematicUpdateSummary,
}

#[pymethods]
impl PyKinematicUpdateSummary {
    #[getter]
    fn innovation_rms_m(&self) -> f64 {
        self.inner.innovation_rms_m
    }
    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner.used_sats.clone()
    }
    #[getter]
    fn ssr_bias_exclusions(&self) -> Vec<PyPppSsrBiasExclusion> {
        self.inner
            .ssr_bias_exclusions
            .iter()
            .cloned()
            .map(|inner| PyPppSsrBiasExclusion { inner })
            .collect()
    }
    #[getter]
    fn unplaced_observations(&self) -> Vec<PppUnplacedObservationRow> {
        self.inner
            .unplaced_observations
            .iter()
            .cloned()
            .map(|inner| PppUnplacedObservationRow { inner })
            .collect()
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "PppKinematicEpochStatus",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyKinematicEpochStatus {
    UPDATED,
    PREDICTED_ONLY,
}

impl From<core::KinematicEpochStatus> for PyKinematicEpochStatus {
    fn from(value: core::KinematicEpochStatus) -> Self {
        match value {
            core::KinematicEpochStatus::Updated => Self::UPDATED,
            core::KinematicEpochStatus::PredictedOnly => Self::PREDICTED_ONLY,
        }
    }
}

#[pymethods]
impl PyKinematicEpochStatus {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::UPDATED => "updated",
            Self::PREDICTED_ONLY => "predicted_only",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppKinematicEpochSolution", eq)]
#[derive(Clone, PartialEq)]
pub struct PyKinematicEpochSolution {
    inner: core::KinematicEpochSolution,
}

#[pymethods]
impl PyKinematicEpochSolution {
    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }
    #[getter]
    fn clock_m(&self) -> f64 {
        self.inner.clock_m
    }
    #[getter]
    fn ztd_residual_m(&self) -> f64 {
        self.inner.ztd_residual_m
    }
    #[getter]
    fn ambiguities_m(&self) -> std::collections::BTreeMap<String, f64> {
        self.inner.ambiguities_m.clone()
    }
    #[getter]
    fn position_covariance_m2(&self) -> [[f64; 3]; 3] {
        self.inner.position_covariance_m2
    }
    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner.used_sats.clone()
    }
    #[getter]
    fn innovation_rms_m(&self) -> f64 {
        self.inner.innovation_rms_m
    }
    #[getter]
    fn ssr_bias_exclusions(&self) -> Vec<PyPppSsrBiasExclusion> {
        self.inner
            .ssr_bias_exclusions
            .iter()
            .cloned()
            .map(|inner| PyPppSsrBiasExclusion { inner })
            .collect()
    }
    #[getter]
    fn unplaced_observations(&self) -> Vec<PppUnplacedObservationRow> {
        self.inner
            .unplaced_observations
            .iter()
            .cloned()
            .map(|inner| PppUnplacedObservationRow { inner })
            .collect()
    }
    #[getter]
    fn status(&self) -> PyKinematicEpochStatus {
        self.inner.status.into()
    }
}

#[pyfunction]
fn correct_kinematic_state(
    py: Python<'_>,
    source: &PySp3,
    epoch_index: usize,
    epoch: &PyPppEpoch,
    state: &mut PyKinematicState,
    covariance_m2: Vec<Vec<f64>>,
    config: &PyKinematicConfig,
) -> PyResult<(PyKinematicUpdateSummary, Vec<Vec<f64>>)> {
    let mut covariance_m2 = covariance_m2;
    let inner = core::correct_kinematic_state(
        &source.inner,
        epoch_index,
        &epoch.inner,
        &mut state.inner,
        &mut covariance_m2,
        &config.inner,
    )
    .map_err(|error| kinematic_error(py, error))?;
    Ok((PyKinematicUpdateSummary { inner }, covariance_m2))
}

#[pyfunction]
fn solve_kinematic_ppp(
    py: Python<'_>,
    source: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    config: &PyKinematicConfig,
) -> PyResult<Vec<PyKinematicEpochSolution>> {
    let epochs = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    core::solve_kinematic_ppp(&source.inner, &epochs, config.inner.clone())
        .map(|solutions| {
            solutions
                .into_iter()
                .map(|inner| PyKinematicEpochSolution { inner })
                .collect()
        })
        .map_err(|error| kinematic_error(py, error))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDualFrequencyObservation>()?;
    module.add_class::<PyDualFrequencyEpoch>()?;
    module.add_class::<PyCycleSlipConfig>()?;
    module.add_class::<PyRunningMeanVariance>()?;
    module.add_class::<PySatelliteCycleSlipState>()?;
    module.add_class::<PyMelbourneWubbenaUpdate>()?;
    module.add_class::<PyGeometryFreeUpdate>()?;
    module.add_class::<PyCycleSlipFlagObservation>()?;
    module.add_class::<PyCycleSlipFlagEpoch>()?;
    module.add_class::<PyWideLanePrepOptions>()?;
    module.add_class::<PyPreparedFloatObservation>()?;
    module.add_class::<PyPreparedFloatEpoch>()?;
    module.add_class::<PyPppSplitArc>()?;
    module.add_class::<PyWideLanePrepResult>()?;
    module.add_class::<PyFloatCycleSlipObservation>()?;
    module.add_class::<PyFloatCycleSlipEpoch>()?;
    module.add_class::<PyFloatCycleSlipTaggedObservation>()?;
    module.add_class::<PyFloatCycleSlipTaggedEpoch>()?;
    module.add_class::<PyTecConfig>()?;
    module.add_class::<PyTecLevelingSample>()?;
    module.add_class::<PyLeveledTecSample>()?;
    module.add_class::<PyTecLevelingResult>()?;
    module.add_class::<PyIonosphericPiercePoint>()?;
    module.add_class::<PyCodeSlantTecEstimate>()?;
    module.add_class::<PyPhaseSlantTecEstimate>()?;
    module.add_class::<PyTecObservation>()?;
    module.add_class::<PyTecEpoch>()?;
    module.add_class::<PyTecEstimateSample>()?;
    module.add_class::<PyTecSatelliteArc>()?;
    module.add_class::<PyTecEstimate>()?;
    module.add_class::<PyRaimConfig>()?;
    module.add_class::<PyRaimStatus>()?;
    module.add_class::<PySatelliteTestStatistic>()?;
    module.add_class::<PyRaimResult>()?;
    module.add_class::<PyRaimGeometryRow>()?;
    module.add_class::<PyRaimIdentification>()?;
    module.add_class::<PyProtectionLevels>()?;
    module.add_class::<PyRaimFdeStatus>()?;
    module.add_class::<PyRaimFdeResult>()?;
    module.add_class::<PyKinematicState>()?;
    module.add_class::<PyKinematicPositionProcessNoise>()?;
    module.add_class::<PyKinematicMotionModel>()?;
    module.add_class::<PyKinematicProcessNoise>()?;
    module.add_class::<PyKinematicConfig>()?;
    module.add_class::<PyKinematicUpdateSummary>()?;
    module.add_class::<PyKinematicEpochStatus>()?;
    module.add_class::<PyKinematicEpochSolution>()?;
    module.add_function(wrap_pyfunction!(melbourne_wubbena_cycles, module)?)?;
    module.add_function(wrap_pyfunction!(geometry_free_m, module)?)?;
    module.add_function(wrap_pyfunction!(update_melbourne_wubbena, module)?)?;
    module.add_function(wrap_pyfunction!(update_geometry_free, module)?)?;
    module.add_function(wrap_pyfunction!(detect_cycle_slips, module)?)?;
    module.add_function(wrap_pyfunction!(prepare_widelane_fixed_epochs, module)?)?;
    module.add_function(wrap_pyfunction!(split_float_cycle_slip_epochs, module)?)?;
    module.add_function(wrap_pyfunction!(code_geometry_free_m, module)?)?;
    module.add_function(wrap_pyfunction!(phase_geometry_free_m, module)?)?;
    module.add_function(wrap_pyfunction!(
        slant_tec_from_code_geometry_free_m,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        slant_tec_from_phase_geometry_free_m,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(estimate_code_slant_tec, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_phase_slant_tec, module)?)?;
    module.add_function(wrap_pyfunction!(thin_shell_mapping_function, module)?)?;
    module.add_function(wrap_pyfunction!(vertical_tec_from_slant_tec, module)?)?;
    module.add_function(wrap_pyfunction!(level_slant_tec_arc, module)?)?;
    module.add_function(wrap_pyfunction!(ionospheric_pierce_point, module)?)?;
    module.add_function(wrap_pyfunction!(estimate_tec, module)?)?;
    module.add_function(wrap_pyfunction!(global_test, module)?)?;
    module.add_function(wrap_pyfunction!(global_test_with_geometry, module)?)?;
    module.add_function(wrap_pyfunction!(per_satellite_statistics, module)?)?;
    module.add_function(wrap_pyfunction!(protection_levels, module)?)?;
    module.add_function(wrap_pyfunction!(protection_levels_with_weights, module)?)?;
    module.add_function(wrap_pyfunction!(fault_free_accuracy_bounds, module)?)?;
    module.add_function(wrap_pyfunction!(fde_float_epoch, module)?)?;
    module.add_function(wrap_pyfunction!(solve_float_epoch_with_raim, module)?)?;
    module.add_function(wrap_pyfunction!(predict_kinematic_state, module)?)?;
    module.add_function(wrap_pyfunction!(correct_kinematic_state, module)?)?;
    module.add_function(wrap_pyfunction!(solve_kinematic_ppp, module)?)?;
    module.add(
        "DEFAULT_IONOSPHERIC_SHELL_HEIGHT_M",
        core::DEFAULT_IONOSPHERIC_SHELL_HEIGHT_M,
    )?;
    module.add("ELECTRONS_PER_TECU_M2", core::ELECTRONS_PER_TECU_M2)?;
    module.add(
        "TEC_GROUP_DELAY_COEFFICIENT",
        core::TEC_GROUP_DELAY_COEFFICIENT,
    )?;
    module.add(
        "DEFAULT_MINIMUM_ARC_LENGTH",
        core::DEFAULT_MINIMUM_ARC_LENGTH,
    )?;
    module.add(
        "DEFAULT_RUNNING_STATISTIC_K_FACTOR",
        core::DEFAULT_RUNNING_STATISTIC_K_FACTOR,
    )?;
    Ok(())
}
