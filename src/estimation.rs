//! Estimation and detection primitive bindings.
//!
//! Thin wrappers over `sidereon_core::estimation::primitives`: alpha-beta
//! filtering, scalar Kalman gains, normalized-innovation gates, MAD spread,
//! EWMA, and CA-CFAR utilities. The binding only marshals scalar values and a
//! numpy sample vector for MAD.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::estimation::primitives::{
    alpha_beta_apply_measurement as core_alpha_beta_apply_measurement,
    alpha_beta_filter_step as core_alpha_beta_filter_step,
    alpha_beta_predict as core_alpha_beta_predict,
    alpha_beta_steady_state_gains as core_alpha_beta_steady_state_gains,
    cfar_ca_false_alarm_probability as core_cfar_ca_false_alarm_probability,
    cfar_ca_multiplier_from_pfa as core_cfar_ca_multiplier_from_pfa,
    cfar_ca_pfa_from_multiplier as core_cfar_ca_pfa_from_multiplier,
    cfar_ca_threshold as core_cfar_ca_threshold, ewma_update as core_ewma_update,
    ewma_update_power_of_two as core_ewma_update_power_of_two,
    kalman_cv_steady_state_gains as core_kalman_cv_steady_state_gains,
    mad_spread as core_mad_spread, nis_expected_value as core_nis_expected_value,
    nis_gate_test as core_nis_gate_test, nis_gate_threshold as core_nis_gate_threshold,
    nis_statistic as core_nis_statistic, normalized_innovation as core_normalized_innovation,
    AlphaBetaGains, AlphaBetaState, AlphaBetaStep, NisGate, PrimitiveError as CorePrimitiveError,
    ScalarKalmanGains, MAD_GAUSSIAN_CONSISTENCY,
};

use crate::PrimitiveError;

fn primitive_error(err: CorePrimitiveError) -> PyErr {
    PrimitiveError::new_err(err.to_string())
}

/// State of a scalar level plus rate alpha-beta estimator.
#[pyclass(module = "sidereon._sidereon", name = "AlphaBetaState")]
#[derive(Clone, Copy)]
pub struct PyAlphaBetaState {
    inner: AlphaBetaState,
}

impl From<AlphaBetaState> for PyAlphaBetaState {
    fn from(inner: AlphaBetaState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAlphaBetaState {
    /// Create a scalar alpha-beta state.
    ///
    /// `level` carries the tracked scalar quantity and `rate` carries its time
    /// derivative in level units per second when `dt` is seconds.
    #[new]
    fn new(level: f64, rate: f64) -> Self {
        Self {
            inner: AlphaBetaState { level, rate },
        }
    }

    /// Level estimate.
    #[getter]
    fn level(&self) -> f64 {
        self.inner.level
    }

    /// Rate estimate, in level units per `dt` unit.
    #[getter]
    fn rate(&self) -> f64 {
        self.inner.rate
    }

    fn __repr__(&self) -> String {
        format!(
            "AlphaBetaState(level={}, rate={})",
            self.inner.level, self.inner.rate
        )
    }
}

/// Alpha-beta gain pair for one scalar channel.
#[pyclass(module = "sidereon._sidereon", name = "AlphaBetaGains")]
#[derive(Clone, Copy)]
pub struct PyAlphaBetaGains {
    inner: AlphaBetaGains,
}

impl From<AlphaBetaGains> for PyAlphaBetaGains {
    fn from(inner: AlphaBetaGains) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAlphaBetaGains {
    /// Create a scalar alpha-beta gain pair.
    ///
    /// `alpha` is the level gain and `beta` maps an innovation to a rate update
    /// as `beta * innovation / dt`.
    #[new]
    fn new(alpha: f64, beta: f64) -> Self {
        Self {
            inner: AlphaBetaGains { alpha, beta },
        }
    }

    /// Level gain.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Rate gain.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    fn __repr__(&self) -> String {
        format!(
            "AlphaBetaGains(alpha={}, beta={})",
            self.inner.alpha, self.inner.beta
        )
    }
}

/// One alpha-beta predict/update result.
#[pyclass(module = "sidereon._sidereon", name = "AlphaBetaStep")]
#[derive(Clone, Copy)]
pub struct PyAlphaBetaStep {
    inner: AlphaBetaStep,
}

impl From<AlphaBetaStep> for PyAlphaBetaStep {
    fn from(inner: AlphaBetaStep) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAlphaBetaStep {
    /// Predicted state before the measurement update.
    #[getter]
    fn predicted(&self) -> PyAlphaBetaState {
        self.inner.predicted.into()
    }

    /// Updated state after applying the measurement.
    #[getter]
    fn updated(&self) -> PyAlphaBetaState {
        self.inner.updated.into()
    }

    /// Innovation `measurement - predicted.level`.
    #[getter]
    fn innovation(&self) -> f64 {
        self.inner.innovation
    }

    fn __repr__(&self) -> String {
        format!(
            "AlphaBetaStep(innovation={}, updated={})",
            self.inner.innovation,
            self.updated().__repr__()
        )
    }
}

/// Steady-state gains for a scalar constant-velocity Kalman filter.
#[pyclass(module = "sidereon._sidereon", name = "ScalarKalmanGains")]
#[derive(Clone, Copy)]
pub struct PyScalarKalmanGains {
    inner: ScalarKalmanGains,
}

impl From<ScalarKalmanGains> for PyScalarKalmanGains {
    fn from(inner: ScalarKalmanGains) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyScalarKalmanGains {
    /// Position gain `K_x`.
    #[getter]
    fn position_gain(&self) -> f64 {
        self.inner.position_gain
    }

    /// Rate gain `K_v`, in inverse `dt` units.
    #[getter]
    fn rate_gain(&self) -> f64 {
        self.inner.rate_gain
    }

    fn __repr__(&self) -> String {
        format!(
            "ScalarKalmanGains(position_gain={}, rate_gain={})",
            self.inner.position_gain, self.inner.rate_gain
        )
    }
}

/// Normalized innovation squared gate result.
#[pyclass(module = "sidereon._sidereon", name = "NisGate")]
#[derive(Clone, Copy)]
pub struct PyNisGate {
    inner: NisGate,
}

impl From<NisGate> for PyNisGate {
    fn from(inner: NisGate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyNisGate {
    /// Normalized innovation squared statistic.
    #[getter]
    fn nis(&self) -> f64 {
        self.inner.nis
    }

    /// Chi-square gate threshold for the selected confidence and DOF.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Whether `nis <= threshold`.
    #[getter]
    fn in_gate(&self) -> bool {
        self.inner.in_gate
    }

    /// Measurement degrees of freedom.
    #[getter]
    fn dof(&self) -> usize {
        self.inner.dof
    }

    fn __repr__(&self) -> String {
        format!(
            "NisGate(nis={}, threshold={}, in_gate={}, dof={})",
            self.inner.nis, self.inner.threshold, self.inner.in_gate, self.inner.dof
        )
    }
}

/// Compute steady-state alpha-beta gains from dimensionless tracking index.
#[pyfunction]
fn alpha_beta_steady_state_gains(tracking_index: f64) -> PyResult<PyAlphaBetaGains> {
    core_alpha_beta_steady_state_gains(tracking_index)
        .map(Into::into)
        .map_err(primitive_error)
}

/// Project an alpha-beta state by a positive time step.
#[pyfunction]
fn alpha_beta_predict(state: &PyAlphaBetaState, dt: f64) -> PyResult<PyAlphaBetaState> {
    core_alpha_beta_predict(state.inner, dt)
        .map(Into::into)
        .map_err(primitive_error)
}

/// Apply one measurement to a predicted alpha-beta state.
#[pyfunction]
fn alpha_beta_apply_measurement(
    predicted: &PyAlphaBetaState,
    measurement: f64,
    dt: f64,
    gains: &PyAlphaBetaGains,
) -> PyResult<PyAlphaBetaState> {
    core_alpha_beta_apply_measurement(predicted.inner, measurement, dt, gains.inner)
        .map(Into::into)
        .map_err(primitive_error)
}

/// Run one alpha-beta predict/update step.
#[pyfunction]
fn alpha_beta_filter_step(
    state: &PyAlphaBetaState,
    measurement: f64,
    dt: f64,
    gains: &PyAlphaBetaGains,
) -> PyResult<PyAlphaBetaStep> {
    core_alpha_beta_filter_step(state.inner, measurement, dt, gains.inner)
        .map(Into::into)
        .map_err(primitive_error)
}

/// Compute steady-state scalar constant-velocity Kalman gains.
#[pyfunction]
fn kalman_cv_steady_state_gains(
    tracking_index: f64,
    dt: f64,
    measurement_variance: f64,
) -> PyResult<PyScalarKalmanGains> {
    core_kalman_cv_steady_state_gains(tracking_index, dt, measurement_variance)
        .map(Into::into)
        .map_err(primitive_error)
}

/// Scalar normalized innovation `innovation / sqrt(innovation_variance)`.
#[pyfunction]
fn normalized_innovation(innovation: f64, innovation_variance: f64) -> PyResult<f64> {
    core_normalized_innovation(innovation, innovation_variance).map_err(primitive_error)
}

/// Scalar normalized innovation squared statistic.
#[pyfunction(name = "nis")]
fn nis_py(innovation: f64, innovation_variance: f64) -> PyResult<f64> {
    core_nis_statistic(innovation, innovation_variance).map_err(primitive_error)
}

/// Scalar normalized innovation squared statistic.
#[pyfunction]
fn nis_statistic(innovation: f64, innovation_variance: f64) -> PyResult<f64> {
    core_nis_statistic(innovation, innovation_variance).map_err(primitive_error)
}

/// Expected NIS value for the selected degrees of freedom.
#[pyfunction]
fn nis_expected_value(dof: usize) -> PyResult<f64> {
    core_nis_expected_value(dof).map_err(primitive_error)
}

/// Chi-square NIS gate threshold for confidence and degrees of freedom.
#[pyfunction]
fn nis_gate_threshold(dof: usize, confidence: f64) -> PyResult<f64> {
    core_nis_gate_threshold(dof, confidence).map_err(primitive_error)
}

/// Test one innovation against a chi-square NIS gate.
#[pyfunction]
fn nis_gate_test(
    innovation: f64,
    innovation_variance: f64,
    dof: usize,
    confidence: f64,
) -> PyResult<PyNisGate> {
    core_nis_gate_test(innovation, innovation_variance, dof, confidence)
        .map(Into::into)
        .map_err(primitive_error)
}

/// Median absolute deviation spread with Gaussian consistency scaling.
#[pyfunction]
fn mad_spread(values: PyReadonlyArray1<'_, f64>, scale_floor: f64) -> PyResult<f64> {
    let values = values
        .as_slice()
        .map_err(|err| PrimitiveError::new_err(err.to_string()))?;
    core_mad_spread(values, scale_floor).map_err(primitive_error)
}

/// Exponentially weighted moving-average update with gain `alpha`.
#[pyfunction]
fn ewma_update(previous: f64, sample: f64, alpha: f64) -> PyResult<f64> {
    core_ewma_update(previous, sample, alpha).map_err(primitive_error)
}

/// Exponentially weighted moving-average update with `alpha = 1 / 2**shift`.
#[pyfunction]
fn ewma_update_power_of_two(previous: f64, sample: f64, shift: u32) -> PyResult<f64> {
    core_ewma_update_power_of_two(previous, sample, shift).map_err(primitive_error)
}

/// CA-CFAR threshold multiplier from target false-alarm probability.
#[pyfunction]
fn cfar_ca_multiplier_from_pfa(
    searched_cells: usize,
    false_alarm_probability: f64,
) -> PyResult<f64> {
    core_cfar_ca_multiplier_from_pfa(searched_cells, false_alarm_probability)
        .map_err(primitive_error)
}

/// CA-CFAR false-alarm probability from a threshold multiplier.
#[pyfunction]
fn cfar_ca_pfa_from_multiplier(searched_cells: usize, multiplier: f64) -> PyResult<f64> {
    core_cfar_ca_pfa_from_multiplier(searched_cells, multiplier).map_err(primitive_error)
}

/// CA-CFAR absolute threshold from noise level and target false-alarm probability.
#[pyfunction]
fn cfar_ca_threshold(
    searched_cells: usize,
    false_alarm_probability: f64,
    noise_level: f64,
) -> PyResult<f64> {
    core_cfar_ca_threshold(searched_cells, false_alarm_probability, noise_level)
        .map_err(primitive_error)
}

/// CA-CFAR false-alarm probability from absolute threshold and noise level.
#[pyfunction]
fn cfar_ca_false_alarm_probability(
    searched_cells: usize,
    threshold: f64,
    noise_level: f64,
) -> PyResult<f64> {
    core_cfar_ca_false_alarm_probability(searched_cells, threshold, noise_level)
        .map_err(primitive_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAlphaBetaState>()?;
    m.add_class::<PyAlphaBetaGains>()?;
    m.add_class::<PyAlphaBetaStep>()?;
    m.add_class::<PyScalarKalmanGains>()?;
    m.add_class::<PyNisGate>()?;
    m.add("MAD_GAUSSIAN_CONSISTENCY", MAD_GAUSSIAN_CONSISTENCY)?;
    m.add_function(wrap_pyfunction!(alpha_beta_steady_state_gains, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_beta_predict, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_beta_apply_measurement, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_beta_filter_step, m)?)?;
    m.add_function(wrap_pyfunction!(kalman_cv_steady_state_gains, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_innovation, m)?)?;
    m.add_function(wrap_pyfunction!(nis_py, m)?)?;
    m.add_function(wrap_pyfunction!(nis_statistic, m)?)?;
    m.add_function(wrap_pyfunction!(nis_expected_value, m)?)?;
    m.add_function(wrap_pyfunction!(nis_gate_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(nis_gate_test, m)?)?;
    m.add_function(wrap_pyfunction!(mad_spread, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_update, m)?)?;
    m.add_function(wrap_pyfunction!(ewma_update_power_of_two, m)?)?;
    m.add_function(wrap_pyfunction!(cfar_ca_multiplier_from_pfa, m)?)?;
    m.add_function(wrap_pyfunction!(cfar_ca_pfa_from_multiplier, m)?)?;
    m.add_function(wrap_pyfunction!(cfar_ca_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(cfar_ca_false_alarm_probability, m)?)?;
    Ok(())
}
