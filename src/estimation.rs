//! Estimation and detection primitive bindings.
//!
//! Thin wrappers over `sidereon_core::estimation::primitives`: alpha-beta
//! filtering, scalar Kalman gains, normalized-innovation gates, MAD spread,
//! EWMA, and CA-CFAR utilities. The binding only marshals scalar values and a
//! numpy sample vector for MAD.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
use sidereon_core::estimation::{
    smooth_track_rts as core_smooth_track_rts, SmoothedTrack, SmoothedTrackEpoch,
    TrackCoordinateFrame, TrackError, TrackFilter, TrackFilterConfig, TrackGatedUpdate,
    TrackInnovation, TrackPrediction, TrackRtsEpoch, TrackRtsHistory, TrackRtsHistoryBuilder,
    TrackState, TrackUpdate,
};

use crate::np_array;
use crate::PrimitiveError;

fn primitive_error(err: CorePrimitiveError) -> PyErr {
    PrimitiveError::new_err(err.to_string())
}

fn track_error(err: TrackError) -> PyErr {
    PrimitiveError::new_err(err.to_string())
}

fn vector_from_array(values: &PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    let view = values.as_array();
    if view.is_empty() {
        return Err(PrimitiveError::new_err(format!("{name} must not be empty")));
    }
    let mut out = Vec::with_capacity(view.len());
    for (index, value) in view.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(PrimitiveError::new_err(format!(
                "{name}[{index}] must be finite"
            )));
        }
        out.push(value);
    }
    Ok(out)
}

fn square_matrix_from_array(
    values: &PyReadonlyArray2<'_, f64>,
    name: &str,
    expected: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let view = values.as_array();
    if view.nrows() != view.ncols() {
        return Err(PrimitiveError::new_err(format!("{name} must be square")));
    }
    if let Some(expected) = expected {
        if view.nrows() != expected {
            return Err(PrimitiveError::new_err(format!(
                "{name} must have shape ({expected}, {expected})"
            )));
        }
    }
    let mut out = vec![vec![0.0; view.ncols()]; view.nrows()];
    for row in 0..view.nrows() {
        for col in 0..view.ncols() {
            let value = view[[row, col]];
            if !value.is_finite() {
                return Err(PrimitiveError::new_err(format!(
                    "{name}[{row}, {col}] must be finite"
                )));
            }
            out[row][col] = value;
        }
    }
    Ok(out)
}

fn matrix_to_array<'py>(py: Python<'py>, values: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let rows = values.len();
    let cols = values.first().map_or(0, Vec::len);
    let mut array = Array2::<f64>::zeros((rows, cols));
    for (row_index, row) in values.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            array[[row_index, col_index]] = *value;
        }
    }
    PyArray2::from_owned_array(py, array)
}

/// Cartesian frame used by a no-IMU track filter.
#[pyclass(
    module = "sidereon._sidereon",
    name = "TrackCoordinateFrame",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
#[allow(non_camel_case_types)]
pub enum PyTrackCoordinateFrame {
    /// Earth-Centered-Earth-Fixed position in metres.
    ECEF,
    /// Local East-North-Up position in metres.
    ENU,
    /// Caller-defined fixed Cartesian frame in metres.
    CALLER_DEFINED_CARTESIAN,
}

impl From<PyTrackCoordinateFrame> for TrackCoordinateFrame {
    fn from(frame: PyTrackCoordinateFrame) -> Self {
        match frame {
            PyTrackCoordinateFrame::ECEF => Self::Ecef,
            PyTrackCoordinateFrame::ENU => Self::Enu,
            PyTrackCoordinateFrame::CALLER_DEFINED_CARTESIAN => Self::CallerDefinedCartesian,
        }
    }
}

impl From<TrackCoordinateFrame> for PyTrackCoordinateFrame {
    fn from(frame: TrackCoordinateFrame) -> Self {
        match frame {
            TrackCoordinateFrame::Ecef => Self::ECEF,
            TrackCoordinateFrame::Enu => Self::ENU,
            TrackCoordinateFrame::CallerDefinedCartesian => Self::CALLER_DEFINED_CARTESIAN,
        }
    }
}

#[pymethods]
impl PyTrackCoordinateFrame {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::ECEF => "ecef",
            Self::ENU => "enu",
            Self::CALLER_DEFINED_CARTESIAN => "caller_defined_cartesian",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::ECEF => "TrackCoordinateFrame.ECEF",
            Self::ENU => "TrackCoordinateFrame.ENU",
            Self::CALLER_DEFINED_CARTESIAN => "TrackCoordinateFrame.CALLER_DEFINED_CARTESIAN",
        }
    }
}

/// Configuration for a no-IMU constant-velocity track filter.
#[pyclass(module = "sidereon._sidereon", name = "TrackFilterConfig")]
#[derive(Clone)]
pub struct PyTrackFilterConfig {
    inner: TrackFilterConfig,
}

impl From<TrackFilterConfig> for PyTrackFilterConfig {
    fn from(inner: TrackFilterConfig) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackFilterConfig {
    /// Build from position, velocity, covariance, and acceleration PSD.
    #[new]
    fn new(
        frame: PyTrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: PyReadonlyArray1<'_, f64>,
        initial_velocity_m_s: PyReadonlyArray1<'_, f64>,
        initial_covariance: PyReadonlyArray2<'_, f64>,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> PyResult<Self> {
        let position = vector_from_array(&initial_position_m, "initial_position_m")?;
        let velocity = vector_from_array(&initial_velocity_m_s, "initial_velocity_m_s")?;
        let covariance = square_matrix_from_array(&initial_covariance, "initial_covariance", None)?;
        TrackFilterConfig::from_position_velocity(
            frame.into(),
            initial_t_s,
            position,
            velocity,
            covariance,
            acceleration_variance_spectral_density_m2_s3,
        )
        .map(Into::into)
        .map_err(track_error)
    }

    /// Build from a position fix and an uncertain zero initial velocity.
    #[staticmethod]
    fn from_position(
        frame: PyTrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: PyReadonlyArray1<'_, f64>,
        position_covariance_m2: PyReadonlyArray2<'_, f64>,
        initial_velocity_variance_m2_s2: f64,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> PyResult<Self> {
        let position = vector_from_array(&initial_position_m, "initial_position_m")?;
        let covariance =
            square_matrix_from_array(&position_covariance_m2, "position_covariance_m2", None)?;
        TrackFilterConfig::from_position(
            frame.into(),
            initial_t_s,
            position,
            covariance,
            initial_velocity_variance_m2_s2,
            acceleration_variance_spectral_density_m2_s3,
        )
        .map(Into::into)
        .map_err(track_error)
    }

    #[getter]
    fn frame(&self) -> PyTrackCoordinateFrame {
        self.inner.frame.into()
    }

    #[getter]
    fn initial_t_s(&self) -> f64 {
        self.inner.initial_t_s
    }

    #[getter]
    fn initial_position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.initial_position_m)
    }

    #[getter]
    fn initial_velocity_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.initial_velocity_m_s)
    }

    #[getter]
    fn initial_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.initial_covariance)
    }

    #[getter]
    fn acceleration_variance_spectral_density_m2_s3(&self) -> f64 {
        self.inner.acceleration_variance_spectral_density_m2_s3
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackFilterConfig(frame={}, initial_t_s={}, dimension={})",
            self.frame().label(),
            self.inner.initial_t_s,
            self.inner.dimension()
        )
    }
}

/// Track state and covariance over `[position, velocity]`.
#[pyclass(module = "sidereon._sidereon", name = "TrackState")]
#[derive(Clone)]
pub struct PyTrackState {
    inner: TrackState,
}

impl From<TrackState> for PyTrackState {
    fn from(inner: TrackState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackState {
    /// Build a track state.
    #[new]
    fn new(
        frame: PyTrackCoordinateFrame,
        t_s: f64,
        position_m: PyReadonlyArray1<'_, f64>,
        velocity_m_s: PyReadonlyArray1<'_, f64>,
        covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let position = vector_from_array(&position_m, "position_m")?;
        let velocity = vector_from_array(&velocity_m_s, "velocity_m_s")?;
        let covariance = square_matrix_from_array(&covariance, "covariance", None)?;
        TrackState::new(frame.into(), t_s, position, velocity, covariance)
            .map(Into::into)
            .map_err(track_error)
    }

    #[getter]
    fn frame(&self) -> PyTrackCoordinateFrame {
        self.inner.frame.into()
    }

    #[getter]
    fn t_s(&self) -> f64 {
        self.inner.t_s
    }

    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_m)
    }

    #[getter]
    fn velocity_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_m_s)
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.covariance)
    }

    #[getter]
    fn state_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.state_vector())
    }

    #[getter]
    fn position_covariance_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.position_covariance_m2())
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn state_dimension(&self) -> usize {
        self.inner.state_dimension()
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackState(frame={}, t_s={}, dimension={})",
            self.frame().label(),
            self.inner.t_s,
            self.inner.dimension()
        )
    }
}

/// Prediction result from `TrackFilter.predict`.
#[pyclass(module = "sidereon._sidereon", name = "TrackPrediction")]
#[derive(Clone)]
pub struct PyTrackPrediction {
    inner: TrackPrediction,
}

impl From<TrackPrediction> for PyTrackPrediction {
    fn from(inner: TrackPrediction) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackPrediction {
    #[getter]
    fn dt_s(&self) -> f64 {
        self.inner.dt_s
    }

    #[getter]
    fn transition<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.transition)
    }

    #[getter]
    fn process_noise<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.process_noise)
    }

    #[getter]
    fn predicted(&self) -> PyTrackState {
        self.inner.predicted.clone().into()
    }

    fn __repr__(&self) -> String {
        format!("TrackPrediction(dt_s={})", self.inner.dt_s)
    }
}

/// Innovation report for a pending or applied track update.
#[pyclass(module = "sidereon._sidereon", name = "TrackInnovation")]
#[derive(Clone)]
pub struct PyTrackInnovation {
    inner: TrackInnovation,
}

impl From<TrackInnovation> for PyTrackInnovation {
    fn from(inner: TrackInnovation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackInnovation {
    #[getter]
    fn innovation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.innovation)
    }

    #[getter]
    fn innovation_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.innovation_covariance)
    }

    #[getter]
    fn nis(&self) -> f64 {
        self.inner.nis
    }

    /// Evaluate this innovation against a chi-square NIS gate.
    fn gate(&self, confidence: f64) -> PyResult<PyNisGate> {
        self.inner
            .gate(confidence)
            .map(Into::into)
            .map_err(track_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackInnovation(nis={}, dof={})",
            self.inner.nis,
            self.inner.innovation.len()
        )
    }
}

/// Update result from a covariance-weighted correction.
#[pyclass(module = "sidereon._sidereon", name = "TrackUpdate")]
#[derive(Clone)]
pub struct PyTrackUpdate {
    inner: TrackUpdate,
}

impl From<TrackUpdate> for PyTrackUpdate {
    fn from(inner: TrackUpdate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackUpdate {
    #[getter]
    fn predicted(&self) -> PyTrackState {
        self.inner.predicted.clone().into()
    }

    #[getter]
    fn updated(&self) -> PyTrackState {
        self.inner.updated.clone().into()
    }

    #[getter]
    fn innovation(&self) -> PyTrackInnovation {
        self.inner.innovation.clone().into()
    }

    #[getter]
    fn kalman_gain<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.kalman_gain)
    }

    fn __repr__(&self) -> String {
        format!("TrackUpdate(nis={})", self.inner.innovation.nis)
    }
}

/// Gated update result.
#[pyclass(module = "sidereon._sidereon", name = "TrackGatedUpdate")]
#[derive(Clone)]
pub struct PyTrackGatedUpdate {
    inner: TrackGatedUpdate,
}

impl From<TrackGatedUpdate> for PyTrackGatedUpdate {
    fn from(inner: TrackGatedUpdate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackGatedUpdate {
    #[getter]
    fn gate(&self) -> PyNisGate {
        self.inner.gate.into()
    }

    #[getter]
    fn update(&self) -> Option<PyTrackUpdate> {
        self.inner.update.clone().map(Into::into)
    }

    #[getter]
    fn state(&self) -> PyTrackState {
        self.inner.state.clone().into()
    }

    fn __repr__(&self) -> String {
        format!("TrackGatedUpdate(in_gate={})", self.inner.gate.in_gate)
    }
}

/// One epoch in a recorded RTS history.
#[pyclass(module = "sidereon._sidereon", name = "TrackRtsEpoch")]
#[derive(Clone)]
pub struct PyTrackRtsEpoch {
    inner: TrackRtsEpoch,
}

impl From<TrackRtsEpoch> for PyTrackRtsEpoch {
    fn from(inner: TrackRtsEpoch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackRtsEpoch {
    #[getter]
    fn t_s(&self) -> f64 {
        self.inner.t_s
    }

    #[getter]
    fn predicted(&self) -> PyTrackState {
        self.inner.predicted.clone().into()
    }

    #[getter]
    fn updated(&self) -> PyTrackState {
        self.inner.updated.clone().into()
    }

    #[getter]
    fn transition_from_previous<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .transition_from_previous
            .as_ref()
            .map(|transition| matrix_to_array(py, transition))
    }

    fn __repr__(&self) -> String {
        format!("TrackRtsEpoch(t_s={})", self.inner.t_s)
    }
}

/// Recorded forward-pass history accepted by `smooth_track_rts`.
#[pyclass(module = "sidereon._sidereon", name = "TrackRtsHistory")]
#[derive(Clone)]
pub struct PyTrackRtsHistory {
    inner: TrackRtsHistory,
}

impl From<TrackRtsHistory> for PyTrackRtsHistory {
    fn from(inner: TrackRtsHistory) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrackRtsHistory {
    #[getter]
    fn epochs(&self) -> Vec<PyTrackRtsEpoch> {
        self.inner
            .epochs
            .clone()
            .into_iter()
            .map(Into::into)
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.epochs.len()
    }

    fn __repr__(&self) -> String {
        format!("TrackRtsHistory(epochs={})", self.inner.epochs.len())
    }
}

/// Builder for recording a forward filter pass before RTS smoothing.
#[pyclass(module = "sidereon._sidereon", name = "TrackRtsHistoryBuilder")]
#[derive(Clone)]
pub struct PyTrackRtsHistoryBuilder {
    inner: TrackRtsHistoryBuilder,
}

#[pymethods]
impl PyTrackRtsHistoryBuilder {
    /// Start an empty history for manual recording.
    #[new]
    fn new() -> Self {
        Self {
            inner: TrackRtsHistoryBuilder::empty(),
        }
    }

    /// Start a history from the filter's current state.
    #[staticmethod]
    fn from_filter(filter: &PyTrackFilter) -> PyResult<Self> {
        TrackRtsHistoryBuilder::from_filter(&filter.inner)
            .map(|inner| Self { inner })
            .map_err(track_error)
    }

    /// Return a validated history.
    fn finish(&self) -> PyResult<PyTrackRtsHistory> {
        self.inner
            .clone()
            .finish()
            .map(Into::into)
            .map_err(track_error)
    }

    fn __repr__(&self) -> &'static str {
        "TrackRtsHistoryBuilder()"
    }
}

/// One epoch in a smoothed track.
#[pyclass(module = "sidereon._sidereon", name = "SmoothedTrackEpoch")]
#[derive(Clone)]
pub struct PySmoothedTrackEpoch {
    inner: SmoothedTrackEpoch,
}

impl From<SmoothedTrackEpoch> for PySmoothedTrackEpoch {
    fn from(inner: SmoothedTrackEpoch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySmoothedTrackEpoch {
    #[getter]
    fn t_s(&self) -> f64 {
        self.inner.t_s
    }

    #[getter]
    fn state(&self) -> PyTrackState {
        self.inner.state.clone().into()
    }

    #[getter]
    fn rts_gain_to_next<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .rts_gain_to_next
            .as_ref()
            .map(|gain| matrix_to_array(py, gain))
    }

    fn __repr__(&self) -> String {
        format!("SmoothedTrackEpoch(t_s={})", self.inner.t_s)
    }
}

/// Smoothed track returned by fixed-interval RTS smoothing.
#[pyclass(module = "sidereon._sidereon", name = "SmoothedTrack")]
#[derive(Clone)]
pub struct PySmoothedTrack {
    inner: SmoothedTrack,
}

impl From<SmoothedTrack> for PySmoothedTrack {
    fn from(inner: SmoothedTrack) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySmoothedTrack {
    #[getter]
    fn epochs(&self) -> Vec<PySmoothedTrackEpoch> {
        self.inner
            .epochs
            .clone()
            .into_iter()
            .map(Into::into)
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.epochs.len()
    }

    fn __repr__(&self) -> String {
        format!("SmoothedTrack(epochs={})", self.inner.epochs.len())
    }
}

/// Stateful no-IMU constant-velocity track filter.
#[pyclass(module = "sidereon._sidereon", name = "TrackFilter")]
#[derive(Clone)]
pub struct PyTrackFilter {
    inner: TrackFilter,
}

#[pymethods]
impl PyTrackFilter {
    /// Build a filter from a validated configuration.
    #[new]
    fn new(config: &PyTrackFilterConfig) -> PyResult<Self> {
        TrackFilter::new(config.inner.clone())
            .map(|inner| Self { inner })
            .map_err(track_error)
    }

    /// Build from a position fix and an uncertain zero initial velocity.
    #[staticmethod]
    fn from_position(
        frame: PyTrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: PyReadonlyArray1<'_, f64>,
        position_covariance_m2: PyReadonlyArray2<'_, f64>,
        initial_velocity_variance_m2_s2: f64,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> PyResult<Self> {
        let config = PyTrackFilterConfig::from_position(
            frame,
            initial_t_s,
            initial_position_m,
            position_covariance_m2,
            initial_velocity_variance_m2_s2,
            acceleration_variance_spectral_density_m2_s3,
        )?;
        Self::new(&config)
    }

    #[getter]
    fn state(&self) -> PyTrackState {
        self.inner.state().clone().into()
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[getter]
    fn acceleration_variance_spectral_density_m2_s3(&self) -> f64 {
        self.inner.acceleration_variance_spectral_density_m2_s3()
    }

    /// Advance the filter with the constant-velocity prediction model.
    fn predict(&mut self, dt_s: f64) -> PyResult<PyTrackPrediction> {
        self.inner
            .predict(dt_s)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Advance the filter and record the prediction for RTS smoothing.
    fn predict_recorded(
        &mut self,
        dt_s: f64,
        history: &mut PyTrackRtsHistoryBuilder,
    ) -> PyResult<PyTrackPrediction> {
        self.inner
            .predict_recorded(dt_s, &mut history.inner)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Compute the position innovation without applying the update.
    fn position_innovation(
        &self,
        observation_position_m: PyReadonlyArray1<'_, f64>,
        observation_covariance_m2: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyTrackInnovation> {
        let observation = vector_from_array(&observation_position_m, "observation_position_m")?;
        let covariance = square_matrix_from_array(
            &observation_covariance_m2,
            "observation_covariance_m2",
            Some(self.inner.dimension()),
        )?;
        self.inner
            .position_innovation(&observation, &covariance)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Compute the full state innovation without applying the update.
    fn state_innovation(
        &self,
        observation_state: PyReadonlyArray1<'_, f64>,
        observation_covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyTrackInnovation> {
        let observation = vector_from_array(&observation_state, "observation_state")?;
        let covariance = square_matrix_from_array(
            &observation_covariance,
            "observation_covariance",
            Some(self.inner.state().state_dimension()),
        )?;
        self.inner
            .state_innovation(&observation, &covariance)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Apply a position fix plus covariance.
    fn update_position(
        &mut self,
        observation_position_m: PyReadonlyArray1<'_, f64>,
        observation_covariance_m2: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyTrackUpdate> {
        let observation = vector_from_array(&observation_position_m, "observation_position_m")?;
        let covariance = square_matrix_from_array(
            &observation_covariance_m2,
            "observation_covariance_m2",
            Some(self.inner.dimension()),
        )?;
        self.inner
            .update_position(&observation, &covariance)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Apply a full position-and-velocity state fix plus covariance.
    fn update_state(
        &mut self,
        observation_state: PyReadonlyArray1<'_, f64>,
        observation_covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<PyTrackUpdate> {
        let observation = vector_from_array(&observation_state, "observation_state")?;
        let covariance = square_matrix_from_array(
            &observation_covariance,
            "observation_covariance",
            Some(self.inner.state().state_dimension()),
        )?;
        self.inner
            .update_state(&observation, &covariance)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Apply a gated position fix plus covariance.
    fn update_position_gated(
        &mut self,
        observation_position_m: PyReadonlyArray1<'_, f64>,
        observation_covariance_m2: PyReadonlyArray2<'_, f64>,
        confidence: f64,
    ) -> PyResult<PyTrackGatedUpdate> {
        let observation = vector_from_array(&observation_position_m, "observation_position_m")?;
        let covariance = square_matrix_from_array(
            &observation_covariance_m2,
            "observation_covariance_m2",
            Some(self.inner.dimension()),
        )?;
        self.inner
            .update_position_gated(&observation, &covariance, confidence)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Apply a position update and record the epoch for RTS smoothing.
    fn update_position_recorded(
        &mut self,
        observation_position_m: PyReadonlyArray1<'_, f64>,
        observation_covariance_m2: PyReadonlyArray2<'_, f64>,
        history: &mut PyTrackRtsHistoryBuilder,
    ) -> PyResult<PyTrackUpdate> {
        let observation = vector_from_array(&observation_position_m, "observation_position_m")?;
        let covariance = square_matrix_from_array(
            &observation_covariance_m2,
            "observation_covariance_m2",
            Some(self.inner.dimension()),
        )?;
        self.inner
            .update_position_recorded(&observation, &covariance, &mut history.inner)
            .map(Into::into)
            .map_err(track_error)
    }

    /// Apply a gated position update and record accepted or rejected epochs.
    fn update_position_gated_recorded(
        &mut self,
        observation_position_m: PyReadonlyArray1<'_, f64>,
        observation_covariance_m2: PyReadonlyArray2<'_, f64>,
        confidence: f64,
        history: &mut PyTrackRtsHistoryBuilder,
    ) -> PyResult<PyTrackGatedUpdate> {
        let observation = vector_from_array(&observation_position_m, "observation_position_m")?;
        let covariance = square_matrix_from_array(
            &observation_covariance_m2,
            "observation_covariance_m2",
            Some(self.inner.dimension()),
        )?;
        self.inner
            .update_position_gated_recorded(
                &observation,
                &covariance,
                confidence,
                &mut history.inner,
            )
            .map(Into::into)
            .map_err(track_error)
    }

    /// Record the current predicted state as an epoch without measurement update.
    fn record_prediction_only(&self, history: &mut PyTrackRtsHistoryBuilder) -> PyResult<()> {
        self.inner
            .record_prediction_only(&mut history.inner)
            .map_err(track_error)
    }

    fn __repr__(&self) -> String {
        format!(
            "TrackFilter(frame={}, t_s={}, dimension={})",
            PyTrackCoordinateFrame::from(self.inner.state().frame).label(),
            self.inner.state().t_s,
            self.inner.dimension()
        )
    }
}

/// Apply fixed-interval RTS smoothing to a recorded no-IMU track history.
#[pyfunction]
fn smooth_track_rts(history: &PyTrackRtsHistory) -> PyResult<PySmoothedTrack> {
    core_smooth_track_rts(&history.inner)
        .map(Into::into)
        .map_err(track_error)
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
    m.add_class::<PyTrackCoordinateFrame>()?;
    m.add_class::<PyTrackFilterConfig>()?;
    m.add_class::<PyTrackState>()?;
    m.add_class::<PyTrackPrediction>()?;
    m.add_class::<PyTrackInnovation>()?;
    m.add_class::<PyTrackUpdate>()?;
    m.add_class::<PyTrackGatedUpdate>()?;
    m.add_class::<PyTrackFilter>()?;
    m.add_class::<PyTrackRtsEpoch>()?;
    m.add_class::<PyTrackRtsHistory>()?;
    m.add_class::<PyTrackRtsHistoryBuilder>()?;
    m.add_class::<PySmoothedTrackEpoch>()?;
    m.add_class::<PySmoothedTrack>()?;
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
    m.add_function(wrap_pyfunction!(smooth_track_rts, m)?)?;
    Ok(())
}
