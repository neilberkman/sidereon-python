//! GNSS/INS fusion binding.
//!
//! The classes in this module are thin value and handle wrappers around
//! [`sidereon_core::fusion`] and [`sidereon_core::inertial`]. The stateful
//! filter stays opaque to Python while checkpoints move across the boundary as
//! versioned bytes produced by the core codec.

use std::str::FromStr;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyModule};

use sidereon_core::fusion::{
    smooth_fusion_rts as core_smooth_fusion_rts,
    velocity_match_outage as core_velocity_match_outage, EkfCorrectionReport, EkfUpdateOptions,
    ErrorStateLayout, FusionFilterKind, FusionRtsEpoch, FusionRtsHistory, FusionRtsHistoryBuilder,
    FusionUpdate, GnssFixMeasurement, GnssFixStatus, GnssFixStatusWeighting,
    IggIiiMeasurementReweighting, InertialFilter, InertialFilterConfig, InertialFilterSnapshot,
    InnovationGate, InnovationGateReport, InsFilterState, LooseCouplingConfig,
    NonHolonomicConstraintConfig, SerializableFusionState, SmoothedFusionEpoch,
    SmoothedFusionTrajectory, StationaryDetectorConfig, StationaryUpdateConfig,
    TightCarrierPhaseObservation, TightClockState, TightCouplingConfig, TightFilterSnapshot,
    TightGnssEpoch, TightGnssObservation, TightRangeRateObservation, TimeSyncHistoryConfig,
    TimeSyncHistoryStatus, TimeSyncUpdate, UkfUpdateOptions, UnscentedTransformOptions,
    VelocityMatchState, VelocityMatchedTrajectory, VelocityMatchingConfig,
    YangPredictionAdaptiveFactor,
};
use sidereon_core::inertial::{
    ImuGrade, ImuSample, ImuSampleKind, ImuSpec, MechanizationConfig, NavState,
};
use sidereon_core::GnssSatelliteId;

use crate::ephemeris::with_observable_source;
use crate::marshal::{mat3_to_array, matrix3_from_array, FinitePolicy};
use crate::np_array;

fn fusion_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_satellite_id(token: &str) -> PyResult<GnssSatelliteId> {
    GnssSatelliteId::from_str(token)
        .map_err(|_| PyValueError::new_err(format!("invalid GNSS satellite token {token:?}")))
}

fn identity3() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn square_matrix_from_array(
    values: &PyReadonlyArray2<'_, f64>,
    name: &str,
    expected: Option<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let view = values.as_array();
    if view.nrows() != view.ncols() {
        return Err(PyValueError::new_err(format!("{name} must be square")));
    }
    if let Some(expected) = expected {
        if view.nrows() != expected {
            return Err(PyValueError::new_err(format!(
                "{name} must have shape ({expected}, {expected})"
            )));
        }
    }
    let mut out = vec![vec![0.0; view.ncols()]; view.nrows()];
    for row in 0..view.nrows() {
        for col in 0..view.ncols() {
            let value = view[[row, col]];
            if !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name}[{row}, {col}] must be finite"
                )));
            }
            out[row][col] = value;
        }
    }
    Ok(out)
}

fn matrix_to_array<'py>(py: Python<'py>, values: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let nrows = values.len();
    let ncols = values.first().map_or(0, Vec::len);
    let mut array = Array2::<f64>::zeros((nrows, ncols));
    for (row_index, row) in values.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            array[[row_index, col_index]] = *value;
        }
    }
    PyArray2::from_owned_array(py, array)
}

fn bytes_from_source(source: &Bound<'_, PyAny>, name: &str) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    if let Ok(bytes) = source.downcast::<PyByteArray>() {
        // SAFETY: the bytearray is copied immediately and no Python code runs
        // before the copy completes.
        return Ok(unsafe { bytes.as_bytes() }.to_vec());
    }
    Err(PyValueError::new_err(format!(
        "{name} expects bytes or bytearray"
    )))
}

/// Coarse IMU class used to select a built-in stochastic preset.
#[pyclass(module = "sidereon._sidereon", name = "ImuGrade", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyImuGrade {
    /// Low-cost MEMS class.
    MEMS,
    /// Tactical class.
    TACTICAL,
    /// Navigation class.
    NAVIGATION,
}

impl From<PyImuGrade> for ImuGrade {
    fn from(value: PyImuGrade) -> Self {
        match value {
            PyImuGrade::MEMS => Self::Mems,
            PyImuGrade::TACTICAL => Self::Tactical,
            PyImuGrade::NAVIGATION => Self::Navigation,
        }
    }
}

#[pymethods]
impl PyImuGrade {
    /// Stable lowercase grade label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::MEMS => "mems",
            Self::TACTICAL => "tactical",
            Self::NAVIGATION => "navigation",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::MEMS => "ImuGrade.MEMS",
            Self::TACTICAL => "ImuGrade.TACTICAL",
            Self::NAVIGATION => "ImuGrade.NAVIGATION",
        }
    }
}

/// Datasheet-level IMU stochastic parameters.
#[pyclass(module = "sidereon._sidereon", name = "ImuSpec")]
#[derive(Clone, Copy)]
pub struct PyImuSpec {
    inner: ImuSpec,
}

impl PyImuSpec {
    fn inner(&self) -> ImuSpec {
        self.inner
    }
}

#[pymethods]
impl PyImuSpec {
    /// Build an IMU specification from datasheet values.
    #[new]
    #[pyo3(signature = (
        accel_vrw_mps_sqrt_s,
        gyro_arw_rad_sqrt_s,
        accel_bias_instab_mps2,
        gyro_bias_instab_rps,
        accel_bias_tau_s,
        gyro_bias_tau_s,
        accel_scale_instab_ppm=None,
        gyro_scale_instab_ppm=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        accel_vrw_mps_sqrt_s: f64,
        gyro_arw_rad_sqrt_s: f64,
        accel_bias_instab_mps2: f64,
        gyro_bias_instab_rps: f64,
        accel_bias_tau_s: f64,
        gyro_bias_tau_s: f64,
        accel_scale_instab_ppm: Option<f64>,
        gyro_scale_instab_ppm: Option<f64>,
    ) -> PyResult<Self> {
        let inner = ImuSpec::datasheet(
            accel_vrw_mps_sqrt_s,
            gyro_arw_rad_sqrt_s,
            accel_bias_instab_mps2,
            gyro_bias_instab_rps,
            accel_bias_tau_s,
            gyro_bias_tau_s,
            accel_scale_instab_ppm,
            gyro_scale_instab_ppm,
        );
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Return the built-in preset for an IMU grade.
    #[staticmethod]
    fn preset(grade: PyImuGrade) -> Self {
        Self {
            inner: ImuSpec::preset(grade.into()),
        }
    }

    /// Representative low-cost MEMS preset.
    #[staticmethod]
    fn mems() -> Self {
        Self {
            inner: ImuSpec::mems(),
        }
    }

    /// Representative tactical preset.
    #[staticmethod]
    fn tactical() -> Self {
        Self {
            inner: ImuSpec::tactical(),
        }
    }

    /// Representative navigation preset.
    #[staticmethod]
    fn navigation() -> Self {
        Self {
            inner: ImuSpec::navigation(),
        }
    }

    /// Accelerometer velocity random walk in m/s per square-root second.
    #[getter]
    fn accel_vrw_mps_sqrt_s(&self) -> f64 {
        self.inner.accel_vrw_mps_sqrt_s
    }

    /// Gyroscope angular random walk in rad per square-root second.
    #[getter]
    fn gyro_arw_rad_sqrt_s(&self) -> f64 {
        self.inner.gyro_arw_rad_sqrt_s
    }

    /// Accelerometer bias instability in m/s^2.
    #[getter]
    fn accel_bias_instab_mps2(&self) -> f64 {
        self.inner.accel_bias_instab_mps2
    }

    /// Gyroscope bias instability in rad/s.
    #[getter]
    fn gyro_bias_instab_rps(&self) -> f64 {
        self.inner.gyro_bias_instab_rps
    }

    /// Accelerometer bias time constant in seconds.
    #[getter]
    fn accel_bias_tau_s(&self) -> f64 {
        self.inner.accel_bias_tau_s
    }

    /// Gyroscope bias time constant in seconds.
    #[getter]
    fn gyro_bias_tau_s(&self) -> f64 {
        self.inner.gyro_bias_tau_s
    }

    /// Optional accelerometer scale instability in parts per million.
    #[getter]
    fn accel_scale_instab_ppm(&self) -> Option<f64> {
        self.inner.accel_scale_instab_ppm
    }

    /// Optional gyroscope scale instability in parts per million.
    #[getter]
    fn gyro_scale_instab_ppm(&self) -> Option<f64> {
        self.inner.gyro_scale_instab_ppm
    }

    fn __repr__(&self) -> String {
        format!(
            "ImuSpec(accel_vrw_mps_sqrt_s={}, gyro_arw_rad_sqrt_s={})",
            self.inner.accel_vrw_mps_sqrt_s, self.inner.gyro_arw_rad_sqrt_s
        )
    }
}

/// Strapdown mechanization options used during inertial propagation.
#[pyclass(module = "sidereon._sidereon", name = "MechanizationConfig")]
#[derive(Clone, Copy)]
pub struct PyMechanizationConfig {
    inner: MechanizationConfig,
}

impl PyMechanizationConfig {
    fn inner(&self) -> MechanizationConfig {
        self.inner
    }
}

#[pymethods]
impl PyMechanizationConfig {
    /// Build the default ECEF strapdown mechanization config.
    #[new]
    fn new() -> Self {
        Self {
            inner: MechanizationConfig::default(),
        }
    }

    /// Coning correction mode label.
    #[getter]
    fn coning_correction(&self) -> &'static str {
        "off"
    }

    fn __repr__(&self) -> &'static str {
        "MechanizationConfig(coning_correction='off')"
    }
}

/// Fusion filter family selector.
#[pyclass(module = "sidereon._sidereon", name = "FusionFilterKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyFusionFilterKind {
    /// Extended Kalman filter update.
    EKF,
    /// Unscented Kalman filter update.
    UKF,
}

impl From<PyFusionFilterKind> for FusionFilterKind {
    fn from(value: PyFusionFilterKind) -> Self {
        match value {
            PyFusionFilterKind::EKF => Self::Ekf,
            PyFusionFilterKind::UKF => Self::Ukf,
        }
    }
}

impl From<FusionFilterKind> for PyFusionFilterKind {
    fn from(value: FusionFilterKind) -> Self {
        match value {
            FusionFilterKind::Ekf => Self::EKF,
            FusionFilterKind::Ukf => Self::UKF,
        }
    }
}

#[pymethods]
impl PyFusionFilterKind {
    /// Stable lowercase filter label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::EKF => "ekf",
            Self::UKF => "ukf",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::EKF => "FusionFilterKind.EKF",
            Self::UKF => "FusionFilterKind.UKF",
        }
    }
}

/// Error-state covariance layout.
#[pyclass(module = "sidereon._sidereon", name = "ErrorStateLayout", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyErrorStateLayout {
    /// Fifteen-state layout `dr, dv, psi, b_a, b_g`.
    FIFTEEN,
    /// Twenty-one-state layout adding IMU scale-factor states.
    TWENTY_ONE,
}

impl From<PyErrorStateLayout> for ErrorStateLayout {
    fn from(value: PyErrorStateLayout) -> Self {
        match value {
            PyErrorStateLayout::FIFTEEN => Self::Fifteen,
            PyErrorStateLayout::TWENTY_ONE => Self::TwentyOne,
        }
    }
}

impl From<ErrorStateLayout> for PyErrorStateLayout {
    fn from(value: ErrorStateLayout) -> Self {
        match value {
            ErrorStateLayout::Fifteen => Self::FIFTEEN,
            ErrorStateLayout::TwentyOne => Self::TWENTY_ONE,
        }
    }
}

#[pymethods]
impl PyErrorStateLayout {
    /// State dimension for this layout.
    #[getter]
    fn dimension(&self) -> usize {
        ErrorStateLayout::from(*self).dimension()
    }

    /// Stable lowercase layout label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::FIFTEEN => "fifteen",
            Self::TWENTY_ONE => "twenty_one",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::FIFTEEN => "ErrorStateLayout.FIFTEEN",
            Self::TWENTY_ONE => "ErrorStateLayout.TWENTY_ONE",
        }
    }
}

/// Navigation state used by the ECEF strapdown mechanizer.
#[pyclass(module = "sidereon._sidereon", name = "NavState")]
#[derive(Clone, Copy)]
pub struct PyNavState {
    inner: NavState,
}

impl From<NavState> for PyNavState {
    fn from(inner: NavState) -> Self {
        Self { inner }
    }
}

impl PyNavState {
    fn inner(&self) -> NavState {
        self.inner
    }
}

#[pymethods]
impl PyNavState {
    /// Build a navigation state.
    #[new]
    #[pyo3(signature = (
        t_j2000_s,
        position_ecef_m,
        velocity_ecef_mps,
        attitude_body_to_ecef=None,
        *,
        accel_bias_mps2=[0.0; 3],
        gyro_bias_rps=[0.0; 3],
    ))]
    fn new(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        velocity_ecef_mps: [f64; 3],
        attitude_body_to_ecef: Option<PyReadonlyArray2<'_, f64>>,
        accel_bias_mps2: [f64; 3],
        gyro_bias_rps: [f64; 3],
    ) -> PyResult<Self> {
        let attitude = match attitude_body_to_ecef {
            Some(values) => matrix3_from_array(
                &values,
                "attitude_body_to_ecef",
                FinitePolicy::RequireFinite,
            )?,
            None => identity3(),
        };
        let inner = NavState::new(t_j2000_s, position_ecef_m, velocity_ecef_mps, attitude)
            .and_then(|state| state.with_biases(accel_bias_mps2, gyro_bias_rps))
            .map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Return a copy of this state with closed-loop IMU bias estimates.
    fn with_biases(&self, accel_bias_mps2: [f64; 3], gyro_bias_rps: [f64; 3]) -> PyResult<Self> {
        self.inner
            .with_biases(accel_bias_mps2, gyro_bias_rps)
            .map(Self::from)
            .map_err(fusion_err)
    }

    /// State time in seconds since J2000.
    #[getter]
    fn t_j2000_s(&self) -> f64 {
        self.inner.t_j2000_s
    }

    /// IMU ECEF position in metres.
    #[getter]
    fn position_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_ecef_m)
    }

    /// IMU ECEF velocity in metres per second.
    #[getter]
    fn velocity_ecef_mps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_ecef_mps)
    }

    /// Body-to-ECEF direction cosine matrix.
    #[getter]
    fn attitude_body_to_ecef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.attitude_body_to_ecef)
    }

    /// Closed-loop accelerometer bias estimate in m/s^2.
    #[getter]
    fn accel_bias_mps2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.accel_bias_mps2)
    }

    /// Closed-loop gyroscope bias estimate in rad/s.
    #[getter]
    fn gyro_bias_rps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.gyro_bias_rps)
    }

    fn __repr__(&self) -> String {
        format!("NavState(t_j2000_s={})", self.inner.t_j2000_s)
    }
}

/// One IMU sample tagged by its end time in seconds since J2000.
#[pyclass(module = "sidereon._sidereon", name = "ImuSample")]
#[derive(Clone, Copy)]
pub struct PyImuSample {
    inner: ImuSample,
}

impl PyImuSample {
    fn inner(&self) -> ImuSample {
        self.inner
    }
}

#[pymethods]
impl PyImuSample {
    /// Build an IMU sample from specific force and angular rate.
    #[staticmethod]
    fn rate(t_j2000_s: f64, specific_force_mps2: [f64; 3], angular_rate_rps: [f64; 3]) -> Self {
        Self {
            inner: ImuSample::rate(t_j2000_s, specific_force_mps2, angular_rate_rps),
        }
    }

    /// Build an IMU sample from sensor-provided increments.
    #[staticmethod]
    fn increment(
        t_j2000_s: f64,
        delta_velocity_mps: [f64; 3],
        delta_theta_rad: [f64; 3],
        dt_s: f64,
    ) -> Self {
        Self {
            inner: ImuSample::increment(t_j2000_s, delta_velocity_mps, delta_theta_rad, dt_s),
        }
    }

    /// Sample end time in seconds since J2000.
    #[getter]
    fn t_j2000_s(&self) -> f64 {
        self.inner.t_j2000_s
    }

    /// Stable payload-kind label.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            ImuSampleKind::Rate { .. } => "rate",
            ImuSampleKind::Increment { .. } => "increment",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ImuSample(t_j2000_s={}, kind='{}')",
            self.inner.t_j2000_s,
            self.kind()
        )
    }
}

/// Closed-loop INS filter state.
#[pyclass(module = "sidereon._sidereon", name = "InsFilterState")]
#[derive(Clone)]
pub struct PyInsFilterState {
    inner: InsFilterState,
}

impl From<InsFilterState> for PyInsFilterState {
    fn from(inner: InsFilterState) -> Self {
        Self { inner }
    }
}

impl PyInsFilterState {
    fn inner(&self) -> InsFilterState {
        self.inner.clone()
    }
}

#[pymethods]
impl PyInsFilterState {
    /// Build a filter state from diagonal covariance entries.
    #[staticmethod]
    fn from_diagonal(
        nominal: &PyNavState,
        layout: PyErrorStateLayout,
        diagonal: Vec<f64>,
    ) -> PyResult<Self> {
        InsFilterState::from_diagonal(nominal.inner(), layout.into(), &diagonal)
            .map(Self::from)
            .map_err(fusion_err)
    }

    /// Build a filter state from a full covariance matrix.
    #[staticmethod]
    fn from_covariance(
        nominal: &PyNavState,
        layout: PyErrorStateLayout,
        covariance: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let layout_core = ErrorStateLayout::from(layout);
        let covariance =
            square_matrix_from_array(&covariance, "covariance", Some(layout_core.dimension()))?;
        InsFilterState::new(nominal.inner(), layout_core, covariance)
            .map(Self::from)
            .map_err(fusion_err)
    }

    /// Nonlinear mechanized navigation state.
    #[getter]
    fn nominal(&self) -> PyNavState {
        PyNavState::from(self.inner.nominal)
    }

    /// Error-state covariance matrix.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.covariance)
    }

    /// Selected error-state layout.
    #[getter]
    fn layout(&self) -> PyErrorStateLayout {
        self.inner.layout().into()
    }

    /// Error-state dimension.
    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn __repr__(&self) -> String {
        format!("InsFilterState(dimension={})", self.inner.dimension())
    }
}

/// Innovation screening options for EKF and UKF updates.
#[pyclass(module = "sidereon._sidereon", name = "InnovationGate")]
#[derive(Clone, Copy)]
pub struct PyInnovationGate {
    inner: InnovationGate,
}

impl PyInnovationGate {
    fn inner(&self) -> InnovationGate {
        self.inner
    }
}

#[pymethods]
impl PyInnovationGate {
    /// Build normalized-innovation screening options.
    #[new]
    #[pyo3(signature = (threshold_sigma, min_rows=1))]
    fn new(threshold_sigma: f64, min_rows: usize) -> PyResult<Self> {
        let inner = InnovationGate {
            threshold_sigma,
            min_rows,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Rejection threshold in normalized-innovation sigma.
    #[getter]
    fn threshold_sigma(&self) -> f64 {
        self.inner.threshold_sigma
    }

    /// Minimum accepted rows required to apply an update.
    #[getter]
    fn min_rows(&self) -> usize {
        self.inner.min_rows
    }
}

/// Generic EKF correction options.
#[pyclass(module = "sidereon._sidereon", name = "EkfUpdateOptions")]
#[derive(Clone, Copy)]
pub struct PyEkfUpdateOptions {
    inner: EkfUpdateOptions,
}

impl PyEkfUpdateOptions {
    fn inner(&self) -> EkfUpdateOptions {
        self.inner
    }
}

#[pymethods]
impl PyEkfUpdateOptions {
    /// Build EKF update options.
    #[new]
    #[pyo3(signature = (innovation_gate=None))]
    fn new(innovation_gate: Option<&PyInnovationGate>) -> Self {
        Self {
            inner: EkfUpdateOptions {
                innovation_gate: innovation_gate.map(PyInnovationGate::inner),
            },
        }
    }

    /// Innovation gate options when screening is active.
    #[getter]
    fn innovation_gate(&self) -> Option<PyInnovationGate> {
        self.inner
            .innovation_gate
            .map(|inner| PyInnovationGate { inner })
    }
}

/// Scaled unscented-transform parameters.
#[pyclass(module = "sidereon._sidereon", name = "UnscentedTransformOptions")]
#[derive(Clone, Copy)]
pub struct PyUnscentedTransformOptions {
    inner: UnscentedTransformOptions,
}

impl PyUnscentedTransformOptions {
    fn inner(&self) -> UnscentedTransformOptions {
        self.inner
    }
}

#[pymethods]
impl PyUnscentedTransformOptions {
    /// Build unscented-transform options.
    #[new]
    #[pyo3(signature = (alpha=0.5, beta=2.0, kappa=0.0))]
    fn new(alpha: f64, beta: f64, kappa: f64) -> Self {
        Self {
            inner: UnscentedTransformOptions { alpha, beta, kappa },
        }
    }

    /// Sigma-point spread around the mean.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Prior-distribution shape parameter.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    /// Secondary sigma-point scaling parameter.
    #[getter]
    fn kappa(&self) -> f64 {
        self.inner.kappa
    }
}

/// UKF measurement-correction options.
#[pyclass(module = "sidereon._sidereon", name = "UkfUpdateOptions")]
#[derive(Clone, Copy)]
pub struct PyUkfUpdateOptions {
    inner: UkfUpdateOptions,
}

impl PyUkfUpdateOptions {
    fn inner(&self) -> UkfUpdateOptions {
        self.inner
    }
}

#[pymethods]
impl PyUkfUpdateOptions {
    /// Build UKF update options.
    #[new]
    #[pyo3(signature = (transform=None, innovation_gate=None))]
    fn new(
        transform: Option<&PyUnscentedTransformOptions>,
        innovation_gate: Option<&PyInnovationGate>,
    ) -> Self {
        Self {
            inner: UkfUpdateOptions {
                transform: transform
                    .map(PyUnscentedTransformOptions::inner)
                    .unwrap_or_default(),
                innovation_gate: innovation_gate.map(PyInnovationGate::inner),
            },
        }
    }

    /// Unscented-transform parameters.
    #[getter]
    fn transform(&self) -> PyUnscentedTransformOptions {
        PyUnscentedTransformOptions {
            inner: self.inner.transform,
        }
    }

    /// Innovation gate options when screening is active.
    #[getter]
    fn innovation_gate(&self) -> Option<PyInnovationGate> {
        self.inner
            .innovation_gate
            .map(|inner| PyInnovationGate { inner })
    }
}

/// Upstream GNSS fix class used by loose measurement weighting.
#[pyclass(module = "sidereon._sidereon", name = "GnssFixStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyGnssFixStatus {
    /// Code-only or standalone GNSS fix.
    SINGLE,
    /// Float carrier-phase ambiguity solution.
    FLOAT,
    /// Fixed carrier-phase ambiguity solution.
    FIXED,
}

impl From<PyGnssFixStatus> for GnssFixStatus {
    fn from(value: PyGnssFixStatus) -> Self {
        match value {
            PyGnssFixStatus::SINGLE => Self::Single,
            PyGnssFixStatus::FLOAT => Self::Float,
            PyGnssFixStatus::FIXED => Self::Fixed,
        }
    }
}

impl From<GnssFixStatus> for PyGnssFixStatus {
    fn from(value: GnssFixStatus) -> Self {
        match value {
            GnssFixStatus::Single => Self::SINGLE,
            GnssFixStatus::Float => Self::FLOAT,
            GnssFixStatus::Fixed => Self::FIXED,
        }
    }
}

#[pymethods]
impl PyGnssFixStatus {
    /// Stable lowercase fix-status label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::SINGLE => "single",
            Self::FLOAT => "float",
            Self::FIXED => "fixed",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::SINGLE => "GnssFixStatus.SINGLE",
            Self::FLOAT => "GnssFixStatus.FLOAT",
            Self::FIXED => "GnssFixStatus.FIXED",
        }
    }
}

/// Per-fix-status sigma multipliers applied to loose GNSS covariance.
#[pyclass(module = "sidereon._sidereon", name = "GnssFixStatusWeighting")]
#[derive(Clone, Copy)]
pub struct PyGnssFixStatusWeighting {
    inner: GnssFixStatusWeighting,
}

impl PyGnssFixStatusWeighting {
    fn inner(&self) -> GnssFixStatusWeighting {
        self.inner
    }
}

#[pymethods]
impl PyGnssFixStatusWeighting {
    /// Build per-fix-status loose GNSS sigma multipliers.
    #[new]
    #[pyo3(signature = (
        single_sigma_multiplier=1.0,
        float_sigma_multiplier=1.0,
        fixed_sigma_multiplier=1.0,
    ))]
    fn new(
        single_sigma_multiplier: f64,
        float_sigma_multiplier: f64,
        fixed_sigma_multiplier: f64,
    ) -> PyResult<Self> {
        let inner = GnssFixStatusWeighting {
            single_sigma_multiplier,
            float_sigma_multiplier,
            fixed_sigma_multiplier,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Sigma multiplier for standalone GNSS fixes.
    #[getter]
    fn single_sigma_multiplier(&self) -> f64 {
        self.inner.single_sigma_multiplier
    }

    /// Sigma multiplier for float carrier-phase fixes.
    #[getter]
    fn float_sigma_multiplier(&self) -> f64 {
        self.inner.float_sigma_multiplier
    }

    /// Sigma multiplier for fixed carrier-phase fixes.
    #[getter]
    fn fixed_sigma_multiplier(&self) -> f64 {
        self.inner.fixed_sigma_multiplier
    }
}

/// IGG-III measurement variance inflation for loose GNSS updates.
#[pyclass(module = "sidereon._sidereon", name = "IggIiiMeasurementReweighting")]
#[derive(Clone, Copy)]
pub struct PyIggIiiMeasurementReweighting {
    inner: IggIiiMeasurementReweighting,
}

impl PyIggIiiMeasurementReweighting {
    fn inner(&self) -> IggIiiMeasurementReweighting {
        self.inner
    }
}

#[pymethods]
impl PyIggIiiMeasurementReweighting {
    /// Build IGG-III break points for robust loose updates.
    #[new]
    #[pyo3(signature = (k0_sigma=2.0, k1_sigma=5.0))]
    fn new(k0_sigma: f64, k1_sigma: f64) -> PyResult<Self> {
        let inner = IggIiiMeasurementReweighting { k0_sigma, k1_sigma };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Common loose-GNSS break points from the core defaults.
    #[staticmethod]
    fn standard() -> Self {
        Self {
            inner: IggIiiMeasurementReweighting::standard(),
        }
    }

    /// Lower standardized-innovation break point in sigma.
    #[getter]
    fn k0_sigma(&self) -> f64 {
        self.inner.k0_sigma
    }

    /// Upper standardized-innovation break point in sigma.
    #[getter]
    fn k1_sigma(&self) -> f64 {
        self.inner.k1_sigma
    }
}

/// Yang prediction adaptive factor for loose GNSS updates.
#[pyclass(module = "sidereon._sidereon", name = "YangPredictionAdaptiveFactor")]
#[derive(Clone, Copy)]
pub struct PyYangPredictionAdaptiveFactor {
    inner: YangPredictionAdaptiveFactor,
}

impl PyYangPredictionAdaptiveFactor {
    fn inner(&self) -> YangPredictionAdaptiveFactor {
        self.inner
    }
}

#[pymethods]
impl PyYangPredictionAdaptiveFactor {
    /// Build the two-segment prediction inflation settings.
    #[new]
    #[pyo3(signature = (threshold=1.0, outlier_gate_probability=0.99))]
    fn new(threshold: f64, outlier_gate_probability: f64) -> PyResult<Self> {
        let inner = YangPredictionAdaptiveFactor {
            threshold,
            outlier_gate_probability,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Conservative core default for prediction inflation and outlier gating.
    #[staticmethod]
    fn standard() -> Self {
        Self {
            inner: YangPredictionAdaptiveFactor::standard(),
        }
    }

    /// Two-segment threshold for the predicted-residual statistic.
    #[getter]
    fn threshold(&self) -> f64 {
        self.inner.threshold
    }

    /// Chi-square probability used for the Mahalanobis outlier gate.
    #[getter]
    fn outlier_gate_probability(&self) -> f64 {
        self.inner.outlier_gate_probability
    }
}

/// Windowed accel and gyro magnitude detector for stationary updates.
#[pyclass(module = "sidereon._sidereon", name = "StationaryDetectorConfig")]
#[derive(Clone, Copy)]
pub struct PyStationaryDetectorConfig {
    inner: StationaryDetectorConfig,
}

impl PyStationaryDetectorConfig {
    fn inner(&self) -> StationaryDetectorConfig {
        self.inner
    }
}

#[pymethods]
impl PyStationaryDetectorConfig {
    /// Build a stationary detector over a trailing IMU epoch window.
    #[new]
    fn new(
        window_len: usize,
        max_specific_force_norm_error_mps2: f64,
        max_body_rate_wrt_ecef_norm_rps: f64,
    ) -> PyResult<Self> {
        let inner = StationaryDetectorConfig {
            window_len,
            max_specific_force_norm_error_mps2,
            max_body_rate_wrt_ecef_norm_rps,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Required propagated IMU epochs before the detector can fire.
    #[getter]
    fn window_len(&self) -> usize {
        self.inner.window_len
    }

    /// Maximum specific-force norm error from local gravity.
    #[getter]
    fn max_specific_force_norm_error_mps2(&self) -> f64 {
        self.inner.max_specific_force_norm_error_mps2
    }

    /// Maximum body angular-rate norm relative to ECEF.
    #[getter]
    fn max_body_rate_wrt_ecef_norm_rps(&self) -> f64 {
        self.inner.max_body_rate_wrt_ecef_norm_rps
    }
}

/// Stationarity detector and pseudo-measurement sigmas for ZUPT/ZARU.
#[pyclass(module = "sidereon._sidereon", name = "StationaryUpdateConfig")]
#[derive(Clone, Copy)]
pub struct PyStationaryUpdateConfig {
    inner: StationaryUpdateConfig,
}

impl PyStationaryUpdateConfig {
    fn inner(&self) -> StationaryUpdateConfig {
        self.inner
    }
}

#[pymethods]
impl PyStationaryUpdateConfig {
    /// Build stationary zero-velocity and zero-angular-rate update settings.
    #[new]
    fn new(
        detector: &PyStationaryDetectorConfig,
        zero_velocity_sigma_mps: f64,
        zero_angular_rate_sigma_rps: f64,
    ) -> PyResult<Self> {
        let inner = StationaryUpdateConfig {
            detector: detector.inner(),
            zero_velocity_sigma_mps,
            zero_angular_rate_sigma_rps,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Detector thresholds over a trailing IMU epoch window.
    #[getter]
    fn detector(&self) -> PyStationaryDetectorConfig {
        PyStationaryDetectorConfig {
            inner: self.inner.detector,
        }
    }

    /// One-sigma zero-velocity pseudo-measurement noise in m/s.
    #[getter]
    fn zero_velocity_sigma_mps(&self) -> f64 {
        self.inner.zero_velocity_sigma_mps
    }

    /// One-sigma zero-angular-rate pseudo-measurement noise in rad/s.
    #[getter]
    fn zero_angular_rate_sigma_rps(&self) -> f64 {
        self.inner.zero_angular_rate_sigma_rps
    }
}

/// Non-holonomic wheeled-vehicle velocity constraint settings.
#[pyclass(module = "sidereon._sidereon", name = "NonHolonomicConstraintConfig")]
#[derive(Clone, Copy)]
pub struct PyNonHolonomicConstraintConfig {
    inner: NonHolonomicConstraintConfig,
}

impl PyNonHolonomicConstraintConfig {
    fn inner(&self) -> NonHolonomicConstraintConfig {
        self.inner
    }
}

#[pymethods]
impl PyNonHolonomicConstraintConfig {
    /// Build wheeled-vehicle lateral and vertical velocity constraints.
    #[new]
    fn new(
        lateral_velocity_sigma_mps: f64,
        vertical_velocity_sigma_mps: f64,
        min_speed_mps: f64,
        max_body_rate_wrt_ecef_norm_rps: f64,
    ) -> PyResult<Self> {
        let inner = NonHolonomicConstraintConfig {
            lateral_velocity_sigma_mps,
            vertical_velocity_sigma_mps,
            min_speed_mps,
            max_body_rate_wrt_ecef_norm_rps,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// One-sigma lateral body velocity pseudo-measurement noise in m/s.
    #[getter]
    fn lateral_velocity_sigma_mps(&self) -> f64 {
        self.inner.lateral_velocity_sigma_mps
    }

    /// One-sigma vertical body velocity pseudo-measurement noise in m/s.
    #[getter]
    fn vertical_velocity_sigma_mps(&self) -> f64 {
        self.inner.vertical_velocity_sigma_mps
    }

    /// Minimum ECEF speed required before applying the constraint.
    #[getter]
    fn min_speed_mps(&self) -> f64 {
        self.inner.min_speed_mps
    }

    /// Maximum body angular-rate norm relative to ECEF.
    #[getter]
    fn max_body_rate_wrt_ecef_norm_rps(&self) -> f64 {
        self.inner.max_body_rate_wrt_ecef_norm_rps
    }
}

/// Endpoint matching settings for a GNSS outage span.
#[pyclass(module = "sidereon._sidereon", name = "VelocityMatchingConfig")]
#[derive(Clone, Copy)]
pub struct PyVelocityMatchingConfig {
    inner: VelocityMatchingConfig,
}

impl PyVelocityMatchingConfig {
    fn inner(&self) -> VelocityMatchingConfig {
        self.inner
    }
}

#[pymethods]
impl PyVelocityMatchingConfig {
    /// Build endpoint velocity matching settings for a GNSS outage span.
    #[new]
    fn new(max_outage_duration_s: f64) -> PyResult<Self> {
        let inner = VelocityMatchingConfig {
            max_outage_duration_s,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Maximum outage interval accepted by the matcher.
    #[getter]
    fn max_outage_duration_s(&self) -> f64 {
        self.inner.max_outage_duration_s
    }
}

/// One position/velocity sample used by velocity matching.
#[pyclass(module = "sidereon._sidereon", name = "VelocityMatchState")]
#[derive(Clone, Copy)]
pub struct PyVelocityMatchState {
    inner: VelocityMatchState,
}

impl PyVelocityMatchState {
    fn inner(&self) -> VelocityMatchState {
        self.inner
    }
}

#[pymethods]
impl PyVelocityMatchState {
    /// Build one velocity-matching state sample.
    #[new]
    fn new(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        velocity_ecef_mps: [f64; 3],
    ) -> PyResult<Self> {
        VelocityMatchState::new(t_j2000_s, position_ecef_m, velocity_ecef_mps)
            .map(|inner| Self { inner })
            .map_err(fusion_err)
    }

    /// Sample epoch in seconds since J2000.
    #[getter]
    fn t_j2000_s(&self) -> f64 {
        self.inner.t_j2000_s
    }

    /// INS position in ECEF metres.
    #[getter]
    fn position_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_ecef_m)
    }

    /// INS velocity in ECEF metres per second.
    #[getter]
    fn velocity_ecef_mps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_ecef_mps)
    }
}

/// Output from endpoint velocity matching across one outage.
#[pyclass(module = "sidereon._sidereon", name = "VelocityMatchedTrajectory")]
#[derive(Clone)]
pub struct PyVelocityMatchedTrajectory {
    inner: VelocityMatchedTrajectory,
}

impl From<VelocityMatchedTrajectory> for PyVelocityMatchedTrajectory {
    fn from(inner: VelocityMatchedTrajectory) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVelocityMatchedTrajectory {
    /// Corrected states in input order.
    #[getter]
    fn states(&self) -> Vec<PyVelocityMatchState> {
        self.inner
            .states
            .iter()
            .copied()
            .map(|inner| PyVelocityMatchState { inner })
            .collect()
    }

    /// Position correction applied at the return-fix endpoint.
    #[getter]
    fn endpoint_position_correction_ecef_m(&self) -> [f64; 3] {
        self.inner.endpoint_position_correction_ecef_m
    }

    /// Velocity correction applied at the return-fix endpoint.
    #[getter]
    fn endpoint_velocity_correction_ecef_mps(&self) -> [f64; 3] {
        self.inner.endpoint_velocity_correction_ecef_mps
    }
}

/// Loose-coupled GNSS update options.
#[pyclass(module = "sidereon._sidereon", name = "LooseCouplingConfig")]
#[derive(Clone, Copy)]
pub struct PyLooseCouplingConfig {
    inner: LooseCouplingConfig,
}

impl PyLooseCouplingConfig {
    fn inner(&self) -> LooseCouplingConfig {
        self.inner
    }
}

#[pymethods]
impl PyLooseCouplingConfig {
    /// Build loose-coupling options.
    #[new]
    #[pyo3(signature = (
        lever_arm_body_m=[0.0; 3],
        update_options=None,
        fix_status_weighting=None,
        measurement_reweighting=None,
        prediction_adaptation=None,
        stationary_updates=None,
        non_holonomic=None,
    ))]
    fn new(
        lever_arm_body_m: [f64; 3],
        update_options: Option<&PyEkfUpdateOptions>,
        fix_status_weighting: Option<&PyGnssFixStatusWeighting>,
        measurement_reweighting: Option<&PyIggIiiMeasurementReweighting>,
        prediction_adaptation: Option<&PyYangPredictionAdaptiveFactor>,
        stationary_updates: Option<&PyStationaryUpdateConfig>,
        non_holonomic: Option<&PyNonHolonomicConstraintConfig>,
    ) -> PyResult<Self> {
        let inner = LooseCouplingConfig {
            lever_arm_body_m,
            update_options: update_options
                .map(PyEkfUpdateOptions::inner)
                .unwrap_or_default(),
            fix_status_weighting: fix_status_weighting
                .map(PyGnssFixStatusWeighting::inner)
                .unwrap_or_default(),
            measurement_reweighting: measurement_reweighting
                .map(PyIggIiiMeasurementReweighting::inner),
            prediction_adaptation: prediction_adaptation.map(PyYangPredictionAdaptiveFactor::inner),
            stationary_updates: stationary_updates.map(PyStationaryUpdateConfig::inner),
            non_holonomic: non_holonomic.map(PyNonHolonomicConstraintConfig::inner),
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Body-frame vector from IMU origin to GNSS antenna phase center, metres.
    #[getter]
    fn lever_arm_body_m(&self) -> [f64; 3] {
        self.inner.lever_arm_body_m
    }

    /// Generic EKF correction options.
    #[getter]
    fn update_options(&self) -> PyEkfUpdateOptions {
        PyEkfUpdateOptions {
            inner: self.inner.update_options,
        }
    }

    /// Per-fix-status sigma multipliers applied to GNSS covariance.
    #[getter]
    fn fix_status_weighting(&self) -> PyGnssFixStatusWeighting {
        PyGnssFixStatusWeighting {
            inner: self.inner.fix_status_weighting,
        }
    }

    /// Optional IGG-III measurement variance inflation settings.
    #[getter]
    fn measurement_reweighting(&self) -> Option<PyIggIiiMeasurementReweighting> {
        self.inner
            .measurement_reweighting
            .map(|inner| PyIggIiiMeasurementReweighting { inner })
    }

    /// Optional Yang predicted-covariance adaptation settings.
    #[getter]
    fn prediction_adaptation(&self) -> Option<PyYangPredictionAdaptiveFactor> {
        self.inner
            .prediction_adaptation
            .map(|inner| PyYangPredictionAdaptiveFactor { inner })
    }

    /// Optional stationary zero-velocity and zero-angular-rate updates.
    #[getter]
    fn stationary_updates(&self) -> Option<PyStationaryUpdateConfig> {
        self.inner
            .stationary_updates
            .map(|inner| PyStationaryUpdateConfig { inner })
    }

    /// Optional wheeled-vehicle lateral and vertical velocity constraints.
    #[getter]
    fn non_holonomic(&self) -> Option<PyNonHolonomicConstraintConfig> {
        self.inner
            .non_holonomic
            .map(|inner| PyNonHolonomicConstraintConfig { inner })
    }
}

/// Tight raw GNSS update options.
#[pyclass(module = "sidereon._sidereon", name = "TightCouplingConfig")]
#[derive(Clone, Copy)]
pub struct PyTightCouplingConfig {
    inner: TightCouplingConfig,
}

impl PyTightCouplingConfig {
    fn inner(&self) -> TightCouplingConfig {
        self.inner
    }
}

#[pymethods]
impl PyTightCouplingConfig {
    /// Build tight-coupling options.
    #[new]
    #[pyo3(signature = (
        lever_arm_body_m=[0.0; 3],
        light_time=true,
        sagnac=true,
        initial_clock_bias_variance_m2=1.0e12,
        initial_clock_drift_variance_m2_s2=1.0e6,
        clock_bias_random_walk_m2_s=1.0,
        clock_drift_random_walk_m2_s3=1.0e-2,
        update_options=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lever_arm_body_m: [f64; 3],
        light_time: bool,
        sagnac: bool,
        initial_clock_bias_variance_m2: f64,
        initial_clock_drift_variance_m2_s2: f64,
        clock_bias_random_walk_m2_s: f64,
        clock_drift_random_walk_m2_s3: f64,
        update_options: Option<&PyEkfUpdateOptions>,
    ) -> PyResult<Self> {
        let inner = TightCouplingConfig {
            lever_arm_body_m,
            light_time,
            sagnac,
            initial_clock_bias_variance_m2,
            initial_clock_drift_variance_m2_s2,
            clock_bias_random_walk_m2_s,
            clock_drift_random_walk_m2_s3,
            update_options: update_options
                .map(PyEkfUpdateOptions::inner)
                .unwrap_or_default(),
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Body-frame vector from IMU origin to GNSS antenna phase center, metres.
    #[getter]
    fn lever_arm_body_m(&self) -> [f64; 3] {
        self.inner.lever_arm_body_m
    }

    /// Whether code and carrier rows apply light-time correction.
    #[getter]
    fn light_time(&self) -> bool {
        self.inner.light_time
    }

    /// Whether code and carrier rows apply Earth-rotation correction.
    #[getter]
    fn sagnac(&self) -> bool {
        self.inner.sagnac
    }

    /// Initial receiver-clock bias variance in square metres.
    #[getter]
    fn initial_clock_bias_variance_m2(&self) -> f64 {
        self.inner.initial_clock_bias_variance_m2
    }

    /// Initial receiver-clock drift variance in square metres per square second.
    #[getter]
    fn initial_clock_drift_variance_m2_s2(&self) -> f64 {
        self.inner.initial_clock_drift_variance_m2_s2
    }

    /// Receiver-clock bias random-walk spectral density in square metres per second.
    #[getter]
    fn clock_bias_random_walk_m2_s(&self) -> f64 {
        self.inner.clock_bias_random_walk_m2_s
    }

    /// Receiver-clock drift random-walk spectral density in square metres per cubic second.
    #[getter]
    fn clock_drift_random_walk_m2_s3(&self) -> f64 {
        self.inner.clock_drift_random_walk_m2_s3
    }

    /// Generic EKF correction options used by tight raw GNSS updates.
    #[getter]
    fn update_options(&self) -> PyEkfUpdateOptions {
        PyEkfUpdateOptions {
            inner: self.inner.update_options,
        }
    }
}

/// Configuration for a stateful inertial filter.
#[pyclass(module = "sidereon._sidereon", name = "InertialFilterConfig")]
#[derive(Clone, Copy)]
pub struct PyInertialFilterConfig {
    inner: InertialFilterConfig,
}

impl PyInertialFilterConfig {
    fn inner(&self) -> InertialFilterConfig {
        self.inner
    }
}

#[pymethods]
impl PyInertialFilterConfig {
    /// Build a filter configuration.
    #[new]
    #[pyo3(signature = (
        imu_spec,
        filter_kind=PyFusionFilterKind::EKF,
        mechanization=None,
        loose=None,
        tight=None,
        ukf_update_options=None,
        *,
        imu_to_body_dcm=None,
    ))]
    fn new(
        imu_spec: &PyImuSpec,
        filter_kind: PyFusionFilterKind,
        mechanization: Option<&PyMechanizationConfig>,
        loose: Option<&PyLooseCouplingConfig>,
        tight: Option<&PyTightCouplingConfig>,
        ukf_update_options: Option<&PyUkfUpdateOptions>,
        imu_to_body_dcm: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        let mut inner = InertialFilterConfig::new(imu_spec.inner()).map_err(fusion_err)?;
        inner.filter_kind = filter_kind.into();
        if let Some(imu_to_body_dcm) = imu_to_body_dcm {
            inner.imu_to_body_dcm = matrix3_from_array(
                &imu_to_body_dcm,
                "imu_to_body_dcm",
                FinitePolicy::RequireFinite,
            )?;
        }
        if let Some(mechanization) = mechanization {
            inner.mechanization = mechanization.inner();
        }
        if let Some(loose) = loose {
            inner.loose = loose.inner();
        }
        if let Some(tight) = tight {
            inner.tight = tight.inner();
        }
        if let Some(ukf) = ukf_update_options {
            inner.ukf_update_options = ukf.inner();
        }
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// IMU stochastic model used for covariance prediction.
    #[getter]
    fn imu_spec(&self) -> PyImuSpec {
        PyImuSpec {
            inner: self.inner.imu_spec,
        }
    }

    /// Measurement-update algorithm used by loose and tight GNSS updates.
    #[getter]
    fn filter_kind(&self) -> PyFusionFilterKind {
        self.inner.filter_kind.into()
    }

    /// Direction cosine matrix rotating IMU sensor axes into body axes.
    #[getter]
    fn imu_to_body_dcm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.imu_to_body_dcm)
    }

    /// Strapdown mechanization options.
    #[getter]
    fn mechanization(&self) -> PyMechanizationConfig {
        PyMechanizationConfig {
            inner: self.inner.mechanization,
        }
    }

    /// Loose GNSS update options.
    #[getter]
    fn loose(&self) -> PyLooseCouplingConfig {
        PyLooseCouplingConfig {
            inner: self.inner.loose,
        }
    }

    /// Tight raw GNSS update options.
    #[getter]
    fn tight(&self) -> PyTightCouplingConfig {
        PyTightCouplingConfig {
            inner: self.inner.tight,
        }
    }

    /// UKF correction options used when the filter kind is UKF.
    #[getter]
    fn ukf_update_options(&self) -> PyUkfUpdateOptions {
        PyUkfUpdateOptions {
            inner: self.inner.ukf_update_options,
        }
    }
}

/// GNSS PVT measurement used by the loose-coupled INS update.
#[pyclass(module = "sidereon._sidereon", name = "GnssFixMeasurement")]
#[derive(Clone)]
pub struct PyGnssFixMeasurement {
    inner: GnssFixMeasurement,
}

impl PyGnssFixMeasurement {
    fn inner(&self) -> GnssFixMeasurement {
        self.inner.clone()
    }
}

#[pymethods]
impl PyGnssFixMeasurement {
    /// Build a position-only GNSS fix measurement.
    #[staticmethod]
    #[pyo3(signature = (
        t_j2000_s,
        position_ecef_m,
        position_covariance_m2,
        satellites_used,
        *,
        fix_status=PyGnssFixStatus::SINGLE,
    ))]
    fn position(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        position_covariance_m2: PyReadonlyArray2<'_, f64>,
        satellites_used: usize,
        fix_status: PyGnssFixStatus,
    ) -> PyResult<Self> {
        let covariance = matrix3_from_array(
            &position_covariance_m2,
            "position_covariance_m2",
            FinitePolicy::RequireFinite,
        )?;
        GnssFixMeasurement::position(t_j2000_s, position_ecef_m, covariance, satellites_used)
            .map(|inner| inner.with_fix_status(fix_status.into()))
            .map(|inner| Self { inner })
            .map_err(fusion_err)
    }

    /// Build a position and velocity GNSS fix measurement.
    #[staticmethod]
    #[pyo3(signature = (
        t_j2000_s,
        position_ecef_m,
        velocity_ecef_mps,
        covariance,
        satellites_used,
        *,
        fix_status=PyGnssFixStatus::SINGLE,
    ))]
    fn position_velocity(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        velocity_ecef_mps: [f64; 3],
        covariance: PyReadonlyArray2<'_, f64>,
        satellites_used: usize,
        fix_status: PyGnssFixStatus,
    ) -> PyResult<Self> {
        let covariance = square_matrix_from_array(&covariance, "covariance", Some(6))?;
        GnssFixMeasurement::position_velocity(
            t_j2000_s,
            position_ecef_m,
            velocity_ecef_mps,
            covariance,
            satellites_used,
        )
        .map(|inner| inner.with_fix_status(fix_status.into()))
        .map(|inner| Self { inner })
        .map_err(fusion_err)
    }

    /// Return a copy tagged with a different upstream fix status.
    fn with_fix_status(&self, fix_status: PyGnssFixStatus) -> Self {
        Self {
            inner: self.inner.clone().with_fix_status(fix_status.into()),
        }
    }

    /// Measurement epoch in seconds since J2000.
    #[getter]
    fn t_j2000_s(&self) -> f64 {
        self.inner.t_j2000_s
    }

    /// GNSS antenna position in ECEF metres.
    #[getter]
    fn position_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_ecef_m)
    }

    /// Optional GNSS antenna velocity in ECEF metres per second.
    #[getter]
    fn velocity_ecef_mps(&self) -> Option<[f64; 3]> {
        self.inner.velocity_ecef_mps
    }

    /// Measurement covariance in row order.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.covariance)
    }

    /// Number of satellites used by the upstream fix.
    #[getter]
    fn satellites_used(&self) -> usize {
        self.inner.satellites_used
    }

    /// Whether the upstream solver reported a successful fix.
    #[getter]
    fn solution_valid(&self) -> bool {
        self.inner.solution_valid
    }

    /// Upstream ambiguity or code-only fix class.
    #[getter]
    fn fix_status(&self) -> PyGnssFixStatus {
        self.inner.fix_status.into()
    }
}

/// Doppler-derived range-rate row for one satellite in a tight update.
#[pyclass(module = "sidereon._sidereon", name = "TightRangeRateObservation")]
#[derive(Clone, Copy)]
pub struct PyTightRangeRateObservation {
    inner: TightRangeRateObservation,
}

impl PyTightRangeRateObservation {
    fn inner(&self) -> TightRangeRateObservation {
        self.inner
    }
}

#[pymethods]
impl PyTightRangeRateObservation {
    /// Build a tight range-rate observation.
    #[new]
    fn new(
        measured_range_rate_m_s: f64,
        sigma_m_s: f64,
        satellite_clock_drift_m_s: f64,
    ) -> PyResult<Self> {
        let inner = TightRangeRateObservation {
            measured_range_rate_m_s,
            sigma_m_s,
            satellite_clock_drift_m_s,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Measured pseudorange rate in metres per second.
    #[getter]
    fn measured_range_rate_m_s(&self) -> f64 {
        self.inner.measured_range_rate_m_s
    }

    /// One-sigma range-rate uncertainty in metres per second.
    #[getter]
    fn sigma_m_s(&self) -> f64 {
        self.inner.sigma_m_s
    }

    /// Satellite clock drift as an equivalent range-rate bias.
    #[getter]
    fn satellite_clock_drift_m_s(&self) -> f64 {
        self.inner.satellite_clock_drift_m_s
    }
}

/// Carrier-phase range row with a caller-supplied float ambiguity.
#[pyclass(module = "sidereon._sidereon", name = "TightCarrierPhaseObservation")]
#[derive(Clone, Copy)]
pub struct PyTightCarrierPhaseObservation {
    inner: TightCarrierPhaseObservation,
}

impl PyTightCarrierPhaseObservation {
    fn inner(&self) -> TightCarrierPhaseObservation {
        self.inner
    }
}

#[pymethods]
impl PyTightCarrierPhaseObservation {
    /// Build a tight carrier-phase observation.
    #[new]
    fn new(phase_range_m: f64, sigma_m: f64, float_ambiguity_m: f64) -> PyResult<Self> {
        let inner = TightCarrierPhaseObservation {
            phase_range_m,
            sigma_m,
            float_ambiguity_m,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Carrier phase converted to range units in metres.
    #[getter]
    fn phase_range_m(&self) -> f64 {
        self.inner.phase_range_m
    }

    /// One-sigma carrier-phase range uncertainty in metres.
    #[getter]
    fn sigma_m(&self) -> f64 {
        self.inner.sigma_m
    }

    /// Current float ambiguity estimate for this arc, in metres.
    #[getter]
    fn float_ambiguity_m(&self) -> f64 {
        self.inner.float_ambiguity_m
    }
}

/// Raw GNSS observation for one satellite in a tight update.
#[pyclass(module = "sidereon._sidereon", name = "TightGnssObservation")]
#[derive(Clone, Copy)]
pub struct PyTightGnssObservation {
    inner: TightGnssObservation,
}

impl PyTightGnssObservation {
    fn inner(&self) -> TightGnssObservation {
        self.inner
    }
}

#[pymethods]
impl PyTightGnssObservation {
    /// Build a tight raw GNSS observation.
    #[new]
    #[pyo3(signature = (
        satellite_id,
        pseudorange_m,
        pseudorange_sigma_m,
        *,
        range_rate=None,
        carrier_phase=None,
        ionosphere_delay_m=0.0,
        troposphere_delay_m=0.0,
    ))]
    fn new(
        satellite_id: &str,
        pseudorange_m: f64,
        pseudorange_sigma_m: f64,
        range_rate: Option<&PyTightRangeRateObservation>,
        carrier_phase: Option<&PyTightCarrierPhaseObservation>,
        ionosphere_delay_m: f64,
        troposphere_delay_m: f64,
    ) -> PyResult<Self> {
        let inner = TightGnssObservation {
            satellite_id: parse_satellite_id(satellite_id)?,
            pseudorange_m,
            pseudorange_sigma_m,
            range_rate: range_rate.map(PyTightRangeRateObservation::inner),
            carrier_phase: carrier_phase.map(PyTightCarrierPhaseObservation::inner),
            ionosphere_delay_m,
            troposphere_delay_m,
        };
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Satellite identifier as an SP3/RINEX token.
    #[getter]
    fn satellite_id(&self) -> String {
        self.inner.satellite_id.to_string()
    }

    /// Measured code pseudorange in metres.
    #[getter]
    fn pseudorange_m(&self) -> f64 {
        self.inner.pseudorange_m
    }

    /// One-sigma pseudorange uncertainty in metres.
    #[getter]
    fn pseudorange_sigma_m(&self) -> f64 {
        self.inner.pseudorange_sigma_m
    }

    /// Optional Doppler-derived range-rate row.
    #[getter]
    fn range_rate(&self) -> Option<PyTightRangeRateObservation> {
        self.inner
            .range_rate
            .map(|inner| PyTightRangeRateObservation { inner })
    }

    /// Optional carrier-phase row with a caller-supplied ambiguity.
    #[getter]
    fn carrier_phase(&self) -> Option<PyTightCarrierPhaseObservation> {
        self.inner
            .carrier_phase
            .map(|inner| PyTightCarrierPhaseObservation { inner })
    }

    /// Ionospheric code delay correction in metres.
    #[getter]
    fn ionosphere_delay_m(&self) -> f64 {
        self.inner.ionosphere_delay_m
    }

    /// Tropospheric delay correction in metres.
    #[getter]
    fn troposphere_delay_m(&self) -> f64 {
        self.inner.troposphere_delay_m
    }
}

/// One receiver epoch of raw GNSS observations for a tight update.
#[pyclass(module = "sidereon._sidereon", name = "TightGnssEpoch")]
#[derive(Clone)]
pub struct PyTightGnssEpoch {
    inner: TightGnssEpoch,
}

impl PyTightGnssEpoch {
    fn inner(&self) -> TightGnssEpoch {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTightGnssEpoch {
    /// Build a tight raw GNSS epoch.
    #[new]
    fn new(
        py: Python<'_>,
        t_j2000_s: f64,
        observations: Vec<Py<PyTightGnssObservation>>,
    ) -> PyResult<Self> {
        let observations = observations
            .iter()
            .map(|observation| observation.borrow(py).inner())
            .collect();
        TightGnssEpoch::new(t_j2000_s, observations)
            .map(|inner| Self { inner })
            .map_err(fusion_err)
    }

    /// Measurement epoch in seconds since J2000.
    #[getter]
    fn t_j2000_s(&self) -> f64 {
        self.inner.t_j2000_s
    }

    /// Number of satellite observations in this epoch.
    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.observations.len()
    }
}

/// Diagnostics from an innovation screen.
#[pyclass(module = "sidereon._sidereon", name = "InnovationGateReport")]
#[derive(Clone)]
pub struct PyInnovationGateReport {
    inner: InnovationGateReport,
}

impl From<InnovationGateReport> for PyInnovationGateReport {
    fn from(inner: InnovationGateReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyInnovationGateReport {
    /// Rejection threshold in sigma.
    #[getter]
    fn threshold_sigma(&self) -> f64 {
        self.inner.threshold_sigma
    }

    /// Minimum accepted rows requested by the gate.
    #[getter]
    fn min_rows(&self) -> usize {
        self.inner.min_rows
    }

    /// Number of input measurement rows.
    #[getter]
    fn input_rows(&self) -> usize {
        self.inner.input_rows
    }

    /// Number of accepted rows.
    #[getter]
    fn accepted_rows(&self) -> usize {
        self.inner.accepted_rows
    }

    /// Number of rejected rows.
    #[getter]
    fn rejected_rows(&self) -> usize {
        self.inner.rejected_rows
    }

    /// Largest accepted absolute normalized innovation, if any row was accepted.
    #[getter]
    fn max_abs_normalized_innovation(&self) -> Option<f64> {
        self.inner.max_abs_normalized_innovation
    }

    /// Largest rejected absolute normalized innovation, if any row was rejected.
    #[getter]
    fn max_rejected_abs_normalized_innovation(&self) -> Option<f64> {
        self.inner.max_rejected_abs_normalized_innovation
    }

    /// Whether too few rows remained to apply the update.
    #[getter]
    fn coasted(&self) -> bool {
        self.inner.coasted
    }
}

/// Diagnostics from one EKF or UKF correction attempt.
#[pyclass(module = "sidereon._sidereon", name = "EkfCorrectionReport")]
#[derive(Clone)]
pub struct PyEkfCorrectionReport {
    inner: EkfCorrectionReport,
}

impl From<EkfCorrectionReport> for PyEkfCorrectionReport {
    fn from(inner: EkfCorrectionReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEkfCorrectionReport {
    /// Whether the correction was applied.
    #[getter]
    fn applied(&self) -> bool {
        self.inner.applied
    }

    /// Normalized innovation squared.
    #[getter]
    fn normalized_innovation_squared(&self) -> f64 {
        self.inner.normalized_innovation_squared
    }

    /// Number of accepted rows.
    #[getter]
    fn accepted_rows(&self) -> usize {
        self.inner.accepted_rows
    }

    /// Number of rejected rows.
    #[getter]
    fn rejected_rows(&self) -> usize {
        self.inner.rejected_rows
    }

    /// Optional innovation gate diagnostics.
    #[getter]
    fn innovation_gate(&self) -> Option<PyInnovationGateReport> {
        self.inner.innovation_gate.clone().map(Into::into)
    }

    /// Innovation covariance matrix.
    #[getter]
    fn innovation_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.innovation_covariance)
    }

    /// Kalman gain matrix.
    #[getter]
    fn kalman_gain<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.kalman_gain)
    }

    /// Error-state estimate applied to the closed-loop nominal state.
    #[getter]
    fn dx<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.dx)
    }
}

/// Result of one fusion update.
#[pyclass(module = "sidereon._sidereon", name = "FusionUpdate")]
#[derive(Clone)]
pub struct PyFusionUpdate {
    inner: FusionUpdate,
}

impl From<FusionUpdate> for PyFusionUpdate {
    fn from(inner: FusionUpdate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFusionUpdate {
    /// Whether the update modified the state.
    #[getter]
    fn applied(&self) -> bool {
        self.inner.applied
    }

    /// Normalized innovation squared.
    #[getter]
    fn nis(&self) -> f64 {
        self.inner.nis
    }

    /// Number of measurement rows.
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    /// Number of rows accepted by screening.
    #[getter]
    fn accepted_rows(&self) -> usize {
        self.inner.accepted_rows
    }

    /// Number of rows rejected by screening.
    #[getter]
    fn rejected_rows(&self) -> usize {
        self.inner.rejected_rows
    }

    /// Detailed correction report.
    #[getter]
    fn ekf(&self) -> PyEkfCorrectionReport {
        self.inner.ekf.clone().into()
    }
}

/// Receiver-clock state reported by the tight filter.
#[pyclass(module = "sidereon._sidereon", name = "TightClockState")]
#[derive(Clone, Copy)]
pub struct PyTightClockState {
    inner: TightClockState,
}

impl From<TightClockState> for PyTightClockState {
    fn from(inner: TightClockState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTightClockState {
    /// Receiver-clock range bias in metres.
    #[getter]
    fn bias_m(&self) -> f64 {
        self.inner.bias_m
    }

    /// Receiver-clock drift in metres per second.
    #[getter]
    fn drift_m_s(&self) -> f64 {
        self.inner.drift_m_s
    }

    /// Two-by-two clock covariance matrix.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows = [
            [self.inner.covariance[0][0], self.inner.covariance[0][1]],
            [self.inner.covariance[1][0], self.inner.covariance[1][1]],
        ];
        let mut array = Array2::<f64>::zeros((2, 2));
        for row in 0..2 {
            for col in 0..2 {
                array[[row, col]] = rows[row][col];
            }
        }
        PyArray2::from_owned_array(py, array)
    }
}

/// Retained-history limits for bounded-latency time synchronization.
#[pyclass(module = "sidereon._sidereon", name = "TimeSyncHistoryConfig")]
#[derive(Clone, Copy)]
pub struct PyTimeSyncHistoryConfig {
    inner: TimeSyncHistoryConfig,
}

impl PyTimeSyncHistoryConfig {
    fn inner(&self) -> TimeSyncHistoryConfig {
        self.inner
    }
}

#[pymethods]
impl PyTimeSyncHistoryConfig {
    /// Build retained-history limits for time-synchronized replay.
    #[new]
    #[pyo3(signature = (imu_capacity=256, checkpoint_capacity=64))]
    fn new(imu_capacity: usize, checkpoint_capacity: usize) -> PyResult<Self> {
        let inner = TimeSyncHistoryConfig::new(imu_capacity, checkpoint_capacity);
        inner.validate().map_err(fusion_err)?;
        Ok(Self { inner })
    }

    /// Number of recent IMU samples retained for replay.
    #[getter]
    fn imu_capacity(&self) -> usize {
        self.inner.imu_capacity
    }

    /// Number of recent filter checkpoints retained at GNSS epochs.
    #[getter]
    fn checkpoint_capacity(&self) -> usize {
        self.inner.checkpoint_capacity
    }
}

/// Current retained-history occupancy for time synchronization.
#[pyclass(module = "sidereon._sidereon", name = "TimeSyncHistoryStatus")]
#[derive(Clone, Copy)]
pub struct PyTimeSyncHistoryStatus {
    inner: TimeSyncHistoryStatus,
}

impl From<TimeSyncHistoryStatus> for PyTimeSyncHistoryStatus {
    fn from(inner: TimeSyncHistoryStatus) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTimeSyncHistoryStatus {
    /// Configured IMU sample capacity.
    #[getter]
    fn imu_capacity(&self) -> usize {
        self.inner.imu_capacity
    }

    /// Number of retained IMU samples.
    #[getter]
    fn imu_len(&self) -> usize {
        self.inner.imu_len
    }

    /// Configured checkpoint capacity.
    #[getter]
    fn checkpoint_capacity(&self) -> usize {
        self.inner.checkpoint_capacity
    }

    /// Number of retained filter checkpoints.
    #[getter]
    fn checkpoint_len(&self) -> usize {
        self.inner.checkpoint_len
    }

    /// Oldest retained IMU sample end epoch, if any.
    #[getter]
    fn oldest_imu_epoch_j2000_s(&self) -> Option<f64> {
        self.inner.oldest_imu_epoch_j2000_s
    }

    /// Newest retained IMU sample end epoch, if any.
    #[getter]
    fn newest_imu_epoch_j2000_s(&self) -> Option<f64> {
        self.inner.newest_imu_epoch_j2000_s
    }

    /// Oldest retained checkpoint epoch, if any.
    #[getter]
    fn oldest_checkpoint_epoch_j2000_s(&self) -> Option<f64> {
        self.inner.oldest_checkpoint_epoch_j2000_s
    }

    /// Newest retained checkpoint epoch, if any.
    #[getter]
    fn newest_checkpoint_epoch_j2000_s(&self) -> Option<f64> {
        self.inner.newest_checkpoint_epoch_j2000_s
    }
}

/// Result of a time-synchronized GNSS update.
#[pyclass(module = "sidereon._sidereon", name = "TimeSyncUpdate")]
#[derive(Clone)]
pub struct PyTimeSyncUpdate {
    inner: TimeSyncUpdate,
}

impl From<TimeSyncUpdate> for PyTimeSyncUpdate {
    fn from(inner: TimeSyncUpdate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTimeSyncUpdate {
    /// Update report for the newly supplied measurement.
    #[getter]
    fn update(&self) -> PyFusionUpdate {
        self.inner.update.clone().into()
    }

    /// Whether the measurement epoch was older than the current propagated epoch.
    #[getter]
    fn late_measurement(&self) -> bool {
        self.inner.late_measurement
    }

    /// Number of IMU segments replayed while applying the measurement.
    #[getter]
    fn replayed_imu_segments(&self) -> usize {
        self.inner.replayed_imu_segments
    }

    /// Checkpoint epoch used as the replay start.
    #[getter]
    fn restored_checkpoint_epoch_j2000_s(&self) -> f64 {
        self.inner.restored_checkpoint_epoch_j2000_s
    }

    /// Filter epoch after replay is complete.
    #[getter]
    fn current_epoch_j2000_s(&self) -> f64 {
        self.inner.current_epoch_j2000_s
    }
}

/// Snapshot of tight receiver-clock augmentation and augmented covariance.
#[pyclass(module = "sidereon._sidereon", name = "TightFilterSnapshot")]
#[derive(Clone)]
pub struct PyTightFilterSnapshot {
    inner: TightFilterSnapshot,
}

impl From<TightFilterSnapshot> for PyTightFilterSnapshot {
    fn from(inner: TightFilterSnapshot) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTightFilterSnapshot {
    /// Receiver-clock range bias in metres.
    #[getter]
    fn clock_bias_m(&self) -> f64 {
        self.inner.clock_bias_m
    }

    /// Receiver-clock drift in metres per second.
    #[getter]
    fn clock_drift_m_s(&self) -> f64 {
        self.inner.clock_drift_m_s
    }

    /// Full covariance over INS error-state and tight clock states.
    #[getter]
    fn augmented_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.augmented_covariance)
    }
}

/// Closed-loop inertial filter checkpoint used by replay and RTS smoothing.
#[pyclass(module = "sidereon._sidereon", name = "InertialFilterSnapshot")]
#[derive(Clone)]
pub struct PyInertialFilterSnapshot {
    inner: InertialFilterSnapshot,
}

impl From<InertialFilterSnapshot> for PyInertialFilterSnapshot {
    fn from(inner: InertialFilterSnapshot) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyInertialFilterSnapshot {
    /// Closed-loop INS state and covariance.
    #[getter]
    fn state(&self) -> PyInsFilterState {
        self.inner.state.clone().into()
    }

    /// Most recent body angular rate relative to ECEF in body axes.
    #[getter]
    fn last_body_rate_wrt_ecef_rps(&self) -> [f64; 3] {
        self.inner.last_body_rate_wrt_ecef_rps
    }

    /// Tight receiver-clock checkpoint.
    #[getter]
    fn tight(&self) -> PyTightFilterSnapshot {
        self.inner.tight.clone().into()
    }
}

/// One epoch in a recorded fusion RTS history.
#[pyclass(module = "sidereon._sidereon", name = "FusionRtsEpoch")]
#[derive(Clone)]
pub struct PyFusionRtsEpoch {
    inner: FusionRtsEpoch,
}

impl From<FusionRtsEpoch> for PyFusionRtsEpoch {
    fn from(inner: FusionRtsEpoch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFusionRtsEpoch {
    /// Epoch in seconds since J2000.
    #[getter]
    fn t_j2000_s(&self) -> f64 {
        self.inner.t_j2000_s
    }

    /// Pre-update checkpoint at this epoch.
    #[getter]
    fn predicted(&self) -> PyInertialFilterSnapshot {
        self.inner.predicted.clone().into()
    }

    /// Post-update checkpoint at this epoch.
    #[getter]
    fn updated(&self) -> PyInertialFilterSnapshot {
        self.inner.updated.clone().into()
    }

    /// Error-state transition from the previous updated epoch, if present.
    #[getter]
    fn transition_from_previous<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .transition_from_previous
            .as_ref()
            .map(|transition| matrix_to_array(py, transition))
    }

    fn __repr__(&self) -> String {
        format!("FusionRtsEpoch(t_j2000_s={})", self.inner.t_j2000_s)
    }
}

/// Recorded forward-pass history accepted by `smooth_fusion_rts`.
#[pyclass(module = "sidereon._sidereon", name = "FusionRtsHistory")]
#[derive(Clone)]
pub struct PyFusionRtsHistory {
    inner: FusionRtsHistory,
}

impl From<FusionRtsHistory> for PyFusionRtsHistory {
    fn from(inner: FusionRtsHistory) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFusionRtsHistory {
    /// Recorded epochs in forward time order.
    #[getter]
    fn epochs(&self) -> Vec<PyFusionRtsEpoch> {
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
        format!("FusionRtsHistory(epochs={})", self.inner.epochs.len())
    }
}

/// Builder for recording a forward fusion pass before RTS smoothing.
#[pyclass(module = "sidereon._sidereon", name = "FusionRtsHistoryBuilder")]
#[derive(Clone)]
pub struct PyFusionRtsHistoryBuilder {
    inner: FusionRtsHistoryBuilder,
}

#[pymethods]
impl PyFusionRtsHistoryBuilder {
    /// Start an empty history for manual recording.
    #[new]
    fn new() -> Self {
        Self {
            inner: FusionRtsHistoryBuilder::empty(),
        }
    }

    /// Start a history from the filter's current checkpoint.
    #[staticmethod]
    fn from_filter(filter: &PyInertialFilter) -> PyResult<Self> {
        FusionRtsHistoryBuilder::from_filter(&filter.inner)
            .map(|inner| Self { inner })
            .map_err(fusion_err)
    }

    /// Return a validated history.
    fn finish(&self) -> PyResult<PyFusionRtsHistory> {
        self.inner
            .clone()
            .finish()
            .map(Into::into)
            .map_err(fusion_err)
    }

    fn __repr__(&self) -> &'static str {
        "FusionRtsHistoryBuilder()"
    }
}

/// One epoch in a smoothed fusion trajectory.
#[pyclass(module = "sidereon._sidereon", name = "SmoothedFusionEpoch")]
#[derive(Clone)]
pub struct PySmoothedFusionEpoch {
    inner: SmoothedFusionEpoch,
}

impl From<SmoothedFusionEpoch> for PySmoothedFusionEpoch {
    fn from(inner: SmoothedFusionEpoch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySmoothedFusionEpoch {
    /// Epoch in seconds since J2000.
    #[getter]
    fn t_j2000_s(&self) -> f64 {
        self.inner.t_j2000_s
    }

    /// Smoothed closed-loop checkpoint.
    #[getter]
    fn snapshot(&self) -> PyInertialFilterSnapshot {
        self.inner.snapshot.clone().into()
    }

    /// Error-state and tight-clock correction applied by the smoother.
    #[getter]
    fn error_state_correction<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.error_state_correction)
    }

    /// Smoothed covariance over the INS error-state and tight clock states.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_array(py, &self.inner.covariance)
    }

    /// RTS gain from this epoch to the next, absent at the final epoch.
    #[getter]
    fn rts_gain_to_next<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .rts_gain_to_next
            .as_ref()
            .map(|gain| matrix_to_array(py, gain))
    }

    fn __repr__(&self) -> String {
        format!("SmoothedFusionEpoch(t_j2000_s={})", self.inner.t_j2000_s)
    }
}

/// Smoothed fusion trajectory returned by fixed-interval RTS smoothing.
#[pyclass(module = "sidereon._sidereon", name = "SmoothedFusionTrajectory")]
#[derive(Clone)]
pub struct PySmoothedFusionTrajectory {
    inner: SmoothedFusionTrajectory,
}

impl From<SmoothedFusionTrajectory> for PySmoothedFusionTrajectory {
    fn from(inner: SmoothedFusionTrajectory) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySmoothedFusionTrajectory {
    /// Smoothed epochs in the same order as the recorded history.
    #[getter]
    fn epochs(&self) -> Vec<PySmoothedFusionEpoch> {
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
        format!(
            "SmoothedFusionTrajectory(epochs={})",
            self.inner.epochs.len()
        )
    }
}

/// Apply fixed-interval RTS smoothing to recorded fusion history.
#[pyfunction]
fn smooth_fusion_rts(history: &PyFusionRtsHistory) -> PyResult<PySmoothedFusionTrajectory> {
    core_smooth_fusion_rts(&history.inner)
        .map(Into::into)
        .map_err(fusion_err)
}

/// Stateful closed-loop INS filter with loose and tight GNSS update methods.
#[pyclass(module = "sidereon._sidereon", name = "InertialFilter")]
#[derive(Clone)]
pub struct PyInertialFilter {
    inner: InertialFilter,
}

#[pymethods]
impl PyInertialFilter {
    /// Build a filter with default calibration and loose-coupling settings.
    #[new]
    fn new(state: &PyInsFilterState, imu_spec: &PyImuSpec) -> PyResult<Self> {
        InertialFilter::new(state.inner(), imu_spec.inner())
            .map(|inner| Self { inner })
            .map_err(fusion_err)
    }

    /// Build a filter with explicit configuration.
    #[staticmethod]
    fn with_config(state: &PyInsFilterState, config: &PyInertialFilterConfig) -> PyResult<Self> {
        InertialFilter::with_config(state.inner(), config.inner())
            .map(|inner| Self { inner })
            .map_err(fusion_err)
    }

    /// Decode a filter from versioned checkpoint bytes and a compatible config.
    #[staticmethod]
    fn from_encoded_state(
        payload: &Bound<'_, PyAny>,
        config: &PyInertialFilterConfig,
    ) -> PyResult<Self> {
        let bytes = bytes_from_source(payload, "from_encoded_state")?;
        let state = SerializableFusionState::decode_versioned(&bytes).map_err(fusion_err)?;
        let snapshot = state.to_snapshot().map_err(fusion_err)?;
        let mut filter = InertialFilter::with_config(snapshot.state.clone(), config.inner())
            .map_err(fusion_err)?;
        filter
            .restore_serializable_state(&state)
            .map_err(fusion_err)?;
        Ok(Self { inner: filter })
    }

    /// Current INS filter state.
    #[getter]
    fn state(&self) -> PyInsFilterState {
        self.inner.state().clone().into()
    }

    /// Immutable filter configuration.
    #[getter]
    fn config(&self) -> PyInertialFilterConfig {
        PyInertialFilterConfig {
            inner: *self.inner.config(),
        }
    }

    /// Most recent body angular rate relative to ECEF in body axes.
    #[getter]
    fn last_body_rate_wrt_ecef_rps(&self) -> [f64; 3] {
        self.inner.last_body_rate_wrt_ecef_rps()
    }

    /// Capture a closed-loop checkpoint for inspection or later restore.
    fn snapshot(&self) -> PyInertialFilterSnapshot {
        self.inner.snapshot().into()
    }

    /// Restore the filter from a closed-loop checkpoint.
    fn restore_snapshot(&mut self, snapshot: &PyInertialFilterSnapshot) -> PyResult<()> {
        self.inner
            .restore_snapshot(&snapshot.inner)
            .map_err(fusion_err)
    }

    /// Propagate the nominal INS state and covariance with one IMU sample.
    fn propagate(&mut self, sample: &PyImuSample) -> PyResult<PyInsFilterState> {
        self.inner.propagate(sample.inner()).map_err(fusion_err)?;
        Ok(self.inner.state().clone().into())
    }

    /// Propagate and record the transition for later RTS smoothing.
    fn propagate_recorded(
        &mut self,
        sample: &PyImuSample,
        history: &mut PyFusionRtsHistoryBuilder,
    ) -> PyResult<PyInsFilterState> {
        self.inner
            .propagate_recorded(sample.inner(), &mut history.inner)
            .map_err(fusion_err)?;
        Ok(self.inner.state().clone().into())
    }

    /// Apply a loose GNSS PVT update at the current propagated epoch.
    fn update_loose(&mut self, measurement: &PyGnssFixMeasurement) -> PyResult<PyFusionUpdate> {
        self.inner
            .update_loose(&measurement.inner())
            .map(Into::into)
            .map_err(fusion_err)
    }

    /// Apply a loose GNSS update and record checkpoints for RTS smoothing.
    fn update_loose_recorded(
        &mut self,
        measurement: &PyGnssFixMeasurement,
        history: &mut PyFusionRtsHistoryBuilder,
    ) -> PyResult<PyFusionUpdate> {
        self.inner
            .update_loose_recorded(&measurement.inner(), &mut history.inner)
            .map(Into::into)
            .map_err(fusion_err)
    }

    /// Apply a loose GNSS update at the measurement epoch, replaying history if needed.
    fn update_loose_time_sync(
        &mut self,
        measurement: &PyGnssFixMeasurement,
    ) -> PyResult<PyTimeSyncUpdate> {
        self.inner
            .update_loose_time_sync(&measurement.inner())
            .map(Into::into)
            .map_err(fusion_err)
    }

    /// Apply a gated zero-velocity and zero-angular-rate update.
    fn update_stationary(&mut self) -> PyResult<Option<PyFusionUpdate>> {
        self.inner
            .update_stationary()
            .map(|update| update.map(Into::into))
            .map_err(fusion_err)
    }

    /// Apply a stationary update and record checkpoints when an update applies.
    fn update_stationary_recorded(
        &mut self,
        history: &mut PyFusionRtsHistoryBuilder,
    ) -> PyResult<Option<PyFusionUpdate>> {
        self.inner
            .update_stationary_recorded(&mut history.inner)
            .map(|update| update.map(Into::into))
            .map_err(fusion_err)
    }

    /// Apply a gated wheeled-vehicle non-holonomic constraint update.
    fn update_non_holonomic(&mut self) -> PyResult<Option<PyFusionUpdate>> {
        self.inner
            .update_non_holonomic()
            .map(|update| update.map(Into::into))
            .map_err(fusion_err)
    }

    /// Apply a non-holonomic constraint and record checkpoints when an update applies.
    fn update_non_holonomic_recorded(
        &mut self,
        history: &mut PyFusionRtsHistoryBuilder,
    ) -> PyResult<Option<PyFusionUpdate>> {
        self.inner
            .update_non_holonomic_recorded(&mut history.inner)
            .map(|update| update.map(Into::into))
            .map_err(fusion_err)
    }

    /// Apply a tight raw GNSS update at the current propagated epoch.
    fn update_tight(
        &mut self,
        source: &Bound<'_, PyAny>,
        epoch: &PyTightGnssEpoch,
    ) -> PyResult<PyFusionUpdate> {
        with_observable_source(source, |source| {
            self.inner
                .update_tight(source, &epoch.inner())
                .map(Into::into)
                .map_err(fusion_err)
        })
    }

    /// Apply a tight raw GNSS update and record checkpoints for RTS smoothing.
    fn update_tight_recorded(
        &mut self,
        source: &Bound<'_, PyAny>,
        epoch: &PyTightGnssEpoch,
        history: &mut PyFusionRtsHistoryBuilder,
    ) -> PyResult<PyFusionUpdate> {
        with_observable_source(source, |source| {
            self.inner
                .update_tight_recorded(source, &epoch.inner(), &mut history.inner)
                .map(Into::into)
                .map_err(fusion_err)
        })
    }

    /// Apply a tight raw GNSS update at the measurement epoch, replaying history if needed.
    fn update_tight_time_sync(
        &mut self,
        source: &Bound<'_, PyAny>,
        epoch: &PyTightGnssEpoch,
    ) -> PyResult<PyTimeSyncUpdate> {
        with_observable_source(source, |source| {
            self.inner
                .update_tight_time_sync(source, &epoch.inner())
                .map(Into::into)
                .map_err(fusion_err)
        })
    }

    /// Borrow the current receiver-clock state carried by tight coupling.
    fn tight_clock_state(&self) -> PyResult<PyTightClockState> {
        self.inner
            .tight_clock_state()
            .map(Into::into)
            .map_err(fusion_err)
    }

    /// Replace retained-history capacities for later time-sync replay.
    fn configure_time_sync_history(&mut self, config: &PyTimeSyncHistoryConfig) -> PyResult<()> {
        self.inner
            .configure_time_sync_history(config.inner())
            .map_err(fusion_err)
    }

    /// Return current retained-history capacity and occupancy.
    fn time_sync_history_status(&self) -> PyTimeSyncHistoryStatus {
        self.inner.time_sync_history_status().into()
    }

    /// Encode the current filter checkpoint with the versioned binary codec.
    fn encode_state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.encode_state().map_err(fusion_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Restore this filter from versioned checkpoint bytes.
    fn restore_encoded_state(&mut self, payload: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = bytes_from_source(payload, "restore_encoded_state")?;
        self.inner.restore_encoded_state(&bytes).map_err(fusion_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "InertialFilter(t_j2000_s={})",
            self.inner.state().nominal.t_j2000_s
        )
    }
}

/// Blend a first good post-outage fix back over an outage span.
#[pyfunction]
fn velocity_match_outage(
    py: Python<'_>,
    states: Vec<Py<PyVelocityMatchState>>,
    first_good_fix: &PyGnssFixMeasurement,
    config: &PyVelocityMatchingConfig,
) -> PyResult<PyVelocityMatchedTrajectory> {
    let states = states
        .iter()
        .map(|state| state.borrow(py).inner())
        .collect::<Vec<_>>();
    core_velocity_match_outage(&states, &first_good_fix.inner(), config.inner())
        .map(Into::into)
        .map_err(fusion_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyImuGrade>()?;
    m.add_class::<PyImuSpec>()?;
    m.add_class::<PyMechanizationConfig>()?;
    m.add_class::<PyFusionFilterKind>()?;
    m.add_class::<PyErrorStateLayout>()?;
    m.add_class::<PyNavState>()?;
    m.add_class::<PyImuSample>()?;
    m.add_class::<PyInsFilterState>()?;
    m.add_class::<PyInnovationGate>()?;
    m.add_class::<PyEkfUpdateOptions>()?;
    m.add_class::<PyUnscentedTransformOptions>()?;
    m.add_class::<PyUkfUpdateOptions>()?;
    m.add_class::<PyGnssFixStatus>()?;
    m.add_class::<PyGnssFixStatusWeighting>()?;
    m.add_class::<PyIggIiiMeasurementReweighting>()?;
    m.add_class::<PyYangPredictionAdaptiveFactor>()?;
    m.add_class::<PyStationaryDetectorConfig>()?;
    m.add_class::<PyStationaryUpdateConfig>()?;
    m.add_class::<PyNonHolonomicConstraintConfig>()?;
    m.add_class::<PyVelocityMatchingConfig>()?;
    m.add_class::<PyVelocityMatchState>()?;
    m.add_class::<PyVelocityMatchedTrajectory>()?;
    m.add_class::<PyLooseCouplingConfig>()?;
    m.add_class::<PyTightCouplingConfig>()?;
    m.add_class::<PyInertialFilterConfig>()?;
    m.add_class::<PyGnssFixMeasurement>()?;
    m.add_class::<PyTightRangeRateObservation>()?;
    m.add_class::<PyTightCarrierPhaseObservation>()?;
    m.add_class::<PyTightGnssObservation>()?;
    m.add_class::<PyTightGnssEpoch>()?;
    m.add_class::<PyInnovationGateReport>()?;
    m.add_class::<PyEkfCorrectionReport>()?;
    m.add_class::<PyFusionUpdate>()?;
    m.add_class::<PyTightClockState>()?;
    m.add_class::<PyTimeSyncHistoryConfig>()?;
    m.add_class::<PyTimeSyncHistoryStatus>()?;
    m.add_class::<PyTimeSyncUpdate>()?;
    m.add_class::<PyTightFilterSnapshot>()?;
    m.add_class::<PyInertialFilterSnapshot>()?;
    m.add_class::<PyFusionRtsEpoch>()?;
    m.add_class::<PyFusionRtsHistory>()?;
    m.add_class::<PyFusionRtsHistoryBuilder>()?;
    m.add_class::<PySmoothedFusionEpoch>()?;
    m.add_class::<PySmoothedFusionTrajectory>()?;
    m.add_class::<PyInertialFilter>()?;
    m.add_function(wrap_pyfunction!(smooth_fusion_rts, m)?)?;
    m.add_function(wrap_pyfunction!(velocity_match_outage, m)?)?;
    Ok(())
}
