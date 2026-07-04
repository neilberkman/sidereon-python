//! Source-localization bindings.
//!
//! Wraps `sidereon_core::source_localization`: sensors in caller-chosen 2D or
//! 3D Cartesian metres, arrival times in seconds, propagation speeds in metres
//! per second, and covariance/CRLB outputs in the corresponding units.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::source_localization::{
    chan_ho_initial_guess as core_chan_ho_initial_guess, locate_source as core_locate_source,
    source_crlb as core_source_crlb, source_dop as core_source_dop, Loss, Sensor, SourceCovariance,
    SourceCrlb, SourceInitialGuess, SourceLocalizationError as CoreSourceLocalizationError,
    SourceLocateOptions, SourceResidual, SourceSensorInfluence, SourceSolution, SourceSolveMode,
};

use crate::events::PyDop;
use crate::geometry_quality::PyGeometryQuality;
use crate::propagation::PyLoss;
use crate::SourceLocalizationError;

fn source_error(err: CoreSourceLocalizationError) -> PyErr {
    SourceLocalizationError::new_err(err.to_string())
}

fn array_from_vec<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, values)
}

fn matrix_from_rows<'py>(
    py: Python<'py>,
    rows: &[Vec<f64>],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cols = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != cols) {
        return Err(SourceLocalizationError::new_err(
            "source covariance rows must have equal length",
        ));
    }
    let mut array = Array2::<f64>::zeros((rows.len(), cols));
    for (row_index, row) in rows.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            array[[row_index, col_index]] = *value;
        }
    }
    Ok(PyArray2::from_owned_array(py, array))
}

fn loss_from_py(loss: PyLoss) -> Loss {
    loss.to_trf_loss()
}

fn loss_to_py(loss: Loss) -> PyLoss {
    match loss {
        Loss::Linear => PyLoss::LINEAR,
        Loss::SoftL1 => PyLoss::SOFT_L1,
        Loss::Huber => PyLoss::HUBER,
        Loss::Cauchy => PyLoss::CAUCHY,
        Loss::Arctan => PyLoss::ARCTAN,
    }
}

fn loss_label(loss: Loss) -> &'static str {
    match loss {
        Loss::Linear => "linear",
        Loss::SoftL1 => "soft_l1",
        Loss::Huber => "huber",
        Loss::Cauchy => "cauchy",
        Loss::Arctan => "arctan",
    }
}

fn sensors_to_core(py: Python<'_>, sensors: &[Py<PySensor>]) -> Vec<Sensor> {
    sensors
        .iter()
        .map(|sensor| sensor.borrow(py).inner.clone())
        .collect()
}

/// A known sensor position for source localization.
#[pyclass(module = "sidereon._sidereon", name = "Sensor")]
#[derive(Clone)]
pub struct PySensor {
    inner: Sensor,
}

impl From<Sensor> for PySensor {
    fn from(inner: Sensor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySensor {
    /// Create a sensor.
    ///
    /// `position_m` is a 2D or 3D Cartesian position in metres. When
    /// `propagation_speed_m_s` is set, that speed overrides the call-level speed
    /// for this sensor's timing residual.
    #[new]
    #[pyo3(signature = (position_m, propagation_speed_m_s=None))]
    fn new(position_m: Vec<f64>, propagation_speed_m_s: Option<f64>) -> Self {
        let inner = match propagation_speed_m_s {
            Some(speed) => Sensor::with_speed(position_m, speed),
            None => Sensor::new(position_m),
        };
        Self { inner }
    }

    /// Sensor position in caller-chosen Cartesian metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array_from_vec(py, &self.inner.position_m)
    }

    /// Optional per-sensor propagation speed, metres per second.
    #[getter]
    fn propagation_speed_m_s(&self) -> Option<f64> {
        self.inner.propagation_speed_m_s
    }

    fn __repr__(&self) -> String {
        format!(
            "Sensor(position_m={:?}, propagation_speed_m_s={:?})",
            self.inner.position_m, self.inner.propagation_speed_m_s
        )
    }
}

/// Source-localization residual mode.
#[pyclass(module = "sidereon._sidereon", name = "SourceSolveMode")]
#[derive(Clone, Copy)]
pub struct PySourceSolveMode {
    inner: SourceSolveMode,
}

impl From<SourceSolveMode> for PySourceSolveMode {
    fn from(inner: SourceSolveMode) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceSolveMode {
    /// Create the default absolute time-of-arrival mode.
    #[new]
    fn new() -> Self {
        Self {
            inner: SourceSolveMode::Toa,
        }
    }

    /// Absolute time-of-arrival mode.
    #[staticmethod]
    fn toa() -> Self {
        Self {
            inner: SourceSolveMode::Toa,
        }
    }

    /// Time-difference-of-arrival mode using a reference sensor index.
    #[staticmethod]
    fn tdoa(reference_sensor: usize) -> Self {
        Self {
            inner: SourceSolveMode::Tdoa { reference_sensor },
        }
    }

    /// Stable mode label, either `"toa"` or `"tdoa"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            SourceSolveMode::Toa => "toa",
            SourceSolveMode::Tdoa { .. } => "tdoa",
        }
    }

    /// Reference sensor index for TDOA, or `None` for ToA.
    #[getter]
    fn reference_sensor(&self) -> Option<usize> {
        match self.inner {
            SourceSolveMode::Toa => None,
            SourceSolveMode::Tdoa { reference_sensor } => Some(reference_sensor),
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            SourceSolveMode::Toa => "SourceSolveMode.toa()".to_string(),
            SourceSolveMode::Tdoa { reference_sensor } => {
                format!("SourceSolveMode.tdoa(reference_sensor={reference_sensor})")
            }
        }
    }
}

/// Options for source localization.
#[pyclass(module = "sidereon._sidereon", name = "SourceLocateOptions")]
#[derive(Clone)]
pub struct PySourceLocateOptions {
    inner: SourceLocateOptions,
}

impl From<SourceLocateOptions> for PySourceLocateOptions {
    fn from(inner: SourceLocateOptions) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceLocateOptions {
    /// Create source-localization solver options.
    ///
    /// `timing_sigma_s` and `f_scale_s` are seconds. The optional tolerances and
    /// `max_nfev` are passed to the trust-region least-squares solver.
    #[new]
    #[pyo3(signature = (
        mode=None,
        timing_sigma_s=1.0,
        loss=PyLoss::LINEAR,
        f_scale_s=1.0,
        ftol=None,
        xtol=None,
        gtol=None,
        max_nfev=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        mode: Option<Py<PySourceSolveMode>>,
        timing_sigma_s: f64,
        loss: PyLoss,
        f_scale_s: f64,
        ftol: Option<f64>,
        xtol: Option<f64>,
        gtol: Option<f64>,
        max_nfev: Option<usize>,
    ) -> Self {
        let mode = mode
            .as_ref()
            .map(|mode| mode.borrow(py).inner)
            .unwrap_or(SourceSolveMode::Toa);
        Self {
            inner: SourceLocateOptions {
                mode,
                timing_sigma_s,
                loss: loss_from_py(loss),
                f_scale_s,
                ftol,
                xtol,
                gtol,
                max_nfev,
            },
        }
    }

    /// ToA or TDOA residual mode.
    #[getter]
    fn mode(&self) -> PySourceSolveMode {
        self.inner.mode.into()
    }

    /// Timing standard deviation used for covariance and influence scores.
    #[getter]
    fn timing_sigma_s(&self) -> f64 {
        self.inner.timing_sigma_s
    }

    /// Robust loss function for the trust-region solver.
    #[getter]
    fn loss(&self) -> PyLoss {
        loss_to_py(self.inner.loss)
    }

    /// Robust residual scale, seconds.
    #[getter]
    fn f_scale_s(&self) -> f64 {
        self.inner.f_scale_s
    }

    /// Optional function tolerance.
    #[getter]
    fn ftol(&self) -> Option<f64> {
        self.inner.ftol
    }

    /// Optional step tolerance.
    #[getter]
    fn xtol(&self) -> Option<f64> {
        self.inner.xtol
    }

    /// Optional gradient tolerance.
    #[getter]
    fn gtol(&self) -> Option<f64> {
        self.inner.gtol
    }

    /// Optional maximum residual evaluations.
    #[getter]
    fn max_nfev(&self) -> Option<usize> {
        self.inner.max_nfev
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceLocateOptions(mode={}, timing_sigma_s={}, loss={}, f_scale_s={})",
            self.mode().__repr__(),
            self.inner.timing_sigma_s,
            loss_label(self.inner.loss),
            self.inner.f_scale_s
        )
    }
}

/// Closed-form initial guess used by the source-localization solver.
#[pyclass(module = "sidereon._sidereon", name = "SourceInitialGuess")]
#[derive(Clone)]
pub struct PySourceInitialGuess {
    inner: SourceInitialGuess,
}

impl From<SourceInitialGuess> for PySourceInitialGuess {
    fn from(inner: SourceInitialGuess) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceInitialGuess {
    /// Initial source position in metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array_from_vec(py, &self.inner.position_m)
    }

    /// Initial origin time in seconds, when estimated.
    #[getter]
    fn origin_time_s(&self) -> Option<f64> {
        self.inner.origin_time_s
    }

    /// Seed residual RMS in seconds.
    #[getter]
    fn residual_rms_s(&self) -> f64 {
        self.inner.residual_rms_s
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceInitialGuess(position_m={:?}, origin_time_s={:?}, residual_rms_s={})",
            self.inner.position_m, self.inner.origin_time_s, self.inner.residual_rms_s
        )
    }
}

/// One source-localization timing residual.
#[pyclass(module = "sidereon._sidereon", name = "SourceResidual")]
#[derive(Clone)]
pub struct PySourceResidual {
    inner: SourceResidual,
}

impl From<SourceResidual> for PySourceResidual {
    fn from(inner: SourceResidual) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceResidual {
    /// Sensor index in the caller's input order.
    #[getter]
    fn sensor_index(&self) -> usize {
        self.inner.sensor_index
    }

    /// Reference sensor index for TDOA residuals, or `None` for ToA.
    #[getter]
    fn reference_sensor_index(&self) -> Option<usize> {
        self.inner.reference_sensor_index
    }

    /// Residual in seconds.
    #[getter]
    fn residual_s(&self) -> f64 {
        self.inner.residual_s
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceResidual(sensor_index={}, reference_sensor_index={:?}, residual_s={})",
            self.inner.sensor_index, self.inner.reference_sensor_index, self.inner.residual_s
        )
    }
}

/// Per-sensor leave-one-out influence diagnostic.
#[pyclass(module = "sidereon._sidereon", name = "SourceSensorInfluence")]
#[derive(Clone)]
pub struct PySourceSensorInfluence {
    inner: SourceSensorInfluence,
}

impl From<SourceSensorInfluence> for PySourceSensorInfluence {
    fn from(inner: SourceSensorInfluence) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceSensorInfluence {
    /// Sensor index in the caller's input order.
    #[getter]
    fn sensor_index(&self) -> usize {
        self.inner.sensor_index
    }

    /// ToA residual at the full solution, seconds.
    #[getter]
    fn residual_s(&self) -> f64 {
        self.inner.residual_s
    }

    /// Held-out ToA residual after solving without this sensor, seconds.
    #[getter]
    fn leave_one_out_residual_s(&self) -> Option<f64> {
        self.inner.leave_one_out_residual_s
    }

    /// Source-position change after omitting this sensor, metres.
    #[getter]
    fn position_delta_m(&self) -> Option<f64> {
        self.inner.position_delta_m
    }

    /// Origin-time change after omitting this sensor, seconds.
    #[getter]
    fn origin_time_delta_s(&self) -> Option<f64> {
        self.inner.origin_time_delta_s
    }

    /// First-derivative robust-loss weight at the full solution.
    #[getter]
    fn loss_weight(&self) -> f64 {
        self.inner.loss_weight
    }

    /// Normalized influence score.
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceSensorInfluence(sensor_index={}, residual_s={}, score={})",
            self.inner.sensor_index, self.inner.residual_s, self.inner.score
        )
    }
}

/// Source-localization state covariance or CRLB.
#[pyclass(module = "sidereon._sidereon", name = "SourceCovariance")]
#[derive(Clone)]
pub struct PySourceCovariance {
    inner: SourceCovariance,
}

impl From<SourceCovariance> for PySourceCovariance {
    fn from(inner: SourceCovariance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceCovariance {
    /// Full state covariance in solver state order.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        matrix_from_rows(py, &self.inner.state)
    }

    /// Position covariance block in square metres.
    #[getter]
    fn position_m2<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        matrix_from_rows(py, &self.inner.position_m2)
    }

    /// Origin-time variance in square seconds, when origin time is in the state.
    #[getter]
    fn origin_time_s2(&self) -> Option<f64> {
        self.inner.origin_time_s2
    }

    /// Timing sigma used to scale this covariance, seconds.
    #[getter]
    fn timing_sigma_s(&self) -> f64 {
        self.inner.timing_sigma_s
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceCovariance(state_rows={}, position_dim={}, timing_sigma_s={})",
            self.inner.state.len(),
            self.inner.position_m2.len(),
            self.inner.timing_sigma_s
        )
    }
}

/// Source-location solution.
#[pyclass(module = "sidereon._sidereon", name = "SourceSolution")]
#[derive(Clone)]
pub struct PySourceSolution {
    inner: SourceSolution,
}

impl From<SourceSolution> for PySourceSolution {
    fn from(inner: SourceSolution) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceSolution {
    /// Estimated source position in metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        array_from_vec(py, &self.inner.position_m)
    }

    /// Estimated origin time in seconds.
    #[getter]
    fn origin_time_s(&self) -> Option<f64> {
        self.inner.origin_time_s
    }

    /// State covariance scaled by `SourceLocateOptions.timing_sigma_s`.
    #[getter]
    fn covariance(&self) -> Option<PySourceCovariance> {
        self.inner.covariance.clone().map(Into::into)
    }

    /// Alias for the covariance interpreted as the timing CRLB.
    #[getter]
    fn crlb(&self) -> Option<PySourceCovariance> {
        self.inner.crlb().cloned().map(Into::into)
    }

    /// Solver residual rows in seconds.
    #[getter]
    fn residuals(&self) -> Vec<PySourceResidual> {
        self.inner
            .residuals
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Per-sensor influence diagnostics.
    #[getter]
    fn per_sensor_influence(&self) -> Vec<PySourceSensorInfluence> {
        self.inner
            .per_sensor_influence
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Geometry observability and covariance-validation diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    /// Closed-form seed used by the iterative solve.
    #[getter]
    fn initial_guess(&self) -> PySourceInitialGuess {
        self.inner.initial_guess.clone().into()
    }

    /// Trust-region termination status code.
    #[getter]
    fn status(&self) -> i32 {
        self.inner.status
    }

    /// Residual evaluations used by the solver.
    #[getter]
    fn nfev(&self) -> usize {
        self.inner.nfev
    }

    /// Jacobian evaluations used by the solver.
    #[getter]
    fn njev(&self) -> usize {
        self.inner.njev
    }

    /// Final least-squares cost.
    #[getter]
    fn cost(&self) -> f64 {
        self.inner.cost
    }

    /// Infinity norm of the final gradient.
    #[getter]
    fn optimality(&self) -> f64 {
        self.inner.optimality
    }

    fn __repr__(&self) -> String {
        format!(
            "SourceSolution(position_m={:?}, origin_time_s={:?}, status={}, cost={})",
            self.inner.position_m, self.inner.origin_time_s, self.inner.status, self.inner.cost
        )
    }
}

/// Source CRLB and timing DOP for a proposed sensor/source geometry.
#[pyclass(module = "sidereon._sidereon", name = "SourceCrlb")]
#[derive(Clone)]
pub struct PySourceCrlb {
    inner: SourceCrlb,
}

impl From<SourceCrlb> for PySourceCrlb {
    fn from(inner: SourceCrlb) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySourceCrlb {
    /// Timing DOP scalars for the proposed geometry.
    #[getter]
    fn dop(&self) -> PyDop {
        PyDop::from_core(self.inner.dop.clone())
    }

    /// State covariance scaled by the requested timing sigma.
    #[getter]
    fn covariance(&self) -> PySourceCovariance {
        self.inner.covariance.clone().into()
    }

    fn __repr__(&self) -> String {
        "SourceCrlb()".to_string()
    }
}

/// Locate a source from sensor arrival times.
///
/// Sensor and source positions are in a caller-chosen 2D or 3D Cartesian frame,
/// metres. Arrival times and origin time are seconds. The call-level
/// propagation speed is metres per second.
#[pyfunction]
#[pyo3(signature = (sensors, arrival_times_s, propagation_speed_m_s, options=None))]
fn locate_source(
    py: Python<'_>,
    sensors: Vec<Py<PySensor>>,
    arrival_times_s: PyReadonlyArray1<'_, f64>,
    propagation_speed_m_s: f64,
    options: Option<Py<PySourceLocateOptions>>,
) -> PyResult<PySourceSolution> {
    let sensors = sensors_to_core(py, &sensors);
    let arrival_times_s = arrival_times_s
        .as_slice()
        .map_err(|err| SourceLocalizationError::new_err(err.to_string()))?
        .to_vec();
    let options = options
        .as_ref()
        .map(|options| options.borrow(py).inner.clone())
        .unwrap_or_default();
    py.allow_threads(|| {
        core_locate_source(&sensors, &arrival_times_s, propagation_speed_m_s, &options)
    })
    .map(Into::into)
    .map_err(source_error)
}

/// Compute the closed-form seed used by `locate_source`.
#[pyfunction]
#[pyo3(signature = (sensors, arrival_times_s, propagation_speed_m_s, mode=None))]
fn chan_ho_initial_guess(
    py: Python<'_>,
    sensors: Vec<Py<PySensor>>,
    arrival_times_s: PyReadonlyArray1<'_, f64>,
    propagation_speed_m_s: f64,
    mode: Option<Py<PySourceSolveMode>>,
) -> PyResult<PySourceInitialGuess> {
    let sensors = sensors_to_core(py, &sensors);
    let arrival_times_s = arrival_times_s
        .as_slice()
        .map_err(|err| SourceLocalizationError::new_err(err.to_string()))?
        .to_vec();
    let mode = mode
        .as_ref()
        .map(|mode| mode.borrow(py).inner)
        .unwrap_or(SourceSolveMode::Toa);
    core_chan_ho_initial_guess(&sensors, &arrival_times_s, propagation_speed_m_s, mode)
        .map(Into::into)
        .map_err(source_error)
}

/// Compute timing DOP for a proposed source location.
#[pyfunction]
fn source_dop(
    py: Python<'_>,
    sensors: Vec<Py<PySensor>>,
    source_position_m: PyReadonlyArray1<'_, f64>,
    propagation_speed_m_s: f64,
) -> PyResult<PyDop> {
    let sensors = sensors_to_core(py, &sensors);
    let source_position_m = source_position_m
        .as_slice()
        .map_err(|err| SourceLocalizationError::new_err(err.to_string()))?
        .to_vec();
    core_source_dop(&sensors, &source_position_m, propagation_speed_m_s)
        .map(PyDop::from_core)
        .map_err(source_error)
}

/// Compute a timing CRLB for a proposed source location.
#[pyfunction]
fn source_crlb(
    py: Python<'_>,
    sensors: Vec<Py<PySensor>>,
    source_position_m: PyReadonlyArray1<'_, f64>,
    propagation_speed_m_s: f64,
    timing_sigma_s: f64,
) -> PyResult<PySourceCrlb> {
    let sensors = sensors_to_core(py, &sensors);
    let source_position_m = source_position_m
        .as_slice()
        .map_err(|err| SourceLocalizationError::new_err(err.to_string()))?
        .to_vec();
    core_source_crlb(
        &sensors,
        &source_position_m,
        propagation_speed_m_s,
        timing_sigma_s,
    )
    .map(Into::into)
    .map_err(source_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySensor>()?;
    m.add_class::<PySourceSolveMode>()?;
    m.add_class::<PySourceLocateOptions>()?;
    m.add_class::<PySourceInitialGuess>()?;
    m.add_class::<PySourceResidual>()?;
    m.add_class::<PySourceSensorInfluence>()?;
    m.add_class::<PySourceCovariance>()?;
    m.add_class::<PySourceSolution>()?;
    m.add_class::<PySourceCrlb>()?;
    m.add_function(wrap_pyfunction!(locate_source, m)?)?;
    m.add_function(wrap_pyfunction!(chan_ho_initial_guess, m)?)?;
    m.add_function(wrap_pyfunction!(source_dop, m)?)?;
    m.add_function(wrap_pyfunction!(source_crlb, m)?)?;
    Ok(())
}
