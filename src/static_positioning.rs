//! Multi-epoch static positioning binding.
//!
//! Static epochs are built from the binding's existing [`SppConfig`] marshaling,
//! then forwarded to [`sidereon_core::positioning::solve_static`]. The binding
//! does no measurement modeling of its own.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::astro::math::least_squares::Status;
use sidereon_core::positioning::{
    solve_static as core_solve_static, EphemerisSource, RobustConfig, StaticCovariance,
    StaticEpoch, StaticInfluenceStatus, StaticSolution, StaticSolveOptions,
};

use crate::events::PyWgs84Geodetic;
use crate::geometry_quality::PyGeometryQuality;
use crate::marshal::{mat3_to_array, PyGnssSystem};
use crate::rinex::PyBroadcastEphemeris;
use crate::spp::{PySppConfig, PySppRobustConfig};
use crate::{np_array, to_solve_err, PySp3};

type PyStaticEpochInfluence = (
    usize,
    usize,
    &'static str,
    Option<[f64; 3]>,
    Option<f64>,
    Option<f64>,
    f64,
);
type PyStaticSatelliteInfluence = (
    usize,
    String,
    &'static str,
    Option<[f64; 3]>,
    Option<f64>,
    Option<f64>,
    f64,
    f64,
    f64,
    f64,
);
type PyStaticSatelliteBatchInfluence = (
    String,
    usize,
    &'static str,
    Option<[f64; 3]>,
    Option<f64>,
    Option<f64>,
    f64,
);

fn status_label(status: Status) -> &'static str {
    match status {
        Status::GradientTolerance => "gradient_tolerance",
        Status::CostTolerance => "cost_tolerance",
        Status::StepTolerance => "step_tolerance",
        Status::MaxEvaluations => "max_evaluations",
    }
}

fn influence_status_label(status: StaticInfluenceStatus) -> &'static str {
    match status {
        StaticInfluenceStatus::Solved => "solved",
        StaticInfluenceStatus::TooFewMeasurements => "too_few_measurements",
        StaticInfluenceStatus::SingularGeometry => "singular_geometry",
        StaticInfluenceStatus::InvalidInput => "invalid_input",
        StaticInfluenceStatus::EphemerisUnavailable => "ephemeris_unavailable",
        StaticInfluenceStatus::SolveFailed => "solve_failed",
    }
}

fn vec_matrix_to_array<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    let mut array = Array2::<f64>::zeros((nrows, ncols));
    for (row_index, row) in rows.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            array[[row_index, col_index]] = *value;
        }
    }
    PyArray2::from_owned_array(py, array)
}

fn with_static_ephemeris_source<R>(
    source: &Bound<'_, PyAny>,
    f: impl FnOnce(&dyn EphemerisSource) -> PyResult<R>,
) -> PyResult<R> {
    if let Ok(sp3) = source.extract::<PyRef<'_, PySp3>>() {
        f(&sp3.inner)
    } else if let Ok(broadcast) = source.extract::<PyRef<'_, PyBroadcastEphemeris>>() {
        f(&broadcast.inner)
    } else {
        Err(PyTypeError::new_err(
            "source must be Sp3 or BroadcastEphemeris",
        ))
    }
}

/// One receive epoch for a multi-epoch static positioning solve.
#[pyclass(module = "sidereon._sidereon", name = "StaticEpoch")]
#[derive(Clone)]
pub struct PyStaticEpoch {
    inner: StaticEpoch,
}

impl PyStaticEpoch {
    fn to_core(&self) -> StaticEpoch {
        self.inner.clone()
    }
}

#[pymethods]
impl PyStaticEpoch {
    /// Build a static epoch from an existing SPP config.
    #[new]
    #[pyo3(signature = (config, weights=None))]
    fn new(config: &PySppConfig, weights: Option<Vec<f64>>) -> Self {
        let mut inner = StaticEpoch::from_solve_inputs(config.to_inputs());
        inner.weights = weights;
        Self { inner }
    }

    /// Number of pseudorange measurements in the epoch.
    #[getter]
    fn measurement_count(&self) -> usize {
        self.inner.measurements.len()
    }

    /// Receive epoch, seconds since J2000 in the ephemeris time scale.
    #[getter]
    fn t_rx_j2000_s(&self) -> f64 {
        self.inner.t_rx_j2000_s
    }

    /// GPS second of day for the receive epoch.
    #[getter]
    fn t_rx_second_of_day_s(&self) -> f64 {
        self.inner.t_rx_second_of_day_s
    }

    /// Fractional day of year for the receive epoch.
    #[getter]
    fn day_of_year(&self) -> f64 {
        self.inner.day_of_year
    }

    /// Initial receiver clock range bias for this epoch, metres.
    #[getter]
    fn clock_initial_m(&self) -> f64 {
        self.inner.clock_initial_m
    }

    /// Optional row weight multipliers.
    #[getter]
    fn weights(&self) -> Option<Vec<f64>> {
        self.inner.weights.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticEpoch(measurements={}, t_rx_j2000_s={})",
            self.inner.measurements.len(),
            self.inner.t_rx_j2000_s
        )
    }
}

/// Options for a multi-epoch static positioning solve.
#[pyclass(module = "sidereon._sidereon", name = "StaticSolveOptions")]
#[derive(Clone, Copy)]
pub struct PyStaticSolveOptions {
    inner: StaticSolveOptions,
}

impl PyStaticSolveOptions {
    fn core_options(&self) -> StaticSolveOptions {
        self.inner
    }
}

#[pymethods]
impl PyStaticSolveOptions {
    /// Create static solve options.
    #[new]
    #[pyo3(signature = (
        initial_position_m=[0.0; 3],
        with_geodetic=false,
        robust=None,
    ))]
    fn new(
        initial_position_m: [f64; 3],
        with_geodetic: bool,
        robust: Option<&PySppRobustConfig>,
    ) -> Self {
        Self {
            inner: StaticSolveOptions {
                initial_position_m,
                with_geodetic,
                robust: robust.map(PySppRobustConfig::inner),
            },
        }
    }

    /// Initial shared receiver ECEF position, metres.
    #[getter]
    fn initial_position_m(&self) -> [f64; 3] {
        self.inner.initial_position_m
    }

    /// Whether the solution includes geodetic coordinates.
    #[getter]
    fn with_geodetic(&self) -> bool {
        self.inner.with_geodetic
    }

    /// Huber/IRLS config when static robust reweighting is enabled.
    #[getter]
    fn robust(&self) -> Option<PySppRobustConfig> {
        self.inner
            .robust
            .map(|inner: RobustConfig| PySppRobustConfig { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticSolveOptions(initial_position_m={:?}, with_geodetic={})",
            self.inner.initial_position_m, self.inner.with_geodetic
        )
    }
}

/// State covariance for a static solution.
#[pyclass(module = "sidereon._sidereon", name = "StaticCovariance")]
#[derive(Clone)]
pub struct PyStaticCovariance {
    inner: StaticCovariance,
}

impl From<StaticCovariance> for PyStaticCovariance {
    fn from(inner: StaticCovariance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStaticCovariance {
    /// Full state covariance in square metres.
    #[getter]
    fn state_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        vec_matrix_to_array(py, &self.inner.state_m2)
    }

    /// ECEF position covariance block in square metres.
    #[getter]
    fn position_ecef_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_ecef_m2)
    }

    /// Local ENU position covariance block in square metres.
    #[getter]
    fn position_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_enu_m2)
    }

    fn __repr__(&self) -> String {
        format!("StaticCovariance(parameters={})", self.inner.state_m2.len())
    }
}

/// Iteration and redundancy metadata for a static solution.
#[pyclass(module = "sidereon._sidereon", name = "StaticSolutionMetadata")]
#[derive(Clone)]
pub struct PyStaticSolutionMetadata {
    inner: sidereon_core::positioning::StaticSolutionMetadata,
}

impl From<sidereon_core::positioning::StaticSolutionMetadata> for PyStaticSolutionMetadata {
    fn from(inner: sidereon_core::positioning::StaticSolutionMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStaticSolutionMetadata {
    /// Number of accepted trust-region iterations.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    /// Whether the final solve reached a convergence criterion.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Stable termination status label.
    #[getter]
    fn status(&self) -> &'static str {
        status_label(self.inner.status)
    }

    /// Number of robust outer iterations.
    #[getter]
    fn outer_iterations(&self) -> usize {
        self.inner.outer_iterations
    }

    /// Final MAD robust scale, metres, when robust reweighting ran.
    #[getter]
    fn final_robust_scale_m(&self) -> Option<f64> {
        self.inner.final_robust_scale_m
    }

    /// Number of measurements used by the final solve.
    #[getter]
    fn used_measurements(&self) -> usize {
        self.inner.used_measurements
    }

    /// Number of fitted state parameters.
    #[getter]
    fn n_parameters(&self) -> usize {
        self.inner.n_parameters
    }

    /// Degrees of freedom.
    #[getter]
    fn redundancy(&self) -> isize {
        self.inner.redundancy
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticSolutionMetadata(iterations={}, converged={}, redundancy={})",
            self.inner.iterations, self.inner.converged, self.inner.redundancy
        )
    }
}

/// Multi-epoch static receiver solution.
#[pyclass(module = "sidereon._sidereon", name = "StaticSolution")]
pub struct PyStaticSolution {
    inner: StaticSolution,
}

#[pymethods]
impl PyStaticSolution {
    /// ECEF receiver position as a numpy `(3,)` array, metres.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(
            py,
            &[
                self.inner.position.x_m,
                self.inner.position.y_m,
                self.inner.position.z_m,
            ],
        )
    }

    /// ECEF X coordinate, metres.
    #[getter]
    fn x_m(&self) -> f64 {
        self.inner.position.x_m
    }

    /// ECEF Y coordinate, metres.
    #[getter]
    fn y_m(&self) -> f64 {
        self.inner.position.y_m
    }

    /// ECEF Z coordinate, metres.
    #[getter]
    fn z_m(&self) -> f64 {
        self.inner.position.z_m
    }

    /// Geodetic solution if requested.
    #[getter]
    fn geodetic(&self) -> Option<PyWgs84Geodetic> {
        self.inner.geodetic.map(PyWgs84Geodetic::from_core)
    }

    /// Epoch-local receiver clocks as `(epoch_index, system, clock_s)`.
    #[getter]
    fn per_epoch_clock(&self) -> Vec<(usize, PyGnssSystem, f64)> {
        self.inner
            .per_epoch_clock
            .iter()
            .map(|clock| (clock.epoch_index, clock.system.into(), clock.clock_s))
            .collect()
    }

    /// State and position covariance blocks.
    #[getter]
    fn covariance(&self) -> PyStaticCovariance {
        self.inner.covariance.clone().into()
    }

    /// Post-fit residual rows as `(epoch_index, satellite, residual_m, base_weight, effective_weight, robust_weight_ratio)`.
    #[getter]
    fn residuals_m(&self) -> Vec<(usize, String, f64, f64, f64, f64)> {
        self.inner
            .residuals_m
            .iter()
            .map(|residual| {
                (
                    residual.epoch_index,
                    residual.satellite_id.to_string(),
                    residual.residual_m,
                    residual.base_weight,
                    residual.effective_weight,
                    residual.robust_weight_ratio,
                )
            })
            .collect()
    }

    /// Residual RMS in metres.
    #[getter]
    fn residual_rms_m(&self) -> f64 {
        self.inner.residual_rms_m()
    }

    /// Used satellites by epoch, in solver row order.
    #[getter]
    fn used_sats(&self) -> Vec<Vec<String>> {
        self.inner
            .used_sats
            .iter()
            .map(|epoch| epoch.iter().map(ToString::to_string).collect())
            .collect()
    }

    /// Rejected satellites by epoch as `(satellite, reason)` rows.
    #[getter]
    fn rejected_sats(&self) -> Vec<Vec<(String, String)>> {
        self.inner
            .rejected_sats
            .iter()
            .map(|epoch| {
                epoch
                    .iter()
                    .map(|sat| (sat.satellite_id.to_string(), format!("{:?}", sat.reason)))
                    .collect()
            })
            .collect()
    }

    /// Leave-one-epoch diagnostics as tuple rows.
    #[getter]
    fn per_epoch_influence(&self) -> Vec<PyStaticEpochInfluence> {
        self.inner
            .per_epoch_influence
            .iter()
            .map(|row| {
                (
                    row.epoch_index,
                    row.omitted_measurements,
                    influence_status_label(row.status),
                    row.position_delta_m,
                    row.position_delta_norm_m,
                    row.residual_rms_m,
                    row.min_robust_weight_ratio,
                )
            })
            .collect()
    }

    /// Leave-one-satellite diagnostics as tuple rows.
    #[getter]
    fn per_satellite_influence(&self) -> Vec<PyStaticSatelliteInfluence> {
        self.inner
            .per_satellite_influence
            .iter()
            .map(|row| {
                (
                    row.epoch_index,
                    row.satellite_id.to_string(),
                    influence_status_label(row.status),
                    row.position_delta_m,
                    row.position_delta_norm_m,
                    row.residual_rms_m,
                    row.residual_m,
                    row.base_weight,
                    row.effective_weight,
                    row.robust_weight_ratio,
                )
            })
            .collect()
    }

    /// Leave-one-satellite-across-batch diagnostics as tuple rows.
    #[getter]
    fn per_satellite_batch_influence(&self) -> Vec<PyStaticSatelliteBatchInfluence> {
        self.inner
            .per_satellite_batch_influence
            .iter()
            .map(|row| {
                (
                    row.satellite_id.to_string(),
                    row.omitted_measurements,
                    influence_status_label(row.status),
                    row.position_delta_m,
                    row.position_delta_norm_m,
                    row.residual_rms_m,
                    row.min_robust_weight_ratio,
                )
            })
            .collect()
    }

    /// Geometry observability and covariance-validation diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    /// Iteration and redundancy metadata.
    #[getter]
    fn metadata(&self) -> PyStaticSolutionMetadata {
        self.inner.metadata.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticSolution(position=[{:.3}, {:.3}, {:.3}], epochs={})",
            self.inner.position.x_m,
            self.inner.position.y_m,
            self.inner.position.z_m,
            self.inner.used_sats.len()
        )
    }
}

/// Solve one static receiver position from multiple pseudorange epochs.
#[pyfunction]
#[pyo3(signature = (source, epochs, options=None))]
fn solve_static(
    source: &Bound<'_, PyAny>,
    epochs: Vec<PyStaticEpoch>,
    options: Option<&PyStaticSolveOptions>,
) -> PyResult<PyStaticSolution> {
    let epochs = epochs
        .iter()
        .map(PyStaticEpoch::to_core)
        .collect::<Vec<_>>();
    let options = options
        .map(PyStaticSolveOptions::core_options)
        .unwrap_or_default();
    let inner = with_static_ephemeris_source(source, |source| {
        core_solve_static(source, &epochs, options).map_err(to_solve_err)
    })?;
    Ok(PyStaticSolution { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStaticEpoch>()?;
    m.add_class::<PyStaticSolveOptions>()?;
    m.add_class::<PyStaticCovariance>()?;
    m.add_class::<PyStaticSolutionMetadata>()?;
    m.add_class::<PyStaticSolution>()?;
    m.add_function(wrap_pyfunction!(solve_static, m)?)?;
    Ok(())
}
