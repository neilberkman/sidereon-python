//! SP3-anchored precise orbit-fit bindings.

use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::orbit_determination::{
    fit_precise_ephemeris_sample_orbit as core_fit_precise_ephemeris_sample_orbit,
    fit_sp3_precise_orbit as core_fit_sp3_precise_orbit, OrbitArcSpan, OrbitFitCovariance,
    OrbitFitOptions, OrbitFitReport, OrbitFitSolution, OrbitResidualLedger, OrbitResidualStats,
};
use sidereon_core::GnssSatelliteId;

use crate::ephemeris::{PyPreciseEphemerisSample, PySp3};
use crate::forces::PyDragParameters;
use crate::frames::PyTimeScale;
use crate::geometry_quality::PyGeometryQuality;
use crate::marshal::{rows_to_array, PyGnssSystem};
use crate::propagation::{PyForceModelKind, PyIntegrator};
use crate::relative::PyCartesianState;
use crate::space_weather::PySpaceWeatherTable;

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite token {token:?}: {err}")))
}

fn to_orbit_fit_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Options controlling precise-orbit fitting.
#[pyclass(module = "sidereon._sidereon", name = "OrbitFitOptions")]
#[derive(Clone)]
pub struct PyOrbitFitOptions {
    inner: OrbitFitOptions,
}

impl PyOrbitFitOptions {
    fn inner(&self) -> OrbitFitOptions {
        self.inner.clone()
    }
}

#[pymethods]
impl PyOrbitFitOptions {
    /// Build precise-orbit fit options.
    #[new]
    #[pyo3(signature = (
        force_model=None,
        integrator=PyIntegrator::DP54,
        abs_tol=1.0e-9,
        rel_tol=1.0e-12,
        initial_step_s=60.0,
        min_step_s=1.0e-6,
        max_step_s=3600.0,
        max_steps=1_000_000,
        min_ledger_samples=3,
        drag=None,
        space_weather_table=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        force_model: Option<&PyForceModelKind>,
        integrator: PyIntegrator,
        abs_tol: f64,
        rel_tol: f64,
        initial_step_s: f64,
        min_step_s: f64,
        max_step_s: f64,
        max_steps: u32,
        min_ledger_samples: usize,
        drag: Option<Py<PyDragParameters>>,
        space_weather_table: Option<Py<PySpaceWeatherTable>>,
    ) -> Self {
        let mut inner = OrbitFitOptions::default();
        if let Some(force_model) = force_model {
            inner.force_model = force_model.inner();
        }
        inner.integrator = integrator.into();
        inner.integrator_options.abs_tol = abs_tol;
        inner.integrator_options.rel_tol = rel_tol;
        inner.integrator_options.initial_step = initial_step_s;
        inner.integrator_options.min_step = min_step_s;
        inner.integrator_options.max_step = max_step_s;
        inner.integrator_options.max_steps = max_steps;
        inner.min_ledger_samples = min_ledger_samples;
        inner.drag = drag.map(|value| value.borrow(py).inner());
        inner.space_weather = space_weather_table.map(|value| value.borrow(py).source());
        Self { inner }
    }

    /// Minimum residual count before a ledger entry is no longer low sample count.
    #[getter]
    fn min_ledger_samples(&self) -> usize {
        self.inner.min_ledger_samples
    }
}

/// State covariance result for a fitted initial state.
#[pyclass(module = "sidereon._sidereon", name = "OrbitFitCovariance")]
#[derive(Clone)]
pub struct PyOrbitFitCovariance {
    inner: OrbitFitCovariance,
}

impl From<OrbitFitCovariance> for PyOrbitFitCovariance {
    fn from(inner: OrbitFitCovariance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOrbitFitCovariance {
    /// Stable covariance kind: `estimated` or `unbounded`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            OrbitFitCovariance::Estimated { .. } => "estimated",
            OrbitFitCovariance::Unbounded => "unbounded",
        }
    }

    /// Whether the arc has no finite residual-scaled covariance bound.
    #[getter]
    fn is_unbounded(&self) -> bool {
        matches!(self.inner, OrbitFitCovariance::Unbounded)
    }

    /// Estimated row-major state covariance, or `None` when unbounded.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        match &self.inner {
            OrbitFitCovariance::Estimated { matrix } => Some(rows_to_array(py, matrix.as_ref())),
            OrbitFitCovariance::Unbounded => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("OrbitFitCovariance(kind={})", self.kind())
    }
}

/// Initial-state fit result for one satellite.
#[pyclass(module = "sidereon._sidereon", name = "OrbitFitSolution")]
#[derive(Clone)]
pub struct PyOrbitFitSolution {
    inner: OrbitFitSolution,
}

impl From<OrbitFitSolution> for PyOrbitFitSolution {
    fn from(inner: OrbitFitSolution) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOrbitFitSolution {
    /// Satellite fitted by this solution.
    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    /// Estimated inertial initial state.
    #[getter]
    fn initial_state(&self) -> PyCartesianState {
        PyCartesianState::from_inner(self.inner.initial_state)
    }

    /// Fitted state covariance, or an unbounded marker for short arcs.
    #[getter]
    fn covariance(&self) -> PyOrbitFitCovariance {
        self.inner.covariance.clone().into()
    }

    /// Singular-value geometry diagnostics for the final design matrix.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    /// Three-dimensional RMS residual of the automatically seeded state, metres.
    #[getter]
    fn seed_rms_3d_m(&self) -> f64 {
        self.inner.seed_rms_3d_m
    }

    /// Three-dimensional RMS residual of the fitted state, metres.
    #[getter]
    fn fit_rms_3d_m(&self) -> f64 {
        self.inner.fit_rms_3d_m
    }

    /// Accepted nonlinear least-squares iterations.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }
}

/// Arc span covered by a residual ledger.
#[pyclass(module = "sidereon._sidereon", name = "OrbitArcSpan")]
#[derive(Clone, Copy)]
pub struct PyOrbitArcSpan {
    inner: OrbitArcSpan,
}

impl From<OrbitArcSpan> for PyOrbitArcSpan {
    fn from(inner: OrbitArcSpan) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOrbitArcSpan {
    /// Time scale shared by all residual epochs.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale.into()
    }

    /// First residual epoch, seconds since J2000 in `time_scale`.
    #[getter]
    fn start_j2000_s(&self) -> f64 {
        self.inner.start_j2000_s
    }

    /// Last residual epoch, seconds since J2000 in `time_scale`.
    #[getter]
    fn end_j2000_s(&self) -> f64 {
        self.inner.end_j2000_s
    }

    /// Arc duration, seconds.
    #[getter]
    fn duration_s(&self) -> f64 {
        self.inner.duration_s
    }
}

/// RTN residual RMS summary.
#[pyclass(module = "sidereon._sidereon", name = "OrbitResidualStats")]
#[derive(Clone, Copy)]
pub struct PyOrbitResidualStats {
    inner: OrbitResidualStats,
}

impl From<OrbitResidualStats> for PyOrbitResidualStats {
    fn from(inner: OrbitResidualStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOrbitResidualStats {
    /// Radial RMS residual, metres.
    #[getter]
    fn radial_rms_m(&self) -> f64 {
        self.inner.radial_rms_m
    }

    /// Along-track RMS residual, metres.
    #[getter]
    fn along_rms_m(&self) -> f64 {
        self.inner.along_rms_m
    }

    /// Cross-track RMS residual, metres.
    #[getter]
    fn cross_rms_m(&self) -> f64 {
        self.inner.cross_rms_m
    }

    /// Three-dimensional RMS residual, metres.
    #[getter]
    fn rms_3d_m(&self) -> f64 {
        self.inner.rms_3d_m
    }

    /// Number of residual epochs.
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    /// Whether this entry has fewer samples than the configured ledger threshold.
    #[getter]
    fn low_sample_count(&self) -> bool {
        self.inner.low_sample_count
    }
}

/// Residual RMS ledger grouped by satellite and constellation.
#[pyclass(module = "sidereon._sidereon", name = "OrbitResidualLedger")]
#[derive(Clone)]
pub struct PyOrbitResidualLedger {
    inner: OrbitResidualLedger,
}

impl From<OrbitResidualLedger> for PyOrbitResidualLedger {
    fn from(inner: OrbitResidualLedger) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOrbitResidualLedger {
    /// Per-satellite RTN residual RMS values.
    #[getter]
    fn per_sat(&self) -> Vec<(String, PyOrbitResidualStats)> {
        self.inner
            .per_sat
            .iter()
            .map(|(sat, stats)| (sat.to_string(), (*stats).into()))
            .collect()
    }

    /// Per-constellation RTN residual RMS values.
    #[getter]
    fn per_constellation(&self) -> Vec<(PyGnssSystem, PyOrbitResidualStats)> {
        self.inner
            .per_constellation
            .iter()
            .map(|(system, stats)| ((*system).into(), (*stats).into()))
            .collect()
    }

    /// Time span covered by all residuals.
    #[getter]
    fn arc_span(&self) -> PyOrbitArcSpan {
        self.inner.arc_span.into()
    }
}

/// Batch orbit-fit report.
#[pyclass(module = "sidereon._sidereon", name = "OrbitFitReport")]
#[derive(Clone)]
pub struct PyOrbitFitReport {
    inner: OrbitFitReport,
}

impl From<OrbitFitReport> for PyOrbitFitReport {
    fn from(inner: OrbitFitReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOrbitFitReport {
    /// Fitted solutions in satellite order.
    #[getter]
    fn fits(&self) -> Vec<PyOrbitFitSolution> {
        self.inner.fits.values().cloned().map(Into::into).collect()
    }

    /// RTN residual RMS ledger.
    #[getter]
    fn ledger(&self) -> PyOrbitResidualLedger {
        self.inner.ledger.clone().into()
    }

    /// Number of fitted satellites.
    #[getter]
    fn fit_count(&self) -> usize {
        self.inner.fits.len()
    }
}

/// Fit one satellite from a parsed SP3 product.
#[pyfunction]
#[pyo3(signature = (product, satellite, options=None))]
fn fit_sp3_precise_orbit(
    product: &PySp3,
    satellite: &str,
    options: Option<&PyOrbitFitOptions>,
) -> PyResult<PyOrbitFitReport> {
    let satellite = parse_satellite(satellite)?;
    let options = options.map(PyOrbitFitOptions::inner).unwrap_or_default();
    core_fit_sp3_precise_orbit(&product.inner, satellite, &options)
        .map(Into::into)
        .map_err(to_orbit_fit_err)
}

/// Fit one satellite from precise ephemeris sample handles.
#[pyfunction]
#[pyo3(signature = (samples, satellite, options=None))]
fn fit_precise_ephemeris_sample_orbit(
    py: Python<'_>,
    samples: Vec<Py<PyPreciseEphemerisSample>>,
    satellite: &str,
    options: Option<&PyOrbitFitOptions>,
) -> PyResult<PyOrbitFitReport> {
    let samples = samples
        .iter()
        .map(|sample| sample.borrow(py).to_core())
        .collect::<Vec<_>>();
    let satellite = parse_satellite(satellite)?;
    let options = options.map(PyOrbitFitOptions::inner).unwrap_or_default();
    core_fit_precise_ephemeris_sample_orbit(&samples, satellite, &options)
        .map(Into::into)
        .map_err(to_orbit_fit_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrbitFitOptions>()?;
    m.add_class::<PyOrbitFitCovariance>()?;
    m.add_class::<PyOrbitFitSolution>()?;
    m.add_class::<PyOrbitArcSpan>()?;
    m.add_class::<PyOrbitResidualStats>()?;
    m.add_class::<PyOrbitResidualLedger>()?;
    m.add_class::<PyOrbitFitReport>()?;
    m.add_function(wrap_pyfunction!(fit_sp3_precise_orbit, m)?)?;
    m.add_function(wrap_pyfunction!(fit_precise_ephemeris_sample_orbit, m)?)?;
    Ok(())
}
