//! Classical range-reliability bindings.
//!
//! The calculations delegate to `sidereon_core::quality` and return the core
//! reliability diagnostics with Python `None` for observations that cannot be
//! checked by the design.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::quality::{
    reliability_araim as core_reliability_araim, reliability_design as core_reliability_design,
    wtest_noncentrality as core_wtest_noncentrality, ObservationReliability, RangeReliabilityRow,
    ReliabilityOptions, ReliabilityReport, ReliabilitySummary,
};

use crate::araim::{PyAraimGeometry, PyIsm};

fn to_value_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Baarda w-test detection constants.
#[pyclass(module = "sidereon._sidereon", name = "WtestNoncentrality")]
#[derive(Clone, Copy)]
pub struct PyWtestNoncentrality {
    delta0: f64,
    lambda0: f64,
}

#[pymethods]
impl PyWtestNoncentrality {
    /// One-dimensional noncentrality amplitude.
    #[getter]
    fn delta0(&self) -> f64 {
        self.delta0
    }

    /// Noncentrality parameter, equal to `delta0 * delta0`.
    #[getter]
    fn lambda0(&self) -> f64 {
        self.lambda0
    }

    /// Return a compact representation of the w-test constants.
    fn __repr__(&self) -> String {
        format!(
            "WtestNoncentrality(delta0={:.15}, lambda0={:.15})",
            self.delta0, self.lambda0
        )
    }
}

/// Options for classical reliability design.
#[pyclass(module = "sidereon._sidereon", name = "ReliabilityOptions")]
#[derive(Clone, Copy)]
pub struct PyReliabilityOptions {
    inner: ReliabilityOptions,
}

impl PyReliabilityOptions {
    fn inner_or_default(value: Option<&Self>) -> ReliabilityOptions {
        value.map(|options| options.inner).unwrap_or_default()
    }
}

#[pymethods]
impl PyReliabilityOptions {
    /// Build reliability options.
    ///
    /// Omitted values use the core defaults. `alpha` is the two-sided false
    /// alarm probability, `beta` is missed-detection probability, and
    /// `lambda0_override` bypasses the w-test calculation when supplied.
    #[new]
    #[pyo3(signature = (alpha=None, beta=None, lambda0_override=None, min_redundancy=None))]
    fn new(
        alpha: Option<f64>,
        beta: Option<f64>,
        lambda0_override: Option<f64>,
        min_redundancy: Option<f64>,
    ) -> Self {
        let defaults = ReliabilityOptions::default();
        Self {
            inner: ReliabilityOptions {
                alpha: alpha.unwrap_or(defaults.alpha),
                beta: beta.unwrap_or(defaults.beta),
                lambda0_override,
                min_redundancy: min_redundancy.unwrap_or(defaults.min_redundancy),
            },
        }
    }

    /// Two-sided false-alarm probability for the w-test.
    #[getter]
    fn alpha(&self) -> f64 {
        self.inner.alpha
    }

    /// Missed-detection probability for the target bias.
    #[getter]
    fn beta(&self) -> f64 {
        self.inner.beta
    }

    /// Precomputed noncentrality parameter, if supplied.
    #[getter]
    fn lambda0_override(&self) -> Option<f64> {
        self.inner.lambda0_override
    }

    /// Redundancy floor below which an observation is uncheckable.
    #[getter]
    fn min_redundancy(&self) -> f64 {
        self.inner.min_redundancy
    }

    /// Return a compact representation of the options.
    fn __repr__(&self) -> String {
        format!(
            "ReliabilityOptions(alpha={}, beta={}, min_redundancy={})",
            self.inner.alpha, self.inner.beta, self.inner.min_redundancy
        )
    }
}

/// One range observation row for reliability design.
#[pyclass(module = "sidereon._sidereon", name = "RangeReliabilityRow")]
#[derive(Clone)]
pub struct PyRangeReliabilityRow {
    inner: RangeReliabilityRow,
}

#[pymethods]
impl PyRangeReliabilityRow {
    /// Build one linearized range row.
    ///
    /// `design_row` contains the partial derivatives for this observation and
    /// `sigma_m` is the externally supplied one-sigma range error in metres.
    #[new]
    fn new(id: String, design_row: Vec<f64>, sigma_m: f64) -> Self {
        Self {
            inner: RangeReliabilityRow {
                id,
                design_row,
                sigma_m,
            },
        }
    }

    /// Observation identifier echoed into the reliability report.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Linearized design row for this range observation.
    #[getter]
    fn design_row(&self) -> Vec<f64> {
        self.inner.design_row.clone()
    }

    /// Externally supplied one-sigma range error, metres.
    #[getter]
    fn sigma_m(&self) -> f64 {
        self.inner.sigma_m
    }

    /// Return a compact representation of the range row.
    fn __repr__(&self) -> String {
        format!(
            "RangeReliabilityRow(id={:?}, parameters={}, sigma_m={})",
            self.inner.id,
            self.inner.design_row.len(),
            self.inner.sigma_m
        )
    }
}

/// Reliability diagnostics for one observation.
#[pyclass(module = "sidereon._sidereon", name = "ObservationReliability")]
#[derive(Clone)]
pub struct PyObservationReliability {
    inner: ObservationReliability,
}

impl From<ObservationReliability> for PyObservationReliability {
    fn from(inner: ObservationReliability) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObservationReliability {
    /// Observation identifier echoed from the input row.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Redundancy number for this observation.
    #[getter]
    fn redundancy(&self) -> f64 {
        self.inner.redundancy
    }

    /// Minimal detectable bias in metres, or `None` when uncheckable.
    #[getter]
    fn mdb_m(&self) -> Option<f64> {
        self.inner.mdb_m
    }

    /// External effect in local ENU or first state coordinates, or `None`.
    #[getter]
    fn external_enu_m(&self) -> Option<(f64, f64, f64)> {
        self.inner
            .external_enu_m
            .map(|value| (value[0], value[1], value[2]))
    }

    /// Bias-to-noise ratio in state space, or `None` when uncheckable.
    #[getter]
    fn bias_to_noise(&self) -> Option<f64> {
        self.inner.bias_to_noise
    }

    /// True when the observation redundancy is below the configured floor.
    #[getter]
    fn uncheckable(&self) -> bool {
        self.inner.uncheckable
    }

    /// Return a compact representation of the observation reliability.
    fn __repr__(&self) -> String {
        format!(
            "ObservationReliability(id={:?}, redundancy={:.6}, uncheckable={})",
            self.inner.id, self.inner.redundancy, self.inner.uncheckable
        )
    }
}

/// Aggregate diagnostics for a reliability design.
#[pyclass(module = "sidereon._sidereon", name = "ReliabilitySummary")]
#[derive(Clone)]
pub struct PyReliabilitySummary {
    inner: ReliabilitySummary,
}

impl From<ReliabilitySummary> for PyReliabilitySummary {
    fn from(inner: ReliabilitySummary) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyReliabilitySummary {
    /// Number of observations in the design.
    #[getter]
    fn n_obs(&self) -> usize {
        self.inner.n_obs
    }

    /// Number of estimated parameters in the design.
    #[getter]
    fn n_params(&self) -> usize {
        self.inner.n_params
    }

    /// Algebraic degrees of freedom, `n_obs - n_params`.
    #[getter]
    fn dof(&self) -> usize {
        self.inner.dof
    }

    /// Sum of all per-observation redundancy numbers.
    #[getter]
    fn sum_redundancy(&self) -> f64 {
        self.inner.sum_redundancy
    }

    /// Noncentrality parameter used for MDB calculations.
    #[getter]
    fn lambda0(&self) -> f64 {
        self.inner.lambda0
    }

    /// Largest finite MDB in the design as `(id, mdb_m)`.
    #[getter]
    fn max_mdb_m(&self) -> Option<(String, f64)> {
        self.inner.max_mdb_m.clone()
    }

    /// Smallest redundancy number as `(id, redundancy)`.
    #[getter]
    fn min_redundancy(&self) -> (String, f64) {
        self.inner.min_redundancy.clone()
    }

    /// Number of observations reported as uncheckable.
    #[getter]
    fn n_uncheckable(&self) -> usize {
        self.inner.n_uncheckable
    }

    /// Return a compact representation of the reliability summary.
    fn __repr__(&self) -> String {
        format!(
            "ReliabilitySummary(n_obs={}, n_params={}, n_uncheckable={})",
            self.inner.n_obs, self.inner.n_params, self.inner.n_uncheckable
        )
    }
}

/// Full reliability design report.
#[pyclass(module = "sidereon._sidereon", name = "ReliabilityReport")]
#[derive(Clone)]
pub struct PyReliabilityReport {
    inner: ReliabilityReport,
}

impl From<ReliabilityReport> for PyReliabilityReport {
    fn from(inner: ReliabilityReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyReliabilityReport {
    /// Per-observation reliability diagnostics in input order.
    #[getter]
    fn per_observation(&self) -> Vec<PyObservationReliability> {
        self.inner
            .per_observation
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Aggregate design diagnostics.
    #[getter]
    fn summary(&self) -> PyReliabilitySummary {
        self.inner.summary.clone().into()
    }

    /// Return a compact representation of the reliability report.
    fn __repr__(&self) -> String {
        format!(
            "ReliabilityReport(observations={}, dof={})",
            self.inner.per_observation.len(),
            self.inner.summary.dof
        )
    }
}

/// Compute Baarda w-test constants.
///
/// `power` is detection probability. Pass `beta` by keyword to provide the
/// missed-detection probability directly. The returned object carries both
/// `delta0` and `lambda0`.
#[pyfunction]
#[pyo3(signature = (alpha, power=None, *, beta=None))]
fn wtest_noncentrality(
    alpha: f64,
    power: Option<f64>,
    beta: Option<f64>,
) -> PyResult<PyWtestNoncentrality> {
    let missed_detection = match (power, beta) {
        (Some(_), Some(_)) => {
            return Err(PyValueError::new_err(
                "provide either power or beta, not both",
            ))
        }
        (Some(power), None) => 1.0 - power,
        (None, Some(beta)) => beta,
        (None, None) => ReliabilityOptions::default().beta,
    };
    let lambda0 = core_wtest_noncentrality(alpha, missed_detection).map_err(to_value_err)?;
    Ok(PyWtestNoncentrality {
        delta0: lambda0.sqrt(),
        lambda0,
    })
}

/// Compute reliability from supplied range geometry.
#[pyfunction]
#[pyo3(signature = (rows, options=None))]
fn reliability_design(
    py: Python<'_>,
    rows: Vec<Py<PyRangeReliabilityRow>>,
    options: Option<&PyReliabilityOptions>,
) -> PyResult<PyReliabilityReport> {
    let rows = rows
        .iter()
        .map(|row| row.borrow(py).inner.clone())
        .collect::<Vec<_>>();
    let options = PyReliabilityOptions::inner_or_default(options);
    core_reliability_design(&rows, &options)
        .map(Into::into)
        .map_err(to_value_err)
}

/// Compute reliability for ARAIM geometry and an integrity support model.
#[pyfunction]
#[pyo3(signature = (geometry, ism, options=None))]
fn reliability_araim(
    geometry: &PyAraimGeometry,
    ism: &PyIsm,
    options: Option<&PyReliabilityOptions>,
) -> PyResult<PyReliabilityReport> {
    let options = PyReliabilityOptions::inner_or_default(options);
    core_reliability_araim(&geometry.inner, &ism.inner, &options)
        .map(Into::into)
        .map_err(to_value_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyWtestNoncentrality>()?;
    m.add_class::<PyReliabilityOptions>()?;
    m.add_class::<PyObservationReliability>()?;
    m.add_class::<PyRangeReliabilityRow>()?;
    m.add_class::<PyReliabilitySummary>()?;
    m.add_class::<PyReliabilityReport>()?;
    m.add_function(wrap_pyfunction!(wtest_noncentrality, m)?)?;
    m.add_function(wrap_pyfunction!(reliability_design, m)?)?;
    m.add_function(wrap_pyfunction!(reliability_araim, m)?)?;
    Ok(())
}
