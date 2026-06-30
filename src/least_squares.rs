//! Generic data-driven trust-region least-squares binding.
//!
//! Thin INTERFACE over the standalone `trust-region-least-squares` crate. A
//! caller selects a built-in residual kind (linear / polynomial / exponential),
//! hands over the data arrays, and the whole trust-region iteration runs in Rust:
//! the residual and its 2-point Jacobian are evaluated entirely inside the crate,
//! so there is NO per-iteration Python callback on this path. The binding only
//! marshals the inputs into a [`DataProblem`], picks the SVD backend, calls
//! [`solve_data_problem`] / [`solve_data_problem_drop_one`], and packages the
//! [`TrfResult`]. No solver logic lives here.
//!
//! The default `backend="native"` uses the in-crate `nalgebra` SVD (a legitimate
//! independent decomposition, not bit-identical to SciPy). `backend="lapack"`
//! injects the crate's [`LapackSvd`], which loads the host LAPACK/BLAS configured
//! through the crate's environment variables
//! (`TRUST_REGION_LEAST_SQUARES_LAPACK_PATH` and the numpy BLAS path) to
//! reproduce a pinned SciPy runtime bit-for-bit.

use numpy::ndarray::Array2;
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use trust_region_least_squares::batch::{solve_data_problem_drop_one, solve_data_problem_drop_one_with};
use trust_region_least_squares::data::{
    solve_data_problem, solve_data_problem_with, BuiltinResidual, DataProblem,
};
use trust_region_least_squares::hostlapack::LapackSvd;
use trust_region_least_squares::loss::Loss;
use trust_region_least_squares::trf::{TrfError, TrfResult, XScale};

use crate::SolveError;

/// Map a [`TrfError`] to a Python exception. Shape/validation failures (empty,
/// mismatched length, bad scale) are `ValueError`; numerical backend failures are
/// [`SolveError`].
fn to_trf_err(err: TrfError) -> PyErr {
    match err {
        TrfError::Svd(_) | TrfError::InvalidSvdOutput(_) => SolveError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// Parse the SciPy `loss` selector string into the crate's [`Loss`].
fn parse_loss(loss: &str) -> PyResult<Loss> {
    match loss {
        "linear" => Ok(Loss::Linear),
        "soft_l1" => Ok(Loss::SoftL1),
        "huber" => Ok(Loss::Huber),
        "cauchy" => Ok(Loss::Cauchy),
        "arctan" => Ok(Loss::Arctan),
        other => Err(PyValueError::new_err(format!(
            "unknown loss {other:?}; expected one of 'linear', 'soft_l1', 'huber', 'cauchy', 'arctan'"
        ))),
    }
}

/// Resolve the `x_scale` argument: the float `1.0`, the string `"jac"`, or a
/// per-parameter sequence of positive finite values, into the crate's [`XScale`].
fn parse_x_scale(x_scale: &Bound<'_, PyAny>) -> PyResult<XScale> {
    if let Ok(text) = x_scale.extract::<String>() {
        return if text == "jac" {
            Ok(XScale::Jac)
        } else {
            Err(PyValueError::new_err(format!(
                "unknown x_scale {text:?}; expected 1.0, 'jac', or a sequence of positive values"
            )))
        };
    }
    if let Ok(scalar) = x_scale.extract::<f64>() {
        if scalar == 1.0 {
            return Ok(XScale::Unit);
        }
        return Err(PyValueError::new_err(
            "scalar x_scale must be 1.0; pass a per-parameter sequence for other scales",
        ));
    }
    let values: Vec<f64> = x_scale.extract().map_err(|_| {
        PyValueError::new_err("x_scale must be 1.0, 'jac', or a sequence of positive values")
    })?;
    Ok(XScale::Values(values))
}

/// Build the [`BuiltinResidual`] for the requested kind from the supplied data
/// arrays, rejecting a missing or wrong-shaped array with `ValueError`.
fn build_kind(
    kind: &str,
    a: Option<PyReadonlyArray2<'_, f64>>,
    b: Option<PyReadonlyArray1<'_, f64>>,
    t: Option<PyReadonlyArray1<'_, f64>>,
    y: Option<PyReadonlyArray1<'_, f64>>,
    degree: Option<usize>,
) -> PyResult<BuiltinResidual> {
    match kind {
        "linear" => {
            let a = a.ok_or_else(|| {
                PyValueError::new_err("kind='linear' requires the design matrix `a` (m, n)")
            })?;
            let b = b.ok_or_else(|| {
                PyValueError::new_err("kind='linear' requires the right-hand side `b` (m,)")
            })?;
            let view = a.as_array();
            let m = view.nrows();
            let n = view.ncols();
            let mut flat = Vec::with_capacity(m * n);
            for row in view.outer_iter() {
                for value in row.iter() {
                    flat.push(*value);
                }
            }
            let b = b
                .as_slice()
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .to_vec();
            Ok(BuiltinResidual::Linear { a: flat, b, m, n })
        }
        "polynomial" => {
            let t = require_pairs("polynomial", t, &y)?;
            let degree = degree.ok_or_else(|| {
                PyValueError::new_err("kind='polynomial' requires the integer `degree`")
            })?;
            let y = pairs_y(y)?;
            Ok(BuiltinResidual::Polynomial { degree, t, y })
        }
        "exponential" => {
            let t = require_pairs("exponential", t, &y)?;
            let y = pairs_y(y)?;
            Ok(BuiltinResidual::Exponential { t, y })
        }
        other => Err(PyValueError::new_err(format!(
            "unknown kind {other:?}; expected one of 'linear', 'polynomial', 'exponential'"
        ))),
    }
}

fn require_pairs(
    kind: &str,
    t: Option<PyReadonlyArray1<'_, f64>>,
    y: &Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Vec<f64>> {
    let t = t.ok_or_else(|| {
        PyValueError::new_err(format!("kind='{kind}' requires the sample abscissae `t` (m,)"))
    })?;
    if y.is_none() {
        return Err(PyValueError::new_err(format!(
            "kind='{kind}' requires the sample ordinates `y` (m,)"
        )));
    }
    Ok(t.as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .to_vec())
}

fn pairs_y(y: Option<PyReadonlyArray1<'_, f64>>) -> PyResult<Vec<f64>> {
    Ok(y
        .ok_or_else(|| {
            PyValueError::new_err("kind requires the sample ordinates `y` (m,)")
        })?
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .to_vec())
}

/// A converged trust-region least-squares solve, mirroring the fields
/// `scipy.optimize.least_squares` reports on its result object.
#[pyclass(module = "sidereon._sidereon", name = "LeastSquaresResult")]
#[derive(Clone)]
pub struct PyLeastSquaresResult {
    x: Vec<f64>,
    cost: f64,
    fun: Vec<f64>,
    jac: Vec<f64>,
    n: usize,
    grad: Vec<f64>,
    optimality: f64,
    nfev: usize,
    njev: usize,
    status: i32,
}

impl PyLeastSquaresResult {
    fn from_result(result: TrfResult) -> Self {
        let n = result.x.len();
        Self {
            x: result.x,
            cost: result.cost,
            fun: result.fun,
            jac: result.jac,
            n,
            grad: result.grad,
            optimality: result.optimality,
            nfev: result.nfev,
            njev: result.njev,
            status: result.status,
        }
    }
}

#[pymethods]
impl PyLeastSquaresResult {
    /// Solution parameter vector, numpy `(n,)` float64.
    #[getter]
    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.x)
    }

    /// Final cost `0.5 * sum(rho(residual^2))`.
    #[getter]
    fn cost(&self) -> f64 {
        self.cost
    }

    /// Residual vector at the solution, numpy `(m,)` float64.
    #[getter]
    fn fun<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.fun)
    }

    /// Jacobian at the solution, numpy `(m, n)` float64.
    #[getter]
    fn jac<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let m = if self.n == 0 { 0 } else { self.jac.len() / self.n };
        let array = Array2::from_shape_vec((m, self.n), self.jac.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(array.to_pyarray(py))
    }

    /// Gradient `J^T f` at the solution, numpy `(n,)` float64.
    #[getter]
    fn grad<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.grad)
    }

    /// First-order optimality `||grad||_inf` at the solution.
    #[getter]
    fn optimality(&self) -> f64 {
        self.optimality
    }

    /// Number of residual evaluations.
    #[getter]
    fn nfev(&self) -> usize {
        self.nfev
    }

    /// Number of Jacobian evaluations.
    #[getter]
    fn njev(&self) -> usize {
        self.njev
    }

    /// SciPy-compatible termination status: 0 max evaluations, 1 gtol, 2 ftol,
    /// 3 xtol, 4 both ftol and xtol.
    #[getter]
    fn status(&self) -> i32 {
        self.status
    }

    /// Whether the solve converged (any positive `status`).
    #[getter]
    fn success(&self) -> bool {
        self.status > 0
    }

    fn __repr__(&self) -> String {
        format!(
            "LeastSquaresResult(cost={}, optimality={}, nfev={}, njev={}, status={}, success={})",
            self.cost,
            self.optimality,
            self.nfev,
            self.njev,
            self.status,
            self.status > 0
        )
    }
}

/// The result of a leave-one-out sweep: the base solve plus one re-solve per
/// masked residual row, and the per-row cost deltas (the RAIM/FDE pattern).
#[pyclass(module = "sidereon._sidereon", name = "LeastSquaresDropOneReport")]
#[derive(Clone)]
pub struct PyLeastSquaresDropOneReport {
    base: PyLeastSquaresResult,
    drops: Vec<PyLeastSquaresResult>,
    cost_delta: Vec<f64>,
}

#[pymethods]
impl PyLeastSquaresDropOneReport {
    /// The solve over the full residual.
    #[getter]
    fn base(&self) -> PyLeastSquaresResult {
        self.base.clone()
    }

    /// One re-solve per masked residual row, in residual-row order.
    #[getter]
    fn drops(&self) -> Vec<PyLeastSquaresResult> {
        self.drops.clone()
    }

    /// `cost_delta[i] = drops[i].cost - base.cost`, numpy `(m,)` float64.
    #[getter]
    fn cost_delta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.cost_delta)
    }

    fn __len__(&self) -> usize {
        self.drops.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "LeastSquaresDropOneReport(base_cost={}, n_drops={})",
            self.base.cost,
            self.drops.len()
        )
    }
}

/// Assemble a [`DataProblem`] from the marshalled inputs.
#[allow(clippy::too_many_arguments)]
fn build_problem(
    kind: &str,
    x0: Vec<f64>,
    a: Option<PyReadonlyArray2<'_, f64>>,
    b: Option<PyReadonlyArray1<'_, f64>>,
    t: Option<PyReadonlyArray1<'_, f64>>,
    y: Option<PyReadonlyArray1<'_, f64>>,
    degree: Option<usize>,
    loss: &str,
    f_scale: f64,
    x_scale: &Bound<'_, PyAny>,
    ftol: f64,
    xtol: f64,
    gtol: f64,
    max_nfev: Option<usize>,
) -> PyResult<DataProblem> {
    let residual = build_kind(kind, a, b, t, y, degree)?;
    let mut problem = DataProblem::new(residual, x0);
    problem.loss = parse_loss(loss)?;
    problem.f_scale = f_scale;
    problem.x_scale = parse_x_scale(x_scale)?;
    problem.ftol = ftol;
    problem.xtol = xtol;
    problem.gtol = gtol;
    problem.max_nfev = max_nfev;
    Ok(problem)
}

/// Solve a built-in data-driven least-squares problem.
///
/// `kind` selects the residual model: `'linear'` (pass `a` (m, n) and `b` (m,)),
/// `'polynomial'` (pass `t`, `y`, and `degree`), or `'exponential'` (pass `t`
/// and `y`, fitting `amp * exp(rate * t) + offset`). `x0` is the starting
/// parameter vector. The residual and its 2-point Jacobian are evaluated entirely
/// in Rust; no Python callback runs inside the trust-region loop.
///
/// `loss` is one of `'linear'`, `'soft_l1'`, `'huber'`, `'cauchy'`, `'arctan'`;
/// `f_scale` rescales the residual for a robust loss; `x_scale` is `1.0`,
/// `'jac'`, or a per-parameter sequence. `backend='native'` uses the in-crate
/// SVD; `backend='lapack'` injects the host LAPACK/BLAS for bit-for-bit SciPy
/// parity (configured via the crate's environment variables).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    kind,
    x0,
    *,
    a=None,
    b=None,
    t=None,
    y=None,
    degree=None,
    loss="linear",
    f_scale=1.0,
    x_scale=None,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-10,
    max_nfev=None,
    backend="native",
))]
fn least_squares(
    py: Python<'_>,
    kind: &str,
    x0: Vec<f64>,
    a: Option<PyReadonlyArray2<'_, f64>>,
    b: Option<PyReadonlyArray1<'_, f64>>,
    t: Option<PyReadonlyArray1<'_, f64>>,
    y: Option<PyReadonlyArray1<'_, f64>>,
    degree: Option<usize>,
    loss: &str,
    f_scale: f64,
    x_scale: Option<&Bound<'_, PyAny>>,
    ftol: f64,
    xtol: f64,
    gtol: f64,
    max_nfev: Option<usize>,
    backend: &str,
) -> PyResult<PyLeastSquaresResult> {
    let unit = 1.0f64.into_pyobject(py)?.into_any();
    let x_scale = x_scale.unwrap_or(&unit);
    let problem = build_problem(
        kind, x0, a, b, t, y, degree, loss, f_scale, x_scale, ftol, xtol, gtol, max_nfev,
    )?;
    // The problem is fully owned (the input arrays were copied into `problem`
    // under the GIL), so the trust-region iteration runs with the GIL released
    // and reacquires it only to package the result.
    let result = match backend {
        "native" => py
            .allow_threads(|| solve_data_problem(&problem))
            .map_err(to_trf_err)?,
        "lapack" => py
            .allow_threads(|| solve_data_problem_with(&problem, &LapackSvd::from_env()))
            .map_err(to_trf_err)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown backend {other:?}; expected 'native' or 'lapack'"
            )))
        }
    };
    Ok(PyLeastSquaresResult::from_result(result))
}

/// Leave-one-out (jackknife / RAIM-FDE) over a built-in data-driven problem.
///
/// Solves the base problem over all rows, then re-solves with each residual row
/// masked in turn, fanning the independent re-solves across a thread pool. The
/// per-row solve at index `i` is bit-identical to an independent serial drop-`i`
/// solve. Arguments match [`least_squares`].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    kind,
    x0,
    *,
    a=None,
    b=None,
    t=None,
    y=None,
    degree=None,
    loss="linear",
    f_scale=1.0,
    x_scale=None,
    ftol=1e-8,
    xtol=1e-8,
    gtol=1e-10,
    max_nfev=None,
    backend="native",
))]
fn least_squares_drop_one(
    py: Python<'_>,
    kind: &str,
    x0: Vec<f64>,
    a: Option<PyReadonlyArray2<'_, f64>>,
    b: Option<PyReadonlyArray1<'_, f64>>,
    t: Option<PyReadonlyArray1<'_, f64>>,
    y: Option<PyReadonlyArray1<'_, f64>>,
    degree: Option<usize>,
    loss: &str,
    f_scale: f64,
    x_scale: Option<&Bound<'_, PyAny>>,
    ftol: f64,
    xtol: f64,
    gtol: f64,
    max_nfev: Option<usize>,
    backend: &str,
) -> PyResult<PyLeastSquaresDropOneReport> {
    let unit = 1.0f64.into_pyobject(py)?.into_any();
    let x_scale = x_scale.unwrap_or(&unit);
    let problem = build_problem(
        kind, x0, a, b, t, y, degree, loss, f_scale, x_scale, ftol, xtol, gtol, max_nfev,
    )?;
    // The problem is fully owned, so the drop-one sweep (a base solve plus one
    // re-solve per masked row) runs with the GIL released and reacquires it only
    // to package the report.
    let report = match backend {
        "native" => py
            .allow_threads(|| solve_data_problem_drop_one(&problem))
            .map_err(to_trf_err)?,
        "lapack" => py
            .allow_threads(|| solve_data_problem_drop_one_with(&problem, &LapackSvd::from_env()))
            .map_err(to_trf_err)?,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown backend {other:?}; expected 'native' or 'lapack'"
            )))
        }
    };
    Ok(PyLeastSquaresDropOneReport {
        base: PyLeastSquaresResult::from_result(report.base),
        drops: report
            .drops
            .into_iter()
            .map(PyLeastSquaresResult::from_result)
            .collect(),
        cost_delta: report.cost_delta,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLeastSquaresResult>()?;
    m.add_class::<PyLeastSquaresDropOneReport>()?;
    m.add_function(wrap_pyfunction!(least_squares, m)?)?;
    m.add_function(wrap_pyfunction!(least_squares_drop_one, m)?)?;
    Ok(())
}
