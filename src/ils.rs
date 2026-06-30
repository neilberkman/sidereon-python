//! Integer-ambiguity resolution kernels: standalone LAMBDA and bounded
//! integer-least-squares search.
//!
//! Thin marshaling over [`sidereon_core::ils`]. The float ambiguities cross as a
//! numpy `(n,)` array, the covariance as a numpy `(n, n)` array, and the result
//! as an `IlsResult` object. All search math - Gaussian elimination, LtDL
//! decorrelation, the MLAMBDA depth-first search, scoring and the ratio test -
//! lives in `sidereon-core`; this module adds no modeling of its own. The fixed
//! integer vector is a small-magnitude cycle count, so it surfaces as
//! `list[int]`.

use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::ils::{
    bounded_ils_search as core_bounded_ils_search, lambda_ils_search as core_lambda_ils_search,
    IlsError, IlsResult,
};

use crate::SolveError;

/// Map an [`IlsError`] onto the binding's exception taxonomy: malformed inputs
/// (shape, non-finite, out-of-domain option) raise `ValueError`; engine
/// failures (singular covariance, exhausted lattice, non-convergence) raise
/// `SolveError`.
fn to_ils_err(err: IlsError) -> PyErr {
    match err {
        IlsError::InvalidDimensions { .. }
        | IlsError::NonFinite
        | IlsError::InvalidInput { .. } => PyValueError::new_err(err.to_string()),
        IlsError::Singular
        | IlsError::NoCandidates(_)
        | IlsError::TooManyCandidates { .. }
        | IlsError::SearchLimitExceeded => SolveError::new_err(err.to_string()),
    }
}

/// Read the float ambiguity vector as a contiguous slice.
fn float_cycles_slice<'a>(float_cycles: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    float_cycles
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Convert a numpy `(n, n)` array into the row-major `Vec<Vec<f64>>` the core
/// kernels expect. Squareness and finiteness are validated by the core, so this
/// only rejects a non-square shape early for a clearer message.
fn covariance_rows(covariance: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Vec<f64>>> {
    let view = covariance.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(PyValueError::new_err(format!(
            "covariance must be a square (n, n) array, got shape {shape:?}"
        )));
    }
    Ok(view.rows().into_iter().map(|row| row.to_vec()).collect())
}

/// Outcome of an integer-least-squares search.
#[pyclass(module = "sidereon._sidereon", name = "IlsResult")]
pub struct PyIlsResult {
    inner: IlsResult,
}

#[pymethods]
impl PyIlsResult {
    /// Best integer ambiguity vector, parallel to the input `float_cycles`.
    #[getter]
    fn fixed(&self) -> Vec<i64> {
        self.inner.fixed.clone()
    }

    /// Whether the ratio test passes at the requested threshold.
    #[getter]
    fn fixed_status(&self) -> bool {
        self.inner.fixed_status
    }

    /// Runner-up / best score ratio.
    #[getter]
    fn ratio(&self) -> f64 {
        self.inner.ratio
    }

    /// Best (lowest) quadratic score `Δᵀ Q⁻¹ Δ`.
    #[getter]
    fn best_score(&self) -> f64 {
        self.inner.best_score
    }

    /// Runner-up score, or `None` when no second lattice point exists.
    #[getter]
    fn second_best_score(&self) -> Option<f64> {
        self.inner.second_best_score
    }

    /// Number of lattice points evaluated.
    #[getter]
    fn candidates_evaluated(&self) -> usize {
        self.inner.candidates_evaluated
    }

    /// Symmetrized covariance actually used, as a numpy `(n, n)` array.
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_array(py, &self.inner.covariance)
    }

    /// Symmetrized inverse covariance, as a numpy `(n, n)` array.
    #[getter]
    fn covariance_inverse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows_to_array(py, &self.inner.covariance_inverse)
    }

    fn __repr__(&self) -> String {
        format!(
            "IlsResult(fixed={:?}, fixed_status={}, ratio={}, best_score={}, \
             candidates_evaluated={})",
            self.inner.fixed,
            self.inner.fixed_status,
            self.inner.ratio,
            self.inner.best_score,
            self.inner.candidates_evaluated
        )
    }
}

/// Build a numpy `(n, n)` array from a row-major `Vec<Vec<f64>>` produced by the
/// core kernels (always square and rectangular).
fn rows_to_array<'py>(py: Python<'py>, rows: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let n = rows.len();
    let mut array = numpy::ndarray::Array2::<f64>::zeros((n, n));
    for (i, row) in rows.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            array[[i, j]] = value;
        }
    }
    PyArray2::from_owned_array(py, array)
}

/// Resolve integer ambiguities with the LAMBDA method (RTKLIB `lambda()` port).
///
/// `float_cycles` is the real-valued ambiguity vector, `covariance` its
/// `(n, n)` covariance, and `ratio_threshold` the ratio-test acceptance
/// threshold (RTKLIB's default is `3.0`). Finds the true integer-least-squares
/// optimum and runner-up for any positive-definite covariance - no search box,
/// no combinatorial blow-up. Raises `ValueError` on a malformed shape or a
/// non-finite / out-of-domain input, and `SolveError` on a singular covariance
/// or a non-converging search.
#[pyfunction]
#[pyo3(signature = (float_cycles, covariance, ratio_threshold=3.0))]
fn lambda_ils_search(
    float_cycles: PyReadonlyArray1<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
    ratio_threshold: f64,
) -> PyResult<PyIlsResult> {
    let cycles = float_cycles_slice(&float_cycles)?;
    let cov = covariance_rows(&covariance)?;
    let inner = core_lambda_ils_search(cycles, &cov, ratio_threshold).map_err(to_ils_err)?;
    Ok(PyIlsResult { inner })
}

/// Resolve integer ambiguities with a bounded lattice search.
///
/// `float_cycles` and `covariance` match [`lambda_ils_search`]. `radius` is the
/// per-ambiguity integer search half-width (the lattice spans `radius` integers
/// either side of each rounded float), `candidate_limit` caps the number of
/// lattice points evaluated before the search aborts, and `ratio_threshold` is
/// the ratio-test acceptance threshold. Correct in the weakly-correlated regime
/// and a drop-in match for `lambda_ils_search` there; on strongly-correlated
/// geometry prefer `lambda_ils_search`. Raises `ValueError` on a malformed
/// shape or a non-finite / out-of-domain input, and `SolveError` on a singular
/// covariance or a lattice that exceeds `candidate_limit`.
#[pyfunction]
#[pyo3(signature = (float_cycles, covariance, radius=1, candidate_limit=100_000, ratio_threshold=3.0))]
fn bounded_ils_search(
    float_cycles: PyReadonlyArray1<'_, f64>,
    covariance: PyReadonlyArray2<'_, f64>,
    radius: i64,
    candidate_limit: usize,
    ratio_threshold: f64,
) -> PyResult<PyIlsResult> {
    let cycles = float_cycles_slice(&float_cycles)?;
    let cov = covariance_rows(&covariance)?;
    let inner = core_bounded_ils_search(cycles, &cov, radius, candidate_limit, ratio_threshold)
        .map_err(to_ils_err)?;
    Ok(PyIlsResult { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIlsResult>()?;
    m.add_function(wrap_pyfunction!(lambda_ils_search, m)?)?;
    m.add_function(wrap_pyfunction!(bounded_ils_search, m)?)?;
    Ok(())
}
