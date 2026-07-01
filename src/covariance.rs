//! Jacobian-derived covariance and 2-D error-ellipse geometry binding.
//!
//! Thin INTERFACE over `sidereon_core`'s least-squares covariance primitives
//! ([`normal_covariance`], [`hessian_trace`], [`covariance_from_jacobian`]) and the
//! domain-neutral [`error_ellipse_2x2`]. The binding marshals numpy matrices into
//! the core's `nalgebra` types and packages the results; every number is produced
//! by the core, no linear algebra lives here.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use nalgebra::DMatrix;

use sidereon_core::astro::math::least_squares::{
    covariance_from_jacobian as core_covariance_from_jacobian, hessian_trace as core_hessian_trace,
    normal_covariance as core_normal_covariance, SolveError as CoreSolveError,
};
use sidereon_core::geometry::{error_ellipse_2x2 as core_error_ellipse_2x2, DopError};

use crate::SolveError;

/// Map a core least-squares [`CoreSolveError`]: a malformed input/shape is a
/// `ValueError`; a rank-deficient (singular) Jacobian is a [`SolveError`].
fn to_covariance_err(err: CoreSolveError) -> PyErr {
    match err {
        CoreSolveError::InvalidInput { .. } => PyValueError::new_err(err.to_string()),
        CoreSolveError::SingularJacobian => SolveError::new_err(err.to_string()),
    }
}

/// Read a 2-D numpy array into a row-major `nalgebra` [`DMatrix`].
fn dmatrix_from_array(name: &str, arr: &PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let view = arr.as_array();
    let rows = view.nrows();
    let cols = view.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for value in view.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} must contain only finite values"
            )));
        }
        flat.push(*value);
    }
    // `view.iter()` yields row-major order; `DMatrix::from_row_slice` reads the
    // same order.
    Ok(DMatrix::from_row_slice(rows, cols, &flat))
}

/// Pack an `nalgebra` [`DMatrix`] into a `(rows, cols)` numpy float64 array.
fn dmatrix_to_array<'py>(py: Python<'py>, matrix: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut out = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            out[[r, c]] = matrix[(r, c)];
        }
    }
    out.to_pyarray(py)
}

/// Parameter covariance from a design (Jacobian) matrix via the Gauss-Newton
/// normal equations `variance_scale * (J^T J)^-1`, formed from the thin SVD of
/// `J` (so the conditioning stays at `cond(J)`, not `cond(J)^2`).
///
/// `jacobian` is an `(m, n)` array with `m >= n`. Pass the post-fit reduced
/// chi-square as `variance_scale` for the fitted covariance, or `1.0` for the
/// bare cofactor. Raises `ValueError` on a rank-deficient Jacobian or a bad
/// shape. This is the same quantity `scipy.optimize.curve_fit` reports as `pcov`.
#[pyfunction]
#[pyo3(signature = (jacobian, variance_scale=1.0))]
fn normal_covariance<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray2<'_, f64>,
    variance_scale: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let jacobian = dmatrix_from_array("jacobian", &jacobian)?;
    let cov = core_normal_covariance(&jacobian, variance_scale).map_err(to_covariance_err)?;
    Ok(dmatrix_to_array(py, &cov))
}

/// Trace of the Gauss-Newton Hessian approximation `J^T J`, i.e. the sum of the
/// squared column norms of `jacobian`. No inverse is formed.
#[pyfunction]
fn hessian_trace(jacobian: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    let jacobian = dmatrix_from_array("jacobian", &jacobian)?;
    Ok(core_hessian_trace(&jacobian))
}

/// Fitted parameter covariance directly from a converged solve's design
/// (Jacobian) matrix and post-fit `cost`.
///
/// Scales `(J^T J)^-1` by the post-fit reduced chi-square `2 * cost / (m - n)`,
/// the same scale `scipy.optimize.curve_fit` applies, with the redundancy taken
/// from the Jacobian's own `(m, n)` shape. Requires positive redundancy `m > n`.
/// Delegates straight to the core `covariance_from_jacobian`.
#[pyfunction]
#[pyo3(signature = (jacobian, cost))]
fn covariance_from_jacobian<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray2<'_, f64>,
    cost: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !cost.is_finite() {
        return Err(PyValueError::new_err("cost must be finite"));
    }
    let jacobian = dmatrix_from_array("jacobian", &jacobian)?;
    let cov = core_covariance_from_jacobian(&jacobian, cost).map_err(to_covariance_err)?;
    Ok(dmatrix_to_array(py, &cov))
}

/// A 2-D confidence error ellipse from a 2x2 covariance block.
#[pyclass(module = "sidereon._sidereon", name = "ErrorEllipse")]
#[derive(Clone)]
pub struct PyErrorEllipse {
    confidence: f64,
    chi_square_scale: f64,
    semi_major: f64,
    semi_minor: f64,
    orientation_rad: f64,
}

#[pymethods]
impl PyErrorEllipse {
    /// Requested confidence probability in `(0, 1)`.
    #[getter]
    fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Two-degree-of-freedom chi-square scale `-2 ln(1 - confidence)`.
    #[getter]
    fn chi_square_scale(&self) -> f64 {
        self.chi_square_scale
    }

    /// Semi-major axis length (same unit as the square root of the covariance).
    #[getter]
    fn semi_major(&self) -> f64 {
        self.semi_major
    }

    /// Semi-minor axis length.
    #[getter]
    fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    /// Semi-major-axis orientation, radians, from the first axis toward the
    /// second.
    #[getter]
    fn orientation_rad(&self) -> f64 {
        self.orientation_rad
    }

    fn __repr__(&self) -> String {
        format!(
            "ErrorEllipse(confidence={}, semi_major={}, semi_minor={}, orientation_rad={})",
            self.confidence, self.semi_major, self.semi_minor, self.orientation_rad
        )
    }
}

/// Confidence ellipse from an arbitrary 2x2 covariance block.
///
/// `covariance` is a `(2, 2)` array; `confidence` is in `(0, 1)`. The semi-axes
/// are scaled by the two-degree-of-freedom chi-square quantile
/// `-2 ln(1 - confidence)` applied to the eigenvalues of the symmetrized block.
/// Returns an [`ErrorEllipse`] with semi-major / semi-minor / orientation.
#[pyfunction]
#[pyo3(signature = (covariance, confidence))]
fn error_ellipse_2x2(
    covariance: PyReadonlyArray2<'_, f64>,
    confidence: f64,
) -> PyResult<PyErrorEllipse> {
    let view = covariance.as_array();
    if view.shape() != [2, 2] {
        return Err(PyValueError::new_err("covariance must have shape (2, 2)"));
    }
    let block = [[view[[0, 0]], view[[0, 1]]], [view[[1, 0]], view[[1, 1]]]];
    let ellipse = core_error_ellipse_2x2(block, confidence).map_err(|err| match err {
        DopError::InvalidInput { field, reason } => {
            PyValueError::new_err(format!("invalid {field}: {reason}"))
        }
        other => SolveError::new_err(other.to_string()),
    })?;
    Ok(PyErrorEllipse {
        confidence: ellipse.confidence,
        chi_square_scale: ellipse.chi_square_scale,
        semi_major: ellipse.semi_major,
        semi_minor: ellipse.semi_minor,
        orientation_rad: ellipse.orientation_rad,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyErrorEllipse>()?;
    m.add_function(wrap_pyfunction!(normal_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(hessian_trace, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_from_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(error_ellipse_2x2, m)?)?;
    Ok(())
}
