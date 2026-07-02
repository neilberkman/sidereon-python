//! Equinoctial-family element binding.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use pyo3::Bound;

use sidereon_core::astro::equinoctial as core;

use crate::elements::PyClassicalElements;
use crate::marshal::{fixed_array_from_any, FinitePolicy};
use crate::np_array;

type StateArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

fn to_equinoctial_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(module = "sidereon._sidereon", name = "RetrogradeFactor", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
/// Direction convention for equinoctial-family element conversions.
pub enum PyRetrogradeFactor {
    PROGRADE,
    RETROGRADE,
}

impl From<PyRetrogradeFactor> for core::RetrogradeFactor {
    fn from(value: PyRetrogradeFactor) -> Self {
        match value {
            PyRetrogradeFactor::PROGRADE => Self::Prograde,
            PyRetrogradeFactor::RETROGRADE => Self::Retrograde,
        }
    }
}

impl From<core::RetrogradeFactor> for PyRetrogradeFactor {
    fn from(value: core::RetrogradeFactor) -> Self {
        match value {
            core::RetrogradeFactor::Prograde => Self::PROGRADE,
            core::RetrogradeFactor::Retrograde => Self::RETROGRADE,
        }
    }
}

#[pymethods]
impl PyRetrogradeFactor {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::PROGRADE => "prograde",
            Self::RETROGRADE => "retrograde",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::PROGRADE => "RetrogradeFactor.PROGRADE",
            Self::RETROGRADE => "RetrogradeFactor.RETROGRADE",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "EquinoctialElements")]
#[derive(Clone, Copy)]
/// Classical equinoctial orbital elements.
///
/// `lambda_` is the mean longitude in radians. Use `lambda_rad` as an alias when a unit-bearing name reads better.
pub struct PyEquinoctialElements {
    inner: core::EquinoctialElements,
}

impl PyEquinoctialElements {
    fn from_inner(inner: core::EquinoctialElements) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEquinoctialElements {
    /// Build classical equinoctial elements.
    #[new]
    #[pyo3(signature = (a, h, k, p, q, lambda_, retrograde=PyRetrogradeFactor::PROGRADE))]
    fn new(
        a: f64,
        h: f64,
        k: f64,
        p: f64,
        q: f64,
        lambda_: f64,
        retrograde: PyRetrogradeFactor,
    ) -> Self {
        Self {
            inner: core::EquinoctialElements {
                a,
                h,
                k,
                p,
                q,
                lambda: lambda_,
                retrograde: retrograde.into(),
            },
        }
    }

    #[getter]
    fn a(&self) -> f64 {
        self.inner.a
    }

    #[getter]
    fn h(&self) -> f64 {
        self.inner.h
    }

    #[getter]
    fn k(&self) -> f64 {
        self.inner.k
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    #[getter]
    fn q(&self) -> f64 {
        self.inner.q
    }

    #[getter]
    fn lambda_(&self) -> f64 {
        self.inner.lambda
    }

    #[getter]
    fn lambda_rad(&self) -> f64 {
        self.inner.lambda
    }

    #[getter]
    fn retrograde(&self) -> PyRetrogradeFactor {
        self.inner.retrograde.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "EquinoctialElements(a={:.3}, h={:.6}, k={:.6}, p={:.6}, q={:.6}, lambda_rad={:.6}, retrograde={})",
            self.inner.a,
            self.inner.h,
            self.inner.k,
            self.inner.p,
            self.inner.q,
            self.inner.lambda,
            PyRetrogradeFactor::from(self.inner.retrograde).label()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "ModifiedEquinoctialElements")]
#[derive(Clone, Copy)]
/// Modified equinoctial orbital elements.
///
/// `l` is the true longitude in radians. Use `l_rad` as an alias when a unit-bearing name reads better.
pub struct PyModifiedEquinoctialElements {
    inner: core::ModifiedEquinoctialElements,
}

impl PyModifiedEquinoctialElements {
    fn from_inner(inner: core::ModifiedEquinoctialElements) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyModifiedEquinoctialElements {
    /// Build modified equinoctial elements.
    #[new]
    #[pyo3(signature = (p, f, g, h, k, l, retrograde=PyRetrogradeFactor::PROGRADE))]
    fn new(p: f64, f: f64, g: f64, h: f64, k: f64, l: f64, retrograde: PyRetrogradeFactor) -> Self {
        Self {
            inner: core::ModifiedEquinoctialElements {
                p,
                f,
                g,
                h,
                k,
                l,
                retrograde: retrograde.into(),
            },
        }
    }

    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    #[getter]
    fn f(&self) -> f64 {
        self.inner.f
    }

    #[getter]
    fn g(&self) -> f64 {
        self.inner.g
    }

    #[getter]
    fn h(&self) -> f64 {
        self.inner.h
    }

    #[getter]
    fn k(&self) -> f64 {
        self.inner.k
    }

    #[getter]
    fn l(&self) -> f64 {
        self.inner.l
    }

    #[getter]
    fn l_rad(&self) -> f64 {
        self.inner.l
    }

    #[getter]
    fn retrograde(&self) -> PyRetrogradeFactor {
        self.inner.retrograde.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "ModifiedEquinoctialElements(p={:.3}, f={:.6}, g={:.6}, h={:.6}, k={:.6}, l_rad={:.6}, retrograde={})",
            self.inner.p,
            self.inner.f,
            self.inner.g,
            self.inner.h,
            self.inner.k,
            self.inner.l,
            PyRetrogradeFactor::from(self.inner.retrograde).label()
        )
    }
}

#[pyfunction]
#[pyo3(signature = (elements, retrograde=PyRetrogradeFactor::PROGRADE))]
/// Convert classical orbital elements to equinoctial elements.
///
/// The optional retrograde factor selects the singularity convention.
fn coe2eq(
    elements: &PyClassicalElements,
    retrograde: PyRetrogradeFactor,
) -> PyResult<PyEquinoctialElements> {
    core::coe2eq(elements.inner(), retrograde.into())
        .map(PyEquinoctialElements::from_inner)
        .map_err(to_equinoctial_err)
}

#[pyfunction]
/// Convert equinoctial elements to classical orbital elements.
fn eq2coe(elements: &PyEquinoctialElements) -> PyResult<PyClassicalElements> {
    core::eq2coe(&elements.inner)
        .map(PyClassicalElements::from_inner)
        .map_err(to_equinoctial_err)
}

#[pyfunction]
#[pyo3(signature = (elements, retrograde=PyRetrogradeFactor::PROGRADE))]
/// Convert classical orbital elements to modified equinoctial elements.
///
/// The optional retrograde factor selects the singularity convention.
fn coe2mee(
    elements: &PyClassicalElements,
    retrograde: PyRetrogradeFactor,
) -> PyResult<PyModifiedEquinoctialElements> {
    core::coe2mee(elements.inner(), retrograde.into())
        .map(PyModifiedEquinoctialElements::from_inner)
        .map_err(to_equinoctial_err)
}

#[pyfunction]
/// Convert modified equinoctial elements to classical orbital elements.
fn mee2coe(elements: &PyModifiedEquinoctialElements) -> PyResult<PyClassicalElements> {
    core::mee2coe(&elements.inner)
        .map(PyClassicalElements::from_inner)
        .map_err(to_equinoctial_err)
}

#[pyfunction]
#[pyo3(signature = (r, v, mu, retrograde=PyRetrogradeFactor::PROGRADE))]
/// Convert position and velocity vectors to equinoctial elements.
///
/// `r` and `v` may be numpy arrays or ordinary Python sequences of three finite floats.
fn rv2eq(
    r: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    mu: f64,
    retrograde: PyRetrogradeFactor,
) -> PyResult<PyEquinoctialElements> {
    let r = fixed_array_from_any::<3>("r", r, FinitePolicy::RequireFinite)?;
    let v = fixed_array_from_any::<3>("v", v, FinitePolicy::RequireFinite)?;
    core::rv2eq(r, v, mu, retrograde.into())
        .map(PyEquinoctialElements::from_inner)
        .map_err(to_equinoctial_err)
}

#[pyfunction]
/// Convert equinoctial elements to position and velocity vectors.
///
/// Returns `(r_km, v_km_s)` as numpy arrays.
fn eq2rv<'py>(
    py: Python<'py>,
    elements: &PyEquinoctialElements,
    mu: f64,
) -> PyResult<StateArrays<'py>> {
    let (r, v) = core::eq2rv(&elements.inner, mu).map_err(to_equinoctial_err)?;
    Ok((np_array(py, &r), np_array(py, &v)))
}

#[pyfunction]
#[pyo3(signature = (r, v, mu, retrograde=PyRetrogradeFactor::PROGRADE))]
/// Convert position and velocity vectors to modified equinoctial elements.
///
/// `r` and `v` may be numpy arrays or ordinary Python sequences of three finite floats.
fn rv2mee(
    r: &Bound<'_, PyAny>,
    v: &Bound<'_, PyAny>,
    mu: f64,
    retrograde: PyRetrogradeFactor,
) -> PyResult<PyModifiedEquinoctialElements> {
    let r = fixed_array_from_any::<3>("r", r, FinitePolicy::RequireFinite)?;
    let v = fixed_array_from_any::<3>("v", v, FinitePolicy::RequireFinite)?;
    core::rv2mee(r, v, mu, retrograde.into())
        .map(PyModifiedEquinoctialElements::from_inner)
        .map_err(to_equinoctial_err)
}

#[pyfunction]
/// Convert modified equinoctial elements to position and velocity vectors.
///
/// Returns `(r_km, v_km_s)` as numpy arrays.
fn mee2rv<'py>(
    py: Python<'py>,
    elements: &PyModifiedEquinoctialElements,
    mu: f64,
) -> PyResult<StateArrays<'py>> {
    let (r, v) = core::mee2rv(&elements.inner, mu).map_err(to_equinoctial_err)?;
    Ok((np_array(py, &r), np_array(py, &v)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRetrogradeFactor>()?;
    m.add_class::<PyEquinoctialElements>()?;
    m.add_class::<PyModifiedEquinoctialElements>()?;
    m.add_function(wrap_pyfunction!(coe2eq, m)?)?;
    m.add_function(wrap_pyfunction!(eq2coe, m)?)?;
    m.add_function(wrap_pyfunction!(coe2mee, m)?)?;
    m.add_function(wrap_pyfunction!(mee2coe, m)?)?;
    m.add_function(wrap_pyfunction!(rv2eq, m)?)?;
    m.add_function(wrap_pyfunction!(eq2rv, m)?)?;
    m.add_function(wrap_pyfunction!(rv2mee, m)?)?;
    m.add_function(wrap_pyfunction!(mee2rv, m)?)?;
    Ok(())
}
