//! Residual-distribution diagnostics binding: sample moments and normality
//! tests.
//!
//! Thin INTERFACE over `sidereon_core::quality::normality`. It marshals a 1-D
//! residual array into a slice and calls the core
//! [`skewness`](sidereon_core::quality::normality::skewness) /
//! [`kurtosis`](sidereon_core::quality::normality::kurtosis) /
//! [`moments`](sidereon_core::quality::normality::moments) /
//! [`jarque_bera`](sidereon_core::quality::normality::jarque_bera) /
//! [`shapiro_wilk`](sidereon_core::quality::normality::shapiro_wilk) primitives.
//! The statistics match `scipy.stats`; no statistics are computed here.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::quality::normality::{
    jarque_bera as core_jarque_bera, kurtosis as core_kurtosis, moments as core_moments,
    shapiro_wilk as core_shapiro_wilk, skewness as core_skewness, NormalityError,
};

/// Map a [`NormalityError`] to a Python `ValueError`, preserving the message.
fn to_normality_err(err: NormalityError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Read a residual array into a slice, mapping a non-contiguous array to a clear
/// error.
fn residual_slice<'a>(x: &'a PyReadonlyArray1<'_, f64>) -> PyResult<&'a [f64]> {
    x.as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Sample mean, variance, skewness, and kurtosis of a residual set.
#[pyclass(module = "sidereon._sidereon", name = "MomentStats")]
#[derive(Clone)]
pub struct PyMomentStats {
    mean: f64,
    variance: f64,
    skewness: f64,
    kurtosis_excess: f64,
}

#[pymethods]
impl PyMomentStats {
    /// Arithmetic mean.
    #[getter]
    fn mean(&self) -> f64 {
        self.mean
    }

    /// Population (biased) variance, the second central moment.
    #[getter]
    fn variance(&self) -> f64 {
        self.variance
    }

    /// Sample skewness (biased or bias-corrected per the `bias` flag).
    #[getter]
    fn skewness(&self) -> f64 {
        self.skewness
    }

    /// Sample kurtosis. Excess (Gaussian -> 0) when `fisher` is set, else Pearson
    /// (Gaussian -> 3).
    #[getter]
    fn kurtosis_excess(&self) -> f64 {
        self.kurtosis_excess
    }

    fn __repr__(&self) -> String {
        format!(
            "MomentStats(mean={}, variance={}, skewness={}, kurtosis_excess={})",
            self.mean, self.variance, self.skewness, self.kurtosis_excess
        )
    }
}

/// A normality test result: the test statistic and its upper-tail p-value.
#[pyclass(module = "sidereon._sidereon", name = "NormalityTest")]
#[derive(Clone)]
pub struct PyNormalityTest {
    statistic: f64,
    p_value: f64,
}

#[pymethods]
impl PyNormalityTest {
    /// The test statistic (Jarque-Bera `JB`, or Shapiro-Wilk `W`).
    #[getter]
    fn statistic(&self) -> f64 {
        self.statistic
    }

    /// Upper-tail p-value for the null hypothesis of normality.
    #[getter]
    fn p_value(&self) -> f64 {
        self.p_value
    }

    fn __repr__(&self) -> String {
        format!(
            "NormalityTest(statistic={}, p_value={})",
            self.statistic, self.p_value
        )
    }
}

/// Sample skewness of a residual set.
///
/// `bias=True` (default) returns the Fisher-Pearson coefficient
/// (`scipy.stats.skew`); `bias=False` applies the sample correction
/// (`scipy.stats.skew(bias=False)`, needs at least three residuals).
#[pyfunction]
#[pyo3(signature = (x, bias=true))]
fn skewness(x: PyReadonlyArray1<'_, f64>, bias: bool) -> PyResult<f64> {
    core_skewness(residual_slice(&x)?, bias).map_err(to_normality_err)
}

/// Sample kurtosis of a residual set.
///
/// `fisher=True` (default) returns the excess kurtosis `m4/m2^2 - 3`
/// (`scipy.stats.kurtosis`); `fisher=False` returns the Pearson kurtosis.
/// `bias=False` applies the sample correction (needs at least four residuals).
#[pyfunction]
#[pyo3(signature = (x, fisher=true, bias=true))]
fn kurtosis(x: PyReadonlyArray1<'_, f64>, fisher: bool, bias: bool) -> PyResult<f64> {
    core_kurtosis(residual_slice(&x)?, fisher, bias).map_err(to_normality_err)
}

/// Mean, variance, skewness, and kurtosis of a residual set in one pass.
///
/// `fisher` and `bias` select the kurtosis convention and the bias correction,
/// exactly as in [`skewness`] and [`kurtosis`].
#[pyfunction]
#[pyo3(signature = (x, fisher=true, bias=true))]
fn moments(x: PyReadonlyArray1<'_, f64>, fisher: bool, bias: bool) -> PyResult<PyMomentStats> {
    let stats = core_moments(residual_slice(&x)?, fisher, bias).map_err(to_normality_err)?;
    Ok(PyMomentStats {
        mean: stats.mean,
        variance: stats.variance,
        skewness: stats.skewness,
        kurtosis_excess: stats.kurtosis_excess,
    })
}

/// Jarque-Bera normality test on a residual set (`scipy.stats.jarque_bera`).
///
/// Uses the biased skewness and biased excess kurtosis; the p-value is the
/// closed-form chi-square(2) survival function `exp(-statistic/2)`. Needs at
/// least two residuals.
#[pyfunction]
fn jarque_bera(x: PyReadonlyArray1<'_, f64>) -> PyResult<PyNormalityTest> {
    let test = core_jarque_bera(residual_slice(&x)?).map_err(to_normality_err)?;
    Ok(PyNormalityTest {
        statistic: test.statistic,
        p_value: test.p_value,
    })
}

/// Shapiro-Wilk W test for normality (`scipy.stats.shapiro`, Royston AS R94).
///
/// `statistic` is the `W` statistic in `(0, 1]`; closer to one is more Gaussian.
/// Needs at least three residuals.
#[pyfunction]
fn shapiro_wilk(x: PyReadonlyArray1<'_, f64>) -> PyResult<PyNormalityTest> {
    let test = core_shapiro_wilk(residual_slice(&x)?).map_err(to_normality_err)?;
    Ok(PyNormalityTest {
        statistic: test.w,
        p_value: test.p_value,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMomentStats>()?;
    m.add_class::<PyNormalityTest>()?;
    m.add_function(wrap_pyfunction!(skewness, m)?)?;
    m.add_function(wrap_pyfunction!(kurtosis, m)?)?;
    m.add_function(wrap_pyfunction!(moments, m)?)?;
    m.add_function(wrap_pyfunction!(jarque_bera, m)?)?;
    m.add_function(wrap_pyfunction!(shapiro_wilk, m)?)?;
    Ok(())
}
