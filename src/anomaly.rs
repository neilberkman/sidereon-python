//! Kepler anomaly conversion binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::anomaly as core;

use crate::elements::PyClassicalElements;

fn to_anomaly_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(module = "sidereon._sidereon", name = "KeplerSolution")]
#[derive(Clone, Copy)]
/// Result of an iterative Kepler equation solve.
pub struct PyKeplerSolution {
    anomaly: f64,
    iterations: usize,
}

#[pymethods]
impl PyKeplerSolution {
    #[getter]
    fn anomaly(&self) -> f64 {
        self.anomaly
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "KeplerSolution(anomaly={:.16e}, iterations={})",
            self.anomaly, self.iterations
        )
    }
}

#[pyfunction]
/// Solve Kepler's equation for eccentric anomaly in radians.
///
/// The result includes the anomaly and iteration count.
fn solve_kepler(mean_anomaly_rad: f64, eccentricity: f64) -> PyResult<PyKeplerSolution> {
    core::solve_kepler(mean_anomaly_rad, eccentricity)
        .map(|solution| PyKeplerSolution {
            anomaly: solution.anomaly,
            iterations: solution.iterations,
        })
        .map_err(to_anomaly_err)
}

#[pyfunction]
/// Convert mean anomaly to eccentric anomaly, in radians.
fn mean_to_eccentric(mean_anomaly_rad: f64, eccentricity: f64) -> PyResult<f64> {
    core::mean_to_eccentric(mean_anomaly_rad, eccentricity).map_err(to_anomaly_err)
}

#[pyfunction]
/// Convert eccentric anomaly to mean anomaly, in radians.
fn eccentric_to_mean(eccentric_anomaly_rad: f64, eccentricity: f64) -> PyResult<f64> {
    core::eccentric_to_mean(eccentric_anomaly_rad, eccentricity).map_err(to_anomaly_err)
}

#[pyfunction]
/// Convert eccentric anomaly to true anomaly, in radians.
fn eccentric_to_true(eccentric_anomaly_rad: f64, eccentricity: f64) -> PyResult<f64> {
    core::eccentric_to_true(eccentric_anomaly_rad, eccentricity).map_err(to_anomaly_err)
}

#[pyfunction]
/// Convert true anomaly to eccentric anomaly, in radians.
fn true_to_eccentric(true_anomaly_rad: f64, eccentricity: f64) -> PyResult<f64> {
    core::true_to_eccentric(true_anomaly_rad, eccentricity).map_err(to_anomaly_err)
}

#[pyfunction]
/// Convert mean anomaly to true anomaly, in radians.
fn mean_to_true(mean_anomaly_rad: f64, eccentricity: f64) -> PyResult<f64> {
    core::mean_to_true(mean_anomaly_rad, eccentricity).map_err(to_anomaly_err)
}

#[pyfunction]
/// Convert true anomaly to mean anomaly, in radians.
fn true_to_mean(true_anomaly_rad: f64, eccentricity: f64) -> PyResult<f64> {
    core::true_to_mean(true_anomaly_rad, eccentricity).map_err(to_anomaly_err)
}

#[pyfunction]
/// Propagate classical elements through a two-body Kepler step.
///
/// `mu_km3_s2` is the gravitational parameter and `dt_s` is the time step.
fn propagate_kepler(
    elements: &PyClassicalElements,
    mu_km3_s2: f64,
    dt_s: f64,
) -> PyResult<PyClassicalElements> {
    core::propagate_kepler(elements.inner(), mu_km3_s2, dt_s)
        .map(PyClassicalElements::from_inner)
        .map_err(to_anomaly_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKeplerSolution>()?;
    m.add_function(wrap_pyfunction!(solve_kepler, m)?)?;
    m.add_function(wrap_pyfunction!(mean_to_eccentric, m)?)?;
    m.add_function(wrap_pyfunction!(eccentric_to_mean, m)?)?;
    m.add_function(wrap_pyfunction!(eccentric_to_true, m)?)?;
    m.add_function(wrap_pyfunction!(true_to_eccentric, m)?)?;
    m.add_function(wrap_pyfunction!(mean_to_true, m)?)?;
    m.add_function(wrap_pyfunction!(true_to_mean, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_kepler, m)?)?;
    Ok(())
}
