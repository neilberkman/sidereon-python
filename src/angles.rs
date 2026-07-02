//! Angular geometry binding.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::angles as core;

use crate::marshal::{fixed_array, FinitePolicy};

fn to_angles_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
fn angular_separation(a: PyReadonlyArray1<'_, f64>, b: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    let a = fixed_array::<3>("a", &a, FinitePolicy::RequireFinite)?;
    let b = fixed_array::<3>("b", &b, FinitePolicy::RequireFinite)?;
    core::angular_separation(a, b).map_err(to_angles_err)
}

#[pyfunction]
fn angular_separation_coords(
    lon1_rad: f64,
    lat1_rad: f64,
    lon2_rad: f64,
    lat2_rad: f64,
) -> PyResult<f64> {
    core::angular_separation_coords((lon1_rad, lat1_rad), (lon2_rad, lat2_rad))
        .map_err(to_angles_err)
}

#[pyfunction]
fn position_angle(
    from_lon_rad: f64,
    from_lat_rad: f64,
    to_lon_rad: f64,
    to_lat_rad: f64,
) -> PyResult<f64> {
    core::position_angle((from_lon_rad, from_lat_rad), (to_lon_rad, to_lat_rad))
        .map_err(to_angles_err)
}

#[pyfunction]
fn beta_angle(
    orbit_normal: PyReadonlyArray1<'_, f64>,
    sun_vector: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let normal = fixed_array::<3>("orbit_normal", &orbit_normal, FinitePolicy::RequireFinite)?;
    let sun = fixed_array::<3>("sun_vector", &sun_vector, FinitePolicy::RequireFinite)?;
    core::beta_angle(normal, sun).map_err(to_angles_err)
}

#[pyfunction]
fn beta_angle_from_state(
    position_km: PyReadonlyArray1<'_, f64>,
    velocity_km_s: PyReadonlyArray1<'_, f64>,
    sun_vector: PyReadonlyArray1<'_, f64>,
) -> PyResult<f64> {
    let r = fixed_array::<3>("position_km", &position_km, FinitePolicy::RequireFinite)?;
    let v = fixed_array::<3>("velocity_km_s", &velocity_km_s, FinitePolicy::RequireFinite)?;
    let sun = fixed_array::<3>("sun_vector", &sun_vector, FinitePolicy::RequireFinite)?;
    core::beta_angle_from_state(r, v, sun).map_err(to_angles_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(angular_separation, m)?)?;
    m.add_function(wrap_pyfunction!(angular_separation_coords, m)?)?;
    m.add_function(wrap_pyfunction!(position_angle, m)?)?;
    m.add_function(wrap_pyfunction!(beta_angle, m)?)?;
    m.add_function(wrap_pyfunction!(beta_angle_from_state, m)?)?;
    Ok(())
}
