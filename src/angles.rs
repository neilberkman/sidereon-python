//! Angular geometry binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::astro::angles as core;

use crate::marshal::{fixed_array_from_any, FinitePolicy};

fn to_angles_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
/// Return the angular separation between two 3-D vectors, in degrees.
///
/// Inputs may be numpy arrays or ordinary Python sequences of three finite floats.
fn angular_separation(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<f64> {
    let a = fixed_array_from_any::<3>("a", a, FinitePolicy::RequireFinite)?;
    let b = fixed_array_from_any::<3>("b", b, FinitePolicy::RequireFinite)?;
    core::angular_separation(a, b).map_err(to_angles_err)
}

#[pyfunction]
/// Return the great-circle angular separation between two lon/lat coordinate pairs.
///
/// Longitudes and latitudes are supplied in radians; the result is in degrees.
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
/// Return the position angle from one lon/lat coordinate to another, in degrees.
///
/// The angle is measured eastward from north at the starting coordinate.
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
/// Return the solar beta angle for an orbit normal and Sun vector, in degrees.
///
/// Inputs may be numpy arrays or ordinary Python sequences of three finite floats.
fn beta_angle(orbit_normal: &Bound<'_, PyAny>, sun_vector: &Bound<'_, PyAny>) -> PyResult<f64> {
    let normal =
        fixed_array_from_any::<3>("orbit_normal", orbit_normal, FinitePolicy::RequireFinite)?;
    let sun = fixed_array_from_any::<3>("sun_vector", sun_vector, FinitePolicy::RequireFinite)?;
    core::beta_angle(normal, sun).map_err(to_angles_err)
}

#[pyfunction]
/// Return the solar beta angle from an inertial position, velocity, and Sun vector.
///
/// Each vector may be a numpy array or an ordinary Python sequence of three floats.
fn beta_angle_from_state(
    position_km: &Bound<'_, PyAny>,
    velocity_km_s: &Bound<'_, PyAny>,
    sun_vector: &Bound<'_, PyAny>,
) -> PyResult<f64> {
    let r = fixed_array_from_any::<3>("position_km", position_km, FinitePolicy::RequireFinite)?;
    let v = fixed_array_from_any::<3>("velocity_km_s", velocity_km_s, FinitePolicy::RequireFinite)?;
    let sun = fixed_array_from_any::<3>("sun_vector", sun_vector, FinitePolicy::RequireFinite)?;
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
