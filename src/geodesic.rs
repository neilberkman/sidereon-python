//! WGS84 geodesic direct and inverse binding.
//!
//! Thin marshaling over `sidereon_core::geodesic`: inputs and outputs are
//! degree angles plus metre distances, and the Karney solver stays wholly in
//! the core.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::geodesic::{
    geodesic_direct as core_geodesic_direct, geodesic_inverse as core_geodesic_inverse,
};

use crate::GeodesicError;

fn to_geodesic_err<E: std::fmt::Display>(err: E) -> PyErr {
    GeodesicError::new_err(err.to_string())
}

/// Solve the WGS84 inverse geodesic problem.
///
/// Inputs are point 1 latitude and longitude followed by point 2 latitude and
/// longitude, all degrees. Returns `(distance_m, azimuth1_deg, azimuth2_deg)`.
/// Raises `GeodesicError` when an input is non-finite or a latitude is outside
/// `[-90, 90]` degrees.
#[pyfunction]
fn geodesic_inverse(
    lat1_deg: f64,
    lon1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
) -> PyResult<(f64, f64, f64)> {
    core_geodesic_inverse(lat1_deg, lon1_deg, lat2_deg, lon2_deg).map_err(to_geodesic_err)
}

/// Solve the WGS84 direct geodesic problem.
///
/// Inputs are point 1 latitude, longitude, forward azimuth, and geodesic
/// distance. Angles are degrees and distance is metres. Returns
/// `(lat2_deg, lon2_deg, azimuth2_deg)`. Raises `GeodesicError` when an input is
/// non-finite or latitude is outside `[-90, 90]` degrees.
#[pyfunction]
fn geodesic_direct(
    lat1_deg: f64,
    lon1_deg: f64,
    azi1_deg: f64,
    s12_m: f64,
) -> PyResult<(f64, f64, f64)> {
    core_geodesic_direct(lat1_deg, lon1_deg, azi1_deg, s12_m).map_err(to_geodesic_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(geodesic_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(geodesic_direct, m)?)?;
    Ok(())
}
