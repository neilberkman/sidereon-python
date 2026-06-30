//! Geoid undulation lookup and orthometric/ellipsoidal height conversion.
//!
//! Thin INTERFACE over `sidereon_core::geoid`. It marshals the grid origin,
//! spacing, dimensions, and samples (or a grid text blob) into the core
//! [`GeoidGrid`](sidereon_core::geoid::GeoidGrid) and exposes the bilinear
//! undulation query, plus the zero-setup built-in-grid helpers
//! [`geoid_undulation`](sidereon_core::geoid::geoid_undulation) /
//! [`orthometric_height_m`](sidereon_core::geoid::orthometric_height_m) /
//! [`ellipsoidal_height_m`](sidereon_core::geoid::ellipsoidal_height_m). All
//! interpolation lives in the core; no geoid logic lives here.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::geoid::{
    ellipsoidal_height_m as core_ellipsoidal_height_m, geoid_undulation as core_geoid_undulation,
    orthometric_height_m as core_orthometric_height_m, GeoidGrid,
};

fn to_geoid_err<E: std::fmt::Debug>(err: E) -> PyErr {
    PyValueError::new_err(format!("{err:?}"))
}

/// A regular latitude/longitude grid of geoid undulation samples (metres) with
/// bilinear interpolation. Construct from explicit arrays or via
/// [`GeoidGrid.from_text`]; wraps [`sidereon_core::geoid::GeoidGrid`] unchanged.
#[pyclass(module = "sidereon._sidereon", name = "GeoidGrid")]
pub struct PyGeoidGrid {
    inner: GeoidGrid,
}

#[pymethods]
impl PyGeoidGrid {
    /// Build a geoid grid from its origin, spacing, dimensions, and row-major
    /// samples (metres). `values_m` must have exactly `n_lat * n_lon` entries in
    /// latitude-ascending-outer, longitude-ascending-inner order. Raises
    /// `ValueError` on a zero dimension, a sample-count mismatch, a non-finite or
    /// non-positive spacing, or a non-finite sample.
    #[new]
    #[pyo3(signature = (lat_min_deg, lon_min_deg, dlat_deg, dlon_deg, n_lat, n_lon, values_m))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lat_min_deg: f64,
        lon_min_deg: f64,
        dlat_deg: f64,
        dlon_deg: f64,
        n_lat: usize,
        n_lon: usize,
        values_m: Vec<f64>,
    ) -> PyResult<Self> {
        let inner = GeoidGrid::new(
            lat_min_deg,
            lon_min_deg,
            dlat_deg,
            dlon_deg,
            n_lat,
            n_lon,
            values_m,
        )
        .map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Parse a geoid grid from the documented whitespace-delimited text format:
    /// a six-field header `lat_min lon_min dlat dlon n_lat n_lon` followed by
    /// `n_lat * n_lon` undulation samples in metres. Raises `ValueError` on a
    /// malformed grid.
    #[staticmethod]
    fn from_text(text: &str) -> PyResult<Self> {
        let inner = GeoidGrid::from_text(text).map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Bilinearly interpolated undulation `N` (metres) at a geodetic position in
    /// degrees (latitude positive north, longitude positive east).
    fn undulation_deg(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.inner.undulation_deg(lat_deg, lon_deg)
    }

    /// Bilinearly interpolated undulation `N` (metres) at a geodetic position in
    /// radians (latitude positive north, longitude positive east).
    fn undulation_rad(&self, lat_rad: f64, lon_rad: f64) -> f64 {
        self.inner.undulation_rad(lat_rad, lon_rad)
    }

    fn __repr__(&self) -> String {
        "GeoidGrid(...)".to_string()
    }
}

/// Geoid undulation `N` (metres above the WGS84 ellipsoid) at a geodetic
/// position in radians, from the coarse built-in global grid.
#[pyfunction]
#[pyo3(signature = (lat_rad, lon_rad))]
fn geoid_undulation(lat_rad: f64, lon_rad: f64) -> f64 {
    core_geoid_undulation(lat_rad, lon_rad)
}

/// Orthometric height `H = h - N` (metres above mean sea level) from an
/// ellipsoidal height and a geodetic position in radians, using the built-in
/// grid's undulation.
#[pyfunction]
#[pyo3(signature = (ellipsoidal_height_m, lat_rad, lon_rad))]
fn orthometric_height_m(ellipsoidal_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    core_orthometric_height_m(ellipsoidal_height_m, lat_rad, lon_rad)
}

/// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
/// orthometric height and a geodetic position in radians, using the built-in
/// grid's undulation.
#[pyfunction]
#[pyo3(signature = (orthometric_height_m, lat_rad, lon_rad))]
fn ellipsoidal_height_m(orthometric_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    core_ellipsoidal_height_m(orthometric_height_m, lat_rad, lon_rad)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeoidGrid>()?;
    m.add_function(wrap_pyfunction!(geoid_undulation, m)?)?;
    m.add_function(wrap_pyfunction!(orthometric_height_m, m)?)?;
    m.add_function(wrap_pyfunction!(ellipsoidal_height_m, m)?)?;
    Ok(())
}
