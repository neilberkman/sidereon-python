//! Observational-astronomy geometry primitives.
//!
//! Thin INTERFACE over `sidereon_core::astro::observation`. It marshals the
//! already-resolved geometry (Earth-fixed vectors, angles, ephemeris-derived
//! positions) into the core sub-solar point, terminator, parallactic angle,
//! visual magnitude, and sub-observer point kernels and packages the result.
//! Every number is produced by the core; no observation geometry lives here.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::astro::observation::{
    parallactic_angle_deg as core_parallactic_angle_deg,
    satellite_visual_magnitude as core_satellite_visual_magnitude,
    sub_observer_point as core_sub_observer_point, sub_solar_point as core_sub_solar_point,
    terminator_latitude_deg as core_terminator_latitude_deg, SurfacePoint,
};

use crate::marshal::{fixed_array, FinitePolicy};

fn to_obs_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// A point on a body surface as geocentric latitude and longitude (degrees).
#[pyclass(module = "sidereon._sidereon", name = "SurfacePoint")]
#[derive(Clone, Copy)]
pub struct PySurfacePoint {
    inner: SurfacePoint,
}

impl PySurfacePoint {
    fn from_inner(inner: SurfacePoint) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySurfacePoint {
    /// Build a surface point from geocentric latitude/longitude (degrees).
    #[new]
    fn new(latitude_deg: f64, longitude_deg: f64) -> Self {
        Self {
            inner: SurfacePoint {
                latitude_deg,
                longitude_deg,
            },
        }
    }

    /// Geocentric latitude, degrees on `[-90, 90]`.
    #[getter]
    fn latitude_deg(&self) -> f64 {
        self.inner.latitude_deg
    }

    /// Longitude, degrees on `(-180, 180]`.
    #[getter]
    fn longitude_deg(&self) -> f64 {
        self.inner.longitude_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "SurfacePoint(latitude_deg={:.6}, longitude_deg={:.6})",
            self.inner.latitude_deg, self.inner.longitude_deg
        )
    }
}

/// Sub-solar point: the geographic point where the Sun is at the zenith.
///
/// `sun_ecef` is the geocentric Sun position `(3,)` in an Earth-fixed frame (any
/// length unit; only its direction matters). Raises `ValueError` on a zero or
/// non-finite vector.
#[pyfunction]
#[pyo3(signature = (sun_ecef))]
fn sub_solar_point(sun_ecef: PyReadonlyArray1<'_, f64>) -> PyResult<PySurfacePoint> {
    let sun_ecef = fixed_array::<3>("sun_ecef", &sun_ecef, FinitePolicy::RequireFinite)?;
    core_sub_solar_point(sun_ecef)
        .map(PySurfacePoint::from_inner)
        .map_err(to_obs_err)
}

/// Latitude (degrees) of the day-night terminator at a given longitude.
///
/// `sub_solar` is the sub-solar point; `longitude_deg` the query longitude.
/// Raises `ValueError` on non-finite input.
#[pyfunction]
#[pyo3(signature = (sub_solar, longitude_deg))]
fn terminator_latitude_deg(sub_solar: &PySurfacePoint, longitude_deg: f64) -> PyResult<f64> {
    core_terminator_latitude_deg(sub_solar.inner, longitude_deg).map_err(to_obs_err)
}

/// Parallactic angle (degrees) of a target at a station.
///
/// `observer_latitude_deg` is the observer geodetic latitude, `hour_angle_deg`
/// the local hour angle (positive west of the meridian), and
/// `declination_deg` the target declination. Raises `ValueError` on non-finite
/// input.
#[pyfunction]
#[pyo3(signature = (observer_latitude_deg, hour_angle_deg, declination_deg))]
fn parallactic_angle_deg(
    observer_latitude_deg: f64,
    hour_angle_deg: f64,
    declination_deg: f64,
) -> PyResult<f64> {
    core_parallactic_angle_deg(observer_latitude_deg, hour_angle_deg, declination_deg)
        .map_err(to_obs_err)
}

/// Apparent visual magnitude of a sunlit body from a diffuse-sphere phase law.
///
/// `range_km` and `reference_range_km` must be positive; `phase_angle_deg` is
/// the solar phase angle (clamped to `[0, 180]`); `standard_magnitude` is the
/// body's brightness at the reference range and zero phase. Raises `ValueError`
/// on invalid input.
#[pyfunction]
#[pyo3(signature = (range_km, phase_angle_deg, standard_magnitude, reference_range_km=1000.0))]
fn satellite_visual_magnitude(
    range_km: f64,
    phase_angle_deg: f64,
    standard_magnitude: f64,
    reference_range_km: f64,
) -> PyResult<f64> {
    core_satellite_visual_magnitude(
        range_km,
        phase_angle_deg,
        standard_magnitude,
        reference_range_km,
    )
    .map_err(to_obs_err)
}

/// Sub-observer point (planetary central meridian) on a rotating body.
///
/// `observer_from_body` is the observer position `(3,)` relative to the body
/// center in an inertial (ICRF/J2000) frame; `pole_ra_deg` / `pole_dec_deg` /
/// `prime_meridian_deg` are the body's IAU orientation. Raises `ValueError` on a
/// zero or non-finite vector.
#[pyfunction]
#[pyo3(signature = (observer_from_body, pole_ra_deg, pole_dec_deg, prime_meridian_deg))]
fn sub_observer_point(
    observer_from_body: PyReadonlyArray1<'_, f64>,
    pole_ra_deg: f64,
    pole_dec_deg: f64,
    prime_meridian_deg: f64,
) -> PyResult<PySurfacePoint> {
    let observer_from_body = fixed_array::<3>(
        "observer_from_body",
        &observer_from_body,
        FinitePolicy::RequireFinite,
    )?;
    core_sub_observer_point(
        observer_from_body,
        pole_ra_deg,
        pole_dec_deg,
        prime_meridian_deg,
    )
    .map(PySurfacePoint::from_inner)
    .map_err(to_obs_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySurfacePoint>()?;
    m.add_function(wrap_pyfunction!(sub_solar_point, m)?)?;
    m.add_function(wrap_pyfunction!(terminator_latitude_deg, m)?)?;
    m.add_function(wrap_pyfunction!(parallactic_angle_deg, m)?)?;
    m.add_function(wrap_pyfunction!(satellite_visual_magnitude, m)?)?;
    m.add_function(wrap_pyfunction!(sub_observer_point, m)?)?;
    Ok(())
}
