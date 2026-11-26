//! Geocentric ephemerides, precession, and coordinate time conversions.

use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::bodies::sun_moon::{self as bodies, SunMoonError};
use sidereon_core::astro::frames::transforms;
use sidereon_core::astro::time::scales;

use crate::earth_orientation::frame_error;
use crate::frames::PyInstant;
use crate::marshal::rows3_to_array;
use crate::{binding_error, SolveError};

fn sun_moon_error(error: SunMoonError) -> PyErr {
    match error {
        SunMoonError::InvalidInput { field, reason } => binding_error(
            SolveError::new_err(error.to_string()),
            "invalid_input",
            |detail| {
                detail.set_item("field", field)?;
                detail.set_item("reason", reason)
            },
        ),
        SunMoonError::FrameTransform(source) => binding_error(
            SolveError::new_err(error.to_string()),
            "frame_transform",
            |detail| {
                detail.set_item(
                    "source",
                    frame_error(source).value(detail.py()).getattr("detail")?,
                )
            },
        ),
    }
}

/// Analytic Sun and Moon positions in GCRS, in metres, at Julian centuries TT from J2000.
#[pyfunction]
fn sun_moon_gcrs(t_tt_centuries: f64) -> PyResult<([f64; 3], [f64; 3])> {
    let result = bodies::sun_moon_gcrs(t_tt_centuries).map_err(sun_moon_error)?;
    Ok((result.sun, result.moon))
}

/// Analytic Sun and Moon positions in GCRS, in metres, at a UTC instant.
#[pyfunction]
fn sun_moon_gcrs_at(instant: &PyInstant) -> PyResult<([f64; 3], [f64; 3])> {
    let result = bodies::sun_moon_gcrs_at(&instant.time_scales()?).map_err(sun_moon_error)?;
    Ok((result.sun, result.moon))
}

/// Analytic Sun and Moon positions in GCRS, in metres, at TDB seconds from J2000.
#[pyfunction]
fn sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds: f64) -> PyResult<([f64; 3], [f64; 3])> {
    let result = bodies::sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds).map_err(sun_moon_error)?;
    Ok((result.sun, result.moon))
}

/// Rotate GCRS components into the mean equator and equinox of date.
#[pyfunction]
fn gcrs_to_mean_of_date_matrix<'py>(
    py: Python<'py>,
    instant: &PyInstant,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let matrix =
        transforms::gcrs_to_mean_of_date_matrix(&instant.time_scales()?).map_err(frame_error)?;
    Ok(rows3_to_array(py, &matrix))
}

/// Rotate mean equator and equinox of date components into GCRS.
#[pyfunction]
fn mean_of_date_to_gcrs_matrix<'py>(
    py: Python<'py>,
    instant: &PyInstant,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let matrix =
        transforms::mean_of_date_to_gcrs_matrix(&instant.time_scales()?).map_err(frame_error)?;
    Ok(rows3_to_array(py, &matrix))
}

/// Return the periodic TDB offset from TT, in seconds.
#[pyfunction]
fn tdb_minus_tt_seconds(tt_j2000_s: f64) -> f64 {
    scales::tdb_minus_tt_seconds(tt_j2000_s)
}

/// Convert TT seconds from J2000 to TDB seconds from J2000.
#[pyfunction]
fn tdb_from_tt_j2000_seconds(tt_j2000_s: f64) -> f64 {
    scales::tdb_from_tt_j2000_seconds(tt_j2000_s)
}

/// Convert TDB seconds from J2000 to TT seconds from J2000.
#[pyfunction]
fn tt_from_tdb_j2000_seconds(tdb_j2000_s: f64) -> f64 {
    scales::tt_from_tdb_j2000_seconds(tdb_j2000_s)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sun_moon_gcrs, m)?)?;
    m.add_function(wrap_pyfunction!(sun_moon_gcrs_at, m)?)?;
    m.add_function(wrap_pyfunction!(sun_moon_gcrs_at_tdb_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(gcrs_to_mean_of_date_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(mean_of_date_to_gcrs_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(tdb_minus_tt_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(tdb_from_tt_j2000_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(tt_from_tdb_j2000_seconds, m)?)?;
    Ok(())
}
