//! Standalone tropospheric-delay binding.
//!
//! Thin marshaling over [`sidereon_core::atmosphere::troposphere`]: the
//! Saastamoinen zenith hydrostatic/wet delays, the Niell (NMF) mapping factors,
//! and the composed slant delay, exposed as standalone calls rather than only as
//! a side effect of an SPP/PPP solve. No delay or mapping numerics live here;
//! the numbers are exactly what `sidereon-core` produces.
//!
//! Epochs cross the boundary as unix-microsecond UTC stamps (the binding's epoch
//! convention). The epoch only feeds the Niell seasonal day-of-year term; its
//! Julian date is taken from the engine's parity-critical UTC time-scale path.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon::passes::UtcInstant;
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeModelError, TimeScale};
use sidereon_core::atmosphere::troposphere::{
    tropo_mapping, tropo_slant, tropo_zenith, MappingModel, Met, TropoModel,
};
use sidereon_core::{Error as CoreError, FrameValueError, Wgs84Geodetic};

fn tropo_error(err: CoreError) -> PyErr {
    crate::core_error_detail::attach_core_error_detail(PyValueError::new_err(err.to_string()), &err)
}

fn attach_typed_input_detail(
    error: PyErr,
    family: &str,
    kind: &str,
    message: &str,
    field: Option<&str>,
    reason: Option<&str>,
) -> PyErr {
    Python::with_gil(|py| {
        let detail = PyDict::new(py);
        let result = (|| -> PyResult<()> {
            detail.set_item("family", family)?;
            detail.set_item("kind", kind)?;
            detail.set_item("message", message)?;
            if let Some(field) = field {
                detail.set_item("field", field)?;
            }
            if let Some(reason) = reason {
                detail.set_item("reason", reason)?;
            }
            error.value(py).setattr("detail", detail)?;
            Ok(())
        })();
        match result {
            Ok(()) => error,
            Err(py_error) => py_error,
        }
    })
}

fn frame_value_error(error: FrameValueError) -> PyErr {
    let message = error.to_string();
    match error {
        FrameValueError::InvalidInput { field, reason } => attach_typed_input_detail(
            PyValueError::new_err(message.clone()),
            "FrameValueError",
            "invalid_input",
            &message,
            Some(field),
            Some(reason),
        ),
    }
}

fn time_model_error(error: TimeModelError) -> PyErr {
    let message = error.to_string();
    match error {
        TimeModelError::InvalidInput { field, reason } => attach_typed_input_detail(
            PyValueError::new_err(message.clone()),
            "TimeModelError",
            "invalid_input",
            &message,
            Some(field),
            Some(reason),
        ),
    }
}

/// Build the core [`Instant`] for the Niell seasonal term from a unix-microsecond
/// UTC stamp. Only the Julian date is consumed by the troposphere model; the
/// scale tag is immaterial to the day-of-year, so the engine's resolved JD split
/// is used directly.
fn instant_from_unix_micros(epoch_unix_us: i64) -> PyResult<Instant> {
    let scales = UtcInstant::from_unix_microseconds(epoch_unix_us).time_scales();
    let split =
        JulianDateSplit::new(scales.jd_whole, scales.tt_fraction).map_err(time_model_error)?;
    Ok(Instant::from_julian_date(TimeScale::Tt, split))
}

fn receiver(lat_rad: f64, lon_rad: f64, height_m: f64) -> PyResult<Wgs84Geodetic> {
    Wgs84Geodetic::new(lat_rad, lon_rad, height_m).map_err(frame_value_error)
}

/// Zenith hydrostatic and wet tropospheric delays (positive metres).
///
/// Returns `(dry_m, wet_m)` from the Saastamoinen model: `lat_rad` and
/// `height_m` set the hydrostatic gravity correction; `pressure_hpa`,
/// `temperature_k`, and `relative_humidity` (unit fraction in `[0, 1]`) drive
/// the formulas. Raises `ValueError` on out-of-domain input; core refusals have
/// the complete typed `CoreError` dictionary on the exception's `detail` field.
#[pyfunction]
fn tropo_zenith_delay(
    lat_rad: f64,
    height_m: f64,
    pressure_hpa: f64,
    temperature_k: f64,
    relative_humidity: f64,
) -> PyResult<(f64, f64)> {
    let rx = receiver(lat_rad, 0.0, height_m)?;
    let met = Met::new(pressure_hpa, temperature_k, relative_humidity).map_err(tropo_error)?;
    let z = tropo_zenith(TropoModel::Saastamoinen, rx, met).map_err(tropo_error)?;
    Ok((z.dry_m, z.wet_m))
}

/// Niell hydrostatic and wet mapping factors at an elevation (dimensionless).
///
/// Returns `(dry, wet)`. The mapping depends on `elevation_rad`, the receiver
/// `lat_rad` and `height_m`, and the seasonal day-of-year taken from
/// `epoch_unix_us` (unix-microsecond UTC). Elevations outside the Niell mapping
/// validity range of 3 to 90 degrees raise `ValueError` with the core error
/// message and complete typed detail on the exception's `detail` field.
#[pyfunction]
fn tropo_mapping_factors(
    elevation_rad: f64,
    lat_rad: f64,
    height_m: f64,
    epoch_unix_us: i64,
) -> PyResult<(f64, f64)> {
    let rx = receiver(lat_rad, 0.0, height_m)?;
    let epoch = instant_from_unix_micros(epoch_unix_us)?;
    let m = tropo_mapping(MappingModel::Niell, elevation_rad, rx, epoch).map_err(tropo_error)?;
    Ok((m.dry, m.wet))
}

/// Full slant tropospheric delay (positive metres).
///
/// Composes the Saastamoinen zenith delays with the Niell mapping at
/// `elevation_rad`. The receiver `lat_rad`/`lon_rad`/`height_m` and the
/// meteorology drive the zenith terms; `epoch_unix_us` (unix-microsecond UTC)
/// sets the seasonal day-of-year. A sub-horizon (`<= 0`) elevation saturates the
/// slant to exactly `0.0` rather than erroring; out-of-domain meteorology or a
/// non-finite input still raises `ValueError`. Core refusals carry complete
/// typed detail on the exception's `detail` field.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn tropo_slant_delay(
    elevation_rad: f64,
    lat_rad: f64,
    lon_rad: f64,
    height_m: f64,
    pressure_hpa: f64,
    temperature_k: f64,
    relative_humidity: f64,
    epoch_unix_us: i64,
) -> PyResult<f64> {
    let rx = receiver(lat_rad, lon_rad, height_m)?;
    let met = Met::new(pressure_hpa, temperature_k, relative_humidity).map_err(tropo_error)?;
    let epoch = instant_from_unix_micros(epoch_unix_us)?;
    tropo_slant(elevation_rad, rx, met, epoch).map_err(tropo_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tropo_zenith_delay, m)?)?;
    m.add_function(wrap_pyfunction!(tropo_mapping_factors, m)?)?;
    m.add_function(wrap_pyfunction!(tropo_slant_delay, m)?)?;
    Ok(())
}
