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
use pyo3::types::PyModule;

use sidereon::passes::UtcInstant;
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::atmosphere::troposphere::{
    tropo_mapping, tropo_slant, tropo_zenith, MappingModel, Met, TropoModel,
};
use sidereon_core::Wgs84Geodetic;

fn invalid<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Build the core [`Instant`] for the Niell seasonal term from a unix-microsecond
/// UTC stamp. Only the Julian date is consumed by the troposphere model; the
/// scale tag is immaterial to the day-of-year, so the engine's resolved JD split
/// is used directly.
fn instant_from_unix_micros(epoch_unix_us: i64) -> PyResult<Instant> {
    let scales = UtcInstant::from_unix_microseconds(epoch_unix_us).time_scales();
    let split = JulianDateSplit::new(scales.jd_whole, scales.tt_fraction).map_err(invalid)?;
    Ok(Instant::from_julian_date(TimeScale::Tt, split))
}

fn receiver(lat_rad: f64, lon_rad: f64, height_m: f64) -> PyResult<Wgs84Geodetic> {
    Wgs84Geodetic::new(lat_rad, lon_rad, height_m).map_err(invalid)
}

/// Zenith hydrostatic and wet tropospheric delays (positive metres).
///
/// Returns `(dry_m, wet_m)` from the Saastamoinen model: `lat_rad` and
/// `height_m` set the hydrostatic gravity correction; `pressure_hpa`,
/// `temperature_k`, and `relative_humidity` (unit fraction in `[0, 1]`) drive
/// the formulas. Raises `ValueError` on out-of-domain meteorology.
#[pyfunction]
fn tropo_zenith_delay(
    lat_rad: f64,
    height_m: f64,
    pressure_hpa: f64,
    temperature_k: f64,
    relative_humidity: f64,
) -> PyResult<(f64, f64)> {
    let rx = receiver(lat_rad, 0.0, height_m)?;
    let met = Met::new(pressure_hpa, temperature_k, relative_humidity).map_err(invalid)?;
    let z = tropo_zenith(TropoModel::Saastamoinen, rx, met).map_err(invalid)?;
    Ok((z.dry_m, z.wet_m))
}

/// Niell hydrostatic and wet mapping factors at an elevation (dimensionless).
///
/// Returns `(dry, wet)`. The mapping depends on `elevation_rad`, the receiver
/// `lat_rad` and `height_m`, and the seasonal day-of-year taken from
/// `epoch_unix_us` (unix-microsecond UTC). Raises `ValueError` below the horizon
/// or on out-of-domain input.
#[pyfunction]
fn tropo_mapping_factors(
    elevation_rad: f64,
    lat_rad: f64,
    height_m: f64,
    epoch_unix_us: i64,
) -> PyResult<(f64, f64)> {
    let rx = receiver(lat_rad, 0.0, height_m)?;
    let epoch = instant_from_unix_micros(epoch_unix_us)?;
    let m = tropo_mapping(MappingModel::Niell, elevation_rad, rx, epoch).map_err(invalid)?;
    Ok((m.dry, m.wet))
}

/// Full slant tropospheric delay (positive metres).
///
/// Composes the Saastamoinen zenith delays with the Niell mapping at
/// `elevation_rad`. The receiver `lat_rad`/`lon_rad`/`height_m` and the
/// meteorology drive the zenith terms; `epoch_unix_us` (unix-microsecond UTC)
/// sets the seasonal day-of-year. Raises `ValueError` on out-of-domain input.
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
    let met = Met::new(pressure_hpa, temperature_k, relative_humidity).map_err(invalid)?;
    let epoch = instant_from_unix_micros(epoch_unix_us)?;
    tropo_slant(elevation_rad, rx, met, epoch).map_err(invalid)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(tropo_zenith_delay, m)?)?;
    m.add_function(wrap_pyfunction!(tropo_mapping_factors, m)?)?;
    m.add_function(wrap_pyfunction!(tropo_slant_delay, m)?)?;
    Ok(())
}
