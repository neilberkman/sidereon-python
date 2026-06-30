//! Leap-second accessor binding.
//!
//! Thin INTERFACE over `sidereon_core`'s time-scale offset accessors
//! ([`gps_utc_offset_s`](sidereon_core::astro::time::scales::gps_utc_offset_s) and
//! [`tai_utc_offset_s`](sidereon_core::astro::time::scales::tai_utc_offset_s)).
//! Each takes a UTC Julian date and returns the offset in seconds; no leap-second
//! table lives here.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::time::scales::{
    gps_utc_offset_s as core_gps_utc_offset_s, tai_utc_offset_s as core_tai_utc_offset_s,
};

/// GPS - UTC (the GNSS leap-second offset since the GPS epoch) at a UTC Julian
/// date, seconds. This is the IS-GPS-200 quantity (18 s from 2017-01-01 onward).
#[pyfunction]
#[pyo3(signature = (jd_utc))]
fn gps_utc_offset_s(jd_utc: f64) -> f64 {
    core_gps_utc_offset_s(jd_utc)
}

/// TAI - UTC (the IERS / Bulletin C leap-second offset) at a UTC Julian date,
/// seconds. This is `gps_utc_offset_s + 19` (37 s from 2017-01-01 onward); use
/// [`gps_utc_offset_s`] for the leap seconds a GPS receiver applies.
#[pyfunction]
#[pyo3(signature = (jd_utc))]
fn tai_utc_offset_s(jd_utc: f64) -> f64 {
    core_tai_utc_offset_s(jd_utc)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gps_utc_offset_s, m)?)?;
    m.add_function(wrap_pyfunction!(tai_utc_offset_s, m)?)?;
    Ok(())
}
