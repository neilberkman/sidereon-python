//! Station displacement tide models (solid-earth, pole, ocean loading).
//!
//! Thin INTERFACE over `sidereon_core::tides`. It marshals the ITRF station
//! vector, calendar date/hour, and the model's external drivers (Sun/Moon
//! vectors, pole coordinates, or per-station BLQ coefficients) into
//! [`solid_earth_tide`](sidereon_core::tides::solid_earth_tide),
//! [`solid_earth_pole_tide`](sidereon_core::tides::solid_earth_pole_tide), and
//! [`ocean_tide_loading`](sidereon_core::tides::ocean_tide_loading), returning
//! the displacement vectors as numpy arrays. No tide arithmetic lives here.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::tides::{
    ocean_tide_loading as core_ocean_tide_loading, solid_earth_pole_tide as core_pole_tide,
    solid_earth_tide as core_solid_earth_tide, OceanLoadingBlq, TideError, NUM_OCEAN_CONSTITUENTS,
};

use crate::marshal::{fixed_array, FinitePolicy};

/// Map a core tide failure to a Python `ValueError`, preserving the engine
/// message.
fn tide_err(err: TideError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Read a `[3][NUM_OCEAN_CONSTITUENTS]` BLQ block from a sequence of three rows.
fn blq_block(name: &str, rows: &[Vec<f64>]) -> PyResult<[[f64; NUM_OCEAN_CONSTITUENTS]; 3]> {
    if rows.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have exactly 3 rows (radial, west, south), got {}",
            rows.len()
        )));
    }
    let mut block = [[0.0; NUM_OCEAN_CONSTITUENTS]; 3];
    for (component, row) in rows.iter().enumerate() {
        if row.len() != NUM_OCEAN_CONSTITUENTS {
            return Err(PyValueError::new_err(format!(
                "{name}[{component}] must have exactly {NUM_OCEAN_CONSTITUENTS} constituents, got {}",
                row.len()
            )));
        }
        block[component].copy_from_slice(row);
    }
    Ok(block)
}

/// Solid-earth tide displacement of an ITRF station, numpy `(3,)` ECEF metres.
///
/// `station_ecef_m` is the geocentric station position (m, ITRF); `year`,
/// `month`, `day` and the UTC fractional hour `fhr` set the epoch; `sun_ecef_m`
/// and `moon_ecef_m` are the geocentric Sun and Moon positions (same units as
/// the station). Returns the displacement to project onto the line of sight.
#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, fhr, sun_ecef_m, moon_ecef_m))]
#[allow(clippy::too_many_arguments)]
fn solid_earth_tide<'py>(
    py: Python<'py>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    sun_ecef_m: PyReadonlyArray1<'_, f64>,
    moon_ecef_m: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xsta = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let xsun = fixed_array::<3>("sun_ecef_m", &sun_ecef_m, FinitePolicy::RequireFinite)?;
    let xmon = fixed_array::<3>("moon_ecef_m", &moon_ecef_m, FinitePolicy::RequireFinite)?;
    let d = core_solid_earth_tide(&xsta, year, month, day, fhr, &xsun, &xmon).map_err(tide_err)?;
    Ok(PyArray1::from_slice(py, &d))
}

/// Solid-earth pole tide displacement of an ITRF station, numpy `(3,)` ECEF
/// metres.
///
/// `xp_arcsec` / `yp_arcsec` are the IERS polar-motion coordinates in arcseconds
/// at the epoch.
#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, fhr, xp_arcsec, yp_arcsec))]
#[allow(clippy::too_many_arguments)]
fn solid_earth_pole_tide<'py>(
    py: Python<'py>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xsta = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let d = core_pole_tide(&xsta, year, month, day, fhr, xp_arcsec, yp_arcsec).map_err(tide_err)?;
    Ok(PyArray1::from_slice(py, &d))
}

/// Ocean tide loading displacement of an ITRF station, numpy `(3,)` ECEF metres.
///
/// `amplitude_m` and `phase_deg` are the station's BLQ coefficients, each three
/// rows (radial/up, west, south) of 11 constituents in BLQ column order
/// `M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa`. Amplitudes are metres, phases are
/// Greenwich phase lags in degrees.
#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, fhr, amplitude_m, phase_deg))]
#[allow(clippy::too_many_arguments)]
fn ocean_tide_loading<'py>(
    py: Python<'py>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    amplitude_m: Vec<Vec<f64>>,
    phase_deg: Vec<Vec<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xsta = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let blq = OceanLoadingBlq {
        amplitude_m: blq_block("amplitude_m", &amplitude_m)?,
        phase_deg: blq_block("phase_deg", &phase_deg)?,
    };
    let d = core_ocean_tide_loading(&xsta, year, month, day, fhr, &blq).map_err(tide_err)?;
    Ok(PyArray1::from_slice(py, &d))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solid_earth_tide, m)?)?;
    m.add_function(wrap_pyfunction!(solid_earth_pole_tide, m)?)?;
    m.add_function(wrap_pyfunction!(ocean_tide_loading, m)?)?;
    Ok(())
}
