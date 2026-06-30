//! Satellite-ground Doppler and range-rate binding.
//!
//! Thin INTERFACE over `sidereon_core::astro::doppler`. It resolves the precise
//! time scales from an [`Instant`](crate::frames::PyInstant), forwards the GCRS
//! state and station geodetic coordinates to
//! [`range_rate_and_ratio`](sidereon_core::astro::doppler::range_rate_and_ratio)
//! and [`doppler_shift`](sidereon_core::astro::doppler::doppler_shift), and
//! packages the result. No frame transport, range-rate, or Doppler arithmetic
//! lives here.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::astro::doppler::{
    doppler_shift as core_doppler_shift, range_rate_and_ratio as core_range_rate_and_ratio,
    DopplerShift,
};

use crate::frames::PyInstant;
use crate::marshal::{fixed_array, FinitePolicy};
use crate::to_solve_err;

/// Range rate, Doppler ratio, and carrier Doppler shift for a satellite-ground
/// link at one epoch.
#[pyclass(module = "sidereon._sidereon", name = "DopplerShift")]
#[derive(Clone)]
pub struct PyDopplerShift {
    range_rate_km_s: f64,
    doppler_hz: f64,
    doppler_ratio: f64,
}

impl From<DopplerShift> for PyDopplerShift {
    fn from(value: DopplerShift) -> Self {
        Self {
            range_rate_km_s: value.range_rate_km_s,
            doppler_hz: value.doppler_hz,
            doppler_ratio: value.doppler_ratio,
        }
    }
}

#[pymethods]
impl PyDopplerShift {
    /// Range rate in km/s; positive means receding from the station.
    #[getter]
    fn range_rate_km_s(&self) -> f64 {
        self.range_rate_km_s
    }

    /// Carrier Doppler shift in Hz (`frequency_hz * doppler_ratio`).
    #[getter]
    fn doppler_hz(&self) -> f64 {
        self.doppler_hz
    }

    /// Dimensionless Doppler ratio; positive means approaching the station.
    #[getter]
    fn doppler_ratio(&self) -> f64 {
        self.doppler_ratio
    }

    fn __repr__(&self) -> String {
        format!(
            "DopplerShift(range_rate_km_s={:.6}, doppler_hz={:.6}, doppler_ratio={:.6e})",
            self.range_rate_km_s, self.doppler_hz, self.doppler_ratio
        )
    }
}

/// Range rate and Doppler ratio from a GCRS satellite state at `epoch`.
///
/// `gcrs_position_km` / `gcrs_velocity_km_s` are `(3,)` GCRS vectors (km and
/// km/s); the station is geodetic latitude/longitude in degrees and altitude in
/// km. Returns `(range_rate_km_s, doppler_ratio)`: positive range rate means
/// receding, positive ratio means approaching.
#[pyfunction]
#[pyo3(signature = (
    gcrs_position_km,
    gcrs_velocity_km_s,
    station_lat_deg,
    station_lon_deg,
    station_alt_km,
    epoch,
))]
fn range_rate_and_ratio(
    gcrs_position_km: PyReadonlyArray1<'_, f64>,
    gcrs_velocity_km_s: PyReadonlyArray1<'_, f64>,
    station_lat_deg: f64,
    station_lon_deg: f64,
    station_alt_km: f64,
    epoch: &PyInstant,
) -> PyResult<(f64, f64)> {
    let position = fixed_array::<3>(
        "gcrs_position_km",
        &gcrs_position_km,
        FinitePolicy::RequireFinite,
    )?;
    let velocity = fixed_array::<3>(
        "gcrs_velocity_km_s",
        &gcrs_velocity_km_s,
        FinitePolicy::RequireFinite,
    )?;
    core_range_rate_and_ratio(
        position,
        velocity,
        station_lat_deg,
        station_lon_deg,
        station_alt_km,
        &epoch.time_scales(),
    )
    .map_err(to_solve_err)
}

/// Range rate, Doppler ratio, and carrier Doppler shift from a GCRS state.
///
/// Inputs match [`range_rate_and_ratio`]; `frequency_hz` is the carrier on which
/// the shift is reported. Returns a [`DopplerShift`].
#[pyfunction]
#[pyo3(signature = (
    gcrs_position_km,
    gcrs_velocity_km_s,
    station_lat_deg,
    station_lon_deg,
    station_alt_km,
    epoch,
    frequency_hz,
))]
#[allow(clippy::too_many_arguments)]
fn doppler_shift(
    gcrs_position_km: PyReadonlyArray1<'_, f64>,
    gcrs_velocity_km_s: PyReadonlyArray1<'_, f64>,
    station_lat_deg: f64,
    station_lon_deg: f64,
    station_alt_km: f64,
    epoch: &PyInstant,
    frequency_hz: f64,
) -> PyResult<PyDopplerShift> {
    let position = fixed_array::<3>(
        "gcrs_position_km",
        &gcrs_position_km,
        FinitePolicy::RequireFinite,
    )?;
    let velocity = fixed_array::<3>(
        "gcrs_velocity_km_s",
        &gcrs_velocity_km_s,
        FinitePolicy::RequireFinite,
    )?;
    let shift = core_doppler_shift(
        position,
        velocity,
        station_lat_deg,
        station_lon_deg,
        station_alt_km,
        &epoch.time_scales(),
        frequency_hz,
    )
    .map_err(to_solve_err)?;
    Ok(PyDopplerShift::from(shift))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDopplerShift>()?;
    m.add_function(wrap_pyfunction!(range_rate_and_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(doppler_shift, m)?)?;
    Ok(())
}
