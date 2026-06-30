//! Neutral-atmosphere density via NRLMSISE-00.
//!
//! Thin INTERFACE over `sidereon_core::astro::atmosphere`. This layer assembles
//! the [`NrlmsiseInput`](sidereon_core::astro::atmosphere::NrlmsiseInput) struct
//! and calls
//! [`nrlmsise00_with_lst`](sidereon_core::astro::atmosphere::nrlmsise00_with_lst),
//! which derives local solar time internally when `lst` is `None`, then unpacks
//! the result. All numeric logic, the local-solar-time derivation, and the
//! model's default switch configuration live in the core engine; the binding
//! adds none.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::astro::atmosphere::{nrlmsise00_with_lst, NrlmsiseInput};

use crate::to_solve_err;

/// One NRLMSISE-00 neutral-atmosphere evaluation.
#[pyclass(module = "sidereon._sidereon", name = "NeutralDensity")]
#[derive(Clone)]
pub struct PyNeutralDensity {
    density_kg_m3: f64,
    temperature_k: f64,
}

#[pymethods]
impl PyNeutralDensity {
    /// Total mass density at the requested altitude, kg/m^3.
    #[getter]
    fn density_kg_m3(&self) -> f64 {
        self.density_kg_m3
    }

    /// Temperature at the requested altitude, kelvin.
    #[getter]
    fn temperature_k(&self) -> f64 {
        self.temperature_k
    }

    fn __repr__(&self) -> String {
        format!(
            "NeutralDensity(density_kg_m3={:.6e}, temperature_k={:.3})",
            self.density_kg_m3, self.temperature_k
        )
    }
}

/// Evaluate NRLMSISE-00 neutral-atmosphere density and temperature.
///
/// `alt_km` is geodetic altitude, `g_lat_deg` / `g_long_deg` geodetic latitude
/// and longitude, `doy` the day of year (1-366), `sec` seconds in the UT day,
/// `f107` the previous-day F10.7 flux, `f107a` the centered 81-day F10.7
/// average, and `ap` the daily magnetic Ap index. Local solar time defaults to
/// the core-derived value (`sec/3600 + g_long/15`) when `lst` is omitted; pass
/// `lst` to override it. The model runs with its default switch configuration
/// (metric output).
#[pyfunction]
#[pyo3(signature = (g_lat_deg, g_long_deg, alt_km, year, doy, sec, f107, f107a, ap, lst=None))]
#[allow(clippy::too_many_arguments)]
fn atmosphere_density(
    g_lat_deg: f64,
    g_long_deg: f64,
    alt_km: f64,
    year: i32,
    doy: i32,
    sec: f64,
    f107: f64,
    f107a: f64,
    ap: f64,
    lst: Option<f64>,
) -> PyResult<PyNeutralDensity> {
    let input = NrlmsiseInput {
        year,
        doy,
        sec,
        alt: alt_km,
        g_lat: g_lat_deg,
        g_long: g_long_deg,
        // Overwritten by `nrlmsise00_with_lst`, which sets `lst` to the supplied
        // value or derives it; the placeholder here is never read.
        lst: 0.0,
        f107a,
        f107,
        ap,
        ap_array: None,
    };
    let output = nrlmsise00_with_lst(&input, lst).map_err(to_solve_err)?;
    Ok(PyNeutralDensity {
        density_kg_m3: output.density(),
        temperature_k: output.temperature_alt(),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNeutralDensity>()?;
    m.add_function(wrap_pyfunction!(atmosphere_density, m)?)?;
    Ok(())
}
