//! Canonical solver defaults, surfaced verbatim from `sidereon-core`.
//!
//! These module attributes expose the engine's own default constants so callers
//! (and the binding's own option objects) read a single source of truth. The
//! values are imported directly from the core; the binding defines none of its
//! own. The drift test asserts that the option classes' default constructors
//! return exactly these values.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::astro::atmosphere::{DEFAULT_AP, DEFAULT_F107, DEFAULT_F107A};
use sidereon_core::carrier_phase::DEFAULT_HATCH_WINDOW_CAP;
use sidereon_core::positioning::{
    SurfaceMet, DEFAULT_HUBER_K, DEFAULT_ROBUST_MAX_OUTER, DEFAULT_ROBUST_OUTER_TOL_M,
    DEFAULT_ROBUST_SCALE_FLOOR_M,
};
use sidereon_core::precise_positioning::defaults as ppp_defaults;
use sidereon_core::rtk_filter::defaults as rtk_defaults;
use sidereon_core::rtk_filter::defaults::{CODE_SIGMA_M, MAX_ITERATIONS, PHASE_SIGMA_M};

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // RTK least-squares defaults (sidereon_core::rtk_filter::defaults).
    m.add("RTK_DEFAULT_MAX_ITERATIONS", MAX_ITERATIONS)?;
    m.add("RTK_DEFAULT_CODE_SIGMA_M", CODE_SIGMA_M)?;
    m.add("RTK_DEFAULT_PHASE_SIGMA_M", PHASE_SIGMA_M)?;
    m.add("RTK_DEFAULT_POSITION_TOL_M", rtk_defaults::POSITION_TOL_M)?;
    m.add("RTK_DEFAULT_AMBIGUITY_TOL_M", rtk_defaults::AMBIGUITY_TOL_M)?;
    m.add("RTK_DEFAULT_RATIO_THRESHOLD", rtk_defaults::RATIO_THRESHOLD)?;
    m.add(
        "RTK_DEFAULT_PARTIAL_MIN_AMBIGUITIES",
        rtk_defaults::PARTIAL_MIN_AMBIGUITIES,
    )?;

    // SPP robust (Huber/IRLS) defaults (sidereon_core::spp).
    m.add("SPP_DEFAULT_HUBER_K", DEFAULT_HUBER_K)?;
    m.add(
        "SPP_DEFAULT_ROBUST_SCALE_FLOOR_M",
        DEFAULT_ROBUST_SCALE_FLOOR_M,
    )?;
    m.add("SPP_DEFAULT_ROBUST_MAX_OUTER", DEFAULT_ROBUST_MAX_OUTER)?;
    m.add("SPP_DEFAULT_ROBUST_OUTER_TOL_M", DEFAULT_ROBUST_OUTER_TOL_M)?;

    // PPP solve defaults (sidereon_core::precise_positioning::defaults).
    m.add("PPP_DEFAULT_MAX_ITERATIONS", ppp_defaults::MAX_ITERATIONS)?;
    m.add(
        "PPP_DEFAULT_POSITION_TOLERANCE_M",
        ppp_defaults::POSITION_TOLERANCE_M,
    )?;
    m.add(
        "PPP_DEFAULT_CLOCK_TOLERANCE_M",
        ppp_defaults::CLOCK_TOLERANCE_M,
    )?;
    m.add(
        "PPP_DEFAULT_AMBIGUITY_TOLERANCE_M",
        ppp_defaults::AMBIGUITY_TOLERANCE_M,
    )?;
    m.add("PPP_DEFAULT_ZTD_TOLERANCE_M", ppp_defaults::ZTD_TOLERANCE_M)?;
    m.add("PPP_DEFAULT_RATIO_THRESHOLD", ppp_defaults::RATIO_THRESHOLD)?;

    // NRLMSISE-00 solar-flux/geomagnetic defaults
    // (sidereon_core::astro::atmosphere).
    m.add("ATMOSPHERE_DEFAULT_F107", DEFAULT_F107)?;
    m.add("ATMOSPHERE_DEFAULT_F107A", DEFAULT_F107A)?;
    m.add("ATMOSPHERE_DEFAULT_AP", DEFAULT_AP)?;

    // Hatch carrier-smoothing window cap (sidereon_core::carrier_phase).
    m.add("HATCH_DEFAULT_WINDOW_CAP", DEFAULT_HATCH_WINDOW_CAP)?;

    // Standard-atmosphere surface meteorology defaults
    // (sidereon_core::spp::SurfaceMet::default()). Shared by SPP troposphere
    // input, PPP troposphere options, and PPP SPP-seed auto-init.
    let surface_met = SurfaceMet::default();
    m.add("SURFACE_MET_DEFAULT_PRESSURE_HPA", surface_met.pressure_hpa)?;
    m.add(
        "SURFACE_MET_DEFAULT_TEMPERATURE_K",
        surface_met.temperature_k,
    )?;
    m.add(
        "SURFACE_MET_DEFAULT_RELATIVE_HUMIDITY",
        surface_met.relative_humidity,
    )?;
    Ok(())
}
