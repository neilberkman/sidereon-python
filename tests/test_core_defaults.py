"""The binding's exposed solver defaults equal the core constants, bit-for-bit.

Every default the option classes construct with must come from `sidereon-core`,
not from a literal re-typed in the binding (which could silently drift from the
engine). The module-level `*_DEFAULT_*` attributes are surfaced verbatim from the
core constants (`sidereon_core::rtk_filter::defaults`,
`sidereon_core::positioning`, `sidereon_core::precise_positioning::defaults`,
`sidereon_core::astro::atmosphere`, and `sidereon_core::carrier_phase`); this
test asserts the option classes' default constructors return exactly those
values, and that the values match the documented canonical numbers.
"""

import struct

import sidereon


def _bits(x):
    """Raw IEEE-754 float64 bit pattern, for exact (bit-equal) comparison."""
    return struct.pack("<d", float(x))


def test_rtk_max_iterations_default_is_core_constant():
    # Float and fixed RTK options both default to the single core constant.
    assert (
        sidereon.RtkFloatOptions().max_iterations == sidereon.RTK_DEFAULT_MAX_ITERATIONS
    )
    assert (
        sidereon.RtkFixedOptions().max_iterations == sidereon.RTK_DEFAULT_MAX_ITERATIONS
    )
    # Canonical value (sidereon_core::rtk_filter::defaults::MAX_ITERATIONS).
    assert sidereon.RTK_DEFAULT_MAX_ITERATIONS == 10
    assert isinstance(sidereon.RTK_DEFAULT_MAX_ITERATIONS, int)


def test_rtk_measurement_sigma_defaults_are_core_constants():
    # The canonical RTKLIB-demo5 measurement sigmas, bit-equal to the core.
    assert _bits(sidereon.RTK_DEFAULT_CODE_SIGMA_M) == _bits(0.3)
    assert _bits(sidereon.RTK_DEFAULT_PHASE_SIGMA_M) == _bits(0.003)


def test_spp_robust_defaults_are_core_constants():
    cfg = sidereon.SppRobustConfig()
    # The default-constructed robust config reads each field from the core.
    assert _bits(cfg.huber_k) == _bits(sidereon.SPP_DEFAULT_HUBER_K)
    assert _bits(cfg.scale_floor_m) == _bits(sidereon.SPP_DEFAULT_ROBUST_SCALE_FLOOR_M)
    assert cfg.max_outer == sidereon.SPP_DEFAULT_ROBUST_MAX_OUTER
    assert _bits(cfg.outer_tol_m) == _bits(sidereon.SPP_DEFAULT_ROBUST_OUTER_TOL_M)


def test_spp_robust_default_canonical_values():
    # Canonical values (sidereon_core::positioning / spp::config).
    assert _bits(sidereon.SPP_DEFAULT_HUBER_K) == _bits(1.345)
    assert _bits(sidereon.SPP_DEFAULT_ROBUST_SCALE_FLOOR_M) == _bits(1.0)
    assert sidereon.SPP_DEFAULT_ROBUST_MAX_OUTER == 5
    assert _bits(sidereon.SPP_DEFAULT_ROBUST_OUTER_TOL_M) == _bits(1e-4)
    assert isinstance(sidereon.SPP_DEFAULT_ROBUST_MAX_OUTER, int)


def test_rtk_tolerance_defaults_are_core_constants():
    # Float and fixed RTK options default their tolerances to the single core
    # constants (sidereon_core::rtk_filter::defaults).
    float_opts = sidereon.RtkFloatOptions()
    fixed_opts = sidereon.RtkFixedOptions()
    assert _bits(float_opts.position_tol_m) == _bits(
        sidereon.RTK_DEFAULT_POSITION_TOL_M
    )
    assert _bits(float_opts.ambiguity_tol_m) == _bits(
        sidereon.RTK_DEFAULT_AMBIGUITY_TOL_M
    )
    assert _bits(fixed_opts.position_tol_m) == _bits(
        sidereon.RTK_DEFAULT_POSITION_TOL_M
    )
    assert _bits(fixed_opts.ambiguity_tol_m) == _bits(
        sidereon.RTK_DEFAULT_AMBIGUITY_TOL_M
    )
    assert _bits(fixed_opts.ratio_threshold) == _bits(
        sidereon.RTK_DEFAULT_RATIO_THRESHOLD
    )
    assert (
        fixed_opts.partial_min_ambiguities
        == sidereon.RTK_DEFAULT_PARTIAL_MIN_AMBIGUITIES
    )


def test_rtk_tolerance_default_canonical_values():
    # Canonical values (sidereon_core::rtk_filter::defaults).
    assert _bits(sidereon.RTK_DEFAULT_POSITION_TOL_M) == _bits(1.0e-4)
    assert _bits(sidereon.RTK_DEFAULT_AMBIGUITY_TOL_M) == _bits(1.0e-4)
    assert _bits(sidereon.RTK_DEFAULT_RATIO_THRESHOLD) == _bits(3.0)
    assert sidereon.RTK_DEFAULT_PARTIAL_MIN_AMBIGUITIES == 4
    assert isinstance(sidereon.RTK_DEFAULT_PARTIAL_MIN_AMBIGUITIES, int)


def test_ppp_defaults_are_core_constants():
    # PPP float options default every state tolerance and the iteration count to
    # the core constants (sidereon_core::precise_positioning::defaults).
    opts = sidereon.PppFloatOptions()
    assert opts.max_iterations == sidereon.PPP_DEFAULT_MAX_ITERATIONS
    assert _bits(opts.position_tolerance_m) == _bits(
        sidereon.PPP_DEFAULT_POSITION_TOLERANCE_M
    )
    assert _bits(opts.clock_tolerance_m) == _bits(
        sidereon.PPP_DEFAULT_CLOCK_TOLERANCE_M
    )
    assert _bits(opts.ambiguity_tolerance_m) == _bits(
        sidereon.PPP_DEFAULT_AMBIGUITY_TOLERANCE_M
    )
    assert _bits(opts.ztd_tolerance_m) == _bits(sidereon.PPP_DEFAULT_ZTD_TOLERANCE_M)
    # The fixed-ambiguity acceptance ratio defaults to the core constant too.
    fixed = sidereon.PppFixedAmbiguityOptions({}, {})
    assert _bits(fixed.ratio_threshold) == _bits(sidereon.PPP_DEFAULT_RATIO_THRESHOLD)


def test_ppp_default_canonical_values():
    # Canonical values (sidereon_core::precise_positioning::defaults).
    assert sidereon.PPP_DEFAULT_MAX_ITERATIONS == 8
    assert isinstance(sidereon.PPP_DEFAULT_MAX_ITERATIONS, int)
    assert _bits(sidereon.PPP_DEFAULT_POSITION_TOLERANCE_M) == _bits(1.0e-4)
    assert _bits(sidereon.PPP_DEFAULT_CLOCK_TOLERANCE_M) == _bits(1.0e-4)
    assert _bits(sidereon.PPP_DEFAULT_AMBIGUITY_TOLERANCE_M) == _bits(1.0e-4)
    assert _bits(sidereon.PPP_DEFAULT_ZTD_TOLERANCE_M) == _bits(1.0e-4)
    assert _bits(sidereon.PPP_DEFAULT_RATIO_THRESHOLD) == _bits(3.0)


def test_atmosphere_solar_flux_default_canonical_values():
    # Canonical NRLMSISE-00 solar-flux/geomagnetic defaults
    # (sidereon_core::astro::atmosphere).
    assert _bits(sidereon.ATMOSPHERE_DEFAULT_F107) == _bits(150.0)
    assert _bits(sidereon.ATMOSPHERE_DEFAULT_F107A) == _bits(150.0)
    assert _bits(sidereon.ATMOSPHERE_DEFAULT_AP) == _bits(4.0)


def test_hatch_window_cap_default_canonical_value():
    # Canonical Hatch carrier-smoothing window cap
    # (sidereon_core::carrier_phase::DEFAULT_HATCH_WINDOW_CAP).
    assert sidereon.HATCH_DEFAULT_WINDOW_CAP == 100
    assert isinstance(sidereon.HATCH_DEFAULT_WINDOW_CAP, int)


def test_surface_met_defaults_are_core_constants():
    # Every met-carrying option object that exposes a surface-meteorology default
    # reads the single core triad (sidereon_core::spp::SurfaceMet::default()), so
    # the binding holds no divergent copy of the standard-atmosphere values.
    met = sidereon.SppSurfaceMet()
    assert _bits(met.pressure_hpa) == _bits(sidereon.SURFACE_MET_DEFAULT_PRESSURE_HPA)
    assert _bits(met.temperature_k) == _bits(sidereon.SURFACE_MET_DEFAULT_TEMPERATURE_K)
    assert _bits(met.relative_humidity) == _bits(
        sidereon.SURFACE_MET_DEFAULT_RELATIVE_HUMIDITY
    )

    # The PPP troposphere options carry the same triad when enabled (met is only
    # constructed in the enabled branch).
    tropo = sidereon.PppTroposphereOptions(enabled=True)
    assert _bits(tropo.pressure_hpa) == _bits(sidereon.SURFACE_MET_DEFAULT_PRESSURE_HPA)
    assert _bits(tropo.temperature_k) == _bits(
        sidereon.SURFACE_MET_DEFAULT_TEMPERATURE_K
    )
    assert _bits(tropo.relative_humidity) == _bits(
        sidereon.SURFACE_MET_DEFAULT_RELATIVE_HUMIDITY
    )


def test_surface_met_default_canonical_values():
    # Canonical standard-atmosphere values
    # (sidereon_core::spp::SurfaceMet::default()).
    assert _bits(sidereon.SURFACE_MET_DEFAULT_PRESSURE_HPA) == _bits(1013.25)
    assert _bits(sidereon.SURFACE_MET_DEFAULT_TEMPERATURE_K) == _bits(288.15)
    assert _bits(sidereon.SURFACE_MET_DEFAULT_RELATIVE_HUMIDITY) == _bits(0.5)
