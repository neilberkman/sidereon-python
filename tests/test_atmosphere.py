"""NRLMSISE-00 neutral-atmosphere density.

`atmosphere_density` is a thin wrapper over
`sidereon_core::astro::atmosphere::nrlmsise00` (with local solar time from the
core `local_solar_time` helper). With an explicit `lst` it reproduces the core's
own metric reference assertion (sea-level total mass density); the derived-`lst`
default path is checked for the physical invariant that density falls with
altitude.
"""

import pytest
import sidereon


def test_metric_sea_level_matches_core_reference():
    # Mirrors sidereon-core's `nrlmsise00_metric_units` test: reference case 10,
    # sea level, lst pinned to 16.0 h. Total mass density 1.26106566 kg/m^3.
    out = sidereon.atmosphere_density(
        g_lat_deg=60.0,
        g_long_deg=-70.0,
        alt_km=0.0,
        year=0,
        doy=172,
        sec=29000.0,
        f107=150.0,
        f107a=150.0,
        ap=4.0,
        lst=16.0,
    )
    assert abs(out.density_kg_m3 - 1.26106566111855011) < 1.0e-6
    assert 270.0 < out.temperature_k < 290.0


def test_default_lst_is_derived_and_density_falls_with_altitude():
    # No lst supplied: the binding derives it from the core helper.
    def rho(alt_km):
        return sidereon.atmosphere_density(
            g_lat_deg=60.0,
            g_long_deg=-70.0,
            alt_km=alt_km,
            year=0,
            doy=172,
            sec=29000.0,
            f107=150.0,
            f107a=150.0,
            ap=4.0,
        )

    sea = rho(0.0)
    mid = rho(200.0)
    high = rho(400.0)
    for out in (sea, mid, high):
        assert out.density_kg_m3 > 0.0
        assert out.temperature_k > 0.0
    # Total mass density is monotonically decreasing with altitude here.
    assert sea.density_kg_m3 > mid.density_kg_m3 > high.density_kg_m3
    # Thermospheric temperature at 400 km is hot.
    assert 500.0 < high.temperature_k < 1500.0


def test_altitude_above_model_domain_raises():
    with pytest.raises(sidereon.SidereonError):
        sidereon.atmosphere_density(
            g_lat_deg=0.0,
            g_long_deg=0.0,
            alt_km=2000.0,  # above the NRLMSISE-00 1000 km domain
            year=0,
            doy=172,
            sec=29000.0,
            f107=150.0,
            f107a=150.0,
            ap=4.0,
        )
