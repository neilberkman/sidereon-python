"""Galileo NeQuick-G and the civil-time ionosphere dispatcher delegate to core.

``galileo_nequick_g_native`` is the native (no-Instant) entry; the
``ionosphere_delay_*`` functions build the epoch ``Instant`` via
``Instant::from_utc_civil`` and call the core ``ionosphere_delay`` dispatcher.
"""

import pytest
import sidereon

F_L1 = 1575.42e6
F_E1 = 1575.42e6
# Representative broadcast NeQuick-G coefficients (mid solar activity).
AI0, AI1, AI2 = 236.831641, -0.39362878, 0.00402826


def test_nequick_native_positive_and_dispersive():
    delay_l1 = sidereon.galileo_nequick_g_native(
        AI0,
        AI1,
        AI2,
        lat_deg=40.0,
        lon_deg=-3.0,
        el_deg=30.0,
        t_gal_s=43200.0,
        day_of_year=180.0,
        frequency_hz=F_E1,
    )
    assert delay_l1 > 0.0
    # Lower frequency -> larger group delay (1/f^2 dispersion).
    delay_low = sidereon.galileo_nequick_g_native(
        AI0,
        AI1,
        AI2,
        lat_deg=40.0,
        lon_deg=-3.0,
        el_deg=30.0,
        t_gal_s=43200.0,
        day_of_year=180.0,
        frequency_hz=F_E1 / 2.0,
    )
    assert delay_low > delay_l1


def test_nequick_native_rejects_bad_elevation():
    with pytest.raises(ValueError):
        sidereon.galileo_nequick_g_native(
            AI0,
            AI1,
            AI2,
            lat_deg=40.0,
            lon_deg=-3.0,
            el_deg=200.0,
            t_gal_s=43200.0,
            day_of_year=180.0,
            frequency_hz=F_E1,
        )


def test_ionosphere_delay_nequick_civil_epoch_positive():
    delay = sidereon.ionosphere_delay_nequick(
        AI0,
        AI1,
        AI2,
        lat_deg=40.0,
        lon_deg=-3.0,
        azimuth_deg=120.0,
        elevation_deg=30.0,
        year=2020,
        month=6,
        day=24,
        hour=12,
        minute=0,
        second=0.0,
        frequency_hz=F_E1,
    )
    assert delay > 0.0


def test_ionosphere_delay_klobuchar_civil_epoch_positive():
    alpha = [1.0e-8, 0.0, -6.0e-8, 0.0]
    beta = [9.0e4, 0.0, -2.0e5, 0.0]
    delay = sidereon.ionosphere_delay_klobuchar(
        alpha,
        beta,
        lat_deg=40.0,
        lon_deg=-3.0,
        azimuth_deg=120.0,
        elevation_deg=30.0,
        year=2020,
        month=6,
        day=24,
        hour=12,
        minute=0,
        second=0.0,
        frequency_hz=F_L1,
    )
    assert delay > 0.0


def test_ionosphere_delay_rejects_bad_epoch():
    with pytest.raises(ValueError):
        sidereon.ionosphere_delay_nequick(
            AI0,
            AI1,
            AI2,
            lat_deg=40.0,
            lon_deg=-3.0,
            azimuth_deg=120.0,
            elevation_deg=30.0,
            year=2020,
            month=13,
            day=99,
            hour=99,
            minute=0,
            second=0.0,
            frequency_hz=F_E1,
        )


# --- full three-dimensional NeQuick-G slant integration --------------------
#
# `nequick_g_stec_tecu` / `nequick_g_delay_m` are the full NeQuick 2 profiler
# integrated along the receiver-to-satellite ray, distinct from the compact
# broadcast-driven `galileo_nequick_g_native` above.

# A receiver near Madrid and a satellite high overhead.
_STATION = dict(station_lon_deg=-3.0, station_lat_deg=40.0, station_height_m=0.0)
_SATELLITE = dict(
    satellite_lon_deg=-3.0, satellite_lat_deg=40.0, satellite_height_m=20_200_000.0
)


def test_nequick_full_stec_positive():
    stec = sidereon.nequick_g_stec_tecu(
        AI0,
        AI1,
        AI2,
        month=6,
        utc_hours=12.0,
        **_STATION,
        **_SATELLITE,
    )
    assert stec > 0.0


def test_nequick_full_delay_matches_stec_dispersion():
    stec = sidereon.nequick_g_stec_tecu(
        AI0, AI1, AI2, month=6, utc_hours=12.0, **_STATION, **_SATELLITE
    )
    delay = sidereon.nequick_g_delay_m(
        AI0,
        AI1,
        AI2,
        month=6,
        utc_hours=12.0,
        **_STATION,
        **_SATELLITE,
        frequency_hz=F_E1,
    )
    # The delay is the dispersive 40.3e16 / f^2 mapping of the slant TEC.
    expected = stec * (40.3e16 / (F_E1 * F_E1))
    assert delay == pytest.approx(expected, rel=1e-12)
    assert delay > 0.0


def test_nequick_full_delay_dispersive_in_frequency():
    delay_l1 = sidereon.nequick_g_delay_m(
        AI0,
        AI1,
        AI2,
        month=6,
        utc_hours=12.0,
        **_STATION,
        **_SATELLITE,
        frequency_hz=F_E1,
    )
    delay_low = sidereon.nequick_g_delay_m(
        AI0,
        AI1,
        AI2,
        month=6,
        utc_hours=12.0,
        **_STATION,
        **_SATELLITE,
        frequency_hz=F_E1 / 2.0,
    )
    # Lower frequency -> larger group delay (1/f^2 dispersion).
    assert delay_low > delay_l1


def test_nequick_full_rejects_bad_month():
    with pytest.raises(ValueError):
        sidereon.nequick_g_stec_tecu(
            AI0, AI1, AI2, month=13, utc_hours=12.0, **_STATION, **_SATELLITE
        )


def test_nequick_full_rejects_bad_utc_hours():
    with pytest.raises(ValueError):
        sidereon.nequick_g_stec_tecu(
            AI0, AI1, AI2, month=6, utc_hours=25.0, **_STATION, **_SATELLITE
        )


def test_nequick_full_delay_rejects_nonpositive_frequency():
    with pytest.raises(ValueError):
        sidereon.nequick_g_delay_m(
            AI0,
            AI1,
            AI2,
            month=6,
            utc_hours=12.0,
            **_STATION,
            **_SATELLITE,
            frequency_hz=0.0,
        )
