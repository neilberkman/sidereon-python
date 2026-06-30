"""EGM96 geoid binding tests (genuine embedded 1-degree model)."""

import math

import pytest
import sidereon


def test_undulation_height_round_trip():
    lat = math.radians(45.0)
    lon = math.radians(10.0)
    n = sidereon.egm96_undulation(lat, lon)
    # H = h - N and h = H + N are exact inverses through the same undulation.
    ortho = sidereon.egm96_orthometric_height_m(100.0, lat, lon)
    assert ortho == pytest.approx(100.0 - n)
    ellip = sidereon.egm96_ellipsoidal_height_m(ortho, lat, lon)
    assert ellip == pytest.approx(100.0)


def test_undulation_in_plausible_global_band():
    # EGM96 geoid undulation lies within roughly +/- 110 m globally.
    for lat_deg in (-80.0, -30.0, 0.0, 30.0, 80.0):
        for lon_deg in (0.0, 90.0, 180.0, 270.0):
            n = sidereon.egm96_undulation(math.radians(lat_deg), math.radians(lon_deg))
            assert -120.0 < n < 120.0


def test_egm96_differs_from_coarse_builtin():
    # The genuine 1-degree EGM96 model is a finer grid than the coarse 30-degree
    # built-in, so they generally disagree at a mid-latitude point.
    lat = math.radians(37.0)
    lon = math.radians(-122.0)
    assert sidereon.egm96_undulation(lat, lon) != sidereon.geoid_undulation(lat, lon)
