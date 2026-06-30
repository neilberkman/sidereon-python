"""Geoid undulation and height conversion delegate to ``sidereon_core::geoid``."""

import math

import pytest
import sidereon


def test_builtin_undulation_and_height_inverse():
    lat_rad = math.radians(40.0)
    lon_rad = math.radians(-105.0)
    n = sidereon.geoid_undulation(lat_rad, lon_rad)
    assert math.isfinite(n)
    h = 1650.0  # ellipsoidal height, metres
    ortho = sidereon.orthometric_height_m(h, lat_rad, lon_rad)
    assert ortho == pytest.approx(h - n, abs=1e-9)
    # ellipsoidal_height_m is the exact inverse of orthometric_height_m.
    assert sidereon.ellipsoidal_height_m(ortho, lat_rad, lon_rad) == pytest.approx(
        h, abs=1e-9
    )


def test_grid_new_bilinear_interpolation():
    # 2x2 grid: values [[0, 1], [2, 3]] over lat 0..1, lon 0..1.
    grid = sidereon.GeoidGrid(
        lat_min_deg=0.0,
        lon_min_deg=0.0,
        dlat_deg=1.0,
        dlon_deg=1.0,
        n_lat=2,
        n_lon=2,
        values_m=[0.0, 1.0, 2.0, 3.0],
    )
    assert grid.undulation_deg(0.0, 0.0) == pytest.approx(0.0)
    assert grid.undulation_deg(1.0, 1.0) == pytest.approx(3.0)
    assert grid.undulation_deg(0.5, 0.5) == pytest.approx(1.5)
    # The radian entry agrees with the degree entry.
    assert grid.undulation_rad(math.radians(0.5), math.radians(0.5)) == pytest.approx(
        1.5
    )


def test_grid_from_text_round_trip():
    text = "0 0 1 1 2 2\n0 1 2 3\n"
    grid = sidereon.GeoidGrid.from_text(text)
    assert grid.undulation_deg(0.5, 0.5) == pytest.approx(1.5)


def test_grid_rejects_sample_count_mismatch():
    with pytest.raises(ValueError):
        sidereon.GeoidGrid(0.0, 0.0, 1.0, 1.0, 2, 2, [0.0, 1.0, 2.0])
