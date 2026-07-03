"""Geoid undulation and height conversion delegate to ``sidereon_core::geoid``."""

import math

import numpy as np
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


def test_batch_undulations_and_grid_height_conversions():
    grid = sidereon.GeoidGrid(0.0, 0.0, 1.0, 1.0, 2, 2, [0.0, 1.0, 2.0, 3.0])
    points_deg = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float64)
    np.testing.assert_allclose(
        grid.undulations_deg(points_deg),
        np.array([0.0, 1.5, 3.0]),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        grid.undulations_rad(np.radians(points_deg)),
        np.array([0.0, 1.5, 3.0]),
        rtol=0.0,
        atol=0.0,
    )
    assert grid.orthometric_height_deg(100.0, 0.5, 0.5) == pytest.approx(98.5)
    assert grid.ellipsoidal_height_deg(98.5, 0.5, 0.5) == pytest.approx(100.0)
    assert grid.orthometric_height_rad(
        100.0, math.radians(0.5), math.radians(0.5)
    ) == pytest.approx(98.5)
    assert grid.ellipsoidal_height_rad(
        98.5, math.radians(0.5), math.radians(0.5)
    ) == pytest.approx(100.0)


def test_egm96_batch_and_dac_loader_are_pinned():
    points = np.array([[40.0, -105.0], [0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(
        sidereon.egm96_undulations_deg(points),
        np.array([-17.21, 17.16]),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sidereon.egm96_undulations_rad(np.radians(points)),
        np.array([-17.21, 17.16]),
        rtol=0.0,
        atol=1e-12,
    )

    dac_bytes = (123).to_bytes(2, "big", signed=True) * (721 * 1440)
    grid = sidereon.GeoidGrid.from_egm96_dac(dac_bytes)
    assert grid.undulation_deg(40.0, -105.0) == pytest.approx(1.23)
    np.testing.assert_allclose(grid.undulations_deg(points), np.array([1.23, 1.23]))


def test_grid_rejects_sample_count_mismatch():
    with pytest.raises(ValueError):
        sidereon.GeoidGrid(0.0, 0.0, 1.0, 1.0, 2, 2, [0.0, 1.0, 2.0])
