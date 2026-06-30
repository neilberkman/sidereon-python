"""Observation-geometry helpers delegate to ``sidereon_core::astro::observation``.

The binding marshals the resolved geometry and packages the core result; these
tests check the documented limiting cases.
"""

import numpy as np
import pytest
import sidereon


def test_sub_solar_point_on_axes():
    # Sun on +x -> the sub-solar point is at (0, 0).
    point = sidereon.sub_solar_point(np.array([1.0, 0.0, 0.0]))
    assert point.latitude_deg == pytest.approx(0.0, abs=1e-9)
    assert point.longitude_deg == pytest.approx(0.0, abs=1e-9)
    # Sun on +z -> sub-solar latitude is +90.
    pole = sidereon.sub_solar_point(np.array([0.0, 0.0, 5.0]))
    assert pole.latitude_deg == pytest.approx(90.0, abs=1e-9)


def test_terminator_crosses_equator_at_quadrature():
    sub_solar = sidereon.SurfacePoint(latitude_deg=23.0, longitude_deg=0.0)
    # 90 degrees from the sub-solar meridian the terminator crosses the equator.
    lat = sidereon.terminator_latitude_deg(sub_solar, 90.0)
    assert lat == pytest.approx(0.0, abs=1e-6)


def test_parallactic_angle_zero_on_meridian():
    # On the meridian (hour angle 0) the parallactic angle is 0.
    assert sidereon.parallactic_angle_deg(45.0, 0.0, 10.0) == pytest.approx(
        0.0, abs=1e-9
    )


def test_visual_magnitude_dims_with_range_and_phase():
    near = sidereon.satellite_visual_magnitude(1000.0, 0.0, 7.0, 1000.0)
    far = sidereon.satellite_visual_magnitude(2000.0, 0.0, 7.0, 1000.0)
    assert near == pytest.approx(7.0, abs=1e-9)  # at the reference range, zero phase
    assert far > near  # larger magnitude == fainter at greater range
    # Larger phase angle is fainter still.
    assert sidereon.satellite_visual_magnitude(1000.0, 90.0, 7.0, 1000.0) > near


def test_visual_magnitude_rejects_non_positive_range():
    with pytest.raises(ValueError):
        sidereon.satellite_visual_magnitude(0.0, 0.0, 7.0, 1000.0)


def test_sub_observer_point_returns_surface_point():
    point = sidereon.sub_observer_point(np.array([7000.0, 0.0, 0.0]), 0.0, 90.0, 0.0)
    assert -90.0 <= point.latitude_deg <= 90.0
    assert -180.0 < point.longitude_deg <= 180.0
