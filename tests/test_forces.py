"""Standalone two-body and J2 acceleration against core-generated values."""

import numpy as np
import pytest
import sidereon
from _helpers import core_goldens, hex_to_f64

POSITION_KM = [7000.0, -1210.0, 1300.0]
VELOCITY_KM_S = [0.0, 0.0, 0.0]


def _expected_acceleration(name):
    expected = core_goldens()["orbital_dynamics"]["radiation"]["standalone_gravity"]
    return np.array([hex_to_f64(value) for value in expected[name]])


def test_two_body_acceleration_is_bit_exact():
    got = sidereon.force_twobody_acceleration(POSITION_KM, VELOCITY_KM_S)
    expected = _expected_acceleration("two_body")
    assert np.array_equal(got, expected), f"got {got!r} want {expected!r}"


def test_j2_acceleration_is_bit_exact():
    got = sidereon.force_j2_acceleration(POSITION_KM, VELOCITY_KM_S)
    expected = _expected_acceleration("j2")
    assert np.array_equal(got, expected), f"got {got!r} want {expected!r}"


def test_two_body_points_to_earth_center():
    """The conservative two-body term is antiparallel to the position vector."""
    accel = sidereon.force_twobody_acceleration(POSITION_KM, VELOCITY_KM_S)
    r = np.array(POSITION_KM)
    # Acceleration along -r: the cross product vanishes and the dot is negative.
    assert np.linalg.norm(np.cross(accel, r)) < 1e-12 * np.linalg.norm(
        accel
    ) * np.linalg.norm(r)
    assert float(np.dot(accel, r)) < 0.0


def test_zero_position_raises():
    with pytest.raises(ValueError):
        sidereon.force_twobody_acceleration([0.0, 0.0, 0.0], VELOCITY_KM_S)
    with pytest.raises(ValueError):
        sidereon.force_j2_acceleration([0.0, 0.0, 0.0], VELOCITY_KM_S)
