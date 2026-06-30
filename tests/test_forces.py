"""Standalone force-model acceleration through the binding.

Two-body and J2 acceleration are pure wrappers over `sidereon_core::astro::forces`.
The bar is bit-exact against the core's own force-wrapper unit test
(`crates/sidereon-core/src/astro/forces/{two_body,j2}.rs`): the same input state
must reproduce the same IEEE-754 acceleration to the bit.
"""

import struct

import numpy as np
import pytest
import sidereon


def _bits(u64):
    return struct.unpack("<d", struct.pack("<Q", u64))[0]


# Input state shared by both core unit tests: position (km), velocity (km/s).
POSITION_KM = [7000.0, -1210.0, 1300.0]
VELOCITY_KM_S = [0.0, 0.0, 0.0]

# Frozen acceleration bits from the core force-wrapper unit tests (km/s^2).
TWO_BODY_BITS = (
    13_798_562_943_973_640_097,
    4_563_548_234_789_153_053,
    13_787_359_517_156_423_902,
)
J2_BITS = (
    13_754_131_348_549_160_135,
    4_519_025_615_523_880_849,
    13_750_824_904_549_515_386,
)


def test_two_body_acceleration_is_bit_exact():
    got = sidereon.force_twobody_acceleration(POSITION_KM, VELOCITY_KM_S)
    expected = np.array([_bits(b) for b in TWO_BODY_BITS])
    assert np.array_equal(got, expected), f"got {got!r} want {expected!r}"


def test_j2_acceleration_is_bit_exact():
    got = sidereon.force_j2_acceleration(POSITION_KM, VELOCITY_KM_S)
    expected = np.array([_bits(b) for b in J2_BITS])
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
