"""Lambert transfer + initial orbit determination (Gibbs / Herrick-Gibbs /
Gauss) through the binding reproduce the crate-side Vallado reference vectors.

The expected numbers are lifted verbatim from the core unit tests
(`astro::lambert` and `astro::iod`); these only confirm the binding marshals the
numpy 3-vectors through to the same answer.
"""

import math

import numpy as np
import pytest
import sidereon

RE = 6378.1363


def assert_float64_bits(actual, expected):
    """Assert exact IEEE-754 equality, including the sign bit of zero."""
    actual_bits = np.asarray(actual, dtype=np.float64).view(np.uint64)
    expected_bits = np.asarray(expected, dtype=np.float64).view(np.uint64)
    np.testing.assert_array_equal(actual_bits, expected_bits)


def test_lambert_battin_short_high():
    # crate `battin_short_high`.
    r1 = np.array([2.5 * RE, 0.0, 0.0], dtype=np.float64)
    r2 = np.array([1.9151111 * RE, 1.6069690 * RE, 0.0], dtype=np.float64)
    v1 = np.array([0.0, 4.999792554221911, 0.0], dtype=np.float64)
    v1t, v2t = sidereon.lambert_battin(
        r1,
        r2,
        v1,
        92854.234,
        direction_of_motion="short",
        direction_of_energy="high",
        nrev=1,
    )
    np.testing.assert_allclose(
        v1t, [-0.8696153795282852, 6.3351545812502374, 0.0], atol=1e-10
    )
    np.testing.assert_allclose(
        v2t, [-3.405994961791248, 5.41198791828363, 0.0], atol=1e-10
    )


def test_lambert_battin_accepts_enum_directions():
    r1 = np.array([2.5 * RE, 0.0, 0.0], dtype=np.float64)
    r2 = np.array([1.9151111 * RE, 1.6069690 * RE, 0.0], dtype=np.float64)
    v1 = np.array([0.0, 4.999792554221911, 0.0], dtype=np.float64)
    via_str = sidereon.lambert_battin(
        r1,
        r2,
        v1,
        92854.234,
        direction_of_motion="short",
        direction_of_energy="high",
        nrev=1,
    )
    via_enum = sidereon.lambert_battin(
        r1,
        r2,
        v1,
        92854.234,
        direction_of_motion=sidereon.DirectionOfMotion.SHORT,
        direction_of_energy=sidereon.DirectionOfEnergy.HIGH,
        nrev=1,
    )
    np.testing.assert_array_equal(via_str[0], via_enum[0])
    np.testing.assert_array_equal(via_str[1], via_enum[1])


def test_lambert_battin_rejects_unknown_direction_label():
    r1 = np.array([2.5 * RE, 0.0, 0.0], dtype=np.float64)
    r2 = np.array([1.9151111 * RE, 1.6069690 * RE, 0.0], dtype=np.float64)
    v1 = np.array([0.0, 4.999792554221911, 0.0], dtype=np.float64)
    with pytest.raises(ValueError):
        sidereon.lambert_battin(r1, r2, v1, 92854.234, direction_of_motion="sideways")


def test_lambert_battin_nonpositive_tof_raises_solve_error():
    r1 = np.array([2.5 * RE, 0.0, 0.0], dtype=np.float64)
    r2 = np.array([1.9151111 * RE, 1.6069690 * RE, 0.0], dtype=np.float64)
    v1 = np.array([0.0, 4.999792554221911, 0.0], dtype=np.float64)
    with pytest.raises(sidereon.SolveError):
        sidereon.lambert_battin(r1, r2, v1, 0.0)


def test_gibbs_example_7_3():
    # crate `gibbs_example_7_3`.
    r1 = np.array([0.0, 0.0, 6378.1363], dtype=np.float64)
    r2 = np.array([0.0, -4464.696, -5102.509], dtype=np.float64)
    r3 = np.array([0.0, 5740.323, 3189.068], dtype=np.float64)
    v2, theta12, theta23, copa = sidereon.gibbs(r1, r2, r3)
    assert_float64_bits(v2, [0.0, 5.5311472050176125, -5.191806413494606])
    assert math.isclose(math.degrees(theta12), 138.81407085944375, rel_tol=1e-9)
    assert math.isclose(math.degrees(theta23), 160.24053069723146, rel_tol=1e-9)
    assert abs(copa) < 1e-9


def test_hgibbs_example_7_4():
    # crate `hgibbs_example_7_4`.
    r1 = np.array([3419.85564, 6019.82602, 2784.60022], dtype=np.float64)
    r2 = np.array([2935.91195, 6326.18324, 2660.59584], dtype=np.float64)
    r3 = np.array([2434.95202, 6597.38674, 2521.52311], dtype=np.float64)
    jd1 = 0.0
    jd2 = (60.0 + 16.48) / 86400.0
    jd3 = (120.0 + 33.04) / 86400.0
    v2, theta12, theta23, _copa = sidereon.hgibbs(r1, r2, r3, jd1, jd2, jd3)
    assert_float64_bits(
        v2, [-6.441557227511062, 3.777559606719521, -1.7205675602414345]
    )
    assert math.isclose(math.degrees(theta12), 4.499996147374992, rel_tol=1e-9)
    assert math.isclose(math.degrees(theta23), 4.499998402168982, rel_tol=1e-9)


def test_gauss_angles_example_7_2():
    # crate `gauss_example_7_2`.
    d2r = math.radians
    decl = np.array([d2r(18.667717), d2r(35.664741), d2r(36.996583)], dtype=np.float64)
    rtasc = np.array([d2r(0.939913), d2r(45.025748), d2r(67.886655)], dtype=np.float64)
    jd = np.array([2_456_159.5, 2_456_159.5, 2_456_159.5], dtype=np.float64)
    jdf = np.array(
        [0.4864351851851852, 0.49199074074074073, 0.4947685185185185], dtype=np.float64
    )
    rseci = np.array(
        [
            [4054.881, 2748.195, 4074.237],
            [3956.224, 2888.232, 4074.364],
            [3905.073, 2956.935, 4074.430],
        ],
        dtype=np.float64,
    )
    r2, v2 = sidereon.gauss_angles(decl, rtasc, jd, jdf, rseci)
    np.testing.assert_allclose(
        r2, [6313.378130210396, 5247.50563344895, 6467.707164431651], rtol=1e-9
    )
    np.testing.assert_allclose(
        v2, [-4.185488280436629, 4.7884929168898145, 1.721714659663034], rtol=1e-9
    )


def test_gibbs_rejects_collinear_vectors():
    r1 = np.array([0.0, 0.0, 6378.1363], dtype=np.float64)
    r2 = np.array([0.0, 1000.0, 0.0], dtype=np.float64)
    r3 = np.array([0.0, 2000.0, 0.0], dtype=np.float64)
    with pytest.raises(sidereon.SolveError):
        sidereon.gibbs(r1, r2, r3)


def test_iod_rejects_wrong_shape():
    bad = np.array([0.0, 0.0], dtype=np.float64)  # only 2 components
    ok = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        sidereon.gibbs(bad, ok, ok)
