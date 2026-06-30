"""rv2coe / coe2rv element conversions delegate to the core Vallado algorithms.

The binding adds no element logic; it marshals the state vectors and gravitational
parameter into ``sidereon_core::astro::elements`` and packages the result. These
tests use the Vallado Algorithm 9 worked example and check the round trip.
"""

import numpy as np
import pytest
import sidereon

# Standard Earth gravitational parameter (km^3 / s^2), Vallado.
MU_EARTH = 398600.4418


def test_rv2coe_matches_vallado_example():
    # Vallado, Algorithm 9 worked example.
    r = np.array([6524.834, 6862.875, 6448.296])
    v = np.array([4.901327, 5.533756, -1.976341])
    coe = sidereon.rv2coe(r, v, MU_EARTH)
    assert coe.ecc == pytest.approx(0.832853, abs=1e-5)
    assert np.degrees(coe.incl) == pytest.approx(87.870, abs=1e-2)
    assert np.degrees(coe.raan) == pytest.approx(227.898, abs=1e-2)
    assert coe.orbit_type == sidereon.OrbitType.ELLIPTICAL_INCLINED
    assert coe.orbit_type.label == "elliptical_inclined"


def test_rv2coe_coe2rv_round_trip():
    r = np.array([-6045.0, -3490.0, 2500.0])
    v = np.array([-3.457, 6.618, 2.533])
    coe = sidereon.rv2coe(r, v, MU_EARTH)
    r_back, v_back = sidereon.coe2rv(coe, MU_EARTH)
    np.testing.assert_allclose(r_back, r, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(v_back, v, rtol=1e-9, atol=1e-9)


def test_classical_elements_constructor_and_coe2rv():
    # Build an elliptical-inclined element set directly and propagate to a state.
    coe = sidereon.ClassicalElements(
        p=11067.79,
        ecc=0.83285,
        incl=np.radians(87.87),
        raan=np.radians(227.89),
        argp=np.radians(53.38),
        nu=np.radians(92.335),
    )
    assert coe.a > 0.0
    r, v = sidereon.coe2rv(coe, MU_EARTH)
    assert r.shape == (3,)
    assert v.shape == (3,)
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(v))


def test_rv2coe_rejects_non_finite():
    with pytest.raises(ValueError):
        sidereon.rv2coe(
            np.array([np.nan, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), MU_EARTH
        )


def test_coe2rv_rejects_non_positive_mu():
    coe = sidereon.ClassicalElements(11067.79, 0.1, 0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        sidereon.coe2rv(coe, -1.0)
