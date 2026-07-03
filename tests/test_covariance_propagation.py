"""Covariance transport bindings delegate to the core propagator."""

import numpy as np
import pytest
import sidereon


def _state():
    return sidereon.CartesianState(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 1.0])


def _covariance():
    return np.diag([1.0e-6, 2.0e-6, 3.0e-6, 1.0e-10, 2.0e-10, 3.0e-10])


def test_covariance_propagation_and_interpolation_are_pinned():
    state = _state()
    covariance = _covariance()
    labeled = sidereon.LabeledCovariance6(covariance, sidereon.CovarianceFrame.INERTIAL)

    eph = sidereon.propagate_covariance(
        state,
        labeled,
        np.array([0.0, 60.0, 120.0], dtype=np.float64),
        force_model=sidereon.ForceModel.TWO_BODY_J2,
    )

    assert len(eph) == 3
    assert eph.epoch_count == 3
    assert not eph.is_empty
    assert eph.nodes[-1].frame == sidereon.CovarianceFrame.INERTIAL
    np.testing.assert_allclose(
        eph.nodes[-1].state.position_km,
        np.array([6941.43403351, 897.48866190, 119.66425402]),
        rtol=0.0,
        atol=5e-9,
    )
    np.testing.assert_allclose(
        np.diag(eph.nodes[-1].covariance),
        np.array(
            [
                2.48985214e-06,
                4.83113781e-06,
                7.24575067e-06,
                1.03430613e-10,
                1.96788387e-10,
                2.95048526e-10,
            ]
        ),
        rtol=0.0,
        atol=5e-14,
    )
    np.testing.assert_allclose(eph.covariance_at(60.0), eph.nodes[1].covariance)


def test_covariance_conversions_and_rtn_helpers_are_pinned():
    state = _state()
    covariance = _covariance()

    np.testing.assert_allclose(
        np.diag(sidereon.covariance6_km_to_m(covariance)),
        np.array([1.0, 2.0, 3.0, 1.0e-4, 2.0e-4, 3.0e-4]),
        rtol=0.0,
        atol=1e-18,
    )
    np.testing.assert_allclose(
        sidereon.covariance6_m_to_km(sidereon.covariance6_km_to_m(covariance)),
        covariance,
    )

    rtn = sidereon.eci_to_rtn_covariance6(covariance, state)
    np.testing.assert_allclose(
        np.diag(rtn),
        np.array(
            [
                1.00000000e-06,
                2.01746725e-06,
                2.98253275e-06,
                1.00000000e-10,
                2.01746725e-10,
                2.98253275e-10,
            ]
        ),
        rtol=0.0,
        atol=5e-14,
    )
    np.testing.assert_allclose(
        sidereon.rtn_to_eci_covariance6(rtn, state), covariance, atol=1e-18
    )

    interp = sidereon.interpolate_covariance6(covariance, covariance * 2.0, 0.5)
    assert np.all(np.diag(interp) > np.diag(covariance))


def test_process_noise_constructor_validation():
    assert sidereon.ProcessNoise.none().kind == "none"
    noise = sidereon.ProcessNoise.rtn_acceleration_psd(1e-18, 2e-18, 3e-18)
    assert noise.kind == "rtn_acceleration_psd"
    assert noise.q_normal_km2_s3 == pytest.approx(3e-18)
    with pytest.raises(ValueError):
        sidereon.ProcessNoise(q_radial_km2_s3=1e-18)
