"""Jacobian-derived covariance and error-ellipse binding tests."""

import numpy as np
import pytest
import sidereon


def test_normal_covariance_matches_explicit_inverse():
    rng = np.random.default_rng(3)
    jac = rng.standard_normal((10, 3))
    cov = sidereon.normal_covariance(jac, 2.0)
    expected = 2.0 * np.linalg.inv(jac.T @ jac)
    np.testing.assert_allclose(cov, expected, rtol=1e-9, atol=1e-12)
    # Covariance is symmetric.
    np.testing.assert_allclose(cov, cov.T, atol=1e-12)


def test_normal_covariance_default_scale_is_cofactor():
    jac = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cov = sidereon.normal_covariance(jac)
    expected = np.linalg.inv(jac.T @ jac)
    np.testing.assert_allclose(cov, expected, rtol=1e-9)


def test_normal_covariance_rank_deficient_raises():
    # A column duplicated -> rank deficient -> unbounded covariance.
    jac = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with pytest.raises(sidereon.SolveError):
        sidereon.normal_covariance(jac)


def test_hessian_trace_is_sum_of_squared_columns():
    jac = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    trace = sidereon.hessian_trace(jac)
    assert trace == pytest.approx(float(np.trace(jac.T @ jac)))


def test_covariance_from_jacobian_scales_by_reduced_chi_square():
    rng = np.random.default_rng(4)
    m, n = 12, 3
    jac = rng.standard_normal((m, n))
    residual = rng.standard_normal(m)
    cost = 0.5 * float(residual @ residual)
    cov = sidereon.covariance_from_jacobian(jac, cost)
    s_sq = 2.0 * cost / (m - n)
    expected = s_sq * np.linalg.inv(jac.T @ jac)
    np.testing.assert_allclose(cov, expected, rtol=1e-9, atol=1e-12)


def test_covariance_from_jacobian_requires_redundancy():
    jac = np.array([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        sidereon.covariance_from_jacobian(jac, 0.0)


def test_error_ellipse_axis_aligned():
    cov = np.array([[4.0, 0.0], [0.0, 1.0]])
    ell = sidereon.error_ellipse_2x2(cov, 0.95)
    scale = -2.0 * np.log(1.0 - 0.95)
    assert ell.chi_square_scale == pytest.approx(scale)
    assert ell.semi_major == pytest.approx(np.sqrt(4.0 * scale))
    assert ell.semi_minor == pytest.approx(np.sqrt(1.0 * scale))
    # Major axis lies along the first (x) axis, orientation 0.
    assert ell.orientation_rad == pytest.approx(0.0)
    assert ell.confidence == pytest.approx(0.95)


def test_error_ellipse_rejects_bad_shape_and_confidence():
    with pytest.raises(ValueError, match="shape"):
        sidereon.error_ellipse_2x2(np.eye(3), 0.95)
    with pytest.raises(ValueError):
        sidereon.error_ellipse_2x2(np.eye(2), 1.5)


def test_covariance_from_jacobian_matches_scipy_curve_fit_pcov():
    scipy_stats = pytest.importorskip("scipy.optimize")
    # curve_fit's pcov is (J^T J)^-1 * s_sq, formed from the SVD of J. Fit a
    # line and compare its reported pcov to ours from the same Jacobian.
    rng = np.random.default_rng(5)
    t = np.linspace(0.0, 10.0, 25)
    y = 2.0 + 0.5 * t + rng.standard_normal(25) * 0.2

    def model(tt, a, b):
        return a + b * tt

    popt, pcov = scipy_stats.curve_fit(model, t, y)
    residual = model(t, *popt) - y
    jac = np.column_stack([np.ones_like(t), t])
    cost = 0.5 * float(residual @ residual)
    ours = sidereon.covariance_from_jacobian(jac, cost)
    np.testing.assert_allclose(ours, pcov, rtol=1e-6, atol=1e-10)
