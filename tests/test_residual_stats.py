"""Residual-distribution diagnostics binding tests.

The structural checks run everywhere. The moment definitions match scipy.stats,
so where SciPy is installed the values are cross-checked within a tight tolerance
(the reductions differ only in summation order, so this is tolerance-close, not
bit-exact)."""

import numpy as np
import pytest
import sidereon

RESID = np.array(
    [0.12, -0.21, 0.05, 0.31, -0.11, 0.22, -0.06, 0.16, -0.26, 0.0, 0.13, -0.09]
)


def test_skewness_kurtosis_moments_consistent():
    mo = sidereon.moments(RESID)
    assert mo.mean == pytest.approx(float(np.mean(RESID)))
    assert mo.variance == pytest.approx(float(np.var(RESID)))
    assert mo.skewness == pytest.approx(sidereon.skewness(RESID))
    assert mo.kurtosis_excess == pytest.approx(sidereon.kurtosis(RESID))


def test_kurtosis_fisher_vs_pearson():
    fisher = sidereon.kurtosis(RESID, fisher=True)
    pearson = sidereon.kurtosis(RESID, fisher=False)
    assert pearson == pytest.approx(fisher + 3.0)


def test_jarque_bera_and_shapiro_fields():
    jb = sidereon.jarque_bera(RESID)
    assert jb.statistic >= 0.0
    assert 0.0 <= jb.p_value <= 1.0
    sw = sidereon.shapiro_wilk(RESID)
    assert 0.0 < sw.statistic <= 1.0
    assert 0.0 <= sw.p_value <= 1.0


def test_zero_variance_raises():
    with pytest.raises(ValueError):
        sidereon.skewness(np.zeros(8))


def test_insufficient_data_raises():
    with pytest.raises(ValueError):
        sidereon.shapiro_wilk(np.array([1.0, 2.0]))


def test_matches_scipy_stats():
    stats = pytest.importorskip("scipy.stats")
    assert sidereon.skewness(RESID) == pytest.approx(stats.skew(RESID), rel=1e-10)
    assert sidereon.kurtosis(RESID) == pytest.approx(stats.kurtosis(RESID), rel=1e-10)
    assert sidereon.skewness(RESID, bias=False) == pytest.approx(
        stats.skew(RESID, bias=False), rel=1e-10
    )
    jb = sidereon.jarque_bera(RESID)
    jb_ref = stats.jarque_bera(RESID)
    assert jb.statistic == pytest.approx(jb_ref.statistic, rel=1e-9)
    sw = sidereon.shapiro_wilk(RESID)
    sw_ref = stats.shapiro(RESID)
    assert sw.statistic == pytest.approx(sw_ref.statistic, rel=1e-8)
