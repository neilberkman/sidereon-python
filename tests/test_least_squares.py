"""Generic data-driven trust-region least-squares binding tests.

The native-backend numeric checks use exactly-solvable problems whose optimum is
analytic, so they run everywhere without SciPy. The bit-exact-vs-SciPy checks are
pinned to Linux x86_64 (where SciPy ships its bundled OpenBLAS and the crate's
host-LAPACK backend can reproduce the SciPy trajectory bit-for-bit) and skip
elsewhere rather than chasing macOS arm64 ULP differences.
"""

import glob
import os
import platform

import numpy as np
import pytest
import sidereon


def _scipy_bitexact_env():
    """Return the scipy OpenBLAS path on Linux x86_64 with SciPy present, else
    None. Setting ``TRUST_REGION_LEAST_SQUARES_LAPACK_PATH`` to it lets the
    crate's ``backend='lapack'`` reproduce the SciPy numerical trajectory."""
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return None
    try:
        import scipy
    except ImportError:
        return None
    site = os.path.dirname(os.path.dirname(scipy.__file__))
    libs = glob.glob(os.path.join(site, "scipy.libs", "*openblas*"))
    return libs[0] if libs else None


BITEXACT = _scipy_bitexact_env()
bitexact_only = pytest.mark.skipif(
    BITEXACT is None,
    reason="bit-exact-vs-scipy is pinned to Linux x86_64 with SciPy installed",
)


def test_linear_exact_fit():
    # y = 1 + 2x sampled exactly: the least-squares optimum is [1, 2] with zero
    # residual.
    a = np.array([[1.0, x] for x in (0.0, 1.0, 2.0, 3.0, 4.0)])
    b = np.array([1.0 + 2.0 * x for x in (0.0, 1.0, 2.0, 3.0, 4.0)])
    result = sidereon.least_squares("linear", [0.0, 0.0], a=a, b=b)
    assert result.success
    assert result.status > 0
    np.testing.assert_allclose(result.x, [1.0, 2.0], atol=1e-9)
    assert result.cost < 1e-18
    assert result.jac.shape == (5, 2)
    assert result.fun.shape == (5,)
    assert result.grad.shape == (2,)
    assert result.nfev >= 1
    assert result.njev >= 1


def test_polynomial_exact_fit():
    t = np.linspace(-2.0, 2.0, 9)
    y = 3.0 - 1.5 * t + 0.5 * t**2
    result = sidereon.least_squares("polynomial", [0.0, 0.0, 0.0], t=t, y=y, degree=2)
    assert result.success
    np.testing.assert_allclose(result.x, [3.0, -1.5, 0.5], atol=1e-7)


def test_exponential_recovers_parameters():
    t = np.linspace(0.0, 1.5, 12)
    y = 2.0 * np.exp(0.7 * t) + 0.4
    result = sidereon.least_squares("exponential", [1.0, 1.0, 0.0], t=t, y=y)
    assert result.success
    np.testing.assert_allclose(result.x, [2.0, 0.7, 0.4], atol=1e-6)


def test_drop_one_report_shape_and_base():
    a = np.array([[1.0, x] for x in (0.0, 1.0, 2.0, 3.0, 4.0)])
    b = np.array([1.0 + 2.0 * x for x in (0.0, 1.0, 2.0, 3.0, 4.0)])
    report = sidereon.least_squares_drop_one("linear", [0.0, 0.0], a=a, b=b)
    np.testing.assert_allclose(report.base.x, [1.0, 2.0], atol=1e-9)
    assert len(report.drops) == 5
    assert report.cost_delta.shape == (5,)
    # Each drop is an independent solve over m - 1 rows.
    for drop in report.drops:
        assert drop.fun.shape == (4,)


def test_drop_one_flags_the_outlier():
    # A clean line with one corrupted row: removing that row drops the cost the
    # most, which is the RAIM/FDE signal.
    xs = np.arange(6.0)
    a = np.array([[1.0, x] for x in xs])
    b = 1.0 + 2.0 * xs
    b[3] += 5.0  # inject an outlier on row 3
    report = sidereon.least_squares_drop_one("linear", [0.0, 0.0], a=a, b=b)
    worst = int(np.argmin(report.cost_delta))
    assert worst == 3


def test_robust_loss_downweights_outlier():
    xs = np.arange(8.0)
    a = np.array([[1.0, x] for x in xs])
    b = 1.0 + 2.0 * xs
    b[5] += 10.0
    linear = sidereon.least_squares("linear", [0.0, 0.0], a=a, b=b)
    huber = sidereon.least_squares(
        "linear", [0.0, 0.0], a=a, b=b, loss="huber", f_scale=1.0
    )
    # A robust loss pulls the slope back toward the clean trend (2.0); plain
    # least squares is dragged further by the outlier.
    assert abs(huber.x[1] - 2.0) < abs(linear.x[1] - 2.0)


@pytest.mark.parametrize("loss", ["linear", "soft_l1", "huber", "cauchy", "arctan"])
def test_all_losses_run(loss):
    xs = np.arange(8.0)
    a = np.array([[1.0, x] for x in xs])
    b = 1.0 + 2.0 * xs
    result = sidereon.least_squares(
        "linear", [0.0, 0.0], a=a, b=b, loss=loss, f_scale=1.5
    )
    assert result.success


def test_x_scale_jac_and_sequence_run():
    a = np.array([[1.0, x] for x in (0.0, 1.0, 2.0, 3.0, 4.0)])
    b = np.array([1.0 + 2.0 * x for x in (0.0, 1.0, 2.0, 3.0, 4.0)])
    jac = sidereon.least_squares("linear", [0.0, 0.0], a=a, b=b, x_scale="jac")
    seq = sidereon.least_squares("linear", [0.0, 0.0], a=a, b=b, x_scale=[1.0, 2.0])
    np.testing.assert_allclose(jac.x, [1.0, 2.0], atol=1e-8)
    np.testing.assert_allclose(seq.x, [1.0, 2.0], atol=1e-8)


def test_unknown_kind_and_loss_raise():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="unknown kind"):
        sidereon.least_squares("quadratic", [0.0, 0.0], a=a, b=b)
    with pytest.raises(ValueError, match="unknown loss"):
        sidereon.least_squares("linear", [0.0, 0.0], a=a, b=b, loss="tukey")
    with pytest.raises(ValueError, match="unknown backend"):
        sidereon.least_squares("linear", [0.0, 0.0], a=a, b=b, backend="cuda")


def test_missing_data_arrays_raise():
    with pytest.raises(ValueError, match="requires the design matrix"):
        sidereon.least_squares("linear", [0.0, 0.0])
    with pytest.raises(ValueError, match="requires the integer `degree`"):
        sidereon.least_squares(
            "polynomial", [0.0], t=np.array([0.0, 1.0]), y=np.array([0.0, 1.0])
        )


def test_underdetermined_raises():
    # m < n: the dense exact trust-region solve requires at least as many
    # residuals as parameters.
    a = np.array([[1.0, 0.0, 0.0]])
    b = np.array([1.0])
    with pytest.raises(ValueError):
        sidereon.least_squares("linear", [0.0, 0.0, 0.0], a=a, b=b)


# --- bit-exact-vs-scipy (Linux x86_64 only) --------------------------------


def _run_bitexact(kind, x0, build_residual, **kwargs):
    import scipy.optimize

    os.environ["TRUST_REGION_LEAST_SQUARES_LAPACK_PATH"] = BITEXACT
    ours = sidereon.least_squares(kind, x0, backend="lapack", **kwargs)
    ref = scipy.optimize.least_squares(
        build_residual,
        np.asarray(x0, dtype=float),
        method="trf",
        jac="2-point",
        loss=kwargs.get("loss", "linear"),
        f_scale=kwargs.get("f_scale", 1.0),
    )
    return ours, ref


@bitexact_only
def test_linear_bitexact_vs_scipy():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((12, 3))
    b = rng.standard_normal(12)
    ours, ref = _run_bitexact("linear", [0.0, 0.0, 0.0], lambda x: a @ x - b, a=a, b=b)
    assert np.array_equal(ours.x, ref.x)
    assert ours.cost == ref.cost


@bitexact_only
def test_robust_loss_bitexact_vs_scipy():
    rng = np.random.default_rng(1)
    t = np.linspace(0.0, 3.0, 20)
    y = 1.0 + 2.0 * t + rng.standard_normal(20) * 0.1
    y[7] += 8.0

    def resid(x):
        return (x[0] + x[1] * t) - y

    a = np.column_stack([np.ones_like(t), t])
    ours, ref = _run_bitexact(
        "linear",
        [0.0, 0.0],
        resid,
        a=a,
        b=y,
        loss="soft_l1",
        f_scale=1.0,
    )
    assert np.array_equal(ours.x, ref.x)
