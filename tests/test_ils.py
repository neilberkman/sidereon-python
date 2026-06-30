"""Standalone integer-ambiguity resolution (LAMBDA / bounded ILS) through the
binding reproduces the crate-side reference behavior.

The numbers are the core's; these only confirm the binding marshals the
array-in / IlsResult-out shapes faithfully and maps the error taxonomy.
"""

import numpy as np
import pytest
import sidereon


def test_bounded_ils_fixes_a_well_separated_lattice_point():
    # Float ambiguities very close to integers, tight diagonal covariance: the
    # nearest lattice point dominates and the ratio test passes (crate
    # `fixes_a_well_separated_lattice_point`).
    float_cycles = np.array([3.02, -1.98, 5.01], dtype=np.float64)
    cov = np.diag([0.01, 0.01, 0.01]).astype(np.float64)
    r = sidereon.bounded_ils_search(
        float_cycles, cov, radius=1, candidate_limit=200_000
    )
    assert r.fixed == [3, -2, 5]
    assert r.fixed_status
    assert r.ratio > 3.0
    assert r.candidates_evaluated == 27  # 3^3
    assert r.covariance.shape == (3, 3)
    assert r.covariance_inverse.shape == (3, 3)
    assert "IlsResult(" in repr(r)


def test_bounded_ils_refuses_an_ambiguous_lattice():
    # Half-integer floats: nearest points are equidistant -> low ratio (crate
    # `refuses_an_ambiguous_lattice`).
    float_cycles = np.array([0.5, 0.5], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    r = sidereon.bounded_ils_search(
        float_cycles, cov, radius=1, candidate_limit=200_000
    )
    assert not r.fixed_status
    assert r.ratio < 3.0


def test_lambda_matches_bounded_on_weakly_correlated_geometry():
    # On a weakly-correlated (diagonal) covariance the two kernels select the
    # identical integer vector and ratio.
    float_cycles = np.array([3.02, -1.98, 5.01], dtype=np.float64)
    cov = np.diag([0.01, 0.01, 0.01]).astype(np.float64)
    lam = sidereon.lambda_ils_search(float_cycles, cov)
    bnd = sidereon.bounded_ils_search(
        float_cycles, cov, radius=1, candidate_limit=200_000
    )
    assert lam.fixed == bnd.fixed
    assert lam.fixed_status == bnd.fixed_status


def test_lambda_default_ratio_threshold_is_three():
    float_cycles = np.array([0.5, 0.5], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    # The RTKLIB default threshold is 3.0; an ambiguous lattice fails it.
    assert not sidereon.lambda_ils_search(float_cycles, cov).fixed_status


def test_bounded_ils_too_many_candidates_raises_solve_error():
    # 3^3 = 27 lattice points, limit 10 -> engine error (crate
    # `errors_when_the_lattice_exceeds_the_candidate_limit`).
    float_cycles = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    cov = np.eye(3, dtype=np.float64)
    with pytest.raises(sidereon.SolveError):
        sidereon.bounded_ils_search(float_cycles, cov, radius=1, candidate_limit=10)


def test_ils_singular_covariance_raises_solve_error():
    float_cycles = np.array([0.0, 0.0], dtype=np.float64)
    singular = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)
    with pytest.raises(sidereon.SolveError):
        sidereon.lambda_ils_search(float_cycles, singular)


def test_ils_non_square_covariance_raises_value_error():
    float_cycles = np.array([0.0, 0.0], dtype=np.float64)
    rect = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError):
        sidereon.lambda_ils_search(float_cycles, rect)


def test_ils_non_finite_input_raises_value_error():
    float_cycles = np.array([np.nan, 0.0], dtype=np.float64)
    cov = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError):
        sidereon.bounded_ils_search(float_cycles, cov)
