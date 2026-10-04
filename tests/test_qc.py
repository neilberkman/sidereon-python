"""Quality control: RAIM and fault detection and exclusion (FDE) through the binding.

Both are pure wrappers over `sidereon_core::quality`. RAIM is checked on
deterministic synthetic residuals (clean set passes, a single blunder is flagged
with the correct worst satellite). FDE is checked end-to-end on the real-GPS
epoch of the SPP trace fixture, with the pseudorange set in `core_goldens.json`
that is consistent with its own core solution to a few nanometres, so
injecting a gross blunder on one satellite must drive the FDE loop to exclude
exactly that satellite and recover the reference position, while the clean set
passes with no exclusions.
"""

import json
import math
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, core_goldens, hex_to_f64

TRACE = os.path.join(CORE_FIXTURES, "spp_trace_L0_minimal.json")
# The satellites this trace solves on, with pseudoranges the core's default
# model fits with zero residuals (to the last bits of a 2e7 m range): the
# trace's own pseudoranges, each moved by its post-fit residual until none is
# left, written by `scripts/core_goldens`. The trace was synthesized without
# the relativistic clock term and with the geometric light time, which the
# core's default model applies as RTKLIB does, so its raw pseudoranges no
# longer fit with zero residuals.
_CONSISTENT = core_goldens()["spp_consistent"]
CONSISTENT_SATS = _CONSISTENT["satellites"]


def _trace():
    with open(TRACE) as fh:
        return json.load(fh)["fixture"]


def _config(fx, observations):
    inp = fx["inputs"]
    return sidereon.SppConfig(
        observations=[sidereon.SppObservation(s, p) for s, p in observations],
        t_rx_j2000_s=hex_to_f64(inp["t_rx_j2000_s"]),
        t_rx_second_of_day_s=hex_to_f64(inp["t_rx_sod_s"]),
        day_of_year=hex_to_f64(inp["doy"]),
        initial_guess=[hex_to_f64(x) for x in fx["frozen"]["initial_guess_x0"]],
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
        klobuchar=sidereon.SppKlobucharCoeffs(
            alpha=[hex_to_f64(x) for x in inp["klobuchar_alpha"]],
            beta=[hex_to_f64(x) for x in inp["klobuchar_beta"]],
        ),
        with_geodetic=True,
    )


def _consistent_observations(fx):
    return [
        (sat, hex_to_f64(p))
        for sat, p in zip(CONSISTENT_SATS, _CONSISTENT["pseudoranges_m"])
    ]


def _truth(fx):
    return np.array([hex_to_f64(x) for x in _CONSISTENT["position_m"]])


def test_raim_passes_clean_and_flags_a_blunder():
    # Synthetic residuals carry no estimator variances, so the test states
    # unit weighting (sigma = 1 m) rather than the solution-variance default.
    used = ["G01", "G02", "G03", "G04", "G05", "G06"]
    clean = [0.4, -0.6, 0.3, 0.1, -0.2, 0.5]
    ok = sidereon.qc_raim(used, clean, 0.05, weights="unit")
    assert ok.testable and not ok.fault_detected

    faulted = list(clean)
    faulted[2] = 80.0
    bad = sidereon.qc_raim(used, faulted, 0.05, weights="unit")
    assert bad.fault_detected
    assert bad.worst_sat == "G03"
    assert bad.dof == len(used) - (3 + 1)


def test_raim_direct_round_trip_from_lists_matches_core_values():
    used = ["G01", "G02", "G03", "G04", "G05", "G06"]
    residuals = [0.4, -0.6, 0.3, 0.1, -0.2, 0.5]
    result = sidereon.raim(used, residuals, weights="unit")

    assert isinstance(result, sidereon.RaimResult)
    assert not result.fault_detected
    assert result.testable
    assert result.test_statistic == pytest.approx(0.91)
    assert result.threshold == pytest.approx(13.815510557964274)
    assert result.dof == 2
    assert result.rms_m == pytest.approx(math.sqrt(0.91 / len(residuals)))
    assert result.reduced_chi_square == pytest.approx(0.455)
    assert result.normalized_residuals["G02"] == pytest.approx(-0.6)
    assert result.worst_sat == "G02"


def test_raim_typed_input_and_solution_path_use_real_spp_residuals():
    fx = _trace()
    sp3 = _load_sp3(fx)
    solution = sidereon.solve_spp(sp3, _config(fx, _consistent_observations(fx)))

    typed = sidereon.RaimInput(
        solution.used_sats, solution.residuals_m, solution.pseudorange_variances_m2
    )
    direct = sidereon.raim(typed, p_fa=0.01)
    from_solution = sidereon.raim_for_solution(solution, p_fa=0.01)

    # Both default to the solve's own variances and count one GPS clock, so
    # they are the same test of the same residuals.
    assert direct.test_statistic == from_solution.test_statistic
    assert direct.threshold == from_solution.threshold
    assert direct.testable
    assert not direct.fault_detected
    assert from_solution.testable
    assert not from_solution.fault_detected
    assert from_solution.dof == len(solution.used_sats) - 4
    assert from_solution.rms_m < 1.0e-6


def test_raim_not_testable_with_too_few_satellites():
    res = sidereon.qc_raim(
        ["G01", "G02", "G03", "G04"], [0.1, 0.2, -0.1, 0.05], 0.05, weights="unit"
    )
    assert not res.testable
    assert not res.fault_detected


def _load_sp3(fx):
    with open(os.path.join(CORE_FIXTURES, "sp3", fx["inputs"]["sp3_file"]), "rb") as fh:
        return sidereon.load_sp3(fh.read())


def test_fde_clean_set_makes_no_exclusions():
    fx = _trace()
    result = sidereon.qc_fde(
        _load_sp3(fx),
        _config(fx, _consistent_observations(fx)),
        p_fa=0.01,
        max_exclusions=3,
    )
    assert result.excluded == []
    assert result.iterations == 0
    assert float(np.linalg.norm(result.position - _truth(fx))) < 1e-6


def test_fde_excludes_the_blunder_and_recovers_truth():
    fx = _trace()
    sp3 = _load_sp3(fx)

    observations = [list(o) for o in _consistent_observations(fx)]
    blunder_sat = "G08"
    for row in observations:
        if row[0] == blunder_sat:
            row[1] += 5000.0

    result = sidereon.qc_fde(
        sp3, _config(fx, observations), p_fa=0.01, max_exclusions=3
    )

    assert blunder_sat in result.excluded
    assert blunder_sat not in result.used_sats
    assert result.iterations >= 1
    assert float(np.linalg.norm(result.position - _truth(fx))) < 1e-6


# --- solution-variance RAIM, the accepted detection test, typed refusals -----
#
# RAIM defaults to the variances the solve weighted each residual by
# (`SppSolution.pseudorange_variances_m2`), RTKLIB demo5 `valsol`'s
# `sum (v / sigma)^2`, and FDE chooses each exclusion by re-solving without each
# satellite, RTKLIB demo5 `raim_fde`'s rule. The checks here are identities of
# those definitions evaluated on the core's own outputs.


def _blunder(fx, blunder_sat, bias_m):
    return [
        (sat, p + (bias_m if sat == blunder_sat else 0.0))
        for sat, p in _consistent_observations(fx)
    ]


def test_spp_solution_reports_the_variances_it_weighted_by():
    fx = _trace()
    solution = sidereon.solve_spp(
        _load_sp3(fx), _config(fx, _consistent_observations(fx))
    )
    variances = solution.pseudorange_variances_m2
    weights = solution.weights

    assert len(variances) == len(weights) == len(solution.used_sats)
    assert all(math.isfinite(v) and v > 0.0 for v in variances)
    # On the least-squares path each weight is the inverse variance, one IEEE
    # division, so the two agree bit for bit.
    assert weights == [1.0 / v for v in variances]


def test_solution_raim_is_the_chi_square_over_the_solution_variances():
    fx = _trace()
    solution = sidereon.solve_spp(
        _load_sp3(fx), _config(fx, _blunder(fx, "G08", 300.0))
    )
    result = sidereon.raim_for_solution(solution)

    statistic = 0.0
    normalized = {}
    for sat, residual, variance in zip(
        solution.used_sats, solution.residuals_m, solution.pseudorange_variances_m2
    ):
        value = residual / math.sqrt(variance)
        normalized[sat] = value
        statistic += value * value

    assert result.test_statistic == statistic
    assert result.normalized_residuals == normalized
    assert result.dof == len(solution.used_sats) - 4
    assert result.threshold == sidereon.chi2_inv(
        1.0 - sidereon.RAIM_DEFAULT_P_FA, result.dof
    )
    assert result.fault_detected


def test_raim_weighting_modes_and_the_missing_variance_refusal():
    used = ["G01", "G02", "G03", "G04", "G05", "G06"]
    residuals = [0.4, -0.6, 0.3, 0.1, -0.2, 0.5]

    assert sidereon.RaimWeights().is_solution
    assert sidereon.RaimWeights.solution().is_solution
    assert not sidereon.RaimWeights.solution().is_unit

    # The default reads the estimator's variances and refuses residuals
    # without them rather than assuming a sigma.
    with pytest.raises(sidereon.QualityError, match="variances") as missing:
        sidereon.qc_raim(used, residuals)
    assert isinstance(missing.value, ValueError)
    assert missing.value.kind == "missing_variances"
    typed = sidereon.RaimInput(used, residuals)
    assert typed.variances_m2 is None
    with pytest.raises(sidereon.QualityError, match="variances") as typed_missing:
        sidereon.raim(typed)
    assert typed_missing.value.kind == "missing_variances"
    with pytest.raises(sidereon.QualityError, match="variance") as misaligned:
        sidereon.raim(used, residuals, variances_m2=[1.0] * (len(used) - 1))
    assert misaligned.value.kind == "invalid_variance"
    with pytest.raises(sidereon.QualityError, match="variance") as invalid:
        sidereon.raim(used, residuals, variances_m2=[1.0] * (len(used) - 1) + [0.0])
    assert invalid.value.kind == "invalid_variance"
    manual = sidereon.QualityError("hand-built")
    assert isinstance(manual, ValueError)
    assert manual.kind is None
    with pytest.raises(sidereon.QualityError) as bad_probability:
        sidereon.qc_raim(used, residuals, p_fa=1.0, variances_m2=[1.0] * len(used))
    assert bad_probability.value.kind == "invalid_probability"
    with pytest.raises(ValueError, match="RAIM weighting"):
        sidereon.raim(used, residuals, weights="elevation")

    by_label = sidereon.raim(used, residuals, weights="unit")
    by_class = sidereon.raim(used, residuals, weights=sidereon.RaimWeights.unit())
    assert by_class.test_statistic == by_label.test_statistic
    # Unit variances divide each residual by sqrt(1), exactly unit weighting.
    unit_variances = sidereon.raim(used, residuals, variances_m2=[1.0] * len(used))
    assert unit_variances.test_statistic == by_label.test_statistic
    assert unit_variances.normalized_residuals == by_label.normalized_residuals


def test_fde_result_carries_the_accepted_detection_test():
    fx = _trace()
    sp3 = _load_sp3(fx)

    # Every default: detection at p_fa = 1e-3 over the solve's own variances,
    # one exclusion chosen by re-solving without each satellite.
    result = sidereon.qc_fde(sp3, _config(fx, _blunder(fx, "G08", 5000.0)))

    assert result.excluded == ["G08"]
    assert result.iterations == 1
    assert result.solution.used_sats == result.used_sats
    accepted = result.raim
    assert accepted.testable
    assert not accepted.fault_detected
    assert accepted.dof == len(result.used_sats) - 4
    again = sidereon.raim_for_solution(result.solution)
    assert accepted.test_statistic == again.test_statistic
    assert accepted.threshold == again.threshold
    assert float(np.linalg.norm(result.position - _truth(fx))) < 1e-6


def test_fde_with_no_exclusion_budget_raises_with_the_flagged_solution():
    fx = _trace()
    sp3 = _load_sp3(fx)

    with pytest.raises(sidereon.FdeFaultUnresolvedError) as caught:
        sidereon.qc_fde(sp3, _config(fx, _blunder(fx, "G08", 5000.0)), max_exclusions=0)

    err = caught.value
    assert isinstance(err, sidereon.SolveError)
    assert err.reason == "exclusion_budget_exhausted"
    assert err.excluded == []
    assert "G08" in err.solution.used_sats
    assert err.raim.fault_detected
    again = sidereon.raim_for_solution(err.solution)
    assert err.raim.test_statistic == again.test_statistic


def test_fde_budget_and_cap_arguments_are_validated():
    fx = _trace()
    sp3 = _load_sp3(fx)
    config = _config(fx, _consistent_observations(fx))

    # The budget has one name, the core's.
    with pytest.raises(TypeError, match="max_iterations"):
        sidereon.qc_fde(sp3, config, max_iterations=1)
    with pytest.raises(ValueError):
        sidereon.qc_fde(sp3, config, max_exclusion_rms_m=0.0)

    hand_built = sidereon.FdeFaultUnresolvedError("hand-built")
    assert hand_built.reason is None
    assert hand_built.solution is None
    assert hand_built.excluded is None
    assert hand_built.raim is None


# --- standalone range RAIM/FDE design over a linearized measurement set -----
#
# `qc_raim_fde_design` is a pure wrapper over `sidereon_core::quality::
# raim_fde_design`: it runs the protected weighted least squares, the global
# chi-square test, and the leave-one-out FDE loop on a generic linearized set,
# independent of any full solve. A single-state set with identical geometry
# rows is mutually consistent, so a clean set passes and a gross blunder on one
# row must be excluded.


def _clean_rows():
    # State dimension 1; every row observes the same partial, so a consistent
    # set has residual = g * dx for a common dx (here dx = 0.5).
    return [sidereon.RangeFdeRow(f"S{i}", 0.5, [1.0], 1.0) for i in range(1, 6)]


def test_range_fde_clean_set_no_exclusion():
    result = sidereon.qc_raim_fde_design(_clean_rows(), p_fa=1e-3)
    assert result.excluded == []
    assert result.iterations == 0
    assert result.global_test.fault_detected is False
    assert result.global_test.testable is True
    assert result.global_test.dof == 4
    assert result.state_correction == pytest.approx([0.5])
    assert len(result.diagnostics) == 5
    assert all(not d.excluded for d in result.diagnostics)


def test_range_fde_excludes_single_blunder():
    rows = _clean_rows()
    # Inject a gross blunder on the last row.
    rows[-1] = sidereon.RangeFdeRow("S5", 10.0, [1.0], 1.0)
    result = sidereon.qc_raim_fde_design(rows, p_fa=1e-3)
    assert "S5" in result.excluded
    assert result.iterations >= 1
    # After excluding the blunder the protected set is consistent again.
    assert result.global_test.fault_detected is False
    # The protected state correction recovers the consistent value.
    assert result.state_correction == pytest.approx([0.5])
    blunder = next(d for d in result.diagnostics if d.id == "S5")
    assert blunder.excluded is True
    assert abs(blunder.normalized_residual) > 1.0


def test_range_fde_max_exclusions_caps_loop():
    rows = _clean_rows()
    rows[-1] = sidereon.RangeFdeRow("S5", 10.0, [1.0], 1.0)
    # Budget of zero forbids any removal; the fault is left unresolved.
    result = sidereon.qc_raim_fde_design(rows, p_fa=1e-3, max_exclusions=0)
    assert result.excluded == []
    assert result.iterations == 0
    assert result.global_test.fault_detected is True


def test_range_fde_rejects_ragged_design_rows():
    rows = [
        sidereon.RangeFdeRow("S1", 0.5, [1.0, 0.0], 1.0),
        sidereon.RangeFdeRow("S2", 0.5, [1.0], 1.0),
    ]
    with pytest.raises(ValueError):
        sidereon.qc_raim_fde_design(rows)
