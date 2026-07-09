"""Quality control: RAIM and fault detection and exclusion (FDE) through the binding.

Both are pure wrappers over `sidereon_core::quality`. RAIM is checked on
deterministic synthetic residuals (clean set passes, a single blunder is flagged
with the correct worst satellite). FDE is checked end-to-end against the
committed real-GPS SPP trace fixture: the consistent satellite set solves with
zero residuals, so injecting a gross blunder on one satellite must drive the FDE
loop to exclude exactly that satellite and recover the reference position, while
the clean set passes with no exclusions.
"""

import json
import math
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, hex_to_f64

TRACE = os.path.join(CORE_FIXTURES, "spp_trace_L0_minimal.json")
# The mutually consistent satellite subset this trace solves on (zero residuals).
CONSISTENT_SATS = ["G08", "G10", "G16", "G18", "G20", "G21", "G26", "G27"]


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
    obs = {o["sat_id"]: hex_to_f64(o["p_meas_m"]) for o in fx["inputs"]["observations"]}
    return [(s, obs[s]) for s in CONSISTENT_SATS]


def _truth(fx):
    return np.array([hex_to_f64(x) for x in fx["final_solution"]["x"][:3]])


def test_raim_passes_clean_and_flags_a_blunder():
    used = ["G01", "G02", "G03", "G04", "G05", "G06"]
    clean = [0.4, -0.6, 0.3, 0.1, -0.2, 0.5]
    ok = sidereon.qc_raim(used, clean, 0.05)
    assert ok.testable and not ok.fault_detected

    faulted = list(clean)
    faulted[2] = 80.0
    bad = sidereon.qc_raim(used, faulted, 0.05)
    assert bad.fault_detected
    assert bad.worst_sat == "G03"
    assert bad.dof == len(used) - (3 + 1)


def test_raim_direct_round_trip_from_lists_matches_core_values():
    used = ["G01", "G02", "G03", "G04", "G05", "G06"]
    residuals = [0.4, -0.6, 0.3, 0.1, -0.2, 0.5]
    result = sidereon.raim(used, residuals)

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


def test_raim_not_testable_with_too_few_satellites():
    res = sidereon.qc_raim(["G01", "G02", "G03", "G04"], [0.1, 0.2, -0.1, 0.05], 0.05)
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
        max_iterations=3,
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
        sp3, _config(fx, observations), p_fa=0.01, max_iterations=3
    )

    assert blunder_sat in result.excluded
    assert blunder_sat not in result.used_sats
    assert result.iterations >= 1
    assert float(np.linalg.norm(result.position - _truth(fx))) < 1e-6


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
