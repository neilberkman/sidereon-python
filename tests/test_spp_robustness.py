"""SPP robustness and integrity surface through the binding.

These exercise the parity additions that route the plain `solve_spp` through the
core `SolvePolicy` and `SolveInputs.robust`:

- `SppConfig(robust=...)` delegates to `sidereon_core::positioning::RobustConfig`
  (Huber/IRLS). On a clean set it tracks the static solve; on a set with one
  gross blunder on a satellite whose own residual keeps at least half of the
  bias (its redundancy, from `core_goldens.json`), it down-weights the outlier
  and lands far closer to truth than the static variance-weighted solve. A
  blunder on the low-redundancy G08 is beyond raw-residual reweighting; that
  case checks delegation to the core robust solve only.
- `solve_spp(..., coarse_search_seeds=n)` delegates to
  `SolvePolicy.coarse_search_seeds`: from an antipodal cold start the static solve
  is refused, while the golden-spiral seed lattice recovers the reference fix.
- `solve_spp(..., max_pdop=p)` delegates to `SolvePolicy.validation.max_pdop`: a
  tight ceiling refuses the geometry; a non-positive ceiling is a value error.

The companion integrity entries already covered elsewhere are FDE (test_qc.py:
`qc_fde` excludes a deliberately corrupted satellite) and the
precise-with-broadcast fallback (test_broadcast_fallback.py:
`solve_with_fallback` picks broadcast when no precise product covers the epoch).

All numbers are the crate's own; nothing is computed here.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, STATUS_LABELS, core_goldens, hex_to_f64

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
AGREEMENT_BOUND_M = 1.0e-6


def _trace():
    with open(TRACE) as fh:
        return json.load(fh)["fixture"]


def _load_sp3(fx):
    path = os.path.join(CORE_FIXTURES, "sp3", fx["inputs"]["sp3_file"])
    with open(path, "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _truth(fx):
    return np.array([hex_to_f64(x) for x in _CONSISTENT["position_m"]])


def _consistent_observations(fx):
    return [
        (sat, hex_to_f64(p))
        for sat, p in zip(CONSISTENT_SATS, _CONSISTENT["pseudoranges_m"])
    ]


def _config(fx, observations, initial_guess, robust=None):
    inp = fx["inputs"]
    return sidereon.SppConfig(
        observations=[sidereon.SppObservation(s, p) for s, p in observations],
        t_rx_j2000_s=hex_to_f64(inp["t_rx_j2000_s"]),
        t_rx_second_of_day_s=hex_to_f64(inp["t_rx_sod_s"]),
        day_of_year=hex_to_f64(inp["doy"]),
        initial_guess=initial_guess,
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
        klobuchar=sidereon.SppKlobucharCoeffs(
            alpha=[hex_to_f64(x) for x in inp["klobuchar_alpha"]],
            beta=[hex_to_f64(x) for x in inp["klobuchar_beta"]],
        ),
        with_geodetic=True,
        robust=robust,
    )


def _warm_guess(fx):
    return [hex_to_f64(x) for x in fx["frozen"]["initial_guess_x0"]]


# --- SppRobustConfig value type -------------------------------------------


def test_robust_config_defaults_and_getters():
    cfg = sidereon.SppRobustConfig()
    # Mirrors the core RobustConfig::default() (textbook ~95%-efficiency Huber k).
    assert cfg.huber_k == pytest.approx(1.345)
    assert cfg.scale_floor_m > 0.0
    assert cfg.max_outer >= 1
    assert cfg.outer_tol_m > 0.0
    assert "SppRobustConfig(" in repr(cfg)

    custom = sidereon.SppRobustConfig(
        huber_k=2.0, scale_floor_m=3.0, max_outer=7, outer_tol_m=1e-4
    )
    assert custom.huber_k == 2.0
    assert custom.scale_floor_m == 3.0
    assert custom.max_outer == 7
    assert custom.outer_tol_m == pytest.approx(1e-4)


def test_config_robust_round_trips():
    fx = _trace()
    plain = _config(fx, _consistent_observations(fx), _warm_guess(fx))
    assert plain.robust is None

    rcfg = sidereon.SppRobustConfig(huber_k=1.5)
    withr = _config(fx, _consistent_observations(fx), _warm_guess(fx), robust=rcfg)
    assert withr.robust is not None
    assert withr.robust.huber_k == 1.5


# --- robust (Huber/IRLS) solve --------------------------------------------


def test_robust_solve_tracks_static_on_a_clean_set():
    fx = _trace()
    sp3 = _load_sp3(fx)
    obs = _consistent_observations(fx)
    guess = _warm_guess(fx)
    truth = _truth(fx)

    static = sidereon.solve_spp(sp3, _config(fx, obs, guess))
    robust = sidereon.solve_spp(
        sp3, _config(fx, obs, guess, robust=sidereon.SppRobustConfig())
    )

    # The clean consistent set has near-zero residuals, so reweighting keeps full
    # weight and the robust fix stays on truth alongside the static one.
    assert np.linalg.norm(static.position - truth) < AGREEMENT_BOUND_M
    assert np.linalg.norm(robust.position - truth) < 1.0e-3


# A satellite's redundancy is the share of a bias on its pseudorange that the
# static variance-weighted solve leaves in its own residual (the diagonal of
# I - H (H^T W H)^-1 H^T W); `scripts/core_goldens` measures it for this
# scenario's bias. The rest of the bias moves the position and shows up in the
# other satellites' residuals. Huber reweighting of raw residuals down-weights
# the satellites whose residuals stand out, so it can reject a bias only on a
# satellite that keeps enough of the bias in its own residual.
_ROBUST_FAULTS = core_goldens()["spp_robust_faults"]
BLUNDER_M = 300.0
IDENTIFIABLE_REDUNDANCY = 0.5
IDENTIFIABLE_SATS = sorted(
    sat
    for sat, share in _ROBUST_FAULTS["redundancy"].items()
    if share >= IDENTIFIABLE_REDUNDANCY
)


def _blunder(fx, blunder_sat):
    return [
        (sat, pr + (BLUNDER_M if sat == blunder_sat else 0.0))
        for sat, pr in _consistent_observations(fx)
    ]


def test_robust_fault_golden_covers_this_scenario():
    assert _ROBUST_FAULTS["bias_m"] == BLUNDER_M
    assert sorted(_ROBUST_FAULTS["redundancy"]) == sorted(CONSISTENT_SATS)
    assert all(0.0 < share < 1.0 for share in _ROBUST_FAULTS["redundancy"].values())
    # The satellites the reweighting is expected to reject are not an empty set.
    assert len(IDENTIFIABLE_SATS) >= 3


@pytest.mark.parametrize("blunder_sat", IDENTIFIABLE_SATS)
def test_robust_solve_downweights_a_gross_blunder(blunder_sat):
    fx = _trace()
    sp3 = _load_sp3(fx)
    guess = _warm_guess(fx)
    truth = _truth(fx)
    observations = _blunder(fx, blunder_sat)

    static = sidereon.solve_spp(sp3, _config(fx, observations, guess))
    # The default reweighting runs until it settles; its outer-solve cap is a
    # safeguard, not a working budget.
    robust = sidereon.solve_spp(
        sp3, _config(fx, observations, guess, robust=sidereon.SppRobustConfig())
    )

    static_err = float(np.linalg.norm(static.position - truth))
    robust_err = float(np.linalg.norm(robust.position - truth))

    # The static variance-weighted solve smears the 300 m blunder across the
    # geometry; the settled IRLS reweighting down-weights the outlier and
    # recovers a near-truth fix.
    assert robust.converged
    assert robust.status == "selection_settled"
    assert static_err > 50.0
    assert robust_err < 10.0
    assert robust_err < static_err / 10.0


def test_robust_solve_on_a_low_redundancy_blunder_matches_core():
    # G08 sits 18.6 degrees above the horizon, alone in its part of the sky: its
    # own residual keeps only about 17% of a bias on it (redundancy 0.168 in the
    # golden), and the rest lands on the other satellites, G27 most of all.
    # Huber reweighting of raw residuals then down-weights G27 rather than G08,
    # and the settled Huber estimate lies about 314 m from truth; no
    # reweighting budget changes that. Only delegation to the core robust solve
    # is checked here, against the core's own result for the same inputs.
    golden = _ROBUST_FAULTS["g08_robust_default"]
    assert _ROBUST_FAULTS["redundancy"]["G08"] < IDENTIFIABLE_REDUNDANCY

    fx = _trace()
    sp3 = _load_sp3(fx)
    robust = sidereon.solve_spp(
        sp3,
        _config(
            fx,
            _blunder(fx, "G08"),
            _warm_guess(fx),
            robust=sidereon.SppRobustConfig(),
        ),
    )

    expected_position = np.array([hex_to_f64(x) for x in golden["position_m"]])
    assert np.array_equal(
        robust.position.view(np.int64), expected_position.view(np.int64)
    )
    assert robust.rx_clock_s == hex_to_f64(golden["rx_clock_s"])
    assert robust.used_sats == golden["used_sats"]
    expected_residuals = np.array([hex_to_f64(x) for x in golden["residuals_m"]])
    assert np.array_equal(
        np.array(robust.residuals_m).view(np.int64),
        expected_residuals.view(np.int64),
    )
    assert robust.status == STATUS_LABELS[golden["status"]]
    assert robust.converged == golden["converged"]


def test_robust_composes_with_batch():
    fx = _trace()
    sp3 = _load_sp3(fx)
    guess = _warm_guess(fx)
    obs = _consistent_observations(fx)

    rcfg = sidereon.SppRobustConfig()
    single = sidereon.solve_spp(sp3, _config(fx, obs, guess, robust=rcfg))
    batch = sidereon.solve_spp_batch(
        sp3, [_config(fx, obs, guess, robust=rcfg) for _ in range(3)]
    )
    assert len(batch) == 3
    for sol in batch:
        assert np.array_equal(
            sol.position.view(np.int64), single.position.view(np.int64)
        )


# --- coarse-search cold start ---------------------------------------------


def test_coarse_search_recovers_an_antipodal_cold_start():
    fx = _trace()
    sp3 = _load_sp3(fx)
    obs = _consistent_observations(fx)
    truth = _truth(fx)

    # An antipodal prior puts the static single solve in the wrong convergence
    # basin: it is refused outright.
    bad_guess = [-truth[0], -truth[1], -truth[2], 0.0]
    with pytest.raises(sidereon.SolveError):
        sidereon.solve_spp(sp3, _config(fx, obs, bad_guess))

    # The golden-spiral seed lattice lands one seed in the basin and the best
    # redundant converged candidate recovers the reference fix.
    recovered = sidereon.solve_spp(
        sp3, _config(fx, obs, bad_guess), coarse_search_seeds=24
    )
    assert recovered.geodetic is not None
    assert np.linalg.norm(recovered.position - truth) < AGREEMENT_BOUND_M


def test_coarse_search_rejects_zero_seeds():
    fx = _trace()
    sp3 = _load_sp3(fx)
    obs = _consistent_observations(fx)
    cfg = _config(fx, obs, _warm_guess(fx))
    with pytest.raises(ValueError):
        sidereon.solve_spp(sp3, cfg, coarse_search_seeds=0)


# --- max_pdop validation gate ---------------------------------------------


def test_max_pdop_refuses_a_loose_geometry():
    fx = _trace()
    sp3 = _load_sp3(fx)
    obs = _consistent_observations(fx)
    guess = _warm_guess(fx)

    # A generous ceiling leaves the accepted fix unchanged.
    base = sidereon.solve_spp(sp3, _config(fx, obs, guess))
    gated = sidereon.solve_spp(sp3, _config(fx, obs, guess), max_pdop=100.0)
    assert np.array_equal(gated.position.view(np.int64), base.position.view(np.int64))

    # A ceiling below the geometry's actual PDOP refuses the fix.
    assert base.dop is not None and base.dop.pdop > 0.1
    with pytest.raises(sidereon.SolveError):
        sidereon.solve_spp(sp3, _config(fx, obs, guess), max_pdop=0.1)


def test_max_pdop_rejects_non_positive():
    fx = _trace()
    sp3 = _load_sp3(fx)
    obs = _consistent_observations(fx)
    with pytest.raises(ValueError):
        sidereon.solve_spp(sp3, _config(fx, obs, _warm_guess(fx)), max_pdop=-1.0)
