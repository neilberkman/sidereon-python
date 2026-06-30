"""SPP solve through the binding reproduces the crate-side reference numbers.

The fixture `spp_trace_L0_minimal.json` carries the exact inputs (as float64 bit
patterns) AND the converged reference solution the crate asserts on. We feed the
binding the same inputs and require the same answer, within the crate's own
agreement bound (AGREEMENT_BOUND_M = 1e-6 m, the independent-solve tolerance the
fixture documents). No truth is invented here.
"""

import json
import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES, hex_to_f64

# The independent-solve agreement bound the crate uses for this fixture.
AGREEMENT_BOUND_M = 1.0e-6


def _load_fixture():
    path = os.path.join(CORE_FIXTURES, "spp_trace_L0_minimal.json")
    with open(path) as fh:
        return json.load(fh)["fixture"]


def _load_sp3_and_config(fx):
    inp = fx["inputs"]

    sp3_path = os.path.join(CORE_FIXTURES, "sp3", inp["sp3_file"])
    with open(sp3_path, "rb") as fh:
        sp3 = sidereon.load_sp3(fh.read())

    observations = [
        sidereon.SppObservation(o["sat_id"], hex_to_f64(o["p_meas_m"]))
        for o in inp["observations"]
    ]
    initial_guess = [hex_to_f64(x) for x in fx["frozen"]["initial_guess_x0"]]

    config = sidereon.SppConfig(
        observations=observations,
        t_rx_j2000_s=hex_to_f64(inp["t_rx_j2000_s"]),
        t_rx_second_of_day_s=hex_to_f64(inp["t_rx_sod_s"]),
        day_of_year=hex_to_f64(inp["doy"]),
        initial_guess=initial_guess,
        corrections=sidereon.SppCorrections(
            # L0_minimal: geometry + clock + Sagnac only, no iono, no tropo.
            ionosphere=False,
            troposphere=False,
        ),
        klobuchar=sidereon.SppKlobucharCoeffs(
            alpha=[hex_to_f64(x) for x in inp["klobuchar_alpha"]],
            beta=[hex_to_f64(x) for x in inp["klobuchar_beta"]],
        ),
        met=sidereon.SppSurfaceMet(
            pressure_hpa=hex_to_f64(inp["met"]["pressure_hpa"]),
            temperature_k=hex_to_f64(inp["met"]["temperature_k"]),
            relative_humidity=hex_to_f64(inp["met"]["relative_humidity"]),
        ),
        with_geodetic=True,
    )
    return sp3, config


def test_spp_matches_reference():
    fx = _load_fixture()
    sp3, config = _load_sp3_and_config(fx)
    assert sp3.epoch_count == 96
    sol = sidereon.solve_spp(sp3, config)

    expected = [hex_to_f64(x) for x in fx["final_solution"]["x"]]
    got = sol.position
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float64
    assert np.linalg.norm(got - np.array(expected[:3])) < AGREEMENT_BOUND_M

    expected_clock_s = hex_to_f64(fx["final_solution"]["rx_clock_s"])
    assert abs(sol.rx_clock_s - expected_clock_s) < 1.0e-9

    # Pythonic surface is populated.
    assert sol.geodetic is not None
    assert len(sol.used_sats) == len(sol.residuals_m)
    assert "SppSolution(" in repr(sol)


def test_spp_solution_exposes_dop():
    fx = _load_fixture()
    sp3, config = _load_sp3_and_config(fx)
    sol = sidereon.solve_spp(sp3, config)

    # A rank-sufficient solve carries DOP diagnostics; the getter maps the
    # core Option<Dop> onto the existing Dop pyclass (None if rank-deficient).
    dop = sol.dop
    assert dop is not None
    assert isinstance(dop, sidereon.Dop)
    for scalar in (dop.gdop, dop.pdop, dop.hdop, dop.vdop, dop.tdop):
        assert np.isfinite(scalar) and scalar > 0.0
    # gdop^2 == pdop^2 + tdop^2 by construction.
    assert abs(dop.gdop**2 - (dop.pdop**2 + dop.tdop**2)) < 1e-9 * dop.gdop**2
    # The solution's system-tagged TDOP agrees with the geometry's scalar.
    if sol.system_tdops:
        assert abs(sol.system_tdops[0][1] - dop.tdop) < 1e-9

    # Absolute per-system clocks: the reference entry equals rx_clock_s.
    clocks = sol.system_clocks_s
    assert len(clocks) >= 1
    assert clocks[0][1] == sol.rx_clock_s


def _bits(x):
    return np.float64(x).view(np.int64)


def test_solve_spp_batch_is_bit_identical_to_single_and_serial():
    """The GIL-released parallel batch must reproduce solve_spp exactly.

    Each epoch is independent, so the rayon fan-out (parallel=True) and the
    serial path (parallel=False) must both be bit-for-bit identical to calling
    solve_spp once per config. This is the core's bit-identity guarantee
    surfaced through the binding.
    """
    fx = _load_fixture()
    sp3, config = _load_sp3_and_config(fx)

    # A stream of independent receive epochs (same epoch repeated is enough to
    # prove bit-identity; the solver carries no cross-epoch state).
    configs = [config for _ in range(5)]
    single = sidereon.solve_spp(sp3, config)

    for parallel in (True, False):
        batch = sidereon.solve_spp_batch(sp3, configs, parallel=parallel)
        assert len(batch) == len(configs)
        for sol in batch:
            assert _bits(sol.rx_clock_s) == _bits(single.rx_clock_s)
            assert np.array_equal(
                sol.position.view(np.int64), single.position.view(np.int64)
            )

    # Empty batch is well-defined.
    assert sidereon.solve_spp_batch(sp3, []) == []
