"""Moving-baseline RTK delegates to ``sidereon_core::rtk_filter::moving_baseline``.

This mirrors the core ``recovers_baseline_per_epoch_as_base_moves`` test: a base
walking along a track with a constant true baseline to the rover, perfect
synthetic double-difference observations, and the LAMBDA integer fix recovering
the baseline at each epoch.
"""

import numpy as np
import pytest
import sidereon

C_M_S = 299792458.0
F_L1_HZ = 1575.42e6
LAMBDA = C_M_S / F_L1_HZ

# (id, tx position metres, integer cycle bias); G01 is the reference.
SATS = [
    ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
    ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 4),
    ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
    ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 9),
    ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -3),
]
AMBIGUITY_IDS = ["G02", "G03", "G04", "G05"]


def _range_m(sat, recv):
    d = np.asarray(sat) - np.asarray(recv)
    return float(np.linalg.norm(d))


def _sat_meas(pos, sat_id, cycles, base, rover):
    return sidereon.RtkSatMeasurement(
        sat=sat_id,
        sd_ambiguity_id=sat_id,
        base_code_m=_range_m(pos, base),
        base_phase_m=_range_m(pos, base),
        rover_code_m=_range_m(pos, rover),
        rover_phase_m=_range_m(pos, rover) + cycles * LAMBDA,
        base_tx_pos=pos,
        rover_tx_pos=pos,
        pos=pos,
    )


def _epoch(base, baseline):
    rover = [base[i] + baseline[i] for i in range(3)]
    references = [_sat_meas(SATS[0][1], SATS[0][0], SATS[0][2], base, rover)]
    nonref = [_sat_meas(p, sid, c, base, rover) for (sid, p, c) in SATS[1:]]
    return sidereon.RtkEpoch(references=references, nonref=nonref, dt_s=0.0)


def _moving_epoch(base, baseline):
    return sidereon.MovingBaselineEpoch(
        base_position_m=base,
        epoch=_epoch(base, baseline),
        ambiguity_ids=AMBIGUITY_IDS,
        ambiguity_satellites={sid: sid for sid in AMBIGUITY_IDS},
        wavelengths_m={sid: LAMBDA for sid in AMBIGUITY_IDS},
        offsets_m={sid: 0.0 for sid in AMBIGUITY_IDS},
    )


def test_moving_baseline_recovers_constant_baseline_as_base_moves():
    bases = [
        [4_075_580.0, 931_854.0, 4_801_568.0],
        [4_075_585.0, 931_860.0, 4_801_572.0],
        [4_075_590.0, 931_867.0, 4_801_575.0],
    ]
    truth = [1.2, -0.85, 0.91]
    epochs = [_moving_epoch(b, truth) for b in bases]

    model = sidereon.RtkMeasurementModel(
        code_sigma_m=0.3, phase_sigma_m=0.003, sagnac=False, stochastic="simple"
    )
    solutions = sidereon.solve_moving_baseline(
        epochs,
        model,
        float_options=sidereon.RtkFloatOptions(
            position_tol_m=1e-3, ambiguity_tol_m=1e-6, max_iterations=10
        ),
        fixed_options=sidereon.RtkFixedOptions(
            position_tol_m=1e-3,
            ambiguity_tol_m=1e-6,
            max_iterations=10,
            ratio_threshold=3.0,
        ),
        initial_baseline_m=[-30.0, 25.0, -10.0],
        warm_start=True,
    )

    assert len(solutions) == 3
    for solution, base in zip(solutions, bases):
        assert solution.fixed
        assert solution.integer_status == sidereon.IntegerStatus.FIXED
        np.testing.assert_allclose(solution.baseline, truth, atol=1e-3)
        np.testing.assert_allclose(solution.base_position, base, atol=1e-6)
        assert solution.baseline_length_m == pytest.approx(
            float(np.linalg.norm(truth)), abs=1e-3
        )
