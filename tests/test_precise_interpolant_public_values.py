"""Public numerical contract for a sample-backed precise interpolant."""

import math

import numpy as np
import pytest
import sidereon

C_M_S = 299_792_458.0
STEP_S = 900.0
BASE = np.asarray([20_200_000.0, 13_400_000.0, 21_700_000.0])
SLOPE_PER_NODE = np.asarray([12.0, -8.0, 5.0])
CURVE_PER_NODE2 = np.asarray([0.03, -0.02, 0.01])


def _samples():
    rows = []
    for index in range(15):
        position = BASE + SLOPE_PER_NODE * index + CURVE_PER_NODE2 * index * index
        rows.append(
            sidereon.PreciseEphemerisSample(
                "G01",
                index * STEP_S,
                position.tolist(),
                1.0e-6 + index * 1.0e-10,
            )
        )
    return rows


def _position_at_index(index):
    return BASE + SLOPE_PER_NODE * index + CURVE_PER_NODE2 * index * index


def test_precise_interpolant_scalar_batch_and_exact_query_values():
    interpolant = sidereon.PreciseEphemerisInterpolant.from_samples(_samples())
    widened = interpolant.with_interpolation_options(13.0)
    assert interpolant.gap_threshold_factor == 1.5
    assert widened.gap_threshold_factor == 13.0
    assert interpolant.satellites == ["G01"]

    epoch_s = 5.5 * STEP_S
    node_index = epoch_s / STEP_S
    expected_position = _position_at_index(node_index)
    expected_clock = 1.0e-6 + node_index * 1.0e-10
    state = interpolant.position_at_j2000_seconds("G01", epoch_s)
    np.testing.assert_allclose(state.position_m, expected_position, rtol=0.0, atol=2e-8)
    assert state.clock_s == pytest.approx(expected_clock, rel=0.0, abs=1e-16)
    # These public samples carry position and clock only, so no velocity is supplied.
    assert state.velocity_m_s is None

    epochs = np.asarray([epoch_s], dtype=np.float64)
    batch = interpolant.observable_states_at_j2000_s(["G01"], epochs)
    assert batch.statuses == [sidereon.ObservableStateElementStatus.VALID]
    np.testing.assert_allclose(
        batch.positions_ecef_m[0], expected_position, rtol=0.0, atol=2e-8
    )
    assert batch.clocks_s[0] == pytest.approx(expected_clock, rel=0.0, abs=1e-16)
    shared = interpolant.observable_states_at_shared_j2000_s(["G01"], epoch_s)
    np.testing.assert_array_equal(shared.positions_ecef_m, batch.positions_ecef_m)
    np.testing.assert_array_equal(shared.clocks_s, batch.clocks_s)

    query = sidereon.ExactEpochQuery.from_binary_j2000_seconds(epoch_s)
    exact_state = interpolant.position_at_epoch_query("G01", query)
    np.testing.assert_array_equal(exact_state.position_m, state.position_m)
    selected = interpolant.selected_state_at_epoch_query("G01", query, query)
    assert selected is not None
    np.testing.assert_allclose(
        selected.position_ecef_m, expected_position, rtol=0.0, atol=2e-8
    )
    assert selected.clock_s == pytest.approx(expected_clock, rel=0.0, abs=1e-16)
    assert selected.group_delay_s is None
    assert selected.degraded_reason is None
    assert interpolant.transmit_epoch_clock_at_epoch_query("G01", query, query) == (
        pytest.approx(expected_clock, rel=0.0, abs=1e-16),
        None,
    )
    assert interpolant.ephemeris_variance_at_epoch_query("G01", query, query) == 0.0

    # The core's peph2pos term uses the exact query advanced by its 1 ms step.
    after_query = query.checked_add_binary_seconds(0.001)
    after_state = interpolant.position_at_epoch_query("G01", after_query)
    velocity_over_step = (
        np.asarray(after_state.position_m) - np.asarray(exact_state.position_m)
    ) / 0.001
    expected_relativity = (
        -2.0
        * math.fsum(
            float(a * b) for a, b in zip(exact_state.position_m, velocity_over_step)
        )
        / C_M_S
        / C_M_S
    )
    relativity = interpolant.clock_relativity_for_state_at_epoch_query(
        "G01", query, exact_state.position_m
    )
    assert relativity.kind == "term"
    assert relativity.term_s == pytest.approx(expected_relativity, rel=2e-10, abs=1e-25)
