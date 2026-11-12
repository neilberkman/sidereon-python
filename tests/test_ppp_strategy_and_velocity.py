import math

import pytest
import sidereon


def _velocity_row(
    satellite: str, position: list[float]
) -> sidereon.PppVelocityObservation:
    receiver = sidereon.PppReceiverVelocityState(
        [0.0, 0.0, 0.0], [3.0, -2.0, 0.5], 0.25
    )
    seed = sidereon.PppVelocityObservation(
        satellite, position, [0.0, 0.0, 0.0], 0.0, 0.1, 0.0
    )
    measured = sidereon.predict_range_rate_m_s(seed, receiver).range_rate_m_s
    return sidereon.PppVelocityObservation(
        satellite, position, [0.0, 0.0, 0.0], measured, 0.1, 0.0
    )


def test_ppp_strategy_defaults_hashes_and_value_objects_compare() -> None:
    assert (
        sidereon.PppAutoInitStrategy.default() == sidereon.PppAutoInitStrategy.REFERENCE
    )
    assert hash(sidereon.PppAutoInitStrategy.REFERENCE) == hash(
        sidereon.PppAutoInitStrategy.REFERENCE
    )
    assert sidereon.PppAutoInitOptions() == sidereon.PppAutoInitOptions()
    assert sidereon.PppMeasurementWeights() == sidereon.PppMeasurementWeights()
    assert (
        sidereon.PppSatelliteClockCorrections()
        == sidereon.PppSatelliteClockCorrections()
    )
    assert sidereon.VMF_SITE_MAX_SAMPLES == 8
    assert sidereon.VMF_SITE_SINGLE_SAMPLE_CLAMP_DAYS == 0.25


def test_nested_core_value_equality_is_covered_by_parent_objects() -> None:
    base = sidereon.PppObservation("G01", "G01#1", 20.0, 21.0)
    with_signals = sidereon.PppObservation(
        "G01", "G01#1", 20.0, 21.0, signals=("1C", "2W", "1C", "2W")
    )
    assert base == sidereon.PppObservation("G01", "G01#1", 20.0, 21.0)
    assert base != with_signals  # FloatObservationSignals participates.

    niell = sidereon.PppTroposphereOptions(enabled=True)
    vmf = sidereon.PppTroposphereOptions(
        enabled=True, vmf1_samples=[(60_000.0, 0.001, 0.0005)]
    )
    assert niell == sidereon.PppTroposphereOptions(enabled=True)
    assert niell != vmf  # TropoMapping and VmfSiteSeries participate.

    explicit_a = sidereon.PppAutoInitOptions(
        initial_guess_position_m=[1.0, 2.0, 3.0], initial_guess_clock_m=4.0
    )
    explicit_b = sidereon.PppAutoInitOptions(
        initial_guess_position_m=[1.0, 2.0, 3.0], initial_guess_clock_m=5.0
    )
    assert explicit_a != explicit_b  # PppInitialGuess participates.


def test_precise_velocity_prediction_and_solve_exact_geometry() -> None:
    receiver = sidereon.PppReceiverVelocityState.stationary_at([0.0, 0.0, 0.0])
    row = sidereon.PppVelocityObservation(
        "G01", [20_200_000.0, 0.0, 0.0], [10.0, 2.0, 3.0], 0.0, 0.1, 1.5
    )
    prediction = sidereon.predict_range_rate_m_s(row, receiver)
    assert prediction.los_unit == [1.0, 0.0, 0.0]
    assert prediction.range_rate_m_s == 8.5

    rows = [
        _velocity_row("G01", [20_200_000.0, 0.0, 0.0]),
        _velocity_row("G02", [0.0, 21_100_000.0, 0.0]),
        _velocity_row("G03", [0.0, 0.0, 22_300_000.0]),
        _velocity_row("G04", [-20_900_000.0, 19_700_000.0, 21_500_000.0]),
        _velocity_row("G05", [18_600_000.0, -20_400_000.0, 23_100_000.0]),
    ]
    solution = sidereon.solve_velocity_least_squares(rows, [0.0, 0.0, 0.0])
    assert solution.velocity_m_s == pytest.approx([3.0, -2.0, 0.5], abs=1e-12)
    assert solution.clock_drift_m_s == pytest.approx(0.25, abs=1e-12)
    assert solution.used_sats == ["G01", "G02", "G03", "G04", "G05"]
    assert solution.downweighted_sats == []
    assert solution.robust_iterations == 0
    assert solution.robust_scale_m_s is None
    assert math.isfinite(solution.residual_rms_m_s)


def test_precise_velocity_refusals_are_exact() -> None:
    coincident = sidereon.PppVelocityObservation(
        "G01", [1.0, 2.0, 3.0], [0.0, 0.0, 0.0], 0.0, 1.0, 0.0
    )
    receiver = sidereon.PppReceiverVelocityState.stationary_at([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="^receiver and satellite positions coincide$"):
        sidereon.predict_range_rate_m_s(coincident, receiver)

    with pytest.raises(
        sidereon.SolveError,
        match="^too few velocity observations: required 4, got 0$",
    ) as first:
        sidereon.solve_velocity_least_squares([], [0.0, 0.0, 0.0])
    with pytest.raises(sidereon.SolveError) as equal:
        sidereon.solve_velocity_least_squares([], [0.0, 0.0, 0.0])
    with pytest.raises(sidereon.SolveError) as unequal:
        sidereon.solve_velocity_least_squares(
            [], [0.0, 0.0, 0.0], sidereon.PppVelocityConfig(5)
        )

    assert isinstance(first.value.detail, dict)
    assert first.value.detail == equal.value.detail
    assert first.value.detail != unequal.value.detail
    assert first.value.detail == {
        "family": "VelocitySolveError",
        "message": "too few velocity observations: required 4, got 0",
        "kind": "too_few_observations",
        "required": 4,
        "actual": 0,
    }


def test_ppp_value_classes_compare_by_value_and_are_unhashable():
    first = sidereon.PppSatelliteClockCorrections()
    second = sidereon.PppSatelliteClockCorrections()
    assert first == second
    assert first is not second
    with pytest.raises(TypeError):
        hash(first)
