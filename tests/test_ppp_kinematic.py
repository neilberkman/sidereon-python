from pathlib import Path

import pytest
import sidereon


def empty_epoch(observations=()):
    return sidereon.PppEpoch(
        sidereon.PppCivilDateTime(2026, 4, 30, 9, 45, 0.0),
        2_461_160.0,
        -0.09375,
        830_000_000.0,
        observations,
    )


def source():
    return sidereon.load_sp3(
        Path("tests/fixtures/sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3")
    )


def test_kinematic_defaults_variants_and_value_equality():
    state = sidereon.PppKinematicState.default()
    assert (state.position_m, state.clock_m, state.ztd_residual_m) == (
        [0.0, 0.0, 0.0],
        0.0,
        0.0,
    )
    assert state.ambiguities_m == {}
    assert state.dimension() == 5
    assert state == sidereon.PppKinematicState()
    assert state != sidereon.PppKinematicState(clock_m=1.0)

    random_walk = sidereon.PppKinematicPositionProcessNoise.default()
    acceleration = sidereon.PppKinematicPositionProcessNoise.white_noise_acceleration(
        0.25
    )
    assert (random_walk.kind, random_walk.spectral_density_m2_s) == (
        "random_walk",
        1.0,
    )
    assert random_walk.spectral_density_m2_s3 is None
    assert (acceleration.kind, acceleration.spectral_density_m2_s3) == (
        "white_noise_acceleration",
        0.25,
    )
    assert acceleration != random_walk

    hold = sidereon.PppKinematicMotionModel.default()
    velocity = sidereon.PppKinematicMotionModel.constant_velocity([3.0, -2.0, 1.0])
    assert (hold.kind, hold.velocity_m_s) == ("hold", None)
    assert (velocity.kind, velocity.velocity_m_s) == (
        "constant_velocity",
        [3.0, -2.0, 1.0],
    )
    assert hold == sidereon.PppKinematicMotionModel.hold()

    noise = sidereon.PppKinematicProcessNoise.default()
    config = sidereon.PppKinematicConfig.default()
    assert (
        noise.clock_white_m2_s,
        noise.ztd_random_walk_m2_s,
        noise.ambiguity_hold_m2_s,
    ) == (100.0, 1.0e-6, 0.0)
    assert config.initial_covariance_m2 == [
        [100_000_000.0 if row == column else 0.0 for column in range(5)]
        for row in range(5)
    ]
    assert config.new_ambiguity_variance_m2 == 100_000_000.0
    assert (config.weights.code, config.weights.phase) == (1.0, 100.0)
    assert config == sidereon.PppKinematicConfig()


def test_predict_kinematic_state_exact_mean_covariance_and_ambiguity_alignment():
    state = sidereon.PppKinematicState(position_m=[1.0, 2.0, 3.0])
    noise = sidereon.PppKinematicProcessNoise(
        sidereon.PppKinematicPositionProcessNoise.random_walk(3.0),
        5.0,
        7.0,
        11.0,
    )
    config = sidereon.PppKinematicConfig(
        motion=sidereon.PppKinematicMotionModel.constant_velocity([3.0, -2.0, 1.0]),
        process_noise=noise,
        new_ambiguity_variance_m2=99.0,
    )
    covariance = sidereon.predict_kinematic_state(
        state,
        [[4.0 if row == column else 0.0 for column in range(5)] for row in range(5)],
        2.0,
        ["G08#1"],
        config,
    )
    assert state.position_m == [7.0, -2.0, 5.0]
    assert state.ambiguities_m == {"G08#1": 0.0}
    assert state.dimension() == 6
    assert [covariance[index][index] for index in range(6)] == [
        10.0,
        10.0,
        10.0,
        14.0,
        18.0,
        121.0,
    ]


def test_kinematic_refusals_have_complete_structured_details():
    state = sidereon.PppKinematicState.default()
    config = sidereon.PppKinematicConfig.default()
    indefinite = [[0.0] * 5 for _ in range(5)]
    indefinite[0][0] = indefinite[1][1] = 1.0
    indefinite[0][1] = indefinite[1][0] = 2.0
    with pytest.raises(sidereon.SolveError) as predicted:
        sidereon.predict_kinematic_state(state, indefinite, 0.0, [], config)
    assert predicted.value.detail == {
        "family": "KinematicSolveError",
        "kind": "invalid_input",
        "field": "kinematic PPP covariance positive semidefinite",
        "reason": "must be positive semidefinite within tolerance",
    }

    summary, corrected_covariance = sidereon.correct_kinematic_state(
        source(), 0, empty_epoch(), state, config.initial_covariance_m2, config
    )
    assert (summary.innovation_rms_m, summary.used_sats) == (0.0, [])
    assert summary.ssr_bias_exclusions == []
    assert summary.unplaced_observations == []
    assert corrected_covariance == config.initial_covariance_m2
    unplaceable = sidereon.PppObservation("G01", "G01", 0.0, 0.0)
    solution = sidereon.solve_kinematic_ppp(
        source(), [empty_epoch([unplaceable])], config
    )[0]
    assert solution.status == sidereon.PppKinematicEpochStatus.PREDICTED_ONLY
    assert solution.status.label == "predicted_only"
    assert solution.position_m == [0.0, 0.0, 0.0]
    assert solution.position_covariance_m2 == [
        [100_000_000.0 if row == column else 0.0 for column in range(3)]
        for row in range(3)
    ]
    assert len(solution.unplaced_observations) == 1
    assert solution.unplaced_observations[0].reason == "code_not_positive"
    assert (
        solution
        == sidereon.solve_kinematic_ppp(source(), [empty_epoch([unplaceable])], config)[
            0
        ]
    )

    with pytest.raises(sidereon.SolveError) as solved:
        sidereon.solve_kinematic_ppp(source(), [], config)
    assert solved.value.detail == {
        "family": "KinematicSolveError",
        "kind": "invalid_input",
        "field": "kinematic PPP epochs",
        "reason": "must not be empty",
    }
