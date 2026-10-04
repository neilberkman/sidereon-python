import numpy as np
import pytest
import sidereon


def _state():
    return sidereon.NavState(
        0.0,
        [6_378_137.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        np.eye(3, dtype=np.float64),
    )


def test_strapdown_mechanizer_and_function_advance_state():
    start = _state()
    increment = sidereon.CorrectedImuIncrement(
        0.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.01
    )
    stepped = sidereon.mechanize_ecef(start, increment, sidereon.MechanizationConfig())
    default_stepped = sidereon.mechanize_ecef(start, increment)
    assert stepped.t_j2000_s == 0.01
    assert np.all(np.isfinite(stepped.position_ecef_m))
    assert default_stepped.t_j2000_s == stepped.t_j2000_s
    np.testing.assert_array_equal(
        default_stepped.position_ecef_m, stepped.position_ecef_m
    )

    mechanizer = sidereon.StrapdownMechanizer(start)
    propagated = mechanizer.propagate(
        sidereon.ImuSample.increment(0.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.01)
    )
    assert propagated.t_j2000_s == 0.01
    assert mechanizer.state.t_j2000_s == 0.01
    assert start.attitude_yaw_pitch_roll_rad == (0.0, 0.0, 0.0)
    assert isinstance(start.attitude_yaw_pitch_roll_rad, tuple)
    assert np.allclose(start.attitude_yaw_pitch_roll_rad, [0.0, 0.0, 0.0])
    assert start.attitude_quaternion_body_to_ecef == (1.0, 0.0, 0.0, 0.0)


def test_inertial_frame_helpers_and_typed_refusals():
    assert (
        sidereon.normal_gravity_mps2(0.0, 0.0)
        == sidereon.WGS84_NORMAL_GRAVITY_EQUATOR_MPS2
    )
    gravity = sidereon.gravity_ecef_mps2([6_378_137.0, 0.0, 0.0])
    assert gravity[0] < 0.0
    np.testing.assert_array_equal(gravity[1:], [0.0, 0.0])
    assert sidereon.gauss_markov_bias_decay(2.0, float("inf")) == 1.0
    assert sidereon.gauss_markov_bias_variance_increment(3.0, 2.0, float("inf")) == 18.0

    with pytest.raises(sidereon.InertialError) as refusal:
        sidereon.normal_gravity_mps2(2.0, 0.0)
    assert refusal.value.kind == "invalid_input"
    assert refusal.value.field == "lat_rad"

    try:
        sidereon.ImuRateRandomWalk(-1.0, 0.0)
    except sidereon.InertialError as error:
        assert error.kind == "invalid_input"
        assert error.field == "accel_mps2_sqrt_s"
        assert error.details["reason"] == "must be non-negative"
    else:
        raise AssertionError("invalid IMU noise density was accepted")


def test_imu_simulator_exposes_core_seeded_increment_route():
    spec = sidereon.ImuSpec.mems()
    simulator = sidereon.ImuSimulator(spec, seed=17)
    sample = simulator.sample_increment(
        sidereon.CorrectedImuIncrement(0.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.01)
    )
    assert sample.kind == "increment"
    assert sample.t_j2000_s == 0.01

    sequence = sidereon.simulate_imu_samples_from_increments(
        [sidereon.CorrectedImuIncrement(0.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.01)],
        spec,
        seed=17,
    )
    assert len(sequence.samples) == 1
    assert len(sequence.bias_history) == 1


def test_imu_simulator_rate_mode_and_truth_trajectory_route():
    start = _state()
    increment = sidereon.CorrectedImuIncrement(
        0.01, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.01
    )
    end = sidereon.mechanize_ecef(start, increment, sidereon.MechanizationConfig())
    reconstructed = sidereon.true_imu_increment_between(start, end)
    assert reconstructed.t_j2000_s == end.t_j2000_s
    assert reconstructed.dt_s == increment.dt_s
    options = sidereon.ImuSimulationOptions(
        seed=23,
        output=sidereon.ImuSimulationOutput.RATE,
        initial_bias=sidereon.ImuBias([0.01, 0.0, 0.0], [0.0, 0.0, 0.0]),
        rate_random_walk=sidereon.ImuRateRandomWalk(1.0e-5, 1.0e-6),
    )

    sequence = sidereon.simulate_imu_samples(
        [start, end], sidereon.ImuSpec.mems(), options=options
    )

    assert len(sequence.samples) == 1
    assert sequence.samples[0].kind == "rate"
    assert sequence.samples[0].specific_force_mps2 is not None
    assert sequence.samples[0].delta_velocity_mps is None
    assert sequence.bias_history[0].accel_mps2 != (0.01, 0.0, 0.0)
    assert len(sequence.rate_random_walk_history) == 1


def test_imu_error_model_stochastic_helpers_and_rotation_functions():
    spec = sidereon.ImuSpec.mems()
    assert spec.accel_bias_decay(0.0) == 1.0
    assert spec.gyro_bias_decay(0.0) == 1.0
    assert spec.accel_bias_variance_increment(0.0) == 0.0
    assert spec.gyro_bias_variance_increment(0.0) == 0.0
    assert sidereon.RANDOM_WALK_BIAS_TAU_S == float("inf")
    assert sidereon.DEFAULT_IMU_SIM_SEED == 0x4D595DF4D0F33173
    assert (
        sidereon.MechanizationConfig().coning_correction
        == sidereon.ConingCorrection.OFF
    )

    raw_sample = sidereon.ImuSample.rate(0.1, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    corrected = sidereon.ImuErrorModel().correct_sample(raw_sample, 0.0)
    assert corrected.dt_s == 0.1
    np.testing.assert_allclose(corrected.delta_velocity_mps, [0.1, 0.2, 0.3])

    quaternion = sidereon.AttitudeQuaternion(2.0, 0.0, 0.0, 0.0)
    assert quaternion.components == (1.0, 0.0, 0.0, 0.0)
    dcm = sidereon.quaternion_to_dcm(quaternion)
    np.testing.assert_array_equal(dcm, np.eye(3))
    assert sidereon.dcm_to_quaternion(dcm) == quaternion
    assert sidereon.attitude_yaw_pitch_roll_rad(dcm) == (0.0, 0.0, 0.0)
    assert isinstance(sidereon.attitude_yaw_pitch_roll_rad(dcm), tuple)
    np.testing.assert_array_equal(sidereon.reorthonormalize_dcm(dcm), dcm)
    np.testing.assert_array_equal(sidereon.rodrigues_delta_dcm([0.0, 0.0, 0.0]), dcm)

    with pytest.raises(sidereon.InertialError) as refusal:
        sidereon.ImuErrorModel().correct_sample(raw_sample, 0.1)
    assert refusal.value.kind == "non_monotonic_sample"

    with pytest.raises(sidereon.InertialError) as calibration_refusal:
        sidereon.ImuCalibration(accel_scale_misalignment=-np.eye(3))
    assert calibration_refusal.value.kind == "singular_calibration"

    with pytest.raises(sidereon.InertialError) as attitude_refusal:
        sidereon.AttitudeQuaternion(0.0, 0.0, 0.0, 0.0)
    assert attitude_refusal.value.kind == "degenerate_attitude"
