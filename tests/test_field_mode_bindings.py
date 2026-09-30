"""Fusion field-mode binding parity."""

import math
import struct

import numpy as np
import sidereon
from _helpers import core_goldens

WGS84_A_M = 6_378_137.0


def _bits(value):
    return int.from_bytes(struct.pack(">d", float(value)), "big")


def _array_bits(values):
    return [_bits(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def _golden_bits(hex_values):
    return [int(value, 16) for value in hex_values]


def _fusion_golden(name):
    """The core's run of the same filter steps, written by
    `scripts/core_goldens` to `fixtures/core_goldens.json`."""
    return core_goldens()["fusion"][name]


def _assert_update_matches(update, golden):
    assert update.applied is golden["applied"]
    assert (update.rows, update.accepted_rows, update.rejected_rows) == (
        golden["rows"],
        golden["accepted_rows"],
        golden["rejected_rows"],
    )
    assert _bits(update.nis) == int(golden["nis"], 16)
    assert _bits(update.ekf.normalized_innovation_squared) == int(golden["ekf_nis"], 16)


def _spec():
    return sidereon.ImuSpec(0.0, 0.0, 0.0, 0.0, math.inf, math.inf)


def _state(t_j2000_s=0.0, position=None, velocity=None, diagonal=None):
    if position is None:
        position = [WGS84_A_M, 0.0, 0.0]
    if velocity is None:
        velocity = [0.0, 0.0, 0.0]
    if diagonal is None:
        diagonal = [1.0] * 15
    return sidereon.InsFilterState.from_diagonal(
        sidereon.NavState(t_j2000_s, position, velocity),
        sidereon.ErrorStateLayout.FIFTEEN,
        diagonal,
    )


def _position_velocity_fix(fix_status=sidereon.GnssFixStatus.SINGLE):
    return sidereon.GnssFixMeasurement.position_velocity(
        0.0,
        [WGS84_A_M + 1.0, 2.0, -3.0],
        [0.4, -0.2, 0.1],
        np.eye(6, dtype=np.float64),
        8,
        fix_status=fix_status,
    )


def test_loose_field_mode_default_omission_matches_plain_bits():
    loose_default = sidereon.LooseCouplingConfig()
    assert loose_default.fix_status_weighting.single_sigma_multiplier == 1.0
    assert loose_default.fix_status_weighting.float_sigma_multiplier == 1.0
    assert loose_default.fix_status_weighting.fixed_sigma_multiplier == 1.0
    assert loose_default.stationary_updates is None
    assert loose_default.non_holonomic is None

    plain = sidereon.InertialFilter.with_config(
        _state(), sidereon.InertialFilterConfig(_spec())
    )
    explicit = sidereon.InertialFilter.with_config(
        _state(), sidereon.InertialFilterConfig(_spec(), loose=loose_default)
    )

    plain_update = plain.update_loose(_position_velocity_fix())
    explicit_update = explicit.update_loose(_position_velocity_fix())

    assert plain_update.nis == explicit_update.nis
    plain_counts = (
        plain_update.rows,
        plain_update.accepted_rows,
        plain_update.rejected_rows,
    )
    assert plain_counts == (6, 6, 0)
    assert (explicit_update.rows, explicit_update.accepted_rows) == (6, 6)
    golden = _fusion_golden("loose_default")
    _assert_update_matches(plain_update, golden["update"])
    assert _array_bits(plain.state.nominal.position_ecef_m) == _golden_bits(
        golden["state"]["position_ecef_m"]
    )
    assert _array_bits(plain.state.nominal.velocity_ecef_mps) == _golden_bits(
        golden["state"]["velocity_ecef_mps"]
    )
    assert _array_bits(np.diag(plain.state.covariance)) == _golden_bits(
        golden["state"]["covariance_diagonal"]
    )
    assert _array_bits(explicit.state.nominal.position_ecef_m) == _array_bits(
        plain.state.nominal.position_ecef_m
    )
    assert _array_bits(np.diag(explicit.state.covariance)[:6]) == _array_bits(
        np.diag(plain.state.covariance)[:6]
    )


def test_stationary_zupt_zaru_update_matches_core_bits():
    loose = sidereon.LooseCouplingConfig(
        stationary_updates=sidereon.StationaryUpdateConfig(
            sidereon.StationaryDetectorConfig(1, 100.0, 1.0),
            0.5,
            0.05,
        )
    )
    assert loose.stationary_updates.detector.window_len == 1
    assert loose.stationary_updates.zero_velocity_sigma_mps == 0.5
    assert loose.stationary_updates.zero_angular_rate_sigma_rps == 0.05

    filter_ = sidereon.InertialFilter.with_config(
        _state(), sidereon.InertialFilterConfig(_spec(), loose=loose)
    )
    filter_.propagate(
        sidereon.ImuSample.increment(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0)
    )

    update = filter_.update_stationary()

    assert update is not None
    assert (update.rows, update.accepted_rows, update.rejected_rows) == (6, 6, 0)
    golden = _fusion_golden("stationary")
    _assert_update_matches(update, golden["update"])
    assert _array_bits(filter_.state.nominal.position_ecef_m) == _golden_bits(
        golden["state"]["position_ecef_m"]
    )
    assert _array_bits(filter_.state.nominal.velocity_ecef_mps) == _golden_bits(
        golden["state"]["velocity_ecef_mps"]
    )
    assert _array_bits(filter_.state.nominal.gyro_bias_rps) == _golden_bits(
        golden["state"]["gyro_bias_rps"]
    )
    assert _array_bits(np.diag(filter_.state.covariance)) == _golden_bits(
        golden["state"]["covariance_diagonal"]
    )
    assert (
        sidereon.InertialFilter.with_config(
            _state(), sidereon.InertialFilterConfig(_spec())
        ).update_stationary()
        is None
    )


def test_fix_status_weighting_covariance_ordering_matches_core_bits():
    weighting = sidereon.GnssFixStatusWeighting(3.0, 2.0, 1.0)
    loose = sidereon.LooseCouplingConfig(fix_status_weighting=weighting)
    results = {}

    for status in (
        sidereon.GnssFixStatus.SINGLE,
        sidereon.GnssFixStatus.FLOAT,
        sidereon.GnssFixStatus.FIXED,
    ):
        filter_ = sidereon.InertialFilter.with_config(
            _state(), sidereon.InertialFilterConfig(_spec(), loose=loose)
        )
        update = filter_.update_loose(_position_velocity_fix(fix_status=status))
        results[status.label] = (update, filter_)
        assert update.applied is True
        assert update.rows == 6

    golden = _fusion_golden("fix_status_weighting")
    for label, (update, filter_) in results.items():
        _assert_update_matches(update, golden[label]["update"])
        assert _array_bits(filter_.state.nominal.position_ecef_m) == _golden_bits(
            golden[label]["state"]["position_ecef_m"]
        )
        assert _array_bits(np.diag(filter_.state.covariance)) == _golden_bits(
            golden[label]["state"]["covariance_diagonal"]
        )
    fixed_x = results["fixed"][1].state.covariance[0, 0]
    float_x = results["float"][1].state.covariance[0, 0]
    single_x = results["single"][1].state.covariance[0, 0]
    assert fixed_x < float_x < single_x
    assert (
        _position_velocity_fix()
        .with_fix_status(sidereon.GnssFixStatus.FLOAT)
        .fix_status
        == sidereon.GnssFixStatus.FLOAT
    )


def test_velocity_matching_and_imu_to_body_dcm_surface_match_core_bits():
    states = [
        sidereon.VelocityMatchState(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        sidereon.VelocityMatchState(1.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        sidereon.VelocityMatchState(2.0, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
    ]
    first_good = sidereon.GnssFixMeasurement.position_velocity(
        2.0,
        [4.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        np.eye(6, dtype=np.float64),
        8,
    )
    matched = sidereon.velocity_match_outage(
        states, first_good, sidereon.VelocityMatchingConfig(5.0)
    )

    assert _array_bits(matched.endpoint_position_correction_ecef_m) == [
        0x4000000000000000,
        0x3FF0000000000000,
        0x0000000000000000,
    ]
    assert _array_bits(matched.endpoint_velocity_correction_ecef_mps) == [
        0x3FF0000000000000,
        0x0000000000000000,
        0x0000000000000000,
    ]
    assert _array_bits(matched.states[1].position_ecef_m) == [
        0x3FFC000000000000,
        0x3FE0000000000000,
        0x0000000000000000,
    ]
    assert _array_bits(matched.states[1].velocity_ecef_mps) == [
        0x4002000000000000,
        0x3FE8000000000000,
        0x0000000000000000,
    ]

    imu_to_body = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    config = sidereon.InertialFilterConfig(_spec(), imu_to_body_dcm=imu_to_body)
    assert _array_bits(config.imu_to_body_dcm) == [
        0x0000000000000000,
        0xBFF0000000000000,
        0x0000000000000000,
        0x3FF0000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x0000000000000000,
        0x3FF0000000000000,
    ]
