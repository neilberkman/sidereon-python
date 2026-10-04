"""Public fusion errors preserve typed core payloads on ``ValueError``."""

import numpy as np
import pytest
import sidereon

WGS84_A_M = 6_378_137.0


def _state(layout=sidereon.ErrorStateLayout.FIFTEEN, diagonal=None):
    if diagonal is None:
        diagonal = [1.0] * layout.dimension
    nominal = sidereon.NavState(0.0, [WGS84_A_M, 0.0, 0.0], [0.0, 0.0, 0.0])
    return sidereon.InsFilterState.from_diagonal(nominal, layout, diagonal)


def _filter(layout=sidereon.ErrorStateLayout.FIFTEEN):
    return sidereon.InertialFilter(_state(layout), sidereon.ImuSpec.mems())


def _assert_detail(error, family, kind, fields):
    assert isinstance(error, ValueError)
    assert error.detail["family"] == family
    assert error.detail["kind"] == kind
    for name, value in fields.items():
        assert error.detail[name] == value


def _fnv1a64(data):
    value = 0xCBF29CE484222325
    for byte in data:
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


def _with_checksum(data):
    return data + _fnv1a64(data).to_bytes(8, "little")


def test_fusion_configuration_state_and_propagation_errors_keep_core_fields():
    with pytest.raises(ValueError) as history_error:
        sidereon.TimeSyncHistoryConfig(imu_capacity=0)
    _assert_detail(
        history_error.value,
        "fusion",
        "invalid_input",
        {"field": "imu_capacity", "reason": "must be positive"},
    )

    nominal = sidereon.NavState(0.0, [WGS84_A_M, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError) as dimension_error:
        sidereon.InsFilterState.from_diagonal(
            nominal, sidereon.ErrorStateLayout.FIFTEEN, [1.0] * 14
        )
    _assert_detail(
        dimension_error.value,
        "fusion",
        "dimension_mismatch",
        {"field": "covariance_diagonal", "expected": 15, "actual": 14},
    )
    assert str(dimension_error.value) == (
        "invalid fusion dimension covariance_diagonal: expected 15, got 14"
    )

    filter_ = _filter()
    checkpoint = filter_.encode_state()
    with pytest.raises(ValueError) as propagation_error:
        filter_.propagate(
            sidereon.ImuSample.increment(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0)
        )
    _assert_detail(
        propagation_error.value,
        "fusion",
        "invalid_input",
        {"field": "imu_samples", "reason": "must be strictly ordered by epoch"},
    )
    assert filter_.encode_state() == checkpoint

    copied = _filter()
    copied.restore_snapshot(filter_.snapshot())
    assert copied.encode_state() == checkpoint

    measurement = sidereon.GnssFixMeasurement.position_velocity(
        0.0,
        [WGS84_A_M + 1.0, 2.0, -3.0],
        [0.4, -0.2, 0.1],
        np.eye(6, dtype=np.float64),
        8,
    )
    filter_.update_loose(measurement)
    after_update = filter_.encode_state()
    with pytest.raises(ValueError) as update_error:
        filter_.update_loose(measurement)
    _assert_detail(
        update_error.value,
        "fusion",
        "invalid_input",
        {
            "field": "t_j2000_s",
            "reason": "GNSS measurement epochs must be strictly increasing",
        },
    )
    assert filter_.encode_state() == after_update


def test_fusion_state_codec_errors_keep_wire_fields_and_valid_roundtrip():
    source = _filter()
    config = sidereon.InertialFilterConfig(sidereon.ImuSpec.mems())
    encoded = source.encode_state()
    restored = sidereon.InertialFilter.from_encoded_state(encoded, config)
    assert restored.encode_state() == encoded

    with pytest.raises(ValueError) as truncated_error:
        sidereon.InertialFilter.from_encoded_state(b"", config)
    _assert_detail(
        truncated_error.value,
        "fusion_state_codec",
        "truncated",
        {"offset": 0, "needed": 18, "actual": 0},
    )
    assert str(truncated_error.value) == (
        "fusion state payload truncated at 0, needed 18 bytes, got 0"
    )

    invalid_magic = bytearray(encoded)
    invalid_magic[0] ^= 0xFF
    with pytest.raises(ValueError) as magic_error:
        sidereon.InertialFilter.from_encoded_state(invalid_magic, config)
    _assert_detail(
        magic_error.value,
        "fusion_state_codec",
        "invalid_magic",
        {},
    )

    unsupported_version = bytearray(encoded[:-8])
    unsupported_version[8:10] = (0).to_bytes(2, "little")
    with pytest.raises(ValueError) as version_error:
        sidereon.InertialFilter.from_encoded_state(
            _with_checksum(unsupported_version), config
        )
    _assert_detail(
        version_error.value,
        "fusion_state_codec",
        "unsupported_version",
        {"version": 0},
    )

    trailing_data = _with_checksum(encoded[:-8] + b"\x00")
    with pytest.raises(ValueError) as trailing_error:
        sidereon.InertialFilter.from_encoded_state(trailing_data, config)
    _assert_detail(
        trailing_error.value,
        "fusion_state_codec",
        "trailing_bytes",
        {"remaining": 1},
    )

    corrupt = bytearray(encoded)
    corrupt[-1] ^= 0x01
    with pytest.raises(ValueError) as checksum_error:
        source.restore_encoded_state(corrupt)
    _assert_detail(
        checksum_error.value,
        "fusion_state_codec",
        "checksum",
        {
            "expected": int.from_bytes(encoded[-8:], "little") ^ (1 << 56),
            "found": _fnv1a64(encoded[:-8]),
        },
    )
    assert source.encode_state() == encoded


def test_core_covariance_refusal_has_typed_payload_and_acceptance_control():
    nominal = sidereon.NavState(0.0, [WGS84_A_M, 0.0, 0.0], [0.0, 0.0, 0.0])
    covariance = np.eye(15, dtype=np.float64)
    covariance[0, 0] = 1.0
    covariance[0, 1] = 2.0
    covariance[1, 0] = 2.0
    covariance[1, 1] = 1.0
    with pytest.raises(ValueError) as covariance_error:
        sidereon.InsFilterState.from_covariance(
            nominal, sidereon.ErrorStateLayout.FIFTEEN, covariance
        )
    _assert_detail(
        covariance_error.value,
        "fusion",
        "non_positive_semidefinite",
        {"field": "covariance"},
    )

    accepted = sidereon.InsFilterState.from_covariance(
        nominal, sidereon.ErrorStateLayout.FIFTEEN, np.eye(15, dtype=np.float64)
    )
    assert accepted.dimension == 15
