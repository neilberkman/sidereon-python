"""Exact public OEM DTO coverage from the pinned core fixture."""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES

OEM_KVN = os.path.join(CORE_FIXTURES, "oem", "gps.kvn")


def _read_fixture():
    with open(OEM_KVN, encoding="utf-8") as stream:
        return stream.read()


def test_public_kvn_route_retains_every_oem_field_and_comment():
    parsed = sidereon.parse_oem_kvn(_read_fixture())
    assert parsed.ccsds_oem_vers == "2.0"
    assert parsed.comments == ["Annotated OEM fixture for a GPS navigation spacecraft."]
    assert parsed.classification is None
    assert parsed.creation_date == "2026-06-28T12:00:00.000"
    assert parsed.originator == "SIDEREON TEST"
    assert parsed.message_id is None
    assert parsed.skipped_states == []
    assert len(parsed.segments) == 1

    segment = parsed.segments[0]
    metadata = segment.metadata
    assert metadata.comments == []
    assert metadata.object_name == "GPS BIIRM-8"
    assert metadata.object_id == "2005-038A"
    assert metadata.center_name == "EARTH"
    assert metadata.ref_frame == "EME2000"
    assert metadata.ref_frame_epoch is None
    assert metadata.time_system == "GPS"
    assert metadata.start_time == "2026-06-28T00:00:00.000"
    assert metadata.stop_time == "2026-06-28T00:30:00.000"
    assert metadata.useable_start_time == "2026-06-28T00:00:00.000"
    assert metadata.useable_stop_time == "2026-06-28T00:30:00.000"
    assert metadata.interpolation == "LAGRANGE"
    assert metadata.interpolation_degree == 5

    assert segment.data_comments == [
        (0, "Epoch X Y Z X_DOT Y_DOT Z_DOT with one acceleration-bearing sample.")
    ]
    assert segment.covariance_comments == []
    assert [state.epoch for state in segment.states] == [
        "2026-06-28T00:00:00.000",
        "2026-06-28T00:15:00.000",
        "2026-06-28T00:30:00.000",
    ]
    np.testing.assert_array_equal(
        segment.states[0].position_km,
        [15600.123456, -21000.654321, 20100.111111],
    )
    np.testing.assert_array_equal(
        segment.states[0].velocity_km_s, [2.102345, 1.305678, -2.987654]
    )
    assert segment.states[0].acceleration_km_s2 is None
    np.testing.assert_array_equal(
        segment.states[1].position_km,
        [17450.223456, -19750.754321, 17210.211111],
    )
    np.testing.assert_array_equal(
        segment.states[1].velocity_km_s, [2.008765, 1.504321, -3.112345]
    )
    assert segment.states[1].acceleration_km_s2 is None
    np.testing.assert_array_equal(
        segment.states[2].position_km,
        [19200.323456, -18200.854321, 14200.311111],
    )
    np.testing.assert_array_equal(
        segment.states[2].velocity_km_s, [1.812345, 1.701234, -3.201234]
    )
    np.testing.assert_array_equal(
        segment.states[2].acceleration_km_s2, [0.000001, -0.000002, 0.000003]
    )

    assert len(segment.covariances) == 1
    covariance = segment.covariances[0]
    assert covariance.epoch == "2026-06-28T00:15:00.000"
    assert covariance.cov_ref_frame == "RTN"
    np.testing.assert_array_equal(
        covariance.lower_triangle,
        [
            0.0001,
            0.0,
            0.0002,
            0.0,
            0.0,
            0.0003,
            0.0,
            0.0,
            0.0,
            0.00000001,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00000002,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00000003,
        ],
    )

    edited = sidereon.Oem(
        segments=[
            sidereon.OemSegment(
                metadata,
                segment.states,
                covariances=segment.covariances,
                data_comments=[(1, "between state vectors")],
                covariance_comments=[(0, "covariance data")],
            )
        ],
        originator=parsed.originator,
        comments=parsed.comments,
        ccsds_oem_vers=parsed.ccsds_oem_vers,
        creation_date=parsed.creation_date,
        classification="U",
        message_id="OEM-TEST-1",
    )
    for encode, parse in (
        (sidereon.Oem.to_kvn_string, sidereon.parse_oem_kvn),
        (sidereon.Oem.to_xml_string, sidereon.parse_oem_xml),
    ):
        encoded = encode(edited)
        recovered = parse(encoded)
        assert recovered == edited
        assert recovered.segments[0].data_comments == [(1, "between state vectors")]
        assert recovered.segments[0].covariance_comments == [(0, "covariance data")]


def test_public_kvn_route_retains_each_skipped_state_reason():
    fixture = _read_fixture()
    wrong_count = fixture.replace(
        "2026-06-28T00:15:00.000 17450.223456",
        "2026-06-28T00:15:00.000 17450.223456 extra",
        1,
    )
    count_error = sidereon.parse_oem_kvn(wrong_count).skipped_states
    assert len(count_error) == 1
    assert count_error[0].line == 22
    assert count_error[0].segment == 0
    assert count_error[0].text == (
        "2026-06-28T00:15:00.000 17450.223456 extra -19750.754321 "
        "17210.211111 2.008765 1.504321 -3.112345"
    )
    assert count_error[0].reason == "item_count"
    assert count_error[0].item_count == 8
    assert count_error[0].field is None
    assert count_error[0].kind is None

    bad_number = fixture.replace(
        "2026-06-28T00:00:00.000 15600.123456",
        "2026-06-28T00:00:00.000 invalid",
        1,
    )
    field_error = sidereon.parse_oem_kvn(bad_number).skipped_states
    assert len(field_error) == 1
    assert field_error[0].line == 21
    assert field_error[0].segment == 0
    assert field_error[0].text == (
        "2026-06-28T00:00:00.000 invalid -21000.654321 20100.111111 "
        "2.102345 1.305678 -2.987654"
    )
    assert field_error[0].reason == "invalid_field"
    assert field_error[0].item_count is None
    assert field_error[0].field == "X"
    assert field_error[0].kind == "FloatParse"
