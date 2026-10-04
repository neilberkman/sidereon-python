"""Exact public coverage for CDM message-level metadata and optionals."""

import sidereon
from test_cdm import KVN_PATH, XML_PATH, _load


def _assert_fixture_metadata(cdm):
    assert cdm.ccsds_cdm_vers == "1.0"
    assert cdm.comments == []
    assert cdm.creation_date == "2010-03-12T22:31:12.000"
    assert cdm.originator == "JSPOC"
    assert cdm.message_for == "SATELLITE A"
    assert cdm.message_id == "201113719185"
    assert cdm.relative_comments == ["Relative Metadata/Data"]
    assert cdm.tca == "2010-03-13T22:37:52.618"
    assert cdm.miss_distance_m == 715.0
    assert cdm.relative_speed_m_s == 14762.0
    assert cdm.relative_position_rtn_m == (27.4, -70.2, 711.8)
    assert cdm.relative_velocity_rtn_m_s == (-7.2, -14692.0, -1437.2)
    assert cdm.start_screen_period == "2010-03-12T18:29:32.212"
    assert cdm.stop_screen_period == "2010-03-15T18:29:32.212"
    assert cdm.screen_volume_frame == "RTN"
    assert cdm.screen_volume_shape == "ELLIPSOID"
    assert cdm.screen_volume_m == (200.0, 1000.0, 1000.0)
    assert cdm.screen_entry_time == "2010-03-13T22:37:52.222"
    assert cdm.screen_exit_time == "2010-03-13T22:37:52.824"
    assert cdm.collision_probability == 4.835e-5
    assert cdm.collision_probability_method == "FOSTER-1992"
    assert cdm.hard_body_radius_m is None


def test_public_kvn_and_xml_readers_retain_exact_message_fields_and_times():
    kvn = sidereon.parse_cdm_kvn(_load(KVN_PATH))
    xml_fixture = sidereon.parse_cdm_xml(_load(XML_PATH))
    _assert_fixture_metadata(kvn)

    # The committed XML fixture omits screening-period/volume fields. Confirm
    # those stay absent while common message metadata keeps its exact value.
    assert xml_fixture.ccsds_cdm_vers == "1.0"
    assert xml_fixture.creation_date == "2010-03-12T22:31:12.000"
    assert xml_fixture.message_id == "201113719185"
    assert xml_fixture.tca == "2010-03-13T22:37:52.618"
    assert xml_fixture.relative_comments == ["Relative Metadata/Data"]
    assert xml_fixture.start_screen_period is None
    assert xml_fixture.stop_screen_period is None
    assert xml_fixture.screen_volume_frame is None
    assert xml_fixture.screen_volume_shape is None
    assert xml_fixture.screen_volume_m == (None, None, None)
    assert xml_fixture.screen_entry_time is None
    assert xml_fixture.screen_exit_time is None

    # The KVN parser result carries those fields; the public XML writer and
    # parser must preserve them too.
    xml_roundtrip = sidereon.parse_cdm_xml(kvn.to_xml_string())
    _assert_fixture_metadata(xml_roundtrip)
    assert xml_roundtrip == kvn


def test_absent_message_optionals_and_componentwise_zeros_round_trip():
    parsed = sidereon.parse_cdm_kvn(_load(KVN_PATH))
    cdm = sidereon.Cdm(
        parsed.object1,
        parsed.object2,
        message_id=parsed.message_id,
        relative_position_rtn_m=(0.0, None, 0.0),
        relative_velocity_rtn_m_s=(None, 0.0, None),
        screen_volume_m=(0.0, None, 0.0),
    )

    for encode, parse in (
        (sidereon.Cdm.to_kvn_string, sidereon.parse_cdm_kvn),
        (sidereon.Cdm.to_xml_string, sidereon.parse_cdm_xml),
    ):
        reparsed = parse(encode(cdm))
        assert reparsed == cdm
        assert reparsed.ccsds_cdm_vers is None
        assert reparsed.comments == []
        assert reparsed.creation_date is None
        assert reparsed.originator is None
        assert reparsed.message_for is None
        assert reparsed.message_id == parsed.message_id
        assert reparsed.relative_comments == []
        assert reparsed.tca is None
        assert reparsed.miss_distance_m is None
        assert reparsed.relative_speed_m_s is None
        assert reparsed.relative_position_rtn_m == (0.0, None, 0.0)
        assert reparsed.relative_velocity_rtn_m_s == (None, 0.0, None)
        assert reparsed.start_screen_period is None
        assert reparsed.stop_screen_period is None
        assert reparsed.screen_volume_frame is None
        assert reparsed.screen_volume_shape is None
        assert reparsed.screen_volume_m == (0.0, None, 0.0)
        assert reparsed.screen_entry_time is None
        assert reparsed.screen_exit_time is None
        assert reparsed.collision_probability is None
        assert reparsed.collision_probability_method is None
        assert reparsed.hard_body_radius_m is None
