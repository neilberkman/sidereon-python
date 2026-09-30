"""OMM binding reproduces core KVN/XML/JSON parse and encode outputs.

The fixture ``omm.json`` is emitted by
``SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test omm_python_fixture``.
It covers the committed CelesTrak OMM KVN/XML/JSON files for near-Earth and
deep-space objects.
"""

import json
import os
from datetime import datetime, timezone

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES, hex_to_f64


def _fixture():
    with open(os.path.join(FIXTURES, "omm.json")) as fh:
        return json.load(fh)


FX = _fixture()


def _load(path):
    with open(path) as fh:
        return fh.read()


def _path(rel):
    return os.path.join(CORE_FIXTURES, "omm", os.path.basename(rel))


def _assert_epoch(epoch, ref):
    assert epoch.year == ref["year"]
    assert epoch.month == ref["month"]
    assert epoch.day == ref["day"]
    assert epoch.hour == ref["hour"]
    assert epoch.minute == ref["minute"]
    assert epoch.second == ref["second"]
    assert epoch.microsecond == ref["microsecond"]
    assert epoch.iso8601 == ref["iso8601"]
    assert epoch == sidereon.OmmEpoch(
        ref["year"],
        ref["month"],
        ref["day"],
        ref["hour"],
        ref["minute"],
        ref["second"],
        ref["microsecond"],
    )
    assert "OmmEpoch" in repr(epoch)


def _assert_omm(omm, ref):
    assert omm.ccsds_omm_vers == ref["ccsds_omm_vers"]
    assert omm.creation_date == ref["creation_date"]
    assert omm.originator == ref["originator"]
    assert omm.object_name == ref["object_name"]
    assert omm.object_id == ref["object_id"]
    assert omm.center_name == ref["center_name"]
    assert omm.ref_frame == ref["ref_frame"]
    assert omm.time_system == ref["time_system"]
    assert omm.mean_element_theory == ref["mean_element_theory"]
    _assert_epoch(omm.epoch, ref["epoch"])
    assert omm.mean_motion == hex_to_f64(ref["mean_motion_hex"])
    assert omm.eccentricity == hex_to_f64(ref["eccentricity_hex"])
    assert omm.inclination_deg == hex_to_f64(ref["inclination_deg_hex"])
    assert omm.ra_of_asc_node_deg == hex_to_f64(ref["ra_of_asc_node_deg_hex"])
    assert omm.arg_of_pericenter_deg == hex_to_f64(ref["arg_of_pericenter_deg_hex"])
    assert omm.mean_anomaly_deg == hex_to_f64(ref["mean_anomaly_deg_hex"])
    assert omm.ephemeris_type == ref["ephemeris_type"]
    assert omm.classification_type == ref["classification_type"]
    assert omm.norad_cat_id == ref["norad_cat_id"]
    assert omm.element_set_no == ref["element_set_no"]
    assert omm.rev_at_epoch == ref["rev_at_epoch"]
    assert omm.bstar == hex_to_f64(ref["bstar_hex"])
    assert omm.mean_motion_dot == hex_to_f64(ref["mean_motion_dot_hex"])
    assert omm.mean_motion_ddot == hex_to_f64(ref["mean_motion_ddot_hex"])
    assert "Omm" in repr(omm)


def test_parse_omm_kvn_xml_json_match_reference_fields_and_encode():
    for fixture in FX["fixtures"]:
        kvn = sidereon.parse_omm_kvn(_load(_path(fixture["kvn_fixture"])))
        xml = sidereon.parse_omm_xml(_load(_path(fixture["xml_fixture"])))
        json_omm = sidereon.parse_omm_json(_load(_path(fixture["json_fixture"])))

        _assert_omm(kvn, fixture["from_kvn"])
        _assert_omm(xml, fixture["from_xml"])
        _assert_omm(json_omm, fixture["from_json"])

        assert kvn.to_kvn_string() == fixture["encoded_kvn"]
        assert xml.to_xml_string() == fixture["encoded_xml"]
        assert json.loads(json_omm.to_json_string()) == json.loads(
            fixture["encoded_json"]
        )

        assert sidereon.parse_omm_kvn(kvn.to_kvn_string()) == kvn
        assert sidereon.parse_omm_xml(xml.to_xml_string()) == xml
        assert sidereon.parse_omm_json(json_omm.to_json_string()) == json_omm


def test_parse_omm_epoch_preserves_fraction_and_utc_leap_second():
    epoch = sidereon.parse_omm_epoch("2026-06-17T04:32:52.099296123456789")
    assert epoch.microsecond == 99_296
    assert epoch.femtosecond == 123_456_789
    assert epoch.iso8601 == "2026-06-17T04:32:52.099296123456789"
    round_trip = sidereon.parse_omm_epoch(epoch.iso8601)
    assert (
        round_trip.year,
        round_trip.month,
        round_trip.day,
        round_trip.hour,
        round_trip.minute,
        round_trip.second,
        round_trip.microsecond,
        round_trip.femtosecond,
    ) == (
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second,
        epoch.microsecond,
        epoch.femtosecond,
    )

    ordinary = sidereon.parse_omm_epoch("2026-06-17T04:32:52.099296")
    assert ordinary.iso8601 == "2026-06-17T04:32:52.099296"

    leap = sidereon.parse_omm_epoch("2016-12-31T23:59:60.000000Z")
    assert leap.second == 60
    assert leap.iso8601 == "2016-12-31T23:59:60.000000"


def test_parse_omm_epoch_invalid_text_keeps_typed_core_detail():
    with pytest.raises(sidereon.OmmParseError) as raised:
        sidereon.parse_omm_epoch("not-an-epoch")
    assert raised.value.detail == {
        "family": "omm",
        "kind": "Epoch",
        "message": 'invalid seconds in "not-an-epoch"',
    }

    with pytest.raises(sidereon.OmmParseError) as raised:
        sidereon.parse_omm_epoch("2016-12-30T23:59:60.000000Z")
    assert raised.value.detail == {
        "family": "omm",
        "kind": "InvalidField",
        "field": "civil datetime",
        "validation_kind": "InvalidCivilTime",
    }


def test_omm_encodings_share_orbital_content():
    for fixture in FX["fixtures"]:
        kvn = sidereon.parse_omm_kvn(_load(_path(fixture["kvn_fixture"])))
        xml = sidereon.parse_omm_xml(_load(_path(fixture["xml_fixture"])))
        json_omm = sidereon.parse_omm_json(_load(_path(fixture["json_fixture"])))
        for other in (xml, json_omm):
            assert other.epoch == kvn.epoch
            assert other.norad_cat_id == kvn.norad_cat_id
            assert np.float64(other.mean_motion).view(np.uint64) == np.float64(
                kvn.mean_motion
            ).view(np.uint64)
            assert np.float64(other.eccentricity).view(np.uint64) == np.float64(
                kvn.eccentricity
            ).view(np.uint64)
            assert np.float64(other.bstar).view(np.uint64) == np.float64(
                kvn.bstar
            ).view(np.uint64)


def test_constructed_omm_value_matches_parsed_kvn():
    ref = FX["fixtures"][0]["from_kvn"]
    e = ref["epoch"]
    epoch = sidereon.OmmEpoch(
        e["year"],
        e["month"],
        e["day"],
        e["hour"],
        e["minute"],
        e["second"],
        e["microsecond"],
    )
    omm = sidereon.Omm(
        epoch,
        hex_to_f64(ref["mean_motion_hex"]),
        hex_to_f64(ref["eccentricity_hex"]),
        hex_to_f64(ref["inclination_deg_hex"]),
        hex_to_f64(ref["ra_of_asc_node_deg_hex"]),
        hex_to_f64(ref["arg_of_pericenter_deg_hex"]),
        hex_to_f64(ref["mean_anomaly_deg_hex"]),
        ref["norad_cat_id"],
        ccsds_omm_vers=ref["ccsds_omm_vers"],
        creation_date=ref["creation_date"],
        originator=ref["originator"],
        object_name=ref["object_name"],
        object_id=ref["object_id"],
        center_name=ref["center_name"],
        ref_frame=ref["ref_frame"],
        time_system=ref["time_system"],
        mean_element_theory=ref["mean_element_theory"],
        ephemeris_type=ref["ephemeris_type"],
        classification_type=ref["classification_type"],
        element_set_no=ref["element_set_no"],
        rev_at_epoch=ref["rev_at_epoch"],
        bstar=hex_to_f64(ref["bstar_hex"]),
        mean_motion_dot=hex_to_f64(ref["mean_motion_dot_hex"]),
        mean_motion_ddot=hex_to_f64(ref["mean_motion_ddot_hex"]),
    )
    parsed = sidereon.parse_omm_kvn(_load(_path(FX["fixtures"][0]["kvn_fixture"])))
    assert omm == parsed
    assert omm.to_kvn_string() == FX["fixtures"][0]["encoded_kvn"]


def test_omm_satellite_propagation_matches_improved_tle():
    omm = sidereon.parse_omm_json(_load(_path("25544.json")))
    line1, line2 = _load(_path("25544.tle")).splitlines()[-2:]
    tle = sidereon.Tle(line1, line2, sidereon.OpsMode.IMPROVED)
    satellite = omm.to_satellite()
    with pytest.raises(TypeError):
        sidereon.OmmSatellite()
    epoch = datetime(
        omm.epoch.year,
        omm.epoch.month,
        omm.epoch.day,
        omm.epoch.hour,
        omm.epoch.minute,
        omm.epoch.second,
        omm.epoch.microsecond,
        tzinfo=timezone.utc,
    )
    epoch_us = int(epoch.timestamp() * 1_000_000)
    epochs = np.asarray(
        [epoch_us, epoch_us + 60_000_000, epoch_us + 300_000_000], dtype=np.int64
    )

    omm_states = satellite.propagate(epochs)
    tle_states = tle.propagate(epochs)
    assert isinstance(satellite, sidereon.OmmSatellite)
    assert isinstance(omm_states, sidereon.TlePropagation)
    assert omm_states.position_km.dtype == np.float64
    assert omm_states.velocity_km_s.dtype == np.float64
    assert omm_states.position_km.shape == (3, 3)
    assert omm_states.velocity_km_s.shape == (3, 3)
    max_position_difference = float(
        np.max(np.abs(omm_states.position_km - tle_states.position_km))
    )
    max_velocity_difference = float(
        np.max(np.abs(omm_states.velocity_km_s - tle_states.velocity_km_s))
    )
    np.testing.assert_allclose(max_position_difference, 0.0, rtol=0, atol=2e-7)
    np.testing.assert_allclose(max_velocity_difference, 0.0, rtol=0, atol=2e-10)
    repeated = satellite.propagate(epochs)
    assert omm_states.position_km.tobytes() == repeated.position_km.tobytes()
    assert omm_states.velocity_km_s.tobytes() == repeated.velocity_km_s.tobytes()

    empty = satellite.propagate(np.asarray([], dtype=np.int64))
    assert len(empty) == 0
    assert empty.position_km.shape == (0, 3)
    assert empty.velocity_km_s.shape == (0, 3)


@pytest.mark.parametrize(
    ("changes", "kind", "field"),
    [
        (
            {"center_name": "MARS"},
            "IncompatibleMetadata",
            "CENTER_NAME",
        ),
        ({"mean_motion": None}, "MissingField", "MEAN_MOTION"),
        ({"bstar": None}, "MissingField", "BSTAR"),
    ],
)
def test_omm_satellite_rejects_incompatible_or_incomplete_input(changes, kind, field):
    source = sidereon.parse_omm_json(_load(_path("25544.json")))
    values = {
        "epoch": source.epoch,
        "mean_motion": source.mean_motion,
        "eccentricity": source.eccentricity,
        "inclination_deg": source.inclination_deg,
        "ra_of_asc_node_deg": source.ra_of_asc_node_deg,
        "arg_of_pericenter_deg": source.arg_of_pericenter_deg,
        "mean_anomaly_deg": source.mean_anomaly_deg,
        "norad_cat_id": source.norad_cat_id,
        "mean_element_theory": source.mean_element_theory,
        "bstar": source.bstar,
    }
    values.update(changes)
    omm = sidereon.Omm(**values)

    with pytest.raises(sidereon.OmmParseError) as raised:
        omm.to_satellite()
    assert raised.value.detail == {
        "family": "omm",
        "kind": kind,
        "field": field,
        **({"value": "MARS"} if kind == "IncompatibleMetadata" else {}),
    }


def test_omm_parse_and_constructor_errors_are_typed():
    with pytest.raises(sidereon.OmmParseError):
        sidereon.parse_omm_kvn("CCSDS_OMM_VERS = 2.0\n")
    with pytest.raises(sidereon.ParseError):
        sidereon.parse_omm_xml("<not xml")
    with pytest.raises(sidereon.OmmParseError):
        sidereon.parse_omm_json("{}")
    with pytest.raises(ValueError):
        sidereon.OmmEpoch(2026, 13, 1, 0, 0, 0, 0)
    with pytest.raises(ValueError):
        sidereon.Omm(
            sidereon.OmmEpoch(2026, 1, 1, 0, 0, 0, 0),
            float("nan"),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1,
        )
