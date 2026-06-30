"""Area 7 CDM binding reproduces the engine parse/encode surface.

The fixture ``cdm.json`` is emitted by
``SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test cdm_python_fixture``.
It points at the committed CCSDS example KVN/XML files and records the core
parse results plus the core encoded KVN/XML text.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES, hex_to_f64


def _fixture():
    with open(os.path.join(FIXTURES, "cdm.json")) as fh:
        return json.load(fh)


FX = _fixture()
KVN_PATH = os.path.join(CORE_FIXTURES, "cdm", os.path.basename(FX["kvn_fixture"]))
XML_PATH = os.path.join(CORE_FIXTURES, "cdm", os.path.basename(FX["xml_fixture"]))


def _bits(arr):
    return np.asarray(arr, dtype=np.float64).view(np.uint64)


def _expect_bits(hex_list):
    return np.asarray([int(h, 16) for h in hex_list], dtype=np.uint64)


def _load(path):
    with open(path) as fh:
        return fh.read()


def _assert_object(obj, ref):
    assert obj.object_designator == ref["object_designator"]
    assert obj.catalog_name == ref["catalog_name"]
    assert obj.object_name == ref["object_name"]
    assert obj.international_designator == ref["international_designator"]
    assert obj.object_type == ref["object_type"]
    assert obj.ref_frame == ref["ref_frame"]
    assert np.array_equal(_bits(obj.position_km), _expect_bits(ref["position_km_hex"]))
    assert np.array_equal(
        _bits(obj.velocity_km_s), _expect_bits(ref["velocity_km_s_hex"])
    )
    assert np.array_equal(
        _bits(obj.covariance_rtn), _expect_bits(ref["covariance_rtn_hex"])
    )
    assert "CdmObject" in repr(obj)


def _assert_cdm(cdm, ref):
    assert cdm.creation_date == ref["creation_date"]
    assert cdm.originator == ref["originator"]
    assert cdm.message_id == ref["message_id"]
    assert cdm.tca == ref["tca"]
    assert cdm.miss_distance_m == hex_to_f64(ref["miss_distance_m_hex"])
    assert cdm.relative_speed_m_s == hex_to_f64(ref["relative_speed_m_s_hex"])
    assert cdm.collision_probability == hex_to_f64(ref["collision_probability_hex"])
    assert cdm.collision_probability_method == ref["collision_probability_method"]
    assert cdm.hard_body_radius_m is None
    _assert_object(cdm.object1, ref["object1"])
    _assert_object(cdm.object2, ref["object2"])
    assert "Cdm" in repr(cdm)


def _assert_roundtrip_content(reparsed, original):
    assert reparsed.creation_date == original.creation_date
    assert reparsed.originator == original.originator
    assert reparsed.message_id == original.message_id
    assert reparsed.tca == original.tca
    assert reparsed.miss_distance_m == original.miss_distance_m
    assert reparsed.relative_speed_m_s == original.relative_speed_m_s
    assert reparsed.collision_probability == original.collision_probability
    assert np.array_equal(
        _bits(reparsed.object1.position_km), _bits(original.object1.position_km)
    )
    assert np.array_equal(
        _bits(reparsed.object2.velocity_km_s), _bits(original.object2.velocity_km_s)
    )
    assert np.array_equal(
        _bits(reparsed.object1.covariance_rtn), _bits(original.object1.covariance_rtn)
    )
    assert np.array_equal(
        _bits(reparsed.object2.covariance_rtn), _bits(original.object2.covariance_rtn)
    )


def test_parse_cdm_kvn_matches_reference_fields_and_encode():
    cdm = sidereon.parse_cdm_kvn(_load(KVN_PATH))
    _assert_cdm(cdm, FX["from_kvn"])
    assert cdm.to_kvn_string() == FX["encoded_kvn"]
    _assert_roundtrip_content(sidereon.parse_cdm_kvn(cdm.to_kvn_string()), cdm)


def test_parse_cdm_xml_matches_reference_fields_and_encode():
    cdm = sidereon.parse_cdm_xml(_load(XML_PATH))
    _assert_cdm(cdm, FX["from_xml"])
    assert cdm.to_xml_string() == FX["encoded_xml"]
    _assert_roundtrip_content(sidereon.parse_cdm_xml(cdm.to_xml_string()), cdm)


def test_kvn_and_xml_messages_have_same_orbital_content():
    kvn = sidereon.parse_cdm_kvn(_load(KVN_PATH))
    xml = sidereon.parse_cdm_xml(_load(XML_PATH))
    assert np.array_equal(
        _bits(kvn.object1.position_km), _bits(xml.object1.position_km)
    )
    assert np.array_equal(
        _bits(kvn.object2.velocity_km_s), _bits(xml.object2.velocity_km_s)
    )
    assert np.array_equal(
        _bits(kvn.object1.covariance_rtn), _bits(xml.object1.covariance_rtn)
    )
    assert kvn.collision_probability == xml.collision_probability


def _rebuild_object(obj):
    """Reconstruct a CdmObject from another's full field set via the getters."""
    return sidereon.CdmObject(
        obj.position_km,
        obj.velocity_km_s,
        obj.covariance_rtn,
        object_designator=obj.object_designator,
        catalog_name=obj.catalog_name,
        object_name=obj.object_name,
        international_designator=obj.international_designator,
        object_type=obj.object_type,
        operator_contact_position=obj.operator_contact_position,
        operator_organization=obj.operator_organization,
        operator_phone=obj.operator_phone,
        operator_email=obj.operator_email,
        ephemeris_name=obj.ephemeris_name,
        covariance_method=obj.covariance_method,
        maneuverable=obj.maneuverable,
        orbit_center=obj.orbit_center,
        ref_frame=obj.ref_frame,
        gravity_model=obj.gravity_model,
        atmospheric_model=obj.atmospheric_model,
        n_body_perturbations=obj.n_body_perturbations,
        solar_rad_pressure=obj.solar_rad_pressure,
        earth_tides=obj.earth_tides,
        intrack_thrust=obj.intrack_thrust,
        velocity_covariance_rtn=obj.velocity_covariance_rtn,
    )


def test_constructed_cdm_value_encodes_like_parsed_kvn():
    parsed = sidereon.parse_cdm_kvn(_load(KVN_PATH))
    # Rebuild every object field from the parsed message through the public
    # getters, so a hand-constructed CDM is value-equal and encodes identically.
    cdm = sidereon.Cdm(
        _rebuild_object(parsed.object1),
        _rebuild_object(parsed.object2),
        creation_date=parsed.creation_date,
        originator=parsed.originator,
        message_id=parsed.message_id,
        tca=parsed.tca,
        miss_distance_m=parsed.miss_distance_m,
        relative_speed_m_s=parsed.relative_speed_m_s,
        collision_probability=parsed.collision_probability,
        collision_probability_method=parsed.collision_probability_method,
        hard_body_radius_m=parsed.hard_body_radius_m,
    )
    assert cdm == parsed
    assert cdm.to_kvn_string() == FX["encoded_kvn"]


def test_full_metadata_and_velocity_covariance_round_trip():
    velocity_covariance = np.arange(1.0, 16.0, dtype=np.float64) * 1e-3
    obj = sidereon.CdmObject(
        np.array([1.0, 2.0, 3.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 0.1, 2.0, 0.01, 0.02, 3.0]),
        object_designator="2020-001A",
        catalog_name="SATCAT",
        object_name="ALPHA",
        international_designator="2020-001A",
        object_type="PAYLOAD",
        operator_contact_position="FLIGHT DYNAMICS",
        operator_organization="ACME",
        operator_phone="+1-555-0100",
        operator_email="ops@example.test",
        ephemeris_name="EPHEM-1",
        covariance_method="CALCULATED",
        maneuverable="YES",
        orbit_center="EARTH",
        ref_frame="ITRF",
        gravity_model="EGM-96: 36D 36O",
        atmospheric_model="JACCHIA 70",
        n_body_perturbations="MOON,SUN",
        solar_rad_pressure="YES",
        earth_tides="YES",
        intrack_thrust="NO",
        velocity_covariance_rtn=velocity_covariance,
    )

    # The numpy getter returns the 15-element velocity block when present.
    assert np.array_equal(
        _bits(obj.velocity_covariance_rtn), _bits(velocity_covariance)
    )

    other = sidereon.CdmObject(
        np.array([4.0, 5.0, 6.0]),
        np.array([0.4, 0.5, 0.6]),
        np.array([2.0, 0.2, 4.0, 0.02, 0.04, 6.0]),
        object_name="BETA",
    )
    # Absent velocity covariance surfaces as None.
    assert other.velocity_covariance_rtn is None

    cdm = sidereon.Cdm(
        obj,
        other,
        creation_date="2020-01-01T00:00:00",
        originator="ACME",
        message_id="MSG-1",
        tca="2020-01-02T03:04:05.000000",
        miss_distance_m=123.0,
        relative_speed_m_s=14762.0,
        collision_probability=4.835e-5,
        collision_probability_method="FOSTER-1992",
    )

    reparsed = sidereon.parse_cdm_kvn(cdm.to_kvn_string()).object1
    assert reparsed.object_designator == "2020-001A"
    assert reparsed.operator_contact_position == "FLIGHT DYNAMICS"
    assert reparsed.operator_organization == "ACME"
    assert reparsed.operator_phone == "+1-555-0100"
    assert reparsed.operator_email == "ops@example.test"
    assert reparsed.ephemeris_name == "EPHEM-1"
    assert reparsed.covariance_method == "CALCULATED"
    assert reparsed.maneuverable == "YES"
    assert reparsed.orbit_center == "EARTH"
    assert reparsed.gravity_model == "EGM-96: 36D 36O"
    assert reparsed.atmospheric_model == "JACCHIA 70"
    assert reparsed.n_body_perturbations == "MOON,SUN"
    assert reparsed.solar_rad_pressure == "YES"
    assert reparsed.earth_tides == "YES"
    assert reparsed.intrack_thrust == "NO"
    assert np.array_equal(
        _bits(reparsed.velocity_covariance_rtn), _bits(velocity_covariance)
    )

    # The same content round-trips through the XML serializer.
    reparsed_xml = sidereon.parse_cdm_xml(cdm.to_xml_string()).object1
    assert reparsed_xml.operator_organization == "ACME"
    assert np.array_equal(
        _bits(reparsed_xml.velocity_covariance_rtn), _bits(velocity_covariance)
    )


def test_velocity_covariance_shape_is_validated():
    with pytest.raises(ValueError):
        sidereon.CdmObject(
            np.zeros(3),
            np.zeros(3),
            np.zeros(6),
            velocity_covariance_rtn=np.zeros(14),
        )


def test_cdm_parse_and_shape_errors_are_typed():
    with pytest.raises(sidereon.CdmParseError):
        sidereon.parse_cdm_kvn("OBJECT = OBJECT1\nX = 1.0 [km]\n")
    with pytest.raises(sidereon.ParseError):
        sidereon.parse_cdm_xml("<segment></segment><segment></segment>")
    with pytest.raises(ValueError):
        sidereon.CdmObject(np.zeros(2), np.zeros(3), np.zeros(6))
    with pytest.raises(ValueError):
        sidereon.CdmObject(np.zeros(3), np.zeros(3), np.zeros(5))
