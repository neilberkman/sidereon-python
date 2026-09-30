"""Typed detail payloads for SPK and CCSDS codec errors."""

import json
import os
import struct

import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES


def _read(path):
    with open(path, encoding="utf-8") as source:
        return source.read()


def test_spk_parse_and_query_errors_keep_core_fields_and_legacy_catches():
    with pytest.raises(sidereon.SpkParseError) as truncated:
        sidereon.load_spk(b"")

    truncated_error = truncated.value
    assert isinstance(truncated_error, sidereon.ParseError)
    assert truncated_error.detail["family"] == "spk"
    assert truncated_error.detail["kind"] == "Truncated"
    assert truncated_error.detail["needed"] > truncated_error.detail["actual"]

    kernel_path = os.path.join(FIXTURES, "spk", "horizons_eros_type21.bsp")
    spk = sidereon.load_spk(kernel_path)
    with pytest.raises(ValueError) as unknown:
        spk.state(999999, 10, 757339200.0)
    unknown_error = unknown.value
    assert unknown_error.detail == {
        "family": "spk",
        "kind": "UnknownBody",
        "body": 999999,
    }

    segment = spk.segments[0]
    with pytest.raises(sidereon.SolveError) as outside:
        spk.state(segment.target, segment.center, segment.stop_et + 1.0)
    outside_error = outside.value
    detail = outside_error.detail
    assert detail["family"] == "spk"
    query_et = segment.stop_et + 1.0
    assert detail == {
        "family": "spk",
        "kind": "CoverageGap",
        "target": segment.target,
        "center": segment.center,
        "et": query_et,
        "et_bits_hex": struct.pack(">d", query_et).hex(),
    }

    assert (
        spk.state(segment.target, segment.center, segment.start_et).target
        == segment.target
    )
    assert outside_error.detail is detail
    assert sidereon.SpkParseError("manual").detail is None


def test_cdm_duplicate_fields_and_successful_parse_retain_full_duplicate_values():
    path = os.path.join(CORE_FIXTURES, "cdm", "ccsds_example2.kvn")
    assert sidereon.parse_cdm_kvn(_read(path)).message_id

    with pytest.raises(sidereon.CdmParseError) as refused:
        sidereon.parse_cdm_kvn("CCSDS_CDM_VERS = 1.0\nCCSDS_CDM_VERS = 2.0\n")
    error = refused.value
    assert isinstance(error, sidereon.ParseError)
    assert error.detail == {
        "family": "cdm",
        "kind": "DuplicateField",
        "field": "CCSDS_CDM_VERS",
        "first": "1.0",
        "second": "2.0",
    }
    assert sidereon.parse_cdm_kvn(_read(path)).message_id
    assert error.detail["second"] == "2.0"
    assert sidereon.CdmParseError("manual").detail is None


def test_oem_invalid_declared_unit_keeps_core_fields_and_valid_fixture_succeeds():
    path = os.path.join(CORE_FIXTURES, "oem", "gps.xml")
    source = _read(path)
    assert sidereon.parse_oem_xml(source).segments
    invalid = source.replace("<X>15600.123456</X>", '<X units="m">15600.123456</X>', 1)
    assert invalid != source

    with pytest.raises(sidereon.OemParseError) as refused:
        sidereon.parse_oem_xml(invalid)
    detail = refused.value.detail
    assert detail["family"] == "oem"
    assert detail["kind"] == "UnitMismatch"
    assert detail["field"] == "X"
    assert detail["unit"] == "m"
    assert detail["expected"] == "km"
    assert sidereon.OemParseError("manual").detail is None


def test_opm_duplicate_fields_and_successful_parse_retain_full_values():
    path = os.path.join(CORE_FIXTURES, "opm", "osprey.kvn")
    assert sidereon.parse_opm_kvn(_read(path)).state.epoch

    changed = _read(path).replace("X = 6878.137\n", "X = 6878.137\nX = 6879.137\n", 1)
    assert changed != _read(path)
    with pytest.raises(sidereon.OpmParseError) as refused:
        sidereon.parse_opm_kvn(changed)
    assert refused.value.detail == {
        "family": "opm",
        "kind": "DuplicateField",
        "field": "X",
        "first": "6878.137",
        "second": "6879.137",
    }
    assert sidereon.OpmParseError("manual").detail is None


def _copy_omm(base, *, object_name=None, comments=None):
    return sidereon.Omm(
        base.epoch,
        base.mean_motion,
        base.eccentricity,
        base.inclination_deg,
        base.ra_of_asc_node_deg,
        base.arg_of_pericenter_deg,
        base.mean_anomaly_deg,
        base.norad_cat_id,
        ccsds_omm_vers=base.ccsds_omm_vers,
        creation_date=base.creation_date,
        originator=base.originator,
        object_name=object_name if object_name is not None else base.object_name,
        object_id=base.object_id,
        center_name=base.center_name,
        ref_frame=base.ref_frame,
        time_system=base.time_system,
        mean_element_theory=base.mean_element_theory,
        ephemeris_type=base.ephemeris_type,
        classification_type=base.classification_type,
        element_set_no=base.element_set_no,
        rev_at_epoch=base.rev_at_epoch,
        bstar=base.bstar,
        mean_motion_dot=base.mean_motion_dot,
        mean_motion_ddot=base.mean_motion_ddot,
        classification=base.classification,
        message_id=base.message_id,
        ref_frame_epoch=base.ref_frame_epoch,
        semi_major_axis_km=base.semi_major_axis_km,
        gm_km3_s2=base.gm_km3_s2,
        spacecraft=base.spacecraft,
        bterm_m2_kg=base.bterm_m2_kg,
        agom_m2_kg=base.agom_m2_kg,
        covariance=base.covariance,
        user_defined=base.user_defined,
        comments=comments if comments is not None else base.comments,
    )


def test_omm_writer_text_and_nested_csv_errors_keep_exact_payloads():
    path = os.path.join(CORE_FIXTURES, "omm", "25544.kvn")
    base = sidereon.parse_omm_kvn(_read(path))
    assert sidereon.parse_omm_kvn(base.to_kvn_string()).epoch == base.epoch

    bad_text = _copy_omm(base, object_name="object\nname")
    with pytest.raises(sidereon.OmmParseError) as text_refused:
        bad_text.to_kvn_string()
    text_detail = text_refused.value.detail
    assert text_detail["family"] == "omm"
    assert text_detail["kind"] == "UnwritableText"
    assert text_detail["field"] == "OBJECT_NAME"
    assert text_detail["value"] == "object\nname"
    assert text_detail["issue"] == "LineBreak"

    comments = sidereon.OmmComments(metadata=["metadata comment"])
    with pytest.raises(sidereon.OmmParseError) as csv_refused:
        sidereon.encode_omm_csv([_copy_omm(base, comments=comments)])
    csv_detail = csv_refused.value.detail
    assert csv_detail["kind"] == "InRecord"
    assert csv_detail["index"] == 0
    assert csv_detail["source"]["kind"] == "UnwritableText"
    assert csv_detail["source"]["field"] == "COMMENT"
    assert csv_detail["source"]["value"] == "metadata comment"
    assert csv_detail["source"]["issue"] == "CommentNotCarried"
    assert sidereon.OmmParseError("manual").detail is None


def test_omm_skipped_json_record_retains_typed_error_after_array_is_dropped():
    path = os.path.join(CORE_FIXTURES, "omm", "25544.json")
    valid_json = _read(path).strip()
    valid_records = json.loads(valid_json)
    assert len(valid_records) == 1
    valid_record = json.dumps(valid_records[0])
    array = sidereon.parse_omm_json_array(json.dumps([valid_records[0], {}]))
    assert len(array) == 1
    skipped = array.skipped[0]
    assert skipped.index == 1
    assert skipped.reason
    del array

    assert sidereon.parse_omm_json(valid_record).epoch
    expected = {
        "family": "omm",
        "kind": "MissingField",
        "field": "EPOCH",
    }
    detail = skipped.detail
    assert detail == expected
    detail["kind"] = "caller mutation"
    assert skipped.detail == expected
    assert skipped.index == 1
    assert skipped.reason
