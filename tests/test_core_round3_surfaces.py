"""Surfaces the binding carries from the core's format, policy and UT1 work.

Each test reads through the binding and checks what the core retains or
reports: typed lookup outcomes, strict and lenient reads, the records a
multi-record reader skipped, and the new typed errors.
"""

import json
import os
import pathlib

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES

BIAS_PATH = (
    pathlib.Path(CORE_FIXTURES) / "bias" / "COD0OPSFIN_20261330000_01D_01D_OSB.BIA"
)


def _bias_bytes():
    with open(BIAS_PATH, "rb") as fh:
        return fh.read()


def _unique_record_index(bias, predicate):
    """Index of the first record matching `predicate` whose target and
    observables no other record shares."""
    records = bias.records
    keys = [(r.kind, r.target, r.obs1, r.obs2) for r in records]
    for index, record in enumerate(records):
        if predicate(record) and keys.count(keys[index]) == 1:
            return index
    raise AssertionError("no matching record in the fixture")


def test_ut1_error_and_validity_mode_are_exposed():
    assert issubclass(sidereon.Ut1OutsideCoverageError, sidereon.SidereonError)
    assert issubclass(sidereon.Ut1OutsideCoverageError, ValueError)
    assert sidereon.Ut1OutsideCoverageError("x").reason is None
    assert sidereon.ValidityMode.STRICT != sidereon.ValidityMode.PERMISSIVE


def test_bias_sinex_header_metadata_and_record_families():
    bias = sidereon.parse_bias_sinex(_bias_bytes())
    assert bias.mode == "absolute"
    assert bias.time_scale == sidereon.TimeScale.GPST
    assert bias.time_system_label == "G"
    families = {record.family for record in bias.records}
    assert {"code", "phase"} <= families
    units = {record.unit for record in bias.records}
    assert "ns" in units
    first = bias.records[0]
    assert first.line is not None
    assert first.valid_from == (2026, 133, 0)
    assert first.raw_epochs == ("2026:133:00000", "2026:134:00000")


def test_bias_lookups_return_typed_outcomes():
    bias = sidereon.parse_bias_sinex(_bias_bytes())
    code_index = _unique_record_index(
        bias,
        lambda r: (
            r.kind == "OSB"
            and r.target == "G01"
            and r.obs1 == "C1C"
            and r.slope is None
        ),
    )
    code = bias.code_osb_seconds("G01", "C1C", 2026, 133, 0)
    assert code.status == "available"
    assert code.is_available
    assert code.value == bias.records[code_index].value
    assert code.records == [code_index]
    assert "BiasLookup(" in repr(code)

    absent = bias.code_osb_seconds("G01", "C9Z", 2026, 133, 0)
    assert absent.status == "absent"
    assert absent.value is None
    assert absent.records == []

    other_scale = bias.code_osb_seconds(
        "G01", "C1C", 2026, 133, 0, time_scale=sidereon.TimeScale.UTC
    )
    assert other_scale.status == "unsupported_scale"
    assert other_scale.product_time_scale == sidereon.TimeScale.GPST
    assert other_scale.query_time_scale == sidereon.TimeScale.UTC

    phase_index = _unique_record_index(
        bias,
        lambda r: (
            r.kind == "OSB"
            and r.target == "G01"
            and r.obs1 == "L1C"
            and r.unit == "ns"
            and r.slope is None
        ),
    )
    assert bias.records[phase_index].family == "phase"
    assert bias.records[phase_index].is_phase
    required = bias.phase_osb_cycles("G01", "L1C", 2026, 133, 0)
    assert required.status == "carrier_frequency_required"
    assert required.record == phase_index
    converted = bias.phase_osb_cycles("G01", "L1C", 2026, 133, 0, carrier_hz=1575.42e6)
    assert converted.status == "available"
    assert converted.records == [phase_index]
    assert converted.value == pytest.approx(
        bias.records[phase_index].value * 1575.42e6, rel=1e-12
    )


def test_bias_sinex_policy_reads_a_departure_only_when_lenient():
    departed = _bias_bytes() + b"\n"
    with pytest.raises(sidereon.BiasError):
        sidereon.parse_bias_sinex(departed)
    with pytest.raises(sidereon.BiasError) as strict_error:
        sidereon.parse_bias_sinex(departed, sidereon.BiasReadPolicy.STRICT)
    assert strict_error.value.kind == "departure"
    assert type(strict_error.value.kind) is str
    assert type(strict_error.value.details) is dict
    assert set(strict_error.value.details) == {"departure"}
    assert strict_error.value.details["departure"]["kind"] == "content_after_footer"
    assert strict_error.value.details["departure"]["line"] > 0
    lenient = sidereon.parse_bias_sinex(
        departed, policy=sidereon.BiasReadPolicy.LENIENT
    )
    assert lenient.record_count == sidereon.parse_bias_sinex(_bias_bytes()).record_count
    assert all(isinstance(notice, str) for notice in lenient.notices)
    assert any(
        notice["kind"] == "departure"
        and notice["departure"]["kind"] == "content_after_footer"
        and notice["departure"]["line"] > 0
        for notice in lenient.notice_details
    )
    lossy = sidereon.parse_bias_sinex_lossy(departed, sidereon.BiasReadPolicy.LENIENT)
    assert lossy.value.record_count == lenient.record_count


def test_bias_writers_restate_exact_source_and_preserve_non_utf8_bytes():
    sinex_bytes = _bias_bytes().decode("latin-1").encode("utf-8")
    sinex = sidereon.parse_bias_sinex(sinex_bytes)
    sinex_text = sidereon.write_bias_sinex(sinex)
    sinex_output = sidereon.write_bias_sinex_bytes(sinex)
    assert sinex_text == sinex_bytes.decode("utf-8")
    assert type(sinex_output) is bytes
    assert sinex_output == sinex_bytes

    core_fixtures = pathlib.Path(CORE_FIXTURES)
    dcb_path = core_fixtures / "bias" / "P1C1_RINEX.DCB"
    dcb_bytes = dcb_path.read_bytes()
    dcb = sidereon.parse_code_dcb(dcb_bytes, options=None)
    assert sidereon.write_code_dcb(dcb) == dcb_bytes.decode("utf-8")
    dcb_output = sidereon.write_code_dcb_bytes(dcb)
    assert type(dcb_output) is bytes
    assert dcb_output == dcb_bytes

    invalid_dcb_bytes = dcb_bytes.replace(b"DIFFERENTIAL", b"DIFFERENTI\xffL", 1)
    assert invalid_dcb_bytes != dcb_bytes
    invalid_dcb = sidereon.parse_code_dcb(
        invalid_dcb_bytes, options=None, policy=sidereon.BiasReadPolicy.LENIENT
    )
    preserved_dcb = sidereon.write_code_dcb_bytes(invalid_dcb)
    assert type(preserved_dcb) is bytes
    assert preserved_dcb == invalid_dcb_bytes
    with pytest.raises(sidereon.BiasError) as dcb_refusal:
        sidereon.write_code_dcb(invalid_dcb)
    assert dcb_refusal.value.kind == "invalid_utf8_line"
    assert dcb_refusal.value.details["line"] > 0

    invalid_bytes = (core_fixtures / "bias" / "CODE.BIA").read_bytes()
    invalid = sidereon.parse_bias_sinex(
        invalid_bytes, policy=sidereon.BiasReadPolicy.LENIENT
    )
    preserved = sidereon.write_bias_sinex_bytes(invalid)
    assert type(preserved) is bytes
    assert preserved == invalid_bytes
    with pytest.raises(sidereon.BiasError) as refusal:
        sidereon.write_bias_sinex(invalid)
    assert refusal.value.kind == "invalid_utf8_line"
    assert refusal.value.details["line"] > 0


def _tle_fixture():
    with open(os.path.join(FIXTURES, "tle_roundtrip.json")) as fh:
        return json.load(fh)


def test_tle_file_lists_rejected_records_under_each_policy():
    fx = _tle_fixture()
    good = (fx["tle"]["line1"], fx["tle"]["line2"])
    flipped = (fx["checksum_case"]["line1"], fx["checksum_case"]["line2"])
    text = "\n".join(
        [
            "ISS (ZARYA)",
            *good,
            "ISS FLIPPED",
            *flipped,
            flipped[1],
        ]
    )

    strict = sidereon.parse_tle_file(text)
    assert [named.name for named in strict.satellites] == ["ISS (ZARYA)"]
    assert strict.satellites[0].line_number == 2
    assert strict.skipped == len(strict.rejected) == 2
    invalid, orphan = strict.rejected
    assert invalid.issue == "invalid"
    assert invalid.name == "ISS FLIPPED"
    assert invalid.line_number == 4
    assert orphan.issue == "orphan_line2"
    assert orphan.line_number == 7
    assert "RejectedTleRecord(" in repr(invalid)

    lenient = sidereon.parse_tle_file(text, policy=sidereon.TlePolicy.LENIENT)
    assert [named.name for named in lenient.satellites] == [
        "ISS (ZARYA)",
        "ISS FLIPPED",
    ]
    [orphan_only] = lenient.rejected
    assert orphan_only.issue == "orphan_line2"
    warnings = lenient.satellites[1].tle.checksum_warnings
    assert [w.kind for w in warnings] == ["mismatch"]


def test_tle_blank_fields_read_as_none():
    fx = _tle_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    assert tle.rev_number == fx["elements"]["rev_number"]
    assert tle.ephemeris_type == 0
    assert tle.bstar_text == " 31745-4"
    assert tle.mean_motion_double_dot_text == " 00000-0"
    assert tle.to_lines() == (fx["encoded"]["line1"], fx["encoded"]["line2"])


def _omm_json_object():
    path = os.path.join(CORE_FIXTURES, "omm", "25544.json")
    with open(path) as fh:
        data = json.load(fh)
    return data[0] if isinstance(data, list) else data


def test_omm_json_arrays_report_skipped_records():
    record = _omm_json_object()
    pair = json.dumps([record, record])
    with pytest.raises(sidereon.OmmParseError):
        sidereon.parse_omm_json(pair)
    array = sidereon.parse_omm_json_array(pair)
    assert len(array) == 2
    assert array.skipped == []
    single = sidereon.parse_omm_json(json.dumps(record))
    assert array.omms[0] == single
    assert single.ccsds_omm_vers is None
    assert single.user_defined == []
    assert single.covariance is None

    with_bad = sidereon.parse_omm_json_array(json.dumps([record, {"OBJECT_NAME": 1}]))
    assert len(with_bad) == 1
    [skipped] = with_bad.skipped
    assert skipped.index == 1
    assert skipped.reason

    encoded = sidereon.encode_omm_json_array(array.omms)
    assert sidereon.parse_omm_json_array(encoded).omms == array.omms


def test_omm_csv_round_trips_a_record():
    omm = sidereon.parse_omm_json(json.dumps(_omm_json_object()))
    text = sidereon.encode_omm_csv([omm])
    assert sidereon.parse_omm_csv(text) == omm
    array = sidereon.parse_omm_csv_array(text)
    assert array.omms == [omm]
    assert array.skipped == []


def test_omm_optional_tle_parameters_stay_absent():
    with open(os.path.join(CORE_FIXTURES, "omm", "25544.kvn")) as fh:
        kvn = sidereon.parse_omm_kvn(fh.read())
    rebuilt = sidereon.Omm(
        kvn.epoch,
        kvn.mean_motion,
        kvn.eccentricity,
        kvn.inclination_deg,
        kvn.ra_of_asc_node_deg,
        kvn.arg_of_pericenter_deg,
        kvn.mean_anomaly_deg,
        None,
        ccsds_omm_vers="2.0",
        object_name="NO CATALOG NUMBER",
        object_id=kvn.object_id,
        center_name="EARTH",
        ref_frame="TEME",
        time_system="UTC",
        mean_element_theory="SGP4",
    )
    assert rebuilt.norad_cat_id is None
    assert rebuilt.ephemeris_type is None
    assert rebuilt.classification_type is None
    assert rebuilt.bstar is None
    reparsed = sidereon.parse_omm_kvn(rebuilt.to_kvn_string())
    assert reparsed.norad_cat_id is None
    assert reparsed.ephemeris_type is None
    assert reparsed.bstar is None


def test_rtcm_decode_is_total_and_the_stream_reports_every_skip():
    frame = bytes.fromhex("d300153ee7d30302aa3c6d183e4605ff0c02ef2b54843a98d8b487")
    assert len(sidereon.decode_rtcm(frame)) == 1
    with pytest.raises(sidereon.RtcmParseError):
        sidereon.decode_rtcm(b"\x00" + frame)
    corrupt = bytearray(frame)
    corrupt[-1] ^= 0xFF
    stream = sidereon.decode_rtcm_stream(frame + bytes(corrupt))
    assert [m.message_number for m in stream.messages] == [1006]
    assert stream.diagnostics.crc_failures == 1
    assert stream.diagnostics.departures == []
    assert not stream.diagnostics.is_clean
    clean = sidereon.decode_rtcm_stream(frame, sidereon.RtcmPolicy.STRICT)
    assert clean.diagnostics.is_clean


def test_rtcm_trailing_bits_are_a_departure():
    station = sidereon.decode_rtcm(
        bytes.fromhex("d300153ee7d30302aa3c6d183e4605ff0c02ef2b54843a98d8b487")
    )[0].station_coordinates
    assert station.trailing_bits == []
    tail = sidereon.RtcmStationCoordinates(
        message_number=station.message_number,
        reference_station_id=station.reference_station_id,
        itrf_realization_year=station.itrf_realization_year,
        gps_indicator=station.gps_indicator,
        glonass_indicator=station.glonass_indicator,
        galileo_indicator=station.galileo_indicator,
        reference_station_indicator=station.reference_station_indicator,
        ecef_x=station.ecef_x,
        ecef_y=station.ecef_y,
        ecef_z=station.ecef_z,
        single_receiver_oscillator=station.single_receiver_oscillator,
        reserved=station.reserved,
        quarter_cycle_indicator=station.quarter_cycle_indicator,
        antenna_height=station.antenna_height,
        trailing_bits=[True] * 8,
    )
    message = sidereon.RtcmMessage.from_station_coordinates(tail)
    with pytest.raises(sidereon.RtcmEncodeError):
        message.encode()
    body, written = message.encode_with_policy(sidereon.RtcmPolicy.LENIENT)
    assert [d.kind for d in written] == ["trailing_bits"]
    assert written[0].message_number == station.message_number
    with pytest.raises(sidereon.RtcmParseError):
        sidereon.decode_rtcm_message(body)
    decoded, read = sidereon.decode_rtcm_message_with_policy(
        body, sidereon.RtcmPolicy.LENIENT
    )
    assert [d.kind for d in read] == ["trailing_bits"]
    assert decoded.encode(sidereon.RtcmPolicy.LENIENT) == body


def test_sbas_block_encode_policy_and_pad_bits():
    body = bytes.fromhex("5306000000000000000000000000000000000000000000000000000040")
    block = sidereon.decode_sbas_block(body, sidereon.SbasWireForm.BODY226)
    assert block.pad_bits == 0
    assert block.encode() == body
    encoded, written = block.encode_with_policy(sidereon.SbasPolicy.STRICT)
    assert encoded == body
    assert written == []

    other_preamble = bytes([0x11]) + body[1:]
    with pytest.raises(sidereon.RtcmParseError):
        sidereon.decode_sbas_block(other_preamble, sidereon.SbasWireForm.BODY226)
    lenient, read = sidereon.decode_sbas_block_with_policy(
        other_preamble, sidereon.SbasWireForm.BODY226, sidereon.SbasPolicy.LENIENT
    )
    assert [d.kind for d in read] == ["unrecognized_preamble"]
    assert read[0].preamble == 0x11
    assert lenient.encode(sidereon.SbasPolicy.LENIENT) == other_preamble
    with pytest.raises(sidereon.RtcmEncodeError) as strict_encode:
        lenient.encode()
    assert strict_encode.value.kind == "sbas_unrecognized_preamble"
    assert strict_encode.value.details == {"preamble": 0x11}
    with pytest.raises(sidereon.RtcmEncodeError) as strict_policy_encode:
        lenient.encode_with_policy(sidereon.SbasPolicy.STRICT)
    assert strict_policy_encode.value.kind == "sbas_unrecognized_preamble"
    assert strict_policy_encode.value.details == {"preamble": 0x11}


def test_sbas_log_reader_accounts_for_every_line():
    body_hex = "5306000000000000000000000000000000000000000000000000000040"
    text = f"% header comment\n2360 259200 120 1 : {body_hex}\n\n"
    log = sidereon.parse_sbas_rtklib_log(text)
    assert len(log.blocks) == 1
    block = log.blocks[0]
    assert block.declared_message_type == 1
    assert block.message_type == 1
    assert log.refused_lines == []
    assert log.departures == []
    assert (1, "comment") in log.skipped_lines

    declared_wrong = f"2360 259200 120 2 : {body_hex}\n"
    with pytest.raises(sidereon.RtcmParseError):
        sidereon.parse_sbas_rtklib_log(declared_wrong)
    lenient = sidereon.parse_sbas_rtklib_log(
        declared_wrong, policy=sidereon.SbasPolicy.LENIENT
    )
    [departure] = lenient.departures
    assert departure.kind == "declared_message_type"
    assert departure.declared == 2
    assert departure.carried == 1
    assert departure.line == 1


def test_ssr_bias_queries_name_the_signal():
    store = sidereon.SsrCorrectionStore()
    assert store.code_bias("G01", "1C") is None
    assert store.code_bias("G01", "C1C") is None
    assert store.phase_bias("G01", 0, source=sidereon.SsrSource.RTCM_SSR) is None
    with pytest.raises(ValueError):
        store.code_bias("G01", 0)
    with pytest.raises(ValueError):
        store.code_bias("G01", "1C", source=sidereon.SsrSource.RTCM_SSR)
    with pytest.raises(ValueError):
        store.code_bias("G01", "not a code")


def test_space_weather_policy_defaults_follow_the_core():
    path = os.path.join(CORE_FIXTURES, "space_weather", "SW-All-20260702-trim.csv")
    table = sidereon.load_space_weather(path)
    assert table.drag_policy == {
        "allow_interpolated": True,
        "allow_not_observed": False,
        "allow_daily_predicted": True,
        "allow_monthly_predicted": True,
        "require_geomagnetic": True,
    }
    lenient = table.with_drag_policy(allow_not_observed=True, require_geomagnetic=False)
    assert lenient.drag_policy["allow_not_observed"] is True
    assert lenient.drag_policy["require_geomagnetic"] is False
    assert table.drag_policy["require_geomagnetic"] is True
    # 2003-10-30 12:00, with every Ap history bin inside observed rows.
    epoch = 1398 * 86400.0
    history = table.ap_history_at_with_policy(epoch)
    assert len(history.ap) == 7
    assert history.ap == table.ap_array_at(epoch)
    assert sidereon.ObservationClass.NOT_OBSERVED.label == "not_observed"


def test_spk_state_carries_velocity_and_kernel_sets_take_precedence():
    path = os.path.join(CORE_FIXTURES, "spk", "horizons_eros_type21.bsp")
    spk = sidereon.load_spk(path)
    segment = spk.segments[0]
    et = 0.5 * (segment.start_et + segment.stop_et)
    state = spk.state(segment.target, segment.center, et)
    assert state.velocity_km_s.shape == (3,)
    in_frame = spk.state_in_frame(segment.target, segment.center, et, segment.frame)
    np.testing.assert_array_equal(in_frame.position_km, state.position_km)
    kernels = sidereon.SpkKernels([spk])
    assert len(kernels) == 1
    from_set = kernels.state(segment.target, segment.center, et)
    np.testing.assert_array_equal(from_set.position_km, state.position_km)
    np.testing.assert_array_equal(from_set.velocity_km_s, state.velocity_km_s)
    same = kernels.state(segment.target, segment.target, et)
    np.testing.assert_array_equal(same.position_km, np.zeros(3))
    assert sidereon.spk_inertial_frame_name(1) == "J2000"
    assert sidereon.spk_inertial_frame_name(99) is None
    np.testing.assert_array_equal(sidereon.spk_inertial_frame_rotation(1, 1), np.eye(3))
    with pytest.raises(ValueError):
        sidereon.spk_inertial_frame_rotation(1, 99)


def test_ppp_observation_signals_round_trip():
    obs = sidereon.PppObservation(
        satellite_id="G01",
        ambiguity_id="G01",
        code_m=2.2e7,
        phase_m=2.2e7,
        signals=("C1C", "2W", "L1C", "L2W"),
    )
    assert obs.signals == ("1C", "2W", "1C", "2W")
    assert sidereon.PppObservation("G01", "G01", 2.2e7, 2.2e7).signals is None
    with pytest.raises(ValueError):
        sidereon.PppObservation(
            "G01", "G01", 2.2e7, 2.2e7, signals=("1c", "2W", "1C", "2W")
        )


def test_spp_config_carries_the_pseudorange_code():
    cfg = sidereon.SppConfig([], 0.0, 0.0, 1.0, [0.0, 0.0, 0.0, 0.0])
    assert cfg.pseudorange_code == sidereon.PseudorangeCode.SINGLE_FREQUENCY
    iflc = sidereon.SppConfig(
        [],
        0.0,
        0.0,
        1.0,
        [0.0, 0.0, 0.0, 0.0],
        pseudorange_code=sidereon.PseudorangeCode.IONOSPHERE_FREE,
    )
    assert iflc.pseudorange_code == sidereon.PseudorangeCode.IONOSPHERE_FREE
    assert (
        sidereon.StaticEpoch(iflc).pseudorange_code
        == sidereon.PseudorangeCode.IONOSPHERE_FREE
    )


def test_nav_message_selectors_include_navic_and_unclassified_galileo():
    assert sidereon.NavMessage.NAVIC_LNAV.label == "navic_lnav"
    assert sidereon.NavMessage.GALILEO_UNCLASSIFIED.label == "galileo_unclassified"
