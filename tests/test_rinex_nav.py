"""RINEX NAV parsing through the Python binding uses real committed fixtures."""

import collections
import json
import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES

NAV_FIXTURES = os.path.join(FIXTURES, "nav")
MESSAGE_BY_GOLDEN = {
    "GPS_LNAV": sidereon.NavMessage.GPS_LNAV,
    "GPS_CNAV": sidereon.NavMessage.GPS_CNAV,
    "GPS_CNAV2": sidereon.NavMessage.GPS_CNAV2,
    "QZSS_CNAV": sidereon.NavMessage.QZSS_CNAV,
    "QZSS_CNAV2": sidereon.NavMessage.QZSS_CNAV2,
    "GAL_INAV": sidereon.NavMessage.GALILEO_INAV,
    "GAL_FNAV": sidereon.NavMessage.GALILEO_FNAV,
    "BDS_D1": sidereon.NavMessage.BEIDOU_D1,
    "BDS_D2": sidereon.NavMessage.BEIDOU_D2,
}


def _read_nav(name):
    with open(os.path.join(NAV_FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


def _count_by(records, key):
    return collections.Counter(key(record) for record in records)


def _float_from_hex_bits(value):
    return struct.unpack(">d", int(value, 16).to_bytes(8, "big"))[0]


def _float_bits(value):
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def _golden_case(name):
    path = os.path.join(CORE_FIXTURES, "broadcast_golden.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return next(case for case in doc["cases"] if case["name"] == name)


def _cnav_golden_case(name):
    path = os.path.join(CORE_FIXTURES, "cnav_broadcast_golden.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return next(case for case in doc["cases"] if case["name"] == name)


def _matching_record(records, case):
    message = MESSAGE_BY_GOLDEN[case["message"]]
    expected = case["elements_hex"]
    matches = [
        record
        for record in records
        if record.satellite == case["sat"]
        and record.message == message
        and _float_bits(record.elements.toe_sow) == int(expected["toe_sow"], 16)
        and _float_bits(record.elements.sqrt_a) == int(expected["sqrt_a"], 16)
        and _float_bits(record.elements.e) == int(expected["e"], 16)
    ]
    assert len(matches) == 1
    return matches[0]


def test_parse_mixed_nav_records_and_default_store():
    text = _read_nav("ESBC00DNK_R_20201770000_01D_MN.rnx")

    records = sidereon.parse_rinex_nav_records(text)
    assert len(records) == 2216
    assert _count_by(records, lambda r: r.satellite[0]) == {
        "G": 257,
        "E": 1602,
        "C": 357,
    }

    galileo_messages = _count_by(
        [r for r in records if r.satellite.startswith("E")],
        lambda r: r.message.label,
    )
    assert galileo_messages["galileo_inav"] == 821
    assert galileo_messages["galileo_fnav"] == 781

    g01 = next(r for r in records if r.satellite == "G01")
    assert g01.message == sidereon.NavMessage.GPS_LNAV
    assert g01.week == 2111
    assert 5100.0 < g01.elements.sqrt_a < 5200.0
    assert 0.0 < g01.elements.e < 0.05
    assert g01.clock.toc_sow == g01.elements.toe_sow
    assert g01.fit_interval_s == 14400.0

    store = sidereon.parse_rinex_nav(text)
    assert store.leap_seconds == 18.0
    assert store.record_count > 0
    assert store.glonass_record_count == 0
    assert all(r.sv_health == 0.0 for r in store.records)
    assert all(r.message != sidereon.NavMessage.GALILEO_FNAV for r in store.records)
    assert any(r.satellite == "C05" for r in store.records)

    iono = store.iono_corrections
    assert isinstance(iono.gps.alpha, np.ndarray)
    assert iono.gps.alpha.shape == (4,)
    assert iono.gps.beta.shape == (4,)
    assert abs(iono.gps.alpha[0] - 4.6566e-09) < 1e-19
    assert abs(iono.gps.beta[0] - 8.1920e04) < 1e-3
    assert iono.beidou is None
    assert "BroadcastEphemeris(" in repr(store)


def test_parse_brdc_header_ionosphere_and_glonass_records():
    name = "BRDC00GOP_R_20210010000_01D_MN.rnx"
    path = os.path.join(NAV_FIXTURES, name)
    text = _read_nav(name)

    records = sidereon.parse_rinex_nav_records(text)
    assert _count_by(records, lambda r: r.satellite[0]) == {"C": 1, "E": 1}

    glonass = sidereon.parse_rinex_glonass_records(text)
    assert len(glonass) == 1
    r10 = glonass[0]
    assert r10.satellite == "R10"
    assert r10.sv_health == 0.0
    assert r10.position_m.shape == (3,)
    assert r10.velocity_m_s.shape == (3,)
    assert r10.acceleration_m_s2.shape == (3,)
    assert 25_000_000.0 < np.linalg.norm(r10.position_m) < 26_000_000.0

    iono = sidereon.parse_rinex_iono_corrections(text)
    assert iono.gps is not None
    assert iono.beidou is not None
    np.testing.assert_allclose(
        iono.beidou.alpha,
        np.array([1.1180e-08, 2.9800e-08, -4.1720e-07, 6.5570e-07]),
    )
    np.testing.assert_allclose(
        iono.beidou.beta,
        np.array([1.4130e05, -5.2430e05, 1.6380e06, -4.5880e05]),
    )
    assert sidereon.parse_rinex_leap_seconds(text) == 18.0

    from_path = sidereon.load_rinex_nav(path)
    from_bytes = sidereon.load_rinex_nav(text.encode("utf-8"))
    assert from_path.glonass_record_count == 1
    assert from_bytes.glonass_records[0].satellite == "R10"


def test_parse_rinex_v4_nav_fixture_records():
    text = _read_nav("KMS300DNK_R_20221591000_01H_MN.rnx")

    records = sidereon.parse_rinex_nav_records(text)
    assert len(records) == 174
    assert _count_by(records, lambda r: r.satellite[0]) == {
        "G": 30,
        "E": 108,
        "C": 36,
    }
    assert _count_by(records, lambda r: r.message.label) == {
        "gps_lnav": 30,
        "galileo_inav": 55,
        "galileo_fnav": 53,
        "beidou_d1": 33,
        "beidou_d2": 3,
    }

    store = sidereon.parse_rinex_nav(text)
    assert store.record_count > 0
    assert store.leap_seconds == 18.0


@pytest.mark.parametrize(
    "case_name",
    ["gps_at_toe", "gal_plus_2h", "bds_geo_plus_2h", "bds_meo_week_fold"],
)
def test_broadcast_record_evaluate_matches_core_golden(case_name):
    case = _golden_case(case_name)
    records = sidereon.parse_rinex_nav_records(
        _read_nav("ESBC00DNK_R_20201770000_01D_MN.rnx")
    )
    record = _matching_record(records, case)

    t_sow_s = _float_from_hex_bits(case["t_sow_hex"])
    state = record.evaluate(t_sow_s)
    expected = case["expect_hex"]

    assert isinstance(state.position_m, np.ndarray)
    assert state.position_m.shape == (3,)
    assert state.position_m.dtype == np.float64
    assert _float_bits(state.t_sow_s) == int(case["t_sow_hex"], 16)
    assert _float_bits(state.x_m) == int(expected["x_m"], 16)
    assert _float_bits(state.y_m) == int(expected["y_m"], 16)
    assert _float_bits(state.z_m) == int(expected["z_m"], 16)
    assert [_float_bits(value) for value in state.position_m] == [
        int(expected["x_m"], 16),
        int(expected["y_m"], 16),
        int(expected["z_m"], 16),
    ]
    assert _float_bits(state.clock_s) == int(expected["dt_clock_total_s"], 16)
    assert _float_bits(state.clock_polynomial_s) == int(expected["dt_clock_poly_s"], 16)
    assert _float_bits(state.relativistic_clock_s) == int(expected["dt_rel_s"], 16)
    assert _float_bits(state.group_delay_s) == int(expected["tgd_s"], 16)
    assert state.kepler_iterations == case["kepler_iterations"]
    assert "BroadcastEvaluation(" in repr(state)


def test_broadcast_record_evaluate_rejects_non_finite_epoch():
    record = sidereon.parse_rinex_nav_records(
        _read_nav("ESBC00DNK_R_20201770000_01D_MN.rnx")
    )[0]

    with pytest.raises(ValueError):
        record.evaluate(float("nan"))


def test_cnav_rinex4_eval_and_accessors_match_core_golden():
    path = os.path.join(CORE_FIXTURES, "nav", "BRD400DLR_S_20261800000_01H_MN_trim.rnx")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    records = sidereon.parse_rinex_nav_records(text)
    case = _cnav_golden_case("g01_gps_cnav_toe")
    record = _matching_record(records, case)

    assert record.is_cnav_family
    assert record.cnav is not None
    assert record.cnav.ura_ed_index == 0
    assert record.cnav.ura_ed_m == pytest.approx(2.0)
    assert record.cnav.ura_ned_m(
        record.cnav.top_week, record.cnav.top_tow_s
    ) == pytest.approx(1.0)
    assert record.group_delays.cnav_isc_l1ca_s == pytest.approx(-2.910383045673e-10)
    assert record.cnav_single_frequency_correction_s(
        sidereon.CnavSignal.L1_CA
    ) == pytest.approx(record.group_delay_s)
    assert sidereon.cnav_ura_nominal_m(1) == pytest.approx(2.8)
    assert sidereon.cnav_ura_ned_m(
        record.cnav, record.cnav.top_week, record.cnav.top_tow_s
    ) == pytest.approx(1.0)

    state = record.evaluate(_float_from_hex_bits(case["t_sow_hex"]))
    expected = case["expect_hex"]
    assert _float_bits(state.x_m) == int(expected["x_m"], 16)
    assert _float_bits(state.y_m) == int(expected["y_m"], 16)
    assert _float_bits(state.z_m) == int(expected["z_m"], 16)
    assert _float_bits(state.clock_s) == int(expected["dt_clock_total_s"], 16)


def test_rinex4_lenient_parse_lint_and_message_preference_are_exposed():
    path = os.path.join(CORE_FIXTURES, "nav", "BRD400DLR_S_20261800000_01H_MN_trim.rnx")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    parsed = sidereon.parse_rinex_nav_lenient(text)
    assert parsed.record_count == 2
    assert parsed.skipped_count == 4
    assert {skipped.satellite for skipped in parsed.skipped} == {"G01", "G03", "J02"}

    report = sidereon.lint_rinex_nav(text)
    assert not report.is_clean
    assert report.count(sidereon.RinexLintSeverity.ERROR) == 4
    assert [finding.code for finding in report.findings[:4]] == ["NAV-B01"] * 4
    assert all(finding.kind == "NavDroppedBlock" for finding in report.findings[:4])

    store = sidereon.parse_rinex_nav(text)
    assert store.message_preference == sidereon.NavMessagePreference.PREFER_LEGACY
    store.set_message_preference(sidereon.NavMessagePreference.PREFER_MODERN)
    assert store.message_preference == sidereon.NavMessagePreference.PREFER_MODERN


def test_rinex_nav_parse_error_is_typed():
    bogus = (
        "     3.05           OBSERVATION DATA   M                   RINEX VERSION / TYPE\n"  # noqa: E501
        "                                                            END OF HEADER\n"
    )
    with pytest.raises(sidereon.RinexNavParseError):
        sidereon.parse_rinex_nav_records(bogus)


def test_public_header_helpers_surface_malformed_fields():
    bad_iono = (
        "     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n"  # noqa: E501
        "GPSA not-a-float                                             IONOSPHERIC CORR\n"  # noqa: E501
        "     XXX                                                         END OF HEADER\n"  # noqa: E501
    )
    bad_leap = (
        "     3.05           NAVIGATION DATA     M                   RINEX VERSION / TYPE\n"  # noqa: E501
        "bad                                                        LEAP SECONDS\n"
        "     XXX                                                         END OF HEADER\n"  # noqa: E501
    )

    with pytest.raises(sidereon.RinexNavParseError):
        sidereon.parse_rinex_iono_corrections(bad_iono)
    with pytest.raises(sidereon.RinexNavParseError):
        sidereon.parse_rinex_leap_seconds(bad_leap)


def test_encode_rinex_nav_round_trips_records():
    text = _read_nav("BRDC00GOP_R_20210010000_01D_MN.rnx")
    records = sidereon.parse_rinex_nav_records(text)
    assert records, "fixture should yield at least one Keplerian record"

    reparsed = sidereon.parse_rinex_nav_records(sidereon.encode_rinex_nav(records))
    assert len(reparsed) == len(records)
    for original, again in zip(records, reparsed):
        assert again.satellite == original.satellite
        assert again.message == original.message
        assert again.week == original.week
        assert _float_bits(again.elements.sqrt_a) == _float_bits(
            original.elements.sqrt_a
        )
        assert _float_bits(again.clock.af0) == _float_bits(original.clock.af0)
