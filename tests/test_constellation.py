"""GNSS constellation identity catalog through the binding.

The catalog is a pure wrapper over ``sidereon_core::constellation``: it turns
already fetched CelesTrak ``gps-ops`` OMM JSON and NAVCEN status HTML into
normalized identity records, merges the two sources, exports the mapping CSV,
and validates against an SP3 id list. The fixtures are the same committed
samples the Rust core asserts on (``gps_ops_sample.json``,
``navcen_gps_sample.html``), copied under ``tests/fixtures/constellation``; the
binding must reproduce the core's records and bytes exactly.
"""

import os

import pytest
import sidereon
from _helpers import FIXTURES

CONST_FIXTURES = os.path.join(FIXTURES, "constellation")


def _read(name, mode="r"):
    with open(os.path.join(CONST_FIXTURES, name), mode) as fh:
        return fh.read()


def _records():
    return sidereon.from_celestrak_json(_read("gps_ops_sample.json"))


def _statuses():
    return sidereon.parse_navcen(_read("navcen_gps_sample.html"))


def _merged():
    return sidereon.merge_navcen(_records(), _statuses())


def test_from_celestrak_json_builds_identity_records():
    records = _records()
    assert [r.prn for r in records] == [3, 5, 13, 19]

    by_prn = {r.prn: r for r in records}
    prn3 = by_prn[3]
    assert prn3.norad_id == 40294
    assert prn3.sp3_id == "G03"
    assert prn3.system == sidereon.GnssSystem.GPS
    assert prn3.svn is None
    assert prn3.active is True
    assert prn3.usable is True
    assert prn3.source.celestrak is not None
    assert prn3.source.celestrak.block_type == "IIF"
    assert prn3.source.celestrak.group == "gps-ops"
    assert prn3.source.navcen is None


def test_parse_navcen_parses_status_rows():
    statuses = _statuses()
    assert [(s.prn, s.svn) for s in statuses] == [(3, 69), (5, 50), (13, 43), (19, 59)]

    by_prn = {s.prn: s for s in statuses}
    prn19 = by_prn[19]
    assert prn19.usable is False
    assert prn19.active_nanu is True
    assert prn19.nanu_type == "UNUSABLE"
    assert prn19.system == sidereon.GnssSystem.GPS


def test_parse_navcen_accepts_bytes():
    from_str = sidereon.parse_navcen(_read("navcen_gps_sample.html"))
    from_bytes = sidereon.parse_navcen(_read("navcen_gps_sample.html", "rb"))
    assert from_str == from_bytes


def test_merge_navcen_fills_svn_and_usability():
    merged = _merged()
    assert [(r.prn, r.svn, r.usable) for r in merged] == [
        (3, 69, True),
        (5, 50, True),
        (13, None, True),
        (19, 59, False),
    ]

    by_prn = {r.prn: r for r in merged}
    # PRN 13's NAVCEN block type (IIR) conflicts with the CelesTrak block III,
    # so the NAVCEN row is recorded as a conflict and not merged: svn stays None.
    assert by_prn[13].source.navcen is None
    assert by_prn[13].source.navcen_conflict is not None
    # PRN 19 merges cleanly and inherits the unusable NANU.
    assert by_prn[19].source.navcen is not None
    assert by_prn[19].source.navcen.svn == 59


def test_to_csv_lower_matches_reference_bytes():
    merged = _merged()
    assert sidereon.to_csv(merged, "lower") == (
        "prn,norad_cat_id,active,sp3_id\n"
        "3,40294,true,G03\n"
        "5,35752,true,G05\n"
        "13,68791,true,G13\n"
        "19,28190,false,G19\n"
    )


def test_to_csv_defaults_to_lower_and_supports_title():
    merged = _merged()
    assert sidereon.to_csv(merged) == sidereon.to_csv(merged, "lower")
    title = sidereon.to_csv(merged, "title")
    assert "True" in title and "False" in title
    with pytest.raises(TypeError):
        sidereon.to_csv(merged, "yes")


def test_validate_against_sp3_ids_reports_findings():
    merged = _merged()
    report = sidereon.validate_against_sp3_ids(merged, ["G03", "G05", "G13"])
    assert report.inactive_unusable_prns == [(sidereon.GnssSystem.GPS, 19)]
    assert report.missing_sp3_ids == []
    assert report.extra_sp3_ids == []
    assert report.duplicate_prns == []
    assert report.duplicate_norad_ids == []
    assert sidereon.is_valid(report) is False


def test_validate_clean_catalog_is_valid():
    # The base CelesTrak records (all active+usable, unique PRNs/NORADs) carry no
    # findings without an SP3 product to compare against.
    report = sidereon.validate(_records())
    assert sidereon.is_valid(report) is True


def test_diff_detects_svn_and_usability_changes():
    previous = _records()
    current = _merged()
    report = sidereon.diff(previous, current)
    assert sidereon.changed(report) is True

    # PRNs 3, 5, 19 gained an SVN from the NAVCEN merge (13 did not: conflict).
    svn_prns = {c.prn for c in report.svn_changed}
    assert svn_prns == {3, 5, 19}
    by_prn = {c.prn: c for c in report.svn_changed}
    assert by_prn[19].from_ is None
    assert by_prn[19].to == 59

    # PRN 19 flipped usable True -> False.
    usability_prns = {c.prn: (c.from_, c.to) for c in report.usability_changed}
    assert usability_prns == {19: (True, False)}

    assert report.added == []
    assert report.removed == []


def test_diff_of_identical_snapshots_is_unchanged():
    records = _records()
    assert sidereon.changed(sidereon.diff(records, records)) is False


def test_gps_sp3_id_renders_canonical_token():
    assert sidereon.gps_sp3_id(7) == "G07"
    assert sidereon.gps_sp3_id(13) == "G13"


def test_constellation_errors_are_typed():
    with pytest.raises(sidereon.OmmParseError):
        sidereon.from_celestrak_json("{ not json")
    with pytest.raises(sidereon.ConstellationError):
        sidereon.parse_navcen("<html>no gps rows</html>")
    assert issubclass(sidereon.ConstellationError, sidereon.SidereonError)
