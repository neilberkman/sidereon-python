"""TLE encode round-trip, checksum warnings, and the exception hierarchy.

The fixture `tle_roundtrip.json` is emitted by the crate's validated round-trip
harness (`SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test
tle_python_fixture`); it carries the committed ISS TLE, the engine's parsed
element fields, the lines `tle::encode` reproduces, and the advisory
checksum-warning case. The binding parses the same TLE and must reproduce the
same encoded lines and warnings.
"""

import json
import os

import pytest
import sidereon
from _helpers import FIXTURES


def _load_fixture():
    with open(os.path.join(FIXTURES, "tle_roundtrip.json")) as fh:
        return json.load(fh)


def test_to_lines_reproduces_engine_encoding():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"], opsmode=fx["opsmode"])

    line1, line2 = tle.to_lines()

    assert isinstance(line1, str) and isinstance(line2, str)
    assert line1 == fx["encoded"]["line1"]
    assert line2 == fx["encoded"]["line2"]
    # The committed ISS lines round-trip character-exact.
    assert line1 == fx["tle"]["line1"]
    assert line2 == fx["tle"]["line2"]


def test_element_properties_match_reference():
    fx = _load_fixture()
    el = fx["elements"]
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])

    assert tle.catalog_number == el["catalog_number"]
    assert tle.classification == el["classification"]
    assert tle.international_designator == el["international_designator"]
    assert tle.epoch_year == el["epoch_year"]
    assert tle.epoch_day_of_year == el["epoch_day_of_year"]
    assert tle.inclination_deg == el["inclination_deg"]
    assert tle.raan_deg == el["raan_deg"]
    assert tle.eccentricity == el["eccentricity"]
    assert tle.arg_perigee_deg == el["arg_perigee_deg"]
    assert tle.mean_anomaly_deg == el["mean_anomaly_deg"]
    assert tle.mean_motion_rev_per_day == el["mean_motion"]
    assert tle.mean_motion_dot == el["mean_motion_dot"]
    assert tle.mean_motion_double_dot == el["mean_motion_double_dot"]
    assert tle.bstar == el["bstar"]
    assert tle.rev_number == el["rev_number"]


def test_clean_tle_has_no_checksum_warnings():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    assert tle.checksum_warnings == []


def test_checksum_warnings_match_reference():
    fx = _load_fixture()
    case = fx["checksum_case"]
    tle = sidereon.Tle(case["line1"], case["line2"])

    warnings = tle.checksum_warnings
    assert len(warnings) == len(case["warnings"])
    for got, want in zip(warnings, case["warnings"]):
        assert got.line_label == want["line_label"]
        assert got.expected == want["expected"]
        assert got.computed == want["computed"]
        assert "ChecksumWarning(" in repr(got)

    # Value-like equality on ChecksumWarning.
    first = warnings[0]
    twin = sidereon.Tle(case["line1"], case["line2"]).checksum_warnings[0]
    assert first == twin


def test_exception_hierarchy_shape():
    assert issubclass(sidereon.ParseError, sidereon.SidereonError)
    assert issubclass(sidereon.Sp3ParseError, sidereon.ParseError)
    assert issubclass(sidereon.TleParseError, sidereon.ParseError)
    assert issubclass(sidereon.SolveError, sidereon.SidereonError)


def test_bad_tle_raises_tle_parse_error():
    with pytest.raises(sidereon.TleParseError):
        sidereon.Tle("not a tle", "also not a tle")
    # Still catchable through the base classes.
    with pytest.raises(sidereon.ParseError):
        sidereon.Tle("not a tle", "also not a tle")
    with pytest.raises(sidereon.SidereonError):
        sidereon.Tle("not a tle", "also not a tle")


def test_garbage_sp3_raises_sp3_parse_error():
    with pytest.raises(sidereon.Sp3ParseError):
        sidereon.load_sp3(b"not a valid sp3 file")
    with pytest.raises(sidereon.SidereonError):
        sidereon.load_sp3(b"not a valid sp3 file")
