"""CCSDS OEM binding: parse, serialize, and round-trip the core fixtures.

The KVN and XML fixtures are the same committed files the Rust core asserts on
(``CORE_FIXTURES/oem``). The binding parses them, re-encodes through the core
writer, and re-parses, requiring structural equality each hop.
"""

import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

OEM_DIR = os.path.join(CORE_FIXTURES, "oem")
KVN_PATH = os.path.join(OEM_DIR, "gps.kvn")
XML_PATH = os.path.join(OEM_DIR, "gps.xml")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def oem_from_kvn():
    return sidereon.parse_oem_kvn(_read(KVN_PATH))


def test_parse_oem_kvn_surface(oem_from_kvn):
    oem = oem_from_kvn
    assert oem.ccsds_oem_vers == "2.0"
    assert oem.originator == "SIDEREON TEST"
    assert oem.skipped_states == 0
    assert len(oem.segments) == 1
    assert "Oem(" in repr(oem)

    segment = oem.segments[0]
    meta = segment.metadata
    assert meta.object_name == "GPS BIIRM-8"
    assert meta.object_id == "2005-038A"
    assert meta.center_name == "EARTH"
    assert meta.ref_frame == "EME2000"
    assert meta.time_system == "GPS"
    assert meta.interpolation == "LAGRANGE"
    assert meta.interpolation_degree == 5

    assert len(segment.states) == 3
    first = segment.states[0]
    assert first.epoch == "2026-06-28T00:00:00.000"
    np.testing.assert_array_equal(
        first.position_km, np.array([15600.123456, -21000.654321, 20100.111111])
    )
    np.testing.assert_array_equal(
        first.velocity_km_s, np.array([2.102345, 1.305678, -2.987654])
    )
    assert first.acceleration_km_s2 is None

    # The final state carries an acceleration triple.
    last = segment.states[-1]
    assert last.acceleration_km_s2 is not None
    np.testing.assert_array_equal(
        last.acceleration_km_s2, np.array([0.000001, -0.000002, 0.000003])
    )


def test_parse_oem_covariance_block(oem_from_kvn):
    covariances = oem_from_kvn.segments[0].covariances
    assert len(covariances) == 1
    cov = covariances[0]
    assert cov.epoch == "2026-06-28T00:15:00.000"
    assert cov.cov_ref_frame == "RTN"
    assert cov.matrix.shape == (6, 6)
    # Diagonal carries the declared variances; the matrix is symmetric.
    assert cov.matrix[0, 0] == 0.0001
    assert cov.matrix[5, 5] == 0.00000003
    np.testing.assert_array_equal(cov.matrix, cov.matrix.T)


def test_oem_kvn_round_trip(oem_from_kvn):
    encoded = oem_from_kvn.to_kvn_string()
    assert sidereon.parse_oem_kvn(encoded) == oem_from_kvn


def test_oem_xml_round_trip(oem_from_kvn):
    encoded = oem_from_kvn.to_xml_string()
    assert sidereon.parse_oem_xml(encoded) == oem_from_kvn


def test_oem_kvn_and_xml_fixtures_agree(oem_from_kvn):
    assert sidereon.parse_oem_xml(_read(XML_PATH)) == oem_from_kvn


def test_oem_constructible_and_round_trips():
    state = sidereon.OemState(
        epoch="2026-06-28T00:00:00.000",
        position_km=np.array([7000.0, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.5, 0.0]),
    )
    meta = sidereon.OemMetadata(
        object_name="BUILT-1",
        object_id="2026-099A",
        center_name="EARTH",
        ref_frame="EME2000",
        time_system="UTC",
        start_time="2026-06-28T00:00:00.000",
        stop_time="2026-06-28T01:00:00.000",
    )
    segment = sidereon.OemSegment(metadata=meta, states=[state])
    oem = sidereon.Oem(segments=[segment], originator="UNIT TEST")
    assert sidereon.parse_oem_kvn(oem.to_kvn_string()) == oem


def test_oem_value_objects_are_hashable(oem_from_kvn):
    meta = oem_from_kvn.segments[0].metadata
    state = oem_from_kvn.segments[0].states[0]
    # Defining __eq__ without __hash__ would make these unusable as set members.
    assert {meta, state}
    assert hash(meta) == hash(oem_from_kvn.segments[0].metadata)


def test_parse_oem_kvn_rejects_garbage():
    with pytest.raises(sidereon.OemParseError):
        sidereon.parse_oem_kvn("not an OEM at all")
    # The message-specific error is a ParseError subclass.
    assert issubclass(sidereon.OemParseError, sidereon.ParseError)
