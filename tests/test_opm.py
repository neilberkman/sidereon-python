"""CCSDS OPM binding: parse, serialize, and round-trip the core fixtures.

The KVN and XML fixtures are the same committed files the Rust core asserts on
(``CORE_FIXTURES/opm``). The binding parses them, re-encodes through the core
writer, and re-parses, requiring structural equality each hop.
"""

import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

OPM_DIR = os.path.join(CORE_FIXTURES, "opm")
KVN_PATH = os.path.join(OPM_DIR, "osprey.kvn")
XML_PATH = os.path.join(OPM_DIR, "osprey.xml")


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


@pytest.fixture(scope="module")
def opm_from_kvn():
    return sidereon.parse_opm_kvn(_read(KVN_PATH))


def test_parse_opm_kvn_surface(opm_from_kvn):
    opm = opm_from_kvn
    assert opm.ccsds_opm_vers == "2.0"
    assert opm.originator == "SIDEREON TEST"
    assert "Opm(" in repr(opm)

    meta = opm.metadata
    assert meta.object_name == "OSPREY-1"
    assert meta.object_id == "2026-045A"
    assert meta.center_name == "EARTH"
    assert meta.ref_frame == "EME2000"
    assert meta.time_system == "UTC"

    state = opm.state
    assert state.epoch == "2026-06-28T12:00:00.000"
    np.testing.assert_array_equal(
        state.position_km, np.array([6878.137, -120.25, 410.75])
    )
    np.testing.assert_array_equal(state.velocity_km_s, np.array([0.125, 7.612, 1.034]))


def test_parse_opm_keplerian_anomaly(opm_from_kvn):
    kep = opm_from_kvn.keplerian
    assert kep is not None
    assert kep.semi_major_axis_km == 6878.137
    assert kep.eccentricity == 0.0012
    assert kep.inclination_deg == 51.64
    assert kep.gm_km3_s2 == 398600.4418
    # The fixture carries a TRUE_ANOMALY, so mean reads back as None.
    assert kep.true_anomaly_deg == 42.0
    assert kep.mean_anomaly_deg is None


def test_parse_opm_spacecraft_and_covariance(opm_from_kvn):
    spacecraft = opm_from_kvn.spacecraft
    assert spacecraft is not None
    assert spacecraft.mass_kg == 425.0
    assert spacecraft.drag_coeff == 2.2

    cov = opm_from_kvn.covariance
    assert cov is not None
    assert cov.cov_ref_frame == "EME2000"
    assert cov.matrix.shape == (6, 6)
    assert cov.matrix[0, 0] == 0.01
    np.testing.assert_array_equal(cov.matrix, cov.matrix.T)


def test_parse_opm_maneuvers(opm_from_kvn):
    maneuvers = opm_from_kvn.maneuvers
    assert len(maneuvers) == 2
    first = maneuvers[0]
    assert first.epoch_ignition == "2026-06-28T12:15:00.000"
    assert first.duration_s == 12.5
    assert first.delta_mass_kg == -0.42
    assert first.ref_frame == "TNW"
    np.testing.assert_array_equal(first.dv_km_s, np.array([0.0005, 0.001, 0.0]))


def test_opm_kvn_round_trip(opm_from_kvn):
    encoded = opm_from_kvn.to_kvn_string()
    assert sidereon.parse_opm_kvn(encoded) == opm_from_kvn


def test_opm_xml_round_trip(opm_from_kvn):
    encoded = opm_from_kvn.to_xml_string()
    assert sidereon.parse_opm_xml(encoded) == opm_from_kvn


def test_opm_kvn_and_xml_fixtures_agree(opm_from_kvn):
    assert sidereon.parse_opm_xml(_read(XML_PATH)) == opm_from_kvn


def test_opm_constructible_and_round_trips():
    meta = sidereon.OpmMetadata(
        object_name="BUILT-2",
        object_id="2026-100A",
        center_name="EARTH",
        ref_frame="EME2000",
        time_system="UTC",
    )
    state = sidereon.OpmState(
        epoch="2026-06-28T00:00:00.000",
        position_km=np.array([6878.137, 0.0, 0.0]),
        velocity_km_s=np.array([0.0, 7.612, 1.034]),
    )
    opm = sidereon.Opm(metadata=meta, state=state, originator="UNIT TEST")
    assert sidereon.parse_opm_kvn(opm.to_kvn_string()) == opm


def test_opm_keplerian_requires_exactly_one_anomaly():
    with pytest.raises(ValueError):
        sidereon.OpmKeplerian(6878.137, 0.0012, 51.64, 120.5, 87.2, 398600.4418)
    with pytest.raises(ValueError):
        sidereon.OpmKeplerian(
            6878.137,
            0.0012,
            51.64,
            120.5,
            87.2,
            398600.4418,
            true_anomaly_deg=42.0,
            mean_anomaly_deg=41.0,
        )
    mean_only = sidereon.OpmKeplerian(
        6878.137, 0.0012, 51.64, 120.5, 87.2, 398600.4418, mean_anomaly_deg=41.0
    )
    assert mean_only.mean_anomaly_deg == 41.0
    assert mean_only.true_anomaly_deg is None


def test_opm_value_objects_are_hashable(opm_from_kvn):
    assert {opm_from_kvn.metadata, opm_from_kvn.state}
    assert hash(opm_from_kvn.maneuvers[0]) == hash(opm_from_kvn.maneuvers[0])


def test_parse_opm_kvn_rejects_garbage():
    with pytest.raises(sidereon.OpmParseError):
        sidereon.parse_opm_kvn("not an OPM at all")
    assert issubclass(sidereon.OpmParseError, sidereon.ParseError)
