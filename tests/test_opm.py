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
    assert cov.lower_triangle.shape == (21,)
    assert cov.lower_triangle[0] == 0.01
    matrix = cov.to_covariance6()
    assert matrix.shape == (6, 6)
    assert matrix[0, 0] == 0.01
    np.testing.assert_array_equal(matrix, matrix.T)


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
    # The KVN fixture states two comments the XML fixture does not: one before
    # OBJECT_NAME, which belongs to the metadata block, and one before the
    # first maneuver. Both are retained, and every other item agrees.
    assert opm_from_kvn.comments == []
    assert opm_from_kvn.metadata.comments == [
        "Annotated OPM fixture for a low Earth orbit servicing spacecraft."
    ]
    assert opm_from_kvn.maneuvers[0].comments == ["Two planned trim burns."]
    xml = sidereon.parse_opm_xml(_read(XML_PATH))
    assert xml.metadata.comments == []
    assert xml.maneuvers[0].comments == []
    assert xml.ccsds_opm_vers == opm_from_kvn.ccsds_opm_vers
    assert xml.creation_date == opm_from_kvn.creation_date
    assert xml.originator == opm_from_kvn.originator
    assert xml.metadata.object_name == opm_from_kvn.metadata.object_name
    assert xml.metadata.ref_frame == opm_from_kvn.metadata.ref_frame
    assert xml.state == opm_from_kvn.state
    assert xml.keplerian == opm_from_kvn.keplerian
    assert xml.spacecraft == opm_from_kvn.spacecraft
    assert xml.covariance == opm_from_kvn.covariance
    assert len(xml.maneuvers) == len(opm_from_kvn.maneuvers)
    for got, want in zip(xml.maneuvers, opm_from_kvn.maneuvers):
        assert got.epoch_ignition == want.epoch_ignition
        np.testing.assert_array_equal(got.dv_km_s, want.dv_km_s)


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


def test_opm_float_hash_matches_equality_without_changing_stored_bits():
    positive_state = sidereon.OpmState(
        epoch="2026-06-28T00:00:00.000",
        position_km=np.array([0.0, 1.0, 2.0]),
        velocity_km_s=np.array([0.0, 4.0, 5.0]),
    )
    negative_state = sidereon.OpmState(
        epoch="2026-06-28T00:00:00.000",
        position_km=np.array([-0.0, 1.0, 2.0]),
        velocity_km_s=np.array([-0.0, 4.0, 5.0]),
    )
    assert positive_state == negative_state
    assert hash(positive_state) == hash(negative_state)
    assert len({positive_state, negative_state}) == 1
    state_values = {positive_state: "found"}
    assert state_values[negative_state] == "found"
    assert not np.signbit(positive_state.position_km[0])
    assert np.signbit(negative_state.position_km[0])

    metadata = sidereon.OpmMetadata(
        object_name="HASH-TEST",
        object_id="2026-001A",
        center_name="EARTH",
        ref_frame="EME2000",
        time_system="UTC",
    )

    def keplerian(anomaly):
        return sidereon.OpmKeplerian(
            7000.0,
            0.001,
            45.0,
            120.0,
            30.0,
            398600.0,
            true_anomaly_deg=anomaly,
        )

    positive_keplerian = keplerian(0.0)
    negative_keplerian = keplerian(-0.0)
    assert positive_keplerian == negative_keplerian
    assert hash(positive_keplerian) == hash(negative_keplerian)

    positive_spacecraft = sidereon.OpmSpacecraft(mass_kg=0.0, drag_coeff=2.2)
    negative_spacecraft = sidereon.OpmSpacecraft(mass_kg=-0.0, drag_coeff=2.2)
    assert positive_spacecraft == negative_spacecraft
    assert hash(positive_spacecraft) == hash(negative_spacecraft)

    positive_covariance_values = np.ones(21)
    negative_covariance_values = np.ones(21)
    positive_covariance_values[0] = 0.0
    negative_covariance_values[0] = -0.0
    positive_covariance = sidereon.OpmCovariance(positive_covariance_values)
    negative_covariance = sidereon.OpmCovariance(negative_covariance_values)
    assert positive_covariance == negative_covariance
    assert hash(positive_covariance) == hash(negative_covariance)

    def maneuver(duration, dv_sign):
        return sidereon.OpmManeuver(
            "2026-06-28T00:00:00.000",
            duration,
            0.1,
            "EME2000",
            np.array([dv_sign * 0.0, 0.02, 0.03]),
        )

    positive_maneuver = maneuver(0.0, 1.0)
    negative_maneuver = maneuver(-0.0, -1.0)
    assert positive_maneuver == negative_maneuver
    assert hash(positive_maneuver) == hash(negative_maneuver)

    positive_message = sidereon.Opm(
        metadata,
        positive_state,
        keplerian=positive_keplerian,
        spacecraft=positive_spacecraft,
        covariance=positive_covariance,
        maneuvers=[positive_maneuver],
    )
    negative_message = sidereon.Opm(
        metadata,
        negative_state,
        keplerian=negative_keplerian,
        spacecraft=negative_spacecraft,
        covariance=negative_covariance,
        maneuvers=[negative_maneuver],
    )
    assert positive_message == negative_message
    assert hash(positive_message) == hash(negative_message)
    assert len({positive_message, negative_message}) == 1
    message_values = {positive_message: "nested"}
    assert message_values[negative_message] == "nested"

    distinct = sidereon.OpmState(
        epoch="2026-06-28T00:00:00.000",
        position_km=np.array([1.0, 1.0, 2.0]),
        velocity_km_s=np.array([3.0, 4.0, 5.0]),
    )
    assert positive_state != distinct
    assert len({positive_state, distinct}) == 2

    nan_payload = np.array([0x7FF8000000000084, 0, 0], dtype=np.uint64).view(np.float64)
    nan_state = sidereon.OpmState(
        epoch="2026-06-28T00:00:00.000",
        position_km=nan_payload,
        velocity_km_s=np.array([3.0, 4.0, 5.0]),
    )
    hash(nan_state)
    assert int(nan_state.position_km.view(np.uint64)[0]) == 0x7FF8000000000084
    assert nan_state != nan_state


def test_parse_opm_kvn_rejects_garbage():
    with pytest.raises(sidereon.OpmParseError):
        sidereon.parse_opm_kvn("not an OPM at all")
    assert issubclass(sidereon.OpmParseError, sidereon.ParseError)
