"""Public OPM parser and writers retain every message and nested DTO field."""

import numpy as np
import sidereon
from test_opm import KVN_PATH, _read


def test_public_opm_routes_retain_every_fixture_field_and_optional_value():
    base = sidereon.parse_opm_kvn(_read(KVN_PATH))
    assert base.ccsds_opm_vers == "2.0"
    assert base.comments == []
    assert base.classification is None
    assert base.creation_date == "2026-06-28T12:30:00.000"
    assert base.originator == "SIDEREON TEST"
    assert base.message_id is None

    assert base.metadata.comments == [
        "Annotated OPM fixture for a low Earth orbit servicing spacecraft."
    ]
    assert base.metadata.object_name == "OSPREY-1"
    assert base.metadata.object_id == "2026-045A"
    assert base.metadata.center_name == "EARTH"
    assert base.metadata.ref_frame == "EME2000"
    assert base.metadata.ref_frame_epoch is None
    assert base.metadata.time_system == "UTC"

    assert base.state.comments == []
    assert base.state.epoch == "2026-06-28T12:00:00.000"
    np.testing.assert_array_equal(
        base.state.position_km, np.array([6878.137, -120.25, 410.75])
    )
    np.testing.assert_array_equal(
        base.state.velocity_km_s, np.array([0.125, 7.612, 1.034])
    )

    assert base.keplerian.comments == []
    assert base.keplerian.semi_major_axis_km == 6878.137
    assert base.keplerian.eccentricity == 0.0012
    assert base.keplerian.inclination_deg == 51.64
    assert base.keplerian.ra_of_asc_node_deg == 120.5
    assert base.keplerian.arg_of_pericenter_deg == 87.2
    assert base.keplerian.true_anomaly_deg == 42.0
    assert base.keplerian.mean_anomaly_deg is None
    assert base.keplerian.gm_km3_s2 == 398600.4418

    assert base.spacecraft.comments == []
    assert base.spacecraft.mass_kg == 425.0
    assert base.spacecraft.solar_rad_area_m2 == 9.5
    assert base.spacecraft.solar_rad_coeff == 1.21
    assert base.spacecraft.drag_area_m2 == 7.2
    assert base.spacecraft.drag_coeff == 2.2

    assert base.covariance.comments == []
    assert base.covariance.cov_ref_frame == "EME2000"
    np.testing.assert_array_equal(
        base.covariance.lower_triangle,
        np.array(
            [
                0.01,
                0.0,
                0.02,
                0.0,
                0.0,
                0.03,
                0.0,
                0.0,
                0.0,
                0.000001,
                0.0,
                0.0,
                0.0,
                0.0,
                0.000002,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.000003,
            ]
        ),
    )

    assert len(base.maneuvers) == 2
    first, second = base.maneuvers
    assert first.comments == ["Two planned trim burns."]
    assert first.epoch_ignition == "2026-06-28T12:15:00.000"
    assert first.duration_s == 12.5
    assert first.delta_mass_kg == -0.42
    assert first.ref_frame == "TNW"
    np.testing.assert_array_equal(first.dv_km_s, np.array([0.0005, 0.001, 0.0]))
    assert second.comments == []
    assert second.epoch_ignition == "2026-06-28T13:45:00.000"
    assert second.duration_s == 8.0
    assert second.delta_mass_kg == -0.27
    assert second.ref_frame == "TNW"
    np.testing.assert_array_equal(second.dv_km_s, np.array([-0.0002, 0.0008, 0.0001]))

    assert base.user_defined == []
    assert base.user_defined_comments == []

    metadata = sidereon.OpmMetadata(
        "OSPREY-1",
        "2026-045A",
        "EARTH",
        "EME2000",
        "UTC",
        ref_frame_epoch="J2000",
        comments=["metadata details"],
    )
    state = sidereon.OpmState(
        "2026-06-28T12:00:00.000",
        np.array([6878.137, -120.25, 410.75]),
        np.array([0.125, 7.612, 1.034]),
        comments=["state details"],
    )
    keplerian = sidereon.OpmKeplerian(
        6878.137,
        0.0012,
        51.64,
        120.5,
        87.2,
        398600.4418,
        mean_anomaly_deg=41.0,
        comments=["Keplerian details"],
    )
    spacecraft = sidereon.OpmSpacecraft(
        mass_kg=425.0,
        solar_rad_area_m2=9.5,
        solar_rad_coeff=1.21,
        drag_area_m2=7.2,
        drag_coeff=2.2,
        comments=["spacecraft details"],
    )
    covariance = sidereon.OpmCovariance(
        base.covariance.lower_triangle,
        cov_ref_frame="EME2000",
        comments=["covariance details"],
    )
    maneuvers = [
        sidereon.OpmManeuver(
            first.epoch_ignition,
            first.duration_s,
            first.delta_mass_kg,
            first.ref_frame,
            first.dv_km_s,
            comments=["maneuver 1 details"],
        ),
        sidereon.OpmManeuver(
            second.epoch_ignition,
            second.duration_s,
            second.delta_mass_kg,
            second.ref_frame,
            second.dv_km_s,
            comments=["maneuver 2 details"],
        ),
    ]
    opm = sidereon.Opm(
        metadata,
        state,
        ccsds_opm_vers="2.0",
        creation_date="2026-06-28T12:30:00.000",
        originator="SIDEREON TEST",
        classification="U",
        message_id="OPM-TEST-1",
        comments=["header details"],
        keplerian=keplerian,
        spacecraft=spacecraft,
        covariance=covariance,
        maneuvers=maneuvers,
        user_defined=[("MISSION", "SERVICING"), ("MODE", "AUTONOMOUS")],
        user_defined_comments=["user parameter details"],
    )

    for encode, parse in (
        (sidereon.Opm.to_kvn_string, sidereon.parse_opm_kvn),
        (sidereon.Opm.to_xml_string, sidereon.parse_opm_xml),
    ):
        reparsed = parse(encode(opm))
        assert reparsed == opm
        assert reparsed.keplerian.mean_anomaly_deg == 41.0
        assert reparsed.keplerian.true_anomaly_deg is None
        assert reparsed.user_defined == [
            ("MISSION", "SERVICING"),
            ("MODE", "AUTONOMOUS"),
        ]
        assert reparsed.user_defined_comments == ["user parameter details"]
