"""Public OMM routes preserve every nested DTO and message field."""

import numpy as np
import sidereon
from test_omm import _load, _path


def test_public_kvn_and_xml_routes_retain_nested_omm_dto_fields():
    base = sidereon.parse_omm_kvn(_load(_path("25544.kvn")))
    assert base.ccsds_omm_vers == "2.0"
    assert base.classification is None
    assert base.creation_date is None
    assert base.originator is None
    assert base.message_id is None
    assert base.object_name == "ISS (ZARYA)"
    assert base.object_id == "1998-067A"
    assert base.center_name == "EARTH"
    assert base.ref_frame == "TEME"
    assert base.ref_frame_epoch is None
    assert base.time_system == "UTC"
    assert base.mean_element_theory == "SGP/SGP4"
    assert base.epoch.iso8601 == "2026-06-17T04:32:52.099296"
    assert (base.epoch.year, base.epoch.month, base.epoch.day) == (2026, 6, 17)
    assert (base.epoch.hour, base.epoch.minute, base.epoch.second) == (4, 32, 52)
    assert base.epoch.microsecond == 99_296
    assert base.epoch.femtosecond == 0
    assert base.mean_motion == 15.49273435
    assert base.semi_major_axis_km is None
    assert base.eccentricity == 0.0004737
    assert base.inclination_deg == 51.6332
    assert base.ra_of_asc_node_deg == 300.0813
    assert base.arg_of_pericenter_deg == 195.1146
    assert base.mean_anomaly_deg == 164.9702
    assert base.gm_km3_s2 is None
    assert base.spacecraft is None
    assert base.ephemeris_type == 0
    assert base.classification_type == "U"
    assert base.norad_cat_id == 25544
    assert base.element_set_no == 999
    assert base.rev_at_epoch == 57175
    assert base.bstar == 0.00017172
    assert base.bterm_m2_kg is None
    assert base.mean_motion_dot == 0.00009113
    assert base.mean_motion_ddot == 0.0
    assert base.agom_m2_kg is None
    assert base.covariance is None
    assert base.user_defined == []
    assert base.comments == sidereon.OmmComments()
    spacecraft = sidereon.OmmSpacecraft(
        comments=["spacecraft details"],
        mass_kg=1250.5,
        solar_rad_area_m2=22.25,
        solar_rad_coeff=1.125,
        drag_area_m2=18.5,
        drag_coeff=2.25,
    )
    covariance = sidereon.OmmCovariance(
        np.arange(1.0, 22.0, dtype=np.float64) / 8.0,
        cov_ref_frame="TEME",
        comments=["covariance details"],
    )
    omm = sidereon.Omm(
        base.epoch,
        15.125,
        base.eccentricity,
        base.inclination_deg,
        base.ra_of_asc_node_deg,
        base.arg_of_pericenter_deg,
        base.mean_anomaly_deg,
        90001,
        ccsds_omm_vers="2.0",
        classification="U",
        creation_date="2026-06-17T00:00:00.000",
        originator="SIDEREON TEST",
        message_id="CDM-OMM-1",
        object_name="ISS (ZARYA)",
        object_id="2024-001A",
        center_name="EARTH",
        ref_frame="TEME",
        ref_frame_epoch="J2000",
        time_system="UTC",
        mean_element_theory="SGP/SGP4",
        semi_major_axis_km=7000.25,
        gm_km3_s2=398600.5,
        spacecraft=spacecraft,
        ephemeris_type=0,
        classification_type="U",
        element_set_no=1024,
        rev_at_epoch=12345,
        bstar=0.000125,
        bterm_m2_kg=0.00025,
        mean_motion_dot=0.000375,
        mean_motion_ddot=0.0005,
        agom_m2_kg=0.000625,
        covariance=covariance,
        user_defined=[("MISSION", "TEST FLIGHT"), ("NOTE", "RETAIN THIS")],
        comments=sidereon.OmmComments(
            header=["header details"],
            metadata=["metadata details"],
            mean_elements=["element details"],
            tle_parameters=["tle details"],
            user_defined=["user parameter details"],
        ),
    )

    for encode, parse in (
        (sidereon.Omm.to_kvn_string, sidereon.parse_omm_kvn),
        (sidereon.Omm.to_xml_string, sidereon.parse_omm_xml),
    ):
        reparsed = parse(encode(omm))
        assert reparsed == omm
        assert reparsed.spacecraft == spacecraft
        assert reparsed.covariance == covariance
        assert reparsed.user_defined == [
            ("MISSION", "TEST FLIGHT"),
            ("NOTE", "RETAIN THIS"),
        ]
        assert reparsed.comments == omm.comments
