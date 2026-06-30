"""Events, body-angle, and DOP bindings reproduce engine fixture bits."""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES, hex_to_f64


def _fixture():
    with open(os.path.join(FIXTURES, "events_bodies_dop.json")) as fh:
        return json.load(fh)


FX = _fixture()


def _bits(arr):
    return np.asarray(arr, dtype=np.float64).view(np.uint64)


def _expect_bits(hex_values):
    return np.asarray([int(h, 16) for h in hex_values], dtype=np.uint64)


def _assert_bits_within_one_ulp(value, expected_hex):
    assert abs(int(_bits(value)) - int(expected_hex, 16)) <= 1


def _array3(entries, key):
    return np.asarray(
        [[hex_to_f64(h) for h in entry[key]] for entry in entries], dtype=np.float64
    )


def _sp3_path(name):
    return os.path.join(CORE_FIXTURES, "sp3", name)


def test_shadow_fraction_and_status_match_reference_bits():
    sat = _array3(FX["eclipse"], "satellite_position_km_hex")
    sun = _array3(FX["eclipse"], "sun_position_km_hex")
    fractions = sidereon.shadow_fraction(sat, sun)
    assert fractions.shape == (len(FX["eclipse"]),)
    assert fractions.dtype == np.float64
    assert np.array_equal(
        _bits(fractions),
        _expect_bits([case["shadow_fraction_hex"] for case in FX["eclipse"]]),
    )

    statuses = sidereon.eclipse_status(sat, sun)
    expected = [
        getattr(sidereon.EclipseStatus, case["status"]) for case in FX["eclipse"]
    ]
    assert statuses == expected
    assert repr(sidereon.EclipseStatus.UMBRA) == "EclipseStatus.UMBRA"


def test_angle_helpers_match_reference_bits():
    cases = FX["angles"]
    sat = _array3(cases, "satellite_position_km_hex")
    sun = _array3(cases, "sun_position_km_hex")
    moon = _array3(cases, "moon_position_km_hex")
    observer = _array3(cases, "observer_position_km_hex")

    checks = [
        (sidereon.sun_angle(sat, sun), "sun_angle_deg_hex"),
        (sidereon.moon_angle(sat, moon), "moon_angle_deg_hex"),
        (sidereon.sun_elevation(sat, sun), "sun_elevation_deg_hex"),
        (sidereon.phase_angle(sat, sun, observer), "phase_angle_deg_hex"),
        (sidereon.earth_angular_radius(sat), "earth_angular_radius_deg_hex"),
    ]
    for got, key in checks:
        assert got.shape == (len(cases),)
        assert got.dtype == np.float64
        assert np.array_equal(_bits(got), _expect_bits([case[key] for case in cases]))


def test_gnss_dop_matches_reference_bits():
    case = FX["dop"]
    los = np.asarray(
        [[hex_to_f64(h) for h in row] for row in case["line_of_sight_hex"]],
        dtype=np.float64,
    )
    weights = np.asarray([hex_to_f64(h) for h in case["weights_hex"]], dtype=np.float64)
    receiver = sidereon.Wgs84Geodetic(
        hex_to_f64(case["receiver"]["lat_rad_hex"]),
        hex_to_f64(case["receiver"]["lon_rad_hex"]),
        hex_to_f64(case["receiver"]["height_m_hex"]),
    )
    assert receiver == sidereon.Wgs84Geodetic(
        receiver.lat_rad, receiver.lon_rad, receiver.height_m
    )
    assert "Wgs84Geodetic" in repr(receiver)

    dop = sidereon.gnss_dop(los, weights, receiver)
    constructed = sidereon.Dop.from_line_of_sight(los, receiver, weights)
    assert constructed == dop
    assert dop == sidereon.gnss_dop(los, weights, receiver)
    assert "Dop" in repr(dop)
    for attr in ("gdop", "pdop", "hdop", "vdop", "tdop"):
        assert _bits(getattr(dop, attr)) == _expect_bits([case[f"{attr}_hex"]])[0]


def test_dop_from_az_el_matches_symmetric_rust_geometry_bits():
    receiver = sidereon.Wgs84Geodetic(0.0, 0.0)
    azimuth_deg = np.asarray([45.0, 225.0, 135.0, 315.0], dtype=np.float64)
    elevation_deg = np.asarray(
        [
            35.264389682754654,
            35.264389682754654,
            -35.264389682754654,
            -35.264389682754654,
        ],
        dtype=np.float64,
    )

    dop = sidereon.Dop.from_az_el(azimuth_deg, elevation_deg, receiver)

    expected = {
        "gdop": "0x3ff94c583ada5b53",
        "pdop": "0x3ff8000000000000",
        "hdop": "0x3ff3988e1409212e",
        "vdop": "0x3febb67ae8584caa",
        "tdop": "0x3fe0000000000000",
    }
    for attr, bits in expected.items():
        assert _bits(getattr(dop, attr)) == _expect_bits([bits])[0]


def test_gnss_dop_series_matches_real_sp3_fixture():
    sp3 = sidereon.load_sp3(_sp3_path("GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"))
    with open(os.path.join(CORE_FIXTURES, "spp_trace_L2_tropo.json")) as fh:
        trace = json.load(fh)
    station_ecef_m = np.asarray(
        [
            hex_to_f64(value)
            for value in trace["fixture"]["final_solution"]["truth_x"][:3]
        ],
        dtype=np.float64,
    )
    t0 = (2459024.5 - 2451545.0) * 86400.0 + 0.5 * 86400.0
    epochs = t0 + np.arange(13, dtype=np.float64) * 300.0

    series = sidereon.gnss_dop_series(
        sp3,
        station_ecef_m,
        epochs,
        elevation_mask_deg=5.0,
        systems=["G"],
        weighting=sidereon.DopWeighting.UNIT,
    )

    assert series.epoch_count == 13
    assert len(series) == 13
    assert "DopSeries" in repr(series)
    assert repr(sidereon.DopWeighting.ELEVATION) == "DopWeighting.ELEVATION"
    assert sidereon.DopWeighting.UNIT.label == "unit"
    assert np.array_equal(series.step_index, np.arange(13, dtype=np.int64))
    assert np.array_equal(series.j2000_seconds, epochs)
    assert np.array_equal(
        series.satellite_count,
        np.asarray([9, 9, 9, 9, 10, 11, 11, 11, 11, 11, 11, 11, 11], dtype=np.int64),
    )
    assert series.satellites[0] == [
        "G21",
        "G16",
        "G26",
        "G20",
        "G27",
        "G18",
        "G10",
        "G08",
        "G07",
    ]

    expected_first = {
        "gdop": "0x4000c042642e3cbc",
        "pdop": "0x3ffd34cde2c7e400",
        "hdop": "0x3ff257e7df379517",
        "vdop": "0x3ff6ba2ad4e284af",
        "tdop": "0x3ff069acbf06750f",
    }
    for attr, bits in expected_first.items():
        _assert_bits_within_one_ulp(getattr(series, attr)[0], bits)


def test_events_bad_inputs_raise_builtin_errors():
    with pytest.raises(ValueError):
        sidereon.shadow_fraction(np.zeros((0, 3)), np.zeros((0, 3)))
    with pytest.raises(ValueError):
        sidereon.sun_angle(np.zeros((1, 2)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        sidereon.phase_angle(np.zeros((1, 3)), np.zeros((2, 3)), np.zeros((1, 3)))
    with pytest.raises(ValueError):
        sidereon.Wgs84Geodetic(np.pi, 0.0)
    with pytest.raises(ValueError):
        sidereon.gnss_dop(
            np.zeros((3, 3)), np.ones(3), sidereon.Wgs84Geodetic(0.0, 0.0)
        )
    with pytest.raises(ValueError):
        sidereon.gnss_dop(
            np.zeros((4, 3)),
            np.asarray([1.0, 1.0, 1.0, 0.0]),
            sidereon.Wgs84Geodetic(0.0, 0.0),
        )
    with pytest.raises(ValueError):
        sidereon.Dop.from_az_el(
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            sidereon.Wgs84Geodetic(0.0, 0.0),
        )
