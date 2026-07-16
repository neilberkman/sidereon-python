"""Frames + time binding reproduces the engine numbers bit-for-bit.

The fixture ``frames_time.json`` is emitted by the crate's env-gated harness
(``SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test
frames_time_python_fixture``); it carries, per real UTC epoch, the resolved
TT/UT1/TDB scales, sidereal time, nutation/precession, and the engine's frame
transforms on a shared sample state, all as IEEE-754 hex bits. The binding
resolves the same instants and applies the same transforms and must return the
identical bits -- a wrapper that diverges is a wrapper bug, not a new answer.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES, hex_to_f64


def _fixture():
    with open(os.path.join(FIXTURES, "frames_time.json")) as fh:
        return json.load(fh)


FX = _fixture()
SAMPLE_POS = np.asarray([hex_to_f64(h) for h in FX["sample"]["position_km_hex"]])
SAMPLE_VEL = np.asarray([hex_to_f64(h) for h in FX["sample"]["velocity_km_s_hex"]])

# Independent Skyfield 1.49 oracle. Unlike ``frames_time.json``, these values
# were captured from Skyfield with ``float.hex()`` and were not emitted by
# Sidereon. Keep this gate separate from the binding-vs-engine fixture checks.
SKYFIELD_TEME_POS = np.asarray(
    [
        hex_to_f64(h)
        for h in (
            "0x40ace86c23dffb6b",
            "0x409f7fa61c81cb47",
            "0x40b4bd8359159cde",
        )
    ]
)
SKYFIELD_TEME_VEL = np.asarray(
    [
        hex_to_f64(h)
        for h in (
            "0xc00b2ffb7cf9ad7d",
            "0x401b7a8751f7fc4a",
            "0xbfceb36925f07cb4",
        )
    ]
)
SKYFIELD_GCRS_POS = np.asarray(
    [
        hex_to_f64(h)
        for h in (
            "0x40ad0bd9193713e1",
            "0x409f41a3b2073733",
            "0x40b4b6ffad1289d1",
        )
    ]
)
SKYFIELD_GCRS_VEL = np.asarray(
    [
        hex_to_f64(h)
        for h in (
            "0xc00af690723d6cb1",
            "0x401b88e06212f969",
            "0xbfcde8575471eaf0",
        )
    ]
)
SKYFIELD_ITRS_POS = np.asarray(
    [
        hex_to_f64(h)
        for h in (
            "0xc092d5d32b319db8",
            "0x40af8b3b3a722474",
            "0x40b4bd8359159cdb",
        )
    ]
)


def _epochs_array():
    return np.asarray([e["unix_micros"] for e in FX["epochs"]], dtype=np.int64)


def test_instant_scales_match_reference_bits():
    for e in FX["epochs"]:
        inst = sidereon.Instant.from_unix_micros(e["unix_micros"])
        assert inst.jd_whole == hex_to_f64(e["jd_whole_hex"])
        assert inst.tt_jd == hex_to_f64(e["tt_jd_hex"])
        assert inst.ut1_jd == hex_to_f64(e["ut1_jd_hex"])
        assert inst.tdb_jd == hex_to_f64(e["tdb_jd_hex"])
        assert inst.tt_fraction == hex_to_f64(e["tt_fraction_hex"])
        assert inst.ut1_fraction == hex_to_f64(e["ut1_fraction_hex"])
        assert inst.tdb_fraction == hex_to_f64(e["tdb_fraction_hex"])
        assert inst.delta_t_seconds == hex_to_f64(e["delta_t_seconds_hex"])
        assert inst.mean_obliquity_radians == hex_to_f64(
            e["mean_obliquity_radians_hex"]
        )


def test_instant_sidereal_time_matches_reference_bits():
    for e in FX["epochs"]:
        inst = sidereon.Instant.from_unix_micros(e["unix_micros"])
        assert inst.gmst_radians() == hex_to_f64(e["gmst_radians_hex"])
        assert inst.gast_radians() == hex_to_f64(e["gast_radians_hex"])


def test_instant_nutation_precession_match_reference_bits():
    for e in FX["epochs"]:
        inst = sidereon.Instant.from_unix_micros(e["unix_micros"])
        dpsi, deps = inst.nutation_angles()
        assert dpsi == hex_to_f64(e["nutation_dpsi_hex"])
        assert deps == hex_to_f64(e["nutation_deps_hex"])

        prec = inst.precession_matrix()
        assert prec.shape == (3, 3)
        assert prec.dtype == np.float64
        nut = inst.nutation_matrix()
        assert nut.shape == (3, 3)
        for i in range(3):
            for j in range(3):
                assert prec[i, j] == hex_to_f64(e["precession_matrix_hex"][i][j])
                assert nut[i, j] == hex_to_f64(e["nutation_matrix_hex"][i][j])


def test_instant_from_utc_equals_unix_path():
    for e in FX["epochs"]:
        c = e["calendar"]
        second = c["second"] + c["microsecond"] / 1_000_000.0
        inst = sidereon.Instant.from_utc(
            c["year"], c["month"], c["day"], c["hour"], c["minute"], second
        )
        assert inst.unix_micros == e["unix_micros"]
        assert inst == sidereon.Instant.from_unix_micros(e["unix_micros"])
        assert inst.tt_jd == hex_to_f64(e["tt_jd_hex"])


def test_instant_jd_split_recombines():
    e = FX["epochs"][0]
    inst = sidereon.Instant.from_unix_micros(e["unix_micros"])
    split = inst.tt_jd_split
    assert split.whole == inst.jd_whole
    assert split.fraction == inst.tt_fraction
    assert split.jd == split.whole + split.fraction
    assert inst.ut1_jd_split.fraction == inst.ut1_fraction
    assert inst.tdb_jd_split.fraction == inst.tdb_fraction


def test_teme_to_gcrs_matches_reference_bits():
    epochs = _epochs_array()
    n = len(epochs)
    pos = np.tile(SAMPLE_POS, (n, 1))
    vel = np.tile(SAMPLE_VEL, (n, 1))

    for compat, key in (
        (True, "teme_to_gcrs_skyfield"),
        (False, "teme_to_gcrs_direct"),
    ):
        result = sidereon.teme_to_gcrs(pos, vel, epochs, skyfield_compat=compat)
        assert result.epoch_count == n
        assert len(result) == n
        gcrs_pos = result.position_km
        gcrs_vel = result.velocity_km_s
        assert gcrs_pos.shape == (n, 3)
        assert gcrs_vel.shape == (n, 3)
        assert gcrs_pos.dtype == np.float64
        for idx, e in enumerate(FX["epochs"]):
            ref = e[key]
            for axis in range(3):
                assert gcrs_pos[idx, axis] == hex_to_f64(ref["position_hex"][axis])
                assert gcrs_vel[idx, axis] == hex_to_f64(ref["velocity_hex"][axis])


def test_teme_to_gcrs_matches_skyfield_1_49_at_zero_ulp():
    epoch = sidereon.Instant.from_utc(2018, 7, 4, 0, 0, 0.0).unix_micros
    result = sidereon.teme_to_gcrs(
        SKYFIELD_TEME_POS.reshape(1, 3),
        SKYFIELD_TEME_VEL.reshape(1, 3),
        np.asarray([epoch], dtype=np.int64),
        skyfield_compat=True,
    )

    np.testing.assert_array_equal(result.position_km[0], SKYFIELD_GCRS_POS)
    np.testing.assert_array_equal(result.velocity_km_s[0], SKYFIELD_GCRS_VEL)


def test_gcrs_to_itrs_matches_reference_bits():
    epochs = _epochs_array()
    n = len(epochs)
    pos = np.tile(SAMPLE_POS, (n, 1))

    for compat, key in (
        (True, "gcrs_to_itrs_skyfield_hex"),
        (False, "gcrs_to_itrs_direct_hex"),
    ):
        itrs = sidereon.gcrs_to_itrs(pos, epochs, skyfield_compat=compat)
        assert itrs.shape == (n, 3)
        for idx, e in enumerate(FX["epochs"]):
            for axis in range(3):
                assert itrs[idx, axis] == hex_to_f64(e[key][axis])


def test_gcrs_to_itrs_matches_skyfield_1_49_at_zero_ulp():
    epoch = sidereon.Instant.from_utc(2018, 7, 4, 0, 0, 0.0).unix_micros
    result = sidereon.gcrs_to_itrs(
        SKYFIELD_GCRS_POS.reshape(1, 3),
        np.asarray([epoch], dtype=np.int64),
        skyfield_compat=True,
    )

    np.testing.assert_array_equal(result[0], SKYFIELD_ITRS_POS)


def test_itrs_to_gcrs_matches_reference_bits():
    epochs = _epochs_array()
    n = len(epochs)
    pos = np.tile(SAMPLE_POS, (n, 1))
    gcrs = sidereon.itrs_to_gcrs(pos, epochs)
    assert gcrs.shape == (n, 3)
    for idx, e in enumerate(FX["epochs"]):
        for axis in range(3):
            assert gcrs[idx, axis] == hex_to_f64(e["itrs_to_gcrs_hex"][axis])


def test_geodetic_ecef_round_match_reference_bits():
    g2e = FX["geodetic_to_ecef"]
    geo_in = np.asarray([[hex_to_f64(h) for h in g2e["input_hex"]]])
    ecef = sidereon.geodetic_to_ecef(geo_in)
    assert ecef.shape == (1, 3)
    for axis in range(3):
        assert ecef[0, axis] == hex_to_f64(g2e["ecef_km_hex"][axis])

    e2g = FX["ecef_to_geodetic"]
    ecef_in = np.asarray([[hex_to_f64(h) for h in e2g["input_km_hex"]]])
    geo = sidereon.ecef_to_geodetic(ecef_in)
    assert geo.shape == (1, 3)
    for axis in range(3):
        assert geo[0, axis] == hex_to_f64(e2g["geodetic_hex"][axis])


def test_leap_seconds_match_reference_bits():
    for case in FX["leap_seconds_cases"]:
        value = sidereon.leap_seconds(case["year"], case["month"], case["day"])
        assert value == hex_to_f64(case["value_hex"])


def test_leap_seconds_batch():
    cases = FX["leap_seconds_cases"]
    dates = np.asarray(
        [[c["year"], c["month"], c["day"]] for c in cases], dtype=np.int64
    )
    values = sidereon.leap_seconds_batch(dates)
    assert values.shape == (len(cases),)
    assert values.dtype == np.float64
    for i, c in enumerate(cases):
        assert values[i] == hex_to_f64(c["value_hex"])


def test_leap_second_table_info():
    info = sidereon.leap_second_table_info()
    ref = FX["leap_second_table"]
    assert info.source == ref["source"]
    assert info.first_mjd == ref["first_mjd"]
    assert info.last_mjd == ref["last_mjd"]
    assert info.entries == ref["entries"]
    assert "LeapSecondTable" in repr(info)


def test_ut1_coverage_info():
    info = sidereon.ut1_coverage_info()
    ref = FX["ut1_coverage"]
    assert info.source == ref["source"]
    assert info.first_mjd == ref["first_mjd"]
    assert info.last_mjd == ref["last_mjd"]
    assert info.entries == ref["entries"]
    assert info.first_jd_tt == hex_to_f64(ref["first_jd_tt_hex"])
    assert info.last_jd_tt == hex_to_f64(ref["last_jd_tt_hex"])


def test_gnss_week_tow():
    ref = FX["gnss_week_tow"]
    wt = sidereon.GnssWeekTow(
        sidereon.TimeScale.GPST, ref["input_week"], hex_to_f64(ref["input_tow_s_hex"])
    )
    assert wt.system == sidereon.TimeScale.GPST
    assert wt.week == ref["input_week"]

    norm = wt.normalized()
    assert norm.week == ref["normalized_week"]
    assert norm.tow_s == hex_to_f64(ref["normalized_tow_s_hex"])
    assert wt.unrolled_week(2) == ref["unrolled_week_2_rollovers"]
    assert norm == sidereon.GnssWeekTow(
        sidereon.TimeScale.GPST,
        ref["normalized_week"],
        hex_to_f64(ref["normalized_tow_s_hex"]),
    )


def test_time_scale_enum():
    assert sidereon.TimeScale.GPST.abbrev == "GPST"
    assert sidereon.TimeScale.UTC != sidereon.TimeScale.TAI
    assert repr(sidereon.TimeScale.TDB) == "TimeScale.TDB"


def test_transform_errors():
    epochs = _epochs_array()
    n = len(epochs)

    # Empty epochs grid.
    with pytest.raises(ValueError):
        sidereon.gcrs_to_itrs(np.zeros((0, 3)), np.asarray([], dtype=np.int64))

    # Wrong column count.
    with pytest.raises(ValueError):
        sidereon.gcrs_to_itrs(np.zeros((n, 2)), epochs)

    # Length mismatch between positions and epochs.
    with pytest.raises(ValueError):
        sidereon.gcrs_to_itrs(np.tile(SAMPLE_POS, (n + 1, 1)), epochs)

    # Bad calendar field.
    with pytest.raises(ValueError):
        sidereon.Instant.from_utc(2020, 13, 1)
