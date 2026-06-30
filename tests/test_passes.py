"""Dense-sample pass finding reproduces the engine reference arc.

The fixture ``pass_finder.json`` is emitted by the crate's validated arc test
(``SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test pass_finder_arc``);
it carries a committed real ISS TLE, a real ground station, a window, the finder
options, and the engine's found passes (AOS/LOS/culmination unix microseconds and
the culmination elevation). The binding parses the same TLE and must return the
same passes -- a wrapper that diverges is a wrapper bug, not a new answer.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES, hex_to_f64

MAX_ELEVATION_TOL_DEG = 1.0e-9


def _assert_reference_elevation_matches(got_deg, expected):
    assert (
        abs(got_deg - hex_to_f64(expected["max_elevation_deg_hex"]))
        < MAX_ELEVATION_TOL_DEG
    )


def _load_fixture():
    with open(os.path.join(FIXTURES, "pass_finder.json")) as fh:
        return json.load(fh)


def _find(fx):
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    station = sidereon.GroundStation(
        latitude_deg=fx["station"]["latitude_deg"],
        longitude_deg=fx["station"]["longitude_deg"],
        altitude_m=fx["station"]["altitude_m"],
    )
    opts = fx["options"]
    return tle.find_passes(
        station,
        fx["window"]["start_unix_us"],
        fx["window"]["end_unix_us"],
        elevation_mask_deg=opts["elevation_mask_deg"],
        step_seconds=opts["coarse_step_seconds"],
        time_tolerance_s=opts["time_tolerance_seconds"],
    )


def _tle_and_station(fx):
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    station = sidereon.GroundStation(
        latitude_deg=fx["station"]["latitude_deg"],
        longitude_deg=fx["station"]["longitude_deg"],
        altitude_m=fx["station"]["altitude_m"],
    )
    return tle, station


def test_passes_match_reference_bits():
    fx = _load_fixture()
    passes = _find(fx)

    assert len(passes) == len(fx["passes"])
    assert len(passes) >= 2

    for got, expected in zip(passes, fx["passes"]):
        assert got.aos_unix_us == expected["aos_unix_us"]
        assert got.los_unix_us == expected["los_unix_us"]
        assert got.culmination_unix_us == expected["culmination_unix_us"]
        _assert_reference_elevation_matches(got.max_elevation_deg, expected)
        # AOS precedes culmination precedes LOS; duration is positive.
        assert got.aos_unix_us <= got.culmination_unix_us <= got.los_unix_us
        assert got.duration_s > 0.0
        assert "SatellitePass(" in repr(got)


def test_visibility_series_matches_reference_passes_and_culminations():
    fx = _load_fixture()
    tle, station = _tle_and_station(fx)
    opts = fx["options"]
    epochs = np.asarray(
        [fx["window"]["start_unix_us"]]
        + [p["culmination_unix_us"] for p in fx["passes"]]
        + [fx["window"]["end_unix_us"]],
        dtype=np.int64,
    )

    series = tle.visibility_series(
        station,
        epochs,
        elevation_mask_deg=opts["elevation_mask_deg"],
        step_seconds=opts["coarse_step_seconds"],
        time_tolerance_s=opts["time_tolerance_seconds"],
    )
    look = tle.look_angles(station, epochs)

    assert isinstance(series.epoch_unix_us, np.ndarray)
    assert series.epoch_unix_us.dtype == np.int64
    assert np.array_equal(series.epoch_unix_us, epochs)
    assert series.azimuth_deg.dtype == np.float64
    assert series.elevation_deg.shape == epochs.shape
    assert series.range_km.shape == epochs.shape
    assert series.visible.dtype == np.bool_
    assert series.visible.shape == epochs.shape
    assert series.epoch_count == len(epochs)
    assert len(series) == len(epochs)
    assert "VisibilitySeries(" in repr(series)

    assert np.array_equal(series.azimuth_deg, look.azimuth_deg)
    assert np.array_equal(series.elevation_deg, look.elevation_deg)
    assert np.array_equal(series.range_km, look.range_km)

    assert series.pass_count == len(fx["passes"])
    assert len(series.passes) == len(fx["passes"])
    for got, expected in zip(series.passes, fx["passes"]):
        assert got.aos_unix_us == expected["aos_unix_us"]
        assert got.los_unix_us == expected["los_unix_us"]
        assert got.culmination_unix_us == expected["culmination_unix_us"]
        _assert_reference_elevation_matches(got.max_elevation_deg, expected)

    for index, expected in enumerate(fx["passes"], start=1):
        assert bool(series.visible[index])
        _assert_reference_elevation_matches(series.elevation_deg[index], expected)


def test_higher_mask_keeps_fewer_passes():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    station = sidereon.GroundStation(
        latitude_deg=fx["station"]["latitude_deg"],
        longitude_deg=fx["station"]["longitude_deg"],
        altitude_m=fx["station"]["altitude_m"],
    )
    start = fx["window"]["start_unix_us"]
    end = fx["window"]["end_unix_us"]

    low = tle.find_passes(
        station, start, end, elevation_mask_deg=0.0, step_seconds=10.0
    )
    high = tle.find_passes(
        station, start, end, elevation_mask_deg=40.0, step_seconds=10.0
    )

    assert len(high) <= len(low)
    for p in high:
        assert p.max_elevation_deg >= 40.0


def test_bad_window_raises():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    station = sidereon.GroundStation(latitude_deg=51.5, longitude_deg=-0.1)
    with pytest.raises(ValueError):
        tle.find_passes(station, 1_000, 1_000)  # end == start


def test_non_positive_step_raises():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    station = sidereon.GroundStation(latitude_deg=51.5, longitude_deg=-0.1)
    with pytest.raises(ValueError):
        tle.find_passes(station, 0, 1_000_000, step_seconds=0.0)


def _passes_key(passes):
    return [(p.aos_unix_us, p.culmination_unix_us, p.los_unix_us) for p in passes]


# ISS (NORAD 25544), epoch 2018-07-03. The ISS is a near-earth satellite, and
# this core's SGP4 only branches on OpsMode in the deep-space periodics path
# (`nodep < 0 && opsmode == 'a'`; the AFSPC gsto branch is commented out and gsto
# is unused for near-earth sats). So for the ISS, afspc and improved are
# bit-identical -- a "differ per mode" assertion is impossible regardless of the
# fix. The ISS still exercises the *consistency* half of the fix.
_ISS_L1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
_ISS_L2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"
_ISS_START_US = 1_530_576_000_000_000  # 2018-07-03 00:00:00 UTC
_ISS_END_US = _ISS_START_US + 24 * 3_600 * 1_000_000  # +24h

# NORAD 23599 from the core's own SGP4 verification set: a deep-space satellite
# (period ~322 min, e=0.578) for which this core's OpsMode branch *is* observable
# -- afspc vs improved diverge by ~1.1 km in TEME. This is the TLE that can prove
# `find_passes`/`visibility_series` honor the Tle's OpsMode.
_DS_L1 = "1 23599U 95029B   06171.76535463  .00085586  12891-6  12956-2 0  2905"
_DS_L2 = "2 23599   6.9327   0.2849 5782022 274.4436  25.2425  4.47796565123555"
_DS_START_US = 1_150_827_726_640_032  # TLE epoch 2006-06-20 18:22 UTC
_DS_END_US = _DS_START_US + 24 * 3_600 * 1_000_000  # +24h


def test_pass_finder_honors_tle_opsmode():
    """find_passes/visibility_series must use the Tle satellite's own OpsMode.

    Regression for the bug where these routed through the core ElementSet pass
    finder, which rebuilds the satellite with a hardcoded OpsMode::Afspc, so an
    `improved` Tle silently got AFSPC passes -- inconsistent with its look-angle
    geometry (which already went through the satellite-based path).

    Uses NORAD 23599 (deep-space) where the core's OpsMode branch is observable;
    a southern mid-latitude station (40S) that the orbit actually passes over.
    """
    station = sidereon.GroundStation(
        latitude_deg=-40.0, longitude_deg=0.0, altitude_m=0.0
    )
    afspc = sidereon.Tle(_DS_L1, _DS_L2, "afspc")
    improved = sidereon.Tle(_DS_L1, _DS_L2, "improved")

    afspc_passes = afspc.find_passes(
        station, _DS_START_US, _DS_END_US, elevation_mask_deg=5.0, step_seconds=60.0
    )
    improved_passes = improved.find_passes(
        station, _DS_START_US, _DS_END_US, elevation_mask_deg=5.0, step_seconds=60.0
    )

    # Window actually contains passes to compare.
    assert len(afspc_passes) >= 1
    assert len(improved_passes) >= 1

    # The two modes produce different SGP4 trajectories for this satellite, so the
    # pass timings must differ. Pre-fix both went through the hardcoded-AFSPC
    # ElementSet finder and were identical -- exactly the bug.
    assert _passes_key(afspc_passes) != _passes_key(improved_passes)

    # Consistency: the improved Tle's passes must agree with its own look-angle
    # geometry. Sample look_angles at the improved finder's culmination instants;
    # the elevation there must equal the pass max_elevation_deg. Pre-fix the
    # finder reported AFSPC culminations/elevations while look_angles returned
    # improved geometry, so these diverged.
    improved_culms = np.asarray(
        [p.culmination_unix_us for p in improved_passes], dtype=np.int64
    )
    improved_looks = improved.look_angles(station, improved_culms)
    for pass_, el in zip(improved_passes, improved_looks.elevation_deg):
        assert abs(el - pass_.max_elevation_deg) < 1.0e-9

    # Sharpen the proof: the *afspc* look-angle elevations at those same instants
    # differ from the improved ones -- so honoring opsmode is not a no-op here.
    afspc_looks = afspc.look_angles(station, improved_culms)
    assert np.any(
        np.abs(
            np.asarray(afspc_looks.elevation_deg)
            - np.asarray(improved_looks.elevation_deg)
        )
        > 1.0e-6
    )

    # visibility_series must take the same satellite-based path as find_passes.
    grid = np.asarray(
        [_DS_START_US] + list(improved_culms) + [_DS_END_US], dtype=np.int64
    )
    series = improved.visibility_series(
        station, grid, elevation_mask_deg=5.0, step_seconds=60.0
    )
    assert _passes_key(series.passes) == _passes_key(improved_passes)


def test_iss_passes_consistent_with_look_angles_and_opsmode_insensitive():
    """ISS (near-earth) is OpsMode-insensitive in this core, and its passes are
    consistent with its look-angles -- the consistency half of the opsmode fix.

    ISS TLE, 24h window, mid-latitude (40N) station per the regression spec.
    """
    station = sidereon.GroundStation(
        latitude_deg=40.0, longitude_deg=-75.0, altitude_m=0.0
    )
    afspc = sidereon.Tle(_ISS_L1, _ISS_L2, "afspc")
    improved = sidereon.Tle(_ISS_L1, _ISS_L2, "improved")

    afspc_passes = afspc.find_passes(
        station, _ISS_START_US, _ISS_END_US, elevation_mask_deg=10.0, step_seconds=30.0
    )
    improved_passes = improved.find_passes(
        station, _ISS_START_US, _ISS_END_US, elevation_mask_deg=10.0, step_seconds=30.0
    )
    assert len(improved_passes) >= 2

    # The ISS is near-earth, so OpsMode has no observable effect in this core:
    # afspc and improved must be bit-identical (a correct property, not the bug).
    assert _passes_key(afspc_passes) == _passes_key(improved_passes)

    # The improved Tle's passes are consistent with its own look-angle geometry.
    improved_culms = np.asarray(
        [p.culmination_unix_us for p in improved_passes], dtype=np.int64
    )
    improved_looks = improved.look_angles(station, improved_culms)
    for pass_, el in zip(improved_passes, improved_looks.elevation_deg):
        assert abs(el - pass_.max_elevation_deg) < 1.0e-9

    # find_passes and visibility_series share the satellite-based path.
    grid = np.asarray(
        [_ISS_START_US] + list(improved_culms) + [_ISS_END_US], dtype=np.int64
    )
    series = improved.visibility_series(
        station, grid, elevation_mask_deg=10.0, step_seconds=30.0
    )
    assert _passes_key(series.passes) == _passes_key(improved_passes)


def test_visibility_series_rejects_non_increasing_grid():
    fx = _load_fixture()
    tle, station = _tle_and_station(fx)
    epochs = np.asarray([fx["window"]["start_unix_us"]] * 2, dtype=np.int64)
    with pytest.raises(ValueError):
        tle.visibility_series(station, epochs)
