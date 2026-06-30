"""Built-once SGP4 fleet `Constellation` reproduces the single-`Tle` paths.

`Constellation` is the binding's fleet form of the WASM `Constellation` and
Elixir's `Sidereon.Constellation`: built from already-parsed `Tle` objects (no
text parsing, no I/O), it batches the core geometry over a shared ground station
and epoch grid. The reference for every value is the single-`Tle` path
(`Tle.propagate` / `Tle.look_angles` / `Tle.ground_track` / `Tle.find_passes`
and the `visible_from_satellites` free function), so a divergence is a wrapper
bug, not a new answer. The fleet fixture is the committed Vallado SGP4 subset
shared with `test_batch.py`.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES


def _load_fixture():
    with open(os.path.join(FIXTURES, "batch_fleet.json")) as fh:
        return json.load(fh)


def _opsmode(name):
    return {
        "afspc": sidereon.OpsMode.AFSPC,
        "improved": sidereon.OpsMode.IMPROVED,
    }[name]


def _tles(fx):
    mode = _opsmode(fx["opsmode"])
    return [sidereon.Tle(t["line1"], t["line2"], mode) for t in fx["tles"]]


def _epochs(fx):
    return np.asarray(fx["epochs_unix_us"], dtype=np.int64)


def _station():
    return sidereon.GroundStation(
        latitude_deg=40.0, longitude_deg=-75.0, altitude_m=0.0
    )


def _bits_equal(a, b):
    """True iff two float64 arrays are identical down to the bit."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    return a.shape == b.shape and bool((a.view(np.uint64) == b.view(np.uint64)).all())


def test_constructor_takes_parsed_tles_not_strings():
    fx = _load_fixture()
    tles = _tles(fx)
    const = sidereon.Constellation(tles)
    assert const.satellite_count == len(tles)
    assert len(const) == len(tles)
    assert "Constellation(" in repr(const)

    # The parsing boundary is `Tle(...)`: the constructor takes already-parsed
    # satellites, never raw TLE line strings.
    with pytest.raises(TypeError):
        sidereon.Constellation([(fx["tles"][0]["line1"], fx["tles"][0]["line2"])])
    with pytest.raises(TypeError):
        sidereon.Constellation([fx["tles"][0]["line1"]])


def test_catalog_numbers_preserve_fleet_order():
    fx = _load_fixture()
    tles = _tles(fx)
    const = sidereon.Constellation(tles)
    assert const.catalog_numbers == [t.catalog_number for t in tles]


def test_propagate_matches_per_tle_bitexact():
    fx = _load_fixture()
    tles = _tles(fx)
    epochs = _epochs(fx)
    const = sidereon.Constellation(tles)

    batch = const.propagate(epochs)
    assert isinstance(batch, sidereon.BatchPropagation)
    assert batch.satellite_count == len(tles)
    assert batch.epoch_count == len(epochs)
    assert batch.position_km.shape == (len(tles), len(epochs), 3)

    pos = batch.position_km
    vel = batch.velocity_km_s
    for i, tle in enumerate(tles):
        ref = tle.propagate(epochs)
        assert _bits_equal(pos[i], ref.position_km)
        assert _bits_equal(vel[i], ref.velocity_km_s)


def test_propagate_matches_propagate_batch_serial_bitexact():
    fx = _load_fixture()
    tles = _tles(fx)
    epochs = _epochs(fx)
    const = sidereon.Constellation(tles)

    batch = const.propagate(epochs)
    pairs = [(t["line1"], t["line2"]) for t in fx["tles"]]
    serial = sidereon.propagate_batch(
        pairs, epochs, opsmode=fx["opsmode"], parallel=False
    )
    assert _bits_equal(batch.position_km, serial.position_km)
    assert _bits_equal(batch.velocity_km_s, serial.velocity_km_s)


def test_look_angle_arcs_match_per_tle_bitexact():
    fx = _load_fixture()
    tles = _tles(fx)
    epochs = _epochs(fx)
    station = _station()
    const = sidereon.Constellation(tles)

    arcs = const.look_angle_arcs(station, epochs)
    assert len(arcs) == len(tles)
    for i, tle in enumerate(tles):
        ref = tle.look_angles(station, epochs)
        assert isinstance(arcs[i], sidereon.LookAngles)
        assert arcs[i].epoch_count == len(epochs)
        assert _bits_equal(arcs[i].azimuth_deg, ref.azimuth_deg)
        assert _bits_equal(arcs[i].elevation_deg, ref.elevation_deg)
        assert _bits_equal(arcs[i].range_km, ref.range_km)


def test_ground_tracks_match_per_tle():
    fx = _load_fixture()
    tles = _tles(fx)
    epochs = _epochs(fx)
    const = sidereon.Constellation(tles)

    tracks = const.ground_tracks(epochs)
    assert len(tracks) == len(tles)
    for i, tle in enumerate(tles):
        ref = tle.ground_track(epochs)  # list[Wgs84Geodetic], radians/metres
        gt = tracks[i]
        assert isinstance(gt, sidereon.GroundTrack)
        assert gt.epoch_count == len(epochs)
        lat = np.degrees([g.lat_rad for g in ref])
        lon = np.degrees([g.lon_rad for g in ref])
        alt = np.asarray([g.height_m for g in ref]) / 1000.0
        np.testing.assert_allclose(gt.latitude_deg, lat, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(gt.longitude_deg, lon, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(gt.altitude_km, alt, rtol=0.0, atol=1e-12)


def test_passes_match_per_tle_and_carry_fleet_index():
    fx = _load_fixture()
    tles = _tles(fx)
    epochs = _epochs(fx)
    station = _station()
    const = sidereon.Constellation(tles)

    start, end = int(epochs[0]), int(epochs[-1])
    fleet = const.passes(station, start, end, elevation_mask_deg=0.0)

    expected = []
    for i, tle in enumerate(tles):
        for p in tle.find_passes(station, start, end, elevation_mask_deg=0.0):
            expected.append(
                (
                    i,
                    p.aos_unix_us,
                    p.los_unix_us,
                    p.culmination_unix_us,
                    p.max_elevation_deg,
                )
            )

    got = [
        (
            fp.satellite_index,
            fp.aos_unix_us,
            fp.los_unix_us,
            fp.culmination_unix_us,
            fp.max_elevation_deg,
        )
        for fp in fleet
    ]
    assert got == expected
    # Every pass points back into the fleet by index, and duration is consistent.
    for fp in fleet:
        assert 0 <= fp.satellite_index < len(tles)
        assert fp.duration_s == (fp.los_unix_us - fp.aos_unix_us) / 1.0e6
        assert "FleetPass(" in repr(fp)


def test_passes_rejects_bad_window():
    fx = _load_fixture()
    const = sidereon.Constellation(_tles(fx))
    station = _station()
    with pytest.raises(ValueError):
        const.passes(station, 10, 10)
    with pytest.raises(ValueError):
        const.passes(station, 100, 200, step_seconds=0.0)


def test_visible_matches_free_function_and_sorts_descending():
    fx = _load_fixture()
    tles = _tles(fx)
    epochs = _epochs(fx)
    station = _station()
    const = sidereon.Constellation(tles)
    epoch = int(epochs[len(epochs) // 2])

    vis = const.visible(station, epoch, -90.0)
    # The constellation's ids are the catalog numbers, in fleet order.
    ref = sidereon.visible_from_satellites(
        tles, const.catalog_numbers, station, epoch, min_elevation_deg=-90.0
    )
    assert [v.catalog_number for v in vis] == [v.catalog_number for v in ref]
    for a, b in zip(vis, ref):
        assert a.azimuth_deg == b.azimuth_deg
        assert a.elevation_deg == b.elevation_deg
        assert a.range_km == b.range_km
        assert _bits_equal(a.position_km, b.position_km)

    # Sorted by elevation, highest first.
    elevs = [v.elevation_deg for v in vis]
    assert elevs == sorted(elevs, reverse=True)

    # A high mask filters the fleet down; default mask is 0.
    masked = const.visible(station, epoch, 89.999)
    assert len(masked) <= len(vis)
    default_mask = const.visible(station, epoch)
    explicit_zero = const.visible(station, epoch, 0.0)
    assert [v.catalog_number for v in default_mask] == [
        v.catalog_number for v in explicit_zero
    ]


def test_empty_constellation_and_empty_grid():
    empty = sidereon.Constellation([])
    assert empty.satellite_count == 0
    assert empty.catalog_numbers == []
    grid = np.asarray([1_151_193_600_000_000], dtype=np.int64)
    batch = empty.propagate(grid)
    assert batch.satellite_count == 0
    assert empty.look_angle_arcs(_station(), grid) == []
    assert empty.ground_tracks(grid) == []

    fx = _load_fixture()
    const = sidereon.Constellation(_tles(fx))
    no_epochs = np.asarray([], dtype=np.int64)
    batch = const.propagate(no_epochs)
    assert batch.satellite_count == len(fx["tles"])
    assert batch.epoch_count == 0
