"""GIL-free rayon-parallel fleet propagation reproduces the engine bit-for-bit.

The fixture ``batch_fleet.json`` carries a committed subset of the Vallado
SGP4-VER.TLE verification set (the eight satellites sharing TLE epoch 06176,
lifted from ``crates/sidereon-core/tests/sgp4_verification.json``) plus a shared
epoch grid in unix microseconds. The reference for every batched value is the
single-``Tle`` path (``Tle.propagate`` / ``Tle.look_angles``), which is itself
pinned to the engine's IEEE-754 goldens: row ``i`` of a batch must equal,
bit-for-bit, the single-satellite arc for TLE ``i``. The parallel and serial
batches must also be bit-identical to each other -- the rayon fan-out reorders
no arithmetic.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES

# The Vallado verification set carries element sets whose column-69 checksum
# disagrees with the line (catalog numbers 33333 to 33335); `twoline2rv` reads
# them, which the lenient policy mirrors. The strict default refuses them.
LENIENT = sidereon.TlePolicy.LENIENT


def _load_fixture():
    with open(os.path.join(FIXTURES, "batch_fleet.json")) as fh:
        return json.load(fh)


def _epochs_int64(fx):
    return np.asarray(fx["epochs_unix_us"], dtype=np.int64)


def _tle_pairs(fx):
    return [(t["line1"], t["line2"]) for t in fx["tles"]]


def _opsmode(name):
    return {
        "afspc": sidereon.OpsMode.AFSPC,
        "improved": sidereon.OpsMode.IMPROVED,
    }[name]


def _bits_equal(a, b):
    """True iff two float64 arrays are identical down to the bit (NaN-safe)."""
    return a.shape == b.shape and bool((a.view(np.uint64) == b.view(np.uint64)).all())


def test_propagate_batch_matches_single_tle_bits():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    opsmode = fx["opsmode"]

    batch = sidereon.propagate_batch(tles, epochs, opsmode=opsmode, policy=LENIENT)

    n_sats, n_epochs = len(tles), len(epochs)
    assert batch.satellite_count == n_sats
    assert batch.epoch_count == n_epochs
    assert len(batch) == n_sats
    assert isinstance(batch.position_km, np.ndarray)
    assert batch.position_km.dtype == np.float64
    assert batch.position_km.shape == (n_sats, n_epochs, 3)
    assert batch.velocity_km_s.shape == (n_sats, n_epochs, 3)
    assert "BatchPropagation(" in repr(batch)

    for i, (line1, line2) in enumerate(tles):
        ref = sidereon.Tle(line1, line2, opsmode=opsmode, policy=LENIENT).propagate(
            epochs
        )
        assert _bits_equal(batch.position_km[i], ref.position_km)
        assert _bits_equal(batch.velocity_km_s[i], ref.velocity_km_s)


def test_propagate_batch_parallel_equals_serial_bits():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    opsmode = fx["opsmode"]

    parallel = sidereon.propagate_batch(
        tles, epochs, opsmode=opsmode, parallel=True, policy=LENIENT
    )
    serial = sidereon.propagate_batch(
        tles, epochs, opsmode=opsmode, parallel=False, policy=LENIENT
    )

    assert _bits_equal(parallel.position_km, serial.position_km)
    assert _bits_equal(parallel.velocity_km_s, serial.velocity_km_s)


def test_propagate_batch_opsmode_enum_matches_string_alias():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)

    legacy = sidereon.propagate_batch(
        tles, epochs, opsmode=fx["opsmode"], policy=LENIENT
    )
    enum_batch = sidereon.propagate_batch(
        tles, epochs, opsmode=_opsmode(fx["opsmode"]), policy=LENIENT
    )

    assert _bits_equal(enum_batch.position_km, legacy.position_km)
    assert _bits_equal(enum_batch.velocity_km_s, legacy.velocity_km_s)


def test_look_angles_batch_matches_single_tle_bits():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    opsmode = fx["opsmode"]
    station = sidereon.GroundStation(
        latitude_deg=51.5074, longitude_deg=-0.1278, altitude_m=11.0
    )

    batch = sidereon.look_angles_batch(
        tles, station, epochs, opsmode=opsmode, policy=LENIENT
    )

    n_sats, n_epochs = len(tles), len(epochs)
    assert batch.satellite_count == n_sats
    assert batch.epoch_count == n_epochs
    assert batch.azimuth_deg.shape == (n_sats, n_epochs)
    assert batch.elevation_deg.shape == (n_sats, n_epochs)
    assert batch.range_km.shape == (n_sats, n_epochs)
    assert "BatchLookAngles(" in repr(batch)

    for i, (line1, line2) in enumerate(tles):
        ref = sidereon.Tle(line1, line2, opsmode=opsmode, policy=LENIENT).look_angles(
            station, epochs
        )
        assert _bits_equal(batch.azimuth_deg[i], ref.azimuth_deg)
        assert _bits_equal(batch.elevation_deg[i], ref.elevation_deg)
        assert _bits_equal(batch.range_km[i], ref.range_km)


def test_look_angles_batch_parallel_equals_serial_bits():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    opsmode = fx["opsmode"]
    station = sidereon.GroundStation(latitude_deg=40.0, longitude_deg=-105.0)

    parallel = sidereon.look_angles_batch(
        tles, station, epochs, opsmode=opsmode, parallel=True, policy=LENIENT
    )
    serial = sidereon.look_angles_batch(
        tles, station, epochs, opsmode=opsmode, parallel=False, policy=LENIENT
    )

    assert _bits_equal(parallel.azimuth_deg, serial.azimuth_deg)
    assert _bits_equal(parallel.elevation_deg, serial.elevation_deg)
    assert _bits_equal(parallel.range_km, serial.range_km)


def test_look_angles_batch_opsmode_enum_matches_string_alias():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    station = sidereon.GroundStation(latitude_deg=40.0, longitude_deg=-105.0)

    legacy = sidereon.look_angles_batch(
        tles, station, epochs, opsmode=fx["opsmode"], policy=LENIENT
    )
    enum_batch = sidereon.look_angles_batch(
        tles, station, epochs, opsmode=_opsmode(fx["opsmode"]), policy=LENIENT
    )

    assert _bits_equal(enum_batch.azimuth_deg, legacy.azimuth_deg)
    assert _bits_equal(enum_batch.elevation_deg, legacy.elevation_deg)
    assert _bits_equal(enum_batch.range_km, legacy.range_km)


def test_ground_tracks_batch_matches_single_tle_bits():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    opsmode = fx["opsmode"]

    batch = sidereon.ground_tracks_batch(tles, epochs, opsmode=opsmode, policy=LENIENT)

    n_sats, n_epochs = len(tles), len(epochs)
    assert batch.satellite_count == n_sats
    assert batch.epoch_count == n_epochs
    assert len(batch) == n_sats
    assert batch.latitude_deg.shape == (n_sats, n_epochs)
    assert batch.longitude_deg.shape == (n_sats, n_epochs)
    assert batch.altitude_km.shape == (n_sats, n_epochs)
    assert "BatchGroundTracks(" in repr(batch)

    for i, (line1, line2) in enumerate(tles):
        ref = sidereon.Tle(line1, line2, opsmode=opsmode, policy=LENIENT).ground_track(
            epochs
        )
        lat = np.asarray([np.degrees(g.lat_rad) for g in ref], dtype=np.float64)
        lon = np.asarray([np.degrees(g.lon_rad) for g in ref], dtype=np.float64)
        alt = np.asarray([g.height_m / 1000.0 for g in ref], dtype=np.float64)
        assert _bits_equal(batch.latitude_deg[i], lat)
        assert _bits_equal(batch.longitude_deg[i], lon)
        assert _bits_equal(batch.altitude_km[i], alt)


def test_ground_tracks_batch_parallel_equals_serial_bits():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    opsmode = fx["opsmode"]

    parallel = sidereon.ground_tracks_batch(
        tles, epochs, opsmode=opsmode, parallel=True, policy=LENIENT
    )
    serial = sidereon.ground_tracks_batch(
        tles, epochs, opsmode=opsmode, parallel=False, policy=LENIENT
    )

    assert _bits_equal(parallel.latitude_deg, serial.latitude_deg)
    assert _bits_equal(parallel.longitude_deg, serial.longitude_deg)
    assert _bits_equal(parallel.altitude_km, serial.altitude_km)


def test_ground_tracks_batch_opsmode_enum_matches_string_alias():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)

    legacy = sidereon.ground_tracks_batch(
        tles, epochs, opsmode=fx["opsmode"], policy=LENIENT
    )
    enum_batch = sidereon.ground_tracks_batch(
        tles, epochs, opsmode=_opsmode(fx["opsmode"]), policy=LENIENT
    )

    assert _bits_equal(enum_batch.latitude_deg, legacy.latitude_deg)
    assert _bits_equal(enum_batch.longitude_deg, legacy.longitude_deg)
    assert _bits_equal(enum_batch.altitude_km, legacy.altitude_km)


def test_batch_look_angles_and_ground_tracks_reject_out_of_range_instants():
    fx = _load_fixture()
    tles = _tle_pairs(fx)[:2]
    station = sidereon.GroundStation(latitude_deg=40.0, longitude_deg=-105.0)
    msg = "invalid look-angle input datetime: invalid UTC instant"
    for extreme in (-(1 << 63), (1 << 63) - 1):
        epochs = np.asarray([extreme], dtype=np.int64)
        with pytest.raises(sidereon.SolveError, match=msg):
            sidereon.look_angles_batch(
                tles, station, epochs, opsmode=fx["opsmode"], policy=LENIENT
            )
        with pytest.raises(sidereon.SolveError, match=msg):
            sidereon.ground_tracks_batch(
                tles, epochs, opsmode=fx["opsmode"], policy=LENIENT
            )


def test_empty_fleet_returns_empty_batches():
    fx = _load_fixture()
    epochs = _epochs_int64(fx)
    station = sidereon.GroundStation(latitude_deg=40.0, longitude_deg=-105.0)

    prop = sidereon.propagate_batch([], epochs)
    looks = sidereon.look_angles_batch([], station, epochs)
    tracks = sidereon.ground_tracks_batch([], epochs)

    assert prop.satellite_count == 0
    assert prop.epoch_count == len(epochs)
    assert prop.position_km.shape == (0, len(epochs), 3)
    assert prop.velocity_km_s.shape == (0, len(epochs), 3)
    assert looks.satellite_count == 0
    assert looks.epoch_count == len(epochs)
    assert looks.azimuth_deg.shape == (0, len(epochs))
    assert looks.elevation_deg.shape == (0, len(epochs))
    assert looks.range_km.shape == (0, len(epochs))
    assert tracks.satellite_count == 0
    assert tracks.epoch_count == len(epochs)
    assert tracks.latitude_deg.shape == (0, len(epochs))
    assert tracks.longitude_deg.shape == (0, len(epochs))
    assert tracks.altitude_km.shape == (0, len(epochs))


def test_empty_epochs_return_empty_batches():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = np.asarray([], dtype=np.int64)
    station = sidereon.GroundStation(latitude_deg=40.0, longitude_deg=-105.0)

    prop = sidereon.propagate_batch(tles, epochs, policy=LENIENT)
    looks = sidereon.look_angles_batch(tles, station, epochs, policy=LENIENT)
    tracks = sidereon.ground_tracks_batch(tles, epochs, policy=LENIENT)

    assert prop.satellite_count == len(tles)
    assert prop.epoch_count == 0
    assert prop.position_km.shape == (len(tles), 0, 3)
    assert prop.velocity_km_s.shape == (len(tles), 0, 3)
    assert looks.satellite_count == len(tles)
    assert looks.epoch_count == 0
    assert looks.azimuth_deg.shape == (len(tles), 0)
    assert looks.elevation_deg.shape == (len(tles), 0)
    assert looks.range_km.shape == (len(tles), 0)
    assert tracks.satellite_count == len(tles)
    assert tracks.epoch_count == 0
    assert tracks.latitude_deg.shape == (len(tles), 0)
    assert tracks.longitude_deg.shape == (len(tles), 0)
    assert tracks.altitude_km.shape == (len(tles), 0)


def test_bad_tle_in_fleet_raises_with_index():
    fx = _load_fixture()
    tles = _tle_pairs(fx)
    epochs = _epochs_int64(fx)
    # Corrupt the second satellite; the error must name its index.
    tles[1] = ("not a tle", "also not a tle")
    with pytest.raises(sidereon.SidereonError) as excinfo:
        sidereon.propagate_batch(tles, epochs, opsmode=fx["opsmode"])
    assert "satellite 1" in str(excinfo.value)


def test_strict_policy_refuses_the_verification_set_checksum_mismatch():
    fx = _load_fixture()
    with pytest.raises(sidereon.TleParseError, match="checksum"):
        sidereon.propagate_batch(_tle_pairs(fx), _epochs_int64(fx))
