"""Batched SGP4 propagation + topocentric look angles reproduce the engine.

The fixture `sgp4_topocentric.json` is emitted by the crate's validated arc test
(`SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test
sgp4_topocentric_arc`); it carries a committed ISS TLE, a ground station, an
epoch grid (unix microseconds), and the engine's reference TEME states and
look angles as IEEE-754 hex bits. The binding parses the same TLE, propagates
the same epochs, and must return the identical bits.
"""

import json
import os

import numpy as np
import sidereon
from _helpers import FIXTURES, hex_to_f64


def _load_fixture():
    with open(os.path.join(FIXTURES, "sgp4_topocentric.json")) as fh:
        return json.load(fh)


def _epochs_int64(fx):
    return np.asarray([e["unix_microseconds"] for e in fx["epochs"]], dtype=np.int64)


def _opsmode(name):
    return {
        "afspc": sidereon.OpsMode.AFSPC,
        "improved": sidereon.OpsMode.IMPROVED,
    }[name]


def test_propagation_matches_reference_bits():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"], opsmode=fx["opsmode"])
    epochs = _epochs_int64(fx)

    prop = tle.propagate(epochs)

    assert isinstance(prop.position_km, np.ndarray)
    assert prop.position_km.dtype == np.float64
    assert prop.position_km.shape == (len(fx["epochs"]), 3)
    assert prop.velocity_km_s.shape == (len(fx["epochs"]), 3)
    assert prop.epoch_count == len(fx["epochs"])
    assert len(prop) == len(fx["epochs"])

    for i, epoch in enumerate(fx["epochs"]):
        expected_pos = [hex_to_f64(h) for h in epoch["position_km_hex"]]
        expected_vel = [hex_to_f64(h) for h in epoch["velocity_km_s_hex"]]
        for axis in range(3):
            assert prop.position_km[i, axis] == expected_pos[axis]
            assert prop.velocity_km_s[i, axis] == expected_vel[axis]


def test_look_angles_match_reference_bits():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"], opsmode=fx["opsmode"])
    epochs = _epochs_int64(fx)
    station = sidereon.GroundStation(
        latitude_deg=fx["station"]["latitude_deg"],
        longitude_deg=fx["station"]["longitude_deg"],
        altitude_m=fx["station"]["altitude_m"],
    )

    look = tle.look_angles(station, epochs)

    assert look.azimuth_deg.dtype == np.float64
    assert look.azimuth_deg.shape == (len(fx["epochs"]),)
    assert look.epoch_count == len(fx["epochs"])

    for i, epoch in enumerate(fx["epochs"]):
        assert look.azimuth_deg[i] == hex_to_f64(epoch["azimuth_deg_hex"])
        assert look.elevation_deg[i] == hex_to_f64(epoch["elevation_deg_hex"])
        assert look.range_km[i] == hex_to_f64(epoch["range_km_hex"])


def test_opsmode_enum_matches_string_alias():
    fx = _load_fixture()
    epochs = _epochs_int64(fx)
    legacy = sidereon.Tle(
        fx["tle"]["line1"], fx["tle"]["line2"], opsmode=fx["opsmode"]
    ).propagate(epochs)
    enum_prop = sidereon.Tle(
        fx["tle"]["line1"], fx["tle"]["line2"], opsmode=_opsmode(fx["opsmode"])
    ).propagate(epochs)

    assert np.array_equal(enum_prop.position_km, legacy.position_km)
    assert np.array_equal(enum_prop.velocity_km_s, legacy.velocity_km_s)
    assert sidereon.OpsMode.AFSPC.label == "afspc"
    assert repr(sidereon.OpsMode.IMPROVED) == "OpsMode.IMPROVED"


def test_tle_properties_expose_elements():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    assert tle.catalog_number == "25544"
    assert tle.epoch_year == 2018
    assert tle.inclination_deg == 51.6414
    assert "Tle(" in repr(tle)


def test_bad_tle_raises():
    import pytest

    with pytest.raises(sidereon.SidereonError):
        sidereon.Tle("not a tle", "also not a tle")


def test_unknown_opsmode_raises():
    import pytest

    with pytest.raises(ValueError):
        sidereon.Tle(
            "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993",
            "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106",
            opsmode="bogus",
        )


def test_empty_epochs_return_empty_tle_results():
    fx = _load_fixture()
    tle = sidereon.Tle(fx["tle"]["line1"], fx["tle"]["line2"])
    epochs = np.asarray([], dtype=np.int64)
    station = sidereon.GroundStation(
        latitude_deg=fx["station"]["latitude_deg"],
        longitude_deg=fx["station"]["longitude_deg"],
        altitude_m=fx["station"]["altitude_m"],
    )

    prop = tle.propagate(epochs)
    look = tle.look_angles(station, epochs)

    assert prop.epoch_count == 0
    assert len(prop) == 0
    assert prop.position_km.shape == (0, 3)
    assert prop.velocity_km_s.shape == (0, 3)
    assert look.epoch_count == 0
    assert len(look) == 0
    assert look.azimuth_deg.shape == (0,)
    assert look.elevation_deg.shape == (0,)
    assert look.range_km.shape == (0,)
