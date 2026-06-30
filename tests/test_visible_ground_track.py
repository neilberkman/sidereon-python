"""Constellation visibility (opsmode-preserving) and ground-track wrappers.

`visible_from_satellites` and `Tle.ground_track` are thin marshals over the core
`passes::visible_from_satellites` / `passes::ground_track`. They are cross-checked
against the binding's own already-validated paths (`Tle.look_angles`,
`Tle.propagate`, and the frame transform chain), so a divergence is a wrapper
bug, not a new answer.
"""

import numpy as np
import pytest
import sidereon

# ISS (NORAD 25544), epoch 2018-07-03. Near-earth: opsmode-insensitive in this
# core, used for the consistency checks.
_ISS_L1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
_ISS_L2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"
_ISS_EPOCH_US = 1_530_576_000_000_000  # 2018-07-03 00:00:00 UTC

# NORAD 23599: deep-space satellite for which this core's SGP4 OpsMode branch is
# observable (afspc vs improved diverge by ~1.1 km in TEME).
_DS_L1 = "1 23599U 95029B   06171.76535463  .00085586  12891-6  12956-2 0  2905"
_DS_L2 = "2 23599   6.9327   0.2849 5782022 274.4436  25.2425  4.47796565123555"
_DS_EPOCH_US = 1_150_827_726_640_032  # 2006-06-20 18:22 UTC


def _station():
    return sidereon.GroundStation(
        latitude_deg=40.0, longitude_deg=-75.0, altitude_m=0.0
    )


def test_visible_matches_look_angles_and_propagate():
    """Each VisibleSatellite's geometry equals the satellite's own look_angles /
    propagate at the same instant (bit-exact: same core path)."""
    tle = sidereon.Tle(_ISS_L1, _ISS_L2)
    st = _station()
    epoch = _ISS_EPOCH_US

    # No mask so the satellite is always returned regardless of geometry.
    vis = sidereon.visible_from_satellites(
        [tle], ["ISS"], st, epoch, min_elevation_deg=-90.0
    )
    assert len(vis) == 1
    v = vis[0]
    assert v.catalog_number == "ISS"
    assert "VisibleSatellite(" in repr(v)

    grid = np.asarray([epoch], dtype=np.int64)
    look = tle.look_angles(st, grid)
    assert v.azimuth_deg == look.azimuth_deg[0]
    assert v.elevation_deg == look.elevation_deg[0]
    assert v.range_km == look.range_km[0]

    prop = tle.propagate(grid)
    assert isinstance(v.position_km, np.ndarray)
    assert v.position_km.shape == (3,)
    assert np.array_equal(v.position_km, prop.position_km[0])


def test_visible_honors_per_satellite_opsmode():
    """The afspc and improved builds of the deep-space TLE must produce different
    VisibleSatellite geometry -- proof opsmode is preserved end-to-end (the whole
    point versus the element-based AFSPC-hardcoded path). Each entry also matches
    its own look_angles."""
    st = sidereon.GroundStation(latitude_deg=-40.0, longitude_deg=0.0, altitude_m=0.0)
    afspc = sidereon.Tle(_DS_L1, _DS_L2, "afspc")
    improved = sidereon.Tle(_DS_L1, _DS_L2, "improved")
    # The deep-space periodics that the opsmode branch governs are zero at the TLE
    # epoch; step 12 h out so the two modes have actually diverged.
    epoch = _DS_EPOCH_US + 12 * 3_600 * 1_000_000

    vis = sidereon.visible_from_satellites(
        [afspc, improved], ["afspc", "improved"], st, epoch, min_elevation_deg=-90.0
    )
    assert len(vis) == 2
    by_id = {v.catalog_number: v for v in vis}

    # OpsMode is observable for this satellite, so the two builds diverge.
    assert by_id["afspc"].elevation_deg != by_id["improved"].elevation_deg
    grid = np.asarray([epoch], dtype=np.int64)
    assert not np.array_equal(by_id["afspc"].position_km, by_id["improved"].position_km)

    # Each entry agrees with its own satellite's look-angle geometry.
    for tle, key in ((afspc, "afspc"), (improved, "improved")):
        look = tle.look_angles(st, grid)
        assert by_id[key].azimuth_deg == look.azimuth_deg[0]
        assert by_id[key].elevation_deg == look.elevation_deg[0]
        assert by_id[key].range_km == look.range_km[0]


def test_visible_filters_and_sorts_by_elevation_descending():
    tle = sidereon.Tle(_ISS_L1, _ISS_L2)
    st = _station()

    # Sweep a day so some epoch puts the ISS above the horizon.
    epochs = [_ISS_EPOCH_US + k * 600_000_000 for k in range(150)]
    seen_above = False
    for ep in epochs:
        below = sidereon.visible_from_satellites(
            [tle], ["ISS"], st, ep, min_elevation_deg=10.0
        )
        full = sidereon.visible_from_satellites(
            [tle], ["ISS"], st, ep, min_elevation_deg=-90.0
        )
        assert len(full) == 1
        if full[0].elevation_deg >= 10.0:
            assert len(below) == 1
            seen_above = True
        else:
            assert len(below) == 0
    assert seen_above

    # Sorting: a multi-satellite call is returned by descending elevation.
    ds = sidereon.Tle(_DS_L1, _DS_L2)
    multi = sidereon.visible_from_satellites(
        [tle, ds], ["iss", "ds"], st, _ISS_EPOCH_US, min_elevation_deg=-90.0
    )
    elevs = [v.elevation_deg for v in multi]
    assert elevs == sorted(elevs, reverse=True)


def test_visible_ids_length_mismatch_raises():
    tle = sidereon.Tle(_ISS_L1, _ISS_L2)
    st = _station()
    with pytest.raises(sidereon.SidereonError):
        sidereon.visible_from_satellites([tle], ["a", "b"], st, _ISS_EPOCH_US)


def test_ground_track_matches_frame_transform_chain():
    """Ground track equals propagate -> teme_to_gcrs -> gcrs_to_itrs ->
    ecef_to_geodetic (the same core transforms ground_track composes, with the
    direct km path skyfield_compat=False)."""
    tle = sidereon.Tle(_ISS_L1, _ISS_L2)
    epochs = np.asarray(
        [_ISS_EPOCH_US + k * 120_000_000 for k in range(5)], dtype=np.int64
    )

    track = tle.ground_track(epochs)
    assert len(track) == len(epochs)

    prop = tle.propagate(epochs)
    gcrs = sidereon.teme_to_gcrs(
        prop.position_km, prop.velocity_km_s, epochs, skyfield_compat=False
    )
    itrs = sidereon.gcrs_to_itrs(gcrs.position_km, epochs, skyfield_compat=False)
    lla = sidereon.ecef_to_geodetic(itrs)  # columns: [lat_deg, lon_deg, alt_km]

    for i, g in enumerate(track):
        assert type(g).__name__ == "Wgs84Geodetic"
        assert np.isclose(np.degrees(g.lat_rad), lla[i, 0], atol=1e-9, rtol=0.0)
        assert np.isclose(np.degrees(g.lon_rad), lla[i, 1], atol=1e-9, rtol=0.0)
        assert np.isclose(g.height_m, lla[i, 2] * 1000.0, atol=1e-6, rtol=0.0)


def test_ground_track_altitude_is_physical_for_iss():
    tle = sidereon.Tle(_ISS_L1, _ISS_L2)
    epochs = np.asarray(
        [_ISS_EPOCH_US + k * 300_000_000 for k in range(8)], dtype=np.int64
    )
    track = tle.ground_track(epochs)
    for g in track:
        # ISS ellipsoidal height stays in a low-earth-orbit band.
        assert 300_000.0 < g.height_m < 460_000.0
        assert -np.pi / 2 <= g.lat_rad <= np.pi / 2


def test_ground_track_empty_epochs_returns_empty():
    tle = sidereon.Tle(_ISS_L1, _ISS_L2)
    track = tle.ground_track(np.asarray([], dtype=np.int64))
    assert track == []
