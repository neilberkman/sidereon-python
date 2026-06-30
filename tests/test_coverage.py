"""One-epoch satellite/station coverage grid.

`coverage_look_angles` and the `CoverageGrid` reductions are thin wrappers over
`sidereon_core::astro::coverage`. The grid is built from the same scalar
look-angle kernel `Tle.look_angles` uses, so the test asserts each cell is
bit-exact to that per-pair kernel and that the visibility reductions match a
hand recomputation from the grid.
"""

import numpy as np
import sidereon

# Two ISS TLEs (identical here, so both rows are the same satellite) and two
# stations: London and New York.
ISS_L1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
ISS_L2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"

EPOCH_US = int(np.datetime64("2018-07-03T20:00:00", "us").astype("int64"))


def _tles():
    return [sidereon.Tle(ISS_L1, ISS_L2), sidereon.Tle(ISS_L1, ISS_L2)]


def _stations():
    return [
        sidereon.GroundStation(latitude_deg=51.5, longitude_deg=-0.1, altitude_m=11.0),
        sidereon.GroundStation(latitude_deg=40.7, longitude_deg=-74.0, altitude_m=10.0),
    ]


def test_grid_cells_match_scalar_look_angle_kernel():
    tles = _tles()
    stations = _stations()
    grid = sidereon.coverage_look_angles(tles, stations, EPOCH_US)

    assert grid.n_satellites == 2
    assert grid.n_stations == 2

    az = grid.azimuth_deg()
    el = grid.elevation_deg()
    rng = grid.range_km()
    assert az.shape == (2, 2)

    epochs = np.asarray([EPOCH_US], dtype=np.int64)
    for sat_index, tle in enumerate(tles):
        for station_index, station in enumerate(stations):
            look = tle.look_angles(station, epochs)
            assert az[sat_index, station_index] == look.azimuth_deg[0]
            assert el[sat_index, station_index] == look.elevation_deg[0]
            assert rng[sat_index, station_index] == look.range_km[0]


def test_reductions_match_grid():
    grid = sidereon.coverage_look_angles(_tles(), _stations(), EPOCH_US)
    el = grid.elevation_deg()

    mask = grid.visible_mask(0.0)
    assert len(mask) == 2 and all(len(row) == 2 for row in mask)
    for sat_index in range(2):
        for station_index in range(2):
            expected = el[sat_index, station_index] >= 0.0
            assert mask[sat_index][station_index] == bool(expected)

    counts = grid.access_counts(0.0)
    assert list(counts) == [
        sum(1 for sat in range(2) if mask[sat][station]) for station in range(2)
    ]

    max_el = grid.max_elevation()
    for station_index in range(2):
        column = el[:, station_index]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            assert max_el[station_index] is None
        else:
            assert max_el[station_index] == float(np.max(finite))


def test_empty_inputs_give_empty_grid():
    grid = sidereon.coverage_look_angles([], _stations(), EPOCH_US)
    assert grid.n_satellites == 0
    assert grid.azimuth_deg().shape == (0, 2)
    assert grid.access_counts(0.0) == []
