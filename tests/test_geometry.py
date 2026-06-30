"""SP3-backed GNSS geometry: visibility, visibility series, and passes.

These thin wrappers delegate to `sidereon_core::geometry`. The tests load the
same real MGEX precise product the observable tests use and assert the
delegation's invariants: the elevation mask is honored, rows are elevation
sorted, the constellation filter narrows the set, and the single-epoch
visibility count equals the first point of a degenerate one-sample series (so a
wrapper that drops or mis-marshals rows is caught).
"""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES

SP3_PATH = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")
RECEIVER = np.asarray([4_500_000.0, 500_000.0, 4_500_000.0], dtype=np.float64)
MASK_DEG = 10.0


def _load_sp3():
    with open(SP3_PATH, "rb") as fh:
        return sidereon.load_sp3(fh.read())


def test_visible_honors_mask_and_is_elevation_sorted():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    t = float(axis[5])

    rows = sidereon.visible(sp3, RECEIVER, t, MASK_DEG)
    assert len(rows) > 0
    elevations = [row.elevation_deg for row in rows]
    # Every returned satellite clears the mask.
    assert all(e >= MASK_DEG for e in elevations)
    # Rows are sorted by descending elevation (core ordering).
    assert elevations == sorted(elevations, reverse=True)
    # Azimuths are within [0, 360).
    assert all(0.0 <= row.azimuth_deg < 360.0 for row in rows)
    # Identifiers are canonical "<letter><prn>" tokens.
    assert all(len(row.satellite) >= 2 for row in rows)


def test_visible_constellation_filter_narrows_the_set():
    sp3 = _load_sp3()
    t = float(sp3.epochs_j2000_seconds[5])

    all_rows = sidereon.visible(sp3, RECEIVER, t, MASK_DEG)
    gps_rows = sidereon.visible(sp3, RECEIVER, t, MASK_DEG, systems=["G"])

    assert all(row.satellite.startswith("G") for row in gps_rows)
    gps_ids = {row.satellite for row in gps_rows}
    all_gps_ids = {row.satellite for row in all_rows if row.satellite.startswith("G")}
    # The filter returns exactly the GPS subset of the unfiltered scan.
    assert gps_ids == all_gps_ids
    assert len(gps_rows) <= len(all_rows)


def test_visibility_series_matches_single_epoch_count():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    t = float(axis[5])
    step = int(round(float(axis[1] - axis[0])))

    # A degenerate one-sample window at t: its only point's count must equal the
    # standalone visibility scan at the same instant.
    series = sidereon.visibility_series(sp3, RECEIVER, t, t, step, MASK_DEG)
    rows = sidereon.visible(sp3, RECEIVER, t, MASK_DEG)
    assert len(series) == 1
    assert series[0].step_index == 0
    assert series[0].n_visible == len(rows)


def test_visibility_series_over_window_is_monotone_indexed():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    start, end = float(axis[5]), float(axis[15])
    step = int(round(float(axis[1] - axis[0])))

    series = sidereon.visibility_series(sp3, RECEIVER, start, end, step, MASK_DEG)
    assert len(series) > 1
    indices = [point.step_index for point in series]
    assert indices == sorted(indices)
    assert all(point.n_visible >= 0 for point in series)


def test_passes_segments_are_well_formed():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    start, end = float(axis[0]), float(axis[-1])
    step = int(round(float(axis[1] - axis[0])))

    passes = sidereon.passes(sp3, RECEIVER, start, end, step, MASK_DEG)
    assert len(passes) > 0
    for p in passes:
        assert len(p.satellite) >= 2
        assert p.rise_step_index <= p.peak_step_index <= p.set_step_index
        assert p.peak_elevation_deg >= MASK_DEG
    # Passes are ordered by rise index (core ordering).
    rises = [p.rise_step_index for p in passes]
    assert rises == sorted(rises)
