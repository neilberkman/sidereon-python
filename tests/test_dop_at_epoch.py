"""SP3-backed exact DOP: single epoch and uniform window series.

`gnss_dop_at_epoch` and `gnss_dop_series_uniform` are thin wrappers over
`sidereon_core::geometry::dop_at_epoch` / `dop_series`. The tests load the same
real MGEX precise product the geometry tests use and assert the delegation's
invariants: the single-epoch result is bit-exact to the array-grid `gnss_dop_series`
at that epoch, and the uniform series' first point is bit-exact to the
single-epoch entry at the window start.
"""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES

SP3_PATH = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")
RECEIVER = np.asarray([4_500_000.0, 500_000.0, 4_500_000.0], dtype=np.float64)
MASK_DEG = 10.0
NODE_STEP_S = 900


def _load_sp3():
    with open(SP3_PATH, "rb") as fh:
        return sidereon.load_sp3(fh.read())


def test_dop_at_epoch_reports_dop_and_satellites():
    sp3 = _load_sp3()
    t = float(sp3.epochs_j2000_seconds[5])

    result = sidereon.gnss_dop_at_epoch(sp3, RECEIVER, t, elevation_mask_deg=MASK_DEG)
    assert result.dop.pdop > 0.0
    assert result.dop.gdop > 0.0
    assert len(result.satellites) >= 4
    assert all(len(sat) >= 2 for sat in result.satellites)
    assert "DopAtEpoch" in repr(result)


def test_dop_at_epoch_matches_array_series_bit_exact():
    sp3 = _load_sp3()
    t = float(sp3.epochs_j2000_seconds[5])

    at = sidereon.gnss_dop_at_epoch(sp3, RECEIVER, t, elevation_mask_deg=MASK_DEG)
    series = sidereon.gnss_dop_series(
        sp3, RECEIVER, np.asarray([t], dtype=np.float64), elevation_mask_deg=MASK_DEG
    )
    assert len(series) == 1
    assert at.dop.gdop == series.gdop[0]
    assert at.dop.pdop == series.pdop[0]
    assert at.dop.hdop == series.hdop[0]
    assert at.dop.vdop == series.vdop[0]
    assert at.dop.tdop == series.tdop[0]


def test_uniform_series_first_point_matches_dop_at_epoch():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    start = float(axis[4])
    end = float(axis[8])

    points = sidereon.gnss_dop_series_uniform(
        sp3, RECEIVER, start, end, NODE_STEP_S, elevation_mask_deg=MASK_DEG
    )
    assert len(points) >= 1
    first = points[0]
    assert first.step_index == 0

    at_start = sidereon.gnss_dop_at_epoch(
        sp3, RECEIVER, start, elevation_mask_deg=MASK_DEG
    )
    assert first.geometry.dop.gdop == at_start.dop.gdop
    assert first.geometry.dop.pdop == at_start.dop.pdop
    assert first.geometry.satellites == at_start.satellites


def test_uniform_series_step_indices_are_monotone():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    points = sidereon.gnss_dop_series_uniform(
        sp3,
        RECEIVER,
        float(axis[4]),
        float(axis[10]),
        NODE_STEP_S,
        elevation_mask_deg=MASK_DEG,
    )
    indices = [p.step_index for p in points]
    assert indices == sorted(indices)
    assert all(p.geometry.dop.pdop > 0.0 for p in points)
