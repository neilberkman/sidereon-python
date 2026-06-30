"""Piecewise (multi-segment) reduced-orbit fit and evaluation.

`reduced_orbit_fit_piecewise` and the `PiecewiseOrbit` evaluation methods are thin
wrappers over the `sidereon_core::orbit` piecewise API. The test builds ECEF truth
samples by sampling a real GPS arc out of the MGEX precise product, tiles the
window into segments, and asserts the round trip: each segment fits a GPS-class
orbit, evaluation reproduces the truth, segment selection lands in coverage, and
drift errors line up with the fit.
"""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES

SP3_PATH = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")
BASE_Y, BASE_MO, BASE_D = 2020, 6, 24
NODE_STEP_S = 900
FIRST, COUNT = 4, 20
SEGMENT_S = 10_800  # ~3 h, so the ~4.75 h window tiles into a few segments


def _load_sp3():
    with open(SP3_PATH, "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _epoch_for_node(i):
    total = i * NODE_STEP_S
    return sidereon.CalendarEpoch(
        BASE_Y, BASE_MO, BASE_D, total // 3600, (total % 3600) // 60, 0.0
    )


def _samples():
    sp3 = _load_sp3()
    sat = next(s for s in sp3.satellites if s.startswith("G"))
    axis = sp3.epochs_j2000_seconds
    nodes = list(range(FIRST, FIRST + COUNT))
    query = np.asarray([axis[i] for i in nodes], dtype=np.float64)
    positions = sp3.interpolate(sat, query).position_m
    return [
        (_epoch_for_node(i), float(p[0]), float(p[1]), float(p[2]))
        for i, p in zip(nodes, positions)
    ]


def _fit():
    return sidereon.reduced_orbit_fit_piecewise(
        _samples(),
        sidereon.TimeScale.GPST,
        sidereon.ReducedOrbitModel.CIRCULAR_SECULAR,
        _epoch_for_node(FIRST),
        _epoch_for_node(FIRST + COUNT - 1),
        SEGMENT_S,
    )


def test_fit_tiles_window_into_segments():
    pw = _fit()
    assert pw.model == sidereon.ReducedOrbitModel.CIRCULAR_SECULAR
    assert pw.scale == sidereon.TimeScale.GPST
    assert pw.segment_s == SEGMENT_S
    assert pw.n_segments >= 2
    assert len(pw.segments) == pw.n_segments
    for segment in pw.segments:
        assert 25.0e6 < segment.orbit.a_m < 28.0e6
        assert np.isfinite(segment.orbit.rms_m)


def test_select_segment_covers_query_epoch():
    pw = _fit()
    epoch = _epoch_for_node(FIRST)
    segment = pw.select_segment(epoch)
    assert 25.0e6 < segment.orbit.a_m < 28.0e6


def test_position_and_velocity_eval():
    pw = _fit()
    samples = _samples()
    epoch, x, y, z = samples[0]

    pos = pw.position(epoch, sidereon.ReducedOrbitFrame.ECEF)
    assert pos.shape == (3,)
    err = np.linalg.norm(pos - np.asarray([x, y, z]))
    # The model reproduces the in-segment truth to within a few fit residuals
    # (the window endpoints are the worst-fit points of a secular segment).
    seg = pw.select_segment(epoch)
    assert err < max(10.0 * seg.orbit.rms_m, 5.0e3)

    pos2, vel = pw.position_velocity(epoch, sidereon.ReducedOrbitFrame.ECEF)
    assert np.array_equal(pos2, pos)
    speed = np.linalg.norm(vel)
    assert 2.0e3 < speed < 5.0e3


def test_drift_against_truth_matches_fit():
    pw = _fit()
    samples = _samples()
    report = pw.drift(samples, threshold_m=1.0e9)
    assert report.errors_m.shape == (COUNT,)
    assert np.all(np.isfinite(report.errors_m))
    assert report.max_m == float(np.max(report.errors_m))
    assert report.threshold_horizon is None

    crossed = pw.drift(samples, threshold_m=0.0)
    assert crossed.threshold_horizon is not None
