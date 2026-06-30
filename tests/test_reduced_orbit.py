"""Compact mean-element (reduced-orbit) fit / eval / drift.

`reduced_orbit_fit` and the `ReducedOrbit` evaluation methods are thin wrappers
over the `sidereon_core::orbit` public API. The test builds ECEF truth samples by
sampling a real GPS satellite arc out of the MGEX precise product, fits the
circular secular model, and asserts the round trip: a GPS-class semi-major axis
and orbital speed, a small fit residual, and a drift evaluation whose errors line
up with the fit.
"""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES

SP3_PATH = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")

# The product is GRG0MGXFIN_2020 DOY176 0000 = 2020-06-24 00:00:00 GPST, on a
# 15-minute (900 s) node grid; node i is i*900 s into that GPST day.
BASE_Y, BASE_MO, BASE_D = 2020, 6, 24
NODE_STEP_S = 900
FIRST, COUNT = 4, 20  # interior nodes (neighbors available for interpolation)


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
    return sidereon.reduced_orbit_fit(
        _samples(), sidereon.TimeScale.GPST, sidereon.ReducedOrbitModel.CIRCULAR_SECULAR
    )


def test_fit_recovers_gps_class_elements():
    orbit = _fit()
    assert orbit.model == sidereon.ReducedOrbitModel.CIRCULAR_SECULAR
    assert orbit.scale == sidereon.TimeScale.GPST
    assert orbit.n_samples == COUNT
    # GPS semi-major axis is ~26,560 km.
    assert 25.0e6 < orbit.a_m < 28.0e6
    # A reduced-orbit fit to a clean 5 h arc has a small residual.
    assert np.isfinite(orbit.rms_m) and orbit.rms_m < 1.0e5
    assert np.isfinite(orbit.max_m) and orbit.max_m >= orbit.rms_m


def test_position_and_velocity_eval():
    orbit = _fit()
    samples = _samples()
    epoch, x, y, z = samples[0]

    pos = orbit.position(epoch, sidereon.ReducedOrbitFrame.ECEF)
    assert pos.shape == (3,)
    # The model reproduces the truth sample to within a few fit residuals.
    err = np.linalg.norm(pos - np.asarray([x, y, z]))
    assert err < max(10.0 * orbit.rms_m, 5.0e3)

    pos2, vel = orbit.position_velocity(epoch, sidereon.ReducedOrbitFrame.ECEF)
    assert pos2.shape == (3,) and vel.shape == (3,)
    assert np.array_equal(pos2, pos)
    # ECEF speed of a GPS satellite is a few km/s.
    speed = np.linalg.norm(vel)
    assert 2.0e3 < speed < 5.0e3


def test_drift_against_truth_matches_fit():
    orbit = _fit()
    samples = _samples()

    report = orbit.drift(samples, threshold_m=1.0e9)
    assert report.errors_m.shape == (COUNT,)
    assert np.all(np.isfinite(report.errors_m))
    assert report.max_m == float(np.max(report.errors_m))
    assert abs(report.rms_m - float(np.sqrt(np.mean(report.errors_m**2)))) < 1.0e-6
    # No sample crosses the (huge) threshold.
    assert report.threshold_horizon is None

    # A zero threshold is crossed at the first sample.
    crossed = orbit.drift(samples, threshold_m=0.0)
    assert crossed.threshold_horizon is not None


def test_sp3_source_fit_and_drift_match_manual_sampling():
    sp3 = _load_sp3()
    sat = next(s for s in sp3.satellites if s.startswith("G"))
    t0 = _epoch_for_node(FIRST)
    t1 = _epoch_for_node(FIRST + COUNT - 1)
    model = sidereon.ReducedOrbitModel.CIRCULAR_SECULAR

    source_fit = sidereon.reduced_orbit_fit_sp3_source(
        sp3, sat, t0, t1, float(NODE_STEP_S), model
    )
    manual_fit = sidereon.reduced_orbit_fit(_samples(), sidereon.TimeScale.GPST, model)

    assert source_fit.requested_samples == COUNT
    assert source_fit.orbit.n_samples == manual_fit.n_samples
    assert abs(source_fit.orbit.a_m - manual_fit.a_m) < 1.0e-9
    assert abs(source_fit.orbit.rms_m - manual_fit.rms_m) < 1.0e-9
    assert abs(source_fit.orbit.max_m - manual_fit.max_m) < 1.0e-9

    source_drift = sidereon.reduced_orbit_drift_sp3_source(
        source_fit.orbit, sp3, sat, t0, t1, float(NODE_STEP_S), 1.0e9
    )
    manual_drift = source_fit.orbit.drift(_samples(), threshold_m=1.0e9)

    assert source_drift.requested_samples == COUNT
    np.testing.assert_allclose(
        source_drift.report.errors_m, manual_drift.errors_m, rtol=0.0, atol=1.0e-9
    )
    assert abs(source_drift.report.max_m - manual_drift.max_m) < 1.0e-9
    assert abs(source_drift.report.rms_m - manual_drift.rms_m) < 1.0e-9
