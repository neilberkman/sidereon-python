"""Observable prediction and velocity from a broadcast NAV source.

`observe` / `observe_broadcast` and `solve_velocity_broadcast` are thin wrappers
over `sidereon_core::observables::predict` and `sidereon_core::velocity::solve`
with a broadcast (RINEX NAV) ephemeris source rather than SP3. The real 2020
DOY177 ESBC00DNK broadcast product and the matching COD MGEX precise SP3 (the
broadcast-comparison fixtures) let us check two things: the SP3 and broadcast
predictions of the same satellite geometry agree to broadcast-orbit accuracy,
and feeding the broadcast predictor's own geometric range-rates back through the
velocity solve recovers a static receiver.
"""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES

NAV = os.path.join(CORE_FIXTURES, "nav/ESBC00DNK_R_20201770000_01D_MN.rnx")
SP3 = os.path.join(CORE_FIXTURES, "sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3")
# Ground receiver (ECEF, m) near the broadcast station's latitude band, so GPS
# satellites are above the horizon at the chosen epoch.
RECEIVER = np.asarray([3_513_000.0, 778_000.0, 5_248_000.0], dtype=np.float64)
MASK_DEG = 10.0


def _load():
    broadcast = sidereon.load_rinex_nav(NAV)
    with open(SP3, "rb") as fh:
        precise = sidereon.load_sp3(fh.read())
    return broadcast, precise


def _mid_epoch(precise):
    axis = precise.epochs_j2000_seconds
    return float(axis[len(axis) // 2])


def _l1_hz():
    return sidereon.carrier_frequency_hz(
        sidereon.GnssSystem.GPS, sidereon.CarrierBand.L1
    )


def test_broadcast_and_sp3_observables_agree_for_visible_gps():
    broadcast, precise = _load()
    t = _mid_epoch(precise)
    carrier = _l1_hz()

    visible = sidereon.visible(precise, RECEIVER, t, MASK_DEG, systems=["G"])
    assert len(visible) >= 4

    compared = 0
    for row in visible:
        sat = row.satellite
        sp3_obs = sidereon.observe(precise, sat, RECEIVER, t, carrier)
        try:
            bc_obs = sidereon.observe_broadcast(broadcast, sat, RECEIVER, t, carrier)
        except sidereon.SidereonError:
            # Not every visible PRN has a usable broadcast record at this epoch.
            continue
        compared += 1
        # The SP3 predictor's elevation matches the visibility scan it came from.
        assert abs(sp3_obs.elevation_deg - row.elevation_deg) < 1e-6
        assert 0.0 <= sp3_obs.azimuth_deg < 360.0
        # Broadcast vs precise geometric range differ only by the broadcast orbit
        # error (a few meters), not by a wrapper bug (kilometers).
        assert abs(sp3_obs.geometric_range_m - bc_obs.geometric_range_m) < 50.0
        assert abs(sp3_obs.elevation_deg - bc_obs.elevation_deg) < 0.1
        assert bc_obs.los_unit.shape == (3,)
        assert bc_obs.sat_pos_ecef_m.shape == (3,)
        assert bc_obs.sat_velocity_m_s.shape == (3,)
    assert compared >= 4


def test_solve_velocity_broadcast_recovers_static_receiver():
    broadcast, precise = _load()
    t = _mid_epoch(precise)
    carrier = _l1_hz()

    visible = sidereon.visible(precise, RECEIVER, t, MASK_DEG, systems=["G"])
    observations = []
    for row in visible:
        try:
            obs = sidereon.observe_broadcast(
                broadcast, row.satellite, RECEIVER, t, carrier
            )
        except sidereon.SidereonError:
            continue
        # Feed back the broadcast predictor's own geometric range-rate; a static
        # receiver with sat clock drift folded out must solve to ~zero velocity.
        observations.append(
            sidereon.VelocityObservation(row.satellite, obs.range_rate_m_s, carrier)
        )
    assert len(observations) >= 4

    solution = sidereon.solve_velocity_broadcast(broadcast, observations, RECEIVER, t)
    assert solution.velocity_m_s.shape == (3,)
    assert solution.residuals_m_s.shape == (len(observations),)
    assert np.all(np.isfinite(solution.velocity_m_s))
    # Closed loop on consistent data recovers the (zero) receiver velocity.
    assert solution.speed_m_s < 1.0e-2
