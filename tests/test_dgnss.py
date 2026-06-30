"""End-to-end code-differential GNSS (DGPS) through the binding.

DGNSS is a pure wrapper over `sidereon_core::dgnss`. The bar is a real
end-to-end differential solve synthesized from the committed multi-GNSS SP3
product: a surveyed base and a nearby rover both observe the same satellites;
pseudoranges are synthesized as `geometric_range - c*dt_sat` (the same synthesis
the GLONASS SPP test uses). The base turns its pseudoranges into corrections, the
rover applies them and solves, and the recovered rover position and base/rover
baseline must match the known synthesis truth. The differential cancels the
satellite-common terms, so the baseline is recovered far tighter than the
absolute light-time-omitted single-point error.
"""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES

SP3_FILE = "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
DOY = 176.0
EPOCH_INDEX = 48
C_M_S = 299792458.0
ELEVATION_MASK_DEG = 10.0

# Surveyed base (Moscow-ish) and a rover offset by a known ECEF baseline.
BASE_LAT_DEG, BASE_LON_DEG, BASE_HEIGHT_M = 55.75, 37.62, 200.0
ROVER_OFFSET_M = np.array([30.0, -40.0, 20.0])


def _geodetic_to_ecef(lat_deg, lon_deg, h_m):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + h_m) * np.cos(lat) * np.cos(lon),
            (n + h_m) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - e2) + h_m) * np.sin(lat),
        ]
    )


def _sp3():
    with open(os.path.join(CORE_FIXTURES, "sp3", SP3_FILE), "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _synth(sp3, position):
    """Synthesize `(token, pseudorange)` for the GPS satellites above the mask at
    the receiver `position` (ECEF), pseudorange = geometric range - c*dt_sat."""
    t_rx = float(sp3.epochs_j2000_seconds[EPOCH_INDEX])
    up = position / np.linalg.norm(position)
    obs = []
    for sat in sp3.satellites:
        if not sat.startswith("G"):
            continue
        interp = sp3.interpolate(sat, np.array([t_rx]))
        pos = interp.position_m[0]
        dt = float(interp.clock_s[0])
        if not np.isfinite(pos).all() or not np.isfinite(dt):
            continue
        los = pos - position
        rng = float(np.linalg.norm(los))
        el = np.degrees(np.arcsin(float(np.dot(los, up)) / rng))
        if el < ELEVATION_MASK_DEG:
            continue
        obs.append((sat, rng - C_M_S * dt))
    return obs, t_rx


def _config(t_rx):
    return sidereon.SppConfig(
        observations=[],
        t_rx_j2000_s=t_rx,
        t_rx_second_of_day_s=0.0,
        day_of_year=DOY,
        initial_guess=[6378137.0, 0.0, 0.0, 0.0],
        with_geodetic=True,
    )


def test_dgnss_solve_recovers_rover_and_baseline():
    sp3 = _sp3()
    base = _geodetic_to_ecef(BASE_LAT_DEG, BASE_LON_DEG, BASE_HEIGHT_M)
    rover = base + ROVER_OFFSET_M

    base_obs, t_rx = _synth(sp3, base)
    rover_obs, _ = _synth(sp3, rover)
    assert len(base_obs) >= 5 and len(rover_obs) >= 5

    sol = sidereon.dgnss_solve(sp3, list(base), base_obs, rover_obs, _config(t_rx))

    assert len(sol.used_sats) >= 4
    pos_err = float(np.linalg.norm(sol.position - rover))
    assert np.isfinite(pos_err) and pos_err < 50.0, (
        f"rover recovered within {pos_err:.3f} m"
    )

    # The differential cancels the satellite-common (light-time-omitted) error,
    # so the baseline is recovered to the metre.
    true_baseline = float(np.linalg.norm(ROVER_OFFSET_M))
    assert abs(sol.baseline_m - true_baseline) < 5.0
    baseline_err = float(np.linalg.norm(sol.baseline_vector_m - ROVER_OFFSET_M))
    assert baseline_err < 5.0, f"baseline vector within {baseline_err:.3f} m"


def test_dgnss_corrections_near_zero_for_clock_free_base():
    """The base pseudoranges are synthesized as the exact modeled value (no base
    receiver clock), so every per-satellite correction is ~0."""
    sp3 = _sp3()
    base = _geodetic_to_ecef(BASE_LAT_DEG, BASE_LON_DEG, BASE_HEIGHT_M)
    base_obs, t_rx = _synth(sp3, base)

    prc = sidereon.dgnss_pseudorange_corrections(sp3, list(base), base_obs, t_rx)
    assert len(prc) >= 5
    # The only residual is the light-time/Sagnac term the simple synthesis omits.
    assert max(abs(v) for v in prc.values()) < 200.0


def test_dgnss_apply_corrections_round_trip_and_drop():
    sp3 = _sp3()
    base = _geodetic_to_ecef(BASE_LAT_DEG, BASE_LON_DEG, BASE_HEIGHT_M)
    base_obs, t_rx = _synth(sp3, base)
    prc = sidereon.dgnss_pseudorange_corrections(sp3, list(base), base_obs, t_rx)

    # A rover observation for a satellite with no correction is dropped; the rest
    # are corrected by exactly their PRC. Pick a token guaranteed absent from the
    # corrections so the drop path is always exercised.
    absent = next(f"G{n:02d}" for n in range(1, 40) if f"G{n:02d}" not in prc)
    rover_obs = list(base_obs) + [(absent, 2.2e7)]
    corrected, dropped = sidereon.dgnss_apply_corrections(rover_obs, prc)

    corrected_map = dict(corrected)
    for token, pr in base_obs:
        if token in prc:
            assert abs(corrected_map[token] - (pr - prc[token])) < 1e-6
    # Every rover token without a matching correction is dropped (in rover order),
    # and the synthesized absent satellite is guaranteed among them.
    assert dropped == [token for token, _pr in rover_obs if token not in prc]
    assert absent in dropped
    assert absent not in corrected_map
