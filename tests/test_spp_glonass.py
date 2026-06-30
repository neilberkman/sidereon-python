"""End-to-end GLONASS single-point positioning through the SPP binding.

GLONASS is a first-class SPP constellation in `sidereon-core`: each GLONASS
satellite is FDMA, so its L1 Klobuchar ionosphere delay scales by `(f_L1/f_k)^2`
where `f_k` is resolved from the satellite's FDMA frequency channel `k`. That
channel map (slot -> k) rides on `SolveInputs.glonass_channels`, surfaced here as
the optional `SppConfig(glonass_channels=...)` dict.

These tests hold the same uniform end-to-end bar the other bindings use
(sidereon-wasm test/glonass_spp.test.mjs): pseudoranges are *synthesized from the
committed multi-GNSS SP3 product itself* -- geometric range to each satellite
(sampled via the engine's own SP3 interpolation) plus its broadcast clock term --
so an actual GLONASS solve is checked against the known synthesis truth, not a
fabricated number. Neglected light-time / Sagnac terms leave a few-hundred-metre
residual, well inside the loose kilometre-scale bound asserted here.

Behaviours proven:
  * GLONASS pseudoranges solve end-to-end and recover the synthesis truth (iono
    off -- no channels needed);
  * ionosphere on with no channel map -> the core's `IonosphereUnsupported`
    error, naming the GLONASS satellite;
  * supplying the channel map lifts that gate and GLONASS still recovers truth;
  * an out-of-range FDMA channel is rejected exactly like a missing one;
  * `glonass_channels` is a bit-for-bit no-op on a GPS-only solve.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, hex_to_f64

# Multi-GNSS final product with real GLONASS ephemeris (same fixture the WASM
# binding synthesizes from).
GLONASS_SP3 = "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
# Day-of-year for the SP3 epoch (2020-06-24).
GLONASS_DOY = 176.0
# SP3 epoch index we synthesize at (mid-arc, away from the interpolation edges).
EPOCH_INDEX = 48
# Moscow: a GLONASS-favourable latitude, lots of birds above the mask.
RX_LAT_DEG, RX_LON_DEG, RX_HEIGHT_M = 55.75, 37.62, 200.0
# Elevation mask for the synthesized observation set.
ELEVATION_MASK_DEG = 10.0
# Loose bound: the synthesis omits light-time/Sagnac, leaving a sub-km residual.
POSITION_TOLERANCE_M = 2000.0

C_M_S = 299792458.0
# A representable channel outside the valid FDMA range [-7, +6]; the core must
# still reject it, proving the value (not just the key) reaches the solver.
OUT_OF_RANGE_CHANNEL = 9
# Substring of the core's `SppError::IonosphereUnsupported` Display message.
NO_CARRIER = "no modeled carrier frequency"


def _geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """WGS84 geodetic -> ECEF metres."""
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


def _glonass_sp3():
    path = os.path.join(CORE_FIXTURES, "sp3", GLONASS_SP3)
    with open(path, "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _glonass_scenario(sp3):
    """Synthesize a self-consistent GLONASS observation set at one SP3 epoch.

    Returns `(rx, observations, channels)` where `rx` is the ECEF truth, each
    observation's pseudorange is `geometric_range - c * dt_sat`, and `channels`
    assigns a valid FDMA channel (0) to every contributing slot.
    """
    t_rx = float(sp3.epochs_j2000_seconds[EPOCH_INDEX])
    rx = _geodetic_to_ecef(RX_LAT_DEG, RX_LON_DEG, RX_HEIGHT_M)
    up = rx / np.linalg.norm(rx)

    observations = []
    channels = {}
    for sat in sp3.satellites:
        if not sat.startswith("R"):
            continue
        interp = sp3.interpolate(sat, np.array([t_rx]))
        pos = interp.position_m[0]
        dt_sat = float(interp.clock_s[0])
        if not np.isfinite(pos).all() or not np.isfinite(dt_sat):
            continue
        los = pos - rx
        rng = float(np.linalg.norm(los))
        el_deg = np.degrees(np.arcsin(float(np.dot(los, up)) / rng))
        if el_deg < ELEVATION_MASK_DEG:
            continue
        observations.append(sidereon.SppObservation(sat, rng - C_M_S * dt_sat))
        channels[int(sat[1:])] = 0
    return rx, observations, channels, t_rx


def _config(observations, t_rx, *, ionosphere, glonass_channels):
    return sidereon.SppConfig(
        observations=observations,
        t_rx_j2000_s=t_rx,
        t_rx_second_of_day_s=0.0,
        day_of_year=GLONASS_DOY,
        # Generic Earth-surface seed (equator/prime meridian), thousands of km
        # from truth; the ionosphere model needs a non-degenerate receiver radius
        # at the first iteration.
        initial_guess=[6378137.0, 0.0, 0.0, 0.0],
        corrections=sidereon.SppCorrections(ionosphere=ionosphere, troposphere=False),
        klobuchar=sidereon.SppKlobucharCoeffs(
            alpha=[1e-8, 0.0, 0.0, 0.0], beta=[1e5, 0.0, 0.0, 0.0]
        ),
        glonass_channels=glonass_channels,
        with_geodetic=True,
    )


def test_glonass_solves_end_to_end_iono_off():
    """Synthesized GLONASS pseudoranges recover the truth (no channels needed)."""
    sp3 = _glonass_sp3()
    rx, observations, _channels, t_rx = _glonass_scenario(sp3)
    assert len(observations) >= 4, "enough visible GLONASS satellites"

    cfg = _config(observations, t_rx, ionosphere=False, glonass_channels=None)
    sol = sidereon.solve_spp(sp3, cfg)

    assert len(sol.used_sats) >= 4
    assert all(s.startswith("R") for s in sol.used_sats)
    assert len(sol.used_sats) == len(sol.residuals_m)
    err = float(np.linalg.norm(sol.position - rx))
    assert np.isfinite(err) and err < POSITION_TOLERANCE_M, (
        f"recovered within {err:.1f} m"
    )


def test_glonass_iono_without_channel_is_rejected():
    """Ionosphere on with no channel map -> the core's typed error, named sat."""
    sp3 = _glonass_sp3()
    _rx, observations, _channels, t_rx = _glonass_scenario(sp3)

    cfg = _config(observations, t_rx, ionosphere=True, glonass_channels=None)
    with pytest.raises(sidereon.SolveError) as exc:
        sidereon.solve_spp(sp3, cfg)
    msg = str(exc.value)
    assert NO_CARRIER in msg
    # The rejected satellite is named (e.g. "R07").
    assert any(f"R{n:02d}" in msg for n in range(1, 25))


def test_glonass_channel_map_lifts_the_gate_and_recovers_truth():
    """A valid FDMA channel map lets the ionosphere-corrected solve recover truth."""
    sp3 = _glonass_sp3()
    rx, observations, channels, t_rx = _glonass_scenario(sp3)

    cfg = _config(observations, t_rx, ionosphere=True, glonass_channels=channels)
    sol = sidereon.solve_spp(sp3, cfg)

    assert len(sol.used_sats) >= 4
    assert all(s.startswith("R") for s in sol.used_sats)
    err = float(np.linalg.norm(sol.position - rx))
    assert np.isfinite(err) and err < POSITION_TOLERANCE_M, (
        f"recovered within {err:.1f} m"
    )


def test_glonass_out_of_range_channel_is_rejected():
    """An out-of-range FDMA channel is rejected exactly like a missing one --
    the value, not just the key, is threaded into the core and range-checked."""
    sp3 = _glonass_sp3()
    _rx, observations, channels, t_rx = _glonass_scenario(sp3)

    bad = {slot: OUT_OF_RANGE_CHANNEL for slot in channels}
    cfg = _config(observations, t_rx, ionosphere=True, glonass_channels=bad)
    with pytest.raises(sidereon.SolveError) as exc:
        sidereon.solve_spp(sp3, cfg)
    assert NO_CARRIER in str(exc.value)


def test_glonass_channels_is_noop_for_gps_only_solve():
    """A populated channel map must not perturb a solve that observes no GLONASS
    satellite: bit-for-bit identical position and clock, GPS golden reproduced."""
    path = os.path.join(CORE_FIXTURES, "spp_trace_L0_minimal.json")
    with open(path) as fh:
        fx = json.load(fh)["fixture"]
    inp = fx["inputs"]
    sp3_path = os.path.join(CORE_FIXTURES, "sp3", inp["sp3_file"])
    with open(sp3_path, "rb") as fh:
        sp3 = sidereon.load_sp3(fh.read())

    observations = [
        sidereon.SppObservation(o["sat_id"], hex_to_f64(o["p_meas_m"]))
        for o in inp["observations"]
    ]

    def _gps_config(glonass_channels):
        return sidereon.SppConfig(
            observations=observations,
            t_rx_j2000_s=hex_to_f64(inp["t_rx_j2000_s"]),
            t_rx_second_of_day_s=hex_to_f64(inp["t_rx_sod_s"]),
            day_of_year=hex_to_f64(inp["doy"]),
            initial_guess=[hex_to_f64(x) for x in fx["frozen"]["initial_guess_x0"]],
            corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
            klobuchar=sidereon.SppKlobucharCoeffs(
                alpha=[hex_to_f64(x) for x in inp["klobuchar_alpha"]],
                beta=[hex_to_f64(x) for x in inp["klobuchar_beta"]],
            ),
            glonass_channels=glonass_channels,
            with_geodetic=True,
        )

    without = sidereon.solve_spp(sp3, _gps_config(None))
    with_channels = sidereon.solve_spp(sp3, _gps_config({1: 0, 2: 3, 7: -7}))

    assert np.array_equal(with_channels.position, without.position)
    assert with_channels.rx_clock_s == without.rx_clock_s

    expected = np.array([hex_to_f64(x) for x in fx["final_solution"]["x"][:3]])
    err = float(np.linalg.norm(without.position - expected))
    assert err < 1.0e-6, f"GPS golden still reproduced within {err} m"
