"""End-to-end code-differential GNSS (DGPS) through the binding.

DGNSS is a pure wrapper over `sidereon_core::dgnss`. The bar is a real
end-to-end differential solve synthesized from the committed multi-GNSS SP3
product: a surveyed base and a nearby rover both observe the same satellites;
pseudoranges are synthesized as `geometric_range - c*dt_sat` (the same synthesis
the GLONASS SPP test uses). The base turns its pseudoranges into corrections, the
rover applies them and solves, and the rover solution must equal, bit for bit,
the core's solve of the same inputs in `fixtures/core_goldens.json`.
"""

import os

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES, core_goldens, hex_to_f64

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


def _bits(values):
    return np.asarray(values, dtype=np.float64).ravel().view(np.uint64).tolist()


def _golden_bits(hex_values):
    return [int(value, 16) for value in hex_values]


def _golden_observations(rows):
    return [(sat, hex_to_f64(bits)) for sat, bits in rows]


def test_dgnss_solve_matches_core():
    """`scripts/core_goldens` synthesizes this scenario the way `_synth` does and
    solves it through the core. The binding solves the inputs it wrote, so the
    comparison does not depend on how numpy rounds the synthesis, and must
    return the core's solution bit for bit."""
    golden = core_goldens()["dgnss"]
    sp3 = _sp3()
    t_rx = hex_to_f64(golden["t_rx_j2000_s"])
    base = [hex_to_f64(bits) for bits in golden["base_position_m"]]
    base_obs = _golden_observations(golden["base_observations"])
    rover_obs = _golden_observations(golden["rover_observations"])

    # The scenario is the one this module synthesizes: the same epoch and the
    # same satellites above the mask at the base and the rover.
    assert t_rx == float(sp3.epochs_j2000_seconds[EPOCH_INDEX])
    python_base = _geodetic_to_ecef(BASE_LAT_DEG, BASE_LON_DEG, BASE_HEIGHT_M)
    assert [sat for sat, _ in _synth(sp3, python_base)[0]] == [
        sat for sat, _ in base_obs
    ]
    assert [sat for sat, _ in _synth(sp3, python_base + ROVER_OFFSET_M)[0]] == [
        sat for sat, _ in rover_obs
    ]

    sol = sidereon.dgnss_solve(sp3, base, base_obs, rover_obs, _config(t_rx))

    expected = golden["solution"]
    assert sol.used_sats == expected["used_sats"]
    assert sol.dropped_sats == expected["dropped_sats"]
    assert _bits(sol.position) == _golden_bits(expected["position_m"])
    assert _bits([sol.rx_clock_s]) == _golden_bits([expected["rx_clock_s"]])
    assert _bits(sol.residuals_m) == _golden_bits(expected["residuals_m"])
    assert _bits(sol.baseline_vector_m) == _golden_bits(expected["baseline_vector_m"])
    assert _bits([sol.baseline_m]) == _golden_bits([expected["baseline_m"]])


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


def test_dgnss_invalid_base_position_preserves_fields():
    import pytest

    sp3 = _sp3()
    with pytest.raises(ValueError) as exc_info:
        sidereon.dgnss_pseudorange_corrections(
            sp3,
            [float("nan"), 0.0, 0.0],
            [],
            float(sp3.epochs_j2000_seconds[EPOCH_INDEX]),
        )

    error = exc_info.value
    assert type(error) is ValueError
    assert str(error) == "invalid DGNSS input base_position_m[0]: not finite"
    assert error.detail == {
        "family": "DgnssError",
        "kind": "invalid_input",
        "field": "base_position_m[0]",
        "reason": "not finite",
        "message": "invalid DGNSS input base_position_m[0]: not finite",
    }


def test_dgnss_public_solve_retains_nested_spp_refusal():
    import pytest

    sp3 = _sp3()
    t_rx = float(sp3.epochs_j2000_seconds[EPOCH_INDEX])
    base = _geodetic_to_ecef(BASE_LAT_DEG, BASE_LON_DEG, BASE_HEIGHT_M)

    base_obs, _ = _synth(sp3, base)
    rover_obs = base_obs[:1]

    with pytest.raises(sidereon.SolveError) as exc_info:
        sidereon.dgnss_solve(sp3, base.tolist(), base_obs, rover_obs, _config(t_rx))

    error = exc_info.value
    assert str(error) == (
        "only 1 usable satellites; need at least 4 (3 position + 1 clock per GNSS)"
    )
    assert error.detail == {
        "family": "DgnssError",
        "kind": "spp",
        "message": str(error),
        "cause": {
            "family": "SppError",
            "kind": "too_few_satellites",
            "used": 1,
            "required": 4,
            "message": str(error),
        },
    }


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
