"""Broadcast-ephemeris SPP and precise-with-broadcast fallback (#2).

Reproduces the crate's reference-arc behavior through the binding on the 2020
DOY177 IGS data (ESBC00DNK GPS C1C observations, the ESBC mixed broadcast
navigation, the COD MGEX final precise SP3) at the first observation epoch:

- a broadcast-only fix and a precise fix on the same pseudoranges agree within a
  LABELED few-meter bound (the broadcast signal-in-space accuracy delta, not a
  bit-exact claim),
- with a precise product covering the epoch, the fallback reports a PRECISE-exact
  source with zero staleness,
- with no covering precise product, the fallback drops to BROADCAST, records WHY
  (never a silent substitution), and is BIT-IDENTICAL to the broadcast-only solve,
- a stale-but-within-cap precise product is used as a degraded PRECISE source,
- a broadcast solve that cannot converge raises the typed `FallbackError`.

Fixtures are the crate's own goldens, reused verbatim.
"""

import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

NAV = os.path.join(CORE_FIXTURES, "nav", "ESBC00DNK_R_20201770000_01D_MN.rnx")
OBS = os.path.join(CORE_FIXTURES, "obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx")
COD_SP3 = os.path.join(CORE_FIXTURES, "sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3")
IGS_2026_SP3 = os.path.join(
    CORE_FIXTURES, "sp3", "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
)
PRIOR_DAY_SP3 = os.path.join(CORE_FIXTURES, "sp3", "GAP_G01_20201760000_15M.sp3")

# The crate's labeled broadcast-vs-precise position bound for this arc: the two
# orbit/clock sources legitimately differ at the meter level, so this is a
# documented accuracy delta, not a bit-exact claim.
BROADCAST_VS_PRECISE_BOUND_M = 20.0


def _bits(value):
    return struct.pack("<d", float(value))


def _civil_to_j2000_s(t):
    """Civil epoch -> J2000 seconds in the file time scale (the crate's recipe)."""
    year, month, day = t.year, t.month, t.day
    a = (14 - month) // 12
    y = year + 4800 - a
    m = month + 12 * a - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd_whole = jdn - 0.5
    day_seconds = t.hour * 3600.0 + t.minute * 60.0 + t.second
    # J2000 = JD 2451545.0; seconds from the J2000 instant.
    return (jd_whole - 2451545.0) * 86400.0 + day_seconds


def _first_epoch_config():
    obs = sidereon.load_rinex_obs(OBS)
    epoch = obs.epochs[0]
    t = epoch.epoch

    t_rx = _civil_to_j2000_s(t)
    sod = t.hour * 3600.0 + t.minute * 60.0 + t.second

    flt = sidereon.ObservationFilter([(sidereon.GnssSystem.GPS, ["C1C"])])
    series = obs.observation_values(0, flt)
    observations = []
    for sat, code, value in zip(series.satellites, series.codes, series.values):
        if not sat.startswith("G") or code != "C1C" or not np.isfinite(value):
            continue
        observations.append(sidereon.SppObservation(sat, float(value)))
    assert len(observations) >= 5, f"need a redundant GPS set, got {len(observations)}"

    approx = obs.header.approx_position_m
    initial_guess = [float(approx[0]), float(approx[1]), float(approx[2]), 0.0]

    return sidereon.SppConfig(
        observations=observations,
        t_rx_j2000_s=t_rx,
        t_rx_second_of_day_s=sod,
        day_of_year=177.0 + sod / 86400.0,
        initial_guess=initial_guess,
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=True),
        with_geodetic=True,
    )


@pytest.fixture(scope="module")
def store():
    return sidereon.load_rinex_nav(NAV)


@pytest.fixture(scope="module")
def config():
    return _first_epoch_config()


def test_solve_broadcast_converges(store, config):
    sol = sidereon.solve_broadcast(store, config)
    assert sol.geodetic is not None
    assert len(sol.used_sats) >= 5
    assert np.all(np.isfinite(sol.position))


def test_broadcast_fde_runs_real_broadcast_spp_path(store, config):
    result = sidereon.qc_fde_broadcast(
        store, config, p_fa=0.01, max_iterations=2, max_pdop=20.0
    )
    alias = sidereon.fde_broadcast(
        store, config, p_fa=0.01, max_iterations=2, max_pdop=20.0
    )

    assert result.excluded == []
    assert result.iterations == 0
    assert result.used_sats == alias.used_sats
    assert np.all(np.isfinite(result.position))
    assert result.geodetic is not None


def test_broadcast_vs_precise_within_labeled_bound(store, config):
    broadcast = sidereon.solve_broadcast(store, config)
    with open(COD_SP3, "rb") as fh:
        cod = sidereon.load_sp3(fh.read())

    sourced = sidereon.solve_with_fallback([cod], store, config)
    assert sourced.source == sidereon.FixSource.PRECISE
    assert sourced.is_precise_exact
    assert sourced.staleness is not None
    assert sourced.staleness.staleness_s == 0.0
    assert sourced.broadcast_reason is None

    delta = float(np.linalg.norm(sourced.solution.position - broadcast.position))
    # Non-tautological: a degenerate source would collapse the two fixes.
    assert delta > 0.01, f"broadcast and precise implausibly identical ({delta} m)"
    assert delta < BROADCAST_VS_PRECISE_BOUND_M, f"delta {delta} m over bound"


def test_fallback_to_broadcast_when_no_precise_is_bit_identical(store, config):
    broadcast = sidereon.solve_broadcast(store, config)
    sourced = sidereon.solve_with_fallback([], store, config)

    assert sourced.source == sidereon.FixSource.BROADCAST
    assert sourced.is_broadcast
    assert not sourced.is_precise
    assert sourced.broadcast_reason == sidereon.BroadcastReason.PRECISE_UNAVAILABLE
    assert sourced.staleness is None
    assert sourced.attempted_staleness is None
    # The typed selection reason is surfaced, never silently dropped.
    assert sourced.selection_error is not None
    assert "empty" in sourced.selection_error.lower()

    # The broadcast fix is bit-for-bit the broadcast-only solve.
    assert _bits(sourced.solution.rx_clock_s) == _bits(broadcast.rx_clock_s)
    assert np.array_equal(
        sourced.solution.position.view(np.int64), broadcast.position.view(np.int64)
    )


def test_fallback_drops_to_broadcast_when_precise_does_not_cover_epoch(store, config):
    broadcast = sidereon.solve_broadcast(store, config)
    with open(IGS_2026_SP3, "rb") as fh:
        future = sidereon.load_sp3(fh.read())  # 2026 product, after the 2020 epoch.

    sourced = sidereon.solve_with_fallback([future], store, config)
    assert sourced.source == sidereon.FixSource.BROADCAST
    assert sourced.broadcast_reason == sidereon.BroadcastReason.PRECISE_UNAVAILABLE
    # The only product is later than the epoch: no prior to degrade to.
    assert sourced.selection_error is not None
    assert "before" in sourced.selection_error.lower()
    assert np.array_equal(
        sourced.solution.position.view(np.int64), broadcast.position.view(np.int64)
    )


def test_fallback_uses_degraded_precise_when_stale_product_serves_epoch(store, config):
    with open(PRIOR_DAY_SP3, "rb") as fh:
        prior = sidereon.load_sp3(
            fh.read()
        )  # DOY176, ends just before the DOY177 epoch.

    sourced = sidereon.solve_with_fallback(
        [prior], store, config, sidereon.StalenessPolicy.days(3.0)
    )
    assert sourced.source == sidereon.FixSource.PRECISE
    assert sourced.is_precise
    assert not sourced.is_precise_exact
    meta = sourced.staleness
    assert meta is not None
    assert meta.kind == sidereon.DegradationKind.NEAREST_PRIOR
    assert 0.0 < meta.staleness_s < sidereon.StalenessPolicy.days(3.0).max_staleness_s


def test_fallback_surfaces_typed_error_when_broadcast_cannot_solve(store):
    # No observations and no precise product: the broadcast fallback solve has
    # nothing to fix on and raises the typed FallbackError.
    empty = sidereon.SppConfig(
        observations=[],
        t_rx_j2000_s=646228800.0,
        t_rx_second_of_day_s=0.0,
        day_of_year=177.0,
        initial_guess=[0.0, 0.0, 0.0, 0.0],
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
        with_geodetic=False,
    )
    with pytest.raises(sidereon.FallbackError):
        sidereon.solve_with_fallback([], store, empty)
