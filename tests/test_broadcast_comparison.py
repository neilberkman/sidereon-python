"""Broadcast-vs-precise accuracy (SISRE) through the binding.

`broadcast_comparison` is a pure wrapper over
`sidereon_core::broadcast_comparison`. This holds the same physical-truth gate as
the core integration test (`crates/sidereon-core/tests/broadcast_comparison.rs`):
the real 2020 DOY177 ESBC00DNK broadcast navigation product differenced against
the COD MGEX final precise SP3, GPS, the full UTC day at a 15 min step, must
reproduce GPS broadcast accuracy (overall 3D orbit RMS ~1-2 m, dominated by
along-track and radial). The comparison-epoch axis is taken from the core's
committed golden so the inputs match the core test exactly.
"""

import json
import math
import os

import sidereon
from _helpers import CORE_FIXTURES, hex_to_f64

NAV = "nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
SP3 = "sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
GOLDEN = os.path.join(CORE_FIXTURES, "broadcast_comparison_golden.json")

J2000_JD = 2451545.0
SECONDS_PER_DAY = 86400.0


def _load_products():
    broadcast = sidereon.load_rinex_nav(os.path.join(CORE_FIXTURES, NAV))
    with open(os.path.join(CORE_FIXTURES, SP3), "rb") as fh:
        precise = sidereon.load_sp3(fh.read())
    return broadcast, precise


def _inputs():
    with open(GOLDEN) as fh:
        inputs = json.load(fh)["inputs"]
    satellites = inputs["satellites"]
    step_s = 2.0 * inputs["velocity_half_s"]
    epochs_j2000_s = [hex_to_f64(row[0]) for row in inputs["epochs"]]
    return satellites, step_s, epochs_j2000_s


def _run():
    satellites, step_s, epochs_j2000_s = _inputs()
    broadcast, precise = _load_products()
    return sidereon.broadcast_comparison(
        broadcast, precise, satellites, epochs_j2000_s, step_s
    )


def _split_jd(t_j2000_s):
    """Day-anchored split Julian date, matching the per-epoch marshalling."""
    jd = J2000_JD + t_j2000_s / SECONDS_PER_DAY
    jd_whole = math.floor(jd - 0.5) + 0.5
    return jd_whole, jd - jd_whole


def test_window_driver_matches_explicit_epoch_axis():
    satellites, step_s, epochs_j2000_s = _inputs()
    broadcast, precise = _load_products()

    t0 = epochs_j2000_s[0]
    t1 = epochs_j2000_s[-1]
    jd_whole, fraction = _split_jd(t0)

    windowed = sidereon.broadcast_comparison_window(
        broadcast,
        precise,
        satellites,
        t0,
        t1,
        jd_whole,
        fraction,
        step_s,
    )
    explicit = sidereon.broadcast_comparison(
        broadcast, precise, satellites, epochs_j2000_s, step_s
    )

    # The window builds the same regular sample grid, so the compared-epoch count
    # matches the explicit-axis path exactly.
    assert windowed.overall.count == explicit.overall.count
    assert windowed.overall.count > 1000

    # The window advances a single precise anchor in lockstep, while the explicit
    # path re-floors a day-anchored split per instant; both name the same instants,
    # so the orbit/clock statistics agree to split-representation noise (sub-cm).
    assert abs(windowed.overall.orbit_3d_rms_m - explicit.overall.orbit_3d_rms_m) < 1e-2
    assert abs(windowed.overall.clock_rms_m - explicit.overall.clock_rms_m) < 1e-2

    # And the window result reproduces the GPS broadcast-accuracy band the core
    # physical-truth gate asserts on.
    assert 0.3 < windowed.overall.orbit_3d_rms_m < 3.0


def test_gps_orbit_agreement_is_broadcast_accuracy_class():
    overall = _run().overall
    assert overall.count > 1000, f"too few compared epochs: {overall.count}"
    rms = overall.orbit_3d_rms_m
    assert 0.3 < rms < 3.0, f"GPS orbit RMS out of band: {rms} m"
    assert overall.orbit_3d_max_m < 6.0, (
        f"GPS orbit max out of band: {overall.orbit_3d_max_m} m"
    )


def test_rac_decomposition_is_orthonormal():
    overall = _run().overall
    rms = overall.orbit_3d_rms_m
    radial, along, cross = (
        overall.radial_rms_m,
        overall.along_rms_m,
        overall.cross_rms_m,
    )
    assert radial > 0.0 and along > 0.0 and cross > 0.0
    quad = (radial * radial + along * along + cross * cross) ** 0.5
    assert abs(rms - quad) < 1e-6, "RAC quadrature mismatch"


def test_removing_clock_datum_shrinks_clock_error():
    overall = _run().overall
    raw = overall.clock_rms_m
    datum_removed = overall.clock_datum_removed_rms_m
    assert 0.0 < raw < 50.0, f"raw clock RMS out of band: {raw} m"
    assert datum_removed > 0.0
    assert datum_removed < raw, "datum removal did not shrink the clock error"


def test_per_satellite_and_missing_are_populated():
    report = _run()
    per_sat = report.per_satellite
    assert len(per_sat) > 20
    # `missing` lists only satellites that actually skipped epochs, so every
    # listed count must be positive (the contract, not a tautology), and any
    # missing satellite must also appear in the per-satellite stats.
    sat_tokens = {sat for sat, _stats in per_sat}
    for sat, count in report.missing:
        assert count > 0, f"{sat} listed in missing with non-positive count {count}"
        assert sat in sat_tokens, f"{sat} missing but absent from per_satellite"
    assert any(stats.count > 0 for _sat, stats in per_sat)
