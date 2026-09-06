import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

_FIXTURE = "GAP_G01_20201760000_15M.sp3"
_MID_HOLE_J2000_S = 646_260_300.0


def _gapped_sp3_path():
    return os.path.join(CORE_FIXTURES, "sp3", _FIXTURE)


def test_load_sp3_interpolation_policy():
    path = _gapped_sp3_path()
    sp3_default = sidereon.load_sp3(path)
    assert sp3_default.gap_threshold_factor == 1.5

    # Midpoint of 12-spacing G01 hole is refused under default policy.
    midpoint = np.array([_MID_HOLE_J2000_S])
    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        sp3_default.interpolate("G01", midpoint)

    # Factor 13 spans the 12-spacing hole and serves the midpoint.
    sp3_wide = sidereon.load_sp3(path, gap_threshold_factor=13.0)
    assert sp3_wide.gap_threshold_factor == 13.0
    interp = sp3_wide.interpolate("G01", midpoint)
    assert np.all(np.isfinite(interp.position_m))

    # A contiguous satellite produces identical interpolation under either policy.
    query_g02 = np.array([_MID_HOLE_J2000_S + 450.0])
    interp_default = sp3_default.interpolate("G02", query_g02)
    interp_wide = sp3_wide.interpolate("G02", query_g02)
    np.testing.assert_array_equal(interp_wide.position_m, interp_default.position_m)

    # Invalid factor (<= 1.0 or non-finite) raises ValueError.
    for invalid in (1.0, 0.5, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="greater than 1.0"):
            sidereon.load_sp3(path, gap_threshold_factor=invalid)


def test_sp3_with_interpolation_options():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)
    midpoint = np.array([_MID_HOLE_J2000_S])

    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        sp3.interpolate("G01", midpoint)

    sp3_wide = sp3.with_interpolation_options(13.0)
    assert sp3_wide.gap_threshold_factor == 13.0
    assert np.all(np.isfinite(sp3_wide.interpolate("G01", midpoint).position_m))

    with pytest.raises(ValueError, match="greater than 1.0"):
        sp3.with_interpolation_options(1.0)


def test_sp3_stencil_extent_follows_gap_threshold():
    path = _gapped_sp3_path()
    sp3_default = sidereon.load_sp3(path)
    before_s, after_s = sp3_default.stencil_extent()
    assert (before_s, after_s) == (11.0 * 900.0, 11.0 * 900.0)

    sp3_wide = sidereon.load_sp3(path, gap_threshold_factor=13.0)
    before_wide, after_wide = sp3_wide.stencil_extent()
    assert (before_wide, after_wide) == (19800.0, 19800.0)


def test_check_continuity_interpolation_policy():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)

    # Default policy checks residuals without bridging the gap.
    res_def = sp3.check_continuity(orbit_class=None, residual_tolerance_m=1.0)
    d_def = next(
        d
        for d in res_def["defects"]
        if d["satellite"] == "G01" and d["from_j2000_s"] == 646_254_900.0
    )
    assert d_def["magnitude"] > 20.0

    # Wide policy bridges the gap, altering the hold-out replay.
    res_wide = sp3.check_continuity(
        orbit_class=None, residual_tolerance_m=1.0, gap_threshold_factor=13.0
    )
    d_wide = next(
        d
        for d in res_wide["defects"]
        if d["satellite"] == "G01" and d["from_j2000_s"] == 646_254_900.0
    )
    assert d_wide["magnitude"] < 15.0
    assert d_wide["magnitude"] != d_def["magnitude"]

    with pytest.raises(ValueError, match="greater than 1.0"):
        sp3.check_continuity(
            orbit_class=None, residual_tolerance_m=1.0, gap_threshold_factor=1.0
        )


def test_continuity_verdict_interpolation_policy():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)
    axis = sp3.epochs_j2000_seconds
    start, stop = float(axis[10]), float(axis[20])

    verdict_def = sp3.continuity_verdict(
        start, stop, orbit_class=None, residual_tolerance_m=1.0
    )
    assert verdict_def["decision"] in ("accept", "refuse")

    verdict_wide = sp3.continuity_verdict(
        start,
        stop,
        orbit_class=None,
        residual_tolerance_m=1.0,
        gap_threshold_factor=13.0,
    )
    assert verdict_wide["decision"] in ("accept", "refuse")

    with pytest.raises(ValueError, match="greater than 1.0"):
        sp3.continuity_verdict(
            start,
            stop,
            orbit_class=None,
            residual_tolerance_m=1.0,
            gap_threshold_factor=1.0,
        )


def test_precise_ephemeris_samples_interpolation_policy():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)
    samples = sp3.precise_ephemeris_samples()

    pes_def = sidereon.PreciseEphemerisSamples.from_samples(samples)
    assert pes_def.gap_threshold_factor == 1.5
    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        sidereon.PreciseEphemerisInterpolant.from_precise_ephemeris_samples(
            pes_def
        ).position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)

    pes_wide = sidereon.PreciseEphemerisSamples.from_samples(
        samples, gap_threshold_factor=13.0
    )
    assert pes_wide.gap_threshold_factor == 13.0
    state = sidereon.PreciseEphemerisInterpolant.from_precise_ephemeris_samples(
        pes_wide
    ).position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)
    assert np.all(np.isfinite(state.position_m))

    pes_wide2 = pes_def.with_interpolation_options(13.0)
    assert pes_wide2.gap_threshold_factor == 13.0
    state2 = sidereon.PreciseEphemerisInterpolant.from_precise_ephemeris_samples(
        pes_wide2
    ).position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)
    assert np.all(np.isfinite(state2.position_m))

    with pytest.raises(ValueError, match="greater than 1.0"):
        sidereon.PreciseEphemerisSamples.from_samples(samples, gap_threshold_factor=1.0)
    with pytest.raises(ValueError, match="greater than 1.0"):
        pes_def.with_interpolation_options(1.0)


def test_precise_ephemeris_interpolant_from_sp3():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)

    interp_def = sidereon.PreciseEphemerisInterpolant.from_sp3(sp3)
    assert interp_def.gap_threshold_factor == 1.5
    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        interp_def.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)

    interp_wide = sidereon.PreciseEphemerisInterpolant.from_sp3(
        sp3, gap_threshold_factor=13.0
    )
    assert interp_wide.gap_threshold_factor == 13.0
    state = interp_wide.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)
    assert np.all(np.isfinite(state.position_m))

    interp_wide2 = interp_def.with_interpolation_options(13.0)
    assert interp_wide2.gap_threshold_factor == 13.0
    assert np.all(
        np.isfinite(
            interp_wide2.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S).position_m
        )
    )

    with pytest.raises(ValueError, match="greater than 1.0"):
        sidereon.PreciseEphemerisInterpolant.from_sp3(sp3, gap_threshold_factor=1.0)
    with pytest.raises(ValueError, match="greater than 1.0"):
        interp_def.with_interpolation_options(1.0)


def test_precise_ephemeris_interpolant_from_samples():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)
    samples = sp3.precise_ephemeris_samples()

    interp_def = sidereon.PreciseEphemerisInterpolant.from_samples(samples)
    assert interp_def.gap_threshold_factor == 1.5
    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        interp_def.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)

    interp_wide = sidereon.PreciseEphemerisInterpolant.from_samples(
        samples, gap_threshold_factor=13.0
    )
    assert interp_wide.gap_threshold_factor == 13.0
    state = interp_wide.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)
    assert np.all(np.isfinite(state.position_m))

    with pytest.raises(ValueError, match="greater than 1.0"):
        sidereon.PreciseEphemerisInterpolant.from_samples(
            samples, gap_threshold_factor=1.0
        )


def test_precise_ephemeris_interpolant_from_precise_ephemeris_samples():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)
    samples_src = sidereon.PreciseEphemerisSamples.from_samples(
        sp3.precise_ephemeris_samples()
    )

    interp_def = sidereon.PreciseEphemerisInterpolant.from_precise_ephemeris_samples(
        samples_src
    )
    assert interp_def.gap_threshold_factor == 1.5
    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        interp_def.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)

    interp_wide = sidereon.PreciseEphemerisInterpolant.from_precise_ephemeris_samples(
        samples_src, gap_threshold_factor=13.0
    )
    assert interp_wide.gap_threshold_factor == 13.0
    state = interp_wide.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)
    assert np.all(np.isfinite(state.position_m))

    with pytest.raises(ValueError, match="greater than 1.0"):
        sidereon.PreciseEphemerisInterpolant.from_precise_ephemeris_samples(
            samples_src, gap_threshold_factor=1.0
        )


def test_sp3_precise_interpolant_artifact_bytes():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)

    bytes_def = sp3.precise_interpolant_artifact_bytes()
    artifact_def = sidereon.PreciseInterpolantArtifact.from_bytes(bytes_def)
    assert artifact_def.gap_threshold_factor == 1.5
    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        artifact_def.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)

    bytes_wide = sp3.precise_interpolant_artifact_bytes(gap_threshold_factor=13.0)
    artifact_wide = sidereon.PreciseInterpolantArtifact.from_bytes(bytes_wide)
    assert artifact_wide.gap_threshold_factor == 13.0
    state = artifact_wide.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)
    assert np.all(np.isfinite(state.position_m))

    with pytest.raises(ValueError, match="greater than 1.0"):
        sp3.precise_interpolant_artifact_bytes(gap_threshold_factor=1.0)


def test_build_precise_interpolant_artifact_bytes():
    path = _gapped_sp3_path()
    sp3 = sidereon.load_sp3(path)

    bytes_def = sidereon.build_precise_interpolant_artifact_bytes(sp3)
    artifact_def = sidereon.PreciseInterpolantArtifact.from_bytes(bytes_def)
    assert artifact_def.gap_threshold_factor == 1.5
    with pytest.raises(sidereon.SolveError, match="epoch out of range"):
        artifact_def.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)

    bytes_wide = sidereon.build_precise_interpolant_artifact_bytes(
        sp3, gap_threshold_factor=13.0
    )
    artifact_wide = sidereon.PreciseInterpolantArtifact.from_bytes(bytes_wide)
    assert artifact_wide.gap_threshold_factor == 13.0
    state = artifact_wide.position_at_j2000_seconds("G01", _MID_HOLE_J2000_S)
    assert np.all(np.isfinite(state.position_m))

    with pytest.raises(ValueError, match="greater than 1.0"):
        sidereon.build_precise_interpolant_artifact_bytes(sp3, gap_threshold_factor=1.0)
