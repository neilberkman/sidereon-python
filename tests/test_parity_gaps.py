"""Parity-gap coverage: lenient OMM catalog, the TimeOffsetErrorCode discriminant,
and a smoke test exercising every newly wrapped core symbol once.

These close the binding-against-core parity audit gaps (the Python interface was
the only one missing several of these). The numbers are the core's; these only
confirm the binding marshals the new shapes through and registers the symbols.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _read_const(name):
    with open(os.path.join(CORE_FIXTURES, "constellation", name)) as fh:
        return fh.read()


def _combined_gnss_feed():
    """A raw combined feed: real GPS + GLONASS CelesTrak OMM records merged into
    one JSON array, mirroring CelesTrak's `gnss` group."""
    gps = json.loads(_read_const("gps_ops_sample.json"))
    glonass = json.loads(_read_const("glonass_ops_sample.json"))
    return json.dumps(gps + glonass), len(gps), len(glonass)


# --- Lenient OMM catalog ---------------------------------------------------


def test_lenient_omm_partitions_a_combined_feed():
    feed, n_gps, n_glonass = _combined_gnss_feed()

    catalog = sidereon.from_celestrak_omm_lenient(feed, sidereon.GnssSystem.GPS)
    assert isinstance(catalog, sidereon.Catalog)
    # Every GPS record resolved; every GLONASS entry was skipped (not aborted).
    assert len(catalog.records) == n_gps
    assert len(catalog.skipped) == n_glonass
    for rec in catalog.records:
        assert rec.system == sidereon.GnssSystem.GPS
        assert rec.sp3_id.startswith("G")
    # Skipped entries carry identity, not just a count.
    for skipped in catalog.skipped:
        assert isinstance(skipped, sidereon.SkippedOmm)
        assert skipped.norad_id > 0
    assert "Catalog(" in repr(catalog)
    assert "SkippedOmm(" in repr(catalog.skipped[0])


def test_lenient_omm_all_resolvable_has_empty_skipped():
    gps = _read_const("gps_ops_sample.json")
    catalog = sidereon.from_celestrak_omm_lenient(gps, sidereon.GnssSystem.GPS)
    assert catalog.skipped == []
    # Matches the strict builder exactly when nothing is skipped.
    strict = sidereon.from_celestrak_json(gps, sidereon.GnssSystem.GPS)
    assert [r.sp3_id for r in catalog.records] == [r.sp3_id for r in strict]


def test_lenient_omm_malformed_json_still_raises():
    with pytest.raises(sidereon.OmmParseError):
        sidereon.from_celestrak_omm_lenient("not json", sidereon.GnssSystem.GPS)


# --- TimeOffsetErrorCode discriminant --------------------------------------


def test_timescale_offset_error_carries_code():
    with pytest.raises(ValueError) as excinfo:
        sidereon.timescale_offset(sidereon.TimeScale.UTC, sidereon.TimeScale.GPST)
    assert excinfo.value.code == sidereon.TimeOffsetErrorCode.EPOCH_REQUIRED
    assert excinfo.value.code.label == "epoch_required"


def test_timescale_offset_at_nonfinite_epoch_code():
    with pytest.raises(ValueError) as excinfo:
        sidereon.timescale_offset_at(
            sidereon.TimeScale.UTC, sidereon.TimeScale.GPST, float("nan")
        )
    assert excinfo.value.code == sidereon.TimeOffsetErrorCode.NON_FINITE_EPOCH


def test_timescale_offset_tdb_is_unsupported():
    with pytest.raises(ValueError) as excinfo:
        sidereon.timescale_offset(sidereon.TimeScale.TT, sidereon.TimeScale.TDB)
    assert excinfo.value.code == sidereon.TimeOffsetErrorCode.UNSUPPORTED


def test_time_offset_error_code_values_mirror_core():
    # The enum is eq_int; its discriminants mirror the core repr(u8) contract
    # (0 reserved for "no error", 1/2/3 for the variants).
    assert sidereon.TimeOffsetErrorCode.EPOCH_REQUIRED == 1
    assert sidereon.TimeOffsetErrorCode.UNSUPPORTED == 2
    assert sidereon.TimeOffsetErrorCode.NON_FINITE_EPOCH == 3


# --- Smoke: every newly wrapped symbol is reachable and runs ---------------


def test_smoke_all_new_symbols():
    # Standalone covariance ops (match the WASM surface).
    cov = np.eye(3, dtype=np.float64)
    assert sidereon.covariance_is_symmetric(cov)
    assert sidereon.covariance_is_positive_semidefinite(cov)
    r = np.array([7000.0, 0.0, 0.0], dtype=np.float64)
    v = np.array([0.0, 7.5, 0.0], dtype=np.float64)
    eci = sidereon.rtn_to_eci_covariance(cov, r, v)
    assert eci.shape == (3, 3)

    # LAMBDA / bounded ILS.
    float_cycles = np.array([1.01, 2.02], dtype=np.float64)
    qcov = np.diag([0.01, 0.01]).astype(np.float64)
    assert sidereon.lambda_ils_search(float_cycles, qcov).fixed == [1, 2]
    assert sidereon.bounded_ils_search(float_cycles, qcov).fixed == [1, 2]

    # Lambert + IOD.
    re = 6378.1363
    v1t, v2t = sidereon.lambert_battin(
        np.array([2.5 * re, 0.0, 0.0]),
        np.array([1.9151111 * re, 1.6069690 * re, 0.0]),
        np.array([0.0, 4.999792554221911, 0.0]),
        92854.234,
        nrev=1,
    )
    assert v1t.shape == (3,) and v2t.shape == (3,)
    gv2, *_ = sidereon.gibbs(
        np.array([0.0, 0.0, 6378.1363]),
        np.array([0.0, -4464.696, -5102.509]),
        np.array([0.0, 5740.323, 3189.068]),
    )
    assert gv2.shape == (3,)
    hv2, *_ = sidereon.hgibbs(
        np.array([3419.85564, 6019.82602, 2784.60022]),
        np.array([2935.91195, 6326.18324, 2660.59584]),
        np.array([2434.95202, 6597.38674, 2521.52311]),
        0.0,
        (60.0 + 16.48) / 86400.0,
        (120.0 + 33.04) / 86400.0,
    )
    assert hv2.shape == (3,)

    # Direction enums + TimeOffsetErrorCode are importable enum members.
    assert sidereon.DirectionOfMotion.SHORT.label == "short"
    assert sidereon.DirectionOfEnergy.HIGH.label == "high"
    assert sidereon.TimeOffsetErrorCode.EPOCH_REQUIRED.label == "epoch_required"


def test_diff_exposes_fdma_channel_changed():
    # A diff over identical GLONASS records: no FDMA-channel change, but the
    # field (and its inclusion in changed()) must be reachable so a real channel
    # reassignment is never silently dropped.
    gps = sidereon.from_celestrak_json(
        _read_const("gps_ops_sample.json"), sidereon.GnssSystem.GPS
    )
    d = sidereon.diff(gps, gps)
    assert d.fdma_channel_changed == []
    assert not sidereon.changed(d)
    assert "fdma_channel_changed=" in repr(d)
