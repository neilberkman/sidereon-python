"""Vectorized batch-observe binding tests.

Each batch entry must be exactly what the scalar ``observe`` returns for the same
request, so the batch is checked element-for-element against it."""

import os

import numpy as np
import pytest
import sidereon

try:
    from _helpers import CORE_FIXTURES

    SP3_PATH = os.path.join(
        CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    )
    _HAVE_SP3 = os.path.exists(SP3_PATH)
except Exception:
    _HAVE_SP3 = False

pytestmark = pytest.mark.skipif(
    not _HAVE_SP3, reason="core SP3 fixture unavailable (set SIDEREON_CORE_FIXTURES)"
)

CARRIER_HZ = 1_575_420_000.0
RECEIVER = np.array([4027894.0, 307045.0, 4919474.0])


def _sp3():
    with open(SP3_PATH, "rb") as handle:
        return sidereon.load_sp3(handle.read())


def test_batch_matches_scalar_observe():
    sp3 = _sp3()
    sats = list(sp3.satellites)[:6]
    epoch = float(sp3.epochs_j2000_seconds[2] + 450.0)
    receivers = np.tile(RECEIVER, (len(sats), 1))
    epochs = np.full(len(sats), epoch)
    batch = sidereon.observe_batch(sp3, sats, receivers, epochs, CARRIER_HZ)
    assert len(batch) == len(sats)
    for sat, got in zip(sats, batch):
        ref = sidereon.observe(sp3, sat, RECEIVER, epoch, CARRIER_HZ)
        assert got is not None
        assert got.geometric_range_m == ref.geometric_range_m
        assert got.doppler_hz == ref.doppler_hz
        assert got.elevation_deg == ref.elevation_deg


def test_parallel_and_serial_agree_bitexact():
    sp3 = _sp3()
    sats = list(sp3.satellites)[:8]
    epoch = float(sp3.epochs_j2000_seconds[3] + 100.0)
    receivers = np.tile(RECEIVER, (len(sats), 1))
    epochs = np.full(len(sats), epoch)
    par = sidereon.observe_batch(
        sp3, sats, receivers, epochs, CARRIER_HZ, parallel=True
    )
    ser = sidereon.observe_batch(
        sp3, sats, receivers, epochs, CARRIER_HZ, parallel=False
    )
    for a, b in zip(par, ser):
        if a is None:
            assert b is None
        else:
            assert a.geometric_range_m == b.geometric_range_m
            assert a.sat_pos_ecef_m.tolist() == b.sat_pos_ecef_m.tolist()


def test_length_mismatch_raises():
    sp3 = _sp3()
    sats = list(sp3.satellites)[:3]
    with pytest.raises(ValueError, match="same length"):
        sidereon.observe_batch(
            sp3, sats, np.tile(RECEIVER, (2, 1)), np.zeros(3), CARRIER_HZ
        )


def test_invalid_input_raises_not_masked_as_none():
    """A malformed per-request input (a non-finite epoch) is a core
    ``InvalidInput``; it must raise ``ValueError`` rather than be silently
    folded to a ``None`` entry the way a no-ephemeris gap is."""
    sp3 = _sp3()
    sats = list(sp3.satellites)[:1]
    receivers = np.tile(RECEIVER, (1, 1))
    epochs = np.array([np.nan])
    for parallel in (False, True):
        with pytest.raises(ValueError):
            sidereon.observe_batch(
                sp3, sats, receivers, epochs, CARRIER_HZ, parallel=parallel
            )


def test_structured_ephemeris_failure_raises_not_masked_as_none():
    """An out-of-span epoch and an unknown satellite are structured SP3 ephemeris
    failures (``ObservablesError::Ephemeris``), not the expected no-data gap, so
    they raise ``SolveError`` instead of being masked as a ``None`` entry."""
    sp3 = _sp3()
    sats = list(sp3.satellites)[:1]
    receivers = np.tile(RECEIVER, (1, 1))
    far_epoch = float(sp3.epochs_j2000_seconds[0] - 10.0 * 86400.0)
    for parallel in (False, True):
        with pytest.raises(sidereon.SolveError):
            sidereon.observe_batch(
                sp3,
                sats,
                receivers,
                np.array([far_epoch]),
                CARRIER_HZ,
                parallel=parallel,
            )
        # "R26" parses as a valid GLONASS token but is absent from this product,
        # so the SP3 source returns a structured "unknown satellite" failure.
        with pytest.raises(sidereon.SolveError):
            sidereon.observe_batch(
                sp3,
                ["R26"],
                receivers,
                np.array([float(sp3.epochs_j2000_seconds[2])]),
                CARRIER_HZ,
                parallel=parallel,
            )
