"""Product-staleness selection (#1) through the binding.

The selection layer is pure and deterministic in `sidereon-core`; these tests
check that the binding marshals it faithfully:

- an exact (present-product) selection is BIT-IDENTICAL to querying the caller's
  product directly (the wrapper adds no modeling),
- a degraded selection carries the correct `StalenessMetadata` (kind, source
  epoch, staleness), and an IONEX diurnal shift evaluates bit-identically to the
  unshifted product at the unshifted epoch (TEC values unchanged; only the epoch
  axis moves),
- requests the cap/coverage cannot satisfy raise the typed `SelectionError`.

Fixtures are the crate's own goldens, reused verbatim.
"""

import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

SP3 = os.path.join(CORE_FIXTURES, "sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3")
# Two maps at 646228800 s and 646236000 s J2000 (same UTC day, 2 h apart).
IONEX = os.path.join(CORE_FIXTURES, "ionex", "synthetic_2map_7x7.20i")
DAY_S = 86_400


def _bits(value):
    return struct.pack("<d", float(value))


@pytest.fixture(scope="module")
def sp3():
    with open(SP3, "rb") as fh:
        return sidereon.load_sp3(fh.read())


@pytest.fixture(scope="module")
def ionex():
    with open(IONEX, "rb") as fh:
        return sidereon.load_ionex(fh.read())


# --- StalenessPolicy --------------------------------------------------------


def test_policy_constructors():
    assert sidereon.StalenessPolicy(123.0).max_staleness_s == 123.0
    assert sidereon.StalenessPolicy.seconds(50.0).max_staleness_s == 50.0
    assert sidereon.StalenessPolicy.days(2.0).max_staleness_s == 2.0 * DAY_S
    # The documented default cap is three days.
    assert sidereon.StalenessPolicy.default_policy().max_staleness_s == 3.0 * DAY_S


# --- SP3 selection ----------------------------------------------------------


def test_select_sp3_exact_is_bit_identical_to_direct_interpolation(sp3):
    axis = sp3.epochs_j2000_seconds
    epoch = float(axis[10])
    sat = sp3.satellites[0]

    sel = sidereon.select_sp3([sp3], epoch)
    meta = sel.metadata
    assert sel.metadata.kind == sidereon.DegradationKind.EXACT
    assert meta.kind.is_exact
    assert meta.staleness_s == 0.0
    assert meta.staleness_days == 0.0
    assert meta.requested_epoch_j2000_s == epoch
    assert meta.source_epoch_j2000_s == epoch

    # The selected product's interpolation must be bit-for-bit the caller's.
    state = sel.position_at_j2000_seconds(sat, epoch)
    direct = sp3.interpolate(sat, np.array([epoch], dtype=np.float64))
    for got, want in zip(state.position_m, direct.position_m[0]):
        assert _bits(got) == _bits(want)
    if state.clock_s is not None:
        assert _bits(state.clock_s) == _bits(direct.clock_s[0])

    # The product handed back round-trips byte-identically.
    assert sel.sp3.to_sp3_string() == sp3.to_sp3_string()


def test_select_sp3_nearest_prior_reports_staleness(sp3):
    axis = sp3.epochs_j2000_seconds
    last = float(axis[-1])
    requested = last + 3600.0  # 1 h past coverage, within a 3-day cap.

    sel = sidereon.select_sp3([sp3], requested, sidereon.StalenessPolicy.days(3.0))
    meta = sel.metadata
    assert meta.kind == sidereon.DegradationKind.NEAREST_PRIOR
    assert not meta.kind.is_exact
    assert meta.source_epoch_j2000_s == last
    assert meta.requested_epoch_j2000_s == requested
    assert meta.staleness_s == requested - last
    assert meta.staleness_days == (requested - last) / DAY_S


def test_select_sp3_beyond_cap_raises(sp3):
    axis = sp3.epochs_j2000_seconds
    requested = float(axis[-1]) + 10 * DAY_S  # past the default 3-day cap.
    with pytest.raises(sidereon.SelectionError):
        sidereon.select_sp3([sp3], requested)


def test_select_sp3_no_prior_raises(sp3):
    axis = sp3.epochs_j2000_seconds
    requested = float(axis[0]) - 10 * DAY_S  # only later products exist.
    with pytest.raises(sidereon.SelectionError):
        sidereon.select_sp3([sp3], requested, sidereon.StalenessPolicy.days(30.0))


def test_select_sp3_empty_set_raises():
    with pytest.raises(sidereon.SelectionError):
        sidereon.select_sp3([], 0.0)


# --- IONEX selection --------------------------------------------------------

# Arbitrary but valid slant geometry; the test asserts AGREEMENT between the
# selection and the direct product, not a particular golden value.
_LAT, _LON, _AZ, _EL, _FREQ = 12.0, -30.0, 45.0, 35.0, 1.57542e9


def test_select_ionex_exact_is_bit_identical(ionex):
    epoch = int(ionex.map_epochs_j2000_s[0])
    sel = sidereon.select_ionex([ionex], epoch)
    meta = sel.metadata
    assert meta.kind == sidereon.DegradationKind.EXACT
    assert meta.staleness_s == 0.0
    assert meta.requested_epoch_j2000_s == float(epoch)

    got = sel.slant_delay(_LAT, _LON, _AZ, _EL, epoch, _FREQ)
    want = ionex.slant_delay(_LAT, _LON, _AZ, _EL, epoch, _FREQ)
    assert _bits(got) == _bits(want)


def test_select_ionex_diurnal_shift_is_persistent_and_bit_exact(ionex):
    base = int(ionex.map_epochs_j2000_s[0])
    requested = base + DAY_S  # next day: no covering map, shift a prior day on.

    sel = sidereon.select_ionex([ionex], requested, sidereon.StalenessPolicy.days(3.0))
    meta = sel.metadata
    assert meta.kind == sidereon.DegradationKind.DIURNAL_SHIFT
    assert meta.staleness_s == float(DAY_S)
    assert meta.staleness_days == 1.0
    assert meta.source_epoch_j2000_s == float(base)
    assert meta.requested_epoch_j2000_s == float(requested)

    # Diurnal persistence: the shifted grid at the shifted epoch evaluates
    # bit-for-bit the same as the original grid at the original epoch (TEC values
    # are unchanged; only the epoch axis advanced by a whole day).
    shifted = sel.slant_delay(_LAT, _LON, _AZ, _EL, requested, _FREQ)
    original = ionex.slant_delay(_LAT, _LON, _AZ, _EL, base, _FREQ)
    assert _bits(shifted) == _bits(original)


def test_select_ionex_beyond_cap_raises(ionex):
    base = int(ionex.map_epochs_j2000_s[0])
    requested = base + 5 * DAY_S  # past the default 3-day cap.
    with pytest.raises(sidereon.SelectionError):
        sidereon.select_ionex([ionex], requested)


def test_select_ionex_over_range_smoke(ionex):
    lo = int(ionex.map_epochs_j2000_s[0])
    hi = int(ionex.map_epochs_j2000_s[-1])
    sel = sidereon.select_ionex_over_range([ionex], lo, hi)
    assert sel.metadata.kind == sidereon.DegradationKind.EXACT
    assert sel.metadata.staleness_s == 0.0
