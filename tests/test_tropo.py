"""Standalone tropospheric delay through the binding.

Saastamoinen zenith (dry+wet), Niell (NMF) mapping, and the composed slant delay
are pure wrappers over `sidereon_core::tropo`. Checked against the core's own
`troposphere_golden.json` (RTKLIB recipe), reused verbatim.

Bit-exactness boundary (mirrors the core's own contract):
  * The zenith hydrostatic/wet delays depend only on meteorology, latitude, and
    height -- no epoch -- so they are asserted **0-ULP** against the golden.
  * At an elevation of 90 deg the Niell mapping factors are exactly 1.0 (so the
    zenith slant is also epoch-independent) -- asserted **0-ULP**.
  * Below the zenith the mapping carries the Niell seasonal day-of-year term. The
    binding derives the day-of-year from a UTC epoch, which reconstructs the
    golden's integer day-of-year only to within the float granularity of a split
    Julian date; that path is checked against the golden within a tight labeled
    tolerance, exactly the bound the core's public wrapper documents.
"""

import datetime
import json
import math
import os

import pytest
import sidereon
from _helpers import CORE_FIXTURES

GOLDEN = os.path.join(CORE_FIXTURES, "troposphere_golden.json")

# 2000-01-28 00:00:00 UTC reconstructs the golden's integer day-of-year 28 for
# the day-of-year-28 cases (used only for the mapping/slant reconstruction band).
DOY28_EPOCH_US = int(
    datetime.datetime(2000, 1, 28, tzinfo=datetime.timezone.utc).timestamp() * 1e6
)
# Labeled physical bound for the epoch->day-of-year reconstruction (see module
# docstring). The Niell seasonal term enters so weakly that this is generous.
RECONSTRUCTION_TOL_M = 1e-4


def _cases():
    with open(GOLDEN) as fh:
        return json.load(fh)["slant_cases"]


def _inp(case, key):
    return float.fromhex(case["inputs"][key])


def _exp(case, key):
    return float.fromhex(case["expect"][key])


def _valid_zenith_cases():
    """Cases with positive elevation and a non-gated (non-zero) slant: their
    golden zenith delays are the true standalone Saastamoinen zenith values."""
    out = []
    for c in _cases():
        if _inp(c, "el_deg") > 0.0 and _exp(c, "slant_m") != 0.0:
            out.append(c)
    return out


def test_zenith_delay_is_bit_exact():
    cases = _valid_zenith_cases()
    assert len(cases) >= 10
    for c in cases:
        dry, wet = sidereon.tropo_zenith_delay(
            _inp(c, "lat_rad"),
            _inp(c, "height_m"),
            _inp(c, "pressure_hpa"),
            _inp(c, "temperature_k"),
            _inp(c, "relative_humidity"),
        )
        assert dry == _exp(c, "zhd_m"), (
            f"{c['name']} dry: {dry.hex()} != {_exp(c, 'zhd_m').hex()}"
        )
        assert wet == _exp(c, "zwd_m"), (
            f"{c['name']} wet: {wet.hex()} != {_exp(c, 'zwd_m').hex()}"
        )


def _zenith_case():
    return next(c for c in _cases() if c["name"] == "zenith_midlat")


def test_zenith_mapping_is_unity_bit_exact():
    """At 90 deg elevation the Niell mapping factors are exactly 1.0, regardless
    of day-of-year, so this is epoch-independent and asserted to the bit."""
    c = _zenith_case()
    dry, wet = sidereon.tropo_mapping_factors(
        _inp(c, "el_rad"), _inp(c, "lat_rad"), _inp(c, "height_m"), DOY28_EPOCH_US
    )
    assert dry == 1.0 and wet == 1.0


def test_zenith_slant_is_bit_exact():
    """The slant at 90 deg is zhd + zwd (mapping = 1), epoch-independent: 0-ULP."""
    c = _zenith_case()
    slant = sidereon.tropo_slant_delay(
        _inp(c, "el_rad"),
        _inp(c, "lat_rad"),
        _inp(c, "lon_rad"),
        _inp(c, "height_m"),
        _inp(c, "pressure_hpa"),
        _inp(c, "temperature_k"),
        _inp(c, "relative_humidity"),
        DOY28_EPOCH_US,
    )
    assert slant == _exp(c, "slant_m"), f"{slant.hex()} != {_exp(c, 'slant_m').hex()}"


def test_below_zenith_mapping_and_slant_match_golden_within_reconstruction_band():
    """An el=30 deg day-of-year-28 case: the Niell mapping/slant track the golden
    within the epoch->day-of-year reconstruction bound."""
    c = next(c for c in _cases() if c["name"] == "el30_midlat")
    assert c["inputs"]["doy_repr"] == "28.0"

    mh, mw = sidereon.tropo_mapping_factors(
        _inp(c, "el_rad"), _inp(c, "lat_rad"), _inp(c, "height_m"), DOY28_EPOCH_US
    )
    assert abs(mh - _exp(c, "mh")) < 1e-9
    assert abs(mw - _exp(c, "mw")) < 1e-9

    slant = sidereon.tropo_slant_delay(
        _inp(c, "el_rad"),
        _inp(c, "lat_rad"),
        _inp(c, "lon_rad"),
        _inp(c, "height_m"),
        _inp(c, "pressure_hpa"),
        _inp(c, "temperature_k"),
        _inp(c, "relative_humidity"),
        DOY28_EPOCH_US,
    )
    assert abs(slant - _exp(c, "slant_m")) < RECONSTRUCTION_TOL_M


# Standalone mapping now surfaces the core's checked Niell validity range:
# 3 to 90 degrees. The slant-delay helper stays permissive for solver transients.
TROPO_MIN_MAPPING_ELEVATION_RAD = math.radians(3.0)


def test_below_horizon_mapping_rejects_with_core_error():
    """Low elevations surface the core mapping-domain error."""
    c = _zenith_case()
    lat = _inp(c, "lat_rad")
    height = _inp(c, "height_m")

    floored = sidereon.tropo_mapping_factors(
        TROPO_MIN_MAPPING_ELEVATION_RAD, lat, height, DOY28_EPOCH_US
    )
    assert all(math.isfinite(v) for v in floored)
    for el in (0.0, math.radians(1.0), math.radians(-5.0)):
        with pytest.raises(ValueError, match="below mapping validity"):
            sidereon.tropo_mapping_factors(el, lat, height, DOY28_EPOCH_US)


def test_below_horizon_slant_saturates_to_zero():
    """The composed slant below the horizon saturates to exactly 0.0 m."""
    c = _zenith_case()
    for el in (0.0, -0.1):
        slant = sidereon.tropo_slant_delay(
            el,
            _inp(c, "lat_rad"),
            _inp(c, "lon_rad"),
            _inp(c, "height_m"),
            _inp(c, "pressure_hpa"),
            _inp(c, "temperature_k"),
            _inp(c, "relative_humidity"),
            DOY28_EPOCH_US,
        )
        assert slant == 0.0
