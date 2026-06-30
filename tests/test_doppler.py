"""Satellite-ground Doppler and range-rate.

`doppler_shift` and `range_rate_and_ratio` are thin wrappers over
`sidereon_core::astro::doppler`. The state matches the core's
`numerical_derivative_oracle` fixture; the test asserts the two entries agree on
the shared quantities and that the carrier shift is exactly the ratio times the
frequency, so a wrapper that reorders arguments or drops the transport term is
caught.
"""

import numpy as np
import sidereon

# Matches the core doppler oracle test state.
GCRS_POS_KM = np.asarray(
    [3700.211211203995390, 2015.912218120605530, 5309.513078070447591], dtype=np.float64
)
GCRS_VEL_KM_S = np.asarray(
    [-3.398428894395407, 6.869656830559572, -0.239850181126689], dtype=np.float64
)
STATION_LAT_DEG = 40.0
STATION_LON_DEG = -74.0
STATION_ALT_KM = 0.0
FREQUENCY_HZ = 437.0e6


def _epoch():
    return sidereon.Instant.from_utc(2018, 7, 4, 0, 0, 30.0)


def test_doppler_shift_and_range_rate_agree():
    epoch = _epoch()
    shift = sidereon.doppler_shift(
        GCRS_POS_KM,
        GCRS_VEL_KM_S,
        STATION_LAT_DEG,
        STATION_LON_DEG,
        STATION_ALT_KM,
        epoch,
        FREQUENCY_HZ,
    )
    rr, ratio = sidereon.range_rate_and_ratio(
        GCRS_POS_KM,
        GCRS_VEL_KM_S,
        STATION_LAT_DEG,
        STATION_LON_DEG,
        STATION_ALT_KM,
        epoch,
    )
    # The two entries share the same kernel, bit for bit.
    assert shift.range_rate_km_s == rr
    assert shift.doppler_ratio == ratio
    # The carrier shift is exactly the ratio applied to the frequency.
    assert shift.doppler_hz == ratio * FREQUENCY_HZ


def test_doppler_quantities_are_physical():
    epoch = _epoch()
    shift = sidereon.doppler_shift(
        GCRS_POS_KM,
        GCRS_VEL_KM_S,
        STATION_LAT_DEG,
        STATION_LON_DEG,
        STATION_ALT_KM,
        epoch,
        FREQUENCY_HZ,
    )
    assert np.isfinite(shift.range_rate_km_s)
    assert np.isfinite(shift.doppler_hz)
    assert np.isfinite(shift.doppler_ratio)
    # A LEO-class link range rate is a few km/s in magnitude.
    assert abs(shift.range_rate_km_s) < 10.0
    # Receding (positive range rate) means approaching is false: the ratio is the
    # negated range rate over c, so the signs are opposite.
    assert np.sign(shift.doppler_ratio) == -np.sign(shift.range_rate_km_s)
