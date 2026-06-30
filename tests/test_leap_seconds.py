"""Leap-second accessor binding tests."""

import sidereon

# 2017-01-01 00:00 UTC as a UTC Julian date (after the 2017 leap second).
JD_2017 = 2457754.5
# 1999-01-01 00:00 UTC (TAI-UTC = 32 s, GPS-UTC = 13 s).
JD_1999 = 2451179.5


def test_2017_offsets():
    assert sidereon.gps_utc_offset_s(JD_2017) == 18.0
    assert sidereon.tai_utc_offset_s(JD_2017) == 37.0


def test_constant_19s_difference():
    for jd in (JD_1999, JD_2017):
        assert sidereon.tai_utc_offset_s(jd) - sidereon.gps_utc_offset_s(jd) == 19.0


def test_historical_offset():
    assert sidereon.tai_utc_offset_s(JD_1999) == 32.0
    assert sidereon.gps_utc_offset_s(JD_1999) == 13.0
