"""RINEX clock parsing through the Python binding uses a real committed fixture."""

import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES

CLK_FIXTURES = os.path.join(FIXTURES, "clk")
CLK = "synthetic_rinex_clock.clk"


def _read_clk():
    with open(os.path.join(CLK_FIXTURES, CLK), encoding="utf-8") as fh:
        return fh.read()


def _float_bits(value):
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def test_parse_rinex_clock_fixture_series_and_interpolation():
    clock = sidereon.parse_rinex_clock(_read_clk())

    assert clock.satellites == ["G05", "G24"]
    assert clock.satellite_count == 2
    assert clock.sample_count == 5
    assert "RinexClock(" in repr(clock)

    by_sat = {series.satellite: series for series in clock.series}
    g05 = by_sat["G05"]
    g24 = clock.series_for("G24")
    assert g24 is not None
    assert "ClockSeries(" in repr(g05)

    assert isinstance(g05.gps_seconds, np.ndarray)
    assert g05.gps_seconds.dtype == np.float64
    assert g05.gps_seconds.shape == (3,)
    assert g05.bias_s.shape == (3,)
    assert len(g05) == 3
    assert len(g24) == 2
    assert np.all(np.diff(g05.gps_seconds) == 30.0)
    assert _float_bits(g05.bias_s[1]) == 0xBF2A36E36F0D4275

    epoch = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 30.0)
    assert epoch.gps_seconds == g05.gps_seconds[1]
    assert _float_bits(clock.clock_s("G05", epoch)) == 0xBF2A36E36F0D4275

    g24_exact = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 0.0)
    g24_mid = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 15.0)
    assert _float_bits(clock.clock_s("G24", g24_exact)) == 0x3F0A36E2EB1C432D
    assert (
        _float_bits(clock.clock_s_at_gps_seconds("G24", g24_mid.gps_seconds))
        == 0x3F0A36E4A2EA40CA
    )
    assert clock.clock_s("G99", epoch) is None
    assert clock.clock_s("G05", sidereon.ClockEpoch(2026, 5, 13, 1, 0, 0.0)) is None

    with pytest.raises(ValueError):
        clock.clock_s_at_gps_seconds("G05", float("nan"))


def test_load_rinex_clock_accepts_path_and_bytes():
    path = os.path.join(CLK_FIXTURES, CLK)
    text = _read_clk()

    assert sidereon.load_rinex_clock(path).sample_count == 5
    assert sidereon.load_rinex_clock(text.encode("utf-8")).satellites == ["G05", "G24"]


def test_strict_rinex_clock_parse_errors_and_lossy_variant_skips_bad_rows():
    short_as = "AS G05  2026 05 13 00 00  0.000000  1\n"
    with pytest.raises(
        sidereon.RinexClockParseError, match="malformed RINEX AS clock record"
    ):
        sidereon.parse_rinex_clock(short_as)

    text = (
        "AS G05  2026 05 13 00 00  0.000000  1   1.0e-04\n"
        "AS G06  2026 05 13 00 00  bad-second  1   2.0e-04\n"
    )
    with pytest.raises(sidereon.RinexClockParseError, match="second=bad-second"):
        sidereon.parse_rinex_clock(text)

    lossy = sidereon.parse_rinex_clock_lossy(text)
    assert lossy.satellites == ["G05"]
    assert _float_bits(
        lossy.clock_s("G05", sidereon.ClockEpoch(2026, 5, 13, 0, 0, 0.0))
    ) == _float_bits(1.0e-4)
    assert sidereon.load_rinex_clock_lossy(short_as.encode("utf-8")).sample_count == 0


def test_to_rinex_string_round_trips_series():
    clock = sidereon.parse_rinex_clock(_read_clk())
    reparsed = sidereon.parse_rinex_clock(clock.to_rinex_string())

    assert reparsed.satellites == clock.satellites
    assert reparsed.sample_count == clock.sample_count
    by_sat = {series.satellite: series for series in reparsed.series}
    for series in clock.series:
        again = by_sat[series.satellite]
        np.testing.assert_array_equal(again.gps_seconds, series.gps_seconds)
        np.testing.assert_array_equal(again.bias_s, series.bias_s)
