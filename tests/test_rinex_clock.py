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


# Fixed-column record builders. The layout mirrors the core writer exactly: a
# 40-column `AS`/`AR` header (record type, 4-column name, civil epoch on the
# microsecond grid, value count) followed by 19-column E19.12 value fields,
# with values 3..6 on a continuation line.
def _record(
    name,
    values,
    record_type="AS",
    year=2026,
    month=5,
    day=13,
    hour=0,
    minute=0,
    second=0,
    microsecond=0,
):
    head = (
        f"{record_type} {name:<4} {year:04d} {month:02d} {day:02d} "
        f"{hour:02d} {minute:02d} {second:>2d}.{microsecond:06d}  {len(values)}   "
    )
    assert len(head) == 40, head
    lines = [head + " ".join(f"{value:>19}" for value in values[:2])]
    if len(values) > 2:
        lines.append(" ".join(f"{value:>19}" for value in values[2:]))
    return "".join(f"{line}\n" for line in lines)


def _clk_text(records, time_system=None):
    header = [f"{'     3.00           C':<60}RINEX VERSION / TYPE"]
    if time_system is not None:
        header.append(f"{time_system:<60}TIME SYSTEM ID")
    header.append(f"{'':<60}END OF HEADER")
    return "".join(f"{line}\n" for line in header) + "".join(records)


SIX_VALUES = [
    " 1.234567890123E-04",
    "-0.000000000000E+00",
    " 2.000000000000E-10",
    "-3.000000000000E-11",
    " 4.000000000000E-12",
    "-5.000000000000E-13",
]


def test_six_declared_values_survive_continuation_and_round_trip():
    text = _clk_text([_record("G01", SIX_VALUES)])
    clock = sidereon.parse_rinex_clock(text)

    series = clock.series_for("G01")
    assert len(series) == 1
    point = series.points[0]
    assert point.value_count == 6
    assert len(point.additional_values) == 5
    assert _float_bits(point.bias_s) == _float_bits(1.234567890123e-4)
    # A declared signed zero is data, not an absent value.
    assert _float_bits(point.bias_sigma_s) == _float_bits(-0.0)
    assert point.rate == 2.0e-10
    assert point.rate_sigma == -3.0e-11
    assert point.acceleration_per_s == 4.0e-12
    assert point.acceleration_sigma_per_s == -5.0e-13
    assert series.additional_values == [point.additional_values]

    # An unedited product restates its input byte for byte.
    assert clock.to_rinex_string() == text

    # Editing the record makes the writer state every value itself; each one,
    # the signed zero included, reads back to the same bits.
    values = [point.bias_s, *point.additional_values]
    clock.set_record_values(0, values)
    assert clock.records[0].reading.kind == "Edited"
    serialized = clock.to_rinex_string()
    # Parent record plus one continuation record for values 3..6.
    assert serialized.splitlines()[-2:] == [
        "AS G01  2026 05 13 00 00  0.000000  6    1.234567890123E-04 "
        "-0.000000000000E+00",
        " 0.200000000000E-09 -0.300000000000E-10  0.400000000000E-11 "
        "-0.500000000000E-12",
    ]
    reparsed = sidereon.parse_rinex_clock(serialized).series_for("G01").points[0]
    assert [
        _float_bits(value) for value in [reparsed.bias_s, *reparsed.additional_values]
    ] == [_float_bits(value) for value in values]
    assert _float_bits(reparsed.bias_sigma_s) == _float_bits(-0.0)


def test_declared_zero_and_absent_values_stay_distinct():
    text = _clk_text(
        [
            _record("G01", [" 1.000000000000E-04"]),
            _record("G02", [" 1.000000000000E-04", " 0.000000000000E+00"]),
        ]
    )
    clock = sidereon.parse_rinex_clock(text)

    absent = clock.series_for("G01").points[0]
    declared_zero = clock.series_for("G02").points[0]

    assert absent.value_count == 1
    assert absent.additional_values == []
    assert absent.bias_sigma_s is None
    assert declared_zero.value_count == 2
    assert declared_zero.additional_values == [0.0]
    assert declared_zero.bias_sigma_s == 0.0
    assert _float_bits(declared_zero.bias_sigma_s) == _float_bits(0.0)
    assert declared_zero.rate is None
    assert declared_zero.acceleration_sigma_per_s is None


def test_utc_rows_are_retained_with_unavailable_gps_seconds():
    text = _clk_text(
        [
            _record("G05", [" 1.000000000000E-04"]),
            _record("G05", [" 2.000000000000E-04"], second=30),
        ],
        time_system="UTC",
    )
    clock = sidereon.parse_rinex_clock(text)

    assert clock.time_scale == sidereon.TimeScale.UTC
    series = clock.series_for("G05")
    assert len(series) == 2
    assert series.bias_s.shape == (2,)
    assert series.gps_seconds.shape == (2,)
    assert series.gps_seconds_valid.dtype == np.bool_
    # Every row is kept; none of them projects into GPS seconds.
    assert np.all(np.isnan(series.gps_seconds))
    assert not series.gps_seconds_valid.any()
    assert [point.gps_seconds for point in series.points] == [None, None]
    np.testing.assert_array_equal(series.bias_s, np.array([1.0e-4, 2.0e-4]))

    for epoch in series.epochs:
        assert epoch.scale == sidereon.TimeScale.UTC
        assert epoch.repr_kind == "JulianDate"
        assert epoch.nanos is None
        assert epoch.jd_whole is not None and epoch.fraction is not None
        assert epoch.gps_seconds is None

    # Civil query fields are read in the file's own scale.
    midpoint = clock.clock_s("G05", sidereon.ClockEpoch(2026, 5, 13, 0, 0, 15.0))
    assert midpoint == pytest.approx(1.5e-4, abs=1.0e-18)
    utc_instant = sidereon.ClockInstant.from_civil(
        sidereon.TimeScale.UTC, 2026, 5, 13, 0, 0, 15.0
    )
    assert clock.clock_s_at_instant("G05", utc_instant) == midpoint
    assert clock.clock_s_at_instant("G05", series.epochs[0]) == 1.0e-4

    # A GPST-tagged instant is on another timeline and is not converted.
    gpst_instant = sidereon.ClockInstant.from_civil(
        sidereon.TimeScale.GPST, 2026, 5, 13, 0, 0, 15.0
    )
    assert clock.clock_s_at_instant("G05", gpst_instant) is None
    assert clock.clock_s_at_gps_seconds("G05", gpst_instant.gps_seconds) is None


def test_qzsst_rows_keep_their_scale_and_answer_on_the_gpst_timeline():
    text = _clk_text(
        [
            _record("J02", [" 1.000000000000E-04"]),
            _record("J02", [" 3.000000000000E-04"], second=30),
        ],
        time_system="QZS",
    )
    clock = sidereon.parse_rinex_clock(text)

    assert clock.time_scale == sidereon.TimeScale.QZSST
    series = clock.series_for("J02")
    assert len(series) == 2
    assert [epoch.scale for epoch in series.epochs] == [
        sidereon.TimeScale.QZSST,
        sidereon.TimeScale.QZSST,
    ]
    # QZSST shares the GPST alignment to TAI, so the core projects its samples
    # to GPS seconds, on the same timeline the GPS-seconds query answers on.
    assert series.gps_seconds_valid.all()
    np.testing.assert_array_equal(
        series.gps_seconds,
        np.array(
            [
                sidereon.ClockEpoch(2026, 5, 13, 0, 0, 0.0).gps_seconds,
                sidereon.ClockEpoch(2026, 5, 13, 0, 0, 30.0).gps_seconds,
            ]
        ),
    )
    assert [point.gps_seconds for point in series.points] == list(series.gps_seconds)

    # QZSST is synchronous with GPST, so the core answers a GPS-seconds query.
    mid = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 15.0).gps_seconds
    assert clock.clock_s_at_gps_seconds("J02", mid) == pytest.approx(
        2.0e-4, abs=1.0e-12
    )
    assert clock.clock_s("J02", sidereon.ClockEpoch(2026, 5, 13, 0, 0, 0.0)) == 1.0e-4


def test_gpst_epoch_views_are_exact_and_detached():
    clock = sidereon.parse_rinex_clock(_read_clk())
    series = clock.series_for("G05")

    assert clock.time_scale == sidereon.TimeScale.GPST
    assert len(series.epochs) == len(series) == len(series.points)
    for index, epoch in enumerate(series.epochs):
        assert epoch.scale == sidereon.TimeScale.GPST
        assert epoch.repr_kind == "JulianDate"
        assert epoch.nanos is None
        assert epoch.gps_seconds == series.gps_seconds[index]
        assert series.gps_seconds_valid[index]
    assert series.epochs[0] == sidereon.ClockInstant.from_civil(
        sidereon.TimeScale.GPST, 2026, 5, 13, 0, 0, 0.0
    )
    assert series.epochs[0] != series.epochs[1]

    # Returned views are detached copies of the stored sample.
    assert series.points[0] is not series.points[0]
    assert series.points[0] == series.points[0]
    assert "ClockPoint(" in repr(series.points[0])
    assert "ClockInstant(" in repr(series.epochs[0])


def test_committed_fixture_reports_its_unmodelled_receiver_record():
    clock = sidereon.parse_rinex_clock(_read_clk())

    assert clock.skipped_record_count == 1
    skipped = clock.skipped_records[0]
    assert skipped.record_type == "AR"
    assert skipped.line == 4
    assert "AR" in skipped.message
    assert clock.diagnostics == []
    assert clock.diagnostic_count == 0
    assert clock.sample_count == 5


def test_multi_line_unmodelled_record_is_reported_once():
    text = _clk_text(
        [
            _record("G01", [" 1.000000000000E-04"]),
            _record("ALIC", SIX_VALUES, record_type="AR"),
            _record("G01", [" 1.500000000000E-04"], second=30),
        ]
    )
    clock = sidereon.parse_rinex_clock(text)

    assert clock.sample_count == 2
    assert clock.skipped_record_count == 1
    skipped = clock.skipped_records[0]
    assert skipped.record_type == "AR"
    # One logical record - parent line plus its continuation - reported once,
    # at the parent line.
    parent_line = next(
        number
        for number, line in enumerate(text.splitlines(), 1)
        if line.startswith("AR ")
    )
    assert skipped.line == parent_line
    assert clock.diagnostics == []
    # The record outside the series is retained with its continuation line and
    # every value, and the writer restates it.
    assert [record.record_type for record in clock.records] == [
        sidereon.ClockRecordType.AS,
        sidereon.ClockRecordType.AR,
        sidereon.ClockRecordType.AS,
    ]
    alic = clock.records[1]
    assert alic.name == "ALIC"
    assert alic.line == parent_line
    assert alic.line_count == 2
    assert [_float_bits(value) for value in alic.values] == [
        _float_bits(float(value)) for value in SIX_VALUES
    ]
    assert clock.to_rinex_string() == text


def test_strict_parse_refuses_malformed_continuation_lossy_keeps_diagnostics():
    bad = list(SIX_VALUES)
    bad[3] = "bad-value"
    text = _clk_text(
        [
            _record("G01", [" 1.000000000000E-04"]),
            _record("G02", bad),
            _record("G03", [" 2.000000000000E-04"]),
        ]
    )
    lines = text.splitlines()
    continuation_line = next(
        number for number, line in enumerate(lines, 1) if "bad-value" in line
    )

    with pytest.raises(sidereon.RinexClockParseError) as strict:
        sidereon.parse_rinex_clock(text)
    detail = strict.value.detail
    assert detail is not None
    assert detail.kind == "MalformedContinuation"
    assert detail.line == continuation_line
    assert detail.reason == "invalid numeric field"
    assert detail.record == lines[continuation_line - 1].strip()
    assert detail.details() == {
        "line": continuation_line,
        "reason": "invalid numeric field",
        "record": lines[continuation_line - 1].strip(),
    }
    assert detail.field is None
    assert detail.value is None
    assert detail.time_scale is None

    lossy = sidereon.parse_rinex_clock_lossy(text)
    assert lossy.satellites == ["G01", "G03"]
    assert lossy.diagnostic_count == 1
    diagnostic = lossy.diagnostics[0]
    assert diagnostic.line == continuation_line
    assert diagnostic.error.kind == "MalformedContinuation"
    assert diagnostic.error == detail
    assert "line" in diagnostic.message


def test_lossy_diagnostics_stay_in_input_order_with_their_variants():
    text = _clk_text(
        [
            "AS X01  2026 05 13 00 00  0.000000  1    1.000000000000E-04\n",
            _record("G01", [" 1.000000000000E-04"]),
            "AS G02  2026 05 13 00 00  0.000000  1\n",
        ]
    )
    lossy = sidereon.parse_rinex_clock_lossy(text)

    assert lossy.satellites == ["G01"]
    assert [diagnostic.error.kind for diagnostic in lossy.diagnostics] == [
        "BadField",
        "MalformedAsRecord",
    ]
    first, second = lossy.diagnostics
    assert first.line < second.line
    assert first.error.field == "satellite"
    assert first.error.value == "X01"
    assert second.error.reason == "expected at least 10 fields"
    assert second.error.record.startswith("AS G02")


def test_unsupported_header_time_system_is_a_named_bad_field():
    text = _clk_text([_record("G01", [" 1.000000000000E-04"])], time_system="TCG")

    with pytest.raises(sidereon.RinexClockParseError) as excinfo:
        sidereon.parse_rinex_clock(text)
    detail = excinfo.value.detail
    assert detail.kind == "BadField"
    assert detail.field == "time_system"
    assert detail.value == "TCG"
    assert detail.line == 2


def test_parsed_value_is_restated_and_an_unstatable_edit_is_refused():
    # 0.30000000000000004 needs 17 significant digits. A product read from text
    # restates the field it read, so the parsed product writes back exactly.
    text = _clk_text([_record("G01", ["0.30000000000000004"])])
    clock = sidereon.parse_rinex_clock(text)
    assert _float_bits(clock.series_for("G01").points[0].bias_s) == _float_bits(
        0.1 + 0.2
    )
    assert clock.to_rinex_string() == text

    # Writing the value itself needs a 19-column field that reads back to the
    # same bits; none does, so an edit to it is refused and changes nothing.
    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        clock.set_record_values(0, [0.1 + 0.2])
    assert isinstance(excinfo.value, ValueError)
    detail = excinfo.value.detail
    assert detail.kind == "InvalidInput"
    assert detail.field == "bias"
    assert detail.line is None
    assert detail.reason == (
        "value cannot be represented in Fortran E19.12 format without loss of precision"
    )
    assert detail.details() == {"field": "bias", "reason": detail.reason}
    assert clock.records[0].reading.kind == "Columns"
    assert clock.to_rinex_string() == text

    # A product built from the value holds it as a typed value, so the writer
    # refuses it by name.
    built = sidereon.RinexClock.from_clock_points(
        sidereon.TimeScale.GPST,
        [("G01", [sidereon.ClockPoint(clock.series_for("G01").epochs[0], 0.1 + 0.2)])],
    )
    with pytest.raises(sidereon.RinexClockWriteError) as write_error:
        built.to_rinex_string()
    assert isinstance(write_error.value, ValueError)
    assert isinstance(write_error.value, sidereon.SidereonError)
    assert write_error.value.detail == detail


def test_query_refusals_carry_named_invalid_input_details():
    clock = sidereon.parse_rinex_clock(_read_clk())

    for bad in (float("nan"), float("inf")):
        with pytest.raises(sidereon.RinexClockQueryError) as excinfo:
            clock.clock_s_at_gps_seconds("G05", bad)
        assert isinstance(excinfo.value, ValueError)
        detail = excinfo.value.detail
        assert detail.kind == "InvalidInput"
        assert detail.field == "gps_seconds"
        assert detail.reason == "must be finite"
        assert detail.details() == {
            "field": "gps_seconds",
            "reason": "must be finite",
        }

    # GPS seconds past the civil year 9999 name no clock epoch; the core refuses
    # them instead of panicking.
    for bad in (1.0e20, -1.0e20):
        with pytest.raises(sidereon.RinexClockQueryError) as excinfo:
            clock.clock_s_at_gps_seconds("G05", bad)
        detail = excinfo.value.detail
        assert detail.kind == "InvalidInput"
        assert detail.field == "gps_seconds"
        assert detail.reason == (
            "outside the civil years 1 through 9999 that a clock epoch can name"
        )

    # A civil query the product's scale cannot read is a query refusal too; it
    # is still a RinexClockParseError, which clock_s raised before.
    with pytest.raises(sidereon.RinexClockQueryError) as excinfo:
        clock.clock_s("G05", sidereon.ClockEpoch(2016, 12, 31, 23, 59, 60.0))
    assert isinstance(excinfo.value, sidereon.RinexClockParseError)
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.detail.details() == {
        "field": "epoch",
        "reason": "invalid civil clock epoch",
    }


def test_epoch_constructors_reject_invalid_fields_without_core_detail():
    # Both constructors sit on core helpers that return an optional value and
    # no error payload, so neither fabricates a typed detail.
    with pytest.raises(ValueError) as epoch_error:
        sidereon.ClockEpoch(2026, 13, 13, 0, 0, 0.0)
    assert getattr(epoch_error.value, "detail", None) is None

    with pytest.raises(ValueError) as instant_error:
        sidereon.ClockInstant.from_civil(
            sidereon.TimeScale.UTC, 2026, 13, 13, 0, 0, 0.0
        )
    assert getattr(instant_error.value, "detail", None) is None


def test_clock_exceptions_keep_their_pre_existing_catch_surface():
    assert issubclass(sidereon.RinexClockParseError, sidereon.ParseError)
    assert issubclass(sidereon.RinexClockWriteError, sidereon.SidereonError)
    assert issubclass(sidereon.RinexClockWriteError, ValueError)
    # clock_s raised RinexClockParseError and the instant and GPS-seconds
    # queries raised ValueError; the query error is both.
    assert issubclass(sidereon.RinexClockQueryError, sidereon.RinexClockParseError)
    assert issubclass(sidereon.RinexClockQueryError, ValueError)
    assert issubclass(sidereon.RinexClockEditError, sidereon.SidereonError)
    assert issubclass(sidereon.RinexClockEditError, ValueError)
    # A hand-built instance carries no core context.
    assert sidereon.RinexClockParseError("x").detail is None
    assert sidereon.RinexClockWriteError("x").detail is None
    assert sidereon.RinexClockQueryError("x").detail is None
    assert sidereon.RinexClockEditError("x").detail is None
