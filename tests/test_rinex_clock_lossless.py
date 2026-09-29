"""Lossless RINEX clock products through the Python binding.

The fixtures are the core's ``clk/lossless`` set - the RINEX clock 3.00 and
3.04 specification examples, byte for byte with their CRLF terminators, and the
header and first epoch of four IGS analysis-centre products - plus the two
products in ``clk``. They are read as bytes so their line terminators reach the
parser unchanged.
"""

import os
import struct

import pytest
import sidereon
from _helpers import CORE_FIXTURES

LOSSLESS = (
    "rinex_clock300_table_a17.clk",
    "rinex_clock300_table_a18.clk",
    "rinex_clock304_table_a17.clk",
    "rinex_clock304_another_example.clk",
    "rinex_clock304_table_a18.clk",
    "COD0OPSRAP_20262600000_first_epoch_excerpt.clk",
    "EMR0OPSRAP_20262600000_first_epoch_excerpt.clk",
    "ESA0OPSRAP_20262600000_first_epoch_excerpt.clk",
    "GBM0MGXRAP_20262600000_first_epoch_excerpt.clk",
)
OTHER = (
    "synthetic_rinex_clock.clk",
    "IGS0OPSFIN_20261330000_90M_30S_CLK.CLK",
)
EMR = "EMR0OPSRAP_20262600000_first_epoch_excerpt.clk"

AS = sidereon.ClockRecordType.AS
AR = sidereon.ClockRecordType.AR


def _path(name):
    folder = ("clk",) if name in OTHER else ("clk", "lossless")
    return os.path.join(CORE_FIXTURES, *folder, name)


def _text(name):
    with open(_path(name), "rb") as fh:
        return fh.read().decode("utf-8")


def _v300(system):
    """A RINEX clock 3.00 `RINEX VERSION / TYPE` line for a satellite system."""
    prefix = "     3.00           C                   " + system
    return f"{prefix:<60}RINEX VERSION / TYPE\n"


def _v304(system):
    """A RINEX clock 3.04 `RINEX VERSION / TYPE` line for a satellite system."""
    prefix = "3.04                 C                    " + system
    return f"{prefix:<65}RINEX VERSION / TYPE\n"


def _bits(value):
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def _header_field(clock, label, payload_prefix):
    for record in clock.header_records:
        if record.label == label and record.payload.startswith(payload_prefix):
            assert record.field is not None
            return record.field
    raise AssertionError(f"no typed {label} record starting {payload_prefix!r}")


def _notices(clock):
    return [
        (notice.kind, notice.system, notice.line, notice.records, notice.first_line)
        for notice in clock.notices
    ]


@pytest.mark.parametrize("name", LOSSLESS + OTHER)
def test_every_fixture_restates_its_bytes(name):
    text = _text(name)
    strict = sidereon.load_rinex_clock(_path(name))
    assert strict.to_rinex_string() == text
    lossy = sidereon.parse_rinex_clock_lossy(text)
    assert lossy.diagnostics == []
    assert lossy == strict
    assert lossy.to_rinex_string() == text
    assert sidereon.parse_rinex_clock(strict.to_rinex_string()) == strict
    # A product read from text writes the same bytes under any policy.
    text_again, departures = strict.to_rinex_string_with_policy(
        sidereon.ClockWritePolicy.lenient()
    )
    assert text_again == text
    assert departures == []


def test_spec_300_example_reads_every_header_record_and_record():
    clock = sidereon.load_rinex_clock(_path("rinex_clock300_table_a17.clk"))
    assert clock.version == 3.0
    assert clock.layout == sidereon.ClockLayout.V300
    assert clock.satellite_system == "G"
    assert clock.time_system == sidereon.ClockTimeSystem.GPS
    assert clock.time_system_status.kind == "Declared"
    assert clock.time_scale == sidereon.TimeScale.GPST
    assert clock.notices == []

    header = clock.header_records
    assert len(header) == 26
    assert all(record.reading == "Columns" for record in header)
    assert all(record.label_column == 60 for record in header)
    assert header[0].line == 1
    assert header[0].field.kind == "VersionType"
    assert header[0].field.details() == {
        "version": 3.0,
        "file_type": "CLOCK DATA",
        "satellite_system": "GPS",
    }
    assert _header_field(clock, "SYS / # / OBS TYPES", "G").details() == {
        "system": "G",
        "count": 4,
        "descriptors": ["C1W", "L1W", "C2W", "L2W"],
    }
    assert _header_field(clock, "LEAP SECONDS", "").details() == {"seconds": 10}
    assert _header_field(clock, "SYS / DCBS APPLIED", "G").details() == {
        "system": "G",
        "program": "CC2NONCC",
        "source": "p1c1bias.hist @ goby.nrl.navy.mil",
    }
    clock_ref = _header_field(clock, "# OF CLK REF", "")
    assert clock_ref.kind == "ClockRefCount"
    assert clock_ref.details() == {
        "count": 1,
        "start": sidereon.ClockEpoch(1994, 7, 14, 0, 0, 0.0),
        "stop": sidereon.ClockEpoch(1994, 7, 14, 20, 59, 0.0),
    }
    assert _header_field(clock, "ANALYSIS CLK REF", "TIDB").details() == {
        "name": "TIDB",
        "identifier": "50103M108",
        "constraint_s": -0.123456789012,
    }
    assert _header_field(clock, "SOLN STA NAME / NUM", "AREQ").details() == {
        "name": "AREQ",
        "identifier": "42202M005",
        "xyz_mm": [-1234567890, 1234567890, -1234567890],
    }
    assert _header_field(clock, "# OF SOLN SATS", "").details() == {"count": 27}
    assert len(_header_field(clock, "PRN LIST", "G01").details()["prns"]) == 15

    records = clock.records
    assert clock.record_count == len(records) == 5
    areq = records[0]
    assert areq.record_type == AR
    assert areq.name == "AREQ"
    assert areq.satellite is None
    assert areq.line == 27
    assert areq.line_count == 2
    assert areq.reading.kind == "Columns"
    assert areq.reading.layout == sidereon.ClockLayout.V300
    assert areq.continuation_reading.layout == sidereon.ClockLayout.V300
    assert areq.declared_count == 6
    assert areq.values == [
        -0.123456789012,
        -1.23456789012,
        -12.3456789012,
        -123.456789012,
        -1234.56789012,
        -12345.6789012,
    ]
    assert areq.clock_point() is None
    assert records[2].values == [
        -0.0123456789012,
        -0.00123456789012,
        -0.000123456789012,
        -0.0000123456789012,
    ]
    g16 = records[1]
    assert g16.satellite == "G16"
    assert g16.continuation_reading is None
    assert g16.clock_point() == clock.series_for("G16").points[0]
    assert clock.satellites == ["G16"]
    assert clock.series_for("G16").additional_values == [[-0.0123456789012]]
    assert [(skip.line, skip.record_type) for skip in clock.skipped_records] == [
        (27, "AR"),
        (30, "AR"),
        (32, "AR"),
        (33, "AR"),
    ]
    assert clock.source_line(29).startswith("AS G16  1994 07 14 20 59")
    assert clock.source_line(1000) is None


def test_a_304_file_without_time_system_takes_the_300_default_and_can_declare_it():
    name = "rinex_clock304_table_a18.clk"
    text = _text(name)
    clock = sidereon.load_rinex_clock(_path(name))
    assert clock.layout == sidereon.ClockLayout.V304
    assert clock.time_system == sidereon.ClockTimeSystem.GPS
    assert clock.time_system_status.kind == "Defaulted"
    assert clock.time_scale == sidereon.TimeScale.GPST
    assert _notices(clock) == [
        ("HeaderRecordNonconforming", None, 7, None, None),
        ("TimeSystemDefaulted", sidereon.ClockTimeSystem.GPS, None, None, None),
        ("TimeSystemMissing", None, None, None, None),
    ]
    records = clock.records
    assert len(records) == 4
    assert records[2].record_type == sidereon.ClockRecordType.DR
    assert records[2].civil_epoch == sidereon.ClockEpoch(1995, 7, 14, 22, 23, 14.5)
    assert records[2].epoch == sidereon.ClockInstant.from_civil(
        sidereon.TimeScale.GPST, 1995, 7, 14, 22, 23, 14.5
    )
    assert records[2].values == [-1.23456789012, 0.123456789012]
    assert clock.series == []

    # Declaring the time system writes one TIME SYSTEM ID record at the 3.04
    # columns, before LEAP SECONDS GNSS as Table A15 orders them, with the
    # file's CRLF terminator; every other line is unchanged.
    clock.set_time_system(sidereon.ClockTimeSystem.GPS)
    split_at = -1
    for _ in range(4):
        split_at = text.index("\r\n", split_at + 1)
    split_at += 2
    tsid = f"{'   GPS':<65}TIME SYSTEM ID\r\n"
    expected = text[:split_at] + tsid + text[split_at:]
    assert clock.to_rinex_string() == expected
    assert clock.time_system_status.kind == "Declared"
    inserted = clock.header_records[4]
    assert inserted.line is None
    assert inserted.label == "TIME SYSTEM ID"
    assert inserted.reading == "Columns"
    assert inserted.field.details() == {"label": "GPS"}
    reparsed = sidereon.parse_rinex_clock(expected)
    assert reparsed.time_scale == sidereon.TimeScale.GPST
    assert _notices(reparsed) == [("HeaderRecordNonconforming", None, 8, None, None)]


def test_surplus_sigma_is_kept_and_an_edit_may_not_drop_it():
    clock = sidereon.load_rinex_clock(_path(EMR))
    assert clock.version == 2.0
    assert clock.time_system_status.kind == "Declared"
    assert clock.record_count == 153
    assert _notices(clock) == [("SurplusValues", None, None, 49, 239)]
    as_records = [record for record in clock.records if record.record_type == AS]
    assert len(as_records) == 49
    assert all(
        record.declared_count == 1 and len(record.surplus_values) == 1
        for record in as_records
    )
    g01 = as_records[0]
    assert g01.line == 239
    assert g01.bias_s == 0.170710878415e-3
    [surplus] = g01.surplus_values
    assert (surplus.position, surplus.value) == (1, 5.556437046250e-12)
    assert clock.series_for("G01").points[0].additional_values == []

    g01_index = next(
        index for index, record in enumerate(clock.records) if record.name == "G01"
    )
    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        clock.set_record_values(g01_index, [1.0e-4])
    assert excinfo.value.detail.details() == {
        "field": "values",
        "reason": (
            "the record carries values beyond its declared count; the new values "
            "must restate them"
        ),
    }
    assert clock == sidereon.load_rinex_clock(_path(EMR))

    clock.set_record_values(g01_index, [0.170710878415e-3, 5.556437046250e-12])
    assert (
        "\nAS G01  2026 09 17 00 00  0.000000  2    0.170710878415E-03  "
        "0.555643704625E-11\n"
    ) in clock.to_rinex_string()
    assert clock.series_for("G01").points[0].additional_values == [5.556437046250e-12]
    assert _notices(clock) == [("SurplusValues", None, None, 48, 240)]


def test_edit_records_applies_a_whole_batch_or_nothing():
    def restate_sigma(record):
        if not record.surplus_values:
            return None
        return [record.bias_s, record.surplus_values[0].value]

    clock = sidereon.load_rinex_clock(_path(EMR))
    assert clock.edit_records(restate_sigma) == 49
    assert clock.notices == []
    assert all(
        len(point.additional_values) == 1
        for series in clock.series
        for point in series.points
    )

    # One refused edit in the batch - G02's bias alone would drop its sigma -
    # leaves every record unchanged.
    fresh = sidereon.load_rinex_clock(_path(EMR))

    def refuse_g02(record):
        if record.name == "G02":
            return [1.0e-4]
        return restate_sigma(record)

    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        fresh.edit_records(refuse_g02)
    assert excinfo.value.detail.field == "values"
    assert fresh == sidereon.load_rinex_clock(_path(EMR))

    # An exception from the callback propagates before anything changes.
    def fail(record):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        fresh.edit_records(fail)
    assert fresh == sidereon.load_rinex_clock(_path(EMR))


def test_retain_records_removes_records_with_their_continuation_lines():
    name = "rinex_clock300_table_a17.clk"
    text = _text(name)
    clock = sidereon.load_rinex_clock(_path(name))
    assert clock.retain_records(lambda record: record.record_type != AR) == 4
    assert [record.name for record in clock.records] == ["G16"]
    assert clock.skipped_records == []
    lines = text.splitlines(keepends=True)
    # Header (26 lines) and the G16 record at line 29.
    assert clock.to_rinex_string() == "".join(lines[:26]) + lines[28]

    def fail(record):
        raise ValueError("keep failed")

    again = sidereon.load_rinex_clock(_path(name))
    with pytest.raises(ValueError, match="keep failed"):
        again.retain_records(fail)
    assert again.to_rinex_string() == text


def test_inserted_records_are_written_in_the_file_layout():
    name = "rinex_clock304_table_a17.clk"
    clock = sidereon.load_rinex_clock(_path(name))
    areq = sidereon.ClockRecord(
        AR, "AREQ00USA", sidereon.ClockEpoch(1994, 7, 14, 21, 0, 0.0), [0.5]
    )
    assert areq.line is None
    assert areq.reading.kind == "Edited"
    clock.insert_record(5, areq)
    expected = (
        _text(name)
        + "AR AREQ00USA 1994 07 14 21 00  0.000000  1    0.500000000000E+00\r\n"
    )
    assert clock.to_rinex_string() == expected
    assert clock.record_count == 6

    # A nine-character name does not fit the 3.00 layout; nothing changes.
    v300_name = "rinex_clock300_table_a17.clk"
    v300 = sidereon.load_rinex_clock(_path(v300_name))
    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        v300.insert_record(0, areq)
    assert excinfo.value.detail.details() == {
        "field": "name",
        "reason": "wider than the name field of the layout",
    }
    assert v300.to_rinex_string() == _text(v300_name)

    # A satellite record goes into the series.
    g05 = sidereon.ClockRecord(
        AS, "G05", sidereon.ClockEpoch(1994, 7, 14, 20, 59, 0.0), [0.25, 0.5]
    )
    v300.insert_record(1, g05)
    assert v300.series_for("G05").points[0].bias_s == 0.25
    assert (
        "\nAS G05  1994 07 14 20 59  0.000000  2    0.250000000000E+00  "
        "0.500000000000E+00\nAS G16"
    ) in v300.to_rinex_string()


def test_unknown_record_types_and_unreadable_lines_are_retained():
    text = (
        _v300("G")
        + """\
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
XX G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
IB G01  2026 05 13 00 00  0.000000  3    0.100000000000E-03  0.1E-10
   0.1E-11
free text that is not a record
AR AREQ 2026 05 13 00 00  0.000000  2    0.100000000000E-03  not-a-number

AS G01  2026 05 13 00 00 30.000000  1    0.200000000000E-03
"""
    )
    with pytest.raises(sidereon.RinexClockParseError) as excinfo:
        sidereon.parse_rinex_clock(text)
    assert excinfo.value.detail.details() == {
        "line": 5,
        "field": "record_type",
        "value": "XX",
    }

    clock = sidereon.parse_rinex_clock_lossy(text)
    assert clock.to_rinex_string() == text
    assert [diagnostic.line for diagnostic in clock.diagnostics] == [5, 6, 7, 8, 9]
    assert clock.diagnostics[4].error.details() == {
        "line": 9,
        "field": "sigma",
        "value": "not-a-number",
    }
    assert clock.record_count == 2
    assert len(clock.series_for("G01")) == 2
    assert clock.source_line(8) == "free text that is not a record"

    # Removing a record leaves the retained unreadable lines in place.
    removed = clock.remove_record(1)
    assert removed.line == 11
    assert clock.to_rinex_string() == text.replace(
        "\nAS G01  2026 05 13 00 00 30.000000  1    0.200000000000E-03\n", "\n"
    )
    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        clock.remove_record(99)
    assert excinfo.value.detail.details() == {
        "field": "index",
        "reason": "no record at this index",
    }


def test_glo_is_utc_and_a_leap_second_label_is_an_epoch():
    text = (
        _v300("R")
        + """\
TESTPGM             TESTAGENCY          20260917 000000 UTC PGM / RUN BY / DATE
A COMMENT THAT MUST SURVIVE                                 COMMENT
   GLO                                                      TIME SYSTEM ID
    18                                                      LEAP SECONDS
     2    AR    AS                                          # / TYPES OF DATA
                                                            END OF HEADER
AR ONSA 2026 09 17 00 00  0.000000  1    0.100000000000E-03
AS R01  2026 09 17 00 00  0.000000  1    0.200000000000E-03
AS R01  2026 09 17 00 00 30.000000  1    0.300000000000E-03
"""
    )
    clock = sidereon.parse_rinex_clock(text)
    assert clock.time_system == sidereon.ClockTimeSystem.GLO
    assert clock.time_scale == sidereon.TimeScale.UTC
    assert clock.to_rinex_string() == text
    assert clock.notices == []
    assert _header_field(clock, "COMMENT", "").details() == {
        "text": "A COMMENT THAT MUST SURVIVE"
    }

    text_304 = (
        _v304("R")
        + """\
   GLO                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AS R01       2016 12 31 23 59 60.000000  1    0.200000000000E-03
AS R01       2017 01 01 00 00  0.000000  1    0.300000000000E-03
"""
    )
    clock_304 = sidereon.parse_rinex_clock(text_304)
    assert clock_304.time_scale == sidereon.TimeScale.UTC
    assert len(clock_304.series_for("R01")) == 2
    leap = sidereon.ClockEpoch(2016, 12, 31, 23, 59, 60.5)
    # The label names no GPS-time epoch, and it is still a valid query here.
    assert leap.gps_seconds is None
    half = clock_304.clock_s("R01", leap)
    assert half == pytest.approx(2.5e-4, abs=1.0e-15)
    assert clock_304.records[0].civil_epoch == sidereon.ClockEpoch(
        2016, 12, 31, 23, 59, 60.0
    )
    assert clock_304.to_rinex_string() == text_304

    # GLONASS system time (UTC(SU) + 3 h) has no RINEX clock label.
    assert sidereon.ClockTimeSystem.GLO.time_scale == sidereon.TimeScale.UTC
    assert sidereon.ClockTimeSystem.for_time_scale(sidereon.TimeScale.GLONASST) is None
    epoch = sidereon.ClockInstant.from_civil(
        sidereon.TimeScale.GLONASST, 2026, 9, 17, 3, 0, 0.0
    )
    built = sidereon.RinexClock.from_instant_series_rows(
        sidereon.TimeScale.GLONASST, [("R01", [(epoch, 2.0e-4)])]
    )
    with pytest.raises(sidereon.RinexClockWriteError) as excinfo:
        built.to_rinex_string()
    detail = excinfo.value.detail
    assert detail.kind == "UnsupportedTimeScale"
    assert detail.time_scale == sidereon.TimeScale.GLONASST


def test_a_utc_leap_second_label_is_queryable_and_a_gps_product_refuses_it():
    text = (
        _v300("G")
        + """\
   UTC                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G05  2016 12 31 23 59 30.000000  1    0.000000000000E+00
AS G05  2016 12 31 23 59 60.000000  1    0.300000000000E+02
AS G05  2017 01 01 00 00  0.000000  1    0.310000000000E+02
"""
    )
    clock = sidereon.parse_rinex_clock(text)
    leap = sidereon.ClockEpoch(2016, 12, 31, 23, 59, 60.0)
    assert clock.clock_s("G05", leap) == 30.0
    half = clock.clock_s("G05", sidereon.ClockEpoch(2016, 12, 31, 23, 59, 60.5))
    assert half == pytest.approx(30.5, abs=1.0e-9)
    before = clock.clock_s("G05", sidereon.ClockEpoch(2016, 12, 31, 23, 59, 45.0))
    assert before == pytest.approx(15.0, abs=1.0e-9)

    # Declaring GPS time would leave the 23:59:60 record naming no epoch, so the
    # change is refused and nothing changes.
    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        clock.set_time_system(sidereon.ClockTimeSystem.GPS)
    assert excinfo.value.detail.details() == {
        "line": 5,
        "field": "epoch",
        "value": "2016 12 31 23 59 60",
    }
    assert clock.time_scale == sidereon.TimeScale.UTC
    assert clock.to_rinex_string() == text

    gps = sidereon.parse_rinex_clock(
        text.replace("   UTC", "   GPS").replace(
            "AS G05  2016 12 31 23 59 60.000000  1    0.300000000000E+02\n", ""
        )
    )
    with pytest.raises(sidereon.RinexClockQueryError) as excinfo:
        gps.clock_s("G05", leap)
    assert excinfo.value.detail.details() == {
        "field": "epoch",
        "reason": "invalid civil clock epoch",
    }


def test_navic_time_keeps_civil_epochs_without_an_instant():
    text = (
        _v304("I")
        + """\
   IRN                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AS I01       2026 05 13 00 00  0.000000  1   -0.232835122007E-05
"""
    )
    clock = sidereon.parse_rinex_clock(text)
    assert clock.time_system == sidereon.ClockTimeSystem.IRN
    assert sidereon.ClockTimeSystem.IRN.time_scale is None
    assert clock.time_scale is None
    assert _notices(clock) == [
        ("TimeSystemWithoutScale", sidereon.ClockTimeSystem.IRN, None, None, None)
    ]
    [record] = clock.records
    assert record.epoch is None
    assert record.clock_point() is None
    assert record.civil_epoch == sidereon.ClockEpoch(2026, 5, 13, 0, 0, 0.0)
    assert clock.series == []
    assert clock.to_rinex_string() == text
    with pytest.raises(sidereon.RinexClockQueryError) as excinfo:
        clock.clock_s("I01", sidereon.ClockEpoch(2026, 5, 13, 0, 0, 0.0))
    assert excinfo.value.detail.field == "time_system"


def test_an_unrecognised_time_system_is_not_read_as_gps_time():
    text = (
        _v300("G")
        + """\
   XYZ                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
"""
    )
    error = {"line": 2, "field": "time_system", "value": "XYZ"}
    with pytest.raises(sidereon.RinexClockParseError) as excinfo:
        sidereon.parse_rinex_clock(text)
    assert excinfo.value.detail.details() == error

    clock = sidereon.parse_rinex_clock_lossy(text)
    status = clock.time_system_status
    assert (status.kind, status.label, status.labels) == ("Unrecognized", "XYZ", None)
    assert clock.time_system is None
    assert clock.time_scale is None
    assert clock.series == []
    assert clock.diagnostics[0].error.details() == error
    assert clock.to_rinex_string() == text

    clock.set_time_system(sidereon.ClockTimeSystem.GPS)
    assert clock.time_scale == sidereon.TimeScale.GPST
    assert len(clock.series_for("G01")) == 1
    assert clock.diagnostics == []
    assert clock.to_rinex_string() == text.replace("   XYZ", "   GPS")


def test_two_ar_records_at_one_epoch_are_both_retained():
    for text, layout in (
        (
            _v300("G")
            + """\
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
AR AREQ 1994 07 14 20 59  0.000000  2    0.100000000000E-03  0.100000000000E-10
AR AREQ 1994 07 14 20 59  0.000000  2    0.200000000000E-03  0.200000000000E-10
AS G16  1994 07 14 20 59  0.000000  1   -0.123456789012E+00
""",
            sidereon.ClockLayout.V300,
        ),
        (
            _v304("G")
            + """\
   GPS                                                           TIME SYSTEM ID
                                                                 END OF HEADER
AR AREQ00USA 1994 07 14 20 59  0.000000  2    0.100000000000E-03   0.100000000000E-10
AR AREQ00USA 1994 07 14 20 59  0.000000  2    0.200000000000E-03   0.200000000000E-10
AS G16       1994 07 14 20 59  0.000000  1   -0.123456789012E+00
""",
            sidereon.ClockLayout.V304,
        ),
    ):
        clock = sidereon.parse_rinex_clock(text)
        assert clock.layout == layout
        ar = [record for record in clock.records if record.record_type == AR]
        assert len(ar) == 2
        assert ar[0].civil_epoch == ar[1].civil_epoch
        assert ar[0].values == [1.0e-4, 1.0e-11]
        assert ar[1].values == [2.0e-4, 2.0e-11]
        assert ar[0].reading.layout == layout
        assert [(skip.line, skip.record_type) for skip in clock.skipped_records] == [
            (4, "AR"),
            (5, "AR"),
        ]
        assert clock.to_rinex_string() == text


def test_records_in_the_other_layout_are_read_and_reported():
    text = (
        _v300("G")
        + """\
   GPS                                                      TIME SYSTEM ID
                                                            END OF HEADER
AS G01  2026 05 13 00 00  0.000000  1    0.100000000000E-03
AS G02       2026 05 13 00 00  0.000000  1    0.200000000000E-03
"""
    )
    clock = sidereon.parse_rinex_clock(text)
    assert [record.reading.layout for record in clock.records] == [
        sidereon.ClockLayout.V300,
        sidereon.ClockLayout.V304,
    ]
    assert _notices(clock) == [("OtherLayoutRecords", None, None, 1, 5)]
    assert clock.series_for("G02").points[0].bias_s == 2.0e-4
    assert clock.to_rinex_string() == text


def test_an_epoch_off_the_microsecond_grid_is_refused_or_reported():
    epoch = sidereon.ClockInstant.from_civil(
        sidereon.TimeScale.GPST, 2026, 5, 13, 0, 0, 30.0000001
    )
    clock = sidereon.RinexClock.from_instant_series_rows(
        sidereon.TimeScale.GPST, [("G05", [(epoch, 1.0e-4)])]
    )
    assert clock.time_system_status.kind == "Constructed"
    assert clock.header_records == []
    with pytest.raises(sidereon.RinexClockWriteError) as excinfo:
        clock.to_rinex_string()
    assert excinfo.value.detail.details() == {
        "field": "epoch",
        "reason": "the epoch field cannot restate this instant without rounding it",
    }
    with pytest.raises(sidereon.RinexClockWriteError):
        clock.to_rinex_string_with_policy()

    policy = sidereon.ClockWritePolicy(
        nearest_microsecond_epochs=sidereon.ClockWriteLeniency.ALLOW
    )
    assert policy == sidereon.ClockWritePolicy.lenient()
    result = clock.to_rinex_string_with_policy(policy)
    assert len(result) == 2
    text, departures = result
    assert text == result.text
    [departure] = departures
    assert departure == result.departures[0]
    assert departure.kind == "EpochAtNearestMicrosecond"
    assert departure.record == 0
    assert departure.name == "G05"
    assert departure.epoch == epoch
    assert departure.written == "2026 05 13 00 00 30.000000"
    assert departure.details() == {
        "record": 0,
        "name": "G05",
        "epoch": epoch,
        "written": "2026 05 13 00 00 30.000000",
    }
    assert "AS G05  2026 05 13 00 00 30.000000" in text

    strict = sidereon.ClockWritePolicy()
    assert strict == sidereon.ClockWritePolicy.strict()
    assert strict.nearest_microsecond_epochs == sidereon.ClockWriteLeniency.STRICT
    assert (
        strict.with_nearest_microsecond_epochs(sidereon.ClockWriteLeniency.ALLOW)
        == policy
    )


def test_products_built_from_rows_keep_every_declared_value():
    parsed = sidereon.load_rinex_clock(_path("synthetic_rinex_clock.clk"))

    rebuilt = sidereon.RinexClock.from_series_rows(parsed.series_rows())
    assert rebuilt.time_system_status.kind == "Constructed"
    assert rebuilt.layout == sidereon.ClockLayout.V300
    assert rebuilt != parsed
    reread = sidereon.parse_rinex_clock(rebuilt.to_rinex_string())
    assert reread.series_rows() == parsed.series_rows()

    from_points = sidereon.RinexClock.from_clock_points(
        sidereon.TimeScale.GPST,
        [(series.satellite, series.points) for series in parsed.series],
    )
    reread = sidereon.parse_rinex_clock(from_points.to_rinex_string())
    for series in parsed.series:
        again = reread.series_for(series.satellite)
        assert again.epochs == series.epochs
        assert [
            [_bits(point.bias_s), *map(_bits, point.additional_values)]
            for point in again.points
        ] == [
            [_bits(point.bias_s), *map(_bits, point.additional_values)]
            for point in series.points
        ]

    instants = sidereon.RinexClock.from_instant_series_rows(
        sidereon.TimeScale.GPST, parsed.instant_series_rows()
    )
    assert instants.series_rows() == parsed.series_rows()

    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        sidereon.RinexClock.from_series_rows([("G01", [(30.0, 1.0), (0.0, 2.0)])])
    assert excinfo.value.detail.details() == {
        "field": "gps_seconds",
        "reason": "must be strictly increasing",
    }


def test_records_and_points_are_checked_as_the_core_checks_them():
    epoch = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 0.0)
    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        sidereon.ClockRecord(AS, "X01", epoch, [1.0e-4])
    assert excinfo.value.detail.details() == {
        "field": "satellite",
        "reason": "not a RINEX satellite identifier",
    }

    instant = sidereon.ClockInstant.from_civil(
        sidereon.TimeScale.GPST, 2026, 5, 13, 0, 0, 0.0
    )
    point = sidereon.ClockPoint(instant, float("nan"))
    with pytest.raises(sidereon.RinexClockEditError) as excinfo:
        point.validate()
    assert excinfo.value.detail.details() == {
        "field": "bias_s",
        "reason": "must be finite",
    }
    with pytest.raises(sidereon.RinexClockEditError):
        sidereon.RinexClock.from_clock_points(
            sidereon.TimeScale.GPST, [("G01", [point])]
        )
    good = sidereon.ClockPoint(instant, 1.0e-4, [2.0e-12])
    good.validate()
    assert good.value_count == 2
    assert good.bias_sigma_s == 2.0e-12


def test_clock_enums_mirror_the_core_labels():
    assert sidereon.ClockTimeSystem.from_label("BDT") == sidereon.ClockTimeSystem.BDS
    assert sidereon.ClockTimeSystem.from_label("XYZ") is None
    assert sidereon.ClockTimeSystem.BDS.label == "BDS"
    assert sidereon.ClockTimeSystem.QZS.time_scale == sidereon.TimeScale.QZSST
    assert (
        sidereon.ClockTimeSystem.for_time_scale(sidereon.TimeScale.BDT)
        == sidereon.ClockTimeSystem.BDS
    )
    assert sidereon.ClockRecordType.from_code("MS") == sidereon.ClockRecordType.MS
    assert sidereon.ClockRecordType.from_code("XX") is None
    assert sidereon.ClockRecordType.CR.code == "CR"
    assert sidereon.ClockLayout.for_version(3.04) == sidereon.ClockLayout.V304
    assert sidereon.ClockLayout.for_version(3.02) == sidereon.ClockLayout.V300
    assert sidereon.ClockLayout.V304.label_column == 65
    assert sidereon.ClockLayout.V300.name_width == 4
