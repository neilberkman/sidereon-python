import os
import pathlib
import struct
from decimal import Decimal

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES

ANTEX_FIXTURES = os.path.join(FIXTURES, "antex")
TEST_ID = "TESTANT             TESTSER"


def _antex_path(name):
    return os.path.join(ANTEX_FIXTURES, name)


def _core_antex(name):
    with open(os.path.join(CORE_FIXTURES, "antex", name), "rb") as fh:
        return fh.read().decode("ascii")


def _mm(values):
    # ANTEX millimetres become metres as `mm * 1e-3`, the arithmetic of RTKLIB
    # `readantex`, which the core matches bit for bit.
    return np.asarray(values, dtype=np.float64) * 1e-3


def _bits(value):
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


def _rec(prefix, tag):
    return f"{prefix:<60}{tag}"


def _one_antenna(records):
    lines = [
        _rec("", "START OF ANTENNA"),
        _rec(TEST_ID, "TYPE / SERIAL NO"),
        *records,
        _rec("", "END OF ANTENNA"),
    ]
    return "\n".join(lines)


def _grid_records(dazi, zen):
    return [_rec(dazi, "DAZI"), _rec(zen, "ZEN1 / ZEN2 / DZEN")]


def _frequency_section(label, pco, rows):
    return [
        _rec(f"   {label}", "START OF FREQUENCY"),
        _rec(pco, "NORTH / EAST / UP"),
        *rows,
        _rec(f"   {label}", "END OF FREQUENCY"),
    ]


def _trimmed_lines(text):
    return [line.rstrip() for line in text.splitlines()]


def test_load_antex_from_path_and_lookup_satellite_pco_pcv():
    antex = sidereon.load_antex(pathlib.Path(_antex_path("igs20_wettzell_trim.atx")))
    epoch = sidereon.AntexDateTime(2020, 6, 25)

    g05 = antex.satellite_antenna("G05", epoch)

    assert antex.antenna_count == 10
    assert g05 is not None
    assert g05.kind == sidereon.AntennaKind.SATELLITE
    assert g05.kind.label == "satellite"
    assert g05.serial == "G05"
    assert g05.valid_at(epoch)
    assert g05.valid_from == sidereon.AntexDateTime(2009, 8, 17)
    assert g05.valid_until is None
    assert "G01" in g05.frequencies
    np.testing.assert_array_equal(g05.pco("G01"), _mm([-3.30, -0.30, 742.63]))
    assert g05.pcv("G01", 9.0) == -9.50 * 1e-3
    assert antex.satellite_antenna("G99", epoch) is None


def test_antex_receiver_lookup_from_bytes():
    with open(_antex_path("igs20_wettzell_trim.atx"), "rb") as fh:
        antex = sidereon.load_antex(fh.read())

    receiver = antex.antenna("LEIAR25.R3      LEIT")

    assert receiver is not None
    assert receiver.kind == sidereon.AntennaKind.RECEIVER
    assert receiver.antenna_type == "LEIAR25.R3      LEIT"
    assert receiver.serial == ""
    assert receiver.valid_from is None
    np.testing.assert_array_equal(receiver.pco("G01"), _mm([-0.05, 0.95, 160.96]))
    assert receiver.pcv("G01", 10.0) == 0.99 * 1e-3


def test_load_second_antex_fixture_and_receiver_pco():
    antex = sidereon.load_antex(_antex_path("igs20_pasa_scoa_gps.atx"))
    receiver_id = next(id for id in antex.antenna_ids if id.startswith("LEIAR20"))
    receiver = antex.antenna(receiver_id)

    assert receiver is not None
    assert receiver.kind == sidereon.AntennaKind.RECEIVER
    assert receiver.antenna_type == "LEIAR20         LEIM"
    assert receiver.serial == ""
    np.testing.assert_array_equal(receiver.pco("G01"), _mm([0.50, 0.13, 124.88]))
    assert receiver.pcv("G01", 20.0) == -0.99 * 1e-3


def test_antex_validation_and_lookup_errors():
    antex = sidereon.load_antex(_antex_path("igs20_wettzell_trim.atx"))
    receiver = antex.antenna("LEIAR25.R3      LEIT")

    assert receiver is not None
    with pytest.raises(ValueError, match="invalid ANTEX datetime"):
        sidereon.AntexDateTime(2020, 2, 30)
    with pytest.raises(ValueError, match="unknown frequency"):
        receiver.pco("UNKNOWN")
    with pytest.raises(ValueError, match="zenith_deg must be finite"):
        receiver.pcv("G01", float("nan"))


def test_to_antex_string_round_trips_through_load():
    antex = sidereon.load_antex(_antex_path("igs20_wettzell_trim.atx"))
    text = antex.to_antex_string()
    assert isinstance(text, str)
    assert "ANTEX VERSION" in text

    reparsed = sidereon.load_antex(text.encode("ascii"))
    assert reparsed.antenna_count == antex.antenna_count
    assert reparsed.antenna_ids == antex.antenna_ids

    epoch = sidereon.AntexDateTime(2020, 6, 25)
    g05 = antex.satellite_antenna("G05", epoch)
    g05_again = reparsed.satellite_antenna("G05", epoch)
    assert g05 is not None and g05_again is not None
    np.testing.assert_array_equal(g05_again.pco("G01"), g05.pco("G01"))


def test_antex_error_names_a_write_field_only_on_a_write_refusal():
    # `to_antex_string` sets `field` and `reason` from the core's refusal; any
    # other instance keeps the class defaults.
    assert sidereon.AntexParseError.field is None
    assert sidereon.AntexParseError.reason is None
    assert sidereon.AntexParseError.detail is None
    hand_built = sidereon.AntexParseError("hand built")
    assert hand_built.field is None
    assert hand_built.reason is None
    assert hand_built.detail is None
    assert issubclass(sidereon.AntexParseError, sidereon.ParseError)
    assert issubclass(sidereon.AntexWriteError, sidereon.AntexParseError)
    assert issubclass(sidereon.AntexWriteError, ValueError)
    assert issubclass(sidereon.AntexQueryError, sidereon.SidereonError)
    assert issubclass(sidereon.AntexQueryError, ValueError)
    assert sidereon.AntexWriteError("x").detail is None
    assert sidereon.AntexQueryError("x").detail is None


def test_relative_pcv_type_reference_antenna_and_metadata_are_retained():
    text = _core_antex("igs_01_relative_trim.atx")
    antex = sidereon.load_antex(text.encode("ascii"))
    assert antex.skipped_records == 0

    header = antex.header
    assert _bits(header.version.version) == _bits(1.3)
    assert header.version.system == "M"
    pcv = header.pcv_type
    assert pcv.pcv_type == sidereon.AntexPcvType.RELATIVE
    assert pcv.reference_antenna_type == "AOAD/M_T"
    assert pcv.reference_antenna_serial == ""
    assert pcv.reference_antenna == "AOAD/M_T"
    assert len(header.comments) == 5
    assert header.comments[1] == (
        "igs_01.pcv (version from July 2007) converted to ANTEX"
    )
    assert header.end_of_header

    [block_i] = antex.antenna("BLOCK I").calibrations
    assert (
        block_i.method,
        block_i.agency,
        block_i.antennas_calibrated,
        block_i.date,
    ) == ("", "", 0, "21-APR-04")
    ash = antex.antenna("ASH700699.L1    NONE")
    [field] = ash.calibrations
    assert (field.method, field.agency, field.antennas_calibrated, field.date) == (
        "FIELD",
        "IGEX",
        None,
        "12-NOV-98",
    )
    # 51.50 mm reads as 51.50 * 1e-3, one unit in the last place away from
    # 51.50 / 1000.
    up = ash.pco("G01")[2]
    assert _bits(up) == _bits(51.5 * 1e-3)
    assert _bits(up) != _bits(51.5 / 1000.0)

    # Blocks keep file order, and the writer restates every record.
    assert [block.id for block in antex.antenna_blocks] == [
        "BLOCK I",
        "ASH700699.L1    NONE",
    ]
    encoded = antex.to_antex_string()
    assert _trimmed_lines(encoded) == _trimmed_lines(text)
    assert sidereon.load_antex(encoded.encode("ascii")) == antex


def test_absolute_pcv_type_and_fractional_validity_are_retained():
    text = _core_antex("igs20_block_i_g03_trim.atx")
    antex = sidereon.load_antex(text.encode("ascii"))
    assert antex.skipped_records == 0
    assert antex.header.pcv_type.pcv_type == sidereon.AntexPcvType.ABSOLUTE
    assert antex.header.pcv_type.reference_antenna is None

    g03 = antex.antenna("BLOCK I             G03                 G011      1985-093A")
    assert g03.valid_from == sidereon.AntexDateTime(1985, 10, 9)
    until = g03.valid_until
    assert until == sidereon.AntexDateTime(
        1994, 4, 17, 23, 59, 59, fraction_digits=9999999, fraction_scale=7
    )
    assert (until.second, until.fraction_digits, until.fraction_scale) == (
        59,
        9999999,
        7,
    )
    assert until.fraction == Decimal("0.9999999")
    assert "59.9999999" in repr(until)
    assert until.nanosecond == 999_999_900

    # The bound is the instant 0.1 microsecond before midnight.
    inside = sidereon.AntexDateTime(
        1994, 4, 17, 23, 59, 59, fraction_digits=5, fraction_scale=1
    )
    after = sidereon.AntexDateTime(
        1994, 4, 17, 23, 59, 59, fraction_digits=99999995, fraction_scale=8
    )
    assert inside < until < after
    assert g03.valid_at(inside)
    assert g03.valid_at(until)
    assert not g03.valid_at(after)
    assert antex.satellite_antenna("G03", inside) is not None
    assert antex.antenna_at(g03.id, after) is None
    assert antex.antenna_intervals(g03.id) == [g03]

    assert g03.dazi_deg == 0.0
    grid = g03.zenith_grid
    assert (grid.start_deg, grid.end_deg, grid.step_deg) == (0.0, 14.0, 1.0)
    assert (g03.zenith_start_deg, g03.zenith_end_deg, g03.zenith_step_deg) == (
        0.0,
        14.0,
        1.0,
    )
    assert g03.has_frequency_count
    assert g03.sinex_code == "IGS20_2434"
    assert g03.frequencies == ["G01", "G02"]
    sample = g03.frequency("G01").pcv_samples[1]
    assert sample.grid == sidereon.AntexPcvGrid.NO_AZIMUTH
    assert sample.azimuth_deg is None
    assert sample.zenith_deg == 1.0
    assert _bits(sample.value_m) == _bits(-2.6 * 1e-3)
    assert _bits(sample.value_m) != _bits(-2.6 / 1000.0)

    encoded = antex.to_antex_string()
    assert _trimmed_lines(encoded) == _trimmed_lines(text)
    assert _rec("  1994     4    17    23    59   59.9999999", "VALID UNTIL") in encoded
    assert sidereon.load_antex(encoded.encode("ascii")) == antex


def _valid_from_with_seconds(seconds):
    return _one_antenna(
        [_rec(f"  2020     1     1     0     0{seconds:>13}", "VALID FROM")]
    )


def _valid_from(seconds):
    antex = sidereon.load_antex(_valid_from_with_seconds(seconds).encode("ascii"))
    return antex, antex.antenna(TEST_ID).valid_from


@pytest.mark.parametrize(
    ("field", "second", "digits", "scale", "written"),
    [
        ("0.1234567890", 0, 123_456_789, 9, "0.123456789"),
        (".123456789012", 0, 123_456_789_012, 12, ".123456789012"),
        ("59.99D0", 59, 99, 2, "59.9900000"),
        ("1.2345678E-9", 0, 12_345_678, 16, "1.2345678E-9"),
        ("7E-13", 0, 7, 13, "7.E-13"),
    ],
)
def test_validity_seconds_are_kept_exactly_and_restated(
    field, second, digits, scale, written
):
    antex, valid_from = _valid_from(field)
    assert valid_from.second == second
    assert valid_from.fraction_digits == digits
    assert valid_from.fraction_scale == scale
    assert valid_from.fraction == Decimal(digits).scaleb(-scale)
    encoded = antex.to_antex_string()
    assert _rec(f"  2020     1     1     0     0{written:>13}", "VALID FROM") in encoded
    assert sidereon.load_antex(encoded.encode("ascii")) == antex


def test_validity_seconds_order_by_value_and_an_unwritable_second_is_refused():
    assert _valid_from("0.05")[1] < _valid_from("0.5")[1]
    assert _valid_from("0.50")[1] == _valid_from(".5")[1]
    assert _valid_from("1.2345678E-9")[1] < _valid_from("0.000000002")[1]
    assert _valid_from("1.2345678E-9")[1].nanosecond is None

    # A value whose digits fit the field only without a decimal point is
    # refused by name rather than written in a form Fortran reads differently.
    antex, valid_from = _valid_from("123456789E-99")
    assert (valid_from.fraction_digits, valid_from.fraction_scale) == (
        123_456_789,
        99,
    )
    with pytest.raises(sidereon.AntexWriteError) as excinfo:
        antex.to_antex_string()
    error = excinfo.value
    assert isinstance(error, sidereon.AntexParseError)
    assert isinstance(error, ValueError)
    assert error.detail.kind == "Unwritable"
    assert error.detail.field == "valid_from"
    assert error.field == "valid_from"
    assert error.reason == error.detail.reason


def test_malformed_validity_and_repeated_records_are_refused_by_name():
    text = _one_antenna(
        [_rec("  2016    12    31    23    59   60.0000000", "VALID FROM")]
    )
    with pytest.raises(sidereon.AntexParseError) as excinfo:
        sidereon.load_antex(text.encode("ascii"))
    assert excinfo.value.detail.details() == {
        "antenna_id": TEST_ID,
        "record": "VALID FROM",
        "field": "second",
        "value": "60.0000000",
    }
    with pytest.raises(ValueError, match="invalid ANTEX datetime"):
        sidereon.AntexDateTime(2016, 12, 31, 23, 59, 60)

    valid_from = _rec("  2020     1     1     0     0    0.0000000", "VALID FROM")
    other = _rec("  2021     1     1     0     0    0.0000000", "VALID FROM")
    assert sidereon.load_antex(_one_antenna([valid_from, valid_from]).encode())
    with pytest.raises(sidereon.AntexParseError) as excinfo:
        sidereon.load_antex(_one_antenna([valid_from, other]).encode("ascii"))
    detail = excinfo.value.detail
    assert detail.kind == "RepeatedRecord"
    assert (detail.antenna_id, detail.record) == (TEST_ID, "VALID FROM")

    header = "\n".join(
        [
            _rec("     1.4            M", "ANTEX VERSION / SYST"),
            _rec("X", "PCV TYPE / REFANT"),
            _rec("", "END OF HEADER"),
        ]
    )
    with pytest.raises(sidereon.AntexParseError) as excinfo:
        sidereon.load_antex(header.encode("ascii"))
    assert excinfo.value.detail.details() == {
        "antenna_id": None,
        "record": "PCV TYPE / REFANT",
        "field": "pcv type",
        "value": "X",
    }


def test_repeated_frequency_labels_keep_every_section_and_refuse_ambiguity():
    records = _grid_records("     0.0", "     0.0  10.0   5.0")
    records.append(_rec("     3", "# OF FREQUENCIES"))
    for label, mm in (("G01", "1.00"), ("G02", "3.00"), ("G01", "2.00")):
        records.extend(
            _frequency_section(
                label,
                f"      {mm}      {mm}      {mm}",
                [f"   NOAZI    {mm}    {mm}    {mm}"],
            )
        )
    antex = sidereon.load_antex(_one_antenna(records).encode("ascii"))
    assert antex.skipped_records == 0
    antenna = antex.antenna(TEST_ID)
    assert antenna.frequencies == ["G01", "G02", "G01"]
    sections = antenna.frequency_sections
    np.testing.assert_array_equal(sections[0].pco_m, [0.001, 0.001, 0.001])
    np.testing.assert_array_equal(sections[2].pco_m, [0.002, 0.002, 0.002])

    for lookup in (
        lambda: antenna.pco("G01"),
        lambda: antenna.pcv("G01", 5.0),
        lambda: antenna.frequency("G01"),
    ):
        with pytest.raises(sidereon.AntexQueryError) as excinfo:
            lookup()
        assert isinstance(excinfo.value, ValueError)
        assert excinfo.value.detail.details() == {
            "antenna_id": TEST_ID,
            "frequency": "G01",
            "sections": 2,
        }
    np.testing.assert_array_equal(antenna.pco("G02"), [0.003, 0.003, 0.003])
    assert antenna.frequency("G02") == sections[1]

    encoded = antex.to_antex_string()
    starts = [
        line[:60].strip()
        for line in encoded.splitlines()
        if line[60:].rstrip() == "START OF FREQUENCY"
    ]
    assert starts == ["G01", "G02", "G01"]
    assert sidereon.load_antex(encoded.encode("ascii")) == antex

    with pytest.raises(sidereon.AntexQueryError) as excinfo:
        antenna.pco("G05")
    assert excinfo.value.detail.kind == "UnknownFrequency"


def test_rms_sections_are_retained():
    records = _grid_records("     0.0", "     0.0  10.0   5.0")
    records.extend(
        _frequency_section(
            "G01",
            "      1.00      2.00      3.00",
            ["   NOAZI    1.00    2.00    3.00"],
        )
    )
    records.extend(
        [
            _rec("   G01", "START OF FREQ RMS"),
            _rec("      0.10      0.20      0.30", "NORTH / EAST / UP"),
            "   NOAZI    0.05    0.06    0.07",
            _rec("   G01", "END OF FREQ RMS"),
        ]
    )
    antex = sidereon.load_antex(_one_antenna(records).encode("ascii"))
    assert antex.skipped_records == 0
    frequency = antex.antenna(TEST_ID).frequency("G01")
    rms = frequency.rms
    np.testing.assert_array_equal(rms.pco_m, [0.1 * 1e-3, 0.2 * 1e-3, 0.3 * 1e-3])
    assert [sample.value_m for sample in rms.pcv_samples] == [
        0.05 * 1e-3,
        0.06 * 1e-3,
        0.07 * 1e-3,
    ]
    assert [sample.zenith_deg for sample in rms.pcv_samples] == [0.0, 5.0, 10.0]
    np.testing.assert_array_equal(frequency.pco_m, [0.001, 0.002, 0.003])
    assert len(frequency.pcv_samples) == 3
    assert sidereon.load_antex(antex.to_antex_string().encode("ascii")) == antex


def test_comments_and_block_order_are_retained_and_nothing_is_invented():
    source = [
        _rec("header note", "COMMENT"),
        _rec("", "END OF HEADER"),
        _rec("after the header", "COMMENT"),
        _rec("", "START OF ANTENNA"),
        _rec("before the type", "COMMENT"),
        _rec("ZZTEST              SER", "TYPE / SERIAL NO"),
        _rec("     0.0", "DAZI"),
        _rec("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        _rec("     1", "# OF FREQUENCIES"),
        _rec("IGS20", "SINEX CODE"),
        _rec("about ZZ", "COMMENT"),
        _rec("G01", "START OF FREQUENCY"),
        _rec("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        _rec("  inside the frequency section", "COMMENT"),
        "   NOAZI    1.00    2.00    3.00",
        _rec("", "END OF FREQUENCY"),
        _rec("", "END OF ANTENNA"),
        _rec("between blocks", "COMMENT"),
        _rec("", "START OF ANTENNA"),
        _rec("AATEST              SER", "TYPE / SERIAL NO"),
        _rec("   G01", "START OF FREQUENCY"),
        _rec("      1.00      2.00      3.00", "NORTH / EAST / UP"),
        _rec("   G01", "END OF FREQUENCY"),
        _rec("", "END OF ANTENNA"),
        _rec("at the end", "COMMENT"),
    ]
    antex = sidereon.load_antex("\n".join(source).encode("ascii"))
    assert antex.skipped_records == 0
    header = antex.header
    assert header.comments == ["header note"]
    assert header.end_of_header
    assert header.version is None
    assert header.pcv_type is None
    assert [(c.blocks_before, c.text) for c in antex.outer_comments] == [
        (0, "after the header"),
        (1, "between blocks"),
        (2, "at the end"),
    ]
    zz = antex.antenna("ZZTEST              SER")
    assert zz.leading_comments == ["before the type"]
    assert zz.comments == ["about ZZ", "  inside the frequency section"]
    assert zz.has_frequency_count
    aa = antex.antenna("AATEST              SER")
    assert aa.dazi_deg is None
    assert aa.zenith_grid is None
    assert aa.zenith_start_deg is None
    assert aa.calibrations == []
    assert not aa.has_frequency_count
    assert [block.id for block in antex.antenna_blocks] == [
        "ZZTEST              SER",
        "AATEST              SER",
    ]

    # The comment read inside the frequency section follows SINEX CODE, and
    # the frequency code sits in columns 4-6; nothing the source lacks is
    # written.
    expected = [
        _rec("header note", "COMMENT"),
        _rec("", "END OF HEADER"),
        _rec("after the header", "COMMENT"),
        _rec("", "START OF ANTENNA"),
        _rec("before the type", "COMMENT"),
        _rec("ZZTEST              SER", "TYPE / SERIAL NO"),
        _rec("     0.0", "DAZI"),
        _rec("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        _rec("     1", "# OF FREQUENCIES"),
        _rec("IGS20", "SINEX CODE"),
        _rec("about ZZ", "COMMENT"),
        _rec("  inside the frequency section", "COMMENT"),
        _rec("   G01", "START OF FREQUENCY"),
        _rec("      0.00      0.00      0.00", "NORTH / EAST / UP"),
        "   NOAZI    1.00    2.00    3.00",
        _rec("   G01", "END OF FREQUENCY"),
        _rec("", "END OF ANTENNA"),
        _rec("between blocks", "COMMENT"),
        _rec("", "START OF ANTENNA"),
        _rec("AATEST              SER", "TYPE / SERIAL NO"),
        _rec("   G01", "START OF FREQUENCY"),
        _rec("      1.00      2.00      3.00", "NORTH / EAST / UP"),
        _rec("   G01", "END OF FREQUENCY"),
        _rec("", "END OF ANTENNA"),
        _rec("at the end", "COMMENT"),
    ]
    assert antex.to_antex_string() == "\n".join(expected) + "\n"


def test_inconsistent_records_are_counted():
    records = _grid_records("     0.0", "     0.0  10.0   5.0")
    records.extend(
        _frequency_section(
            "G01", "      0.00      0.00      0.00", ["   NOAZI    1.00"]
        )
    )
    stray = ["stray text", *records]
    assert sidereon.load_antex(_one_antenna(stray).encode("ascii")).skipped_records == 1
    count = [*records, _rec("     2", "# OF FREQUENCIES")]
    assert sidereon.load_antex(_one_antenna(count).encode("ascii")).skipped_records == 1
