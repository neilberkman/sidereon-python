"""Value equality for the public RINEX clock DTOs."""

from pathlib import Path

import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _fixture(name):
    return Path(CORE_FIXTURES, "clk", "lossless", name).read_bytes().decode("utf-8")


def _v300(system="G"):
    prefix = "     3.00           C                   " + system
    return f"{prefix:<60}RINEX VERSION / TYPE\n"


def _header(payload, label):
    return f"{payload:<60}{label}\n"


def _clock(record, *, comment="same"):
    return (
        _v300()
        + _header("   GPS", "TIME SYSTEM ID")
        + _header(comment, "COMMENT")
        + _header("", "END OF HEADER")
        + record
    )


def _parse_bad_satellite(satellite):
    return _clock(
        f"AS {satellite}  2026 05 13 00 00  0.000000  1    1.000000000000E-04\n"
    )


def _strict_detail(text):
    with pytest.raises(sidereon.RinexClockParseError) as excinfo:
        sidereon.parse_rinex_clock(text)
    assert excinfo.value.detail is not None
    return excinfo.value.detail


def test_rinex_clock_value_equality_uses_public_values():
    # Epochs compare their civil fields, including the fractional second.
    epoch = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 30.0)
    same_epoch = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 30.0)
    next_epoch = sidereon.ClockEpoch(2026, 5, 13, 0, 0, 30.5)
    assert epoch == same_epoch
    assert epoch != next_epoch

    # Skips come from independent public reads; changing valid header input
    # shifts the source line and therefore changes the retained skip value.
    source = _fixture("rinex_clock300_table_a17.clk")
    parsed = sidereon.parse_rinex_clock(source)
    reparsed = sidereon.load_rinex_clock(
        Path(CORE_FIXTURES, "clk", "lossless", "rinex_clock300_table_a17.clk")
    )
    assert parsed.skipped_records[0] == reparsed.skipped_records[0]
    newline = "\r\n" if "\r\n" in source else "\n"
    lines = source.splitlines(keepends=True)
    end_index = next(
        index
        for index, line in enumerate(lines)
        if line[60:].rstrip("\r\n").strip() == "END OF HEADER"
    )
    lines.insert(end_index, f"{'equality proof':<60}COMMENT{newline}")
    shifted = sidereon.parse_rinex_clock("".join(lines))
    assert parsed.skipped_records[0].line + 1 == shifted.skipped_records[0].line
    assert parsed.skipped_records[0] != shifted.skipped_records[0]

    # Lossy diagnostics and their typed errors compare the retained source
    # line and error payload, not just the displayed kind.
    bad_x01 = _parse_bad_satellite("X01")
    bad_x02 = _parse_bad_satellite("X02")
    diag_a = sidereon.parse_rinex_clock_lossy(bad_x01).diagnostics[0]
    diag_b = sidereon.parse_rinex_clock_lossy(bad_x01).diagnostics[0]
    diag_changed = sidereon.parse_rinex_clock_lossy(bad_x02).diagnostics[0]
    assert diag_a == diag_b
    assert diag_a.error == _strict_detail(bad_x01)
    assert diag_a != diag_changed
    assert diag_a.error != _strict_detail(bad_x02)

    # Time-system notices/statuses differ across real parse outcomes.
    defaulted_text = _fixture("rinex_clock304_table_a18.clk")
    defaulted = sidereon.parse_rinex_clock(defaulted_text)
    defaulted_again = sidereon.parse_rinex_clock(defaulted_text)
    declared = sidereon.parse_rinex_clock(_fixture("rinex_clock300_table_a17.clk"))
    assert defaulted.time_system_status == defaulted_again.time_system_status
    assert defaulted.time_system_status != declared.time_system_status
    default_notice = next(
        n for n in defaulted.notices if n.kind == "TimeSystemDefaulted"
    )
    assert default_notice == next(
        n for n in defaulted_again.notices if n.kind == "TimeSystemDefaulted"
    )
    irn_text = (
        "3.04                 C                    I".ljust(60)
        + "RINEX VERSION / TYPE\n"
        + _header("   IRN", "TIME SYSTEM ID")
        + _header("", "END OF HEADER")
        + "AS I01       2026 05 13 00 00  0.000000  1   -0.232835122007E-05\n"
    )
    irn_notice = sidereon.parse_rinex_clock(irn_text).notices[0]
    assert default_notice != irn_notice

    # Header records compare their full retained source text and typed reading.
    same_header = sidereon.parse_rinex_clock(
        _clock(
            "AS G01  2026 05 13 00 00  0.000000  1    1.000000000000E-04\n",
            comment="same",
        )
    )
    changed_header = sidereon.parse_rinex_clock(
        _clock(
            "AS G01  2026 05 13 00 00  0.000000  1    1.000000000000E-04\n",
            comment="changed",
        )
    )
    comment_record = next(r for r in same_header.header_records if r.label == "COMMENT")
    same_comment_record = next(
        r
        for r in sidereon.parse_rinex_clock(
            _clock(
                "AS G01  2026 05 13 00 00  0.000000  1    1.000000000000E-04\n",
                comment="same",
            )
        ).header_records
        if r.label == "COMMENT"
    )
    changed_comment_record = next(
        r for r in changed_header.header_records if r.label == "COMMENT"
    )
    assert comment_record == same_comment_record
    assert comment_record != changed_comment_record

    # Writer policies and their emitted departures use core value equality.
    strict_policy = sidereon.ClockWritePolicy.strict()
    lenient_policy = sidereon.ClockWritePolicy.lenient()
    assert strict_policy == sidereon.ClockWritePolicy()
    assert strict_policy != lenient_policy
    assert lenient_policy == sidereon.ClockWritePolicy(
        nearest_microsecond_epochs=sidereon.ClockWriteLeniency.ALLOW
    )

    def departure(name):
        instant = sidereon.ClockInstant.from_civil(
            sidereon.TimeScale.GPST, 2026, 5, 13, 0, 0, 30.0000001
        )
        product = sidereon.RinexClock.from_instant_series_rows(
            sidereon.TimeScale.GPST, [(name, [(instant, 1.0e-4)])]
        )
        return product.to_rinex_string_with_policy(lenient_policy).departures[0]

    departure_a = departure("G05")
    assert departure_a == departure("G05")
    assert departure_a != departure("G06")

    # Records and reading views compare across independent parses; a changed
    # bias and a typed edited record produce different public values.
    record_text = "AS G01  2026 05 13 00 00  0.000000  1    1.000000000000E-04\n"
    record_a = sidereon.parse_rinex_clock(_clock(record_text)).records[0]
    record_same = sidereon.parse_rinex_clock(_clock(record_text)).records[0]
    record_changed = sidereon.parse_rinex_clock(
        _clock(record_text.replace("1.000000000000E-04", "2.000000000000E-04"))
    ).records[0]
    assert record_a == record_same
    assert record_a != record_changed
    edited = sidereon.ClockRecord(
        sidereon.ClockRecordType.AS,
        "G01",
        epoch,
        [1.0e-4],
    )
    assert record_a.reading == record_same.reading
    assert record_a.reading != edited.reading

    # A valid EMR record carries a surplus sigma; changing that source number
    # yields a structurally different ClockSurplusValue.
    surplus_text = _fixture("EMR0OPSRAP_20262600000_first_epoch_excerpt.clk")
    surplus_other_text = surplus_text.replace(
        "5.556437046250E-12", "5.556437046251E-12", 1
    )
    assert surplus_other_text != surplus_text
    surplus_a = next(
        r for r in sidereon.parse_rinex_clock(surplus_text).records if r.surplus_values
    ).surplus_values[0]
    surplus_same = next(
        r for r in sidereon.parse_rinex_clock(surplus_text).records if r.surplus_values
    ).surplus_values[0]
    surplus_changed = next(
        r
        for r in sidereon.parse_rinex_clock(surplus_other_text).records
        if r.surplus_values
    ).surplus_values[0]
    assert surplus_a == surplus_same
    assert surplus_a != surplus_changed
