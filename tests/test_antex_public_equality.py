"""Direct equality coverage for public ANTEX values and parse details."""

import sidereon


def _record(content, label):
    assert len(content) <= 60
    return f"{content:<60}{label}"


def _antex(
    *,
    version="1.4",
    system="M",
    header_comment="header note",
    pcv_type="R",
    reference_type="AOAD/M_T",
    reference_serial="",
    outer_comment="outside note",
    grid="     0.0  10.0   5.0",
    calibration_date="01-JAN-20",
    pco="      1.00      2.00      3.00",
    rms_pco="      0.10      0.20      0.30",
    sample="   NOAZI    1.00    2.00    3.00",
):
    """Return a valid, small ANTEX document with all nested value types."""
    pcv_record = f"{pcv_type:<1}{'':19}{reference_type:<20}{reference_serial:<20}"
    calibration = f"{'FIELD':<20}{'IGS':<20}{2:6d}{'':4}{calibration_date:<10}"
    lines = [
        _record(f"     {version:<3}            {system}", "ANTEX VERSION / SYST"),
        _record(pcv_record, "PCV TYPE / REFANT"),
        _record(header_comment, "COMMENT"),
        _record("", "END OF HEADER"),
        _record(outer_comment, "COMMENT"),
        _record("", "START OF ANTENNA"),
        _record(f"{'TESTANT':<20}{'SERIAL-1':<20}", "TYPE / SERIAL NO"),
        _record(calibration, "METH / BY / # / DATE"),
        _record("     1", "# OF FREQUENCIES"),
        _record(grid, "ZEN1 / ZEN2 / DZEN"),
        _record("   G01", "START OF FREQUENCY"),
        _record(pco, "NORTH / EAST / UP"),
        sample,
        _record("   G01", "END OF FREQUENCY"),
        _record("   G01", "START OF FREQ RMS"),
        _record(rms_pco, "NORTH / EAST / UP"),
        _record("   NOAZI    0.01    0.02    0.03", ""),
        _record("   G01", "END OF FREQ RMS"),
        _record("", "END OF ANTENNA"),
        _record("after block", "COMMENT"),
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def _invalid_pcv_type(value):
    text = "\n".join(
        [
            _record("     1.4            M", "ANTEX VERSION / SYST"),
            _record(value, "PCV TYPE / REFANT"),
            _record("", "END OF HEADER"),
        ]
    )
    try:
        sidereon.load_antex(text.encode("ascii"))
    except sidereon.AntexParseError as error:
        assert error.detail is not None
        return error.detail
    raise AssertionError("invalid PCV type unexpectedly parsed")


def _assert_eq_and_ne(equal_left, equal_again, different):
    # These wrappers define __eq__ but rely on Python's normal __ne__ fallback.
    assert equal_left == equal_again
    assert not (equal_left != equal_again)
    assert equal_left != different
    assert not (equal_left == different)


def test_public_antex_value_and_error_detail_equality_protocol():
    source = _antex()
    parsed = sidereon.load_antex(source)
    repeated = sidereon.load_antex(source)

    header_changed = sidereon.load_antex(_antex(header_comment="changed header"))
    version_changed = sidereon.load_antex(_antex(system="G"))
    pcv_type_changed = sidereon.load_antex(
        _antex(reference_type="OTHER ANT", reference_serial="SERIAL-2")
    )
    outer_comment_changed = sidereon.load_antex(_antex(outer_comment="changed outer"))
    grid_changed = sidereon.load_antex(_antex(grid="     0.0  12.0   6.0"))
    calibration_changed = sidereon.load_antex(_antex(calibration_date="02-JAN-20"))
    frequency_changed = sidereon.load_antex(
        _antex(pco="      1.50      2.00      3.00")
    )
    rms_changed = sidereon.load_antex(_antex(rms_pco="      0.11      0.20      0.30"))
    sample_changed = sidereon.load_antex(
        _antex(sample="   NOAZI    1.00    2.50    3.00")
    )

    _assert_eq_and_ne(parsed, repeated, header_changed)

    antenna = parsed.antenna("TESTANT             SERIAL-1")
    repeated_antenna = repeated.antenna("TESTANT             SERIAL-1")
    changed_antenna = grid_changed.antenna("TESTANT             SERIAL-1")
    assert antenna is not None and repeated_antenna is not None
    assert changed_antenna is not None
    _assert_eq_and_ne(antenna, repeated_antenna, changed_antenna)

    _assert_eq_and_ne(parsed.header, repeated.header, header_changed.header)
    _assert_eq_and_ne(
        parsed.header.version,
        repeated.header.version,
        version_changed.header.version,
    )
    _assert_eq_and_ne(
        parsed.header.pcv_type,
        repeated.header.pcv_type,
        pcv_type_changed.header.pcv_type,
    )

    _assert_eq_and_ne(
        parsed.outer_comments[0],
        repeated.outer_comments[0],
        outer_comment_changed.outer_comments[0],
    )
    assert len(parsed.outer_comments) == 2

    assert antenna.zenith_grid is not None
    assert repeated_antenna.zenith_grid is not None
    assert changed_antenna.zenith_grid is not None
    _assert_eq_and_ne(
        antenna.zenith_grid,
        repeated_antenna.zenith_grid,
        changed_antenna.zenith_grid,
    )
    _assert_eq_and_ne(
        antenna.calibrations[0],
        repeated_antenna.calibrations[0],
        calibration_changed.antenna("TESTANT             SERIAL-1").calibrations[0],
    )

    frequency = antenna.frequency_sections[0]
    repeated_frequency = repeated_antenna.frequency_sections[0]
    _assert_eq_and_ne(
        frequency,
        repeated_frequency,
        frequency_changed.antenna("TESTANT             SERIAL-1").frequency_sections[0],
    )
    assert frequency.rms is not None and repeated_frequency.rms is not None
    changed_rms = (
        rms_changed.antenna("TESTANT             SERIAL-1").frequency_sections[0].rms
    )
    assert changed_rms is not None
    _assert_eq_and_ne(frequency.rms, repeated_frequency.rms, changed_rms)
    _assert_eq_and_ne(
        frequency.pcv_samples[1],
        repeated_frequency.pcv_samples[1],
        sample_changed.antenna("TESTANT             SERIAL-1")
        .frequency_sections[0]
        .pcv_samples[1],
    )

    _assert_eq_and_ne(
        _invalid_pcv_type("X"),
        _invalid_pcv_type("X"),
        _invalid_pcv_type("Y"),
    )
