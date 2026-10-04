"""Exact public assertions for the ANTEX PCV sample variants."""

import numpy as np
import sidereon


def _record(content, label):
    return f"{content:<60}{label}"


def test_public_parser_retains_noazi_and_azimuth_data_and_rms_samples():
    source = [
        _record("     1.4            M", "ANTEX VERSION / SYST"),
        _record("A", "PCV TYPE / REFANT"),
        _record("", "END OF HEADER"),
        _record("", "START OF ANTENNA"),
        _record("TESTANT             SERIAL-1", "TYPE / SERIAL NO"),
        _record("     90.0", "DAZI"),
        _record("     0.0  10.0   5.0", "ZEN1 / ZEN2 / DZEN"),
        _record("     1", "# OF FREQUENCIES"),
        _record("G01", "START OF FREQUENCY"),
        _record("      1.00      2.00      3.00", "NORTH / EAST / UP"),
        "   NOAZI    1.00    2.00    3.00",
        "    90.0    4.00    5.00    6.00",
        _record("G01", "END OF FREQUENCY"),
        _record("G01", "START OF FREQ RMS"),
        _record("      0.10      0.20      0.30", "NORTH / EAST / UP"),
        "   NOAZI    0.01    0.02    0.03",
        "    90.0    0.04    0.05    0.06",
        _record("G01", "END OF FREQ RMS"),
        _record("", "END OF ANTENNA"),
    ]
    parsed = sidereon.load_antex(("\n".join(source) + "\n").encode("ascii"))
    antenna = parsed.antenna("TESTANT             SERIAL-1")
    assert antenna is not None
    assert antenna.kind == sidereon.AntennaKind.RECEIVER
    assert antenna.antenna_type == "TESTANT"
    assert antenna.serial == "SERIAL-1"
    assert antenna.dazi_deg == 90.0
    assert (
        antenna.zenith_grid.start_deg,
        antenna.zenith_grid.end_deg,
        antenna.zenith_grid.step_deg,
    ) == (0.0, 10.0, 5.0)
    assert antenna.has_frequency_count
    assert antenna.sinex_code is None
    assert antenna.valid_from is None
    assert antenna.valid_until is None
    assert antenna.calibrations == []
    assert antenna.leading_comments == []
    assert antenna.comments == []
    assert antenna.frequencies == ["G01"]

    frequency = antenna.frequency_sections[0]
    assert frequency.frequency == "G01"
    np.testing.assert_array_equal(frequency.pco_m, [0.001, 0.002, 0.003])
    assert [
        (sample.grid, sample.azimuth_deg, sample.zenith_deg, sample.value_m)
        for sample in frequency.pcv_samples
    ] == [
        (sidereon.AntexPcvGrid.NO_AZIMUTH, None, 0.0, 0.001),
        (sidereon.AntexPcvGrid.NO_AZIMUTH, None, 5.0, 0.002),
        (sidereon.AntexPcvGrid.NO_AZIMUTH, None, 10.0, 0.003),
        (sidereon.AntexPcvGrid.AZIMUTH, 90.0, 0.0, 0.004),
        (sidereon.AntexPcvGrid.AZIMUTH, 90.0, 5.0, 0.005),
        (sidereon.AntexPcvGrid.AZIMUTH, 90.0, 10.0, 0.006),
    ]
    assert frequency.rms is not None
    np.testing.assert_array_equal(frequency.rms.pco_m, [0.0001, 0.0002, 0.0003])
    assert [
        (sample.grid, sample.azimuth_deg, sample.zenith_deg, sample.value_m)
        for sample in frequency.rms.pcv_samples
    ] == [
        (sidereon.AntexPcvGrid.NO_AZIMUTH, None, 0.0, 0.00001),
        (sidereon.AntexPcvGrid.NO_AZIMUTH, None, 5.0, 0.00002),
        (sidereon.AntexPcvGrid.NO_AZIMUTH, None, 10.0, 0.00003),
        (sidereon.AntexPcvGrid.AZIMUTH, 90.0, 0.0, 0.00004),
        (sidereon.AntexPcvGrid.AZIMUTH, 90.0, 5.0, 0.00005),
        (sidereon.AntexPcvGrid.AZIMUTH, 90.0, 10.0, 0.00006),
    ]

    encoded = parsed.to_antex_string()
    reparsed = sidereon.load_antex(encoded.encode("ascii"))
    assert reparsed == parsed
