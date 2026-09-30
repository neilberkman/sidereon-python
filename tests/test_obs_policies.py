"""RINEX OBS policy surfaces: corrections, event timing, and fallible writing.

Every fixture built here is RINEX text the core parser accepts, laid out to the
columns in RINEX 3.05 Table A2, or Table A1 for the one version 2.11 fixture,
and every assertion names an exact value the binding exposes. The subjects are
the parts of the observation API where a wrapper can quietly lose something: a
correction the header does not state, a header record an event changes part-way
through a file, an epoch that carries slips or event text rather than
measurements, and a writer refusal. The repair round trip reads one committed
station file, since CRINEX is checked against real text rather than a hand-built
header.
"""

import math
import os
import re

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES

OBS_FIXTURES = os.path.join(FIXTURES, "obs")
ESBC_RNX = "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"

OBS_FIELD_WIDTH = 16


def header_line(body, label):
    """A header record: 60 columns of content, then its label."""
    return f"{body:<60}{label}"


def obs_field(value=None, lli=None, ssi=None):
    """One F14.3,I1,I1 observation field; a blank value blanks the whole field."""
    if value is None:
        return " " * OBS_FIELD_WIDTH
    return f"{value:14.3f}{'' if lli is None else lli:1}{'' if ssi is None else ssi:1}"


def obs_record(satellite, fields):
    return satellite + "".join(fields)


def epoch_line(minute, second, flag, count, hour=0):
    return f"> 2020 01 01 {hour:02d} {minute:02d}{second:11.7f}{flag:3d}{count:3d}"


def untimed_event(flag, records):
    """A flag>1 event that leaves its epoch fields blank, plus its records."""
    return "\n".join([f">{'':30}{flag}{len(records):3d}", *records])


def obs_text(version, code_headers, body_lines, extra_headers=()):
    version_line = f"{version:9.2f}           OBSERVATION DATA    M (MIXED)"
    lines = [header_line(version_line, "RINEX VERSION / TYPE")]
    lines.extend(code_headers)
    lines.extend(extra_headers)
    lines.append(header_line("", "END OF HEADER"))
    lines.extend(body_lines)
    return "\n".join(lines) + "\n"


def time_of_first_obs(year, month, day, hour, minute, second, scale="GPS"):
    """A `TIME OF FIRST OBS` record: `5I6,F13.7` then the scale at 49-51."""
    body = f"{year:6d}{month:6d}{day:6d}{hour:6d}{minute:6d}{second:13.7f}{scale:>8}"
    return header_line(body, "TIME OF FIRST OBS")


def read_fixture(name):
    with open(os.path.join(OBS_FIXTURES, name), encoding="utf-8") as handle:
        return handle.read()


def synthetic_scenario():
    """The scenario `tests/test_018_domain_exposure.py` simulates: two epochs at
    a 30 s cadence, five synthetic GPS satellites on one signal, every error
    term disabled, so the observables are the geometry and nothing else.
    """
    start_j2000_s = sidereon.j2000_seconds(2026, 1, 1, 0, 0, 0.0)
    u60 = math.pi / 3.0
    disabled_clock = {
        "enabled": False,
        "bias_s": 0.0,
        "drift_s_s": 0.0,
        "power_law_coefficients": [0.0] * 5,
    }

    def orbit(prn, raan_rad, inclination_rad, mean_anomaly_rad):
        return {
            "satellite_id": {"system": "Gps", "prn": prn},
            "semi_major_axis_m": 26_560_000.0,
            "eccentricity": 0.0,
            "inclination_rad": inclination_rad,
            "raan_rad": raan_rad,
            "arg_perigee_rad": 0.0,
            "mean_anomaly_rad": mean_anomaly_rad,
            "epoch_j2000_s": start_j2000_s,
            "clock_bias_s": 0.0,
            "clock_drift_s_s": 0.0,
        }

    return {
        "schema_version": sidereon.scenario_schema_version(),
        "seed": 0x515C_1E7E_0B5E_A11D,
        "epochs": {
            "start_j2000_s": start_j2000_s,
            "count": 2,
            "cadence_s": 30.0,
        },
        "receiver": {
            "kind": "static_geodetic",
            "position": {"lat_rad": 0.0, "lon_rad": 0.0, "height_m": 0.0},
        },
        "constellation": {
            "kind": "synthetic_keplerian",
            "satellites": [
                orbit(1, 0.0, 0.0, 0.0),
                orbit(2, 0.0, 0.0, u60),
                orbit(3, 0.0, 0.0, -u60),
                orbit(4, 0.0, math.pi / 2.0, u60),
                orbit(5, 0.0, math.pi / 2.0, -u60),
            ],
        },
        "signals": [
            {
                "system": "Gps",
                "code_observable": "C1C",
                "phase_observable": "L1C",
                "doppler_observable": "D1C",
                "carrier_hz": 1_575_420_000.0,
                "carrier_phase_bias_cycles": 0.0,
            }
        ],
        "error_budget": {
            "receiver_clock": disabled_clock,
            "satellite_clock": disabled_clock,
            "ionosphere": {"kind": "off"},
            "troposphere": {"kind": "off"},
            "thermal_noise": {
                "enabled": False,
                "pseudorange_sigma_m": 0.0,
                "carrier_phase_sigma_m": 0.0,
                "doppler_sigma_hz": 0.0,
            },
            "multipath": {
                "enabled": False,
                "amplitude_m": 0.0,
                "reflector_height_m": 0.0,
                "phase_rad": 0.0,
            },
            "elevation_mask_deg": -5.0,
        },
    }


def row_index(series, satellite, code):
    return next(
        index
        for index, (sat, candidate) in enumerate(zip(series.satellites, series.codes))
        if sat == satellite and candidate == code
    )


# --- Phase-shift corrections ------------------------------------------------


def _three_correction_states_text():
    """One header stating a correction, contradicting itself, and declaring
    an alignment unknown, so all three states appear in one epoch's rows.

    `G L1C  0.25000` states 0.25 cycles. The two `G L2W` records, one with a
    blank correction and one with 0.5, are a contradiction RINEX gives no order
    to resolve. `E` alone names only its constellation, which RINEX 3.05
    section 5.2.12 gives as "the phase alignment is unknown".
    """
    return obs_text(
        3.05,
        [
            header_line("G    2 L1C L2W", "SYS / # / OBS TYPES"),
            header_line("E    1 L1C", "SYS / # / OBS TYPES"),
        ],
        [
            epoch_line(0, 0.0, 0, 2),
            obs_record("G01", [obs_field(100000.125, 0, 7), obs_field()]),
            obs_record("E01", [obs_field(200000.250, 0, 6)]),
        ],
        extra_headers=[
            header_line("G L1C  0.25000", "SYS / PHASE SHIFT"),
            header_line("G L2W", "SYS / PHASE SHIFT"),
            header_line("G L2W  0.50000", "SYS / PHASE SHIFT"),
            header_line("E", "SYS / PHASE SHIFT"),
        ],
    )


def test_stated_unknown_and_ambiguous_corrections_stay_apart():
    obs = sidereon.parse_rinex_obs(_three_correction_states_text())
    phase = obs.carrier_phase_rows(0)

    # Three carrier rows: two GPS codes and one Galileo code. None is dropped
    # for wanting a correction, and every parallel array has one entry per row.
    assert len(phase) == 3
    for array in (
        phase.value_cycles,
        phase.frequency_hz,
        phase.wavelength_m,
        phase.value_m,
        phase.phase_shift_cycles,
        phase.phase_shift_valid,
        phase.lli,
        phase.ssi,
    ):
        assert len(array) == 3
    assert len(phase.satellites) == 3
    assert len(phase.codes) == 3
    assert len(phase.phase_shift_results) == 3
    assert len(phase.phase_shift_statuses) == 3

    stated = row_index(phase, "G01", "L1C")
    ambiguous = row_index(phase, "G01", "L2W")
    unknown = row_index(phase, "E01", "L1C")

    assert phase.phase_shift_statuses == [
        phase.phase_shift_results[index].status for index in range(3)
    ]

    held = phase.phase_shift_results[stated]
    assert held.status == "available"
    assert held.is_available is True
    assert held.cycles == 0.25
    assert held.value == 0.25
    assert held.corrections == []
    assert phase.phase_shift_cycles[stated] == 0.25
    assert bool(phase.phase_shift_valid[stated]) is True

    conflict = phase.phase_shift_results[ambiguous]
    assert conflict.status == "ambiguous"
    assert conflict.is_ambiguous is True
    assert conflict.is_available is False
    assert conflict.cycles is None
    # The blank record is kept as None beside the stated 0.5, in header order.
    assert conflict.corrections == [None, 0.5]
    assert math.isnan(phase.phase_shift_cycles[ambiguous])
    assert bool(phase.phase_shift_valid[ambiguous]) is False

    absent = phase.phase_shift_results[unknown]
    assert absent.status == "unknown"
    assert absent.is_unknown is True
    assert absent.cycles is None
    assert absent.corrections == []
    assert math.isnan(phase.phase_shift_cycles[unknown])
    assert bool(phase.phase_shift_valid[unknown]) is False

    # The blank L2W field keeps its row and reads as absent, not as zero.
    assert math.isnan(phase.value_cycles[ambiguous])
    assert math.isnan(phase.value_m[ambiguous])
    assert phase.value_cycles[stated] == 100000.125
    assert phase.frequency_hz[stated] == 1575420000.0

    # The contradiction is counted, not swallowed, and both records are kept.
    assert obs.skipped_records == 1
    l2w = [shift for shift in obs.header.phase_shifts if shift.code == "L2W"]
    assert [shift.correction_cycles for shift in l2w] == [None, 0.5]


def test_a_stated_zero_correction_is_not_an_unknown_one():
    """A blank correction on a record that names a code says none was applied,
    which reads as 0.0; only a record with no code says the alignment is
    unknown. The two must not collapse into one another.
    """
    text = obs_text(
        3.05,
        [header_line("G    1 L1C", "SYS / # / OBS TYPES")],
        [
            epoch_line(0, 0.0, 0, 1),
            obs_record("G01", [obs_field(100000.125, 0, 7)]),
        ],
        extra_headers=[header_line("G L1C", "SYS / PHASE SHIFT")],
    )
    obs = sidereon.parse_rinex_obs(text)

    record = obs.header.phase_shifts[0]
    assert record.code == "L1C"
    assert record.correction_cycles is None
    assert record.covers_every_satellite() is True
    assert record.satellite_count() == 0

    result = obs.carrier_phase_rows(0).phase_shift_results[0]
    assert result.status == "available"
    assert result.cycles == 0.0
    assert result.is_unknown is False


def test_a_file_with_no_phase_shift_record_states_no_correction():
    text = obs_text(
        3.05,
        [header_line("G    1 L1C", "SYS / # / OBS TYPES")],
        [
            epoch_line(0, 0.0, 0, 1),
            obs_record("G01", [obs_field(100000.125, 0, 7)]),
        ],
    )
    obs = sidereon.parse_rinex_obs(text)
    assert obs.header.phase_shifts == []

    result = obs.carrier_phase_rows(0).phase_shift_results[0]
    assert result.status == "available"
    assert result.cycles == 0.0


def test_phase_shift_record_keeps_satellites_it_cannot_represent():
    """A record may name a satellite designator the engine's id does not hold.
    It is kept as written so the record writes back whole, counted as a skip,
    and it restricts the record to satellites that are not this one.

    The satellite-token range is 01..99 for every constellation, so an
    extended slot such as R28 is held; a designator numbered 00 names no
    satellite at all.
    """
    text = obs_text(
        3.05,
        [header_line("R    1 L1C", "SYS / # / OBS TYPES")],
        [],
        extra_headers=[
            header_line("R L1C  0.25000  03 R01 R28 R00", "SYS / PHASE SHIFT"),
        ],
    )
    obs = sidereon.parse_rinex_obs(text)

    record = obs.header.phase_shifts[0]
    assert record.system == sidereon.GnssSystem.GLONASS
    assert record.code == "L1C"
    assert record.correction_cycles == 0.25
    assert record.satellites == ["R01", "R28"]
    assert record.unrepresentable_satellites == ["R00"]
    assert record.covers_every_satellite() is False
    assert record.satellite_count() == 3
    assert obs.skipped_records == 1


# --- GLONASS code-phase biases ----------------------------------------------


def _glonass_bias_obs(bias_records):
    return sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("R    1 C1C", "SYS / # / OBS TYPES")],
            [],
            extra_headers=bias_records,
        )
    )


def test_glonass_bias_separates_absent_blank_stated_and_contradicted():
    # No record at all: the header gives the signal no bias.
    absent = _glonass_bias_obs([])
    assert absent.header.glonass_cod_phs_bis is None
    none_result = absent.header.glonass_code_phase_bias("C1C")
    assert none_result.status == "none"
    assert none_result.is_none is True
    assert none_result.value is None
    assert none_result.corrections == []

    # A blank record declares the alignment unknown (RINEX 3.05 5.2.16).
    blank = _glonass_bias_obs([header_line("", "GLONASS COD/PHS/BIS")])
    assert blank.header.glonass_cod_phs_bis == []
    unknown = blank.header.glonass_code_phase_bias("C1C")
    assert unknown.status == "unknown"
    assert unknown.is_unknown is True
    assert unknown.value is None

    # A stated bias.
    stated = _glonass_bias_obs(
        [header_line(f" C1C {0.25:8.3f}", "GLONASS COD/PHS/BIS")]
    )
    assert stated.header.glonass_cod_phs_bis == [("C1C", 0.25)]
    available = stated.header.glonass_code_phase_bias("C1C")
    assert available.status == "available"
    assert available.is_available is True
    assert available.value == 0.25
    assert available.bias_m == 0.25

    # A blank bias for the signal beside a stated one: kept in header order,
    # the blank one as None, and read as ambiguous rather than picked between.
    contradicted = _glonass_bias_obs(
        [
            header_line(f" C1C {'':8}", "GLONASS COD/PHS/BIS"),
            header_line(f" C1C {0.0:8.3f}", "GLONASS COD/PHS/BIS"),
        ]
    )
    assert contradicted.header.glonass_cod_phs_bis == [("C1C", None), ("C1C", 0.0)]
    ambiguous = contradicted.header.glonass_code_phase_bias("C1C")
    assert ambiguous.status == "ambiguous"
    assert ambiguous.is_ambiguous is True
    assert ambiguous.value is None
    assert ambiguous.corrections == [None, 0.0]
    assert contradicted.skipped_records == 1

    # A code the records do not name has no bias, whatever the others say.
    other = contradicted.header.glonass_code_phase_bias("C2C")
    assert other.status == "none"


def test_rinex4_ignores_the_deprecated_bias_and_phase_shift_records():
    """RINEX 4.00, 4.01 and 4.02 Table A2 say of both records that "the lines
    should be ignored by RINEX decoders and encoders". The records are still
    parsed and kept, so the file writes back whole, but they state nothing.
    """
    text = obs_text(
        4.02,
        [header_line("G    1 L1C", "SYS / # / OBS TYPES")],
        [
            epoch_line(0, 0.0, 0, 1),
            obs_record("G01", [obs_field(100000.125, 0, 7)]),
        ],
        extra_headers=[
            header_line("G L1C  0.25000", "SYS / PHASE SHIFT"),
            header_line(f" C1C {0.25:8.3f}", "GLONASS COD/PHS/BIS"),
        ],
    )
    obs = sidereon.parse_rinex_obs(text)

    # Kept as written.
    assert obs.header.phase_shifts[0].correction_cycles == 0.25
    assert obs.header.glonass_cod_phs_bis == [("C1C", 0.25)]
    # Applied as nothing.
    result = obs.carrier_phase_rows(0).phase_shift_results[0]
    assert result.status == "available"
    assert result.cycles == 0.0
    assert obs.header.glonass_code_phase_bias("C1C").status == "none"


# --- Event timing -----------------------------------------------------------


def _event_phase_shift_text():
    """A file whose flag 4 event replaces the GPS L1C phase shift part-way
    through, so the epochs before it keep 0.25 and the ones after take 0.5.
    """
    return obs_text(
        3.05,
        [header_line("G    1 L1C", "SYS / # / OBS TYPES")],
        [
            epoch_line(0, 0.0, 0, 1),
            obs_record("G01", [obs_field(100000.125, 0, 7)]),
            epoch_line(1, 0.0, 4, 1),
            header_line("G L1C  0.50000", "SYS / PHASE SHIFT"),
            epoch_line(2, 0.0, 0, 1),
            obs_record("G01", [obs_field(100100.125, 0, 7)]),
        ],
        extra_headers=[header_line("G L1C  0.25000", "SYS / PHASE SHIFT")],
    )


def test_an_event_changes_the_correction_from_its_own_epoch_onward():
    obs = sidereon.parse_rinex_obs(_event_phase_shift_text())
    assert obs.epoch_count == 3

    # The file header keeps what it declared; the event does not rewrite it.
    assert [shift.correction_cycles for shift in obs.header.phase_shifts] == [0.25]

    before = obs.carrier_phase_rows(0).phase_shift_results[0]
    after = obs.carrier_phase_rows(2).phase_shift_results[0]
    assert before.cycles == 0.25
    assert after.cycles == 0.5

    # header_at agrees with the rows it produced.
    assert [shift.correction_cycles for shift in obs.header_at(0).phase_shifts] == [
        0.25
    ]
    assert [shift.correction_cycles for shift in obs.header_at(2).phase_shifts] == [0.5]
    # The event's own epoch already carries its records.
    assert [shift.correction_cycles for shift in obs.header_at(1).phase_shifts] == [0.5]

    timeline = obs.header_timeline()
    assert timeline.segment_count == 2
    assert len(timeline) == 2
    assert timeline.segment_index(0) == 0
    assert timeline.segment_index(1) == 1
    assert timeline.segment_index(2) == 1
    segments = timeline.segments()
    assert [first for first, _ in segments] == [0, 1]
    assert [shift.correction_cycles for shift in segments[0][1].phase_shifts] == [0.25]
    assert [shift.correction_cycles for shift in segments[1][1].phase_shifts] == [0.5]
    assert [shift.correction_cycles for shift in timeline.at(0).phase_shifts] == [0.25]
    assert [shift.correction_cycles for shift in timeline.at(2).phase_shifts] == [0.5]

    with pytest.raises(sidereon.RinexObsParseError):
        obs.header_at(3)


def test_a_file_with_no_event_has_one_timeline_segment():
    obs = sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("G    1 L1C", "SYS / # / OBS TYPES")],
            [
                epoch_line(0, 0.0, 0, 1),
                obs_record("G01", [obs_field(100000.125, 0, 7)]),
            ],
        )
    )
    timeline = obs.header_timeline()
    assert timeline.segment_count == 1
    assert timeline.segment_index(0) == 0


def _list_change_text():
    """A flag 4 event declaring a longer, reordered GPS list, then a shorter
    one, leaving GLONASS's alone. Ported from the core's own fixture for this.
    """
    return obs_text(
        3.05,
        [
            header_line("G    2 C1C L1C", "SYS / # / OBS TYPES"),
            header_line("R    1 C1C", "SYS / # / OBS TYPES"),
        ],
        [
            epoch_line(0, 0.0, 0, 2),
            obs_record("G01", [obs_field(20000000.0), obs_field(100000.0)]),
            obs_record("R02", [obs_field(21000000.0)]),
            untimed_event(
                4, [header_line("G    3 L1C C1C S1C", "SYS / # / OBS TYPES")]
            ),
            epoch_line(0, 30.0, 0, 2),
            obs_record(
                "G01",
                [obs_field(100030.0), obs_field(20000030.0), obs_field(45.0)],
            ),
            obs_record("R02", [obs_field(21000030.0)]),
            untimed_event(4, [header_line("G    1 C1C", "SYS / # / OBS TYPES")]),
            epoch_line(1, 0.0, 0, 1),
            obs_record("G01", [obs_field(20000060.0)]),
        ],
    )


def test_the_union_list_reads_values_and_the_declared_list_is_per_epoch():
    obs = sidereon.parse_rinex_obs(_list_change_text())
    gps = sidereon.GnssSystem.GPS

    # The union holds every code any list declares, first-declared order. It is
    # what epoch value vectors are aligned to.
    assert obs.header.obs_codes(gps) == ["C1C", "L1C", "S1C"]
    assert obs.obs_codes(gps) == ["C1C", "L1C", "S1C"]
    # The file header itself declared only two of them.
    assert obs.header.declared_obs_codes(gps) == ["C1C", "L1C"]
    assert obs.header.obs_codes(sidereon.GnssSystem.GLONASS) == ["C1C"]

    # Each epoch's own declaration, from the header in effect there.
    assert obs.header_at(0).declared_obs_codes(gps) == ["C1C", "L1C"]
    assert obs.header_at(2).declared_obs_codes(gps) == ["L1C", "C1C", "S1C"]
    assert obs.header_at(4).declared_obs_codes(gps) == ["C1C"]

    # Values sit at their code's place in the union, blank where a list does
    # not declare the code. A reordered list does not move a value's meaning.
    def values(epoch_index):
        return [value.value for value in obs.epoch(epoch_index).sats["G01"]]

    assert values(0) == [20000000.0, 100000.0, None]
    assert values(2) == [20000030.0, 100030.0, 45.0]
    assert values(4) == [20000060.0, None, None]

    rows = obs.observation_values(2)
    assert rows.codes == ["C1C", "L1C", "S1C", "C1C"]
    assert rows.satellites == ["G01", "G01", "G01", "R02"]


# --- Event and cycle-slip epochs --------------------------------------------


def _mixed_record_kinds_text():
    """An observation epoch, a cycle slip epoch and an untimed flag 4 event,
    so every kind of epoch record appears in one product.
    """
    return obs_text(
        3.05,
        [header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
        [
            epoch_line(0, 0.0, 0, 1),
            obs_record(
                "G01", [obs_field(20000000.0, 0, 7), obs_field(100000.125, 0, 7)]
            ),
            epoch_line(0, 30.0, 6, 1),
            obs_record("G01", [obs_field(), obs_field(1.0)]),
            untimed_event(4, [header_line("A COMMENT ON THE SITE", "COMMENT")]),
            epoch_line(1, 0.0, 0, 1),
            obs_record(
                "G01", [obs_field(20000100.0, 1, 7), obs_field(100300.250, 1, 7)]
            ),
        ],
    )


def test_event_and_cycle_slip_epochs_carry_no_observation_rows():
    obs = sidereon.parse_rinex_obs(_mixed_record_kinds_text())
    assert obs.epoch_count == 4

    slips = obs.epoch(1)
    assert slips.flag == sidereon.CYCLE_SLIP_FLAG == 6
    assert slips.declared_record_count == 1
    assert slips.satellites == []
    assert slips.satellite_count == 0
    assert slips.sats == {}
    assert slips.special_records == []
    # The slips are held apart from measurements, in the observation layout,
    # with a blank field staying blank rather than becoming a zero slip.
    assert slips.cycle_slip_satellites == ["G01"]
    assert slips.cycle_slip_count == 1
    assert [value.value for value in slips.cycle_slips["G01"]] == [None, 1.0]
    assert [value.lli for value in slips.cycle_slips["G01"]] == [None, None]
    assert [value.ssi for value in slips.cycle_slips["G01"]] == [None, None]
    # A cycle slip epoch has a time; only an event may leave its fields blank.
    assert slips.epoch is not None
    assert slips.epoch.second == 30.0

    event = obs.epoch(2)
    assert event.flag == 4
    assert event.epoch is None
    assert event.declared_record_count == 1
    assert event.sats == {}
    assert event.cycle_slips == {}
    # The records are kept exactly as the file wrote them, label and all.
    assert event.special_records == [header_line("A COMMENT ON THE SITE", "COMMENT")]

    # Neither kind of record produces an observation row.
    for epoch_index in (1, 2):
        assert len(obs.observation_values(epoch_index)) == 0
        assert len(obs.carrier_phase_rows(epoch_index)) == 0
        assert len(obs.pseudoranges(epoch_index)) == 0
        assert obs.observation_values(epoch_index).satellites == []
        assert obs.carrier_phase_rows(epoch_index).phase_shift_results == []

    # The observation epochs on either side are untouched, and their indices
    # are the ones the event and slip records sit between.
    assert len(obs.observation_values(0)) == 2
    assert len(obs.observation_values(3)) == 2
    assert obs.epoch(0).epoch.minute == 0
    assert obs.epoch(3).epoch.minute == 1


def test_optional_epoch_line_fields_are_kept_or_absent():
    """A receiver clock offset and the RINEX 4.02 picosecond extension are
    optional; an epoch without them says None rather than 0.
    """
    without = sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("G    1 C1C", "SYS / # / OBS TYPES")],
            [
                epoch_line(0, 0.0, 0, 1),
                obs_record("G01", [obs_field(20000000.0, 0, 7)]),
            ],
        )
    )
    assert without.epoch(0).rcv_clock_offset_s is None
    assert without.epoch(0).epoch_picoseconds is None

    with_offset = sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("G    1 C1C", "SYS / # / OBS TYPES")],
            [
                # The epoch line runs to column 35; the F15.12 clock offset
                # starts at column 42.
                epoch_line(0, 0.0, 0, 1) + " " * 6 + f"{0.000123456789:15.12f}",
                obs_record("G01", [obs_field(20000000.0, 0, 7)]),
            ],
        )
    )
    assert with_offset.epoch(0).rcv_clock_offset_s == pytest.approx(
        0.000123456789, abs=1e-15
    )


def test_blank_observation_fields_keep_their_row_and_read_as_absent():
    obs = sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
            [
                epoch_line(0, 0.0, 0, 1),
                obs_record("G01", [obs_field(), obs_field(100000.125)]),
            ],
        )
    )
    values = obs.epoch(0).sats["G01"]
    assert [value.value for value in values] == [None, 100000.125]
    assert [value.lli for value in values] == [None, None]
    assert [value.ssi for value in values] == [None, None]

    rows = obs.observation_values(0)
    assert len(rows) == 2
    assert rows.codes == ["C1C", "L1C"]
    assert math.isnan(rows.values[0])
    assert rows.values[1] == 100000.125
    assert math.isnan(rows.lli[1])
    assert math.isnan(rows.ssi[1])
    assert rows.kinds == [
        sidereon.ObservationKind.PSEUDORANGE,
        sidereon.ObservationKind.CARRIER_PHASE,
    ]

    # A blank pseudorange is not a range of zero: the satellite is left out.
    assert len(obs.pseudoranges(0)) == 0


# --- LEAP SECONDS time system ------------------------------------------------


def _leap_obs(version, week_day_token):
    return sidereon.parse_rinex_obs(
        obs_text(
            version,
            [header_line("G    1 C1C", "SYS / # / OBS TYPES")],
            [],
            extra_headers=[header_line(week_day_token, "LEAP SECONDS")],
        )
    )


def _leap_body(current, delta=None, week=None, day=None, token=""):
    def field(value):
        return f"{value:6d}" if value is not None else " " * 6

    return f"{current:6d}{field(delta)}{field(week)}{field(day)}{token}"


def test_leap_seconds_time_system_is_kept_as_written():
    # Absent identifier: the field is optional and its absence is not `GPS`.
    absent = _leap_obs(3.05, _leap_body(18, week=2300, day=1))
    assert absent.header.leap_seconds.current == 18
    assert absent.header.leap_seconds.delta_future is None
    assert absent.header.leap_seconds.week == 2300
    assert absent.header.leap_seconds.day == 1
    assert absent.header.leap_seconds.time_system is None

    # An explicit GPS is a token the header carries, not an inferred default.
    explicit = _leap_obs(3.05, _leap_body(18, week=2300, day=1, token="GPS"))
    assert explicit.header.leap_seconds.time_system == "GPS"

    # BeiDou's token is kept in the spelling its version uses: BDS in 3.03 and
    # 3.04, BDT from 3.05. Neither is rewritten as the other.
    bds = _leap_obs(3.03, _leap_body(4, week=1000, day=6, token="BDS"))
    bdt = _leap_obs(3.05, _leap_body(4, week=1000, day=6, token="BDT"))
    assert bds.header.leap_seconds.time_system == "BDS"
    assert bdt.header.leap_seconds.time_system == "BDT"

    # A file with no LEAP SECONDS record says so.
    bare = sidereon.parse_rinex_obs(
        obs_text(3.05, [header_line("G    1 C1C", "SYS / # / OBS TYPES")], [])
    )
    assert bare.header.leap_seconds is None


def test_a_leap_seconds_token_the_version_forbids_is_refused_on_parse():
    # RINEX 3.05 renamed BDS to BDT, so BDS is not a 3.05 token.
    with pytest.raises(sidereon.RinexObsParseError):
        _leap_obs(3.05, _leap_body(4, week=1000, day=6, token="BDS"))
    # And an identifier that is no time system at all.
    with pytest.raises(sidereon.RinexObsParseError):
        _leap_obs(3.05, _leap_body(4, week=1000, day=6, token="XYZ"))


# --- Fallible writing and downgrade -----------------------------------------


def test_downgrade_refuses_a_target_that_is_not_a_version_two():
    obs = sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("G    1 C1C", "SYS / # / OBS TYPES")],
            [
                epoch_line(0, 0.0, 0, 1),
                obs_record("G01", [obs_field(20000000.0, 0, 7)]),
            ],
        )
    )
    with pytest.raises(sidereon.RinexObsWriteError) as raised:
        obs.downgrade_to_rinex2(3.05)

    detail = raised.value.detail
    assert detail is not None
    assert detail.kind == "NotVersionTwo"
    assert detail.version == 3.05
    assert detail.details() == {"version": 3.05}
    # The typed payload carries the value; the message is not the only record.
    assert "3.05" in detail.message
    # A write refusal is still a RINEX OBS error for a caller catching broadly.
    assert isinstance(raised.value, sidereon.RinexObsParseError)


def test_downgrade_refuses_an_observable_version_two_cannot_represent():
    """BeiDou C1X is on B1C. No version 2 name reads back as that carrier, so
    writing it under one would move the measurement to another signal. The
    refusal names the code rather than guessing an alias.
    """
    text = (
        "     3.05           OBSERVATION DATA    C"
        "                   RINEX VERSION / TYPE\n"
        "C    2 C1X L1X                                              "
        "SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
        "> 2020 06 24 00 00  0.0000000  0  1\n"
        "C01  22000000.000          10.000  \n"
    )
    obs = sidereon.parse_rinex_obs(text)
    assert obs.header.obs_codes(sidereon.GnssSystem.BEIDOU) == ["C1X", "L1X"]

    for version in (2.11, 2.12):
        with pytest.raises(sidereon.RinexObsWriteError) as raised:
            obs.downgrade_to_rinex2(version)
        detail = raised.value.detail
        assert detail.kind == "ObservableNotRepresentable"
        assert detail.system == sidereon.GnssSystem.BEIDOU
        assert detail.code == "C1X"
        assert detail.version == version
        assert detail.details() == {
            "system": sidereon.GnssSystem.BEIDOU,
            "code": "C1X",
            "version": version,
        }

    # The refusal leaves the source product as it was.
    assert obs.header.obs_codes(sidereon.GnssSystem.BEIDOU) == ["C1X", "L1X"]


def test_downgrade_refuses_a_leap_seconds_token_the_target_version_lacks():
    """BDT is a 3.05 token. Version 2 permits only GPS, so the downgrade is
    refused rather than dropping or rewriting the identifier.
    """
    obs = _leap_obs(3.05, _leap_body(4, week=1000, day=6, token="BDT"))
    assert obs.header.leap_seconds.time_system == "BDT"

    with pytest.raises(sidereon.RinexObsWriteError) as raised:
        obs.downgrade_to_rinex2(2.11)

    detail = raised.value.detail
    assert detail.kind == "LeapSecondsTimeSystemNotInVersion"
    assert detail.time_system == "BDT"
    assert detail.version == 2.11
    assert detail.details() == {"time_system": "BDT", "version": 2.11}

    # The check is version-aware, not a blanket refusal: BDT is a 3.05 token,
    # so writing the same product as 3.05 keeps it and reads it back.
    written = obs.to_rinex_string()
    assert sidereon.parse_rinex_obs(written).header.leap_seconds.time_system == "BDT"


def test_downgrade_reports_every_change_and_keeps_slips():
    """A 3.02 BeiDou B1I product becomes a version 2 one by renaming its codes
    to what the version 2 column reads back as. Each rename is reported, and
    the cycle slips survive the rewrite.
    """
    text = (
        "     3.02           OBSERVATION DATA    C"
        "                   RINEX VERSION / TYPE\n"
        "C    2 C1I L1I                                              "
        "SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
        "> 2020 06 24 00 00  0.0000000  0  1\n"
        "C01  22000000.000 7        10.00015\n"
        "> 2020 06 24 00 00 30.0000000  6  1\n"
        "C01                       100.000  \n"
    )
    obs = sidereon.parse_rinex_obs(text)
    beidou = sidereon.GnssSystem.BEIDOU

    downgraded, changes = obs.downgrade_to_rinex2(2.11)
    assert downgraded.header.version == 2.11
    assert downgraded.header.obs_codes(beidou) == ["C2I", "L2I"]

    renames = {
        (change.details()["from_code"], change.details()["to_code"])
        for change in changes
        if change.kind == "CodeRenamed"
    }
    assert renames == {("C1I", "C2I"), ("L1I", "L2I")}
    for change in changes:
        if change.kind == "CodeRenamed":
            assert change.system == beidou
            assert change.epoch_index is None

    # Observations keep their values and both indicators.
    values = downgraded.epoch(0).sats["C01"]
    assert [value.value for value in values] == [22000000.0, 10.0]
    assert [value.lli for value in values] == [None, 1]
    assert [value.ssi for value in values] == [7, 5]

    # The flag 6 records stay slips, with the blank field still blank.
    slips = downgraded.epoch(1)
    assert slips.flag == sidereon.CYCLE_SLIP_FLAG
    assert [value.value for value in slips.cycle_slips["C01"]] == [None, 100.0]
    assert slips.sats == {}

    # The source product is not changed by the downgrade.
    assert obs.header.version == 3.02
    assert obs.header.obs_codes(beidou) == ["C1I", "L1I"]

    # The free-function form gives the same result.
    also, also_changes = sidereon.downgrade_to_rinex2(obs, 2.11)
    assert also.header.obs_codes(beidou) == ["C2I", "L2I"]
    assert len(also_changes) == len(changes)


def test_downgrade_change_payloads_are_native_values_under_their_own_keys():
    """A scale-factored value is stored physical, so it can carry more decimals
    than a version 2 field holds. The downgrade removes the record and rounds
    the value, and reports both with the values themselves: a count, an epoch
    index, a RINEX designator, a code, and the two float values.
    """
    obs = sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
            [
                epoch_line(0, 0.0, 0, 1),
                obs_record(
                    "G01", [obs_field(20000000.0, 0, 7), obs_field(100000.125, 0, 7)]
                ),
            ],
            extra_headers=[header_line("G 1000   1 L1C", "SYS / SCALE FACTOR")],
        )
    )
    # The record declares one code, so the factor reaches that code and no
    # other: `L1C` is held physical, and `C1C` is the number its field states.
    assert obs.header.scale_factors[0].codes == ["L1C"]
    assert obs.epoch(0).sats["G01"][1].value == pytest.approx(100.000125)
    assert obs.epoch(0).sats["G01"][0].value == 20000000.0

    downgraded, changes = obs.downgrade_to_rinex2(2.11)
    by_kind = {change.kind: change for change in changes}
    assert "ScaleFactorsRemoved" in by_kind
    assert "ValueRounded" in by_kind

    removed = by_kind["ScaleFactorsRemoved"]
    assert removed.details() == {"count": 1}
    # A variant that names no epoch and no constellation answers None for both.
    assert removed.epoch_index is None
    assert removed.system is None

    rounded = by_kind["ValueRounded"]
    payload = rounded.details()
    assert list(payload) == [
        "epoch_index",
        "satellite",
        "code",
        "from_value",
        "to_value",
    ]
    assert payload["epoch_index"] == 0
    assert payload["satellite"] == "G01"
    assert payload["code"] == "L1C"
    assert payload["from_value"] == pytest.approx(100.000125)
    assert payload["to_value"] == 100.0
    # Native values, not text: a count is an int, a value is a float, and the
    # satellite is the RINEX designator rather than a debug rendering.
    assert isinstance(removed.details()["count"], int)
    assert isinstance(payload["from_value"], float)
    assert isinstance(payload["satellite"], str)
    assert rounded.epoch_index == 0
    assert rounded.system is None

    # The downgraded product holds the rounded value, and the scale factor
    # record is gone rather than carried into a version that cannot state it.
    assert downgraded.header.scale_factors == []
    assert downgraded.epoch(0).sats["G01"][1].value == 100.0
    assert obs.header.scale_factors[0].factor == 1000.0


def test_written_text_reads_back_as_the_same_product():
    text = _three_correction_states_text()
    obs = sidereon.parse_rinex_obs(text)
    written = obs.to_rinex_string()
    reparsed = sidereon.parse_rinex_obs(written)

    assert reparsed.epoch_count == obs.epoch_count
    assert reparsed.header.version == obs.header.version
    assert reparsed.header.obs_codes(sidereon.GnssSystem.GPS) == obs.header.obs_codes(
        sidereon.GnssSystem.GPS
    )
    assert [
        (shift.code, shift.correction_cycles) for shift in reparsed.header.phase_shifts
    ] == [(shift.code, shift.correction_cycles) for shift in obs.header.phase_shifts]

    before = obs.carrier_phase_rows(0)
    after = reparsed.carrier_phase_rows(0)
    assert after.codes == before.codes
    assert after.phase_shift_statuses == before.phase_shift_statuses
    assert [result.corrections for result in after.phase_shift_results] == [
        result.corrections for result in before.phase_shift_results
    ]
    np.testing.assert_array_equal(after.value_cycles, before.value_cycles)


def test_the_write_error_type_is_registered_and_always_answers_detail():
    """The RINEX writer and the scenario observation writer raise the same
    exception, so a caller handles one type for both. It subclasses the OBS
    parse error, so code that caught that before still catches these.
    """
    assert issubclass(sidereon.RinexObsWriteError, sidereon.RinexObsParseError)
    # A hand-built instance carries no core context, and reading `.detail` on
    # it gives None rather than raising AttributeError.
    assert sidereon.RinexObsWriteError.detail is None
    assert sidereon.RinexObsWriteError("built by hand").detail is None


def test_simulated_observations_serialize_and_read_back():
    """`SyntheticObservationSet.to_rinex_string()` is the other caller of the
    fallible writer. The text it returns parses as the product the simulator
    built: the same header identity, the same epochs and satellites, and every
    observable at the three decimals a RINEX field holds.
    """
    output = sidereon.simulate_scenario(synthetic_scenario())
    assert output.observation_count() == 10

    text = output.to_rinex_string()
    obs = sidereon.parse_rinex_obs(text)

    assert obs.header.version == 3.05
    assert obs.header.marker_name == "SYNTHETIC"
    assert obs.header.marker_type == "SIMULATED"
    assert obs.header.n_satellites == 5
    assert obs.header.interval_s == pytest.approx(30.0)
    assert obs.header.obs_codes(sidereon.GnssSystem.GPS) == ["C1C", "L1C", "D1C"]

    assert obs.epoch_count == 2
    assert obs.skipped_records == 0
    assert [epoch.flag for epoch in obs.epochs] == [0, 0]
    assert [epoch.satellites for epoch in obs.epochs] == [
        ["G01", "G02", "G03", "G04", "G05"],
        ["G01", "G02", "G03", "G04", "G05"],
    ]

    # Every observation the simulator produced is in the text, under its own
    # code, rounded to the field's three decimals and no further.
    codes = obs.header.obs_codes(sidereon.GnssSystem.GPS)
    arrays = output.observations
    for index in range(output.observation_count()):
        epoch = obs.epoch(int(arrays.epoch_index[index]))
        values = epoch.sats[arrays.satellite_id[index]]
        code = values[codes.index(arrays.code_observable[index])]
        phase = values[codes.index(arrays.phase_observable[index])]
        doppler = values[codes.index(arrays.doppler_observable[index])]
        assert code.value == pytest.approx(arrays.pseudorange_m[index], rel=0, abs=5e-4)
        assert phase.value == pytest.approx(
            arrays.carrier_phase_cycles[index], rel=0, abs=5e-4
        )
        assert doppler.value == pytest.approx(arrays.doppler_hz[index], rel=0, abs=5e-4)

    # The same simulator output writes the same text every time.
    assert sidereon.simulate_scenario(synthetic_scenario()).to_rinex_string() == text


# --- Header records kept for rewrite ----------------------------------------


def test_header_keeps_records_it_states_no_value_for():
    """`PRN / # OF OBS` counts and unretained labels are diagnostics about the
    text, kept so a caller can see what the product does and does not hold.
    """
    text = obs_text(
        3.05,
        [header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
        [
            epoch_line(0, 0.0, 0, 1),
            obs_record(
                "G01", [obs_field(20000000.0, 0, 7), obs_field(100000.125, 0, 7)]
            ),
        ],
        extra_headers=[
            header_line("   G01     1     1", "PRN / # OF OBS"),
            header_line("A FREE COMMENT", "COMMENT"),
        ],
    )
    obs = sidereon.parse_rinex_obs(text)
    assert obs.header.prn_obs_counts == {"G01": [1, 1]}
    assert obs.header.comments == ["A FREE COMMENT"]
    assert obs.header.unretained_header_labels == []


def test_scale_factor_records_are_exposed_as_written():
    text = obs_text(
        3.05,
        [header_line("G    2 C1C L1C", "SYS / # / OBS TYPES")],
        [],
        extra_headers=[header_line("G 1000   1 L1C", "SYS / SCALE FACTOR")],
    )
    obs = sidereon.parse_rinex_obs(text)
    factor = obs.header.scale_factors[0]
    assert factor.system == sidereon.GnssSystem.GPS
    assert factor.factor == 1000.0
    assert factor.codes == ["L1C"]


# --- QC over the same products ----------------------------------------------


def test_observation_qc_counts_events_and_slips_apart_from_epochs():
    obs = sidereon.parse_rinex_obs(_mixed_record_kinds_text())
    report = sidereon.observation_qc(obs)

    assert report.total_epoch_records == 4
    # Flags 0 and 1 are observation epochs; everything above 1 is an event
    # record, the cycle slip epoch among them.
    assert report.observation_epochs == 2
    assert report.event_records == 2
    assert report.power_failure_epochs == 0
    assert report.skipped_records == 0

    kinds = {note.kind for note in report.notes}
    assert kinds <= {
        "non_monotonic_epoch",
        "interval_unresolved",
        "event_header_records_unread",
    }
    # Two observation epochs 60 s apart resolve an interval, and the event
    # records read, so neither of those notes is raised.
    assert "event_header_records_unread" not in kinds
    for note in report.notes:
        if note.kind == "non_monotonic_epoch":
            assert isinstance(note.epoch_index, int)
        else:
            assert note.epoch_index is None
        assert note.kind in repr(note)


def test_observation_qc_notes_an_unresolvable_interval():
    obs = sidereon.parse_rinex_obs(
        obs_text(
            3.05,
            [header_line("G    1 C1C", "SYS / # / OBS TYPES")],
            [
                epoch_line(0, 0.0, 0, 1),
                obs_record("G01", [obs_field(20000000.0, 0, 7)]),
            ],
        )
    )
    report = sidereon.observation_qc(obs)
    assert report.observation_epochs == 1
    assert [note.kind for note in report.notes] == ["interval_unresolved"]
    assert report.notes[0].epoch_index is None


def test_lint_findings_carry_their_native_payload_not_only_text():
    obs_source = _mixed_record_kinds_text()
    report = sidereon.lint_rinex_obs(obs_source)

    by_kind = {}
    for finding in report.findings:
        by_kind.setdefault(finding.kind, []).append(finding)

    # The event epoch is reported with its own flag as a number.
    assert "ObsEventEpoch" in by_kind
    assert len(by_kind["ObsEventEpoch"]) == 1
    event = by_kind["ObsEventEpoch"][0]
    assert event.code == "OBS-B07"
    assert event.details() == {"flag": 4}
    assert event.at.epoch_index == 2
    assert event.severity.label == "info"
    # Every finding answers `details()`, and the keys are its own payload.
    for finding in report.findings:
        assert isinstance(finding.details(), dict)
        assert finding.code
        assert finding.kind != "Unknown"


def test_a_time_of_first_obs_mismatch_reports_typed_epochs_and_scales():
    """The richest lint payload carries four values, none of them text: the
    declared and observed epochs as `ObsEpochTime`, each with its `TimeScale`.
    """
    text = obs_text(
        3.05,
        [header_line("G    1 C1C", "SYS / # / OBS TYPES")],
        [
            epoch_line(0, 0.0, 0, 1),
            obs_record("G01", [obs_field(20000000.0, 0, 7)]),
        ],
        extra_headers=[time_of_first_obs(2020, 1, 1, 1, 0, 0.0)],
    )
    obs = sidereon.parse_rinex_obs(text)
    declared, declared_scale = obs.header.time_of_first_obs
    assert (declared.hour, declared.minute) == (1, 0)
    assert declared_scale == sidereon.TimeScale.GPST

    findings = [
        finding
        for finding in sidereon.lint_rinex_obs(text).findings
        if finding.kind == "ObsTimeOfFirstMismatch"
    ]
    assert len(findings) == 1
    payload = findings[0].details()
    assert findings[0].code == "OBS-H07"
    assert list(payload) == [
        "declared",
        "declared_scale",
        "observed",
        "observed_scale",
    ]
    assert isinstance(payload["declared"], sidereon.ObsEpochTime)
    assert isinstance(payload["observed"], sidereon.ObsEpochTime)
    assert payload["declared"] == declared
    assert (payload["declared"].hour, payload["declared"].minute) == (1, 0)
    assert (payload["observed"].hour, payload["observed"].minute) == (0, 0)
    assert payload["declared_scale"] == sidereon.TimeScale.GPST
    assert payload["observed_scale"] == sidereon.TimeScale.GPST


def crx2rnx_spelling(text):
    """The lines of RINEX observation text `text` with every observation
    spelled as the reference `crx2rnx` spells it.

    CRINEX carries an observation as the integer its three decimals scale to,
    not as the characters written, and the decoder restates that integer in
    the `F14.3` layout `crx2rnx` writes: a negative value above -1 without the
    zero before the decimal point (`-.920`), and zero without a sign. The
    RINEX writer keeps the zero (`-0.920`). Both spell the same value. The
    header passes through CRINEX as written, and a RINEX 3 epoch line opens
    with `>`, so only the satellite lines after `END OF HEADER` are respelled.
    """

    def respell(match):
        digits = match.group(1)
        return "  0.000" if digits == "000" else f"  -.{digits}"

    lines = text.splitlines()
    end = next(
        index
        for index, line in enumerate(lines)
        if line[60:].strip() == "END OF HEADER"
    )
    body = [
        line if line.startswith(">") else re.sub(r" -0\.(\d{3})", respell, line)
        for line in lines[end + 1 :]
    ]
    return lines[: end + 1] + body


def test_a_repair_writes_crinex_that_decodes_to_the_repaired_product():
    """`RinexObsRepair.to_crinex_string()` writes the repaired product as RINEX
    and encodes that text. Decoding the result gives back that text with each
    observation in the spelling `crx2rnx` uses, and it parses as the product
    the repair holds, value for value.
    """
    text = read_fixture(ESBC_RNX)
    repair = sidereon.repair_rinex_obs(
        text,
        sidereon.RinexRepairOptions(
            set_interval=True,
            set_time_of_last_obs=True,
            set_obs_counts=True,
            drop_empty_records=True,
            drop_unsupported=True,
        ),
    )
    repaired = repair.repaired

    crinex = repair.to_crinex_string()
    assert crinex.splitlines()[0][60:].strip() == "CRINEX VERS   / TYPE"
    # The text handed to the encoder is the repaired product's own RINEX.
    written = repaired.to_rinex_string()
    assert crinex == sidereon.encode_crinex(written)

    decoded_text = sidereon.decode_crinex(crinex)
    # The repaired product holds values in (-1, 0), so the two spellings
    # differ in this file, and that is the only difference.
    assert decoded_text.splitlines() != written.splitlines()
    assert decoded_text.splitlines() == crx2rnx_spelling(written)

    decoded = sidereon.parse_rinex_obs(decoded_text)
    assert decoded.epoch_count == repaired.epoch_count
    assert decoded.header.version == repaired.header.version
    assert [epoch.satellites for epoch in decoded.epochs] == [
        epoch.satellites for epoch in repaired.epochs
    ]
    for system in repaired.header.systems:
        assert decoded.header.obs_codes(system) == repaired.header.obs_codes(system)

    for decoded_epoch, repaired_epoch in zip(decoded.epochs, repaired.epochs):
        decoded_sats = decoded_epoch.sats
        repaired_sats = repaired_epoch.sats
        for satellite in repaired_epoch.satellites:
            assert [
                (value.value, value.lli, value.ssi) for value in decoded_sats[satellite]
            ] == [
                (value.value, value.lli, value.ssi)
                for value in repaired_sats[satellite]
            ]


def rinex2_epoch_line(year, month, day, hour, minute, second, flag, satellites):
    """A version 2 epoch record in the eight fixed fields the parser reads at
    columns 1-3, 4-6, 7-9, 10-12, 13-15, 15-26, 28-29 and 29-32, then the
    satellites it names, three columns each from column 32.
    """
    head = (
        f" {year % 100:02d} {month:2d} {day:2d} {hour:2d} {minute:2d}"
        f"{second:11.7f}  {flag:1d}{len(satellites):3d}"
    )
    return head + "".join(satellites)


def _rinex2_text_with_scale_factor():
    """Version 2.11 text the parser accepts whole, carrying one
    `SYS / SCALE FACTOR` record.

    `SYS / SCALE FACTOR` has no version gate on the reading side: the parser
    takes it in a version 2 header and keeps it, so the product holds a record
    a version 2 file's writer has no record to put it in. The factor is 10 over
    every code of GPS, and both values divide by it exactly, so nothing here
    turns on float rounding.
    """
    return obs_text(
        2.11,
        [header_line("     2    C1    L1", "# / TYPES OF OBSERV")],
        [
            rinex2_epoch_line(2020, 1, 1, 0, 0, 0.0, 0, ["G01"]),
            obs_record("", [obs_field(20000000.0, 0, 7), obs_field(100000.0, 0, 7)]),
        ],
        extra_headers=[
            time_of_first_obs(2020, 1, 1, 0, 0, 0.0),
            header_line("G   10", "SYS / SCALE FACTOR"),
        ],
    )


def test_a_repair_of_accepted_version_two_text_reaches_the_writer_refusal():
    """The parser accepting a record and the writer having a field for it are
    two different questions, and `SYS / SCALE FACTOR` in a version 2 header is
    where they part.

    The reader takes the record without asking the version and keeps it on the
    product. A repair under the default options parses and mechanically edits;
    it removes no scale factor, and it serializes nothing, so the product
    reaches `to_crinex_string()` still carrying the record. The version 2
    writer has no record to write it as and refuses with
    `ScaleFactorsInVersionTwo`, counting the records it could not place.

    So this refusal is reachable from public calls alone, on text this binding
    accepts: no hand-built product, no setter, no unexposed constructor.
    """
    text = _rinex2_text_with_scale_factor()

    # The parse keeps everything, so no parse or unretained-record failure
    # stands between this text and the writer.
    source = sidereon.parse_rinex_obs(text)
    assert source.header.version == 2.11
    assert source.header.unretained_header_labels == []
    assert source.header.scale_factors[0].factor == 10.0
    assert source.header.scale_factors[0].codes == []

    repair = sidereon.repair_rinex_obs(text)
    # The repair carried the record through rather than removing it, which is
    # what leaves the product unwritable.
    retained = repair.repaired.header.scale_factors
    assert len(retained) == 1
    assert retained[0].system == sidereon.GnssSystem.GPS
    assert retained[0].factor == 10.0
    assert repair.repaired.header.version == 2.11

    with pytest.raises(sidereon.RinexObsWriteError) as raised:
        repair.to_crinex_string()

    detail = raised.value.detail
    assert detail is not None
    assert detail.kind == "ScaleFactorsInVersionTwo"
    assert detail.details() == {"count": 1}
    # A native count under its own key, not text to be parsed back out.
    assert isinstance(detail.details()["count"], int)
    assert not isinstance(detail.details()["count"], bool)
    # The variant names no epoch, constellation or satellite, and answers None
    # for each rather than raising.
    assert detail.epoch_index is None
    assert detail.system is None
    assert detail.satellite is None
    # The broad types a caller may already be catching still catch it.
    assert isinstance(raised.value, sidereon.RinexObsParseError)
    assert isinstance(raised.value, sidereon.ParseError)
    assert isinstance(raised.value, sidereon.SidereonError)

    # Writing the repaired product directly is the same refusal: the CRINEX
    # call composes the RINEX writer and the encoder, and the writer refuses
    # first, so the encoder is never reached.
    with pytest.raises(sidereon.RinexObsWriteError) as direct:
        repair.repaired.to_rinex_string()
    assert direct.value.detail.kind == "ScaleFactorsInVersionTwo"
    assert direct.value.detail.details() == {"count": 1}
