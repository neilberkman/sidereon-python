"""Station displacement tide models (solid-earth, pole, ocean loading).

`solid_earth_tide`, `solid_earth_pole_tide`, and `ocean_tide_loading` are thin
wrappers over `sidereon_core::tides`. The solid-earth test replays the core's
Dehant golden cases and asserts the returned displacement matches the recorded
IERS reference vector; the pole and ocean tests assert the delegation's
structural invariants.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

DEHANT = os.path.join(CORE_FIXTURES, "tides", "tides_dehant_golden.json")
N_CONSTITUENTS = 11


def _fhr(year, month, day, hour):  # convenience for readable epochs
    return float(hour)


def test_solid_earth_tide_matches_dehant_golden():
    with open(DEHANT) as fh:
        golden = json.load(fh)
    cases = golden["cases"]
    assert len(cases) > 0
    for case in cases:
        # case_4 is a known fixture transcription artifact (its expected vector
        # is a verbatim copy of case_3's while its Sun vector is unphysical); the
        # core's own golden test excludes it for the same reason.
        if case["id"] == "case_4_2017_01_15":
            continue
        inp = case["inputs"]
        xsta = np.asarray(inp["xsta_m"]["values"], dtype=np.float64)
        xsun = np.asarray(inp["xsun_m"]["values"], dtype=np.float64)
        xmon = np.asarray(inp["xmon_m"]["values"], dtype=np.float64)
        date = inp["date_utc"]
        fhr = inp["fhr_hours"]["value"]
        out = sidereon.station_tide_displacement(
            xsta,
            date["year"],
            date["month"],
            date["day"],
            fhr,
            xsun,
            xmon,
            constants=sidereon.StationTideConstants.IERS_ROUTINE,
        )
        expected = np.asarray(case["expected"]["dxtide_m"]["values"], dtype=np.float64)
        assert out.shape == (3,)
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1.0e-12)


def test_station_tide_constants_select_core_variant():
    station = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    sun = np.asarray([1.0e11, 2.0e10, 3.0e10], dtype=np.float64)
    moon = np.asarray([2.0e8, -1.0e8, 1.0e8], dtype=np.float64)
    default = sidereon.station_tide_displacement(station, 2009, 4, 13, 0.0, sun, moon)
    conventions = sidereon.station_tide_displacement(
        station,
        2009,
        4,
        13,
        0.0,
        sun,
        moon,
        sidereon.StationTideConstants.CONVENTIONS,
    )
    routine = sidereon.station_tide_displacement(
        station,
        2009,
        4,
        13,
        0.0,
        sun,
        moon,
        sidereon.StationTideConstants.IERS_ROUTINE,
    )
    np.testing.assert_array_equal(default, conventions)
    assert default.shape == routine.shape == (3,)
    assert np.all(np.isfinite(routine))


def test_station_displacement_reports_permissive_ut1_degradation():
    station = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    result = sidereon.station_displacement_ecef_m(
        station,
        2500,
        1,
        1,
        0,
        0,
        0.0,
        validity=sidereon.ValidityMode.PERMISSIVE,
    )
    assert result.ut1_degraded
    assert result.degradation_reason == "after_coverage"
    assert np.all(np.isfinite(result.ecef_m))
    assert result.solid_earth_tide_ecef_m is not None
    assert result.pole_tide_ecef_m is None
    assert result.ocean_loading_ecef_m is None

    empty = sidereon.station_displacement_ecef_m(
        station, 2009, 4, 13, 0, 0, 0.0, solid_earth_tide=False
    )
    np.testing.assert_array_equal(empty.ecef_m, np.zeros(3))
    assert empty.solid_earth_tide_ecef_m is None

    with pytest.raises(sidereon.TideEvaluationError) as refusal:
        sidereon.station_displacement_ecef_m(
            station,
            2500,
            1,
            1,
            0,
            0,
            0.0,
            validity=sidereon.ValidityMode.STRICT,
        )
    assert refusal.value.kind == "frame_transform"
    assert refusal.value.details == {
        "source_kind": "Ut1OutsideCoverage",
        "reason": "after_coverage",
    }


def test_station_tide_input_refusal_preserves_core_variant():
    station = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    with pytest.raises(sidereon.TideEvaluationError) as refusal:
        sidereon.station_displacement_ecef_m(
            station,
            2009,
            4,
            13,
            0,
            0,
            0.0,
            solid_earth_tide=False,
            pole_tide=True,
        )
    assert refusal.value.kind == "missing_input"
    assert refusal.value.details["field"] == "polar motion"

    with pytest.raises(sidereon.TideEvaluationError) as transform_refusal:
        sidereon.station_displacement_ecef_m(
            station,
            2009,
            4,
            13,
            0,
            0,
            0.0,
            solid_earth_tide=False,
            pole_tide=True,
            polar_motion_arcsec=(float("inf"), 0.0),
        )
    assert transform_refusal.value.kind == "invalid_input"
    assert transform_refusal.value.details["field"] == "polar motion xp"
    assert transform_refusal.value.details["reason"] == "NonFinite"


def test_station_displacement_typed_epoch_options_and_batch_rows():
    position = sidereon.StationDisplacementPosition.from_ecef_m(
        [4_075_578.385, 931_852.89, 4_801_570.154]
    )
    geodetic = sidereon.StationDisplacementPosition.from_geodetic(0.5, 0.2, 100.0)
    assert geodetic is not None
    epoch = sidereon.StationDisplacementEpoch.from_utc(2009, 4, 13, 0, 0, 0.0)
    polar_epoch = epoch.with_polar_motion_arcsec(0.12, 0.34)
    assert polar_epoch.polar_motion.xp_arcsec == 0.12
    options = sidereon.StationDisplacementOptions(solid_earth_tide=False)

    expected = sidereon.station_displacement_ecef_m_at_epoch(position, epoch, options)
    assert expected.ecef_m == [0.0, 0.0, 0.0]
    rows = sidereon.station_displacement_ecef_m_batch(
        position,
        [epoch, sidereon.StationDisplacementEpoch.from_utc(2009, 2, 30, 0, 0, 0.0)],
        options,
    )
    assert rows[0].value.ecef_m == expected.ecef_m
    assert rows[0].error_kind is None
    assert rows[1].value is None
    assert rows[1].error_kind == "invalid_input"
    assert rows[1].error_details == [
        ("field", "civil datetime"),
        ("reason", "InvalidCivilDate"),
    ]
    assert [item.label for item in sidereon.ocean_loading_constituents()] == [
        "M2",
        "S2",
        "N2",
        "K2",
        "K1",
        "O1",
        "P1",
        "Q1",
        "Mf",
        "Mm",
        "Ssa",
    ]


def test_solid_earth_pole_tide_returns_finite_vector():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    out = sidereon.solid_earth_pole_tide(xsta, 2009, 4, 13, 0.0, 0.12, 0.34)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_ocean_tide_loading_zero_coefficients_is_zero():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    zero = [[0.0] * N_CONSTITUENTS for _ in range(3)]
    out = sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 0.0, zero, zero)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_ocean_tide_loading_nonzero_coefficients_is_finite():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    amplitude = [
        [
            0.003,
            0.001,
            0.0007,
            0.0003,
            0.002,
            0.0015,
            0.0006,
            0.0002,
            0.0004,
            0.0002,
            0.0001,
        ],
        [
            0.001,
            0.0005,
            0.0003,
            0.0001,
            0.0008,
            0.0006,
            0.0002,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ],
        [
            0.002,
            0.0008,
            0.0005,
            0.0002,
            0.0012,
            0.0009,
            0.0004,
            0.0001,
            0.0002,
            0.0001,
            0.0001,
        ],
    ]
    phase = [[10.0 * (c + 1) for c in range(N_CONSTITUENTS)] for _ in range(3)]
    out = sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 6.0, amplitude, phase)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.linalg.norm(out) > 0.0


def test_ocean_tide_loading_rejects_wrong_shape():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    two_rows = [[0.0] * N_CONSTITUENTS for _ in range(2)]
    three_rows = [[0.0] * N_CONSTITUENTS for _ in range(3)]
    with pytest.raises(ValueError):
        sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 0.0, two_rows, three_rows)

    short_row = [[0.0] * (N_CONSTITUENTS - 1) for _ in range(3)]
    with pytest.raises(ValueError):
        sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 0.0, short_row, three_rows)


# Real ZIM2 public BLQ block (Onsala Space Observatory ocean tide loading
# provider) in the standard column order, the block the core tests read.
ZIM2_BLQ_BLOCK = """
$$ Station: ZIM2, Zimmerwald
$$ Source: Onsala Space Observatory ocean tide loading provider, ZIM2 public BLQ
$$ Ocean model: GOT4.7, long-period tides from FES99
$$ Column order: M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa
ZIM2
 0.00693 0.00228 0.00148 0.00061 0.00220 0.00094 0.00070 0.00001 0.00047 0.00025 0.00019
 0.00272 0.00076 0.00061 0.00020 0.00036 0.00025 0.00011 0.00005 0.00004 0.00001 0.00002
 0.00061 0.00026 0.00010 0.00009 0.00025 0.00002 0.00008 0.00003 0.00002 0.00000 0.00001
-72.3 -44.2 -90.8 -44.1 -62.9 -94.5 -64.3 171.0 3.4 3.6 1.1
 84.3 115.4 63.3 113.7 98.6 20.7 94.2 -44.5 -170.0 -162.7 -177.8
-29.3 1.7 -44.0 -4.2 44.2 -39.1 43.7 170.1 -93.3 -118.3 -176.4
"""


def test_blq_block_round_trips_with_its_comments_and_station_column():
    block = sidereon.parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK)
    assert sidereon.OceanLoadingBlq.from_blq_block(ZIM2_BLQ_BLOCK) == block
    assert block.station == "ZIM2"
    coefficients = block.coefficients
    assert coefficients.amplitude_m[0][0] == 0.00693
    assert coefficients.phase_deg[2][10] == -176.4
    assert [comment.line for comment in block.comments] == [
        "$$ Station: ZIM2, Zimmerwald",
        "$$ Source: Onsala Space Observatory ocean tide loading provider, "
        "ZIM2 public BLQ",
        "$$ Ocean model: GOT4.7, long-period tides from FES99",
        "$$ Column order: M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa",
    ]
    assert {comment.placement for comment in block.comments} == {"before_station"}

    encoded = block.to_blq_block()
    # The station is written from the third column, where the provider's files
    # put it and RTKLIB `readblq` reads it.
    assert "\n  ZIM2\n" in encoded
    assert sidereon.parse_ocean_loading_blq_block(encoded) == block

    # The coefficients drive the displacement model unchanged.
    xsta = np.asarray([4331297.3, 567555.6, 4633133.7], dtype=np.float64)
    out = sidereon.ocean_tide_loading(
        xsta,
        2023,
        6,
        1,
        12.0,
        coefficients.amplitude_m,
        coefficients.phase_deg,
    )
    assert out.shape == (3,) and np.all(np.isfinite(out))


def test_blq_comments_keep_their_places_across_blocks():
    text = """$$ header
AAA
$$ before row 0
 1 0 0 0 0 0 0 0 0 0 0
 2 0 0 0 0 0 0 0 0 0 0
# between rows 2 and 3
 3 0 0 0 0 0 0 0 0 0 0
 4 0 0 0 0 0 0 0 0 0 0
 5 0 0 0 0 0 0 0 0 0 0
 6 0 0 0 0 0 0 0 0 0 0
$$ between blocks
BBB
 7 0 0 0 0 0 0 0 0 0 0
 8 0 0 0 0 0 0 0 0 0 0
 9 0 0 0 0 0 0 0 0 0 0
 10 0 0 0 0 0 0 0 0 0 0
 11 0 0 0 0 0 0 0 0 0 0
 12 0 0 0 0 0 0 0 0 0 0
! trailing
"""
    blocks = sidereon.parse_ocean_loading_blq_blocks(text)
    assert [
        (comment.placement, comment.row, comment.line) for comment in blocks[0].comments
    ] == [
        ("before_station", None, "$$ header"),
        ("before_row", 0, "$$ before row 0"),
        ("before_row", 2, "# between rows 2 and 3"),
    ]
    assert [
        (comment.placement, comment.row, comment.line) for comment in blocks[1].comments
    ] == [
        ("before_station", None, "$$ between blocks"),
        ("after_rows", None, "! trailing"),
    ]
    written = sidereon.write_ocean_loading_blq_blocks(blocks)
    assert sidereon.parse_ocean_loading_blq_blocks(written) == blocks

    # A comment after the rows of a block that is not the last would be read
    # as the next block's, so the writer refuses it.
    with pytest.raises(sidereon.BlqError) as excinfo:
        sidereon.write_ocean_loading_blq_blocks([blocks[1], blocks[0]])
    detail = excinfo.value.detail
    assert (detail.kind, detail.reason, detail.block) == (
        "BlqWrite",
        "AfterRowsBeforeAnotherBlock",
        0,
    )


def test_blq_writer_refuses_what_the_parser_would_not_read_back():
    valid = sidereon.parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK)

    def refusal(block):
        with pytest.raises(sidereon.BlqError) as excinfo:
            block.to_blq_block()
        assert isinstance(excinfo.value, ValueError)
        return excinfo.value.detail.details()

    def with_station(station):
        return sidereon.OceanLoadingBlqBlock(
            station, valid.coefficients, valid.comments
        )

    assert refusal(with_station(" ZIM2")) == {
        "block": 0,
        "reason": "StationSurroundingWhitespace",
    }
    assert refusal(with_station("-12.5"))["reason"] == "StationReadsAsCoefficientRow"

    amplitude = valid.coefficients.amplitude_m
    amplitude[0][0] = float("nan")
    nan_block = sidereon.OceanLoadingBlqBlock(
        "ZIM2",
        sidereon.OceanLoadingBlq(amplitude, valid.coefficients.phase_deg),
        valid.comments,
    )
    assert refusal(nan_block) == {
        "block": 0,
        "reason": "NonFiniteCoefficient",
        "row": 0,
        "constituent": "M2",
    }

    last = len(valid.comments)
    header = sidereon.OceanLoadingBlqBlock(
        "ZIM2",
        valid.coefficients,
        [
            *valid.comments,
            sidereon.OceanLoadingBlqComment(
                "$$ Column order: nonsense", "before_station"
            ),
        ],
    )
    assert refusal(header) == {
        "block": 0,
        "reason": "InvalidHeader",
        "index": last,
        "header": {"reason": "UnsupportedConstituent", "constituent": "NONSENSE"},
    }
    beyond = sidereon.OceanLoadingBlqBlock(
        "ZIM2",
        valid.coefficients,
        [sidereon.OceanLoadingBlqComment("$$ after the rows", "before_row", 6)],
    )
    assert refusal(beyond)["reason"] == "CommentPlacementOutOfRange"

    with pytest.raises(ValueError, match="needs a row"):
        sidereon.OceanLoadingBlqComment("$$ x", "before_row")


def test_blq_parse_refusals_are_typed():
    with pytest.raises(sidereon.BlqError) as excinfo:
        sidereon.parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK.replace("Ssa", "M4"))
    detail = excinfo.value.detail
    assert (detail.kind, detail.reason) == ("BlqParse", "UnsupportedConstituent")
    assert detail.details()["constituent"] == "M4"
    assert isinstance(detail.line, int)
    assert sidereon.BlqError("x").detail is None
