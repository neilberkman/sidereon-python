"""Public ClockHeaderField payload coverage for the RINEX 3.04 examples."""

from pathlib import Path

import sidereon


def _fixture(name: str) -> Path:
    return Path(__file__).parent / "fixtures" / "clk" / "lossless" / name


def test_public_header_records_expose_every_clock_header_field_payload():
    clock = sidereon.load_rinex_clock(str(_fixture("rinex_clock304_table_a17.clk")))
    expected = [
        ("VersionType", {"version": 3.04, "file_type": "C", "satellite_system": "G"}),
        (
            "ProgramRunByDate",
            {
                "program": "TORINEXC V9.9",
                "run_by": "USNO",
                "date": "19960403  001000 UTC",
            },
        ),
        ("Comment", {"text": "EXAMPLE OF A CLOCK DATA ANALYSIS FILE"}),
        (
            "Comment",
            {"text": "IN THIS CASE ANALYSIS RESULTS FROM GPS ONLY ARE INCLUDED"},
        ),
        ("Comment", {"text": "No re-alignment of the clocks has been applied."}),
        (
            "ObservationTypes",
            {"system": "G", "count": 4, "descriptors": ["C1W", "L1W", "C2W", "L2W"]},
        ),
        ("TimeSystem", {"label": "GPS"}),
        ("LeapSeconds", {"seconds": 10}),
        (
            "DcbsApplied",
            {
                "system": "G",
                "program": "CC2NONCC",
                "source": "p1c1bias.hist @ goby.nrl.navy.mil",
            },
        ),
        (
            "PcvsApplied",
            {
                "system": "G",
                "program": "PAGES",
                "source": "igs05.atx @ igscb.jpl.nasa.gov",
            },
        ),
        ("TypesOfData", {"count": 2, "types": ["AS", "AR"]}),
        ("AnalysisCenter", {"designator": "USN", "name": "USNO USING GIPSY/OASIS-II"}),
        (
            "ClockRefCount",
            {
                "count": 1,
                "start": sidereon.ClockEpoch(1994, 7, 14, 0, 0, 0.0),
                "stop": sidereon.ClockEpoch(1994, 7, 14, 20, 59, 0.0),
            },
        ),
        (
            "AnalysisClockRef",
            {
                "name": "USNO",
                "identifier": "40451S003",
                "constraint_s": -0.123456789012,
            },
        ),
        (
            "ClockRefCount",
            {
                "count": 1,
                "start": sidereon.ClockEpoch(1994, 7, 14, 21, 0, 0.0),
                "stop": sidereon.ClockEpoch(1994, 7, 14, 21, 59, 0.0),
            },
        ),
        (
            "AnalysisClockRef",
            {
                "name": "TIDB",
                "identifier": "50103M108",
                "constraint_s": -0.123456789012,
            },
        ),
        ("SolutionStationCount", {"count": 4, "frame": "ITRF96"}),
        (
            "SolutionStation",
            {
                "name": "GOLD",
                "identifier": "40405S031",
                "xyz_mm": [1234567890, -1234567890, -1234567890],
            },
        ),
        (
            "SolutionStation",
            {
                "name": "AREQ",
                "identifier": "42202M005",
                "xyz_mm": [-1234567890, 1234567890, -1234567890],
            },
        ),
        (
            "SolutionStation",
            {
                "name": "TIDB",
                "identifier": "50103M108",
                "xyz_mm": [1234567890, -1234567890, 1234567890],
            },
        ),
        (
            "SolutionStation",
            {
                "name": "HARK",
                "identifier": "30302M007",
                "xyz_mm": [-1234567890, 1234567890, -1234567890],
            },
        ),
        (
            "SolutionStation",
            {
                "name": "USNO",
                "identifier": "40451S003",
                "xyz_mm": [1234567890, -1234567890, -1234567890],
            },
        ),
        ("SolutionSatelliteCount", {"count": 27}),
        (
            "PrnList",
            {
                "prns": [
                    "G01",
                    "G02",
                    "G03",
                    "G04",
                    "G05",
                    "G06",
                    "G07",
                    "G08",
                    "G09",
                    "G10",
                    "G13",
                    "G14",
                    "G15",
                    "G16",
                    "G17",
                    "G18",
                ]
            },
        ),
        (
            "PrnList",
            {
                "prns": [
                    "G19",
                    "G21",
                    "G22",
                    "G23",
                    "G24",
                    "G25",
                    "G26",
                    "G27",
                    "G29",
                    "G30",
                    "G31",
                ]
            },
        ),
        ("EndOfHeader", {}),
    ]
    assert [(r.field.kind, r.field.details()) for r in clock.header_records] == expected
    reparsed = sidereon.load_rinex_clock(str(_fixture("rinex_clock304_table_a17.clk")))
    assert [record.field for record in clock.header_records] == [
        record.field for record in reparsed.header_records
    ]
    assert {kind for kind, _ in expected} == {
        "VersionType",
        "ProgramRunByDate",
        "Comment",
        "ObservationTypes",
        "TimeSystem",
        "LeapSeconds",
        "DcbsApplied",
        "PcvsApplied",
        "TypesOfData",
        "AnalysisCenter",
        "ClockRefCount",
        "AnalysisClockRef",
        "SolutionStationCount",
        "SolutionStation",
        "SolutionSatelliteCount",
        "PrnList",
        "EndOfHeader",
    }


def test_public_header_records_expose_rinex304_gnss_and_station_fields():
    clock = sidereon.load_rinex_clock(str(_fixture("rinex_clock304_table_a18.clk")))
    fields = [(r.field.kind, r.field.details()) for r in clock.header_records]
    assert ("LeapSecondsGnss", {"seconds": 10}) in fields
    assert ("StationNameNum", {"name": "USNO", "identifier": "40451S003"}) in fields
    assert (
        "StationClockRef",
        {"text": "UTC(USNO) MASTER CLOCK VIA CONTINUOUS CABLE MONITOR"},
    ) in fields


def test_public_header_records_keep_blank_optional_fields_and_continuations():
    source = _fixture("rinex_clock304_table_a17.clk").read_text()
    lines = source.splitlines()
    end_line = next(line for line in lines if "END OF HEADER" in line)
    continuation = "        L5Q".ljust(65) + "SYS / # / OBS TYPES"
    source_with_continuation = source.replace(
        end_line, continuation + "\n" + end_line, 1
    )
    continued = sidereon.parse_rinex_clock(source_with_continuation)
    observation = [
        record.field.details()
        for record in continued.header_records
        if record.field.kind == "ObservationTypes"
    ]
    assert observation[-1] == {
        "system": None,
        "count": None,
        "descriptors": ["L5Q"],
    }

    clock_ref = next(line for line in lines if "# OF CLK REF" in line)
    blank_clock_ref = "     1".ljust(65) + "# OF CLK REF"
    source_with_blank_clock_ref = source.replace(clock_ref, blank_clock_ref, 1)
    blank_epoch = sidereon.parse_rinex_clock(source_with_blank_clock_ref)
    ref_details = next(
        record.field.details()
        for record in blank_epoch.header_records
        if record.field.kind == "ClockRefCount"
    )
    assert ref_details == {"count": 1, "start": None, "stop": None}

    analysis_ref = next(line for line in lines if "ANALYSIS CLK REF" in line)
    blank_constraint = "USNO      40451S003".ljust(65) + "ANALYSIS CLK REF"
    source_with_blank_constraint = source.replace(analysis_ref, blank_constraint, 1)
    no_constraint = sidereon.parse_rinex_clock(source_with_blank_constraint)
    constraint_details = next(
        record.field.details()
        for record in no_constraint.header_records
        if record.field.kind == "AnalysisClockRef"
    )
    assert constraint_details == {
        "name": "USNO",
        "identifier": "40451S003",
        "constraint_s": None,
    }
