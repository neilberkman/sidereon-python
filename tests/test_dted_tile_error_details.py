"""Public DTED tile routes preserve typed parser and lookup errors."""

import struct
from pathlib import Path

import pytest
import sidereon
from _helpers import CORE_FIXTURES

FIXTURE = Path(CORE_FIXTURES) / "dted" / "tiles" / "n36_w107_1arc_v3.dt2"
DATA_OFFSET = 3428
BLOCK_LEN = 22


def tile_bytes():
    return bytearray(FIXTURE.read_bytes())


def put(data, offset, replacement):
    data[offset : offset + len(replacement)] = replacement
    return data


def checksum_block(data):
    start = DATA_OFFSET
    data[start + BLOCK_LEN - 4 : start + BLOCK_LEN] = struct.pack(
        ">i", sum(data[start : start + BLOCK_LEN - 4])
    )
    return data


def assert_tile_error(path, kind, fields, message):
    with pytest.raises(sidereon.TerrainError) as caught:
        sidereon.DtedTile.from_path(path)
    error = caught.value
    assert isinstance(error, ValueError)
    assert isinstance(error, sidereon.SidereonError)
    assert (error.detail.family, error.detail.kind) == ("DtedTileError", kind)
    assert error.detail.details() == fields
    assert error.detail.message == message


def assert_elevation_error(path, longitude, latitude, kind, fields, message):
    tile = sidereon.DtedTile.from_path(path)
    with pytest.raises(sidereon.TerrainError) as caught:
        tile.height_m(latitude, longitude)
    error = caught.value
    assert isinstance(error, ValueError)
    assert (error.detail.family, error.detail.kind) == ("DtedTileError", kind)
    assert error.detail.details() == fields
    assert error.detail.message == message


def test_public_tile_reader_retains_parser_fields_and_messages(tmp_path):
    cases = [
        (
            "missing-uhl",
            put(tile_bytes(), 0, b"NOPE"),
            "MissingUhl1",
            lambda p: {"path": str(p)},
            lambda p: f"{p} missing UHL1 header",
        ),
        (
            "bad-encoding",
            put(tile_bytes(), 4, b"\xff"),
            "InvalidEncoding",
            {"text": "invalid utf-8 sequence of 1 bytes from index 0"},
            lambda _p: "invalid utf-8 sequence of 1 bytes from index 0",
        ),
        (
            "bad-field",
            put(tile_bytes(), 47, b"abcd"),
            "InvalidField",
            {"text": "invalid digit found in string"},
            lambda _p: "invalid digit found in string",
        ),
        (
            "bad-dimensions",
            put(tile_bytes(), 47, b"0001"),
            "InvalidDimensions",
            lambda p: {"path": str(p), "lon_count": 1, "lat_count": 5},
            lambda p: (
                f"{p} has invalid DTED dimensions lon_count=1 lat_count=5; "
                + "both must be at least 2"
            ),
        ),
        (
            "truncated",
            tile_bytes()[:DATA_OFFSET],
            "Truncated",
            lambda p: {"path": str(p), "actual": DATA_OFFSET, "expected": 3538},
            lambda p: f"{p} has 3428 bytes but expected at least 3538",
        ),
        (
            "bad-hemisphere",
            put(tile_bytes(), 4, b"1070000X"),
            "InvalidHemisphere",
            {"hemisphere": "X"},
            lambda _p: "invalid DTED hemisphere X",
        ),
        (
            "out-of-range",
            put(tile_bytes(), 4, b"9990000W"),
            "CoordinateOutOfRange",
            {"field": "longitude of origin", "text": "9990000W"},
            lambda _p: 'DTED longitude of origin "9990000W" is out of range',
        ),
        (
            "wrong-hemisphere",
            put(tile_bytes(), 4, b"1060000N"),
            "WrongHemisphere",
            {"field": "longitude of origin", "hemisphere": "N", "expected": "E or W"},
            lambda _p: "DTED longitude of origin has hemisphere N, expected E or W",
        ),
        (
            "fractional-origin",
            put(tile_bytes(), 4, b"1073000W"),
            "OriginNotWholeDegree",
            {"field": "longitude of origin", "text": "1073000W"},
            lambda _p: 'DTED longitude of origin "1073000W" is not a whole degree',
        ),
        (
            "interval-mismatch",
            put(tile_bytes(), 20, b"8999"),
            "IntervalCountMismatch",
            {
                "field": "longitude data interval",
                "interval_tenths_arcsec": 8999,
                "count": 5,
            },
            lambda _p: (
                "DTED longitude data interval of 8999 tenths of an arc "
                + "second over 5 postings does not span one degree"
            ),
        ),
    ]
    for name, data, kind, fields, message in cases:
        path = tmp_path / f"n36_w107_{name}.dt2"
        path.write_bytes(data)
        assert_tile_error(
            path, kind, fields(path) if callable(fields) else fields, message(path)
        )

    short = tmp_path / "short.dt2"
    short.write_bytes(b"short")
    assert_tile_error(
        short,
        "TooShort",
        {"path": str(short)},
        f"{short} is too short for DTED headers",
    )

    missing = tmp_path / "missing.dt2"
    with pytest.raises(sidereon.TerrainError) as caught:
        sidereon.DtedTile.from_path(missing)
    detail = caught.value.detail
    assert (detail.family, detail.kind) == ("DtedTileError", "Io")
    assert set(detail.details()) == {"path", "message"}
    assert detail.details()["path"] == str(missing)
    assert detail.details()["message"] in detail.message


def test_public_tile_lookup_retains_profile_fields_and_success(tmp_path):
    cases = [
        (
            "outside",
            tile_bytes(),
            -105.5,
            36.5,
            "Outside",
            {
                "longitude": -105.5,
                "latitude": 36.5,
                "origin_longitude": -107.0,
                "origin_latitude": 36.0,
            },
            "point (-105.5,36.5) is outside DTED tile (-107,36)",
        ),
        (
            "sentinel",
            put(tile_bytes(), DATA_OFFSET, b"\x00"),
            -107.0,
            36.0,
            "MissingDataSentinel",
            {"longitude_index": 0},
            "DTED block 0 missing data sentinel",
        ),
        (
            "checksum",
            put(tile_bytes(), DATA_OFFSET + 18, struct.pack(">i", 961)),
            -107.0,
            36.0,
            "Checksum",
            {"longitude_index": 0, "checksum": 961, "sum": 960},
            "DTED checksum failed for block 0: expected 961, found 960",
        ),
        (
            "profile-count",
            checksum_block(put(tile_bytes(), DATA_OFFSET + 4, b"\x00\x01")),
            -107.0,
            36.0,
            "ProfileLongitudeCountMismatch",
            {"longitude_index": 0, "declared": 1},
            "DTED block 0 declares longitude count 1",
        ),
        (
            "partial-profile",
            checksum_block(put(tile_bytes(), DATA_OFFSET + 6, b"\x00\x01")),
            -107.0,
            36.0,
            "UnsupportedPartialProfile",
            {"longitude_index": 0, "first_latitude_index": 1},
            "DTED block 0 is a partial profile starting at latitude count 1",
        ),
        (
            "null-posting",
            checksum_block(put(tile_bytes(), DATA_OFFSET + 8, b"\xff\xff")),
            -107.0,
            36.0,
            "NullPosting",
            {"longitude_index": 0, "latitude_index": 0},
            "DTED posting lon=0 lat=0 is a null (unknown) elevation",
        ),
    ]
    for name, data, longitude, latitude, kind, fields, message in cases:
        path = tmp_path / f"n36_w107_{name}.dt2"
        path.write_bytes(data)
        assert_elevation_error(path, longitude, latitude, kind, fields, message)

    assert sidereon.DtedTile.from_path(FIXTURE).height_m(36.0, -107.0) == -20


def test_public_tile_lookup_error_details_compare_by_core_value():
    tile = sidereon.DtedTile.from_path(FIXTURE)

    def outside(latitude, longitude):
        with pytest.raises(sidereon.TerrainError) as caught:
            tile.height_m(latitude, longitude)
        assert caught.value.detail.kind == "Outside"
        return caught.value.detail

    positive_zero = outside(0.0, 0.0)
    repeated_positive_zero = outside(0.0, 0.0)
    negative_zero = outside(-0.0, 0.0)

    assert positive_zero == repeated_positive_zero
    # Outside compares binary64 coordinate bits, so the zero sign bit matters.
    assert positive_zero != negative_zero
