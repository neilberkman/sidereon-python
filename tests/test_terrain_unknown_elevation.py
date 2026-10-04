"""DTED null postings are unknown elevations; tiles keep their horizontal datum.

A posting holding the DTED null (all bits set, MIL-PRF-89020B 3.11.3.1) has no
height. A lookup that gives it weight raises `TerrainError` naming the tile and
posting, on the raw DTED reader and on the terrain store alike. A tile whose
DSI record states a datum other than WGS84 still reads as a tile, but no WGS84
query is answered from it and the store refuses to hold it.
"""

import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

TILE = os.path.join(CORE_FIXTURES, "dted", "tiles", "n36_w107_1arc_v3.dt2")
TILE_NAME = "n36_w107_1arc_v3.dt2"
# The committed tiles hold 5 x 5 postings; data records start after the
# 80-byte UHL, 648-byte DSI and 2700-byte ACC.
DATA_OFFSET = 3428
BLOCK_LEN = 12 + 2 * 5


def _tile_bytes():
    with open(TILE, "rb") as handle:
        return bytearray(handle.read())


def _with_null_posting(lon_posting, lat_posting):
    tile = _tile_bytes()
    block = DATA_OFFSET + lon_posting * BLOCK_LEN
    start = block + 8 + 2 * lat_posting
    tile[start : start + 2] = b"\xff\xff"
    checksum = sum(tile[block : block + BLOCK_LEN - 4])
    tile[block + BLOCK_LEN - 4 : block + BLOCK_LEN] = struct.pack(">i", checksum)
    return bytes(tile)


UNKNOWN = {
    "lat_index": 36,
    "lon_index": -107,
    "latitude_posting": 3,
    "longitude_posting": 2,
}


def test_null_postings_are_unknown_elevations_in_the_reader_and_the_store(tmp_path):
    (tmp_path / TILE_NAME).write_bytes(_with_null_posting(2, 3))
    store = sidereon.dted_tree_to_mmap_store(tmp_path)
    # Payload offset 4096; posting (lon 2, lat 3) is element 2 * 5 + 3.
    stored = 4096 + 2 * (2 * 5 + 3)
    assert sidereon.TERRAIN_STORE_NULL_POSTING == -32767
    assert (
        struct.unpack("<h", store[stored : stored + 2])[0]
        == sidereon.TERRAIN_STORE_NULL_POSTING
    )

    mmap = sidereon.MmapTerrain.from_bytes(store)
    dted = sidereon.DtedTerrain(tmp_path)
    nearest = sidereon.DtedLookupOptions(sidereon.DtedInterpolation.NEAREST_POSTING)
    bilinear = sidereon.DtedLookupOptions(sidereon.DtedInterpolation.BILINEAR)
    for longitude, latitude, options, want in [
        (-106.5, 36.75, nearest, None),
        (-106.5, 36.75, bilinear, None),
        (-106.375, 36.625, bilinear, None),
        (-106.625, 36.875, bilinear, None),
        (-106.25, 36.75, bilinear, -5.0),
        (-106.25, 36.75, nearest, -5.0),
    ]:
        if want is not None:
            assert mmap.height_m_with_options(longitude, latitude, options) == want
            assert dted.height_m(latitude, longitude, options) == want
            continue
        for lookup, args in (
            (mmap.height_m_with_options, (longitude, latitude, options)),
            (mmap.orthometric_height_m_with_options, (longitude, latitude, options)),
            (dted.height_m, (latitude, longitude, options)),
        ):
            with pytest.raises(sidereon.TerrainError) as excinfo:
                lookup(*args)
            error = excinfo.value
            assert isinstance(error, ValueError)
            assert isinstance(error, sidereon.SidereonError)
            assert error.point_index is None
            detail = error.detail
            assert (detail.family, detail.kind) == ("Error", "UnknownTerrainElevation")
            assert detail.details() == UNKNOWN
            assert (detail.lat_index, detail.lon_index) == (36, -107)
            assert (detail.latitude_posting, detail.longitude_posting) == (3, 2)

    # A batch names the row it stopped at.
    points = np.asarray([[-106.25, 36.75], [-106.5, 36.75]], dtype=np.float64)
    for batch, prefix in (
        (mmap.height_batch, "terrain store point 1: "),
        (dted.height_batch, "terrain point 1: "),
    ):
        with pytest.raises(sidereon.TerrainError) as excinfo:
            batch(points, nearest)
        assert str(excinfo.value).startswith(prefix)
        assert excinfo.value.point_index == 1
        assert excinfo.value.detail.details() == UNKNOWN

    # The validity form keeps every known height and marks the unknown one.
    for batch in (mmap.height_batch_with_validity, dted.height_batch_with_validity):
        heights, valid = batch(points, nearest)
        assert heights.dtype == np.float64 and valid.dtype == np.bool_
        assert heights[0] == -5.0
        assert np.isnan(heights[1])
        assert valid.tolist() == [True, False]

    # The datum conversion reports the lookup it could not make.
    with pytest.raises(sidereon.TerrainError) as excinfo:
        mmap.ellipsoidal_height_m(-106.375, 36.625)
    assert excinfo.value.detail.kind == "UnknownTerrainElevation"

    # The single-tile reader names the posting.
    tile = sidereon.DtedTile.from_path(tmp_path / TILE_NAME)
    with pytest.raises(sidereon.TerrainError) as excinfo:
        tile.height_m(36.75, -106.5)
    detail = excinfo.value.detail
    assert (detail.family, detail.kind) == ("DtedTileError", "NullPosting")
    assert detail.details() == {"longitude_index": 2, "latitude_index": 3}
    assert (detail.latitude_posting, detail.longitude_posting) == (3, 2)


@pytest.mark.parametrize(
    ("field", "kind", "text", "compatible"),
    [
        (b"WGS84", "Wgs84", None, True),
        (b"     ", "Unstated", None, True),
        (b"WGS72", "Wgs72", None, False),
        (b"NAD27", "Other", "NAD27", False),
    ],
)
def test_horizontal_datum_is_kept_and_refused_for_wgs84_queries(
    tmp_path, field, kind, text, compatible
):
    tile = _tile_bytes()
    # DSI character 145, after the 80-byte UHL.
    tile[224:229] = field
    path = tmp_path / TILE_NAME
    path.write_bytes(bytes(tile))

    read = sidereon.DtedTile.from_path(path)
    datum = read.horizontal_datum
    assert (datum.kind, datum.text, datum.is_wgs84_compatible) == (
        kind,
        text,
        compatible,
    )
    assert read.height_m(36.0, -107.0) == -20.0

    terrain = sidereon.DtedTerrain(tmp_path)
    if compatible:
        assert np.isfinite(terrain.height_m(36.125, -106.875))
        return
    with pytest.raises(sidereon.TerrainError) as excinfo:
        terrain.height_m(36.125, -106.875)
    detail = excinfo.value.detail
    assert detail.kind == "NonWgs84TerrainTile"
    assert (detail.lat_index, detail.lon_index) == (36, -107)
    assert detail.datum == datum

    # The store records no datum, so it refuses the tile rather than holding
    # it as WGS84.
    with pytest.raises(ValueError, match="NonWgs84Tile"):
        sidereon.dted_tree_to_mmap_store(tmp_path)


def test_terrain_error_defaults_carry_no_core_context():
    error = sidereon.TerrainError("hand built")
    assert error.detail is None
    assert error.point_index is None
