"""Typed detail payloads for space-weather and terrain refusals."""

import struct
from pathlib import Path

import pytest
import sidereon
from _helpers import CORE_FIXTURES

WEATHER_CSV = (
    b"DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,AP5,"
    b"AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,"
    b"F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n"
    b"2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,"
    b"24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n"
    b"2024-05-10,2556,2,40,50,60,70,67,57,47,37,428,27,48,80,132,111,67,39,22,"
    b"66,1.8,7,121,190.2,187.1,OBS,151.2,150.9,148.0,147.6\n"
    b"2024-05-11,2556,3,33,30,27,23,20,17,13,10,173,18,15,12,9,7,6,5,4,"
    b"10,0.8,3,119,176.3,173.0,OBS,152.3,151.1,149.0,148.2\n"
    b"2024-06-01,2557,24,,,,,,,,,,,,,,,,,,,,,118,171.0,168.0,PRM,153.0,152.0,"
    b"150.0,149.0\n"
)


def test_space_weather_failures_retain_variant_fields_and_epoch_bits():
    table = sidereon.load_space_weather(WEATHER_CSV)
    coverage = table.coverage()

    with pytest.raises(sidereon.SpaceWeatherError) as before:
        table.sample_at(coverage.first_j2000_s - 0.5)
    assert before.value.detail.kind == "BeforeCoverage"
    assert before.value.detail.details() == {
        "requested_j2000_s": coverage.first_j2000_s - 0.5,
        "first_j2000_s": coverage.first_j2000_s,
    }

    with pytest.raises(sidereon.SpaceWeatherError) as after:
        table.sample_at(coverage.end_j2000_s)
    assert after.value.detail.kind == "AfterCoverage"
    assert after.value.detail.details() == {
        "requested_j2000_s": coverage.end_j2000_s,
        "end_j2000_s": coverage.end_j2000_s,
    }

    missing_epoch = coverage.first_j2000_s + 3.5 * 86400.0
    with pytest.raises(sidereon.SpaceWeatherError) as missing:
        table.sample_at(missing_epoch)
    assert missing.value.detail.kind == "MissingData"
    assert missing.value.detail.details() == {
        "year": 2024,
        "month": 5,
        "day": 12,
        "field": "record",
    }

    monthly_epoch = coverage.end_j2000_s - 0.5 * 86400.0
    with pytest.raises(sidereon.SpaceWeatherError) as rejected:
        table.sample_at(monthly_epoch)
    assert rejected.value.detail.kind == "RejectedByPolicy"
    assert rejected.value.detail.details() == {
        "class": sidereon.ObservationClass.MONTHLY_PREDICTED,
        "year": 2024,
        "month": 6,
        "day": 1,
    }

    bad_epoch = float("nan")
    with pytest.raises(sidereon.SpaceWeatherError) as invalid:
        table.sample_at(bad_epoch)
    assert invalid.value.detail.kind == "InvalidEpoch"
    assert (
        invalid.value.detail.details()["epoch_j2000_s_bits"]
        == struct.unpack(">Q", struct.pack(">d", bad_epoch))[0]
    )

    with pytest.raises(sidereon.SpaceWeatherError) as malformed:
        sidereon.load_space_weather(WEATHER_CSV.split(b"\n", 1)[0] + b"\n")
    detail = malformed.value.detail
    assert detail.kind == "Malformed"
    fields = detail.details()
    assert fields["line"] == 1
    assert fields["reason"]
    fields["reason"] = "caller mutation"
    assert detail.details()["reason"] != "caller mutation"


def test_terrain_store_parse_and_tile_id_mismatch_keep_payloads():
    missing_path = Path(CORE_FIXTURES) / "dted" / "missing.dt2"
    with pytest.raises(ValueError) as io_error:
        sidereon.MmapTerrain.from_path(missing_path)
    io_detail = io_error.value.detail
    assert io_detail.kind == "Io"
    assert io_detail.path == str(missing_path)
    assert io_detail.details()["path"] == str(missing_path)
    assert io_detail.details()["message"]

    with pytest.raises(ValueError) as parse_error:
        sidereon.MmapTerrain.from_bytes(b"not a terrain store")
    detail = parse_error.value.detail
    assert isinstance(detail, sidereon.TerrainStoreError)
    assert detail.kind == "Parse"
    assert detail.details()["reason"]

    tiles = Path(CORE_FIXTURES) / "dted" / "tiles"
    source = tiles / "n36_w107_1arc_v3.dt2"
    wrong_id = sidereon.DtedTileListEntry.from_indices(35, -107, str(source))
    with pytest.raises(ValueError) as mismatch:
        sidereon.dted_tile_list_to_mmap_store([wrong_id])
    mismatch_detail = mismatch.value.detail
    assert mismatch_detail.kind == "TileIdMismatch"
    fields = mismatch_detail.details()
    assert fields["path"] == str(source)
    assert (fields["expected"].lat_index, fields["expected"].lon_index) == (35, -107)
    assert (fields["found"].lat_index, fields["found"].lon_index) == (36, -107)


def test_terrain_store_version_domain_bounds_and_checksums_keep_fields(tmp_path):
    tiles = Path(CORE_FIXTURES) / "dted" / "tiles"
    valid = bytearray(sidereon.dted_tree_to_mmap_store(tiles))
    index_offset = struct.unpack_from("<Q", valid, 16)[0]

    unsupported = bytearray(valid)
    struct.pack_into("<H", unsupported, 8, 2)
    with pytest.raises(ValueError) as version:
        sidereon.MmapTerrain.from_bytes(bytes(unsupported))
    assert version.value.detail.kind == "UnsupportedVersion"
    assert version.value.detail.details() == {"version": 2}

    out_of_range = bytearray(valid)
    struct.pack_into("<i", out_of_range, index_offset, 90)
    with pytest.raises(ValueError) as domain:
        sidereon.MmapTerrain.from_bytes(bytes(out_of_range))
    assert domain.value.detail.kind == "TileIdOutOfRange"
    assert domain.value.detail.details() == {"lat_index": 90, "lon_index": -107}

    bad_bounds = bytearray(valid)
    struct.pack_into("<d", bad_bounds, index_offset + 40, 36.25)
    with pytest.raises(ValueError) as bounds:
        sidereon.MmapTerrain.from_bytes(bytes(bad_bounds))
    assert bounds.value.detail.kind == "TileBoundsMismatch"
    assert bounds.value.detail.details() == {
        "lat_index": 36,
        "lon_index": -107,
        "field": "min_latitude_deg",
    }

    bad_payload_checksum = bytearray(valid)
    checksum_offset = index_offset + 32
    expected = struct.unpack_from("<Q", valid, checksum_offset)[0]
    struct.pack_into("<Q", bad_payload_checksum, checksum_offset, expected ^ 1)
    with pytest.raises(ValueError) as checksum:
        sidereon.MmapTerrain.from_bytes(bytes(bad_payload_checksum))
    fields = checksum.value.detail.details()
    assert checksum.value.detail.kind == "Checksum"
    assert (fields["lat_index"], fields["lon_index"]) == (36, -107)
    assert fields["expected"] == expected ^ 1
    assert fields["found"] == expected

    path = tmp_path / "attested.tmm"
    path.write_bytes(valid)
    claimed = sidereon.terrain_store_checksum64(bytes(valid)) ^ 1
    reader = sidereon.MmapTerrain.from_path_attested(path, claimed)
    with pytest.raises(ValueError) as attested:
        reader.verify()
    assert attested.value.detail.kind == "AttestedChecksumMismatch"
    assert attested.value.detail.details() == {
        "expected": claimed,
        "found": sidereon.terrain_store_checksum64(bytes(valid)),
    }

    unsupported_datum = bytearray(valid)
    unsupported_datum[10] = 255
    with pytest.raises(ValueError) as datum_tag:
        sidereon.MmapTerrain.from_bytes(bytes(unsupported_datum))
    assert datum_tag.value.detail.kind == "UnsupportedDatum"
    assert datum_tag.value.detail.details() == {"tag": 255}


def test_terrain_store_duplicate_non_wgs84_and_nested_tile_errors(tmp_path):
    tiles = Path(CORE_FIXTURES) / "dted" / "tiles"
    source = tiles / "n36_w107_1arc_v3.dt2"
    entry = sidereon.DtedTileListEntry.from_indices(36, -107, str(source))
    with pytest.raises(ValueError) as duplicate:
        sidereon.dted_tile_list_to_mmap_store([entry, entry])
    assert duplicate.value.detail.kind == "DuplicateTile"
    assert duplicate.value.detail.details() == {"lat_index": 36, "lon_index": -107}

    non_wgs84_path = tmp_path / "n36_w107_1arc_v3.dt2"
    non_wgs84 = bytearray(source.read_bytes())
    non_wgs84[224:229] = b"WGS72"
    non_wgs84_path.write_bytes(non_wgs84)
    non_wgs84_entry = sidereon.DtedTileListEntry.from_indices(
        36, -107, str(non_wgs84_path)
    )
    with pytest.raises(ValueError) as datum:
        sidereon.dted_tile_list_to_mmap_store([non_wgs84_entry])
    assert datum.value.detail.kind == "NonWgs84Tile"
    assert datum.value.detail.path == str(non_wgs84_path)
    assert datum.value.detail.details() == {
        "path": str(non_wgs84_path),
        "datum_kind": "Wgs72",
    }

    invalid_path = tmp_path / "invalid.dt2"
    invalid_path.write_bytes(b"x" * 4000)
    invalid_entry = sidereon.DtedTileListEntry.from_indices(36, -107, str(invalid_path))
    with pytest.raises(ValueError) as nested:
        sidereon.dted_tile_list_to_mmap_store([invalid_entry])
    assert nested.value.detail.kind == "Tile"
    fields = nested.value.detail.details()
    assert fields["path"] == str(invalid_path)
    assert fields["error"]["kind"] == "MissingUhl1"
    assert fields["error"]["path"] == str(invalid_path)


def test_terrain_datum_geoid_error_is_nested_and_lossless(tmp_path):
    with pytest.raises(ValueError) as invalid_grid:
        sidereon.Egm96FifteenMinuteGeoid.from_ww15mgh_dac_bytes(b"")
    outer = invalid_grid.value.detail
    assert isinstance(outer, sidereon.TerrainDatumError)
    assert outer.kind == "Geoid"
    fields = outer.details()
    assert fields["error_kind"] == "Parse"
    assert fields["error"]["reason"]
    fields["error"]["reason"] = "caller mutation"
    assert outer.details()["error"]["reason"] != "caller mutation"

    missing_path = tmp_path / "missing-WW15MGH.DAC"
    with pytest.raises(ValueError) as missing:
        sidereon.Egm96FifteenMinuteGeoid.from_ww15mgh_dac_path(missing_path)
    missing_detail = missing.value.detail
    assert missing_detail.kind == "MissingEgm96Dac"
    assert missing_detail.path == str(missing_path)
    assert missing_detail.details()["path"] == str(missing_path)
    assert missing_detail.details()["remediation"] == missing_detail.remediation

    tiles = Path(CORE_FIXTURES) / "dted" / "tiles"
    terrain = sidereon.MmapTerrain.from_bytes(sidereon.dted_tree_to_mmap_store(tiles))
    with pytest.raises(sidereon.TerrainError) as lookup:
        terrain.ellipsoidal_height_m(0.0, 0.0)
    assert str(lookup.value) == (
        "Terrain: terrain lookup failed: missing terrain tile (0,0)"
    )
    assert lookup.value.detail.kind == "MissingTerrainTile"
    assert isinstance(lookup.value.datum_error, sidereon.TerrainDatumError)
    assert lookup.value.datum_error.kind == "Terrain"
    assert lookup.value.datum_error.details()["error_kind"] == "MissingTerrainTile"
    assert lookup.value.datum_error.details()["error"] == {
        "lat_index": 0,
        "lon_index": 0,
    }

    with pytest.raises(sidereon.TerrainError) as invalid_input:
        terrain.ellipsoidal_height_m(float("nan"), 0.0)
    assert invalid_input.value.detail.kind == "InvalidInput"
    assert invalid_input.value.datum_error.details() == {
        "error_kind": "InvalidInput",
        "error": {"reason": "longitude_deg must be finite"},
    }
    assert str(invalid_input.value).endswith(
        "invalid input: longitude_deg must be finite"
    )
