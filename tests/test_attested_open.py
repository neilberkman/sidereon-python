"""Attested path-open parity for large binary artifact readers."""

import struct
from pathlib import Path

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES

TERRAIN_DATA_OFFSET_OFFSET = 24
INTERPOLANT_INDEX_OFFSET_OFFSET = 16
INTERPOLANT_HEADER_CHECKSUM_OFFSET = 40
INTERPOLANT_POS_KX_OFFSET_OFFSET = 24


def _u64_at(data, offset):
    return struct.unpack_from("<Q", data, offset)[0]


@pytest.fixture(scope="module")
def terrain_store_bytes():
    root = Path(CORE_FIXTURES) / "dted" / "tiles"
    return sidereon.dted_tree_to_mmap_store(root)


@pytest.fixture(scope="module")
def precise_artifact_bytes():
    path = Path(FIXTURES) / "sp3" / "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    return sidereon.build_precise_interpolant_artifact_bytes(sidereon.load_sp3(path))


def test_terrain_attested_open_skips_payload_hash_and_verify_detects_corruption(
    tmp_path, terrain_store_bytes
):
    corrupt = bytearray(terrain_store_bytes)
    data_offset = _u64_at(corrupt, TERRAIN_DATA_OFFSET_OFFSET)
    corrupt[data_offset + 1] ^= 1
    path = tmp_path / "corrupt-terrain.tmm"
    path.write_bytes(corrupt)

    with pytest.raises(ValueError, match="Checksum"):
        sidereon.MmapTerrain.from_path(path)

    claim = sidereon.terrain_store_checksum64(terrain_store_bytes)
    attested = sidereon.MmapTerrain.from_path_attested(path, claim)
    assert attested.digest_provenance() == "attested"
    assert attested.checksum64() == claim

    with pytest.raises(ValueError, match="Checksum"):
        attested.verify()
    assert attested.digest_provenance() == "attested"


def test_precise_attested_open_skips_payload_hash_and_verify_detects_corruption(
    tmp_path, precise_artifact_bytes
):
    corrupt = bytearray(precise_artifact_bytes)
    declared = _u64_at(corrupt, INTERPOLANT_HEADER_CHECKSUM_OFFSET)
    index_offset = _u64_at(corrupt, INTERPOLANT_INDEX_OFFSET_OFFSET)
    pos_kx_offset = _u64_at(corrupt, index_offset + INTERPOLANT_POS_KX_OFFSET_OFFSET)
    corrupt[pos_kx_offset + 1] ^= 1
    path = tmp_path / "corrupt-interpolant.spi"
    path.write_bytes(corrupt)

    with pytest.raises(sidereon.PreciseInterpolantArtifactCorruptError):
        sidereon.PreciseInterpolantArtifact.from_path(path)

    attested = sidereon.PreciseInterpolantArtifact.from_path_attested(path, declared)
    assert attested.digest_provenance == "attested"
    assert attested.checksum64 == declared

    with pytest.raises(sidereon.PreciseInterpolantArtifactCorruptError):
        attested.verify()
    assert attested.digest_provenance == "attested"


def test_precise_attested_open_rejects_claim_that_differs_from_header(
    tmp_path, precise_artifact_bytes
):
    declared = _u64_at(precise_artifact_bytes, INTERPOLANT_HEADER_CHECKSUM_OFFSET)
    claimed = declared ^ 1
    path = tmp_path / "pristine-interpolant.spi"
    path.write_bytes(precise_artifact_bytes)

    with pytest.raises(
        sidereon.PreciseInterpolantArtifactError,
        match=f"{claimed:#x}.*{declared:#x}",
    ):
        sidereon.PreciseInterpolantArtifact.from_path_attested(path, claimed)


def test_pristine_terrain_attested_and_verified_queries_are_identical(
    tmp_path, terrain_store_bytes
):
    path = tmp_path / "pristine-terrain.tmm"
    path.write_bytes(terrain_store_bytes)
    claim = sidereon.terrain_store_checksum64(terrain_store_bytes)

    verified = sidereon.MmapTerrain.from_path(path)
    attested = sidereon.MmapTerrain.from_path_attested(path, claim)
    assert verified.digest_provenance() == "verified"
    assert attested.digest_provenance() == "attested"
    assert attested.checksum64() == claim
    assert attested.height_m(-105.5, 36.5) == verified.height_m(-105.5, 36.5)

    attested.verify()
    assert attested.digest_provenance() == "verified"
    assert attested.checksum64() == claim


def test_pristine_precise_attested_and_verified_queries_are_identical(
    tmp_path, precise_artifact_bytes
):
    path = tmp_path / "pristine-interpolant.spi"
    path.write_bytes(precise_artifact_bytes)
    claim = _u64_at(precise_artifact_bytes, INTERPOLANT_HEADER_CHECKSUM_OFFSET)

    verified = sidereon.PreciseInterpolantArtifact.from_path(path)
    attested = sidereon.PreciseInterpolantArtifact.from_path_attested(path, claim)
    assert verified.digest_provenance == "verified"
    assert attested.digest_provenance == "attested"
    assert attested.checksum64 == claim

    expected = verified.position_at_j2000_seconds("G01", 830_818_800.0)
    found = attested.position_at_j2000_seconds("G01", 830_818_800.0)
    assert np.array_equal(found.position_m, expected.position_m)
    assert found.clock_s == expected.clock_s

    attested.verify()
    assert attested.digest_provenance == "verified"
    assert attested.checksum64 == claim


@pytest.mark.parametrize("claim", [-1, 2**64, 1.5, "1", None])
def test_attested_open_rejects_malformed_checksum_claims(
    tmp_path, terrain_store_bytes, precise_artifact_bytes, claim
):
    terrain_path = tmp_path / "terrain.tmm"
    terrain_path.write_bytes(terrain_store_bytes)
    precise_path = tmp_path / "interpolant.spi"
    precise_path.write_bytes(precise_artifact_bytes)

    for artifact_type, path in (
        (sidereon.MmapTerrain, terrain_path),
        (sidereon.PreciseInterpolantArtifact, precise_path),
    ):
        with pytest.raises(ValueError, match="claimed_checksum64"):
            artifact_type.from_path_attested(path, claim)
