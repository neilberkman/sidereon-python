"""Fail-closed Unix-compress transport regressions."""

import base64
from pathlib import Path

import ncompress
import pytest
from sidereon import distribution

FIXTURE = Path(__file__).parent / "fixtures" / "gnss_data" / "igs22376.sp3.Z.b64"
MAXBITS_9_FIXTURE = (
    Path(__file__).parent
    / "fixtures"
    / "gnss_data"
    / "ncompress-current-b9-counter.Z.b64"
)


def test_real_unix_compress_sp3_rejects_incomplete_final_code():
    archive = base64.b64decode(b"".join(FIXTURE.read_bytes().split()), validate=True)

    body = distribution._decompress(archive, "unix_compress", 20_000)
    assert len(body) == 10_161
    assert body.endswith(b"EOF\n")

    # ncompress alone returns a plausible 10,160-byte prefix ending in a bare
    # EOF record here. The compression boundary must reject the partial code.
    with pytest.raises(
        distribution.DecompressionFailure,
        match="invalid or truncated Unix-compress product",
    ):
        distribution._decompress(archive[:-1], "unix_compress", 20_000)


def test_current_ncompress_maxbits_9_archive_round_trips():
    """Lock the fixed 9-bit behavior merged upstream in ncompress PR 34."""
    archive = base64.b64decode(
        b"".join(MAXBITS_9_FIXTURE.read_bytes().split()), validate=True
    )
    expected = bytes(range(256)) + bytes(range(14))

    assert archive[:3] == b"\x1f\x9d\x89"
    assert distribution._decompress(archive, "unix_compress", len(expected)) == expected


@pytest.mark.parametrize(
    "archive",
    [
        b"\x1f\x9d\x10A",
        bytes(bytearray(ncompress.compress(b"A"))[:-1] + b"\x80"),
    ],
    ids=["incomplete-first-code", "nonzero-terminal-padding"],
)
def test_terminal_bit_validation_precedes_decompression(monkeypatch, archive):
    def unexpected_decompress(_source, _output):
        pytest.fail("ncompress must not run for a structurally incomplete archive")

    monkeypatch.setattr(distribution.ncompress, "decompress", unexpected_decompress)
    with pytest.raises(
        distribution.DecompressionFailure,
        match="invalid or truncated Unix-compress product",
    ):
        distribution._decompress(archive, "unix_compress", 1_000)


def test_empty_stream_encodings_are_consistent_with_the_elixir_decoder():
    assert distribution._decompress(b"", "unix_compress", 0) == b""
    assert distribution._decompress(ncompress.compress(b""), "unix_compress", 0) == b""
