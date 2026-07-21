"""Internal bounded decompression primitives shared by Python acquisition APIs."""

from __future__ import annotations

import zlib

_GZIP_MAGIC = b"\x1f\x8b"
_OUTPUT_CHUNK_BYTES = 64 * 1024
_MIN_INPUT_CHUNK_BYTES = 128
_MAX_INPUT_CHUNK_BYTES = 64 * 1024


class GzipIntegrityError(ValueError):
    """A gzip member sequence is malformed, corrupt, or incomplete."""


class GzipSizeLimitError(ValueError):
    """A gzip member sequence expands beyond the caller's cumulative cap."""


def gunzip_members(compressed: bytes, max_bytes: int) -> bytes:
    """Decode a complete RFC 1952 member sequence under one output cap.

    Every member must have a valid header, DEFLATE stream, CRC32, ISIZE, and
    end marker. Bytes after one member are accepted only when they begin the
    next gzip member; arbitrary prefix, inter-member, and trailing data fail.
    The output cap is cumulative across all members.
    """
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")
    if not compressed:
        raise GzipIntegrityError("truncated gzip member (empty input)")

    output = bytearray()
    source = memoryview(compressed)
    cursor = 0
    member_index = 0

    while cursor < len(source):
        member_index += 1
        if (
            len(source) - cursor < len(_GZIP_MAGIC)
            or source[cursor : cursor + len(_GZIP_MAGIC)] != _GZIP_MAGIC
        ):
            if member_index == 1:
                raise GzipIntegrityError("invalid gzip header")
            raise GzipIntegrityError("non-member data after gzip member")

        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        input_chunk_bytes = _MIN_INPUT_CHUNK_BYTES

        try:
            while True:
                available = max_bytes - len(output)
                max_length = min(_OUTPUT_CHUNK_BYTES, available + 1)
                if cursor < len(source):
                    end = min(len(source), cursor + input_chunk_bytes)
                    pending = source[cursor:end]
                    cursor = end
                else:
                    pending = memoryview(b"")

                chunk = decompressor.decompress(pending, max_length)
                output.extend(chunk)

                if len(output) > max_bytes:
                    raise GzipSizeLimitError(
                        f"decompressed output exceeded cap of {max_bytes} bytes"
                    )

                if decompressor.eof:
                    # zlib may read ahead into the next member. Rewind by the
                    # unconsumed suffix length and start the next member from
                    # the original immutable input, avoiding repeated copies
                    # of the full remaining archive.
                    cursor -= len(decompressor.unused_data)
                    break

                if decompressor.unconsumed_tail:
                    cursor -= len(decompressor.unconsumed_tail)
                    continue

                if pending:
                    # Geometric growth keeps each supplied slice bounded while
                    # retaining large chunks for ordinary products. Starting
                    # small prevents a hostile sequence of tiny valid members
                    # from repeatedly feeding the full remaining archive.
                    input_chunk_bytes = min(
                        input_chunk_bytes * 2, _MAX_INPUT_CHUNK_BYTES
                    )
                    continue

                # A full output chunk can coincide with consumption of the
                # final supplied input byte. Give zlib empty-input calls while
                # they make a full chunk of progress; otherwise EOF is absent.
                if len(chunk) == max_length:
                    continue
                raise GzipIntegrityError(
                    f"truncated gzip member {member_index} (no end-of-stream marker)"
                )
        except zlib.error as exc:
            raise GzipIntegrityError(
                f"corrupt gzip member {member_index}: {exc}"
            ) from exc

    return bytes(output)
