"""Private bounded-input helpers shared by Python download paths."""

from __future__ import annotations

_STREAM_CHUNK_BYTES = 64 * 1024


def append_bounded(buffer: bytearray, chunk: bytes, limit: int) -> bool:
    """Retain at most ``limit + 1`` bytes and report whether the cap was crossed.

    Keeping the one-byte probe makes an exact-size body distinguishable from an
    oversized body without duplicating an arbitrarily large transport-provided
    chunk in Sidereon's accumulator.
    """
    if limit < 0:
        raise ValueError("limit must be non-negative")

    probe_limit = limit + 1
    remaining = max(0, probe_limit - len(buffer))
    if remaining:
        buffer.extend(memoryview(chunk)[:remaining])
    return len(buffer) > limit
