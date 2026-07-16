"""Atomic exact-product cache transactions.

The cache binds a complete distributor-independent product identity, the
resolved distribution source, and SHA-256 digests and lengths for validated
product, distributor archive, and provenance bytes. Native publication uses a
bounded cross-process lock and one atomic reader-visible commit marker.

Transport and product-format validation remain caller responsibilities. A
cache hit returns authenticated bytes; callers must still parse the product
and interpret the authenticated provenance before use.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Iterator, Optional

from ._exact_cache import (
    CONTROL_DIRECTORY,
    CacheFiles,
    CacheFormatError,
    CacheLockTimeout,
    committed_files,
)
from ._exact_cache import (
    ExactCache as _ExactCache,
)


class ExactProductCache(_ExactCache):
    """One lock-owning native exact-product cache transaction."""

    def __init__(self, path: Path, identity, source, timeout_s: float = 30.0) -> None:
        super().__init__(path, identity, source, timeout_s)


@contextlib.contextmanager
def entry_lock(
    path: Path, identity, source, timeout_s: float = 30.0
) -> Iterator[ExactProductCache]:
    """Hold the bounded cross-process writer lock for one exact cache entry."""
    cache = ExactProductCache(path, identity, source, timeout_s)
    try:
        yield cache
    finally:
        cache.close()


def read(path: Path, identity, source) -> Optional[CacheFiles]:
    """Read a complete digest-verified entry without taking the writer lock."""
    return committed_files(path, identity, source)


__all__ = [
    "CONTROL_DIRECTORY",
    "CacheFiles",
    "CacheFormatError",
    "CacheLockTimeout",
    "ExactProductCache",
    "entry_lock",
    "read",
]
