"""Python adapter for the shared native exact-product cache contract."""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from . import _sidereon  # type: ignore[attr-defined]

CONTROL_DIRECTORY = _sidereon._EXACT_CACHE_CONTROL_DIRECTORY


class CacheLockTimeout(OSError):
    """The per-entry cross-process cache lock was not acquired in time."""


class CacheFormatError(OSError):
    """The shared atomic cache commit or immutable entry is invalid."""


@dataclass(frozen=True)
class CacheFiles:
    """Paths and authenticated bytes from one immutable transaction."""

    product: Path
    archive: Path
    provenance: Path
    entry_id: str
    product_bytes: bytes
    archive_bytes: bytes
    provenance_bytes: bytes


def _identity_json(identity) -> str:
    return json.dumps(identity.to_dict(), sort_keys=True, separators=(",", ":"))


def validate_identity(identity) -> None:
    """Validate a complete product identity with the shared Rust catalog."""
    _sidereon.data_validate_product_identity(_identity_json(identity))


def _files(value) -> CacheFiles:
    (
        product,
        archive,
        provenance,
        entry_id,
        product_bytes,
        archive_bytes,
        provenance_bytes,
    ) = value
    return CacheFiles(
        product=Path(product),
        archive=Path(archive),
        provenance=Path(provenance),
        entry_id=entry_id,
        product_bytes=bytes(product_bytes),
        archive_bytes=bytes(archive_bytes),
        provenance_bytes=bytes(provenance_bytes),
    )


class ExactCache:
    """Lock-owning adapter over the common Rust cache implementation."""

    def __init__(self, path: Path, identity, source, timeout_s: float) -> None:
        if (
            not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or timeout_s < 0
        ):
            raise ValueError("cache lock timeout must be finite and non-negative")
        try:
            self._native = _sidereon._ExactProductCache(
                str(path), _identity_json(identity), source.value, float(timeout_s)
            )
        except TimeoutError as error:
            raise CacheLockTimeout(str(error)) from None

    def committed_files(self) -> Optional[CacheFiles]:
        try:
            value = self._native.read()
        except OSError as error:
            raise CacheFormatError(str(error)) from None
        return None if value is None else _files(value)

    def publish(self, product: bytes, archive: bytes, provenance: bytes) -> CacheFiles:
        return _files(self._native.publish(product, archive, provenance))

    def cleanup_abandoned(self) -> None:
        self._native.cleanup_abandoned()

    def close(self) -> None:
        self._native.close()


@contextlib.contextmanager
def entry_lock(path: Path, identity, source, timeout_s: float) -> Iterator[ExactCache]:
    """Hold the common bounded cross-process lock for one exact cache entry."""
    cache = ExactCache(path, identity, source, timeout_s)
    try:
        yield cache
    finally:
        cache.close()


def committed_files(path: Path, identity, source) -> Optional[CacheFiles]:
    """Read a committed immutable entry without waiting for a writer lock."""
    try:
        value = _sidereon.data_exact_cache_read(
            str(path), _identity_json(identity), source.value
        )
    except OSError as error:
        raise CacheFormatError(str(error)) from None
    return None if value is None else _files(value)
