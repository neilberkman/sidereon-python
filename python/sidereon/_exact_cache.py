"""Python adapter for the shared native exact-product cache contract."""

from __future__ import annotations

import contextlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Union

from . import _sidereon  # type: ignore[attr-defined]

CONTROL_DIRECTORY = _sidereon._EXACT_CACHE_CONTROL_DIRECTORY


class CacheLockTimeout(OSError):
    """The per-entry cross-process cache lock was not acquired in time."""


class CacheSingleFlightTimeout(CacheLockTimeout):
    """A live single-flight owner did not publish within the bounded wait."""


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


@dataclass(frozen=True)
class SingleFlightOptions:
    """Bounded timing policy for exact-cache single-flight coordination."""

    poll_interval_s: float = 0.05
    heartbeat_interval_s: float = 5.0
    liveness_timeout_s: float = 30.0
    wait_timeout_s: float = 30.0 * 60.0

    def __post_init__(self) -> None:
        values = (
            ("poll_interval_s", self.poll_interval_s),
            ("heartbeat_interval_s", self.heartbeat_interval_s),
            ("liveness_timeout_s", self.liveness_timeout_s),
            ("wait_timeout_s", self.wait_timeout_s),
        )
        for name, value in values:
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.heartbeat_interval_s >= self.liveness_timeout_s:
            raise ValueError(
                "heartbeat_interval_s must be shorter than liveness_timeout_s"
            )

    def _timing_s(self) -> tuple[float, float, float, float]:
        return (
            float(self.poll_interval_s),
            float(self.heartbeat_interval_s),
            float(self.liveness_timeout_s),
            float(self.wait_timeout_s),
        )


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


class ExactCacheOwner:
    """Exclusive right to fetch and publish one single-flight cache miss."""

    def __init__(self, native) -> None:
        self._native = native

    def heartbeat(self) -> None:
        self._native.heartbeat()

    def publish(self, product: bytes, archive: bytes, provenance: bytes) -> CacheFiles:
        return _files(self._native.publish(product, archive, provenance))

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


@contextlib.contextmanager
def open_single_flight(
    path: Path,
    identity,
    source,
    options: Optional[SingleFlightOptions] = None,
) -> Iterator[Union[CacheFiles, ExactCacheOwner]]:
    """Return a verified hit or hold ownership of one cache miss."""
    timing = SingleFlightOptions() if options is None else options
    try:
        hit, native_owner = _sidereon.data_exact_cache_open_single_flight(
            str(path),
            _identity_json(identity),
            source.value,
            timing._timing_s(),
        )
    except TimeoutError as error:
        raise CacheSingleFlightTimeout(str(error)) from None
    if hit is not None:
        if native_owner is not None:  # pragma: no cover - native enum invariant
            raise RuntimeError("single-flight open returned both hit and owner")
        yield _files(hit)
        return
    if native_owner is None:  # pragma: no cover - native enum invariant
        raise RuntimeError("single-flight open returned neither hit nor owner")
    owner = ExactCacheOwner(native_owner)
    try:
        yield owner
    finally:
        owner.close()


def committed_files(path: Path, identity, source) -> Optional[CacheFiles]:
    """Read a committed immutable entry without waiting for a writer lock."""
    try:
        value = _sidereon.data_exact_cache_read(
            str(path), _identity_json(identity), source.value
        )
    except OSError as error:
        raise CacheFormatError(str(error)) from None
    return None if value is None else _files(value)
