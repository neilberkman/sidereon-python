"""Crash-consistent storage for one exact-product cache entry.

The acquisition layer supplies already validated product, archive, and
provenance bytes.  This module publishes them as an immutable transaction and
makes one small commit record the only reader-visible transition.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterator, Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - the supported contract is POSIX
    fcntl = None  # type: ignore[assignment]


_ENTRY_RE = re.compile(r"^[0-9a-f]{32}$")
_CONTROL_DIRECTORY = ".sidereon-cache-v2"
_LOCK_FILENAME = ".sidereon-cache.lock"
_MARKER_FILENAME = "current.json"


class CacheLockTimeout(OSError):
    """The per-entry cross-process cache lock was not acquired in time."""


class CacheFormatError(OSError):
    """The atomic cache commit record or immutable entry is malformed."""


@dataclass(frozen=True)
class CacheFiles:
    """Paths belonging to one immutable committed transaction."""

    product: Path
    archive: Path
    provenance: Path
    entry_id: str
    provenance_bytes: Optional[bytes] = None


def _control_directory(path: Path) -> Path:
    return path.parent / _CONTROL_DIRECTORY


def _marker_path(path: Path) -> Path:
    return _control_directory(path) / _MARKER_FILENAME


def _entry_files(path: Path, entry_id: str) -> CacheFiles:
    entry = _control_directory(path) / "entries" / entry_id
    return CacheFiles(
        product=entry / path.name,
        archive=entry / f"{path.name}.archive",
        provenance=entry / f"{path.name}.provenance.json",
        entry_id=entry_id,
    )


@contextlib.contextmanager
def entry_lock(path: Path, timeout_s: float) -> Iterator[None]:
    """Hold the interoperable POSIX advisory lock for an exact cache entry."""
    if (
        not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s < 0
    ):
        raise ValueError("cache lock timeout must be non-negative")
    if fcntl is None:  # pragma: no cover - Linux and macOS provide fcntl
        raise OSError("cross-process exact-product cache locking is unavailable")
    _mkdirs_durable(path.parent)
    lock_path = path.parent / _LOCK_FILENAME
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fsync(fd)
        _sync_directory(path.parent)
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CacheLockTimeout(
                        f"timed out waiting for exact-product cache lock {path.name}"
                    ) from None
                time.sleep(min(0.01, remaining))
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def committed_files(path: Path) -> Optional[CacheFiles]:
    """Resolve the current immutable transaction, if one was committed."""
    marker_path = _marker_path(path)
    try:
        marker_bytes = marker_path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise CacheFormatError(f"cannot read cache commit for {path.name}") from error
    try:
        marker = json.loads(marker_bytes)
        entry_id = marker["entry"]
        provenance_sha256 = marker["provenance_sha256"]
        valid = (
            marker["schema_version"] == 2
            and isinstance(entry_id, str)
            and _ENTRY_RE.fullmatch(entry_id) is not None
            and isinstance(provenance_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", provenance_sha256) is not None
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        valid = False
    if not valid:
        raise CacheFormatError(f"invalid cache commit for {path.name}")
    files = _entry_files(path, entry_id)
    try:
        provenance = files.provenance.read_bytes()
    except OSError as error:
        raise CacheFormatError(
            f"cannot read committed provenance for {path.name}"
        ) from error
    if hashlib.sha256(provenance).hexdigest() != provenance_sha256:
        raise CacheFormatError(f"cache commit digest mismatch for {path.name}")
    return replace(files, provenance_bytes=provenance)


def publish(
    path: Path, product: bytes, archive: bytes, provenance: bytes
) -> CacheFiles:
    """Durably publish one immutable entry and atomically replace its marker."""
    control = _control_directory(path)
    entries = control / "entries"
    _mkdirs_durable(entries)
    entry_id = secrets.token_hex(16)
    files = _entry_files(path, entry_id)
    entry_directory = files.product.parent
    entry_directory.mkdir(mode=0o700)
    _sync_directory(entries)
    marker_published = False
    marker_temp = control / f".{_MARKER_FILENAME}.{entry_id}.tmp"
    try:
        _write_exclusive(files.product, product)
        _hit_failpoint("after_payload")
        _write_exclusive(files.archive, archive)
        _hit_failpoint("after_archive")
        _write_exclusive(files.provenance, provenance)
        _hit_failpoint("after_metadata")
        _sync_directory(entry_directory)
        _sync_directory(entries)
        _hit_failpoint("after_entry_sync")

        marker = json.dumps(
            {
                "schema_version": 2,
                "entry": entry_id,
                "provenance_sha256": hashlib.sha256(provenance).hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _write_exclusive(marker_temp, marker)
        _hit_failpoint("after_marker_write")
        os.replace(marker_temp, _marker_path(path))
        marker_published = True
        _hit_failpoint("after_marker_rename")
        _sync_directory(control)
        _hit_failpoint("after_commit_sync")
        return files
    finally:
        if not marker_published:
            try:
                marker_temp.unlink()
            except OSError:
                pass
            try:
                shutil.rmtree(entry_directory)
            except OSError:
                pass


def cleanup_abandoned(path: Path) -> None:
    """Remove uncommitted artifacts while the caller holds the entry lock."""
    control = _control_directory(path)
    entries = control / "entries"
    if not control.exists():
        return
    current: Optional[str] = None
    marker = _marker_path(path)
    try:
        value = json.loads(marker.read_bytes())
        candidate = value.get("entry")
        if isinstance(candidate, str) and _ENTRY_RE.fullmatch(candidate):
            current = candidate
    except FileNotFoundError:
        pass
    except (OSError, ValueError, TypeError):
        # Preserve all entries when an existing marker cannot be interpreted.
        return
    try:
        children = tuple(entries.iterdir())
    except FileNotFoundError:
        children = ()
    for child in children:
        if child.name != current:
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError:
                pass
    try:
        marker_temps = tuple(control.glob(f".{_MARKER_FILENAME}.*.tmp"))
    except OSError:
        marker_temps = ()
    for item in marker_temps:
        try:
            item.unlink()
        except OSError:
            pass


def _mkdirs_durable(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        _sync_directory(directory.parent)


def _write_exclusive(path: Path, content: bytes) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _hit_failpoint(_step: str) -> None:
    """Test seam replaced only inside crash-boundary subprocesses."""
