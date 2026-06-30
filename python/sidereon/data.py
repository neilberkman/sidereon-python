"""Optional fetch-and-cache layer for GNSS products (IONEX, SP3).

``sidereon.data`` downloads, decompresses, checksums, and records provenance for
precise- and ionosphere-product files, then hands back a local file path (or a
parsed handle). It is one-directional: the numerical layers never call into this
module, so a solve never depends on network availability. You fetch once, then
point the solver at the cached file.

This is the Python counterpart of the Elixir ``Sidereon.GNSS.Data`` module,
scoped to the IONEX (CODE rapid + predicted) and merged-SP3 capabilities. The
product tokens, archive URLs, filename convention, cache layout, and offline
semantics are taken verbatim from that reference.

Quick start::

    import sidereon.data as data

    # Newest available predicted ionosphere map, parsed:
    ionex = data.fetch_ionex("cod_prd1", date.today())

    # Merged current-day SP3 from several centers + the merge audit report:
    sp3, report = data.fetch_merged_sp3(date.today(), ["igs_ult", "gfz_ult"])

Cache-first ``fetch`` returns a local file path; a verified cache hit returns
with no network. Pass ``offline=True`` to forbid all network access (a verified
cache hit is returned, a miss raises :class:`OfflineCacheMiss`).

Failures raise a typed exception from the :class:`DataError` hierarchy rather
than returning sentinels.
"""

from __future__ import annotations

import datetime as _dt
import gzip as _gzip
import hashlib as _hashlib
import json as _json
import os as _os
import zlib as _zlib
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union
from urllib.parse import urlsplit

import httpx
import platformdirs

import sidereon

__all__ = [
    "DataError",
    "UnknownCenter",
    "UnsupportedProduct",
    "OfflineCacheMiss",
    "FileNotFoundOnArchive",
    "HttpStatusError",
    "RedirectNotAllowed",
    "NetworkError",
    "ChecksumMismatch",
    "DownloadSizeExceeded",
    "DecompressError",
    "CacheNotWritable",
    "IncompatibleSources",
    "NoProducts",
    "Product",
    "MergeReport",
    "default_cache_dir",
    "centers",
    "content_types",
    "gps_week",
    "day_of_year",
    "canonical_filename",
    "archive_url",
    "mgex_ionex",
    "rapid_ionex",
    "predicted_ionex",
    "ops_ultra_sp3",
    "mgex_sp3",
    "product",
    "fetch",
    "fetch_ionex",
    "fetch_merged_sp3",
    "fetch_merged_sp3_file",
    "write_sp3",
]


# --- errors --------------------------------------------------------------


class DataError(Exception):
    """Base class for every fetch/cache failure in :mod:`sidereon.data`."""


class UnknownCenter(DataError):
    """The analysis-center code is not in the catalog."""


class UnsupportedProduct(DataError):
    """The requested center/content/sample combination is not buildable."""


class OfflineCacheMiss(DataError):
    """``offline=True`` and the product is not present in the cache."""


class FileNotFoundOnArchive(DataError):
    """The archive returned 404 for the product URL."""


class HttpStatusError(DataError):
    """A non-2xx, non-404 HTTP status was returned."""

    def __init__(self, status: int, url: str) -> None:
        self.status = status
        self.url = url
        super().__init__(f"HTTP {status} for {url}")


class RedirectNotAllowed(DataError):
    """A 3xx redirect was refused (redirects are not followed)."""

    def __init__(self, status: int, url: str) -> None:
        self.status = status
        self.url = url
        super().__init__(f"redirect ({status}) refused for {url}")


class NetworkError(DataError):
    """A connection, timeout, or DNS failure reaching the archive."""


class ChecksumMismatch(DataError):
    """A decompressed file failed SHA-256 verification."""

    def __init__(self, expected: str, got: str) -> None:
        self.expected = expected
        self.got = got
        super().__init__(f"checksum mismatch: expected {expected}, got {got}")


class DownloadSizeExceeded(DataError):
    """The compressed payload exceeded the buffered-bytes cap."""


class DecompressError(DataError):
    """The gzip payload was corrupt or exceeded the decompression cap."""


class CacheNotWritable(DataError):
    """The cache directory could not be created or written."""


class IncompatibleSources(DataError):
    """Fetched SP3 sources exist but could not be merged into one frame."""

    def __init__(self, centers: Sequence[str], reason: object) -> None:
        self.centers = list(centers)
        self.reason = reason
        super().__init__(f"incompatible SP3 sources {self.centers}: {reason}")


class NoProducts(DataError):
    """No center contributed a product to a merged SP3 fetch."""

    def __init__(self, reasons: Sequence["AbsentCenter"]) -> None:
        self.reasons = list(reasons)
        detail = ", ".join(f"{r.center}={r.reason}" for r in self.reasons)
        super().__init__(f"no SP3 products available ({detail})")


# --- catalog -------------------------------------------------------------

# Mirrors Sidereon.GNSS.Data.Catalog @centers for the scoped centers. Each entry
# carries the archive protocol, host, root URL, per-content product token,
# directory layout, default sampling, and (for the AIUB recent-products root)
# the uncompressed flag where the archive serves plain files.
_CENTERS: dict[str, dict] = {
    # IONEX low-latency CODE maps on the AIUB /CODE recent-products root.
    "cod_rap": {
        "protocol": "http",
        "host": "ftp.aiub.unibe.ch",
        "root": "http://ftp.aiub.unibe.ch",
        "tokens": {"ionex": "COD0OPSRAP"},
        "layouts": {"ionex": "aiub_code_root"},
        "spans": {"ionex": "01D"},
        "samples": {"ionex": "01H"},
    },
    "cod_prd1": {
        "protocol": "http",
        "host": "ftp.aiub.unibe.ch",
        "root": "http://ftp.aiub.unibe.ch",
        "tokens": {"ionex": "COD0OPSPRD"},
        "layouts": {"ionex": "aiub_code_root"},
        "spans": {"ionex": "01D"},
        "samples": {"ionex": "01H"},
    },
    "cod_prd2": {
        "protocol": "http",
        "host": "ftp.aiub.unibe.ch",
        "root": "http://ftp.aiub.unibe.ch",
        "tokens": {"ionex": "COD0OPSPRD"},
        "layouts": {"ionex": "aiub_code_root"},
        "spans": {"ionex": "01D"},
        "samples": {"ionex": "01H"},
    },
    "esa": {
        "protocol": "https",
        "host": "navigation-office.esa.int",
        "root": "https://navigation-office.esa.int/products/gnss-products",
        "tokens": {"sp3": "ESA0MGNFIN", "ionex": "ESA0OPSFIN"},
        "layouts": {"sp3": "gps_week", "ionex": "gps_week"},
        "samples": {"sp3": "05M", "ionex": "02H"},
    },
    "cod": {
        "protocol": "http",
        "host": "ftp.aiub.unibe.ch",
        "root": "http://ftp.aiub.unibe.ch",
        "tokens": {"sp3": "COD0MGXFIN", "ionex": "COD0OPSFIN"},
        "layouts": {"sp3": "aiub_code_mgex_year", "ionex": "aiub_code_year"},
        "samples": {"sp3": "05M", "ionex": "01H"},
    },
    "gfz": {
        "protocol": "https",
        "host": "isdc-data.gfz.de",
        "root": "https://isdc-data.gfz.de/gnss/products",
        "tokens": {"sp3": "GFZ0OPSRAP"},
        "layouts": {"sp3": "gfz_rapid_week"},
        "samples": {"sp3": "15M"},
    },
    # Ultra-rapid SP3 centers (02D span, sub-daily issue times).
    "igs_ult": {
        "protocol": "https",
        "host": "igs.bkg.bund.de",
        "root": "https://igs.bkg.bund.de/root_ftp/IGS",
        "tokens": {"sp3": "IGS0OPSULT"},
        "layouts": {"sp3": "bkg_products_week"},
        "spans": {"sp3": "02D"},
        "samples": {"sp3": "15M"},
        "issues": ("0000", "0600", "1200", "1800"),
    },
    "cod_ult": {
        "protocol": "http",
        "host": "ftp.aiub.unibe.ch",
        "root": "http://ftp.aiub.unibe.ch",
        "tokens": {"sp3": "COD0OPSULT"},
        "layouts": {"sp3": "aiub_code_root"},
        "spans": {"sp3": "01D"},
        "samples": {"sp3": "05M"},
        "issues": ("0000",),
        "compression": {"sp3": "none"},
    },
    "esa_ult": {
        "protocol": "https",
        "host": "navigation-office.esa.int",
        "root": "https://navigation-office.esa.int/products/gnss-products",
        "tokens": {"sp3": "ESA0OPSULT"},
        "layouts": {"sp3": "gps_week"},
        "spans": {"sp3": "02D"},
        "samples": {"sp3": "15M"},
        "issues": ("0000", "0600", "1200", "1800"),
    },
    "gfz_ult": {
        "protocol": "https",
        "host": "isdc-data.gfz.de",
        "root": "https://isdc-data.gfz.de/gnss/products",
        "tokens": {"sp3": "GFZ0OPSULT"},
        "layouts": {"sp3": "gfz_ultra_week"},
        "spans": {"sp3": "02D"},
        "samples": {"sp3": "05M"},
        "issues": ("0000", "0600", "1200", "1800"),
    },
}

# Content type -> filename form.
#   code: the content code; ext: the file extension (case as published);
#   kind: "sampled" (AAAVPPPTTT_DATE_LEN_SMP_CNT.EXT).
_CONTENT: dict[str, dict] = {
    "sp3": {"code": "ORB", "ext": "SP3", "kind": "sampled"},
    "ionex": {"code": "GIM", "ext": "INX", "kind": "sampled"},
}

_ALLOWED_HOSTS = frozenset(c["host"] for c in _CENTERS.values())

_DEFAULT_MAX_DECOMPRESSED_BYTES = 500 * 1024 * 1024
_DEFAULT_MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF_S = 0.5

_OPSULT_ISSUES = ("0000", "0600", "1200", "1800")

# datetime.date.toordinal() of 1980-01-06, the GPS epoch.
_GPS_EPOCH_ORDINAL = _dt.date(1980, 1, 6).toordinal()


def centers() -> list[str]:
    """All supported analysis-center codes."""
    return list(_CENTERS.keys())


def content_types() -> list[str]:
    """All supported content-type codes."""
    return list(_CONTENT.keys())


def _center_def(code: str) -> dict:
    if not isinstance(code, str):
        raise UnknownCenter(f"unknown center: {code!r}")
    try:
        return _CENTERS[code]
    except KeyError:
        raise UnknownCenter(f"unknown center: {code!r}") from None


def gps_week(date: _dt.date) -> int:
    """The GPS week number for a calendar date (week 0 began 1980-01-06)."""
    return (date.toordinal() - _GPS_EPOCH_ORDINAL) // 7


def day_of_year(date: _dt.date) -> int:
    """The day-of-year (1-366) for a calendar date."""
    return date.timetuple().tm_yday


def _pad3(n: int) -> str:
    return f"{n:03d}"


def _date_block(date: _dt.date, issue: Optional[str]) -> str:
    iss = issue if issue is not None else "0000"
    return f"{date.year}{_pad3(day_of_year(date))}{iss}"


def predicted_day_offset(center: str) -> int:
    """Day offset a predicted IONEX alias maps to relative to its target date.

    ``cod_prd1`` is the current/near-future day (offset 0); ``cod_prd2`` is the
    day after (offset +1). Every other center returns 0.
    """
    if center == "cod_prd1":
        return 0
    if center == "cod_prd2":
        return 1
    return 0


def _as_date(target: Union[_dt.date, _dt.datetime]) -> _dt.date:
    if isinstance(target, _dt.datetime):
        return target.date()
    if isinstance(target, _dt.date):
        return target
    raise UnsupportedProduct(f"target must be a date or datetime, got {target!r}")


def _as_naive_datetime(target: Union[_dt.date, _dt.datetime]) -> _dt.datetime:
    if isinstance(target, _dt.datetime):
        return target.replace(tzinfo=None)
    if isinstance(target, _dt.date):
        return _dt.datetime(target.year, target.month, target.day)
    raise UnsupportedProduct(f"target must be a date or datetime, got {target!r}")


def _validate_sample(sample: str) -> None:
    if (
        not isinstance(sample, str)
        or len(sample) != 3
        or not sample[:2].isdigit()
        or not ("A" <= sample[2] <= "Z")
    ):
        raise UnsupportedProduct(f"invalid sample code: {sample!r}")


def _validate_issue(issue: str) -> None:
    if not isinstance(issue, str) or len(issue) != 4 or not issue.isdigit():
        raise UnsupportedProduct(f"invalid issue time: {issue!r}")
    hh, mm = int(issue[:2]), int(issue[2:])
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise UnsupportedProduct(f"invalid issue time: {issue!r}")


def _default_sample(center: str, content: str) -> str:
    cdef = _center_def(center)
    samples = cdef.get("samples", {})
    if content not in samples:
        raise UnsupportedProduct(f"no default sample for {center}/{content}")
    return samples[content]


# --- product -------------------------------------------------------------


@dataclass(frozen=True)
class Product:
    """A GNSS product specification.

    A pure value identifying one archived file: the analysis center, content
    type, calendar date, temporal sampling, and optional sub-daily issue time.
    It resolves deterministically (no network) to a canonical filename and a
    full archive URL.
    """

    center: str
    content: str
    date: _dt.date
    sample: str
    issue: Optional[str] = None

    def __post_init__(self) -> None:
        cdef = _center_def(self.center)
        if self.content not in _CONTENT:
            raise UnsupportedProduct(f"unknown content type: {self.content!r}")
        if self.content not in cdef.get("tokens", {}):
            raise UnsupportedProduct(f"{self.center} does not serve {self.content}")
        _validate_sample(self.sample)
        issues = cdef.get("issues")
        if issues is not None:
            if self.issue is None:
                raise UnsupportedProduct(f"{self.center} requires an issue time")
            _validate_issue(self.issue)
            if self.issue not in issues:
                raise UnsupportedProduct(
                    f"{self.center} does not publish issue {self.issue!r}"
                )
        elif self.issue is not None:
            raise UnsupportedProduct(f"{self.center} does not take an issue time")

    @property
    def gps_week(self) -> int:
        return gps_week(self.date)

    @property
    def day_of_year(self) -> int:
        return day_of_year(self.date)

    def canonical_filename(self) -> str:
        """The canonical IGS long-name filename (no ``.gz`` suffix)."""
        cdef = _center_def(self.center)
        descriptor = _CONTENT[self.content]
        token = cdef["tokens"][self.content]
        span = cdef.get("spans", {}).get(self.content, "01D")
        block = _date_block(self.date, self.issue)
        return (
            f"{token}_{block}_{span}_{self.sample}_"
            f"{descriptor['code']}.{descriptor['ext']}"
        )

    def _compression(self) -> str:
        cdef = _center_def(self.center)
        return cdef.get("compression", {}).get(self.content, "gzip")

    def _protocol(self) -> str:
        return _center_def(self.center)["protocol"]

    def _dir_path(self) -> str:
        cdef = _center_def(self.center)
        layout = cdef["layouts"][self.content]
        date = self.date
        if layout == "gfz_rapid_week":
            return f"rapid/w{gps_week(date)}"
        if layout == "gfz_ultra_week":
            return f"ultra/w{gps_week(date)}"
        if layout == "gps_week":
            return f"{gps_week(date)}"
        if layout == "bkg_products_week":
            return f"products/{gps_week(date)}"
        if layout == "aiub_code_mgex_year":
            return f"CODE_MGEX/CODE/{date.year}"
        if layout == "aiub_code_year":
            return f"CODE/{date.year}"
        if layout == "aiub_code_root":
            return "CODE"
        raise UnsupportedProduct(f"unknown layout: {layout!r}")

    def archive_url(self) -> str:
        """The full, compressed (``.gz`` where gzipped) archive URL."""
        cdef = _center_def(self.center)
        filename = self.canonical_filename()
        suffix = ".gz" if self._compression() == "gzip" else ""
        return f"{cdef['root']}/{self._dir_path()}/{filename}{suffix}"


# --- product builders ----------------------------------------------------


def product(
    center: str,
    content: str,
    date: _dt.date,
    sample: Optional[str] = None,
    *,
    issue: Optional[str] = None,
) -> Product:
    """Build a :class:`Product` for any center/content/date/sample."""
    if sample is None:
        sample = _default_sample(center, content)
    return Product(
        center=center, content=content, date=date, sample=sample, issue=issue
    )


def mgex_ionex(center: str, date: _dt.date, *, sample: Optional[str] = None) -> Product:
    """Build an IONEX product for ``center`` on ``date`` (single exact day)."""
    return product(center, "ionex", date, sample)


def rapid_ionex(date: _dt.date, *, sample: Optional[str] = None) -> Product:
    """Build the CODE rapid IONEX product (``COD0OPSRAP``) for a UTC day."""
    return product("cod_rap", "ionex", date, sample)


def predicted_ionex(
    center: str, date: _dt.date, *, sample: Optional[str] = None
) -> Product:
    """Build a CODE predicted IONEX product (``COD0OPSPRD``) for a UTC day.

    ``center`` is ``"cod_prd1"`` (1-day-ahead) or ``"cod_prd2"`` (2-day-ahead).
    The horizon is encoded by offsetting the target day; both aliases serve the
    same ``COD0OPSPRD`` token.
    """
    if center not in ("cod_prd1", "cod_prd2"):
        raise UnknownCenter(
            f"predicted_ionex center must be cod_prd1 or cod_prd2, got {center!r}"
        )
    target = date + _dt.timedelta(days=predicted_day_offset(center))
    return product(center, "ionex", target, sample)


def mgex_sp3(center: str, date: _dt.date, *, sample: Optional[str] = None) -> Product:
    """Build an MGEX/precise SP3 product for a center and date."""
    return product(center, "sp3", date, sample)


def ops_ultra_sp3(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    *,
    sample: Optional[str] = None,
    issue: Optional[str] = None,
    available_issues: Optional[Sequence[tuple[_dt.date, str]]] = None,
) -> Product:
    """Build an ultra-rapid OPS SP3 product.

    Pass a ``date`` with an explicit ``issue`` (defaults to ``"0000"``), or a
    ``datetime`` target and the latest issue not after that time is selected. If
    ``available_issues`` is given, selection falls back to the newest issue
    present in that list.
    """
    cdef = _center_def(center)
    if "issues" not in cdef or "sp3" not in cdef.get("tokens", {}):
        raise UnsupportedProduct(f"{center} is not an ultra-rapid SP3 center")
    if sample is None:
        sample = _default_sample(center, "sp3")
    if isinstance(target, _dt.datetime):
        if issue is not None:
            return Product(center, "sp3", target.date(), sample, issue)
        date, issue = _latest_ultra_issue(
            center, _as_naive_datetime(target), available_issues
        )
        return Product(center, "sp3", date, sample, issue)
    date = _as_date(target)
    if issue is None:
        issue = "0000"
    return Product(center, "sp3", date, sample, issue)


def _issue_epoch(date: _dt.date, issue: str) -> _dt.datetime:
    return _dt.datetime(date.year, date.month, date.day, int(issue[:2]), int(issue[2:]))


def _ultra_issue_candidates(
    center: str, target: _dt.datetime
) -> list[tuple[_dt.date, str]]:
    """Candidate ultra issues at or before ``target``, newest first."""
    cdef = _center_def(center)
    issues = cdef.get("issues")
    if issues is None:
        raise UnsupportedProduct(f"{center} is not an ultra-rapid SP3 center")
    target_date = target.date()
    candidates: list[tuple[_dt.datetime, _dt.date, str]] = []
    for date in (target_date, target_date - _dt.timedelta(days=1)):
        for issue in issues:
            epoch = _issue_epoch(date, issue)
            if epoch <= target:
                candidates.append((epoch, date, issue))
    candidates.sort(key=lambda c: c[0], reverse=True)
    return [(date, issue) for (_epoch, date, issue) in candidates]


def _latest_ultra_issue(
    center: str,
    target: _dt.datetime,
    available: Optional[Sequence[tuple[_dt.date, str]]],
) -> tuple[_dt.date, str]:
    candidates = _ultra_issue_candidates(center, target)
    if available is None:
        if not candidates:
            raise UnsupportedProduct(f"no ultra issue at or before {target}")
        return candidates[0]
    available_set = {(d, i) for (d, i) in available}
    for cand in candidates:
        if cand in available_set:
            return cand
    raise UnsupportedProduct(f"no available ultra issue at or before {target}")


def canonical_filename(
    center: str,
    content: str,
    date: _dt.date,
    sample: Optional[str] = None,
    *,
    issue: Optional[str] = None,
) -> str:
    """The canonical IGS long-name filename for a center/content/date/sample."""
    return product(center, content, date, sample, issue=issue).canonical_filename()


def archive_url(
    center: str,
    content: str,
    date: _dt.date,
    sample: Optional[str] = None,
    *,
    issue: Optional[str] = None,
) -> str:
    """The full, compressed archive URL for a center/content/date/sample."""
    return product(center, content, date, sample, issue=issue).archive_url()


# --- gim candidate days --------------------------------------------------


def _gim_date_candidates(
    center: str, target: Union[_dt.date, _dt.datetime], lookback: int
) -> list[_dt.date]:
    cdef = _center_def(center)
    if "ionex" not in cdef.get("tokens", {}):
        raise UnsupportedProduct(f"{center} does not serve ionex")
    base = _as_date(target) + _dt.timedelta(days=predicted_day_offset(center))
    return [base - _dt.timedelta(days=back) for back in range(lookback + 1)]


# --- cache ---------------------------------------------------------------


def default_cache_dir() -> str:
    """The default cache directory, ``user_cache_dir("sidereon")/gnss``."""
    return _os.path.join(platformdirs.user_cache_dir("sidereon"), "gnss")


def _resolve_cache_dir(cache_dir: Optional[str]) -> str:
    return cache_dir if cache_dir is not None else default_cache_dir()


def _sha256(data: bytes) -> str:
    return _hashlib.sha256(data).hexdigest()


def _validate_cache_name(filename: str) -> None:
    if (
        filename in ("", ".", "..")
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or ".." in filename
        or _os.path.isabs(filename)
    ):
        raise CacheNotWritable(f"unsafe cache name: {filename!r}")


def _cache_path(cache_dir: str, filename: str) -> str:
    _validate_cache_name(filename)
    return _os.path.join(cache_dir, filename)


def _provenance_path(path: str) -> str:
    return path + ".provenance.json"


def _read_provenance(path: str) -> Optional[dict]:
    try:
        with open(_provenance_path(path), "rb") as handle:
            return _json.loads(handle.read())
    except FileNotFoundError:
        return None
    except (ValueError, OSError):
        return None


def _classify(path: str, expected_sha256: Optional[str]) -> tuple[str, object]:
    """Classify a cache entry: ('hit'|'absent'|'stale'|'unverified', detail)."""
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except FileNotFoundError:
        return ("absent", None)

    got = _sha256(data)
    if expected_sha256 is not None:
        if got == expected_sha256.lower():
            return ("hit", path)
        return ("stale", ChecksumMismatch(expected_sha256.lower(), got))

    prov = _read_provenance(path)
    if prov and isinstance(prov.get("sha256_decompressed"), str):
        recorded = prov["sha256_decompressed"].lower()
        if got == recorded:
            return ("hit", path)
        return ("stale", ChecksumMismatch(recorded, got))
    return ("unverified", path)


def _gunzip(compressed: bytes, max_bytes: int) -> bytes:
    """Decompress gzip bytes, aborting if the output would exceed max_bytes."""
    decompressor = _zlib.decompressobj(16 + 15)
    out = bytearray()
    try:
        out += decompressor.decompress(compressed, max_bytes + 1)
        if len(out) > max_bytes:
            raise DecompressError(
                f"decompressed output exceeded cap of {max_bytes} bytes"
            )
        while decompressor.unconsumed_tail:
            chunk = decompressor.decompress(decompressor.unconsumed_tail, max_bytes + 1)
            out += chunk
            if len(out) > max_bytes:
                raise DecompressError(
                    f"decompressed output exceeded cap of {max_bytes} bytes"
                )
        out += decompressor.flush()
    except _zlib.error as exc:
        raise DecompressError(f"corrupt gzip: {exc}") from exc
    if len(out) > max_bytes:
        raise DecompressError(f"decompressed output exceeded cap of {max_bytes} bytes")
    # A truncated gzip stream decompresses without raising but never reaches the
    # end-of-stream trailer; refuse to cache a partial product.
    if not decompressor.eof:
        raise DecompressError("truncated gzip stream (no end-of-stream marker)")
    if decompressor.unused_data:
        raise DecompressError("trailing bytes after gzip end-of-stream")
    return bytes(out)


def _ensure_dir(directory: str) -> None:
    try:
        _os.makedirs(directory, exist_ok=True)
    except OSError as exc:
        raise CacheNotWritable(f"cannot create cache dir {directory}: {exc}") from exc


def _write_temp(directory: str, data: bytes) -> str:
    import tempfile

    try:
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
        with _os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            _os.fsync(handle.fileno())
        return tmp
    except OSError as exc:
        raise CacheNotWritable(f"cannot write to cache dir {directory}: {exc}") from exc


def _commit(path: str, decompressed: bytes, provenance: dict) -> str:
    """Atomically commit the file and its required provenance sidecar."""
    directory = _os.path.dirname(path)
    sidecar = _provenance_path(path)
    _ensure_dir(directory)
    json_bytes = _json.dumps(provenance, indent=2).encode("utf-8")

    data_tmp = _write_temp(directory, decompressed)
    try:
        prov_tmp = _write_temp(directory, json_bytes)
    except CacheNotWritable:
        _silent_remove(data_tmp)
        raise

    # Publish the provenance sidecar first, then the data file, so a data file is
    # never visible without its provenance (a reader keys on the data file and
    # treats data-without-sidecar as unverified). A crash between the two leaves
    # at most an orphan sidecar, which is harmless and overwritten on the next
    # commit. On failure we only clean up our own temp files and never unlink a
    # path that may already be a valid published product.
    try:
        _os.replace(prov_tmp, sidecar)
    except OSError as exc:
        _silent_remove(prov_tmp)
        _silent_remove(data_tmp)
        raise CacheNotWritable(f"cannot commit provenance for {path}: {exc}") from exc

    try:
        _os.replace(data_tmp, path)
    except OSError as exc:
        _silent_remove(data_tmp)
        raise CacheNotWritable(f"cannot commit {path}: {exc}") from exc
    return path


def _silent_remove(path: str) -> None:
    try:
        _os.remove(path)
    except OSError:
        pass


# --- download ------------------------------------------------------------


def _check_host(url: str, protocol: str) -> None:
    parts = urlsplit(url)
    if parts.hostname not in _ALLOWED_HOSTS:
        raise NetworkError(f"host not allowed: {parts.hostname}")
    if parts.scheme != protocol:
        raise NetworkError(f"scheme mismatch: {parts.scheme} != {protocol} for {url}")


def _download(url: str, protocol: str, opts: dict) -> bytes:
    _check_host(url, protocol)
    retries = opts.get("retries", _DEFAULT_RETRIES)
    backoff = opts.get("backoff_s", _DEFAULT_BACKOFF_S)
    timeout = opts.get("timeout_s", _DEFAULT_TIMEOUT_S)
    max_bytes = opts.get("max_compressed_bytes", _DEFAULT_MAX_COMPRESSED_BYTES)

    attempt = 0
    while True:
        attempt += 1
        try:
            return _download_once(url, timeout, max_bytes)
        except (FileNotFoundOnArchive, DownloadSizeExceeded, RedirectNotAllowed):
            raise
        except HttpStatusError as exc:
            transient = exc.status in (408, 429) or exc.status >= 500
            if transient and attempt < retries:
                _sleep(backoff * (2 ** (attempt - 1)))
                continue
            raise
        except NetworkError:
            if attempt < retries:
                _sleep(backoff * (2 ** (attempt - 1)))
                continue
            raise


def _sleep(seconds: float) -> None:
    if seconds > 0:
        import time

        time.sleep(seconds)


def _download_once(url: str, timeout: float, max_bytes: int) -> bytes:
    try:
        with httpx.stream(
            "GET",
            url,
            follow_redirects=False,
            timeout=timeout,
        ) as response:
            status = response.status_code
            if status == 200:
                buf = bytearray()
                for chunk in response.iter_bytes():
                    buf += chunk
                    if len(buf) > max_bytes:
                        response.close()
                        raise DownloadSizeExceeded(
                            f"compressed payload exceeded {max_bytes} bytes"
                        )
                return bytes(buf)
            if status == 404:
                raise FileNotFoundOnArchive(url)
            if 300 <= status < 400:
                raise RedirectNotAllowed(status, url)
            raise HttpStatusError(status, url)
    except httpx.HTTPError as exc:
        raise NetworkError(f"network error for {url}: {exc}") from exc


def _provenance(
    url: str, protocol: str, compression: str, downloaded: bytes, decompressed: bytes
) -> dict:
    return {
        "source_url": url,
        "protocol": protocol,
        "compression": compression,
        "sha256_downloaded": _sha256(downloaded),
        "sha256_compressed": _sha256(downloaded),
        "sha256_decompressed": _sha256(decompressed),
        "size_downloaded": len(downloaded),
        "size_compressed": len(downloaded),
        "size_decompressed": len(decompressed),
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "fetcher": "sidereon.data",
    }


# --- fetch ---------------------------------------------------------------


def fetch(
    product: Product,
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    sha256: Optional[str] = None,
    max_decompressed_bytes: int = _DEFAULT_MAX_DECOMPRESSED_BYTES,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    retries: int = _DEFAULT_RETRIES,
    backoff_s: float = _DEFAULT_BACKOFF_S,
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES,
) -> str:
    """Fetch a product, returning the local path to its decompressed file.

    Cache-first: a verified cache hit returns immediately with no network. A
    corrupt cache hit is re-downloaded online and is terminal offline; an
    unverifiable hit (no provenance sidecar and no caller checksum) is
    re-downloaded online and is a miss offline (an untrusted file is never
    returned). Raises a :class:`DataError` subclass on failure.
    """
    resolved = _resolve_cache_dir(cache_dir)
    filename = product.canonical_filename()
    path = _cache_path(resolved, filename)

    state, detail = _classify(path, sha256)
    if state == "hit":
        return path
    if state == "absent":
        if offline:
            raise OfflineCacheMiss(f"not cached: {filename}")
        return _download_and_cache(
            product,
            path,
            sha256,
            max_decompressed_bytes,
            dict(
                timeout_s=timeout_s,
                retries=retries,
                backoff_s=backoff_s,
                max_compressed_bytes=max_compressed_bytes,
            ),
        )
    if state == "unverified":
        if offline:
            raise OfflineCacheMiss(
                f"cached but unverifiable (no provenance sidecar): {filename}"
            )
        return _download_and_cache(
            product,
            path,
            sha256,
            max_decompressed_bytes,
            dict(
                timeout_s=timeout_s,
                retries=retries,
                backoff_s=backoff_s,
                max_compressed_bytes=max_compressed_bytes,
            ),
        )
    # stale
    if offline:
        raise detail  # ChecksumMismatch
    return _download_and_cache(
        product,
        path,
        sha256,
        max_decompressed_bytes,
        dict(
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
            max_compressed_bytes=max_compressed_bytes,
        ),
    )


def _download_and_cache(
    product: Product,
    path: str,
    sha256: Optional[str],
    max_decompressed_bytes: int,
    opts: dict,
) -> str:
    url = product.archive_url()
    protocol = product._protocol()
    compression = product._compression()
    downloaded = _download(url, protocol, opts)
    if compression == "gzip":
        decompressed = _gunzip(downloaded, max_decompressed_bytes)
    else:
        decompressed = downloaded
    if sha256 is not None:
        got = _sha256(decompressed)
        if got != sha256.lower():
            raise ChecksumMismatch(sha256.lower(), got)
    provenance = _provenance(url, protocol, compression, downloaded, decompressed)
    return _commit(path, decompressed, provenance)


def fetch_ionex(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    sample: Optional[str] = None,
    lookback: int = 2,
    **fetch_opts,
) -> "sidereon.Ionex":
    """Fetch the newest available IONEX map for a target day, parsed.

    Walks candidate days newest-first (the rapid map lands a day or two late,
    predicted maps are published ahead of their target day) and returns the
    first hit parsed via :func:`sidereon.load_ionex`. Raises the last error when
    every candidate misses (preserving :class:`OfflineCacheMiss` offline).
    """
    dates = _gim_date_candidates(center, target, lookback)
    last_error: Optional[DataError] = None
    for date in dates:
        prod = product(center, "ionex", date, sample)
        try:
            path = fetch(prod, cache_dir=cache_dir, offline=offline, **fetch_opts)
        except (FileNotFoundOnArchive, OfflineCacheMiss) as exc:
            # Expected absence for this candidate day; walk to the next one.
            # Integrity and transport failures (checksum, decompress, HTTP, cache)
            # are real problems and propagate rather than being masked as a miss.
            last_error = exc
            continue
        return sidereon.load_ionex(path)
    if last_error is not None:
        raise last_error
    raise UnsupportedProduct("no candidate IONEX days to try")


# --- merged SP3 ----------------------------------------------------------


@dataclass(frozen=True)
class AbsentCenter:
    """A center that did not contribute to a merged SP3 fetch."""

    center: str
    filename: Optional[str]
    reason: str


@dataclass(frozen=True)
class Contributor:
    """A center that contributed an SP3 product to a merged fetch."""

    center: str
    filename: str
    date: _dt.date
    issue: Optional[str]


@dataclass
class MergeReport:
    """Audit report for a merged SP3 fetch.

    Carries the per-center contribution audit plus the binding's own SP3 merge
    report (``merge_report``) when more than one center contributed.
    """

    contributors: list[Contributor]
    absent: list[AbsentCenter]
    source_count: int
    single_product: bool
    merged: bool
    merge_report: Optional["sidereon.Sp3MergeReport"] = field(default=None)


def _ultra_center(center: str) -> bool:
    return "issues" in _CENTERS.get(center, {})


def _sp3_candidates(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    sample: Optional[str],
) -> list[Product]:
    cdef = _center_def(center)
    if "sp3" not in cdef.get("tokens", {}):
        raise UnsupportedProduct(f"{center} does not serve sp3")
    eff_sample = sample if sample is not None else _default_sample(center, "sp3")

    if _ultra_center(center) and isinstance(target, _dt.datetime):
        candidates = _ultra_issue_candidates(center, _as_naive_datetime(target))
        return [
            Product(center, "sp3", date, eff_sample, issue)
            for (date, issue) in candidates
        ]
    date = _as_date(target)
    if _ultra_center(center):
        return [Product(center, "sp3", date, eff_sample, "0000")]
    return [Product(center, "sp3", date, eff_sample)]


def _fetch_center_sp3(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    sample: Optional[str],
    fetch_kwargs: dict,
):
    try:
        candidates = _sp3_candidates(center, target, sample)
    except UnsupportedProduct as exc:
        # The center does not publish SP3 at all: a clean absence for the merge.
        # UnknownCenter (a caller mistake) propagates rather than being recorded.
        return ("absent", AbsentCenter(center, None, _reason_str(exc)))

    last: Optional[tuple[str, DataError]] = None
    for prod in candidates:
        filename = prod.canonical_filename()
        try:
            path = fetch(prod, **fetch_kwargs)
        except (FileNotFoundOnArchive, OfflineCacheMiss) as exc:
            # Expected absence for this candidate; try the next. Integrity,
            # cache, and transport failures are real and propagate instead of
            # being silently recorded as an absent center.
            last = (filename, exc)
            continue
        sp3 = sidereon.load_sp3(path)
        return (
            "ok",
            Contributor(center, filename, prod.date, prod.issue),
            sp3,
        )
    if last is not None:
        return ("absent", AbsentCenter(center, last[0], _reason_str(last[1])))
    return ("absent", AbsentCenter(center, None, "no_candidate"))


def _reason_str(exc: DataError) -> str:
    if isinstance(exc, OfflineCacheMiss):
        return "offline_miss"
    if isinstance(exc, FileNotFoundOnArchive):
        return "not_published"
    if isinstance(exc, ChecksumMismatch):
        return "checksum"
    if isinstance(exc, HttpStatusError):
        return f"http_status:{exc.status}"
    return type(exc).__name__


def fetch_merged_sp3(
    target: Union[_dt.date, _dt.datetime],
    centers: Sequence[str],
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    systems: Optional[Sequence[str]] = None,
    epoch_interval_s: Optional[float] = None,
    sample: Optional[str] = None,
    **fetch_opts,
) -> tuple["sidereon.Sp3", MergeReport]:
    """Fetch SP3 from several centers and merge the available ones.

    ``centers`` are tried in precedence order; a missing or not-yet-published
    center is recorded in the report and does not abort the call. Returns the
    parsed merged :class:`sidereon.Sp3` and a :class:`MergeReport`. Raises
    :class:`NoProducts` when no center contributes and
    :class:`IncompatibleSources` when the fetched sources cannot be combined.
    """
    if not isinstance(centers, (list, tuple)):
        raise UnsupportedProduct("centers must be a list of center codes")

    # Validate every center up front so an unknown code raises rather than being
    # silently recorded as an absent contributor.
    for center in centers:
        _center_def(center)

    fetch_kwargs = dict(cache_dir=cache_dir, offline=offline, **fetch_opts)
    results = [
        _fetch_center_sp3(center, target, sample, fetch_kwargs) for center in centers
    ]

    contributors = [r for r in results if r[0] == "ok"]
    absent = [r[1] for r in results if r[0] == "absent"]

    if not contributors:
        raise NoProducts(absent)

    if len(contributors) == 1:
        _state, info, sp3 = contributors[0]
        report = MergeReport(
            contributors=[info],
            absent=absent,
            source_count=1,
            single_product=True,
            merged=False,
            merge_report=None,
        )
        return sp3, report

    sources = [c[2] for c in contributors]
    options = _merge_options(systems, epoch_interval_s)
    try:
        merged, merge_report = sidereon.merge_sp3(sources, options)
    except sidereon.SidereonError as exc:
        raise IncompatibleSources([c[1].center for c in contributors], exc) from exc

    report = MergeReport(
        contributors=[c[1] for c in contributors],
        absent=absent,
        source_count=len(contributors),
        single_product=False,
        merged=True,
        merge_report=merge_report,
    )
    return merged, report


def _merge_options(
    systems: Optional[Sequence[str]], epoch_interval_s: Optional[float]
) -> Optional["sidereon.Sp3MergeOptions"]:
    if systems is None and epoch_interval_s is None:
        return None
    kwargs: dict = {}
    if systems is not None:
        kwargs["systems"] = list(systems)
    if epoch_interval_s is not None:
        kwargs["target_epoch_interval_s"] = epoch_interval_s
    return sidereon.Sp3MergeOptions(**kwargs)


def write_sp3(sp3: "sidereon.Sp3", path: str, *, gzip: bool = False) -> str:
    """Write an SP3 product to ``path`` atomically, the inverse of fetch.

    Pass ``gzip=True`` to gzip-compress the output (pair it with a ``.gz``
    extension on ``path``). Returns the written path.
    """
    data = sp3.to_sp3_string().encode("ascii")
    if gzip:
        data = _gzip.compress(data)
    directory = _os.path.dirname(_os.path.abspath(path))
    _ensure_dir(directory)
    tmp = _write_temp(directory, data)
    try:
        _os.replace(tmp, path)
    except OSError as exc:
        _silent_remove(tmp)
        raise CacheNotWritable(f"cannot write {path}: {exc}") from exc
    return path


def fetch_merged_sp3_file(
    target: Union[_dt.date, _dt.datetime],
    centers: Sequence[str],
    path: str,
    *,
    gzip: bool = False,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    systems: Optional[Sequence[str]] = None,
    epoch_interval_s: Optional[float] = None,
    sample: Optional[str] = None,
    **fetch_opts,
) -> str:
    """Fetch the merged SP3 from several centers and persist it to ``path``.

    Composes :func:`fetch_merged_sp3` with :func:`write_sp3`. Returns the written
    path. Nothing is written if the fetch/merge step raises.
    """
    merged, _report = fetch_merged_sp3(
        target,
        centers,
        cache_dir=cache_dir,
        offline=offline,
        systems=systems,
        epoch_interval_s=epoch_interval_s,
        sample=sample,
        **fetch_opts,
    )
    return write_sp3(merged, path, gzip=gzip)
