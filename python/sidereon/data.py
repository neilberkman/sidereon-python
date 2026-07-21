"""Optional fetch-and-cache layer for GNSS products and terrain.

``sidereon.data`` downloads, decompresses, checksums, and records provenance for
precise-orbit, ionosphere, and terrain products, then hands back a local file
path or parsed handle. It is one-directional: numerical layers never call into
this module, so a solve or terrain lookup never depends on network availability.
Fetch once, then point the solver or terrain reader at the cached data.

Catalog tokens, archive URLs, filenames, terrain tile paths, and HGT-to-DTED
conversion come from the core catalog and converter. This module owns Python
transport, cache IO, checksum verification, provenance, offline behavior, and
typed errors.

Quick start::

    import sidereon.data as data

    # Newest available predicted ionosphere map, parsed:
    ionex = data.fetch_ionex("cod_prd1", date.today())

    # Merged current-day SP3 from several centers + the merge audit report:
    sp3, report = data.fetch_merged_sp3(date.today(), ["igs_ult", "gfz_ult"])

    # Fetch a terrain tile, then read the terrain cache with the core reader:
    terrain_root = data.default_terrain_cache_dir()
    path = data.fetch_dted(36.5, -106.5, cache_dir=terrain_root)
    if path is not None:
        terrain = sidereon.DtedTerrain(terrain_root)
        height_m = terrain.height_m(36.5, -106.5)

Bulk terrain workflow::

    report = data.prefetch_dted_bbox(36.0, -107.0, 37.0, -106.0)
    offline_report = data.prefetch_dted_tiles(
        ["N36W107", "N37W107"], offline=True
    )

Cache-first ``fetch`` returns a local file path; a verified cache hit returns
with no network. Pass ``offline=True`` to forbid all network access (a verified
cache hit is returned, a miss raises :class:`OfflineCacheMiss`).

Failures raise a typed exception from the :class:`DataError` hierarchy rather
than returning sentinels, except that terrain no-coverage returns ``None`` by
default because the terrain reader treats an absent ocean tile as sea level.
"""

from __future__ import annotations

import datetime as _dt
import gzip as _gzip
import hashlib as _hashlib
import json as _json
import math as _math
import os as _os
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Iterator, Mapping, Optional, Sequence, Union
from urllib.parse import urljoin, urlsplit

import httpx
import platformdirs

import sidereon
from sidereon._compression import GzipIntegrityError as _GzipIntegrityError
from sidereon._compression import GzipSizeLimitError as _GzipSizeLimitError
from sidereon._compression import gunzip_members as _gunzip_members
from sidereon._ingress import _STREAM_CHUNK_BYTES as _STREAM_CHUNK_BYTES
from sidereon._ingress import append_bounded as _append_bounded
from sidereon._sidereon import (
    data_allowed_hosts as _core_data_allowed_hosts,
)
from sidereon._sidereon import (
    data_archive_compression as _core_data_archive_compression,
)
from sidereon._sidereon import (
    data_archive_url as _core_data_archive_url,
)
from sidereon._sidereon import (
    data_canonical_filename as _core_data_canonical_filename,
)
from sidereon._sidereon import (
    data_center_entry as _core_data_center_entry,
)
from sidereon._sidereon import (
    data_centers as _core_data_centers,
)
from sidereon._sidereon import (
    data_content_types as _core_data_content_types,
)
from sidereon._sidereon import (
    data_day_of_year as _core_data_day_of_year,
)
from sidereon._sidereon import (
    data_default_sample as _core_data_default_sample,
)
from sidereon._sidereon import (
    data_default_sample_for_date as _core_data_default_sample_for_date,
)
from sidereon._sidereon import (
    data_dted_block_dir as _core_data_dted_block_dir,
)
from sidereon._sidereon import (
    data_dted_cache_relpath as _core_data_dted_cache_relpath,
)
from sidereon._sidereon import (
    data_dted_tile_filename as _core_data_dted_tile_filename,
)
from sidereon._sidereon import (
    data_gim_date_candidates as _core_data_gim_date_candidates,
)
from sidereon._sidereon import (
    data_gps_week as _core_data_gps_week,
)
from sidereon._sidereon import (
    data_hgt_to_dted as _core_data_hgt_to_dted,
)
from sidereon._sidereon import (
    data_parse_skadi_tile_id as _core_data_parse_skadi_tile_id,
)
from sidereon._sidereon import (
    data_predicted_day_offset as _core_data_predicted_day_offset,
)
from sidereon._sidereon import (
    data_product_sample as _core_data_product_sample,
)
from sidereon._sidereon import (
    data_product_solution_class as _core_data_product_solution_class,
)
from sidereon._sidereon import (
    data_skadi_archive_url as _core_data_skadi_archive_url,
)
from sidereon._sidereon import (
    data_skadi_band as _core_data_skadi_band,
)
from sidereon._sidereon import (
    data_skadi_source_entry as _core_data_skadi_source_entry,
)
from sidereon._sidereon import (
    data_skadi_tile_id as _core_data_skadi_tile_id,
)
from sidereon._sidereon import (
    data_sp3_content_start_convention as _core_data_sp3_content_start_convention,
)
from sidereon._sidereon import (
    data_space_weather_archive_url as _core_data_space_weather_archive_url,
)
from sidereon._sidereon import (
    data_space_weather_cache_relpath as _core_data_space_weather_cache_relpath,
)
from sidereon._sidereon import (
    data_space_weather_filename as _core_data_space_weather_filename,
)
from sidereon._sidereon import (
    data_space_weather_source_entry as _core_data_space_weather_source_entry,
)
from sidereon._sidereon import (
    data_supported_samples as _core_data_supported_samples,
)
from sidereon._sidereon import (
    data_terrain_tile_index as _core_data_terrain_tile_index,
)
from sidereon._sidereon import (
    data_ultra_issue_candidates as _core_data_ultra_issue_candidates,
)
from sidereon._sidereon import (
    data_ultra_sp3_locations as _core_data_ultra_sp3_locations,
)
from sidereon._sidereon import (
    sp3_merge_input_identity as _core_sp3_merge_input_identity,
)

if TYPE_CHECKING:
    from sidereon.distribution import (
        AcquisitionProvenance,
        DistributionSource,
        ProductIdentity,
        SourceFailure,
    )

__all__ = [
    "DataError",
    "UnknownCenter",
    "UnsupportedProduct",
    "InvalidCoordinate",
    "InvalidTileIndex",
    "InvalidTileId",
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
    "NoCoverage",
    "Product",
    "ArtifactIdentity",
    "AcquisitionFacts",
    "Sp3MergeInputIdentity",
    "MergeReport",
    "TerrainSourceEntry",
    "SpaceWeatherSourceEntry",
    "TerrainFetchReport",
    "default_cache_dir",
    "default_terrain_cache_dir",
    "centers",
    "content_types",
    "allowed_hosts",
    "gps_week",
    "day_of_year",
    "default_sample_for_date",
    "supported_samples",
    "product_solution_class",
    "Sp3ContentStartConvention",
    "sp3_content_start_convention",
    "canonical_filename",
    "archive_url",
    "mgex_ionex",
    "rapid_ionex",
    "predicted_ionex",
    "ops_ultra_sp3",
    "mgex_sp3",
    "product",
    "skadi_source_entry",
    "skadi_tile_id",
    "skadi_band",
    "skadi_archive_url",
    "space_weather_source_entry",
    "space_weather_filename",
    "space_weather_archive_url",
    "space_weather_cache_relpath",
    "terrain_tile_index",
    "dted_tile_filename",
    "dted_block_dir",
    "dted_cache_relpath",
    "parse_skadi_tile_id",
    "hgt_to_dted",
    "fetch",
    "fetch_dted",
    "fetch_space_weather",
    "prefetch_dted_bbox",
    "prefetch_dted_tiles",
    "populate_terrain_cache",
    "fetch_ionex",
    "fetch_merged_sp3",
    "fetch_merged_sp3_file",
    "sp3_merge_input_identity",
    "verify_merge_report",
    "write_sp3",
    "DistributionSource",
    "Distribution",
    "ProductIdentity",
    "ProductRequest",
    "EarthdataAuth",
    "SourceFailure",
    "AcquisitionProvenance",
    "AcquiredProduct",
    "AcquisitionError",
    "UnsupportedDistribution",
    "HttpAcquisitionError",
    "ProductNotPublished",
    "AuthenticationRequired",
    "AuthenticationFailed",
    "AuthorizationDenied",
    "RedirectPolicyFailure",
    "RetiredEndpoint",
    "MalformedUrl",
    "TransportFailure",
    "InvalidContentType",
    "ErrorDocument",
    "ContentLengthMismatch",
    "DecompressionFailure",
    "ProductValidationFailure",
    "CacheReadFailure",
    "CacheWriteFailure",
    "AllDistributorsFailed",
    "ExactProductSetError",
    "identity",
    "request",
    "cddis_url",
    "validate_exact_product_set",
    "acquire",
]

_DISTRIBUTION_EXPORTS = frozenset(__all__[__all__.index("DistributionSource") :])


# --- errors --------------------------------------------------------------


class DataError(Exception):
    """Base class for every fetch/cache failure in :mod:`sidereon.data`."""


class UnknownCenter(DataError):
    """The analysis-center code is not in the catalog."""


class UnsupportedProduct(DataError):
    """The requested center/content/sample combination is not buildable."""


class InvalidCoordinate(DataError):
    """A terrain coordinate is non-finite or outside the supported range."""


class InvalidTileIndex(DataError):
    """A terrain tile index is outside the supported one-degree cell range."""


class InvalidTileId(DataError):
    """A Skadi terrain tile id is malformed."""


class OfflineCacheMiss(DataError):
    """``offline=True`` and the product is not present in the cache."""

    code = "offline_cache_miss"


class FileNotFoundOnArchive(DataError):
    """A candidate URL returned HTTP 404."""

    def __init__(self, url: str, status: int = 404) -> None:
        self.status = status
        self.url = url
        super().__init__(f"HTTP {status} for candidate URL {url}")


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
    """A cached data file failed SHA-256 verification."""

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


class NoCoverage(DataError):
    """The terrain archive has no tile for a valid ocean/no-data cell."""

    def __init__(self, tile_id: str) -> None:
        self.tile_id = tile_id
        super().__init__(f"no terrain coverage for {tile_id}")


# --- catalog -------------------------------------------------------------

_ALLOWED_HOSTS = frozenset(_core_data_allowed_hosts())

_DEFAULT_MAX_DECOMPRESSED_BYTES = 500 * 1024 * 1024
_DEFAULT_MAX_COMPRESSED_BYTES = 64 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 30.0
_DEFAULT_RETRIES = 3
_DEFAULT_BACKOFF_S = 0.5
_MAX_REDIRECTS = 5
_AIUB_WEB_HOST = "www.aiub.unibe.ch"
_AIUB_DOWNLOAD_HOST = "download.aiub.unibe.ch"
_AIUB_OBJECT_STORE_SUFFIX = ".s3.cloud.switch.ch"


@dataclass(frozen=True)
class TerrainSourceEntry:
    """Catalog facts for the terrain archive source."""

    protocol: str
    host: str
    compression: str
    root_url: str


@dataclass(frozen=True)
class SpaceWeatherSourceEntry:
    """Catalog facts for the CelesTrak space-weather source."""

    protocol: str
    host: str
    compression: str
    root_url: str


@dataclass(frozen=True)
class TerrainFetchReport:
    """Partitioned result from a terrain region or bulk cache population."""

    fetched: list[str]
    cached: list[str]
    no_coverage: list[str]
    errors: list[tuple[str, DataError]]


def centers() -> list[str]:
    """All supported analysis-center codes."""
    return list(_core_data_centers())


def content_types() -> list[str]:
    """All supported content-type codes."""
    return list(_core_data_content_types())


def allowed_hosts() -> list[str]:
    """Archive hosts allowed by the core public data catalog."""
    return sorted(_ALLOWED_HOSTS)


def _catalog_error(exc: ValueError) -> DataError:
    message = str(exc)
    if message.startswith("unknown analysis center"):
        return UnknownCenter(message)
    return UnsupportedProduct(message)


def _terrain_catalog_error(exc: ValueError) -> DataError:
    message = str(exc)
    if message.startswith("invalid terrain coordinate"):
        return InvalidCoordinate(message)
    if message.startswith("invalid terrain tile index"):
        return InvalidTileIndex(message)
    if message.startswith("invalid skadi tile id"):
        return InvalidTileId(message)
    return UnsupportedProduct(message)


def _hgt_conversion_error(exc: ValueError) -> DataError:
    message = str(exc)
    if message.startswith("invalid terrain tile index"):
        return InvalidTileIndex(message)
    return DecompressError(message)


def _center_def(code: str) -> dict:
    if not isinstance(code, str):
        raise UnknownCenter(f"unknown center: {code!r}")
    try:
        protocol, host, products, issues = _core_data_center_entry(code)
    except ValueError as exc:
        raise _catalog_error(exc) from None
    return {
        "protocol": protocol,
        "host": host,
        "products": set(products),
        "issues": tuple(issues),
    }


def gps_week(date: _dt.date) -> int:
    """The GPS week number for a calendar date (week 0 began 1980-01-06)."""
    try:
        return int(_core_data_gps_week(date.year, date.month, date.day))
    except ValueError as exc:
        raise UnsupportedProduct(str(exc)) from None


def day_of_year(date: _dt.date) -> int:
    """The day-of-year (1-366) for a calendar date."""
    try:
        return int(_core_data_day_of_year(date.year, date.month, date.day))
    except ValueError as exc:
        raise UnsupportedProduct(str(exc)) from None


def skadi_source_entry() -> TerrainSourceEntry:
    """Catalog facts for the Skadi SRTM source."""
    protocol, host, compression, root_url = _core_data_skadi_source_entry()
    return TerrainSourceEntry(protocol, host, compression, root_url)


def _space_weather_catalog_error(exc: ValueError) -> DataError:
    return UnsupportedProduct(str(exc))


def space_weather_source_entry() -> SpaceWeatherSourceEntry:
    """Catalog facts for the CelesTrak space-weather source."""
    protocol, host, compression, root_url = _core_data_space_weather_source_entry()
    return SpaceWeatherSourceEntry(protocol, host, compression, root_url)


def space_weather_filename(product: str = "sw_all") -> str:
    """Core-derived CelesTrak space-weather filename for ``product``."""
    try:
        return _core_data_space_weather_filename(product)
    except ValueError as exc:
        raise _space_weather_catalog_error(exc) from None


def space_weather_archive_url(product: str = "sw_all") -> str:
    """Core-derived CelesTrak space-weather source URL for ``product``."""
    try:
        return _core_data_space_weather_archive_url(product)
    except ValueError as exc:
        raise _space_weather_catalog_error(exc) from None


def space_weather_cache_relpath(product: str = "sw_all") -> str:
    """Core-derived cache path below the GNSS cache root."""
    try:
        return _core_data_space_weather_cache_relpath(product)
    except ValueError as exc:
        raise _space_weather_catalog_error(exc) from None


def skadi_tile_id(lat_index: int, lon_index: int) -> str:
    """Core-derived Skadi tile id, for example ``N36W107``."""
    try:
        return _core_data_skadi_tile_id(lat_index, lon_index)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None


def skadi_band(lat_index: int) -> str:
    """Core-derived Skadi latitude band, for example ``N36``."""
    try:
        return _core_data_skadi_band(lat_index)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None


def skadi_archive_url(lat_index: int, lon_index: int) -> str:
    """Core-derived terrain source URL for one tile."""
    try:
        return _core_data_skadi_archive_url(lat_index, lon_index)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None


def terrain_tile_index(lat_deg: float, lon_deg: float) -> tuple[int, int]:
    """Core-derived tile index covering a latitude/longitude coordinate."""
    try:
        lat_index, lon_index = _core_data_terrain_tile_index(lat_deg, lon_deg)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None
    return int(lat_index), int(lon_index)


def dted_tile_filename(lat_index: int, lon_index: int) -> str:
    """Core-derived DTED tile filename read by :class:`sidereon.DtedTerrain`."""
    try:
        return _core_data_dted_tile_filename(lat_index, lon_index)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None


def dted_block_dir(lat_index: int, lon_index: int) -> str:
    """Core-derived DTED ten-degree cache block directory."""
    try:
        return _core_data_dted_block_dir(lat_index, lon_index)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None


def dted_cache_relpath(lat_index: int, lon_index: int) -> str:
    """Core-derived DTED cache path below a terrain root."""
    try:
        return _core_data_dted_cache_relpath(lat_index, lon_index)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None


def parse_skadi_tile_id(tile_id: str) -> tuple[int, int]:
    """Parse a Skadi tile id through core validation."""
    try:
        lat_index, lon_index = _core_data_parse_skadi_tile_id(tile_id)
    except ValueError as exc:
        raise _terrain_catalog_error(exc) from None
    return int(lat_index), int(lon_index)


def hgt_to_dted(lat_index: int, lon_index: int, hgt: bytes) -> bytes:
    """Convert decompressed SRTM1 HGT bytes to deterministic DTED bytes."""
    try:
        return bytes(_core_data_hgt_to_dted(lat_index, lon_index, hgt))
    except ValueError as exc:
        raise _hgt_conversion_error(exc) from None


def predicted_day_offset(center: str) -> int:
    """Day offset a predicted IONEX alias maps to relative to its target date.

    ``cod_prd1`` is the current/near-future day (offset 0); ``cod_prd2`` is the
    day after (offset +1). Every other center returns 0.
    """
    try:
        return int(_core_data_predicted_day_offset(center))
    except ValueError as exc:
        raise _catalog_error(exc) from None


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
    try:
        return _core_data_default_sample(center, content)
    except ValueError as exc:
        raise _catalog_error(exc) from None


def default_sample_for_date(center: str, content: str, date: _dt.date) -> str:
    """Published default sample token for one center/product/date."""
    try:
        return _core_data_default_sample_for_date(
            center, content, date.year, date.month, date.day
        )
    except (AttributeError, ValueError) as exc:
        raise _catalog_error(ValueError(str(exc))) from None


def supported_samples(
    center: str,
    content: str,
    date: _dt.date,
    issue: Optional[str] = None,
) -> list[str]:
    """Officially cataloged sample tokens for one product date and issue.

    The result is product-, date-, and issue-aware. For an issue-based product
    line, omitting ``issue`` queries the ``0000`` issue; product construction
    still requires an explicit issue. Unsupported centers, products, eras, and
    issue values raise :class:`UnsupportedProduct`.
    """
    try:
        return list(
            _core_data_supported_samples(
                center, content, date.year, date.month, date.day, issue
            )
        )
    except (AttributeError, ValueError) as exc:
        raise _catalog_error(ValueError(str(exc))) from None


def product_solution_class(center: str, content: str) -> str:
    """Solution class for a supported center/product family.

    This is product-aware: for example, IGS SP3 is ``"final"`` while IGS
    broadcast navigation is ``"broadcast"``.
    """
    try:
        return _core_data_product_solution_class(center, content)
    except ValueError as exc:
        raise _catalog_error(exc) from None


class Sp3ContentStartConvention(Enum):
    """Relationship between an SP3 filename epoch and its first content epoch."""

    FILENAME_EPOCH = "filename_epoch"
    FILENAME_EPOCH_MINUS_ONE_DAY = "filename_epoch_minus_one_day"

    @property
    def content_start_offset_s(self) -> int:
        """Seconds added to the filename epoch to obtain the content start."""
        if self is Sp3ContentStartConvention.FILENAME_EPOCH:
            return 0
        return -86_400


def sp3_content_start_convention(
    center: str, date: _dt.date, issue: Optional[str] = None
) -> Sp3ContentStartConvention:
    """Cataloged first-content convention for one exact SP3 product issue.

    ``issue`` is required for ultra-rapid centers, must be one of that center's
    published issues, and must be omitted for product lines without issue times.
    """
    try:
        code, offset_s = _core_data_sp3_content_start_convention(
            center, date.year, date.month, date.day, issue
        )
    except (AttributeError, ValueError) as exc:
        raise _catalog_error(ValueError(str(exc))) from None

    try:
        convention = Sp3ContentStartConvention(code)
    except ValueError as exc:
        raise RuntimeError(
            f"core returned unknown SP3 content-start code {code!r}"
        ) from exc
    if convention.content_start_offset_s != offset_s:
        raise RuntimeError(
            "core returned inconsistent SP3 content-start code and offset: "
            f"{code!r}, {offset_s}"
        )
    return convention


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
    span: Optional[str] = None
    pattern: Optional[str] = None
    filename: Optional[str] = None
    cache_filename: Optional[str] = None
    url: Optional[str] = None
    compression: Optional[str] = None

    def __post_init__(self) -> None:
        cdef = _center_def(self.center)
        if self.content not in content_types():
            raise UnsupportedProduct(f"unknown content type: {self.content!r}")
        if self.content not in cdef["products"]:
            raise UnsupportedProduct(f"{self.center} does not serve {self.content}")
        _validate_sample(self.sample)
        issues = cdef["issues"]
        if issues:
            if self.issue is None:
                raise UnsupportedProduct(f"{self.center} requires an issue time")
            _validate_issue(self.issue)
            if self.issue not in issues:
                raise UnsupportedProduct(
                    f"{self.center} does not publish issue {self.issue!r}"
                )
        elif self.issue is not None:
            raise UnsupportedProduct(f"{self.center} does not take an issue time")
        self.canonical_filename()

    @property
    def gps_week(self) -> int:
        return gps_week(self.date)

    @property
    def day_of_year(self) -> int:
        return day_of_year(self.date)

    def canonical_filename(self) -> str:
        """The canonical official filename, without transport compression."""
        if self.filename is not None:
            return self.filename
        try:
            return _core_data_canonical_filename(
                self.center,
                self.content,
                self.date.year,
                self.date.month,
                self.date.day,
                self.sample,
                self.issue,
            )
        except ValueError as exc:
            raise _catalog_error(exc) from None

    def _compression(self) -> str:
        if self.compression is not None:
            return self.compression
        try:
            # Do not report a center-wide current compression for an era whose
            # direct archive layout is deliberately unsupported.
            if self.url is None:
                self.archive_url()
            return _core_data_archive_compression(self.center, self.content)
        except ValueError as exc:
            raise _catalog_error(exc) from None

    def archive_compression(self) -> str:
        """Transport compression used by this product's direct archive."""
        return self._compression()

    def _protocol(self) -> str:
        return _center_def(self.center)["protocol"]

    def archive_url(self) -> str:
        """The full, compressed (``.gz`` where gzipped) archive URL."""
        if self.url is not None:
            return self.url
        try:
            return _core_data_archive_url(
                self.center,
                self.content,
                self.date.year,
                self.date.month,
                self.date.day,
                self.sample,
                self.issue,
            )
        except ValueError as exc:
            raise _catalog_error(exc) from None


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
        try:
            sample = _core_data_product_sample(
                center,
                content,
                date.year,
                date.month,
                date.day,
                issue,
            )
        except ValueError as exc:
            raise _catalog_error(exc) from None
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
    present in that list. When ``sample`` is omitted, the core catalog selects
    the published cadence for that exact issue, including intraday transitions.
    """
    cdef = _center_def(center)
    if not cdef["issues"] or "sp3" not in cdef["products"]:
        raise UnsupportedProduct(f"{center} is not an ultra-rapid SP3 center")
    if isinstance(target, _dt.datetime):
        if issue is not None:
            date = target.date()
        else:
            date, issue = _latest_ultra_issue(
                center, _as_naive_datetime(target), available_issues
            )
    else:
        date = _as_date(target)
        if issue is None:
            issue = "0000"
    if sample is None:
        try:
            locations = _core_data_ultra_sp3_locations(
                center, date.year, date.month, date.day, issue
            )
        except ValueError as exc:
            raise _catalog_error(exc) from None
        if not locations:
            raise UnsupportedProduct(
                f"{center} has no ultra-rapid SP3 location for {date} issue {issue}"
            )
        # The core orders candidates with the exact published cadence for this
        # issue first. This matters at intraday cadence transitions, which a
        # date-only default cannot represent.
        sample = locations[0][2]
    return Product(center, "sp3", date, sample, issue)


def _issue_epoch(date: _dt.date, issue: str) -> _dt.datetime:
    return _dt.datetime(date.year, date.month, date.day, int(issue[:2]), int(issue[2:]))


def _ultra_issue_candidates(
    center: str, target: _dt.datetime
) -> list[tuple[_dt.date, str]]:
    """Candidate ultra issues at or before ``target``, newest first."""
    try:
        rows = _core_data_ultra_issue_candidates(
            center,
            target.year,
            target.month,
            target.day,
            target.hour,
            target.minute,
            target.second,
        )
    except ValueError as exc:
        raise _catalog_error(exc) from None
    return [(_dt.date(year, month, day), issue) for (year, month, day, issue) in rows]


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
    """The canonical official filename for a center/content/date/sample."""
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
    date = _as_date(target)
    try:
        rows = _core_data_gim_date_candidates(
            center, date.year, date.month, date.day, lookback
        )
    except ValueError as exc:
        raise _catalog_error(exc) from None
    return [_dt.date(year, month, day) for (year, month, day) in rows]


# --- cache ---------------------------------------------------------------


def default_cache_dir() -> str:
    """The default cache directory, ``user_cache_dir("sidereon")/gnss``."""
    return _os.path.join(platformdirs.user_cache_dir("sidereon"), "gnss")


def default_terrain_cache_dir() -> str:
    """The default terrain cache root, ``user_cache_dir("sidereon")/terrain``."""
    return _os.path.join(platformdirs.user_cache_dir("sidereon"), "terrain")


def _resolve_cache_dir(cache_dir: Optional[str]) -> str:
    return cache_dir if cache_dir is not None else default_cache_dir()


def _resolve_terrain_cache_dir(cache_dir: Optional[str]) -> str:
    return cache_dir if cache_dir is not None else default_terrain_cache_dir()


def _sha256(data: bytes) -> str:
    return _hashlib.sha256(data).hexdigest()


def _validate_cache_component(component: str) -> None:
    if (
        component in ("", ".", "..")
        or "/" in component
        or "\\" in component
        or "\x00" in component
        or ".." in component
        or _os.path.isabs(component)
    ):
        raise CacheNotWritable(f"unsafe cache path component: {component!r}")


def _validate_cache_name(filename: str) -> None:
    _validate_cache_component(filename)


def _cache_path(cache_dir: str, filename: str) -> str:
    _validate_cache_name(filename)
    return _os.path.join(cache_dir, filename)


def _terrain_cache_path(cache_dir: str, relpath: str) -> str:
    parts = relpath.split("/")
    if len(parts) != 2:
        raise CacheNotWritable(f"unsafe terrain cache path: {relpath!r}")
    for part in parts:
        _validate_cache_component(part)
    return _os.path.join(cache_dir, parts[0], parts[1])


def _catalog_rel_cache_path(cache_dir: str, relpath: str) -> str:
    parts = relpath.split("/")
    if len(parts) != 2:
        raise CacheNotWritable(f"unsafe cache path: {relpath!r}")
    for part in parts:
        _validate_cache_component(part)
    return _os.path.join(cache_dir, parts[0], parts[1])


def _provenance_path(path: str) -> str:
    return path + ".provenance.json"


def _no_coverage_path(path: str) -> str:
    return path + ".no_coverage.json"


def _read_provenance(path: str) -> Optional[dict]:
    try:
        with open(_provenance_path(path), "rb") as handle:
            return _json.loads(handle.read())
    except FileNotFoundError:
        return None
    except (ValueError, OSError):
        return None


def _fetched_at(provenance: Optional[dict]) -> Optional[_dt.datetime]:
    if not provenance:
        return None
    value = provenance.get("fetched_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = _dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def _fresh_enough(path: str, max_age_s: float) -> bool:
    fetched_at = _fetched_at(_read_provenance(path))
    if fetched_at is None:
        return False
    age_s = (_dt.datetime.now(_dt.timezone.utc) - fetched_at).total_seconds()
    return age_s <= max_age_s


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
    if prov and isinstance(
        prov.get("sha256_data", prov.get("sha256_decompressed")), str
    ):
        recorded = prov.get("sha256_data", prov.get("sha256_decompressed")).lower()
        if got == recorded:
            return ("hit", path)
        return ("stale", ChecksumMismatch(recorded, got))
    return ("unverified", path)


def _gunzip(compressed: bytes, max_bytes: int) -> bytes:
    """Decompress a complete gzip member sequence under one output cap."""
    try:
        return _gunzip_members(compressed, max_bytes)
    except _GzipSizeLimitError as exc:
        raise DecompressError(str(exc)) from None
    except _GzipIntegrityError as exc:
        raise DecompressError(str(exc)) from None


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


def _commit_json_sidecar(path: str, payload: dict) -> str:
    directory = _os.path.dirname(path)
    _ensure_dir(directory)
    json_bytes = _json.dumps(payload, indent=2).encode("utf-8")
    tmp = _write_temp(directory, json_bytes)
    try:
        _os.replace(tmp, path)
    except OSError as exc:
        _silent_remove(tmp)
        raise CacheNotWritable(f"cannot commit {path}: {exc}") from exc
    return path


def _commit_no_coverage_marker(path: str, tile_id: str, url: str, protocol: str) -> str:
    return _commit_json_sidecar(
        _no_coverage_path(path),
        {
            "source_url": url,
            "protocol": protocol,
            "status": 404,
            "tile_id": tile_id,
            "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
    )


def _read_no_coverage_marker(path: str, tile_id: str, url: str, protocol: str) -> bool:
    try:
        with open(_no_coverage_path(path), "rb") as handle:
            marker = _json.loads(handle.read())
    except FileNotFoundError:
        return False
    except (ValueError, OSError):
        return False
    return (
        isinstance(marker, dict)
        and marker.get("status") == 404
        and marker.get("tile_id") == tile_id
        and marker.get("source_url") == url
        and marker.get("protocol") == protocol
    )


def _delete_no_coverage_marker(path: str) -> None:
    _silent_remove(_no_coverage_path(path))


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
    current_url = url
    for redirect_count in range(_MAX_REDIRECTS + 1):
        try:
            with httpx.stream(
                "GET",
                current_url,
                follow_redirects=False,
                timeout=timeout,
            ) as response:
                status = response.status_code
                if status == 200:
                    buf = bytearray()
                    for chunk in response.iter_bytes(chunk_size=_STREAM_CHUNK_BYTES):
                        if _append_bounded(buf, chunk, max_bytes):
                            response.close()
                            raise DownloadSizeExceeded(
                                f"compressed payload exceeded {max_bytes} bytes"
                            )
                    return bytes(buf)
                if status == 404:
                    raise FileNotFoundOnArchive(current_url, status)
                if 300 <= status < 400:
                    location = response.headers.get("location")
                    if location is None or redirect_count == _MAX_REDIRECTS:
                        raise RedirectNotAllowed(status, current_url)
                    current_url = _validated_redirect_url(current_url, status, location)
                    continue
                raise HttpStatusError(status, current_url)
        except httpx.HTTPError as exc:
            raise NetworkError(f"network error for {current_url}: {exc}") from exc
    raise RedirectNotAllowed(310, current_url)


def _validated_redirect_url(source_url: str, status: int, location: str) -> str:
    target_url = urljoin(source_url, location)
    source = urlsplit(source_url)
    target = urlsplit(target_url)
    target_host = target.hostname or ""
    if source.scheme == "https" and target.scheme == "https":
        if source.hostname == _AIUB_WEB_HOST and target_host == _AIUB_DOWNLOAD_HOST:
            return target_url
        if source.hostname in {_AIUB_WEB_HOST, _AIUB_DOWNLOAD_HOST} and (
            target_host.endswith(_AIUB_OBJECT_STORE_SUFFIX)
        ):
            return target_url
    raise RedirectNotAllowed(status, source_url)


def _provenance(
    url: str, protocol: str, compression: str, downloaded: bytes, decompressed: bytes
) -> dict:
    data_digest = _sha256(decompressed)
    return {
        "source_url": url,
        "protocol": protocol,
        "compression": compression,
        "sha256_data": data_digest,
        "size_data": len(decompressed),
        "sha256_downloaded": _sha256(downloaded),
        "sha256_compressed": _sha256(downloaded),
        "sha256_decompressed": data_digest,
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
    path = _cache_path(resolved, product.cache_filename or filename)

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


# --- space weather -------------------------------------------------------


def fetch_space_weather(
    product: str = "sw_all",
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    sha256: Optional[str] = None,
    max_age_s: float = 86_400.0,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    retries: int = _DEFAULT_RETRIES,
    backoff_s: float = _DEFAULT_BACKOFF_S,
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES,
) -> "sidereon.SpaceWeatherTable":
    """Fetch and load a CelesTrak space-weather table.

    The product is mutable, so a verified cache hit is reused only while its
    provenance ``fetched_at`` is no older than ``max_age_s``. ``offline=True``
    returns a verified cached file at any age. A caller-supplied ``sha256`` pins
    an exact snapshot and is verified on every cache read and fetch.
    """
    if not _math.isfinite(max_age_s) or max_age_s < 0.0:
        raise UnsupportedProduct("max_age_s must be finite and non-negative")

    resolved = _resolve_cache_dir(cache_dir)
    relpath = space_weather_cache_relpath(product)
    path = _catalog_rel_cache_path(resolved, relpath)
    filename = space_weather_filename(product)

    state, detail = _classify(path, sha256)
    if state == "hit":
        if offline or sha256 is not None or _fresh_enough(path, max_age_s):
            return sidereon.load_space_weather(path)
    elif state == "absent":
        if offline:
            raise OfflineCacheMiss(f"not cached: {filename}")
    elif state == "unverified":
        if offline:
            raise OfflineCacheMiss(
                f"cached but unverifiable (no provenance sidecar): {filename}"
            )
    else:
        if sha256 is not None or offline:
            raise detail

    path = _download_space_weather_and_cache(
        product,
        path,
        sha256,
        dict(
            timeout_s=timeout_s,
            retries=retries,
            backoff_s=backoff_s,
            max_compressed_bytes=max_compressed_bytes,
        ),
    )
    return sidereon.load_space_weather(path)


def _download_space_weather_and_cache(
    product: str,
    path: str,
    sha256: Optional[str],
    opts: dict,
) -> str:
    source = space_weather_source_entry()
    url = space_weather_archive_url(product)
    downloaded = _download(url, source.protocol, opts)
    if source.compression != "none":
        raise UnsupportedProduct(
            f"unsupported space-weather compression: {source.compression}"
        )
    if sha256 is not None:
        got = _sha256(downloaded)
        if got != sha256.lower():
            raise ChecksumMismatch(sha256.lower(), got)
    provenance = _provenance(
        url, source.protocol, source.compression, downloaded, downloaded
    )
    return _commit(path, downloaded, provenance)


# --- terrain -------------------------------------------------------------


def fetch_dted(
    lat: float,
    lon: float,
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    sha256: Optional[str] = None,
    strict: bool = False,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    retries: int = _DEFAULT_RETRIES,
    backoff_s: float = _DEFAULT_BACKOFF_S,
    max_compressed_bytes: int = _DEFAULT_MAX_COMPRESSED_BYTES,
    max_decompressed_bytes: int = _DEFAULT_MAX_DECOMPRESSED_BYTES,
) -> Optional[str]:
    """Fetch the DTED tile covering ``lat``/``lon`` and return its local path.

    A verified cache hit returns without a request. A known ocean/no-coverage
    tile returns ``None`` by default, or raises :class:`NoCoverage` when
    ``strict=True``. The returned file is written below the terrain cache root
    in the block-directory layout read by :class:`sidereon.DtedTerrain`.
    """
    lat_index, lon_index = terrain_tile_index(lat, lon)
    state, value = _fetch_dted_tile(
        lat_index,
        lon_index,
        cache_dir=cache_dir,
        offline=offline,
        sha256=sha256,
        strict=strict,
        timeout_s=timeout_s,
        retries=retries,
        backoff_s=backoff_s,
        max_compressed_bytes=max_compressed_bytes,
        max_decompressed_bytes=max_decompressed_bytes,
    )
    if state == "no_coverage":
        return None
    return str(value)


def _fetch_dted_tile(
    lat_index: int,
    lon_index: int,
    *,
    cache_dir: Optional[str],
    offline: bool,
    sha256: Optional[str],
    strict: bool,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    max_compressed_bytes: int,
    max_decompressed_bytes: int,
) -> tuple[str, object]:
    tile_id = skadi_tile_id(lat_index, lon_index)
    url = skadi_archive_url(lat_index, lon_index)
    relpath = dted_cache_relpath(lat_index, lon_index)
    source = skadi_source_entry()
    root = _resolve_terrain_cache_dir(cache_dir)
    path = _terrain_cache_path(root, relpath)

    state, detail = _classify(path, sha256)
    if state == "hit":
        return ("cached", path)
    if state == "absent":
        if _read_no_coverage_marker(path, tile_id, url, source.protocol):
            if strict:
                raise NoCoverage(tile_id)
            return ("no_coverage", tile_id)
        if offline:
            raise OfflineCacheMiss(f"not cached: {tile_id}")
    elif state == "unverified":
        if offline:
            raise OfflineCacheMiss(
                f"cached but unverifiable (no provenance sidecar): {tile_id}"
            )
    elif offline:
        raise detail

    try:
        downloaded = _download(
            url,
            source.protocol,
            dict(
                timeout_s=timeout_s,
                retries=retries,
                backoff_s=backoff_s,
                max_compressed_bytes=max_compressed_bytes,
            ),
        )
    except FileNotFoundOnArchive:
        _commit_no_coverage_marker(path, tile_id, url, source.protocol)
        if strict:
            raise NoCoverage(tile_id) from None
        return ("no_coverage", tile_id)

    if source.compression == "gzip":
        hgt = _gunzip(downloaded, max_decompressed_bytes)
    elif source.compression == "none":
        hgt = downloaded
    else:
        raise UnsupportedProduct(
            f"unsupported terrain compression: {source.compression}"
        )

    dt2 = hgt_to_dted(lat_index, lon_index, hgt)
    got = _sha256(dt2)
    if sha256 is not None and got != sha256.lower():
        raise ChecksumMismatch(sha256.lower(), got)

    provenance = _terrain_provenance(
        url=url,
        source=source,
        downloaded=downloaded,
        hgt=hgt,
        dt2=dt2,
        tile_id=tile_id,
        lat_index=lat_index,
        lon_index=lon_index,
    )
    committed = _commit(path, dt2, provenance)
    _delete_no_coverage_marker(path)
    return ("fetched", committed)


def _terrain_provenance(
    *,
    url: str,
    source: TerrainSourceEntry,
    downloaded: bytes,
    hgt: bytes,
    dt2: bytes,
    tile_id: str,
    lat_index: int,
    lon_index: int,
) -> dict:
    hgt_gz_digest = _sha256(downloaded)
    hgt_digest = _sha256(hgt)
    dt2_digest = _sha256(dt2)
    return {
        "source_url": url,
        "protocol": source.protocol,
        "compression": source.compression,
        "sha256_data": dt2_digest,
        "size_data": len(dt2),
        "sha256_downloaded": hgt_gz_digest,
        "sha256_compressed": hgt_gz_digest,
        "sha256_decompressed": hgt_digest,
        "size_downloaded": len(downloaded),
        "size_compressed": len(downloaded),
        "size_decompressed": len(hgt),
        "sha256_hgt_gz": hgt_gz_digest,
        "sha256_hgt": hgt_digest,
        "sha256_dt2": dt2_digest,
        "size_dt2": len(dt2),
        "converter": "sidereon-core hgt_to_dted v1",
        "tile_id": tile_id,
        "lat_index": lat_index,
        "lon_index": lon_index,
        "fetched_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "fetcher": "sidereon.data",
    }


def prefetch_dted_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    **opts,
) -> TerrainFetchReport:
    """Fetch every DTED tile intersecting an inclusive bounding box."""
    if max_lat < min_lat:
        raise ValueError("max_lat must be greater than or equal to min_lat")
    if max_lon < min_lon:
        raise ValueError("max_lon must be greater than or equal to min_lon")
    lat_min, lon_min = terrain_tile_index(min_lat, min_lon)
    lat_max, lon_max = terrain_tile_index(max_lat, max_lon)
    tiles = [
        (lat_index, lon_index)
        for lat_index in range(min(lat_min, lat_max), max(lat_min, lat_max) + 1)
        for lon_index in range(min(lon_min, lon_max), max(lon_min, lon_max) + 1)
    ]
    return _prefetch_dted_tile_indices(
        tiles, cache_dir=cache_dir, offline=offline, **opts
    )


def prefetch_dted_tiles(
    tiles: Union[Iterable[tuple[int, int]], Iterable[str], str],
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    **opts,
) -> TerrainFetchReport:
    """Fetch an iterable of ``(lat_index, lon_index)`` pairs or tile-id strings."""
    entries = [tiles] if isinstance(tiles, str) else list(tiles)
    fetched: list[str] = []
    cached: list[str] = []
    no_coverage: list[str] = []
    errors: list[tuple[str, DataError]] = []
    valid_tiles: list[tuple[int, int]] = []

    for entry in entries:
        try:
            lat_index, lon_index = _coerce_tile_entry(entry)
            valid_tiles.append((lat_index, lon_index))
        except DataError as exc:
            errors.append((str(entry), exc))

    report = _prefetch_dted_tile_indices(
        valid_tiles, cache_dir=cache_dir, offline=offline, **opts
    )
    fetched.extend(report.fetched)
    cached.extend(report.cached)
    no_coverage.extend(report.no_coverage)
    errors.extend(report.errors)
    return TerrainFetchReport(fetched, cached, no_coverage, errors)


def populate_terrain_cache(
    region: object,
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    **opts,
) -> TerrainFetchReport:
    """Populate the terrain cache for a bounding box or explicit tile iterable."""
    if isinstance(region, Mapping):
        return prefetch_dted_bbox(
            region["min_lat"],
            region["min_lon"],
            region["max_lat"],
            region["max_lon"],
            cache_dir=cache_dir,
            offline=offline,
            **opts,
        )
    if (
        isinstance(region, (list, tuple))
        and len(region) == 4
        and all(isinstance(value, (int, float)) for value in region)
    ):
        min_lat, min_lon, max_lat, max_lon = region
        return prefetch_dted_bbox(
            float(min_lat),
            float(min_lon),
            float(max_lat),
            float(max_lon),
            cache_dir=cache_dir,
            offline=offline,
            **opts,
        )
    return prefetch_dted_tiles(region, cache_dir=cache_dir, offline=offline, **opts)


def _prefetch_dted_tile_indices(
    tiles: Iterable[tuple[int, int]],
    *,
    cache_dir: Optional[str],
    offline: bool,
    **opts,
) -> TerrainFetchReport:
    fetched: list[str] = []
    cached: list[str] = []
    no_coverage: list[str] = []
    errors: list[tuple[str, DataError]] = []
    seen: set[tuple[int, int]] = set()
    strict = bool(opts.pop("strict", False))

    for lat_index, lon_index in tiles:
        if (lat_index, lon_index) in seen:
            continue
        seen.add((lat_index, lon_index))
        try:
            tile_id = skadi_tile_id(lat_index, lon_index)
            state, value = _fetch_dted_tile(
                lat_index,
                lon_index,
                cache_dir=cache_dir,
                offline=offline,
                sha256=opts.get("sha256"),
                strict=strict,
                timeout_s=opts.get("timeout_s", _DEFAULT_TIMEOUT_S),
                retries=opts.get("retries", _DEFAULT_RETRIES),
                backoff_s=opts.get("backoff_s", _DEFAULT_BACKOFF_S),
                max_compressed_bytes=opts.get(
                    "max_compressed_bytes", _DEFAULT_MAX_COMPRESSED_BYTES
                ),
                max_decompressed_bytes=opts.get(
                    "max_decompressed_bytes", _DEFAULT_MAX_DECOMPRESSED_BYTES
                ),
            )
        except NoCoverage as exc:
            no_coverage.append(exc.tile_id)
        except DataError as exc:
            key = _tile_error_key(lat_index, lon_index)
            errors.append((key, exc))
            continue
        if state == "fetched":
            fetched.append(str(value))
        elif state == "cached":
            cached.append(str(value))
        elif state == "no_coverage":
            no_coverage.append(str(value))
        else:
            errors.append((tile_id, UnsupportedProduct(f"unexpected state: {state}")))

    return TerrainFetchReport(fetched, cached, no_coverage, errors)


def _coerce_tile_entry(entry: object) -> tuple[int, int]:
    if isinstance(entry, str):
        return parse_skadi_tile_id(entry)
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        try:
            lat_index = int(entry[0])
            lon_index = int(entry[1])
        except (TypeError, ValueError) as exc:
            raise InvalidTileIndex(f"invalid terrain tile entry: {entry!r}") from exc
        skadi_tile_id(lat_index, lon_index)
        return lat_index, lon_index
    raise InvalidTileIndex(f"invalid terrain tile entry: {entry!r}")


def _tile_error_key(lat_index: int, lon_index: int) -> str:
    try:
        return skadi_tile_id(lat_index, lon_index)
    except DataError:
        return f"{lat_index},{lon_index}"


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
    predicted maps are published ahead of their target day). Every candidate
    uses the exact-identity acquisition path, including semantic date/cadence
    validation and source-specific cache isolation. Raises the last absence
    when every explicitly permitted lookback candidate misses.
    """
    from sidereon import distribution

    dates = _gim_date_candidates(center, target, lookback)
    last_error: Optional[DataError] = None
    acquire_opts = dict(fetch_opts)
    if "max_compressed_bytes" in acquire_opts:
        acquire_opts["max_archive_bytes"] = acquire_opts.pop("max_compressed_bytes")
    if "max_decompressed_bytes" in acquire_opts:
        acquire_opts["max_product_bytes"] = acquire_opts.pop("max_decompressed_bytes")
    for date in dates:
        prod = product(center, "ionex", date, sample)
        exact = distribution.request(prod, [distribution.Distribution.direct()])
        try:
            acquired = distribution.acquire(
                exact,
                cache_dir=cache_dir,
                offline=offline,
                **acquire_opts,
            )
        except (distribution.ProductNotPublished, OfflineCacheMiss) as exc:
            # This API explicitly permits lookback. Integrity, cache, and
            # transport failures remain terminal instead of becoming absence.
            last_error = exc
            continue
        return sidereon.load_ionex(acquired.path)
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
    pattern: Optional[str] = None
    url: Optional[str] = None
    http_status: Optional[int] = None

    def to_dict(self) -> dict:
        """Return the secret-free public absence record."""
        return {
            "center": self.center,
            "filename": self.filename,
            "reason": self.reason,
            "pattern": self.pattern,
            "url": self.url,
            "http_status": self.http_status,
        }


_PRODUCT_IDENTITY_FIELDS = {
    "family",
    "analysis_center",
    "publisher",
    "solution_class",
    "campaign",
    "filename_version",
    "date",
    "issue",
    "span",
    "sample",
    "official_filename",
    "format",
    "format_version",
    "prediction_horizon_days",
}
_ARTIFACT_IDENTITY_FIELDS = {
    "schema_version",
    "requested_identity",
    "resolved_identity",
    "distribution_source",
    "official_filename",
    "product_sha256",
    "product_byte_length",
    "archive_sha256",
    "archive_byte_length",
    "compression",
}
_ACQUISITION_FACTS_FIELDS = {
    "schema_version",
    "retrieved_at",
    "cache_hit",
    "original_url",
    "final_url",
    "etag",
    "last_modified",
    "attempts",
}
_SOURCE_FAILURE_FIELDS = {"source", "error_type", "message", "url", "status"}


def _exact_fields(
    value: object, expected: set[str], description: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{description} must be a string-keyed mapping")
    if set(value) != expected:
        raise ValueError(f"invalid {description} fields")
    return value


def _exact_str(value: object, description: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise ValueError(f"{description} must be a string")
    return value


def _optional_exact_str(value: object, description: str) -> Optional[str]:
    if value is None:
        return None
    return _exact_str(value, description)


def _exact_int(value: object, description: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{description} must be an integer")
    return value


def _exact_float(
    value: object, description: str, *, nonnegative: bool = False
) -> float:
    if type(value) is not float or not _math.isfinite(value):
        raise ValueError(f"{description} must be a finite float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{description} must be non-negative")
    return value


def _exact_schema_version(value: object, description: str) -> None:
    if type(value) is not int or value != 1:
        raise ValueError(f"unsupported {description} schema version")


def _exact_date(value: object, description: str) -> _dt.date:
    text = _exact_str(value, description)
    parsed = _dt.date.fromisoformat(text)
    if parsed.isoformat() != text:
        raise ValueError(f"{description} is not canonical ISO-8601")
    return parsed


def _exact_product_identity(value: object) -> "ProductIdentity":
    from sidereon import distribution

    fields = _exact_fields(value, _PRODUCT_IDENTITY_FIELDS, "product identity")
    for name in (
        "family",
        "analysis_center",
        "publisher",
        "solution_class",
        "campaign",
        "issue",
        "span",
        "sample",
        "official_filename",
        "format",
    ):
        _exact_str(fields[name], f"product identity {name}")
    _exact_int(fields["filename_version"], "product identity filename_version")
    _exact_date(fields["date"], "product identity date")
    _optional_exact_str(fields["format_version"], "product identity format_version")
    horizon = fields["prediction_horizon_days"]
    if horizon is not None:
        _exact_int(horizon, "product identity prediction_horizon_days")
    return distribution.ProductIdentity.from_dict(fields)


def _exact_public_url(value: object, description: str) -> Optional[str]:
    from sidereon import distribution

    result = _optional_exact_str(value, description)
    if result is None:
        return None
    if result != distribution._sanitize_url(result) or not result.startswith(
        ("https://", "http://")
    ):
        raise ValueError(f"{description} must be a sanitized public URL")
    return result


def _exact_source_failure(value: object) -> "SourceFailure":
    from sidereon import distribution

    fields = _exact_fields(value, _SOURCE_FAILURE_FIELDS, "source failure")
    source = distribution.DistributionSource(
        _exact_str(fields["source"], "source failure source")
    )
    error_type = _exact_str(fields["error_type"], "source failure error_type")
    message = _exact_str(fields["message"], "source failure message")
    url = _exact_public_url(fields["url"], "source failure URL")
    status_value = fields["status"]
    status = (
        None
        if status_value is None
        else _exact_int(status_value, "source failure status", minimum=0)
    )
    return distribution.SourceFailure(source, error_type, message, url, status)


@dataclass(frozen=True)
class ArtifactIdentity:
    """Stable identity of one exact, verified SP3 artifact.

    Only reproducible fields belong here. Retrieval timestamps, HTTP metadata,
    cache status, failures, credentials, and local paths are deliberately kept
    in :class:`AcquisitionFacts` or omitted entirely.
    """

    requested_identity: "ProductIdentity"
    resolved_identity: "ProductIdentity"
    distribution_source: "DistributionSource"
    official_filename: str
    product_sha256: str
    product_byte_length: int
    archive_sha256: str
    archive_byte_length: int
    compression: str

    @classmethod
    def _from_provenance(
        cls, provenance: "AcquisitionProvenance"
    ) -> "ArtifactIdentity":
        return cls(
            requested_identity=provenance.requested_identity,
            resolved_identity=provenance.resolved_identity,
            distribution_source=provenance.distribution_source,
            official_filename=provenance.official_filename,
            product_sha256=provenance.sha256,
            product_byte_length=provenance.byte_length,
            archive_sha256=provenance.archive_sha256,
            archive_byte_length=provenance.archive_byte_length,
            compression=provenance.archive_compression,
        )

    def to_dict(self) -> dict:
        """Return the complete, secret-free reproducible identity."""
        return {
            "schema_version": 1,
            "requested_identity": self.requested_identity.to_dict(),
            "resolved_identity": self.resolved_identity.to_dict(),
            "distribution_source": self.distribution_source.value,
            "official_filename": self.official_filename,
            "product_sha256": self.product_sha256,
            "product_byte_length": self.product_byte_length,
            "archive_sha256": self.archive_sha256,
            "archive_byte_length": self.archive_byte_length,
            "compression": self.compression,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ArtifactIdentity":
        """Restore a persisted artifact identity for core verification."""
        from sidereon import distribution

        fields = _exact_fields(value, _ARTIFACT_IDENTITY_FIELDS, "artifact identity")
        _exact_schema_version(fields["schema_version"], "artifact identity")
        return cls(
            requested_identity=_exact_product_identity(fields["requested_identity"]),
            resolved_identity=_exact_product_identity(fields["resolved_identity"]),
            distribution_source=distribution.DistributionSource(
                _exact_str(fields["distribution_source"], "distribution source")
            ),
            official_filename=_exact_str(
                fields["official_filename"], "artifact official filename"
            ),
            product_sha256=_exact_str(
                fields["product_sha256"], "artifact product SHA-256"
            ),
            product_byte_length=_exact_int(
                fields["product_byte_length"],
                "artifact product byte length",
                minimum=1,
            ),
            archive_sha256=_exact_str(
                fields["archive_sha256"], "artifact archive SHA-256"
            ),
            archive_byte_length=_exact_int(
                fields["archive_byte_length"],
                "artifact archive byte length",
                minimum=1,
            ),
            compression=_exact_str(fields["compression"], "artifact compression"),
        )


@dataclass(frozen=True)
class AcquisitionFacts:
    """Secret-free observations about how one exact artifact was acquired."""

    retrieved_at: str
    cache_hit: bool
    original_url: Optional[str]
    final_url: Optional[str]
    etag: Optional[str]
    last_modified: Optional[str]
    attempts: tuple["SourceFailure", ...] = ()

    @classmethod
    def _from_provenance(
        cls, provenance: "AcquisitionProvenance"
    ) -> "AcquisitionFacts":
        return cls(
            retrieved_at=provenance.retrieved_at,
            cache_hit=provenance.cache_hit,
            original_url=provenance.original_url,
            final_url=provenance.final_url,
            etag=provenance.etag,
            last_modified=provenance.last_modified,
            attempts=provenance.attempts,
        )

    def to_dict(self) -> dict:
        """Return observations without credentials, cookies, or local paths."""
        return {
            "schema_version": 1,
            "retrieved_at": self.retrieved_at,
            "cache_hit": self.cache_hit,
            "original_url": self.original_url,
            "final_url": self.final_url,
            "etag": self.etag,
            "last_modified": self.last_modified,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AcquisitionFacts":
        """Restore persisted secret-free acquisition observations."""
        fields = _exact_fields(value, _ACQUISITION_FACTS_FIELDS, "acquisition facts")
        _exact_schema_version(fields["schema_version"], "acquisition facts")
        attempts = fields["attempts"]
        if type(attempts) is not list:
            raise ValueError("acquisition attempts must be a list")
        cache_hit = fields["cache_hit"]
        if type(cache_hit) is not bool:
            raise ValueError("acquisition cache_hit must be a boolean")
        retrieved_at = _exact_str(fields["retrieved_at"], "retrieval time")
        retrieved = _dt.datetime.fromisoformat(retrieved_at)
        if retrieved.tzinfo is None or retrieved.isoformat() != retrieved_at:
            raise ValueError("retrieval time must be canonical ISO-8601 with an offset")
        return cls(
            retrieved_at=retrieved_at,
            cache_hit=cache_hit,
            original_url=_exact_public_url(fields["original_url"], "original URL"),
            final_url=_exact_public_url(fields["final_url"], "final URL"),
            etag=_optional_exact_str(fields["etag"], "ETag"),
            last_modified=_optional_exact_str(fields["last_modified"], "Last-Modified"),
            attempts=tuple(_exact_source_failure(attempt) for attempt in attempts),
        )


@dataclass(frozen=True)
class Sp3MergeInputIdentity:
    """Complete canonical identity returned by the shared Rust core.

    Iteration yields ``(schema_version, stable_id)`` to preserve compatibility
    with the original two-value return contract.
    """

    schema_version: int
    stable_id: str
    canonical_contributors: tuple[ArtifactIdentity, ...]
    precedence_contributors: Optional[tuple[ArtifactIdentity, ...]]

    def __iter__(self) -> Iterator[Union[int, str]]:
        yield self.schema_version
        yield self.stable_id


@dataclass(frozen=True)
class Contributor:
    """A center that contributed an SP3 product to a merged fetch."""

    center: str
    filename: str
    date: _dt.date
    issue: Optional[str]
    pattern: Optional[str] = None
    artifact_identity: Optional[ArtifactIdentity] = None
    acquisition_facts: Optional[AcquisitionFacts] = None

    def to_dict(self) -> dict:
        """Return a complete, portable contributor record."""
        if self.artifact_identity is None or self.acquisition_facts is None:
            raise ValueError("SP3 contributor provenance is incomplete")
        return {
            "center": self.center,
            "filename": self.filename,
            "date": self.date.isoformat(),
            "issue": self.issue,
            "pattern": self.pattern,
            "artifact_identity": self.artifact_identity.to_dict(),
            "acquisition_facts": self.acquisition_facts.to_dict(),
        }


def _merge_flag_to_dict(flag: "sidereon.Sp3MergeFlag") -> dict:
    return {
        "satellite": flag.satellite,
        "jd_whole": flag.jd_whole,
        "jd_fraction": flag.jd_fraction,
        "sources": flag.sources,
    }


def _transform_to_dict(
    transform: Optional[tuple[list[float], float, list[float]]],
    *,
    rates: bool,
) -> Optional[dict]:
    if transform is None:
        return None
    translation, scale, rotation = transform
    if rates:
        return {
            "translation_mm_per_year": translation,
            "scale_ppb_per_year": scale,
            "rotation_mas_per_year": rotation,
        }
    return {
        "translation_mm": translation,
        "scale_ppb": scale,
        "rotation_mas": rotation,
    }


def _frame_reconciliation_to_dict(
    reconciliation: "sidereon.Sp3FrameReconciliation",
) -> dict:
    span = reconciliation.epoch_year_span
    return {
        "source_index": reconciliation.source_index,
        "source_label": reconciliation.source_label,
        "target_label": reconciliation.target_label,
        "method": reconciliation.method,
        "asserted_label_set": reconciliation.asserted_label_set,
        "source_frame": reconciliation.source_frame,
        "target_frame": reconciliation.target_frame,
        "catalog_source_frame": reconciliation.catalog_source_frame,
        "catalog_target_frame": reconciliation.catalog_target_frame,
        "catalog_inverse": reconciliation.catalog_inverse,
        "reference_epoch_year": reconciliation.reference_epoch_year,
        "parameters": _transform_to_dict(reconciliation.parameters, rates=False),
        "rates": _transform_to_dict(reconciliation.rates, rates=True),
        "provenance": reconciliation.provenance,
        "epoch_year_span": None if span is None else list(span),
        "records_affected": reconciliation.records_affected,
        "identity": reconciliation.identity,
    }


def _agreement_cell_to_dict(metric: "sidereon.Sp3AgreementMetric") -> dict:
    return {
        "satellite": metric.satellite,
        "jd_whole": metric.jd_whole,
        "jd_fraction": metric.jd_fraction,
        "position_members": metric.position_members,
        "position_rms_m": metric.position_rms_m,
        "position_max_m": metric.position_max_m,
        "clock_members": metric.clock_members,
        "clock_rms_s": metric.clock_rms_s,
        "clock_max_s": metric.clock_max_s,
    }


def _agreement_epoch_to_dict(epoch: "sidereon.Sp3EpochAgreement") -> dict:
    return {
        "jd_whole": epoch.jd_whole,
        "jd_fraction": epoch.jd_fraction,
        "satellites": epoch.satellites,
        "position_rms_m": epoch.position_rms_m,
        "position_max_m": epoch.position_max_m,
        "clock_rms_s": epoch.clock_rms_s,
        "clock_max_s": epoch.clock_max_s,
    }


def _merge_result_to_dict(report: "sidereon.Sp3MergeReport") -> dict:
    return {
        "frame_reconciliations": [
            _frame_reconciliation_to_dict(item) for item in report.frame_reconciliations
        ],
        "quarantined": [_merge_flag_to_dict(item) for item in report.quarantined],
        "single_source": [_merge_flag_to_dict(item) for item in report.single_source],
        "position_outliers": [
            _merge_flag_to_dict(item) for item in report.position_outliers
        ],
        "clock_outliers": [_merge_flag_to_dict(item) for item in report.clock_outliers],
        "agreement": {
            "position_rms_m": report.position_agreement_rms_m,
            "position_max_m": report.position_agreement_max_m,
            "clock_rms_s": report.clock_agreement_rms_s,
            "clock_max_s": report.clock_agreement_max_s,
            "cells": [_agreement_cell_to_dict(item) for item in report.agreement],
            "epochs": [
                _agreement_epoch_to_dict(item) for item in report.agreement_epochs
            ],
        },
    }


@dataclass
class MergeReport:
    """Audit report for a merged SP3 fetch.

    Carries the per-center contribution audit plus the binding's own SP3 merge
    report (``merge_report``). Single contributors deliberately follow the same
    merge path so every supplied merge option is applied consistently.
    """

    contributors: list[Contributor]
    absent: list[AbsentCenter]
    source_count: int
    single_product: bool
    merged: bool
    merge_report: Optional["sidereon.Sp3MergeReport"] = field(default=None)
    stable_input_identity: Optional[str] = None
    input_identity_schema_version: Optional[int] = None
    merge_policy: Optional[dict] = None
    requested_centers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return the public merge acquisition report as JSON-safe values."""
        if (
            self.stable_input_identity is None
            or self.input_identity_schema_version is None
            or self.merge_policy is None
        ):
            raise ValueError("merged-SP3 input identity is incomplete")
        if self.merge_report is None:
            raise ValueError("merged-SP3 result audit is incomplete")
        return {
            "schema_version": 1,
            "contributors": [
                contributor.to_dict() for contributor in self.contributors
            ],
            "absent": [center.to_dict() for center in self.absent],
            "requested_centers": list(self.requested_centers),
            "source_count": self.source_count,
            "single_product": self.single_product,
            "merged": self.merged,
            "stable_input_identity": self.stable_input_identity,
            "input_identity_schema_version": self.input_identity_schema_version,
            "merge_policy": self.merge_policy,
            "merge_report": _merge_result_to_dict(self.merge_report),
        }


def _ultra_center(center: str) -> bool:
    return bool(_center_def(center)["issues"])


def _sp3_candidates(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    sample: Optional[str],
) -> list[Product]:
    cdef = _center_def(center)
    if "sp3" not in cdef["products"]:
        raise UnsupportedProduct(f"{center} does not serve sp3")
    date = _as_date(target)
    eff_sample = (
        sample if sample is not None else default_sample_for_date(center, "sp3", date)
    )

    if _ultra_center(center) and isinstance(target, _dt.datetime):
        candidates = _ultra_issue_candidates(center, _as_naive_datetime(target))
        return [
            product
            for date, issue in candidates
            for product in _sp3_products_for_issue(center, date, issue, sample)
        ]
    if _ultra_center(center):
        return _sp3_products_for_issue(center, date, "0000", sample)
    return [Product(center, "sp3", date, eff_sample)]


def _sp3_products_for_issue(
    center: str,
    date: _dt.date,
    issue: str,
    requested_sample: Optional[str],
) -> list[Product]:
    if requested_sample is not None:
        return [
            Product(
                center,
                "sp3",
                date,
                requested_sample,
                issue,
                pattern="requested_sample",
            )
        ]
    try:
        locations = _core_data_ultra_sp3_locations(
            center, date.year, date.month, date.day, issue
        )
    except ValueError as exc:
        raise _catalog_error(exc) from None
    return [
        Product(
            center=center,
            content="sp3",
            date=date,
            sample=candidate_sample,
            issue=issue,
            span=span,
            pattern=pattern,
            filename=filename,
            url=url,
            compression=compression,
        )
        for pattern, span, candidate_sample, filename, url, compression in locations
    ]


def _fetch_center_sp3(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    sample: Optional[str],
    fetch_kwargs: dict,
):
    from sidereon import distribution

    # Unsupported center/product combinations are caller configuration errors,
    # not publication absence, and must fail before any acquisition attempt.
    candidates = _sp3_candidates(center, target, sample)

    last: Optional[tuple[Product, str, DataError]] = None
    candidate_attempts = []
    for prod in candidates:
        filename = prod.canonical_filename()
        try:
            acquired = distribution._acquire_catalog_product(prod, **fetch_kwargs)
        except (distribution.ProductNotPublished, OfflineCacheMiss) as exc:
            # Expected absence for this candidate; try the next. Integrity,
            # cache, and transport failures are real and propagate instead of
            # being silently recorded as an absent center.
            last = (prod, filename, exc)
            candidate_attempts.append(
                distribution.SourceFailure(
                    source=distribution.DistributionSource.DIRECT,
                    error_type=getattr(exc, "code", type(exc).__name__),
                    message=str(exc),
                    url=getattr(exc, "url", None),
                    status=getattr(exc, "status", None),
                )
            )
            continue
        sp3 = sidereon.load_sp3(acquired.path)
        artifact_identity = ArtifactIdentity._from_provenance(acquired.provenance)
        acquisition_facts = AcquisitionFacts._from_provenance(acquired.provenance)
        acquisition_facts = replace(
            acquisition_facts,
            attempts=tuple(candidate_attempts) + acquisition_facts.attempts,
        )
        return (
            "ok",
            Contributor(
                center,
                filename,
                prod.date,
                prod.issue,
                prod.pattern or "canonical",
                artifact_identity,
                acquisition_facts,
            ),
            sp3,
        )
    if last is not None:
        return (
            "absent",
            AbsentCenter(
                center,
                last[1],
                _reason_str(last[2]),
                last[0].pattern,
                last[0].archive_url(),
                getattr(last[2], "status", None),
            ),
        )
    return ("absent", AbsentCenter(center, None, "no_candidate"))


def _reason_str(exc: DataError) -> str:
    if isinstance(exc, OfflineCacheMiss):
        return "offline_miss"
    if isinstance(exc, FileNotFoundOnArchive) or getattr(exc, "code", None) == (
        "product_not_published"
    ):
        return "candidate_not_found"
    if isinstance(exc, ChecksumMismatch):
        return "checksum"
    if isinstance(exc, HttpStatusError):
        return f"http_status:{exc.status}"
    return type(exc).__name__


def sp3_merge_input_identity(
    contributors: Sequence[ArtifactIdentity],
    merge_options: Optional["sidereon.Sp3MergeOptions"] = None,
) -> Sp3MergeInputIdentity:
    """Return the complete canonical identity for exact inputs and policy.

    Contributor and mapping insertion order do not affect mean or median merge
    identities. Contributor order is semantic source priority for precedence
    merges and therefore does affect their identity. Observational acquisition
    facts are not accepted and cannot perturb the identity. The shared Rust core
    validates every complete artifact record and fails closed on malformed or
    inconsistent provenance.
    """
    artifacts = list(contributors)
    if not all(isinstance(item, ArtifactIdentity) for item in artifacts):
        raise TypeError("contributors must contain ArtifactIdentity values")
    encoded = [
        _json.dumps(
            item.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        for item in artifacts
    ]
    schema_version, stable_id, canonical_indices, precedence_indices = (
        _core_sp3_merge_input_identity(encoded, merge_options)
    )
    return Sp3MergeInputIdentity(
        schema_version=schema_version,
        stable_id=stable_id,
        canonical_contributors=tuple(artifacts[index] for index in canonical_indices),
        precedence_contributors=(
            None
            if precedence_indices is None
            else tuple(artifacts[index] for index in precedence_indices)
        ),
    )


def _merge_options_from_policy(value: Mapping[str, object]):
    expected_fields = {
        "schema_version",
        "position_tolerance_m",
        "clock_tolerance_s",
        "min_agree",
        "clock_min_common",
        "combine",
        "precedence_scope",
        "outlier_reject",
        "target_epoch_interval_s",
        "systems",
        "asserted_frame_label_sets",
        "helmert",
        "precedence_artifact_sha256",
    }
    fields = _exact_fields(value, expected_fields, "merged-SP3 policy")
    _exact_schema_version(fields["schema_version"], "merged-SP3 policy")
    position_tolerance_m = _exact_float(
        fields["position_tolerance_m"],
        "merged-SP3 position tolerance",
        nonnegative=True,
    )
    clock_tolerance_s = _exact_float(
        fields["clock_tolerance_s"],
        "merged-SP3 clock tolerance",
        nonnegative=True,
    )
    min_agree = _exact_int(fields["min_agree"], "merged-SP3 min_agree", minimum=1)
    clock_min_common = _exact_int(
        fields["clock_min_common"],
        "merged-SP3 clock_min_common",
        minimum=1,
    )
    combine = _exact_str(fields["combine"], "merged-SP3 combine")
    if combine not in {"mean", "median", "precedence"}:
        raise ValueError("invalid merged-SP3 combine")
    precedence_scope = _exact_str(
        fields["precedence_scope"], "merged-SP3 precedence_scope"
    )
    if precedence_scope not in {"cell", "satellite_arc"}:
        raise ValueError("invalid merged-SP3 precedence_scope")
    outlier_value = fields["outlier_reject"]
    outlier = None
    if outlier_value is not None:
        outlier_fields = _exact_fields(
            outlier_value,
            {"position_tolerance_m", "clock_tolerance_s"},
            "merged-SP3 outlier policy",
        )
        outlier = sidereon.Sp3OutlierRejectOptions(
            _exact_float(
                outlier_fields["position_tolerance_m"],
                "merged-SP3 outlier position tolerance",
                nonnegative=True,
            ),
            _exact_float(
                outlier_fields["clock_tolerance_s"],
                "merged-SP3 outlier clock tolerance",
                nonnegative=True,
            ),
        )
    target_interval = fields["target_epoch_interval_s"]
    if target_interval is not None:
        target_interval = _exact_float(
            target_interval, "merged-SP3 target epoch interval"
        )
        if (
            target_interval < 1.0
            or abs(target_interval - round(target_interval)) > 1e-6
        ):
            raise ValueError("invalid merged-SP3 target epoch interval")
    systems = fields["systems"]
    if systems is not None:
        if type(systems) is not list or not systems:
            raise ValueError("invalid merged-SP3 systems policy")
        if any(
            type(system) is not str or system not in {"G", "R", "E", "C", "J", "I", "S"}
            for system in systems
        ):
            raise ValueError("invalid merged-SP3 systems policy")
        if systems != sorted(set(systems)):
            raise ValueError("merged-SP3 systems policy is not canonical")
    frame_sets = fields["asserted_frame_label_sets"]
    if type(frame_sets) is not list:
        raise ValueError("invalid merged-SP3 frame policy")
    normalized_frame_sets = []
    for labels in frame_sets:
        if type(labels) is not list or len(labels) < 2:
            raise ValueError("invalid merged-SP3 frame policy")
        if any(
            type(label) is not str or not label or label != label.strip()
            for label in labels
        ):
            raise ValueError("invalid merged-SP3 frame policy")
        canonical_labels = sorted(set(labels))
        if len(canonical_labels) < 2 or labels != canonical_labels:
            raise ValueError("merged-SP3 frame policy is not canonical")
        normalized_frame_sets.append(canonical_labels)
    if frame_sets != sorted(normalized_frame_sets):
        raise ValueError("merged-SP3 frame policy is not canonical")
    helmert = fields["helmert"]
    if type(helmert) is not bool:
        raise ValueError("invalid merged-SP3 Helmert policy")
    precedence = fields["precedence_artifact_sha256"]
    if type(precedence) is not list or any(
        type(digest) is not str for digest in precedence
    ):
        raise ValueError("invalid merged-SP3 precedence contributor policy")
    return sidereon.Sp3MergeOptions(
        position_tolerance_m=position_tolerance_m,
        clock_tolerance_s=clock_tolerance_s,
        min_agree=min_agree,
        clock_min_common=clock_min_common,
        combine=combine,
        precedence_scope=precedence_scope,
        outlier_reject=outlier,
        target_epoch_interval_s=target_interval,
        systems=systems,
        asserted_frame_label_sets=frame_sets,
        helmert=helmert,
    )


_MERGE_REPORT_FIELDS = {
    "schema_version",
    "contributors",
    "absent",
    "requested_centers",
    "source_count",
    "single_product",
    "merged",
    "stable_input_identity",
    "input_identity_schema_version",
    "merge_policy",
    "merge_report",
}
_CONTRIBUTOR_FIELDS = {
    "center",
    "filename",
    "date",
    "issue",
    "pattern",
    "artifact_identity",
    "acquisition_facts",
}
_ABSENT_CENTER_FIELDS = {
    "center",
    "filename",
    "reason",
    "pattern",
    "url",
    "http_status",
}
_MERGE_RESULT_FIELDS = {
    "frame_reconciliations",
    "quarantined",
    "single_source",
    "position_outliers",
    "clock_outliers",
    "agreement",
}
_MERGE_FLAG_FIELDS = {"satellite", "jd_whole", "jd_fraction", "sources"}
_AGREEMENT_FIELDS = {
    "position_rms_m",
    "position_max_m",
    "clock_rms_s",
    "clock_max_s",
    "cells",
    "epochs",
}
_AGREEMENT_CELL_FIELDS = {
    "satellite",
    "jd_whole",
    "jd_fraction",
    "position_members",
    "position_rms_m",
    "position_max_m",
    "clock_members",
    "clock_rms_s",
    "clock_max_s",
}
_AGREEMENT_EPOCH_FIELDS = {
    "jd_whole",
    "jd_fraction",
    "satellites",
    "position_rms_m",
    "position_max_m",
    "clock_rms_s",
    "clock_max_s",
}
_FRAME_RECONCILIATION_FIELDS = {
    "source_index",
    "source_label",
    "target_label",
    "method",
    "asserted_label_set",
    "source_frame",
    "target_frame",
    "catalog_source_frame",
    "catalog_target_frame",
    "catalog_inverse",
    "reference_epoch_year",
    "parameters",
    "rates",
    "provenance",
    "epoch_year_span",
    "records_affected",
    "identity",
}
_HELMERT_PARAMETER_FIELDS = {"translation_mm", "scale_ppb", "rotation_mas"}
_HELMERT_RATE_FIELDS = {
    "translation_mm_per_year",
    "scale_ppb_per_year",
    "rotation_mas_per_year",
}
_SYSTEM_ORDER = {"G": 0, "R": 1, "E": 2, "C": 3, "J": 4, "I": 5, "S": 6}
_SATELLITE_PRN_RANGES = {
    "G": (1, 32),
    "R": (1, 27),
    "E": (1, 36),
    "C": (1, 63),
    "J": (1, 9),
    "I": (1, 14),
    "S": (20, 58),
}
_TERRESTRIAL_FRAMES = {"ITRF2020", "ITRF2014", "ITRF2008"}


def _contributor_matches_catalog(
    contributor: Mapping[str, object], artifact: ArtifactIdentity
) -> bool:
    from sidereon import distribution

    requested = artifact.requested_identity
    center = _exact_str(contributor["center"], "contributor center")
    filename = _exact_str(contributor["filename"], "contributor filename")
    date = _exact_date(contributor["date"], "contributor date")
    issue = contributor["issue"]
    if issue is not None:
        issue = _exact_str(issue, "contributor issue")
    pattern = _exact_str(contributor["pattern"], "contributor pattern")
    if (
        center != requested.analysis_center
        or date != requested.date
        or ("0000" if issue is None else issue) != requested.issue
    ):
        return False

    if pattern in {"canonical", "requested_sample"}:
        if pattern == "canonical" and _ultra_center(center):
            return False
        if pattern == "requested_sample" and not _ultra_center(center):
            return False
        product = Product(
            center,
            "sp3",
            date,
            requested.sample,
            issue=issue,
            pattern=None if pattern == "canonical" else pattern,
        )
        return (
            filename == product.canonical_filename()
            and distribution._catalog_product_identity(product) == requested
        )

    if issue is None:
        return False
    return any(
        candidate.pattern == pattern
        and candidate.canonical_filename() == filename
        and distribution._catalog_product_identity(candidate) == requested
        for candidate in _sp3_products_for_issue(center, date, issue, None)
    )


def _validate_absent_center(value: object) -> str:
    fields = _exact_fields(value, _ABSENT_CENTER_FIELDS, "absent center")
    center = _exact_str(fields["center"], "absent center code")
    _center_def(center)
    filename = _optional_exact_str(fields["filename"], "absent center filename")
    pattern = _optional_exact_str(fields["pattern"], "absent center pattern")
    url = _exact_public_url(fields["url"], "absent center URL")
    _exact_str(fields["reason"], "absent center reason")
    status = fields["http_status"]
    if status is not None:
        _exact_int(status, "absent center HTTP status", minimum=0)
    if (filename is None) != (url is None) or (
        filename is None and pattern is not None
    ):
        raise ValueError("absent center candidate metadata is incomplete")
    if filename is not None and ("/" in filename or "\\" in filename):
        raise ValueError("absent center filename must not be a path")
    return center


def _optional_nonnegative_float(value: object, description: str) -> Optional[float]:
    if value is None:
        return None
    return _exact_float(value, description, nonnegative=True)


def _satellite_key(value: object) -> tuple[int, int]:
    satellite = _exact_str(value, "SP3 merge satellite")
    if (
        len(satellite) != 3
        or satellite[0] not in _SATELLITE_PRN_RANGES
        or not satellite[1:].isdigit()
    ):
        raise ValueError("invalid SP3 merge satellite")
    prn = int(satellite[1:])
    minimum, maximum = _SATELLITE_PRN_RANGES[satellite[0]]
    if not minimum <= prn <= maximum:
        raise ValueError("invalid SP3 merge satellite")
    return (_SYSTEM_ORDER[satellite[0]], prn)


def _epoch_key(value: Mapping[str, object]) -> tuple[float, float]:
    return (value["jd_whole"], value["jd_fraction"])


def _cell_key(value: Mapping[str, object]) -> tuple[float, float, int, int]:
    system, prn = _satellite_key(value["satellite"])
    return (value["jd_whole"], value["jd_fraction"], system, prn)


def _strictly_ascending(values: Sequence[object]) -> bool:
    return all(first < second for first, second in zip(values, values[1:]))


def _validate_epoch(fields: Mapping[str, object], description: str) -> None:
    jd_whole = _exact_float(fields["jd_whole"], f"{description} jd_whole")
    fraction = _exact_float(fields["jd_fraction"], f"{description} jd_fraction")
    if (
        not 1_721_059.5 <= jd_whole <= 5_373_483.5
        or jd_whole - _math.floor(jd_whole) != 0.5
    ):
        raise ValueError(f"{description} jd_whole is not canonical")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{description} jd_fraction is not canonical")
    if fraction == 1.0:
        try:
            before = sidereon.timescale_offset_at(
                sidereon.TimeScale.UTC, sidereon.TimeScale.TAI, jd_whole
            )
            after = sidereon.timescale_offset_at(
                sidereon.TimeScale.UTC, sidereon.TimeScale.TAI, jd_whole + 1.0
            )
        except (TypeError, ValueError):
            raise ValueError(f"{description} jd_fraction is not canonical") from None
        if after - before != 1.0:
            raise ValueError(f"{description} jd_fraction is not canonical")


def _validate_source_indices(
    value: object, source_count: int, description: str
) -> list[int]:
    if type(value) is not list or not value:
        raise ValueError(f"{description} must be a non-empty list")
    for source in value:
        _exact_int(source, description)
        if source >= source_count:
            raise ValueError(f"{description} contains an unknown source")
    if not _strictly_ascending(value):
        raise ValueError(f"{description} is not strictly ascending")
    return value


def _validate_flags(
    value: object, source_count: int, description: str
) -> list[Mapping[str, object]]:
    if type(value) is not list:
        raise ValueError(f"{description} must be a list")
    flags = []
    for index, item in enumerate(value):
        fields = _exact_fields(item, _MERGE_FLAG_FIELDS, f"{description} flag")
        _satellite_key(fields["satellite"])
        _validate_epoch(fields, f"{description} flag {index}")
        _validate_source_indices(
            fields["sources"], source_count, f"{description} flag sources"
        )
        flags.append(fields)
    if not _strictly_ascending([_cell_key(flag) for flag in flags]):
        raise ValueError(f"{description} flags are not ordered and unique")
    return flags


def _validate_metric_pair(
    rms_value: object, max_value: object, description: str
) -> tuple[float, float]:
    rms = _exact_float(rms_value, f"{description} RMS", nonnegative=True)
    maximum = _exact_float(max_value, f"{description} maximum", nonnegative=True)
    return rms, maximum


def _validate_optional_metric_pair(
    rms_value: object, max_value: object, description: str
) -> tuple[Optional[float], Optional[float]]:
    if rms_value is None and max_value is None:
        return None, None
    if rms_value is None or max_value is None:
        raise ValueError(f"{description} metrics must both be present or absent")
    return _validate_metric_pair(rms_value, max_value, description)


def _validate_member_metric_bound(
    rms: Optional[float],
    maximum: Optional[float],
    members: int,
    description: str,
    *,
    upper_terms: Optional[int] = None,
) -> None:
    if rms is None and maximum is None and members == 0:
        return
    if (
        rms is None
        or maximum is None
        or members <= 0
        or (upper_terms is not None and not 0 <= upper_terms <= members)
    ):
        raise ValueError(f"invalid {description}")
    terms = members if upper_terms is None else upper_terms
    square = maximum * maximum
    upper_sum = 0.0
    for _ in range(terms):
        upper_sum += square
    lower_rms = _math.sqrt(square / members)
    upper_rms = _math.sqrt(upper_sum / members)
    if (
        not _math.isfinite(lower_rms)
        or not _math.isfinite(upper_rms)
        or rms < lower_rms
        or rms > upper_rms
    ):
        raise ValueError(f"invalid {description}")


def _validate_agreement_cells(
    value: object, source_count: int
) -> list[Mapping[str, object]]:
    if type(value) is not list:
        raise ValueError("SP3 agreement cells must be a list")
    cells = []
    for index, item in enumerate(value):
        fields = _exact_fields(item, _AGREEMENT_CELL_FIELDS, "SP3 agreement cell")
        _satellite_key(fields["satellite"])
        _validate_epoch(fields, f"SP3 agreement cell {index}")
        position_members = _exact_int(
            fields["position_members"], "SP3 position member count", minimum=1
        )
        clock_members = _exact_int(fields["clock_members"], "SP3 clock member count")
        if position_members > source_count or clock_members > source_count:
            raise ValueError("SP3 agreement member count exceeds source count")
        position_rms, position_max = _validate_metric_pair(
            fields["position_rms_m"],
            fields["position_max_m"],
            "SP3 position agreement",
        )
        clock_rms, clock_max = _validate_optional_metric_pair(
            fields["clock_rms_s"],
            fields["clock_max_s"],
            "SP3 clock agreement",
        )
        _validate_member_metric_bound(
            position_rms,
            position_max,
            position_members,
            "SP3 position dispersion",
        )
        _validate_member_metric_bound(
            clock_rms,
            clock_max,
            clock_members,
            "SP3 clock dispersion",
        )
        if clock_members == 0 and (clock_rms is not None or clock_max is not None):
            raise ValueError("clockless SP3 agreement cell has clock metrics")
        if clock_members > 0 and (clock_rms is None or clock_max is None):
            raise ValueError("clocked SP3 agreement cell lacks clock metrics")
        if position_members == 1 and (position_rms != 0.0 or position_max != 0.0):
            raise ValueError("single-source SP3 position dispersion is nonzero")
        if clock_members == 1 and (clock_rms != 0.0 or clock_max != 0.0):
            raise ValueError("single-source SP3 clock dispersion is nonzero")
        cells.append(fields)
    if not _strictly_ascending([_cell_key(cell) for cell in cells]):
        raise ValueError("SP3 agreement cells are not ordered and unique")
    return cells


def _validate_agreement_epochs(value: object) -> list[Mapping[str, object]]:
    if type(value) is not list:
        raise ValueError("SP3 agreement epochs must be a list")
    epochs = []
    for index, item in enumerate(value):
        fields = _exact_fields(item, _AGREEMENT_EPOCH_FIELDS, "SP3 agreement epoch")
        _validate_epoch(fields, f"SP3 agreement epoch {index}")
        _exact_int(fields["satellites"], "SP3 agreement epoch satellite count")
        _validate_metric_pair(
            fields["position_rms_m"],
            fields["position_max_m"],
            "SP3 epoch position agreement",
        )
        _validate_optional_metric_pair(
            fields["clock_rms_s"],
            fields["clock_max_s"],
            "SP3 epoch clock agreement",
        )
        epochs.append(fields)
    if not _strictly_ascending([_epoch_key(epoch) for epoch in epochs]):
        raise ValueError("SP3 agreement epochs are not ordered and unique")
    return epochs


def _pooled_rms(metrics: Iterable[tuple[float, int]]) -> Optional[float]:
    sum_squares = 0.0
    members = 0
    for rms, count in metrics:
        # Keep the same simple ordered accumulation used by the Rust core.
        sum_squares += rms * rms * count
        if not _math.isfinite(sum_squares):
            raise ValueError("SP3 agreement arithmetic is not finite")
        members += count
    if members == 0:
        return None
    result = _math.sqrt(sum_squares / members)
    if not _math.isfinite(result):
        raise ValueError("SP3 agreement arithmetic is not finite")
    return result


def _agreement_aggregate(cells: Sequence[Mapping[str, object]]) -> dict:
    position_max = None
    clock_max = None
    for cell in cells:
        value = cell["position_max_m"]
        position_max = value if position_max is None else max(position_max, value)
        if cell["clock_members"] > 0:
            value = cell["clock_max_s"]
            clock_max = value if clock_max is None else max(clock_max, value)
    return {
        "position_rms_m": _pooled_rms(
            (cell["position_rms_m"], cell["position_members"])
            for cell in cells
            if cell["position_members"] >= 2
        ),
        "position_max_m": position_max,
        "clock_rms_s": _pooled_rms(
            (cell["clock_rms_s"], cell["clock_members"])
            for cell in cells
            if cell["clock_members"] >= 2
        ),
        "clock_max_s": clock_max,
    }


def _agreement_epoch_aggregates(
    cells: Sequence[Mapping[str, object]],
) -> list[dict]:
    groups: list[list[Mapping[str, object]]] = []
    for cell in cells:
        if not groups or _epoch_key(groups[-1][0]) != _epoch_key(cell):
            groups.append([])
        groups[-1].append(cell)
    out = []
    for group in groups:
        multi_position = [cell for cell in group if cell["position_members"] >= 2]
        multi_clock = [cell for cell in group if cell["clock_members"] >= 2]
        first = group[0]
        clock_max = None
        for cell in multi_clock:
            value = cell["clock_max_s"]
            clock_max = value if clock_max is None else max(clock_max, value)
        out.append(
            {
                "jd_whole": first["jd_whole"],
                "jd_fraction": first["jd_fraction"],
                "satellites": len(multi_position),
                "position_rms_m": _pooled_rms(
                    (cell["position_rms_m"], cell["position_members"])
                    for cell in multi_position
                )
                or 0.0,
                "position_max_m": max(cell["position_max_m"] for cell in group),
                "clock_rms_s": _pooled_rms(
                    (cell["clock_rms_s"], cell["clock_members"]) for cell in multi_clock
                ),
                "clock_max_s": clock_max,
            }
        )
    return out


def _validate_agreement(
    value: object, source_count: int
) -> tuple[
    Mapping[str, object], list[Mapping[str, object]], list[Mapping[str, object]]
]:
    fields = _exact_fields(value, _AGREEMENT_FIELDS, "SP3 agreement")
    for name in (
        "position_rms_m",
        "position_max_m",
        "clock_rms_s",
        "clock_max_s",
    ):
        _optional_nonnegative_float(fields[name], f"SP3 agreement {name}")
    cells = _validate_agreement_cells(fields["cells"], source_count)
    epochs = _validate_agreement_epochs(fields["epochs"])
    expected = _agreement_aggregate(cells)
    for name in expected:
        if fields[name] != expected[name] or type(fields[name]) is not type(
            expected[name]
        ):
            raise ValueError("SP3 agreement aggregate disagrees with cells")
    if epochs != _agreement_epoch_aggregates(cells):
        raise ValueError("SP3 agreement epoch aggregates disagree with cells")
    return fields, cells, epochs


def _validate_float_vector(value: object, description: str) -> list[float]:
    if type(value) is not list or len(value) != 3:
        raise ValueError(f"{description} must be a three-value list")
    for component in value:
        _exact_float(component, description)
    return value


def _validate_transform(
    value: object, *, rates: bool
) -> Optional[Mapping[str, object]]:
    if value is None:
        return None
    expected = _HELMERT_RATE_FIELDS if rates else _HELMERT_PARAMETER_FIELDS
    fields = _exact_fields(value, expected, "SP3 Helmert transform")
    if rates:
        _validate_float_vector(
            fields["translation_mm_per_year"], "SP3 Helmert translation rates"
        )
        _exact_float(fields["scale_ppb_per_year"], "SP3 Helmert scale rate")
        _validate_float_vector(
            fields["rotation_mas_per_year"], "SP3 Helmert rotation rates"
        )
    else:
        _validate_float_vector(fields["translation_mm"], "SP3 Helmert translation")
        _exact_float(fields["scale_ppb"], "SP3 Helmert scale")
        _validate_float_vector(fields["rotation_mas"], "SP3 Helmert rotation")
    return fields


def _sp3_frame_for_label(label: str) -> Optional[str]:
    if label in {"ITRF2020", "ITRF20", "IGS20", "IGc20"}:
        return "ITRF2020"
    if label in {"ITRF2014", "ITRF14", "IGS14", "IGb14"}:
        return "ITRF2014"
    if label in {"ITRF2008", "ITRF08", "IGS08", "IGb08"}:
        return "ITRF2008"
    return None


def _catalog_transform_dict(transform, *, rates: bool) -> dict:
    if rates:
        return {
            "translation_mm_per_year": transform.translation_mm_per_year.tolist(),
            "scale_ppb_per_year": transform.scale_ppb_per_year,
            "rotation_mas_per_year": transform.rotation_mas_per_year.tolist(),
        }
    return {
        "translation_mm": transform.translation_mm.tolist(),
        "scale_ppb": transform.scale_ppb,
        "rotation_mas": transform.rotation_mas.tolist(),
    }


def _validate_frame_method(
    value: Mapping[str, object], policy: Mapping[str, object]
) -> None:
    source_label = value["source_label"]
    target_label = value["target_label"]
    assertion = next(
        (
            labels
            for labels in policy["asserted_frame_label_sets"]
            if source_label in labels and target_label in labels
        ),
        None,
    )
    if value["method"] == "asserted_equivalence":
        helmert_fields = (
            "source_frame",
            "target_frame",
            "catalog_source_frame",
            "catalog_target_frame",
            "reference_epoch_year",
            "parameters",
            "rates",
            "provenance",
            "epoch_year_span",
        )
        if (
            value["asserted_label_set"] != assertion
            or assertion is None
            or source_label not in assertion
            or target_label not in assertion
            or any(value[name] is not None for name in helmert_fields)
            or value["catalog_inverse"] is not False
            or value["identity"] is not True
        ):
            raise ValueError("invalid asserted SP3 frame reconciliation")
        return

    if value["method"] != "helmert":
        raise ValueError("invalid SP3 frame reconciliation method")
    source_frame = _sp3_frame_for_label(source_label)
    target_frame = _sp3_frame_for_label(target_label)
    if (
        policy["helmert"] is not True
        or assertion is not None
        or source_frame is None
        or target_frame is None
        or value["source_frame"] != source_frame
        or value["target_frame"] != target_frame
        or value["asserted_label_set"] is not None
        or value["identity"] != (source_frame == target_frame)
        or (value["records_affected"] != 0 and value["epoch_year_span"] is None)
    ):
        raise ValueError("invalid Helmert SP3 frame reconciliation")

    if value["identity"]:
        catalog_fields = (
            "catalog_source_frame",
            "catalog_target_frame",
            "reference_epoch_year",
            "parameters",
            "rates",
            "provenance",
        )
        if value["catalog_inverse"] is not False or any(
            value[name] is not None for name in catalog_fields
        ):
            raise ValueError("invalid identity Helmert catalog record")
        return

    catalog_from, catalog_to = (
        (target_frame, source_frame)
        if value["catalog_inverse"]
        else (source_frame, target_frame)
    )
    if (
        value["catalog_source_frame"] != catalog_from
        or value["catalog_target_frame"] != catalog_to
    ):
        raise ValueError("invalid Helmert catalog orientation")
    catalog = sidereon.frame_catalog_entry(catalog_from, catalog_to)
    if catalog is None:
        raise ValueError("missing public Helmert catalog entry")
    if (
        value["reference_epoch_year"] != catalog.reference_epoch_year
        or value["parameters"]
        != _catalog_transform_dict(catalog.parameters, rates=False)
        or value["rates"] != _catalog_transform_dict(catalog.rates, rates=True)
        or value["provenance"] != catalog.provenance
    ):
        raise ValueError("SP3 frame reconciliation disagrees with public catalog")


def _validate_frame_reconciliations(
    value: object, source_count: int, policy: Mapping[str, object]
) -> list[Mapping[str, object]]:
    if type(value) is not list:
        raise ValueError("SP3 frame reconciliations must be a list")
    reconciliations = []
    for item in value:
        fields = _exact_fields(
            item, _FRAME_RECONCILIATION_FIELDS, "SP3 frame reconciliation"
        )
        source_index = _exact_int(
            fields["source_index"], "SP3 frame reconciliation source index"
        )
        if source_index == 0 or source_index >= source_count:
            raise ValueError("SP3 frame reconciliation source index is invalid")
        for name in ("source_label", "target_label"):
            label = _exact_str(fields[name], f"SP3 frame reconciliation {name}")
            if label != label.strip():
                raise ValueError("SP3 frame reconciliation label is not canonical")
        if fields["source_label"] == fields["target_label"]:
            raise ValueError("SP3 frame reconciliation labels are equal")
        method = _exact_str(fields["method"], "SP3 frame reconciliation method")
        if method not in {"asserted_equivalence", "helmert"}:
            raise ValueError("invalid SP3 frame reconciliation method")
        labels = fields["asserted_label_set"]
        if labels is not None:
            if (
                type(labels) is not list
                or len(labels) < 2
                or any(
                    type(label) is not str or not label or label != label.strip()
                    for label in labels
                )
                or labels != sorted(set(labels))
            ):
                raise ValueError("invalid asserted SP3 frame label set")
        for name in (
            "source_frame",
            "target_frame",
            "catalog_source_frame",
            "catalog_target_frame",
        ):
            frame = fields[name]
            if frame is not None and (
                type(frame) is not str or frame not in _TERRESTRIAL_FRAMES
            ):
                raise ValueError("invalid SP3 terrestrial frame")
        if type(fields["catalog_inverse"]) is not bool:
            raise ValueError("SP3 catalog_inverse must be a boolean")
        if fields["reference_epoch_year"] is not None:
            _exact_float(fields["reference_epoch_year"], "SP3 frame reference epoch")
        _validate_transform(fields["parameters"], rates=False)
        _validate_transform(fields["rates"], rates=True)
        _optional_exact_str(fields["provenance"], "SP3 frame provenance")
        span = fields["epoch_year_span"]
        if span is not None:
            if type(span) is not list or len(span) != 2:
                raise ValueError("invalid SP3 frame epoch span")
            first = _exact_float(span[0], "SP3 frame epoch span")
            last = _exact_float(span[1], "SP3 frame epoch span")
            if not 0.0 <= first <= last < 10_000.0:
                raise ValueError("invalid SP3 frame epoch span")
        _exact_int(fields["records_affected"], "SP3 reconciled record count")
        if type(fields["identity"]) is not bool:
            raise ValueError("SP3 frame identity must be a boolean")
        _validate_frame_method(fields, policy)
        reconciliations.append(fields)
    if not _strictly_ascending([item["source_index"] for item in reconciliations]):
        raise ValueError("SP3 frame reconciliations are not ordered and unique")
    if len({item["target_label"] for item in reconciliations}) > 1:
        raise ValueError("SP3 frame reconciliation targets disagree")
    return reconciliations


def _contested_minimum(policy: Mapping[str, object]) -> int:
    if policy["combine"] == "precedence" and policy["outlier_reject"] is not None:
        return max(policy["min_agree"], 2)
    return policy["min_agree"]


def _within_scaled_relative_policy_bound(
    value: float, bound: float, scale: float
) -> bool:
    if bound == 0.0:
        return value == 0.0
    return value / (scale * (1.0 + 1.0e-12)) <= bound


def _validate_selected_member_dispersion(
    cell: Mapping[str, object],
    *,
    position_selects_member: bool,
    clock_selects_member: bool,
) -> None:
    if position_selects_member:
        _validate_member_metric_bound(
            cell["position_rms_m"],
            cell["position_max_m"],
            cell["position_members"],
            "selected SP3 position dispersion",
            upper_terms=cell["position_members"] - 1,
        )
    if clock_selects_member and cell["clock_members"] > 0:
        _validate_member_metric_bound(
            cell["clock_rms_s"],
            cell["clock_max_s"],
            cell["clock_members"],
            "selected SP3 clock dispersion",
            upper_terms=cell["clock_members"] - 1,
        )


def _validate_agreement_policy(
    cells: Sequence[Mapping[str, object]], policy: Mapping[str, object]
) -> None:
    # Mean uses naive floating summation in core. Without the absolute source
    # coordinates, a persisted report cannot derive a safe roundoff bound.
    if policy["combine"] == "mean":
        return
    if policy["combine"] == "precedence":
        if policy["outlier_reject"] is not None:
            position_bound = policy["outlier_reject"]["position_tolerance_m"]
            clock_bound = policy["outlier_reject"]["clock_tolerance_s"]
        else:
            position_bound = policy["position_tolerance_m"]
            clock_bound = policy["clock_tolerance_s"]
        for cell in cells:
            _validate_selected_member_dispersion(
                cell,
                position_selects_member=True,
                clock_selects_member=True,
            )
            if cell["position_max_m"] > position_bound:
                raise ValueError("SP3 position agreement exceeds merge policy")
            if cell["clock_max_s"] is not None and cell["clock_max_s"] > clock_bound:
                raise ValueError("SP3 clock agreement exceeds merge policy")
        return

    for cell in cells:
        _validate_selected_member_dispersion(
            cell,
            position_selects_member=False,
            clock_selects_member=cell["clock_members"] % 2 == 1,
        )
        if not _within_scaled_relative_policy_bound(
            cell["position_max_m"],
            policy["position_tolerance_m"],
            _math.sqrt(3.0),
        ):
            raise ValueError("SP3 position agreement exceeds merge policy")
        if cell["clock_max_s"] is not None and not (
            _within_scaled_relative_policy_bound(
                cell["clock_max_s"], policy["clock_tolerance_s"], 1.0
            )
        ):
            raise ValueError("SP3 clock agreement exceeds merge policy")


def _j2000_second_key(value: Mapping[str, object]) -> int:
    day_seconds = (value["jd_whole"] - 2_451_545.0) * 86_400.0
    within_day = value["jd_fraction"] * 86_400.0
    nearest = round(within_day)
    if value["jd_fraction"] == nearest / 86_400.0:
        return int(day_seconds + nearest)
    whole_second = _math.floor(within_day)
    fractional_second = within_day - whole_second
    return int(_math.floor(day_seconds + whole_second + fractional_second))


def _validate_epoch_grid(
    records: Sequence[Mapping[str, object]], interval: Optional[float]
) -> None:
    if not records:
        return
    keyed_epochs = [
        (_j2000_second_key(record), _epoch_key(record)) for record in records
    ]
    exact_epochs_by_second: dict[int, set[tuple[float, float]]] = {}
    for second, exact_epoch in keyed_epochs:
        exact_epochs_by_second.setdefault(second, set()).add(exact_epoch)
    if any(len(epochs) > 1 for epochs in exact_epochs_by_second.values()):
        raise ValueError("SP3 merge report contains aliased epochs")
    if interval is None:
        return
    step = round(interval)
    anchor = keyed_epochs[0][0]
    if not all((second - anchor) % step == 0 for second, _ in keyed_epochs):
        raise ValueError("SP3 merge report is off the requested epoch grid")


def _validate_precedence_flags(
    quarantined: Sequence[Mapping[str, object]],
    single_source: Sequence[Mapping[str, object]],
    position_outliers: Sequence[Mapping[str, object]],
    clock_outliers: Sequence[Mapping[str, object]],
    policy: Mapping[str, object],
) -> None:
    if policy["combine"] != "precedence":
        return
    outliers = list(position_outliers) + list(clock_outliers)
    if policy["outlier_reject"] is None and any(
        0 in flag["sources"] for flag in outliers
    ):
        raise ValueError("precedence outlier contains the preferred source")
    if policy["precedence_scope"] != "satellite_arc":
        return
    all_flags = (
        list(quarantined)
        + list(single_source)
        + list(position_outliers)
        + list(clock_outliers)
    )
    satellites = {flag["satellite"] for flag in single_source}
    for satellite in satellites:
        owners = {
            flag["sources"][0]
            for flag in single_source
            if flag["satellite"] == satellite
        }
        if len(owners) != 1:
            raise ValueError("satellite-arc precedence owner is inconsistent")
        owner = next(iter(owners))
        mentioned = [
            source
            for flag in all_flags
            if flag["satellite"] == satellite
            for source in flag["sources"]
        ]
        if any(source < owner for source in mentioned):
            raise ValueError("satellite-arc precedence owner is inconsistent")
        if policy["outlier_reject"] is None and any(
            flag["satellite"] == satellite and owner in flag["sources"]
            for flag in outliers
        ):
            raise ValueError("satellite-arc precedence rejects its owner")


def _validate_accepted_cells(
    cells: Sequence[Mapping[str, object]],
    single_source: Sequence[Mapping[str, object]],
    position_outliers: Sequence[Mapping[str, object]],
    clock_outliers: Sequence[Mapping[str, object]],
    source_count: int,
    policy: Mapping[str, object],
) -> None:
    single_by_key = {_cell_key(flag): flag for flag in single_source}
    position_by_key = {_cell_key(flag): flag for flag in position_outliers}
    clock_by_key = {_cell_key(flag): flag for flag in clock_outliers}
    required = _contested_minimum(policy)
    for cell in cells:
        key = _cell_key(cell)
        single = single_by_key.get(key)
        position = position_by_key.get(key)
        clock = clock_by_key.get(key)
        position_sources = cell["position_members"] + (
            0 if position is None else len(position["sources"])
        )
        clock_sources = cell["clock_members"] + (
            0 if clock is None else len(clock["sources"])
        )
        if clock_sources > position_sources:
            raise ValueError("clock contributor count exceeds position contributors")
        if single is not None and (
            cell["position_members"] != 1 or position is not None
        ):
            raise ValueError("single-source and outlier flags contradict")
        if position is not None and (
            source_count < 2
            or position_sources > source_count
            or cell["position_members"] < required
        ):
            raise ValueError("invalid SP3 position outlier")
        if cell["position_members"] == 1 and single is None and position is None:
            raise ValueError("single-member SP3 cell lacks an audit flag")
        if cell["position_members"] > 1 and (
            single is not None or cell["position_members"] < policy["min_agree"]
        ):
            raise ValueError("invalid SP3 position consensus")
        if clock is None:
            if 1 < cell["clock_members"] < policy["min_agree"]:
                raise ValueError("invalid SP3 clock consensus")
        elif (
            source_count < 2
            or (cell["clock_members"] > 0 and clock_sources > source_count)
            or (cell["clock_members"] > 0 and cell["clock_members"] < required)
            or (
                cell["clock_members"] == 0
                and not (
                    policy["combine"] == "precedence"
                    and policy["outlier_reject"] is not None
                    and len(clock["sources"]) >= 2
                )
            )
        ):
            raise ValueError("invalid SP3 clock outlier")


def _validate_merge_result(
    value: object, source_count: int, policy: Mapping[str, object]
) -> None:
    fields = _exact_fields(value, _MERGE_RESULT_FIELDS, "SP3 merge result")
    _validate_frame_reconciliations(
        fields["frame_reconciliations"], source_count, policy
    )
    quarantined = _validate_flags(fields["quarantined"], source_count, "quarantined")
    single_source = _validate_flags(
        fields["single_source"], source_count, "single-source"
    )
    position_outliers = _validate_flags(
        fields["position_outliers"], source_count, "position-outlier"
    )
    clock_outliers = _validate_flags(
        fields["clock_outliers"], source_count, "clock-outlier"
    )
    _, cells, epochs = _validate_agreement(fields["agreement"], source_count)
    accepted_keys = {_cell_key(cell) for cell in cells}
    quarantined_keys = {_cell_key(flag) for flag in quarantined}
    single_keys = {_cell_key(flag) for flag in single_source}
    position_keys = {_cell_key(flag) for flag in position_outliers}
    clock_keys = {_cell_key(flag) for flag in clock_outliers}
    if any(len(flag["sources"]) < 2 for flag in quarantined):
        raise ValueError("quarantined SP3 cells require multiple sources")
    if quarantined and _contested_minimum(policy) < 2:
        raise ValueError("quarantine is impossible under the merge policy")
    if any(len(flag["sources"]) != 1 for flag in single_source):
        raise ValueError("single-source SP3 flags require one source")
    if not quarantined_keys.isdisjoint(
        accepted_keys | single_keys | position_keys | clock_keys
    ):
        raise ValueError("quarantined SP3 flags contradict accepted records")
    if not (
        single_keys <= accepted_keys
        and position_keys <= accepted_keys
        and clock_keys <= accepted_keys
    ):
        raise ValueError("SP3 merge flags refer to unaccepted cells")
    if not single_keys.isdisjoint(position_keys | clock_keys):
        raise ValueError("single-source SP3 flags contradict outliers")
    _validate_agreement_policy(cells, policy)
    _validate_epoch_grid(
        list(quarantined)
        + list(single_source)
        + list(position_outliers)
        + list(clock_outliers)
        + list(cells)
        + list(epochs),
        policy["target_epoch_interval_s"],
    )
    _validate_precedence_flags(
        quarantined,
        single_source,
        position_outliers,
        clock_outliers,
        policy,
    )
    _validate_accepted_cells(
        cells,
        single_source,
        position_outliers,
        clock_outliers,
        source_count,
        policy,
    )
    if source_count == 1:
        if (
            quarantined
            or position_outliers
            or clock_outliers
            or accepted_keys != single_keys
            or any(
                cell["position_members"] != 1 or cell["clock_members"] > 1
                for cell in cells
            )
        ):
            raise ValueError("invalid single-product SP3 merge report")
    systems = policy["systems"]
    if systems is not None and any(
        record["satellite"][0] not in systems
        for record in list(quarantined)
        + list(single_source)
        + list(position_outliers)
        + list(clock_outliers)
        + list(cells)
    ):
        raise ValueError("SP3 merge report contains a filtered system")


def verify_merge_report(value: Mapping[str, object]) -> bool:
    """Verify a persisted :meth:`MergeReport.to_dict` record with the core.

    Returns ``False`` for incomplete, malformed, internally inconsistent, or
    identity-mismatched reports. Observational acquisition facts are parsed but
    never enter the stable identity.
    """
    from sidereon import distribution

    try:
        report = _exact_fields(value, _MERGE_REPORT_FIELDS, "merged-SP3 report")
        _exact_schema_version(report["schema_version"], "merged-SP3 report")
        contributors_value = report["contributors"]
        policy_value = report["merge_policy"]
        if type(contributors_value) is not list:
            raise ValueError("contributors must be a list")
        if not isinstance(policy_value, Mapping):
            raise ValueError("merge policy must be a mapping")
        artifacts = []
        contributor_centers = []
        for contributor in contributors_value:
            contributor = _exact_fields(
                contributor, _CONTRIBUTOR_FIELDS, "SP3 contributor"
            )
            artifact_value = contributor["artifact_identity"]
            facts_value = contributor["acquisition_facts"]
            if not isinstance(artifact_value, Mapping) or not isinstance(
                facts_value, Mapping
            ):
                raise ValueError("contributor provenance must be mappings")
            artifact = ArtifactIdentity.from_dict(artifact_value)
            AcquisitionFacts.from_dict(facts_value)
            distribution._validate_requested_identity(artifact.requested_identity)
            if not _contributor_matches_catalog(contributor, artifact):
                raise ValueError("contributor fields disagree with artifact identity")
            artifacts.append(artifact)
            contributor_centers.append(artifact.requested_identity.analysis_center)
        if not artifacts or len(set(contributor_centers)) != len(contributor_centers):
            raise ValueError("contributors must contain unique centers")

        absent_value = report["absent"]
        if type(absent_value) is not list:
            raise ValueError("absent centers must be a list")
        absent_centers = [_validate_absent_center(item) for item in absent_value]
        if len(set(absent_centers)) != len(absent_centers):
            raise ValueError("absent centers must be unique")

        requested_centers = report["requested_centers"]
        if type(requested_centers) is not list or any(
            type(center) is not str for center in requested_centers
        ):
            raise ValueError("requested centers must be a list of strings")
        if len(set(requested_centers)) != len(requested_centers):
            raise ValueError("requested centers must be unique")
        for center in requested_centers:
            _center_def(center)
        if set(requested_centers) != set(contributor_centers) | set(absent_centers):
            raise ValueError("contributors and absent centers do not partition request")
        if set(contributor_centers) & set(absent_centers):
            raise ValueError("a center cannot be both contributor and absent")
        order = {center: index for index, center in enumerate(requested_centers)}
        if contributor_centers != sorted(contributor_centers, key=order.__getitem__):
            raise ValueError("contributors are not in request order")
        if absent_centers != sorted(absent_centers, key=order.__getitem__):
            raise ValueError("absent centers are not in request order")

        source_count = _exact_int(report["source_count"], "source count", minimum=1)
        single_product = report["single_product"]
        merged = report["merged"]
        if source_count != len(artifacts):
            raise ValueError("source count disagrees with contributors")
        if type(single_product) is not bool or single_product != (len(artifacts) == 1):
            raise ValueError("single-product flag disagrees with contributors")
        if merged is not True:
            raise ValueError("merged flag must be true")
        options = _merge_options_from_policy(policy_value)
        precedence = policy_value["precedence_artifact_sha256"]
        expected_precedence = (
            [artifact.product_sha256 for artifact in artifacts]
            if options.combine.label == "precedence"
            else []
        )
        if precedence != expected_precedence:
            raise ValueError("precedence contributors disagree with contributor order")
        _validate_merge_result(report["merge_report"], source_count, policy_value)
        identity = sp3_merge_input_identity(artifacts, options)
        return (
            type(report["input_identity_schema_version"]) is int
            and report["input_identity_schema_version"] == identity.schema_version
            and type(report["stable_input_identity"]) is str
            and report["stable_input_identity"] == identity.stable_id
        )
    except (KeyError, TypeError, ValueError, OverflowError, DataError):
        return False


def _merge_policy_to_dict(
    merge_options: Optional["sidereon.Sp3MergeOptions"],
    artifacts: Sequence[ArtifactIdentity],
) -> dict:
    options = merge_options or sidereon.Sp3MergeOptions()
    outlier = options.outlier_reject
    combine = options.combine.label
    canonical_frame_sets = sorted(
        sorted(set(labels)) for labels in options.asserted_frame_label_sets
    )
    return {
        "schema_version": 1,
        "position_tolerance_m": (
            0.0 if options.position_tolerance_m == 0.0 else options.position_tolerance_m
        ),
        "clock_tolerance_s": (
            0.0 if options.clock_tolerance_s == 0.0 else options.clock_tolerance_s
        ),
        "min_agree": options.min_agree,
        "clock_min_common": options.clock_min_common,
        "combine": combine,
        "precedence_scope": options.precedence_scope.label,
        "outlier_reject": (
            None
            if outlier is None
            else {
                "position_tolerance_m": (
                    0.0
                    if outlier.position_tolerance_m == 0.0
                    else outlier.position_tolerance_m
                ),
                "clock_tolerance_s": (
                    0.0
                    if outlier.clock_tolerance_s == 0.0
                    else outlier.clock_tolerance_s
                ),
            }
        ),
        "target_epoch_interval_s": options.target_epoch_interval_s,
        "systems": None if options.systems is None else sorted(set(options.systems)),
        "asserted_frame_label_sets": canonical_frame_sets,
        "helmert": options.helmert,
        "precedence_artifact_sha256": (
            [artifact.product_sha256 for artifact in artifacts]
            if combine == "precedence"
            else []
        ),
    }


def fetch_merged_sp3(
    target: Union[_dt.date, _dt.datetime],
    centers: Sequence[str],
    *,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    systems: Optional[Sequence[str]] = None,
    epoch_interval_s: Optional[float] = None,
    sample: Optional[str] = None,
    merge_options: Optional["sidereon.Sp3MergeOptions"] = None,
    **fetch_opts,
) -> tuple["sidereon.Sp3", MergeReport]:
    """Fetch SP3 from several centers and merge the available ones.

    ``centers`` are tried in precedence order; a missing or not-yet-published
    center is recorded in the report and does not abort the call. Returns the
    parsed merged :class:`sidereon.Sp3` and a :class:`MergeReport`. Raises
    :class:`NoProducts` when no center contributes and
    :class:`IncompatibleSources` when the fetched sources cannot be combined.
    Ultra-rapid centers probe only officially cataloged dated variants after
    archive misses. A second candidate exists only for an explicitly evidenced
    publication overlap. Pass a complete :class:`Sp3MergeOptions` as
    ``merge_options``; single contributors follow the same merge path.
    """
    if not isinstance(centers, (list, tuple)):
        raise UnsupportedProduct("centers must be a list of center codes")
    if len(set(centers)) != len(centers):
        raise UnsupportedProduct("centers must not contain duplicates")

    # Validate every center and its SP3 capability before any cache or network
    # acquisition. A known non-SP3 center is caller configuration, not absence.
    for center in centers:
        if "sp3" not in _center_def(center)["products"]:
            raise UnsupportedProduct(f"{center} does not serve sp3")

    fetch_kwargs = dict(cache_dir=cache_dir, offline=offline, **fetch_opts)
    if "max_compressed_bytes" in fetch_kwargs:
        fetch_kwargs["max_archive_bytes"] = fetch_kwargs.pop("max_compressed_bytes")
    if "max_decompressed_bytes" in fetch_kwargs:
        fetch_kwargs["max_product_bytes"] = fetch_kwargs.pop("max_decompressed_bytes")
    results = [
        _fetch_center_sp3(center, target, sample, fetch_kwargs) for center in centers
    ]

    contributors = [r for r in results if r[0] == "ok"]
    absent = [r[1] for r in results if r[0] == "absent"]

    if not contributors:
        raise NoProducts(absent)

    sources = [c[2] for c in contributors]
    if merge_options is not None and (
        systems is not None or epoch_interval_s is not None
    ):
        raise UnsupportedProduct(
            "merge_options cannot be combined with systems or epoch_interval_s"
        )
    options = merge_options or _merge_options(systems, epoch_interval_s)
    try:
        merged, merge_report = sidereon.merge_sp3(sources, options)
    except sidereon.SidereonError as exc:
        raise IncompatibleSources([c[1].center for c in contributors], exc) from exc

    artifact_identities = []
    for contributor in contributors:
        artifact_identity = contributor[1].artifact_identity
        if artifact_identity is None:  # pragma: no cover - internal invariant
            raise UnsupportedProduct("SP3 contributor provenance is incomplete")
        artifact_identities.append(artifact_identity)
    input_identity_schema_version, stable_input_identity = sp3_merge_input_identity(
        artifact_identities, options
    )
    report = MergeReport(
        contributors=[c[1] for c in contributors],
        absent=absent,
        source_count=len(contributors),
        single_product=len(contributors) == 1,
        merged=True,
        requested_centers=list(centers),
        merge_report=merge_report,
        stable_input_identity=stable_input_identity,
        input_identity_schema_version=input_identity_schema_version,
        merge_policy=_merge_policy_to_dict(options, artifact_identities),
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
    return_report: bool = False,
    cache_dir: Optional[str] = None,
    offline: bool = False,
    systems: Optional[Sequence[str]] = None,
    epoch_interval_s: Optional[float] = None,
    sample: Optional[str] = None,
    merge_options: Optional["sidereon.Sp3MergeOptions"] = None,
    **fetch_opts,
) -> Union[str, tuple[str, MergeReport]]:
    """Fetch the merged SP3 from several centers and persist it to ``path``.

    Composes :func:`fetch_merged_sp3` with :func:`write_sp3`. Returns the written
    path by default. Pass ``return_report=True`` to receive ``(path, report)``
    and retain the exact contributor provenance and stable input identity.
    Nothing is written if the fetch/merge step raises.
    """
    merged, report = fetch_merged_sp3(
        target,
        centers,
        cache_dir=cache_dir,
        offline=offline,
        systems=systems,
        epoch_interval_s=epoch_interval_s,
        sample=sample,
        merge_options=merge_options,
        **fetch_opts,
    )
    written = write_sp3(merged, path, gzip=gzip)
    if return_report:
        return written, report
    return written


def __getattr__(name: str):
    """Lazily expose exact-distribution APIs without an import cycle."""
    if name in _DISTRIBUTION_EXPORTS:
        from sidereon import distribution

        return getattr(distribution, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
