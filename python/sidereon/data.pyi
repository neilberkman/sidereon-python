"""Type stubs for the optional GNSS data-provisioning layer."""

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence, Union

import sidereon

__all__: list[str]

class DataError(Exception): ...
class UnknownCenter(DataError): ...
class UnsupportedProduct(DataError): ...
class InvalidCoordinate(DataError): ...
class InvalidTileIndex(DataError): ...
class InvalidTileId(DataError): ...
class OfflineCacheMiss(DataError): ...
class FileNotFoundOnArchive(DataError): ...

class HttpStatusError(DataError):
    status: int
    url: str
    def __init__(self, status: int, url: str) -> None: ...

class RedirectNotAllowed(DataError):
    status: int
    url: str
    def __init__(self, status: int, url: str) -> None: ...

class NetworkError(DataError): ...

class ChecksumMismatch(DataError):
    expected: str
    got: str
    def __init__(self, expected: str, got: str) -> None: ...

class DownloadSizeExceeded(DataError): ...
class DecompressError(DataError): ...
class CacheNotWritable(DataError): ...

class NoCoverage(DataError):
    tile_id: str
    def __init__(self, tile_id: str) -> None: ...

class IncompatibleSources(DataError):
    centers: list[str]
    reason: object
    def __init__(self, centers: Sequence[str], reason: object) -> None: ...

class NoProducts(DataError):
    reasons: list[AbsentCenter]
    def __init__(self, reasons: Sequence[AbsentCenter]) -> None: ...

@dataclass(frozen=True)
class Product:
    center: str
    content: str
    date: _dt.date
    sample: str
    issue: Optional[str] = ...
    @property
    def gps_week(self) -> int: ...
    @property
    def day_of_year(self) -> int: ...
    def canonical_filename(self) -> str: ...
    def archive_url(self) -> str: ...

@dataclass(frozen=True)
class AbsentCenter:
    center: str
    filename: Optional[str]
    reason: str

@dataclass(frozen=True)
class Contributor:
    center: str
    filename: str
    date: _dt.date
    issue: Optional[str]

@dataclass
class MergeReport:
    contributors: list[Contributor]
    absent: list[AbsentCenter]
    source_count: int
    single_product: bool
    merged: bool
    merge_report: Optional[sidereon.Sp3MergeReport] = ...

@dataclass(frozen=True)
class TerrainSourceEntry:
    protocol: str
    host: str
    compression: str
    root_url: str

@dataclass(frozen=True)
class SpaceWeatherSourceEntry:
    protocol: str
    host: str
    compression: str
    root_url: str

@dataclass(frozen=True)
class TerrainFetchReport:
    fetched: list[str]
    cached: list[str]
    no_coverage: list[str]
    errors: list[tuple[str, DataError]]

def default_cache_dir() -> str: ...
def default_terrain_cache_dir() -> str: ...
def centers() -> list[str]: ...
def content_types() -> list[str]: ...
def gps_week(date: _dt.date) -> int: ...
def day_of_year(date: _dt.date) -> int: ...
def predicted_day_offset(center: str) -> int: ...
def skadi_source_entry() -> TerrainSourceEntry: ...
def space_weather_source_entry() -> SpaceWeatherSourceEntry: ...
def space_weather_filename(product: str = ...) -> str: ...
def space_weather_archive_url(product: str = ...) -> str: ...
def space_weather_cache_relpath(product: str = ...) -> str: ...
def skadi_tile_id(lat_index: int, lon_index: int) -> str: ...
def skadi_band(lat_index: int) -> str: ...
def skadi_archive_url(lat_index: int, lon_index: int) -> str: ...
def terrain_tile_index(lat_deg: float, lon_deg: float) -> tuple[int, int]: ...
def dted_tile_filename(lat_index: int, lon_index: int) -> str: ...
def dted_block_dir(lat_index: int, lon_index: int) -> str: ...
def dted_cache_relpath(lat_index: int, lon_index: int) -> str: ...
def parse_skadi_tile_id(tile_id: str) -> tuple[int, int]: ...
def hgt_to_dted(lat_index: int, lon_index: int, hgt: bytes) -> bytes: ...
def canonical_filename(
    center: str,
    content: str,
    date: _dt.date,
    sample: Optional[str] = ...,
    *,
    issue: Optional[str] = ...,
) -> str: ...
def archive_url(
    center: str,
    content: str,
    date: _dt.date,
    sample: Optional[str] = ...,
    *,
    issue: Optional[str] = ...,
) -> str: ...
def product(
    center: str,
    content: str,
    date: _dt.date,
    sample: Optional[str] = ...,
    *,
    issue: Optional[str] = ...,
) -> Product: ...
def mgex_ionex(
    center: str, date: _dt.date, *, sample: Optional[str] = ...
) -> Product: ...
def rapid_ionex(date: _dt.date, *, sample: Optional[str] = ...) -> Product: ...
def predicted_ionex(
    center: str, date: _dt.date, *, sample: Optional[str] = ...
) -> Product: ...
def mgex_sp3(
    center: str, date: _dt.date, *, sample: Optional[str] = ...
) -> Product: ...
def ops_ultra_sp3(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    *,
    sample: Optional[str] = ...,
    issue: Optional[str] = ...,
    available_issues: Optional[Sequence[tuple[_dt.date, str]]] = ...,
) -> Product: ...
def fetch(
    product: Product,
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    sha256: Optional[str] = ...,
    max_decompressed_bytes: int = ...,
    timeout_s: float = ...,
    retries: int = ...,
    backoff_s: float = ...,
    max_compressed_bytes: int = ...,
) -> str: ...
def fetch_dted(
    lat: float,
    lon: float,
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    sha256: Optional[str] = ...,
    strict: bool = ...,
    timeout_s: float = ...,
    retries: int = ...,
    backoff_s: float = ...,
    max_compressed_bytes: int = ...,
    max_decompressed_bytes: int = ...,
) -> Optional[str]: ...
def fetch_space_weather(
    product: str = ...,
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    sha256: Optional[str] = ...,
    max_age_s: float = ...,
    timeout_s: float = ...,
    retries: int = ...,
    backoff_s: float = ...,
    max_compressed_bytes: int = ...,
) -> sidereon.SpaceWeatherTable: ...
def prefetch_dted_bbox(
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    **opts: object,
) -> TerrainFetchReport: ...
def prefetch_dted_tiles(
    tiles: Union[Iterable[tuple[int, int]], Iterable[str], str],
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    **opts: object,
) -> TerrainFetchReport: ...
def populate_terrain_cache(
    region: Union[
        Mapping[str, float],
        tuple[float, float, float, float],
        Iterable[tuple[int, int]],
        Iterable[str],
    ],
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    **opts: object,
) -> TerrainFetchReport: ...
def fetch_ionex(
    center: str,
    target: Union[_dt.date, _dt.datetime],
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    sample: Optional[str] = ...,
    lookback: int = ...,
    **fetch_opts: object,
) -> sidereon.Ionex: ...
def fetch_merged_sp3(
    target: Union[_dt.date, _dt.datetime],
    centers: Sequence[str],
    *,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    systems: Optional[Sequence[str]] = ...,
    epoch_interval_s: Optional[float] = ...,
    sample: Optional[str] = ...,
    **fetch_opts: object,
) -> tuple[sidereon.Sp3, MergeReport]: ...
def fetch_merged_sp3_file(
    target: Union[_dt.date, _dt.datetime],
    centers: Sequence[str],
    path: str,
    *,
    gzip: bool = ...,
    cache_dir: Optional[str] = ...,
    offline: bool = ...,
    systems: Optional[Sequence[str]] = ...,
    epoch_interval_s: Optional[float] = ...,
    sample: Optional[str] = ...,
    **fetch_opts: object,
) -> str: ...
def write_sp3(sp3: sidereon.Sp3, path: str, *, gzip: bool = ...) -> str: ...
