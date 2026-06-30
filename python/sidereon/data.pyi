"""Type stubs for the optional GNSS data-provisioning layer."""

import datetime as _dt
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import sidereon

__all__: list[str]

class DataError(Exception): ...
class UnknownCenter(DataError): ...
class UnsupportedProduct(DataError): ...
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

def default_cache_dir() -> str: ...
def centers() -> list[str]: ...
def content_types() -> list[str]: ...
def gps_week(date: _dt.date) -> int: ...
def day_of_year(date: _dt.date) -> int: ...
def predicted_day_offset(center: str) -> int: ...
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
