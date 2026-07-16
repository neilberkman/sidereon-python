"""Types for explicit exact-product distribution acquisition."""

import datetime as dt
import enum
import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, Union

import httpx

from . import data as _data

class DistributionSource(enum.Enum):
    DIRECT: DistributionSource
    NASA_CDDIS: DistributionSource
    LOCAL_FILE: DistributionSource
    IN_MEMORY: DistributionSource
    @property
    def value(self) -> str: ...

@dataclass(frozen=True)
class Distribution:
    source: DistributionSource
    path: Optional[str] = ...
    content: Optional[bytes] = ...
    compression: Optional[str] = ...
    @classmethod
    def direct(cls) -> Distribution: ...
    @classmethod
    def nasa_cddis(cls) -> Distribution: ...
    @classmethod
    def local_file(
        cls,
        path: Union[str, os.PathLike[str]],
        *,
        compression: str = ...,
    ) -> Distribution: ...
    @classmethod
    def in_memory(cls, content: bytes, *, compression: str = ...) -> Distribution: ...

@dataclass(frozen=True)
class ProductIdentity:
    family: str
    analysis_center: str
    publisher: str
    solution_class: str
    campaign: str
    filename_version: int
    date: dt.date
    issue: str
    span: str
    sample: str
    official_filename: str
    format: str
    format_version: Optional[str] = ...
    prediction_horizon_days: Optional[int] = ...
    @property
    def key(self) -> str: ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ProductIdentity: ...

@dataclass(frozen=True)
class ProductRequest:
    identity: ProductIdentity
    distributors: Tuple[Distribution, ...]

@dataclass(frozen=True)
class EarthdataAuth:
    bearer_token: Optional[str] = ...
    use_netrc: bool = ...
    netrc_path: Optional[str] = ...
    @classmethod
    def bearer(cls, token: str) -> EarthdataAuth: ...
    @classmethod
    def from_netrc(
        cls, path: Optional[Union[str, os.PathLike[str]]] = ...
    ) -> EarthdataAuth: ...
    @property
    def configured(self) -> bool: ...

@dataclass(frozen=True)
class SourceFailure:
    source: DistributionSource
    error_type: str
    message: str
    url: Optional[str] = ...
    status: Optional[int] = ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> SourceFailure: ...

@dataclass(frozen=True)
class AcquisitionProvenance:
    requested_identity: ProductIdentity
    resolved_identity: ProductIdentity
    publisher: str
    distribution_source: DistributionSource
    official_filename: str
    original_url: Optional[str]
    final_url: Optional[str]
    retrieved_at: str
    byte_length: int
    sha256: str
    etag: Optional[str]
    last_modified: Optional[str]
    cache_hit: bool
    archive_compression: str
    archive_byte_length: int
    archive_sha256: str
    attempts: Tuple[SourceFailure, ...] = ...
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AcquisitionProvenance: ...

@dataclass(frozen=True)
class AcquiredProduct:
    path: str
    provenance: AcquisitionProvenance

class AcquisitionError(_data.DataError):
    code: str

class UnsupportedDistribution(AcquisitionError): ...

class HttpAcquisitionError(AcquisitionError):
    status: int
    url: str
    def __init__(self, status: int, url: str, message: str) -> None: ...

class ProductNotPublished(HttpAcquisitionError): ...
class AuthenticationRequired(HttpAcquisitionError): ...
class AuthenticationFailed(HttpAcquisitionError): ...
class AuthorizationDenied(HttpAcquisitionError): ...
class RedirectPolicyFailure(AcquisitionError): ...
class RetiredEndpoint(HttpAcquisitionError): ...
class MalformedUrl(AcquisitionError): ...

class TransportFailure(AcquisitionError):
    kind: str
    url: str
    status: Optional[int]
    def __init__(self, kind: str, url: str) -> None: ...

class InvalidContentType(AcquisitionError): ...
class ErrorDocument(AcquisitionError): ...
class ContentLengthMismatch(AcquisitionError): ...
class DecompressionFailure(AcquisitionError): ...
class ProductValidationFailure(AcquisitionError): ...
class CacheReadFailure(AcquisitionError): ...
class CacheWriteFailure(AcquisitionError): ...

class AllDistributorsFailed(AcquisitionError):
    attempts: Tuple[SourceFailure, ...]
    def __init__(self, attempts: Sequence[SourceFailure]) -> None: ...

class ExactProductSetError(AcquisitionError):
    missing: Tuple[ProductIdentity, ...]
    unexpected: Tuple[ProductIdentity, ...]
    duplicate_expected: Tuple[ProductIdentity, ...]
    duplicate_available: Tuple[ProductIdentity, ...]

def identity(product: _data.Product) -> ProductIdentity: ...
def request(
    product: _data.Product,
    distributors: Sequence[Union[Distribution, DistributionSource]],
) -> ProductRequest: ...
def cddis_url(product_identity: ProductIdentity) -> str: ...
def validate_exact_product_set(
    expected: Sequence[ProductIdentity], available: Sequence[ProductIdentity]
) -> None: ...
def acquire(
    exact_request: ProductRequest,
    *,
    cache_dir: Optional[Union[str, os.PathLike[str]]] = ...,
    offline: bool = ...,
    earthdata_auth: Optional[EarthdataAuth] = ...,
    sha256: Optional[str] = ...,
    timeout_s: float = ...,
    retries: int = ...,
    backoff_s: float = ...,
    max_archive_bytes: int = ...,
    max_product_bytes: int = ...,
    http_client: Optional[httpx.Client] = ...,
) -> AcquiredProduct: ...

__all__: list[str]
