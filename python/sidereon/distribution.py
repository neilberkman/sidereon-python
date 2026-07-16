"""Exact GNSS product acquisition from explicit public distributors.

Product identity is independent from transport.  This module keeps that
boundary explicit while adding authenticated NASA CDDIS/Earthdata access to
the existing direct-analysis-center and caller-provided input paths.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import netrc
import os
import re
import tempfile
import threading
import time
import zlib
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union, cast
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

import sidereon
from sidereon import data as _data


class DistributionSource(Enum):
    """Public distributor or caller-provided input for an exact product."""

    DIRECT = "direct"
    NASA_CDDIS = "nasa_cddis"
    LOCAL_FILE = "local_file"
    IN_MEMORY = "in_memory"


@dataclass(frozen=True)
class Distribution:
    """One allowed distributor and any source-specific caller input."""

    source: DistributionSource
    path: Optional[str] = None
    content: Optional[bytes] = field(default=None, repr=False, compare=False)
    compression: Optional[str] = None

    def __post_init__(self) -> None:
        if self.source is DistributionSource.LOCAL_FILE:
            if not self.path or self.content is not None:
                raise ValueError("local_file distribution requires path only")
        elif self.source is DistributionSource.IN_MEMORY:
            if self.content is None or self.path is not None:
                raise ValueError("in_memory distribution requires content only")
        elif self.path is not None or self.content is not None:
            raise ValueError("network distributions do not accept path or content")
        if self.compression not in (None, "none", "gzip", "auto"):
            raise ValueError("compression must be none, gzip, auto, or None")

    @classmethod
    def direct(cls) -> "Distribution":
        return cls(DistributionSource.DIRECT)

    @classmethod
    def nasa_cddis(cls) -> "Distribution":
        return cls(DistributionSource.NASA_CDDIS)

    @classmethod
    def local_file(
        cls, path: Union[str, os.PathLike[str]], *, compression: str = "auto"
    ) -> "Distribution":
        return cls(
            DistributionSource.LOCAL_FILE, path=os.fspath(path), compression=compression
        )

    @classmethod
    def in_memory(cls, content: bytes, *, compression: str = "auto") -> "Distribution":
        return cls(
            DistributionSource.IN_MEMORY,
            content=bytes(content),
            compression=compression,
        )


@dataclass(frozen=True)
class ProductIdentity:
    """Exact public GNSS product identity, independent of distributor."""

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
    format_version: Optional[str] = None
    prediction_horizon_days: Optional[int] = None

    @property
    def key(self) -> str:
        """Stable cache key that retains every exact-request discriminator."""
        _validate_requested_identity(self)
        fields = (
            self.family,
            self.analysis_center,
            self.publisher,
            self.solution_class,
            self.campaign,
            str(self.filename_version),
            self.date.isoformat(),
            self.issue,
            self.span,
            self.sample,
            self.official_filename,
            self.format,
            self.format_version or "",
            ""
            if self.prediction_horizon_days is None
            else str(self.prediction_horizon_days),
        )
        digest = hashlib.sha256("\x00".join(fields).encode("ascii")).hexdigest()[:20]
        return f"{self.publisher.lower()}-{self.solution_class}-{digest}"

    def to_dict(self) -> dict:
        value = asdict(self)
        value["date"] = self.date.isoformat()
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ProductIdentity":
        fields = dict(value)
        fields["date"] = dt.date.fromisoformat(str(fields["date"]))
        return cls(**fields)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ProductRequest:
    """Exact product identity and ordered acceptable distributors."""

    identity: ProductIdentity
    distributors: Tuple[Distribution, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "distributors", tuple(self.distributors))
        if not self.distributors:
            raise ValueError("exact product request requires at least one distributor")


@dataclass(frozen=True)
class EarthdataAuth:
    """Caller-supplied Earthdata credential mechanism.

    Bearer tokens and netrc-derived passwords are excluded from repr/equality
    and are never copied into failures or provenance.
    """

    bearer_token: Optional[str] = field(default=None, repr=False, compare=False)
    use_netrc: bool = False
    netrc_path: Optional[str] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.bearer_token is not None and not self.bearer_token.strip():
            raise ValueError("bearer_token must not be blank")
        if self.bearer_token is not None and self.use_netrc:
            raise ValueError("choose bearer_token or netrc, not both")

    @classmethod
    def bearer(cls, token: str) -> "EarthdataAuth":
        return cls(bearer_token=token)

    @classmethod
    def from_netrc(
        cls, path: Optional[Union[str, os.PathLike[str]]] = None
    ) -> "EarthdataAuth":
        return cls(use_netrc=True, netrc_path=None if path is None else os.fspath(path))

    @property
    def configured(self) -> bool:
        return self.bearer_token is not None or self.use_netrc


@dataclass(frozen=True)
class SourceFailure:
    """Sanitized structured failure from one explicitly allowed source."""

    source: DistributionSource
    error_type: str
    message: str
    url: Optional[str] = None
    status: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "error_type": self.error_type,
            "message": self.message,
            "url": self.url,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "SourceFailure":
        return cls(
            source=DistributionSource(str(value["source"])),
            error_type=str(value["error_type"]),
            message=str(value["message"]),
            url=None if value.get("url") is None else str(value["url"]),
            status=(None if value.get("status") is None else int(str(value["status"]))),
        )


@dataclass(frozen=True)
class AcquisitionProvenance:
    """Reproducible, secret-free provenance for a successful acquisition."""

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
    attempts: Tuple[SourceFailure, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["requested_identity"] = self.requested_identity.to_dict()
        value["resolved_identity"] = self.resolved_identity.to_dict()
        value["distribution_source"] = self.distribution_source.value
        value["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        value["schema_version"] = 1
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "AcquisitionProvenance":
        attempts = cast(Sequence[Mapping[str, object]], value.get("attempts", []))
        return cls(
            requested_identity=ProductIdentity.from_dict(
                cast(Mapping[str, object], value["requested_identity"])
            ),
            resolved_identity=ProductIdentity.from_dict(
                cast(Mapping[str, object], value["resolved_identity"])
            ),
            publisher=str(value["publisher"]),
            distribution_source=DistributionSource(str(value["distribution_source"])),
            official_filename=str(value["official_filename"]),
            original_url=None
            if value.get("original_url") is None
            else str(value["original_url"]),
            final_url=None
            if value.get("final_url") is None
            else str(value["final_url"]),
            retrieved_at=str(value["retrieved_at"]),
            byte_length=int(str(value["byte_length"])),
            sha256=str(value["sha256"]),
            etag=None if value.get("etag") is None else str(value["etag"]),
            last_modified=(
                None
                if value.get("last_modified") is None
                else str(value["last_modified"])
            ),
            cache_hit=bool(value["cache_hit"]),
            archive_compression=str(value["archive_compression"]),
            archive_byte_length=int(str(value["archive_byte_length"])),
            archive_sha256=str(value["archive_sha256"]),
            attempts=tuple(SourceFailure.from_dict(item) for item in attempts),
        )


@dataclass(frozen=True)
class AcquiredProduct:
    """Verified local product path plus acquisition provenance."""

    path: str
    provenance: AcquisitionProvenance


class AcquisitionError(_data.DataError):
    """Base class for exact-acquisition failures."""

    code = "acquisition_error"


class UnsupportedDistribution(AcquisitionError):
    code = "unsupported_distribution"


class HttpAcquisitionError(AcquisitionError):
    """HTTP failure carrying only a sanitized public URL and status."""

    def __init__(self, status: int, url: str, message: str) -> None:
        self.status = status
        self.url = _sanitize_url(url)
        super().__init__(f"{message} ({status}) at {self.url}")


class ProductNotPublished(HttpAcquisitionError):
    code = "product_not_published"


class AuthenticationRequired(HttpAcquisitionError):
    code = "authentication_required"


class AuthenticationFailed(HttpAcquisitionError):
    code = "authentication_failed"


class AuthorizationDenied(HttpAcquisitionError):
    code = "authorization_denied"


class RedirectPolicyFailure(AcquisitionError):
    code = "redirect_policy_failure"


class RetiredEndpoint(HttpAcquisitionError):
    code = "retired_endpoint"


class MalformedUrl(AcquisitionError):
    code = "malformed_url"


class TransportFailure(AcquisitionError):
    code = "transport_failure"

    def __init__(self, kind: str, url: str) -> None:
        self.kind = kind
        self.url = _sanitize_url(url)
        self.status: Optional[int] = None
        super().__init__(f"{kind} transport failure for {self.url}")


class InvalidContentType(AcquisitionError):
    code = "invalid_content_type"


class ErrorDocument(AcquisitionError):
    code = "error_document"


class ContentLengthMismatch(AcquisitionError):
    code = "content_length_mismatch"


class DecompressionFailure(AcquisitionError):
    code = "decompression_failure"


class ProductValidationFailure(AcquisitionError):
    code = "product_validation_failure"


class CacheReadFailure(AcquisitionError):
    code = "cache_read_failure"


class CacheWriteFailure(AcquisitionError):
    code = "cache_write_failure"


class AllDistributorsFailed(AcquisitionError):
    code = "all_distributors_failed"

    def __init__(self, attempts: Sequence[SourceFailure]) -> None:
        self.attempts = tuple(attempts)
        detail = ", ".join(
            f"{item.source.value}={item.error_type}" for item in self.attempts
        )
        super().__init__(f"all explicitly allowed distributors failed ({detail})")


_SOLUTION_NAMES = {
    "FIN": "final",
    "RAP": "rapid",
    "ULT": "ultra_rapid",
    "PRD": "predicted",
}
_PREDICTION_HORIZON = {"cod_prd1": 1, "cod_prd2": 2}
_AIUB_OBJECT_STORE_SUFFIX = ".s3.cloud.switch.ch"
_MAX_REDIRECTS = 8
_DEFAULT_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_DEFAULT_MAX_PRODUCT_BYTES = 500 * 1024 * 1024

_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}


def identity(product: _data.Product) -> ProductIdentity:
    """Resolve a legacy :class:`sidereon.data.Product` to exact identity."""
    filename = product.canonical_filename()
    if not _safe_filename(filename):
        raise _data.UnsupportedProduct(f"unsafe official filename: {filename!r}")
    parts = filename.split("_")
    if len(parts) < 5 or len(parts[0]) < 10 or len(parts[1]) != 11:
        raise _data.UnsupportedProduct(f"unsupported official filename: {filename!r}")
    token, timestamp, span, sample = parts[:4]
    suffix = parts[-1]
    publisher = token[:3]
    try:
        filename_version = int(token[3])
    except ValueError:
        raise _data.UnsupportedProduct(
            f"invalid filename version: {filename!r}"
        ) from None
    campaign = token[4:7]
    solution_token = token[7:10]
    solution = _SOLUTION_NAMES.get(solution_token)
    if solution is None:
        raise _data.UnsupportedProduct(
            f"unsupported solution token: {solution_token!r}"
        )
    expected_suffix = {"sp3": "ORB.SP3", "ionex": "GIM.INX"}.get(product.content)
    if expected_suffix is None or suffix != expected_suffix:
        raise _data.UnsupportedProduct(
            "exact distribution acquisition currently supports SP3 and IONEX"
        )
    return ProductIdentity(
        family=product.content,
        analysis_center=product.center,
        publisher=publisher,
        solution_class=solution,
        campaign=campaign,
        filename_version=filename_version,
        date=product.date,
        issue=timestamp[-4:],
        span=span,
        sample=sample,
        official_filename=filename,
        format="SP3" if product.content == "sp3" else "IONEX",
        prediction_horizon_days=_PREDICTION_HORIZON.get(product.center),
    )


def request(
    product: _data.Product,
    distributors: Sequence[Union[Distribution, DistributionSource]],
) -> ProductRequest:
    """Build an exact request without permitting product substitution."""
    normalized = tuple(
        item if isinstance(item, Distribution) else Distribution(item)
        for item in distributors
    )
    product_identity = identity(product)
    _validate_requested_identity(product_identity)
    return ProductRequest(product_identity, normalized)


def cddis_url(product_identity: ProductIdentity) -> str:
    """Official NASA CDDIS HTTPS URL for an exact SP3 or IONEX product."""
    _validate_requested_identity(product_identity)
    filename = product_identity.official_filename
    if not _safe_filename(filename):
        raise MalformedUrl("unsafe official filename")
    if product_identity.family == "sp3":
        week = _data.gps_week(product_identity.date)
        return f"https://cddis.nasa.gov/archive/gnss/products/{week}/{filename}.gz"
    if product_identity.family == "ionex":
        doy = _data.day_of_year(product_identity.date)
        return (
            "https://cddis.nasa.gov/archive/gnss/products/ionex/"
            f"{product_identity.date.year}/{doy:03d}/{filename}.gz"
        )
    raise UnsupportedDistribution(
        f"NASA CDDIS source does not support {product_identity.family}"
    )


def acquire(
    exact_request: ProductRequest,
    *,
    cache_dir: Optional[Union[str, os.PathLike[str]]] = None,
    offline: bool = False,
    earthdata_auth: Optional[EarthdataAuth] = None,
    sha256: Optional[str] = None,
    timeout_s: float = 30.0,
    retries: int = 3,
    backoff_s: float = 0.5,
    max_archive_bytes: int = _DEFAULT_MAX_ARCHIVE_BYTES,
    max_product_bytes: int = _DEFAULT_MAX_PRODUCT_BYTES,
    http_client: Optional[httpx.Client] = None,
) -> AcquiredProduct:
    """Acquire the exact product from only the ordered allowed distributors."""
    _validate_requested_identity(exact_request.identity)
    if retries < 1:
        raise ValueError("retries must be at least one")
    if timeout_s <= 0 or backoff_s < 0:
        raise ValueError("timeout_s must be positive and backoff_s non-negative")
    if max_archive_bytes <= 0 or max_product_bytes <= 0:
        raise ValueError("byte limits must be positive")
    auth = earthdata_auth or EarthdataAuth()
    root = Path(cache_dir) if cache_dir is not None else Path(_data.default_cache_dir())
    attempts: list[SourceFailure] = []
    last_error: Optional[_data.DataError] = None

    for distributor in exact_request.distributors:
        path = _cache_path(root, exact_request.identity, distributor.source)
        lock = _path_lock(path)
        with lock:
            cache_error: Optional[_data.DataError] = None
            try:
                cached = _load_cached(
                    path,
                    exact_request.identity,
                    distributor.source,
                    attempts,
                    sha256,
                )
            except _data.DataError as caught:
                cached = None
                cache_error = caught
            if cached is not None:
                return cached
            if offline and distributor.source in (
                DistributionSource.DIRECT,
                DistributionSource.NASA_CDDIS,
            ):
                error: _data.DataError = cache_error or _data.OfflineCacheMiss(
                    f"exact product not cached from {distributor.source.value}"
                )
            else:
                try:
                    return _acquire_one(
                        exact_request.identity,
                        distributor,
                        path,
                        auth,
                        sha256,
                        timeout_s,
                        retries,
                        backoff_s,
                        max_archive_bytes,
                        max_product_bytes,
                        http_client,
                        attempts,
                    )
                except (_data.DataError, OSError) as caught:
                    error = _normalize_error(caught)
            failure = _source_failure(distributor.source, error)
            attempts.append(failure)
            last_error = error

    if len(attempts) == 1 and last_error is not None:
        raise last_error
    raise AllDistributorsFailed(attempts)


@dataclass(frozen=True)
class _Download:
    archive: bytes
    original_url: str
    final_url: str
    etag: Optional[str]
    last_modified: Optional[str]
    content_type: Optional[str]


def _acquire_one(
    requested: ProductIdentity,
    distributor: Distribution,
    path: Path,
    auth: EarthdataAuth,
    expected_sha256: Optional[str],
    timeout_s: float,
    retries: int,
    backoff_s: float,
    max_archive_bytes: int,
    max_product_bytes: int,
    http_client: Optional[httpx.Client],
    attempts: Sequence[SourceFailure],
) -> AcquiredProduct:
    source = distributor.source
    if source is DistributionSource.DIRECT:
        product = _product_from_identity(requested)
        original_url = product.archive_url()
        compression = product.archive_compression()
        download = _download_http(
            original_url,
            source,
            auth,
            timeout_s,
            retries,
            backoff_s,
            max_archive_bytes,
            http_client,
        )
        archive = download.archive
    elif source is DistributionSource.NASA_CDDIS:
        original_url = cddis_url(requested)
        compression = "gzip"
        download = _download_http(
            original_url,
            source,
            auth,
            timeout_s,
            retries,
            backoff_s,
            max_archive_bytes,
            http_client,
        )
        archive = download.archive
    elif source is DistributionSource.LOCAL_FILE:
        original_url = None
        try:
            archive = Path(distributor.path or "").read_bytes()
        except OSError as exc:
            raise CacheReadFailure("unable to read caller-provided local file") from exc
        if len(archive) > max_archive_bytes:
            raise _data.DownloadSizeExceeded(
                f"caller-provided archive exceeded {max_archive_bytes} bytes"
            )
        compression = _detect_compression(archive, distributor.compression)
        download = _Download(archive, "", "", None, None, None)
    elif source is DistributionSource.IN_MEMORY:
        original_url = None
        archive = distributor.content or b""
        if len(archive) > max_archive_bytes:
            raise _data.DownloadSizeExceeded(
                f"caller-provided archive exceeded {max_archive_bytes} bytes"
            )
        compression = _detect_compression(archive, distributor.compression)
        download = _Download(archive, "", "", None, None, None)
    else:  # pragma: no cover - Enum makes this unreachable
        raise UnsupportedDistribution(f"unsupported source {source!r}")

    content = _decompress(archive, compression, max_product_bytes)
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise _data.ChecksumMismatch(expected_sha256.lower(), digest)
    resolved = _validate_product(requested, content)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    provenance = AcquisitionProvenance(
        requested_identity=requested,
        resolved_identity=resolved,
        publisher=requested.publisher,
        distribution_source=source,
        official_filename=requested.official_filename,
        original_url=None if original_url is None else _sanitize_url(original_url),
        final_url=None if not download.final_url else _sanitize_url(download.final_url),
        retrieved_at=now,
        byte_length=len(content),
        sha256=digest,
        etag=download.etag,
        last_modified=download.last_modified,
        cache_hit=False,
        archive_compression=compression,
        archive_byte_length=len(archive),
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        attempts=tuple(attempts),
    )
    _commit_cache(path, content, archive, provenance)
    return AcquiredProduct(str(path), provenance)


def _product_from_identity(value: ProductIdentity) -> _data.Product:
    issue = value.issue if value.solution_class == "ultra_rapid" else None
    return _data.Product(
        value.analysis_center,
        value.family,
        value.date,
        value.sample,
        issue=issue,
        filename=value.official_filename,
    )


def _validate_requested_identity(value: ProductIdentity) -> None:
    """Reject caller-constructed identities that disagree with the catalog."""
    if not isinstance(value, ProductIdentity):
        raise ProductValidationFailure("invalid requested product identity")
    try:
        issue = value.issue if value.solution_class == "ultra_rapid" else None
        expected = identity(
            _data.Product(
                value.analysis_center,
                value.family,
                value.date,
                value.sample,
                issue=issue,
            )
        )
    except (ValueError, TypeError, _data.DataError):
        raise ProductValidationFailure("invalid requested product identity") from None
    if replace(value, format_version=None) != expected:
        raise ProductValidationFailure(
            "requested identity disagrees with its official filename"
        )


def _download_http(
    original_url: str,
    source: DistributionSource,
    auth: EarthdataAuth,
    timeout_s: float,
    retries: int,
    backoff_s: float,
    max_bytes: int,
    supplied_client: Optional[httpx.Client],
) -> _Download:
    last_error: Optional[AcquisitionError] = None
    own_client = supplied_client is None
    client = supplied_client or httpx.Client(follow_redirects=False)
    try:
        for attempt in range(retries):
            try:
                return _download_http_once(
                    client, original_url, source, auth, timeout_s, max_bytes
                )
            except AcquisitionError as error:
                last_error = error
                if not _retryable(error) or attempt + 1 == retries:
                    raise
                if backoff_s:
                    time.sleep(backoff_s * (2**attempt))
        raise last_error or TransportFailure("other", original_url)
    finally:
        if own_client:
            client.close()


def _download_http_once(
    client: httpx.Client,
    original_url: str,
    source: DistributionSource,
    auth: EarthdataAuth,
    timeout_s: float,
    max_bytes: int,
) -> _Download:
    current = original_url
    for redirect_count in range(_MAX_REDIRECTS + 1):
        _validate_url(current, source)
        headers = _auth_headers(current, source, auth)
        try:
            with client.stream(
                "GET",
                current,
                headers=headers,
                follow_redirects=False,
                timeout=timeout_s,
            ) as response:
                status = response.status_code
                if 300 <= status < 400:
                    location = response.headers.get("location")
                    if location is None or redirect_count == _MAX_REDIRECTS:
                        raise RedirectPolicyFailure(
                            f"redirect policy failed for {_sanitize_url(current)}"
                        )
                    current = _redirect_target(current, location, source)
                    continue
                if status == 401:
                    kind = (
                        AuthenticationFailed
                        if auth.configured
                        else AuthenticationRequired
                    )
                    public_url = _sanitize_url(current)
                    raise kind(
                        status,
                        current,
                        f"authentication rejected for {public_url}",
                    )
                if status == 403:
                    raise AuthorizationDenied(status, current, "authorization denied")
                if status == 404:
                    raise ProductNotPublished(
                        status, current, "exact product is not published"
                    )
                if status == 410:
                    raise RetiredEndpoint(status, current, "retired endpoint")
                if status < 200 or status >= 300:
                    error = TransportFailure(f"http_{status}", current)
                    error.status = status
                    raise error
                archive = bytearray()
                chunks = (
                    (response.content,)
                    if response.is_stream_consumed
                    else response.iter_raw()
                )
                for chunk in chunks:
                    archive.extend(chunk)
                    if len(archive) > max_bytes:
                        raise _data.DownloadSizeExceeded(
                            f"archive payload exceeded {max_bytes} bytes"
                        )
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        expected = int(declared)
                    except ValueError:
                        raise ContentLengthMismatch(
                            f"invalid Content-Length for {_sanitize_url(current)}"
                        ) from None
                    if expected != len(archive):
                        raise ContentLengthMismatch(
                            f"Content-Length mismatch for {_sanitize_url(current)}"
                        )
                content_type = response.headers.get("content-type")
                _reject_error_document(bytes(archive), content_type, current)
                return _Download(
                    bytes(archive),
                    original_url,
                    current,
                    response.headers.get("etag"),
                    response.headers.get("last-modified"),
                    content_type,
                )
        except httpx.InvalidURL:
            raise MalformedUrl(f"malformed URL {_sanitize_url(current)}") from None
        except httpx.TimeoutException:
            raise TransportFailure("timeout", current) from None
        except httpx.ConnectError:
            raise TransportFailure("connection", current) from None
        except httpx.RequestError:
            raise TransportFailure("other", current) from None
    raise RedirectPolicyFailure(f"too many redirects for {_sanitize_url(original_url)}")


def _auth_headers(
    url: str, source: DistributionSource, auth: EarthdataAuth
) -> Mapping[str, str]:
    if source is not DistributionSource.NASA_CDDIS:
        return {}
    host = (urlsplit(url).hostname or "").lower()
    if host not in {"cddis.nasa.gov", "urs.earthdata.nasa.gov"}:
        return {}
    if auth.bearer_token is not None:
        return {"authorization": f"Bearer {auth.bearer_token}"}
    if auth.use_netrc and host == "urs.earthdata.nasa.gov":
        try:
            credentials = netrc.netrc(auth.netrc_path).authenticators(host)
        except (netrc.NetrcParseError, OSError):
            raise AuthenticationFailed(
                0, url, "unable to read Earthdata netrc credentials"
            ) from None
        if credentials is None:
            raise AuthenticationRequired(
                0, url, "Earthdata netrc has no URS credentials"
            )
        login, _, password = credentials
        token = base64.b64encode(f"{login}:{password}".encode()).decode("ascii")
        return {"authorization": f"Basic {token}"}
    return {}


def _validate_url(url: str, source: DistributionSource) -> None:
    parts = urlsplit(url)
    allowed_schemes = (
        {"https"} if source is DistributionSource.NASA_CDDIS else {"http", "https"}
    )
    if (
        parts.scheme not in allowed_schemes
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise MalformedUrl(f"malformed or insecure URL {_sanitize_url(url)}")
    host = parts.hostname.lower()
    if source is DistributionSource.NASA_CDDIS:
        if host not in {"cddis.nasa.gov", "urs.earthdata.nasa.gov"}:
            raise RedirectPolicyFailure(f"NASA redirect host refused: {host}")
    elif host not in _data.allowed_hosts() and not host.endswith(
        _AIUB_OBJECT_STORE_SUFFIX
    ):
        raise RedirectPolicyFailure(f"direct-source host refused: {host}")


def _redirect_target(current: str, location: str, source: DistributionSource) -> str:
    target = urljoin(current, location)
    current_parts = urlsplit(current)
    target_parts = urlsplit(target)
    source_host = (current_parts.hostname or "").lower()
    target_host = (target_parts.hostname or "").lower()
    if target_parts.scheme != "https" or not target_host:
        raise RedirectPolicyFailure(
            f"insecure redirect refused for {_sanitize_url(current)}"
        )
    if source is DistributionSource.NASA_CDDIS:
        allowed = {"cddis.nasa.gov", "urs.earthdata.nasa.gov"}
        if source_host in allowed and target_host in allowed:
            return target
    else:
        if source_host == target_host:
            return target
        if (
            source_host == "www.aiub.unibe.ch"
            and target_host == "download.aiub.unibe.ch"
        ):
            return target
        if source_host in {
            "www.aiub.unibe.ch",
            "download.aiub.unibe.ch",
        } and target_host.endswith(_AIUB_OBJECT_STORE_SUFFIX):
            return target
    raise RedirectPolicyFailure(f"redirect host refused for {_sanitize_url(current)}")


def _sanitize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port is not None:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "<invalid-url>"


def _reject_error_document(
    content: bytes, content_type: Optional[str], url: str
) -> None:
    media_type = (content_type or "").split(";", 1)[0].strip().lower()
    prefix = content[:512].lstrip().lower()
    if media_type in {"text/html", "application/xhtml+xml"}:
        raise InvalidContentType(f"HTML content type refused for {_sanitize_url(url)}")
    if prefix.startswith((b"<!doctype html", b"<html", b"<head", b"<body")):
        raise ErrorDocument(f"HTML error document refused for {_sanitize_url(url)}")


def _retryable(error: AcquisitionError) -> bool:
    if isinstance(error, TransportFailure):
        status = getattr(error, "status", None)
        return error.kind in {
            "timeout",
            "connection",
            "other",
            "http_408",
            "http_429",
        } or (isinstance(status, int) and status >= 500)
    return False


def _detect_compression(content: bytes, requested: Optional[str]) -> str:
    if requested in (None, "auto"):
        return "gzip" if content.startswith(b"\x1f\x8b") else "none"
    return requested


def _decompress(content: bytes, compression: str, limit: int) -> bytes:
    if compression == "none":
        if len(content) > limit:
            raise _data.DownloadSizeExceeded(f"product exceeded {limit} bytes")
        return content
    if compression != "gzip":
        raise DecompressionFailure(f"unsupported compression {compression!r}")
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        output = decompressor.decompress(content, limit + 1)
        if len(output) > limit or decompressor.unconsumed_tail:
            raise _data.DownloadSizeExceeded(
                f"decompressed product exceeded {limit} bytes"
            )
        output += decompressor.flush(limit + 1 - len(output))
    except zlib.error:
        raise DecompressionFailure("invalid or truncated gzip product") from None
    if len(output) > limit:
        raise _data.DownloadSizeExceeded(f"decompressed product exceeded {limit} bytes")
    if not decompressor.eof or decompressor.unused_data:
        raise DecompressionFailure("invalid, truncated, or concatenated gzip product")
    return output


def _validate_product(requested: ProductIdentity, content: bytes) -> ProductIdentity:
    if not content:
        raise ProductValidationFailure("empty product")
    try:
        if requested.family == "sp3":
            sp3 = sidereon.load_sp3(content)
            if sp3.epoch_count == 0:
                raise ProductValidationFailure("SP3 product has no epochs")
            version = _sp3_version(content)
            _validate_sp3_metadata(requested, content)
            format_version = f"SP3-{version}"
        elif requested.family == "ionex":
            ionex = sidereon.load_ionex(content)
            if len(ionex.map_epochs_j2000_s) == 0:
                raise ProductValidationFailure("IONEX product has no maps")
            version = _ionex_version(content)
            _validate_ionex_metadata(requested, content)
            format_version = f"IONEX-{version}"
        else:
            raise UnsupportedDistribution(
                f"validation is unavailable for {requested.family}"
            )
    except ProductValidationFailure:
        raise
    except Exception as exc:
        if isinstance(exc, _data.DataError):
            raise
        raise ProductValidationFailure(
            f"{requested.format} parse or semantic validation failed"
        ) from None
    if (
        requested.format_version is not None
        and requested.format_version != format_version
    ):
        raise ProductValidationFailure(
            "parsed format version differs from exact request"
        )
    return replace(requested, format_version=format_version)


def _sp3_version(content: bytes) -> str:
    if len(content) < 2 or content[:1] != b"#" or content[1:2].lower() not in b"abcd":
        raise ProductValidationFailure("invalid SP3 version header")
    return content[1:2].decode("ascii").lower()


def _validate_sp3_metadata(requested: ProductIdentity, content: bytes) -> None:
    first = content.splitlines()[0].decode("ascii", "strict")
    match = re.match(
        r"^#[a-dA-D][PV](\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})", first
    )
    if match is None:
        raise ProductValidationFailure("SP3 first epoch is malformed")
    year, month, day, hour, minute = map(int, match.groups())
    if (year, month, day) != (
        requested.date.year,
        requested.date.month,
        requested.date.day,
    ) or f"{hour:02d}{minute:02d}" != requested.issue:
        raise ProductValidationFailure("SP3 coverage start differs from exact request")
    lines = content.splitlines()
    if len(lines) < 2:
        raise ProductValidationFailure("SP3 header is truncated")
    second = lines[1].decode("ascii", "strict")
    try:
        cadence = float(second[24:38])
    except ValueError:
        raise ProductValidationFailure("SP3 cadence is malformed") from None
    expected = _sample_seconds(requested.sample)
    if expected is not None and abs(cadence - expected) > 1e-6:
        raise ProductValidationFailure("SP3 cadence differs from exact request")


def _ionex_version(content: bytes) -> str:
    first = content.splitlines()[0].decode("ascii", "strict")
    if "IONEX VERSION / TYPE" not in first:
        raise ProductValidationFailure("invalid IONEX version header")
    version = first[:20].strip()
    if not version:
        raise ProductValidationFailure("IONEX version is missing")
    return version


def _validate_ionex_metadata(requested: ProductIdentity, content: bytes) -> None:
    text = content.decode("ascii", "strict")
    first_epoch: Optional[Tuple[int, int, int, int, int]] = None
    interval: Optional[int] = None
    map_epochs: list[dt.datetime] = []
    for line in text.splitlines():
        label = line[60:].strip() if len(line) >= 60 else ""
        if label == "EPOCH OF FIRST MAP":
            fields = line[:36].split()
            if len(fields) >= 5:
                first_epoch = tuple(map(int, fields[:5]))  # type: ignore[assignment]
        elif label == "INTERVAL":
            try:
                interval = int(line[:10].strip())
            except ValueError:
                raise ProductValidationFailure("IONEX cadence is malformed") from None
        elif label == "EPOCH OF CURRENT MAP":
            fields = line[:36].split()
            if len(fields) >= 5:
                year, month, day, hour, minute = map(int, fields[:5])
                map_epochs.append(
                    dt.datetime(
                        year,
                        month,
                        day,
                        hour,
                        minute,
                        tzinfo=dt.timezone.utc,
                    )
                )
    if first_epoch is None and map_epochs:
        epoch = map_epochs[0]
        first_epoch = (epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute)
    if interval is None and len(map_epochs) >= 2:
        interval = int((map_epochs[1] - map_epochs[0]).total_seconds())
    if first_epoch is None:
        raise ProductValidationFailure("IONEX first-map epoch is missing")
    year, month, day, hour, minute = first_epoch
    if (year, month, day) != (
        requested.date.year,
        requested.date.month,
        requested.date.day,
    ) or f"{hour:02d}{minute:02d}" != requested.issue:
        raise ProductValidationFailure(
            "IONEX coverage start differs from exact request"
        )
    expected = _sample_seconds(requested.sample)
    if interval is not None and expected is not None and interval != expected:
        raise ProductValidationFailure("IONEX cadence differs from exact request")


def _sample_seconds(sample: str) -> Optional[int]:
    if len(sample) != 3 or not sample[:2].isdigit():
        return None
    value = int(sample[:2])
    return {"S": value, "M": value * 60, "H": value * 3600, "D": value * 86400}.get(
        sample[2]
    )


def _cache_path(
    root: Path, product_identity: ProductIdentity, source: DistributionSource
) -> Path:
    _validate_requested_identity(product_identity)
    filename = product_identity.official_filename
    if not _safe_filename(filename):
        raise CacheWriteFailure("unsafe official filename")
    return (
        root
        / "products"
        / "v1"
        / source.value
        / product_identity.family
        / product_identity.key
        / filename
    )


def _safe_filename(filename: str) -> bool:
    return (
        bool(filename)
        and filename not in {".", ".."}
        and Path(filename).name == filename
        and ".." not in filename
    )


def _path_lock(path: Path) -> threading.Lock:
    key = str(path)
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def _load_cached(
    path: Path,
    requested: ProductIdentity,
    source: DistributionSource,
    attempts: Sequence[SourceFailure],
    expected_sha256: Optional[str],
) -> Optional[AcquiredProduct]:
    if not path.exists():
        return None
    sidecar = _provenance_path(path)
    try:
        content = path.read_bytes()
        archive = _archive_path(path).read_bytes()
        provenance = AcquisitionProvenance.from_dict(json.loads(sidecar.read_text()))
    except OSError:
        raise CacheReadFailure(
            f"cannot read cached exact product {path.name}"
        ) from None
    except (ValueError, KeyError, TypeError):
        raise CacheReadFailure(
            f"invalid cached provenance for exact product {path.name}"
        ) from None
    if (
        provenance.requested_identity != requested
        or provenance.distribution_source is not source
    ):
        raise CacheReadFailure(f"cached identity mismatch for {path.name}")
    digest = hashlib.sha256(content).hexdigest()
    if provenance.sha256 != digest or provenance.byte_length != len(content):
        raise CacheReadFailure(f"cached content checksum mismatch for {path.name}")
    if provenance.archive_sha256 != hashlib.sha256(
        archive
    ).hexdigest() or provenance.archive_byte_length != len(archive):
        raise CacheReadFailure(f"cached archive checksum mismatch for {path.name}")
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise _data.ChecksumMismatch(expected_sha256.lower(), digest)
    try:
        resolved = _validate_product(requested, content)
    except AcquisitionError:
        raise CacheReadFailure(
            f"cached product validation failed for {path.name}"
        ) from None
    if resolved != provenance.resolved_identity:
        raise CacheReadFailure(f"cached resolved identity mismatch for {path.name}")
    return AcquiredProduct(
        str(path), replace(provenance, cache_hit=True, attempts=tuple(attempts))
    )


def _provenance_path(path: Path) -> Path:
    return Path(f"{path}.provenance.json")


def _archive_path(path: Path) -> Path:
    return Path(f"{path}.archive")


def _commit_cache(
    path: Path,
    content: bytes,
    archive: bytes,
    provenance: AcquisitionProvenance,
) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise CacheWriteFailure(
            f"cannot create cache directory {path.parent}"
        ) from None
    temporary: list[Path] = []
    try:
        data_tmp = _temp_file(path.parent, content)
        temporary.append(data_tmp)
        archive_tmp = _temp_file(path.parent, archive)
        temporary.append(archive_tmp)
        provenance_tmp = _temp_file(
            path.parent,
            json.dumps(provenance.to_dict(), indent=2, sort_keys=True).encode("utf-8"),
        )
        temporary.append(provenance_tmp)
        os.replace(archive_tmp, _archive_path(path))
        temporary.remove(archive_tmp)
        os.replace(provenance_tmp, _provenance_path(path))
        temporary.remove(provenance_tmp)
        os.replace(data_tmp, path)
        temporary.remove(data_tmp)
    except OSError:
        raise CacheWriteFailure(
            f"cannot atomically commit exact product {path.name}"
        ) from None
    finally:
        for item in temporary:
            try:
                item.unlink()
            except OSError:
                pass


def _temp_file(directory: Path, content: bytes) -> Path:
    name: Optional[str] = None
    try:
        fd, name = tempfile.mkstemp(prefix=".sidereon-", dir=str(directory))
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        return Path(name)
    except OSError:
        if name is not None:
            try:
                Path(name).unlink()
            except OSError:
                pass
        raise CacheWriteFailure(f"cannot write temporary file in {directory}") from None


def _normalize_error(error: BaseException) -> AcquisitionError:
    if isinstance(error, AcquisitionError):
        return error
    if isinstance(error, _data.ChecksumMismatch):
        normalized: AcquisitionError = ProductValidationFailure(
            "content checksum differs from caller pin"
        )
        normalized.code = "checksum_mismatch"
        return normalized
    if isinstance(error, _data.DownloadSizeExceeded):
        normalized = ProductValidationFailure("product exceeded configured byte limit")
        normalized.code = "download_size_exceeded"
        return normalized
    if isinstance(error, _data.OfflineCacheMiss):
        normalized = CacheReadFailure(str(error))
        normalized.code = "offline_cache_miss"
        return normalized
    if isinstance(error, OSError):
        return CacheReadFailure("caller-provided source could not be read")
    return ProductValidationFailure("unexpected acquisition failure")


def _source_failure(
    source: DistributionSource, error: _data.DataError
) -> SourceFailure:
    return SourceFailure(
        source=source,
        error_type=getattr(error, "code", error.__class__.__name__),
        message=str(error),
        url=getattr(error, "url", None),
        status=getattr(error, "status", None),
    )


__all__ = [
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
    "identity",
    "request",
    "cddis_url",
    "acquire",
]
