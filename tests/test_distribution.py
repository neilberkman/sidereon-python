"""Deterministic exact-distributor acquisition tests (no public network)."""

import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx
import pytest
import sidereon.data as data
import sidereon.distribution as distribution
from _helpers import CORE_FIXTURES

SP3_DATE = dt.date(2020, 6, 25)
IONEX_DATE = dt.date(2020, 6, 24)


def _sp3_bytes():
    path = os.path.join(CORE_FIXTURES, "sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3")
    with open(path, "rb") as handle:
        return handle.read()


def _ionex_bytes():
    path = os.path.join(CORE_FIXTURES, "ionex", "synthetic_2map_7x7.20i")
    with open(path, "rb") as handle:
        return handle.read()


def _ionex_bytes_for(date, interval_hours=1):
    lines = _ionex_bytes().decode("ascii").splitlines()
    map_index = 0
    for index, line in enumerate(lines):
        if line.rstrip().endswith("EPOCH OF CURRENT MAP"):
            hour = map_index * interval_hours
            lines[index] = (
                f"{date.year:6d}{date.month:6d}{date.day:6d}"
                f"{hour:6d}{0:6d}{0:6d}{line[36:]}"
            )
            map_index += 1
    return ("\n".join(lines) + "\n").encode("ascii")


def _predicted_ionex_request(center, target, *sources):
    product = data.predicted_ionex(center, target)
    return distribution.request(
        product, sources or [distribution.Distribution.direct()]
    )


def _sp3_request(*sources):
    product = data.mgex_sp3("cod", SP3_DATE)
    return distribution.request(
        product, sources or [distribution.Distribution.nasa_cddis()]
    )


def _client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def _gzip_response(request, body=None, **headers):
    archive = gzip.compress(body or _sp3_bytes(), mtime=0)
    headers = {key.replace("_", "-"): value for key, value in headers.items()}
    return httpx.Response(
        200,
        request=request,
        content=archive,
        headers={"content-type": "application/gzip", **headers},
    )


def test_identity_is_independent_of_distributor_and_paths_are_exact():
    product = data.mgex_sp3("cod", SP3_DATE)
    exact = distribution.request(
        product,
        [
            distribution.Distribution.direct(),
            distribution.Distribution.nasa_cddis(),
        ],
    )

    assert exact.identity.publisher == "COD"
    assert exact.identity.solution_class == "final"
    assert exact.identity.campaign == "MGX"
    assert exact.identity.issue == "0000"
    assert exact.identity.sample == "05M"
    assert exact.identity.official_filename == (
        "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    )
    assert distribution.cddis_url(exact.identity) == (
        "https://cddis.nasa.gov/archive/gnss/products/2111/"
        "COD0MGXFIN_20201770000_01D_05M_ORB.SP3.gz"
    )
    assert [item.source for item in exact.distributors] == [
        distribution.DistributionSource.DIRECT,
        distribution.DistributionSource.NASA_CDDIS,
    ]
    assert data.acquire is distribution.acquire
    assert data.ProductIdentity is distribution.ProductIdentity


def test_ionex_cddis_year_day_path_and_parsed_acquisition(tmp_path):
    product = data.mgex_ionex("esa", IONEX_DATE)
    exact = distribution.request(product, [distribution.Distribution.nasa_cddis()])
    assert distribution.cddis_url(exact.identity) == (
        "https://cddis.nasa.gov/archive/gnss/products/ionex/2020/176/"
        "ESA0OPSFIN_20201760000_01D_02H_GIM.INX.gz"
    )

    def handler(request):
        return _gzip_response(request, _ionex_bytes(), etag='"ionex-etag"')

    with _client(handler) as client:
        result = distribution.acquire(exact, cache_dir=tmp_path, http_client=client)

    assert result.provenance.resolved_identity.format_version == "IONEX-1.1"
    assert result.provenance.etag == '"ionex-etag"'
    assert result.provenance.requested_identity == exact.identity
    with open(result.path, "rb") as handle:
        assert handle.read() == _ionex_bytes()


def test_predicted_ionex_direct_path_and_semantic_identity(tmp_path):
    target = dt.date(2026, 7, 15)
    exact = _predicted_ionex_request("cod_prd1", target)
    expected_url = (
        "https://www.aiub.unibe.ch/download/CODE/IONO/P1/2026/"
        "COD0OPSPRD_20261960000_01D_01H_GIM.INX.gz"
    )

    def handler(request):
        assert str(request.url) == expected_url
        return _gzip_response(request, _ionex_bytes_for(exact.identity.date))

    with _client(handler) as client:
        result = distribution.acquire(exact, cache_dir=tmp_path, http_client=client)
    assert result.provenance.requested_identity == exact.identity
    assert result.provenance.resolved_identity.date == exact.identity.date
    assert result.provenance.original_url == expected_url


def test_predicted_ionex_wrong_date_is_typed_validation_failure(tmp_path):
    exact = _predicted_ionex_request("cod_prd2", dt.date(2026, 7, 15))
    wrong = _ionex_bytes_for(exact.identity.date - dt.timedelta(days=1))
    request = distribution.ProductRequest(
        exact.identity, (distribution.Distribution.in_memory(wrong),)
    )
    with pytest.raises(distribution.ProductValidationFailure):
        distribution.acquire(request, cache_dir=tmp_path)


def test_predicted_tiers_with_same_filename_cannot_share_cache(tmp_path):
    p1 = _predicted_ionex_request("cod_prd1", dt.date(2026, 7, 16)).identity
    p2 = _predicted_ionex_request("cod_prd2", dt.date(2026, 7, 15)).identity
    assert p1.official_filename == p2.official_filename
    assert p1.key != p2.key
    assert distribution._cache_path(
        tmp_path, p1, distribution.DistributionSource.DIRECT
    ) != distribution._cache_path(tmp_path, p2, distribution.DistributionSource.DIRECT)

    seeded = distribution.ProductRequest(
        p1,
        (distribution.Distribution.in_memory(_ionex_bytes_for(p1.date)),),
    )
    distribution.acquire(seeded, cache_dir=tmp_path)
    with pytest.raises(data.OfflineCacheMiss):
        distribution.acquire(
            _predicted_ionex_request("cod_prd2", dt.date(2026, 7, 15)),
            cache_dir=tmp_path,
            offline=True,
        )


def test_exact_predicted_ionex_404_does_not_look_back(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(404, request=request)

    with _client(handler) as client, pytest.raises(distribution.ProductNotPublished):
        distribution.acquire(
            _predicted_ionex_request("cod_prd1", dt.date(2026, 7, 15)),
            cache_dir=tmp_path,
            http_client=client,
            retries=1,
        )
    assert len(calls) == 1


def test_concurrent_different_predicted_products_are_immutable(tmp_path):
    requests = []
    expected = {}
    for center, target in (
        ("cod_prd1", dt.date(2026, 7, 15)),
        ("cod_prd2", dt.date(2026, 7, 15)),
    ):
        identity = _predicted_ionex_request(center, target).identity
        content = _ionex_bytes_for(identity.date)
        exact = distribution.ProductRequest(
            identity, (distribution.Distribution.in_memory(content),)
        )
        requests.append(exact)
        expected[identity.analysis_center] = content

    results = []
    failures = []
    threads = [
        threading.Thread(
            target=lambda exact=exact: _thread_acquire(
                results, failures, exact, tmp_path, None
            )
        )
        for exact in requests
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert len({result.path for result in results}) == 2
    snapshots = {result.path: Path(result.path).read_bytes() for result in results}
    for result in results:
        center = result.provenance.requested_identity.analysis_center
        assert snapshots[result.path] == expected[center]
    assert {path: Path(path).read_bytes() for path in snapshots} == snapshots


def test_earthdata_redirect_cookie_client_and_bearer_are_secret_safe(tmp_path):
    token = "test-token-that-must-never-escape"
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["authorization"] == f"Bearer {token}"
        if len(calls) == 1:
            return httpx.Response(
                302,
                request=request,
                headers={
                    "location": (
                        "https://urs.earthdata.nasa.gov/oauth/authorize"
                        "?state=secret-query-value"
                    ),
                    "set-cookie": "session=not-provenance; Secure; HttpOnly",
                },
            )
        if len(calls) == 2:
            return httpx.Response(
                302,
                request=request,
                headers={
                    "location": (
                        distribution.cddis_url(_sp3_request().identity)
                        + "?download-ticket=secret-query-value"
                    )
                },
            )
        return _gzip_response(request, etag='"sp3-etag"')

    auth = distribution.EarthdataAuth.bearer(token)
    with _client(handler) as client:
        result = distribution.acquire(
            _sp3_request(),
            cache_dir=tmp_path,
            earthdata_auth=auth,
            http_client=client,
        )

    serialized = json.dumps(result.provenance.to_dict())
    assert len(calls) == 3
    assert token not in repr(auth)
    assert token not in serialized
    assert "secret-query-value" not in serialized
    assert result.provenance.final_url.endswith(".SP3.gz")
    assert "?" not in result.provenance.final_url


@pytest.mark.parametrize(
    ("status", "auth", "error_type"),
    [
        (401, None, distribution.AuthenticationRequired),
        (
            401,
            distribution.EarthdataAuth.bearer("bad-secret"),
            distribution.AuthenticationFailed,
        ),
        (403, None, distribution.AuthorizationDenied),
        (404, None, distribution.ProductNotPublished),
        (410, None, distribution.RetiredEndpoint),
    ],
)
def test_http_statuses_remain_distinct(tmp_path, status, auth, error_type):
    def handler(request):
        return httpx.Response(status, request=request)

    with _client(handler) as client, pytest.raises(error_type) as caught:
        distribution.acquire(
            _sp3_request(),
            cache_dir=tmp_path,
            earthdata_auth=auth,
            http_client=client,
            retries=1,
        )
    assert "bad-secret" not in str(caught.value)
    assert caught.value.status == status
    assert "?" not in caught.value.url


def test_timeout_malformed_url_and_retired_endpoint_are_not_not_published(
    tmp_path, monkeypatch
):
    def timeout(request):
        raise httpx.ReadTimeout("synthetic transport detail", request=request)

    with (
        _client(timeout) as client,
        pytest.raises(distribution.TransportFailure) as caught,
    ):
        distribution.acquire(
            _sp3_request(),
            cache_dir=tmp_path / "timeout",
            http_client=client,
            retries=1,
        )
    assert caught.value.kind == "timeout"

    monkeypatch.setattr(distribution, "cddis_url", lambda _identity: "not a URL")
    with pytest.raises(distribution.MalformedUrl):
        distribution.acquire(_sp3_request(), cache_dir=tmp_path / "bad-url", retries=1)

    assert distribution._sanitize_url(
        "https://example.test:not-a-port/path?secret"
    ) == ("<invalid-url>")


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (
            lambda request: httpx.Response(
                200,
                request=request,
                content=b"<html>login failed</html>",
                headers={"content-type": "text/html"},
            ),
            distribution.InvalidContentType,
        ),
        (
            lambda request: httpx.Response(
                200,
                request=request,
                content=b"<!doctype html><title>error</title>",
                headers={"content-type": "application/octet-stream"},
            ),
            distribution.ErrorDocument,
        ),
        (
            lambda request: httpx.Response(
                200,
                request=request,
                content=gzip.compress(_sp3_bytes(), mtime=0)[:-8],
                headers={"content-type": "application/gzip"},
            ),
            distribution.DecompressionFailure,
        ),
    ],
)
def test_error_documents_and_truncated_compression_never_enter_cache(
    tmp_path, response, error_type
):
    with _client(response) as client, pytest.raises(error_type):
        distribution.acquire(
            _sp3_request(), cache_dir=tmp_path, http_client=client, retries=1
        )
    assert not list(tmp_path.rglob("*.SP3"))


def test_decompression_stops_at_the_configured_product_limit(tmp_path):
    exact = distribution.ProductRequest(
        _sp3_request().identity,
        (
            distribution.Distribution.in_memory(
                gzip.compress(_sp3_bytes(), mtime=0), compression="gzip"
            ),
        ),
    )

    with pytest.raises(distribution.ProductValidationFailure) as caught:
        distribution.acquire(exact, cache_dir=tmp_path, max_product_bytes=32)

    assert caught.value.code == "download_size_exceeded"
    assert not list(tmp_path.rglob("*.SP3"))


def test_content_length_and_semantic_identity_mismatches_are_typed(tmp_path):
    def wrong_length(request):
        return httpx.Response(
            200,
            request=request,
            content=gzip.compress(_sp3_bytes(), mtime=0),
            headers={"content-length": "2", "content-type": "application/gzip"},
        )

    with (
        _client(wrong_length) as client,
        pytest.raises(distribution.ContentLengthMismatch),
    ):
        distribution.acquire(
            _sp3_request(), cache_dir=tmp_path / "length", http_client=client, retries=1
        )

    wrong_identity = dataclasses.replace(
        _sp3_request().identity, date=dt.date(2020, 6, 24)
    )
    exact = distribution.ProductRequest(
        wrong_identity, (distribution.Distribution.nasa_cddis(),)
    )
    with (
        _client(_gzip_response) as client,
        pytest.raises(distribution.ProductValidationFailure),
    ):
        distribution.acquire(
            exact, cache_dir=tmp_path / "semantic", http_client=client, retries=1
        )

    inconsistent = dataclasses.replace(_sp3_request().identity, publisher="ESA")
    with pytest.raises(distribution.ProductValidationFailure):
        distribution.cddis_url(inconsistent)

    unsafe = dataclasses.replace(
        _sp3_request().identity, official_filename="../escape.SP3"
    )
    called = False

    def should_not_run(request):
        nonlocal called
        called = True
        return _gzip_response(request)

    with (
        _client(should_not_run) as client,
        pytest.raises(distribution.ProductValidationFailure),
    ):
        distribution.acquire(
            distribution.ProductRequest(
                unsafe, (distribution.Distribution.nasa_cddis(),)
            ),
            cache_dir=tmp_path / "unsafe",
            http_client=client,
        )
    assert not called
    assert not (tmp_path / "unsafe").exists()


def test_local_bytes_and_cddis_match_hashes_and_parsed_identity(tmp_path):
    exact_local = _sp3_request(distribution.Distribution.in_memory(_sp3_bytes()))
    local = distribution.acquire(exact_local, cache_dir=tmp_path / "local")

    with _client(_gzip_response) as client:
        remote = distribution.acquire(
            _sp3_request(), cache_dir=tmp_path / "remote", http_client=client
        )

    assert local.provenance.sha256 == remote.provenance.sha256
    assert local.provenance.resolved_identity == remote.provenance.resolved_identity
    assert (
        local.provenance.distribution_source
        is distribution.DistributionSource.IN_MEMORY
    )
    assert (
        remote.provenance.distribution_source
        is distribution.DistributionSource.NASA_CDDIS
    )


def test_cache_hit_revalidates_content_and_round_trips_provenance(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return _gzip_response(request, last_modified="Thu, 25 Jun 2020 00:00:00 GMT")

    exact = _sp3_request()
    with _client(handler) as client:
        first = distribution.acquire(exact, cache_dir=tmp_path, http_client=client)
        second = distribution.acquire(exact, cache_dir=tmp_path, http_client=client)

    assert calls == 1
    assert not first.provenance.cache_hit
    assert second.provenance.cache_hit
    assert second.provenance.resolved_identity == first.provenance.resolved_identity
    assert second.provenance.last_modified == "Thu, 25 Jun 2020 00:00:00 GMT"
    assert os.path.exists(first.path + ".archive")


def test_offline_corrupt_cache_is_a_cache_read_failure(tmp_path):
    with _client(_gzip_response) as client:
        result = distribution.acquire(
            _sp3_request(), cache_dir=tmp_path, http_client=client
        )
    with open(result.path, "ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(distribution.CacheReadFailure):
        distribution.acquire(_sp3_request(), cache_dir=tmp_path, offline=True)


def test_verified_cache_is_kept_when_remote_transport_would_fail(tmp_path):
    with _client(_gzip_response) as client:
        first = distribution.acquire(
            _sp3_request(), cache_dir=tmp_path, http_client=client
        )

    def fail_if_called(_request):
        raise AssertionError("verified cache hit must not make a remote request")

    with _client(fail_if_called) as client:
        second = distribution.acquire(
            _sp3_request(), cache_dir=tmp_path, http_client=client
        )
    assert second.path == first.path
    assert second.provenance.cache_hit


def test_atomic_commit_failure_leaves_no_visible_product_or_temp(tmp_path, monkeypatch):
    real_replace = distribution.os.replace
    replaces = 0

    def fail_second(source, target):
        nonlocal replaces
        replaces += 1
        if replaces == 2:
            raise OSError("simulated interruption")
        return real_replace(source, target)

    monkeypatch.setattr(distribution.os, "replace", fail_second)
    with (
        _client(_gzip_response) as client,
        pytest.raises(distribution.CacheWriteFailure),
    ):
        distribution.acquire(
            _sp3_request(), cache_dir=tmp_path, http_client=client, retries=1
        )

    assert not list(tmp_path.rglob("*.SP3"))
    assert not list(tmp_path.rglob(".sidereon-*"))


def test_concurrent_requests_download_once(tmp_path):
    calls = 0
    calls_lock = threading.Lock()

    def handler(request):
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        return _gzip_response(request)

    results = []
    failures = []
    exact = _sp3_request()
    with _client(handler) as client:
        threads = [
            threading.Thread(
                target=lambda: _thread_acquire(
                    results, failures, exact, tmp_path, client
                )
            )
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    assert failures == []
    assert len(results) == 8
    assert calls == 1
    assert len({result.path for result in results}) == 1


def _thread_acquire(results, failures, exact, cache_dir, client):
    try:
        results.append(
            distribution.acquire(exact, cache_dir=cache_dir, http_client=client)
        )
    except Exception as error:  # pragma: no cover - reported by assertion
        failures.append(error)


def test_explicit_fallback_records_only_same_product_failures(tmp_path):
    exact = _sp3_request(
        distribution.Distribution.nasa_cddis(),
        distribution.Distribution.direct(),
    )

    def handler(request):
        if request.url.host == "cddis.nasa.gov":
            return httpx.Response(404, request=request)
        return _gzip_response(request)

    with _client(handler) as client:
        result = distribution.acquire(exact, cache_dir=tmp_path, http_client=client)

    assert result.provenance.requested_identity == exact.identity
    assert result.provenance.resolved_identity.official_filename == (
        exact.identity.official_filename
    )
    assert (
        result.provenance.distribution_source is distribution.DistributionSource.DIRECT
    )
    assert len(result.provenance.attempts) == 1
    assert result.provenance.attempts[0].error_type == "product_not_published"


def test_all_explicit_sources_failed_preserves_each_public_reason(tmp_path):
    exact = _sp3_request(
        distribution.Distribution.nasa_cddis(),
        distribution.Distribution.direct(),
    )

    def handler(request):
        status = 404 if request.url.host == "cddis.nasa.gov" else 403
        return httpx.Response(status, request=request)

    with (
        _client(handler) as client,
        pytest.raises(distribution.AllDistributorsFailed) as caught,
    ):
        distribution.acquire(exact, cache_dir=tmp_path, http_client=client, retries=1)

    assert [failure.error_type for failure in caught.value.attempts] == [
        "product_not_published",
        "authorization_denied",
    ]
    assert caught.value.attempts[0].source is distribution.DistributionSource.NASA_CDDIS
    assert caught.value.attempts[1].source is distribution.DistributionSource.DIRECT


def test_distinct_product_identities_have_distinct_cache_entries(tmp_path):
    first = _sp3_request().identity
    second = distribution.identity(
        data.ops_ultra_sp3("cod_ult", SP3_DATE, issue="0000")
    )
    first_path = distribution._cache_path(
        tmp_path, first, distribution.DistributionSource.NASA_CDDIS
    )
    second_path = distribution._cache_path(
        tmp_path, second, distribution.DistributionSource.NASA_CDDIS
    )
    assert first_path != second_path
    assert first.key != second.key
    assert (
        hashlib.sha256(str(first_path).encode()).digest()
        != hashlib.sha256(str(second_path).encode()).digest()
    )
