"""Exact-distributor availability fallback parity regressions."""

import base64
import datetime as dt
from pathlib import Path

import httpx
import pytest
from sidereon import data, distribution

FIXTURE = Path(__file__).parent / "fixtures" / "gnss_data" / "igs22376.sp3.Z.b64"
PRODUCT_DATE = dt.date(2022, 11, 26)


def _body() -> bytes:
    archive = base64.b64decode(b"".join(FIXTURE.read_bytes().split()), validate=True)
    return distribution._decompress(archive, "unix_compress", 20_000)


def _request(*sources: distribution.Distribution) -> distribution.ProductRequest:
    return distribution.request(data.mgex_sp3("igs", PRODUCT_DATE), sources)


def test_retired_endpoint_falls_through_to_another_exact_distributor(tmp_path):
    request = _request(
        distribution.Distribution.nasa_cddis(),
        distribution.Distribution.in_memory(_body(), compression="none"),
    )

    def handler(http_request):
        return httpx.Response(410, request=http_request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        acquired = distribution.acquire(
            request, cache_dir=tmp_path, http_client=client, retries=1
        )

    assert (
        acquired.provenance.distribution_source
        is distribution.DistributionSource.IN_MEMORY
    )
    assert len(acquired.provenance.attempts) == 1
    assert acquired.provenance.attempts[0].error_type == "retired_endpoint"
    assert acquired.provenance.attempts[0].status == 410


def test_offline_cache_miss_falls_through_to_a_later_source_cache(tmp_path):
    in_memory = distribution.Distribution.in_memory(_body(), compression="none")
    distribution.acquire(_request(in_memory), cache_dir=tmp_path)

    acquired = distribution.acquire(
        _request(distribution.Distribution.nasa_cddis(), in_memory),
        cache_dir=tmp_path,
        offline=True,
    )

    assert (
        acquired.provenance.distribution_source
        is distribution.DistributionSource.IN_MEMORY
    )
    assert len(acquired.provenance.attempts) == 1
    assert acquired.provenance.attempts[0].error_type == "offline_cache_miss"


def test_exhausted_retryable_transport_falls_through(tmp_path):
    calls = 0
    request = _request(
        distribution.Distribution.nasa_cddis(),
        distribution.Distribution.in_memory(_body(), compression="none"),
    )

    def handler(http_request):
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out", request=http_request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        acquired = distribution.acquire(
            request,
            cache_dir=tmp_path,
            http_client=client,
            retries=2,
            backoff_s=0,
        )

    assert calls == 2
    assert (
        acquired.provenance.distribution_source
        is distribution.DistributionSource.IN_MEMORY
    )
    assert len(acquired.provenance.attempts) == 1
    assert acquired.provenance.attempts[0].error_type == "transport_failure"
    assert "timeout transport failure" in acquired.provenance.attempts[0].message


def test_caller_transport_failure_is_terminal(tmp_path):
    calls = 0
    request = _request(
        distribution.Distribution.nasa_cddis(),
        distribution.Distribution.in_memory(_body(), compression="none"),
    )

    def handler(http_request):
        nonlocal calls
        calls += 1
        raise httpx.LocalProtocolError("caller supplied an invalid header")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(distribution.TransportFailure) as caught:
            distribution.acquire(
                request,
                cache_dir=tmp_path,
                http_client=client,
                retries=3,
                backoff_s=0,
            )

    assert calls == 1
    assert caught.value.kind == "other"
    assert distribution._distributor_fallback_kind(caught.value) is None


@pytest.mark.parametrize("status", [600, 99999])
def test_non_http_server_status_is_not_retryable_or_fallback_eligible(status):
    error = distribution.TransportFailure(f"http_{status}", "https://example.test")
    error.status = status

    assert not distribution._retryable(error)
    assert distribution._distributor_fallback_kind(error) is None


@pytest.mark.parametrize("token", ["abc\rdef", "abc\ndef", "abc\r\ndef"])
def test_bearer_token_rejects_header_line_breaks(token):
    with pytest.raises(ValueError, match="must not contain CR or LF"):
        distribution.EarthdataAuth.bearer(token)


def test_first_availability_failure_is_preserved_when_later_source_is_absent(
    tmp_path,
):
    calls = 0
    request = _request(
        distribution.Distribution.nasa_cddis(),
        distribution.Distribution.nasa_cddis(),
    )

    def handler(http_request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("timed out", request=http_request)
        return httpx.Response(404, request=http_request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(distribution.TransportFailure) as caught:
            distribution.acquire(
                request,
                cache_dir=tmp_path,
                http_client=client,
                retries=1,
                backoff_s=0,
            )

    assert calls == 2
    assert caught.value.kind == "timeout"
