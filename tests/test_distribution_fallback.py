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


def test_unexpected_caller_transport_error_is_observed_without_changing_propagation(
    tmp_path,
):
    calls = 0
    diagnostics = []
    secret = "private token and message"
    request = _request(
        distribution.Distribution.nasa_cddis(),
        distribution.Distribution.in_memory(_body(), compression="none"),
    )

    def handler(http_request):
        nonlocal calls
        calls += 1
        raise ValueError(f"{secret}: {http_request.url}")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError) as caught:
            distribution.acquire(
                request,
                cache_dir=tmp_path,
                http_client=client,
                http_client_exception_diagnostics=diagnostics.append,
                retries=3,
                backoff_s=0,
            )

    assert calls == 1
    assert str(caught.value).startswith(secret)
    assert distribution._distributor_fallback_kind(caught.value) is None
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert set(diagnostic) == {
        "exception_class",
        "client_callsite",
        "stack_frames",
    }
    assert diagnostic["exception_class"] == "builtins.ValueError"
    callsite = diagnostic["client_callsite"]
    assert callsite == (__name__, "handler", 1)
    frames = diagnostic["stack_frames"]
    assert isinstance(frames, tuple) and 1 <= len(frames) <= 8
    assert all(isinstance(frame, tuple) and len(frame) == 3 for frame in frames)
    assert callsite in frames
    assert secret not in repr(diagnostic)
    assert "https://" not in repr(diagnostic)


def test_transport_diagnostic_failure_does_not_mask_transport_error(tmp_path):
    secret = "observer failure details"
    request = _request(distribution.Distribution.nasa_cddis())

    def handler(_http_request):
        raise ValueError("original private transport error")

    def observer(_diagnostic):
        raise RuntimeError(secret)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError) as caught:
            distribution.acquire(
                request,
                cache_dir=tmp_path,
                http_client=client,
                http_client_exception_diagnostics=observer,
                retries=3,
                backoff_s=0,
            )

    assert str(caught.value) == "original private transport error"
    assert secret not in str(caught.value)


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
