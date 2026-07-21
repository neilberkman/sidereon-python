"""Focused SP3 catalog, exact-validation, and fallback regressions."""

import datetime as dt
import gzip
import hashlib
import os

import httpx
import ncompress
import pytest
import sidereon
from _helpers import CORE_FIXTURES, sp3_bytes_for_date
from sidereon import data, distribution

SP3_DATE = dt.date(2026, 6, 25)


def _base_sp3() -> bytes:
    path = os.path.join(CORE_FIXTURES, "sp3", "COD0MGXFIN_20201770000_01D_05M_ORB.SP3")
    with open(path, "rb") as handle:
        return sp3_bytes_for_date(handle.read(), SP3_DATE)


def _with_epoch_count(content: bytes, count: int) -> bytes:
    lines = content.decode("ascii").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("*  ")]
    if not 0 < count <= len(starts):
        raise ValueError("test epoch count is outside the fixture")
    if count < len(starts):
        lines = lines[: starts[count]] + ["EOF"]
    lines[0] = lines[0][:32] + f"{count:7d}" + lines[0][39:]
    return ("\n".join(lines) + "\n").encode("ascii")


def _with_header_cadence(content: bytes, token: str) -> bytes:
    lines = content.decode("ascii").splitlines()
    assert len(token) == 14
    lines[1] = lines[1][:24] + token + lines[1][38:]
    return ("\n".join(lines) + "\n").encode("ascii")


def _with_irregular_grid(content: bytes) -> bytes:
    lines = content.decode("ascii").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("*  ")]
    line = lines[starts[1]]
    lines[starts[1]] = line[:17] + "  6" + line[20:]
    return ("\n".join(lines) + "\n").encode("ascii")


def _with_excess_epoch(content: bytes) -> bytes:
    lines = content.decode("ascii").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("*  ")]
    eof = lines.index("EOF")
    extra = list(lines[starts[-1] : eof])
    extra[0] = "*  2026  6 26  0  5  0.00000000"
    lines = lines[:eof] + extra + ["EOF"]
    lines[0] = lines[0][:32] + f"{290:7d}" + lines[0][39:]
    return ("\n".join(lines) + "\n").encode("ascii")


def _exact_request(*sources: distribution.Distribution) -> distribution.ProductRequest:
    product = data.mgex_sp3("cod", SP3_DATE)
    return distribution.request(
        product,
        sources or (distribution.Distribution.direct(),),
    )


def _gzip_response(request: httpx.Request, body: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        request=request,
        content=gzip.compress(body, mtime=0),
        headers={"content-type": "application/gzip"},
    )


def _gps_date(week: int, day: int = 0) -> dt.date:
    return dt.date(1980, 1, 6) + dt.timedelta(weeks=week, days=day)


def test_igs_final_catalog_covers_verified_naming_eras_and_padded_cddis_weeks():
    eras = [
        (730, "igs07300.sp3"),
        (999, "igs09990.sp3"),
        (2237, "igs22370.sp3"),
    ]
    for week, expected in eras:
        product = data.mgex_sp3("igs", _gps_date(week))
        assert product.canonical_filename() == expected
        exact = distribution.identity(product)
        assert exact.solution_class == "final"
        assert distribution.cddis_url(exact).endswith(f"/{week:04d}/{expected}.Z")
        assert distribution._distribution_location(
            exact,
            distribution.DistributionSource.NASA_CDDIS,
        )[1:] == (f"{expected}.Z", "unix_compress")
        with pytest.raises(data.UnsupportedProduct):
            product.archive_url()
        with pytest.raises(data.UnsupportedProduct):
            product.archive_compression()
        with pytest.raises(distribution.UnsupportedDistribution):
            distribution._distribution_location(
                exact,
                distribution.DistributionSource.DIRECT,
            )

    current = data.mgex_sp3("igs", _gps_date(2238))
    assert current.canonical_filename() == ("IGS0OPSFIN_20223310000_01D_15M_ORB.SP3")
    assert current.archive_url().endswith(
        "/2238/IGS0OPSFIN_20223310000_01D_15M_ORB.SP3.gz"
    )
    with pytest.raises(data.UnsupportedProduct):
        data.mgex_sp3("igs", _gps_date(729, 6))


def test_igs_broadcast_navigation_and_product_aware_solution_class_are_preserved():
    before_final_orbits = _gps_date(729, 6)
    nav = data.product("igs", "nav", before_final_orbits)
    assert nav.canonical_filename() == "BRDC00WRD_R_19940010000_01D_MN.rnx"
    assert nav.archive_url() == (
        "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/1994/001/"
        "BRDC00WRD_R_19940010000_01D_MN.rnx.gz"
    )
    assert data.product_solution_class("igs", "nav") == "broadcast"
    assert data.product_solution_class("igs", "sp3") == "final"
    with pytest.raises(data.UnsupportedProduct):
        data.product_solution_class("igs", "clk")


def test_date_aware_gfz_default_and_verified_code_family_routes():
    assert data.default_sample_for_date("gfz", "sp3", dt.date(2021, 5, 17)) == "15M"
    assert data.default_sample_for_date("gfz", "sp3", dt.date(2021, 5, 18)) == "05M"
    date = dt.date(2026, 4, 30)
    assert data.product("cod", "sp3", date).archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE_MGEX/CODE/2026/"
        "COD0MGXFIN_20261200000_01D_05M_ORB.SP3.gz"
    )
    assert data.product("cod", "clk", date).archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE_MGEX/CODE/2026/"
        "COD0MGXFIN_20261200000_01D_30S_CLK.CLK.gz"
    )
    assert data.product("cod", "ionex", date).archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE/2026/"
        "COD0OPSFIN_20261200000_01D_01H_GIM.INX.gz"
    )
    assert data.rapid_ionex(date).archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE/"
        "COD0OPSRAP_20261200000_01D_01H_GIM.INX.gz"
    )


def test_verified_sp3_family_floors_and_pretransition_cddis_guards():
    for center, day_before, first_day, expected_sample in [
        ("esa", dt.date(2014, 1, 4), dt.date(2014, 1, 5), "05M"),
        ("gfz", dt.date(2020, 5, 12), dt.date(2020, 5, 13), "15M"),
    ]:
        with pytest.raises(data.UnsupportedProduct):
            data.mgex_sp3(center, day_before)
        product = data.mgex_sp3(center, first_day)
        assert product.sample == expected_sample
        with pytest.raises(distribution.UnsupportedDistribution):
            distribution._distribution_location(
                distribution.identity(product),
                distribution.DistributionSource.NASA_CDDIS,
            )

    current_esa = data.mgex_sp3("esa", dt.date(2024, 6, 24))
    with pytest.raises(distribution.UnsupportedDistribution):
        distribution._distribution_location(
            distribution.identity(current_esa),
            distribution.DistributionSource.NASA_CDDIS,
        )

    for center, day_before, first_day, expected_url in [
        (
            "esa",
            dt.date(2014, 1, 4),
            dt.date(2014, 1, 5),
            "https://navigation-office.esa.int/products/gnss-products/1774/"
            "ESA0MGNFIN_20140050000_01D_30S_CLK.CLK.gz",
        ),
        (
            "gfz",
            dt.date(2020, 5, 12),
            dt.date(2020, 5, 13),
            "https://isdc-data.gfz.de/gnss/products/rapid/w2105/"
            "GFZ0OPSRAP_20201340000_01D_30S_CLK.CLK.gz",
        ),
    ]:
        with pytest.raises(data.UnsupportedProduct):
            data.product(center, "clk", day_before)
        assert data.product(center, "clk", first_day).archive_url() == expected_url

    with pytest.raises(data.UnsupportedProduct):
        data.ops_ultra_sp3("cod_ult", _gps_date(2237, 6), issue="0000")
    cod_ult = data.ops_ultra_sp3("cod_ult", _gps_date(2238), issue="0000")
    assert cod_ult.canonical_filename() == ("COD0OPSULT_20223310000_01D_05M_ORB.SP3")

    with pytest.raises(data.UnsupportedProduct):
        data.ops_ultra_sp3("igs_ult", _gps_date(2237, 6), issue="0000")
    igs_ult = data.ops_ultra_sp3("igs_ult", _gps_date(2238), issue="0000")
    assert igs_ult.sample == "15M"
    assert igs_ult.canonical_filename() == ("IGS0OPSULT_20223310000_02D_15M_ORB.SP3")

    for center, day_before, first_day in [
        ("esa_ult", dt.date(2022, 10, 3), dt.date(2022, 10, 4)),
        ("gfz_ult", dt.date(2020, 10, 5), dt.date(2020, 10, 6)),
    ]:
        with pytest.raises(data.UnsupportedProduct):
            data.ops_ultra_sp3(center, day_before, issue="0000")
        product = data.ops_ultra_sp3(center, first_day, issue="0000")
        assert product.sample == "15M"
        with pytest.raises(distribution.UnsupportedDistribution):
            distribution._distribution_location(
                distribution.identity(product),
                distribution.DistributionSource.NASA_CDDIS,
            )


def test_ultra_rapid_defaults_and_candidates_follow_issue_aware_cadence_eras():
    assert data.default_sample_for_date("esa_ult", "sp3", dt.date(2024, 9, 3)) == "15M"
    assert data.default_sample_for_date("esa_ult", "sp3", dt.date(2025, 2, 2)) == "15M"

    esa_0600 = data.ops_ultra_sp3("esa_ult", dt.date(2025, 2, 2), issue="0600")
    esa_1200 = data.ops_ultra_sp3("esa_ult", dt.date(2025, 2, 2), issue="1200")
    assert esa_0600.sample == "15M"
    assert esa_1200.sample == "05M"
    assert esa_0600.canonical_filename().endswith("_02D_15M_ORB.SP3")
    assert esa_1200.canonical_filename().endswith("_02D_05M_ORB.SP3")
    assert [
        product.sample
        for product in data._sp3_products_for_issue(
            "esa_ult", dt.date(2025, 2, 2), "0600", None
        )[:2]
    ] == ["15M", "05M"]
    assert [
        product.sample
        for product in data._sp3_products_for_issue(
            "esa_ult", dt.date(2025, 2, 2), "1200", None
        )[:2]
    ] == ["05M", "15M"]

    assert data.default_sample_for_date("gfz_ult", "sp3", dt.date(2021, 5, 15)) == "15M"
    assert data.default_sample_for_date("gfz_ult", "sp3", dt.date(2021, 5, 16)) == "05M"
    assert (
        data.ops_ultra_sp3("gfz_ult", dt.date(2021, 5, 15), issue="2100").sample
        == "15M"
    )
    assert (
        data.ops_ultra_sp3("gfz_ult", dt.date(2021, 5, 16), issue="0000").sample
        == "05M"
    )


def test_generic_product_default_uses_the_exact_ultra_issue_cadence():
    transition = dt.date(2025, 2, 2)
    for issue, expected_sample in [("0600", "15M"), ("1200", "05M")]:
        product = data.product("esa_ult", "sp3", transition, issue=issue)
        assert product.sample == expected_sample
        assert product.canonical_filename() == (
            f"ESA0OPSULT_2025033{issue}_02D_{expected_sample}_ORB.SP3"
        )
        assert (
            data.canonical_filename("esa_ult", "sp3", transition, issue=issue)
            == product.canonical_filename()
        )


def test_exact_sp3_accepts_half_open_and_inclusive_regular_grids():
    request = sidereon.ExactSp3Request(SP3_DATE, "01D", "05M", expected_agency="AIUB")
    inclusive_sp3, inclusive = sidereon.parse_exact_sp3(_base_sp3(), request)
    half_open_sp3, half_open = sidereon.parse_exact_sp3(
        _with_epoch_count(_base_sp3(), 288), request
    )
    assert inclusive is sidereon.ExactSp3Coverage.INCLUSIVE
    assert half_open is sidereon.ExactSp3Coverage.HALF_OPEN
    assert inclusive_sp3.epoch_count == inclusive_sp3.declared_epoch_count == 289
    assert half_open_sp3.epoch_count == half_open_sp3.declared_epoch_count == 288
    assert inclusive_sp3.declared_start_j2000_s is not None
    assert sidereon.validate_exact_sp3(inclusive_sp3, request) is inclusive


@pytest.mark.parametrize(
    "content, message",
    [
        (_with_epoch_count(_base_sp3(), 287), "span mismatch"),
        (_with_excess_epoch(_base_sp3()), "span mismatch"),
        (_with_irregular_grid(_base_sp3()), "epoch grid is irregular"),
        (_with_header_cadence(_base_sp3(), "    0.00000000"), "must be positive"),
        (_with_header_cadence(_base_sp3(), "           NaN"), "is not finite"),
    ],
    ids=["287", "excess", "irregular", "zero-cadence", "nan-cadence"],
)
def test_exact_sp3_rejects_bad_span_grid_and_header_cadence(content, message):
    request = sidereon.ExactSp3Request(SP3_DATE, "01D", "05M")
    with pytest.raises(sidereon.ExactSp3ValidationError, match=message):
        sidereon.parse_exact_sp3(content, request)


@pytest.mark.parametrize("sample", ["00S", "05X", "NaN"])
def test_exact_sp3_request_rejects_zero_and_unknown_sample_tokens(sample):
    with pytest.raises(sidereon.ExactSp3ValidationError):
        sidereon.ExactSp3Request(SP3_DATE, "01D", sample)


def test_exact_sp3_request_rejects_non_sp3_identity():
    ionex = distribution.identity(data.mgex_ionex("cod", SP3_DATE))
    with pytest.raises(sidereon.ExactSp3ValidationError):
        sidereon.ExactSp3Request.from_identity(ionex)


def test_historical_cddis_unix_compress_round_trip_and_bounded_failure():
    content = b"historical IGS product bytes" * 50
    archive = ncompress.compress(content)
    assert distribution._detect_compression(archive, "auto") == "unix_compress"
    assert distribution._decompress(archive, "unix_compress", len(content)) == content
    with pytest.raises(data.DownloadSizeExceeded):
        distribution._decompress(archive, "unix_compress", len(content) - 1)
    with pytest.raises(distribution.DecompressionFailure):
        distribution._decompress(b"\x1f\x9dcorrupt", "unix_compress", 1_000)


def test_unix_compress_output_limit_stops_further_archive_input(monkeypatch):
    archive = ncompress.compress(b"A")

    def fake_decompress(source, output):
        assert source.read(1) == archive[:1]
        output.write(b"overflow")
        assert source.read(1) == b""

    monkeypatch.setattr(distribution.ncompress, "decompress", fake_decompress)
    with pytest.raises(data.DownloadSizeExceeded):
        distribution._decompress(archive, "unix_compress", 1)


def test_publication_absence_falls_back_to_the_same_exact_identity(tmp_path):
    calls = []
    request = _exact_request(
        distribution.Distribution.nasa_cddis(), distribution.Distribution.direct()
    )

    def handler(http_request):
        calls.append(http_request.url.host)
        if http_request.url.host == "cddis.nasa.gov":
            return httpx.Response(404, request=http_request)
        return _gzip_response(http_request, _base_sp3())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        acquired = distribution.acquire(request, cache_dir=tmp_path, http_client=client)
    assert len(calls) == 2
    assert acquired.provenance.attempts[0].error_type == "product_not_published"
    assert acquired.provenance.resolved_identity.format_version == "SP3-d"


@pytest.mark.parametrize(
    "bad_content",
    [
        b"not an SP3 product",
        _with_epoch_count(_base_sp3(), 287),
        _with_header_cadence(_base_sp3(), "  900.00000000"),
        _with_irregular_grid(_base_sp3()),
        _base_sp3()[:-4],
    ],
)
def test_integrity_failure_is_terminal_even_when_later_source_is_valid(
    tmp_path, bad_content
):
    calls = []
    request = _exact_request(
        distribution.Distribution.direct(), distribution.Distribution.nasa_cddis()
    )

    def handler(http_request):
        calls.append(http_request.url.host)
        body = bad_content if len(calls) == 1 else _base_sp3()
        return _gzip_response(http_request, body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(distribution.ProductValidationFailure):
            distribution.acquire(request, cache_dir=tmp_path, http_client=client)
    assert len(calls) == 1


def test_digest_failure_is_terminal_even_when_later_source_matches(tmp_path):
    original = _base_sp3()
    changed = original.replace(b"/* CODE MGEX", b"/* C0DE MGEX", 1)
    calls = []
    request = _exact_request(
        distribution.Distribution.direct(), distribution.Distribution.nasa_cddis()
    )

    def handler(http_request):
        calls.append(http_request.url.host)
        return _gzip_response(http_request, changed if len(calls) == 1 else original)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(distribution.ProductValidationFailure) as caught:
            distribution.acquire(
                request,
                cache_dir=tmp_path,
                http_client=client,
                sha256=hashlib.sha256(original).hexdigest(),
            )
    assert len(calls) == 1
    assert caught.value.code == "checksum_mismatch"


def test_absence_followed_by_integrity_failure_preserves_integrity(tmp_path):
    calls = []
    request = _exact_request(
        distribution.Distribution.nasa_cddis(), distribution.Distribution.direct()
    )

    def handler(http_request):
        calls.append(http_request.url.host)
        if len(calls) == 1:
            return httpx.Response(404, request=http_request)
        return _gzip_response(http_request, b"malformed SP3")

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(distribution.ProductValidationFailure):
            distribution.acquire(request, cache_dir=tmp_path, http_client=client)
    assert len(calls) == 2


def test_candidate_absence_then_integrity_failure_is_terminal(monkeypatch):
    calls = []

    def fake_acquire(product, **_kwargs):
        calls.append(product.pattern)
        if len(calls) == 1:
            raise distribution.ProductNotPublished(
                404,
                product.archive_url(),
                "not published",
            )
        if len(calls) == 2:
            raise distribution.ProductValidationFailure("malformed candidate")
        raise AssertionError("an integrity failure must stop candidate fallback")

    monkeypatch.setattr(distribution, "_acquire_catalog_product", fake_acquire)
    with pytest.raises(
        distribution.ProductValidationFailure,
        match="malformed candidate",
    ):
        data._fetch_center_sp3("esa_ult", dt.date(2026, 7, 13), None, {})
    assert calls == ["primary_02D_05M", "alternate_02D_15M"]


def test_known_unsupported_center_product_fails_before_http(tmp_path):
    calls = 0

    def handler(http_request):
        nonlocal calls
        calls += 1
        return httpx.Response(404, request=http_request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(data.UnsupportedProduct):
            data.fetch_merged_sp3(
                SP3_DATE,
                ["cod_prd1"],
                cache_dir=str(tmp_path),
                http_client=client,
            )
    assert calls == 0


def test_mixed_center_list_prevalidates_sp3_capability_before_http(tmp_path):
    calls = 0

    def handler(http_request):
        nonlocal calls
        calls += 1
        return httpx.Response(404, request=http_request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(
            data.UnsupportedProduct,
            match="cod_prd1 does not serve sp3",
        ):
            data.fetch_merged_sp3(
                SP3_DATE,
                ["cod", "cod_prd1"],
                cache_dir=str(tmp_path),
                http_client=client,
            )
    assert calls == 0
