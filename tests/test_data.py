"""Tests for the optional GNSS data-provisioning layer (``sidereon.data``).

The offline tests always run with no network: they seed a temporary cache_dir
with real core fixtures under their canonical IGS long-names, then exercise the
cache-first fetch, parse, and merge paths plus the typed-error taxonomy. The
URL/filename builders are unit-tested against the exact core-derived strings.

The network tests (``@pytest.mark.network``) hit a live archive and are excluded
by default; run them with ``pytest -m network``.
"""

import copy
import datetime as dt
import functools
import gzip
import hashlib
import json
import math
import os
import shutil
from dataclasses import replace

import httpx
import numpy as np
import pytest
import sidereon
import sidereon.data as data
import sidereon.distribution as distribution
from _helpers import CORE_FIXTURES, FIXTURES, sp3_bytes_for_date


def test_unusable_sample_report_requires_reason_and_epoch_to_agree():
    placed = {
        "kind": "unusable_sample",
        "satellite": "G01",
        "from_j2000_s": 1.0,
        "to_j2000_s": 1.0,
        "magnitude": None,
        "bound": None,
        "epoch_j2000_s": 1.0,
        "sample_index": 0,
        "reason": "non_finite_position",
    }
    unplaced = {
        **placed,
        "from_j2000_s": None,
        "to_j2000_s": None,
        "epoch_j2000_s": None,
        "reason": "epoch_not_placed",
    }

    assert data._validate_continuity_defect(placed, "placed defect")
    assert data._validate_continuity_defect(unplaced, "unplaced defect")

    for invalid in (
        {**placed, "reason": "epoch_not_placed"},
        {
            **unplaced,
            "reason": "non_finite_position",
        },
    ):
        with pytest.raises(ValueError, match="reason disagrees with epoch"):
            data._validate_continuity_defect(invalid, "invalid defect")


# --- fixtures ------------------------------------------------------------


def _core_sp3(name):
    return os.path.join(CORE_FIXTURES, "sp3", name)


def _core_ionex(name):
    return os.path.join(CORE_FIXTURES, "ionex", name)


def _ionex_bytes_for(date, interval_hours=1):
    with open(_core_ionex("synthetic_2map_7x7.20i"), "rb") as handle:
        lines = handle.read().decode("ascii").splitlines()
    map_index = 0
    for index, line in enumerate(lines):
        if line.rstrip().endswith("EPOCH OF CURRENT MAP"):
            lines[index] = (
                f"{date.year:6d}{date.month:6d}{date.day:6d}"
                f"{map_index * interval_hours:6d}{0:6d}{0:6d}{line[36:]}"
            )
            map_index += 1
    return ("\n".join(lines) + "\n").encode("ascii")


def _seed_exact_direct_ionex(cache_dir, product):
    exact = distribution.request(product, [distribution.Distribution.direct()])

    def handler(request):
        return httpx.Response(
            200,
            request=request,
            content=gzip.compress(_ionex_bytes_for(product.date), mtime=0),
            headers={"content-type": "application/gzip"},
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return distribution.acquire(exact, cache_dir=cache_dir, http_client=client)


def _sp3_payload():
    with open(_core_sp3("COD0MGXFIN_20201770000_01D_05M_ORB.SP3"), "rb") as handle:
        return handle.read()


def _archive_for_catalog_product(product, content):
    identity = distribution._catalog_product_identity(product)
    _, compression = distribution._catalog_direct_location(product, identity)
    agency = {
        "cod": "AIUB",
        "cod_ult": "AIUB",
        "esa": "ESOC",
        "esa_ult": "ESOC",
        "gfz": "GFZ",
        "gfz_ult": "GFZ",
        "igs": "IGS",
        "igs_ult": "IGS",
    }[product.center]
    content = sp3_bytes_for_date(content, product.date, agency)
    return gzip.compress(content, mtime=0) if compression == "gzip" else content


def _seed_exact_direct_sp3(cache_dir, product, content=None):
    payload = _sp3_payload() if content is None else content
    archive = _archive_for_catalog_product(product, payload)

    def handler(request):
        return httpx.Response(200, request=request, content=archive)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return distribution._acquire_catalog_product(
            product, cache_dir=cache_dir, http_client=client
        )


def _provenance_for_catalog_product(product):
    requested = distribution._catalog_product_identity(product)
    resolved = replace(requested, format_version="SP3-d")
    return distribution.AcquisitionProvenance(
        requested_identity=requested,
        resolved_identity=resolved,
        publisher=requested.publisher,
        distribution_source=distribution.DistributionSource.DIRECT,
        official_filename=requested.official_filename,
        original_url=product.archive_url(),
        final_url=product.archive_url(),
        retrieved_at="2026-07-16T00:00:00+00:00",
        byte_length=1,
        sha256="11" * 32,
        etag=None,
        last_modified=None,
        cache_hit=False,
        archive_compression=product.archive_compression(),
        archive_byte_length=1,
        archive_sha256="22" * 32,
    )


def _core_dted_tile():
    return os.path.join(CORE_FIXTURES, "dted", "tiles", "n36_w107_1arc_v3.dt2")


def _seed(cache_dir, product, source_path):
    """Seed a real fixture into the cache as a verified hit.

    A genuine cache entry carries a provenance sidecar recording the decompressed
    sha256; offline reads require it, so the seed writes one (mirroring the
    committer) rather than relying on an untrusted bare file.
    """
    dest = os.path.join(cache_dir, product.canonical_filename())
    shutil.copyfile(source_path, dest)
    with open(dest, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    with open(dest + ".provenance.json", "w") as handle:
        json.dump({"sha256_decompressed": digest}, handle)
    return dest


def _seed_bare(cache_dir, product, source_path):
    """Seed a fixture with NO provenance sidecar (an untrusted hand-placed file)."""
    dest = os.path.join(cache_dir, product.canonical_filename())
    shutil.copyfile(source_path, dest)
    return dest


def _terrain_path(cache_dir, lat_index=36, lon_index=-107):
    return os.path.join(cache_dir, data.dted_cache_relpath(lat_index, lon_index))


def _seed_terrain(cache_dir, source_path=None, lat_index=36, lon_index=-107):
    dest = _terrain_path(cache_dir, lat_index, lon_index)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(source_path or _core_dted_tile(), dest)
    with open(dest, "rb") as handle:
        payload = handle.read()
    digest = hashlib.sha256(payload).hexdigest()
    with open(dest + ".provenance.json", "w") as handle:
        json.dump(
            {
                "sha256_data": digest,
                "tile_id": data.skadi_tile_id(lat_index, lon_index),
            },
            handle,
        )
    return dest


def _seed_terrain_bare(cache_dir, lat_index=36, lon_index=-107):
    dest = _terrain_path(cache_dir, lat_index, lon_index)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(_core_dted_tile(), dest)
    return dest


class _StubResponse:
    def __init__(self, status_code, body=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}
        self.closed = False

    def iter_bytes(self, chunk_size=None):
        yield self._body

    def close(self):
        self.closed = True


class _StubStream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self.response

    def __exit__(self, exc_type, exc, traceback):
        return False


def _stub_http(monkeypatch, responses):
    calls = []
    queue = list(responses)

    def stream(method, url, follow_redirects, timeout):
        calls.append(
            {
                "method": method,
                "url": url,
                "follow_redirects": follow_redirects,
                "timeout": timeout,
            }
        )
        if not queue:
            raise AssertionError("unexpected HTTP request")
        response = queue.pop(0)
        status, body = response[:2]
        headers = response[2] if len(response) == 3 else None
        return _StubStream(_StubResponse(status, body, headers))

    monkeypatch.setattr(data.httpx, "stream", stream)
    return calls


POSTINGS = 3601
HGT_LEN = POSTINGS * POSTINGS * 2
DTED_LEN = 25_981_042
# The synthetic tile's one SRTM void is written as the DTED null (0xFFFF), as
# MIL-PRF-89020B 3.10.9.1 requires; written as 0, the digest was
# 708c193f768f3d859b71da20a81059db4e6077494481c4484d0bc238af096d77.
SYNTHETIC_DTED_SHA256 = (
    "953bbb5ae33c646910a1498276cb7a3a9dc000bd2a8a58e58d5bf320a71a8a89"
)

SPACE_WEATHER_CSV = (
    "DATE,BSRN,ND,KP1,KP2,KP3,KP4,KP5,KP6,KP7,KP8,KP_SUM,AP1,AP2,AP3,AP4,"
    "AP5,AP6,AP7,AP8,AP_AVG,CP,C9,ISN,F10.7_OBS,F10.7_ADJ,F10.7_DATA_TYPE,"
    "F10.7_OBS_CENTER81,F10.7_OBS_LAST81,F10.7_ADJ_CENTER81,F10.7_ADJ_LAST81\n"
    "2024-05-09,2556,1,23,27,30,33,40,50,47,37,287,9,12,15,18,27,48,39,22,"
    "24,1.2,5,120,165.1,162.0,OBS,150.1,149.8,147.0,146.6\n"
    "2024-05-10,2556,2,40,50,60,70,67,57,47,37,428,27,48,80,132,111,67,39,"
    "22,66,1.8,7,121,190.2,187.1,OBS,151.2,150.9,148.0,147.6\n"
    "2024-05-11,2556,3,33,30,27,23,20,17,13,10,173,18,15,12,9,7,6,5,4,10,"
    "0.8,3,119,176.3,173.0,OBS,152.3,151.1,149.0,148.2\n"
    "2024-06-01,2557,24,,,,,,,,,,,,,,,,,,,,,118,171.0,168.0,PRM,153.0,152.0,"
    "150.0,149.0\n"
).encode()


def _synthetic_hgt_sample(row, col):
    if (row, col) == (2366, 2345):
        return -32768
    if (row, col) == (3500, 200):
        return -415
    if (row, col) == (1600, 3000):
        return -1
    if (row, col) == (0, 3600):
        return 8848
    return ((row * 37 + col * 19) % 5000) - 1000


def _expected_posting(lat_posting, lon_posting):
    """The height a converted posting reads back as, or None for the SRTM void,
    which reads back as an unknown elevation."""
    sample = _synthetic_hgt_sample(POSTINGS - 1 - lat_posting, lon_posting)
    return None if sample == -32768 else sample


@functools.lru_cache(maxsize=1)
def _synthetic_hgt():
    rows = np.arange(POSTINGS, dtype=np.int32)[:, None]
    cols = np.arange(POSTINGS, dtype=np.int32)[None, :]
    samples = ((rows * 37 + cols * 19) % 5000 - 1000).astype(">i2")
    samples[2366, 2345] = -32768
    samples[3500, 200] = -415
    samples[1600, 3000] = -1
    samples[0, 3600] = 8848
    payload = samples.tobytes()
    assert len(payload) == HGT_LEN
    return payload


@functools.lru_cache(maxsize=1)
def _synthetic_hgt_gz():
    return gzip.compress(_synthetic_hgt(), mtime=0)


def _seed_space_weather(cache_dir, product="sw_all", fetched_at=None, payload=None):
    relpath = data.space_weather_cache_relpath(product)
    dest = os.path.join(cache_dir, relpath)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    body = payload or SPACE_WEATHER_CSV
    with open(dest, "wb") as handle:
        handle.write(body)
    digest = hashlib.sha256(body).hexdigest()
    with open(dest + ".provenance.json", "w") as handle:
        json.dump(
            {
                "source_url": data.space_weather_archive_url(product),
                "sha256_data": digest,
                "fetched_at": (
                    fetched_at or dt.datetime.now(dt.timezone.utc).isoformat()
                ),
                "fetcher": "sidereon.data",
            },
            handle,
        )
    return dest


# An IONEX day whose canonical predicted name we seed; the offset is applied by
# predicted_ionex, so we build the product and seed at its resolved date.
IONEX_DATE = dt.date(2026, 6, 14)
# The real CODE final SP3 fixture is for doy 177 of 2020 (2020-06-25).
SP3_DATE = dt.date(2026, 6, 25)


# --- builder / URL unit tests (no network) -------------------------------


def test_predicted_ionex_filenames_match_reference():
    p1 = data.predicted_ionex("cod_prd1", IONEX_DATE)
    p2 = data.predicted_ionex("cod_prd2", IONEX_DATE)
    assert p1.canonical_filename() == "COD0OPSP0D_20261650000_01D_01H_GIM.INX"
    assert p2.canonical_filename() == "COD0OPSP1D_20261660000_01D_01H_GIM.INX"


def test_predicted_ionex_urls_use_the_aiub_prd_archive():
    p1 = data.predicted_ionex("cod_prd1", IONEX_DATE)
    assert p1.archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE/IONO/PRD/"
        "COD0OPSP0D_20261650000_01D_01H_GIM.INX.gz"
    )
    p2 = data.predicted_ionex("cod_prd2", IONEX_DATE)
    assert p2.archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE/IONO/PRD/"
        "COD0OPSP1D_20261660000_01D_01H_GIM.INX.gz"
    )
    boundary = data.predicted_ionex("cod_prd2", dt.date(2026, 12, 31))
    assert boundary.date == dt.date(2027, 1, 1)
    assert boundary.archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE/IONO/PRD/"
        "COD0OPSP1D_20270010000_01D_01H_GIM.INX.gz"
    )


def test_rapid_ionex_filename_and_url_match_reference():
    p = data.rapid_ionex(dt.date(2026, 6, 13))
    assert p.canonical_filename() == "COD0OPSRAP_20261640000_01D_01H_GIM.INX"
    assert p.archive_url() == (
        "https://www.aiub.unibe.ch/download/CODE/"
        "COD0OPSRAP_20261640000_01D_01H_GIM.INX.gz"
    )


def test_sp3_center_filenames_match_reference():
    # GFZ rapid SP3
    assert (
        data.canonical_filename("gfz", "sp3", dt.date(2020, 6, 24), "15M")
        == "GFZ0OPSRAP_20201760000_01D_15M_ORB.SP3"
    )
    # IGS ultra-rapid SP3 with a sub-daily issue
    assert (
        data.product(
            "igs_ult", "sp3", dt.date(2024, 9, 3), "15M", issue="0600"
        ).canonical_filename()
        == "IGS0OPSULT_20242470600_02D_15M_ORB.SP3"
    )
    # CODE final MGEX SP3
    assert (
        data.mgex_sp3("cod", SP3_DATE).canonical_filename()
        == "COD0MGXFIN_20261760000_01D_05M_ORB.SP3"
    )


def test_sp3_center_urls_match_reference():
    assert data.archive_url("gfz", "sp3", dt.date(2020, 6, 24), "15M") == (
        "https://isdc-data.gfz.de/gnss/products/rapid/w2111/"
        "GFZ0OPSRAP_20201760000_01D_15M_ORB.SP3.gz"
    )
    # CODE ultra-rapid SP3 is served uncompressed on AIUB's HTTPS /CODE root.
    assert data.archive_url(
        "cod_ult", "sp3", dt.date(2026, 7, 14), "05M", issue="0000"
    ) == (
        "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT_20261950000_01D_05M_ORB.SP3"
    )
    assert data.archive_url(
        "igs_ult", "sp3", dt.date(2024, 9, 3), "15M", issue="0600"
    ) == (
        "https://igs.bkg.bund.de/root_ftp/IGS/products/2330/"
        "IGS0OPSULT_20242470600_02D_15M_ORB.SP3.gz"
    )


def test_gps_week_and_doy():
    assert data.gps_week(dt.date(2020, 6, 24)) == 2111
    assert data.day_of_year(dt.date(2020, 6, 24)) == 176


def test_space_weather_catalog_paths_match_reference():
    source = data.space_weather_source_entry()
    assert source.protocol == "https"
    assert source.host == "celestrak.org"
    assert source.compression == "none"
    assert data.space_weather_filename() == "SW-All.csv"
    assert data.space_weather_filename("sw_last5") == "SW-Last5Years.csv"
    assert data.space_weather_cache_relpath() == "space-weather/SW-All.csv"
    assert data.space_weather_archive_url("sw_last5") == (
        "https://celestrak.org/SpaceData/SW-Last5Years.csv"
    )


def test_fetch_space_weather_fresh_cache_returns_loaded_table(tmp_path, monkeypatch):
    _seed_space_weather(tmp_path)
    calls = _stub_http(monkeypatch, [])

    table = data.fetch_space_weather(cache_dir=str(tmp_path), offline=False)

    assert calls == []
    sample = table.sample_at(sidereon.j2000_seconds(2024, 5, 10, 12, 0, 0.0))
    assert sample.space_weather.f107 == 165.1
    assert sample.space_weather.ap == 66.0


def test_fetch_space_weather_offline_returns_stale_cache(tmp_path, monkeypatch):
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)).isoformat()
    _seed_space_weather(tmp_path, fetched_at=old)
    calls = _stub_http(monkeypatch, [])

    table = data.fetch_space_weather(
        cache_dir=str(tmp_path),
        offline=True,
        max_age_s=1.0,
    )

    assert calls == []
    assert table.coverage().first_j2000_s < table.coverage().end_j2000_s


def test_fetch_space_weather_expired_cache_refetches(tmp_path, monkeypatch):
    old = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)).isoformat()
    path = _seed_space_weather(tmp_path, fetched_at=old)
    calls = _stub_http(monkeypatch, [(200, SPACE_WEATHER_CSV)])

    table = data.fetch_space_weather(cache_dir=str(tmp_path), max_age_s=1.0)

    assert len(calls) == 1
    assert calls[0]["url"] == data.space_weather_archive_url("sw_all")
    weather = table.space_weather_at(sidereon.j2000_seconds(2024, 5, 10, 0, 0, 0.0))
    assert weather.ap == 66.0
    with open(path + ".provenance.json") as handle:
        provenance = json.load(handle)
    assert provenance["source_url"] == data.space_weather_archive_url("sw_all")
    assert provenance["fetcher"] == "sidereon.data"


def test_fetch_space_weather_sha256_mismatch_is_terminal(tmp_path, monkeypatch):
    _seed_space_weather(tmp_path)
    calls = _stub_http(monkeypatch, [(200, SPACE_WEATHER_CSV)])

    with pytest.raises(data.ChecksumMismatch):
        data.fetch_space_weather(cache_dir=str(tmp_path), sha256="0" * 64)

    assert calls == []


def test_unknown_center_raises():
    with pytest.raises(data.UnknownCenter):
        data.product("nope", "ionex", IONEX_DATE)
    with pytest.raises(data.UnknownCenter):
        data.predicted_ionex("cod_rap", IONEX_DATE)


def test_unknown_center_in_fetch_ionex_raises(tmp_path):
    with pytest.raises(data.UnknownCenter):
        data.fetch_ionex("nope", IONEX_DATE, offline=True, cache_dir=str(tmp_path))


# --- offline fetch_ionex -------------------------------------------------


def test_fetch_ionex_offline_reads_cache_and_parses(tmp_path):
    cache = str(tmp_path)
    prod = data.predicted_ionex("cod_prd1", IONEX_DATE)
    _seed_exact_direct_ionex(cache, prod)

    ionex = data.fetch_ionex("cod_prd1", IONEX_DATE, offline=True, cache_dir=cache)
    assert isinstance(ionex, sidereon.Ionex)


def test_fetch_ionex_offline_walks_back_to_older_cached_day(tmp_path):
    cache = str(tmp_path)
    # Only the day-before is cached; the newest candidate is absent. The
    # newest-first walk must fall back to it.
    older = data.predicted_ionex("cod_prd1", IONEX_DATE - dt.timedelta(days=1))
    _seed_exact_direct_ionex(cache, older)

    ionex = data.fetch_ionex("cod_prd1", IONEX_DATE, offline=True, cache_dir=cache)
    assert isinstance(ionex, sidereon.Ionex)


def test_fetch_ionex_offline_empty_cache_raises_offline_miss(tmp_path):
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch_ionex("cod_prd1", IONEX_DATE, offline=True, cache_dir=str(tmp_path))


def test_fetch_ionex_uses_exact_semantic_validation(tmp_path):
    product = data.predicted_ionex("cod_prd1", IONEX_DATE)
    wrong = _ionex_bytes_for(product.date + dt.timedelta(days=1))

    def handler(request):
        return httpx.Response(
            200,
            request=request,
            content=gzip.compress(wrong, mtime=0),
            headers={"content-type": "application/gzip"},
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(distribution.ProductValidationFailure),
    ):
        data.fetch_ionex(
            "cod_prd1",
            IONEX_DATE,
            lookback=2,
            cache_dir=str(tmp_path),
            http_client=client,
        )


# --- offline merged SP3 --------------------------------------------------


def test_fetch_merged_sp3_offline_single_contributor(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(cache, prod)

    sp3, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], offline=True, cache_dir=cache
    )
    assert isinstance(sp3, sidereon.Sp3)
    assert report.single_product is True
    assert report.source_count == 1
    assert report.merged is True
    assert report.merge_report is not None
    assert [c.center for c in report.contributors] == ["cod"]


def test_fetch_merged_sp3_forwards_complete_merge_options(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(cache, prod)
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        outlier_reject=sidereon.Sp3OutlierRejectOptions(0.5, 5.0e-9),
        min_agree=1,
    )

    merged, report = data.fetch_merged_sp3(
        SP3_DATE,
        ["cod"],
        offline=True,
        cache_dir=cache,
        merge_options=options,
    )

    assert merged.epoch_count > 0
    assert report.merge_report is not None
    assert report.merge_policy["combine"] == "precedence"
    assert report.merge_policy["precedence_artifact_sha256"] == [
        report.contributors[0].artifact_identity.product_sha256
    ]


def test_merged_sp3_exposes_exact_two_contributor_provenance(tmp_path):
    cache = str(tmp_path)
    products = [
        data.mgex_sp3("cod", SP3_DATE),
        data.mgex_sp3("esa", SP3_DATE),
    ]
    payload = _sp3_payload()
    archives = {
        product.archive_url(): _archive_for_catalog_product(product, payload)
        for product in products
    }

    def handler(request):
        return httpx.Response(200, request=request, content=archives[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, report = data.fetch_merged_sp3(
            SP3_DATE,
            ["cod", "esa"],
            cache_dir=cache,
            http_client=client,
        )

    assert report.source_count == 2
    assert report.input_identity_schema_version == 1
    assert report.stable_input_identity.startswith("sidereon-sp3-merge-input-v1:")
    artifacts = [item.artifact_identity for item in report.contributors]
    assert all(artifact is not None for artifact in artifacts)
    assert [artifact.requested_identity.analysis_center for artifact in artifacts] == [
        "cod",
        "esa",
    ]
    assert all(
        artifact.resolved_identity.format_version == "SP3-d" for artifact in artifacts
    )
    assert all(len(artifact.product_sha256) == 64 for artifact in artifacts)
    assert all(len(artifact.archive_sha256) == 64 for artifact in artifacts)
    assert all(item.acquisition_facts is not None for item in report.contributors)
    assert report.merge_policy["combine"] == "mean"
    assert report.merge_policy["precedence_artifact_sha256"] == []
    persisted = report.to_dict()["contributors"][0]["artifact_identity"]
    assert data.ArtifactIdentity.from_dict(persisted) == artifacts[0]

    _, precedence_report = data.fetch_merged_sp3(
        SP3_DATE,
        ["cod", "esa"],
        cache_dir=cache,
        offline=True,
        merge_options=sidereon.Sp3MergeOptions(combine="precedence"),
    )
    precedence_record = precedence_report.to_dict()
    assert data.verify_merge_report(precedence_record)
    precedence_record["contributors"].reverse()
    assert not data.verify_merge_report(precedence_record)


def test_merged_sp3_cache_hit_does_not_change_stable_identity(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    archive = _archive_for_catalog_product(product, _sp3_payload())

    def handler(request):
        return httpx.Response(200, request=request, content=archive)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, miss = data.fetch_merged_sp3(
            SP3_DATE, ["cod"], cache_dir=str(tmp_path), http_client=client
        )
    _, hit = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], cache_dir=str(tmp_path), offline=True
    )

    assert miss.contributors[0].acquisition_facts.cache_hit is False
    assert hit.contributors[0].acquisition_facts.cache_hit is True
    assert miss.stable_input_identity == hit.stable_input_identity


def test_sp3_merge_input_identity_binds_artifact_set_order_and_policy(tmp_path):
    first = data.mgex_sp3("cod", SP3_DATE)
    second = data.mgex_sp3("esa", SP3_DATE)
    first_acquired = _seed_exact_direct_sp3(str(tmp_path / "first"), first)
    second_acquired = _seed_exact_direct_sp3(str(tmp_path / "second"), second)
    first_artifact = data.ArtifactIdentity._from_provenance(first_acquired.provenance)
    second_artifact = data.ArtifactIdentity._from_provenance(second_acquired.provenance)

    original = data.sp3_merge_input_identity([first_artifact, second_artifact])
    reversed_order = data.sp3_merge_input_identity([second_artifact, first_artifact])
    changed_policy = data.sp3_merge_input_identity(
        [first_artifact, second_artifact],
        sidereon.Sp3MergeOptions(combine="median"),
    )
    precedence = sidereon.Sp3MergeOptions(combine="precedence")
    precedence_forward = data.sp3_merge_input_identity(
        [first_artifact, second_artifact], precedence
    )
    precedence_reverse = data.sp3_merge_input_identity(
        [second_artifact, first_artifact], precedence
    )
    first_mapping = first_artifact.to_dict()
    reversed_mapping = dict(reversed(list(first_mapping.items())))
    mapped = data._core_sp3_merge_input_identity([json.dumps(first_mapping)], None)
    reversed_mapped = data._core_sp3_merge_input_identity(
        [json.dumps(reversed_mapping)], None
    )

    assert original == reversed_order
    assert original != changed_policy
    assert precedence_forward != precedence_reverse
    assert mapped == reversed_mapped


def test_sp3_merge_input_identity_matches_all_surface_golden_contract():
    with open(
        os.path.join(FIXTURES, "sp3-merge-input-v1.json"), encoding="utf-8"
    ) as handle:
        fixture = json.load(handle)

    def product_identity(value):
        return distribution.ProductIdentity(
            family=value["family"],
            analysis_center=value["analysis_center"],
            publisher=value["publisher"],
            solution_class=value["solution"],
            campaign=value["campaign"],
            filename_version=value["version"],
            date=dt.date.fromisoformat(value["date"]),
            issue=value["issue"],
            span=value["span"],
            sample=value["sample"],
            official_filename=value["official_filename"],
            format=value["format"],
            format_version=value["format_version"],
            prediction_horizon_days=value["prediction_horizon_days"],
        )

    def artifact(value):
        return data.ArtifactIdentity(
            requested_identity=product_identity(value["requested_identity"]),
            resolved_identity=product_identity(value["resolved_identity"]),
            distribution_source=distribution.DistributionSource(
                value["distribution_source"]
            ),
            official_filename=value["official_filename"],
            product_sha256=value["product_sha256"],
            product_byte_length=value["product_byte_length"],
            archive_sha256=value["archive_sha256"],
            archive_byte_length=value["archive_byte_length"],
            compression=value["compression"],
        )

    def options(combine, *, reverse_sets=False, negative_zero=False):
        value = fixture["complete_policy"]
        frame = value["frame_reconciliation"]
        label_sets = frame["asserted_equivalent_label_sets"]
        systems = value["systems"]
        if reverse_sets:
            label_sets = [list(reversed(labels)) for labels in reversed(label_sets)]
            systems = list(reversed(systems))
        return sidereon.Sp3MergeOptions(
            position_tolerance_m=(
                -0.0 if negative_zero else value["position_tolerance_m"]
            ),
            clock_tolerance_s=value["clock_tolerance_s"],
            min_agree=value["min_agree"],
            clock_min_common=value["clock_min_common"],
            combine=combine,
            precedence_scope=value["precedence_scope"],
            outlier_reject=sidereon.Sp3OutlierRejectOptions(
                value["outlier_reject"]["position_tolerance_m"],
                value["outlier_reject"]["clock_tolerance_s"],
            ),
            target_epoch_interval_s=value["target_epoch_interval_s"],
            systems=systems,
            asserted_frame_label_sets=label_sets,
            helmert=frame["helmert"],
        )

    esa = artifact(fixture["artifacts"]["esa"])
    cod = artifact(fixture["artifacts"]["cod"])
    expected = fixture["expected"]

    mean = data.sp3_merge_input_identity([esa, cod], options("mean"))
    mean_reversed = data.sp3_merge_input_identity([cod, esa], options("mean"))
    mean_reordered_sets = data.sp3_merge_input_identity(
        [esa, cod], options("mean", reverse_sets=True)
    )
    median = data.sp3_merge_input_identity([esa, cod], options("median"))
    precedence = data.sp3_merge_input_identity([esa, cod], options("precedence"))
    precedence_reversed = data.sp3_merge_input_identity(
        [cod, esa], options("precedence")
    )
    single = data.sp3_merge_input_identity([esa], options("mean"))

    assert mean.schema_version == fixture["schema_version"]
    assert mean.stable_id == expected["mean_esa_cod"]
    assert mean_reversed.stable_id == expected["mean_esa_cod"]
    assert mean_reordered_sets.stable_id == expected["mean_esa_cod"]
    assert median.stable_id == expected["median_esa_cod"]
    assert precedence.stable_id == expected["precedence_esa_cod"]
    assert precedence_reversed.stable_id == expected["precedence_cod_esa"]
    assert single.stable_id == expected["single_mean_esa"]
    assert mean.canonical_contributors == mean_reversed.canonical_contributors
    assert mean.precedence_contributors is None
    assert precedence.canonical_contributors == mean.canonical_contributors
    assert precedence.precedence_contributors == (esa, cod)
    assert precedence_reversed.precedence_contributors == (cod, esa)
    assert tuple(mean) == (mean.schema_version, mean.stable_id)

    negative_zero = data.sp3_merge_input_identity(
        [esa, cod], options("mean", negative_zero=True)
    )
    assert negative_zero.stable_id == mean.stable_id
    assert options("mean", negative_zero=True).position_tolerance_m == 0.0

    mutations = fixture["required_mutations"]
    assert (
        data.sp3_merge_input_identity(
            [replace(esa, product_sha256=mutations["changed_product_sha256"]), cod],
            options("mean"),
        ).stable_id
        != mean.stable_id
    )
    assert (
        data.sp3_merge_input_identity(
            [
                replace(
                    esa,
                    resolved_identity=replace(
                        esa.resolved_identity,
                        format_version=mutations["changed_resolved_format_version"],
                    ),
                ),
                cod,
            ],
            options("mean"),
        ).stable_id
        != mean.stable_id
    )
    changed_policy = options("mean")
    changed_policy = sidereon.Sp3MergeOptions(
        position_tolerance_m=changed_policy.position_tolerance_m,
        clock_tolerance_s=mutations["changed_clock_tolerance_s"],
        min_agree=changed_policy.min_agree,
        clock_min_common=changed_policy.clock_min_common,
        combine=changed_policy.combine,
        precedence_scope=changed_policy.precedence_scope,
        outlier_reject=changed_policy.outlier_reject,
        target_epoch_interval_s=changed_policy.target_epoch_interval_s,
        systems=changed_policy.systems,
        asserted_frame_label_sets=changed_policy.asserted_frame_label_sets,
        helmert=changed_policy.helmert,
    )
    assert (
        data.sp3_merge_input_identity([esa, cod], changed_policy).stable_id
        != mean.stable_id
    )
    with pytest.raises(ValueError, match="product SHA-256"):
        data.sp3_merge_input_identity(
            [replace(esa, product_sha256=mutations["malformed_product_sha256"])]
        )
    # Merge policy identity uses the same exact 10-nanosecond interval rule as
    # the core, including fractional-second intervals.
    fractional = sidereon.Sp3MergeOptions(
        target_epoch_interval_s=mutations["fractional_target_epoch_interval_s"]
    )
    assert (
        fractional.target_epoch_interval_s
        == mutations["fractional_target_epoch_interval_s"]
    )
    assert data.sp3_merge_input_identity([esa, cod], fractional).stable_id
    off_tick = sidereon.Sp3MergeOptions(target_epoch_interval_s=1.0e-9)
    with pytest.raises(
        ValueError,
        match=(
            "target_epoch_interval_s 0.000000001 s is not an SP3 epoch interval: "
            "it is not a whole number of the 10-nanosecond ticks an SP3 epoch states"
        ),
    ) as caught:
        data.sp3_merge_input_identity([esa, cod], off_tick)
    assert caught.value.kind == "sp3_epoch_interval"
    assert caught.value.field == "target_epoch_interval_s"
    assert caught.value.value in {"1e-09", "1e-9"}
    with pytest.raises(ValueError, match="must not be empty"):
        sidereon.Sp3MergeOptions(systems=mutations["empty_systems"])


def test_sp3_merge_input_identity_changes_with_contributor_bytes(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    original_payload = _sp3_payload()
    changed_payload = original_payload.replace(
        b"/* CODE MGEX orbits", b"/* C0DE MGEX orbits", 1
    )
    assert changed_payload != original_payload
    original = _seed_exact_direct_sp3(
        str(tmp_path / "original"), product, original_payload
    )
    changed = _seed_exact_direct_sp3(
        str(tmp_path / "changed"), product, changed_payload
    )
    original_artifact = data.ArtifactIdentity._from_provenance(original.provenance)
    changed_artifact = data.ArtifactIdentity._from_provenance(changed.provenance)

    assert data.sp3_merge_input_identity(
        [original_artifact]
    ) != data.sp3_merge_input_identity([changed_artifact])


def test_sp3_merge_input_identity_accepts_zero_tolerances(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    acquired = _seed_exact_direct_sp3(str(tmp_path), product)
    artifact = data.ArtifactIdentity._from_provenance(acquired.provenance)
    policy = sidereon.Sp3MergeOptions(
        position_tolerance_m=0.0,
        clock_tolerance_s=0.0,
    )

    schema_version, stable_id = data.sp3_merge_input_identity([artifact], policy)

    assert schema_version == 1
    assert stable_id.startswith("sidereon-sp3-merge-input-v1:")


def test_sp3_merge_input_identity_rejects_malformed_provenance(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    acquired = _seed_exact_direct_sp3(str(tmp_path), product)
    artifact = data.ArtifactIdentity._from_provenance(acquired.provenance)

    with pytest.raises(ValueError, match="product SHA-256"):
        data.sp3_merge_input_identity(
            [replace(artifact, product_sha256="not-a-digest")]
        )


def test_merged_sp3_report_serialization_excludes_secrets_and_local_paths(tmp_path):
    cache = str(tmp_path / "cache-with-secret-name")
    product = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(cache, product)
    _, report = data.fetch_merged_sp3(SP3_DATE, ["cod"], cache_dir=cache, offline=True)

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert cache not in serialized
    assert "authorization" not in serialized.lower()
    assert "cookie" not in serialized.lower()
    assert "temporary" not in serialized.lower()
    assert "stable_input_identity" in serialized
    persisted = json.loads(serialized)
    assert data.verify_merge_report(persisted)

    observational_change = json.loads(serialized)
    observational_change["contributors"][0]["acquisition_facts"]["retrieved_at"] = (
        "2099-01-01T00:00:00+00:00"
    )
    observational_change["contributors"][0]["acquisition_facts"]["cache_hit"] = False
    assert data.verify_merge_report(observational_change)

    changed_artifact = json.loads(serialized)
    changed_artifact["contributors"][0]["artifact_identity"]["product_sha256"] = (
        "00" * 32
    )
    assert not data.verify_merge_report(changed_artifact)


def test_verify_merged_sp3_report_rejects_every_nested_schema_escape(tmp_path):
    products = [data.mgex_sp3("cod", SP3_DATE), data.mgex_sp3("esa", SP3_DATE)]
    payload = _sp3_payload()
    archives = {
        product.archive_url(): _archive_for_catalog_product(product, payload)
        for product in products
    }

    def handler(request):
        return httpx.Response(200, request=request, content=archives[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, report = data.fetch_merged_sp3(
            SP3_DATE,
            ["cod", "esa"],
            cache_dir=str(tmp_path),
            http_client=client,
            merge_options=sidereon.Sp3MergeOptions(
                combine="precedence",
                outlier_reject=sidereon.Sp3OutlierRejectOptions(),
            ),
        )
    persisted = json.loads(json.dumps(report.to_dict()))
    assert data.verify_merge_report(persisted)

    def changed(mutator):
        value = copy.deepcopy(persisted)
        mutator(value)
        return value

    unknown_fields = [
        lambda value: value.__setitem__("authorization", "secret"),
        lambda value: value["contributors"][0].__setitem__("local_path", "/tmp/x"),
        lambda value: value["contributors"][0]["artifact_identity"].__setitem__(
            "cookie", "secret"
        ),
        lambda value: value["contributors"][0]["artifact_identity"][
            "requested_identity"
        ].__setitem__("token", "secret"),
        lambda value: value["contributors"][0]["acquisition_facts"].__setitem__(
            "temporary_path", "/tmp/x"
        ),
        lambda value: value["merge_policy"].__setitem__("credentials", "secret"),
        lambda value: value["merge_policy"]["outlier_reject"].__setitem__(
            "password", "secret"
        ),
        lambda value: value["merge_report"].__setitem__("secret", "value"),
        lambda value: value["merge_report"]["agreement"].__setitem__(
            "cookie", "secret"
        ),
        lambda value: value["merge_report"]["agreement"]["cells"][0].__setitem__(
            "temporary_path", "/tmp/x"
        ),
        lambda value: value["merge_report"]["agreement"]["epochs"][0].__setitem__(
            "authorization", "secret"
        ),
        lambda value: value["absent"].append(
            {
                "center": "gfz",
                "filename": None,
                "reason": "no_candidate",
                "pattern": None,
                "url": None,
                "http_status": None,
                "cache_dir": "/tmp/x",
            }
        ),
    ]
    for mutation in unknown_fields:
        assert not data.verify_merge_report(changed(mutation))

    attempt = {
        "source": "direct",
        "error_type": "product_not_published",
        "message": "exact product is not published",
        "url": persisted["contributors"][0]["acquisition_facts"]["original_url"],
        "status": 404,
    }
    with_attempt = copy.deepcopy(persisted)
    with_attempt["contributors"][0]["acquisition_facts"]["attempts"].append(attempt)
    assert data.verify_merge_report(with_attempt)
    with_attempt["contributors"][0]["acquisition_facts"]["attempts"][0][
        "authorization"
    ] = "secret"
    assert not data.verify_merge_report(with_attempt)

    coercions = [
        lambda value: value.__setitem__("schema_version", "1"),
        lambda value: value.__setitem__("source_count", "2"),
        lambda value: value["contributors"][0]["artifact_identity"].__setitem__(
            "schema_version", "1"
        ),
        lambda value: value["contributors"][0]["artifact_identity"].__setitem__(
            "product_byte_length", "1597406"
        ),
        lambda value: value["contributors"][0]["acquisition_facts"].__setitem__(
            "cache_hit", 1
        ),
        lambda value: value["merge_policy"].__setitem__("position_tolerance_m", 0),
        lambda value: value["merge_policy"].__setitem__("min_agree", 2.0),
        lambda value: value["merge_report"]["agreement"]["cells"][0].__setitem__(
            "position_members", 2.0
        ),
        lambda value: value["merge_report"]["agreement"]["epochs"][0].__setitem__(
            "satellites", True
        ),
        lambda value: value.__setitem__("input_identity_schema_version", "1"),
    ]
    for mutation in coercions:
        assert not data.verify_merge_report(changed(mutation))

    inconsistencies = [
        lambda value: value["contributors"][0].__setitem__("center", "gfz"),
        lambda value: value["contributors"][0].__setitem__("date", "2020-06-26"),
        lambda value: value["contributors"][0].__setitem__("issue", "0600"),
        lambda value: value["contributors"][0].__setitem__("pattern", "alias_latest"),
        lambda value: value["contributors"][0].__setitem__("filename", "other.SP3"),
        lambda value: value.__setitem__("source_count", 1),
        lambda value: value.__setitem__("single_product", True),
        lambda value: value.__setitem__("merged", False),
        lambda value: value.__setitem__("requested_centers", ["cod"]),
        lambda value: value["requested_centers"].append("gfz"),
        lambda value: value["absent"].append(
            {
                "center": "cod",
                "filename": None,
                "reason": "no_candidate",
                "pattern": None,
                "url": None,
                "http_status": None,
            }
        ),
        lambda value: value["contributors"].reverse(),
        lambda value: value["merge_policy"]["precedence_artifact_sha256"].__setitem__(
            0, "00" * 32
        ),
        lambda value: value["merge_report"]["agreement"].__setitem__(
            "position_rms_m", 2.0
        ),
        lambda value: value["merge_report"]["agreement"]["epochs"][0].__setitem__(
            "position_max_m", -1.0
        ),
        lambda value: value["merge_report"]["agreement"]["cells"].reverse(),
        lambda value: value["merge_report"]["agreement"]["cells"][0].__setitem__(
            "satellite", "G33"
        ),
        lambda value: value["merge_report"]["agreement"]["cells"][0].__setitem__(
            "jd_fraction", 1.000_001
        ),
        lambda value: value["contributors"][0]["acquisition_facts"].__setitem__(
            "original_url",
            persisted["contributors"][0]["acquisition_facts"]["original_url"]
            + "?token=secret",
        ),
    ]
    for mutation in inconsistencies:
        assert not data.verify_merge_report(changed(mutation))


def test_verify_merged_sp3_report_rejects_unverifiable_summary_and_impossible_counts(
    tmp_path,
):
    product = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(str(tmp_path), product)
    _, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], cache_dir=str(tmp_path), offline=True
    )
    persisted = report.to_dict()
    assert data.verify_merge_report(persisted)
    assert data.verify_merge_report(json.loads(json.dumps(persisted)))

    old_summary = copy.deepcopy(persisted)
    old_summary["merge_report"] = {
        "schema_version": 1,
        "frame_reconciliation_count": 10**30,
        "quarantined_count": 10**30,
        "single_source_count": 10**30,
        "position_outlier_count": 10**30,
        "clock_outlier_count": 10**30,
        "agreement_count": 10**30,
        "position_agreement_rms_m": 0.0,
        "position_agreement_max_m": 0.0,
        "clock_agreement_rms_s": 0.0,
        "clock_agreement_max_s": 0.0,
    }
    assert not data.verify_merge_report(old_summary)

    impossible_outliers = copy.deepcopy(persisted)
    impossible_outliers["merge_report"]["position_outliers"] = [
        copy.deepcopy(impossible_outliers["merge_report"]["single_source"][0])
    ]
    impossible_outliers["merge_report"]["clock_outliers"] = [
        copy.deepcopy(impossible_outliers["merge_report"]["single_source"][1])
    ]
    assert not data.verify_merge_report(impossible_outliers)

    fabricated_dispersion = copy.deepcopy(persisted)
    fabricated_dispersion["merge_report"]["agreement"]["position_rms_m"] = 1.0
    fabricated_dispersion["merge_report"]["agreement"]["position_max_m"] = 1.0
    assert not data.verify_merge_report(fabricated_dispersion)

    missing_single_source_audit = copy.deepcopy(persisted)
    missing_single_source_audit["merge_report"]["single_source"] = []
    assert not data.verify_merge_report(missing_single_source_audit)


def _persisted_artifacts(record):
    return [
        data.ArtifactIdentity.from_dict(contributor["artifact_identity"])
        for contributor in record["contributors"]
    ]


def _rebind_persisted_merge_policy(record, options):
    artifacts = _persisted_artifacts(record)
    identity = data.sp3_merge_input_identity(artifacts, options)
    record["merge_policy"] = data._merge_policy_to_dict(options, artifacts)
    record["input_identity_schema_version"] = identity.schema_version
    record["stable_input_identity"] = identity.stable_id


def _empty_merge_audit():
    """The report schema 3 fields of a merge that dropped, omitted and withheld
    nothing and was asked for no continuity check or provenance."""
    return {
        "dropped_input_epochs": [],
        "omitted_epochs": [],
        "arc_withheld": [],
        "clock_omissions": [],
        "continuity": None,
        "provenance": None,
    }


_MERGE_AUDIT_FIELDS = tuple(_empty_merge_audit())


def _as_schema(record, schema):
    """The record as a writer of report ``schema`` spelled it: schemas 1 and 2
    carry neither the audit fields nor merge policy schema 2's reporting
    options."""
    out = copy.deepcopy(record)
    out["schema_version"] = schema
    if schema < 3:
        for name in _MERGE_AUDIT_FIELDS:
            del out["merge_report"][name]
        policy = out["merge_policy"]
        del policy["verify_continuity"]
        del policy["provenance"]
        policy["schema_version"] = 1
    return out


def _single_source_merge_result(entries):
    flags = []
    cells = []
    epochs = []
    for satellite, jd_whole, jd_fraction in entries:
        flags.append(
            {
                "satellite": satellite,
                "jd_whole": jd_whole,
                "jd_fraction": jd_fraction,
                "sources": [0],
            }
        )
        cells.append(
            {
                "satellite": satellite,
                "jd_whole": jd_whole,
                "jd_fraction": jd_fraction,
                "position_members": 1,
                "position_rms_m": 0.0,
                "position_max_m": 0.0,
                "clock_members": 1,
                "clock_rms_s": 0.0,
                "clock_max_s": 0.0,
            }
        )
        # Report schema 2: an epoch with no multi-source position consensus
        # has no position spread.
        epoch = {
            "jd_whole": jd_whole,
            "jd_fraction": jd_fraction,
            "satellites": 0,
            "position_rms_m": None,
            "position_max_m": None,
            "clock_rms_s": None,
            "clock_max_s": None,
        }
        if not epochs or (jd_whole, jd_fraction) != (
            epochs[-1]["jd_whole"],
            epochs[-1]["jd_fraction"],
        ):
            epochs.append(epoch)
    return {
        "frame_reconciliations": [],
        "quarantined": [],
        "single_source": flags,
        "position_outliers": [],
        "clock_outliers": [],
        "agreement": {
            "position_rms_m": None,
            "position_max_m": 0.0,
            "clock_rms_s": None,
            "clock_max_s": 0.0,
            "cells": cells,
            "epochs": epochs,
        },
        **_empty_merge_audit(),
    }


def _two_source_merge_result(frame_reconciliations=None):
    return {
        "frame_reconciliations": frame_reconciliations or [],
        "quarantined": [],
        "single_source": [],
        "position_outliers": [],
        "clock_outliers": [],
        "agreement": {
            "position_rms_m": 0.0,
            "position_max_m": 0.0,
            "clock_rms_s": 0.0,
            "clock_max_s": 0.0,
            "cells": [
                {
                    "satellite": "G01",
                    "jd_whole": 2_460_000.5,
                    "jd_fraction": 0.5,
                    "position_members": 2,
                    "position_rms_m": 0.0,
                    "position_max_m": 0.0,
                    "clock_members": 2,
                    "clock_rms_s": 0.0,
                    "clock_max_s": 0.0,
                }
            ],
            "epochs": [
                {
                    "jd_whole": 2_460_000.5,
                    "jd_fraction": 0.5,
                    "satellites": 1,
                    "position_rms_m": 0.0,
                    "position_max_m": 0.0,
                    "clock_rms_s": 0.0,
                    "clock_max_s": 0.0,
                }
            ],
        },
        **_empty_merge_audit(),
    }


def test_verify_merged_sp3_report_checks_satellite_leap_and_epoch_grid_domains(
    tmp_path,
):
    product = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(str(tmp_path), product)
    _, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], cache_dir=str(tmp_path), offline=True
    )
    persisted = report.to_dict()

    # The core identifier takes 01..99 under every system letter, so an
    # extended slot such as R28, which real products carry, is a valid cell.
    for satellite in ("G01", "G99", "R28", "E99", "C64", "J10", "I99", "S01", "S99"):
        boundary = copy.deepcopy(persisted)
        boundary["merge_report"] = _single_source_merge_result(
            [(satellite, 2_457_753.5, 1.0)]
        )
        assert data.verify_merge_report(boundary)
        assert data.verify_merge_report(json.loads(json.dumps(boundary)))

    for satellite in (
        "G00",
        "R00",
        "S00",
        "L01",
        "X05",
        "G999",
        "G1",
    ):
        invalid = copy.deepcopy(persisted)
        invalid["merge_report"] = _single_source_merge_result(
            [(satellite, 2_460_000.5, 0.5)]
        )
        assert not data.verify_merge_report(invalid)

    for jd_whole, jd_fraction in (
        (2_460_000.5, 1.0),
        (1_721_058.5, 0.5),
        (5_373_484.5, 0.5),
    ):
        invalid = copy.deepcopy(persisted)
        invalid["merge_report"] = _single_source_merge_result(
            [("G01", jd_whole, jd_fraction)]
        )
        assert not data.verify_merge_report(invalid)

    for jd_whole in (1_721_059.5, 5_373_483.5):
        boundary = copy.deepcopy(persisted)
        boundary["merge_report"] = _single_source_merge_result([("G01", jd_whole, 0.5)])
        assert data.verify_merge_report(boundary)

    aliased = copy.deepcopy(persisted)
    aliased["merge_report"] = _single_source_merge_result(
        [
            ("G01", 2_457_753.5, 1.0),
            ("G01", 2_457_754.5, 0.0),
        ]
    )
    assert not data.verify_merge_report(aliased)

    gridded = copy.deepcopy(persisted)
    _rebind_persisted_merge_policy(
        gridded, sidereon.Sp3MergeOptions(target_epoch_interval_s=300.0)
    )
    gridded["merge_report"] = _single_source_merge_result(
        [("G01", 2_460_000.5, second / 86_400.0) for second in (11, 311, 611)]
    )
    assert data.verify_merge_report(gridded)

    off_grid = copy.deepcopy(gridded)
    off_grid["merge_report"] = _single_source_merge_result(
        [("G01", 2_460_000.5, second / 86_400.0) for second in (11, 312, 611)]
    )
    assert not data.verify_merge_report(off_grid)


def test_verify_merged_sp3_report_checks_frames_systems_and_precedence(tmp_path):
    products = [data.mgex_sp3("cod", SP3_DATE), data.mgex_sp3("esa", SP3_DATE)]
    payload = _sp3_payload()
    archives = {
        product.archive_url(): _archive_for_catalog_product(product, payload)
        for product in products
    }

    def handler(request):
        return httpx.Response(200, request=request, content=archives[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, report = data.fetch_merged_sp3(
            SP3_DATE,
            ["cod", "esa"],
            cache_dir=str(tmp_path),
            http_client=client,
        )
    persisted = report.to_dict()

    filtered = copy.deepcopy(persisted)
    _rebind_persisted_merge_policy(filtered, sidereon.Sp3MergeOptions(systems=["G"]))
    filtered["merge_report"] = _two_source_merge_result()
    assert data.verify_merge_report(filtered)
    filtered["merge_report"]["agreement"]["cells"][0]["satellite"] = "E01"
    assert not data.verify_merge_report(filtered)

    precedence = copy.deepcopy(persisted)
    _rebind_persisted_merge_policy(
        precedence,
        sidereon.Sp3MergeOptions(combine="precedence", min_agree=1),
    )
    precedence["merge_report"] = _two_source_merge_result()
    precedence["merge_report"]["agreement"]["cells"][0]["position_members"] = 1
    precedence["merge_report"]["agreement"]["position_rms_m"] = None
    precedence["merge_report"]["agreement"]["epochs"][0]["satellites"] = 0
    precedence["merge_report"]["agreement"]["epochs"][0]["position_rms_m"] = None
    precedence["merge_report"]["agreement"]["epochs"][0]["position_max_m"] = None
    precedence["merge_report"]["position_outliers"] = [
        {
            "satellite": "G01",
            "jd_whole": 2_460_000.5,
            "jd_fraction": 0.5,
            "sources": [1],
        }
    ]
    assert data.verify_merge_report(precedence)
    precedence["merge_report"]["position_outliers"][0]["sources"] = [0]
    assert not data.verify_merge_report(precedence)

    asserted = copy.deepcopy(persisted)
    _rebind_persisted_merge_policy(
        asserted,
        sidereon.Sp3MergeOptions(asserted_frame_label_sets=[["IGS14", "ITRF2"]]),
    )
    asserted_frame = {
        "source_index": 1,
        "source_label": "ITRF2",
        "target_label": "IGS14",
        "method": "asserted_equivalence",
        "asserted_label_set": ["IGS14", "ITRF2"],
        "source_frame": None,
        "target_frame": None,
        "catalog_source_frame": None,
        "catalog_target_frame": None,
        "catalog_inverse": False,
        "reference_epoch_year": None,
        "parameters": None,
        "rates": None,
        "provenance": None,
        "epoch_year_span": None,
        "records_affected": 1,
        "identity": True,
    }
    asserted["merge_report"] = _two_source_merge_result([asserted_frame])
    assert data.verify_merge_report(asserted)
    bad_assertion = copy.deepcopy(asserted)
    bad_assertion["merge_report"]["frame_reconciliations"][0]["catalog_inverse"] = True
    assert not data.verify_merge_report(bad_assertion)

    catalog = sidereon.frame_catalog_entry("ITRF2014", "ITRF2008")
    helmert = copy.deepcopy(persisted)
    _rebind_persisted_merge_policy(helmert, sidereon.Sp3MergeOptions(helmert=True))
    helmert_frame = {
        "source_index": 1,
        "source_label": "ITRF2008",
        "target_label": "ITRF2014",
        "method": "helmert",
        "asserted_label_set": None,
        "source_frame": "ITRF2008",
        "target_frame": "ITRF2014",
        "catalog_source_frame": "ITRF2014",
        "catalog_target_frame": "ITRF2008",
        "catalog_inverse": True,
        "reference_epoch_year": catalog.reference_epoch_year,
        "parameters": {
            "translation_mm": catalog.parameters.translation_mm.tolist(),
            "scale_ppb": catalog.parameters.scale_ppb,
            "rotation_mas": catalog.parameters.rotation_mas.tolist(),
        },
        "rates": {
            "translation_mm_per_year": catalog.rates.translation_mm_per_year.tolist(),
            "scale_ppb_per_year": catalog.rates.scale_ppb_per_year,
            "rotation_mas_per_year": catalog.rates.rotation_mas_per_year.tolist(),
        },
        "provenance": catalog.provenance,
        "epoch_year_span": [2020.0, 2020.0],
        "records_affected": 1,
        "identity": False,
    }
    helmert["merge_report"] = _two_source_merge_result([helmert_frame])
    assert data.verify_merge_report(helmert)
    assert data.verify_merge_report(json.loads(json.dumps(helmert)))
    invalid_helmert = []
    for span in ([-1.0, 2020.0], [2020.0, 10_000.0]):
        invalid = copy.deepcopy(helmert)
        invalid["merge_report"]["frame_reconciliations"][0]["epoch_year_span"] = span
        invalid_helmert.append(invalid)
    bad_catalog = copy.deepcopy(helmert)
    bad_catalog["merge_report"]["frame_reconciliations"][0]["parameters"][
        "scale_ppb"
    ] += 1.0
    invalid_helmert.append(bad_catalog)
    assert all(not data.verify_merge_report(value) for value in invalid_helmert)


def test_verify_merged_sp3_report_checks_dispersion_numeric_invariants(tmp_path):
    products = [
        data.mgex_sp3(center, SP3_DATE, sample="05M")
        for center in ("cod", "esa", "gfz")
    ]
    payload = _sp3_payload()
    archives = {
        product.archive_url(): _archive_for_catalog_product(product, payload)
        for product in products
    }

    def handler(request):
        return httpx.Response(200, request=request, content=archives[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, report = data.fetch_merged_sp3(
            SP3_DATE,
            ["cod", "esa", "gfz"],
            sample="05M",
            cache_dir=str(tmp_path),
            http_client=client,
        )
    persisted = report.to_dict()

    def with_position_metric(members, rms, maximum):
        value = copy.deepcopy(persisted)
        result = _two_source_merge_result()
        cell = result["agreement"]["cells"][0]
        epoch = result["agreement"]["epochs"][0]
        cell["position_members"] = members
        cell["position_rms_m"] = rms
        cell["position_max_m"] = maximum
        result["agreement"]["position_rms_m"] = rms
        result["agreement"]["position_max_m"] = maximum
        epoch["position_rms_m"] = rms
        epoch["position_max_m"] = maximum
        value["merge_report"] = result
        return value

    impossible = with_position_metric(2, 0.0, 0.2)
    assert not data.verify_merge_report(impossible)

    underflow = with_position_metric(2, 0.0, 1.0e-300)
    assert data.verify_merge_report(underflow)

    square = 0.3 * 0.3
    equal_distance_rms = math.sqrt((square + square + square) / 3)
    assert equal_distance_rms > 0.3
    rounded = with_position_metric(3, equal_distance_rms, 0.3)
    assert data.verify_merge_report(rounded)

    precedence = with_position_metric(2, 0.8, 0.8)
    _rebind_persisted_merge_policy(
        precedence,
        sidereon.Sp3MergeOptions(
            combine="precedence",
            position_tolerance_m=1.0,
            clock_tolerance_s=1.0,
        ),
    )
    assert not data.verify_merge_report(precedence)

    odd_median_clock = with_position_metric(3, 0.0, 0.0)
    clock_cell = odd_median_clock["merge_report"]["agreement"]["cells"][0]
    clock_epoch = odd_median_clock["merge_report"]["agreement"]["epochs"][0]
    clock_cell["clock_members"] = 3
    clock_cell["clock_rms_s"] = 0.8
    clock_cell["clock_max_s"] = 0.8
    odd_median_clock["merge_report"]["agreement"]["clock_rms_s"] = 0.8
    odd_median_clock["merge_report"]["agreement"]["clock_max_s"] = 0.8
    clock_epoch["clock_rms_s"] = 0.8
    clock_epoch["clock_max_s"] = 0.8
    _rebind_persisted_merge_policy(
        odd_median_clock,
        sidereon.Sp3MergeOptions(
            combine="median",
            position_tolerance_m=1.0,
            clock_tolerance_s=1.0,
        ),
    )
    assert not data.verify_merge_report(odd_median_clock)

    huge_tolerance = with_position_metric(3, 0.0, 0.0)
    _rebind_persisted_merge_policy(
        huge_tolerance,
        sidereon.Sp3MergeOptions(
            combine="median",
            position_tolerance_m=1.0e308,
            clock_tolerance_s=1.0e308,
        ),
    )
    assert data.verify_merge_report(huge_tolerance)

    large = 9.0e153
    overflow = with_position_metric(2, large, large)
    first_cell = overflow["merge_report"]["agreement"]["cells"][0]
    second_cell = copy.deepcopy(first_cell)
    second_cell["satellite"] = "G02"
    overflow["merge_report"]["agreement"]["cells"].append(second_cell)
    overflow["merge_report"]["agreement"]["epochs"][0]["satellites"] = 2
    assert not data.verify_merge_report(overflow)


def test_verify_merged_sp3_report_accepts_exact_absent_center_partition(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    archive = _archive_for_catalog_product(product, _sp3_payload())

    def handler(request):
        if str(request.url) == product.archive_url():
            return httpx.Response(200, request=request, content=archive)
        return httpx.Response(404, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, report = data.fetch_merged_sp3(
            SP3_DATE,
            ["cod", "esa"],
            cache_dir=str(tmp_path),
            http_client=client,
        )

    persisted = report.to_dict()
    assert persisted["requested_centers"] == ["cod", "esa"]
    assert [center["center"] for center in persisted["absent"]] == ["esa"]
    assert data.verify_merge_report(persisted)

    persisted["absent"][0]["center"] = "gfz"
    assert not data.verify_merge_report(persisted)


def test_merge_report_schema_3_is_written_and_schemas_1_and_2_verify_by_their_rules(
    tmp_path,
):
    product = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(str(tmp_path), product)
    _, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], cache_dir=str(tmp_path), offline=True
    )
    persisted = report.to_dict()
    assert persisted["schema_version"] == 3
    assert persisted["merge_policy"]["schema_version"] == 2
    assert persisted["merge_policy"]["verify_continuity"] is None
    assert persisted["merge_policy"]["provenance"] is None
    for name, empty in _empty_merge_audit().items():
        assert persisted["merge_report"][name] == empty
    assert data.verify_merge_report(persisted)

    # Every cell of a single-product merge is single-source, so no epoch has a
    # multi-source position spread to report.
    for epoch in persisted["merge_report"]["agreement"]["epochs"]:
        assert epoch["satellites"] == 0
        assert epoch["position_rms_m"] is None
        assert epoch["position_max_m"] is None

    entries = [("G01", 2_460_000.5, 0.5), ("G02", 2_460_000.5, 0.5)]
    current = copy.deepcopy(persisted)
    current["merge_report"] = _single_source_merge_result(entries)
    assert data.verify_merge_report(current)

    # Schema 2 carried the same agreement without the audit fields or the
    # policy's reporting options, and still verifies as schema 2.
    schema_2 = _as_schema(current, 2)
    assert data.verify_merge_report(schema_2)
    assert data.verify_merge_report(json.loads(json.dumps(schema_2)))

    # Schema 1 spelled the same epoch as a 0.0 spread, with the maximum over
    # every cell. A record written that way still verifies as schema 1.
    legacy = _as_schema(current, 1)
    for epoch in legacy["merge_report"]["agreement"]["epochs"]:
        epoch["position_rms_m"] = 0.0
        epoch["position_max_m"] = 0.0
    assert data.verify_merge_report(legacy)
    assert data.verify_merge_report(json.loads(json.dumps(legacy)))

    # Each schema is held to its own spelling and fields, and no other version
    # is read.
    assert not data.verify_merge_report(_as_schema(current, 1))
    mixed_current = copy.deepcopy(legacy)
    mixed_current["schema_version"] = 2
    assert not data.verify_merge_report(mixed_current)
    audit_in_schema_2 = copy.deepcopy(current)
    audit_in_schema_2["schema_version"] = 2
    assert not data.verify_merge_report(audit_in_schema_2)
    schema_3_without_audit = copy.deepcopy(schema_2)
    schema_3_without_audit["schema_version"] = 3
    assert not data.verify_merge_report(schema_3_without_audit)
    old_policy_in_schema_3 = copy.deepcopy(current)
    del old_policy_in_schema_3["merge_policy"]["verify_continuity"]
    del old_policy_in_schema_3["merge_policy"]["provenance"]
    old_policy_in_schema_3["merge_policy"]["schema_version"] = 1
    assert not data.verify_merge_report(old_policy_in_schema_3)
    new_policy_in_schema_2 = copy.deepcopy(schema_2)
    new_policy_in_schema_2["merge_policy"] = copy.deepcopy(current["merge_policy"])
    assert not data.verify_merge_report(new_policy_in_schema_2)
    for version in (0, 4, True, 2.0, 3.0):
        unknown = copy.deepcopy(current)
        unknown["schema_version"] = version
        assert not data.verify_merge_report(unknown)


def _two_source_clock_only_merge_result(*, quarantined_orbit):
    """A schema 2 merge result whose G01 cell is a clock-only record.

    Without ``quarantined_orbit`` G01 is a single-source orbit and G02 a
    single-source clock-only record, the result of merging a product holding
    G01 with one holding G02's clock beside the missing-orbit sentinel. With
    it, both sources held G01 orbits that disagreed (quarantined) while their
    clocks agreed, so G01 is kept as a two-member clock-only record.
    """
    jd_whole, jd_fraction = 2_460_000.5, 0.5

    def flag(satellite, sources):
        return {
            "satellite": satellite,
            "jd_whole": jd_whole,
            "jd_fraction": jd_fraction,
            "sources": sources,
        }

    def cell(satellite, position_members, clock_members):
        orbit = position_members > 0
        return {
            "satellite": satellite,
            "jd_whole": jd_whole,
            "jd_fraction": jd_fraction,
            "position_members": position_members,
            "position_rms_m": 0.0 if orbit else None,
            "position_max_m": 0.0 if orbit else None,
            "clock_members": clock_members,
            "clock_rms_s": 0.0,
            "clock_max_s": 0.0,
        }

    if quarantined_orbit:
        quarantined = [flag("G01", [0, 1])]
        single_source = []
        cells = [cell("G01", 0, 2)]
        clock_rms, clock_max = 0.0, 0.0
        position_max = None
    else:
        quarantined = []
        single_source = [flag("G01", [0]), flag("G02", [1])]
        cells = [cell("G01", 1, 1), cell("G02", 0, 1)]
        clock_rms, clock_max = None, None
        position_max = 0.0
    return {
        "frame_reconciliations": [],
        "quarantined": quarantined,
        "single_source": single_source,
        "position_outliers": [],
        "clock_outliers": [],
        "agreement": {
            "position_rms_m": None,
            "position_max_m": position_max,
            "clock_rms_s": clock_rms,
            "clock_max_s": 0.0,
            "cells": cells,
            "epochs": [
                {
                    "jd_whole": jd_whole,
                    "jd_fraction": jd_fraction,
                    "satellites": 0,
                    "position_rms_m": None,
                    "position_max_m": None,
                    "clock_rms_s": clock_rms,
                    "clock_max_s": clock_max,
                }
            ],
        },
        **_empty_merge_audit(),
    }


def test_verify_merged_sp3_report_accepts_clock_only_cells(tmp_path):
    products = [data.mgex_sp3("cod", SP3_DATE), data.mgex_sp3("esa", SP3_DATE)]
    payload = _sp3_payload()
    archives = {
        product.archive_url(): _archive_for_catalog_product(product, payload)
        for product in products
    }

    def handler(request):
        return httpx.Response(200, request=request, content=archives[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        _, report = data.fetch_merged_sp3(
            SP3_DATE,
            ["cod", "esa"],
            cache_dir=str(tmp_path),
            http_client=client,
        )
    persisted = report.to_dict()

    for quarantined_orbit in (False, True):
        record = copy.deepcopy(persisted)
        record["merge_report"] = _two_source_clock_only_merge_result(
            quarantined_orbit=quarantined_orbit
        )
        assert data.verify_merge_report(record)
        assert data.verify_merge_report(json.loads(json.dumps(record)))

        # Schema 2 already held clock-only cells; schema 1 predates them.
        assert data.verify_merge_report(_as_schema(record, 2))
        assert not data.verify_merge_report(_as_schema(record, 1))

    clock_only = copy.deepcopy(persisted)
    clock_only["merge_report"] = _two_source_clock_only_merge_result(
        quarantined_orbit=False
    )

    fabricated_orbit = copy.deepcopy(clock_only)
    fabricated_orbit["merge_report"]["agreement"]["cells"][1]["position_rms_m"] = 0.0
    fabricated_orbit["merge_report"]["agreement"]["cells"][1]["position_max_m"] = 0.0
    assert not data.verify_merge_report(fabricated_orbit)

    no_record = copy.deepcopy(clock_only)
    no_record["merge_report"]["agreement"]["cells"][1]["clock_members"] = 0
    no_record["merge_report"]["agreement"]["cells"][1]["clock_rms_s"] = None
    no_record["merge_report"]["agreement"]["cells"][1]["clock_max_s"] = None
    assert not data.verify_merge_report(no_record)

    # G02's clock came from one source, which the core always flags.
    unaudited = copy.deepcopy(clock_only)
    unaudited["merge_report"]["single_source"].pop()
    assert not data.verify_merge_report(unaudited)

    # A quarantined orbit cannot sit beside an accepted orbit for the same cell.
    orbit_beside_quarantine = copy.deepcopy(persisted)
    result = _two_source_clock_only_merge_result(quarantined_orbit=True)
    orbit_cell = result["agreement"]["cells"][0]
    orbit_cell["position_members"] = 2
    orbit_cell["position_rms_m"] = 0.0
    orbit_cell["position_max_m"] = 0.0
    result["agreement"]["position_rms_m"] = 0.0
    result["agreement"]["position_max_m"] = 0.0
    result["agreement"]["epochs"][0]["satellites"] = 1
    result["agreement"]["epochs"][0]["position_rms_m"] = 0.0
    result["agreement"]["epochs"][0]["position_max_m"] = 0.0
    orbit_beside_quarantine["merge_report"] = result
    assert not data.verify_merge_report(orbit_beside_quarantine)


def test_single_product_schema_3_report_holds_a_clock_only_record(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(str(tmp_path), product)
    _, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], cache_dir=str(tmp_path), offline=True
    )
    record = report.to_dict()
    result = _single_source_merge_result([("G01", 2_460_000.5, 0.5)])
    cell = result["agreement"]["cells"][0]
    cell["position_members"] = 0
    cell["position_rms_m"] = None
    cell["position_max_m"] = None
    result["agreement"]["position_max_m"] = None
    record["merge_report"] = result

    # One product's clock beside its missing-orbit sentinel: a clock-only
    # record, which schemas 2 and 3 hold and schema 1 predates.
    assert data.verify_merge_report(record)
    assert data.verify_merge_report(_as_schema(record, 2))
    assert not data.verify_merge_report(_as_schema(record, 1))


def test_merge_report_leap_second_label_on_the_next_day_boundary(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(str(tmp_path), product)
    _, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], cache_dir=str(tmp_path), offline=True
    )
    persisted = report.to_dict()

    # 2016-12-31 ends with a positive leap second. The core holds a UTC
    # 23:59:60.x label on 2017-01-01's boundary with the negative fraction
    # (x - 1) / 86400, down to 23:59:60.0 at -1/86400.
    for fraction in (-0.5 / 86_400.0, -1.0 / 86_400.0, -1.0e-9):
        leap = copy.deepcopy(persisted)
        leap["merge_report"] = _single_source_merge_result(
            [("G01", 2_457_754.5, fraction)]
        )
        assert data.verify_merge_report(leap)
        assert data.verify_merge_report(json.loads(json.dumps(leap)))

    for jd_whole, fraction in (
        # The day before 2460000.5 ends without a leap second.
        (2_460_000.5, -0.5 / 86_400.0),
        # Earlier than the leap second itself.
        (2_457_754.5, -1.5 / 86_400.0),
    ):
        invalid = copy.deepcopy(persisted)
        invalid["merge_report"] = _single_source_merge_result(
            [("G01", jd_whole, fraction)]
        )
        assert not data.verify_merge_report(invalid)


def test_merge_report_schema_3_writes_one_failure_vocabulary(tmp_path):
    cache = str(tmp_path)
    _seed_exact_direct_sp3(cache, data.mgex_sp3("cod", SP3_DATE))
    _, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod", "esa"], offline=True, cache_dir=cache
    )
    persisted = report.to_dict()
    absent = persisted["absent"][0]
    assert absent["reason"] == "offline_cache_miss"
    assert absent["filename"] is not None and absent["http_status"] is None
    assert data.verify_merge_report(persisted)
    assert data.verify_merge_report(_as_schema(persisted, 2))

    def with_absence(record, **changes):
        value = copy.deepcopy(record)
        value["absent"][0].update(changes)
        return value

    def with_attempt(record, error_type, status=None, **extra):
        value = copy.deepcopy(record)
        value["contributors"][0]["acquisition_facts"]["attempts"].append(
            {
                "source": "direct",
                "error_type": error_type,
                "message": "failure",
                "url": None,
                "status": status,
                **extra,
            }
        )
        return value

    # Schema 3 writes and reads only the canonical spellings.
    for legacy in ("offline_miss", "candidate_not_found", "http_status:503"):
        assert not data.verify_merge_report(with_absence(persisted, reason=legacy))
        assert data.verify_merge_report(
            with_absence(_as_schema(persisted, 2), reason=legacy)
        )
    for legacy in ("transport", "decompression_failed", "acquisition", "unknown"):
        assert not data.verify_merge_report(with_attempt(persisted, legacy))
        assert data.verify_merge_report(with_attempt(_as_schema(persisted, 2), legacy))
    for canonical in (
        "transport_failure",
        "decompression_failure",
        "http_client_failure",
    ):
        assert data.verify_merge_report(with_attempt(persisted, canonical))
    assert data.verify_merge_report(with_attempt(persisted, "http_status", 503))

    # Unknown spellings are refused in every schema, and a canonical reason
    # must agree with the candidate it names.
    for schema in (2, 3):
        record = persisted if schema == 3 else _as_schema(persisted, 2)
        assert not data.verify_merge_report(with_absence(record, reason="gone"))
        assert not data.verify_merge_report(with_attempt(record, "mystery"))
    untried = with_absence(
        persisted, reason="no_candidate", filename=None, pattern=None, url=None
    )
    assert data.verify_merge_report(untried)
    for value in (
        with_absence(persisted, reason="http_status"),
        with_absence(persisted, reason="offline_cache_miss", http_status=404),
        with_absence(persisted, reason="no_candidate"),
        with_attempt(persisted, "http_status"),
    ):
        assert not data.verify_merge_report(value)

    # An HTTP status is 100-599 (RFC 9110 section 15), in every schema.
    for status in (100, 599):
        assert data.verify_merge_report(with_attempt(persisted, "http_status", status))
        assert data.verify_merge_report(
            with_absence(persisted, reason="http_status", http_status=status)
        )
    for schema in (2, 3):
        record = persisted if schema == 3 else _as_schema(persisted, 2)
        for status in (0, 99, 600, 999):
            assert not data.verify_merge_report(
                with_attempt(record, "http_status", status)
            )
            assert not data.verify_merge_report(
                with_absence(record, reason="product_not_published", http_status=status)
            )
    assert data.verify_merge_report(
        with_absence(_as_schema(persisted, 2), reason="http_status:599")
    )
    for legacy in ("http_status:600", "http_status:99", "http_status:0503"):
        assert not data.verify_merge_report(
            with_absence(_as_schema(persisted, 2), reason=legacy)
        )

    # A failure no canonical case describes is unclassified, with its raw
    # text in a required non-empty detail; schemas 1 and 2 predate it.
    raw = "SomeError: something the classifier has no case for"
    assert data.verify_merge_report(
        with_attempt(persisted, "unclassified_failure", detail=raw)
    )
    assert data.verify_merge_report(
        with_absence(persisted, reason="unclassified", detail=raw)
    )
    for value in (
        with_attempt(persisted, "unclassified_failure"),
        with_attempt(persisted, "unclassified_failure", detail=""),
        with_attempt(persisted, "unclassified_failure", detail=None),
        with_attempt(persisted, "transport_failure", detail=raw),
        with_absence(persisted, reason="unclassified"),
        with_absence(persisted, reason="unclassified", detail=""),
        with_absence(
            persisted,
            reason="unclassified",
            detail=raw,
            filename=None,
            pattern=None,
            url=None,
        ),
        with_attempt(_as_schema(persisted, 2), "unclassified_failure", detail=raw),
        with_absence(_as_schema(persisted, 2), reason="unclassified", detail=raw),
    ):
        assert not data.verify_merge_report(value)


def test_unclassifiable_failures_are_recorded_as_unclassified_with_their_text():
    direct = distribution.DistributionSource.DIRECT

    class SurpriseError(Exception):
        pass

    for error, raw in (
        (SurpriseError("the unexpected"), "SurpriseError: the unexpected"),
        (distribution.AcquisitionError("base"), "acquisition_error: base"),
    ):
        failure = distribution._source_failure(direct, error)
        assert failure.error_type == "unclassified_failure"
        assert failure.detail == raw
        assert failure.to_dict()["detail"] == raw
        assert distribution.SourceFailure.from_dict(failure.to_dict()) == failure

    classified = distribution._source_failure(
        direct, distribution.ProductNotPublished(404, "https://example.test/x", "gone")
    )
    assert classified.error_type == "product_not_published"
    assert classified.detail is None
    assert "detail" not in classified.to_dict()

    class MalformedCodeError(Exception):
        def __init__(self, code, message="foreign failure"):
            super().__init__(message)
            self.code = code

    for malformed_code in (["future"], {"future": "failure"}, 17, ""):
        malformed = distribution._source_failure(
            direct, MalformedCodeError(malformed_code)
        )
        assert malformed.error_type == "unclassified_failure"
        assert malformed.detail == "MalformedCodeError: foreign failure"
        assert isinstance(malformed.detail, str)
        assert malformed.message == "foreign failure"
        assert distribution.SourceFailure.from_dict(malformed.to_dict()) == malformed

    class EmptyTextMalformedCodeError(MalformedCodeError):
        def __str__(self):
            return ""

    empty_text = distribution._source_failure(
        direct, EmptyTextMalformedCodeError({"future": "failure"})
    )
    assert empty_text.error_type == "unclassified_failure"
    assert empty_text.detail == "EmptyTextMalformedCodeError"
    assert isinstance(empty_text.detail, str)
    assert empty_text.message == "unclassified_failure"
    assert distribution.SourceFailure.from_dict(empty_text.to_dict()) == empty_text

    # The public serializer/deserializer pair preserves optional None values,
    # but rejects malformed persisted records instead of coercing or dropping
    # fields. Classified failures cannot carry unclassified raw detail.
    assert distribution.SourceFailure.from_dict(classified.to_dict()) == classified
    no_location = distribution._source_failure(
        direct, distribution.ProductValidationFailure("content rejected")
    )
    assert no_location.url is None and no_location.status is None
    assert distribution.SourceFailure.from_dict(no_location.to_dict()) == no_location
    for invalid_record in (
        {**classified.to_dict(), "detail": "unexpected detail"},
        {**classified.to_dict(), "status": True},
        {**classified.to_dict(), "status": 99},
        {**classified.to_dict(), "status": 600},
        {**classified.to_dict(), "error_type": "future_failure"},
        {**empty_text.to_dict(), "detail": None},
        {key: value for key, value in empty_text.to_dict().items() if key != "detail"},
        {**empty_text.to_dict(), "detail": ""},
    ):
        with pytest.raises(ValueError):
            distribution.SourceFailure.from_dict(invalid_record)

    status = distribution._source_failure(
        direct, distribution.HttpStatusFailure(503, "https://example.test/x")
    )
    assert (status.error_type, status.status) == ("http_status", 503)
    # A status outside 100-599 is no HTTP status and is not recorded as one.
    invalid = distribution.TransportFailure("http_600", "https://example.test/x")
    invalid.status = 600
    recorded = distribution._source_failure(direct, invalid)
    assert (recorded.error_type, recorded.status) == ("transport_failure", None)
    assert "http_600" in recorded.message

    assert data._absence_reason(data.HttpStatusError(503, "u")) == (
        "http_status",
        None,
    )
    reason, detail = data._absence_reason(data.HttpStatusError(600, "u"))
    assert reason == "unclassified"
    assert detail == "HttpStatusError: HTTP 600 for u"
    reason, detail = data._absence_reason(data.NetworkError("down"))
    assert (reason, detail) == ("unclassified", "NetworkError: down")
    absent = data.AbsentCenter(
        "esa", "F.SP3", reason, None, "https://x/F", None, detail
    )
    assert absent.to_dict()["detail"] == detail
    assert "detail" not in data.AbsentCenter("esa", None, "no_candidate").to_dict()


def test_merge_report_schema_3_fixture_from_the_elixir_interface():
    """The schema 3 record the Elixir interface writes verifies here, and a
    tampered copy does not."""
    with open(os.path.join(FIXTURES, "sp3-merge-report-v3.json")) as handle:
        record = json.load(handle)
    assert record["schema_version"] == 3
    # ESA contributed after a transport failure; COD was absent, its candidate
    # not published.
    assert record["requested_centers"] == ["esa", "cod"]
    assert [item["reason"] for item in record["absent"]] == ["product_not_published"]
    assert record["absent"][0]["http_status"] == 404
    attempts = record["contributors"][0]["acquisition_facts"]["attempts"]
    assert [item["error_type"] for item in attempts] == [
        "transport_failure",
        "unclassified_failure",
    ]
    assert "detail" not in attempts[0]
    assert attempts[1]["detail"] == "{:tls_alert, :bad_record_mac}"
    assert data.verify_merge_report(record)

    def changed(mutation):
        value = copy.deepcopy(record)
        mutation(value)
        return value

    tampered = [
        lambda value: value["merge_report"]["continuity"].__setitem__("attested", True),
        lambda value: value["merge_report"]["continuity"].__setitem__(
            "pairs_checked", 0
        ),
        lambda value: value["merge_report"]["provenance"]["cells"].pop(),
        lambda value: value["merge_policy"].__setitem__("provenance", "summary"),
        lambda value: value["merge_policy"]["verify_continuity"].__setitem__(
            "gap_threshold_factor", 1.0
        ),
        lambda value: value.__setitem__("stable_input_identity", "0" * 16),
        # The absent center and the contributor's failed attempt are held to
        # the canonical vocabulary and the HTTP status range.
        lambda value: value["absent"][0].__setitem__("reason", "candidate_not_found"),
        lambda value: value["absent"][0].__setitem__("http_status", 700),
        lambda value: value["absent"].clear(),
        lambda value: value["contributors"][0]["acquisition_facts"]["attempts"][
            0
        ].__setitem__("error_type", "transport"),
        lambda value: value["contributors"][0]["acquisition_facts"]["attempts"][
            0
        ].__setitem__("status", 99),
        # An unclassified failure keeps its raw text; without it the record is
        # refused.
        lambda value: value["contributors"][0]["acquisition_facts"]["attempts"][1].pop(
            "detail"
        ),
    ]
    for mutation in tampered:
        assert not data.verify_merge_report(changed(mutation))

    # Summary mode omits per-cell membership but still has to refer to the
    # accepted agreement grid. Rebind the stable identity after changing the
    # requested provenance mode so each mutation starts from a valid record.
    summary = copy.deepcopy(record)
    summary["merge_policy"]["provenance"] = "summary"
    summary["merge_report"]["provenance"]["mode"] = "summary"
    summary["merge_report"]["provenance"]["cells"] = []
    _rebind_persisted_merge_policy(
        summary, data._merge_options_from_policy(summary["merge_policy"])
    )
    assert data.verify_merge_report(summary)

    def summary_changed(mutation):
        value = copy.deepcopy(summary)
        mutation(value)
        return value

    absent_satellite = summary_changed(
        lambda value: value["merge_report"]["provenance"]["transitions"][0].__setitem__(
            "satellite", "G02"
        )
    )
    assert not data.verify_merge_report(absent_satellite)
    absent_epoch = summary_changed(
        lambda value: value["merge_report"]["provenance"]["transitions"][0][
            "epoch"
        ].__setitem__(
            "jd_whole",
            value["merge_report"]["provenance"]["transitions"][0]["epoch"]["jd_whole"]
            + 1,
        )
    )
    assert not data.verify_merge_report(absent_epoch)
    absent_coverage_span = summary_changed(
        lambda value: [
            value["merge_report"]["provenance"]["coverage"][0][endpoint].__setitem__(
                "jd_whole",
                value["merge_report"]["provenance"]["coverage"][0][endpoint]["jd_whole"]
                + 1,
            )
            for endpoint in ("first_epoch", "last_epoch")
        ]
    )
    assert not data.verify_merge_report(absent_coverage_span)
    off_grid_coverage_span = summary_changed(
        lambda value: [
            value["merge_report"]["provenance"]["coverage"][0][endpoint].__setitem__(
                "jd_fraction",
                value["merge_report"]["provenance"]["coverage"][0][endpoint][
                    "jd_fraction"
                ]
                + 0.001,
            )
            for endpoint in ("first_epoch", "last_epoch")
        ]
    )
    assert not data.verify_merge_report(off_grid_coverage_span)

    # Canonical class bounds are delegated to the core; changing the class and
    # all redundant bound fields remains valid, while a self-consistent forged
    # value is refused.
    for orbit_class in ("meo_gnss", "geosynchronous", "leo"):
        candidate = copy.deepcopy(record)
        bound = data._core_sp3_orbit_class_speed_bound_m_s(orbit_class)
        candidate["merge_policy"]["verify_continuity"]["orbit_class"] = orbit_class
        for defect in candidate["merge_report"]["continuity"]["defects"]:
            if defect["kind"] == "speed_bound":
                defect["displacement_m"] = (bound + 1.0) * defect["interval_s"]
                speed = defect["displacement_m"] / defect["interval_s"]
                defect["implied_speed_m_s"] = defect["magnitude"] = speed
                defect["bound"] = defect["bound_m_s"] = bound
        for violation in candidate["merge_report"]["continuity"]["violations"]:
            defect = violation["defect"]
            if defect["kind"] == "speed_bound":
                defect["displacement_m"] = (bound + 1.0) * defect["interval_s"]
                speed = defect["displacement_m"] / defect["interval_s"]
                defect["implied_speed_m_s"] = defect["magnitude"] = speed
                defect["bound"] = defect["bound_m_s"] = bound
        _rebind_persisted_merge_policy(
            candidate, data._merge_options_from_policy(candidate["merge_policy"])
        )
        assert data.verify_merge_report(candidate)

    wrong_class_bound = copy.deepcopy(record)
    for defect in wrong_class_bound["merge_report"]["continuity"]["defects"]:
        defect["bound"] = defect["bound_m_s"] = 1.0
    for violation in wrong_class_bound["merge_report"]["continuity"]["violations"]:
        violation["defect"]["bound"] = violation["defect"]["bound_m_s"] = 1.0
    assert not data.verify_merge_report(wrong_class_bound)


def _fetch_two_contributor_merge(tmp_path, merge_options=None):
    """Merge COD and ESA copies of one product through `fetch_merged_sp3`."""
    products = [data.mgex_sp3("cod", SP3_DATE), data.mgex_sp3("esa", SP3_DATE)]
    payload = _sp3_payload()
    archives = {
        product.archive_url(): _archive_for_catalog_product(product, payload)
        for product in products
    }

    def handler(request):
        return httpx.Response(200, request=request, content=archives[str(request.url)])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return data.fetch_merged_sp3(
            SP3_DATE,
            ["cod", "esa"],
            cache_dir=str(tmp_path),
            http_client=client,
            merge_options=merge_options,
        )


def _epoch_of(instant):
    return {
        "time_scale": instant.scale.abbrev,
        "jd_whole": instant.jd_whole,
        "jd_fraction": instant.fraction,
    }


def test_merge_report_schema_3_round_trips_provenance_and_continuity(tmp_path):
    options = sidereon.Sp3MergeOptions(
        systems=["J"],
        provenance="full",
        verify_continuity=sidereon.Sp3ContinuityOptions(),
    )
    _, report = _fetch_two_contributor_merge(tmp_path, options)
    persisted = report.to_dict()
    native = report.merge_report

    assert persisted["schema_version"] == 3
    policy = persisted["merge_policy"]
    assert policy["schema_version"] == 2
    assert policy["provenance"] == "full"
    assert policy["verify_continuity"] == {
        "orbit_class": "meo_gnss",
        "residual_tolerance_m": 1.0,
        "gap_threshold_factor": None,
    }
    assert data.verify_merge_report(persisted)
    assert data.verify_merge_report(json.loads(json.dumps(persisted)))

    # Nothing the binding reports is lost on the way to the record.
    result = persisted["merge_report"]
    provenance = result["provenance"]
    assert provenance["mode"] == "full"
    assert len(provenance["cells"]) == len(native.provenance.cells)
    assert len(provenance["cells"]) == len(result["agreement"]["cells"]) > 0
    for stored, cell in zip(provenance["cells"], native.provenance.cells):
        assert stored["epoch"] == _epoch_of(cell.epoch)
        assert stored["satellite"] == cell.satellite
        assert stored["position"] == {
            "kind": "combined",
            "rule": "mean",
            "members": list(cell.position.members),
        }
    assert [
        (item["from_source"], item["to_source"], item["reason"])
        for item in provenance["transitions"]
    ] == [
        (item.from_source, item.to_source, item.reason)
        for item in native.provenance.transitions
    ]
    assert [item["cells_contributed"] for item in provenance["coverage"]] == [
        item.cells_contributed for item in native.provenance.coverage
    ]
    continuity = result["continuity"]
    assert continuity["attested"] == native.continuity.attested
    assert len(continuity["defects"]) == len(native.continuity.defects)
    assert len(continuity["violations"]) == len(native.continuity.violations)
    for stored, violation in zip(
        continuity["violations"], native.continuity.violations
    ):
        assert stored["sources"] == violation.sources
        assert [cell["epoch_j2000_s"] for cell in stored["cells"]] == [
            cell.epoch_j2000_s for cell in violation.cells
        ]
    assert continuity["splices"] == [
        item for item in continuity["violations"] if item["crosses_contributors"]
    ]

    # Schema 2 and 1 spellings of the same merge still verify by their rules.
    assert data.verify_merge_report(_as_schema(persisted, 2))

    def changed(mutation):
        value = copy.deepcopy(persisted)
        mutation(value)
        return value

    first_cell = result["agreement"]["cells"][0]
    first_epoch = {
        "time_scale": provenance["cells"][0]["epoch"]["time_scale"],
        "jd_whole": first_cell["jd_whole"],
        "jd_fraction": first_cell["jd_fraction"],
    }
    inconsistencies = [
        lambda value: value["merge_report"].__setitem__("provenance", None),
        lambda value: value["merge_policy"].__setitem__("provenance", "summary"),
        lambda value: value["merge_policy"].__setitem__("provenance", None),
        lambda value: value["merge_report"]["provenance"]["cells"][0][
            "position"
        ].__setitem__("members", [0]),
        lambda value: value["merge_report"]["provenance"]["cells"][0][
            "position"
        ].__setitem__("rule", "median"),
        lambda value: value["merge_report"]["provenance"]["cells"].pop(),
        lambda value: value["merge_report"]["provenance"]["cells"][0][
            "epoch"
        ].__setitem__("time_scale", "UT1"),
        lambda value: value["merge_report"]["provenance"]["coverage"][0].__setitem__(
            "cells_selected", 1
        ),
        lambda value: value["merge_report"]["provenance"]["coverage"][0].__setitem__(
            "cells_contributed",
            value["merge_report"]["provenance"]["coverage"][0]["cells_contributed"] - 1,
        ),
        lambda value: value["merge_report"]["provenance"]["transitions"][0].__setitem__(
            "reason", "whim"
        ),
        lambda value: value["merge_report"].__setitem__("continuity", None),
        lambda value: value["merge_policy"].__setitem__("verify_continuity", None),
        lambda value: value["merge_policy"]["verify_continuity"].__setitem__(
            "residual_tolerance_m", None
        ),
        # The core's interpolation options take a factor greater than 1.
        lambda value: value["merge_policy"]["verify_continuity"].__setitem__(
            "gap_threshold_factor", 1.0
        ),
        lambda value: value["merge_report"]["continuity"].__setitem__(
            "attested", not value["merge_report"]["continuity"]["attested"]
        ),
        lambda value: value["merge_report"]["continuity"].__setitem__("splices", [{}]),
        lambda value: value["merge_report"]["omitted_epochs"].append(first_epoch),
        lambda value: value["merge_report"]["arc_withheld"].append(
            {"satellite": first_cell["satellite"], "epoch": first_epoch, "sources": [1]}
        ),
        lambda value: value["merge_report"]["dropped_input_epochs"].append(
            {
                "source": 1,
                "epoch_index": 0,
                "epoch": first_epoch,
                "reason": "off_target_grid",
            }
        ),
        lambda value: value["merge_report"]["clock_omissions"].append(
            {
                "epoch": first_epoch,
                "satellite": first_cell["satellite"],
                "source": 1,
                "reason": "no_consensus",
                "preferred": None,
                "cell_has_clock": False,
            }
        ),
        lambda value: value["merge_report"]["clock_omissions"].append(
            {
                "epoch": first_epoch,
                "satellite": first_cell["satellite"],
                "source": 0,
                "reason": "datum_not_observable",
                "preferred": None,
                "cell_has_clock": True,
            }
        ),
        lambda value: value["merge_report"]["clock_omissions"].append(
            {
                "epoch": first_epoch,
                "satellite": first_cell["satellite"],
                "source": 1,
                "reason": "preferred_source_without_clock",
                "preferred": 0,
                "cell_has_clock": True,
            }
        ),
    ]
    for mutation in inconsistencies:
        assert not data.verify_merge_report(changed(mutation))
    explicit_factor = changed(
        lambda value: value["merge_policy"]["verify_continuity"].__setitem__(
            "gap_threshold_factor", 1.5
        )
    )
    assert data.verify_merge_report(explicit_factor)


def test_merge_report_schema_3_verifies_core_withheld_arcs_omissions_and_splices(
    tmp_path,
):
    """Core merges of the coverage fixture, recorded in a schema 3 report.

    Satellite-arc precedence omits the epochs past source A's end and
    withholds and omits source B's positions and clocks there; cell precedence
    over a 0.8 m handover yields continuity violations across contributors.
    Each record verifies, and each audit it carries is checked against the
    agreement it sits beside.
    """
    from _sp3_merge import coverage_product

    _, report = _fetch_two_contributor_merge(tmp_path)
    persisted = report.to_dict()
    a = coverage_product("a_0_72", list(range(0, 72)))
    b = coverage_product("b_60_24", list(range(60, 84)))
    b_displaced = coverage_product("b_60_24_displaced_0_8", list(range(60, 84)), 0.8)

    def recorded(sources, options):
        _merged, merge_report = sidereon.merge_sp3(sources, options)
        record = copy.deepcopy(persisted)
        _rebind_persisted_merge_policy(record, options)
        record["merge_report"] = data._merge_result_to_dict(merge_report)
        return record

    arc = recorded(
        [a, b],
        sidereon.Sp3MergeOptions(
            combine="precedence",
            precedence_scope="satellite_arc",
            min_agree=1,
            position_tolerance_m=5.0,
            provenance="full",
        ),
    )
    assert data.verify_merge_report(arc)
    assert data.verify_merge_report(json.loads(json.dumps(arc)))
    arc_result = arc["merge_report"]
    assert len(arc_result["omitted_epochs"]) == 12
    assert len(arc_result["arc_withheld"]) == 6 * 12
    assert len(arc_result["clock_omissions"]) == 6 * 12
    assert all(
        item["reason"] == "datum_not_observable" and item["source"] == 1
        for item in arc_result["clock_omissions"]
    )
    assert len(arc_result["provenance"]["cells"]) == len(
        arc_result["agreement"]["cells"]
    )

    accepted = arc_result["agreement"]["cells"][0]
    accepted_epoch = {
        "time_scale": arc_result["omitted_epochs"][0]["time_scale"],
        "jd_whole": accepted["jd_whole"],
        "jd_fraction": accepted["jd_fraction"],
    }

    def changed(record, mutation):
        value = copy.deepcopy(record)
        mutation(value["merge_report"])
        return value

    arc_inconsistencies = [
        # A withheld position is one the product does not hold.
        lambda value: (
            value["arc_withheld"][0].__setitem__("epoch", accepted_epoch)
            or value["arc_withheld"][0].__setitem__("satellite", accepted["satellite"])
        ),
        # An omitted epoch holds no accepted cell, and they are in time order.
        lambda value: value["omitted_epochs"].__setitem__(0, accepted_epoch),
        lambda value: value["omitted_epochs"].reverse(),
        # Source 0 is the clock datum.
        lambda value: value["clock_omissions"][0].__setitem__("source", 0),
        lambda value: value["clock_omissions"][0].__setitem__("cell_has_clock", True),
        lambda value: value["clock_omissions"].reverse(),
        # The full provenance of an accepted cell names its writer.
        lambda value: value["provenance"]["cells"][0].__setitem__(
            "position", {"kind": "single_source", "source": 1}
        ),
    ]
    for mutation in arc_inconsistencies:
        assert not data.verify_merge_report(changed(arc, mutation))
    # Cell precedence always prefers a source that carries the cell.
    cell_policy = copy.deepcopy(arc)
    _rebind_persisted_merge_policy(
        cell_policy,
        sidereon.Sp3MergeOptions(
            combine="precedence",
            precedence_scope="cell",
            min_agree=1,
            position_tolerance_m=5.0,
            provenance="full",
        ),
    )
    assert not data.verify_merge_report(cell_policy)

    splice = recorded(
        [a, b_displaced],
        sidereon.Sp3MergeOptions(
            combine="precedence",
            precedence_scope="cell",
            min_agree=1,
            position_tolerance_m=5.0,
            verify_continuity=sidereon.Sp3ContinuityOptions(),
            provenance="summary",
        ),
    )
    assert data.verify_merge_report(splice)
    assert data.verify_merge_report(json.loads(json.dumps(splice)))
    continuity = splice["merge_report"]["continuity"]
    assert not continuity["attested"]
    assert continuity["splices"]
    assert splice["merge_report"]["provenance"]["cells"] == []

    index = continuity["violations"].index(continuity["splices"][0])
    splice_inconsistencies = [
        lambda value: value["continuity"]["violations"][index].__setitem__(
            "crosses_contributors", False
        ),
        lambda value: value["continuity"]["violations"][index]["cells"].pop(),
        lambda value: value["continuity"]["violations"][index].__setitem__(
            "sources", [1]
        ),
        lambda value: value["continuity"]["violations"][index]["cells"][0].__setitem__(
            "role", "pair_end"
        ),
        lambda value: value["continuity"]["splices"].pop(),
        lambda value: value["continuity"]["violations"].pop(),
        lambda value: value["continuity"]["defects"][0].__setitem__(
            "magnitude", value["continuity"]["defects"][0]["magnitude"] * 2.0
        ),
        lambda value: value["provenance"].__setitem__(
            "cells", copy.deepcopy(arc["merge_report"]["provenance"]["cells"])
        ),
        # Each finding is on a sample the residual check examined.
        lambda value: value["continuity"].__setitem__("residuals_checked", 0),
    ]
    for mutation in splice_inconsistencies:
        assert not data.verify_merge_report(changed(splice, mutation))


def test_ultra_sp3_candidates_include_only_evidenced_current_product():
    candidates = data._sp3_candidates("esa_ult", dt.date(2026, 7, 13), None)

    assert [candidate.pattern for candidate in candidates] == ["primary_02D_05M"]


def test_code_ultra_sp3_candidates_exclude_the_moving_latest_snapshot():
    candidates = data._sp3_candidates("cod_ult", dt.date(2026, 7, 14), None)

    assert [candidate.pattern for candidate in candidates] == ["primary_01D_05M"]
    assert [candidate.archive_url() for candidate in candidates] == [
        "https://www.aiub.unibe.ch/download/CODE/"
        "COD0OPSULT_20261950000_01D_05M_ORB.SP3",
    ]


def test_aiub_download_follows_only_validated_object_store_redirect(monkeypatch):
    source = "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT.SP3"
    download = "https://download.aiub.unibe.ch/CODE/COD0OPSULT.SP3"
    target = "https://zhw-b.s3.cloud.switch.ch/aiub/CODE/COD0OPSULT.SP3"
    calls = _stub_http(
        monkeypatch,
        [
            (302, b"", {"location": download}),
            (301, b"", {"location": target}),
            (200, b"#dP fixture"),
        ],
    )

    assert data._download_once(source, 1.0, 1024) == b"#dP fixture"
    assert [call["url"] for call in calls] == [source, download, target]


def test_aiub_download_rejects_untrusted_redirect_target(monkeypatch):
    source = "https://www.aiub.unibe.ch/download/CODE/COD0OPSULT.SP3"
    _stub_http(
        monkeypatch,
        [(302, b"", {"location": "https://example.com/COD0OPSULT.SP3"})],
    )

    with pytest.raises(data.RedirectNotAllowed):
        data._download_once(source, 1.0, 1024)


def test_gfz_documented_cadence_overlap_falls_back_after_dated_absence(monkeypatch):
    calls = []

    def fake_acquire(product, **_kwargs):
        calls.append(product.pattern)
        if product.pattern == "primary_02D_15M":
            raise distribution.ProductNotPublished(
                404, product.archive_url(), "not published"
            )
        return distribution.AcquiredProduct(
            _core_sp3("COD0MGXFIN_20201770000_01D_05M_ORB.SP3"),
            _provenance_for_catalog_product(product),
        )

    monkeypatch.setattr(distribution, "_acquire_catalog_product", fake_acquire)
    result = data._fetch_center_sp3("gfz_ult", dt.date(2021, 5, 15), None, {})

    assert result[0] == "ok"
    assert result[1].pattern == "alternate_02D_05M"
    assert calls == ["primary_02D_15M", "alternate_02D_05M"]
    attempts = result[1].acquisition_facts.attempts
    assert len(attempts) == 1
    assert attempts[0].source is distribution.DistributionSource.DIRECT
    assert attempts[0].error_type == "product_not_published"
    assert attempts[0].status == 404
    assert attempts[0].url.endswith("_02D_15M_ORB.SP3.gz")


def test_ultra_sp3_all_variants_missing_records_absence(monkeypatch):
    calls = []

    def fake_acquire(product, **_kwargs):
        calls.append(product.pattern)
        raise distribution.ProductNotPublished(
            404, product.archive_url(), "not published"
        )

    monkeypatch.setattr(distribution, "_acquire_catalog_product", fake_acquire)
    result = data._fetch_center_sp3("gfz_ult", dt.date(2021, 5, 15), None, {})

    assert result[0] == "absent"
    assert result[1].reason == "product_not_published"
    assert result[1].pattern == "alternate_02D_05M"
    assert result[1].url is not None
    assert result[1].url.endswith("_02D_05M_ORB.SP3.gz")
    assert result[1].http_status == 404
    assert calls == [
        "primary_02D_15M",
        "alternate_02D_05M",
    ]


def test_fetch_merged_sp3_offline_records_absent_centers(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(cache, prod)

    # esa is requested but not cached: it must be reported absent, not abort.
    sp3, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod", "esa"], offline=True, cache_dir=cache
    )
    assert isinstance(sp3, sidereon.Sp3)
    assert report.source_count == 1
    assert [a.center for a in report.absent] == ["esa"]
    assert report.absent[0].reason == "offline_cache_miss"


def test_fetch_merged_sp3_offline_empty_cache_raises_no_products(tmp_path):
    with pytest.raises(data.NoProducts) as excinfo:
        data.fetch_merged_sp3(SP3_DATE, ["cod"], offline=True, cache_dir=str(tmp_path))
    reasons = excinfo.value.reasons
    assert [r.center for r in reasons] == ["cod"]
    assert reasons[0].reason == "offline_cache_miss"


def test_fetch_merged_sp3_rejects_legacy_digest_only_cache(tmp_path):
    product = data.mgex_sp3("cod", SP3_DATE)
    _seed(
        str(tmp_path),
        product,
        _core_sp3("COD0MGXFIN_20201770000_01D_05M_ORB.SP3"),
    )

    with pytest.raises(data.NoProducts) as excinfo:
        data.fetch_merged_sp3(SP3_DATE, ["cod"], cache_dir=str(tmp_path), offline=True)

    assert excinfo.value.reasons[0].reason == "offline_cache_miss"


def test_fetch_merged_sp3_unknown_center_raises(tmp_path):
    with pytest.raises(data.UnknownCenter):
        data.fetch_merged_sp3(
            SP3_DATE, ["cod", "bogus"], offline=True, cache_dir=str(tmp_path)
        )


def test_fetch_merged_sp3_file_offline_writes_nothing_on_miss(tmp_path):
    out = str(tmp_path / "merged.sp3")
    with pytest.raises(data.NoProducts):
        data.fetch_merged_sp3_file(
            SP3_DATE, ["cod"], out, offline=True, cache_dir=str(tmp_path)
        )
    assert not os.path.exists(out)


def test_fetch_merged_sp3_file_offline_writes_single_contributor(tmp_path):
    cache = str(tmp_path / "cache")
    os.makedirs(cache)
    prod = data.mgex_sp3("cod", SP3_DATE)
    _seed_exact_direct_sp3(cache, prod)

    out = str(tmp_path / "merged.sp3")
    written = data.fetch_merged_sp3_file(
        SP3_DATE, ["cod"], out, offline=True, cache_dir=cache
    )
    assert written == out
    assert os.path.exists(out)
    # The written file is a standard SP3 the loader round-trips.
    reloaded = sidereon.load_sp3(out)
    assert isinstance(reloaded, sidereon.Sp3)

    written_with_report, report = data.fetch_merged_sp3_file(
        SP3_DATE,
        ["cod"],
        out,
        offline=True,
        cache_dir=cache,
        return_report=True,
    )
    assert written_with_report == out
    assert report.stable_input_identity is not None
    assert report.contributors[0].artifact_identity is not None


# --- cache behavior ------------------------------------------------------


def test_fetch_verified_hit_uses_provenance_sidecar(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_ionex("esa", dt.date(2024, 6, 24))
    path = _seed(cache, prod, _core_ionex("synthetic_2map_7x7.20i"))

    # A cache entry with a matching provenance sidecar is a verified hit offline.
    returned = data.fetch(prod, offline=True, cache_dir=cache)
    assert returned == path


def test_fetch_offline_unverified_no_sidecar_is_miss(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_ionex("esa", dt.date(2024, 6, 24))
    # A bare hand-placed file with no provenance is untrusted; offline it must be
    # a miss, not silently returned.
    _seed_bare(cache, prod, _core_ionex("synthetic_2map_7x7.20i"))
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch(prod, offline=True, cache_dir=cache)


def test_fetch_offline_caller_checksum_mismatch_raises(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_ionex("esa", dt.date(2024, 6, 24))
    _seed(cache, prod, _core_ionex("synthetic_2map_7x7.20i"))

    with pytest.raises(data.ChecksumMismatch):
        data.fetch(prod, offline=True, cache_dir=cache, sha256="00" * 32)


def test_default_cache_dir_is_under_user_cache():
    d = data.default_cache_dir()
    assert d.endswith(os.path.join("sidereon", "gnss"))


# --- terrain data layer --------------------------------------------------


def test_terrain_derivation_comes_from_core():
    source = data.skadi_source_entry()
    assert source.protocol == "https"
    assert source.host in data._ALLOWED_HOSTS
    assert source.compression == "gzip"
    assert data.skadi_tile_id(36, -107) == "N36W107"
    assert data.skadi_band(36) == "N36"
    assert data.skadi_archive_url(36, -107).endswith("/skadi/N36/N36W107.hgt.gz")
    assert data.dted_cache_relpath(36, -107) == ("n30_w100/n36_w107_1arc_v3.dt2")
    assert data.terrain_tile_index(90.0, 180.0) == (89, 179)
    assert data.parse_skadi_tile_id("S01E010") == (-1, 10)
    with pytest.raises(data.InvalidTileId):
        data.parse_skadi_tile_id("n36w107")


def test_fetch_dted_cache_hit_uses_zero_network(tmp_path, monkeypatch):
    cache = str(tmp_path)
    path = _seed_terrain(cache)

    def fail_stream(*args, **kwargs):
        raise AssertionError("cache hit must not touch network")

    monkeypatch.setattr(data.httpx, "stream", fail_stream)
    assert data.fetch_dted(36.5, -106.5, cache_dir=cache) == path


def test_fetch_dted_offline_hit_reads_verified_cache(tmp_path, monkeypatch):
    cache = str(tmp_path)
    path = _seed_terrain(cache)

    def fail_stream(*args, **kwargs):
        raise AssertionError("offline hit must not touch network")

    monkeypatch.setattr(data.httpx, "stream", fail_stream)
    assert data.fetch_dted(36.5, -106.5, cache_dir=cache, offline=True) == path


def test_fetch_dted_offline_miss_and_unverified_file_are_misses(tmp_path):
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch_dted(36.5, -106.5, cache_dir=str(tmp_path), offline=True)

    cache = str(tmp_path / "bare")
    _seed_terrain_bare(cache)
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch_dted(36.5, -106.5, cache_dir=cache, offline=True)


def test_fetch_dted_checksum_failure_is_typed(tmp_path):
    cache = str(tmp_path)
    _seed_terrain(cache)

    with pytest.raises(data.ChecksumMismatch):
        data.fetch_dted(
            36.5,
            -106.5,
            cache_dir=cache,
            offline=True,
            sha256="00" * 32,
        )


def test_fetch_dted_ocean_404_writes_negative_cache(tmp_path, monkeypatch):
    cache = str(tmp_path)
    calls = _stub_http(monkeypatch, [(404, b"")])

    assert data.fetch_dted(0.5, -159.5, cache_dir=cache) is None
    assert len(calls) == 1

    relpath = data.dted_cache_relpath(0, -160)
    marker_path = os.path.join(cache, relpath) + ".no_coverage.json"
    with open(marker_path) as handle:
        marker = json.load(handle)
    assert marker["status"] == 404
    assert marker["tile_id"] == "N00W160"
    assert marker["source_url"] == data.skadi_archive_url(0, -160)

    def fail_stream(*args, **kwargs):
        raise AssertionError("offline no-coverage marker must not touch network")

    monkeypatch.setattr(data.httpx, "stream", fail_stream)
    assert data.fetch_dted(0.5, -159.5, cache_dir=cache, offline=True) is None
    with pytest.raises(data.NoCoverage):
        data.fetch_dted(0.5, -159.5, cache_dir=cache, offline=True, strict=True)
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch_dted(1.5, -159.5, cache_dir=cache, offline=True)


def test_fetch_dted_ignores_stale_no_coverage_marker_online(tmp_path, monkeypatch):
    cache = str(tmp_path)
    path = _terrain_path(cache, 0, -160)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".no_coverage.json", "w") as handle:
        json.dump(
            {
                "source_url": "https://s3.amazonaws.com/old",
                "protocol": "https",
                "status": 404,
                "tile_id": "N00W160",
            },
            handle,
        )
    calls = _stub_http(monkeypatch, [(404, b"")])

    assert data.fetch_dted(0.5, -159.5, cache_dir=cache) is None
    assert len(calls) == 1


def test_fetch_dted_rejects_disallowed_host_before_request(tmp_path, monkeypatch):
    monkeypatch.setattr(
        data,
        "skadi_archive_url",
        lambda lat_index, lon_index: "https://example.invalid/x",
    )

    def fail_stream(*args, **kwargs):
        raise AssertionError("disallowed host must fail before request")

    monkeypatch.setattr(data.httpx, "stream", fail_stream)
    with pytest.raises(data.NetworkError):
        data.fetch_dted(36.5, -106.5, cache_dir=str(tmp_path))


def test_fetch_dted_rejects_oversized_compressed_payload(tmp_path, monkeypatch):
    _stub_http(monkeypatch, [(200, b"abcde")])

    with pytest.raises(data.DownloadSizeExceeded):
        data.fetch_dted(
            36.5,
            -106.5,
            cache_dir=str(tmp_path),
            max_compressed_bytes=4,
        )


def test_legacy_fetch_retains_only_limit_plus_one_from_one_oversized_chunk(
    tmp_path, monkeypatch
):
    _stub_http(monkeypatch, [(200, b"x" * 2_000_000)])
    original = data._append_bounded
    retained_sizes = []

    def tracking_append(buffer, chunk, limit):
        overflow = original(buffer, chunk, limit)
        retained_sizes.append(len(buffer))
        return overflow

    monkeypatch.setattr(data, "_append_bounded", tracking_append)

    with pytest.raises(data.DownloadSizeExceeded):
        data.fetch_dted(
            36.5,
            -106.5,
            cache_dir=str(tmp_path),
            max_compressed_bytes=4,
        )

    assert retained_sizes == [5]


def test_fetch_dted_rejects_wrong_hgt_length(tmp_path, monkeypatch):
    _stub_http(monkeypatch, [(200, gzip.compress(b"short", mtime=0))])

    with pytest.raises(data.DecompressError):
        data.fetch_dted(36.5, -106.5, cache_dir=str(tmp_path))


def test_legacy_gunzip_uses_the_complete_bounded_member_contract():
    first = b"A" * 100_000
    second = b"B" * 90_000
    first_member = gzip.compress(first, mtime=0)
    second_member = gzip.compress(second, mtime=0)
    archive = first_member + second_member

    assert data._gunzip(archive, len(first) + len(second)) == first + second

    with pytest.raises(data.DecompressError, match="exceeded cap"):
        data._gunzip(archive, len(first) + len(second) - 1)

    invalid = [
        first_member + second_member[:-1],
        first_member + bytes([second_member[0] ^ 0x01]) + second_member[1:],
        first_member + b"trailing garbage",
    ]
    for candidate in invalid:
        with pytest.raises(data.DecompressError):
            data._gunzip(candidate, len(first) + len(second))


def test_fetch_dted_conversion_reference_and_terrain_reader(tmp_path, monkeypatch):
    cache = str(tmp_path)
    calls = _stub_http(monkeypatch, [(200, _synthetic_hgt_gz())])

    path = data.fetch_dted(36.5, -106.5, cache_dir=cache)

    assert len(calls) == 1
    assert path == _terrain_path(cache)
    with open(path, "rb") as handle:
        dt2 = handle.read()
    assert len(dt2) == DTED_LEN
    assert hashlib.sha256(dt2).hexdigest() == SYNTHETIC_DTED_SHA256

    with open(path + ".provenance.json") as handle:
        provenance = json.load(handle)
    assert provenance["sha256_data"] == SYNTHETIC_DTED_SHA256
    assert provenance["sha256_dt2"] == SYNTHETIC_DTED_SHA256
    assert provenance["tile_id"] == "N36W107"
    assert provenance["size_dt2"] == DTED_LEN

    terrain = sidereon.DtedTerrain(cache)
    nearest = sidereon.DtedLookupOptions(sidereon.DtedInterpolation.NEAREST_POSTING)
    for lat_posting, lon_posting in [
        (0, 0),
        (100, 200),
        (1234, 2345),
        (2000, 3000),
        (3600, 3600),
    ]:
        lat = 36.0 + lat_posting / 3600.0
        lon = -107.0 + lon_posting / 3600.0
        expected = _expected_posting(lat_posting, lon_posting)
        if expected is None:
            with pytest.raises(sidereon.TerrainError) as excinfo:
                terrain.height_m(lat, lon, nearest)
            assert isinstance(excinfo.value, ValueError)
            detail = excinfo.value.detail
            assert (detail.family, detail.kind) == ("Error", "UnknownTerrainElevation")
            assert detail.details() == {
                "lat_index": 36,
                "lon_index": -107,
                "latitude_posting": lat_posting,
                "longitude_posting": lon_posting,
            }
            assert excinfo.value.point_index is None
            continue
        assert terrain.height_m(lat, lon, nearest) == pytest.approx(expected)


def test_a_tile_converted_with_voids_as_zero_is_reconverted(tmp_path, monkeypatch):
    # hgt_to_dted v1 wrote SRTM voids as 0 m, so its tiles read as sea level at
    # a void. Such a tile is not a cache hit: offline it is a miss, online it
    # is fetched and converted again with voids as the DTED null.
    cache = str(tmp_path)
    path = _seed_terrain(cache)
    with open(path + ".provenance.json") as handle:
        provenance = json.load(handle)
    provenance["converter"] = "sidereon-core hgt_to_dted v1"
    with open(path + ".provenance.json", "w") as handle:
        json.dump(provenance, handle)

    with pytest.raises(data.OfflineCacheMiss, match="voids written as 0 m"):
        data.fetch_dted(36.5, -106.5, cache_dir=cache, offline=True)

    calls = _stub_http(monkeypatch, [(200, _synthetic_hgt_gz())])
    assert data.fetch_dted(36.5, -106.5, cache_dir=cache) == path
    assert len(calls) == 1
    with open(path + ".provenance.json") as handle:
        provenance = json.load(handle)
    assert provenance["converter"] == "sidereon-core hgt_to_dted v2"
    assert provenance["sha256_dt2"] == SYNTHETIC_DTED_SHA256

    # The reconverted tile is a cache hit.
    assert data.fetch_dted(36.5, -106.5, cache_dir=cache, offline=True) == path


def test_prefetch_dted_bbox_and_tiles_report_results(tmp_path, monkeypatch):
    cache = str(tmp_path)
    cached_path = _seed_terrain(cache)
    calls = _stub_http(monkeypatch, [(404, b"")])

    report = data.prefetch_dted_bbox(36.0, -107.0, 36.0, -106.0, cache_dir=cache)

    assert report.cached == [cached_path]
    assert report.fetched == []
    assert report.no_coverage == ["N36W106"]
    assert report.errors == []
    assert len(calls) == 1

    tile_report = data.prefetch_dted_tiles(
        ["N36W107", "bad-tile"], cache_dir=cache, offline=True
    )
    assert tile_report.cached == [cached_path]
    assert tile_report.fetched == []
    assert tile_report.no_coverage == []
    assert len(tile_report.errors) == 1
    assert isinstance(tile_report.errors[0][1], data.InvalidTileId)


def test_populate_terrain_cache_accepts_bbox_mapping(tmp_path, monkeypatch):
    cache = str(tmp_path)
    cached_path = _seed_terrain(cache)

    def fail_stream(*args, **kwargs):
        raise AssertionError("single cached bbox must not touch network")

    monkeypatch.setattr(data.httpx, "stream", fail_stream)
    report = data.populate_terrain_cache(
        {"min_lat": 36.5, "min_lon": -106.5, "max_lat": 36.5, "max_lon": -106.5},
        cache_dir=cache,
        offline=True,
    )
    assert report.cached == [cached_path]
    assert report.fetched == []
    assert report.no_coverage == []
    assert report.errors == []


# --- network tests (excluded by default) ---------------------------------


@pytest.mark.network
def test_network_fetch_code_ultra_day_195(tmp_path):
    product = data.ops_ultra_sp3("cod_ult", dt.date(2026, 7, 14))
    path = data.fetch(product, cache_dir=str(tmp_path))

    assert os.path.getsize(path) == 1_473_962
    with open(path, "rb") as handle:
        assert handle.read(3) == b"#dP"
    sp3 = sidereon.load_sp3(path)
    assert sp3.epoch_count == 289


@pytest.mark.network
def test_network_fetch_ionex_predicted(tmp_path):
    # Predicted maps are published ahead of their target day, so today's
    # cod_prd1 should resolve. The newest-first walk tolerates frontier lag.
    today = dt.datetime.now(dt.timezone.utc).date()
    ionex = data.fetch_ionex("cod_prd1", today, cache_dir=str(tmp_path), lookback=3)
    assert isinstance(ionex, sidereon.Ionex)


@pytest.mark.network
def test_network_fetch_merged_sp3_ultra(tmp_path):
    target = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=12)
    sp3, report = data.fetch_merged_sp3(
        target, ["igs_ult", "gfz_ult"], cache_dir=str(tmp_path)
    )
    assert isinstance(sp3, sidereon.Sp3)
    assert report.source_count >= 1


def test_data_catalog_errors_preserve_exact_core_detail():
    with pytest.raises(data.UnknownCenter) as unknown_center:
        data.product("not-a-center", "ionex", IONEX_DATE)
    assert unknown_center.value.detail == {
        "family": "DataCatalogError",
        "kind": "unknown_center",
        "value": "not-a-center",
        "message": 'unknown analysis center "not-a-center"',
    }

    with pytest.raises(data.UnsupportedProduct) as unsupported:
        data.product("cod_rap", "sp3", IONEX_DATE)
    assert unsupported.value.detail["family"] == "DataCatalogError"
    assert unsupported.value.detail["kind"] == "unsupported_product"
    assert unsupported.value.detail["center"] == "cod_rap"
    assert unsupported.value.detail["product_type"] == "sp3"
    assert unsupported.value.detail["message"] == str(unsupported.value)

    with pytest.raises(data.InvalidCoordinate) as invalid_coordinate:
        data.terrain_tile_index(91.0, 0.0)
    detail = invalid_coordinate.value.detail
    assert detail["family"] == "DataCatalogError"
    assert detail["kind"] == "invalid_coordinate"
    assert detail["lat_deg_bits"] == 4636103972657037312
    assert detail["lon_deg_bits"] == 0
    assert detail["message"] == str(invalid_coordinate.value)

    with pytest.raises(data.InvalidTileIndex) as invalid_index:
        data.skadi_tile_id(91, 0)
    assert invalid_index.value.detail == {
        "family": "DataCatalogError",
        "kind": "invalid_tile_index",
        "lat_index": 91,
        "lon_index": 0,
        "message": "invalid terrain tile index lat=91 lon=0",
    }

    with pytest.raises(data.InvalidTileId) as invalid_id:
        data.parse_skadi_tile_id("bad")
    assert invalid_id.value.detail == {
        "family": "DataCatalogError",
        "kind": "invalid_tile_id",
        "value": "bad",
        "message": 'invalid skadi tile id "bad"',
    }
