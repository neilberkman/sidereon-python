"""Tests for the optional GNSS data-provisioning layer (``sidereon.data``).

The offline tests always run with no network: they seed a temporary cache_dir
with real core fixtures under their canonical IGS long-names, then exercise the
cache-first fetch, parse, and merge paths plus the typed-error taxonomy. The
URL/filename builders are unit-tested against the exact core-derived strings.

The network tests (``@pytest.mark.network``) hit a live archive and are excluded
by default; run them with ``pytest -m network``.
"""

import datetime as dt
import functools
import gzip
import hashlib
import json
import os
import shutil

import numpy as np
import pytest
import sidereon
import sidereon.data as data
from _helpers import CORE_FIXTURES

# --- fixtures ------------------------------------------------------------


def _core_sp3(name):
    return os.path.join(CORE_FIXTURES, "sp3", name)


def _core_ionex(name):
    return os.path.join(CORE_FIXTURES, "ionex", name)


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
    def __init__(self, status_code, body=b""):
        self.status_code = status_code
        self._body = body
        self.closed = False

    def iter_bytes(self):
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
        status, body = queue.pop(0)
        return _StubStream(_StubResponse(status, body))

    monkeypatch.setattr(data.httpx, "stream", stream)
    return calls


POSTINGS = 3601
HGT_LEN = POSTINGS * POSTINGS * 2
DTED_LEN = 25_981_042
SYNTHETIC_DTED_SHA256 = (
    "708c193f768f3d859b71da20a81059db4e6077494481c4484d0bc238af096d77"
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
    sample = _synthetic_hgt_sample(POSTINGS - 1 - lat_posting, lon_posting)
    return 0 if sample == -32768 else sample


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
SP3_DATE = dt.date(2020, 6, 25)


# --- builder / URL unit tests (no network) -------------------------------


def test_predicted_ionex_filenames_match_reference():
    p1 = data.predicted_ionex("cod_prd1", IONEX_DATE)
    p2 = data.predicted_ionex("cod_prd2", IONEX_DATE)
    assert p1.canonical_filename() == "COD0OPSPRD_20261650000_01D_01H_GIM.INX"
    assert p2.canonical_filename() == "COD0OPSPRD_20261660000_01D_01H_GIM.INX"


def test_predicted_ionex_urls_use_aiub_code_root_with_gz():
    p1 = data.predicted_ionex("cod_prd1", IONEX_DATE)
    assert p1.archive_url() == (
        "http://ftp.aiub.unibe.ch/CODE/COD0OPSPRD_20261650000_01D_01H_GIM.INX.gz"
    )


def test_rapid_ionex_filename_and_url_match_reference():
    p = data.rapid_ionex(dt.date(2026, 6, 13))
    assert p.canonical_filename() == "COD0OPSRAP_20261640000_01D_01H_GIM.INX"
    assert p.archive_url() == (
        "http://ftp.aiub.unibe.ch/CODE/COD0OPSRAP_20261640000_01D_01H_GIM.INX.gz"
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
        == "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    )


def test_sp3_center_urls_match_reference():
    assert data.archive_url("gfz", "sp3", dt.date(2020, 6, 24), "15M") == (
        "https://isdc-data.gfz.de/gnss/products/rapid/w2111/"
        "GFZ0OPSRAP_20201760000_01D_15M_ORB.SP3.gz"
    )
    # CODE ultra-rapid SP3 is served uncompressed on the AIUB /CODE root.
    assert data.archive_url(
        "cod_ult", "sp3", dt.date(2026, 6, 11), "05M", issue="0000"
    ) == ("http://ftp.aiub.unibe.ch/CODE/COD0OPSULT_20261620000_01D_05M_ORB.SP3")
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
    # Seed the newest candidate day (offset 0 for cod_prd1) with a real IONEX.
    prod = data.predicted_ionex("cod_prd1", IONEX_DATE)
    _seed(cache, prod, _core_ionex("synthetic_2map_7x7.20i"))

    ionex = data.fetch_ionex("cod_prd1", IONEX_DATE, offline=True, cache_dir=cache)
    assert isinstance(ionex, sidereon.Ionex)


def test_fetch_ionex_offline_walks_back_to_older_cached_day(tmp_path):
    cache = str(tmp_path)
    # Only the day-before is cached; the newest candidate is absent. The
    # newest-first walk must fall back to it.
    older = data.predicted_ionex("cod_prd1", IONEX_DATE - dt.timedelta(days=1))
    _seed(cache, older, _core_ionex("synthetic_2map_7x7.20i"))

    ionex = data.fetch_ionex("cod_prd1", IONEX_DATE, offline=True, cache_dir=cache)
    assert isinstance(ionex, sidereon.Ionex)


def test_fetch_ionex_offline_empty_cache_raises_offline_miss(tmp_path):
    with pytest.raises(data.OfflineCacheMiss):
        data.fetch_ionex("cod_prd1", IONEX_DATE, offline=True, cache_dir=str(tmp_path))


# --- offline merged SP3 --------------------------------------------------


def test_fetch_merged_sp3_offline_single_contributor(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_sp3("cod", SP3_DATE)
    _seed(cache, prod, _core_sp3("COD0MGXFIN_20201770000_01D_05M_ORB.SP3"))

    sp3, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod"], offline=True, cache_dir=cache
    )
    assert isinstance(sp3, sidereon.Sp3)
    assert report.single_product is True
    assert report.source_count == 1
    assert report.merged is False
    assert [c.center for c in report.contributors] == ["cod"]


def test_fetch_merged_sp3_offline_records_absent_centers(tmp_path):
    cache = str(tmp_path)
    prod = data.mgex_sp3("cod", SP3_DATE)
    _seed(cache, prod, _core_sp3("COD0MGXFIN_20201770000_01D_05M_ORB.SP3"))

    # esa is requested but not cached: it must be reported absent, not abort.
    sp3, report = data.fetch_merged_sp3(
        SP3_DATE, ["cod", "esa"], offline=True, cache_dir=cache
    )
    assert isinstance(sp3, sidereon.Sp3)
    assert report.source_count == 1
    assert [a.center for a in report.absent] == ["esa"]
    assert report.absent[0].reason == "offline_miss"


def test_fetch_merged_sp3_offline_empty_cache_raises_no_products(tmp_path):
    with pytest.raises(data.NoProducts) as excinfo:
        data.fetch_merged_sp3(SP3_DATE, ["cod"], offline=True, cache_dir=str(tmp_path))
    reasons = excinfo.value.reasons
    assert [r.center for r in reasons] == ["cod"]
    assert reasons[0].reason == "offline_miss"


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
    _seed(cache, prod, _core_sp3("COD0MGXFIN_20201770000_01D_05M_ORB.SP3"))

    out = str(tmp_path / "merged.sp3")
    written = data.fetch_merged_sp3_file(
        SP3_DATE, ["cod"], out, offline=True, cache_dir=cache
    )
    assert written == out
    assert os.path.exists(out)
    # The written file is a standard SP3 the loader round-trips.
    reloaded = sidereon.load_sp3(out)
    assert isinstance(reloaded, sidereon.Sp3)


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


def test_fetch_dted_rejects_wrong_hgt_length(tmp_path, monkeypatch):
    _stub_http(monkeypatch, [(200, gzip.compress(b"short", mtime=0))])

    with pytest.raises(data.DecompressError):
        data.fetch_dted(36.5, -106.5, cache_dir=str(tmp_path))


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
        assert terrain.height_m(lat, lon, nearest) == pytest.approx(
            _expected_posting(lat_posting, lon_posting)
        )


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
