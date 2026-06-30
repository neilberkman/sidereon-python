"""Tests for the optional GNSS data-provisioning layer (``sidereon.data``).

The offline tests always run with no network: they seed a temporary cache_dir
with real core fixtures under their canonical IGS long-names, then exercise the
cache-first fetch, parse, and merge paths plus the typed-error taxonomy. The
URL/filename builders are unit-tested against the exact strings the Elixir
``Sidereon.GNSS.Data`` reference documents.

The network tests (``@pytest.mark.network``) hit a live archive and are excluded
by default; run them with ``pytest -m network``.
"""

import datetime as dt
import hashlib
import json
import os
import shutil

import pytest
import sidereon
import sidereon.data as data
from _helpers import CORE_FIXTURES

# --- fixtures ------------------------------------------------------------


def _core_sp3(name):
    return os.path.join(CORE_FIXTURES, "sp3", name)


def _core_ionex(name):
    return os.path.join(CORE_FIXTURES, "ionex", name)


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
