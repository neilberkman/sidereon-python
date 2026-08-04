"""Publication-lag resilience surface (core 0.36.0) through the binding.

The cross-line predicted-IONEX walk, the publication-status parsers, and the
newest-published-issue selection are pure and deterministic in
`sidereon-core`; these tests check that the binding marshals them faithfully
against the archive listings recorded live during the 2026-08-04 publication
lag (the same fixtures the core pins).
"""

import datetime as dt
from pathlib import Path

import pytest
from sidereon import data

FIXTURES = Path(__file__).parent / "fixtures" / "listings"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_predicted_ionex_line_candidates_share_the_map_date() -> None:
    map_date = dt.date(2026, 8, 5)  # 2026 day-of-year 217
    candidates = data.predicted_ionex_line_candidates(map_date)
    assert [candidate.center for candidate in candidates] == ["cod_prd1", "cod_prd2"]
    assert all(candidate.date == map_date for candidate in candidates)
    # Same official filename family, distinct lines: the single-line builder
    # agrees with the walk's second candidate.
    two_day = data.predicted_ionex("cod_prd2", map_date - dt.timedelta(days=1))
    assert two_day == candidates[1]


def test_predicted_ionex_line_candidates_cross_the_year_boundary() -> None:
    map_date = dt.date(2027, 1, 1)
    candidates = data.predicted_ionex_line_candidates(map_date)
    assert all(candidate.date == map_date for candidate in candidates)


def test_recorded_p1_gap_resolves_to_p2() -> None:
    objects = data.parse_archive_listing(_fixture("aiub-iono-p1p2-20260804.csv"))

    gap = data.predicted_ionex_line_candidates(dt.date(2026, 8, 5))
    assert data.resolve_first_published(gap, objects) == 1  # P1 absent, P2 present
    assert gap[1].center == "cod_prd2"

    both = data.predicted_ionex_line_candidates(dt.date(2026, 8, 4))
    assert data.resolve_first_published(both, objects) == 0  # P1 preferred


def test_newest_published_product_reports_the_recorded_gfz_lag() -> None:
    objects = data.parse_archive_listing(_fixture("gfz-ultra-w2430-20260804.html"))
    newest = data.newest_published_product("gfz_ult", "sp3", objects)
    assert newest == data.PublishedProduct(
        date=dt.date(2026, 8, 3),
        issue="0300",
        filename="GFZ0OPSULT_20262150300_02D_05M_ORB.SP3",
        observed_at="2026-08-04 08:20",
    )
    age = data.published_issue_age(newest, dt.datetime(2026, 8, 4, 7, 8, 0))
    assert age == dt.timedelta(hours=28, minutes=8)


def test_parse_archive_listing_refuses_unrecognized_bodies() -> None:
    for body in ("", "This mirror has moved.", "<html><h1>503</h1></html>"):
        with pytest.raises(data.DataError):
            data.parse_archive_listing(body)


def test_publication_listing_urls_are_bounded() -> None:
    urls = data.publication_listing_urls("gfz_ult", "sp3", dt.date(2026, 8, 4))
    assert urls == [
        "https://isdc-data.gfz.de/gnss/products/ultra/w2430/",
        "https://isdc-data.gfz.de/gnss/products/ultra/w2429/",
    ]
    aiub = data.publication_listing_urls("cod_prd1", "ionex", dt.date(2026, 8, 4))
    assert aiub == ["https://www.aiub.unibe.ch/download/full_listing.csv"]


def test_wum_nrt_center_is_cataloged_with_its_verified_conventions() -> None:
    assert "wum_nrt" in data.centers()
    entry = data._center_def("wum_nrt")
    assert entry["protocol"] == "ftp"
    assert entry["host"] == "igs.gnsswhu.cn"
    assert entry["products"] == {"sp3"}
    assert len(entry["issues"]) == 24  # hourly rhythm

    product = data.ops_ultra_sp3("wum_nrt", dt.date(2026, 8, 3), issue="0500")
    assert product.canonical_filename() == "WUM0MGXNRT_20262150500_02D_05M_ORB.SP3"
    assert data.product_solution_class("wum_nrt", "sp3") == "near_real_time"

    # The era gate refuses dates before the archive-verified NRT start.
    with pytest.raises(data.DataError):
        data.ops_ultra_sp3("wum_nrt", dt.date(2024, 7, 2), issue="0000")
