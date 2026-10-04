"""Precision-preserving IONEX queries using scale-tagged ClockInstant values."""

from __future__ import annotations

import json
import os
import struct

import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _fixture_grid():
    with open(os.path.join(CORE_FIXTURES, "ionex_golden.json")) as handle:
        golden = json.load(handle)
    path = os.path.join(CORE_FIXTURES, "ionex", golden["ionex_file"]["name"])
    with open(path, "rb") as handle:
        return sidereon.load_ionex(handle.read())


def _bits(value: float) -> bytes:
    return struct.pack("<d", value)


def _query_geometry():
    with open(os.path.join(CORE_FIXTURES, "ionex_golden.json")) as handle:
        case = json.load(handle)["cases"][0]["inputs"]
    return (
        float.fromhex(case["lat_deg"]),
        float.fromhex(case["lon_deg"]),
        float.fromhex(case["az_deg"]),
        float.fromhex(case["el_deg"]),
        float.fromhex(case["frequency_hz"]),
    )


def _record(data: str, label: str) -> str:
    return f"{data:<60}{label:<20}\n"


def _ionex_with_missing_query_node():
    epoch = "  2020     1     1     0     0     0"
    lines = [
        _record("     1.0            IONOSPHERE MAPS     GPS", "IONEX VERSION / TYPE"),
        _record("typed detail default control", "PGM / RUN BY / DATE"),
        _record(epoch, "EPOCH OF FIRST MAP"),
        _record(epoch, "EPOCH OF LAST MAP"),
        _record("  3600", "INTERVAL"),
        _record("     1", "# OF MAPS IN FILE"),
        _record("  COSZ", "MAPPING FUNCTION"),
        _record("     0.0", "ELEVATION CUTOFF"),
        _record("  6371.0", "BASE RADIUS"),
        _record("     2", "MAP DIMENSION"),
        _record("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"),
        _record("     1.0   0.0  -1.0", "LAT1 / LAT2 / DLAT"),
        _record("     0.0   1.0   1.0", "LON1 / LON2 / DLON"),
        _record("    -1", "EXPONENT"),
        _record("", "END OF HEADER"),
        _record("     1", "START OF TEC MAP"),
        _record(epoch, "EPOCH OF CURRENT MAP"),
        _record("     1.0   0.0   1.0   1.0 450.0", "LAT/LON1/LON2/DLON/H"),
        f"{10:5d}{'9999':>5}\n",
        _record("     0.0   0.0   1.0   1.0 450.0", "LAT/LON1/LON2/DLON/H"),
        f"{30:5d}{40:5d}\n",
        _record("     1", "END OF TEC MAP"),
        _record("", "END OF FILE"),
    ]
    return sidereon.load_ionex("".join(lines).encode("utf-8"))


def _instant(scale, whole_seconds: int, fractional_nanoseconds: int = 0):
    return sidereon.ClockInstant.from_nanos(
        scale,
        whole_seconds * 1_000_000_000 + fractional_nanoseconds,
    )


def test_fractional_scalar_query_blends_maps_and_respects_its_scale():
    grid = _fixture_grid()
    lat, lon, azimuth, elevation, frequency = _query_geometry()
    first_map_s = int(grid.map_epochs_j2000_s[0])

    # The fixture maps are two hours apart. UTC and GPST labels below name the
    # same 1800.5-second instant (GPS-UTC is 18 seconds on these map dates).
    utc_half = _instant(sidereon.TimeScale.UTC, first_map_s + 1800, 500_000_000)
    gpst_half = _instant(sidereon.TimeScale.GPST, first_map_s + 1818, 500_000_000)
    utc_whole = _instant(sidereon.TimeScale.UTC, first_map_s + 1800)
    utc_next = _instant(sidereon.TimeScale.UTC, first_map_s + 1801)

    whole_delay = grid.slant_delay_at_instant(
        lat,
        lon,
        azimuth,
        elevation,
        utc_whole,
        frequency,
    )
    fractional_delay = grid.slant_delay_at_instant(
        lat,
        lon,
        azimuth,
        elevation,
        utc_half,
        frequency,
    )
    next_delay = grid.slant_delay_at_instant(
        lat,
        lon,
        azimuth,
        elevation,
        utc_next,
        frequency,
    )
    gpst_delay = sidereon.ionex_slant_delay_at_instant(
        grid,
        lat,
        lon,
        azimuth,
        elevation,
        gpst_half,
        frequency,
    )
    assert whole_delay < fractional_delay < next_delay
    assert _bits(gpst_delay) == _bits(fractional_delay)

    policy_result = grid.slant_delay_at_instant_with_policy(
        lat,
        lon,
        azimuth,
        elevation,
        utc_half,
        frequency,
        sidereon.IonexSlantPolicy.default_policy(),
    )
    assert _bits(policy_result.delay_m) == _bits(fractional_delay)
    assert policy_result.is_valid


def test_fractional_batch_preserves_order_and_typed_per_request_refusals():
    grid = _fixture_grid()
    lat, lon, azimuth, elevation, frequency = _query_geometry()
    epochs = [int(value) for value in grid.map_epochs_j2000_s]
    halfway = _instant(sidereon.TimeScale.UTC, epochs[0] + 1800, 500_000_000)
    after_last = _instant(sidereon.TimeScale.UTC, epochs[-1], 1)

    requests = [
        sidereon.IonexInstantSlantRequest(
            lat,
            lon,
            azimuth,
            elevation,
            halfway,
            frequency,
        ),
        sidereon.IonexInstantSlantRequest(
            lat,
            lon,
            azimuth,
            999.0,
            halfway,
            frequency,
        ),
        sidereon.IonexInstantSlantRequest(
            lat,
            lon,
            azimuth,
            elevation,
            after_last,
            frequency,
        ),
    ]
    results = grid.slant_delays_at_instants_batch_results(requests)
    top_level_results = sidereon.ionex_slant_delay_results_at_instants(grid, requests)
    assert len(results) == len(requests)
    assert [result.is_ok for result in top_level_results] == [
        result.is_ok for result in results
    ]
    assert results[0].is_ok
    assert _bits(results[0].delay_m) == _bits(
        grid.slant_delay_at_instant(lat, lon, azimuth, elevation, halfway, frequency)
    )
    assert not results[1].is_ok
    assert results[1].refusal.kind == "INVALID_INPUT"
    assert not results[2].is_ok
    assert results[2].refusal.kind == "COVERAGE"
    assert results[2].refusal.coverage_error == (
        sidereon.IonexCoverageError.EPOCH_AFTER_LAST_MAP
    )

    with pytest.raises(ValueError, match="out of coverage"):
        grid.slant_delay_at_instant(
            lat,
            lon,
            azimuth,
            elevation,
            after_last,
            frequency,
        )


def test_instant_selection_and_range_keep_fraction_and_scale_equivalence():
    grid = _fixture_grid()
    lat, lon, azimuth, elevation, frequency = _query_geometry()
    first_map_s = int(grid.map_epochs_j2000_s[0])
    query_utc = _instant(sidereon.TimeScale.UTC, first_map_s + 1800, 500_000_000)
    query_gpst = _instant(sidereon.TimeScale.GPST, first_map_s + 1818, 500_000_000)

    selected = sidereon.select_ionex_at_instant([grid], query_gpst)
    assert selected.metadata.kind == sidereon.DegradationKind.EXACT
    assert selected.metadata.requested_epoch_j2000_s == first_map_s + 1800.5
    selected_delay = selected.slant_delay_at_instant(
        lat,
        lon,
        azimuth,
        elevation,
        query_gpst,
        frequency,
    )
    direct_delay = grid.slant_delay_at_instant(
        lat,
        lon,
        azimuth,
        elevation,
        query_utc,
        frequency,
    )
    assert _bits(selected_delay) == _bits(direct_delay)

    start_utc = _instant(sidereon.TimeScale.UTC, first_map_s + 1800, 250_000_000)
    end_gpst = _instant(sidereon.TimeScale.GPST, first_map_s + 1818, 750_000_000)
    selected_range = sidereon.select_ionex_over_instant_range(
        [grid],
        start_utc,
        end_gpst,
    )
    assert selected_range.metadata.kind == sidereon.DegradationKind.EXACT
    assert selected_range.metadata.requested_epoch_j2000_s == first_map_s + 1800.75
    range_delay = selected_range.slant_delay_at_instant(
        lat,
        lon,
        azimuth,
        elevation,
        query_utc,
        frequency,
    )
    assert _bits(range_delay) == _bits(direct_delay)


def _assert_epoch_detail(detail, kind, scale, utc_j2000_s=None):
    assert isinstance(detail, sidereon.IonexEpochErrorDetail)
    assert detail.kind == kind
    assert detail.scale == scale
    assert detail.utc_j2000_s == utc_j2000_s


def test_selection_detail_is_additive_and_keeps_legacy_instant_ionex_cause():
    with pytest.raises(sidereon.SolveError) as hand_solve:
        raise sidereon.SolveError("hand-built solve error")
    assert hand_solve.value.detail is None

    with pytest.raises(sidereon.SelectionError) as hand_selection:
        raise sidereon.SelectionError("hand-built selection error")
    assert hand_selection.value.detail is None

    ionex = _ionex_with_missing_query_node()
    epoch = int(ionex.map_epochs_j2000_s[0])
    with pytest.raises(
        sidereon.SolveError, match="IONEX nodes not available"
    ) as solve_error:
        ionex.slant_delay_at_instant(
            0.5,
            0.5,
            0.0,
            90.0,
            _instant(sidereon.TimeScale.UTC, epoch),
            1_575_420_000.0,
        )
    assert solve_error.value.detail is None

    with pytest.raises(
        sidereon.SelectionError, match="product set is empty"
    ) as selection_error:
        sidereon.select_ionex_at_instant([], _instant(sidereon.TimeScale.UTC, epoch))
    assert selection_error.value.detail is None
    assert selection_error.value.selection_detail == {
        "family": "SelectionError",
        "kind": "empty_product_set",
        "message": "product set is empty",
    }


def test_instant_scalar_policy_batch_and_selected_query_keep_epoch_causes():
    grid = _fixture_grid()
    lat, lon, azimuth, elevation, frequency = _query_geometry()
    first_map_s = int(grid.map_epochs_j2000_s[0])
    tdb = sidereon.ClockInstant.from_nanos(
        sidereon.TimeScale.TDB,
        first_map_s * 1_000_000_000,
    )
    utc = _instant(sidereon.TimeScale.UTC, first_map_s)
    gpst = _instant(sidereon.TimeScale.GPST, first_map_s + 18)
    out_of_range_utc = sidereon.ClockInstant.from_nanos(
        sidereon.TimeScale.UTC,
        2**127 - 1,
    )

    with pytest.raises(sidereon.SolveError) as scalar_error:
        grid.slant_delay_at_instant(lat, lon, azimuth, elevation, tdb, frequency)
    _assert_epoch_detail(
        scalar_error.value.detail,
        "no_exact_utc_offset",
        sidereon.TimeScale.TDB,
    )
    assert str(scalar_error.value) == (
        "invalid input: IONEX map epoch in TDB has no exact offset to UTC"
    )

    with pytest.raises(sidereon.SolveError) as policy_error:
        grid.slant_delay_at_instant_with_policy(
            lat,
            lon,
            azimuth,
            elevation,
            tdb,
            frequency,
            sidereon.IonexSlantPolicy.default_policy(),
        )
    _assert_epoch_detail(
        policy_error.value.detail,
        "no_exact_utc_offset",
        sidereon.TimeScale.TDB,
    )
    assert str(policy_error.value) == str(scalar_error.value)

    requests = [
        sidereon.IonexInstantSlantRequest(lat, lon, azimuth, elevation, utc, frequency),
        sidereon.IonexInstantSlantRequest(lat, lon, azimuth, elevation, tdb, frequency),
        sidereon.IonexInstantSlantRequest(
            lat, lon, azimuth, elevation, out_of_range_utc, frequency
        ),
        sidereon.IonexInstantSlantRequest(
            lat, lon, azimuth, elevation, gpst, frequency
        ),
    ]
    results = grid.slant_delays_at_instants_batch_results(requests)
    assert [result.is_ok for result in results] == [True, False, False, True]
    _assert_epoch_detail(
        results[1].refusal.epoch_error,
        "no_exact_utc_offset",
        sidereon.TimeScale.TDB,
    )
    _assert_epoch_detail(
        results[2].refusal.epoch_error,
        "out_of_range",
        sidereon.TimeScale.UTC,
    )
    assert results[1].refusal.kind == "EPOCH"
    assert results[2].refusal.kind == "EPOCH"
    assert results[1].refusal.message == str(scalar_error.value)
    assert results[2].refusal.message == (
        "invalid input: IONEX map epoch in UTC is outside the i64 J2000 seconds"
    )

    selected = sidereon.select_ionex_at_instant([grid], utc)
    with pytest.raises(sidereon.SolveError) as selected_error:
        selected.slant_delay_at_instant(
            lat,
            lon,
            azimuth,
            elevation,
            tdb,
            frequency,
        )
    _assert_epoch_detail(
        selected_error.value.detail,
        "no_exact_utc_offset",
        sidereon.TimeScale.TDB,
    )
    assert str(selected_error.value) == str(scalar_error.value)


def test_instant_selectors_keep_epoch_scale_and_range_causes():
    grid = _fixture_grid()
    first_map_s = int(grid.map_epochs_j2000_s[0])
    valid_utc = _instant(sidereon.TimeScale.UTC, first_map_s)
    tdb = sidereon.ClockInstant.from_nanos(
        sidereon.TimeScale.TDB,
        first_map_s * 1_000_000_000,
    )
    out_of_range_utc = sidereon.ClockInstant.from_nanos(
        sidereon.TimeScale.UTC,
        2**127 - 1,
    )

    with pytest.raises(sidereon.SelectionError) as instant_error:
        sidereon.select_ionex_at_instant([grid], tdb)
    _assert_epoch_detail(
        instant_error.value.detail,
        "no_exact_utc_offset",
        sidereon.TimeScale.TDB,
    )
    assert instant_error.value.selection_detail["family"] == "SelectionError"
    assert instant_error.value.selection_detail["kind"] == "ionex_epoch"
    assert (
        instant_error.value.selection_detail["cause"]["kind"] == "no_exact_utc_offset"
    )
    assert instant_error.value.selection_detail["cause"]["scale"] == "TDB"
    assert str(instant_error.value) == (
        "IONEX map epoch in TDB has no exact offset to UTC"
    )

    with pytest.raises(sidereon.SelectionError) as range_error:
        sidereon.select_ionex_over_instant_range(
            [grid],
            valid_utc,
            out_of_range_utc,
        )
    _assert_epoch_detail(
        range_error.value.detail,
        "out_of_range",
        sidereon.TimeScale.UTC,
    )
    assert str(range_error.value) == (
        "IONEX map epoch in UTC is outside the i64 J2000 seconds"
    )
