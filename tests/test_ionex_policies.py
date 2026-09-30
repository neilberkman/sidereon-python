"""Policy-aware and settled IONEX tests for Python bindings.

Tests policy-aware slant evaluation, validity masks, header metadata,
structured parser warnings, precision-loss rejections, and batch evaluation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sidereon


def _ionex_record(data: str, label: str) -> str:
    """Standard 80-column IONEX record."""
    return f"{data:<60}{label:<20}\n"


def _i5(val: int | str) -> str:
    """Format single value into exact 5-character I5 fixed-width field."""
    if isinstance(val, int):
        return f"{val:5d}"
    return f"{val:>5}"


def _i5_row(*vals: int | str) -> str:
    """Format values as exact 5-character fixed-width columns."""
    return "".join(_i5(v) for v in vals)


LAYOUT_LAT = "     1.0   0.0  -1.0"
LAYOUT_LON = "     0.0   1.0   1.0"
LAYOUT_EPOCH_0 = "  2020     1     1     0     0     0"
LAYOUT_EPOCH_1 = "  2020     1     1     1     0     0"


def _layout_header(
    maps: int = 1,
    last_epoch: str = LAYOUT_EPOCH_0,
    extra: str = "",
    mapping: str = "COSZ",
    lat_axis: str = LAYOUT_LAT,
) -> str:
    """A header declaring `mapping` in its one `MAPPING FUNCTION` record.

    The reader refuses a second `MAPPING FUNCTION` record that gives another
    code, so a file declaring anything but `COSZ` passes `mapping` here rather
    than adding its own record through `extra`.
    """
    t = [
        _ionex_record(
            "     1.0            IONOSPHERE MAPS     GPS",
            "IONEX VERSION / TYPE",
        ),
        _ionex_record("synthetic policy test", "PGM / RUN BY / DATE"),
        _ionex_record(LAYOUT_EPOCH_0, "EPOCH OF FIRST MAP"),
        _ionex_record(last_epoch, "EPOCH OF LAST MAP"),
        _ionex_record("  3600", "INTERVAL"),
        _ionex_record(f"{maps:6d}", "# OF MAPS IN FILE"),
        _ionex_record(f"  {mapping}", "MAPPING FUNCTION"),
        _ionex_record("     0.0", "ELEVATION CUTOFF"),
        _ionex_record("", "OBSERVABLES USED"),
        _ionex_record("  6371.0", "BASE RADIUS"),
        _ionex_record("     2", "MAP DIMENSION"),
        _ionex_record("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"),
        _ionex_record(lat_axis, "LAT1 / LAT2 / DLAT"),
        _ionex_record(LAYOUT_LON, "LON1 / LON2 / DLON"),
    ]
    if extra:
        t.append(extra)
    t.append(_ionex_record("", "END OF HEADER"))
    return "".join(t)


def _layout_band(
    lat: float, lon1: float, lon2: float, dlon: float, h: float, values: str
) -> str:
    prefix = _ionex_record(
        f"  {lat:6.1f}{lon1:6.1f}{lon2:6.1f}{dlon:6.1f}{h:6.1f}",
        "LAT/LON1/LON2/DLON/H",
    )
    return f"{prefix}{values}\n"


def _layout_bands(north: str, south: str) -> str:
    return _layout_band(1.0, 0.0, 1.0, 1.0, 450.0, north) + _layout_band(
        0.0, 0.0, 1.0, 1.0, 450.0, south
    )


def _layout_map(kind: str, index: int, epoch: str | None, body: str) -> str:
    t = [_ionex_record(f"{index:6d}", f"START OF {kind} MAP")]
    if epoch is not None:
        t.append(_ionex_record(epoch, "EPOCH OF CURRENT MAP"))
    t.append(body)
    t.append(_ionex_record(f"{index:6d}", f"END OF {kind} MAP"))
    return "".join(t)


def _layout_end() -> str:
    return _ionex_record("", "END OF FILE")


def _exponent_record(exp: int) -> str:
    return _ionex_record(f"{exp:6d}", "EXPONENT")


def _one_map_file(
    extra_header: str,
    body: str,
    mapping: str = "COSZ",
    lat_axis: str = LAYOUT_LAT,
) -> str:
    return (
        _layout_header(1, LAYOUT_EPOCH_0, extra_header, mapping, lat_axis)
        + _layout_map("TEC", 1, LAYOUT_EPOCH_0, body)
        + _layout_end()
    )


def _assert_bitwise_float_equal(got: float, want: float, label: str = "") -> None:
    got_u64 = np.array([got], dtype=np.float64).view(np.uint64)[0]
    want_u64 = np.array([want], dtype=np.float64).view(np.uint64)[0]
    assert got_u64 == want_u64, (
        f"{label} mismatch: {got!r} ({got.hex()}) != {want!r} ({want.hex()})"
    )


def test_missing_cells_map_to_nan_and_false_mask_and_roundtrip():
    # 9999 represents missing/non-available cell in standard IONEX.
    # North band has values [10, 9999], South band has [9999, 40].
    # Exact 5-character columns: "   10 9999" and " 9999   40".
    body = _layout_bands(_i5_row(10, 9999), _i5_row(9999, 40))
    text = _one_map_file(_exponent_record(0), body)
    ionex = sidereon.load_ionex(text.encode("utf-8"))

    tec = ionex.tec_maps
    assert tec.shape == (1, 2, 2)
    assert tec[0, 0, 0] == 10.0
    assert math.isnan(tec[0, 0, 1])
    assert math.isnan(tec[0, 1, 0])
    assert tec[0, 1, 1] == 40.0

    mask = ionex.tec_mask
    assert mask.shape == (1, 2, 2)
    assert mask.dtype == bool
    assert bool(mask[0, 0, 0]) is True
    assert bool(mask[0, 0, 1]) is False
    assert bool(mask[0, 1, 0]) is False
    assert bool(mask[0, 1, 1]) is True

    # tec_valid is documented alias for tec_mask
    np.testing.assert_array_equal(ionex.tec_valid, mask)

    # Re-serialization with 9999 and re-parse preserves NaN and false mask
    reparsed = sidereon.load_ionex(ionex.to_ionex_string().encode("utf-8"))
    np.testing.assert_array_equal(reparsed.tec_mask, mask)
    np.testing.assert_array_equal(reparsed.tec_valid, mask)
    _assert_bitwise_float_equal(reparsed.tec_maps[0, 0, 0], 10.0)
    assert math.isnan(reparsed.tec_maps[0, 0, 1])
    assert math.isnan(reparsed.tec_maps[0, 1, 0])
    _assert_bitwise_float_equal(reparsed.tec_maps[0, 1, 1], 40.0)


def test_tec_rms_height_missing_cells_full_roundtrip():
    # Construct whole-grid samples with missing cells in TEC, RMS, and height
    epochs = np.array([631108800], dtype=np.int64)  # 2020-01-01 00:00:00 UTC
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)

    tec_vals = np.array([[[10.0, 999.0], [999.0, 40.0]]], dtype=np.float64)
    tec_mask = np.array([[[True, False], [False, True]]], dtype=bool)

    rms_vals = np.array([[[1.5, 999.0], [2.5, 3.5]]], dtype=np.float64)
    rms_mask = np.array([[[True, False], [True, True]]], dtype=bool)

    hgt_vals = np.array([[[450.0, 450.0], [999.0, 450.0]]], dtype=np.float64)
    hgt_mask = np.array([[[True, True], [False, True]]], dtype=bool)

    grid = sidereon.TecGridSamples(
        epochs,
        lats,
        lons,
        -1.0,
        1.0,
        450.0,
        6371.0,
        0,
        tec_vals,
        rms_maps=rms_vals,
        height_maps=hgt_vals,
        tec_mask=tec_mask,
        rms_mask=rms_mask,
        height_mask=hgt_mask,
    )
    ionex = sidereon.Ionex.from_samples(grid)

    assert ionex.has_rms
    assert ionex.has_height
    assert math.isnan(ionex.tec_maps[0, 0, 1])
    assert math.isnan(ionex.rms_maps[0, 0, 1])
    assert math.isnan(ionex.height_maps[0, 1, 0])

    _assert_bitwise_float_equal(ionex.tec_maps[0, 0, 0], 10.0)
    _assert_bitwise_float_equal(ionex.rms_maps[0, 0, 0], 1.5)
    _assert_bitwise_float_equal(ionex.height_maps[0, 0, 0], 450.0)

    # Samples roundtrip
    extracted = ionex.tec_grid_samples()
    assert extracted.has_rms
    assert extracted.has_height
    np.testing.assert_array_equal(extracted.tec_mask, tec_mask)
    np.testing.assert_array_equal(extracted.rms_mask, rms_mask)
    np.testing.assert_array_equal(extracted.height_mask, hgt_mask)

    # Serializer roundtrip via to_ionex_string
    text = ionex.to_ionex_string()
    reparsed = sidereon.load_ionex(text.encode("utf-8"))

    assert reparsed.has_rms
    assert reparsed.has_height
    np.testing.assert_array_equal(reparsed.tec_mask, tec_mask)
    np.testing.assert_array_equal(reparsed.rms_mask, rms_mask)
    np.testing.assert_array_equal(reparsed.height_mask, hgt_mask)

    assert math.isnan(reparsed.tec_maps[0, 0, 1])
    assert math.isnan(reparsed.rms_maps[0, 0, 1])
    assert math.isnan(reparsed.height_maps[0, 1, 0])

    _assert_bitwise_float_equal(reparsed.tec_maps[0, 0, 0], 10.0)
    _assert_bitwise_float_equal(reparsed.rms_maps[0, 0, 0], 1.5)
    _assert_bitwise_float_equal(reparsed.height_maps[0, 0, 0], 450.0)


def test_present_map_with_all_false_cells_distinct_from_absent_maps():
    epochs = np.array([631108800], dtype=np.int64)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)
    tec = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)

    # 1. Product with ABSENT RMS and height maps
    grid_absent = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec
    )
    ionex_absent = sidereon.Ionex.from_samples(grid_absent)
    assert not ionex_absent.has_rms
    assert not ionex_absent.has_height
    assert ionex_absent.rms_maps.shape == (0, 0, 0)
    assert ionex_absent.rms_mask.shape == (0, 0, 0)
    assert ionex_absent.height_maps.shape == (0, 0, 0)
    assert ionex_absent.height_mask.shape == (0, 0, 0)

    # 2. Product with PRESENT RMS and height maps, but ALL cells are false/missing
    all_dummy = np.zeros((1, 2, 2), dtype=np.float64)
    all_false_mask = np.zeros((1, 2, 2), dtype=bool)

    grid_present_empty = sidereon.TecGridSamples(
        epochs,
        lats,
        lons,
        -1.0,
        1.0,
        450.0,
        6371.0,
        0,
        tec,
        rms_maps=all_dummy,
        rms_mask=all_false_mask,
        height_maps=all_dummy,
        height_mask=all_false_mask,
    )
    ionex_present_empty = sidereon.Ionex.from_samples(grid_present_empty)
    # Present distinction is preserved
    assert ionex_present_empty.has_rms
    assert ionex_present_empty.has_height
    assert ionex_present_empty.rms_maps.shape == (1, 2, 2)
    assert ionex_present_empty.rms_mask.shape == (1, 2, 2)
    assert ionex_present_empty.height_maps.shape == (1, 2, 2)
    assert ionex_present_empty.height_mask.shape == (1, 2, 2)

    assert np.all(np.isnan(ionex_present_empty.rms_maps))
    assert not np.any(ionex_present_empty.rms_mask)
    assert np.all(np.isnan(ionex_present_empty.height_maps))
    assert not np.any(ionex_present_empty.height_mask)


def test_mask_alias_refusal_and_validation():
    epochs = np.array([0], dtype=np.int64)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)
    vals = np.array([[[12.345, 999.0], [0.0, -5.5]]], dtype=np.float64)
    mask = np.array([[[True, False], [True, True]]], dtype=bool)

    # 1. Reject conflicting simultaneous aliases
    with pytest.raises(ValueError, match="cannot supply both tec_mask and tec_valid"):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            -1,
            vals,
            tec_mask=mask,
            tec_valid=mask,
        )

    # 2. Reject mask supplied when corresponding optional map values are absent
    with pytest.raises(
        ValueError, match="cannot supply rms_mask or rms_valid when rms_maps is absent"
    ):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            -1,
            vals,
            rms_mask=mask,
        )

    with pytest.raises(
        ValueError,
        match="cannot supply height_mask or height_valid when height_maps is absent",
    ):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            -1,
            vals,
            height_valid=mask,
        )

    # 3. Reject zero-dimensional axes
    empty_vals = np.zeros((0, 2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="dimensions must be non-zero"):
        sidereon.TecGridSamples(
            np.array([], dtype=np.int64),
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            -1,
            empty_vals,
        )

    # 4. Reject dimension mismatch without altering numbers
    bad_shape_vals = np.zeros((1, 3, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="does not match grid dimensions"):
        sidereon.TecGridSamples(
            epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, -1, bad_shape_vals
        )

    # 5. False-masked cell maps to NaN and present values preserve bit patterns
    grid = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, -1, vals, tec_mask=mask
    )
    ionex = sidereon.Ionex.from_samples(grid)
    assert math.isnan(ionex.tec_maps[0, 0, 1])
    assert not ionex.tec_mask[0, 0, 1]
    _assert_bitwise_float_equal(ionex.tec_maps[0, 0, 0], 12.345)
    _assert_bitwise_float_equal(ionex.tec_maps[0, 1, 0], 0.0)
    _assert_bitwise_float_equal(ionex.tec_maps[0, 1, 1], -5.5)

    # 6. Wrong-sized mask fails
    bad_mask = np.ones((1, 2, 3), dtype=bool)
    with pytest.raises(ValueError, match="mask shape"):
        sidereon.TecGridSamples(
            epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, -1, vals, tec_mask=bad_mask
        )

    # 7. True-masked NaN fails validation
    nan_vals = np.array([[[float("nan"), 1.0], [2.0, 3.0]]], dtype=np.float64)
    all_true_mask = np.ones((1, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="must be finite"):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            -1,
            nan_vals,
            tec_mask=all_true_mask,
        )

    # 8. True-masked Inf fails validation
    inf_vals = np.array([[[float("inf"), 1.0], [2.0, 3.0]]], dtype=np.float64)
    with pytest.raises(ValueError, match="must be finite"):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            -1,
            inf_vals,
            tec_mask=all_true_mask,
        )

    # 9. Omitted mask assumes all present and rejects NaN
    with pytest.raises(ValueError, match="not finite"):
        sidereon.TecGridSamples(
            epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, -1, nan_vals
        )


def test_full_header_roundtrip_and_mapping_declarations():
    epochs = np.array([631108800], dtype=np.int64)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)
    tec = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)

    # 1. Absent mapping function
    h_absent = sidereon.IonexHeader(mapping_function=None)
    assert h_absent.mapping_function is None
    assert h_absent.mapping_function_code is None
    assert h_absent.mapping_declaration.is_absent
    assert not h_absent.mapping_declaration.is_declared
    assert h_absent.mapping_declaration.kind == "ABSENT"

    g_absent = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=h_absent
    )
    i_absent = sidereon.Ionex.from_samples(g_absent)
    assert i_absent.mapping_function is None
    assert i_absent.mapping_declaration.is_absent
    reparsed_absent = sidereon.load_ionex(i_absent.to_ionex_string().encode("utf-8"))
    assert reparsed_absent.header.mapping_function is None
    assert reparsed_absent.header.mapping_declaration.is_absent
    assert reparsed_absent.mapping_function is None
    assert reparsed_absent.mapping_declaration.is_absent

    # 2. Declared COSZ
    h_cosz = sidereon.IonexHeader(mapping_function="COSZ")
    assert h_cosz.mapping_function is not None
    assert h_cosz.mapping_function.kind == "COSZ"
    assert h_cosz.mapping_function.code == "COSZ"
    assert h_cosz.mapping_declaration.is_declared
    assert h_cosz.mapping_declaration.kind == "DECLARED"

    g_cosz = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=h_cosz
    )
    i_cosz = sidereon.Ionex.from_samples(g_cosz)
    assert i_cosz.mapping_function is not None
    assert i_cosz.mapping_function.kind == "COSZ"
    reparsed_cosz = sidereon.load_ionex(i_cosz.to_ionex_string().encode("utf-8"))
    assert reparsed_cosz.header.mapping_function.kind == "COSZ"
    assert reparsed_cosz.mapping_function.kind == "COSZ"
    assert reparsed_cosz.mapping_declaration.is_declared

    # 3. Declared NONE
    h_none = sidereon.IonexHeader(
        mapping_function=sidereon.IonexMappingFunction.no_mapping()
    )
    assert h_none.mapping_function.kind == "NONE"
    assert h_none.mapping_function.code == "NONE"
    g_none = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=h_none
    )
    i_none = sidereon.Ionex.from_samples(g_none)
    reparsed_none = sidereon.load_ionex(i_none.to_ionex_string().encode("utf-8"))
    assert reparsed_none.header.mapping_function.kind == "NONE"
    assert reparsed_none.mapping_function.kind == "NONE"

    # 4. Declared QFAC
    h_qfac = sidereon.IonexHeader(
        mapping_function=sidereon.IonexMappingFunction.q_factor()
    )
    assert h_qfac.mapping_function.kind == "QFAC"
    assert h_qfac.mapping_function.code == "QFAC"
    g_qfac = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=h_qfac
    )
    i_qfac = sidereon.Ionex.from_samples(g_qfac)
    reparsed_qfac = sidereon.load_ionex(i_qfac.to_ionex_string().encode("utf-8"))
    assert reparsed_qfac.header.mapping_function.kind == "QFAC"
    assert reparsed_qfac.mapping_function.kind == "QFAC"

    # 5. Declared Other with exact custom text
    h_other = sidereon.IonexHeader(
        mapping_function=sidereon.IonexMappingFunction.other("MOD")
    )
    assert h_other.mapping_function.kind == "OTHER"
    assert h_other.mapping_function.code == "MOD"
    assert h_other.mapping_declaration.code == "MOD"
    g_other = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=h_other
    )
    i_other = sidereon.Ionex.from_samples(g_other)
    reparsed_other = sidereon.load_ionex(i_other.to_ionex_string().encode("utf-8"))
    assert reparsed_other.header.mapping_function.kind == "OTHER"
    assert reparsed_other.header.mapping_function.code == "MOD"
    assert reparsed_other.mapping_function.kind == "OTHER"

    # 6. All header fields read and write correctly
    hdr = sidereon.IonexHeader(
        version=1.1,
        satellite_system="GPS",
        program="SIDEREON",
        run_by="TEST",
        date="2026-09-22",
        descriptions=["Test Desc"],
        comments=["Comment 1"],
        interval_s=3600,
        elevation_cutoff_deg=10.0,
        observables_used="L1+L2",
        station_count=42,
        satellite_count=32,
        maps_in_file=1,
    )
    assert hdr.version == 1.1
    assert hdr.satellite_system == "GPS"
    assert hdr.program == "SIDEREON"
    assert hdr.run_by == "TEST"
    assert hdr.date == "2026-09-22"
    assert hdr.descriptions == ["Test Desc"]
    assert hdr.comments == ["Comment 1"]
    assert hdr.interval_s == 3600
    assert hdr.elevation_cutoff_deg == 10.0
    assert hdr.observables_used == "L1+L2"
    assert hdr.station_count == 42
    assert hdr.satellite_count == 32
    assert hdr.maps_in_file == 1


def test_default_strict_refusal_for_missing_node():
    # Map has node [1.0, 1.0] missing. Exact I5 formatting: "   10 9999", "   30   40"
    body = _layout_bands(_i5_row(10, 9999), _i5_row(30, 40))
    text = _one_map_file(_exponent_record(0), body)
    ionex = sidereon.load_ionex(text.encode("utf-8"))
    epoch0 = int(ionex.map_epochs_j2000_s[0])

    default_policy = sidereon.IonexSlantPolicy()
    assert default_policy.coverage == sidereon.IonexCoveragePolicy.STRICT
    assert default_policy.missing_nodes == sidereon.IonexMissingNodePolicy.STRICT
    assert default_policy.mapping == sidereon.IonexMappingPolicy.SINGLE_LAYER

    # Scalar slant_delay raises sidereon.SolveError on missing nodes
    with pytest.raises(sidereon.SolveError, match="IONEX nodes not available"):
        ionex.slant_delay(0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0)

    # Top-level ionex_slant_delay raises sidereon.SolveError as well
    with pytest.raises(sidereon.SolveError, match="IONEX nodes not available"):
        sidereon.ionex_slant_delay(ionex, 0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0)

    # Batch evaluation reports refusal with MISSING_NODES kind and node_gap payload
    req = sidereon.IonexSlantRequest(0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0)
    batch_results = ionex.slant_delays_batch_results([req], default_policy)
    assert len(batch_results) == 1
    res = batch_results[0]
    assert not res.is_ok
    assert res.evaluation is None
    refusal = res.refusal
    assert refusal is not None
    assert refusal.kind == "MISSING_NODES"
    gap = refusal.node_gap
    assert gap is not None
    assert gap.earlier is not None
    assert gap.earlier.map_number == 1
    assert gap.earlier.lat_index == 0
    assert gap.earlier.lon_index == 0
    assert gap.earlier.lon_index_next == 1
    assert gap.earlier.missing == (False, True, False, False)
    assert gap.later is None


def test_two_time_bracket_missing_nodes():
    # Map 1: missing node at [1.0, 1.0] (north band, lon 1)
    bands_1 = _layout_bands(_i5_row(10, 9999), _i5_row(30, 40))
    # Map 2: missing node at [0.0, 0.0] (south band, lon 0)
    bands_2 = _layout_bands(_i5_row(10, 20), _i5_row(9999, 40))

    header = _layout_header(2, LAYOUT_EPOCH_1, _exponent_record(0))
    text = (
        header
        + _layout_map("TEC", 1, LAYOUT_EPOCH_0, bands_1)
        + _layout_map("TEC", 2, LAYOUT_EPOCH_1, bands_2)
        + _layout_end()
    )
    ionex = sidereon.load_ionex(text.encode("utf-8"))
    assert len(ionex.map_epochs_j2000_s) == 2
    epoch0 = int(ionex.map_epochs_j2000_s[0])
    epoch1 = int(ionex.map_epochs_j2000_s[1])
    mid_epoch = (epoch0 + epoch1) // 2

    # Query between maps under strict policy -> both brackets are weighted and missing
    req = sidereon.IonexSlantRequest(0.5, 0.5, 0.0, 90.0, mid_epoch, 1575420000.0)
    strict_results = ionex.slant_delays_batch_results([req])
    assert not strict_results[0].is_ok
    refusal = strict_results[0].refusal
    assert refusal.kind == "MISSING_NODES"
    gap = refusal.node_gap
    assert gap is not None
    # Both earlier and later records are retained
    assert gap.earlier is not None
    assert gap.earlier.map_number == 1
    assert gap.earlier.missing == (False, True, False, False)
    assert gap.earlier.missing_01 is True
    assert gap.earlier.missing_00 is False

    assert gap.later is not None
    assert gap.later.map_number == 2
    assert gap.later.missing == (False, False, True, False)
    assert gap.later.missing_10 is True
    assert gap.later.missing_11 is False

    # Under renormalize policy, interpolation succeeds with degraded status
    renorm_policy = sidereon.IonexSlantPolicy().with_missing_nodes(
        sidereon.IonexMissingNodePolicy.RENORMALIZE
    )
    eval_res = ionex.slant_delay_with_policy(
        0.5, 0.5, 0.0, 90.0, mid_epoch, 1575420000.0, policy=renorm_policy
    )
    assert eval_res.delay_m > 0.0
    assert eval_res.status.is_degraded
    degraded = eval_res.status.degraded
    assert degraded.earlier is not None and degraded.earlier.map_number == 1
    assert degraded.later is not None and degraded.later.map_number == 2


def test_explicit_renormalization_retains_node_indices_and_missing_flags():
    body = _layout_bands(_i5_row(10, 9999), _i5_row(30, 40))
    text = _one_map_file(_exponent_record(0), body)
    ionex = sidereon.load_ionex(text.encode("utf-8"))
    epoch0 = int(ionex.map_epochs_j2000_s[0])

    renorm_policy = sidereon.IonexSlantPolicy().with_missing_nodes(
        sidereon.IonexMissingNodePolicy.RENORMALIZE
    )
    assert renorm_policy.missing_nodes == sidereon.IonexMissingNodePolicy.RENORMALIZE

    eval_result = ionex.slant_delay_with_policy(
        0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0, policy=renorm_policy
    )
    assert eval_result.delay_m > 0.0
    status = eval_result.status
    assert not status.is_valid
    assert status.is_degraded
    assert status.degraded is not None

    gap = status.degraded
    assert gap.earlier is not None
    nodes = gap.earlier
    assert nodes.map_number == 1
    assert nodes.lat_index == 0
    assert nodes.lon_index == 0
    assert nodes.lon_index_next == 1
    assert nodes.missing == (False, True, False, False)
    assert nodes.missing_01 is True
    assert nodes.missing_00 is False


def test_single_result_retains_held_degraded_and_assumed_mapping():
    body = _layout_bands(_i5_row(10, 9999), _i5_row(30, 40))
    text = _one_map_file(_exponent_record(0), body, mapping="NONE")
    ionex = sidereon.load_ionex(text.encode("utf-8"))
    epoch0 = int(ionex.map_epochs_j2000_s[0])

    multi_policy = (
        sidereon.IonexSlantPolicy()
        .with_coverage(sidereon.IonexCoveragePolicy.HOLD)
        .with_missing_nodes(sidereon.IonexMissingNodePolicy.RENORMALIZE)
        .with_mapping(sidereon.IonexMappingPolicy.SINGLE_LAYER)
    )

    # Query with epoch after map epoch -> Held
    eval_result = ionex.slant_delay_with_policy(
        0.5, 0.5, 0.0, 90.0, epoch0 + 3600, 1575420000.0, policy=multi_policy
    )
    status = eval_result.status

    # All three fields are retained together
    assert status.held is not None
    assert status.is_held
    assert status.held == sidereon.IonexCoverageError.EPOCH_AFTER_LAST_MAP

    assert status.degraded is not None
    assert status.is_degraded
    assert status.degraded.earlier is not None

    assert status.assumed_mapping is not None
    assert status.is_assumed_mapping
    assert status.has_assumed_mapping
    assert not status.is_nominal
    assert status.assumed_mapping == sidereon.IonexAssumedMapping.NO_MAPPING

    # Not valid because held and degraded are set
    assert not status.is_valid
    assert not eval_result.is_valid
    assert not eval_result.is_nominal
    assert eval_result.held == status.held
    assert eval_result.degraded == status.degraded
    assert eval_result.assumed_mapping == status.assumed_mapping


def test_assumed_mapping_alone_is_valid():
    body = _layout_bands(_i5_row(10, 20), _i5_row(30, 40))
    text = _one_map_file(_exponent_record(0), body, mapping="NONE")
    ionex = sidereon.load_ionex(text.encode("utf-8"))
    epoch0 = int(ionex.map_epochs_j2000_s[0])

    eval_result = ionex.slant_delay_with_policy(
        0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0
    )
    status = eval_result.status
    assert status.held is None
    assert status.degraded is None
    assert status.assumed_mapping == sidereon.IonexAssumedMapping.NO_MAPPING

    # Assumed mapping alone does NOT invalidate nominal coverage/availability
    assert status.is_valid
    assert eval_result.is_valid
    assert not status.is_nominal
    assert not eval_result.is_nominal
    assert status.is_assumed_mapping
    assert status.has_assumed_mapping
    assert eval_result.held is None
    assert eval_result.degraded is None
    assert eval_result.assumed_mapping == sidereon.IonexAssumedMapping.NO_MAPPING


def test_strict_declared_unsupported_mapping_refuses():
    body = _layout_bands(_i5_row(10, 20), _i5_row(30, 40))
    text = _one_map_file(_exponent_record(0), body, mapping="NONE")
    ionex = sidereon.load_ionex(text.encode("utf-8"))
    epoch0 = int(ionex.map_epochs_j2000_s[0])

    declared_policy = sidereon.IonexSlantPolicy().with_mapping(
        sidereon.IonexMappingPolicy.DECLARED
    )
    with pytest.raises(sidereon.SolveError, match="mapping|NONE"):
        ionex.slant_delay_with_policy(
            0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0, policy=declared_policy
        )

    # Batch result gives exact typed refusal
    req = sidereon.IonexSlantRequest(0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0)
    results = ionex.slant_delays_batch_results([req], policy=declared_policy)
    assert not results[0].is_ok
    refusal = results[0].refusal
    assert refusal is not None
    assert refusal.kind == "MAPPING_FUNCTION"
    assert refusal.mapping_declaration is not None
    assert refusal.mapping_declaration.is_declared
    assert not refusal.mapping_declaration.is_absent
    assert refusal.mapping_declaration.kind == "DECLARED"
    assert refusal.mapping_declaration.code == "NONE"
    assert (
        refusal.mapping_declaration.function
        == sidereon.IonexMappingFunction.no_mapping()
    )
    assert "NONE" in refusal.message or "mapping" in refusal.message


def test_height_map_refusals():
    epochs = np.array([631108800], dtype=np.int64)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)
    tec = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)

    # 1. Varying height map (e.g. 450 vs 500)
    varying_heights = np.array([[[450.0, 500.0], [450.0, 450.0]]], dtype=np.float64)
    grid_var = sidereon.TecGridSamples(
        epochs,
        lats,
        lons,
        -1.0,
        1.0,
        450.0,
        6371.0,
        0,
        tec,
        height_maps=varying_heights,
    )
    ionex_var = sidereon.Ionex.from_samples(grid_var)
    assert ionex_var.has_height

    req = sidereon.IonexSlantRequest(0.5, 0.5, 0.0, 45.0, 631108800, 1575420000.0)
    res_var = ionex_var.slant_delays_batch_results([req])
    assert not res_var[0].is_ok
    assert res_var[0].refusal.kind == "VARYING_HEIGHTS"
    assert res_var[0].refusal.map_number == 1
    assert res_var[0].refusal.lat_index == 0
    assert res_var[0].refusal.lon_index == 1

    # 2. Unavailable height map cell (missing node in height map)
    h_missing = np.array([[[450.0, 450.0], [450.0, float("nan")]]], dtype=np.float64)
    h_mask = np.array([[[True, True], [True, False]]], dtype=bool)
    grid_unavail = sidereon.TecGridSamples(
        epochs,
        lats,
        lons,
        -1.0,
        1.0,
        450.0,
        6371.0,
        0,
        tec,
        height_maps=h_missing,
        height_mask=h_mask,
    )
    ionex_unavail = sidereon.Ionex.from_samples(grid_unavail)
    res_unavail = ionex_unavail.slant_delays_batch_results([req])
    assert not res_unavail[0].is_ok
    assert res_unavail[0].refusal.kind == "HEIGHT_NOT_AVAILABLE"
    assert res_unavail[0].refusal.map_number == 1
    assert res_unavail[0].refusal.lat_index == 1
    assert res_unavail[0].refusal.lon_index == 1


def test_all_seven_warning_variants():
    bands = _layout_bands(_i5_row(1, 2), _i5_row(3, 4))
    clean = (
        _layout_header(2, LAYOUT_EPOCH_1, _exponent_record(0))
        + _layout_map("TEC", 1, LAYOUT_EPOCH_0, bands)
        + _layout_map("TEC", 2, LAYOUT_EPOCH_1, bands)
        + _layout_end()
    )

    # 1. MissingRecord
    no_end = clean.replace(_layout_end(), "")
    res = sidereon.load_ionex_with_warnings(no_end.encode("utf-8"))
    eof_warn = next(w for w in res.warnings if w.kind == "MISSING_RECORD")
    assert eof_warn.label == "END OF FILE"

    # 2. VersionRecordNotFirst
    comment_first = _ionex_record("a comment first", "COMMENT") + clean
    res = sidereon.load_ionex_with_warnings(comment_first.encode("utf-8"))
    ver_warn = next(w for w in res.warnings if w.kind == "VERSION_RECORD_NOT_FIRST")
    assert ver_warn.line == 2

    # 3. EpochMismatch (precision-preserving diagnostic epoch)
    wrong_last = (
        _layout_header(2, LAYOUT_EPOCH_0, _exponent_record(0))
        + _layout_map("TEC", 1, LAYOUT_EPOCH_0, bands)
        + _layout_map("TEC", 2, LAYOUT_EPOCH_1, bands)
        + _layout_end()
    )
    res = sidereon.load_ionex_with_warnings(wrong_last.encode("utf-8"))
    epoch_warn = next(w for w in res.warnings if w.kind == "EPOCH_MISMATCH")
    assert epoch_warn.label == "EPOCH OF LAST MAP"
    assert epoch_warn.line == 4
    assert epoch_warn.declared_epoch is not None
    assert epoch_warn.maps_epoch is not None
    assert epoch_warn.declared_epoch.scale == "UTC"
    assert epoch_warn.declared_epoch.jd_whole == 2458849.0
    assert epoch_warn.declared_epoch.fraction == 0.5
    assert epoch_warn.declared_epoch.j2000_seconds == 631108800
    assert epoch_warn.maps_epoch.scale == "UTC"
    assert epoch_warn.maps_epoch.jd_whole == 2458849.0
    assert abs(epoch_warn.maps_epoch.fraction - (0.5 + 1.0 / 24.0)) < 1e-12
    assert epoch_warn.maps_epoch.j2000_seconds == 631112400

    # 4. MapCountMismatch
    wrong_count = (
        _layout_header(3, LAYOUT_EPOCH_1, _exponent_record(0))
        + _layout_map("TEC", 1, LAYOUT_EPOCH_0, bands)
        + _layout_map("TEC", 2, LAYOUT_EPOCH_1, bands)
        + _layout_end()
    )
    res = sidereon.load_ionex_with_warnings(wrong_count.encode("utf-8"))
    count_warn = next(w for w in res.warnings if w.kind == "MAP_COUNT_MISMATCH")
    assert count_warn.line == 6
    assert count_warn.declared_count == 3
    assert count_warn.tec_maps == 2
    assert count_warn.all_maps == 2

    # 5. NotANumberValue
    nan_bands = _layout_bands(_i5_row(1, "nan"), _i5_row(3, 4))
    nan_text = _one_map_file(_exponent_record(0), nan_bands)
    res = sidereon.load_ionex_with_warnings(nan_text.encode("utf-8"))
    nan_warn = next(w for w in res.warnings if w.kind == "NOT_A_NUMBER_VALUE")
    assert nan_warn.line == 20
    assert nan_warn.data_kind == "TEC"
    assert nan_warn.map_number == 1
    assert nan_warn.lat_deg == 1.0
    assert nan_warn.lon_deg == 1.0

    # 6. IntervalMismatch
    wrong_interval = (
        _layout_header(2, LAYOUT_EPOCH_1, _exponent_record(0)).replace(
            "  3600", "  1800"
        )
        + _layout_map("TEC", 1, LAYOUT_EPOCH_0, bands)
        + _layout_map("TEC", 2, LAYOUT_EPOCH_1, bands)
        + _layout_end()
    )
    res = sidereon.load_ionex_with_warnings(wrong_interval.encode("utf-8"))
    interval_warn = next(w for w in res.warnings if w.kind == "INTERVAL_MISMATCH")
    assert interval_warn.line == 5
    assert interval_warn.declared_s == 1800
    assert interval_warn.spacing_s == 3600
    assert interval_warn.map_number == 2

    # 7. ExponentCarriedIntoMap
    changed_bands = _exponent_record(-2) + _layout_bands(
        _i5_row(100, 200), _i5_row(300, 400)
    )
    carried_bands = _layout_bands(_i5_row(100, 200), _i5_row(300, 400))
    carried_text = (
        _layout_header(2, LAYOUT_EPOCH_1, "")
        + _layout_map("TEC", 1, LAYOUT_EPOCH_0, changed_bands)
        + _layout_map("TEC", 2, LAYOUT_EPOCH_1, carried_bands)
        + _layout_end()
    )
    res = sidereon.load_ionex_with_warnings(carried_text.encode("utf-8"))
    exp_warn = next(w for w in res.warnings if w.kind == "EXPONENT_CARRIED_INTO_MAP")
    assert exp_warn.line == 26
    assert exp_warn.exponent == -2
    assert exp_warn.map_number == 2
    assert exp_warn.data_kind == "TEC"
    assert exp_warn.set_by_line == 16


def test_batch_mixed_success_failure_preserves_order_and_details():
    # Three latitude rows, 1, 0 and -1, over longitudes 0 and 1: two cells.
    # The one missing node, latitude 1 longitude 1, is a corner of the
    # northern cell only, so a pierce point inside the southern cell weights
    # four available nodes and one inside the northern cell weights the
    # missing one.
    #
    # The pierce point is not the receiver. Even at 90 degrees elevation the
    # earth-central angle is -asin(Re/(Re+H) * cos(pi/2)), about -5.7e-17 rad
    # rather than zero, so a receiver on a grid edge can put its pierce point
    # just outside the grid, and a lower elevation moves it by tenths of a
    # degree. Every request meant to succeed therefore sits well inside the
    # southern cell: at 88 degrees elevation the pierce point moves about
    # 0.13 degrees, at 85 degrees about 0.33 degrees.
    lat_axis = f"  {1.0:6.1f}{-1.0:6.1f}{-1.0:6.1f}"
    body = _layout_bands(_i5_row(10, 9999), _i5_row(30, 40)) + _layout_band(
        -1.0, 0.0, 1.0, 1.0, 450.0, _i5_row(50, 60)
    )
    text = _one_map_file(_exponent_record(0), body, lat_axis=lat_axis)
    ionex = sidereon.load_ionex(text.encode("utf-8"))
    epoch0 = int(ionex.map_epochs_j2000_s[0])

    requests = [
        # 0. Zenith inside the southern cell -> Ok
        sidereon.IonexSlantRequest(-0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0),
        # 1. Invalid receiver latitude 95.0 deg -> Refusal INVALID_INPUT
        sidereon.IonexSlantRequest(95.0, 0.5, 0.0, 90.0, epoch0, 1575420000.0),
        # 2. Pierce point near (-0.41, 0.59), southern cell -> Ok
        sidereon.IonexSlantRequest(-0.5, 0.5, 45.0, 88.0, epoch0, 1575420000.0),
        # 3. Nonfinite receiver coordinate NaN -> Refusal INVALID_INPUT
        sidereon.IonexSlantRequest(float("nan"), 0.5, 0.0, 90.0, epoch0, 1575420000.0),
        # 4. Pierce point near (-0.5, 0.83), southern cell -> Ok
        sidereon.IonexSlantRequest(-0.5, 0.5, 90.0, 85.0, epoch0, 1575420000.0),
        # 5. Core-level bad frequency -1.0 Hz -> Refusal INVALID_INPUT
        sidereon.IonexSlantRequest(-0.5, 0.5, 0.0, 90.0, epoch0, -1.0),
        # 6. Zenith inside the northern cell, which weights the missing node
        #    -> Refusal MISSING_NODES
        sidereon.IonexSlantRequest(0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0),
        # 7. Out of coverage epoch before first map -> Refusal COVERAGE
        sidereon.IonexSlantRequest(-0.5, 0.5, 0.0, 90.0, epoch0 - 3600, 1575420000.0),
        # 8. Out of coverage epoch after last map -> Refusal COVERAGE
        sidereon.IonexSlantRequest(-0.5, 0.5, 0.0, 90.0, epoch0 + 3600, 1575420000.0),
        # 9. Zenith inside the southern cell, looking south -> Ok
        sidereon.IonexSlantRequest(-0.5, 0.5, 180.0, 90.0, epoch0, 1575420000.0),
    ]

    results = ionex.slant_delays_batch_results(requests)
    assert len(results) == 10

    # Check top-level ionex_slant_delay_results produces identical structure
    top_results = sidereon.ionex_slant_delay_results(ionex, requests)
    assert len(top_results) == 10

    for res_list in (results, top_results):
        # 0: Ok
        assert res_list[0].is_ok
        assert res_list[0].evaluation is not None
        assert res_list[0].delay_m is not None and res_list[0].delay_m > 0.0
        assert res_list[0].status.is_valid
        assert res_list[0].unwrap().delay_m == res_list[0].delay_m

        # 1: Invalid latitude
        assert not res_list[1].is_ok
        assert res_list[1].evaluation is None
        assert res_list[1].refusal.kind == "INVALID_INPUT"
        with pytest.raises(ValueError, match="invalid receiver"):
            res_list[1].unwrap()

        # 2: Ok
        assert res_list[2].is_ok
        assert res_list[2].delay_m is not None and res_list[2].delay_m > 0.0

        # 3: Nonfinite coordinate
        assert not res_list[3].is_ok
        assert res_list[3].refusal.kind == "INVALID_INPUT"
        with pytest.raises(ValueError, match="invalid receiver"):
            res_list[3].unwrap()

        # 4: Ok
        assert res_list[4].is_ok
        assert res_list[4].delay_m is not None and res_list[4].delay_m > 0.0

        # 5: Bad frequency
        assert not res_list[5].is_ok
        assert res_list[5].refusal.kind == "INVALID_INPUT"
        with pytest.raises(ValueError, match="frequency"):
            res_list[5].unwrap()

        # 6: Missing nodes
        assert not res_list[6].is_ok
        assert res_list[6].refusal.kind == "MISSING_NODES"
        assert res_list[6].refusal.node_gap is not None
        with pytest.raises(ValueError, match="nodes not available"):
            res_list[6].unwrap()

        # 7: Coverage before
        assert not res_list[7].is_ok
        assert res_list[7].refusal.kind == "COVERAGE"
        assert (
            res_list[7].refusal.coverage_error
            == sidereon.IonexCoverageError.EPOCH_BEFORE_FIRST_MAP
        )

        # 8: Coverage after
        assert not res_list[8].is_ok
        assert res_list[8].refusal.kind == "COVERAGE"
        assert (
            res_list[8].refusal.coverage_error
            == sidereon.IonexCoverageError.EPOCH_AFTER_LAST_MAP
        )

        # 9: Ok
        assert res_list[9].is_ok
        assert res_list[9].delay_m is not None and res_list[9].delay_m > 0.0


def test_writer_rejects_precision_loss():
    epochs = np.array([631108800], dtype=np.int64)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)

    # 12345.6 cannot fit in standard I5 at any exponent without loss
    wide_vals = np.array([[[12345.6, 1.0], [2.0, 3.0]]], dtype=np.float64)
    grid = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, -1, wide_vals
    )
    ionex = sidereon.Ionex.from_samples(grid)

    assert issubclass(sidereon.IonexWriteError, sidereon.IonexParseError)
    assert issubclass(sidereon.IonexWriteError, ValueError)
    with pytest.raises(
        sidereon.IonexWriteError,
        match=(
            r"is not, for any EXPONENT k, a whole number of 10\^k units "
            r"within I5 other than 9999"
        ),
    ) as excinfo:
        ionex.to_ionex_string()
    assert excinfo.value.detail == str(excinfo.value)


def test_mapping_function_custom_preserves_exact_text_and_whitespace():
    # 1. Constructor preserves exact custom code text including whitespace
    mf_trailing = sidereon.IonexMappingFunction("MOD ")
    assert mf_trailing.code == "MOD "
    assert mf_trailing.kind == "OTHER"
    assert "MOD " in repr(mf_trailing)
    assert str(mf_trailing) == "MOD "

    mf_empty = sidereon.IonexMappingFunction("")
    assert mf_empty.code == ""
    assert mf_empty.kind == "OTHER"

    mf_spaces = sidereon.IonexMappingFunction("   ")
    assert mf_spaces.code == "   "
    assert mf_spaces.kind == "OTHER"

    # PyIonexMappingFunction.other remains exact
    mf_other = sidereon.IonexMappingFunction.other("CUSTOM ")
    assert mf_other.code == "CUSTOM "
    assert mf_other.kind == "OTHER"

    # Exact canonical codes match kind
    assert sidereon.IonexMappingFunction("NONE").kind == "NONE"
    assert sidereon.IonexMappingFunction("COSZ").kind == "COSZ"
    assert sidereon.IonexMappingFunction("QFAC").kind == "QFAC"
    # Near-canonical codes with whitespace remain OTHER and retain exact chars
    assert sidereon.IonexMappingFunction(" NONE").kind == "OTHER"
    assert sidereon.IonexMappingFunction("NONE ").kind == "OTHER"
    assert sidereon.IonexMappingFunction("NONE ").code == "NONE "

    # 2. The header constructor keeps the exact string and does not turn an empty
    # string into None
    hdr_trailing = sidereon.IonexHeader(mapping_function="MOD ")
    assert hdr_trailing.mapping_function is not None
    assert hdr_trailing.mapping_function.code == "MOD "
    assert hdr_trailing.mapping_declaration.is_declared
    assert not hdr_trailing.mapping_declaration.is_absent
    assert hdr_trailing.mapping_declaration.code == "MOD "

    hdr_empty = sidereon.IonexHeader(mapping_function="")
    assert hdr_empty.mapping_function is not None
    assert hdr_empty.mapping_function.code == ""
    assert hdr_empty.mapping_declaration.is_declared
    assert not hdr_empty.mapping_declaration.is_absent

    hdr_spaces = sidereon.IonexHeader(mapping_function="   ")
    assert hdr_spaces.mapping_function is not None
    assert hdr_spaces.mapping_function.code == "   "
    assert hdr_spaces.mapping_declaration.is_declared
    assert not hdr_spaces.mapping_declaration.is_absent

    # Explicit absence is None
    hdr_absent = sidereon.IonexHeader(mapping_function=None)
    assert hdr_absent.mapping_function is None
    assert hdr_absent.mapping_declaration.is_absent
    assert not hdr_absent.mapping_declaration.is_declared
    assert hdr_absent.mapping_declaration.code is None

    # 3. Fallible core writer named-refuses unrepresentable raw codes
    epochs = np.array([631108800], dtype=np.int64)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)
    tec = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)

    # Trailing whitespace is unrepresentable in writer
    grid_trailing = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=hdr_trailing
    )
    ionex_trailing = sidereon.Ionex.from_samples(grid_trailing)
    with pytest.raises(ValueError, match="MAPPING FUNCTION code.*cannot be written"):
        ionex_trailing.to_ionex_string()

    # Empty string is unrepresentable in writer
    grid_empty = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=hdr_empty
    )
    ionex_empty = sidereon.Ionex.from_samples(grid_empty)
    with pytest.raises(ValueError, match="MAPPING FUNCTION code.*cannot be written"):
        ionex_empty.to_ionex_string()

    # Whitespace-only string is unrepresentable in writer
    grid_spaces = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=hdr_spaces
    )
    ionex_spaces = sidereon.Ionex.from_samples(grid_spaces)
    with pytest.raises(ValueError, match="MAPPING FUNCTION code.*cannot be written"):
        ionex_spaces.to_ionex_string()

    # Valid custom code (<=4 chars, no whitespace) succeeds and roundtrips
    hdr_valid_custom = sidereon.IonexHeader(mapping_function="MOD")
    grid_valid_custom = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, header=hdr_valid_custom
    )
    ionex_valid_custom = sidereon.Ionex.from_samples(grid_valid_custom)
    text = ionex_valid_custom.to_ionex_string()
    # `MAPPING FUNCTION` is `2X,A4`: two blanks, the code left-justified in
    # its four columns, blanks to column 60, then the label.
    assert f"{'  MOD':<60}MAPPING FUNCTION" in text.splitlines()
    reparsed_custom = sidereon.load_ionex(text.encode("utf-8"))
    assert reparsed_custom.header.mapping_function.kind == "OTHER"
    assert reparsed_custom.header.mapping_function.code == "MOD"


def test_optional_maps_roundtrip_and_canonical_empty_shape_validation():
    epochs = np.array([631108800], dtype=np.int64)
    lats = np.array([1.0, 0.0], dtype=np.float64)
    lons = np.array([0.0, 1.0], dtype=np.float64)
    tec = np.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=np.float64)

    # 1. Extraction -> constructor -> Ionex round trip for absent optional maps
    grid_absent = sidereon.TecGridSamples(
        epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec
    )
    ionex_absent = sidereon.Ionex.from_samples(grid_absent)
    assert not ionex_absent.has_rms
    assert not ionex_absent.has_height
    assert ionex_absent.rms_maps.shape == (0, 0, 0)
    assert ionex_absent.rms_mask.shape == (0, 0, 0)
    assert ionex_absent.height_maps.shape == (0, 0, 0)
    assert ionex_absent.height_mask.shape == (0, 0, 0)

    # Rebuild TecGridSamples from the exact (0, 0, 0) arrays and masks the getters
    # emit
    reconstructed_grid = sidereon.TecGridSamples(
        map_epochs_j2000_s=ionex_absent.map_epochs_j2000_s,
        lat_nodes_deg=ionex_absent.lat_nodes_deg,
        lon_nodes_deg=ionex_absent.lon_nodes_deg,
        dlat_deg=ionex_absent.dlat_deg,
        dlon_deg=ionex_absent.dlon_deg,
        shell_height_km=ionex_absent.shell_height_km,
        base_radius_km=ionex_absent.base_radius_km,
        exponent=ionex_absent.exponent,
        tec_maps=ionex_absent.tec_maps,
        rms_maps=ionex_absent.rms_maps,
        height_maps=ionex_absent.height_maps,
        tec_mask=ionex_absent.tec_mask,
        rms_mask=ionex_absent.rms_mask,
        height_mask=ionex_absent.height_mask,
        header=ionex_absent.header,
    )
    reconstructed_ionex = sidereon.Ionex.from_samples(reconstructed_grid)
    assert not reconstructed_ionex.has_rms
    assert not reconstructed_ionex.has_height
    assert reconstructed_ionex.rms_maps.shape == (0, 0, 0)
    assert reconstructed_ionex.rms_mask.shape == (0, 0, 0)
    assert reconstructed_ionex.height_maps.shape == (0, 0, 0)
    assert reconstructed_ionex.height_mask.shape == (0, 0, 0)
    np.testing.assert_array_equal(reconstructed_ionex.tec_maps, ionex_absent.tec_maps)
    np.testing.assert_array_equal(reconstructed_ionex.tec_mask, ionex_absent.tec_mask)

    # 2. Transparent roundtrip for PRESENT ALL-MISSING optional maps
    all_dummy = np.zeros((1, 2, 2), dtype=np.float64)
    all_false_mask = np.zeros((1, 2, 2), dtype=bool)
    grid_present_empty = sidereon.TecGridSamples(
        epochs,
        lats,
        lons,
        -1.0,
        1.0,
        450.0,
        6371.0,
        0,
        tec,
        rms_maps=all_dummy,
        rms_mask=all_false_mask,
        height_maps=all_dummy,
        height_mask=all_false_mask,
    )
    ionex_present_empty = sidereon.Ionex.from_samples(grid_present_empty)
    assert ionex_present_empty.has_rms
    assert ionex_present_empty.has_height

    reconstructed_present_empty_grid = sidereon.TecGridSamples(
        map_epochs_j2000_s=ionex_present_empty.map_epochs_j2000_s,
        lat_nodes_deg=ionex_present_empty.lat_nodes_deg,
        lon_nodes_deg=ionex_present_empty.lon_nodes_deg,
        dlat_deg=ionex_present_empty.dlat_deg,
        dlon_deg=ionex_present_empty.dlon_deg,
        shell_height_km=ionex_present_empty.shell_height_km,
        base_radius_km=ionex_present_empty.base_radius_km,
        exponent=ionex_present_empty.exponent,
        tec_maps=ionex_present_empty.tec_maps,
        rms_maps=ionex_present_empty.rms_maps,
        height_maps=ionex_present_empty.height_maps,
        tec_mask=ionex_present_empty.tec_mask,
        rms_mask=ionex_present_empty.rms_mask,
        height_mask=ionex_present_empty.height_mask,
        header=ionex_present_empty.header,
    )
    reconstructed_present_empty_ionex = sidereon.Ionex.from_samples(
        reconstructed_present_empty_grid
    )
    assert reconstructed_present_empty_ionex.has_rms
    assert reconstructed_present_empty_ionex.has_height
    assert reconstructed_present_empty_ionex.rms_maps.shape == (1, 2, 2)
    assert reconstructed_present_empty_ionex.height_maps.shape == (1, 2, 2)
    assert np.all(np.isnan(reconstructed_present_empty_ionex.rms_maps))
    assert not np.any(reconstructed_present_empty_ionex.rms_mask)

    # 3. Transparent roundtrip for PRESENT POPULATED optional maps
    populated_rms = np.array([[[1.5, 2.5], [3.5, 4.5]]], dtype=np.float64)
    populated_mask = np.ones((1, 2, 2), dtype=bool)
    grid_populated = sidereon.TecGridSamples(
        epochs,
        lats,
        lons,
        -1.0,
        1.0,
        450.0,
        6371.0,
        0,
        tec,
        rms_maps=populated_rms,
        rms_mask=populated_mask,
    )
    ionex_populated = sidereon.Ionex.from_samples(grid_populated)
    assert ionex_populated.has_rms
    reconstructed_pop_grid = sidereon.TecGridSamples(
        map_epochs_j2000_s=ionex_populated.map_epochs_j2000_s,
        lat_nodes_deg=ionex_populated.lat_nodes_deg,
        lon_nodes_deg=ionex_populated.lon_nodes_deg,
        dlat_deg=ionex_populated.dlat_deg,
        dlon_deg=ionex_populated.dlon_deg,
        shell_height_km=ionex_populated.shell_height_km,
        base_radius_km=ionex_populated.base_radius_km,
        exponent=ionex_populated.exponent,
        tec_maps=ionex_populated.tec_maps,
        rms_maps=ionex_populated.rms_maps,
        rms_mask=ionex_populated.rms_mask,
    )
    reconstructed_pop_ionex = sidereon.Ionex.from_samples(reconstructed_pop_grid)
    assert reconstructed_pop_ionex.has_rms
    np.testing.assert_array_equal(
        reconstructed_pop_ionex.rms_maps, ionex_populated.rms_maps
    )
    np.testing.assert_array_equal(
        reconstructed_pop_ionex.rms_mask, ionex_populated.rms_mask
    )

    # 4. Reject malformed partial-empty dimensions
    for bad_shape in [(0, 2, 2), (1, 0, 2), (1, 2, 0)]:
        bad_rms = np.zeros(bad_shape, dtype=np.float64)
        with pytest.raises(ValueError, match="partial empty dimensions"):
            sidereon.TecGridSamples(
                epochs, lats, lons, -1.0, 1.0, 450.0, 6371.0, 0, tec, rms_maps=bad_rms
            )
        with pytest.raises(ValueError, match="partial empty dimensions"):
            sidereon.TecGridSamples(
                epochs,
                lats,
                lons,
                -1.0,
                1.0,
                450.0,
                6371.0,
                0,
                tec,
                height_maps=bad_rms,
            )

    # 5. Reject mask mismatch with canonical empty cube (0, 0, 0)
    empty_cube = np.zeros((0, 0, 0), dtype=np.float64)
    non_empty_mask = np.zeros((1, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            0,
            tec,
            rms_maps=empty_cube,
            rms_mask=non_empty_mask,
        )
    with pytest.raises(ValueError, match="does not match"):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            0,
            tec,
            height_maps=empty_cube,
            height_mask=non_empty_mask,
        )

    # 6. Reject mask supplied when values are None (mask without values)
    empty_mask = np.zeros((0, 0, 0), dtype=bool)
    with pytest.raises(
        ValueError, match="cannot supply rms_mask or rms_valid when rms_maps is absent"
    ):
        sidereon.TecGridSamples(
            epochs,
            lats,
            lons,
            -1.0,
            1.0,
            450.0,
            6371.0,
            0,
            tec,
            rms_maps=None,
            rms_mask=empty_mask,
        )


def test_runtime_stubs_parity_and_nominal_evaluation():
    # 1. IonexSlantPolicy.default_policy()
    default_pol = sidereon.IonexSlantPolicy.default_policy()
    assert default_pol.coverage == sidereon.IonexCoveragePolicy.STRICT
    assert default_pol.missing_nodes == sidereon.IonexMissingNodePolicy.STRICT
    assert default_pol.mapping == sidereon.IonexMappingPolicy.SINGLE_LAYER

    # 2. Ionex.skipped_records, parse_str, parse_str_with_warnings
    body = _layout_bands(_i5_row(10, 20), _i5_row(30, 40))
    extra = _exponent_record(0) + _ionex_record("  COSZ", "MAPPING FUNCTION")
    text = _one_map_file(extra, body)

    ionex_str = sidereon.Ionex.parse_str(text)
    assert ionex_str.skipped_records == 0
    assert ionex_str.shell_height_km == 450.0

    parsed_with_warns = sidereon.Ionex.parse_str_with_warnings(text)
    assert parsed_with_warns.ionex.skipped_records == 0
    assert isinstance(parsed_with_warns.warnings, list)

    # 3. Fully nominal evaluation
    epoch0 = int(ionex_str.map_epochs_j2000_s[0])
    eval_nom = ionex_str.slant_delay_with_policy(
        0.5, 0.5, 0.0, 90.0, epoch0, 1575420000.0, policy=default_pol
    )
    assert eval_nom.is_valid
    assert eval_nom.is_nominal
    assert eval_nom.status.is_valid
    assert eval_nom.status.is_nominal
    assert eval_nom.status.held is None
    assert eval_nom.status.degraded is None
    assert eval_nom.status.assumed_mapping is None
    assert not eval_nom.status.is_held
    assert not eval_nom.status.is_degraded
    assert not eval_nom.status.is_assumed_mapping
    assert not eval_nom.status.has_assumed_mapping
    assert eval_nom.held is None
    assert eval_nom.degraded is None
    assert eval_nom.assumed_mapping is None
