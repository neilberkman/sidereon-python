import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _sp3_path(name):
    return os.path.join(CORE_FIXTURES, "sp3", name)


def _load_sp3(name):
    return sidereon.load_sp3(_sp3_path(name))


def _mini_sp3(label, records):
    sats = "".join(sat for sat, _, _ in records)
    sats += "  0" * (17 - len(records))
    lines = [
        f"#cP2020  6 25  0  0  0.00000000       1 ORBIT {label} FIT  TST",
        "## 2111 432000.00000000   900.00000000 59025 0.0000000000000",
        f"+   {len(records):2}   {sats}",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
        "/* TEST SP3-c FIXTURE",
        "*  2020  6 25  0  0  0.00000000",
    ]
    for sat, position_km, clock_us in records:
        x, y, z = position_km
        lines.append(f"P{sat}{x:14.6f}{y:14.6f}{z:14.6f}{clock_us:14.6f}")
    lines.append("EOF")
    return sidereon.load_sp3(("\n".join(lines) + "\n").encode("ascii"))


def test_merge_sp3_real_gbm_products_spans_union_coverage_and_interpolates():
    full = _load_sp3("GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3")
    trim = _load_sp3("GBM_BDS_C21_C08_trim.sp3")
    options = sidereon.Sp3MergeOptions(
        position_tolerance_m=1.0e-6,
        clock_min_common=1,
        systems=["C"],
    )

    merged, report = sidereon.merge_sp3([full, trim], options)

    full_axis = full.epochs_j2000_seconds
    trim_axis = trim.epochs_j2000_seconds
    merged_axis = merged.epochs_j2000_seconds
    assert merged_axis[0] == min(full_axis[0], trim_axis[0])
    assert merged_axis[-1] == max(full_axis[-1], trim_axis[-1])
    assert {"C08", "C21"}.issubset(merged.satellites)
    assert all(sat.startswith("C") for sat in merged.satellites)
    assert report.quarantined_count == 0
    assert report.position_outlier_count == 0
    assert report.single_source_count > 0

    query = np.asarray([(merged_axis[-3] + merged_axis[-2]) / 2.0], dtype=np.float64)
    expected = trim.interpolate("C21", query)
    actual = merged.interpolate("C21", query)
    np.testing.assert_allclose(
        actual.position_m, expected.position_m, rtol=0.0, atol=1.0e-6
    )
    assert np.isfinite(actual.clock_s[0])


def test_merge_sp3_degenerate_single_source_reports_single_source_cells():
    sp3 = _load_sp3("degenerate_coincident_5sat.sp3")
    options = sidereon.Sp3MergeOptions(min_agree=1, clock_min_common=1)

    merged, report = sidereon.merge_sp3([sp3], options)

    assert merged.epoch_count == sp3.epoch_count
    assert merged.satellites == sp3.satellites
    assert report.single_source_count == sp3.epoch_count * len(sp3.satellites)
    assert report.quarantined_count == 0
    first_flag = report.single_source[0]
    assert first_flag.satellite in sp3.satellites
    assert first_flag.sources == [0]
    assert np.isfinite(first_flag.epoch_j2000_seconds)

    axis = sp3.epochs_j2000_seconds
    query = np.asarray([(axis[0] + axis[1]) / 2.0], dtype=np.float64)
    expected = sp3.interpolate("G01", query)
    actual = merged.interpolate("G01", query)
    np.testing.assert_allclose(
        actual.position_m, expected.position_m, rtol=0.0, atol=1.0e-9
    )
    np.testing.assert_allclose(actual.clock_s, expected.clock_s, rtol=0.0, atol=1.0e-15)


def test_merge_sp3_rejects_empty_sources_and_mismatched_frames():
    with pytest.raises(ValueError, match="at least one"):
        sidereon.merge_sp3([])

    cod = _load_sp3("COD0MGXFIN_20201770000_01D_05M_ORB.SP3")
    gbm = _load_sp3("GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3")
    with pytest.raises(ValueError, match="mismatched coordinate systems"):
        sidereon.merge_sp3([cod, gbm])


def test_merge_sp3_asserted_frame_equivalence_reports_no_math():
    a = _mini_sp3("IGS14", [("G01", [15000.0, -20000.0, 5000.0], 100.0)])
    b = _mini_sp3("ITRF2", [("G02", [16000.0, -21000.0, 6000.0], 200.0)])
    options = sidereon.Sp3MergeOptions(
        asserted_frame_label_sets=[["IGS14", "ITRF2"]],
    )

    merged, report = sidereon.merge_sp3([a, b], options)

    assert {"G01", "G02"} == set(merged.satellites)
    assert report.frame_reconciliation_count == 1
    reconciliation = report.frame_reconciliations[0]
    assert reconciliation.method == "asserted_equivalence"
    assert reconciliation.source_index == 1
    assert reconciliation.source_label == "ITRF2"
    assert reconciliation.target_label == "IGS14"
    assert reconciliation.asserted_label_set == ["IGS14", "ITRF2"]
    assert reconciliation.parameters is None
    assert reconciliation.rates is None
    assert reconciliation.records_affected == 1


def test_merge_sp3_helmert_frame_reconciliation_reports_table_values():
    a = _mini_sp3("IGS14", [("G01", [14000.0, -19000.0, 4000.0], 100.0)])
    b = _mini_sp3("IGS20", [("G02", [15000.0, -20000.0, 5000.0], 200.0)])
    options = sidereon.Sp3MergeOptions(min_agree=1, helmert=True)

    merged, report = sidereon.merge_sp3([a, b], options)

    got = merged.state("G02", 0).position_m
    expected = np.array(
        [14_999_999.992_3, -19_999_999.993_048_087, 5_000_000.000_396_175],
        dtype=np.float64,
    )
    np.testing.assert_allclose(got, expected, rtol=0.0, atol=2.0e-9)
    reconciliation = report.frame_reconciliations[0]
    assert reconciliation.method == "helmert"
    assert reconciliation.source_frame == "ITRF2020"
    assert reconciliation.target_frame == "ITRF2014"
    assert reconciliation.catalog_source_frame == "ITRF2020"
    assert reconciliation.catalog_target_frame == "ITRF2014"
    assert reconciliation.catalog_inverse is False
    assert reconciliation.parameters == ([-1.4, -0.9, 1.4], -0.42, [0.0, 0.0, 0.0])
    assert reconciliation.rates == ([0.0, -0.1, 0.2], 0.0, [0.0, 0.0, 0.0])
    assert "ITRF2020 to past ITRFs" in reconciliation.provenance
    assert reconciliation.records_affected == 1


def test_merge_sp3_helmert_inverse_reports_catalog_direction():
    a = _mini_sp3("IGS20", [("G01", [14000.0, -19000.0, 4000.0], 100.0)])
    b = _mini_sp3("IGS14", [("G02", [15000.0, -20000.0, 5000.0], 200.0)])
    options = sidereon.Sp3MergeOptions(min_agree=1, helmert=True)

    _merged, report = sidereon.merge_sp3([a, b], options)

    reconciliation = report.frame_reconciliations[0]
    assert reconciliation.method == "helmert"
    assert reconciliation.source_frame == "ITRF2014"
    assert reconciliation.target_frame == "ITRF2020"
    assert reconciliation.catalog_source_frame == "ITRF2020"
    assert reconciliation.catalog_target_frame == "ITRF2014"
    assert reconciliation.catalog_inverse is True
    assert reconciliation.parameters == ([-1.4, -0.9, 1.4], -0.42, [0.0, 0.0, 0.0])


def test_merge_sp3_helmert_identity_reconciliation_is_bit_equal():
    a = _mini_sp3("IGS20", [("G01", [14000.0, -19000.0, 4000.0], 100.0)])
    b = _mini_sp3("IGc20", [("G02", [15000.125, -20000.5, 5000.25], 200.0)])
    original = b.state("G02", 0).position_m.copy()
    options = sidereon.Sp3MergeOptions(min_agree=1, helmert=True)

    merged, report = sidereon.merge_sp3([a, b], options)

    np.testing.assert_array_equal(merged.state("G02", 0).position_m, original)
    reconciliation = report.frame_reconciliations[0]
    assert reconciliation.identity is True
    assert reconciliation.parameters is None


def test_sp3_merge_options_accept_string_selectors_and_validate_systems():
    guard = sidereon.Sp3OutlierRejectOptions(0.5, 5.0e-9)
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="satellite_arc",
        outlier_reject=guard,
        systems=["GPS", "C"],
    )
    assert options.combine == sidereon.Sp3MergeCombine.PRECEDENCE
    assert options.precedence_scope == sidereon.Sp3MergePrecedenceScope.SATELLITE_ARC
    assert options.outlier_reject.position_tolerance_m == 0.5
    assert options.systems == ["G", "C"]
    assert options.asserted_frame_label_sets == []
    assert options.helmert is False

    with pytest.raises(ValueError, match="unknown GNSS system"):
        sidereon.Sp3MergeOptions(systems=["X"])

    with pytest.raises(ValueError, match="at least two labels"):
        sidereon.Sp3MergeOptions(asserted_frame_label_sets=[["IGS14"]])


def test_precedence_outlier_guard_rejects_corrupt_preferred_cell():
    preferred = _mini_sp3("IGS14", [("G01", [16000.0, -20000.0, 5000.0], 1000.0)])
    agreeing_a = _mini_sp3("IGS14", [("G01", [15000.0, -20000.0, 5000.0], 100.0)])
    agreeing_b = _mini_sp3("IGS14", [("G01", [15000.0001, -20000.0, 5000.0], 100.0)])
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        outlier_reject=sidereon.Sp3OutlierRejectOptions(0.5, 5.0e-9),
    )

    merged, report = sidereon.merge_sp3([preferred, agreeing_a, agreeing_b], options)

    np.testing.assert_allclose(
        merged.state("G01", 0).position_m,
        agreeing_a.state("G01", 0).position_m,
        rtol=0.0,
        atol=0.0,
    )
    assert report.position_outliers[0].sources == [0]
    assert report.clock_outlier_count == 0


def test_sp3_prediction_summary_is_exposed():
    sp3 = _load_sp3("degenerate_coincident_5sat.sp3")
    summary = sp3.prediction_summary()

    assert len(summary.epochs) == sp3.epoch_count
    assert all(epoch.observed for epoch in summary.epochs)
    assert summary.observed_through_j2000_seconds == sp3.epochs_j2000_seconds[-1]
