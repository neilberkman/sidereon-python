import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _sp3_path(name):
    return os.path.join(CORE_FIXTURES, "sp3", name)


def _load_sp3(name):
    return sidereon.load_sp3(_sp3_path(name))


def test_merge_sp3_real_gbm_products_spans_common_coverage_and_interpolates():
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
    assert merged_axis[0] == max(full_axis[0], trim_axis[0])
    assert merged_axis[-1] == min(full_axis[-1], trim_axis[-1])
    assert {"C08", "C21"}.issubset(merged.satellites)
    assert all(sat.startswith("C") for sat in merged.satellites)
    assert report.quarantined_count == 0
    assert report.position_outlier_count == 0
    assert report.single_source_count > 0

    query = np.asarray([(merged_axis[20] + merged_axis[21]) / 2.0], dtype=np.float64)
    expected = full.interpolate("C21", query)
    actual = merged.interpolate("C21", query)
    np.testing.assert_allclose(
        actual.position_m, expected.position_m, rtol=0.0, atol=1.0e-6
    )
    np.testing.assert_allclose(actual.clock_s, expected.clock_s, rtol=0.0, atol=1.0e-12)


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


def test_sp3_merge_options_accept_string_selectors_and_validate_systems():
    options = sidereon.Sp3MergeOptions(combine="precedence", systems=["GPS", "C"])
    assert options.combine == sidereon.Sp3MergeCombine.PRECEDENCE
    assert options.systems == ["G", "C"]

    with pytest.raises(ValueError, match="unknown GNSS system"):
        sidereon.Sp3MergeOptions(systems=["X"])
