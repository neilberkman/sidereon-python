import math
from pathlib import Path

import pytest
import sidereon


def dual(**changes):
    values = dict(
        satellite_id="G01",
        ambiguity_id="G01",
        p1_m=20_000_002.0,
        p2_m=20_000_000.0,
        phi1_cyc=110_000_000.0,
        phi2_cyc=85_000_000.0,
        f1_hz=1_575_420_000.0,
        f2_hz=1_227_600_000.0,
    )
    values.update(changes)
    return sidereon.PppDualFrequencyObservation(**values)


def residual(satellite, code_m, phase_m=0.0):
    return sidereon.PppFloatResidual(0, satellite, satellite, code_m, phase_m, 1.0, 1.0)


def test_tec_exact_combinations_leveling_and_pierce_point():
    observation = dual()
    assert sidereon.code_geometry_free_m(observation) == 2.0
    expected = 2.0 / (
        sidereon.TEC_GROUP_DELAY_COEFFICIENT
        * (1.0 / observation.f1_hz**2 - 1.0 / observation.f2_hz**2)
    )
    assert (
        sidereon.slant_tec_from_code_geometry_free_m(
            2.0, observation.f1_hz, observation.f2_hz
        )
        == expected
    )
    samples = [
        sidereon.PppTecLevelingSample(10.0, 13.0, math.pi / 2),
        sidereon.PppTecLevelingSample(12.0, 15.0, math.pi / 2),
    ]
    leveled = sidereon.level_slant_tec_arc(samples, sidereon.PppTecConfig())
    assert leveled.phase_bias_tecu == 3.0
    assert [sample.vertical_tec_tecu for sample in leveled.samples] == [10.0, 12.0]
    point = sidereon.ionospheric_pierce_point(
        0.4, -1.2, math.pi / 2, 0.7, sidereon.PppTecConfig()
    )
    assert point.latitude_rad == 0.39999999999999997
    assert point.longitude_rad == -1.2
    assert point.shell_height_m == 350_000.0


def test_tec_refusals_have_exact_structured_details():
    with pytest.raises(sidereon.SolveError) as invalid:
        sidereon.thin_shell_mapping_function(0.5, sidereon.PppTecConfig(-1.0))
    assert invalid.value.detail == {
        "family": "TecError",
        "kind": "invalid_shell_height",
    }
    with pytest.raises(sidereon.SolveError) as empty:
        sidereon.estimate_tec([], sidereon.PppTecConfig())
    assert empty.value.detail == {"family": "TecError", "kind": "no_epochs"}


def test_cycle_slip_state_updates_equality_and_refusals():
    statistic = sidereon.PppRunningMeanVariance()
    statistic.push(1.0)
    statistic.push(3.0)
    assert (statistic.sample_count, statistic.mean, statistic.variance) == (2, 2.0, 2.0)
    assert statistic.standard_deviation() == math.sqrt(2.0)
    state = sidereon.PppSatelliteCycleSlipState()
    config = sidereon.PppCycleSlipConfig()
    first = sidereon.update_geometry_free(state, dual(), config, 30.0)
    assert (first.slip, first.reset, state.previous_epoch_time_s) == (
        False,
        False,
        30.0,
    )
    assert state == state
    with pytest.raises(sidereon.SolveError) as backward:
        sidereon.update_geometry_free(state, dual(), config, 20.0)
    assert backward.value.detail == {
        "family": "CycleSlipError",
        "kind": "epochs_not_ordered",
    }
    with pytest.raises(sidereon.SolveError) as bad_config:
        sidereon.PppCycleSlipConfig(minimum_arc_length=0).validate()
    assert bad_config.value.detail == {
        "family": "CycleSlipConfigError",
        "kind": "invalid_minimum_arc_length",
    }


def test_precise_cycle_slip_collision_names_and_prep():
    assert sidereon.detect_cycle_slips.__doc__.startswith("Detect cycle slips")
    epochs = [sidereon.PppDualFrequencyEpoch([dual()], gap_time_s=0.0)]
    flags = sidereon.ppp_detect_cycle_slips(epochs, sidereon.PppCycleSlipConfig())
    assert flags[0].observations[0].satellite_id == "G01"
    assert flags[0].observations[0].slip is False
    assert sidereon.ppp_geometry_free_m(dual()) == sidereon.phase_geometry_free_m(
        dual()
    )
    float_epochs = [
        sidereon.PppFloatCycleSlipEpoch(
            [sidereon.PppFloatCycleSlipObservation("G01", "arc", None)]
        )
    ]
    tagged = sidereon.split_float_cycle_slip_epochs(
        float_epochs, sidereon.CycleSlipOptions()
    )
    assert tagged[0].observations[0].ambiguity_id == "arc"
    with pytest.raises(sidereon.SolveError) as short:
        sidereon.prepare_widelane_fixed_epochs(
            epochs,
            sidereon.PppWideLanePrepOptions(2, 0.25),
            "error",
            sidereon.CycleSlipOptions(),
        )
    assert short.value.detail["family"] == "WideLanePrepError"
    assert short.value.detail["kind"] == "too_few_wide_lane_epochs"
    assert short.value.detail["count"] == 1
    assert short.value.detail["minimum"] == 2


def test_raim_exact_global_values_status_equality_and_refusals():
    rows = [residual("G01", 1.0), residual("G02", 2.0), residual("G03", 3.0)]
    config = sidereon.PppRaimConfig(chi_square_threshold=20.0)
    result = sidereon.global_test(rows, 5, config)
    assert result.status == sidereon.PppRaimStatus.PASSED
    assert (result.test_statistic, result.threshold, result.redundancy) == (
        14.0,
        20.0,
        1,
    )
    assert result == sidereon.global_test(rows, 5, config)
    assert result != sidereon.global_test(rows, 6, config)
    with pytest.raises(sidereon.SolveError) as invalid:
        sidereon.global_test(rows, 5, sidereon.PppRaimConfig(measurement_sigma_m=0.0))
    assert invalid.value.detail == {
        "family": "RaimError",
        "kind": "invalid_config",
        "field": "measurement_sigma_m",
        "reason": "must be positive and finite",
    }


def test_raim_geometry_and_protection_refusals():
    geometry = [
        sidereon.PppRaimGeometryRow("G01", [1.0, 0.0, 0.0]),
        sidereon.PppRaimGeometryRow("G02", [0.0, 1.0, 0.0]),
        sidereon.PppRaimGeometryRow("G03", [0.0, 0.0, 1.0]),
    ]
    rows = [residual("G01", 0.0), residual("G02", 0.0), residual("G03", 5.0)]
    with pytest.raises(sidereon.SolveError) as singular:
        sidereon.protection_levels(
            geometry, sidereon.Wgs84Geodetic(0.0, 0.0), sidereon.PppRaimConfig()
        )
    assert singular.value.detail["family"] == "RaimError"
    assert singular.value.detail["kind"] == "invalid_geometry"
    assert (
        singular.value.detail["reason"]
        == "protection levels require positive redundancy"
    )
    with pytest.raises(sidereon.SolveError) as mismatch:
        sidereon.per_satellite_statistics(rows[:2], geometry, sidereon.PppRaimConfig())
    assert mismatch.value.detail["kind"] == "invalid_geometry"


def test_raim_fde_entry_points_preserve_float_solve_error_details():
    source = sidereon.load_sp3(
        Path("tests/fixtures/sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3")
    )
    epoch = sidereon.PppEpoch(
        sidereon.PppCivilDateTime(2026, 4, 30, 9, 45, 0.0),
        2_461_160.0,
        -0.09375,
        830_000_000.0,
        [],
    )
    state = sidereon.PppFloatState([0.0, 0.0, 0.0], [0.0], {})
    for solve in (sidereon.fde_float_epoch, sidereon.solve_float_epoch_with_raim):
        with pytest.raises(sidereon.SolveError) as refused:
            solve(
                source,
                epoch,
                state,
                sidereon.PppFloatConfig(),
                sidereon.PppRaimConfig(),
            )
        assert refused.value.detail["family"] == "PppFloatSolveError"
