"""RTK float and validated-fixed solves through the binding reproduce the engine.

The fixture `rtk_wtzr.json` is emitted by the crate's validated WTZR/WTZZ static
GPS RTK integration test (`SIDEREON_DUMP_FIXTURES=1 cargo test --test
rtk_real_arc ...`); it carries the fully built epoch inputs and the engine's own
float and validated-fixed reference baselines. The binding drives the same
engine path, so it must return the identical baselines (bit-exact).
"""

import json
import os

import _helpers
import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES


def _fixture():
    with open(os.path.join(FIXTURES, "rtk_wtzr.json")) as fh:
        return json.load(fh)


WTZR_MARKER_M = np.array([4075580.3111, 931854.0543, 4801568.2808])
WTZZ_MARKER_M = np.array([4075579.1913, 931853.3696, 4801569.1897])
WTZR_OBS = "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"
WTZZ_OBS = "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"
WTZR_WTZZ_SP3 = "GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3"


def _core_fixture(*parts):
    return os.path.join(_helpers.CORE_FIXTURES, *parts)


def _wettzell_rinex_inputs():
    sp3 = sidereon.load_sp3(_core_fixture("sp3", WTZR_WTZZ_SP3))
    base_obs = sidereon.load_rinex_obs(_core_fixture("obs", WTZR_OBS))
    rover_obs = sidereon.load_rinex_obs(_core_fixture("obs", WTZZ_OBS))
    base_arp_m = _arp_position(WTZR_MARKER_M, base_obs)
    rover_arp_m = _arp_position(WTZZ_MARKER_M, rover_obs)
    return sp3, base_obs, rover_obs, base_arp_m, rover_arp_m - base_arp_m


def _arp_position(marker_m, obs):
    delta_hen = obs.header.antenna_delta_hen_m
    assert delta_hen is not None
    height_m, east_m, north_m = delta_hen
    assert east_m == 0.0
    assert north_m == 0.0
    return marker_m + marker_m / np.linalg.norm(marker_m) * height_m


def _real_arc_model():
    return sidereon.RtkMeasurementModel(
        code_sigma_m=2.0,
        phase_sigma_m=0.01,
        sagnac=True,
        stochastic=sidereon.RtkStochasticModel.SIMPLE,
        elevation_weighting=True,
    )


def _real_arc_float_options():
    return sidereon.RtkFloatOptions(
        position_tol_m=1.0e-4,
        ambiguity_tol_m=1.0e-4,
        max_iterations=10,
    )


def _real_arc_fixed_options():
    return sidereon.RtkFixedOptions(
        position_tol_m=1.0e-4,
        ambiguity_tol_m=1.0e-4,
        max_iterations=10,
        ratio_threshold=3.0,
        partial_ambiguity_resolution=False,
        partial_min_ambiguities=4,
    )


def _vector_error_m(vector, truth):
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64) - truth))


def _assert_square_covariance(covariance, dim):
    assert covariance.ndim == 2
    assert len(covariance) == dim
    assert len(covariance[0]) == dim
    assert np.all(np.diag(covariance) > 0.0)


def _sat(row):
    return sidereon.RtkSatMeasurement(
        sat=row["sat"],
        sd_ambiguity_id=row["sd_ambiguity_id"],
        base_code_m=row["base_code_m"],
        base_phase_m=row["base_phase_m"],
        rover_code_m=row["rover_code_m"],
        rover_phase_m=row["rover_phase_m"],
        base_tx_pos=row["base_tx_pos"],
        rover_tx_pos=row["rover_tx_pos"],
        pos=row["pos"],
    )


def _epochs(fx):
    return [
        sidereon.RtkEpoch(
            references=[_sat(row) for row in epoch["references"]],
            nonref=[_sat(row) for row in epoch["nonref"]],
            dt_s=epoch["dt_s"],
            velocity_mps=epoch.get("velocity_mps"),
        )
        for epoch in fx["epochs"]
    ]


def _model(fx):
    model = fx["model"]
    stochastic = model["stochastic"]
    return sidereon.RtkMeasurementModel(
        code_sigma_m=model["code_sigma_m"],
        phase_sigma_m=model["phase_sigma_m"],
        sagnac=model["sagnac"],
        stochastic=stochastic["kind"],
        elevation_weighting=stochastic.get("elevation_weighting", False),
    )


def _float_options(fx):
    opts = fx["float_opts"]
    return sidereon.RtkFloatOptions(
        position_tol_m=opts["position_tol_m"],
        ambiguity_tol_m=opts["ambiguity_tol_m"],
        max_iterations=opts["max_iterations"],
    )


def _fixed_options(fx):
    opts = fx["fixed_opts"]
    return sidereon.RtkFixedOptions(
        position_tol_m=opts["position_tol_m"],
        ambiguity_tol_m=opts["ambiguity_tol_m"],
        max_iterations=opts["max_iterations"],
        ratio_threshold=opts["ratio_threshold"],
        partial_ambiguity_resolution=opts["partial_ambiguity_resolution"],
        partial_min_ambiguities=opts["partial_min_ambiguities"],
    )


def _residual_options(fx):
    opts = fx["residual_opts"]
    return sidereon.RtkResidualValidationOptions(
        threshold_sigma=opts["threshold_sigma"],
        max_exclusions=opts["max_exclusions"],
    )


def _integer_status(name):
    return {
        "Fixed": sidereon.IntegerStatus.FIXED,
        "NotFixed": sidereon.IntegerStatus.NOT_FIXED,
    }[name]


def test_rtk_stochastic_model_enum_and_string_alias():
    enum_model = sidereon.RtkMeasurementModel(
        code_sigma_m=0.5,
        phase_sigma_m=0.005,
        stochastic=sidereon.RtkStochasticModel.SIMPLE,
        elevation_weighting=True,
    )
    legacy_model = sidereon.RtkMeasurementModel(
        code_sigma_m=0.5,
        phase_sigma_m=0.005,
        stochastic="simple",
        elevation_weighting=True,
    )

    assert enum_model.stochastic == sidereon.RtkStochasticModel.SIMPLE
    assert legacy_model.stochastic == sidereon.RtkStochasticModel.SIMPLE
    assert enum_model.elevation_weighting
    assert sidereon.RtkStochasticModel.RTKLIB.label == "rtklib"
    assert repr(sidereon.RtkStochasticModel.SIMPLE) == "RtkStochasticModel.SIMPLE"


def test_rtk_float_matches_reference():
    fx = _fixture()
    config = sidereon.RtkFloatConfig(
        epochs=_epochs(fx),
        base=fx["base_arp_m"],
        ambiguity_ids=fx["ambiguity_ids"],
        model=_model(fx),
        initial_baseline_m=fx["initial_baseline_m"],
        options=_float_options(fx),
    )
    sol = sidereon.solve_rtk_float(config)
    expected = np.array(fx["expected"]["float_baseline_m"])
    assert isinstance(sol.baseline, np.ndarray)
    assert sol.baseline.dtype == np.float64
    assert np.array_equal(sol.baseline, expected)
    assert sol.converged
    assert isinstance(sol.geometry_quality, sidereon.GeometryQuality)
    _assert_square_covariance(sol.ambiguity_covariance, len(sol.ambiguities_m))
    assert "RtkFloatSolution(" in repr(sol)


def test_rtk_fixed_matches_reference():
    fx = _fixture()
    config = sidereon.RtkFixedConfig(
        epochs=_epochs(fx),
        base=fx["base_arp_m"],
        ambiguity_ids=fx["ambiguity_ids"],
        ambiguity_satellites=fx["ambiguity_satellites"],
        wavelengths_m=fx["wavelengths_m"],
        offsets_m=fx["offsets_m"],
        model=_model(fx),
        float_options=_float_options(fx),
        fixed_options=_fixed_options(fx),
        residual_options=_residual_options(fx),
        float_only_systems=fx["float_only_systems"],
        initial_baseline_m=fx["initial_baseline_m"],
    )
    sol = sidereon.solve_rtk_fixed(config)
    exp = fx["expected"]
    assert np.array_equal(sol.fixed_baseline, np.array(exp["fixed_baseline_m"]))
    assert np.array_equal(
        sol.float_baseline, np.array(exp["validated_float_baseline_m"])
    )
    assert sol.integer_status == _integer_status(exp["fixed_integer_status"])
    assert "RtkFixedSolution(" in repr(sol)


# --- sequential RTK arc driver ---------------------------------------------
#
# `solve_rtk_arc` is the high-level raw-epochs driver: it selects references
# once, builds the sequential filter, and runs the per-epoch update/search/hold
# loop. Re-shaping the committed static WTZR/WTZZ arc into raw base/rover epochs
# and driving the sequential filter must converge the final reported baseline
# onto the static fixed reference baseline.


def _arc_sd_key(key):
    # The static fixture keys wavelengths/offsets by the double-difference id
    # (`<sd>|ref=G30`); the arc keys them by the single-difference ambiguity id.
    return key.split("|", 1)[0]


def _arc_epochs(fx, lli_by_epoch_side_sat=None):
    lli_by_epoch_side_sat = lli_by_epoch_side_sat or {}
    out = []
    for epoch_index, epoch in enumerate(fx["epochs"]):
        rows = epoch["references"] + epoch["nonref"]
        base = [
            sidereon.RtkArcObservation(
                r["sat"],
                r["sd_ambiguity_id"],
                r["base_code_m"],
                r["base_phase_m"],
                lli=lli_by_epoch_side_sat.get((epoch_index, "base", r["sat"])),
            )
            for r in rows
        ]
        rover = [
            sidereon.RtkArcObservation(
                r["sat"],
                r["sd_ambiguity_id"],
                r["rover_code_m"],
                r["rover_phase_m"],
                lli=lli_by_epoch_side_sat.get((epoch_index, "rover", r["sat"])),
            )
            for r in rows
        ]
        out.append(
            sidereon.RtkArcEpoch(
                base=base,
                rover=rover,
                satellite_positions_m={r["sat"]: r["pos"] for r in rows},
                base_satellite_positions_m={r["sat"]: r["base_tx_pos"] for r in rows},
                rover_satellite_positions_m={r["sat"]: r["rover_tx_pos"] for r in rows},
                velocity_mps=epoch.get("velocity_mps"),
            )
        )
    return out


def _arc_config(fx, preprocessing=None, extra_ambiguity_ids=()):
    wavelengths_m = {_arc_sd_key(k): v for k, v in fx["wavelengths_m"].items()}
    offsets_m = {_arc_sd_key(k): v for k, v in fx["offsets_m"].items()}
    for ambiguity_id in extra_ambiguity_ids:
        sd_id = _arc_sd_key(ambiguity_id)
        sat = sd_id.split("@", 1)[0]
        source_id = sd_id if sd_id in wavelengths_m else sat
        wavelengths_m[ambiguity_id] = wavelengths_m[source_id]
        offsets_m[ambiguity_id] = offsets_m[source_id]
    return sidereon.RtkArcConfig(
        base=fx["base_arp_m"],
        model=_model(fx),
        wavelengths_m=wavelengths_m,
        offsets_m=offsets_m,
        baseline_prior_sigma_m=30.0,
        ambiguity_prior_sigma_m=30.0,
        initial_baseline_m=[0.0, 0.0, 0.0],
        preprocessing=preprocessing,
    )


def test_solve_static_rtk_arc_smoke():
    fx = _fixture()
    config = sidereon.RtkStaticArcConfig(
        _arc_config(fx, extra_ambiguity_ids=fx["wavelengths_m"].keys()),
        float_options=_float_options(fx),
        fixed_options=_fixed_options(fx),
        residual_options=_residual_options(fx),
    )
    sol = sidereon.solve_static_rtk_arc(_arc_epochs(fx), config)

    assert sol.references == {"G": "G30"}
    assert sol.ambiguity_ids
    assert sol.float_solution.converged
    assert sol.fixed_solution.integer_status == _integer_status(
        fx["expected"]["fixed_integer_status"]
    )
    assert np.array_equal(
        sol.fixed_solution.fixed_baseline,
        np.array(fx["expected"]["fixed_baseline_m"]),
    )
    assert isinstance(sol.geometry_quality, sidereon.GeometryQuality)
    assert "RtkStaticArcSolution(" in repr(sol)


def test_rtk_arc_converges_to_static_fixed_baseline():
    fx = _fixture()
    sol = sidereon.solve_rtk_arc(_arc_epochs(fx), _arc_config(fx))

    assert len(sol.epochs) == len(fx["epochs"])
    # Reference is the highest-elevation GPS satellite, selected once for the arc.
    assert sol.references == {"G": "G30"}
    assert sol.final_state.epoch_count == len(fx["epochs"])

    last = sol.epochs[-1]
    assert isinstance(last.reported_baseline, np.ndarray)
    assert last.reported_baseline.dtype == np.float64
    # The sequential arc resolves the integers and rides the static reference.
    assert last.integer_fixed is True
    assert isinstance(last.geometry_quality, sidereon.GeometryQuality)
    expected_fixed = np.array(fx["expected"]["fixed_baseline_m"])
    assert np.allclose(last.reported_baseline_m, expected_fixed, atol=1e-6)
    assert "RtkArcSolution(" in repr(sol)
    assert "RtkArcEpochSolution(" in repr(last)


def test_rtk_arc_is_deterministic():
    fx = _fixture()
    a = sidereon.solve_rtk_arc(_arc_epochs(fx), _arc_config(fx))
    b = sidereon.solve_rtk_arc(_arc_epochs(fx), _arc_config(fx))
    assert a.epochs[-1].reported_baseline_m == b.epochs[-1].reported_baseline_m
    assert a.references == b.references


def test_rtk_arc_preprocessing_metadata_fields_are_exposed():
    fx = _fixture()
    slip_sat = "G05"
    slip_epoch = 3
    lli = {(slip_epoch, "rover", slip_sat): 1}

    obs = sidereon.RtkArcObservation("G01", "G01", 1.0, 2.0, lli=1)
    assert obs.lli == 1

    preprocessing = sidereon.RtkArcPreprocessing(
        cycle_slip="split_arc",
        hatch_window_cap=8,
        elevation_mask_deg=10.0,
    )
    assert preprocessing.cycle_slip == "split_arc"
    assert preprocessing.hatch_window_cap == 8
    assert preprocessing.elevation_mask_deg == 10.0

    split_sol = sidereon.solve_rtk_arc(
        _arc_epochs(fx, lli_by_epoch_side_sat=lli),
        _arc_config(
            fx,
            preprocessing=preprocessing,
            extra_ambiguity_ids=[f"{slip_sat}@rover#1", f"{slip_sat}@rover#2"],
        ),
    )

    assert split_sol.dropped_sats == []
    assert split_sol.elevation_masked_sats == ["G08", "G09", "G18", "G21", "G27"]
    split_arcs = split_sol.split_cycle_slip_arcs
    assert [arc.receiver for arc in split_arcs] == ["rover", "rover"]
    assert [arc.satellite_id for arc in split_arcs] == [slip_sat, slip_sat]
    assert [arc.ambiguity_id for arc in split_arcs] == [
        f"{slip_sat}@rover#1",
        f"{slip_sat}@rover#2",
    ]
    assert split_arcs[0].start_epoch_index == 0
    assert split_arcs[0].end_epoch_index == slip_epoch - 1
    assert split_arcs[1].start_epoch_index == slip_epoch
    assert split_arcs[1].end_epoch_index == len(fx["epochs"]) - 1

    covariance = split_sol.measurement_covariance
    dim = 3 + len(split_sol.final_state.sd_ambiguity_ids)
    assert len(covariance) == dim * dim
    assert all(isinstance(value, float) for value in covariance)
    covariance_matrix = split_sol.measurement_covariance_matrix
    assert covariance_matrix.ndim == 2
    assert len(covariance_matrix) == dim
    assert len(covariance_matrix[0]) == dim
    np.testing.assert_allclose(
        covariance_matrix.ravel(),
        np.array(covariance),
        rtol=0.0,
        atol=0.0,
    )

    drop_sol = sidereon.solve_rtk_arc(
        _arc_epochs(fx, lli_by_epoch_side_sat=lli),
        _arc_config(
            fx,
            preprocessing=sidereon.RtkArcPreprocessing(cycle_slip="drop_satellite"),
        ),
    )
    assert drop_sol.dropped_sats == [slip_sat]
    assert drop_sol.split_cycle_slip_arcs == []
    assert all(slip_sat not in epoch.used_satellite_ids for epoch in drop_sol.epochs)


def test_rtk_arc_update_options_round_trip_getters():
    opts = sidereon.RtkArcUpdateOptions(
        hold_sigma_m=1e-3,
        ratio_threshold=2.5,
        dynamics="velocity_propagated",
        report_residuals=True,
    )
    assert opts.hold_sigma_m == 1e-3
    assert opts.ratio_threshold == 2.5
    assert opts.report_residuals is True


def test_rtk_arc_rejects_unknown_dynamics():
    with pytest.raises(ValueError):
        sidereon.RtkArcUpdateOptions(dynamics="bogus")


def _arc_config_with_screen(fx, threshold_sigma):
    options = sidereon.RtkArcUpdateOptions(
        innovation_threshold_sigma=threshold_sigma,
        innovation_min_rows=1,
    )
    return sidereon.RtkArcConfig(
        base=fx["base_arp_m"],
        model=_model(fx),
        wavelengths_m={_arc_sd_key(k): v for k, v in fx["wavelengths_m"].items()},
        offsets_m={_arc_sd_key(k): v for k, v in fx["offsets_m"].items()},
        baseline_prior_sigma_m=30.0,
        ambiguity_prior_sigma_m=30.0,
        initial_baseline_m=[0.0, 0.0, 0.0],
        update_options=options,
    )


def test_rtk_arc_innovation_screen_is_absent_by_default():
    fx = _fixture()
    sol = sidereon.solve_rtk_arc(_arc_epochs(fx), _arc_config(fx))
    # No screen configured: every epoch carries no screen diagnostics.
    assert all(epoch.innovation_screen is None for epoch in sol.epochs)


def test_rtk_arc_exposes_innovation_screen_when_enabled():
    fx = _fixture()
    sol = sidereon.solve_rtk_arc(
        _arc_epochs(fx), _arc_config_with_screen(fx, threshold_sigma=6.0)
    )

    screens = [epoch.innovation_screen for epoch in sol.epochs]
    assert all(screen is not None for screen in screens)

    for screen in screens:
        assert screen.threshold_sigma == 6.0
        assert screen.min_rows == 1
        # The row accounting is internally consistent per epoch.
        assert screen.input_rows == screen.accepted_rows + screen.rejected_rows
        assert screen.rejected_rows == (
            screen.rejected_code_rows + screen.rejected_phase_rows
        )
        assert isinstance(screen.coasted, bool)
        assert "RtkArcInnovationScreen(" in repr(screen)

    # The screen actually ran: at least one epoch presented rows to it, and the
    # optional largest-innovation diagnostic surfaces as a float when rows exist.
    assert any(screen.input_rows > 0 for screen in screens)
    populated = next(screen for screen in screens if screen.input_rows > 0)
    assert populated.max_abs_normalized_innovation is not None


def test_rtk_arc_rejects_empty_arc():
    with pytest.raises(sidereon.SolveError):
        sidereon.solve_rtk_arc([], _arc_config(_fixture()))


def test_rinex_rtk_arc_options_round_trip_getters():
    pair = sidereon.RtkRinexSignalPair.gps_l1_c()
    assert pair.system == sidereon.GnssSystem.GPS
    assert pair.code_observable == "C1C"
    assert pair.phase_observable == "L1C"

    opts = sidereon.RtkRinexArcOptions(
        signal_pairs=[pair],
        max_epochs=12,
        min_common_satellites=5,
        include_prediction_time=False,
    )
    assert opts.signal_pairs[0].code_observable == "C1C"
    assert opts.max_epochs == 12
    assert opts.min_common_satellites == 5
    assert opts.include_prediction_time is False

    dual_pair = sidereon.RtkRinexDualSignalPair.gps_l1_l2_cw()
    assert dual_pair.system == sidereon.GnssSystem.GPS
    assert dual_pair.code1_observable == "C1C"
    assert dual_pair.phase1_observable == "L1C"
    assert dual_pair.code2_observable == "C2W"
    assert dual_pair.phase2_observable == "L2W"

    dual_opts = sidereon.RtkRinexDualArcOptions(
        signal_pairs=[dual_pair],
        max_epochs=12,
        min_common_satellites=5,
        include_prediction_time=False,
    )
    assert dual_opts.signal_pairs[0].code2_observable == "C2W"
    assert dual_opts.max_epochs == 12
    assert dual_opts.min_common_satellites == 5
    assert dual_opts.include_prediction_time is False


def test_rinex_rtk_static_convenience_solves_real_wettzell_arc():
    sp3, base_obs, rover_obs, base_arp_m, truth_baseline_m = _wettzell_rinex_inputs()
    arc_options = sidereon.RtkRinexArcOptions(
        max_epochs=120,
        include_prediction_time=False,
    )

    arc = sidereon.build_rinex_rtk_arc(sp3, base_obs, rover_obs, arc_options)
    assert len(arc.epochs) == 120
    assert arc.skipped_epoch_count == 0
    assert arc.wavelengths_m
    assert set(arc.offsets_m.values()) == {0.0}

    solution = sidereon.solve_static_rinex_rtk_baseline(
        sp3,
        base_obs,
        rover_obs,
        base_arp_m.tolist(),
        model=_real_arc_model(),
        arc_options=arc_options,
        preprocessing=sidereon.RtkArcPreprocessing(cycle_slip="split_arc"),
        float_options=_real_arc_float_options(),
        fixed_options=_real_arc_fixed_options(),
    )

    assert solution.references == {"G": "G30"}
    assert len(solution.split_cycle_slip_arcs) == 4
    assert solution.float_solution.converged
    assert _vector_error_m(solution.float_solution.baseline, truth_baseline_m) < 0.08
    _assert_square_covariance(
        solution.float_solution.ambiguity_covariance,
        len(solution.float_solution.ambiguities_m),
    )

    fixed = solution.fixed_solution
    assert fixed.integer_status == sidereon.IntegerStatus.NOT_FIXED
    assert fixed.integer_ratio is not None
    assert fixed.integer_ratio < 3.0
    assert _vector_error_m(fixed.fixed_baseline, truth_baseline_m) < 0.01


def test_rinex_wide_lane_fixed_convenience_solves_real_wettzell_arc():
    sp3, base_obs, rover_obs, base_arp_m, truth_baseline_m = _wettzell_rinex_inputs()
    arc_options = sidereon.RtkRinexDualArcOptions(
        max_epochs=120,
        include_prediction_time=False,
    )

    arc = sidereon.build_dual_frequency_rinex_rtk_arc(
        sp3,
        base_obs,
        rover_obs,
        arc_options,
    )
    assert len(arc.epochs) == 120
    assert arc.skipped_epoch_count == 0

    solution = sidereon.solve_wide_lane_fixed_rinex_rtk_baseline(
        sp3,
        base_obs,
        rover_obs,
        base_arp_m.tolist(),
        model=_real_arc_model(),
        arc_options=arc_options,
        float_options=_real_arc_float_options(),
        fixed_options=_real_arc_fixed_options(),
    )

    assert solution.wide_lane_fixed
    assert solution.integer_status == sidereon.IntegerStatus.FIXED
    assert solution.integer_ratio is not None
    assert solution.integer_ratio > 3.0
    assert solution.wide_lane_ambiguities_cycles
    assert _vector_error_m(solution.fixed_baseline, truth_baseline_m) < 0.01
    assert _vector_error_m(solution.float_baseline, truth_baseline_m) < 0.1
    _assert_square_covariance(
        solution.float_ambiguity_covariance,
        len(solution.solution.float_solution.ambiguities_m),
    )


_F_L1_HZ = 1_575_420_000.0
_F_L2_HZ = 1_227_600_000.0
_DUAL_BASE_M = [3_512_900.0, 780_500.0, 5_248_700.0]
_DUAL_POSITIONS_M = {
    "G01": [14_350_000.0, 3_190_000.0, 21_440_000.0],
    "G02": [20_000_000.0, 3_000_000.0, 18_000_000.0],
    "G03": [9_000_000.0, 9_000_000.0, 22_000_000.0],
    "G04": [16_000_000.0, -4_000_000.0, 21_000_000.0],
}


def _dual_frequency_observation(ambiguity_id, p1_m, p2_m, phi1_cycles):
    return sidereon.RtkDualFrequencyObservation(
        ambiguity_id=ambiguity_id,
        p1_m=p1_m,
        p2_m=p2_m,
        phi1_cycles=phi1_cycles,
        phi2_cycles=0.0,
        f1_hz=_F_L1_HZ,
        f2_hz=_F_L2_HZ,
    )


def _dual_frequency_satellite(
    satellite_id,
    base_p1_m,
    base_p2_m,
    base_phi1_cycles,
    rover_p1_m,
    rover_p2_m,
    rover_phi1_cycles,
):
    return sidereon.RtkDualFrequencySatelliteObservation(
        satellite_id=satellite_id,
        base=_dual_frequency_observation(
            satellite_id,
            base_p1_m,
            base_p2_m,
            base_phi1_cycles,
        ),
        rover=_dual_frequency_observation(
            satellite_id,
            rover_p1_m,
            rover_p2_m,
            rover_phi1_cycles,
        ),
    )


def _dual_frequency_observations():
    return [
        _dual_frequency_satellite(
            "G01", 20_000_020.0, 20_000_022.0, 2.0, 20_000_050.0, 20_000_052.5, 5.0
        ),
        _dual_frequency_satellite(
            "G02", 20_000_010.0, 20_000_012.0, 1.0, 20_000_042.0, 20_000_044.5, 7.0
        ),
        _dual_frequency_satellite(
            "G03", 19_999_980.0, 19_999_982.0, -2.0, 20_000_005.0, 20_000_007.5, 0.0
        ),
        _dual_frequency_satellite(
            "G04", 20_000_040.0, 20_000_042.0, 4.0, 20_000_073.0, 20_000_075.5, 8.0
        ),
    ]


def _dual_frequency_arc():
    return [
        sidereon.RtkDualFrequencyArcEpoch(
            jd_whole=2_460_100.5,
            jd_fraction=jd_fraction,
            epoch_sort_key=f"{idx:03}",
            gap_time_s=float(idx),
            observations=_dual_frequency_observations(),
            satellite_positions_m=_DUAL_POSITIONS_M,
        )
        for idx, jd_fraction in enumerate([0.25, 0.251, 0.252])
    ]


def _wide_lane_arc_config(cycle_slip=None):
    return sidereon.RtkWideLaneArcConfig(
        base=_DUAL_BASE_M,
        options=sidereon.RtkWideLaneOptions(
            min_epochs=2,
            tolerance_cycles=0.5,
            skip_short_fragments=False,
        ),
        cycle_slip=cycle_slip,
    )


def test_fix_wide_lane_rtk_arc_smoke():
    config = _wide_lane_arc_config()
    assert config.options.min_epochs == 2

    sol = sidereon.fix_wide_lane_rtk_arc(_dual_frequency_arc(), config)

    assert sol.references == {"G": "G01"}
    assert sol.wide_lane_cycles
    assert len(sol.epochs) == 3
    assert sol.dropped_sats == []
    assert sol.split_cycle_slip_arcs == []
    assert isinstance(sol.geometry_quality, sidereon.GeometryQuality)
    assert "RtkWideLaneArcSolution(" in repr(sol)


def test_prepare_ionosphere_free_rtk_arc_smoke():
    epochs = _dual_frequency_arc()
    wide_lane = sidereon.fix_wide_lane_rtk_arc(epochs, _wide_lane_arc_config())
    config = sidereon.RtkIonosphereFreeArcConfig(
        base=_DUAL_BASE_M,
        initial_baseline_m=[0.0, 0.0, 0.0],
    )

    sol = sidereon.prepare_ionosphere_free_rtk_arc(
        epochs,
        wide_lane.wide_lane_cycles,
        config,
    )

    assert sol.references == wide_lane.references
    assert len(sol.epochs) == len(epochs)
    assert sol.epochs[0].base_count == 4
    assert sol.wavelengths_m
    assert sol.offsets_m
    assert "RtkIonosphereFreeArcSolution(" in repr(sol)
