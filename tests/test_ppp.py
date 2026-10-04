"""Static float PPP solve through the binding reproduces the engine.

The fixture `ppp_esbc.json` is emitted by the crate's ESBC troposphere-corrected
float-PPP integration test (`SIDEREON_DUMP_FIXTURES=1 cargo test --test
ppp_real_arc ...`); it carries the built epoch arc, initial state and the
troposphere-corrected config. `scripts/ppp_esbc_expected` then solves the arc
with the core entry points the binding calls, from inputs built as the Python
constructors build them, and writes each solution into `expected`. The binding
loads the same committed SP3 product and must return those solutions bit for
bit.
"""

import json
import os
import struct

import numpy as np
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES


def _load_fixture():
    with open(os.path.join(FIXTURES, "ppp_esbc.json")) as fh:
        return json.load(fh)


def _load_sp3(fx):
    sp3_path = os.path.join(CORE_FIXTURES, "sp3", fx["sp3_file"])
    with open(sp3_path, "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _epochs(fx):
    out = []
    for epoch in fx["epochs"]:
        civil = sidereon.PppCivilDateTime(
            epoch["civil"]["year"],
            epoch["civil"]["month"],
            epoch["civil"]["day"],
            epoch["civil"]["hour"],
            epoch["civil"]["minute"],
            epoch["civil"]["second"],
        )
        observations = [
            sidereon.PppObservation(
                satellite_id=obs["satellite_id"],
                ambiguity_id=obs["ambiguity_id"],
                code_m=obs["code_m"],
                phase_m=obs["phase_m"],
                freq1_hz=obs["freq1_hz"],
                freq2_hz=obs["freq2_hz"],
            )
            for obs in epoch["observations"]
        ]
        out.append(
            sidereon.PppEpoch(
                civil,
                epoch["jd_whole"],
                epoch["jd_fraction"],
                epoch["t_rx_j2000_s"],
                observations,
            )
        )
    return out


def _state(fx):
    state = fx["initial_state"]
    return sidereon.PppFloatState(
        position_m=state["position_m"],
        clocks_m=state["clocks_m"],
        ambiguities_m=dict(state["ambiguities_m"]),
        ztd_m=state["ztd_m"],
        tropo_gradient_north_m=state.get("tropo_gradient_north_m", 0.0),
        tropo_gradient_east_m=state.get("tropo_gradient_east_m", 0.0),
        residual_ionosphere_m=state.get("residual_ionosphere_m"),
    )


def _weights(raw):
    return sidereon.PppMeasurementWeights(
        code=raw["code"],
        phase=raw["phase"],
        elevation_weighting=raw["elevation_weighting"],
    )


def _tropo(raw):
    return sidereon.PppTroposphereOptions(
        enabled=raw["enabled"],
        estimate_ztd=raw["estimate_ztd"],
        estimate_tropo_gradients=raw.get("estimate_tropo_gradients", False),
        pressure_hpa=raw["pressure_hpa"],
        temperature_k=raw["temperature_k"],
        relative_humidity=raw["relative_humidity"],
    )


def _options(raw):
    return sidereon.PppFloatOptions(
        max_iterations=raw["max_iterations"],
        position_tolerance_m=raw["position_tolerance_m"],
        clock_tolerance_m=raw["clock_tolerance_m"],
        ambiguity_tolerance_m=raw["ambiguity_tolerance_m"],
        ztd_tolerance_m=raw["ztd_tolerance_m"],
    )


def _float_config(fx):
    config = fx["config"]
    return sidereon.PppFloatConfig(
        weights=_weights(config["weights"]),
        tropo=_tropo(config["tropo"]),
        options=_options(config["opts"]),
        residual_screen=config["residual_screen"],
        elevation_cutoff_deg=config.get("elevation_cutoff_deg"),
        estimate_residual_ionosphere=config.get("estimate_residual_ionosphere", False),
    )


def _fixed_config(fx):
    config = fx["fixed_config"]
    ambiguity = config["ambiguity"]
    return sidereon.PppFixedConfig(
        ambiguity=sidereon.PppFixedAmbiguityOptions(
            wavelengths_m=ambiguity["wavelengths_m"],
            offsets_m=ambiguity["offsets_m"],
            ratio_threshold=ambiguity["ratio_threshold"],
        ),
        weights=_weights(config["weights"]),
        tropo=_tropo(config["tropo"]),
        options=_options(config["opts"]),
        elevation_cutoff_deg=config.get("elevation_cutoff_deg"),
        estimate_residual_ionosphere=config.get("estimate_residual_ionosphere", False),
    )


def _bits(value):
    return "0x" + struct.pack(">d", float(value)).hex()


def _assert_position_bits(position, expected_bits):
    assert [_bits(value) for value in position] == expected_bits


def _assert_outcome(sol, expected):
    """The solve outcome `scripts/ppp_esbc_expected` writes for a solution."""
    _assert_position_bits(sol.position, expected["position_bits"])
    assert sol.status.label == expected["status"]
    assert sol.iterations == expected["iterations"]
    assert sol.converged is expected["converged"]
    assert [_bits(clock) for clock in sol.epoch_clocks_m] == expected[
        "epoch_clocks_bits"
    ]
    assert sol.solved_epoch_indices == expected["solved_epoch_indices"]
    assert [
        {
            "epoch_index": row.epoch_index,
            "satellite_id": row.satellite_id,
            "ambiguity_id": row.ambiguity_id,
            "reason": row.reason,
        }
        for row in sol.unplaced_observations
    ] == expected["unplaced_observations"]
    assert len(sol.residuals_m) == expected["residuals"]["count"]
    assert [
        {
            "epoch_index": row.epoch_index,
            "satellite_id": row.satellite_id,
            "ambiguity_id": row.ambiguity_id,
            "code_m": _bits(row.code_m),
            "phase_m": _bits(row.phase_m),
            "code_weight": _bits(row.code_weight),
            "phase_weight": _bits(row.phase_weight),
        }
        for row in sol.residuals_m[:5]
    ] == expected["residuals"]["first"]
    assert len(sol.ssr_bias_exclusions) == expected["ssr_bias_exclusion_count"]
    if "residual_screen_removals" in expected:
        assert [list(row) for row in sol.residual_screen_removals] == expected[
            "residual_screen_removals"
        ]
        assert sol.residual_screen is expected["residual_screen"]


def _integer_status(name):
    return {
        "Fixed": sidereon.IntegerStatus.FIXED,
        "NotFixed": sidereon.IntegerStatus.NOT_FIXED,
    }[name]


def _assert_matrix_close(actual, expected):
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    assert np.array_equal(actual, np.array(expected))


def _assert_temporal_correlation(actual, expected):
    assert actual.lag1_autocorrelation == expected["lag1_autocorrelation"]
    assert actual.decorrelation_time_epochs == expected["decorrelation_time_epochs"]
    assert actual.decorrelation_time_s == expected["decorrelation_time_s"]
    assert actual.nominal_sample_count == expected["nominal_sample_count"]
    assert actual.effective_sample_count == expected["effective_sample_count"]
    assert actual.variance_inflation_factor == expected["variance_inflation_factor"]
    assert actual.arcs_used == expected["arcs_used"]


def _assert_ppp_metadata(sol, expected):
    _assert_matrix_close(
        sol.position_covariance_ecef_m2,
        expected["position_covariance_ecef_m2"],
    )
    _assert_matrix_close(
        sol.position_covariance_enu_m2,
        expected["position_covariance_enu_m2"],
    )
    _assert_matrix_close(
        sol.formal_position_covariance_ecef_m2,
        expected["formal_position_covariance_ecef_m2"],
    )
    _assert_matrix_close(
        sol.formal_position_covariance_enu_m2,
        expected["formal_position_covariance_enu_m2"],
    )
    _assert_matrix_close(
        sol.temporal_position_covariance_ecef_m2,
        expected["temporal_position_covariance_ecef_m2"],
    )
    _assert_matrix_close(
        sol.temporal_position_covariance_enu_m2,
        expected["temporal_position_covariance_enu_m2"],
    )
    assert sol.posterior_variance_factor == expected["posterior_variance_factor"]
    assert (
        sol.position_covariance_scale_factor
        == expected["position_covariance_scale_factor"]
    )
    assert (
        sol.temporal_position_covariance_scale_factor
        == expected["temporal_position_covariance_scale_factor"]
    )
    _assert_temporal_correlation(
        sol.temporal_correlation,
        expected["temporal_correlation"],
    )
    assert sol.tropo_gradient_north_m == expected["tropo_gradient_north_m"]
    assert sol.tropo_gradient_east_m == expected["tropo_gradient_east_m"]
    if expected.get("tropo_gradient_covariance_m2") is None:
        assert sol.tropo_gradient_covariance_m2 is None
    else:
        _assert_matrix_close(
            sol.tropo_gradient_covariance_m2,
            expected["tropo_gradient_covariance_m2"],
        )
    if expected.get("formal_tropo_gradient_covariance_m2") is None:
        assert sol.formal_tropo_gradient_covariance_m2 is None
    else:
        _assert_matrix_close(
            sol.formal_tropo_gradient_covariance_m2,
            expected["formal_tropo_gradient_covariance_m2"],
        )
    assert sol.residual_ionosphere_m == expected["residual_ionosphere_m"]


def test_ppp_float_matches_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)

    sol = sidereon.solve_ppp_float(
        sp3,
        epochs=_epochs(fx),
        initial_state=_state(fx),
        config=_float_config(fx),
    )
    assert isinstance(sol.position, np.ndarray)
    assert sol.position.dtype == np.float64
    assert sol.converged
    assert len(sol.used_sats) > 0
    assert "PppFloatSolution(" in repr(sol)
    _assert_ppp_metadata(sol, fx["expected"]["float_solution"])
    _assert_outcome(sol, fx["expected"]["float_solution"])
    options = sol.solve_options
    opts = fx["config"]["opts"]
    assert options.max_iterations == opts["max_iterations"]
    assert options.position_tolerance_m == opts["position_tolerance_m"]
    assert options.clock_tolerance_m == opts["clock_tolerance_m"]
    assert options.ambiguity_tolerance_m == opts["ambiguity_tolerance_m"]
    assert options.ztd_tolerance_m == opts["ztd_tolerance_m"]


def test_ppp_fixed_matches_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    epochs = _epochs(fx)
    float_sol = sidereon.solve_ppp_float(
        sp3,
        epochs=epochs,
        initial_state=_state(fx),
        config=_float_config(fx),
    )

    sol = sidereon.solve_ppp_fixed(
        sp3,
        epochs=epochs,
        float_solution=float_sol,
        config=_fixed_config(fx),
    )

    exp = fx["expected"]
    assert isinstance(sol.position, np.ndarray)
    assert sol.position.dtype == np.float64
    _assert_outcome(sol, exp["fixed_solution"])
    _assert_outcome(sol.float_solution, exp["fixed_float_solution"])
    assert sol.integer_status == _integer_status(exp["fixed_integer_status"])
    assert _bits(sol.integer_ratio) == exp["fixed_solution"]["integer_ratio_bits"]
    assert sol.integer_candidates == exp["fixed_integer_candidates"]
    assert sol.fixed_ambiguities_cycles == exp["fixed_ambiguities_cycles"]
    assert sol.fixed_ambiguities_m == exp["fixed_ambiguities_m"]
    assert "PppFixedSolution(" in repr(sol)
    _assert_ppp_metadata(sol, exp["fixed_solution"])
    _assert_ppp_metadata(sol.float_solution, exp["fixed_float_solution"])


def test_ppp_023_option_contracts():
    state = sidereon.PppFloatState(
        position_m=[1.0, 2.0, 3.0],
        clocks_m=[4.0],
        ambiguities_m={"G01": 5.0},
        ztd_m=0.2,
        tropo_gradient_north_m=0.03,
        tropo_gradient_east_m=-0.04,
        residual_ionosphere_m={"G01": 0.5},
    )
    assert state.tropo_gradient_north_m == 0.03
    assert state.tropo_gradient_east_m == -0.04
    assert state.residual_ionosphere_m == {"G01": 0.5}

    tropo = sidereon.PppTroposphereOptions(
        enabled=True,
        estimate_ztd=True,
        estimate_tropo_gradients=True,
    )
    assert tropo.estimate_tropo_gradients is True

    float_config = sidereon.PppFloatConfig(
        tropo=tropo,
        elevation_cutoff_deg=12.5,
        estimate_residual_ionosphere=True,
    )
    assert float_config.elevation_cutoff_deg == 12.5
    assert float_config.estimate_residual_ionosphere is True

    fixed_config = sidereon.PppFixedConfig(
        sidereon.PppFixedAmbiguityOptions({"G01": 0.19}, {"G01": 0.0}),
        elevation_cutoff_deg=10.0,
        estimate_residual_ionosphere=True,
    )
    assert fixed_config.elevation_cutoff_deg == 10.0
    assert fixed_config.estimate_residual_ionosphere is True


def test_ppp_elevation_cutoff_matches_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    config = _float_config(fx)
    config = sidereon.PppFloatConfig(
        weights=_weights(fx["config"]["weights"]),
        tropo=_tropo(fx["config"]["tropo"]),
        options=_options(fx["config"]["opts"]),
        residual_screen=config.residual_screen,
        elevation_cutoff_deg=10.0,
    )

    sol = sidereon.solve_ppp_float(
        sp3,
        epochs=_epochs(fx),
        initial_state=_state(fx),
        config=config,
    )

    expected = fx["expected"]["float_elevation_cutoff_10_deg"]
    _assert_position_bits(sol.position, expected["position_bits"])
    assert len(sol.used_sats) == expected["used_sat_count"]


def test_ppp_tropo_gradients_match_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    config = sidereon.PppFloatConfig(
        weights=_weights(fx["config"]["weights"]),
        tropo=sidereon.PppTroposphereOptions(
            enabled=fx["config"]["tropo"]["enabled"],
            estimate_ztd=fx["config"]["tropo"]["estimate_ztd"],
            estimate_tropo_gradients=True,
            pressure_hpa=fx["config"]["tropo"]["pressure_hpa"],
            temperature_k=fx["config"]["tropo"]["temperature_k"],
            relative_humidity=fx["config"]["tropo"]["relative_humidity"],
        ),
        options=_options(fx["config"]["opts"]),
        residual_screen=fx["config"]["residual_screen"],
    )

    sol = sidereon.solve_ppp_float(
        sp3,
        epochs=_epochs(fx),
        initial_state=_state(fx),
        config=config,
    )

    expected = fx["expected"]["float_tropo_gradients"]
    _assert_ppp_metadata(sol, expected)
    _assert_outcome(sol, expected)


# --- SPP-seeded auto-initialization drivers --------------------------------
#
# `solve_ppp_auto_init_float` / `solve_ppp_auto_init_fixed` seed the float state
# from a per-epoch SPP solve instead of taking an explicit `PppFloatState`.
# `scripts/ppp_esbc_expected` runs the same auto-init solves through the core.


def test_ppp_auto_init_float_recovers_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    sol = sidereon.solve_ppp_auto_init_float(
        sp3,
        epochs=_epochs(fx),
        config=_float_config(fx),
    )
    assert sol.converged
    _assert_outcome(sol, fx["expected"]["auto_init_float"])
    assert len(sol.used_sats) > 0


def test_ppp_auto_init_empty_epochs_preserves_typed_error_detail():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)

    try:
        sidereon.solve_ppp_auto_init_float(sp3, epochs=[], config=_float_config(fx))
    except sidereon.SolveError as error:
        assert str(error) == "PPP auto-init requires at least one epoch"
        assert error.detail == {
            "family": "PppAutoInitError",
            "kind": "empty_epochs",
            "message": "PPP auto-init requires at least one epoch",
        }
    else:
        raise AssertionError("empty PPP arc unexpectedly solved")


def test_ppp_auto_init_fixed_recovers_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    sol = sidereon.solve_ppp_auto_init_fixed(
        sp3,
        epochs=_epochs(fx),
        float_config=_float_config(fx),
        fixed_config=_fixed_config(fx),
    )
    exp = fx["expected"]["auto_init_fixed"]
    _assert_outcome(sol, exp)
    _assert_outcome(sol.float_solution, exp["float_solution"])
    assert sol.integer_status == _integer_status(exp["integer_status"])
    assert _bits(sol.integer_ratio) == exp["integer_ratio_bits"]


def test_ppp_auto_init_explicit_guess_matches_spp_seed():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    # Seed the auto-init with the auto-init float position explicitly.
    spp_sol = sidereon.solve_ppp_auto_init_float(
        sp3, epochs=_epochs(fx), config=_float_config(fx)
    )
    options = sidereon.PppAutoInitOptions(
        initial_guess_position_m=list(spp_sol.position),
        initial_guess_clock_m=0.0,
    )
    assert options.initial_guess_position_m == list(spp_sol.position)
    assert options.initial_guess_clock_m == 0.0
    sol = sidereon.solve_ppp_auto_init_float(
        sp3,
        epochs=_epochs(fx),
        config=_float_config(fx),
        options=options,
    )
    assert sol.converged
    _assert_outcome(sol, fx["expected"]["auto_init_explicit_guess"])


def test_ppp_auto_init_options_defaults():
    options = sidereon.PppAutoInitOptions()
    assert options.initial_guess_position_m is None
    assert options.spp_troposphere is False
    assert list(options.spp_initial_guess) == [0.0, 0.0, 0.0, 0.0]


def test_ppp_unplaced_codes_are_listed_and_their_epochs_left_unsolved():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    # As `scripts/ppp_esbc_expected` builds it: the first observation of epoch 3
    # reads 0.0 and every observation of epoch 7 reads -1.0.
    for index, obs in enumerate(fx["epochs"][3]["observations"]):
        if index == 0:
            obs["code_m"] = 0.0
    for obs in fx["epochs"][7]["observations"]:
        obs["code_m"] = -1.0
    epochs = _epochs(fx)
    raw = fx["config"]
    config = sidereon.PppFloatConfig(
        weights=_weights(raw["weights"]),
        tropo=_tropo(raw["tropo"]),
        options=_options(raw["opts"]),
        residual_screen=True,
    )

    sol = sidereon.solve_ppp_float(
        sp3, epochs=epochs, initial_state=_state(fx), config=config
    )
    expected = fx["expected"]["unplaced_code"]
    _assert_outcome(sol, expected)

    rows = sol.unplaced_observations
    assert len(rows) == 1 + len(fx["epochs"][7]["observations"])
    first = rows[0]
    assert (first.epoch_index, first.satellite_id, first.ambiguity_id) == (
        3,
        fx["epochs"][3]["observations"][0]["satellite_id"],
        fx["epochs"][3]["observations"][0]["ambiguity_id"],
    )
    assert first.reason == "code_not_positive"
    assert "code is zero or negative" in first.message
    assert "PppUnplacedObservation(" in repr(first)
    assert {row.epoch_index for row in rows[1:]} == {7}
    assert 7 not in sol.solved_epoch_indices
    assert len(sol.epoch_clocks_m) == len(sol.solved_epoch_indices)

    fixed = sidereon.solve_ppp_fixed(
        sp3, epochs=epochs, float_solution=sol, config=_fixed_config(fx)
    )
    _assert_outcome(fixed, expected["fixed_solution"])


def test_ppp_solution_record_classes():
    assert sidereon.PppFloatStatus.STATE_TOLERANCE.label == "state_tolerance"
    assert sidereon.PppFloatStatus.MAX_ITERATIONS.label == "max_iterations"
    for name in (
        "PppFloatResidual",
        "PppSsrBiasExclusion",
        "PppSsrTransmitTimeFailure",
        "PppSsrObservationApplication",
        "PppSsrSignalReport",
    ):
        assert hasattr(sidereon, name)
