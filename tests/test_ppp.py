"""Static float PPP solve through the binding reproduces the engine.

The fixture `ppp_esbc.json` is emitted by the crate's ESBC troposphere-corrected
float-PPP integration test (`SIDEREON_DUMP_FIXTURES=1 cargo test --test
ppp_real_arc ...`); it carries the built epoch arc, initial state, the
troposphere-corrected config, and the engine's reference position. The binding
loads the same committed SP3 product and must return the identical position.
"""

import json
import os

import numpy as np
import pytest
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


def _integer_status(name):
    return {
        "Fixed": sidereon.IntegerStatus.FIXED,
        "NotFixed": sidereon.IntegerStatus.NOT_FIXED,
    }[name]


def _assert_matrix_close(actual, expected):
    assert isinstance(actual, np.ndarray)
    assert actual.dtype == np.float64
    assert np.allclose(actual, np.array(expected), rtol=0.0, atol=1.0e-12)


def _assert_temporal_correlation(actual, expected):
    assert actual.lag1_autocorrelation == pytest.approx(
        expected["lag1_autocorrelation"], abs=1.0e-15
    )
    assert actual.decorrelation_time_epochs == pytest.approx(
        expected["decorrelation_time_epochs"], abs=1.0e-15
    )
    assert actual.decorrelation_time_s == expected["decorrelation_time_s"]
    assert actual.nominal_sample_count == expected["nominal_sample_count"]
    assert actual.effective_sample_count == pytest.approx(
        expected["effective_sample_count"], abs=1.0e-15
    )
    assert actual.variance_inflation_factor == pytest.approx(
        expected["variance_inflation_factor"], abs=1.0e-15
    )
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
    assert sol.posterior_variance_factor == pytest.approx(
        expected["posterior_variance_factor"], abs=1.0e-15
    )
    assert sol.position_covariance_scale_factor == pytest.approx(
        expected["position_covariance_scale_factor"], abs=1.0e-15
    )
    assert sol.temporal_position_covariance_scale_factor == pytest.approx(
        expected["temporal_position_covariance_scale_factor"], abs=1.0e-15
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
    expected = np.array(fx["expected"]["position_m"])
    assert isinstance(sol.position, np.ndarray)
    assert sol.position.dtype == np.float64
    # Static PPP eliminates per-epoch clocks via Schur reduction (0.22), which is
    # equivalent to the old dense solve to ~1e-9 m but not bit-identical. A
    # micrometre tolerance is far below any requirement and above formulation noise.
    assert np.allclose(sol.position, expected, rtol=0.0, atol=1.0e-6)
    assert sol.converged
    assert len(sol.used_sats) > 0
    assert "PppFloatSolution(" in repr(sol)
    _assert_ppp_metadata(sol, fx["expected"]["float_solution"])


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
    assert np.allclose(
        sol.position, np.array(exp["fixed_position_m"]), rtol=0.0, atol=1.0e-6
    )
    assert np.allclose(
        sol.float_solution.position,
        np.array(exp["fixed_float_position_m"]),
        rtol=0.0,
        atol=1.0e-6,
    )
    assert sol.integer_status == _integer_status(exp["fixed_integer_status"])
    assert sol.integer_ratio == pytest.approx(exp["fixed_integer_ratio"], rel=1.0e-6)
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
    assert np.allclose(sol.position, np.array(expected["position_m"]), atol=1.0e-6)
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
    assert np.allclose(sol.position, np.array(expected["position_m"]), atol=1.0e-6)
    _assert_ppp_metadata(sol, expected)


# --- SPP-seeded auto-initialization drivers --------------------------------
#
# `solve_ppp_auto_init_float` / `solve_ppp_auto_init_fixed` seed the float state
# from a per-epoch SPP solve instead of taking an explicit `PppFloatState`. The
# float solve converges to the same data-determined optimum, so the auto-init
# result reproduces the explicitly-seeded reference position.


def test_ppp_auto_init_float_recovers_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    sol = sidereon.solve_ppp_auto_init_float(
        sp3,
        epochs=_epochs(fx),
        config=_float_config(fx),
    )
    expected = np.array(fx["expected"]["position_m"])
    assert sol.converged
    assert np.allclose(sol.position, expected, atol=1e-6)
    assert len(sol.used_sats) > 0


def test_ppp_auto_init_fixed_recovers_reference():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    sol = sidereon.solve_ppp_auto_init_fixed(
        sp3,
        epochs=_epochs(fx),
        float_config=_float_config(fx),
        fixed_config=_fixed_config(fx),
    )
    exp = fx["expected"]
    assert np.allclose(sol.position, np.array(exp["fixed_position_m"]), atol=1e-6)
    assert np.allclose(
        sol.float_solution.position, np.array(exp["position_m"]), atol=1e-6
    )
    assert sol.integer_status == _integer_status(exp["fixed_integer_status"])


def test_ppp_auto_init_explicit_guess_matches_spp_seed():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    # Seed the auto-init from the SPP-derived solution explicitly; with the same
    # converged optimum, the position still recovers the reference.
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
    assert np.allclose(sol.position, np.array(fx["expected"]["position_m"]), atol=1e-6)


def test_ppp_auto_init_options_defaults():
    options = sidereon.PppAutoInitOptions()
    assert options.initial_guess_position_m is None
    assert options.spp_troposphere is False
    assert list(options.spp_initial_guess) == [0.0, 0.0, 0.0, 0.0]
