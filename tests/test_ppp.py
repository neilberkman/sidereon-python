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
    )


def _integer_status(name):
    return {
        "Fixed": sidereon.IntegerStatus.FIXED,
        "NotFixed": sidereon.IntegerStatus.NOT_FIXED,
    }[name]


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
    assert np.array_equal(sol.position, expected)
    assert sol.converged
    assert len(sol.used_sats) > 0
    assert "PppFloatSolution(" in repr(sol)


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
    assert np.array_equal(sol.position, np.array(exp["fixed_position_m"]))
    assert np.array_equal(
        sol.float_solution.position,
        np.array(exp["fixed_float_position_m"]),
    )
    assert sol.integer_status == _integer_status(exp["fixed_integer_status"])
    assert sol.integer_ratio == exp["fixed_integer_ratio"]
    assert sol.integer_candidates == exp["fixed_integer_candidates"]
    assert sol.fixed_ambiguities_cycles == exp["fixed_ambiguities_cycles"]
    assert sol.fixed_ambiguities_m == exp["fixed_ambiguities_m"]
    assert "PppFixedSolution(" in repr(sol)


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
