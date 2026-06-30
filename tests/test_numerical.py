"""Numerical state-vector propagation reproduces the engine bit-for-bit.

The fixture ``numerical_propagation.json`` is emitted by the crate's validated
arc test (``SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test
numerical_propagation reference_arc``); it carries an initial ECI state, the
integrator options, an output epoch grid, and the engine's reference ephemeris
as IEEE-754 hex bits. The binding propagates the same state with the same
options and must return the identical bits -- a wrapper that diverges is a
wrapper bug, not a new answer.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES, hex_to_f64


def _load_fixture():
    with open(os.path.join(FIXTURES, "numerical_propagation.json")) as fh:
        return json.load(fh)


def _propagate(fx):
    opts = fx["options"]
    return sidereon.propagate_state(
        hex_to_f64(fx["epoch_s_hex"]),
        np.asarray([hex_to_f64(h) for h in fx["position_km_hex"]], dtype=np.float64),
        np.asarray([hex_to_f64(h) for h in fx["velocity_km_s_hex"]], dtype=np.float64),
        np.asarray(
            [hex_to_f64(s["time_s_hex"]) for s in fx["samples"]], dtype=np.float64
        ),
        force_model=fx["force_model"],
        integrator=fx["integrator"],
        abs_tol=opts["abs_tol"],
        rel_tol=opts["rel_tol"],
        initial_step_s=opts["initial_step_s"],
        min_step_s=opts["min_step_s"],
        max_step_s=opts["max_step_s"],
        max_steps=opts["max_steps"],
    )


def _force_model(name):
    return {
        "two_body": sidereon.ForceModel.TWO_BODY,
        "two_body_j2": sidereon.ForceModel.TWO_BODY_J2,
    }[name]


def _integrator(name):
    return {
        "dp54": sidereon.Integrator.DP54,
        "rk4": sidereon.Integrator.RK4,
    }[name]


def test_propagation_matches_reference_bits():
    fx = _load_fixture()
    eph = _propagate(fx)

    n = len(fx["samples"])
    assert isinstance(eph.position_km, np.ndarray)
    assert eph.position_km.dtype == np.float64
    assert eph.position_km.shape == (n, 3)
    assert eph.velocity_km_s.shape == (n, 3)
    assert eph.times_s.shape == (n,)
    assert eph.states.shape == (n, 6)
    assert eph.epoch_count == n
    assert len(eph) == n

    for i, sample in enumerate(fx["samples"]):
        expected_pos = [hex_to_f64(h) for h in sample["position_km_hex"]]
        expected_vel = [hex_to_f64(h) for h in sample["velocity_km_s_hex"]]
        assert eph.times_s[i] == hex_to_f64(sample["time_s_hex"])
        for axis in range(3):
            assert eph.position_km[i, axis] == expected_pos[axis]
            assert eph.velocity_km_s[i, axis] == expected_vel[axis]
            # states column layout is [x, y, z, vx, vy, vz].
            assert eph.states[i, axis] == expected_pos[axis]
            assert eph.states[i, axis + 3] == expected_vel[axis]


def test_selector_enums_match_string_aliases():
    fx = _load_fixture()
    opts = fx["options"]
    legacy = _propagate(fx)
    enum_eph = sidereon.propagate_state(
        hex_to_f64(fx["epoch_s_hex"]),
        np.asarray([hex_to_f64(h) for h in fx["position_km_hex"]], dtype=np.float64),
        np.asarray([hex_to_f64(h) for h in fx["velocity_km_s_hex"]], dtype=np.float64),
        np.asarray(
            [hex_to_f64(s["time_s_hex"]) for s in fx["samples"]], dtype=np.float64
        ),
        force_model=_force_model(fx["force_model"]),
        integrator=_integrator(fx["integrator"]),
        abs_tol=opts["abs_tol"],
        rel_tol=opts["rel_tol"],
        initial_step_s=opts["initial_step_s"],
        min_step_s=opts["min_step_s"],
        max_step_s=opts["max_step_s"],
        max_steps=opts["max_steps"],
    )

    assert np.array_equal(enum_eph.position_km, legacy.position_km)
    assert np.array_equal(enum_eph.velocity_km_s, legacy.velocity_km_s)
    assert sidereon.ForceModel.TWO_BODY_J2.label == "two_body_j2"
    assert repr(sidereon.Integrator.DP54) == "Integrator.DP54"


def test_first_sample_is_initial_state():
    fx = _load_fixture()
    eph = _propagate(fx)
    expected_pos = [hex_to_f64(h) for h in fx["position_km_hex"]]
    for axis in range(3):
        assert eph.position_km[0, axis] == expected_pos[axis]


def test_two_body_circular_returns_to_start():
    # Analytic cross-check independent of the fixture: a circular orbit returns
    # to its start after exactly one period.
    mu = 398600.4418
    r = 7000.0
    v = (mu / r) ** 0.5
    period = 2.0 * np.pi * (r**3 / mu) ** 0.5
    eph = sidereon.propagate_state(
        0.0,
        np.asarray([r, 0.0, 0.0]),
        np.asarray([0.0, v, 0.0]),
        np.asarray([0.0, period]),
        force_model="two_body",
        integrator="dp54",
        abs_tol=1e-12,
        rel_tol=1e-12,
    )
    assert abs(eph.position_km[1, 0] - r) < 1e-6
    assert abs(eph.position_km[1, 1]) < 1e-6


def test_bad_position_shape_raises():
    with pytest.raises(ValueError):
        sidereon.propagate_state(
            0.0,
            np.asarray([7000.0, 0.0]),  # length 2, not 3
            np.asarray([0.0, 7.5, 0.0]),
            np.asarray([0.0, 60.0]),
        )


def test_empty_times_returns_empty_ephemeris():
    eph = sidereon.propagate_state(
        0.0,
        np.asarray([7000.0, 0.0, 0.0]),
        np.asarray([0.0, 7.5, 0.0]),
        np.asarray([], dtype=np.float64),
    )

    assert eph.epoch_count == 0
    assert len(eph) == 0
    assert eph.times_s.shape == (0,)
    assert eph.position_km.shape == (0, 3)
    assert eph.velocity_km_s.shape == (0, 3)
    assert eph.states.shape == (0, 6)


def test_unknown_force_model_raises():
    with pytest.raises(ValueError):
        sidereon.propagate_state(
            0.0,
            np.asarray([7000.0, 0.0, 0.0]),
            np.asarray([0.0, 7.5, 0.0]),
            np.asarray([0.0, 60.0]),
            force_model="newtonian_soup",
        )
