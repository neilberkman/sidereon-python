"""Area 6 conjunction binding reproduces the engine numbers bit-for-bit.

The fixture ``conjunction.json`` is emitted by the crate's env-gated harness
(``SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test
conjunction_python_fixture``). It carries encounter-frame geometry, B-plane
covariance, all Pc methods, and RTN->ECI covariance rotation as IEEE-754 hex
bits from the core APIs the binding calls.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES, hex_to_f64


def _fixture():
    with open(os.path.join(FIXTURES, "conjunction.json")) as fh:
        return json.load(fh)


FX = _fixture()


def _bits(arr):
    """uint64 bit view of a float64 numpy array, for exact comparison."""
    return np.asarray(arr, dtype=np.float64).view(np.uint64)


def _expect_bits(hex_list):
    return np.asarray([int(h, 16) for h in hex_list], dtype=np.uint64)


def _vector(hex_list):
    return np.asarray([hex_to_f64(h) for h in hex_list], dtype=np.float64)


def _matrix(hex_rows):
    return np.asarray([_vector(row) for row in hex_rows], dtype=np.float64)


def _state(entry):
    return sidereon.ConjunctionState(
        _vector(entry["position_km_hex"]),
        _vector(entry["velocity_km_s_hex"]),
        _matrix(entry["covariance_km2_hex"]),
    )


OBJ1 = _state(FX["object1"])
OBJ2 = _state(FX["object2"])


def test_conjunction_state_properties_are_numpy_and_value_like():
    assert OBJ1.position_km.shape == (3,)
    assert OBJ1.velocity_km_s.dtype == np.float64
    assert OBJ1.covariance_km2.shape == (3, 3)
    assert OBJ1 == _state(FX["object1"])
    assert "ConjunctionState" in repr(OBJ1)


def test_encounter_frame_matches_reference_bits():
    frame = sidereon.encounter_frame(
        OBJ1.position_km,
        OBJ1.velocity_km_s,
        OBJ2.position_km,
        OBJ2.velocity_km_s,
    )
    ref = FX["frame"]
    assert np.array_equal(_bits(frame.x_hat), _expect_bits(ref["x_hat_hex"]))
    assert np.array_equal(_bits(frame.y_hat), _expect_bits(ref["y_hat_hex"]))
    assert np.array_equal(_bits(frame.z_hat), _expect_bits(ref["z_hat_hex"]))
    assert np.array_equal(
        _bits(frame.relative_position_km), _expect_bits(ref["relative_position_km_hex"])
    )
    assert np.array_equal(
        _bits(frame.relative_velocity_km_s),
        _expect_bits(ref["relative_velocity_km_s_hex"]),
    )
    assert frame.miss_km == hex_to_f64(ref["miss_km_hex"])
    assert frame.relative_speed_km_s == hex_to_f64(ref["relative_speed_km_s_hex"])
    assert frame == sidereon.encounter_frame(
        OBJ1.position_km,
        OBJ1.velocity_km_s,
        OBJ2.position_km,
        OBJ2.velocity_km_s,
    )
    assert "EncounterFrame" in repr(frame)


def test_encounter_plane_covariance_matches_reference_bits():
    frame = sidereon.encounter_frame(
        OBJ1.position_km,
        OBJ1.velocity_km_s,
        OBJ2.position_km,
        OBJ2.velocity_km_s,
    )
    combined = _matrix(FX["combined_covariance_km2_hex"])
    projected = sidereon.encounter_plane_covariance(frame, combined)
    assert projected.shape == (2, 2)
    assert projected.dtype == np.float64
    assert np.array_equal(
        _bits(projected.ravel()),
        _expect_bits(sum(FX["encounter_plane_covariance_hex"], [])),
    )


def test_collision_probability_methods_match_reference_bits():
    hbr = hex_to_f64(FX["hard_body_radius_km_hex"])
    for entry in FX["collision_probability"]:
        method = getattr(sidereon.PcMethod, entry["method"])
        result = sidereon.collision_probability(OBJ1, OBJ2, hbr, method)
        assert result.pc == hex_to_f64(entry["pc_hex"])
        assert result.miss_km == hex_to_f64(entry["miss_km_hex"])
        assert result.relative_speed_km_s == hex_to_f64(
            entry["relative_speed_km_s_hex"]
        )
        assert result.sigma_x_km == hex_to_f64(entry["sigma_x_km_hex"])
        assert result.sigma_z_km == hex_to_f64(entry["sigma_z_km_hex"])
        assert result == sidereon.collision_probability(OBJ1, OBJ2, hbr, method)
        assert "CollisionProbability" in repr(result)


def test_pc_method_enum():
    assert sidereon.PcMethod.FOSTER_EQUAL_AREA.label == "foster_equal_area"
    assert sidereon.PcMethod.FOSTER_EQUAL_AREA != sidereon.PcMethod.ALFANO_2005
    assert repr(sidereon.PcMethod.ALFANO_2005) == "PcMethod.ALFANO_2005"


def test_rtn_to_eci_covariance_matches_reference_bits():
    ref = FX["rtn"]
    eci = sidereon.rtn_to_eci_covariance(
        _matrix(ref["covariance_rtn_hex"]),
        _vector(ref["position_km_hex"]),
        _vector(ref["velocity_km_s_hex"]),
    )
    assert eci.shape == (3, 3)
    assert eci.dtype == np.float64
    assert np.array_equal(
        _bits(eci.ravel()), _expect_bits(sum(ref["covariance_eci_hex"], []))
    )
    assert (
        sidereon.covariance_is_symmetric(_matrix(ref["covariance_rtn_hex"]))
        is ref["symmetric"]
    )
    assert (
        sidereon.covariance_is_positive_semidefinite(_matrix(ref["covariance_rtn_hex"]))
        is ref["positive_semidefinite"]
    )


def test_conjunction_errors_are_typed():
    with pytest.raises(ValueError):
        sidereon.ConjunctionState(np.zeros(2), np.zeros(3), np.eye(3))

    with pytest.raises(ValueError):
        sidereon.rtn_to_eci_covariance(np.eye(3), np.zeros(3), np.ones(3))

    with pytest.raises(sidereon.SolveError):
        sidereon.encounter_frame(np.zeros(3), np.ones(3), np.ones(3), np.ones(3))

    with pytest.raises(ValueError):
        sidereon.collision_probability(OBJ1, OBJ2, -1.0)

    with pytest.raises(ValueError):
        sidereon.collision_probability(OBJ1, OBJ2, 0.0)

    bad_state = sidereon.ConjunctionState(
        np.array([np.nan, 0.0, 0.0]), OBJ1.velocity_km_s, OBJ1.covariance_km2
    )
    with pytest.raises(ValueError):
        sidereon.collision_probability(bad_state, OBJ2, 0.02)

    with pytest.raises(ValueError):
        sidereon.encounter_frame(
            np.array([np.inf, 0.0, 0.0]),
            OBJ1.velocity_km_s,
            OBJ2.position_km,
            OBJ2.velocity_km_s,
        )

    frame = sidereon.encounter_frame(
        OBJ1.position_km,
        OBJ1.velocity_km_s,
        OBJ2.position_km,
        OBJ2.velocity_km_s,
    )
    bad_cov = OBJ1.covariance_km2.copy()
    bad_cov[0, 0] = np.nan
    with pytest.raises(ValueError):
        sidereon.encounter_plane_covariance(frame, bad_cov)
