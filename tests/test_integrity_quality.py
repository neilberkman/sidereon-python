import pytest
import sidereon


def test_chi2_isf_and_joint_gate_options():
    assert sidereon.chi2_isf(0.05, 2) == pytest.approx(5.991464547107979)
    assert (
        sidereon.EkfUpdateOptions(
            joint_innovation_gate_probability=0.05
        ).joint_innovation_gate_probability
        == 0.05
    )
    assert (
        sidereon.UkfUpdateOptions(
            joint_innovation_gate_probability=0.01
        ).joint_innovation_gate_probability
        == 0.01
    )


def test_add_v3_model_and_constellation_error():
    import math

    model = sidereon.SatelliteIsmModel.add_v3_0(
        sidereon.GnssSystem.GPS, math.pi / 2, 0.75, 0.5, 0.5, 1e-5
    )
    # Core Appendix A test gives local variance at zenith.
    assert model.effective_sigma_int_m**2 - 0.75**2 == pytest.approx(
        0.27847436379754925, abs=2e-15
    )
    with pytest.raises(ValueError) as exc:
        sidereon.SatelliteIsmModel.add_v3_0(
            sidereon.GnssSystem.BEIDOU, math.pi / 6, 0.75, 0.5, 0.5, 1e-5
        )
    assert exc.value.detail == {
        "family": "AraimError",
        "kind": "unsupported_constellation",
        "system": sidereon.GnssSystem.BEIDOU,
    }


def test_checked_coverage_empty_and_invalid_threshold():
    grid = sidereon.coverage_look_angles([], [], 0)
    assert grid.try_visible_mask(0.0) == []
    assert grid.try_access_counts(0.0) == []
    with pytest.deprecated_call():
        assert grid.visible_mask(0.0) == []
    with pytest.deprecated_call():
        assert grid.access_counts(0.0) == []
    with pytest.raises(ValueError):
        grid.try_visible_mask(float("nan"))
    with pytest.raises(ValueError):
        grid.try_access_counts(float("inf"))


def test_raim_levels_against_core_oracle():
    import json
    import os
    from pathlib import Path

    fixture = json.loads(
        (
            Path(os.environ["SIDEREON_CORE_FIXTURES"]) / "ppp_raim_oracles.json"
        ).read_text()
    )
    rows = [
        sidereon.PppRaimGeometryRow(f"G{i:02}", los)
        for i, los in enumerate(fixture["line_of_sight_ecef"], 1)
    ]
    lat, lon = fixture["inputs"]["receiver_geodetic_rad"]
    receiver = sidereon.Wgs84Geodetic(lat, lon)
    config = sidereon.PppRaimConfig()
    levels = sidereon.protection_levels(rows, receiver, config)
    oracle = fixture["slope_cases"][0]
    assert levels.hpl_m == pytest.approx(oracle["hpl_slope"], abs=1e-8)
    assert levels.vpl_m == pytest.approx(oracle["vpl_slope"], abs=1e-8)
    accuracy = sidereon.fault_free_accuracy_bounds(rows, [1.0] * 6, receiver, config)
    assert accuracy.hpl_m == pytest.approx(oracle["hpl_dop_two_sided"], abs=1e-9)
    assert accuracy.vpl_m == pytest.approx(oracle["vpl_dop_two_sided"], abs=1e-9)
    weighted = sidereon.protection_levels_with_weights(
        rows, [0.1] * 6, receiver, config
    )
    assert weighted.hpl_m == pytest.approx(
        fixture["weighted_protection"]["hpl_slope_m"], abs=1e-7
    )
    with pytest.raises(sidereon.SolveError) as exc:
        sidereon.protection_levels(rows[:4], receiver, config)
    assert exc.value.detail["kind"] == "invalid_geometry"
    with pytest.raises(sidereon.SolveError) as exc:
        sidereon.protection_levels(
            rows[:5], receiver, sidereon.PppRaimConfig(chi_square_threshold=1e15)
        )
    assert exc.value.detail["kind"] == "unsupported_noncentrality"


def test_raim_unsupported_residual_degrees_of_freedom():
    rows = [
        sidereon.PppRaimGeometryRow("G01", [1.0, 0.0, 0.0]),
        sidereon.PppRaimGeometryRow("G02", [0.0, 1.0, 0.0]),
        sidereon.PppRaimGeometryRow("G03", [0.0, 0.0, 1.0]),
        sidereon.PppRaimGeometryRow("G04", [-1.0, 0.0, 0.0]),
        sidereon.PppRaimGeometryRow("G05", [0.0, -1.0, 0.0]),
    ]
    with pytest.raises(sidereon.SolveError) as exc:
        sidereon.protection_levels(
            rows * 20_001, sidereon.Wgs84Geodetic(0.0, 0.0), sidereon.PppRaimConfig()
        )
    assert exc.value.detail == {
        "family": "RaimError",
        "kind": "unsupported_dof",
        "dof": 100001,
        "max": 100000,
    }
