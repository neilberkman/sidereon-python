import math

import numpy as np
import pytest
import sidereon


def _source_arrivals(sensor_positions, source, origin_s, speed_m_s):
    source = np.asarray(source, dtype=np.float64)
    return np.asarray(
        [
            origin_s
            + np.linalg.norm(np.asarray(pos, dtype=np.float64) - source) / speed_m_s
            for pos in sensor_positions
        ],
        dtype=np.float64,
    )


def test_geometry_quality_source_solution_nominal_smoke():
    positions = [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
        [2.0, 2.0, 2.0],
    ]
    sensors = [sidereon.Sensor(pos) for pos in positions]
    source = np.asarray([0.4, 0.6, 0.5], dtype=np.float64)
    arrivals = _source_arrivals(positions, source, 1.25, 1.0)

    solution = sidereon.locate_source(
        sensors,
        arrivals,
        1.0,
        sidereon.SourceLocateOptions(timing_sigma_s=0.001),
    )

    quality = solution.geometry_quality
    assert isinstance(quality, sidereon.GeometryQuality)
    assert sidereon.SourceGeometryQuality is sidereon.GeometryQuality
    assert quality.tier == sidereon.ObservabilityTier.NOMINAL
    assert quality.tier.label == "nominal"
    assert quality.rank == 4
    assert quality.redundancy == 1
    assert quality.raim_checkable is True
    assert quality.covariance_validated is True
    assert math.isfinite(quality.condition_number)
    assert math.isfinite(quality.gdop)
    assert "GeometryQuality(" in repr(quality)


def test_source_rank_deficient_geometry_yields_singular_error():
    sensors = [
        sidereon.Sensor([0.0, 0.0]),
        sidereon.Sensor([100.0, 0.0]),
        sidereon.Sensor([200.0, 0.0]),
        sidereon.Sensor([300.0, 0.0]),
    ]

    with pytest.raises(sidereon.SourceLocalizationError) as excinfo:
        sidereon.source_dop(sensors, np.asarray([50.0, 0.0], dtype=np.float64), 300.0)

    message = str(excinfo.value)
    assert "source geometry failed" in message
    assert "singular" in message


def _distance(a, b):
    return float(
        np.linalg.norm(
            np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        )
    )


def _rtk_row(sat, pos, base, rover, ambiguity_m):
    return sidereon.RtkSatMeasurement(
        sat=sat,
        sd_ambiguity_id=sat,
        base_code_m=_distance(pos, base),
        base_phase_m=_distance(pos, base),
        rover_code_m=_distance(pos, rover),
        rover_phase_m=_distance(pos, rover) + ambiguity_m,
        base_tx_pos=pos,
        rover_tx_pos=pos,
        pos=pos,
    )


def _rtk_float_config(sats):
    base = [4_075_580.0, 931_854.0, 4_801_568.0]
    baseline = [1.2, -0.85, 0.91]
    rover = [base[i] + baseline[i] for i in range(3)]
    rows = [_rtk_row(sat, pos, base, rover, ambiguity) for sat, pos, ambiguity in sats]
    epoch = sidereon.RtkEpoch(references=[rows[0]], nonref=rows[1:], dt_s=0.0)
    return sidereon.RtkFloatConfig(
        epochs=[epoch],
        base=base,
        ambiguity_ids=[sat for sat, _pos, _ambiguity in sats[1:]],
        model=sidereon.RtkMeasurementModel(
            code_sigma_m=0.3,
            phase_sigma_m=0.003,
            sagnac=False,
            stochastic=sidereon.RtkStochasticModel.SIMPLE,
            elevation_weighting=False,
        ),
        initial_baseline_m=baseline,
        options=sidereon.RtkFloatOptions(
            position_tol_m=1.0e-9,
            ambiguity_tol_m=1.0e-9,
            max_iterations=5,
        ),
    )


def test_geometry_quality_rtk_float_nominal_smoke():
    sats = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0.0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 0.6),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -1.4),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 1.0),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -0.3),
    ]

    solution = sidereon.solve_rtk_float(_rtk_float_config(sats))
    quality = solution.geometry_quality

    assert quality.tier == sidereon.ObservabilityTier.NOMINAL
    assert quality.rank == 7
    assert quality.redundancy == 1
    assert quality.raim_checkable is True
    assert quality.covariance_validated is True


def test_rtk_rank_deficient_geometry_yields_singular_error():
    repeated = [-12_000_000.0, 18_000_000.0, 19_000_000.0]
    sats = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0.0),
        ("G02", repeated, 0.6),
        ("G03", repeated, -1.4),
        ("G04", repeated, 1.0),
    ]

    with pytest.raises(sidereon.SolveError) as excinfo:
        sidereon.solve_rtk_float(_rtk_float_config(sats))

    message = str(excinfo.value)
    assert "RTK float geometry is singular" in message
