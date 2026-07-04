"""Sidereon 0.13 binding coverage.

The cases here use synthetic, in-repo vectors only. Numeric array parity uses
``.tobytes()`` where the API returns numpy arrays.
"""

import math

import numpy as np
import pytest
import sidereon


def _sample_rows():
    rows = []
    for sat_index, sat in enumerate(("G01", "G02")):
        base = np.array(
            [
                20_200_000.0 + 10_000.0 * sat_index,
                13_400_000.0 - 8_000.0 * sat_index,
                21_700_000.0 + 6_000.0 * sat_index,
            ],
            dtype=np.float64,
        )
        slope = np.array([12.0, -8.0, 5.0], dtype=np.float64) * (sat_index + 1.0)
        curve = np.array([0.03, -0.02, 0.01], dtype=np.float64)
        for i in range(15):
            epoch = float(i * 900.0)
            position = base + slope * i + curve * i * i
            rows.append(
                sidereon.PreciseEphemerisSample(
                    sat,
                    epoch,
                    position.tolist(),
                    1.0e-6 + sat_index * 1.0e-7 + i * 1.0e-10,
                )
            )
    return rows


def _arrivals(sensors, source, origin, speed):
    source = np.asarray(source, dtype=np.float64)
    out = []
    for sensor in sensors:
        sensor_speed = sensor.propagation_speed_m_s or speed
        out.append(origin + np.linalg.norm(source - sensor.position_m) / sensor_speed)
    return np.asarray(out, dtype=np.float64)


def test_observable_state_batch_and_interpolant_parity():
    samples = _sample_rows()
    source = sidereon.PreciseEphemerisSamples.from_samples(samples)
    handle = sidereon.PreciseEphemerisInterpolant.from_precise_ephemeris_samples(
        source
    )
    direct_handle = sidereon.PreciseEphemerisInterpolant.from_samples(samples)

    assert handle.time_scale == sidereon.TimeScale.GPST
    assert direct_handle.satellites == source.satellites == ["G01", "G02"]

    sats = ["G01", "G02", "G03"]
    epochs = np.asarray([5.5 * 900.0, 6.0 * 900.0, 5.5 * 900.0], dtype=np.float64)
    source_batch = sidereon.observable_states_at_j2000_s(source, sats, epochs)
    handle_batch = handle.observable_states_at_j2000_s(sats, epochs)

    assert (
        source_batch.positions_ecef_m.tobytes()
        == handle_batch.positions_ecef_m.tobytes()
    )
    assert source_batch.clocks_s.tobytes() == handle_batch.clocks_s.tobytes()
    assert source_batch.statuses == [
        sidereon.ObservableStateElementStatus.VALID,
        sidereon.ObservableStateElementStatus.VALID,
        sidereon.ObservableStateElementStatus.GAP,
    ]
    assert source_batch.element_results[:2] == [None, None]
    assert source_batch.element_result(2) is not None
    assert np.isnan(source_batch.positions_ecef_m[2]).all()
    assert np.isnan(np.asarray(sidereon.OBSERVABLE_STATE_MISSING_POSITION_ECEF_M)).all()

    scalar = handle.position_at_j2000_seconds("G01", float(epochs[0]))
    assert (
        scalar.position_m.tobytes()
        == source_batch.positions_ecef_m[0].copy().tobytes()
    )
    assert scalar.clock_s == source_batch.clocks_s[0]

    shared = sidereon.observable_states_at_shared_j2000_s(
        handle, ["G01", "G02"], float(epochs[0])
    )
    per_sat = sidereon.observable_states_at_j2000_s(
        handle,
        ["G01", "G02"],
        np.asarray([epochs[0], epochs[0]], dtype=np.float64),
    )
    assert shared.positions_ecef_m.tobytes() == per_sat.positions_ecef_m.tobytes()
    assert shared.clocks_s.tobytes() == per_sat.clocks_s.tobytes()


def test_estimation_primitives_match_core_reference_values():
    assert issubclass(sidereon.PrimitiveError, ValueError)

    gains = sidereon.alpha_beta_steady_state_gains(4.0)
    assert gains.alpha == pytest.approx(0.864_145_399_682_717_8, abs=1.0e-12)
    assert gains.beta == pytest.approx(0.737_169_180_900_238_8, abs=1.0e-12)

    kalman = sidereon.kalman_cv_steady_state_gains(4.0, 1.0, 1.0)
    assert kalman.position_gain == pytest.approx(gains.alpha, abs=1.0e-12)
    assert kalman.rate_gain == pytest.approx(gains.beta, abs=1.0e-12)

    step = sidereon.alpha_beta_filter_step(
        sidereon.AlphaBetaState(5.0, 2.0),
        8.0,
        2.0,
        sidereon.AlphaBetaGains(0.6, 0.8),
    )
    assert step.predicted.level == 9.0
    assert step.predicted.rate == 2.0
    assert step.innovation == -1.0
    assert step.updated.level == 8.4
    assert step.updated.rate == 1.6
    predicted = sidereon.alpha_beta_predict(sidereon.AlphaBetaState(5.0, 2.0), 2.0)
    assert predicted.level == 9.0
    assert (
        sidereon.alpha_beta_apply_measurement(
            sidereon.AlphaBetaState(9.0, 2.0),
            8.0,
            2.0,
            sidereon.AlphaBetaGains(0.6, 0.8),
        ).rate
        == 1.6
    )

    gate = sidereon.nis_gate_test(1.0, 1.0, 1, 0.95)
    assert gate.nis == 1.0
    assert gate.threshold == pytest.approx(3.841_458_820_694_124, abs=1.0e-12)
    assert gate.in_gate is True
    assert gate.dof == 1
    assert sidereon.normalized_innovation(2.0, 4.0) == 1.0
    assert sidereon.nis(2.0, 4.0) == 1.0
    assert sidereon.nis_statistic(2.0, 4.0) == 1.0
    assert sidereon.nis_expected_value(3) == 3.0
    assert sidereon.nis_gate_threshold(1, 0.95) == pytest.approx(
        3.841_458_820_694_124, abs=1.0e-12
    )

    q75 = 0.674_489_750_196_081_7
    values = np.asarray([-2.0 * q75, -q75, 0.0, q75, 2.0 * q75], dtype=np.float64)
    assert sidereon.MAD_GAUSSIAN_CONSISTENCY == pytest.approx(
        1.482_602_218_505_602, abs=1.0e-15
    )
    assert sidereon.mad_spread(values, 1.0e-12) == pytest.approx(1.0, abs=1.0e-12)
    assert sidereon.ewma_update(16.0, 2.0, 1.0 / 16.0) == pytest.approx(
        15.125, abs=1.0e-12
    )
    assert sidereon.ewma_update_power_of_two(16.0, 2.0, 4) == pytest.approx(
        15.125, abs=1.0e-12
    )

    multiplier = sidereon.cfar_ca_multiplier_from_pfa(4, 1.0e-3)
    assert multiplier == pytest.approx(18.493_653_007_613_965, abs=1.0e-12)
    threshold = sidereon.cfar_ca_threshold(4, 1.0e-3, 5.0)
    assert threshold == pytest.approx(5.0 * multiplier, abs=1.0e-12)
    assert sidereon.cfar_ca_pfa_from_multiplier(4, multiplier) == pytest.approx(
        1.0e-3, abs=1.0e-12
    )
    assert sidereon.cfar_ca_false_alarm_probability(4, threshold, 5.0) == pytest.approx(
        1.0e-3, abs=1.0e-12
    )

    with pytest.raises(sidereon.PrimitiveError):
        sidereon.normalized_innovation(1.0, 0.0)


def test_source_localization_known_vectors():
    assert issubclass(sidereon.SourceLocalizationError, ValueError)

    sensors_3d = [
        sidereon.Sensor([0.0, 0.0, 0.0]),
        sidereon.Sensor([1200.0, 0.0, 0.0]),
        sidereon.Sensor([0.0, 900.0, 0.0]),
        sidereon.Sensor([0.0, 0.0, 700.0]),
        sidereon.Sensor([1100.0, 800.0, 600.0]),
    ]
    source_3d = np.asarray([320.0, 260.0, 180.0], dtype=np.float64)
    origin = 12.5
    speed = 343.0
    times_3d = _arrivals(sensors_3d, source_3d, origin, speed)

    seed = sidereon.chan_ho_initial_guess(sensors_3d, times_3d, speed)
    assert np.allclose(seed.position_m, source_3d, atol=1.0e-8)
    assert seed.origin_time_s == pytest.approx(origin, abs=1.0e-10)
    assert seed.residual_rms_s < 1.0e-11

    solution = sidereon.locate_source(
        sensors_3d,
        times_3d,
        speed,
        sidereon.SourceLocateOptions(timing_sigma_s=0.001),
    )
    assert np.allclose(solution.position_m, source_3d, atol=1.0e-7)
    assert solution.origin_time_s == pytest.approx(origin, abs=1.0e-10)
    assert solution.covariance is not None
    assert solution.crlb is not None
    assert len(solution.residuals) == len(sensors_3d)
    assert len(solution.per_sensor_influence) == len(sensors_3d)
    assert solution.geometry_quality.residual_count == len(sensors_3d)

    sensors_2d = [
        sidereon.Sensor([0.0, 0.0]),
        sidereon.Sensor([1000.0, 0.0]),
        sidereon.Sensor([0.0, 800.0]),
        sidereon.Sensor([900.0, 900.0]),
    ]
    source_2d = np.asarray([300.0, 260.0], dtype=np.float64)
    tdoa_times = _arrivals(sensors_2d, source_2d, 4.0, 340.0)
    tdoa_solution = sidereon.locate_source(
        sensors_2d,
        tdoa_times,
        340.0,
        sidereon.SourceLocateOptions(mode=sidereon.SourceSolveMode.tdoa(0)),
    )
    assert np.allclose(tdoa_solution.position_m, source_2d, atol=1.0e-7)
    assert len(tdoa_solution.residuals) == len(sensors_2d) - 1

    square = [
        sidereon.Sensor([100.0, 0.0]),
        sidereon.Sensor([-100.0, 0.0]),
        sidereon.Sensor([0.0, 100.0]),
        sidereon.Sensor([0.0, -100.0]),
    ]
    dop = sidereon.source_dop(square, np.asarray([0.0, 0.0], dtype=np.float64), 10.0)
    assert dop.pdop == pytest.approx(10.0, abs=1.0e-12)
    assert dop.hdop == pytest.approx(10.0, abs=1.0e-12)
    assert dop.vdop == 0.0
    assert dop.tdop == pytest.approx(0.5, abs=1.0e-12)
    assert dop.gdop == pytest.approx(math.sqrt(100.25), abs=1.0e-12)

    crlb = sidereon.source_crlb(
        square, np.asarray([0.0, 0.0], dtype=np.float64), 10.0, 0.01
    )
    assert crlb.dop.pdop == pytest.approx(dop.pdop, abs=1.0e-12)
    assert crlb.covariance.position_m2[0, 0] == pytest.approx(0.005, abs=1.0e-15)
    assert crlb.covariance.position_m2[1, 1] == pytest.approx(0.005, abs=1.0e-15)
    assert crlb.covariance.origin_time_s2 == pytest.approx(0.000025, abs=1.0e-18)
