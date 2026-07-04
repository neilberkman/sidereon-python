import math

import numpy as np
import pytest
import sidereon


def test_015_position_metrics_isotropic_cep_and_non_psd_error():
    sigma = 3.25
    covariance = np.eye(3, dtype=np.float64) * sigma * sigma

    metrics = sidereon.metrics_from_enu_covariance_m2(covariance)

    assert metrics.sigma_e_m == sigma
    assert metrics.sigma_n_m == sigma
    assert metrics.sigma_u_m == sigma
    assert metrics.ellipse.semi_major_m == sigma
    assert metrics.ellipse.semi_minor_m == sigma
    assert metrics.cep_m.radius_m == pytest.approx(1.177410 * sigma, rel=1.0e-6)

    non_psd = np.asarray(
        [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="not positive semidefinite"):
        sidereon.metrics_from_enu_covariance_m2(non_psd)


def test_015_sidereal_undercovered_flags_passthrough():
    series = np.asarray([0.25, -0.5, 0.75, -1.0, 1.25], dtype=np.float64)
    options = sidereon.SiderealFilterOptions(min_coverage=2)

    output = sidereon.sidereal_filter(series, 4.0, options)

    assert output.filtered.tobytes() == series.tobytes()
    assert output.coverage == [1, 0, 0, 0]
    assert output.under_covered.tolist() == [True, True, True, True]
    assert np.isnan(output.template).all()


def test_015_midas_synthetic_velocity_matches_rust_reference():
    rate = np.asarray([0.01, -0.02, 0.03], dtype=np.float64)
    noise = [0.001, -0.002, 0.003, 0.0, 0.003, -0.002, 0.001]
    samples = []
    for year in range(7):
        t = float(year)
        position = np.asarray(
            [
                rate[0] * t + noise[year],
                rate[1] * t + 2.0 * noise[year],
                rate[2] * t - noise[year],
            ],
            dtype=np.float64,
        )
        samples.append(sidereon.PositionSample(t, position))

    velocity = sidereon.velocity_midas(sidereon.PositionSeries.enu(samples))

    np.testing.assert_allclose(
        velocity.rate_enu_m_per_yr,
        rate,
        rtol=0.0,
        atol=1.0e-16,
    )
    expected_sigma = (
        3.0
        * math.sqrt(math.pi / 2.0)
        * 1.482_602_218_505_602
        * 0.003
        / math.sqrt(6.0 / 4.0)
    )
    assert velocity.sigma_enu_m_per_yr[0] == pytest.approx(
        expected_sigma,
        abs=2.0e-17,
    )


def test_015_power_law_white_fm_slope_exact_and_short_fit_flagged():
    noise_type = sidereon.PowerLawNoiseType.WHITE_FM

    assert sidereon.allan_deviation_power_law_slope(noise_type) == -0.5
    assert sidereon.modified_allan_deviation_power_law_slope(noise_type) == -0.5
    assert sidereon.allan_variance_power_law_tau_exponent(noise_type) == -1

    series = sidereon.AllanSeries.fractional_frequency(
        np.asarray([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    )
    adev = sidereon.overlapping_adev(series, 1.0, [1])
    mdev = sidereon.modified_adev(series, 1.0, [1])
    fit = sidereon.fit_power_law_noise(
        adev,
        mdev,
        sidereon.PowerLawNoiseOptions(min_points_per_octave=3),
    )

    assert fit.regions == []
    assert np.isnan(fit.coefficients).all()
    assert fit.dominant_per_octave[0].dominance.kind == "flagged"
    assert (
        fit.dominant_per_octave[0].dominance.flag
        == sidereon.PowerLawOctaveFlag.UNDER_SAMPLED
    )


def test_015_sparse_orbit_fit_reports_unbounded_covariance_and_low_sample_ledger():
    start = sidereon.j2000_seconds(2026, 6, 1, 0, 0, 0.0)
    epochs_j2000 = np.asarray([start, start + 600.0], dtype=np.float64)
    unix0 = sidereon.Instant.from_utc(2026, 6, 1, 0, 0, 0.0).unix_micros
    epochs_unix_us = np.asarray([unix0, unix0 + 600_000_000], dtype=np.int64)

    truth = sidereon.propagate_state(
        start,
        np.asarray([7078.0, 0.0, 820.0], dtype=np.float64),
        np.asarray([0.15, 7.35, 1.0], dtype=np.float64),
        epochs_j2000,
        force_model=sidereon.ForceModelKind.two_body(),
        integrator=sidereon.Integrator.DP54,
        abs_tol=1.0e-12,
        rel_tol=1.0e-13,
        initial_step_s=10.0,
        max_step_s=60.0,
    )
    ecef_km = sidereon.gcrs_to_itrs(
        truth.position_km,
        epochs_unix_us,
        skyfield_compat=False,
    )
    samples = [
        sidereon.PreciseEphemerisSample(
            "G11",
            float(epoch),
            ecef_km[index] * 1000.0,
            time_scale=sidereon.TimeScale.UTC,
        )
        for index, epoch in enumerate(epochs_j2000)
    ]
    options = sidereon.OrbitFitOptions(
        force_model=sidereon.ForceModelKind.two_body(),
        integrator=sidereon.Integrator.DP54,
        abs_tol=1.0e-12,
        rel_tol=1.0e-13,
        initial_step_s=10.0,
        max_step_s=60.0,
        min_ledger_samples=3,
    )

    report = sidereon.fit_precise_ephemeris_sample_orbit(samples, "G11", options)
    fit = report.fits[0]
    stats = report.ledger.per_sat[0][1]

    assert fit.covariance.kind == "unbounded"
    assert fit.covariance.is_unbounded is True
    assert fit.covariance.matrix is None
    assert stats.n == 2
    assert stats.low_sample_count is True


def test_015_composite_force_selection_matches_two_body_bits():
    position = np.asarray([7078.0, -30.0, 820.0], dtype=np.float64)
    velocity = np.asarray([0.2, 7.35, 1.05], dtype=np.float64)
    epochs = np.asarray([0.0, 120.0, 240.0, 360.0], dtype=np.float64)
    kwargs = dict(
        integrator=sidereon.Integrator.RK4,
        initial_step_s=30.0,
        max_step_s=30.0,
    )

    legacy = sidereon.propagate_state(
        0.0,
        position,
        velocity,
        epochs,
        force_model=sidereon.ForceModel.TWO_BODY,
        **kwargs,
    )
    composite = sidereon.propagate_state(
        0.0,
        position,
        velocity,
        epochs,
        force_model=sidereon.ForceModelKind.composite(
            sidereon.ForceModelComponents.earth_two_body()
        ),
        **kwargs,
    )

    assert composite.states.tobytes() == legacy.states.tobytes()
    assert sidereon.ForceModelKind.earth_phase_a().label == "composite"
