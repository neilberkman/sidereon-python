import math

import numpy as np
import pytest
import sidereon


def _ecef_covariance_from_enu(covariance_enu_m2, receiver):
    sin_lat = math.sin(receiver.lat_rad)
    cos_lat = math.cos(receiver.lat_rad)
    sin_lon = math.sin(receiver.lon_rad)
    cos_lon = math.cos(receiver.lon_rad)
    rotation = np.asarray(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return rotation.T @ covariance_enu_m2 @ rotation


def _assert_percentile_radius(actual, expected):
    assert actual.probability == pytest.approx(expected.probability, rel=0.0, abs=0.0)
    assert actual.radius_m == pytest.approx(expected.radius_m, rel=1.0e-12)
    assert actual.approx_m == pytest.approx(expected.approx_m, rel=1.0e-12)
    assert actual.approx_valid is expected.approx_valid


def _assert_metrics_close(actual, expected):
    assert actual.sigma_e_m == pytest.approx(expected.sigma_e_m, rel=1.0e-12)
    assert actual.sigma_n_m == pytest.approx(expected.sigma_n_m, rel=1.0e-12)
    assert actual.sigma_u_m == pytest.approx(expected.sigma_u_m, rel=1.0e-12)
    assert actual.drms_m == pytest.approx(expected.drms_m, rel=1.0e-12)
    assert actual.two_drms_m == pytest.approx(expected.two_drms_m, rel=1.0e-12)
    assert actual.vep_m == pytest.approx(expected.vep_m, rel=1.0e-12)
    assert actual.mrse_m == pytest.approx(expected.mrse_m, rel=1.0e-12)
    assert actual.ellipse.semi_major_m == pytest.approx(
        expected.ellipse.semi_major_m, rel=1.0e-12
    )
    assert actual.ellipse.semi_minor_m == pytest.approx(
        expected.ellipse.semi_minor_m, rel=1.0e-12
    )
    assert actual.ellipse.orientation_rad == pytest.approx(
        expected.ellipse.orientation_rad, abs=1.0e-12
    )
    _assert_percentile_radius(actual.cep_m, expected.cep_m)
    _assert_percentile_radius(actual.r95_m, expected.r95_m)
    _assert_percentile_radius(actual.r99_m, expected.r99_m)
    _assert_percentile_radius(actual.sep_m, expected.sep_m)


def test_position_error_metrics_circular_covariance_oracles_and_helpers():
    sigma_m = 3.25
    covariance = np.eye(3, dtype=np.float64) * sigma_m * sigma_m

    metrics = sidereon.metrics_from_enu_covariance_m2(covariance)

    expected_cep50 = math.sqrt(2.0 * math.log(2.0)) * sigma_m
    expected_r95 = math.sqrt(-2.0 * math.log(1.0 - 0.95)) * sigma_m
    expected_drms = math.sqrt(2.0) * sigma_m
    assert metrics.cep_m.radius_m == pytest.approx(expected_cep50, rel=1.0e-12)
    assert metrics.r95_m.radius_m == pytest.approx(expected_r95, rel=1.0e-12)
    assert metrics.drms_m == pytest.approx(expected_drms, rel=1.0e-12)
    assert metrics.two_drms_m == pytest.approx(2.0 * expected_drms, rel=1.0e-12)

    ellipse = sidereon.error_ellipse_from_enu_m2(covariance)
    assert ellipse.semi_major_m == pytest.approx(sigma_m, rel=1.0e-12)
    assert ellipse.semi_minor_m == pytest.approx(sigma_m, rel=1.0e-12)
    assert ellipse.orientation_rad == pytest.approx(0.0, abs=1.0e-12)

    horizontal = sidereon.horizontal_radius_at(covariance, 0.95)
    spherical = sidereon.spherical_radius_at(covariance, 0.5)
    vertical = sidereon.vertical_radius_at(sigma_m * sigma_m, 0.5)
    _assert_percentile_radius(horizontal, metrics.r95_m)
    _assert_percentile_radius(spherical, metrics.sep_m)
    assert vertical == pytest.approx(0.6744897501960817 * sigma_m, rel=1.0e-12)
    assert metrics.vep_m == pytest.approx(0.674490 * sigma_m, rel=1.0e-12)

    covariance_value = sidereon.PositionCovariance(covariance * 2.0, covariance)
    from_position_covariance = sidereon.metrics_from_position_covariance(
        covariance_value
    )
    _assert_metrics_close(from_position_covariance, metrics)


def test_position_error_metrics_elongated_covariance_ellipse_oracle():
    covariance = np.asarray(
        [[9.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.44]],
        dtype=np.float64,
    )
    ellipse = sidereon.error_ellipse_from_enu_m2(covariance)

    trace = covariance[0, 0] + covariance[1, 1]
    delta = math.sqrt(
        (covariance[0, 0] - covariance[1, 1]) ** 2 + 4.0 * covariance[0, 1] ** 2
    )
    major_lambda = 0.5 * (trace + delta)
    minor_lambda = 0.5 * (trace - delta)
    assert ellipse.semi_major_m == pytest.approx(math.sqrt(major_lambda), rel=1.0e-12)
    assert ellipse.semi_minor_m == pytest.approx(math.sqrt(minor_lambda), rel=1.0e-12)
    assert ellipse.orientation_rad == pytest.approx(0.5 * math.atan2(4.0, 5.0))

    metrics = sidereon.metrics_from_enu_covariance_m2(covariance)
    assert metrics.cep_m.approx_valid
    expected_cep_approx = 0.6152 * math.sqrt(major_lambda) + 0.5620 * math.sqrt(
        minor_lambda
    )
    assert metrics.cep_m.approx_m == pytest.approx(expected_cep_approx, rel=1.0e-12)
    assert (
        abs(metrics.cep_m.radius_m - metrics.cep_m.approx_m) / metrics.cep_m.radius_m
        < 0.03
    )


def test_position_error_metrics_ecef_and_kinematic_paths_agree_with_rotated_enu():
    receiver = sidereon.Wgs84Geodetic(0.0, 0.0, 0.0)
    covariance_enu = np.asarray(
        [[5.0, 0.25, 0.1], [0.25, 2.0, -0.2], [0.1, -0.2, 1.25]],
        dtype=np.float64,
    )
    covariance_ecef = _ecef_covariance_from_enu(covariance_enu, receiver)

    from_enu = sidereon.metrics_from_enu_covariance_m2(covariance_enu)
    from_ecef = sidereon.metrics_from_ecef_covariance_m2(covariance_ecef, receiver)
    _assert_metrics_close(from_ecef, from_enu)

    solution = sidereon.KinematicSolution(
        np.asarray([6_378_137.0, 0.0, 0.0], dtype=np.float64),
        covariance_ecef,
    )
    from_solution = sidereon.metrics_from_kinematic_solution(solution)
    _assert_metrics_close(from_solution, from_enu)
