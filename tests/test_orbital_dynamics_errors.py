import math
import struct
from pathlib import Path

import pytest
import sidereon as s
from _helpers import CORE_FIXTURES


def test_gnss_geometry_refusals():
    with pytest.raises(s.GnssAttitudeError) as caught:
        s.nominal_yaw([float("nan"), 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    assert caught.value.detail == {"kind": "non_finite", "field": "position_m"}
    with pytest.raises(s.GnssAttitudeError) as caught:
        s.dyb_axes([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
    assert caught.value.detail == {"kind": "degenerate_geometry", "t_s": None}


def test_gnss_law_refusals():
    propagator = s.EcksteinHechlerPropagator.from_mean_elements(
        0.0,
        s.CircularElements(26560.0, 0.0, 0.0, 0.4, 0.0, 4.9),
        s.ZonalHarmonics.earth(),
    )
    trajectory = s.OrbitAttitudeTrajectory.from_eckstein_hechler(propagator)
    cases = [
        (
            s.GnssAttitudeLaw.gps_iir(-0.1),
            {"kind": "invalid_parameter", "field": "yaw_rate_rad_s", "value": -0.1},
        ),
        (
            s.GnssAttitudeLaw.for_block(s.GnssBlock.beidou3_meo()),
            {"kind": "beidou_manufacturer_required", "t_s": 0.0},
        ),
        (
            s.GnssAttitudeLaw.galileo_iov(1.57, 1.56),
            {"kind": "turn_end_not_found", "t_s": 0.0},
        ),
        (
            s.GnssAttitudeLaw.gps_iia(100.0, 0.01, s.GPS_SHADOW_LIMIT_RAD),
            {
                "kind": "outside_validated_domain",
                "t_s": -30000.0,
                "reason": (
                    "GPS IIA maximum yaw rate above 1e5 times the orbital rate, where "
                    "the shadow crossing's yaw is set only to the rate times the "
                    "uncertainty of the shadow entry"
                ),
            },
        ),
    ]
    for law, detail in cases:
        with pytest.raises(s.GnssAttitudeError) as caught:
            s.gnss_attitude(law, trajectory, detail.get("t_s", 0.0))
        assert caught.value.detail == detail


def test_gnss_block_refusals():
    cases = [
        ("BDS-3S", {"kind": "unsupported", "name": "BDS-3S"}),
        ("BDS-3", {"kind": "beidou_orbit_class_required", "block_type": "BDS-3"}),
    ]
    for name, detail in cases:
        with pytest.raises(s.GnssBlockError) as caught:
            s.GnssBlock.from_constellation_block_type(s.GnssSystem.BEIDOU, name)
        assert caught.value.detail == detail


def test_de_file_and_state_refusals(tmp_path):
    with pytest.raises(s.DeError) as caught:
        s.DeEphemeris.load(tmp_path / "missing.440")
    assert caught.value.detail == {
        "kind": "io",
        "message": "No such file or directory (os error 2)",
        "raw_os_error": 2,
    }

    path = Path(CORE_FIXTURES) / "bodies/de440/lnxp1990.440"
    data = bytearray(path.read_bytes())
    # The first Sun coefficient in the first little-endian DE440 data record.
    struct.pack_into("<d", data, 2 * 8144 + 8 * (753 - 1), float("inf"))
    ephemeris = s.DeEphemeris.from_bytes(bytes(data))
    with pytest.raises(s.DeError) as caught:
        ephemeris.sun_moon_at_tdb_seconds(-315921100.0)
    assert caught.value.detail == {"kind": "non_finite_state"}


def test_antumbra_axis_gradient_refusal():
    with pytest.raises(s.ShadowGradientError) as caught:
        s.shadow_fraction_with_gradient([-2e6, 0.0, 0.0], [150e6, 0.0, 0.0])
    assert caught.value.detail == {"kind": "numerical_failure"}
    assert str(caught.value) == "shadow fraction gradient is not finite"


@pytest.mark.parametrize(
    "a,e,i,detail",
    [
        (
            6000.0,
            0.0,
            0.7,
            {
                "kind": "inside_reference_sphere",
                "semi_major_axis_km": 6000.0,
                "reference_radius_km": 6378.137,
            },
        ),
        (
            7000.0,
            0.2,
            0.7,
            {"kind": "eccentricity_too_large", "eccentricity": 0.2, "limit": 0.1},
        ),
        (7000.0, 0.0, 0.0, {"kind": "nearly_equatorial", "inclination_rad": 0.0}),
        (
            7000.0,
            0.0,
            1.1071487177940904,
            {
                "kind": "near_critical_inclination",
                "inclination_rad": 1.1071487177940904,
            },
        ),
        (7000.0, 0.0, float("nan"), {"kind": "non_finite", "field": "mean elements"}),
    ],
)
def test_eckstein_hechler_domain_refusals(a, e, i, detail):
    with pytest.raises(s.AnalyticPropagationError) as caught:
        s.EcksteinHechlerPropagator.from_mean_elements(
            0.0, s.CircularElements(a, e, 0.0, i, 0.0, 4.9), s.ZonalHarmonics.earth()
        )
    assert caught.value.detail == detail


def test_analytic_conversion_refusals():
    gravity = s.ZonalHarmonics.earth()
    for e, mean_anomaly, detail in [
        (1.0, 0.0, {"kind": "not_elliptic", "field": "elements"}),
        (
            0.998046875,
            0.001953125,
            {"kind": "kepler_not_converged", "field": "eccentric anomaly"},
        ),
    ]:
        elements = s.AnalyticKeplerianElements(7000.0, e, 0.7, 0.2, 0.3, mean_anomaly)
        with pytest.raises(s.AnalyticPropagationError) as caught:
            elements.to_cartesian(gravity.mu_km3_s2)
        assert caught.value.detail == detail

    retrograde = s.CartesianState(0.0, [7000.0, 0.0, 0.0], [0.0, -7.546, 0.0])
    with pytest.raises(s.AnalyticPropagationError) as caught:
        s.brouwer_lyddane_mean_elements(
            retrograde, gravity, s.MeanConversionOptions.BROUWER_LYDDANE_DEFAULT
        )
    assert caught.value.detail == {
        "kind": "retrograde_equatorial",
        "inclination_rad": math.pi,
    }

    propagator = s.EcksteinHechlerPropagator.from_mean_elements(
        0.0, s.CircularElements(7000.0, 0.001, 0.0, 0.7, 0.0, 4.9), gravity
    )
    with pytest.raises(s.AnalyticPropagationError) as caught:
        s.eckstein_hechler_mean_elements(
            propagator.propagate(0.0), gravity, s.MeanConversionOptions(1e-11, 1)
        )
    assert caught.value.detail == {
        "kind": "mean_elements_not_converged",
        "theory": s.AnalyticTheory.ECKSTEIN_HECHLER,
        "iterations": 1,
    }


def test_brouwer_lyddane_overflow_refusal():
    propagator = s.BrouwerLyddanePropagator.from_mean_elements(
        0.0,
        s.AnalyticKeplerianElements(7000.0, 0.03, 0.7, 0.2, 0.3, 0.4),
        s.ZonalHarmonics.earth(),
        1e-17,
    )
    with pytest.raises(s.AnalyticPropagationError) as caught:
        propagator.propagate(1e308)
    assert caught.value.detail == {
        "kind": "non_finite_result",
        "theory": s.AnalyticTheory.BROUWER_LYDDANE,
        "field": "a_km",
    }


def test_decay_refusals():
    drag = s.DragParameters.from_bc_factor_m2_kg(0.02)
    context = s.PropagationContext()
    for position, extra, detail in [
        (
            [7000.0, 0.0, 0.0],
            {},
            {"kind": "no_decay_within_horizon", "horizon_s": 20.0},
        ),
        (
            [7000.0, 0.0, 0.0],
            {"max_scan_samples": 1},
            {"kind": "scan_budget_exhausted", "samples": 1, "scanned_s": 10.0},
        ),
        (
            [0.0, 0.0, 0.0],
            {},
            {
                "kind": "propagation",
                "source": {
                    "kind": "numerical_failure",
                    "message": "Zero position magnitude",
                },
            },
        ),
    ]:
        with pytest.raises(s.SolveError) as caught:
            s.estimate_decay_with_context(
                0.0,
                position,
                [0.0, 7.5, 0.0],
                drag,
                context,
                scan_step_s=10.0,
                max_duration_s=20.0,
                **extra,
            )
        assert caught.value.detail == detail


def test_composite_force_recovers_finite_sum():
    limit = float.fromhex("0x1.fffffffffffffp1023")
    initial = s.CartesianState(0.0, [7000.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    positive, negative = [
        s.EmpiricalRtnAcceleration().with_term(
            s.RtnAxis.RADIAL, s.EmpiricalHarmonic.CONSTANT, coefficient
        )
        for coefficient in (limit, -limit)
    ]
    recovered = s.CompositeForceModel([positive, positive, negative])
    assert recovered.acceleration(initial) == [limit, 0.0, 0.0]
    overflowing = s.CompositeForceModel([positive, positive])
    with pytest.raises(s.SolveError) as caught:
        overflowing.acceleration(initial)
    assert caught.value.detail == {
        "kind": "numerical_failure",
        "message": "acceleration not finite",
    }


def test_sourced_drag_weather_refusal():
    table = s.load_space_weather(
        Path(CORE_FIXTURES) / "space_weather/SW-All-20260702-trim.csv"
    )
    force = s.SourcedDragForce(s.DragParameters.from_bc_factor_m2_kg(0.02), table)
    initial = s.CartesianState(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    with pytest.raises(s.SolveError) as caught:
        force.acceleration_with_parameters([], initial)
    assert caught.value.detail == {
        "kind": "force_model_failure",
        "message": "space weather: space-weather lookup before coverage",
    }
