import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest
import sidereon as s
from _helpers import CORE_FIXTURES, FIXTURES, hex_to_f64


@pytest.fixture(scope="module")
def orbital():
    path = Path(FIXTURES) / "core_goldens.json"
    return json.loads(path.read_text())["orbital_dynamics"]


def exact(actual, expected):
    if isinstance(expected, list):
        assert len(actual) == len(expected)
        for item, reference in zip(actual, expected, strict=True):
            exact(item, reference)
    elif isinstance(expected, str) and expected.startswith("0x"):
        assert "0x" + struct.pack(">d", actual).hex() == expected
    else:
        assert actual == expected


def test_geocentric_bodies_and_coordinate_time(orbital):
    fx = orbital["celestial"]
    instant = s.Instant.from_utc(2020, 6, 24, 12, 0, 0.0)
    exact(s.sun_moon_gcrs(0.0), fx["sun_moon"][0])
    exact(s.sun_moon_gcrs_at(instant), fx["sun_moon"][1])
    exact(s.sun_moon_gcrs_at_tdb_seconds(0.0), fx["sun_moon"][2])
    exact(s.gcrs_to_mean_of_date_matrix(instant), fx["gcrs_to_mean"])
    exact(s.mean_of_date_to_gcrs_matrix(instant), fx["mean_to_gcrs"])
    exact(
        [
            s.tdb_minus_tt_seconds(0.0),
            s.tdb_from_tt_j2000_seconds(0.0),
            s.tt_from_tdb_j2000_seconds(0.0),
        ],
        fx["time"],
    )


def test_de_ephemeris_and_context(orbital):
    fx = orbital["celestial"]["de"]
    path = Path(CORE_FIXTURES) / "bodies/de440/lnxp1990.440"
    ephemeris = s.DeEphemeris.load(path)
    binary = s.DeEphemeris.from_bytes(path.read_bytes())
    epoch = hex_to_f64(fx["epoch"])
    exact(ephemeris.coverage_tdb_seconds(), fx["coverage"])
    assert ephemeris.version() == fx["version"]
    assert ephemeris.record_size() == fx["record_size"]
    assert ephemeris.constants() == {
        key: hex_to_f64(value) for key, value in fx["constants"].items()
    }
    exact(ephemeris.astronomical_unit_m(), fx["astronomical_unit_m"])
    exact(ephemeris.earth_moon_mass_ratio(), fx["earth_moon_mass_ratio"])
    bodies = (
        s.DeBody.SUN,
        s.DeBody.MERCURY,
        s.DeBody.VENUS,
        s.DeBody.EARTH_MOON,
        s.DeBody.EARTH,
        s.DeBody.MOON,
        s.DeBody.MARS,
        s.DeBody.JUPITER,
        s.DeBody.SATURN,
        s.DeBody.URANUS,
        s.DeBody.NEPTUNE,
        s.DeBody.PLUTO,
        s.DeBody.SOLAR_SYSTEM,
    )
    exact([ephemeris.gm_m3_s2(body) for body in bodies], fx["gm"])
    states = ephemeris.sun_moon_at_tdb_seconds(epoch)
    exact([states.sun.position_m, states.sun.velocity_m_s], fx["sun_state"])
    exact([states.moon.position_m, states.moon.velocity_m_s], fx["moon_state"])
    assert states == binary.sun_moon_at_tdb_seconds(epoch)
    assert states.sun != states.moon
    assert states.sun == binary.sun_moon_at_tdb_seconds(epoch).sun
    assert states != binary.sun_moon_at_tdb_seconds(epoch + 1.0)
    exact(ephemeris.sun_moon_positions_at_tdb_seconds(epoch), fx["positions"])
    context = s.PropagationContext()
    assert context.de_ephemeris() is None
    assert context.ut1_departure() is None
    attached = context.with_de_ephemeris(ephemeris)
    assert attached.de_ephemeris() is not None
    exact(attached.de_ephemeris().coverage_tdb_seconds(), fx["coverage"])

    with pytest.raises(s.DeError) as caught:
        s.DeEphemeris.from_bytes(b"")
    assert caught.value.detail == {
        "kind": "invalid_file",
        "reason": "incomplete first header record",
    }
    outside = hex_to_f64(fx["coverage"][0]) - 1.0
    with pytest.raises(s.DeError) as caught:
        ephemeris.sun_moon_at_tdb_seconds(outside)
    assert caught.value.detail == {
        "kind": "outside_coverage",
        "epoch_tdb_seconds": outside,
    }
    with pytest.raises(s.DeError) as caught:
        ephemeris.sun_moon_positions_at_tdb_seconds(float("nan"))
    assert caught.value.detail == {"kind": "non_finite_epoch"}


def test_shadow_fractions_and_gradients(orbital):
    fx = orbital["terrestrial"]["eclipse"]
    satellite = [7_000.0, 0.0, 0.0]
    sun = [-150_000_000.0, 0.0, 0.0]
    moon = [-400_000.0, 0.0, 0.0]
    model = s.EarthShadowModel.WGS84_OBLATE
    exact(s.moon_shadow_fraction(satellite, sun, moon), fx[0])
    exact(s.shadow_fraction_with_moon(satellite, sun, moon), fx[1])
    exact(s.shadow_fraction_with_model_and_moon(satellite, sun, moon, model), fx[2])
    exact(s.shadow_fraction_with_gradient(satellite, sun), fx[3])
    exact(s.shadow_fraction_with_model_and_gradient(satellite, sun, model), fx[4])
    exact(
        s.shadow_fraction_with_model_moon_and_gradient(satellite, sun, moon, model),
        fx[5],
    )
    assert s.status_with_moon(satellite, sun, moon) == s.EclipseStatus.UMBRA
    assert (
        s.status_with_model_and_moon(satellite, sun, moon, model)
        == s.EclipseStatus.UMBRA
    )
    assert s.IAU_2015_SOLAR_RADIUS_KM == 695_700.0
    with pytest.raises(s.ShadowGradientError) as caught:
        s.shadow_fraction_with_gradient([0.0, 0.0, 0.0], sun)
    assert caught.value.detail == {
        "kind": "eclipse",
        "source": {
            "kind": "invalid_input",
            "field": "sat_pos",
            "reason": "zero vector",
        },
    }


def test_iers_earth_orientation(orbital):
    fx = orbital["terrestrial"]["orientation"]
    first = s.Iers2010EopSample(59024, 0.1, 0.2, -0.2, 0.001, 0.0001, -0.0001)
    second = s.Iers2010EopSample(59025, 0.1, 0.2, -0.2, 0.001, 0.0001, -0.0001)
    assert first == s.Iers2010EopSample(59024, 0.1, 0.2, -0.2, 0.001, 0.0001, -0.0001)
    assert first != second
    assert [
        first.mjd_utc,
        first.xp_arcsec,
        first.yp_arcsec,
        first.ut1_minus_utc_s,
        first.lod_s,
        first.dx_arcsec,
        first.dy_arcsec,
        first.pole_rates_rad_s,
    ] == [59024, 0.1, 0.2, -0.2, 0.001, 0.0001, -0.0001, None]
    provider = s.Iers2010EarthOrientationProvider([first, second])
    assert provider == s.Iers2010EarthOrientationProvider([first, second])
    assert provider != provider.with_validity(s.ValidityMode.PERMISSIVE)
    result = provider.orientation_at_utc_mjd(59024, 43200.0)
    exact(
        [
            result.iers2010_quaternion(),
            result.earth_rotation_vector_itrf_rad_s(),
            result.gcrf_to_itrf_matrix(),
            result.itrf_to_gcrf_matrix(),
            result.gcrf_to_itrf_rotation_rate_matrix(),
            result.itrf_to_gcrf_rotation_rate_matrix(),
        ],
        fx,
    )
    assert result == provider.orientation_at_utc_mjd(59024, 43200.0)
    assert result != provider.orientation_at_utc_mjd(59024, 0.0)
    exact(
        provider.orientation_at_tdb_seconds(646272069.184).iers2010_quaternion(),
        orbital["terrestrial"]["orientation_tdb"],
    )
    assert result.ut1_degraded() is None
    with pytest.raises(s.SolveError) as caught:
        provider.orientation_at_utc_mjd(59024, -1.0)
    assert caught.value.detail == {
        "kind": "invalid_input",
        "field": "seconds_in_day",
        "reason": "outside the UTC day",
    }
    with pytest.raises(s.SolveError) as caught:
        s.Iers2010EarthOrientationProvider([second, first])
    assert caught.value.detail == {
        "kind": "invalid_input",
        "field": "samples",
        "reason": "MJDs must be strictly increasing",
    }


def test_icgem_gravity(orbital):
    fx = orbital["terrestrial"]
    field = s.IcgemGravityField.parse(fx["field"])
    assert field == s.IcgemGravityField.parse_truncated(fx["field"], 2, 2)
    assert field != s.IcgemGravityField.parse_truncated(fx["field"], 2, 0)
    assert (
        field.header()
        == s.IcgemGravityField.parse_truncated(fx["field"], 2, 0).header()
    )
    subset = s.IcgemGravityField.parse_truncated(fx["field"], 2, 0)
    assert subset.stored_max_degree() == 2
    assert subset.stored_max_order() == 0
    assert subset.static_coefficient(2, 0).c == -4.8e-4
    assert field == field.with_assumed_tide_system(s.TideSystem.TIDE_FREE)
    header = field.header()
    assert [
        header.model_name,
        header.gravity_constant_m3_s2,
        header.radius_m,
        header.max_degree,
        header.errors,
        header.normalization,
        header.tide_system,
        header.format,
    ] == [
        "TEST",
        3.986e14,
        6378000.0,
        2,
        s.IcgemErrors.NO,
        s.IcgemNormalization.FULLY_NORMALIZED,
        s.TideSystem.TIDE_FREE,
        s.IcgemFormat.ICGEM1_0,
    ]
    assert field.stored_max_degree() == field.stored_max_order() == 2
    assert field.static_sigmas(2, 0) is None
    assert not field.is_time_variable()
    assert field.time_variable_terms() == []
    coefficient = field.static_coefficient(2, 0)
    assert [coefficient.degree, coefficient.order, coefficient.c, coefficient.s] == [
        2,
        0,
        -4.8e-4,
        0.0,
    ]
    assert field.coefficient_at(2, 2, 0.0).c == 0.0
    gravity = field.gravity(2, 2)
    assert gravity == field.gravity_in_tide_system(2, 2, s.TideSystem.TIDE_FREE)
    assert gravity != field.gravity(2, 0)
    assert not gravity.is_time_variable()
    assert gravity.coefficient_at(2, 0, 0.0).c == -4.8e-4
    position = [7000.0, 1000.0, 200.0]
    exact(
        [
            gravity.body_fixed_acceleration_km_s2(position),
            gravity.body_fixed_acceleration_at_km_s2(position, 0.0),
            gravity.body_fixed_potential_km2_s2(position),
            gravity.body_fixed_potential_at_km2_s2(position, 0.0),
        ],
        fx["gravity"],
    )
    assert s.IcgemSigmas().calibrated is None
    assert s.IcgemSigmas().formal is None
    assert s.IcgemTermKind.CONSTANT.label == "constant"
    assert s.IcgemTermKind.TREND.label == "trend"
    assert s.IcgemTermKind.cosine(2.0).period_years == 2.0
    assert s.IcgemTermKind.sine(3.0).period_years == 3.0

    with pytest.raises(s.IcgemError) as caught:
        s.IcgemGravityField.parse("")
    assert caught.value.detail == {"kind": "missing_end_of_head"}
    with pytest.raises(s.IcgemError) as caught:
        field.with_assumed_tide_system(s.TideSystem.ZERO_TIDE)
    assert caught.value.detail == {
        "kind": "tide_system_conflict",
        "stated": s.TideSystem.TIDE_FREE,
        "assumed": s.TideSystem.ZERO_TIDE,
    }


POSITION = [7000.0, 1000.0, 200.0]
VELOCITY = [-1.0, 7.5, 0.8]
SUN = [130_000_000.0, 10_000_000.0, 5_000_000.0]
MOON = [300_000.0, -100_000.0, 50_000.0]


def check_partials(actual, expected):
    exact(actual.wrt_position, expected["position"])
    exact(actual.wrt_velocity, expected["velocity"])


def test_estimable_force_evaluations(orbital):
    fx = orbital["radiation"]
    state = s.CartesianState(0.0, POSITION, VELOCITY)
    empirical = s.EmpiricalRtnAcceleration().with_term(
        s.RtnAxis.RADIAL, s.EmpiricalHarmonic.CONSTANT, 1e-8
    )
    srp = s.SolarRadiationPressure(1.5, 0.02)
    erp = s.EarthRadiationPressure(1.5, 0.02)
    ecom1 = s.Ecom1SolarRadiationPressure(s.Ecom1Coefficients(d0=2e-9))
    ecom2 = s.Ecom2SolarRadiationPressure.d2b1(s.Ecom2D2b1Coefficients(d0=2e-9))
    for name, force in [
        ("empirical", empirical),
        ("srp", srp),
        ("erp", erp),
        ("ecom1", ecom1),
        ("ecom2", ecom2),
    ]:
        f = fx[name]
        parameters = force.parameters()
        assert force.parameter_count() == len(f["parameters"])
        exact([[p.name, p.value] for p in parameters], f["parameters"])
        values = [p.value for p in parameters]
        exact(force.acceleration(state), f["acceleration"])
        exact(
            force.acceleration_with_parameters(values, state),
            f["acceleration_with_parameters"],
        )
        check_partials(force.acceleration_partials(values, state), f["partials"])
        exact(force.parameter_partials(values, state), f["parameter_partials"])
    for key, force in [("ecom1", ecom1), ("ecom2", ecom2)]:
        day = force.with_moon_eclipse(False)
        assert not day.moon_eclipse_enabled()
        assert force.moon_eclipse_enabled()
        assert force.shadow_function() == s.ShadowFunction.APPARENT_DISC
        assert force.earth_shadow_model() == s.EarthShadowModel.WGS84_OBLATE
        conical = force.with_shadow_function(s.ShadowFunction.CONICAL)
        assert conical.shadow_function() == s.ShadowFunction.CONICAL
        assert not conical.moon_eclipse_enabled()
        assert (
            force.with_earth_shadow_model(
                s.EarthShadowModel.SPHERICAL
            ).earth_shadow_model()
            == s.EarthShadowModel.SPHERICAL
        )
        a, d, p = fx[key + "_sun"]
        exact(day.acceleration_with_sun(POSITION, VELOCITY, SUN), a)
        check_partials(day.acceleration_partials_with_sun(POSITION, VELOCITY, SUN), d)
        exact(day.parameter_partials_with_sun(POSITION, VELOCITY, SUN), p)
        a, d, p = fx[key + "_sun_moon"]
        exact(force.acceleration_with_sun_moon(POSITION, VELOCITY, SUN, MOON), a)
        check_partials(
            force.acceleration_partials_with_sun_moon(POSITION, VELOCITY, SUN, MOON), d
        )
        exact(force.parameter_partials_with_sun_moon(POSITION, VELOCITY, SUN, MOON), p)
    for name, geometry in (
        ("geometry", s.ecom_geometry(POSITION, VELOCITY, SUN)),
        (
            "geometry_with_shadow",
            s.ecom_geometry_with_shadow(
                POSITION, VELOCITY, SUN, MOON, s.EarthShadowModel.WGS84_OBLATE
            ),
        ),
    ):
        exact(
            [
                geometry.dyb.d,
                geometry.dyb.y,
                geometry.dyb.b,
                geometry.delta_u_rad,
                geometry.argument_of_latitude_rad,
                geometry.lighting_ratio,
            ],
            fx[name],
        )

    plain_geometry = s.ecom_geometry(POSITION, VELOCITY, SUN)
    assert plain_geometry == s.ecom_geometry(
        tuple(POSITION), tuple(VELOCITY), tuple(SUN)
    )
    assert plain_geometry != s.ecom_geometry(
        POSITION, VELOCITY, [-value for value in SUN]
    )

    assert not erp.moon_eclipse_enabled()
    erp = erp.with_moon_eclipse(True)
    assert erp.moon_eclipse_enabled()
    assert s.ForceModelComponents().earth_radiation_pressure is None
    assert (
        s.ForceModelComponents()
        .with_earth_radiation_pressure(erp)
        .earth_radiation_pressure
        == erp
    )
    exact(erp.acceleration_with_sun_moon(state, SUN, MOON), fx["erp_sun_moon"])
    exact(srp.acceleration_with_sun_moon(POSITION, SUN, MOON), fx["srp_sun_moon"])
    assert srp.cr == 1.5
    assert srp.area_to_mass_m2_kg == 0.02
    assert srp.au_km == 149_597_870.7
    assert srp.moon_eclipse_enabled()
    assert srp.shadow_function() == s.ShadowFunction.APPARENT_DISC
    assert srp.earth_shadow_model() == s.EarthShadowModel.WGS84_OBLATE
    assert (
        srp.with_shadow_function(s.ShadowFunction.CONICAL).shadow_function()
        == s.ShadowFunction.CONICAL
    )
    assert not srp.with_moon_eclipse(False).moon_eclipse_enabled()
    assert (
        srp.with_earth_shadow_model(s.EarthShadowModel.SPHERICAL).earth_shadow_model()
        == s.EarthShadowModel.SPHERICAL
    )
    assert s.SolarRadiationPressure.CR_PARAMETER == "srp.cr"
    assert s.ForceParameter("srp.cr", 1.5) == s.ForceParameter("srp.cr", 1.5)
    assert s.ForceParameter("srp.cr", 1.5) != s.ForceParameter("srp.cr", 1.0)
    zeros = [[0.0] * 3 for _ in range(3)]
    assert s.AccelerationPartials.zeros() == s.AccelerationPartials.position_only(zeros)
    assert s.AccelerationPartials.zeros().wrt_position == zeros
    assert s.AccelerationPartials.zeros().wrt_velocity == zeros
    assert (
        s.AccelerationPartials.position_only(np.eye(3)).wrt_position
        == np.eye(3).tolist()
    )
    assert (
        s.AccelerationPartials.position_only(np.eye(3))
        != s.AccelerationPartials.zeros()
    )
    with pytest.raises(s.SolveError) as caught:
        srp.acceleration_with_parameters([], state)
    assert caught.value.detail == {
        "kind": "invalid_input",
        "message": "force model takes 1 parameter values, got 0",
    }


def test_empirical_term_selection_and_ecom_coefficients():
    axes = s.RtnAxis.ALL
    terms = s.EmpiricalHarmonic.ALL
    assert axes == [s.RtnAxis.RADIAL, s.RtnAxis.ALONG_TRACK, s.RtnAxis.CROSS_TRACK]
    assert terms == [
        s.EmpiricalHarmonic.CONSTANT,
        s.EmpiricalHarmonic.COSINE,
        s.EmpiricalHarmonic.SINE,
    ]
    assert s.EmpiricalRtnAcceleration().terms() == []
    assert s.EmpiricalRtnAcceleration.constant_terms().terms() == [
        (axis, terms[0], 0.0) for axis in axes
    ]
    assert s.EmpiricalRtnAcceleration.all_terms().terms() == [
        (axis, term, 0.0) for axis in axes for term in terms
    ]
    selected = s.EmpiricalRtnAcceleration.with_zero_terms([(axes[0], terms[0])])
    assert selected.coefficient_km_s2(axes[0], terms[0]) == 0.0
    assert selected.coefficient_km_s2(axes[0], terms[1]) is None
    assert (
        s.EmpiricalRtnAcceleration.parameter_name(axes[0], terms[0])
        == "empirical.radial.constant"
    )
    updated = selected.with_term(axes[0], terms[0], 1e-8)
    assert updated.terms() == [(s.RtnAxis.RADIAL, s.EmpiricalHarmonic.CONSTANT, 1e-8)]
    assert selected.terms() == [(s.RtnAxis.RADIAL, s.EmpiricalHarmonic.CONSTANT, 0.0)]
    assert selected != s.EmpiricalRtnAcceleration()
    for cls, count, names in [
        (
            s.Ecom1Coefficients,
            s.ECOM1_COEFFICIENT_COUNT,
            ["d0", "dc", "ds", "y0", "yc", "ys", "b0", "bc", "bs"],
        ),
        (
            s.Ecom2D2b1Coefficients,
            s.ECOM2_D2B1_COEFFICIENT_COUNT,
            ["b0", "b1c", "b1s", "d0", "d2c", "d2s", "y0"],
        ),
        (
            s.Ecom2D4b1Coefficients,
            s.ECOM2_D4B1_COEFFICIENT_COUNT,
            ["b0", "b1c", "b1s", "d0", "d2c", "d4c", "d2s", "d4s", "y0"],
        ),
    ]:
        assert count == len(names)
        assert cls.NAMES == [name.capitalize() for name in names]
        prefix = "ecom1." if cls is s.Ecom1Coefficients else "ecom2."
        assert cls.PARAMETER_NAMES == [prefix + name for name in names]
        values = [float(index) for index in range(1, count + 1)]
        coefficients = cls.from_array(values)
        assert [getattr(coefficients, name) for name in names] == values
        assert coefficients.to_array() == values
        assert coefficients == cls(**dict(zip(names, values, strict=True)))
        assert coefficients != cls()
        assert cls().to_array() == [0.0] * count
    ecom1 = s.Ecom1SolarRadiationPressure(s.Ecom1Coefficients(d0=2e-9))
    assert ecom1.coefficients().to_array() == [2e-9, *([0.0] * 8)]
    assert ecom1 == s.Ecom1SolarRadiationPressure(
        s.Ecom1Coefficients.from_array([2e-9, *([0.0] * 8)])
    )
    assert ecom1 != ecom1.with_moon_eclipse(False)
    model = s.Ecom2SolarRadiationPressure.d4b1(s.Ecom2D4b1Coefficients(d4s=2e-9))
    assert (model.n_d(), model.n_b()) == (2, 1)
    assert model.coefficients() == [0, 0, 0, 0, 0, 0, 0, 2e-9, 0]
    assert s.Ecom2SolarRadiationPressure.coefficient_count(2, 1) == 9
    assert (
        s.Ecom2SolarRadiationPressure.parameter_names(2, 1)
        == s.Ecom2D4b1Coefficients.PARAMETER_NAMES
    )
    assert model == s.Ecom2SolarRadiationPressure(2, 1, model.coefficients())


def test_analytic_orbits_and_attitude(orbital):
    fx = orbital["analytic"]
    gravity = s.ZonalHarmonics.earth()
    g_fields = ("mu_km3_s2", "re_km", "c20", "c30", "c40", "c50", "c60")
    exact([getattr(gravity, field) for field in g_fields], fx["gravity"])
    assert gravity == s.ZonalHarmonics.default()
    assert gravity == s.ZonalHarmonics(*(hex_to_f64(value) for value in fx["gravity"]))
    circular = s.CircularElements(7000.0, 0.001, 0.002, 0.7, 0.3, 0.4)
    keplerian = s.AnalyticKeplerianElements(7000.0, 0.03, 0.7, 0.2, 0.3, 0.4)
    exact(circular.eccentricity(), fx["eccentricity"])
    exact(circular.to_cartesian(gravity.mu_km3_s2), fx["circular_cartesian"])
    exact(keplerian.to_cartesian(gravity.mu_km3_s2), fx["keplerian_cartesian"])
    assert s.MeanConversionOptions.ECKSTEIN_HECHLER_DEFAULT.tolerance == 1e-11
    assert s.MeanConversionOptions.ECKSTEIN_HECHLER_DEFAULT.max_iterations == 100
    assert s.MeanConversionOptions.BROUWER_LYDDANE_DEFAULT.tolerance == 1e-13
    assert s.MeanConversionOptions.BROUWER_LYDDANE_DEFAULT.max_iterations == 200
    for prefix, cls, elements, fields, more in [
        (
            "eckstein_hechler",
            s.EcksteinHechlerPropagator,
            circular,
            ("a_km", "ex", "ey", "i_rad", "raan_rad", "mean_latitude_argument_rad"),
            [],
        ),
        (
            "brouwer_lyddane",
            s.BrouwerLyddanePropagator,
            keplerian,
            ("a_km", "e", "i_rad", "arg_perigee_rad", "raan_rad", "mean_anomaly_rad"),
            [1e-17],
        ),
    ]:
        values = fx[prefix]
        propagate = cls.from_mean_elements(0.0, elements, gravity, *more)
        assert propagate.epoch_tdb_seconds() == 0.0
        assert propagate.mean_elements() == elements
        assert propagate.gravity() == gravity
        if more:
            assert propagate.m2_rad_s2() == 1e-17
        for index, state in enumerate(propagate.ephemeris([0.0, 600.0])):
            exact([state.position_km, state.velocity_km_s], values[index])
        exact(
            [getattr(propagate.osculating_elements(600.0), field) for field in fields],
            values[2],
        )
        osculating = getattr(s, prefix + "_osculating_elements")(elements, gravity)
        exact([getattr(osculating, field) for field in fields], values[3])
        initial = propagate.propagate(0.0)
        exact([initial.position_km, initial.velocity_km_s], values[0])
        options = getattr(s.MeanConversionOptions, prefix.upper() + "_DEFAULT")
        assert options == s.MeanConversionOptions(
            1e-13 if more else 1e-11, 200 if more else 100
        )
        mean = getattr(s, prefix + "_mean_elements")(initial, gravity, options)
        assert mean.iterations == values[4]
        exact([getattr(mean.elements, field) for field in fields], values[5])
        assert cls(initial, gravity, *more, options).mean_elements() == mean.elements
        converted = type(elements).from_cartesian(
            initial.position_km, initial.velocity_km_s, gravity.mu_km3_s2
        )
        exact([getattr(converted, field) for field in fields], values[6])
        with pytest.raises(s.AnalyticPropagationError) as caught:
            elements.to_cartesian(0.0)
        assert caught.value.detail == {
            "kind": "invalid_parameter",
            "field": "mu_km3_s2",
            "reason": "must be positive",
        }
    assert str(s.AnalyticTheory.ECKSTEIN_HECHLER) == "Eckstein-Hechler"
    assert str(s.AnalyticTheory.BROUWER_LYDDANE) == "Brouwer-Lyddane"
    bl = s.BrouwerLyddanePropagator.from_mean_elements(0.0, keplerian, gravity, 1e-17)
    sample = s.OrbitAttitudeTrajectory.from_brouwer_lyddane(bl).sample(600.0)
    assert sample.position_m == [
        hex_to_f64(value) * 1000.0 for value in fx["brouwer_lyddane"][1][0]
    ]
    eh = s.EcksteinHechlerPropagator.from_mean_elements(0.0, circular, gravity)
    trajectory = s.OrbitAttitudeTrajectory.from_eckstein_hechler(eh)
    turn_orbit = s.EcksteinHechlerPropagator.from_mean_elements(
        0.0, s.CircularElements(26560.0, 0.0, 0.0, 0.409, 0.0, 4.9), gravity
    )
    turn = s.gnss_attitude(
        s.GnssAttitudeLaw.for_block(s.GnssBlock.GPS_IIR),
        s.OrbitAttitudeTrajectory.from_eckstein_hechler(turn_orbit),
        0.0,
    )
    assert turn.mode == s.AttitudeMode.NOON_TURN
    exact([turn.yaw_rad, turn.body_x, turn.body_y, turn.body_z], fx["turn"])
    law = s.GnssAttitudeLaw.orbit_normal()
    attitude = s.gnss_attitude(law, trajectory, 600.0)
    assert attitude.mode == s.AttitudeMode.ORBIT_NORMAL
    exact(
        [
            attitude.yaw_rad,
            attitude.nominal_yaw_rad,
            attitude.beta_rad,
            attitude.orbit_angle_rad,
            attitude.body_x,
            attitude.body_y,
            attitude.body_z,
        ],
        fx["attitude"],
    )
    sample = trajectory.sample(600.0)
    assert sample == s.AttitudeSample(
        sample.position_m, sample.velocity_m_s, sample.sun_position_m
    )
    block = s.GnssBlock.from_igs_name("BEIDOU-3G-CAST")
    assert block == s.GnssBlock.BEIDOU3_GEO
    assert block.igs_name() == str(block) == "BEIDOU-3G-CAST"
    assert s.GnssAttitudeLaw.for_block(block) == law
    with pytest.raises(s.GnssAttitudeError) as caught:
        s.GnssAttitudeLaw.for_block(s.GnssBlock.GPS_IIA)
    assert caught.value.detail == {"kind": "no_default_yaw_rate", "prn": None}
    with pytest.raises(s.GnssBlockError) as caught:
        s.GnssBlock.from_igs_name("bad")
    assert caught.value.detail == {"kind": "unknown", "name": "bad"}


def test_variational_and_force_parameter_fit(orbital):
    fx = orbital["estimation"]
    initial = s.CartesianState(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.5])
    f = s.EmpiricalRtnAcceleration().with_term(
        s.RtnAxis.RADIAL, s.EmpiricalHarmonic.CONSTANT, 1e-8
    )
    eq = s.VariationalEquations(f, [1e-8], [0])
    epoch = s.VariationalEpoch.from_tdb_seconds(0.0, 120.0)
    assert epoch.tdb_j2000_s == 120.0
    exact(epoch.elapsed_tt_s, fx["variational"]["epoch"])
    assert epoch == s.VariationalEpoch(120.0, hex_to_f64(fx["variational"]["epoch"]))
    assert epoch != s.VariationalEpoch(120.0, 0.0)
    assert eq.force is f
    assert eq.parameters == [1e-8]
    assert eq.sensitivity_parameters == [0]
    (result,) = s.propagate_variational(initial, [epoch], eq, s.Integrator.DP54)
    exact(
        [result.state.position_km, result.state.velocity_km_s],
        fx["variational"]["state"],
    )
    exact(result.state_transition, fx["variational"]["stm"])
    exact(result.sensitivity[:, 0], fx["variational"]["sensitivity"])
    same_epoch = s.VariationalEpoch(120.0, hex_to_f64(fx["variational"]["epoch"]))
    (same_result,) = s.propagate_variational(
        initial, [same_epoch], eq, s.Integrator.DP54
    )
    (at_start,) = s.propagate_variational(
        initial, [s.VariationalEpoch(0.0, 0.0)], eq, s.Integrator.DP54
    )
    assert result == same_result
    assert result != at_start
    with pytest.raises(s.SolveError) as caught:
        s.propagate_variational(
            initial, [epoch], s.VariationalEquations(f, [], []), s.Integrator.DP54
        )
    assert caught.value.detail == {
        "kind": "invalid_input",
        "message": "force model takes 1 parameter values, got 0",
    }
    fit_fx = fx["fit"]
    samples = [
        s.InertialPositionSample(
            s.ClockInstant.from_nanos(s.TimeScale.TDB, sec * 10**9),
            [hex_to_f64(value) for value in position],
        )
        for sec, position in zip(fit_fx["epochs"], fit_fx["observations"], strict=True)
    ]
    options = s.OrbitParameterFitOptions(force_model=s.ForceModelKind.two_body())
    fit = s.fit_inertial_orbit_parameters(samples, "G01", None, options)
    assert fit.satellite == "G01"
    assert fit.parameter_names == s.STATE_PARAMETER_NAMES
    assert fit.force_parameters == []
    assert fit.force_parameter("bad") is None
    assert fit.iterations == fit_fx["iterations"]
    assert fit.covariance.is_estimated
    assert fit.covariance != s.OrbitParameterCovariance.UNBOUNDED
    unbounded = s.OrbitParameterCovariance.UNBOUNDED
    assert not unbounded.is_estimated
    assert [
        unbounded.matrix,
        unbounded.correlation,
        unbounded.variance_factor,
        unbounded.degrees_of_freedom,
    ] == [None] * 4
    assert fit.covariance.degrees_of_freedom == 9
    exact(
        [
            fit.covariance.matrix,
            fit.covariance.correlation,
            fit.covariance.variance_factor,
            fit.covariance.degrees_of_freedom,
        ],
        fit_fx["covariance"],
    )
    exact(fit.residual_stats.rms_3d_m, fit_fx["rms_3d_m"])
    exact(fit.sigma("x_km"), fit_fx["sigma"])
    exact(fit.correlation("x_km", "x_km"), fit_fx["correlation"])
    exact(fit.condition_number, fit_fx["condition_number"])
    exact(fit.seed_rms_3d_m, fit_fx["seed_rms_3d_m"])
    exact(
        [fit.initial_state.position_km, fit.initial_state.velocity_km_s],
        fit_fx["initial_state"],
    )
    exact(
        [
            [r.epoch_tdb_j2000_s, r.elapsed_tt_s, r.radial_m, r.along_m, r.cross_m]
            for r in fit.residuals
        ],
        fit_fx["residuals"],
    )
    assert fit.residuals[0].variational_epoch() == s.VariationalEpoch(
        hex_to_f64(fit_fx["initial_epoch"]), 0.0
    )
    validated, degradation = s.fit_inertial_orbit_parameters_with_validity(
        samples, "G01", None, options, s.ValidityMode.STRICT
    )
    assert validated == fit
    assert validated.residuals == fit.residuals
    assert validated.covariance == fit.covariance
    assert fit.residuals[0] != fit.residuals[1]
    assert degradation is None
    with pytest.raises(s.OrbitParameterFitError) as caught:
        s.fit_inertial_orbit_parameters([], "G01", None, options)
    assert caught.value.detail == {
        "kind": "too_few_samples",
        "satellite": "G01",
        "got": 0,
        "required": 2,
    }


def test_context_covariance_and_decay():
    initial = s.CartesianState(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0])
    context = s.PropagationContext()
    options = s.IntegratorOptions()
    assert [
        options.abs_tol,
        options.rel_tol,
        options.min_step,
        options.max_step,
        options.initial_step,
        options.max_steps,
        options.dense_output,
    ] == [1e-9, 1e-12, 1e-6, 3600.0, 60.0, 1_000_000, False]
    prop = s.StatePropagator(
        initial, s.ForceModelKind.two_body(), s.Integrator.DP54, options
    )
    covariance = np.eye(6)
    state, actual = prop.propagate_state_with_covariance_with_context(
        covariance, 0.0, context
    )
    assert state.position_km.tolist() == [7000.0, 0.0, 0.0]
    np.testing.assert_array_equal(actual, covariance)
    np.testing.assert_array_equal(
        prop.state_transition_matrix_to_with_context(0.0, context), np.eye(6)
    )
    ephemeris = prop.propagate_covariance_with_context(
        s.LabeledCovariance6(covariance), [0.0], context
    )
    assert len(ephemeris) == 1
    np.testing.assert_array_equal(ephemeris.covariance_at(0.0), covariance)
    assert prop.ephemeris_with_context([0.0], context)[0].velocity_km_s.tolist() == [
        0.0,
        7.5,
        0.0,
    ]
    drag = s.DragParameters.from_bc_factor_m2_kg(0.02)
    decay = s.estimate_decay_with_context(
        0.0, [6400.0, 0.0, 0.0], [0.0, 7.8, 0.0], drag, context
    )
    assert decay.time_to_decay_s == 0.0
    with pytest.raises(s.SolveError) as caught:
        s.estimate_decay_with_context(
            0.0, [6400.0, 0.0, 0.0], [0.0, 7.8, 0.0], drag, context, scan_step_s=0.0
        )
    assert caught.value.detail == {"kind": "invalid_config", "field": "scan_step_s"}


def test_time_variable_icgem_terms(orbital):
    static = orbital["terrestrial"]["field"]
    variable = static.replace(
        "gfc 2 0 -4.8e-4 0", "gfct 2 0 -4.8e-4 0 20000101\ntrnd 2 0 1e-7 0"
    )
    field = s.IcgemGravityField.parse(variable)
    assert field.is_time_variable()
    assert field.static_coefficient(2, 0) is None
    constant, trend = field.time_variable_terms()
    assert constant.kind == s.IcgemTermKind.CONSTANT
    assert trend.kind == s.IcgemTermKind.TREND
    assert constant != trend
    assert (
        constant.line,
        constant.degree,
        constant.order,
        constant.c,
        constant.s,
        constant.reference_tt_seconds,
        constant.valid_from_tt_seconds,
        constant.valid_until_tt_seconds,
        constant.sigmas,
    ) == (9, 2, 0, -4.8e-4, 0.0, -43200.0, -float("inf"), float("inf"), s.IcgemSigmas())
    assert field.coefficient_at(2, 0, -43200.0).c == -4.8e-4
    gravity = field.gravity(2, 2)
    assert gravity.is_time_variable()
    assert gravity.coefficient_at(2, 0, -43200.0).c == -4.8e-4
    with pytest.raises(s.IcgemError) as caught:
        field.coefficient_at(2, 0, float("nan"))
    assert caught.value.detail == {"kind": "non_finite_epoch"}


def test_central_gravity_drag_and_force_composition(orbital):
    force = s.ForceModel.TWO_BODY
    drag = s.DragParameters.from_bc_factor_m2_kg(0.02)
    empirical = s.EmpiricalRtnAcceleration().with_term(
        s.RtnAxis.RADIAL, s.EmpiricalHarmonic.CONSTANT, 1e-8
    )
    composite = s.CompositeForceModel([force, empirical])
    state = s.CartesianState(0.0, POSITION, VELOCITY)
    for name, model in [
        ("gravity", force),
        ("drag", drag),
        ("drag", drag.to_force()),
        ("composite", composite),
    ]:
        fx = orbital["radiation"][name]
        exact([[p.name, p.value] for p in model.parameters()], fx["parameters"])
        assert model.parameter_count() == len(fx["parameters"])
        parameters = [hex_to_f64(value) for _, value in fx["parameters"]]
        exact(model.acceleration(state), fx["acceleration"])
        exact(
            model.acceleration_with_parameters(parameters, state),
            fx["acceleration_with_parameters"],
        )
        check_partials(model.acceleration_partials(parameters, state), fx["partials"])
        exact(model.parameter_partials(parameters, state), fx["parameter_partials"])


def test_state_propagation_with_additional_forces(orbital):
    epoch = 120_614_400.0
    targets = [epoch + 120.0, epoch + 240.0]
    position = [6778.0, 0.0, 0.0]
    velocity = [0.0, 7.5, 0.5]
    initial = s.CartesianState(epoch, position, velocity)
    gravity = s.ForceModelKind.two_body_j2()
    additional = [
        s.EmpiricalRtnAcceleration().with_term(
            axis, s.EmpiricalHarmonic.CONSTANT, value
        )
        for axis, value in [(s.RtnAxis.RADIAL, 1e-8), (s.RtnAxis.ALONG_TRACK, 2e-8)]
    ]
    drag = s.DragParameters.from_bc_factor_m2_kg(0.02)
    weather = s.load_space_weather(
        Path(CORE_FIXTURES) / "space_weather/SW-All-20260702-trim.csv"
    )
    context = s.PropagationContext()
    options = s.IntegratorOptions(dense_output=True)

    for case, drag_parameters, table in (
        ("base", None, None),
        ("fixed", drag, None),
        ("sourced", drag, weather),
    ):
        expected = orbital["additional_force_propagation"][case]
        arc = s.propagate_state(
            epoch,
            np.array(position),
            np.array(velocity),
            np.array(targets),
            force_model=gravity,
            drag=drag_parameters,
            space_weather_table=table,
            additional_forces=additional,
        )
        exact(list(zip(arc.position_km, arc.velocity_km_s)), expected["ephemeris"])
        propagator = s.StatePropagator(
            initial, gravity, s.Integrator.DP54, options, drag_parameters, table
        )
        states = propagator.ephemeris_with_context(
            targets, context, additional_forces=additional
        )
        exact(
            [[state.position_km, state.velocity_km_s] for state in states],
            expected["ephemeris"],
        )
        result = propagator.propagate_to(
            targets[-1], context, additional_forces=additional
        )
        final = result.final_state
        assert final.epoch_tdb_seconds == targets[-1]
        exact([final.position_km, final.velocity_km_s], expected["final_state"])
        assert result.dense is not None
        assert result.dense.coverage == (epoch, targets[-1])
        sample = result.dense.eval(targets[0])
        exact([sample.position_km, sample.velocity_km_s], expected["dense_state"])

    with pytest.raises(s.SolveError) as caught:
        s.propagate_state(
            epoch,
            np.array(position),
            np.array(velocity),
            np.array(targets),
            space_weather_table=weather,
            additional_forces=additional,
        )
    assert caught.value.detail == {
        "kind": "invalid_input",
        "message": "space weather source without drag",
    }
    with pytest.raises(TypeError, match="expected a concrete force model"):
        s.propagate_state(
            epoch,
            np.array(position),
            np.array(velocity),
            np.array(targets),
            additional_forces=[gravity],
        )


def test_released_force_configuration_identity():
    for factory in (
        lambda: s.SolarRadiationPressure(1.2, 0.02),
        s.ForceModelComponents,
        s.ForceModelKind.two_body,
        lambda: s.DragParameters.from_bc_factor_m2_kg(0.02),
        lambda: s.DragForce.from_bc_factor_m2_kg(0.02),
    ):
        first, second = factory(), factory()
        assert first != second
        assert {first: 1, second: 2} == {first: 1, second: 2}
        assert len({first, first, second}) == 2


def test_force_parameter_estimate(orbital):
    fx = orbital["estimation"]
    estimate = fx["force_fit"]
    axis = s.RtnAxis.RADIAL
    term = s.EmpiricalHarmonic.CONSTANT
    force = s.EmpiricalRtnAcceleration.with_zero_terms([(axis, term)])
    name = "empirical.radial.constant"
    options = s.OrbitParameterFitOptions(
        force_model=s.ForceModelKind.two_body(),
        additional_forces=[force],
        estimated_parameters=[name],
    )
    assert options.additional_forces == [force]
    assert options.estimated_parameters == [name]
    assert options.drag is None and options.space_weather is None
    assert options.force_model.label == "two_body"
    assert options.integrator == s.Integrator.DP54
    assert options.integrator_options == s.IntegratorOptions()
    assert options.propagation_context.de_ephemeris() is None
    assert options.observation_sigma_m == 1.0
    assert options.max_iterations == 25
    assert options.convergence_threshold == 1e-3
    assert options.max_condition_number == 1e12
    defaults = s.OrbitParameterFitOptions()
    assert defaults.force_model.label == "composite"
    assert defaults.integrator.label == "dp54"
    assert defaults.estimated_parameters == []
    assert options.min_ledger_samples == 3
    samples = [
        s.InertialPositionSample(
            s.ClockInstant.from_nanos(s.TimeScale.TDB, sec * 10**9),
            [hex_to_f64(value) for value in position],
        )
        for sec, position in zip(
            fx["fit"]["epochs"], estimate["observations"], strict=True
        )
    ]
    exact(samples[0].position_gcrf_m, estimate["observations"][0])
    assert samples[0].epoch == s.ClockInstant.from_nanos(s.TimeScale.TDB, 0)
    assert samples[0] == s.InertialPositionSample(
        s.ClockInstant.from_nanos(s.TimeScale.TDB, 0),
        [hex_to_f64(value) for value in estimate["observations"][0]],
    )
    assert samples[0] != samples[1]
    fit = s.fit_inertial_orbit_parameters(samples, "G01", None, options)
    assert fit.parameter_names == [
        "x_km",
        "y_km",
        "z_km",
        "vx_km_s",
        "vy_km_s",
        "vz_km_s",
        name,
    ]
    (parameter,) = fit.force_parameters
    assert fit.force_parameter(name) == parameter
    assert parameter.name == estimate["name"]
    exact(parameter.initial_value, estimate["initial"])
    exact(parameter.value, estimate["value"])
    exact(fit.condition_number, estimate["condition"])


def test_gnss_yaw_geometry_and_published_laws(orbital):
    nominal = s.nominal_yaw([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    exact(
        [
            nominal.yaw_rad,
            nominal.beta_rad,
            nominal.body_x,
            nominal.body_y,
            nominal.body_z,
        ],
        orbital["analytic"]["nominal"],
    )
    axes = s.dyb_axes([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    assert [axes.d, axes.y, axes.b] == [
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
    ]
    with pytest.raises(s.GnssAttitudeError) as caught:
        s.dyb_axes([0.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    assert caught.value.detail == {"kind": "zero_length", "field": "position_m"}

    def radians(degrees):
        return degrees * math.pi / 180.0

    for name, degrees in {
        "GPS_IIA_DEFAULT_YAW_BIAS_RAD": 0.5,
        "GPS_IIR_DEFAULT_YAW_RATE_RAD_S": 0.2,
        "GPS_IIF_DEFAULT_YAW_RATE_RAD_S": 0.11,
        "GPS_IIF_DEFAULT_YAW_BIAS_RAD": -0.7,
        "GPS_SHADOW_LIMIT_RAD": 166.75,
        "GLONASS_M_DEFAULT_YAW_RATE_RAD_S": 0.25,
        "GLONASS_M_SHADOW_LIMIT_RAD": 165.8,
        "GALILEO_IOV_BETA_X_RAD": 15.0,
        "GALILEO_IOV_BETA_Y_RAD": 2.0,
        "GPS_III_BETA_X_RAD": 15.0,
        "GPS_III_BETA_Y_RAD": 5.8,
        "BEIDOU3_SECM_BETA0_RAD": 3.0,
        "BEIDOU2_YS_TO_ON_BETA_RAD": 4.1,
        "BEIDOU2_ON_TO_YS_BETA_RAD": 3.9,
    }.items():
        assert getattr(s, name) == radians(degrees)

    turn = s.CosineTurnStartAngle
    for value, beta, inclusive, angle, threshold, period in [
        (s.GALILEO_FOC_TURN, 4.1, False, turn.COLINEARITY, 10.0, 5656.0),
        (s.BEIDOU3_CAST_MEO_TURN, 3.0, True, turn.ORBIT_ANGLE, 6.0, 3090.0),
        (s.BEIDOU3_CAST_IGSO_TURN, 3.0, True, turn.ORBIT_ANGLE, 6.0, 5740.0),
    ]:
        assert value == s.CosineTurnParameters(
            radians(beta), inclusive, angle, radians(threshold), period
        )
        assert [
            value.beta_limit_rad,
            value.beta_limit_inclusive,
            value.start_angle,
            value.start_angle_rad,
            value.period_s,
        ] == [
            radians(beta),
            inclusive,
            angle,
            radians(threshold),
            period,
        ]

    block = s.GnssBlock
    law = s.GnssAttitudeLaw
    for kind, name, system in [
        (block.GPS_IIA, "BLOCK IIA", s.GnssSystem.GPS),
        (block.GPS_IIR, "BLOCK IIR-B", s.GnssSystem.GPS),
        (block.GPS_IIRM, "BLOCK IIR-M", s.GnssSystem.GPS),
        (block.GPS_IIF, "BLOCK IIF", s.GnssSystem.GPS),
        (block.GPS_III, "BLOCK IIIA", s.GnssSystem.GPS),
        (block.GLONASS_M, "GLONASS-M", s.GnssSystem.GLONASS),
        (block.GALILEO_IOV, "GALILEO-1", s.GnssSystem.GALILEO),
        (block.GALILEO_FOC, "GALILEO-2", s.GnssSystem.GALILEO),
        (block.BEIDOU2_GEO, "BEIDOU-2G", s.GnssSystem.BEIDOU),
        (block.BEIDOU2_IGSO, "BEIDOU-2I", s.GnssSystem.BEIDOU),
        (block.BEIDOU2_MEO, "BEIDOU-2M", s.GnssSystem.BEIDOU),
        (block.BEIDOU3_GEO, "BEIDOU-3G-CAST", s.GnssSystem.BEIDOU),
        (block.BEIDOU3_IGSO, "BEIDOU-3I", s.GnssSystem.BEIDOU),
        (block.beidou3_meo(), "BEIDOU-3M", s.GnssSystem.BEIDOU),
        (
            block.beidou3_meo(s.BeidouManufacturer.CAST),
            "BEIDOU-3M-CAST",
            s.GnssSystem.BEIDOU,
        ),
        (
            block.beidou3_meo(s.BeidouManufacturer.SECM),
            "BEIDOU-3M-SECM",
            s.GnssSystem.BEIDOU,
        ),
    ]:
        assert kind == block.from_igs_name(name)
        assert hash(kind) == hash(block.from_igs_name(name))
        assert (kind.system(), kind.igs_name(), str(kind)) == (system, name, name)
    assert block.GPS_IIA < block.GPS_IIR
    for orbit, expected in [
        (s.BeidouOrbitClass.GEO, block.BEIDOU2_GEO),
        (s.BeidouOrbitClass.IGSO, block.BEIDOU2_IGSO),
        (s.BeidouOrbitClass.MEO, block.BEIDOU2_MEO),
    ]:
        assert (
            block.from_constellation_block_type(s.GnssSystem.BEIDOU, "BDS-2", orbit)
            == expected
        )
    yaw_rate = 0.1211 * (math.pi / 180.0)
    assert s.gps_iia_default_yaw_rate_rad_s(1) == yaw_rate
    assert s.gps_iia_default_yaw_rate_rad_s(0) is None
    assert nominal == s.nominal_yaw([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0])
    assert axes == s.dyb_axes([1.0, 0.0, 0.0], [1.0, 1.0, 0.0])
    for kind, expected in [
        (block.GPS_IIA, law.gps_iia(yaw_rate, radians(0.5), radians(166.75))),
        (block.GPS_IIR, law.gps_iir(radians(0.2))),
        (block.GPS_IIF, law.gps_iif(radians(0.11), radians(-0.7), radians(166.75))),
        (block.GPS_III, law.galileo_iov(radians(15.0), radians(5.8))),
        (block.GLONASS_M, law.glonass_m(radians(0.25), radians(165.8))),
        (block.GALILEO_FOC, law.cosine_turn(s.GALILEO_FOC_TURN)),
        (block.BEIDOU2_MEO, law.beidou_orbit_normal_switch(radians(4.1), radians(3.9))),
        (
            block.beidou3_meo(s.BeidouManufacturer.SECM),
            law.fixed_beta_yaw_steering(radians(3.0)),
        ),
        (
            block.beidou3_meo(),
            law.beidou3_meo_unknown_manufacturer(s.BEIDOU3_CAST_MEO_TURN, radians(3.0)),
        ),
    ]:
        assert law.for_block(kind, 1) == expected
        assert expected != law.orbit_normal()


def test_gnss_attitude_law_fields():
    law = s.GnssAttitudeLaw
    yaw = law.for_block(s.GnssBlock.GPS_IIA, prn=5)
    assert yaw.variant == "gps_iia"
    assert yaw.yaw_rate_rad_s == math.radians(0.118)
    assert yaw.yaw_bias_rad == 0.5 * math.pi / 180.0
    assert yaw.shadow_limit_rad == 166.75 * math.pi / 180.0

    iir = law.gps_iir(0.01)
    assert (
        iir.variant,
        iir.yaw_rate_rad_s,
        iir.yaw_bias_rad,
        iir.shadow_limit_rad,
    ) == ("gps_iir", 0.01, None, None)
    iif = law.gps_iif(0.02, -0.03, 2.9)
    assert (
        iif.variant,
        iif.yaw_rate_rad_s,
        iif.yaw_bias_rad,
        iif.shadow_limit_rad,
    ) == ("gps_iif", 0.02, -0.03, 2.9)
    glonass = law.glonass_m(0.03, 2.8)
    assert (
        glonass.variant,
        glonass.yaw_rate_rad_s,
        glonass.yaw_bias_rad,
        glonass.shadow_limit_rad,
    ) == ("glonass_m", 0.03, None, 2.8)
    iov = law.galileo_iov(0.04, 0.05)
    assert (iov.variant, iov.beta_x_rad, iov.beta_y_rad) == ("galileo_iov", 0.04, 0.05)
    switch = law.beidou_orbit_normal_switch(0.06, 0.07)
    assert (switch.variant, switch.ys_to_on_beta_rad, switch.on_to_ys_beta_rad) == (
        "beidou_orbit_normal_switch",
        0.06,
        0.07,
    )
    fixed = law.fixed_beta_yaw_steering(0.08)
    assert (fixed.variant, fixed.beta0_rad) == ("fixed_beta_yaw_steering", 0.08)
    params = s.CosineTurnParameters(
        0.09, True, s.CosineTurnStartAngle.ORBIT_ANGLE, 0.1, 20.0
    )
    cosine = law.cosine_turn(params)
    assert (cosine.variant, cosine.cosine_turn_parameters, cosine.cast) == (
        "cosine_turn",
        params,
        None,
    )
    combined = law.beidou3_meo_unknown_manufacturer(params, 0.11)
    assert (combined.variant, combined.cast, combined.secm_beta0_rad) == (
        "beidou3_meo_unknown_manufacturer",
        params,
        0.11,
    )
    normal = law.orbit_normal()
    assert normal.variant == "orbit_normal"
    assert normal.yaw_rate_rad_s is normal.beta_x_rad is normal.cast is None


def test_legacy_drag_and_decay_value_errors_keep_details():
    source = s.load_space_weather(
        Path(CORE_FIXTURES) / "space_weather/SW-All-20260702-trim.csv"
    )
    drag = s.DragParameters.from_bc_factor_m2_kg(0.02)
    position, velocity = [6778.0, 0.0, 0.0], [0.0, 7.5, 0.5]
    with pytest.raises(ValueError) as caught:
        s.SourcedDragForce(drag, source).acceleration(float("nan"), position, velocity)
    assert caught.value.detail == {
        "kind": "invalid_input",
        "message": "epoch_tdb_seconds not finite",
    }
    for table in (None, source):
        reentry = [6400.0, 0.0, 0.0]
        estimate = s.estimate_decay(
            0.0, reentry, velocity, drag, space_weather_table=table
        )
        assert estimate.time_to_decay_s == estimate.epoch_tdb_seconds == 0.0
        assert estimate.reentry_position_km.tolist() == reentry
        assert estimate.reentry_velocity_km_s.tolist() == velocity
        with pytest.raises(ValueError) as caught:
            s.estimate_decay(
                0.0,
                position,
                velocity,
                drag,
                scan_step_s=0.0,
                space_weather_table=table,
            )
        assert caught.value.detail == {"kind": "invalid_config", "field": "scan_step_s"}


def test_explicit_orientation_refusal_and_sourced_decay():
    source = s.load_space_weather(
        Path(CORE_FIXTURES) / "space_weather/SW-All-20260702-trim.csv"
    )
    drag = s.DragParameters.from_bc_factor_m2_kg(0.02)
    initial = s.CartesianState(0.0, [6400.0, 0.0, 0.0], [0.0, 7.8, 0.0])
    result = s.estimate_decay_with_source_and_context(
        0.0,
        initial.position_km,
        initial.velocity_km_s,
        drag,
        source,
        s.PropagationContext(),
    )
    assert result.time_to_decay_s == 0.0
    assert result.epoch_tdb_seconds == 0.0
    assert result.reentry_position_km.tolist() == [6400.0, 0.0, 0.0]
    assert result.reentry_velocity_km_s.tolist() == [0.0, 7.8, 0.0]

    provider = s.Iers2010EarthOrientationProvider(
        [
            s.Iers2010EopSample(59024, 0.1, 0.2, -0.2, 0.001, 0.0, 0.0),
            s.Iers2010EopSample(59025, 0.1, 0.2, -0.2, 0.001, 0.0, 0.0),
        ]
    )
    context = s.PropagationContext().with_body_fixed_frame_provider(provider)
    for force in (drag.to_force(), s.SourcedDragForce(drag, source)):
        with pytest.raises(s.SolveError) as caught:
            force.acceleration_with_parameters(
                [parameter.value for parameter in force.parameters()], initial, context
            )
        assert caught.value.detail == {
            "kind": "ut1_outside_coverage",
            "reason": "before_coverage",
        }

    product = s.load_sp3(
        Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    )
    with pytest.raises(s.OrbitParameterFitError) as caught:
        s.fit_sp3_ecef_orbit_parameters_with_validity(
            product,
            "G01",
            provider,
            None,
            s.OrbitParameterFitOptions(),
            s.ValidityMode.PERMISSIVE,
        )
    assert caught.value.detail == {
        "kind": "ut1_validity_mismatch",
        "fit": s.ValidityMode.PERMISSIVE,
        "provider": "orientation",
        "provider_mode": s.ValidityMode.STRICT,
    }


def test_sp3_fit_with_explicit_earth_orientation(orbital):
    expected = orbital["terrestrial_parameter_fit"]
    product = s.load_sp3(
        Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    )
    provider = s.Iers2010EarthOrientationProvider(
        [
            s.Iers2010EopSample(day, 0.1, 0.2, -0.2, 0.001, 0.0, 0.0)
            for day in [61160, 61161]
        ]
    )
    options = s.OrbitParameterFitOptions()
    fit = s.fit_sp3_ecef_orbit_parameters(product, "G01", provider, None, options)
    assert fit.satellite == "G01"
    assert len(fit.residuals) == expected["observation_count"]
    exact(fit.initial_state.position_km, expected["position"])
    exact(fit.initial_state.velocity_km_s, expected["velocity"])
    exact(fit.condition_number, expected["condition"])
    exact(fit.residual_stats.rms_3d_m, expected["rms_3d_m"])
    permissive = provider.with_validity(s.ValidityMode.PERMISSIVE)
    fit_permissive, departure = s.fit_sp3_ecef_orbit_parameters_with_validity(
        product, "G01", permissive, None, options, s.ValidityMode.PERMISSIVE
    )
    exact(fit_permissive.initial_state.position_km, expected["position"])
    exact(fit_permissive.initial_state.velocity_km_s, expected["velocity"])
    assert departure is None


def test_gnss_attitude_trajectory_sources(orbital):
    fx = orbital["attitude_trajectories"]
    precise = s.load_sp3(
        Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    )
    iers = s.Iers2010EarthOrientationProvider(
        [
            s.Iers2010EopSample(day, 0.1, 0.2, -0.2, 0.001, 0.0, 0.0)
            for day in (61160, 61161)
        ]
    )
    nav = s.load_rinex_nav(Path(FIXTURES) / "nav/ESBC00DNK_R_20201770000_01D_MN.rnx")
    origin = s.ClockInstant.from_nanos(s.TimeScale.TDB, 0)
    nodes = s.OrbitTrajectory(
        "ITRF2020",
        origin,
        [
            s.TrajectorySample("G01", origin, [20e6, 10e6, 2e6]),
            s.TrajectorySample(
                "G01",
                s.ClockInstant.from_nanos(s.TimeScale.TDB, 43_200_000_000_000),
                [20e6, 10.02e6, 2e6],
            ),
        ],
    )
    analytic = s.BrouwerLyddanePropagator.from_mean_elements(
        0.0,
        s.AnalyticKeplerianElements(7000.0, 0.03, 0.7, 0.2, 0.3, 0.4),
        s.ZonalHarmonics.earth(),
        1e-17,
    )
    de = s.DeEphemeris.load(Path(CORE_FIXTURES) / "bodies/de440/lnxp1990.440")
    propagator = s.StatePropagator(
        s.CartesianState(0.0, [7000.0, 1000.0, 200.0], [-1.0, 7.5, 0.8]),
        s.ForceModelKind.two_body(398600.4418),
        s.Integrator.DP54,
        s.IntegratorOptions(dense_output=True),
    )
    result = propagator.propagate_to(300.0)
    final = result.final_state
    assert final.epoch_tdb_seconds == 300.0
    exact([final.position_km, final.velocity_km_s], fx["final_state"])
    assert isinstance(result, s.PropagationResult)
    assert isinstance(result.dense, s.DenseOutput)
    assert result.dense.coverage is not None
    exact(result.dense.coverage, fx["dense_coverage"])
    backward_dense = propagator.propagate_to(-300.0).dense
    assert backward_dense is not None
    assert backward_dense.coverage is not None
    exact(backward_dense.coverage, fx["backward_dense_coverage"])
    interpolated = result.dense.eval(123.0)
    assert interpolated.epoch_tdb_seconds == 123.0
    exact([interpolated.position_km, interpolated.velocity_km_s], fx["dense_state"])
    for endpoint in (0.0, 300.0):
        assert result.dense.eval(endpoint).epoch_tdb_seconds == endpoint
    with pytest.raises(s.SolveError) as caught:
        result.dense.eval(301.0)
    assert caught.value.detail == {
        "kind": "output_out_of_range",
        "t": 301.0,
        "t_min": 0.0,
        "t_max": 300.0,
    }
    with pytest.raises(s.SolveError) as caught:
        result.dense.eval(float("nan"))
    detail = caught.value.detail
    assert math.isnan(detail["t"])
    assert math.isnan(detail["theta"])
    assert detail == {
        "kind": "segment_out_of_range",
        "t": detail["t"],
        "t_start": 0.0,
        "t_end": hex_to_f64(fx["dense_first_step_end"]),
        "theta": detail["theta"],
    }
    empty_dense = propagator.propagate_to(0.0).dense
    assert empty_dense is not None
    assert empty_dense.coverage is None
    with pytest.raises(s.SolveError) as caught:
        empty_dense.eval(0.0)
    assert caught.value.detail == {"kind": "empty"}

    with_context = propagator.propagate_to(300.0, s.PropagationContext()).final_state
    exact([with_context.position_km, with_context.velocity_km_s], fx["final_state"])
    for integrator, dense_output in [
        (s.Integrator.RK4, True),
        (s.Integrator.DP54, False),
    ]:
        no_dense = s.StatePropagator(
            s.CartesianState(0.0, [7000.0, 1000.0, 200.0], [-1.0, 7.5, 0.8]),
            s.ForceModelKind.two_body(398600.4418),
            integrator,
            s.IntegratorOptions(dense_output=dense_output),
        )
        assert no_dense.propagate_to(300.0).dense is None

    trajectory = s.OrbitAttitudeTrajectory
    analytic_trajectory = trajectory.from_brouwer_lyddane(analytic)
    dense = trajectory.from_dense_output(result.dense)
    broadcast = s.BroadcastOrbit(nav, s.ClockInstant.from_nanos(s.TimeScale.GPST, 0))
    broadcast_epoch, position, clock = fx["broadcast_state"]
    state = broadcast.state_at_j2000_s("G01", hex_to_f64(broadcast_epoch))
    exact([state.position_ecef_m, state.clock_s], [position, clock])
    cases = {
        "sp3": trajectory.from_sp3(precise, "G01", iers),
        "predicted": trajectory.from_orbit_source(
            s.PredictedSp3.from_prediction_flags(precise), "G01", iers
        ),
        "broadcast": trajectory.from_broadcast(nav, "G01"),
        "broadcast_source": trajectory.from_orbit_source(broadcast, "G01"),
        "sampled_source": trajectory.from_orbit_source(
            nodes, "G01"
        ).with_velocity_half_step_s(43200.0),
        "analytic": analytic_trajectory,
        "analytic_de": analytic_trajectory.with_de_ephemeris(de),
        "dense": dense,
    }
    del precise, iers, nav, nodes, analytic, de, result, broadcast
    law = s.GnssAttitudeLaw.orbit_normal()
    for name, orbit in cases.items():
        epoch = hex_to_f64(fx[name]["epoch"])
        sample = orbit.sample(epoch)
        exact(
            [sample.position_m, sample.velocity_m_s, sample.sun_position_m],
            fx[name]["sample"],
        )
        attitude = s.gnss_attitude(law, orbit, epoch)
        assert attitude.mode == s.AttitudeMode.ORBIT_NORMAL
        exact(
            [
                attitude.yaw_rad,
                attitude.nominal_yaw_rad,
                attitude.beta_rad,
                attitude.orbit_angle_rad,
                attitude.body_x,
                attitude.body_y,
                attitude.body_z,
                attitude.dyb.d,
                attitude.dyb.y,
                attitude.dyb.b,
            ],
            fx[name]["attitude"],
        )

    assert s.gnss_attitude(law, dense, 123.0) == s.gnss_attitude(
        law, cases["dense"], 123.0
    )
    assert s.gnss_attitude(law, dense, 123.0) != s.gnss_attitude(
        law, cases["analytic"], 600.0
    )
    assert dense.sample(301.0) is None
    with pytest.raises(s.GnssAttitudeError) as caught:
        s.gnss_attitude(law, dense, 301.0)
    assert caught.value.detail == {"kind": "trajectory_unavailable", "t_s": 301.0}
    for step, reason in [
        (0.0, "not positive and finite"),
        (float("nan"), "not positive and finite"),
        (1e-9, "below the 1e-8 s resolution of SP3 epochs"),
    ]:
        with pytest.raises(s.OrbitScoreError) as caught:
            cases["sp3"].with_velocity_half_step_s(step)
        assert caught.value.detail == {
            "kind": "invalid_input",
            "field": "velocity_half_step_s",
            "reason": reason,
        }


def test_parameter_fit_refusals(orbital):
    fx = orbital["estimation"]["fit"]
    samples = [
        s.InertialPositionSample(
            s.ClockInstant.from_nanos(s.TimeScale.TDB, epoch * 10**9),
            [hex_to_f64(component) for component in position],
        )
        for epoch, position in zip(fx["epochs"], fx["observations"], strict=True)
    ]
    seed = s.CartesianState(
        hex_to_f64(fx["initial_epoch"]), [7000.0, 0.0, 0.0], [0.0, 7.5, 0.5]
    )
    first = samples[0].epoch
    next_epoch = samples[1].epoch
    no_effect = s.SolarRadiationPressure(1.0, 5e-324)
    radial = s.EmpiricalRtnAcceleration.with_zero_terms(
        [(s.RtnAxis.RADIAL, s.EmpiricalHarmonic.CONSTANT)]
    )
    cases = [
        (
            samples,
            None,
            {"estimated_parameters": ["bad"]},
            {
                "kind": "invalid_parameter",
                "name": "bad",
                "reason": "is not a parameter of the force models",
            },
        ),
        (
            samples,
            None,
            {"max_iterations": 0},
            {
                "kind": "invalid_option",
                "field": "max_iterations",
                "reason": "not positive",
            },
        ),
        (
            samples[:1] * 2,
            None,
            {},
            {"kind": "non_monotonic_epochs", "satellite": "G01"},
        ),
        (
            samples,
            s.CartesianState(1.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.5]),
            {},
            {
                "kind": "invalid_epoch",
                "satellite": "G01",
                "reason": "initial-state seed epoch must match the first sample",
            },
        ),
        (
            [s.InertialPositionSample(first, [float("nan"), 0.0, 0.0]), *samples[1:]],
            None,
            {},
            {
                "kind": "invalid_observation",
                "satellite": "G01",
                "reason": "position components must be finite",
            },
        ),
        (
            samples,
            None,
            {"integrator_options": s.IntegratorOptions(max_steps=1)},
            {
                "kind": "propagation",
                "satellite": "G01",
                "source": {"kind": "max_steps_exceeded"},
            },
        ),
        (
            samples,
            None,
            {"observation_sigma_m": 1e-320, "max_iterations": 1},
            {
                "kind": "did_not_converge",
                "satellite": "G01",
                "iterations": 1,
                "last_step_sigmas": float("inf"),
            },
        ),
        (
            samples,
            None,
            {"additional_forces": [no_effect], "estimated_parameters": ["srp.cr"]},
            {"kind": "unobservable_parameter", "satellite": "G01", "name": "srp.cr"},
        ),
        (
            samples[:2],
            seed,
            {
                "additional_forces": [radial],
                "estimated_parameters": ["empirical.radial.constant"],
            },
            {
                "kind": "singular_normal_equations",
                "satellite": "G01",
                "condition_number": float("inf"),
                "weakest": [],
            },
        ),
        (
            [samples[0], s.InertialPositionSample(next_epoch, [1.7e308] * 3)],
            seed,
            {},
            {
                "kind": "numerical_failure",
                "satellite": "G01",
                "quantity": "sum of squared residuals",
            },
        ),
    ]
    for observations, initial_state, kwargs, detail in cases:
        options = s.OrbitParameterFitOptions(
            force_model=s.ForceModelKind.two_body(), **kwargs
        )
        with pytest.raises(s.OrbitParameterFitError) as caught:
            s.fit_inertial_orbit_parameters(observations, "G01", initial_state, options)
        assert caught.value.detail == detail
    with pytest.raises(s.OrbitParameterFitError) as caught:
        s.fit_inertial_orbit_parameters(
            samples,
            "G01",
            None,
            s.OrbitParameterFitOptions(
                force_model=s.ForceModelKind.two_body(),
                max_condition_number=1.5,
            ),
        )
    expected = orbital["estimation"]["conditioning_refusal"]
    assert caught.value.detail == {
        **expected,
        "condition_number": hex_to_f64(expected["condition_number"]),
        "limit": hex_to_f64(expected["limit"]),
        "weakest": [(name, hex_to_f64(value)) for name, value in expected["weakest"]],
    }


def test_strict_parameter_fit_refuses_drag_ut1_departure():
    start = 900_000_000
    epochs = list(range(start, start + 241, 60))
    initial = s.CartesianState(start, [6778.0, 0.0, 0.0], [0.0, 7.5, 0.5])
    drag = s.DragParameters.from_bc_factor_m2_kg(0.02)
    gravity = s.ForceModelKind.two_body()
    propagator = s.StatePropagator(initial, gravity, s.Integrator.DP54, drag=drag)
    states = propagator.ephemeris_with_context(epochs, s.PropagationContext())
    samples = [
        s.InertialPositionSample(
            s.ClockInstant.from_nanos(s.TimeScale.TDB, epoch * 10**9),
            state.position_km * 1000.0,
        )
        for epoch, state in zip(epochs, states, strict=True)
    ]
    with pytest.raises(s.OrbitParameterFitError) as caught:
        s.fit_inertial_orbit_parameters(
            samples,
            "G01",
            initial,
            s.OrbitParameterFitOptions(force_model=gravity, drag=drag),
        )
    assert caught.value.detail == {
        "kind": "ut1_outside_coverage",
        "reason": "after_coverage",
    }


def test_parameter_fit_rtn_and_orientation_refusals():
    position = [7000.0, 0.0, 0.0]
    samples = [
        s.InertialPositionSample(
            s.ClockInstant.from_nanos(s.TimeScale.TDB, t * 10**9), [7e6, 0.0, 0.0]
        )
        for t in (0, 120, 240)
    ]
    seed = s.CartesianState(2.0**-36, position, [0.0, 0.0, 0.0])
    with pytest.raises(s.OrbitParameterFitError) as caught:
        s.fit_inertial_orbit_parameters(
            samples,
            "G01",
            seed,
            s.OrbitParameterFitOptions(
                force_model=s.ForceModelKind.two_body(5e-324),
            ),
        )
    assert caught.value.detail == {
        "kind": "rtn_frame",
        "satellite": "G01",
        "reason": {"kind": "parallel_position_velocity"},
    }
    near_origin = 2.0**-110
    tiny_samples = [
        s.InertialPositionSample(sample.epoch, [near_origin * 1000.0, 0.0, 0.0])
        for sample in samples
    ]
    tiny_seed = s.CartesianState(2.0**-36, [near_origin, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(s.OrbitParameterFitError) as caught:
        s.fit_inertial_orbit_parameters(
            tiny_samples,
            "G01",
            tiny_seed,
            s.OrbitParameterFitOptions(
                force_model=s.ForceModelKind.two_body(0.0),
            ),
        )
    assert caught.value.detail == {
        "kind": "rtn_frame",
        "satellite": "G01",
        "reason": {"kind": "zero_position"},
    }

    provider = s.Iers2010EarthOrientationProvider(
        [
            s.Iers2010EopSample(day, 0.1, 0.2, -0.2, 0.001, 0.0, 0.0)
            for day in (61161, 61162)
        ]
    )
    precise = s.load_sp3(
        Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    )
    with pytest.raises(s.OrbitParameterFitError) as caught:
        s.fit_sp3_ecef_orbit_parameters(
            precise, "G01", provider, None, s.OrbitParameterFitOptions()
        )
    assert caught.value.detail == {
        "kind": "frame",
        "satellite": "G01",
        "source": {
            "kind": "ut1_outside_coverage",
            "reason": "before_coverage",
        },
    }


def test_geopotential_zonal_and_sourced_drag_partials(orbital):
    fx = orbital["radiation"]
    weather = s.load_space_weather(
        Path(CORE_FIXTURES) / "space_weather/SW-All-20260702-trim.csv"
    )
    sourced = s.SourcedDragForce(s.DragParameters.from_bc_factor_m2_kg(0.02), weather)
    exact(
        sourced.acceleration(120614400.0, POSITION, VELOCITY),
        fx["sourced_drag"]["acceleration"],
    )
    field = s.IcgemGravityField.parse(orbital["terrestrial"]["field"])
    for name, model, epoch in [
        ("sourced_drag", sourced, 120614400.0),
        ("geopotential", field.gravity(2, 2), 0.0),
        ("zonal", s.ForceModel.TWO_BODY_J2, 0.0),
    ]:
        expected = fx[name]
        state = s.CartesianState(epoch, POSITION, VELOCITY)
        assert model.parameter_count() == len(expected["parameters"])
        exact(
            [[value.name, value.value] for value in model.parameters()],
            expected["parameters"],
        )
        parameters = [hex_to_f64(value) for _, value in expected["parameters"]]
        exact(
            model.acceleration_with_parameters(parameters, state),
            expected["acceleration_with_parameters"],
        )
        check_partials(
            model.acceleration_partials(parameters, state), expected["partials"]
        )
        exact(
            model.parameter_partials(parameters, state), expected["parameter_partials"]
        )
