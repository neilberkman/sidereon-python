use std::path::Path;
use std::sync::Arc;

use serde_json::{json, Value};
use sidereon_core::astro::attitude::gnss::{
    self, AttitudeTrajectory, GnssAttitudeLaw, OrbitAttitudeTrajectory,
};
use sidereon_core::astro::bodies::sun_moon::{
    sun_moon_gcrs, sun_moon_gcrs_at, sun_moon_gcrs_at_tdb_seconds,
};
use sidereon_core::astro::conjunction::PcMethod;
use sidereon_core::astro::covariance::{Covariance6, Mat6};
use sidereon_core::astro::de::{DeBody, DeEphemeris};
use sidereon_core::astro::events::eclipse::{self, EarthShadowModel};
use sidereon_core::astro::forces::{
    ecom_geometry, ecom_geometry_with_shadow, AccelerationPartials, CompositeForceModel, DragForce,
    DragParameters, EarthRadiationPressure, Ecom1Coefficients, Ecom1SolarRadiationPressure,
    Ecom2D2b1Coefficients, Ecom2SolarRadiationPressure, EmpiricalHarmonic,
    EmpiricalRtnAcceleration, ForceModel, IcgemGravityField, J2Gravity, RtnAxis,
    SolarRadiationPressure, SpaceWeather, SpaceWeatherSource, TwoBodyGravity,
};
use sidereon_core::astro::frames::transforms::{
    gcrs_to_mean_of_date_matrix, mean_of_date_to_gcrs_matrix,
};
use sidereon_core::astro::frames::{
    EarthOrientationProvider, Iers2010EarthOrientationProvider, Iers2010EopSample,
    TdbEarthOrientationProvider,
};
use sidereon_core::astro::propagator::analytic::*;
use sidereon_core::astro::propagator::{
    propagate_variational, ForceModelKind, IntegratorKind, IntegratorOptions, PropagationContext,
    StatePropagator, VariationalEpoch, VariationalEquations,
};
use sidereon_core::astro::propagator::{
    CovarianceFrame, CovariancePropagationOptions, LabeledCovariance6,
};
use sidereon_core::astro::sgp4::JulianDate;
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::tca::{
    find_tca_conjunctions_with_propagated_covariance_from_tles, TcaFinderOptions,
    TcaPropagatedCovariancePcOptions, TcaTle,
};
use sidereon_core::astro::time::scales::{
    tdb_from_tt_j2000_seconds, tdb_minus_tt_seconds, tt_from_tdb_j2000_seconds, TimeScales,
};
use sidereon_core::astro::time::{j2000_seconds, timescale_offset_s, Instant, TimeScale};
use sidereon_core::ephemeris::{BroadcastEphemeris, Sp3};
use sidereon_core::orbit_determination::{
    fit_inertial_orbit_parameters, fit_sp3_ecef_orbit_parameters, InertialPositionSample,
    OrbitParameterCovariance, OrbitParameterFitError, OrbitParameterFitOptions,
};
use sidereon_core::orbit_scoring::{
    score_orbit, BroadcastOrbit, ClockDatum, GapPolicy, OrbitScoreOptions, OrbitSource,
    OrbitTrajectory, PredictedSp3, ScoreEpochs, TrajectorySample, TriadVelocity,
};

use crate::{hex, hexes};

fn rows(matrix: [[f64; 3]; 3]) -> Vec<Value> {
    matrix.iter().map(|row| hexes(row)).collect()
}

fn partials(value: AccelerationPartials) -> Value {
    let matrix = |get: &dyn Fn(usize, usize) -> f64| {
        rows(std::array::from_fn(|row| {
            std::array::from_fn(|col| get(row, col))
        }))
    };
    json!({
        "position": matrix(&|r, c| value.wrt_position[(r, c)]),
        "velocity": matrix(&|r, c| value.wrt_velocity[(r, c)]),
    })
}

fn celestial(core: &Path) -> Value {
    let ts = TimeScales::from_utc(2020, 6, 24, 12, 0, 0.0).unwrap();
    let bodies = [
        sun_moon_gcrs(0.0).unwrap(),
        sun_moon_gcrs_at(&ts).unwrap(),
        sun_moon_gcrs_at_tdb_seconds(0.0).unwrap(),
    ];
    let de = DeEphemeris::load(core.join("bodies/de440/lnxp1990.440")).unwrap();
    let (start, end) = de.coverage_tdb_seconds();
    let epoch = start + 500.0;
    let state = de.sun_moon_at_tdb_seconds(epoch).unwrap();
    let pos = de.sun_moon_positions_at_tdb_seconds(epoch).unwrap();
    json!({
        "sun_moon": bodies.iter().map(|b| [hexes(&b.sun), hexes(&b.moon)]).collect::<Vec<_>>(),
        "gcrs_to_mean": rows(gcrs_to_mean_of_date_matrix(&ts).unwrap()),
        "mean_to_gcrs": rows(mean_of_date_to_gcrs_matrix(&ts).unwrap()),
        "time": [hex(tdb_minus_tt_seconds(0.0)), hex(tdb_from_tt_j2000_seconds(0.0)), hex(tt_from_tdb_j2000_seconds(0.0))],
        "de": {
            "coverage": hexes(&[start, end]),
            "epoch": hex(epoch),
            "version": de.version(),
            "record_size": de.record_size(),
            "constants": de.constants().iter().map(|(key, value)| (key, hex(*value))).collect::<std::collections::BTreeMap<_, _>>(),
            "astronomical_unit_m": hex(de.astronomical_unit_m()),
            "earth_moon_mass_ratio": hex(de.earth_moon_mass_ratio()),
            "gm": ([DeBody::Sun, DeBody::Mercury, DeBody::Venus, DeBody::EarthMoon, DeBody::Earth, DeBody::Moon, DeBody::Mars, DeBody::Jupiter, DeBody::Saturn, DeBody::Uranus, DeBody::Neptune, DeBody::Pluto, DeBody::SolarSystem]).map(|body| de.gm_m3_s2(body).map(hex)),
            "sun_state": [hexes(&state.sun.position_m), hexes(&state.sun.velocity_m_s)],
            "moon_state": [hexes(&state.moon.position_m), hexes(&state.moon.velocity_m_s)],
            "positions": [hexes(&pos.sun), hexes(&pos.moon)],
        },
    })
}

fn radiation(core: &Path) -> Value {
    let position = [7000.0, 1000.0, 200.0];
    let velocity = [-1.0, 7.5, 0.8];
    let sun = [130_000_000.0, 10_000_000.0, 5_000_000.0];
    let moon = [300_000.0, -100_000.0, 50_000.0];
    let state = CartesianState::new(0.0, position, velocity);
    let ctx = PropagationContext::default();
    let ecom1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: 2e-9,
        ..Default::default()
    })
    .unwrap();
    let ecom2 = Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients {
        d0: 2e-9,
        ..Default::default()
    })
    .unwrap();
    let empirical = EmpiricalRtnAcceleration::new()
        .with_term(RtnAxis::Radial, EmpiricalHarmonic::Constant, 1e-8)
        .unwrap();
    let srp = SolarRadiationPressure::new(1.5, 0.02).unwrap();
    let erp = EarthRadiationPressure::new(1.5, 0.02).unwrap();
    let model_goldens_at = |model: &dyn ForceModel, state: &CartesianState| {
        let parameters: Vec<_> = model.parameters().iter().map(|p| p.value).collect();
        json!({
            "parameters": model.parameters().iter().map(|p| json!([p.name, hex(p.value)])).collect::<Vec<_>>(),
            "acceleration": hexes(model.acceleration(state, &ctx).unwrap().as_slice()),
            "acceleration_with_parameters": hexes(model.acceleration_with_parameters(&parameters, state, &ctx).unwrap().as_slice()),
            "partials": partials(model.acceleration_partials(&parameters, state, &ctx).unwrap()),
            "parameter_partials": model.parameter_partials(&parameters, state, &ctx).unwrap().iter().map(|v| hexes(v.as_slice())).collect::<Vec<_>>(),
        })
    };
    let model_goldens = |model: &dyn ForceModel| model_goldens_at(model, &state);
    let drag = DragParameters::from_bc_factor_m2_kg(
        0.02,
        SpaceWeather::default(),
        DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
    )
    .unwrap();
    let mut composite = CompositeForceModel::new();
    composite.add(Box::new(TwoBodyGravity::default()));
    composite.add(Box::new(empirical));
    let zonal = ForceModelKind::two_body_j2().build().unwrap();
    let field = IcgemGravityField::parse(STATIC_FIELD)
        .unwrap()
        .gravity(2, 2)
        .unwrap();
    let weather = sidereon_core::astro::space_weather::parse(
        &std::fs::read(core.join("space_weather/SW-All-20260702-trim.csv")).unwrap(),
    )
    .unwrap();
    let sourced = sidereon_core::astro::forces::SourcedDragForce::new(
        drag,
        sidereon_core::astro::forces::SpaceWeatherSource::Table(std::sync::Arc::new(weather.value)),
    );
    let sourced_state = CartesianState::new(120614400.0, position, velocity);
    let p = position.into();
    let v = velocity.into();
    let s = sun.into();
    let m = moon.into();
    let model = EarthShadowModel::Wgs84Oblate;
    let a1 = ecom1.with_moon_eclipse(false);
    let a2 = ecom2.clone().with_moon_eclipse(false);
    let geometry = |g: sidereon_core::astro::forces::EcomGeometry| {
        json!([
            hexes(&g.dyb.d),
            hexes(&g.dyb.y),
            hexes(&g.dyb.b),
            g.delta_u_rad.map(hex),
            g.argument_of_latitude_rad.map(hex),
            hex(g.lighting_ratio)
        ])
    };
    let standalone_state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [0.0; 3]);
    let standalone_acceleration = |model: &dyn ForceModel| {
        hexes(
            model
                .acceleration(&standalone_state, &ctx)
                .unwrap()
                .as_slice(),
        )
    };
    json!({
        "ecom1": model_goldens(&ecom1),
        "ecom2": model_goldens(&ecom2),
        "empirical": model_goldens(&empirical),
        "srp": model_goldens(&srp),
        "erp": model_goldens(&erp),
        "drag": model_goldens(&drag.to_force()),
        "sourced_drag": model_goldens_at(&sourced, &sourced_state),
        "geopotential": model_goldens(&field),
        "zonal": model_goldens(zonal.as_ref()),
        "composite": model_goldens(&composite),
        "gravity": model_goldens(&TwoBodyGravity::default()),
        "standalone_gravity": {
            "two_body": standalone_acceleration(&TwoBodyGravity::default()),
            "j2": standalone_acceleration(&J2Gravity::default()),
        },
        "geometry": geometry(ecom_geometry(p, v, s).unwrap()),
        "geometry_with_shadow": geometry(ecom_geometry_with_shadow(p, v, s, Some(m), model).unwrap()),
        "ecom1_sun": [
            hexes(a1.acceleration_with_sun(p, v, s).unwrap().as_slice()),
            partials(a1.acceleration_partials_with_sun(p, v, s).unwrap()),
            json!(a1.parameter_partials_with_sun(p, v, s).unwrap().iter().map(|value| hexes(value.as_slice())).collect::<Vec<_>>()),
        ],
        "ecom1_sun_moon": [
            hexes(ecom1.acceleration_with_sun_moon(p, v, s, m).unwrap().as_slice()),
            partials(ecom1.acceleration_partials_with_sun_moon(p, v, s, m).unwrap()),
            json!(ecom1.parameter_partials_with_sun_moon(p, v, s, m).unwrap().iter().map(|value| hexes(value.as_slice())).collect::<Vec<_>>()),
        ],
        "ecom2_sun": [
            hexes(a2.acceleration_with_sun(p, v, s).unwrap().as_slice()),
            partials(a2.acceleration_partials_with_sun(p, v, s).unwrap()),
            json!(a2.parameter_partials_with_sun(p, v, s).unwrap().iter().map(|value| hexes(value.as_slice())).collect::<Vec<_>>()),
        ],
        "ecom2_sun_moon": [
            hexes(ecom2.acceleration_with_sun_moon(p, v, s, m).unwrap().as_slice()),
            partials(ecom2.acceleration_partials_with_sun_moon(p, v, s, m).unwrap()),
            json!(ecom2.parameter_partials_with_sun_moon(p, v, s, m).unwrap().iter().map(|value| hexes(value.as_slice())).collect::<Vec<_>>()),
        ],
        "srp_sun_moon": hexes(srp.acceleration_with_sun_moon(p, s, m).unwrap().as_slice()),
        "erp_sun_moon": hexes(erp.with_moon_eclipse(true).acceleration_with_sun_moon(&state, s, m).unwrap().as_slice()),
    })
}

fn analytic() -> Value {
    let gravity = ZonalHarmonics::earth();
    let circular = CircularElements {
        a_km: 7000.0,
        ex: 0.001,
        ey: 0.002,
        i_rad: 0.7,
        raan_rad: 0.3,
        mean_latitude_argument_rad: 0.4,
    };
    let keplerian = KeplerianElements {
        a_km: 7000.0,
        e: 0.03,
        i_rad: 0.7,
        arg_perigee_rad: 0.2,
        raan_rad: 0.3,
        mean_anomaly_rad: 0.4,
    };
    let eh = EcksteinHechlerPropagator::from_mean_elements(0.0, circular, &gravity).unwrap();
    let bl = BrouwerLyddanePropagator::from_mean_elements(0.0, keplerian, &gravity, 1e-17).unwrap();
    let pv = |state: CartesianState| {
        [
            hexes(&state.position_array()),
            hexes(&state.velocity_array()),
        ]
    };
    let cf = |c: CircularElements| {
        hexes(&[
            c.a_km,
            c.ex,
            c.ey,
            c.i_rad,
            c.raan_rad,
            c.mean_latitude_argument_rad,
        ])
    };
    let kf = |k: KeplerianElements| {
        hexes(&[
            k.a_km,
            k.e,
            k.i_rad,
            k.arg_perigee_rad,
            k.raan_rad,
            k.mean_anomaly_rad,
        ])
    };
    let e0 = eh.propagate(0.0).unwrap();
    let b0 = bl.propagate(0.0).unwrap();
    let eh_mean = eckstein_hechler_mean_elements(
        &e0,
        &gravity,
        MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT,
    )
    .unwrap();
    let bl_mean = brouwer_lyddane_mean_elements(
        &b0,
        &gravity,
        MeanConversionOptions::BROUWER_LYDDANE_DEFAULT,
    )
    .unwrap();
    let e_apparent = eckstein_hechler_osculating_elements(&circular, &gravity).unwrap();
    let b_apparent = brouwer_lyddane_osculating_elements(&keplerian, &gravity).unwrap();
    let attitude_trajectory = OrbitAttitudeTrajectory::from_eckstein_hechler(&eh);
    let attitude =
        gnss::gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &attitude_trajectory, 600.0).unwrap();
    let turn_orbit = EcksteinHechlerPropagator::from_mean_elements(
        0.0,
        CircularElements {
            a_km: 26560.0,
            ex: 0.0,
            ey: 0.0,
            i_rad: 0.409,
            raan_rad: 0.0,
            mean_latitude_argument_rad: 4.9,
        },
        &gravity,
    )
    .unwrap();
    let turn_trajectory = OrbitAttitudeTrajectory::from_eckstein_hechler(&turn_orbit);
    let turn_law = GnssAttitudeLaw::for_block(gnss::GnssBlock::GpsIIR, None).unwrap();
    let turn = gnss::gnss_attitude(&turn_law, &turn_trajectory, 0.0).unwrap();
    assert_eq!(turn.mode, gnss::AttitudeMode::NoonTurn);
    let (circular_position, circular_velocity) = circular.to_cartesian(gravity.mu_km3_s2).unwrap();
    let (keplerian_position, keplerian_velocity) =
        keplerian.to_cartesian(gravity.mu_km3_s2).unwrap();
    let nominal = gnss::nominal_yaw([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]).unwrap();
    json!({
        "gravity": hexes(&[gravity.mu_km3_s2, gravity.re_km, gravity.c20, gravity.c30, gravity.c40, gravity.c50, gravity.c60]),
        "eccentricity": hex(circular.eccentricity()),
        "circular_cartesian": [hexes(&circular_position), hexes(&circular_velocity)],
        "keplerian_cartesian": [hexes(&keplerian_position), hexes(&keplerian_velocity)],
        "nominal": [hex(nominal.yaw_rad), hex(nominal.beta_rad), hexes(&nominal.body_x), hexes(&nominal.body_y), hexes(&nominal.body_z)],
        "eckstein_hechler": [
            pv(e0), pv(eh.propagate(600.0).unwrap()), cf(eh.osculating_elements(600.0).unwrap()),
            cf(e_apparent), json!(eh_mean.iterations), cf(eh_mean.elements),
            cf(CircularElements::from_cartesian(e0.position_array(), e0.velocity_array(), gravity.mu_km3_s2).unwrap()),
        ],
        "brouwer_lyddane": [
            pv(b0), pv(bl.propagate(600.0).unwrap()), kf(bl.osculating_elements(600.0).unwrap()),
            kf(b_apparent), json!(bl_mean.iterations), kf(bl_mean.elements),
            kf(KeplerianElements::from_cartesian(b0.position_array(), b0.velocity_array(), gravity.mu_km3_s2).unwrap()),
        ],
        "turn": [hex(turn.yaw_rad), hexes(&turn.body_x), hexes(&turn.body_y), hexes(&turn.body_z)],
        "attitude": [hex(attitude.yaw_rad), hex(attitude.nominal_yaw_rad), hex(attitude.beta_rad), hex(attitude.orbit_angle_rad),
            hexes(&attitude.body_x), hexes(&attitude.body_y), hexes(&attitude.body_z)],
    })
}

fn attitude_case(trajectory: OrbitAttitudeTrajectory<'_>, epoch: f64) -> Value {
    let sample = trajectory
        .sample(epoch)
        .unwrap_or_else(|| panic!("trajectory unavailable at {epoch}"));
    let attitude = gnss::gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &trajectory, epoch).unwrap();
    let dyb = attitude.dyb.unwrap();
    json!({
        "epoch": hex(epoch),
        "sample": [hexes(&sample.position_m), hexes(&sample.velocity_m_s), hexes(&sample.sun_position_m)],
        "attitude": [hex(attitude.yaw_rad), hex(attitude.nominal_yaw_rad),
            hex(attitude.beta_rad), hex(attitude.orbit_angle_rad), hexes(&attitude.body_x),
            hexes(&attitude.body_y), hexes(&attitude.body_z), hexes(&dyb.d), hexes(&dyb.y), hexes(&dyb.b)],
    })
}

fn attitude_trajectories(core: &Path, fixtures: &Path) -> Value {
    let satellite = "G01".parse().unwrap();
    let product = Sp3::parse(
        &std::fs::read(fixtures.join("sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3")).unwrap(),
    )
    .unwrap();
    let iers = Iers2010EarthOrientationProvider::new(
        [61160, 61161]
            .map(|mjd_utc| Iers2010EopSample {
                mjd_utc,
                xp_arcsec: 0.1,
                yp_arcsec: 0.2,
                ut1_minus_utc_s: -0.2,
                lod_s: 0.001,
                dx_arcsec: 0.0,
                dy_arcsec: 0.0,
                pole_rates_rad_s: None,
            })
            .to_vec(),
    )
    .unwrap();
    let gpst_to_tdb = |gpst| {
        tdb_from_tt_j2000_seconds(
            gpst + timescale_offset_s(TimeScale::Gpst, TimeScale::Tt).unwrap(),
        )
    };
    let precise_epoch = gpst_to_tdb(product.epochs_j2000_seconds()[0] + 3456.0);
    let predicted = PredictedSp3::from_prediction_flags(&product).unwrap();

    let nav = BroadcastEphemeris::from_nav(
        &std::fs::read_to_string(fixtures.join("nav/ESBC00DNK_R_20201770000_01D_MN.rnx")).unwrap(),
    )
    .unwrap();
    let broadcast_gpst = j2000_seconds(2020, 6, 25, 4, 15, 0.0);
    let broadcast_epoch = gpst_to_tdb(broadcast_gpst);
    let default = TdbEarthOrientationProvider::new();
    let broadcast = BroadcastOrbit::new(&nav, Instant::from_nanos(TimeScale::Gpst, 0)).unwrap();

    let broadcast_state = broadcast
        .state_at_j2000_s(satellite, broadcast_gpst)
        .unwrap();

    let origin = Instant::from_nanos(TimeScale::Tdb, 0);
    let nodes = OrbitTrajectory::new(
        "ITRF2020",
        origin,
        [
            TrajectorySample {
                satellite,
                epoch: origin,
                position_ecef_m: [20e6, 10e6, 2e6],
                clock_s: None,
            },
            TrajectorySample {
                satellite,
                epoch: Instant::from_nanos(TimeScale::Tdb, 43_200_000_000_000),
                position_ecef_m: [20e6, 10.02e6, 2e6],
                clock_s: None,
            },
        ],
    )
    .unwrap();
    let sampled = OrbitAttitudeTrajectory::from_orbit_source(&nodes, satellite, &default)
        .with_velocity_half_step_s(43200.0)
        .unwrap();

    let gravity = ZonalHarmonics::earth();
    let analytic = BrouwerLyddanePropagator::from_mean_elements(
        0.0,
        KeplerianElements {
            a_km: 7000.0,
            e: 0.03,
            i_rad: 0.7,
            arg_perigee_rad: 0.2,
            raan_rad: 0.3,
            mean_anomaly_rad: 0.4,
        },
        &gravity,
        1e-17,
    )
    .unwrap();
    let de = DeEphemeris::load(core.join("bodies/de440/lnxp1990.440")).unwrap();

    let de_epoch = de.coverage_tdb_seconds().0 + 500.0;
    let mut options = IntegratorOptions::default();
    options.dense_output = true;
    let propagator = StatePropagator::new(
        0.0,
        [7000.0, 1000.0, 200.0],
        [-1.0, 7.5, 0.8],
        ForceModelKind::TwoBody {
            mu_km3_s2: 398600.4418,
        },
        IntegratorKind::Dp54,
    )
    .with_options(options);
    let result = propagator.propagate_to(300.0).unwrap();
    let backward = propagator.propagate_to(-300.0).unwrap();
    let dense = result.dense.as_ref().unwrap();
    let (start, end) = dense.coverage_tdb_seconds().unwrap();
    let (backward_start, backward_end) = backward.dense.unwrap().coverage_tdb_seconds().unwrap();
    let interpolated = dense.eval(123.0).unwrap();
    let segment = &dense.segments[0];
    json!({
        "sp3": attitude_case(OrbitAttitudeTrajectory::from_sp3(&product, satellite, &iers), precise_epoch),
        "predicted": attitude_case(OrbitAttitudeTrajectory::from_orbit_source(&predicted, satellite, &iers), precise_epoch),
        "broadcast": attitude_case(OrbitAttitudeTrajectory::from_broadcast(&nav, satellite, &default), broadcast_epoch),
        "broadcast_state": [hex(broadcast_gpst), hexes(&broadcast_state.position_ecef_m), json!(broadcast_state.clock_s.map(hex))],
        "broadcast_source": attitude_case(OrbitAttitudeTrajectory::from_orbit_source(&broadcast, satellite, &default), broadcast_epoch),
        "sampled_source": attitude_case(sampled, 0.0),
        "analytic": attitude_case(OrbitAttitudeTrajectory::from_brouwer_lyddane(&analytic), 600.0),
        "analytic_de": attitude_case(OrbitAttitudeTrajectory::from_brouwer_lyddane(&analytic).with_de_ephemeris(&de), de_epoch),
        "dense": attitude_case(OrbitAttitudeTrajectory::from_dense_output(dense), 123.0),
        "dense_state": [hexes(&interpolated.position_array()), hexes(&interpolated.velocity_array())],
        "dense_coverage": hexes(&[start, end]),
        "backward_dense_coverage": hexes(&[backward_start, backward_end]),
        "dense_first_step_end": hex(segment.t_end()),
        "final_state": [hexes(&result.final_state.position_array()), hexes(&result.final_state.velocity_array())],
    })
}

fn broadcast_scores(core: &Path, fixtures: &Path) -> Value {
    let reference = Sp3::parse(
        &std::fs::read(core.join("sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3")).unwrap(),
    )
    .unwrap();
    let nav = BroadcastEphemeris::from_nav(
        &std::fs::read_to_string(fixtures.join("nav/ESBC00DNK_R_20201770000_01D_MN.rnx")).unwrap(),
    )
    .unwrap();
    let first_epoch = j2000_seconds(2020, 6, 25, 4, 15, 0.0) as i128;
    let epochs = [first_epoch, first_epoch + 300, first_epoch + 600];
    let instant = |seconds| Instant::from_nanos(TimeScale::Gpst, seconds * 1_000_000_000);
    let source = BroadcastOrbit::new(&nav, instant(first_epoch)).unwrap();
    let mut options = OrbitScoreOptions::new(vec![-1.0, 600.0]);
    options.epochs = ScoreEpochs::Instants(epochs.map(instant).to_vec());
    options.clock_datum = ClockDatum::Unadjusted;
    options.triad = TriadVelocity::EarthFixed;
    options.gaps = GapPolicy::Refuse;
    let report = score_orbit(&source, &reference, &["G01".parse().unwrap()], &options).unwrap();
    let bin = &report.per_satellite[0].bins[0];
    let stats = [
        bin.radial,
        bin.transverse,
        bin.normal,
        bin.orbit_3d,
        bin.sisre_orbit,
        bin.clock,
        bin.sisre,
    ]
    .map(|field| {
        let field = field.unwrap();
        hexes(&[field.mean_m, field.rms_m, field.max_abs_m])
    });
    json!({
        "epochs": epochs,
        "count": bin.count,
        "clock_count": bin.clock_count,
        "statistics": stats,
    })
}

fn additional_force_propagation(core: &Path) -> Value {
    let weather = sidereon_core::astro::space_weather::parse(
        &std::fs::read(core.join("space_weather/SW-All-20260702-trim.csv")).unwrap(),
    )
    .unwrap();
    let acceleration = |axis, coefficient| -> Arc<dyn ForceModel> {
        Arc::new(
            EmpiricalRtnAcceleration::new()
                .with_term(axis, EmpiricalHarmonic::Constant, coefficient)
                .unwrap(),
        )
    };
    let additional = [
        acceleration(RtnAxis::Radial, 1e-8),
        acceleration(RtnAxis::AlongTrack, 2e-8),
    ];
    let epoch = 120_614_400.0;
    let targets = [epoch + 120.0, epoch + 240.0];
    let mut options = IntegratorOptions::default();
    options.dense_output = true;
    let base = || {
        StatePropagator::new(
            epoch,
            [6778.0, 0.0, 0.0],
            [0.0, 7.5, 0.5],
            ForceModelKind::two_body_j2(),
            IntegratorKind::Dp54,
        )
        .with_options(options)
    };
    let drag = DragParameters::from_bc_factor_m2_kg(
        0.02,
        SpaceWeather::default(),
        DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
    )
    .unwrap();
    let fixed = base().with_drag(drag);
    let sourced = base()
        .with_drag(drag)
        .with_space_weather(SpaceWeatherSource::Table(Arc::new(weather.value)));
    let state = |value: CartesianState| {
        [
            hexes(&value.position_array()),
            hexes(&value.velocity_array()),
        ]
    };
    let context = PropagationContext::default();
    let evaluate = |propagator: StatePropagator| {
        let states = propagator
            .ephemeris_with_additional_forces(&targets, &additional, &context)
            .unwrap();
        let result = propagator
            .propagate_to_with_additional_forces(targets[1], &additional, &context)
            .unwrap();
        let dense = result.dense.unwrap();
        json!({
            "ephemeris": states.into_iter().map(state).collect::<Vec<_>>(),
            "final_state": state(result.final_state),
            "dense_state": state(dense.eval(targets[0]).unwrap()),
        })
    };
    json!({
        "base": evaluate(base()),
        "fixed": evaluate(fixed),
        "sourced": evaluate(sourced),
    })
}

const STATIC_FIELD: &str = "product_type gravity_field
modelname TEST
earth_gravity_constant 3.986e14
radius 6378000
max_degree 2
errors no
tide_system tide_free
end_of_head
gfc 2 0 -4.8e-4 0
gfc 2 1 0 0
gfc 2 2 0 0
";

fn terrestrial() -> Value {
    let day = 59024;
    let sample = |mjd_utc| Iers2010EopSample {
        mjd_utc,
        xp_arcsec: 0.1,
        yp_arcsec: 0.2,
        ut1_minus_utc_s: -0.2,
        lod_s: 0.001,
        dx_arcsec: 0.0001,
        dy_arcsec: -0.0001,
        pole_rates_rad_s: None,
    };
    let provider =
        Iers2010EarthOrientationProvider::new(vec![sample(day), sample(day + 1)]).unwrap();
    let orientation = provider.orientation_at_utc_mjd(day, 43200.0).unwrap();
    let orientation_tdb = provider.orientation_at_tdb_seconds(646272069.184).unwrap();
    let field = IcgemGravityField::parse(STATIC_FIELD).unwrap();
    let gravity = field.gravity(2, 2).unwrap();
    let position = [7000.0, 1000.0, 200.0];
    let s = [7_000.0, 0.0, 0.0];
    let sun = [-150_000_000.0, 0.0, 0.0];
    let moon = [-400_000.0, 0.0, 0.0];
    let model = EarthShadowModel::Wgs84Oblate;
    let gradient = |(value, vector): (f64, [f64; 3])| json!([hex(value), hexes(&vector)]);
    json!({
        "orientation": [
            hexes(&orientation.iers2010_quaternion().unwrap()), hexes(&orientation.earth_rotation_vector_itrf_rad_s()),
            json!(rows(orientation.gcrf_to_itrf_matrix())), json!(rows(orientation.itrf_to_gcrf_matrix())),
            json!(rows(orientation.gcrf_to_itrf_rotation_rate_matrix())), json!(rows(orientation.itrf_to_gcrf_rotation_rate_matrix())),
        ],
        "orientation_tdb": hexes(&orientation_tdb.iers2010_quaternion().unwrap()),
        "field": STATIC_FIELD,
        "gravity": [
            hexes(&gravity.body_fixed_acceleration_km_s2(position).unwrap()),
            hexes(&gravity.body_fixed_acceleration_at_km_s2(position, 0.0).unwrap()),
            hex(gravity.body_fixed_potential_km2_s2(position).unwrap()),
            hex(gravity.body_fixed_potential_at_km2_s2(position, 0.0).unwrap()),
        ],
        "eclipse": [
            hex(eclipse::moon_shadow_fraction(s, sun, moon).unwrap()),
            hex(eclipse::shadow_fraction_with_moon(s, sun, moon).unwrap()),
            hex(eclipse::shadow_fraction_with_model_and_moon(s, sun, moon, model).unwrap()),
            gradient(eclipse::shadow_fraction_with_gradient(s, sun).unwrap()),
            gradient(eclipse::shadow_fraction_with_model_and_gradient(s, sun, model).unwrap()),
            gradient(eclipse::shadow_fraction_with_model_moon_and_gradient(s, sun, moon, model).unwrap()),
        ],
    })
}

fn estimation() -> Value {
    let initial = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.5]);
    let f = EmpiricalRtnAcceleration::new()
        .with_term(RtnAxis::Radial, EmpiricalHarmonic::Constant, 1e-8)
        .unwrap();
    let equations = VariationalEquations {
        force: &f,
        parameters: &[1e-8],
        sensitivity_parameters: &[0],
    };
    let options = IntegratorOptions::default();
    let epochs = [VariationalEpoch::from_tdb_seconds(0.0, 120.0)];
    let ctx = PropagationContext::default();
    let result = propagate_variational(
        initial,
        &epochs,
        &equations,
        IntegratorKind::Dp54,
        &options,
        &ctx,
    )
    .unwrap();
    let e = &result[0];
    let pv = |state: CartesianState| {
        [
            hexes(&state.position_array()),
            hexes(&state.velocity_array()),
        ]
    };
    let epoch_seconds = [0, 120, 240, 360, 480];
    let propagate = StatePropagator::new(
        0.0,
        initial.position_array(),
        initial.velocity_array(),
        ForceModelKind::two_body(),
        IntegratorKind::Dp54,
    );
    let instants: Vec<_> = epoch_seconds
        .iter()
        .map(|&epoch| Instant::from_nanos(TimeScale::Tdb, i128::from(epoch) * 1_000_000_000))
        .collect();
    let sample_states = propagate.ephemeris(&epoch_seconds.map(f64::from)).unwrap();
    let samples: Vec<_> = sample_states
        .iter()
        .zip(instants)
        .map(|(state, epoch)| InertialPositionSample {
            epoch,
            position_gcrf_m: state.position_array().map(|value| value * 1000.0),
        })
        .collect();
    let mut fit_options = OrbitParameterFitOptions::default();
    fit_options.force_model = ForceModelKind::two_body();
    let fit = fit_inertial_orbit_parameters(&samples, "G01".parse().unwrap(), None, &fit_options)
        .unwrap();
    let OrbitParameterCovariance::Estimated {
        matrix: ref covariance,
        ref correlation,
        variance_factor,
        degrees_of_freedom,
    } = fit.covariance
    else {
        panic!("expected estimated covariance");
    };
    let matrix = |get: &dyn Fn(usize, usize) -> f64| {
        (0..6)
            .map(|row| (0..6).map(|col| hex(get(row, col))).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    };
    fit_options.max_condition_number = 1.5;
    let OrbitParameterFitError::IllConditionedNormalEquations {
        satellite,
        condition_number,
        limit,
        weakest,
    } = fit_inertial_orbit_parameters(&samples, "G01".parse().unwrap(), None, &fit_options)
        .unwrap_err()
    else {
        panic!("expected a condition number refusal");
    };
    let conditioning_refusal = json!({
        "kind": "ill_conditioned_normal_equations",
        "satellite": satellite.to_string(),
        "condition_number": hex(condition_number),
        "limit": hex(limit),
        "weakest": weakest.iter().map(|(name, value)| (name, hex(*value))).collect::<Vec<_>>(),
    });
    fit_options.max_condition_number = OrbitParameterFitOptions::default().max_condition_number;
    let mut force = CompositeForceModel::new();
    force.add(Box::new(TwoBodyGravity::default()));
    force.add(Box::new(f));
    let parameter_epochs: Vec<_> = epoch_seconds
        .iter()
        .map(|&epoch| VariationalEpoch::from_tdb_seconds(0.0, f64::from(epoch)))
        .collect();
    let equations = VariationalEquations {
        force: &force,
        parameters: &[1e-8],
        sensitivity_parameters: &[0],
    };
    let perturbed = propagate_variational(
        initial,
        &parameter_epochs,
        &equations,
        IntegratorKind::Dp54,
        &options,
        &ctx,
    )
    .unwrap();
    let perturbed_samples: Vec<_> = samples
        .iter()
        .zip(perturbed)
        .map(|(sample, state)| InertialPositionSample {
            epoch: sample.epoch,
            position_gcrf_m: state.state.position_array().map(|value| value * 1000.0),
        })
        .collect();
    let parameter_name =
        EmpiricalRtnAcceleration::parameter_name(RtnAxis::Radial, EmpiricalHarmonic::Constant);
    fit_options.additional_forces = vec![std::sync::Arc::new(
        EmpiricalRtnAcceleration::with_zero_terms(&[(
            RtnAxis::Radial,
            EmpiricalHarmonic::Constant,
        )]),
    )];
    fit_options.estimated_parameters = vec![parameter_name.to_string()];
    let fit_parameter = fit_inertial_orbit_parameters(
        &perturbed_samples,
        "G01".parse().unwrap(),
        None,
        &fit_options,
    )
    .unwrap();
    let estimate = fit_parameter.force_parameter(parameter_name).unwrap();
    json!({
        "variational": {
            "epoch": hex(epochs[0].elapsed_tt_s),
            "state": pv(e.state),
            "stm": (0..6).map(|row| (0..6).map(|col| hex(e.state_transition[(row,col)])).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "sensitivity": hexes(e.sensitivity.as_slice()),
        },
        "force_fit": {
            "observations": perturbed_samples.iter().map(|sample| hexes(&sample.position_gcrf_m)).collect::<Vec<_>>(),
            "name": estimate.name,
            "initial": hex(estimate.initial_value),
            "value": hex(estimate.value),
            "condition": hex(fit_parameter.condition_number),
        },
        "conditioning_refusal": conditioning_refusal,
        "fit": {
            "covariance": [matrix(&|r,c| covariance[(r,c)]), matrix(&|r,c| correlation[(r,c)]), json!(hex(variance_factor)), json!(degrees_of_freedom)],
            "rms_3d_m": hex(fit.residual_stats.rms_3d_m),
            "epochs": epoch_seconds,
            "observations": samples.iter().map(|sample| hexes(&sample.position_gcrf_m)).collect::<Vec<_>>(),
            "initial_state": pv(fit.initial_state),
            "initial_epoch": hex(fit.initial_state.epoch_tdb_seconds),
            "iterations": fit.iterations,
            "condition_number": hex(fit.condition_number),
            "seed_rms_3d_m": hex(fit.seed_rms_3d_m),
            "residuals": fit.residuals.iter().map(|residual| hexes(&[residual.epoch_tdb_j2000_s, residual.elapsed_tt_s, residual.radial_m, residual.along_m, residual.cross_m])).collect::<Vec<_>>(),
            "sigma": fit.sigma("x_km").map(hex),
            "correlation": fit.correlation("x_km", "x_km").map(hex),
        },
    })
}

fn terrestrial_parameter_fit(fixtures: &Path) -> Value {
    let bytes =
        std::fs::read(fixtures.join("sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3")).unwrap();
    let product = sidereon_core::ephemeris::Sp3::parse(&bytes).unwrap();
    let provider = Iers2010EarthOrientationProvider::new(
        [61160, 61161]
            .map(|mjd_utc| Iers2010EopSample {
                mjd_utc,
                xp_arcsec: 0.1,
                yp_arcsec: 0.2,
                ut1_minus_utc_s: -0.2,
                lod_s: 0.001,
                dx_arcsec: 0.0,
                dy_arcsec: 0.0,
                pole_rates_rad_s: None,
            })
            .to_vec(),
    )
    .unwrap();
    let fit = fit_sp3_ecef_orbit_parameters(
        &product,
        "G01".parse().unwrap(),
        &provider,
        None,
        &OrbitParameterFitOptions::default(),
    )
    .unwrap();
    json!({
        "position": hexes(&fit.initial_state.position_array()),
        "velocity": hexes(&fit.initial_state.velocity_array()),
        "condition": hex(fit.condition_number),
        "rms_3d_m": hex(fit.residual_stats.rms_3d_m),
        "observation_count": fit.residuals.len(),
    })
}

fn diagonal_covariance(diagonal: [f64; 6]) -> Covariance6 {
    Covariance6::try_from_matrix(std::array::from_fn(|row| {
        std::array::from_fn(|col| if row == col { diagonal[row] } else { 0.0 })
    }))
    .unwrap()
}

fn covariance_transport() -> Value {
    let propagator = StatePropagator::new(
        0.0,
        [7000.0, 0.0, 0.0],
        [0.0, 7.5, 1.0],
        ForceModelKind::two_body_j2(),
        IntegratorKind::Dp54,
    );
    let initial = LabeledCovariance6 {
        covariance: diagonal_covariance([1e-6, 2e-6, 3e-6, 1e-10, 2e-10, 3e-10]),
        frame: CovarianceFrame::Inertial,
    };
    let ephemeris = propagator
        .propagate_covariance(
            initial,
            &[0.0, 60.0, 120.0],
            &CovariancePropagationOptions::default(),
        )
        .unwrap();
    let node = ephemeris.nodes().last().unwrap();
    let matrix_hex = |matrix: &Mat6| matrix.iter().map(|row| hexes(row)).collect::<Vec<_>>();

    let tle1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
    let primary2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";
    let secondary2 = "2 25544  51.6414 100.0000 0003435 262.6267 010.0000 15.54005638121103";
    let covariance = diagonal_covariance([1.0e-6, 1.2e-6, 1.4e-6, 1.0e-10, 1.1e-10, 1.2e-10]);
    let mut finder = TcaFinderOptions::default();
    finder.coarse_step_seconds = 60.0;
    finder.time_tolerance_seconds = 1e-3;
    let conjunctions = find_tca_conjunctions_with_propagated_covariance_from_tles(
        TcaTle::new(tle1, primary2),
        TcaTle::new(tle1, secondary2),
        JulianDate(2458303.0, 0.0),
        JulianDate(2458304.0, 0.0),
        finder,
        TcaPropagatedCovariancePcOptions::new(
            0.02,
            PcMethod::FosterEqualArea,
            covariance,
            covariance,
        ),
    )
    .unwrap();
    let collision = conjunctions[0].collision_probability;
    json!({
        "position": hexes(&node.state.position_array()),
        "covariance": matrix_hex(node.covariance.as_matrix()),
        "interpolated": matrix_hex(ephemeris.covariance_at(90.0).unwrap().as_matrix()),
        "conjunction_count": conjunctions.len(),
        "collision": hexes(&[collision.pc, collision.miss_km, collision.sigma_x_km, collision.sigma_z_km]),
    })
}

pub fn section(core: &Path, fixtures: &Path) -> Value {
    json!({
        "celestial": celestial(core),
        "radiation": radiation(core),
        "analytic": analytic(),
        "attitude_trajectories": attitude_trajectories(core, fixtures),
        "broadcast_scores": broadcast_scores(core, fixtures),
        "additional_force_propagation": additional_force_propagation(core),
        "terrestrial": terrestrial(),
        "estimation": estimation(),
        "covariance_transport": covariance_transport(),
        "terrestrial_parameter_fit": terrestrial_parameter_fit(fixtures),
    })
}
