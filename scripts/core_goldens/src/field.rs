//! Goldens for the RTK static reference-station, fusion and DGNSS binding
//! tests. Each function builds its inputs as the Python test named in its doc
//! builds them through the binding, with every default the binding's
//! constructors fill in, and calls the core entry point the binding calls.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::{json, Value};
use sidereon_core::dgnss::{solve_position, CodeObservation};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::fusion::{
    smooth_fusion_rts, EkfUpdateOptions, ErrorStateLayout, FusionFilterKind,
    FusionRtsHistoryBuilder, FusionUpdate, GnssFixMeasurement, GnssFixStatus,
    GnssFixStatusWeighting, IggIiiMeasurementReweighting, InertialFilter, InertialFilterConfig,
    InnovationGate, InsFilterState, LooseCouplingConfig, StationaryDetectorConfig,
    StationaryUpdateConfig, TimeSyncHistoryConfig, YangPredictionAdaptiveFactor,
};
use sidereon_core::inertial::{ImuSample, ImuSpec, NavState};
use sidereon_core::positioning::{
    solve_static_reference_station_rinex, Corrections, KlobucharCoeffs, PseudorangeCode, QzssClock,
    SolveInputs, StaticReferenceCarrierRinexOptions, StaticReferenceStationRinexOptions,
    SurfaceMet, TroposphereModel,
};
use sidereon_core::rinex::observations::RinexObs;
use sidereon_core::rtk::BaselineReferenceSelection;
use sidereon_core::rtk_filter::defaults::{
    AMBIGUITY_TOL_M, MAX_ITERATIONS, POSITION_TOL_M, RATIO_THRESHOLD,
};
use sidereon_core::rtk_filter::{
    CycleSlipPolicy, DynamicsModel, FixedSolveOpts, FloatSolveOpts, MeasModel,
    ResidualValidationOpts, RtkArcConfig, RtkArcPreprocessing, RtkRinexArcOptions,
    RtkStaticArcConfig, SearchOpts, StochasticModel, UpdateOpts, ValidatedFixedSolveOpts,
};

const WGS84_A_M: f64 = 6_378_137.0;
const C_M_S: f64 = 299_792_458.0;

fn hex(value: f64) -> Value {
    Value::String(format!("{:#018x}", value.to_bits()))
}

fn hexes(values: &[f64]) -> Value {
    Value::Array(values.iter().copied().map(hex).collect())
}

fn diagonal(matrix: &[Vec<f64>]) -> Vec<f64> {
    matrix.iter().enumerate().map(|(i, row)| row[i]).collect()
}

fn load_sp3(path: &Path) -> Sp3 {
    sidereon::load_sp3(&std::fs::read(path).expect("read SP3")).expect("parse SP3")
}

/// The sections this module adds to `core_goldens.json`.
pub fn sections(core: &Path) -> Vec<(&'static str, Value)> {
    vec![
        ("rtk_static_reference", rtk_static_reference(core)),
        ("fusion", fusion()),
        ("dgnss", dgnss(core)),
    ]
}

// --- RTK static reference station -----------------------------------------

const WTZR_MARKER_M: [f64; 3] = [4075580.3111, 931854.0543, 4801568.2808];
const WTZR_OBS: &str = "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx";
const WTZZ_OBS: &str = "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx";
const WTZR_WTZZ_SP3: &str = "GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3";

fn load_obs(path: &Path) -> RinexObs {
    RinexObs::parse(&std::fs::read_to_string(path).expect("read RINEX OBS"))
        .expect("parse RINEX OBS")
}

/// The antenna reference point `tests/test_rtk.py::_arp_position` forms: the
/// marker moved along its own direction by the header's antenna height.
fn arp_position(marker_m: [f64; 3], obs: &RinexObs) -> [f64; 3] {
    let delta = obs
        .header()
        .antenna_delta_hen_m
        .expect("antenna delta H/E/N");
    assert_eq!(delta[1], 0.0, "east antenna offset");
    assert_eq!(delta[2], 0.0, "north antenna offset");
    let norm =
        (marker_m[0] * marker_m[0] + marker_m[1] * marker_m[1] + marker_m[2] * marker_m[2]).sqrt();
    [
        marker_m[0] + marker_m[0] / norm * delta[0],
        marker_m[1] + marker_m[1] / norm * delta[0],
        marker_m[2] + marker_m[2] / norm * delta[0],
    ]
}

/// `tests/test_rtk.py::test_static_reference_station_rinex_matches_core_oracle`:
/// `solve_static_reference_station_rinex(sp3, base_obs, rover_obs, base_arp_m,
/// model=RtkMeasurementModel(2.0, 0.01, sagnac=True, stochastic=SIMPLE,
/// elevation_weighting=True), arc_options=RtkRinexArcOptions(max_epochs=24,
/// include_prediction_time=False), preprocessing=RtkArcPreprocessing(
/// cycle_slip="split_arc"), float_options=RtkFloatOptions(1e-4, 1e-4, 10),
/// fixed_options=RtkFixedOptions(1e-4, 1e-4, 10, 3.0, True, 4),
/// enable_code_dgnss=False, enable_carrier_rtk=True, with_geodetic=True)`,
/// every other argument at its binding default. The reference position is
/// written out, and the test solves from it, so the solve does not depend on
/// how numpy rounds the marker norm.
fn rtk_static_reference(core: &Path) -> Value {
    let sp3 = load_sp3(&core.join("sp3").join(WTZR_WTZZ_SP3));
    let base_obs = load_obs(&core.join("obs").join(WTZR_OBS));
    let rover_obs = load_obs(&core.join("obs").join(WTZZ_OBS));
    let reference_position_m = arp_position(WTZR_MARKER_M, &base_obs);

    // RtkRinexArcOptions(max_epochs=24, include_prediction_time=False)
    let signal_pairs = RtkRinexArcOptions::gps_l1_c().signal_pairs;
    let mut arc_options = RtkRinexArcOptions::new(signal_pairs.clone(), Some(24), 4, false);
    arc_options.signal_pairs = signal_pairs;
    arc_options.max_epochs = Some(24);
    arc_options.min_common_satellites = 4;
    arc_options.include_prediction_time = false;

    let model = MeasModel {
        code_sigma_m: 2.0,
        phase_sigma_m: 0.01,
        sagnac: true,
        stochastic: StochasticModel::Simple {
            elevation_weighting: true,
        },
    };
    // RtkArcUpdateOptions() defaults.
    let update_opts = UpdateOpts {
        hold_sigma_m: 1.0e-4,
        position_tol_m: POSITION_TOL_M,
        ambiguity_tol_m: AMBIGUITY_TOL_M,
        max_iterations: MAX_ITERATIONS,
        process_noise_baseline_sigma_m: 0.0,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: Vec::new(),
        report_residuals: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: RATIO_THRESHOLD,
        },
    };
    let preprocessing = RtkArcPreprocessing {
        cycle_slip: Some(CycleSlipPolicy::SplitArc),
        hatch_window_cap: None,
        elevation_mask_deg: None,
    };
    let initial_baseline_m = [0.0; 3];
    let baseline_prior_sigma_m = 30.0;
    let ambiguity_prior_sigma_m = 30.0;
    let wavelengths_m: BTreeMap<String, f64> = BTreeMap::new();
    let offsets_m: BTreeMap<String, f64> = BTreeMap::new();
    let mut arc = RtkArcConfig::new(
        reference_position_m,
        BaselineReferenceSelection::Auto,
        model,
        baseline_prior_sigma_m,
        ambiguity_prior_sigma_m,
        initial_baseline_m,
        wavelengths_m.clone(),
        offsets_m.clone(),
        update_opts.clone(),
        preprocessing.clone(),
    );
    arc.base_m = reference_position_m;
    arc.reference = BaselineReferenceSelection::Auto;
    arc.model = model;
    arc.baseline_prior_sigma_m = baseline_prior_sigma_m;
    arc.ambiguity_prior_sigma_m = ambiguity_prior_sigma_m;
    arc.initial_baseline_m = initial_baseline_m;
    arc.wavelengths_m = wavelengths_m;
    arc.offsets_m = offsets_m;
    arc.update_opts = update_opts;
    arc.preprocessing = preprocessing;

    let opts = ValidatedFixedSolveOpts {
        float: FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
        },
        fixed: FixedSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: true,
            partial_min_ambiguities: 4,
        },
        residual: ResidualValidationOpts {
            threshold_sigma: None,
            max_exclusions: 0,
        },
    };
    let mut static_config = RtkStaticArcConfig::new(arc.clone(), opts);
    static_config.arc = arc;
    static_config.opts = opts;

    let mut carrier =
        StaticReferenceCarrierRinexOptions::new(arc_options.clone(), static_config.clone());
    carrier.arc_options = arc_options;
    carrier.static_config = static_config;
    let mut options = StaticReferenceStationRinexOptions::new(None, Some(carrier.clone()), true);
    options.code_options = None;
    options.carrier_options = Some(carrier);
    options.with_geodetic = true;

    let solution = solve_static_reference_station_rinex(
        &sp3,
        &base_obs,
        &rover_obs,
        reference_position_m,
        &options,
    )
    .expect("static reference-station solve");
    let covariance: Vec<f64> = solution
        .covariance
        .position_ecef_m2
        .iter()
        .flatten()
        .copied()
        .collect();
    let carrier_solution = solution
        .carrier_solution
        .as_ref()
        .expect("carrier solution");
    json!({
        "reference_position_m": hexes(&reference_position_m),
        "position_m": hexes(&solution.position.as_array()),
        "baseline_vector_m": hexes(&solution.baseline_vector_m),
        "baseline_m": hex(solution.baseline_m),
        "position_covariance_ecef_m2": hexes(&covariance),
        "integer_ratio": carrier_solution.integer_ratio.map(hex),
        "diagnostic_count": solution.diagnostics.len(),
    })
}

// --- fusion -----------------------------------------------------------------

fn update_block(update: &FusionUpdate) -> Value {
    json!({
        "applied": update.applied,
        "nis": hex(update.nis),
        "rows": update.rows,
        "accepted_rows": update.accepted_rows,
        "rejected_rows": update.rejected_rows,
        "ekf_nis": hex(update.ekf.normalized_innovation_squared),
    })
}

fn state_block(state: &InsFilterState) -> Value {
    json!({
        "position_ecef_m": hexes(&state.nominal.position_ecef_m),
        "velocity_ecef_mps": hexes(&state.nominal.velocity_ecef_mps),
        "gyro_bias_rps": hexes(&state.nominal.gyro_bias_rps),
        "covariance_diagonal": hexes(&diagonal(&state.covariance)),
    })
}

/// `NavState(t, position, velocity)`: identity attitude and zero biases.
fn nav_state(t_j2000_s: f64, position: [f64; 3], velocity: [f64; 3]) -> NavState {
    let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    NavState::new(t_j2000_s, position, velocity, identity)
        .and_then(|state| state.with_biases([0.0; 3], [0.0; 3]))
        .expect("nav state")
}

/// `InsFilterState.from_diagonal(NavState(0.0, [a, 0, 0], [0, 0, 0]),
/// ErrorStateLayout.FIFTEEN, [1.0] * 15)`, the `_state()` and
/// `_filter_state()` of the fusion tests.
fn unit_state() -> InsFilterState {
    InsFilterState::from_diagonal(
        nav_state(0.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3]),
        ErrorStateLayout::Fifteen,
        &[1.0; 15],
    )
    .expect("filter state")
}

/// `ImuSpec(0.0, 0.0, 0.0, 0.0, math.inf, math.inf)`, the `_spec()` of
/// `tests/test_field_mode_bindings.py`.
fn ideal_spec() -> ImuSpec {
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, f64::INFINITY, f64::INFINITY, None, None);
    spec.validate().expect("IMU spec");
    spec
}

/// `InertialFilterConfig(spec, loose=loose)`; `None` leaves the loose
/// configuration at its core default, as omitting `loose` does.
fn filter_config(spec: ImuSpec, loose: Option<LooseCouplingConfig>) -> InertialFilterConfig {
    let mut config = InertialFilterConfig::new(spec).expect("filter config");
    config.filter_kind = FusionFilterKind::Ekf;
    if let Some(loose) = loose {
        config.loose = loose;
    }
    config.validate().expect("filter config");
    config
}

/// `LooseCouplingConfig(...)` with the keyword arguments given and the rest at
/// the binding defaults.
fn loose_config(
    update_options: Option<EkfUpdateOptions>,
    fix_status_weighting: Option<GnssFixStatusWeighting>,
    measurement_reweighting: Option<IggIiiMeasurementReweighting>,
    prediction_adaptation: Option<YangPredictionAdaptiveFactor>,
    stationary_updates: Option<StationaryUpdateConfig>,
) -> LooseCouplingConfig {
    let mut loose = LooseCouplingConfig::default();
    loose.lever_arm_body_m = [0.0; 3];
    loose.update_options = update_options.unwrap_or_default();
    loose.fix_status_weighting = fix_status_weighting.unwrap_or_default();
    loose.measurement_reweighting = measurement_reweighting;
    loose.prediction_adaptation = prediction_adaptation;
    loose.stationary_updates = stationary_updates;
    loose.non_holonomic = None;
    loose.validate().expect("loose config");
    loose
}

/// `_position_velocity_fix(fix_status)` of `tests/test_field_mode_bindings.py`.
fn position_velocity_fix(fix_status: GnssFixStatus) -> GnssFixMeasurement {
    let mut identity6 = vec![vec![0.0; 6]; 6];
    for (index, row) in identity6.iter_mut().enumerate() {
        row[index] = 1.0;
    }
    GnssFixMeasurement::position_velocity(
        0.0,
        [WGS84_A_M + 1.0, 2.0, -3.0],
        [0.4, -0.2, 0.1],
        identity6,
        8,
    )
    .expect("position-velocity fix")
    .with_fix_status(fix_status)
}

/// `GnssFixMeasurement.position(t, position, np.eye(3) * scale, satellites)`.
fn position_fix(
    t_j2000_s: f64,
    position: [f64; 3],
    scale_m2: f64,
    satellites_used: usize,
) -> GnssFixMeasurement {
    let covariance = [
        [scale_m2, 0.0, 0.0],
        [0.0, scale_m2, 0.0],
        [0.0, 0.0, scale_m2],
    ];
    GnssFixMeasurement::position(t_j2000_s, position, covariance, satellites_used)
        .expect("position fix")
        .with_fix_status(GnssFixStatus::Single)
}

fn fusion() -> Value {
    json!({
        "loose_default": loose_default(),
        "stationary": stationary(),
        "fix_status_weighting": fix_status_weighting(),
        "checkpoint_ukf_time_sync": checkpoint_ukf_time_sync(),
        "robust_loose_rts": robust_loose_rts(),
    })
}

/// `test_loose_field_mode_default_omission_matches_plain_bits`: the plain
/// filter, `InertialFilter.with_config(_state(), InertialFilterConfig(_spec()))`,
/// after `update_loose(_position_velocity_fix())`.
fn loose_default() -> Value {
    let mut filter = InertialFilter::with_config(unit_state(), filter_config(ideal_spec(), None))
        .expect("filter");
    let update = filter
        .update_loose(&position_velocity_fix(GnssFixStatus::Single))
        .expect("loose update");
    json!({
        "update": update_block(&update),
        "state": state_block(filter.state()),
    })
}

/// `test_stationary_zupt_zaru_update_matches_core_bits`: a stationary update
/// config of `StationaryDetectorConfig(1, 100.0, 1.0)`, 0.5 m/s and 0.05 rad/s,
/// one `ImuSample.increment(1.0, [0, 0, 0], [0, 0, 0], 1.0)` propagation and
/// `update_stationary()`.
fn stationary() -> Value {
    let mut detector = StationaryDetectorConfig::new(1, 100.0, 1.0);
    detector.window_len = 1;
    detector.max_specific_force_norm_error_mps2 = 100.0;
    detector.max_body_rate_wrt_ecef_norm_rps = 1.0;
    detector.validate().expect("detector");
    let mut stationary = StationaryUpdateConfig::new(detector, 0.5, 0.05);
    stationary.detector = detector;
    stationary.zero_velocity_sigma_mps = 0.5;
    stationary.zero_angular_rate_sigma_rps = 0.05;
    stationary.validate().expect("stationary config");
    let loose = loose_config(None, None, None, None, Some(stationary));
    let mut filter =
        InertialFilter::with_config(unit_state(), filter_config(ideal_spec(), Some(loose)))
            .expect("filter");
    filter
        .propagate(ImuSample::increment(1.0, [0.0; 3], [0.0; 3], 1.0))
        .expect("propagate");
    let update = filter
        .update_stationary()
        .expect("stationary update")
        .expect("stationary update applies");
    json!({
        "update": update_block(&update),
        "state": state_block(filter.state()),
    })
}

/// `test_fix_status_weighting_covariance_ordering_matches_core_bits`:
/// `GnssFixStatusWeighting(3.0, 2.0, 1.0)` and one `update_loose` of
/// `_position_velocity_fix(fix_status=status)` on a fresh filter per status.
fn fix_status_weighting() -> Value {
    let weighting = GnssFixStatusWeighting {
        single_sigma_multiplier: 3.0,
        float_sigma_multiplier: 2.0,
        fixed_sigma_multiplier: 1.0,
    };
    weighting.validate().expect("weighting");
    let loose = loose_config(None, Some(weighting), None, None, None);
    let mut out = serde_json::Map::new();
    for (label, status) in [
        ("single", GnssFixStatus::Single),
        ("float", GnssFixStatus::Float),
        ("fixed", GnssFixStatus::Fixed),
    ] {
        let mut filter =
            InertialFilter::with_config(unit_state(), filter_config(ideal_spec(), Some(loose)))
                .expect("filter");
        let update = filter
            .update_loose(&position_velocity_fix(status))
            .expect("loose update");
        out.insert(
            label.into(),
            json!({
                "update": update_block(&update),
                "state": state_block(filter.state()),
            }),
        );
    }
    Value::Object(out)
}

/// `tests/test_018_domain_exposure.py::
/// test_fusion_filter_checkpoint_loose_ukf_tight_and_time_sync_bits`: the EKF
/// `InertialFilter(state, ImuSpec.mems())` and the UKF filter each after
/// `update_loose(_zero_fix(0.0, 4.0))`, and the time-synchronized late update
/// `update_loose_time_sync(_zero_fix(0.5, 25.0))` after four rate samples at
/// 0.25 s steps with `TimeSyncHistoryConfig(8, 4)`.
fn checkpoint_ukf_time_sync() -> Value {
    let spec = ImuSpec::mems();
    let zero = [WGS84_A_M, 0.0, 0.0];

    let mut ekf = InertialFilter::new(unit_state(), spec).expect("filter");
    let ekf_update = ekf
        .update_loose(&position_fix(0.0, zero, 4.0, 5))
        .expect("EKF loose update");

    let mut ukf_config = InertialFilterConfig::new(spec).expect("filter config");
    ukf_config.filter_kind = FusionFilterKind::Ukf;
    ukf_config.validate().expect("filter config");
    let mut ukf = InertialFilter::with_config(unit_state(), ukf_config).expect("UKF filter");
    let ukf_update = ukf
        .update_loose(&position_fix(0.0, zero, 4.0, 5))
        .expect("UKF loose update");

    let mut replay = InertialFilter::new(unit_state(), spec).expect("filter");
    let history = TimeSyncHistoryConfig::new(8, 4);
    history.validate().expect("time-sync history");
    replay
        .configure_time_sync_history(history)
        .expect("configure time sync");
    for t_j2000_s in [0.25, 0.5, 0.75, 1.0] {
        replay
            .propagate(ImuSample::rate(t_j2000_s, [0.0; 3], [0.0; 3]))
            .expect("propagate");
    }
    let replayed = replay
        .update_loose_time_sync(&position_fix(0.5, zero, 25.0, 5))
        .expect("time-sync update");
    json!({
        "ekf_update": update_block(&ekf_update),
        "ukf_update": update_block(&ukf_update),
        "time_sync": {
            "late_measurement": replayed.late_measurement,
            "replayed_imu_segments": replayed.replayed_imu_segments,
            "restored_checkpoint_epoch_j2000_s": hex(replayed.restored_checkpoint_epoch_j2000_s),
            "current_epoch_j2000_s": hex(replayed.current_epoch_j2000_s),
            "update": update_block(&replayed.update),
        },
    })
}

/// `tests/test_018_domain_exposure.py::test_fusion_robust_loose_recorded_rts_bits`:
/// a loose configuration with `EkfUpdateOptions(InnovationGate(4.0, 2))`, the
/// standard IGG-III reweighting and Yang adaptation, one recorded rate
/// propagation to 1.0 s, one recorded position update and the RTS smoothing of
/// the recorded history.
fn robust_loose_rts() -> Value {
    let spec = ImuSpec::mems();
    let gate = InnovationGate {
        threshold_sigma: 4.0,
        min_rows: 2,
    };
    gate.validate().expect("gate");
    let mut update_options = EkfUpdateOptions::default();
    update_options.innovation_gate = Some(gate);
    let loose = loose_config(
        Some(update_options),
        None,
        Some(IggIiiMeasurementReweighting::standard()),
        Some(YangPredictionAdaptiveFactor::standard()),
        None,
    );
    let mut filter = InertialFilter::with_config(unit_state(), filter_config(spec, Some(loose)))
        .expect("filter");
    let mut history = FusionRtsHistoryBuilder::from_filter(&filter).expect("history");
    let snapshot = filter.snapshot();
    filter.restore_snapshot(&snapshot).expect("restore");
    filter
        .propagate_recorded(ImuSample::rate(1.0, [0.0; 3], [0.0; 3]), &mut history)
        .expect("recorded propagate");
    let update = filter
        .update_loose_recorded(
            &position_fix(1.0, [6_378_137.35, 0.2, -0.1], 0.5, 7),
            &mut history,
        )
        .expect("recorded loose update");
    let recorded = history.clone().finish().expect("history");
    let smoothed = smooth_fusion_rts(&recorded).expect("RTS smoothing");
    let gate_report = update.ekf.innovation_gate.as_ref().expect("gate report");
    json!({
        "update": update_block(&update),
        "gate_max_abs_normalized_innovation": gate_report.max_abs_normalized_innovation.map(hex),
        "state": state_block(filter.state()),
        "recorded_epoch_count": recorded.epochs.len(),
        "recorded_transition_diagonal": hexes(&diagonal(
            recorded.epochs[1]
                .transition_from_previous
                .as_ref()
                .expect("transition"),
        )),
        "smoothed": smoothed.epochs.iter().map(|epoch| json!({
            "position_ecef_m": hexes(&epoch.snapshot.state.nominal.position_ecef_m),
            "error_state_correction": hexes(&epoch.error_state_correction),
            "covariance_diagonal": hexes(&diagonal(&epoch.covariance)),
        })).collect::<Vec<_>>(),
    })
}

// --- DGNSS ------------------------------------------------------------------

const DGNSS_SP3: &str = "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3";
const DGNSS_EPOCH_INDEX: usize = 48;
const DGNSS_ELEVATION_MASK_DEG: f64 = 10.0;
const DGNSS_ROVER_OFFSET_M: [f64; 3] = [30.0, -40.0, 20.0];

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// `tests/test_dgnss.py::_geodetic_to_ecef`.
fn geodetic_to_ecef(lat_deg: f64, lon_deg: f64, h_m: f64) -> [f64; 3] {
    let a = 6378137.0;
    let f = 1.0 / 298.257223563;
    let e2 = f * (2.0 - f);
    let lat = lat_deg.to_radians();
    let lon = lon_deg.to_radians();
    let n = a / (1.0 - e2 * lat.sin().powi(2)).sqrt();
    [
        (n + h_m) * lat.cos() * lon.cos(),
        (n + h_m) * lat.cos() * lon.sin(),
        (n * (1.0 - e2) + h_m) * lat.sin(),
    ]
}

/// `tests/test_dgnss.py::_synth`: for each GPS satellite of the product above a
/// 10 degree mask at the receiver, `(token, geometric range - c * dt_sat)`.
fn synthesize(sp3: &Sp3, position: [f64; 3], t_rx: f64) -> Vec<(String, f64)> {
    let length = norm(position);
    let up = [
        position[0] / length,
        position[1] / length,
        position[2] / length,
    ];
    let mut out = Vec::new();
    for sat in sp3.satellites().iter().copied() {
        let token = sat.to_string();
        if !token.starts_with('G') {
            continue;
        }
        let state = sp3
            .position_at_j2000_seconds(sat, t_rx)
            .expect("SP3 interpolation");
        let pos = state.position.as_array();
        let Some(dt) = state.clock_s else { continue };
        if !pos.iter().all(|value| value.is_finite()) || !dt.is_finite() {
            continue;
        }
        let los = [
            pos[0] - position[0],
            pos[1] - position[1],
            pos[2] - position[2],
        ];
        let range = norm(los);
        let up_component = los[0] * up[0] + los[1] * up[1] + los[2] * up[2];
        let elevation_deg = (up_component / range).asin().to_degrees();
        if elevation_deg < DGNSS_ELEVATION_MASK_DEG {
            continue;
        }
        out.push((token, range - C_M_S * dt));
    }
    out
}

fn observations_json(observations: &[(String, f64)]) -> Value {
    Value::Array(
        observations
            .iter()
            .map(|(sat, pseudorange)| json!([sat, hex(*pseudorange)]))
            .collect(),
    )
}

fn code_observations(observations: &[(String, f64)]) -> Vec<CodeObservation> {
    observations
        .iter()
        .map(|(sat, pseudorange)| CodeObservation::new(sat.clone(), *pseudorange))
        .collect()
}

/// `tests/test_dgnss.py::test_dgnss_solve_recovers_rover_and_baseline`:
/// `dgnss_solve(sp3, base, base_obs, rover_obs, SppConfig(observations=[],
/// t_rx_j2000_s=t_rx, t_rx_second_of_day_s=0.0, day_of_year=176.0,
/// initial_guess=[6378137.0, 0.0, 0.0, 0.0], with_geodetic=True))`. The
/// synthesized inputs are written out and the test solves from them, so the
/// solve does not depend on how numpy rounds the synthesis.
fn dgnss(core: &Path) -> Value {
    let sp3 = load_sp3(&core.join("sp3").join(DGNSS_SP3));
    let t_rx = sp3.epochs_j2000_seconds()[DGNSS_EPOCH_INDEX];
    let base = geodetic_to_ecef(55.75, 37.62, 200.0);
    let rover = [
        base[0] + DGNSS_ROVER_OFFSET_M[0],
        base[1] + DGNSS_ROVER_OFFSET_M[1],
        base[2] + DGNSS_ROVER_OFFSET_M[2],
    ];
    let base_obs = synthesize(&sp3, base, t_rx);
    let rover_obs = synthesize(&sp3, rover, t_rx);
    let inputs = SolveInputs {
        observations: Vec::new(),
        t_rx_j2000_s: t_rx,
        t_rx_second_of_day_s: 0.0,
        day_of_year: 176.0,
        initial_guess: [6378137.0, 0.0, 0.0, 0.0],
        corrections: Corrections {
            ionosphere: false,
            troposphere: false,
        },
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet::default(),
        troposphere_model: TroposphereModel::Rtklib,
        robust: None,
        pseudorange_code: PseudorangeCode::SingleFrequency,
        qzss_clock: QzssClock::Gps,
    };
    let solution = solve_position(
        &sp3,
        base,
        &code_observations(&base_obs),
        &code_observations(&rover_obs),
        inputs,
        true,
    )
    .expect("DGNSS solve");
    let receiver = &solution.solution;
    let position = [
        receiver.position.x_m,
        receiver.position.y_m,
        receiver.position.z_m,
    ];
    json!({
        "t_rx_j2000_s": hex(t_rx),
        "base_position_m": hexes(&base),
        "rover_position_m": hexes(&rover),
        "base_observations": observations_json(&base_obs),
        "rover_observations": observations_json(&rover_obs),
        "solution": {
            "position_m": hexes(&position),
            "rx_clock_s": hex(receiver.rx_clock_s),
            "used_sats": receiver.used_sats.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "residuals_m": hexes(&receiver.residuals_m),
            "baseline_vector_m": hexes(&solution.baseline_vector_m),
            "baseline_m": hex(solution.baseline_m),
            "dropped_sats": solution.dropped_sats,
        },
    })
}
