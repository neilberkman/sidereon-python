//! Writes the solutions `tests/test_ppp.py` checks beyond the ones the core's
//! ESBC fixture dump states: the float, fixed and fixed-float solution
//! metadata, the float solve with a 10 degree elevation cutoff, and the float
//! solve estimating troposphere gradients.
//!
//! The core dump (`SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test
//! ppp_real_arc`) writes the fixture's inputs and its core-level `expected`
//! values. This program reads those inputs, builds the solve inputs exactly as
//! the Python constructors build them, solves with the same `sidereon` entry
//! points the binding calls, and adds the binding-level values to the
//! `expected` block. Run it from this directory after regenerating the core
//! dump:
//!
//!     cargo run --release --locked -- ../../tests/fixtures/ppp_esbc.json <core-fixtures>/sp3
//!
//! It rewrites the fixture in place and changes no input.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

use serde_json::{json, Map, Value};
use sidereon_core::atmosphere::troposphere::Met;
use sidereon_core::positioning::SurfaceMet;
use sidereon_core::ppp_corrections::{
    build as build_ppp_corrections, CivilDateTime, PppCorrectionEpoch, PppCorrectionObservation,
    PppCorrectionsOptions,
};
use sidereon_core::precise_positioning::{
    FixedAmbiguityOptions, FixedSolution, FixedSolveConfig, FloatEpoch, FloatObservation,
    FloatResidual, FloatSolution, FloatSolveConfig, FloatSolveOptions, FloatState, FloatStatus,
    MeasurementWeights, PositionCovariance, PppAutoInitOptions, PppCorrectionLookup,
    PppInitialGuess, RangeCorrections, TemporalCorrelationSummary, TropoMapping,
    TroposphereOptions, UnplacedObservation, UnplacedObservationReason,
};
use sidereon_core::GnssSatelliteId;

#[path = "../../core_source_identity.rs"]
mod core_source_identity;

fn f(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn epochs(fx: &Value) -> Vec<FloatEpoch> {
    fx["epochs"]
        .as_array()
        .expect("epochs")
        .iter()
        .map(|epoch| {
            let civil = &epoch["civil"];
            FloatEpoch {
                epoch: CivilDateTime {
                    year: civil["year"].as_i64().expect("year") as i32,
                    month: civil["month"].as_u64().expect("month") as u8,
                    day: civil["day"].as_u64().expect("day") as u8,
                    hour: civil["hour"].as_u64().expect("hour") as u8,
                    minute: civil["minute"].as_u64().expect("minute") as u8,
                    second: f(&civil["second"]),
                },
                jd_whole: f(&epoch["jd_whole"]),
                jd_fraction: f(&epoch["jd_fraction"]),
                t_rx_j2000_s: f(&epoch["t_rx_j2000_s"]),
                observations: epoch["observations"]
                    .as_array()
                    .expect("observations")
                    .iter()
                    .map(|obs| {
                        let satellite_id = obs["satellite_id"].as_str().expect("sat").to_string();
                        FloatObservation {
                            sat: GnssSatelliteId::from_str(&satellite_id).expect("satellite"),
                            satellite_id,
                            ambiguity_id: obs["ambiguity_id"]
                                .as_str()
                                .expect("ambiguity")
                                .to_string(),
                            code_m: f(&obs["code_m"]),
                            phase_m: f(&obs["phase_m"]),
                            freq1_hz: f(&obs["freq1_hz"]),
                            freq2_hz: f(&obs["freq2_hz"]),
                            glonass_channel: None,
                            signals: None,
                        }
                    })
                    .collect(),
            }
        })
        .collect()
}

fn state(fx: &Value) -> FloatState {
    let raw = &fx["initial_state"];
    let position: Vec<f64> = raw["position_m"]
        .as_array()
        .expect("position")
        .iter()
        .map(f)
        .collect();
    FloatState {
        position_m: [position[0], position[1], position[2]],
        clocks_m: raw["clocks_m"]
            .as_array()
            .expect("clocks")
            .iter()
            .map(f)
            .collect(),
        ambiguities_m: raw["ambiguities_m"]
            .as_array()
            .expect("ambiguities")
            .iter()
            .map(|pair| (pair[0].as_str().expect("id").to_string(), f(&pair[1])))
            .collect(),
        ztd_m: f(&raw["ztd_m"]),
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    }
}

fn weights(raw: &Value) -> MeasurementWeights {
    MeasurementWeights {
        code: f(&raw["code"]),
        phase: f(&raw["phase"]),
        elevation_weighting: raw["elevation_weighting"].as_bool().expect("flag"),
    }
}

/// `PppTroposphereOptions(...)` as the binding builds it.
fn tropo(raw: &Value, estimate_tropo_gradients: bool) -> TroposphereOptions {
    if !raw["enabled"].as_bool().expect("enabled") {
        return TroposphereOptions::disabled();
    }
    let met = Met::new(
        f(&raw["pressure_hpa"]),
        f(&raw["temperature_k"]),
        f(&raw["relative_humidity"]),
    )
    .expect("met");
    let mut tropo = TroposphereOptions::new(met);
    tropo.enabled = true;
    tropo.estimate_ztd = raw["estimate_ztd"].as_bool().expect("ztd");
    tropo.estimate_tropo_gradients = estimate_tropo_gradients;
    tropo.met = met;
    tropo.mapping = TropoMapping::Niell;
    tropo
}

fn options(raw: &Value) -> FloatSolveOptions {
    let mut opts = FloatSolveOptions::default();
    opts.max_iterations = raw["max_iterations"].as_u64().expect("iterations") as usize;
    opts.position_tolerance_m = f(&raw["position_tolerance_m"]);
    opts.clock_tolerance_m = f(&raw["clock_tolerance_m"]);
    opts.ambiguity_tolerance_m = f(&raw["ambiguity_tolerance_m"]);
    opts.ztd_tolerance_m = f(&raw["ztd_tolerance_m"]);
    opts
}

/// `PppFloatConfig(...)` as the binding builds it.
fn float_config(
    fx: &Value,
    elevation_cutoff_deg: Option<f64>,
    estimate_tropo_gradients: bool,
) -> FloatSolveConfig {
    let raw = &fx["config"];
    let weights = weights(&raw["weights"]);
    let tropo = tropo(&raw["tropo"], estimate_tropo_gradients);
    let opts = options(&raw["opts"]);
    let residual_screen = raw["residual_screen"].as_bool().expect("screen");
    let corrections = RangeCorrections::disabled();
    let mut config = FloatSolveConfig::new(
        weights,
        tropo.clone(),
        corrections.clone(),
        opts,
        elevation_cutoff_deg,
        residual_screen,
        false,
    );
    config.weights = weights;
    config.tropo = tropo;
    config.corrections = corrections;
    config.opts = opts;
    config.elevation_cutoff_deg = elevation_cutoff_deg;
    config.residual_screen = residual_screen;
    config.estimate_residual_ionosphere = false;
    config
}

/// `PppFixedConfig(...)` as the binding builds it.
fn fixed_config(fx: &Value) -> FixedSolveConfig {
    let raw = &fx["fixed_config"];
    let ambiguity_raw = &raw["ambiguity"];
    let to_map = |value: &Value| -> BTreeMap<String, f64> {
        value
            .as_object()
            .expect("map")
            .iter()
            .map(|(key, value)| (key.clone(), f(value)))
            .collect()
    };
    let ratio_threshold = f(&ambiguity_raw["ratio_threshold"]);
    let mut ambiguity = FixedAmbiguityOptions::new(ratio_threshold);
    ambiguity.wavelengths_m = to_map(&ambiguity_raw["wavelengths_m"]);
    ambiguity.offsets_m = to_map(&ambiguity_raw["offsets_m"]);
    ambiguity.ratio_threshold = ratio_threshold;
    let weights = weights(&raw["weights"]);
    let tropo = tropo(&raw["tropo"], false);
    let opts = options(&raw["opts"]);
    let corrections = RangeCorrections::disabled();
    let mut config = FixedSolveConfig::new(
        weights,
        tropo.clone(),
        corrections.clone(),
        opts,
        None,
        ambiguity.clone(),
        false,
    );
    config.weights = weights;
    config.tropo = tropo;
    config.corrections = corrections;
    config.opts = opts;
    config.elevation_cutoff_deg = None;
    config.ambiguity = ambiguity;
    config.estimate_residual_ionosphere = false;
    config
}

fn matrix3(matrix: &[[f64; 3]; 3]) -> Value {
    json!(matrix)
}

fn temporal(summary: &TemporalCorrelationSummary) -> Value {
    json!({
        "lag1_autocorrelation": summary.lag1_autocorrelation,
        "decorrelation_time_epochs": summary.decorrelation_time_epochs,
        "decorrelation_time_s": summary.decorrelation_time_s,
        "nominal_sample_count": summary.nominal_sample_count,
        "effective_sample_count": summary.effective_sample_count,
        "variance_inflation_factor": summary.variance_inflation_factor,
        "arcs_used": summary.arcs_used,
    })
}

#[allow(clippy::too_many_arguments)]
fn metadata(
    covariance: &PositionCovariance,
    formal: &PositionCovariance,
    temporal_covariance: &PositionCovariance,
    posterior_variance_factor: f64,
    position_covariance_scale_factor: f64,
    temporal_position_covariance_scale_factor: f64,
    correlation: &TemporalCorrelationSummary,
    gradients: (Option<f64>, Option<f64>),
    gradient_covariances: (Option<[[f64; 2]; 2]>, Option<[[f64; 2]; 2]>),
    residual_ionosphere_m: &BTreeMap<String, f64>,
) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert(
        "position_covariance_ecef_m2".into(),
        matrix3(&covariance.ecef_m2),
    );
    out.insert(
        "position_covariance_enu_m2".into(),
        matrix3(&covariance.enu_m2),
    );
    out.insert(
        "formal_position_covariance_ecef_m2".into(),
        matrix3(&formal.ecef_m2),
    );
    out.insert(
        "formal_position_covariance_enu_m2".into(),
        matrix3(&formal.enu_m2),
    );
    out.insert(
        "posterior_variance_factor".into(),
        json!(posterior_variance_factor),
    );
    out.insert(
        "position_covariance_scale_factor".into(),
        json!(position_covariance_scale_factor),
    );
    out.insert(
        "temporal_position_covariance_ecef_m2".into(),
        matrix3(&temporal_covariance.ecef_m2),
    );
    out.insert(
        "temporal_position_covariance_enu_m2".into(),
        matrix3(&temporal_covariance.enu_m2),
    );
    out.insert(
        "temporal_position_covariance_scale_factor".into(),
        json!(temporal_position_covariance_scale_factor),
    );
    out.insert("temporal_correlation".into(), temporal(correlation));
    out.insert("tropo_gradient_north_m".into(), json!(gradients.0));
    out.insert("tropo_gradient_east_m".into(), json!(gradients.1));
    out.insert("residual_ionosphere_m".into(), json!(residual_ionosphere_m));
    out.insert(
        "tropo_gradient_covariance_m2".into(),
        json!(gradient_covariances.0),
    );
    out.insert(
        "formal_tropo_gradient_covariance_m2".into(),
        json!(gradient_covariances.1),
    );
    out
}

fn hex(value: f64) -> Value {
    Value::String(format!("{:#018x}", value.to_bits()))
}

fn status_label(status: FloatStatus) -> &'static str {
    match status {
        FloatStatus::StateTolerance => "state_tolerance",
        FloatStatus::MaxIterations => "max_iterations",
    }
}

/// The residual rows the tests compare: the count and the first five rows,
/// every value as its IEEE-754 bits.
fn residual_block(rows: &[FloatResidual]) -> Value {
    json!({
        "count": rows.len(),
        "first": rows.iter().take(5).map(|row| json!({
            "epoch_index": row.epoch_index,
            "satellite_id": row.satellite_id,
            "ambiguity_id": row.ambiguity_id,
            "code_m": hex(row.code_m),
            "phase_m": hex(row.phase_m),
            "code_weight": hex(row.code_weight),
            "phase_weight": hex(row.phase_weight),
        })).collect::<Vec<_>>(),
    })
}

fn unplaced_block(rows: &[UnplacedObservation]) -> Value {
    Value::Array(
        rows.iter()
            .map(|row| {
                json!({
                    "epoch_index": row.epoch_index,
                    "satellite_id": row.satellite_id,
                    "ambiguity_id": row.ambiguity_id,
                    "reason": match row.reason {
                        UnplacedObservationReason::CodeNotPositive => "code_not_positive",
                        _ => panic!("unplaced reason this program does not name"),
                    },
                })
            })
            .collect(),
    )
}

/// The solve outcome of a float solution beyond its covariance metadata.
#[allow(clippy::too_many_arguments)]
fn solve_block(
    position_m: &[f64; 3],
    status: FloatStatus,
    iterations: usize,
    converged: bool,
    epoch_clocks_m: &[f64],
    solved_epoch_indices: &[usize],
    unplaced: &[UnplacedObservation],
    residuals: &[FloatResidual],
    ssr_bias_exclusions: usize,
) -> Map<String, Value> {
    let mut out = Map::new();
    out.insert(
        "position_bits".into(),
        Value::Array(position_m.iter().copied().map(hex).collect()),
    );
    out.insert("status".into(), json!(status_label(status)));
    out.insert("iterations".into(), json!(iterations));
    out.insert("converged".into(), json!(converged));
    out.insert(
        "epoch_clocks_bits".into(),
        Value::Array(epoch_clocks_m.iter().copied().map(hex).collect()),
    );
    out.insert("solved_epoch_indices".into(), json!(solved_epoch_indices));
    out.insert("unplaced_observations".into(), unplaced_block(unplaced));
    out.insert("residuals".into(), residual_block(residuals));
    out.insert(
        "ssr_bias_exclusion_count".into(),
        json!(ssr_bias_exclusions),
    );
    out
}

fn float_outcome(sol: &FloatSolution) -> Map<String, Value> {
    let mut out = solve_block(
        &sol.position_m,
        sol.status,
        sol.iterations,
        sol.converged,
        &sol.epoch_clocks_m,
        &sol.solved_epoch_indices,
        &sol.unplaced_observations,
        &sol.residuals_m,
        sol.ssr_bias_exclusions.len(),
    );
    out.insert(
        "residual_screen_removals".into(),
        json!(sol.residual_screen_removals),
    );
    out.insert("residual_screen".into(), json!(sol.residual_screen));
    out
}

fn fixed_outcome(sol: &FixedSolution) -> Map<String, Value> {
    let mut out = solve_block(
        &sol.position_m,
        sol.status,
        sol.iterations,
        sol.converged,
        &sol.epoch_clocks_m,
        &sol.solved_epoch_indices,
        &sol.unplaced_observations,
        &sol.residuals_m,
        sol.ssr_bias_exclusions.len(),
    );
    out.insert("integer_ratio_bits".into(), hex(sol.integer.integer_ratio));
    out
}

fn float_metadata(sol: &FloatSolution) -> Map<String, Value> {
    metadata(
        &sol.position_covariance,
        &sol.formal_position_covariance,
        &sol.temporal_position_covariance,
        sol.posterior_variance_factor,
        sol.position_covariance_scale_factor,
        sol.temporal_position_covariance_scale_factor,
        &sol.temporal_correlation,
        (sol.tropo_gradient_north_m, sol.tropo_gradient_east_m),
        (
            sol.tropo_gradient_covariance_m2,
            sol.formal_tropo_gradient_covariance_m2,
        ),
        &sol.residual_ionosphere_m,
    )
}

fn fixed_metadata(sol: &FixedSolution) -> Map<String, Value> {
    metadata(
        &sol.position_covariance,
        &sol.formal_position_covariance,
        &sol.temporal_position_covariance,
        sol.posterior_variance_factor,
        sol.position_covariance_scale_factor,
        sol.temporal_position_covariance_scale_factor,
        &sol.temporal_correlation,
        (sol.tropo_gradient_north_m, sol.tropo_gradient_east_m),
        (
            sol.tropo_gradient_covariance_m2,
            sol.formal_tropo_gradient_covariance_m2,
        ),
        &sol.residual_ionosphere_m,
    )
}

/// The paired engine revision locked by this program's Cargo graph.
fn core_revision() -> String {
    core_source_identity::revision_from_lock(include_str!("../Cargo.lock"))
        .unwrap_or_else(|error| panic!("resolve engine source revision: {error}"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let fixture = PathBuf::from(args.next().expect("fixture path"));
    let sp3_dir = PathBuf::from(args.next().expect("SP3 fixture directory"));
    let mut fx: Value =
        serde_json::from_str(&std::fs::read_to_string(&fixture).expect("read fixture"))
            .expect("parse fixture");
    let sp3_file = fx["sp3_file"].as_str().expect("sp3_file").to_string();
    let sp3 = sidereon::load_sp3(&std::fs::read(sp3_dir.join(sp3_file)).expect("read SP3"))
        .expect("parse SP3");
    let epochs = epochs(&fx);
    let initial = state(&fx);

    let float = sidereon::solve_ppp_float(
        &sp3,
        &epochs,
        initial.clone(),
        float_config(&fx, None, false),
    )
    .expect("float solve");
    let fixed = sidereon::solve_ppp_fixed(&sp3, &epochs, float.clone(), fixed_config(&fx))
        .expect("fixed solve");
    let cutoff = sidereon::solve_ppp_float(
        &sp3,
        &epochs,
        initial.clone(),
        float_config(&fx, Some(10.0), false),
    )
    .expect("elevation cutoff solve");
    let gradients = sidereon::solve_ppp_float(
        &sp3,
        &epochs,
        initial.clone(),
        float_config(&fx, None, true),
    )
    .expect("troposphere gradient solve");

    // The ESBC arc is GPS only and its observations state no carriers (the
    // ionosphere-free rows need none), so the wind-up rows take the GPS L1 and
    // L2 carriers, as `tests/test_ppp_range_corrections.py` passes them.
    const GPS_L1_HZ: f64 = 1_575_420_000.0;
    const GPS_L2_HZ: f64 = 1_227_600_000.0;
    let corr_epochs: Vec<PppCorrectionEpoch> = epochs
        .iter()
        .map(|epoch| PppCorrectionEpoch {
            epoch: epoch.epoch,
            t_rx_j2000_s: epoch.t_rx_j2000_s,
            observations: epoch
                .observations
                .iter()
                .map(|obs| {
                    assert_eq!(obs.sat.system, sidereon_core::GnssSystem::Gps);
                    PppCorrectionObservation {
                        sat: obs.sat,
                        freq1_hz: GPS_L1_HZ,
                        freq2_hz: GPS_L2_HZ,
                        glonass_channel: None,
                    }
                })
                .collect(),
        })
        .collect();
    let mut corr_opts = PppCorrectionsOptions::new();
    corr_opts.solid_earth_tide = true;
    corr_opts.phase_windup = true;
    let ppp_corr = build_ppp_corrections(&sp3, &corr_epochs, initial.position_m, &corr_opts)
        .expect("ppp corrections build");
    let lookup = PppCorrectionLookup::from_options(ppp_corr, &corr_opts);
    let range_corrections = RangeCorrections {
        receiver_antenna: None,
        sat_clock_relativity: true,
        satellite_clock: None,
        ppp: lookup,
    };
    let mut float_corr_config = float_config(&fx, None, false);
    float_corr_config.corrections = range_corrections;
    let float_with_corrections =
        sidereon::solve_ppp_float(&sp3, &epochs, initial.clone(), float_corr_config)
            .expect("float solve with corrections");

    // `solve_ppp_auto_init_float(sp3, epochs, config)` and the fixed form,
    // with the options `PppAutoInitOptions()` builds.
    let auto_float = sidereon_core::precise_positioning::solve_ppp_auto_init_float(
        &sp3,
        &epochs,
        PppAutoInitOptions::default(),
        float_config(&fx, None, false),
    )
    .expect("auto-init float solve");
    let auto_fixed = sidereon_core::precise_positioning::solve_ppp_auto_init_fixed(
        &sp3,
        &epochs,
        PppAutoInitOptions::default(),
        float_config(&fx, None, false),
        fixed_config(&fx),
    )
    .expect("auto-init fixed solve");
    // `PppAutoInitOptions(initial_guess_position_m=<auto-init float position>,
    // initial_guess_clock_m=0.0)`.
    let mut guess_options = PppAutoInitOptions::default();
    guess_options.initial_guess = Some(PppInitialGuess {
        position_m: auto_float.position_m,
        clock_m: 0.0,
    });
    guess_options.spp_met = SurfaceMet::default();
    let auto_guess = sidereon_core::precise_positioning::solve_ppp_auto_init_float(
        &sp3,
        &epochs,
        guess_options,
        float_config(&fx, None, false),
    )
    .expect("auto-init explicit-guess solve");

    // The arc with codes that place no transmission epoch: the first
    // observation of epoch 3 reads 0.0 and every observation of epoch 7 reads
    // -1.0, solved with the residual screen on.
    let mut unplaced_epochs = epochs.clone();
    unplaced_epochs[3].observations[0].code_m = 0.0;
    for observation in &mut unplaced_epochs[7].observations {
        observation.code_m = -1.0;
    }
    let mut unplaced_config = float_config(&fx, None, false);
    unplaced_config.residual_screen = true;
    let unplaced =
        sidereon::solve_ppp_float(&sp3, &unplaced_epochs, initial.clone(), unplaced_config)
            .expect("unplaced-code float solve");
    let unplaced_fixed =
        sidereon::solve_ppp_fixed(&sp3, &unplaced_epochs, unplaced.clone(), fixed_config(&fx))
            .expect("unplaced-code fixed solve");

    let expected = fx["expected"].as_object_mut().expect("expected block");
    let mut float_block = float_metadata(&float);
    float_block.extend(float_outcome(&float));
    expected.insert("float_solution".into(), Value::Object(float_block));
    let mut float_corr_block = float_metadata(&float_with_corrections);
    float_corr_block.extend(float_outcome(&float_with_corrections));
    expected.insert(
        "float_with_corrections".into(),
        Value::Object(float_corr_block),
    );
    let mut fixed_block = fixed_metadata(&fixed);
    fixed_block.extend(fixed_outcome(&fixed));
    expected.insert("fixed_solution".into(), Value::Object(fixed_block));
    let mut fixed_float_block = float_metadata(&fixed.float_solution);
    fixed_float_block.extend(float_outcome(&fixed.float_solution));
    expected.insert(
        "fixed_float_solution".into(),
        Value::Object(fixed_float_block),
    );
    expected.insert(
        "float_elevation_cutoff_10_deg".into(),
        json!({
            "position_m": cutoff.position_m,
            "position_bits": cutoff.position_m.iter().copied().map(hex).collect::<Vec<_>>(),
            "used_sat_count": cutoff.used_sats.len(),
        }),
    );
    expected.insert(
        "auto_init_float".into(),
        Value::Object(float_outcome(&auto_float)),
    );
    let mut auto_fixed_block = fixed_outcome(&auto_fixed);
    auto_fixed_block.insert(
        "float_solution".into(),
        Value::Object(float_outcome(&auto_fixed.float_solution)),
    );
    auto_fixed_block.insert(
        "integer_status".into(),
        json!(format!("{:?}", auto_fixed.integer.integer_status)),
    );
    expected.insert("auto_init_fixed".into(), Value::Object(auto_fixed_block));
    expected.insert(
        "auto_init_explicit_guess".into(),
        Value::Object(float_outcome(&auto_guess)),
    );
    let mut unplaced_block = float_outcome(&unplaced);
    unplaced_block.insert(
        "fixed_solution".into(),
        Value::Object(fixed_outcome(&unplaced_fixed)),
    );
    expected.insert("unplaced_code".into(), Value::Object(unplaced_block));
    let mut gradient_block = float_metadata(&gradients);
    gradient_block.extend(float_outcome(&gradients));
    gradient_block.insert("position_m".into(), json!(gradients.position_m));
    expected.insert(
        "float_tropo_gradients".into(),
        Value::Object(gradient_block),
    );

    fx.as_object_mut()
        .expect("fixture object")
        .insert("core_revision".into(), json!(core_revision()));
    std::fs::write(
        &fixture,
        serde_json::to_string_pretty(&fx).expect("serialize fixture"),
    )
    .expect("write fixture");
    eprintln!(
        "wrote the binding-level PPP solutions to {}",
        fixture.display()
    );
}
