//! Writes `tests/fixtures/core_goldens.json`: the results of core computations
//! that binding tests compare with bit for bit. Each section builds its inputs
//! as the Python test builds them, through the same core entry points the
//! binding calls, so a test that feeds the same inputs through the binding has
//! to reproduce the section exactly.
//!
//! Run from this directory after an engine change:
//!
//!     cargo run --release --locked -- <core-fixtures> ../../tests/fixtures
//!
//! `<core-fixtures>` is `crates/sidereon-core/tests/fixtures` of the core the
//! binding builds against. The program reads `scenario_base.json` from the
//! binding fixtures and writes `core_goldens.json` next to it.

#[path = "../../core_source_identity.rs"]
mod core_source_identity;
mod diagnostics;
mod field;
mod parity_v3;
mod sp3_merge;

use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde_json::{json, Value};
use sidereon_core::astro::sgp4::{
    fit_tle, FitConfig, FitEpoch, FitSample, JulianDate, Satellite, TleMetadata,
};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::frequencies::rinex_band_frequency_hz;
use sidereon_core::observables::{
    predict, pseudorange_transmit_epoch_j2000_s, pseudorange_transmit_satellite_state,
    PredictOptions,
};
use sidereon_core::positioning::{
    solve_static, solve_with_doppler_velocity, solve_with_policy, Corrections, DopplerObservation,
    KlobucharCoeffs, Observation, ReceiverSolution, RobustConfig, SolveInputs, SolvePolicy,
    StaticEpoch, StaticSolveOptions, SurfaceMet,
};
use sidereon_core::ppp_corrections::{
    build as build_ppp_corrections, CivilDateTime, PppCorrectionEpoch, PppCorrectionObservation,
    PppCorrectionsOptions, SatelliteAntenna, SatelliteAntennaFrequency, SatelliteAntennaOptions,
};
use sidereon_core::quality::SolutionValidationOptions;
use sidereon_core::scenario::{simulate_scenario, Scenario};
use sidereon_core::velocity::{
    range_rate_to_doppler, solve as solve_velocity, VelocityObservable, VelocityObservation,
    VelocitySolution, VelocitySolveOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const C_M_S: f64 = 299_792_458.0;
const F_L1_HZ: f64 = 1_575_420_000.0;
const F_L2_HZ: f64 = 1_227_600_000.0;

/// The satellites of the SPP trace fixture that the binding's QC and
/// robustness tests solve on.
const CONSISTENT_SATS: [&str; 8] = ["G08", "G10", "G16", "G18", "G20", "G21", "G26", "G27"];

/// The velocity scenario of the core `velocity` module test.
const VEL_T_RX_J2000_S: f64 = 646_272_000.0;
const VEL_RECEIVER: [f64; 3] = [4_500_000.0, 500_000.0, 4_500_000.0];
const VEL_TRUE: [f64; 3] = [12.0, -7.0, 3.0];
const VEL_DRIFT_TRUE: f64 = 1.0e-9;
const VEL_SP3: &str = "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3";

fn hex(value: f64) -> Value {
    Value::String(format!("{:#018x}", value.to_bits()))
}

fn hexes(values: &[f64]) -> Value {
    Value::Array(values.iter().copied().map(hex).collect())
}

fn from_hex(value: &Value) -> f64 {
    let text = value.as_str().expect("hex bit string");
    f64::from_bits(u64::from_str_radix(text.trim_start_matches("0x"), 16).expect("hex bits"))
}

fn load_sp3(core: &Path, name: &str) -> Sp3 {
    Sp3::parse(&std::fs::read(core.join("sp3").join(name)).expect("read SP3")).expect("parse SP3")
}

/// The SPP trace fixture's inputs.
struct Trace {
    doc: Value,
}

impl Trace {
    fn inputs(&self) -> &Value {
        &self.doc["fixture"]["inputs"]
    }

    fn observations(&self) -> Vec<(String, f64)> {
        self.inputs()["observations"]
            .as_array()
            .expect("observations")
            .iter()
            .map(|obs| {
                (
                    obs["sat_id"].as_str().expect("sat").to_string(),
                    from_hex(&obs["p_meas_m"]),
                )
            })
            .collect()
    }

    fn four(&self, value: &Value) -> [f64; 4] {
        let values: Vec<f64> = value
            .as_array()
            .expect("array")
            .iter()
            .map(from_hex)
            .collect();
        [values[0], values[1], values[2], values[3]]
    }

    /// `SppConfig(...)` of the binding tests: no ionosphere, no troposphere,
    /// the frozen initial guess, and the trace's Klobuchar and met values.
    fn inputs_for(&self, observations: &[(String, f64)], met: bool) -> SolveInputs {
        let inputs = self.inputs();
        let mut solve = SolveInputs::default();
        solve.observations = observations
            .iter()
            .map(|(sat, pseudorange_m)| Observation {
                satellite_id: GnssSatelliteId::from_str(sat).expect("satellite"),
                pseudorange_m: *pseudorange_m,
            })
            .collect();
        solve.t_rx_j2000_s = from_hex(&inputs["t_rx_j2000_s"]);
        solve.t_rx_second_of_day_s = from_hex(&inputs["t_rx_sod_s"]);
        solve.day_of_year = from_hex(&inputs["doy"]);
        solve.initial_guess = self.four(&self.doc["fixture"]["frozen"]["initial_guess_x0"]);
        solve.corrections = Corrections {
            ionosphere: false,
            troposphere: false,
        };
        solve.klobuchar = KlobucharCoeffs {
            alpha: self.four(&inputs["klobuchar_alpha"]),
            beta: self.four(&inputs["klobuchar_beta"]),
        };
        solve.met = if met {
            SurfaceMet {
                pressure_hpa: from_hex(&inputs["met"]["pressure_hpa"]),
                temperature_k: from_hex(&inputs["met"]["temperature_k"]),
                relative_humidity: from_hex(&inputs["met"]["relative_humidity"]),
            }
        } else {
            SurfaceMet::default()
        };
        solve
    }
}

/// The policy `solve_spp(..., max_pdop=None, coarse_search_seeds=None)` builds.
fn default_policy() -> SolvePolicy {
    SolvePolicy {
        validation: SolutionValidationOptions::default(),
        coarse_search_seeds: None,
    }
}

fn spp(sp3: &Sp3, inputs: &SolveInputs) -> ReceiverSolution {
    solve_with_policy(sp3, inputs, true, default_policy()).expect("SPP solve")
}

fn position(solution: &ReceiverSolution) -> [f64; 3] {
    [
        solution.position.x_m,
        solution.position.y_m,
        solution.position.z_m,
    ]
}

/// The default SPP solve of the trace inputs, as `tests/test_spp.py` makes it.
fn trace_default(sp3: &Sp3, trace: &Trace) -> Value {
    let solution = spp(sp3, &trace.inputs_for(&trace.observations(), true));
    json!({
        "position_m": hexes(&position(&solution)),
        "rx_clock_s": hex(solution.rx_clock_s),
        "used_sats": solution.used_sats.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "status": format!("{:?}", solution.metadata.status),
        "converged": solution.metadata.converged,
        "iterations": solution.metadata.iterations,
    })
}

/// Pseudoranges of the consistent trace satellites that the default model
/// fits to within one unit in the last place of a pseudorange: each round
/// subtracts the residuals of the solve from the pseudoranges and stops when
/// the largest residual no longer falls. For these pseudoranges, between 2^24
/// and 2^25 m, one unit in the last place is 2^-28 m (3.7e-9 m), and the rounds
/// end there. The file records the largest residual it reached as
/// `max_residual_m`. The solution then sits where the corrected pseudoranges
/// place it.
fn consistent(sp3: &Sp3, trace: &Trace) -> Value {
    let all: std::collections::BTreeMap<String, f64> = trace.observations().into_iter().collect();
    let mut observations: Vec<(String, f64)> = CONSISTENT_SATS
        .iter()
        .map(|sat| (sat.to_string(), all[*sat]))
        .collect();
    // Each round subtracts the post-fit residuals; the rounds stop when the
    // largest residual no longer falls, which is where the solver's own
    // convergence tolerance leaves it.
    let mut best: Option<(f64, Vec<(String, f64)>, ReceiverSolution)> = None;
    for round in 0..30 {
        let solution = spp(sp3, &trace.inputs_for(&observations, false));
        let max_residual = solution
            .residuals_m
            .iter()
            .fold(0.0_f64, |acc, residual| acc.max(residual.abs()));
        eprintln!("consistent round {round}: largest residual {max_residual:e} m");
        if let Some((previous, _, _)) = &best {
            if max_residual >= *previous {
                break;
            }
        }
        let next: Vec<(String, f64)> = observations
            .iter()
            .map(|(sat, p)| {
                let residual = solution
                    .used_sats
                    .iter()
                    .zip(&solution.residuals_m)
                    .find(|(used, _)| used.to_string() == *sat)
                    .map(|(_, r)| *r)
                    .expect("used satellite");
                (sat.clone(), p - residual)
            })
            .collect();
        best = Some((max_residual, observations.clone(), solution));
        observations = next;
    }
    let (max_residual, observations, solution) = best.expect("at least one round");
    json!({
        "satellites": observations.iter().map(|(sat, _)| sat.clone()).collect::<Vec<_>>(),
        "pseudoranges_m": hexes(&observations.iter().map(|(_, p)| *p).collect::<Vec<_>>()),
        "position_m": hexes(&position(&solution)),
        "rx_clock_s": hex(solution.rx_clock_s),
        "max_residual_m": max_residual,
    })
}

/// The bias `tests/test_spp_robustness.py` adds to one consistent pseudorange.
const ROBUST_BIAS_M: f64 = 300.0;

/// The robust-scenario inputs of `tests/test_spp_robustness.py`: the
/// consistent pseudoranges with `ROBUST_BIAS_M` added to `faulted`, solved
/// statically or with the given Huber configuration.
fn robust_scenario_inputs(
    trace: &Trace,
    consistent: &[(String, f64)],
    faulted: &str,
    robust: Option<RobustConfig>,
) -> SolveInputs {
    let observations: Vec<(String, f64)> = consistent
        .iter()
        .map(|(sat, p)| {
            let bias = if sat == faulted { ROBUST_BIAS_M } else { 0.0 };
            (sat.clone(), p + bias)
        })
        .collect();
    let mut inputs = trace.inputs_for(&observations, false);
    inputs.robust = robust;
    inputs
}

fn own_residual(solution: &ReceiverSolution, satellite: &str) -> f64 {
    solution
        .used_sats
        .iter()
        .zip(&solution.residuals_m)
        .find(|(used, _)| used.to_string() == satellite)
        .map(|(_, residual)| *residual)
        .expect("faulted satellite is used")
}

fn distance_m(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Per-satellite redundancy of the robust scenario, and the default robust
/// solve with the bias on G08.
///
/// A satellite's redundancy is the share of a bias on its pseudorange that the
/// static variance-weighted solve leaves in that satellite's own residual,
/// `(r_j(biased) - r_j(clean)) / bias`, with the bias of the robust test. It is
/// the diagonal of `I - H (H^T W H)^-1 H^T W` at this geometry and these
/// weights. The rest of the bias moves the position. A Huber reweighting of raw
/// residuals can only down-weight a biased satellite whose own residual
/// stands out, so a satellite with a small redundancy hides its bias in the
/// other satellites' residuals, and the reweighting down-weights them instead.
fn robust_faults(sp3: &Sp3, trace: &Trace, consistent: &Value) -> Value {
    let observations: Vec<(String, f64)> = consistent["satellites"]
        .as_array()
        .expect("consistent satellites")
        .iter()
        .zip(
            consistent["pseudoranges_m"]
                .as_array()
                .expect("consistent pseudoranges"),
        )
        .map(|(sat, p)| (sat.as_str().expect("satellite").to_string(), from_hex(p)))
        .collect();
    let truth: Vec<f64> = consistent["position_m"]
        .as_array()
        .expect("consistent position")
        .iter()
        .map(from_hex)
        .collect();
    let truth = [truth[0], truth[1], truth[2]];
    let clean = spp(sp3, &trace.inputs_for(&observations, false));

    let mut redundancy = serde_json::Map::new();
    for (satellite, _) in &observations {
        let biased = spp(
            sp3,
            &robust_scenario_inputs(trace, &observations, satellite, None),
        );
        let share =
            (own_residual(&biased, satellite) - own_residual(&clean, satellite)) / ROBUST_BIAS_M;
        redundancy.insert(satellite.clone(), json!(share));
        let static_error = distance_m(position(&biased), truth);
        let mut report = format!(
            "robust faults {satellite}: redundancy {share:.6} static error {static_error:.6} m"
        );
        for max_outer in [RobustConfig::default().max_outer, 50] {
            let mut config = RobustConfig::default();
            config.max_outer = max_outer;
            let robust = spp(
                sp3,
                &robust_scenario_inputs(trace, &observations, satellite, Some(config)),
            );
            report.push_str(&format!(
                "; max_outer {max_outer}: robust error {:.6} m status {:?} converged {}",
                distance_m(position(&robust), truth),
                robust.metadata.status,
                robust.metadata.converged
            ));
        }
        eprintln!("{report}");
    }

    let g08 = spp(
        sp3,
        &robust_scenario_inputs(trace, &observations, "G08", Some(RobustConfig::default())),
    );
    json!({
        "bias_m": ROBUST_BIAS_M,
        "redundancy": Value::Object(redundancy),
        "g08_robust_default": {
            "position_m": hexes(&position(&g08)),
            "rx_clock_s": hex(g08.rx_clock_s),
            "used_sats": g08.used_sats.iter().map(ToString::to_string).collect::<Vec<_>>(),
            "residuals_m": hexes(&g08.residuals_m),
            "status": format!("{:?}", g08.metadata.status),
            "converged": g08.metadata.converged,
        },
    })
}

/// `tests/test_017_domain_exposure.py::test_static_positioning_solution_bits`:
/// three epochs of the trace offset by 0, 0.2 and -0.1 m.
fn static_three_epochs(sp3: &Sp3, trace: &Trace) -> Value {
    let epochs: Vec<StaticEpoch> = [0.0, 0.2, -0.1]
        .iter()
        .map(|offset| {
            let observations: Vec<(String, f64)> = trace
                .observations()
                .into_iter()
                .map(|(sat, p)| (sat, p + offset))
                .collect();
            let mut epoch = StaticEpoch::from_solve_inputs(trace.inputs_for(&observations, true));
            epoch.weights = None;
            epoch
        })
        .collect();
    let guess = trace.four(&trace.doc["fixture"]["frozen"]["initial_guess_x0"]);
    let mut options = StaticSolveOptions::default();
    options.initial_position_m = [guess[0], guess[1], guess[2]];
    options.with_geodetic = true;
    options.robust = None;
    let solution = solve_static(sp3, &epochs, options).expect("static solve");
    let covariance: Vec<f64> = solution
        .covariance
        .position_ecef_m2
        .iter()
        .flatten()
        .copied()
        .collect();
    json!({
        "position_m": hexes(&[solution.position.x_m, solution.position.y_m, solution.position.z_m]),
        "per_epoch_clock": solution
            .per_epoch_clock
            .iter()
            .map(|clock| json!([clock.epoch_index, clock.system.as_str(), hex(clock.clock_s)]))
            .collect::<Vec<_>>(),
        "position_covariance_ecef_m2": hexes(&covariance),
        "residual_rms_m": hex(solution.residual_rms_m()),
        "used_measurements": solution.metadata.used_measurements,
        "n_parameters": solution.metadata.n_parameters,
        "redundancy": solution.metadata.redundancy,
        "converged": solution.metadata.converged,
        "status": format!("{:?}", solution.metadata.status),
        "used_sat_counts": solution.used_sats.iter().map(Vec::len).collect::<Vec<_>>(),
        "residual_count": solution.residuals_m.len(),
    })
}

fn velocity_block(solution: &VelocitySolution) -> Value {
    let state: Vec<f64> = solution
        .state_covariance
        .iter()
        .flatten()
        .copied()
        .collect();
    json!({
        "used_sats": solution.used_sats.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "velocity_m_s": hexes(&solution.velocity_m_s),
        "speed_m_s": hex(solution.speed_m_s),
        "clock_drift_s_s": hex(solution.clock_drift_s_s),
        "residuals_m_s": hexes(&solution.residuals_m_s.iter().map(|(_, r)| *r).collect::<Vec<_>>()),
        "state_covariance": hexes(&state),
    })
}

/// The core `velocity` module scenario: range rates synthesized with the
/// core's `predict` from a true velocity and clock drift, and the range-rate
/// and GLONASS-channel Doppler solves of them.
fn velocity(core: &Path) -> Value {
    let sp3 = load_sp3(core, VEL_SP3);
    let mut planning = PredictOptions::default();
    planning.light_time = false;
    let satellites: Vec<GnssSatelliteId> = sp3
        .satellites()
        .iter()
        .copied()
        .filter(|sat| sat.system == GnssSystem::Gps)
        .filter(|sat| {
            predict(&sp3, *sat, VEL_RECEIVER, VEL_T_RX_J2000_S, planning)
                .map(|obs| obs.elevation_deg >= 5.0)
                .unwrap_or(false)
        })
        .collect();
    let range_rates: Vec<VelocityObservation> = satellites
        .iter()
        .map(|&sat| {
            let obs = predict(
                &sp3,
                sat,
                VEL_RECEIVER,
                VEL_T_RX_J2000_S,
                PredictOptions::default(),
            )
            .expect("predict");
            let along = obs.los_unit[0] * VEL_TRUE[0]
                + obs.los_unit[1] * VEL_TRUE[1]
                + obs.los_unit[2] * VEL_TRUE[2];
            VelocityObservation {
                satellite_id: sat,
                value: obs.range_rate_m_s - along + C_M_S * VEL_DRIFT_TRUE,
                carrier_hz: F_L1_HZ,
                sat_clock_drift_s_s: 0.0,
            }
        })
        .collect();
    let range_rate_solution = solve_velocity(
        &sp3,
        &range_rates,
        VEL_RECEIVER,
        VEL_T_RX_J2000_S,
        VelocitySolveOptions::default(),
    )
    .expect("range-rate solve");
    let dopplers: Vec<VelocityObservation> = range_rates
        .iter()
        .enumerate()
        .map(|(index, obs)| {
            let channel = (index % 14) as i8 - 7;
            let carrier_hz = rinex_band_frequency_hz(GnssSystem::Glonass, '1', Some(channel))
                .expect("GLONASS G1 carrier");
            VelocityObservation {
                satellite_id: obs.satellite_id,
                value: range_rate_to_doppler(obs.value, carrier_hz).expect("Doppler"),
                carrier_hz,
                sat_clock_drift_s_s: 0.0,
            }
        })
        .collect();
    let mut doppler_options = VelocitySolveOptions::default();
    doppler_options.observable = VelocityObservable::Doppler;
    let doppler_solution = solve_velocity(
        &sp3,
        &dopplers,
        VEL_RECEIVER,
        VEL_T_RX_J2000_S,
        doppler_options,
    )
    .expect("Doppler solve");
    json!({
        "satellites": satellites.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "range_rate_m_s": hexes(&range_rates.iter().map(|obs| obs.value).collect::<Vec<_>>()),
        "range_rate_solution": velocity_block(&range_rate_solution),
        "doppler_solution": velocity_block(&doppler_solution),
    })
}

/// `tests/test_017_domain_exposure.py::test_velocity_covariance_and_spp_doppler_bits`:
/// the default SPP solve of the trace, L1 Doppler predicted at its position for
/// each used satellite with light time and Sagnac, and the combined SPP and
/// Doppler velocity solve of them.
fn spp_doppler(sp3: &Sp3, trace: &Trace) -> Value {
    let inputs = trace.inputs_for(&trace.observations(), true);
    let receiver = spp(sp3, &inputs);
    let receiver_ecef_m = position(&receiver);
    let mut options = PredictOptions::default();
    options.carrier_hz = F_L1_HZ;
    options.light_time = true;
    options.sagnac = true;
    let dopplers: Vec<DopplerObservation> = receiver
        .used_sats
        .iter()
        .map(|&sat| DopplerObservation {
            satellite_id: sat,
            doppler_hz: predict(sp3, sat, receiver_ecef_m, inputs.t_rx_j2000_s, options)
                .expect("predict")
                .doppler_hz,
            carrier_hz: F_L1_HZ,
            sat_clock_drift_s_s: 0.0,
        })
        .collect();
    let combined =
        solve_with_doppler_velocity(sp3, &inputs, &dopplers, true).expect("SPP Doppler solve");
    assert!(combined.velocity_error.is_none(), "velocity solve failed");
    let velocity = combined.velocity.expect("velocity");
    json!({
        "used_sats": receiver.used_sats.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "rx_clock_drift_s_s": combined.receiver.rx_clock_drift_s_s.map(hex),
        "velocity_m_s": hexes(&velocity.velocity_m_s),
        "clock_drift_s_s": hex(velocity.clock_drift_s_s),
        "velocity_used_sats": velocity.used_sats.iter().map(ToString::to_string).collect::<Vec<_>>(),
    })
}

/// What moves the SPP Doppler velocity of [`spp_doppler`] off zero, from
/// solves that each change one thing, all at the SPP position of the trace:
///
/// - `unplaced_sagnac_on`: Dopplers from `predict` with light time and Sagnac,
///   solved by `velocity::solve` with the same light time and Sagnac, which
///   predicts each row as `predict` does. The model on both sides is the same.
/// - `unplaced_sagnac_off`: the same with Sagnac off on both sides.
/// - `placement_sagnac_off`: Dopplers whose range rate is the line-of-sight
///   rate of each satellite placed at the transmission epoch of its
///   pseudorange (the state the placed solve reads), with no Sagnac term,
///   solved by `velocity::solve` with Sagnac off. Each row differs from the
///   solve's own prediction only by where the satellite is placed.
///
/// The combined solve reads the placed, unrotated state with the rate of the
/// first-order Sagnac term, where `predict` rotates the state into the
/// reception frame; `spp_doppler` holds its velocity.
fn spp_doppler_terms(sp3: &Sp3, trace: &Trace) -> Value {
    let inputs = trace.inputs_for(&trace.observations(), true);
    let receiver = spp(sp3, &inputs);
    let receiver_ecef_m = position(&receiver);
    let t_rx = inputs.t_rx_j2000_s;
    let pseudoranges: std::collections::BTreeMap<GnssSatelliteId, f64> = inputs
        .observations
        .iter()
        .map(|obs| (obs.satellite_id, obs.pseudorange_m))
        .collect();
    let predicted = |sagnac: bool| -> Vec<(
        GnssSatelliteId,
        sidereon_core::observables::PredictedObservables,
    )> {
        let mut options = PredictOptions::default();
        options.carrier_hz = F_L1_HZ;
        options.light_time = true;
        options.sagnac = sagnac;
        receiver
            .used_sats
            .iter()
            .map(|&sat| {
                (
                    sat,
                    predict(sp3, sat, receiver_ecef_m, t_rx, options).expect("predict"),
                )
            })
            .collect()
    };
    let solve = |dopplers: Vec<VelocityObservation>, sagnac: bool| -> VelocitySolution {
        let mut options = VelocitySolveOptions::default();
        options.observable = VelocityObservable::Doppler;
        options.light_time = true;
        options.sagnac = sagnac;
        solve_velocity(sp3, &dopplers, receiver_ecef_m, t_rx, options).expect("velocity solve")
    };
    let doppler_rows = |rows: &[(GnssSatelliteId, f64)]| -> Vec<VelocityObservation> {
        rows.iter()
            .map(|&(sat, doppler_hz)| VelocityObservation {
                satellite_id: sat,
                value: doppler_hz,
                carrier_hz: F_L1_HZ,
                sat_clock_drift_s_s: 0.0,
            })
            .collect()
    };
    let block = |solution: &VelocitySolution| -> Value {
        json!({
            "velocity_m_s": hexes(&solution.velocity_m_s),
            "speed_m_s": solution.speed_m_s,
            "clock_drift_s_s": hex(solution.clock_drift_s_s),
        })
    };

    let with_sagnac = predicted(true);
    let unplaced_on = solve(
        doppler_rows(
            &with_sagnac
                .iter()
                .map(|(sat, p)| (*sat, p.doppler_hz))
                .collect::<Vec<_>>(),
        ),
        true,
    );
    let without_sagnac = predicted(false);
    let unplaced_off = solve(
        doppler_rows(
            &without_sagnac
                .iter()
                .map(|(sat, p)| (*sat, p.doppler_hz))
                .collect::<Vec<_>>(),
        ),
        false,
    );
    // The placed rows keep whatever `predict` adds to the geometric range rate
    // and change only the line-of-sight rate: Doppler = -range rate * f / c.
    let mut placement_rate_change_m_s = Vec::new();
    let placed_rows: Vec<(GnssSatelliteId, f64)> = without_sagnac
        .iter()
        .map(|(sat, p)| {
            let t_tx = pseudorange_transmit_epoch_j2000_s(sp3, *sat, t_rx, pseudoranges[sat])
                .expect("transmit epoch");
            let state =
                pseudorange_transmit_satellite_state(sp3, *sat, receiver_ecef_m, t_rx, t_tx, false)
                    .expect("placed state");
            let placed_rate = state.los_unit[0] * state.velocity_m_s[0]
                + state.los_unit[1] * state.velocity_m_s[1]
                + state.los_unit[2] * state.velocity_m_s[2];
            let predicted_rate = p.los_unit[0] * p.sat_velocity_m_s[0]
                + p.los_unit[1] * p.sat_velocity_m_s[1]
                + p.los_unit[2] * p.sat_velocity_m_s[2];
            let change = placed_rate - predicted_rate;
            placement_rate_change_m_s.push(change);
            (*sat, p.doppler_hz - change * F_L1_HZ / C_M_S)
        })
        .collect();
    let placement_off = solve(doppler_rows(&placed_rows), false);
    json!({
        "unplaced_sagnac_on": block(&unplaced_on),
        "unplaced_sagnac_off": block(&unplaced_off),
        "placement_sagnac_off": block(&placement_off),
        "placement_rate_change_m_s": placement_rate_change_m_s,
    })
}

fn civil(year: i32, month: u8, day: u8) -> CivilDateTime {
    CivilDateTime {
        year,
        month,
        day,
        hour: 0,
        minute: 0,
        second: 0.0,
    }
}

/// `tests/test_ppp_corrections.py::test_ppp_corrections_are_bit_exact`.
fn ppp_corrections(core: &Path) -> Value {
    let sp3 = load_sp3(core, VEL_SP3);
    let sat = GnssSatelliteId::from_str("G21").expect("satellite");
    let epoch = PppCorrectionEpoch {
        epoch: CivilDateTime {
            year: 2020,
            month: 6,
            day: 24,
            hour: 12,
            minute: 0,
            second: 0.0,
        },
        t_rx_j2000_s: (2_459_025.0 - 2_451_545.0) * 86_400.0,
        observations: vec![PppCorrectionObservation {
            sat,
            freq1_hz: F_L1_HZ,
            freq2_hz: F_L2_HZ,
            glonass_channel: None,
        }],
    };
    let antennas = vec![SatelliteAntenna {
        sat,
        valid_from: Some(civil(2020, 1, 1)),
        valid_until: Some(civil(2021, 1, 1)),
        frequencies: vec![
            SatelliteAntennaFrequency {
                label: "G01".to_string(),
                pco_m: [0.1, -0.2, 1.0],
                noazi_pcv_m: vec![(0.0, 0.001), (5.0, 0.002), (10.0, 0.004)],
            },
            SatelliteAntennaFrequency {
                label: "G02".to_string(),
                pco_m: [-0.1, 0.3, 0.5],
                noazi_pcv_m: vec![(0.0, -0.001), (5.0, -0.002), (10.0, -0.003)],
            },
        ],
    }];
    let mut antenna = SatelliteAntennaOptions::new(
        "G01".to_string(),
        F_L1_HZ,
        "G02".to_string(),
        F_L2_HZ,
        antennas.clone(),
    );
    antenna.antennas = antennas;
    let mut options = PppCorrectionsOptions::new();
    options.solid_earth_tide = true;
    options.phase_windup = true;
    options.satellite_antenna = Some(antenna);
    let corrections = build_ppp_corrections(
        &sp3,
        &[epoch],
        [3_512_900.0, 780_500.0, 5_248_700.0],
        &options,
    )
    .expect("PPP corrections");
    json!({
        "tide": hexes(&corrections.tide[0].vector_m),
        "windup_m": hex(corrections.windup_m[0].value_m),
        "sat_pco_ecef": hexes(&corrections.sat_pco_ecef[0].vector_m),
        "sat_pcv_m": hex(corrections.sat_pcv_m[0].value_m),
    })
}

/// `tests/test_018_domain_exposure.py`'s base scenario, read from
/// `scenario_base.json`.
fn scenario(fixtures: &Path) -> Value {
    let text = std::fs::read_to_string(fixtures.join("scenario_base.json")).expect("scenario");
    let scenario: Scenario = serde_json::from_str(&text).expect("parse scenario");
    let output = simulate_scenario(&scenario).expect("simulate scenario");
    let bytes = serde_json::to_vec(&output).expect("serialize output");
    let doc: Value = serde_json::from_slice(&bytes).expect("reparse output");
    let engine_version = doc["engine_version"].as_str().expect("engine version");
    let version = engine_version.split(':').next().expect("version");
    let observations = &doc["observations"];
    let first_three = |key: &str| -> Value {
        hexes(
            &observations[key]
                .as_array()
                .expect("observable array")
                .iter()
                .take(3)
                .map(|v| v.as_f64().expect("number"))
                .collect::<Vec<_>>(),
        )
    };
    json!({
        "bytes_without_version": bytes.len() - version.len(),
        "pseudorange_m": first_three("pseudorange_m"),
        "doppler_hz": first_three("doppler_hz"),
    })
}

fn tle_fit_parity() -> Value {
    const LINE1: &str = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
    const LINE2: &str = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";
    const BASE_UNIX_US: i64 = 1_530_645_957_304_128;
    const OFFSETS_S: [i64; 6] = [-1800, 0, 1800, 3600, 5400, 7200];
    const UNIX_US_PER_SECOND: i64 = 1_000_000;
    const UNIX_US_PER_DAY: i64 = 86_400 * UNIX_US_PER_SECOND;

    let satellite = Satellite::from_tle(LINE1, LINE2).expect("parse ISS TLE");
    let samples: Vec<FitSample> = OFFSETS_S
        .iter()
        .map(|offset_s| {
            let unix_us = BASE_UNIX_US + offset_s * UNIX_US_PER_SECOND;
            let jd = JulianDate::from_unix_microseconds(unix_us);
            let prediction = satellite.propagate_jd(jd).expect("propagate TLE sample");

            let unix_day = unix_us.div_euclid(UNIX_US_PER_DAY);
            let day_us = unix_us.rem_euclid(UNIX_US_PER_DAY);
            let seconds_of_day = (day_us / UNIX_US_PER_SECOND) as f64
                + (day_us % UNIX_US_PER_SECOND) as f64 / UNIX_US_PER_SECOND as f64;
            let jd_value = 2_440_587.5 + unix_day as f64 + seconds_of_day / 86_400.0;
            let jd_whole = jd_value.floor();

            FitSample {
                epoch: JulianDate(jd_whole, jd_value - jd_whole),
                position_teme_km: prediction.position,
                velocity_teme_km_s: Some(prediction.velocity),
            }
        })
        .collect();

    let mut config = FitConfig::default();
    config.epoch = FitEpoch::Sample(1);
    config.fit_bstar = false;
    config.use_velocity = true;
    config.metadata = TleMetadata {
        catalog_number: 25544,
        international_designator: "98067A".to_owned(),
        object_name: "ISS".to_owned(),
        ..TleMetadata::default()
    };
    let fit = fit_tle(&samples, &config).expect("fit TLE");
    json!({
        "line1": fit.line1,
        "line2": fit.line2,
        "stats": {
            "rms_position_km": hex(fit.stats.rms_position_km),
            "max_position_km": hex(fit.stats.max_position_km),
            "rms_position_axes_km": hexes(&fit.stats.rms_position_axes_km),
            "rms_velocity_km_s": fit.stats.rms_velocity_km_s.map(hex),
            "tle_rms_position_km": hex(fit.stats.tle_rms_position_km),
            "status": fit.stats.status,
            "nfev": fit.stats.nfev,
            "njev": fit.stats.njev,
            "cost": hex(fit.stats.cost),
            "optimality": hex(fit.stats.optimality),
            "bstar_observable": fit.stats.bstar_observable,
            "seed_refine_passes": fit.stats.seed_refine_passes,
        }
    })
}

/// The paired engine revision locked by this program's Cargo graph.
fn core_revision() -> String {
    core_source_identity::revision_from_lock(include_str!("../Cargo.lock"))
        .unwrap_or_else(|error| panic!("resolve engine source revision: {error}"))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let core = PathBuf::from(args.next().expect("core fixtures directory"));
    let fixtures = PathBuf::from(args.next().expect("binding fixtures directory"));
    let trace = Trace {
        doc: serde_json::from_str(
            &std::fs::read_to_string(core.join("spp_trace_L0_minimal.json")).expect("trace"),
        )
        .expect("parse trace"),
    };
    let sp3 = load_sp3(
        &core,
        trace.inputs()["sp3_file"].as_str().expect("sp3_file"),
    );
    let spp_consistent = consistent(&sp3, &trace);
    let spp_robust_faults = robust_faults(&sp3, &trace, &spp_consistent);
    let doc = json!({
        "source": "scripts/core_goldens",
        "core_revision": core_revision(),
        "spp_trace_default": trace_default(&sp3, &trace),
        "spp_consistent": spp_consistent,
        "spp_robust_faults": spp_robust_faults,
        "static_three_epochs": static_three_epochs(&sp3, &trace),
        "velocity": velocity(&core),
        "spp_doppler": spp_doppler(&sp3, &trace),
        "spp_doppler_terms": spp_doppler_terms(&sp3, &trace),
        "tle_fit": tle_fit_parity(),
        "ppp_corrections": ppp_corrections(&core),
        "scenario": scenario(&fixtures),
    });
    let mut doc = doc;
    let object = doc.as_object_mut().expect("goldens object");
    for (key, value) in field::sections(&core) {
        assert!(
            object.insert(key.to_string(), value).is_none(),
            "section {key} written twice"
        );
    }
    for (key, value) in diagnostics::sections(&fixtures) {
        assert!(
            object.insert(key.to_string(), value).is_none(),
            "section {key} written twice"
        );
    }
    for (key, value) in parity_v3::sections(&core, &fixtures) {
        assert!(
            object.insert(key.to_string(), value).is_none(),
            "section {key} written twice"
        );
    }
    for (key, value) in sp3_merge::sections() {
        assert!(
            object.insert(key.to_string(), value).is_none(),
            "section {key} written twice"
        );
    }
    let out = fixtures.join("core_goldens.json");
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&doc).expect("serialize") + "\n",
    )
    .expect("write goldens");
    eprintln!("wrote {}", out.display());
}
