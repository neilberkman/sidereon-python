//! Direct core baselines for the binding's WTZR/WTZZ RTK fixture.
use serde_json::{json, Value};
use sidereon_core::rtk::BaselineReferenceSelection;
use sidereon_core::rtk_filter::*;
use std::collections::BTreeMap;
use std::path::Path;

fn vec3(v: &Value) -> [f64; 3] {
    std::array::from_fn(|i| v[i].as_f64().unwrap())
}
fn map_f64(v: &Value) -> BTreeMap<String, f64> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_f64().unwrap()))
        .collect()
}
fn map_str(v: &Value) -> BTreeMap<String, String> {
    v.as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_owned()))
        .collect()
}
fn h(v: &[f64]) -> Vec<String> {
    v.iter().map(|x| format!("{:#018x}", x.to_bits())).collect()
}
fn sat(v: &Value) -> SatMeas {
    SatMeas {
        sat: v["sat"].as_str().unwrap().into(),
        sd_ambiguity_id: v["sd_ambiguity_id"].as_str().unwrap().into(),
        base_code_m: v["base_code_m"].as_f64().unwrap(),
        base_phase_m: v["base_phase_m"].as_f64().unwrap(),
        rover_code_m: v["rover_code_m"].as_f64().unwrap(),
        rover_phase_m: v["rover_phase_m"].as_f64().unwrap(),
        base_tx_pos: vec3(&v["base_tx_pos"]),
        rover_tx_pos: vec3(&v["rover_tx_pos"]),
        pos: vec3(&v["pos"]),
    }
}
fn rows(v: &Value, key: &str) -> Vec<SatMeas> {
    v[key].as_array().unwrap().iter().map(sat).collect()
}
fn arc_obs(r: &SatMeas, rover: bool) -> RtkArcObservation {
    RtkArcObservation {
        satellite_id: r.sat.clone(),
        ambiguity_id: r.sd_ambiguity_id.clone(),
        code_m: if rover { r.rover_code_m } else { r.base_code_m },
        phase_m: if rover {
            r.rover_phase_m
        } else {
            r.base_phase_m
        },
        lli: None,
    }
}
pub fn generate(fixtures: &Path) -> Value {
    let f: Value =
        serde_json::from_slice(&std::fs::read(fixtures.join("rtk_wtzr.json")).unwrap()).unwrap();
    let epochs: Vec<Epoch> = f["epochs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| Epoch {
            references: rows(e, "references"),
            nonref: rows(e, "nonref"),
            dt_s: e["dt_s"].as_f64().unwrap(),
            velocity_mps: e["velocity_mps"]
                .as_array()
                .map(|_| vec3(&e["velocity_mps"])),
        })
        .collect();
    let base = vec3(&f["base_arp_m"]);
    let initial = vec3(&f["initial_baseline_m"]);
    let ids: Vec<String> = f["ambiguity_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_owned())
        .collect();
    let satellites = map_str(&f["ambiguity_satellites"]);
    let wavelengths = map_f64(&f["wavelengths_m"]);
    let offsets = map_f64(&f["offsets_m"]);
    let model = MeasModel {
        code_sigma_m: f["model"]["code_sigma_m"].as_f64().unwrap(),
        phase_sigma_m: f["model"]["phase_sigma_m"].as_f64().unwrap(),
        sagnac: f["model"]["sagnac"].as_bool().unwrap(),
        stochastic: StochasticModel::Simple {
            elevation_weighting: f["model"]["stochastic"]["elevation_weighting"]
                .as_bool()
                .unwrap(),
        },
    };
    let fo = &f["float_opts"];
    let float = FloatSolveOpts {
        position_tol_m: fo["position_tol_m"].as_f64().unwrap(),
        ambiguity_tol_m: fo["ambiguity_tol_m"].as_f64().unwrap(),
        max_iterations: fo["max_iterations"].as_u64().unwrap() as usize,
    };
    let fo = &f["fixed_opts"];
    let fixed = FixedSolveOpts {
        position_tol_m: fo["position_tol_m"].as_f64().unwrap(),
        ambiguity_tol_m: fo["ambiguity_tol_m"].as_f64().unwrap(),
        max_iterations: fo["max_iterations"].as_u64().unwrap() as usize,
        ratio_threshold: fo["ratio_threshold"].as_f64().unwrap(),
        partial_ambiguity_resolution: fo["partial_ambiguity_resolution"].as_bool().unwrap(),
        partial_min_ambiguities: fo["partial_min_ambiguities"].as_u64().unwrap() as usize,
    };
    let residual = ResidualValidationOpts {
        threshold_sigma: f["residual_opts"]["threshold_sigma"].as_f64(),
        max_exclusions: f["residual_opts"]["max_exclusions"].as_u64().unwrap() as usize,
    };
    let opts = ValidatedFixedSolveOpts {
        float,
        fixed,
        residual,
    };
    let fl = solve_float_baseline(&epochs, base, &ids, initial, &model, float, None).unwrap();
    let empty: Vec<String> = vec![];
    let fixed_sol = solve_fixed_baseline_validated(
        &epochs,
        base,
        AmbiguitySet {
            ids: &ids,
            satellites: &satellites,
            scale: AmbiguityScale {
                wavelengths_m: &wavelengths,
                offsets_m: &offsets,
            },
            float_only_systems: &empty,
        },
        initial,
        &model,
        opts,
        None,
    )
    .unwrap();
    let raw: Vec<RtkArcEpoch> = epochs
        .iter()
        .map(|e| {
            let all: Vec<_> = e.references.iter().chain(&e.nonref).collect();
            RtkArcEpoch {
                base: all.iter().map(|r| arc_obs(r, false)).collect(),
                rover: all.iter().map(|r| arc_obs(r, true)).collect(),
                satellite_positions_m: all.iter().map(|r| (r.sat.clone(), r.pos)).collect(),
                base_satellite_positions_m: all
                    .iter()
                    .map(|r| (r.sat.clone(), r.base_tx_pos))
                    .collect(),
                rover_satellite_positions_m: all
                    .iter()
                    .map(|r| (r.sat.clone(), r.rover_tx_pos))
                    .collect(),
                velocity_mps: e.velocity_mps,
                prediction_time_s: None,
                prediction_epoch: None,
            }
        })
        .collect();
    let mut w = BTreeMap::new();
    let mut o = BTreeMap::new();
    for (k, v) in &wavelengths {
        w.insert(k.split('|').next().unwrap().to_owned(), *v);
        w.insert(k.clone(), *v);
    }
    for (k, v) in &offsets {
        o.insert(k.split('|').next().unwrap().to_owned(), *v);
        o.insert(k.clone(), *v);
    }
    let update = UpdateOpts {
        hold_sigma_m: 1e-4,
        position_tol_m: defaults::POSITION_TOL_M,
        ambiguity_tol_m: defaults::AMBIGUITY_TOL_M,
        max_iterations: defaults::MAX_ITERATIONS,
        process_noise_baseline_sigma_m: 0.,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: vec![],
        report_residuals: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: defaults::RATIO_THRESHOLD,
        },
    };
    let arc = RtkArcConfig::new(
        base,
        BaselineReferenceSelection::Auto,
        model,
        30.,
        30.,
        [0.; 3],
        w,
        o,
        update,
        RtkArcPreprocessing::default(),
    );
    let static_sol = solve_static_rtk_arc(&raw, &RtkStaticArcConfig::new(arc, opts)).unwrap();
    json!({"float_baseline_m":h(&fl.baseline_m),"validated_float_baseline_m":h(&fixed_sol.float_solution.baseline_m),"fixed_baseline_m":h(&fixed_sol.fixed_solution.baseline_m),"static_fixed_baseline_m":h(&static_sol.fixed_solution.fixed_solution.baseline_m)})
}
