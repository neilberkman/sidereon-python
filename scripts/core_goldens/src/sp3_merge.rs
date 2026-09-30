//! Goldens for the SP3 merge audit trail: per-epoch provenance, omitted
//! epochs, withheld arc cells, clock omissions, dropped input epochs, the
//! continuity post-condition with its contributor attribution and node
//! selection, agreement aggregates and per-satellite coverage.
//!
//! The scenarios are the ones `crates/sidereon-core/tests/
//! sp3_merge_per_epoch_provenance.rs` and `tests/sp3_merge_coverage.rs` build.
//! The coverage fixture's circular trajectories are computed here once and
//! written as exact bits, so the Python tests rebuild byte-identical SP3 text
//! without evaluating a cosine; each built text's FNV-1a 64 digest is written
//! too, so a test can prove its inputs are these.

use serde_json::{json, Map, Value};
use sidereon_core::astro::time::{Instant, InstantRepr};
use sidereon_core::ephemeris::{
    check_continuity, merge, CellSelection, ClockOmission, ClockOmissionReason, ContinuityDefect,
    ContinuityOptions, DroppedEpochReason, DroppedInputEpoch, EpochWindow, InterpolationNodes,
    MergeCombine, MergeContinuityCellRole, MergeContinuityReport, MergeFlag, MergeOptions,
    MergePrecedenceScope, MergeProvenance, MergeReport, OrbitClass, OutlierRejectOptions,
    ProvenanceMode, Sp3, Sp3ChannelCoverage, Sp3Coverage, StencilExtent, TransitionReason,
    WindowContinuityDecision,
};

use super::{hex, hexes};

const STEP_S: usize = 300;
/// Highest epoch index any coverage scenario uses, plus one.
const FIXTURE_EPOCHS: usize = 84;

/// FNV-1a 64 of `bytes`, as sixteen lowercase hex digits.
fn fnv1a64(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

fn epoch(instant: &Instant) -> Value {
    match instant.repr {
        InstantRepr::JulianDate(split) => json!([hex(split.jd_whole), hex(split.fraction)]),
        InstantRepr::Nanos(nanos) => json!({ "nanos": nanos.to_string() }),
    }
}

fn epochs(instants: &[Instant]) -> Value {
    Value::Array(instants.iter().map(epoch).collect())
}

fn optional_hex(value: Option<f64>) -> Value {
    value.map_or(Value::Null, hex)
}

fn combine_label(rule: MergeCombine) -> &'static str {
    match rule {
        MergeCombine::Mean => "mean",
        MergeCombine::Median => "median",
        MergeCombine::Precedence => "precedence",
    }
}

fn flags(flags: &[MergeFlag]) -> Value {
    Value::Array(
        flags
            .iter()
            .map(|flag| {
                json!({
                    "epoch": epoch(&flag.epoch),
                    "satellite": flag.satellite.to_string(),
                    "sources": flag.sources,
                })
            })
            .collect(),
    )
}

fn clock_omission(omission: &ClockOmission) -> Value {
    let (reason, preferred) = match omission.reason {
        ClockOmissionReason::DatumNotObservable => ("datum_not_observable", None),
        ClockOmissionReason::PreferredSourceWithoutClock { preferred } => {
            ("preferred_source_without_clock", preferred)
        }
        ClockOmissionReason::NoConsensus => ("no_consensus", None),
    };
    json!({
        "epoch": epoch(&omission.epoch),
        "satellite": omission.satellite.to_string(),
        "source": omission.source,
        "reason": reason,
        "preferred_source": preferred,
        "cell_has_clock": omission.cell_has_clock,
    })
}

fn dropped_input_epoch(dropped: &DroppedInputEpoch) -> Value {
    json!({
        "source": dropped.source,
        "epoch_index": dropped.epoch_index,
        "epoch": epoch(&dropped.epoch),
        "reason": match dropped.reason {
            DroppedEpochReason::OffTargetGrid => "off_target_grid",
            DroppedEpochReason::NotOnTickAxis => "not_on_tick_axis",
        },
    })
}

fn selection(selection: &CellSelection) -> Value {
    match selection {
        CellSelection::SingleSource { source } => json!({
            "kind": "single_source",
            "source": source,
            "members": [source],
            "rule": null,
        }),
        CellSelection::Precedence { source, members } => json!({
            "kind": "precedence",
            "source": source,
            "members": members,
            "rule": null,
        }),
        CellSelection::Combined { rule, members } => json!({
            "kind": "combined",
            "source": null,
            "members": members,
            "rule": combine_label(*rule),
        }),
    }
}

fn optional_selection(value: Option<&CellSelection>) -> Value {
    value.map_or(Value::Null, selection)
}

fn provenance(record: &MergeProvenance) -> Value {
    json!({
        "mode": match record.mode {
            ProvenanceMode::Summary => "summary",
            ProvenanceMode::Full => "full",
        },
        "cells": record.cells.iter().map(|cell| json!({
            "epoch": epoch(&cell.epoch),
            "satellite": cell.satellite.to_string(),
            "position": optional_selection(cell.position.as_ref()),
            "clock": optional_selection(cell.clock.as_ref()),
        })).collect::<Vec<_>>(),
        "transitions": record.transitions.iter().map(|transition| json!({
            "satellite": transition.satellite.to_string(),
            "epoch": epoch(&transition.epoch),
            "from_source": transition.from_source,
            "to_source": transition.to_source,
            "reason": match transition.reason {
                TransitionReason::SoleAvailability => "sole_availability",
                TransitionReason::Precedence => "precedence",
                TransitionReason::OutlierRejection => "outlier_rejection",
                TransitionReason::ConsensusChange => "consensus_change",
            },
        })).collect::<Vec<_>>(),
        "coverage": record.coverage.iter().map(|coverage| json!({
            "source": coverage.source,
            "cells_contributed": coverage.cells_contributed,
            "cells_selected": coverage.cells_selected,
            "first_epoch": coverage.first_epoch.as_ref().map_or(Value::Null, epoch),
            "last_epoch": coverage.last_epoch.as_ref().map_or(Value::Null, epoch),
            "cells_absent": coverage.cells_absent,
        })).collect::<Vec<_>>(),
    })
}

fn defect(defect: &ContinuityDefect) -> Value {
    match defect {
        ContinuityDefect::DuplicateEpoch {
            sat,
            epoch_j2000_s,
            occurrences,
        } => json!({
            "kind": "duplicate_epoch",
            "satellite": sat.to_string(),
            "epoch_j2000_s": hex(*epoch_j2000_s),
            "occurrences": occurrences,
        }),
        ContinuityDefect::SingleSampleSeries { sat } => json!({
            "kind": "single_sample_series",
            "satellite": sat.to_string(),
        }),
        ContinuityDefect::UnusableSample {
            sat,
            sample_index,
            epoch_j2000_s,
            reason,
        } => json!({
            "kind": "unusable_sample",
            "satellite": sat.to_string(),
            "sample_index": sample_index,
            "epoch_j2000_s": epoch_j2000_s.map_or(Value::Null, hex),
            "reason": match reason {
                sidereon_core::ephemeris::UnusableSampleReason::EpochNotPlaced => "epoch_not_placed",
                sidereon_core::ephemeris::UnusableSampleReason::NonFinitePosition => "non_finite_position",
                _ => "unknown",
            },
        }),
        ContinuityDefect::SpeedBound {
            sat,
            from_j2000_s,
            to_j2000_s,
            interval_s,
            displacement_m,
            implied_speed_m_s,
            bound_m_s,
        } => json!({
            "kind": "speed_bound",
            "satellite": sat.to_string(),
            "from_j2000_s": hex(*from_j2000_s),
            "to_j2000_s": hex(*to_j2000_s),
            "interval_s": hex(*interval_s),
            "displacement_m": hex(*displacement_m),
            "implied_speed_m_s": hex(*implied_speed_m_s),
            "bound_m_s": hex(*bound_m_s),
        }),
        ContinuityDefect::HoldOutResidual {
            sat,
            epoch_j2000_s,
            preceding_j2000_s,
            residual_m,
            tolerance_m,
            node_epochs_j2000_s,
        } => json!({
            "kind": "hold_out_residual",
            "satellite": sat.to_string(),
            "epoch_j2000_s": hex(*epoch_j2000_s),
            "preceding_j2000_s": hex(*preceding_j2000_s),
            "residual_m": hex(*residual_m),
            "tolerance_m": hex(*tolerance_m),
            "node_epochs_j2000_s": hexes(node_epochs_j2000_s),
        }),
    }
}

fn continuity(report: &MergeContinuityReport) -> Value {
    json!({
        "attested": report.attested(),
        "pairs_checked": report.report.pairs_checked,
        "residuals_checked": report.report.residuals_checked,
        "residuals_skipped": report.report.residuals_skipped,
        "defects": report.report.defects.iter().map(defect).collect::<Vec<_>>(),
        "violations": report.violations.iter().map(|violation| json!({
            "defect": defect(&violation.defect),
            "from_sources": violation.from_sources,
            "to_sources": violation.to_sources,
            "cells": violation.cells.iter().map(|cell| json!({
                "epoch_j2000_s": hex(cell.epoch_j2000_s),
                "role": match cell.role {
                    MergeContinuityCellRole::HeldOut => "held_out",
                    MergeContinuityCellRole::InterpolationNode => "interpolation_node",
                    MergeContinuityCellRole::PairEnd => "pair_end",
                    MergeContinuityCellRole::RepeatedEpoch => "repeated_epoch",
                },
                "selection": optional_selection(cell.selection.as_ref()),
            })).collect::<Vec<_>>(),
            "sources": violation.sources,
            "crosses_contributors": violation.crosses_contributors,
        })).collect::<Vec<_>>(),
    })
}

fn report(report: &MergeReport) -> Value {
    json!({
        "quarantined": flags(&report.quarantined),
        "single_source": flags(&report.single_source),
        "position_outliers": flags(&report.position_outliers),
        "clock_outliers": flags(&report.clock_outliers),
        "arc_withheld": flags(&report.arc_withheld),
        "omitted_epochs": epochs(&report.omitted_epochs),
        "clock_omissions": report.clock_omissions.iter().map(clock_omission).collect::<Vec<_>>(),
        "dropped_input_epochs": report
            .dropped_input_epochs
            .iter()
            .map(dropped_input_epoch)
            .collect::<Vec<_>>(),
        "provenance": report.provenance.as_ref().map_or(Value::Null, provenance),
        "continuity": report.continuity.as_ref().map_or(Value::Null, continuity),
        "single_source_fraction": optional_hex(report.single_source_fraction()),
        "position_agreement_rms_m": optional_hex(report.position_agreement_rms_m()),
        "position_agreement_max_m": optional_hex(report.position_agreement_max_m()),
        "clock_agreement_rms_s": optional_hex(report.clock_agreement_rms_s()),
        "clock_agreement_max_s": optional_hex(report.clock_agreement_max_s()),
        "per_epoch_agreement": report.per_epoch_agreement().iter().map(|entry| json!({
            "epoch": epoch(&entry.epoch),
            "satellites": entry.satellites,
            "position_rms_m": optional_hex(entry.position_rms_m),
            "position_max_m": optional_hex(entry.position_max_m),
            "clock_rms_s": optional_hex(entry.clock_rms_s),
            "clock_max_s": optional_hex(entry.clock_max_s),
        })).collect::<Vec<_>>(),
    })
}

fn product(sp3: &Sp3) -> Value {
    json!({
        "epochs": epochs(&sp3.epochs),
        "num_epochs": sp3.header.num_epochs,
        "epoch_interval_s": hex(sp3.header.epoch_interval_s),
        "mjd": sp3.header.mjd,
        "mjd_fraction": hex(sp3.header.mjd_fraction),
        "seconds_of_week": hex(sp3.header.seconds_of_week),
        "declared_start_j2000_s": optional_hex(sp3.declared_start_j2000_s()),
        "text_fnv1a64": sp3
            .to_sp3_string()
            .ok()
            .map_or(Value::Null, |text| Value::String(fnv1a64(text.as_bytes()))),
    })
}

fn channel(coverage: &Sp3ChannelCoverage) -> Value {
    json!({
        "epochs": coverage.epochs,
        "spans": coverage.spans.iter().map(|span| json!({
            "first_index": span.first_index,
            "last_index": span.last_index,
            "first_epoch": epoch(&span.first_epoch),
            "last_epoch": epoch(&span.last_epoch),
        })).collect::<Vec<_>>(),
        "gaps": coverage.gaps.iter().map(|gap| json!({
            "after_index": gap.after_index,
            "before_index": gap.before_index,
            "missing_epochs": gap.missing_epochs,
        })).collect::<Vec<_>>(),
    })
}

fn coverage(coverage: &Sp3Coverage) -> Value {
    json!({
        "grid": {
            "interval_s": optional_hex(coverage.grid.interval_s),
            "agrees_with_header": coverage.grid.agrees_with_header,
            "out_of_order": coverage.grid.out_of_order,
            "unplaced": coverage.grid.unplaced,
        },
        "satellites": coverage.satellites.iter().map(|satellite| json!({
            "satellite": satellite.satellite.to_string(),
            "declared": satellite.declared,
            "positions": channel(&satellite.positions),
            "clocks": channel(&satellite.clocks),
        })).collect::<Vec<_>>(),
    })
}

/// A merge outcome: the product and report, or the refusal's message.
fn merged(sources: &[Sp3], options: &MergeOptions) -> Value {
    match merge(sources, options) {
        Ok((product_value, report_value)) => json!({
            "product": product(&product_value),
            "report": report(&report_value),
        }),
        Err(error) => json!({ "error": error.to_string() }),
    }
}

fn decision_label(decision: WindowContinuityDecision) -> &'static str {
    match decision {
        WindowContinuityDecision::Accept => "accept",
        WindowContinuityDecision::Refuse => "refuse",
    }
}

// --- the coverage fixture ---------------------------------------------------

/// What one satellite carries at one epoch of a built product.
#[derive(Clone, Copy, PartialEq)]
enum Record {
    Full,
    NoClock,
    ClockOnly,
    Absent,
}

/// `[x_km without offset, y_km, z_km]` of each of G01..G06 at each fixture
/// epoch index.
fn trajectory() -> Vec<[[f64; 3]; 6]> {
    (0..FIXTURE_EPOCHS)
        .map(|index| {
            let seconds = (index * STEP_S) as f64;
            let mut row = [[0.0; 3]; 6];
            for prn in 1..=6u8 {
                let angle = seconds * core::f64::consts::TAU / 43_200.0 + f64::from(prn) * 0.3;
                row[usize::from(prn - 1)] = [
                    26_560.0 * angle.cos(),
                    26_560.0 * angle.sin() * 0.6,
                    26_560.0 * angle.sin() * 0.8,
                ];
            }
            row
        })
        .collect()
}

fn epoch_fields(index: usize) -> String {
    format!(
        "2020  6 25 {:2}{:3}  0.00000000",
        index / 12,
        (index % 12) * 5
    )
}

/// The product `crates/sidereon-core/tests/sp3_merge_coverage.rs` builds, with
/// its header interval written as `interval_s` (the core test assigns the
/// header field after parsing; writing it into line 2 gives the parser the same
/// value).
fn product_text(
    trajectory: &[[[f64; 3]; 6]],
    indices: &[usize],
    offset_m: f64,
    interval_s: f64,
    record: &dyn Fn(u8, usize) -> Record,
) -> String {
    let first = indices[0];
    let mut text = format!(
        "#cP{}     {:3} ORBIT IGS14 FIT  TST\n\
         ## 2111 {:14.8}{:15.8} 59025 {:.13}\n\
         +    6   G01G02G03G04G05G06  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* SYNTHETIC SP3 COVERAGE FIXTURE\n",
        epoch_fields(first),
        indices.len(),
        345_600.0 + (first * STEP_S) as f64,
        interval_s,
        (first * STEP_S) as f64 / 86_400.0
    );
    for &index in indices {
        text.push_str(&format!("*  {}\n", epoch_fields(index)));
        for prn in 1..=6u8 {
            let [x0, y, z] = trajectory[index][usize::from(prn - 1)];
            let x = x0 + offset_m / 1000.0;
            let clock = 10.0 + f64::from(prn);
            match record(prn, index) {
                Record::Full => {
                    text.push_str(&format!("PG{prn:02}{x:14.6}{y:14.6}{z:14.6}{clock:14.6}\n"))
                }
                Record::NoClock => text.push_str(&format!(
                    "PG{prn:02}{x:14.6}{y:14.6}{z:14.6}{:14.6}\n",
                    999_999.999_999
                )),
                Record::ClockOnly => text.push_str(&format!(
                    "PG{prn:02}{:14.6}{:14.6}{:14.6}{clock:14.6}\n",
                    0.0, 0.0, 0.0
                )),
                Record::Absent => {}
            }
        }
    }
    text.push_str("EOF\n");
    text
}

/// Parses `text` and records its digest under `name`.
fn built(texts: &mut Map<String, Value>, name: &str, text: String) -> Sp3 {
    texts.insert(name.to_string(), Value::String(fnv1a64(text.as_bytes())));
    Sp3::parse(text.as_bytes()).expect("parse built SP3")
}

fn precedence(scope: MergePrecedenceScope) -> MergeOptions {
    let mut options = MergeOptions::default();
    options.combine = MergeCombine::Precedence;
    options.precedence_scope = scope;
    options.min_agree = 1;
    options.position_tolerance_m = 5.0;
    options
}

fn full(_: u8, _: usize) -> Record {
    Record::Full
}

fn range(first: usize, count: usize) -> Vec<usize> {
    (first..first + count).collect()
}

fn verified(scope: MergePrecedenceScope) -> MergeOptions {
    let mut options = precedence(scope);
    options.verify_continuity = Some(ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss));
    options
}

/// The node selection of every satellite for a set of windows, by index into
/// the merged product's epochs.
fn node_selection(nodes: &InterpolationNodes, sp3: &Sp3, windows: &[(usize, usize)]) -> Value {
    let axis = sp3.epochs_j2000_seconds();
    let mut out = Vec::new();
    for &(from, through) in windows {
        let window = EpochWindow::new(axis[from], axis[through]).expect("window");
        let mut per_satellite = Map::new();
        for &satellite in sp3.satellites() {
            per_satellite.insert(
                satellite.to_string(),
                hexes(&nodes.selected_nodes(satellite, window)),
            );
        }
        out.push(json!({
            "from_index": from,
            "through_index": through,
            "selected_nodes": Value::Object(per_satellite),
        }));
    }
    Value::Array(out)
}

fn coverage_cases(trajectory: &[[[f64; 3]; 6]]) -> (Map<String, Value>, Map<String, Value>) {
    let mut texts = Map::new();
    let mut cases = Map::new();
    let a = built(
        &mut texts,
        "a_0_72",
        product_text(trajectory, &range(0, 72), 0.0, 300.0, &full),
    );
    let b = built(
        &mut texts,
        "b_60_24",
        product_text(trajectory, &range(60, 24), 0.0, 300.0, &full),
    );
    let b_displaced = built(
        &mut texts,
        "b_60_24_displaced_0_8",
        product_text(trajectory, &range(60, 24), 0.8, 300.0, &full),
    );

    // Hold-out attribution and window verdicts on the 0.8 m handover.
    {
        let sources = [a.clone(), b_displaced.clone()];
        let (merged_product, merged_report) =
            merge(&sources, &verified(MergePrecedenceScope::Cell)).expect("merge");
        let axis = merged_product.epochs_j2000_seconds();
        let continuity_report = merged_report
            .continuity
            .as_ref()
            .expect("verification requested");
        let merge_decisions: Vec<&str> = axis
            .iter()
            .map(|&epoch_s| {
                let window = EpochWindow::new(epoch_s, epoch_s).expect("window");
                decision_label(
                    merged_report
                        .continuity_verdict_for_window(window)
                        .expect("verification requested")
                        .decision,
                )
            })
            .collect();
        let plain = check_continuity(
            &merged_product.precise_ephemeris_samples(),
            &ContinuityOptions::for_orbit_class(OrbitClass::MeoGnss),
        )
        .expect("valid orbit-class continuity options");
        let stencil = StencilExtent::for_sp3(&merged_product).expect("stencil");
        let plain_decisions: Vec<&str> = axis
            .iter()
            .map(|&epoch_s| {
                let window = EpochWindow::new(epoch_s, epoch_s).expect("window");
                decision_label(plain.verdict_for_window(window, stencil).decision)
            })
            .collect();
        let ranges: Vec<Value> = [(51usize, 66usize), (51, 68)]
            .iter()
            .map(|&(from, through)| {
                let window = EpochWindow::new(axis[from], axis[through]).expect("window");
                let verdict = merged_report
                    .continuity_verdict_for_window(window)
                    .expect("verification requested");
                json!({
                    "from_index": from,
                    "through_index": through,
                    "decision": decision_label(verdict.decision),
                    "influencing_defects": verdict.influencing_defects.len(),
                    "influencing_splices": verdict.influencing_splices.len(),
                })
            })
            .collect();
        let windows = [(0usize, 0usize), (51, 66), (51, 68), (70, 70), (83, 83)];
        cases.insert(
            "hold_out_attribution".to_string(),
            json!({
                "merge": {
                    "product": product(&merged_product),
                    "report": report(&merged_report),
                },
                "merge_single_epoch_decisions": merge_decisions,
                "plain_single_epoch_decisions": plain_decisions,
                "range_verdicts": ranges,
                "stencil_before_s": hex(stencil.before_s()),
                "stencil_after_s": hex(stencil.after_s()),
                "report_nodes": node_selection(&continuity_report.nodes, &merged_product, &windows),
                "product_nodes": node_selection(
                    &InterpolationNodes::for_sp3(&merged_product),
                    &merged_product,
                    &windows,
                ),
            }),
        );
    }

    cases.insert(
        "undisplaced".to_string(),
        merged(
            &[a.clone(), b.clone()],
            &verified(MergePrecedenceScope::Cell),
        ),
    );

    {
        let sources = [a.clone(), b.clone()];
        let (merged_product, merged_report) =
            merge(&sources, &precedence(MergePrecedenceScope::Cell)).expect("merge");
        cases.insert(
            "cell_precedence_coverage".to_string(),
            json!({
                "product": product(&merged_product),
                "report": report(&merged_report),
                "coverage": coverage(&merged_product.satellite_coverage()),
            }),
        );
    }

    {
        let sources = [a.clone(), b.clone()];
        let (merged_product, merged_report) =
            merge(&sources, &precedence(MergePrecedenceScope::SatelliteArc)).expect("merge");
        cases.insert(
            "satellite_arc_omits".to_string(),
            json!({
                "product": product(&merged_product),
                "report": report(&merged_report),
                "coverage": coverage(&merged_product.satellite_coverage()),
            }),
        );
    }

    let skipping = built(
        &mut texts,
        "skip_0_1_2_4_5",
        product_text(trajectory, &[0, 1, 2, 4, 5], 0.0, 300.0, &full),
    );
    cases.insert(
        "skipped_epoch".to_string(),
        merged(&[skipping], &precedence(MergePrecedenceScope::Cell)),
    );

    let patterned = built(
        &mut texts,
        "patterned_coverage",
        product_text(
            trajectory,
            &[0, 1, 2, 3, 4, 6, 7, 8],
            0.0,
            300.0,
            &|prn, index| match (prn, index) {
                (1, _) => Record::Full,
                (2, 2 | 3) => Record::Absent,
                (2, _) => Record::Full,
                (3, 7 | 8) => Record::NoClock,
                (3, _) => Record::Full,
                (4, 0 | 1) => Record::ClockOnly,
                (4, _) => Record::Full,
                (_, _) => Record::Absent,
            },
        ),
    );
    cases.insert(
        "product_coverage".to_string(),
        json!({ "coverage": coverage(&patterned.satellite_coverage()) }),
    );

    cases.insert(
        "reversed_satellite_arc".to_string(),
        merged(
            &[b.clone(), a.clone()],
            &precedence(MergePrecedenceScope::SatelliteArc),
        ),
    );
    cases.insert(
        "single_starting_at_five".to_string(),
        merged(&[b.clone()], &precedence(MergePrecedenceScope::Cell)),
    );

    {
        let indices = range(60, 4);
        let near = built(
            &mut texts,
            "near_no_clock_60_4",
            product_text(trajectory, &indices, 0.0, 300.0, &|_, _| Record::NoClock),
        );
        let far = built(
            &mut texts,
            "far_no_clock_60_4",
            product_text(trajectory, &indices, 1_000.0, 300.0, &|_, _| {
                Record::NoClock
            }),
        );
        let mut options = MergeOptions::default();
        options.min_agree = 2;
        cases.insert("epochless".to_string(), merged(&[near, far], &options));
    }

    {
        let even: Vec<usize> = (0..24).step_by(2).collect();
        let odd: Vec<usize> = (1..24).step_by(2).collect();
        let even_product = built(
            &mut texts,
            "even_0_24",
            product_text(trajectory, &even, 0.0, 300.0, &full),
        );
        let odd_product = built(
            &mut texts,
            "odd_1_24",
            product_text(trajectory, &odd, 0.0, 300.0, &full),
        );
        cases.insert(
            "phase_offset".to_string(),
            merged(
                &[even_product, odd_product],
                &precedence(MergePrecedenceScope::Cell),
            ),
        );
    }

    let twelve = built(
        &mut texts,
        "a_0_12",
        product_text(trajectory, &range(0, 12), 0.0, 300.0, &full),
    );
    {
        let mut options = precedence(MergePrecedenceScope::Cell);
        options.target_epoch_interval_s = Some(900.0);
        cases.insert(
            "explicit_target".to_string(),
            merged(std::slice::from_ref(&twelve), &options),
        );
        let mut options = precedence(MergePrecedenceScope::Cell);
        options.target_epoch_interval_s = Some(450.5);
        cases.insert(
            "fractional_target".to_string(),
            merged(std::slice::from_ref(&twelve), &options),
        );
    }

    {
        let whole = built(
            &mut texts,
            "a_0_1",
            product_text(trajectory, &range(0, 1), 0.0, 300.0, &full),
        );
        let text = whole
            .to_sp3_string()
            .expect("write")
            .replace("25  0  0  0.00000000", "25  0  0  0.50000000");
        let half = built(&mut texts, "a_0_1_half_second", text);
        cases.insert(
            "half_second_apart".to_string(),
            merged(&[whole, half], &precedence(MergePrecedenceScope::Cell)),
        );
    }

    {
        let indices = range(0, 12);
        let first = built(
            &mut texts,
            "clock_a_0_12",
            product_text(trajectory, &indices, 0.0, 300.0, &full),
        );
        let second = first.clone();
        let third = built(
            &mut texts,
            "clock_c_0_12",
            product_text(trajectory, &indices, 0.0, 300.0, &|prn, _| {
                if prn <= 4 {
                    Record::Full
                } else {
                    Record::NoClock
                }
            }),
        );
        cases.insert(
            "clock_omissions".to_string(),
            merged(&[first, second, third], &MergeOptions::default()),
        );
    }

    {
        let off_grid = built(
            &mut texts,
            "gapped_declared_600",
            product_text(trajectory, &[0, 1, 2, 4, 5], 0.0, 600.0, &full),
        );
        let wrong_header = built(
            &mut texts,
            "uniform_declared_900",
            product_text(trajectory, &range(0, 4), 0.0, 900.0, &full),
        );
        let zero_gapped = built(
            &mut texts,
            "gapped_declared_0",
            product_text(trajectory, &[0, 1, 2, 4, 5], 0.0, 0.0, &full),
        );
        let zero_uniform = built(
            &mut texts,
            "uniform_declared_0",
            product_text(trajectory, &range(0, 4), 0.0, 0.0, &full),
        );
        let out_of_order = built(
            &mut texts,
            "out_of_order_0_2_1_3",
            product_text(trajectory, &[0, 2, 1, 3], 0.0, 300.0, &full),
        );
        cases.insert(
            "gapped_off_declared_grid".to_string(),
            json!({
                "merge": merged(
                    std::slice::from_ref(&off_grid),
                    &precedence(MergePrecedenceScope::Cell),
                ),
                "coverage": coverage(&off_grid.satellite_coverage()),
            }),
        );
        cases.insert(
            "uniform_wrong_header".to_string(),
            json!({
                "merge": merged(
                    std::slice::from_ref(&wrong_header),
                    &precedence(MergePrecedenceScope::Cell),
                ),
                "coverage": coverage(&wrong_header.satellite_coverage()),
            }),
        );
        cases.insert(
            "zero_interval_gapped".to_string(),
            json!({ "coverage": coverage(&zero_gapped.satellite_coverage()) }),
        );
        cases.insert(
            "zero_interval_uniform".to_string(),
            json!({ "coverage": coverage(&zero_uniform.satellite_coverage()) }),
        );
        cases.insert(
            "out_of_order".to_string(),
            json!({ "coverage": coverage(&out_of_order.satellite_coverage()) }),
        );
    }

    (texts, cases)
}

// --- the per-epoch provenance fixture ----------------------------------------

const PROVENANCE_HEADER_TAIL: &str = "\
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
%i    0    0    0    0      0      0      0      0         0\n\
%i    0    0    0    0      0      0      0      0         0\n\
/* TEST SP3-c FIXTURE\n";

/// `source` of `sp3_merge_per_epoch_provenance.rs`: G01 at 00:00 and 00:15
/// with the given X coordinates, kilometres.
fn provenance_source_text(positions_km: [f64; 2]) -> String {
    format!(
        "#cP2020  6 25  0  0  0.00000000       2 ORBIT IGS14 FIT  TST\n\
         ## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n\
         {PROVENANCE_HEADER_TAIL}\
         *  2020  6 25  0  0  0.00000000\n\
         PG01 {:13.6} -20000.000000   5000.000000    100.000000\n\
         *  2020  6 25  0 15  0.00000000\n\
         PG01 {:13.6} -20000.000000   5000.000000    100.000000\n\
         EOF\n",
        positions_km[0], positions_km[1]
    )
}

/// `late_source`: G01 at 00:15 only.
fn provenance_late_text(position_km: f64) -> String {
    format!(
        "#cP2020  6 25  0 15  0.00000000       1 ORBIT IGS14 FIT  TST\n\
         ## 2111 432900.00000000   900.00000000 59025 0.0000000000000\n\
         {PROVENANCE_HEADER_TAIL}\
         *  2020  6 25  0 15  0.00000000\n\
         PG01 {position_km:13.6} -20000.000000   5000.000000    100.000000\n\
         EOF\n"
    )
}

/// The source of the forced-switch case: G01 at 00:00 only.
fn provenance_early_text() -> String {
    format!(
        "#cP2020  6 25  0  0  0.00000000       1 ORBIT IGS14 FIT  TST\n\
         ## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n\
         {PROVENANCE_HEADER_TAIL}\
         *  2020  6 25  0  0  0.00000000\n\
         PG01  15000.000000 -20000.000000   5000.000000    100.000000\n\
         EOF\n"
    )
}

fn provenance_precedence(mode: Option<ProvenanceMode>) -> MergeOptions {
    let mut options = MergeOptions::default();
    options.combine = MergeCombine::Precedence;
    options.precedence_scope = MergePrecedenceScope::Cell;
    options.min_agree = 1;
    options.provenance = mode;
    options
}

fn provenance_cases() -> (Map<String, Value>, Map<String, Value>) {
    let mut texts = Map::new();
    let mut cases = Map::new();
    let steady = built(
        &mut texts,
        "source_15000_15100",
        provenance_source_text([15_000.0, 15_100.0]),
    );
    let late = built(&mut texts, "late_15100", provenance_late_text(15_100.0));
    let late_half = built(&mut texts, "late_15100_5", provenance_late_text(15_100.5));
    let early = built(&mut texts, "early_15000", provenance_early_text());
    let wild = built(
        &mut texts,
        "source_15000_25000",
        provenance_source_text([15_000.0, 25_000.0]),
    );

    cases.insert(
        "not_requested".to_string(),
        merged(std::slice::from_ref(&steady), &provenance_precedence(None)),
    );
    cases.insert(
        "single_contributor".to_string(),
        merged(
            std::slice::from_ref(&steady),
            &provenance_precedence(Some(ProvenanceMode::Full)),
        ),
    );
    cases.insert(
        "forced_switch".to_string(),
        merged(
            &[early, late.clone()],
            &provenance_precedence(Some(ProvenanceMode::Full)),
        ),
    );
    {
        let mut options = MergeOptions::default();
        options.combine = MergeCombine::Precedence;
        options.precedence_scope = MergePrecedenceScope::Cell;
        options.min_agree = 2;
        options.position_tolerance_m = 1.0;
        options.outlier_reject = Some(OutlierRejectOptions::new(1.0, 1.0e-6));
        options.provenance = Some(ProvenanceMode::Full);
        cases.insert(
            "outlier_rejection".to_string(),
            merged(&[wild, steady.clone(), steady.clone()], &options),
        );
    }
    {
        let mut options = MergeOptions::default();
        options.provenance = Some(ProvenanceMode::Full);
        cases.insert(
            "combined".to_string(),
            merged(&[steady.clone(), steady.clone()], &options),
        );
    }
    cases.insert(
        "full_mode".to_string(),
        merged(
            &[steady.clone(), late.clone()],
            &provenance_precedence(Some(ProvenanceMode::Full)),
        ),
    );
    cases.insert(
        "summary_mode".to_string(),
        merged(
            &[steady.clone(), late],
            &provenance_precedence(Some(ProvenanceMode::Summary)),
        ),
    );
    cases.insert(
        "product_without_provenance".to_string(),
        merged(
            &[steady.clone(), late_half.clone()],
            &provenance_precedence(None),
        ),
    );
    cases.insert(
        "product_with_provenance".to_string(),
        merged(
            &[steady, late_half],
            &provenance_precedence(Some(ProvenanceMode::Full)),
        ),
    );
    (texts, cases)
}

pub fn sections() -> Vec<(&'static str, Value)> {
    let trajectory = trajectory();
    let (coverage_texts, coverage_cases) = coverage_cases(&trajectory);
    let (provenance_texts, provenance_cases) = provenance_cases();
    let trajectory_bits: Vec<Value> = trajectory
        .iter()
        .map(|row| {
            Value::Array(
                row.iter()
                    .map(|position| hexes(position))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    vec![(
        "sp3_merge",
        json!({
            "coverage_fixture": {
                "step_s": STEP_S,
                "trajectory_km": trajectory_bits,
                "texts_fnv1a64": Value::Object(coverage_texts),
                "cases": Value::Object(coverage_cases),
            },
            "provenance_fixture": {
                "texts_fnv1a64": Value::Object(provenance_texts),
                "cases": Value::Object(provenance_cases),
            },
        }),
    )]
}
