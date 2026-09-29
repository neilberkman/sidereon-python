"""Per-epoch SP3 merge provenance through the binding: which contributor
supplied each cell, where selection changed, and what each contributor covered.

Ported from `crates/sidereon-core/tests/sp3_merge_per_epoch_provenance.rs`.
The sources are that file's SP3 text; each merge outcome is compared whole with
the one `scripts/core_goldens` recorded from the core for the same inputs
(delegation), and each test then asserts the core test's own properties on the
binding's typed records.
"""

import sidereon
from _sp3_merge import (
    merged,
    provenance_case,
    provenance_early,
    provenance_late,
    provenance_source,
)


def _steady():
    return provenance_source("source_15000_15100", [15_000.0, 15_100.0])


def _precedence(mode):
    return sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        min_agree=1,
        provenance=mode,
    )


def _merge(case, sources, options):
    outcome, result = merged(sources, options)
    assert outcome == provenance_case(case)
    assert result is not None
    return result


def test_provenance_is_absent_unless_requested():
    options = _precedence(None)
    assert options.provenance is None
    _merged, report = _merge("not_requested", [_steady()], options)
    assert report.provenance is None


def test_a_single_contributor_merge_records_it_for_every_epoch_with_no_transition():
    options = _precedence("full")
    assert options.provenance == sidereon.Sp3ProvenanceMode.FULL
    _merged, report = _merge("single_contributor", [_steady()], options)

    record = report.provenance
    assert record.mode == sidereon.Sp3ProvenanceMode.FULL
    assert len(record.cells) == 2
    for cell in record.cells:
        assert cell.position is not None
        assert cell.position.selected_source == 0

    assert len(record.transitions) == 1
    assert record.transitions[0].from_source is None
    assert record.transitions[0].to_source == 0

    coverage = record.coverage[0]
    assert coverage.cells_contributed == 2
    assert coverage.cells_selected == 2
    assert coverage.cells_absent == 0
    assert coverage.first_epoch is not None
    assert coverage.last_epoch is not None


def test_a_forced_precedence_switch_records_one_transition_naming_both_sides():
    early = provenance_early("early_15000")
    late = provenance_late("late_15100", 15_100.0)
    _merged, report = _merge(
        "forced_switch", [early, late], _precedence(sidereon.Sp3ProvenanceMode.FULL)
    )

    record = report.provenance
    assert len(record.cells) == 2
    assert record.cells[0].position.selected_source == 0
    assert record.cells[1].position.selected_source == 1

    changes = [t for t in record.transitions if t.from_source is not None]
    assert len(changes) == 1
    assert changes[0].from_source == 0
    assert changes[0].to_source == 1
    assert changes[0].reason == "sole_availability"

    assert record.coverage[0].cells_contributed == 1
    assert record.coverage[0].cells_absent == 1
    assert record.coverage[1].cells_contributed == 1
    assert record.coverage[1].cells_absent == 1


def test_outlier_rejection_is_recorded_as_its_own_reason():
    wild = provenance_source("source_15000_25000", [15_000.0, 25_000.0])
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        min_agree=2,
        position_tolerance_m=1.0,
        outlier_reject=sidereon.Sp3OutlierRejectOptions(1.0, 1.0e-6),
        provenance="full",
    )
    _merged, report = _merge("outlier_rejection", [wild, _steady(), _steady()], options)

    record = report.provenance
    assert record.cells[0].position.selected_source == 0
    assert record.cells[1].position.selected_source == 1

    changes = [t for t in record.transitions if t.from_source is not None]
    assert len(changes) == 1
    assert changes[0].reason == "outlier_rejection"
    assert report.position_outliers


def test_a_combined_cell_names_no_single_supplier():
    options = sidereon.Sp3MergeOptions(provenance="full")
    _merged, report = _merge("combined", [_steady(), _steady()], options)

    record = report.provenance
    for cell in record.cells:
        position = cell.position
        assert position.kind == "combined"
        assert position.rule == sidereon.Sp3MergeCombine.MEAN
        assert position.selected_source is None
        assert position.members == [0, 1]
    for coverage in record.coverage:
        assert coverage.cells_selected == 0
        assert coverage.cells_contributed == 2


def test_summary_and_full_modes_agree_on_every_transition_they_both_describe():
    late = provenance_late("late_15100", 15_100.0)
    _full_merged, full_report = _merge(
        "full_mode", [_steady(), late], _precedence("full")
    )
    _summary_merged, summary_report = _merge(
        "summary_mode", [_steady(), late], _precedence("summary")
    )

    full = full_report.provenance
    summary = summary_report.provenance
    assert summary.mode == sidereon.Sp3ProvenanceMode.SUMMARY
    assert full.transitions == summary.transitions
    assert full.coverage == summary.coverage
    assert summary.cells == []
    assert full.cells


def test_the_merged_product_is_byte_identical_whether_or_not_provenance_is_enabled():
    late = provenance_late("late_15100_5", 15_100.5)
    without, report_without = _merge(
        "product_without_provenance", [_steady(), late], _precedence(None)
    )
    with_provenance, report_with = _merge(
        "product_with_provenance", [_steady(), late], _precedence("full")
    )

    assert without.to_sp3_string() == with_provenance.to_sp3_string()

    def agreement(report):
        return [
            (
                metric.jd_whole,
                metric.jd_fraction,
                metric.satellite,
                metric.position_members,
                metric.position_rms_m,
                metric.position_max_m,
                metric.clock_members,
                metric.clock_rms_s,
                metric.clock_max_s,
            )
            for metric in report.agreement
        ]

    def flags(values):
        return [(flag.epoch, flag.satellite, flag.sources) for flag in values]

    assert agreement(report_without) == agreement(report_with)
    assert flags(report_without.single_source) == flags(report_with.single_source)
    assert flags(report_without.quarantined) == flags(report_with.quarantined)
    assert flags(report_without.position_outliers) == flags(
        report_with.position_outliers
    )
    assert report_without.provenance is None
    assert report_with.provenance is not None
