"""What an SP3 merge says about each satellite, through the binding: which
contributors a continuity finding rests on, where positions and clocks are and
are not, and which epochs and cells it did not write.

Ported from `crates/sidereon-core/tests/sp3_merge_coverage.rs`. The fixture is
that file's: six GPS satellites on circular trajectories on a 300-second grid
on 2020-06-25; source A carries epochs 0-71, source B epochs 60-83, and B can be
displaced bodily along X. Each merge or coverage outcome is compared whole with
the one `scripts/core_goldens` recorded from the core for the same inputs
(delegation), and each test then asserts the core test's own properties on the
binding's typed records. The core's integer-nanosecond epoch case has no
Python counterpart: nothing in the binding rebuilds a product's epochs in
another representation.
"""

import pytest
import sidereon
from _sp3_merge import (
    ABSENT,
    CLOCK_ONLY,
    COVERAGE,
    FULL,
    NO_CLOCK,
    STEP_S,
    coverage,
    coverage_case,
    coverage_product,
    f64_hex,
    fnv1a64,
    merged,
    node_selection,
)


def _a():
    return coverage_product("a_0_72", list(range(0, 72)))


def _b(offset_m=0.0):
    if offset_m == 0.0:
        return coverage_product("b_60_24", list(range(60, 84)))
    return coverage_product("b_60_24_displaced_0_8", list(range(60, 84)), offset_m)


def _precedence(scope, verify=False):
    return sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope=scope,
        min_agree=1,
        position_tolerance_m=5.0,
        verify_continuity=sidereon.Sp3ContinuityOptions() if verify else None,
    )


def _merge(expected, sources, options):
    outcome, result = merged(sources, options)
    assert outcome == expected
    assert result is not None
    return result


def _gap(after, before, missing):
    return sidereon.Sp3CoverageGap(after, before, missing)


def test_a_hold_out_residual_is_attributed_to_every_node_its_prediction_used():
    case = coverage_case("hold_out_attribution")
    options = _precedence("cell", verify=True)
    assert options.verify_continuity.orbit_class == "meo_gnss"
    assert options.verify_continuity.residual_tolerance_m == 1.0
    merged_product, report = _merge(case["merge"], [_a(), _b(0.8)], options)

    start = float(merged_product.epochs_j2000_seconds[0])
    continuity = report.continuity
    assert not continuity.attested

    inside_b = [
        violation
        for violation in continuity.violations
        if violation.from_sources == [1] and violation.to_sources == [1]
    ]
    assert inside_b
    for violation in inside_b:
        defect = violation.defect
        assert defect.kind == "hold_out_residual"
        nodes = defect.node_epochs_j2000_s
        assert len(nodes) == 11
        assert all(b - a == 2.0 * STEP_S for a, b in zip(nodes, nodes[1:]))
        assert nodes[0] <= start + 71.0 * STEP_S

        assert violation.crosses_contributors
        assert violation.sources == [0, 1]
        assert len(violation.cells) == 12
        held_out = [cell for cell in violation.cells if cell.role == "held_out"]
        assert len(held_out) == 1
        assert held_out[0].epoch_j2000_s == defect.epoch_j2000_s
        assert held_out[0].selection.kind == "single_source"
        assert held_out[0].selection.selected_source == 1
        assert held_out[0].selection.members == [1]
        assert any(
            cell.role == "interpolation_node"
            and cell.selection is not None
            and cell.selection.kind == "precedence"
            and cell.selection.selected_source == 0
            and cell.selection.members == [0, 1]
            for cell in violation.cells
        )

    # The dict view of the same findings carries every field the typed one does.
    epoch = float(merged_product.epochs_j2000_seconds[-1])
    verdict = report.continuity_verdict(merged_product, epoch, epoch)
    typed_splices = continuity.splices
    assert len(verdict["all_splices"]) == len(typed_splices)
    for as_dict, typed in zip(verdict["all_splices"], typed_splices):
        assert as_dict["sources"] == typed.sources
        assert [cell["role"] for cell in as_dict["cells"]] == [
            cell.role for cell in typed.cells
        ]
        assert [cell["epoch_j2000_s"] for cell in as_dict["cells"]] == [
            cell.epoch_j2000_s for cell in typed.cells
        ]
        assert as_dict["defect"]["kind"] == typed.defect.kind
        if typed.defect.kind == "hold_out_residual":
            assert (
                as_dict["defect"]["node_epochs_j2000_s"]
                == typed.defect.node_epochs_j2000_s
            )
            assert as_dict["defect"]["residual_m"] == typed.defect.residual_m


def test_the_undisplaced_merge_attests():
    _merged, report = _merge(
        coverage_case("undisplaced"), [_a(), _b()], _precedence("cell", verify=True)
    )
    assert report.continuity.attested
    assert report.continuity.defects == []
    assert report.continuity.violations == []


def test_coverage_separates_positions_from_clocks_and_names_each_omitted_clock():
    case = coverage_case("cell_precedence_coverage")
    merged_product, report = _merge(
        {"product": case["product"], "report": case["report"]},
        [_a(), _b()],
        _precedence("cell"),
    )
    satellite_coverage = merged_product.satellite_coverage()
    assert coverage(satellite_coverage) == case["coverage"]

    assert merged_product.epoch_count == 84
    assert report.omitted_epochs == []
    assert report.arc_withheld == []
    assert report.continuity is None
    epochs = merged_product.epochs_j2000_seconds
    assert report.splices_influencing(float(epochs[0]), float(epochs[-1])) is None

    assert satellite_coverage.grid.interval_s == 300.0
    assert satellite_coverage.grid.agrees_with_header
    assert len(satellite_coverage.satellites) == 6
    instants = merged_product.epoch_instants
    for satellite in satellite_coverage.satellites:
        assert satellite.declared
        assert satellite.positions.epochs == 84
        assert satellite.positions.is_complete

        clocks = satellite.clocks
        assert clocks.epochs == 72
        assert len(clocks.spans) == 1
        assert clocks.spans[0].first_index == 0
        assert clocks.spans[0].last_index == 71
        assert clocks.last_epoch == instants[71]
        assert clocks.gaps == [_gap(71, None, 12)]

    assert len(report.clock_omissions) == 6 * 12
    for omission in report.clock_omissions:
        assert omission.reason == "datum_not_observable"
        assert omission.preferred_source is None
        assert omission.source == 1
        assert not omission.cell_has_clock
        assert instants.index(omission.epoch) >= 72


def test_satellite_arc_precedence_omits_and_reports_empty_epochs():
    case = coverage_case("satellite_arc_omits")
    a, b = _a(), _b()
    merged_product, report = _merge(
        {"product": case["product"], "report": case["report"]},
        [a, b],
        _precedence("satellite_arc"),
    )
    assert coverage(merged_product.satellite_coverage()) == case["coverage"]

    assert merged_product.epoch_count == 72
    assert merged_product.header.num_epochs == 72
    assert merged_product.epoch_instants == a.epoch_instants
    assert report.omitted_epochs == b.epoch_instants[12:]
    assert report.omitted_epochs_j2000_seconds == [
        float(value) for value in b.epochs_j2000_seconds[12:]
    ]
    assert len(report.arc_withheld) == 6 * 12
    for flag in report.arc_withheld:
        assert flag.sources == [1]
        assert flag.epoch in report.omitted_epochs
    assert len(report.clock_omissions) == 6 * 12
    for omission in report.clock_omissions:
        assert omission.reason == "datum_not_observable"
        assert omission.source == 1
        assert omission.epoch in report.omitted_epochs

    for satellite in merged_product.satellite_coverage().satellites:
        assert satellite.positions.is_complete
        assert satellite.clocks.is_complete
        assert satellite.positions.epochs == 72

    reread = sidereon.load_sp3(merged_product.to_sp3_string().encode("ascii"))
    assert reread.epoch_instants == merged_product.epoch_instants


def test_a_product_that_skips_an_epoch_merges_on_its_grid():
    skipping = coverage_product("skip_0_1_2_4_5", [0, 1, 2, 4, 5])
    merged_product, report = _merge(
        coverage_case("skipped_epoch"), [skipping], _precedence("cell")
    )
    assert merged_product.header.epoch_interval_s == 300.0
    assert merged_product.epoch_count == 5
    assert report.omitted_epochs == []


def _pattern(prn, index):
    if prn == 1:
        return FULL
    if prn == 2:
        return ABSENT if index in (2, 3) else FULL
    if prn == 3:
        return NO_CLOCK if index in (7, 8) else FULL
    if prn == 4:
        return CLOCK_ONLY if index in (0, 1) else FULL
    return ABSENT


def test_coverage_of_a_product_reports_spans_and_gaps_per_channel():
    # Epoch 5 is not in the product at all.
    sp3 = coverage_product(
        "patterned_coverage", [0, 1, 2, 3, 4, 6, 7, 8], record=_pattern
    )
    result = sp3.satellite_coverage()
    assert coverage(result) == coverage_case("product_coverage")["coverage"]

    assert result.grid.interval_s == 300.0
    assert result.grid.agrees_with_header
    assert result.grid.out_of_order == []
    assert result.grid.unplaced == []
    assert len(result.satellites) == 6
    by_prn = {
        int(satellite.satellite[1:]): satellite for satellite in result.satellites
    }

    g01 = by_prn[1]
    assert g01.positions.epochs == 8
    assert len(g01.positions.spans) == 2
    first_span = g01.positions.spans[0]
    assert (first_span.first_index, first_span.last_index) == (0, 4)
    assert first_span.epochs == 5
    assert g01.positions.gaps == [_gap(4, 5, 0)]
    assert not g01.positions.is_complete
    assert g01.clocks == g01.positions

    g02 = by_prn[2]
    assert g02.positions.epochs == 6
    assert g02.positions.gaps == [_gap(1, 4, 2), _gap(4, 5, 0)]

    g03 = by_prn[3]
    assert g03.positions.epochs == 8
    assert g03.clocks.epochs == 6
    assert g03.clocks.gaps[-1] == _gap(5, None, 2)

    g04 = by_prn[4]
    assert g04.positions.epochs == 6
    assert g04.clocks.epochs == 8
    assert g04.positions.gaps[0] == _gap(None, 2, 2)

    for prn in (5, 6):
        empty = by_prn[prn]
        assert empty.declared
        assert empty.positions.epochs == 0
        assert empty.positions.spans == []
        assert empty.positions.gaps == [_gap(None, None, 8)]
        assert empty.positions.first_epoch is None
        assert empty.clocks == empty.positions


def test_merge_window_verdicts_refuse_only_windows_that_use_the_handover():
    case = coverage_case("hold_out_attribution")
    merged_product, report = _merge(
        case["merge"], [_a(), _b(0.8)], _precedence("cell", verify=True)
    )
    epochs = [float(value) for value in merged_product.epochs_j2000_seconds]
    assert len(epochs) == 84

    decisions = []
    for index, epoch in enumerate(epochs):
        verdict = report.continuity_verdict(merged_product, epoch, epoch)
        typed = report.continuity.verdict(epoch, epoch)
        assert verdict == typed
        expected = "refuse" if index >= 68 else "accept"
        assert verdict["decision"] == expected, index
        decisions.append(verdict["decision"])
    assert decisions == case["merge_single_epoch_decisions"]

    early = report.continuity_verdict(merged_product, epochs[51], epochs[66])
    assert early["decision"] == "accept"
    reaching = report.continuity_verdict(merged_product, epochs[51], epochs[68])
    assert reaching["decision"] == "refuse"
    assert reaching["influencing_splices"]
    splices = report.splices_influencing(epochs[51], epochs[68])
    assert len(splices) == len(reaching["influencing_splices"])
    assert all(splice.crosses_contributors for splice in splices)
    for expected in case["range_verdicts"]:
        first, last = expected["from_index"], expected["through_index"]
        verdict = report.continuity_verdict(merged_product, epochs[first], epochs[last])
        assert verdict["decision"] == expected["decision"]
        assert len(verdict["influencing_defects"]) == expected["influencing_defects"]
        assert len(verdict["influencing_splices"]) == expected["influencing_splices"]

    # The same findings checked on the product alone bound a window's reach
    # by the stencil extent: single-epoch windows are refused from epoch 69.
    before_s, after_s = merged_product.stencil_extent()
    assert before_s == 3_300.0
    assert f64_hex(before_s) == case["stencil_before_s"]
    assert f64_hex(after_s) == case["stencil_after_s"]
    plain = []
    for index, epoch in enumerate(epochs):
        verdict = merged_product.continuity_verdict(epoch, epoch)
        expected = "refuse" if index >= 69 else "accept"
        assert verdict["decision"] == expected, index
        plain.append(verdict["decision"])
    assert plain == case["plain_single_epoch_decisions"]


def test_interpolation_nodes_select_exactly_what_the_window_interpolates():
    case = coverage_case("hold_out_attribution")
    merged_product, report = _merge(
        case["merge"], [_a(), _b(0.8)], _precedence("cell", verify=True)
    )
    windows = [
        (entry["from_index"], entry["through_index"]) for entry in case["report_nodes"]
    ]
    product_nodes = sidereon.Sp3InterpolationNodes.for_sp3(merged_product)
    assert report.continuity.nodes == product_nodes
    assert (
        node_selection(report.continuity.nodes, merged_product, windows)
        == case["report_nodes"]
    )
    assert (
        node_selection(product_nodes, merged_product, windows) == case["product_nodes"]
    )

    epochs = merged_product.epochs_j2000_seconds
    with pytest.raises(ValueError, match="start must not follow its end"):
        product_nodes.selected_nodes("G01", float(epochs[1]), float(epochs[0]))
    with pytest.raises(ValueError, match="invalid satellite token"):
        product_nodes.selected_nodes(
            "not a satellite", float(epochs[0]), float(epochs[0])
        )


def test_a_reversed_satellite_arc_merge_starting_at_five_writes_and_reads_back():
    a, b = _a(), _b()
    merged_product, report = _merge(
        coverage_case("reversed_satellite_arc"), [b, a], _precedence("satellite_arc")
    )
    assert merged_product.epoch_instants == b.epoch_instants
    assert report.omitted_epochs == a.epoch_instants[:60]
    assert merged_product.header.mjd == 59025
    assert merged_product.header.mjd_fraction == 0.208_333_333_333_3
    assert merged_product.header.seconds_of_week == 363_600.0

    text = merged_product.to_sp3_string()
    assert text.startswith("#cP2020  6 25  5  0  0.00000000"), text
    assert "\n## 2111 363600.00000000   300.00000000 59025 0.2083333333333\n" in text
    reread = sidereon.load_sp3(text.encode("ascii"))
    assert reread.epoch_instants == merged_product.epoch_instants
    assert reread.header.mjd_fraction == merged_product.header.mjd_fraction


def test_a_merge_starting_at_five_writes():
    merged_product, _report = _merge(
        coverage_case("single_starting_at_five"), [_b()], _precedence("cell")
    )
    reread = sidereon.load_sp3(merged_product.to_sp3_string().encode("ascii"))
    assert reread.epoch_instants == merged_product.epoch_instants


def test_a_merge_that_accepts_no_cell_writes_an_epochless_product():
    indices = list(range(60, 64))
    near = coverage_product(
        "near_no_clock_60_4", indices, record=lambda _p, _i: NO_CLOCK
    )
    far = coverage_product(
        "far_no_clock_60_4", indices, 1_000.0, record=lambda _p, _i: NO_CLOCK
    )
    merged_product, report = _merge(
        coverage_case("epochless"), [near, far], sidereon.Sp3MergeOptions(min_agree=2)
    )

    assert merged_product.epoch_count == 0
    assert len(report.omitted_epochs) == 4
    assert len(report.quarantined) == 4 * 6
    start = float(near.epochs_j2000_seconds[0])
    assert merged_product.declared_start_j2000_s == start
    assert merged_product.header.mjd_fraction == 0.208_333_333_333_3

    text = merged_product.to_sp3_string()
    assert text.startswith("#cP2020  6 25  5  0  0.00000000       0 "), text
    reread = sidereon.load_sp3(text.encode("ascii"))
    assert reread.epoch_count == 0
    assert reread.declared_start_j2000_s == start
    assert reread.to_sp3_string() == text


def test_the_default_grid_holds_every_epoch_of_phase_offset_inputs():
    even = coverage_product("even_0_24", list(range(0, 24, 2)))
    odd = coverage_product("odd_1_24", list(range(1, 24, 2)))
    merged_product, report = _merge(
        coverage_case("phase_offset"), [even, odd], _precedence("cell")
    )
    assert merged_product.header.epoch_interval_s == 300.0
    assert merged_product.epoch_count == 24
    assert report.dropped_input_epochs == []


def test_an_explicit_target_reports_every_input_epoch_it_drops():
    twelve = coverage_product("a_0_12", list(range(0, 12)))
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        min_agree=1,
        position_tolerance_m=5.0,
        target_epoch_interval_s=900.0,
    )
    merged_product, report = _merge(coverage_case("explicit_target"), [twelve], options)
    assert merged_product.epoch_count == 4
    assert len(report.dropped_input_epochs) == 8
    for dropped in report.dropped_input_epochs:
        assert dropped.source == 0
        assert dropped.epoch_index % 3 != 0
        assert dropped.reason == "off_target_grid"
        assert dropped.epoch == twelve.epoch_instants[dropped.epoch_index]


def test_a_fractional_target_on_the_tick_axis_is_merged_not_refused():
    # 450.5 s is a whole number of the 10-nanosecond ticks an SP3 interval
    # states, so the core merges on it; only the 00:00 epoch lies on that grid.
    twelve = coverage_product("a_0_12", list(range(0, 12)))
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        min_agree=1,
        position_tolerance_m=5.0,
        target_epoch_interval_s=450.5,
    )
    assert options.target_epoch_interval_s == 450.5
    merged_product, report = _merge(
        coverage_case("fractional_target"), [twelve], options
    )
    assert merged_product.header.epoch_interval_s == 450.5
    assert merged_product.epoch_count + len(report.dropped_input_epochs) == 12
    assert all(
        dropped.reason == "off_target_grid" for dropped in report.dropped_input_epochs
    )


def test_invalid_target_interval_reports_typed_json_safe_core_evidence():
    product = coverage_product("a_0_12", list(range(0, 12)))
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        min_agree=1,
        target_epoch_interval_s=1.0e-9,
    )

    with pytest.raises(ValueError) as caught:
        sidereon.merge_sp3([product], options)

    assert caught.value.kind == "sp3_epoch_interval"
    assert caught.value.field == "target_epoch_interval_s"
    assert caught.value.value == "1e-09" or caught.value.value == "1e-9"

    with pytest.raises(ValueError) as tolerance_error:
        sidereon.Sp3MergeOptions(position_tolerance_m=-1.0)
    assert tolerance_error.value.kind == "sp3_merge_tolerance"
    assert tolerance_error.value.field == "Position"
    assert tolerance_error.value.value == "-1.0"


def test_inputs_half_a_second_apart_are_separate_epochs():
    whole = coverage_product("a_0_1", [0])
    text = whole.to_sp3_string().replace("25  0  0  0.00000000", "25  0  0  0.50000000")
    assert (
        fnv1a64(text.encode("ascii")) == COVERAGE["texts_fnv1a64"]["a_0_1_half_second"]
    )
    half = sidereon.load_sp3(text.encode("ascii"))
    merged_product, report = _merge(
        coverage_case("half_second_apart"), [whole, half], _precedence("cell")
    )
    assert merged_product.epoch_count == 2
    assert merged_product.header.epoch_interval_s == 0.5
    assert report.dropped_input_epochs == []
    written = merged_product.to_sp3_string()
    assert "*  2020  6 25  0  0  0.00000000\n" in written
    assert "*  2020  6 25  0  0  0.50000000\n" in written


def test_each_source_clock_left_out_of_a_clocked_cell_is_reported():
    indices = list(range(0, 12))
    first = coverage_product("clock_a_0_12", indices)
    third = coverage_product(
        "clock_c_0_12", indices, record=lambda prn, _i: FULL if prn <= 4 else NO_CLOCK
    )
    merged_product, report = _merge(
        coverage_case("clock_omissions"),
        [first, first, third],
        sidereon.Sp3MergeOptions(),
    )
    assert merged_product.epoch_count == 12
    assert len(report.clock_omissions) == 12 * 4
    for omission in report.clock_omissions:
        assert omission.source == 2
        assert omission.reason == "datum_not_observable"
        assert omission.preferred_source is None
        assert omission.cell_has_clock
        assert int(omission.satellite[1:]) <= 4


def test_a_gapped_product_merges_only_on_its_declared_grid():
    case = coverage_case("gapped_off_declared_grid")
    off_grid = coverage_product(
        "gapped_declared_600", [0, 1, 2, 4, 5], interval_s=600.0
    )
    outcome, result = merged([off_grid], _precedence("cell"))
    assert outcome == case["merge"]
    assert result is None
    assert "lie on no grid" in outcome["error"]
    off_grid_coverage = off_grid.satellite_coverage()
    assert coverage(off_grid_coverage) == case["coverage"]
    assert off_grid_coverage.grid.interval_s is None
    assert not off_grid_coverage.grid.agrees_with_header

    case = coverage_case("uniform_wrong_header")
    wrong_header = coverage_product(
        "uniform_declared_900", list(range(0, 4)), interval_s=900.0
    )
    merged_product, _report = _merge(case["merge"], [wrong_header], _precedence("cell"))
    assert merged_product.header.epoch_interval_s == 300.0
    wrong_coverage = wrong_header.satellite_coverage()
    assert coverage(wrong_coverage) == case["coverage"]
    assert wrong_coverage.grid.interval_s == 300.0
    assert not wrong_coverage.grid.agrees_with_header
    assert wrong_coverage.satellites[0].positions.is_complete


def test_coverage_with_a_zero_interval():
    gapped = coverage_product("gapped_declared_0", [0, 1, 2, 4, 5], interval_s=0.0)
    result = gapped.satellite_coverage()
    assert coverage(result) == coverage_case("zero_interval_gapped")["coverage"]
    assert result.grid.interval_s is None
    assert not result.grid.agrees_with_header
    assert len(result.satellites[0].positions.spans) == 1
    assert result.satellites[0].positions.is_complete

    uniform = coverage_product("uniform_declared_0", list(range(0, 4)), interval_s=0.0)
    result = uniform.satellite_coverage()
    assert coverage(result) == coverage_case("zero_interval_uniform")["coverage"]
    assert result.grid.interval_s == 300.0
    assert not result.grid.agrees_with_header


def test_coverage_reports_epochs_out_of_order():
    sp3 = coverage_product("out_of_order_0_2_1_3", [0, 2, 1, 3])
    result = sp3.satellite_coverage()
    assert coverage(result) == coverage_case("out_of_order")["coverage"]
    assert result.grid.out_of_order == [2]
    assert result.grid.interval_s is None
    g01 = result.satellites[0].positions
    assert len(g01.spans) == 2
    assert g01.gaps == [_gap(1, 2, 0)]


def test_continuity_and_provenance_options_refuse_unknown_labels():
    with pytest.raises(ValueError, match="unknown orbit class"):
        sidereon.Sp3ContinuityOptions(orbit_class="heliocentric")
    with pytest.raises(ValueError, match="unknown SP3 merge provenance mode"):
        sidereon.Sp3MergeOptions(provenance="everything")

    disabled = sidereon.Sp3ContinuityOptions(
        orbit_class=None, residual_tolerance_m=None, gap_threshold_factor=2.0
    )
    assert disabled.orbit_class is None
    assert disabled.residual_tolerance_m is None
    assert disabled.gap_threshold_factor == 2.0
    options = sidereon.Sp3MergeOptions(
        verify_continuity=disabled, provenance=sidereon.Sp3ProvenanceMode.SUMMARY
    )
    assert options.verify_continuity.gap_threshold_factor == 2.0
    assert options.provenance == sidereon.Sp3ProvenanceMode.SUMMARY
