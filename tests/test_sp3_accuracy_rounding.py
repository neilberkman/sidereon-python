"""Public SP3 accuracy-rounding writer contract on synthetic records."""

from pathlib import Path

import pytest
import sidereon
import sidereon.data as data

_FIXTURES = Path(__file__).parent / "fixtures" / "sp3" / "accuracy_rounding"


def _load(name):
    return sidereon.load_sp3((_FIXTURES / name).read_bytes())


def _replace_position_x_exponent(text, satellite, exponent):
    suffix = f" {str(exponent).rjust(2)} {'':2} {'':2} {'':3} "
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"P{satellite}"):
            lines[index] = line[:60] + suffix
    return "\n".join(lines) + "\n"


def _replace_codes(text, satellite, *, position=None, velocity=None):
    def code(value, width):
        return " " * width if value is None else f"{value:>{width}d}"

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if position is not None and line.startswith(f"P{satellite}"):
            lines[index] = (
                line[:60]
                + " "
                + f"{code(position[0], 2)} {code(position[1], 2)} "
                + f"{code(position[2], 2)} {code(position[3], 3)}  "
            )
        elif velocity is not None and line.startswith(f"V{satellite}"):
            lines[index] = (
                line[:60]
                + " "
                + f"{code(velocity[0], 2)} {code(velocity[1], 2)} "
                + f"{code(velocity[2], 2)} {code(velocity[3], 3)}  "
            )
    return "\n".join(lines) + "\n"


def _replace_bases(text, *, position=None, clock=None):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("%f ") and line.split()[0] == "%f":
            fields = line.split()
            if position is not None:
                fields[1] = f"{position:.7f}"
            if clock is not None:
                fields[2] = f"{clock:.9f}"
            lines[index] = f"%f {fields[1]:>10} {fields[2]:>12} {fields[3]} {fields[4]}"
            break
    return "\n".join(lines) + "\n"


def _options():
    return sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        min_agree=1,
        clock_min_common=1,
        position_tolerance_m=5.0,
        clock_tolerance_s=50e-9,
        outlier_reject=sidereon.Sp3OutlierRejectOptions(
            position_tolerance_m=5.0, clock_tolerance_s=50e-9
        ),
        target_epoch_interval_s=300.0,
        systems=["G"],
    )


@pytest.mark.parametrize("order", [("A.sp3", "B.sp3"), ("B.sp3", "A.sp3")])
def test_accuracy_rounding_writer_preserves_decoded_sigmas_after_merge(order):
    sources = [_load(name) for name in order]
    for source in sources:
        assert isinstance(source.to_sp3_string(), str)

    merged, _ = sidereon.merge_sp3(sources, _options())
    text, report = merged.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    reread = sidereon.load_sp3(text.encode("ascii"))

    assert isinstance(report, sidereon.Sp3DetailedRoundingReport)
    assert isinstance(report.record_values, sidereon.Sp3RoundingReport)
    assert isinstance(report.accuracy_sigmas, sidereon.Sp3AccuracyRoundingReport)
    assert (
        tuple(
            getattr(report.accuracy_sigmas, field)
            for field in (
                "position_sigma_x_m",
                "position_sigma_y_m",
                "position_sigma_z_m",
                "velocity_sigma_x_m_s",
                "velocity_sigma_y_m_s",
                "velocity_sigma_z_m_s",
                "clock_sigma_m",
                "clock_rate_sigma_m_s",
            )
        )
        == (0.0,) * 8
    )
    for field in (
        "position_sigma_x_m",
        "position_sigma_y_m",
        "position_sigma_z_m",
        "velocity_sigma_x_m_s",
        "velocity_sigma_y_m_s",
        "velocity_sigma_z_m_s",
        "clock_sigma_m",
        "clock_rate_sigma_m_s",
    ):
        assert getattr(report.accuracy_sigmas, field) >= 0.0

    original = merged.record_accuracy("G12", 0).p.position_sigma_m[0]
    restored = reread.record_accuracy("G12", 0).p.position_sigma_m[0]
    assert restored.kind == original.kind == "known"
    assert original.value == pytest.approx(0.018189894035458565, rel=1e-15)
    assert restored.value >= original.value
    assert reread.state("G12", 0).position_m == merged.state("G12", 0).position_m
    assert reread.record_accuracy("G01", 0).p.position_sigma_m[0].kind == "unknown"


@pytest.mark.parametrize("reverse", [False, True])
def test_equal_sigmas_with_compatible_defined_bases_write_in_either_order(reverse):
    a_text = _replace_bases(
        _load("A-anchor.sp3").to_sp3_string(), position=2.0, clock=2.0
    )
    a_text = _replace_codes(a_text, "G01", position=(2, None, None, 2))
    b_text = _replace_bases(
        _load("B-anchor.sp3").to_sp3_string(), position=4.0, clock=4.0
    )
    b_text = _replace_codes(b_text, "G12", position=(1, None, None, 1))
    a = sidereon.load_sp3(a_text.encode("ascii"))
    b = sidereon.load_sp3(b_text.encode("ascii"))
    a_roundtrip = sidereon.load_sp3(a.to_sp3_string().encode("ascii"))
    b_roundtrip = sidereon.load_sp3(b.to_sp3_string().encode("ascii"))
    for source, restored, satellite in (
        (a, a_roundtrip, "G01"),
        (b, b_roundtrip, "G12"),
    ):
        source_accuracy = source.record_accuracy(satellite, 0).p
        restored_accuracy = restored.record_accuracy(satellite, 0).p
        assert restored_accuracy.position_sigma_m == source_accuracy.position_sigma_m
        assert restored_accuracy.clock_sigma_m == source_accuracy.clock_sigma_m
    sources = [b, a] if reverse else [a, b]
    merged, _ = sidereon.merge_sp3(sources, _options())

    strict = sidereon.load_sp3(merged.to_sp3_string().encode("ascii"))
    rounded_text, report = merged.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    rounded = sidereon.load_sp3(rounded_text.encode("ascii"))
    assert rounded.to_sp3_string() == strict.to_sp3_string()
    assert (
        tuple(
            getattr(report.accuracy_sigmas, field)
            for field in (
                "position_sigma_x_m",
                "position_sigma_y_m",
                "position_sigma_z_m",
                "velocity_sigma_x_m_s",
                "velocity_sigma_y_m_s",
                "velocity_sigma_z_m_s",
                "clock_sigma_m",
                "clock_rate_sigma_m_s",
            )
        )
        == (0.0,) * 8
    )
    for satellite in ("G01", "G12"):
        source = merged.record_accuracy(satellite, 0).p
        output = rounded.record_accuracy(satellite, 0).p
        assert output.position_sigma_m == source.position_sigma_m
        assert output.clock_sigma_m == source.clock_sigma_m
    assert (
        merged.record_accuracy("G01", 0).p.position_sigma_m[0].value
        == merged.record_accuracy("G12", 0).p.position_sigma_m[0].value
    )
    assert (
        merged.record_accuracy("G01", 0).p.clock_sigma_m.value
        == merged.record_accuracy("G12", 0).p.clock_sigma_m.value
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_accuracy_rounding_reports_a_nonzero_conservative_sigma_increase(reverse):
    first = (
        (_FIXTURES / "A.sp3")
        .read_text("ascii")
        .replace("%f  0.0000000", "%f  2.0000000", 1)
    )
    first = _replace_position_x_exponent(first, "G01", 1)
    a = sidereon.load_sp3(first.encode("ascii"))
    b = _load("B.sp3")
    merged, _ = sidereon.merge_sp3([b, a] if reverse else [a, b], _options())

    text, report = merged.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    reread = sidereon.load_sp3(text.encode("ascii"))
    sigma = reread.record_accuracy("G12", 0).p.position_sigma_m[0]

    if not reverse:
        assert reread.header.pos_vel_base == 2.0
        assert sigma.value == pytest.approx(0.032, rel=0.0, abs=0.0)
        assert report.accuracy_sigmas.position_sigma_x_m == pytest.approx(
            0.013810105964541435, rel=1e-14
        )
    else:
        before = merged.record_accuracy("G01", 0).p.position_sigma_m[0]
        after = reread.record_accuracy("G01", 0).p.position_sigma_m[0]
        assert after.kind == before.kind == "known"
        assert after.value > before.value
        assert report.accuracy_sigmas.position_sigma_x_m == pytest.approx(
            after.value - before.value, rel=1e-14
        )


def test_accuracy_rounding_refuses_when_no_conservative_exponent_fits():
    a_text = (
        (_FIXTURES / "A.sp3")
        .read_text("ascii")
        .replace("%f  0.0000000", "%f  1.2500000", 1)
    )
    a_text = _replace_position_x_exponent(a_text, "G01", 1)
    b_text = (
        (_FIXTURES / "B.sp3")
        .read_text("ascii")
        .replace("%f  1.2500000", "%f  2.0000000", 1)
    )
    b_text = _replace_position_x_exponent(b_text, "G12", 98)
    sources = [
        sidereon.load_sp3(a_text.encode("ascii")),
        sidereon.load_sp3(b_text.encode("ascii")),
    ]
    for source in sources:
        assert isinstance(source.to_sp3_string(), str)
    merged, _ = sidereon.merge_sp3(sources, _options())

    with pytest.raises(sidereon.Sp3WriteError) as excinfo:
        merged.to_sp3_string_with_accuracy_rounding(
            sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
        )
    assert excinfo.value.detail.kind == "AccuracyNotRepresentable"


def test_accuracy_report_maps_all_position_velocity_clock_and_rate_fields():
    direct = _load("B-velocity.sp3")
    direct_text, direct_report = direct.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    direct_reread = sidereon.load_sp3(direct_text.encode("ascii"))
    report_fields = (
        "position_sigma_x_m",
        "position_sigma_y_m",
        "position_sigma_z_m",
        "velocity_sigma_x_m_s",
        "velocity_sigma_y_m_s",
        "velocity_sigma_z_m_s",
        "clock_sigma_m",
        "clock_rate_sigma_m_s",
    )
    assert (
        tuple(getattr(direct_report.accuracy_sigmas, field) for field in report_fields)
        == (0.0,) * 8
    )
    direct_source = direct.record_accuracy("G12", 0)
    direct_output = direct_reread.record_accuracy("G12", 0)
    assert direct_source.v is not None and direct_output.v is not None
    assert all(value.kind == "known" for value in direct_source.v.velocity_sigma_m_s)
    assert all(value.kind == "known" for value in direct_output.v.velocity_sigma_m_s)
    assert direct_output.v.clock_rate_sigma_m_s.kind == "known"

    merged, _ = sidereon.merge_sp3(
        [_load("A-velocity.sp3"), _load("B-velocity.sp3")], _options()
    )
    text, report = merged.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    reread = sidereon.load_sp3(text.encode("ascii"))
    source = merged.record_accuracy("G12", 0)
    output = reread.record_accuracy("G12", 0)
    assert source.p is not None and output.p is not None
    assert source.v is None and output.v is None
    position_changes = tuple(
        after.value - before.value
        for before, after in zip(source.p.position_sigma_m, output.p.position_sigma_m)
    )
    clock_change = output.p.clock_sigma_m.value - source.p.clock_sigma_m.value
    assert report.accuracy_sigmas.position_sigma_x_m == position_changes[0] > 0.0
    assert report.accuracy_sigmas.position_sigma_y_m == position_changes[1] > 0.0
    assert report.accuracy_sigmas.position_sigma_z_m == position_changes[2] > 0.0
    assert report.accuracy_sigmas.clock_sigma_m == clock_change > 0.0
    assert (
        tuple(
            getattr(report.accuracy_sigmas, field)
            for field in report_fields[3:6] + ("clock_rate_sigma_m_s",)
        )
        == (0.0,) * 4
    )
    assert tuple(value.value for value in output.p.position_sigma_m) == (0.032,) * 3
    assert output.p.clock_sigma_m.value == 2.0 * (299_792_458.0 * 1e-12)


def test_accuracy_rounding_preserves_clock_anchor_uncertainty():
    merged, _ = sidereon.merge_sp3(
        [_load("A-anchor.sp3"), _load("B-anchor.sp3")], _options()
    )
    text, report = merged.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    reread = sidereon.load_sp3(text.encode("ascii"))

    source_accuracy = merged.record_accuracy("G12", 0).p.clock_sigma_m
    output_accuracy = reread.record_accuracy("G12", 0).p.clock_sigma_m
    assert source_accuracy.kind == output_accuracy.kind == "known"
    assert output_accuracy.value >= source_accuracy.value
    assert reread.state("G12", 0).clock_s == merged.state("G12", 0).clock_s
    assert report.accuracy_sigmas.clock_sigma_m >= 0.0


def test_accuracy_rounding_respects_i2_i3_finite_lower_bounds_and_sentinels():
    a_text = _replace_bases(_load("A-anchor.sp3").to_sp3_string(), position=2.0)
    a_text = _replace_codes(a_text, "G01", position=(-9, None, None, None))
    b_text = _replace_bases(_load("B-anchor.sp3").to_sp3_string(), position=4.0)
    b_text = _replace_codes(b_text, "G12", position=(-5, None, None, None))
    lower_axis, _ = sidereon.merge_sp3(
        [
            sidereon.load_sp3(a_text.encode("ascii")),
            sidereon.load_sp3(b_text.encode("ascii")),
        ],
        _options(),
    )
    axis_text, axis_report = lower_axis.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    axis_codes = (
        sidereon.load_sp3(axis_text.encode("ascii")).record_accuracy_codes("G12", 0).p
    )
    assert axis_codes.axis_exponents[0] == -9
    assert axis_report.accuracy_sigmas.position_sigma_x_m > 0.0

    a_clock = _replace_bases(_load("A-anchor.sp3").to_sp3_string(), clock=2.0)
    a_clock = _replace_codes(a_clock, "G01", position=(None, None, None, -99))
    b_clock = _replace_bases(_load("B-anchor.sp3").to_sp3_string(), clock=4.0)
    b_clock = _replace_codes(b_clock, "G12", position=(None, None, None, -50))
    lower_clock, _ = sidereon.merge_sp3(
        [
            sidereon.load_sp3(a_clock.encode("ascii")),
            sidereon.load_sp3(b_clock.encode("ascii")),
        ],
        _options(),
    )
    clock_text, clock_report = lower_clock.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    clock_codes = (
        sidereon.load_sp3(clock_text.encode("ascii")).record_accuracy_codes("G12", 0).p
    )
    assert clock_codes.clock_exponent == -99
    assert clock_report.accuracy_sigmas.clock_sigma_m > 0.0

    a_clock = _replace_bases(_load("A-anchor.sp3").to_sp3_string(), clock=1.025)
    a_clock = _replace_codes(a_clock, "G01", position=(None, None, None, 1))
    b_clock = _replace_bases(_load("B-anchor.sp3").to_sp3_string(), clock=2.0)
    b_clock = _replace_codes(b_clock, "G12", position=(None, None, None, 998))
    upper_clock, _ = sidereon.merge_sp3(
        [
            sidereon.load_sp3(a_clock.encode("ascii")),
            sidereon.load_sp3(b_clock.encode("ascii")),
        ],
        _options(),
    )
    with pytest.raises(sidereon.Sp3WriteError) as excinfo:
        upper_clock.to_sp3_string_with_accuracy_rounding(
            sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
        )
    assert excinfo.value.detail.kind == "AccuracyNotRepresentable"


def test_accuracy_rounding_preserves_sentinels_and_refuses_invalid_bases():
    text = _replace_codes(
        _replace_bases(_load("B.sp3").to_sp3_string(), position=2.0, clock=2.0),
        "G12",
        position=(99, 99, 99, 999),
    )
    sentinel = sidereon.load_sp3(text.encode("ascii"))
    rounded, report = sentinel.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    codes = sidereon.load_sp3(rounded.encode("ascii")).record_accuracy_codes("G12", 0).p
    assert tuple(codes.axis_exponents) == (99, 99, 99)
    assert codes.clock_exponent == 999
    assert (
        tuple(
            getattr(report.accuracy_sigmas, name)
            for name in (
                "position_sigma_x_m",
                "position_sigma_y_m",
                "position_sigma_z_m",
                "velocity_sigma_x_m_s",
                "velocity_sigma_y_m_s",
                "velocity_sigma_z_m_s",
                "clock_sigma_m",
                "clock_rate_sigma_m_s",
            )
        )
        == (0.0,) * 8
    )

    invalid_text = _replace_codes(
        _replace_bases(_load("B.sp3").to_sp3_string(), position=0.0),
        "G12",
        position=(1, 99, 99, 999),
    )
    invalid = sidereon.load_sp3(invalid_text.encode("ascii"))
    assert (
        invalid.record_accuracy("G12", 0).p.position_sigma_m[0].kind == "invalid_base"
    )
    invalid_output, invalid_report = invalid.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    invalid_codes = (
        sidereon.load_sp3(invalid_output.encode("ascii"))
        .record_accuracy_codes("G12", 0)
        .p
    )
    assert invalid_codes.axis_exponents[0] == 1
    assert invalid_report.accuracy_sigmas.position_sigma_x_m == 0.0

    invalid_source = _replace_codes(
        _load("A-anchor.sp3").to_sp3_string(), "G01", position=(13, 13, 13, 13)
    )
    known_source = _replace_codes(
        _load("B-anchor.sp3").to_sp3_string(), "G12", position=(13, 13, 13, 13)
    )
    for source_texts in (
        (invalid_source, known_source),
        (known_source, invalid_source),
    ):
        merged, _ = sidereon.merge_sp3(
            [sidereon.load_sp3(source.encode("ascii")) for source in source_texts],
            _options(),
        )
        with pytest.raises(sidereon.Sp3WriteError) as excinfo:
            merged.to_sp3_string_with_accuracy_rounding(
                sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
            )
        assert excinfo.value.detail.kind == "AccuracyNotRepresentable"


def test_accuracy_rounding_requires_a_known_policy_and_data_writer_is_atomic(tmp_path):
    sp3 = _load("B.sp3")

    with pytest.raises(TypeError):
        sp3.to_sp3_string_with_accuracy_rounding()
    with pytest.raises(ValueError, match="unknown SP3 accuracy rounding policy"):
        sp3.to_sp3_string_with_accuracy_rounding("lossy")

    path = tmp_path / "rounded.sp3.gz"
    written, report = data.write_sp3_with_accuracy_rounding(
        sp3,
        str(path),
        policy=sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE,
        gzip=True,
    )
    assert written == str(path)
    assert path.is_file()
    assert isinstance(report, sidereon.Sp3DetailedRoundingReport)
