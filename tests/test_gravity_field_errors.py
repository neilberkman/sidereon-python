import pytest
import sidereon as s

HEADER = """product_type gravity_field
modelname TEST
earth_gravity_constant 3.986e14
radius 6378000
max_degree 2
errors no
tide_system tide_free
end_of_head
"""
RECORDS = "gfc 2 0 -4.8e-4 0\ngfc 2 1 0 0\ngfc 2 2 0 0\n"


@pytest.mark.parametrize(
    "header,records,detail",
    [
        (
            HEADER.replace("radius 6378000", "radius"),
            RECORDS,
            dict(kind="missing_field", line=4, keyword="radius", field="value"),
        ),
        (
            HEADER.replace("radius 6378000", "radius bad"),
            RECORDS,
            dict(
                kind="invalid_field",
                line=4,
                keyword="radius",
                field="value",
                value="bad",
            ),
        ),
        (
            HEADER.replace("radius 6378000\n", ""),
            RECORDS,
            dict(kind="missing_header_keyword", keyword="radius"),
        ),
        (
            HEADER.replace("radius 6378000", "radius 6378000\nradius 6378000"),
            RECORDS,
            dict(
                kind="repeated_header_keyword", line=5, first_line=4, keyword="radius"
            ),
        ),
        (
            HEADER.replace("end_of_head", "format icgem2.0\nend_of_head"),
            RECORDS + "dot 2 0 1e-7 0\n",
            dict(kind="keyword_not_in_format", line=13, keyword="dot"),
        ),
        (
            HEADER,
            RECORDS.replace("gfc", "gcf", 1),
            dict(kind="unknown_record_keyword", line=9, keyword="gcf"),
        ),
        (
            HEADER,
            RECORDS.replace("gfc ", "", 1),
            dict(kind="missing_record_keyword", line=9),
        ),
        (
            HEADER,
            RECORDS.replace("gfc 2 0", "gfc 3 0"),
            dict(kind="degree_above_maximum", line=9, degree=3, max_degree=2),
        ),
        (
            HEADER,
            RECORDS.replace("gfc 2 0", "gfc 2 3"),
            dict(kind="order_above_degree", line=9, degree=2, order=3),
        ),
        (
            HEADER,
            RECORDS + "gfc 2 0 0 0\n",
            dict(kind="repeated_coefficient", line=12, first_line=9, degree=2, order=0),
        ),
        (
            HEADER,
            RECORDS.replace("gfc 2 2 0 0", "comment"),
            dict(kind="missing_coefficient", degree=2, order=2),
        ),
        (HEADER, "", dict(kind="too_few_records", max_degree=2, required=3, lines=0)),
        (
            HEADER,
            RECORDS + "trnd 2 0 1e-7 0\n",
            dict(kind="uncovered_term", line=12, keyword="trnd", degree=2, order=0),
        ),
        (
            HEADER,
            RECORDS.replace("gfc 2 0 -4.8e-4 0", "gfct 2 0 -4.8e-4 0 20000101")
            + "trnd 2 0 1e-7 0\ntrnd 2 0 2e-7 0\n",
            dict(
                kind="overlapping_term",
                line=13,
                first_line=12,
                keyword="trnd",
                degree=2,
                order=0,
            ),
        ),
    ],
    ids=lambda value: value["kind"] if isinstance(value, dict) else None,
)
def test_icgem_parser_refusals(header, records, detail):
    with pytest.raises(s.IcgemError) as caught:
        s.IcgemGravityField.parse(header + records)
    assert caught.value.detail == detail


def test_icgem_normalization_refusal():
    header = HEADER.replace("max_degree 2", "max_degree 170").replace(
        "end_of_head", "norm unnormalized\nend_of_head"
    )
    records = [
        f"gfc {degree} {order} {int(degree == order == 170)} 0"
        for degree in range(2, 171)
        for order in range(degree + 1)
    ]
    with pytest.raises(s.IcgemError) as caught:
        s.IcgemGravityField.parse(header + "\n".join(records))
    assert caught.value.detail == {
        "kind": "normalization_out_of_range",
        "line": 14712,
        "degree": 170,
        "order": 170,
    }


def test_icgem_coefficient_refusals():
    text = HEADER + RECORDS
    with pytest.raises(s.IcgemError) as caught:
        s.IcgemGravityField.parse_truncated(text, 3, 2)
    assert caught.value.detail == {
        "kind": "truncation_out_of_range",
        "degree": 3,
        "order": 2,
        "max_degree": 2,
    }

    field = s.IcgemGravityField.parse(text.replace("tide_system tide_free\n", ""))
    with pytest.raises(s.IcgemError) as caught:
        field.gravity_in_tide_system(2, 2, s.TideSystem.ZERO_TIDE)
    assert caught.value.detail == {"kind": "unknown_tide_system"}

    variable = (
        HEADER.replace("end_of_head", "format icgem2.0\nend_of_head")
        + RECORDS.replace(
            "gfc 2 0 -4.8e-4 0", "gfct 2 0 -4.8e-4 0 20000101.0000 20300101.0000"
        )
        + "acos 2 0 1e-11 0 20000101.0000 20300101.0000 1e-320\n"
    )
    field = s.IcgemGravityField.parse(variable)
    for epoch, kind in [
        (-43201.0, "epoch_outside_validity"),
        (-43199.0, "non_finite_coefficient"),
    ]:
        with pytest.raises(s.IcgemError) as caught:
            field.coefficient_at(2, 0, epoch)
        assert caught.value.detail == {
            "kind": kind,
            "degree": 2,
            "order": 0,
            "epoch_tt_seconds": epoch,
        }


def test_icgem_declared_sigmas_and_normalization():
    header = HEADER.replace("errors no", "errors calibrated_and_formal").replace(
        "end_of_head", "norm unnormalized\nend_of_head"
    )
    field = s.IcgemGravityField.parse(
        header + "gfc 2 0 0 0 1 2 3 4\ngfc 2 1 0 0 0 0 0 0\ngfc 2 2 0 0 0 0 0 0\n"
    )
    assert field.header().errors == s.IcgemErrors.CALIBRATED_AND_FORMAL
    assert field.header().normalization == s.IcgemNormalization.UNNORMALIZED
    sigmas = field.static_sigmas(2, 0)
    factor = 1.0 / 5.0**0.5
    assert sigmas.calibrated == [factor, 2.0 * factor]
    assert sigmas.formal == [3.0 * factor, 4.0 * factor]
    assert sigmas != s.IcgemSigmas()
