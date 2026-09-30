"""Tests for typed format diagnostics across core entry points."""

import json
import os

import pytest
import sidereon
from _helpers import FIXTURES


@pytest.fixture
def inputs():
    path = os.path.join(FIXTURES, "format_diagnostics_inputs.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture
def goldens():
    path = os.path.join(FIXTURES, "core_goldens.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)["format_diagnostics"]


def _assert_field_error_matches(field_error, golden):
    if golden is None:
        assert field_error is None
        return
    assert field_error is not None
    assert field_error.kind.label == golden["kind"]
    assert field_error.field == golden["field"]
    assert field_error.message == golden["message"]
    assert field_error.min == golden["min"]
    assert field_error.max == golden["max"]
    assert field_error.upper_inclusive == golden["upper_inclusive"]
    assert field_error.value == golden["value"]
    assert field_error.year == golden["year"]
    assert field_error.month == golden["month"]
    assert field_error.day == golden["day"]
    assert field_error.hour == golden["hour"]
    assert field_error.minute == golden["minute"]
    assert field_error.second == golden["second"]


def _assert_skip_matches(skip, golden):
    assert skip.at.line == golden["line"]
    assert skip.at.record_index == golden["record_index"]
    assert skip.at.satellite == golden["satellite"]
    reason = skip.reason
    g_reason = golden["reason"]
    assert reason.kind.label == g_reason["kind"]
    assert reason.record_type == g_reason["record_type"]
    assert reason.unit == g_reason["unit"]
    assert reason.block == g_reason["block"]
    assert reason.detail == g_reason["detail"]
    assert reason.message == g_reason["message"]
    _assert_field_error_matches(reason.field_error, g_reason["field_error"])


def _assert_warning_matches(warning, golden):
    assert warning.at.line == golden["line"]
    assert warning.at.record_index == golden["record_index"]
    assert warning.at.satellite == golden["satellite"]
    assert warning.kind.label == golden["kind"]


def _assert_diagnostics_matches(diagnostics, golden):
    assert diagnostics.skip_count == len(golden["skips"])
    assert diagnostics.warning_count == len(golden["warnings"])
    assert len(diagnostics.skips) == len(golden["skips"])
    assert len(diagnostics.warnings) == len(golden["warnings"])
    for skip, g_skip in zip(diagnostics.skips, golden["skips"]):
        _assert_skip_matches(skip, g_skip)
    for warning, g_warning in zip(diagnostics.warnings, golden["warnings"]):
        _assert_warning_matches(warning, g_warning)


def test_enum_labels_are_snake_case():
    enums = (
        (
            sidereon.FieldErrorKind,
            (
                (sidereon.FieldErrorKind.MISSING, "MISSING", "missing"),
                (sidereon.FieldErrorKind.NON_FINITE, "NON_FINITE", "non_finite"),
                (sidereon.FieldErrorKind.NOT_POSITIVE, "NOT_POSITIVE", "not_positive"),
                (sidereon.FieldErrorKind.NEGATIVE, "NEGATIVE", "negative"),
                (sidereon.FieldErrorKind.OUT_OF_RANGE, "OUT_OF_RANGE", "out_of_range"),
                (sidereon.FieldErrorKind.FLOAT_PARSE, "FLOAT_PARSE", "float_parse"),
                (sidereon.FieldErrorKind.INT_PARSE, "INT_PARSE", "int_parse"),
                (
                    sidereon.FieldErrorKind.INVALID_CIVIL_DATE,
                    "INVALID_CIVIL_DATE",
                    "invalid_civil_date",
                ),
                (
                    sidereon.FieldErrorKind.INVALID_CIVIL_TIME,
                    "INVALID_CIVIL_TIME",
                    "invalid_civil_time",
                ),
            ),
        ),
        (
            sidereon.FormatSkipReasonKind,
            (
                (
                    sidereon.FormatSkipReasonKind.UNREPRESENTABLE_SATELLITE,
                    "UNREPRESENTABLE_SATELLITE",
                    "unrepresentable_satellite",
                ),
                (
                    sidereon.FormatSkipReasonKind.UNSUPPORTED_RECORD_TYPE,
                    "UNSUPPORTED_RECORD_TYPE",
                    "unsupported_record_type",
                ),
                (
                    sidereon.FormatSkipReasonKind.MALFORMED_FIELD,
                    "MALFORMED_FIELD",
                    "malformed_field",
                ),
                (
                    sidereon.FormatSkipReasonKind.OUT_OF_RANGE_EPOCH,
                    "OUT_OF_RANGE_EPOCH",
                    "out_of_range_epoch",
                ),
                (
                    sidereon.FormatSkipReasonKind.TRUNCATED,
                    "TRUNCATED",
                    "truncated",
                ),
                (
                    sidereon.FormatSkipReasonKind.UNSUPPORTED_UNIT,
                    "UNSUPPORTED_UNIT",
                    "unsupported_unit",
                ),
                (
                    sidereon.FormatSkipReasonKind.UNKNOWN_BLOCK,
                    "UNKNOWN_BLOCK",
                    "unknown_block",
                ),
                (
                    sidereon.FormatSkipReasonKind.INCONSISTENT_RECORD,
                    "INCONSISTENT_RECORD",
                    "inconsistent_record",
                ),
            ),
        ),
        (
            sidereon.FormatWarningKind,
            (
                (sidereon.FormatWarningKind.CHECKSUM, "CHECKSUM", "checksum"),
                (sidereon.FormatWarningKind.CLAMPED, "CLAMPED", "clamped"),
                (sidereon.FormatWarningKind.DEGRADED, "DEGRADED", "degraded"),
                (sidereon.FormatWarningKind.MISMATCH, "MISMATCH", "mismatch"),
                (sidereon.FormatWarningKind.OVERLAP, "OVERLAP", "overlap"),
                (
                    sidereon.FormatWarningKind.MISSING_METADATA,
                    "MISSING_METADATA",
                    "missing_metadata",
                ),
            ),
        ),
    )
    for cls, variants in enums:
        for variant, expected_name, expected_label in variants:
            assert getattr(cls, expected_name) == variant
            assert variant.label == expected_label
            assert repr(variant) == f"{cls.__name__}.{expected_name}"


def test_nmea_format_diagnostics(inputs, goldens):
    log = sidereon.parse_nmea(inputs["nmea"].encode("utf-8"))
    _assert_diagnostics_matches(log.diagnostics, goldens["nmea"])
    assert not log.diagnostics.is_empty()


def test_bias_sinex_format_diagnostics(inputs, goldens):
    parsed = sidereon.parse_bias_sinex_lossy(
        inputs["bias_sinex"].encode("utf-8"),
        policy=sidereon.BiasReadPolicy.LENIENT,
    )
    _assert_diagnostics_matches(parsed.diagnostics, goldens["bias_sinex"])
    assert parsed.skip_count == parsed.diagnostics.skip_count
    assert parsed.warning_count == parsed.diagnostics.warning_count
    assert any(
        notice["kind"] == "departure"
        and notice["departure"]["kind"] == "header_layout"
        and notice["departure"]["reason"]
        for notice in parsed.value.notice_details
    )
    assert not parsed.diagnostics.is_empty()


def test_bias_sinex_format_departure_is_refused_by_strict_policy(inputs):
    bias_bytes = inputs["bias_sinex"].encode("utf-8")
    with pytest.raises(sidereon.BiasError) as strict_error:
        sidereon.parse_bias_sinex_lossy(
            bias_bytes, policy=sidereon.BiasReadPolicy.STRICT
        )
    assert strict_error.value.kind == "departure"
    assert strict_error.value.details["departure"]["kind"] == "header_layout"
    assert strict_error.value.details["departure"]["reason"]

    with pytest.raises(sidereon.BiasError):
        sidereon.parse_bias_sinex_lossy(bias_bytes)


def test_space_weather_format_diagnostics(inputs, goldens):
    table = sidereon.load_space_weather(inputs["space_weather"].encode("utf-8"))
    _assert_diagnostics_matches(table.diagnostics, goldens["space_weather"])
    assert not table.diagnostics.is_empty()


def test_nmea_diagnostics_alias():
    assert sidereon.NmeaDiagnostics is sidereon.FormatDiagnostics


def test_empty_diagnostics():
    clean_nmea = (
        b"$GPRMC,123520,A,4807.038,N,01131.000,E,22.4,84.4,230394,3.1,W,A,S*72\n"
    )
    log = sidereon.parse_nmea(clean_nmea)
    diag = log.diagnostics
    assert diag.is_empty()
    assert diag.skip_count == 0
    assert diag.warning_count == 0
    assert diag.skips == []
    assert diag.warnings == []
    assert diag == log.diagnostics
    assert repr(diag) == "FormatDiagnostics(skip_count=0, warning_count=0)"


def test_format_diagnostics_repr_and_equality(inputs):
    log = sidereon.parse_nmea(inputs["nmea"].encode("utf-8"))
    diag = log.diagnostics
    assert repr(diag) == (
        f"FormatDiagnostics(skip_count={diag.skip_count},"
        f" warning_count={diag.warning_count})"
    )
    assert diag == diag
    skip = diag.skips[0]
    assert skip == skip
    assert repr(skip).startswith("FormatSkip(")
    assert repr(skip.at).startswith("FormatRecordRef(")
    assert repr(skip.reason).startswith("FormatSkipReason(")
    warning = diag.warnings[0]
    assert warning == warning
    assert repr(warning).startswith("FormatWarning(")
