//! Goldens for typed format diagnostics. Each function parses test inputs
//! through the core entry points the binding calls and writes the resulting
//! diagnostics as JSON bit-for-bit against the binding tests.

use std::path::Path;

use serde_json::{json, Value};
use sidereon_core::bias::BiasReadPolicy;
use sidereon_core::nmea::{Diagnostics, FieldError, Skip, SkipReason, Warning, WarningKind};

fn serialize_field_error(err: &FieldError) -> Value {
    let (kind, min, max, upper_inclusive, value, year, month, day, hour, minute, second) = match err
    {
        FieldError::Missing { .. } => (
            "missing", None, None, None, None, None, None, None, None, None, None,
        ),
        FieldError::NonFinite { .. } => (
            "non_finite",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        FieldError::NotPositive { .. } => (
            "not_positive",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        FieldError::Negative { .. } => (
            "negative", None, None, None, None, None, None, None, None, None, None,
        ),
        FieldError::OutOfRange {
            min,
            max,
            upper_inclusive,
            ..
        } => (
            "out_of_range",
            Some(*min),
            Some(*max),
            Some(*upper_inclusive),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        FieldError::FloatParse { value, .. } => (
            "float_parse",
            None,
            None,
            None,
            Some(value.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        FieldError::IntParse { value, .. } => (
            "int_parse",
            None,
            None,
            None,
            Some(value.clone()),
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        FieldError::InvalidCivilDate {
            year, month, day, ..
        } => (
            "invalid_civil_date",
            None,
            None,
            None,
            None,
            Some(*year),
            Some(*month),
            Some(*day),
            None,
            None,
            None,
        ),
        FieldError::InvalidCivilTime {
            hour,
            minute,
            second,
            ..
        } => (
            "invalid_civil_time",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(*hour),
            Some(*minute),
            Some(*second),
        ),
    };
    json!({
        "kind": kind,
        "field": err.field(),
        "message": err.to_string(),
        "min": min,
        "max": max,
        "upper_inclusive": upper_inclusive,
        "value": value,
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
    })
}

fn serialize_skip_reason(reason: &SkipReason) -> Value {
    let (kind, record_type, unit, block, detail, field_error, message) = match reason {
        SkipReason::UnrepresentableSatellite => (
            "unrepresentable_satellite",
            None,
            None,
            None,
            None,
            None,
            "unrepresentable satellite".to_string(),
        ),
        SkipReason::UnsupportedRecordType(t) => (
            "unsupported_record_type",
            Some(*t),
            None,
            None,
            None,
            None,
            format!("unsupported record type: {t}"),
        ),
        SkipReason::MalformedField(err) => (
            "malformed_field",
            None,
            None,
            None,
            None,
            Some(serialize_field_error(err)),
            err.to_string(),
        ),
        SkipReason::OutOfRangeEpoch => (
            "out_of_range_epoch",
            None,
            None,
            None,
            None,
            None,
            "epoch out of range".to_string(),
        ),
        SkipReason::Truncated => (
            "truncated",
            None,
            None,
            None,
            None,
            None,
            "truncated record".to_string(),
        ),
        SkipReason::UnsupportedUnit(u) => (
            "unsupported_unit",
            None,
            Some(u.clone()),
            None,
            None,
            None,
            format!("unsupported unit: {u}"),
        ),
        SkipReason::UnknownBlock(b) => (
            "unknown_block",
            None,
            None,
            Some(b.clone()),
            None,
            None,
            format!("unknown block: {b}"),
        ),
        SkipReason::InconsistentRecord(d) => (
            "inconsistent_record",
            None,
            None,
            None,
            Some(*d),
            None,
            format!("inconsistent record: {d}"),
        ),
    };
    json!({
        "kind": kind,
        "record_type": record_type,
        "unit": unit,
        "block": block,
        "detail": detail,
        "field_error": field_error,
        "message": message,
    })
}

fn serialize_skip(skip: &Skip) -> Value {
    json!({
        "line": skip.at.line,
        "record_index": skip.at.record_index,
        "satellite": skip.at.satellite,
        "reason": serialize_skip_reason(&skip.reason),
    })
}

fn serialize_warning(warning: &Warning) -> Value {
    let kind = match warning.kind {
        WarningKind::Checksum => "checksum",
        WarningKind::Clamped => "clamped",
        WarningKind::Degraded => "degraded",
        WarningKind::Mismatch => "mismatch",
        WarningKind::Overlap => "overlap",
        WarningKind::MissingMetadata => "missing_metadata",
    };
    json!({
        "line": warning.at.line,
        "record_index": warning.at.record_index,
        "satellite": warning.at.satellite,
        "kind": kind,
    })
}

fn serialize_diagnostics(diagnostics: &Diagnostics) -> Value {
    json!({
        "skips": diagnostics.skips.iter().map(serialize_skip).collect::<Vec<_>>(),
        "warnings": diagnostics.warnings.iter().map(serialize_warning).collect::<Vec<_>>(),
    })
}

pub fn sections(fixtures: &Path) -> Vec<(&'static str, Value)> {
    let inputs_path = fixtures.join("format_diagnostics_inputs.json");
    let inputs_text = std::fs::read_to_string(&inputs_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", inputs_path.display()));
    let inputs: Value =
        serde_json::from_str(&inputs_text).expect("parse format diagnostics inputs JSON");

    let nmea_str = inputs["nmea"].as_str().expect("nmea input string");
    let nmea_log = sidereon_core::nmea::parse_nmea(nmea_str.as_bytes());

    let bias_str = inputs["bias_sinex"]
        .as_str()
        .expect("bias_sinex input string");
    let bias_parsed =
        sidereon::parse_bias_sinex_lossy_with_policy(bias_str.as_bytes(), BiasReadPolicy::Lenient)
            .expect("parse bias sinex lossy");

    let sw_str = inputs["space_weather"]
        .as_str()
        .expect("space_weather input string");
    let sw_parsed =
        sidereon_core::astro::space_weather::parse(sw_str.as_bytes()).expect("parse space weather");

    let section_val = json!({
        "nmea": serialize_diagnostics(&nmea_log.diagnostics),
        "bias_sinex": serialize_diagnostics(&bias_parsed.diagnostics),
        "space_weather": serialize_diagnostics(&sw_parsed.diagnostics),
    });

    vec![("format_diagnostics", section_val)]
}
