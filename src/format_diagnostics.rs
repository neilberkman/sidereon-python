//! Typed format diagnostics for non-fatal parser findings.

use pyo3::prelude::*;
use sidereon_core::nmea::{
    Diagnostics, FieldError, RecordRef, Skip, SkipReason, Warning, WarningKind,
};

/// Category of validation error on a parser or solver input field.
#[pyclass(module = "sidereon._sidereon", name = "FieldErrorKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[allow(non_camel_case_types)]
pub enum PyFieldErrorKind {
    /// A required field was absent or contained only whitespace.
    MISSING,
    /// A floating-point field was NaN or infinite.
    NON_FINITE,
    /// A value that must be strictly greater than zero was not positive.
    NOT_POSITIVE,
    /// A value that must be at least zero was negative.
    NEGATIVE,
    /// A finite value fell outside the supplied inclusive or half-open bounds.
    OUT_OF_RANGE,
    /// Text could not be parsed as a finite floating-point value.
    FLOAT_PARSE,
    /// Text could not be parsed as the requested integer type.
    INT_PARSE,
    /// Calendar fields do not identify a date in the supported civil range.
    INVALID_CIVIL_DATE,
    /// Clock fields do not identify a valid time of day.
    INVALID_CIVIL_TIME,
}

#[pymethods]
impl PyFieldErrorKind {
    /// Stable lowercase identifier for this field error kind.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::MISSING => "missing",
            Self::NON_FINITE => "non_finite",
            Self::NOT_POSITIVE => "not_positive",
            Self::NEGATIVE => "negative",
            Self::OUT_OF_RANGE => "out_of_range",
            Self::FLOAT_PARSE => "float_parse",
            Self::INT_PARSE => "int_parse",
            Self::INVALID_CIVIL_DATE => "invalid_civil_date",
            Self::INVALID_CIVIL_TIME => "invalid_civil_time",
        }
    }

    fn __repr__(&self) -> String {
        format!("FieldErrorKind.{}", self.label().to_uppercase())
    }
}

impl From<&FieldError> for PyFieldErrorKind {
    fn from(err: &FieldError) -> Self {
        match err {
            FieldError::Missing { .. } => Self::MISSING,
            FieldError::NonFinite { .. } => Self::NON_FINITE,
            FieldError::NotPositive { .. } => Self::NOT_POSITIVE,
            FieldError::Negative { .. } => Self::NEGATIVE,
            FieldError::OutOfRange { .. } => Self::OUT_OF_RANGE,
            FieldError::FloatParse { .. } => Self::FLOAT_PARSE,
            FieldError::IntParse { .. } => Self::INT_PARSE,
            FieldError::InvalidCivilDate { .. } => Self::INVALID_CIVIL_DATE,
            FieldError::InvalidCivilTime { .. } => Self::INVALID_CIVIL_TIME,
        }
    }
}

/// Describes why a named parser or solver input was rejected.
#[pyclass(module = "sidereon._sidereon", name = "FieldError")]
#[derive(Clone, PartialEq)]
pub struct PyFieldError {
    pub(crate) inner: FieldError,
}

#[pymethods]
impl PyFieldError {
    /// Category identifying which constraint was violated.
    #[getter]
    fn kind(&self) -> PyFieldErrorKind {
        PyFieldErrorKind::from(&self.inner)
    }

    /// Stable field name associated with this validation failure.
    #[getter]
    fn field(&self) -> &'static str {
        self.inner.field()
    }

    /// Human-readable description of the validation failure.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// Lower bound that the input had to meet, when bounded.
    #[getter]
    fn min(&self) -> Option<f64> {
        match &self.inner {
            FieldError::OutOfRange { min, .. } => Some(*min),
            _ => None,
        }
    }

    /// Upper bound that the input had to meet, when bounded.
    #[getter]
    fn max(&self) -> Option<f64> {
        match &self.inner {
            FieldError::OutOfRange { max, .. } => Some(*max),
            _ => None,
        }
    }

    /// Whether the upper bound is inclusive, when bounded.
    #[getter]
    fn upper_inclusive(&self) -> Option<bool> {
        match &self.inner {
            FieldError::OutOfRange {
                upper_inclusive, ..
            } => Some(*upper_inclusive),
            _ => None,
        }
    }

    /// Raw text token that failed parsing, when applicable.
    #[getter]
    fn value(&self) -> Option<String> {
        match &self.inner {
            FieldError::FloatParse { value, .. } | FieldError::IntParse { value, .. } => {
                Some(value.clone())
            }
            _ => None,
        }
    }

    /// Calendar year supplied by the caller, when applicable.
    #[getter]
    fn year(&self) -> Option<i64> {
        match &self.inner {
            FieldError::InvalidCivilDate { year, .. } => Some(*year),
            _ => None,
        }
    }

    /// Calendar month supplied by the caller, when applicable.
    #[getter]
    fn month(&self) -> Option<i64> {
        match &self.inner {
            FieldError::InvalidCivilDate { month, .. } => Some(*month),
            _ => None,
        }
    }

    /// Calendar day supplied by the caller, when applicable.
    #[getter]
    fn day(&self) -> Option<i64> {
        match &self.inner {
            FieldError::InvalidCivilDate { day, .. } => Some(*day),
            _ => None,
        }
    }

    /// Hour in the supplied civil time, when applicable.
    #[getter]
    fn hour(&self) -> Option<i64> {
        match &self.inner {
            FieldError::InvalidCivilTime { hour, .. } => Some(*hour),
            _ => None,
        }
    }

    /// Minute in the supplied civil time, when applicable.
    #[getter]
    fn minute(&self) -> Option<i64> {
        match &self.inner {
            FieldError::InvalidCivilTime { minute, .. } => Some(*minute),
            _ => None,
        }
    }

    /// Seconds in the supplied civil time, when applicable.
    #[getter]
    fn second(&self) -> Option<f64> {
        match &self.inner {
            FieldError::InvalidCivilTime { second, .. } => Some(*second),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FieldError(kind=FieldErrorKind.{}, field={:?}, message={:?})",
            self.kind().label().to_uppercase(),
            self.field(),
            self.message()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Reference to a record in an input stream.
#[pyclass(module = "sidereon._sidereon", name = "FormatRecordRef")]
#[derive(Clone, PartialEq)]
pub struct PyFormatRecordRef {
    pub(crate) inner: RecordRef,
}

#[pymethods]
impl PyFormatRecordRef {
    /// One-based input line number, when known.
    #[getter]
    fn line(&self) -> Option<usize> {
        self.inner.line
    }

    /// Logical record index, when known.
    #[getter]
    fn record_index(&self) -> Option<usize> {
        self.inner.record_index
    }

    /// Raw satellite token, when known.
    #[getter]
    fn satellite(&self) -> Option<String> {
        self.inner.satellite.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "FormatRecordRef(line={:?}, record_index={:?}, satellite={:?})",
            self.inner.line, self.inner.record_index, self.inner.satellite
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Typed reason category why a forgiving parser skipped a record.
#[pyclass(
    module = "sidereon._sidereon",
    name = "FormatSkipReasonKind",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[allow(non_camel_case_types)]
pub enum PyFormatSkipReasonKind {
    /// The record names a satellite that cannot be represented downstream.
    UNREPRESENTABLE_SATELLITE,
    /// The record type is outside the reader's supported subset.
    UNSUPPORTED_RECORD_TYPE,
    /// A field failed typed validation.
    MALFORMED_FIELD,
    /// The epoch lies outside the representable range for the target format.
    OUT_OF_RANGE_EPOCH,
    /// The record ended before all required fields were available.
    TRUNCATED,
    /// The record names a unit outside the reader's supported set.
    UNSUPPORTED_UNIT,
    /// A logical block is not modeled by this reader.
    UNKNOWN_BLOCK,
    /// The record is internally inconsistent.
    INCONSISTENT_RECORD,
}

#[pymethods]
impl PyFormatSkipReasonKind {
    /// Stable lowercase identifier for this skip reason kind.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::UNREPRESENTABLE_SATELLITE => "unrepresentable_satellite",
            Self::UNSUPPORTED_RECORD_TYPE => "unsupported_record_type",
            Self::MALFORMED_FIELD => "malformed_field",
            Self::OUT_OF_RANGE_EPOCH => "out_of_range_epoch",
            Self::TRUNCATED => "truncated",
            Self::UNSUPPORTED_UNIT => "unsupported_unit",
            Self::UNKNOWN_BLOCK => "unknown_block",
            Self::INCONSISTENT_RECORD => "inconsistent_record",
        }
    }

    fn __repr__(&self) -> String {
        format!("FormatSkipReasonKind.{}", self.label().to_uppercase())
    }
}

impl From<&SkipReason> for PyFormatSkipReasonKind {
    fn from(reason: &SkipReason) -> Self {
        match reason {
            SkipReason::UnrepresentableSatellite => Self::UNREPRESENTABLE_SATELLITE,
            SkipReason::UnsupportedRecordType(_) => Self::UNSUPPORTED_RECORD_TYPE,
            SkipReason::MalformedField(_) => Self::MALFORMED_FIELD,
            SkipReason::OutOfRangeEpoch => Self::OUT_OF_RANGE_EPOCH,
            SkipReason::Truncated => Self::TRUNCATED,
            SkipReason::UnsupportedUnit(_) => Self::UNSUPPORTED_UNIT,
            SkipReason::UnknownBlock(_) => Self::UNKNOWN_BLOCK,
            SkipReason::InconsistentRecord(_) => Self::INCONSISTENT_RECORD,
        }
    }
}

/// Typed reason and associated details explaining why a record was skipped.
#[pyclass(module = "sidereon._sidereon", name = "FormatSkipReason")]
#[derive(Clone, PartialEq)]
pub struct PyFormatSkipReason {
    pub(crate) inner: SkipReason,
}

#[pymethods]
impl PyFormatSkipReason {
    /// Category of skip reason.
    #[getter]
    fn kind(&self) -> PyFormatSkipReasonKind {
        PyFormatSkipReasonKind::from(&self.inner)
    }

    /// Unsupported record type identifier, when applicable.
    #[getter]
    fn record_type(&self) -> Option<&'static str> {
        match &self.inner {
            SkipReason::UnsupportedRecordType(t) => Some(t),
            _ => None,
        }
    }

    /// Unsupported unit string, when applicable.
    #[getter]
    fn unit(&self) -> Option<String> {
        match &self.inner {
            SkipReason::UnsupportedUnit(u) => Some(u.clone()),
            _ => None,
        }
    }

    /// Unrecognized block name, when applicable.
    #[getter]
    fn block(&self) -> Option<String> {
        match &self.inner {
            SkipReason::UnknownBlock(b) => Some(b.clone()),
            _ => None,
        }
    }

    /// Inconsistent record detail description, when applicable.
    #[getter]
    fn detail(&self) -> Option<&'static str> {
        match &self.inner {
            SkipReason::InconsistentRecord(d) => Some(d),
            _ => None,
        }
    }

    /// Typed field error, when applicable.
    #[getter]
    fn field_error(&self) -> Option<PyFieldError> {
        match &self.inner {
            SkipReason::MalformedField(err) => Some(PyFieldError { inner: err.clone() }),
            _ => None,
        }
    }

    /// Plain description built from the kind and payload.
    #[getter]
    fn message(&self) -> String {
        match &self.inner {
            SkipReason::UnrepresentableSatellite => "unrepresentable satellite".to_string(),
            SkipReason::UnsupportedRecordType(t) => format!("unsupported record type: {t}"),
            SkipReason::MalformedField(err) => err.to_string(),
            SkipReason::OutOfRangeEpoch => "epoch out of range".to_string(),
            SkipReason::Truncated => "truncated record".to_string(),
            SkipReason::UnsupportedUnit(u) => format!("unsupported unit: {u}"),
            SkipReason::UnknownBlock(b) => format!("unknown block: {b}"),
            SkipReason::InconsistentRecord(d) => format!("inconsistent record: {d}"),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FormatSkipReason(kind=FormatSkipReasonKind.{}, message={:?})",
            self.kind().label().to_uppercase(),
            self.message()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A record skipped during parsing and the reason it was skipped.
#[pyclass(module = "sidereon._sidereon", name = "FormatSkip")]
#[derive(Clone, PartialEq)]
pub struct PyFormatSkip {
    pub(crate) inner: Skip,
}

#[pymethods]
impl PyFormatSkip {
    /// Location where the record was skipped.
    #[getter]
    fn at(&self) -> PyFormatRecordRef {
        PyFormatRecordRef {
            inner: self.inner.at.clone(),
        }
    }

    /// Why the record was skipped.
    #[getter]
    fn reason(&self) -> PyFormatSkipReason {
        PyFormatSkipReason {
            inner: self.inner.reason.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "FormatSkip(at={}, reason={})",
            self.at().__repr__(),
            self.reason().__repr__()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Advisory warning categories.
#[pyclass(module = "sidereon._sidereon", name = "FormatWarningKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[allow(non_camel_case_types)]
pub enum PyFormatWarningKind {
    /// A checksum did not match the record body.
    CHECKSUM,
    /// A value was clamped to fit the target range.
    CLAMPED,
    /// A value lost precision or fidelity during conversion.
    DEGRADED,
    /// A declared count or mode did not match decoded records.
    MISMATCH,
    /// Published validity intervals overlap.
    OVERLAP,
    /// A required-by-format metadata block was absent but recoverable.
    MISSING_METADATA,
}

#[pymethods]
impl PyFormatWarningKind {
    /// Stable lowercase identifier for this warning category.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::CHECKSUM => "checksum",
            Self::CLAMPED => "clamped",
            Self::DEGRADED => "degraded",
            Self::MISMATCH => "mismatch",
            Self::OVERLAP => "overlap",
            Self::MISSING_METADATA => "missing_metadata",
        }
    }

    fn __repr__(&self) -> String {
        format!("FormatWarningKind.{}", self.label().to_uppercase())
    }
}

impl From<WarningKind> for PyFormatWarningKind {
    fn from(kind: WarningKind) -> Self {
        match kind {
            WarningKind::Checksum => Self::CHECKSUM,
            WarningKind::Clamped => Self::CLAMPED,
            WarningKind::Degraded => Self::DEGRADED,
            WarningKind::Mismatch => Self::MISMATCH,
            WarningKind::Overlap => Self::OVERLAP,
            WarningKind::MissingMetadata => Self::MISSING_METADATA,
        }
    }
}

/// An advisory warning attached to a record.
#[pyclass(module = "sidereon._sidereon", name = "FormatWarning")]
#[derive(Clone, PartialEq)]
pub struct PyFormatWarning {
    pub(crate) inner: Warning,
}

#[pymethods]
impl PyFormatWarning {
    /// Where the warning came from.
    #[getter]
    fn at(&self) -> PyFormatRecordRef {
        PyFormatRecordRef {
            inner: self.inner.at.clone(),
        }
    }

    /// The warning category.
    #[getter]
    fn kind(&self) -> PyFormatWarningKind {
        PyFormatWarningKind::from(self.inner.kind)
    }

    fn __repr__(&self) -> String {
        format!(
            "FormatWarning(at={}, kind=FormatWarningKind.{})",
            self.at().__repr__(),
            self.kind().label().to_uppercase()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Non-fatal parser diagnostics holding skipped records and advisory warnings.
#[pyclass(module = "sidereon._sidereon", name = "FormatDiagnostics")]
#[derive(Clone, PartialEq)]
pub struct PyFormatDiagnostics {
    pub(crate) inner: Diagnostics,
}

impl PyFormatDiagnostics {
    pub(crate) fn from_inner(inner: Diagnostics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFormatDiagnostics {
    /// Records skipped during parsing.
    #[getter]
    fn skips(&self) -> Vec<PyFormatSkip> {
        self.inner
            .skips
            .iter()
            .cloned()
            .map(|inner| PyFormatSkip { inner })
            .collect()
    }

    /// Advisory warnings collected during parsing.
    #[getter]
    fn warnings(&self) -> Vec<PyFormatWarning> {
        self.inner
            .warnings
            .iter()
            .cloned()
            .map(|inner| PyFormatWarning { inner })
            .collect()
    }

    /// Number of skipped records.
    #[getter]
    fn skip_count(&self) -> usize {
        self.inner.skips.len()
    }

    /// Number of advisory warnings.
    #[getter]
    fn warning_count(&self) -> usize {
        self.inner.warnings.len()
    }

    /// Returns True when no skips or warnings are present.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "FormatDiagnostics(skip_count={}, warning_count={})",
            self.inner.skips.len(),
            self.inner.warnings.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFieldErrorKind>()?;
    m.add_class::<PyFieldError>()?;
    m.add_class::<PyFormatRecordRef>()?;
    m.add_class::<PyFormatSkipReasonKind>()?;
    m.add_class::<PyFormatSkipReason>()?;
    m.add_class::<PyFormatSkip>()?;
    m.add_class::<PyFormatWarningKind>()?;
    m.add_class::<PyFormatWarning>()?;
    m.add_class::<PyFormatDiagnostics>()?;
    Ok(())
}
