//! RINEX clock (`.CLK`) binding: lossless products, typed views, edits and
//! policy-aware writing.
//!
//! A PyO3 surface over `sidereon_core::rinex::clock`. The core keeps the text
//! it reads as the product's authority; this module copies the typed views the
//! core derives from that text into Python value objects, forwards edits to the
//! core setters, and maps every core refusal to a typed exception carrying the
//! core error as `detail`. It runs no clock parsing, arithmetic or validation
//! of its own apart from reading Python arguments.

use numpy::PyArray1;
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyModule, PyTuple, PyType};

use sidereon_core::astro::time::{Instant as CoreInstant, InstantRepr, TimeScale as CoreTimeScale};
use sidereon_core::rinex::clock::{
    civil_to_clock_instant as core_civil_to_clock_instant,
    civil_to_gps_seconds as core_civil_to_gps_seconds, ClockEpoch as CoreClockEpoch,
    ClockHeaderField as CoreClockHeaderField, ClockHeaderReading as CoreClockHeaderReading,
    ClockHeaderRecord as CoreClockHeaderRecord, ClockLayout as CoreClockLayout,
    ClockPoint as CoreClockPoint, ClockRecord as CoreClockRecord,
    ClockRecordReading as CoreClockRecordReading, ClockRecordType as CoreClockRecordType,
    ClockSurplusValue as CoreClockSurplusValue, ClockTimeSystem as CoreClockTimeSystem,
    ClockTimeSystemStatus as CoreClockTimeSystemStatus,
    ClockWriteDeparture as CoreClockWriteDeparture, ClockWriteLeniency as CoreClockWriteLeniency,
    ClockWritePolicy as CoreClockWritePolicy, RinexClock as CoreRinexClock,
    RinexClockDiagnostic as CoreRinexClockDiagnostic, RinexClockError,
    RinexClockNotice as CoreRinexClockNotice, RinexClockSkip as CoreRinexClockSkip,
};

use crate::frames::PyTimeScale;
use crate::np_array;
use crate::rinex::{text_from_source, RinexTextKind};
use crate::{
    rinex_clock_edit_error_type, rinex_clock_query_error_type, rinex_clock_write_error_type,
    RinexClockParseError,
};

/// The variant name at the head of a derived `Debug` rendering.
///
/// Several core clock enums are `#[non_exhaustive]`, so a core newer than this
/// binding can hand back a variant no arm here names. Its derived `Debug`
/// output starts with the variant name, which is the `kind` reported for it.
fn debug_variant_name<T: std::fmt::Debug>(value: &T) -> String {
    format!("{value:?}")
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// Raise `ty` with the core error's message and its typed payload as `detail`.
fn clock_error(py: Python<'_>, ty: PyResult<Bound<'_, PyType>>, err: RinexClockError) -> PyErr {
    let ty = match ty {
        Ok(ty) => ty,
        Err(e) => return e,
    };
    let py_err = PyErr::from_type(ty, err.to_string());
    let detail = match (PyRinexClockErrorDetail { inner: err }).into_pyobject(py) {
        Ok(detail) => detail,
        Err(e) => return e,
    };
    if let Err(e) = py_err.value(py).setattr("detail", detail) {
        return e;
    }
    py_err
}

/// A strict-parse refusal: `RinexClockParseError`.
fn to_clock_parse_err(py: Python<'_>, err: RinexClockError) -> PyErr {
    clock_error(py, Ok(py.get_type::<RinexClockParseError>()), err)
}

/// A query refusal: `RinexClockQueryError`, a subclass of
/// `RinexClockParseError` (which `clock_s` raised before) and `ValueError`
/// (which the instant and GPS-seconds queries raised before).
fn to_clock_query_err(py: Python<'_>, err: RinexClockError) -> PyErr {
    clock_error(py, rinex_clock_query_error_type(py), err)
}

/// A write refusal: `RinexClockWriteError`.
fn to_clock_write_err(py: Python<'_>, err: RinexClockError) -> PyErr {
    clock_error(py, rinex_clock_write_error_type(py), err)
}

/// A refused record, point, construction or edit: `RinexClockEditError`.
fn to_clock_edit_err(py: Python<'_>, err: RinexClockError) -> PyErr {
    clock_error(py, rinex_clock_edit_error_type(py), err)
}

/// GPS seconds for a clock epoch, through the core point projection.
fn clock_instant_gps_seconds(epoch: CoreInstant) -> Option<f64> {
    CoreClockPoint::new(epoch, 0.0, Vec::new()).gps_seconds()
}

/// The typed payload a RINEX clock failure carries on its `detail` attribute.
///
/// `kind` is the core `RinexClockError` variant name and `details()` that
/// variant's own payload under its own keys, so a caller reads the native
/// values rather than parsing `message`. `line`, `reason`, `record`,
/// `record_type`, `field`, `value` and `time_scale` are conveniences that
/// return None for a variant that names no such field.
///
/// `RinexClockParseError`, `RinexClockQueryError`, `RinexClockWriteError` and
/// `RinexClockEditError` raised from a core refusal always carry one. A failure
/// with no core error behind it - an unreadable file, text that is not UTF-8,
/// an argument Python itself rejects - and a hand-built exception carry none,
/// so the class attribute defaults to None and `except ... as e: e.detail`
/// never raises.
#[pyclass(module = "sidereon._sidereon", name = "RinexClockErrorDetail")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyRinexClockErrorDetail {
    inner: RinexClockError,
}

impl From<RinexClockError> for PyRinexClockErrorDetail {
    fn from(inner: RinexClockError) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexClockErrorDetail {
    /// Error kind string matching the core `RinexClockError` variant name.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            RinexClockError::MalformedAsRecord { .. } => "MalformedAsRecord",
            RinexClockError::MissingContinuation { .. } => "MissingContinuation",
            RinexClockError::MalformedContinuation { .. } => "MalformedContinuation",
            RinexClockError::BadField { .. } => "BadField",
            RinexClockError::InvalidInput { .. } => "InvalidInput",
            RinexClockError::UnsupportedTimeScale { .. } => "UnsupportedTimeScale",
        }
    }

    /// Formatted core error message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// One-based input line number, for the variants that name one.
    #[getter]
    fn line(&self) -> Option<usize> {
        match &self.inner {
            RinexClockError::MalformedAsRecord { line, .. }
            | RinexClockError::MissingContinuation { line, .. }
            | RinexClockError::MalformedContinuation { line, .. }
            | RinexClockError::BadField { line, .. } => Some(*line),
            RinexClockError::InvalidInput { .. } | RinexClockError::UnsupportedTimeScale { .. } => {
                None
            }
        }
    }

    /// Human-readable failure reason, for the variants that carry one.
    #[getter]
    fn reason(&self) -> Option<&'static str> {
        match &self.inner {
            RinexClockError::MalformedAsRecord { reason, .. }
            | RinexClockError::MalformedContinuation { reason, .. }
            | RinexClockError::InvalidInput { reason, .. } => Some(reason),
            _ => None,
        }
    }

    /// The full record text, for the malformed-record variants.
    #[getter]
    fn record(&self) -> Option<&str> {
        match &self.inner {
            RinexClockError::MalformedAsRecord { record, .. }
            | RinexClockError::MalformedContinuation { record, .. } => Some(record),
            _ => None,
        }
    }

    /// Record type of the parent record, for `MissingContinuation`.
    #[getter]
    fn record_type(&self) -> Option<&str> {
        match &self.inner {
            RinexClockError::MissingContinuation { record_type, .. } => Some(record_type),
            _ => None,
        }
    }

    /// Field name, for `BadField` and `InvalidInput`.
    #[getter]
    fn field(&self) -> Option<&'static str> {
        match &self.inner {
            RinexClockError::BadField { field, .. }
            | RinexClockError::InvalidInput { field, .. } => Some(field),
            _ => None,
        }
    }

    /// Source field value, for `BadField`.
    #[getter]
    fn value(&self) -> Option<&str> {
        match &self.inner {
            RinexClockError::BadField { value, .. } => Some(value),
            _ => None,
        }
    }

    /// The time scale named by `UnsupportedTimeScale`.
    #[getter]
    fn time_scale(&self) -> Option<PyTimeScale> {
        match &self.inner {
            RinexClockError::UnsupportedTimeScale { scale } => Some((*scale).into()),
            _ => None,
        }
    }

    /// Dictionary containing all payload fields of this error variant.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            RinexClockError::MalformedAsRecord {
                line,
                reason,
                record,
            }
            | RinexClockError::MalformedContinuation {
                line,
                reason,
                record,
            } => {
                dict.set_item("line", *line)?;
                dict.set_item("reason", *reason)?;
                dict.set_item("record", record.clone())?;
            }
            RinexClockError::MissingContinuation { line, record_type } => {
                dict.set_item("line", *line)?;
                dict.set_item("record_type", record_type.clone())?;
            }
            RinexClockError::BadField { line, field, value } => {
                dict.set_item("line", *line)?;
                dict.set_item("field", *field)?;
                dict.set_item("value", value.clone())?;
            }
            RinexClockError::InvalidInput { field, reason } => {
                dict.set_item("field", *field)?;
                dict.set_item("reason", *reason)?;
            }
            RinexClockError::UnsupportedTimeScale { scale } => {
                dict.set_item("scale", PyTimeScale::from(*scale))?;
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexClockErrorDetail(kind=\"{}\", message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &PyRinexClockErrorDetail) -> bool {
        self.inner == other.inner
    }
}

/// A time system a RINEX clock `TIME SYSTEM ID` record names.
///
/// `GLO` has the hours of UTC in every version, as RINEX clock 3.00, RINEX
/// 3.05 section 4.1.2 and RTKLIB read it, so its `time_scale` is UTC. `IRN`
/// has no core time scale.
#[pyclass(module = "sidereon._sidereon", name = "ClockTimeSystem", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyClockTimeSystem {
    /// GPS system time.
    GPS,
    /// GLONASS time as RINEX reports it, with the hours of UTC.
    GLO,
    /// Galileo system time.
    GAL,
    /// QZSS system time.
    QZS,
    /// BeiDou system time (`BDS`; the spelling `BDT` is also read).
    BDS,
    /// IRNSS system time.
    IRN,
    /// Coordinated Universal Time.
    UTC,
    /// International Atomic Time.
    TAI,
}

impl From<PyClockTimeSystem> for CoreClockTimeSystem {
    fn from(system: PyClockTimeSystem) -> Self {
        match system {
            PyClockTimeSystem::GPS => Self::Gps,
            PyClockTimeSystem::GLO => Self::Glo,
            PyClockTimeSystem::GAL => Self::Gal,
            PyClockTimeSystem::QZS => Self::Qzs,
            PyClockTimeSystem::BDS => Self::Bds,
            PyClockTimeSystem::IRN => Self::Irn,
            PyClockTimeSystem::UTC => Self::Utc,
            PyClockTimeSystem::TAI => Self::Tai,
        }
    }
}

/// The Python value of a core time system. The core enum is
/// `#[non_exhaustive]`; a system this binding has no value for is refused by
/// label rather than mapped onto another one.
fn clock_time_system_to_py(system: CoreClockTimeSystem) -> PyResult<PyClockTimeSystem> {
    Ok(match system {
        CoreClockTimeSystem::Gps => PyClockTimeSystem::GPS,
        CoreClockTimeSystem::Glo => PyClockTimeSystem::GLO,
        CoreClockTimeSystem::Gal => PyClockTimeSystem::GAL,
        CoreClockTimeSystem::Qzs => PyClockTimeSystem::QZS,
        CoreClockTimeSystem::Bds => PyClockTimeSystem::BDS,
        CoreClockTimeSystem::Irn => PyClockTimeSystem::IRN,
        CoreClockTimeSystem::Utc => PyClockTimeSystem::UTC,
        CoreClockTimeSystem::Tai => PyClockTimeSystem::TAI,
        other => {
            return Err(PyValueError::new_err(format!(
                "RINEX clock time system {} has no ClockTimeSystem value in this binding",
                other.label()
            )))
        }
    })
}

#[pymethods]
impl PyClockTimeSystem {
    /// The label this system is written with (`GPS`, `GLO`, ...).
    #[getter]
    fn label(&self) -> &'static str {
        CoreClockTimeSystem::from(*self).label()
    }

    /// The time scale record epochs are read in; None for `IRN`.
    #[getter]
    fn time_scale(&self) -> Option<PyTimeScale> {
        CoreClockTimeSystem::from(*self)
            .time_scale()
            .map(PyTimeScale::from)
    }

    /// Read a `TIME SYSTEM ID` label; None for a label no system has.
    #[staticmethod]
    fn from_label(label: &str) -> PyResult<Option<Self>> {
        CoreClockTimeSystem::from_label(label)
            .map(clock_time_system_to_py)
            .transpose()
    }

    /// The system a product built in `scale` is written with; None for a
    /// scale no RINEX clock time system names, GLONASS system time among
    /// them.
    #[staticmethod]
    fn for_time_scale(scale: PyTimeScale) -> PyResult<Option<Self>> {
        CoreClockTimeSystem::for_time_scale(CoreTimeScale::from(scale))
            .map(clock_time_system_to_py)
            .transpose()
    }
}

/// Column layout of a RINEX clock file: the 80-column layout of versions
/// before 3.04, or the 85-column layout of 3.04 and later.
#[pyclass(module = "sidereon._sidereon", name = "ClockLayout", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyClockLayout {
    /// The 80-column layout of versions before 3.04.
    V300,
    /// The 85-column layout of version 3.04 and later.
    V304,
}

impl From<CoreClockLayout> for PyClockLayout {
    fn from(layout: CoreClockLayout) -> Self {
        match layout {
            CoreClockLayout::V300 => Self::V300,
            CoreClockLayout::V304 => Self::V304,
        }
    }
}

impl From<PyClockLayout> for CoreClockLayout {
    fn from(layout: PyClockLayout) -> Self {
        match layout {
            PyClockLayout::V300 => Self::V300,
            PyClockLayout::V304 => Self::V304,
        }
    }
}

#[pymethods]
impl PyClockLayout {
    /// Zero-based column where a header label starts (60 or 65).
    #[getter]
    fn label_column(&self) -> usize {
        CoreClockLayout::from(*self).label_column()
    }

    /// Width of the receiver or satellite name field in a data record.
    #[getter]
    fn name_width(&self) -> usize {
        CoreClockLayout::from(*self).name_width()
    }

    /// The layout a declared format version uses.
    #[staticmethod]
    fn for_version(version: f64) -> Self {
        CoreClockLayout::for_version(version).into()
    }
}

/// A RINEX clock data record type (Table A16).
#[pyclass(module = "sidereon._sidereon", name = "ClockRecordType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyClockRecordType {
    /// Analysis result for a receiver clock.
    AR,
    /// Analysis result for a satellite clock.
    AS,
    /// Calibration measurement for a receiver.
    CR,
    /// Discontinuity measurement for a receiver.
    DR,
    /// Monitor measurement for a broadcast satellite clock.
    MS,
}

impl From<CoreClockRecordType> for PyClockRecordType {
    fn from(record_type: CoreClockRecordType) -> Self {
        match record_type {
            CoreClockRecordType::Ar => Self::AR,
            CoreClockRecordType::As => Self::AS,
            CoreClockRecordType::Cr => Self::CR,
            CoreClockRecordType::Dr => Self::DR,
            CoreClockRecordType::Ms => Self::MS,
        }
    }
}

impl From<PyClockRecordType> for CoreClockRecordType {
    fn from(record_type: PyClockRecordType) -> Self {
        match record_type {
            PyClockRecordType::AR => Self::Ar,
            PyClockRecordType::AS => Self::As,
            PyClockRecordType::CR => Self::Cr,
            PyClockRecordType::DR => Self::Dr,
            PyClockRecordType::MS => Self::Ms,
        }
    }
}

#[pymethods]
impl PyClockRecordType {
    /// The two-letter code as written in the file.
    #[getter]
    fn code(&self) -> &'static str {
        CoreClockRecordType::from(*self).code()
    }

    /// Read a two-letter record type code; None for any other text.
    #[staticmethod]
    fn from_code(code: &str) -> Option<Self> {
        CoreClockRecordType::from_code(code).map(Self::from)
    }
}

/// Whether the RINEX clock writer may emit one kind of departure from what a
/// product states (`ALLOW`, and report it) or refuses to write it (`STRICT`).
#[pyclass(module = "sidereon._sidereon", name = "ClockWriteLeniency", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyClockWriteLeniency {
    /// Refuse to write, naming what cannot be stated.
    STRICT,
    /// Write, and report the departure as a `ClockWriteDeparture`.
    ALLOW,
}

impl From<CoreClockWriteLeniency> for PyClockWriteLeniency {
    fn from(leniency: CoreClockWriteLeniency) -> Self {
        match leniency {
            CoreClockWriteLeniency::Strict => Self::STRICT,
            CoreClockWriteLeniency::Allow => Self::ALLOW,
        }
    }
}

impl From<PyClockWriteLeniency> for CoreClockWriteLeniency {
    fn from(leniency: PyClockWriteLeniency) -> Self {
        match leniency {
            PyClockWriteLeniency::STRICT => Self::Strict,
            PyClockWriteLeniency::ALLOW => Self::Allow,
        }
    }
}

/// The policy the RINEX clock writer applies to departures from what a
/// product states.
///
/// The default allows none, so `to_rinex_string_with_policy` with it writes
/// exactly what `to_rinex_string` writes, or refuses the same product. Values
/// are never approximated under any policy: a value no 19-column field states
/// exactly is refused.
#[pyclass(module = "sidereon._sidereon", name = "ClockWritePolicy")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PyClockWritePolicy {
    inner: CoreClockWritePolicy,
}

#[pymethods]
impl PyClockWritePolicy {
    /// Build a policy; every departure defaults to `STRICT`.
    #[new]
    #[pyo3(signature = (*, nearest_microsecond_epochs=PyClockWriteLeniency::STRICT))]
    fn new(nearest_microsecond_epochs: PyClockWriteLeniency) -> Self {
        Self {
            inner: CoreClockWritePolicy::strict()
                .with_nearest_microsecond_epochs(nearest_microsecond_epochs.into()),
        }
    }

    /// A policy that allows no departure.
    #[staticmethod]
    fn strict() -> Self {
        Self {
            inner: CoreClockWritePolicy::strict(),
        }
    }

    /// A policy that allows every departure.
    #[staticmethod]
    fn lenient() -> Self {
        Self {
            inner: CoreClockWritePolicy::lenient(),
        }
    }

    /// Whether an epoch no microsecond text states exactly is written as the
    /// nearest microsecond text (`ALLOW`) or refused (`STRICT`).
    #[getter]
    fn nearest_microsecond_epochs(&self) -> PyClockWriteLeniency {
        self.inner.nearest_microsecond_epochs.into()
    }

    /// This policy with `nearest_microsecond_epochs` replaced.
    fn with_nearest_microsecond_epochs(&self, leniency: PyClockWriteLeniency) -> Self {
        Self {
            inner: self.inner.with_nearest_microsecond_epochs(leniency.into()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockWritePolicy(nearest_microsecond_epochs=ClockWriteLeniency.{:?})",
            PyClockWriteLeniency::from(self.inner.nearest_microsecond_epochs)
        )
    }

    fn __eq__(&self, other: &PyClockWritePolicy) -> bool {
        self.inner == other.inner
    }
}

/// A departure from what a product states that the writer emitted under a
/// `ClockWritePolicy`.
///
/// `kind` is the core variant name. `EpochAtNearestMicrosecond` names the
/// record by its index in `RinexClock.records` order (`record`), the name it
/// is written with, the epoch the product holds (`epoch`, None when the record
/// has no instant) and the epoch fields as written, separated by single blanks
/// (`written`).
#[pyclass(module = "sidereon._sidereon", name = "ClockWriteDeparture")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyClockWriteDeparture {
    inner: CoreClockWriteDeparture,
}

#[pymethods]
impl PyClockWriteDeparture {
    /// Departure kind, the core variant name.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            CoreClockWriteDeparture::EpochAtNearestMicrosecond { .. } => {
                "EpochAtNearestMicrosecond".to_string()
            }
            other => debug_variant_name(other),
        }
    }

    /// Index of the record in `RinexClock.records` order.
    #[getter]
    fn record(&self) -> Option<usize> {
        match &self.inner {
            CoreClockWriteDeparture::EpochAtNearestMicrosecond { record, .. } => Some(*record),
            _ => None,
        }
    }

    /// Satellite or receiver name the record is written with.
    #[getter]
    fn name(&self) -> Option<String> {
        match &self.inner {
            CoreClockWriteDeparture::EpochAtNearestMicrosecond { name, .. } => Some(name.clone()),
            _ => None,
        }
    }

    /// The epoch the product holds, when the record has an instant.
    #[getter]
    fn epoch(&self) -> Option<PyClockInstant> {
        match &self.inner {
            CoreClockWriteDeparture::EpochAtNearestMicrosecond { epoch, .. } => {
                epoch.map(|inner| PyClockInstant { inner })
            }
            _ => None,
        }
    }

    /// The epoch fields as written: year, month, day, hour, minute and
    /// seconds, separated by single blanks.
    #[getter]
    fn written(&self) -> Option<String> {
        match &self.inner {
            CoreClockWriteDeparture::EpochAtNearestMicrosecond { written, .. } => {
                Some(written.clone())
            }
            _ => None,
        }
    }

    /// Formatted core departure text.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// Dictionary containing every field of this departure.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            CoreClockWriteDeparture::EpochAtNearestMicrosecond {
                record,
                name,
                epoch,
                written,
            } => {
                dict.set_item("record", *record)?;
                dict.set_item("name", name.as_str())?;
                dict.set_item("epoch", epoch.map(|inner| PyClockInstant { inner }))?;
                dict.set_item("written", written.as_str())?;
            }
            other => {
                dict.set_item("debug", format!("{other:?}"))?;
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockWriteDeparture(kind=\"{}\", message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &PyClockWriteDeparture) -> bool {
        self.inner == other.inner
    }
}

/// The text a policy-aware RINEX clock write produced, with every departure
/// the policy allowed and the writer emitted, in record order.
///
/// Unpacks as `(text, departures)`.
#[pyclass(module = "sidereon._sidereon", name = "ClockWriteResult")]
#[derive(Clone)]
pub struct PyClockWriteResult {
    text: String,
    departures: Vec<PyClockWriteDeparture>,
}

#[pymethods]
impl PyClockWriteResult {
    /// The RINEX clock text.
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    /// The departures written, in record order; empty when the text states
    /// exactly what the product holds.
    #[getter]
    fn departures(&self) -> Vec<PyClockWriteDeparture> {
        self.departures.clone()
    }

    fn __len__(&self) -> usize {
        2
    }

    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        PyTuple::new(
            py,
            [
                self.text.clone().into_pyobject(py)?.into_any(),
                self.departures.clone().into_pyobject(py)?.into_any(),
            ],
        )?
        .try_iter()
    }

    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<PyObject> {
        match idx {
            0 | -2 => Ok(self.text.clone().into_pyobject(py)?.into_any().unbind()),
            1 | -1 => Ok(self
                .departures
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => Err(PyIndexError::new_err("ClockWriteResult index out of range")),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockWriteResult(bytes={}, departures={})",
            self.text.len(),
            self.departures.len()
        )
    }
}

/// How a product's time system was established.
///
/// `kind` is the core variant name: `Declared` (a `TIME SYSTEM ID` record),
/// `Defaulted` (no such record; the RINEX clock 3.00 default applies),
/// `Unrecognized` (a label no system has, in `label`), `Conflicting` (several
/// records naming different systems, their labels in file order in `labels`)
/// or `Constructed` (built from rows in a stated time scale).
#[pyclass(module = "sidereon._sidereon", name = "ClockTimeSystemStatus")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyClockTimeSystemStatus {
    inner: CoreClockTimeSystemStatus,
}

#[pymethods]
impl PyClockTimeSystemStatus {
    /// Status kind, the core variant name.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            CoreClockTimeSystemStatus::Declared => "Declared".to_string(),
            CoreClockTimeSystemStatus::Defaulted => "Defaulted".to_string(),
            CoreClockTimeSystemStatus::Unrecognized { .. } => "Unrecognized".to_string(),
            CoreClockTimeSystemStatus::Conflicting { .. } => "Conflicting".to_string(),
            CoreClockTimeSystemStatus::Constructed => "Constructed".to_string(),
            other => debug_variant_name(other),
        }
    }

    /// The unrecognized label as written; None for every other kind.
    #[getter]
    fn label(&self) -> Option<String> {
        match &self.inner {
            CoreClockTimeSystemStatus::Unrecognized { label } => Some(label.clone()),
            _ => None,
        }
    }

    /// The distinct conflicting labels in file order; None for every other
    /// kind.
    #[getter]
    fn labels(&self) -> Option<Vec<String>> {
        match &self.inner {
            CoreClockTimeSystemStatus::Conflicting { labels } => Some(labels.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("ClockTimeSystemStatus({:?})", self.inner)
    }

    fn __eq__(&self, other: &PyClockTimeSystemStatus) -> bool {
        self.inner == other.inner
    }
}

/// A finding about how a product was read that does not stop it being read.
///
/// `kind` is the core variant name: `TimeSystemDefaulted` (`system`),
/// `TimeSystemMissing`, `TimeSystemWithoutScale` (`system`),
/// `HeaderRecordNonconforming`, `HeaderRecordUninterpreted` and
/// `HeaderRecordUnknownLabel` (`line`), and `SurplusValues`,
/// `OtherLayoutRecords` and `WhitespaceRecords` (`records`, `first_line`).
/// Accessors a kind does not carry return None.
#[pyclass(module = "sidereon._sidereon", name = "ClockNotice")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyClockNotice {
    inner: CoreRinexClockNotice,
}

#[pymethods]
impl PyClockNotice {
    /// Notice kind, the core variant name.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            CoreRinexClockNotice::TimeSystemDefaulted { .. } => "TimeSystemDefaulted".to_string(),
            CoreRinexClockNotice::TimeSystemMissing => "TimeSystemMissing".to_string(),
            CoreRinexClockNotice::TimeSystemWithoutScale { .. } => {
                "TimeSystemWithoutScale".to_string()
            }
            CoreRinexClockNotice::HeaderRecordNonconforming { .. } => {
                "HeaderRecordNonconforming".to_string()
            }
            CoreRinexClockNotice::HeaderRecordUninterpreted { .. } => {
                "HeaderRecordUninterpreted".to_string()
            }
            CoreRinexClockNotice::HeaderRecordUnknownLabel { .. } => {
                "HeaderRecordUnknownLabel".to_string()
            }
            CoreRinexClockNotice::SurplusValues { .. } => "SurplusValues".to_string(),
            CoreRinexClockNotice::OtherLayoutRecords { .. } => "OtherLayoutRecords".to_string(),
            CoreRinexClockNotice::WhitespaceRecords { .. } => "WhitespaceRecords".to_string(),
            other => debug_variant_name(other),
        }
    }

    /// The time system a time-system notice names.
    #[getter]
    fn system(&self) -> PyResult<Option<PyClockTimeSystem>> {
        match &self.inner {
            CoreRinexClockNotice::TimeSystemDefaulted { system }
            | CoreRinexClockNotice::TimeSystemWithoutScale { system } => {
                clock_time_system_to_py(*system).map(Some)
            }
            _ => Ok(None),
        }
    }

    /// One-based line number a header-record notice names.
    #[getter]
    fn line(&self) -> Option<usize> {
        match &self.inner {
            CoreRinexClockNotice::HeaderRecordNonconforming { line }
            | CoreRinexClockNotice::HeaderRecordUninterpreted { line }
            | CoreRinexClockNotice::HeaderRecordUnknownLabel { line } => Some(*line),
            _ => None,
        }
    }

    /// Number of records a record notice counts.
    #[getter]
    fn records(&self) -> Option<usize> {
        match &self.inner {
            CoreRinexClockNotice::SurplusValues { records, .. }
            | CoreRinexClockNotice::OtherLayoutRecords { records, .. }
            | CoreRinexClockNotice::WhitespaceRecords { records, .. } => Some(*records),
            _ => None,
        }
    }

    /// One-based line number of the first record a record notice counts.
    #[getter]
    fn first_line(&self) -> Option<usize> {
        match &self.inner {
            CoreRinexClockNotice::SurplusValues { first_line, .. }
            | CoreRinexClockNotice::OtherLayoutRecords { first_line, .. }
            | CoreRinexClockNotice::WhitespaceRecords { first_line, .. } => Some(*first_line),
            _ => None,
        }
    }

    /// Formatted core notice text.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// Dictionary containing every field of this notice.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            CoreRinexClockNotice::TimeSystemDefaulted { system }
            | CoreRinexClockNotice::TimeSystemWithoutScale { system } => {
                dict.set_item("system", clock_time_system_to_py(*system)?)?;
            }
            CoreRinexClockNotice::TimeSystemMissing => {}
            CoreRinexClockNotice::HeaderRecordNonconforming { line }
            | CoreRinexClockNotice::HeaderRecordUninterpreted { line }
            | CoreRinexClockNotice::HeaderRecordUnknownLabel { line } => {
                dict.set_item("line", *line)?;
            }
            CoreRinexClockNotice::SurplusValues {
                records,
                first_line,
            }
            | CoreRinexClockNotice::OtherLayoutRecords {
                records,
                first_line,
            }
            | CoreRinexClockNotice::WhitespaceRecords {
                records,
                first_line,
            } => {
                dict.set_item("records", *records)?;
                dict.set_item("first_line", *first_line)?;
            }
            other => {
                dict.set_item("debug", format!("{other:?}"))?;
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockNotice(kind=\"{}\", message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &PyClockNotice) -> bool {
        self.inner == other.inner
    }
}

/// The typed reading of one header record's fields.
///
/// `kind` is the core variant name and `details()` that variant's fields under
/// their own keys:
///
/// - `VersionType`: `version`, `file_type`, `satellite_system`
/// - `ProgramRunByDate`: `program`, `run_by`, `date`
/// - `Comment`, `StationClockRef`: `text`
/// - `ObservationTypes`: `system` and `count` (None on a continuation line),
///   `descriptors`
/// - `TimeSystem`: `label`
/// - `LeapSeconds`, `LeapSecondsGnss`: `seconds`
/// - `DcbsApplied`, `PcvsApplied`: `system`, `program`, `source`
/// - `TypesOfData`: `count`, `types`
/// - `StationNameNum`: `name`, `identifier`
/// - `AnalysisCenter`: `designator`, `name`
/// - `ClockRefCount`: `count`, `start` and `stop` (`ClockEpoch`, None when
///   blank)
/// - `AnalysisClockRef`: `name`, `identifier`, `constraint_s` (None when blank)
/// - `SolutionStationCount`: `count`, `frame`
/// - `SolutionStation`: `name`, `identifier`, `xyz_mm` (three integers)
/// - `SolutionSatelliteCount`: `count`
/// - `PrnList`: `prns`
/// - `EndOfHeader`: no fields
#[pyclass(module = "sidereon._sidereon", name = "ClockHeaderField")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyClockHeaderField {
    inner: CoreClockHeaderField,
}

#[pymethods]
impl PyClockHeaderField {
    /// Field kind, the core variant name.
    #[getter]
    fn kind(&self) -> String {
        let name = match &self.inner {
            CoreClockHeaderField::VersionType { .. } => "VersionType",
            CoreClockHeaderField::ProgramRunByDate { .. } => "ProgramRunByDate",
            CoreClockHeaderField::Comment(_) => "Comment",
            CoreClockHeaderField::ObservationTypes { .. } => "ObservationTypes",
            CoreClockHeaderField::TimeSystem { .. } => "TimeSystem",
            CoreClockHeaderField::LeapSeconds(_) => "LeapSeconds",
            CoreClockHeaderField::LeapSecondsGnss(_) => "LeapSecondsGnss",
            CoreClockHeaderField::DcbsApplied { .. } => "DcbsApplied",
            CoreClockHeaderField::PcvsApplied { .. } => "PcvsApplied",
            CoreClockHeaderField::TypesOfData { .. } => "TypesOfData",
            CoreClockHeaderField::StationNameNum { .. } => "StationNameNum",
            CoreClockHeaderField::StationClockRef(_) => "StationClockRef",
            CoreClockHeaderField::AnalysisCenter { .. } => "AnalysisCenter",
            CoreClockHeaderField::ClockRefCount { .. } => "ClockRefCount",
            CoreClockHeaderField::AnalysisClockRef { .. } => "AnalysisClockRef",
            CoreClockHeaderField::SolutionStationCount { .. } => "SolutionStationCount",
            CoreClockHeaderField::SolutionStation { .. } => "SolutionStation",
            CoreClockHeaderField::SolutionSatelliteCount(_) => "SolutionSatelliteCount",
            CoreClockHeaderField::PrnList(_) => "PrnList",
            CoreClockHeaderField::EndOfHeader => "EndOfHeader",
            other => return debug_variant_name(other),
        };
        name.to_string()
    }

    /// Dictionary containing every field of this reading.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            CoreClockHeaderField::VersionType {
                version,
                file_type,
                satellite_system,
            } => {
                dict.set_item("version", *version)?;
                dict.set_item("file_type", file_type.as_str())?;
                dict.set_item("satellite_system", satellite_system.as_str())?;
            }
            CoreClockHeaderField::ProgramRunByDate {
                program,
                run_by,
                date,
            } => {
                dict.set_item("program", program.as_str())?;
                dict.set_item("run_by", run_by.as_str())?;
                dict.set_item("date", date.as_str())?;
            }
            CoreClockHeaderField::Comment(text) | CoreClockHeaderField::StationClockRef(text) => {
                dict.set_item("text", text.as_str())?;
            }
            CoreClockHeaderField::ObservationTypes {
                system,
                count,
                descriptors,
            } => {
                dict.set_item("system", system.map(String::from))?;
                dict.set_item("count", *count)?;
                dict.set_item("descriptors", descriptors.clone())?;
            }
            CoreClockHeaderField::TimeSystem { label } => {
                dict.set_item("label", label.as_str())?;
            }
            CoreClockHeaderField::LeapSeconds(seconds)
            | CoreClockHeaderField::LeapSecondsGnss(seconds) => {
                dict.set_item("seconds", *seconds)?;
            }
            CoreClockHeaderField::DcbsApplied {
                system,
                program,
                source,
            }
            | CoreClockHeaderField::PcvsApplied {
                system,
                program,
                source,
            } => {
                dict.set_item("system", system.as_str())?;
                dict.set_item("program", program.as_str())?;
                dict.set_item("source", source.as_str())?;
            }
            CoreClockHeaderField::TypesOfData { count, types } => {
                dict.set_item("count", *count)?;
                dict.set_item("types", types.clone())?;
            }
            CoreClockHeaderField::StationNameNum { name, identifier } => {
                dict.set_item("name", name.as_str())?;
                dict.set_item("identifier", identifier.as_str())?;
            }
            CoreClockHeaderField::AnalysisCenter { designator, name } => {
                dict.set_item("designator", designator.as_str())?;
                dict.set_item("name", name.as_str())?;
            }
            CoreClockHeaderField::ClockRefCount { count, start, stop } => {
                dict.set_item("count", *count)?;
                dict.set_item("start", start.map(PyClockEpoch::from_core))?;
                dict.set_item("stop", stop.map(PyClockEpoch::from_core))?;
            }
            CoreClockHeaderField::AnalysisClockRef {
                name,
                identifier,
                constraint_s,
            } => {
                dict.set_item("name", name.as_str())?;
                dict.set_item("identifier", identifier.as_str())?;
                dict.set_item("constraint_s", *constraint_s)?;
            }
            CoreClockHeaderField::SolutionStationCount { count, frame } => {
                dict.set_item("count", *count)?;
                dict.set_item("frame", frame.as_str())?;
            }
            CoreClockHeaderField::SolutionStation {
                name,
                identifier,
                xyz_mm,
            } => {
                dict.set_item("name", name.as_str())?;
                dict.set_item("identifier", identifier.as_str())?;
                dict.set_item("xyz_mm", xyz_mm.to_vec())?;
            }
            CoreClockHeaderField::SolutionSatelliteCount(count) => {
                dict.set_item("count", *count)?;
            }
            CoreClockHeaderField::PrnList(prns) => {
                dict.set_item("prns", prns.clone())?;
            }
            CoreClockHeaderField::EndOfHeader => {}
            other => {
                dict.set_item("debug", format!("{other:?}"))?;
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!("ClockHeaderField(kind=\"{}\")", self.kind())
    }

    fn __eq__(&self, other: &PyClockHeaderField) -> bool {
        self.inner == other.inner
    }
}

/// One header line with its exact text and its typed reading.
#[pyclass(module = "sidereon._sidereon", name = "ClockHeaderRecord")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyClockHeaderRecord {
    inner: CoreClockHeaderRecord,
}

#[pymethods]
impl PyClockHeaderRecord {
    /// One-based line number in the text the product was read from; None for
    /// a line written by an edit.
    #[getter]
    fn line(&self) -> Option<usize> {
        self.inner.line()
    }

    /// The complete line without its terminator, exactly as it is written.
    #[getter]
    fn text(&self) -> &str {
        self.inner.text()
    }

    /// The header label, or for an unknown label the text in the label
    /// columns.
    #[getter]
    fn label(&self) -> &str {
        self.inner.label()
    }

    /// Zero-based column where the label starts.
    #[getter]
    fn label_column(&self) -> usize {
        self.inner.label_column()
    }

    /// The text before the label.
    #[getter]
    fn payload(&self) -> &str {
        self.inner.payload()
    }

    /// The typed reading, when the fields read; None otherwise.
    #[getter]
    fn field(&self) -> Option<PyClockHeaderField> {
        self.inner
            .field()
            .cloned()
            .map(|inner| PyClockHeaderField { inner })
    }

    /// How the fields were read: `Columns`, `OtherVersionColumns`,
    /// `Whitespace`, `Uninterpreted` or `UnknownLabel`.
    #[getter]
    fn reading(&self) -> String {
        let reading = self.inner.reading();
        match reading {
            CoreClockHeaderReading::Columns => "Columns".to_string(),
            CoreClockHeaderReading::OtherVersionColumns => "OtherVersionColumns".to_string(),
            CoreClockHeaderReading::Whitespace => "Whitespace".to_string(),
            CoreClockHeaderReading::Uninterpreted => "Uninterpreted".to_string(),
            CoreClockHeaderReading::UnknownLabel => "UnknownLabel".to_string(),
            other => debug_variant_name(&other),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockHeaderRecord(line={:?}, label={:?})",
            self.inner.line(),
            self.inner.label()
        )
    }

    fn __eq__(&self, other: &PyClockHeaderRecord) -> bool {
        self.inner == other.inner
    }
}

/// How a data record line was read.
///
/// `kind` is `Columns` (read at the columns of `layout`; a record read at the
/// columns of the layout its file does not declare reports that layout),
/// `Whitespace` (read as whitespace-separated values) or `Edited` (built or
/// edited through the typed API and written in the product's layout).
/// `layout` is None for every kind but `Columns`.
#[pyclass(module = "sidereon._sidereon", name = "ClockRecordReading")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyClockRecordReading {
    inner: CoreClockRecordReading,
}

#[pymethods]
impl PyClockRecordReading {
    /// Reading kind, the core variant name.
    #[getter]
    fn kind(&self) -> String {
        match self.inner {
            CoreClockRecordReading::Columns(_) => "Columns".to_string(),
            CoreClockRecordReading::Whitespace => "Whitespace".to_string(),
            CoreClockRecordReading::Edited => "Edited".to_string(),
            other => debug_variant_name(&other),
        }
    }

    /// The layout whose columns the line was read at.
    #[getter]
    fn layout(&self) -> Option<PyClockLayout> {
        match self.inner {
            CoreClockRecordReading::Columns(layout) => Some(layout.into()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("ClockRecordReading({:?})", self.inner)
    }

    fn __eq__(&self, other: &PyClockRecordReading) -> bool {
        self.inner == other.inner
    }
}

/// A value present in a record beyond its declared count.
///
/// `position` is the zero-based place in the record's value sequence: 0 bias,
/// 1 bias sigma, 2 rate, 3 rate sigma, 4 acceleration, 5 acceleration sigma.
#[pyclass(module = "sidereon._sidereon", name = "ClockSurplusValue")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyClockSurplusValue {
    inner: CoreClockSurplusValue,
}

#[pymethods]
impl PyClockSurplusValue {
    /// Zero-based position in the record's value sequence.
    #[getter]
    fn position(&self) -> usize {
        self.inner.position
    }

    /// The value, exactly as read.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockSurplusValue(position={}, value={:?})",
            self.inner.position, self.inner.value
        )
    }

    fn __eq__(&self, other: &PyClockSurplusValue) -> bool {
        self.inner == other.inner
    }
}

/// One data record with its typed reading.
///
/// A record read from a file keeps its source lines in the product; this view
/// is derived from them. `values` are the declared values, bias first; values
/// present beyond the declared count are kept separately in `surplus_values`.
///
/// `ClockRecord(record_type, name, epoch, values)` builds a record to insert
/// with `RinexClock.insert_record`. `values` holds the bias followed by up to
/// five further values in the Table A16 order. An `AS` name must be a
/// satellite identifier and is stored in its canonical spelling; other names
/// must be a single ASCII token of at most nine characters. The epoch's second
/// is taken as the shortest decimal that reads back to the given float, with no
/// digit rounded; insertion refuses an epoch finer than the microsecond the
/// epoch field states, and one that names no epoch in the product's time
/// system. The core refusal raises `RinexClockEditError`.
#[pyclass(module = "sidereon._sidereon", name = "ClockRecord")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyClockRecord {
    inner: CoreClockRecord,
}

#[pymethods]
impl PyClockRecord {
    #[new]
    fn new(
        py: Python<'_>,
        record_type: PyClockRecordType,
        name: &str,
        epoch: &PyClockEpoch,
        values: Vec<f64>,
    ) -> PyResult<Self> {
        CoreClockRecord::new(record_type.into(), name, epoch.inner, values)
            .map(|inner| Self { inner })
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Record type.
    #[getter]
    fn record_type(&self) -> PyClockRecordType {
        self.inner.record_type().into()
    }

    /// Receiver or satellite name as written, trimmed.
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    /// Canonical satellite identifier of an `AS` record; None for other types.
    #[getter]
    fn satellite(&self) -> Option<&str> {
        self.inner.satellite()
    }

    /// Civil epoch. The record keeps every digit its seconds field states;
    /// the float second here is the nearest double to it.
    #[getter]
    fn civil_epoch(&self) -> PyClockEpoch {
        PyClockEpoch::from_core(self.inner.civil_epoch())
    }

    /// The epoch as an instant in the product's time scale; None when the
    /// product's time system resolves to no time scale.
    #[getter]
    fn epoch(&self) -> Option<PyClockInstant> {
        self.inner.epoch().map(|inner| PyClockInstant { inner })
    }

    /// Declared number of values.
    #[getter]
    fn declared_count(&self) -> usize {
        self.inner.declared_count()
    }

    /// Declared values, bias first, exactly as read.
    #[getter]
    fn values(&self) -> Vec<f64> {
        self.inner.values().to_vec()
    }

    /// Clock bias, seconds (the first declared value).
    #[getter]
    fn bias_s(&self) -> f64 {
        self.inner.bias_s()
    }

    /// Declared values after the bias.
    #[getter]
    fn additional_values(&self) -> Vec<f64> {
        self.inner.additional_values().to_vec()
    }

    /// Values present beyond the declared count.
    #[getter]
    fn surplus_values(&self) -> Vec<PyClockSurplusValue> {
        self.inner
            .surplus_values()
            .iter()
            .map(|&inner| PyClockSurplusValue { inner })
            .collect()
    }

    /// One-based line number of the record's first line in the source; None
    /// for a record built or edited through the typed API.
    #[getter]
    fn line(&self) -> Option<usize> {
        self.inner.line()
    }

    /// Number of physical source lines the record spans, including blank
    /// lines between the record and its continuation line.
    #[getter]
    fn line_count(&self) -> usize {
        self.inner.line_count()
    }

    /// How the record's first line was read.
    #[getter]
    fn reading(&self) -> PyClockRecordReading {
        PyClockRecordReading {
            inner: self.inner.reading(),
        }
    }

    /// How the continuation line was read; None when the record has none.
    #[getter]
    fn continuation_reading(&self) -> Option<PyClockRecordReading> {
        self.inner
            .continuation_reading()
            .map(|inner| PyClockRecordReading { inner })
    }

    /// The satellite clock sample of an `AS` record with a resolved epoch;
    /// None otherwise.
    fn clock_point(&self) -> Option<PyClockPoint> {
        self.inner.clock_point().map(|inner| PyClockPoint { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockRecord(record_type={}, name={:?}, line={:?}, values={})",
            self.inner.record_type(),
            self.inner.name(),
            self.inner.line(),
            self.inner.values().len()
        )
    }

    fn __eq__(&self, other: &PyClockRecord) -> bool {
        self.inner == other.inner
    }
}

/// A record the product retains outside the satellite series.
///
/// `AR`, `CR`, `DR` and `MS` records are reported here, one per logical record
/// at its parent line. The record itself is retained: `RinexClock.records`
/// returns it with its typed values, and `to_rinex_string` restates it.
#[pyclass(module = "sidereon._sidereon", name = "ClockSkippedRecord")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PyClockSkippedRecord {
    inner: CoreRinexClockSkip,
}

#[pymethods]
impl PyClockSkippedRecord {
    /// One-based input line number of the record's parent line.
    #[getter]
    fn line(&self) -> usize {
        self.inner.line
    }

    /// Two-letter RINEX clock record type, such as `"AR"` or `"MS"`.
    #[getter]
    fn record_type(&self) -> &str {
        &self.inner.record_type
    }

    /// The core report text for this record.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockSkippedRecord(line={}, record_type='{}')",
            self.inner.line, self.inner.record_type
        )
    }

    fn __eq__(&self, other: &PyClockSkippedRecord) -> bool {
        self.inner == other.inner
    }
}

/// A line a lossy read kept without reading it as a record, or a header
/// time-system error, with its typed error.
///
/// Lossy parsing keeps such lines verbatim and `to_rinex_string` restates
/// them. Strict parsing refuses the file on the first one instead, so a
/// strictly parsed product has none.
#[pyclass(module = "sidereon._sidereon", name = "ClockDiagnostic")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyClockDiagnostic {
    inner: CoreRinexClockDiagnostic,
}

#[pymethods]
impl PyClockDiagnostic {
    /// One-based line number of the line that did not read.
    #[getter]
    fn line(&self) -> usize {
        self.inner.line
    }

    /// The typed core error for the line.
    #[getter]
    fn error(&self) -> PyRinexClockErrorDetail {
        PyRinexClockErrorDetail {
            inner: self.inner.error.clone(),
        }
    }

    /// The core diagnostic text for this line.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockDiagnostic(line={}, kind=\"{}\")",
            self.inner.line,
            self.error().kind()
        )
    }

    fn __eq__(&self, other: &PyClockDiagnostic) -> bool {
        self.inner == other.inner
    }
}

/// A scale-tagged instant, exactly as the core stores it, also used by RINEX clock epochs.
///
/// Every stored sample carries the core `Instant`: a `TimeScale` plus either a
/// split Julian date (`jd_whole` with `fraction`) or integer nanoseconds
/// (`nanos`). `repr_kind` names which one. The two halves of a split date are
/// reported separately and never summed here: adding them discards the
/// precision the split exists to keep.
///
/// `nanos` is the raw core count and carries this epoch's own `scale`. The core
/// clock reader and writer interpret it as nanoseconds from J2000
/// (2000-01-01 12:00:00) in that scale, not from the unix epoch; it is a
/// different quantity from `Instant.unix_micros`, which is a UTC unix stamp.
/// The RINEX clock parser only ever builds split-Julian-date epochs, so a
/// parsed product reports `nanos` as None.
#[pyclass(module = "sidereon._sidereon", name = "ClockInstant")]
#[derive(Clone, Copy, Debug)]
pub struct PyClockInstant {
    inner: CoreInstant,
}

impl PyClockInstant {
    pub(crate) fn from_core(inner: CoreInstant) -> Self {
        Self { inner }
    }

    pub(crate) fn to_core(self) -> CoreInstant {
        self.inner
    }
}

#[pymethods]
impl PyClockInstant {
    #[staticmethod]
    fn from_nanos(scale: PyTimeScale, nanos: i128) -> Self {
        Self {
            inner: CoreInstant::from_nanos(scale.into(), nanos),
        }
    }

    #[staticmethod]
    fn from_split_julian_date(scale: PyTimeScale, jd_whole: f64, fraction: f64) -> PyResult<Self> {
        let split = sidereon_core::astro::time::JulianDateSplit::new(jd_whole, fraction)
            .map_err(|error| crate::time_model_error::py_error(error, error.to_string()))?;
        Ok(Self {
            inner: CoreInstant::from_julian_date(scale.into(), split),
        })
    }

    /// Build a clock instant from civil fields read in `scale`.
    ///
    /// The second is taken as the shortest decimal that reads back to the
    /// given float, with every digit kept, as the core reads a record's
    /// seconds field. A `23:59:60` label is accepted for UTC on a day that
    /// ends with a positive leap second; every other scale is continuous and
    /// refuses it.
    ///
    /// The core civil helper returns an optional instant and produces no error
    /// payload of its own, so invalid fields raise a plain `ValueError` with no
    /// `detail` rather than a fabricated typed refusal.
    #[staticmethod]
    fn from_civil(
        scale: PyTimeScale,
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: f64,
    ) -> PyResult<Self> {
        core_civil_to_clock_instant(
            CoreTimeScale::from(scale),
            year,
            month,
            day,
            hour,
            minute,
            second,
        )
        .map(|inner| Self { inner })
        .ok_or_else(|| {
            PyValueError::new_err("invalid civil clock epoch fields for the given time scale")
        })
    }

    /// The time scale this epoch is expressed in.
    #[getter]
    fn scale(&self) -> PyTimeScale {
        self.inner.scale.into()
    }

    /// Core representation tag, `"JulianDate"` or `"Nanos"`.
    #[getter]
    fn repr_kind(&self) -> &'static str {
        match self.inner.repr {
            InstantRepr::JulianDate(_) => "JulianDate",
            InstantRepr::Nanos(_) => "Nanos",
        }
    }

    /// Whole part of the split Julian date, or None for a `Nanos` epoch.
    #[getter]
    fn jd_whole(&self) -> Option<f64> {
        match self.inner.repr {
            InstantRepr::JulianDate(split) => Some(split.jd_whole),
            InstantRepr::Nanos(_) => None,
        }
    }

    /// Day fraction of the split Julian date, or None for a `Nanos` epoch.
    #[getter]
    fn fraction(&self) -> Option<f64> {
        match self.inner.repr {
            InstantRepr::JulianDate(split) => Some(split.fraction),
            InstantRepr::Nanos(_) => None,
        }
    }

    /// Integer nanoseconds, or None for a split-Julian-date epoch.
    #[getter]
    fn nanos(&self) -> Option<i128> {
        match self.inner.repr {
            InstantRepr::Nanos(nanos) => Some(nanos),
            InstantRepr::JulianDate(_) => None,
        }
    }

    /// Seconds since the GPS epoch, or None when the core declines to project.
    ///
    /// The core projection is the one `ClockPoint.gps_seconds` uses: it is
    /// available for a GPST or QZSST split-Julian-date epoch (QZSST shares the
    /// GPST alignment to TAI) and returns None for every other scale, and for
    /// a `Nanos` epoch of any scale.
    #[getter]
    fn gps_seconds(&self) -> Option<f64> {
        clock_instant_gps_seconds(self.inner)
    }

    fn __repr__(&self) -> String {
        match self.inner.repr {
            InstantRepr::JulianDate(split) => format!(
                "ClockInstant(scale=TimeScale.{}, jd_whole={}, fraction={})",
                self.inner.scale.abbrev(),
                split.jd_whole,
                split.fraction
            ),
            InstantRepr::Nanos(nanos) => format!(
                "ClockInstant(scale=TimeScale.{}, nanos={})",
                self.inner.scale.abbrev(),
                nanos
            ),
        }
    }

    fn __eq__(&self, other: &PyClockInstant) -> bool {
        self.inner == other.inner
    }
}

/// One RINEX clock sample, with every value its record declared.
///
/// `bias_s` is the satellite clock bias in seconds. `additional_values` is the
/// remaining declared values in the core order: bias sigma (s), clock rate
/// (dimensionless), clock rate sigma (dimensionless), clock acceleration
/// (s^-1), clock acceleration sigma (s^-1). A record declaring fewer values
/// has a shorter list, and the named accessors report None for a value the
/// record did not declare. A declared zero - including a signed zero - is data
/// and is kept as such. Values a record carries beyond its declared count are
/// not included; `ClockRecord.surplus_values` reports them.
///
/// `ClockPoint(epoch, bias_s, additional_values=None)` builds a point for
/// `RinexClock.from_clock_points`; the core validates it there, or through
/// `validate()`.
#[pyclass(module = "sidereon._sidereon", name = "ClockPoint")]
#[derive(Clone, Debug)]
pub struct PyClockPoint {
    inner: CoreClockPoint,
}

impl PyClockPoint {
    fn additional_value(&self, index: usize) -> Option<f64> {
        self.inner.additional_values.get(index).copied()
    }
}

#[pymethods]
impl PyClockPoint {
    #[new]
    #[pyo3(signature = (epoch, bias_s, additional_values=None))]
    fn new(epoch: &PyClockInstant, bias_s: f64, additional_values: Option<Vec<f64>>) -> Self {
        Self {
            inner: CoreClockPoint::new(epoch.inner, bias_s, additional_values.unwrap_or_default()),
        }
    }

    /// The sample epoch, scale-tagged and in its native representation.
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant {
            inner: self.inner.epoch,
        }
    }

    /// Satellite clock bias in seconds.
    #[getter]
    fn bias_s(&self) -> f64 {
        self.inner.bias_s
    }

    /// Values following the bias, in core order, exactly as declared.
    #[getter]
    fn additional_values(&self) -> Vec<f64> {
        self.inner.additional_values.clone()
    }

    /// Number of values this record declared, the RINEX count field: the bias
    /// plus its additional values.
    #[getter]
    fn value_count(&self) -> usize {
        1 + self.inner.additional_values.len()
    }

    /// Clock bias sigma in seconds, or None when not declared.
    #[getter]
    fn bias_sigma_s(&self) -> Option<f64> {
        self.additional_value(0)
    }

    /// Clock rate (dimensionless), or None when not declared.
    #[getter]
    fn rate(&self) -> Option<f64> {
        self.additional_value(1)
    }

    /// Clock rate sigma (dimensionless), or None when not declared.
    #[getter]
    fn rate_sigma(&self) -> Option<f64> {
        self.additional_value(2)
    }

    /// Clock acceleration in s^-1, or None when not declared.
    #[getter]
    fn acceleration_per_s(&self) -> Option<f64> {
        self.additional_value(3)
    }

    /// Clock acceleration sigma in s^-1, or None when not declared.
    #[getter]
    fn acceleration_sigma_per_s(&self) -> Option<f64> {
        self.additional_value(4)
    }

    /// This sample's epoch as GPS seconds, or None when the core declines to
    /// project it: every sample that is not a GPST or QZSST split-Julian-date
    /// epoch.
    #[getter]
    fn gps_seconds(&self) -> Option<f64> {
        self.inner.gps_seconds()
    }

    /// Check the point as the core does before building a product from it: a
    /// valid epoch instant, a finite bias, at most five finite additional
    /// values. A refusal raises `RinexClockEditError`.
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        self.inner
            .validate()
            .map_err(|err| to_clock_edit_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockPoint(bias_s={}, value_count={}, scale=TimeScale.{})",
            self.inner.bias_s,
            self.value_count(),
            self.inner.epoch.scale.abbrev()
        )
    }

    fn __eq__(&self, other: &PyClockPoint) -> bool {
        self.inner == other.inner
    }
}

/// Civil epoch fields in a RINEX clock product's time system.
///
/// `RinexClock.clock_s` reads the fields in the *product's* time system, as
/// the core does, and `ClockRecord` takes them as a record epoch: the same
/// `ClockEpoch` names a UTC label in a UTC product and a GPST label in a GPS
/// product.
///
/// The constructor refuses only fields that name no civil epoch in any RINEX
/// clock time system: it checks them as the core checks a record epoch, which
/// accepts `23:59:60` on a day that ends with a positive leap second. Whether a
/// leap-second label names an epoch in a given product is the core's decision
/// when the epoch is used: a UTC or `GLO` product answers it, a GPS product
/// refuses it. The constructor raises a plain `ValueError` with no `detail`,
/// since the core civil helper produces no error payload.
///
/// `gps_seconds` is the GPST projection of these fields, computed by the core
/// civil helper, and is the value `clock_s_at_gps_seconds` takes. It is None
/// for fields that name no GPS-time epoch, such as a leap-second label. It
/// describes the fields read as GPS time, which is what they mean in a GPST or
/// QZSST product only; to query another scale by an explicitly tagged instant,
/// build `ClockInstant.from_civil` and call `clock_s_at_instant`.
#[pyclass(module = "sidereon._sidereon", name = "ClockEpoch")]
#[derive(Clone, Copy, Debug)]
pub struct PyClockEpoch {
    inner: CoreClockEpoch,
    gps_seconds: Option<f64>,
}

impl PyClockEpoch {
    fn from_core(inner: CoreClockEpoch) -> Self {
        let gps_seconds = core_civil_to_gps_seconds(
            inner.year,
            inner.month,
            inner.day,
            inner.hour,
            inner.minute,
            inner.second,
        );
        Self { inner, gps_seconds }
    }
}

#[pymethods]
impl PyClockEpoch {
    /// Build civil epoch fields for a RINEX clock record or query.
    #[new]
    fn new(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> PyResult<Self> {
        // The core checks a record epoch (`ClockRecord::new`) under the
        // UTC-like civil policy, the widest any RINEX clock time system uses;
        // the UTC civil helper applies that same check.
        core_civil_to_clock_instant(CoreTimeScale::Utc, year, month, day, hour, minute, second)
            .ok_or_else(|| {
                PyValueError::new_err("invalid civil clock epoch fields for RINEX clock records")
            })?;
        Ok(Self::from_core(CoreClockEpoch {
            year,
            month,
            day,
            hour,
            minute,
            second,
        }))
    }

    /// Calendar year.
    #[getter]
    fn year(&self) -> i32 {
        self.inner.year
    }

    /// Calendar month, 1..12.
    #[getter]
    fn month(&self) -> u8 {
        self.inner.month
    }

    /// Calendar day of month, 1..31.
    #[getter]
    fn day(&self) -> u8 {
        self.inner.day
    }

    /// Hour of day, 0..23.
    #[getter]
    fn hour(&self) -> u8 {
        self.inner.hour
    }

    /// Minute of hour, 0..59.
    #[getter]
    fn minute(&self) -> u8 {
        self.inner.minute
    }

    /// Seconds of minute, including fractional seconds; 60.x on a leap-second
    /// label.
    #[getter]
    fn second(&self) -> f64 {
        self.inner.second
    }

    /// These civil fields read as GPS time, seconds since 1980-01-06 00:00:00;
    /// None when they name no GPS-time epoch.
    #[getter]
    fn gps_seconds(&self) -> Option<f64> {
        self.gps_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockEpoch({:04}-{:02}-{:02}T{:02}:{:02}:{:?})",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second
        )
    }

    fn __eq__(&self, other: &PyClockEpoch) -> bool {
        self.inner == other.inner
    }
}

/// Per-satellite RINEX clock samples, in stored time order.
///
/// Every sample of the satellite series is kept, whatever its time scale.
/// `points` is the complete typed view and `epochs` its scale-tagged epochs;
/// `bias_s`, `gps_seconds`, `gps_seconds_valid` and `additional_values` are
/// row-aligned with it and with each other, and `len()` is the number of
/// samples.
///
/// `gps_seconds` is a numpy `(n,)` float64 convenience holding seconds since
/// the GPS epoch. A sample the core declines to project - any sample that is
/// not a GPST or QZSST split-Julian-date epoch - reads NaN there, with False in
/// the `gps_seconds_valid` bool array. That is the sample being unavailable in
/// GPS seconds, not the sample being absent: it is present in `points`, in
/// `epochs`, and in `bias_s`, with its own time scale.
#[pyclass(module = "sidereon._sidereon", name = "ClockSeries")]
#[derive(Clone)]
pub struct PyClockSeries {
    satellite: String,
    points: Vec<CoreClockPoint>,
    gps_seconds: Vec<f64>,
    gps_seconds_valid: Vec<bool>,
    bias_s: Vec<f64>,
}

impl PyClockSeries {
    fn from_points(satellite: String, points: &[CoreClockPoint]) -> Self {
        let mut gps_seconds = Vec::with_capacity(points.len());
        let mut gps_seconds_valid = Vec::with_capacity(points.len());
        let mut bias_s = Vec::with_capacity(points.len());
        for point in points {
            match point.gps_seconds() {
                Some(seconds) => {
                    gps_seconds.push(seconds);
                    gps_seconds_valid.push(true);
                }
                None => {
                    gps_seconds.push(f64::NAN);
                    gps_seconds_valid.push(false);
                }
            }
            bias_s.push(point.bias_s);
        }
        Self {
            satellite,
            points: points.to_vec(),
            gps_seconds,
            gps_seconds_valid,
            bias_s,
        }
    }
}

#[pymethods]
impl PyClockSeries {
    /// RINEX satellite token such as `"G05"`.
    #[getter]
    fn satellite(&self) -> &str {
        &self.satellite
    }

    /// Sample times as a numpy `(n,)` array, seconds since the GPS epoch, NaN
    /// where the core declines to project the epoch.
    #[getter]
    fn gps_seconds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.gps_seconds)
    }

    /// Row-aligned numpy `(n,)` bool array, False where `gps_seconds` is NaN
    /// because the sample has no GPS-seconds projection.
    #[getter]
    fn gps_seconds_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.gps_seconds_valid)
    }

    /// Satellite clock-bias samples as a numpy `(n,)` array, seconds.
    #[getter]
    fn bias_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.bias_s)
    }

    /// The complete stored samples, in time order.
    #[getter]
    fn points(&self) -> Vec<PyClockPoint> {
        self.points
            .iter()
            .map(|point| PyClockPoint {
                inner: point.clone(),
            })
            .collect()
    }

    /// The stored sample epochs, scale-tagged and row-aligned with `points`.
    #[getter]
    fn epochs(&self) -> Vec<PyClockInstant> {
        self.points
            .iter()
            .map(|point| PyClockInstant { inner: point.epoch })
            .collect()
    }

    /// Per-sample values following the bias, row-aligned with `points`.
    #[getter]
    fn additional_values(&self) -> Vec<Vec<f64>> {
        self.points
            .iter()
            .map(|point| point.additional_values.clone())
            .collect()
    }

    /// Number of clock samples for this satellite.
    fn __len__(&self) -> usize {
        self.points.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockSeries(satellite='{}', samples={})",
            self.satellite,
            self.points.len()
        )
    }
}

/// A RINEX clock product.
///
/// A product read from text keeps the text as its authority: every header
/// line with its exact label and payload, and every body line in order,
/// including blank lines, records of every type (`AR`, `AS`, `CR`, `DR`,
/// `MS`), continuation lines and, in a lossy read, lines that do not read as a
/// record. `header_records`, `records`, `series`, `skipped_records`,
/// `diagnostics` and `notices` are views the core derives from those lines.
/// `to_rinex_string` on an unedited product restates its input byte for byte,
/// line terminators included.
///
/// Edits go through `set_time_system`, `set_record_values`, `insert_record`,
/// `remove_record`, `retain_records` and `edit_records`. The core validates the
/// whole change first; a refused edit raises `RinexClockEditError` and changes
/// nothing, so an accepted edit never leaves the product unwritable.
///
/// `time_scale` is None when the time system is missing, unrecognized,
/// conflicting or has no core time scale (`IRN`); record epochs then keep their
/// civil fields without an instant. Equality compares the retained text and
/// records, so a product rebuilt from series rows does not equal the parsed
/// product it came from; compare `series` for that.
#[pyclass(module = "sidereon._sidereon", name = "RinexClock")]
#[derive(Clone)]
pub struct PyRinexClock {
    inner: CoreRinexClock,
}

#[pymethods]
impl PyRinexClock {
    /// Rebuild a GPST product from `(satellite, [(gps_seconds, bias_s), ...])`
    /// rows, the shape `series_rows` returns.
    ///
    /// Each satellite's GPS seconds must be finite, strictly increasing and
    /// inside the civil years 1 through 9999. A refusal raises
    /// `RinexClockEditError`.
    #[staticmethod]
    fn from_series_rows(py: Python<'_>, rows: Vec<(String, Vec<(f64, f64)>)>) -> PyResult<Self> {
        CoreRinexClock::from_series_rows(rows)
            .map(|inner| Self { inner })
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Build a product in `time_scale` from
    /// `(satellite, [(ClockInstant, bias_s), ...])` rows, the shape
    /// `instant_series_rows` returns.
    #[staticmethod]
    fn from_instant_series_rows(
        py: Python<'_>,
        time_scale: PyTimeScale,
        rows: Vec<(String, Vec<(PyClockInstant, f64)>)>,
    ) -> PyResult<Self> {
        let rows = rows
            .into_iter()
            .map(|(satellite, points)| {
                (
                    satellite,
                    points
                        .into_iter()
                        .map(|(epoch, bias_s)| (epoch.inner, bias_s))
                        .collect(),
                )
            })
            .collect();
        CoreRinexClock::from_instant_series_rows(CoreTimeScale::from(time_scale), rows)
            .map(|inner| Self { inner })
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Build a product in `time_scale` from `(satellite, [ClockPoint, ...])`
    /// rows, keeping every declared value of each point.
    ///
    /// Each satellite's points must be strictly increasing in time; a satellite
    /// named in more than one row keeps the points of every row. The records
    /// are written in the layout the time scale takes (3.00 for GPST, GST, UTC
    /// and TAI; 3.04 for QZSST and BDT). A time scale no RINEX clock time
    /// system names, GLONASS system time among them, builds a product whose
    /// `to_rinex_string` raises `RinexClockWriteError`. A refused point raises
    /// `RinexClockEditError`.
    #[staticmethod]
    fn from_clock_points(
        py: Python<'_>,
        time_scale: PyTimeScale,
        rows: Vec<(String, Vec<PyClockPoint>)>,
    ) -> PyResult<Self> {
        let rows = rows
            .into_iter()
            .map(|(satellite, points)| {
                (
                    satellite,
                    points.into_iter().map(|point| point.inner).collect(),
                )
            })
            .collect();
        CoreRinexClock::from_clock_points(CoreTimeScale::from(time_scale), rows)
            .map(|inner| Self { inner })
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Declared format version; for a product built from rows, the version it
    /// is written in. None when no version is declared.
    #[getter]
    fn version(&self) -> Option<f64> {
        self.inner.version()
    }

    /// Column layout records are read and written in; None when the text
    /// declares no version or a built product's time scale has no RINEX clock
    /// time system.
    #[getter]
    fn layout(&self) -> Option<PyClockLayout> {
        self.inner.layout().map(PyClockLayout::from)
    }

    /// Satellite system code of the `RINEX VERSION / TYPE` record (`G`, `R`,
    /// `E`, `C`, `I`, `J`, `S` or `M`); None when none is written.
    #[getter]
    fn satellite_system(&self) -> Option<String> {
        self.inner.satellite_system().map(String::from)
    }

    /// The product's time system, when one is declared, defaulted or built
    /// in.
    #[getter]
    fn time_system(&self) -> PyResult<Option<PyClockTimeSystem>> {
        self.inner
            .time_system()
            .map(clock_time_system_to_py)
            .transpose()
    }

    /// How the time system was established.
    #[getter]
    fn time_system_status(&self) -> PyClockTimeSystemStatus {
        PyClockTimeSystemStatus {
            inner: self.inner.time_system_status().clone(),
        }
    }

    /// The time scale record epochs are interpreted in; None when the time
    /// system is missing, unrecognized, conflicting or has no core time scale.
    #[getter]
    fn time_scale(&self) -> Option<PyTimeScale> {
        self.inner.time_scale().map(PyTimeScale::from)
    }

    /// Every header line in order with its typed reading. A product built
    /// from rows has no header lines; its header is written from its time
    /// scale.
    #[getter]
    fn header_records(&self) -> Vec<PyClockHeaderRecord> {
        self.inner
            .header_records()
            .into_iter()
            .map(|inner| PyClockHeaderRecord { inner })
            .collect()
    }

    /// Every data record in order, including records repeated for one name
    /// and epoch. Lines a lossy read could not read are not records;
    /// `diagnostics` names them.
    #[getter]
    fn records(&self) -> Vec<PyClockRecord> {
        self.inner
            .records()
            .map(|inner| PyClockRecord { inner })
            .collect()
    }

    /// Number of data records.
    #[getter]
    fn record_count(&self) -> usize {
        self.inner.record_count()
    }

    /// One line of the text the product was read from, by one-based line
    /// number, without its terminator; None past the last line.
    fn source_line(&self, line: usize) -> Option<String> {
        self.inner.source_line(line).map(str::to_owned)
    }

    /// Satellite tokens with at least one sample, in sort order.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner.series().keys().cloned().collect()
    }

    /// Per-satellite clock-bias series in satellite sort order, derived from
    /// the `AS` records whose epoch resolves to an instant. Where records
    /// repeat one satellite and instant, the last in file order is the
    /// sample; every such record remains in `records`.
    #[getter]
    fn series(&self) -> Vec<PyClockSeries> {
        self.inner
            .series()
            .iter()
            .map(|(satellite, points)| PyClockSeries::from_points(satellite.clone(), points))
            .collect()
    }

    /// Number of satellites with clock samples.
    #[getter]
    fn satellite_count(&self) -> usize {
        self.inner.series().len()
    }

    /// Total number of satellite clock samples.
    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.series().values().map(Vec::len).sum()
    }

    /// Records outside the satellite series, one per logical record at its
    /// parent line. Each is retained in `records`.
    #[getter]
    fn skipped_records(&self) -> Vec<PyClockSkippedRecord> {
        self.inner
            .skipped_records()
            .iter()
            .map(|skip| PyClockSkippedRecord {
                inner: skip.clone(),
            })
            .collect()
    }

    /// Number of records outside the satellite series.
    #[getter]
    fn skipped_record_count(&self) -> usize {
        self.inner.skipped_records().len()
    }

    /// Lines a lossy read kept without reading them as records, and header
    /// time-system errors, in input order with their typed core errors.
    #[getter]
    fn diagnostics(&self) -> Vec<PyClockDiagnostic> {
        self.inner
            .diagnostics()
            .iter()
            .map(|diagnostic| PyClockDiagnostic {
                inner: diagnostic.clone(),
            })
            .collect()
    }

    /// Number of diagnostics.
    #[getter]
    fn diagnostic_count(&self) -> usize {
        self.inner.diagnostics().len()
    }

    /// Findings about how the product was read that do not stop it being read.
    #[getter]
    fn notices(&self) -> Vec<PyClockNotice> {
        self.inner
            .notices()
            .iter()
            .map(|notice| PyClockNotice {
                inner: notice.clone(),
            })
            .collect()
    }

    /// Return one satellite's clock series, or `None` if the satellite is absent.
    fn series_for(&self, satellite_id: &str) -> Option<PyClockSeries> {
        self.inner
            .series()
            .get(satellite_id)
            .map(|points| PyClockSeries::from_points(satellite_id.to_string(), points))
    }

    /// GPST and QZSST samples as `[(satellite, [(gps_seconds, bias_s), ...]), ...]`.
    /// Samples on other time scales are omitted rather than converted.
    fn series_rows(&self) -> Vec<(String, Vec<(f64, f64)>)> {
        self.inner.series_rows()
    }

    /// Every sample as `[(satellite, [(ClockInstant, bias_s), ...]), ...]`.
    fn instant_series_rows(&self) -> Vec<(String, Vec<(PyClockInstant, f64)>)> {
        self.inner
            .instant_series_rows()
            .into_iter()
            .map(|(satellite, points)| {
                (
                    satellite,
                    points
                        .into_iter()
                        .map(|(inner, bias_s)| (PyClockInstant { inner }, bias_s))
                        .collect(),
                )
            })
            .collect()
    }

    /// Interpolate one satellite clock bias at a civil epoch.
    ///
    /// The epoch fields are read in this product's time scale, so a UTC or
    /// `GLO` product is queried by UTC labels, including a `23:59:60` label on
    /// a leap-second day, and a GPS product by GPST labels. Interpolation
    /// across a UTC leap second uses elapsed time. Returns None for an unknown
    /// satellite or an epoch outside the stored bracket. A refused epoch, and a
    /// product whose time system resolves to no time scale, raise
    /// `RinexClockQueryError` carrying the typed core `detail`.
    fn clock_s(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyClockEpoch,
    ) -> PyResult<Option<f64>> {
        self.inner
            .clock_s(satellite_id, epoch.inner)
            .map_err(|err| to_clock_query_err(py, err))
    }

    /// Interpolate one satellite clock bias at a scale-tagged instant.
    ///
    /// This is the exact query for a product whose scale is not GPST: build the
    /// instant with `ClockInstant.from_civil`, or reuse one of the product's
    /// own `ClockSeries.epochs`. An instant on a different timeline from the
    /// stored samples returns None rather than being converted. A refused
    /// instant raises `RinexClockQueryError` carrying the typed core `detail`.
    fn clock_s_at_instant(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyClockInstant,
    ) -> PyResult<Option<f64>> {
        self.inner
            .clock_s_at_instant(satellite_id, epoch.inner)
            .map_err(|err| to_clock_query_err(py, err))
    }

    /// Interpolate one satellite clock bias at GPS seconds.
    ///
    /// GPST and QZSST series answer. GPS seconds that are not finite, or are
    /// outside the civil years 1 through 9999, are refused by the core and
    /// raise `RinexClockQueryError` carrying the typed core `detail`.
    fn clock_s_at_gps_seconds(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        gps_seconds: f64,
    ) -> PyResult<Option<f64>> {
        self.inner
            .clock_s_at_gps_seconds(satellite_id, gps_seconds)
            .map_err(|err| to_clock_query_err(py, err))
    }

    /// Write the product as RINEX clock text.
    ///
    /// A product read from text restates every retained line byte for byte,
    /// including header records, blank lines, records of every type and, after
    /// a lossy read, lines that did not read. Records held as typed values are
    /// written in the product's layout; a value no 19-column field states
    /// exactly, an epoch no seconds field states exactly, and a time scale no
    /// RINEX clock time system names are refused, raising
    /// `RinexClockWriteError` carrying the typed core `detail`.
    fn to_rinex_string(&self, py: Python<'_>) -> PyResult<String> {
        self.inner
            .to_rinex_string()
            .map_err(|err| to_clock_write_err(py, err))
    }

    /// Write the product under `policy`, returning the text and every
    /// departure from what the product states that the policy allowed and the
    /// writer emitted.
    ///
    /// With `nearest_microsecond_epochs` allowed, an epoch no microsecond text
    /// states exactly is written as the nearest one and reported. Every other
    /// refusal of `to_rinex_string` still raises `RinexClockWriteError`.
    #[pyo3(signature = (policy=None))]
    fn to_rinex_string_with_policy(
        &self,
        py: Python<'_>,
        policy: Option<PyClockWritePolicy>,
    ) -> PyResult<PyClockWriteResult> {
        let policy = policy.map(|policy| policy.inner).unwrap_or_default();
        let (text, departures) = self
            .inner
            .to_rinex_string_with_policy(policy)
            .map_err(|err| to_clock_write_err(py, err))?;
        Ok(PyClockWriteResult {
            text,
            departures: departures
                .into_iter()
                .map(|inner| PyClockWriteDeparture { inner })
                .collect(),
        })
    }

    /// Declare the product's time system.
    ///
    /// Replaces every `TIME SYSTEM ID` record with one written at the columns
    /// of the product's layout, or inserts one where Table A15 orders it. Every
    /// record epoch is checked in the new time system first; a refusal (a
    /// `23:59:60` label in a continuous scale, a product with no header
    /// section, a product built from rows) raises `RinexClockEditError` and
    /// changes nothing.
    fn set_time_system(&mut self, py: Python<'_>, system: PyClockTimeSystem) -> PyResult<()> {
        self.inner
            .set_time_system(system.into())
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Replace the declared values of the record at `index` (in `records`
    /// order), bias first.
    ///
    /// The record keeps its type, name and epoch, including the exact text of
    /// its seconds field, and is then written in the product's layout. The
    /// edit is refused, and nothing changes, when the record could not then be
    /// written, when `values` does not restate values the source record
    /// carries beyond its declared count, or when no record has that index;
    /// the refusal raises `RinexClockEditError`.
    fn set_record_values(
        &mut self,
        py: Python<'_>,
        index: usize,
        values: Vec<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_record_values(index, values)
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Insert `record` before the record at `index` (in `records` order), or
    /// after the last record when `index` equals `record_count`.
    ///
    /// The record must be writable in the product's layout and its epoch valid
    /// in the product's time system, and it must carry no surplus values. A
    /// refusal raises `RinexClockEditError` and changes nothing.
    fn insert_record(
        &mut self,
        py: Python<'_>,
        index: usize,
        record: &PyClockRecord,
    ) -> PyResult<()> {
        self.inner
            .insert_record(index, record.inner.clone())
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Remove the record at `index` (in `records` order) with every line it
    /// spans, returning it. No record at `index` raises `RinexClockEditError`.
    fn remove_record(&mut self, py: Python<'_>, index: usize) -> PyResult<PyClockRecord> {
        self.inner
            .remove_record(index)
            .map(|inner| PyClockRecord { inner })
            .map_err(|err| to_clock_edit_err(py, err))
    }

    /// Keep the records for which `keep(record)` is true and remove every
    /// other, with every line it spans; returns the number removed. Blank and
    /// unread lines stay.
    ///
    /// `keep` is called once per record, in `records` order, before anything
    /// changes; an exception it raises propagates and leaves the product
    /// unchanged. The product cannot be read from inside `keep`.
    fn retain_records(&mut self, keep: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut decisions = Vec::with_capacity(self.inner.record_count());
        for record in self.inner.records() {
            let verdict = keep.call1((PyClockRecord { inner: record },))?;
            decisions.push(verdict.is_truthy()?);
        }
        let mut index = 0usize;
        Ok(self.inner.retain_records(|_| {
            // The core calls `keep` for exactly the records `records()`
            // yields, in the same order.
            let kept = decisions.get(index).copied().unwrap_or(true);
            index += 1;
            kept
        }))
    }

    /// Replace the declared values of every record for which `edit(record)`
    /// returns a sequence of floats, bias first; returns the number edited.
    /// `edit` returns None to leave a record unchanged.
    ///
    /// `edit` is called once per record, in `records` order, before anything
    /// changes; an exception it raises propagates and leaves the product
    /// unchanged. Each edit follows `set_record_values`, and the core checks
    /// the whole batch first: if one edit is refused, none is applied and the
    /// refusal raises `RinexClockEditError`.
    fn edit_records(&mut self, py: Python<'_>, edit: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut edits: Vec<Option<Vec<f64>>> = Vec::with_capacity(self.inner.record_count());
        for record in self.inner.records() {
            let result = edit.call1((PyClockRecord { inner: record },))?;
            edits.push(if result.is_none() {
                None
            } else {
                Some(result.extract::<Vec<f64>>()?)
            });
        }
        let mut index = 0usize;
        self.inner
            .edit_records(|_| {
                // The core visits the records `records()` yields, in the same
                // order.
                let values = edits.get_mut(index).and_then(Option::take);
                index += 1;
                values
            })
            .map_err(|err| to_clock_edit_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexClock(record_count={}, satellite_count={}, sample_count={})",
            self.inner.record_count(),
            self.inner.series().len(),
            self.inner.series().values().map(Vec::len).sum::<usize>()
        )
    }

    fn __eq__(&self, other: &PyRinexClock) -> bool {
        self.inner == other.inner
    }
}

fn parse_clock_text(py: Python<'_>, text: &str) -> PyResult<PyRinexClock> {
    Ok(PyRinexClock {
        inner: CoreRinexClock::parse(text).map_err(|err| to_clock_parse_err(py, err))?,
    })
}

fn parse_clock_text_lossy(text: &str) -> PyRinexClock {
    PyRinexClock {
        inner: CoreRinexClock::parse_lossy(text),
    }
}

/// Strictly parse RINEX clock text, failing on the first line that does not
/// read.
///
/// Every line is retained. A line that does not read raises
/// `RinexClockParseError` carrying the typed core `detail`.
#[pyfunction]
fn parse_rinex_clock(py: Python<'_>, text: &str) -> PyResult<PyRinexClock> {
    parse_clock_text(py, text)
}

/// Load and strictly parse a RINEX clock file from bytes, bytearray, or a path.
///
/// A line that does not read raises `RinexClockParseError` carrying the typed
/// core `detail`. A source that cannot be read, or is not UTF-8 text, fails
/// before the core parser runs and so carries no `detail`.
#[pyfunction]
fn load_rinex_clock(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<PyRinexClock> {
    let text = text_from_source(
        source,
        "load_rinex_clock",
        "RINEX clock source",
        RinexTextKind::Clock,
    )?;
    parse_clock_text(py, &text)
}

/// Parse RINEX clock text, keeping lines that do not read verbatim with a
/// diagnostic.
///
/// Nothing is dropped: `to_rinex_string` on the result restates the input
/// exactly.
#[pyfunction]
fn parse_rinex_clock_lossy(text: &str) -> PyRinexClock {
    parse_clock_text_lossy(text)
}

/// Load and lossily parse a RINEX clock file from bytes, bytearray, or a path.
#[pyfunction]
fn load_rinex_clock_lossy(source: &Bound<'_, PyAny>) -> PyResult<PyRinexClock> {
    let text = text_from_source(
        source,
        "load_rinex_clock_lossy",
        "RINEX clock source",
        RinexTextKind::Clock,
    )?;
    Ok(parse_clock_text_lossy(&text))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRinexClockErrorDetail>()?;
    m.add_class::<PyClockTimeSystem>()?;
    m.add_class::<PyClockLayout>()?;
    m.add_class::<PyClockRecordType>()?;
    m.add_class::<PyClockWriteLeniency>()?;
    m.add_class::<PyClockWritePolicy>()?;
    m.add_class::<PyClockWriteDeparture>()?;
    m.add_class::<PyClockWriteResult>()?;
    m.add_class::<PyClockTimeSystemStatus>()?;
    m.add_class::<PyClockNotice>()?;
    m.add_class::<PyClockHeaderField>()?;
    m.add_class::<PyClockHeaderRecord>()?;
    m.add_class::<PyClockRecordReading>()?;
    m.add_class::<PyClockSurplusValue>()?;
    m.add_class::<PyClockRecord>()?;
    m.add_class::<PyClockSkippedRecord>()?;
    m.add_class::<PyClockDiagnostic>()?;
    m.add_class::<PyClockInstant>()?;
    m.add_class::<PyClockPoint>()?;
    m.add_class::<PyClockEpoch>()?;
    m.add_class::<PyClockSeries>()?;
    m.add_class::<PyRinexClock>()?;
    m.add_function(wrap_pyfunction!(parse_rinex_clock, m)?)?;
    m.add_function(wrap_pyfunction!(load_rinex_clock, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_clock_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(load_rinex_clock_lossy, m)?)?;
    Ok(())
}
