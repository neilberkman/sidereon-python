//! CCSDS Tracking Data Message (TDM) Python bindings.
//!
//! Provides typed Python value objects for the CCSDS 503.0-B-2 KVN format,
//! strict and policy-aware parsing and serialization, positioned comments,
//! and atomic metadata editing backed solely by raw fields.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule};
use pyo3::PyClass;

use sidereon_core::astro::tdm::{
    encode_kvn, encode_kvn_with_policy, parse_kvn, parse_kvn_with_policy, Tdm, TdmComment,
    TdmDataRecord, TdmDataSection, TdmDeparture, TdmError, TdmField, TdmInputErrorKind,
    TdmLeniency, TdmMetadata, TdmObservable, TdmParticipant, TdmPath, TdmPolicy, TdmScalar,
    TdmSegment, TdmUnit, TdmWarning, TdmWritePolicy,
};

use crate::{TdmParseError, TdmValidationError, TdmWriteError};

fn attach_detail(py: Python<'_>, py_err: &PyErr, err: TdmError) -> PyResult<()> {
    let py_detail = PyTdmErrorDetail { inner: err }.into_pyobject(py)?;
    py_err.value(py).setattr("detail", py_detail)?;
    Ok(())
}

pub(crate) fn to_tdm_parse_err(py: Python<'_>, err: TdmError) -> PyErr {
    let msg = err.to_string();
    let py_err = TdmParseError::new_err(msg);
    if let Err(e) = attach_detail(py, &py_err, err) {
        return e;
    }
    py_err
}

pub(crate) fn to_tdm_write_err(py: Python<'_>, err: TdmError) -> PyErr {
    let msg = err.to_string();
    let py_err = TdmWriteError::new_err(msg);
    if let Err(e) = attach_detail(py, &py_err, err) {
        return e;
    }
    py_err
}

pub(crate) fn to_tdm_validation_err(py: Python<'_>, err: TdmError) -> PyErr {
    let msg = err.to_string();
    let py_err = TdmValidationError::new_err(msg);
    if let Err(e) = attach_detail(py, &py_err, err) {
        return e;
    }
    py_err
}

fn borrow_vec<T, U, F>(py: Python<'_>, values: Option<Vec<Py<U>>>, f: F) -> Vec<T>
where
    U: PyClass,
    F: Fn(&U) -> T,
{
    values
        .unwrap_or_default()
        .iter()
        .map(|value| {
            let borrowed = value.borrow(py);
            f(&*borrowed)
        })
        .collect()
}

fn extract_fields(py: Python<'_>, fields: &Bound<'_, PyAny>) -> PyResult<Vec<TdmField>> {
    let mut result = Vec::new();
    let iter = fields.try_iter()?;
    for item in iter {
        let item = item?;
        if let Ok(field_py) = item.extract::<Py<PyTdmField>>() {
            result.push(field_py.borrow(py).inner.clone());
        } else if let Ok((key, value)) = item.extract::<(String, String)>() {
            result.push(TdmField { key, value });
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "fields must contain TdmField instances or (key, value) pairs",
            ));
        }
    }
    Ok(result)
}

fn extract_comments(
    py: Python<'_>,
    comments: Option<&Bound<'_, PyAny>>,
    default_before_record: usize,
) -> PyResult<Vec<TdmComment>> {
    let Some(comments_obj) = comments else {
        return Ok(Vec::new());
    };
    if comments_obj.is_none() {
        return Ok(Vec::new());
    }
    let mut result = Vec::new();
    let iter = comments_obj.try_iter()?;
    for item in iter {
        let item = item?;
        if let Ok(comment_py) = item.extract::<Py<PyTdmComment>>() {
            result.push(comment_py.borrow(py).inner.clone());
        } else if let Ok(s) = item.extract::<String>() {
            result.push(TdmComment {
                text: s,
                before_record: default_before_record,
            });
        } else if let Ok((text, offset)) = item.extract::<(String, usize)>() {
            result.push(TdmComment {
                text,
                before_record: offset,
            });
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "comments must contain TdmComment, str, or (text, before_record) entries",
            ));
        }
    }
    Ok(result)
}

/// A TDM comment retaining its placement among records or fields.
#[pyclass(module = "sidereon._sidereon", name = "TdmComment")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmComment {
    pub(crate) inner: TdmComment,
}

impl From<TdmComment> for PyTdmComment {
    fn from(inner: TdmComment) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmComment {
    /// Build a positioned TDM comment.
    #[new]
    #[pyo3(signature = (text, before_record=0))]
    fn new(text: String, before_record: usize) -> Self {
        Self {
            inner: TdmComment {
                text,
                before_record,
            },
        }
    }

    /// Comment text (without the COMMENT keyword or leading space).
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    #[setter]
    fn set_text(&mut self, text: String) {
        self.inner.text = text;
    }

    /// Zero-based index of the field or record this comment precedes.
    #[getter]
    fn before_record(&self) -> usize {
        self.inner.before_record
    }

    #[setter]
    fn set_before_record(&mut self, before_record: usize) {
        self.inner.before_record = before_record;
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmComment(text={:?}, before_record={})",
            self.inner.text, self.inner.before_record
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A TDM KVN key/value field preserved in parse order.
#[pyclass(module = "sidereon._sidereon", name = "TdmField")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmField {
    pub(crate) inner: TdmField,
}

impl From<TdmField> for PyTdmField {
    fn from(inner: TdmField) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmField {
    /// Build a raw TDM KVN key/value field.
    #[new]
    fn new(key: String, value: String) -> Self {
        Self {
            inner: TdmField { key, value },
        }
    }

    /// KVN keyword.
    #[getter]
    fn key(&self) -> &str {
        &self.inner.key
    }

    #[setter]
    fn set_key(&mut self, key: String) {
        self.inner.key = key;
    }

    /// KVN value string.
    #[getter]
    fn value(&self) -> &str {
        &self.inner.value
    }

    #[setter]
    fn set_value(&mut self, value: String) {
        self.inner.value = value;
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmField(key={:?}, value={:?})",
            self.inner.key, self.inner.value
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// One named TDM tracking participant.
#[pyclass(module = "sidereon._sidereon", name = "TdmParticipant")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmParticipant {
    pub(crate) inner: TdmParticipant,
}

impl From<TdmParticipant> for PyTdmParticipant {
    fn from(inner: TdmParticipant) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmParticipant {
    /// Build a TDM participant entry.
    #[new]
    fn new(index: u8, name: String) -> Self {
        Self {
            inner: TdmParticipant { index, name },
        }
    }

    /// Numeric suffix from `PARTICIPANT_n`.
    #[getter]
    fn index(&self) -> u8 {
        self.inner.index
    }

    /// Participant name.
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmParticipant(index={}, name={:?})",
            self.inner.index, self.inner.name
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A parsed TDM signal path from `PATH`, `PATH_1`, or `PATH_2`.
#[pyclass(module = "sidereon._sidereon", name = "TdmPath")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmPath {
    pub(crate) inner: TdmPath,
}

impl From<TdmPath> for PyTdmPath {
    fn from(inner: TdmPath) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmPath {
    /// Build a TDM path entry.
    #[new]
    #[pyo3(signature = (key, participants, index=None))]
    fn new(key: String, participants: Vec<u8>, index: Option<u8>) -> Self {
        Self {
            inner: TdmPath {
                key,
                index,
                participants,
            },
        }
    }

    /// Original path keyword.
    #[getter]
    fn key(&self) -> &str {
        &self.inner.key
    }

    /// Path suffix for `PATH_n`, or `None` for `PATH`.
    #[getter]
    fn index(&self) -> Option<u8> {
        self.inner.index
    }

    /// Participant indices listed in path order, as a list of ints.
    ///
    /// A `Vec<u8>` converts to `bytes`, so the indices are built into a list
    /// one int at a time.
    #[getter]
    fn participants<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.participants.iter().copied())
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmPath(key={:?}, participants={:?})",
            self.inner.key, self.inner.participants
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

fn unit_from_label(label: &str) -> TdmUnit {
    match label {
        "km" => TdmUnit::Kilometers,
        "s" => TdmUnit::Seconds,
        "RU" => TdmUnit::RangeUnits,
        "km/s" => TdmUnit::KilometersPerSecond,
        "Hz" => TdmUnit::Hertz,
        "Hz/s" => TdmUnit::HertzPerSecond,
        "deg" => TdmUnit::Degrees,
        "dBW" => TdmUnit::DecibelWatts,
        "dBHz" => TdmUnit::DecibelHertz,
        "m**2" => TdmUnit::SquareMeters,
        "m" => TdmUnit::Meters,
        "s/s" => TdmUnit::SecondsPerSecond,
        "%" => TdmUnit::Percent,
        "K" => TdmUnit::Kelvin,
        "hPa" => TdmUnit::Hectopascals,
        "TECU" => TdmUnit::TotalElectronContentUnits,
        "n/a" => TdmUnit::Dimensionless,
        other => TdmUnit::Unknown(other.to_string()),
    }
}

/// Unit attached to a TDM tracking data record.
#[pyclass(module = "sidereon._sidereon", name = "TdmUnit")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmUnit {
    pub(crate) inner: TdmUnit,
}

impl From<TdmUnit> for PyTdmUnit {
    fn from(inner: TdmUnit) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmUnit {
    /// Build a TDM unit from its canonical label.
    #[new]
    fn new(label: String) -> Self {
        Self {
            inner: unit_from_label(&label),
        }
    }

    /// Canonical unit label.
    #[getter]
    fn label(&self) -> &str {
        self.inner.as_str()
    }

    fn __repr__(&self) -> String {
        format!("TdmUnit({:?})", self.inner.as_str())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Observable family for a TDM tracking data record.
#[pyclass(module = "sidereon._sidereon", name = "TdmObservable")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmObservable {
    pub(crate) inner: TdmObservable,
}

impl From<TdmObservable> for PyTdmObservable {
    fn from(inner: TdmObservable) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmObservable {
    /// A `RANGE` observable.
    #[staticmethod]
    fn range() -> Self {
        Self {
            inner: TdmObservable::Range,
        }
    }

    /// A `DOPPLER_INSTANTANEOUS` observable.
    #[staticmethod]
    fn doppler_instantaneous() -> Self {
        Self {
            inner: TdmObservable::DopplerInstantaneous,
        }
    }

    /// A `DOPPLER_INTEGRATED` observable.
    #[staticmethod]
    fn doppler_integrated() -> Self {
        Self {
            inner: TdmObservable::DopplerIntegrated,
        }
    }

    /// A `RECEIVE_FREQ` observable with optional participant suffix.
    #[staticmethod]
    #[pyo3(signature = (participant=None))]
    fn receive_freq(participant: Option<u8>) -> Self {
        Self {
            inner: TdmObservable::ReceiveFreq { participant },
        }
    }

    /// A `TRANSMIT_FREQ` observable with optional participant suffix.
    #[staticmethod]
    #[pyo3(signature = (participant=None))]
    fn transmit_freq(participant: Option<u8>) -> Self {
        Self {
            inner: TdmObservable::TransmitFreq { participant },
        }
    }

    /// A `TRANSMIT_FREQ_RATE` observable with optional participant suffix.
    #[staticmethod]
    #[pyo3(signature = (participant=None))]
    fn transmit_freq_rate(participant: Option<u8>) -> Self {
        Self {
            inner: TdmObservable::TransmitFreqRate { participant },
        }
    }

    /// An `ANGLE_1` observable.
    #[staticmethod]
    fn angle1() -> Self {
        Self {
            inner: TdmObservable::Angle1,
        }
    }

    /// An `ANGLE_2` observable.
    #[staticmethod]
    fn angle2() -> Self {
        Self {
            inner: TdmObservable::Angle2,
        }
    }

    /// A modeled-as-other TDM data keyword.
    #[staticmethod]
    fn other(name: String) -> Self {
        Self {
            inner: TdmObservable::Other(name),
        }
    }

    /// Stable lowercase observable family.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            TdmObservable::Range => "range",
            TdmObservable::DopplerInstantaneous => "doppler_instantaneous",
            TdmObservable::DopplerIntegrated => "doppler_integrated",
            TdmObservable::ReceiveFreq { .. } => "receive_freq",
            TdmObservable::TransmitFreq { .. } => "transmit_freq",
            TdmObservable::TransmitFreqRate { .. } => "transmit_freq_rate",
            TdmObservable::Angle1 => "angle1",
            TdmObservable::Angle2 => "angle2",
            TdmObservable::Other(_) => "other",
        }
    }

    /// Participant suffix for indexed frequency observables.
    #[getter]
    fn participant(&self) -> Option<u8> {
        match self.inner {
            TdmObservable::ReceiveFreq { participant }
            | TdmObservable::TransmitFreq { participant }
            | TdmObservable::TransmitFreqRate { participant } => participant,
            _ => None,
        }
    }

    /// Keyword carried by an `other` observable.
    #[getter]
    fn other_name(&self) -> Option<&str> {
        match &self.inner {
            TdmObservable::Other(name) => Some(name.as_str()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            TdmObservable::Other(name) => format!("TdmObservable.other({name:?})"),
            _ => format!("TdmObservable(kind={:?})", self.kind()),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A numeric TDM record value plus the exact decimal token used to encode it.
#[pyclass(module = "sidereon._sidereon", name = "TdmScalar")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdmScalar {
    pub(crate) inner: TdmScalar,
}

impl From<TdmScalar> for PyTdmScalar {
    fn from(inner: TdmScalar) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmScalar {
    /// Build a TDM scalar from its decimal token and parsed value.
    #[new]
    fn new(text: String, value: f64) -> Self {
        Self {
            inner: TdmScalar { text, value },
        }
    }

    /// Exact decimal or scientific-notation token read from the message.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    /// Parsed finite `f64` value.
    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmScalar(text={:?}, value={})",
            self.inner.text, self.inner.value
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// One time-tagged TDM tracking data record.
#[pyclass(module = "sidereon._sidereon", name = "TdmDataRecord")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdmDataRecord {
    pub(crate) inner: TdmDataRecord,
}

impl From<TdmDataRecord> for PyTdmDataRecord {
    fn from(inner: TdmDataRecord) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmDataRecord {
    /// Build a TDM data record from canonical pieces.
    #[new]
    fn new(
        observable: PyTdmObservable,
        keyword: String,
        epoch: String,
        value: PyTdmScalar,
        unit: PyTdmUnit,
    ) -> Self {
        Self {
            inner: TdmDataRecord {
                observable: observable.inner,
                keyword,
                epoch,
                value: value.inner,
                unit: unit.inner,
            },
        }
    }

    /// Parsed observable family.
    #[getter]
    fn observable(&self) -> PyTdmObservable {
        self.inner.observable.clone().into()
    }

    /// Original data keyword.
    #[getter]
    fn keyword(&self) -> &str {
        &self.inner.keyword
    }

    /// Raw epoch string.
    #[getter]
    fn epoch(&self) -> &str {
        &self.inner.epoch
    }

    /// Numeric record value and exact source token.
    #[getter]
    fn value(&self) -> PyTdmScalar {
        self.inner.value.clone().into()
    }

    /// Exact decimal or scientific-notation token read from the message.
    #[getter]
    fn value_text(&self) -> &str {
        &self.inner.value.text
    }

    /// Parsed finite `f64` value.
    #[getter]
    fn value_float(&self) -> f64 {
        self.inner.value.value
    }

    /// Unit assigned by CCSDS 503.0-B-2.
    #[getter]
    fn unit(&self) -> PyTdmUnit {
        self.inner.unit.clone().into()
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmDataRecord(keyword={:?}, epoch={:?}, value_text={:?})",
            self.inner.keyword, self.inner.epoch, self.inner.value.text
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A TDM data block.
#[pyclass(module = "sidereon._sidereon", name = "TdmDataSection")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdmDataSection {
    pub(crate) inner: TdmDataSection,
}

impl From<TdmDataSection> for PyTdmDataSection {
    fn from(inner: TdmDataSection) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmDataSection {
    /// Build a TDM data section.
    #[new]
    #[pyo3(signature = (records, comments=None))]
    fn new(
        py: Python<'_>,
        records: Vec<Py<PyTdmDataRecord>>,
        comments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let comments = extract_comments(py, comments, 0)?;
        Ok(Self {
            inner: TdmDataSection {
                comments,
                records: records
                    .iter()
                    .map(|record| record.borrow(py).inner.clone())
                    .collect(),
            },
        })
    }

    /// Data-section positioned comments.
    #[getter]
    fn comments(&self) -> Vec<PyTdmComment> {
        self.inner
            .comments
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[setter]
    fn set_comments(&mut self, py: Python<'_>, comments: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.comments = extract_comments(py, Some(comments), 0)?;
        Ok(())
    }

    /// Data records in message order.
    #[getter]
    fn records(&self) -> Vec<PyTdmDataRecord> {
        self.inner.records.iter().cloned().map(Into::into).collect()
    }

    #[setter]
    fn set_records(&mut self, py: Python<'_>, records: Vec<Py<PyTdmDataRecord>>) {
        self.inner.records = records.iter().map(|r| r.borrow(py).inner.clone()).collect();
    }

    /// Append a data record to the data section.
    fn add_record(&mut self, record: &PyTdmDataRecord) {
        self.inner.records.push(record.inner.clone());
    }

    /// Insert a data record at `index`.
    fn insert_record(&mut self, index: usize, record: &PyTdmDataRecord) -> PyResult<()> {
        if index > self.inner.records.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "record insert index out of range",
            ));
        }
        self.inner.records.insert(index, record.inner.clone());
        for c in self.inner.comments.iter_mut() {
            if c.before_record > index {
                c.before_record += 1;
            }
        }
        Ok(())
    }

    /// Remove a data record at `index`.
    fn remove_record(&mut self, index: usize) -> PyResult<PyTdmDataRecord> {
        if index >= self.inner.records.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "record remove index out of range",
            ));
        }
        let removed = self.inner.records.remove(index);
        for c in self.inner.comments.iter_mut() {
            if c.before_record > index {
                c.before_record -= 1;
            }
        }
        Ok(removed.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmDataSection(records={}, comments={})",
            self.inner.records.len(),
            self.inner.comments.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Leniency rule for parsing or encoding TDM departures.
#[pyclass(module = "sidereon._sidereon", name = "TdmLeniency", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum PyTdmLeniency {
    STRICT,
    FORGIVE,
}

impl From<PyTdmLeniency> for TdmLeniency {
    fn from(l: PyTdmLeniency) -> Self {
        match l {
            PyTdmLeniency::STRICT => Self::Strict,
            PyTdmLeniency::FORGIVE => Self::Forgive,
        }
    }
}

impl From<TdmLeniency> for PyTdmLeniency {
    fn from(l: TdmLeniency) -> Self {
        match l {
            TdmLeniency::Strict => Self::STRICT,
            TdmLeniency::Forgive => Self::FORGIVE,
        }
    }
}

#[pymethods]
impl PyTdmLeniency {
    #[classattr]
    #[pyo3(name = "Strict")]
    const STRICT_ALIAS: PyTdmLeniency = PyTdmLeniency::STRICT;
    #[classattr]
    #[pyo3(name = "Forgive")]
    const FORGIVE_ALIAS: PyTdmLeniency = PyTdmLeniency::FORGIVE;

    #[getter]
    fn name(&self) -> &'static str {
        match self {
            Self::STRICT => "STRICT",
            Self::FORGIVE => "FORGIVE",
        }
    }

    #[getter]
    fn value(&self) -> &'static str {
        match self {
            Self::STRICT => "strict",
            Self::FORGIVE => "forgive",
        }
    }
}

/// Reader policy for departures from CCSDS 503.0-B-2 (8 axes).
#[pyclass(module = "sidereon._sidereon", name = "TdmPolicy")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PyTdmPolicy {
    pub(crate) inner: TdmPolicy,
}

impl From<TdmPolicy> for PyTdmPolicy {
    fn from(inner: TdmPolicy) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmPolicy {
    /// Build a TdmPolicy with explicit axes, defaulting each to STRICT.
    #[new]
    #[pyo3(signature = (
        *,
        non_printable=None,
        missing_keywords=None,
        long_lines=None,
        empty_data_sections=None,
        record_order=None,
        duplicate_records=None,
        keyword_order=None,
        final_terminator=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        non_printable: Option<PyTdmLeniency>,
        missing_keywords: Option<PyTdmLeniency>,
        long_lines: Option<PyTdmLeniency>,
        empty_data_sections: Option<PyTdmLeniency>,
        record_order: Option<PyTdmLeniency>,
        duplicate_records: Option<PyTdmLeniency>,
        keyword_order: Option<PyTdmLeniency>,
        final_terminator: Option<PyTdmLeniency>,
    ) -> Self {
        let mut policy = TdmPolicy::strict();
        if let Some(v) = non_printable {
            policy = policy.with_non_printable(v.into());
        }
        if let Some(v) = missing_keywords {
            policy = policy.with_missing_keywords(v.into());
        }
        if let Some(v) = long_lines {
            policy = policy.with_long_lines(v.into());
        }
        if let Some(v) = empty_data_sections {
            policy = policy.with_empty_data_sections(v.into());
        }
        if let Some(v) = record_order {
            policy = policy.with_record_order(v.into());
        }
        if let Some(v) = duplicate_records {
            policy = policy.with_duplicate_records(v.into());
        }
        if let Some(v) = keyword_order {
            policy = policy.with_keyword_order(v.into());
        }
        if let Some(v) = final_terminator {
            policy = policy.with_final_terminator(v.into());
        }
        Self { inner: policy }
    }

    /// Strict default reader policy.
    #[staticmethod]
    fn strict() -> Self {
        Self {
            inner: TdmPolicy::strict(),
        }
    }

    /// Fully lenient reader policy.
    #[staticmethod]
    fn lenient() -> Self {
        Self {
            inner: TdmPolicy::lenient(),
        }
    }

    /// Default reader policy (strict).
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: TdmPolicy::default(),
        }
    }

    #[getter]
    fn non_printable(&self) -> PyTdmLeniency {
        self.inner.non_printable.into()
    }

    #[getter]
    fn missing_keywords(&self) -> PyTdmLeniency {
        self.inner.missing_keywords.into()
    }

    #[getter]
    fn long_lines(&self) -> PyTdmLeniency {
        self.inner.long_lines.into()
    }

    #[getter]
    fn empty_data_sections(&self) -> PyTdmLeniency {
        self.inner.empty_data_sections.into()
    }

    #[getter]
    fn record_order(&self) -> PyTdmLeniency {
        self.inner.record_order.into()
    }

    #[getter]
    fn duplicate_records(&self) -> PyTdmLeniency {
        self.inner.duplicate_records.into()
    }

    #[getter]
    fn keyword_order(&self) -> PyTdmLeniency {
        self.inner.keyword_order.into()
    }

    #[getter]
    fn final_terminator(&self) -> PyTdmLeniency {
        self.inner.final_terminator.into()
    }

    fn with_non_printable(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_non_printable(leniency.into()),
        }
    }

    fn with_missing_keywords(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_missing_keywords(leniency.into()),
        }
    }

    fn with_long_lines(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_long_lines(leniency.into()),
        }
    }

    fn with_empty_data_sections(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_empty_data_sections(leniency.into()),
        }
    }

    fn with_record_order(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_record_order(leniency.into()),
        }
    }

    fn with_duplicate_records(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_duplicate_records(leniency.into()),
        }
    }

    fn with_keyword_order(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_keyword_order(leniency.into()),
        }
    }

    fn with_final_terminator(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_final_terminator(leniency.into()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmPolicy(non_printable={:?}, missing_keywords={:?}, long_lines={:?}, empty_data_sections={:?}, record_order={:?}, duplicate_records={:?}, keyword_order={:?}, final_terminator={:?})",
            self.non_printable().name(),
            self.missing_keywords().name(),
            self.long_lines().name(),
            self.empty_data_sections().name(),
            self.record_order().name(),
            self.duplicate_records().name(),
            self.keyword_order().name(),
            self.final_terminator().name()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Writer policy for departures from CCSDS 503.0-B-2 (9 axes, including repeated_keywords).
#[pyclass(module = "sidereon._sidereon", name = "TdmWritePolicy")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PyTdmWritePolicy {
    pub(crate) inner: TdmWritePolicy,
}

impl From<TdmWritePolicy> for PyTdmWritePolicy {
    fn from(inner: TdmWritePolicy) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmWritePolicy {
    /// Build a TdmWritePolicy with explicit axes, defaulting each to STRICT.
    #[new]
    #[pyo3(signature = (
        *,
        non_printable=None,
        missing_keywords=None,
        long_lines=None,
        empty_data_sections=None,
        record_order=None,
        duplicate_records=None,
        keyword_order=None,
        final_terminator=None,
        repeated_keywords=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        non_printable: Option<PyTdmLeniency>,
        missing_keywords: Option<PyTdmLeniency>,
        long_lines: Option<PyTdmLeniency>,
        empty_data_sections: Option<PyTdmLeniency>,
        record_order: Option<PyTdmLeniency>,
        duplicate_records: Option<PyTdmLeniency>,
        keyword_order: Option<PyTdmLeniency>,
        final_terminator: Option<PyTdmLeniency>,
        repeated_keywords: Option<PyTdmLeniency>,
    ) -> Self {
        let mut policy = TdmWritePolicy::strict();
        if let Some(v) = non_printable {
            policy = policy.with_non_printable(v.into());
        }
        if let Some(v) = missing_keywords {
            policy = policy.with_missing_keywords(v.into());
        }
        if let Some(v) = long_lines {
            policy = policy.with_long_lines(v.into());
        }
        if let Some(v) = empty_data_sections {
            policy = policy.with_empty_data_sections(v.into());
        }
        if let Some(v) = record_order {
            policy = policy.with_record_order(v.into());
        }
        if let Some(v) = duplicate_records {
            policy = policy.with_duplicate_records(v.into());
        }
        if let Some(v) = keyword_order {
            policy = policy.with_keyword_order(v.into());
        }
        if let Some(v) = final_terminator {
            policy = policy.with_final_terminator(v.into());
        }
        if let Some(v) = repeated_keywords {
            policy = policy.with_repeated_keywords(v.into());
        }
        Self { inner: policy }
    }

    /// Strict default writer policy.
    #[staticmethod]
    fn strict() -> Self {
        Self {
            inner: TdmWritePolicy::strict(),
        }
    }

    /// Fully lenient writer policy.
    #[staticmethod]
    fn lenient() -> Self {
        Self {
            inner: TdmWritePolicy::lenient(),
        }
    }

    /// Default writer policy (strict).
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: TdmWritePolicy::default(),
        }
    }

    /// The reader policy mirrored by this write policy (dropping repeated_keywords).
    fn as_read(&self) -> PyTdmPolicy {
        PyTdmPolicy {
            inner: self.inner.as_read(),
        }
    }

    #[getter]
    fn non_printable(&self) -> PyTdmLeniency {
        self.inner.non_printable.into()
    }

    #[getter]
    fn missing_keywords(&self) -> PyTdmLeniency {
        self.inner.missing_keywords.into()
    }

    #[getter]
    fn long_lines(&self) -> PyTdmLeniency {
        self.inner.long_lines.into()
    }

    #[getter]
    fn empty_data_sections(&self) -> PyTdmLeniency {
        self.inner.empty_data_sections.into()
    }

    #[getter]
    fn record_order(&self) -> PyTdmLeniency {
        self.inner.record_order.into()
    }

    #[getter]
    fn duplicate_records(&self) -> PyTdmLeniency {
        self.inner.duplicate_records.into()
    }

    #[getter]
    fn keyword_order(&self) -> PyTdmLeniency {
        self.inner.keyword_order.into()
    }

    #[getter]
    fn final_terminator(&self) -> PyTdmLeniency {
        self.inner.final_terminator.into()
    }

    #[getter]
    fn repeated_keywords(&self) -> PyTdmLeniency {
        self.inner.repeated_keywords.into()
    }

    fn with_non_printable(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_non_printable(leniency.into()),
        }
    }

    fn with_missing_keywords(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_missing_keywords(leniency.into()),
        }
    }

    fn with_long_lines(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_long_lines(leniency.into()),
        }
    }

    fn with_empty_data_sections(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_empty_data_sections(leniency.into()),
        }
    }

    fn with_record_order(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_record_order(leniency.into()),
        }
    }

    fn with_duplicate_records(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_duplicate_records(leniency.into()),
        }
    }

    fn with_keyword_order(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_keyword_order(leniency.into()),
        }
    }

    fn with_final_terminator(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_final_terminator(leniency.into()),
        }
    }

    fn with_repeated_keywords(&self, leniency: PyTdmLeniency) -> Self {
        Self {
            inner: self.inner.with_repeated_keywords(leniency.into()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmWritePolicy(non_printable={:?}, missing_keywords={:?}, long_lines={:?}, empty_data_sections={:?}, record_order={:?}, duplicate_records={:?}, keyword_order={:?}, final_terminator={:?}, repeated_keywords={:?})",
            self.non_printable().name(),
            self.missing_keywords().name(),
            self.long_lines().name(),
            self.empty_data_sections().name(),
            self.record_order().name(),
            self.duplicate_records().name(),
            self.keyword_order().name(),
            self.final_terminator().name(),
            self.repeated_keywords().name()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A parser warning emitted when a departure from CCSDS 503.0-B-2 is forgiven.
#[pyclass(module = "sidereon._sidereon", name = "TdmWarning")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmWarning {
    pub(crate) inner: TdmWarning,
}

impl From<TdmWarning> for PyTdmWarning {
    fn from(inner: TdmWarning) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmWarning {
    /// Categorical warning identifier.
    #[getter]
    fn kind(&self) -> String {
        let label = match &self.inner {
            TdmWarning::NonPrintableCharacter { .. } => "non_printable_character",
            TdmWarning::LineTooLong { .. } => "line_too_long",
            TdmWarning::RepeatedKeyword { .. } => "repeated_keyword",
            TdmWarning::MissingKeyword { .. } => "missing_keyword",
            TdmWarning::EmptyDataSection { .. } => "empty_data_section",
            TdmWarning::RecordsOutOfOrder { .. } => "records_out_of_order",
            TdmWarning::UnterminatedFinalLine { .. } => "unterminated_final_line",
            TdmWarning::KeywordOutOfOrder { .. } => "keyword_out_of_order",
            TdmWarning::DuplicateRecord { .. } => "duplicate_record",
            other => return crate::marshal::debug_variant_snake(other),
        };
        label.to_string()
    }

    /// One-based source line number, if applicable.
    #[getter]
    fn line(&self) -> Option<usize> {
        match &self.inner {
            TdmWarning::NonPrintableCharacter { line, .. }
            | TdmWarning::LineTooLong { line, .. }
            | TdmWarning::RepeatedKeyword { line, .. }
            | TdmWarning::UnterminatedFinalLine { line, .. }
            | TdmWarning::KeywordOutOfOrder { line, .. } => Some(*line),
            _ => None,
        }
    }

    /// Associated KVN keyword, if applicable.
    #[getter]
    fn keyword(&self) -> Option<&str> {
        match &self.inner {
            TdmWarning::NonPrintableCharacter { keyword, .. }
            | TdmWarning::LineTooLong { keyword, .. }
            | TdmWarning::RepeatedKeyword { keyword, .. }
            | TdmWarning::MissingKeyword { keyword, .. }
            | TdmWarning::RecordsOutOfOrder { keyword, .. }
            | TdmWarning::KeywordOutOfOrder { keyword, .. }
            | TdmWarning::DuplicateRecord { keyword, .. } => Some(keyword.as_str()),
            _ => None,
        }
    }

    /// One-based character column for non-printable character warnings.
    #[getter]
    fn column(&self) -> Option<usize> {
        match &self.inner {
            TdmWarning::NonPrintableCharacter { column, .. } => Some(*column),
            _ => None,
        }
    }

    /// Offending character string for non-printable character warnings.
    #[getter]
    fn character(&self) -> Option<String> {
        match &self.inner {
            TdmWarning::NonPrintableCharacter { character, .. } => Some(character.to_string()),
            _ => None,
        }
    }

    /// Line character length for line_too_long warnings.
    #[getter]
    fn length(&self) -> Option<usize> {
        match &self.inner {
            TdmWarning::LineTooLong { length, .. } => Some(*length),
            _ => None,
        }
    }

    /// Section name ("header", "metadata", or "data") where the warning occurred.
    #[getter]
    fn section(&self) -> Option<&str> {
        match &self.inner {
            TdmWarning::RepeatedKeyword { section, .. }
            | TdmWarning::KeywordOutOfOrder { section, .. } => Some(section),
            _ => None,
        }
    }

    /// One-based segment index, if applicable.
    #[getter]
    fn segment(&self) -> Option<usize> {
        match &self.inner {
            TdmWarning::MissingKeyword { segment, .. } => *segment,
            TdmWarning::EmptyDataSection { segment, .. }
            | TdmWarning::RecordsOutOfOrder { segment, .. }
            | TdmWarning::DuplicateRecord { segment, .. } => Some(*segment),
            _ => None,
        }
    }

    /// Associated record timetag epoch, if applicable.
    #[getter]
    fn epoch(&self) -> Option<&str> {
        match &self.inner {
            TdmWarning::RecordsOutOfOrder { epoch, .. }
            | TdmWarning::DuplicateRecord { epoch, .. } => Some(epoch.as_str()),
            _ => None,
        }
    }

    /// Full human-readable diagnostic message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __str__(&self) -> String {
        self.message()
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmWarning(kind={:?}, message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A departure from CCSDS 503.0-B-2 emitted under an explicit write policy.
#[pyclass(module = "sidereon._sidereon", name = "TdmDeparture")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmDeparture {
    pub(crate) inner: TdmDeparture,
}

impl From<TdmDeparture> for PyTdmDeparture {
    fn from(inner: TdmDeparture) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmDeparture {
    /// Categorical departure identifier.
    #[getter]
    fn kind(&self) -> String {
        let label = match &self.inner {
            TdmDeparture::NonPrintableCharacter { .. } => "non_printable_character",
            TdmDeparture::LineTooLong { .. } => "line_too_long",
            TdmDeparture::MissingKeyword { .. } => "missing_keyword",
            TdmDeparture::EmptyDataSection { .. } => "empty_data_section",
            TdmDeparture::RecordsOutOfOrder { .. } => "records_out_of_order",
            TdmDeparture::DuplicateRecord { .. } => "duplicate_record",
            TdmDeparture::RepeatedKeyword { .. } => "repeated_keyword",
            TdmDeparture::KeywordOutOfOrder { .. } => "keyword_out_of_order",
            TdmDeparture::UnterminatedFinalLine => "unterminated_final_line",
            other => return crate::marshal::debug_variant_snake(other),
        };
        label.to_string()
    }

    /// Associated KVN keyword, if applicable.
    #[getter]
    fn keyword(&self) -> Option<&str> {
        match &self.inner {
            TdmDeparture::NonPrintableCharacter { keyword, .. }
            | TdmDeparture::LineTooLong { keyword, .. }
            | TdmDeparture::MissingKeyword { keyword, .. }
            | TdmDeparture::RecordsOutOfOrder { keyword, .. }
            | TdmDeparture::DuplicateRecord { keyword, .. }
            | TdmDeparture::RepeatedKeyword { keyword, .. }
            | TdmDeparture::KeywordOutOfOrder { keyword, .. } => Some(keyword.as_str()),
            _ => None,
        }
    }

    /// Offending character string for non-printable character departures.
    #[getter]
    fn character(&self) -> Option<String> {
        match &self.inner {
            TdmDeparture::NonPrintableCharacter { character, .. } => Some(character.to_string()),
            _ => None,
        }
    }

    /// Line character length for line_too_long departures.
    #[getter]
    fn length(&self) -> Option<usize> {
        match &self.inner {
            TdmDeparture::LineTooLong { length, .. } => Some(*length),
            _ => None,
        }
    }

    /// Section name ("header", "metadata", or "data") where the departure occurred.
    #[getter]
    fn section(&self) -> Option<&str> {
        match &self.inner {
            TdmDeparture::RepeatedKeyword { section, .. }
            | TdmDeparture::KeywordOutOfOrder { section, .. } => Some(section),
            _ => None,
        }
    }

    /// One-based segment index, if applicable.
    #[getter]
    fn segment(&self) -> Option<usize> {
        match &self.inner {
            TdmDeparture::MissingKeyword { segment, .. } => *segment,
            TdmDeparture::EmptyDataSection { segment, .. }
            | TdmDeparture::RecordsOutOfOrder { segment, .. }
            | TdmDeparture::DuplicateRecord { segment, .. } => Some(*segment),
            _ => None,
        }
    }

    /// Associated record timetag epoch, if applicable.
    #[getter]
    fn epoch(&self) -> Option<&str> {
        match &self.inner {
            TdmDeparture::RecordsOutOfOrder { epoch, .. }
            | TdmDeparture::DuplicateRecord { epoch, .. } => Some(epoch.as_str()),
            _ => None,
        }
    }

    /// Full human-readable diagnostic message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __str__(&self) -> String {
        self.message()
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmDeparture(kind={:?}, message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Category of fine-grained input validation failure.
#[pyclass(module = "sidereon._sidereon", name = "TdmInputErrorKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum PyTdmInputErrorKind {
    MISSING,
    FLOAT_PARSE,
    NON_FINITE,
    NOT_POSITIVE,
    OUT_OF_RANGE,
    INVALID_INDEX,
    UNKNOWN_KEYWORD,
    UNEXPECTED_UNIT,
    NON_INTEGER,
    NEGATIVE,
    NEGATIVE_ZERO,
    UNIT_MISMATCH,
    DECIMAL_MISMATCH,
    UNKNOWN,
}

impl From<TdmInputErrorKind> for PyTdmInputErrorKind {
    fn from(kind: TdmInputErrorKind) -> Self {
        match kind {
            TdmInputErrorKind::Missing => Self::MISSING,
            TdmInputErrorKind::FloatParse => Self::FLOAT_PARSE,
            TdmInputErrorKind::NonFinite => Self::NON_FINITE,
            TdmInputErrorKind::NotPositive => Self::NOT_POSITIVE,
            TdmInputErrorKind::OutOfRange => Self::OUT_OF_RANGE,
            TdmInputErrorKind::InvalidIndex => Self::INVALID_INDEX,
            TdmInputErrorKind::UnknownKeyword => Self::UNKNOWN_KEYWORD,
            TdmInputErrorKind::UnexpectedUnit => Self::UNEXPECTED_UNIT,
            TdmInputErrorKind::NonInteger => Self::NON_INTEGER,
            TdmInputErrorKind::Negative => Self::NEGATIVE,
            TdmInputErrorKind::NegativeZero => Self::NEGATIVE_ZERO,
            TdmInputErrorKind::UnitMismatch => Self::UNIT_MISMATCH,
            TdmInputErrorKind::DecimalMismatch => Self::DECIMAL_MISMATCH,
            _ => Self::UNKNOWN,
        }
    }
}

#[pymethods]
impl PyTdmInputErrorKind {
    #[getter]
    fn kind(&self) -> &'static str {
        match self {
            Self::MISSING => "missing",
            Self::FLOAT_PARSE => "float_parse",
            Self::NON_FINITE => "non_finite",
            Self::NOT_POSITIVE => "not_positive",
            Self::OUT_OF_RANGE => "out_of_range",
            Self::INVALID_INDEX => "invalid_index",
            Self::UNKNOWN_KEYWORD => "unknown_keyword",
            Self::UNEXPECTED_UNIT => "unexpected_unit",
            Self::NON_INTEGER => "non_integer",
            Self::NEGATIVE => "negative",
            Self::NEGATIVE_ZERO => "negative_zero",
            Self::UNIT_MISMATCH => "unit_mismatch",
            Self::DECIMAL_MISMATCH => "decimal_mismatch",
            Self::UNKNOWN => "unknown",
        }
    }

    fn __repr__(&self) -> String {
        format!("TdmInputErrorKind.{}", self.kind().to_uppercase())
    }
}

/// Structured error payload attached to TdmParseError exceptions.
#[pyclass(module = "sidereon._sidereon", name = "TdmErrorDetail")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmErrorDetail {
    pub(crate) inner: TdmError,
}

impl From<TdmError> for PyTdmErrorDetail {
    fn from(inner: TdmError) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmErrorDetail {
    /// Categorical error kind string.
    #[getter]
    fn kind(&self) -> String {
        let label = match &self.inner {
            TdmError::NoSegments => "no_segments",
            TdmError::Section { .. } => "section",
            TdmError::MalformedLine { .. } => "malformed_line",
            TdmError::NonPrintableCharacter { .. } => "non_printable_character",
            TdmError::LineTooLong { .. } => "line_too_long",
            TdmError::MalformedEpoch { .. } => "malformed_epoch",
            TdmError::RecordsOutOfOrder { .. } => "records_out_of_order",
            TdmError::DuplicateRecord { .. } => "duplicate_record",
            TdmError::UnterminatedFinalLine { .. } => "unterminated_final_line",
            TdmError::Unwritable { .. } => "unwritable",
            TdmError::KeywordOutOfOrder { .. } => "keyword_out_of_order",
            TdmError::UndefinedParticipant { .. } => "undefined_participant",
            TdmError::ConflictingKeyword { .. } => "conflicting_keyword",
            TdmError::RepeatedKeyword { .. } => "repeated_keyword",
            TdmError::UndefinedKeyword { .. } => "undefined_keyword",
            TdmError::MissingKeyword { .. } => "missing_keyword",
            TdmError::EmptyDataSection { .. } => "empty_data_section",
            TdmError::EmptyValue { .. } => "empty_value",
            TdmError::InvalidVersion { .. } => "invalid_version",
            TdmError::KeywordNotAssignable { .. } => "keyword_not_assignable",
            TdmError::MalformedRecord { .. } => "malformed_record",
            TdmError::InvalidField { .. } => "invalid_field",
            other => return crate::marshal::debug_variant_snake(other),
        };
        label.to_string()
    }

    /// One-based source line number, if applicable.
    #[getter]
    fn line(&self) -> Option<usize> {
        match &self.inner {
            TdmError::Section { line, .. }
            | TdmError::MalformedLine { line, .. }
            | TdmError::UnterminatedFinalLine { line, .. }
            | TdmError::UndefinedKeyword { line, .. }
            | TdmError::MalformedRecord { line, .. } => Some(*line),
            TdmError::NonPrintableCharacter { line, .. }
            | TdmError::LineTooLong { line, .. }
            | TdmError::MalformedEpoch { line, .. }
            | TdmError::KeywordOutOfOrder { line, .. }
            | TdmError::ConflictingKeyword { line, .. }
            | TdmError::RepeatedKeyword { line, .. }
            | TdmError::EmptyValue { line, .. }
            | TdmError::InvalidVersion { line, .. } => *line,
            _ => None,
        }
    }

    /// Associated KVN keyword, if applicable.
    #[getter]
    fn keyword(&self) -> Option<&str> {
        match &self.inner {
            TdmError::NonPrintableCharacter { keyword, .. }
            | TdmError::LineTooLong { keyword, .. }
            | TdmError::MalformedEpoch { keyword, .. }
            | TdmError::RecordsOutOfOrder { keyword, .. }
            | TdmError::DuplicateRecord { keyword, .. }
            | TdmError::Unwritable { keyword, .. }
            | TdmError::KeywordOutOfOrder { keyword, .. }
            | TdmError::UndefinedParticipant { keyword, .. }
            | TdmError::ConflictingKeyword { keyword, .. }
            | TdmError::RepeatedKeyword { keyword, .. }
            | TdmError::UndefinedKeyword { keyword, .. }
            | TdmError::MissingKeyword { keyword, .. }
            | TdmError::EmptyValue { keyword, .. }
            | TdmError::KeywordNotAssignable { keyword, .. }
            | TdmError::MalformedRecord { keyword, .. }
            | TdmError::InvalidField { keyword, .. } => Some(keyword.as_str()),
            _ => None,
        }
    }

    /// One-based character column for non-printable character errors.
    #[getter]
    fn column(&self) -> Option<usize> {
        match &self.inner {
            TdmError::NonPrintableCharacter { column, .. } => Some(*column),
            _ => None,
        }
    }

    /// Offending character string for non-printable character errors.
    #[getter]
    fn character(&self) -> Option<String> {
        match &self.inner {
            TdmError::NonPrintableCharacter { character, .. } => Some(character.to_string()),
            _ => None,
        }
    }

    /// Line character length for line_too_long errors.
    #[getter]
    fn length(&self) -> Option<usize> {
        match &self.inner {
            TdmError::LineTooLong { length, .. } => Some(*length),
            _ => None,
        }
    }

    /// Raw text associated with malformed line or malformed epoch.
    #[getter]
    fn text(&self) -> Option<&str> {
        match &self.inner {
            TdmError::MalformedLine { text, .. } | TdmError::MalformedEpoch { text, .. } => {
                Some(text.as_str())
            }
            _ => None,
        }
    }

    /// Section detail text for section marker errors.
    #[getter]
    fn detail(&self) -> Option<&str> {
        match &self.inner {
            TdmError::Section { detail, .. } => Some(detail),
            _ => None,
        }
    }

    /// Reason why a field or comment cannot be written.
    #[getter]
    fn reason(&self) -> Option<&str> {
        match &self.inner {
            TdmError::Unwritable { reason, .. } => Some(reason),
            _ => None,
        }
    }

    /// Section name ("header", "metadata", or "data") where the error occurred.
    #[getter]
    fn section(&self) -> Option<&str> {
        match &self.inner {
            TdmError::KeywordOutOfOrder { section, .. }
            | TdmError::ConflictingKeyword { section, .. }
            | TdmError::RepeatedKeyword { section, .. }
            | TdmError::UndefinedKeyword { section, .. } => Some(section),
            _ => None,
        }
    }

    /// One-based segment index, if applicable.
    #[getter]
    fn segment(&self) -> Option<usize> {
        match &self.inner {
            TdmError::RecordsOutOfOrder { segment, .. }
            | TdmError::DuplicateRecord { segment, .. }
            | TdmError::UndefinedParticipant { segment, .. }
            | TdmError::EmptyDataSection { segment, .. } => Some(*segment),
            TdmError::MissingKeyword { segment, .. } => *segment,
            _ => None,
        }
    }

    /// Associated record timetag epoch, if applicable.
    #[getter]
    fn epoch(&self) -> Option<&str> {
        match &self.inner {
            TdmError::RecordsOutOfOrder { epoch, .. } | TdmError::DuplicateRecord { epoch, .. } => {
                Some(epoch.as_str())
            }
            _ => None,
        }
    }

    /// Participant index that could not be resolved.
    #[getter]
    fn index(&self) -> Option<u8> {
        match &self.inner {
            TdmError::UndefinedParticipant { index, .. } => Some(*index),
            _ => None,
        }
    }

    /// First observed value for conflicting keyword errors.
    #[getter]
    fn first(&self) -> Option<&str> {
        match &self.inner {
            TdmError::ConflictingKeyword { first, .. } => Some(first.as_str()),
            _ => None,
        }
    }

    /// Second (disagreeing) value for conflicting keyword errors.
    #[getter]
    fn second(&self) -> Option<&str> {
        match &self.inner {
            TdmError::ConflictingKeyword { second, .. } => Some(second.as_str()),
            _ => None,
        }
    }

    /// Offending value string for invalid version errors.
    #[getter]
    fn value(&self) -> Option<&str> {
        match &self.inner {
            TdmError::InvalidVersion { value, .. } => Some(value.as_str()),
            _ => None,
        }
    }

    /// Validation failure category for invalid field errors.
    #[getter]
    fn input_error_kind(&self) -> Option<PyTdmInputErrorKind> {
        match &self.inner {
            TdmError::InvalidField { kind, .. } => Some((*kind).into()),
            _ => None,
        }
    }

    /// Name of the validation failure category for invalid field errors, in
    /// snake case, including a category newer than `TdmInputErrorKind`, which
    /// reads as `UNKNOWN` there.
    #[getter]
    fn input_error_kind_name(&self) -> Option<String> {
        match &self.inner {
            TdmError::InvalidField { kind, .. } => Some(crate::marshal::debug_variant_snake(kind)),
            _ => None,
        }
    }

    /// Full human-readable diagnostic message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __str__(&self) -> String {
        self.message()
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmErrorDetail(kind={:?}, message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Result of policy-aware TDM parsing.
#[pyclass(module = "sidereon._sidereon", name = "TdmParseResult")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdmParseResult {
    pub(crate) message: PyTdm,
    pub(crate) warnings: Vec<PyTdmWarning>,
}

#[pymethods]
impl PyTdmParseResult {
    /// Parsed TDM message.
    #[getter]
    fn message(&self) -> PyTdm {
        self.message.clone()
    }

    /// Alias for message.
    #[getter]
    fn tdm(&self) -> PyTdm {
        self.message.clone()
    }

    /// Value alias.
    #[getter]
    fn value(&self) -> PyTdm {
        self.message.clone()
    }

    /// Ordered warnings emitted for forgiven departures.
    #[getter]
    fn warnings(&self) -> Vec<PyTdmWarning> {
        self.warnings.clone()
    }

    fn __len__(&self) -> usize {
        2
    }

    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<PyObject> {
        match idx {
            0 => Ok(self.message.clone().into_pyobject(py)?.into_any().unbind()),
            1 => Ok(self.warnings.clone().into_pyobject(py)?.into_any().unbind()),
            _ => Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmParseResult(segments={}, warnings={})",
            self.message.inner.segments.len(),
            self.warnings.len()
        )
    }
}

/// Result of policy-aware TDM serialization.
#[pyclass(module = "sidereon._sidereon", name = "TdmWriteResult")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyTdmWriteResult {
    pub(crate) text: String,
    pub(crate) departures: Vec<PyTdmDeparture>,
}

#[pymethods]
impl PyTdmWriteResult {
    /// Emitted KVN text.
    #[getter]
    fn text(&self) -> &str {
        &self.text
    }

    /// Alias for text.
    #[getter]
    fn value(&self) -> &str {
        &self.text
    }

    /// Ordered departures emitted under policy.
    #[getter]
    fn departures(&self) -> Vec<PyTdmDeparture> {
        self.departures.clone()
    }

    fn __len__(&self) -> usize {
        2
    }

    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<PyObject> {
        match idx {
            0 => Ok(self.text.clone().into_pyobject(py)?.into_any().unbind()),
            1 => Ok(self
                .departures
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmWriteResult(bytes={}, departures={})",
            self.text.len(),
            self.departures.len()
        )
    }
}

/// Result of policy-aware TDM metadata construction or replacement.
#[pyclass(module = "sidereon._sidereon", name = "TdmMetadataResult")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdmMetadataResult {
    pub(crate) metadata: PyTdmMetadata,
    pub(crate) departures: Vec<PyTdmDeparture>,
}

#[pymethods]
impl PyTdmMetadataResult {
    /// Derived TDM metadata block.
    #[getter]
    fn metadata(&self) -> PyTdmMetadata {
        self.metadata.clone()
    }

    /// Value alias.
    #[getter]
    fn value(&self) -> PyTdmMetadata {
        self.metadata.clone()
    }

    /// Ordered departures emitted or forgiven under policy.
    #[getter]
    fn departures(&self) -> Vec<PyTdmDeparture> {
        self.departures.clone()
    }

    fn __len__(&self) -> usize {
        2
    }

    fn __getitem__(&self, py: Python<'_>, idx: isize) -> PyResult<PyObject> {
        match idx {
            0 => Ok(self.metadata.clone().into_pyobject(py)?.into_any().unbind()),
            1 => Ok(self
                .departures
                .clone()
                .into_pyobject(py)?
                .into_any()
                .unbind()),
            _ => Err(pyo3::exceptions::PyIndexError::new_err(
                "index out of range",
            )),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmMetadataResult(fields={}, departures={})",
            self.metadata.inner.fields.len(),
            self.departures.len()
        )
    }
}

/// Metadata extracted from a TDM metadata block.
///
/// Raw ordered fields and positioned comments are the sole authority.
/// All convenience properties (`participants`, `mode`, `paths`, `timetag_ref`,
/// `time_system`, `range_units`) are derived synchronously by the core.
#[pyclass(module = "sidereon._sidereon", name = "TdmMetadata")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdmMetadata {
    pub(crate) inner: TdmMetadata,
}

impl From<TdmMetadata> for PyTdmMetadata {
    fn from(inner: TdmMetadata) -> Self {
        Self { inner }
    }
}

impl PyTdmMetadata {
    fn edit_atomic<F>(
        &mut self,
        py: Python<'_>,
        policy: TdmWritePolicy,
        edit_fn: F,
    ) -> PyResult<Vec<PyTdmDeparture>>
    where
        F: FnOnce(&mut Vec<TdmField>, &mut Vec<TdmComment>),
    {
        let mut candidate_fields = self.inner.fields.clone();
        let mut candidate_comments = self.inner.comments.clone();
        edit_fn(&mut candidate_fields, &mut candidate_comments);
        let (metadata, departures) =
            TdmMetadata::from_raw_with_policy(candidate_fields, candidate_comments, policy)
                .map_err(|e| to_tdm_validation_err(py, e))?;
        self.inner = metadata;
        Ok(departures.into_iter().map(Into::into).collect())
    }

    fn set_scalar_internal(
        &mut self,
        py: Python<'_>,
        key: &str,
        val: Option<&str>,
        policy: TdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        self.edit_atomic(py, policy, |fields, comments| {
            match val {
                Some(v) => {
                    let mut found = false;
                    for f in fields.iter_mut() {
                        if f.key.eq_ignore_ascii_case(key) {
                            f.value = v.to_string();
                            found = true;
                        }
                    }
                    if !found {
                        // Append deterministically
                        fields.push(TdmField {
                            key: key.to_string(),
                            value: v.to_string(),
                        });
                    }
                }
                None => {
                    let mut indices: Vec<usize> = fields
                        .iter()
                        .enumerate()
                        .filter(|(_, f)| f.key.eq_ignore_ascii_case(key))
                        .map(|(i, _)| i)
                        .collect();
                    indices.sort_unstable();
                    for &idx in indices.iter().rev() {
                        fields.remove(idx);
                        for c in comments.iter_mut() {
                            if c.before_record > idx {
                                c.before_record -= 1;
                            }
                        }
                    }
                }
            }
        })
    }
}

#[pymethods]
impl PyTdmMetadata {
    /// Construct metadata strictly from ordered raw fields and positioned comments.
    #[new]
    #[pyo3(signature = (fields, comments=None))]
    fn new(
        py: Python<'_>,
        fields: &Bound<'_, PyAny>,
        comments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let raw_fields = extract_fields(py, fields)?;
        let raw_comments = extract_comments(py, comments, 0)?;
        let inner = TdmMetadata::from_raw(raw_fields, raw_comments)
            .map_err(|e| to_tdm_validation_err(py, e))?;
        Ok(Self { inner })
    }

    /// Construct metadata from raw ordered fields and positioned comments under strict policy.
    #[staticmethod]
    #[pyo3(signature = (fields, comments=None))]
    fn from_raw(
        py: Python<'_>,
        fields: &Bound<'_, PyAny>,
        comments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Self::new(py, fields, comments)
    }

    /// Construct metadata from raw fields and comments under an explicit write policy.
    #[staticmethod]
    #[pyo3(signature = (fields, comments=None, policy=None))]
    fn from_raw_with_policy(
        py: Python<'_>,
        fields: &Bound<'_, PyAny>,
        comments: Option<&Bound<'_, PyAny>>,
        policy: Option<&PyTdmWritePolicy>,
    ) -> PyResult<PyTdmMetadataResult> {
        let raw_fields = extract_fields(py, fields)?;
        let raw_comments = extract_comments(py, comments, 0)?;
        let write_policy = policy
            .map(|p| p.inner)
            .unwrap_or_else(TdmWritePolicy::strict);
        let (metadata, departures) =
            TdmMetadata::from_raw_with_policy(raw_fields, raw_comments, write_policy)
                .map_err(|e| to_tdm_validation_err(py, e))?;
        Ok(PyTdmMetadataResult {
            metadata: PyTdmMetadata { inner: metadata },
            departures: departures.into_iter().map(Into::into).collect(),
        })
    }

    /// Atomically replace raw fields and comments under strict policy.
    #[pyo3(signature = (fields, comments=None))]
    fn replace_raw(
        &mut self,
        py: Python<'_>,
        fields: &Bound<'_, PyAny>,
        comments: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        let raw_fields = extract_fields(py, fields)?;
        let raw_comments = if let Some(c) = comments {
            extract_comments(py, Some(c), 0)?
        } else {
            self.inner.comments.clone()
        };
        self.inner
            .replace_raw(raw_fields, raw_comments)
            .map_err(|e| to_tdm_validation_err(py, e))
    }

    /// Atomically replace raw fields and comments under an explicit write policy.
    #[pyo3(signature = (fields, comments=None, policy=None))]
    fn replace_raw_with_policy(
        &mut self,
        py: Python<'_>,
        fields: &Bound<'_, PyAny>,
        comments: Option<&Bound<'_, PyAny>>,
        policy: Option<&PyTdmWritePolicy>,
    ) -> PyResult<PyTdmMetadataResult> {
        let raw_fields = extract_fields(py, fields)?;
        let raw_comments = if let Some(c) = comments {
            extract_comments(py, Some(c), 0)?
        } else {
            self.inner.comments.clone()
        };
        let write_policy = policy
            .map(|p| p.inner)
            .unwrap_or_else(TdmWritePolicy::strict);
        let departures = self
            .inner
            .replace_raw_with_policy(raw_fields, raw_comments, write_policy)
            .map_err(|e| to_tdm_validation_err(py, e))?;
        Ok(PyTdmMetadataResult {
            metadata: self.clone(),
            departures: departures.into_iter().map(Into::into).collect(),
        })
    }

    /// Positioned metadata comments.
    #[getter]
    fn comments(&self) -> Vec<PyTdmComment> {
        self.inner
            .comments
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Replace all comments under strict policy.
    #[setter]
    fn set_comments(&mut self, py: Python<'_>, comments: &Bound<'_, PyAny>) -> PyResult<()> {
        let raw_comments = extract_comments(py, Some(comments), 0)?;
        self.edit_atomic(py, TdmWritePolicy::strict(), |_fields, c| {
            *c = raw_comments;
        })
        .map(|_| ())
    }

    /// Replace all comments under policy.
    fn set_comments_with_policy(
        &mut self,
        py: Python<'_>,
        comments: &Bound<'_, PyAny>,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        let raw_comments = extract_comments(py, Some(comments), 0)?;
        self.edit_atomic(py, policy.inner, |_fields, c| {
            *c = raw_comments;
        })
    }

    /// Raw metadata fields in parse order.
    #[getter]
    fn fields(&self) -> Vec<PyTdmField> {
        self.inner.fields.iter().cloned().map(Into::into).collect()
    }

    /// Replace all raw fields under strict policy.
    #[setter]
    fn set_fields(&mut self, py: Python<'_>, fields: &Bound<'_, PyAny>) -> PyResult<()> {
        let raw_fields = extract_fields(py, fields)?;
        self.inner
            .replace_raw(raw_fields, self.inner.comments.clone())
            .map_err(|e| to_tdm_validation_err(py, e))
    }

    /// Replace all raw fields under policy.
    fn set_fields_with_policy(
        &mut self,
        py: Python<'_>,
        fields: &Bound<'_, PyAny>,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        let raw_fields = extract_fields(py, fields)?;
        self.inner
            .replace_raw_with_policy(raw_fields, self.inner.comments.clone(), policy.inner)
            .map_err(|e| to_tdm_validation_err(py, e))
            .map(|deps| deps.into_iter().map(Into::into).collect())
    }

    /// Parsed `PARTICIPANT_n` entries derived from raw fields.
    #[getter]
    fn participants(&self) -> Vec<PyTdmParticipant> {
        self.inner
            .participants
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Optional `MODE` value derived from raw fields.
    #[getter]
    fn mode(&self) -> Option<String> {
        self.inner.mode.clone()
    }

    /// Set `MODE` value under strict policy.
    #[setter]
    fn set_mode(&mut self, py: Python<'_>, mode: Option<String>) -> PyResult<()> {
        self.set_scalar_internal(py, "MODE", mode.as_deref(), TdmWritePolicy::strict())
            .map(|_| ())
    }

    /// Set `MODE` value under explicit policy, returning departures.
    fn set_mode_with_policy(
        &mut self,
        py: Python<'_>,
        mode: Option<String>,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        self.set_scalar_internal(py, "MODE", mode.as_deref(), policy.inner)
    }

    /// Parsed `PATH`, `PATH_1`, and `PATH_2` entries derived from raw fields.
    #[getter]
    fn paths(&self) -> Vec<PyTdmPath> {
        self.inner.paths.iter().cloned().map(Into::into).collect()
    }

    /// Optional `TIMETAG_REF` value derived from raw fields.
    #[getter]
    fn timetag_ref(&self) -> Option<String> {
        self.inner.timetag_ref.clone()
    }

    /// Set `TIMETAG_REF` value under strict policy.
    #[setter]
    fn set_timetag_ref(&mut self, py: Python<'_>, timetag_ref: Option<String>) -> PyResult<()> {
        self.set_scalar_internal(
            py,
            "TIMETAG_REF",
            timetag_ref.as_deref(),
            TdmWritePolicy::strict(),
        )
        .map(|_| ())
    }

    /// Set `TIMETAG_REF` value under explicit policy, returning departures.
    fn set_timetag_ref_with_policy(
        &mut self,
        py: Python<'_>,
        timetag_ref: Option<String>,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        self.set_scalar_internal(py, "TIMETAG_REF", timetag_ref.as_deref(), policy.inner)
    }

    /// Optional `TIME_SYSTEM` value derived from raw fields.
    #[getter]
    fn time_system(&self) -> Option<String> {
        self.inner.time_system.clone()
    }

    /// Set `TIME_SYSTEM` value under strict policy.
    #[setter]
    fn set_time_system(&mut self, py: Python<'_>, time_system: Option<String>) -> PyResult<()> {
        self.set_scalar_internal(
            py,
            "TIME_SYSTEM",
            time_system.as_deref(),
            TdmWritePolicy::strict(),
        )
        .map(|_| ())
    }

    /// Set `TIME_SYSTEM` value under explicit policy, returning departures.
    fn set_time_system_with_policy(
        &mut self,
        py: Python<'_>,
        time_system: Option<String>,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        self.set_scalar_internal(py, "TIME_SYSTEM", time_system.as_deref(), policy.inner)
    }

    /// Range unit for `RANGE` records derived from raw fields.
    #[getter]
    fn range_units(&self) -> PyTdmUnit {
        self.inner.range_units.clone().into()
    }

    /// Set `RANGE_UNITS` under strict policy.
    #[setter]
    fn set_range_units(&mut self, py: Python<'_>, range_units: PyTdmUnit) -> PyResult<()> {
        self.set_range_units_with_policy(py, range_units, &PyTdmWritePolicy::strict())
            .map(|_| ())
    }

    /// Set `RANGE_UNITS` under explicit policy, returning departures.
    fn set_range_units_with_policy(
        &mut self,
        py: Python<'_>,
        range_units: PyTdmUnit,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        let label = range_units.inner.as_str();
        self.set_scalar_internal(py, "RANGE_UNITS", Some(label), policy.inner)
    }

    /// Return the last raw metadata value for `key`.
    fn get_last(&self, key: &str) -> Option<String> {
        self.inner.get_last(key).map(ToString::to_string)
    }

    /// Update matching fields in place or append `key = value` under strict policy.
    fn set_field(&mut self, py: Python<'_>, key: &str, value: &str) -> PyResult<()> {
        self.set_scalar_internal(py, key, Some(value), TdmWritePolicy::strict())
            .map(|_| ())
    }

    /// Update matching fields in place or append `key = value` under policy, returning departures.
    fn set_field_with_policy(
        &mut self,
        py: Python<'_>,
        key: &str,
        value: &str,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        self.set_scalar_internal(py, key, Some(value), policy.inner)
    }

    /// Insert field at index, shifting succeeding comment offsets.
    fn insert_field(&mut self, py: Python<'_>, index: usize, field: &PyTdmField) -> PyResult<()> {
        self.insert_field_with_policy(py, index, field, &PyTdmWritePolicy::strict())
            .map(|_| ())
    }

    /// Insert field at index under policy, shifting succeeding comment offsets.
    fn insert_field_with_policy(
        &mut self,
        py: Python<'_>,
        index: usize,
        field: &PyTdmField,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        if index > self.inner.fields.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "insert index out of range",
            ));
        }
        let f = field.inner.clone();
        self.edit_atomic(py, policy.inner, |fields, comments| {
            fields.insert(index, f);
            for c in comments.iter_mut() {
                if c.before_record > index {
                    c.before_record += 1;
                }
            }
        })
    }

    /// Append field to raw fields under strict policy.
    fn append_field(&mut self, py: Python<'_>, field: &PyTdmField) -> PyResult<()> {
        let idx = self.inner.fields.len();
        self.insert_field(py, idx, field)
    }

    /// Append field to raw fields under policy, returning departures.
    fn append_field_with_policy(
        &mut self,
        py: Python<'_>,
        field: &PyTdmField,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        let idx = self.inner.fields.len();
        self.insert_field_with_policy(py, idx, field, policy)
    }

    /// Remove all matching occurrences of `key`, shifting subsequent comment offsets.
    fn remove_field(&mut self, py: Python<'_>, key: &str) -> PyResult<usize> {
        let (count, _) = self.remove_field_with_policy(py, key, &PyTdmWritePolicy::strict())?;
        Ok(count)
    }

    /// Remove all matching occurrences of `key` under policy, returning (count, departures).
    fn remove_field_with_policy(
        &mut self,
        py: Python<'_>,
        key: &str,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<(usize, Vec<PyTdmDeparture>)> {
        let count = self
            .inner
            .fields
            .iter()
            .filter(|f| f.key.eq_ignore_ascii_case(key))
            .count();
        let departures = self.set_scalar_internal(py, key, None, policy.inner)?;
        Ok((count, departures))
    }

    /// Remove the field at `index`, adjusting succeeding comment offsets.
    fn remove_field_at(&mut self, py: Python<'_>, index: usize) -> PyResult<PyTdmField> {
        let (field, _) =
            self.remove_field_at_with_policy(py, index, &PyTdmWritePolicy::strict())?;
        Ok(field)
    }

    /// Remove the field at `index` under policy, returning (removed_field, departures).
    fn remove_field_at_with_policy(
        &mut self,
        py: Python<'_>,
        index: usize,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<(PyTdmField, Vec<PyTdmDeparture>)> {
        if index >= self.inner.fields.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "remove index out of range",
            ));
        }
        let removed = self.inner.fields[index].clone();
        let departures = self.edit_atomic(py, policy.inner, |fields, comments| {
            fields.remove(index);
            for c in comments.iter_mut() {
                if c.before_record > index {
                    c.before_record -= 1;
                }
            }
        })?;
        Ok((removed.into(), departures))
    }

    /// Add a positioned comment to this metadata block.
    fn add_comment(&mut self, py: Python<'_>, text: String, before_record: usize) -> PyResult<()> {
        self.add_comment_with_policy(py, text, before_record, &PyTdmWritePolicy::strict())
            .map(|_| ())
    }

    /// Add a positioned comment under policy, returning departures.
    fn add_comment_with_policy(
        &mut self,
        py: Python<'_>,
        text: String,
        before_record: usize,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<Vec<PyTdmDeparture>> {
        self.edit_atomic(py, policy.inner, |_fields, comments| {
            comments.push(TdmComment {
                text,
                before_record,
            });
            comments.sort_by_key(|c| c.before_record);
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmMetadata(participants={}, fields={}, comments={})",
            self.inner.participants.len(),
            self.inner.fields.len(),
            self.inner.comments.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// One TDM segment consisting of metadata and data blocks.
#[pyclass(module = "sidereon._sidereon", name = "TdmSegment")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdmSegment {
    pub(crate) inner: TdmSegment,
}

impl From<TdmSegment> for PyTdmSegment {
    fn from(inner: TdmSegment) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmSegment {
    /// Build a TDM segment.
    #[new]
    fn new(metadata: PyTdmMetadata, data: PyTdmDataSection) -> Self {
        Self {
            inner: TdmSegment {
                metadata: metadata.inner,
                data: data.inner,
            },
        }
    }

    /// Segment metadata block (getter copy).
    #[getter]
    fn metadata(&self) -> PyTdmMetadata {
        self.inner.metadata.clone().into()
    }

    /// Replace the segment metadata block.
    #[setter]
    fn set_metadata(&mut self, metadata: &PyTdmMetadata) {
        self.inner.metadata = metadata.inner.clone();
    }

    /// Segment data block (getter copy).
    #[getter]
    fn data(&self) -> PyTdmDataSection {
        self.inner.data.clone().into()
    }

    /// Replace the segment data block.
    #[setter]
    fn set_data(&mut self, data: &PyTdmDataSection) {
        self.inner.data = data.inner.clone();
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmSegment(records={}, fields={})",
            self.inner.data.records.len(),
            self.inner.metadata.fields.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A parsed CCSDS Tracking Data Message.
#[pyclass(module = "sidereon._sidereon", name = "Tdm")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyTdm {
    pub(crate) inner: Tdm,
}

impl From<Tdm> for PyTdm {
    fn from(inner: Tdm) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdm {
    /// Build a TDM message.
    #[new]
    #[pyo3(signature = (
        version,
        segments,
        *,
        comments=None,
        creation_date=None,
        originator=None,
        message_id=None,
        header_fields=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        version: String,
        segments: Vec<Py<PyTdmSegment>>,
        comments: Option<&Bound<'_, PyAny>>,
        creation_date: Option<String>,
        originator: Option<String>,
        message_id: Option<String>,
        header_fields: Option<Vec<Py<PyTdmField>>>,
    ) -> PyResult<Self> {
        let comments = extract_comments(py, comments, 1)?;
        Ok(Self {
            inner: Tdm {
                version,
                comments,
                creation_date,
                originator,
                message_id,
                header_fields: borrow_vec(py, header_fields, |field: &PyTdmField| {
                    field.inner.clone()
                }),
                segments: segments
                    .iter()
                    .map(|segment| segment.borrow(py).inner.clone())
                    .collect(),
            },
        })
    }

    /// `CCSDS_TDM_VERS` header value.
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    #[setter]
    fn set_version(&mut self, version: String) {
        self.inner.version = version;
    }

    /// Positioned header comments.
    #[getter]
    fn comments(&self) -> Vec<PyTdmComment> {
        self.inner
            .comments
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[setter]
    fn set_comments(&mut self, py: Python<'_>, comments: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.comments = extract_comments(py, Some(comments), 1)?;
        Ok(())
    }

    /// Optional `CREATION_DATE` header value.
    #[getter]
    fn creation_date(&self) -> Option<String> {
        self.inner.creation_date.clone()
    }

    #[setter]
    fn set_creation_date(&mut self, creation_date: Option<String>) {
        self.inner.creation_date = creation_date;
    }

    /// Optional `ORIGINATOR` header value.
    #[getter]
    fn originator(&self) -> Option<String> {
        self.inner.originator.clone()
    }

    #[setter]
    fn set_originator(&mut self, originator: Option<String>) {
        self.inner.originator = originator;
    }

    /// Optional `MESSAGE_ID` header value.
    #[getter]
    fn message_id(&self) -> Option<String> {
        self.inner.message_id.clone()
    }

    #[setter]
    fn set_message_id(&mut self, message_id: Option<String>) {
        self.inner.message_id = message_id;
    }

    /// Header fields not part of the common modeled header.
    #[getter]
    fn header_fields(&self) -> Vec<PyTdmField> {
        self.inner
            .header_fields
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[setter]
    fn set_header_fields(&mut self, py: Python<'_>, header_fields: Vec<Py<PyTdmField>>) {
        self.inner.header_fields = header_fields
            .iter()
            .map(|f| f.borrow(py).inner.clone())
            .collect();
    }

    /// Metadata/data segments in message order.
    #[getter]
    fn segments(&self) -> Vec<PyTdmSegment> {
        self.inner
            .segments
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[setter]
    fn set_segments(&mut self, py: Python<'_>, segments: Vec<Py<PyTdmSegment>>) {
        self.inner.segments = segments
            .iter()
            .map(|s| s.borrow(py).inner.clone())
            .collect();
    }

    /// Replace the segment at `index`.
    fn set_segment(&mut self, index: usize, segment: &PyTdmSegment) -> PyResult<()> {
        if index >= self.inner.segments.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "segment index out of range",
            ));
        }
        self.inner.segments[index] = segment.inner.clone();
        Ok(())
    }

    /// Alias for `set_segment`.
    fn replace_segment(&mut self, index: usize, segment: &PyTdmSegment) -> PyResult<()> {
        self.set_segment(index, segment)
    }

    /// Encode this message to canonical CCSDS TDM KVN text under strict policy.
    fn to_kvn_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_kvn(&self.inner).map_err(|e| to_tdm_write_err(py, e))
    }

    /// Convenience alias for `to_kvn_string`.
    fn to_kvn(&self, py: Python<'_>) -> PyResult<String> {
        self.to_kvn_string(py)
    }

    /// Encode this message to CCSDS TDM KVN text under an explicit write policy.
    fn to_kvn_string_with_policy(
        &self,
        py: Python<'_>,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<PyTdmWriteResult> {
        let (text, departures) = encode_kvn_with_policy(&self.inner, policy.inner)
            .map_err(|e| to_tdm_write_err(py, e))?;
        Ok(PyTdmWriteResult {
            text,
            departures: departures.into_iter().map(Into::into).collect(),
        })
    }

    /// Convenience alias for `to_kvn_string_with_policy`.
    fn to_kvn_with_policy(
        &self,
        py: Python<'_>,
        policy: &PyTdmWritePolicy,
    ) -> PyResult<PyTdmWriteResult> {
        self.to_kvn_string_with_policy(py, policy)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tdm(version={:?}, segments={}, comments={})",
            self.inner.version,
            self.inner.segments.len(),
            self.inner.comments.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Parse CCSDS TDM KVN text under strict default policy.
#[pyfunction]
fn parse_tdm_kvn(py: Python<'_>, text: &str) -> PyResult<PyTdm> {
    parse_kvn(text)
        .map(Into::into)
        .map_err(|e| to_tdm_parse_err(py, e))
}

/// Parse CCSDS TDM KVN text under an explicit read policy.
#[pyfunction]
#[pyo3(signature = (text, policy=None))]
fn parse_tdm_kvn_with_policy(
    py: Python<'_>,
    text: &str,
    policy: Option<&PyTdmPolicy>,
) -> PyResult<PyTdmParseResult> {
    let read_policy = policy.map(|p| p.inner).unwrap_or_else(TdmPolicy::strict);
    let (tdm, warnings) =
        parse_kvn_with_policy(text, read_policy).map_err(|e| to_tdm_parse_err(py, e))?;
    Ok(PyTdmParseResult {
        message: PyTdm { inner: tdm },
        warnings: warnings.into_iter().map(Into::into).collect(),
    })
}

/// Encode a CCSDS TDM message to canonical KVN text under strict policy.
#[pyfunction]
fn encode_tdm_kvn(py: Python<'_>, tdm: &PyTdm) -> PyResult<String> {
    tdm.to_kvn_string(py)
}

/// Encode a CCSDS TDM message to KVN text under an explicit write policy.
#[pyfunction]
#[pyo3(signature = (tdm, policy=None))]
fn encode_tdm_kvn_with_policy(
    py: Python<'_>,
    tdm: &PyTdm,
    policy: Option<&PyTdmWritePolicy>,
) -> PyResult<PyTdmWriteResult> {
    let write_policy = policy.cloned().unwrap_or_else(PyTdmWritePolicy::strict);
    tdm.to_kvn_string_with_policy(py, &write_policy)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTdmField>()?;
    m.add_class::<PyTdmParticipant>()?;
    m.add_class::<PyTdmPath>()?;
    m.add_class::<PyTdmUnit>()?;
    m.add_class::<PyTdmObservable>()?;
    m.add_class::<PyTdmScalar>()?;
    m.add_class::<PyTdmDataRecord>()?;
    m.add_class::<PyTdmDataSection>()?;
    m.add_class::<PyTdmMetadata>()?;
    m.add_class::<PyTdmSegment>()?;
    m.add_class::<PyTdm>()?;
    m.add_class::<PyTdmComment>()?;
    m.add_class::<PyTdmLeniency>()?;
    m.add_class::<PyTdmPolicy>()?;
    m.add_class::<PyTdmWritePolicy>()?;
    m.add_class::<PyTdmWarning>()?;
    m.add_class::<PyTdmDeparture>()?;
    m.add_class::<PyTdmInputErrorKind>()?;
    m.add_class::<PyTdmErrorDetail>()?;
    m.add_class::<PyTdmParseResult>()?;
    m.add_class::<PyTdmWriteResult>()?;
    m.add_class::<PyTdmMetadataResult>()?;
    m.add_function(wrap_pyfunction!(parse_tdm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_tdm_kvn_with_policy, m)?)?;
    m.add_function(wrap_pyfunction!(encode_tdm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(encode_tdm_kvn_with_policy, m)?)?;
    Ok(())
}
