//! CCSDS TDM binding.
//!
//! Provides typed Python value objects for the core TDM KVN message model and
//! parse/encode entry points. Decimal record tokens are preserved by the core
//! and exposed alongside their parsed `f64` values.

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::PyClass;

use sidereon_core::astro::tdm::{
    encode_kvn, parse_kvn, Tdm, TdmDataRecord, TdmDataSection, TdmField, TdmMetadata,
    TdmObservable, TdmParticipant, TdmPath, TdmScalar, TdmSegment, TdmUnit,
};

use crate::TdmParseError;

fn to_tdm_err<E: std::fmt::Display>(err: E) -> PyErr {
    TdmParseError::new_err(err.to_string())
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

/// A TDM KVN key/value field preserved in parse order.
#[pyclass(module = "sidereon._sidereon", name = "TdmField")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyTdmField {
    inner: TdmField,
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

    /// Trimmed KVN value.
    #[getter]
    fn value(&self) -> &str {
        &self.inner.value
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmField(key={:?}, value={:?})",
            self.inner.key, self.inner.value
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

/// One named TDM tracking participant.
#[pyclass(module = "sidereon._sidereon", name = "TdmParticipant")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyTdmParticipant {
    inner: TdmParticipant,
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
        self == other
    }
}

/// A parsed TDM signal path from `PATH`, `PATH_1`, or `PATH_2`.
#[pyclass(module = "sidereon._sidereon", name = "TdmPath")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyTdmPath {
    inner: TdmPath,
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

    /// Participant indices listed in path order.
    #[getter]
    fn participants(&self) -> Vec<u8> {
        self.inner.participants.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmPath(key={:?}, participants={:?})",
            self.inner.key, self.inner.participants
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
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
#[derive(Clone, PartialEq, Eq)]
pub struct PyTdmUnit {
    inner: TdmUnit,
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
        self == other
    }
}

/// Observable family for a TDM tracking data record.
#[pyclass(module = "sidereon._sidereon", name = "TdmObservable")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyTdmObservable {
    inner: TdmObservable,
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
        self == other
    }
}

/// A numeric TDM record value plus the exact decimal token used to encode it.
#[pyclass(module = "sidereon._sidereon", name = "TdmScalar")]
#[derive(Clone, PartialEq)]
pub struct PyTdmScalar {
    inner: TdmScalar,
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
        self == other
    }
}

/// One time-tagged TDM tracking data record.
#[pyclass(module = "sidereon._sidereon", name = "TdmDataRecord")]
#[derive(Clone, PartialEq)]
pub struct PyTdmDataRecord {
    inner: TdmDataRecord,
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
        self == other
    }
}

/// A TDM data block.
#[pyclass(module = "sidereon._sidereon", name = "TdmDataSection")]
#[derive(Clone, PartialEq)]
pub struct PyTdmDataSection {
    inner: TdmDataSection,
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
        comments: Option<Vec<String>>,
    ) -> Self {
        Self {
            inner: TdmDataSection {
                comments: comments.unwrap_or_default(),
                records: records
                    .iter()
                    .map(|record| record.borrow(py).inner.clone())
                    .collect(),
            },
        }
    }

    /// Data-section comments in parse order.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Data records in parse order.
    #[getter]
    fn records(&self) -> Vec<PyTdmDataRecord> {
        self.inner.records.iter().cloned().map(Into::into).collect()
    }

    fn __repr__(&self) -> String {
        format!("TdmDataSection(records={})", self.inner.records.len())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

fn synthesize_metadata_fields(
    fields: &mut Vec<TdmField>,
    participants: &[TdmParticipant],
    mode: &Option<String>,
    paths: &[TdmPath],
    timetag_ref: &Option<String>,
    time_system: &Option<String>,
    range_units: &TdmUnit,
) {
    if let Some(value) = time_system {
        fields.push(TdmField {
            key: "TIME_SYSTEM".to_string(),
            value: value.clone(),
        });
    }
    for participant in participants {
        fields.push(TdmField {
            key: format!("PARTICIPANT_{}", participant.index),
            value: participant.name.clone(),
        });
    }
    if let Some(value) = mode {
        fields.push(TdmField {
            key: "MODE".to_string(),
            value: value.clone(),
        });
    }
    for path in paths {
        fields.push(TdmField {
            key: path.key.clone(),
            value: path
                .participants
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(","),
        });
    }
    if let Some(value) = timetag_ref {
        fields.push(TdmField {
            key: "TIMETAG_REF".to_string(),
            value: value.clone(),
        });
    }
    if !matches!(range_units, TdmUnit::Kilometers) {
        fields.push(TdmField {
            key: "RANGE_UNITS".to_string(),
            value: range_units.as_str().to_string(),
        });
    }
}

/// Metadata extracted from a TDM metadata block.
#[pyclass(module = "sidereon._sidereon", name = "TdmMetadata")]
#[derive(Clone, PartialEq)]
pub struct PyTdmMetadata {
    inner: TdmMetadata,
}

impl From<TdmMetadata> for PyTdmMetadata {
    fn from(inner: TdmMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTdmMetadata {
    /// Build a TDM metadata block.
    #[new]
    #[pyo3(signature = (
        comments=None,
        fields=None,
        participants=None,
        mode=None,
        paths=None,
        timetag_ref=None,
        time_system=None,
        range_units=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        comments: Option<Vec<String>>,
        fields: Option<Vec<Py<PyTdmField>>>,
        participants: Option<Vec<Py<PyTdmParticipant>>>,
        mode: Option<String>,
        paths: Option<Vec<Py<PyTdmPath>>>,
        timetag_ref: Option<String>,
        time_system: Option<String>,
        range_units: Option<PyTdmUnit>,
    ) -> Self {
        let mut fields = borrow_vec(py, fields, |field: &PyTdmField| field.inner.clone());
        let participants = borrow_vec(py, participants, |participant: &PyTdmParticipant| {
            participant.inner.clone()
        });
        let paths = borrow_vec(py, paths, |path: &PyTdmPath| path.inner.clone());
        let range_units = range_units
            .map(|unit| unit.inner)
            .unwrap_or(TdmUnit::Kilometers);
        if fields.is_empty() {
            synthesize_metadata_fields(
                &mut fields,
                &participants,
                &mode,
                &paths,
                &timetag_ref,
                &time_system,
                &range_units,
            );
        }
        Self {
            inner: TdmMetadata {
                comments: comments.unwrap_or_default(),
                fields,
                participants,
                mode,
                paths,
                timetag_ref,
                time_system,
                range_units,
            },
        }
    }

    /// Metadata comments in parse order.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Raw metadata fields in parse order.
    #[getter]
    fn fields(&self) -> Vec<PyTdmField> {
        self.inner.fields.iter().cloned().map(Into::into).collect()
    }

    /// Parsed `PARTICIPANT_n` entries.
    #[getter]
    fn participants(&self) -> Vec<PyTdmParticipant> {
        self.inner
            .participants
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Optional `MODE` metadata value.
    #[getter]
    fn mode(&self) -> Option<String> {
        self.inner.mode.clone()
    }

    /// Parsed `PATH`, `PATH_1`, and `PATH_2` entries.
    #[getter]
    fn paths(&self) -> Vec<PyTdmPath> {
        self.inner.paths.iter().cloned().map(Into::into).collect()
    }

    /// Optional `TIMETAG_REF` metadata value.
    #[getter]
    fn timetag_ref(&self) -> Option<String> {
        self.inner.timetag_ref.clone()
    }

    /// Optional `TIME_SYSTEM` metadata value.
    #[getter]
    fn time_system(&self) -> Option<String> {
        self.inner.time_system.clone()
    }

    /// Range unit for `RANGE` records.
    #[getter]
    fn range_units(&self) -> PyTdmUnit {
        self.inner.range_units.clone().into()
    }

    /// Return the last raw metadata value for `key`.
    fn get_last(&self, key: &str) -> Option<String> {
        self.inner.get_last(key).map(ToString::to_string)
    }

    fn __repr__(&self) -> String {
        format!(
            "TdmMetadata(participants={}, fields={})",
            self.inner.participants.len(),
            self.inner.fields.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

/// One TDM segment consisting of metadata and data blocks.
#[pyclass(module = "sidereon._sidereon", name = "TdmSegment")]
#[derive(Clone, PartialEq)]
pub struct PyTdmSegment {
    inner: TdmSegment,
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

    /// Segment metadata block.
    #[getter]
    fn metadata(&self) -> PyTdmMetadata {
        self.inner.metadata.clone().into()
    }

    /// Segment data block.
    #[getter]
    fn data(&self) -> PyTdmDataSection {
        self.inner.data.clone().into()
    }

    fn __repr__(&self) -> String {
        format!("TdmSegment(records={})", self.inner.data.records.len())
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

/// A parsed CCSDS Tracking Data Message.
#[pyclass(module = "sidereon._sidereon", name = "Tdm")]
#[derive(Clone, PartialEq)]
pub struct PyTdm {
    inner: Tdm,
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
        comments: Option<Vec<String>>,
        creation_date: Option<String>,
        originator: Option<String>,
        message_id: Option<String>,
        header_fields: Option<Vec<Py<PyTdmField>>>,
    ) -> Self {
        Self {
            inner: Tdm {
                version,
                comments: comments.unwrap_or_default(),
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
        }
    }

    /// `CCSDS_TDM_VERS` header value.
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }

    /// Header comments in parse order.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Optional `CREATION_DATE` header value.
    #[getter]
    fn creation_date(&self) -> Option<String> {
        self.inner.creation_date.clone()
    }

    /// Optional `ORIGINATOR` header value.
    #[getter]
    fn originator(&self) -> Option<String> {
        self.inner.originator.clone()
    }

    /// Optional `MESSAGE_ID` header value.
    #[getter]
    fn message_id(&self) -> Option<String> {
        self.inner.message_id.clone()
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

    /// Encode this message to canonical CCSDS TDM KVN text via the core writer.
    fn to_kvn_string(&self) -> PyResult<String> {
        encode_kvn(&self.inner).map_err(to_tdm_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Tdm(version={:?}, segments={})",
            self.inner.version,
            self.inner.segments.len()
        )
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

/// Parse CCSDS TDM KVN text.
#[pyfunction]
fn parse_tdm_kvn(text: &str) -> PyResult<PyTdm> {
    parse_kvn(text).map(Into::into).map_err(to_tdm_err)
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
    m.add_function(wrap_pyfunction!(parse_tdm_kvn, m)?)?;
    Ok(())
}
