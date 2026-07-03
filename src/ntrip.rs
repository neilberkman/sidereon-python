//! NTRIP sans-I/O binding plus RTCM stream assembler exposure.
//!
//! Network transport lives in `python/sidereon/ntrip.py`. This module only
//! wraps the core request builder, response machine, sourcetable IR, GGA helper,
//! and byte-stream assembler.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use sidereon_core::ntrip::{
    classify_http_response as core_classify_http_response, format_gga as core_format_gga,
    parse_sourcetable as core_parse_sourcetable, CasRecord, ChunkedDecoder, Field, GgaPosition,
    HttpClassification, NetRecord, NtripClientMachine, NtripConfig, NtripCredentials, NtripEvent,
    NtripHandshake, NtripRejection, NtripState, NtripVersion, OtherRecord, Sourcetable,
    SourcetableRecord, StrAuth, StrRecord,
};
use sidereon_core::rtcm::SsrStreamAssembler;

use crate::rtcm::PyRtcmMessage;
use crate::RtcmParseError;

fn to_ntrip_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_version(version: &str) -> PyResult<NtripVersion> {
    match version {
        "rev1" => Ok(NtripVersion::Rev1),
        "rev2" => Ok(NtripVersion::Rev2),
        other => Err(PyValueError::new_err(format!(
            "unknown NTRIP version {other:?}; expected \"rev1\" or \"rev2\""
        ))),
    }
}

fn version_label(version: NtripVersion) -> &'static str {
    match version {
        NtripVersion::Rev1 => "rev1",
        NtripVersion::Rev2 => "rev2",
    }
}

fn state_label(state: NtripState) -> &'static str {
    match state {
        NtripState::Idle => "idle",
        NtripState::AwaitingStatus => "awaiting_status",
        NtripState::AwaitingHeaders => "awaiting_headers",
        NtripState::Streaming => "streaming",
        NtripState::Sourcetable => "sourcetable",
        NtripState::Closed => "closed",
    }
}

fn auth_label(auth: &StrAuth) -> String {
    match auth {
        StrAuth::None => "none".to_string(),
        StrAuth::Basic => "basic".to_string(),
        StrAuth::Digest => "digest".to_string(),
        StrAuth::Other(value) => value.clone(),
    }
}

fn field_raw<T>(field: &Field<T>) -> Option<String> {
    match field {
        Field::Raw(value) => Some(value.clone()),
        Field::Parsed(_) | Field::Empty => None,
    }
}

#[pyclass(module = "sidereon._sidereon", name = "GgaPosition")]
#[derive(Clone)]
pub struct PyGgaPosition {
    inner: GgaPosition,
}

impl PyGgaPosition {
    fn inner(&self) -> GgaPosition {
        self.inner.clone()
    }
}

#[pymethods]
impl PyGgaPosition {
    #[new]
    #[pyo3(signature = (
        lat_deg,
        lon_deg,
        height_m,
        fix_quality=1,
        num_satellites=10,
        hdop=1.0,
    ))]
    fn new(
        lat_deg: f64,
        lon_deg: f64,
        height_m: f64,
        fix_quality: u8,
        num_satellites: u8,
        hdop: f64,
    ) -> Self {
        Self {
            inner: GgaPosition {
                lat_deg,
                lon_deg,
                height_m,
                fix_quality,
                num_satellites,
                hdop,
            },
        }
    }

    #[getter]
    fn lat_deg(&self) -> f64 {
        self.inner.lat_deg
    }

    #[getter]
    fn lon_deg(&self) -> f64 {
        self.inner.lon_deg
    }

    #[getter]
    fn height_m(&self) -> f64 {
        self.inner.height_m
    }

    #[getter]
    fn fix_quality(&self) -> u8 {
        self.inner.fix_quality
    }

    #[getter]
    fn num_satellites(&self) -> u8 {
        self.inner.num_satellites
    }

    #[getter]
    fn hdop(&self) -> f64 {
        self.inner.hdop
    }

    fn __repr__(&self) -> String {
        format!(
            "GgaPosition(lat_deg={}, lon_deg={}, height_m={})",
            self.inner.lat_deg, self.inner.lon_deg, self.inner.height_m
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NtripConfig")]
#[derive(Clone)]
pub struct PyNtripConfig {
    inner: NtripConfig,
}

impl PyNtripConfig {
    fn inner(&self) -> NtripConfig {
        self.inner.clone()
    }
}

#[pymethods]
impl PyNtripConfig {
    #[new]
    #[pyo3(signature = (
        host,
        port=2101,
        mountpoint=String::new(),
        *,
        version="rev2",
        username=None,
        password=None,
        user_agent_product=None,
        gga_interval_s=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        host: String,
        port: u16,
        mountpoint: String,
        version: &str,
        username: Option<String>,
        password: Option<String>,
        user_agent_product: Option<String>,
        gga_interval_s: Option<f64>,
    ) -> PyResult<Self> {
        let credentials = match (username, password) {
            (Some(username), Some(password)) => Some(NtripCredentials { username, password }),
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "username and password must be supplied together",
                ))
            }
        };
        let default = NtripConfig::default();
        Ok(Self {
            inner: NtripConfig {
                host,
                port,
                mountpoint,
                version: parse_version(version)?,
                credentials,
                user_agent_product: user_agent_product.unwrap_or(default.user_agent_product),
                gga_interval_s,
            },
        })
    }

    fn request_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.request_bytes().map_err(to_ntrip_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn request_headers(&self) -> PyResult<(String, Vec<(String, String)>)> {
        self.inner.request_headers().map_err(to_ntrip_err)
    }

    #[getter]
    fn host(&self) -> &str {
        &self.inner.host
    }

    #[getter]
    fn port(&self) -> u16 {
        self.inner.port
    }

    #[getter]
    fn mountpoint(&self) -> &str {
        &self.inner.mountpoint
    }

    #[getter]
    fn version(&self) -> &'static str {
        version_label(self.inner.version)
    }

    #[getter]
    fn gga_interval_s(&self) -> Option<f64> {
        self.inner.gga_interval_s
    }

    fn __repr__(&self) -> String {
        format!(
            "NtripConfig(host={:?}, port={}, mountpoint={:?}, version={:?})",
            self.inner.host,
            self.inner.port,
            self.inner.mountpoint,
            self.version()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "ChunkedDecoder")]
#[derive(Clone, Default)]
pub struct PyChunkedDecoder {
    inner: ChunkedDecoder,
}

#[pymethods]
impl PyChunkedDecoder {
    #[new]
    fn new() -> Self {
        Self {
            inner: ChunkedDecoder::new(),
        }
    }

    fn push<'py>(&mut self, py: Python<'py>, bytes: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
        let decoded = self.inner.push(bytes).map_err(to_ntrip_err)?;
        Ok(PyBytes::new(py, &decoded))
    }

    fn finished(&self) -> bool {
        self.inner.finished()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!("ChunkedDecoder(finished={})", self.inner.finished())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Sourcetable")]
#[derive(Clone)]
pub struct PySourcetable {
    inner: Sourcetable,
}

#[pymethods]
impl PySourcetable {
    #[getter]
    fn records(&self) -> Vec<PySourcetableRecord> {
        self.inner
            .records
            .iter()
            .cloned()
            .map(|inner| PySourcetableRecord { inner })
            .collect()
    }

    fn streams(&self) -> Vec<PyStrRecord> {
        self.inner
            .streams()
            .cloned()
            .map(|inner| PyStrRecord { inner })
            .collect()
    }

    fn to_text(&self) -> PyResult<String> {
        self.inner.to_text().map_err(to_ntrip_err)
    }

    fn __repr__(&self) -> String {
        format!("Sourcetable(records={})", self.inner.records.len())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SourcetableRecord")]
#[derive(Clone)]
pub struct PySourcetableRecord {
    inner: SourcetableRecord,
}

#[pymethods]
impl PySourcetableRecord {
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            SourcetableRecord::Str(_) => "STR",
            SourcetableRecord::Cas(_) => "CAS",
            SourcetableRecord::Net(_) => "NET",
            SourcetableRecord::Other(_) => "OTHER",
        }
    }

    #[getter]
    fn str_record(&self) -> Option<PyStrRecord> {
        match &self.inner {
            SourcetableRecord::Str(inner) => Some(PyStrRecord {
                inner: inner.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn cas_record(&self) -> Option<PyCasRecord> {
        match &self.inner {
            SourcetableRecord::Cas(inner) => Some(PyCasRecord {
                inner: inner.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn net_record(&self) -> Option<PyNetRecord> {
        match &self.inner {
            SourcetableRecord::Net(inner) => Some(PyNetRecord {
                inner: inner.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn other_record(&self) -> Option<PyOtherRecord> {
        match &self.inner {
            SourcetableRecord::Other(inner) => Some(PyOtherRecord {
                inner: inner.clone(),
            }),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("SourcetableRecord(kind={:?})", self.kind())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "StrRecord")]
#[derive(Clone)]
pub struct PyStrRecord {
    inner: StrRecord,
}

#[pymethods]
impl PyStrRecord {
    #[getter]
    fn mountpoint(&self) -> &str {
        &self.inner.mountpoint
    }

    #[getter]
    fn identifier(&self) -> &str {
        &self.inner.identifier
    }

    #[getter]
    fn format(&self) -> &str {
        &self.inner.format
    }

    #[getter]
    fn format_details(&self) -> &str {
        &self.inner.format_details
    }

    #[getter]
    fn carrier(&self) -> Option<u8> {
        self.inner.carrier.value().copied()
    }

    #[getter]
    fn nav_system(&self) -> &str {
        &self.inner.nav_system
    }

    #[getter]
    fn network(&self) -> &str {
        &self.inner.network
    }

    #[getter]
    fn country(&self) -> &str {
        &self.inner.country
    }

    #[getter]
    fn lat_deg(&self) -> Option<f64> {
        self.inner.lat_deg.value().copied()
    }

    #[getter]
    fn lon_deg(&self) -> Option<f64> {
        self.inner.lon_deg.value().copied()
    }

    #[getter]
    fn nmea_required(&self) -> Option<bool> {
        self.inner.nmea_required.value().copied()
    }

    #[getter]
    fn network_solution(&self) -> Option<bool> {
        self.inner.network_solution.value().copied()
    }

    #[getter]
    fn generator(&self) -> &str {
        &self.inner.generator
    }

    #[getter]
    fn compression(&self) -> &str {
        &self.inner.compression
    }

    #[getter]
    fn authentication(&self) -> String {
        auth_label(&self.inner.authentication)
    }

    #[getter]
    fn fee(&self) -> Option<bool> {
        self.inner.fee.value().copied()
    }

    #[getter]
    fn bitrate(&self) -> Option<u32> {
        self.inner.bitrate.value().copied()
    }

    #[getter]
    fn misc(&self) -> &str {
        &self.inner.misc
    }

    #[getter]
    fn raw_nmea_required(&self) -> Option<String> {
        field_raw(&self.inner.nmea_required)
    }

    fn __repr__(&self) -> String {
        format!("StrRecord(mountpoint={:?})", self.inner.mountpoint)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "CasRecord")]
#[derive(Clone)]
pub struct PyCasRecord {
    inner: CasRecord,
}

#[pymethods]
impl PyCasRecord {
    #[getter]
    fn host(&self) -> &str {
        &self.inner.host
    }

    #[getter]
    fn port(&self) -> Option<u16> {
        self.inner.port.value().copied()
    }

    #[getter]
    fn identifier(&self) -> &str {
        &self.inner.identifier
    }

    #[getter]
    fn operator(&self) -> &str {
        &self.inner.operator
    }

    #[getter]
    fn nmea_required(&self) -> Option<bool> {
        self.inner.nmea_required.value().copied()
    }

    #[getter]
    fn country(&self) -> &str {
        &self.inner.country
    }

    #[getter]
    fn lat_deg(&self) -> Option<f64> {
        self.inner.lat_deg.value().copied()
    }

    #[getter]
    fn lon_deg(&self) -> Option<f64> {
        self.inner.lon_deg.value().copied()
    }

    #[getter]
    fn fallback_host(&self) -> &str {
        &self.inner.fallback_host
    }

    #[getter]
    fn fallback_port(&self) -> Option<u16> {
        self.inner.fallback_port.value().copied()
    }

    #[getter]
    fn misc(&self) -> &str {
        &self.inner.misc
    }

    fn __repr__(&self) -> String {
        format!(
            "CasRecord(host={:?}, port={:?})",
            self.inner.host,
            self.port()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NetRecord")]
#[derive(Clone)]
pub struct PyNetRecord {
    inner: NetRecord,
}

#[pymethods]
impl PyNetRecord {
    #[getter]
    fn identifier(&self) -> &str {
        &self.inner.identifier
    }

    #[getter]
    fn operator(&self) -> &str {
        &self.inner.operator
    }

    #[getter]
    fn authentication(&self) -> String {
        auth_label(&self.inner.authentication)
    }

    #[getter]
    fn fee(&self) -> Option<bool> {
        self.inner.fee.value().copied()
    }

    #[getter]
    fn web_net(&self) -> &str {
        &self.inner.web_net
    }

    #[getter]
    fn web_str(&self) -> &str {
        &self.inner.web_str
    }

    #[getter]
    fn web_reg(&self) -> &str {
        &self.inner.web_reg
    }

    #[getter]
    fn misc(&self) -> &str {
        &self.inner.misc
    }

    fn __repr__(&self) -> String {
        format!("NetRecord(identifier={:?})", self.inner.identifier)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "OtherRecord")]
#[derive(Clone)]
pub struct PyOtherRecord {
    inner: OtherRecord,
}

#[pymethods]
impl PyOtherRecord {
    #[getter]
    fn type_tag(&self) -> &str {
        &self.inner.type_tag
    }

    #[getter]
    fn fields(&self) -> Vec<String> {
        self.inner.fields.clone()
    }

    fn __repr__(&self) -> String {
        format!("OtherRecord(type_tag={:?})", self.inner.type_tag)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NtripHandshake")]
#[derive(Clone)]
pub struct PyNtripHandshake {
    inner: NtripHandshake,
}

#[pymethods]
impl PyNtripHandshake {
    #[getter]
    fn version(&self) -> &'static str {
        version_label(self.inner.version)
    }

    #[getter]
    fn chunked(&self) -> bool {
        self.inner.chunked
    }

    #[getter]
    fn headers(&self) -> Vec<(String, String)> {
        self.inner.headers.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "NtripHandshake(version={:?}, chunked={})",
            self.version(),
            self.inner.chunked
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NtripRejection")]
#[derive(Clone)]
pub struct PyNtripRejection {
    inner: NtripRejection,
}

#[pymethods]
impl PyNtripRejection {
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            NtripRejection::Unauthorized => "unauthorized",
            NtripRejection::MountpointNotFound => "mountpoint_not_found",
            NtripRejection::DigestRequired => "digest_required",
            NtripRejection::CasterError { .. } => "caster_error",
            NtripRejection::UnexpectedContentType { .. } => "unexpected_content_type",
            NtripRejection::HttpError { .. } => "http_error",
            NtripRejection::MalformedHandshake { .. } => "malformed_handshake",
        }
    }

    #[getter]
    fn reason(&self) -> Option<String> {
        match &self.inner {
            NtripRejection::CasterError { reason } | NtripRejection::HttpError { reason, .. } => {
                Some(reason.clone())
            }
            _ => None,
        }
    }

    #[getter]
    fn status(&self) -> Option<u16> {
        match self.inner {
            NtripRejection::HttpError { status, .. } => Some(status),
            _ => None,
        }
    }

    #[getter]
    fn content_type(&self) -> Option<String> {
        match &self.inner {
            NtripRejection::UnexpectedContentType { content_type } => Some(content_type.clone()),
            _ => None,
        }
    }

    #[getter]
    fn prefix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        match &self.inner {
            NtripRejection::MalformedHandshake { prefix } => Some(PyBytes::new(py, prefix)),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("NtripRejection(kind={:?})", self.kind())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NtripEvent")]
#[derive(Clone)]
pub struct PyNtripEvent {
    kind: &'static str,
    handshake: Option<PyNtripHandshake>,
    payload: Option<Vec<u8>>,
    sourcetable: Option<PySourcetable>,
    rejection: Option<PyNtripRejection>,
    detail: Option<String>,
}

impl From<NtripEvent> for PyNtripEvent {
    fn from(event: NtripEvent) -> Self {
        match event {
            NtripEvent::Connected(inner) => Self {
                kind: "connected",
                handshake: Some(PyNtripHandshake { inner }),
                payload: None,
                sourcetable: None,
                rejection: None,
                detail: None,
            },
            NtripEvent::Payload(bytes) => Self {
                kind: "payload",
                handshake: None,
                payload: Some(bytes),
                sourcetable: None,
                rejection: None,
                detail: None,
            },
            NtripEvent::Sourcetable(inner) => Self {
                kind: "sourcetable",
                handshake: None,
                payload: None,
                sourcetable: Some(PySourcetable { inner }),
                rejection: None,
                detail: None,
            },
            NtripEvent::Rejected(inner) => Self {
                kind: "rejected",
                handshake: None,
                payload: None,
                sourcetable: None,
                rejection: Some(PyNtripRejection { inner }),
                detail: None,
            },
            NtripEvent::StreamCorrupted { detail } => Self {
                kind: "stream_corrupted",
                handshake: None,
                payload: None,
                sourcetable: None,
                rejection: None,
                detail: Some(detail),
            },
            NtripEvent::StreamEnded => Self {
                kind: "stream_ended",
                handshake: None,
                payload: None,
                sourcetable: None,
                rejection: None,
                detail: None,
            },
        }
    }
}

#[pymethods]
impl PyNtripEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    #[getter]
    fn handshake(&self) -> Option<PyNtripHandshake> {
        self.handshake.clone()
    }

    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.payload
            .as_ref()
            .map(|payload| PyBytes::new(py, payload))
    }

    #[getter]
    fn sourcetable(&self) -> Option<PySourcetable> {
        self.sourcetable.clone()
    }

    #[getter]
    fn rejection(&self) -> Option<PyNtripRejection> {
        self.rejection.clone()
    }

    #[getter]
    fn detail(&self) -> Option<String> {
        self.detail.clone()
    }

    fn __repr__(&self) -> String {
        format!("NtripEvent(kind={:?})", self.kind)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "HttpClassification")]
#[derive(Clone)]
pub struct PyHttpClassification {
    kind: &'static str,
    chunked: Option<bool>,
    rejection: Option<PyNtripRejection>,
}

impl From<HttpClassification> for PyHttpClassification {
    fn from(classification: HttpClassification) -> Self {
        match classification {
            HttpClassification::Stream { chunked } => Self {
                kind: "stream",
                chunked: Some(chunked),
                rejection: None,
            },
            HttpClassification::Sourcetable { chunked } => Self {
                kind: "sourcetable",
                chunked: Some(chunked),
                rejection: None,
            },
            HttpClassification::Rejection(inner) => Self {
                kind: "rejection",
                chunked: None,
                rejection: Some(PyNtripRejection { inner }),
            },
        }
    }
}

#[pymethods]
impl PyHttpClassification {
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    #[getter]
    fn chunked(&self) -> Option<bool> {
        self.chunked
    }

    #[getter]
    fn rejection(&self) -> Option<PyNtripRejection> {
        self.rejection.clone()
    }

    fn __repr__(&self) -> String {
        format!("HttpClassification(kind={:?})", self.kind)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NtripClientMachine")]
#[derive(Clone)]
pub struct PyNtripClientMachine {
    inner: NtripClientMachine,
}

#[pymethods]
impl PyNtripClientMachine {
    #[new]
    fn new(config: &PyNtripConfig) -> Self {
        Self {
            inner: NtripClientMachine::new(config.inner()),
        }
    }

    fn connection_request<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.inner.connection_request().map_err(to_ntrip_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    fn push(&mut self, bytes: &[u8]) -> Vec<PyNtripEvent> {
        self.inner.push(bytes).into_iter().map(Into::into).collect()
    }

    fn finish(&mut self) -> Vec<PyNtripEvent> {
        self.inner.finish().into_iter().map(Into::into).collect()
    }

    fn gga_message<'py>(
        &mut self,
        py: Python<'py>,
        now_s: f64,
        position: &PyGgaPosition,
        utc_seconds_of_day: f64,
    ) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .gga_message(now_s, &position.inner(), utc_seconds_of_day)
            .map(|bytes| PyBytes::new(py, &bytes))
    }

    #[getter]
    fn state(&self) -> &'static str {
        state_label(self.inner.state())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> String {
        format!("NtripClientMachine(state={:?})", self.state())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrStreamAssembler")]
#[derive(Clone, Default)]
pub struct PySsrStreamAssembler {
    inner: SsrStreamAssembler,
}

#[pymethods]
impl PySsrStreamAssembler {
    #[new]
    fn new() -> Self {
        Self {
            inner: SsrStreamAssembler::new(),
        }
    }

    fn push(&mut self, chunk: &[u8]) -> PyResult<Vec<PyRtcmMessage>> {
        let mut messages = Vec::new();
        for result in self.inner.push(chunk) {
            messages.push(PyRtcmMessage::from_inner(
                result.map_err(|err| RtcmParseError::new_err(err.to_string()))?,
            ));
        }
        Ok(messages)
    }

    fn push_lossy(&mut self, chunk: &[u8]) -> Vec<PyRtcmMessage> {
        self.inner
            .push(chunk)
            .into_iter()
            .filter_map(Result::ok)
            .map(PyRtcmMessage::from_inner)
            .collect()
    }

    fn retained_len(&self) -> usize {
        self.inner.retained_len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrStreamAssembler(retained_len={})",
            self.inner.retained_len()
        )
    }
}

#[pyfunction]
fn parse_sourcetable(text: &str) -> PyResult<PySourcetable> {
    core_parse_sourcetable(text)
        .map(|inner| PySourcetable { inner })
        .map_err(to_ntrip_err)
}

#[pyfunction]
fn format_gga<'py>(
    py: Python<'py>,
    position: &PyGgaPosition,
    utc_seconds_of_day: f64,
) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = core_format_gga(&position.inner(), utc_seconds_of_day).map_err(to_ntrip_err)?;
    Ok(PyBytes::new(py, &bytes))
}

#[pyfunction]
fn classify_http_response(
    status: u16,
    reason: &str,
    headers: Vec<(String, String)>,
) -> PyHttpClassification {
    core_classify_http_response(status, reason, &headers).into()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGgaPosition>()?;
    m.add_class::<PyNtripConfig>()?;
    m.add_class::<PyChunkedDecoder>()?;
    m.add_class::<PySourcetable>()?;
    m.add_class::<PySourcetableRecord>()?;
    m.add_class::<PyStrRecord>()?;
    m.add_class::<PyCasRecord>()?;
    m.add_class::<PyNetRecord>()?;
    m.add_class::<PyOtherRecord>()?;
    m.add_class::<PyNtripHandshake>()?;
    m.add_class::<PyNtripRejection>()?;
    m.add_class::<PyNtripEvent>()?;
    m.add_class::<PyHttpClassification>()?;
    m.add_class::<PyNtripClientMachine>()?;
    m.add_class::<PySsrStreamAssembler>()?;
    m.add_function(wrap_pyfunction!(parse_sourcetable, m)?)?;
    m.add_function(wrap_pyfunction!(format_gga, m)?)?;
    m.add_function(wrap_pyfunction!(classify_http_response, m)?)?;
    Ok(())
}
