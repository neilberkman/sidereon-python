//! SBAS and RTCM SSR correction bindings.
//!
//! Bytes decode into core message structs, stores ingest those structs, and
//! corrected ephemeris wrappers call the core corrected-source implementations.

use std::collections::BTreeMap;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyModule};

use sidereon::ephemeris::EphemerisSource;
use sidereon_core::astro::time::model::GnssWeekTow;
use sidereon_core::frame::Wgs84Geodetic;
use sidereon_core::rtcm::{SsrKind, SsrMessage};
use sidereon_core::sbas::{
    parse_ems_lines as core_parse_sbas_ems_lines, parse_ems_log as core_parse_sbas_ems_log,
    parse_rtklib_lines as core_parse_sbas_rtklib_lines,
    parse_rtklib_log as core_parse_sbas_rtklib_log, sat_to_sbas_prn, sbas_prn_to_sat, SbasBlock,
    SbasCorrectedEphemeris, SbasCorrectionStore, SbasDeparture, SbasDoNotUse, SbasFastCorrection,
    SbasFastCorrections, SbasFastDegradation, SbasGeoAlmanac, SbasGeoNav, SbasGeoState, SbasIgp,
    SbasIgpDelay, SbasIgpMask, SbasIgpUnavailableReason, SbasIntegrity, SbasIonoDelays,
    SbasIonoGrid, SbasLineRefusal, SbasLog, SbasLogBlock, SbasLogOptions, SbasLongTermCorrection,
    SbasLongTermCorrections, SbasLongTermHalf, SbasLongTermRecord, SbasMessage,
    SbasMixedCorrections, SbasMixedFastCorrections, SbasNetworkTime, SbasPolicy, SbasPrnMask,
    SbasSkippedLineKind, SbasSolveMode, SbasUnavailableIgp, SbasUnsupported, SbasWireForm,
    SpareBits,
};
use sidereon_core::ssr::{
    GnssSignal, MissingCorrectionAction, OrbitReferencePoint, RegionalPolicy, SignalCode,
    SsrClockCorrection, SsrCorrectedEphemeris, SsrCorrectionSize, SsrCorrectionSizePolicy,
    SsrCorrectionStore, SsrFallbackPolicy, SsrHighRateClock, SsrNavigationMessage,
    SsrOrbitCorrection, SsrRawSignal, SsrSignalKey, SsrSolution, SsrSource,
};
use sidereon_core::GnssSatelliteId;

use crate::exact_time::PyExactEpochQuery;
use crate::frames::PyTimeScale;
use crate::marshal::{debug_variant_snake, PyGnssSystem};
use crate::rinex::PyBroadcastEphemeris;
use crate::rtcm::PyRtcmMessage;
use crate::{to_rtcm_encode_err, RtcmParseError};

type PyCheckedSsrState = (Option<([f64; 3], f64)>, Option<&'static str>);

/// How the SBAS codec and log readers treat a departure from the format,
/// mirroring the core `SbasPolicy`. A CRC mismatch in a framed block is
/// refused under both.
#[pyclass(module = "sidereon._sidereon", name = "SbasPolicy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PySbasPolicy {
    /// Refuse the first departure.
    STRICT,
    /// Read or write the input and report each departure.
    LENIENT,
}

impl From<PySbasPolicy> for SbasPolicy {
    fn from(policy: PySbasPolicy) -> Self {
        match policy {
            PySbasPolicy::STRICT => SbasPolicy::Strict,
            PySbasPolicy::LENIENT => SbasPolicy::Lenient,
        }
    }
}

/// A departure from the SBAS message or log format read or written under
/// `SbasPolicy.LENIENT`: `unrecognized_preamble` (with `preamble`) or
/// `declared_message_type` (with `declared` and `carried`). `line` is the log
/// line for a departure a log reader recorded.
#[pyclass(module = "sidereon._sidereon", name = "SbasDeparture")]
#[derive(Clone)]
pub struct PySbasDeparture {
    inner: SbasDeparture,
    line: Option<usize>,
}

#[pymethods]
impl PySbasDeparture {
    /// Departure kind label.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            SbasDeparture::UnrecognizedPreamble { .. } => "unrecognized_preamble".to_string(),
            SbasDeparture::DeclaredMessageType { .. } => "declared_message_type".to_string(),
            other => debug_variant_snake(other),
        }
    }

    /// The preamble as read, for `unrecognized_preamble`.
    #[getter]
    fn preamble(&self) -> Option<u8> {
        match &self.inner {
            SbasDeparture::UnrecognizedPreamble { preamble } => Some(*preamble),
            _ => None,
        }
    }

    /// The message type the log record's field states, for
    /// `declared_message_type`.
    #[getter]
    fn declared(&self) -> Option<u8> {
        match &self.inner {
            SbasDeparture::DeclaredMessageType { declared, .. } => Some(*declared),
            _ => None,
        }
    }

    /// The message type the record's message carries, for
    /// `declared_message_type`.
    #[getter]
    fn carried(&self) -> Option<u8> {
        match &self.inner {
            SbasDeparture::DeclaredMessageType { carried, .. } => Some(*carried),
            _ => None,
        }
    }

    /// One-based log line, for a departure a log reader recorded.
    #[getter]
    fn line(&self) -> Option<usize> {
        self.line
    }

    /// The departure as the core states it.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasDeparture(kind={:?}, message={:?})",
            self.kind(),
            self.inner.to_string()
        )
    }
}

fn sbas_departures(list: Vec<SbasDeparture>) -> Vec<PySbasDeparture> {
    list.into_iter()
        .map(|inner| PySbasDeparture { inner, line: None })
        .collect()
}

/// A record line an SBAS log reader left unread: `ambiguous_week` (with
/// `week`) or `checksum_mismatch` (with `written` and `computed`).
#[pyclass(module = "sidereon._sidereon", name = "SbasRefusedLine")]
#[derive(Clone)]
pub struct PySbasRefusedLine {
    line: usize,
    reason: SbasLineRefusal,
}

#[pymethods]
impl PySbasRefusedLine {
    /// One-based line number.
    #[getter]
    fn line(&self) -> usize {
        self.line
    }

    /// Refusal kind label.
    #[getter]
    fn reason(&self) -> String {
        match &self.reason {
            SbasLineRefusal::AmbiguousWeek { .. } => "ambiguous_week".to_string(),
            SbasLineRefusal::ChecksumMismatch { .. } => "checksum_mismatch".to_string(),
            other => debug_variant_snake(other),
        }
    }

    /// The 10-bit week as written, for `ambiguous_week`.
    #[getter]
    fn week(&self) -> Option<u32> {
        match self.reason {
            SbasLineRefusal::AmbiguousWeek { week } => Some(week),
            _ => None,
        }
    }

    /// The checksum as written, when it reads as one, for
    /// `checksum_mismatch`.
    #[getter]
    fn written(&self) -> Option<u32> {
        match self.reason {
            SbasLineRefusal::ChecksumMismatch { written, .. } => written,
            _ => None,
        }
    }

    /// The checksum the line's text gives, for `checksum_mismatch`.
    #[getter]
    fn computed(&self) -> Option<u32> {
        match self.reason {
            SbasLineRefusal::ChecksumMismatch { computed, .. } => Some(computed),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasRefusedLine(line={}, reason={:?})",
            self.line,
            self.reason()
        )
    }
}

/// Everything an EMS or RTKLIB SBAS log reader read: the records, every line
/// read as no record, every record line left unread, and under
/// `SbasPolicy.LENIENT` every departure. Each input line appears exactly
/// once.
#[pyclass(module = "sidereon._sidereon", name = "SbasLog")]
#[derive(Clone)]
pub struct PySbasLog {
    inner: SbasLog,
}

#[pymethods]
impl PySbasLog {
    /// Records in input order.
    #[getter]
    fn blocks(&self) -> Vec<PySbasLogBlock> {
        self.inner.blocks.iter().cloned().map(Into::into).collect()
    }

    /// Lines read as no record, as `(line, kind)` with `kind` one of
    /// `blank`, `comment` or `non_record`.
    #[getter]
    fn skipped_lines(&self) -> Vec<(usize, &'static str)> {
        self.inner
            .skipped_lines
            .iter()
            .map(|skipped| {
                let kind = match skipped.kind {
                    SbasSkippedLineKind::Blank => "blank",
                    SbasSkippedLineKind::Comment => "comment",
                    SbasSkippedLineKind::NonRecord => "non_record",
                };
                (skipped.line, kind)
            })
            .collect()
    }

    /// Record lines left unread, with the reason.
    #[getter]
    fn refused_lines(&self) -> Vec<PySbasRefusedLine> {
        self.inner
            .refused_lines
            .iter()
            .map(|refused| PySbasRefusedLine {
                line: refused.line,
                reason: refused.reason,
            })
            .collect()
    }

    /// Departures read under `SbasPolicy.LENIENT`, in input order.
    #[getter]
    fn departures(&self) -> Vec<PySbasDeparture> {
        self.inner
            .departures
            .iter()
            .map(|departure| PySbasDeparture {
                inner: departure.departure.clone(),
                line: Some(departure.line),
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasLog(blocks={}, skipped_lines={}, refused_lines={}, departures={})",
            self.inner.blocks.len(),
            self.inner.skipped_lines.len(),
            self.inner.refused_lines.len(),
            self.inner.departures.len()
        )
    }
}

fn sbas_log_options(policy: PySbasPolicy, reference_week: Option<u32>) -> SbasLogOptions {
    let options = SbasLogOptions::default().with_policy(policy.into());
    match reference_week {
        Some(week) => options.with_reference_week(week),
        None => options,
    }
}

/// A grid point whose latest delay entry makes it unusable.
#[pyclass(module = "sidereon._sidereon", name = "SbasUnavailableIgp")]
#[derive(Clone)]
pub struct PySbasUnavailableIgp {
    inner: SbasUnavailableIgp,
}

#[pymethods]
impl PySbasUnavailableIgp {
    /// Grid latitude in degrees.
    #[getter]
    fn lat_deg(&self) -> f64 {
        self.inner.lat_deg
    }

    /// Grid longitude in degrees.
    #[getter]
    fn lon_deg(&self) -> f64 {
        self.inner.lon_deg
    }

    /// The nine-bit vertical delay as broadcast.
    #[getter]
    fn vertical_delay(&self) -> u16 {
        self.inner.vertical_delay
    }

    /// The four-bit GIVEI as broadcast.
    #[getter]
    fn givei(&self) -> u8 {
        self.inner.givei
    }

    /// Why the point is unusable: `do_not_use` (delay 511) or
    /// `not_monitored` (GIVEI 15).
    #[getter]
    fn reason(&self) -> &'static str {
        match self.inner.reason {
            SbasIgpUnavailableReason::DoNotUse => "do_not_use",
            SbasIgpUnavailableReason::NotMonitored => "not_monitored",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasUnavailableIgp(lat_deg={}, lon_deg={}, reason={:?})",
            self.inner.lat_deg,
            self.inner.lon_deg,
            self.reason()
        )
    }
}

/// The SSR bias key a Python signal names: a RINEX 3 band-and-attribute
/// string (`"1C"`, or an observation code such as `"C1C"`) names the physical
/// signal of the satellite's system; an int is a raw signal index and needs
/// the `source` whose table it belongs to.
fn ssr_signal_key(
    sat: GnssSatelliteId,
    signal: &Bound<'_, PyAny>,
    source: Option<PySsrSource>,
) -> PyResult<SsrSignalKey> {
    if let Ok(text) = signal.extract::<String>() {
        if source.is_some() {
            return Err(PyValueError::new_err(
                "source applies only to a raw integer signal index",
            ));
        }
        let code = SignalCode::parse(&text).ok_or_else(|| {
            PyValueError::new_err(format!(
                "signal {text:?} is not a RINEX 3 band and attribute or observation code"
            ))
        })?;
        return Ok(GnssSignal::new(sat.system, code).into());
    }
    let index: u8 = signal.extract().map_err(|_| {
        PyValueError::new_err(
            "signal must be a RINEX 3 signal code string or a raw index in 0..=255",
        )
    })?;
    let source = source.ok_or_else(|| {
        PyValueError::new_err(
            "a raw signal index needs source (SsrSource.RTCM_SSR or SsrSource.GALILEO_HAS)",
        )
    })?;
    Ok(SsrRawSignal::new(source.into(), sat.system, index).into())
}

fn nav_message_label(message: SsrNavigationMessage) -> String {
    match message {
        SsrNavigationMessage::Rtcm => "rtcm".to_string(),
        SsrNavigationMessage::Has(index) => format!("has:{index}"),
        SsrNavigationMessage::IgsSsr => "igs_ssr".to_string(),
    }
}

fn to_rtcm_err<E: std::fmt::Display>(err: E) -> PyErr {
    RtcmParseError::new_err(err.to_string())
}

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite_id {token:?}: {err}")))
}

fn gnss_week_tow(scale: PyTimeScale, week: u32, tow_s: f64) -> PyResult<GnssWeekTow> {
    GnssWeekTow::new(scale.into(), week, tow_s)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn ssr_kind_label(kind: SsrKind) -> &'static str {
    match kind {
        SsrKind::Orbit => "orbit",
        SsrKind::Clock => "clock",
        SsrKind::CombinedOrbitClock => "combined_orbit_clock",
        SsrKind::CodeBias => "code_bias",
        SsrKind::PhaseBias => "phase_bias",
        SsrKind::Ura => "ura",
        SsrKind::HighRateClock => "high_rate_clock",
    }
}

fn sbas_message_label(message: &SbasMessage) -> &'static str {
    match message {
        SbasMessage::DoNotUse(_) => "do_not_use",
        SbasMessage::PrnMask(_) => "prn_mask",
        SbasMessage::FastCorrections(_) => "fast_corrections",
        SbasMessage::Integrity(_) => "integrity",
        SbasMessage::FastDegradation(_) => "fast_degradation",
        SbasMessage::GeoNav(_) => "geo_nav",
        SbasMessage::NetworkTime(_) => "network_time",
        SbasMessage::GeoAlmanac(_) => "geo_almanac",
        SbasMessage::IgpMask(_) => "igp_mask",
        SbasMessage::MixedCorrections(_) => "mixed_corrections",
        SbasMessage::LongTermCorrections(_) => "long_term_corrections",
        SbasMessage::IonoDelays(_) => "iono_delays",
        SbasMessage::Unsupported(_) => "unsupported",
    }
}

fn sbas_wire_form_label(form: SbasWireForm) -> &'static str {
    match form {
        SbasWireForm::Framed250 => "framed250",
        SbasWireForm::Body226 => "body226",
    }
}

fn py_satellite(sat: GnssSatelliteId) -> String {
    sat.to_string()
}

fn spare_bits(bits: &SpareBits) -> Vec<(u64, u8)> {
    bits.0.clone()
}

#[pyclass(module = "sidereon._sidereon", name = "SbasWireForm", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Wire encoding for an SBAS message block.
pub enum PySbasWireForm {
    FRAMED250,
    BODY226,
}

impl From<PySbasWireForm> for SbasWireForm {
    fn from(form: PySbasWireForm) -> Self {
        match form {
            PySbasWireForm::FRAMED250 => SbasWireForm::Framed250,
            PySbasWireForm::BODY226 => SbasWireForm::Body226,
        }
    }
}

impl From<SbasWireForm> for PySbasWireForm {
    fn from(form: SbasWireForm) -> Self {
        match form {
            SbasWireForm::Framed250 => Self::FRAMED250,
            SbasWireForm::Body226 => Self::BODY226,
        }
    }
}

#[pymethods]
impl PySbasWireForm {
    #[getter]
    fn label(&self) -> &'static str {
        sbas_wire_form_label((*self).into())
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::FRAMED250 => "SbasWireForm.FRAMED250",
            Self::BODY226 => "SbasWireForm.BODY226",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasSolveMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Mode used when applying SBAS corrections to broadcast ephemeris.
pub enum PySbasSolveMode {
    MIXED_AUGMENTATION,
    SBAS_ONLY,
}

impl From<PySbasSolveMode> for SbasSolveMode {
    fn from(mode: PySbasSolveMode) -> Self {
        match mode {
            PySbasSolveMode::MIXED_AUGMENTATION => SbasSolveMode::MixedAugmentation,
            PySbasSolveMode::SBAS_ONLY => SbasSolveMode::SbasOnly,
        }
    }
}

#[pymethods]
impl PySbasSolveMode {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::MIXED_AUGMENTATION => "mixed_augmentation",
            Self::SBAS_ONLY => "sbas_only",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::MIXED_AUGMENTATION => "SbasSolveMode.MIXED_AUGMENTATION",
            Self::SBAS_ONLY => "SbasSolveMode.SBAS_ONLY",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasMessageKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Decoded SBAS message family.
pub enum PySbasMessageKind {
    DO_NOT_USE,
    PRN_MASK,
    FAST_CORRECTIONS,
    INTEGRITY,
    FAST_DEGRADATION,
    GEO_NAV,
    NETWORK_TIME,
    GEO_ALMANAC,
    IGP_MASK,
    MIXED_CORRECTIONS,
    LONG_TERM_CORRECTIONS,
    IONO_DELAYS,
    UNSUPPORTED,
}

fn sbas_message_kind(message: &SbasMessage) -> PySbasMessageKind {
    match message {
        SbasMessage::DoNotUse(_) => PySbasMessageKind::DO_NOT_USE,
        SbasMessage::PrnMask(_) => PySbasMessageKind::PRN_MASK,
        SbasMessage::FastCorrections(_) => PySbasMessageKind::FAST_CORRECTIONS,
        SbasMessage::Integrity(_) => PySbasMessageKind::INTEGRITY,
        SbasMessage::FastDegradation(_) => PySbasMessageKind::FAST_DEGRADATION,
        SbasMessage::GeoNav(_) => PySbasMessageKind::GEO_NAV,
        SbasMessage::NetworkTime(_) => PySbasMessageKind::NETWORK_TIME,
        SbasMessage::GeoAlmanac(_) => PySbasMessageKind::GEO_ALMANAC,
        SbasMessage::IgpMask(_) => PySbasMessageKind::IGP_MASK,
        SbasMessage::MixedCorrections(_) => PySbasMessageKind::MIXED_CORRECTIONS,
        SbasMessage::LongTermCorrections(_) => PySbasMessageKind::LONG_TERM_CORRECTIONS,
        SbasMessage::IonoDelays(_) => PySbasMessageKind::IONO_DELAYS,
        SbasMessage::Unsupported(_) => PySbasMessageKind::UNSUPPORTED,
    }
}

#[pymethods]
impl PySbasMessageKind {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::DO_NOT_USE => "do_not_use",
            Self::PRN_MASK => "prn_mask",
            Self::FAST_CORRECTIONS => "fast_corrections",
            Self::INTEGRITY => "integrity",
            Self::FAST_DEGRADATION => "fast_degradation",
            Self::GEO_NAV => "geo_nav",
            Self::NETWORK_TIME => "network_time",
            Self::GEO_ALMANAC => "geo_almanac",
            Self::IGP_MASK => "igp_mask",
            Self::MIXED_CORRECTIONS => "mixed_corrections",
            Self::LONG_TERM_CORRECTIONS => "long_term_corrections",
            Self::IONO_DELAYS => "iono_delays",
            Self::UNSUPPORTED => "unsupported",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::DO_NOT_USE => "SbasMessageKind.DO_NOT_USE",
            Self::PRN_MASK => "SbasMessageKind.PRN_MASK",
            Self::FAST_CORRECTIONS => "SbasMessageKind.FAST_CORRECTIONS",
            Self::INTEGRITY => "SbasMessageKind.INTEGRITY",
            Self::FAST_DEGRADATION => "SbasMessageKind.FAST_DEGRADATION",
            Self::GEO_NAV => "SbasMessageKind.GEO_NAV",
            Self::NETWORK_TIME => "SbasMessageKind.NETWORK_TIME",
            Self::GEO_ALMANAC => "SbasMessageKind.GEO_ALMANAC",
            Self::IGP_MASK => "SbasMessageKind.IGP_MASK",
            Self::MIXED_CORRECTIONS => "SbasMessageKind.MIXED_CORRECTIONS",
            Self::LONG_TERM_CORRECTIONS => "SbasMessageKind.LONG_TERM_CORRECTIONS",
            Self::IONO_DELAYS => "SbasMessageKind.IONO_DELAYS",
            Self::UNSUPPORTED => "SbasMessageKind.UNSUPPORTED",
        }
    }
}

/// Decoded SBAS message payload.
#[pyclass(module = "sidereon._sidereon", name = "SbasMessage")]
#[derive(Clone)]
pub struct PySbasMessage {
    inner: SbasMessage,
}

impl From<SbasMessage> for PySbasMessage {
    fn from(inner: SbasMessage) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySbasMessage {
    /// Numeric SBAS message type.
    #[getter]
    fn message_type(&self) -> u8 {
        self.inner.message_type()
    }

    /// Decoded SBAS message family.
    #[getter]
    fn kind(&self) -> PySbasMessageKind {
        sbas_message_kind(&self.inner)
    }

    /// Stable decoded message family label.
    #[getter]
    fn kind_label(&self) -> &'static str {
        sbas_message_label(&self.inner)
    }

    /// Message type 0 payload, if this message is type 0.
    #[getter]
    fn do_not_use(&self) -> Option<PySbasDoNotUse> {
        match &self.inner {
            SbasMessage::DoNotUse(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 1 payload, if this message is type 1.
    #[getter]
    fn prn_mask(&self) -> Option<PySbasPrnMask> {
        match &self.inner {
            SbasMessage::PrnMask(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 2 through 5 payload, if present.
    #[getter]
    fn fast_corrections(&self) -> Option<PySbasFastCorrections> {
        match &self.inner {
            SbasMessage::FastCorrections(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 6 payload, if present.
    #[getter]
    fn integrity(&self) -> Option<PySbasIntegrity> {
        match &self.inner {
            SbasMessage::Integrity(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 7 payload, if present.
    #[getter]
    fn fast_degradation(&self) -> Option<PySbasFastDegradation> {
        match &self.inner {
            SbasMessage::FastDegradation(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 9 payload, if present.
    #[getter]
    fn geo_nav(&self) -> Option<PySbasGeoNav> {
        match &self.inner {
            SbasMessage::GeoNav(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 12 payload, if present.
    #[getter]
    fn network_time(&self) -> Option<PySbasNetworkTime> {
        match &self.inner {
            SbasMessage::NetworkTime(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 17 payload, if present.
    #[getter]
    fn geo_almanac(&self) -> Option<PySbasGeoAlmanac> {
        match &self.inner {
            SbasMessage::GeoAlmanac(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 18 payload, if present.
    #[getter]
    fn igp_mask(&self) -> Option<PySbasIgpMask> {
        match &self.inner {
            SbasMessage::IgpMask(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 24 payload, if present.
    #[getter]
    fn mixed_corrections(&self) -> Option<PySbasMixedCorrections> {
        match &self.inner {
            SbasMessage::MixedCorrections(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 25 payload, if present.
    #[getter]
    fn long_term_corrections(&self) -> Option<PySbasLongTermCorrections> {
        match &self.inner {
            SbasMessage::LongTermCorrections(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Message type 26 payload, if present.
    #[getter]
    fn iono_delays(&self) -> Option<PySbasIonoDelays> {
        match &self.inner {
            SbasMessage::IonoDelays(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    /// Unsupported payload, if this message type is not decoded.
    #[getter]
    fn unsupported(&self) -> Option<PySbasUnsupported> {
        match &self.inner {
            SbasMessage::Unsupported(value) => Some((*value).clone().into()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasMessage(message_type={}, kind={})",
            self.inner.message_type(),
            sbas_message_label(&self.inner)
        )
    }
}

macro_rules! sbas_payload {
    ($py:ident, $core:ident, $name:literal) => {
        #[pyclass(module = "sidereon._sidereon", name = $name)]
        #[derive(Clone)]
        /// Decoded SBAS payload wrapper.
        pub struct $py {
            inner: $core,
        }

        impl From<$core> for $py {
            fn from(inner: $core) -> Self {
                Self { inner }
            }
        }
    };
}

sbas_payload!(PySbasDoNotUse, SbasDoNotUse, "SbasDoNotUse");
sbas_payload!(PySbasPrnMask, SbasPrnMask, "SbasPrnMask");
sbas_payload!(
    PySbasFastCorrections,
    SbasFastCorrections,
    "SbasFastCorrections"
);
sbas_payload!(PySbasIntegrity, SbasIntegrity, "SbasIntegrity");
sbas_payload!(
    PySbasFastDegradation,
    SbasFastDegradation,
    "SbasFastDegradation"
);
sbas_payload!(PySbasGeoNav, SbasGeoNav, "SbasGeoNav");
sbas_payload!(PySbasNetworkTime, SbasNetworkTime, "SbasNetworkTime");
sbas_payload!(PySbasGeoAlmanac, SbasGeoAlmanac, "SbasGeoAlmanac");
sbas_payload!(
    PySbasMixedCorrections,
    SbasMixedCorrections,
    "SbasMixedCorrections"
);
sbas_payload!(
    PySbasMixedFastCorrections,
    SbasMixedFastCorrections,
    "SbasMixedFastCorrections"
);
sbas_payload!(
    PySbasLongTermCorrections,
    SbasLongTermCorrections,
    "SbasLongTermCorrections"
);
sbas_payload!(PySbasLongTermHalf, SbasLongTermHalf, "SbasLongTermHalf");
sbas_payload!(
    PySbasLongTermRecord,
    SbasLongTermRecord,
    "SbasLongTermRecord"
);
sbas_payload!(PySbasIgpMask, SbasIgpMask, "SbasIgpMask");
sbas_payload!(PySbasIonoDelays, SbasIonoDelays, "SbasIonoDelays");
sbas_payload!(PySbasIgpDelay, SbasIgpDelay, "SbasIgpDelay");
sbas_payload!(PySbasUnsupported, SbasUnsupported, "SbasUnsupported");

#[pymethods]
impl PySbasDoNotUse {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Raw 212-bit message data packed into bytes.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }
}

#[pymethods]
impl PySbasPrnMask {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Issue of data for the PRN mask.
    #[getter]
    fn iodp(&self) -> u8 {
        self.inner.iodp
    }

    /// Active-state bits for the 210 monitored slots.
    #[getter]
    fn mask(&self) -> Vec<bool> {
        self.inner.mask.to_vec()
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasFastCorrections {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Numeric SBAS message type.
    #[getter]
    fn message_type(&self) -> u8 {
        self.inner.message_type
    }

    /// Issue of data for fast corrections.
    #[getter]
    fn iodf(&self) -> u8 {
        self.inner.iodf
    }

    /// Issue of data for the PRN mask.
    #[getter]
    fn iodp(&self) -> u8 {
        self.inner.iodp
    }

    /// Pseudorange corrections in raw message units.
    #[getter]
    fn prc(&self) -> Vec<i16> {
        self.inner.prc.to_vec()
    }

    /// UDREI values for each correction slot, as a list of ints.
    ///
    /// A `Vec<u8>` converts to `bytes`, so the values are built into a list
    /// one int at a time.
    #[getter]
    fn udrei<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.udrei.iter().copied())
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasIntegrity {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Issue of data for fast corrections, one per block, as a list of ints.
    ///
    /// A `Vec<u8>` converts to `bytes`, so the values are built into a list
    /// one int at a time.
    #[getter]
    fn iodf<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.iodf.iter().copied())
    }

    /// UDREI values for monitored slots, as a list of ints.
    ///
    /// A `Vec<u8>` converts to `bytes`, so the values are built into a list
    /// one int at a time.
    #[getter]
    fn udrei<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.udrei.iter().copied())
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasFastDegradation {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// System latency in seconds.
    #[getter]
    fn system_latency_s(&self) -> u8 {
        self.inner.system_latency_s
    }

    /// Issue of data for the PRN mask.
    #[getter]
    fn iodp(&self) -> u8 {
        self.inner.iodp
    }

    /// Degradation indicators for monitored slots, as a list of ints.
    ///
    /// A `Vec<u8>` converts to `bytes`, so the values are built into a list
    /// one int at a time.
    #[getter]
    fn ai<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.ai.iter().copied())
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasGeoNav {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Time of day in seconds.
    #[getter]
    fn time_of_day_s(&self) -> u16 {
        self.inner.time_of_day_s
    }

    /// User range accuracy index.
    #[getter]
    fn ura(&self) -> u8 {
        self.inner.ura
    }

    /// ECEF X position in raw message units.
    #[getter]
    fn x_m(&self) -> i32 {
        self.inner.x_m
    }

    /// ECEF Y position in raw message units.
    #[getter]
    fn y_m(&self) -> i32 {
        self.inner.y_m
    }

    /// ECEF Z position in raw message units.
    #[getter]
    fn z_m(&self) -> i32 {
        self.inner.z_m
    }

    /// ECEF X velocity in raw message units.
    #[getter]
    fn x_rate_m_s(&self) -> i32 {
        self.inner.x_rate_m_s
    }

    /// ECEF Y velocity in raw message units.
    #[getter]
    fn y_rate_m_s(&self) -> i32 {
        self.inner.y_rate_m_s
    }

    /// ECEF Z velocity in raw message units.
    #[getter]
    fn z_rate_m_s(&self) -> i32 {
        self.inner.z_rate_m_s
    }

    /// ECEF X acceleration in raw message units.
    #[getter]
    fn x_accel_m_s2(&self) -> i16 {
        self.inner.x_accel_m_s2
    }

    /// ECEF Y acceleration in raw message units.
    #[getter]
    fn y_accel_m_s2(&self) -> i16 {
        self.inner.y_accel_m_s2
    }

    /// ECEF Z acceleration in raw message units.
    #[getter]
    fn z_accel_m_s2(&self) -> i16 {
        self.inner.z_accel_m_s2
    }

    /// GEO clock offset term in raw message units.
    #[getter]
    fn a_gf0_s(&self) -> i16 {
        self.inner.a_gf0_s
    }

    /// GEO clock drift term in raw message units.
    #[getter]
    fn a_gf1_s_s(&self) -> i16 {
        self.inner.a_gf1_s_s
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasNetworkTime {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Raw 212-bit message data packed into bytes.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }
}

#[pymethods]
impl PySbasGeoAlmanac {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Raw 212-bit message data packed into bytes.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }
}

#[pymethods]
impl PySbasMixedCorrections {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Fast-correction half of the mixed message.
    #[getter]
    fn fast(&self) -> PySbasMixedFastCorrections {
        self.inner.fast.clone().into()
    }

    /// Long-term half of the mixed message.
    #[getter]
    fn long_term(&self) -> PySbasLongTermHalf {
        self.inner.long_term.clone().into()
    }
}

#[pymethods]
impl PySbasMixedFastCorrections {
    /// Issue of data for fast corrections.
    #[getter]
    fn iodf(&self) -> u8 {
        self.inner.iodf
    }

    /// Issue of data for the PRN mask.
    #[getter]
    fn iodp(&self) -> u8 {
        self.inner.iodp
    }

    /// Correction block id.
    #[getter]
    fn block_id(&self) -> u8 {
        self.inner.block_id
    }

    /// Pseudorange corrections in raw message units.
    #[getter]
    fn prc(&self) -> Vec<i16> {
        self.inner.prc.to_vec()
    }

    /// UDREI values for each correction slot, as a list of ints.
    ///
    /// A `Vec<u8>` converts to `bytes`, so the values are built into a list
    /// one int at a time.
    #[getter]
    fn udrei<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        PyList::new(py, self.inner.udrei.iter().copied())
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasLongTermCorrections {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Two long-term correction halves.
    #[getter]
    fn halves(&self) -> Vec<PySbasLongTermHalf> {
        self.inner.halves.iter().cloned().map(Into::into).collect()
    }
}

#[pymethods]
impl PySbasLongTermHalf {
    /// Whether records include velocity fields.
    #[getter]
    fn velocity_code(&self) -> bool {
        self.inner.velocity_code
    }

    /// Issue of data for the PRN mask.
    #[getter]
    fn iodp(&self) -> u8 {
        self.inner.iodp
    }

    /// Long-term correction records.
    #[getter]
    fn records(&self) -> Vec<PySbasLongTermRecord> {
        self.inner.records.iter().cloned().map(Into::into).collect()
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasLongTermRecord {
    /// Monitored satellite index.
    #[getter]
    fn monitored_index(&self) -> u8 {
        self.inner.monitored_index
    }

    /// Issue of data ephemeris.
    #[getter]
    fn iode(&self) -> u8 {
        self.inner.iode
    }

    /// ECEF X correction in raw message units.
    #[getter]
    fn delta_x(&self) -> i32 {
        self.inner.delta_x
    }

    /// ECEF Y correction in raw message units.
    #[getter]
    fn delta_y(&self) -> i32 {
        self.inner.delta_y
    }

    /// ECEF Z correction in raw message units.
    #[getter]
    fn delta_z(&self) -> i32 {
        self.inner.delta_z
    }

    /// ECEF X rate correction in raw message units.
    #[getter]
    fn delta_x_rate(&self) -> i32 {
        self.inner.delta_x_rate
    }

    /// ECEF Y rate correction in raw message units.
    #[getter]
    fn delta_y_rate(&self) -> i32 {
        self.inner.delta_y_rate
    }

    /// ECEF Z rate correction in raw message units.
    #[getter]
    fn delta_z_rate(&self) -> i32 {
        self.inner.delta_z_rate
    }

    /// Clock offset correction in raw message units.
    #[getter]
    fn delta_a_f0(&self) -> i32 {
        self.inner.delta_a_f0
    }

    /// Clock drift correction in raw message units.
    #[getter]
    fn delta_a_f1(&self) -> i32 {
        self.inner.delta_a_f1
    }

    /// Optional time of day in seconds, scaled by the message definition.
    #[getter]
    fn time_of_day_s(&self) -> Option<u32> {
        self.inner.time_of_day_s
    }
}

#[pymethods]
impl PySbasIgpMask {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// IGP band number.
    #[getter]
    fn band_number(&self) -> u8 {
        self.inner.band_number
    }

    /// Issue of data for ionosphere.
    #[getter]
    fn iodi(&self) -> u8 {
        self.inner.iodi
    }

    /// Active-state bits for the 201 IGP slots.
    #[getter]
    fn mask(&self) -> Vec<bool> {
        self.inner.mask.to_vec()
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasIonoDelays {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// IGP band number.
    #[getter]
    fn band_number(&self) -> u8 {
        self.inner.band_number
    }

    /// IGP block id.
    #[getter]
    fn block_id(&self) -> u8 {
        self.inner.block_id
    }

    /// Issue of data for ionosphere.
    #[getter]
    fn iodi(&self) -> u8 {
        self.inner.iodi
    }

    /// Ionospheric delay entries.
    #[getter]
    fn entries(&self) -> Vec<PySbasIgpDelay> {
        self.inner.entries.iter().cloned().map(Into::into).collect()
    }

    /// Reserved spare-bit values as `(value, width)` pairs.
    #[getter]
    fn reserved(&self) -> Vec<(u64, u8)> {
        spare_bits(&self.inner.reserved)
    }
}

#[pymethods]
impl PySbasIgpDelay {
    /// Vertical delay in raw message units.
    #[getter]
    fn vertical_delay(&self) -> u16 {
        self.inner.vertical_delay
    }

    /// GIVEI value.
    #[getter]
    fn givei(&self) -> u8 {
        self.inner.givei
    }
}

#[pymethods]
impl PySbasUnsupported {
    /// SBAS preamble byte.
    #[getter]
    fn preamble(&self) -> u8 {
        self.inner.preamble
    }

    /// Numeric SBAS message type.
    #[getter]
    fn message_type(&self) -> u8 {
        self.inner.message_type
    }

    /// Raw 212-bit message data packed into bytes.
    #[getter]
    fn data<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.data)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasBlock")]
#[derive(Clone)]
/// Decoded SBAS message block.
pub struct PySbasBlock {
    inner: SbasBlock,
}

#[pymethods]
impl PySbasBlock {
    #[getter]
    fn form(&self) -> PySbasWireForm {
        self.inner.form.into()
    }

    #[getter]
    fn message_type(&self) -> u8 {
        self.inner.message.message_type()
    }

    #[getter]
    fn kind(&self) -> PySbasMessageKind {
        sbas_message_kind(&self.inner.message)
    }

    #[getter]
    fn kind_label(&self) -> &'static str {
        sbas_message_label(&self.inner.message)
    }

    /// Decoded SBAS message payload.
    #[getter]
    fn message(&self) -> PySbasMessage {
        self.inner.message.clone().into()
    }

    /// The six bits completing the last byte of either wire form, as read
    /// and written back.
    #[getter]
    fn pad_bits(&self) -> u8 {
        self.inner.pad_bits
    }

    /// Encode the block back to its wire form. A value the wire form cannot
    /// carry as held (a field wider than its width, a reserved-segment
    /// layout other than the message's, a raw payload that is not 212 bits)
    /// raises `RtcmEncodeError`; under `SbasPolicy.STRICT` so does a preamble
    /// other than 0x53, 0x9A and 0xC6.
    #[pyo3(signature = (policy=PySbasPolicy::STRICT))]
    fn encode<'py>(&self, py: Python<'py>, policy: PySbasPolicy) -> PyResult<Bound<'py, PyBytes>> {
        let (bytes, _) = self
            .inner
            .encode_with_policy(policy.into())
            .map_err(|err| to_rtcm_encode_err(py, err))?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Encode under `policy`, returning the bytes and the departures written.
    fn encode_with_policy<'py>(
        &self,
        py: Python<'py>,
        policy: PySbasPolicy,
    ) -> PyResult<(Bound<'py, PyBytes>, Vec<PySbasDeparture>)> {
        let (bytes, written) = self
            .inner
            .encode_with_policy(policy.into())
            .map_err(|err| to_rtcm_encode_err(py, err))?;
        Ok((PyBytes::new(py, &bytes), sbas_departures(written)))
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasBlock(form={}, message_type={}, kind={})",
            sbas_wire_form_label(self.inner.form),
            self.inner.message.message_type(),
            sbas_message_label(&self.inner.message)
        )
    }
}

impl PySbasBlock {
    fn from_inner(inner: SbasBlock) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasLogBlock")]
#[derive(Clone)]
/// One SBAS log row with epoch, satellite, and raw message bytes.
pub struct PySbasLogBlock {
    inner: SbasLogBlock,
}

#[pymethods]
impl PySbasLogBlock {
    #[getter]
    fn satellite_id(&self) -> String {
        py_satellite(self.inner.satellite_id)
    }

    #[getter]
    fn week(&self) -> u32 {
        self.inner.epoch.week
    }

    #[getter]
    fn tow_s(&self) -> f64 {
        self.inner.epoch.tow_s
    }

    #[getter]
    fn form(&self) -> PySbasWireForm {
        self.inner.form.into()
    }

    #[getter]
    fn bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.bytes)
    }

    /// The message type the record's own field states, `None` for an
    /// eight-field EMS comma line, which carries none.
    #[getter]
    fn declared_message_type(&self) -> Option<u8> {
        self.inner.declared_message_type
    }

    /// The six-bit message type the record's bytes carry.
    #[getter]
    fn message_type(&self) -> Option<u8> {
        self.inner.message_type()
    }

    #[pyo3(signature = (policy=PySbasPolicy::STRICT))]
    fn decode(&self, policy: PySbasPolicy) -> PyResult<PySbasBlock> {
        SbasBlock::decode_with_policy(&self.inner.bytes, self.inner.form, policy.into())
            .map(|(block, _)| PySbasBlock::from_inner(block))
            .map_err(to_rtcm_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasLogBlock(satellite_id={:?}, week={}, tow_s={:.3}, form={})",
            py_satellite(self.inner.satellite_id),
            self.inner.epoch.week,
            self.inner.epoch.tow_s,
            sbas_wire_form_label(self.inner.form)
        )
    }
}

impl From<SbasLogBlock> for PySbasLogBlock {
    fn from(inner: SbasLogBlock) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasFastCorrection")]
#[derive(Clone)]
/// SBAS fast pseudorange correction for one satellite.
pub struct PySbasFastCorrection {
    inner: SbasFastCorrection,
}

#[pymethods]
impl PySbasFastCorrection {
    #[getter]
    fn prc_m(&self) -> f64 {
        self.inner.prc_m
    }

    #[getter]
    fn rrc_m_s(&self) -> f64 {
        self.inner.rrc_m_s
    }

    #[getter]
    fn udrei(&self) -> u8 {
        self.inner.udrei
    }

    #[getter]
    fn t_of_j2000_s(&self) -> f64 {
        self.inner.t_of_j2000_s
    }

    #[getter]
    fn iodf(&self) -> u8 {
        self.inner.iodf
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasFastCorrection(prc_m={:.6}, rrc_m_s={:.6}, udrei={}, iodf={})",
            self.inner.prc_m, self.inner.rrc_m_s, self.inner.udrei, self.inner.iodf
        )
    }
}

impl From<SbasFastCorrection> for PySbasFastCorrection {
    fn from(inner: SbasFastCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasLongTermCorrection")]
#[derive(Clone)]
/// SBAS long-term orbit and clock correction for one satellite.
pub struct PySbasLongTermCorrection {
    inner: SbasLongTermCorrection,
}

#[pymethods]
impl PySbasLongTermCorrection {
    #[getter]
    fn iode(&self) -> u8 {
        self.inner.iode
    }

    #[getter]
    fn delta_ecef_m(&self) -> [f64; 3] {
        self.inner.delta_ecef_m
    }

    #[getter]
    fn delta_ecef_rate_m_s(&self) -> [f64; 3] {
        self.inner.delta_ecef_rate_m_s
    }

    #[getter]
    fn delta_af0_s(&self) -> f64 {
        self.inner.delta_af0_s
    }

    #[getter]
    fn delta_af1_s_s(&self) -> f64 {
        self.inner.delta_af1_s_s
    }

    #[getter]
    fn t0_j2000_s(&self) -> f64 {
        self.inner.t0_j2000_s
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasLongTermCorrection(iode={}, delta_af0_s={:.6e}, t0_j2000_s={:.3})",
            self.inner.iode, self.inner.delta_af0_s, self.inner.t0_j2000_s
        )
    }
}

impl From<SbasLongTermCorrection> for PySbasLongTermCorrection {
    fn from(inner: SbasLongTermCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasIgp")]
#[derive(Clone)]
/// One SBAS ionospheric grid point.
pub struct PySbasIgp {
    inner: SbasIgp,
}

#[pymethods]
impl PySbasIgp {
    #[getter]
    fn lat_deg(&self) -> f64 {
        self.inner.lat_deg
    }

    #[getter]
    fn lon_deg(&self) -> f64 {
        self.inner.lon_deg
    }

    #[getter]
    fn vertical_delay_m(&self) -> f64 {
        self.inner.vertical_delay_m
    }

    #[getter]
    fn give_variance_m2(&self) -> Option<f64> {
        self.inner.give_variance_m2
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasIgp(lat_deg={:.3}, lon_deg={:.3}, vertical_delay_m={:.6})",
            self.inner.lat_deg, self.inner.lon_deg, self.inner.vertical_delay_m
        )
    }
}

impl From<SbasIgp> for PySbasIgp {
    fn from(inner: SbasIgp) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasIonoGrid")]
#[derive(Clone)]
/// SBAS ionospheric grid for one GEO provider.
pub struct PySbasIonoGrid {
    inner: SbasIonoGrid,
}

#[pymethods]
impl PySbasIonoGrid {
    #[getter]
    fn iodi(&self) -> u8 {
        self.inner.iodi
    }

    #[getter]
    fn igps(&self) -> Vec<PySbasIgp> {
        self.inner.igps().iter().cloned().map(Into::into).collect()
    }

    /// Grid points whose latest delay entry makes them unusable (delay 511,
    /// "do not use", or GIVEI 15, "not monitored"), with the raw entry.
    #[getter]
    fn unavailable_igps(&self) -> Vec<PySbasUnavailableIgp> {
        self.inner
            .unavailable_igps()
            .iter()
            .cloned()
            .map(|inner| PySbasUnavailableIgp { inner })
            .collect()
    }

    #[pyo3(signature = (latitude_deg, longitude_deg, height_m, elevation_rad, azimuth_rad, frequency_hz))]
    fn slant_delay_m(
        &self,
        latitude_deg: f64,
        longitude_deg: f64,
        height_m: f64,
        elevation_rad: f64,
        azimuth_rad: f64,
        frequency_hz: f64,
    ) -> PyResult<Option<f64>> {
        let receiver = Wgs84Geodetic::new(
            latitude_deg.to_radians(),
            longitude_deg.to_radians(),
            height_m,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(self
            .inner
            .slant_delay_m(receiver, elevation_rad, azimuth_rad, frequency_hz))
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasIonoGrid(iodi={}, igps={})",
            self.inner.iodi,
            self.inner.igps().len()
        )
    }
}

impl From<SbasIonoGrid> for PySbasIonoGrid {
    fn from(inner: SbasIonoGrid) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasGeoState")]
#[derive(Clone)]
/// SBAS GEO navigation state.
pub struct PySbasGeoState {
    inner: SbasGeoState,
}

#[pymethods]
impl PySbasGeoState {
    #[getter]
    fn position_ecef_m(&self) -> [f64; 3] {
        self.inner.position_ecef_m
    }

    #[getter]
    fn velocity_ecef_m_s(&self) -> [f64; 3] {
        self.inner.velocity_ecef_m_s
    }

    #[getter]
    fn acceleration_ecef_m_s2(&self) -> [f64; 3] {
        self.inner.acceleration_ecef_m_s2
    }

    #[getter]
    fn clock_offset_s(&self) -> f64 {
        self.inner.clock_offset_s
    }

    #[getter]
    fn clock_drift_s_s(&self) -> f64 {
        self.inner.clock_drift_s_s
    }

    #[getter]
    fn t0_j2000_s(&self) -> f64 {
        self.inner.t0_j2000_s
    }

    fn state_at(&self, t_j2000_s: f64) -> ([f64; 3], f64) {
        self.inner.state_at(t_j2000_s)
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasGeoState(t0_j2000_s={:.3}, clock_offset_s={:.6e})",
            self.inner.t0_j2000_s, self.inner.clock_offset_s
        )
    }
}

impl From<SbasGeoState> for PySbasGeoState {
    fn from(inner: SbasGeoState) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasCorrectionStore")]
/// Mutable SBAS correction store.
pub struct PySbasCorrectionStore {
    pub(crate) inner: SbasCorrectionStore,
}

#[pymethods]
impl PySbasCorrectionStore {
    /// Build an empty SBAS correction store.
    #[new]
    fn new() -> Self {
        Self {
            inner: SbasCorrectionStore::new(),
        }
    }

    /// Ingest a decoded SBAS block for a GEO satellite and epoch.
    #[pyo3(signature = (block, geo_satellite_id, week, tow_s, time_scale=PyTimeScale::GPST))]
    fn ingest(
        &mut self,
        block: &PySbasBlock,
        geo_satellite_id: &str,
        week: u32,
        tow_s: f64,
        time_scale: PyTimeScale,
    ) -> PyResult<()> {
        let geo = parse_satellite(geo_satellite_id)?;
        let epoch = gnss_week_tow(time_scale, week, tow_s)?;
        self.inner
            .ingest(&block.inner.message, geo, epoch)
            .map_err(to_rtcm_err)
    }

    fn ready_geos(&self, t_j2000_s: f64) -> Vec<String> {
        self.inner
            .ready_geos(t_j2000_s)
            .into_iter()
            .map(py_satellite)
            .collect()
    }

    /// Return the latest fast correction for a GEO and satellite pair.
    fn fast(
        &self,
        geo_satellite_id: &str,
        satellite_id: &str,
    ) -> PyResult<Option<PySbasFastCorrection>> {
        let geo = parse_satellite(geo_satellite_id)?;
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.fast(geo, sat).cloned().map(Into::into))
    }

    /// Corrections a GEO addressed to active PRN-mask bits that name no
    /// satellite held here - a future-GNSS or unassigned mask number - counted
    /// per 1-based mask number. Such a bit keeps its place among the active
    /// bits, so the corrections after it still reach their own satellites; the
    /// ones addressed to it are applied to no satellite and counted here. None
    /// when the GEO has no partition.
    fn unassigned_mask_corrections(
        &self,
        geo_satellite_id: &str,
    ) -> PyResult<Option<BTreeMap<u8, u64>>> {
        let geo = parse_satellite(geo_satellite_id)?;
        Ok(self.inner.unassigned_mask_corrections(geo).cloned())
    }

    /// Return the latest long-term correction for a GEO and satellite pair.
    fn long_term(
        &self,
        geo_satellite_id: &str,
        satellite_id: &str,
    ) -> PyResult<Option<PySbasLongTermCorrection>> {
        let geo = parse_satellite(geo_satellite_id)?;
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.long_term(geo, sat).cloned().map(Into::into))
    }

    /// Return the ionospheric grid for a GEO satellite.
    fn iono_grid(&self, geo_satellite_id: &str) -> PyResult<Option<PySbasIonoGrid>> {
        let geo = parse_satellite(geo_satellite_id)?;
        Ok(self.inner.iono_grid(geo).cloned().map(Into::into))
    }

    /// Return the navigation state for a GEO satellite.
    fn geo_nav(&self, geo_satellite_id: &str) -> PyResult<Option<PySbasGeoState>> {
        let geo = parse_satellite(geo_satellite_id)?;
        Ok(self.inner.geo_nav(geo).cloned().map(Into::into))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasCorrectedEphemeris")]
/// Broadcast ephemeris source corrected with SBAS messages.
pub struct PySbasCorrectedEphemeris {
    broadcast: Py<PyBroadcastEphemeris>,
    store: Py<PySbasCorrectionStore>,
    geo: GnssSatelliteId,
    mode: SbasSolveMode,
}

#[pymethods]
impl PySbasCorrectedEphemeris {
    /// Build an SBAS-corrected ephemeris source.
    #[new]
    #[pyo3(signature = (broadcast, store, geo_satellite_id, mode=PySbasSolveMode::MIXED_AUGMENTATION))]
    fn new(
        broadcast: Py<PyBroadcastEphemeris>,
        store: Py<PySbasCorrectionStore>,
        geo_satellite_id: &str,
        mode: PySbasSolveMode,
    ) -> PyResult<Self> {
        Ok(Self {
            broadcast,
            store,
            geo: parse_satellite(geo_satellite_id)?,
            mode: mode.into(),
        })
    }

    /// Return corrected ECEF position in metres and clock offset in seconds.
    fn position_clock_at_j2000_s(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        t_j2000_s: f64,
    ) -> PyResult<Option<([f64; 3], f64)>> {
        let sat = parse_satellite(satellite_id)?;
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = SbasCorrectedEphemeris::new(&broadcast.inner, &store.inner, self.geo)
            .with_mode(self.mode);
        Ok(source.position_clock_at_j2000_s(sat, t_j2000_s))
    }

    fn selected_state_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<crate::ephemeris::PyEphemerisQueryState>> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = SbasCorrectedEphemeris::new(&broadcast.inner, &store.inner, self.geo)
            .with_mode(self.mode);
        crate::ephemeris::source_state_at_epoch_query(&source, satellite_id, epoch, selection_epoch)
    }

    fn transmit_epoch_clock_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<(f64, Option<&'static str>)>> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = SbasCorrectedEphemeris::new(&broadcast.inner, &store.inner, self.geo)
            .with_mode(self.mode);
        crate::ephemeris::source_transmit_clock_at_epoch_query(
            &source,
            satellite_id,
            epoch,
            selection_epoch,
        )
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        state_epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<f64> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = SbasCorrectedEphemeris::new(&broadcast.inner, &store.inner, self.geo)
            .with_mode(self.mode);
        crate::ephemeris::source_variance_at_epoch_query(
            &source,
            satellite_id,
            state_epoch,
            selection_epoch,
        )
    }

    fn clock_relativity_for_state_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        position_ecef_m: [f64; 3],
    ) -> PyResult<crate::ephemeris::PyClockRelativity> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = SbasCorrectedEphemeris::new(&broadcast.inner, &store.inner, self.geo)
            .with_mode(self.mode);
        crate::ephemeris::source_clock_relativity_at_epoch_query(
            &source,
            satellite_id,
            epoch,
            position_ecef_m,
        )
    }

    /// Return the active ionospheric grid for the selected GEO.
    fn iono_grid(&self, py: Python<'_>) -> Option<PySbasIonoGrid> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = SbasCorrectedEphemeris::new(&broadcast.inner, &store.inner, self.geo)
            .with_mode(self.mode);
        source.iono_grid().cloned().map(Into::into)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// RTCM SSR message kind.
pub enum PySsrKind {
    ORBIT,
    CLOCK,
    COMBINED_ORBIT_CLOCK,
    CODE_BIAS,
    PHASE_BIAS,
    URA,
    HIGH_RATE_CLOCK,
}

impl From<SsrKind> for PySsrKind {
    fn from(kind: SsrKind) -> Self {
        match kind {
            SsrKind::Orbit => Self::ORBIT,
            SsrKind::Clock => Self::CLOCK,
            SsrKind::CombinedOrbitClock => Self::COMBINED_ORBIT_CLOCK,
            SsrKind::CodeBias => Self::CODE_BIAS,
            SsrKind::PhaseBias => Self::PHASE_BIAS,
            SsrKind::Ura => Self::URA,
            SsrKind::HighRateClock => Self::HIGH_RATE_CLOCK,
        }
    }
}

#[pymethods]
impl PySsrKind {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::ORBIT => "orbit",
            Self::CLOCK => "clock",
            Self::COMBINED_ORBIT_CLOCK => "combined_orbit_clock",
            Self::CODE_BIAS => "code_bias",
            Self::PHASE_BIAS => "phase_bias",
            Self::URA => "ura",
            Self::HIGH_RATE_CLOCK => "high_rate_clock",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::ORBIT => "SsrKind.ORBIT",
            Self::CLOCK => "SsrKind.CLOCK",
            Self::COMBINED_ORBIT_CLOCK => "SsrKind.COMBINED_ORBIT_CLOCK",
            Self::CODE_BIAS => "SsrKind.CODE_BIAS",
            Self::PHASE_BIAS => "SsrKind.PHASE_BIAS",
            Self::URA => "SsrKind.URA",
            Self::HIGH_RATE_CLOCK => "SsrKind.HIGH_RATE_CLOCK",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrSource", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Provider family that produced an SSR correction.
pub enum PySsrSource {
    RTCM_SSR,
    GALILEO_HAS,
    IGS_SSR,
}

impl From<PySsrSource> for SsrSource {
    fn from(source: PySsrSource) -> Self {
        match source {
            PySsrSource::RTCM_SSR => SsrSource::RtcmSsr,
            PySsrSource::GALILEO_HAS => SsrSource::GalileoHas,
            PySsrSource::IGS_SSR => SsrSource::IgsSsr,
        }
    }
}

impl From<SsrSource> for PySsrSource {
    fn from(source: SsrSource) -> Self {
        match source {
            SsrSource::RtcmSsr => Self::RTCM_SSR,
            SsrSource::GalileoHas => Self::GALILEO_HAS,
            SsrSource::IgsSsr => Self::IGS_SSR,
        }
    }
}

#[pymethods]
impl PySsrSource {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::RTCM_SSR => "rtcm_ssr",
            Self::GALILEO_HAS => "galileo_has",
            Self::IGS_SSR => "igs_ssr",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::RTCM_SSR => "SsrSource.RTCM_SSR",
            Self::GALILEO_HAS => "SsrSource.GALILEO_HAS",
            Self::IGS_SSR => "SsrSource.IGS_SSR",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrSolution", eq)]
#[derive(Clone, Copy, PartialEq, Eq)]
/// Provider and solution identity for an SSR correction.
pub struct PySsrSolution {
    inner: SsrSolution,
}

#[pymethods]
impl PySsrSolution {
    #[getter]
    fn source(&self) -> PySsrSource {
        self.inner.source.into()
    }

    #[getter]
    fn provider_id(&self) -> u16 {
        self.inner.provider_id
    }

    #[getter]
    fn solution_id(&self) -> u8 {
        self.inner.solution_id
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrSolution(source={}, provider_id={}, solution_id={})",
            self.source().label(),
            self.inner.provider_id,
            self.inner.solution_id
        )
    }
}

impl From<SsrSolution> for PySsrSolution {
    fn from(inner: SsrSolution) -> Self {
        Self { inner }
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "OrbitReferencePoint",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Orbit reference point used by SSR orbit corrections.
pub enum PyOrbitReferencePoint {
    ANTENNA_PHASE_CENTER,
    CENTER_OF_MASS,
}

#[pymethods]
impl PyOrbitReferencePoint {
    fn __repr__(&self) -> &'static str {
        match self {
            Self::ANTENNA_PHASE_CENTER => "OrbitReferencePoint.ANTENNA_PHASE_CENTER",
            Self::CENTER_OF_MASS => "OrbitReferencePoint.CENTER_OF_MASS",
        }
    }
}

impl From<PyOrbitReferencePoint> for OrbitReferencePoint {
    fn from(value: PyOrbitReferencePoint) -> Self {
        match value {
            PyOrbitReferencePoint::ANTENNA_PHASE_CENTER => OrbitReferencePoint::AntennaPhaseCenter,
            PyOrbitReferencePoint::CENTER_OF_MASS => OrbitReferencePoint::CenterOfMass,
        }
    }
}

impl From<OrbitReferencePoint> for PyOrbitReferencePoint {
    fn from(value: OrbitReferencePoint) -> Self {
        match value {
            OrbitReferencePoint::AntennaPhaseCenter => Self::ANTENNA_PHASE_CENTER,
            OrbitReferencePoint::CenterOfMass => Self::CENTER_OF_MASS,
        }
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "MissingCorrectionAction",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Action taken when an SSR correction is missing.
pub enum PyMissingCorrectionAction {
    DECLINE,
    FALL_BACK_TO_BROADCAST,
}

#[pymethods]
impl PyMissingCorrectionAction {
    fn __repr__(&self) -> &'static str {
        match self {
            Self::DECLINE => "MissingCorrectionAction.DECLINE",
            Self::FALL_BACK_TO_BROADCAST => "MissingCorrectionAction.FALL_BACK_TO_BROADCAST",
        }
    }
}

impl From<PyMissingCorrectionAction> for MissingCorrectionAction {
    fn from(value: PyMissingCorrectionAction) -> Self {
        match value {
            PyMissingCorrectionAction::DECLINE => MissingCorrectionAction::Decline,
            PyMissingCorrectionAction::FALL_BACK_TO_BROADCAST => {
                MissingCorrectionAction::FallBackToBroadcast
            }
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrFallbackPolicy")]
#[derive(Clone)]
/// Fallback policy used by an SSR-corrected ephemeris source.
pub struct PySsrFallbackPolicy {
    inner: SsrFallbackPolicy,
}

#[pymethods]
impl PySsrFallbackPolicy {
    /// Build an SSR fallback policy.
    #[new]
    #[pyo3(signature = (on_missing_correction=PyMissingCorrectionAction::DECLINE, allow_regional_providers=None))]
    fn new(
        on_missing_correction: PyMissingCorrectionAction,
        allow_regional_providers: Option<Vec<u16>>,
    ) -> Self {
        let regional = match allow_regional_providers {
            Some(providers) => RegionalPolicy::AllowProviders(providers.into_iter().collect()),
            None => RegionalPolicy::DeclineRegional,
        };
        Self {
            inner: SsrFallbackPolicy {
                on_missing_correction: on_missing_correction.into(),
                regional,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrFallbackPolicy(on_missing_correction={})",
            match self.inner.on_missing_correction {
                MissingCorrectionAction::Decline => "decline",
                MissingCorrectionAction::FallBackToBroadcast => "fall_back_to_broadcast",
            }
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrOrbitCorrection")]
#[derive(Clone, Copy)]
/// SSR orbit correction for one satellite.
pub struct PySsrOrbitCorrection {
    inner: SsrOrbitCorrection,
}

#[pymethods]
impl PySsrOrbitCorrection {
    #[getter]
    fn solution(&self) -> PySsrSolution {
        self.inner.solution.into()
    }

    #[getter]
    fn iode(&self) -> u32 {
        self.inner.iode
    }

    #[getter]
    fn iod_ssr(&self) -> u8 {
        self.inner.iod_ssr
    }

    #[getter]
    fn crs_regional(&self) -> bool {
        self.inner.crs_regional
    }

    #[getter]
    fn reference_point(&self) -> PyOrbitReferencePoint {
        self.inner.reference_point.into()
    }

    #[getter]
    fn radial_m(&self) -> f64 {
        self.inner.radial_m
    }

    #[getter]
    fn along_m(&self) -> f64 {
        self.inner.along_m
    }

    #[getter]
    fn cross_m(&self) -> f64 {
        self.inner.cross_m
    }

    #[getter]
    fn radial_rate_m_s(&self) -> f64 {
        self.inner.radial_rate_m_s
    }

    #[getter]
    fn along_rate_m_s(&self) -> f64 {
        self.inner.along_rate_m_s
    }

    #[getter]
    fn cross_rate_m_s(&self) -> f64 {
        self.inner.cross_rate_m_s
    }

    #[getter]
    fn ref_epoch_j2000_s(&self) -> f64 {
        self.inner.ref_epoch_j2000_s
    }

    #[getter]
    fn update_interval_s(&self) -> f64 {
        self.inner.update_interval_s
    }

    /// The epoch the correction was transmitted for, seconds past J2000; for
    /// Galileo HAS the TOH epoch. RTCM SSR orbit and clock corrections are
    /// gated by age from it.
    #[getter]
    fn transmitted_epoch_j2000_s(&self) -> f64 {
        self.inner.transmitted_epoch_j2000_s
    }

    /// The navigation message the correction refers to: `rtcm`, or `has:N`
    /// with the Galileo HAS index as transmitted. A reserved HAS index is
    /// stored and not applied.
    #[getter]
    fn nav_message(&self) -> String {
        nav_message_label(self.inner.nav_message)
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrOrbitCorrection(iode={}, iod_ssr={}, radial_m={:.6}, along_m={:.6}, cross_m={:.6})",
            self.inner.iode,
            self.inner.iod_ssr,
            self.inner.radial_m,
            self.inner.along_m,
            self.inner.cross_m
        )
    }
}

impl From<SsrOrbitCorrection> for PySsrOrbitCorrection {
    fn from(inner: SsrOrbitCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrHighRateClock")]
#[derive(Clone, Copy)]
/// SSR high-rate clock correction for one satellite.
pub struct PySsrHighRateClock {
    inner: SsrHighRateClock,
}

#[pymethods]
impl PySsrHighRateClock {
    #[getter]
    fn solution(&self) -> PySsrSolution {
        self.inner.solution.into()
    }

    #[getter]
    fn c0_m(&self) -> f64 {
        self.inner.c0_m
    }

    #[getter]
    fn ref_epoch_j2000_s(&self) -> f64 {
        self.inner.ref_epoch_j2000_s
    }

    #[getter]
    fn update_interval_s(&self) -> f64 {
        self.inner.update_interval_s
    }

    /// The epoch the correction was transmitted for, seconds past J2000.
    #[getter]
    fn transmitted_epoch_j2000_s(&self) -> f64 {
        self.inner.transmitted_epoch_j2000_s
    }

    /// IOD SSR of the high-rate clock.
    #[getter]
    fn iod_ssr(&self) -> u8 {
        self.inner.iod_ssr
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrHighRateClock(c0_m={:.6}, ref_epoch_j2000_s={:.3})",
            self.inner.c0_m, self.inner.ref_epoch_j2000_s
        )
    }
}

impl From<SsrHighRateClock> for PySsrHighRateClock {
    fn from(inner: SsrHighRateClock) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrClockCorrection")]
#[derive(Clone, Copy)]
/// SSR clock correction for one satellite.
pub struct PySsrClockCorrection {
    inner: SsrClockCorrection,
}

#[pymethods]
impl PySsrClockCorrection {
    #[getter]
    fn solution(&self) -> PySsrSolution {
        self.inner.solution.into()
    }

    #[getter]
    fn iod_ssr(&self) -> u8 {
        self.inner.iod_ssr
    }

    #[getter]
    fn c0_m(&self) -> f64 {
        self.inner.c0_m
    }

    #[getter]
    fn c1_m_s(&self) -> f64 {
        self.inner.c1_m_s
    }

    #[getter]
    fn c2_m_s2(&self) -> f64 {
        self.inner.c2_m_s2
    }

    #[getter]
    fn ref_epoch_j2000_s(&self) -> f64 {
        self.inner.ref_epoch_j2000_s
    }

    #[getter]
    fn update_interval_s(&self) -> f64 {
        self.inner.update_interval_s
    }

    #[getter]
    fn high_rate(&self) -> Option<PySsrHighRateClock> {
        self.inner.high_rate.map(Into::into)
    }

    /// The epoch the correction was transmitted for, seconds past J2000.
    #[getter]
    fn transmitted_epoch_j2000_s(&self) -> f64 {
        self.inner.transmitted_epoch_j2000_s
    }

    /// The navigation message the correction refers to: `rtcm`, or `has:N`.
    #[getter]
    fn nav_message(&self) -> String {
        nav_message_label(self.inner.nav_message)
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrClockCorrection(iod_ssr={}, c0_m={:.6}, c1_m_s={:.6e}, c2_m_s2={:.6e})",
            self.inner.iod_ssr, self.inner.c0_m, self.inner.c1_m_s, self.inner.c2_m_s2
        )
    }
}

impl From<SsrClockCorrection> for PySsrClockCorrection {
    fn from(inner: SsrClockCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrMessage")]
#[derive(Clone)]
/// Decoded RTCM SSR message.
pub struct PySsrMessage {
    inner: SsrMessage,
}

#[pymethods]
impl PySsrMessage {
    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }

    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    #[getter]
    fn kind(&self) -> PySsrKind {
        self.inner.kind.into()
    }

    #[getter]
    fn kind_label(&self) -> &'static str {
        ssr_kind_label(self.inner.kind)
    }

    #[getter]
    fn epoch_time_s(&self) -> u32 {
        self.inner.header.epoch_time_s
    }

    #[getter]
    fn update_interval_index(&self) -> u8 {
        self.inner.header.update_interval
    }

    #[getter]
    fn multiple_message(&self) -> bool {
        self.inner.header.multiple_message
    }

    #[getter]
    fn iod_ssr(&self) -> u8 {
        self.inner.header.iod_ssr
    }

    #[getter]
    fn provider_id(&self) -> u16 {
        self.inner.header.provider_id
    }

    #[getter]
    fn solution_id(&self) -> u8 {
        self.inner.header.solution_id
    }

    #[getter]
    fn satellite_reference_datum(&self) -> Option<bool> {
        self.inner.header.satellite_reference_datum
    }

    #[getter]
    fn satellite_count(&self) -> u8 {
        self.inner.header.satellite_count
    }

    #[getter]
    fn orbit_record_count(&self) -> usize {
        self.inner.orbit.len()
    }

    #[getter]
    fn clock_record_count(&self) -> usize {
        self.inner.clock.len()
    }

    #[getter]
    fn ura_record_count(&self) -> usize {
        self.inner.ura.len()
    }

    /// Encode this message back into an RTCM SSR body.
    ///
    /// A satellite field wider than the message's - five bits for GLONASS,
    /// four for the native QZSS messages 1246..1251 and 1268, six otherwise -
    /// raises `RtcmEncodeError` with the core's reason.
    #[pyo3(signature = (policy=crate::rtcm::PyRtcmPolicy::STRICT))]
    fn encode<'py>(
        &self,
        py: Python<'py>,
        policy: crate::rtcm::PyRtcmPolicy,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let (body, _) = self
            .inner
            .encode_with_policy(policy.into())
            .map_err(|err| to_rtcm_encode_err(py, err))?;
        Ok(PyBytes::new(py, &body))
    }

    /// Bits after the last record, kept as read: the fewer than eight zero
    /// bits that align the body, or under `RtcmPolicy.LENIENT` any other
    /// trailing bits and the bits of a record the body cuts short.
    #[getter]
    fn padding_bits(&self) -> Vec<bool> {
        self.inner.padding_bits.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrMessage(message_number={}, kind={}, satellite_count={})",
            self.inner.message_number,
            ssr_kind_label(self.inner.kind),
            self.inner.header.satellite_count
        )
    }
}

impl PySsrMessage {
    fn from_inner(inner: SsrMessage) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrCorrectionStore")]
/// Mutable RTCM SSR correction store.
pub struct PySsrCorrectionStore {
    inner: SsrCorrectionStore,
}

#[pymethods]
impl PySsrCorrectionStore {
    /// Build an empty SSR correction store.
    #[new]
    #[pyo3(signature = (reference_point=PyOrbitReferencePoint::CENTER_OF_MASS))]
    fn new(reference_point: PyOrbitReferencePoint) -> Self {
        Self {
            inner: SsrCorrectionStore::new().with_reference_point(reference_point.into()),
        }
    }

    /// Ingest one decoded SSR message at a GNSS week and time-of-week.
    #[pyo3(signature = (message, week, tow_s, time_scale=PyTimeScale::GPST))]
    fn ingest_ssr(
        &mut self,
        message: &PySsrMessage,
        week: u32,
        tow_s: f64,
        time_scale: PyTimeScale,
    ) -> PyResult<()> {
        let epoch = gnss_week_tow(time_scale, week, tow_s)?;
        self.inner
            .ingest_ssr(&message.inner, epoch)
            .map_err(to_rtcm_err)
    }

    /// Ingest one decoded RTCM message. Non-SSR RTCM messages are ignored.
    #[pyo3(signature = (message, week, tow_s, time_scale=PyTimeScale::GPST))]
    fn ingest(
        &mut self,
        message: &PyRtcmMessage,
        week: u32,
        tow_s: f64,
        time_scale: PyTimeScale,
    ) -> PyResult<()> {
        let epoch = gnss_week_tow(time_scale, week, tow_s)?;
        self.inner
            .ingest(&message.inner, epoch)
            .map_err(to_rtcm_err)
    }

    fn orbit(&self, satellite_id: &str) -> PyResult<Option<PySsrOrbitCorrection>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.orbit(sat).copied().map(Into::into))
    }

    /// Return the latest SSR clock correction for a satellite.
    fn clock(&self, satellite_id: &str) -> PyResult<Option<PySsrClockCorrection>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.clock(sat).copied().map(Into::into))
    }

    /// Return the latest SSR URA index for a satellite.
    fn ura_index(&self, satellite_id: &str) -> PyResult<Option<u8>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.ura_index(sat))
    }

    /// Return the latest SSR code bias for a satellite signal, metres,
    /// ignoring lifetime, staleness and HAS exclusion.
    ///
    /// Biases are keyed by physical signal. `signal` is a RINEX 3 band and
    /// attribute (`"1C"`, or an observation code such as `"C1C"`) of the
    /// satellite's system, or a raw signal index with the `source` whose
    /// table it belongs to; a raw index the table assigns a physical signal is
    /// looked up as that signal.
    #[pyo3(signature = (satellite_id, signal, source=None))]
    fn code_bias(
        &self,
        satellite_id: &str,
        signal: &Bound<'_, PyAny>,
        source: Option<PySsrSource>,
    ) -> PyResult<Option<f64>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self
            .inner
            .code_bias(sat, ssr_signal_key(sat, signal, source)?))
    }

    /// Return the latest SSR phase bias for a satellite signal, metres,
    /// ignoring lifetime, staleness, HAS exclusion and phase continuity.
    /// `signal` and `source` are read as `code_bias` reads them.
    #[pyo3(signature = (satellite_id, signal, source=None))]
    fn phase_bias(
        &self,
        satellite_id: &str,
        signal: &Bound<'_, PyAny>,
        source: Option<PySsrSource>,
    ) -> PyResult<Option<f64>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self
            .inner
            .phase_bias(sat, ssr_signal_key(sat, signal, source)?))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrCorrectedEphemeris")]
/// Broadcast ephemeris source corrected with RTCM SSR messages.
pub struct PySsrCorrectedEphemeris {
    broadcast: Py<PyBroadcastEphemeris>,
    store: Py<PySsrCorrectionStore>,
    fallback: SsrFallbackPolicy,
    max_staleness_s: Option<f64>,
    ut1_validity: crate::PyValidityMode,
    correction_size_policy: PySsrCorrectionSizePolicy,
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "SsrCorrectionSizePolicy",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PySsrCorrectionSizePolicy {
    STRICT,
    LENIENT,
}

impl From<PySsrCorrectionSizePolicy> for SsrCorrectionSizePolicy {
    fn from(value: PySsrCorrectionSizePolicy) -> Self {
        match value {
            PySsrCorrectionSizePolicy::STRICT => Self::Strict,
            PySsrCorrectionSizePolicy::LENIENT => Self::Lenient,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrCorrectionSize")]
#[derive(Clone, Copy)]
pub struct PySsrCorrectionSize {
    inner: SsrCorrectionSize,
}

impl From<SsrCorrectionSize> for PySsrCorrectionSize {
    fn from(inner: SsrCorrectionSize) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySsrCorrectionSize {
    #[getter]
    fn orbit_m(&self) -> f64 {
        self.inner.orbit_m
    }
    #[getter]
    fn clock_m(&self) -> f64 {
        self.inner.clock_m
    }
    #[getter]
    fn orbit_exceeds_limit(&self) -> bool {
        self.inner.orbit_exceeds_limit()
    }
    #[getter]
    fn clock_exceeds_limit(&self) -> bool {
        self.inner.clock_exceeds_limit()
    }
    #[getter]
    fn exceeds_limit(&self) -> bool {
        self.inner.exceeds_limit()
    }
}

#[pymethods]
impl PySsrCorrectedEphemeris {
    /// Build an SSR-corrected ephemeris source.
    #[new]
    ///
    /// `ut1_validity` is the UT1 policy of a centre-of-mass orbit's
    /// antenna-offset conversion: `ValidityMode.STRICT` refuses an instant
    /// outside the UT1 table, which `corrected_state_checked` raises as
    /// `Ut1OutsideCoverageError`; `PERMISSIVE` uses the long-term UT1 and
    /// reports the departure.
    #[pyo3(signature = (broadcast, store, fallback=None, max_staleness_s=None, ut1_validity=crate::PyValidityMode::STRICT, correction_size_policy=PySsrCorrectionSizePolicy::STRICT))]
    fn new(
        broadcast: Py<PyBroadcastEphemeris>,
        store: Py<PySsrCorrectionStore>,
        fallback: Option<&PySsrFallbackPolicy>,
        max_staleness_s: Option<f64>,
        ut1_validity: crate::PyValidityMode,
        correction_size_policy: PySsrCorrectionSizePolicy,
    ) -> PyResult<Self> {
        if let Some(max_staleness_s) = max_staleness_s {
            if !max_staleness_s.is_finite() || max_staleness_s < 0.0 {
                return Err(PyValueError::new_err(
                    "max_staleness_s must be finite and non-negative",
                ));
            }
        }
        Ok(Self {
            broadcast,
            store,
            fallback: fallback
                .map(|fallback| fallback.inner.clone())
                .unwrap_or_default(),
            max_staleness_s,
            ut1_validity,
            correction_size_policy,
        })
    }

    /// Return corrected ECEF position in metres and clock offset in seconds,
    /// or `None` when the source declines the satellite, a UT1 refusal
    /// included.
    fn position_clock_at_j2000_s(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        t_j2000_s: f64,
    ) -> PyResult<Option<([f64; 3], f64)>> {
        let sat = parse_satellite(satellite_id)?;
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        Ok(source.position_clock_at_j2000_s(sat, t_j2000_s))
    }

    /// The corrected state with the UT1 policy's outcome, as
    /// `(state, ut1_degraded)`: `state` is `(position_m, clock_s)` or `None`
    /// when the source declines the satellite, and `ut1_degraded` the
    /// departure a permissive policy accepted. A strict refusal raises
    /// `Ut1OutsideCoverageError` instead of declining the satellite.
    #[allow(clippy::type_complexity)]
    fn corrected_state_checked(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        t_j2000_s: f64,
    ) -> PyResult<PyCheckedSsrState> {
        let sat = parse_satellite(satellite_id)?;
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        match source.corrected_state_checked(sat, t_j2000_s) {
            Ok(checked) => Ok((
                checked.value,
                checked.degraded.map(crate::degrade_reason_label),
            )),
            Err(sidereon_core::Error::Ut1OutsideCoverage(reason)) => Err(
                crate::ut1_outside_coverage_err("SSR antenna-offset conversion", reason),
            ),
            Err(other) => Err(to_rtcm_err(other)),
        }
    }

    fn corrected_state_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<PyCheckedSsrState> {
        let sat = parse_satellite(satellite_id)?;
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        match source.corrected_state_with_group_delay_checked_selected_query(
            sat,
            &epoch.inner,
            &selection_epoch.inner,
        ) {
            Ok(checked) => Ok((
                checked.value.map(|(position, clock, _)| (position, clock)),
                checked.degraded.map(crate::degrade_reason_label),
            )),
            Err(sidereon_core::Error::Ut1OutsideCoverage(reason)) => Err(
                crate::ut1_outside_coverage_err("SSR antenna-offset conversion", reason),
            ),
            Err(other) => Err(to_rtcm_err(other)),
        }
    }

    fn correction_size_refusal_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<PySsrCorrectionSize>> {
        let sat = parse_satellite(satellite_id)?;
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        Ok(source
            .correction_size_refusal_at_epoch_query(sat, &epoch.inner, &selection_epoch.inner)
            .map(Into::into))
    }

    fn selected_state_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<crate::ephemeris::PyEphemerisQueryState>> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        crate::ephemeris::source_state_at_epoch_query(&source, satellite_id, epoch, selection_epoch)
    }

    fn transmit_epoch_clock_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<(f64, Option<&'static str>)>> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        crate::ephemeris::source_transmit_clock_at_epoch_query(
            &source,
            satellite_id,
            epoch,
            selection_epoch,
        )
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        state_epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<f64> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        crate::ephemeris::source_variance_at_epoch_query(
            &source,
            satellite_id,
            state_epoch,
            selection_epoch,
        )
    }

    fn clock_relativity_for_state_at_epoch_query(
        &self,
        py: Python<'_>,
        satellite_id: &str,
        epoch: &PyExactEpochQuery,
        position_ecef_m: [f64; 3],
    ) -> PyResult<crate::ephemeris::PyClockRelativity> {
        let broadcast = self.broadcast.borrow(py);
        let store = self.store.borrow(py);
        let source = self.source(&broadcast, &store);
        crate::ephemeris::source_clock_relativity_at_epoch_query(
            &source,
            satellite_id,
            epoch,
            position_ecef_m,
        )
    }
}

impl PySsrCorrectedEphemeris {
    fn source<'a>(
        &self,
        broadcast: &'a PyBroadcastEphemeris,
        store: &'a PySsrCorrectionStore,
    ) -> SsrCorrectedEphemeris<'a> {
        let mut source = SsrCorrectedEphemeris::new(&broadcast.inner, &store.inner)
            .with_fallback(self.fallback.clone())
            .with_validity(self.ut1_validity.into())
            .with_correction_size_policy(self.correction_size_policy.into());
        if let Some(max_staleness_s) = self.max_staleness_s {
            source = source.with_staleness(sidereon_core::staleness::StalenessPolicy::seconds(
                max_staleness_s,
            ));
        }
        source
    }
}

#[pyfunction]
#[pyo3(signature = (bytes, form, policy=PySbasPolicy::STRICT))]
/// Decode raw SBAS message bytes in the selected wire form. Under
/// `SbasPolicy.LENIENT` a preamble other than 0x53, 0x9A and 0xC6 is read and
/// kept; a CRC mismatch is refused under both.
fn decode_sbas_block(
    bytes: &[u8],
    form: PySbasWireForm,
    policy: PySbasPolicy,
) -> PyResult<PySbasBlock> {
    SbasBlock::decode_with_policy(bytes, form.into(), policy.into())
        .map(|(block, _)| PySbasBlock::from_inner(block))
        .map_err(to_rtcm_err)
}

#[pyfunction]
/// Decode raw SBAS message bytes under `policy`, returning the block and the
/// departures read.
fn decode_sbas_block_with_policy(
    bytes: &[u8],
    form: PySbasWireForm,
    policy: PySbasPolicy,
) -> PyResult<(PySbasBlock, Vec<PySbasDeparture>)> {
    let (block, read) =
        SbasBlock::decode_with_policy(bytes, form.into(), policy.into()).map_err(to_rtcm_err)?;
    Ok((PySbasBlock::from_inner(block), sbas_departures(read)))
}

#[pyfunction]
#[pyo3(signature = (bytes, form, policy=PySbasPolicy::STRICT))]
/// Decode raw SBAS message bytes in the selected wire form.
fn decode_sbas_message(
    bytes: &[u8],
    form: PySbasWireForm,
    policy: PySbasPolicy,
) -> PyResult<PySbasBlock> {
    decode_sbas_block(bytes, form, policy)
}

#[pyfunction]
#[pyo3(signature = (text, *, policy=PySbasPolicy::STRICT, reference_week=None))]
/// Read an EMS SBAS log: every record, every line read as no record, every
/// record line left unread with its reason, and under `SbasPolicy.LENIENT`
/// every departure. `reference_week` resolves a NovAtel OEM3 10-bit week.
fn parse_sbas_ems_log(
    text: &str,
    policy: PySbasPolicy,
    reference_week: Option<u32>,
) -> PyResult<PySbasLog> {
    core_parse_sbas_ems_log(text, sbas_log_options(policy, reference_week))
        .map(|inner| PySbasLog { inner })
        .map_err(to_rtcm_err)
}

#[pyfunction]
#[pyo3(signature = (text, *, policy=PySbasPolicy::STRICT, reference_week=None))]
/// Read an RTKLIB SBAS log as `parse_sbas_ems_log` reads an EMS log.
fn parse_sbas_rtklib_log(
    text: &str,
    policy: PySbasPolicy,
    reference_week: Option<u32>,
) -> PyResult<PySbasLog> {
    core_parse_sbas_rtklib_log(text, sbas_log_options(policy, reference_week))
        .map(|inner| PySbasLog { inner })
        .map_err(to_rtcm_err)
}

#[pyfunction]
/// Parse SBAS EMS log lines into timestamped raw message blocks.
fn parse_sbas_ems_lines(text: &str) -> PyResult<Vec<PySbasLogBlock>> {
    core_parse_sbas_ems_lines(text)
        .map(|blocks| blocks.into_iter().map(Into::into).collect())
        .map_err(to_rtcm_err)
}

#[pyfunction]
/// Parse RTKLIB-style SBAS log lines into timestamped raw message blocks.
fn parse_sbas_rtklib_lines(text: &str) -> PyResult<Vec<PySbasLogBlock>> {
    core_parse_sbas_rtklib_lines(text)
        .map(|blocks| blocks.into_iter().map(Into::into).collect())
        .map_err(to_rtcm_err)
}

#[pyfunction]
/// Convert an SBAS PRN to the package satellite token, if it is valid.
fn sbas_prn_to_satellite_id(prn: u16) -> Option<String> {
    sbas_prn_to_sat(prn).map(py_satellite)
}

#[pyfunction]
/// Convert an SBAS satellite token to its PRN, if it is valid.
fn satellite_id_to_sbas_prn(satellite_id: &str) -> PyResult<Option<u16>> {
    let sat = parse_satellite(satellite_id)?;
    Ok(sat_to_sbas_prn(sat))
}

#[pyfunction]
#[pyo3(signature = (body, policy=crate::rtcm::PyRtcmPolicy::STRICT))]
/// Decode a raw RTCM SSR message body. Under `RtcmPolicy.LENIENT` a body that
/// ends before the records its header counts is read up to its last complete
/// record.
fn decode_ssr_message(body: &[u8], policy: crate::rtcm::PyRtcmPolicy) -> PyResult<PySsrMessage> {
    SsrMessage::decode_with_policy(body, policy.into())
        .map(|(inner, _)| PySsrMessage::from_inner(inner))
        .map_err(to_rtcm_err)
}

#[pyfunction]
#[pyo3(signature = (body, policy=crate::rtcm::PyRtcmPolicy::STRICT))]
/// Decode a raw RTCM SSR message body.
fn decode_ssr(body: &[u8], policy: crate::rtcm::PyRtcmPolicy) -> PyResult<PySsrMessage> {
    decode_ssr_message(body, policy)
}

/// A decoded RTCM message the SSR store refused to ingest.
#[pyclass(module = "sidereon._sidereon", name = "SsrIngestRefusal")]
#[derive(Clone)]
pub struct PySsrIngestRefusal {
    message_number: u16,
    error: String,
}

#[pymethods]
impl PySsrIngestRefusal {
    /// The RTCM message number.
    #[getter]
    fn message_number(&self) -> u16 {
        self.message_number
    }

    /// Why the store refused it, as the core states it.
    #[getter]
    fn error(&self) -> String {
        self.error.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrIngestRefusal(message_number={}, error={:?})",
            self.message_number, self.error
        )
    }
}

/// An SSR correction store built from every readable frame of an RTCM byte
/// stream, with an account of everything that was not read or not applied.
#[pyclass(module = "sidereon._sidereon", name = "SsrRtcmIngest")]
pub struct PySsrRtcmIngest {
    store: Py<PySsrCorrectionStore>,
    diagnostics: sidereon_core::rtcm::StreamDiagnostics,
    trailing_partial_frame_len: usize,
    ingest_refusals: Vec<PySsrIngestRefusal>,
}

#[pymethods]
impl PySsrRtcmIngest {
    /// The store holding every correction that was read and applied.
    #[getter]
    fn store(&self, py: Python<'_>) -> Py<PySsrCorrectionStore> {
        self.store.clone_ref(py)
    }

    /// Bytes passed over while resynchronizing, CRC-24Q failures, frames
    /// whose body did not decode, and departures read under the lenient
    /// policy.
    #[getter]
    fn diagnostics(&self) -> crate::rtcm::PyRtcmStreamDiagnostics {
        crate::rtcm::PyRtcmStreamDiagnostics::from_inner(self.diagnostics.clone())
    }

    /// Bytes from the last preamble whose declared frame ran past the end of
    /// the input.
    #[getter]
    fn trailing_partial_frame_len(&self) -> usize {
        self.trailing_partial_frame_len
    }

    /// Messages that decoded but that the store refused to ingest.
    #[getter]
    fn ingest_refusals(&self) -> Vec<PySsrIngestRefusal> {
        self.ingest_refusals.clone()
    }

    /// True when every byte was read into a message that was applied, with
    /// no departure from the format.
    #[getter]
    fn is_complete(&self) -> bool {
        self.diagnostics.is_clean()
            && self.trailing_partial_frame_len == 0
            && self.ingest_refusals.is_empty()
    }

    fn __repr__(&self) -> String {
        format!(
            "SsrRtcmIngest(resync_bytes={}, skipped_frames={}, ingest_refusals={}, trailing_partial_frame_len={})",
            self.diagnostics.resync_bytes,
            self.diagnostics.skipped_frames.len(),
            self.ingest_refusals.len(),
            self.trailing_partial_frame_len
        )
    }
}

#[pyfunction]
#[pyo3(signature = (bytes, week, tow_s, time_scale=PyTimeScale::GPST))]
/// Build an SSR correction store from every readable frame of framed RTCM
/// bytes, read under `RtcmPolicy.LENIENT`, reporting in the returned
/// `SsrRtcmIngest` the bytes outside frames, CRC-24Q failures, frames that
/// did not decode, a trailing partial frame and the messages the store
/// refused. `ssr_store_from_rtcm_strict` refuses all of them instead.
fn ssr_store_from_rtcm(
    py: Python<'_>,
    bytes: &[u8],
    week: u32,
    tow_s: f64,
    time_scale: PyTimeScale,
) -> PyResult<PySsrRtcmIngest> {
    let epoch = gnss_week_tow(time_scale, week, tow_s)?;
    let ingest = sidereon::ssr_store_from_rtcm(bytes, epoch);
    Ok(PySsrRtcmIngest {
        store: Py::new(
            py,
            PySsrCorrectionStore {
                inner: ingest.store,
            },
        )?,
        diagnostics: ingest.diagnostics,
        trailing_partial_frame_len: ingest.trailing_partial_frame_len,
        ingest_refusals: ingest
            .ingest_refusals
            .into_iter()
            .map(|refusal| PySsrIngestRefusal {
                message_number: refusal.message_number,
                error: refusal.error.to_string(),
            })
            .collect(),
    })
}

#[pyfunction]
#[pyo3(signature = (bytes, week, tow_s, time_scale=PyTimeScale::GPST))]
/// Build an SSR correction store from framed RTCM bytes, refusing with
/// `RtcmParseError` anything it cannot read under `RtcmPolicy.STRICT` and
/// apply in full: bytes outside a CRC-valid frame, a CRC-24Q failure, a
/// trailing partial frame, a frame that does not decode and a message the
/// store refuses.
fn ssr_store_from_rtcm_strict(
    bytes: &[u8],
    week: u32,
    tow_s: f64,
    time_scale: PyTimeScale,
) -> PyResult<PySsrCorrectionStore> {
    let epoch = gnss_week_tow(time_scale, week, tow_s)?;
    sidereon::ssr_store_from_rtcm_strict(bytes, epoch)
        .map(|inner| PySsrCorrectionStore { inner })
        .map_err(to_rtcm_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySbasWireForm>()?;
    m.add_class::<PySbasSolveMode>()?;
    m.add_class::<PySbasMessageKind>()?;
    m.add_class::<PySbasMessage>()?;
    m.add_class::<PySbasDoNotUse>()?;
    m.add_class::<PySbasPrnMask>()?;
    m.add_class::<PySbasFastCorrections>()?;
    m.add_class::<PySbasIntegrity>()?;
    m.add_class::<PySbasFastDegradation>()?;
    m.add_class::<PySbasGeoNav>()?;
    m.add_class::<PySbasNetworkTime>()?;
    m.add_class::<PySbasGeoAlmanac>()?;
    m.add_class::<PySbasMixedCorrections>()?;
    m.add_class::<PySbasMixedFastCorrections>()?;
    m.add_class::<PySbasLongTermCorrections>()?;
    m.add_class::<PySbasLongTermHalf>()?;
    m.add_class::<PySbasLongTermRecord>()?;
    m.add_class::<PySbasIgpMask>()?;
    m.add_class::<PySbasIonoDelays>()?;
    m.add_class::<PySbasIgpDelay>()?;
    m.add_class::<PySbasUnsupported>()?;
    m.add_class::<PySbasBlock>()?;
    m.add_class::<PySbasLogBlock>()?;
    m.add_class::<PySbasFastCorrection>()?;
    m.add_class::<PySbasLongTermCorrection>()?;
    m.add_class::<PySbasIgp>()?;
    m.add_class::<PySbasIonoGrid>()?;
    m.add_class::<PySbasGeoState>()?;
    m.add_class::<PySbasCorrectionStore>()?;
    m.add_class::<PySbasCorrectedEphemeris>()?;
    m.add_class::<PySsrKind>()?;
    m.add_class::<PySsrSource>()?;
    m.add_class::<PySsrSolution>()?;
    m.add_class::<PyOrbitReferencePoint>()?;
    m.add_class::<PyMissingCorrectionAction>()?;
    m.add_class::<PySsrFallbackPolicy>()?;
    m.add_class::<PySsrCorrectionSizePolicy>()?;
    m.add_class::<PySsrCorrectionSize>()?;
    m.add_class::<PySsrOrbitCorrection>()?;
    m.add_class::<PySsrHighRateClock>()?;
    m.add_class::<PySsrClockCorrection>()?;
    m.add_class::<PySsrMessage>()?;
    m.add_class::<PySsrCorrectionStore>()?;
    m.add_class::<PySsrCorrectedEphemeris>()?;
    m.add_function(wrap_pyfunction!(decode_sbas_block, m)?)?;
    m.add_function(wrap_pyfunction!(decode_sbas_message, m)?)?;
    m.add_function(wrap_pyfunction!(parse_sbas_ems_lines, m)?)?;
    m.add_function(wrap_pyfunction!(parse_sbas_rtklib_lines, m)?)?;
    m.add_function(wrap_pyfunction!(sbas_prn_to_satellite_id, m)?)?;
    m.add_function(wrap_pyfunction!(satellite_id_to_sbas_prn, m)?)?;
    m.add_function(wrap_pyfunction!(decode_ssr_message, m)?)?;
    m.add_function(wrap_pyfunction!(decode_ssr, m)?)?;
    m.add_function(wrap_pyfunction!(ssr_store_from_rtcm, m)?)?;
    m.add_function(wrap_pyfunction!(ssr_store_from_rtcm_strict, m)?)?;
    m.add_function(wrap_pyfunction!(decode_sbas_block_with_policy, m)?)?;
    m.add_function(wrap_pyfunction!(parse_sbas_ems_log, m)?)?;
    m.add_function(wrap_pyfunction!(parse_sbas_rtklib_log, m)?)?;
    m.add_class::<PySbasPolicy>()?;
    m.add_class::<PySbasDeparture>()?;
    m.add_class::<PySbasRefusedLine>()?;
    m.add_class::<PySbasLog>()?;
    m.add_class::<PySbasUnavailableIgp>()?;
    m.add_class::<PySsrIngestRefusal>()?;
    m.add_class::<PySsrRtcmIngest>()?;
    Ok(())
}
