//! SBAS and RTCM SSR correction bindings.
//!
//! Bytes decode into core message structs, stores ingest those structs, and
//! corrected ephemeris wrappers call the core corrected-source implementations.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use sidereon::ephemeris::EphemerisSource;
use sidereon_core::astro::time::model::GnssWeekTow;
use sidereon_core::frame::Wgs84Geodetic;
use sidereon_core::rtcm::{SsrKind, SsrMessage};
use sidereon_core::sbas::{
    parse_ems_lines as core_parse_sbas_ems_lines,
    parse_rtklib_lines as core_parse_sbas_rtklib_lines, sat_to_sbas_prn, sbas_prn_to_sat,
    SbasBlock, SbasCorrectedEphemeris, SbasCorrectionStore, SbasDoNotUse, SbasFastCorrection,
    SbasFastCorrections, SbasFastDegradation, SbasGeoAlmanac, SbasGeoNav, SbasGeoState, SbasIgp,
    SbasIgpDelay, SbasIgpMask, SbasIntegrity, SbasIonoDelays, SbasIonoGrid, SbasLogBlock,
    SbasLongTermCorrection, SbasLongTermCorrections, SbasLongTermHalf, SbasLongTermRecord,
    SbasMessage, SbasMixedCorrections, SbasMixedFastCorrections, SbasNetworkTime, SbasPrnMask,
    SbasSolveMode, SbasUnsupported, SbasWireForm, SpareBits,
};
use sidereon_core::ssr::{
    MissingCorrectionAction, OrbitReferencePoint, RegionalPolicy, SsrClockCorrection,
    SsrCorrectedEphemeris, SsrCorrectionStore, SsrFallbackPolicy, SsrHighRateClock,
    SsrOrbitCorrection, SsrSolution, SsrSource,
};
use sidereon_core::GnssSatelliteId;

use crate::frames::PyTimeScale;
use crate::marshal::PyGnssSystem;
use crate::rinex::PyBroadcastEphemeris;
use crate::rtcm::PyRtcmMessage;
use crate::RtcmParseError;

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
        SsrKind::Vtec => "vtec",
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

    /// UDREI values for each correction slot.
    #[getter]
    fn udrei(&self) -> Vec<u8> {
        self.inner.udrei.to_vec()
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

    /// Issue of data for fast corrections, one per block.
    #[getter]
    fn iodf(&self) -> Vec<u8> {
        self.inner.iodf.to_vec()
    }

    /// UDREI values for monitored slots.
    #[getter]
    fn udrei(&self) -> Vec<u8> {
        self.inner.udrei.to_vec()
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

    /// Degradation indicators for monitored slots.
    #[getter]
    fn ai(&self) -> Vec<u8> {
        self.inner.ai.to_vec()
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

    /// UDREI values for each correction slot.
    #[getter]
    fn udrei(&self) -> Vec<u8> {
        self.inner.udrei.to_vec()
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

    /// Encode the block back to its original wire form.
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
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

    fn decode(&self) -> PyResult<PySbasBlock> {
        SbasBlock::decode(&self.inner.bytes, self.inner.form)
            .map(PySbasBlock::from_inner)
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
    inner: SbasCorrectionStore,
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
    VTEC,
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
            SsrKind::Vtec => Self::VTEC,
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
            Self::VTEC => "vtec",
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
            Self::VTEC => "SsrKind.VTEC",
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
}

impl From<SsrSource> for PySsrSource {
    fn from(source: SsrSource) -> Self {
        match source {
            SsrSource::RtcmSsr => Self::RTCM_SSR,
            SsrSource::GalileoHas => Self::GALILEO_HAS,
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
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::RTCM_SSR => "SsrSource.RTCM_SSR",
            Self::GALILEO_HAS => "SsrSource.GALILEO_HAS",
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

    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
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

    /// Return the latest SSR code bias for a satellite signal.
    fn code_bias(&self, satellite_id: &str, signal: u8) -> PyResult<Option<f64>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.code_bias(sat, signal))
    }

    /// Return the latest SSR phase bias for a satellite signal.
    fn phase_bias(&self, satellite_id: &str, signal: u8) -> PyResult<Option<f64>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.phase_bias(sat, signal))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrCorrectedEphemeris")]
/// Broadcast ephemeris source corrected with RTCM SSR messages.
pub struct PySsrCorrectedEphemeris {
    broadcast: Py<PyBroadcastEphemeris>,
    store: Py<PySsrCorrectionStore>,
    fallback: SsrFallbackPolicy,
    max_staleness_s: Option<f64>,
}

#[pymethods]
impl PySsrCorrectedEphemeris {
    /// Build an SSR-corrected ephemeris source.
    #[new]
    #[pyo3(signature = (broadcast, store, fallback=None, max_staleness_s=None))]
    fn new(
        broadcast: Py<PyBroadcastEphemeris>,
        store: Py<PySsrCorrectionStore>,
        fallback: Option<&PySsrFallbackPolicy>,
        max_staleness_s: Option<f64>,
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
        let mut source = SsrCorrectedEphemeris::new(&broadcast.inner, &store.inner)
            .with_fallback(self.fallback.clone());
        if let Some(max_staleness_s) = self.max_staleness_s {
            source = source.with_staleness(sidereon_core::staleness::StalenessPolicy::seconds(
                max_staleness_s,
            ));
        }
        Ok(source.position_clock_at_j2000_s(sat, t_j2000_s))
    }
}

#[pyfunction]
/// Decode raw SBAS message bytes in the selected wire form.
fn decode_sbas_block(bytes: &[u8], form: PySbasWireForm) -> PyResult<PySbasBlock> {
    SbasBlock::decode(bytes, form.into())
        .map(PySbasBlock::from_inner)
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
/// Decode a raw RTCM SSR message body.
fn decode_ssr_message(body: &[u8]) -> PyResult<PySsrMessage> {
    SsrMessage::decode(body)
        .map(PySsrMessage::from_inner)
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
    m.add_class::<PySsrOrbitCorrection>()?;
    m.add_class::<PySsrHighRateClock>()?;
    m.add_class::<PySsrClockCorrection>()?;
    m.add_class::<PySsrMessage>()?;
    m.add_class::<PySsrCorrectionStore>()?;
    m.add_class::<PySsrCorrectedEphemeris>()?;
    m.add_function(wrap_pyfunction!(decode_sbas_block, m)?)?;
    m.add_function(wrap_pyfunction!(parse_sbas_ems_lines, m)?)?;
    m.add_function(wrap_pyfunction!(parse_sbas_rtklib_lines, m)?)?;
    m.add_function(wrap_pyfunction!(sbas_prn_to_satellite_id, m)?)?;
    m.add_function(wrap_pyfunction!(satellite_id_to_sbas_prn, m)?)?;
    m.add_function(wrap_pyfunction!(decode_ssr_message, m)?)?;
    Ok(())
}
