//! RINEX navigation binding: broadcast-store loading and typed NAV records.
//!
//! This module is a PyO3 surface over `sidereon-core`'s RINEX NAV parser. It
//! copies parsed records into Python value objects and exposes the broadcast
//! ephemeris store exactly as the core builds it. It contains no orbit or clock
//! modeling logic.

use std::collections::BTreeMap;
use std::path::PathBuf;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyModule};

use sidereon_core::ephemeris::{
    cnav_ura_ned_m as core_cnav_ura_ned_m, cnav_ura_nominal_m as core_cnav_ura_nominal_m,
    is_beidou_geo, satellite_state, satellite_state_cnav,
    BroadcastEphemeris as CoreBroadcastEphemeris, BroadcastGroupDelayTerm, BroadcastGroupDelays,
    BroadcastRecord, ClockPolynomial, CnavParameters, CnavRates, CnavSignal, GlonassRecord,
    IonoCorrections, KeplerianElements, KlobucharAlphaBeta, NavMessage, NavMessagePreference,
    SatelliteState,
};
use sidereon_core::rinex::crinex::{
    decode as core_decode_crinex, decode_to as core_decode_crinex_to,
    encode_crinex as core_encode_crinex,
};
use sidereon_core::rinex::nav::{
    encode_nav, parse_glonass, parse_iono_corrections, parse_leap_seconds, parse_nav,
    parse_nav_lenient, NavParse, NavParseError, SkippedNavBlock,
};
use sidereon_core::rinex::observations::{
    carrier_phase_rows, observation_values, pseudoranges, CarrierPhaseRow as CoreCarrierPhaseRow,
    CorrectionUnavailable as CoreCorrectionUnavailable,
    ObsDowngradeChange as CoreObsDowngradeChange, ObsEpoch as CoreObsEpoch,
    ObsEpochTime as CoreObsEpochTime, ObsHeader as CoreObsHeader,
    ObsHeaderTimeline as CoreObsHeaderTimeline, ObsLeapSeconds as CoreObsLeapSeconds,
    ObsPhaseShift as CoreObsPhaseShift, ObsScaleFactor as CoreObsScaleFactor,
    ObsValue as CoreObsValue, ObservationFilter as CoreObservationFilter,
    ObservationKind as CoreObservationKind, ObservationValueRow as CoreObservationValueRow,
    RinexObs as CoreRinexObs, RinexObsWriteError as CoreRinexObsWriteError,
    SignalPolicy as CoreSignalPolicy, CYCLE_SLIP_FLAG,
};
use sidereon_core::rinex::qc::{
    lint_nav_text as core_lint_nav_text, lint_obs_text as core_lint_obs_text,
    repair_nav_text as core_repair_nav_text, repair_obs_text as core_repair_obs_text,
    Finding as CoreRinexFinding, FindingRef as CoreFindingRef, LintReport as CoreLintReport,
    NavRepair as CoreNavRepair, ObsRepair as CoreObsRepair, RepairAction as CoreRepairAction,
    RepairOptions as CoreRepairOptions, Severity as CoreRinexSeverity,
};

use crate::exact_time::PyExactEpochQuery;
use sidereon_core::{GnssSatelliteId, GnssSystem};

use crate::frames::PyTimeScale;
use crate::marshal::{option_py_or_default, PyGnssSystem};
use crate::{np_array, CrinexParseError, RinexNavParseError};
use crate::{RinexClockParseError, RinexObsParseError, RinexObsWriteError};

fn to_nav_err(err: NavParseError) -> PyErr {
    RinexNavParseError::new_err(err.to_string())
}

#[derive(Clone, Copy)]
pub(crate) enum RinexTextKind {
    Nav,
    Obs,
    Clock,
    Crinex,
}

fn utf8_err(kind: RinexTextKind, message: String) -> PyErr {
    match kind {
        RinexTextKind::Nav => RinexNavParseError::new_err(message),
        RinexTextKind::Obs => RinexObsParseError::new_err(message),
        RinexTextKind::Clock => RinexClockParseError::new_err(message),
        RinexTextKind::Crinex => CrinexParseError::new_err(message),
    }
}

fn utf8_text(bytes: &[u8], source: &str, kind: RinexTextKind) -> PyResult<String> {
    std::str::from_utf8(bytes)
        .map(str::to_owned)
        .map_err(|err| utf8_err(kind, format!("{source} is not UTF-8 text: {err}")))
}

pub(crate) fn text_from_source(
    source: &Bound<'_, PyAny>,
    function_name: &str,
    source_label: &str,
    kind: RinexTextKind,
) -> PyResult<String> {
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        return utf8_text(bytes.as_bytes(), source_label, kind);
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the bytearray is copied into an owned String synchronously, and
        // no Python code runs before the copy completes.
        return utf8_text(unsafe { buf.as_bytes() }, source_label, kind);
    }
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{function_name} expects bytes, bytearray, or a path (str/os.PathLike)"
        ))
    })?;
    std::fs::read_to_string(&path).map_err(Into::into)
}

fn parse_store_text(text: &str) -> PyResult<PyBroadcastEphemeris> {
    let inner = CoreBroadcastEphemeris::from_nav(text).map_err(to_nav_err)?;
    Ok(PyBroadcastEphemeris {
        inner,
        leap_seconds: parse_leap_seconds(text).map_err(to_nav_err)?,
    })
}

fn satellite_token(sat: sidereon_core::GnssSatelliteId) -> String {
    sat.to_string()
}

fn to_obs_err<E: std::fmt::Display>(err: E) -> PyErr {
    RinexObsParseError::new_err(err.to_string())
}

fn to_crinex_err<E: std::fmt::Display>(err: E) -> PyErr {
    CrinexParseError::new_err(err.to_string())
}

fn parse_obs_text(text: &str) -> PyResult<PyRinexObs> {
    Ok(PyRinexObs {
        inner: CoreRinexObs::parse(text).map_err(to_obs_err)?,
    })
}

fn nan_if_missing(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NAN)
}

fn u8_nan_if_missing(value: Option<u8>) -> f64 {
    value.map(f64::from).unwrap_or(f64::NAN)
}

fn check_epoch_index(obs: &CoreRinexObs, epoch_index: usize) -> PyResult<&CoreObsEpoch> {
    obs.epochs().get(epoch_index).ok_or_else(|| {
        PyValueError::new_err(format!(
            "epoch_index {epoch_index} out of range for {} epochs",
            obs.epochs().len()
        ))
    })
}

fn filter_from_optional(
    py: Python<'_>,
    filter: Option<Py<PyObservationFilter>>,
) -> CoreObservationFilter {
    option_py_or_default(
        py,
        filter.as_ref(),
        |filter| filter.inner.clone(),
        CoreObservationFilter::all,
    )
}

fn policy_from_optional(
    py: Python<'_>,
    obs: &CoreRinexObs,
    policy: Option<Py<PySignalPolicy>>,
) -> PyResult<CoreSignalPolicy> {
    match policy {
        Some(policy) => Ok(policy.borrow(py).inner.clone()),
        None => CoreSignalPolicy::default_for(obs.header().version).map_err(to_obs_err),
    }
}

fn obs_entries_from_py(
    entries: Option<Vec<(PyGnssSystem, Vec<String>)>>,
) -> BTreeMap<GnssSystem, Vec<String>> {
    entries
        .unwrap_or_default()
        .into_iter()
        .map(|(system, codes)| (system.into(), codes))
        .collect()
}

/// Which supported RINEX NAV message a broadcast record carries.
#[pyclass(module = "sidereon._sidereon", name = "NavMessage", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyNavMessage {
    /// GPS legacy LNAV.
    GPS_LNAV,
    /// GPS CNAV.
    GPS_CNAV,
    /// GPS CNAV-2.
    GPS_CNAV2,
    /// QZSS legacy LNAV.
    QZSS_LNAV,
    /// QZSS CNAV.
    QZSS_CNAV,
    /// QZSS CNAV-2.
    QZSS_CNAV2,
    /// Galileo I/NAV.
    GALILEO_INAV,
    /// Galileo F/NAV.
    GALILEO_FNAV,
    /// BeiDou D1.
    BEIDOU_D1,
    /// BeiDou D2.
    BEIDOU_D2,
    /// NavIC LNAV.
    NAVIC_LNAV,
    /// A Galileo record whose data-source word names no message (RINEX 3.05
    /// Table A8), used with the BGD E5b/E1 as RTKLIB's default Galileo
    /// selection uses it.
    GALILEO_UNCLASSIFIED,
}

impl From<NavMessage> for PyNavMessage {
    fn from(message: NavMessage) -> Self {
        match message {
            NavMessage::GpsLnav => Self::GPS_LNAV,
            NavMessage::GpsCnav => Self::GPS_CNAV,
            NavMessage::GpsCnav2 => Self::GPS_CNAV2,
            NavMessage::QzssLnav => Self::QZSS_LNAV,
            NavMessage::QzssCnav => Self::QZSS_CNAV,
            NavMessage::QzssCnav2 => Self::QZSS_CNAV2,
            NavMessage::GalileoInav => Self::GALILEO_INAV,
            NavMessage::GalileoFnav => Self::GALILEO_FNAV,
            NavMessage::BeidouD1 => Self::BEIDOU_D1,
            NavMessage::BeidouD2 => Self::BEIDOU_D2,
            NavMessage::NavicLnav => Self::NAVIC_LNAV,
            NavMessage::GalileoUnclassified => Self::GALILEO_UNCLASSIFIED,
        }
    }
}

impl From<PyNavMessage> for NavMessage {
    fn from(message: PyNavMessage) -> Self {
        match message {
            PyNavMessage::GPS_LNAV => Self::GpsLnav,
            PyNavMessage::GPS_CNAV => Self::GpsCnav,
            PyNavMessage::GPS_CNAV2 => Self::GpsCnav2,
            PyNavMessage::QZSS_LNAV => Self::QzssLnav,
            PyNavMessage::QZSS_CNAV => Self::QzssCnav,
            PyNavMessage::QZSS_CNAV2 => Self::QzssCnav2,
            PyNavMessage::GALILEO_INAV => Self::GalileoInav,
            PyNavMessage::GALILEO_FNAV => Self::GalileoFnav,
            PyNavMessage::BEIDOU_D1 => Self::BeidouD1,
            PyNavMessage::BEIDOU_D2 => Self::BeidouD2,
            PyNavMessage::NAVIC_LNAV => Self::NavicLnav,
            PyNavMessage::GALILEO_UNCLASSIFIED => Self::GalileoUnclassified,
        }
    }
}

#[pymethods]
impl PyNavMessage {
    /// Stable lowercase selector for this NAV message.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::GPS_LNAV => "gps_lnav",
            Self::GPS_CNAV => "gps_cnav",
            Self::GPS_CNAV2 => "gps_cnav2",
            Self::QZSS_LNAV => "qzss_lnav",
            Self::QZSS_CNAV => "qzss_cnav",
            Self::QZSS_CNAV2 => "qzss_cnav2",
            Self::GALILEO_INAV => "galileo_inav",
            Self::GALILEO_FNAV => "galileo_fnav",
            Self::BEIDOU_D1 => "beidou_d1",
            Self::BEIDOU_D2 => "beidou_d2",
            Self::NAVIC_LNAV => "navic_lnav",
            Self::GALILEO_UNCLASSIFIED => "galileo_unclassified",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::GPS_LNAV => "NavMessage.GPS_LNAV",
            Self::GPS_CNAV => "NavMessage.GPS_CNAV",
            Self::GPS_CNAV2 => "NavMessage.GPS_CNAV2",
            Self::QZSS_LNAV => "NavMessage.QZSS_LNAV",
            Self::QZSS_CNAV => "NavMessage.QZSS_CNAV",
            Self::QZSS_CNAV2 => "NavMessage.QZSS_CNAV2",
            Self::GALILEO_INAV => "NavMessage.GALILEO_INAV",
            Self::GALILEO_FNAV => "NavMessage.GALILEO_FNAV",
            Self::BEIDOU_D1 => "NavMessage.BEIDOU_D1",
            Self::BEIDOU_D2 => "NavMessage.BEIDOU_D2",
            Self::NAVIC_LNAV => "NavMessage.NAVIC_LNAV",
            Self::GALILEO_UNCLASSIFIED => "NavMessage.GALILEO_UNCLASSIFIED",
        }
    }
}

/// GPS/QZSS signal selector for CNAV-family group-delay correction.
#[pyclass(module = "sidereon._sidereon", name = "CnavSignal", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyCnavSignal {
    L1_CA,
    L2C,
    L5_I5,
    L5_Q5,
    L1C_PILOT,
    L1C_DATA,
}

impl From<PyCnavSignal> for CnavSignal {
    fn from(signal: PyCnavSignal) -> Self {
        match signal {
            PyCnavSignal::L1_CA => Self::L1Ca,
            PyCnavSignal::L2C => Self::L2C,
            PyCnavSignal::L5_I5 => Self::L5I5,
            PyCnavSignal::L5_Q5 => Self::L5Q5,
            PyCnavSignal::L1C_PILOT => Self::L1Cp,
            PyCnavSignal::L1C_DATA => Self::L1Cd,
        }
    }
}

#[pymethods]
impl PyCnavSignal {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::L1_CA => "l1_ca",
            Self::L2C => "l2c",
            Self::L5_I5 => "l5_i5",
            Self::L5_Q5 => "l5_q5",
            Self::L1C_PILOT => "l1c_pilot",
            Self::L1C_DATA => "l1c_data",
        }
    }
}

/// Broadcast group-delay term selector.
#[pyclass(
    module = "sidereon._sidereon",
    name = "BroadcastGroupDelayTerm",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyBroadcastGroupDelayTerm {
    GPS_TGD,
    GALILEO_BGD_E5A_E1,
    GALILEO_BGD_E5B_E1,
    BEIDOU_TGD1,
    BEIDOU_TGD2,
    CNAV_ISC_L1_CA,
    CNAV_ISC_L2C,
    CNAV_ISC_L5_I5,
    CNAV_ISC_L5_Q5,
    CNAV_ISC_L1C_DATA,
    CNAV_ISC_L1C_PILOT,
}

impl From<PyBroadcastGroupDelayTerm> for BroadcastGroupDelayTerm {
    fn from(term: PyBroadcastGroupDelayTerm) -> Self {
        match term {
            PyBroadcastGroupDelayTerm::GPS_TGD => Self::GpsTgd,
            PyBroadcastGroupDelayTerm::GALILEO_BGD_E5A_E1 => Self::GalileoBgdE5aE1,
            PyBroadcastGroupDelayTerm::GALILEO_BGD_E5B_E1 => Self::GalileoBgdE5bE1,
            PyBroadcastGroupDelayTerm::BEIDOU_TGD1 => Self::BeidouTgd1,
            PyBroadcastGroupDelayTerm::BEIDOU_TGD2 => Self::BeidouTgd2,
            PyBroadcastGroupDelayTerm::CNAV_ISC_L1_CA => Self::CnavIscL1Ca,
            PyBroadcastGroupDelayTerm::CNAV_ISC_L2C => Self::CnavIscL2C,
            PyBroadcastGroupDelayTerm::CNAV_ISC_L5_I5 => Self::CnavIscL5I5,
            PyBroadcastGroupDelayTerm::CNAV_ISC_L5_Q5 => Self::CnavIscL5Q5,
            PyBroadcastGroupDelayTerm::CNAV_ISC_L1C_DATA => Self::CnavIscL1Cd,
            PyBroadcastGroupDelayTerm::CNAV_ISC_L1C_PILOT => Self::CnavIscL1Cp,
        }
    }
}

#[pymethods]
impl PyBroadcastGroupDelayTerm {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::GPS_TGD => "gps_tgd",
            Self::GALILEO_BGD_E5A_E1 => "galileo_bgd_e5a_e1",
            Self::GALILEO_BGD_E5B_E1 => "galileo_bgd_e5b_e1",
            Self::BEIDOU_TGD1 => "beidou_tgd1",
            Self::BEIDOU_TGD2 => "beidou_tgd2",
            Self::CNAV_ISC_L1_CA => "cnav_isc_l1_ca",
            Self::CNAV_ISC_L2C => "cnav_isc_l2c",
            Self::CNAV_ISC_L5_I5 => "cnav_isc_l5_i5",
            Self::CNAV_ISC_L5_Q5 => "cnav_isc_l5_q5",
            Self::CNAV_ISC_L1C_DATA => "cnav_isc_l1c_data",
            Self::CNAV_ISC_L1C_PILOT => "cnav_isc_l1c_pilot",
        }
    }
}

/// Per-signal broadcast group delays from one NAV record.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastGroupDelays")]
#[derive(Clone, Copy)]
pub struct PyBroadcastGroupDelays {
    inner: BroadcastGroupDelays,
}

impl From<BroadcastGroupDelays> for PyBroadcastGroupDelays {
    fn from(inner: BroadcastGroupDelays) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBroadcastGroupDelays {
    #[getter]
    fn gps_tgd_s(&self) -> Option<f64> {
        self.inner.gps_tgd_s
    }

    #[getter]
    fn galileo_bgd_e5a_e1_s(&self) -> Option<f64> {
        self.inner.galileo_bgd_e5a_e1_s
    }

    #[getter]
    fn galileo_bgd_e5b_e1_s(&self) -> Option<f64> {
        self.inner.galileo_bgd_e5b_e1_s
    }

    #[getter]
    fn beidou_tgd1_s(&self) -> Option<f64> {
        self.inner.beidou_tgd1_s
    }

    #[getter]
    fn beidou_tgd2_s(&self) -> Option<f64> {
        self.inner.beidou_tgd2_s
    }

    #[getter]
    fn cnav_isc_l1ca_s(&self) -> Option<f64> {
        self.inner.cnav_isc_l1ca_s
    }

    #[getter]
    fn cnav_isc_l2c_s(&self) -> Option<f64> {
        self.inner.cnav_isc_l2c_s
    }

    #[getter]
    fn cnav_isc_l5i5_s(&self) -> Option<f64> {
        self.inner.cnav_isc_l5i5_s
    }

    #[getter]
    fn cnav_isc_l5q5_s(&self) -> Option<f64> {
        self.inner.cnav_isc_l5q5_s
    }

    #[getter]
    fn cnav_isc_l1cd_s(&self) -> Option<f64> {
        self.inner.cnav_isc_l1cd_s
    }

    #[getter]
    fn cnav_isc_l1cp_s(&self) -> Option<f64> {
        self.inner.cnav_isc_l1cp_s
    }

    fn get(&self, term: PyBroadcastGroupDelayTerm) -> Option<f64> {
        self.inner.get(term.into())
    }

    fn cnav_single_frequency_correction_s(&self, signal: PyCnavSignal) -> Option<f64> {
        self.inner.cnav_single_frequency_correction_s(signal.into())
    }

    fn __repr__(&self) -> &'static str {
        "BroadcastGroupDelays(...)"
    }
}

/// CNAV/CNAV-2 parameters carried by a NAV record.
#[pyclass(module = "sidereon._sidereon", name = "CnavParameters")]
#[derive(Clone, Copy)]
pub struct PyCnavParameters {
    inner: CnavParameters,
}

impl From<CnavParameters> for PyCnavParameters {
    fn from(inner: CnavParameters) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCnavParameters {
    #[getter]
    fn adot_m_s(&self) -> f64 {
        self.inner.adot_m_s
    }

    #[getter]
    fn delta_n0_dot_rad_s2(&self) -> f64 {
        self.inner.delta_n0_dot_rad_s2
    }

    #[getter]
    fn top_week(&self) -> u32 {
        self.inner.top.week
    }

    #[getter]
    fn top_tow_s(&self) -> f64 {
        self.inner.top.tow_s
    }

    #[getter]
    fn top_time_scale(&self) -> PyTimeScale {
        self.inner.top.system.into()
    }

    #[getter]
    fn ura_ed_index(&self) -> i8 {
        self.inner.ura_ed_index
    }

    #[getter]
    fn ura_ned0_index(&self) -> i8 {
        self.inner.ura_ned0_index
    }

    #[getter]
    fn ura_ned1_index(&self) -> u8 {
        self.inner.ura_ned1_index
    }

    #[getter]
    fn ura_ned2_index(&self) -> u8 {
        self.inner.ura_ned2_index
    }

    #[getter]
    fn transmission_time_sow(&self) -> f64 {
        self.inner.transmission_time_sow
    }

    #[getter]
    fn flags(&self) -> Option<u32> {
        self.inner.flags
    }

    #[getter]
    fn ura_ed_m(&self) -> Option<f64> {
        core_cnav_ura_nominal_m(self.inner.ura_ed_index)
    }

    fn ura_ned_m(&self, week: u32, tow_s: f64) -> Option<f64> {
        let t = sidereon_core::astro::time::GnssWeekTow {
            system: self.inner.top.system,
            week,
            tow_s,
        };
        core_cnav_ura_ned_m(&self.inner, t)
    }

    fn __repr__(&self) -> String {
        format!(
            "CnavParameters(ura_ed_index={}, top_week={}, top_tow_s={})",
            self.inner.ura_ed_index, self.inner.top.week, self.inner.top.tow_s
        )
    }
}

/// GPS/QZSS legacy-vs-CNAV selection preference for mixed stores.
#[pyclass(
    module = "sidereon._sidereon",
    name = "NavMessagePreference",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyNavMessagePreference {
    PREFER_LEGACY,
    PREFER_MODERN,
}

impl From<PyNavMessagePreference> for NavMessagePreference {
    fn from(value: PyNavMessagePreference) -> Self {
        match value {
            PyNavMessagePreference::PREFER_LEGACY => Self::PreferLegacy,
            PyNavMessagePreference::PREFER_MODERN => Self::PreferModern,
        }
    }
}

impl From<NavMessagePreference> for PyNavMessagePreference {
    fn from(value: NavMessagePreference) -> Self {
        match value {
            NavMessagePreference::PreferLegacy => Self::PREFER_LEGACY,
            NavMessagePreference::PreferModern => Self::PREFER_MODERN,
        }
    }
}

#[pymethods]
impl PyNavMessagePreference {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::PREFER_LEGACY => "prefer_legacy",
            Self::PREFER_MODERN => "prefer_modern",
        }
    }
}

/// Observation kind inferred from a RINEX observation code.
#[pyclass(module = "sidereon._sidereon", name = "ObservationKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
#[allow(non_camel_case_types)]
pub enum PyObservationKind {
    /// Code pseudorange, metres.
    PSEUDORANGE,
    /// Carrier phase, cycles.
    CARRIER_PHASE,
    /// Doppler, hertz.
    DOPPLER,
    /// Signal strength, dB-Hz.
    SIGNAL_STRENGTH,
    /// Unknown leading RINEX code letter.
    UNKNOWN,
}

impl From<CoreObservationKind> for PyObservationKind {
    fn from(kind: CoreObservationKind) -> Self {
        match kind {
            CoreObservationKind::Pseudorange => Self::PSEUDORANGE,
            CoreObservationKind::CarrierPhase => Self::CARRIER_PHASE,
            CoreObservationKind::Doppler => Self::DOPPLER,
            CoreObservationKind::SignalStrength => Self::SIGNAL_STRENGTH,
            CoreObservationKind::Unknown => Self::UNKNOWN,
        }
    }
}

#[pymethods]
impl PyObservationKind {
    /// Stable lower-case label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::PSEUDORANGE => "pseudorange",
            Self::CARRIER_PHASE => "carrier_phase",
            Self::DOPPLER => "doppler",
            Self::SIGNAL_STRENGTH => "signal_strength",
            Self::UNKNOWN => "unknown",
        }
    }

    /// Units of the parsed value.
    #[getter]
    fn units(&self) -> &'static str {
        match self {
            Self::PSEUDORANGE => "meters",
            Self::CARRIER_PHASE => "cycles",
            Self::DOPPLER => "hz",
            Self::SIGNAL_STRENGTH => "db_hz",
            Self::UNKNOWN => "unknown",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::PSEUDORANGE => "ObservationKind.PSEUDORANGE",
            Self::CARRIER_PHASE => "ObservationKind.CARRIER_PHASE",
            Self::DOPPLER => "ObservationKind.DOPPLER",
            Self::SIGNAL_STRENGTH => "ObservationKind.SIGNAL_STRENGTH",
            Self::UNKNOWN => "ObservationKind.UNKNOWN",
        }
    }
}

/// Civil epoch from a RINEX observation file.
#[pyclass(module = "sidereon._sidereon", name = "ObsEpochTime")]
#[derive(Clone, Copy, Debug)]
pub struct PyObsEpochTime {
    inner: CoreObsEpochTime,
}

impl From<CoreObsEpochTime> for PyObsEpochTime {
    fn from(inner: CoreObsEpochTime) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsEpochTime {
    /// Calendar year in the file time scale.
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

    /// Fractional seconds of minute in the file time scale.
    #[getter]
    fn second(&self) -> f64 {
        self.inner.second
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsEpochTime({:04}-{:02}-{:02}T{:02}:{:02}:{:06.3})",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second
        )
    }

    fn __eq__(&self, other: &PyObsEpochTime) -> bool {
        self.inner == other.inner
    }
}

/// One `SYS / SCALE FACTOR` record from a RINEX OBS header.
#[pyclass(module = "sidereon._sidereon", name = "ObsScaleFactor")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyObsScaleFactor {
    inner: CoreObsScaleFactor,
}

impl From<CoreObsScaleFactor> for PyObsScaleFactor {
    fn from(inner: CoreObsScaleFactor) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsScaleFactor {
    /// Constellation the scale factor applies to.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    /// Factor to divide stored observations by before use.
    #[getter]
    fn factor(&self) -> f64 {
        self.inner.factor
    }

    /// Observation codes affected. Empty means all codes for the system.
    #[getter]
    fn codes(&self) -> Vec<String> {
        self.inner.codes.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsScaleFactor(system={}, factor={}, codes={:?})",
            self.system().label(),
            self.inner.factor,
            self.inner.codes
        )
    }
}

/// One `LEAP SECONDS` header record retained from an observation file.
#[pyclass(module = "sidereon._sidereon", name = "ObsLeapSeconds")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyObsLeapSeconds {
    inner: CoreObsLeapSeconds,
}

impl From<CoreObsLeapSeconds> for PyObsLeapSeconds {
    fn from(inner: CoreObsLeapSeconds) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsLeapSeconds {
    /// Current leap-second count.
    #[getter]
    fn current(&self) -> i64 {
        self.inner.current
    }

    /// Future/past delta field, if present.
    #[getter]
    fn delta_future(&self) -> Option<i64> {
        self.inner.delta_future
    }

    /// GPS week field, if present.
    #[getter]
    fn week(&self) -> Option<i64> {
        self.inner.week
    }

    /// Day field, if present.
    #[getter]
    fn day(&self) -> Option<i64> {
        self.inner.day
    }

    /// The `LEAP SECONDS` time-system identifier from columns 25..27, exactly
    /// as the record wrote it, or None where the record leaves the field out.
    ///
    /// RINEX 3.03 introduced `BDS` and `GPS`; 3.05 renamed BeiDou's token to
    /// `BDT`; RINEX 4 permits only `GPS`. The parser refuses a token its
    /// version does not permit, so a parsed header carries a token that version
    /// allows. The token is returned as written, with no alias normalisation:
    /// `BDS` and `BDT` stay apart, and an absent field stays apart from an
    /// explicit `GPS`.
    #[getter]
    fn time_system(&self) -> Option<&str> {
        self.inner.time_system.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsLeapSeconds(current={}, delta_future={:?}, week={:?}, day={:?}, time_system={:?})",
            self.inner.current,
            self.inner.delta_future,
            self.inner.week,
            self.inner.day,
            self.inner.time_system
        )
    }
}

/// The `GLONASS COD/PHS/BIS` code-phase bias an OBS header gives a signal.
///
/// `status` separates the four answers the header can give: `available`
/// (`value` holds the bias in metres), `none` (the header gives the signal no
/// bias, which includes a file from RINEX 4.00 on, where Table A2 says the
/// record's lines "should be ignored by RINEX decoders and encoders"),
/// `unknown` (a blank record, or a blank bias for the signal: "If the GLONASS
/// code phase alignment is unknown, then all fields within GLONASS COD/PHS/BIS
/// header record are left blank", RINEX 3.05 section 5.2.16), and `ambiguous`
/// (records in one header block give the signal different biases, kept in
/// `corrections` in header order, a blank one as None).
///
/// No record at all and a blank record are different answers: the first is
/// `none`, the second `unknown`.
#[pyclass(module = "sidereon._sidereon", name = "GlonassBiasResult")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyGlonassBiasResult {
    pub(crate) status: &'static str,
    pub(crate) value: Option<f64>,
    pub(crate) corrections: Vec<Option<f64>>,
}

#[pymethods]
impl PyGlonassBiasResult {
    /// Status: 'available', 'none', 'unknown', or 'ambiguous'.
    #[getter]
    fn status(&self) -> &'static str {
        self.status
    }

    /// Alignment bias in metres when available, else None.
    #[getter]
    fn value(&self) -> Option<f64> {
        self.value
    }

    /// Alignment bias in metres when available, else None.
    #[getter]
    fn bias_m(&self) -> Option<f64> {
        self.value
    }

    /// True if an unambiguous bias value is available.
    #[getter]
    fn is_available(&self) -> bool {
        self.status == "available"
    }

    /// True if no record was declared (or record deprecated in RINEX 4).
    #[getter]
    fn is_none(&self) -> bool {
        self.status == "none"
    }

    /// True if header declares alignment unknown (blank record/field).
    #[getter]
    fn is_unknown(&self) -> bool {
        self.status == "unknown"
    }

    /// True if header gives contradictory biases for this code.
    #[getter]
    fn is_ambiguous(&self) -> bool {
        self.status == "ambiguous"
    }

    /// Conflicting bias values in header order when ambiguous.
    #[getter]
    fn corrections(&self) -> Vec<Option<f64>> {
        self.corrections.clone()
    }

    fn __repr__(&self) -> String {
        match self.status {
            "available" => format!(
                "GlonassBiasResult(status=\"available\", value={:?})",
                self.value
            ),
            "none" => "GlonassBiasResult(status=\"none\")".to_string(),
            "unknown" => "GlonassBiasResult(status=\"unknown\")".to_string(),
            "ambiguous" => format!(
                "GlonassBiasResult(status=\"ambiguous\", corrections={:?})",
                self.corrections
            ),
            _ => format!("GlonassBiasResult(status={:?})", self.status),
        }
    }
}

/// One `SYS / PHASE SHIFT` record from a RINEX OBS header.
#[pyclass(module = "sidereon._sidereon", name = "ObsPhaseShift")]
#[derive(Clone)]
pub struct PyObsPhaseShift {
    inner: CoreObsPhaseShift,
}

impl From<CoreObsPhaseShift> for PyObsPhaseShift {
    fn from(inner: CoreObsPhaseShift) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsPhaseShift {
    /// Constellation this correction applies to.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    /// RINEX carrier observation code, such as `L1C`, or None for a record
    /// naming only its constellation, which declares that constellation's phase
    /// alignment unknown (RINEX 3.05 section 5.2.12).
    #[getter]
    fn code(&self) -> Option<&str> {
        self.inner.code.as_deref()
    }

    /// Phase correction in carrier cycles, or None where the record leaves the
    /// field blank: "Correction applied (cycles) or blank if none" (RINEX 3.05
    /// and 4.02 Table A2). A blank field on a record that names a code says no
    /// correction was applied, which reads as 0.0 cycles; it does not say the
    /// alignment is unknown. Only a record with no code says that.
    #[getter]
    fn correction_cycles(&self) -> Option<f64> {
        self.inner.correction_cycles
    }

    /// Satellite tokens this correction is restricted to. Empty means all.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner
            .satellites
            .iter()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// Satellite tokens unrepresentable in the engine, preserved for lossless rewrite.
    #[getter]
    fn unrepresentable_satellites(&self) -> Vec<String> {
        self.inner.unrepresentable_satellites.clone()
    }

    /// True if the record names no satellite restriction (representable or unrepresentable).
    fn covers_every_satellite(&self) -> bool {
        self.inner.covers_every_satellite()
    }

    /// Total satellite count named by this record (representable + unrepresentable).
    fn satellite_count(&self) -> usize {
        self.inner.satellite_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsPhaseShift(system={}, code={:?}, correction_cycles={:?})",
            self.system().label(),
            self.inner.code,
            self.inner.correction_cycles
        )
    }
}

/// Parsed RINEX OBS header.
///
/// Every property returns a detached copy of what the header holds: changing
/// the returned list, array or record changes nothing in the product. There is
/// no header-editing API, so the product stays the one authority for its own
/// values.
///
/// `obs_codes` is the product's union list, which epoch values are aligned to;
/// `declared_obs_codes` is what this particular header declares. On a header
/// from `RinexObs.header_at` or `ObsHeaderTimeline.at`, `declared_obs_codes`,
/// `phase_shifts`, `scale_factors`, `glonass_slots` and `glonass_cod_phs_bis`
/// are the ones in effect at that epoch.
#[pyclass(module = "sidereon._sidereon", name = "ObsHeader")]
#[derive(Clone)]
pub struct PyObsHeader {
    pub(crate) inner: CoreObsHeader,
}

impl From<CoreObsHeader> for PyObsHeader {
    fn from(inner: CoreObsHeader) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsHeader {
    /// RINEX version, for example `3.05`.
    #[getter]
    fn version(&self) -> f64 {
        self.inner.version
    }

    /// Surveyed receiver a-priori ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn approx_position_m<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .approx_position_m
            .map(|position| np_array(py, &position))
    }

    /// Antenna H/E/N offset as a numpy `(3,)` array, metres.
    #[getter]
    fn antenna_delta_hen_m<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .antenna_delta_hen_m
            .map(|delta| np_array(py, &delta))
    }

    /// Nominal epoch interval in seconds, if present.
    #[getter]
    fn interval_s(&self) -> Option<f64> {
        self.inner.interval_s
    }

    /// Marker or station name, if present.
    #[getter]
    fn marker_name(&self) -> Option<&str> {
        self.inner.marker_name.as_deref()
    }

    /// Marker number, if present.
    #[getter]
    fn marker_number(&self) -> Option<&str> {
        self.inner.marker_number.as_deref()
    }

    /// Marker type, if present.
    #[getter]
    fn marker_type(&self) -> Option<&str> {
        self.inner.marker_type.as_deref()
    }

    /// Observer name, if present.
    #[getter]
    fn observer(&self) -> Option<&str> {
        self.inner.observer.as_deref()
    }

    /// Agency name, if present.
    #[getter]
    fn agency(&self) -> Option<&str> {
        self.inner.agency.as_deref()
    }

    /// Receiver serial number, if present.
    #[getter]
    fn receiver_number(&self) -> Option<&str> {
        self.inner.receiver.as_ref().map(|r| r.number.as_str())
    }

    /// Receiver type, if present.
    #[getter]
    fn receiver_type(&self) -> Option<&str> {
        self.inner
            .receiver
            .as_ref()
            .map(|r| r.receiver_type.as_str())
    }

    /// Receiver firmware/version, if present.
    #[getter]
    fn receiver_version(&self) -> Option<&str> {
        self.inner.receiver.as_ref().map(|r| r.version.as_str())
    }

    /// Antenna serial number, if present.
    #[getter]
    fn antenna_number(&self) -> Option<&str> {
        self.inner.antenna.as_ref().map(|a| a.number.as_str())
    }

    /// Antenna type, if present.
    #[getter]
    fn antenna_type(&self) -> Option<&str> {
        self.inner.antenna.as_ref().map(|a| a.antenna_type.as_str())
    }

    /// Program name from PGM / RUN BY / DATE, if present.
    #[getter]
    fn program(&self) -> Option<&str> {
        self.inner
            .program_run_by_date
            .as_ref()
            .map(|p| p.program.as_str())
    }

    /// Agency/user from PGM / RUN BY / DATE, if present.
    #[getter]
    fn run_by(&self) -> Option<&str> {
        self.inner
            .program_run_by_date
            .as_ref()
            .map(|p| p.run_by.as_str())
    }

    /// Date string from PGM / RUN BY / DATE, if present.
    #[getter]
    fn program_date(&self) -> Option<&str> {
        self.inner
            .program_run_by_date
            .as_ref()
            .map(|p| p.date.as_str())
    }

    /// Header comments in file order.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Declared distinct-satellite count, if present.
    #[getter]
    fn n_satellites(&self) -> Option<usize> {
        self.inner.n_satellites
    }

    /// Constellations with union observation-code lists.
    #[getter]
    fn systems(&self) -> Vec<PyGnssSystem> {
        self.inner
            .obs_codes
            .keys()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// Constellations declaring code lists in this header.
    #[getter]
    fn declared_systems(&self) -> Vec<PyGnssSystem> {
        self.inner
            .declared_obs_codes
            .keys()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// RINEX 2 # / TYPES OF OBSERV names as read.
    #[getter]
    fn rinex2_types(&self) -> Vec<String> {
        self.inner.rinex2_types.clone()
    }

    /// Constellation declared by RINEX 2 version record, if single-system.
    #[getter]
    fn rinex2_system(&self) -> Option<PyGnssSystem> {
        self.inner.rinex2_system.map(Into::into)
    }

    /// Signal-strength unit, such as DBHZ.
    #[getter]
    fn signal_strength_unit(&self) -> Option<&str> {
        self.inner.signal_strength_unit.as_deref()
    }

    /// Leap seconds record, if present. A detached copy of the header's record.
    #[getter]
    fn leap_seconds(&self) -> Option<PyObsLeapSeconds> {
        self.inner.leap_seconds.clone().map(Into::into)
    }

    /// `PRN / # OF OBS` counts per satellite, index-aligned with that
    /// satellite's system code list. A blank count field stays None.
    #[getter]
    fn prn_obs_counts(&self) -> BTreeMap<String, Vec<Option<usize>>> {
        self.inner
            .prn_obs_counts
            .iter()
            .map(|(sat, counts)| (sat.to_string(), counts.clone()))
            .collect()
    }

    /// Unretained header labels recorded as drop-on-rewrite disclosure.
    #[getter]
    fn unretained_header_labels(&self) -> Vec<String> {
        self.inner.unretained_header_labels.clone()
    }

    /// Raw GLONASS code-phase bias entries as (code, bias_or_none) pairs.
    #[getter]
    fn glonass_cod_phs_bis(&self) -> Option<Vec<(String, Option<f64>)>> {
        self.inner.glonass_cod_phs_bis.clone()
    }

    /// Carrier phase-shift records in header order.
    #[getter]
    fn phase_shifts(&self) -> Vec<PyObsPhaseShift> {
        self.inner
            .phase_shifts
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Scale-factor records in header order.
    #[getter]
    fn scale_factors(&self) -> Vec<PyObsScaleFactor> {
        self.inner
            .scale_factors
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// GLONASS slot to FDMA frequency-channel entries.
    #[getter]
    fn glonass_slots(&self) -> Vec<(u8, i8)> {
        self.inner
            .glonass_slots
            .iter()
            .map(|(&slot, &channel)| (slot, channel))
            .collect()
    }

    /// First observation epoch and its time scale, if present.
    #[getter]
    fn time_of_first_obs(&self) -> Option<(PyObsEpochTime, PyTimeScale)> {
        self.inner
            .time_of_first_obs
            .map(|(epoch, scale)| (epoch.into(), scale.into()))
    }

    /// Last observation epoch and its time scale, if present.
    #[getter]
    fn time_of_last_obs(&self) -> Option<(PyObsEpochTime, PyTimeScale)> {
        self.inner
            .time_of_last_obs
            .map(|(epoch, scale)| (epoch.into(), scale.into()))
    }

    /// The product's union code list for a constellation: the file header's
    /// codes first, then each code a later event list declares, in the order
    /// first declared. Epoch value vectors are index-aligned to this list, so
    /// it is what reads an epoch's values; it is not what the header declares
    /// at any one epoch. For that, use `declared_obs_codes`.
    fn obs_codes(&self, system: PyGnssSystem) -> Vec<String> {
        self.inner
            .obs_codes
            .get(&system.into())
            .cloned()
            .unwrap_or_default()
    }

    /// The code list this header declares for a constellation, as its
    /// `SYS / # / OBS TYPES` records give it. On a header from
    /// `RinexObs.header_at` or `ObsHeaderTimeline.at`, this is the list in
    /// effect at that epoch, after the event records at or before it.
    fn declared_obs_codes(&self, system: PyGnssSystem) -> Vec<String> {
        self.inner
            .declared_obs_codes
            .get(&system.into())
            .cloned()
            .unwrap_or_default()
    }

    /// Query GLONASS code-phase bias for an observable code.
    fn glonass_code_phase_bias(&self, code: &str) -> PyGlonassBiasResult {
        match self.inner.glonass_code_phase_bias(code) {
            Ok(Some(value)) => PyGlonassBiasResult {
                status: "available",
                value: Some(value),
                corrections: Vec::new(),
            },
            Ok(None) => PyGlonassBiasResult {
                status: "none",
                value: None,
                corrections: Vec::new(),
            },
            Err(CoreCorrectionUnavailable::Unknown) => PyGlonassBiasResult {
                status: "unknown",
                value: None,
                corrections: Vec::new(),
            },
            Err(CoreCorrectionUnavailable::Ambiguous { corrections }) => PyGlonassBiasResult {
                status: "ambiguous",
                value: None,
                corrections,
            },
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsHeader(version={}, systems={}, marker_name={:?})",
            self.inner.version,
            self.inner.obs_codes.len(),
            self.inner.marker_name
        )
    }
}

/// One reconstructed observation value with indicators.
#[pyclass(module = "sidereon._sidereon", name = "ObsValue")]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PyObsValue {
    inner: CoreObsValue,
}

impl From<CoreObsValue> for PyObsValue {
    fn from(inner: CoreObsValue) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsValue {
    /// Observed value (metres, cycles, etc.) or None if blank.
    #[getter]
    fn value(&self) -> Option<f64> {
        self.inner.value
    }

    /// Loss-of-lock indicator (LLI), None when blank.
    #[getter]
    fn lli(&self) -> Option<u8> {
        self.inner.lli
    }

    /// Signal-strength indicator (SSI), None when blank.
    #[getter]
    fn ssi(&self) -> Option<u8> {
        self.inner.ssi
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsValue(value={:?}, lli={:?}, ssi={:?})",
            self.inner.value, self.inner.lli, self.inner.ssi
        )
    }
}

/// One RINEX OBS epoch record: its civil time, its flag, and what it carried.
///
/// The three kinds of record are kept apart. An observation epoch (flag 0 or 1)
/// carries `sats` and nothing else. A cycle slip epoch (flag 6) carries
/// `cycle_slips`, which are slips written in the observation record layout, not
/// measurements, and are never mixed into `sats`. Any other flag above 1 is an
/// event, whose records are kept verbatim in `special_records`; it carries no
/// observations and no slips, and may leave its epoch fields blank, in which
/// case `epoch` is None.
///
/// Every list and map returned is a detached copy of what the epoch holds.
#[pyclass(module = "sidereon._sidereon", name = "ObsEpoch")]
#[derive(Clone)]
pub struct PyObsEpoch {
    pub(crate) inner: CoreObsEpoch,
}

impl From<CoreObsEpoch> for PyObsEpoch {
    fn from(inner: CoreObsEpoch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsEpoch {
    /// Civil epoch in the file time scale, or None for untimed events.
    #[getter]
    fn epoch(&self) -> Option<PyObsEpochTime> {
        self.inner.epoch.map(Into::into)
    }

    /// RINEX epoch flag. `0` is a normal observation epoch, `6` cycle slips.
    #[getter]
    fn flag(&self) -> u8 {
        self.inner.flag
    }

    /// Receiver clock offset from epoch line in seconds, if present.
    #[getter]
    fn rcv_clock_offset_s(&self) -> Option<f64> {
        self.inner.rcv_clock_offset_s
    }

    /// Optional RINEX 4 epoch picoseconds.
    #[getter]
    fn epoch_picoseconds(&self) -> Option<u32> {
        self.inner.epoch_picoseconds
    }

    /// Satellite/special-record count declared on the epoch line.
    #[getter]
    fn declared_record_count(&self) -> usize {
        self.inner.declared_record_count
    }

    /// Event special records as written, verbatim. Empty for observation/cycle slip epochs.
    #[getter]
    fn special_records(&self) -> Vec<String> {
        self.inner.special_records.clone()
    }

    /// Satellite tokens present with measurements at this epoch.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner.sats.keys().map(|sat| sat.to_string()).collect()
    }

    /// Number of satellites present with measurements at this epoch.
    #[getter]
    fn satellite_count(&self) -> usize {
        self.inner.sats.len()
    }

    /// Satellite tokens reporting cycle slips at this epoch. Empty if flag != 6.
    #[getter]
    fn cycle_slip_satellites(&self) -> Vec<String> {
        self.inner
            .cycle_slips
            .keys()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// Number of satellites reporting cycle slips at this epoch.
    #[getter]
    fn cycle_slip_count(&self) -> usize {
        self.inner.cycle_slips.len()
    }

    /// Cycle slips per satellite, held separate from measurements.
    #[getter]
    fn cycle_slips(&self) -> BTreeMap<String, Vec<PyObsValue>> {
        self.inner
            .cycle_slips
            .iter()
            .map(|(sat, values)| {
                (
                    sat.to_string(),
                    values.iter().copied().map(Into::into).collect(),
                )
            })
            .collect()
    }

    /// Measurement values per satellite. Empty for event epochs and cycle slip epochs.
    #[getter]
    fn sats(&self) -> BTreeMap<String, Vec<PyObsValue>> {
        self.inner
            .sats
            .iter()
            .map(|(sat, values)| {
                (
                    sat.to_string(),
                    values.iter().copied().map(Into::into).collect(),
                )
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsEpoch(epoch={:?}, flag={}, satellite_count={}, special_records={})",
            self.epoch(),
            self.inner.flag,
            self.inner.sats.len(),
            self.inner.special_records.len()
        )
    }
}

/// Optional observation-code allow-list for raw OBS and carrier-phase rows.
#[pyclass(module = "sidereon._sidereon", name = "ObservationFilter")]
#[derive(Clone)]
pub struct PyObservationFilter {
    inner: CoreObservationFilter,
}

#[pymethods]
impl PyObservationFilter {
    /// Build a filter from `(GnssSystem, [code, ...])` entries. Empty keeps all.
    #[new]
    #[pyo3(signature = (entries=None))]
    fn new(entries: Option<Vec<(PyGnssSystem, Vec<String>)>>) -> Self {
        Self {
            inner: CoreObservationFilter::from_entries(obs_entries_from_py(entries)),
        }
    }

    /// Filter that keeps every parsed observation.
    #[staticmethod]
    fn all() -> Self {
        Self {
            inner: CoreObservationFilter::all(),
        }
    }

    /// Filter entries as `(GnssSystem, [code, ...])` pairs.
    #[getter]
    fn entries(&self) -> Vec<(PyGnssSystem, Vec<String>)> {
        self.inner
            .codes
            .iter()
            .map(|(&system, codes)| (system.into(), codes.clone()))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("ObservationFilter(entries={})", self.inner.codes.len())
    }
}

/// Per-constellation single-frequency pseudorange code-selection policy.
#[pyclass(module = "sidereon._sidereon", name = "SignalPolicy")]
#[derive(Clone)]
pub struct PySignalPolicy {
    inner: CoreSignalPolicy,
}

impl PySignalPolicy {
    pub(crate) fn inner(&self) -> CoreSignalPolicy {
        self.inner.clone()
    }
}

#[pymethods]
impl PySignalPolicy {
    /// Build a pseudorange policy from `(GnssSystem, [code, ...])` entries.
    #[new]
    #[pyo3(signature = (entries=None))]
    fn new(entries: Option<Vec<(PyGnssSystem, Vec<String>)>>) -> Self {
        Self {
            inner: CoreSignalPolicy {
                codes: obs_entries_from_py(entries),
            },
        }
    }

    /// Core default policy for a RINEX version.
    #[staticmethod]
    fn default_for(version: f64) -> PyResult<Self> {
        Ok(Self {
            inner: CoreSignalPolicy::default_for(version).map_err(to_obs_err)?,
        })
    }

    /// Return a copy with one constellation preference list replaced.
    fn with_override(&self, system: PyGnssSystem, codes: Vec<String>) -> Self {
        Self {
            inner: self.inner.clone().with_override(system.into(), codes),
        }
    }

    /// Policy entries as `(GnssSystem, [code, ...])` pairs.
    #[getter]
    fn entries(&self) -> Vec<(PyGnssSystem, Vec<String>)> {
        self.inner
            .codes
            .iter()
            .map(|(&system, codes)| (system.into(), codes.clone()))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("SignalPolicy(entries={})", self.inner.codes.len())
    }
}

/// Flattened pseudorange rows from one RINEX OBS epoch.
///
/// `satellites` is row-aligned with `ranges_m`; ranges are metres.
#[pyclass(module = "sidereon._sidereon", name = "PseudorangeSeries")]
#[derive(Clone)]
pub struct PyPseudorangeSeries {
    satellites: Vec<String>,
    ranges_m: Vec<f64>,
}

impl PyPseudorangeSeries {
    fn from_rows(rows: Vec<(GnssSatelliteId, f64)>) -> Self {
        let mut satellites = Vec::with_capacity(rows.len());
        let mut ranges_m = Vec::with_capacity(rows.len());
        for (satellite, range_m) in rows {
            satellites.push(satellite.to_string());
            ranges_m.push(range_m);
        }
        Self {
            satellites,
            ranges_m,
        }
    }
}

#[pymethods]
impl PyPseudorangeSeries {
    /// Satellite tokens, row-aligned with `ranges_m`.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.satellites.clone()
    }

    /// Pseudorange values as a numpy `(n,)` array, metres.
    #[getter]
    fn ranges_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.ranges_m.clone())
    }

    /// Number of rows.
    fn __len__(&self) -> usize {
        self.ranges_m.len()
    }

    fn __repr__(&self) -> String {
        format!("PseudorangeSeries(rows={})", self.ranges_m.len())
    }
}

/// Flattened raw observation rows from one RINEX OBS epoch.
///
/// Numeric arrays are row-aligned with `satellites`, `codes`, and `kinds`.
/// Blank RINEX values, LLI, and SSI fields are represented as `NaN`.
#[pyclass(module = "sidereon._sidereon", name = "ObservationValueSeries")]
#[derive(Clone)]
pub struct PyObservationValueSeries {
    satellites: Vec<String>,
    codes: Vec<String>,
    kinds: Vec<PyObservationKind>,
    values: Vec<f64>,
    lli: Vec<f64>,
    ssi: Vec<f64>,
}

impl PyObservationValueSeries {
    fn from_rows(rows: Vec<(GnssSatelliteId, Vec<CoreObservationValueRow>)>) -> Self {
        let row_count = rows.iter().map(|(_, rows)| rows.len()).sum();
        let mut out = Self {
            satellites: Vec::with_capacity(row_count),
            codes: Vec::with_capacity(row_count),
            kinds: Vec::with_capacity(row_count),
            values: Vec::with_capacity(row_count),
            lli: Vec::with_capacity(row_count),
            ssi: Vec::with_capacity(row_count),
        };
        for (satellite, rows) in rows {
            let satellite = satellite.to_string();
            for row in rows {
                out.satellites.push(satellite.clone());
                out.codes.push(row.code);
                out.kinds.push(row.kind.into());
                out.values.push(nan_if_missing(row.value));
                out.lli.push(u8_nan_if_missing(row.lli));
                out.ssi.push(u8_nan_if_missing(row.ssi));
            }
        }
        out
    }
}

#[pymethods]
impl PyObservationValueSeries {
    /// Satellite tokens, row-aligned with all arrays.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.satellites.clone()
    }

    /// RINEX observation codes, row-aligned with all arrays.
    #[getter]
    fn codes(&self) -> Vec<String> {
        self.codes.clone()
    }

    /// Observation kinds, row-aligned with all arrays.
    #[getter]
    fn kinds(&self) -> Vec<PyObservationKind> {
        self.kinds.clone()
    }

    /// Parsed observation values as a numpy `(n,)` array.
    #[getter]
    fn values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.values.clone())
    }

    /// RINEX LLI values as a numpy `(n,)` array, with `NaN` for blanks.
    #[getter]
    fn lli<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.lli.clone())
    }

    /// RINEX SSI values as a numpy `(n,)` array, with `NaN` for blanks.
    #[getter]
    fn ssi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.ssi.clone())
    }

    /// Number of flattened rows.
    fn __len__(&self) -> usize {
        self.values.len()
    }

    fn __repr__(&self) -> String {
        format!("ObservationValueSeries(rows={})", self.values.len())
    }
}

/// The `SYS / PHASE SHIFT` correction the header states for one carrier row.
///
/// RINEX 3 stores phase observations already aligned, so the correction is
/// metadata describing what was applied when the file was written, for
/// reconstructing the original values. It is not a correction to apply again to
/// `CarrierPhaseSeries.value_cycles`.
///
/// `status` is `available` (`cycles` holds the stated correction, 0.0 where no
/// record covers the row or the covering record leaves the field blank),
/// `unknown` (the only record covering the row names just the constellation,
/// declaring the alignment unknown), or `ambiguous` (records in one header
/// block give the row's code different corrections, kept in `corrections` in
/// the order the records give them, a blank one as None). A stated 0.0 and an
/// `unknown` row are different answers and stay apart.
///
/// From RINEX 4.00 the records "should be ignored by RINEX decoders and
/// encoders" (Table A2), so every row of such a file reports `available` with
/// 0.0 cycles whatever its records say; the records themselves stay on
/// `ObsHeader.phase_shifts` and are written back.
#[pyclass(module = "sidereon._sidereon", name = "PhaseShiftResult")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyPhaseShiftResult {
    pub(crate) status: &'static str,
    pub(crate) cycles: Option<f64>,
    pub(crate) corrections: Vec<Option<f64>>,
}

impl From<Result<f64, CoreCorrectionUnavailable>> for PyPhaseShiftResult {
    fn from(res: Result<f64, CoreCorrectionUnavailable>) -> Self {
        match res {
            Ok(cycles) => Self {
                status: "available",
                cycles: Some(cycles),
                corrections: Vec::new(),
            },
            Err(CoreCorrectionUnavailable::Unknown) => Self {
                status: "unknown",
                cycles: None,
                corrections: Vec::new(),
            },
            Err(CoreCorrectionUnavailable::Ambiguous { corrections }) => Self {
                status: "ambiguous",
                cycles: None,
                corrections,
            },
        }
    }
}

#[pymethods]
impl PyPhaseShiftResult {
    /// Status tag: 'available', 'unknown', or 'ambiguous'.
    #[getter]
    fn status(&self) -> &'static str {
        self.status
    }

    /// Phase correction cycles when available, otherwise None.
    #[getter]
    fn cycles(&self) -> Option<f64> {
        self.cycles
    }

    /// Alias for cycles.
    #[getter]
    fn value(&self) -> Option<f64> {
        self.cycles
    }

    /// True if an unambiguous correction value is available.
    #[getter]
    fn is_available(&self) -> bool {
        self.status == "available"
    }

    /// True if header declares alignment unknown (blank record/field).
    #[getter]
    fn is_unknown(&self) -> bool {
        self.status == "unknown"
    }

    /// True if header gives contradictory corrections for this satellite/code.
    #[getter]
    fn is_ambiguous(&self) -> bool {
        self.status == "ambiguous"
    }

    /// Conflicting correction values in header order when ambiguous.
    #[getter]
    fn corrections(&self) -> Vec<Option<f64>> {
        self.corrections.clone()
    }

    fn __repr__(&self) -> String {
        match self.status {
            "available" => format!(
                "PhaseShiftResult(status=\"available\", cycles={:?})",
                self.cycles
            ),
            "ambiguous" => format!(
                "PhaseShiftResult(status=\"ambiguous\", corrections={:?})",
                self.corrections
            ),
            _ => "PhaseShiftResult(status=\"unknown\")".to_string(),
        }
    }
}

/// Flattened carrier-phase rows from one RINEX OBS epoch.
///
/// Numeric arrays and `phase_shift_results` are row-aligned with `satellites`
/// and `codes`, and every list and array has one entry per row. A blank RINEX
/// field and unknown carrier metadata read as `NaN`.
///
/// Every row the epoch holds is kept, including rows whose phase-shift
/// correction is unknown or ambiguous; none is dropped and none is given a
/// substitute correction. An event epoch and a cycle slip epoch hold no
/// observations, so they produce no rows at all.
#[pyclass(module = "sidereon._sidereon", name = "CarrierPhaseSeries")]
#[derive(Clone)]
pub struct PyCarrierPhaseSeries {
    satellites: Vec<String>,
    codes: Vec<String>,
    value_cycles: Vec<f64>,
    frequency_hz: Vec<f64>,
    wavelength_m: Vec<f64>,
    value_m: Vec<f64>,
    phase_shift_cycles: Vec<f64>,
    phase_shift_valid: Vec<bool>,
    phase_shift_results: Vec<PyPhaseShiftResult>,
    lli: Vec<f64>,
    ssi: Vec<f64>,
}

impl PyCarrierPhaseSeries {
    fn from_rows(rows: Vec<(GnssSatelliteId, Vec<CoreCarrierPhaseRow>)>) -> Self {
        let row_count = rows.iter().map(|(_, rows)| rows.len()).sum();
        let mut out = Self {
            satellites: Vec::with_capacity(row_count),
            codes: Vec::with_capacity(row_count),
            value_cycles: Vec::with_capacity(row_count),
            frequency_hz: Vec::with_capacity(row_count),
            wavelength_m: Vec::with_capacity(row_count),
            value_m: Vec::with_capacity(row_count),
            phase_shift_cycles: Vec::with_capacity(row_count),
            phase_shift_valid: Vec::with_capacity(row_count),
            phase_shift_results: Vec::with_capacity(row_count),
            lli: Vec::with_capacity(row_count),
            ssi: Vec::with_capacity(row_count),
        };
        for (satellite, rows) in rows {
            let satellite = satellite.to_string();
            for row in rows {
                let res = PyPhaseShiftResult::from(row.phase_shift_cycles);
                let (cycles, valid) = match res.cycles {
                    Some(c) => (c, true),
                    None => (f64::NAN, false),
                };
                out.satellites.push(satellite.clone());
                out.codes.push(row.code);
                out.value_cycles.push(nan_if_missing(row.value_cycles));
                out.frequency_hz.push(nan_if_missing(row.frequency_hz));
                out.wavelength_m.push(nan_if_missing(row.wavelength_m));
                out.value_m.push(nan_if_missing(row.value_m));
                out.phase_shift_cycles.push(cycles);
                out.phase_shift_valid.push(valid);
                out.phase_shift_results.push(res);
                out.lli.push(u8_nan_if_missing(row.lli));
                out.ssi.push(u8_nan_if_missing(row.ssi));
            }
        }
        out
    }
}

#[pymethods]
impl PyCarrierPhaseSeries {
    /// Satellite tokens, row-aligned with all arrays.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.satellites.clone()
    }

    /// RINEX carrier observation codes, row-aligned with all arrays.
    #[getter]
    fn codes(&self) -> Vec<String> {
        self.codes.clone()
    }

    /// Carrier phase as a numpy `(n,)` array, cycles.
    #[getter]
    fn value_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.value_cycles.clone())
    }

    /// Carrier frequency as a numpy `(n,)` array, hertz.
    #[getter]
    fn frequency_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.frequency_hz.clone())
    }

    /// Carrier wavelength as a numpy `(n,)` array, metres.
    #[getter]
    fn wavelength_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.wavelength_m.clone())
    }

    /// Carrier phase as a numpy `(n,)` array, metres.
    #[getter]
    fn value_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.value_m.clone())
    }

    /// The stated `SYS / PHASE SHIFT` correction per row as a numpy `(n,)`
    /// array, cycles, with `NaN` where the header states no one correction.
    ///
    /// A convenience beside `phase_shift_results`, which is the authoritative
    /// answer: `NaN` here means only "not one value", and does not say whether
    /// the header declared the alignment unknown or gave the row's code several
    /// corrections. Read `phase_shift_valid` or `phase_shift_statuses` before
    /// using a row. These describe values already aligned in the file; they are
    /// not to be re-applied to `value_cycles`.
    #[getter]
    fn phase_shift_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.phase_shift_cycles.clone())
    }

    /// Boolean numpy `(n,)` mask, True where the header states one correction
    /// for the row (including a stated 0.0) and False where it is unknown or
    /// ambiguous.
    #[getter]
    fn phase_shift_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_vec(py, self.phase_shift_valid.clone())
    }

    /// The typed `PhaseShiftResult` for every row, in row order. Authoritative:
    /// it keeps the unavailability reason and every conflicting correction.
    #[getter]
    fn phase_shift_results(&self) -> Vec<PyPhaseShiftResult> {
        self.phase_shift_results.clone()
    }

    /// The `phase_shift_results` status of every row: 'available', 'unknown',
    /// or 'ambiguous'.
    #[getter]
    fn phase_shift_statuses(&self) -> Vec<&'static str> {
        self.phase_shift_results.iter().map(|r| r.status).collect()
    }

    /// RINEX LLI values as a numpy `(n,)` array, with `NaN` for blanks.
    #[getter]
    fn lli<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.lli.clone())
    }

    /// RINEX SSI values as a numpy `(n,)` array, with `NaN` for blanks.
    #[getter]
    fn ssi<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.ssi.clone())
    }

    /// Number of flattened rows.
    fn __len__(&self) -> usize {
        self.value_cycles.len()
    }

    fn __repr__(&self) -> String {
        format!("CarrierPhaseSeries(rows={})", self.value_cycles.len())
    }
}

/// The header in effect at each epoch of a product.
///
/// The file header, and from each event whose records take effect, the header
/// those records lay over the one before. A phase shift, code list, scale
/// factor or GLONASS channel an event declares applies from that event's epoch
/// on; the epochs before it keep the value from before it. Built once by
/// `RinexObs.header_timeline`, so a loop over the epochs looks each header up
/// rather than reading the event records again.
#[pyclass(module = "sidereon._sidereon", name = "ObsHeaderTimeline")]
#[derive(Clone)]
pub struct PyObsHeaderTimeline {
    inner: CoreObsHeaderTimeline,
}

impl From<CoreObsHeaderTimeline> for PyObsHeaderTimeline {
    fn from(inner: CoreObsHeaderTimeline) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsHeaderTimeline {
    /// The header in effect at an epoch index: the file header with every event
    /// at or before it laid over it. Past the last epoch, the header after
    /// every event.
    fn at(&self, epoch_index: usize) -> PyObsHeader {
        self.inner.at(epoch_index).clone().into()
    }

    /// The position in `segments()` of the header in effect at an epoch index.
    fn segment_index(&self, epoch_index: usize) -> usize {
        self.inner.segment_index(epoch_index)
    }

    /// Each header with the index of the first epoch it is in effect at, in
    /// file order, beginning with the file header at index 0.
    fn segments(&self) -> Vec<(usize, PyObsHeader)> {
        self.inner
            .segments()
            .map(|(idx, header)| (idx, header.clone().into()))
            .collect()
    }

    /// Total number of distinct header segments in the timeline.
    #[getter]
    fn segment_count(&self) -> usize {
        self.inner.segments().count()
    }

    fn __len__(&self) -> usize {
        self.segment_count()
    }

    fn __repr__(&self) -> String {
        format!("ObsHeaderTimeline(segments={})", self.segment_count())
    }
}

/// One change `RinexObs.downgrade_to_rinex2` made to turn a product into one a
/// version 2 file can state exactly.
///
/// `kind` is the variant name; `details()` carries that variant's own payload
/// under its own keys, with satellites as RINEX designators, constellations as
/// `GnssSystem` and a nested change (`InEventLists`) as an
/// `ObsDowngradeChange`.
#[pyclass(module = "sidereon._sidereon", name = "ObsDowngradeChange")]
#[derive(Clone, Debug)]
pub struct PyObsDowngradeChange {
    pub(crate) inner: CoreObsDowngradeChange,
}

impl From<CoreObsDowngradeChange> for PyObsDowngradeChange {
    fn from(inner: CoreObsDowngradeChange) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsDowngradeChange {
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Change kind discriminator string.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            CoreObsDowngradeChange::CodeRenamed { .. } => "CodeRenamed",
            CoreObsDowngradeChange::CodeMoved { .. } => "CodeMoved",
            CoreObsDowngradeChange::CodeAdded { .. } => "CodeAdded",
            CoreObsDowngradeChange::CodeListRemoved { .. } => "CodeListRemoved",
            CoreObsDowngradeChange::ValueRounded { .. } => "ValueRounded",
            CoreObsDowngradeChange::CycleSlipRounded { .. } => "CycleSlipRounded",
            CoreObsDowngradeChange::ScaleFactorsRemoved { .. } => "ScaleFactorsRemoved",
            CoreObsDowngradeChange::EpochPicosecondsRemoved { .. } => "EpochPicosecondsRemoved",
            CoreObsDowngradeChange::ClockOffsetRounded { .. } => "ClockOffsetRounded",
            CoreObsDowngradeChange::InEventLists { .. } => "InEventLists",
            CoreObsDowngradeChange::DeprecatedRecordsRemoved { .. } => "DeprecatedRecordsRemoved",
            CoreObsDowngradeChange::EventRecordsRewritten { .. } => "EventRecordsRewritten",
        }
    }

    /// Details dictionary containing all variant-specific fields.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            CoreObsDowngradeChange::CodeRenamed { system, from, to } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("from_code", from.clone())?;
                dict.set_item("to_code", to.clone())?;
            }
            CoreObsDowngradeChange::CodeMoved {
                system,
                code,
                from,
                to,
            } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("code", code.clone())?;
                dict.set_item("from_position", *from)?;
                dict.set_item("to_position", *to)?;
            }
            CoreObsDowngradeChange::CodeAdded { system, code } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("code", code.clone())?;
            }
            CoreObsDowngradeChange::CodeListRemoved { system, codes } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("codes", codes.clone())?;
            }
            CoreObsDowngradeChange::ValueRounded {
                epoch_index,
                satellite,
                code,
                from,
                to,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("code", code.clone())?;
                dict.set_item("from_value", *from)?;
                dict.set_item("to_value", *to)?;
            }
            CoreObsDowngradeChange::CycleSlipRounded {
                epoch_index,
                satellite,
                code,
                from,
                to,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("code", code.clone())?;
                dict.set_item("from_value", *from)?;
                dict.set_item("to_value", *to)?;
            }
            CoreObsDowngradeChange::ScaleFactorsRemoved { count } => {
                dict.set_item("count", *count)?;
            }
            CoreObsDowngradeChange::EpochPicosecondsRemoved {
                epoch_index,
                picoseconds,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("picoseconds", *picoseconds)?;
            }
            CoreObsDowngradeChange::ClockOffsetRounded {
                epoch_index,
                from,
                to,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("from_offset_s", *from)?;
                dict.set_item("to_offset_s", *to)?;
            }
            CoreObsDowngradeChange::InEventLists {
                epoch_index,
                change,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                let sub = PyObsDowngradeChange {
                    inner: *change.clone(),
                };
                dict.set_item("change", sub)?;
            }
            CoreObsDowngradeChange::DeprecatedRecordsRemoved {
                label,
                epoch_index,
                records,
            } => {
                dict.set_item("label", label.clone())?;
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("records", records.clone())?;
            }
            CoreObsDowngradeChange::EventRecordsRewritten {
                epoch_index,
                from,
                to,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("from_records", from.clone())?;
                dict.set_item("to_records", to.clone())?;
            }
        }
        Ok(dict)
    }

    #[getter]
    fn epoch_index(&self) -> Option<usize> {
        match &self.inner {
            CoreObsDowngradeChange::ValueRounded { epoch_index, .. }
            | CoreObsDowngradeChange::CycleSlipRounded { epoch_index, .. }
            | CoreObsDowngradeChange::EpochPicosecondsRemoved { epoch_index, .. }
            | CoreObsDowngradeChange::ClockOffsetRounded { epoch_index, .. }
            | CoreObsDowngradeChange::InEventLists { epoch_index, .. }
            | CoreObsDowngradeChange::EventRecordsRewritten { epoch_index, .. } => {
                Some(*epoch_index)
            }
            CoreObsDowngradeChange::DeprecatedRecordsRemoved { epoch_index, .. } => *epoch_index,
            _ => None,
        }
    }

    #[getter]
    fn system(&self) -> Option<PyGnssSystem> {
        match &self.inner {
            CoreObsDowngradeChange::CodeRenamed { system, .. }
            | CoreObsDowngradeChange::CodeMoved { system, .. }
            | CoreObsDowngradeChange::CodeAdded { system, .. }
            | CoreObsDowngradeChange::CodeListRemoved { system, .. } => Some((*system).into()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("ObsDowngradeChange(kind=\"{}\")", self.kind())
    }
}

/// The typed payload a `RinexObsWriteError` carries on its `detail` attribute.
///
/// `kind` is the core variant name and `details()` that variant's own payload
/// under its own keys, so a caller reads the native values rather than parsing
/// `message`. `epoch_index`, `system`, `satellite`, `version`, `code` and
/// `time_system` are conveniences that return None for a variant that names no
/// such field.
///
/// `RinexObsWriteError` raised from anywhere but a hand-built exception always
/// carries one; the class attribute defaults to None so `except ... as e:
/// e.detail` never raises.
#[pyclass(module = "sidereon._sidereon", name = "RinexObsWriteErrorDetail")]
#[derive(Clone, Debug)]
pub struct PyRinexObsWriteErrorDetail {
    pub(crate) inner: CoreRinexObsWriteError,
}

impl From<CoreRinexObsWriteError> for PyRinexObsWriteErrorDetail {
    fn from(inner: CoreRinexObsWriteError) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexObsWriteErrorDetail {
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    /// Error kind string matching the RinexObsWriteError variant name.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            CoreRinexObsWriteError::CodeListsNotVersionTwo { .. } => "CodeListsNotVersionTwo",
            CoreRinexObsWriteError::NotVersionTwo { .. } => "NotVersionTwo",
            CoreRinexObsWriteError::ScaleFactorsInVersionTwo { .. } => "ScaleFactorsInVersionTwo",
            CoreRinexObsWriteError::ValuesWithoutCodes { .. } => "ValuesWithoutCodes",
            CoreRinexObsWriteError::CountsWithoutCodes { .. } => "CountsWithoutCodes",
            CoreRinexObsWriteError::CodeListNotStated { .. } => "CodeListNotStated",
            CoreRinexObsWriteError::EpochFlagTooWide { .. } => "EpochFlagTooWide",
            CoreRinexObsWriteError::EpochTimeMissing { .. } => "EpochTimeMissing",
            CoreRinexObsWriteError::EpochPicosecondsNotInVersion { .. } => {
                "EpochPicosecondsNotInVersion"
            }
            CoreRinexObsWriteError::TooManyObservationTypes { .. } => "TooManyObservationTypes",
            CoreRinexObsWriteError::CodeListsNotUnion { .. } => "CodeListsNotUnion",
            CoreRinexObsWriteError::ValueOutsideDeclaredList { .. } => "ValueOutsideDeclaredList",
            CoreRinexObsWriteError::DeclaredListNotStated { .. } => "DeclaredListNotStated",
            CoreRinexObsWriteError::EventRecordsUnreadable { .. } => "EventRecordsUnreadable",
            CoreRinexObsWriteError::ObservableNotRepresentable { .. } => {
                "ObservableNotRepresentable"
            }
            CoreRinexObsWriteError::LeapSecondsTimeSystemNotInVersion { .. } => {
                "LeapSecondsTimeSystemNotInVersion"
            }
            CoreRinexObsWriteError::InvalidLeapSecondsTimeSystem { .. } => {
                "InvalidLeapSecondsTimeSystem"
            }
            CoreRinexObsWriteError::ReadBackMismatch { .. } => "ReadBackMismatch",
        }
    }

    /// Formatted error message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// Dictionary containing all payload fields of this write error variant.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            CoreRinexObsWriteError::CodeListsNotVersionTwo {
                system,
                position,
                code,
            } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("position", *position)?;
                dict.set_item("code", code.clone())?;
            }
            CoreRinexObsWriteError::NotVersionTwo { version } => {
                dict.set_item("version", *version)?;
            }
            CoreRinexObsWriteError::ScaleFactorsInVersionTwo { count } => {
                dict.set_item("count", *count)?;
            }
            CoreRinexObsWriteError::ValuesWithoutCodes {
                epoch_index,
                satellite,
                codes,
                values,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("codes", *codes)?;
                dict.set_item("values", *values)?;
            }
            CoreRinexObsWriteError::CountsWithoutCodes {
                satellite,
                codes,
                counts,
            } => {
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("codes", *codes)?;
                dict.set_item("counts", *counts)?;
            }
            CoreRinexObsWriteError::CodeListNotStated { system } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
            }
            CoreRinexObsWriteError::EpochFlagTooWide { epoch_index, flag } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("flag", *flag)?;
            }
            CoreRinexObsWriteError::EpochTimeMissing { epoch_index, flag } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("flag", *flag)?;
            }
            CoreRinexObsWriteError::EpochPicosecondsNotInVersion {
                epoch_index,
                version,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("version", *version)?;
            }
            CoreRinexObsWriteError::TooManyObservationTypes { count } => {
                dict.set_item("count", *count)?;
            }
            CoreRinexObsWriteError::CodeListsNotUnion { system } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
            }
            CoreRinexObsWriteError::ValueOutsideDeclaredList {
                epoch_index,
                satellite,
                code,
            } => {
                dict.set_item("epoch_index", *epoch_index)?;
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("code", code.clone())?;
            }
            CoreRinexObsWriteError::DeclaredListNotStated { system } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
            }
            CoreRinexObsWriteError::EventRecordsUnreadable { message } => {
                dict.set_item("message", message.clone())?;
            }
            CoreRinexObsWriteError::ObservableNotRepresentable {
                system,
                code,
                version,
            } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("code", code.clone())?;
                dict.set_item("version", *version)?;
            }
            CoreRinexObsWriteError::LeapSecondsTimeSystemNotInVersion {
                time_system,
                version,
            } => {
                dict.set_item("time_system", time_system.clone())?;
                dict.set_item("version", *version)?;
            }
            CoreRinexObsWriteError::InvalidLeapSecondsTimeSystem { time_system } => {
                dict.set_item("time_system", time_system.clone())?;
            }
            CoreRinexObsWriteError::ReadBackMismatch { what } => {
                dict.set_item("what", what.clone())?;
            }
        }
        Ok(dict)
    }

    #[getter]
    fn epoch_index(&self) -> Option<usize> {
        match &self.inner {
            CoreRinexObsWriteError::ValuesWithoutCodes { epoch_index, .. }
            | CoreRinexObsWriteError::EpochFlagTooWide { epoch_index, .. }
            | CoreRinexObsWriteError::EpochTimeMissing { epoch_index, .. }
            | CoreRinexObsWriteError::EpochPicosecondsNotInVersion { epoch_index, .. }
            | CoreRinexObsWriteError::ValueOutsideDeclaredList { epoch_index, .. } => {
                Some(*epoch_index)
            }
            _ => None,
        }
    }

    #[getter]
    fn system(&self) -> Option<PyGnssSystem> {
        match &self.inner {
            CoreRinexObsWriteError::CodeListsNotVersionTwo { system, .. }
            | CoreRinexObsWriteError::CodeListNotStated { system }
            | CoreRinexObsWriteError::CodeListsNotUnion { system }
            | CoreRinexObsWriteError::DeclaredListNotStated { system }
            | CoreRinexObsWriteError::ObservableNotRepresentable { system, .. } => {
                Some((*system).into())
            }
            _ => None,
        }
    }

    #[getter]
    fn satellite(&self) -> Option<String> {
        match &self.inner {
            CoreRinexObsWriteError::ValuesWithoutCodes { satellite, .. }
            | CoreRinexObsWriteError::CountsWithoutCodes { satellite, .. }
            | CoreRinexObsWriteError::ValueOutsideDeclaredList { satellite, .. } => {
                Some(satellite.to_string())
            }
            _ => None,
        }
    }

    #[getter]
    fn version(&self) -> Option<f64> {
        match &self.inner {
            CoreRinexObsWriteError::NotVersionTwo { version }
            | CoreRinexObsWriteError::EpochPicosecondsNotInVersion { version, .. }
            | CoreRinexObsWriteError::ObservableNotRepresentable { version, .. }
            | CoreRinexObsWriteError::LeapSecondsTimeSystemNotInVersion { version, .. } => {
                Some(*version)
            }
            _ => None,
        }
    }

    /// The observation code the refusal names, for the variants that name one:
    /// `CodeListsNotVersionTwo`, `ValueOutsideDeclaredList` (None where the
    /// constellation has no list in effect) and `ObservableNotRepresentable`.
    #[getter]
    fn code(&self) -> Option<String> {
        match &self.inner {
            CoreRinexObsWriteError::CodeListsNotVersionTwo { code, .. }
            | CoreRinexObsWriteError::ValueOutsideDeclaredList { code, .. } => code.clone(),
            CoreRinexObsWriteError::ObservableNotRepresentable { code, .. } => Some(code.clone()),
            _ => None,
        }
    }

    /// The `LEAP SECONDS` time-system token the refusal names, as written.
    #[getter]
    fn time_system(&self) -> Option<&str> {
        match &self.inner {
            CoreRinexObsWriteError::LeapSecondsTimeSystemNotInVersion { time_system, .. }
            | CoreRinexObsWriteError::InvalidLeapSecondsTimeSystem { time_system } => {
                Some(time_system.as_str())
            }
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexObsWriteErrorDetail(kind=\"{}\", message={:?})",
            self.kind(),
            self.message()
        )
    }
}

fn attach_obs_write_detail(
    py: Python<'_>,
    py_err: &PyErr,
    err: CoreRinexObsWriteError,
) -> PyResult<()> {
    let py_detail = PyRinexObsWriteErrorDetail { inner: err }.into_pyobject(py)?;
    py_err.value(py).setattr("detail", py_detail)?;
    Ok(())
}

pub(crate) fn to_obs_write_err(py: Python<'_>, err: CoreRinexObsWriteError) -> PyErr {
    let msg = err.to_string();
    let py_err = RinexObsWriteError::new_err(msg);
    if let Err(e) = attach_obs_write_detail(py, &py_err, err) {
        return e;
    }
    py_err
}

/// A parsed RINEX observation product, major version 2, 3 or 4.
///
/// Epochs stay in file order, event records and cycle slip records among them,
/// so epoch indices are stable. Use `observation_values`, `carrier_phase_rows`
/// and `pseudoranges` for row-aligned numeric series, `header_at` or
/// `header_timeline` for the header in effect at an epoch, and
/// `to_rinex_string` or `downgrade_to_rinex2` to write it back.
#[pyclass(module = "sidereon._sidereon", name = "RinexObs")]
#[derive(Clone)]
pub struct PyRinexObs {
    inner: CoreRinexObs,
}

impl PyRinexObs {
    pub(crate) fn from_inner(inner: CoreRinexObs) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &CoreRinexObs {
        &self.inner
    }
}

#[pymethods]
impl PyRinexObs {
    /// Parsed RINEX OBS header.
    #[getter]
    fn header(&self) -> PyObsHeader {
        self.inner.header().clone().into()
    }

    /// Epoch records in file order.
    #[getter]
    fn epochs(&self) -> Vec<PyObsEpoch> {
        self.inner
            .epochs()
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Number of parsed epoch records.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.inner.epochs().len()
    }

    /// Number of satellite records skipped during parsing because they could not be represented.
    #[getter]
    fn skipped_records(&self) -> usize {
        self.inner.skipped_records
    }

    /// Return one epoch by zero-based index.
    fn epoch(&self, epoch_index: usize) -> PyResult<PyObsEpoch> {
        Ok(check_epoch_index(&self.inner, epoch_index)?.clone().into())
    }

    /// The header in effect at one epoch: the file header with the header
    /// records of every event at or before `epoch_index` laid over it.
    ///
    /// Raises `RinexObsParseError` when `epoch_index` is past the last epoch,
    /// and when an event's header record does not read, which a product read
    /// from text never holds. A loop over many epochs uses `header_timeline`,
    /// which reads the event records once.
    fn header_at(&self, epoch_index: usize) -> PyResult<PyObsHeader> {
        self.inner
            .header_at(epoch_index)
            .map(Into::into)
            .map_err(to_obs_err)
    }

    /// The header in effect at every epoch, built once for a loop over them.
    ///
    /// Raises `RinexObsParseError` when an event's header record does not read,
    /// which a product read from text never holds.
    fn header_timeline(&self) -> PyResult<PyObsHeaderTimeline> {
        self.inner
            .header_timeline()
            .map(Into::into)
            .map_err(to_obs_err)
    }

    /// The product's union code list for a constellation, which epoch value
    /// vectors are index-aligned to. Same as `header.obs_codes(system)`.
    fn obs_codes(&self, system: PyGnssSystem) -> Vec<String> {
        self.inner
            .obs_codes(system.into())
            .map(|codes| codes.to_vec())
            .unwrap_or_default()
    }

    /// Labelled raw observation rows for one epoch, flattened across the
    /// epoch's satellites in ascending satellite order.
    ///
    /// An event epoch and a cycle slip epoch hold no observations, so they give
    /// an empty series; an epoch's cycle slips are on `ObsEpoch.cycle_slips`,
    /// never here. Raises `ValueError` when `epoch_index` is past the last
    /// epoch.
    #[pyo3(signature = (epoch_index, filter=None))]
    fn observation_values(
        &self,
        py: Python<'_>,
        epoch_index: usize,
        filter: Option<Py<PyObservationFilter>>,
    ) -> PyResult<PyObservationValueSeries> {
        let epoch = check_epoch_index(&self.inner, epoch_index)?;
        let filter = filter_from_optional(py, filter);
        Ok(PyObservationValueSeries::from_rows(
            observation_values(&self.inner, epoch, &filter).map_err(to_obs_err)?,
        ))
    }

    /// Carrier-phase rows for one epoch, with the carrier metadata and the
    /// `SYS / PHASE SHIFT` correction the header states for each.
    ///
    /// The rows are read with the header in effect at `epoch_index`, from
    /// `header_at`, so a phase shift or GLONASS channel an event declares
    /// applies from that event's epoch on and not before it. Every row is kept,
    /// including ones whose correction is unknown or ambiguous; read
    /// `phase_shift_results` for which. An event epoch and a cycle slip epoch
    /// give an empty series.
    ///
    /// Raises `ValueError` when `epoch_index` is past the last epoch, and
    /// `RinexObsParseError` when an event's header record does not read.
    #[pyo3(signature = (epoch_index, filter=None))]
    fn carrier_phase_rows(
        &self,
        py: Python<'_>,
        epoch_index: usize,
        filter: Option<Py<PyObservationFilter>>,
    ) -> PyResult<PyCarrierPhaseSeries> {
        let epoch = check_epoch_index(&self.inner, epoch_index)?;
        let header = self.inner.header_at(epoch_index).map_err(to_obs_err)?;
        let filter = filter_from_optional(py, filter);
        Ok(PyCarrierPhaseSeries::from_rows(
            carrier_phase_rows(&header, epoch, &filter).map_err(to_obs_err)?,
        ))
    }

    /// Single-frequency pseudoranges for one epoch under a `SignalPolicy`.
    ///
    /// For each satellite, the first code in its system's preference list whose
    /// value is present at the epoch. Satellites whose system has no policy
    /// entry, or that lack every preferred code, are left out. An event epoch
    /// and a cycle slip epoch give an empty series.
    #[pyo3(signature = (epoch_index, policy=None))]
    fn pseudoranges(
        &self,
        py: Python<'_>,
        epoch_index: usize,
        policy: Option<Py<PySignalPolicy>>,
    ) -> PyResult<PyPseudorangeSeries> {
        let epoch = check_epoch_index(&self.inner, epoch_index)?;
        let policy = policy_from_optional(py, &self.inner, policy)?;
        Ok(PyPseudorangeSeries::from_rows(
            pseudoranges(&self.inner, epoch, &policy).map_err(to_obs_err)?,
        ))
    }

    /// Rewrite this product as one a RINEX 2 file of `version` can state
    /// exactly, returning `(product, changes)`.
    ///
    /// `changes` is every change the rewrite made, one `ObsDowngradeChange`
    /// each: a code renamed or moved into the single version 2 column layout, a
    /// list removed, a value or clock offset rounded to the decimals a version
    /// 2 field holds, scale factors or epoch picoseconds removed, records
    /// RINEX 4 declared to be ignored removed, an event's records rewritten.
    /// The source product is not changed.
    ///
    /// What version 2 still cannot state is refused rather than lost: raises
    /// `RinexObsWriteError`, whose `detail` names the variant and carries its
    /// payload. A code on a carrier version 2 cannot represent - BeiDou B1C
    /// against B1I, say - is refused as `ObservableNotRepresentable` rather
    /// than silently written under a version 2 name on another carrier, and a
    /// `LEAP SECONDS` time system the target version does not permit as
    /// `LeapSecondsTimeSystemNotInVersion`.
    fn downgrade_to_rinex2(
        &self,
        py: Python<'_>,
        version: f64,
    ) -> PyResult<(PyRinexObs, Vec<PyObsDowngradeChange>)> {
        let (repaired, changes) = self
            .inner
            .downgrade_to_rinex2(version)
            .map_err(|err| to_obs_write_err(py, err))?;
        Ok((
            PyRinexObs::from_inner(repaired),
            changes.into_iter().map(Into::into).collect(),
        ))
    }

    /// Serialize this product to RINEX observation text. The version the header
    /// carries decides the records written: below 3.0 the file is version 2
    /// throughout, otherwise version 3.
    ///
    /// The text is returned only when reading it back gives this product, field
    /// by field. Nothing is dropped, rounded, wrapped or truncated to make a
    /// product fit: what the text could not carry raises `RinexObsWriteError`,
    /// whose `detail` names the variant and carries its payload. A product that
    /// has to lose something to become a version 2 file goes through
    /// `downgrade_to_rinex2`, which reports every change it made.
    fn to_rinex_string(&self, py: Python<'_>) -> PyResult<String> {
        self.inner
            .to_rinex_string()
            .map_err(|err| to_obs_write_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexObs(version={}, epoch_count={}, skipped_records={})",
            self.inner.header().version,
            self.inner.epochs().len(),
            self.inner.skipped_records
        )
    }
}

/// Evaluated broadcast orbit and satellite clock at one epoch.
///
/// The position is ITRF/ECEF metres as a numpy `(3,)` array. `t_sow_s` is
/// seconds of week in the record's own broadcast time scale; `clock_s` is the
/// total satellite clock offset in seconds after the broadcast group delay.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastEvaluation")]
#[derive(Clone, Copy)]
pub struct PyBroadcastEvaluation {
    inner: SatelliteState,
    t_sow_s: f64,
}

#[pymethods]
impl PyBroadcastEvaluation {
    /// Query epoch, seconds of week in the record's broadcast time scale.
    #[getter]
    fn t_sow_s(&self) -> f64 {
        self.t_sow_s
    }

    /// ITRF/ECEF satellite position as a numpy `(3,)` array, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let position = self
            .inner
            .orbit
            .position()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .as_array();
        Ok(np_array(py, &position))
    }

    /// ITRF/ECEF X coordinate in metres.
    #[getter]
    fn x_m(&self) -> f64 {
        self.inner.orbit.x_m
    }

    /// ITRF/ECEF Y coordinate in metres.
    #[getter]
    fn y_m(&self) -> f64 {
        self.inner.orbit.y_m
    }

    /// ITRF/ECEF Z coordinate in metres.
    #[getter]
    fn z_m(&self) -> f64 {
        self.inner.orbit.z_m
    }

    /// Total satellite clock offset in seconds.
    #[getter]
    fn clock_s(&self) -> f64 {
        self.inner.clock.dt_clock_total_s
    }

    /// Broadcast clock-polynomial component in seconds.
    #[getter]
    fn clock_polynomial_s(&self) -> f64 {
        self.inner.clock.dt_clock_poly_s
    }

    /// Relativistic eccentricity clock component in seconds.
    #[getter]
    fn relativistic_clock_s(&self) -> f64 {
        self.inner.clock.dt_rel_s
    }

    /// Broadcast group delay subtracted from the clock offset, seconds.
    #[getter]
    fn group_delay_s(&self) -> f64 {
        self.inner.clock.tgd_s
    }

    /// Fixed-point iterations used to solve Kepler's equation.
    #[getter]
    fn kepler_iterations(&self) -> usize {
        self.inner.orbit.kepler_iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "BroadcastEvaluation(t_sow_s={}, clock_s={}, position_m=(3,))",
            self.t_sow_s, self.inner.clock.dt_clock_total_s
        )
    }
}

/// Broadcast Keplerian orbital elements.
///
/// Units are SI: angles in radians, correction terms in radians or metres as
/// named, and `toe_sow` in seconds of the constellation week.
#[pyclass(module = "sidereon._sidereon", name = "KeplerianElements")]
#[derive(Clone, Copy)]
pub struct PyKeplerianElements {
    inner: KeplerianElements,
}

impl From<KeplerianElements> for PyKeplerianElements {
    fn from(inner: KeplerianElements) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyKeplerianElements {
    #[getter]
    fn sqrt_a(&self) -> f64 {
        self.inner.sqrt_a
    }

    #[getter]
    fn e(&self) -> f64 {
        self.inner.e
    }

    #[getter]
    fn m0(&self) -> f64 {
        self.inner.m0
    }

    #[getter]
    fn delta_n(&self) -> f64 {
        self.inner.delta_n
    }

    #[getter]
    fn omega0(&self) -> f64 {
        self.inner.omega0
    }

    #[getter]
    fn i0(&self) -> f64 {
        self.inner.i0
    }

    #[getter]
    fn omega(&self) -> f64 {
        self.inner.omega
    }

    #[getter]
    fn omega_dot(&self) -> f64 {
        self.inner.omega_dot
    }

    #[getter]
    fn idot(&self) -> f64 {
        self.inner.idot
    }

    #[getter]
    fn cuc(&self) -> f64 {
        self.inner.cuc
    }

    #[getter]
    fn cus(&self) -> f64 {
        self.inner.cus
    }

    #[getter]
    fn crc(&self) -> f64 {
        self.inner.crc
    }

    #[getter]
    fn crs(&self) -> f64 {
        self.inner.crs
    }

    #[getter]
    fn cic(&self) -> f64 {
        self.inner.cic
    }

    #[getter]
    fn cis(&self) -> f64 {
        self.inner.cis
    }

    #[getter]
    fn toe_sow(&self) -> f64 {
        self.inner.toe_sow
    }

    fn __repr__(&self) -> String {
        format!(
            "KeplerianElements(sqrt_a={}, e={}, toe_sow={})",
            self.inner.sqrt_a, self.inner.e, self.inner.toe_sow
        )
    }
}

/// Broadcast satellite-clock polynomial.
///
/// `af0`, `af1`, and `af2` are seconds, seconds per second, and seconds per
/// second squared. `toc_sow` is seconds of the constellation week.
#[pyclass(module = "sidereon._sidereon", name = "ClockPolynomial")]
#[derive(Clone, Copy)]
pub struct PyClockPolynomial {
    inner: ClockPolynomial,
}

impl From<ClockPolynomial> for PyClockPolynomial {
    fn from(inner: ClockPolynomial) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyClockPolynomial {
    #[getter]
    fn af0(&self) -> f64 {
        self.inner.af0
    }

    #[getter]
    fn af1(&self) -> f64 {
        self.inner.af1
    }

    #[getter]
    fn af2(&self) -> f64 {
        self.inner.af2
    }

    #[getter]
    fn toc_sow(&self) -> f64 {
        self.inner.toc_sow
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockPolynomial(af0={}, af1={}, af2={}, toc_sow={})",
            self.inner.af0, self.inner.af1, self.inner.af2, self.inner.toc_sow
        )
    }
}

/// One GPS, Galileo, or BeiDou broadcast ephemeris record from RINEX NAV.
///
/// The Keplerian elements and clock polynomial use SI units. Call `evaluate`
/// with seconds of week to compute the ECEF position and satellite clock.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastRecord")]
#[derive(Clone, Copy)]
pub struct PyBroadcastRecord {
    inner: BroadcastRecord,
}

impl From<BroadcastRecord> for PyBroadcastRecord {
    fn from(inner: BroadcastRecord) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyBroadcastRecord {
    /// RINEX satellite token such as `"G01"`.
    #[getter]
    fn satellite(&self) -> String {
        satellite_token(self.inner.satellite_id)
    }

    /// Broadcast message type.
    #[getter]
    fn message(&self) -> PyNavMessage {
        self.inner.message.into()
    }

    /// Continuous constellation week number from the broadcast record.
    #[getter]
    fn week(&self) -> u32 {
        self.inner.week
    }

    /// Keplerian orbital elements in SI units.
    #[getter]
    fn elements(&self) -> PyKeplerianElements {
        self.inner.elements.into()
    }

    /// Satellite clock polynomial.
    #[getter]
    fn clock(&self) -> PyClockPolynomial {
        self.inner.clock.into()
    }

    /// Broadcast group delay in seconds.
    #[getter]
    fn group_delay_s(&self) -> f64 {
        self.inner.broadcast_clock_group_delay_s()
    }

    /// Satellite health word, where 0 is healthy for nominal GPS/Galileo.
    #[getter]
    fn sv_health(&self) -> f64 {
        self.inner.sv_health
    }

    /// Signal-in-space accuracy in metres: the stated URA, SISA or BeiDou
    /// URA, or the nominal value of the CNAV URA_ED index. `None` for a blank
    /// or unreadable field and for the CNAV indices that carry no accuracy
    /// prediction.
    #[getter]
    fn sv_accuracy_m(&self) -> Option<f64> {
        self.inner.sv_accuracy_m
    }

    /// GPS/QZSS IODC (BROADCAST ORBIT-6 field 4), as stated.
    #[getter]
    fn iodc(&self) -> Option<f64> {
        self.inner.iodc()
    }

    /// GPS/QZSS codes on L2 (BROADCAST ORBIT-5 field 2), as stated.
    #[getter]
    fn l2_codes(&self) -> Option<f64> {
        self.inner.l2_codes()
    }

    /// GPS/QZSS L2 P data flag (BROADCAST ORBIT-5 field 4), as stated.
    #[getter]
    fn l2p_data_flag(&self) -> Option<f64> {
        self.inner.l2p_data_flag()
    }

    /// The Galileo data-source word (BROADCAST ORBIT-5 field 2) as the
    /// nearest integer.
    #[getter]
    fn galileo_data_sources(&self) -> Option<u32> {
        self.inner.galileo_data_sources()
    }

    /// BeiDou AODC (BROADCAST ORBIT-7 field 2), as stated.
    #[getter]
    fn beidou_aodc(&self) -> Option<f64> {
        self.inner.beidou_aodc()
    }

    /// Transmission time of message, seconds of week: the CNAV `t_tm`, or
    /// BROADCAST ORBIT-7 field 1 as stated.
    #[getter]
    fn transmission_time_sow(&self) -> Option<f64> {
        self.inner.transmission_time_sow()
    }

    /// Every BROADCAST ORBIT field the orbit and clock models do not read,
    /// as stated, keyed by field (`orbit5_field2`, `orbit5_field4`,
    /// `orbit6_field4`, `transmission_time_sow`, `orbit7_field2`,
    /// `orbit7_field3`, `orbit7_field4`); `None` for a blank field.
    #[getter]
    fn stated_fields(&self) -> BTreeMap<&'static str, Option<f64>> {
        let stated = self.inner.stated;
        BTreeMap::from([
            ("orbit5_field2", stated.orbit5_field2),
            ("orbit5_field4", stated.orbit5_field4),
            ("orbit6_field4", stated.orbit6_field4),
            ("transmission_time_sow", stated.transmission_time_sow),
            ("orbit7_field2", stated.orbit7_field2),
            ("orbit7_field3", stated.orbit7_field3),
            ("orbit7_field4", stated.orbit7_field4),
        ])
    }

    /// GPS curve-fit interval in seconds, or `None` when not broadcast.
    #[getter]
    fn fit_interval_s(&self) -> Option<f64> {
        self.inner.fit_interval_s
    }

    /// Broadcast time scale for this record.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale().into()
    }

    /// Native issue-of-data value, `None` for a GPS/QZSS CNAV-family record,
    /// which states no issue of data.
    #[getter]
    fn issue(&self) -> Option<u32> {
        self.inner.issue_of_data.map(|issue| issue.issue)
    }

    /// Navigation message associated with the issue-of-data value, `None`
    /// when the record states no issue of data.
    #[getter]
    fn issue_message(&self) -> Option<PyNavMessage> {
        self.inner.issue_of_data.map(|issue| issue.message.into())
    }

    /// Ephemeris reference week number.
    #[getter]
    fn toe_week(&self) -> u32 {
        self.inner.toe.week
    }

    /// Ephemeris reference seconds of week.
    #[getter]
    fn toe_tow_s(&self) -> f64 {
        self.inner.toe.tow_s
    }

    /// Clock reference week number.
    #[getter]
    fn toc_week(&self) -> u32 {
        self.inner.toc.week
    }

    /// Clock reference seconds of week.
    #[getter]
    fn toc_tow_s(&self) -> f64 {
        self.inner.toc.tow_s
    }

    /// Full record group-delay set.
    #[getter]
    fn group_delays(&self) -> PyBroadcastGroupDelays {
        self.inner.group_delays.into()
    }

    /// CNAV/CNAV-2 extension, if this is a CNAV-family record.
    #[getter]
    fn cnav(&self) -> Option<PyCnavParameters> {
        self.inner.cnav.map(Into::into)
    }

    /// Whether this record is a GPS/QZSS CNAV-family message.
    #[getter]
    fn is_cnav_family(&self) -> bool {
        self.inner.message.is_cnav_family()
    }

    /// CNAV single-frequency clock correction `TGD - ISC`, if available.
    fn cnav_single_frequency_correction_s(&self, signal: PyCnavSignal) -> Option<f64> {
        self.inner
            .group_delays
            .cnav_single_frequency_correction_s(signal.into())
    }

    /// Evaluate the broadcast record at a seconds-of-week epoch.
    ///
    /// `t_sow_s` is seconds of week in this record's broadcast time scale
    /// (GPS/Galileo system time for G/E records, BDT for BeiDou). The result is
    /// ITRF/ECEF metres and satellite clock seconds.
    fn evaluate(&self, t_sow_s: f64) -> PyResult<PyBroadcastEvaluation> {
        if !t_sow_s.is_finite() {
            return Err(PyValueError::new_err("t_sow_s must be finite"));
        }
        let inner = if let Some(cnav) = self.inner.cnav {
            satellite_state_cnav(
                &self.inner.elements,
                &CnavRates {
                    adot_m_s: cnav.adot_m_s,
                    delta_n0_dot_rad_s2: cnav.delta_n0_dot_rad_s2,
                },
                &self.inner.clock,
                &self.inner.constants(),
                t_sow_s,
                self.inner.broadcast_clock_group_delay_s(),
            )
        } else {
            satellite_state(
                &self.inner.elements,
                &self.inner.clock,
                &self.inner.constants(),
                t_sow_s,
                self.inner.broadcast_clock_group_delay_s(),
                is_beidou_geo(self.inner.satellite_id),
            )
        }
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(PyBroadcastEvaluation { inner, t_sow_s })
    }

    fn __repr__(&self) -> String {
        let message = PyNavMessage::from(self.inner.message);
        format!(
            "BroadcastRecord(satellite='{}', message={}, week={})",
            self.inner.satellite_id,
            message.label(),
            self.inner.week
        )
    }
}

/// One GLONASS broadcast state-vector record.
///
/// `toe_utc_j2000_s` is UTC seconds past J2000. Position is PZ-90.11 ECEF
/// metres, velocity is metres per second, and acceleration is metres per second
/// squared.
#[pyclass(module = "sidereon._sidereon", name = "GlonassRecord")]
#[derive(Clone, Copy)]
pub struct PyGlonassRecord {
    inner: GlonassRecord,
}

impl From<GlonassRecord> for PyGlonassRecord {
    fn from(inner: GlonassRecord) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGlonassRecord {
    /// RINEX satellite token such as `"R10"`.
    #[getter]
    fn satellite(&self) -> String {
        satellite_token(self.inner.satellite_id)
    }

    /// Reference epoch in UTC seconds past J2000.
    #[getter]
    fn toe_utc_j2000_s(&self) -> f64 {
        self.inner.toe_utc_j2000_s
    }

    /// PZ-90.11 ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.pos_m)
    }

    /// PZ-90.11 ECEF velocity as a numpy `(3,)` array, metres per second.
    #[getter]
    fn velocity_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.vel_m_s)
    }

    /// Lunisolar acceleration as a numpy `(3,)` array, metres per second squared.
    #[getter]
    fn acceleration_m_s2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.acc_m_s2)
    }

    /// Broadcast clock bias in seconds.
    #[getter]
    fn clock_bias_s(&self) -> f64 {
        self.inner.clk_bias
    }

    /// Relative frequency offset.
    #[getter]
    fn gamma_n(&self) -> f64 {
        self.inner.gamma_n
    }

    /// Satellite health, where 0 is healthy.
    #[getter]
    fn sv_health(&self) -> f64 {
        self.inner.sv_health
    }

    /// FDMA frequency-channel number. A stated value above 128 reads as that
    /// value less 256, as RTKLIB `decode_geph` reads it.
    #[getter]
    fn freq_channel(&self) -> i32 {
        self.inner.freq_channel
    }

    /// The frequency channel field as stated, before a value above 128 is
    /// folded into `freq_channel`.
    #[getter]
    fn stated_freq_channel(&self) -> i32 {
        self.inner.stated_freq_channel
    }

    /// The epoch as the record states it, UTC seconds past J2000;
    /// `toe_utc_j2000_s` is it rounded to the 15-minute grid.
    #[getter]
    fn epoch_utc_j2000_s(&self) -> f64 {
        self.inner.epoch_utc_j2000_s
    }

    /// The reference epoch in GPS time, seconds past J2000, converted with the
    /// leap-second table at that UTC instant.
    #[getter]
    fn toe_gpst_j2000_s(&self) -> f64 {
        self.inner.toe_gpst_j2000_s()
    }

    /// Message frame time as stated: seconds of the UTC week in RINEX 3 and 4,
    /// of the UTC day in RINEX 2. `None` when blank.
    #[getter]
    fn message_frame_time_s(&self) -> Option<f64> {
        self.inner.message_frame_time_s
    }

    /// Age of operational information `E_n`, days.
    #[getter]
    fn age_days(&self) -> Option<f64> {
        self.inner.age_days
    }

    /// Status flags (RINEX 3.05 and 4 BROADCAST ORBIT-4 field 1) as stated.
    #[getter]
    fn status_flags(&self) -> Option<f64> {
        self.inner.status_flags
    }

    /// The status flags as the nearest integer word; `None` when blank or not
    /// known.
    #[getter]
    fn status_flags_word(&self) -> Option<u32> {
        self.inner.status_flags_word()
    }

    /// L1/L2 group delay difference field as stated, seconds, including the
    /// not-known value.
    #[getter]
    fn l1_l2_group_delay_field_s(&self) -> Option<f64> {
        self.inner.l1_l2_group_delay_field_s
    }

    /// L1/L2 group delay difference in seconds, `None` when blank or stated
    /// as not known.
    #[getter]
    fn l1_l2_group_delay_s(&self) -> Option<f64> {
        self.inner.l1_l2_group_delay_s()
    }

    /// Raw accuracy index URAI `F_T` as stated.
    #[getter]
    fn urai(&self) -> Option<f64> {
        self.inner.urai
    }

    /// Health flags (BROADCAST ORBIT-4 field 4) as stated.
    #[getter]
    fn health_flags(&self) -> Option<f64> {
        self.inner.health_flags
    }

    /// The health flags as the nearest integer word.
    #[getter]
    fn health_flags_word(&self) -> Option<u32> {
        self.inner.health_flags_word()
    }

    /// Whether the record reports the satellite healthy by the RINEX 3.05
    /// Table A10 (4.01 Table A15) health fields.
    #[getter]
    fn is_healthy(&self) -> bool {
        self.inner.is_healthy()
    }

    /// RTKLIB `prange`'s G1 single-frequency group delay `-ΔτN / (γ - 1)`,
    /// seconds, `None` without a stated delay.
    #[getter]
    fn single_frequency_group_delay_s(&self) -> Option<f64> {
        self.inner.single_frequency_group_delay_s()
    }

    /// RTKLIB `geph2clk`: the clock bias at a satellite clock time given as
    /// GPS seconds past J2000.
    fn clock_bias_at(&self, t_sv_gpst_j2000_s: f64) -> f64 {
        self.inner.clock_bias_s(t_sv_gpst_j2000_s)
    }

    fn __repr__(&self) -> String {
        format!(
            "GlonassRecord(satellite='{}', toe_utc_j2000_s={})",
            self.inner.satellite_id, self.inner.toe_utc_j2000_s
        )
    }
}

/// Klobuchar alpha and beta ionosphere coefficients.
///
/// `alpha` and `beta` are numpy `(4,)` arrays in the units broadcast by the
/// RINEX NAV header.
#[pyclass(module = "sidereon._sidereon", name = "KlobucharAlphaBeta")]
#[derive(Clone, Copy)]
pub struct PyKlobucharAlphaBeta {
    inner: KlobucharAlphaBeta,
}

impl From<KlobucharAlphaBeta> for PyKlobucharAlphaBeta {
    fn from(inner: KlobucharAlphaBeta) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyKlobucharAlphaBeta {
    /// Alpha coefficients as a numpy `(4,)` array.
    #[getter]
    fn alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.alpha)
    }

    /// Beta coefficients as a numpy `(4,)` array.
    #[getter]
    fn beta<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.beta)
    }

    fn __repr__(&self) -> &'static str {
        "KlobucharAlphaBeta(alpha=(4,), beta=(4,))"
    }
}

/// Broadcast ionosphere coefficients parsed from a RINEX NAV header.
///
/// GPS and BeiDou Klobuchar-8 coefficient sets are exposed independently. A
/// missing header pair is returned as `None`.
#[pyclass(module = "sidereon._sidereon", name = "IonoCorrections")]
#[derive(Clone, Copy)]
pub struct PyIonoCorrections {
    inner: IonoCorrections,
}

impl From<IonoCorrections> for PyIonoCorrections {
    fn from(inner: IonoCorrections) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIonoCorrections {
    /// GPS Klobuchar coefficients, if the header has GPSA and GPSB.
    #[getter]
    fn gps(&self) -> Option<PyKlobucharAlphaBeta> {
        self.inner.gps.map(Into::into)
    }

    /// BeiDou Klobuchar coefficients, if the header has BDSA and BDSB.
    #[getter]
    fn beidou(&self) -> Option<PyKlobucharAlphaBeta> {
        self.inner.beidou.map(Into::into)
    }

    fn __repr__(&self) -> String {
        format!(
            "IonoCorrections(gps={}, beidou={})",
            self.inner.gps.is_some(),
            self.inner.beidou.is_some()
        )
    }
}

/// Parsed broadcast ephemeris store from a RINEX NAV file.
///
/// `records` contains healthy GPS LNAV, Galileo I/NAV, and BeiDou D1/D2 records
/// selected by the core's default SPP policy. `glonass_records` contains healthy
/// GLONASS state-vector records. Epochs are J2000 seconds where named.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastEphemeris")]
pub struct PyBroadcastEphemeris {
    pub(crate) inner: CoreBroadcastEphemeris,
    leap_seconds: Option<f64>,
}

#[pymethods]
impl PyBroadcastEphemeris {
    /// Usable GPS, Galileo, and BeiDou broadcast records in file order.
    #[getter]
    fn records(&self) -> Vec<PyBroadcastRecord> {
        self.inner
            .records()
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// Healthy GLONASS broadcast records in file order.
    #[getter]
    fn glonass_records(&self) -> Vec<PyGlonassRecord> {
        self.inner
            .glonass_records()
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// Broadcast ionosphere coefficients parsed from the NAV header.
    #[getter]
    fn iono_corrections(&self) -> PyIonoCorrections {
        self.inner.iono_corrections().into()
    }

    /// GPS minus UTC leap seconds from the NAV header, if present.
    #[getter]
    fn leap_seconds(&self) -> Option<f64> {
        self.leap_seconds
    }

    fn selected_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<crate::ephemeris::PyEphemerisQueryState>> {
        crate::ephemeris::source_state_at_epoch_query(
            &self.inner,
            satellite,
            epoch,
            selection_epoch,
        )
    }

    fn transmit_epoch_clock_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<Option<(f64, Option<&'static str>)>> {
        crate::ephemeris::source_transmit_clock_at_epoch_query(
            &self.inner,
            satellite,
            epoch,
            selection_epoch,
        )
    }

    fn ephemeris_variance_at_epoch_query(
        &self,
        satellite: &str,
        state_epoch: &PyExactEpochQuery,
        selection_epoch: &PyExactEpochQuery,
    ) -> PyResult<f64> {
        crate::ephemeris::source_variance_at_epoch_query(
            &self.inner,
            satellite,
            state_epoch,
            selection_epoch,
        )
    }

    fn clock_relativity_for_state_at_epoch_query(
        &self,
        satellite: &str,
        epoch: &PyExactEpochQuery,
        position_ecef_m: [f64; 3],
    ) -> PyResult<crate::ephemeris::PyClockRelativity> {
        crate::ephemeris::source_clock_relativity_at_epoch_query(
            &self.inner,
            satellite,
            epoch,
            position_ecef_m,
        )
    }

    /// GPS/QZSS message-family selection preference.
    #[getter]
    fn message_preference(&self) -> PyNavMessagePreference {
        self.inner.message_preference().into()
    }

    /// Set the GPS/QZSS legacy-vs-CNAV selection preference.
    fn set_message_preference(&mut self, preference: PyNavMessagePreference) {
        self.inner.set_message_preference(preference.into());
    }

    /// GLONASS FDMA channels from retained broadcast records, keyed by slot.
    #[getter]
    fn glonass_frequency_channels(&self) -> BTreeMap<u8, i8> {
        self.inner.glonass_frequency_channels()
    }

    /// Number of usable GPS, Galileo, and BeiDou records.
    #[getter]
    fn record_count(&self) -> usize {
        self.inner.records().len()
    }

    /// Number of usable GLONASS records.
    #[getter]
    fn glonass_record_count(&self) -> usize {
        self.inner.glonass_records().len()
    }

    fn __repr__(&self) -> String {
        format!(
            "BroadcastEphemeris(record_count={}, glonass_record_count={})",
            self.inner.records().len(),
            self.inner.glonass_records().len()
        )
    }
}

/// A malformed supported NAV block skipped by lenient parsing.
#[pyclass(module = "sidereon._sidereon", name = "SkippedNavBlock")]
#[derive(Clone)]
pub struct PySkippedNavBlock {
    inner: SkippedNavBlock,
}

impl From<SkippedNavBlock> for PySkippedNavBlock {
    fn from(inner: SkippedNavBlock) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySkippedNavBlock {
    #[getter]
    fn satellite(&self) -> &str {
        &self.inner.satellite
    }

    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }

    /// 1-based line of the block's first line.
    #[getter]
    fn line(&self) -> usize {
        self.inner.line
    }

    fn __repr__(&self) -> String {
        format!(
            "SkippedNavBlock(satellite={:?}, message={:?})",
            self.inner.satellite, self.inner.message
        )
    }
}

/// A departure from the RINEX NAV format that a lenient read read through.
#[pyclass(module = "sidereon._sidereon", name = "NavDiagnostic")]
#[derive(Clone)]
pub struct PyNavDiagnostic {
    line: usize,
    satellite: String,
    message: String,
}

#[pymethods]
impl PyNavDiagnostic {
    /// 1-based line of the departure.
    #[getter]
    fn line(&self) -> usize {
        self.line
    }

    /// Satellite token of the block, empty for a header line.
    #[getter]
    fn satellite(&self) -> &str {
        &self.satellite
    }

    /// The departure as the core states it.
    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    fn __repr__(&self) -> String {
        format!(
            "NavDiagnostic(line={}, satellite={:?}, message={:?})",
            self.line, self.satellite, self.message
        )
    }
}

/// A NAV block of a kind `parse_rinex_nav_lenient` does not return as a
/// broadcast record: GLONASS, SBAS, system time offset, Earth orientation,
/// ionosphere, or a recognized frame that is not decoded.
#[pyclass(module = "sidereon._sidereon", name = "OtherNavBlock")]
#[derive(Clone)]
pub struct PyOtherNavBlock {
    inner: sidereon_core::rinex::nav::OtherNavBlock,
}

#[pymethods]
impl PyOtherNavBlock {
    /// 1-based line of the block's first line.
    #[getter]
    fn line(&self) -> usize {
        self.inner.line
    }

    /// Satellite token of the block.
    #[getter]
    fn satellite(&self) -> &str {
        &self.inner.satellite
    }

    /// The RINEX 4 message token of the block's frame, when it has one.
    #[getter]
    fn message_token(&self) -> Option<&str> {
        self.inner.message_token.as_deref()
    }

    /// The block kind: `glonass`, `sbas`, `system_time_offset`,
    /// `earth_orientation`, `ionosphere` or `not_decoded`.
    #[getter]
    fn kind(&self) -> &'static str {
        use sidereon_core::rinex::nav::OtherNavBlockKind;
        match self.inner.kind {
            OtherNavBlockKind::Glonass => "glonass",
            OtherNavBlockKind::Sbas => "sbas",
            OtherNavBlockKind::SystemTimeOffset => "system_time_offset",
            OtherNavBlockKind::EarthOrientation => "earth_orientation",
            OtherNavBlockKind::Ionosphere => "ionosphere",
            OtherNavBlockKind::NotDecoded => "not_decoded",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OtherNavBlock(line={}, satellite={:?}, kind={:?})",
            self.inner.line,
            self.inner.satellite,
            self.kind()
        )
    }
}

/// Result of lenient RINEX NAV parsing.
#[pyclass(module = "sidereon._sidereon", name = "RinexNavParse")]
#[derive(Clone)]
pub struct PyRinexNavParse {
    inner: NavParse,
}

#[pymethods]
impl PyRinexNavParse {
    #[getter]
    fn records(&self) -> Vec<PyBroadcastRecord> {
        self.inner.records.iter().copied().map(Into::into).collect()
    }

    #[getter]
    fn skipped(&self) -> Vec<PySkippedNavBlock> {
        self.inner.skipped.iter().cloned().map(Into::into).collect()
    }

    #[getter]
    fn record_count(&self) -> usize {
        self.inner.records.len()
    }

    #[getter]
    fn skipped_count(&self) -> usize {
        self.inner.skipped.len()
    }

    /// Departures from the format the lenient read read through, each with
    /// its line.
    #[getter]
    fn departures(&self) -> Vec<PyNavDiagnostic> {
        self.inner
            .departures
            .iter()
            .map(|diagnostic| PyNavDiagnostic {
                line: diagnostic.line,
                satellite: diagnostic.satellite.clone(),
                message: diagnostic.error.to_string(),
            })
            .collect()
    }

    /// Blocks of the kinds this parse does not return as broadcast records.
    #[getter]
    fn other(&self) -> Vec<PyOtherNavBlock> {
        self.inner
            .other
            .iter()
            .cloned()
            .map(|inner| PyOtherNavBlock { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexNavParse(record_count={}, skipped_count={})",
            self.inner.records.len(),
            self.inner.skipped.len()
        )
    }
}

/// RINEX lint severity.
#[pyclass(module = "sidereon._sidereon", name = "RinexLintSeverity", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyRinexLintSeverity {
    FATAL,
    ERROR,
    WARNING,
    INFO,
}

impl From<CoreRinexSeverity> for PyRinexLintSeverity {
    fn from(value: CoreRinexSeverity) -> Self {
        match value {
            CoreRinexSeverity::Fatal => Self::FATAL,
            CoreRinexSeverity::Error => Self::ERROR,
            CoreRinexSeverity::Warning => Self::WARNING,
            CoreRinexSeverity::Info => Self::INFO,
        }
    }
}

impl From<PyRinexLintSeverity> for CoreRinexSeverity {
    fn from(value: PyRinexLintSeverity) -> Self {
        match value {
            PyRinexLintSeverity::FATAL => Self::Fatal,
            PyRinexLintSeverity::ERROR => Self::Error,
            PyRinexLintSeverity::WARNING => Self::Warning,
            PyRinexLintSeverity::INFO => Self::Info,
        }
    }
}

#[pymethods]
impl PyRinexLintSeverity {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::FATAL => "fatal",
            Self::ERROR => "error",
            Self::WARNING => "warning",
            Self::INFO => "info",
        }
    }
}

/// Source location for a RINEX lint finding.
#[pyclass(module = "sidereon._sidereon", name = "RinexFindingRef")]
#[derive(Clone)]
pub struct PyRinexFindingRef {
    inner: CoreFindingRef,
}

impl From<CoreFindingRef> for PyRinexFindingRef {
    fn from(inner: CoreFindingRef) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexFindingRef {
    #[getter]
    fn epoch_index(&self) -> Option<usize> {
        self.inner.epoch_index
    }

    #[getter]
    fn satellite(&self) -> Option<String> {
        self.inner.satellite.clone()
    }

    #[getter]
    fn field(&self) -> Option<&'static str> {
        self.inner.field
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexFindingRef(epoch_index={:?}, satellite={:?}, field={:?})",
            self.inner.epoch_index, self.inner.satellite, self.inner.field
        )
    }
}

fn finding_kind(finding: &CoreRinexFinding) -> &'static str {
    match finding {
        CoreRinexFinding::ObsFatalParse { .. } => "ObsFatalParse",
        CoreRinexFinding::ObsUnpublishedVersion { .. } => "ObsUnpublishedVersion",
        CoreRinexFinding::ObsMissingHeader { .. } => "ObsMissingHeader",
        CoreRinexFinding::ObsMissingObsTypes { .. } => "ObsMissingObsTypes",
        CoreRinexFinding::ObsInvalidObsCode { .. } => "ObsInvalidObsCode",
        CoreRinexFinding::ObsDuplicateObsCode { .. } => "ObsDuplicateObsCode",
        CoreRinexFinding::ObsTimeOfFirstMismatch { .. } => "ObsTimeOfFirstMismatch",
        CoreRinexFinding::ObsTimeOfLastMismatch { .. } => "ObsTimeOfLastMismatch",
        CoreRinexFinding::ObsIntervalMismatch { .. } => "ObsIntervalMismatch",
        CoreRinexFinding::ObsIntervalUnavailable { .. } => "ObsIntervalUnavailable",
        CoreRinexFinding::ObsInvalidInterval { .. } => "ObsInvalidInterval",
        CoreRinexFinding::ObsSatelliteCountMismatch { .. } => "ObsSatelliteCountMismatch",
        CoreRinexFinding::ObsPrnObsCountMismatch { .. } => "ObsPrnObsCountMismatch",
        CoreRinexFinding::ObsGlonassSlotIssue { .. } => "ObsGlonassSlotIssue",
        CoreRinexFinding::ObsPhaseShiftUndeclaredCode { .. } => "ObsPhaseShiftUndeclaredCode",
        CoreRinexFinding::ObsScaleFactorIssue { .. } => "ObsScaleFactorIssue",
        CoreRinexFinding::ObsMarkerTypeIssue { .. } => "ObsMarkerTypeIssue",
        CoreRinexFinding::ObsIdentityFieldIssue { .. } => "ObsIdentityFieldIssue",
        CoreRinexFinding::ObsImplausibleApproxPosition { .. } => "ObsImplausibleApproxPosition",
        CoreRinexFinding::ObsImplausibleAntennaDelta { .. } => "ObsImplausibleAntennaDelta",
        CoreRinexFinding::ObsEpochOrder { .. } => "ObsEpochOrder",
        CoreRinexFinding::ObsDuplicateEpoch { .. } => "ObsDuplicateEpoch",
        CoreRinexFinding::ObsSkippedRecords { .. } => "ObsSkippedRecords",
        CoreRinexFinding::ObsEpochSatCountMismatch { .. } => "ObsEpochSatCountMismatch",
        CoreRinexFinding::ObsEventHeaderUnreadable { .. } => "ObsEventHeaderUnreadable",
        CoreRinexFinding::ObsUnretainedHeader { .. } => "ObsUnretainedHeader",
        CoreRinexFinding::ObsPseudorangeOutOfRange { .. } => "ObsPseudorangeOutOfRange",
        CoreRinexFinding::ObsLossOfLockOutOfRange { .. } => "ObsLossOfLockOutOfRange",
        CoreRinexFinding::ObsEventEpoch { .. } => "ObsEventEpoch",
        CoreRinexFinding::ObsEmptySatelliteRecord { .. } => "ObsEmptySatelliteRecord",
        CoreRinexFinding::ObsEpochGap { .. } => "ObsEpochGap",
        CoreRinexFinding::NavFatalParse { .. } => "NavFatalParse",
        CoreRinexFinding::NavLeapSecondsAbsent { .. } => "NavLeapSecondsAbsent",
        CoreRinexFinding::NavIonoMalformed { .. } => "NavIonoMalformed",
        CoreRinexFinding::NavDroppedBlock { .. } => "NavDroppedBlock",
        CoreRinexFinding::NavDuplicateRecord { .. } => "NavDuplicateRecord",
        CoreRinexFinding::NavUnsortedRecords { .. } => "NavUnsortedRecords",
        CoreRinexFinding::NavImplausibleRecord { .. } => "NavImplausibleRecord",
        CoreRinexFinding::NavUnhealthyRecords { .. } => "NavUnhealthyRecords",
        CoreRinexFinding::NavOutOfScopeRecords { .. } => "NavOutOfScopeRecords",
        _ => "Unknown",
    }
}

/// One RINEX lint finding from the core QC suite.
#[pyclass(module = "sidereon._sidereon", name = "RinexLintFinding")]
#[derive(Clone)]
pub struct PyRinexLintFinding {
    inner: CoreRinexFinding,
}

impl From<CoreRinexFinding> for PyRinexLintFinding {
    fn from(inner: CoreRinexFinding) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexLintFinding {
    #[getter]
    fn kind(&self) -> &'static str {
        finding_kind(&self.inner)
    }

    #[getter]
    fn code(&self) -> &'static str {
        self.inner.code()
    }

    #[getter]
    fn severity(&self) -> PyRinexLintSeverity {
        self.inner.severity().into()
    }

    #[getter]
    fn spec_ref(&self) -> &'static str {
        self.inner.spec_ref()
    }

    #[getter]
    fn at(&self) -> PyRinexFindingRef {
        self.inner.at().clone().into()
    }

    #[getter]
    fn is_repairable(&self) -> bool {
        self.inner.is_repairable()
    }

    #[getter]
    fn detail(&self) -> String {
        format!("{:?}", self.inner)
    }

    /// Details dictionary preserving exact native payloads for every finding variant.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            CoreRinexFinding::ObsFatalParse { message, .. } => {
                dict.set_item("message", message.clone())?;
            }
            CoreRinexFinding::ObsUnpublishedVersion { version, .. } => {
                dict.set_item("version", *version)?;
            }
            CoreRinexFinding::ObsMissingHeader { label, .. } => {
                dict.set_item("label", *label)?;
            }
            CoreRinexFinding::ObsMissingObsTypes { .. } => {}
            CoreRinexFinding::ObsInvalidObsCode { system, code, .. } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("code", code.clone())?;
            }
            CoreRinexFinding::ObsDuplicateObsCode { system, code, .. } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("code", code.clone())?;
            }
            CoreRinexFinding::ObsTimeOfFirstMismatch {
                declared,
                declared_scale,
                observed,
                observed_scale,
                ..
            } => {
                dict.set_item("declared", PyObsEpochTime::from(*declared))?;
                dict.set_item("declared_scale", PyTimeScale::from(*declared_scale))?;
                dict.set_item("observed", PyObsEpochTime::from(*observed))?;
                dict.set_item("observed_scale", PyTimeScale::from(*observed_scale))?;
            }
            CoreRinexFinding::ObsTimeOfLastMismatch {
                declared,
                declared_scale,
                observed,
                observed_scale,
                ..
            } => {
                dict.set_item("declared", PyObsEpochTime::from(*declared))?;
                dict.set_item("declared_scale", PyTimeScale::from(*declared_scale))?;
                dict.set_item("observed", PyObsEpochTime::from(*observed))?;
                dict.set_item("observed_scale", PyTimeScale::from(*observed_scale))?;
            }
            CoreRinexFinding::ObsIntervalMismatch {
                declared_s,
                observed_s,
                ..
            } => {
                dict.set_item("declared_s", *declared_s)?;
                dict.set_item("observed_s", *observed_s)?;
            }
            CoreRinexFinding::ObsSatelliteCountMismatch {
                declared, observed, ..
            } => {
                dict.set_item("declared", *declared)?;
                dict.set_item("observed", *observed)?;
            }
            CoreRinexFinding::ObsPrnObsCountMismatch {
                satellite,
                code,
                declared,
                observed,
                ..
            } => {
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("code", code.clone())?;
                dict.set_item("declared", *declared)?;
                dict.set_item("observed", *observed)?;
            }
            CoreRinexFinding::ObsGlonassSlotIssue {
                satellite, issue, ..
            } => {
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("issue", *issue)?;
            }
            CoreRinexFinding::ObsPhaseShiftUndeclaredCode { system, code, .. } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("code", code.clone())?;
            }
            CoreRinexFinding::ObsScaleFactorIssue { system, code, .. } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("code", code.clone())?;
            }
            CoreRinexFinding::ObsMarkerTypeIssue { marker_type, .. } => {
                dict.set_item("marker_type", marker_type.clone())?;
            }
            CoreRinexFinding::ObsIdentityFieldIssue { label, value, .. } => {
                dict.set_item("label", *label)?;
                dict.set_item("value", value.clone())?;
            }
            CoreRinexFinding::ObsImplausibleApproxPosition { radius_m, .. } => {
                dict.set_item("radius_m", *radius_m)?;
            }
            CoreRinexFinding::ObsImplausibleAntennaDelta {
                component, value_m, ..
            } => {
                dict.set_item("component", *component)?;
                dict.set_item("value_m", *value_m)?;
            }
            CoreRinexFinding::ObsEpochOrder {
                previous, current, ..
            } => {
                dict.set_item("previous", PyObsEpochTime::from(*previous))?;
                dict.set_item("current", PyObsEpochTime::from(*current))?;
            }
            CoreRinexFinding::ObsDuplicateEpoch { epoch, .. } => {
                dict.set_item("epoch", PyObsEpochTime::from(*epoch))?;
            }
            CoreRinexFinding::ObsSkippedRecords { count, .. } => {
                dict.set_item("count", *count)?;
            }
            CoreRinexFinding::ObsEpochSatCountMismatch {
                declared, retained, ..
            } => {
                dict.set_item("declared", *declared)?;
                dict.set_item("retained", *retained)?;
            }
            CoreRinexFinding::ObsUnretainedHeader { label, .. } => {
                dict.set_item("label", label.clone())?;
            }
            CoreRinexFinding::ObsPseudorangeOutOfRange { code, value_m, .. } => {
                dict.set_item("code", code.clone())?;
                dict.set_item("value_m", *value_m)?;
            }
            CoreRinexFinding::ObsLossOfLockOutOfRange { code, lli, .. } => {
                dict.set_item("code", code.clone())?;
                dict.set_item("lli", *lli)?;
            }
            CoreRinexFinding::ObsEventHeaderUnreadable { message, .. } => {
                dict.set_item("message", message.clone())?;
            }
            CoreRinexFinding::ObsEventEpoch { flag, .. } => {
                dict.set_item("flag", *flag)?;
            }
            CoreRinexFinding::ObsEmptySatelliteRecord { .. } => {}
            CoreRinexFinding::ObsEpochGap {
                gap_s, interval_s, ..
            } => {
                dict.set_item("gap_s", *gap_s)?;
                dict.set_item("interval_s", *interval_s)?;
            }
            CoreRinexFinding::ObsIntervalUnavailable { .. } => {}
            CoreRinexFinding::ObsInvalidInterval { declared_s, .. } => {
                dict.set_item("declared_s", *declared_s)?;
            }
            CoreRinexFinding::NavFatalParse { message, .. } => {
                dict.set_item("message", message.clone())?;
            }
            CoreRinexFinding::NavLeapSecondsAbsent { .. } => {}
            CoreRinexFinding::NavIonoMalformed { message, .. } => {
                dict.set_item("message", message.clone())?;
            }
            CoreRinexFinding::NavDroppedBlock {
                satellite, message, ..
            } => {
                dict.set_item("satellite", satellite.clone())?;
                dict.set_item("message", message.clone())?;
            }
            CoreRinexFinding::NavDuplicateRecord {
                satellite,
                same_payload,
                ..
            } => {
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("same_payload", *same_payload)?;
            }
            CoreRinexFinding::NavUnsortedRecords { .. } => {}
            CoreRinexFinding::NavImplausibleRecord {
                satellite,
                field,
                value,
                ..
            } => {
                dict.set_item("satellite", satellite.to_string())?;
                dict.set_item("field", *field)?;
                dict.set_item("value", *value)?;
            }
            CoreRinexFinding::NavUnhealthyRecords { system, count, .. } => {
                dict.set_item("system", PyGnssSystem::from(*system))?;
                dict.set_item("count", *count)?;
            }
            CoreRinexFinding::NavOutOfScopeRecords { class, count, .. } => {
                dict.set_item("class", class.clone())?;
                dict.set_item("count", *count)?;
            }
            _ => {}
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexLintFinding(kind={}, code={}, severity={})",
            self.kind(),
            self.inner.code(),
            self.severity().label()
        )
    }
}

/// RINEX lint report.
#[pyclass(module = "sidereon._sidereon", name = "RinexLintReport")]
#[derive(Clone)]
pub struct PyRinexLintReport {
    inner: CoreLintReport,
}

impl From<CoreLintReport> for PyRinexLintReport {
    fn from(inner: CoreLintReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexLintReport {
    #[getter]
    fn findings(&self) -> Vec<PyRinexLintFinding> {
        self.inner
            .findings
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn decoded_from_crinex(&self) -> bool {
        self.inner.decoded_from_crinex
    }

    #[getter]
    fn is_clean(&self) -> bool {
        self.inner.is_clean()
    }

    fn count(&self, severity: PyRinexLintSeverity) -> usize {
        self.inner.count(severity.into())
    }

    fn __repr__(&self) -> String {
        format!("RinexLintReport(findings={})", self.inner.findings.len())
    }
}

/// RINEX mechanical repair options.
#[pyclass(module = "sidereon._sidereon", name = "RinexRepairOptions")]
#[derive(Clone)]
pub struct PyRinexRepairOptions {
    inner: CoreRepairOptions,
}

impl PyRinexRepairOptions {
    fn inner(&self) -> CoreRepairOptions {
        self.inner.clone()
    }
}

#[pymethods]
impl PyRinexRepairOptions {
    #[new]
    #[pyo3(signature = (
        set_interval=false,
        set_time_of_last_obs=false,
        set_obs_counts=false,
        drop_empty_records=false,
        sort_records=true,
        drop_unsupported=false,
    ))]
    fn new(
        set_interval: bool,
        set_time_of_last_obs: bool,
        set_obs_counts: bool,
        drop_empty_records: bool,
        sort_records: bool,
        drop_unsupported: bool,
    ) -> Self {
        let mut inner = CoreRepairOptions::default();
        inner.file_stamp = None;
        inner.set_interval = set_interval;
        inner.set_time_of_last_obs = set_time_of_last_obs;
        inner.set_obs_counts = set_obs_counts;
        inner.drop_empty_records = drop_empty_records;
        inner.sort_records = sort_records;
        inner.drop_unsupported = drop_unsupported;
        Self { inner }
    }
}

/// One RINEX repair action.
#[pyclass(module = "sidereon._sidereon", name = "RinexRepairAction")]
#[derive(Clone)]
pub struct PyRinexRepairAction {
    inner: CoreRepairAction,
}

impl From<CoreRepairAction> for PyRinexRepairAction {
    fn from(inner: CoreRepairAction) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexRepairAction {
    #[getter]
    fn id(&self) -> &'static str {
        self.inner.id
    }

    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }
}

/// Observation repair result.
#[pyclass(module = "sidereon._sidereon", name = "RinexObsRepair")]
#[derive(Clone)]
pub struct PyRinexObsRepair {
    inner: CoreObsRepair,
}

#[pymethods]
impl PyRinexObsRepair {
    #[getter]
    fn repaired(&self) -> PyRinexObs {
        PyRinexObs::from_inner(self.inner.repaired.clone())
    }

    #[getter]
    fn actions(&self) -> Vec<PyRinexRepairAction> {
        self.inner.actions.iter().cloned().map(Into::into).collect()
    }

    #[getter]
    fn remaining(&self) -> PyRinexLintReport {
        self.inner.remaining.clone().into()
    }

    #[getter]
    fn decoded_from_crinex(&self) -> bool {
        self.inner.decoded_from_crinex
    }

    /// The repaired product as CRINEX text.
    ///
    /// Writes the repaired product as RINEX and encodes that text, the two
    /// public calls the core helper composes. A writer refusal keeps its typed
    /// payload: `RinexObsWriteError` with its `detail`, the same exception
    /// `repaired.to_rinex_string()` raises. A CRINEX encoding failure raises
    /// `CrinexParseError`.
    fn to_crinex_string(&self, py: Python<'_>) -> PyResult<String> {
        let text = self
            .inner
            .repaired
            .to_rinex_string()
            .map_err(|err| to_obs_write_err(py, err))?;
        core_encode_crinex(&text).map_err(to_crinex_err)
    }
}

/// Navigation repair result.
#[pyclass(module = "sidereon._sidereon", name = "RinexNavRepair")]
#[derive(Clone)]
pub struct PyRinexNavRepair {
    inner: CoreNavRepair,
}

#[pymethods]
impl PyRinexNavRepair {
    #[getter]
    fn records(&self) -> Vec<PyBroadcastRecord> {
        self.inner.records.iter().copied().map(Into::into).collect()
    }

    #[getter]
    fn iono_corrections(&self) -> Option<PyIonoCorrections> {
        self.inner.iono.map(Into::into)
    }

    #[getter]
    fn leap_seconds(&self) -> Option<f64> {
        self.inner.leap_seconds
    }

    #[getter]
    fn actions(&self) -> Vec<PyRinexRepairAction> {
        self.inner.actions.iter().cloned().map(Into::into).collect()
    }

    #[getter]
    fn remaining(&self) -> PyRinexLintReport {
        self.inner.remaining.clone().into()
    }
}

/// Parse RINEX NAV text into the default broadcast ephemeris store.
#[pyfunction]
fn parse_rinex_nav(text: &str) -> PyResult<PyBroadcastEphemeris> {
    parse_store_text(text)
}

/// Load a RINEX NAV file from bytes, bytearray, or a path.
#[pyfunction]
fn load_rinex_nav(source: &Bound<'_, PyAny>) -> PyResult<PyBroadcastEphemeris> {
    let text = text_from_source(
        source,
        "load_rinex_nav",
        "RINEX NAV source",
        RinexTextKind::Nav,
    )?;
    parse_store_text(&text)
}

/// Parse all supported GPS, Galileo, and BeiDou broadcast records from NAV text.
///
/// This returns the raw supported Keplerian records before the store's default
/// health and message-policy filter.
#[pyfunction]
fn parse_rinex_nav_records(text: &str) -> PyResult<Vec<PyBroadcastRecord>> {
    Ok(parse_nav(text)
        .map_err(to_nav_err)?
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Parse supported NAV records while reporting malformed blocks that were skipped.
#[pyfunction]
fn parse_rinex_nav_lenient(text: &str) -> PyResult<PyRinexNavParse> {
    Ok(PyRinexNavParse {
        inner: parse_nav_lenient(text).map_err(to_nav_err)?,
    })
}

/// CNAV nominal URA bound in metres for an ED/NED0 index.
#[pyfunction]
fn cnav_ura_nominal_m(index: i8) -> Option<f64> {
    core_cnav_ura_nominal_m(index)
}

/// CNAV time-dependent NED URA bound in metres.
#[pyfunction]
fn cnav_ura_ned_m(params: &PyCnavParameters, week: u32, tow_s: f64) -> Option<f64> {
    params.ura_ned_m(week, tow_s)
}

fn repair_options_from_optional(
    py: Python<'_>,
    options: Option<Py<PyRinexRepairOptions>>,
) -> CoreRepairOptions {
    option_py_or_default(
        py,
        options.as_ref(),
        PyRinexRepairOptions::inner,
        CoreRepairOptions::default,
    )
}

/// Lint RINEX observation text, including CRINEX input.
#[pyfunction]
fn lint_rinex_obs(text: &str) -> PyRinexLintReport {
    core_lint_obs_text(text).into()
}

/// Lint RINEX navigation text.
#[pyfunction]
fn lint_rinex_nav(text: &str) -> PyRinexLintReport {
    core_lint_nav_text(text).into()
}

/// Repair RINEX observation text, including CRINEX input.
#[pyfunction]
#[pyo3(signature = (text, options=None))]
fn repair_rinex_obs(
    py: Python<'_>,
    text: &str,
    options: Option<Py<PyRinexRepairOptions>>,
) -> PyResult<PyRinexObsRepair> {
    let options = repair_options_from_optional(py, options);
    Ok(PyRinexObsRepair {
        inner: core_repair_obs_text(text, &options).map_err(to_obs_err)?,
    })
}

/// Repair RINEX navigation text.
#[pyfunction]
#[pyo3(signature = (text, options=None))]
fn repair_rinex_nav(
    py: Python<'_>,
    text: &str,
    options: Option<Py<PyRinexRepairOptions>>,
) -> PyResult<PyRinexNavRepair> {
    let options = repair_options_from_optional(py, options);
    Ok(PyRinexNavRepair {
        inner: core_repair_nav_text(text, &options).map_err(to_nav_err)?,
    })
}

/// Serialize broadcast navigation records to standard RINEX navigation text:
/// RINEX 3.04, or RINEX 4.02 frames when a CNAV-family record is present, with
/// `PGM / RUN BY / DATE` in the header.
///
/// The inverse of [`parse_rinex_nav_records`]: re-parsing the output yields the
/// same records. GLONASS state-vector records are not part of the Keplerian NAV
/// body and are not emitted. Raises `RinexNavWriteError` for a set that holds
/// both a CNAV-family record, which only RINEX 4 holds, and an unclassified
/// Galileo record, which only RINEX 3 holds.
#[pyfunction]
fn encode_rinex_nav(py: Python<'_>, records: Vec<PyBroadcastRecord>) -> PyResult<String> {
    let records: Vec<BroadcastRecord> = records.into_iter().map(|record| record.inner).collect();
    encode_nav(&records).map_err(|err| crate::to_rinex_nav_write_err(py, err))
}

/// Parse all GLONASS state-vector records from RINEX NAV text.
#[pyfunction]
fn parse_rinex_glonass_records(text: &str) -> PyResult<Vec<PyGlonassRecord>> {
    Ok(parse_glonass(text)
        .map_err(to_nav_err)?
        .into_iter()
        .map(Into::into)
        .collect())
}

/// Parse GPS and BeiDou Klobuchar coefficients from a RINEX NAV header.
#[pyfunction]
fn parse_rinex_iono_corrections(text: &str) -> PyResult<PyIonoCorrections> {
    Ok(parse_iono_corrections(text).map_err(to_nav_err)?.into())
}

/// Parse the NAV header GPS minus UTC leap seconds, if present.
#[pyfunction]
fn parse_rinex_leap_seconds(text: &str) -> PyResult<Option<f64>> {
    parse_leap_seconds(text).map_err(to_nav_err)
}

/// Parse RINEX OBS text into a typed observation product.
#[pyfunction]
fn parse_rinex_obs(text: &str) -> PyResult<PyRinexObs> {
    parse_obs_text(text)
}

/// Load a RINEX OBS file from bytes, bytearray, or a path.
#[pyfunction]
fn load_rinex_obs(source: &Bound<'_, PyAny>) -> PyResult<PyRinexObs> {
    let text = text_from_source(
        source,
        "load_rinex_obs",
        "RINEX OBS source",
        RinexTextKind::Obs,
    )?;
    parse_obs_text(&text)
}

/// Decode Compact RINEX (Hatanaka) OBS text into plain RINEX OBS text.
#[pyfunction]
fn decode_crinex(text: &str) -> PyResult<String> {
    core_decode_crinex(text).map_err(to_crinex_err)
}

/// Decode Compact RINEX (Hatanaka) OBS text into plain RINEX OBS lines.
///
/// Each returned line excludes its trailing newline. This mirrors the core
/// streaming decoder while keeping ownership simple for Python callers.
#[pyfunction]
fn decode_crinex_lines(text: &str) -> PyResult<Vec<String>> {
    let mut lines = Vec::new();
    core_decode_crinex_to(text, |line| lines.push(line.to_owned())).map_err(to_crinex_err)?;
    Ok(lines)
}

/// Encode plain RINEX (Hatanaka) OBS text into Compact RINEX (CRINEX) text.
///
/// The inverse of `decode_crinex`: parses the RINEX 2 or RINEX 3 observation
/// text and emits the canonical all-reset CRINEX stream, so
/// `decode_crinex(encode_crinex(text))` round-trips. Raises `CrinexParseError`
/// on malformed input.
#[pyfunction]
fn encode_crinex(text: &str) -> PyResult<String> {
    core_encode_crinex(text).map_err(to_crinex_err)
}

/// Load and decode a Compact RINEX OBS file from bytes, bytearray, or a path.
#[pyfunction]
fn load_crinex(source: &Bound<'_, PyAny>) -> PyResult<String> {
    let text = text_from_source(
        source,
        "load_crinex",
        "CRINEX source",
        RinexTextKind::Crinex,
    )?;
    decode_crinex(&text)
}

/// Rewrite a RINEX OBS product as one a RINEX 2 file of `version` can state
/// exactly, returning `(product, changes)`.
///
/// The free-function form of `RinexObs.downgrade_to_rinex2`, with the same
/// changes and the same refusals.
#[pyfunction]
fn downgrade_to_rinex2(
    py: Python<'_>,
    obs: &PyRinexObs,
    version: f64,
) -> PyResult<(PyRinexObs, Vec<PyObsDowngradeChange>)> {
    obs.downgrade_to_rinex2(py, version)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNavMessage>()?;
    m.add_class::<PyCnavSignal>()?;
    m.add_class::<PyBroadcastGroupDelayTerm>()?;
    m.add_class::<PyBroadcastGroupDelays>()?;
    m.add_class::<PyCnavParameters>()?;
    m.add_class::<PyNavMessagePreference>()?;
    m.add_class::<PyGnssSystem>()?;
    m.add_class::<PyObservationKind>()?;
    m.add_class::<PyObsEpochTime>()?;
    m.add_class::<PyObsScaleFactor>()?;
    m.add_class::<PyObsLeapSeconds>()?;
    m.add_class::<PyGlonassBiasResult>()?;
    m.add_class::<PyObsPhaseShift>()?;
    m.add_class::<PyObsHeader>()?;
    m.add_class::<PyObsValue>()?;
    m.add_class::<PyObsEpoch>()?;
    m.add_class::<PyPhaseShiftResult>()?;
    m.add_class::<PyObsHeaderTimeline>()?;
    m.add_class::<PyObservationFilter>()?;
    m.add_class::<PySignalPolicy>()?;
    m.add_class::<PyPseudorangeSeries>()?;
    m.add_class::<PyObservationValueSeries>()?;
    m.add_class::<PyCarrierPhaseSeries>()?;
    m.add_class::<PyObsDowngradeChange>()?;
    m.add_class::<PyRinexObsWriteErrorDetail>()?;
    m.add_class::<PyRinexObs>()?;
    m.add_class::<PyBroadcastEvaluation>()?;
    m.add_class::<PyKeplerianElements>()?;
    m.add_class::<PyClockPolynomial>()?;
    m.add_class::<PyBroadcastRecord>()?;
    m.add_class::<PyGlonassRecord>()?;
    m.add_class::<PyKlobucharAlphaBeta>()?;
    m.add_class::<PyIonoCorrections>()?;
    m.add_class::<PyBroadcastEphemeris>()?;
    m.add_class::<PySkippedNavBlock>()?;
    m.add_class::<PyRinexNavParse>()?;
    m.add_class::<PyNavDiagnostic>()?;
    m.add_class::<PyOtherNavBlock>()?;
    m.add_class::<PyRinexLintSeverity>()?;
    m.add_class::<PyRinexFindingRef>()?;
    m.add_class::<PyRinexLintFinding>()?;
    m.add_class::<PyRinexLintReport>()?;
    m.add_class::<PyRinexRepairOptions>()?;
    m.add_class::<PyRinexRepairAction>()?;
    m.add_class::<PyRinexObsRepair>()?;
    m.add_class::<PyRinexNavRepair>()?;
    m.add_function(wrap_pyfunction!(parse_rinex_nav, m)?)?;
    m.add_function(wrap_pyfunction!(load_rinex_nav, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_nav_records, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_nav_lenient, m)?)?;
    m.add_function(wrap_pyfunction!(cnav_ura_nominal_m, m)?)?;
    m.add_function(wrap_pyfunction!(cnav_ura_ned_m, m)?)?;
    m.add_function(wrap_pyfunction!(lint_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(lint_rinex_nav, m)?)?;
    m.add_function(wrap_pyfunction!(repair_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(repair_rinex_nav, m)?)?;
    m.add_function(wrap_pyfunction!(encode_rinex_nav, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_glonass_records, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_iono_corrections, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_leap_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(load_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(downgrade_to_rinex2, m)?)?;
    // The epoch flag whose records report cycle slips rather than
    // measurements; `ObsEpoch.cycle_slips` holds them.
    m.add("CYCLE_SLIP_FLAG", CYCLE_SLIP_FLAG)?;
    m.add_function(wrap_pyfunction!(decode_crinex, m)?)?;
    m.add_function(wrap_pyfunction!(decode_crinex_lines, m)?)?;
    m.add_function(wrap_pyfunction!(encode_crinex, m)?)?;
    m.add_function(wrap_pyfunction!(load_crinex, m)?)?;
    Ok(())
}

#[cfg(test)]
mod rinex_error_projection_tests {
    use super::*;

    macro_rules! assert_error {
        ($inner:expr, $kind:literal, $message:literal, $epoch:expr, $system:expr, $satellite:expr, $version:expr, $code:expr, $time:expr, {$($key:literal => $value:expr),* $(,)?}) => {{
            let d = PyRinexObsWriteErrorDetail { inner: $inner };
            assert_eq!(d.kind(), $kind);
            assert_eq!(d.message(), $message);
            assert_eq!(d.inner, d.inner.clone());
            assert_eq!(d.epoch_index(), $epoch);
            assert_eq!(d.system(), $system);
            assert_eq!(d.satellite(), $satellite);
            assert_eq!(d.version(), $version);
            assert_eq!(d.code(), $code);
            assert_eq!(d.time_system(), $time);
            Python::with_gil(|py| {
                let actual = d.details(py).unwrap();
                let expected = PyDict::new(py);
                $(expected.set_item($key, $value).unwrap();)*
                assert!(actual.eq(&expected).unwrap(), "{}: {:?}", $kind, actual);
            });
        }};
    }

    macro_rules! assert_change {
        ($inner:expr, $kind:literal, $epoch:expr, $system:expr, {$($key:literal => $value:expr),* $(,)?}) => {{
            let d = PyObsDowngradeChange { inner: $inner };
            assert_eq!(d.kind(), $kind);
            assert_eq!(d.inner, d.inner.clone());
            assert_eq!(d.epoch_index(), $epoch);
            assert_eq!(d.system(), $system);
            Python::with_gil(|py| {
                let actual = d.details(py).unwrap();
                let expected = PyDict::new(py);
                $(expected.set_item($key, $value).unwrap();)*
                assert!(actual.eq(&expected).unwrap(), "{}: {:?}", $kind, actual);
            });
        }};
    }

    fn sat() -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, 1).unwrap()
    }

    #[test]
    fn all_write_error_variants_project_literal_messages_fields_and_getters() {
        let s = sat();
        assert_error!(CoreRinexObsWriteError::CodeListsNotVersionTwo { system: GnssSystem::Galileo, position: 2, code: Some("C1X".into()) }, "CodeListsNotVersionTwo", "RINEX OBS version 2 has no observation type that every constellation reads back as its own code at position 2, where Galileo holds \"C1X\"", None, Some(PyGnssSystem::GALILEO), None, None, Some("C1X".into()), None, {"system" => PyGnssSystem::GALILEO, "position" => 2usize, "code" => Some("C1X".to_owned())});
        assert_error!(CoreRinexObsWriteError::CodeListsNotVersionTwo { system: GnssSystem::Gps, position: 2, code: None }, "CodeListsNotVersionTwo", "RINEX OBS version 2 names one list of codes for every constellation, and GPS holds 2 codes where another constellation holds a different number", None, Some(PyGnssSystem::GPS), None, None, None, None, {"system" => PyGnssSystem::GPS, "position" => 2usize, "code" => None::<String>});
        assert_error!(CoreRinexObsWriteError::NotVersionTwo { version: 3.05 }, "NotVersionTwo", "RINEX OBS version 3.05 is not a version 2", None, None, None, Some(3.05), None, None, {"version" => 3.05});
        assert_error!(CoreRinexObsWriteError::ScaleFactorsInVersionTwo { count: 2 }, "ScaleFactorsInVersionTwo", "RINEX OBS version 2 would carry 2 SYS / SCALE FACTOR records, which version 2 readers that do not apply them read as physical values; downgrade_to_rinex2 removes them", None, None, None, None, None, None, {"count" => 2usize});
        assert_error!(CoreRinexObsWriteError::ValuesWithoutCodes { epoch_index: 4, satellite: s, codes: 2, values: 3 }, "ValuesWithoutCodes", "RINEX OBS epoch 4 satellite G01 holds 3 values for 2 observation codes", Some(4), None, Some("G01".into()), None, None, None, {"epoch_index" => 4usize, "satellite" => "G01", "codes" => 2usize, "values" => 3usize});
        assert_error!(CoreRinexObsWriteError::CountsWithoutCodes { satellite: s, codes: 2, counts: 3 }, "CountsWithoutCodes", "RINEX OBS PRN / # OF OBS for G01 holds 3 counts for 2 observation codes", None, None, Some("G01".into()), None, None, None, {"satellite" => "G01", "codes" => 2usize, "counts" => 3usize});
        assert_error!(CoreRinexObsWriteError::CodeListNotStated { system: GnssSystem::BeiDou }, "CodeListNotStated", "RINEX OBS version 2 would not state BeiDou's code list: no observation or PRN / # OF OBS count names BeiDou, so a reader builds no list for it, and the type names do not read as it; downgrade_to_rinex2 removes the list", None, Some(PyGnssSystem::BEIDOU), None, None, None, None, {"system" => PyGnssSystem::BEIDOU});
        assert_error!(CoreRinexObsWriteError::EpochFlagTooWide { epoch_index: 4, flag: 10 }, "EpochFlagTooWide", "RINEX OBS epoch 4 flag 10 does not fit the one-digit flag field", Some(4), None, None, None, None, None, {"epoch_index" => 4usize, "flag" => 10u8});
        assert_error!(CoreRinexObsWriteError::EpochTimeMissing { epoch_index: 4, flag: 0 }, "EpochTimeMissing", "RINEX OBS epoch 4 with flag 0 has no epoch time, which only an event may leave blank", Some(4), None, None, None, None, None, {"epoch_index" => 4usize, "flag" => 0u8});
        assert_error!(CoreRinexObsWriteError::EpochPicosecondsNotInVersion { epoch_index: 4, version: 3.05 }, "EpochPicosecondsNotInVersion", "RINEX OBS epoch 4 carries picoseconds, which a version 3.05 epoch record has no field for", Some(4), None, None, Some(3.05), None, None, {"epoch_index" => 4usize, "version" => 3.05});
        assert_error!(CoreRinexObsWriteError::TooManyObservationTypes { count: 1000 }, "TooManyObservationTypes", "RINEX OBS version 2 would need at least 1000 observation types, more than the 999 its count field declares", None, None, None, None, None, None, {"count" => 1000usize});
        assert_error!(CoreRinexObsWriteError::CodeListsNotUnion { system: GnssSystem::Glonass }, "CodeListsNotUnion", "RINEX OBS GLONASS code list is not the union of the lists the header and its events declare", None, Some(PyGnssSystem::GLONASS), None, None, None, None, {"system" => PyGnssSystem::GLONASS});
        assert_error!(CoreRinexObsWriteError::ValueOutsideDeclaredList { epoch_index: 4, satellite: s, code: Some("L1C".into()) }, "ValueOutsideDeclaredList", "RINEX OBS epoch 4 satellite G01 holds a value under \"L1C\", which the list in effect at that epoch does not declare", Some(4), None, Some("G01".into()), None, Some("L1C".into()), None, {"epoch_index" => 4usize, "satellite" => "G01", "code" => Some("L1C".to_owned())});
        assert_error!(CoreRinexObsWriteError::ValueOutsideDeclaredList { epoch_index: 4, satellite: s, code: None }, "ValueOutsideDeclaredList", "RINEX OBS epoch 4 satellite G01 is of a constellation with no code list in effect at that epoch", Some(4), None, Some("G01".into()), None, None, None, {"epoch_index" => 4usize, "satellite" => "G01", "code" => None::<String>});
        assert_error!(CoreRinexObsWriteError::DeclaredListNotStated { system: GnssSystem::Qzss }, "DeclaredListNotStated", "RINEX OBS QZSS declared code list is not what the version 2 type names state for it", None, Some(PyGnssSystem::QZSS), None, None, None, None, {"system" => PyGnssSystem::QZSS});
        assert_error!(CoreRinexObsWriteError::EventRecordsUnreadable { message: "bad record".into() }, "EventRecordsUnreadable", "RINEX OBS event header records do not read: bad record", None, None, None, None, None, None, {"message" => "bad record"});
        assert_error!(CoreRinexObsWriteError::ObservableNotRepresentable { system: GnssSystem::BeiDou, code: "C1X".into(), version: 2.11 }, "ObservableNotRepresentable", "RINEX OBS BeiDou code C1X is on a carrier that version 2.11 cannot represent", None, Some(PyGnssSystem::BEIDOU), None, Some(2.11), Some("C1X".into()), None, {"system" => PyGnssSystem::BEIDOU, "code" => "C1X", "version" => 2.11});
        assert_error!(CoreRinexObsWriteError::LeapSecondsTimeSystemNotInVersion { time_system: "BDT".into(), version: 2.11 }, "LeapSecondsTimeSystemNotInVersion", "RINEX OBS LEAP SECONDS time system BDT is not supported in version 2.11", None, None, None, Some(2.11), None, Some("BDT"), {"time_system" => "BDT", "version" => 2.11});
        assert_error!(CoreRinexObsWriteError::InvalidLeapSecondsTimeSystem { time_system: "XYZ".into() }, "InvalidLeapSecondsTimeSystem", "RINEX OBS LEAP SECONDS invalid time system identifier: \"XYZ\"", None, None, None, None, None, Some("XYZ"), {"time_system" => "XYZ"});
        assert_error!(CoreRinexObsWriteError::ReadBackMismatch { what: "header.version".into() }, "ReadBackMismatch", "RINEX OBS text would not read back as the product: header.version", None, None, None, None, None, None, {"what" => "header.version"});
    }

    #[test]
    fn all_downgrade_change_variants_project_every_field_and_nested_change() {
        let s = sat();
        let gps = Some(PyGnssSystem::GPS);
        let no_system = None;
        assert_change!(CoreObsDowngradeChange::CodeRenamed { system: GnssSystem::BeiDou, from: "C1I".into(), to: "C2I".into() }, "CodeRenamed", None, Some(PyGnssSystem::BEIDOU), {"system" => PyGnssSystem::BEIDOU, "from_code" => "C1I", "to_code" => "C2I"});
        assert_change!(CoreObsDowngradeChange::CodeMoved { system: GnssSystem::Gps, code: "L1C".into(), from: 1, to: 0 }, "CodeMoved", None, gps, {"system" => PyGnssSystem::GPS, "code" => "L1C", "from_position" => 1usize, "to_position" => 0usize});
        assert_change!(CoreObsDowngradeChange::CodeAdded { system: GnssSystem::Galileo, code: "C1C".into() }, "CodeAdded", None, Some(PyGnssSystem::GALILEO), {"system" => PyGnssSystem::GALILEO, "code" => "C1C"});
        assert_change!(CoreObsDowngradeChange::CodeListRemoved { system: GnssSystem::Glonass, codes: vec!["C1C".into(), "L1C".into()] }, "CodeListRemoved", None, Some(PyGnssSystem::GLONASS), {"system" => PyGnssSystem::GLONASS, "codes" => vec!["C1C", "L1C"]});
        assert_change!(CoreObsDowngradeChange::ValueRounded { epoch_index: 3, satellite: s, code: "L1C".into(), from: 100.000125, to: 100.0 }, "ValueRounded", Some(3), no_system, {"epoch_index" => 3usize, "satellite" => "G01", "code" => "L1C", "from_value" => 100.000125, "to_value" => 100.0});
        assert_change!(CoreObsDowngradeChange::CycleSlipRounded { epoch_index: 3, satellite: s, code: "L1C".into(), from: 0.125125, to: 0.125 }, "CycleSlipRounded", Some(3), no_system, {"epoch_index" => 3usize, "satellite" => "G01", "code" => "L1C", "from_value" => 0.125125, "to_value" => 0.125});
        assert_change!(CoreObsDowngradeChange::ScaleFactorsRemoved { count: 2 }, "ScaleFactorsRemoved", None, no_system, {"count" => 2usize});
        assert_change!(CoreObsDowngradeChange::EpochPicosecondsRemoved { epoch_index: 3, picoseconds: 123456 }, "EpochPicosecondsRemoved", Some(3), no_system, {"epoch_index" => 3usize, "picoseconds" => 123456u32});
        assert_change!(CoreObsDowngradeChange::ClockOffsetRounded { epoch_index: 3, from: 1.123456789123, to: 1.123456789 }, "ClockOffsetRounded", Some(3), no_system, {"epoch_index" => 3usize, "from_offset_s" => 1.123456789123, "to_offset_s" => 1.123456789});

        let nested = PyObsDowngradeChange {
            inner: CoreObsDowngradeChange::InEventLists {
                epoch_index: 7,
                change: Box::new(CoreObsDowngradeChange::CodeMoved {
                    system: GnssSystem::Gps,
                    code: "L1C".into(),
                    from: 1,
                    to: 0,
                }),
            },
        };
        assert_eq!(nested.kind(), "InEventLists");
        assert_eq!(nested.epoch_index(), Some(7));
        assert_eq!(nested.system(), None);
        assert_eq!(nested.inner, nested.inner.clone());
        Python::with_gil(|py| {
            let d = nested.details(py).unwrap();
            assert_eq!(
                d.get_item("epoch_index")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                7
            );
            let inner = d
                .get_item("change")
                .unwrap()
                .unwrap()
                .extract::<PyRef<'_, PyObsDowngradeChange>>()
                .unwrap();
            assert_eq!(inner.kind(), "CodeMoved");
            assert_eq!(
                inner
                    .details(py)
                    .unwrap()
                    .get_item("system")
                    .unwrap()
                    .unwrap()
                    .extract::<PyGnssSystem>()
                    .unwrap(),
                PyGnssSystem::GPS
            );
            assert_eq!(
                inner
                    .details(py)
                    .unwrap()
                    .get_item("code")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "L1C"
            );
            assert_eq!(
                inner
                    .details(py)
                    .unwrap()
                    .get_item("from_position")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                1
            );
            assert_eq!(
                inner
                    .details(py)
                    .unwrap()
                    .get_item("to_position")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                0
            );
        });

        assert_change!(CoreObsDowngradeChange::DeprecatedRecordsRemoved { label: "SYS / PHASE SHIFT".into(), epoch_index: None, records: vec!["G L1C  0.25000".into()] }, "DeprecatedRecordsRemoved", None, no_system, {"label" => "SYS / PHASE SHIFT", "epoch_index" => None::<usize>, "records" => vec!["G L1C  0.25000"]});
        assert_change!(CoreObsDowngradeChange::EventRecordsRewritten { epoch_index: 7, from: vec!["G    1 L1C".into()], to: vec!["     1    C1".into()] }, "EventRecordsRewritten", Some(7), no_system, {"epoch_index" => 7usize, "from_records" => vec!["G    1 L1C"], "to_records" => vec!["     1    C1"]});
    }
}
