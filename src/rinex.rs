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
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyModule};

use sidereon_core::ephemeris::{
    cnav_ura_ned_m as core_cnav_ura_ned_m, cnav_ura_nominal_m as core_cnav_ura_nominal_m,
    is_beidou_geo, satellite_state, satellite_state_cnav,
    BroadcastEphemeris as CoreBroadcastEphemeris, BroadcastGroupDelayTerm, BroadcastGroupDelays,
    BroadcastRecord, ClockPolynomial, CnavParameters, CnavRates, CnavSignal, GlonassRecord,
    IonoCorrections, KeplerianElements, KlobucharAlphaBeta, NavMessage, NavMessagePreference,
    SatelliteState,
};
use sidereon_core::rinex::clock::{
    civil_to_gps_seconds as core_civil_to_gps_seconds, ClockEpoch as CoreClockEpoch,
    ClockPoint as CoreClockPoint, RinexClock as CoreRinexClock, RinexClockError,
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
    ObsEpoch as CoreObsEpoch, ObsEpochTime as CoreObsEpochTime, ObsHeader as CoreObsHeader,
    ObsPhaseShift as CoreObsPhaseShift, ObservationFilter as CoreObservationFilter,
    ObservationKind as CoreObservationKind, ObservationValueRow as CoreObservationValueRow,
    RinexObs as CoreRinexObs, SignalPolicy as CoreSignalPolicy,
};
use sidereon_core::rinex::qc::{
    lint_nav_text as core_lint_nav_text, lint_obs_text as core_lint_obs_text,
    repair_nav_text as core_repair_nav_text, repair_obs_text as core_repair_obs_text,
    repair_obs_to_crinex_string as core_repair_obs_to_crinex_string, Finding as CoreRinexFinding,
    FindingRef as CoreFindingRef, LintReport as CoreLintReport, NavRepair as CoreNavRepair,
    ObsRepair as CoreObsRepair, RepairAction as CoreRepairAction,
    RepairOptions as CoreRepairOptions, Severity as CoreRinexSeverity,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

use crate::frames::PyTimeScale;
use crate::marshal::{option_py_or_default, PyGnssSystem};
use crate::{np_array, CrinexParseError, RinexNavParseError};
use crate::{RinexClockParseError, RinexObsParseError};

fn to_nav_err(err: NavParseError) -> PyErr {
    RinexNavParseError::new_err(err.to_string())
}

#[derive(Clone, Copy)]
enum RinexTextKind {
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

fn text_from_source(
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

fn to_clock_err(err: RinexClockError) -> PyErr {
    RinexClockParseError::new_err(err.to_string())
}

fn parse_obs_text(text: &str) -> PyResult<PyRinexObs> {
    Ok(PyRinexObs {
        inner: CoreRinexObs::parse(text).map_err(to_obs_err)?,
    })
}

fn parse_clock_text(text: &str) -> PyResult<PyRinexClock> {
    Ok(PyRinexClock {
        inner: CoreRinexClock::parse(text).map_err(to_clock_err)?,
    })
}

fn parse_clock_text_lossy(text: &str) -> PyRinexClock {
    PyRinexClock {
        inner: CoreRinexClock::parse_lossy(text),
    }
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

/// Civil GPS-time epoch used by RINEX clock products.
///
/// Calendar fields are interpreted in GPS time. The `gps_seconds` property is
/// seconds since 1980-01-06 00:00:00 GPST.
#[pyclass(module = "sidereon._sidereon", name = "ClockEpoch")]
#[derive(Clone, Copy, Debug)]
pub struct PyClockEpoch {
    inner: CoreClockEpoch,
    gps_seconds: f64,
}

#[pymethods]
impl PyClockEpoch {
    /// Build a GPS-time civil epoch for RINEX clock interpolation.
    #[new]
    fn new(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> PyResult<Self> {
        let gps_seconds = core_civil_to_gps_seconds(year, month, day, hour, minute, second)
            .ok_or_else(|| {
                PyValueError::new_err(
                    "invalid GPS-time clock epoch fields for RINEX clock interpolation",
                )
            })?;
        Ok(Self {
            inner: CoreClockEpoch {
                year,
                month,
                day,
                hour,
                minute,
                second,
            },
            gps_seconds,
        })
    }

    /// Calendar year in GPS time.
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

    /// Fractional seconds of minute in GPS time.
    #[getter]
    fn second(&self) -> f64 {
        self.inner.second
    }

    /// Seconds since the GPS epoch, 1980-01-06 00:00:00 GPST.
    #[getter]
    fn gps_seconds(&self) -> f64 {
        self.gps_seconds
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockEpoch({:04}-{:02}-{:02}T{:02}:{:02}:{:06.3} GPST)",
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

/// Per-satellite RINEX clock-bias samples.
///
/// `gps_seconds` and `bias_s` are row-aligned numpy `(n,)` arrays. Epochs are
/// seconds since the GPS epoch and clock biases are seconds.
#[pyclass(module = "sidereon._sidereon", name = "ClockSeries")]
#[derive(Clone)]
pub struct PyClockSeries {
    satellite: String,
    gps_seconds: Vec<f64>,
    bias_s: Vec<f64>,
}

impl PyClockSeries {
    fn from_points(satellite: String, points: &[CoreClockPoint]) -> Self {
        let mut gps_seconds = Vec::with_capacity(points.len());
        let mut bias_s = Vec::with_capacity(points.len());
        for point in points {
            if let Some(seconds) = point.gps_seconds() {
                gps_seconds.push(seconds);
                bias_s.push(point.bias_s);
            }
        }
        Self {
            satellite,
            gps_seconds,
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

    /// Sample times as a numpy `(n,)` array, seconds since the GPS epoch.
    #[getter]
    fn gps_seconds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.gps_seconds)
    }

    /// Satellite clock-bias samples as a numpy `(n,)` array, seconds.
    #[getter]
    fn bias_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.bias_s)
    }

    /// Number of clock samples for this satellite.
    fn __len__(&self) -> usize {
        self.bias_s.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClockSeries(satellite='{}', samples={})",
            self.satellite,
            self.bias_s.len()
        )
    }
}

/// Parsed RINEX clock product with satellite clock-bias interpolation.
///
/// The product contains per-satellite `AS` clock records. Times are GPS seconds
/// since 1980-01-06 00:00:00 GPST and clock biases are seconds.
#[pyclass(module = "sidereon._sidereon", name = "RinexClock")]
#[derive(Clone)]
pub struct PyRinexClock {
    inner: CoreRinexClock,
}

#[pymethods]
impl PyRinexClock {
    /// Satellite tokens with at least one parsed `AS` clock sample.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner.series.keys().cloned().collect()
    }

    /// Per-satellite clock-bias series in satellite sort order.
    #[getter]
    fn series(&self) -> Vec<PyClockSeries> {
        self.inner
            .series
            .iter()
            .map(|(satellite, points)| PyClockSeries::from_points(satellite.clone(), points))
            .collect()
    }

    /// Number of satellites with clock samples.
    #[getter]
    fn satellite_count(&self) -> usize {
        self.inner.series.len()
    }

    /// Total number of parsed satellite clock samples.
    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.series.values().map(Vec::len).sum()
    }

    /// Return one satellite's clock series, or `None` if the satellite is absent.
    fn series_for(&self, satellite_id: &str) -> Option<PyClockSeries> {
        self.inner
            .series
            .get(satellite_id)
            .map(|points| PyClockSeries::from_points(satellite_id.to_string(), points))
    }

    /// Interpolate one satellite clock bias at a GPS-time civil epoch.
    fn clock_s(&self, satellite_id: &str, epoch: &PyClockEpoch) -> PyResult<Option<f64>> {
        self.inner
            .clock_s(satellite_id, epoch.inner)
            .map_err(to_clock_err)
    }

    /// Interpolate one satellite clock bias at GPS seconds.
    fn clock_s_at_gps_seconds(
        &self,
        satellite_id: &str,
        gps_seconds: f64,
    ) -> PyResult<Option<f64>> {
        if !gps_seconds.is_finite() {
            return Err(PyValueError::new_err("gps_seconds must be finite"));
        }
        self.inner
            .clock_s_at_gps_seconds(satellite_id, gps_seconds)
            .map_err(to_clock_err)
    }

    /// Serialize this product to standard RINEX clock text via the core writer.
    ///
    /// Re-parsing the output with [`parse_rinex_clock`] reproduces the same time
    /// scale and per-satellite series.
    fn to_rinex_string(&self) -> PyResult<String> {
        self.inner
            .to_rinex_string()
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexClock(satellite_count={}, sample_count={})",
            self.inner.series.len(),
            self.inner.series.values().map(Vec::len).sum::<usize>()
        )
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

    /// RINEX carrier observation code, such as `L1C`.
    #[getter]
    fn code(&self) -> &str {
        &self.inner.code
    }

    /// Phase correction in carrier cycles.
    #[getter]
    fn correction_cycles(&self) -> f64 {
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

    fn __repr__(&self) -> String {
        format!(
            "ObsPhaseShift(system={}, code={:?}, correction_cycles={})",
            self.system().label(),
            self.inner.code,
            self.inner.correction_cycles
        )
    }
}

/// Parsed RINEX OBS header.
#[pyclass(module = "sidereon._sidereon", name = "ObsHeader")]
#[derive(Clone)]
pub struct PyObsHeader {
    inner: CoreObsHeader,
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

    /// Constellations with declared observation-code lists.
    #[getter]
    fn systems(&self) -> Vec<PyGnssSystem> {
        self.inner
            .obs_codes
            .keys()
            .copied()
            .map(Into::into)
            .collect()
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

    /// Observation codes for a constellation, in RINEX header order.
    fn obs_codes(&self, system: PyGnssSystem) -> Vec<String> {
        self.inner
            .obs_codes
            .get(&system.into())
            .cloned()
            .unwrap_or_default()
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

/// One RINEX OBS epoch. Observation values are read through `RinexObs` methods.
#[pyclass(module = "sidereon._sidereon", name = "ObsEpoch")]
#[derive(Clone)]
pub struct PyObsEpoch {
    inner: CoreObsEpoch,
}

impl From<CoreObsEpoch> for PyObsEpoch {
    fn from(inner: CoreObsEpoch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObsEpoch {
    /// Civil epoch in the file time scale.
    #[getter]
    fn epoch(&self) -> PyObsEpochTime {
        self.inner.epoch.into()
    }

    /// RINEX epoch flag. `0` is a normal observation epoch.
    #[getter]
    fn flag(&self) -> u8 {
        self.inner.flag
    }

    /// Satellite tokens present at this epoch.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner.sats.keys().map(|sat| sat.to_string()).collect()
    }

    /// Number of satellites present at this epoch.
    #[getter]
    fn satellite_count(&self) -> usize {
        self.inner.sats.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObsEpoch(epoch={:?}, flag={}, satellite_count={})",
            self.epoch(),
            self.inner.flag,
            self.inner.sats.len()
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

/// Flattened carrier-phase rows from one RINEX OBS epoch.
///
/// Numeric arrays are row-aligned with `satellites` and `codes`. Missing values
/// and unknown carrier metadata are represented as `NaN`.
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
            lli: Vec::with_capacity(row_count),
            ssi: Vec::with_capacity(row_count),
        };
        for (satellite, rows) in rows {
            let satellite = satellite.to_string();
            for row in rows {
                out.satellites.push(satellite.clone());
                out.codes.push(row.code);
                out.value_cycles.push(nan_if_missing(row.value_cycles));
                out.frequency_hz.push(nan_if_missing(row.frequency_hz));
                out.wavelength_m.push(nan_if_missing(row.wavelength_m));
                out.value_m.push(nan_if_missing(row.value_m));
                out.phase_shift_cycles.push(row.phase_shift_cycles);
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

    /// Applied phase shift as a numpy `(n,)` array, cycles.
    #[getter]
    fn phase_shift_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_vec(py, self.phase_shift_cycles.clone())
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

/// Parsed RINEX 3 observation file.
///
/// Epochs remain in file order. Use `observation_values`, `carrier_phase_rows`,
/// and `pseudoranges` to extract row-aligned numpy numeric series.
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

    /// Return one epoch by zero-based index.
    fn epoch(&self, epoch_index: usize) -> PyResult<PyObsEpoch> {
        Ok(check_epoch_index(&self.inner, epoch_index)?.clone().into())
    }

    /// Observation codes for a constellation, in header order.
    fn obs_codes(&self, system: PyGnssSystem) -> Vec<String> {
        self.inner
            .obs_codes(system.into())
            .map(|codes| codes.to_vec())
            .unwrap_or_default()
    }

    /// Flatten raw observation values for one epoch.
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

    /// Flatten carrier-phase values for one epoch.
    #[pyo3(signature = (epoch_index, filter=None))]
    fn carrier_phase_rows(
        &self,
        py: Python<'_>,
        epoch_index: usize,
        filter: Option<Py<PyObservationFilter>>,
    ) -> PyResult<PyCarrierPhaseSeries> {
        let epoch = check_epoch_index(&self.inner, epoch_index)?;
        let filter = filter_from_optional(py, filter);
        Ok(PyCarrierPhaseSeries::from_rows(
            carrier_phase_rows(&self.inner, epoch, &filter).map_err(to_obs_err)?,
        ))
    }

    /// Extract single-frequency pseudoranges for one epoch.
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

    /// Serialize this product to standard RINEX 3 observation text via the core
    /// writer.
    ///
    /// Re-parsing the output with [`parse_rinex_obs`] reproduces the same header
    /// and epochs.
    fn to_rinex_string(&self) -> String {
        self.inner.to_rinex_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexObs(version={}, epoch_count={})",
            self.inner.header().version,
            self.inner.epochs().len()
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

    /// Signal-in-space accuracy in metres.
    #[getter]
    fn sv_accuracy_m(&self) -> f64 {
        self.inner.sv_accuracy_m
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

    /// Native issue-of-data value.
    #[getter]
    fn issue(&self) -> u32 {
        self.inner.issue_of_data.issue
    }

    /// Navigation message associated with the issue-of-data value.
    #[getter]
    fn issue_message(&self) -> PyNavMessage {
        self.inner.issue_of_data.message.into()
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

    /// FDMA frequency-channel number.
    #[getter]
    fn freq_channel(&self) -> i32 {
        self.inner.freq_channel
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

    fn __repr__(&self) -> String {
        format!(
            "SkippedNavBlock(satellite={:?}, message={:?})",
            self.inner.satellite, self.inner.message
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
        CoreRinexFinding::ObsEventSpecialRecords { .. } => "ObsEventSpecialRecords",
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
        Self {
            inner: CoreRepairOptions {
                file_stamp: None,
                set_interval,
                set_time_of_last_obs,
                set_obs_counts,
                drop_empty_records,
                sort_records,
                drop_unsupported,
            },
        }
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

    fn to_crinex_string(&self) -> PyResult<String> {
        core_repair_obs_to_crinex_string(&self.inner).map_err(to_crinex_err)
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

/// Serialize broadcast navigation records to standard RINEX 3 navigation text.
///
/// The inverse of [`parse_rinex_nav_records`]: re-parsing the output yields the
/// same records. GLONASS state-vector records are not part of the Keplerian NAV
/// body and are not emitted.
#[pyfunction]
fn encode_rinex_nav(records: Vec<PyBroadcastRecord>) -> String {
    let records: Vec<BroadcastRecord> = records.into_iter().map(|record| record.inner).collect();
    encode_nav(&records)
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

/// Strictly parse RINEX clock text into satellite clock-bias series.
#[pyfunction]
fn parse_rinex_clock(text: &str) -> PyResult<PyRinexClock> {
    parse_clock_text(text)
}

/// Load and strictly parse a RINEX clock file from bytes, bytearray, or a path.
#[pyfunction]
fn load_rinex_clock(source: &Bound<'_, PyAny>) -> PyResult<PyRinexClock> {
    let text = text_from_source(
        source,
        "load_rinex_clock",
        "RINEX clock source",
        RinexTextKind::Clock,
    )?;
    parse_clock_text(&text)
}

/// Parse RINEX clock text while skipping malformed and non-`AS` rows.
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
    m.add_class::<PyObsPhaseShift>()?;
    m.add_class::<PyObsHeader>()?;
    m.add_class::<PyObsEpoch>()?;
    m.add_class::<PyObservationFilter>()?;
    m.add_class::<PySignalPolicy>()?;
    m.add_class::<PyPseudorangeSeries>()?;
    m.add_class::<PyObservationValueSeries>()?;
    m.add_class::<PyCarrierPhaseSeries>()?;
    m.add_class::<PyRinexObs>()?;
    m.add_class::<PyClockEpoch>()?;
    m.add_class::<PyClockSeries>()?;
    m.add_class::<PyRinexClock>()?;
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
    m.add_function(wrap_pyfunction!(parse_rinex_clock, m)?)?;
    m.add_function(wrap_pyfunction!(load_rinex_clock, m)?)?;
    m.add_function(wrap_pyfunction!(parse_rinex_clock_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(load_rinex_clock_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(decode_crinex, m)?)?;
    m.add_function(wrap_pyfunction!(decode_crinex_lines, m)?)?;
    m.add_function(wrap_pyfunction!(encode_crinex, m)?)?;
    m.add_function(wrap_pyfunction!(load_crinex, m)?)?;
    Ok(())
}
