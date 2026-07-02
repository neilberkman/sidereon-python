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
    SbasBlock, SbasCorrectedEphemeris, SbasCorrectionStore, SbasFastCorrection, SbasGeoState,
    SbasIgp, SbasIonoGrid, SbasLogBlock, SbasLongTermCorrection, SbasMessage, SbasSolveMode,
    SbasWireForm,
};
use sidereon_core::ssr::{
    MissingCorrectionAction, OrbitReferencePoint, RegionalPolicy, SsrClockCorrection,
    SsrCorrectedEphemeris, SsrCorrectionStore, SsrFallbackPolicy, SsrHighRateClock,
    SsrOrbitCorrection,
};
use sidereon_core::GnssSatelliteId;

use crate::frames::PyTimeScale;
use crate::marshal::PyGnssSystem;
use crate::rinex::PyBroadcastEphemeris;
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

#[pyclass(module = "sidereon._sidereon", name = "SbasWireForm", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
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

#[pyclass(module = "sidereon._sidereon", name = "SbasBlock")]
#[derive(Clone)]
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
    fn kind(&self) -> &'static str {
        sbas_message_label(&self.inner.message)
    }

    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
    }

    fn __repr__(&self) -> String {
        format!(
            "SbasBlock(form={}, message_type={}, kind={:?})",
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
}

impl From<SbasLogBlock> for PySbasLogBlock {
    fn from(inner: SbasLogBlock) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasFastCorrection")]
#[derive(Clone)]
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
}

impl From<SbasFastCorrection> for PySbasFastCorrection {
    fn from(inner: SbasFastCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasLongTermCorrection")]
#[derive(Clone)]
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
}

impl From<SbasLongTermCorrection> for PySbasLongTermCorrection {
    fn from(inner: SbasLongTermCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasIgp")]
#[derive(Clone)]
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
}

impl From<SbasIgp> for PySbasIgp {
    fn from(inner: SbasIgp) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasIonoGrid")]
#[derive(Clone)]
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
}

impl From<SbasIonoGrid> for PySbasIonoGrid {
    fn from(inner: SbasIonoGrid) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasGeoState")]
#[derive(Clone)]
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
}

impl From<SbasGeoState> for PySbasGeoState {
    fn from(inner: SbasGeoState) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasCorrectionStore")]
pub struct PySbasCorrectionStore {
    inner: SbasCorrectionStore,
}

#[pymethods]
impl PySbasCorrectionStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: SbasCorrectionStore::new(),
        }
    }

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

    fn fast(
        &self,
        geo_satellite_id: &str,
        satellite_id: &str,
    ) -> PyResult<Option<PySbasFastCorrection>> {
        let geo = parse_satellite(geo_satellite_id)?;
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.fast(geo, sat).cloned().map(Into::into))
    }

    fn long_term(
        &self,
        geo_satellite_id: &str,
        satellite_id: &str,
    ) -> PyResult<Option<PySbasLongTermCorrection>> {
        let geo = parse_satellite(geo_satellite_id)?;
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.long_term(geo, sat).cloned().map(Into::into))
    }

    fn iono_grid(&self, geo_satellite_id: &str) -> PyResult<Option<PySbasIonoGrid>> {
        let geo = parse_satellite(geo_satellite_id)?;
        Ok(self.inner.iono_grid(geo).cloned().map(Into::into))
    }

    fn geo_nav(&self, geo_satellite_id: &str) -> PyResult<Option<PySbasGeoState>> {
        let geo = parse_satellite(geo_satellite_id)?;
        Ok(self.inner.geo_nav(geo).cloned().map(Into::into))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SbasCorrectedEphemeris")]
pub struct PySbasCorrectedEphemeris {
    broadcast: Py<PyBroadcastEphemeris>,
    store: Py<PySbasCorrectionStore>,
    geo: GnssSatelliteId,
    mode: SbasSolveMode,
}

#[pymethods]
impl PySbasCorrectedEphemeris {
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
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "OrbitReferencePoint",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyOrbitReferencePoint {
    ANTENNA_PHASE_CENTER,
    CENTER_OF_MASS,
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
pub enum PyMissingCorrectionAction {
    DECLINE,
    FALL_BACK_TO_BROADCAST,
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
pub struct PySsrFallbackPolicy {
    inner: SsrFallbackPolicy,
}

#[pymethods]
impl PySsrFallbackPolicy {
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
}

#[pyclass(module = "sidereon._sidereon", name = "SsrOrbitCorrection")]
#[derive(Clone, Copy)]
pub struct PySsrOrbitCorrection {
    inner: SsrOrbitCorrection,
}

#[pymethods]
impl PySsrOrbitCorrection {
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
}

impl From<SsrOrbitCorrection> for PySsrOrbitCorrection {
    fn from(inner: SsrOrbitCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrHighRateClock")]
#[derive(Clone, Copy)]
pub struct PySsrHighRateClock {
    inner: SsrHighRateClock,
}

#[pymethods]
impl PySsrHighRateClock {
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
}

impl From<SsrHighRateClock> for PySsrHighRateClock {
    fn from(inner: SsrHighRateClock) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrClockCorrection")]
#[derive(Clone, Copy)]
pub struct PySsrClockCorrection {
    inner: SsrClockCorrection,
}

#[pymethods]
impl PySsrClockCorrection {
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
}

impl From<SsrClockCorrection> for PySsrClockCorrection {
    fn from(inner: SsrClockCorrection) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrMessage")]
#[derive(Clone)]
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
}

impl PySsrMessage {
    fn from_inner(inner: SsrMessage) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrCorrectionStore")]
pub struct PySsrCorrectionStore {
    inner: SsrCorrectionStore,
}

#[pymethods]
impl PySsrCorrectionStore {
    #[new]
    #[pyo3(signature = (reference_point=PyOrbitReferencePoint::CENTER_OF_MASS))]
    fn new(reference_point: PyOrbitReferencePoint) -> Self {
        Self {
            inner: SsrCorrectionStore::new().with_reference_point(reference_point.into()),
        }
    }

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

    fn orbit(&self, satellite_id: &str) -> PyResult<Option<PySsrOrbitCorrection>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.orbit(sat).copied().map(Into::into))
    }

    fn clock(&self, satellite_id: &str) -> PyResult<Option<PySsrClockCorrection>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.clock(sat).copied().map(Into::into))
    }

    fn ura_index(&self, satellite_id: &str) -> PyResult<Option<u8>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.ura_index(sat))
    }

    fn code_bias(&self, satellite_id: &str, signal: u8) -> PyResult<Option<f64>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.code_bias(sat, signal))
    }

    fn phase_bias(&self, satellite_id: &str, signal: u8) -> PyResult<Option<f64>> {
        let sat = parse_satellite(satellite_id)?;
        Ok(self.inner.phase_bias(sat, signal))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SsrCorrectedEphemeris")]
pub struct PySsrCorrectedEphemeris {
    broadcast: Py<PyBroadcastEphemeris>,
    store: Py<PySsrCorrectionStore>,
    fallback: SsrFallbackPolicy,
    max_staleness_s: Option<f64>,
}

#[pymethods]
impl PySsrCorrectedEphemeris {
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

    fn corrected_state(
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
fn decode_sbas_block(bytes: &[u8], form: PySbasWireForm) -> PyResult<PySbasBlock> {
    SbasBlock::decode(bytes, form.into())
        .map(PySbasBlock::from_inner)
        .map_err(to_rtcm_err)
}

#[pyfunction]
fn parse_sbas_ems_lines(text: &str) -> PyResult<Vec<PySbasLogBlock>> {
    core_parse_sbas_ems_lines(text)
        .map(|blocks| blocks.into_iter().map(Into::into).collect())
        .map_err(to_rtcm_err)
}

#[pyfunction]
fn parse_sbas_rtklib_lines(text: &str) -> PyResult<Vec<PySbasLogBlock>> {
    core_parse_sbas_rtklib_lines(text)
        .map(|blocks| blocks.into_iter().map(Into::into).collect())
        .map_err(to_rtcm_err)
}

#[pyfunction]
fn sbas_prn_to_satellite_id(prn: u16) -> Option<String> {
    sbas_prn_to_sat(prn).map(py_satellite)
}

#[pyfunction]
fn satellite_id_to_sbas_prn(satellite_id: &str) -> PyResult<Option<u16>> {
    let sat = parse_satellite(satellite_id)?;
    Ok(sat_to_sbas_prn(sat))
}

#[pyfunction]
fn decode_ssr_message(body: &[u8]) -> PyResult<PySsrMessage> {
    SsrMessage::decode(body)
        .map(PySsrMessage::from_inner)
        .map_err(to_rtcm_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySbasWireForm>()?;
    m.add_class::<PySbasSolveMode>()?;
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
