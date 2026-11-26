//! GNSS yaw laws, satellite block identifiers, and inertial attitude geometry.

use std::sync::Arc;

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::attitude::gnss::{self, AttitudeTrajectory, OrbitAttitudeTrajectory};
use sidereon_core::astro::frames::{EarthOrientationProvider, TdbEarthOrientationProvider};
use sidereon_core::orbit_scoring::DEFAULT_VELOCITY_HALF_STEP_S;
use sidereon_core::GnssSatelliteId;

use crate::analytic::{PyBrouwerLyddanePropagator, PyEcksteinHechlerPropagator};
use crate::de::PyDeEphemeris;
use crate::earth_orientation::PyIers2010EarthOrientationProvider;
use crate::ephemeris::{parse_sat, PySp3};
use crate::marshal::PyGnssSystem;
use crate::orbit_scoring::{orbit_score_error, PyOrbitSource};
use crate::rinex::PyBroadcastEphemeris;
use crate::variational::PyDenseOutput;
use crate::{binding_error, SolveError};

/// Part of the yaw law that produced the attitude.
#[pyclass(
    module = "sidereon._sidereon",
    name = "AttitudeMode",
    eq,
    eq_int,
    frozen
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum PyAttitudeMode {
    /// Nominal yaw steering.
    NOMINAL,
    /// A turn around orbit noon.
    NOON_TURN,
    /// A turn around orbit midnight outside a shadow-crossing law.
    MIDNIGHT_TURN,
    /// Apply a shadow-crossing yaw law, including GPS IIA post-shadow recovery.
    SHADOW,
    /// Orbit-normal mode.
    ORBIT_NORMAL,
    /// Yaw steering with a modified Sun elevation (BeiDou-3 SECM).
    MODIFIED_YAW_STEERING,
}

#[pymethods]
impl PyAttitudeMode {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyAttitudeMode> for gnss::AttitudeMode {
    fn from(value: PyAttitudeMode) -> Self {
        match value {
            PyAttitudeMode::NOMINAL => Self::Nominal,
            PyAttitudeMode::NOON_TURN => Self::NoonTurn,
            PyAttitudeMode::MIDNIGHT_TURN => Self::MidnightTurn,
            PyAttitudeMode::SHADOW => Self::Shadow,
            PyAttitudeMode::ORBIT_NORMAL => Self::OrbitNormal,
            PyAttitudeMode::MODIFIED_YAW_STEERING => Self::ModifiedYawSteering,
        }
    }
}

impl TryFrom<gnss::AttitudeMode> for PyAttitudeMode {
    type Error = PyErr;

    fn try_from(value: gnss::AttitudeMode) -> PyResult<Self> {
        match value {
            gnss::AttitudeMode::Nominal => Ok(Self::NOMINAL),
            gnss::AttitudeMode::NoonTurn => Ok(Self::NOON_TURN),
            gnss::AttitudeMode::MidnightTurn => Ok(Self::MIDNIGHT_TURN),
            gnss::AttitudeMode::Shadow => Ok(Self::SHADOW),
            gnss::AttitudeMode::OrbitNormal => Ok(Self::ORBIT_NORMAL),
            gnss::AttitudeMode::ModifiedYawSteering => Ok(Self::MODIFIED_YAW_STEERING),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported AttitudeMode value",
            )),
        }
    }
}

/// Angle whose crossing starts a cosine yaw turn.
#[pyclass(
    module = "sidereon._sidereon",
    name = "CosineTurnStartAngle",
    eq,
    eq_int,
    frozen
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum PyCosineTurnStartAngle {
    /// `cos(beta) cos(in-plane angle)`.
    COLINEARITY,
    /// The in-plane orbit angle from noon or midnight (Wang et al. 2018).
    ORBIT_ANGLE,
}

#[pymethods]
impl PyCosineTurnStartAngle {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyCosineTurnStartAngle> for gnss::CosineTurnStartAngle {
    fn from(value: PyCosineTurnStartAngle) -> Self {
        match value {
            PyCosineTurnStartAngle::COLINEARITY => Self::Colinearity,
            PyCosineTurnStartAngle::ORBIT_ANGLE => Self::OrbitAngle,
        }
    }
}

impl TryFrom<gnss::CosineTurnStartAngle> for PyCosineTurnStartAngle {
    type Error = PyErr;

    fn try_from(value: gnss::CosineTurnStartAngle) -> PyResult<Self> {
        match value {
            gnss::CosineTurnStartAngle::Colinearity => Ok(Self::COLINEARITY),
            gnss::CosineTurnStartAngle::OrbitAngle => Ok(Self::ORBIT_ANGLE),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "unsupported CosineTurnStartAngle value",
            )),
        }
    }
}

/// Manufacturer of a BeiDou-3 satellite.
#[pyclass(
    module = "sidereon._sidereon",
    name = "BeidouManufacturer",
    eq,
    eq_int,
    frozen,
    ord
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum PyBeidouManufacturer {
    /// China Academy of Space Technology.
    CAST,
    /// Shanghai Engineering Center for Microsatellites.
    SECM,
}

#[pymethods]
impl PyBeidouManufacturer {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyBeidouManufacturer> for gnss::BeidouManufacturer {
    fn from(value: PyBeidouManufacturer) -> Self {
        match value {
            PyBeidouManufacturer::CAST => Self::Cast,
            PyBeidouManufacturer::SECM => Self::Secm,
        }
    }
}

impl From<gnss::BeidouManufacturer> for PyBeidouManufacturer {
    fn from(value: gnss::BeidouManufacturer) -> Self {
        match value {
            gnss::BeidouManufacturer::Cast => Self::CAST,
            gnss::BeidouManufacturer::Secm => Self::SECM,
        }
    }
}

/// Orbit class of a BeiDou satellite.
#[pyclass(
    module = "sidereon._sidereon",
    name = "BeidouOrbitClass",
    eq,
    eq_int,
    frozen,
    ord
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum PyBeidouOrbitClass {
    /// Geostationary orbit.
    GEO,
    /// Inclined geosynchronous orbit.
    IGSO,
    /// Medium Earth orbit.
    MEO,
}

#[pymethods]
impl PyBeidouOrbitClass {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyBeidouOrbitClass> for gnss::BeidouOrbitClass {
    fn from(value: PyBeidouOrbitClass) -> Self {
        match value {
            PyBeidouOrbitClass::GEO => Self::Geo,
            PyBeidouOrbitClass::IGSO => Self::Igso,
            PyBeidouOrbitClass::MEO => Self::Meo,
        }
    }
}

impl From<gnss::BeidouOrbitClass> for PyBeidouOrbitClass {
    fn from(value: gnss::BeidouOrbitClass) -> Self {
        match value {
            gnss::BeidouOrbitClass::Geo => Self::GEO,
            gnss::BeidouOrbitClass::Igso => Self::IGSO,
            gnss::BeidouOrbitClass::Meo => Self::MEO,
        }
    }
}

create_exception!(
    _sidereon,
    GnssAttitudeError,
    SolveError,
    "A GNSS attitude evaluation failure."
);

fn attitude_error(error: gnss::GnssAttitudeError) -> PyErr {
    let exception = GnssAttitudeError::new_err(error.to_string());
    match error {
        gnss::GnssAttitudeError::NonFinite { field } => {
            binding_error(exception, "non_finite", |detail| {
                detail.set_item("field", field)
            })
        }
        gnss::GnssAttitudeError::ZeroLength { field } => {
            binding_error(exception, "zero_length", |detail| {
                detail.set_item("field", field)
            })
        }
        gnss::GnssAttitudeError::TrajectoryUnavailable { t_s } => {
            binding_error(exception, "trajectory_unavailable", |detail| {
                detail.set_item("t_s", t_s)
            })
        }
        gnss::GnssAttitudeError::InvalidParameter { field, value } => {
            binding_error(exception, "invalid_parameter", |detail| {
                detail.set_item("field", field)?;
                detail.set_item("value", value)
            })
        }
        gnss::GnssAttitudeError::NoDefaultYawRate { prn } => {
            binding_error(exception, "no_default_yaw_rate", |detail| {
                detail.set_item("prn", prn)
            })
        }
        gnss::GnssAttitudeError::DegenerateGeometry { t_s } => {
            binding_error(exception, "degenerate_geometry", |detail| {
                detail.set_item("t_s", t_s)
            })
        }
        gnss::GnssAttitudeError::BeidouManufacturerRequired { t_s } => {
            binding_error(exception, "beidou_manufacturer_required", |detail| {
                detail.set_item("t_s", t_s)
            })
        }
        gnss::GnssAttitudeError::TurnEndNotFound { t_s } => {
            binding_error(exception, "turn_end_not_found", |detail| {
                detail.set_item("t_s", t_s)
            })
        }
        gnss::GnssAttitudeError::OutsideValidatedDomain { t_s, reason } => {
            binding_error(exception, "outside_validated_domain", |detail| {
                detail.set_item("t_s", t_s)?;
                detail.set_item("reason", reason)
            })
        }
        _ => exception,
    }
}

create_exception!(
    _sidereon,
    GnssBlockError,
    SolveError,
    "An unrecognized or unsupported GNSS block identifier."
);

fn block_error(error: gnss::GnssBlockError) -> PyErr {
    let exception = GnssBlockError::new_err(error.to_string());
    match error {
        gnss::GnssBlockError::Unknown { name } => {
            binding_error(exception, "unknown", |detail| detail.set_item("name", name))
        }
        gnss::GnssBlockError::Unsupported { name } => {
            binding_error(exception, "unsupported", |detail| {
                detail.set_item("name", name)
            })
        }
        gnss::GnssBlockError::BeidouOrbitClassRequired { block_type } => {
            binding_error(exception, "beidou_orbit_class_required", |detail| {
                detail.set_item("block_type", block_type)
            })
        }
        _ => exception,
    }
}

/// A GNSS satellite block with a published attitude law.
#[pyclass(
    module = "sidereon._sidereon",
    name = "GnssBlock",
    eq,
    ord,
    hash,
    frozen
)]
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PyGnssBlock(gnss::GnssBlock);

#[pymethods]
impl PyGnssBlock {
    /// GpsIIA satellite block.
    #[classattr]
    const GPS_IIA: Self = Self(gnss::GnssBlock::GpsIIA);
    /// GpsIIR satellite block.
    #[classattr]
    const GPS_IIR: Self = Self(gnss::GnssBlock::GpsIIR);
    /// GpsIIRM satellite block.
    #[classattr]
    const GPS_IIRM: Self = Self(gnss::GnssBlock::GpsIIRM);
    /// GpsIIF satellite block.
    #[classattr]
    const GPS_IIF: Self = Self(gnss::GnssBlock::GpsIIF);
    /// GpsIII satellite block.
    #[classattr]
    const GPS_III: Self = Self(gnss::GnssBlock::GpsIII);
    /// GlonassM satellite block.
    #[classattr]
    const GLONASS_M: Self = Self(gnss::GnssBlock::GlonassM);
    /// GalileoIov satellite block.
    #[classattr]
    const GALILEO_IOV: Self = Self(gnss::GnssBlock::GalileoIov);
    /// GalileoFoc satellite block.
    #[classattr]
    const GALILEO_FOC: Self = Self(gnss::GnssBlock::GalileoFoc);
    /// Beidou2Geo satellite block.
    #[classattr]
    const BEIDOU2_GEO: Self = Self(gnss::GnssBlock::Beidou2Geo);
    /// Beidou2Igso satellite block.
    #[classattr]
    const BEIDOU2_IGSO: Self = Self(gnss::GnssBlock::Beidou2Igso);
    /// Beidou2Meo satellite block.
    #[classattr]
    const BEIDOU2_MEO: Self = Self(gnss::GnssBlock::Beidou2Meo);
    /// Beidou3Geo satellite block.
    #[classattr]
    const BEIDOU3_GEO: Self = Self(gnss::GnssBlock::Beidou3Geo);
    /// Beidou3Igso satellite block.
    #[classattr]
    const BEIDOU3_IGSO: Self = Self(gnss::GnssBlock::Beidou3Igso);
    /// BeiDou-3 MEO block with an optional manufacturer.
    #[staticmethod]
    #[pyo3(signature = (manufacturer=None))]
    fn beidou3_meo(manufacturer: Option<PyBeidouManufacturer>) -> Self {
        Self(gnss::GnssBlock::Beidou3Meo {
            manufacturer: manufacturer.map(Into::into),
        })
    }

    /// Parse an IGS or ANTEX satellite block name.
    #[staticmethod]
    fn from_igs_name(name: &str) -> PyResult<Self> {
        gnss::GnssBlock::from_igs_name(name)
            .map(Self)
            .map_err(block_error)
    }

    /// Parse a constellation block type and optional BeiDou orbit class.
    #[staticmethod]
    #[pyo3(signature = (system, block_type, beidou_orbit=None))]
    fn from_constellation_block_type(
        system: PyGnssSystem,
        block_type: &str,
        beidou_orbit: Option<PyBeidouOrbitClass>,
    ) -> PyResult<Self> {
        gnss::GnssBlock::from_constellation_block_type(
            system.into(),
            block_type,
            beidou_orbit.map(Into::into),
        )
        .map(Self)
        .map_err(block_error)
    }

    /// Canonical IGS name for the block.
    fn igs_name(&self) -> &'static str {
        self.0.igs_name()
    }

    /// Constellation of the block.
    fn system(&self) -> PyGnssSystem {
        self.0.system().into()
    }

    /// Canonical IGS display name.
    fn __str__(&self) -> String {
        self.0.to_string()
    }
}

/// Parameters shared by the Galileo FOC and BeiDou-3 CAST cosine yaw laws.
#[pyclass(
    module = "sidereon._sidereon",
    name = "CosineTurnParameters",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyCosineTurnParameters(gnss::CosineTurnParameters);

#[pymethods]
impl PyCosineTurnParameters {
    /// Construct cosine turn parameters.
    #[new]
    fn new(
        beta_limit_rad: f64,
        beta_limit_inclusive: bool,
        start_angle: PyCosineTurnStartAngle,
        start_angle_rad: f64,
        period_s: f64,
    ) -> Self {
        Self(gnss::CosineTurnParameters {
            beta_limit_rad,
            beta_limit_inclusive,
            start_angle: start_angle.into(),
            start_angle_rad,
            period_s,
        })
    }

    /// Sun elevation limit in radians.
    #[getter]
    fn beta_limit_rad(&self) -> f64 {
        self.0.beta_limit_rad
    }

    /// Whether the elevation limit is inclusive.
    #[getter]
    fn beta_limit_inclusive(&self) -> bool {
        self.0.beta_limit_inclusive
    }

    /// Type of angle that begins the turn.
    #[getter]
    fn start_angle(&self) -> PyResult<PyCosineTurnStartAngle> {
        self.0.start_angle.try_into()
    }

    /// Angle threshold in radians.
    #[getter]
    fn start_angle_rad(&self) -> f64 {
        self.0.start_angle_rad
    }

    /// Cosine period in seconds, twice the turn duration.
    #[getter]
    fn period_s(&self) -> f64 {
        self.0.period_s
    }
}

/// Parameterized GNSS satellite yaw law.
#[pyclass(module = "sidereon._sidereon", name = "GnssAttitudeLaw", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyGnssAttitudeLaw(gnss::GnssAttitudeLaw);

#[pymethods]
impl PyGnssAttitudeLaw {
    /// Construct the gps iia yaw law.
    #[staticmethod]
    fn gps_iia(yaw_rate_rad_s: f64, yaw_bias_rad: f64, shadow_limit_rad: f64) -> Self {
        Self(gnss::GnssAttitudeLaw::GpsIIA {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        })
    }

    /// Construct the gps iir yaw law.
    #[staticmethod]
    fn gps_iir(yaw_rate_rad_s: f64) -> Self {
        Self(gnss::GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s })
    }

    /// Construct the gps iif yaw law.
    #[staticmethod]
    fn gps_iif(yaw_rate_rad_s: f64, yaw_bias_rad: f64, shadow_limit_rad: f64) -> Self {
        Self(gnss::GnssAttitudeLaw::GpsIIF {
            yaw_rate_rad_s,
            yaw_bias_rad,
            shadow_limit_rad,
        })
    }

    /// Construct the glonass m yaw law.
    #[staticmethod]
    fn glonass_m(yaw_rate_rad_s: f64, shadow_limit_rad: f64) -> Self {
        Self(gnss::GnssAttitudeLaw::GlonassM {
            yaw_rate_rad_s,
            shadow_limit_rad,
        })
    }

    /// Construct the galileo iov yaw law.
    #[staticmethod]
    fn galileo_iov(beta_x_rad: f64, beta_y_rad: f64) -> Self {
        Self(gnss::GnssAttitudeLaw::GalileoIov {
            beta_x_rad,
            beta_y_rad,
        })
    }

    /// Construct the cosine turn yaw law.
    #[staticmethod]
    fn cosine_turn(parameters: &PyCosineTurnParameters) -> Self {
        Self(gnss::GnssAttitudeLaw::CosineTurn(parameters.0))
    }

    /// Construct the beidou orbit normal switch yaw law.
    #[staticmethod]
    fn beidou_orbit_normal_switch(ys_to_on_beta_rad: f64, on_to_ys_beta_rad: f64) -> Self {
        Self(gnss::GnssAttitudeLaw::BeidouOrbitNormalSwitch {
            ys_to_on_beta_rad,
            on_to_ys_beta_rad,
        })
    }

    /// Construct the fixed beta yaw steering yaw law.
    #[staticmethod]
    fn fixed_beta_yaw_steering(beta0_rad: f64) -> Self {
        Self(gnss::GnssAttitudeLaw::FixedBetaYawSteering { beta0_rad })
    }

    /// Construct the beidou3 meo unknown manufacturer yaw law.
    #[staticmethod]
    fn beidou3_meo_unknown_manufacturer(
        cast: &PyCosineTurnParameters,
        secm_beta0_rad: f64,
    ) -> Self {
        Self(gnss::GnssAttitudeLaw::Beidou3MeoUnknownManufacturer {
            cast: cast.0,
            secm_beta0_rad,
        })
    }

    /// Construct the orbit normal yaw law.
    #[staticmethod]
    fn orbit_normal() -> Self {
        Self(gnss::GnssAttitudeLaw::OrbitNormal)
    }

    /// Law variant, named after its constructor.
    #[getter]
    fn variant(&self) -> PyResult<&'static str> {
        use gnss::GnssAttitudeLaw::*;
        Ok(match self.0 {
            GpsIIA { .. } => "gps_iia",
            GpsIIR { .. } => "gps_iir",
            GpsIIF { .. } => "gps_iif",
            GlonassM { .. } => "glonass_m",
            GalileoIov { .. } => "galileo_iov",
            CosineTurn(_) => "cosine_turn",
            BeidouOrbitNormalSwitch { .. } => "beidou_orbit_normal_switch",
            FixedBetaYawSteering { .. } => "fixed_beta_yaw_steering",
            Beidou3MeoUnknownManufacturer { .. } => "beidou3_meo_unknown_manufacturer",
            OrbitNormal => "orbit_normal",
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "unsupported GnssAttitudeLaw variant",
                ))
            }
        })
    }

    /// Maximum yaw rate in radians per second. Returns None for laws without this field.
    #[getter]
    fn yaw_rate_rad_s(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::GpsIIA { yaw_rate_rad_s, .. }
            | gnss::GnssAttitudeLaw::GpsIIR { yaw_rate_rad_s, .. }
            | gnss::GnssAttitudeLaw::GpsIIF { yaw_rate_rad_s, .. }
            | gnss::GnssAttitudeLaw::GlonassM { yaw_rate_rad_s, .. } => Some(yaw_rate_rad_s),
            _ => None,
        }
    }

    /// Yaw bias in radians. Returns None for laws without this field.
    #[getter]
    fn yaw_bias_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::GpsIIA { yaw_bias_rad, .. }
            | gnss::GnssAttitudeLaw::GpsIIF { yaw_bias_rad, .. } => Some(yaw_bias_rad),
            _ => None,
        }
    }

    /// Sun-satellite angle in radians at the shadow limit. Returns None for laws without this field.
    #[getter]
    fn shadow_limit_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::GpsIIA {
                shadow_limit_rad, ..
            }
            | gnss::GnssAttitudeLaw::GpsIIF {
                shadow_limit_rad, ..
            }
            | gnss::GnssAttitudeLaw::GlonassM {
                shadow_limit_rad, ..
            } => Some(shadow_limit_rad),
            _ => None,
        }
    }

    /// Galileo IOV along-track Sun-angle threshold in radians. Returns None for laws without this field.
    #[getter]
    fn beta_x_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::GalileoIov { beta_x_rad, .. } => Some(beta_x_rad),
            _ => None,
        }
    }

    /// Galileo IOV Sun elevation threshold in radians. Returns None for laws without this field.
    #[getter]
    fn beta_y_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::GalileoIov { beta_y_rad, .. } => Some(beta_y_rad),
            _ => None,
        }
    }

    /// Sun elevation threshold for switching to orbit-normal mode, in radians. Returns None for laws without this field.
    #[getter]
    fn ys_to_on_beta_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::BeidouOrbitNormalSwitch {
                ys_to_on_beta_rad, ..
            } => Some(ys_to_on_beta_rad),
            _ => None,
        }
    }

    /// Sun elevation threshold for switching to yaw steering, in radians. Returns None for laws without this field.
    #[getter]
    fn on_to_ys_beta_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::BeidouOrbitNormalSwitch {
                on_to_ys_beta_rad, ..
            } => Some(on_to_ys_beta_rad),
            _ => None,
        }
    }

    /// Fixed Sun elevation magnitude in radians for the yaw-steering law. Returns None for laws without this field.
    #[getter]
    fn beta0_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::FixedBetaYawSteering { beta0_rad, .. } => Some(beta0_rad),
            _ => None,
        }
    }

    /// SECM fixed Sun elevation magnitude in radians. Returns None for laws without this field.
    #[getter]
    fn secm_beta0_rad(&self) -> Option<f64> {
        match self.0 {
            gnss::GnssAttitudeLaw::Beidou3MeoUnknownManufacturer { secm_beta0_rad, .. } => {
                Some(secm_beta0_rad)
            }
            _ => None,
        }
    }

    /// Cosine turn parameters for the cosine turn law, if present.
    #[getter]
    fn cosine_turn_parameters(&self) -> Option<PyCosineTurnParameters> {
        match self.0 {
            gnss::GnssAttitudeLaw::CosineTurn(parameters) => {
                Some(PyCosineTurnParameters(parameters))
            }
            _ => None,
        }
    }

    /// CAST turn parameters for the combined BeiDou-3 law, if present.
    #[getter]
    fn cast(&self) -> Option<PyCosineTurnParameters> {
        match self.0 {
            gnss::GnssAttitudeLaw::Beidou3MeoUnknownManufacturer { cast, .. } => {
                Some(PyCosineTurnParameters(cast))
            }
            _ => None,
        }
    }

    /// Use the published law and its parameters for a satellite block.
    #[staticmethod]
    #[pyo3(signature = (block, prn=None))]
    fn for_block(block: &PyGnssBlock, prn: Option<u8>) -> PyResult<Self> {
        gnss::GnssAttitudeLaw::for_block(block.0, prn)
            .map(Self)
            .map_err(attitude_error)
    }
}

/// Inertial satellite state and Sun position at one epoch.
#[pyclass(module = "sidereon._sidereon", name = "AttitudeSample", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyAttitudeSample(gnss::AttitudeSample);

#[pymethods]
impl PyAttitudeSample {
    /// Build an attitude sample from inertial position, velocity, and Sun vectors.
    #[new]
    fn new(position_m: [f64; 3], velocity_m_s: [f64; 3], sun_position_m: [f64; 3]) -> Self {
        Self(gnss::AttitudeSample {
            position_m,
            velocity_m_s,
            sun_position_m,
        })
    }

    /// Satellite position in metres.
    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.0.position_m
    }

    /// Satellite velocity in metres per second.
    #[getter]
    fn velocity_m_s(&self) -> [f64; 3] {
        self.0.velocity_m_s
    }

    /// Sun position in metres.
    #[getter]
    fn sun_position_m(&self) -> [f64; 3] {
        self.0.sun_position_m
    }
}

/// Sun-fixed D, Y and B unit axes.
#[pyclass(module = "sidereon._sidereon", name = "DybAxes", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyDybAxes(pub(crate) gnss::DybAxes);

#[pymethods]
impl PyDybAxes {
    /// Unit vector from satellite to Sun.
    #[getter]
    fn d(&self) -> [f64; 3] {
        self.0.d
    }

    /// Nominal solar panel axis.
    #[getter]
    fn y(&self) -> [f64; 3] {
        self.0.y
    }

    /// Cross product of D and Y.
    #[getter]
    fn b(&self) -> [f64; 3] {
        self.0.b
    }
}

/// Nominal satellite yaw and body axes.
#[pyclass(module = "sidereon._sidereon", name = "NominalYaw", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyNominalYaw(gnss::NominalYaw);

#[pymethods]
impl PyNominalYaw {
    /// Yaw in radians.
    #[getter]
    fn yaw_rad(&self) -> f64 {
        self.0.yaw_rad
    }

    /// Sun elevation in radians.
    #[getter]
    fn beta_rad(&self) -> f64 {
        self.0.beta_rad
    }

    /// Unit positive X body axis.
    #[getter]
    fn body_x(&self) -> [f64; 3] {
        self.0.body_x
    }

    /// Unit positive Y body axis.
    #[getter]
    fn body_y(&self) -> [f64; 3] {
        self.0.body_y
    }

    /// Unit positive Z body axis.
    #[getter]
    fn body_z(&self) -> [f64; 3] {
        self.0.body_z
    }
}

/// GNSS satellite attitude and Sun geometry.
#[pyclass(module = "sidereon._sidereon", name = "GnssAttitude", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyGnssAttitude(gnss::GnssAttitude);

#[pymethods]
impl PyGnssAttitude {
    /// Active mode of the yaw law.
    #[getter]
    fn mode(&self) -> PyResult<PyAttitudeMode> {
        self.0.mode.try_into()
    }

    /// Yaw in radians.
    #[getter]
    fn yaw_rad(&self) -> f64 {
        self.0.yaw_rad
    }

    /// Nominal yaw in radians.
    #[getter]
    fn nominal_yaw_rad(&self) -> f64 {
        self.0.nominal_yaw_rad
    }

    /// Sun elevation in radians.
    #[getter]
    fn beta_rad(&self) -> f64 {
        self.0.beta_rad
    }

    /// Orbit angle from midnight in radians.
    #[getter]
    fn orbit_angle_rad(&self) -> f64 {
        self.0.orbit_angle_rad
    }

    /// Unit positive X body axis.
    #[getter]
    fn body_x(&self) -> [f64; 3] {
        self.0.body_x
    }

    /// Unit positive Y body axis.
    #[getter]
    fn body_y(&self) -> [f64; 3] {
        self.0.body_y
    }

    /// Unit positive Z body axis.
    #[getter]
    fn body_z(&self) -> [f64; 3] {
        self.0.body_z
    }

    /// Sun-fixed axes if defined.
    #[getter]
    fn dyb(&self) -> Option<PyDybAxes> {
        self.0.dyb.map(PyDybAxes)
    }
}

enum EarthFixedOrbit {
    Sp3(Py<PySp3>),
    Broadcast(Py<PyBroadcastEphemeris>),
    Scoring(PyOrbitSource),
}

enum AttitudeOrbit {
    EcksteinHechler(Py<PyEcksteinHechlerPropagator>),
    BrouwerLyddane(Py<PyBrouwerLyddanePropagator>),
    DenseOutput(PyDenseOutput),
    EarthFixed {
        orbit: EarthFixedOrbit,
        satellite: GnssSatelliteId,
        orientation: Option<PyIers2010EarthOrientationProvider>,
    },
}

/// Orbit and Sun samples in GCRF, queried in TDB seconds past J2000.
#[pyclass(
    module = "sidereon._sidereon",
    name = "OrbitAttitudeTrajectory",
    frozen
)]
#[derive(Clone)]
pub struct PyOrbitAttitudeTrajectory {
    orbit: Arc<AttitudeOrbit>,
    ephemeris: Option<PyDeEphemeris>,
    velocity_half_step_s: f64,
}

impl PyOrbitAttitudeTrajectory {
    fn new(orbit: AttitudeOrbit) -> Self {
        Self {
            orbit: Arc::new(orbit),
            ephemeris: None,
            velocity_half_step_s: DEFAULT_VELOCITY_HALF_STEP_S,
        }
    }

    fn earth_fixed(
        orbit: EarthFixedOrbit,
        satellite: &str,
        orientation: Option<PyIers2010EarthOrientationProvider>,
    ) -> PyResult<Self> {
        Ok(Self::new(AttitudeOrbit::EarthFixed {
            orbit,
            satellite: parse_sat(satellite)?,
            orientation,
        }))
    }

    fn with_core<T>(
        &self,
        py: Python<'_>,
        evaluate: impl FnOnce(OrbitAttitudeTrajectory<'_>) -> PyResult<T>,
    ) -> PyResult<T> {
        let evaluate = |trajectory: OrbitAttitudeTrajectory<'_>| {
            let trajectory = trajectory
                .with_velocity_half_step_s(self.velocity_half_step_s)
                .map_err(orbit_score_error)?;
            let trajectory = match &self.ephemeris {
                Some(ephemeris) => trajectory.with_de_ephemeris(&ephemeris.0),
                None => trajectory,
            };
            evaluate(trajectory)
        };

        match self.orbit.as_ref() {
            AttitudeOrbit::EcksteinHechler(propagator) => evaluate(
                OrbitAttitudeTrajectory::from_eckstein_hechler(&propagator.borrow(py).0),
            ),
            AttitudeOrbit::BrouwerLyddane(propagator) => evaluate(
                OrbitAttitudeTrajectory::from_brouwer_lyddane(&propagator.borrow(py).0),
            ),
            AttitudeOrbit::DenseOutput(output) => {
                evaluate(OrbitAttitudeTrajectory::from_dense_output(&output.0))
            }
            AttitudeOrbit::EarthFixed {
                orbit,
                satellite,
                orientation,
            } => {
                let default_orientation = TdbEarthOrientationProvider::new();
                let provider: &dyn EarthOrientationProvider = match orientation {
                    Some(provider) => provider.0.as_ref(),
                    None => &default_orientation,
                };
                match orbit {
                    EarthFixedOrbit::Sp3(product) => evaluate(OrbitAttitudeTrajectory::from_sp3(
                        &product.borrow(py).inner,
                        *satellite,
                        provider,
                    )),
                    EarthFixedOrbit::Broadcast(ephemeris) => {
                        evaluate(OrbitAttitudeTrajectory::from_broadcast(
                            &ephemeris.borrow(py).inner,
                            *satellite,
                            provider,
                        ))
                    }
                    EarthFixedOrbit::Scoring(source) => source.with_source(py, |source| {
                        evaluate(OrbitAttitudeTrajectory::from_orbit_source(
                            source, *satellite, provider,
                        ))
                    }),
                }
            }
        }
    }
}

#[pymethods]
impl PyOrbitAttitudeTrajectory {
    /// Use a parsed SP3 product and the selected satellite.
    /// An omitted orientation uses the default TDB provider.
    #[staticmethod]
    #[pyo3(signature = (product, satellite, orientation=None))]
    fn from_sp3(
        product: Py<PySp3>,
        satellite: &str,
        orientation: Option<PyIers2010EarthOrientationProvider>,
    ) -> PyResult<Self> {
        Self::earth_fixed(EarthFixedOrbit::Sp3(product), satellite, orientation)
    }

    /// Use a broadcast navigation set and the selected satellite.
    /// An omitted orientation uses the default TDB provider.
    #[staticmethod]
    #[pyo3(signature = (ephemeris, satellite, orientation=None))]
    fn from_broadcast(
        ephemeris: Py<PyBroadcastEphemeris>,
        satellite: &str,
        orientation: Option<PyIers2010EarthOrientationProvider>,
    ) -> PyResult<Self> {
        Self::earth_fixed(
            EarthFixedOrbit::Broadcast(ephemeris),
            satellite,
            orientation,
        )
    }

    /// Use an Earth-fixed orbit scoring source and the selected satellite.
    /// An omitted orientation uses the default TDB provider.
    #[staticmethod]
    #[pyo3(signature = (source, satellite, orientation=None))]
    fn from_orbit_source(
        source: PyOrbitSource,
        satellite: &str,
        orientation: Option<PyIers2010EarthOrientationProvider>,
    ) -> PyResult<Self> {
        Self::earth_fixed(EarthFixedOrbit::Scoring(source), satellite, orientation)
    }

    /// Use an Eckstein-Hechler orbit defined in GCRF.
    #[staticmethod]
    fn from_eckstein_hechler(propagator: Py<PyEcksteinHechlerPropagator>) -> Self {
        Self::new(AttitudeOrbit::EcksteinHechler(propagator))
    }

    /// Use a Brouwer-Lyddane orbit defined in GCRF.
    #[staticmethod]
    fn from_brouwer_lyddane(propagator: Py<PyBrouwerLyddanePropagator>) -> Self {
        Self::new(AttitudeOrbit::BrouwerLyddane(propagator))
    }

    /// Use continuous numerical output defined in GCRF.
    #[staticmethod]
    fn from_dense_output(output: PyDenseOutput) -> Self {
        Self::new(AttitudeOrbit::DenseOutput(output))
    }

    /// Return a trajectory using the supplied JPL DE ephemeris for the Sun.
    fn with_de_ephemeris(&self, ephemeris: PyDeEphemeris) -> Self {
        Self {
            ephemeris: Some(ephemeris),
            ..self.clone()
        }
    }

    /// Return a trajectory with a different velocity half step on the source time axis.
    fn with_velocity_half_step_s(&self, py: Python<'_>, half_step_s: f64) -> PyResult<Self> {
        let trajectory = Self {
            velocity_half_step_s: half_step_s,
            ..self.clone()
        };
        trajectory.with_core(py, |_| Ok(()))?;
        Ok(trajectory)
    }

    /// Sample GCRF position, velocity, and Sun position at TDB seconds past J2000.
    fn sample(&self, py: Python<'_>, t_s: f64) -> PyResult<Option<PyAttitudeSample>> {
        self.with_core(py, |trajectory| {
            Ok(trajectory.sample(t_s).map(PyAttitudeSample))
        })
    }
}

/// Nominal yaw steering from one inertial satellite state and Sun position.
#[pyfunction]
fn nominal_yaw(
    position_m: [f64; 3],
    velocity_m_s: [f64; 3],
    sun_position_m: [f64; 3],
) -> PyResult<PyNominalYaw> {
    gnss::nominal_yaw(position_m, velocity_m_s, sun_position_m)
        .map(PyNominalYaw)
        .map_err(attitude_error)
}

/// Sun-fixed axes from inertial satellite and Sun positions.
#[pyfunction]
fn dyb_axes(position_m: [f64; 3], sun_position_m: [f64; 3]) -> PyResult<PyDybAxes> {
    gnss::dyb_axes(position_m, sun_position_m)
        .map(PyDybAxes)
        .map_err(attitude_error)
}

/// Evaluate a GNSS yaw law in GCRF at TDB seconds past J2000.
#[pyfunction]
fn gnss_attitude(
    py: Python<'_>,
    law: &PyGnssAttitudeLaw,
    trajectory: &PyOrbitAttitudeTrajectory,
    t_s: f64,
) -> PyResult<PyGnssAttitude> {
    trajectory.with_core(py, |trajectory| {
        gnss::gnss_attitude(&law.0, &trajectory, t_s)
            .map(PyGnssAttitude)
            .map_err(attitude_error)
    })
}

/// Published GPS IIA maximum yaw rate for a PRN in radians per second.
#[pyfunction]
fn gps_iia_default_yaw_rate_rad_s(prn: u8) -> Option<f64> {
    gnss::gps_iia_default_yaw_rate_rad_s(prn)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAttitudeMode>()?;
    m.add_class::<PyCosineTurnStartAngle>()?;
    m.add_class::<PyBeidouManufacturer>()?;
    m.add_class::<PyBeidouOrbitClass>()?;
    m.add("GnssAttitudeError", m.py().get_type::<GnssAttitudeError>())?;
    m.add("GnssBlockError", m.py().get_type::<GnssBlockError>())?;
    m.add_class::<PyGnssBlock>()?;
    m.add_class::<PyCosineTurnParameters>()?;
    m.add_class::<PyGnssAttitudeLaw>()?;
    m.add_class::<PyAttitudeSample>()?;
    m.add_class::<PyDybAxes>()?;
    m.add_class::<PyNominalYaw>()?;
    m.add_class::<PyGnssAttitude>()?;
    m.add_class::<PyOrbitAttitudeTrajectory>()?;
    m.add_function(wrap_pyfunction!(nominal_yaw, m)?)?;
    m.add_function(wrap_pyfunction!(dyb_axes, m)?)?;
    m.add_function(wrap_pyfunction!(gnss_attitude, m)?)?;
    m.add_function(wrap_pyfunction!(gps_iia_default_yaw_rate_rad_s, m)?)?;
    m.add(
        "GPS_IIA_DEFAULT_YAW_BIAS_RAD",
        gnss::GPS_IIA_DEFAULT_YAW_BIAS_RAD,
    )?;
    m.add(
        "GPS_IIR_DEFAULT_YAW_RATE_RAD_S",
        gnss::GPS_IIR_DEFAULT_YAW_RATE_RAD_S,
    )?;
    m.add(
        "GPS_IIF_DEFAULT_YAW_RATE_RAD_S",
        gnss::GPS_IIF_DEFAULT_YAW_RATE_RAD_S,
    )?;
    m.add(
        "GPS_IIF_DEFAULT_YAW_BIAS_RAD",
        gnss::GPS_IIF_DEFAULT_YAW_BIAS_RAD,
    )?;
    m.add("GPS_SHADOW_LIMIT_RAD", gnss::GPS_SHADOW_LIMIT_RAD)?;
    m.add(
        "GLONASS_M_DEFAULT_YAW_RATE_RAD_S",
        gnss::GLONASS_M_DEFAULT_YAW_RATE_RAD_S,
    )?;
    m.add(
        "GLONASS_M_SHADOW_LIMIT_RAD",
        gnss::GLONASS_M_SHADOW_LIMIT_RAD,
    )?;
    m.add("GALILEO_IOV_BETA_X_RAD", gnss::GALILEO_IOV_BETA_X_RAD)?;
    m.add("GALILEO_IOV_BETA_Y_RAD", gnss::GALILEO_IOV_BETA_Y_RAD)?;
    m.add("GPS_III_BETA_X_RAD", gnss::GPS_III_BETA_X_RAD)?;
    m.add("GPS_III_BETA_Y_RAD", gnss::GPS_III_BETA_Y_RAD)?;
    m.add(
        "GALILEO_FOC_TURN",
        PyCosineTurnParameters(gnss::GALILEO_FOC_TURN),
    )?;
    m.add(
        "BEIDOU3_CAST_MEO_TURN",
        PyCosineTurnParameters(gnss::BEIDOU3_CAST_MEO_TURN),
    )?;
    m.add(
        "BEIDOU3_CAST_IGSO_TURN",
        PyCosineTurnParameters(gnss::BEIDOU3_CAST_IGSO_TURN),
    )?;
    m.add("BEIDOU3_SECM_BETA0_RAD", gnss::BEIDOU3_SECM_BETA0_RAD)?;
    m.add("BEIDOU2_YS_TO_ON_BETA_RAD", gnss::BEIDOU2_YS_TO_ON_BETA_RAD)?;
    m.add("BEIDOU2_ON_TO_YS_BETA_RAD", gnss::BEIDOU2_ON_TO_YS_BETA_RAD)?;
    Ok(())
}
