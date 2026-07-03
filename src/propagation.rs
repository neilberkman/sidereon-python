//! SGP4 / TLE propagation + topocentric look-angle binding.
//!
//! Marshals a parsed TLE plus a numpy epoch grid into the core's
//! [`sidereon::passes`] arc walkers and returns numpy-native states / angles.
//! No modeling: the satellite is built once by
//! [`sidereon::sgp4::Satellite::from_tle_with_opsmode`] and stepped by
//! [`sidereon::passes::propagate_teme_arc`] / [`sidereon::passes::look_angle_arc`],
//! so the numbers are exactly what the engine produces. Epochs cross the FFI
//! boundary once as a 1-D `int64` array of unix microseconds.

use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon::geometry::visible_at_elevation_mask;
use sidereon::passes::{
    find_passes_for_satellite, ground_track as core_ground_track, look_angle_arc,
    look_angle_batch_parallel, look_angle_batch_serial, propagate_teme_arc,
    propagate_teme_batch_parallel, propagate_teme_batch_serial,
    visible_from_satellites as core_visible_from_satellites, GroundStation, PassFinderOptions,
    UtcInstant, VisibleSatellite,
};
use sidereon::propagator::api::IntegratorOptions;
use sidereon::propagator::{
    propagate_states, IntegratorKind, PropagationConfig, PropagationForceModel,
};
use sidereon::sgp4::{
    fit_tle as core_fit_tle, parse_tle_file_with_opsmode, ElementSet, FitConfig, FitEpoch,
    FitSample, FitStatistics, JulianDate as Sgp4JulianDate, Loss as CoreLoss,
    OpsMode as CoreOpsMode, Satellite, TleFit, TleMetadata, XScale,
};
use sidereon::tle::{
    encode as encode_tle_lines, parse as parse_tle_lines, ChecksumWarning, TleElements,
};

use crate::events::PyWgs84Geodetic;
use crate::forces::PyDragParameters;
use crate::marshal::{
    fixed_array, instants_from_unix_micros, rows3_to_array, rows6_to_array, scalar_rows_to_array2,
    vec3_rows_to_array3, EmptyPolicy, FinitePolicy,
};
use crate::omm::PyOmm;
use crate::space_weather::PySpaceWeatherTable;
use crate::{to_solve_err, to_tle_err, SolveError, TleParseError};

/// SGP4 operation mode for TLE initialization.
#[pyclass(module = "sidereon._sidereon", name = "OpsMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyOpsMode {
    /// AFSPC-compatible mode.
    AFSPC,
    /// Improved Vallado mode.
    IMPROVED,
}

impl PyOpsMode {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "afspc" => Ok(Self::AFSPC),
            "improved" => Ok(Self::IMPROVED),
            other => Err(PyValueError::new_err(format!(
                "unknown opsmode {other:?}; expected \"afspc\" or \"improved\""
            ))),
        }
    }
}

impl From<PyOpsMode> for CoreOpsMode {
    fn from(mode: PyOpsMode) -> Self {
        match mode {
            PyOpsMode::AFSPC => CoreOpsMode::Afspc,
            PyOpsMode::IMPROVED => CoreOpsMode::Improved,
        }
    }
}

#[pymethods]
impl PyOpsMode {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            PyOpsMode::AFSPC => "afspc",
            PyOpsMode::IMPROVED => "improved",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyOpsMode::AFSPC => "OpsMode.AFSPC",
            PyOpsMode::IMPROVED => "OpsMode.IMPROVED",
        }
    }
}

fn extract_opsmode(obj: &Bound<'_, PyAny>) -> PyResult<PyOpsMode> {
    if let Ok(mode) = obj.extract::<PyOpsMode>() {
        return Ok(mode);
    }
    PyOpsMode::from_label(&obj.extract::<String>()?)
}

/// Numerical propagation force model.
#[pyclass(module = "sidereon._sidereon", name = "ForceModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyForceModel {
    /// Point-mass two-body gravity.
    TWO_BODY,
    /// Two-body gravity plus Earth J2 oblateness.
    TWO_BODY_J2,
}

impl PyForceModel {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "two_body" => Ok(Self::TWO_BODY),
            "two_body_j2" => Ok(Self::TWO_BODY_J2),
            other => Err(PyValueError::new_err(format!(
                "unknown force_model {other:?}; expected \"two_body\" or \"two_body_j2\""
            ))),
        }
    }

    pub(crate) fn to_core(self) -> PropagationForceModel {
        match self {
            PyForceModel::TWO_BODY => PropagationForceModel::TwoBody,
            PyForceModel::TWO_BODY_J2 => PropagationForceModel::TwoBodyJ2,
        }
    }
}

#[pymethods]
impl PyForceModel {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            PyForceModel::TWO_BODY => "two_body",
            PyForceModel::TWO_BODY_J2 => "two_body_j2",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyForceModel::TWO_BODY => "ForceModel.TWO_BODY",
            PyForceModel::TWO_BODY_J2 => "ForceModel.TWO_BODY_J2",
        }
    }
}

pub(crate) fn extract_force_model(obj: &Bound<'_, PyAny>) -> PyResult<PyForceModel> {
    if let Ok(model) = obj.extract::<PyForceModel>() {
        return Ok(model);
    }
    PyForceModel::from_label(&obj.extract::<String>()?)
}

/// Numerical propagation integrator.
#[pyclass(module = "sidereon._sidereon", name = "Integrator", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyIntegrator {
    /// Dormand-Prince 5(4) adaptive integrator.
    DP54,
    /// Fixed-step fourth-order Runge-Kutta integrator.
    RK4,
}

impl PyIntegrator {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "dp54" => Ok(Self::DP54),
            "rk4" => Ok(Self::RK4),
            other => Err(PyValueError::new_err(format!(
                "unknown integrator {other:?}; expected \"dp54\" or \"rk4\""
            ))),
        }
    }
}

impl From<PyIntegrator> for IntegratorKind {
    fn from(integrator: PyIntegrator) -> Self {
        match integrator {
            PyIntegrator::DP54 => IntegratorKind::Dp54,
            PyIntegrator::RK4 => IntegratorKind::Rk4,
        }
    }
}

#[pymethods]
impl PyIntegrator {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            PyIntegrator::DP54 => "dp54",
            PyIntegrator::RK4 => "rk4",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyIntegrator::DP54 => "Integrator.DP54",
            PyIntegrator::RK4 => "Integrator.RK4",
        }
    }
}

pub(crate) fn extract_integrator(obj: &Bound<'_, PyAny>) -> PyResult<PyIntegrator> {
    if let Ok(integrator) = obj.extract::<PyIntegrator>() {
        return Ok(integrator);
    }
    PyIntegrator::from_label(&obj.extract::<String>()?)
}

/// A geodetic ground station: WGS84 latitude/longitude in degrees and altitude
/// in metres. Pass to [`Tle.look_angles`].
#[pyclass(module = "sidereon._sidereon", name = "GroundStation")]
#[derive(Clone)]
pub struct PyGroundStation {
    inner: GroundStation,
}

impl PyGroundStation {
    /// The core ground station this handle wraps, for sibling binding modules.
    pub(crate) fn core(&self) -> GroundStation {
        self.inner
    }
}

#[pymethods]
impl PyGroundStation {
    #[new]
    #[pyo3(signature = (latitude_deg, longitude_deg, altitude_m=0.0))]
    fn new(latitude_deg: f64, longitude_deg: f64, altitude_m: f64) -> Self {
        Self {
            inner: GroundStation {
                latitude_deg,
                longitude_deg,
                altitude_m,
            },
        }
    }

    #[getter]
    fn latitude_deg(&self) -> f64 {
        self.inner.latitude_deg
    }

    #[getter]
    fn longitude_deg(&self) -> f64 {
        self.inner.longitude_deg
    }

    #[getter]
    fn altitude_m(&self) -> f64 {
        self.inner.altitude_m
    }

    fn __repr__(&self) -> String {
        format!(
            "GroundStation(latitude_deg={}, longitude_deg={}, altitude_m={})",
            self.inner.latitude_deg, self.inner.longitude_deg, self.inner.altitude_m
        )
    }
}

/// TEME states from a batched SGP4 propagation: `position_km` and
/// `velocity_km_s` as numpy `float64` arrays of shape `(n_epochs, 3)`.
#[pyclass(module = "sidereon._sidereon", name = "TlePropagation")]
pub struct PyTlePropagation {
    positions_km: Vec<[f64; 3]>,
    velocities_km_s: Vec<[f64; 3]>,
}

#[pymethods]
impl PyTlePropagation {
    /// TEME positions as a numpy array of shape `(n_epochs, 3)`, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.positions_km)
    }

    /// TEME velocities as a numpy array of shape `(n_epochs, 3)`, km/s.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.velocities_km_s)
    }

    /// Number of epochs propagated.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.positions_km.len()
    }

    fn __len__(&self) -> usize {
        self.positions_km.len()
    }

    fn __repr__(&self) -> String {
        format!("TlePropagation(epoch_count={})", self.positions_km.len())
    }
}

/// Topocentric look angles from a batched arc: `azimuth_deg`, `elevation_deg`
/// (each numpy `float64` of shape `(n_epochs,)`), and `range_km`.
#[pyclass(module = "sidereon._sidereon", name = "LookAngles")]
pub struct PyLookAngles {
    azimuth_deg: Vec<f64>,
    elevation_deg: Vec<f64>,
    range_km: Vec<f64>,
}

#[pymethods]
impl PyLookAngles {
    /// Azimuth in degrees, clockwise from north, as a numpy array `(n_epochs,)`.
    #[getter]
    fn azimuth_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.azimuth_deg)
    }

    /// Elevation in degrees above the horizon, as a numpy array `(n_epochs,)`.
    #[getter]
    fn elevation_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.elevation_deg)
    }

    /// Slant range in kilometres, as a numpy array `(n_epochs,)`.
    #[getter]
    fn range_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.range_km)
    }

    /// Number of epochs evaluated.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.azimuth_deg.len()
    }

    fn __len__(&self) -> usize {
        self.azimuth_deg.len()
    }

    fn __repr__(&self) -> String {
        format!("LookAngles(epoch_count={})", self.azimuth_deg.len())
    }
}

/// Per-epoch topocentric visibility for one TLE plus the pass list over the
/// grid window.
///
/// `epoch_unix_us` is the UTC unix-microsecond input grid. `azimuth_deg`,
/// `elevation_deg`, and `range_km` have shape `(n_epochs,)`; `visible` is true
/// where `elevation_deg >= elevation_mask_deg`. `passes` is computed by the
/// core dense pass finder over `[epoch_unix_us[0], epoch_unix_us[-1]]`.
#[pyclass(module = "sidereon._sidereon", name = "VisibilitySeries")]
pub struct PyVisibilitySeries {
    epochs_unix_us: Vec<i64>,
    azimuth_deg: Vec<f64>,
    elevation_deg: Vec<f64>,
    range_km: Vec<f64>,
    visible: Vec<bool>,
    passes: Vec<PySatellitePass>,
}

#[pymethods]
impl PyVisibilitySeries {
    /// Epoch grid, UTC unix microseconds, as a numpy `int64` array `(n_epochs,)`.
    #[getter]
    fn epoch_unix_us<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.epochs_unix_us)
    }

    /// Azimuth in degrees clockwise from north, shape `(n_epochs,)`.
    #[getter]
    fn azimuth_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.azimuth_deg)
    }

    /// Elevation in degrees above the horizon, shape `(n_epochs,)`.
    #[getter]
    fn elevation_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.elevation_deg)
    }

    /// Slant range in kilometres, shape `(n_epochs,)`.
    #[getter]
    fn range_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.range_km)
    }

    /// Boolean visibility mask, shape `(n_epochs,)`.
    #[getter]
    fn visible<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.visible)
    }

    /// Dense pass-finder results over the epoch-grid window.
    #[getter]
    fn passes(&self) -> Vec<PySatellitePass> {
        self.passes.clone()
    }

    /// Number of epochs evaluated.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.epochs_unix_us.len()
    }

    /// Number of passes found over the epoch-grid window.
    #[getter]
    fn pass_count(&self) -> usize {
        self.passes.len()
    }

    fn __len__(&self) -> usize {
        self.epochs_unix_us.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "VisibilitySeries(epoch_count={}, pass_count={})",
            self.epochs_unix_us.len(),
            self.passes.len()
        )
    }
}

/// A satellite pass over a ground station: acquisition of signal (rise above the
/// elevation mask), loss of signal (set below it), the culmination time, and the
/// elevation at culmination. Times are unix microseconds (UTC), matching the
/// epoch convention used by [`Tle.propagate`] / [`Tle.look_angles`].
#[pyclass(module = "sidereon._sidereon", name = "SatellitePass")]
#[derive(Clone, Copy)]
pub struct PySatellitePass {
    aos_unix_us: i64,
    los_unix_us: i64,
    culmination_unix_us: i64,
    max_elevation_deg: f64,
}

#[pymethods]
impl PySatellitePass {
    /// Acquisition of signal (rise above the mask), unix microseconds UTC.
    #[getter]
    fn aos_unix_us(&self) -> i64 {
        self.aos_unix_us
    }

    /// Loss of signal (set below the mask), unix microseconds UTC.
    #[getter]
    fn los_unix_us(&self) -> i64 {
        self.los_unix_us
    }

    /// Culmination (maximum elevation) time, unix microseconds UTC.
    #[getter]
    fn culmination_unix_us(&self) -> i64 {
        self.culmination_unix_us
    }

    /// Elevation at culmination, degrees.
    #[getter]
    fn max_elevation_deg(&self) -> f64 {
        self.max_elevation_deg
    }

    /// Pass duration (LOS minus AOS), seconds.
    #[getter]
    fn duration_s(&self) -> f64 {
        (self.los_unix_us - self.aos_unix_us) as f64 / 1.0e6
    }

    fn __repr__(&self) -> String {
        format!(
            "SatellitePass(aos_unix_us={}, los_unix_us={}, culmination_unix_us={}, max_elevation_deg={:.4})",
            self.aos_unix_us, self.los_unix_us, self.culmination_unix_us, self.max_elevation_deg
        )
    }
}

fn to_py_pass(pass: &sidereon::passes::SatellitePass) -> PySatellitePass {
    PySatellitePass {
        aos_unix_us: pass.aos.unix_microseconds(),
        los_unix_us: pass.los.unix_microseconds(),
        culmination_unix_us: pass.culmination.unix_microseconds(),
        max_elevation_deg: pass.max_elevation_deg,
    }
}

/// One satellite above a ground station's horizon at a single instant.
///
/// `catalog_number` is the identity supplied for the corresponding satellite
/// (its `ids[i]`). `azimuth_deg` / `elevation_deg` are topocentric degrees and
/// `range_km` is the slant range; `position_km` is the satellite's TEME position
/// `(3,)` in kilometres at the instant. Returned by [`visible_from_satellites`].
#[pyclass(module = "sidereon._sidereon", name = "VisibleSatellite")]
pub struct PyVisibleSatellite {
    catalog_number: String,
    azimuth_deg: f64,
    elevation_deg: f64,
    range_km: f64,
    position_km: [f64; 3],
}

impl PyVisibleSatellite {
    fn from_core(v: VisibleSatellite) -> Self {
        Self {
            catalog_number: v.catalog_number,
            azimuth_deg: v.azimuth_deg,
            elevation_deg: v.elevation_deg,
            range_km: v.range_km,
            position_km: v.position_km,
        }
    }
}

#[pymethods]
impl PyVisibleSatellite {
    /// The caller-supplied identity for this satellite (its `ids[i]`).
    #[getter]
    fn catalog_number(&self) -> String {
        self.catalog_number.clone()
    }

    /// Topocentric azimuth, degrees clockwise from north.
    #[getter]
    fn azimuth_deg(&self) -> f64 {
        self.azimuth_deg
    }

    /// Topocentric elevation, degrees above the horizon.
    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.elevation_deg
    }

    /// Slant range to the satellite, kilometres.
    #[getter]
    fn range_km(&self) -> f64 {
        self.range_km
    }

    /// Satellite TEME position at the instant, numpy `(3,)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.position_km)
    }

    fn __repr__(&self) -> String {
        format!(
            "VisibleSatellite(catalog_number={:?}, azimuth_deg={:.4}, elevation_deg={:.4}, range_km={:.4})",
            self.catalog_number, self.azimuth_deg, self.elevation_deg, self.range_km
        )
    }
}

/// An advisory TLE checksum discrepancy.
///
/// The TLE grammar does not reject a line on a bad modulo-10 checksum, so each
/// mismatch is surfaced here (via [`Tle.checksum_warnings`]) rather than raised.
/// `line_label` is `"line 1"` or `"line 2"`; `expected` is the digit found in
/// column 69 and `computed` is the digit recomputed from columns 1-68.
#[pyclass(module = "sidereon._sidereon", name = "ChecksumWarning")]
#[derive(Clone)]
pub struct PyChecksumWarning {
    line_label: &'static str,
    expected: u8,
    computed: u8,
}

#[pymethods]
impl PyChecksumWarning {
    /// Which line the discrepancy is on: `"line 1"` or `"line 2"`.
    #[getter]
    fn line_label(&self) -> &'static str {
        self.line_label
    }

    /// The checksum digit found in column 69 of the line.
    #[getter]
    fn expected(&self) -> u8 {
        self.expected
    }

    /// The checksum digit recomputed from columns 1-68.
    #[getter]
    fn computed(&self) -> u8 {
        self.computed
    }

    fn __repr__(&self) -> String {
        format!(
            "ChecksumWarning(line_label={:?}, expected={}, computed={})",
            self.line_label, self.expected, self.computed
        )
    }

    fn __eq__(&self, other: &PyChecksumWarning) -> bool {
        self.line_label == other.line_label
            && self.expected == other.expected
            && self.computed == other.computed
    }
}

impl From<&ChecksumWarning> for PyChecksumWarning {
    fn from(w: &ChecksumWarning) -> Self {
        Self {
            line_label: w.line_label,
            expected: w.expected,
            computed: w.computed,
        }
    }
}

fn vec3_from_vec(name: &str, values: Vec<f64>) -> PyResult<[f64; 3]> {
    let array: [f64; 3] = values
        .try_into()
        .map_err(|_| PyValueError::new_err(format!("{name} must have length 3")))?;
    if !array.iter().all(|value| value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(array)
}

fn extract_loss(obj: &Bound<'_, PyAny>) -> PyResult<PyLoss> {
    if let Ok(loss) = obj.extract::<PyLoss>() {
        return Ok(loss);
    }
    PyLoss::from_label(&obj.extract::<String>()?)
}

/// Robust loss selector for TLE fitting.
#[pyclass(module = "sidereon._sidereon", name = "Loss", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyLoss {
    LINEAR,
    SOFT_L1,
    HUBER,
    CAUCHY,
    ARCTAN,
}

impl PyLoss {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "linear" => Ok(Self::LINEAR),
            "soft_l1" => Ok(Self::SOFT_L1),
            "huber" => Ok(Self::HUBER),
            "cauchy" => Ok(Self::CAUCHY),
            "arctan" => Ok(Self::ARCTAN),
            other => Err(PyValueError::new_err(format!(
                "unknown loss {other:?}; expected linear, soft_l1, huber, cauchy, or arctan"
            ))),
        }
    }
}

impl From<PyLoss> for CoreLoss {
    fn from(loss: PyLoss) -> Self {
        match loss {
            PyLoss::LINEAR => CoreLoss::Linear,
            PyLoss::SOFT_L1 => CoreLoss::SoftL1,
            PyLoss::HUBER => CoreLoss::Huber,
            PyLoss::CAUCHY => CoreLoss::Cauchy,
            PyLoss::ARCTAN => CoreLoss::Arctan,
        }
    }
}

#[pymethods]
impl PyLoss {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::LINEAR => "linear",
            Self::SOFT_L1 => "soft_l1",
            Self::HUBER => "huber",
            Self::CAUCHY => "cauchy",
            Self::ARCTAN => "arctan",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::LINEAR => "Loss.LINEAR",
            Self::SOFT_L1 => "Loss.SOFT_L1",
            Self::HUBER => "Loss.HUBER",
            Self::CAUCHY => "Loss.CAUCHY",
            Self::ARCTAN => "Loss.ARCTAN",
        }
    }
}

/// TLE fit parameter scaling selector.
#[pyclass(module = "sidereon._sidereon", name = "XScale")]
#[derive(Clone)]
pub struct PyXScale {
    inner: XScale,
}

impl PyXScale {
    fn inner(&self) -> XScale {
        self.inner.clone()
    }
}

#[pymethods]
impl PyXScale {
    #[staticmethod]
    fn unit() -> Self {
        Self {
            inner: XScale::Unit,
        }
    }

    #[staticmethod]
    fn jac() -> Self {
        Self { inner: XScale::Jac }
    }

    #[staticmethod]
    fn values(values: Vec<f64>) -> PyResult<Self> {
        if values.is_empty()
            || values
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(PyValueError::new_err(
                "x_scale values must be finite positive numbers",
            ));
        }
        Ok(Self {
            inner: XScale::Values(values),
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            XScale::Unit => "unit",
            XScale::Values(_) => "values",
            XScale::Jac => "jac",
        }
    }

    #[getter]
    fn scale_values(&self) -> Option<Vec<f64>> {
        match &self.inner {
            XScale::Values(values) => Some(values.clone()),
            XScale::Unit | XScale::Jac => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("XScale(kind={:?})", self.kind())
    }
}

/// One target TEME sample for inverse SGP4 fitting.
#[pyclass(module = "sidereon._sidereon", name = "FitSample")]
#[derive(Clone, Copy)]
pub struct PyFitSample {
    inner: FitSample,
}

#[pymethods]
impl PyFitSample {
    #[new]
    #[pyo3(signature = (
        jd_whole,
        jd_fraction,
        position_teme_km,
        velocity_teme_km_s=None,
    ))]
    fn new(
        jd_whole: f64,
        jd_fraction: f64,
        position_teme_km: Vec<f64>,
        velocity_teme_km_s: Option<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FitSample {
                epoch: Sgp4JulianDate(jd_whole, jd_fraction),
                position_teme_km: vec3_from_vec("position_teme_km", position_teme_km)?,
                velocity_teme_km_s: velocity_teme_km_s
                    .map(|value| vec3_from_vec("velocity_teme_km_s", value))
                    .transpose()?,
            },
        })
    }

    #[getter]
    fn jd_whole(&self) -> f64 {
        self.inner.epoch.0
    }

    #[getter]
    fn jd_fraction(&self) -> f64 {
        self.inner.epoch.1
    }

    #[getter]
    fn position_teme_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.position_teme_km)
    }

    #[getter]
    fn velocity_teme_km_s<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .velocity_teme_km_s
            .map(|velocity| PyArray1::from_slice(py, &velocity))
    }

    fn __repr__(&self) -> String {
        format!(
            "FitSample(jd_whole={}, jd_fraction={})",
            self.inner.epoch.0, self.inner.epoch.1
        )
    }
}

/// Selector for the fitted TLE epoch.
#[pyclass(module = "sidereon._sidereon", name = "FitEpoch")]
#[derive(Clone, Copy)]
pub struct PyFitEpoch {
    inner: FitEpoch,
}

impl PyFitEpoch {
    fn inner(&self) -> FitEpoch {
        self.inner
    }
}

#[pymethods]
impl PyFitEpoch {
    #[staticmethod]
    fn midpoint() -> Self {
        Self {
            inner: FitEpoch::Midpoint,
        }
    }

    #[staticmethod]
    fn first() -> Self {
        Self {
            inner: FitEpoch::First,
        }
    }

    #[staticmethod]
    fn last() -> Self {
        Self {
            inner: FitEpoch::Last,
        }
    }

    #[staticmethod]
    fn sample(index: usize) -> Self {
        Self {
            inner: FitEpoch::Sample(index),
        }
    }

    #[staticmethod]
    fn jd(jd_whole: f64, jd_fraction: f64) -> Self {
        Self {
            inner: FitEpoch::Jd(Sgp4JulianDate(jd_whole, jd_fraction)),
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            FitEpoch::Midpoint => "midpoint",
            FitEpoch::First => "first",
            FitEpoch::Last => "last",
            FitEpoch::Sample(_) => "sample",
            FitEpoch::Jd(_) => "jd",
        }
    }

    #[getter]
    fn sample_index(&self) -> Option<usize> {
        match self.inner {
            FitEpoch::Sample(index) => Some(index),
            _ => None,
        }
    }

    #[getter]
    fn jd_pair(&self) -> Option<(f64, f64)> {
        match self.inner {
            FitEpoch::Jd(epoch) => Some((epoch.0, epoch.1)),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("FitEpoch(kind={:?})", self.kind())
    }
}

/// Metadata carried into fitted TLE and OMM encodings.
#[pyclass(module = "sidereon._sidereon", name = "TleFitMetadata")]
#[derive(Clone)]
pub struct PyTleFitMetadata {
    inner: TleMetadata,
}

impl PyTleFitMetadata {
    fn inner(&self) -> TleMetadata {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTleFitMetadata {
    #[new]
    #[pyo3(signature = (
        catalog_number=0,
        classification="U".to_string(),
        international_designator="".to_string(),
        element_set_number=999,
        rev_at_epoch=0,
        object_name="".to_string(),
    ))]
    fn new(
        catalog_number: u32,
        classification: String,
        international_designator: String,
        element_set_number: i32,
        rev_at_epoch: i64,
        object_name: String,
    ) -> Self {
        Self {
            inner: TleMetadata {
                catalog_number,
                classification,
                international_designator,
                element_set_number,
                rev_at_epoch,
                object_name,
            },
        }
    }

    #[getter]
    fn catalog_number(&self) -> u32 {
        self.inner.catalog_number
    }

    #[getter]
    fn classification(&self) -> &str {
        &self.inner.classification
    }

    #[getter]
    fn international_designator(&self) -> &str {
        &self.inner.international_designator
    }

    #[getter]
    fn element_set_number(&self) -> i32 {
        self.inner.element_set_number
    }

    #[getter]
    fn rev_at_epoch(&self) -> i64 {
        self.inner.rev_at_epoch
    }

    #[getter]
    fn object_name(&self) -> &str {
        &self.inner.object_name
    }

    fn __repr__(&self) -> String {
        format!(
            "TleFitMetadata(catalog_number={}, object_name={:?})",
            self.inner.catalog_number, self.inner.object_name
        )
    }
}

/// Diagnostics from inverse SGP4 fitting.
#[pyclass(module = "sidereon._sidereon", name = "FitStatistics")]
#[derive(Clone)]
pub struct PyFitStatistics {
    inner: FitStatistics,
}

#[pymethods]
impl PyFitStatistics {
    #[getter]
    fn rms_position_km(&self) -> f64 {
        self.inner.rms_position_km
    }

    #[getter]
    fn max_position_km(&self) -> f64 {
        self.inner.max_position_km
    }

    #[getter]
    fn rms_position_axes_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.rms_position_axes_km)
    }

    #[getter]
    fn rms_velocity_km_s(&self) -> Option<f64> {
        self.inner.rms_velocity_km_s
    }

    #[getter]
    fn tle_rms_position_km(&self) -> f64 {
        self.inner.tle_rms_position_km
    }

    #[getter]
    fn status(&self) -> i32 {
        self.inner.status
    }

    #[getter]
    fn nfev(&self) -> usize {
        self.inner.nfev
    }

    #[getter]
    fn njev(&self) -> usize {
        self.inner.njev
    }

    #[getter]
    fn cost(&self) -> f64 {
        self.inner.cost
    }

    #[getter]
    fn optimality(&self) -> f64 {
        self.inner.optimality
    }

    #[getter]
    fn bstar_observable(&self) -> bool {
        self.inner.bstar_observable
    }

    #[getter]
    fn seed_refine_passes(&self) -> usize {
        self.inner.seed_refine_passes
    }

    fn __repr__(&self) -> String {
        format!(
            "FitStatistics(rms_position_km={}, bstar_observable={})",
            self.inner.rms_position_km, self.inner.bstar_observable
        )
    }
}

/// Fitted SGP4 element set plus TLE and OMM encodings.
#[pyclass(module = "sidereon._sidereon", name = "TleFit")]
#[derive(Clone)]
pub struct PyTleFit {
    inner: TleFit,
}

impl PyTleFit {
    fn elements(&self) -> &ElementSet {
        &self.inner.elements
    }
}

#[pymethods]
impl PyTleFit {
    #[getter]
    fn line1(&self) -> &str {
        &self.inner.line1
    }

    #[getter]
    fn line2(&self) -> &str {
        &self.inner.line2
    }

    #[getter]
    fn omm(&self) -> PyOmm {
        PyOmm::from_inner(self.inner.omm.clone())
    }

    #[getter]
    fn stats(&self) -> PyFitStatistics {
        PyFitStatistics {
            inner: self.inner.stats.clone(),
        }
    }

    #[getter]
    fn jd_whole(&self) -> f64 {
        self.elements().epoch.0
    }

    #[getter]
    fn jd_fraction(&self) -> f64 {
        self.elements().epoch.1
    }

    #[getter]
    fn bstar(&self) -> f64 {
        self.elements().bstar
    }

    #[getter]
    fn mean_motion_dot(&self) -> f64 {
        self.elements().mean_motion_dot
    }

    #[getter]
    fn mean_motion_double_dot(&self) -> f64 {
        self.elements().mean_motion_double_dot
    }

    #[getter]
    fn eccentricity(&self) -> f64 {
        self.elements().eccentricity
    }

    #[getter]
    fn argument_of_perigee_deg(&self) -> f64 {
        self.elements().argument_of_perigee_deg
    }

    #[getter]
    fn inclination_deg(&self) -> f64 {
        self.elements().inclination_deg
    }

    #[getter]
    fn mean_anomaly_deg(&self) -> f64 {
        self.elements().mean_anomaly_deg
    }

    #[getter]
    fn mean_motion_rev_per_day(&self) -> f64 {
        self.elements().mean_motion_rev_per_day
    }

    #[getter]
    fn right_ascension_deg(&self) -> f64 {
        self.elements().right_ascension_deg
    }

    #[getter]
    fn catalog_number(&self) -> u32 {
        self.elements().catalog_number
    }

    fn to_lines(&self) -> (String, String) {
        (self.inner.line1.clone(), self.inner.line2.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "TleFit(catalog_number={}, rms_position_km={})",
            self.elements().catalog_number,
            self.inner.stats.rms_position_km
        )
    }
}

/// Fit a TLE to a TEME state arc.
#[pyfunction]
#[pyo3(signature = (
    samples,
    *,
    epoch = None,
    fit_bstar = true,
    bstar_seed = 0.0,
    use_velocity = true,
    velocity_weight_s = None,
    weights = None,
    opsmode = PyOpsMode::IMPROVED,
    ftol = None,
    xtol = None,
    gtol = None,
    max_nfev = None,
    x_scale = None,
    loss = PyLoss::LINEAR,
    f_scale = 1.0,
    metadata = None,
))]
#[allow(clippy::too_many_arguments)]
fn fit_tle(
    py: Python<'_>,
    samples: Vec<Py<PyFitSample>>,
    epoch: Option<Py<PyFitEpoch>>,
    fit_bstar: bool,
    bstar_seed: f64,
    use_velocity: bool,
    velocity_weight_s: Option<f64>,
    weights: Option<Vec<f64>>,
    #[pyo3(from_py_with = extract_opsmode)] opsmode: PyOpsMode,
    ftol: Option<f64>,
    xtol: Option<f64>,
    gtol: Option<f64>,
    max_nfev: Option<usize>,
    x_scale: Option<Py<PyXScale>>,
    #[pyo3(from_py_with = extract_loss)] loss: PyLoss,
    f_scale: f64,
    metadata: Option<Py<PyTleFitMetadata>>,
) -> PyResult<PyTleFit> {
    let samples: Vec<FitSample> = samples.iter().map(|s| s.borrow(py).inner).collect();
    let config = FitConfig {
        epoch: epoch
            .as_ref()
            .map(|value| value.borrow(py).inner())
            .unwrap_or_default(),
        fit_bstar,
        bstar_seed,
        use_velocity,
        velocity_weight_s,
        weights,
        opsmode: opsmode.into(),
        ftol,
        xtol,
        gtol,
        max_nfev,
        x_scale: x_scale.as_ref().map(|value| value.borrow(py).inner()),
        loss: loss.into(),
        f_scale,
        metadata: metadata
            .as_ref()
            .map(|value| value.borrow(py).inner())
            .unwrap_or_default(),
    };
    let inner = py
        .allow_threads(move || core_fit_tle(&samples, &config))
        .map_err(|err| SolveError::new_err(err.to_string()))?;
    Ok(PyTleFit { inner })
}

/// A parsed two-line element set, ready to propagate.
///
/// Construct from the two TLE lines. `opsmode` selects the SGP4 operation mode:
/// `OpsMode.AFSPC` (default, matching the engine's topocentric goldens) or
/// `OpsMode.IMPROVED`. Element fields are exposed as read-only properties; the
/// `propagate` and `look_angles` methods take a 1-D numpy `int64` array of unix
/// microseconds (e.g. `np.asarray(times, "datetime64[us]").astype("int64")`).
#[pyclass(module = "sidereon._sidereon", name = "Tle")]
pub struct PyTle {
    elements: TleElements,
    satellite: Satellite,
    checksum_warnings: Vec<ChecksumWarning>,
}

impl PyTle {
    /// The initialized SGP4 satellite this handle wraps, for sibling binding
    /// modules (for example coverage-grid batching).
    pub(crate) fn satellite(&self) -> &Satellite {
        &self.satellite
    }
}

#[pymethods]
impl PyTle {
    #[new]
    #[pyo3(signature = (line1, line2, opsmode=PyOpsMode::AFSPC))]
    fn new(
        line1: &str,
        line2: &str,
        #[pyo3(from_py_with = extract_opsmode)] opsmode: PyOpsMode,
    ) -> PyResult<Self> {
        let mode = CoreOpsMode::from(opsmode);
        let parsed = parse_tle_lines(line1, line2).map_err(to_tle_err)?;
        let satellite = Satellite::from_tle_with_opsmode(line1, line2, mode).map_err(to_tle_err)?;
        Ok(Self {
            elements: parsed.elements,
            satellite,
            checksum_warnings: parsed.checksum_warnings,
        })
    }

    /// Re-encode the parsed elements as the two 69-character TLE lines (with
    /// checksums), via the engine's `tle::encode`. For a well-formed input the
    /// round-trip is character-exact.
    fn to_lines(&self) -> (String, String) {
        encode_tle_lines(&self.elements)
    }

    /// Advisory checksum discrepancies found while parsing, as a list of
    /// [`ChecksumWarning`]. Empty when both lines' checksums are valid.
    #[getter]
    fn checksum_warnings(&self) -> Vec<PyChecksumWarning> {
        self.checksum_warnings
            .iter()
            .map(PyChecksumWarning::from)
            .collect()
    }

    /// Propagate over a 1-D numpy `int64` array of unix-microsecond epochs.
    ///
    /// Returns a [`TlePropagation`] whose `position_km` / `velocity_km_s` are
    /// `(n_epochs, 3)` numpy arrays in the TEME frame.
    fn propagate(&self, epochs_unix_us: PyReadonlyArray1<'_, i64>) -> PyResult<PyTlePropagation> {
        let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
        let predictions = propagate_teme_arc(&self.satellite, &instants).map_err(to_solve_err)?;
        Ok(PyTlePropagation {
            positions_km: predictions.iter().map(|p| p.position).collect(),
            velocities_km_s: predictions.iter().map(|p| p.velocity).collect(),
        })
    }

    /// Topocentric az/el/range from `station` over a 1-D numpy `int64` array of
    /// unix-microsecond epochs.
    fn look_angles(
        &self,
        station: PyGroundStation,
        epochs_unix_us: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<PyLookAngles> {
        let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
        let looks =
            look_angle_arc(&self.satellite, station.inner, &instants).map_err(to_solve_err)?;
        Ok(PyLookAngles {
            azimuth_deg: looks.iter().map(|l| l.azimuth_deg).collect(),
            elevation_deg: looks.iter().map(|l| l.elevation_deg).collect(),
            range_km: looks.iter().map(|l| l.range_km).collect(),
        })
    }

    /// Topocentric visibility arrays and dense pass list over an epoch grid.
    ///
    /// `epochs_unix_us` must be a strictly increasing 1-D numpy `int64` array of
    /// UTC unix microseconds with at least two samples. The azimuth/elevation
    /// geometry is evaluated exactly through the same core path as
    /// [`Tle.look_angles`]. The pass list is computed by the core dense
    /// pass-finder over the window from the first epoch to the last epoch.
    #[pyo3(signature = (
        station,
        epochs_unix_us,
        *,
        elevation_mask_deg = 0.0,
        step_seconds = 30.0,
        time_tolerance_s = 1.0e-3,
    ))]
    fn visibility_series(
        &self,
        station: PyGroundStation,
        epochs_unix_us: PyReadonlyArray1<'_, i64>,
        elevation_mask_deg: f64,
        step_seconds: f64,
        time_tolerance_s: f64,
    ) -> PyResult<PyVisibilitySeries> {
        if !elevation_mask_deg.is_finite() {
            return Err(PyValueError::new_err("elevation_mask_deg must be finite"));
        }
        if !step_seconds.is_finite() || step_seconds <= 0.0 {
            return Err(PyValueError::new_err("step_seconds must be positive"));
        }
        if !time_tolerance_s.is_finite() || time_tolerance_s <= 0.0 {
            return Err(PyValueError::new_err("time_tolerance_s must be positive"));
        }

        let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Reject)?;
        if instants.len() < 2 {
            return Err(PyValueError::new_err(
                "epochs_unix_us must contain at least two samples",
            ));
        }
        if instants.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(PyValueError::new_err(
                "epochs_unix_us must be strictly increasing",
            ));
        }

        let ground_station = station.inner;
        let looks =
            look_angle_arc(&self.satellite, ground_station, &instants).map_err(to_solve_err)?;
        let passes = find_passes_for_satellite(
            &self.satellite,
            ground_station,
            instants[0],
            *instants.last().expect("non-empty instants checked"),
            PassFinderOptions {
                elevation_mask_deg,
                coarse_step_seconds: step_seconds,
                time_tolerance_seconds: time_tolerance_s,
            },
        )
        .map_err(to_solve_err)?;

        Ok(PyVisibilitySeries {
            epochs_unix_us: instants
                .iter()
                .map(|instant| instant.unix_microseconds())
                .collect(),
            azimuth_deg: looks.iter().map(|l| l.azimuth_deg).collect(),
            elevation_deg: looks.iter().map(|l| l.elevation_deg).collect(),
            range_km: looks.iter().map(|l| l.range_km).collect(),
            visible: looks
                .iter()
                .map(|l| visible_at_elevation_mask(l.elevation_deg, elevation_mask_deg))
                .collect(),
            passes: passes.iter().map(to_py_pass).collect(),
        })
    }

    /// Find passes over `station` within `[start_unix_us, end_unix_us)` by dense
    /// elevation sampling.
    ///
    /// `start_unix_us` / `end_unix_us` are UTC unix microseconds. The elevation
    /// is sampled every `step_seconds`, sign changes of `elevation -
    /// elevation_mask_deg` are bracketed, and each AOS/LOS crossing plus the
    /// culmination (the elevation-rate zero) is refined to `time_tolerance_s`.
    /// Returns a list of [`SatellitePass`]. Raises `ValueError` on a
    /// non-positive step or an end at or before the start.
    #[pyo3(signature = (
        station,
        start_unix_us,
        end_unix_us,
        *,
        elevation_mask_deg = 0.0,
        step_seconds = 30.0,
        time_tolerance_s = 1.0e-3,
    ))]
    fn find_passes(
        &self,
        station: PyGroundStation,
        start_unix_us: i64,
        end_unix_us: i64,
        elevation_mask_deg: f64,
        step_seconds: f64,
        time_tolerance_s: f64,
    ) -> PyResult<Vec<PySatellitePass>> {
        if end_unix_us <= start_unix_us {
            return Err(PyValueError::new_err(
                "end_unix_us must be after start_unix_us",
            ));
        }
        if step_seconds <= 0.0 {
            return Err(PyValueError::new_err("step_seconds must be positive"));
        }

        let passes = find_passes_for_satellite(
            &self.satellite,
            station.inner,
            UtcInstant::from_unix_microseconds(start_unix_us),
            UtcInstant::from_unix_microseconds(end_unix_us),
            PassFinderOptions {
                elevation_mask_deg,
                coarse_step_seconds: step_seconds,
                time_tolerance_seconds: time_tolerance_s,
            },
        )
        .map_err(to_solve_err)?;
        Ok(passes.iter().map(to_py_pass).collect())
    }

    /// Sub-satellite (ground-track) WGS84 geodetic points over a 1-D numpy
    /// `int64` array of unix-microsecond epochs.
    ///
    /// Returns a list of [`Wgs84Geodetic`] (geodetic latitude/longitude in
    /// radians, ellipsoidal height in metres), one per epoch, computed by the
    /// core `passes::ground_track`: propagate to TEME, TEME->GCRS->ITRS, then the
    /// shared ECEF->geodetic reduction. Same epoch convention as
    /// [`Tle.propagate`] / [`Tle.look_angles`]; the first propagation or frame
    /// error aborts the whole arc.
    fn ground_track(
        &self,
        epochs_unix_us: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Vec<PyWgs84Geodetic>> {
        let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
        let track = core_ground_track(&self.satellite, &instants).map_err(to_solve_err)?;
        Ok(track.into_iter().map(PyWgs84Geodetic::from_core).collect())
    }

    /// NORAD catalog number (as recorded in the TLE).
    #[getter]
    fn catalog_number(&self) -> String {
        self.elements.catalog_number.clone()
    }

    /// Classification character (`U`/`C`/`S`).
    #[getter]
    fn classification(&self) -> String {
        self.elements.classification.clone()
    }

    /// International designator (COSPAR ID).
    #[getter]
    fn international_designator(&self) -> String {
        self.elements.international_designator.clone()
    }

    /// Four-digit epoch year.
    #[getter]
    fn epoch_year(&self) -> i32 {
        self.elements.epoch_year
    }

    /// Fractional day-of-year of the epoch.
    #[getter]
    fn epoch_day_of_year(&self) -> f64 {
        self.elements.epoch_day_of_year
    }

    /// Inclination in degrees.
    #[getter]
    fn inclination_deg(&self) -> f64 {
        self.elements.inclination_deg
    }

    /// Right ascension of the ascending node, degrees.
    #[getter]
    fn raan_deg(&self) -> f64 {
        self.elements.raan_deg
    }

    /// Orbital eccentricity (dimensionless).
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.elements.eccentricity
    }

    /// Argument of perigee, degrees.
    #[getter]
    fn arg_perigee_deg(&self) -> f64 {
        self.elements.arg_perigee_deg
    }

    /// Mean anomaly at epoch, degrees.
    #[getter]
    fn mean_anomaly_deg(&self) -> f64 {
        self.elements.mean_anomaly_deg
    }

    /// Mean motion, revolutions per day.
    #[getter]
    fn mean_motion_rev_per_day(&self) -> f64 {
        self.elements.mean_motion
    }

    /// First derivative of mean motion (rev/day^2).
    #[getter]
    fn mean_motion_dot(&self) -> f64 {
        self.elements.mean_motion_dot
    }

    /// Second derivative of mean motion (rev/day^3).
    #[getter]
    fn mean_motion_double_dot(&self) -> f64 {
        self.elements.mean_motion_double_dot
    }

    /// B* drag term (TLE dimensionless convention).
    #[getter]
    fn bstar(&self) -> f64 {
        self.elements.bstar
    }

    /// Revolution number at epoch.
    #[getter]
    fn rev_number(&self) -> i32 {
        self.elements.rev_number
    }

    fn __repr__(&self) -> String {
        format!(
            "Tle(catalog_number={:?}, epoch_year={}, epoch_day_of_year={:.8}, inclination_deg={:.4})",
            self.elements.catalog_number,
            self.elements.epoch_year,
            self.elements.epoch_day_of_year,
            self.elements.inclination_deg
        )
    }
}

/// A named satellite from a parsed TLE file: a [`Tle`] paired with its name
/// line.
#[pyclass(module = "sidereon._sidereon", name = "NamedTle")]
pub struct PyNamedTle {
    name: String,
    tle: Py<PyTle>,
}

#[pymethods]
impl PyNamedTle {
    /// The name line preceding the element set (a leading CelesTrak `0 ` marker
    /// stripped). Empty when the record was a bare 2-line set with no name.
    #[getter]
    fn name(&self) -> String {
        self.name.clone()
    }

    /// The parsed satellite, ready to `propagate` / `look_angles`.
    #[getter]
    fn tle(&self, py: Python<'_>) -> Py<PyTle> {
        self.tle.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let catalog = self.tle.borrow(py).elements.catalog_number.clone();
        format!(
            "NamedTle(name={:?}, catalog_number={:?})",
            self.name, catalog
        )
    }
}

/// The result of parsing a multi-record TLE file: the satellites that parsed,
/// plus a count of records that were skipped because their element set failed
/// SGP4 initialization.
#[pyclass(module = "sidereon._sidereon", name = "TleFile")]
pub struct PyTleFile {
    satellites: Vec<Py<PyNamedTle>>,
    skipped: usize,
}

#[pymethods]
impl PyTleFile {
    /// The successfully parsed satellites, in file order, as [`NamedTle`].
    #[getter]
    fn satellites(&self, py: Python<'_>) -> Vec<Py<PyNamedTle>> {
        self.satellites.iter().map(|s| s.clone_ref(py)).collect()
    }

    /// How many complete `(line 1, line 2)` records were found but skipped
    /// because their element set failed SGP4 initialization.
    #[getter]
    fn skipped(&self) -> usize {
        self.skipped
    }

    /// Number of satellites that parsed successfully.
    fn __len__(&self) -> usize {
        self.satellites.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "TleFile(satellites={}, skipped={})",
            self.satellites.len(),
            self.skipped
        )
    }
}

/// Parse a multi-record TLE file (CelesTrak / Space-Track style) into satellites
/// paired with their names.
///
/// Handles, in a single pass, bare 2-line element sets, 3-line sets (a name line
/// followed by lines 1 and 2), and CelesTrak `0 NAME` name lines. Blank lines,
/// CRLF endings, and surrounding whitespace are tolerated. A record whose element
/// set fails SGP4 initialization is skipped and counted in `TleFile.skipped`
/// rather than aborting the whole file.
///
/// `opsmode` selects the SGP4 operation mode (default `OpsMode.AFSPC`); each
/// returned [`Tle`] is initialized with it.
#[pyfunction]
#[pyo3(signature = (text, *, opsmode=PyOpsMode::AFSPC))]
fn parse_tle_file(
    py: Python<'_>,
    text: &str,
    #[pyo3(from_py_with = extract_opsmode)] opsmode: PyOpsMode,
) -> PyResult<PyTleFile> {
    let mode = CoreOpsMode::from(opsmode);
    let parsed = parse_tle_file_with_opsmode(text, mode);
    let mut satellites = Vec::with_capacity(parsed.satellites.len());
    for named in parsed.satellites {
        // The core already initialized this satellite with `mode`; wrap it
        // directly (no re-init) and recover the element fields / checksum
        // advisories by re-parsing its source lines.
        let elements = parse_tle_lines(named.satellite.line1(), named.satellite.line2())
            .map_err(to_tle_err)?;
        let tle = Py::new(
            py,
            PyTle {
                elements: elements.elements,
                satellite: named.satellite,
                checksum_warnings: elements.checksum_warnings,
            },
        )?;
        satellites.push(Py::new(
            py,
            PyNamedTle {
                name: named.name,
                tle,
            },
        )?);
    }
    Ok(PyTleFile {
        satellites,
        skipped: parsed.skipped,
    })
}

/// An ephemeris from numerical state-vector propagation: the requested output
/// epochs plus the Cartesian state at each, as numpy `float64` arrays.
#[pyclass(module = "sidereon._sidereon", name = "Ephemeris")]
pub struct PyEphemeris {
    times_s: Vec<f64>,
    positions_km: Vec<[f64; 3]>,
    velocities_km_s: Vec<[f64; 3]>,
}

#[pymethods]
impl PyEphemeris {
    /// The output epochs (TDB seconds), as a numpy array of shape `(n,)`.
    #[getter]
    fn times_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.times_s)
    }

    /// ECI positions as a numpy array of shape `(n, 3)`, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.positions_km)
    }

    /// ECI velocities as a numpy array of shape `(n, 3)`, km/s.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.velocities_km_s)
    }

    /// The full state ephemeris as a single `(n, 6)` numpy array whose columns
    /// are `[x, y, z, vx, vy, vz]` (km, km/s) -- times by states.
    #[getter]
    fn states<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: Vec<[f64; 6]> = self
            .positions_km
            .iter()
            .zip(self.velocities_km_s.iter())
            .map(|(p, v)| [p[0], p[1], p[2], v[0], v[1], v[2]])
            .collect();
        rows6_to_array(py, &rows)
    }

    /// Number of output epochs.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.times_s.len()
    }

    fn __len__(&self) -> usize {
        self.times_s.len()
    }

    fn __repr__(&self) -> String {
        format!("Ephemeris(epoch_count={})", self.times_s.len())
    }
}

/// Numerically propagate an ECI Cartesian state and sample it at a grid of
/// epochs.
///
/// `position_km` and `velocity_km_s` are length-3 numpy arrays; `times_s` is a
/// 1-D numpy `float64` array of absolute TDB epochs (seconds) at which to sample
/// the trajectory, monotonic in the propagation direction. The state is
/// integrated with the chosen `force_model` (`ForceModel.TWO_BODY` or
/// `ForceModel.TWO_BODY_J2`) and `integrator` (`Integrator.DP54` adaptive,
/// default, or `Integrator.RK4` fixed-step). The
/// tolerance / step keywords are forwarded to the integrator. Bad input
/// (wrong shape, unknown selector, non-positive step) raises
/// `ValueError`; a propagation failure raises `SidereonError`.
#[pyfunction]
#[pyo3(signature = (
    epoch_s,
    position_km,
    velocity_km_s,
    times_s,
    *,
    force_model = PyForceModel::TWO_BODY,
    integrator = PyIntegrator::DP54,
    abs_tol = 1.0e-9,
    rel_tol = 1.0e-12,
    initial_step_s = 60.0,
    min_step_s = 1.0e-6,
    max_step_s = 3600.0,
    max_steps = 1_000_000,
    mu_km3_s2 = None,
    drag = None,
    space_weather_table = None,
))]
#[allow(clippy::too_many_arguments)]
fn propagate_state(
    py: Python<'_>,
    epoch_s: f64,
    position_km: PyReadonlyArray1<'_, f64>,
    velocity_km_s: PyReadonlyArray1<'_, f64>,
    times_s: PyReadonlyArray1<'_, f64>,
    #[pyo3(from_py_with = extract_force_model)] force_model: PyForceModel,
    #[pyo3(from_py_with = extract_integrator)] integrator: PyIntegrator,
    abs_tol: f64,
    rel_tol: f64,
    initial_step_s: f64,
    min_step_s: f64,
    max_step_s: f64,
    max_steps: u32,
    mu_km3_s2: Option<f64>,
    drag: Option<Py<PyDragParameters>>,
    space_weather_table: Option<Py<PySpaceWeatherTable>>,
) -> PyResult<PyEphemeris> {
    let position = fixed_array::<3>("position_km", &position_km, FinitePolicy::AllowNonFinite)?;
    let velocity = fixed_array::<3>(
        "velocity_km_s",
        &velocity_km_s,
        FinitePolicy::AllowNonFinite,
    )?;
    let times = times_s
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let times_vec = times.to_vec();
    if initial_step_s <= 0.0 {
        return Err(PyValueError::new_err("initial_step_s must be positive"));
    }
    let drag = drag.map(|value| value.borrow(py).inner());

    let config = PropagationConfig {
        force_model: force_model.to_core(),
        mu_km3_s2,
        integrator: IntegratorKind::from(integrator),
        options: IntegratorOptions {
            abs_tol,
            rel_tol,
            initial_step: initial_step_s,
            min_step: min_step_s,
            max_step: max_step_s,
            max_steps,
            dense_output: false,
        },
        drag,
        ..PropagationConfig::new(epoch_s, position, velocity)
    };

    let output_times = times_vec.clone();
    let states = if let Some(table) = space_weather_table {
        let source = table.borrow(py).source();
        let propagator = config.to_propagator().with_space_weather(source);
        py.allow_threads(move || propagator.ephemeris(&times_vec))
            .map_err(to_solve_err)?
    } else {
        py.allow_threads(move || propagate_states(&config, &times_vec))
            .map_err(to_solve_err)?
    };
    Ok(PyEphemeris {
        times_s: output_times,
        positions_km: states.iter().map(|s| s.position_array()).collect(),
        velocities_km_s: states.iter().map(|s| s.velocity_array()).collect(),
    })
}

/// TEME states from a batched multi-satellite propagation: `position_km` and
/// `velocity_km_s` as numpy `float64` arrays of shape `(n_satellites, n_epochs,
/// 3)`. Row `i` is satellite `i`'s arc over the shared epoch grid, in the same
/// order as the input TLEs.
#[pyclass(module = "sidereon._sidereon", name = "BatchPropagation")]
pub struct PyBatchPropagation {
    epoch_count: usize,
    positions_km: Vec<Vec<[f64; 3]>>,
    velocities_km_s: Vec<Vec<[f64; 3]>>,
}

#[pymethods]
impl PyBatchPropagation {
    /// TEME positions as a numpy array of shape `(n_satellites, n_epochs, 3)`,
    /// kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        vec3_rows_to_array3(py, &self.positions_km, self.epoch_count)
    }

    /// TEME velocities as a numpy array of shape `(n_satellites, n_epochs, 3)`,
    /// km/s.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        vec3_rows_to_array3(py, &self.velocities_km_s, self.epoch_count)
    }

    /// Number of satellites in the batch.
    #[getter]
    fn satellite_count(&self) -> usize {
        self.positions_km.len()
    }

    /// Number of epochs each satellite was propagated over.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.epoch_count
    }

    fn __len__(&self) -> usize {
        self.positions_km.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchPropagation(satellite_count={}, epoch_count={})",
            self.positions_km.len(),
            self.epoch_count
        )
    }
}

/// Topocentric look angles for a batch of satellites over a shared epoch grid:
/// `azimuth_deg`, `elevation_deg`, and `range_km`, each a numpy `float64` array
/// of shape `(n_satellites, n_epochs)`. Row `i` corresponds to input TLE `i`.
#[pyclass(module = "sidereon._sidereon", name = "BatchLookAngles")]
pub struct PyBatchLookAngles {
    epoch_count: usize,
    azimuth_deg: Vec<Vec<f64>>,
    elevation_deg: Vec<Vec<f64>>,
    range_km: Vec<Vec<f64>>,
}

#[pymethods]
impl PyBatchLookAngles {
    /// Azimuth in degrees, clockwise from north, shape `(n_satellites, n_epochs)`.
    #[getter]
    fn azimuth_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        scalar_rows_to_array2(py, &self.azimuth_deg, self.epoch_count)
    }

    /// Elevation in degrees above the horizon, shape `(n_satellites, n_epochs)`.
    #[getter]
    fn elevation_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        scalar_rows_to_array2(py, &self.elevation_deg, self.epoch_count)
    }

    /// Slant range in kilometres, shape `(n_satellites, n_epochs)`.
    #[getter]
    fn range_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        scalar_rows_to_array2(py, &self.range_km, self.epoch_count)
    }

    /// Number of satellites in the batch.
    #[getter]
    fn satellite_count(&self) -> usize {
        self.azimuth_deg.len()
    }

    /// Number of epochs each satellite was evaluated over.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.epoch_count
    }

    fn __len__(&self) -> usize {
        self.azimuth_deg.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchLookAngles(satellite_count={}, epoch_count={})",
            self.azimuth_deg.len(),
            self.epoch_count
        )
    }
}

/// Build a satellite per `(line1, line2)` pair, parsed once with the GIL held so
/// the rayon compute window touches no Python objects. A parse/init failure
/// names the offending index. Empty fleets are valid and return empty batches.
fn build_satellites(tles: &[(String, String)], mode: CoreOpsMode) -> PyResult<Vec<Satellite>> {
    tles.iter()
        .enumerate()
        .map(|(idx, (line1, line2))| {
            Satellite::from_tle_with_opsmode(line1, line2, mode)
                .map_err(|e| TleParseError::new_err(format!("satellite {idx}: {e}")))
        })
        .collect()
}

/// Collapse a per-satellite batch result into one arc per satellite, surfacing
/// the first failing satellite's index and engine message as a `SidereonError`.
fn unwrap_batch<T, E: std::fmt::Display>(results: Vec<Result<Vec<T>, E>>) -> PyResult<Vec<Vec<T>>> {
    results
        .into_iter()
        .enumerate()
        .map(|(idx, arc)| arc.map_err(|e| SolveError::new_err(format!("satellite {idx}: {e}"))))
        .collect()
}

/// Propagate a fleet of TLEs over a shared epoch grid, releasing the GIL for the
/// whole compute.
///
/// `tles` is a sequence of `(line1, line2)` string pairs; `epochs_unix_us` is a
/// 1-D numpy `int64` array of unix-microsecond UTC epochs shared by every
/// satellite. The TLEs are parsed with the GIL held, then the per-satellite SGP4
/// arcs run inside `Python::allow_threads` -- by default across a rayon thread
/// pool (`parallel=True`), so the fleet saturates all cores with no interpreter
/// lock held. Each satellite's arc is computed by the same serial kernel and
/// rayon's indexed collect preserves order, so the result is bit-identical to
/// the serial path (`parallel=False`) element by element. Returns a
/// [`BatchPropagation`] with `(n_satellites, n_epochs, 3)` TEME state arrays.
/// Empty fleets and empty epoch grids return empty arrays. Raises
/// `SidereonError` (naming the index) if a satellite fails to parse or
/// propagate.
#[pyfunction]
#[pyo3(signature = (tles, epochs_unix_us, *, opsmode=PyOpsMode::AFSPC, parallel=true))]
fn propagate_batch(
    py: Python<'_>,
    tles: Vec<(String, String)>,
    epochs_unix_us: PyReadonlyArray1<'_, i64>,
    #[pyo3(from_py_with = extract_opsmode)] opsmode: PyOpsMode,
    parallel: bool,
) -> PyResult<PyBatchPropagation> {
    let mode = CoreOpsMode::from(opsmode);
    let satellites = build_satellites(&tles, mode)?;
    let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
    let epoch_count = instants.len();

    // GIL released for the entire propagation: the closure owns plain Rust data
    // (satellites + instants) and touches no Python object.
    let results = py.allow_threads(move || {
        if parallel {
            propagate_teme_batch_parallel(&satellites, &instants)
        } else {
            propagate_teme_batch_serial(&satellites, &instants)
        }
    });

    let arcs = unwrap_batch(results)?;
    Ok(PyBatchPropagation {
        epoch_count,
        positions_km: arcs
            .iter()
            .map(|arc| arc.iter().map(|p| p.position).collect())
            .collect(),
        velocities_km_s: arcs
            .iter()
            .map(|arc| arc.iter().map(|p| p.velocity).collect())
            .collect(),
    })
}

/// Topocentric look angles for a fleet of TLEs from one ground station over a
/// shared epoch grid, releasing the GIL for the whole compute.
///
/// Same input/threading contract as [`propagate_batch`]: `tles` parsed with the
/// GIL held, then the rayon look-angle batch runs inside `Python::allow_threads`
/// (`parallel=True` by default), bit-identical to the serial path. Returns a
/// [`BatchLookAngles`] with `(n_satellites, n_epochs)` az/el/range arrays.
#[pyfunction]
#[pyo3(signature = (tles, station, epochs_unix_us, *, opsmode=PyOpsMode::AFSPC, parallel=true))]
fn look_angles_batch(
    py: Python<'_>,
    tles: Vec<(String, String)>,
    station: PyGroundStation,
    epochs_unix_us: PyReadonlyArray1<'_, i64>,
    #[pyo3(from_py_with = extract_opsmode)] opsmode: PyOpsMode,
    parallel: bool,
) -> PyResult<PyBatchLookAngles> {
    let mode = CoreOpsMode::from(opsmode);
    let satellites = build_satellites(&tles, mode)?;
    let ground_station = station.inner;
    let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
    let epoch_count = instants.len();

    let results = py.allow_threads(move || {
        if parallel {
            look_angle_batch_parallel(&satellites, ground_station, &instants)
        } else {
            look_angle_batch_serial(&satellites, ground_station, &instants)
        }
    });

    let arcs = unwrap_batch(results)?;
    Ok(PyBatchLookAngles {
        epoch_count,
        azimuth_deg: arcs
            .iter()
            .map(|arc| arc.iter().map(|l| l.azimuth_deg).collect())
            .collect(),
        elevation_deg: arcs
            .iter()
            .map(|arc| arc.iter().map(|l| l.elevation_deg).collect())
            .collect(),
        range_km: arcs
            .iter()
            .map(|arc| arc.iter().map(|l| l.range_km).collect())
            .collect(),
    })
}

/// Satellites above a ground station's horizon at one instant, honoring each
/// satellite's own SGP4 opsmode.
///
/// `satellites` is a list of already-parsed [`Tle`] objects, each carrying the
/// opsmode it was constructed with; `ids` is a parallel list of identities (one
/// per satellite, same order) that become each result's `catalog_number`. Unlike
/// rebuilding from raw element sets with a hardcoded AFSPC opsmode, this steps
/// each satellite exactly as it was initialized, so an `OpsMode.IMPROVED`
/// satellite is propagated in improved mode end-to-end.
///
/// `station` is the observer; `epoch_unix_us` is the UTC unix-microsecond instant
/// (matching the epoch convention used elsewhere). Results are filtered to
/// `elevation_deg >= min_elevation_deg` and sorted by elevation descending.
/// Per-satellite propagation or frame failures are skipped. Raises `SidereonError`
/// on an invalid station, a non-finite threshold, or an `ids`/`satellites`
/// length mismatch.
#[pyfunction]
#[pyo3(signature = (satellites, ids, station, epoch_unix_us, *, min_elevation_deg=0.0))]
fn visible_from_satellites(
    py: Python<'_>,
    satellites: Vec<Py<PyTle>>,
    ids: Vec<String>,
    station: PyGroundStation,
    epoch_unix_us: i64,
    min_elevation_deg: f64,
) -> PyResult<Vec<PyVisibleSatellite>> {
    let sats: Vec<Satellite> = satellites
        .iter()
        .map(|t| t.borrow(py).satellite.clone())
        .collect();
    let datetime = UtcInstant::from_unix_microseconds(epoch_unix_us);
    let visible =
        core_visible_from_satellites(&sats, &ids, station.inner, datetime, min_elevation_deg)
            .map_err(to_solve_err)?;
    Ok(visible
        .into_iter()
        .map(PyVisibleSatellite::from_core)
        .collect())
}

/// Sub-satellite WGS84 ground track for one satellite over an epoch grid:
/// `latitude_deg`, `longitude_deg`, and `altitude_km`, each a numpy `float64`
/// array of shape `(n_epochs,)` aligned to the input grid. Latitude/longitude
/// are geodetic degrees (longitude in `[-180, 180]`) and altitude is the
/// ellipsoidal height above the WGS84 ellipsoid in kilometres. The per-epoch
/// geometry is the same core reduction as [`Tle.ground_track`].
#[pyclass(module = "sidereon._sidereon", name = "GroundTrack")]
pub struct PyGroundTrack {
    latitude_deg: Vec<f64>,
    longitude_deg: Vec<f64>,
    altitude_km: Vec<f64>,
}

#[pymethods]
impl PyGroundTrack {
    /// Geodetic latitude of the sub-satellite point, degrees north, `(n_epochs,)`.
    #[getter]
    fn latitude_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.latitude_deg)
    }

    /// Geodetic longitude of the sub-satellite point, degrees east in
    /// `[-180, 180]`, `(n_epochs,)`.
    #[getter]
    fn longitude_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.longitude_deg)
    }

    /// Ellipsoidal height above the WGS84 ellipsoid, kilometres, `(n_epochs,)`.
    #[getter]
    fn altitude_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.altitude_km)
    }

    /// Number of epochs evaluated.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.latitude_deg.len()
    }

    fn __len__(&self) -> usize {
        self.latitude_deg.len()
    }

    fn __repr__(&self) -> String {
        format!("GroundTrack(epoch_count={})", self.latitude_deg.len())
    }
}

/// One pass in a [`Constellation.passes`] result: the pass geometry plus the
/// fleet-order `satellite_index` of the satellite it belongs to. Map that index
/// to your own per-satellite metadata (or to [`Constellation.catalog_numbers`]).
/// Times are unix microseconds (UTC), matching every other epoch in the binding.
#[pyclass(module = "sidereon._sidereon", name = "FleetPass")]
#[derive(Clone, Copy)]
pub struct PyFleetPass {
    satellite_index: usize,
    pass: PySatellitePass,
}

#[pymethods]
impl PyFleetPass {
    /// Fleet-order index of the satellite this pass belongs to.
    #[getter]
    fn satellite_index(&self) -> usize {
        self.satellite_index
    }

    /// Acquisition of signal (rise above the mask), unix microseconds UTC.
    #[getter]
    fn aos_unix_us(&self) -> i64 {
        self.pass.aos_unix_us
    }

    /// Loss of signal (set below the mask), unix microseconds UTC.
    #[getter]
    fn los_unix_us(&self) -> i64 {
        self.pass.los_unix_us
    }

    /// Culmination (maximum elevation) time, unix microseconds UTC.
    #[getter]
    fn culmination_unix_us(&self) -> i64 {
        self.pass.culmination_unix_us
    }

    /// Elevation at culmination, degrees.
    #[getter]
    fn max_elevation_deg(&self) -> f64 {
        self.pass.max_elevation_deg
    }

    /// Pass duration (LOS minus AOS), seconds.
    #[getter]
    fn duration_s(&self) -> f64 {
        (self.pass.los_unix_us - self.pass.aos_unix_us) as f64 / 1.0e6
    }

    fn __repr__(&self) -> String {
        format!(
            "FleetPass(satellite_index={}, aos_unix_us={}, los_unix_us={}, culmination_unix_us={}, max_elevation_deg={:.4})",
            self.satellite_index,
            self.pass.aos_unix_us,
            self.pass.los_unix_us,
            self.pass.culmination_unix_us,
            self.pass.max_elevation_deg
        )
    }
}

/// A built-once fleet of already-parsed SGP4 satellites for repeated batch
/// geometry over a shared ground station and epoch grid.
///
/// Build it once from parsed [`Tle`] objects, then call `propagate`, `visible`,
/// `look_angle_arcs`, `ground_tracks`, and `passes` as often as you like: the
/// constellation OWNS its satellites and BORROWS them on each call, so the same
/// instance drives a live scene across frames with no re-parse. Each satellite
/// keeps the opsmode its [`Tle`] was constructed with, and its NORAD catalog
/// number becomes its id in `visible` / `catalog_numbers`. Input order is the
/// fleet order: the leading axis of every batch result and the `satellite_index`
/// of every [`FleetPass`].
///
/// It does no parsing or I/O: TLE text becomes satellites at the interface
/// boundary ([`Tle`] / [`parse_tle_file`]); the constellation only batches the
/// core geometry over the satellites it was handed. This is the Python form of
/// the WASM `Constellation` and Elixir's `Sidereon.Constellation`.
#[pyclass(module = "sidereon._sidereon", name = "Constellation")]
pub struct PyConstellation {
    satellites: Vec<Satellite>,
    ids: Vec<String>,
}

#[pymethods]
impl PyConstellation {
    /// Build a constellation from a sequence of already-parsed [`Tle`] objects,
    /// taking a snapshot of each satellite and its NORAD catalog number. The
    /// `Tle` objects are not consumed (they are cloned), so the caller keeps its
    /// handles.
    #[new]
    fn new(py: Python<'_>, satellites: Vec<Py<PyTle>>) -> Self {
        let mut sats = Vec::with_capacity(satellites.len());
        let mut ids = Vec::with_capacity(satellites.len());
        for tle in &satellites {
            let borrowed = tle.borrow(py);
            sats.push(borrowed.satellite.clone());
            ids.push(borrowed.elements.catalog_number.clone());
        }
        Self {
            satellites: sats,
            ids,
        }
    }

    /// Number of satellites in the constellation (the leading axis of every batch
    /// result).
    #[getter]
    fn satellite_count(&self) -> usize {
        self.satellites.len()
    }

    /// The satellites' NORAD catalog numbers, in fleet order.
    #[getter]
    fn catalog_numbers(&self) -> Vec<String> {
        self.ids.clone()
    }

    /// Propagate the whole constellation over a shared epoch grid in one call,
    /// borrowing it (the constellation is not consumed).
    ///
    /// `epochs_unix_us` is a 1-D numpy `int64` array of unix-microsecond UTC
    /// epochs shared by every satellite. Element `(i, j)` of the result is
    /// satellite `i` propagated to epoch `j`, bit-for-bit identical to the
    /// per-satellite [`Tle.propagate`] path. An empty constellation or empty
    /// epoch grid yields empty arrays. Raises `SidereonError` (naming the
    /// satellite index) if a satellite fails to propagate.
    fn propagate(&self, epochs_unix_us: PyReadonlyArray1<'_, i64>) -> PyResult<PyBatchPropagation> {
        let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
        let epoch_count = instants.len();
        let results = propagate_teme_batch_serial(&self.satellites, &instants);
        let arcs = unwrap_batch(results)?;
        Ok(PyBatchPropagation {
            epoch_count,
            positions_km: arcs
                .iter()
                .map(|arc| arc.iter().map(|p| p.position).collect())
                .collect(),
            velocities_km_s: arcs
                .iter()
                .map(|arc| arc.iter().map(|p| p.velocity).collect())
                .collect(),
        })
    }

    /// Satellites above `min_elevation_deg` from `station` at a single epoch,
    /// each with its catalog number and topocentric az/el/range, sorted by
    /// elevation (highest first).
    ///
    /// `epoch_unix_us` is a UTC unix-microsecond instant. The constellation form
    /// of [`visible_from_satellites`] (Elixir `Constellation.visible_from`).
    /// Per-satellite propagation or frame failures are skipped. Raises
    /// `SidereonError` on an invalid station or threshold.
    #[pyo3(signature = (station, epoch_unix_us, min_elevation_deg=0.0))]
    fn visible(
        &self,
        station: PyGroundStation,
        epoch_unix_us: i64,
        min_elevation_deg: f64,
    ) -> PyResult<Vec<PyVisibleSatellite>> {
        let datetime = UtcInstant::from_unix_microseconds(epoch_unix_us);
        let visible = core_visible_from_satellites(
            &self.satellites,
            &self.ids,
            station.inner,
            datetime,
            min_elevation_deg,
        )
        .map_err(to_solve_err)?;
        Ok(visible
            .into_iter()
            .map(PyVisibleSatellite::from_core)
            .collect())
    }

    /// Topocentric az/el/range arcs from `station` for every satellite over a
    /// shared epoch grid, in fleet order (element `i` is satellite `i`'s arc).
    ///
    /// A satellite that fails to propagate yields an empty [`LookAngles`] arc, so
    /// the result stays index-aligned with the constellation. The batched form of
    /// [`Tle.look_angles`].
    fn look_angle_arcs(
        &self,
        station: PyGroundStation,
        epochs_unix_us: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Vec<PyLookAngles>> {
        let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
        let results = look_angle_batch_serial(&self.satellites, station.inner, &instants);
        Ok(results
            .into_iter()
            .map(|arc| match arc {
                Ok(looks) => PyLookAngles {
                    azimuth_deg: looks.iter().map(|l| l.azimuth_deg).collect(),
                    elevation_deg: looks.iter().map(|l| l.elevation_deg).collect(),
                    range_km: looks.iter().map(|l| l.range_km).collect(),
                },
                Err(_) => PyLookAngles {
                    azimuth_deg: Vec::new(),
                    elevation_deg: Vec::new(),
                    range_km: Vec::new(),
                },
            })
            .collect())
    }

    /// Sub-satellite WGS84 ground tracks for every satellite over a shared epoch
    /// grid, in fleet order (element `i` is satellite `i`'s track).
    ///
    /// Each track is reduced TEME->GCRS->ITRS->geodetic by the engine's
    /// validated transforms (the same path as [`Tle.ground_track`]). A satellite
    /// that fails yields an empty [`GroundTrack`], keeping the result
    /// index-aligned with the constellation.
    fn ground_tracks(
        &self,
        epochs_unix_us: PyReadonlyArray1<'_, i64>,
    ) -> PyResult<Vec<PyGroundTrack>> {
        let instants = instants_from_unix_micros(&epochs_unix_us, EmptyPolicy::Allow)?;
        Ok(self
            .satellites
            .iter()
            .map(|satellite| match core_ground_track(satellite, &instants) {
                Ok(points) => PyGroundTrack {
                    latitude_deg: points.iter().map(|g| g.lat_rad.to_degrees()).collect(),
                    longitude_deg: points.iter().map(|g| g.lon_rad.to_degrees()).collect(),
                    altitude_km: points.iter().map(|g| g.height_m / 1000.0).collect(),
                },
                Err(_) => PyGroundTrack {
                    latitude_deg: Vec::new(),
                    longitude_deg: Vec::new(),
                    altitude_km: Vec::new(),
                },
            })
            .collect())
    }

    /// Passes over `station` within `[start_unix_us, end_unix_us)` for every
    /// satellite, flattened across the constellation: each [`FleetPass`] carries
    /// the fleet-order `satellite_index` it belongs to.
    ///
    /// The elevation is dense-sampled exactly as [`Tle.find_passes`].
    /// `elevation_mask_deg` defaults to 0, `step_seconds` to 30,
    /// `time_tolerance_s` to 1e-3. A satellite that fails to scan contributes no
    /// passes. Raises `ValueError` on a non-positive step or an end at or before
    /// the start.
    #[pyo3(signature = (
        station,
        start_unix_us,
        end_unix_us,
        *,
        elevation_mask_deg = 0.0,
        step_seconds = 30.0,
        time_tolerance_s = 1.0e-3,
    ))]
    fn passes(
        &self,
        station: PyGroundStation,
        start_unix_us: i64,
        end_unix_us: i64,
        elevation_mask_deg: f64,
        step_seconds: f64,
        time_tolerance_s: f64,
    ) -> PyResult<Vec<PyFleetPass>> {
        if end_unix_us <= start_unix_us {
            return Err(PyValueError::new_err(
                "end_unix_us must be after start_unix_us",
            ));
        }
        if step_seconds <= 0.0 {
            return Err(PyValueError::new_err("step_seconds must be positive"));
        }

        let start = UtcInstant::from_unix_microseconds(start_unix_us);
        let end = UtcInstant::from_unix_microseconds(end_unix_us);
        let options = PassFinderOptions {
            elevation_mask_deg,
            coarse_step_seconds: step_seconds,
            time_tolerance_seconds: time_tolerance_s,
        };

        let mut out = Vec::new();
        for (index, satellite) in self.satellites.iter().enumerate() {
            let passes =
                match find_passes_for_satellite(satellite, station.inner, start, end, options) {
                    Ok(passes) => passes,
                    Err(_) => continue,
                };
            for pass in &passes {
                out.push(PyFleetPass {
                    satellite_index: index,
                    pass: to_py_pass(pass),
                });
            }
        }
        Ok(out)
    }

    fn __len__(&self) -> usize {
        self.satellites.len()
    }

    fn __repr__(&self) -> String {
        format!("Constellation(satellite_count={})", self.satellites.len())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpsMode>()?;
    m.add_class::<PyForceModel>()?;
    m.add_class::<PyIntegrator>()?;
    m.add_class::<PyGroundStation>()?;
    m.add_class::<PyChecksumWarning>()?;
    m.add_class::<PyLoss>()?;
    m.add_class::<PyXScale>()?;
    m.add_class::<PyFitSample>()?;
    m.add_class::<PyFitEpoch>()?;
    m.add_class::<PyTleFitMetadata>()?;
    m.add_class::<PyFitStatistics>()?;
    m.add_class::<PyTleFit>()?;
    m.add_class::<PyTle>()?;
    m.add_class::<PyNamedTle>()?;
    m.add_class::<PyTleFile>()?;
    m.add_class::<PyTlePropagation>()?;
    m.add_class::<PyLookAngles>()?;
    m.add_class::<PyVisibilitySeries>()?;
    m.add_class::<PySatellitePass>()?;
    m.add_class::<PyEphemeris>()?;
    m.add_class::<PyBatchPropagation>()?;
    m.add_class::<PyBatchLookAngles>()?;
    m.add_class::<PyVisibleSatellite>()?;
    m.add_class::<PyGroundTrack>()?;
    m.add_class::<PyFleetPass>()?;
    m.add_class::<PyConstellation>()?;
    m.add_function(wrap_pyfunction!(parse_tle_file, m)?)?;
    m.add_function(wrap_pyfunction!(fit_tle, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_state, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(look_angles_batch, m)?)?;
    m.add_function(wrap_pyfunction!(visible_from_satellites, m)?)?;
    Ok(())
}
