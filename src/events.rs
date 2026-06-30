//! Events, body-angle geometry, and standalone GNSS DOP binding.
//!
//! This module marshals numpy vector batches into the core eclipse, angle, and
//! DOP kernels. It contains no modeling formulas: every output is delegated to
//! `sidereon-core`.

use std::collections::BTreeSet;
use std::f64::consts::FRAC_PI_2;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::astro::angles as core_angles;
use sidereon_core::astro::events::eclipse as core_eclipse;
use sidereon_core::astro::frames::transforms::geodetic_to_itrs;
use sidereon_core::geometry::{
    self as core_geometry, DopError, DopOptions, DopWeighting as CoreDopWeighting, LineOfSight,
    VisibilityOptions, Wgs84Geodetic,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

use crate::marshal::{fixed_array, rows3_from_array, EmptyPolicy, FinitePolicy, PyGnssSystem};
use crate::{np_array, to_solve_err, PySp3};

fn ensure_finite(name: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

fn read_positive_weights(weights: &PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    let view = weights.as_array();
    if view.is_empty() {
        return Err(PyValueError::new_err("weights array is empty"));
    }
    let mut out = Vec::with_capacity(view.len());
    for (index, value) in view.iter().copied().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(PyValueError::new_err(format!(
                "weights[{index}] must be finite and positive"
            )));
        }
        out.push(value);
    }
    Ok(out)
}

fn check_same_len(
    left_name: &str,
    left_len: usize,
    right_name: &str,
    right_len: usize,
) -> PyResult<()> {
    if left_len != right_len {
        return Err(PyValueError::new_err(format!(
            "{left_name} ({left_len}) and {right_name} ({right_len}) must have the same length"
        )));
    }
    Ok(())
}

fn weights_or_unit(
    weights: Option<&PyReadonlyArray1<'_, f64>>,
    expected_len: usize,
) -> PyResult<Vec<f64>> {
    match weights {
        Some(weights) => {
            let values = read_positive_weights(weights)?;
            check_same_len("geometry", expected_len, "weights", values.len())?;
            Ok(values)
        }
        None => Ok(vec![1.0; expected_len]),
    }
}

fn dop_from_rows(
    rows: &[[f64; 3]],
    weights: &[f64],
    receiver: &PyWgs84Geodetic,
) -> PyResult<PyDop> {
    let los: Vec<LineOfSight> = rows
        .iter()
        .map(|r| LineOfSight::new(r[0], r[1], r[2]))
        .collect();
    dop_from_los(&los, weights, receiver)
}

fn dop_from_los(
    los: &[LineOfSight],
    weights: &[f64],
    receiver: &PyWgs84Geodetic,
) -> PyResult<PyDop> {
    match core_geometry::dop(los, weights, Wgs84Geodetic::try_from(receiver)?) {
        Ok(dop) => Ok(PyDop::from(dop)),
        Err(DopError::InvalidInput { field, reason }) => Err(PyValueError::new_err(format!(
            "invalid DOP input {field}: {reason}"
        ))),
        Err(DopError::TooFewSatellites) => Err(PyValueError::new_err(
            "at least four geometry rows are required",
        )),
        Err(DopError::Singular) => Err(to_solve_err(DopError::Singular)),
    }
}

fn receiver_ecef_m_from_geodetic(receiver: &PyWgs84Geodetic) -> PyResult<[f64; 3]> {
    let (x_km, y_km, z_km) = geodetic_to_itrs(
        receiver.lat_rad.to_degrees(),
        receiver.lon_rad.to_degrees(),
        receiver.height_m / 1000.0,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok([x_km * 1000.0, y_km * 1000.0, z_km * 1000.0])
}

fn receiver_ecef_m_from_station(station: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
    if let Ok(receiver) = station.extract::<PyRef<'_, PyWgs84Geodetic>>() {
        return receiver_ecef_m_from_geodetic(&receiver);
    }
    if let Ok(values) = station.extract::<PyReadonlyArray1<'_, f64>>() {
        return fixed_array("station", &values, FinitePolicy::RequireFinite);
    }
    let values: Vec<f64> = station.extract().map_err(|_| {
        PyValueError::new_err("station must be Wgs84Geodetic or an ECEF metre vector of length 3")
    })?;
    if values.len() != 3 {
        return Err(PyValueError::new_err(
            "station ECEF metre vector must have length 3",
        ));
    }
    let out = [values[0], values[1], values[2]];
    if out.iter().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(
            "station ECEF metre vector must contain only finite values",
        ));
    }
    Ok(out)
}

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite token {token:?}: {err}")))
}

fn parse_satellites(values: Vec<String>) -> PyResult<Vec<GnssSatelliteId>> {
    if values.is_empty() {
        return Err(PyValueError::new_err("satellites must not be empty"));
    }
    values
        .iter()
        .map(|value| parse_satellite(value))
        .collect::<PyResult<Vec<_>>>()
}

fn parse_systems(values: Vec<String>) -> PyResult<BTreeSet<GnssSystem>> {
    if values.is_empty() {
        return Err(PyValueError::new_err("systems must not be empty"));
    }
    values
        .iter()
        .map(|value| parse_system(value))
        .collect::<PyResult<BTreeSet<_>>>()
}

fn parse_system(value: &str) -> PyResult<GnssSystem> {
    match value.trim().to_ascii_uppercase().as_str() {
        "G" | "GPS" => Ok(GnssSystem::Gps),
        "R" | "GLO" | "GLONASS" => Ok(GnssSystem::Glonass),
        "E" | "GAL" | "GALILEO" => Ok(GnssSystem::Galileo),
        "C" | "BDS" | "BEIDOU" => Ok(GnssSystem::BeiDou),
        "J" | "QZSS" => Ok(GnssSystem::Qzss),
        "I" | "IRNSS" | "NAVIC" => Ok(GnssSystem::Navic),
        "S" | "SBAS" => Ok(GnssSystem::Sbas),
        other => Err(PyValueError::new_err(format!(
            "unknown GNSS system {other:?}; expected one of G, R, E, C, J, I, S"
        ))),
    }
}

/// Illumination state relative to Earth's conical shadow.
#[pyclass(module = "sidereon._sidereon", name = "EclipseStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyEclipseStatus {
    /// Full sunlight.
    SUNLIT,
    /// Partial shadow.
    PENUMBRA,
    /// Full umbra.
    UMBRA,
}

impl From<core_eclipse::EclipseStatus> for PyEclipseStatus {
    fn from(value: core_eclipse::EclipseStatus) -> Self {
        match value {
            core_eclipse::EclipseStatus::Sunlit => PyEclipseStatus::SUNLIT,
            core_eclipse::EclipseStatus::Penumbra => PyEclipseStatus::PENUMBRA,
            core_eclipse::EclipseStatus::Umbra => PyEclipseStatus::UMBRA,
        }
    }
}

#[pymethods]
impl PyEclipseStatus {
    fn __repr__(&self) -> &'static str {
        match self {
            PyEclipseStatus::SUNLIT => "EclipseStatus.SUNLIT",
            PyEclipseStatus::PENUMBRA => "EclipseStatus.PENUMBRA",
            PyEclipseStatus::UMBRA => "EclipseStatus.UMBRA",
        }
    }
}

/// WGS84 receiver geodetic coordinates for DOP.
#[pyclass(module = "sidereon._sidereon", name = "Wgs84Geodetic")]
#[derive(Clone, Copy)]
pub struct PyWgs84Geodetic {
    lat_rad: f64,
    lon_rad: f64,
    height_m: f64,
}

#[pymethods]
impl PyWgs84Geodetic {
    /// Build a WGS84 geodetic coordinate from radians and metres.
    #[new]
    #[pyo3(signature = (lat_rad, lon_rad, height_m=0.0))]
    fn new(lat_rad: f64, lon_rad: f64, height_m: f64) -> PyResult<Self> {
        let lat_rad = ensure_finite("lat_rad", lat_rad)?;
        if !(-FRAC_PI_2..=FRAC_PI_2).contains(&lat_rad) {
            return Err(PyValueError::new_err("lat_rad must be in [-pi/2, pi/2]"));
        }
        Ok(Self {
            lat_rad,
            lon_rad: ensure_finite("lon_rad", lon_rad)?,
            height_m: ensure_finite("height_m", height_m)?,
        })
    }

    /// Geodetic latitude, radians.
    #[getter]
    fn lat_rad(&self) -> f64 {
        self.lat_rad
    }

    /// Geodetic longitude, radians east.
    #[getter]
    fn lon_rad(&self) -> f64 {
        self.lon_rad
    }

    /// Ellipsoidal height above WGS84, metres.
    #[getter]
    fn height_m(&self) -> f64 {
        self.height_m
    }

    fn __repr__(&self) -> String {
        format!(
            "Wgs84Geodetic(lat_rad={}, lon_rad={}, height_m={})",
            self.lat_rad, self.lon_rad, self.height_m
        )
    }

    fn __eq__(&self, other: &PyWgs84Geodetic) -> bool {
        self.lat_rad == other.lat_rad
            && self.lon_rad == other.lon_rad
            && self.height_m == other.height_m
    }
}

impl PyWgs84Geodetic {
    /// Build from a core [`Wgs84Geodetic`]. The core converter already validated
    /// the latitude/longitude ranges, so no re-validation is performed here.
    pub(crate) fn from_core(value: Wgs84Geodetic) -> Self {
        Self {
            lat_rad: value.lat_rad,
            lon_rad: value.lon_rad,
            height_m: value.height_m,
        }
    }
}

impl TryFrom<&PyWgs84Geodetic> for Wgs84Geodetic {
    type Error = PyErr;

    fn try_from(value: &PyWgs84Geodetic) -> Result<Self, Self::Error> {
        Wgs84Geodetic::new(value.lat_rad, value.lon_rad, value.height_m)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

/// DOP row weighting policy for SP3-derived geometry series.
#[pyclass(module = "sidereon._sidereon", name = "DopWeighting", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyDopWeighting {
    /// Unit weights for all satellite rows.
    UNIT,
    /// Elevation weights using `sin(elevation)^2`.
    ELEVATION,
}

impl PyDopWeighting {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "unit" => Ok(Self::UNIT),
            "elevation" => Ok(Self::ELEVATION),
            other => Err(PyValueError::new_err(format!(
                "unknown DOP weighting {other:?}; expected \"unit\" or \"elevation\""
            ))),
        }
    }
}

impl From<PyDopWeighting> for CoreDopWeighting {
    fn from(value: PyDopWeighting) -> Self {
        match value {
            PyDopWeighting::UNIT => CoreDopWeighting::Unit,
            PyDopWeighting::ELEVATION => CoreDopWeighting::Elevation,
        }
    }
}

#[pymethods]
impl PyDopWeighting {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::UNIT => "unit",
            Self::ELEVATION => "elevation",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::UNIT => "DopWeighting.UNIT",
            Self::ELEVATION => "DopWeighting.ELEVATION",
        }
    }
}

fn extract_dop_weighting(obj: &Bound<'_, PyAny>) -> PyResult<PyDopWeighting> {
    if let Ok(value) = obj.extract::<PyDopWeighting>() {
        return Ok(value);
    }
    PyDopWeighting::from_label(&obj.extract::<String>()?)
}

/// GNSS dilution-of-precision scalars.
#[pyclass(module = "sidereon._sidereon", name = "Dop")]
#[derive(Clone)]
pub struct PyDop {
    gdop: f64,
    pdop: f64,
    hdop: f64,
    vdop: f64,
    tdop: f64,
    system_tdops: Vec<(PyGnssSystem, f64)>,
}

impl From<core_geometry::Dop> for PyDop {
    fn from(value: core_geometry::Dop) -> Self {
        Self {
            gdop: value.gdop,
            pdop: value.pdop,
            hdop: value.hdop,
            vdop: value.vdop,
            tdop: value.tdop,
            system_tdops: value
                .system_tdops
                .into_iter()
                .map(|(sys, tdop)| (sys.into(), tdop))
                .collect(),
        }
    }
}

#[pymethods]
impl PyDop {
    /// Compute DOP from ECEF receiver-to-satellite LOS rows.
    ///
    /// `line_of_sight` is a numpy `(n, 3)` array of ECEF unit vectors,
    /// `receiver` is WGS84 geodetic, and optional `weights` is a positive
    /// numpy `(n,)` array. At least four rows are required.
    #[staticmethod]
    #[pyo3(signature = (line_of_sight, receiver, weights=None))]
    fn from_line_of_sight(
        line_of_sight: PyReadonlyArray2<'_, f64>,
        receiver: PyRef<'_, PyWgs84Geodetic>,
        weights: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let rows = rows3_from_array(
            "line_of_sight",
            &line_of_sight,
            EmptyPolicy::Reject,
            FinitePolicy::RequireFinite,
        )?;
        let weights = weights_or_unit(weights.as_ref(), rows.len())?;
        dop_from_rows(&rows, &weights, &receiver)
    }

    /// Compute DOP from topocentric azimuth/elevation rows.
    ///
    /// `azimuth_deg` and `elevation_deg` are numpy `(n,)` arrays in degrees,
    /// using azimuth clockwise from geodetic north. `receiver` defines the local
    /// ENU frame. Optional `weights` is a positive numpy `(n,)` array.
    #[staticmethod]
    #[pyo3(signature = (azimuth_deg, elevation_deg, receiver, weights=None))]
    fn from_az_el(
        azimuth_deg: PyReadonlyArray1<'_, f64>,
        elevation_deg: PyReadonlyArray1<'_, f64>,
        receiver: PyRef<'_, PyWgs84Geodetic>,
        weights: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let azimuth = azimuth_deg
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let elevation = elevation_deg
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        if azimuth.is_empty() {
            return Err(PyValueError::new_err("azimuth_deg array is empty"));
        }
        check_same_len(
            "azimuth_deg",
            azimuth.len(),
            "elevation_deg",
            elevation.len(),
        )?;
        let core_receiver = Wgs84Geodetic::try_from(&*receiver)?;
        let mut los = Vec::with_capacity(azimuth.len());
        for (index, (&az, &el)) in azimuth.iter().zip(elevation.iter()).enumerate() {
            if !az.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "azimuth_deg[{index}] must be finite"
                )));
            }
            if !el.is_finite() || !(-90.0..=90.0).contains(&el) {
                return Err(PyValueError::new_err(format!(
                    "elevation_deg[{index}] must be finite and in [-90, 90]"
                )));
            }
            los.push(
                core_geometry::line_of_sight_from_az_el_deg(az, el, core_receiver)
                    .map_err(to_solve_err)?,
            );
        }
        let weights = weights_or_unit(weights.as_ref(), los.len())?;
        dop_from_los(&los, &weights, &receiver)
    }

    /// Geometric DOP.
    #[getter]
    fn gdop(&self) -> f64 {
        self.gdop
    }

    /// Position DOP.
    #[getter]
    fn pdop(&self) -> f64 {
        self.pdop
    }

    /// Horizontal DOP.
    #[getter]
    fn hdop(&self) -> f64 {
        self.hdop
    }

    /// Vertical DOP.
    #[getter]
    fn vdop(&self) -> f64 {
        self.vdop
    }

    /// Time DOP.
    #[getter]
    fn tdop(&self) -> f64 {
        self.tdop
    }

    /// Per-clock-column time DOP, one entry per estimated receiver clock (one
    /// per constellation in a multi-system solve), as `(system, tdop)` pairs in
    /// clock-column order. For a single-system geometry this is a one-element
    /// list whose value equals `tdop`.
    #[getter]
    fn system_tdops(&self) -> Vec<(PyGnssSystem, f64)> {
        self.system_tdops.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Dop(gdop={}, pdop={}, hdop={}, vdop={}, tdop={}, system_tdops={:?})",
            self.gdop, self.pdop, self.hdop, self.vdop, self.tdop, self.system_tdops
        )
    }

    fn __eq__(&self, other: &PyDop) -> bool {
        self.gdop == other.gdop
            && self.pdop == other.pdop
            && self.hdop == other.hdop
            && self.vdop == other.vdop
            && self.tdop == other.tdop
            && self.system_tdops == other.system_tdops
    }
}

/// DOP values sampled from an SP3 precise product over an epoch grid.
///
/// Arrays contain only samples with finite DOP. `step_index` maps each row back
/// to the input `j2000_seconds` grid, while `j2000_seconds` contains the
/// successful sample epochs in seconds since J2000.
#[pyclass(module = "sidereon._sidereon", name = "DopSeries")]
pub struct PyDopSeries {
    step_index: Vec<i64>,
    j2000_seconds: Vec<f64>,
    gdop: Vec<f64>,
    pdop: Vec<f64>,
    hdop: Vec<f64>,
    vdop: Vec<f64>,
    tdop: Vec<f64>,
    satellite_count: Vec<i64>,
    satellites: Vec<Vec<String>>,
}

#[pymethods]
impl PyDopSeries {
    /// Input epoch indices for finite DOP samples, numpy `(m,)` int64.
    #[getter]
    fn step_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.step_index)
    }

    /// Successful sample epochs, seconds since J2000, numpy `(m,)` float64.
    #[getter]
    fn j2000_seconds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.j2000_seconds)
    }

    /// Geometric DOP samples, numpy `(m,)` float64.
    #[getter]
    fn gdop<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.gdop)
    }

    /// Position DOP samples, numpy `(m,)` float64.
    #[getter]
    fn pdop<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.pdop)
    }

    /// Horizontal DOP samples, numpy `(m,)` float64.
    #[getter]
    fn hdop<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.hdop)
    }

    /// Vertical DOP samples, numpy `(m,)` float64.
    #[getter]
    fn vdop<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.vdop)
    }

    /// Time DOP samples, numpy `(m,)` float64.
    #[getter]
    fn tdop<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.tdop)
    }

    /// Number of satellites used at each finite sample, numpy `(m,)` int64.
    #[getter]
    fn satellite_count<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_slice(py, &self.satellite_count)
    }

    /// Satellite tokens used at each finite sample, index-aligned to the arrays.
    #[getter]
    fn satellites(&self) -> Vec<Vec<String>> {
        self.satellites.clone()
    }

    /// Number of finite DOP samples.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.gdop.len()
    }

    fn __len__(&self) -> usize {
        self.gdop.len()
    }

    fn __repr__(&self) -> String {
        format!("DopSeries(epoch_count={})", self.gdop.len())
    }
}

/// Shadow fraction in `[0, 1]` for satellite and Sun position batches in km.
#[pyfunction]
fn shadow_fraction<'py>(
    py: Python<'py>,
    satellite_position_km: PyReadonlyArray2<'_, f64>,
    sun_position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let satellites = rows3_from_array(
        "satellite_position_km",
        &satellite_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let suns = rows3_from_array(
        "sun_position_km",
        &sun_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    check_same_len(
        "satellite_position_km",
        satellites.len(),
        "sun_position_km",
        suns.len(),
    )?;
    let out: Vec<f64> = satellites
        .iter()
        .zip(suns.iter())
        .map(|(&sat, &sun)| core_eclipse::shadow_fraction(sat, sun).map_err(to_solve_err))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray1::from_vec(py, out))
}

/// Eclipse status for satellite and Sun position batches in km.
#[pyfunction]
fn eclipse_status(
    satellite_position_km: PyReadonlyArray2<'_, f64>,
    sun_position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<PyEclipseStatus>> {
    let satellites = rows3_from_array(
        "satellite_position_km",
        &satellite_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let suns = rows3_from_array(
        "sun_position_km",
        &sun_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    check_same_len(
        "satellite_position_km",
        satellites.len(),
        "sun_position_km",
        suns.len(),
    )?;
    satellites
        .iter()
        .zip(suns.iter())
        .map(|(&sat, &sun)| {
            core_eclipse::status(sat, sun)
                .map(PyEclipseStatus::from)
                .map_err(to_solve_err)
        })
        .collect()
}

/// Angle in degrees between satellite nadir and the Sun direction.
#[pyfunction]
fn sun_angle<'py>(
    py: Python<'py>,
    satellite_position_km: PyReadonlyArray2<'_, f64>,
    sun_position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let satellites = rows3_from_array(
        "satellite_position_km",
        &satellite_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let suns = rows3_from_array(
        "sun_position_km",
        &sun_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    check_same_len(
        "satellite_position_km",
        satellites.len(),
        "sun_position_km",
        suns.len(),
    )?;
    let out: Vec<f64> = satellites
        .iter()
        .zip(suns.iter())
        .map(|(&sat, &sun)| core_angles::sun_angle(sat, sun).map_err(to_solve_err))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray1::from_vec(py, out))
}

/// Angle in degrees between satellite nadir and the Moon direction.
#[pyfunction]
fn moon_angle<'py>(
    py: Python<'py>,
    satellite_position_km: PyReadonlyArray2<'_, f64>,
    moon_position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let satellites = rows3_from_array(
        "satellite_position_km",
        &satellite_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let moons = rows3_from_array(
        "moon_position_km",
        &moon_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    check_same_len(
        "satellite_position_km",
        satellites.len(),
        "moon_position_km",
        moons.len(),
    )?;
    let out: Vec<f64> = satellites
        .iter()
        .zip(moons.iter())
        .map(|(&sat, &moon)| core_angles::moon_angle(sat, moon).map_err(to_solve_err))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray1::from_vec(py, out))
}

/// Sun elevation in degrees above the satellite local horizontal plane.
#[pyfunction]
fn sun_elevation<'py>(
    py: Python<'py>,
    satellite_position_km: PyReadonlyArray2<'_, f64>,
    sun_position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let satellites = rows3_from_array(
        "satellite_position_km",
        &satellite_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let suns = rows3_from_array(
        "sun_position_km",
        &sun_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    check_same_len(
        "satellite_position_km",
        satellites.len(),
        "sun_position_km",
        suns.len(),
    )?;
    let out: Vec<f64> = satellites
        .iter()
        .zip(suns.iter())
        .map(|(&sat, &sun)| core_angles::sun_elevation(sat, sun).map_err(to_solve_err))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray1::from_vec(py, out))
}

/// Sun-satellite-observer phase angle in degrees.
#[pyfunction]
fn phase_angle<'py>(
    py: Python<'py>,
    satellite_position_km: PyReadonlyArray2<'_, f64>,
    sun_position_km: PyReadonlyArray2<'_, f64>,
    observer_position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let satellites = rows3_from_array(
        "satellite_position_km",
        &satellite_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let suns = rows3_from_array(
        "sun_position_km",
        &sun_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let observers = rows3_from_array(
        "observer_position_km",
        &observer_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    check_same_len(
        "satellite_position_km",
        satellites.len(),
        "sun_position_km",
        suns.len(),
    )?;
    check_same_len(
        "satellite_position_km",
        satellites.len(),
        "observer_position_km",
        observers.len(),
    )?;
    let out: Vec<f64> = satellites
        .iter()
        .zip(suns.iter())
        .zip(observers.iter())
        .map(|((&sat, &sun), &observer)| {
            core_angles::phase_angle(sat, sun, observer).map_err(to_solve_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray1::from_vec(py, out))
}

/// Angular radius in degrees of Earth as seen from each satellite position.
#[pyfunction]
fn earth_angular_radius<'py>(
    py: Python<'py>,
    satellite_position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let satellites = rows3_from_array(
        "satellite_position_km",
        &satellite_position_km,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let out: Vec<f64> = satellites
        .iter()
        .map(|&sat| core_angles::earth_angular_radius(sat).map_err(to_solve_err))
        .collect::<PyResult<Vec<_>>>()?;
    Ok(PyArray1::from_vec(py, out))
}

/// GNSS dilution of precision from ECEF line-of-sight unit rows and weights.
#[pyfunction]
fn gnss_dop(
    line_of_sight: PyReadonlyArray2<'_, f64>,
    weights: PyReadonlyArray1<'_, f64>,
    receiver: PyRef<'_, PyWgs84Geodetic>,
) -> PyResult<PyDop> {
    let rows = rows3_from_array(
        "line_of_sight",
        &line_of_sight,
        EmptyPolicy::Reject,
        FinitePolicy::RequireFinite,
    )?;
    let weights = read_positive_weights(&weights)?;
    check_same_len("line_of_sight", rows.len(), "weights", weights.len())?;
    dop_from_rows(&rows, &weights, &receiver)
}

/// Sample SP3-derived GNSS DOP over a J2000 epoch grid.
///
/// `station` is either [`Wgs84Geodetic`] or an ECEF metre vector `(3,)`.
/// `j2000_seconds` is a numpy `(n,)` float64 grid in the SP3 product time
/// scale. Samples with too few satellites or singular geometry are omitted.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    sp3,
    station,
    j2000_seconds,
    *,
    satellites=None,
    elevation_mask_deg=5.0,
    systems=None,
    weighting=PyDopWeighting::UNIT,
    light_time=false,
))]
fn gnss_dop_series(
    sp3: &PySp3,
    station: &Bound<'_, PyAny>,
    j2000_seconds: PyReadonlyArray1<'_, f64>,
    satellites: Option<Vec<String>>,
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
    #[pyo3(from_py_with = extract_dop_weighting)] weighting: PyDopWeighting,
    light_time: bool,
) -> PyResult<PyDopSeries> {
    let epochs = j2000_seconds
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    if epochs.is_empty() {
        return Err(PyValueError::new_err("j2000_seconds array is empty"));
    }
    for (index, value) in epochs.iter().enumerate() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "j2000_seconds[{index}] must be finite"
            )));
        }
    }
    if !elevation_mask_deg.is_finite() {
        return Err(PyValueError::new_err("elevation_mask_deg must be finite"));
    }

    let receiver_ecef_m = receiver_ecef_m_from_station(station)?;
    let explicit_satellites = satellites.map(parse_satellites).transpose()?;
    let systems = systems.map(parse_systems).transpose()?;
    let options = DopOptions {
        visibility: VisibilityOptions {
            elevation_mask_deg,
            systems,
        },
        weighting: weighting.into(),
        light_time,
    };

    let mut out = PyDopSeries {
        step_index: Vec::new(),
        j2000_seconds: Vec::new(),
        gdop: Vec::new(),
        pdop: Vec::new(),
        hdop: Vec::new(),
        vdop: Vec::new(),
        tdop: Vec::new(),
        satellite_count: Vec::new(),
        satellites: Vec::new(),
    };
    let all_satellites = sp3.inner.satellites();
    for (index, &epoch) in epochs.iter().enumerate() {
        let result = core_geometry::dop_at_epoch(
            &sp3.inner,
            all_satellites,
            explicit_satellites.as_deref(),
            receiver_ecef_m,
            epoch,
            &options,
        );
        let Ok(geometry) = result else {
            continue;
        };
        out.step_index.push(index as i64);
        out.j2000_seconds.push(epoch);
        out.gdop.push(geometry.dop.gdop);
        out.pdop.push(geometry.dop.pdop);
        out.hdop.push(geometry.dop.hdop);
        out.vdop.push(geometry.dop.vdop);
        out.tdop.push(geometry.dop.tdop);
        out.satellite_count.push(geometry.satellites.len() as i64);
        out.satellites.push(
            geometry
                .satellites
                .iter()
                .map(ToString::to_string)
                .collect(),
        );
    }
    Ok(out)
}

/// DOP scalars at one epoch plus the satellites that contributed rows.
#[pyclass(module = "sidereon._sidereon", name = "DopAtEpoch")]
#[derive(Clone)]
pub struct PyDopAtEpoch {
    dop: PyDop,
    satellites: Vec<String>,
}

#[pymethods]
impl PyDopAtEpoch {
    /// The [`Dop`] scalars for this epoch.
    #[getter]
    fn dop(&self) -> PyDop {
        self.dop.clone()
    }

    /// Canonical satellite tokens that contributed a line-of-sight row.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.satellites.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DopAtEpoch(n_satellites={}, gdop={}, pdop={})",
            self.satellites.len(),
            self.dop.gdop(),
            self.dop.pdop()
        )
    }
}

/// One sampled point of a uniform DOP series: the input step index and the
/// [`DopAtEpoch`] geometry at that sample.
#[pyclass(module = "sidereon._sidereon", name = "DopSeriesPoint")]
#[derive(Clone)]
pub struct PyDopSeriesPoint {
    step_index: usize,
    geometry: PyDopAtEpoch,
}

#[pymethods]
impl PyDopSeriesPoint {
    /// Zero-based sample index from the series start.
    #[getter]
    fn step_index(&self) -> usize {
        self.step_index
    }

    /// The [`DopAtEpoch`] geometry at this sample.
    #[getter]
    fn geometry(&self) -> PyDopAtEpoch {
        self.geometry.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DopSeriesPoint(step_index={}, n_satellites={})",
            self.step_index,
            self.geometry.satellites.len()
        )
    }
}

fn build_dop_options(
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
    weighting: PyDopWeighting,
    light_time: bool,
) -> PyResult<DopOptions> {
    if !elevation_mask_deg.is_finite() {
        return Err(PyValueError::new_err("elevation_mask_deg must be finite"));
    }
    let systems = systems.map(parse_systems).transpose()?;
    Ok(DopOptions {
        visibility: VisibilityOptions {
            elevation_mask_deg,
            systems,
        },
        weighting: weighting.into(),
        light_time,
    })
}

fn dop_at_epoch_to_py(geometry: core_geometry::DopAtEpoch) -> PyDopAtEpoch {
    PyDopAtEpoch {
        dop: PyDop::from(geometry.dop),
        satellites: geometry
            .satellites
            .iter()
            .map(ToString::to_string)
            .collect(),
    }
}

/// SP3-derived GNSS DOP at one epoch, with the contributing satellites.
///
/// `station` is either [`Wgs84Geodetic`] or an ECEF metre vector `(3,)`;
/// `t_rx_j2000_s` is the receive epoch in seconds since J2000 in the SP3 product
/// time scale. Delegates to the core `dop_at_epoch`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    sp3,
    station,
    t_rx_j2000_s,
    *,
    satellites=None,
    elevation_mask_deg=5.0,
    systems=None,
    weighting=PyDopWeighting::UNIT,
    light_time=false,
))]
fn gnss_dop_at_epoch(
    sp3: &PySp3,
    station: &Bound<'_, PyAny>,
    t_rx_j2000_s: f64,
    satellites: Option<Vec<String>>,
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
    #[pyo3(from_py_with = extract_dop_weighting)] weighting: PyDopWeighting,
    light_time: bool,
) -> PyResult<PyDopAtEpoch> {
    if !t_rx_j2000_s.is_finite() {
        return Err(PyValueError::new_err("t_rx_j2000_s must be finite"));
    }
    let receiver_ecef_m = receiver_ecef_m_from_station(station)?;
    let explicit_satellites = satellites.map(parse_satellites).transpose()?;
    let options = build_dop_options(elevation_mask_deg, systems, weighting, light_time)?;
    let geometry = core_geometry::dop_at_epoch(
        &sp3.inner,
        sp3.inner.satellites(),
        explicit_satellites.as_deref(),
        receiver_ecef_m,
        t_rx_j2000_s,
        &options,
    )
    .map_err(to_solve_err)?;
    Ok(dop_at_epoch_to_py(geometry))
}

/// Sample SP3-derived GNSS DOP over a uniform inclusive `[start, end]` window.
///
/// `step_seconds` is the sample spacing. Singular or underdetermined samples are
/// omitted. Delegates to the core `dop_series`; each returned point carries the
/// core step index and the [`DopAtEpoch`] geometry.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    sp3,
    station,
    start_j2000_s,
    end_j2000_s,
    step_seconds,
    *,
    satellites=None,
    elevation_mask_deg=5.0,
    systems=None,
    weighting=PyDopWeighting::UNIT,
    light_time=false,
))]
fn gnss_dop_series_uniform(
    sp3: &PySp3,
    station: &Bound<'_, PyAny>,
    start_j2000_s: f64,
    end_j2000_s: f64,
    step_seconds: u64,
    satellites: Option<Vec<String>>,
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
    #[pyo3(from_py_with = extract_dop_weighting)] weighting: PyDopWeighting,
    light_time: bool,
) -> PyResult<Vec<PyDopSeriesPoint>> {
    if !start_j2000_s.is_finite() || !end_j2000_s.is_finite() {
        return Err(PyValueError::new_err(
            "start_j2000_s and end_j2000_s must be finite",
        ));
    }
    let receiver_ecef_m = receiver_ecef_m_from_station(station)?;
    let explicit_satellites = satellites.map(parse_satellites).transpose()?;
    let options = build_dop_options(elevation_mask_deg, systems, weighting, light_time)?;
    let points = core_geometry::dop_series(
        &sp3.inner,
        sp3.inner.satellites(),
        explicit_satellites.as_deref(),
        receiver_ecef_m,
        (start_j2000_s, end_j2000_s),
        step_seconds,
        &options,
    )
    .map_err(to_solve_err)?;
    Ok(points
        .into_iter()
        .map(|point| PyDopSeriesPoint {
            step_index: point.step_index,
            geometry: dop_at_epoch_to_py(point.geometry),
        })
        .collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEclipseStatus>()?;
    m.add_class::<PyWgs84Geodetic>()?;
    m.add_class::<PyDopWeighting>()?;
    m.add_class::<PyDop>()?;
    m.add_class::<PyDopSeries>()?;
    m.add_class::<PyDopAtEpoch>()?;
    m.add_class::<PyDopSeriesPoint>()?;
    m.add_function(wrap_pyfunction!(shadow_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(eclipse_status, m)?)?;
    m.add_function(wrap_pyfunction!(sun_angle, m)?)?;
    m.add_function(wrap_pyfunction!(moon_angle, m)?)?;
    m.add_function(wrap_pyfunction!(sun_elevation, m)?)?;
    m.add_function(wrap_pyfunction!(phase_angle, m)?)?;
    m.add_function(wrap_pyfunction!(earth_angular_radius, m)?)?;
    m.add_function(wrap_pyfunction!(gnss_dop, m)?)?;
    m.add_function(wrap_pyfunction!(gnss_dop_series, m)?)?;
    m.add_function(wrap_pyfunction!(gnss_dop_at_epoch, m)?)?;
    m.add_function(wrap_pyfunction!(gnss_dop_series_uniform, m)?)?;
    Ok(())
}
