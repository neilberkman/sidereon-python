//! SBAS protection-level bindings.
//!
//! This module marshals SBAS protection geometry, supplied range-error models,
//! fixed K multipliers, and protection-level outputs to `sidereon_core::sbas_pl`.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::geometry::LineOfSight;
use sidereon_core::sbas_pl::{
    sbas_protection_levels as core_sbas_protection_levels, AirborneModel, DegradationParams,
    ProtectionGeometry, ProtectionRow, SbasErrorModel, SbasKMultipliers, SbasPlError,
    SbasProtection, SbasSisError,
};
use sidereon_core::{GnssSatelliteId, GnssSystem, Wgs84Geodetic};

use crate::events::PyWgs84Geodetic;
use crate::marshal::{fixed_array, FinitePolicy, PyGnssSystem};
use crate::np_array;
use crate::sbas_ssr::PySbasCorrectionStore;

fn to_sbas_pl_err(err: SbasPlError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite_id {token:?}: {err}")))
}

fn system_from_satellite(
    satellite_id: GnssSatelliteId,
    system: Option<PyGnssSystem>,
) -> GnssSystem {
    system.map(Into::into).unwrap_or(satellite_id.system)
}

/// SBAS protection-level failure kind.
#[pyclass(module = "sidereon._sidereon", name = "SbasPlError", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PySbasPlError {
    /// The geometry has too few independent rows.
    INSUFFICIENT_GEOMETRY,
    /// A matrix operation or covariance projection failed.
    NUMERICAL_FAILURE,
    /// The supplied range-error model is outside its valid domain.
    INVALID_ERROR_MODEL,
}

impl From<SbasPlError> for PySbasPlError {
    fn from(value: SbasPlError) -> Self {
        match value {
            SbasPlError::InsufficientGeometry => Self::INSUFFICIENT_GEOMETRY,
            SbasPlError::NumericalFailure => Self::NUMERICAL_FAILURE,
            SbasPlError::InvalidErrorModel => Self::INVALID_ERROR_MODEL,
        }
    }
}

#[pymethods]
impl PySbasPlError {
    /// Stable lowercase label for the error kind.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::INSUFFICIENT_GEOMETRY => "insufficient_geometry",
            Self::NUMERICAL_FAILURE => "numerical_failure",
            Self::INVALID_ERROR_MODEL => "invalid_error_model",
        }
    }

    /// Return a compact representation of the error kind.
    fn __repr__(&self) -> &'static str {
        match self {
            Self::INSUFFICIENT_GEOMETRY => "SbasPlError.INSUFFICIENT_GEOMETRY",
            Self::NUMERICAL_FAILURE => "SbasPlError.NUMERICAL_FAILURE",
            Self::INVALID_ERROR_MODEL => "SbasPlError.INVALID_ERROR_MODEL",
        }
    }
}

/// One SBAS protection-level geometry row.
#[pyclass(module = "sidereon._sidereon", name = "ProtectionRow")]
#[derive(Clone, Copy)]
pub struct PyProtectionRow {
    inner: ProtectionRow,
}

impl From<ProtectionRow> for PyProtectionRow {
    fn from(inner: ProtectionRow) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyProtectionRow {
    /// Build one protection geometry row.
    ///
    /// `satellite_id` is a token such as `"G01"`. `line_of_sight_ecef` is a
    /// receiver-to-satellite ECEF unit vector with three components, and
    /// `elevation_rad` is the receiver elevation angle in radians.
    #[new]
    #[pyo3(signature = (satellite_id, line_of_sight_ecef, elevation_rad, system=None))]
    fn new(
        satellite_id: &str,
        line_of_sight_ecef: PyReadonlyArray1<'_, f64>,
        elevation_rad: f64,
        system: Option<PyGnssSystem>,
    ) -> PyResult<Self> {
        let id = parse_satellite(satellite_id)?;
        let los = fixed_array::<3>(
            "line_of_sight_ecef",
            &line_of_sight_ecef,
            FinitePolicy::RequireFinite,
        )?;
        Ok(Self {
            inner: ProtectionRow {
                id,
                line_of_sight: LineOfSight::new(los[0], los[1], los[2]),
                system: system_from_satellite(id, system),
                elevation_rad,
            },
        })
    }

    /// Satellite token used to match the SBAS error model.
    #[getter]
    fn satellite_id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Receiver-to-satellite ECEF unit vector as a numpy array.
    #[getter]
    fn line_of_sight_ecef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(
            py,
            &[
                self.inner.line_of_sight.e_x,
                self.inner.line_of_sight.e_y,
                self.inner.line_of_sight.e_z,
            ],
        )
    }

    /// Constellation owning this row.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    /// Receiver elevation angle in radians.
    #[getter]
    fn elevation_rad(&self) -> f64 {
        self.inner.elevation_rad
    }

    /// Return a compact representation of the protection row.
    fn __repr__(&self) -> String {
        format!(
            "ProtectionRow(satellite_id={:?}, system={}, elevation_rad={})",
            self.inner.id.to_string(),
            self.inner.system.as_str(),
            self.inner.elevation_rad
        )
    }
}

/// SBAS protection-level geometry and clock-column convention.
#[pyclass(module = "sidereon._sidereon", name = "ProtectionGeometry")]
#[derive(Clone)]
pub struct PyProtectionGeometry {
    inner: ProtectionGeometry,
}

#[pymethods]
impl PyProtectionGeometry {
    /// Build SBAS protection geometry from rows, receiver, and clock systems.
    #[new]
    fn new(
        py: Python<'_>,
        rows: Vec<Py<PyProtectionRow>>,
        receiver: PyRef<'_, PyWgs84Geodetic>,
        clock_systems: Vec<PyGnssSystem>,
    ) -> PyResult<Self> {
        let rows = rows
            .iter()
            .map(|row| row.borrow(py).inner)
            .collect::<Vec<_>>();
        Ok(Self {
            inner: ProtectionGeometry {
                rows,
                receiver: Wgs84Geodetic::try_from(&*receiver)?,
                clock_systems: clock_systems.into_iter().map(Into::into).collect(),
            },
        })
    }

    /// Protection rows in input order.
    #[getter]
    fn rows(&self) -> Vec<PyProtectionRow> {
        self.inner.rows.iter().copied().map(Into::into).collect()
    }

    /// Receiver geodetic position for local ENU projection.
    #[getter]
    fn receiver(&self) -> PyWgs84Geodetic {
        PyWgs84Geodetic::from_core(self.inner.receiver)
    }

    /// Receiver-clock columns in solved-state order.
    #[getter]
    fn clock_systems(&self) -> Vec<PyGnssSystem> {
        self.inner
            .clock_systems
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// Return a compact representation of the protection geometry.
    fn __repr__(&self) -> String {
        format!("ProtectionGeometry(rows={})", self.inner.rows.len())
    }
}

/// Fixed SBAS protection-level multipliers.
#[pyclass(module = "sidereon._sidereon", name = "SbasKMultipliers")]
#[derive(Clone, Copy)]
pub struct PySbasKMultipliers {
    inner: SbasKMultipliers,
}

impl PySbasKMultipliers {
    fn inner_or_default(value: Option<&Self>) -> SbasKMultipliers {
        value
            .map(|multipliers| multipliers.inner)
            .unwrap_or(SbasKMultipliers::PRECISION_APPROACH)
    }
}

#[pymethods]
impl PySbasKMultipliers {
    /// Build SBAS K multipliers.
    ///
    /// Omitted values use the precision-approach constants.
    #[new]
    #[pyo3(signature = (k_h=None, k_v=None))]
    fn new(k_h: Option<f64>, k_v: Option<f64>) -> Self {
        let defaults = SbasKMultipliers::PRECISION_APPROACH;
        Self {
            inner: SbasKMultipliers {
                k_h: k_h.unwrap_or(defaults.k_h),
                k_v: k_v.unwrap_or(defaults.k_v),
            },
        }
    }

    /// Precision-approach SBAS multipliers.
    #[staticmethod]
    fn precision_approach() -> Self {
        Self {
            inner: SbasKMultipliers::PRECISION_APPROACH,
        }
    }

    /// En-route through non-precision-approach SBAS multipliers.
    #[staticmethod]
    fn en_route_npa() -> Self {
        Self {
            inner: SbasKMultipliers::EN_ROUTE_NPA,
        }
    }

    /// Horizontal multiplier.
    #[getter]
    fn k_h(&self) -> f64 {
        self.inner.k_h
    }

    /// Vertical multiplier.
    #[getter]
    fn k_v(&self) -> f64 {
        self.inner.k_v
    }

    /// Return a compact representation of the multipliers.
    fn __repr__(&self) -> String {
        format!(
            "SbasKMultipliers(k_h={}, k_v={})",
            self.inner.k_h, self.inner.k_v
        )
    }
}

/// One satellite's SBAS one-sigma range-error budget.
#[pyclass(module = "sidereon._sidereon", name = "SbasSisError")]
#[derive(Clone, Copy)]
pub struct PySbasSisError {
    inner: SbasSisError,
}

impl From<SbasSisError> for PySbasSisError {
    fn from(inner: SbasSisError) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySbasSisError {
    /// Build one SBAS range-error row.
    #[new]
    #[pyo3(signature = (
        satellite_id,
        sigma_flt_m,
        sigma_uire_m=0.0,
        sigma_air_m=0.0,
        sigma_tropo_m=0.0
    ))]
    fn new(
        satellite_id: &str,
        sigma_flt_m: f64,
        sigma_uire_m: f64,
        sigma_air_m: f64,
        sigma_tropo_m: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SbasSisError {
                id: parse_satellite(satellite_id)?,
                sigma_flt_m,
                sigma_uire_m,
                sigma_air_m,
                sigma_tropo_m,
            },
        })
    }

    /// Satellite token matching a protection geometry row.
    #[getter]
    fn satellite_id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Fast and long-term correction residual sigma, metres.
    #[getter]
    fn sigma_flt_m(&self) -> f64 {
        self.inner.sigma_flt_m
    }

    /// User ionospheric range-error sigma, metres.
    #[getter]
    fn sigma_uire_m(&self) -> f64 {
        self.inner.sigma_uire_m
    }

    /// Airborne receiver noise, divergence, and multipath sigma, metres.
    #[getter]
    fn sigma_air_m(&self) -> f64 {
        self.inner.sigma_air_m
    }

    /// Tropospheric residual sigma, metres.
    #[getter]
    fn sigma_tropo_m(&self) -> f64 {
        self.inner.sigma_tropo_m
    }

    /// Sum-of-squares range variance in square metres, if valid.
    fn variance_m2(&self) -> Option<f64> {
        self.inner.variance_m2()
    }

    /// Total one-sigma range error in metres, if valid.
    fn sigma_m(&self) -> Option<f64> {
        self.inner.sigma_m()
    }

    /// Return a compact representation of the SIS error row.
    fn __repr__(&self) -> String {
        format!(
            "SbasSisError(satellite_id={:?}, sigma_flt_m={})",
            self.inner.id.to_string(),
            self.inner.sigma_flt_m
        )
    }
}

/// Index-aligned SBAS error model for protection geometry rows.
#[pyclass(module = "sidereon._sidereon", name = "SbasErrorModel")]
#[derive(Clone)]
pub struct PySbasErrorModel {
    inner: SbasErrorModel,
}

#[pymethods]
impl PySbasErrorModel {
    /// Build an SBAS error model from supplied per-satellite rows.
    #[new]
    fn new(py: Python<'_>, rows: Vec<Py<PySbasSisError>>) -> Self {
        Self {
            inner: SbasErrorModel::new(rows.iter().map(|row| row.borrow(py).inner).collect()),
        }
    }

    /// Build an SBAS error model from decoded SBAS correction storage.
    #[staticmethod]
    #[pyo3(signature = (
        store,
        geo_satellite_id,
        geometry,
        epoch_j2000_s,
        airborne=None,
        degradation=None
    ))]
    fn from_store(
        store: &PySbasCorrectionStore,
        geo_satellite_id: &str,
        geometry: &PyProtectionGeometry,
        epoch_j2000_s: f64,
        airborne: Option<&PyAirborneModel>,
        degradation: Option<&PyDegradationParams>,
    ) -> PyResult<Self> {
        let airborne = airborne.map(|model| model.inner).unwrap_or_default();
        let degradation = degradation.map(|params| params.inner).unwrap_or_default();
        let geo = parse_satellite(geo_satellite_id)?;
        SbasErrorModel::from_store(
            &store.inner,
            geo,
            &geometry.inner,
            &airborne,
            epoch_j2000_s,
            &degradation,
        )
        .map(|inner| Self { inner })
        .map_err(to_sbas_pl_err)
    }

    /// Per-satellite range-error rows.
    #[getter]
    fn rows(&self) -> Vec<PySbasSisError> {
        self.inner.rows.iter().copied().map(Into::into).collect()
    }

    /// Return the range-error row for one satellite token.
    fn row_for(&self, satellite_id: &str) -> PyResult<Option<PySbasSisError>> {
        let id = parse_satellite(satellite_id)?;
        Ok(self.inner.row_for(id).copied().map(Into::into))
    }

    /// Return a compact representation of the SBAS error model.
    fn __repr__(&self) -> String {
        format!("SbasErrorModel(rows={})", self.inner.rows.len())
    }
}

/// Airborne receiver and multipath contribution model.
#[pyclass(module = "sidereon._sidereon", name = "AirborneModel")]
#[derive(Clone, Copy)]
pub struct PyAirborneModel {
    inner: AirborneModel,
}

#[pymethods]
impl PyAirborneModel {
    /// Build an airborne model from receiver noise and divergence sigma.
    #[new]
    #[pyo3(signature = (sigma_noise_divergence_m=0.36))]
    fn new(sigma_noise_divergence_m: f64) -> Self {
        Self {
            inner: AirborneModel::new(sigma_noise_divergence_m),
        }
    }

    /// Default airborne model.
    #[staticmethod]
    fn aad_a() -> Self {
        Self {
            inner: AirborneModel::aad_a(),
        }
    }

    /// Receiver noise and code-carrier divergence sigma, metres.
    #[getter]
    fn sigma_noise_divergence_m(&self) -> f64 {
        self.inner.sigma_noise_divergence_m
    }

    /// Airborne receiver, divergence, and multipath sigma at elevation.
    fn sigma_air_m(&self, elevation_rad: f64) -> Option<f64> {
        self.inner.sigma_air_m(elevation_rad)
    }

    /// Return a compact representation of the airborne model.
    fn __repr__(&self) -> String {
        format!(
            "AirborneModel(sigma_noise_divergence_m={})",
            self.inner.sigma_noise_divergence_m
        )
    }
}

/// Supplied SBAS degradation terms.
#[pyclass(module = "sidereon._sidereon", name = "DegradationParams")]
#[derive(Clone, Copy)]
pub struct PyDegradationParams {
    inner: DegradationParams,
}

#[pymethods]
impl PyDegradationParams {
    /// Build SBAS degradation parameters.
    #[new]
    #[pyo3(signature = (
        delta_udre=1.0,
        eps_fc_m=0.0,
        eps_rrc_m=0.0,
        eps_ltc_m=0.0,
        eps_er_m=0.0,
        eps_iono_m=0.0,
        rss_udre=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        delta_udre: f64,
        eps_fc_m: f64,
        eps_rrc_m: f64,
        eps_ltc_m: f64,
        eps_er_m: f64,
        eps_iono_m: f64,
        rss_udre: bool,
    ) -> Self {
        Self {
            inner: DegradationParams {
                delta_udre,
                eps_fc_m,
                eps_rrc_m,
                eps_ltc_m,
                eps_er_m,
                eps_iono_m,
                rss_udre,
            },
        }
    }

    /// No extra degradation and no UDRE inflation.
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: DegradationParams::none(),
        }
    }

    /// Variance multiplier applied to the UDRE variance table.
    #[getter]
    fn delta_udre(&self) -> f64 {
        self.inner.delta_udre
    }

    /// Fast-correction degradation term, metres.
    #[getter]
    fn eps_fc_m(&self) -> f64 {
        self.inner.eps_fc_m
    }

    /// Range-rate-correction degradation term, metres.
    #[getter]
    fn eps_rrc_m(&self) -> f64 {
        self.inner.eps_rrc_m
    }

    /// Long-term-correction degradation term, metres.
    #[getter]
    fn eps_ltc_m(&self) -> f64 {
        self.inner.eps_ltc_m
    }

    /// En-route degradation term, metres.
    #[getter]
    fn eps_er_m(&self) -> f64 {
        self.inner.eps_er_m
    }

    /// Ionospheric degradation term added to UIRE, metres.
    #[getter]
    fn eps_iono_m(&self) -> f64 {
        self.inner.eps_iono_m
    }

    /// True when UDRE degradation terms are combined by root-sum-square.
    #[getter]
    fn rss_udre(&self) -> bool {
        self.inner.rss_udre
    }

    /// True when every supplied degradation term is valid.
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Return a compact representation of the degradation parameters.
    fn __repr__(&self) -> String {
        format!(
            "DegradationParams(delta_udre={}, rss_udre={})",
            self.inner.delta_udre, self.inner.rss_udre
        )
    }
}

/// SBAS protection-level output for one geometry snapshot.
#[pyclass(module = "sidereon._sidereon", name = "SbasProtection")]
#[derive(Clone, Copy)]
pub struct PySbasProtection {
    inner: SbasProtection,
}

impl From<SbasProtection> for PySbasProtection {
    fn from(inner: SbasProtection) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySbasProtection {
    /// Horizontal protection level, metres.
    #[getter]
    fn hpl_m(&self) -> f64 {
        self.inner.hpl_m
    }

    /// Vertical protection level, metres.
    #[getter]
    fn vpl_m(&self) -> f64 {
        self.inner.vpl_m
    }

    /// Horizontal one-sigma semi-major axis, metres.
    #[getter]
    fn d_major_m(&self) -> f64 {
        self.inner.d_major_m
    }

    /// Vertical one-sigma standard deviation, metres.
    #[getter]
    fn sigma_u_m(&self) -> f64 {
        self.inner.sigma_u_m
    }

    /// East one-sigma standard deviation, metres.
    #[getter]
    fn d_east_m(&self) -> f64 {
        self.inner.d_east_m
    }

    /// North one-sigma standard deviation, metres.
    #[getter]
    fn d_north_m(&self) -> f64 {
        self.inner.d_north_m
    }

    /// East-north covariance term, square metres.
    #[getter]
    fn d_en_m2(&self) -> f64 {
        self.inner.d_en_m2
    }

    /// Return a compact representation of the protection levels.
    fn __repr__(&self) -> String {
        format!(
            "SbasProtection(hpl_m={}, vpl_m={})",
            self.inner.hpl_m, self.inner.vpl_m
        )
    }
}

/// Compute SBAS horizontal and vertical protection levels.
#[pyfunction]
#[pyo3(signature = (geometry, model, k=None))]
fn sbas_protection_levels(
    geometry: &PyProtectionGeometry,
    model: &PySbasErrorModel,
    k: Option<&PySbasKMultipliers>,
) -> PyResult<PySbasProtection> {
    let k = PySbasKMultipliers::inner_or_default(k);
    core_sbas_protection_levels(&geometry.inner, &model.inner, k)
        .map(Into::into)
        .map_err(to_sbas_pl_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySbasPlError>()?;
    m.add_class::<PyProtectionRow>()?;
    m.add_class::<PyProtectionGeometry>()?;
    m.add_class::<PySbasKMultipliers>()?;
    m.add_class::<PySbasProtection>()?;
    m.add_class::<PySbasErrorModel>()?;
    m.add_class::<PyAirborneModel>()?;
    m.add_class::<PyDegradationParams>()?;
    m.add_class::<PySbasSisError>()?;
    m.add_function(wrap_pyfunction!(sbas_protection_levels, m)?)?;
    Ok(())
}
