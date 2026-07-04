//! ARAIM integrity bindings.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::araim::{
    araim as core_araim, enumerate_fault_modes as core_enumerate_fault_modes, AraimGeometry,
    AraimResult, AraimRow, ConstellationIsm, FaultHypothesis, FaultMode, IntegrityAllocation, Ism,
    SatelliteIsm, SatelliteIsmModel,
};
use sidereon_core::geometry::LineOfSight;
use sidereon_core::{GnssSatelliteId, GnssSystem, Wgs84Geodetic};

use crate::events::PyWgs84Geodetic;
use crate::marshal::{fixed_array, PyGnssSystem};
use crate::np_array;

fn to_araim_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite_id {token:?}: {err}")))
}

fn satellite_tokens(ids: &[GnssSatelliteId]) -> Vec<String> {
    ids.iter().map(ToString::to_string).collect()
}

fn system_from_satellite(
    satellite_id: GnssSatelliteId,
    system: Option<PyGnssSystem>,
) -> GnssSystem {
    system.map(Into::into).unwrap_or(satellite_id.system)
}

/// One satellite row in an ARAIM geometry snapshot.
#[pyclass(module = "sidereon._sidereon", name = "AraimRow")]
#[derive(Clone)]
pub struct PyAraimRow {
    inner: AraimRow,
}

impl From<AraimRow> for PyAraimRow {
    fn from(inner: AraimRow) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAraimRow {
    /// Build one ARAIM geometry row.
    ///
    /// `satellite_id` is a token such as `"G01"`. `line_of_sight_ecef` is a
    /// receiver-to-satellite ECEF unit vector with three components.
    /// `elevation_rad` is the receiver elevation angle in radians. `system`
    /// defaults to the constellation encoded in `satellite_id`.
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
            crate::marshal::FinitePolicy::RequireFinite,
        )?;
        Ok(Self {
            inner: AraimRow {
                id,
                line_of_sight: LineOfSight::new(los[0], los[1], los[2]),
                system: system_from_satellite(id, system),
                elevation_rad,
            },
        })
    }

    /// Satellite token used for ISM lookup and satellite-fault modes.
    #[getter]
    fn satellite_id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Receiver-to-satellite ECEF unit vector as a numpy `(3,)` array.
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

    /// Constellation owning the signal and constellation-fault mode.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    /// Elevation angle at the receiver, radians.
    #[getter]
    fn elevation_rad(&self) -> f64 {
        self.inner.elevation_rad
    }

    fn __repr__(&self) -> String {
        format!(
            "AraimRow(satellite_id={:?}, system={}, elevation_rad={})",
            self.inner.id.to_string(),
            self.inner.system.as_str(),
            self.inner.elevation_rad
        )
    }
}

/// A snapshot geometry and clock-column convention for ARAIM.
#[pyclass(module = "sidereon._sidereon", name = "AraimGeometry")]
#[derive(Clone)]
pub struct PyAraimGeometry {
    inner: AraimGeometry,
}

#[pymethods]
impl PyAraimGeometry {
    /// Build ARAIM geometry from rows, a receiver, and clock systems.
    ///
    /// `rows` are satellite line-of-sight rows. `receiver` is WGS84 geodetic in
    /// radians and metres. `clock_systems` gives receiver-clock columns in the
    /// same order as the solved state.
    #[new]
    fn new(
        py: Python<'_>,
        rows: Vec<Py<PyAraimRow>>,
        receiver: PyRef<'_, PyWgs84Geodetic>,
        clock_systems: Vec<PyGnssSystem>,
    ) -> PyResult<Self> {
        let rows = rows
            .iter()
            .map(|row| row.borrow(py).inner)
            .collect::<Vec<_>>();
        Ok(Self {
            inner: AraimGeometry {
                rows,
                receiver: Wgs84Geodetic::try_from(&*receiver)?,
                clock_systems: clock_systems.into_iter().map(Into::into).collect(),
            },
        })
    }

    /// Satellite rows, index-aligned through all gain matrices.
    #[getter]
    fn rows(&self) -> Vec<PyAraimRow> {
        self.inner.rows.iter().copied().map(Into::into).collect()
    }

    /// Receiver geodetic position for ENU rotation.
    #[getter]
    fn receiver(&self) -> PyWgs84Geodetic {
        PyWgs84Geodetic::from_core(self.inner.receiver)
    }

    /// Receiver-clock columns in the same order as the solved state.
    #[getter]
    fn clock_systems(&self) -> Vec<PyGnssSystem> {
        self.inner
            .clock_systems
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("AraimGeometry(rows={})", self.inner.rows.len())
    }
}

/// Integrity and continuity risk allocation for one ARAIM solve.
#[pyclass(module = "sidereon._sidereon", name = "IntegrityAllocation")]
#[derive(Clone, Copy)]
pub struct PyIntegrityAllocation {
    inner: IntegrityAllocation,
}

#[pymethods]
impl PyIntegrityAllocation {
    /// Build an ARAIM integrity allocation.
    #[new]
    #[pyo3(signature = (
        phmi_total,
        phmi_vert,
        phmi_hor,
        pfa_vert,
        pfa_hor,
        p_threshold_unmonitored,
        max_fault_order,
        p_emt=1.0e-5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        phmi_total: f64,
        phmi_vert: f64,
        phmi_hor: f64,
        pfa_vert: f64,
        pfa_hor: f64,
        p_threshold_unmonitored: f64,
        max_fault_order: usize,
        p_emt: f64,
    ) -> Self {
        Self {
            inner: IntegrityAllocation {
                phmi_total,
                phmi_vert,
                phmi_hor,
                pfa_vert,
                pfa_hor,
                p_threshold_unmonitored,
                p_emt,
                max_fault_order,
            },
        }
    }

    /// LPV-200 allocation from the public ARAIM reference material.
    #[staticmethod]
    fn lpv_200() -> Self {
        Self {
            inner: IntegrityAllocation::lpv_200(),
        }
    }

    /// Total probability of hazardous misleading information.
    #[getter]
    fn phmi_total(&self) -> f64 {
        self.inner.phmi_total
    }

    /// Vertical PHMI allocation.
    #[getter]
    fn phmi_vert(&self) -> f64 {
        self.inner.phmi_vert
    }

    /// Horizontal PHMI allocation.
    #[getter]
    fn phmi_hor(&self) -> f64 {
        self.inner.phmi_hor
    }

    /// Vertical false-alert allocation.
    #[getter]
    fn pfa_vert(&self) -> f64 {
        self.inner.pfa_vert
    }

    /// Horizontal false-alert allocation.
    #[getter]
    fn pfa_hor(&self) -> f64 {
        self.inner.pfa_hor
    }

    /// Maximum acceptable unmonitored fault probability mass.
    #[getter]
    fn p_threshold_unmonitored(&self) -> f64 {
        self.inner.p_threshold_unmonitored
    }

    /// Fault-prior threshold used for the effective monitor threshold.
    #[getter]
    fn p_emt(&self) -> f64 {
        self.inner.p_emt
    }

    /// Maximum enumerated satellite-fault order.
    #[getter]
    fn max_fault_order(&self) -> usize {
        self.inner.max_fault_order
    }

    fn __repr__(&self) -> String {
        format!(
            "IntegrityAllocation(phmi_total={}, max_fault_order={})",
            self.inner.phmi_total, self.inner.max_fault_order
        )
    }
}

/// Per-satellite integrity and accuracy model without an identity.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteIsmModel")]
#[derive(Clone, Copy)]
pub struct PySatelliteIsmModel {
    inner: SatelliteIsmModel,
}

#[pymethods]
impl PySatelliteIsmModel {
    /// Build a per-satellite ISM model.
    #[new]
    #[pyo3(signature = (
        sigma_ura_m,
        sigma_ure_m,
        b_nom_m,
        p_sat,
        effective_sigma_int_m=None,
        effective_sigma_acc_m=None
    ))]
    fn new(
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
        effective_sigma_int_m: Option<f64>,
        effective_sigma_acc_m: Option<f64>,
    ) -> Self {
        Self {
            inner: SatelliteIsmModel {
                sigma_ura_m,
                sigma_ure_m,
                effective_sigma_int_m,
                effective_sigma_acc_m,
                b_nom_m,
                p_sat,
            },
        }
    }

    /// Build a per-satellite ISM model with direct effective range sigmas.
    #[staticmethod]
    fn new_with_effective_sigmas(
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
        effective_sigma_int_m: f64,
        effective_sigma_acc_m: f64,
    ) -> Self {
        Self {
            inner: SatelliteIsmModel::new_with_effective_sigmas(
                sigma_ura_m,
                sigma_ure_m,
                b_nom_m,
                p_sat,
                effective_sigma_int_m,
                effective_sigma_acc_m,
            ),
        }
    }

    /// Integrity one-sigma SIS range error, metres.
    #[getter]
    fn sigma_ura_m(&self) -> f64 {
        self.inner.sigma_ura_m
    }

    /// Accuracy and continuity one-sigma SIS range error, metres.
    #[getter]
    fn sigma_ure_m(&self) -> f64 {
        self.inner.sigma_ure_m
    }

    /// Effective integrity one-sigma range error after local terms, metres.
    #[getter]
    fn effective_sigma_int_m(&self) -> Option<f64> {
        self.inner.effective_sigma_int_m
    }

    /// Effective accuracy one-sigma range error after local terms, metres.
    #[getter]
    fn effective_sigma_acc_m(&self) -> Option<f64> {
        self.inner.effective_sigma_acc_m
    }

    /// Nominal SIS bias bound, metres.
    #[getter]
    fn b_nom_m(&self) -> f64 {
        self.inner.b_nom_m
    }

    /// Prior probability for a satellite fault.
    #[getter]
    fn p_sat(&self) -> f64 {
        self.inner.p_sat
    }
}

/// Per-satellite ISM override.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteIsm")]
#[derive(Clone, Copy)]
pub struct PySatelliteIsm {
    inner: SatelliteIsm,
}

#[pymethods]
impl PySatelliteIsm {
    /// Build a satellite-specific ISM model.
    #[new]
    #[pyo3(signature = (
        satellite_id,
        sigma_ura_m,
        sigma_ure_m,
        b_nom_m,
        p_sat,
        effective_sigma_int_m=None,
        effective_sigma_acc_m=None
    ))]
    fn new(
        satellite_id: &str,
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
        effective_sigma_int_m: Option<f64>,
        effective_sigma_acc_m: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SatelliteIsm {
                id: parse_satellite(satellite_id)?,
                sigma_ura_m,
                sigma_ure_m,
                effective_sigma_int_m,
                effective_sigma_acc_m,
                b_nom_m,
                p_sat,
            },
        })
    }

    /// Build a satellite-specific ISM model with direct effective range sigmas.
    #[staticmethod]
    fn new_with_effective_sigmas(
        satellite_id: &str,
        sigma_ura_m: f64,
        sigma_ure_m: f64,
        b_nom_m: f64,
        p_sat: f64,
        effective_sigma_int_m: f64,
        effective_sigma_acc_m: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SatelliteIsm::new_with_effective_sigmas(
                parse_satellite(satellite_id)?,
                sigma_ura_m,
                sigma_ure_m,
                b_nom_m,
                p_sat,
                effective_sigma_int_m,
                effective_sigma_acc_m,
            ),
        })
    }

    /// Satellite token for this override.
    #[getter]
    fn satellite_id(&self) -> String {
        self.inner.id.to_string()
    }

    /// Integrity one-sigma SIS range error, metres.
    #[getter]
    fn sigma_ura_m(&self) -> f64 {
        self.inner.sigma_ura_m
    }

    /// Accuracy and continuity one-sigma SIS range error, metres.
    #[getter]
    fn sigma_ure_m(&self) -> f64 {
        self.inner.sigma_ure_m
    }

    /// Effective integrity one-sigma range error after local terms, metres.
    #[getter]
    fn effective_sigma_int_m(&self) -> Option<f64> {
        self.inner.effective_sigma_int_m
    }

    /// Effective accuracy one-sigma range error after local terms, metres.
    #[getter]
    fn effective_sigma_acc_m(&self) -> Option<f64> {
        self.inner.effective_sigma_acc_m
    }

    /// Nominal SIS bias bound, metres.
    #[getter]
    fn b_nom_m(&self) -> f64 {
        self.inner.b_nom_m
    }

    /// Prior probability for a satellite fault.
    #[getter]
    fn p_sat(&self) -> f64 {
        self.inner.p_sat
    }
}

/// Per-constellation fault prior and default satellite model.
#[pyclass(module = "sidereon._sidereon", name = "ConstellationIsm")]
#[derive(Clone, Copy)]
pub struct PyConstellationIsm {
    inner: ConstellationIsm,
}

#[pymethods]
impl PyConstellationIsm {
    /// Build a per-constellation ISM model.
    #[new]
    fn new(system: PyGnssSystem, p_const: f64, default_sat: &PySatelliteIsmModel) -> Self {
        Self {
            inner: ConstellationIsm::new(system.into(), p_const, default_sat.inner),
        }
    }

    /// Constellation identity.
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    /// Prior probability for a constellation-wide fault.
    #[getter]
    fn p_const(&self) -> f64 {
        self.inner.p_const
    }

    /// Default satellite model for rows in this constellation.
    #[getter]
    fn default_sat(&self) -> PySatelliteIsmModel {
        PySatelliteIsmModel {
            inner: self.inner.default_sat,
        }
    }
}

/// Parsed integrity support message used by ARAIM.
#[pyclass(module = "sidereon._sidereon", name = "Ism")]
#[derive(Clone)]
pub struct PyIsm {
    inner: Ism,
}

#[pymethods]
impl PyIsm {
    /// Build an ISM from constellation defaults and satellite overrides.
    #[new]
    fn new(
        py: Python<'_>,
        constellations: Vec<Py<PyConstellationIsm>>,
        satellites: Vec<Py<PySatelliteIsm>>,
    ) -> Self {
        Self {
            inner: Ism::new(
                constellations
                    .iter()
                    .map(|item| item.borrow(py).inner)
                    .collect(),
                satellites
                    .iter()
                    .map(|item| item.borrow(py).inner)
                    .collect(),
            ),
        }
    }

    /// Per-constellation defaults and constellation-wide fault priors.
    #[getter]
    fn constellations(&self) -> Vec<PyConstellationIsm> {
        self.inner
            .constellations
            .iter()
            .copied()
            .map(|inner| PyConstellationIsm { inner })
            .collect()
    }

    /// Per-satellite overrides.
    #[getter]
    fn satellites(&self) -> Vec<PySatelliteIsm> {
        self.inner
            .satellites
            .iter()
            .copied()
            .map(|inner| PySatelliteIsm { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Ism(constellations={}, satellites={})",
            self.inner.constellations.len(),
            self.inner.satellites.len()
        )
    }
}

/// One ARAIM fault hypothesis.
#[pyclass(module = "sidereon._sidereon", name = "FaultHypothesis")]
#[derive(Clone)]
pub struct PyFaultHypothesis {
    inner: FaultHypothesis,
}

impl From<FaultHypothesis> for PyFaultHypothesis {
    fn from(inner: FaultHypothesis) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFaultHypothesis {
    /// Satellites excluded by this hypothesis.
    #[getter]
    fn excluded(&self) -> Vec<String> {
        satellite_tokens(&self.inner.excluded)
    }

    /// Constellation excluded by this hypothesis, if any.
    #[getter]
    fn excluded_constellation(&self) -> Option<PyGnssSystem> {
        self.inner.excluded_constellation.map(Into::into)
    }

    /// Prior probability mass for this hypothesis.
    #[getter]
    fn prior(&self) -> f64 {
        self.inner.prior
    }
}

/// Per-hypothesis ARAIM monitor data.
#[pyclass(module = "sidereon._sidereon", name = "FaultMode")]
#[derive(Clone)]
pub struct PyFaultMode {
    inner: FaultMode,
}

impl From<FaultMode> for PyFaultMode {
    fn from(inner: FaultMode) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFaultMode {
    /// Satellites excluded by this mode.
    #[getter]
    fn excluded(&self) -> Vec<String> {
        satellite_tokens(&self.inner.excluded)
    }

    /// Constellation excluded by this mode, if any.
    #[getter]
    fn excluded_constellation(&self) -> Option<PyGnssSystem> {
        self.inner.excluded_constellation.map(Into::into)
    }

    /// Fault prior probability for this mode.
    #[getter]
    fn prior(&self) -> f64 {
        self.inner.prior
    }

    /// Integrity sigma in local `[east, north, up]`, metres.
    #[getter]
    fn sigma_int_enu_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.sigma_int_enu_m)
    }

    /// Nominal bias bound in local `[east, north, up]`, metres.
    #[getter]
    fn bias_enu_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.bias_enu_m)
    }

    /// Separation monitor threshold in local `[east, north, up]`, metres.
    #[getter]
    fn threshold_enu_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.threshold_enu_m)
    }

    /// True when the subset geometry is full-rank.
    #[getter]
    fn monitorable(&self) -> bool {
        self.inner.monitorable
    }
}

/// ARAIM protection-level result.
#[pyclass(module = "sidereon._sidereon", name = "AraimResult")]
#[derive(Clone)]
pub struct PyAraimResult {
    inner: AraimResult,
}

impl From<AraimResult> for PyAraimResult {
    fn from(inner: AraimResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAraimResult {
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

    /// All-in-view horizontal accuracy sigma, metres.
    #[getter]
    fn sigma_acc_h_m(&self) -> f64 {
        self.inner.sigma_acc_h_m
    }

    /// All-in-view vertical accuracy sigma, metres.
    #[getter]
    fn sigma_acc_v_m(&self) -> f64 {
        self.inner.sigma_acc_v_m
    }

    /// Effective monitor threshold, metres.
    #[getter]
    fn emt_m(&self) -> f64 {
        self.inner.emt_m
    }

    /// Per-mode monitor data, including the fault-free mode first.
    #[getter]
    fn fault_modes(&self) -> Vec<PyFaultMode> {
        self.inner
            .fault_modes
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Unenumerated plus unmonitorable fault probability mass.
    #[getter]
    fn p_unmonitored(&self) -> f64 {
        self.inner.p_unmonitored
    }

    /// True when the solve met the allocation and all PL roots converged.
    #[getter]
    fn availability(&self) -> bool {
        self.inner.availability
    }

    fn __repr__(&self) -> String {
        format!(
            "AraimResult(hpl_m={}, vpl_m={}, availability={})",
            self.inner.hpl_m, self.inner.vpl_m, self.inner.availability
        )
    }
}

/// Run the ARAIM MHSS protection-level algorithm.
#[pyfunction]
fn araim(
    geometry: &PyAraimGeometry,
    ism: &PyIsm,
    allocation: &PyIntegrityAllocation,
) -> PyResult<PyAraimResult> {
    core_araim(&geometry.inner, &ism.inner, &allocation.inner)
        .map(Into::into)
        .map_err(to_araim_err)
}

/// Enumerate ARAIM fault modes in the order used by the MHSS solve.
#[pyfunction]
fn enumerate_fault_modes(
    geometry: &PyAraimGeometry,
    ism: &PyIsm,
    allocation: &PyIntegrityAllocation,
) -> PyResult<Vec<PyFaultHypothesis>> {
    Ok(
        core_enumerate_fault_modes(&geometry.inner, &ism.inner, &allocation.inner)
            .map_err(to_araim_err)?
            .into_iter()
            .map(Into::into)
            .collect(),
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAraimRow>()?;
    m.add_class::<PyAraimGeometry>()?;
    m.add_class::<PyIntegrityAllocation>()?;
    m.add_class::<PySatelliteIsmModel>()?;
    m.add_class::<PySatelliteIsm>()?;
    m.add_class::<PyConstellationIsm>()?;
    m.add_class::<PyIsm>()?;
    m.add_class::<PyFaultHypothesis>()?;
    m.add_class::<PyFaultMode>()?;
    m.add_class::<PyAraimResult>()?;
    m.add_function(wrap_pyfunction!(araim, m)?)?;
    m.add_function(wrap_pyfunction!(enumerate_fault_modes, m)?)?;
    Ok(())
}
