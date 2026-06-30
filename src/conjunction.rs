//! Conjunction and covariance binding.
//!
//! Marshals numpy vectors/matrices into the core conjunction and RTN covariance
//! routines, then returns numpy arrays and typed result objects. This module is a
//! pure interface: collision probability, encounter-frame geometry, B-plane
//! covariance projection, PSD checks, and RTN->ECI covariance rotation are all
//! computed by `sidereon-core`.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::conjunction::{
    collision_probability as core_collision_probability, encounter_frame as core_encounter_frame,
    encounter_plane_covariance as core_encounter_plane_covariance, CollisionPc, ConjunctionError,
    ConjunctionState, EncounterFrame, PcMethod,
};
use sidereon_core::astro::covariance::{
    positive_semidefinite, rtn_to_eci, symmetric, RtnFrameError,
};

use crate::marshal::{fixed_array, mat3_to_array, matrix3_from_array, FinitePolicy};
use crate::{np_array, to_solve_err};

fn mat2_to_array<'py>(py: Python<'py>, values: &[[f64; 2]; 2]) -> Bound<'py, PyArray2<f64>> {
    let mut array = Array2::<f64>::zeros((2, 2));
    for i in 0..2 {
        for j in 0..2 {
            array[[i, j]] = values[i][j];
        }
    }
    PyArray2::from_owned_array(py, array)
}

fn rtn_frame_err(err: RtnFrameError) -> PyErr {
    PyValueError::new_err(err.message())
}

fn conjunction_err(err: ConjunctionError) -> PyErr {
    match err {
        ConjunctionError::UndefinedFrame => to_solve_err(err),
        ConjunctionError::NonFinite { .. } | ConjunctionError::NotPositive { .. } => {
            PyValueError::new_err(err.to_string())
        }
    }
}

/// Collision-probability integration method.
#[pyclass(module = "sidereon._sidereon", name = "PcMethod", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyPcMethod {
    /// Foster 2D with the equal-area square approximation.
    FOSTER_EQUAL_AREA,
    /// Foster 2D with polar-grid numerical integration.
    FOSTER_NUMERICAL,
    /// Alfano (2005) 1D Simpson integration.
    ALFANO_2005,
}

impl From<PyPcMethod> for PcMethod {
    fn from(method: PyPcMethod) -> Self {
        match method {
            PyPcMethod::FOSTER_EQUAL_AREA => PcMethod::FosterEqualArea,
            PyPcMethod::FOSTER_NUMERICAL => PcMethod::FosterNumerical,
            PyPcMethod::ALFANO_2005 => PcMethod::Alfano2005,
        }
    }
}

#[pymethods]
impl PyPcMethod {
    /// Stable lowercase identifier for this method.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            PyPcMethod::FOSTER_EQUAL_AREA => "foster_equal_area",
            PyPcMethod::FOSTER_NUMERICAL => "foster_numerical",
            PyPcMethod::ALFANO_2005 => "alfano_2005",
        }
    }

    fn __repr__(&self) -> String {
        match self {
            PyPcMethod::FOSTER_EQUAL_AREA => "PcMethod.FOSTER_EQUAL_AREA".to_string(),
            PyPcMethod::FOSTER_NUMERICAL => "PcMethod.FOSTER_NUMERICAL".to_string(),
            PyPcMethod::ALFANO_2005 => "PcMethod.ALFANO_2005".to_string(),
        }
    }
}

/// One object's conjunction state: ECI position (km), velocity (km/s), and 3x3
/// position covariance (km^2).
#[pyclass(module = "sidereon._sidereon", name = "ConjunctionState")]
#[derive(Clone, PartialEq)]
pub struct PyConjunctionState {
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
    covariance_km2: [[f64; 3]; 3],
}

impl PyConjunctionState {
    fn core(&self) -> ConjunctionState {
        ConjunctionState {
            position_km: self.position_km,
            velocity_km_s: self.velocity_km_s,
            covariance_km2: self.covariance_km2,
        }
    }
}

#[pymethods]
impl PyConjunctionState {
    #[new]
    fn new(
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
        covariance_km2: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            position_km: fixed_array::<3>(
                "position_km",
                &position_km,
                FinitePolicy::AllowNonFinite,
            )?,
            velocity_km_s: fixed_array::<3>(
                "velocity_km_s",
                &velocity_km_s,
                FinitePolicy::AllowNonFinite,
            )?,
            covariance_km2: matrix3_from_array(
                &covariance_km2,
                "covariance_km2",
                FinitePolicy::AllowNonFinite,
            )?,
        })
    }

    /// ECI position as a numpy `(3,)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.position_km)
    }

    /// ECI velocity as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.velocity_km_s)
    }

    /// Position covariance as a numpy `(3, 3)` array, km^2.
    #[getter]
    fn covariance_km2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.covariance_km2)
    }

    fn __repr__(&self) -> String {
        format!(
            "ConjunctionState(position_km=[{}, {}, {}], velocity_km_s=[{}, {}, {}])",
            self.position_km[0],
            self.position_km[1],
            self.position_km[2],
            self.velocity_km_s[0],
            self.velocity_km_s[1],
            self.velocity_km_s[2]
        )
    }

    fn __eq__(&self, other: &PyConjunctionState) -> bool {
        self == other
    }
}

/// Orthonormal encounter frame built from two relative states.
#[pyclass(module = "sidereon._sidereon", name = "EncounterFrame")]
#[derive(Clone, PartialEq)]
pub struct PyEncounterFrame {
    inner: EncounterFrame,
}

#[pymethods]
impl PyEncounterFrame {
    /// In-plane cross-track unit axis.
    #[getter]
    fn x_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.x_hat)
    }

    /// Relative-velocity unit axis.
    #[getter]
    fn y_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.y_hat)
    }

    /// Encounter-plane normal unit axis.
    #[getter]
    fn z_hat<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.z_hat)
    }

    /// Relative position, object2 minus object1, kilometres.
    #[getter]
    fn relative_position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.relative_position_km)
    }

    /// Relative velocity, object2 minus object1, kilometres per second.
    #[getter]
    fn relative_velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.relative_velocity_km_s)
    }

    /// Orthogonal miss distance in the encounter plane, kilometres.
    #[getter]
    fn miss_km(&self) -> f64 {
        self.inner.miss_km
    }

    /// Relative speed, kilometres per second.
    #[getter]
    fn relative_speed_km_s(&self) -> f64 {
        self.inner.relative_speed_km_s
    }

    fn __repr__(&self) -> String {
        format!(
            "EncounterFrame(miss_km={}, relative_speed_km_s={})",
            self.inner.miss_km, self.inner.relative_speed_km_s
        )
    }

    fn __eq__(&self, other: &PyEncounterFrame) -> bool {
        self == other
    }
}

/// Collision-probability result and encounter-plane summary.
#[pyclass(module = "sidereon._sidereon", name = "CollisionProbability")]
#[derive(Clone, PartialEq)]
pub struct PyCollisionProbability {
    inner: CollisionPc,
}

#[pymethods]
impl PyCollisionProbability {
    /// Collision probability.
    #[getter]
    fn pc(&self) -> f64 {
        self.inner.pc
    }

    /// Orthogonal miss distance in the encounter plane, kilometres.
    #[getter]
    fn miss_km(&self) -> f64 {
        self.inner.miss_km
    }

    /// Relative speed, kilometres per second.
    #[getter]
    fn relative_speed_km_s(&self) -> f64 {
        self.inner.relative_speed_km_s
    }

    /// Principal-axis standard deviation in the encounter plane, kilometres.
    #[getter]
    fn sigma_x_km(&self) -> f64 {
        self.inner.sigma_x_km
    }

    /// Principal-axis standard deviation in the encounter plane, kilometres.
    #[getter]
    fn sigma_z_km(&self) -> f64 {
        self.inner.sigma_z_km
    }

    fn __repr__(&self) -> String {
        format!(
            "CollisionProbability(pc={}, miss_km={}, relative_speed_km_s={})",
            self.inner.pc, self.inner.miss_km, self.inner.relative_speed_km_s
        )
    }

    fn __eq__(&self, other: &PyCollisionProbability) -> bool {
        self == other
    }
}

/// Build the encounter frame from two position/velocity states.
#[pyfunction]
fn encounter_frame(
    position1_km: PyReadonlyArray1<'_, f64>,
    velocity1_km_s: PyReadonlyArray1<'_, f64>,
    position2_km: PyReadonlyArray1<'_, f64>,
    velocity2_km_s: PyReadonlyArray1<'_, f64>,
) -> PyResult<PyEncounterFrame> {
    let r1 = fixed_array::<3>("position1_km", &position1_km, FinitePolicy::AllowNonFinite)?;
    let v1 = fixed_array::<3>(
        "velocity1_km_s",
        &velocity1_km_s,
        FinitePolicy::AllowNonFinite,
    )?;
    let r2 = fixed_array::<3>("position2_km", &position2_km, FinitePolicy::AllowNonFinite)?;
    let v2 = fixed_array::<3>(
        "velocity2_km_s",
        &velocity2_km_s,
        FinitePolicy::AllowNonFinite,
    )?;
    let inner = core_encounter_frame(r1, v1, r2, v2).map_err(conjunction_err)?;
    Ok(PyEncounterFrame { inner })
}

/// Project a 3x3 ECI covariance into the encounter B-plane `(x, z)`.
#[pyfunction]
fn encounter_plane_covariance<'py>(
    py: Python<'py>,
    frame: PyRef<'_, PyEncounterFrame>,
    covariance_km2: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cov = matrix3_from_array(
        &covariance_km2,
        "covariance_km2",
        FinitePolicy::AllowNonFinite,
    )?;
    let projected = core_encounter_plane_covariance(&frame.inner, &cov).map_err(conjunction_err)?;
    Ok(mat2_to_array(py, &projected))
}

/// Compute collision probability from two conjunction states and hard-body
/// radius.
#[pyfunction]
#[pyo3(signature = (object1, object2, hard_body_radius_km, method=PyPcMethod::FOSTER_EQUAL_AREA))]
fn collision_probability(
    object1: PyRef<'_, PyConjunctionState>,
    object2: PyRef<'_, PyConjunctionState>,
    hard_body_radius_km: f64,
    method: PyPcMethod,
) -> PyResult<PyCollisionProbability> {
    let inner = core_collision_probability(
        &object1.core(),
        &object2.core(),
        hard_body_radius_km,
        method.into(),
    )
    .map_err(conjunction_err)?;
    Ok(PyCollisionProbability { inner })
}

/// Transform a 3x3 RTN covariance to ECI for the given orbit state.
#[pyfunction]
fn rtn_to_eci_covariance<'py>(
    py: Python<'py>,
    covariance_rtn: PyReadonlyArray2<'_, f64>,
    position_km: PyReadonlyArray1<'_, f64>,
    velocity_km_s: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let cov = matrix3_from_array(
        &covariance_rtn,
        "covariance_rtn",
        FinitePolicy::AllowNonFinite,
    )?;
    let r = fixed_array::<3>("position_km", &position_km, FinitePolicy::AllowNonFinite)?;
    let v = fixed_array::<3>(
        "velocity_km_s",
        &velocity_km_s,
        FinitePolicy::AllowNonFinite,
    )?;
    let eci = rtn_to_eci(&cov, r, v).map_err(rtn_frame_err)?;
    Ok(mat3_to_array(py, &eci))
}

/// Return whether a 3x3 covariance is symmetric within the engine tolerance.
#[pyfunction]
fn covariance_is_symmetric(covariance_km2: PyReadonlyArray2<'_, f64>) -> PyResult<bool> {
    Ok(symmetric(&matrix3_from_array(
        &covariance_km2,
        "covariance_km2",
        FinitePolicy::AllowNonFinite,
    )?))
}

/// Return whether a 3x3 covariance is symmetric positive semidefinite within the
/// engine tolerance.
#[pyfunction]
fn covariance_is_positive_semidefinite(
    covariance_km2: PyReadonlyArray2<'_, f64>,
) -> PyResult<bool> {
    Ok(positive_semidefinite(&matrix3_from_array(
        &covariance_km2,
        "covariance_km2",
        FinitePolicy::AllowNonFinite,
    )?))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPcMethod>()?;
    m.add_class::<PyConjunctionState>()?;
    m.add_class::<PyEncounterFrame>()?;
    m.add_class::<PyCollisionProbability>()?;
    m.add_function(wrap_pyfunction!(encounter_frame, m)?)?;
    m.add_function(wrap_pyfunction!(encounter_plane_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(collision_probability, m)?)?;
    m.add_function(wrap_pyfunction!(rtn_to_eci_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_is_symmetric, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_is_positive_semidefinite, m)?)?;
    Ok(())
}
