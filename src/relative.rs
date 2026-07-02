//! Relative-frame and Clohessy-Wiltshire binding.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::relative as core;
use sidereon_core::astro::state::CartesianState;

use crate::marshal::{fixed_array, mat3_to_array, rows_to_array, FinitePolicy};
use crate::np_array;

fn to_relative_err<E: std::fmt::Debug>(err: E) -> PyErr {
    PyValueError::new_err(format!("{err:?}"))
}

fn state_from_arrays(
    epoch_tdb_seconds: f64,
    position_km: &PyReadonlyArray1<'_, f64>,
    velocity_km_s: &PyReadonlyArray1<'_, f64>,
) -> PyResult<CartesianState> {
    let position = fixed_array::<3>("position_km", position_km, FinitePolicy::RequireFinite)?;
    let velocity = fixed_array::<3>("velocity_km_s", velocity_km_s, FinitePolicy::RequireFinite)?;
    Ok(CartesianState::new(epoch_tdb_seconds, position, velocity))
}

#[pyclass(module = "sidereon._sidereon", name = "CartesianState")]
#[derive(Clone, Copy)]
pub struct PyCartesianState {
    inner: CartesianState,
}

impl PyCartesianState {
    pub(crate) fn from_inner(inner: CartesianState) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &CartesianState {
        &self.inner
    }
}

#[pymethods]
impl PyCartesianState {
    #[new]
    fn new(
        epoch_tdb_seconds: f64,
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: state_from_arrays(epoch_tdb_seconds, &position_km, &velocity_km_s)?,
        })
    }

    #[getter]
    fn epoch_tdb_seconds(&self) -> f64 {
        self.inner.epoch_tdb_seconds
    }

    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_array())
    }

    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_array())
    }

    fn __repr__(&self) -> String {
        format!(
            "CartesianState(epoch_tdb_seconds={:.6}, position_km={:?}, velocity_km_s={:?})",
            self.inner.epoch_tdb_seconds,
            self.inner.position_array(),
            self.inner.velocity_array()
        )
    }
}

fn rotation_to_array<'py>(
    py: Python<'py>,
    rotation: Result<[[f64; 3]; 3], impl std::fmt::Debug>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(mat3_to_array(py, &rotation.map_err(to_relative_err)?))
}

#[pyfunction]
fn rsw_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::rsw_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
fn rtn_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::rtn_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
fn ric_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::ric_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
fn lvlh_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::lvlh_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
fn relative_state(
    chief: &PyCartesianState,
    deputy: &PyCartesianState,
) -> PyResult<PyCartesianState> {
    core::relative_state(chief.inner(), deputy.inner())
        .map(PyCartesianState::from_inner)
        .map_err(to_relative_err)
}

#[pyfunction]
fn absolute_from_relative(
    chief: &PyCartesianState,
    relative: &PyCartesianState,
) -> PyResult<PyCartesianState> {
    core::absolute_from_relative(chief.inner(), relative.inner())
        .map(PyCartesianState::from_inner)
        .map_err(to_relative_err)
}

#[pyfunction]
fn cw_stm<'py>(
    py: Python<'py>,
    mean_motion_rad_s: f64,
    dt_s: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let stm = core::cw_stm(mean_motion_rad_s, dt_s).map_err(to_relative_err)?;
    Ok(rows_to_array(py, &stm))
}

#[pyfunction]
fn cw_propagate(
    relative: &PyCartesianState,
    mean_motion_rad_s: f64,
    dt_s: f64,
) -> PyResult<PyCartesianState> {
    core::cw_propagate(relative.inner(), mean_motion_rad_s, dt_s)
        .map(PyCartesianState::from_inner)
        .map_err(to_relative_err)
}

#[pyfunction]
fn mean_motion_circular(radius_km: f64) -> PyResult<f64> {
    core::mean_motion_circular(radius_km).map_err(to_relative_err)
}

#[pyfunction]
fn mean_motion_from_state(chief: &PyCartesianState) -> PyResult<f64> {
    core::mean_motion_from_state(chief.inner()).map_err(to_relative_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCartesianState>()?;
    m.add_function(wrap_pyfunction!(rsw_to_inertial_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(rtn_to_inertial_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(ric_to_inertial_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(lvlh_to_inertial_rotation, m)?)?;
    m.add_function(wrap_pyfunction!(relative_state, m)?)?;
    m.add_function(wrap_pyfunction!(absolute_from_relative, m)?)?;
    m.add_function(wrap_pyfunction!(cw_stm, m)?)?;
    m.add_function(wrap_pyfunction!(cw_propagate, m)?)?;
    m.add_function(wrap_pyfunction!(mean_motion_circular, m)?)?;
    m.add_function(wrap_pyfunction!(mean_motion_from_state, m)?)?;
    Ok(())
}
