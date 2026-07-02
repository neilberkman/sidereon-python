//! Relative-frame and Clohessy-Wiltshire binding.

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::astro::covariance::RtnFrameError;
use sidereon_core::astro::relative as core;
use sidereon_core::astro::state::CartesianState;

use crate::marshal::{fixed_array_from_any, mat3_to_array, rows_to_array, FinitePolicy};
use crate::np_array;

fn to_relative_err(err: RtnFrameError) -> PyErr {
    let message = match err {
        RtnFrameError::InvalidInput { field, reason } => {
            format!("invalid input for {field}: {reason}")
        }
        other => other.message().to_string(),
    };
    PyValueError::new_err(message)
}

fn state_from_arrays(
    epoch_tdb_seconds: f64,
    position_km: &Bound<'_, PyAny>,
    velocity_km_s: &Bound<'_, PyAny>,
) -> PyResult<CartesianState> {
    let position =
        fixed_array_from_any::<3>("position_km", position_km, FinitePolicy::RequireFinite)?;
    let velocity =
        fixed_array_from_any::<3>("velocity_km_s", velocity_km_s, FinitePolicy::RequireFinite)?;
    Ok(CartesianState::new(epoch_tdb_seconds, position, velocity))
}

#[pyclass(module = "sidereon._sidereon", name = "CartesianState")]
#[derive(Clone, Copy)]
/// Cartesian position and velocity at a TDB epoch.
///
/// Positions are kilometres and velocities are kilometres per second.
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
    /// Build a Cartesian state from epoch, position, and velocity.
    ///
    /// The vectors may be numpy arrays or ordinary Python sequences of three finite floats.
    #[new]
    fn new(
        epoch_tdb_seconds: f64,
        position_km: &Bound<'_, PyAny>,
        velocity_km_s: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: state_from_arrays(epoch_tdb_seconds, position_km, velocity_km_s)?,
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
    rotation: Result<[[f64; 3]; 3], RtnFrameError>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    Ok(mat3_to_array(py, &rotation.map_err(to_relative_err)?))
}

#[pyfunction]
/// Return the rotation from RSW coordinates to inertial coordinates.
///
/// The result is a 3-by-3 numpy array whose rows are the inertial basis vectors.
fn rsw_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::rsw_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
/// Return the rotation from RTN coordinates to inertial coordinates.
///
/// The result is a 3-by-3 numpy array whose rows are the inertial basis vectors.
fn rtn_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::rtn_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
/// Return the rotation from RIC coordinates to inertial coordinates.
///
/// The result is a 3-by-3 numpy array whose rows are the inertial basis vectors.
fn ric_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::ric_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
/// Return the rotation from LVLH coordinates to inertial coordinates.
///
/// The result is a 3-by-3 numpy array whose rows are the inertial basis vectors.
fn lvlh_to_inertial_rotation<'py>(
    py: Python<'py>,
    chief: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    rotation_to_array(py, core::lvlh_to_inertial_rotation(chief.inner()))
}

#[pyfunction]
/// Express the deputy state relative to the chief in the local orbital frame.
///
/// The returned state carries relative position and velocity components.
fn relative_state(
    chief: &PyCartesianState,
    deputy: &PyCartesianState,
) -> PyResult<PyCartesianState> {
    core::relative_state(chief.inner(), deputy.inner())
        .map(PyCartesianState::from_inner)
        .map_err(to_relative_err)
}

#[pyfunction]
/// Rebuild an inertial deputy state from a chief state and local relative state.
///
/// This is the inverse operation for `relative_state` within numerical tolerance.
fn absolute_from_relative(
    chief: &PyCartesianState,
    relative: &PyCartesianState,
) -> PyResult<PyCartesianState> {
    core::absolute_from_relative(chief.inner(), relative.inner())
        .map(PyCartesianState::from_inner)
        .map_err(to_relative_err)
}

#[pyfunction]
/// Return the Clohessy-Wiltshire state-transition matrix.
///
/// `mean_motion_rad_s` is the circular-orbit mean motion and `dt_s` is the propagation interval.
fn cw_stm<'py>(
    py: Python<'py>,
    mean_motion_rad_s: f64,
    dt_s: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let stm = core::cw_stm(mean_motion_rad_s, dt_s).map_err(to_relative_err)?;
    Ok(rows_to_array(py, &stm))
}

#[pyfunction]
/// Propagate a relative state with the Clohessy-Wiltshire equations.
///
/// The returned state uses the same local frame convention as `relative_state`.
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
/// Return circular-orbit mean motion for a radius in kilometres.
fn mean_motion_circular(radius_km: f64) -> PyResult<f64> {
    core::mean_motion_circular(radius_km).map_err(to_relative_err)
}

#[pyfunction]
/// Estimate mean motion from a chief Cartesian state.
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
