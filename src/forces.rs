//! Force-model acceleration binding.
//!
//! Thin marshaling over [`sidereon_core::astro::forces`]: builds a
//! [`CartesianState`] from a position (km) and velocity (km/s) and returns the
//! force model's acceleration (km/s^2) as a numpy `(3,)` array. No acceleration
//! formula lives here; the numbers are exactly what `sidereon-core` produces
//! (0-ULP against the core goldens).

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::forces::{
    DragForce, DragParameters, ForceModel, J2Gravity, SpaceWeather, TwoBodyGravity,
};
use sidereon_core::astro::propagator::api::PropagationContext;
use sidereon_core::astro::propagator::decay::{
    estimate_decay as core_estimate_decay, DecayConfig, DecayEstimate,
};
use sidereon_core::astro::state::CartesianState;

use crate::np_array;

fn acceleration<'py>(
    py: Python<'py>,
    force: &dyn ForceModel,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let state = CartesianState::new(0.0, position_km, velocity_km_s);
    let accel = force
        .acceleration(&state, &PropagationContext::default())
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(np_array(py, &[accel.x, accel.y, accel.z]))
}

#[pyclass(module = "sidereon._sidereon", name = "SpaceWeather")]
#[derive(Clone, Copy)]
pub struct PySpaceWeather {
    inner: SpaceWeather,
}

impl PySpaceWeather {
    fn inner(&self) -> SpaceWeather {
        self.inner
    }
}

#[pymethods]
impl PySpaceWeather {
    #[new]
    #[pyo3(signature = (
        f107=SpaceWeather::default().f107,
        f107a=SpaceWeather::default().f107a,
        ap=SpaceWeather::default().ap,
    ))]
    fn new(f107: f64, f107a: f64, ap: f64) -> Self {
        Self {
            inner: SpaceWeather { f107, f107a, ap },
        }
    }

    #[getter]
    fn f107(&self) -> f64 {
        self.inner.f107
    }

    #[getter]
    fn f107a(&self) -> f64 {
        self.inner.f107a
    }

    #[getter]
    fn ap(&self) -> f64 {
        self.inner.ap
    }

    fn __repr__(&self) -> String {
        format!(
            "SpaceWeather(f107={}, f107a={}, ap={})",
            self.inner.f107, self.inner.f107a, self.inner.ap
        )
    }
}

impl From<SpaceWeather> for PySpaceWeather {
    fn from(inner: SpaceWeather) -> Self {
        Self { inner }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "DragParameters")]
#[derive(Clone, Copy)]
pub struct PyDragParameters {
    inner: DragParameters,
}

impl PyDragParameters {
    pub(crate) fn inner(&self) -> DragParameters {
        self.inner
    }

    fn sw(space_weather: Option<&PySpaceWeather>) -> SpaceWeather {
        space_weather.map(PySpaceWeather::inner).unwrap_or_default()
    }
}

#[pymethods]
impl PyDragParameters {
    #[staticmethod]
    #[pyo3(signature = (cd, area_m2, mass_kg, space_weather=None, cutoff_altitude_km=DragForce::DEFAULT_REENTRY_ALTITUDE_KM))]
    fn from_area_mass(
        cd: f64,
        area_m2: f64,
        mass_kg: f64,
        space_weather: Option<&PySpaceWeather>,
        cutoff_altitude_km: f64,
    ) -> PyResult<Self> {
        DragParameters::from_area_mass(
            cd,
            area_m2,
            mass_kg,
            Self::sw(space_weather),
            cutoff_altitude_km,
        )
        .map(|inner| Self { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (bc_factor_m2_kg, space_weather=None, cutoff_altitude_km=DragForce::DEFAULT_REENTRY_ALTITUDE_KM))]
    fn from_bc_factor_m2_kg(
        bc_factor_m2_kg: f64,
        space_weather: Option<&PySpaceWeather>,
        cutoff_altitude_km: f64,
    ) -> PyResult<Self> {
        DragParameters::from_bc_factor_m2_kg(
            bc_factor_m2_kg,
            Self::sw(space_weather),
            cutoff_altitude_km,
        )
        .map(|inner| Self { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (bc_kg_m2, space_weather=None, cutoff_altitude_km=DragForce::DEFAULT_REENTRY_ALTITUDE_KM))]
    fn from_ballistic_coefficient(
        bc_kg_m2: f64,
        space_weather: Option<&PySpaceWeather>,
        cutoff_altitude_km: f64,
    ) -> PyResult<Self> {
        DragParameters::from_ballistic_coefficient(
            bc_kg_m2,
            Self::sw(space_weather),
            cutoff_altitude_km,
        )
        .map(|inner| Self { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[allow(clippy::wrong_self_convention)] // Public Python API keeps to_force while the Rust wrapper is Copy.
    fn to_force(&self) -> PyDragForce {
        PyDragForce {
            inner: self.inner.to_force(),
        }
    }

    #[getter]
    fn bc_factor_m2_kg(&self) -> f64 {
        self.inner.bc_factor_m2_kg()
    }

    #[getter]
    fn space_weather(&self) -> PySpaceWeather {
        self.inner.space_weather().into()
    }

    #[getter]
    fn cutoff_altitude_km(&self) -> f64 {
        self.inner.cutoff_altitude_km()
    }

    fn __repr__(&self) -> String {
        format!(
            "DragParameters(bc_factor_m2_kg={}, cutoff_altitude_km={})",
            self.inner.bc_factor_m2_kg(),
            self.inner.cutoff_altitude_km()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "DragForce")]
#[derive(Clone, Copy)]
pub struct PyDragForce {
    inner: DragForce,
}

impl PyDragForce {
    fn sw(space_weather: Option<&PySpaceWeather>) -> SpaceWeather {
        space_weather.map(PySpaceWeather::inner).unwrap_or_default()
    }
}

#[pymethods]
impl PyDragForce {
    #[staticmethod]
    #[pyo3(signature = (cd, area_m2, mass_kg, space_weather=None, cutoff_altitude_km=DragForce::DEFAULT_REENTRY_ALTITUDE_KM))]
    fn from_area_mass(
        cd: f64,
        area_m2: f64,
        mass_kg: f64,
        space_weather: Option<&PySpaceWeather>,
        cutoff_altitude_km: f64,
    ) -> PyResult<Self> {
        DragForce::from_area_mass(
            cd,
            area_m2,
            mass_kg,
            Self::sw(space_weather),
            cutoff_altitude_km,
        )
        .map(|inner| Self { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (bc_factor_m2_kg, space_weather=None, cutoff_altitude_km=DragForce::DEFAULT_REENTRY_ALTITUDE_KM))]
    fn from_bc_factor_m2_kg(
        bc_factor_m2_kg: f64,
        space_weather: Option<&PySpaceWeather>,
        cutoff_altitude_km: f64,
    ) -> PyResult<Self> {
        DragForce::from_bc_factor_m2_kg(
            bc_factor_m2_kg,
            Self::sw(space_weather),
            cutoff_altitude_km,
        )
        .map(|inner| Self { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[staticmethod]
    #[pyo3(signature = (bc_kg_m2, space_weather=None, cutoff_altitude_km=DragForce::DEFAULT_REENTRY_ALTITUDE_KM))]
    fn from_ballistic_coefficient(
        bc_kg_m2: f64,
        space_weather: Option<&PySpaceWeather>,
        cutoff_altitude_km: f64,
    ) -> PyResult<Self> {
        DragForce::from_ballistic_coefficient(bc_kg_m2, Self::sw(space_weather), cutoff_altitude_km)
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    #[getter]
    fn bc_factor_m2_kg(&self) -> f64 {
        self.inner.bc_factor_m2_kg()
    }

    #[getter]
    fn space_weather(&self) -> PySpaceWeather {
        self.inner.space_weather().into()
    }

    #[getter]
    fn cutoff_altitude_km(&self) -> f64 {
        self.inner.cutoff_altitude_km()
    }

    fn __repr__(&self) -> String {
        format!(
            "DragForce(bc_factor_m2_kg={}, cutoff_altitude_km={})",
            self.inner.bc_factor_m2_kg(),
            self.inner.cutoff_altitude_km()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "DecayEstimate")]
pub struct PyDecayEstimate {
    inner: DecayEstimate,
}

#[pymethods]
impl PyDecayEstimate {
    #[getter]
    fn time_to_decay_s(&self) -> f64 {
        self.inner.time_to_decay_s
    }

    #[getter]
    fn epoch_tdb_seconds(&self) -> f64 {
        self.inner.reentry_state.epoch_tdb_seconds
    }

    #[getter]
    fn reentry_position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.reentry_state.position_array())
    }

    #[getter]
    fn reentry_velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.reentry_state.velocity_array())
    }

    #[getter]
    fn reentry_altitude_km(&self) -> f64 {
        self.inner.reentry_altitude_km
    }

    fn __repr__(&self) -> String {
        format!(
            "DecayEstimate(time_to_decay_s={}, reentry_altitude_km={})",
            self.inner.time_to_decay_s, self.inner.reentry_altitude_km
        )
    }
}

/// Two-body (point-mass Earth gravity) acceleration in km/s^2.
///
/// `position_km` is the ECI position (km); `velocity_km_s` is accepted for a
/// uniform force-model signature but does not affect the conservative two-body
/// term. Returns a numpy `(3,)` array. Raises `ValueError` on a zero position.
#[pyfunction]
fn force_twobody_acceleration<'py>(
    py: Python<'py>,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    acceleration(py, &TwoBodyGravity::default(), position_km, velocity_km_s)
}

/// J2 oblateness perturbing acceleration in km/s^2.
///
/// `position_km` is the ECI position (km); `velocity_km_s` is accepted for a
/// uniform force-model signature but does not affect the J2 term. Returns a
/// numpy `(3,)` array. Raises `ValueError` on a zero position.
#[pyfunction]
fn force_j2_acceleration<'py>(
    py: Python<'py>,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    acceleration(py, &J2Gravity::default(), position_km, velocity_km_s)
}

#[pyfunction]
fn force_drag_acceleration<'py>(
    py: Python<'py>,
    drag: &PyDragForce,
    epoch_tdb_seconds: f64,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let state = CartesianState::new(epoch_tdb_seconds, position_km, velocity_km_s);
    let accel = drag
        .inner
        .acceleration(&state, &PropagationContext::default())
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(np_array(py, &[accel.x, accel.y, accel.z]))
}

#[pyfunction]
#[pyo3(signature = (
    epoch_tdb_seconds,
    position_km,
    velocity_km_s,
    drag,
    *,
    reentry_altitude_km=DecayConfig::DEFAULT_REENTRY_ALTITUDE_KM,
    scan_step_s=DecayConfig::DEFAULT_SCAN_STEP_S,
    crossing_tolerance_s=DecayConfig::DEFAULT_CROSSING_TOLERANCE_S,
    max_duration_s=DecayConfig::DEFAULT_MAX_DURATION_S,
    max_scan_samples=DecayConfig::DEFAULT_MAX_SCAN_SAMPLES,
))]
#[allow(clippy::too_many_arguments)]
fn estimate_decay(
    py: Python<'_>,
    epoch_tdb_seconds: f64,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
    drag: &PyDragParameters,
    reentry_altitude_km: f64,
    scan_step_s: f64,
    crossing_tolerance_s: f64,
    max_duration_s: f64,
    max_scan_samples: u32,
) -> PyResult<PyDecayEstimate> {
    let initial = CartesianState::new(epoch_tdb_seconds, position_km, velocity_km_s);
    let config = DecayConfig::new(drag.inner())
        .with_reentry_altitude_km(reentry_altitude_km)
        .with_scan_step_s(scan_step_s)
        .with_crossing_tolerance_s(crossing_tolerance_s)
        .with_max_duration_s(max_duration_s)
        .with_max_scan_samples(max_scan_samples);
    py.allow_threads(move || core_estimate_decay(initial, &config))
        .map(|inner| PyDecayEstimate { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpaceWeather>()?;
    m.add_class::<PyDragParameters>()?;
    m.add_class::<PyDragForce>()?;
    m.add_class::<PyDecayEstimate>()?;
    m.add_function(wrap_pyfunction!(force_twobody_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(force_j2_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(force_drag_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_decay, m)?)?;
    Ok(())
}
