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

use sidereon_core::astro::forces::{ForceModel, J2Gravity, TwoBodyGravity};
use sidereon_core::astro::propagator::api::PropagationContext;
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(force_twobody_acceleration, m)?)?;
    m.add_function(wrap_pyfunction!(force_j2_acceleration, m)?)?;
    Ok(())
}
