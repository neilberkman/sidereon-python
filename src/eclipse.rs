//! Lunar occultation and satellite-position derivatives of shadow fractions.

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::constants::astro::IAU_2015_SOLAR_RADIUS_KM;
use sidereon_core::astro::events::eclipse::{
    self, EclipseError, ShadowFunction, ShadowGradientError as CoreShadowGradientError,
};

use crate::events::{PyEarthShadowModel, PyEclipseStatus};
use crate::SolveError;

/// Solar lighting model used by radiation pressure.
#[pyclass(module = "sidereon._sidereon", name = "ShadowFunction", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyShadowFunction {
    /// Apparent-disc occultation.
    APPARENT_DISC,
    /// Earth-only linear conical shadow.
    CONICAL,
}

impl From<PyShadowFunction> for ShadowFunction {
    fn from(value: PyShadowFunction) -> Self {
        match value {
            PyShadowFunction::APPARENT_DISC => Self::ApparentDisc,
            PyShadowFunction::CONICAL => Self::Conical,
        }
    }
}

impl TryFrom<ShadowFunction> for PyShadowFunction {
    type Error = PyErr;

    fn try_from(value: ShadowFunction) -> PyResult<Self> {
        match value {
            ShadowFunction::ApparentDisc => Ok(Self::APPARENT_DISC),
            ShadowFunction::Conical => Ok(Self::CONICAL),
            _ => Err(SolveError::new_err("unsupported shadow function")),
        }
    }
}

fn eclipse_error(error: EclipseError) -> PyErr {
    let EclipseError::InvalidInput { field, reason } = error;
    crate::binding_error(
        SolveError::new_err(error.to_string()),
        "invalid_input",
        |detail| {
            detail.set_item("field", field)?;
            detail.set_item("reason", reason)
        },
    )
}

create_exception!(
    _sidereon,
    ShadowGradientError,
    SolveError,
    "A shadow-fraction gradient evaluation failure."
);

fn gradient_error(error: CoreShadowGradientError) -> PyErr {
    let exception = ShadowGradientError::new_err(error.to_string());
    match error {
        CoreShadowGradientError::Eclipse(source) => {
            crate::binding_error(exception, "eclipse", |detail| {
                let source = eclipse_error(source);
                detail.set_item("source", source.value(detail.py()).getattr("detail")?)
            })
        }
        CoreShadowGradientError::NumericalFailure => {
            crate::binding_error(exception, "numerical_failure", |_| Ok(()))
        }
        _ => exception,
    }
}

/// Fraction of the Sun obscured by the Moon.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn moon_shadow_fraction(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    moon_position_km: [f64; 3],
) -> PyResult<f64> {
    eclipse::moon_shadow_fraction(satellite_position_km, sun_position_km, moon_position_km)
        .map_err(eclipse_error)
}

/// Combined spherical Earth and Moon shadow fraction.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn shadow_fraction_with_moon(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    moon_position_km: [f64; 3],
) -> PyResult<f64> {
    eclipse::shadow_fraction_with_moon(satellite_position_km, sun_position_km, moon_position_km)
        .map_err(eclipse_error)
}

/// Combined Earth and Moon shadow fraction with an explicit Earth figure.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn shadow_fraction_with_model_and_moon(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    moon_position_km: [f64; 3],
    model: PyEarthShadowModel,
) -> PyResult<f64> {
    eclipse::shadow_fraction_with_model_and_moon(
        satellite_position_km,
        sun_position_km,
        moon_position_km,
        model.into(),
    )
    .map_err(eclipse_error)
}

/// Eclipse classification with spherical Earth and Moon occultation.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn status_with_moon(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    moon_position_km: [f64; 3],
) -> PyResult<PyEclipseStatus> {
    eclipse::status_with_moon(satellite_position_km, sun_position_km, moon_position_km)
        .map(Into::into)
        .map_err(eclipse_error)
}

/// Eclipse classification with an explicit Earth figure and Moon occultation.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn status_with_model_and_moon(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    moon_position_km: [f64; 3],
    model: PyEarthShadowModel,
) -> PyResult<PyEclipseStatus> {
    eclipse::status_with_model_and_moon(
        satellite_position_km,
        sun_position_km,
        moon_position_km,
        model.into(),
    )
    .map(Into::into)
    .map_err(eclipse_error)
}

/// Spherical Earth conical shadow and its satellite-position gradient.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn shadow_fraction_with_gradient(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
) -> PyResult<(f64, [f64; 3])> {
    eclipse::shadow_fraction_with_gradient(satellite_position_km, sun_position_km)
        .map_err(gradient_error)
}

/// Conical shadow and its satellite-position gradient for an explicit Earth figure.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn shadow_fraction_with_model_and_gradient(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    model: PyEarthShadowModel,
) -> PyResult<(f64, [f64; 3])> {
    eclipse::shadow_fraction_with_model_and_gradient(
        satellite_position_km,
        sun_position_km,
        model.into(),
    )
    .map_err(gradient_error)
}

/// Combined Earth and Moon shadow and its satellite-position gradient.
///
/// Positions are inertial vectors in kilometres.
#[pyfunction]
fn shadow_fraction_with_model_moon_and_gradient(
    satellite_position_km: [f64; 3],
    sun_position_km: [f64; 3],
    moon_position_km: [f64; 3],
    model: PyEarthShadowModel,
) -> PyResult<(f64, [f64; 3])> {
    eclipse::shadow_fraction_with_model_moon_and_gradient(
        satellite_position_km,
        sun_position_km,
        moon_position_km,
        model.into(),
    )
    .map_err(gradient_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IAU_2015_SOLAR_RADIUS_KM", IAU_2015_SOLAR_RADIUS_KM)?;
    m.add_class::<PyShadowFunction>()?;
    m.add(
        "ShadowGradientError",
        m.py().get_type::<ShadowGradientError>(),
    )?;
    m.add_function(wrap_pyfunction!(moon_shadow_fraction, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_fraction_with_moon, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_fraction_with_model_and_moon, m)?)?;
    m.add_function(wrap_pyfunction!(status_with_moon, m)?)?;
    m.add_function(wrap_pyfunction!(status_with_model_and_moon, m)?)?;
    m.add_function(wrap_pyfunction!(shadow_fraction_with_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(
        shadow_fraction_with_model_and_gradient,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        shadow_fraction_with_model_moon_and_gradient,
        m
    )?)?;
    Ok(())
}
