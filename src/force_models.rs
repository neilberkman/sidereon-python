//! Estimable force parameters, ECOM radiation pressure, and empirical accelerations.

use std::sync::Arc;

use nalgebra::Matrix3;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::error::PropagationError;
use sidereon_core::astro::forces::{self, AccelerationPartials, ForceModel, ForceParameter};
use sidereon_core::astro::propagator::api::PropagationContext;

use crate::earth_orientation::PyPropagationContext;
use crate::eclipse::PyShadowFunction;
use crate::events::PyEarthShadowModel;
use crate::forces::{PyDragForce, PyDragParameters, PySourcedDragForce};
use crate::gnss_attitude::PyDybAxes;
use crate::gravity_field::PySphericalHarmonicGravity;
use crate::propagation::{PyForceModel, PySolarRadiationPressure};
use crate::relative::PyCartesianState;
use crate::{binding_error, degrade_reason_label, SolveError};

pub(crate) fn propagation_error(error: PropagationError) -> PyErr {
    let exception = SolveError::new_err(error.to_string());
    propagation_error_detail(error, exception)
}

pub(crate) fn propagation_error_detail(error: PropagationError, exception: PyErr) -> PyErr {
    let (kind, message) = match error {
        PropagationError::InvalidInput(message) => ("invalid_input", message),
        PropagationError::NumericalFailure(message) => ("numerical_failure", message),
        PropagationError::EventFailure(message) => ("event_failure", message),
        PropagationError::ForceModelFailure(message) => ("force_model_failure", message),
        PropagationError::MaxStepsExceeded => {
            return binding_error(exception, "max_steps_exceeded", |_| Ok(()))
        }
        PropagationError::Ut1OutsideCoverage(reason) => {
            return binding_error(exception, "ut1_outside_coverage", |detail| {
                detail.set_item("reason", degrade_reason_label(reason))
            })
        }
    };
    binding_error(exception, kind, |detail| {
        detail.set_item("message", message)
    })
}

pub(crate) fn with_context<T>(
    context: Option<&PyPropagationContext>,
    action: impl FnOnce(&PropagationContext) -> T,
) -> T {
    match context {
        Some(context) => action(&context.inner),
        None => action(&PropagationContext::default()),
    }
}

pub(crate) fn matrix_rows(matrix: &Matrix3<f64>) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}

fn matrix_from_rows(rows: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_fn(|row, column| rows[row][column])
}
/// An estimable force parameter and its configured value.
#[pyclass(module = "sidereon._sidereon", name = "ForceParameter", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyForceParameter(pub(crate) ForceParameter);

#[pymethods]
impl PyForceParameter {
    /// Pair a parameter name and value.
    #[new]
    fn new(name: String, value: f64) -> Self {
        Self(ForceParameter::new(name, value))
    }

    /// Parameter name.
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    /// Value in the force model's parameter units.
    #[getter]
    fn value(&self) -> f64 {
        self.0.value
    }
}

/// Acceleration derivatives with respect to inertial position and velocity.
#[pyclass(
    module = "sidereon._sidereon",
    name = "AccelerationPartials",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyAccelerationPartials(pub(crate) AccelerationPartials);

#[pymethods]
impl PyAccelerationPartials {
    /// Derivative with respect to position, in inverse square seconds.
    #[getter]
    fn wrt_position(&self) -> [[f64; 3]; 3] {
        matrix_rows(&self.0.wrt_position)
    }

    /// Derivative with respect to velocity, in inverse seconds.
    #[getter]
    fn wrt_velocity(&self) -> [[f64; 3]; 3] {
        matrix_rows(&self.0.wrt_velocity)
    }

    /// Return two zero derivative matrices.
    #[staticmethod]
    fn zeros() -> Self {
        Self(AccelerationPartials::zeros())
    }

    /// Build an acceleration derivative with a zero velocity block.
    #[staticmethod]
    fn position_only(wrt_position: [[f64; 3]; 3]) -> Self {
        Self(AccelerationPartials::position_only(matrix_from_rows(
            wrt_position,
        )))
    }
}

/// Acceleration coefficients for Ecom1 in metres per square second.
#[pyclass(module = "sidereon._sidereon", name = "Ecom1Coefficients", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyEcom1Coefficients(forces::Ecom1Coefficients);

#[pymethods]
impl PyEcom1Coefficients {
    /// Build a coefficient set with omitted coefficients set to zero.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (d0=0.0, dc=0.0, ds=0.0, y0=0.0, yc=0.0, ys=0.0, b0=0.0, bc=0.0, bs=0.0))]
    fn new(
        d0: f64,
        dc: f64,
        ds: f64,
        y0: f64,
        yc: f64,
        ys: f64,
        b0: f64,
        bc: f64,
        bs: f64,
    ) -> Self {
        Self(forces::Ecom1Coefficients {
            d0,
            dc,
            ds,
            y0,
            yc,
            ys,
            b0,
            bc,
            bs,
        })
    }

    /// Coefficient d0 in metres per square second.
    #[getter]
    fn d0(&self) -> f64 {
        self.0.d0
    }

    /// Coefficient dc in metres per square second.
    #[getter]
    fn dc(&self) -> f64 {
        self.0.dc
    }

    /// Coefficient ds in metres per square second.
    #[getter]
    fn ds(&self) -> f64 {
        self.0.ds
    }

    /// Coefficient y0 in metres per square second.
    #[getter]
    fn y0(&self) -> f64 {
        self.0.y0
    }

    /// Coefficient yc in metres per square second.
    #[getter]
    fn yc(&self) -> f64 {
        self.0.yc
    }

    /// Coefficient ys in metres per square second.
    #[getter]
    fn ys(&self) -> f64 {
        self.0.ys
    }

    /// Coefficient b0 in metres per square second.
    #[getter]
    fn b0(&self) -> f64 {
        self.0.b0
    }

    /// Coefficient bc in metres per square second.
    #[getter]
    fn bc(&self) -> f64 {
        self.0.bc
    }

    /// Coefficient bs in metres per square second.
    #[getter]
    fn bs(&self) -> f64 {
        self.0.bs
    }

    /// Read a coefficient sequence in model order.
    #[staticmethod]
    fn from_array(values: [f64; forces::ECOM1_COEFFICIENT_COUNT]) -> Self {
        Self(forces::Ecom1Coefficients::from_array(values))
    }

    /// Return the coefficients in model order.
    fn to_array(&self) -> [f64; forces::ECOM1_COEFFICIENT_COUNT] {
        self.0.to_array()
    }

    /// Coefficient names in model order.
    #[classattr]
    const NAMES: [&'static str; forces::ECOM1_COEFFICIENT_COUNT] = forces::Ecom1Coefficients::NAMES;
    /// Estimable parameter names in model order.
    #[classattr]
    const PARAMETER_NAMES: [&'static str; forces::ECOM1_COEFFICIENT_COUNT] =
        forces::Ecom1Coefficients::PARAMETER_NAMES;
}

/// Acceleration coefficients for Ecom2D2b1 in metres per square second.
#[pyclass(
    module = "sidereon._sidereon",
    name = "Ecom2D2b1Coefficients",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyEcom2D2b1Coefficients(forces::Ecom2D2b1Coefficients);

#[pymethods]
impl PyEcom2D2b1Coefficients {
    /// Build a coefficient set with omitted coefficients set to zero.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (b0=0.0, b1c=0.0, b1s=0.0, d0=0.0, d2c=0.0, d2s=0.0, y0=0.0))]
    fn new(b0: f64, b1c: f64, b1s: f64, d0: f64, d2c: f64, d2s: f64, y0: f64) -> Self {
        Self(forces::Ecom2D2b1Coefficients {
            b0,
            b1c,
            b1s,
            d0,
            d2c,
            d2s,
            y0,
        })
    }

    /// Coefficient b0 in metres per square second.
    #[getter]
    fn b0(&self) -> f64 {
        self.0.b0
    }

    /// Coefficient b1c in metres per square second.
    #[getter]
    fn b1c(&self) -> f64 {
        self.0.b1c
    }

    /// Coefficient b1s in metres per square second.
    #[getter]
    fn b1s(&self) -> f64 {
        self.0.b1s
    }

    /// Coefficient d0 in metres per square second.
    #[getter]
    fn d0(&self) -> f64 {
        self.0.d0
    }

    /// Coefficient d2c in metres per square second.
    #[getter]
    fn d2c(&self) -> f64 {
        self.0.d2c
    }

    /// Coefficient d2s in metres per square second.
    #[getter]
    fn d2s(&self) -> f64 {
        self.0.d2s
    }

    /// Coefficient y0 in metres per square second.
    #[getter]
    fn y0(&self) -> f64 {
        self.0.y0
    }

    /// Read a coefficient sequence in model order.
    #[staticmethod]
    fn from_array(values: [f64; forces::ECOM2_D2B1_COEFFICIENT_COUNT]) -> Self {
        Self(forces::Ecom2D2b1Coefficients::from_array(values))
    }

    /// Return the coefficients in model order.
    fn to_array(&self) -> [f64; forces::ECOM2_D2B1_COEFFICIENT_COUNT] {
        self.0.to_array()
    }

    /// Coefficient names in model order.
    #[classattr]
    const NAMES: [&'static str; forces::ECOM2_D2B1_COEFFICIENT_COUNT] =
        forces::Ecom2D2b1Coefficients::NAMES;
    /// Estimable parameter names in model order.
    #[classattr]
    const PARAMETER_NAMES: [&'static str; forces::ECOM2_D2B1_COEFFICIENT_COUNT] =
        forces::Ecom2D2b1Coefficients::PARAMETER_NAMES;
}

/// Acceleration coefficients for Ecom2D4b1 in metres per square second.
#[pyclass(
    module = "sidereon._sidereon",
    name = "Ecom2D4b1Coefficients",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyEcom2D4b1Coefficients(forces::Ecom2D4b1Coefficients);

#[pymethods]
impl PyEcom2D4b1Coefficients {
    /// Build a coefficient set with omitted coefficients set to zero.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (b0=0.0, b1c=0.0, b1s=0.0, d0=0.0, d2c=0.0, d4c=0.0, d2s=0.0, d4s=0.0, y0=0.0))]
    fn new(
        b0: f64,
        b1c: f64,
        b1s: f64,
        d0: f64,
        d2c: f64,
        d4c: f64,
        d2s: f64,
        d4s: f64,
        y0: f64,
    ) -> Self {
        Self(forces::Ecom2D4b1Coefficients {
            b0,
            b1c,
            b1s,
            d0,
            d2c,
            d4c,
            d2s,
            d4s,
            y0,
        })
    }

    /// Coefficient b0 in metres per square second.
    #[getter]
    fn b0(&self) -> f64 {
        self.0.b0
    }

    /// Coefficient b1c in metres per square second.
    #[getter]
    fn b1c(&self) -> f64 {
        self.0.b1c
    }

    /// Coefficient b1s in metres per square second.
    #[getter]
    fn b1s(&self) -> f64 {
        self.0.b1s
    }

    /// Coefficient d0 in metres per square second.
    #[getter]
    fn d0(&self) -> f64 {
        self.0.d0
    }

    /// Coefficient d2c in metres per square second.
    #[getter]
    fn d2c(&self) -> f64 {
        self.0.d2c
    }

    /// Coefficient d4c in metres per square second.
    #[getter]
    fn d4c(&self) -> f64 {
        self.0.d4c
    }

    /// Coefficient d2s in metres per square second.
    #[getter]
    fn d2s(&self) -> f64 {
        self.0.d2s
    }

    /// Coefficient d4s in metres per square second.
    #[getter]
    fn d4s(&self) -> f64 {
        self.0.d4s
    }

    /// Coefficient y0 in metres per square second.
    #[getter]
    fn y0(&self) -> f64 {
        self.0.y0
    }

    /// Read a coefficient sequence in model order.
    #[staticmethod]
    fn from_array(values: [f64; forces::ECOM2_D4B1_COEFFICIENT_COUNT]) -> Self {
        Self(forces::Ecom2D4b1Coefficients::from_array(values))
    }

    /// Return the coefficients in model order.
    fn to_array(&self) -> [f64; forces::ECOM2_D4B1_COEFFICIENT_COUNT] {
        self.0.to_array()
    }

    /// Coefficient names in model order.
    #[classattr]
    const NAMES: [&'static str; forces::ECOM2_D4B1_COEFFICIENT_COUNT] =
        forces::Ecom2D4b1Coefficients::NAMES;
    /// Estimable parameter names in model order.
    #[classattr]
    const PARAMETER_NAMES: [&'static str; forces::ECOM2_D4B1_COEFFICIENT_COUNT] =
        forces::Ecom2D4b1Coefficients::PARAMETER_NAMES;
}

/// Sun-fixed axes, angular arguments, and lighting for ECOM.
#[pyclass(module = "sidereon._sidereon", name = "EcomGeometry", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyEcomGeometry(forces::EcomGeometry);

#[pymethods]
impl PyEcomGeometry {
    /// Sun-fixed D, Y and B axes.
    #[getter]
    fn dyb(&self) -> PyDybAxes {
        PyDybAxes(self.0.dyb)
    }

    /// Argument of latitude in radians, when defined.
    #[getter]
    fn argument_of_latitude_rad(&self) -> Option<f64> {
        self.0.argument_of_latitude_rad
    }

    /// Sun-relative argument of latitude in radians, when defined.
    #[getter]
    fn delta_u_rad(&self) -> Option<f64> {
        self.0.delta_u_rad
    }

    /// Fraction of incident sunlight.
    #[getter]
    fn lighting_ratio(&self) -> f64 {
        self.0.lighting_ratio
    }
}

/// Evaluate ECOM geometry from inertial satellite state and Sun position.
#[pyfunction]
fn ecom_geometry(
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
    sun_position_km: [f64; 3],
) -> PyResult<PyEcomGeometry> {
    forces::ecom_geometry(
        position_km.into(),
        velocity_km_s.into(),
        sun_position_km.into(),
    )
    .map(PyEcomGeometry)
    .map_err(propagation_error)
}

/// Evaluate ECOM geometry with an explicit Earth figure and optional Moon position.
#[pyfunction]
#[pyo3(signature = (position_km, velocity_km_s, sun_position_km, moon_position_km, earth_shadow_model))]
fn ecom_geometry_with_shadow(
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
    sun_position_km: [f64; 3],
    moon_position_km: Option<[f64; 3]>,
    earth_shadow_model: PyEarthShadowModel,
) -> PyResult<PyEcomGeometry> {
    forces::ecom_geometry_with_shadow(
        position_km.into(),
        velocity_km_s.into(),
        sun_position_km.into(),
        moon_position_km.map(Into::into),
        earth_shadow_model.into(),
    )
    .map(PyEcomGeometry)
    .map_err(propagation_error)
}

/// ECOM1 empirical solar radiation pressure.
#[pyclass(
    module = "sidereon._sidereon",
    name = "Ecom1SolarRadiationPressure",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyEcom1SolarRadiationPressure(pub(crate) forces::Ecom1SolarRadiationPressure);

#[pymethods]
impl PyEcom1SolarRadiationPressure {
    /// Configured ECOM1 acceleration coefficients.
    fn coefficients(&self) -> PyEcom1Coefficients {
        PyEcom1Coefficients(self.0.coefficients())
    }

    /// Build ECOM1 from its acceleration coefficients.
    #[new]
    fn new(coefficients: &PyEcom1Coefficients) -> PyResult<Self> {
        forces::Ecom1SolarRadiationPressure::new(coefficients.0)
            .map(Self)
            .map_err(propagation_error)
    }

    /// Select the eclipse lighting function.
    fn with_shadow_function(&self, function: PyShadowFunction) -> Self {
        Self(self.0.with_shadow_function(function.into()))
    }

    /// Select the Earth figure for occultation.
    fn with_earth_shadow_model(&self, model: PyEarthShadowModel) -> Self {
        Self(self.0.with_earth_shadow_model(model.into()))
    }

    /// Enable or disable lunar occultation.
    fn with_moon_eclipse(&self, enabled: bool) -> Self {
        Self(self.0.with_moon_eclipse(enabled))
    }

    /// Configured eclipse lighting function.
    fn shadow_function(&self) -> PyResult<PyShadowFunction> {
        self.0.shadow_function().try_into()
    }

    /// Configured Earth figure for occultation.
    fn earth_shadow_model(&self) -> PyEarthShadowModel {
        self.0.earth_shadow_model().into()
    }

    /// Whether the model includes lunar occultation.
    fn moon_eclipse_enabled(&self) -> bool {
        self.0.moon_eclipse_enabled()
    }

    /// Evaluate acceleration with explicit Sun position.
    fn acceleration_with_sun(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
    ) -> PyResult<[f64; 3]> {
        self.0
            .acceleration_with_sun(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
            )
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with explicit Sun and Moon positions.
    fn acceleration_with_sun_moon(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
        moon_position_km: [f64; 3],
    ) -> PyResult<[f64; 3]> {
        self.0
            .acceleration_with_sun_moon(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
                moon_position_km.into(),
            )
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration partials with explicit Sun position.
    fn acceleration_partials_with_sun(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
    ) -> PyResult<PyAccelerationPartials> {
        self.0
            .acceleration_partials_with_sun(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
            )
            .map(PyAccelerationPartials)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration partials with explicit Sun and Moon positions.
    fn acceleration_partials_with_sun_moon(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
        moon_position_km: [f64; 3],
    ) -> PyResult<PyAccelerationPartials> {
        self.0
            .acceleration_partials_with_sun_moon(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
                moon_position_km.into(),
            )
            .map(PyAccelerationPartials)
            .map_err(propagation_error)
    }

    /// Evaluate parameter partials with explicit Sun position.
    fn parameter_partials_with_sun(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
    ) -> PyResult<Vec<[f64; 3]>> {
        self.0
            .parameter_partials_with_sun(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
            )
            .map(|partials| partials.into_iter().map(Into::into).collect())
            .map_err(propagation_error)
    }

    /// Evaluate parameter partials with explicit Sun and Moon positions.
    fn parameter_partials_with_sun_moon(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
        moon_position_km: [f64; 3],
    ) -> PyResult<Vec<[f64; 3]>> {
        self.0
            .parameter_partials_with_sun_moon(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
                moon_position_km.into(),
            )
            .map(|partials| partials.into_iter().map(Into::into).collect())
            .map_err(propagation_error)
    }

    /// Return estimable parameters in model order.
    fn parameters(&self) -> Vec<PyForceParameter> {
        self.0
            .parameters()
            .into_iter()
            .map(PyForceParameter)
            .collect()
    }

    /// Return the number of estimable parameters.
    fn parameter_count(&self) -> usize {
        self.0.parameter_count()
    }

    /// Evaluate acceleration in kilometres per square second.
    #[pyo3(signature = (state, context=None))]
    fn acceleration(
        &self,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| self.0.acceleration(state.inner(), ctx))
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with replacement parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_with_parameters(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| {
            self.0
                .acceleration_with_parameters(&parameters, state.inner(), ctx)
        })
        .map(Into::into)
        .map_err(propagation_error)
    }

    /// Evaluate position and velocity derivatives at the supplied parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<PyAccelerationPartials> {
        with_context(context, |ctx| {
            self.0
                .acceleration_partials(&parameters, state.inner(), ctx)
        })
        .map(PyAccelerationPartials)
        .map_err(propagation_error)
    }

    /// Evaluate one acceleration derivative per parameter in model order.
    #[pyo3(signature = (parameters, state, context=None))]
    fn parameter_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<Vec<[f64; 3]>> {
        with_context(context, |ctx| {
            self.0.parameter_partials(&parameters, state.inner(), ctx)
        })
        .map(|partials| partials.into_iter().map(Into::into).collect())
        .map_err(propagation_error)
    }
}

/// ECOM2 empirical solar radiation pressure.
#[pyclass(
    module = "sidereon._sidereon",
    name = "Ecom2SolarRadiationPressure",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyEcom2SolarRadiationPressure(pub(crate) forces::Ecom2SolarRadiationPressure);

#[pymethods]
impl PyEcom2SolarRadiationPressure {
    /// Coefficient layout for the selected D and B harmonic orders.
    #[staticmethod]
    fn coefficient_count(n_d: usize, n_b: usize) -> Option<usize> {
        forces::Ecom2SolarRadiationPressure::coefficient_count(n_d, n_b)
    }

    /// Coefficient layout for the selected D and B harmonic orders.
    #[staticmethod]
    fn parameter_names(n_d: usize, n_b: usize) -> PyResult<Vec<String>> {
        forces::Ecom2SolarRadiationPressure::parameter_names(n_d, n_b).map_err(propagation_error)
    }

    /// Build the ECOM2 D2B1 coefficient layout.
    #[staticmethod]
    fn d2b1(coefficients: &PyEcom2D2b1Coefficients) -> PyResult<Self> {
        forces::Ecom2SolarRadiationPressure::d2b1(coefficients.0)
            .map(Self)
            .map_err(propagation_error)
    }

    /// Build the ECOM2 D4B1 coefficient layout.
    #[staticmethod]
    fn d4b1(coefficients: &PyEcom2D4b1Coefficients) -> PyResult<Self> {
        forces::Ecom2SolarRadiationPressure::d4b1(coefficients.0)
            .map(Self)
            .map_err(propagation_error)
    }

    /// The configured D harmonic order.
    fn n_d(&self) -> usize {
        self.0.n_d()
    }

    /// The configured B harmonic order.
    fn n_b(&self) -> usize {
        self.0.n_b()
    }

    /// Configured ECOM2 acceleration coefficients in model order.
    fn coefficients(&self) -> Vec<f64> {
        self.0.coefficients().to_vec()
    }

    /// Build ECOM2 from its acceleration coefficients.
    #[new]
    fn new(n_d: usize, n_b: usize, coefficients: Vec<f64>) -> PyResult<Self> {
        forces::Ecom2SolarRadiationPressure::new(n_d, n_b, coefficients)
            .map(Self)
            .map_err(propagation_error)
    }

    /// Select the eclipse lighting function.
    fn with_shadow_function(&self, function: PyShadowFunction) -> Self {
        Self(self.0.clone().with_shadow_function(function.into()))
    }

    /// Select the Earth figure for occultation.
    fn with_earth_shadow_model(&self, model: PyEarthShadowModel) -> Self {
        Self(self.0.clone().with_earth_shadow_model(model.into()))
    }

    /// Enable or disable lunar occultation.
    fn with_moon_eclipse(&self, enabled: bool) -> Self {
        Self(self.0.clone().with_moon_eclipse(enabled))
    }

    /// Configured eclipse lighting function.
    fn shadow_function(&self) -> PyResult<PyShadowFunction> {
        self.0.shadow_function().try_into()
    }

    /// Configured Earth figure for occultation.
    fn earth_shadow_model(&self) -> PyEarthShadowModel {
        self.0.earth_shadow_model().into()
    }

    /// Whether the model includes lunar occultation.
    fn moon_eclipse_enabled(&self) -> bool {
        self.0.moon_eclipse_enabled()
    }

    /// Evaluate acceleration with explicit Sun position.
    fn acceleration_with_sun(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
    ) -> PyResult<[f64; 3]> {
        self.0
            .acceleration_with_sun(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
            )
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with explicit Sun and Moon positions.
    fn acceleration_with_sun_moon(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
        moon_position_km: [f64; 3],
    ) -> PyResult<[f64; 3]> {
        self.0
            .acceleration_with_sun_moon(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
                moon_position_km.into(),
            )
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration partials with explicit Sun position.
    fn acceleration_partials_with_sun(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
    ) -> PyResult<PyAccelerationPartials> {
        self.0
            .acceleration_partials_with_sun(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
            )
            .map(PyAccelerationPartials)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration partials with explicit Sun and Moon positions.
    fn acceleration_partials_with_sun_moon(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
        moon_position_km: [f64; 3],
    ) -> PyResult<PyAccelerationPartials> {
        self.0
            .acceleration_partials_with_sun_moon(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
                moon_position_km.into(),
            )
            .map(PyAccelerationPartials)
            .map_err(propagation_error)
    }

    /// Evaluate parameter partials with explicit Sun position.
    fn parameter_partials_with_sun(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
    ) -> PyResult<Vec<[f64; 3]>> {
        self.0
            .parameter_partials_with_sun(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
            )
            .map(|partials| partials.into_iter().map(Into::into).collect())
            .map_err(propagation_error)
    }

    /// Evaluate parameter partials with explicit Sun and Moon positions.
    fn parameter_partials_with_sun_moon(
        &self,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        sun_position_km: [f64; 3],
        moon_position_km: [f64; 3],
    ) -> PyResult<Vec<[f64; 3]>> {
        self.0
            .parameter_partials_with_sun_moon(
                position_km.into(),
                velocity_km_s.into(),
                sun_position_km.into(),
                moon_position_km.into(),
            )
            .map(|partials| partials.into_iter().map(Into::into).collect())
            .map_err(propagation_error)
    }

    /// Return estimable parameters in model order.
    fn parameters(&self) -> Vec<PyForceParameter> {
        self.0
            .parameters()
            .into_iter()
            .map(PyForceParameter)
            .collect()
    }

    /// Return the number of estimable parameters.
    fn parameter_count(&self) -> usize {
        self.0.parameter_count()
    }

    /// Evaluate acceleration in kilometres per square second.
    #[pyo3(signature = (state, context=None))]
    fn acceleration(
        &self,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| self.0.acceleration(state.inner(), ctx))
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with replacement parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_with_parameters(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| {
            self.0
                .acceleration_with_parameters(&parameters, state.inner(), ctx)
        })
        .map(Into::into)
        .map_err(propagation_error)
    }

    /// Evaluate position and velocity derivatives at the supplied parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<PyAccelerationPartials> {
        with_context(context, |ctx| {
            self.0
                .acceleration_partials(&parameters, state.inner(), ctx)
        })
        .map(PyAccelerationPartials)
        .map_err(propagation_error)
    }

    /// Evaluate one acceleration derivative per parameter in model order.
    #[pyo3(signature = (parameters, state, context=None))]
    fn parameter_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<Vec<[f64; 3]>> {
        with_context(context, |ctx| {
            self.0.parameter_partials(&parameters, state.inner(), ctx)
        })
        .map(|partials| partials.into_iter().map(Into::into).collect())
        .map_err(propagation_error)
    }
}

/// Radial, along-track and cross-track axes.
#[pyclass(
    module = "sidereon._sidereon",
    name = "RtnAxis",
    eq,
    eq_int,
    frozen,
    ord
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum PyRtnAxis {
    /// Along the position vector.
    RADIAL,
    /// Along the transverse axis in the direction of orbital motion.
    ALONG_TRACK,
    /// Along the orbital angular momentum `r x v`.
    CROSS_TRACK,
}

impl From<PyRtnAxis> for forces::RtnAxis {
    fn from(value: PyRtnAxis) -> Self {
        match value {
            PyRtnAxis::RADIAL => Self::Radial,
            PyRtnAxis::ALONG_TRACK => Self::AlongTrack,
            PyRtnAxis::CROSS_TRACK => Self::CrossTrack,
        }
    }
}

impl From<forces::RtnAxis> for PyRtnAxis {
    fn from(value: forces::RtnAxis) -> Self {
        match value {
            forces::RtnAxis::Radial => Self::RADIAL,
            forces::RtnAxis::AlongTrack => Self::ALONG_TRACK,
            forces::RtnAxis::CrossTrack => Self::CROSS_TRACK,
        }
    }
}

#[pymethods]
impl PyRtnAxis {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }

    /// Every supported value in parameter order.
    #[classattr]
    #[allow(non_snake_case)]
    fn ALL() -> Vec<Self> {
        forces::RtnAxis::ALL.into_iter().map(Into::into).collect()
    }
}

/// Constant, cosine and sine empirical terms.
#[pyclass(
    module = "sidereon._sidereon",
    name = "EmpiricalHarmonic",
    eq,
    eq_int,
    frozen,
    ord
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[allow(non_camel_case_types)]
pub enum PyEmpiricalHarmonic {
    /// Constant.
    CONSTANT,
    /// Proportional to `cos u`.
    COSINE,
    /// Proportional to `sin u`.
    SINE,
}

impl From<PyEmpiricalHarmonic> for forces::EmpiricalHarmonic {
    fn from(value: PyEmpiricalHarmonic) -> Self {
        match value {
            PyEmpiricalHarmonic::CONSTANT => Self::Constant,
            PyEmpiricalHarmonic::COSINE => Self::Cosine,
            PyEmpiricalHarmonic::SINE => Self::Sine,
        }
    }
}

impl From<forces::EmpiricalHarmonic> for PyEmpiricalHarmonic {
    fn from(value: forces::EmpiricalHarmonic) -> Self {
        match value {
            forces::EmpiricalHarmonic::Constant => Self::CONSTANT,
            forces::EmpiricalHarmonic::Cosine => Self::COSINE,
            forces::EmpiricalHarmonic::Sine => Self::SINE,
        }
    }
}

#[pymethods]
impl PyEmpiricalHarmonic {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }

    /// Every supported value in parameter order.
    #[classattr]
    #[allow(non_snake_case)]
    fn ALL() -> Vec<Self> {
        forces::EmpiricalHarmonic::ALL
            .into_iter()
            .map(Into::into)
            .collect()
    }
}

/// Empirical accelerations resolved in the radial, along-track and cross-track frame.
#[pyclass(
    module = "sidereon._sidereon",
    name = "EmpiricalRtnAcceleration",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyEmpiricalRtnAcceleration(pub(crate) forces::EmpiricalRtnAcceleration);

#[pymethods]
impl PyEmpiricalRtnAcceleration {
    /// Build an empty empirical acceleration.
    #[new]
    fn new() -> Self {
        Self(forces::EmpiricalRtnAcceleration::new())
    }

    /// Build with the three constant terms set to zero.
    #[staticmethod]
    fn constant_terms() -> Self {
        Self(forces::EmpiricalRtnAcceleration::constant_terms())
    }

    /// Build with all nine coefficients set to zero.
    #[staticmethod]
    fn all_terms() -> Self {
        Self(forces::EmpiricalRtnAcceleration::all_terms())
    }

    /// Select empirical terms with zero initial coefficients.
    #[staticmethod]
    fn with_zero_terms(terms: Vec<(PyRtnAxis, PyEmpiricalHarmonic)>) -> Self {
        let terms: Vec<_> = terms
            .into_iter()
            .map(|(axis, harmonic)| (axis.into(), harmonic.into()))
            .collect();
        Self(forces::EmpiricalRtnAcceleration::with_zero_terms(&terms))
    }

    /// Select an empirical term and its coefficient.
    fn with_term(
        &self,
        axis: PyRtnAxis,
        harmonic: PyEmpiricalHarmonic,
        coefficient_km_s2: f64,
    ) -> PyResult<Self> {
        self.0
            .with_term(axis.into(), harmonic.into(), coefficient_km_s2)
            .map(Self)
            .map_err(propagation_error)
    }

    /// Configured coefficient, if the term is enabled.
    fn coefficient_km_s2(&self, axis: PyRtnAxis, harmonic: PyEmpiricalHarmonic) -> Option<f64> {
        self.0.coefficient_km_s2(axis.into(), harmonic.into())
    }

    /// Enabled terms in model parameter order.
    fn terms(&self) -> Vec<(PyRtnAxis, PyEmpiricalHarmonic, f64)> {
        self.0
            .terms()
            .into_iter()
            .map(|(axis, harmonic, value)| (axis.into(), harmonic.into(), value))
            .collect()
    }

    /// Stable estimation name for an RTN term.
    #[staticmethod]
    fn parameter_name(axis: PyRtnAxis, harmonic: PyEmpiricalHarmonic) -> &'static str {
        forces::EmpiricalRtnAcceleration::parameter_name(axis.into(), harmonic.into())
    }

    /// Return estimable parameters in model order.
    fn parameters(&self) -> Vec<PyForceParameter> {
        self.0
            .parameters()
            .into_iter()
            .map(PyForceParameter)
            .collect()
    }

    /// Return the number of estimable parameters.
    fn parameter_count(&self) -> usize {
        self.0.parameter_count()
    }

    /// Evaluate acceleration in kilometres per square second.
    #[pyo3(signature = (state, context=None))]
    fn acceleration(
        &self,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| self.0.acceleration(state.inner(), ctx))
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with replacement parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_with_parameters(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| {
            self.0
                .acceleration_with_parameters(&parameters, state.inner(), ctx)
        })
        .map(Into::into)
        .map_err(propagation_error)
    }

    /// Evaluate position and velocity derivatives at the supplied parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<PyAccelerationPartials> {
        with_context(context, |ctx| {
            self.0
                .acceleration_partials(&parameters, state.inner(), ctx)
        })
        .map(PyAccelerationPartials)
        .map_err(propagation_error)
    }

    /// Evaluate one acceleration derivative per parameter in model order.
    #[pyo3(signature = (parameters, state, context=None))]
    fn parameter_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<Vec<[f64; 3]>> {
        with_context(context, |ctx| {
            self.0.parameter_partials(&parameters, state.inner(), ctx)
        })
        .map(|partials| partials.into_iter().map(Into::into).collect())
        .map_err(propagation_error)
    }
}

/// Cannonball radiation pressure due to Earth albedo and thermal emission.
#[pyclass(
    module = "sidereon._sidereon",
    name = "EarthRadiationPressure",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyEarthRadiationPressure(pub(crate) forces::EarthRadiationPressure);

#[pymethods]
impl PyEarthRadiationPressure {
    /// Build Earth radiation pressure from reflectivity and area-to-mass ratio.
    #[new]
    fn new(cr: f64, area_to_mass_m2_kg: f64) -> PyResult<Self> {
        forces::EarthRadiationPressure::new(cr, area_to_mass_m2_kg)
            .map(Self)
            .map_err(propagation_error)
    }

    /// Reflectivity coefficient.
    #[getter]
    fn cr(&self) -> f64 {
        self.0.cr()
    }

    /// Area-to-mass ratio in square metres per kilogram.
    #[getter]
    fn area_to_mass_m2_kg(&self) -> f64 {
        self.0.area_to_mass_m2_kg()
    }

    /// Solar radiation pressure at one astronomical unit, in newtons per square metre.
    #[getter]
    fn pressure_n_m2(&self) -> f64 {
        self.0.pressure_n_m2()
    }

    /// Earth radius in kilometres.
    #[getter]
    fn earth_radius_km(&self) -> f64 {
        self.0.earth_radius_km()
    }

    /// Astronomical unit in kilometres.
    #[getter]
    fn au_km(&self) -> f64 {
        self.0.au_km()
    }

    /// Enable or disable lunar occultation.
    fn with_moon_eclipse(&self, enabled: bool) -> Self {
        Self(self.0.with_moon_eclipse(enabled))
    }

    /// Whether the model includes lunar occultation.
    fn moon_eclipse_enabled(&self) -> bool {
        self.0.moon_eclipse_enabled()
    }

    /// Evaluate Earth radiation pressure with explicit Sun and Moon positions.
    fn acceleration_with_sun_moon(
        &self,
        state: &PyCartesianState,
        sun_pos_km: [f64; 3],
        moon_pos_km: [f64; 3],
    ) -> PyResult<[f64; 3]> {
        self.0
            .acceleration_with_sun_moon(state.inner(), sun_pos_km.into(), moon_pos_km.into())
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Return estimable parameters in model order.
    fn parameters(&self) -> Vec<PyForceParameter> {
        self.0
            .parameters()
            .into_iter()
            .map(PyForceParameter)
            .collect()
    }

    /// Return the number of estimable parameters.
    fn parameter_count(&self) -> usize {
        self.0.parameter_count()
    }

    /// Evaluate acceleration in kilometres per square second.
    #[pyo3(signature = (state, context=None))]
    fn acceleration(
        &self,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| self.0.acceleration(state.inner(), ctx))
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with replacement parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_with_parameters(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| {
            self.0
                .acceleration_with_parameters(&parameters, state.inner(), ctx)
        })
        .map(Into::into)
        .map_err(propagation_error)
    }

    /// Evaluate position and velocity derivatives at the supplied parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<PyAccelerationPartials> {
        with_context(context, |ctx| {
            self.0
                .acceleration_partials(&parameters, state.inner(), ctx)
        })
        .map(PyAccelerationPartials)
        .map_err(propagation_error)
    }

    /// Evaluate one acceleration derivative per parameter in model order.
    #[pyo3(signature = (parameters, state, context=None))]
    fn parameter_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<Vec<[f64; 3]>> {
        with_context(context, |ctx| {
            self.0.parameter_partials(&parameters, state.inner(), ctx)
        })
        .map(|partials| partials.into_iter().map(Into::into).collect())
        .map_err(propagation_error)
    }
}

/// Sum of concrete force models in specified order.
#[pyclass(module = "sidereon._sidereon", name = "CompositeForceModel")]
#[derive(Clone)]
pub struct PyCompositeForceModel(pub(crate) Arc<forces::CompositeForceModel>);

#[pymethods]
impl PyCompositeForceModel {
    /// Build an additive model from a sequence of force objects.
    #[new]
    fn new(models: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let mut result = forces::CompositeForceModel::new();
        for model in models {
            result.add(Box::new(force_object(&model)?));
        }
        Ok(Self(Arc::new(result)))
    }

    /// Return estimable parameters in model order.
    fn parameters(&self) -> Vec<PyForceParameter> {
        self.0
            .parameters()
            .into_iter()
            .map(PyForceParameter)
            .collect()
    }

    /// Return the number of estimable parameters.
    fn parameter_count(&self) -> usize {
        self.0.parameter_count()
    }

    /// Evaluate acceleration in kilometres per square second.
    #[pyo3(signature = (state, context=None))]
    fn acceleration(
        &self,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| self.0.acceleration(state.inner(), ctx))
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with replacement parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_with_parameters(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| {
            self.0
                .acceleration_with_parameters(&parameters, state.inner(), ctx)
        })
        .map(Into::into)
        .map_err(propagation_error)
    }

    /// Evaluate position and velocity derivatives at the supplied parameter values.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<PyAccelerationPartials> {
        with_context(context, |ctx| {
            self.0
                .acceleration_partials(&parameters, state.inner(), ctx)
        })
        .map(PyAccelerationPartials)
        .map_err(propagation_error)
    }

    /// Evaluate one acceleration derivative per parameter in model order.
    #[pyo3(signature = (parameters, state, context=None))]
    fn parameter_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<Vec<[f64; 3]>> {
        with_context(context, |ctx| {
            self.0.parameter_partials(&parameters, state.inner(), ctx)
        })
        .map(|partials| partials.into_iter().map(Into::into).collect())
        .map_err(propagation_error)
    }
}

pub(crate) fn force_object(value: &Bound<'_, PyAny>) -> PyResult<Arc<dyn ForceModel>> {
    if let Ok(model) = value.extract::<PyForceModel>() {
        return model.core_force().map(Arc::from).map_err(propagation_error);
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyDragParameters>>() {
        return Ok(Arc::new(model.inner().to_force()));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyEcom1SolarRadiationPressure>>() {
        return Ok(Arc::new(model.0));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyEcom2SolarRadiationPressure>>() {
        return Ok(Arc::new(model.0.clone()));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyEmpiricalRtnAcceleration>>() {
        return Ok(Arc::new(model.0));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyEarthRadiationPressure>>() {
        return Ok(Arc::new(model.0));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyCompositeForceModel>>() {
        return Ok(model.0.clone());
    }
    if let Ok(model) = value.extract::<PyRef<'_, PySolarRadiationPressure>>() {
        return Ok(Arc::new(model.inner));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PyDragForce>>() {
        return Ok(Arc::new(model.inner));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PySourcedDragForce>>() {
        return Ok(Arc::new(model.inner.clone()));
    }
    if let Ok(model) = value.extract::<PyRef<'_, PySphericalHarmonicGravity>>() {
        return Ok(model.0.clone());
    }
    Err(PyTypeError::new_err("expected a concrete force model"))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyForceParameter>()?;
    m.add_class::<PyAccelerationPartials>()?;
    m.add("ECOM1_COEFFICIENT_COUNT", forces::ECOM1_COEFFICIENT_COUNT)?;
    m.add_class::<PyEcom1Coefficients>()?;
    m.add(
        "ECOM2_D2B1_COEFFICIENT_COUNT",
        forces::ECOM2_D2B1_COEFFICIENT_COUNT,
    )?;
    m.add_class::<PyEcom2D2b1Coefficients>()?;
    m.add(
        "ECOM2_D4B1_COEFFICIENT_COUNT",
        forces::ECOM2_D4B1_COEFFICIENT_COUNT,
    )?;
    m.add_class::<PyEcom2D4b1Coefficients>()?;
    m.add_class::<PyEcomGeometry>()?;
    m.add_function(wrap_pyfunction!(ecom_geometry, m)?)?;
    m.add_function(wrap_pyfunction!(ecom_geometry_with_shadow, m)?)?;
    m.add_class::<PyEcom1SolarRadiationPressure>()?;
    m.add_class::<PyEcom2SolarRadiationPressure>()?;
    m.add_class::<PyRtnAxis>()?;
    m.add_class::<PyEmpiricalHarmonic>()?;
    m.add_class::<PyEmpiricalRtnAcceleration>()?;
    m.add_class::<PyEarthRadiationPressure>()?;
    m.add_class::<PyCompositeForceModel>()?;
    Ok(())
}
