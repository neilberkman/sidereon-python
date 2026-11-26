//! Estimation of initial orbit state and selected force-model parameters.

use numpy::PyArray2;
use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::covariance::RtnFrameError;
use sidereon_core::astro::math::least_squares::SolveError as LeastSquaresError;
use sidereon_core::astro::propagator::IntegratorKind;
use sidereon_core::astro::time::ValidityMode;
use sidereon_core::orbit_determination::{self as core, OrbitFitError, Ut1ProviderRole};

use crate::earth_orientation::{PyIers2010EarthOrientationProvider, PyPropagationContext};
use crate::ephemeris::{parse_sat, PySp3};
use crate::force_models::force_object;
use crate::forces::PyDragParameters;
use crate::geometry_quality::PyGeometryQuality;
use crate::orbit_determination::PyOrbitResidualStats;
use crate::propagation::{PyForceModelKind, PyIntegrator};
use crate::relative::PyCartesianState;
use crate::rinex_clock::PyClockInstant;
use crate::space_weather::PySpaceWeatherTable;
use crate::variational::{matrix_to_array, PyIntegratorOptions, PyVariationalEpoch};
use crate::{
    binding_error, binding_error_detail, degrade_reason_label, PyValidityMode, SolveError,
};

create_exception!(
    _sidereon,
    OrbitParameterFitError,
    SolveError,
    "A force-parameter orbit fit failure."
);

fn orbit_parameter_fit_error(error: core::OrbitParameterFitError) -> PyErr {
    let exception = OrbitParameterFitError::new_err(error.to_string());
    match error {
        core::OrbitParameterFitError::Fit(error) => orbit_fit_error(exception, error),
        core::OrbitParameterFitError::InvalidParameter { name, reason } => {
            binding_error(exception, "invalid_parameter", |detail| {
                detail.set_item("name", name)?;
                detail.set_item("reason", reason)
            })
        }
        core::OrbitParameterFitError::UnobservableParameter { satellite, name } => {
            binding_error(exception, "unobservable_parameter", |detail| {
                detail.set_item("satellite", satellite.to_string())?;
                detail.set_item("name", name)
            })
        }
        core::OrbitParameterFitError::SingularNormalEquations {
            satellite,
            condition_number,
            weakest,
        } => binding_error(exception, "singular_normal_equations", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("condition_number", condition_number)?;
            detail.set_item("weakest", weakest)
        }),
        core::OrbitParameterFitError::IllConditionedNormalEquations {
            satellite,
            condition_number,
            limit,
            weakest,
        } => binding_error(exception, "ill_conditioned_normal_equations", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("condition_number", condition_number)?;
            detail.set_item("limit", limit)?;
            detail.set_item("weakest", weakest)
        }),
        core::OrbitParameterFitError::NumericalFailure {
            satellite,
            quantity,
        } => binding_error(exception, "numerical_failure", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("quantity", quantity)
        }),
        core::OrbitParameterFitError::DidNotConverge {
            satellite,
            iterations,
            last_step_sigmas,
        } => binding_error(exception, "did_not_converge", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("iterations", iterations)?;
            detail.set_item("last_step_sigmas", last_step_sigmas)
        }),
    }
}

fn orbit_fit_error(exception: PyErr, source: OrbitFitError) -> PyErr {
    match source {
        OrbitFitError::InvalidOption { field, reason } => {
            binding_error(exception, "invalid_option", |detail| {
                detail.set_item("field", field)?;
                detail.set_item("reason", reason)
            })
        }
        OrbitFitError::TooFewSamples {
            satellite,
            got,
            required,
        } => binding_error(exception, "too_few_samples", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("got", got)?;
            detail.set_item("required", required)
        }),
        OrbitFitError::NonMonotonicEpochs { satellite } => {
            binding_error(exception, "non_monotonic_epochs", |detail| {
                detail.set_item("satellite", satellite.to_string())
            })
        }
        OrbitFitError::InvalidEpoch { satellite, reason } => {
            binding_error(exception, "invalid_epoch", |detail| {
                detail.set_item("satellite", satellite.to_string())?;
                detail.set_item("reason", reason)
            })
        }
        OrbitFitError::InvalidObservation { satellite, reason } => {
            binding_error(exception, "invalid_observation", |detail| {
                detail.set_item("satellite", satellite.to_string())?;
                detail.set_item("reason", reason)
            })
        }
        OrbitFitError::Ut1OutsideCoverage(reason) => {
            binding_error(exception, "ut1_outside_coverage", |detail| {
                detail.set_item("reason", degrade_reason_label(reason))
            })
        }
        OrbitFitError::Frame { satellite, source } => binding_error(exception, "frame", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item(
                "source",
                crate::earth_orientation::frame_error(source)
                    .value(detail.py())
                    .getattr("detail")?,
            )
        }),
        OrbitFitError::Propagation { satellite, source } => {
            binding_error(exception, "propagation", |detail| {
                detail.set_item("satellite", satellite.to_string())?;
                detail.set_item(
                    "source",
                    crate::force_models::propagation_error(source)
                        .value(detail.py())
                        .getattr("detail")?,
                )
            })
        }
        OrbitFitError::EmptySelection => binding_error(exception, "empty_selection", |_| Ok(())),
        OrbitFitError::MixedTimeScales => binding_error(exception, "mixed_time_scales", |_| Ok(())),
        OrbitFitError::LeastSquares { satellite, source } => {
            binding_error(exception, "least_squares", |detail| {
                let source = match source {
                    LeastSquaresError::SingularJacobian => {
                        binding_error_detail(detail.py(), "singular_jacobian", |_| Ok(()))
                    }
                    LeastSquaresError::InvalidInput { field, reason } => {
                        binding_error_detail(detail.py(), "invalid_input", |source| {
                            source.set_item("field", field)?;
                            source.set_item("reason", reason)
                        })
                    }
                }?;
                detail.set_item("satellite", satellite.to_string())?;
                detail.set_item("source", source)
            })
        }
        OrbitFitError::SingularGeometry {
            satellite,
            geometry_quality,
        } => binding_error(exception, "singular_geometry", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item(
                "geometry_quality",
                PyGeometryQuality::from(geometry_quality),
            )
        }),
        OrbitFitError::DidNotConverge {
            satellite,
            iterations,
        } => binding_error(exception, "did_not_converge", |detail| {
            detail.set_item("satellite", satellite.to_string())?;
            detail.set_item("iterations", iterations)
        }),
        OrbitFitError::Ut1ValidityMismatch {
            fit,
            provider,
            provider_mode,
        } => binding_error(exception, "ut1_validity_mismatch", |detail| {
            let policy = |mode| match mode {
                ValidityMode::Strict => PyValidityMode::STRICT,
                ValidityMode::Permissive => PyValidityMode::PERMISSIVE,
            };
            let role = match provider {
                Ut1ProviderRole::Orientation => "orientation",
                Ut1ProviderRole::Propagation => "propagation",
            };
            detail.set_item("fit", policy(fit))?;
            detail.set_item("provider", role)?;
            detail.set_item("provider_mode", policy(provider_mode))
        }),
        OrbitFitError::RtnFrame { satellite, reason } => {
            binding_error(exception, "rtn_frame", |detail| {
                let py = detail.py();
                let reason = match reason {
                    RtnFrameError::InvalidInput { field, reason } => {
                        binding_error_detail(py, "invalid_input", |source| {
                            source.set_item("field", field)?;
                            source.set_item("reason", reason)
                        })
                    }
                    RtnFrameError::ZeroPosition => {
                        binding_error_detail(py, "zero_position", |_| Ok(()))
                    }
                    RtnFrameError::ParallelPositionVelocity => {
                        binding_error_detail(py, "parallel_position_velocity", |_| Ok(()))
                    }
                }?;
                detail.set_item("satellite", satellite.to_string())?;
                detail.set_item("reason", reason)
            })
        }
    }
}
/// Observed position in the propagation's inertial frame.
#[pyclass(
    module = "sidereon._sidereon",
    name = "InertialPositionSample",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyInertialPositionSample(core::InertialPositionSample);

#[pymethods]
impl PyInertialPositionSample {
    /// Build an inertial position observation.
    #[new]
    fn new(epoch: PyClockInstant, position_gcrf_m: [f64; 3]) -> Self {
        Self(core::InertialPositionSample {
            epoch: epoch.to_core(),
            position_gcrf_m,
        })
    }

    /// Observation epoch, in any supported time scale.
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.0.epoch)
    }

    /// GCRF position in metres.
    #[getter]
    fn position_gcrf_m(&self) -> [f64; 3] {
        self.0.position_gcrf_m
    }
}

/// An estimated force-model parameter and its initial value.
#[pyclass(
    module = "sidereon._sidereon",
    name = "EstimatedForceParameter",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyEstimatedForceParameter(core::EstimatedForceParameter);

#[pymethods]
impl PyEstimatedForceParameter {
    /// Parameter name.
    #[getter]
    fn name(&self) -> String {
        self.0.name.clone()
    }

    /// Value before estimation.
    #[getter]
    fn initial_value(&self) -> f64 {
        self.0.initial_value
    }

    /// Estimate in the force model's parameter units.
    #[getter]
    fn value(&self) -> f64 {
        self.0.value
    }
}

/// Observed minus fitted position in the satellite's radial, along-track and cross-track frame.
#[pyclass(module = "sidereon._sidereon", name = "OrbitRtnResidual", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitRtnResidual(core::OrbitRtnResidual);

#[pymethods]
impl PyOrbitRtnResidual {
    /// Observation epoch in TDB seconds from J2000.
    #[getter]
    fn epoch_tdb_j2000_s(&self) -> f64 {
        self.0.epoch_tdb_j2000_s
    }

    /// TT seconds since the fitted initial epoch.
    #[getter]
    fn elapsed_tt_s(&self) -> f64 {
        self.0.elapsed_tt_s
    }

    /// Radial residual in metres.
    #[getter]
    fn radial_m(&self) -> f64 {
        self.0.radial_m
    }

    /// Along-track residual in metres.
    #[getter]
    fn along_m(&self) -> f64 {
        self.0.along_m
    }

    /// Cross-track residual in metres.
    #[getter]
    fn cross_m(&self) -> f64 {
        self.0.cross_m
    }

    /// Time coordinates for propagating the fitted state to this observation.
    fn variational_epoch(&self) -> PyVariationalEpoch {
        PyVariationalEpoch(self.0.variational_epoch())
    }
}

/// Estimated initial state, selected force parameters, covariance and post-fit residuals.
#[pyclass(module = "sidereon._sidereon", name = "OrbitParameterFit", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitParameterFit(core::OrbitParameterFit);

#[pymethods]
impl PyOrbitParameterFit {
    /// Satellite identifier.
    #[getter]
    fn satellite(&self) -> String {
        self.0.satellite.to_string()
    }

    /// Estimated initial state at the first observation.
    #[getter]
    fn initial_state(&self) -> PyCartesianState {
        PyCartesianState::from_inner(self.0.initial_state)
    }

    /// Estimated force parameters in covariance order.
    #[getter]
    fn force_parameters(&self) -> Vec<PyEstimatedForceParameter> {
        self.0
            .force_parameters
            .iter()
            .cloned()
            .map(PyEstimatedForceParameter)
            .collect()
    }

    /// Parameter names in covariance order.
    #[getter]
    fn parameter_names(&self) -> Vec<String> {
        self.0.parameter_names.clone()
    }

    /// Estimated or unbounded covariance.
    #[getter]
    fn covariance(&self) -> PyOrbitParameterCovariance {
        PyOrbitParameterCovariance(self.0.covariance.clone())
    }

    /// Residual at each observation epoch.
    #[getter]
    fn residuals(&self) -> Vec<PyOrbitRtnResidual> {
        self.0
            .residuals
            .iter()
            .copied()
            .map(PyOrbitRtnResidual)
            .collect()
    }

    /// RMS residuals and sample counts.
    #[getter]
    fn residual_stats(&self) -> PyOrbitResidualStats {
        self.0.residual_stats.into()
    }

    /// Three-dimensional RMS residual of the initial guess in metres.
    #[getter]
    fn seed_rms_3d_m(&self) -> f64 {
        self.0.seed_rms_3d_m
    }

    /// Number of Gauss-Newton iterations.
    #[getter]
    fn iterations(&self) -> usize {
        self.0.iterations
    }

    /// Condition number of the column-scaled normal equations.
    #[getter]
    fn condition_number(&self) -> f64 {
        self.0.condition_number
    }

    /// Estimated standard deviation for a named parameter, if available.
    fn sigma(&self, name: &str) -> Option<f64> {
        self.0.sigma(name)
    }

    /// Estimated correlation between two named parameters, if available.
    fn correlation(&self, first: &str, second: &str) -> Option<f64> {
        self.0.correlation(first, second)
    }

    /// Estimate of the named force parameter, if present.
    fn force_parameter(&self, name: &str) -> Option<PyEstimatedForceParameter> {
        self.0
            .force_parameter(name)
            .cloned()
            .map(PyEstimatedForceParameter)
    }
}

/// Scaled covariance of an orbit-parameter fit, or an unbounded estimate.
#[pyclass(
    module = "sidereon._sidereon",
    name = "OrbitParameterCovariance",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyOrbitParameterCovariance(core::OrbitParameterCovariance);

#[pymethods]
impl PyOrbitParameterCovariance {
    /// Result of a fit with zero residual degrees of freedom.
    #[classattr]
    const UNBOUNDED: Self = Self(core::OrbitParameterCovariance::Unbounded);

    /// Whether finite covariance data are available.
    #[getter]
    fn is_estimated(&self) -> bool {
        matches!(self.0, core::OrbitParameterCovariance::Estimated { .. })
    }
    /// Formal covariance scaled by the variance factor.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        match &self.0 {
            core::OrbitParameterCovariance::Estimated { matrix, .. } => {
                Some(matrix_to_array(py, matrix))
            }
            core::OrbitParameterCovariance::Unbounded => None,
        }
    }
    /// Matrix of parameter correlation coefficients.
    #[getter]
    fn correlation<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        match &self.0 {
            core::OrbitParameterCovariance::Estimated { correlation, .. } => {
                Some(matrix_to_array(py, correlation))
            }
            core::OrbitParameterCovariance::Unbounded => None,
        }
    }
    /// A-posteriori variance factor.
    #[getter]
    fn variance_factor(&self) -> Option<f64> {
        match self.0 {
            core::OrbitParameterCovariance::Estimated {
                variance_factor, ..
            } => Some(variance_factor),
            core::OrbitParameterCovariance::Unbounded => None,
        }
    }

    /// Residual components minus estimated parameters.
    #[getter]
    fn degrees_of_freedom(&self) -> Option<usize> {
        match self.0 {
            core::OrbitParameterCovariance::Estimated {
                degrees_of_freedom, ..
            } => Some(degrees_of_freedom),
            core::OrbitParameterCovariance::Unbounded => None,
        }
    }
}

/// Force composition, propagation context, and estimation controls.
#[pyclass(module = "sidereon._sidereon", name = "OrbitParameterFitOptions")]
pub struct PyOrbitParameterFitOptions {
    inner: core::OrbitParameterFitOptions,
    space_weather: Option<Py<PySpaceWeatherTable>>,
    additional_forces: Vec<Py<PyAny>>,
    propagation_context: PyPropagationContext,
}

#[pymethods]
impl PyOrbitParameterFitOptions {
    /// Configure orbit fitting and which named force parameters to estimate.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (*, force_model=None, drag=None, space_weather=None, additional_forces=Vec::new(), estimated_parameters=Vec::new(), integrator=None, integrator_options=None, propagation_context=None, observation_sigma_m=None, max_iterations=None, convergence_threshold=None, max_condition_number=None, min_ledger_samples=None))]
    fn new(
        py: Python<'_>,
        force_model: Option<&PyForceModelKind>,
        drag: Option<&PyDragParameters>,
        space_weather: Option<Py<PySpaceWeatherTable>>,
        additional_forces: Vec<Py<PyAny>>,
        estimated_parameters: Vec<String>,
        integrator: Option<PyIntegrator>,
        integrator_options: Option<&PyIntegratorOptions>,
        propagation_context: Option<&PyPropagationContext>,
        observation_sigma_m: Option<f64>,
        max_iterations: Option<usize>,
        convergence_threshold: Option<f64>,
        max_condition_number: Option<f64>,
        min_ledger_samples: Option<usize>,
    ) -> PyResult<Self> {
        let mut inner = core::OrbitParameterFitOptions::default();
        if let Some(force_model) = force_model {
            inner.force_model = force_model.inner();
        }
        inner.drag = drag.map(PyDragParameters::inner);
        inner.space_weather = space_weather
            .as_ref()
            .map(|table| table.borrow(py).source());
        inner.additional_forces = additional_forces
            .iter()
            .map(|force| force_object(force.bind(py)))
            .collect::<PyResult<_>>()?;
        inner.estimated_parameters = estimated_parameters;
        if let Some(integrator) = integrator {
            inner.integrator = integrator.into();
        }
        if let Some(options) = integrator_options {
            inner.integrator_options = options.0;
        }
        let propagation_context = propagation_context.cloned().unwrap_or_default();
        inner.propagation_context = propagation_context.inner.clone();
        if let Some(observation_sigma_m) = observation_sigma_m {
            inner.observation_sigma_m = observation_sigma_m;
        }
        if let Some(max_iterations) = max_iterations {
            inner.max_iterations = max_iterations;
        }
        if let Some(convergence_threshold) = convergence_threshold {
            inner.convergence_threshold = convergence_threshold;
        }
        if let Some(max_condition_number) = max_condition_number {
            inner.max_condition_number = max_condition_number;
        }
        if let Some(min_ledger_samples) = min_ledger_samples {
            inner.min_ledger_samples = min_ledger_samples;
        }
        Ok(Self {
            inner,
            space_weather,
            additional_forces,
            propagation_context,
        })
    }
    /// Built-in force configuration.
    #[getter]
    fn force_model(&self) -> PyForceModelKind {
        PyForceModelKind {
            inner: self.inner.force_model,
        }
    }

    /// Optional atmospheric drag parameters.
    #[getter]
    fn drag(&self) -> Option<PyDragParameters> {
        self.inner.drag.map(|inner| PyDragParameters { inner })
    }

    /// Optional space weather series.
    #[getter]
    fn space_weather(&self, py: Python<'_>) -> Option<Py<PySpaceWeatherTable>> {
        self.space_weather.as_ref().map(|table| table.clone_ref(py))
    }

    /// Additional forces in evaluation order.
    #[getter]
    fn additional_forces(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.additional_forces
            .iter()
            .map(|force| force.clone_ref(py))
            .collect()
    }

    /// Names of the selected force parameters in covariance order.
    #[getter]
    fn estimated_parameters(&self) -> Vec<String> {
        self.inner.estimated_parameters.clone()
    }

    /// Numerical integrator.
    #[getter]
    fn integrator(&self) -> PyIntegrator {
        match self.inner.integrator {
            IntegratorKind::Dp54 => PyIntegrator::DP54,
            IntegratorKind::Rk4 => PyIntegrator::RK4,
        }
    }

    /// Numerical integrator controls.
    #[getter]
    fn integrator_options(&self) -> PyIntegratorOptions {
        PyIntegratorOptions(self.inner.integrator_options)
    }

    /// Context used for force evaluation.
    #[getter]
    fn propagation_context(&self) -> PyPropagationContext {
        self.propagation_context.clone()
    }

    /// A-priori standard deviation per observed position component.
    #[getter]
    fn observation_sigma_m(&self) -> f64 {
        self.inner.observation_sigma_m
    }

    /// Maximum number of fit iterations.
    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    /// Convergence threshold in formal standard deviations.
    #[getter]
    fn convergence_threshold(&self) -> f64 {
        self.inner.convergence_threshold
    }

    /// Maximum condition number for the column-scaled normal equations.
    #[getter]
    fn max_condition_number(&self) -> f64 {
        self.inner.max_condition_number
    }

    /// Minimum sample count for residual statistics.
    #[getter]
    fn min_ledger_samples(&self) -> usize {
        self.inner.min_ledger_samples
    }
}

/// Fit initial state and selected force parameters.
#[pyfunction]
#[pyo3(signature = (samples, satellite, initial_state, options))]
fn fit_inertial_orbit_parameters(
    samples: Vec<PyInertialPositionSample>,
    satellite: &str,
    initial_state: Option<&PyCartesianState>,
    options: &PyOrbitParameterFitOptions,
) -> PyResult<PyOrbitParameterFit> {
    let satellite = parse_sat(satellite)?;
    let initial_state = initial_state.map(|state| *state.inner());
    let samples: Vec<_> = samples.into_iter().map(|sample| sample.0).collect();
    core::fit_inertial_orbit_parameters(&samples, satellite, initial_state, &options.inner)
        .map(PyOrbitParameterFit)
        .map_err(orbit_parameter_fit_error)
}

/// Fit initial state and selected force parameters with an explicit UT1 coverage policy.
#[pyfunction]
#[pyo3(signature = (samples, satellite, initial_state, options, mode))]
fn fit_inertial_orbit_parameters_with_validity(
    samples: Vec<PyInertialPositionSample>,
    satellite: &str,
    initial_state: Option<&PyCartesianState>,
    options: &PyOrbitParameterFitOptions,
    mode: PyValidityMode,
) -> PyResult<(PyOrbitParameterFit, Option<&'static str>)> {
    let satellite = parse_sat(satellite)?;
    let initial_state = initial_state.map(|state| *state.inner());
    let samples: Vec<_> = samples.into_iter().map(|sample| sample.0).collect();
    core::fit_inertial_orbit_parameters_with_validity(
        &samples,
        satellite,
        initial_state,
        &options.inner,
        mode.into(),
    )
    .map(|result| {
        (
            PyOrbitParameterFit(result.value),
            result.degraded.map(degrade_reason_label),
        )
    })
    .map_err(orbit_parameter_fit_error)
}

/// Fit initial state and selected force parameters.
#[pyfunction]
#[pyo3(signature = (product, satellite, orientation_provider, initial_state, options))]
fn fit_sp3_ecef_orbit_parameters(
    product: &PySp3,
    satellite: &str,
    orientation_provider: &PyIers2010EarthOrientationProvider,
    initial_state: Option<&PyCartesianState>,
    options: &PyOrbitParameterFitOptions,
) -> PyResult<PyOrbitParameterFit> {
    let satellite = parse_sat(satellite)?;
    let initial_state = initial_state.map(|state| *state.inner());
    core::fit_sp3_ecef_orbit_parameters(
        &product.inner,
        satellite,
        orientation_provider.0.as_ref(),
        initial_state,
        &options.inner,
    )
    .map(PyOrbitParameterFit)
    .map_err(orbit_parameter_fit_error)
}

/// Fit initial state and selected force parameters with an explicit UT1 coverage policy.
#[pyfunction]
#[pyo3(signature = (product, satellite, orientation_provider, initial_state, options, mode))]
fn fit_sp3_ecef_orbit_parameters_with_validity(
    product: &PySp3,
    satellite: &str,
    orientation_provider: &PyIers2010EarthOrientationProvider,
    initial_state: Option<&PyCartesianState>,
    options: &PyOrbitParameterFitOptions,
    mode: PyValidityMode,
) -> PyResult<(PyOrbitParameterFit, Option<&'static str>)> {
    let satellite = parse_sat(satellite)?;
    let initial_state = initial_state.map(|state| *state.inner());
    core::fit_sp3_ecef_orbit_parameters_with_validity(
        &product.inner,
        satellite,
        orientation_provider.0.as_ref(),
        initial_state,
        &options.inner,
        mode.into(),
    )
    .map(|result| {
        (
            PyOrbitParameterFit(result.value),
            result.degraded.map(degrade_reason_label),
        )
    })
    .map_err(orbit_parameter_fit_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "OrbitParameterFitError",
        m.py().get_type::<OrbitParameterFitError>(),
    )?;
    m.add("STATE_PARAMETER_NAMES", core::STATE_PARAMETER_NAMES)?;
    m.add_class::<PyInertialPositionSample>()?;
    m.add_class::<PyEstimatedForceParameter>()?;
    m.add_class::<PyOrbitRtnResidual>()?;
    m.add_class::<PyOrbitParameterFit>()?;
    m.add_class::<PyOrbitParameterCovariance>()?;
    m.add_class::<PyOrbitParameterFitOptions>()?;
    m.add_function(wrap_pyfunction!(fit_inertial_orbit_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(
        fit_inertial_orbit_parameters_with_validity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(fit_sp3_ecef_orbit_parameters, m)?)?;
    m.add_function(wrap_pyfunction!(
        fit_sp3_ecef_orbit_parameters_with_validity,
        m
    )?)?;
    Ok(())
}
