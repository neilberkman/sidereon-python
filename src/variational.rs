//! Numerical propagation with state-transition and force-parameter sensitivities.

use std::sync::Arc;

use nalgebra::{DMatrix, Matrix6};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use sidereon_core::astro::forces::ForceModel;
use sidereon_core::astro::propagator::dense_output::{DenseOutput, DenseOutputError};
use sidereon_core::astro::propagator::{
    self, CovariancePropagationOptions, IntegratorOptions, StatePropagator, VariationalEpoch,
    VariationalEquations, VariationalState,
};
use sidereon_core::astro::state::CartesianState;

use crate::covariance::{
    PyCovarianceEphemeris, PyCovarianceFrame, PyLabeledCovariance6, PyProcessNoise,
};
use crate::earth_orientation::PyPropagationContext;
use crate::force_models::{force_object, propagation_error, with_context};
use crate::forces::PyDragParameters;
use crate::marshal::{covariance6_from_array, covariance6_to_array, rows_to_array};
use crate::propagation::{PyForceModelKind, PyIntegrator};
use crate::relative::PyCartesianState;
use crate::space_weather::PySpaceWeatherTable;
use crate::{binding_error, SolveError};

pub(crate) fn matrix_to_array<'py>(
    py: Python<'py>,
    matrix: &DMatrix<f64>,
) -> Bound<'py, PyArray2<f64>> {
    let values = numpy::ndarray::Array2::from_shape_fn((matrix.nrows(), matrix.ncols()), |index| {
        matrix[index]
    });
    PyArray2::from_owned_array(py, values)
}
/// Step and error controls for numerical integration.
#[pyclass(module = "sidereon._sidereon", name = "IntegratorOptions", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyIntegratorOptions(pub(crate) IntegratorOptions);

#[pymethods]
impl PyIntegratorOptions {
    /// Build numerical integrator controls.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (abs_tol=None, rel_tol=None, min_step=None, max_step=None, initial_step=None, max_steps=None, dense_output=None))]
    fn new(
        abs_tol: Option<f64>,
        rel_tol: Option<f64>,
        min_step: Option<f64>,
        max_step: Option<f64>,
        initial_step: Option<f64>,
        max_steps: Option<u32>,
        dense_output: Option<bool>,
    ) -> Self {
        let mut options = IntegratorOptions::default();
        if let Some(abs_tol) = abs_tol {
            options.abs_tol = abs_tol;
        }
        if let Some(rel_tol) = rel_tol {
            options.rel_tol = rel_tol;
        }
        if let Some(min_step) = min_step {
            options.min_step = min_step;
        }
        if let Some(max_step) = max_step {
            options.max_step = max_step;
        }
        if let Some(initial_step) = initial_step {
            options.initial_step = initial_step;
        }
        if let Some(max_steps) = max_steps {
            options.max_steps = max_steps;
        }
        if let Some(dense_output) = dense_output {
            options.dense_output = dense_output;
        }
        Self(options)
    }

    /// Absolute error tolerance.
    #[getter]
    fn abs_tol(&self) -> f64 {
        self.0.abs_tol
    }

    /// Relative error tolerance.
    #[getter]
    fn rel_tol(&self) -> f64 {
        self.0.rel_tol
    }

    /// Minimum step size in seconds.
    #[getter]
    fn min_step(&self) -> f64 {
        self.0.min_step
    }

    /// Maximum step size in seconds.
    #[getter]
    fn max_step(&self) -> f64 {
        self.0.max_step
    }

    /// Initial step size in seconds.
    #[getter]
    fn initial_step(&self) -> f64 {
        self.0.initial_step
    }

    /// Maximum number of step outcomes.
    #[getter]
    fn max_steps(&self) -> u32 {
        self.0.max_steps
    }

    /// Whether dense output is requested.
    #[getter]
    fn dense_output(&self) -> bool {
        self.0.dense_output
    }
}

/// Evaluation epoch in TDB with elapsed integration time in TT.
#[pyclass(module = "sidereon._sidereon", name = "VariationalEpoch", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyVariationalEpoch(pub(crate) VariationalEpoch);

#[pymethods]
impl PyVariationalEpoch {
    /// Specify TDB epoch and elapsed TT time explicitly.
    #[new]
    fn new(tdb_j2000_s: f64, elapsed_tt_s: f64) -> Self {
        Self(VariationalEpoch {
            tdb_j2000_s,
            elapsed_tt_s,
        })
    }

    /// TDB seconds from J2000.
    #[getter]
    fn tdb_j2000_s(&self) -> f64 {
        self.0.tdb_j2000_s
    }

    /// Elapsed TT seconds from the initial epoch.
    #[getter]
    fn elapsed_tt_s(&self) -> f64 {
        self.0.elapsed_tt_s
    }

    /// Build the TT and TDB values from initial and target TDB epochs.
    #[staticmethod]
    fn from_tdb_seconds(initial_tdb_j2000_s: f64, tdb_j2000_s: f64) -> Self {
        Self(VariationalEpoch::from_tdb_seconds(
            initial_tdb_j2000_s,
            tdb_j2000_s,
        ))
    }
}

/// Concrete force and parameter values for the variational equations.
#[pyclass(module = "sidereon._sidereon", name = "VariationalEquations")]
pub struct PyVariationalEquations {
    force: Py<PyAny>,
    model: Arc<dyn ForceModel>,
    parameters: Vec<f64>,
    sensitivity_parameters: Vec<usize>,
}

#[pymethods]
impl PyVariationalEquations {
    /// Use replacement force parameters and select sensitivity columns by index.
    #[new]
    fn new(
        force: Bound<'_, PyAny>,
        parameters: Vec<f64>,
        sensitivity_parameters: Vec<usize>,
    ) -> PyResult<Self> {
        let model = force_object(&force)?;
        Ok(Self {
            force: force.unbind(),
            model,
            parameters,
            sensitivity_parameters,
        })
    }

    /// Concrete force model.
    #[getter]
    fn force(&self, py: Python<'_>) -> Py<PyAny> {
        self.force.clone_ref(py)
    }

    /// Replacement parameter values in model order.
    #[getter]
    fn parameters(&self) -> Vec<f64> {
        self.parameters.clone()
    }

    /// Parameter indices in sensitivity column order.
    #[getter]
    fn sensitivity_parameters(&self) -> Vec<usize> {
        self.sensitivity_parameters.clone()
    }
}
/// Propagated state and its derivatives with respect to state and force parameters.
#[pyclass(module = "sidereon._sidereon", name = "VariationalState", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyVariationalState(VariationalState);

#[pymethods]
impl PyVariationalState {
    /// Cartesian state at the requested epoch.
    #[getter]
    fn state(&self) -> PyCartesianState {
        PyCartesianState::from_inner(self.0.state)
    }

    /// Six-by-six derivative with respect to the initial state.
    #[getter]
    fn state_transition<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows: [[f64; 6]; 6] = rows_of_matrix6(&self.0.state_transition);
        rows_to_array(py, &rows)
    }

    /// Derivatives with respect to selected force parameters, one column per index.
    #[getter]
    fn sensitivity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let matrix = &self.0.sensitivity;
        let values =
            numpy::ndarray::Array2::from_shape_fn((6, matrix.ncols()), |index| matrix[index]);
        PyArray2::from_owned_array(py, values)
    }
}

fn rows_of_matrix6(matrix: &Matrix6<f64>) -> [[f64; 6]; 6] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[(row, column)]))
}
/// Propagate states, state-transition matrices, and selected force-parameter sensitivities.
#[pyfunction]
#[pyo3(signature = (initial, epochs, equations, integrator, options=None, context=None))]
fn propagate_variational(
    initial: &PyCartesianState,
    epochs: Vec<PyVariationalEpoch>,
    equations: &PyVariationalEquations,
    integrator: PyIntegrator,
    options: Option<&PyIntegratorOptions>,
    context: Option<&PyPropagationContext>,
) -> PyResult<Vec<PyVariationalState>> {
    let epochs: Vec<_> = epochs.into_iter().map(|epoch| epoch.0).collect();
    let equations = VariationalEquations {
        force: equations.model.as_ref(),
        parameters: &equations.parameters,
        sensitivity_parameters: &equations.sensitivity_parameters,
    };
    let options = options.map(|options| options.0).unwrap_or_default();
    with_context(context, |ctx| {
        propagator::propagate_variational(
            *initial.inner(),
            &epochs,
            &equations,
            integrator.into(),
            &options,
            ctx,
        )
    })
    .map(|states| states.into_iter().map(PyVariationalState).collect())
    .map_err(propagation_error)
}

/// Continuous Cartesian output from the accepted steps of a DP54 arc.
#[pyclass(module = "sidereon._sidereon", name = "DenseOutput", frozen)]
#[derive(Clone)]
pub struct PyDenseOutput(pub(crate) Arc<DenseOutput>);

#[pymethods]
impl PyDenseOutput {
    /// Earliest and latest covered TDB seconds past J2000, or None for an empty arc.
    #[getter]
    fn coverage(&self) -> Option<(f64, f64)> {
        self.0.coverage_tdb_seconds()
    }

    /// Evaluate the Cartesian interpolant at TDB seconds past J2000.
    fn eval(&self, t: f64) -> PyResult<PyCartesianState> {
        self.0
            .eval(t)
            .map(PyCartesianState::from_inner)
            .map_err(dense_output_error)
    }
}

fn dense_output_error(error: DenseOutputError) -> PyErr {
    let exception = SolveError::new_err(error.to_string());
    match error {
        DenseOutputError::Empty => binding_error(exception, "empty", |_| Ok(())),
        DenseOutputError::OutputOutOfRange { t, t_min, t_max } => {
            binding_error(exception, "output_out_of_range", |detail| {
                detail.set_item("t", t)?;
                detail.set_item("t_min", t_min)?;
                detail.set_item("t_max", t_max)
            })
        }
        DenseOutputError::SegmentOutOfRange {
            t,
            t_start,
            t_end,
            theta,
        } => binding_error(exception, "segment_out_of_range", |detail| {
            detail.set_item("t", t)?;
            detail.set_item("t_start", t_start)?;
            detail.set_item("t_end", t_end)?;
            detail.set_item("theta", theta)
        }),
        _ => exception,
    }
}

/// Final Cartesian state and optional continuous output from a numerical arc.
#[pyclass(module = "sidereon._sidereon", name = "PropagationResult", frozen)]
pub struct PyPropagationResult {
    final_state: CartesianState,
    dense: Option<PyDenseOutput>,
}

#[pymethods]
impl PyPropagationResult {
    /// Cartesian state at the requested end epoch.
    #[getter]
    fn final_state(&self) -> PyCartesianState {
        PyCartesianState::from_inner(self.final_state)
    }

    /// Continuous DP54 output if enabled in the integrator options.
    #[getter]
    fn dense(&self) -> Option<PyDenseOutput> {
        self.dense.clone()
    }
}

/// Numerical orbit propagator with a built-in force configuration.
#[pyclass(module = "sidereon._sidereon", name = "StatePropagator")]
pub struct PyStatePropagator(StatePropagator);

#[pymethods]
impl PyStatePropagator {
    /// Build the propagator from a Cartesian state and force configuration.
    #[new]
    #[pyo3(signature = (initial, force_model, integrator, options=None, drag=None, space_weather_table=None))]
    fn new(
        initial: &PyCartesianState,
        force_model: &PyForceModelKind,
        integrator: PyIntegrator,
        options: Option<&PyIntegratorOptions>,
        drag: Option<&PyDragParameters>,
        space_weather_table: Option<&PySpaceWeatherTable>,
    ) -> Self {
        let propagator = StatePropagator {
            initial: *initial.inner(),
            force_model: force_model.inner(),
            integrator: integrator.into(),
            options: options.map(|options| options.0).unwrap_or_default(),
            drag: drag.map(PyDragParameters::inner),
            space_weather: space_weather_table.map(PySpaceWeatherTable::source),
        };
        Self(propagator)
    }

    /// Propagate to an absolute TDB epoch and return the final state and optional dense output.
    ///
    /// Additional forces are summed in list order after configured gravity and drag.
    #[pyo3(signature = (epoch_tdb_seconds, context=None, *, additional_forces=Vec::new()))]
    fn propagate_to(
        &self,
        epoch_tdb_seconds: f64,
        context: Option<&PyPropagationContext>,
        additional_forces: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<PyPropagationResult> {
        let additional_forces = additional_forces
            .iter()
            .map(force_object)
            .collect::<PyResult<Vec<_>>>()?;
        with_context(context, |ctx| {
            self.0
                .propagate_to_with_additional_forces(epoch_tdb_seconds, &additional_forces, ctx)
        })
        .map(|result| PyPropagationResult {
            final_state: result.final_state,
            dense: result.dense.map(|output| PyDenseOutput(Arc::new(output))),
        })
        .map_err(propagation_error)
    }

    /// Evaluate Cartesian states on the specified TDB grid using the context.
    ///
    /// Additional forces are summed in list order after configured gravity and drag.
    #[pyo3(signature = (epochs_tdb_seconds, context, *, additional_forces=Vec::new()))]
    fn ephemeris_with_context(
        &self,
        epochs_tdb_seconds: Vec<f64>,
        context: &PyPropagationContext,
        additional_forces: Vec<Bound<'_, PyAny>>,
    ) -> PyResult<Vec<PyCartesianState>> {
        let additional_forces = additional_forces
            .iter()
            .map(force_object)
            .collect::<PyResult<Vec<_>>>()?;
        self.0
            .ephemeris_with_additional_forces(
                &epochs_tdb_seconds,
                &additional_forces,
                &context.inner,
            )
            .map(|states| {
                states
                    .into_iter()
                    .map(PyCartesianState::from_inner)
                    .collect()
            })
            .map_err(propagation_error)
    }

    /// Evaluate the state-transition matrix at an absolute TDB epoch.
    fn state_transition_matrix_to_with_context<'py>(
        &self,
        py: Python<'py>,
        epoch_tdb_seconds: f64,
        context: &PyPropagationContext,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.0
            .state_transition_matrix_to_with_context(epoch_tdb_seconds, &context.inner)
            .map(|matrix| rows_to_array(py, &matrix))
            .map_err(propagation_error)
    }

    /// Propagate a labeled covariance over TDB epochs with an explicit context.
    #[pyo3(signature = (initial, epochs_tdb_seconds, context, *, process_noise=None, output_frame=PyCovarianceFrame::INERTIAL))]
    fn propagate_covariance_with_context(
        &self,
        initial: &PyLabeledCovariance6,
        epochs_tdb_seconds: Vec<f64>,
        context: &PyPropagationContext,
        process_noise: Option<&PyProcessNoise>,
        output_frame: PyCovarianceFrame,
    ) -> PyResult<PyCovarianceEphemeris> {
        let mut options = CovariancePropagationOptions::default();
        options.output_frame = output_frame.into();
        if let Some(process_noise) = process_noise {
            options.process_noise = process_noise.inner();
        }
        self.0
            .propagate_covariance_with_context(
                initial.inner(),
                &epochs_tdb_seconds,
                &options,
                &context.inner,
            )
            .map(|inner| PyCovarianceEphemeris { inner })
            .map_err(propagation_error)
    }

    /// Propagate an inertial state and covariance over a relative span with the context.
    fn propagate_state_with_covariance_with_context<'py>(
        &self,
        py: Python<'py>,
        covariance0: PyReadonlyArray2<'_, f64>,
        span_seconds: f64,
        context: &PyPropagationContext,
    ) -> PyResult<(PyCartesianState, Bound<'py, PyArray2<f64>>)> {
        let covariance0 = covariance6_from_array(&covariance0, "covariance0")?;
        let (state, covariance) = self
            .0
            .propagate_state_with_covariance_with_context(covariance0, span_seconds, &context.inner)
            .map_err(propagation_error)?;
        Ok((
            PyCartesianState::from_inner(state),
            covariance6_to_array(py, &covariance),
        ))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIntegratorOptions>()?;
    m.add_class::<PyDenseOutput>()?;
    m.add_class::<PyPropagationResult>()?;
    m.add_class::<PyVariationalEpoch>()?;
    m.add_class::<PyVariationalEquations>()?;
    m.add_class::<PyVariationalState>()?;
    m.add_function(wrap_pyfunction!(propagate_variational, m)?)?;
    m.add_class::<PyStatePropagator>()?;
    Ok(())
}
