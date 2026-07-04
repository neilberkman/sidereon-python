//! Jacobian-derived covariance and 2-D error-ellipse geometry binding.
//!
//! Thin INTERFACE over `sidereon_core`'s least-squares covariance primitives
//! ([`normal_covariance`], [`hessian_trace`], [`covariance_from_jacobian`]) and the
//! domain-neutral [`error_ellipse_2x2`]. The binding marshals numpy matrices into
//! the core's `nalgebra` types and packages the results; every number is produced
//! by the core, no linear algebra lives here.

use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use nalgebra::DMatrix;

use sidereon::propagator::api::IntegratorOptions;
use sidereon::propagator::{IntegratorKind, PropagationConfig};
use sidereon_core::astro::covariance::{
    covariance6_km_to_m as core_covariance6_km_to_m,
    covariance6_m_to_km as core_covariance6_m_to_km,
    eci_to_rtn_covariance6 as core_eci_to_rtn_covariance6,
    interpolate_covariance_psd as core_interpolate_covariance_psd,
    rtn_to_eci_covariance6 as core_rtn_to_eci_covariance6, RtnFrameError,
};
use sidereon_core::astro::math::least_squares::{
    covariance_from_jacobian as core_covariance_from_jacobian, hessian_trace as core_hessian_trace,
    normal_covariance as core_normal_covariance, SolveError as CoreSolveError,
};
use sidereon_core::astro::propagator::{
    CovarianceEphemeris as CoreCovarianceEphemeris, CovarianceFrame as CoreCovarianceFrame,
    CovariancePropagationOptions, LabeledCovariance6 as CoreLabeledCovariance6,
    ProcessNoise as CoreProcessNoise,
};
use sidereon_core::geometry::{error_ellipse_2x2 as core_error_ellipse_2x2, DopError};

use crate::forces::PyDragParameters;
use crate::marshal::{covariance6_from_array, covariance6_to_array, option_py_or_default};
use crate::propagation::{PyForceModel, PyIntegrator};
use crate::relative::PyCartesianState;
use crate::space_weather::PySpaceWeatherTable;
use crate::SolveError;

/// Map a core least-squares [`CoreSolveError`]: a malformed input/shape is a
/// `ValueError`; a rank-deficient (singular) Jacobian is a [`SolveError`].
fn to_covariance_err(err: CoreSolveError) -> PyErr {
    match err {
        CoreSolveError::InvalidInput { .. } => PyValueError::new_err(err.to_string()),
        CoreSolveError::SingularJacobian => SolveError::new_err(err.to_string()),
    }
}

/// Read a 2-D numpy array into a row-major `nalgebra` [`DMatrix`].
fn dmatrix_from_array(name: &str, arr: &PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let view = arr.as_array();
    let rows = view.nrows();
    let cols = view.ncols();
    let mut flat = Vec::with_capacity(rows * cols);
    for value in view.iter() {
        if !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name} must contain only finite values"
            )));
        }
        flat.push(*value);
    }
    // `view.iter()` yields row-major order; `DMatrix::from_row_slice` reads the
    // same order.
    Ok(DMatrix::from_row_slice(rows, cols, &flat))
}

/// Pack an `nalgebra` [`DMatrix`] into a `(rows, cols)` numpy float64 array.
fn dmatrix_to_array<'py>(py: Python<'py>, matrix: &DMatrix<f64>) -> Bound<'py, PyArray2<f64>> {
    let rows = matrix.nrows();
    let cols = matrix.ncols();
    let mut out = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            out[[r, c]] = matrix[(r, c)];
        }
    }
    out.to_pyarray(py)
}

fn to_rtn_err(err: RtnFrameError) -> PyErr {
    let message = match err {
        RtnFrameError::InvalidInput { field, reason } => {
            format!("invalid input for {field}: {reason}")
        }
        other => other.message().to_string(),
    };
    PyValueError::new_err(message)
}

fn extract_covariance_frame(obj: &Bound<'_, PyAny>) -> PyResult<PyCovarianceFrame> {
    if let Ok(frame) = obj.extract::<PyCovarianceFrame>() {
        return Ok(frame);
    }
    PyCovarianceFrame::from_label(&obj.extract::<String>()?)
}

/// Frame used by a 6x6 Cartesian covariance.
#[pyclass(module = "sidereon._sidereon", name = "CovarianceFrame", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyCovarianceFrame {
    /// Propagator inertial axes.
    INERTIAL,
    /// Radial, transverse, normal axes at the associated state.
    RTN,
}

impl PyCovarianceFrame {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "inertial" => Ok(Self::INERTIAL),
            "rtn" => Ok(Self::RTN),
            other => Err(PyValueError::new_err(format!(
                "unknown covariance frame {other:?}; expected \"inertial\" or \"rtn\""
            ))),
        }
    }
}

impl From<PyCovarianceFrame> for CoreCovarianceFrame {
    fn from(frame: PyCovarianceFrame) -> Self {
        match frame {
            PyCovarianceFrame::INERTIAL => CoreCovarianceFrame::Inertial,
            PyCovarianceFrame::RTN => CoreCovarianceFrame::Rtn,
        }
    }
}

impl From<CoreCovarianceFrame> for PyCovarianceFrame {
    fn from(frame: CoreCovarianceFrame) -> Self {
        match frame {
            CoreCovarianceFrame::Inertial => Self::INERTIAL,
            CoreCovarianceFrame::Rtn => Self::RTN,
        }
    }
}

#[pymethods]
impl PyCovarianceFrame {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::INERTIAL => "inertial",
            Self::RTN => "rtn",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::INERTIAL => "CovarianceFrame.INERTIAL",
            Self::RTN => "CovarianceFrame.RTN",
        }
    }
}

/// Validated 6x6 covariance with its reference frame.
#[pyclass(module = "sidereon._sidereon", name = "LabeledCovariance6")]
#[derive(Clone, Copy)]
pub struct PyLabeledCovariance6 {
    inner: CoreLabeledCovariance6,
}

impl PyLabeledCovariance6 {
    pub(crate) fn inner(&self) -> CoreLabeledCovariance6 {
        self.inner
    }
}

#[pymethods]
impl PyLabeledCovariance6 {
    #[new]
    #[pyo3(signature = (covariance, frame=PyCovarianceFrame::INERTIAL))]
    fn new(
        covariance: PyReadonlyArray2<'_, f64>,
        #[pyo3(from_py_with = extract_covariance_frame)] frame: PyCovarianceFrame,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: CoreLabeledCovariance6 {
                covariance: covariance6_from_array(&covariance, "covariance")?,
                frame: frame.into(),
            },
        })
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        covariance6_to_array(py, &self.inner.covariance)
    }

    #[getter]
    fn frame(&self) -> PyCovarianceFrame {
        self.inner.frame.into()
    }

    fn __repr__(&self) -> String {
        format!("LabeledCovariance6(frame={})", self.frame().label())
    }
}

/// Process-noise model for covariance propagation.
#[pyclass(module = "sidereon._sidereon", name = "ProcessNoise")]
#[derive(Clone, Copy)]
pub struct PyProcessNoise {
    inner: CoreProcessNoise,
}

impl PyProcessNoise {
    pub(crate) fn inner(&self) -> CoreProcessNoise {
        self.inner
    }
}

#[pymethods]
impl PyProcessNoise {
    #[new]
    #[pyo3(signature = (q_radial_km2_s3=None, q_transverse_km2_s3=None, q_normal_km2_s3=None))]
    fn new(
        q_radial_km2_s3: Option<f64>,
        q_transverse_km2_s3: Option<f64>,
        q_normal_km2_s3: Option<f64>,
    ) -> PyResult<Self> {
        match (q_radial_km2_s3, q_transverse_km2_s3, q_normal_km2_s3) {
            (None, None, None) => Ok(Self {
                inner: CoreProcessNoise::None,
            }),
            (Some(q_radial_km2_s3), Some(q_transverse_km2_s3), Some(q_normal_km2_s3)) => Ok(Self {
                inner: CoreProcessNoise::RtnAccelerationPsd {
                    q_radial_km2_s3,
                    q_transverse_km2_s3,
                    q_normal_km2_s3,
                },
            }),
            _ => Err(PyValueError::new_err(
                "all RTN acceleration PSD values must be supplied together",
            )),
        }
    }

    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: CoreProcessNoise::None,
        }
    }

    #[staticmethod]
    fn rtn_acceleration_psd(
        q_radial_km2_s3: f64,
        q_transverse_km2_s3: f64,
        q_normal_km2_s3: f64,
    ) -> Self {
        Self {
            inner: CoreProcessNoise::RtnAccelerationPsd {
                q_radial_km2_s3,
                q_transverse_km2_s3,
                q_normal_km2_s3,
            },
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CoreProcessNoise::None => "none",
            CoreProcessNoise::RtnAccelerationPsd { .. } => "rtn_acceleration_psd",
        }
    }

    #[getter]
    fn q_radial_km2_s3(&self) -> Option<f64> {
        match self.inner {
            CoreProcessNoise::RtnAccelerationPsd {
                q_radial_km2_s3, ..
            } => Some(q_radial_km2_s3),
            CoreProcessNoise::None => None,
        }
    }

    #[getter]
    fn q_transverse_km2_s3(&self) -> Option<f64> {
        match self.inner {
            CoreProcessNoise::RtnAccelerationPsd {
                q_transverse_km2_s3,
                ..
            } => Some(q_transverse_km2_s3),
            CoreProcessNoise::None => None,
        }
    }

    #[getter]
    fn q_normal_km2_s3(&self) -> Option<f64> {
        match self.inner {
            CoreProcessNoise::RtnAccelerationPsd {
                q_normal_km2_s3, ..
            } => Some(q_normal_km2_s3),
            CoreProcessNoise::None => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("ProcessNoise(kind={:?})", self.kind())
    }
}

/// One propagated state and covariance node.
#[pyclass(module = "sidereon._sidereon", name = "CovarianceNode")]
#[derive(Clone, Copy)]
pub struct PyCovarianceNode {
    state: PyCartesianState,
    covariance: sidereon_core::astro::covariance::Covariance6,
    frame: CoreCovarianceFrame,
}

#[pymethods]
impl PyCovarianceNode {
    #[getter]
    fn state(&self) -> PyCartesianState {
        self.state
    }

    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        covariance6_to_array(py, &self.covariance)
    }

    #[getter]
    fn frame(&self) -> PyCovarianceFrame {
        self.frame.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "CovarianceNode(epoch_tdb_seconds={}, frame={})",
            self.state.inner().epoch_tdb_seconds,
            self.frame().label()
        )
    }
}

/// Propagated covariance ephemeris.
#[pyclass(module = "sidereon._sidereon", name = "CovarianceEphemeris")]
#[derive(Clone)]
pub struct PyCovarianceEphemeris {
    inner: CoreCovarianceEphemeris,
}

#[pymethods]
impl PyCovarianceEphemeris {
    #[getter]
    fn nodes(&self) -> Vec<PyCovarianceNode> {
        self.inner
            .nodes()
            .iter()
            .map(|node| PyCovarianceNode {
                state: PyCartesianState::from_inner(node.state),
                covariance: node.covariance,
                frame: node.frame,
            })
            .collect()
    }

    #[getter]
    fn epoch_count(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    fn covariance_at<'py>(
        &self,
        py: Python<'py>,
        epoch_tdb_seconds: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let covariance = self
            .inner
            .covariance_at(epoch_tdb_seconds)
            .map_err(crate::to_solve_err)?;
        Ok(covariance6_to_array(py, &covariance))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("CovarianceEphemeris(epoch_count={})", self.inner.len())
    }
}

/// Parameter covariance from a design (Jacobian) matrix via the Gauss-Newton
/// normal equations `variance_scale * (J^T J)^-1`, formed from the thin SVD of
/// `J` (so the conditioning stays at `cond(J)`, not `cond(J)^2`).
///
/// `jacobian` is an `(m, n)` array with `m >= n`. Pass the post-fit reduced
/// chi-square as `variance_scale` for the fitted covariance, or `1.0` for the
/// bare cofactor. Raises `ValueError` on a rank-deficient Jacobian or a bad
/// shape. This is the same quantity `scipy.optimize.curve_fit` reports as `pcov`.
#[pyfunction]
#[pyo3(signature = (jacobian, variance_scale=1.0))]
fn normal_covariance<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray2<'_, f64>,
    variance_scale: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let jacobian = dmatrix_from_array("jacobian", &jacobian)?;
    let cov = core_normal_covariance(&jacobian, variance_scale).map_err(to_covariance_err)?;
    Ok(dmatrix_to_array(py, &cov))
}

/// Trace of the Gauss-Newton Hessian approximation `J^T J`, i.e. the sum of the
/// squared column norms of `jacobian`. No inverse is formed.
#[pyfunction]
fn hessian_trace(jacobian: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
    let jacobian = dmatrix_from_array("jacobian", &jacobian)?;
    Ok(core_hessian_trace(&jacobian))
}

/// Fitted parameter covariance directly from a converged solve's design
/// (Jacobian) matrix and post-fit `cost`.
///
/// Scales `(J^T J)^-1` by the post-fit reduced chi-square `2 * cost / (m - n)`,
/// the same scale `scipy.optimize.curve_fit` applies, with the redundancy taken
/// from the Jacobian's own `(m, n)` shape. Requires positive redundancy `m > n`.
/// Delegates straight to the core `covariance_from_jacobian`.
#[pyfunction]
#[pyo3(signature = (jacobian, cost))]
fn covariance_from_jacobian<'py>(
    py: Python<'py>,
    jacobian: PyReadonlyArray2<'_, f64>,
    cost: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    if !cost.is_finite() {
        return Err(PyValueError::new_err("cost must be finite"));
    }
    let jacobian = dmatrix_from_array("jacobian", &jacobian)?;
    let cov = core_covariance_from_jacobian(&jacobian, cost).map_err(to_covariance_err)?;
    Ok(dmatrix_to_array(py, &cov))
}

/// A 2-D confidence error ellipse from a 2x2 covariance block.
#[pyclass(module = "sidereon._sidereon", name = "ErrorEllipse")]
#[derive(Clone)]
pub(crate) struct PyErrorEllipse {
    confidence: f64,
    chi_square_scale: f64,
    semi_major: f64,
    semi_minor: f64,
    orientation_rad: f64,
}

impl PyErrorEllipse {
    pub(crate) fn from_one_sigma_m(ellipse: sidereon_core::error_metrics::ErrorEllipse) -> Self {
        Self {
            confidence: f64::NAN,
            chi_square_scale: 1.0,
            semi_major: ellipse.semi_major_m,
            semi_minor: ellipse.semi_minor_m,
            orientation_rad: ellipse.orientation_rad,
        }
    }
}

#[pymethods]
impl PyErrorEllipse {
    /// Requested confidence probability in `(0, 1)`.
    #[getter]
    fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Two-degree-of-freedom chi-square scale `-2 ln(1 - confidence)`.
    #[getter]
    fn chi_square_scale(&self) -> f64 {
        self.chi_square_scale
    }

    /// Semi-major axis length (same unit as the square root of the covariance).
    #[getter]
    fn semi_major(&self) -> f64 {
        self.semi_major
    }

    /// Semi-major axis length in metres for position-error metrics.
    #[getter]
    fn semi_major_m(&self) -> f64 {
        self.semi_major
    }

    /// Semi-minor axis length.
    #[getter]
    fn semi_minor(&self) -> f64 {
        self.semi_minor
    }

    /// Semi-minor axis length in metres for position-error metrics.
    #[getter]
    fn semi_minor_m(&self) -> f64 {
        self.semi_minor
    }

    /// Semi-major-axis orientation, radians, from the first axis toward the
    /// second.
    #[getter]
    fn orientation_rad(&self) -> f64 {
        self.orientation_rad
    }

    fn __repr__(&self) -> String {
        format!(
            "ErrorEllipse(confidence={}, semi_major={}, semi_minor={}, orientation_rad={})",
            self.confidence, self.semi_major, self.semi_minor, self.orientation_rad
        )
    }
}

/// Confidence ellipse from an arbitrary 2x2 covariance block.
///
/// `covariance` is a `(2, 2)` array; `confidence` is in `(0, 1)`. The semi-axes
/// are scaled by the two-degree-of-freedom chi-square quantile
/// `-2 ln(1 - confidence)` applied to the eigenvalues of the symmetrized block.
/// Returns an [`ErrorEllipse`] with semi-major / semi-minor / orientation.
#[pyfunction]
#[pyo3(signature = (covariance, confidence))]
fn error_ellipse_2x2(
    covariance: PyReadonlyArray2<'_, f64>,
    confidence: f64,
) -> PyResult<PyErrorEllipse> {
    let view = covariance.as_array();
    if view.shape() != [2, 2] {
        return Err(PyValueError::new_err("covariance must have shape (2, 2)"));
    }
    let block = [[view[[0, 0]], view[[0, 1]]], [view[[1, 0]], view[[1, 1]]]];
    let ellipse = core_error_ellipse_2x2(block, confidence).map_err(|err| match err {
        DopError::InvalidInput { field, reason } => {
            PyValueError::new_err(format!("invalid {field}: {reason}"))
        }
        other => SolveError::new_err(other.to_string()),
    })?;
    Ok(PyErrorEllipse {
        confidence: ellipse.confidence,
        chi_square_scale: ellipse.chi_square_scale,
        semi_major: ellipse.semi_major,
        semi_minor: ellipse.semi_minor,
        orientation_rad: ellipse.orientation_rad,
    })
}

/// Convert a km-based 6x6 state covariance to m-based units.
#[pyfunction]
fn covariance6_km_to_m<'py>(
    py: Python<'py>,
    covariance: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let covariance = covariance6_from_array(&covariance, "covariance")?;
    let converted = core_covariance6_km_to_m(&covariance)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok(covariance6_to_array(py, &converted))
}

/// Convert an m-based 6x6 state covariance to km-based units.
#[pyfunction]
fn covariance6_m_to_km<'py>(
    py: Python<'py>,
    covariance: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let covariance = covariance6_from_array(&covariance, "covariance")?;
    let converted = core_covariance6_m_to_km(&covariance)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok(covariance6_to_array(py, &converted))
}

/// PSD-safe interpolation between two 6x6 state covariances.
#[pyfunction]
fn interpolate_covariance6<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'_, f64>,
    b: PyReadonlyArray2<'_, f64>,
    u: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a = covariance6_from_array(&a, "a")?;
    let b = covariance6_from_array(&b, "b")?;
    let interpolated = core_interpolate_covariance_psd(&a, &b, u)
        .map_err(|err| PyValueError::new_err(format!("{err:?}")))?;
    Ok(covariance6_to_array(py, &interpolated))
}

/// Transform an inertial 6x6 covariance to RTN at a Cartesian state.
#[pyfunction]
fn eci_to_rtn_covariance6<'py>(
    py: Python<'py>,
    covariance: PyReadonlyArray2<'_, f64>,
    state: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let covariance = covariance6_from_array(&covariance, "covariance")?;
    let transformed =
        core_eci_to_rtn_covariance6(&covariance, state.inner()).map_err(to_rtn_err)?;
    Ok(covariance6_to_array(py, &transformed))
}

/// Transform an RTN 6x6 covariance to inertial axes at a Cartesian state.
#[pyfunction]
fn rtn_to_eci_covariance6<'py>(
    py: Python<'py>,
    covariance: PyReadonlyArray2<'_, f64>,
    state: &PyCartesianState,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let covariance = covariance6_from_array(&covariance, "covariance")?;
    let transformed =
        core_rtn_to_eci_covariance6(&covariance, state.inner()).map_err(to_rtn_err)?;
    Ok(covariance6_to_array(py, &transformed))
}

/// Propagate a Cartesian state covariance to requested TDB epochs.
#[pyfunction]
#[pyo3(signature = (
    initial_state,
    initial_covariance,
    epochs_tdb_seconds,
    *,
    force_model = PyForceModel::TWO_BODY,
    integrator = PyIntegrator::DP54,
    abs_tol = 1.0e-9,
    rel_tol = 1.0e-12,
    initial_step_s = 60.0,
    min_step_s = 1.0e-6,
    max_step_s = 3600.0,
    max_steps = 1_000_000,
    mu_km3_s2 = None,
    drag = None,
    space_weather_table = None,
    process_noise = None,
    output_frame = PyCovarianceFrame::INERTIAL,
))]
#[allow(clippy::too_many_arguments)]
fn propagate_covariance(
    py: Python<'_>,
    initial_state: &PyCartesianState,
    initial_covariance: &PyLabeledCovariance6,
    epochs_tdb_seconds: PyReadonlyArray1<'_, f64>,
    #[pyo3(from_py_with = crate::propagation::extract_force_model)] force_model: PyForceModel,
    #[pyo3(from_py_with = crate::propagation::extract_integrator)] integrator: PyIntegrator,
    abs_tol: f64,
    rel_tol: f64,
    initial_step_s: f64,
    min_step_s: f64,
    max_step_s: f64,
    max_steps: u32,
    mu_km3_s2: Option<f64>,
    drag: Option<Py<PyDragParameters>>,
    space_weather_table: Option<Py<PySpaceWeatherTable>>,
    process_noise: Option<Py<PyProcessNoise>>,
    #[pyo3(from_py_with = extract_covariance_frame)] output_frame: PyCovarianceFrame,
) -> PyResult<PyCovarianceEphemeris> {
    if initial_step_s <= 0.0 {
        return Err(PyValueError::new_err("initial_step_s must be positive"));
    }
    let epochs = epochs_tdb_seconds
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?
        .to_vec();
    let process_noise = option_py_or_default(
        py,
        process_noise.as_ref(),
        PyProcessNoise::inner,
        CoreProcessNoise::default,
    );
    let drag = drag.map(|value| value.borrow(py).inner());
    let state = *initial_state.inner();

    let config = PropagationConfig {
        force_model: force_model.to_core(),
        mu_km3_s2,
        integrator: IntegratorKind::from(integrator),
        options: IntegratorOptions {
            abs_tol,
            rel_tol,
            initial_step: initial_step_s,
            min_step: min_step_s,
            max_step: max_step_s,
            max_steps,
            dense_output: false,
        },
        drag,
        ..PropagationConfig::new(
            state.epoch_tdb_seconds,
            state.position_array(),
            state.velocity_array(),
        )
    };
    let options = CovariancePropagationOptions {
        process_noise,
        output_frame: output_frame.into(),
    };
    let initial_covariance = initial_covariance.inner();
    let inner = if let Some(table) = space_weather_table {
        let source = table.borrow(py).source();
        let propagator = config.to_propagator().with_space_weather(source);
        py.allow_threads(move || {
            propagator.propagate_covariance(initial_covariance, &epochs, &options)
        })
        .map_err(crate::to_solve_err)?
    } else {
        let propagator = config.to_propagator();
        py.allow_threads(move || {
            propagator.propagate_covariance(initial_covariance, &epochs, &options)
        })
        .map_err(crate::to_solve_err)?
    };
    Ok(PyCovarianceEphemeris { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyErrorEllipse>()?;
    m.add_class::<PyCovarianceFrame>()?;
    m.add_class::<PyLabeledCovariance6>()?;
    m.add_class::<PyProcessNoise>()?;
    m.add_class::<PyCovarianceNode>()?;
    m.add_class::<PyCovarianceEphemeris>()?;
    m.add_function(wrap_pyfunction!(normal_covariance, m)?)?;
    m.add_function(wrap_pyfunction!(hessian_trace, m)?)?;
    m.add_function(wrap_pyfunction!(covariance_from_jacobian, m)?)?;
    m.add_function(wrap_pyfunction!(error_ellipse_2x2, m)?)?;
    m.add_function(wrap_pyfunction!(covariance6_km_to_m, m)?)?;
    m.add_function(wrap_pyfunction!(covariance6_m_to_km, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_covariance6, m)?)?;
    m.add_function(wrap_pyfunction!(eci_to_rtn_covariance6, m)?)?;
    m.add_function(wrap_pyfunction!(rtn_to_eci_covariance6, m)?)?;
    m.add_function(wrap_pyfunction!(propagate_covariance, m)?)?;
    Ok(())
}
