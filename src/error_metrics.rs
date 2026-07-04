//! Position error-metric bindings.

use std::collections::BTreeMap;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::error_metrics::{
    metrics_from_ecef_covariance_m2 as core_metrics_from_ecef_covariance_m2,
    metrics_from_enu_covariance_m2 as core_metrics_from_enu_covariance_m2,
    metrics_from_kinematic_solution as core_metrics_from_kinematic_solution, ErrorMetricsError,
    PercentileRadius, PositionErrorMetrics,
};
use sidereon_core::precise_positioning::{KinematicEpochSolution, KinematicEpochStatus};

use crate::covariance::PyErrorEllipse;
use crate::events::PyWgs84Geodetic;
use crate::marshal::{fixed_array, mat3_to_array, matrix3_from_array, FinitePolicy};
use crate::np_array;

fn to_metrics_err(err: ErrorMetricsError) -> PyErr {
    match err {
        ErrorMetricsError::NonFinite => {
            PyValueError::new_err("covariance contains non-finite values")
        }
        ErrorMetricsError::NotPositiveSemidefinite => {
            PyValueError::new_err("covariance is not positive semidefinite")
        }
        ErrorMetricsError::InvalidProbability => {
            PyValueError::new_err("probability must be in the open interval (0, 1)")
        }
        ErrorMetricsError::Rotation(source) => {
            PyValueError::new_err(format!("ECEF to ENU rotation failed: {source}"))
        }
    }
}

/// Percentile circle or sphere radius.
#[pyclass(module = "sidereon._sidereon", name = "PercentileRadius")]
#[derive(Clone, Copy)]
pub struct PyPercentileRadius {
    inner: PercentileRadius,
}

impl From<PercentileRadius> for PyPercentileRadius {
    fn from(inner: PercentileRadius) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPercentileRadius {
    /// Probability mass inside the radius.
    #[getter]
    fn probability(&self) -> f64 {
        self.inner.probability
    }

    /// Exact radius, metres.
    #[getter]
    fn radius_m(&self) -> f64 {
        self.inner.radius_m
    }

    /// Approximate radius, metres, when a named approximation applies.
    #[getter]
    fn approx_m(&self) -> f64 {
        self.inner.approx_m
    }

    /// Whether `approx_m` is inside the approximation's stated ratio range.
    #[getter]
    fn approx_valid(&self) -> bool {
        self.inner.approx_valid
    }

    fn __repr__(&self) -> String {
        format!(
            "PercentileRadius(probability={}, radius_m={}, approx_valid={})",
            self.inner.probability, self.inner.radius_m, self.inner.approx_valid
        )
    }
}

/// Standard position error metrics from a covariance matrix.
#[pyclass(module = "sidereon._sidereon", name = "PositionErrorMetrics")]
#[derive(Clone, Copy)]
pub struct PyPositionErrorMetrics {
    inner: PositionErrorMetrics,
}

impl From<PositionErrorMetrics> for PyPositionErrorMetrics {
    fn from(inner: PositionErrorMetrics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPositionErrorMetrics {
    /// Horizontal one-sigma error ellipse, metres.
    #[getter]
    fn ellipse(&self) -> PyErrorEllipse {
        PyErrorEllipse::from_one_sigma_m(self.inner.ellipse)
    }

    /// East standard deviation, metres.
    #[getter]
    fn sigma_e_m(&self) -> f64 {
        self.inner.sigma_e_m
    }

    /// North standard deviation, metres.
    #[getter]
    fn sigma_n_m(&self) -> f64 {
        self.inner.sigma_n_m
    }

    /// Up standard deviation, metres.
    #[getter]
    fn sigma_u_m(&self) -> f64 {
        self.inner.sigma_u_m
    }

    /// Horizontal 50 percent circle radius.
    #[getter]
    fn cep_m(&self) -> PyPercentileRadius {
        self.inner.cep_m.into()
    }

    /// Horizontal 95 percent circle radius.
    #[getter]
    fn r95_m(&self) -> PyPercentileRadius {
        self.inner.r95_m.into()
    }

    /// Horizontal 99 percent circle radius.
    #[getter]
    fn r99_m(&self) -> PyPercentileRadius {
        self.inner.r99_m.into()
    }

    /// Distance root mean square, metres.
    #[getter]
    fn drms_m(&self) -> f64 {
        self.inner.drms_m
    }

    /// Twice the distance root mean square, metres.
    #[getter]
    fn two_drms_m(&self) -> f64 {
        self.inner.two_drms_m
    }

    /// Vertical 50 percent one-dimensional radius, metres.
    #[getter]
    fn vep_m(&self) -> f64 {
        self.inner.vep_m
    }

    /// Three-dimensional 50 percent sphere radius.
    #[getter]
    fn sep_m(&self) -> PyPercentileRadius {
        self.inner.sep_m.into()
    }

    /// Mean radial spherical error, metres.
    #[getter]
    fn mrse_m(&self) -> f64 {
        self.inner.mrse_m
    }

    fn __repr__(&self) -> String {
        format!(
            "PositionErrorMetrics(cep_m={}, r95_m={}, mrse_m={})",
            self.inner.cep_m.radius_m, self.inner.r95_m.radius_m, self.inner.mrse_m
        )
    }
}

/// Kinematic position solution value accepted by `metrics_from_kinematic_solution`.
#[pyclass(module = "sidereon._sidereon", name = "KinematicSolution")]
#[derive(Clone, Copy)]
pub struct PyKinematicSolution {
    position_m: [f64; 3],
    position_covariance_m2: [[f64; 3]; 3],
}

impl PyKinematicSolution {
    fn inner(&self) -> KinematicEpochSolution {
        KinematicEpochSolution {
            position_m: self.position_m,
            clock_m: 0.0,
            ztd_residual_m: 0.0,
            ambiguities_m: BTreeMap::new(),
            position_covariance_m2: self.position_covariance_m2,
            used_sats: Vec::new(),
            innovation_rms_m: 0.0,
            status: KinematicEpochStatus::Updated,
        }
    }
}

#[pymethods]
impl PyKinematicSolution {
    /// Build a kinematic position solution from ECEF position and covariance.
    #[new]
    fn new(
        position_m: PyReadonlyArray1<'_, f64>,
        position_covariance_m2: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            position_m: fixed_array("position_m", &position_m, FinitePolicy::RequireFinite)?,
            position_covariance_m2: matrix3_from_array(
                &position_covariance_m2,
                "position_covariance_m2",
                FinitePolicy::RequireFinite,
            )?,
        })
    }

    /// Receiver ECEF position, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.position_m)
    }

    /// ECEF position covariance, square metres.
    #[getter]
    fn position_covariance_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.position_covariance_m2)
    }

    fn __repr__(&self) -> String {
        format!(
            "KinematicSolution(position_m=[{}, {}, {}])",
            self.position_m[0], self.position_m[1], self.position_m[2]
        )
    }
}

/// Compute position error metrics from an ENU covariance in square metres.
#[pyfunction]
fn metrics_from_enu_covariance_m2(
    covariance_enu_m2: PyReadonlyArray2<'_, f64>,
) -> PyResult<PyPositionErrorMetrics> {
    let covariance = matrix3_from_array(
        &covariance_enu_m2,
        "covariance_enu_m2",
        FinitePolicy::AllowNonFinite,
    )?;
    core_metrics_from_enu_covariance_m2(covariance)
        .map(Into::into)
        .map_err(to_metrics_err)
}

/// Rotate an ECEF covariance to ENU and compute position error metrics.
#[pyfunction]
fn metrics_from_ecef_covariance_m2(
    covariance_ecef_m2: PyReadonlyArray2<'_, f64>,
    receiver: &PyWgs84Geodetic,
) -> PyResult<PyPositionErrorMetrics> {
    let covariance = matrix3_from_array(
        &covariance_ecef_m2,
        "covariance_ecef_m2",
        FinitePolicy::AllowNonFinite,
    )?;
    core_metrics_from_ecef_covariance_m2(covariance, receiver.try_into()?)
        .map(Into::into)
        .map_err(to_metrics_err)
}

/// Compute position error metrics from a kinematic position solution.
#[pyfunction]
fn metrics_from_kinematic_solution(
    solution: &PyKinematicSolution,
) -> PyResult<PyPositionErrorMetrics> {
    core_metrics_from_kinematic_solution(&solution.inner())
        .map(Into::into)
        .map_err(to_metrics_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPercentileRadius>()?;
    m.add_class::<PyPositionErrorMetrics>()?;
    m.add_class::<PyKinematicSolution>()?;
    m.add_function(wrap_pyfunction!(metrics_from_enu_covariance_m2, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_from_ecef_covariance_m2, m)?)?;
    m.add_function(wrap_pyfunction!(metrics_from_kinematic_solution, m)?)?;
    Ok(())
}
