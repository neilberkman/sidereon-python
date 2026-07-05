//! Uncertainty-aware geofencing binding.
//!
//! Thin marshaling over [`sidereon_core::geofence`]: WGS84 vertices and
//! positions cross as [`Wgs84Geodetic`](crate::events::PyWgs84Geodetic),
//! uncertainty crosses as typed covariance/radius variants, and every
//! containment, distance, probability, and crossing result is produced by the
//! core.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::error_metrics::PercentileRadius;
use sidereon_core::geofence::{
    containment, containment_probability, containment_probability_with_options, crossing,
    crossing_probability, crossing_probability_with_options, distance_to_boundary, CrossingEvent,
    CrossingKind, Fence, GeofenceError as CoreGeofenceError, GeofencePositionEstimate,
    PositionUncertainty, ProbabilityHysteresis, ProbabilityMethod, ProbabilityOptions,
};
use sidereon_core::Wgs84Geodetic;

use crate::events::PyWgs84Geodetic;
use crate::marshal::{mat3_to_array, matrix3_from_array, FinitePolicy};
use crate::GeofenceError;

fn to_geofence_err(err: CoreGeofenceError) -> PyErr {
    GeofenceError::new_err(err.to_string())
}

fn py_position(position: &PyWgs84Geodetic) -> PyResult<Wgs84Geodetic> {
    Wgs84Geodetic::try_from(position)
}

/// Probability integration method for geofence containment.
#[pyclass(
    module = "sidereon._sidereon",
    name = "GeofenceProbabilityMethod",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyGeofenceProbabilityMethod {
    /// Closed-form Gaussian half-space probability at the nearest boundary.
    BOUNDARY_NORMAL,
    /// Fixed angular quadrature over the local planarized fence.
    PLANAR_QUADRATURE,
}

impl From<PyGeofenceProbabilityMethod> for ProbabilityMethod {
    fn from(value: PyGeofenceProbabilityMethod) -> Self {
        match value {
            PyGeofenceProbabilityMethod::BOUNDARY_NORMAL => Self::BoundaryNormal,
            PyGeofenceProbabilityMethod::PLANAR_QUADRATURE => Self::PlanarQuadrature,
        }
    }
}

impl From<ProbabilityMethod> for PyGeofenceProbabilityMethod {
    fn from(value: ProbabilityMethod) -> Self {
        match value {
            ProbabilityMethod::BoundaryNormal => Self::BOUNDARY_NORMAL,
            ProbabilityMethod::PlanarQuadrature => Self::PLANAR_QUADRATURE,
        }
    }
}

#[pymethods]
impl PyGeofenceProbabilityMethod {
    /// Stable lowercase method label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::BOUNDARY_NORMAL => "boundary_normal",
            Self::PLANAR_QUADRATURE => "planar_quadrature",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::BOUNDARY_NORMAL => "GeofenceProbabilityMethod.BOUNDARY_NORMAL",
            Self::PLANAR_QUADRATURE => "GeofenceProbabilityMethod.PLANAR_QUADRATURE",
        }
    }
}

/// Options for geofence probability integration.
#[pyclass(module = "sidereon._sidereon", name = "GeofenceProbabilityOptions")]
#[derive(Clone, Copy)]
pub struct PyGeofenceProbabilityOptions {
    inner: ProbabilityOptions,
}

impl From<&PyGeofenceProbabilityOptions> for ProbabilityOptions {
    fn from(value: &PyGeofenceProbabilityOptions) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyGeofenceProbabilityOptions {
    /// Create probability options.
    #[new]
    #[pyo3(signature = (method=PyGeofenceProbabilityMethod::BOUNDARY_NORMAL))]
    fn new(method: PyGeofenceProbabilityMethod) -> Self {
        Self {
            inner: ProbabilityOptions {
                method: method.into(),
            },
        }
    }

    /// Probability integration method.
    #[getter]
    fn method(&self) -> PyGeofenceProbabilityMethod {
        self.inner.method.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "GeofenceProbabilityOptions(method={})",
            self.method().label()
        )
    }
}

/// Position uncertainty accepted by probabilistic geofencing.
#[pyclass(module = "sidereon._sidereon", name = "GeofencePositionUncertainty")]
#[derive(Clone, Copy)]
pub struct PyGeofencePositionUncertainty {
    inner: PositionUncertainty,
}

impl From<&PyGeofencePositionUncertainty> for PositionUncertainty {
    fn from(value: &PyGeofencePositionUncertainty) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyGeofencePositionUncertainty {
    /// Build uncertainty from a local ENU covariance matrix in square metres.
    #[staticmethod]
    fn enu_covariance_m2(covariance: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PositionUncertainty::EnuCovarianceM2(matrix3_from_array(
                &covariance,
                "covariance",
                FinitePolicy::RequireFinite,
            )?),
        })
    }

    /// Build uncertainty from an ECEF covariance matrix in square metres.
    #[staticmethod]
    fn ecef_covariance_m2(covariance: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            inner: PositionUncertainty::EcefCovarianceM2(matrix3_from_array(
                &covariance,
                "covariance",
                FinitePolicy::RequireFinite,
            )?),
        })
    }

    /// Build isotropic horizontal uncertainty from a percentile radius.
    #[staticmethod]
    fn horizontal_radius(probability: f64, radius_m: f64) -> Self {
        Self {
            inner: PositionUncertainty::HorizontalRadius(PercentileRadius {
                probability,
                radius_m,
                approx_m: radius_m,
                approx_valid: true,
            }),
        }
    }

    /// Build circular-error-probable horizontal uncertainty from a CEP radius.
    #[staticmethod]
    fn cep_radius_m(radius_m: f64) -> Self {
        Self {
            inner: PositionUncertainty::CepRadiusM(radius_m),
        }
    }

    /// Stable variant label.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            PositionUncertainty::EnuCovarianceM2(_) => "enu_covariance_m2",
            PositionUncertainty::EcefCovarianceM2(_) => "ecef_covariance_m2",
            PositionUncertainty::PositionCovariance(_) => "position_covariance",
            PositionUncertainty::HorizontalRadius(_) => "horizontal_radius",
            PositionUncertainty::CepRadiusM(_) => "cep_radius_m",
        }
    }

    /// Covariance matrix for covariance-backed variants, otherwise `None`.
    fn covariance<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        match self.inner {
            PositionUncertainty::EnuCovarianceM2(covariance)
            | PositionUncertainty::EcefCovarianceM2(covariance) => {
                Some(mat3_to_array(py, &covariance))
            }
            PositionUncertainty::PositionCovariance(covariance) => {
                Some(mat3_to_array(py, &covariance.enu_m2))
            }
            PositionUncertainty::HorizontalRadius(_) | PositionUncertainty::CepRadiusM(_) => None,
        }
    }

    fn __repr__(&self) -> String {
        format!("GeofencePositionUncertainty(kind={:?})", self.kind())
    }
}

/// One position estimate and its uncertainty for probabilistic crossing.
#[pyclass(module = "sidereon._sidereon", name = "GeofencePositionEstimate")]
#[derive(Clone, Copy)]
pub struct PyGeofencePositionEstimate {
    inner: GeofencePositionEstimate,
}

impl From<&PyGeofencePositionEstimate> for GeofencePositionEstimate {
    fn from(value: &PyGeofencePositionEstimate) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyGeofencePositionEstimate {
    /// Create a geofence position estimate.
    #[new]
    fn new(
        position: PyRef<'_, PyWgs84Geodetic>,
        uncertainty: &PyGeofencePositionUncertainty,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: GeofencePositionEstimate {
                position: py_position(&position)?,
                uncertainty: uncertainty.into(),
            },
        })
    }

    /// Estimated WGS84 geodetic position.
    #[getter]
    fn position(&self) -> PyWgs84Geodetic {
        PyWgs84Geodetic::from_core(self.inner.position)
    }

    /// Position uncertainty for this estimate.
    #[getter]
    fn uncertainty(&self) -> PyGeofencePositionUncertainty {
        PyGeofencePositionUncertainty {
            inner: self.inner.uncertainty,
        }
    }

    fn __repr__(&self) -> String {
        "GeofencePositionEstimate(...)".to_string()
    }
}

/// Probability hysteresis thresholds for geofence crossing detection.
#[pyclass(module = "sidereon._sidereon", name = "GeofenceProbabilityHysteresis")]
#[derive(Clone, Copy)]
pub struct PyGeofenceProbabilityHysteresis {
    inner: ProbabilityHysteresis,
}

impl From<&PyGeofenceProbabilityHysteresis> for ProbabilityHysteresis {
    fn from(value: &PyGeofenceProbabilityHysteresis) -> Self {
        value.inner
    }
}

#[pymethods]
impl PyGeofenceProbabilityHysteresis {
    /// Create hysteresis thresholds.
    #[new]
    #[pyo3(signature = (
        enter_confidence=ProbabilityHysteresis::default().enter_confidence,
        leave_confidence=ProbabilityHysteresis::default().leave_confidence,
    ))]
    fn new(enter_confidence: f64, leave_confidence: f64) -> PyResult<Self> {
        Ok(Self {
            inner: ProbabilityHysteresis::new(enter_confidence, leave_confidence)
                .map_err(to_geofence_err)?,
        })
    }

    /// Required inside probability before an entered event is emitted.
    #[getter]
    fn enter_confidence(&self) -> f64 {
        self.inner.enter_confidence
    }

    /// Required outside probability before a left event is emitted.
    #[getter]
    fn leave_confidence(&self) -> f64 {
        self.inner.leave_confidence
    }

    fn __repr__(&self) -> String {
        format!(
            "GeofenceProbabilityHysteresis(enter_confidence={}, leave_confidence={})",
            self.inner.enter_confidence, self.inner.leave_confidence
        )
    }
}

/// Direction of a geofence crossing.
#[pyclass(
    module = "sidereon._sidereon",
    name = "GeofenceCrossingKind",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyGeofenceCrossingKind {
    /// The sequence entered the fence.
    ENTERED,
    /// The sequence left the fence.
    LEFT,
}

impl From<CrossingKind> for PyGeofenceCrossingKind {
    fn from(value: CrossingKind) -> Self {
        match value {
            CrossingKind::Entered => Self::ENTERED,
            CrossingKind::Left => Self::LEFT,
        }
    }
}

#[pymethods]
impl PyGeofenceCrossingKind {
    /// Stable lowercase event label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::ENTERED => "entered",
            Self::LEFT => "left",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::ENTERED => "GeofenceCrossingKind.ENTERED",
            Self::LEFT => "GeofenceCrossingKind.LEFT",
        }
    }
}

/// One geofence crossing event.
#[pyclass(module = "sidereon._sidereon", name = "GeofenceCrossingEvent")]
#[derive(Clone, Copy)]
pub struct PyGeofenceCrossingEvent {
    inner: CrossingEvent,
}

impl From<CrossingEvent> for PyGeofenceCrossingEvent {
    fn from(inner: CrossingEvent) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGeofenceCrossingEvent {
    /// Index of the sample that first satisfied the crossing condition.
    #[getter]
    fn sample_index(&self) -> usize {
        self.inner.sample_index
    }

    /// Direction of the crossing.
    #[getter]
    fn kind(&self) -> PyGeofenceCrossingKind {
        self.inner.kind.into()
    }

    /// Inside probability at the event sample.
    #[getter]
    fn inside_probability(&self) -> f64 {
        self.inner.inside_probability
    }

    fn __repr__(&self) -> String {
        format!(
            "GeofenceCrossingEvent(sample_index={}, kind={}, inside_probability={})",
            self.inner.sample_index,
            self.kind().label(),
            self.inner.inside_probability
        )
    }
}

/// Geodesic polygon fence on WGS84.
#[pyclass(module = "sidereon._sidereon", name = "Geofence")]
pub struct PyGeofence {
    inner: Fence,
}

#[pymethods]
impl PyGeofence {
    /// Construct a fence from WGS84 geodetic vertices.
    #[new]
    fn new(vertices: Vec<PyRef<'_, PyWgs84Geodetic>>) -> PyResult<Self> {
        let vertices = vertices
            .iter()
            .map(|vertex| py_position(vertex))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: Fence::new(vertices).map_err(to_geofence_err)?,
        })
    }

    /// Fence vertices in open polygon form.
    #[getter]
    fn vertices(&self) -> Vec<PyWgs84Geodetic> {
        self.inner
            .vertices()
            .iter()
            .copied()
            .map(PyWgs84Geodetic::from_core)
            .collect()
    }

    /// Number of vertices.
    #[getter]
    fn vertex_count(&self) -> usize {
        self.inner.vertices().len()
    }

    /// Number of polygon edges.
    #[getter]
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Whether the core's planar path can evaluate this position.
    fn planar_fast_path_applies(&self, position: PyRef<'_, PyWgs84Geodetic>) -> PyResult<bool> {
        Ok(self.inner.planar_fast_path_applies(py_position(&position)?))
    }

    /// Boolean containment for one position.
    fn contains(&self, position: PyRef<'_, PyWgs84Geodetic>) -> PyResult<bool> {
        containment(py_position(&position)?, &self.inner).map_err(to_geofence_err)
    }

    /// Signed distance from a position to the fence boundary, metres.
    fn distance_to_boundary(&self, position: PyRef<'_, PyWgs84Geodetic>) -> PyResult<f64> {
        distance_to_boundary(py_position(&position)?, &self.inner).map_err(to_geofence_err)
    }

    /// Containment probability for one uncertain position.
    #[pyo3(signature = (position, uncertainty, options=None))]
    fn containment_probability(
        &self,
        position: PyRef<'_, PyWgs84Geodetic>,
        uncertainty: &PyGeofencePositionUncertainty,
        options: Option<&PyGeofenceProbabilityOptions>,
    ) -> PyResult<f64> {
        let position = py_position(&position)?;
        let uncertainty = uncertainty.into();
        match options {
            Some(options) => containment_probability_with_options(
                position,
                uncertainty,
                &self.inner,
                options.into(),
            ),
            None => containment_probability(position, uncertainty, &self.inner),
        }
        .map_err(to_geofence_err)
    }

    /// Boolean crossing events over a position sequence.
    fn crossing(
        &self,
        positions: Vec<PyRef<'_, PyWgs84Geodetic>>,
    ) -> PyResult<Vec<PyGeofenceCrossingEvent>> {
        let positions = positions
            .iter()
            .map(|position| py_position(position))
            .collect::<PyResult<Vec<_>>>()?;
        crossing(&positions, &self.inner)
            .map(|events| events.into_iter().map(Into::into).collect())
            .map_err(to_geofence_err)
    }

    /// Probabilistic crossing events over uncertain position estimates.
    #[pyo3(signature = (samples, hysteresis, options=None))]
    fn crossing_probability(
        &self,
        samples: Vec<PyRef<'_, PyGeofencePositionEstimate>>,
        hysteresis: &PyGeofenceProbabilityHysteresis,
        options: Option<&PyGeofenceProbabilityOptions>,
    ) -> PyResult<Vec<PyGeofenceCrossingEvent>> {
        let samples = samples
            .iter()
            .map(|sample| sample.inner)
            .collect::<Vec<_>>();
        let hysteresis = hysteresis.into();
        match options {
            Some(options) => {
                crossing_probability_with_options(&samples, &self.inner, hysteresis, options.into())
            }
            None => crossing_probability(&samples, &self.inner, hysteresis),
        }
        .map(|events| events.into_iter().map(Into::into).collect())
        .map_err(to_geofence_err)
    }

    fn __repr__(&self) -> String {
        format!("Geofence(vertices={})", self.inner.vertices().len())
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyGeofenceProbabilityMethod>()?;
    m.add_class::<PyGeofenceProbabilityOptions>()?;
    m.add_class::<PyGeofencePositionUncertainty>()?;
    m.add_class::<PyGeofencePositionEstimate>()?;
    m.add_class::<PyGeofenceProbabilityHysteresis>()?;
    m.add_class::<PyGeofenceCrossingKind>()?;
    m.add_class::<PyGeofenceCrossingEvent>()?;
    m.add_class::<PyGeofence>()?;
    Ok(())
}
