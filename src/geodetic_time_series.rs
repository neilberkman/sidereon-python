//! Geodetic position time-series bindings.

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::geodetic_time_series::{
    detect_steps as core_detect_steps, fit_trajectory as core_fit_trajectory,
    network_field as core_network_field, velocity_midas as core_velocity_midas,
    GeodeticTimeSeriesError, MidasComponentStats, MidasOptions, MotionField, NetworkFrame,
    NetworkStation, PositionFrame, PositionSample, PositionSeries, StationMotion, StepCandidate,
    StepDetectionHeuristic, StepDetectionOptions, TimeSeriesQuality, Trajectory,
    TrajectoryComponent, TrajectoryFitOptions, TrajectoryModel, TrajectoryTerm, Velocity,
};

use crate::events::PyWgs84Geodetic;
use crate::geometry_quality::PyGeometryQuality;
use crate::marshal::{fixed_array, mat3_to_array, matrix3_from_array, FinitePolicy};
use crate::np_array;
use crate::propagation::PyLoss;

fn to_series_err(err: GeodeticTimeSeriesError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn matrix_vec_to_array<'py>(py: Python<'py>, values: &[Vec<f64>]) -> Bound<'py, PyArray2<f64>> {
    let rows = values.len();
    let cols = values.first().map(Vec::len).unwrap_or(0);
    let mut array = Array2::<f64>::zeros((rows, cols));
    for (row_index, row) in values.iter().enumerate() {
        for (col_index, value) in row.iter().copied().enumerate() {
            array[[row_index, col_index]] = value;
        }
    }
    array.to_pyarray(py)
}

/// One position sample in a station time series.
#[pyclass(module = "sidereon._sidereon", name = "PositionSample")]
#[derive(Clone, Copy)]
pub struct PyPositionSample {
    inner: PositionSample,
}

impl PyPositionSample {
    fn inner(&self) -> PositionSample {
        self.inner
    }
}

#[pymethods]
impl PyPositionSample {
    /// Build one position sample from decimal-year epoch and metre coordinates.
    #[new]
    #[pyo3(signature = (epoch_year, position_m, covariance_m2=None))]
    fn new(
        epoch_year: f64,
        position_m: PyReadonlyArray1<'_, f64>,
        covariance_m2: Option<PyReadonlyArray2<'_, f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: PositionSample {
                epoch_year,
                position_m: fixed_array("position_m", &position_m, FinitePolicy::RequireFinite)?,
                covariance_m2: covariance_m2
                    .as_ref()
                    .map(|value| {
                        matrix3_from_array(value, "covariance_m2", FinitePolicy::RequireFinite)
                    })
                    .transpose()?,
            },
        })
    }

    /// Decimal-year epoch.
    #[getter]
    fn epoch_year(&self) -> f64 {
        self.inner.epoch_year
    }

    /// Position vector, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_m)
    }

    /// Coordinate covariance in square metres, if supplied.
    #[getter]
    fn covariance_m2<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .covariance_m2
            .map(|covariance| mat3_to_array(py, &covariance))
    }

    fn __repr__(&self) -> String {
        format!("PositionSample(epoch_year={})", self.inner.epoch_year)
    }
}

/// Station position series in ENU or ECEF coordinates.
#[pyclass(module = "sidereon._sidereon", name = "PositionSeries")]
#[derive(Clone)]
pub struct PyPositionSeries {
    frame: PositionFrame,
    samples: Vec<PositionSample>,
}

impl PyPositionSeries {
    fn core(&self) -> PositionSeries<'_> {
        PositionSeries {
            frame: self.frame,
            samples: &self.samples,
        }
    }
}

#[pymethods]
impl PyPositionSeries {
    /// Build a local ENU position series from sample handles.
    #[staticmethod]
    fn enu(py: Python<'_>, samples: Vec<Py<PyPositionSample>>) -> Self {
        Self {
            frame: PositionFrame::Enu,
            samples: samples
                .iter()
                .map(|sample| sample.borrow(py).inner())
                .collect(),
        }
    }

    /// Build an ECEF position series with a geodetic local-frame reference.
    #[staticmethod]
    fn ecef(
        py: Python<'_>,
        reference: &PyWgs84Geodetic,
        samples: Vec<Py<PyPositionSample>>,
    ) -> PyResult<Self> {
        Ok(Self {
            frame: PositionFrame::Ecef {
                reference: reference.try_into()?,
            },
            samples: samples
                .iter()
                .map(|sample| sample.borrow(py).inner())
                .collect(),
        })
    }

    /// Coordinate frame label, `enu` or `ecef`.
    #[getter]
    fn frame(&self) -> &'static str {
        match self.frame {
            PositionFrame::Enu => "enu",
            PositionFrame::Ecef { .. } => "ecef",
        }
    }

    /// Geodetic reference for ECEF input, or `None` for ENU input.
    #[getter]
    fn reference(&self) -> Option<PyWgs84Geodetic> {
        match self.frame {
            PositionFrame::Enu => None,
            PositionFrame::Ecef { reference } => Some(PyWgs84Geodetic::from_core(reference)),
        }
    }

    /// Number of samples.
    fn __len__(&self) -> usize {
        self.samples.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PositionSeries(frame={}, samples={})",
            self.frame(),
            self.samples.len()
        )
    }
}

/// MIDAS robust velocity-estimator options.
#[pyclass(module = "sidereon._sidereon", name = "MidasOptions")]
#[derive(Clone, Copy)]
pub struct PyMidasOptions {
    inner: MidasOptions,
}

impl PyMidasOptions {
    fn inner(&self) -> MidasOptions {
        self.inner
    }
}

#[pymethods]
impl PyMidasOptions {
    /// Build MIDAS options.
    #[new]
    #[pyo3(signature = (
        dominant_period_years=1.0,
        period_tolerance_years=0.001,
        min_pairs=3,
    ))]
    fn new(dominant_period_years: f64, period_tolerance_years: f64, min_pairs: usize) -> Self {
        Self {
            inner: MidasOptions {
                dominant_period_years,
                period_tolerance_years,
                min_pairs,
            },
        }
    }

    /// Dominant period used for pair selection, years.
    #[getter]
    fn dominant_period_years(&self) -> f64 {
        self.inner.dominant_period_years
    }

    /// Allowed absolute difference from the dominant period, years.
    #[getter]
    fn period_tolerance_years(&self) -> f64 {
        self.inner.period_tolerance_years
    }

    /// Minimum retained pair count required for each component.
    #[getter]
    fn min_pairs(&self) -> usize {
        self.inner.min_pairs
    }
}

/// Qualitative strength of a time-series estimate.
#[pyclass(module = "sidereon._sidereon", name = "TimeSeriesQuality", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyTimeSeriesQuality {
    /// The series spans enough dominant periods for the estimator.
    NOMINAL,
    /// The estimate is usable but has limited span.
    SHORT_SPAN,
}

impl From<TimeSeriesQuality> for PyTimeSeriesQuality {
    fn from(value: TimeSeriesQuality) -> Self {
        match value {
            TimeSeriesQuality::Nominal => Self::NOMINAL,
            TimeSeriesQuality::ShortSpan => Self::SHORT_SPAN,
        }
    }
}

#[pymethods]
impl PyTimeSeriesQuality {
    /// Stable lowercase quality label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::NOMINAL => "nominal",
            Self::SHORT_SPAN => "short_span",
        }
    }
}

/// MIDAS diagnostics for one ENU component.
#[pyclass(module = "sidereon._sidereon", name = "MidasComponentStats")]
#[derive(Clone, Copy)]
pub struct PyMidasComponentStats {
    inner: MidasComponentStats,
}

impl From<MidasComponentStats> for PyMidasComponentStats {
    fn from(inner: MidasComponentStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMidasComponentStats {
    /// Pair slopes selected before trimming.
    #[getter]
    fn pair_count(&self) -> usize {
        self.inner.pair_count
    }

    /// Pair slopes retained after trimming.
    #[getter]
    fn retained_pair_count(&self) -> usize {
        self.inner.retained_pair_count
    }

    /// Robust standard deviation of retained pair slopes, metres per year.
    #[getter]
    fn slope_sigma_m_per_yr(&self) -> f64 {
        self.inner.slope_sigma_m_per_yr
    }

    /// Effective number of independent slope samples.
    #[getter]
    fn effective_pair_count(&self) -> f64 {
        self.inner.effective_pair_count
    }
}

/// Robust ENU velocity estimate.
#[pyclass(module = "sidereon._sidereon", name = "Velocity")]
#[derive(Clone)]
pub struct PyVelocity {
    inner: Velocity,
}

impl From<Velocity> for PyVelocity {
    fn from(inner: Velocity) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyVelocity {
    /// Velocity components `[east, north, up]`, metres per year.
    #[getter]
    fn rate_enu_m_per_yr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.rate_enu_m_per_yr)
    }

    /// One-sigma MIDAS uncertainties, metres per year.
    #[getter]
    fn sigma_enu_m_per_yr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.sigma_enu_m_per_yr)
    }

    /// Diagonal ENU velocity covariance, square metres per square year.
    #[getter]
    fn covariance_enu_m2_per_yr2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.covariance_enu_m2_per_yr2)
    }

    /// Per-component MIDAS slope statistics.
    #[getter]
    fn component_stats(&self) -> Vec<PyMidasComponentStats> {
        self.inner
            .component_stats
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// Accepted sample count.
    #[getter]
    fn sample_count(&self) -> usize {
        self.inner.sample_count
    }

    /// Series span, years.
    #[getter]
    fn span_years(&self) -> f64 {
        self.inner.span_years
    }

    /// Estimate quality flag.
    #[getter]
    fn quality(&self) -> PyTimeSeriesQuality {
        self.inner.quality.into()
    }
}

/// Linear trajectory model options.
#[pyclass(module = "sidereon._sidereon", name = "TrajectoryModel")]
#[derive(Clone)]
pub struct PyTrajectoryModel {
    inner: TrajectoryModel,
}

impl PyTrajectoryModel {
    fn inner(&self) -> TrajectoryModel {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTrajectoryModel {
    /// Build a trajectory model.
    #[new]
    #[pyo3(signature = (
        reference_epoch_year=None,
        include_annual=true,
        include_semiannual=true,
        offset_epochs_year=None,
    ))]
    fn new(
        reference_epoch_year: Option<f64>,
        include_annual: bool,
        include_semiannual: bool,
        offset_epochs_year: Option<Vec<f64>>,
    ) -> Self {
        Self {
            inner: TrajectoryModel {
                reference_epoch_year,
                include_annual,
                include_semiannual,
                offset_epochs_year: offset_epochs_year.unwrap_or_default(),
            },
        }
    }

    /// Optional reference epoch, years.
    #[getter]
    fn reference_epoch_year(&self) -> Option<f64> {
        self.inner.reference_epoch_year
    }

    /// Whether annual sine and cosine terms are included.
    #[getter]
    fn include_annual(&self) -> bool {
        self.inner.include_annual
    }

    /// Whether semiannual sine and cosine terms are included.
    #[getter]
    fn include_semiannual(&self) -> bool {
        self.inner.include_semiannual
    }

    /// Known offset epochs, years.
    #[getter]
    fn offset_epochs_year(&self) -> Vec<f64> {
        self.inner.offset_epochs_year.clone()
    }
}

/// Least-squares controls for trajectory fitting.
#[pyclass(module = "sidereon._sidereon", name = "TrajectoryFitOptions")]
#[derive(Clone, Copy)]
pub struct PyTrajectoryFitOptions {
    inner: TrajectoryFitOptions,
}

impl PyTrajectoryFitOptions {
    fn inner(&self) -> TrajectoryFitOptions {
        self.inner
    }
}

#[pymethods]
impl PyTrajectoryFitOptions {
    /// Build trajectory least-squares options.
    #[new]
    #[pyo3(signature = (loss=PyLoss::LINEAR, f_scale_m=1.0, max_nfev=None))]
    fn new(loss: PyLoss, f_scale_m: f64, max_nfev: Option<usize>) -> Self {
        Self {
            inner: TrajectoryFitOptions {
                loss: loss.to_trf_loss(),
                f_scale_m,
                max_nfev,
            },
        }
    }

    /// Robust-loss scale in metres.
    #[getter]
    fn f_scale_m(&self) -> f64 {
        self.inner.f_scale_m
    }

    /// Optional maximum residual evaluations.
    #[getter]
    fn max_nfev(&self) -> Option<usize> {
        self.inner.max_nfev
    }
}

/// One trajectory model term.
#[pyclass(module = "sidereon._sidereon", name = "TrajectoryTerm")]
#[derive(Clone, Copy)]
pub struct PyTrajectoryTerm {
    inner: TrajectoryTerm,
}

impl From<TrajectoryTerm> for PyTrajectoryTerm {
    fn from(inner: TrajectoryTerm) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrajectoryTerm {
    /// Stable term kind label.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            TrajectoryTerm::Position => "position",
            TrajectoryTerm::Velocity => "velocity",
            TrajectoryTerm::AnnualSin => "annual_sin",
            TrajectoryTerm::AnnualCos => "annual_cos",
            TrajectoryTerm::SemiannualSin => "semiannual_sin",
            TrajectoryTerm::SemiannualCos => "semiannual_cos",
            TrajectoryTerm::Offset { .. } => "offset",
        }
    }

    /// Offset index for offset terms.
    #[getter]
    fn index(&self) -> Option<usize> {
        match self.inner {
            TrajectoryTerm::Offset { index, .. } => Some(index),
            _ => None,
        }
    }

    /// Offset epoch for offset terms, years.
    #[getter]
    fn epoch_year(&self) -> Option<f64> {
        match self.inner {
            TrajectoryTerm::Offset { epoch_year, .. } => Some(epoch_year),
            _ => None,
        }
    }
}

/// Fitted trajectory coefficients for one ENU component.
#[pyclass(module = "sidereon._sidereon", name = "TrajectoryComponent")]
#[derive(Clone)]
pub struct PyTrajectoryComponent {
    inner: TrajectoryComponent,
}

impl From<TrajectoryComponent> for PyTrajectoryComponent {
    fn from(inner: TrajectoryComponent) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrajectoryComponent {
    /// Position at the reference epoch, metres.
    #[getter]
    fn position_m(&self) -> f64 {
        self.inner.position_m
    }

    /// Linear velocity, metres per year.
    #[getter]
    fn velocity_m_per_yr(&self) -> f64 {
        self.inner.velocity_m_per_yr
    }

    /// Annual sine coefficient, metres.
    #[getter]
    fn annual_sin_m(&self) -> Option<f64> {
        self.inner.annual_sin_m
    }

    /// Annual cosine coefficient, metres.
    #[getter]
    fn annual_cos_m(&self) -> Option<f64> {
        self.inner.annual_cos_m
    }

    /// Semiannual sine coefficient, metres.
    #[getter]
    fn semiannual_sin_m(&self) -> Option<f64> {
        self.inner.semiannual_sin_m
    }

    /// Semiannual cosine coefficient, metres.
    #[getter]
    fn semiannual_cos_m(&self) -> Option<f64> {
        self.inner.semiannual_cos_m
    }

    /// Heaviside offset coefficients, metres.
    #[getter]
    fn offsets_m(&self) -> Vec<f64> {
        self.inner.offsets_m.clone()
    }
}

/// Trajectory least-squares result.
#[pyclass(module = "sidereon._sidereon", name = "Trajectory")]
#[derive(Clone)]
pub struct PyTrajectory {
    inner: Trajectory,
}

impl From<Trajectory> for PyTrajectory {
    fn from(inner: Trajectory) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTrajectory {
    /// Reference epoch used for position and harmonic phases.
    #[getter]
    fn reference_epoch_year(&self) -> f64 {
        self.inner.reference_epoch_year
    }

    /// Parameter terms within each component block.
    #[getter]
    fn terms(&self) -> Vec<PyTrajectoryTerm> {
        self.inner.terms.iter().copied().map(Into::into).collect()
    }

    /// ENU component coefficients ordered east, north, up.
    #[getter]
    fn components(&self) -> Vec<PyTrajectoryComponent> {
        self.inner
            .components
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Full parameter covariance in solver order.
    #[getter]
    fn parameter_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_vec_to_array(py, &self.inner.parameter_covariance)
    }

    /// Root-mean-square residuals `[east, north, up]`, metres.
    #[getter]
    fn residual_rms_enu_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.residual_rms_enu_m)
    }

    /// Design observability and covariance diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    /// Trust-region termination status.
    #[getter]
    fn status(&self) -> i32 {
        self.inner.status
    }

    /// Residual evaluations used by the solver.
    #[getter]
    fn nfev(&self) -> usize {
        self.inner.nfev
    }

    /// Jacobian evaluations used by the solver.
    #[getter]
    fn njev(&self) -> usize {
        self.inner.njev
    }

    /// Final least-squares cost.
    #[getter]
    fn cost(&self) -> f64 {
        self.inner.cost
    }

    /// Infinity norm of the final gradient.
    #[getter]
    fn optimality(&self) -> f64 {
        self.inner.optimality
    }
}

/// Controls for step detection.
#[pyclass(module = "sidereon._sidereon", name = "StepDetectionOptions")]
#[derive(Clone, Copy)]
pub struct PyStepDetectionOptions {
    inner: StepDetectionOptions,
}

impl PyStepDetectionOptions {
    fn inner(&self) -> StepDetectionOptions {
        self.inner
    }
}

#[pymethods]
impl PyStepDetectionOptions {
    /// Build step detection options.
    #[new]
    #[pyo3(signature = (
        window_years=0.75,
        score_threshold=8.0,
        min_offset_m=1.0e-4,
        min_samples_each_side=4,
        min_separation_years=0.25,
        midas=None,
    ))]
    fn new(
        window_years: f64,
        score_threshold: f64,
        min_offset_m: f64,
        min_samples_each_side: usize,
        min_separation_years: f64,
        midas: Option<&PyMidasOptions>,
    ) -> Self {
        Self {
            inner: StepDetectionOptions {
                window_years,
                score_threshold,
                min_offset_m,
                min_samples_each_side,
                min_separation_years,
                midas: midas.map(PyMidasOptions::inner).unwrap_or_default(),
            },
        }
    }
}

/// Heuristic used to generate a step candidate.
#[pyclass(
    module = "sidereon._sidereon",
    name = "StepDetectionHeuristic",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyStepDetectionHeuristic {
    /// Difference of residual medians after MIDAS detrending.
    DETRENDED_SLIDING_MEDIAN,
}

impl From<StepDetectionHeuristic> for PyStepDetectionHeuristic {
    fn from(value: StepDetectionHeuristic) -> Self {
        match value {
            StepDetectionHeuristic::DetrendedSlidingMedian => Self::DETRENDED_SLIDING_MEDIAN,
        }
    }
}

#[pymethods]
impl PyStepDetectionHeuristic {
    /// Stable lowercase heuristic label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::DETRENDED_SLIDING_MEDIAN => "detrended_sliding_median",
        }
    }
}

/// Candidate displacement step.
#[pyclass(module = "sidereon._sidereon", name = "StepCandidate")]
#[derive(Clone, Copy)]
pub struct PyStepCandidate {
    inner: StepCandidate,
}

impl From<StepCandidate> for PyStepCandidate {
    fn from(inner: StepCandidate) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStepCandidate {
    /// Candidate step epoch, years.
    #[getter]
    fn epoch_year(&self) -> f64 {
        self.inner.epoch_year
    }

    /// Estimated ENU offset after minus before, metres.
    #[getter]
    fn offset_enu_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.offset_enu_m)
    }

    /// Robust normalized offset score.
    #[getter]
    fn score(&self) -> f64 {
        self.inner.score
    }

    /// Number of samples before the candidate used by the score.
    #[getter]
    fn before_count(&self) -> usize {
        self.inner.before_count
    }

    /// Number of samples after the candidate used by the score.
    #[getter]
    fn after_count(&self) -> usize {
        self.inner.after_count
    }

    /// Heuristic label for the diagnostic.
    #[getter]
    fn heuristic(&self) -> PyStepDetectionHeuristic {
        self.inner.heuristic.into()
    }
}

/// Network field frame and filtering controls.
#[pyclass(module = "sidereon._sidereon", name = "NetworkFrame")]
#[derive(Clone, Copy)]
pub struct PyNetworkFrame {
    inner: NetworkFrame,
}

impl PyNetworkFrame {
    fn inner(&self) -> NetworkFrame {
        self.inner
    }
}

#[pymethods]
impl PyNetworkFrame {
    /// Build network frame controls.
    #[new]
    #[pyo3(signature = (origin, remove_common_mode=false))]
    fn new(origin: &PyWgs84Geodetic, remove_common_mode: bool) -> PyResult<Self> {
        Ok(Self {
            inner: NetworkFrame {
                origin: origin.try_into()?,
                remove_common_mode,
            },
        })
    }

    /// Geodetic origin defining the output ENU frame.
    #[getter]
    fn origin(&self) -> PyWgs84Geodetic {
        PyWgs84Geodetic::from_core(self.inner.origin)
    }

    /// Whether common-mode velocity was removed.
    #[getter]
    fn remove_common_mode(&self) -> bool {
        self.inner.remove_common_mode
    }
}

/// Station input for network field estimation.
#[pyclass(module = "sidereon._sidereon", name = "NetworkStation")]
#[derive(Clone)]
pub struct PyNetworkStation {
    id: String,
    reference: sidereon_core::Wgs84Geodetic,
    series: PyPositionSeries,
}

#[pymethods]
impl PyNetworkStation {
    /// Build one network station input.
    #[new]
    fn new(id: String, reference: &PyWgs84Geodetic, series: &PyPositionSeries) -> PyResult<Self> {
        Ok(Self {
            id,
            reference: reference.try_into()?,
            series: series.clone(),
        })
    }

    /// Caller-provided station identifier.
    #[getter]
    fn id(&self) -> String {
        self.id.clone()
    }

    /// Station reference position.
    #[getter]
    fn reference(&self) -> PyWgs84Geodetic {
        PyWgs84Geodetic::from_core(self.reference)
    }

    /// Station position series.
    #[getter]
    fn series(&self) -> PyPositionSeries {
        self.series.clone()
    }
}

/// One station motion in a network field.
#[pyclass(module = "sidereon._sidereon", name = "StationMotion")]
#[derive(Clone)]
pub struct PyStationMotion {
    inner: StationMotion,
}

impl From<StationMotion> for PyStationMotion {
    fn from(inner: StationMotion) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStationMotion {
    /// Station identifier copied from the input.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Velocity in the network frame after optional common-mode removal.
    #[getter]
    fn rate_enu_m_per_yr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.rate_enu_m_per_yr)
    }

    /// Velocity in the network frame before common-mode removal.
    #[getter]
    fn raw_rate_enu_m_per_yr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.raw_rate_enu_m_per_yr)
    }

    /// One-sigma uncertainty in the network frame.
    #[getter]
    fn sigma_enu_m_per_yr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.sigma_enu_m_per_yr)
    }

    /// Station-local MIDAS velocity before rotation into the network frame.
    #[getter]
    fn local_velocity(&self) -> PyVelocity {
        self.inner.local_velocity.clone().into()
    }
}

/// Network motion field.
#[pyclass(module = "sidereon._sidereon", name = "MotionField")]
#[derive(Clone)]
pub struct PyMotionField {
    inner: MotionField,
}

impl From<MotionField> for PyMotionField {
    fn from(inner: MotionField) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMotionField {
    /// Output frame and filtering controls.
    #[getter]
    fn frame(&self) -> PyNetworkFrame {
        PyNetworkFrame {
            inner: self.inner.frame,
        }
    }

    /// Station motions in accepted input order.
    #[getter]
    fn stations(&self) -> Vec<PyStationMotion> {
        self.inner
            .stations
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    /// Unweighted mean velocity removed from station rates.
    #[getter]
    fn common_mode_enu_m_per_yr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.common_mode_enu_m_per_yr)
    }
}

/// Estimate robust station velocity with MIDAS.
#[pyfunction]
#[pyo3(signature = (series, options=None))]
fn velocity_midas(
    series: &PyPositionSeries,
    options: Option<&PyMidasOptions>,
) -> PyResult<PyVelocity> {
    let options = options.map(PyMidasOptions::inner).unwrap_or_default();
    core_velocity_midas(&series.core(), options)
        .map(Into::into)
        .map_err(to_series_err)
}

/// Fit a linear geodetic trajectory model.
#[pyfunction]
#[pyo3(signature = (series, model=None, options=None))]
fn fit_trajectory(
    series: &PyPositionSeries,
    model: Option<&PyTrajectoryModel>,
    options: Option<&PyTrajectoryFitOptions>,
) -> PyResult<PyTrajectory> {
    let model = model.map(PyTrajectoryModel::inner).unwrap_or_default();
    let options = options
        .map(PyTrajectoryFitOptions::inner)
        .unwrap_or_default();
    core_fit_trajectory(&series.core(), &model, options)
        .map(Into::into)
        .map_err(to_series_err)
}

/// Detect candidate displacement steps.
#[pyfunction]
#[pyo3(signature = (series, options=None))]
fn detect_steps(
    series: &PyPositionSeries,
    options: Option<&PyStepDetectionOptions>,
) -> PyResult<Vec<PyStepCandidate>> {
    let options = options
        .map(PyStepDetectionOptions::inner)
        .unwrap_or_default();
    core_detect_steps(&series.core(), options)
        .map(|values| values.into_iter().map(Into::into).collect())
        .map_err(to_series_err)
}

/// Estimate a network motion field.
#[pyfunction]
fn network_field(
    py: Python<'_>,
    stations: Vec<Py<PyNetworkStation>>,
    frame: &PyNetworkFrame,
) -> PyResult<PyMotionField> {
    let borrowed = stations
        .iter()
        .map(|station| station.borrow(py))
        .collect::<Vec<_>>();
    let core_stations = borrowed
        .iter()
        .map(|station| NetworkStation {
            id: station.id.as_str(),
            reference: station.reference,
            series: station.series.core(),
        })
        .collect::<Vec<_>>();
    core_network_field(&core_stations, frame.inner())
        .map(Into::into)
        .map_err(to_series_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPositionSample>()?;
    m.add_class::<PyPositionSeries>()?;
    m.add_class::<PyMidasOptions>()?;
    m.add_class::<PyTimeSeriesQuality>()?;
    m.add_class::<PyMidasComponentStats>()?;
    m.add_class::<PyVelocity>()?;
    m.add_class::<PyTrajectoryModel>()?;
    m.add_class::<PyTrajectoryFitOptions>()?;
    m.add_class::<PyTrajectoryTerm>()?;
    m.add_class::<PyTrajectoryComponent>()?;
    m.add_class::<PyTrajectory>()?;
    m.add_class::<PyStepDetectionOptions>()?;
    m.add_class::<PyStepDetectionHeuristic>()?;
    m.add_class::<PyStepCandidate>()?;
    m.add_class::<PyNetworkFrame>()?;
    m.add_class::<PyNetworkStation>()?;
    m.add_class::<PyStationMotion>()?;
    m.add_class::<PyMotionField>()?;
    m.add_function(wrap_pyfunction!(velocity_midas, m)?)?;
    m.add_function(wrap_pyfunction!(fit_trajectory, m)?)?;
    m.add_function(wrap_pyfunction!(detect_steps, m)?)?;
    m.add_function(wrap_pyfunction!(network_field, m)?)?;
    Ok(())
}
