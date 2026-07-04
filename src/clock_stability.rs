//! Allan-family clock-stability bindings.

use numpy::PyArray1;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::clock_stability::{
    allan_deviation as core_allan_deviation,
    compute_allan_deviations as core_compute_allan_deviations,
    hadamard_deviation as core_hadamard_deviation, modified_adev as core_modified_adev,
    overlapping_adev as core_overlapping_adev, time_deviation as core_time_deviation,
    AllanDeviationCurves, AllanError, AllanEstimatorSet, AllanInput, AllanOptions, AllanResult,
    AllanSeries, GapPolicy, TauGrid,
};

use crate::np_array;

fn to_allan_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn vec_from_array(values: PyReadonlyArray1<'_, f64>, name: &str) -> PyResult<Vec<f64>> {
    values
        .as_slice()
        .map(|slice| slice.to_vec())
        .map_err(|err| PyValueError::new_err(format!("{name} must be contiguous: {err}")))
}

/// Tagged input samples for Allan-family estimators.
#[pyclass(module = "sidereon._sidereon", name = "AllanSeries")]
#[derive(Clone)]
pub struct PyAllanSeries {
    data: PyAllanSeriesData,
}

#[derive(Clone)]
enum PyAllanSeriesData {
    PhaseSeconds(Vec<f64>),
    FractionalFrequency(Vec<f64>),
    PhaseSecondsWithGaps(Vec<Option<f64>>),
    FractionalFrequencyWithGaps(Vec<Option<f64>>),
}

impl PyAllanSeries {
    fn with_core<T>(
        &self,
        f: impl FnOnce(AllanSeries<'_>) -> Result<T, AllanError>,
    ) -> PyResult<T> {
        let result = match &self.data {
            PyAllanSeriesData::PhaseSeconds(values) => f(AllanSeries::PhaseSeconds(values)),
            PyAllanSeriesData::FractionalFrequency(values) => {
                f(AllanSeries::FractionalFrequency(values))
            }
            PyAllanSeriesData::PhaseSecondsWithGaps(values) => {
                f(AllanSeries::PhaseSecondsWithGaps(values))
            }
            PyAllanSeriesData::FractionalFrequencyWithGaps(values) => {
                f(AllanSeries::FractionalFrequencyWithGaps(values))
            }
        };
        result.map_err(to_allan_err)
    }

    fn len(&self) -> usize {
        match &self.data {
            PyAllanSeriesData::PhaseSeconds(values)
            | PyAllanSeriesData::FractionalFrequency(values) => values.len(),
            PyAllanSeriesData::PhaseSecondsWithGaps(values)
            | PyAllanSeriesData::FractionalFrequencyWithGaps(values) => values.len(),
        }
    }

    fn missing_count(&self) -> usize {
        match &self.data {
            PyAllanSeriesData::PhaseSeconds(_) | PyAllanSeriesData::FractionalFrequency(_) => 0,
            PyAllanSeriesData::PhaseSecondsWithGaps(values)
            | PyAllanSeriesData::FractionalFrequencyWithGaps(values) => {
                values.iter().filter(|value| value.is_none()).count()
            }
        }
    }
}

#[pymethods]
impl PyAllanSeries {
    /// Build a phase-deviation series in seconds from a numpy `float64` array.
    #[staticmethod]
    fn phase_seconds(values: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            data: PyAllanSeriesData::PhaseSeconds(vec_from_array(values, "values")?),
        })
    }

    /// Build a fractional-frequency series from a numpy `float64` array.
    #[staticmethod]
    fn fractional_frequency(values: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        Ok(Self {
            data: PyAllanSeriesData::FractionalFrequency(vec_from_array(values, "values")?),
        })
    }

    /// Build a phase-deviation series in seconds with missing samples.
    #[staticmethod]
    fn phase_seconds_with_gaps(values: Vec<Option<f64>>) -> Self {
        Self {
            data: PyAllanSeriesData::PhaseSecondsWithGaps(values),
        }
    }

    /// Build a fractional-frequency series with missing samples.
    #[staticmethod]
    fn fractional_frequency_with_gaps(values: Vec<Option<f64>>) -> Self {
        Self {
            data: PyAllanSeriesData::FractionalFrequencyWithGaps(values),
        }
    }

    /// Number of supplied samples.
    #[getter]
    fn sample_count(&self) -> usize {
        self.len()
    }

    /// Number of missing samples.
    #[getter]
    fn missing_sample_count(&self) -> usize {
        self.missing_count()
    }

    /// Stable input-kind label.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.data {
            PyAllanSeriesData::PhaseSeconds(_) => "phase_seconds",
            PyAllanSeriesData::FractionalFrequency(_) => "fractional_frequency",
            PyAllanSeriesData::PhaseSecondsWithGaps(_) => "phase_seconds_with_gaps",
            PyAllanSeriesData::FractionalFrequencyWithGaps(_) => "fractional_frequency_with_gaps",
        }
    }

    fn __len__(&self) -> usize {
        self.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "AllanSeries(kind={}, sample_count={}, missing_sample_count={})",
            self.kind(),
            self.len(),
            self.missing_count()
        )
    }
}

/// Averaging-factor grid for requested Allan-family estimator points.
#[pyclass(module = "sidereon._sidereon", name = "TauGrid")]
#[derive(Clone)]
pub struct PyTauGrid {
    inner: TauGrid,
}

impl PyTauGrid {
    fn inner(&self) -> TauGrid {
        self.inner.clone()
    }
}

#[pymethods]
impl PyTauGrid {
    /// Use averaging factors `1, 2, 4, 8, ...` while terms exist.
    #[staticmethod]
    fn octave() -> Self {
        Self {
            inner: TauGrid::Octave,
        }
    }

    /// Use every positive averaging factor while terms exist.
    #[staticmethod]
    fn all() -> Self {
        Self {
            inner: TauGrid::All,
        }
    }

    /// Use caller-supplied positive averaging factors.
    #[staticmethod]
    fn explicit(averaging_factors: Vec<usize>) -> Self {
        Self {
            inner: TauGrid::Explicit(averaging_factors),
        }
    }

    /// Stable tau-grid label.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            TauGrid::Octave => "octave",
            TauGrid::All => "all",
            TauGrid::Explicit(_) => "explicit",
        }
    }

    /// Explicit averaging factors, or `None` for generated grids.
    #[getter]
    fn averaging_factors(&self) -> Option<Vec<usize>> {
        match &self.inner {
            TauGrid::Explicit(values) => Some(values.clone()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            TauGrid::Explicit(values) => format!("TauGrid.explicit({values:?})"),
            TauGrid::Octave => "TauGrid.octave()".to_string(),
            TauGrid::All => "TauGrid.all()".to_string(),
        }
    }
}

/// Missing-sample policy for gapped Allan-family input.
#[pyclass(module = "sidereon._sidereon", name = "GapPolicy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyGapPolicy {
    /// Reject missing samples before estimation.
    REJECT,
    /// Omit estimator terms that cross a missing sample.
    OMIT_TERMS,
}

impl From<PyGapPolicy> for GapPolicy {
    fn from(value: PyGapPolicy) -> Self {
        match value {
            PyGapPolicy::REJECT => Self::Reject,
            PyGapPolicy::OMIT_TERMS => Self::OmitTerms,
        }
    }
}

impl From<GapPolicy> for PyGapPolicy {
    fn from(value: GapPolicy) -> Self {
        match value {
            GapPolicy::Reject => Self::REJECT,
            GapPolicy::OmitTerms => Self::OMIT_TERMS,
        }
    }
}

#[pymethods]
impl PyGapPolicy {
    /// Stable policy label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::REJECT => "reject",
            Self::OMIT_TERMS => "omit_terms",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::REJECT => "GapPolicy.REJECT",
            Self::OMIT_TERMS => "GapPolicy.OMIT_TERMS",
        }
    }
}

/// Estimator identifier for combined Allan-family options.
#[pyclass(module = "sidereon._sidereon", name = "AllanEstimator", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyAllanEstimator {
    /// Plain non-overlapping Allan deviation.
    ADEV,
    /// Fully overlapping Allan deviation.
    OVERLAPPING_ADEV,
    /// Modified Allan deviation.
    MDEV,
    /// Overlapping Hadamard deviation.
    HDEV,
    /// Time deviation.
    TDEV,
}

#[pymethods]
impl PyAllanEstimator {
    /// Stable estimator label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::ADEV => "adev",
            Self::OVERLAPPING_ADEV => "overlapping_adev",
            Self::MDEV => "mdev",
            Self::HDEV => "hdev",
            Self::TDEV => "tdev",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::ADEV => "AllanEstimator.ADEV",
            Self::OVERLAPPING_ADEV => "AllanEstimator.OVERLAPPING_ADEV",
            Self::MDEV => "AllanEstimator.MDEV",
            Self::HDEV => "AllanEstimator.HDEV",
            Self::TDEV => "AllanEstimator.TDEV",
        }
    }
}

/// Boolean estimator selection for the combined Allan-family driver.
#[pyclass(module = "sidereon._sidereon", name = "AllanEstimatorSet")]
#[derive(Clone, Copy)]
pub struct PyAllanEstimatorSet {
    inner: AllanEstimatorSet,
}

impl PyAllanEstimatorSet {
    fn inner(&self) -> AllanEstimatorSet {
        self.inner
    }
}

#[pymethods]
impl PyAllanEstimatorSet {
    /// Build an estimator selection.
    #[new]
    #[pyo3(signature = (adev=false, overlapping_adev=true, mdev=true, hdev=true, tdev=true))]
    fn new(adev: bool, overlapping_adev: bool, mdev: bool, hdev: bool, tdev: bool) -> Self {
        Self {
            inner: AllanEstimatorSet {
                adev,
                overlapping_adev,
                mdev,
                hdev,
                tdev,
            },
        }
    }

    /// Select no estimators.
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: AllanEstimatorSet::none(),
        }
    }

    /// Select the standard overlapping estimators plus TDEV.
    #[staticmethod]
    fn standard() -> Self {
        Self {
            inner: AllanEstimatorSet::standard(),
        }
    }

    /// Select every implemented estimator.
    #[staticmethod]
    fn all() -> Self {
        Self {
            inner: AllanEstimatorSet::all(),
        }
    }

    /// Whether plain non-overlapping Allan deviation is selected.
    #[getter]
    fn adev(&self) -> bool {
        self.inner.adev
    }

    /// Whether fully overlapping Allan deviation is selected.
    #[getter]
    fn overlapping_adev(&self) -> bool {
        self.inner.overlapping_adev
    }

    /// Whether modified Allan deviation is selected.
    #[getter]
    fn mdev(&self) -> bool {
        self.inner.mdev
    }

    /// Whether overlapping Hadamard deviation is selected.
    #[getter]
    fn hdev(&self) -> bool {
        self.inner.hdev
    }

    /// Whether time deviation is selected.
    #[getter]
    fn tdev(&self) -> bool {
        self.inner.tdev
    }

    fn __repr__(&self) -> String {
        format!(
            "AllanEstimatorSet(adev={}, overlapping_adev={}, mdev={}, hdev={}, tdev={})",
            self.inner.adev,
            self.inner.overlapping_adev,
            self.inner.mdev,
            self.inner.hdev,
            self.inner.tdev
        )
    }
}

/// Options for the combined Allan-family driver.
#[pyclass(module = "sidereon._sidereon", name = "AllanOptions")]
#[derive(Clone)]
pub struct PyAllanOptions {
    inner: AllanOptions,
}

impl PyAllanOptions {
    fn inner(&self) -> AllanOptions {
        self.inner.clone()
    }
}

#[pymethods]
impl PyAllanOptions {
    /// Build Allan-family estimator options.
    #[new]
    #[pyo3(signature = (estimators=None, tau_grid=None, gap_policy=PyGapPolicy::REJECT))]
    fn new(
        estimators: Option<&PyAllanEstimatorSet>,
        tau_grid: Option<&PyTauGrid>,
        gap_policy: PyGapPolicy,
    ) -> Self {
        Self {
            inner: AllanOptions {
                estimators: estimators.map(|value| value.inner()).unwrap_or_default(),
                tau_grid: tau_grid.map(|value| value.inner()).unwrap_or_default(),
                gap_policy: gap_policy.into(),
            },
        }
    }

    /// Selected estimators.
    #[getter]
    fn estimators(&self) -> PyAllanEstimatorSet {
        PyAllanEstimatorSet {
            inner: self.inner.estimators,
        }
    }

    /// Averaging-factor grid.
    #[getter]
    fn tau_grid(&self) -> PyTauGrid {
        PyTauGrid {
            inner: self.inner.tau_grid.clone(),
        }
    }

    /// Missing-sample policy.
    #[getter]
    fn gap_policy(&self) -> PyGapPolicy {
        self.inner.gap_policy.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "AllanOptions(estimators={:?}, tau_grid={}, gap_policy={})",
            self.inner.estimators,
            self.tau_grid().kind(),
            self.gap_policy().label()
        )
    }
}

/// Input package for the combined Allan-family driver.
#[pyclass(module = "sidereon._sidereon", name = "AllanInput")]
#[derive(Clone)]
pub struct PyAllanInput {
    series: PyAllanSeries,
    tau0_s: f64,
    options: AllanOptions,
}

#[pymethods]
impl PyAllanInput {
    /// Build an Allan-family combined-driver input.
    #[new]
    #[pyo3(signature = (series, tau0_s, options=None))]
    fn new(series: &PyAllanSeries, tau0_s: f64, options: Option<&PyAllanOptions>) -> Self {
        Self {
            series: series.clone(),
            tau0_s,
            options: options.map(|value| value.inner()).unwrap_or_default(),
        }
    }

    /// Tagged sample series.
    #[getter]
    fn series(&self) -> PyAllanSeries {
        self.series.clone()
    }

    /// Basic sampling interval in seconds.
    #[getter]
    fn tau0_s(&self) -> f64 {
        self.tau0_s
    }

    /// Estimator, tau-grid, and gap options.
    #[getter]
    fn options(&self) -> PyAllanOptions {
        PyAllanOptions {
            inner: self.options.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "AllanInput(series={}, tau0_s={})",
            self.series.kind(),
            self.tau0_s
        )
    }
}

/// One Allan-family estimator curve.
#[pyclass(module = "sidereon._sidereon", name = "AllanResult")]
#[derive(Clone)]
pub struct PyAllanResult {
    inner: AllanResult,
}

impl From<AllanResult> for PyAllanResult {
    fn from(inner: AllanResult) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAllanResult {
    /// Averaging times as a numpy `(n,)` array, seconds.
    #[getter]
    fn tau_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.tau_s)
    }

    /// Deviation value for each averaging time as a numpy `(n,)` array.
    #[getter]
    fn deviation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.deviation)
    }

    /// Number of estimator terms used at each averaging time.
    #[getter]
    fn n(&self) -> Vec<usize> {
        self.inner.n.clone()
    }

    /// Number of tau points in the curve.
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("AllanResult(points={})", self.inner.len())
    }
}

/// Combined output from the Allan-family driver.
#[pyclass(module = "sidereon._sidereon", name = "AllanDeviationCurves")]
#[derive(Clone)]
pub struct PyAllanDeviationCurves {
    inner: AllanDeviationCurves,
}

impl From<AllanDeviationCurves> for PyAllanDeviationCurves {
    fn from(inner: AllanDeviationCurves) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAllanDeviationCurves {
    /// Plain non-overlapping Allan deviation, if requested.
    #[getter]
    fn adev(&self) -> Option<PyAllanResult> {
        self.inner.adev.clone().map(Into::into)
    }

    /// Fully overlapping Allan deviation, if requested.
    #[getter]
    fn overlapping_adev(&self) -> Option<PyAllanResult> {
        self.inner.overlapping_adev.clone().map(Into::into)
    }

    /// Modified Allan deviation, if requested.
    #[getter]
    fn mdev(&self) -> Option<PyAllanResult> {
        self.inner.mdev.clone().map(Into::into)
    }

    /// Overlapping Hadamard deviation, if requested.
    #[getter]
    fn hdev(&self) -> Option<PyAllanResult> {
        self.inner.hdev.clone().map(Into::into)
    }

    /// Time deviation, if requested.
    #[getter]
    fn tdev(&self) -> Option<PyAllanResult> {
        self.inner.tdev.clone().map(Into::into)
    }

    fn __repr__(&self) -> &'static str {
        "AllanDeviationCurves()"
    }
}

/// Compute the requested Allan-family curves.
#[pyfunction]
fn compute_allan_deviations(input: &PyAllanInput) -> PyResult<PyAllanDeviationCurves> {
    input
        .series
        .with_core(|series| {
            core_compute_allan_deviations(&AllanInput {
                series,
                tau0_s: input.tau0_s,
                options: input.options.clone(),
            })
        })
        .map(Into::into)
}

/// Plain non-overlapping Allan deviation for explicit averaging factors.
#[pyfunction]
fn allan_deviation(
    series: &PyAllanSeries,
    tau0_s: f64,
    averaging_factors: Vec<usize>,
) -> PyResult<PyAllanResult> {
    series
        .with_core(|series| core_allan_deviation(series, tau0_s, &averaging_factors))
        .map(Into::into)
}

/// Fully overlapping Allan deviation for explicit averaging factors.
#[pyfunction]
fn overlapping_adev(
    series: &PyAllanSeries,
    tau0_s: f64,
    averaging_factors: Vec<usize>,
) -> PyResult<PyAllanResult> {
    series
        .with_core(|series| core_overlapping_adev(series, tau0_s, &averaging_factors))
        .map(Into::into)
}

/// Modified Allan deviation for explicit averaging factors.
#[pyfunction]
fn modified_adev(
    series: &PyAllanSeries,
    tau0_s: f64,
    averaging_factors: Vec<usize>,
) -> PyResult<PyAllanResult> {
    series
        .with_core(|series| core_modified_adev(series, tau0_s, &averaging_factors))
        .map(Into::into)
}

/// Overlapping Hadamard deviation for explicit averaging factors.
#[pyfunction]
fn hadamard_deviation(
    series: &PyAllanSeries,
    tau0_s: f64,
    averaging_factors: Vec<usize>,
) -> PyResult<PyAllanResult> {
    series
        .with_core(|series| core_hadamard_deviation(series, tau0_s, &averaging_factors))
        .map(Into::into)
}

/// Time deviation for explicit averaging factors.
#[pyfunction]
fn time_deviation(
    series: &PyAllanSeries,
    tau0_s: f64,
    averaging_factors: Vec<usize>,
) -> PyResult<PyAllanResult> {
    series
        .with_core(|series| core_time_deviation(series, tau0_s, &averaging_factors))
        .map(Into::into)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAllanSeries>()?;
    m.add_class::<PyTauGrid>()?;
    m.add_class::<PyGapPolicy>()?;
    m.add_class::<PyAllanEstimator>()?;
    m.add_class::<PyAllanEstimatorSet>()?;
    m.add_class::<PyAllanOptions>()?;
    m.add_class::<PyAllanInput>()?;
    m.add_class::<PyAllanResult>()?;
    m.add_class::<PyAllanDeviationCurves>()?;
    m.add_function(wrap_pyfunction!(compute_allan_deviations, m)?)?;
    m.add_function(wrap_pyfunction!(allan_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(overlapping_adev, m)?)?;
    m.add_function(wrap_pyfunction!(modified_adev, m)?)?;
    m.add_function(wrap_pyfunction!(hadamard_deviation, m)?)?;
    m.add_function(wrap_pyfunction!(time_deviation, m)?)?;
    Ok(())
}
