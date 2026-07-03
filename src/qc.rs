//! Quality-control binding: residual chi-square RAIM and fault detection and
//! exclusion (FDE).
//!
//! Thin marshaling over [`sidereon_core::quality`]: [`qc_raim`] runs the
//! residual-based chi-square integrity test over a solution's used satellites and
//! residuals; [`qc_fde`] delegates to the core [`fde_spp`] driver, which runs the
//! single-point solve over the shrinking observation set and excludes the worst
//! satellite until RAIM passes (or the exclusion budget is exhausted). No
//! statistics, solve, or exclusion loop lives here; the numbers are exactly what
//! `sidereon-core` produces.

use std::collections::BTreeMap;

use numpy::PyArray1;
use pyo3::exceptions::{PyDeprecationWarning, PyValueError};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::positioning::ReceiverSolution;
use sidereon_core::qc_obs::{
    observation_qc_with_options as core_observation_qc_with_options, IntervalSource,
    ObservationDataGap, ObservationQcNote, ObservationQcOptions, ObservationQcReport,
    SatelliteObservationQc, SatelliteSignalQc, SnrStats, SsiHistogram, SystemSignalQc,
};
use sidereon_core::quality::{
    self, fde_spp, raim_fde_design as core_raim_fde_design, FdeError, FdeOptions, FdeSppError,
    FdeSppOptions, RaimInput, RaimOptions, RaimWeights, RangeChiSquareTest, RangeFdeOptions,
    RangeFdeResult, RangeFdeRow, RangeMeasurementDiagnostic, SolutionValidationOptions,
    DEFAULT_P_FA,
};

use crate::marshal::PyGnssSystem;
use crate::rinex::{PyObsEpochTime, PyRinexObs};
use crate::spp::{PySppConfig, PySppRobustConfig};
use crate::{np_array, PySp3, SolveError};

fn raim_weights(weights: Option<BTreeMap<String, f64>>) -> RaimWeights {
    match weights {
        None => RaimWeights::Unit,
        Some(map) => RaimWeights::BySatellite(map),
    }
}

fn raim_options(
    p_fa: f64,
    weights: Option<BTreeMap<String, f64>>,
    n_systems: Option<isize>,
) -> RaimOptions {
    RaimOptions {
        p_fa,
        weights: raim_weights(weights),
        n_systems,
    }
}

/// The result of a residual chi-square RAIM test.
#[pyclass(module = "sidereon._sidereon", name = "RaimResult")]
pub struct PyRaimResult {
    inner: quality::RaimResult,
}

#[pymethods]
impl PyRaimResult {
    /// `True` when the test statistic exceeds the chi-square threshold.
    #[getter]
    fn fault_detected(&self) -> bool {
        self.inner.fault_detected
    }

    /// Weighted residual sum of squares.
    #[getter]
    fn test_statistic(&self) -> f64 {
        self.inner.test_statistic
    }

    /// Chi-square threshold, `None` when the geometry is not testable.
    #[getter]
    fn threshold(&self) -> Option<f64> {
        self.inner.threshold
    }

    /// Degrees of freedom, `n_used - (3 + n_systems)`.
    #[getter]
    fn dof(&self) -> isize {
        self.inner.dof
    }

    /// `False` when `dof <= 0`.
    #[getter]
    fn testable(&self) -> bool {
        self.inner.testable
    }

    /// Per-satellite standardized residuals as a dict of `token -> value`.
    #[getter]
    fn normalized_residuals(&self) -> BTreeMap<String, f64> {
        self.inner.normalized_residuals.clone()
    }

    /// Satellite token with the largest absolute standardized residual.
    #[getter]
    fn worst_sat(&self) -> Option<String> {
        self.inner.worst_sat.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RaimResult(fault_detected={}, test_statistic={:.6}, dof={})",
            self.inner.fault_detected, self.inner.test_statistic, self.inner.dof
        )
    }
}

/// Residual-based chi-square RAIM over a solution's used satellites and residuals.
///
/// `used_sats` are the satellite tokens in residual order; `residuals_m` are the
/// post-fit pseudorange residuals (metres). `p_fa` is the false-alarm
/// probability; `weights` is an optional dict of per-satellite inverse-variance
/// weights (unit weights when omitted); `n_systems` optionally overrides the
/// number of distinct GNSS clock systems. Returns a `RaimResult`. Raises
/// `ValueError` on malformed input.
#[pyfunction]
#[pyo3(signature = (used_sats, residuals_m, p_fa, weights=None, n_systems=None))]
fn qc_raim(
    used_sats: Vec<String>,
    residuals_m: Vec<f64>,
    p_fa: f64,
    weights: Option<BTreeMap<String, f64>>,
    n_systems: Option<isize>,
) -> PyResult<PyRaimResult> {
    let input = RaimInput {
        used_sats,
        residuals_m,
    };
    let options = raim_options(p_fa, weights, n_systems);
    let inner =
        quality::raim(&input, &options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRaimResult { inner })
}

/// The result of a fault-detection-and-exclusion loop.
#[pyclass(module = "sidereon._sidereon", name = "FdeResult")]
pub struct PyFdeResult {
    solution: ReceiverSolution,
    excluded: Vec<String>,
    iterations: usize,
}

#[pymethods]
impl PyFdeResult {
    /// Final accepted ECEF position as a numpy array `[x_m, y_m, z_m]`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = &self.solution.position;
        np_array(py, &[p.x_m, p.y_m, p.z_m])
    }

    /// Receiver clock bias in seconds.
    #[getter]
    fn rx_clock_s(&self) -> f64 {
        self.solution.rx_clock_s
    }

    /// `(lat_rad, lon_rad, height_m)` if the solve was asked for geodetic.
    #[getter]
    fn geodetic(&self) -> Option<(f64, f64, f64)> {
        self.solution
            .geodetic
            .map(|g| (g.lat_rad, g.lon_rad, g.height_m))
    }

    /// Satellite tokens used in the final accepted solution.
    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.solution
            .used_sats
            .iter()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// Post-fit residuals (metres), index-aligned to `used_sats`.
    #[getter]
    fn residuals_m(&self) -> Vec<f64> {
        self.solution.residuals_m.clone()
    }

    /// Excluded satellite tokens, in exclusion order.
    #[getter]
    fn excluded(&self) -> Vec<String> {
        self.excluded.clone()
    }

    /// Number of exclusions performed before RAIM passed.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "FdeResult(used_sats={}, excluded={:?}, iterations={})",
            self.solution.used_sats.len(),
            self.excluded,
            self.iterations
        )
    }
}

/// Fault detection and exclusion over an SPP solve.
///
/// `config` is the `SppConfig` to solve; its observation set is the starting set
/// the loop shrinks. After each solve RAIM is run (with `p_fa`, optional
/// per-satellite `weights`, and optional `n_systems`); if a fault is detected the
/// worst satellite is excluded and the solve re-run, up to `max_iterations`
/// exclusions. `max_pdop` optionally caps the accepted geometry. Returns an
/// `FdeResult`. Raises `SolveError` if a solve fails or the fault is unresolved,
/// and `ValueError` on malformed input.
#[pyfunction]
#[pyo3(signature = (sp3, config, p_fa, max_iterations, weights=None, n_systems=None, max_pdop=None))]
#[allow(clippy::too_many_arguments)]
fn qc_fde(
    sp3: &PySp3,
    config: &PySppConfig,
    p_fa: f64,
    max_iterations: usize,
    weights: Option<BTreeMap<String, f64>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
) -> PyResult<PyFdeResult> {
    let inputs = config.to_inputs();
    let with_geodetic = config.with_geodetic_flag();
    let options = FdeSppOptions {
        fde: FdeOptions {
            raim: raim_options(p_fa, weights, n_systems),
            max_iterations,
        },
        validation: SolutionValidationOptions {
            max_pdop,
            ..Default::default()
        },
    };

    match fde_spp(&sp3.inner, &inputs, with_geodetic, &options) {
        Ok(result) => Ok(PyFdeResult {
            solution: result.solution,
            excluded: result.excluded,
            iterations: result.iterations,
        }),
        Err(FdeError::Solve(FdeSppError::Spp(err))) => Err(SolveError::new_err(err.to_string())),
        Err(FdeError::Solve(FdeSppError::Validation(err))) => {
            Err(SolveError::new_err(err.to_string()))
        }
        Err(FdeError::FaultUnresolved(stat)) => Err(SolveError::new_err(format!(
            "FDE fault unresolved: test statistic {stat} still exceeds threshold after exhausting the exclusion budget"
        ))),
        Err(FdeError::Raim(err)) => Err(PyValueError::new_err(err.to_string())),
    }
}

#[pyfunction]
#[pyo3(signature = (sp3, config, robust, p_fa, max_iterations, weights=None, n_systems=None, max_pdop=None))]
#[allow(clippy::too_many_arguments)]
/// Run robust SPP with RAIM fault detection and exclusion.
///
/// This wraps the core robust SPP FDE driver and returns the final accepted result.
fn solve_spp_robust_fde(
    sp3: &PySp3,
    config: &PySppConfig,
    robust: &PySppRobustConfig,
    p_fa: f64,
    max_iterations: usize,
    weights: Option<BTreeMap<String, f64>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
) -> PyResult<PyFdeResult> {
    solve_spp_robust_fde_impl(
        sp3,
        config,
        robust,
        p_fa,
        max_iterations,
        weights,
        n_systems,
        max_pdop,
    )
}

#[pyfunction]
#[pyo3(signature = (sp3, config, robust, p_fa, max_iterations, weights=None, n_systems=None, max_pdop=None))]
#[allow(clippy::too_many_arguments)]
/// Deprecated alias for `solve_spp_robust_fde`.
fn spp_robust_fde_driver(
    py: Python<'_>,
    sp3: &PySp3,
    config: &PySppConfig,
    robust: &PySppRobustConfig,
    p_fa: f64,
    max_iterations: usize,
    weights: Option<BTreeMap<String, f64>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
) -> PyResult<PyFdeResult> {
    let warning = py.get_type::<PyDeprecationWarning>();
    PyErr::warn(
        py,
        &warning,
        c_str!("spp_robust_fde_driver is deprecated; use solve_spp_robust_fde"),
        2,
    )?;
    solve_spp_robust_fde_impl(
        sp3,
        config,
        robust,
        p_fa,
        max_iterations,
        weights,
        n_systems,
        max_pdop,
    )
}

#[allow(clippy::too_many_arguments)]
fn solve_spp_robust_fde_impl(
    sp3: &PySp3,
    config: &PySppConfig,
    robust: &PySppRobustConfig,
    p_fa: f64,
    max_iterations: usize,
    weights: Option<BTreeMap<String, f64>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
) -> PyResult<PyFdeResult> {
    let inputs = config.to_inputs();
    let with_geodetic = config.with_geodetic_flag();
    let options = FdeSppOptions {
        fde: FdeOptions {
            raim: raim_options(p_fa, weights, n_systems),
            max_iterations,
        },
        validation: SolutionValidationOptions {
            max_pdop,
            ..Default::default()
        },
    };

    match quality::spp_robust_fde_driver(
        &sp3.inner,
        &inputs,
        with_geodetic,
        robust.inner(),
        &options,
    ) {
        Ok(result) => Ok(PyFdeResult {
            solution: result.solution,
            excluded: result.excluded,
            iterations: result.iterations,
        }),
        Err(FdeError::Solve(FdeSppError::Spp(err))) => Err(SolveError::new_err(err.to_string())),
        Err(FdeError::Solve(FdeSppError::Validation(err))) => {
            Err(SolveError::new_err(err.to_string()))
        }
        Err(FdeError::FaultUnresolved(stat)) => Err(SolveError::new_err(format!(
            "FDE fault unresolved: test statistic {stat} still exceeds threshold after exhausting the exclusion budget"
        ))),
        Err(FdeError::Raim(err)) => Err(PyValueError::new_err(err.to_string())),
    }
}

// --- generic range RAIM/FDE design over a linearized measurement set -------

/// One linearized range measurement for [`qc_raim_fde_design`].
///
/// `design_row` is this measurement's row of the geometry matrix `H` (the
/// partials of the predicted range with respect to each estimated state
/// parameter); `residual_m` is the observed-minus-computed range; `weight` is the
/// inverse-variance weight `1 / sigma^2`. Every row must share the same
/// `design_row` length, which is the number of estimated state parameters.
#[pyclass(module = "sidereon._sidereon", name = "RangeFdeRow")]
#[derive(Clone)]
pub struct PyRangeFdeRow {
    inner: RangeFdeRow,
}

#[pymethods]
impl PyRangeFdeRow {
    /// Create one linearized range measurement row.
    #[new]
    fn new(id: String, residual_m: f64, design_row: Vec<f64>, weight: f64) -> Self {
        Self {
            inner: RangeFdeRow {
                id,
                residual_m,
                design_row,
                weight,
            },
        }
    }

    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    #[getter]
    fn residual_m(&self) -> f64 {
        self.inner.residual_m
    }

    #[getter]
    fn design_row(&self) -> Vec<f64> {
        self.inner.design_row.clone()
    }

    #[getter]
    fn weight(&self) -> f64 {
        self.inner.weight
    }

    fn __repr__(&self) -> String {
        format!(
            "RangeFdeRow(id={:?}, residual_m={:.6}, weight={:.6})",
            self.inner.id, self.inner.residual_m, self.inner.weight
        )
    }
}

/// Global chi-square consistency test over a protected measurement set.
#[pyclass(module = "sidereon._sidereon", name = "RangeChiSquareTest")]
pub struct PyRangeChiSquareTest {
    inner: RangeChiSquareTest,
}

#[pymethods]
impl PyRangeChiSquareTest {
    /// Weighted sum of squared post-fit residuals, `v^T W v`.
    #[getter]
    fn weighted_sum_squares(&self) -> f64 {
        self.inner.weighted_sum_squares
    }

    /// Redundancy, `n_used - n_state`.
    #[getter]
    fn dof(&self) -> isize {
        self.inner.dof
    }

    /// Chi-square threshold, `None` when `dof <= 0`.
    #[getter]
    fn threshold(&self) -> Option<f64> {
        self.inner.threshold
    }

    /// `False` when `dof <= 0` (no redundancy to test against).
    #[getter]
    fn testable(&self) -> bool {
        self.inner.testable
    }

    /// `True` when the test statistic exceeds the threshold (a fault remains).
    #[getter]
    fn fault_detected(&self) -> bool {
        self.inner.fault_detected
    }

    fn __repr__(&self) -> String {
        format!(
            "RangeChiSquareTest(weighted_sum_squares={:.6}, dof={}, fault_detected={})",
            self.inner.weighted_sum_squares, self.inner.dof, self.inner.fault_detected
        )
    }
}

/// Per-measurement diagnostics from [`qc_raim_fde_design`], in input order.
#[pyclass(module = "sidereon._sidereon", name = "RangeMeasurementDiagnostic")]
pub struct PyRangeMeasurementDiagnostic {
    inner: RangeMeasurementDiagnostic,
}

#[pymethods]
impl PyRangeMeasurementDiagnostic {
    /// Measurement identifier, echoed from the input row.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Whether the FDE loop excluded this measurement from the protected solve.
    #[getter]
    fn excluded(&self) -> bool {
        self.inner.excluded
    }

    /// Post-fit residual against the protected state correction, metres.
    #[getter]
    fn post_fit_residual_m(&self) -> f64 {
        self.inner.post_fit_residual_m
    }

    /// Standardized post-fit residual `post_fit_residual_m * sqrt(weight)`.
    #[getter]
    fn normalized_residual(&self) -> f64 {
        self.inner.normalized_residual
    }

    fn __repr__(&self) -> String {
        format!(
            "RangeMeasurementDiagnostic(id={:?}, excluded={}, normalized_residual={:.6})",
            self.inner.id, self.inner.excluded, self.inner.normalized_residual
        )
    }
}

/// Result of a standalone range RAIM/FDE design solve.
#[pyclass(module = "sidereon._sidereon", name = "RangeFdeResult")]
pub struct PyRangeFdeResult {
    inner: RangeFdeResult,
}

#[pymethods]
impl PyRangeFdeResult {
    /// Protected weighted-least-squares state correction `dx`, length `n_state`.
    #[getter]
    fn state_correction(&self) -> Vec<f64> {
        self.inner.state_correction.clone()
    }

    /// Protected state covariance `(H^T W H)^-1` for the accepted set, row-major.
    #[getter]
    fn state_covariance(&self) -> Vec<Vec<f64>> {
        self.inner.state_covariance.clone()
    }

    /// Global chi-square consistency test for the accepted set.
    #[getter]
    fn global_test(&self) -> PyRangeChiSquareTest {
        PyRangeChiSquareTest {
            inner: self.inner.global_test,
        }
    }

    /// Excluded measurement identifiers, in exclusion order.
    #[getter]
    fn excluded(&self) -> Vec<String> {
        self.inner.excluded.clone()
    }

    /// Per-measurement diagnostics, in input order.
    #[getter]
    fn diagnostics(&self) -> Vec<PyRangeMeasurementDiagnostic> {
        self.inner
            .diagnostics
            .iter()
            .map(|inner| PyRangeMeasurementDiagnostic {
                inner: inner.clone(),
            })
            .collect()
    }

    /// Number of exclusions performed.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "RangeFdeResult(n_state={}, excluded={:?}, iterations={})",
            self.inner.state_correction.len(),
            self.inner.excluded,
            self.inner.iterations
        )
    }
}

/// Standalone range RAIM/FDE over a generic linearized measurement set.
///
/// `rows` is a list of `RangeFdeRow` linearizing a range solve about a nominal
/// state. The protected weighted least squares `dx = (H^T W H)^-1 H^T W r` is
/// solved, the global chi-square consistency test run, and (on a detected fault)
/// the leave-one-out fault-detection-and-exclusion loop run. `p_fa` is the
/// false-alarm probability; `max_exclusions` caps the number of removals (`None`
/// for unbounded); `min_redundancy` is the redundancy floor an exclusion must
/// leave behind. Returns a `RangeFdeResult`. Raises `ValueError` on malformed or
/// rank-deficient input.
#[pyfunction]
#[pyo3(signature = (rows, p_fa=DEFAULT_P_FA, max_exclusions=None, min_redundancy=1))]
fn qc_raim_fde_design(
    py: Python<'_>,
    rows: Vec<Py<PyRangeFdeRow>>,
    p_fa: f64,
    max_exclusions: Option<usize>,
    min_redundancy: usize,
) -> PyResult<PyRangeFdeResult> {
    let rows: Vec<RangeFdeRow> = rows
        .iter()
        .map(|row| row.borrow(py).inner.clone())
        .collect();
    let options = RangeFdeOptions {
        p_fa,
        max_exclusions: max_exclusions.unwrap_or(usize::MAX),
        min_redundancy,
    };
    let inner =
        core_raim_fde_design(&rows, &options).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyRangeFdeResult { inner })
}

#[pyfunction]
fn chi2_inv(p: f64, dof: usize) -> PyResult<f64> {
    quality::chi2_inv(p, dof).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Source of the interval used by observation QC gap detection.
#[pyclass(module = "sidereon._sidereon", name = "IntervalSource", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyIntervalSource {
    OVERRIDE,
    HEADER,
    INFERRED,
    UNRESOLVED,
}

impl From<IntervalSource> for PyIntervalSource {
    fn from(value: IntervalSource) -> Self {
        match value {
            IntervalSource::Override => Self::OVERRIDE,
            IntervalSource::Header => Self::HEADER,
            IntervalSource::Inferred => Self::INFERRED,
            IntervalSource::Unresolved => Self::UNRESOLVED,
        }
    }
}

#[pymethods]
impl PyIntervalSource {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::OVERRIDE => "override",
            Self::HEADER => "header",
            Self::INFERRED => "inferred",
            Self::UNRESOLVED => "unresolved",
        }
    }
}

/// RINEX SSI digit histogram.
#[pyclass(module = "sidereon._sidereon", name = "SsiHistogram")]
#[derive(Clone, Copy)]
pub struct PySsiHistogram {
    inner: SsiHistogram,
}

impl From<SsiHistogram> for PySsiHistogram {
    fn from(inner: SsiHistogram) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySsiHistogram {
    #[getter]
    fn counts(&self) -> Vec<u64> {
        self.inner.counts.to_vec()
    }
}

/// Raw S-code signal-strength statistics.
#[pyclass(module = "sidereon._sidereon", name = "SnrStats")]
#[derive(Clone, Copy)]
pub struct PySnrStats {
    inner: SnrStats,
}

impl From<SnrStats> for PySnrStats {
    fn from(inner: SnrStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySnrStats {
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    #[getter]
    fn mean(&self) -> f64 {
        self.inner.mean
    }

    #[getter]
    fn min(&self) -> f64 {
        self.inner.min
    }

    #[getter]
    fn max(&self) -> f64 {
        self.inner.max
    }

    #[getter]
    fn std(&self) -> Option<f64> {
        self.inner.std
    }
}

/// One detected observation data gap.
#[pyclass(module = "sidereon._sidereon", name = "ObservationDataGap")]
#[derive(Clone)]
pub struct PyObservationDataGap {
    inner: ObservationDataGap,
}

impl From<ObservationDataGap> for PyObservationDataGap {
    fn from(inner: ObservationDataGap) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObservationDataGap {
    #[getter]
    fn start_epoch(&self) -> PyObsEpochTime {
        self.inner.start_epoch.into()
    }

    #[getter]
    fn end_epoch(&self) -> PyObsEpochTime {
        self.inner.end_epoch.into()
    }

    #[getter]
    fn nominal_interval_s(&self) -> f64 {
        self.inner.nominal_interval_s
    }

    #[getter]
    fn observed_delta_s(&self) -> f64 {
        self.inner.observed_delta_s
    }

    #[getter]
    fn missing_epochs(&self) -> usize {
        self.inner.missing_epochs
    }
}

/// Per-satellite observation QC counts.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteObservationQc")]
#[derive(Clone)]
pub struct PySatelliteObservationQc {
    inner: SatelliteObservationQc,
}

impl From<SatelliteObservationQc> for PySatelliteObservationQc {
    fn from(inner: SatelliteObservationQc) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySatelliteObservationQc {
    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    #[getter]
    fn epochs_with_observations(&self) -> usize {
        self.inner.epochs_with_observations
    }

    #[getter]
    fn value_observations(&self) -> usize {
        self.inner.value_observations
    }
}

/// Per-satellite, per-code observation QC counts.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteSignalQc")]
#[derive(Clone)]
pub struct PySatelliteSignalQc {
    inner: SatelliteSignalQc,
}

impl From<SatelliteSignalQc> for PySatelliteSignalQc {
    fn from(inner: SatelliteSignalQc) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySatelliteSignalQc {
    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    #[getter]
    fn code(&self) -> &str {
        &self.inner.code
    }

    #[getter]
    fn value_observations(&self) -> usize {
        self.inner.value_observations
    }

    #[getter]
    fn ssi(&self) -> Option<PySsiHistogram> {
        self.inner.ssi.map(Into::into)
    }

    #[getter]
    fn snr(&self) -> Option<PySnrStats> {
        self.inner.snr.map(Into::into)
    }
}

/// Per-system, per-code observation QC counts.
#[pyclass(module = "sidereon._sidereon", name = "SystemSignalQc")]
#[derive(Clone)]
pub struct PySystemSignalQc {
    inner: SystemSignalQc,
}

impl From<SystemSignalQc> for PySystemSignalQc {
    fn from(inner: SystemSignalQc) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySystemSignalQc {
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    #[getter]
    fn code(&self) -> &str {
        &self.inner.code
    }

    #[getter]
    fn value_observations(&self) -> usize {
        self.inner.value_observations
    }

    #[getter]
    fn ssi(&self) -> Option<PySsiHistogram> {
        self.inner.ssi.map(Into::into)
    }

    #[getter]
    fn snr(&self) -> Option<PySnrStats> {
        self.inner.snr.map(Into::into)
    }
}

/// Non-fatal observation QC note.
#[pyclass(module = "sidereon._sidereon", name = "ObservationQcNote")]
#[derive(Clone, Copy)]
pub struct PyObservationQcNote {
    inner: ObservationQcNote,
}

impl From<ObservationQcNote> for PyObservationQcNote {
    fn from(inner: ObservationQcNote) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObservationQcNote {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            ObservationQcNote::NonMonotonicEpoch { .. } => "non_monotonic_epoch",
            ObservationQcNote::IntervalUnresolved => "interval_unresolved",
        }
    }

    #[getter]
    fn epoch_index(&self) -> Option<usize> {
        match self.inner {
            ObservationQcNote::NonMonotonicEpoch { epoch_index } => Some(epoch_index),
            ObservationQcNote::IntervalUnresolved => None,
        }
    }
}

/// Aggregate RINEX observation QC report.
#[pyclass(module = "sidereon._sidereon", name = "ObservationQcReport")]
#[derive(Clone)]
pub struct PyObservationQcReport {
    inner: ObservationQcReport,
}

impl From<ObservationQcReport> for PyObservationQcReport {
    fn from(inner: ObservationQcReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObservationQcReport {
    #[getter]
    fn total_epoch_records(&self) -> usize {
        self.inner.total_epoch_records
    }

    #[getter]
    fn observation_epochs(&self) -> usize {
        self.inner.observation_epochs
    }

    #[getter]
    fn event_records(&self) -> usize {
        self.inner.event_records
    }

    #[getter]
    fn power_failure_epochs(&self) -> usize {
        self.inner.power_failure_epochs
    }

    #[getter]
    fn skipped_records(&self) -> usize {
        self.inner.skipped_records
    }

    #[getter]
    fn interval_s(&self) -> Option<f64> {
        self.inner.interval_s
    }

    #[getter]
    fn interval_source(&self) -> PyIntervalSource {
        self.inner.interval_source.into()
    }

    #[getter]
    fn missing_epochs(&self) -> usize {
        self.inner.missing_epochs
    }

    #[getter]
    fn data_gaps(&self) -> Vec<PyObservationDataGap> {
        self.inner
            .data_gaps
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn satellites(&self) -> Vec<PySatelliteObservationQc> {
        self.inner
            .satellites
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn satellite_signals(&self) -> Vec<PySatelliteSignalQc> {
        self.inner
            .satellite_signals
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn system_signals(&self) -> Vec<PySystemSignalQc> {
        self.inner
            .system_signals
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn notes(&self) -> Vec<PyObservationQcNote> {
        self.inner.notes.iter().copied().map(Into::into).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "ObservationQcReport(observation_epochs={}, satellites={})",
            self.inner.observation_epochs,
            self.inner.satellites.len()
        )
    }
}

/// Run RINEX observation quality-control rollups.
#[pyfunction]
#[pyo3(signature = (obs, interval_override_s=None, gap_factor=1.5))]
fn observation_qc(
    obs: &PyRinexObs,
    interval_override_s: Option<f64>,
    gap_factor: f64,
) -> PyResult<PyObservationQcReport> {
    let report = core_observation_qc_with_options(
        obs.inner(),
        ObservationQcOptions {
            interval_override_s,
            gap_factor,
        },
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(report.into())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRaimResult>()?;
    m.add_class::<PyFdeResult>()?;
    m.add_class::<PyRangeFdeRow>()?;
    m.add_class::<PyRangeChiSquareTest>()?;
    m.add_class::<PyRangeMeasurementDiagnostic>()?;
    m.add_class::<PyRangeFdeResult>()?;
    m.add_class::<PyIntervalSource>()?;
    m.add_class::<PySsiHistogram>()?;
    m.add_class::<PySnrStats>()?;
    m.add_class::<PyObservationDataGap>()?;
    m.add_class::<PySatelliteObservationQc>()?;
    m.add_class::<PySatelliteSignalQc>()?;
    m.add_class::<PySystemSignalQc>()?;
    m.add_class::<PyObservationQcNote>()?;
    m.add_class::<PyObservationQcReport>()?;
    m.add_function(wrap_pyfunction!(qc_raim, m)?)?;
    m.add_function(wrap_pyfunction!(qc_fde, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp_robust_fde, m)?)?;
    m.add_function(wrap_pyfunction!(spp_robust_fde_driver, m)?)?;
    m.add_function(wrap_pyfunction!(qc_raim_fde_design, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_inv, m)?)?;
    m.add_function(wrap_pyfunction!(observation_qc, m)?)?;
    Ok(())
}
