//! Quality-control binding: residual chi-square RAIM and fault detection and
//! exclusion (FDE).
//!
//! Thin marshaling over [`sidereon_core::quality`]: [`qc_raim`] runs the
//! residual-based chi-square integrity test over a solution's used satellites and
//! residuals; [`qc_fde`] delegates to the core [`fde_spp`] driver, which tests the
//! solve against its own pseudorange variances and, on a detected fault, excludes
//! by RTKLIB demo5 `raim_fde`'s leave-one-out rule until RAIM passes (or the
//! exclusion budget is spent). No statistics, solve, or exclusion loop lives
//! here; the numbers are exactly what `sidereon-core` produces.

use std::collections::BTreeMap;

use numpy::PyArray1;
use pyo3::exceptions::{PyDeprecationWarning, PyValueError};
use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::positioning::{residual_rms as core_residual_rms, ReceiverSolution};
use sidereon_core::qc_obs::{
    observation_qc_with_options as core_observation_qc_with_options,
    render_html as core_observation_qc_render_html, render_text as core_observation_qc_render_text,
    ClockJump, CycleSlipQc, IntervalSource, MpStats, MultipathReport, ObservationDataGap,
    ObservationQcFinding, ObservationQcNote, ObservationQcOptions, ObservationQcReport,
    SatelliteMultipathQc, SatelliteObservationQc, SatelliteSignalQc, SnrStats, SsiHistogram,
    SystemCycleSlipQc, SystemMultipathQc, SystemSignalQc,
};
use sidereon_core::quality::{
    self, fde_spp, raim_fde_design as core_raim_fde_design, FdeError, FdeOptions, FdeResult,
    FdeSppError, FdeSppOptions, FdeUnresolved, FdeUnresolvedReason, RaimInput, RaimOptions,
    RaimWeights, RangeChiSquareTest, RangeFdeOptions, RangeFdeResult, RangeFdeRow,
    RangeMeasurementDiagnostic, SolutionValidationOptions, DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
    DEFAULT_P_FA,
};

use crate::marshal::PyGnssSystem;
use crate::observables::PyRaimWeights;
use crate::rinex::{PyBroadcastEphemeris, PyObsEpochTime, PyRinexLintSeverity, PyRinexObs};
use crate::spp::{PySppConfig, PySppRobustConfig, PySppSolution};
use crate::{np_array, quality_err, PySp3, SolveError};

/// The RAIM weighting a Python `weights` argument names.
///
/// `None` selects the core default, `RaimWeights::Solution`: each residual is
/// divided by the standard deviation the estimator weighted it by, as RTKLIB
/// demo5 `valsol` forms `sum (v / sigma)^2`. A `RaimWeights` instance is used
/// as it is; the labels `"solution"` and `"unit"` name those modes; a dict maps
/// satellite tokens to inverse-variance weights (`RaimWeights.by_satellite`),
/// with omitted satellites at unit weight.
fn raim_weights(weights: Option<&Bound<'_, PyAny>>) -> PyResult<RaimWeights> {
    let Some(weights) = weights.filter(|weights| !weights.is_none()) else {
        return Ok(RaimWeights::default());
    };
    if let Ok(typed) = weights.extract::<PyRef<'_, PyRaimWeights>>() {
        return Ok(typed.inner().clone());
    }
    if let Ok(label) = weights.extract::<String>() {
        return match label.as_str() {
            "solution" => Ok(RaimWeights::Solution),
            "unit" => Ok(RaimWeights::Unit),
            other => Err(PyValueError::new_err(format!(
                "unknown RAIM weighting {other:?}; expected \"solution\", \"unit\", a RaimWeights, or a dict of per-satellite inverse-variance weights"
            ))),
        };
    }
    Ok(RaimWeights::BySatellite(
        weights.extract::<BTreeMap<String, f64>>()?,
    ))
}

fn raim_options(
    p_fa: f64,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
) -> PyResult<RaimOptions> {
    let mut options = RaimOptions::default();
    options.p_fa = p_fa;
    options.weights = raim_weights(weights)?;
    options.n_systems = n_systems;
    Ok(options)
}

/// Typed input for standalone residual chi-square RAIM.
#[pyclass(module = "sidereon._sidereon", name = "RaimInput")]
#[derive(Clone)]
pub struct PyRaimInput {
    inner: RaimInput,
}

#[pymethods]
impl PyRaimInput {
    /// Create a standalone RAIM input from satellite tokens, post-fit
    /// residuals in metres and, optionally, the variance in square metres the
    /// estimator weighted each residual by, in residual order. The default
    /// solution weighting reads the variances and refuses an input without
    /// them.
    #[new]
    #[pyo3(signature = (used_sats, residuals_m, variances_m2=None))]
    fn new(used_sats: Vec<String>, residuals_m: Vec<f64>, variances_m2: Option<Vec<f64>>) -> Self {
        Self {
            inner: RaimInput {
                used_sats,
                residuals_m,
                variances_m2,
            },
        }
    }

    /// Satellite tokens in residual order.
    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner.used_sats.clone()
    }

    /// Post-fit residuals in metres.
    #[getter]
    fn residuals_m(&self) -> Vec<f64> {
        self.inner.residuals_m.clone()
    }

    /// Residual variances in square metres, in residual order, or `None`.
    #[getter]
    fn variances_m2(&self) -> Option<Vec<f64>> {
        self.inner.variances_m2.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RaimInput(used_sats={}, variances={})",
            self.inner.used_sats.len(),
            self.inner.variances_m2.is_some()
        )
    }
}

/// The result of a residual chi-square RAIM test.
#[pyclass(module = "sidereon._sidereon", name = "RaimResult")]
pub struct PyRaimResult {
    inner: quality::RaimResult,
    rms_m: f64,
    reduced_chi_square: Option<f64>,
}

#[pymethods]
impl PyRaimResult {
    /// `True` when the test statistic exceeds the chi-square threshold.
    #[getter]
    fn fault_detected(&self) -> bool {
        self.inner.fault_detected
    }

    /// Weighted residual sum of squares: `sum (r / sigma)^2` under the
    /// solution weighting, `sum w r^2` under unit or per-satellite weights.
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

    /// Per-satellite weighted residuals as a dict of `token -> value`: `r /
    /// sigma` under the solution weighting, `r sqrt(w)` otherwise. They are not
    /// standardized by each residual's redundancy, so a satellite the geometry
    /// leans on shows a small value here even when it carries the fault.
    #[getter]
    fn normalized_residuals(&self) -> BTreeMap<String, f64> {
        self.inner.normalized_residuals.clone()
    }

    /// Satellite token with the largest absolute weighted residual. This is not
    /// a fault identification: a fault on a low-redundancy satellite spreads
    /// onto the other residuals and can leave a healthy satellite here, which
    /// is why FDE chooses exclusions by re-solving without each candidate.
    #[getter]
    fn worst_sat(&self) -> Option<String> {
        self.inner.worst_sat.clone()
    }

    /// Root-mean-square residual in metres, unweighted.
    #[getter]
    fn rms_m(&self) -> f64 {
        self.rms_m
    }

    /// Reduced chi-square, `test_statistic / dof`, when `dof > 0`.
    #[getter]
    fn reduced_chi_square(&self) -> Option<f64> {
        self.reduced_chi_square
    }

    fn __repr__(&self) -> String {
        format!(
            "RaimResult(fault_detected={}, test_statistic={:.6}, dof={})",
            self.inner.fault_detected, self.inner.test_statistic, self.inner.dof
        )
    }
}

fn py_raim_result(input: RaimInput, options: RaimOptions) -> PyResult<PyRaimResult> {
    let inner = quality::raim(&input, &options).map_err(quality_err)?;
    Ok(py_raim_result_from(&input.residuals_m, inner))
}

/// Wrap a core RAIM result with the unweighted residual RMS and the reduced
/// chi-square `test_statistic / dof` it documents.
fn py_raim_result_from(residuals_m: &[f64], inner: quality::RaimResult) -> PyRaimResult {
    let reduced_chi_square = (inner.dof > 0).then(|| inner.test_statistic / inner.dof as f64);
    PyRaimResult {
        inner,
        rms_m: core_residual_rms(residuals_m),
        reduced_chi_square,
    }
}

/// Direct post-solve residual chi-square RAIM.
///
/// `used_sats` are satellite tokens in residual order (or a `RaimInput`, with
/// `residuals_m` and `variances_m2` omitted); `residuals_m` are the post-fit
/// pseudorange residuals in metres; `variances_m2` are the variances in square
/// metres the estimator weighted each residual by. `weights` selects the
/// weighting: `None` (the core default) reads `variances_m2` and forms RTKLIB
/// demo5 `valsol`'s `sum (r / sigma)^2`, refusing residuals without variances
/// with `ValueError`; `"unit"` treats every sigma as 1 m; a dict (or
/// `RaimWeights.by_satellite`) gives per-satellite inverse-variance weights,
/// with omitted satellites at unit weight. `n_systems` overrides the number of
/// receiver clock parameters, otherwise the distinct system letters.
#[pyfunction]
#[pyo3(signature = (used_sats, residuals_m=None, p_fa=DEFAULT_P_FA, weights=None, n_systems=None, variances_m2=None))]
fn raim(
    used_sats: &Bound<'_, PyAny>,
    residuals_m: Option<Vec<f64>>,
    p_fa: f64,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    variances_m2: Option<Vec<f64>>,
) -> PyResult<PyRaimResult> {
    let input = if let Ok(typed) = used_sats.extract::<PyRef<'_, PyRaimInput>>() {
        if residuals_m.is_some() || variances_m2.is_some() {
            return Err(PyValueError::new_err(
                "residuals_m and variances_m2 must be omitted when used_sats is RaimInput",
            ));
        }
        typed.inner.clone()
    } else {
        let used_sats = used_sats.extract::<Vec<String>>()?;
        let residuals_m = residuals_m.ok_or_else(|| {
            PyValueError::new_err("residuals_m is required when input is a satellite sequence")
        })?;
        RaimInput {
            used_sats,
            residuals_m,
            variances_m2,
        }
    };
    let options = raim_options(p_fa, weights, n_systems)?;
    py_raim_result(input, options)
}

/// Run residual chi-square RAIM over an existing SPP solution object.
///
/// The residuals, their variances (`SppSolution.pseudorange_variances_m2`) and
/// the solve's clock count come from the solution; `n_systems` overrides the
/// clock count. `weights` defaults to the solution weighting, the chi-square
/// of the residuals over the variances the solve weighted them by.
#[pyfunction]
#[pyo3(signature = (solution, p_fa=DEFAULT_P_FA, weights=None, n_systems=None))]
fn raim_for_solution(
    solution: &PySppSolution,
    p_fa: f64,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
) -> PyResult<PyRaimResult> {
    let options = raim_options(p_fa, weights, n_systems)?;
    let inner = quality::raim_for_solution(solution.inner(), &options).map_err(quality_err)?;
    Ok(py_raim_result_from(&solution.inner().residuals_m, inner))
}

/// Residual-based chi-square RAIM over a solution's used satellites and residuals.
///
/// `used_sats` are the satellite tokens in residual order; `residuals_m` are the
/// post-fit pseudorange residuals (metres). `p_fa` is the false-alarm
/// probability. `weights` and `variances_m2` are as for `raim`: the default
/// solution weighting reads `variances_m2`, the variances (square metres) the
/// estimator weighted each residual by, and refuses residuals without them;
/// `"unit"` or a dict of per-satellite inverse-variance weights select the
/// other modes. `n_systems` optionally overrides the number of receiver clock
/// parameters. Returns a `RaimResult`. Raises `ValueError` on malformed input.
#[pyfunction]
#[pyo3(signature = (used_sats, residuals_m, p_fa=DEFAULT_P_FA, weights=None, n_systems=None, variances_m2=None))]
fn qc_raim(
    used_sats: Vec<String>,
    residuals_m: Vec<f64>,
    p_fa: f64,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    variances_m2: Option<Vec<f64>>,
) -> PyResult<PyRaimResult> {
    let input = RaimInput {
        used_sats,
        residuals_m,
        variances_m2,
    };
    let options = raim_options(p_fa, weights, n_systems)?;
    py_raim_result(input, options)
}

/// The result of a fault-detection-and-exclusion loop.
#[pyclass(module = "sidereon._sidereon", name = "FdeResult")]
pub struct PyFdeResult {
    solution: ReceiverSolution,
    excluded: Vec<String>,
    iterations: usize,
    raim: quality::RaimResult,
}

impl From<FdeResult<ReceiverSolution>> for PyFdeResult {
    fn from(result: FdeResult<ReceiverSolution>) -> Self {
        Self {
            solution: result.solution,
            excluded: result.excluded,
            iterations: result.iterations,
            raim: result.raim,
        }
    }
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

    /// The accepted solution as an `SppSolution`, with every field the solve
    /// reports.
    #[getter]
    fn solution(&self) -> PySppSolution {
        PySppSolution::from_solution(self.solution.clone())
    }

    /// Excluded satellite tokens, in exclusion order.
    #[getter]
    fn excluded(&self) -> Vec<String> {
        self.excluded.clone()
    }

    /// Number of exclusions performed, `len(excluded)`.
    #[getter]
    fn iterations(&self) -> usize {
        self.iterations
    }

    /// The detection test of the accepted solution. `testable` is `False`
    /// when the accepted set has no redundancy left to test.
    #[getter]
    fn raim(&self) -> PyRaimResult {
        py_raim_result_from(&self.solution.residuals_m, self.raim.clone())
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

/// The label of why FDE stopped with a fault still detected:
/// `exclusion_budget_exhausted` or `no_admissible_exclusion`.
fn unresolved_reason_label(reason: FdeUnresolvedReason) -> String {
    match reason {
        FdeUnresolvedReason::ExclusionBudgetExhausted => "exclusion_budget_exhausted".to_string(),
        FdeUnresolvedReason::NoAdmissibleExclusion => "no_admissible_exclusion".to_string(),
        // The core enum is non-exhaustive; a variant a later core adds keeps
        // its own name rather than collapsing into a catch-all label.
        other => snake_case(&format!("{other:?}")),
    }
}

fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (index, c) in name.chars().enumerate() {
        if c.is_ascii_uppercase() {
            if index > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Map an unresolved fault into `FdeFaultUnresolvedError`, carrying the core
/// reason, the last solution, the exclusions made and its detection test.
fn fde_fault_unresolved_err(py: Python<'_>, unresolved: FdeUnresolved<ReceiverSolution>) -> PyErr {
    let reason = unresolved_reason_label(unresolved.reason);
    let threshold = unresolved
        .raim
        .threshold
        .map_or_else(|| "none".to_string(), |threshold| threshold.to_string());
    let message = format!(
        "FDE fault unresolved ({reason}): test statistic {} exceeds threshold {threshold} after excluding {:?}",
        unresolved.raim.test_statistic, unresolved.excluded
    );
    let ty = match crate::fde_fault_unresolved_error_type(py) {
        Ok(ty) => ty,
        Err(e) => return e,
    };
    let err = PyErr::from_type(ty, message);
    let raim = py_raim_result_from(&unresolved.solution.residuals_m, unresolved.raim);
    let solution = PySppSolution::from_solution(unresolved.solution);
    match set_fde_unresolved_fields(py, &err, reason, solution, unresolved.excluded, raim) {
        Ok(()) => err,
        Err(e) => e,
    }
}

fn set_fde_unresolved_fields(
    py: Python<'_>,
    err: &PyErr,
    reason: String,
    solution: PySppSolution,
    excluded: Vec<String>,
    raim: PyRaimResult,
) -> PyResult<()> {
    let value = err.value(py);
    value.setattr("reason", reason)?;
    value.setattr("solution", Py::new(py, solution)?)?;
    value.setattr("excluded", excluded)?;
    value.setattr("raim", Py::new(py, raim)?)?;
    Ok(())
}

/// Map a core FDE outcome: the accepted `FdeResult`, `SelectionUnsettledError`
/// or `SolveError` for a solve that failed on the full set and that no
/// leave-one-out re-solve cured, `FdeFaultUnresolvedError` for a fault still
/// detected when the loop stopped, and `ValueError` for invalid RAIM options.
fn fde_outcome(
    py: Python<'_>,
    outcome: Result<FdeResult<ReceiverSolution>, FdeError<ReceiverSolution, FdeSppError>>,
) -> PyResult<PyFdeResult> {
    match outcome {
        Ok(result) => Ok(result.into()),
        Err(FdeError::Solve(FdeSppError::Spp(err))) => Err(crate::spp_solve_err(err)),
        Err(FdeError::Solve(FdeSppError::Validation(err))) => {
            Err(SolveError::new_err(err.to_string()))
        }
        Err(FdeError::FaultUnresolved(unresolved)) => {
            Err(fde_fault_unresolved_err(py, *unresolved))
        }
        Err(FdeError::Raim(err)) => Err(quality_err(err)),
    }
}

/// The exclusion budget: `max_exclusions`, or the core default, RTKLIB demo5's
/// single exclusion, when it is not given.
fn exclusion_budget(max_exclusions: Option<usize>) -> usize {
    max_exclusions.unwrap_or_else(|| FdeOptions::default().max_exclusions)
}

/// The core FDE options for one binding call.
#[allow(clippy::too_many_arguments)]
fn fde_spp_options(
    p_fa: f64,
    max_exclusions: Option<usize>,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
    max_exclusion_rms_m: f64,
) -> PyResult<FdeSppOptions> {
    let mut validation = SolutionValidationOptions::default();
    validation.max_pdop = max_pdop;
    let mut fde = FdeOptions::new(
        raim_options(p_fa, weights, n_systems)?,
        exclusion_budget(max_exclusions),
    );
    fde.max_exclusion_rms_m = max_exclusion_rms_m;
    Ok(FdeSppOptions::new(fde, validation))
}

/// Fault detection and exclusion over an SPP solve.
///
/// `config` is the `SppConfig` to solve; its observation set is the starting
/// set. The solve is tested with residual chi-square RAIM at `p_fa` (default
/// `1e-3`, RTKLIB demo5's `chisqr` alpha): by default the residuals over the
/// variances the solve weighted them by (`SppSolution.pseudorange_variances_m2`),
/// as RTKLIB `valsol` forms `sum (v / sigma)^2`; `weights` (`"unit"`, a dict of
/// per-satellite inverse-variance weights, or a `RaimWeights`) and `n_systems`
/// override that. On a detected fault each satellite of the flagged solution is
/// left out in turn and the rest re-solved, as RTKLIB demo5 `raim_fde` does, and
/// the re-solve with the smallest unweighted residual RMS, no larger than
/// `max_exclusion_rms_m` (default 100 m, demo5's initial `rms`) and using at
/// least five satellites, is kept. `max_exclusions` (default 1, demo5's single
/// exclusion) bounds the exclusions; a larger budget repeats the
/// test and the search on the remaining set. A full-set solve that fails as
/// RTKLIB `estpos` can fail (no settling, singular geometry, a refused
/// geometry) with at least six observations is searched the same way.
/// `max_pdop` optionally caps the accepted geometry.
///
/// Returns an `FdeResult`. Raises `FdeFaultUnresolvedError` (a `SolveError`)
/// when a fault is still detected when the loop stops, carrying the last
/// solution, the exclusions and its `RaimResult`; `SolveError` (or
/// `SelectionUnsettledError`) when the solve fails and no exclusion cures it;
/// and `ValueError` on malformed input or options.
#[pyfunction]
#[pyo3(signature = (
    sp3,
    config,
    p_fa=DEFAULT_P_FA,
    max_exclusions=None,
    weights=None,
    n_systems=None,
    max_pdop=None,
    *,
    max_exclusion_rms_m=DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
))]
#[allow(clippy::too_many_arguments)]
fn qc_fde(
    py: Python<'_>,
    sp3: &PySp3,
    config: &PySppConfig,
    p_fa: f64,
    max_exclusions: Option<usize>,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
    max_exclusion_rms_m: f64,
) -> PyResult<PyFdeResult> {
    let options = fde_spp_options(
        p_fa,
        max_exclusions,
        weights,
        n_systems,
        max_pdop,
        max_exclusion_rms_m,
    )?;
    fde_outcome(
        py,
        fde_spp(
            &sp3.inner,
            &config.to_inputs(),
            config.with_geodetic_flag(),
            &options,
        ),
    )
}

/// Fault detection and exclusion over an SPP solve using broadcast ephemeris;
/// the arguments, result and errors are those of `qc_fde`.
#[pyfunction]
#[pyo3(signature = (
    broadcast,
    config,
    p_fa=DEFAULT_P_FA,
    max_exclusions=None,
    weights=None,
    n_systems=None,
    max_pdop=None,
    *,
    max_exclusion_rms_m=DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
))]
#[allow(clippy::too_many_arguments)]
fn qc_fde_broadcast(
    py: Python<'_>,
    broadcast: &PyBroadcastEphemeris,
    config: &PySppConfig,
    p_fa: f64,
    max_exclusions: Option<usize>,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
    max_exclusion_rms_m: f64,
) -> PyResult<PyFdeResult> {
    let options = fde_spp_options(
        p_fa,
        max_exclusions,
        weights,
        n_systems,
        max_pdop,
        max_exclusion_rms_m,
    )?;
    fde_outcome(
        py,
        fde_spp(
            &broadcast.inner,
            &config.to_inputs(),
            config.with_geodetic_flag(),
            &options,
        ),
    )
}

/// Alias for `qc_fde_broadcast`.
#[pyfunction]
#[pyo3(signature = (
    broadcast,
    config,
    p_fa=DEFAULT_P_FA,
    max_exclusions=None,
    weights=None,
    n_systems=None,
    max_pdop=None,
    *,
    max_exclusion_rms_m=DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
))]
#[allow(clippy::too_many_arguments)]
fn fde_broadcast(
    py: Python<'_>,
    broadcast: &PyBroadcastEphemeris,
    config: &PySppConfig,
    p_fa: f64,
    max_exclusions: Option<usize>,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
    max_exclusion_rms_m: f64,
) -> PyResult<PyFdeResult> {
    qc_fde_broadcast(
        py,
        broadcast,
        config,
        p_fa,
        max_exclusions,
        weights,
        n_systems,
        max_pdop,
        max_exclusion_rms_m,
    )
}

#[pyfunction]
#[pyo3(signature = (
    sp3,
    config,
    robust,
    p_fa=DEFAULT_P_FA,
    max_exclusions=None,
    weights=None,
    n_systems=None,
    max_pdop=None,
    *,
    max_exclusion_rms_m=DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
))]
#[allow(clippy::too_many_arguments)]
/// Run robust SPP with RAIM fault detection and exclusion.
///
/// This wraps the core robust SPP FDE driver: every solve, the first and each
/// leave-one-out re-solve, is the Huber-reweighted solve `robust` configures.
/// Detection standardizes the residuals by the pseudorange variances, not by
/// the Huber-reduced weights. The other arguments, the result and the errors
/// are those of `qc_fde`.
fn solve_spp_robust_fde(
    py: Python<'_>,
    sp3: &PySp3,
    config: &PySppConfig,
    robust: &PySppRobustConfig,
    p_fa: f64,
    max_exclusions: Option<usize>,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
    max_exclusion_rms_m: f64,
) -> PyResult<PyFdeResult> {
    solve_spp_robust_fde_impl(
        py,
        sp3,
        config,
        robust,
        fde_spp_options(
            p_fa,
            max_exclusions,
            weights,
            n_systems,
            max_pdop,
            max_exclusion_rms_m,
        )?,
    )
}

#[pyfunction]
#[pyo3(signature = (
    sp3,
    config,
    robust,
    p_fa=DEFAULT_P_FA,
    max_exclusions=None,
    weights=None,
    n_systems=None,
    max_pdop=None,
    *,
    max_exclusion_rms_m=DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
))]
#[allow(clippy::too_many_arguments)]
/// Deprecated alias for `solve_spp_robust_fde`.
fn spp_robust_fde_driver(
    py: Python<'_>,
    sp3: &PySp3,
    config: &PySppConfig,
    robust: &PySppRobustConfig,
    p_fa: f64,
    max_exclusions: Option<usize>,
    weights: Option<&Bound<'_, PyAny>>,
    n_systems: Option<isize>,
    max_pdop: Option<f64>,
    max_exclusion_rms_m: f64,
) -> PyResult<PyFdeResult> {
    let warning = py.get_type::<PyDeprecationWarning>();
    PyErr::warn(
        py,
        &warning,
        c_str!("spp_robust_fde_driver is deprecated; use solve_spp_robust_fde"),
        2,
    )?;
    solve_spp_robust_fde_impl(
        py,
        sp3,
        config,
        robust,
        fde_spp_options(
            p_fa,
            max_exclusions,
            weights,
            n_systems,
            max_pdop,
            max_exclusion_rms_m,
        )?,
    )
}

fn solve_spp_robust_fde_impl(
    py: Python<'_>,
    sp3: &PySp3,
    config: &PySppConfig,
    robust: &PySppRobustConfig,
    options: FdeSppOptions,
) -> PyResult<PyFdeResult> {
    fde_outcome(
        py,
        quality::spp_robust_fde_driver(
            &sp3.inner,
            &config.to_inputs(),
            config.with_geodetic_flag(),
            robust.inner(),
            &options,
        ),
    )
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
/// RTKLIB demo5 `raim_fde`'s exclusion run: each active row is left out in turn,
/// in input order, the rest re-solved, and the one whose unweighted post-fit
/// residual RMS is smallest (and no larger than `max_exclusion_rms_m`, default
/// 100 m) is excluded, ties to the later row. `p_fa` is the false-alarm
/// probability; `max_exclusions` caps the number of removals (default 1,
/// RTKLIB's single exclusion; `None` for unbounded); `min_redundancy` is the
/// redundancy floor an exclusion must leave behind. Returns a `RangeFdeResult`.
/// Raises `ValueError` on malformed or rank-deficient input.
#[pyfunction]
#[pyo3(signature = (
    rows,
    p_fa=DEFAULT_P_FA,
    max_exclusions=Some(RangeFdeOptions::default().max_exclusions),
    min_redundancy=RangeFdeOptions::default().min_redundancy,
    max_exclusion_rms_m=DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
))]
fn qc_raim_fde_design(
    py: Python<'_>,
    rows: Vec<Py<PyRangeFdeRow>>,
    p_fa: f64,
    max_exclusions: Option<usize>,
    min_redundancy: usize,
    max_exclusion_rms_m: f64,
) -> PyResult<PyRangeFdeResult> {
    let rows: Vec<RangeFdeRow> = rows
        .iter()
        .map(|row| row.borrow(py).inner.clone())
        .collect();
    let mut options = RangeFdeOptions::default();
    options.p_fa = p_fa;
    options.max_exclusions = max_exclusions.unwrap_or(usize::MAX);
    options.min_redundancy = min_redundancy;
    options.max_exclusion_rms_m = max_exclusion_rms_m;
    let inner = core_raim_fde_design(&rows, &options).map_err(quality_err)?;
    Ok(PyRangeFdeResult { inner })
}

#[pyfunction]
fn chi2_inv(p: f64, dof: usize) -> PyResult<f64> {
    quality::chi2_inv(p, dof).map_err(quality_err)
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

/// One detected receiver-clock jump.
#[pyclass(module = "sidereon._sidereon", name = "ClockJump")]
#[derive(Clone, Copy)]
pub struct PyClockJump {
    inner: ClockJump,
}

impl From<ClockJump> for PyClockJump {
    fn from(inner: ClockJump) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyClockJump {
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    #[getter]
    fn epoch(&self) -> PyObsEpochTime {
        self.inner.epoch.into()
    }

    #[getter]
    fn delta_s(&self) -> f64 {
        self.inner.delta_s
    }
}

/// RMS statistics for one multipath series.
#[pyclass(module = "sidereon._sidereon", name = "MpStats")]
#[derive(Clone, Copy)]
pub struct PyMpStats {
    inner: MpStats,
}

impl From<MpStats> for PyMpStats {
    fn from(inner: MpStats) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMpStats {
    #[getter]
    fn n(&self) -> usize {
        self.inner.n
    }

    #[getter]
    fn rms_m(&self) -> f64 {
        self.inner.rms_m
    }
}

/// Per-satellite MP1/MP2 multipath RMS.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteMultipathQc")]
#[derive(Clone)]
pub struct PySatelliteMultipathQc {
    inner: SatelliteMultipathQc,
}

impl From<SatelliteMultipathQc> for PySatelliteMultipathQc {
    fn from(inner: SatelliteMultipathQc) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySatelliteMultipathQc {
    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    #[getter]
    fn mp1(&self) -> Option<PyMpStats> {
        self.inner.mp1.map(Into::into)
    }

    #[getter]
    fn mp2(&self) -> Option<PyMpStats> {
        self.inner.mp2.map(Into::into)
    }
}

/// Per-system MP1/MP2 multipath RMS.
#[pyclass(module = "sidereon._sidereon", name = "SystemMultipathQc")]
#[derive(Clone)]
pub struct PySystemMultipathQc {
    inner: SystemMultipathQc,
}

impl From<SystemMultipathQc> for PySystemMultipathQc {
    fn from(inner: SystemMultipathQc) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySystemMultipathQc {
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    #[getter]
    fn mp1(&self) -> Option<PyMpStats> {
        self.inner.mp1.map(Into::into)
    }

    #[getter]
    fn mp2(&self) -> Option<PyMpStats> {
        self.inner.mp2.map(Into::into)
    }
}

/// MP1/MP2 multipath RMS report.
#[pyclass(module = "sidereon._sidereon", name = "MultipathReport")]
#[derive(Clone)]
pub struct PyMultipathReport {
    inner: MultipathReport,
}

impl From<MultipathReport> for PyMultipathReport {
    fn from(inner: MultipathReport) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyMultipathReport {
    #[getter]
    fn satellites(&self) -> Vec<PySatelliteMultipathQc> {
        self.inner
            .satellites
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn systems(&self) -> Vec<PySystemMultipathQc> {
        self.inner.systems.iter().cloned().map(Into::into).collect()
    }
}

/// Per-constellation cycle-slip counts.
#[pyclass(module = "sidereon._sidereon", name = "SystemCycleSlipQc")]
#[derive(Clone, Copy)]
pub struct PySystemCycleSlipQc {
    inner: SystemCycleSlipQc,
}

impl From<SystemCycleSlipQc> for PySystemCycleSlipQc {
    fn from(inner: SystemCycleSlipQc) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySystemCycleSlipQc {
    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    #[getter]
    fn observations(&self) -> usize {
        self.inner.observations
    }

    #[getter]
    fn slips(&self) -> usize {
        self.inner.slips
    }

    #[getter]
    fn observations_per_slip(&self) -> Option<f64> {
        self.inner.observations_per_slip
    }
}

/// Aggregate dual-frequency cycle-slip counts.
#[pyclass(module = "sidereon._sidereon", name = "CycleSlipQc")]
#[derive(Clone)]
pub struct PyCycleSlipQc {
    inner: CycleSlipQc,
}

impl From<CycleSlipQc> for PyCycleSlipQc {
    fn from(inner: CycleSlipQc) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCycleSlipQc {
    #[getter]
    fn observations(&self) -> usize {
        self.inner.observations
    }

    #[getter]
    fn total_slips(&self) -> usize {
        self.inner.total_slips
    }

    #[getter]
    fn observations_per_slip(&self) -> Option<f64> {
        self.inner.observations_per_slip
    }

    #[getter]
    fn by_system(&self) -> Vec<PySystemCycleSlipQc> {
        self.inner
            .by_system
            .iter()
            .copied()
            .map(Into::into)
            .collect()
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
            ObservationQcNote::EventHeaderRecordsUnread => "event_header_records_unread",
        }
    }

    #[getter]
    fn epoch_index(&self) -> Option<usize> {
        match self.inner {
            ObservationQcNote::NonMonotonicEpoch { epoch_index } => Some(epoch_index),
            ObservationQcNote::IntervalUnresolved | ObservationQcNote::EventHeaderRecordsUnread => {
                None
            }
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            ObservationQcNote::NonMonotonicEpoch { epoch_index } => {
                format!(
                    "ObservationQcNote(kind=\"non_monotonic_epoch\", epoch_index={epoch_index})"
                )
            }
            ObservationQcNote::IntervalUnresolved => {
                "ObservationQcNote(kind=\"interval_unresolved\")".to_string()
            }
            ObservationQcNote::EventHeaderRecordsUnread => {
                "ObservationQcNote(kind=\"event_header_records_unread\")".to_string()
            }
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
    fn clock_jumps(&self) -> Vec<PyClockJump> {
        self.inner
            .clock_jumps
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn cycle_slips(&self) -> PyCycleSlipQc {
        self.inner.cycle_slips.clone().into()
    }

    #[getter]
    fn multipath(&self) -> PyMultipathReport {
        self.inner.multipath.clone().into()
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
    fn lint_findings(&self) -> Vec<PyObservationQcFinding> {
        self.inner
            .lint_findings
            .iter()
            .cloned()
            .map(Into::into)
            .collect()
    }

    #[getter]
    fn notes(&self) -> Vec<PyObservationQcNote> {
        self.inner.notes.iter().copied().map(Into::into).collect()
    }

    fn render_text(&self) -> String {
        core_observation_qc_render_text(&self.inner)
    }

    fn render_html(&self) -> String {
        core_observation_qc_render_html(&self.inner)
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "ObservationQcReport(observation_epochs={}, satellites={})",
            self.inner.observation_epochs,
            self.inner.satellites.len()
        )
    }
}

/// Compact RINEX lint finding retained by an observation QC report.
#[pyclass(module = "sidereon._sidereon", name = "ObservationQcFinding")]
#[derive(Clone)]
pub struct PyObservationQcFinding {
    inner: ObservationQcFinding,
}

impl From<ObservationQcFinding> for PyObservationQcFinding {
    fn from(inner: ObservationQcFinding) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObservationQcFinding {
    #[getter]
    fn code(&self) -> &str {
        &self.inner.code
    }

    #[getter]
    fn severity(&self) -> PyRinexLintSeverity {
        self.inner.severity.into()
    }

    #[getter]
    fn spec_ref(&self) -> &str {
        &self.inner.spec_ref
    }

    fn __repr__(&self) -> String {
        format!(
            "ObservationQcFinding(code={}, severity={:?})",
            self.inner.code, self.inner.severity
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
    let mut options = ObservationQcOptions::default();
    options.interval_override_s = interval_override_s;
    options.gap_factor = gap_factor;
    let report = core_observation_qc_with_options(obs.inner(), options)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok(report.into())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRaimInput>()?;
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
    m.add_class::<PyClockJump>()?;
    m.add_class::<PyMpStats>()?;
    m.add_class::<PySatelliteMultipathQc>()?;
    m.add_class::<PySystemMultipathQc>()?;
    m.add_class::<PyMultipathReport>()?;
    m.add_class::<PySystemCycleSlipQc>()?;
    m.add_class::<PyCycleSlipQc>()?;
    m.add_class::<PyObservationQcFinding>()?;
    m.add_class::<PyObservationQcNote>()?;
    m.add_class::<PyObservationQcReport>()?;
    m.add_function(wrap_pyfunction!(qc_raim, m)?)?;
    m.add_function(wrap_pyfunction!(raim, m)?)?;
    m.add_function(wrap_pyfunction!(raim_for_solution, m)?)?;
    m.add_function(wrap_pyfunction!(qc_fde, m)?)?;
    m.add_function(wrap_pyfunction!(qc_fde_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(fde_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp_robust_fde, m)?)?;
    m.add_function(wrap_pyfunction!(spp_robust_fde_driver, m)?)?;
    m.add_function(wrap_pyfunction!(qc_raim_fde_design, m)?)?;
    m.add_function(wrap_pyfunction!(chi2_inv, m)?)?;
    m.add_function(wrap_pyfunction!(observation_qc, m)?)?;
    Ok(())
}
