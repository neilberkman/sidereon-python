//! RTK baseline binding: static float and validated fixed solves.
//!
//! The RTK epoch input is a structured record (per-satellite base/rover
//! code+phase plus transmit-time satellite positions) that the engine builds
//! from RINEX+SP3 upstream. This binding only MARSHALS that record: Python dicts
//! are deserialized into mirror structs and rebuilt into the `sidereon-core`
//! input types, then handed to `sidereon::solve_rtk_float` /
//! `sidereon::solve_rtk_fixed`. No modeling happens here.

use std::collections::BTreeMap;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::positioning::{
    solve_static_reference_station_rinex as core_solve_static_reference_station_rinex,
    RinexSppOptions, StaticReferenceCarrierRinexOptions, StaticReferenceCarrierSolution,
    StaticReferenceCodeSolution, StaticReferenceEpochDiagnostic, StaticReferenceFixStatus,
    StaticReferenceModeError, StaticReferenceModeReport, StaticReferenceModeStatus,
    StaticReferenceStationCovariance, StaticReferenceStationMode,
    StaticReferenceStationRinexOptions, StaticReferenceStationSolution,
};
use sidereon_core::rtk::{BaselineReferenceSelection, CycleSlipReceiver};
use sidereon_core::rtk_filter::defaults::{
    AMBIGUITY_TOL_M, MAX_ITERATIONS, PARTIAL_MIN_AMBIGUITIES, POSITION_TOL_M, RATIO_THRESHOLD,
};
use sidereon_core::rtk_filter::{
    build_dual_frequency_rinex_rtk_arc as core_build_dual_frequency_rinex_rtk_arc,
    build_rinex_rtk_arc as core_build_rinex_rtk_arc,
    fix_wide_lane_rtk_arc as core_fix_wide_lane_rtk_arc,
    prepare_ionosphere_free_rtk_arc as core_prepare_ionosphere_free_rtk_arc,
    solve_moving_baseline as core_solve_moving_baseline, solve_rtk_arc as core_solve_rtk_arc,
    solve_static_rtk_arc as core_solve_static_rtk_arc,
    solve_wide_lane_fixed_rtk_arc as core_solve_wide_lane_fixed_rtk_arc, AmbiguityScale,
    AmbiguitySet, CycleSlipOptions, CycleSlipPolicy, CycleSlipSplitArc, DynamicsModel, Epoch,
    FilterState, FixedSolveOpts, FloatBaselineSolution, FloatResidual, FloatSolveOpts,
    IntegerSearchMeta, IntegerStatus as RtkIntegerStatus, MeasModel, MovingBaselineEpoch,
    MovingBaselineEpochSolution, MovingBaselineOpts, MovingBaselineStatus, ResidualValidationOpts,
    RtkArcConfig, RtkArcEpoch, RtkArcEpochSolution, RtkArcObservation, RtkArcPreprocessing,
    RtkArcSolution, RtkDualCycleSlipConfig, RtkDualFrequencyArcEpoch, RtkDualFrequencyObservation,
    RtkDualFrequencySatelliteObservation, RtkIonosphereFreeArcConfig, RtkIonosphereFreeArcSolution,
    RtkRinexArc as CoreRtkRinexArc, RtkRinexArcOptions as CoreRtkRinexArcOptions,
    RtkRinexDualArcOptions, RtkRinexDualFrequencyArc as CoreRtkRinexDualFrequencyArc,
    RtkRinexDualSignalPair, RtkRinexSignalPair, RtkStaticArcConfig, RtkStaticArcSolution,
    RtkWideLaneArcConfig, RtkWideLaneArcSolution, RtkWideLaneFixedArcConfig,
    RtkWideLaneFixedArcSolution, RtkWideLaneFixedArcSolveConfig, RtkWideLaneFixedStaticArcSolution,
    SatMeas, SearchOpts, StochasticModel, UpdateOpts, ValidatedFixedBaselineSolution,
    ValidatedFixedSolveOpts, WideLaneOptions,
};

use crate::ephemeris::with_observable_source;
use crate::geometry_quality::PyGeometryQuality;
use crate::marshal::{mat3_to_array, option_py_or_default, PyGnssSystem};
use crate::rinex::PyRinexObs;
use crate::{np_array, to_solve_err, PySp3};

// --- input value/config objects -------------------------------------------

/// Integer ambiguity-fix status.
#[pyclass(module = "sidereon._sidereon", name = "IntegerStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub(crate) enum PyIntegerStatus {
    /// The ambiguity search accepted an integer fix.
    FIXED,
    /// The ambiguity search rejected the integer fix.
    NOT_FIXED,
}

impl From<RtkIntegerStatus> for PyIntegerStatus {
    fn from(status: RtkIntegerStatus) -> Self {
        match status {
            RtkIntegerStatus::Fixed => PyIntegerStatus::FIXED,
            RtkIntegerStatus::NotFixed => PyIntegerStatus::NOT_FIXED,
        }
    }
}

#[pymethods]
impl PyIntegerStatus {
    fn __repr__(&self) -> &'static str {
        match self {
            PyIntegerStatus::FIXED => "IntegerStatus.FIXED",
            PyIntegerStatus::NOT_FIXED => "IntegerStatus.NOT_FIXED",
        }
    }
}

/// RTK stochastic measurement weighting model.
#[pyclass(module = "sidereon._sidereon", name = "RtkStochasticModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyRtkStochasticModel {
    /// Simple sigma model, optionally elevation weighted.
    SIMPLE,
    /// RTKLIB-compatible stochastic model.
    RTKLIB,
}

impl PyRtkStochasticModel {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "simple" => Ok(Self::SIMPLE),
            "rtklib" => Ok(Self::RTKLIB),
            other => Err(PyValueError::new_err(format!(
                "unknown stochastic model {other:?}; expected \"simple\" or \"rtklib\""
            ))),
        }
    }
}

impl From<StochasticModel> for PyRtkStochasticModel {
    fn from(model: StochasticModel) -> Self {
        match model {
            StochasticModel::Simple { .. } => Self::SIMPLE,
            StochasticModel::Rtklib => Self::RTKLIB,
        }
    }
}

#[pymethods]
impl PyRtkStochasticModel {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            PyRtkStochasticModel::SIMPLE => "simple",
            PyRtkStochasticModel::RTKLIB => "rtklib",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            PyRtkStochasticModel::SIMPLE => "RtkStochasticModel.SIMPLE",
            PyRtkStochasticModel::RTKLIB => "RtkStochasticModel.RTKLIB",
        }
    }
}

fn extract_stochastic_model(obj: &Bound<'_, PyAny>) -> PyResult<PyRtkStochasticModel> {
    if let Ok(model) = obj.extract::<PyRtkStochasticModel>() {
        return Ok(model);
    }
    PyRtkStochasticModel::from_label(&obj.extract::<String>()?)
}

fn default_rtk_model() -> MeasModel {
    MeasModel {
        code_sigma_m: sidereon_core::rtk_filter::defaults::CODE_SIGMA_M,
        phase_sigma_m: sidereon_core::rtk_filter::defaults::PHASE_SIGMA_M,
        sagnac: true,
        stochastic: StochasticModel::Rtklib,
    }
}

fn flat_square_to_array<'py>(
    py: Python<'py>,
    values: &[f64],
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = (values.len() as f64).sqrt() as usize;
    if n * n != values.len() {
        return Err(PyValueError::new_err(
            "covariance length is not a square matrix",
        ));
    }
    let mut array = Array2::<f64>::zeros((n, n));
    for row in 0..n {
        for col in 0..n {
            array[[row, col]] = values[row * n + col];
        }
    }
    Ok(PyArray2::from_owned_array(py, array))
}

/// One satellite's base/rover measurements for an RTK epoch.
#[pyclass(module = "sidereon._sidereon", name = "RtkSatMeasurement")]
#[derive(Clone)]
pub struct PyRtkSatMeasurement {
    inner: SatMeas,
}

#[pymethods]
impl PyRtkSatMeasurement {
    /// Create one base/rover code+phase measurement row.
    #[new]
    #[pyo3(signature = (
        sat,
        sd_ambiguity_id,
        base_code_m,
        base_phase_m,
        rover_code_m,
        rover_phase_m,
        base_tx_pos,
        rover_tx_pos,
        pos,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        sat: String,
        sd_ambiguity_id: String,
        base_code_m: f64,
        base_phase_m: f64,
        rover_code_m: f64,
        rover_phase_m: f64,
        base_tx_pos: [f64; 3],
        rover_tx_pos: [f64; 3],
        pos: [f64; 3],
    ) -> Self {
        Self {
            inner: SatMeas {
                sat,
                sd_ambiguity_id,
                base_code_m,
                base_phase_m,
                rover_code_m,
                rover_phase_m,
                base_tx_pos,
                rover_tx_pos,
                pos,
            },
        }
    }

    #[getter]
    fn sat(&self) -> &str {
        &self.inner.sat
    }

    #[getter]
    fn sd_ambiguity_id(&self) -> &str {
        &self.inner.sd_ambiguity_id
    }

    #[getter]
    fn base_code_m(&self) -> f64 {
        self.inner.base_code_m
    }

    #[getter]
    fn base_phase_m(&self) -> f64 {
        self.inner.base_phase_m
    }

    #[getter]
    fn rover_code_m(&self) -> f64 {
        self.inner.rover_code_m
    }

    #[getter]
    fn rover_phase_m(&self) -> f64 {
        self.inner.rover_phase_m
    }

    #[getter]
    fn base_tx_pos(&self) -> [f64; 3] {
        self.inner.base_tx_pos
    }

    #[getter]
    fn rover_tx_pos(&self) -> [f64; 3] {
        self.inner.rover_tx_pos
    }

    #[getter]
    fn pos(&self) -> [f64; 3] {
        self.inner.pos
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkSatMeasurement(sat={:?}, sd_ambiguity_id={:?})",
            self.inner.sat, self.inner.sd_ambiguity_id
        )
    }
}

impl PyRtkSatMeasurement {
    fn to_core(&self) -> SatMeas {
        self.inner.clone()
    }
}

/// One RTK epoch with reference and non-reference satellite rows.
#[pyclass(module = "sidereon._sidereon", name = "RtkEpoch")]
#[derive(Clone)]
pub struct PyRtkEpoch {
    inner: Epoch,
}

#[pymethods]
impl PyRtkEpoch {
    /// Create one RTK epoch.
    #[new]
    #[pyo3(signature = (references, nonref, dt_s, velocity_mps=None))]
    fn new(
        py: Python<'_>,
        references: Vec<Py<PyRtkSatMeasurement>>,
        nonref: Vec<Py<PyRtkSatMeasurement>>,
        dt_s: f64,
        velocity_mps: Option<[f64; 3]>,
    ) -> Self {
        let references = references
            .iter()
            .map(|row| row.borrow(py).to_core())
            .collect();
        let nonref = nonref.iter().map(|row| row.borrow(py).to_core()).collect();
        Self {
            inner: Epoch {
                references,
                nonref,
                velocity_mps,
                dt_s,
            },
        }
    }

    #[getter]
    fn reference_count(&self) -> usize {
        self.inner.references.len()
    }

    #[getter]
    fn nonref_count(&self) -> usize {
        self.inner.nonref.len()
    }

    #[getter]
    fn velocity_mps(&self) -> Option<[f64; 3]> {
        self.inner.velocity_mps
    }

    #[getter]
    fn dt_s(&self) -> f64 {
        self.inner.dt_s
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkEpoch(references={}, nonref={}, dt_s={:.3})",
            self.inner.references.len(),
            self.inner.nonref.len(),
            self.inner.dt_s
        )
    }
}

impl PyRtkEpoch {
    fn to_core(&self) -> Epoch {
        self.inner.clone()
    }
}

/// RTK measurement weighting and correction model.
#[pyclass(module = "sidereon._sidereon", name = "RtkMeasurementModel")]
#[derive(Clone, Copy)]
pub struct PyRtkMeasurementModel {
    inner: MeasModel,
}

#[pymethods]
impl PyRtkMeasurementModel {
    /// Create an RTK measurement model.
    #[new]
    #[pyo3(signature = (
        code_sigma_m,
        phase_sigma_m,
        sagnac=true,
        stochastic=PyRtkStochasticModel::SIMPLE,
        elevation_weighting=false,
    ))]
    fn new(
        code_sigma_m: f64,
        phase_sigma_m: f64,
        sagnac: bool,
        #[pyo3(from_py_with = extract_stochastic_model)] stochastic: PyRtkStochasticModel,
        elevation_weighting: bool,
    ) -> PyResult<Self> {
        let stochastic = match stochastic {
            PyRtkStochasticModel::SIMPLE => StochasticModel::Simple {
                elevation_weighting,
            },
            PyRtkStochasticModel::RTKLIB => StochasticModel::Rtklib,
        };
        Ok(Self {
            inner: MeasModel {
                code_sigma_m,
                phase_sigma_m,
                sagnac,
                stochastic,
            },
        })
    }

    #[getter]
    fn code_sigma_m(&self) -> f64 {
        self.inner.code_sigma_m
    }

    #[getter]
    fn phase_sigma_m(&self) -> f64 {
        self.inner.phase_sigma_m
    }

    #[getter]
    fn sagnac(&self) -> bool {
        self.inner.sagnac
    }

    #[getter]
    fn stochastic(&self) -> PyRtkStochasticModel {
        PyRtkStochasticModel::from(self.inner.stochastic)
    }

    #[getter]
    fn elevation_weighting(&self) -> bool {
        match self.inner.stochastic {
            StochasticModel::Simple {
                elevation_weighting,
            } => elevation_weighting,
            StochasticModel::Rtklib => false,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkMeasurementModel(code_sigma_m={:.3}, phase_sigma_m={:.5}, stochastic={:?})",
            self.inner.code_sigma_m,
            self.inner.phase_sigma_m,
            self.stochastic().label()
        )
    }
}

/// Iteration controls for an RTK float solve.
#[pyclass(module = "sidereon._sidereon", name = "RtkFloatOptions")]
#[derive(Clone, Copy)]
pub struct PyRtkFloatOptions {
    inner: FloatSolveOpts,
}

#[pymethods]
impl PyRtkFloatOptions {
    /// Create RTK float solve controls.
    #[new]
    #[pyo3(signature = (position_tol_m=POSITION_TOL_M, ambiguity_tol_m=AMBIGUITY_TOL_M, max_iterations=MAX_ITERATIONS))]
    fn new(position_tol_m: f64, ambiguity_tol_m: f64, max_iterations: usize) -> Self {
        Self {
            inner: FloatSolveOpts {
                position_tol_m,
                ambiguity_tol_m,
                max_iterations,
            },
        }
    }

    #[getter]
    fn position_tol_m(&self) -> f64 {
        self.inner.position_tol_m
    }

    #[getter]
    fn ambiguity_tol_m(&self) -> f64 {
        self.inner.ambiguity_tol_m
    }

    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkFloatOptions(position_tol_m={:.3e}, ambiguity_tol_m={:.3e}, max_iterations={})",
            self.inner.position_tol_m, self.inner.ambiguity_tol_m, self.inner.max_iterations
        )
    }
}

impl Default for PyRtkFloatOptions {
    fn default() -> Self {
        Self::new(POSITION_TOL_M, AMBIGUITY_TOL_M, MAX_ITERATIONS)
    }
}

/// Iteration and integer-search controls for RTK fixed solving.
#[pyclass(module = "sidereon._sidereon", name = "RtkFixedOptions")]
#[derive(Clone, Copy)]
pub struct PyRtkFixedOptions {
    inner: FixedSolveOpts,
}

#[pymethods]
impl PyRtkFixedOptions {
    /// Create RTK fixed solve controls.
    #[new]
    #[pyo3(signature = (
        position_tol_m=POSITION_TOL_M,
        ambiguity_tol_m=AMBIGUITY_TOL_M,
        max_iterations=MAX_ITERATIONS,
        ratio_threshold=RATIO_THRESHOLD,
        partial_ambiguity_resolution=false,
        partial_min_ambiguities=PARTIAL_MIN_AMBIGUITIES,
    ))]
    fn new(
        position_tol_m: f64,
        ambiguity_tol_m: f64,
        max_iterations: usize,
        ratio_threshold: f64,
        partial_ambiguity_resolution: bool,
        partial_min_ambiguities: usize,
    ) -> Self {
        Self {
            inner: FixedSolveOpts {
                position_tol_m,
                ambiguity_tol_m,
                max_iterations,
                ratio_threshold,
                partial_ambiguity_resolution,
                partial_min_ambiguities,
            },
        }
    }

    #[getter]
    fn position_tol_m(&self) -> f64 {
        self.inner.position_tol_m
    }

    #[getter]
    fn ambiguity_tol_m(&self) -> f64 {
        self.inner.ambiguity_tol_m
    }

    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    #[getter]
    fn ratio_threshold(&self) -> f64 {
        self.inner.ratio_threshold
    }

    #[getter]
    fn partial_ambiguity_resolution(&self) -> bool {
        self.inner.partial_ambiguity_resolution
    }

    #[getter]
    fn partial_min_ambiguities(&self) -> usize {
        self.inner.partial_min_ambiguities
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkFixedOptions(ratio_threshold={:.3}, partial_ambiguity_resolution={})",
            self.inner.ratio_threshold, self.inner.partial_ambiguity_resolution
        )
    }
}

impl Default for PyRtkFixedOptions {
    fn default() -> Self {
        Self::new(
            POSITION_TOL_M,
            AMBIGUITY_TOL_M,
            MAX_ITERATIONS,
            RATIO_THRESHOLD,
            false,
            PARTIAL_MIN_AMBIGUITIES,
        )
    }
}

/// Residual validation controls for RTK fixed solving.
#[pyclass(module = "sidereon._sidereon", name = "RtkResidualValidationOptions")]
#[derive(Clone, Copy)]
pub struct PyRtkResidualValidationOptions {
    inner: ResidualValidationOpts,
}

#[pymethods]
impl PyRtkResidualValidationOptions {
    /// Create residual-validation controls.
    #[new]
    #[pyo3(signature = (threshold_sigma=None, max_exclusions=0))]
    fn new(threshold_sigma: Option<f64>, max_exclusions: usize) -> Self {
        Self {
            inner: ResidualValidationOpts {
                threshold_sigma,
                max_exclusions,
            },
        }
    }

    #[getter]
    fn threshold_sigma(&self) -> Option<f64> {
        self.inner.threshold_sigma
    }

    #[getter]
    fn max_exclusions(&self) -> usize {
        self.inner.max_exclusions
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkResidualValidationOptions(threshold_sigma={:?}, max_exclusions={})",
            self.inner.threshold_sigma, self.inner.max_exclusions
        )
    }
}

impl Default for PyRtkResidualValidationOptions {
    fn default() -> Self {
        Self::new(None, 0)
    }
}

fn validated_fixed_opts_from_py(
    py: Python<'_>,
    float_options: Option<&Py<PyRtkFloatOptions>>,
    fixed_options: Option<&Py<PyRtkFixedOptions>>,
    residual_options: Option<&Py<PyRtkResidualValidationOptions>>,
) -> ValidatedFixedSolveOpts {
    let float = option_py_or_default(
        py,
        float_options,
        |value| value.inner,
        || PyRtkFloatOptions::default().inner,
    );
    let fixed = option_py_or_default(
        py,
        fixed_options,
        |value| value.inner,
        || PyRtkFixedOptions::default().inner,
    );
    let residual = option_py_or_default(
        py,
        residual_options,
        |value| value.inner,
        || PyRtkResidualValidationOptions::default().inner,
    );
    ValidatedFixedSolveOpts {
        float,
        fixed,
        residual,
    }
}

/// Complete typed input bundle for an RTK float solve.
#[pyclass(module = "sidereon._sidereon", name = "RtkFloatConfig")]
pub struct PyRtkFloatConfig {
    epochs: Vec<Epoch>,
    base: [f64; 3],
    ambiguity_ids: Vec<String>,
    model: MeasModel,
    initial_baseline_m: [f64; 3],
    opts: FloatSolveOpts,
}

#[pymethods]
impl PyRtkFloatConfig {
    /// Create an RTK float solve configuration.
    #[new]
    #[pyo3(signature = (
        epochs,
        base,
        ambiguity_ids,
        model,
        initial_baseline_m=[0.0; 3],
        options=None,
    ))]
    fn new(
        py: Python<'_>,
        epochs: Vec<Py<PyRtkEpoch>>,
        base: [f64; 3],
        ambiguity_ids: Vec<String>,
        model: &PyRtkMeasurementModel,
        initial_baseline_m: [f64; 3],
        options: Option<Py<PyRtkFloatOptions>>,
    ) -> Self {
        let epochs = epochs
            .iter()
            .map(|epoch| epoch.borrow(py).to_core())
            .collect();
        let opts = option_py_or_default(
            py,
            options.as_ref(),
            |value| value.inner,
            || PyRtkFloatOptions::default().inner,
        );
        Self {
            epochs,
            base,
            ambiguity_ids,
            model: model.inner,
            initial_baseline_m,
            opts,
        }
    }

    #[getter]
    fn epoch_count(&self) -> usize {
        self.epochs.len()
    }

    #[getter]
    fn base(&self) -> [f64; 3] {
        self.base
    }

    #[getter]
    fn ambiguity_ids(&self) -> Vec<String> {
        self.ambiguity_ids.clone()
    }

    #[getter]
    fn initial_baseline_m(&self) -> [f64; 3] {
        self.initial_baseline_m
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkFloatConfig(epochs={}, ambiguity_ids={})",
            self.epochs.len(),
            self.ambiguity_ids.len()
        )
    }
}

/// Complete typed input bundle for an RTK fixed solve.
#[pyclass(module = "sidereon._sidereon", name = "RtkFixedConfig")]
pub struct PyRtkFixedConfig {
    epochs: Vec<Epoch>,
    base: [f64; 3],
    ambiguity_ids: Vec<String>,
    ambiguity_satellites: BTreeMap<String, String>,
    wavelengths_m: BTreeMap<String, f64>,
    offsets_m: BTreeMap<String, f64>,
    model: MeasModel,
    opts: ValidatedFixedSolveOpts,
    float_only_systems: Vec<String>,
    initial_baseline_m: [f64; 3],
}

#[pymethods]
impl PyRtkFixedConfig {
    /// Create an RTK fixed solve configuration.
    #[new]
    #[pyo3(signature = (
        epochs,
        base,
        ambiguity_ids,
        ambiguity_satellites,
        wavelengths_m,
        offsets_m,
        model,
        float_options=None,
        fixed_options=None,
        residual_options=None,
        float_only_systems=Vec::new(),
        initial_baseline_m=[0.0; 3],
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        epochs: Vec<Py<PyRtkEpoch>>,
        base: [f64; 3],
        ambiguity_ids: Vec<String>,
        ambiguity_satellites: BTreeMap<String, String>,
        wavelengths_m: BTreeMap<String, f64>,
        offsets_m: BTreeMap<String, f64>,
        model: &PyRtkMeasurementModel,
        float_options: Option<Py<PyRtkFloatOptions>>,
        fixed_options: Option<Py<PyRtkFixedOptions>>,
        residual_options: Option<Py<PyRtkResidualValidationOptions>>,
        float_only_systems: Vec<String>,
        initial_baseline_m: [f64; 3],
    ) -> Self {
        let epochs = epochs
            .iter()
            .map(|epoch| epoch.borrow(py).to_core())
            .collect();
        let opts = validated_fixed_opts_from_py(
            py,
            float_options.as_ref(),
            fixed_options.as_ref(),
            residual_options.as_ref(),
        );
        Self {
            epochs,
            base,
            ambiguity_ids,
            ambiguity_satellites,
            wavelengths_m,
            offsets_m,
            model: model.inner,
            opts,
            float_only_systems,
            initial_baseline_m,
        }
    }

    #[getter]
    fn epoch_count(&self) -> usize {
        self.epochs.len()
    }

    #[getter]
    fn base(&self) -> [f64; 3] {
        self.base
    }

    #[getter]
    fn ambiguity_ids(&self) -> Vec<String> {
        self.ambiguity_ids.clone()
    }

    #[getter]
    fn float_only_systems(&self) -> Vec<String> {
        self.float_only_systems.clone()
    }

    #[getter]
    fn initial_baseline_m(&self) -> [f64; 3] {
        self.initial_baseline_m
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkFixedConfig(epochs={}, ambiguity_ids={}, float_only_systems={})",
            self.epochs.len(),
            self.ambiguity_ids.len(),
            self.float_only_systems.len()
        )
    }
}

// --- result objects --------------------------------------------------------

/// Static float RTK baseline solution.
///
/// `baseline` is the rover-minus-base ECEF baseline as a numpy `float64` array
/// of shape `(3,)`, metres. Ambiguity estimates are keyed by ambiguity id.
#[pyclass(module = "sidereon._sidereon", name = "RtkFloatSolution")]
pub struct PyRtkFloatSolution {
    inner: FloatBaselineSolution,
}

#[pymethods]
impl PyRtkFloatSolution {
    /// Baseline (rover minus base) as a numpy array `[dx, dy, dz]` metres.
    #[getter]
    fn baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.baseline_m)
    }

    #[getter]
    fn baseline_m(&self) -> [f64; 3] {
        self.inner.baseline_m
    }

    /// Baseline covariance matrix, metres squared.
    #[getter]
    fn baseline_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.baseline_covariance_m2)
    }

    /// Float single-difference ambiguities in metres, keyed by ambiguity id.
    #[getter]
    fn ambiguities_m(&self) -> BTreeMap<String, f64> {
        self.inner.ambiguities_m.iter().cloned().collect()
    }

    /// Float ambiguity covariance matrix, metres squared.
    #[getter]
    fn ambiguity_covariance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        flat_square_to_array(py, &self.inner.ambiguity_covariance_m)
    }

    #[getter]
    fn code_rms_m(&self) -> f64 {
        self.inner.code_rms_m
    }

    #[getter]
    fn phase_rms_m(&self) -> f64 {
        self.inner.phase_rms_m
    }

    #[getter]
    fn weighted_rms_m(&self) -> f64 {
        self.inner.weighted_rms_m
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn n_observations(&self) -> usize {
        self.inner.n_observations
    }

    /// Geometry observability and covariance-validation diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkFloatSolution(baseline=[{:.4}, {:.4}, {:.4}], phase_rms_m={:.4}, converged={})",
            self.inner.baseline_m[0],
            self.inner.baseline_m[1],
            self.inner.baseline_m[2],
            self.inner.phase_rms_m,
            self.inner.converged
        )
    }
}

/// Validated fixed RTK baseline solution.
///
/// `fixed_baseline` and `float_baseline` are rover-minus-base ECEF baselines as
/// numpy `float64` arrays of shape `(3,)`, metres.
#[pyclass(module = "sidereon._sidereon", name = "RtkFixedSolution")]
pub struct PyRtkFixedSolution {
    inner: ValidatedFixedBaselineSolution,
}

#[pymethods]
impl PyRtkFixedSolution {
    /// Fixed (integer-resolved) baseline as a numpy array `[dx, dy, dz]` metres.
    #[getter]
    fn fixed_baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.fixed_solution.baseline_m)
    }

    #[getter]
    fn fixed_baseline_m(&self) -> [f64; 3] {
        self.inner.fixed_solution.baseline_m
    }

    /// Fixed baseline covariance matrix, metres squared.
    #[getter]
    fn fixed_baseline_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.fixed_solution.baseline_covariance_m2)
    }

    /// The underlying float baseline as a numpy array `[dx, dy, dz]` metres.
    #[getter]
    fn float_baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.float_solution.baseline_m)
    }

    #[getter]
    fn float_baseline_m(&self) -> [f64; 3] {
        self.inner.float_solution.baseline_m
    }

    /// Integer ambiguity-fix status.
    #[getter]
    fn integer_status(&self) -> PyIntegerStatus {
        self.inner.fixed_solution.search.integer_status.into()
    }

    #[getter]
    fn integer_ratio(&self) -> Option<f64> {
        self.inner.fixed_solution.search.integer_ratio
    }

    #[getter]
    fn integer_candidates(&self) -> usize {
        self.inner.fixed_solution.search.integer_candidates
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.fixed_solution.converged
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkFixedSolution(fixed_baseline=[{:.4}, {:.4}, {:.4}], integer_status={:?})",
            self.inner.fixed_solution.baseline_m[0],
            self.inner.fixed_solution.baseline_m[1],
            self.inner.fixed_solution.baseline_m[2],
            self.inner.fixed_solution.search.integer_status
        )
    }
}

// --- solve entry points ----------------------------------------------------

#[pyfunction]
#[pyo3(signature = (config))]
fn solve_rtk_float(config: &PyRtkFloatConfig) -> PyResult<PyRtkFloatSolution> {
    let inner = sidereon::solve_rtk_float(
        &config.epochs,
        config.base,
        &config.ambiguity_ids,
        config.initial_baseline_m,
        &config.model,
        config.opts,
        None,
    )
    .map_err(to_solve_err)?;
    Ok(PyRtkFloatSolution { inner })
}

#[pyfunction]
#[pyo3(signature = (config))]
fn solve_rtk_fixed(config: &PyRtkFixedConfig) -> PyResult<PyRtkFixedSolution> {
    let ambiguities = AmbiguitySet {
        ids: &config.ambiguity_ids,
        satellites: &config.ambiguity_satellites,
        scale: AmbiguityScale {
            wavelengths_m: &config.wavelengths_m,
            offsets_m: &config.offsets_m,
        },
        float_only_systems: &config.float_only_systems,
    };

    let inner = sidereon::solve_rtk_fixed(
        &config.epochs,
        config.base,
        ambiguities,
        config.initial_baseline_m,
        &config.model,
        config.opts,
        None,
    )
    .map_err(to_solve_err)?;
    Ok(PyRtkFixedSolution { inner })
}

// --- moving-baseline RTK ---------------------------------------------------

/// One moving-baseline epoch: the base receiver's own ECEF position this epoch,
/// the double-difference observations, and the ambiguity set to resolve.
///
/// Both receivers move, so the base position is supplied per epoch (typically
/// the base's own navigation fix) rather than held constant. The observation
/// epoch and ambiguity set are exactly the inputs the static fixed-base solver
/// takes.
#[pyclass(module = "sidereon._sidereon", name = "MovingBaselineEpoch")]
pub struct PyMovingBaselineEpoch {
    base_position_m: [f64; 3],
    epoch: Epoch,
    ambiguity_ids: Vec<String>,
    ambiguity_satellites: BTreeMap<String, String>,
    wavelengths_m: BTreeMap<String, f64>,
    offsets_m: BTreeMap<String, f64>,
    float_only_systems: Vec<String>,
}

#[pymethods]
impl PyMovingBaselineEpoch {
    /// Create one moving-baseline epoch.
    #[new]
    #[pyo3(signature = (
        base_position_m,
        epoch,
        ambiguity_ids,
        ambiguity_satellites,
        wavelengths_m,
        offsets_m,
        float_only_systems=Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        base_position_m: [f64; 3],
        epoch: &PyRtkEpoch,
        ambiguity_ids: Vec<String>,
        ambiguity_satellites: BTreeMap<String, String>,
        wavelengths_m: BTreeMap<String, f64>,
        offsets_m: BTreeMap<String, f64>,
        float_only_systems: Vec<String>,
    ) -> Self {
        Self {
            base_position_m,
            epoch: epoch.to_core(),
            ambiguity_ids,
            ambiguity_satellites,
            wavelengths_m,
            offsets_m,
            float_only_systems,
        }
    }

    #[getter]
    fn base_position_m(&self) -> [f64; 3] {
        self.base_position_m
    }

    #[getter]
    fn ambiguity_ids(&self) -> Vec<String> {
        self.ambiguity_ids.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "MovingBaselineEpoch(base_position_m=[{:.3}, {:.3}, {:.3}], ambiguity_ids={})",
            self.base_position_m[0],
            self.base_position_m[1],
            self.base_position_m[2],
            self.ambiguity_ids.len()
        )
    }
}

/// One solved moving-baseline epoch.
///
/// `baseline` is the rover-minus-base ECEF baseline as a numpy `float64` array
/// of shape `(3,)`, metres: the integer-fixed baseline when `fixed` is true,
/// otherwise the float baseline.
#[pyclass(module = "sidereon._sidereon", name = "MovingBaselineEpochSolution")]
pub struct PyMovingBaselineEpochSolution {
    inner: MovingBaselineEpochSolution,
}

#[pymethods]
impl PyMovingBaselineEpochSolution {
    /// Base receiver ECEF position used for this epoch, numpy `[x, y, z]` metres.
    #[getter]
    fn base_position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.base_position_m)
    }

    #[getter]
    fn base_position_m(&self) -> [f64; 3] {
        self.inner.base_position_m
    }

    /// Baseline (rover minus base) as a numpy array `[dx, dy, dz]` metres.
    #[getter]
    fn baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.baseline_m)
    }

    #[getter]
    fn baseline_m(&self) -> [f64; 3] {
        self.inner.baseline_m
    }

    /// Euclidean baseline length, metres.
    #[getter]
    fn baseline_length_m(&self) -> f64 {
        self.inner.baseline_length_m
    }

    /// Whether the integer ambiguities were fixed this epoch.
    #[getter]
    fn fixed(&self) -> bool {
        matches!(self.inner.status, MovingBaselineStatus::Fixed)
    }

    /// Integer ambiguity-fix status this epoch.
    #[getter]
    fn integer_status(&self) -> PyIntegerStatus {
        self.inner.fixed.search.integer_status.into()
    }

    #[getter]
    fn integer_ratio(&self) -> Option<f64> {
        self.inner.fixed.search.integer_ratio
    }

    fn __repr__(&self) -> String {
        format!(
            "MovingBaselineEpochSolution(baseline=[{:.4}, {:.4}, {:.4}], baseline_length_m={:.4}, \
             fixed={})",
            self.inner.baseline_m[0],
            self.inner.baseline_m[1],
            self.inner.baseline_m[2],
            self.inner.baseline_length_m,
            matches!(self.inner.status, MovingBaselineStatus::Fixed),
        )
    }
}

/// Solve a sequence of moving-baseline RTK epochs, each against its own base
/// position.
///
/// `epochs` is a list of `MovingBaselineEpoch`. With `warm_start` enabled, each
/// solved baseline seeds the next epoch's float linearization point for
/// continuity; the first epoch always starts from `initial_baseline_m`. Raises
/// `SolveError` (tagged with the failing epoch index) on a solve failure.
#[pyfunction]
#[pyo3(signature = (
    epochs,
    model,
    float_options=None,
    fixed_options=None,
    initial_baseline_m=[0.0; 3],
    warm_start=true,
))]
fn solve_moving_baseline(
    py: Python<'_>,
    epochs: Vec<Py<PyMovingBaselineEpoch>>,
    model: &PyRtkMeasurementModel,
    float_options: Option<Py<PyRtkFloatOptions>>,
    fixed_options: Option<Py<PyRtkFixedOptions>>,
    initial_baseline_m: [f64; 3],
    warm_start: bool,
) -> PyResult<Vec<PyMovingBaselineEpochSolution>> {
    let float = option_py_or_default(
        py,
        float_options.as_ref(),
        |value| value.inner,
        || PyRtkFloatOptions::default().inner,
    );
    let fixed = option_py_or_default(
        py,
        fixed_options.as_ref(),
        |value| value.inner,
        || PyRtkFixedOptions::default().inner,
    );
    let opts = MovingBaselineOpts {
        model: model.inner,
        float,
        fixed,
        initial_baseline_m,
        warm_start,
    };

    // Hold the borrows alive for the duration of the solve so the borrowed
    // `MovingBaselineEpoch` views into each epoch's owned data stay valid.
    let borrows: Vec<PyRef<'_, PyMovingBaselineEpoch>> =
        epochs.iter().map(|epoch| epoch.borrow(py)).collect();
    let mb_epochs: Vec<MovingBaselineEpoch<'_>> = borrows
        .iter()
        .map(|epoch| MovingBaselineEpoch {
            base_position_m: epoch.base_position_m,
            epoch: &epoch.epoch,
            ambiguities: AmbiguitySet {
                ids: &epoch.ambiguity_ids,
                satellites: &epoch.ambiguity_satellites,
                scale: AmbiguityScale {
                    wavelengths_m: &epoch.wavelengths_m,
                    offsets_m: &epoch.offsets_m,
                },
                float_only_systems: &epoch.float_only_systems,
            },
        })
        .collect();

    let solutions = core_solve_moving_baseline(&mb_epochs, opts, None).map_err(to_solve_err)?;
    Ok(solutions
        .into_iter()
        .map(|inner| PyMovingBaselineEpochSolution { inner })
        .collect())
}

// --- sequential RTK arc driver ---------------------------------------------

/// One raw single-frequency code/carrier observation at a receiver, for the
/// sequential RTK arc driver.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcObservation")]
#[derive(Clone)]
pub struct PyRtkArcObservation {
    inner: RtkArcObservation,
}

#[pymethods]
impl PyRtkArcObservation {
    /// Create one raw base/rover code+carrier observation.
    ///
    /// `ambiguity_id` is the ambiguity-arc id: a clean arc uses the satellite id;
    /// a cycle-slip split carries a distinct id (e.g. `"G05#2"`) so the
    /// single-difference key resets.
    ///
    /// `lli` is the optional loss-of-lock indicator consumed only by the optional
    /// cycle-slip preprocessing (`RtkArcPreprocessing.cycle_slip`): bit 0 set marks
    /// a slip on this satellite at this epoch. `None` (default) is no-LLI and
    /// leaves the solve unchanged.
    #[new]
    #[pyo3(signature = (satellite_id, ambiguity_id, code_m, phase_m, lli=None))]
    fn new(
        satellite_id: String,
        ambiguity_id: String,
        code_m: f64,
        phase_m: f64,
        lli: Option<i64>,
    ) -> Self {
        Self {
            inner: RtkArcObservation {
                satellite_id,
                ambiguity_id,
                code_m,
                phase_m,
                lli,
            },
        }
    }

    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    #[getter]
    fn code_m(&self) -> f64 {
        self.inner.code_m
    }

    #[getter]
    fn phase_m(&self) -> f64 {
        self.inner.phase_m
    }

    #[getter]
    fn lli(&self) -> Option<i64> {
        self.inner.lli
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcObservation(satellite_id={:?}, ambiguity_id={:?})",
            self.inner.satellite_id, self.inner.ambiguity_id
        )
    }
}

/// One raw RTK arc epoch: paired base/rover observations and the satellite
/// positions needed to form double differences.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcEpoch")]
#[derive(Clone)]
pub struct PyRtkArcEpoch {
    inner: RtkArcEpoch,
}

#[pymethods]
impl PyRtkArcEpoch {
    /// Create one raw RTK arc epoch.
    ///
    /// `satellite_positions_m` are the shared receive-time satellite ECEF
    /// positions (used for the variance model and reference geometry). The
    /// per-receiver `base_satellite_positions_m` / `rover_satellite_positions_m`
    /// default to the shared map when left empty. `prediction_time_s` is the
    /// optional epoch time coordinate (seconds) used to form prediction deltas.
    #[new]
    #[pyo3(signature = (
        base,
        rover,
        satellite_positions_m,
        base_satellite_positions_m=BTreeMap::new(),
        rover_satellite_positions_m=BTreeMap::new(),
        velocity_mps=None,
        prediction_time_s=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        base: Vec<Py<PyRtkArcObservation>>,
        rover: Vec<Py<PyRtkArcObservation>>,
        satellite_positions_m: BTreeMap<String, [f64; 3]>,
        base_satellite_positions_m: BTreeMap<String, [f64; 3]>,
        rover_satellite_positions_m: BTreeMap<String, [f64; 3]>,
        velocity_mps: Option<[f64; 3]>,
        prediction_time_s: Option<f64>,
    ) -> Self {
        let base = base.iter().map(|o| o.borrow(py).inner.clone()).collect();
        let rover = rover.iter().map(|o| o.borrow(py).inner.clone()).collect();
        Self {
            inner: RtkArcEpoch {
                base,
                rover,
                satellite_positions_m,
                base_satellite_positions_m,
                rover_satellite_positions_m,
                velocity_mps,
                prediction_time_s,
            },
        }
    }

    #[getter]
    fn base_count(&self) -> usize {
        self.inner.base.len()
    }

    #[getter]
    fn rover_count(&self) -> usize {
        self.inner.rover.len()
    }

    #[getter]
    fn velocity_mps(&self) -> Option<[f64; 3]> {
        self.inner.velocity_mps
    }

    #[getter]
    fn prediction_time_s(&self) -> Option<f64> {
        self.inner.prediction_time_s
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcEpoch(base={}, rover={})",
            self.inner.base.len(),
            self.inner.rover.len()
        )
    }
}

/// Map a cycle-slip policy selector string into the core [`CycleSlipPolicy`].
fn extract_cycle_slip_policy(label: &str) -> PyResult<CycleSlipPolicy> {
    match label {
        "error" => Ok(CycleSlipPolicy::Error),
        "drop_satellite" => Ok(CycleSlipPolicy::DropSatellite),
        "split_arc" => Ok(CycleSlipPolicy::SplitArc),
        other => Err(PyValueError::new_err(format!(
            "unknown cycle-slip policy {other:?}; expected \"error\", \"drop_satellite\", or \
             \"split_arc\""
        ))),
    }
}

/// Selector string for a core [`CycleSlipPolicy`] (inverse of
/// [`extract_cycle_slip_policy`]).
fn cycle_slip_policy_label(policy: CycleSlipPolicy) -> &'static str {
    match policy {
        CycleSlipPolicy::Error => "error",
        CycleSlipPolicy::DropSatellite => "drop_satellite",
        CycleSlipPolicy::SplitArc => "split_arc",
    }
}

/// Selector string for a core [`CycleSlipReceiver`].
fn cycle_slip_receiver_label(receiver: CycleSlipReceiver) -> &'static str {
    match receiver {
        CycleSlipReceiver::Base => "base",
        CycleSlipReceiver::Rover => "rover",
    }
}

/// Map a dynamics-model selector string into the core [`DynamicsModel`].
fn extract_dynamics_model(label: &str) -> PyResult<DynamicsModel> {
    match label {
        "constant_position" => Ok(DynamicsModel::ConstantPosition),
        "velocity_propagated" => Ok(DynamicsModel::VelocityPropagated),
        other => Err(PyValueError::new_err(format!(
            "unknown dynamics model {other:?}; expected \"constant_position\" or \
             \"velocity_propagated\""
        ))),
    }
}

/// Per-epoch sequential-update controls for the RTK arc driver.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcUpdateOptions")]
#[derive(Clone)]
pub struct PyRtkArcUpdateOptions {
    inner: UpdateOpts,
}

#[pymethods]
impl PyRtkArcUpdateOptions {
    /// Create RTK arc per-epoch update controls.
    ///
    /// `dynamics` is `"constant_position"` (default) or `"velocity_propagated"`.
    /// `process_noise_baseline_sigma_m` at `0.0` is the static filter.
    #[new]
    #[pyo3(signature = (
        hold_sigma_m=1.0e-4,
        position_tol_m=POSITION_TOL_M,
        ambiguity_tol_m=AMBIGUITY_TOL_M,
        max_iterations=MAX_ITERATIONS,
        process_noise_baseline_sigma_m=0.0,
        dynamics="constant_position".to_string(),
        float_only_systems=Vec::new(),
        report_residuals=false,
        ar_arming_sigma_m=None,
        ratio_threshold=RATIO_THRESHOLD,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        hold_sigma_m: f64,
        position_tol_m: f64,
        ambiguity_tol_m: f64,
        max_iterations: usize,
        process_noise_baseline_sigma_m: f64,
        dynamics: String,
        float_only_systems: Vec<String>,
        report_residuals: bool,
        ar_arming_sigma_m: Option<f64>,
        ratio_threshold: f64,
    ) -> PyResult<Self> {
        let dynamics_model = extract_dynamics_model(&dynamics)?;
        Ok(Self {
            inner: UpdateOpts {
                hold_sigma_m,
                position_tol_m,
                ambiguity_tol_m,
                max_iterations,
                process_noise_baseline_sigma_m,
                dynamics_model,
                float_only_systems,
                report_residuals,
                receiver_antenna_corrections: None,
                ar_arming_sigma_m,
                search: SearchOpts { ratio_threshold },
            },
        })
    }

    #[getter]
    fn hold_sigma_m(&self) -> f64 {
        self.inner.hold_sigma_m
    }

    #[getter]
    fn report_residuals(&self) -> bool {
        self.inner.report_residuals
    }

    #[getter]
    fn ratio_threshold(&self) -> f64 {
        self.inner.search.ratio_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcUpdateOptions(hold_sigma_m={:.3e}, ratio_threshold={:.3})",
            self.inner.hold_sigma_m, self.inner.search.ratio_threshold
        )
    }
}

impl Default for PyRtkArcUpdateOptions {
    fn default() -> Self {
        Self::new(
            1.0e-4,
            POSITION_TOL_M,
            AMBIGUITY_TOL_M,
            MAX_ITERATIONS,
            0.0,
            "constant_position".to_string(),
            Vec::new(),
            false,
            None,
            RATIO_THRESHOLD,
        )
        .expect("constant_position is a valid dynamics model")
    }
}

/// Optional preprocessing chained ahead of the sequential RTK arc solve.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcPreprocessing")]
#[derive(Clone, Default)]
pub struct PyRtkArcPreprocessing {
    inner: RtkArcPreprocessing,
}

#[pymethods]
impl PyRtkArcPreprocessing {
    /// Create optional RTK arc preprocessing controls.
    ///
    /// Every stage is opt-in; the all-default value leaves the solve identical to
    /// the bare core solve. When set, stages run before the sequential filter in
    /// this fixed order: cycle-slip handling, then Hatch code smoothing, then
    /// elevation masking.
    ///
    /// `cycle_slip` is `"error"`, `"drop_satellite"`, or `"split_arc"` (or `None`
    /// to skip cycle-slip handling); it reads each observation's `lli`.
    /// `hatch_window_cap` is the code-smoothing window cap (`None` skips
    /// smoothing). `elevation_mask_deg` masks satellites at the base receiver below
    /// the given elevation (`None` skips masking).
    #[new]
    #[pyo3(signature = (cycle_slip=None, hatch_window_cap=None, elevation_mask_deg=None))]
    fn new(
        cycle_slip: Option<String>,
        hatch_window_cap: Option<usize>,
        elevation_mask_deg: Option<f64>,
    ) -> PyResult<Self> {
        let cycle_slip = cycle_slip
            .map(|label| extract_cycle_slip_policy(&label))
            .transpose()?;
        Ok(Self {
            inner: RtkArcPreprocessing {
                cycle_slip,
                hatch_window_cap,
                elevation_mask_deg,
            },
        })
    }

    #[getter]
    fn cycle_slip(&self) -> Option<&'static str> {
        self.inner.cycle_slip.map(cycle_slip_policy_label)
    }

    #[getter]
    fn hatch_window_cap(&self) -> Option<usize> {
        self.inner.hatch_window_cap
    }

    #[getter]
    fn elevation_mask_deg(&self) -> Option<f64> {
        self.inner.elevation_mask_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcPreprocessing(cycle_slip={:?}, hatch_window_cap={:?}, elevation_mask_deg={:?})",
            self.inner.cycle_slip.map(cycle_slip_policy_label),
            self.inner.hatch_window_cap,
            self.inner.elevation_mask_deg,
        )
    }
}

fn reference_selection(
    reference_satellite: Option<String>,
    reference_per_system: Option<BTreeMap<String, String>>,
) -> BaselineReferenceSelection {
    match (reference_per_system, reference_satellite) {
        (Some(per_system), _) => BaselineReferenceSelection::PerSystem(per_system),
        (None, Some(sat)) => BaselineReferenceSelection::Satellite(sat),
        (None, None) => BaselineReferenceSelection::Auto,
    }
}

/// Complete typed configuration for a sequential RTK arc solve.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcConfig")]
pub struct PyRtkArcConfig {
    inner: RtkArcConfig,
}

#[pymethods]
impl PyRtkArcConfig {
    /// Create a sequential RTK arc configuration.
    ///
    /// Reference selection is geometry-based per constellation by default;
    /// pass `reference_per_system` (a constellation-letter to satellite-id map)
    /// or `reference_satellite` (single-system data only) to fix it.
    #[new]
    #[pyo3(signature = (
        base,
        model,
        wavelengths_m,
        offsets_m,
        baseline_prior_sigma_m,
        ambiguity_prior_sigma_m,
        initial_baseline_m=[0.0; 3],
        update_options=None,
        reference_satellite=None,
        reference_per_system=None,
        preprocessing=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        base: [f64; 3],
        model: &PyRtkMeasurementModel,
        wavelengths_m: BTreeMap<String, f64>,
        offsets_m: BTreeMap<String, f64>,
        baseline_prior_sigma_m: f64,
        ambiguity_prior_sigma_m: f64,
        initial_baseline_m: [f64; 3],
        update_options: Option<Py<PyRtkArcUpdateOptions>>,
        reference_satellite: Option<String>,
        reference_per_system: Option<BTreeMap<String, String>>,
        preprocessing: Option<Py<PyRtkArcPreprocessing>>,
    ) -> Self {
        let reference = reference_selection(reference_satellite, reference_per_system);
        let update_opts = option_py_or_default(
            py,
            update_options.as_ref(),
            |value| value.inner.clone(),
            || PyRtkArcUpdateOptions::default().inner,
        );
        let preprocessing = option_py_or_default(
            py,
            preprocessing.as_ref(),
            |value| value.inner.clone(),
            RtkArcPreprocessing::default,
        );
        Self {
            inner: RtkArcConfig {
                base_m: base,
                reference,
                model: model.inner,
                baseline_prior_sigma_m,
                ambiguity_prior_sigma_m,
                initial_baseline_m,
                wavelengths_m,
                offsets_m,
                update_opts,
                preprocessing,
            },
        }
    }

    #[getter]
    fn base(&self) -> [f64; 3] {
        self.inner.base_m
    }

    #[getter]
    fn initial_baseline_m(&self) -> [f64; 3] {
        self.inner.initial_baseline_m
    }

    /// Optional preprocessing chained ahead of the solve.
    #[getter]
    fn preprocessing(&self) -> PyRtkArcPreprocessing {
        PyRtkArcPreprocessing {
            inner: self.inner.preprocessing.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcConfig(base=[{:.3}, {:.3}, {:.3}], ambiguities={})",
            self.inner.base_m[0],
            self.inner.base_m[1],
            self.inner.base_m[2],
            self.inner.wavelengths_m.len()
        )
    }
}

/// LAMBDA integer-search diagnostics for one RTK arc epoch.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcIntegerSearch")]
#[derive(Clone)]
pub struct PyRtkArcIntegerSearch {
    inner: IntegerSearchMeta,
}

#[pymethods]
impl PyRtkArcIntegerSearch {
    #[getter]
    fn integer_status(&self) -> PyIntegerStatus {
        self.inner.integer_status.into()
    }

    #[getter]
    fn integer_method(&self) -> &str {
        self.inner.integer_method
    }

    #[getter]
    fn integer_ratio(&self) -> Option<f64> {
        self.inner.integer_ratio
    }

    #[getter]
    fn integer_best_score(&self) -> Option<f64> {
        self.inner.integer_best_score
    }

    #[getter]
    fn integer_second_best_score(&self) -> Option<f64> {
        self.inner.integer_second_best_score
    }

    #[getter]
    fn integer_candidates(&self) -> usize {
        self.inner.integer_candidates
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcIntegerSearch(integer_status={:?}, integer_candidates={})",
            self.inner.integer_status, self.inner.integer_candidates
        )
    }
}

/// One public residual row at an RTK arc epoch's reported solution.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcResidual")]
#[derive(Clone)]
pub struct PyRtkArcResidual {
    inner: FloatResidual,
}

#[pymethods]
impl PyRtkArcResidual {
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }
    #[getter]
    fn reference_satellite_id(&self) -> &str {
        &self.inner.reference_satellite_id
    }
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }
    #[getter]
    fn code_m(&self) -> f64 {
        self.inner.code_m
    }
    #[getter]
    fn phase_m(&self) -> f64 {
        self.inner.phase_m
    }
    #[getter]
    fn code_sigma_m(&self) -> f64 {
        self.inner.code_sigma_m
    }
    #[getter]
    fn phase_sigma_m(&self) -> f64 {
        self.inner.phase_sigma_m
    }
    #[getter]
    fn code_normalized(&self) -> f64 {
        self.inner.code_normalized
    }
    #[getter]
    fn phase_normalized(&self) -> f64 {
        self.inner.phase_normalized
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcResidual(satellite_id={:?}, reference_satellite_id={:?})",
            self.inner.satellite_id, self.inner.reference_satellite_id
        )
    }
}

/// One epoch's reported baseline/ambiguity solution from the RTK arc driver.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcEpochSolution")]
#[derive(Clone)]
pub struct PyRtkArcEpochSolution {
    inner: RtkArcEpochSolution,
}

#[pymethods]
impl PyRtkArcEpochSolution {
    /// Ambiguity-conditioned reported baseline, numpy `[dx, dy, dz]` metres.
    #[getter]
    fn reported_baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.reported_baseline_m)
    }

    #[getter]
    fn reported_baseline_m(&self) -> [f64; 3] {
        self.inner.reported_baseline_m
    }

    /// Carried float (Kalman posterior) baseline, numpy `[dx, dy, dz]` metres.
    #[getter]
    fn float_baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.float_baseline_m)
    }

    #[getter]
    fn float_baseline_m(&self) -> [f64; 3] {
        self.inner.float_baseline_m
    }

    #[getter]
    fn integer_fixed(&self) -> bool {
        self.inner.integer_fixed
    }

    #[getter]
    fn integer_ratio(&self) -> f64 {
        self.inner.integer_ratio
    }

    #[getter]
    fn newly_fixed(&self) -> Vec<String> {
        self.inner.newly_fixed.clone()
    }

    #[getter]
    fn fixed_ids(&self) -> Vec<String> {
        self.inner.fixed_ids.clone()
    }

    /// Reported single-difference ambiguities as `(id, metres)` in column order.
    #[getter]
    fn sd_ambiguities_m(&self) -> Vec<(String, f64)> {
        self.inner.sd_ambiguities_m.clone()
    }

    #[getter]
    fn fixed_double_difference_ids(&self) -> Vec<String> {
        self.inner.fixed_double_difference_ids.clone()
    }

    #[getter]
    fn used_satellite_ids(&self) -> Vec<String> {
        self.inner.used_satellite_ids.clone()
    }

    /// LAMBDA search diagnostics, or `None` when no search ran this epoch.
    #[getter]
    fn search(&self) -> Option<PyRtkArcIntegerSearch> {
        self.inner
            .search
            .clone()
            .map(|inner| PyRtkArcIntegerSearch { inner })
    }

    /// Public residual rows at the reported solution (empty unless enabled).
    #[getter]
    fn residuals(&self) -> Vec<PyRtkArcResidual> {
        self.inner
            .residuals
            .iter()
            .map(|inner| PyRtkArcResidual {
                inner: inner.clone(),
            })
            .collect()
    }

    /// Geometry observability and covariance-validation diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcEpochSolution(reported_baseline=[{:.4}, {:.4}, {:.4}], integer_fixed={})",
            self.inner.reported_baseline_m[0],
            self.inner.reported_baseline_m[1],
            self.inner.reported_baseline_m[2],
            self.inner.integer_fixed
        )
    }
}

/// The carried sequential filter state after the last RTK arc epoch.
#[pyclass(module = "sidereon._sidereon", name = "RtkFilterState")]
#[derive(Clone)]
pub struct PyRtkFilterState {
    inner: FilterState,
}

#[pymethods]
impl PyRtkFilterState {
    #[getter]
    fn version(&self) -> u16 {
        self.inner.version
    }

    /// Per-constellation reference single-difference ambiguity ids.
    #[getter]
    fn references(&self) -> BTreeMap<String, String> {
        self.inner.references.clone()
    }

    /// Single-difference ambiguity ids, in information-matrix column order.
    #[getter]
    fn sd_ambiguity_ids(&self) -> Vec<String> {
        self.inner.sd_ambiguity_ids.clone()
    }

    /// Float (Kalman posterior) baseline, numpy `[dx, dy, dz]` metres.
    #[getter]
    fn baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.baseline_m)
    }

    #[getter]
    fn baseline_m(&self) -> [f64; 3] {
        self.inner.baseline_m
    }

    /// Float single-difference ambiguities (metres), parallel to the ids.
    #[getter]
    fn sd_ambiguities_m(&self) -> Vec<f64> {
        self.inner.sd_ambiguities_m.clone()
    }

    /// Row-major `n x n` information matrix, `n = 3 + sd_ambiguity_ids.len()`.
    #[getter]
    fn information(&self) -> Vec<f64> {
        self.inner.information.clone()
    }

    #[getter]
    fn ambiguity_prior_sigma_m(&self) -> f64 {
        self.inner.ambiguity_prior_sigma_m
    }

    #[getter]
    fn epoch_count(&self) -> usize {
        self.inner.epoch_count
    }

    /// Held fixed integer double-difference ambiguities (id to cycles).
    #[getter]
    fn fixed_cycles(&self) -> BTreeMap<String, i64> {
        self.inner.fixed_cycles.clone()
    }

    /// Held fixed double-difference ambiguities (id to metres).
    #[getter]
    fn fixed_m(&self) -> BTreeMap<String, f64> {
        self.inner.fixed_m.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkFilterState(sd_ambiguity_ids={}, epoch_count={})",
            self.inner.sd_ambiguity_ids.len(),
            self.inner.epoch_count
        )
    }
}

/// Split-arc metadata produced by cycle-slip preprocessing under the
/// `"split_arc"` policy.
#[pyclass(module = "sidereon._sidereon", name = "CycleSlipSplitArc")]
#[derive(Clone)]
pub struct PyCycleSlipSplitArc {
    inner: CycleSlipSplitArc,
}

#[pymethods]
impl PyCycleSlipSplitArc {
    /// Receiver side the split occurred on (`"base"` or `"rover"`).
    #[getter]
    fn receiver(&self) -> &'static str {
        cycle_slip_receiver_label(self.inner.receiver)
    }

    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    #[getter]
    fn start_epoch_index(&self) -> usize {
        self.inner.start_epoch_index
    }

    #[getter]
    fn end_epoch_index(&self) -> usize {
        self.inner.end_epoch_index
    }

    #[getter]
    fn n_epochs(&self) -> usize {
        self.inner.n_epochs
    }

    fn __repr__(&self) -> String {
        format!(
            "CycleSlipSplitArc(satellite_id={:?}, ambiguity_id={:?}, start_epoch_index={}, \
             end_epoch_index={})",
            self.inner.satellite_id,
            self.inner.ambiguity_id,
            self.inner.start_epoch_index,
            self.inner.end_epoch_index,
        )
    }
}

/// Full sequential RTK arc solution.
#[pyclass(module = "sidereon._sidereon", name = "RtkArcSolution")]
pub struct PyRtkArcSolution {
    inner: RtkArcSolution,
}

#[pymethods]
impl PyRtkArcSolution {
    /// Per-constellation reference single-difference ambiguity ids.
    #[getter]
    fn references(&self) -> BTreeMap<String, String> {
        self.inner.references.clone()
    }

    /// Per-epoch reported solutions, in input order.
    #[getter]
    fn epochs(&self) -> Vec<PyRtkArcEpochSolution> {
        self.inner
            .epochs
            .iter()
            .map(|inner| PyRtkArcEpochSolution {
                inner: inner.clone(),
            })
            .collect()
    }

    /// Final carried filter state after the last epoch.
    #[getter]
    fn final_state(&self) -> PyRtkFilterState {
        PyRtkFilterState {
            inner: self.inner.final_state.clone(),
        }
    }

    /// Satellites dropped during cycle-slip preprocessing under the
    /// `"drop_satellite"` policy, sorted. Empty when cycle-slip preprocessing is
    /// disabled or no slip occurred.
    #[getter]
    fn dropped_sats(&self) -> Vec<String> {
        self.inner.dropped_sats.clone()
    }

    /// Split-arc metadata produced by cycle-slip preprocessing under the
    /// `"split_arc"` policy. Empty otherwise.
    #[getter]
    fn split_cycle_slip_arcs(&self) -> Vec<PyCycleSlipSplitArc> {
        self.inner
            .split_cycle_slip_arcs
            .iter()
            .map(|inner| PyCycleSlipSplitArc {
                inner: inner.clone(),
            })
            .collect()
    }

    /// Satellites masked below the elevation mask in any epoch, sorted. Empty when
    /// elevation masking is disabled.
    #[getter]
    fn elevation_masked_sats(&self) -> Vec<String> {
        self.inner.elevation_masked_sats.clone()
    }

    /// Posterior measurement covariance (row-major `n x n`, metres squared): the
    /// inverse of `final_state`'s information matrix. Empty only if that inversion
    /// is singular.
    #[getter]
    fn measurement_covariance(&self) -> Vec<f64> {
        self.inner.measurement_covariance.clone()
    }

    /// Posterior measurement covariance as a square numpy matrix.
    #[getter]
    fn measurement_covariance_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        flat_square_to_array(py, &self.inner.measurement_covariance)
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkArcSolution(references={}, epochs={})",
            self.inner.references.len(),
            self.inner.epochs.len()
        )
    }
}

/// Solve a sequential RTK baseline arc from raw rover+base epochs.
///
/// `epochs` is a list of `RtkArcEpoch`; `config` is an `RtkArcConfig`. Reference
/// satellites are selected once for the whole arc; an epoch whose update fails
/// aborts the arc at that epoch. Returns an `RtkArcSolution`. Raises `SolveError`
/// on an empty arc, too few satellites, reference-selection failure, or a
/// per-epoch update failure.
#[pyfunction]
#[pyo3(signature = (epochs, config))]
fn solve_rtk_arc(
    py: Python<'_>,
    epochs: Vec<Py<PyRtkArcEpoch>>,
    config: &PyRtkArcConfig,
) -> PyResult<PyRtkArcSolution> {
    let epochs: Vec<RtkArcEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).inner.clone())
        .collect();
    let inner = core_solve_rtk_arc(&epochs, &config.inner).map_err(to_solve_err)?;
    Ok(PyRtkArcSolution { inner })
}

/// Complete typed configuration for a static RTK arc solve.
#[pyclass(module = "sidereon._sidereon", name = "RtkStaticArcConfig")]
pub struct PyRtkStaticArcConfig {
    inner: RtkStaticArcConfig,
}

#[pymethods]
impl PyRtkStaticArcConfig {
    /// Create a static RTK arc configuration.
    #[new]
    #[pyo3(signature = (
        arc,
        float_options=None,
        fixed_options=None,
        residual_options=None,
    ))]
    fn new(
        py: Python<'_>,
        arc: &PyRtkArcConfig,
        float_options: Option<Py<PyRtkFloatOptions>>,
        fixed_options: Option<Py<PyRtkFixedOptions>>,
        residual_options: Option<Py<PyRtkResidualValidationOptions>>,
    ) -> Self {
        let opts = validated_fixed_opts_from_py(
            py,
            float_options.as_ref(),
            fixed_options.as_ref(),
            residual_options.as_ref(),
        );
        Self {
            inner: RtkStaticArcConfig {
                arc: arc.inner.clone(),
                opts,
            },
        }
    }

    #[getter]
    fn base(&self) -> [f64; 3] {
        self.inner.arc.base_m
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkStaticArcConfig(base=[{:.3}, {:.3}, {:.3}])",
            self.inner.arc.base_m[0], self.inner.arc.base_m[1], self.inner.arc.base_m[2]
        )
    }
}

/// Static arc float and fixed RTK solution.
#[pyclass(module = "sidereon._sidereon", name = "RtkStaticArcSolution")]
pub struct PyRtkStaticArcSolution {
    inner: RtkStaticArcSolution,
}

#[pymethods]
impl PyRtkStaticArcSolution {
    #[getter]
    fn references(&self) -> BTreeMap<String, String> {
        self.inner.references.clone()
    }

    #[getter]
    fn ambiguity_ids(&self) -> Vec<String> {
        self.inner.ambiguity_ids.clone()
    }

    #[getter]
    fn ambiguity_satellites(&self) -> BTreeMap<String, String> {
        self.inner.ambiguity_satellites.clone()
    }

    #[getter]
    fn float_solution(&self) -> PyRtkFloatSolution {
        PyRtkFloatSolution {
            inner: self.inner.float_solution.clone(),
        }
    }

    #[getter]
    fn fixed_solution(&self) -> PyRtkFixedSolution {
        PyRtkFixedSolution {
            inner: self.inner.fixed_solution.clone(),
        }
    }

    #[getter]
    fn dropped_sats(&self) -> Vec<String> {
        self.inner.dropped_sats.clone()
    }

    #[getter]
    fn split_cycle_slip_arcs(&self) -> Vec<PyCycleSlipSplitArc> {
        self.inner
            .split_cycle_slip_arcs
            .iter()
            .map(|inner| PyCycleSlipSplitArc {
                inner: inner.clone(),
            })
            .collect()
    }

    #[getter]
    fn elevation_masked_sats(&self) -> Vec<String> {
        self.inner.elevation_masked_sats.clone()
    }

    /// Geometry observability and covariance-validation diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkStaticArcSolution(references={}, ambiguity_ids={})",
            self.inner.references.len(),
            self.inner.ambiguity_ids.len()
        )
    }
}

/// Solve one static RTK baseline over a raw rover+base arc.
#[pyfunction]
#[pyo3(signature = (epochs, config))]
fn solve_static_rtk_arc(
    py: Python<'_>,
    epochs: Vec<Py<PyRtkArcEpoch>>,
    config: &PyRtkStaticArcConfig,
) -> PyResult<PyRtkStaticArcSolution> {
    let epochs: Vec<RtkArcEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).inner.clone())
        .collect();
    let inner = core_solve_static_rtk_arc(&epochs, &config.inner).map_err(to_solve_err)?;
    Ok(PyRtkStaticArcSolution { inner })
}

/// One single-frequency RINEX code/carrier pair used to build RTK arc records.
#[pyclass(module = "sidereon._sidereon", name = "RtkRinexSignalPair")]
#[derive(Clone)]
pub struct PyRtkRinexSignalPair {
    inner: RtkRinexSignalPair,
}

#[pymethods]
impl PyRtkRinexSignalPair {
    /// Create a RINEX signal pair for one constellation.
    #[new]
    fn new(system: PyGnssSystem, code_observable: String, phase_observable: String) -> Self {
        Self {
            inner: RtkRinexSignalPair {
                system: system.into(),
                code_observable,
                phase_observable,
            },
        }
    }

    /// GPS `C1C` plus `L1C`.
    #[staticmethod]
    fn gps_l1_c() -> Self {
        Self {
            inner: RtkRinexSignalPair::gps_l1_c(),
        }
    }

    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    #[getter]
    fn code_observable(&self) -> &str {
        &self.inner.code_observable
    }

    #[getter]
    fn phase_observable(&self) -> &str {
        &self.inner.phase_observable
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkRinexSignalPair(code_observable={:?}, phase_observable={:?})",
            self.inner.code_observable, self.inner.phase_observable
        )
    }
}

/// Options for building single-frequency RTK arc records from RINEX.
#[pyclass(module = "sidereon._sidereon", name = "RtkRinexArcOptions")]
#[derive(Clone)]
pub struct PyRtkRinexArcOptions {
    inner: CoreRtkRinexArcOptions,
}

#[pymethods]
impl PyRtkRinexArcOptions {
    /// Create RINEX RTK arc build options.
    #[new]
    #[pyo3(signature = (
        signal_pairs=None,
        max_epochs=None,
        min_common_satellites=4,
        include_prediction_time=true,
    ))]
    fn new(
        py: Python<'_>,
        signal_pairs: Option<Vec<Py<PyRtkRinexSignalPair>>>,
        max_epochs: Option<usize>,
        min_common_satellites: usize,
        include_prediction_time: bool,
    ) -> Self {
        let signal_pairs = signal_pairs
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|pair| pair.borrow(py).inner.clone())
                    .collect()
            })
            .unwrap_or_else(|| CoreRtkRinexArcOptions::gps_l1_c().signal_pairs);
        Self {
            inner: CoreRtkRinexArcOptions {
                signal_pairs,
                max_epochs,
                min_common_satellites,
                include_prediction_time,
            },
        }
    }

    /// GPS `C1C` plus `L1C` defaults.
    #[staticmethod]
    fn gps_l1_c() -> Self {
        Self {
            inner: CoreRtkRinexArcOptions::gps_l1_c(),
        }
    }

    #[getter]
    fn signal_pairs(&self) -> Vec<PyRtkRinexSignalPair> {
        self.inner
            .signal_pairs
            .iter()
            .cloned()
            .map(|inner| PyRtkRinexSignalPair { inner })
            .collect()
    }

    #[getter]
    fn max_epochs(&self) -> Option<usize> {
        self.inner.max_epochs
    }

    #[getter]
    fn min_common_satellites(&self) -> usize {
        self.inner.min_common_satellites
    }

    #[getter]
    fn include_prediction_time(&self) -> bool {
        self.inner.include_prediction_time
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkRinexArcOptions(signal_pairs={}, max_epochs={:?})",
            self.inner.signal_pairs.len(),
            self.inner.max_epochs
        )
    }
}

/// Single-frequency RTK arc records built from RINEX.
#[pyclass(module = "sidereon._sidereon", name = "RinexRtkArc")]
pub struct PyRinexRtkArc {
    inner: CoreRtkRinexArc,
}

#[pymethods]
impl PyRinexRtkArc {
    #[getter]
    fn epochs(&self) -> Vec<PyRtkArcEpoch> {
        self.inner
            .epochs
            .iter()
            .cloned()
            .map(|inner| PyRtkArcEpoch { inner })
            .collect()
    }

    #[getter]
    fn wavelengths_m(&self) -> BTreeMap<String, f64> {
        self.inner.wavelengths_m.clone()
    }

    #[getter]
    fn offsets_m(&self) -> BTreeMap<String, f64> {
        self.inner.offsets_m.clone()
    }

    #[getter]
    fn skipped_epoch_count(&self) -> usize {
        self.inner.skipped_epoch_count
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexRtkArc(epochs={}, ambiguities={})",
            self.inner.epochs.len(),
            self.inner.wavelengths_m.len()
        )
    }
}

/// One dual-frequency RINEX signal selection for one constellation.
#[pyclass(module = "sidereon._sidereon", name = "RtkRinexDualSignalPair")]
#[derive(Clone)]
pub struct PyRtkRinexDualSignalPair {
    inner: RtkRinexDualSignalPair,
}

#[pymethods]
impl PyRtkRinexDualSignalPair {
    /// Create a dual-frequency RINEX signal selection.
    #[new]
    #[pyo3(signature = (
        system,
        code1_observable,
        phase1_observable,
        code2_observable,
        phase2_observable,
    ))]
    fn new(
        system: PyGnssSystem,
        code1_observable: String,
        phase1_observable: String,
        code2_observable: String,
        phase2_observable: String,
    ) -> Self {
        Self {
            inner: RtkRinexDualSignalPair {
                system: system.into(),
                code1_observable,
                phase1_observable,
                code2_observable,
                phase2_observable,
            },
        }
    }

    /// GPS `C1C`/`L1C` plus `C2W`/`L2W`.
    #[staticmethod]
    fn gps_l1_l2_cw() -> Self {
        Self {
            inner: RtkRinexDualSignalPair::gps_l1_l2_cw(),
        }
    }

    #[getter]
    fn system(&self) -> PyGnssSystem {
        self.inner.system.into()
    }

    #[getter]
    fn code1_observable(&self) -> &str {
        &self.inner.code1_observable
    }

    #[getter]
    fn phase1_observable(&self) -> &str {
        &self.inner.phase1_observable
    }

    #[getter]
    fn code2_observable(&self) -> &str {
        &self.inner.code2_observable
    }

    #[getter]
    fn phase2_observable(&self) -> &str {
        &self.inner.phase2_observable
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkRinexDualSignalPair(code1_observable={:?}, code2_observable={:?})",
            self.inner.code1_observable, self.inner.code2_observable
        )
    }
}

/// Options for building dual-frequency RTK arc records from RINEX.
#[pyclass(module = "sidereon._sidereon", name = "RtkRinexDualArcOptions")]
#[derive(Clone)]
pub struct PyRtkRinexDualArcOptions {
    inner: RtkRinexDualArcOptions,
}

#[pymethods]
impl PyRtkRinexDualArcOptions {
    /// Create dual-frequency RINEX RTK arc build options.
    #[new]
    #[pyo3(signature = (
        signal_pairs=None,
        max_epochs=None,
        min_common_satellites=4,
        include_prediction_time=true,
    ))]
    fn new(
        py: Python<'_>,
        signal_pairs: Option<Vec<Py<PyRtkRinexDualSignalPair>>>,
        max_epochs: Option<usize>,
        min_common_satellites: usize,
        include_prediction_time: bool,
    ) -> Self {
        let signal_pairs = signal_pairs
            .map(|pairs| {
                pairs
                    .iter()
                    .map(|pair| pair.borrow(py).inner.clone())
                    .collect()
            })
            .unwrap_or_else(|| RtkRinexDualArcOptions::gps_l1_l2_cw().signal_pairs);
        Self {
            inner: RtkRinexDualArcOptions {
                signal_pairs,
                max_epochs,
                min_common_satellites,
                include_prediction_time,
            },
        }
    }

    /// GPS `C1C`/`L1C` plus `C2W`/`L2W` defaults.
    #[staticmethod]
    fn gps_l1_l2_cw() -> Self {
        Self {
            inner: RtkRinexDualArcOptions::gps_l1_l2_cw(),
        }
    }

    #[getter]
    fn signal_pairs(&self) -> Vec<PyRtkRinexDualSignalPair> {
        self.inner
            .signal_pairs
            .iter()
            .cloned()
            .map(|inner| PyRtkRinexDualSignalPair { inner })
            .collect()
    }

    #[getter]
    fn max_epochs(&self) -> Option<usize> {
        self.inner.max_epochs
    }

    #[getter]
    fn min_common_satellites(&self) -> usize {
        self.inner.min_common_satellites
    }

    #[getter]
    fn include_prediction_time(&self) -> bool {
        self.inner.include_prediction_time
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkRinexDualArcOptions(signal_pairs={}, max_epochs={:?})",
            self.inner.signal_pairs.len(),
            self.inner.max_epochs
        )
    }
}

/// Dual-frequency RTK arc records built from RINEX.
#[pyclass(module = "sidereon._sidereon", name = "RinexDualFrequencyRtkArc")]
pub struct PyRinexDualFrequencyRtkArc {
    inner: CoreRtkRinexDualFrequencyArc,
}

#[pymethods]
impl PyRinexDualFrequencyRtkArc {
    #[getter]
    fn epochs(&self) -> Vec<PyRtkDualFrequencyArcEpoch> {
        self.inner
            .epochs
            .iter()
            .cloned()
            .map(|inner| PyRtkDualFrequencyArcEpoch { inner })
            .collect()
    }

    #[getter]
    fn skipped_epoch_count(&self) -> usize {
        self.inner.skipped_epoch_count
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexDualFrequencyRtkArc(epochs={})",
            self.inner.epochs.len()
        )
    }
}

fn rinex_arc_options_from_py(
    py: Python<'_>,
    options: Option<Py<PyRtkRinexArcOptions>>,
) -> CoreRtkRinexArcOptions {
    option_py_or_default(
        py,
        options.as_ref(),
        |value| value.inner.clone(),
        CoreRtkRinexArcOptions::gps_l1_c,
    )
}

fn rinex_dual_arc_options_from_py(
    py: Python<'_>,
    options: Option<Py<PyRtkRinexDualArcOptions>>,
) -> RtkRinexDualArcOptions {
    option_py_or_default(
        py,
        options.as_ref(),
        |value| value.inner.clone(),
        RtkRinexDualArcOptions::gps_l1_l2_cw,
    )
}

fn model_from_optional(py: Python<'_>, model: Option<Py<PyRtkMeasurementModel>>) -> MeasModel {
    option_py_or_default(py, model.as_ref(), |value| value.inner, default_rtk_model)
}

fn arc_update_options_from_optional(
    py: Python<'_>,
    update_options: Option<Py<PyRtkArcUpdateOptions>>,
) -> UpdateOpts {
    option_py_or_default(
        py,
        update_options.as_ref(),
        |value| value.inner.clone(),
        || PyRtkArcUpdateOptions::default().inner,
    )
}

fn preprocessing_from_optional(
    py: Python<'_>,
    preprocessing: Option<Py<PyRtkArcPreprocessing>>,
) -> RtkArcPreprocessing {
    option_py_or_default(
        py,
        preprocessing.as_ref(),
        |value| value.inner.clone(),
        RtkArcPreprocessing::default,
    )
}

#[allow(clippy::too_many_arguments)]
fn static_arc_config_from_parts(
    py: Python<'_>,
    base: [f64; 3],
    model: MeasModel,
    wavelengths_m: BTreeMap<String, f64>,
    offsets_m: BTreeMap<String, f64>,
    initial_baseline_m: [f64; 3],
    baseline_prior_sigma_m: f64,
    ambiguity_prior_sigma_m: f64,
    update_options: Option<Py<PyRtkArcUpdateOptions>>,
    preprocessing: Option<Py<PyRtkArcPreprocessing>>,
    float_options: Option<Py<PyRtkFloatOptions>>,
    fixed_options: Option<Py<PyRtkFixedOptions>>,
    residual_options: Option<Py<PyRtkResidualValidationOptions>>,
) -> RtkStaticArcConfig {
    RtkStaticArcConfig {
        arc: RtkArcConfig {
            base_m: base,
            reference: BaselineReferenceSelection::Auto,
            model,
            baseline_prior_sigma_m,
            ambiguity_prior_sigma_m,
            initial_baseline_m,
            wavelengths_m,
            offsets_m,
            update_opts: arc_update_options_from_optional(py, update_options),
            preprocessing: preprocessing_from_optional(py, preprocessing),
        },
        opts: validated_fixed_opts_from_py(
            py,
            float_options.as_ref(),
            fixed_options.as_ref(),
            residual_options.as_ref(),
        ),
    }
}

/// Build single-frequency RTK arc records from parsed RINEX OBS products.
#[pyfunction]
#[pyo3(signature = (ephemeris, base_obs, rover_obs, options=None))]
fn build_rinex_rtk_arc(
    py: Python<'_>,
    ephemeris: &Bound<'_, PyAny>,
    base_obs: &PyRinexObs,
    rover_obs: &PyRinexObs,
    options: Option<Py<PyRtkRinexArcOptions>>,
) -> PyResult<PyRinexRtkArc> {
    let options = rinex_arc_options_from_py(py, options);
    with_observable_source(ephemeris, |source| {
        core_build_rinex_rtk_arc(source, base_obs.inner(), rover_obs.inner(), &options)
            .map(|inner| PyRinexRtkArc { inner })
            .map_err(to_solve_err)
    })
}

/// Build dual-frequency RTK arc records from parsed RINEX OBS products.
#[pyfunction]
#[pyo3(signature = (ephemeris, base_obs, rover_obs, options=None))]
fn build_dual_frequency_rinex_rtk_arc(
    py: Python<'_>,
    ephemeris: &Bound<'_, PyAny>,
    base_obs: &PyRinexObs,
    rover_obs: &PyRinexObs,
    options: Option<Py<PyRtkRinexDualArcOptions>>,
) -> PyResult<PyRinexDualFrequencyRtkArc> {
    let options = rinex_dual_arc_options_from_py(py, options);
    with_observable_source(ephemeris, |source| {
        core_build_dual_frequency_rinex_rtk_arc(
            source,
            base_obs.inner(),
            rover_obs.inner(),
            &options,
        )
        .map(|inner| PyRinexDualFrequencyRtkArc { inner })
        .map_err(to_solve_err)
    })
}

/// Solve a static single-frequency RTK baseline directly from RINEX OBS.
#[pyfunction]
#[pyo3(signature = (
    ephemeris,
    base_obs,
    rover_obs,
    base,
    model=None,
    arc_options=None,
    preprocessing=None,
    update_options=None,
    float_options=None,
    fixed_options=None,
    residual_options=None,
    initial_baseline_m=[0.0; 3],
    baseline_prior_sigma_m=30.0,
    ambiguity_prior_sigma_m=30.0,
))]
#[allow(clippy::too_many_arguments)]
fn solve_static_rinex_rtk_baseline(
    py: Python<'_>,
    ephemeris: &Bound<'_, PyAny>,
    base_obs: &PyRinexObs,
    rover_obs: &PyRinexObs,
    base: [f64; 3],
    model: Option<Py<PyRtkMeasurementModel>>,
    arc_options: Option<Py<PyRtkRinexArcOptions>>,
    preprocessing: Option<Py<PyRtkArcPreprocessing>>,
    update_options: Option<Py<PyRtkArcUpdateOptions>>,
    float_options: Option<Py<PyRtkFloatOptions>>,
    fixed_options: Option<Py<PyRtkFixedOptions>>,
    residual_options: Option<Py<PyRtkResidualValidationOptions>>,
    initial_baseline_m: [f64; 3],
    baseline_prior_sigma_m: f64,
    ambiguity_prior_sigma_m: f64,
) -> PyResult<PyRtkStaticArcSolution> {
    let arc_options = rinex_arc_options_from_py(py, arc_options);
    let model = model_from_optional(py, model);
    with_observable_source(ephemeris, |source| {
        let arc =
            core_build_rinex_rtk_arc(source, base_obs.inner(), rover_obs.inner(), &arc_options)
                .map_err(to_solve_err)?;
        let config = static_arc_config_from_parts(
            py,
            base,
            model,
            arc.wavelengths_m.clone(),
            arc.offsets_m.clone(),
            initial_baseline_m,
            baseline_prior_sigma_m,
            ambiguity_prior_sigma_m,
            update_options,
            preprocessing,
            float_options,
            fixed_options,
            residual_options,
        );
        core_solve_static_rtk_arc(&arc.epochs, &config)
            .map(|inner| PyRtkStaticArcSolution { inner })
            .map_err(to_solve_err)
    })
}

fn reference_station_mode_label(mode: StaticReferenceStationMode) -> &'static str {
    match mode {
        StaticReferenceStationMode::CodeDgnss => "code_dgnss",
        StaticReferenceStationMode::CarrierFloat => "carrier_float",
        StaticReferenceStationMode::CarrierFixed => "carrier_fixed",
    }
}

fn reference_station_fix_status_label(status: StaticReferenceFixStatus) -> &'static str {
    match status {
        StaticReferenceFixStatus::CodeDgnss => "code_dgnss",
        StaticReferenceFixStatus::CarrierFloat => "carrier_float",
        StaticReferenceFixStatus::CarrierFixed => "carrier_fixed",
    }
}

fn reference_mode_status_label(status: StaticReferenceModeStatus) -> &'static str {
    match status {
        StaticReferenceModeStatus::Solved => "solved",
        StaticReferenceModeStatus::Failed => "failed",
    }
}

fn reference_mode_error_kind(error: &StaticReferenceModeError) -> &'static str {
    match error {
        StaticReferenceModeError::RinexAssembly { .. } => "rinex_assembly",
        StaticReferenceModeError::NoMatchedCodeEpochs => "no_matched_code_epochs",
        StaticReferenceModeError::CodeDgnss { .. } => "code_dgnss",
        StaticReferenceModeError::StaticSolve { .. } => "static_solve",
        StaticReferenceModeError::CarrierArc { .. } => "carrier_arc",
        StaticReferenceModeError::CarrierSolve { .. } => "carrier_solve",
        StaticReferenceModeError::Frame { .. } => "frame",
        StaticReferenceModeError::CorrectedObservation { .. } => "corrected_observation",
        StaticReferenceModeError::InvalidCorrectedSatelliteId { .. } => {
            "invalid_corrected_satellite_id"
        }
    }
}

/// Failure detail for one attempted static reference-station mode.
#[pyclass(module = "sidereon._sidereon", name = "StaticReferenceModeError")]
#[derive(Clone)]
pub struct PyStaticReferenceModeError {
    inner: StaticReferenceModeError,
}

#[pymethods]
impl PyStaticReferenceModeError {
    /// Stable variant label.
    #[getter]
    fn kind(&self) -> &'static str {
        reference_mode_error_kind(&self.inner)
    }

    /// Human-readable failure text.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// RINEX side when `kind == "rinex_assembly"`.
    #[getter]
    fn side(&self) -> Option<&'static str> {
        match &self.inner {
            StaticReferenceModeError::RinexAssembly { side, .. } => Some(*side),
            _ => None,
        }
    }

    /// Conversion field when `kind == "frame"`.
    #[getter]
    fn field(&self) -> Option<&'static str> {
        match &self.inner {
            StaticReferenceModeError::Frame { field, .. } => Some(*field),
            _ => None,
        }
    }

    /// Source failure text when the core variant carries one.
    #[getter]
    fn reason(&self) -> Option<String> {
        match &self.inner {
            StaticReferenceModeError::RinexAssembly { reason, .. }
            | StaticReferenceModeError::CodeDgnss { reason }
            | StaticReferenceModeError::StaticSolve { reason }
            | StaticReferenceModeError::CarrierArc { reason }
            | StaticReferenceModeError::CarrierSolve { reason }
            | StaticReferenceModeError::Frame { reason, .. }
            | StaticReferenceModeError::CorrectedObservation { reason } => Some(reason.clone()),
            StaticReferenceModeError::NoMatchedCodeEpochs
            | StaticReferenceModeError::InvalidCorrectedSatelliteId { .. } => None,
        }
    }

    /// Invalid satellite id when `kind == "invalid_corrected_satellite_id"`.
    #[getter]
    fn satellite_id(&self) -> Option<String> {
        match &self.inner {
            StaticReferenceModeError::InvalidCorrectedSatelliteId { satellite_id } => {
                Some(satellite_id.clone())
            }
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticReferenceModeError(kind={}, message={:?})",
            reference_mode_error_kind(&self.inner),
            self.inner.to_string()
        )
    }
}

/// Position covariance for a static reference-station coordinate.
#[pyclass(
    module = "sidereon._sidereon",
    name = "StaticReferenceStationCovariance"
)]
#[derive(Clone)]
pub struct PyStaticReferenceStationCovariance {
    inner: StaticReferenceStationCovariance,
}

#[pymethods]
impl PyStaticReferenceStationCovariance {
    /// ECEF covariance matrix, metres squared.
    #[getter]
    fn position_ecef<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_ecef_m2)
    }

    /// Local ENU covariance matrix, metres squared.
    #[getter]
    fn position_enu<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_enu_m2)
    }

    fn __repr__(&self) -> String {
        "StaticReferenceStationCovariance()".to_string()
    }
}

/// Per-epoch diagnostic row from the selected static station mode.
#[pyclass(module = "sidereon._sidereon", name = "StaticReferenceEpochDiagnostic")]
#[derive(Clone)]
pub struct PyStaticReferenceEpochDiagnostic {
    inner: StaticReferenceEpochDiagnostic,
}

#[pymethods]
impl PyStaticReferenceEpochDiagnostic {
    #[getter]
    fn mode(&self) -> &'static str {
        reference_station_mode_label(self.inner.mode)
    }

    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    #[getter]
    fn used_satellites(&self) -> Vec<String> {
        self.inner.used_satellites.clone()
    }

    #[getter]
    fn rejected_satellite_count(&self) -> usize {
        self.inner.rejected_satellite_count
    }

    #[getter]
    fn code_residual_rms_m(&self) -> Option<f64> {
        self.inner.code_residual_rms_m
    }

    #[getter]
    fn phase_residual_rms_m(&self) -> Option<f64> {
        self.inner.phase_residual_rms_m
    }

    #[getter]
    fn residual_rms_m(&self) -> Option<f64> {
        self.inner.residual_rms_m
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticReferenceEpochDiagnostic(mode={}, epoch_index={}, used={})",
            reference_station_mode_label(self.inner.mode),
            self.inner.epoch_index,
            self.inner.used_satellites.len()
        )
    }
}

/// Per-mode attempt report for the static reference-station wrapper.
#[pyclass(module = "sidereon._sidereon", name = "StaticReferenceModeReport")]
#[derive(Clone)]
pub struct PyStaticReferenceModeReport {
    inner: StaticReferenceModeReport,
}

#[pymethods]
impl PyStaticReferenceModeReport {
    #[getter]
    fn mode(&self) -> &'static str {
        reference_station_mode_label(self.inner.mode)
    }

    #[getter]
    fn status(&self) -> &'static str {
        reference_mode_status_label(self.inner.status)
    }

    #[getter]
    fn used_epochs(&self) -> usize {
        self.inner.used_epochs
    }

    #[getter]
    fn skipped_epochs(&self) -> usize {
        self.inner.skipped_epochs
    }

    #[getter]
    fn used_measurements(&self) -> usize {
        self.inner.used_measurements
    }

    #[getter]
    fn error(&self) -> Option<PyStaticReferenceModeError> {
        self.inner
            .error
            .clone()
            .map(|inner| PyStaticReferenceModeError { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticReferenceModeReport(mode={}, status={})",
            reference_station_mode_label(self.inner.mode),
            reference_mode_status_label(self.inner.status)
        )
    }
}

/// Code-DGNSS detail from a static reference-station solve.
#[pyclass(module = "sidereon._sidereon", name = "StaticReferenceCodeSolution")]
#[derive(Clone)]
pub struct PyStaticReferenceCodeSolution {
    inner: StaticReferenceCodeSolution,
}

#[pymethods]
impl PyStaticReferenceCodeSolution {
    /// ECEF coordinate as a numpy array `[x, y, z]` metres.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = self.inner.position;
        np_array(py, &[p.x_m, p.y_m, p.z_m])
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position.as_array()
    }

    #[getter]
    fn covariance(&self) -> PyStaticReferenceStationCovariance {
        PyStaticReferenceStationCovariance {
            inner: self.inner.covariance,
        }
    }

    /// Rover-minus-reference baseline vector as a numpy array, metres.
    #[getter]
    fn baseline_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.baseline_vector_m)
    }

    #[getter]
    fn baseline_vector_m(&self) -> [f64; 3] {
        self.inner.baseline_vector_m
    }

    #[getter]
    fn baseline_m(&self) -> f64 {
        self.inner.baseline_m
    }

    #[getter]
    fn diagnostics(&self) -> Vec<PyStaticReferenceEpochDiagnostic> {
        self.inner
            .diagnostics
            .iter()
            .cloned()
            .map(|inner| PyStaticReferenceEpochDiagnostic { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticReferenceCodeSolution(baseline_m={:.3}, epochs={})",
            self.inner.baseline_m,
            self.inner.diagnostics.len()
        )
    }
}

/// Carrier RTK detail from a static reference-station solve.
#[pyclass(module = "sidereon._sidereon", name = "StaticReferenceCarrierSolution")]
#[derive(Clone)]
pub struct PyStaticReferenceCarrierSolution {
    inner: StaticReferenceCarrierSolution,
}

#[pymethods]
impl PyStaticReferenceCarrierSolution {
    /// ECEF coordinate as a numpy array `[x, y, z]` metres.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = self.inner.position;
        np_array(py, &[p.x_m, p.y_m, p.z_m])
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position.as_array()
    }

    #[getter]
    fn covariance(&self) -> PyStaticReferenceStationCovariance {
        PyStaticReferenceStationCovariance {
            inner: self.inner.covariance,
        }
    }

    /// Selected rover-minus-reference baseline vector as a numpy array, metres.
    #[getter]
    fn baseline_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.baseline_vector_m)
    }

    #[getter]
    fn baseline_vector_m(&self) -> [f64; 3] {
        self.inner.baseline_vector_m
    }

    #[getter]
    fn baseline_m(&self) -> f64 {
        self.inner.baseline_m
    }

    #[getter]
    fn integer_status(&self) -> PyIntegerStatus {
        self.inner.integer_status.into()
    }

    #[getter]
    fn integer_ratio(&self) -> Option<f64> {
        self.inner.integer_ratio
    }

    #[getter]
    fn rtk_solution(&self) -> PyRtkStaticArcSolution {
        PyRtkStaticArcSolution {
            inner: self.inner.rtk_solution.clone(),
        }
    }

    #[getter]
    fn diagnostics(&self) -> Vec<PyStaticReferenceEpochDiagnostic> {
        self.inner
            .diagnostics
            .iter()
            .cloned()
            .map(|inner| PyStaticReferenceEpochDiagnostic { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticReferenceCarrierSolution(baseline_m={:.4}, integer_status={})",
            self.inner.baseline_m,
            self.integer_status().__repr__()
        )
    }
}

/// Static reference-station coordinate with covariance and mode diagnostics.
#[pyclass(module = "sidereon._sidereon", name = "StaticReferenceStationSolution")]
pub struct PyStaticReferenceStationSolution {
    inner: StaticReferenceStationSolution,
}

#[pymethods]
impl PyStaticReferenceStationSolution {
    #[getter]
    fn mode(&self) -> &'static str {
        reference_station_mode_label(self.inner.mode)
    }

    #[getter]
    fn fix_status(&self) -> &'static str {
        reference_station_fix_status_label(self.inner.fix_status)
    }

    /// ECEF coordinate as a numpy array `[x, y, z]` metres.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = self.inner.position;
        np_array(py, &[p.x_m, p.y_m, p.z_m])
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position.as_array()
    }

    #[getter]
    fn covariance(&self) -> PyStaticReferenceStationCovariance {
        PyStaticReferenceStationCovariance {
            inner: self.inner.covariance,
        }
    }

    /// Selected rover-minus-reference baseline vector as a numpy array, metres.
    #[getter]
    fn baseline_vector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.baseline_vector_m)
    }

    #[getter]
    fn baseline_vector_m(&self) -> [f64; 3] {
        self.inner.baseline_vector_m
    }

    #[getter]
    fn baseline_m(&self) -> f64 {
        self.inner.baseline_m
    }

    #[getter]
    fn code_solution(&self) -> Option<PyStaticReferenceCodeSolution> {
        self.inner
            .code_solution
            .clone()
            .map(|inner| PyStaticReferenceCodeSolution { inner })
    }

    #[getter]
    fn carrier_solution(&self) -> Option<PyStaticReferenceCarrierSolution> {
        self.inner
            .carrier_solution
            .clone()
            .map(|inner| PyStaticReferenceCarrierSolution { inner })
    }

    #[getter]
    fn mode_reports(&self) -> Vec<PyStaticReferenceModeReport> {
        self.inner
            .mode_reports
            .iter()
            .cloned()
            .map(|inner| PyStaticReferenceModeReport { inner })
            .collect()
    }

    #[getter]
    fn diagnostics(&self) -> Vec<PyStaticReferenceEpochDiagnostic> {
        self.inner
            .diagnostics
            .iter()
            .cloned()
            .map(|inner| PyStaticReferenceEpochDiagnostic { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "StaticReferenceStationSolution(mode={}, baseline_m={:.4}, epochs={})",
            reference_station_mode_label(self.inner.mode),
            self.inner.baseline_m,
            self.inner.diagnostics.len()
        )
    }
}

/// Solve one static reference-station coordinate from paired RINEX observations.
#[pyfunction]
#[pyo3(signature = (
    sp3,
    reference_obs,
    rover_obs,
    reference_position_m,
    *,
    model=None,
    arc_options=None,
    preprocessing=None,
    update_options=None,
    float_options=None,
    fixed_options=None,
    residual_options=None,
    initial_baseline_m=[0.0; 3],
    baseline_prior_sigma_m=30.0,
    ambiguity_prior_sigma_m=30.0,
    enable_code_dgnss=true,
    enable_carrier_rtk=true,
    with_geodetic=false,
))]
#[allow(clippy::too_many_arguments)]
fn solve_static_reference_station_rinex(
    py: Python<'_>,
    sp3: &PySp3,
    reference_obs: &PyRinexObs,
    rover_obs: &PyRinexObs,
    reference_position_m: [f64; 3],
    model: Option<Py<PyRtkMeasurementModel>>,
    arc_options: Option<Py<PyRtkRinexArcOptions>>,
    preprocessing: Option<Py<PyRtkArcPreprocessing>>,
    update_options: Option<Py<PyRtkArcUpdateOptions>>,
    float_options: Option<Py<PyRtkFloatOptions>>,
    fixed_options: Option<Py<PyRtkFixedOptions>>,
    residual_options: Option<Py<PyRtkResidualValidationOptions>>,
    initial_baseline_m: [f64; 3],
    baseline_prior_sigma_m: f64,
    ambiguity_prior_sigma_m: f64,
    enable_code_dgnss: bool,
    enable_carrier_rtk: bool,
    with_geodetic: bool,
) -> PyResult<PyStaticReferenceStationSolution> {
    let code_options = if enable_code_dgnss {
        Some(
            RinexSppOptions::default_for(rover_obs.inner())
                .map_err(|error| PyValueError::new_err(error.to_string()))?,
        )
    } else {
        None
    };
    let carrier_options = if enable_carrier_rtk {
        let arc_options = rinex_arc_options_from_py(py, arc_options);
        let model = model_from_optional(py, model);
        let static_config = static_arc_config_from_parts(
            py,
            reference_position_m,
            model,
            BTreeMap::new(),
            BTreeMap::new(),
            initial_baseline_m,
            baseline_prior_sigma_m,
            ambiguity_prior_sigma_m,
            update_options,
            preprocessing,
            float_options,
            fixed_options,
            residual_options,
        );
        Some(StaticReferenceCarrierRinexOptions {
            arc_options,
            static_config,
        })
    } else {
        None
    };
    let options = StaticReferenceStationRinexOptions {
        code_options,
        carrier_options,
        with_geodetic,
    };
    let inner = core_solve_static_reference_station_rinex(
        &sp3.inner,
        reference_obs.inner(),
        rover_obs.inner(),
        reference_position_m,
        &options,
    )
    .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(PyStaticReferenceStationSolution { inner })
}

/// Static dual-frequency wide-lane fixed RTK solution built from RINEX.
#[pyclass(module = "sidereon._sidereon", name = "RinexWideLaneFixedRtkSolution")]
pub struct PyRinexWideLaneFixedRtkSolution {
    inner: RtkWideLaneFixedStaticArcSolution,
}

#[pymethods]
impl PyRinexWideLaneFixedRtkSolution {
    #[getter]
    fn wide_lane(&self) -> PyRtkWideLaneArcSolution {
        PyRtkWideLaneArcSolution {
            inner: self.inner.wide_lane.clone(),
        }
    }

    #[getter]
    fn ionosphere_free(&self) -> PyRtkIonosphereFreeArcSolution {
        PyRtkIonosphereFreeArcSolution {
            inner: self.inner.ionosphere_free.clone(),
        }
    }

    #[getter]
    fn solution(&self) -> PyRtkStaticArcSolution {
        PyRtkStaticArcSolution {
            inner: self.inner.solution.clone(),
        }
    }

    /// Integer-fixed baseline as a numpy array `[dx, dy, dz]` metres.
    #[getter]
    fn fixed_baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(
            py,
            &self.inner.solution.fixed_solution.fixed_solution.baseline_m,
        )
    }

    #[getter]
    fn fixed_baseline_m(&self) -> [f64; 3] {
        self.inner.solution.fixed_solution.fixed_solution.baseline_m
    }

    /// Underlying float baseline as a numpy array `[dx, dy, dz]` metres.
    #[getter]
    fn float_baseline<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.solution.float_solution.baseline_m)
    }

    #[getter]
    fn float_baseline_m(&self) -> [f64; 3] {
        self.inner.solution.float_solution.baseline_m
    }

    #[getter]
    fn integer_status(&self) -> PyIntegerStatus {
        self.inner
            .solution
            .fixed_solution
            .fixed_solution
            .search
            .integer_status
            .into()
    }

    #[getter]
    fn integer_ratio(&self) -> Option<f64> {
        self.inner
            .solution
            .fixed_solution
            .fixed_solution
            .search
            .integer_ratio
    }

    #[getter]
    fn float_ambiguity_covariance<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        flat_square_to_array(
            py,
            &self.inner.solution.float_solution.ambiguity_covariance_m,
        )
    }

    #[getter]
    fn wide_lane_fixed(&self) -> bool {
        self.inner.metadata.wide_lane_fixed
    }

    #[getter]
    fn wide_lane_ambiguities_cycles(&self) -> BTreeMap<String, i64> {
        self.inner.metadata.wide_lane_ambiguities_cycles.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexWideLaneFixedRtkSolution(fixed_baseline=[{:.4}, {:.4}, {:.4}], integer_status={:?})",
            self.inner.solution.fixed_solution.fixed_solution.baseline_m[0],
            self.inner.solution.fixed_solution.fixed_solution.baseline_m[1],
            self.inner.solution.fixed_solution.fixed_solution.baseline_m[2],
            self.inner.solution.fixed_solution.fixed_solution.search.integer_status
        )
    }
}

/// Solve a static dual-frequency wide-lane fixed RTK baseline from RINEX OBS.
#[pyfunction]
#[pyo3(signature = (
    ephemeris,
    base_obs,
    rover_obs,
    base,
    model=None,
    arc_options=None,
    update_options=None,
    float_options=None,
    fixed_options=None,
    residual_options=None,
    initial_baseline_m=[0.0; 3],
    baseline_prior_sigma_m=30.0,
    ambiguity_prior_sigma_m=30.0,
    apply_troposphere=true,
))]
#[allow(clippy::too_many_arguments)]
fn solve_wide_lane_fixed_rinex_rtk_baseline(
    py: Python<'_>,
    ephemeris: &Bound<'_, PyAny>,
    base_obs: &PyRinexObs,
    rover_obs: &PyRinexObs,
    base: [f64; 3],
    model: Option<Py<PyRtkMeasurementModel>>,
    arc_options: Option<Py<PyRtkRinexDualArcOptions>>,
    update_options: Option<Py<PyRtkArcUpdateOptions>>,
    float_options: Option<Py<PyRtkFloatOptions>>,
    fixed_options: Option<Py<PyRtkFixedOptions>>,
    residual_options: Option<Py<PyRtkResidualValidationOptions>>,
    initial_baseline_m: [f64; 3],
    baseline_prior_sigma_m: f64,
    ambiguity_prior_sigma_m: f64,
    apply_troposphere: bool,
) -> PyResult<PyRinexWideLaneFixedRtkSolution> {
    let arc_options = rinex_dual_arc_options_from_py(py, arc_options);
    let model = model_from_optional(py, model);
    with_observable_source(ephemeris, |source| {
        let arc = core_build_dual_frequency_rinex_rtk_arc(
            source,
            base_obs.inner(),
            rover_obs.inner(),
            &arc_options,
        )
        .map_err(to_solve_err)?;
        let static_config = static_arc_config_from_parts(
            py,
            base,
            model,
            BTreeMap::new(),
            BTreeMap::new(),
            initial_baseline_m,
            baseline_prior_sigma_m,
            ambiguity_prior_sigma_m,
            update_options,
            None,
            float_options,
            fixed_options,
            residual_options,
        );
        let config = RtkWideLaneFixedArcConfig {
            wide_lane: RtkWideLaneArcConfig {
                base_m: base,
                reference: BaselineReferenceSelection::Auto,
                options: WideLaneOptions {
                    min_epochs: 2,
                    tolerance_cycles: 0.5,
                    skip_short_fragments: false,
                },
                cycle_slip: Some(RtkDualCycleSlipConfig {
                    policy: CycleSlipPolicy::DropSatellite,
                    options: CycleSlipOptions::default(),
                }),
            },
            ionosphere_free: RtkIonosphereFreeArcConfig {
                base_m: base,
                initial_baseline_m,
                reference: BaselineReferenceSelection::Auto,
                apply_troposphere,
            },
            solve: RtkWideLaneFixedArcSolveConfig::Static(static_config),
        };
        let inner =
            core_solve_wide_lane_fixed_rtk_arc(&arc.epochs, &config).map_err(to_solve_err)?;
        match inner {
            RtkWideLaneFixedArcSolution::Static(inner) => {
                Ok(PyRinexWideLaneFixedRtkSolution { inner })
            }
            RtkWideLaneFixedArcSolution::Sequential(_) => Err(PyTypeError::new_err(
                "wide-lane RINEX convenience expected a static solution",
            )),
        }
    })
}

/// One dual-frequency observation at a receiver.
#[pyclass(module = "sidereon._sidereon", name = "RtkDualFrequencyObservation")]
#[derive(Clone)]
pub struct PyRtkDualFrequencyObservation {
    inner: RtkDualFrequencyObservation,
}

#[pymethods]
impl PyRtkDualFrequencyObservation {
    /// Create one dual-frequency code+carrier observation.
    #[new]
    #[pyo3(signature = (
        ambiguity_id,
        p1_m,
        p2_m,
        phi1_cycles,
        phi2_cycles,
        f1_hz,
        f2_hz,
        lli1=None,
        lli2=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        ambiguity_id: String,
        p1_m: f64,
        p2_m: f64,
        phi1_cycles: f64,
        phi2_cycles: f64,
        f1_hz: f64,
        f2_hz: f64,
        lli1: Option<i64>,
        lli2: Option<i64>,
    ) -> Self {
        Self {
            inner: RtkDualFrequencyObservation {
                ambiguity_id,
                p1_m,
                p2_m,
                phi1_cycles,
                phi2_cycles,
                f1_hz,
                f2_hz,
                lli1,
                lli2,
            },
        }
    }

    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    #[getter]
    fn p1_m(&self) -> f64 {
        self.inner.p1_m
    }

    #[getter]
    fn p2_m(&self) -> f64 {
        self.inner.p2_m
    }

    #[getter]
    fn phi1_cycles(&self) -> f64 {
        self.inner.phi1_cycles
    }

    #[getter]
    fn phi2_cycles(&self) -> f64 {
        self.inner.phi2_cycles
    }

    #[getter]
    fn f1_hz(&self) -> f64 {
        self.inner.f1_hz
    }

    #[getter]
    fn f2_hz(&self) -> f64 {
        self.inner.f2_hz
    }

    #[getter]
    fn lli1(&self) -> Option<i64> {
        self.inner.lli1
    }

    #[getter]
    fn lli2(&self) -> Option<i64> {
        self.inner.lli2
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkDualFrequencyObservation(ambiguity_id={:?})",
            self.inner.ambiguity_id
        )
    }
}

/// One satellite's paired base and rover dual-frequency observations.
#[pyclass(
    module = "sidereon._sidereon",
    name = "RtkDualFrequencySatelliteObservation"
)]
#[derive(Clone)]
pub struct PyRtkDualFrequencySatelliteObservation {
    inner: RtkDualFrequencySatelliteObservation,
}

#[pymethods]
impl PyRtkDualFrequencySatelliteObservation {
    /// Create one satellite row for a dual-frequency RTK arc epoch.
    #[new]
    #[pyo3(signature = (satellite_id, base, rover))]
    fn new(
        satellite_id: String,
        base: &PyRtkDualFrequencyObservation,
        rover: &PyRtkDualFrequencyObservation,
    ) -> Self {
        Self {
            inner: RtkDualFrequencySatelliteObservation {
                satellite_id,
                base: base.inner.clone(),
                rover: rover.inner.clone(),
            },
        }
    }

    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    #[getter]
    fn base(&self) -> PyRtkDualFrequencyObservation {
        PyRtkDualFrequencyObservation {
            inner: self.inner.base.clone(),
        }
    }

    #[getter]
    fn rover(&self) -> PyRtkDualFrequencyObservation {
        PyRtkDualFrequencyObservation {
            inner: self.inner.rover.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkDualFrequencySatelliteObservation(satellite_id={:?})",
            self.inner.satellite_id
        )
    }
}

/// One dual-frequency RTK arc epoch.
#[pyclass(module = "sidereon._sidereon", name = "RtkDualFrequencyArcEpoch")]
#[derive(Clone)]
pub struct PyRtkDualFrequencyArcEpoch {
    inner: RtkDualFrequencyArcEpoch,
}

#[pymethods]
impl PyRtkDualFrequencyArcEpoch {
    /// Create one dual-frequency RTK arc epoch.
    #[new]
    #[pyo3(signature = (
        jd_whole,
        jd_fraction,
        observations,
        satellite_positions_m,
        epoch_sort_key=None,
        gap_time_s=None,
        base_satellite_positions_m=BTreeMap::new(),
        rover_satellite_positions_m=BTreeMap::new(),
        velocity_mps=None,
        prediction_time_s=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        jd_whole: f64,
        jd_fraction: f64,
        observations: Vec<Py<PyRtkDualFrequencySatelliteObservation>>,
        satellite_positions_m: BTreeMap<String, [f64; 3]>,
        epoch_sort_key: Option<String>,
        gap_time_s: Option<f64>,
        base_satellite_positions_m: BTreeMap<String, [f64; 3]>,
        rover_satellite_positions_m: BTreeMap<String, [f64; 3]>,
        velocity_mps: Option<[f64; 3]>,
        prediction_time_s: Option<f64>,
    ) -> Self {
        let observations = observations
            .iter()
            .map(|observation| observation.borrow(py).inner.clone())
            .collect();
        Self {
            inner: RtkDualFrequencyArcEpoch {
                jd_whole,
                jd_fraction,
                epoch_sort_key,
                gap_time_s,
                observations,
                satellite_positions_m,
                base_satellite_positions_m,
                rover_satellite_positions_m,
                velocity_mps,
                prediction_time_s,
            },
        }
    }

    #[getter]
    fn jd_whole(&self) -> f64 {
        self.inner.jd_whole
    }

    #[getter]
    fn jd_fraction(&self) -> f64 {
        self.inner.jd_fraction
    }

    #[getter]
    fn epoch_sort_key(&self) -> Option<String> {
        self.inner.epoch_sort_key.clone()
    }

    #[getter]
    fn gap_time_s(&self) -> Option<f64> {
        self.inner.gap_time_s
    }

    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.observations.len()
    }

    #[getter]
    fn velocity_mps(&self) -> Option<[f64; 3]> {
        self.inner.velocity_mps
    }

    #[getter]
    fn prediction_time_s(&self) -> Option<f64> {
        self.inner.prediction_time_s
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkDualFrequencyArcEpoch(observations={})",
            self.inner.observations.len()
        )
    }
}

/// Wide-lane integer estimation controls.
#[pyclass(module = "sidereon._sidereon", name = "RtkWideLaneOptions")]
#[derive(Clone, Copy)]
pub struct PyRtkWideLaneOptions {
    inner: WideLaneOptions,
}

#[pymethods]
impl PyRtkWideLaneOptions {
    /// Create wide-lane integer estimation controls.
    #[new]
    #[pyo3(signature = (min_epochs, tolerance_cycles, skip_short_fragments=false))]
    fn new(min_epochs: usize, tolerance_cycles: f64, skip_short_fragments: bool) -> Self {
        Self {
            inner: WideLaneOptions {
                min_epochs,
                tolerance_cycles,
                skip_short_fragments,
            },
        }
    }

    #[getter]
    fn min_epochs(&self) -> usize {
        self.inner.min_epochs
    }

    #[getter]
    fn tolerance_cycles(&self) -> f64 {
        self.inner.tolerance_cycles
    }

    #[getter]
    fn skip_short_fragments(&self) -> bool {
        self.inner.skip_short_fragments
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkWideLaneOptions(min_epochs={}, tolerance_cycles={:.3}, \
             skip_short_fragments={})",
            self.inner.min_epochs, self.inner.tolerance_cycles, self.inner.skip_short_fragments
        )
    }
}

/// Optional dual-frequency cycle-slip preprocessing controls.
#[pyclass(module = "sidereon._sidereon", name = "RtkDualCycleSlipConfig")]
#[derive(Clone, Copy)]
pub struct PyRtkDualCycleSlipConfig {
    inner: RtkDualCycleSlipConfig,
}

#[pymethods]
impl PyRtkDualCycleSlipConfig {
    /// Create dual-frequency cycle-slip preprocessing controls.
    #[new]
    #[pyo3(signature = (
        policy,
        gf_threshold_m=0.05,
        mw_threshold_cycles=4.0,
        min_arc_gap_s=300.0,
    ))]
    fn new(
        policy: String,
        gf_threshold_m: f64,
        mw_threshold_cycles: f64,
        min_arc_gap_s: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: RtkDualCycleSlipConfig {
                policy: extract_cycle_slip_policy(&policy)?,
                options: CycleSlipOptions {
                    gf_threshold_m,
                    mw_threshold_cycles,
                    min_arc_gap_s,
                },
            },
        })
    }

    #[getter]
    fn policy(&self) -> &'static str {
        cycle_slip_policy_label(self.inner.policy)
    }

    #[getter]
    fn gf_threshold_m(&self) -> f64 {
        self.inner.options.gf_threshold_m
    }

    #[getter]
    fn mw_threshold_cycles(&self) -> f64 {
        self.inner.options.mw_threshold_cycles
    }

    #[getter]
    fn min_arc_gap_s(&self) -> f64 {
        self.inner.options.min_arc_gap_s
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkDualCycleSlipConfig(policy={:?})",
            cycle_slip_policy_label(self.inner.policy)
        )
    }
}

/// Complete typed configuration for wide-lane RTK arc fixing.
#[pyclass(module = "sidereon._sidereon", name = "RtkWideLaneArcConfig")]
pub struct PyRtkWideLaneArcConfig {
    inner: RtkWideLaneArcConfig,
}

#[pymethods]
impl PyRtkWideLaneArcConfig {
    /// Create a wide-lane RTK arc configuration.
    #[new]
    #[pyo3(signature = (
        base,
        options,
        reference_satellite=None,
        reference_per_system=None,
        cycle_slip=None,
    ))]
    fn new(
        py: Python<'_>,
        base: [f64; 3],
        options: &PyRtkWideLaneOptions,
        reference_satellite: Option<String>,
        reference_per_system: Option<BTreeMap<String, String>>,
        cycle_slip: Option<Py<PyRtkDualCycleSlipConfig>>,
    ) -> Self {
        let cycle_slip = cycle_slip.as_ref().map(|config| config.borrow(py).inner);
        Self {
            inner: RtkWideLaneArcConfig {
                base_m: base,
                reference: reference_selection(reference_satellite, reference_per_system),
                options: options.inner,
                cycle_slip,
            },
        }
    }

    #[getter]
    fn base(&self) -> [f64; 3] {
        self.inner.base_m
    }

    #[getter]
    fn options(&self) -> PyRtkWideLaneOptions {
        PyRtkWideLaneOptions {
            inner: self.inner.options,
        }
    }

    #[getter]
    fn cycle_slip(&self) -> Option<PyRtkDualCycleSlipConfig> {
        self.inner
            .cycle_slip
            .map(|inner| PyRtkDualCycleSlipConfig { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkWideLaneArcConfig(base=[{:.3}, {:.3}, {:.3}])",
            self.inner.base_m[0], self.inner.base_m[1], self.inner.base_m[2]
        )
    }
}

/// Wide-lane RTK arc fixing result.
#[pyclass(module = "sidereon._sidereon", name = "RtkWideLaneArcSolution")]
pub struct PyRtkWideLaneArcSolution {
    inner: RtkWideLaneArcSolution,
}

#[pymethods]
impl PyRtkWideLaneArcSolution {
    #[getter]
    fn references(&self) -> BTreeMap<String, String> {
        self.inner.references.clone()
    }

    #[getter]
    fn wide_lane_cycles(&self) -> BTreeMap<String, i64> {
        self.inner.wide_lane_cycles.clone()
    }

    #[getter]
    fn epochs(&self) -> Vec<PyRtkDualFrequencyArcEpoch> {
        self.inner
            .epochs
            .iter()
            .map(|inner| PyRtkDualFrequencyArcEpoch {
                inner: inner.clone(),
            })
            .collect()
    }

    #[getter]
    fn dropped_sats(&self) -> Vec<String> {
        self.inner.dropped_sats.clone()
    }

    #[getter]
    fn split_cycle_slip_arcs(&self) -> Vec<PyCycleSlipSplitArc> {
        self.inner
            .split_cycle_slip_arcs
            .iter()
            .map(|inner| PyCycleSlipSplitArc {
                inner: inner.clone(),
            })
            .collect()
    }

    /// Geometry observability and covariance-validation diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkWideLaneArcSolution(references={}, wide_lane_cycles={})",
            self.inner.references.len(),
            self.inner.wide_lane_cycles.len()
        )
    }
}

/// Fix wide-lane RTK ambiguities over a dual-frequency arc.
#[pyfunction]
#[pyo3(signature = (epochs, config))]
fn fix_wide_lane_rtk_arc(
    py: Python<'_>,
    epochs: Vec<Py<PyRtkDualFrequencyArcEpoch>>,
    config: &PyRtkWideLaneArcConfig,
) -> PyResult<PyRtkWideLaneArcSolution> {
    let epochs: Vec<RtkDualFrequencyArcEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).inner.clone())
        .collect();
    let inner = core_fix_wide_lane_rtk_arc(&epochs, &config.inner).map_err(to_solve_err)?;
    Ok(PyRtkWideLaneArcSolution { inner })
}

/// Complete typed configuration for ionosphere-free RTK arc setup.
#[pyclass(module = "sidereon._sidereon", name = "RtkIonosphereFreeArcConfig")]
pub struct PyRtkIonosphereFreeArcConfig {
    inner: RtkIonosphereFreeArcConfig,
}

#[pymethods]
impl PyRtkIonosphereFreeArcConfig {
    /// Create an ionosphere-free RTK arc setup configuration.
    #[new]
    #[pyo3(signature = (
        base,
        initial_baseline_m=[0.0; 3],
        reference_satellite=None,
        reference_per_system=None,
        apply_troposphere=false,
    ))]
    fn new(
        base: [f64; 3],
        initial_baseline_m: [f64; 3],
        reference_satellite: Option<String>,
        reference_per_system: Option<BTreeMap<String, String>>,
        apply_troposphere: bool,
    ) -> Self {
        Self {
            inner: RtkIonosphereFreeArcConfig {
                base_m: base,
                initial_baseline_m,
                reference: reference_selection(reference_satellite, reference_per_system),
                apply_troposphere,
            },
        }
    }

    #[getter]
    fn base(&self) -> [f64; 3] {
        self.inner.base_m
    }

    #[getter]
    fn initial_baseline_m(&self) -> [f64; 3] {
        self.inner.initial_baseline_m
    }

    #[getter]
    fn apply_troposphere(&self) -> bool {
        self.inner.apply_troposphere
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkIonosphereFreeArcConfig(base=[{:.3}, {:.3}, {:.3}], \
             apply_troposphere={})",
            self.inner.base_m[0],
            self.inner.base_m[1],
            self.inner.base_m[2],
            self.inner.apply_troposphere
        )
    }
}

/// Ionosphere-free RTK arc setup result.
#[pyclass(module = "sidereon._sidereon", name = "RtkIonosphereFreeArcSolution")]
pub struct PyRtkIonosphereFreeArcSolution {
    inner: RtkIonosphereFreeArcSolution,
}

#[pymethods]
impl PyRtkIonosphereFreeArcSolution {
    #[getter]
    fn references(&self) -> BTreeMap<String, String> {
        self.inner.references.clone()
    }

    #[getter]
    fn epochs(&self) -> Vec<PyRtkArcEpoch> {
        self.inner
            .epochs
            .iter()
            .map(|inner| PyRtkArcEpoch {
                inner: inner.clone(),
            })
            .collect()
    }

    #[getter]
    fn wavelengths_m(&self) -> BTreeMap<String, f64> {
        self.inner.wavelengths_m.clone()
    }

    #[getter]
    fn offsets_m(&self) -> BTreeMap<String, f64> {
        self.inner.offsets_m.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtkIonosphereFreeArcSolution(references={}, epochs={})",
            self.inner.references.len(),
            self.inner.epochs.len()
        )
    }
}

/// Prepare ionosphere-free single-frequency RTK arc inputs from dual-frequency data.
#[pyfunction]
#[pyo3(signature = (epochs, wide_lane_cycles, config))]
fn prepare_ionosphere_free_rtk_arc(
    py: Python<'_>,
    epochs: Vec<Py<PyRtkDualFrequencyArcEpoch>>,
    wide_lane_cycles: BTreeMap<String, i64>,
    config: &PyRtkIonosphereFreeArcConfig,
) -> PyResult<PyRtkIonosphereFreeArcSolution> {
    let epochs: Vec<RtkDualFrequencyArcEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).inner.clone())
        .collect();
    let inner = core_prepare_ionosphere_free_rtk_arc(&epochs, &wide_lane_cycles, &config.inner)
        .map_err(to_solve_err)?;
    Ok(PyRtkIonosphereFreeArcSolution { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRtkFloatSolution>()?;
    m.add_class::<PyRtkFixedSolution>()?;
    m.add_class::<PyMovingBaselineEpoch>()?;
    m.add_class::<PyMovingBaselineEpochSolution>()?;
    m.add_function(wrap_pyfunction!(solve_moving_baseline, m)?)?;
    m.add_function(wrap_pyfunction!(solve_rtk_float, m)?)?;
    m.add_function(wrap_pyfunction!(solve_rtk_fixed, m)?)?;
    m.add_class::<PyIntegerStatus>()?;
    m.add_class::<PyRtkStochasticModel>()?;
    m.add_class::<PyRtkSatMeasurement>()?;
    m.add_class::<PyRtkEpoch>()?;
    m.add_class::<PyRtkMeasurementModel>()?;
    m.add_class::<PyRtkFloatOptions>()?;
    m.add_class::<PyRtkFixedOptions>()?;
    m.add_class::<PyRtkResidualValidationOptions>()?;
    m.add_class::<PyRtkFloatConfig>()?;
    m.add_class::<PyRtkFixedConfig>()?;
    m.add_class::<PyRtkArcObservation>()?;
    m.add_class::<PyRtkArcEpoch>()?;
    m.add_class::<PyRtkArcUpdateOptions>()?;
    m.add_class::<PyRtkArcPreprocessing>()?;
    m.add_class::<PyRtkArcConfig>()?;
    m.add_class::<PyRtkArcIntegerSearch>()?;
    m.add_class::<PyRtkArcResidual>()?;
    m.add_class::<PyRtkArcEpochSolution>()?;
    m.add_class::<PyCycleSlipSplitArc>()?;
    m.add_class::<PyRtkFilterState>()?;
    m.add_class::<PyRtkArcSolution>()?;
    m.add_function(wrap_pyfunction!(solve_rtk_arc, m)?)?;
    m.add_class::<PyRtkStaticArcConfig>()?;
    m.add_class::<PyRtkStaticArcSolution>()?;
    m.add_function(wrap_pyfunction!(solve_static_rtk_arc, m)?)?;
    m.add_class::<PyRtkRinexSignalPair>()?;
    m.add_class::<PyRtkRinexArcOptions>()?;
    m.add_class::<PyRinexRtkArc>()?;
    m.add_function(wrap_pyfunction!(build_rinex_rtk_arc, m)?)?;
    m.add_function(wrap_pyfunction!(solve_static_rinex_rtk_baseline, m)?)?;
    m.add_class::<PyStaticReferenceStationCovariance>()?;
    m.add_class::<PyStaticReferenceEpochDiagnostic>()?;
    m.add_class::<PyStaticReferenceModeError>()?;
    m.add_class::<PyStaticReferenceModeReport>()?;
    m.add_class::<PyStaticReferenceCodeSolution>()?;
    m.add_class::<PyStaticReferenceCarrierSolution>()?;
    m.add_class::<PyStaticReferenceStationSolution>()?;
    m.add_function(wrap_pyfunction!(solve_static_reference_station_rinex, m)?)?;
    m.add_class::<PyRtkRinexDualSignalPair>()?;
    m.add_class::<PyRtkRinexDualArcOptions>()?;
    m.add_class::<PyRinexDualFrequencyRtkArc>()?;
    m.add_function(wrap_pyfunction!(build_dual_frequency_rinex_rtk_arc, m)?)?;
    m.add_class::<PyRinexWideLaneFixedRtkSolution>()?;
    m.add_function(wrap_pyfunction!(
        solve_wide_lane_fixed_rinex_rtk_baseline,
        m
    )?)?;
    m.add_class::<PyRtkDualFrequencyObservation>()?;
    m.add_class::<PyRtkDualFrequencySatelliteObservation>()?;
    m.add_class::<PyRtkDualFrequencyArcEpoch>()?;
    m.add_class::<PyRtkWideLaneOptions>()?;
    m.add_class::<PyRtkDualCycleSlipConfig>()?;
    m.add_class::<PyRtkWideLaneArcConfig>()?;
    m.add_class::<PyRtkWideLaneArcSolution>()?;
    m.add_function(wrap_pyfunction!(fix_wide_lane_rtk_arc, m)?)?;
    m.add_class::<PyRtkIonosphereFreeArcConfig>()?;
    m.add_class::<PyRtkIonosphereFreeArcSolution>()?;
    m.add_function(wrap_pyfunction!(prepare_ionosphere_free_rtk_arc, m)?)?;
    Ok(())
}
