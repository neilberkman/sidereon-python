//! PPP arc binding: static multi-epoch float PPP.
//!
//! Marshals the structured ionosphere-free epoch records, initial state, and
//! solve config from Python dicts into the `sidereon-core` PPP input types and
//! calls `sidereon::solve_ppp_float`. The SP3 product is the ephemeris source.
//! No modeling lives here.

use std::collections::BTreeMap;
use std::str::FromStr;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon_core::atmosphere::troposphere::Met;
use sidereon_core::positioning::SurfaceMet;
use sidereon_core::ppp_corrections::CivilDateTime;
use sidereon_core::precise_positioning::defaults::{
    AMBIGUITY_TOLERANCE_M, CLOCK_TOLERANCE_M, MAX_ITERATIONS, POSITION_TOLERANCE_M,
    RATIO_THRESHOLD, ZTD_TOLERANCE_M,
};
use sidereon_core::precise_positioning::{
    solve_ppp_auto_init_fixed as core_solve_ppp_auto_init_fixed,
    solve_ppp_auto_init_float as core_solve_ppp_auto_init_float, FixedAmbiguityOptions,
    FixedSolution, FixedSolveConfig, FloatEpoch, FloatObservation, FloatObservationSignals,
    FloatResidual, FloatSolution, FloatSolveConfig, FloatSolveOptions, FloatState, FloatStatus,
    IntegerStatus as PppIntegerStatus, MeasurementWeights, PcvSample, PppAutoInitOptions,
    PppCorrectionLookup, PppInitialGuess, RangeCorrections, ReceiverAntennaFrequency,
    ReceiverAntennaOptions, SatelliteClockCorrections, SsrBiasExclusion, SsrIfCombinationStatus,
    SsrObsApplicationReport, SsrObsSignalReport, SsrTransmitTimeFailure,
    TemporalCorrelationSummary, TropoMapping, TroposphereOptions, UnplacedObservation,
    UnplacedObservationReason, VmfSiteSample, VmfSiteSeries,
};
use sidereon_core::ssr::{
    SignalCode, SsrCodeBiasQueryResult, SsrPhaseBiasQueryResult, SsrSolution,
};

use crate::marshal::{debug_variant_snake, mat3_to_array, option_py_or_default};
use crate::ppp_corrections::{PyPppCorrections, PyPppCorrectionsOptions};
use crate::sbas_ssr::{PySsrCorrectionSize, PySsrSolution};
use sidereon_core::GnssSatelliteId;

fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    GnssSatelliteId::from_str(token)
        .map_err(|_| PyValueError::new_err(format!("invalid satellite token: {token}")))
}

/// A PPP observation left out before the solve because no transmission epoch
/// can be placed from it.
#[pyclass(module = "sidereon._sidereon", name = "PppUnplacedObservation")]
#[derive(Clone)]
pub struct PppUnplacedObservationRow {
    inner: UnplacedObservation,
}

#[pymethods]
impl PppUnplacedObservationRow {
    /// Input epoch index of the observation.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    /// Satellite token of the observation.
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    /// Ambiguity id of the observation.
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    /// Why no transmission epoch is placed: `code_not_positive` for a zero
    /// or negative code, which RTKLIB reads as no pseudorange.
    #[getter]
    fn reason(&self) -> String {
        match self.inner.reason {
            UnplacedObservationReason::CodeNotPositive => "code_not_positive".to_string(),
            UnplacedObservationReason::SsrCorrectionExceedsLimit(_) => {
                "ssr_correction_exceeds_limit".to_string()
            }
            other => debug_variant_snake(&other),
        }
    }

    #[getter]
    fn correction_size(&self) -> Option<PySsrCorrectionSize> {
        match self.inner.reason {
            UnplacedObservationReason::SsrCorrectionExceedsLimit(size) => Some(size.into()),
            _ => None,
        }
    }

    /// Why no transmission epoch is placed, in words.
    #[getter]
    fn message(&self) -> String {
        let reason = match self.inner.reason {
            UnplacedObservationReason::CodeNotPositive => {
                "the code is zero or negative, which places no transmission epoch".to_string()
            }
            UnplacedObservationReason::SsrCorrectionExceedsLimit(size) => format!(
                "the SSR correction exceeds its application limit (orbit {} m, clock {} m)",
                size.orbit_m, size.clock_m
            ),
            other => format!("{other:?}"),
        };
        format!(
            "epoch {} observation {} ({}) left out: {reason}",
            self.inner.epoch_index, self.inner.ambiguity_id, self.inner.satellite_id
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "PppUnplacedObservation(epoch_index={}, satellite_id={:?}, ambiguity_id={:?}, reason={:?})",
            self.inner.epoch_index,
            self.inner.satellite_id,
            self.inner.ambiguity_id,
            self.reason()
        )
    }
}

fn unplaced_rows(rows: &[UnplacedObservation]) -> Vec<PppUnplacedObservationRow> {
    rows.iter()
        .cloned()
        .map(|inner| PppUnplacedObservationRow { inner })
        .collect()
}

/// The four tracking codes of a PPP observation as
/// `(code1, code2, phase1, phase2)`.
type SignalCodes = (String, String, String, String);

/// Termination status of a PPP float or fixed solve.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyPppFloatStatus {
    /// Every active state update met its configured tolerance.
    STATE_TOLERANCE,
    /// The solve reached `max_iterations` first.
    MAX_ITERATIONS,
}

impl From<FloatStatus> for PyPppFloatStatus {
    fn from(status: FloatStatus) -> Self {
        match status {
            FloatStatus::StateTolerance => Self::STATE_TOLERANCE,
            FloatStatus::MaxIterations => Self::MAX_ITERATIONS,
        }
    }
}

#[pymethods]
impl PyPppFloatStatus {
    /// Snake-case label: `state_tolerance` or `max_iterations`.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::STATE_TOLERANCE => "state_tolerance",
            Self::MAX_ITERATIONS => "max_iterations",
        }
    }

    fn __repr__(&self) -> String {
        format!("PppFloatStatus.{}", self.label().to_ascii_uppercase())
    }
}

/// One residual row of a PPP float or fixed solution.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatResidual", eq)]
#[derive(Clone, PartialEq)]
pub struct PyPppFloatResidual {
    inner: FloatResidual,
}

#[pymethods]
impl PyPppFloatResidual {
    /// Input epoch index of the row.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    /// Satellite token of the observation.
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    /// Ambiguity id of the observation.
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    /// Code prefit residual, metres.
    #[getter]
    fn code_m(&self) -> f64 {
        self.inner.code_m
    }

    /// Phase prefit residual, metres.
    #[getter]
    fn phase_m(&self) -> f64 {
        self.inner.phase_m
    }

    /// Code inverse-sigma row weight, including elevation scaling.
    #[getter]
    fn code_weight(&self) -> f64 {
        self.inner.code_weight
    }

    /// Phase inverse-sigma row weight, including elevation scaling.
    #[getter]
    fn phase_weight(&self) -> f64 {
        self.inner.phase_weight
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatResidual(epoch_index={}, ambiguity_id={:?}, code_m={:.6}, phase_m={:.6})",
            self.inner.epoch_index, self.inner.ambiguity_id, self.inner.code_m, self.inner.phase_m
        )
    }
}

fn residual_rows(rows: &[FloatResidual]) -> Vec<PyPppFloatResidual> {
    rows.iter()
        .cloned()
        .map(|inner| PyPppFloatResidual { inner })
        .collect()
}

/// Why the SSR/HAS biases recorded for an observation do not hold at its
/// transmission time.
#[pyclass(module = "sidereon._sidereon", name = "PppSsrTransmitTimeFailure")]
#[derive(Clone)]
pub struct PyPppSsrTransmitTimeFailure {
    inner: SsrTransmitTimeFailure,
}

#[pymethods]
impl PyPppSsrTransmitTimeFailure {
    /// `source_without_ssr_corrections`, `transmit_time_unavailable`,
    /// `orbit_clock_solution`, `bias_record` or `source`.
    #[getter]
    fn kind(&self) -> String {
        debug_variant_snake(&self.inner)
    }

    /// Transmission time, seconds since J2000, where the failure states one.
    #[getter]
    fn transmit_time_j2000_s(&self) -> Option<f64> {
        match &self.inner {
            SsrTransmitTimeFailure::OrbitClockSolution {
                transmit_time_j2000_s,
                ..
            }
            | SsrTransmitTimeFailure::BiasRecord {
                transmit_time_j2000_s,
                ..
            }
            | SsrTransmitTimeFailure::Source {
                transmit_time_j2000_s,
                ..
            } => Some(*transmit_time_j2000_s),
            _ => None,
        }
    }

    /// The orbit and clock solution the source applies at the transmission
    /// time, for `orbit_clock_solution`; `None` there when it applies none.
    #[getter]
    fn applied_solution(&self) -> Option<PySsrSolution> {
        match &self.inner {
            SsrTransmitTimeFailure::OrbitClockSolution { applied, .. } => {
                applied.map(PySsrSolution::from)
            }
            _ => None,
        }
    }

    /// Signal of the record, for `bias_record`.
    #[getter]
    fn signal(&self) -> Option<String> {
        match &self.inner {
            SsrTransmitTimeFailure::BiasRecord { signal, .. } => Some(signal.to_string()),
            _ => None,
        }
    }

    /// Status of the query for that signal at the transmission time, for
    /// `bias_record`.
    #[getter]
    fn status(&self) -> Option<String> {
        match &self.inner {
            SsrTransmitTimeFailure::BiasRecord { status, .. } => Some(debug_variant_snake(status)),
            _ => None,
        }
    }

    /// The source's error message, for `source`.
    #[getter]
    fn error(&self) -> Option<String> {
        match &self.inner {
            SsrTransmitTimeFailure::Source { error, .. } => Some(error.to_string()),
            _ => None,
        }
    }

    /// The core value in full, as its `Debug` text.
    #[getter]
    fn detail(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn __repr__(&self) -> String {
        format!("PppSsrTransmitTimeFailure(kind={:?})", self.kind())
    }
}

/// The query of one signal's SSR/HAS bias for an observation.
#[pyclass(module = "sidereon._sidereon", name = "PppSsrSignalReport")]
#[derive(Clone)]
pub struct PyPppSsrSignalReport {
    epoch_index: usize,
    satellite_id: String,
    ambiguity_id: String,
    signal: String,
    query_signal: String,
    source_signal: Option<String>,
    status: String,
    bias_m: Option<f64>,
    bias_cycles: Option<f64>,
    solution: Option<SsrSolution>,
    iod_ssr: Option<u8>,
    ref_epoch_j2000_s: Option<f64>,
    query: String,
}

impl PyPppSsrSignalReport {
    fn code(report: &SsrObsSignalReport<SsrCodeBiasQueryResult>) -> Self {
        let q = &report.query_result;
        Self {
            epoch_index: report.epoch_index,
            satellite_id: report.sat.to_string(),
            ambiguity_id: report.ambiguity_id.clone(),
            signal: report.signal.to_string(),
            query_signal: q.signal.to_string(),
            source_signal: q.source_signal.as_ref().map(ToString::to_string),
            status: debug_variant_snake(&q.status),
            bias_m: q.bias_m,
            bias_cycles: None,
            solution: q.solution,
            iod_ssr: q.iod_ssr,
            ref_epoch_j2000_s: q.ref_epoch_j2000_s,
            query: format!("{q:?}"),
        }
    }

    fn phase(report: &SsrObsSignalReport<SsrPhaseBiasQueryResult>) -> Self {
        let q = &report.query_result;
        Self {
            epoch_index: report.epoch_index,
            satellite_id: report.sat.to_string(),
            ambiguity_id: report.ambiguity_id.clone(),
            signal: report.signal.to_string(),
            query_signal: q.signal.to_string(),
            source_signal: q.source_signal.as_ref().map(ToString::to_string),
            status: debug_variant_snake(&q.status),
            bias_m: q.bias_m,
            bias_cycles: q.bias_cycles,
            solution: q.solution,
            iod_ssr: q.iod_ssr,
            ref_epoch_j2000_s: q.ref_epoch_j2000_s,
            query: format!("{q:?}"),
        }
    }
}

#[pymethods]
impl PyPppSsrSignalReport {
    /// Input epoch index of the observation.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.epoch_index
    }

    /// Satellite token of the observation.
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.satellite_id
    }

    /// Ambiguity id of the observation.
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.ambiguity_id
    }

    /// The physical signal requested.
    #[getter]
    fn signal(&self) -> &str {
        &self.signal
    }

    /// The signal key the store was queried with.
    #[getter]
    fn query_signal(&self) -> &str {
        &self.query_signal
    }

    /// The raw signal of the record, where the store keeps one.
    #[getter]
    fn source_signal(&self) -> Option<&str> {
        self.source_signal.as_deref()
    }

    /// Query status in snake case, such as `available`, `missing` or
    /// `expired`.
    #[getter]
    fn status(&self) -> &str {
        &self.status
    }

    /// Bias in metres, when the query resolved one.
    #[getter]
    fn bias_m(&self) -> Option<f64> {
        self.bias_m
    }

    /// Phase bias in cycles, for a phase query that resolved one.
    #[getter]
    fn bias_cycles(&self) -> Option<f64> {
        self.bias_cycles
    }

    /// Solution of the record.
    #[getter]
    fn solution(&self) -> Option<PySsrSolution> {
        self.solution.map(PySsrSolution::from)
    }

    /// IOD SSR of the record.
    #[getter]
    fn iod_ssr(&self) -> Option<u8> {
        self.iod_ssr
    }

    /// Reference epoch of the record, seconds since J2000.
    #[getter]
    fn ref_epoch_j2000_s(&self) -> Option<f64> {
        self.ref_epoch_j2000_s
    }

    /// The core query result in full, as its `Debug` text, including the
    /// lifetime, resolution details and phase continuity state.
    #[getter]
    fn query(&self) -> &str {
        &self.query
    }

    fn __repr__(&self) -> String {
        format!(
            "PppSsrSignalReport(ambiguity_id={:?}, signal={:?}, status={:?})",
            self.ambiguity_id, self.signal, self.status
        )
    }
}

fn combination_status_ut1_reason(status: &SsrIfCombinationStatus) -> Option<&'static str> {
    match status {
        SsrIfCombinationStatus::Ut1OutsideCoverage(reason) => {
            Some(crate::degrade_reason_label(*reason))
        }
        _ => None,
    }
}

/// How SSR/HAS biases were applied to one observation.
#[pyclass(module = "sidereon._sidereon", name = "PppSsrObservationApplication")]
#[derive(Clone)]
pub struct PyPppSsrObservationApplication {
    inner: SsrObsApplicationReport,
}

#[pymethods]
impl PyPppSsrObservationApplication {
    /// Input epoch index of the observation.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    /// Satellite token of the observation.
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    /// Ambiguity id of the observation.
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    /// Transmission time, seconds since J2000, when it could be predicted.
    #[getter]
    fn transmit_time_j2000_s(&self) -> Option<f64> {
        self.inner.transmit_time_j2000_s
    }

    /// Solution of the orbit and clock corrections the source applies there.
    #[getter]
    fn applied_orbit_clock_solution(&self) -> Option<PySsrSolution> {
        self.inner
            .applied_orbit_clock_solution
            .map(PySsrSolution::from)
    }

    /// Tracking codes of the observation as `(code1, code2, phase1, phase2)`.
    #[getter]
    fn observation_signals(&self) -> Option<SignalCodes> {
        self.inner.observation_signals.as_ref().map(|signals| {
            (
                signals.code1.to_string(),
                signals.code2.to_string(),
                signals.phase1.to_string(),
                signals.phase2.to_string(),
            )
        })
    }

    /// Code combination status in snake case, such as `applied` or
    /// `signal_unavailable`.
    #[getter]
    fn code_status(&self) -> String {
        debug_variant_snake(&self.inner.code_status)
    }

    /// UT1 degrade reason when `code_status` is `ut1_outside_coverage`.
    #[getter]
    fn code_status_ut1_reason(&self) -> Option<&'static str> {
        combination_status_ut1_reason(&self.inner.code_status)
    }

    /// Applied ionosphere-free code bias, metres.
    #[getter]
    fn applied_code_if_m(&self) -> Option<f64> {
        self.inner.applied_code_if_m
    }

    #[getter]
    fn code1_report(&self) -> Option<PyPppSsrSignalReport> {
        self.inner
            .code1_report
            .as_ref()
            .map(PyPppSsrSignalReport::code)
    }

    #[getter]
    fn code2_report(&self) -> Option<PyPppSsrSignalReport> {
        self.inner
            .code2_report
            .as_ref()
            .map(PyPppSsrSignalReport::code)
    }

    /// Phase combination status in snake case.
    #[getter]
    fn phase_status(&self) -> String {
        debug_variant_snake(&self.inner.phase_status)
    }

    /// UT1 degrade reason when `phase_status` is `ut1_outside_coverage`.
    #[getter]
    fn phase_status_ut1_reason(&self) -> Option<&'static str> {
        combination_status_ut1_reason(&self.inner.phase_status)
    }

    /// Applied ionosphere-free phase bias, metres.
    #[getter]
    fn applied_phase_if_m(&self) -> Option<f64> {
        self.inner.applied_phase_if_m
    }

    #[getter]
    fn phase1_report(&self) -> Option<PyPppSsrSignalReport> {
        self.inner
            .phase1_report
            .as_ref()
            .map(PyPppSsrSignalReport::phase)
    }

    #[getter]
    fn phase2_report(&self) -> Option<PyPppSsrSignalReport> {
        self.inner
            .phase2_report
            .as_ref()
            .map(PyPppSsrSignalReport::phase)
    }

    fn __repr__(&self) -> String {
        format!(
            "PppSsrObservationApplication(ambiguity_id={:?}, code_status={:?}, phase_status={:?})",
            self.inner.ambiguity_id,
            self.code_status(),
            self.phase_status()
        )
    }
}

/// An observation left out of a PPP solve because an SSR/HAS bias its
/// corrections require was not resolved.
#[pyclass(module = "sidereon._sidereon", name = "PppSsrBiasExclusion")]
#[derive(Clone)]
pub struct PyPppSsrBiasExclusion {
    inner: SsrBiasExclusion,
}

#[pymethods]
impl PyPppSsrBiasExclusion {
    /// Input epoch index of the observation.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    /// Satellite token of the observation.
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    /// Ambiguity id of the observation.
    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    /// A required SSR code bias is absent.
    #[getter]
    fn code_bias_missing(&self) -> bool {
        self.inner.code_bias_missing
    }

    /// A required SSR phase bias is absent.
    #[getter]
    fn phase_bias_missing(&self) -> bool {
        self.inner.phase_bias_missing
    }

    /// Why the recorded biases do not hold at the transmission time.
    #[getter]
    fn transmit_time_failure(&self) -> Option<PyPppSsrTransmitTimeFailure> {
        self.inner
            .transmit_time_failure
            .clone()
            .map(|inner| PyPppSsrTransmitTimeFailure { inner })
    }

    /// The bias application row for the observation.
    #[getter]
    fn application(&self) -> Option<PyPppSsrObservationApplication> {
        self.inner
            .application
            .clone()
            .map(|inner| PyPppSsrObservationApplication { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "PppSsrBiasExclusion(epoch_index={}, ambiguity_id={:?}, code_bias_missing={}, phase_bias_missing={})",
            self.inner.epoch_index,
            self.inner.ambiguity_id,
            self.inner.code_bias_missing,
            self.inner.phase_bias_missing
        )
    }
}

fn exclusion_rows(rows: &[SsrBiasExclusion]) -> Vec<PyPppSsrBiasExclusion> {
    rows.iter()
        .cloned()
        .map(|inner| PyPppSsrBiasExclusion { inner })
        .collect()
}

fn parse_signal_code(field: &str, code: &str) -> PyResult<SignalCode> {
    SignalCode::parse(code).ok_or_else(|| {
        PyValueError::new_err(format!(
            "{field} {code:?} is not a RINEX 3 band and attribute (\"1C\") or observation code (\"C1C\")"
        ))
    })
}

fn signals_from_codes(codes: SignalCodes) -> PyResult<FloatObservationSignals> {
    Ok(FloatObservationSignals {
        code1: parse_signal_code("signals code1", &codes.0)?,
        code2: parse_signal_code("signals code2", &codes.1)?,
        phase1: parse_signal_code("signals phase1", &codes.2)?,
        phase2: parse_signal_code("signals phase2", &codes.3)?,
    })
}
use crate::rtk::PyIntegerStatus;
use crate::{np_array, PySp3};

impl From<PppIntegerStatus> for PyIntegerStatus {
    fn from(status: PppIntegerStatus) -> Self {
        match status {
            PppIntegerStatus::Fixed => PyIntegerStatus::FIXED,
            PppIntegerStatus::NotFixed => PyIntegerStatus::NOT_FIXED,
        }
    }
}

fn mat2_to_array<'py>(py: Python<'py>, matrix: &[[f64; 2]; 2]) -> Bound<'py, PyArray2<f64>> {
    let mut array = Array2::<f64>::zeros((2, 2));
    for row in 0..2 {
        for col in 0..2 {
            array[[row, col]] = matrix[row][col];
        }
    }
    PyArray2::from_owned_array(py, array)
}

// --- input value/config objects -------------------------------------------

/// Civil epoch timestamp for a PPP epoch.
#[pyclass(module = "sidereon._sidereon", name = "PppCivilDateTime")]
#[derive(Clone, Copy)]
pub struct PyPppCivilDateTime {
    inner: CivilDateTime,
}

#[pymethods]
impl PyPppCivilDateTime {
    /// Create a civil timestamp used by the PPP model.
    #[new]
    fn new(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> Self {
        Self {
            inner: CivilDateTime {
                year,
                month,
                day,
                hour,
                minute,
                second,
            },
        }
    }

    #[getter]
    fn year(&self) -> i32 {
        self.inner.year
    }

    #[getter]
    fn month(&self) -> u8 {
        self.inner.month
    }

    #[getter]
    fn day(&self) -> u8 {
        self.inner.day
    }

    #[getter]
    fn hour(&self) -> u8 {
        self.inner.hour
    }

    #[getter]
    fn minute(&self) -> u8 {
        self.inner.minute
    }

    #[getter]
    fn second(&self) -> f64 {
        self.inner.second
    }

    fn __repr__(&self) -> String {
        format!(
            "PppCivilDateTime({:04}-{:02}-{:02} {:02}:{:02}:{:06.3})",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second
        )
    }
}

/// One ionosphere-free code/phase observation in a PPP epoch.
#[pyclass(module = "sidereon._sidereon", name = "PppObservation")]
#[derive(Clone)]
pub struct PyPppObservation {
    inner: FloatObservation,
}

#[pymethods]
impl PyPppObservation {
    /// Create one PPP code/phase observation.
    #[new]
    #[pyo3(signature = (
        satellite_id,
        ambiguity_id,
        code_m,
        phase_m,
        freq1_hz=0.0,
        freq2_hz=0.0,
        glonass_channel=None,
        signals=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        satellite_id: String,
        ambiguity_id: String,
        code_m: f64,
        phase_m: f64,
        freq1_hz: f64,
        freq2_hz: f64,
        glonass_channel: Option<i8>,
        signals: Option<SignalCodes>,
    ) -> PyResult<Self> {
        let sat = GnssSatelliteId::from_str(&satellite_id).map_err(|_| {
            PyValueError::new_err(format!("invalid satellite token: {satellite_id}"))
        })?;
        Ok(Self {
            inner: FloatObservation {
                sat,
                satellite_id,
                ambiguity_id,
                code_m,
                phase_m,
                freq1_hz,
                freq2_hz,
                glonass_channel,
                signals: signals.map(signals_from_codes).transpose()?,
            },
        })
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
    fn freq1_hz(&self) -> f64 {
        self.inner.freq1_hz
    }

    #[getter]
    fn freq2_hz(&self) -> f64 {
        self.inner.freq2_hz
    }

    #[getter]
    fn glonass_channel(&self) -> Option<i8> {
        self.inner.glonass_channel
    }

    /// The tracking codes of the two pseudoranges and two carrier phases as
    /// `(code1, code2, phase1, phase2)` in RINEX 3 band-and-attribute form
    /// (`"1C"`), or `None` when the observation does not state them. An SSR
    /// bias applies only to an observation of the bias's exact signal.
    #[getter]
    fn signals(&self) -> Option<SignalCodes> {
        self.inner.signals.map(|signals| {
            (
                signals.code1.to_string(),
                signals.code2.to_string(),
                signals.phase1.to_string(),
                signals.phase2.to_string(),
            )
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PppObservation(satellite_id={:?}, ambiguity_id={:?})",
            self.inner.satellite_id, self.inner.ambiguity_id
        )
    }
}

impl PyPppObservation {
    fn to_core(&self) -> FloatObservation {
        self.inner.clone()
    }
}

/// One static PPP epoch.
#[pyclass(module = "sidereon._sidereon", name = "PppEpoch")]
#[derive(Clone)]
pub struct PyPppEpoch {
    inner: FloatEpoch,
}

#[pymethods]
impl PyPppEpoch {
    /// Create a PPP epoch.
    #[new]
    #[pyo3(signature = (civil, jd_whole, jd_fraction, t_rx_j2000_s, observations))]
    fn new(
        py: Python<'_>,
        civil: &PyPppCivilDateTime,
        jd_whole: f64,
        jd_fraction: f64,
        t_rx_j2000_s: f64,
        observations: Vec<Py<PyPppObservation>>,
    ) -> Self {
        let observations = observations
            .iter()
            .map(|obs| obs.borrow(py).to_core())
            .collect();
        Self {
            inner: FloatEpoch {
                epoch: civil.inner,
                jd_whole,
                jd_fraction,
                t_rx_j2000_s,
                observations,
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
    fn t_rx_j2000_s(&self) -> f64 {
        self.inner.t_rx_j2000_s
    }

    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.observations.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppEpoch(t_rx_j2000_s={:.3}, observations={})",
            self.inner.t_rx_j2000_s,
            self.inner.observations.len()
        )
    }
}

impl PyPppEpoch {
    fn to_core(&self) -> FloatEpoch {
        self.inner.clone()
    }
}

/// Initial PPP state.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatState")]
#[derive(Clone)]
pub struct PyPppFloatState {
    inner: FloatState,
}

#[pymethods]
impl PyPppFloatState {
    /// Create an initial PPP state.
    #[new]
    #[pyo3(signature = (
        position_m,
        clocks_m,
        ambiguities_m,
        ztd_m=0.0,
        tropo_gradient_north_m=0.0,
        tropo_gradient_east_m=0.0,
        residual_ionosphere_m=None,
    ))]
    fn new(
        position_m: [f64; 3],
        clocks_m: Vec<f64>,
        ambiguities_m: BTreeMap<String, f64>,
        ztd_m: f64,
        tropo_gradient_north_m: f64,
        tropo_gradient_east_m: f64,
        residual_ionosphere_m: Option<BTreeMap<String, f64>>,
    ) -> Self {
        Self {
            inner: FloatState {
                position_m,
                clocks_m,
                ambiguities_m,
                ztd_m,
                tropo_gradient_north_m,
                tropo_gradient_east_m,
                residual_ionosphere_m: residual_ionosphere_m.unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }

    #[getter]
    fn clocks_m(&self) -> Vec<f64> {
        self.inner.clocks_m.clone()
    }

    #[getter]
    fn ambiguities_m(&self) -> BTreeMap<String, f64> {
        self.inner.ambiguities_m.clone()
    }

    #[getter]
    fn ztd_m(&self) -> f64 {
        self.inner.ztd_m
    }

    #[getter]
    fn tropo_gradient_north_m(&self) -> f64 {
        self.inner.tropo_gradient_north_m
    }

    #[getter]
    fn tropo_gradient_east_m(&self) -> f64 {
        self.inner.tropo_gradient_east_m
    }

    #[getter]
    fn residual_ionosphere_m(&self) -> BTreeMap<String, f64> {
        self.inner.residual_ionosphere_m.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatState(position_m=[{:.3}, {:.3}, {:.3}], clocks={})",
            self.inner.position_m[0],
            self.inner.position_m[1],
            self.inner.position_m[2],
            self.inner.clocks_m.len()
        )
    }
}

/// PPP measurement weights.
#[pyclass(module = "sidereon._sidereon", name = "PppMeasurementWeights")]
#[derive(Clone, Copy)]
pub struct PyPppMeasurementWeights {
    inner: MeasurementWeights,
}

#[pymethods]
impl PyPppMeasurementWeights {
    /// Create PPP measurement weights.
    #[new]
    #[pyo3(signature = (code=1.0, phase=100.0, elevation_weighting=false))]
    fn new(code: f64, phase: f64, elevation_weighting: bool) -> Self {
        Self {
            inner: MeasurementWeights {
                code,
                phase,
                elevation_weighting,
            },
        }
    }

    #[getter]
    fn code(&self) -> f64 {
        self.inner.code
    }

    #[getter]
    fn phase(&self) -> f64 {
        self.inner.phase
    }

    #[getter]
    fn elevation_weighting(&self) -> bool {
        self.inner.elevation_weighting
    }

    fn __repr__(&self) -> String {
        format!(
            "PppMeasurementWeights(code={:.3}, phase={:.3}, elevation_weighting={})",
            self.inner.code, self.inner.phase, self.inner.elevation_weighting
        )
    }
}

impl Default for PyPppMeasurementWeights {
    fn default() -> Self {
        Self::new(1.0, 100.0, false)
    }
}

/// PPP troposphere controls.
#[pyclass(module = "sidereon._sidereon", name = "PppTroposphereOptions")]
#[derive(Clone, Copy)]
pub struct PyPppTroposphereOptions {
    inner: TroposphereOptions,
}

#[pymethods]
impl PyPppTroposphereOptions {
    /// Create PPP troposphere controls.
    #[new]
    #[pyo3(signature = (
        enabled=false,
        estimate_ztd=false,
        estimate_tropo_gradients=false,
        pressure_hpa=SurfaceMet::default().pressure_hpa,
        temperature_k=SurfaceMet::default().temperature_k,
        relative_humidity=SurfaceMet::default().relative_humidity,
        vmf1_samples=None,
    ))]
    fn new(
        enabled: bool,
        estimate_ztd: bool,
        estimate_tropo_gradients: bool,
        pressure_hpa: f64,
        temperature_k: f64,
        relative_humidity: f64,
        vmf1_samples: Option<Vec<(f64, f64, f64)>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: if enabled {
                // Default to the climatological Niell mapping; if the caller
                // supplies a 6-hourly VMF1 site-wise `a`-coefficient series
                // (`(mjd, ah, aw)` rows) switch to the VMF1 mapping.
                let mapping = match vmf1_samples {
                    Some(rows) => {
                        let samples: Vec<VmfSiteSample> = rows
                            .into_iter()
                            .map(|(mjd, ah, aw)| VmfSiteSample { mjd, ah, aw })
                            .collect();
                        let series = VmfSiteSeries::new(&samples)
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                        TropoMapping::Vmf1(series)
                    }
                    None => TropoMapping::Niell,
                };
                let met = Met::new(pressure_hpa, temperature_k, relative_humidity)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?;
                let mut tropo = TroposphereOptions::new(met);
                tropo.enabled = true;
                tropo.estimate_ztd = estimate_ztd;
                tropo.estimate_tropo_gradients = estimate_tropo_gradients;
                tropo.met = met;
                tropo.mapping = mapping;
                tropo
            } else {
                TroposphereOptions::disabled()
            },
        })
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }

    #[getter]
    fn estimate_ztd(&self) -> bool {
        self.inner.estimate_ztd
    }

    #[getter]
    fn estimate_tropo_gradients(&self) -> bool {
        self.inner.estimate_tropo_gradients
    }

    #[getter]
    fn pressure_hpa(&self) -> f64 {
        self.inner.met.pressure_hpa
    }

    #[getter]
    fn temperature_k(&self) -> f64 {
        self.inner.met.temperature_k
    }

    #[getter]
    fn relative_humidity(&self) -> f64 {
        self.inner.met.relative_humidity
    }

    /// The tropospheric mapping function in effect: `"niell"` or `"vmf1"`.
    #[getter]
    fn mapping(&self) -> &'static str {
        match self.inner.mapping {
            TropoMapping::Niell => "niell",
            TropoMapping::Vmf1(_) => "vmf1",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PppTroposphereOptions(enabled={}, estimate_ztd={}, mapping={:?})",
            self.inner.enabled,
            self.inner.estimate_ztd,
            self.mapping(),
        )
    }
}

impl Default for PyPppTroposphereOptions {
    fn default() -> Self {
        Self {
            inner: TroposphereOptions::disabled(),
        }
    }
}

/// Iteration and convergence controls for PPP.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatOptions")]
#[derive(Clone, Copy)]
pub struct PyPppFloatOptions {
    inner: FloatSolveOptions,
}

#[pymethods]
impl PyPppFloatOptions {
    /// Create PPP solve controls.
    #[new]
    #[pyo3(signature = (
        max_iterations=MAX_ITERATIONS,
        position_tolerance_m=POSITION_TOLERANCE_M,
        clock_tolerance_m=CLOCK_TOLERANCE_M,
        ambiguity_tolerance_m=AMBIGUITY_TOLERANCE_M,
        ztd_tolerance_m=ZTD_TOLERANCE_M,
    ))]
    fn new(
        max_iterations: usize,
        position_tolerance_m: f64,
        clock_tolerance_m: f64,
        ambiguity_tolerance_m: f64,
        ztd_tolerance_m: f64,
    ) -> Self {
        let mut inner = FloatSolveOptions::default();
        inner.max_iterations = max_iterations;
        inner.position_tolerance_m = position_tolerance_m;
        inner.clock_tolerance_m = clock_tolerance_m;
        inner.ambiguity_tolerance_m = ambiguity_tolerance_m;
        inner.ztd_tolerance_m = ztd_tolerance_m;
        Self { inner }
    }

    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    #[getter]
    fn position_tolerance_m(&self) -> f64 {
        self.inner.position_tolerance_m
    }

    #[getter]
    fn clock_tolerance_m(&self) -> f64 {
        self.inner.clock_tolerance_m
    }

    #[getter]
    fn ambiguity_tolerance_m(&self) -> f64 {
        self.inner.ambiguity_tolerance_m
    }

    #[getter]
    fn ztd_tolerance_m(&self) -> f64 {
        self.inner.ztd_tolerance_m
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatOptions(max_iterations={}, position_tolerance_m={:.3e})",
            self.inner.max_iterations, self.inner.position_tolerance_m
        )
    }
}

impl Default for PyPppFloatOptions {
    fn default() -> Self {
        Self::new(
            MAX_ITERATIONS,
            POSITION_TOLERANCE_M,
            CLOCK_TOLERANCE_M,
            AMBIGUITY_TOLERANCE_M,
            ZTD_TOLERANCE_M,
        )
    }
}

/// One ANTEX PCV sample.
#[pyclass(module = "sidereon._sidereon", name = "PppPcvSample")]
#[derive(Clone)]
pub struct PyPppPcvSample {
    pub(crate) inner: PcvSample,
}

#[pymethods]
impl PyPppPcvSample {
    #[new]
    #[pyo3(signature = (zenith_deg, value_m, azimuth_deg=None))]
    fn new(zenith_deg: f64, value_m: f64, azimuth_deg: Option<f64>) -> Self {
        Self {
            inner: PcvSample {
                azimuth_deg,
                zenith_deg,
                value_m,
            },
        }
    }

    /// Zenith angle in degrees.
    #[getter]
    fn zenith_deg(&self) -> f64 {
        self.inner.zenith_deg
    }

    /// Receiver PCV correction in meters.
    #[getter]
    fn value_m(&self) -> f64 {
        self.inner.value_m
    }

    /// Azimuth angle in degrees, or None for a no-azimuth sample.
    #[getter]
    fn azimuth_deg(&self) -> Option<f64> {
        self.inner.azimuth_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "PppPcvSample(zenith_deg={:?}, value_m={:?}, azimuth_deg={:?})",
            self.inner.zenith_deg, self.inner.value_m, self.inner.azimuth_deg
        )
    }
}

/// Receiver antenna calibration at one frequency.
#[pyclass(module = "sidereon._sidereon", name = "PppReceiverAntennaFrequency")]
#[derive(Clone)]
pub struct PyPppReceiverAntennaFrequency {
    pub(crate) inner: ReceiverAntennaFrequency,
}

#[pymethods]
impl PyPppReceiverAntennaFrequency {
    #[new]
    #[pyo3(signature = (label, pco_m, pcv_samples))]
    fn new(label: String, pco_m: [f64; 3], pcv_samples: Vec<PyPppPcvSample>) -> Self {
        Self {
            inner: ReceiverAntennaFrequency {
                label,
                pco_m,
                pcv_samples: pcv_samples.into_iter().map(|s| s.inner).collect(),
            },
        }
    }

    /// Frequency label searched by the configured signal selectors.
    #[getter]
    fn label(&self) -> String {
        self.inner.label.clone()
    }

    /// Receiver phase-center offset in local north/east/up meters.
    #[getter]
    fn pco_m(&self) -> [f64; 3] {
        self.inner.pco_m
    }

    /// ANTEX PCV samples.
    #[getter]
    fn pcv_samples(&self) -> Vec<PyPppPcvSample> {
        self.inner
            .pcv_samples
            .iter()
            .copied()
            .map(|inner| PyPppPcvSample { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppReceiverAntennaFrequency(label={:?}, pco_m={:?}, pcv_samples={})",
            self.inner.label,
            self.inner.pco_m,
            self.inner.pcv_samples.len()
        )
    }
}

/// Receiver antenna correction options.
#[pyclass(module = "sidereon._sidereon", name = "PppReceiverAntennaOptions")]
#[derive(Clone)]
pub struct PyPppReceiverAntennaOptions {
    pub(crate) inner: ReceiverAntennaOptions,
}

#[pymethods]
impl PyPppReceiverAntennaOptions {
    #[new]
    #[pyo3(signature = (freq1_label, freq1_hz, freq2_label, freq2_hz, frequencies))]
    fn new(
        freq1_label: String,
        freq1_hz: f64,
        freq2_label: String,
        freq2_hz: f64,
        frequencies: Vec<PyPppReceiverAntennaFrequency>,
    ) -> Self {
        let core_frequencies = frequencies.into_iter().map(|f| f.inner).collect();
        let inner = ReceiverAntennaOptions::new(
            freq1_label,
            freq1_hz,
            freq2_label,
            freq2_hz,
            core_frequencies,
        );
        Self { inner }
    }

    /// Label selecting the first receiver-frequency calibration record.
    #[getter]
    fn freq1_label(&self) -> String {
        self.inner.freq1_label.clone()
    }

    /// First carrier frequency in hertz.
    #[getter]
    fn freq1_hz(&self) -> f64 {
        self.inner.freq1_hz
    }

    /// Label selecting the second receiver-frequency calibration record.
    #[getter]
    fn freq2_label(&self) -> String {
        self.inner.freq2_label.clone()
    }

    /// Second carrier frequency in hertz.
    #[getter]
    fn freq2_hz(&self) -> f64 {
        self.inner.freq2_hz
    }

    /// Frequency calibration records.
    #[getter]
    fn frequencies(&self) -> Vec<PyPppReceiverAntennaFrequency> {
        self.inner
            .frequencies
            .iter()
            .cloned()
            .map(|inner| PyPppReceiverAntennaFrequency { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppReceiverAntennaOptions(freq1_label={:?}, freq1_hz={:?}, freq2_label={:?}, freq2_hz={:?}, frequencies={})",
            self.inner.freq1_label,
            self.inner.freq1_hz,
            self.inner.freq2_label,
            self.inner.freq2_hz,
            self.inner.frequencies.len()
        )
    }
}

/// Fine satellite clock series, keyed by satellite token.
#[pyclass(module = "sidereon._sidereon", name = "PppSatelliteClockCorrections")]
#[derive(Clone)]
pub struct PyPppSatelliteClockCorrections {
    pub(crate) inner: SatelliteClockCorrections,
}

#[pymethods]
impl PyPppSatelliteClockCorrections {
    #[new]
    #[pyo3(signature = (series))]
    fn new(series: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut core_series = BTreeMap::new();
        for (key, val) in series.iter() {
            let token: String = key.extract()?;
            let sat = parse_sat(&token)?;
            let records: Vec<(f64, f64)> = val.extract()?;
            core_series.insert(sat, records);
        }
        Ok(Self {
            inner: SatelliteClockCorrections {
                series: core_series,
            },
        })
    }

    /// Per-satellite clock records as (GPS seconds, clock bias seconds).
    #[getter]
    fn series<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (sat, records) in &self.inner.series {
            dict.set_item(sat.to_string(), records.clone())?;
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "PppSatelliteClockCorrections(satellites={})",
            self.inner.series.len()
        )
    }
}

/// Indexed static PPP correction lookup tables.
#[pyclass(module = "sidereon._sidereon", name = "PppCorrectionLookup")]
#[derive(Clone)]
pub struct PyPppCorrectionLookup {
    pub(crate) inner: PppCorrectionLookup,
}

#[pymethods]
impl PyPppCorrectionLookup {
    /// Create PPP correction lookup tables.
    #[new]
    #[pyo3(signature = (
        tide=None,
        pole_tide=None,
        ocean_loading=None,
        windup_m=None,
        sat_pcv_m=None,
        code_bias_m=None,
        sat_pco_ecef=None,
        ssr_code_bias_m=None,
        phase_bias_m=None,
        tide_enabled=false,
        pole_tide_enabled=false,
        ocean_loading_enabled=false,
        windup_enabled=false,
        satellite_antenna_enabled=false,
        code_bias_enabled=false,
        ssr_code_bias_enabled=false,
        phase_bias_enabled=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        tide: Option<BTreeMap<usize, [f64; 3]>>,
        pole_tide: Option<BTreeMap<usize, [f64; 3]>>,
        ocean_loading: Option<BTreeMap<usize, [f64; 3]>>,
        windup_m: Option<BTreeMap<(String, usize), f64>>,
        sat_pcv_m: Option<BTreeMap<(String, usize), f64>>,
        code_bias_m: Option<BTreeMap<(String, usize), f64>>,
        sat_pco_ecef: Option<BTreeMap<(String, usize), [f64; 3]>>,
        ssr_code_bias_m: Option<BTreeMap<(String, usize, String), f64>>,
        phase_bias_m: Option<BTreeMap<(String, usize, String), f64>>,
        tide_enabled: bool,
        pole_tide_enabled: bool,
        ocean_loading_enabled: bool,
        windup_enabled: bool,
        satellite_antenna_enabled: bool,
        code_bias_enabled: bool,
        ssr_code_bias_enabled: bool,
        phase_bias_enabled: bool,
    ) -> PyResult<Self> {
        let mut core_windup_m = BTreeMap::new();
        if let Some(map) = windup_m {
            for ((sat_str, epoch), val) in map {
                let sat = parse_sat(&sat_str)?;
                core_windup_m.insert((sat, epoch), val);
            }
        }
        let mut core_sat_pcv_m = BTreeMap::new();
        if let Some(map) = sat_pcv_m {
            for ((sat_str, epoch), val) in map {
                let sat = parse_sat(&sat_str)?;
                core_sat_pcv_m.insert((sat, epoch), val);
            }
        }
        let mut core_code_bias_m = BTreeMap::new();
        if let Some(map) = code_bias_m {
            for ((sat_str, epoch), val) in map {
                let sat = parse_sat(&sat_str)?;
                core_code_bias_m.insert((sat, epoch), val);
            }
        }
        let mut core_sat_pco_ecef = BTreeMap::new();
        if let Some(map) = sat_pco_ecef {
            for ((sat_str, epoch), val) in map {
                let sat = parse_sat(&sat_str)?;
                core_sat_pco_ecef.insert((sat, epoch), val);
            }
        }
        let mut core_ssr_code_bias_m = BTreeMap::new();
        if let Some(map) = ssr_code_bias_m {
            for ((sat_str, epoch, amb), val) in map {
                let sat = parse_sat(&sat_str)?;
                core_ssr_code_bias_m.insert((sat, epoch, amb), val);
            }
        }
        let mut core_phase_bias_m = BTreeMap::new();
        if let Some(map) = phase_bias_m {
            for ((sat_str, epoch, amb), val) in map {
                let sat = parse_sat(&sat_str)?;
                core_phase_bias_m.insert((sat, epoch, amb), val);
            }
        }
        Ok(Self {
            inner: PppCorrectionLookup {
                tide: tide.unwrap_or_default(),
                pole_tide: pole_tide.unwrap_or_default(),
                ocean_loading: ocean_loading.unwrap_or_default(),
                windup_m: core_windup_m,
                sat_pco_ecef: core_sat_pco_ecef,
                sat_pcv_m: core_sat_pcv_m,
                code_bias_m: core_code_bias_m,
                ssr_code_bias_m: core_ssr_code_bias_m,
                phase_bias_m: core_phase_bias_m,
                ssr_code_bias_records: BTreeMap::new(),
                phase_bias_records: BTreeMap::new(),
                tide_enabled,
                pole_tide_enabled,
                ocean_loading_enabled,
                windup_enabled,
                satellite_antenna_enabled,
                code_bias_enabled,
                ssr_code_bias_enabled,
                phase_bias_enabled,
                ssr_bias_report: None,
            },
        })
    }

    /// Convert precomputed correction tables into an indexed lookup.
    #[staticmethod]
    fn from_corrections(corrections: &PyPppCorrections, options: &PyPppCorrectionsOptions) -> Self {
        Self {
            inner: PppCorrectionLookup::from_options(corrections.inner.clone(), &options.inner),
        }
    }

    /// ECEF solid-earth tide vectors keyed by epoch index.
    #[getter]
    fn tide<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (&epoch, &[x, y, z]) in &self.inner.tide {
            dict.set_item(epoch, (x, y, z))?;
        }
        Ok(dict)
    }

    /// ECEF pole-tide vectors keyed by epoch index.
    #[getter]
    fn pole_tide<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (&epoch, &[x, y, z]) in &self.inner.pole_tide {
            dict.set_item(epoch, (x, y, z))?;
        }
        Ok(dict)
    }

    /// ECEF ocean-loading vectors keyed by epoch index.
    #[getter]
    fn ocean_loading<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (&epoch, &[x, y, z]) in &self.inner.ocean_loading {
            dict.set_item(epoch, (x, y, z))?;
        }
        Ok(dict)
    }

    /// Ionosphere-free phase wind-up meters keyed by (satellite, epoch index).
    #[getter]
    fn windup_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for ((sat, epoch), &val) in &self.inner.windup_m {
            dict.set_item((sat.to_string(), *epoch), val)?;
        }
        Ok(dict)
    }

    /// Ionosphere-free satellite PCV meters keyed by (satellite, epoch index).
    #[getter]
    fn sat_pcv_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for ((sat, epoch), &val) in &self.inner.sat_pcv_m {
            dict.set_item((sat.to_string(), *epoch), val)?;
        }
        Ok(dict)
    }

    /// Clock-datum code-bias meters keyed by (satellite, epoch index).
    #[getter]
    fn code_bias_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for ((sat, epoch), &val) in &self.inner.code_bias_m {
            dict.set_item((sat.to_string(), *epoch), val)?;
        }
        Ok(dict)
    }

    /// Ionosphere-free satellite PCO vectors in ECEF meters keyed by (satellite, epoch index).
    #[getter]
    fn sat_pco_ecef<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for ((sat, epoch), &[x, y, z]) in &self.inner.sat_pco_ecef {
            dict.set_item((sat.to_string(), *epoch), (x, y, z))?;
        }
        Ok(dict)
    }

    /// SSR ionosphere-free code bias meters keyed by (satellite, epoch index, ambiguity id).
    #[getter]
    fn ssr_code_bias_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for ((sat, epoch, amb), &val) in &self.inner.ssr_code_bias_m {
            dict.set_item((sat.to_string(), *epoch, amb.clone()), val)?;
        }
        Ok(dict)
    }

    /// SSR ionosphere-free phase bias meters keyed by (satellite, epoch index, ambiguity id).
    #[getter]
    fn phase_bias_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for ((sat, epoch, amb), &val) in &self.inner.phase_bias_m {
            dict.set_item((sat.to_string(), *epoch, amb.clone()), val)?;
        }
        Ok(dict)
    }

    /// Solid-earth tide requirement flag.
    #[getter]
    fn tide_enabled(&self) -> bool {
        self.inner.tide_enabled
    }

    /// Pole-tide requirement flag.
    #[getter]
    fn pole_tide_enabled(&self) -> bool {
        self.inner.pole_tide_enabled
    }

    /// Ocean-loading requirement flag.
    #[getter]
    fn ocean_loading_enabled(&self) -> bool {
        self.inner.ocean_loading_enabled
    }

    /// Phase wind-up requirement flag.
    #[getter]
    fn windup_enabled(&self) -> bool {
        self.inner.windup_enabled
    }

    /// Satellite antenna requirement flag.
    #[getter]
    fn satellite_antenna_enabled(&self) -> bool {
        self.inner.satellite_antenna_enabled
    }

    /// Code-bias requirement flag.
    #[getter]
    fn code_bias_enabled(&self) -> bool {
        self.inner.code_bias_enabled
    }

    /// SSR code-bias requirement flag.
    #[getter]
    fn ssr_code_bias_enabled(&self) -> bool {
        self.inner.ssr_code_bias_enabled
    }

    /// SSR phase-bias requirement flag.
    #[getter]
    fn phase_bias_enabled(&self) -> bool {
        self.inner.phase_bias_enabled
    }

    fn __repr__(&self) -> String {
        format!(
            "PppCorrectionLookup(tide={}, windup_m={}, sat_pco_ecef={}, sat_pcv_m={}, code_bias_m={})",
            self.inner.tide.len(),
            self.inner.windup_m.len(),
            self.inner.sat_pco_ecef.len(),
            self.inner.sat_pcv_m.len(),
            self.inner.code_bias_m.len(),
        )
    }
}

/// Range-correction options and precomputed correction tables.
#[pyclass(module = "sidereon._sidereon", name = "PppRangeCorrections")]
#[derive(Clone)]
pub struct PyPppRangeCorrections {
    pub(crate) inner: RangeCorrections,
}

#[pymethods]
impl PyPppRangeCorrections {
    /// Create range corrections.
    #[new]
    #[pyo3(signature = (receiver_antenna=None, sat_clock_relativity=false, satellite_clock=None, ppp=None))]
    fn new(
        receiver_antenna: Option<PyPppReceiverAntennaOptions>,
        sat_clock_relativity: bool,
        satellite_clock: Option<PyPppSatelliteClockCorrections>,
        ppp: Option<PyPppCorrectionLookup>,
    ) -> Self {
        Self {
            inner: RangeCorrections {
                receiver_antenna: receiver_antenna.map(|a| a.inner),
                sat_clock_relativity,
                satellite_clock: satellite_clock.map(|c| c.inner),
                ppp: ppp.map(|p| p.inner).unwrap_or_default(),
            },
        }
    }

    /// Create an explicit all-off correction set.
    #[staticmethod]
    fn disabled() -> Self {
        Self {
            inner: RangeCorrections::disabled(),
        }
    }

    /// Optional receiver antenna calibration.
    #[getter]
    fn receiver_antenna(&self) -> Option<PyPppReceiverAntennaOptions> {
        self.inner
            .receiver_antenna
            .as_ref()
            .cloned()
            .map(|inner| PyPppReceiverAntennaOptions { inner })
    }

    /// Enables relativistic satellite range correction.
    #[getter]
    fn sat_clock_relativity(&self) -> bool {
        self.inner.sat_clock_relativity
    }

    /// Optional external satellite clock corrections.
    #[getter]
    fn satellite_clock(&self) -> Option<PyPppSatelliteClockCorrections> {
        self.inner
            .satellite_clock
            .as_ref()
            .cloned()
            .map(|inner| PyPppSatelliteClockCorrections { inner })
    }

    /// Precomputed PPP correction tables.
    #[getter]
    fn ppp(&self) -> PyPppCorrectionLookup {
        PyPppCorrectionLookup {
            inner: self.inner.ppp.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PppRangeCorrections(receiver_antenna={}, sat_clock_relativity={}, satellite_clock={})",
            self.inner.receiver_antenna.is_some(),
            self.inner.sat_clock_relativity,
            self.inner.satellite_clock.is_some()
        )
    }
}

/// Complete typed configuration for a PPP float solve.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatConfig")]
pub struct PyPppFloatConfig {
    inner: FloatSolveConfig,
}

#[pymethods]
impl PyPppFloatConfig {
    /// Create a PPP float solve configuration.
    #[new]
    #[pyo3(signature = (
        weights=None,
        tropo=None,
        options=None,
        residual_screen=false,
        elevation_cutoff_deg=None,
        estimate_residual_ionosphere=false,
        corrections=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        weights: Option<Py<PyPppMeasurementWeights>>,
        tropo: Option<Py<PyPppTroposphereOptions>>,
        options: Option<Py<PyPppFloatOptions>>,
        residual_screen: bool,
        elevation_cutoff_deg: Option<f64>,
        estimate_residual_ionosphere: bool,
        corrections: Option<Py<PyPppRangeCorrections>>,
    ) -> Self {
        let weights = option_py_or_default(
            py,
            weights.as_ref(),
            |value| value.inner,
            || PyPppMeasurementWeights::default().inner,
        );
        let tropo = option_py_or_default(
            py,
            tropo.as_ref(),
            |value| value.inner,
            || PyPppTroposphereOptions::default().inner,
        );
        let opts = option_py_or_default(
            py,
            options.as_ref(),
            |value| value.inner,
            || PyPppFloatOptions::default().inner,
        );
        let corrections = option_py_or_default(
            py,
            corrections.as_ref(),
            |value| value.inner.clone(),
            RangeCorrections::disabled,
        );
        let mut inner = FloatSolveConfig::new(
            weights,
            tropo,
            corrections.clone(),
            opts,
            elevation_cutoff_deg,
            residual_screen,
            estimate_residual_ionosphere,
        );
        inner.weights = weights;
        inner.tropo = tropo;
        inner.corrections = corrections;
        inner.opts = opts;
        inner.elevation_cutoff_deg = elevation_cutoff_deg;
        inner.residual_screen = residual_screen;
        inner.estimate_residual_ionosphere = estimate_residual_ionosphere;
        Self { inner }
    }

    #[getter]
    fn residual_screen(&self) -> bool {
        self.inner.residual_screen
    }

    #[getter]
    fn elevation_cutoff_deg(&self) -> Option<f64> {
        self.inner.elevation_cutoff_deg
    }

    #[getter]
    fn estimate_residual_ionosphere(&self) -> bool {
        self.inner.estimate_residual_ionosphere
    }

    /// Code and phase weights of the solve.
    #[getter]
    fn weights(&self) -> PyPppMeasurementWeights {
        PyPppMeasurementWeights {
            inner: self.inner.weights,
        }
    }

    /// Troposphere model and estimation options of the solve.
    #[getter]
    fn tropo(&self) -> PyPppTroposphereOptions {
        PyPppTroposphereOptions {
            inner: self.inner.tropo,
        }
    }

    /// Iteration cap and state tolerances of the solve.
    #[getter]
    fn options(&self) -> PyPppFloatOptions {
        PyPppFloatOptions {
            inner: self.inner.opts,
        }
    }

    /// Range corrections applied during the float solve.
    #[getter]
    fn corrections(&self) -> PyPppRangeCorrections {
        PyPppRangeCorrections {
            inner: self.inner.corrections.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatConfig(residual_screen={}, max_iterations={})",
            self.inner.residual_screen, self.inner.opts.max_iterations
        )
    }
}

/// Integer ambiguity controls for PPP fixed solving.
#[pyclass(module = "sidereon._sidereon", name = "PppFixedAmbiguityOptions")]
pub struct PyPppFixedAmbiguityOptions {
    inner: FixedAmbiguityOptions,
}

#[pymethods]
impl PyPppFixedAmbiguityOptions {
    /// Create PPP fixed ambiguity-search controls.
    #[new]
    #[pyo3(signature = (wavelengths_m, offsets_m, ratio_threshold=RATIO_THRESHOLD))]
    fn new(
        wavelengths_m: BTreeMap<String, f64>,
        offsets_m: BTreeMap<String, f64>,
        ratio_threshold: f64,
    ) -> Self {
        let mut inner = FixedAmbiguityOptions::new(ratio_threshold);
        inner.wavelengths_m = wavelengths_m;
        inner.offsets_m = offsets_m;
        inner.ratio_threshold = ratio_threshold;
        Self { inner }
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
    fn ratio_threshold(&self) -> f64 {
        self.inner.ratio_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFixedAmbiguityOptions(ambiguities={}, ratio_threshold={:.3})",
            self.inner.wavelengths_m.len(),
            self.inner.ratio_threshold
        )
    }
}

/// Complete typed configuration for a PPP fixed solve.
#[pyclass(module = "sidereon._sidereon", name = "PppFixedConfig")]
pub struct PyPppFixedConfig {
    inner: FixedSolveConfig,
}

#[pymethods]
impl PyPppFixedConfig {
    /// Create a PPP fixed solve configuration.
    #[new]
    #[pyo3(signature = (
        ambiguity,
        weights=None,
        tropo=None,
        options=None,
        elevation_cutoff_deg=None,
        estimate_residual_ionosphere=false,
        corrections=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        ambiguity: &PyPppFixedAmbiguityOptions,
        weights: Option<Py<PyPppMeasurementWeights>>,
        tropo: Option<Py<PyPppTroposphereOptions>>,
        options: Option<Py<PyPppFloatOptions>>,
        elevation_cutoff_deg: Option<f64>,
        estimate_residual_ionosphere: bool,
        corrections: Option<Py<PyPppRangeCorrections>>,
    ) -> Self {
        let weights = option_py_or_default(
            py,
            weights.as_ref(),
            |value| value.inner,
            || PyPppMeasurementWeights::default().inner,
        );
        let tropo = option_py_or_default(
            py,
            tropo.as_ref(),
            |value| value.inner,
            || PyPppTroposphereOptions::default().inner,
        );
        let opts = option_py_or_default(
            py,
            options.as_ref(),
            |value| value.inner,
            || PyPppFloatOptions::default().inner,
        );
        let corrections = option_py_or_default(
            py,
            corrections.as_ref(),
            |value| value.inner.clone(),
            RangeCorrections::disabled,
        );
        let mut inner = FixedSolveConfig::new(
            weights,
            tropo,
            corrections.clone(),
            opts,
            elevation_cutoff_deg,
            ambiguity.inner.clone(),
            estimate_residual_ionosphere,
        );
        inner.weights = weights;
        inner.tropo = tropo;
        inner.corrections = corrections;
        inner.opts = opts;
        inner.elevation_cutoff_deg = elevation_cutoff_deg;
        inner.ambiguity = ambiguity.inner.clone();
        inner.estimate_residual_ionosphere = estimate_residual_ionosphere;
        Self { inner }
    }

    #[getter]
    fn elevation_cutoff_deg(&self) -> Option<f64> {
        self.inner.elevation_cutoff_deg
    }

    #[getter]
    fn estimate_residual_ionosphere(&self) -> bool {
        self.inner.estimate_residual_ionosphere
    }

    /// Code and phase weights of the solve.
    #[getter]
    fn weights(&self) -> PyPppMeasurementWeights {
        PyPppMeasurementWeights {
            inner: self.inner.weights,
        }
    }

    /// Troposphere model and estimation options of the solve.
    #[getter]
    fn tropo(&self) -> PyPppTroposphereOptions {
        PyPppTroposphereOptions {
            inner: self.inner.tropo,
        }
    }

    /// Iteration cap and state tolerances of the solve.
    #[getter]
    fn options(&self) -> PyPppFloatOptions {
        PyPppFloatOptions {
            inner: self.inner.opts,
        }
    }

    /// Integer ambiguity resolution options of the solve.
    #[getter]
    fn ambiguity(&self) -> PyPppFixedAmbiguityOptions {
        PyPppFixedAmbiguityOptions {
            inner: self.inner.ambiguity.clone(),
        }
    }

    /// Range corrections applied during the fixed solve.
    #[getter]
    fn corrections(&self) -> PyPppRangeCorrections {
        PyPppRangeCorrections {
            inner: self.inner.corrections.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFixedConfig(ambiguities={}, max_iterations={})",
            self.inner.ambiguity.wavelengths_m.len(),
            self.inner.opts.max_iterations
        )
    }
}

// --- result object ---------------------------------------------------------

/// Temporal-correlation summary for a static PPP residual sequence.
#[pyclass(module = "sidereon._sidereon", name = "PppTemporalCorrelationSummary")]
#[derive(Clone, Copy)]
pub struct PyPppTemporalCorrelationSummary {
    inner: TemporalCorrelationSummary,
}

impl From<TemporalCorrelationSummary> for PyPppTemporalCorrelationSummary {
    fn from(inner: TemporalCorrelationSummary) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPppTemporalCorrelationSummary {
    #[getter]
    fn lag1_autocorrelation(&self) -> f64 {
        self.inner.lag1_autocorrelation
    }

    #[getter]
    fn decorrelation_time_epochs(&self) -> f64 {
        self.inner.decorrelation_time_epochs
    }

    #[getter]
    fn decorrelation_time_s(&self) -> Option<f64> {
        self.inner.decorrelation_time_s
    }

    #[getter]
    fn nominal_sample_count(&self) -> usize {
        self.inner.nominal_sample_count
    }

    #[getter]
    fn effective_sample_count(&self) -> f64 {
        self.inner.effective_sample_count
    }

    #[getter]
    fn variance_inflation_factor(&self) -> f64 {
        self.inner.variance_inflation_factor
    }

    #[getter]
    fn arcs_used(&self) -> usize {
        self.inner.arcs_used
    }

    fn __repr__(&self) -> String {
        format!(
            "PppTemporalCorrelationSummary(lag1_autocorrelation={:.3}, variance_inflation_factor={:.3})",
            self.inner.lag1_autocorrelation, self.inner.variance_inflation_factor
        )
    }
}

/// Static float PPP solution.
///
/// `position` is the receiver ECEF position as a numpy `float64` array of shape
/// `(3,)`, metres. Float ambiguities and residual RMS values are in metres.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatSolution")]
pub struct PyPppFloatSolution {
    inner: FloatSolution,
}

#[pymethods]
impl PyPppFloatSolution {
    /// Receiver clock of each solved epoch, metres, in `solved_epoch_indices`
    /// order.
    #[getter]
    fn epoch_clocks_m(&self) -> Vec<f64> {
        self.inner.epoch_clocks_m.clone()
    }

    /// The input epoch index of each solved epoch. An epoch left with no
    /// observations, by the caller, the elevation cutoff or the residual
    /// screen, is not solved.
    #[getter]
    fn solved_epoch_indices(&self) -> Vec<usize> {
        self.inner.solved_epoch_indices.clone()
    }

    /// Observations left out before the solve because no transmission epoch
    /// can be placed from them, each as a `PppUnplacedObservation`.
    #[getter]
    fn unplaced_observations(&self) -> Vec<PppUnplacedObservationRow> {
        unplaced_rows(&self.inner.unplaced_observations)
    }

    /// Observations the residual screen removed, as `(input epoch index,
    /// ambiguity id)`.
    #[getter]
    fn residual_screen_removals(&self) -> Vec<(usize, String)> {
        self.inner.residual_screen_removals.clone()
    }

    /// Whether the residual screen ran.
    #[getter]
    fn residual_screen(&self) -> bool {
        self.inner.residual_screen
    }

    /// Residual rows in epoch and observation order, each a
    /// `PppFloatResidual`.
    #[getter]
    fn residuals_m(&self) -> Vec<PyPppFloatResidual> {
        residual_rows(&self.inner.residuals_m)
    }

    /// Whether the solve met its state tolerances or reached its iteration
    /// cap.
    #[getter]
    fn status(&self) -> PyPppFloatStatus {
        self.inner.status.into()
    }

    /// Observations left out because an SSR/HAS bias the corrections require
    /// was not resolved, each a `PppSsrBiasExclusion`.
    #[getter]
    fn ssr_bias_exclusions(&self) -> Vec<PyPppSsrBiasExclusion> {
        exclusion_rows(&self.inner.ssr_bias_exclusions)
    }

    /// The iteration and convergence options the solve ran with.
    #[getter]
    fn solve_options(&self) -> PyPppFloatOptions {
        PyPppFloatOptions {
            inner: self.inner.solve_options,
        }
    }

    /// ECEF position as a numpy array `[x_m, y_m, z_m]`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_m)
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }

    /// Float ambiguities in metres, keyed by ambiguity id.
    #[getter]
    fn ambiguities_m(&self) -> BTreeMap<String, f64> {
        self.inner.ambiguities_m.clone()
    }

    #[getter]
    fn ztd_residual_m(&self) -> Option<f64> {
        self.inner.ztd_residual_m
    }

    #[getter]
    fn residual_ionosphere_m(&self) -> BTreeMap<String, f64> {
        self.inner.residual_ionosphere_m.clone()
    }

    #[getter]
    fn tropo_gradient_north_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_north_m
    }

    #[getter]
    fn tropo_gradient_east_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_east_m
    }

    #[getter]
    fn tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn formal_tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .formal_tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn position_covariance_ecef_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.ecef_m2)
    }

    #[getter]
    fn position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.enu_m2)
    }

    #[getter]
    fn formal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.ecef_m2)
    }

    #[getter]
    fn formal_position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.enu_m2)
    }

    #[getter]
    fn posterior_variance_factor(&self) -> f64 {
        self.inner.posterior_variance_factor
    }

    #[getter]
    fn position_covariance_scale_factor(&self) -> f64 {
        self.inner.position_covariance_scale_factor
    }

    #[getter]
    fn temporal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.ecef_m2)
    }

    #[getter]
    fn temporal_position_covariance_enu_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.enu_m2)
    }

    #[getter]
    fn temporal_position_covariance_scale_factor(&self) -> f64 {
        self.inner.temporal_position_covariance_scale_factor
    }

    #[getter]
    fn temporal_correlation(&self) -> PyPppTemporalCorrelationSummary {
        self.inner.temporal_correlation.into()
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
    fn used_sats(&self) -> Vec<String> {
        self.inner.used_sats.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatSolution(position=[{:.3}, {:.3}, {:.3}], phase_rms_m={:.4}, converged={})",
            self.inner.position_m[0],
            self.inner.position_m[1],
            self.inner.position_m[2],
            self.inner.phase_rms_m,
            self.inner.converged
        )
    }
}

/// Static integer-fixed PPP solution.
///
/// `position` is the receiver ECEF position as a numpy `float64` array of shape
/// `(3,)`, metres. Fixed ambiguities are exposed in cycles and metres.
#[pyclass(module = "sidereon._sidereon", name = "PppFixedSolution")]
pub struct PyPppFixedSolution {
    inner: FixedSolution,
}

#[pymethods]
impl PyPppFixedSolution {
    /// Receiver clock of each solved epoch, metres, in `solved_epoch_indices`
    /// order.
    #[getter]
    fn epoch_clocks_m(&self) -> Vec<f64> {
        self.inner.epoch_clocks_m.clone()
    }

    /// The input epoch index of each solved epoch. An epoch left with no
    /// observations, by the caller, the elevation cutoff or the residual
    /// screen, is not solved.
    #[getter]
    fn solved_epoch_indices(&self) -> Vec<usize> {
        self.inner.solved_epoch_indices.clone()
    }

    /// Observations left out before the solve because no transmission epoch
    /// can be placed from them, each as a `PppUnplacedObservation`.
    #[getter]
    fn unplaced_observations(&self) -> Vec<PppUnplacedObservationRow> {
        unplaced_rows(&self.inner.unplaced_observations)
    }

    /// Residual rows in epoch and observation order, each a
    /// `PppFloatResidual`.
    #[getter]
    fn residuals_m(&self) -> Vec<PyPppFloatResidual> {
        residual_rows(&self.inner.residuals_m)
    }

    /// Whether the solve met its state tolerances or reached its iteration
    /// cap.
    #[getter]
    fn status(&self) -> PyPppFloatStatus {
        self.inner.status.into()
    }

    /// Observations left out because an SSR/HAS bias the corrections require
    /// was not resolved, each a `PppSsrBiasExclusion`.
    #[getter]
    fn ssr_bias_exclusions(&self) -> Vec<PyPppSsrBiasExclusion> {
        exclusion_rows(&self.inner.ssr_bias_exclusions)
    }

    /// ECEF position as a numpy array `[x_m, y_m, z_m]`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_m)
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }

    /// Fixed ambiguities in cycles, keyed by ambiguity id.
    #[getter]
    fn fixed_ambiguities_cycles(&self) -> BTreeMap<String, i64> {
        self.inner.fixed_ambiguities_cycles.clone()
    }

    /// Fixed ambiguities in metres, keyed by ambiguity id.
    #[getter]
    fn fixed_ambiguities_m(&self) -> BTreeMap<String, f64> {
        self.inner.fixed_ambiguities_m.clone()
    }

    #[getter]
    fn ztd_residual_m(&self) -> Option<f64> {
        self.inner.ztd_residual_m
    }

    #[getter]
    fn residual_ionosphere_m(&self) -> BTreeMap<String, f64> {
        self.inner.residual_ionosphere_m.clone()
    }

    #[getter]
    fn tropo_gradient_north_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_north_m
    }

    #[getter]
    fn tropo_gradient_east_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_east_m
    }

    #[getter]
    fn tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn formal_tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .formal_tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn position_covariance_ecef_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.ecef_m2)
    }

    #[getter]
    fn position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.enu_m2)
    }

    #[getter]
    fn formal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.ecef_m2)
    }

    #[getter]
    fn formal_position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.enu_m2)
    }

    #[getter]
    fn posterior_variance_factor(&self) -> f64 {
        self.inner.posterior_variance_factor
    }

    #[getter]
    fn position_covariance_scale_factor(&self) -> f64 {
        self.inner.position_covariance_scale_factor
    }

    #[getter]
    fn temporal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.ecef_m2)
    }

    #[getter]
    fn temporal_position_covariance_enu_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.enu_m2)
    }

    #[getter]
    fn temporal_position_covariance_scale_factor(&self) -> f64 {
        self.inner.temporal_position_covariance_scale_factor
    }

    #[getter]
    fn temporal_correlation(&self) -> PyPppTemporalCorrelationSummary {
        self.inner.temporal_correlation.into()
    }

    /// The float solution that seeded integer search.
    #[getter]
    fn float_solution(&self) -> PyPppFloatSolution {
        PyPppFloatSolution {
            inner: self.inner.float_solution.clone(),
        }
    }

    #[getter]
    fn integer_status(&self) -> PyIntegerStatus {
        self.inner.integer.integer_status.into()
    }

    #[getter]
    fn integer_ratio(&self) -> f64 {
        self.inner.integer.integer_ratio
    }

    #[getter]
    fn integer_candidates(&self) -> usize {
        self.inner.integer.integer_candidates
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
    fn used_sats(&self) -> Vec<String> {
        self.inner.used_sats.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFixedSolution(position=[{:.3}, {:.3}, {:.3}], integer_status={:?}, converged={})",
            self.inner.position_m[0],
            self.inner.position_m[1],
            self.inner.position_m[2],
            self.inner.integer.integer_status,
            self.inner.converged
        )
    }
}

/// SPP-seeded auto-initialization policy for the raw-epochs PPP drivers.
///
/// With no explicit guess the driver seeds the static float state from a
/// per-epoch single-point-positioning solve (ionosphere off, optional
/// troposphere) using `spp_initial_guess` as the cold start. Supplying
/// `initial_guess_position_m` skips the SPP/mean stages and uses that position
/// (with `initial_guess_clock_m`, duplicated across epochs) directly.
#[pyclass(module = "sidereon._sidereon", name = "PppAutoInitOptions")]
#[derive(Clone, Copy, Default)]
pub struct PyPppAutoInitOptions {
    inner: PppAutoInitOptions,
}

#[pymethods]
impl PyPppAutoInitOptions {
    /// Create PPP auto-init controls.
    #[new]
    #[pyo3(signature = (
        spp_initial_guess=[0.0; 4],
        spp_troposphere=false,
        pressure_hpa=SurfaceMet::default().pressure_hpa,
        temperature_k=SurfaceMet::default().temperature_k,
        relative_humidity=SurfaceMet::default().relative_humidity,
        initial_guess_position_m=None,
        initial_guess_clock_m=0.0,
    ))]
    fn new(
        spp_initial_guess: [f64; 4],
        spp_troposphere: bool,
        pressure_hpa: f64,
        temperature_k: f64,
        relative_humidity: f64,
        initial_guess_position_m: Option<[f64; 3]>,
        initial_guess_clock_m: f64,
    ) -> Self {
        let initial_guess = initial_guess_position_m.map(|position_m| PppInitialGuess {
            position_m,
            clock_m: initial_guess_clock_m,
        });
        let mut inner = PppAutoInitOptions::default();
        inner.initial_guess = initial_guess;
        inner.spp_initial_guess = spp_initial_guess;
        inner.spp_troposphere = spp_troposphere;
        inner.spp_met = SurfaceMet {
            pressure_hpa,
            temperature_k,
            relative_humidity,
        };
        Self { inner }
    }

    #[getter]
    fn spp_initial_guess(&self) -> [f64; 4] {
        self.inner.spp_initial_guess
    }

    #[getter]
    fn spp_troposphere(&self) -> bool {
        self.inner.spp_troposphere
    }

    #[getter]
    fn initial_guess_position_m(&self) -> Option<[f64; 3]> {
        self.inner.initial_guess.map(|guess| guess.position_m)
    }

    #[getter]
    fn initial_guess_clock_m(&self) -> Option<f64> {
        self.inner.initial_guess.map(|guess| guess.clock_m)
    }

    fn __repr__(&self) -> String {
        format!(
            "PppAutoInitOptions(spp_troposphere={}, has_initial_guess={})",
            self.inner.spp_troposphere,
            self.inner.initial_guess.is_some()
        )
    }
}

// --- solve entry points ----------------------------------------------------

#[pyfunction]
#[pyo3(signature = (sp3, epochs, initial_state, config))]
fn solve_ppp_float(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    initial_state: &PyPppFloatState,
    config: &PyPppFloatConfig,
) -> PyResult<PyPppFloatSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let inner = sidereon_core::precise_positioning::solve_float_epochs(
        &sp3.inner,
        &epochs,
        initial_state.inner.clone(),
        config.inner.clone(),
    )
    .map_err(|error| crate::solve_error_detail::ppp_float_error(py, error))?;
    Ok(PyPppFloatSolution { inner })
}

#[pyfunction]
#[pyo3(signature = (sp3, epochs, float_solution, config))]
fn solve_ppp_fixed(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    float_solution: &PyPppFloatSolution,
    config: &PyPppFixedConfig,
) -> PyResult<PyPppFixedSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let inner = sidereon_core::precise_positioning::solve_fixed_from_float(
        &sp3.inner,
        &epochs,
        float_solution.inner.clone(),
        config.inner.clone(),
    )
    .map_err(|error| crate::solve_error_detail::ppp_fixed_error(py, error))?;
    Ok(PyPppFixedSolution { inner })
}

/// Solve a static multi-epoch float PPP arc from raw epochs, auto-initializing
/// the float state from the SPP seed described by `options`.
///
/// Unlike [`solve_ppp_float`], no explicit initial `FloatState` is supplied: the
/// driver seeds it (per-epoch SPP position/clock, phase-minus-code ambiguities,
/// zero ZTD) and then runs the same static float solve. The SP3 product is both
/// the SPP seed ephemeris and the PPP observable ephemeris. Raises `SolveError`
/// on a seed or float-solve failure.
#[pyfunction]
#[pyo3(signature = (sp3, epochs, config, options=None))]
fn solve_ppp_auto_init_float(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    config: &PyPppFloatConfig,
    options: Option<Py<PyPppAutoInitOptions>>,
) -> PyResult<PyPppFloatSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |value| value.inner,
        || PyPppAutoInitOptions::default().inner,
    );
    let inner = core_solve_ppp_auto_init_float(&sp3.inner, &epochs, options, config.inner.clone())
        .map_err(|error| crate::solve_error_detail::ppp_auto_init_error(py, error))?;
    Ok(PyPppFloatSolution { inner })
}

/// Solve a static integer-fixed PPP arc from raw epochs: auto-init seed, the
/// float solve, then the LAMBDA integer fix and ambiguity-conditioned re-solve.
///
/// This is the auto-initialized counterpart of [`solve_ppp_fixed`]: the float
/// arc is seeded from the SPP auto-init `options` (no explicit `FloatState` or
/// float solution is supplied), then the integer search and fixed re-solve run.
/// Raises `SolveError` on a seed, float-solve, or fixed-solve failure.
#[pyfunction]
#[pyo3(signature = (sp3, epochs, float_config, fixed_config, options=None))]
fn solve_ppp_auto_init_fixed(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    float_config: &PyPppFloatConfig,
    fixed_config: &PyPppFixedConfig,
    options: Option<Py<PyPppAutoInitOptions>>,
) -> PyResult<PyPppFixedSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |value| value.inner,
        || PyPppAutoInitOptions::default().inner,
    );
    let inner = core_solve_ppp_auto_init_fixed(
        &sp3.inner,
        &epochs,
        options,
        float_config.inner.clone(),
        fixed_config.inner.clone(),
    )
    .map_err(|error| crate::solve_error_detail::ppp_auto_init_error(py, error))?;
    Ok(PyPppFixedSolution { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPppFloatSolution>()?;
    m.add_class::<PppUnplacedObservationRow>()?;
    m.add_class::<PyPppFloatStatus>()?;
    m.add_class::<PyPppFloatResidual>()?;
    m.add_class::<PyPppSsrTransmitTimeFailure>()?;
    m.add_class::<PyPppSsrSignalReport>()?;
    m.add_class::<PyPppSsrObservationApplication>()?;
    m.add_class::<PyPppSsrBiasExclusion>()?;
    m.add_class::<PyPppTemporalCorrelationSummary>()?;
    m.add_function(wrap_pyfunction!(solve_ppp_float, m)?)?;
    m.add_class::<PyPppCivilDateTime>()?;
    m.add_class::<PyPppObservation>()?;
    m.add_class::<PyPppEpoch>()?;
    m.add_class::<PyPppFloatState>()?;
    m.add_class::<PyPppMeasurementWeights>()?;
    m.add_class::<PyPppTroposphereOptions>()?;
    m.add_class::<PyPppFloatOptions>()?;
    m.add_class::<PyPppPcvSample>()?;
    m.add_class::<PyPppReceiverAntennaFrequency>()?;
    m.add_class::<PyPppReceiverAntennaOptions>()?;
    m.add_class::<PyPppSatelliteClockCorrections>()?;
    m.add_class::<PyPppCorrectionLookup>()?;
    m.add_class::<PyPppRangeCorrections>()?;
    m.add_class::<PyPppFloatConfig>()?;
    m.add_class::<PyPppFixedAmbiguityOptions>()?;
    m.add_class::<PyPppFixedConfig>()?;
    m.add_class::<PyPppFixedSolution>()?;
    m.add_function(wrap_pyfunction!(solve_ppp_fixed, m)?)?;
    m.add_class::<PyPppAutoInitOptions>()?;
    m.add_function(wrap_pyfunction!(solve_ppp_auto_init_float, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ppp_auto_init_fixed, m)?)?;
    Ok(())
}
