//! Precise-product utilities beyond single-file SP3 parsing.
//!
//! This module marshals Python inputs into core precise-product APIs. The SP3
//! merge path returns the existing `Sp3` binding.

use std::collections::BTreeSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};
use serde::Deserialize;

use sidereon_core::astro::time::civil::seconds_between_splits;
use sidereon_core::astro::time::{Instant, InstantRepr};
use sidereon_core::constants::J2000_JD;
use sidereon_core::data::ArchiveCompression;
use sidereon_core::ephemeris::{
    merge, AgreementMetric, CellProvenance, CellSelection, ClockOmission, ClockOmissionReason,
    ContinuityDefect, ContinuityOptions, ContributorCoverage, DroppedEpochReason,
    DroppedInputEpoch, EpochAgreement, EpochWindow, InterpolationNodes, MergeCombine,
    MergeContinuityCell, MergeContinuityCellRole, MergeContinuityReport, MergeContinuityViolation,
    MergeFlag, MergeOptions, MergePrecedenceScope, MergeProvenance, MergeReport, OrbitClass,
    OutlierRejectOptions, PrecedenceTransition, ProvenanceMode, Sp3ArtifactIdentity,
    Sp3ChannelCoverage, Sp3Coverage, Sp3CoverageGap, Sp3CoverageSpan, Sp3EpochGrid,
    Sp3EpochIntervalError, Sp3FrameLabelSet, Sp3FrameReconciliation, Sp3FrameReconciliationOptions,
    Sp3MergeInputIdentity, Sp3MergeInputIdentityError, Sp3SatelliteCoverage, TransitionReason,
};
use sidereon_core::Error as CoreError;
use sidereon_core::GnssSystem;

use crate::ephemeris::{continuity_options, continuity_verdict_to_py, parse_sat};
use crate::marshal::option_py_or_default;
use crate::rinex_clock::PyClockInstant;
use crate::PySp3;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sp3ArtifactIdentityInput {
    schema_version: u8,
    requested_identity: serde_json::Value,
    resolved_identity: serde_json::Value,
    distribution_source: String,
    official_filename: String,
    product_sha256: String,
    product_byte_length: u64,
    archive_sha256: String,
    archive_byte_length: u64,
    compression: String,
}

fn artifact_identity(json: &str) -> PyResult<Sp3ArtifactIdentity> {
    let input: Sp3ArtifactIdentityInput =
        serde_json::from_str(json).map_err(|error| PyValueError::new_err(error.to_string()))?;
    if input.schema_version != 1 {
        return Err(PyValueError::new_err(
            "unsupported SP3 artifact identity schema version",
        ));
    }
    let requested_json = serde_json::to_string(&input.requested_identity)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let resolved_json = serde_json::to_string(&input.resolved_identity)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let compression = match input.compression.as_str() {
        "gzip" => ArchiveCompression::Gzip,
        "unix_compress" => ArchiveCompression::UnixCompress,
        "none" => ArchiveCompression::None,
        _ => return Err(PyValueError::new_err("unknown archive compression")),
    };
    Ok(Sp3ArtifactIdentity {
        requested_identity: crate::exact_cache::identity(&requested_json)?,
        resolved_identity: crate::exact_cache::identity(&resolved_json)?,
        distribution_source: crate::exact_cache::source(&input.distribution_source)?,
        official_filename: input.official_filename,
        product_sha256: input.product_sha256,
        product_byte_length: input.product_byte_length,
        archive_sha256: input.archive_sha256,
        archive_byte_length: input.archive_byte_length,
        compression,
    })
}

/// How agreeing SP3 sources are combined in `merge_sp3`.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeCombine", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub enum PySp3MergeCombine {
    /// Arithmetic mean of agreeing sources.
    MEAN,
    /// Component-wise median of agreeing sources.
    MEDIAN,
    /// Highest-precedence agreeing source, using input order.
    PRECEDENCE,
}

impl PySp3MergeCombine {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "mean" => Ok(Self::MEAN),
            "median" => Ok(Self::MEDIAN),
            "precedence" => Ok(Self::PRECEDENCE),
            other => Err(PyValueError::new_err(format!(
                "unknown SP3 merge combine {other:?}; expected \"mean\", \"median\", or \"precedence\""
            ))),
        }
    }
}

impl From<PySp3MergeCombine> for MergeCombine {
    fn from(value: PySp3MergeCombine) -> Self {
        match value {
            PySp3MergeCombine::MEAN => MergeCombine::Mean,
            PySp3MergeCombine::MEDIAN => MergeCombine::Median,
            PySp3MergeCombine::PRECEDENCE => MergeCombine::Precedence,
        }
    }
}

impl From<MergeCombine> for PySp3MergeCombine {
    fn from(value: MergeCombine) -> Self {
        match value {
            MergeCombine::Mean => Self::MEAN,
            MergeCombine::Median => Self::MEDIAN,
            MergeCombine::Precedence => Self::PRECEDENCE,
        }
    }
}

#[pymethods]
impl PySp3MergeCombine {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::MEAN => "mean",
            Self::MEDIAN => "median",
            Self::PRECEDENCE => "precedence",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::MEAN => "Sp3MergeCombine.MEAN",
            Self::MEDIAN => "Sp3MergeCombine.MEDIAN",
            Self::PRECEDENCE => "Sp3MergeCombine.PRECEDENCE",
        }
    }
}

fn extract_merge_combine(obj: &Bound<'_, PyAny>) -> PyResult<PySp3MergeCombine> {
    if let Ok(value) = obj.extract::<PySp3MergeCombine>() {
        return Ok(value);
    }
    PySp3MergeCombine::from_label(&obj.extract::<String>()?)
}

/// Scope used by precedence-mode SP3 source selection.
#[pyclass(
    module = "sidereon._sidereon",
    name = "Sp3MergePrecedenceScope",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PySp3MergePrecedenceScope {
    /// Select the highest-precedence source present in each cell.
    CELL,
    /// Keep one source owner for an entire satellite arc.
    SATELLITE_ARC,
}

impl PySp3MergePrecedenceScope {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "cell" => Ok(Self::CELL),
            "satellite_arc" => Ok(Self::SATELLITE_ARC),
            other => Err(PyValueError::new_err(format!(
                "unknown SP3 precedence scope {other:?}; expected \"cell\" or \"satellite_arc\""
            ))),
        }
    }
}

impl From<PySp3MergePrecedenceScope> for MergePrecedenceScope {
    fn from(value: PySp3MergePrecedenceScope) -> Self {
        match value {
            PySp3MergePrecedenceScope::CELL => MergePrecedenceScope::Cell,
            PySp3MergePrecedenceScope::SATELLITE_ARC => MergePrecedenceScope::SatelliteArc,
        }
    }
}

#[pymethods]
impl PySp3MergePrecedenceScope {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::CELL => "cell",
            Self::SATELLITE_ARC => "satellite_arc",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::CELL => "Sp3MergePrecedenceScope.CELL",
            Self::SATELLITE_ARC => "Sp3MergePrecedenceScope.SATELLITE_ARC",
        }
    }
}

fn extract_precedence_scope(obj: &Bound<'_, PyAny>) -> PyResult<PySp3MergePrecedenceScope> {
    if let Ok(value) = obj.extract::<PySp3MergePrecedenceScope>() {
        return Ok(value);
    }
    PySp3MergePrecedenceScope::from_label(&obj.extract::<String>()?)
}

/// Optional tolerances that guard contested precedence cells against outliers.
#[pyclass(module = "sidereon._sidereon", name = "Sp3OutlierRejectOptions")]
#[derive(Clone)]
pub struct PySp3OutlierRejectOptions {
    position_tolerance_m: f64,
    clock_tolerance_s: f64,
}

#[pymethods]
impl PySp3OutlierRejectOptions {
    #[new]
    #[pyo3(signature = (position_tolerance_m=0.5, clock_tolerance_s=5.0e-9))]
    fn new(py: Python<'_>, position_tolerance_m: f64, clock_tolerance_s: f64) -> PyResult<Self> {
        require_nonnegative_finite(py, "OutlierPosition", position_tolerance_m)?;
        require_nonnegative_finite(py, "OutlierClock", clock_tolerance_s)?;
        Ok(Self {
            position_tolerance_m: normalize_nonnegative_zero(position_tolerance_m),
            clock_tolerance_s: normalize_nonnegative_zero(clock_tolerance_s),
        })
    }

    #[getter]
    fn position_tolerance_m(&self) -> f64 {
        self.position_tolerance_m
    }

    #[getter]
    fn clock_tolerance_s(&self) -> f64 {
        self.clock_tolerance_s
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3OutlierRejectOptions(position_tolerance_m={}, clock_tolerance_s={})",
            self.position_tolerance_m, self.clock_tolerance_s
        )
    }
}

/// How much per-epoch provenance `merge_sp3` records.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ProvenanceMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub enum PySp3ProvenanceMode {
    /// Precedence transitions and per-contributor coverage only.
    SUMMARY,
    /// Everything in `SUMMARY`, plus one record per accepted cell.
    FULL,
}

impl PySp3ProvenanceMode {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "summary" => Ok(Self::SUMMARY),
            "full" => Ok(Self::FULL),
            other => Err(PyValueError::new_err(format!(
                "unknown SP3 merge provenance mode {other:?}; expected \"summary\" or \"full\""
            ))),
        }
    }
}

impl From<PySp3ProvenanceMode> for ProvenanceMode {
    fn from(value: PySp3ProvenanceMode) -> Self {
        match value {
            PySp3ProvenanceMode::SUMMARY => ProvenanceMode::Summary,
            PySp3ProvenanceMode::FULL => ProvenanceMode::Full,
        }
    }
}

impl From<ProvenanceMode> for PySp3ProvenanceMode {
    fn from(value: ProvenanceMode) -> Self {
        match value {
            ProvenanceMode::Summary => Self::SUMMARY,
            ProvenanceMode::Full => Self::FULL,
        }
    }
}

#[pymethods]
impl PySp3ProvenanceMode {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::SUMMARY => "summary",
            Self::FULL => "full",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::SUMMARY => "Sp3ProvenanceMode.SUMMARY",
            Self::FULL => "Sp3ProvenanceMode.FULL",
        }
    }
}

fn extract_provenance_mode(obj: &Bound<'_, PyAny>) -> PyResult<PySp3ProvenanceMode> {
    if let Ok(value) = obj.extract::<PySp3ProvenanceMode>() {
        return Ok(value);
    }
    PySp3ProvenanceMode::from_label(&obj.extract::<String>()?)
}

/// Continuity checks run on a merged product as a merge post-condition.
///
/// The same checks as `Sp3.check_continuity`, with the same arguments:
/// `orbit_class` (`"meo_gnss"`, `"geosynchronous"`, `"leo"`, or `None` to
/// disable the speed gate), `residual_tolerance_m` (`None` disables the
/// hold-out residual check) and `gap_threshold_factor` (`None` keeps the core
/// default). The defaults are the core `ContinuityOptions::for_orbit_class`
/// settings for GNSS MEO. A residual tolerance must be finite and
/// non-negative.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ContinuityOptions")]
#[derive(Clone)]
pub struct PySp3ContinuityOptions {
    orbit_class: Option<String>,
    residual_tolerance_m: Option<f64>,
    gap_threshold_factor: Option<f64>,
    inner: ContinuityOptions,
}

#[pymethods]
impl PySp3ContinuityOptions {
    #[new]
    #[pyo3(signature = (
        orbit_class = Some("meo_gnss".to_string()),
        residual_tolerance_m = Some(1.0),
        gap_threshold_factor = None,
    ))]
    fn new(
        py: Python<'_>,
        orbit_class: Option<String>,
        residual_tolerance_m: Option<f64>,
        gap_threshold_factor: Option<f64>,
    ) -> PyResult<Self> {
        // A NaN or infinite tolerance, which no residual exceeds, or a
        // negative one, which every residual exceeds, would disable or
        // saturate the check silently; `None` is how the check is turned off.
        let inner = continuity_options(
            py,
            orbit_class.as_deref(),
            residual_tolerance_m,
            gap_threshold_factor,
        )?;
        Ok(Self {
            orbit_class,
            residual_tolerance_m,
            gap_threshold_factor,
            inner,
        })
    }

    /// Orbit class of the speed gate, or `None` when the gate is disabled.
    #[getter]
    fn orbit_class(&self) -> Option<String> {
        self.orbit_class.clone()
    }

    /// Hold-out residual tolerance, metres, or `None` when the check is
    /// disabled.
    #[getter]
    fn residual_tolerance_m(&self) -> Option<f64> {
        self.residual_tolerance_m
    }

    /// Hold-out interpolation gap threshold factor, or `None` for the core
    /// default.
    #[getter]
    fn gap_threshold_factor(&self) -> Option<f64> {
        self.gap_threshold_factor
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ContinuityOptions(orbit_class={:?}, residual_tolerance_m={:?}, gap_threshold_factor={:?})",
            self.orbit_class, self.residual_tolerance_m, self.gap_threshold_factor
        )
    }
}

/// Controls for merging SP3 precise orbit and clock products.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeOptions")]
#[derive(Clone)]
pub struct PySp3MergeOptions {
    position_tolerance_m: f64,
    clock_tolerance_s: f64,
    min_agree: usize,
    clock_min_common: usize,
    combine: PySp3MergeCombine,
    precedence_scope: PySp3MergePrecedenceScope,
    outlier_reject: Option<PySp3OutlierRejectOptions>,
    target_epoch_interval_s: Option<f64>,
    systems: Option<BTreeSet<GnssSystem>>,
    asserted_frame_label_sets: Vec<Vec<String>>,
    helmert: bool,
    provenance: Option<PySp3ProvenanceMode>,
    verify_continuity: Option<PySp3ContinuityOptions>,
}

#[pymethods]
impl PySp3MergeOptions {
    /// Create SP3 merge controls.
    ///
    /// `position_tolerance_m` is metres, `clock_tolerance_s` is seconds,
    /// `target_epoch_interval_s` is seconds or `None`, and `systems` is an
    /// optional sequence of RINEX system letters or names.
    ///
    /// `target_epoch_interval_s` must be a whole number of the 10-nanosecond
    /// ticks an SP3 interval states; the core merge refuses one that is not.
    /// `provenance` (`Sp3ProvenanceMode` or `"summary"`/`"full"`) records
    /// per-epoch provenance on the report; `None` records nothing.
    /// `verify_continuity` runs the continuity checks on the merged product and
    /// attributes each violation to its contributors; `None` runs none.
    /// Neither changes the merged product.
    #[new]
    #[pyo3(signature = (
        position_tolerance_m=0.5,
        clock_tolerance_s=5.0e-9,
        min_agree=2,
        clock_min_common=5,
        combine=PySp3MergeCombine::MEAN,
        precedence_scope=PySp3MergePrecedenceScope::CELL,
        outlier_reject=None,
        target_epoch_interval_s=None,
        systems=None,
        asserted_frame_label_sets=None,
        helmert=false,
        provenance=None,
        verify_continuity=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        position_tolerance_m: f64,
        clock_tolerance_s: f64,
        min_agree: usize,
        clock_min_common: usize,
        #[pyo3(from_py_with = extract_merge_combine)] combine: PySp3MergeCombine,
        #[pyo3(from_py_with = extract_precedence_scope)]
        precedence_scope: PySp3MergePrecedenceScope,
        outlier_reject: Option<&PySp3OutlierRejectOptions>,
        target_epoch_interval_s: Option<f64>,
        systems: Option<Vec<String>>,
        asserted_frame_label_sets: Option<Vec<Vec<String>>>,
        helmert: bool,
        provenance: Option<&Bound<'_, PyAny>>,
        verify_continuity: Option<&PySp3ContinuityOptions>,
    ) -> PyResult<Self> {
        require_nonnegative_finite(py, "Position", position_tolerance_m)?;
        require_nonnegative_finite(py, "Clock", clock_tolerance_s)?;
        if min_agree == 0 {
            return Err(PyValueError::new_err("min_agree must be at least 1"));
        }
        if clock_min_common == 0 {
            return Err(PyValueError::new_err("clock_min_common must be at least 1"));
        }
        let provenance = provenance
            .filter(|value| !value.is_none())
            .map(extract_provenance_mode)
            .transpose()?;

        let systems = systems.map(parse_systems).transpose()?;
        let asserted_frame_label_sets = parse_asserted_frame_label_sets(asserted_frame_label_sets)?;

        Ok(Self {
            position_tolerance_m: normalize_nonnegative_zero(position_tolerance_m),
            clock_tolerance_s: normalize_nonnegative_zero(clock_tolerance_s),
            min_agree,
            clock_min_common,
            combine,
            precedence_scope,
            outlier_reject: outlier_reject.cloned(),
            target_epoch_interval_s,
            systems,
            asserted_frame_label_sets,
            helmert,
            provenance,
            verify_continuity: verify_continuity.cloned(),
        })
    }

    /// Maximum agreeing-source 3D position difference, metres.
    #[getter]
    fn position_tolerance_m(&self) -> f64 {
        self.position_tolerance_m
    }

    /// Maximum agreeing-source clock difference after datum alignment, seconds.
    #[getter]
    fn clock_tolerance_s(&self) -> f64 {
        self.clock_tolerance_s
    }

    /// Minimum agreeing sources required when several sources cover one cell.
    #[getter]
    fn min_agree(&self) -> usize {
        self.min_agree
    }

    /// Minimum common clocked satellites for clock-datum alignment.
    #[getter]
    fn clock_min_common(&self) -> usize {
        self.clock_min_common
    }

    /// Consensus combination policy.
    #[getter]
    fn combine(&self) -> PySp3MergeCombine {
        self.combine
    }

    /// Precedence source-selection scope.
    #[getter]
    fn precedence_scope(&self) -> PySp3MergePrecedenceScope {
        self.precedence_scope
    }

    /// Optional contested-cell outlier guard.
    #[getter]
    fn outlier_reject(&self) -> Option<PySp3OutlierRejectOptions> {
        self.outlier_reject.clone()
    }

    /// Output epoch spacing in seconds, or `None` for the finest input grid.
    #[getter]
    fn target_epoch_interval_s(&self) -> Option<f64> {
        self.target_epoch_interval_s
    }

    /// Optional system filter as RINEX letters (`G`, `R`, `E`, `C`, `J`, `I`, `S`).
    #[getter]
    fn systems(&self) -> Option<Vec<String>> {
        self.systems.as_ref().map(|systems| {
            systems
                .iter()
                .map(|system| system.letter().to_string())
                .collect()
        })
    }

    /// Caller-asserted coordinate-label sets that may merge without frame math.
    #[getter]
    fn asserted_frame_label_sets(&self) -> Vec<Vec<String>> {
        self.asserted_frame_label_sets.clone()
    }

    /// Whether catalog Helmert reconciliation is enabled for known labels.
    #[getter]
    fn helmert(&self) -> bool {
        self.helmert
    }

    /// Per-epoch provenance recorded on the report, or `None` for none.
    #[getter]
    fn provenance(&self) -> Option<PySp3ProvenanceMode> {
        self.provenance
    }

    /// Continuity post-condition checks, or `None` when none run.
    #[getter]
    fn verify_continuity(&self) -> Option<PySp3ContinuityOptions> {
        self.verify_continuity.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeOptions(position_tolerance_m={}, clock_tolerance_s={}, min_agree={}, combine={:?}, precedence_scope={:?}, outlier_reject={}, helmert={}, provenance={:?}, verify_continuity={})",
            self.position_tolerance_m,
            self.clock_tolerance_s,
            self.min_agree,
            self.combine.label(),
            self.precedence_scope.label(),
            self.outlier_reject.is_some(),
            self.helmert,
            self.provenance.map(|mode| mode.label()),
            self.verify_continuity.is_some()
        )
    }
}

impl PySp3MergeOptions {
    fn to_core(&self) -> MergeOptions {
        // Mutate-a-default rather than a struct literal: MergeOptions is
        // non-exhaustive, so any future option the core learns stays at its
        // default without this conversion having to name it.
        let mut options = MergeOptions::default();
        options.position_tolerance_m = self.position_tolerance_m;
        options.clock_tolerance_s = self.clock_tolerance_s;
        options.min_agree = self.min_agree;
        options.clock_min_common = self.clock_min_common;
        options.combine = self.combine.into();
        options.precedence_scope = self.precedence_scope.into();
        options.outlier_reject = self.outlier_reject.as_ref().map(|reject_options| {
            let mut o = OutlierRejectOptions::new(
                reject_options.position_tolerance_m,
                reject_options.clock_tolerance_s,
            );
            o.position_tolerance_m = reject_options.position_tolerance_m;
            o.clock_tolerance_s = reject_options.clock_tolerance_s;
            o
        });
        options.target_epoch_interval_s = self.target_epoch_interval_s;
        options.systems = self.systems.clone();
        let mut frame_recon = Sp3FrameReconciliationOptions::default();
        frame_recon.asserted_equivalent_label_sets = self
            .asserted_frame_label_sets
            .iter()
            .map(|labels| Sp3FrameLabelSet::new(labels.iter().cloned()))
            .collect();
        frame_recon.helmert = self.helmert;
        options.frame_reconciliation = frame_recon;
        options.provenance = self.provenance.map(ProvenanceMode::from);
        options.verify_continuity = self
            .verify_continuity
            .as_ref()
            .map(|continuity| continuity.inner.clone());
        options
    }
}

/// One SP3 merge audit flag for an epoch and satellite.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeFlag")]
#[derive(Clone)]
pub struct PySp3MergeFlag {
    epoch: Instant,
    epoch_j2000_seconds: f64,
    jd_whole: f64,
    jd_fraction: f64,
    satellite: String,
    sources: Vec<usize>,
}

#[pymethods]
impl PySp3MergeFlag {
    /// Flagged epoch, scale-tagged, in the core's own representation.
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.epoch)
    }

    /// Flagged epoch as seconds since J2000 in the product time scale.
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        self.epoch_j2000_seconds
    }

    /// Canonical half-integer Julian day containing the flagged epoch.
    #[getter]
    fn jd_whole(&self) -> f64 {
        self.jd_whole
    }

    /// Fraction within `jd_whole`, retaining the leap-second boundary value 1.0.
    #[getter]
    fn jd_fraction(&self) -> f64 {
        self.jd_fraction
    }

    /// Satellite token, for example `G01`.
    #[getter]
    fn satellite(&self) -> String {
        self.satellite.clone()
    }

    /// Source indices from the input `sources` sequence.
    #[getter]
    fn sources(&self) -> Vec<usize> {
        self.sources.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeFlag(epoch_j2000_seconds={}, satellite={:?}, sources={:?})",
            self.epoch_j2000_seconds, self.satellite, self.sources
        )
    }
}

fn epoch_seconds(epoch: &Instant) -> f64 {
    instant_to_j2000_seconds(epoch).unwrap_or(f64::NAN)
}

/// One source's clock for a merged cell that the merge did not write, with
/// the reason.
///
/// `reason` is `"datum_not_observable"` (the source's datum offset to source
/// 0 could not be estimated at this epoch and is never extrapolated),
/// `"preferred_source_without_clock"` (precedence writes a clock only from the
/// preferred source, named by `preferred_source` when the merge had one, and
/// it had none on the reference datum), or `"no_consensus"` (the clocks
/// disagreed and no agreeing subset met the consensus rule).
#[pyclass(module = "sidereon._sidereon", name = "Sp3ClockOmission")]
#[derive(Clone)]
pub struct PySp3ClockOmission {
    inner: ClockOmission,
}

#[pymethods]
impl PySp3ClockOmission {
    /// The epoch, scale-tagged, in the core's own representation.
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.epoch)
    }

    /// The epoch as seconds since J2000 in the product time scale; NaN for
    /// an epoch held as integer nanoseconds.
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        epoch_seconds(&self.inner.epoch)
    }

    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    /// Index into the `merge_sp3` sources of the source whose clock was not
    /// written.
    #[getter]
    fn source(&self) -> usize {
        self.inner.source
    }

    #[getter]
    fn reason(&self) -> &'static str {
        match self.inner.reason {
            ClockOmissionReason::DatumNotObservable => "datum_not_observable",
            ClockOmissionReason::PreferredSourceWithoutClock { .. } => {
                "preferred_source_without_clock"
            }
            ClockOmissionReason::NoConsensus => "no_consensus",
        }
    }

    /// The preferred source of a `"preferred_source_without_clock"` omission,
    /// when the merge had one for the satellite; `None` for the other reasons.
    #[getter]
    fn preferred_source(&self) -> Option<usize> {
        match self.inner.reason {
            ClockOmissionReason::PreferredSourceWithoutClock { preferred } => preferred,
            ClockOmissionReason::DatumNotObservable | ClockOmissionReason::NoConsensus => None,
        }
    }

    /// Whether the merged cell carries a clock from other sources.
    #[getter]
    fn cell_has_clock(&self) -> bool {
        self.inner.cell_has_clock
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ClockOmission(satellite={:?}, source={}, reason={:?}, cell_has_clock={})",
            self.satellite(),
            self.inner.source,
            self.reason(),
            self.inner.cell_has_clock
        )
    }
}

/// An input epoch that took no part in a merge, with the reason:
/// `"off_target_grid"` (off an explicit `target_epoch_interval_s` grid) or
/// `"not_on_tick_axis"` (no SP3 record states it exactly).
#[pyclass(module = "sidereon._sidereon", name = "Sp3DroppedInputEpoch")]
#[derive(Clone)]
pub struct PySp3DroppedInputEpoch {
    inner: DroppedInputEpoch,
}

#[pymethods]
impl PySp3DroppedInputEpoch {
    /// Index into the `merge_sp3` sources.
    #[getter]
    fn source(&self) -> usize {
        self.inner.source
    }

    /// Index into that source's epochs.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    /// The epoch, scale-tagged, in the core's own representation.
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.epoch)
    }

    /// The epoch as seconds since J2000; NaN for an epoch held as integer
    /// nanoseconds.
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        epoch_seconds(&self.inner.epoch)
    }

    #[getter]
    fn reason(&self) -> &'static str {
        match self.inner.reason {
            DroppedEpochReason::OffTargetGrid => "off_target_grid",
            DroppedEpochReason::NotOnTickAxis => "not_on_tick_axis",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3DroppedInputEpoch(source={}, epoch_index={}, reason={:?})",
            self.inner.source,
            self.inner.epoch_index,
            self.reason()
        )
    }
}

/// How the merge arrived at the value it wrote for one channel of one cell.
///
/// `kind` is `"single_source"` (one source carried the cell),
/// `"precedence"` (precedence picked `selected_source` out of the agreeing
/// `members`) or `"combined"` (the value combines the `members` by `rule`;
/// no single source supplied it, so `selected_source` is `None`).
#[pyclass(module = "sidereon._sidereon", name = "Sp3CellSelection")]
#[derive(Clone)]
pub struct PySp3CellSelection {
    inner: CellSelection,
}

impl PySp3CellSelection {
    fn from_core(inner: &CellSelection) -> Self {
        Self {
            inner: inner.clone(),
        }
    }
}

#[pymethods]
impl PySp3CellSelection {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            CellSelection::SingleSource { .. } => "single_source",
            CellSelection::Precedence { .. } => "precedence",
            CellSelection::Combined { .. } => "combined",
        }
    }

    /// The single source whose value was written; `None` for a combined
    /// value.
    #[getter]
    fn selected_source(&self) -> Option<usize> {
        self.inner.selected_source()
    }

    /// Every source in the accepted consensus, ascending.
    #[getter]
    fn members(&self) -> Vec<usize> {
        self.inner.members()
    }

    /// The combining rule of a `"combined"` value; `None` otherwise.
    #[getter]
    fn rule(&self) -> Option<PySp3MergeCombine> {
        match self.inner {
            CellSelection::Combined { rule, .. } => Some(rule.into()),
            CellSelection::SingleSource { .. } | CellSelection::Precedence { .. } => None,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3CellSelection(kind={:?}, selected_source={:?}, members={:?})",
            self.kind(),
            self.inner.selected_source(),
            self.inner.members()
        )
    }
}

/// Provenance of one accepted `(epoch, satellite)` cell, recorded as the merge
/// decided it.
#[pyclass(module = "sidereon._sidereon", name = "Sp3CellProvenance")]
#[derive(Clone)]
pub struct PySp3CellProvenance {
    inner: CellProvenance,
}

#[pymethods]
impl PySp3CellProvenance {
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.epoch)
    }

    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        epoch_seconds(&self.inner.epoch)
    }

    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    /// How the written position was arrived at; `None` when the cell carries
    /// no position.
    #[getter]
    fn position(&self) -> Option<PySp3CellSelection> {
        self.inner
            .position
            .as_ref()
            .map(PySp3CellSelection::from_core)
    }

    /// How the written clock was arrived at; `None` when the cell carries no
    /// clock.
    #[getter]
    fn clock(&self) -> Option<PySp3CellSelection> {
        self.inner.clock.as_ref().map(PySp3CellSelection::from_core)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3CellProvenance(satellite={:?}, epoch_j2000_seconds={})",
            self.satellite(),
            self.epoch_j2000_seconds()
        )
    }
}

/// One change in which source supplied a satellite's position.
///
/// `reason` is `"sole_availability"`, `"precedence"`, `"outlier_rejection"`
/// or `"consensus_change"`. `from_source` is `None` at a satellite's first
/// accepted cell; `to_source` is `None` when the new cell is combined.
#[pyclass(module = "sidereon._sidereon", name = "Sp3PrecedenceTransition")]
#[derive(Clone)]
pub struct PySp3PrecedenceTransition {
    inner: PrecedenceTransition,
}

#[pymethods]
impl PySp3PrecedenceTransition {
    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    /// The epoch at which the new source took over.
    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.epoch)
    }

    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        epoch_seconds(&self.inner.epoch)
    }

    #[getter(from_source)]
    fn transition_from_source(&self) -> Option<usize> {
        self.inner.from_source
    }

    #[getter]
    fn to_source(&self) -> Option<usize> {
        self.inner.to_source
    }

    #[getter]
    fn reason(&self) -> &'static str {
        match self.inner.reason {
            TransitionReason::SoleAvailability => "sole_availability",
            TransitionReason::Precedence => "precedence",
            TransitionReason::OutlierRejection => "outlier_rejection",
            TransitionReason::ConsensusChange => "consensus_change",
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3PrecedenceTransition(satellite={:?}, from_source={:?}, to_source={:?}, reason={:?})",
            self.satellite(),
            self.inner.from_source,
            self.inner.to_source,
            self.reason()
        )
    }
}

/// What one contributor supplied to the merged product.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ContributorCoverage")]
#[derive(Clone)]
pub struct PySp3ContributorCoverage {
    inner: ContributorCoverage,
}

#[pymethods]
impl PySp3ContributorCoverage {
    /// Index into the `merge_sp3` sources.
    #[getter]
    fn source(&self) -> usize {
        self.inner.source
    }

    /// Accepted cells where this source was in the position or clock
    /// consensus.
    #[getter]
    fn cells_contributed(&self) -> usize {
        self.inner.cells_contributed
    }

    /// Accepted cells whose written position came from this source alone;
    /// always zero under a combining rule.
    #[getter]
    fn cells_selected(&self) -> usize {
        self.inner.cells_selected
    }

    #[getter]
    fn first_epoch(&self) -> Option<PyClockInstant> {
        self.inner.first_epoch.map(PyClockInstant::from_core)
    }

    #[getter]
    fn last_epoch(&self) -> Option<PyClockInstant> {
        self.inner.last_epoch.map(PyClockInstant::from_core)
    }

    /// Accepted cells this source contributed nothing to.
    #[getter]
    fn cells_absent(&self) -> usize {
        self.inner.cells_absent
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ContributorCoverage(source={}, cells_contributed={}, cells_selected={}, cells_absent={})",
            self.inner.source,
            self.inner.cells_contributed,
            self.inner.cells_selected,
            self.inner.cells_absent
        )
    }
}

/// Per-epoch merge provenance, present on the report only when
/// `Sp3MergeOptions.provenance` requested it.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeProvenance")]
#[derive(Clone)]
pub struct PySp3MergeProvenance {
    inner: MergeProvenance,
}

#[pymethods]
impl PySp3MergeProvenance {
    /// The mode that produced this record.
    #[getter]
    fn mode(&self) -> PySp3ProvenanceMode {
        self.inner.mode.into()
    }

    /// One record per accepted cell, in output order; empty under `SUMMARY`.
    #[getter]
    fn cells(&self) -> Vec<PySp3CellProvenance> {
        self.inner
            .cells
            .iter()
            .cloned()
            .map(|inner| PySp3CellProvenance { inner })
            .collect()
    }

    /// Every change of supplying source, in output order.
    #[getter]
    fn transitions(&self) -> Vec<PySp3PrecedenceTransition> {
        self.inner
            .transitions
            .iter()
            .cloned()
            .map(|inner| PySp3PrecedenceTransition { inner })
            .collect()
    }

    /// What each input contributed, in source order.
    #[getter]
    fn coverage(&self) -> Vec<PySp3ContributorCoverage> {
        self.inner
            .coverage
            .iter()
            .cloned()
            .map(|inner| PySp3ContributorCoverage { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeProvenance(mode={:?}, cells={}, transitions={}, coverage={})",
            self.mode().label(),
            self.inner.cells.len(),
            self.inner.transitions.len(),
            self.inner.coverage.len()
        )
    }
}

/// One continuity defect with every field its kind carries.
///
/// `kind` is `"duplicate_epoch"` (`epoch_j2000_s`, `occurrences`),
/// `"single_sample_series"` (no epoch), `"speed_bound"` (`from_j2000_s`,
/// `to_j2000_s`, `interval_s`, `displacement_m`, `implied_speed_m_s`,
/// `bound_m_s`) or `"hold_out_residual"` (`epoch_j2000_s`,
/// `preceding_j2000_s`, `residual_m`, `tolerance_m`, `node_epochs_j2000_s`).
/// A field the kind does not carry is `None`.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ContinuityDefect")]
#[derive(Clone)]
pub struct PySp3ContinuityDefect {
    inner: ContinuityDefect,
}

#[pymethods]
impl PySp3ContinuityDefect {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            ContinuityDefect::DuplicateEpoch { .. } => "duplicate_epoch",
            ContinuityDefect::SingleSampleSeries { .. } => "single_sample_series",
            ContinuityDefect::UnusableSample { .. } => "unusable_sample",
            ContinuityDefect::SpeedBound { .. } => "speed_bound",
            ContinuityDefect::HoldOutResidual { .. } => "hold_out_residual",
        }
    }

    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite().to_string()
    }

    /// The repeated epoch, or the held-out sample's epoch, seconds since J2000.
    #[getter]
    fn epoch_j2000_s(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::DuplicateEpoch { epoch_j2000_s, .. }
            | ContinuityDefect::HoldOutResidual { epoch_j2000_s, .. } => Some(*epoch_j2000_s),
            ContinuityDefect::UnusableSample { epoch_j2000_s, .. } => *epoch_j2000_s,
            ContinuityDefect::SingleSampleSeries { .. } | ContinuityDefect::SpeedBound { .. } => {
                None
            }
        }
    }

    /// How many samples carried the repeated epoch.
    #[getter]
    fn occurrences(&self) -> Option<usize> {
        match &self.inner {
            ContinuityDefect::DuplicateEpoch { occurrences, .. } => Some(*occurrences),
            _ => None,
        }
    }

    #[getter]
    fn sample_index(&self) -> Option<usize> {
        match &self.inner {
            ContinuityDefect::UnusableSample { sample_index, .. } => Some(*sample_index),
            _ => None,
        }
    }

    #[getter]
    fn reason(&self) -> Option<&'static str> {
        match &self.inner {
            ContinuityDefect::UnusableSample { reason, .. } => Some(match reason {
                sidereon_core::ephemeris::UnusableSampleReason::EpochNotPlaced => {
                    "epoch_not_placed"
                }
                sidereon_core::ephemeris::UnusableSampleReason::NonFinitePosition => {
                    "non_finite_position"
                }
                _ => "unknown",
            }),
            _ => None,
        }
    }

    /// Earlier epoch of the speed-gated pair, seconds since J2000.
    #[getter(from_j2000_s)]
    fn speed_bound_from_j2000_s(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::SpeedBound { from_j2000_s, .. } => Some(*from_j2000_s),
            _ => None,
        }
    }

    /// Later epoch of the speed-gated pair, seconds since J2000.
    #[getter]
    fn to_j2000_s(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::SpeedBound { to_j2000_s, .. } => Some(*to_j2000_s),
            _ => None,
        }
    }

    /// Elapsed interval of the speed-gated pair, seconds.
    #[getter]
    fn interval_s(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::SpeedBound { interval_s, .. } => Some(*interval_s),
            _ => None,
        }
    }

    /// 3D chord displacement over the interval, metres.
    #[getter]
    fn displacement_m(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::SpeedBound { displacement_m, .. } => Some(*displacement_m),
            _ => None,
        }
    }

    /// Implied earth-fixed chord speed, metres per second.
    #[getter]
    fn implied_speed_m_s(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::SpeedBound {
                implied_speed_m_s, ..
            } => Some(*implied_speed_m_s),
            _ => None,
        }
    }

    /// The speed bound it exceeded, metres per second.
    #[getter]
    fn bound_m_s(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::SpeedBound { bound_m_s, .. } => Some(*bound_m_s),
            _ => None,
        }
    }

    /// Epoch of the sample before the held-out one, seconds since J2000.
    #[getter]
    fn preceding_j2000_s(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::HoldOutResidual {
                preceding_j2000_s, ..
            } => Some(*preceding_j2000_s),
            _ => None,
        }
    }

    /// Distance between the stored record and its prediction, metres.
    #[getter]
    fn residual_m(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::HoldOutResidual { residual_m, .. } => Some(*residual_m),
            _ => None,
        }
    }

    /// The residual tolerance it exceeded, metres.
    #[getter]
    fn tolerance_m(&self) -> Option<f64> {
        match &self.inner {
            ContinuityDefect::HoldOutResidual { tolerance_m, .. } => Some(*tolerance_m),
            _ => None,
        }
    }

    /// Epochs of the retained nodes the prediction used, seconds since J2000,
    /// ascending.
    #[getter]
    fn node_epochs_j2000_s(&self) -> Option<Vec<f64>> {
        match &self.inner {
            ContinuityDefect::HoldOutResidual {
                node_epochs_j2000_s,
                ..
            } => Some(node_epochs_j2000_s.clone()),
            _ => None,
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ContinuityDefect(kind={:?}, satellite={:?})",
            self.kind(),
            self.satellite()
        )
    }
}

/// One merged cell a continuity finding rests on.
///
/// `role` is `"held_out"`, `"interpolation_node"`, `"pair_end"` or
/// `"repeated_epoch"`; `selection` is how the merge arrived at the value it
/// wrote there, or `None` when it recorded none.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeContinuityCell")]
#[derive(Clone)]
pub struct PySp3MergeContinuityCell {
    inner: MergeContinuityCell,
}

#[pymethods]
impl PySp3MergeContinuityCell {
    #[getter]
    fn epoch_j2000_s(&self) -> f64 {
        self.inner.epoch_j2000_s
    }

    #[getter]
    fn role(&self) -> &'static str {
        continuity_cell_role(self.inner.role)
    }

    #[getter]
    fn selection(&self) -> Option<PySp3CellSelection> {
        self.inner
            .selection
            .as_ref()
            .map(PySp3CellSelection::from_core)
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeContinuityCell(epoch_j2000_s={}, role={:?})",
            self.inner.epoch_j2000_s,
            self.role()
        )
    }
}

pub(crate) fn continuity_cell_role(role: MergeContinuityCellRole) -> &'static str {
    match role {
        MergeContinuityCellRole::HeldOut => "held_out",
        MergeContinuityCellRole::InterpolationNode => "interpolation_node",
        MergeContinuityCellRole::PairEnd => "pair_end",
        MergeContinuityCellRole::RepeatedEpoch => "repeated_epoch",
    }
}

/// One continuity violation in a merged product, attributed to the
/// contributors whose records it rests on.
///
/// `from_sources`/`to_sources` are the sources written to the earlier and
/// later record of the offending pair; `cells` is every merged cell the
/// finding rests on, ascending by epoch; `sources` is every source written to
/// one of those cells; `crosses_contributors` marks a splice between
/// contributors rather than one contributor's own discontinuity.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeContinuityViolation")]
#[derive(Clone)]
pub struct PySp3MergeContinuityViolation {
    inner: MergeContinuityViolation,
}

#[pymethods]
impl PySp3MergeContinuityViolation {
    #[getter]
    fn defect(&self) -> PySp3ContinuityDefect {
        PySp3ContinuityDefect {
            inner: self.inner.defect.clone(),
        }
    }

    #[getter(from_sources)]
    fn violation_from_sources(&self) -> Vec<usize> {
        self.inner.from_sources.clone()
    }

    #[getter]
    fn to_sources(&self) -> Vec<usize> {
        self.inner.to_sources.clone()
    }

    #[getter]
    fn cells(&self) -> Vec<PySp3MergeContinuityCell> {
        self.inner
            .cells
            .iter()
            .cloned()
            .map(|inner| PySp3MergeContinuityCell { inner })
            .collect()
    }

    #[getter]
    fn sources(&self) -> Vec<usize> {
        self.inner.sources.clone()
    }

    #[getter]
    fn crosses_contributors(&self) -> bool {
        self.inner.crosses_contributors
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeContinuityViolation(kind={:?}, satellite={:?}, sources={:?}, crosses_contributors={})",
            self.defect().kind(),
            self.inner.defect.satellite().to_string(),
            self.inner.sources,
            self.inner.crosses_contributors
        )
    }
}

fn violations_to_py<'a>(
    violations: impl IntoIterator<Item = &'a MergeContinuityViolation>,
) -> Vec<PySp3MergeContinuityViolation> {
    violations
        .into_iter()
        .cloned()
        .map(|inner| PySp3MergeContinuityViolation { inner })
        .collect()
}

fn window(from_j2000_s: f64, through_j2000_s: f64) -> PyResult<EpochWindow> {
    EpochWindow::new(from_j2000_s, through_j2000_s)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// Each satellite's position nodes in an SP3 product, for asking exactly
/// which of them the interpolations of an evaluation window select.
///
/// Build with `Sp3InterpolationNodes.for_sp3(product)`; a merge continuity
/// report carries the merged product's as `nodes`.
#[pyclass(module = "sidereon._sidereon", name = "Sp3InterpolationNodes")]
#[derive(Clone)]
pub struct PySp3InterpolationNodes {
    inner: InterpolationNodes,
}

#[pymethods]
impl PySp3InterpolationNodes {
    /// The position node series of every satellite in `sp3`, read under the
    /// product's own interpolation options.
    #[staticmethod]
    fn for_sp3(sp3: &PySp3) -> Self {
        Self {
            inner: InterpolationNodes::for_sp3(&sp3.inner),
        }
    }

    /// Epochs of the nodes some position query of `satellite` in the inclusive
    /// window selects, ascending, seconds since J2000; empty when no query in
    /// the window is served for it.
    fn selected_nodes(
        &self,
        satellite: &str,
        from_j2000_s: f64,
        through_j2000_s: f64,
    ) -> PyResult<Vec<f64>> {
        Ok(self.inner.selected_nodes(
            parse_sat(satellite)?,
            window(from_j2000_s, through_j2000_s)?,
        ))
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> &'static str {
        "Sp3InterpolationNodes(...)"
    }
}

/// Continuity verification of a merged product, run as a merge
/// post-condition when `Sp3MergeOptions.verify_continuity` asked for it.
///
/// Reporting, never refusing: whether a product with violations is acceptable
/// is the caller's decision.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeContinuityReport")]
#[derive(Clone)]
pub struct PySp3MergeContinuityReport {
    inner: MergeContinuityReport,
}

#[pymethods]
impl PySp3MergeContinuityReport {
    /// Whether the merged product is attested continuous.
    #[getter]
    fn attested(&self) -> bool {
        self.inner.attested()
    }

    /// Every defect the continuity check found, ordered by satellite then
    /// epoch.
    #[getter]
    fn defects(&self) -> Vec<PySp3ContinuityDefect> {
        self.inner
            .report
            .defects
            .iter()
            .cloned()
            .map(|inner| PySp3ContinuityDefect { inner })
            .collect()
    }

    /// Adjacent pairs the speed gate examined.
    #[getter]
    fn pairs_checked(&self) -> usize {
        self.inner.report.pairs_checked
    }

    /// Samples the hold-out residual check examined.
    #[getter]
    fn residuals_checked(&self) -> usize {
        self.inner.report.residuals_checked
    }

    /// Samples the residual check could not evaluate.
    #[getter]
    fn residuals_skipped(&self) -> usize {
        self.inner.report.residuals_skipped
    }

    /// Each violation, attributed to the contributors it rests on.
    #[getter]
    fn violations(&self) -> Vec<PySp3MergeContinuityViolation> {
        violations_to_py(&self.inner.violations)
    }

    /// The violations that sit across a change of contributor.
    #[getter]
    fn splices(&self) -> Vec<PySp3MergeContinuityViolation> {
        violations_to_py(self.inner.splices())
    }

    /// The merged product's position nodes.
    #[getter]
    fn nodes(&self) -> PySp3InterpolationNodes {
        PySp3InterpolationNodes {
            inner: self.inner.nodes.clone(),
        }
    }

    /// Violations the interpolations of the inclusive window rest on: their
    /// selected nodes include the held-out, repeated or pair-end record, or
    /// straddle a handover between the violation's records.
    fn violations_influencing(
        &self,
        from_j2000_s: f64,
        through_j2000_s: f64,
    ) -> PyResult<Vec<PySp3MergeContinuityViolation>> {
        Ok(violations_to_py(self.inner.violations_influencing(window(
            from_j2000_s,
            through_j2000_s,
        )?)))
    }

    /// Contributor-changing violations that influence the inclusive window.
    fn splices_influencing(
        &self,
        from_j2000_s: f64,
        through_j2000_s: f64,
    ) -> PyResult<Vec<PySp3MergeContinuityViolation>> {
        Ok(violations_to_py(self.inner.splices_influencing(window(
            from_j2000_s,
            through_j2000_s,
        )?)))
    }

    /// The window-scoped decision over every violation, as the dict
    /// `Sp3MergeReport.continuity_verdict` returns.
    fn verdict(
        &self,
        py: Python<'_>,
        from_j2000_s: f64,
        through_j2000_s: f64,
    ) -> PyResult<PyObject> {
        continuity_verdict_to_py(
            py,
            self.inner
                .verdict_for_window(window(from_j2000_s, through_j2000_s)?),
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeContinuityReport(attested={}, defects={}, violations={})",
            self.inner.attested(),
            self.inner.report.defects.len(),
            self.inner.violations.len()
        )
    }
}

// --- per-satellite coverage of a product ----------------------------------

/// The grid a product's epochs lie on.
///
/// `interval_s` is the grid step, or `None` when the epochs lie on no grid
/// (steps that are not whole multiples of the declared interval, or epochs out
/// of order); `agrees_with_header` says whether the declared interval is that
/// step; `out_of_order` and `unplaced` are epoch indices that do not follow
/// the epoch before them in time, or that no SP3 record states exactly.
#[pyclass(module = "sidereon._sidereon", name = "Sp3EpochGrid")]
#[derive(Clone)]
pub struct PySp3EpochGrid {
    inner: Sp3EpochGrid,
}

#[pymethods]
impl PySp3EpochGrid {
    #[getter]
    fn interval_s(&self) -> Option<f64> {
        self.inner.interval_s
    }

    #[getter]
    fn agrees_with_header(&self) -> bool {
        self.inner.agrees_with_header
    }

    #[getter]
    fn out_of_order(&self) -> Vec<usize> {
        self.inner.out_of_order.clone()
    }

    #[getter]
    fn unplaced(&self) -> Vec<usize> {
        self.inner.unplaced.clone()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3EpochGrid(interval_s={:?}, agrees_with_header={})",
            self.inner.interval_s, self.inner.agrees_with_header
        )
    }
}

/// A run of product epochs at which a satellite carries a channel, by index
/// into the product's epochs, inclusive.
#[pyclass(module = "sidereon._sidereon", name = "Sp3CoverageSpan")]
#[derive(Clone)]
pub struct PySp3CoverageSpan {
    inner: Sp3CoverageSpan,
}

#[pymethods]
impl PySp3CoverageSpan {
    #[getter]
    fn first_index(&self) -> usize {
        self.inner.first_index
    }

    #[getter]
    fn last_index(&self) -> usize {
        self.inner.last_index
    }

    #[getter]
    fn first_epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.first_epoch)
    }

    #[getter]
    fn last_epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.inner.last_epoch)
    }

    /// Number of product epochs in the run.
    #[getter]
    fn epochs(&self) -> usize {
        self.inner.epochs()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3CoverageSpan(first_index={}, last_index={})",
            self.inner.first_index, self.inner.last_index
        )
    }
}

/// A stretch of a product in which a satellite carries no value for a
/// channel.
///
/// `after_index` is the last epoch index before the gap that carries the
/// channel (`None` when the gap opens the product), `before_index` the first
/// after it (`None` when it closes the product), and `missing_epochs` the
/// product epochs inside it; zero when the gap is only a break in the
/// product's own epoch list.
#[pyclass(module = "sidereon._sidereon", name = "Sp3CoverageGap")]
#[derive(Clone)]
pub struct PySp3CoverageGap {
    inner: Sp3CoverageGap,
}

#[pymethods]
impl PySp3CoverageGap {
    #[new]
    #[pyo3(signature = (after_index, before_index, missing_epochs))]
    fn new(after_index: Option<usize>, before_index: Option<usize>, missing_epochs: usize) -> Self {
        Self {
            inner: Sp3CoverageGap {
                after_index,
                before_index,
                missing_epochs,
            },
        }
    }

    #[getter]
    fn after_index(&self) -> Option<usize> {
        self.inner.after_index
    }

    #[getter]
    fn before_index(&self) -> Option<usize> {
        self.inner.before_index
    }

    #[getter]
    fn missing_epochs(&self) -> usize {
        self.inner.missing_epochs
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3CoverageGap(after_index={:?}, before_index={:?}, missing_epochs={})",
            self.inner.after_index, self.inner.before_index, self.inner.missing_epochs
        )
    }
}

/// Where a satellite carries one channel (positions or clocks) in a product.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ChannelCoverage")]
#[derive(Clone)]
pub struct PySp3ChannelCoverage {
    inner: Sp3ChannelCoverage,
}

#[pymethods]
impl PySp3ChannelCoverage {
    /// Number of product epochs at which the channel is carried.
    #[getter]
    fn epochs(&self) -> usize {
        self.inner.epochs
    }

    /// Maximal runs of carried epochs, in file order.
    #[getter]
    fn spans(&self) -> Vec<PySp3CoverageSpan> {
        self.inner
            .spans
            .iter()
            .cloned()
            .map(|inner| PySp3CoverageSpan { inner })
            .collect()
    }

    /// Every stretch without the channel, in file order.
    #[getter]
    fn gaps(&self) -> Vec<PySp3CoverageGap> {
        self.inner
            .gaps
            .iter()
            .copied()
            .map(|inner| PySp3CoverageGap { inner })
            .collect()
    }

    #[getter]
    fn first_epoch(&self) -> Option<PyClockInstant> {
        self.inner.first_epoch().map(PyClockInstant::from_core)
    }

    #[getter]
    fn last_epoch(&self) -> Option<PyClockInstant> {
        self.inner.last_epoch().map(PyClockInstant::from_core)
    }

    /// Whether the channel is carried at every product epoch without a break.
    #[getter]
    fn is_complete(&self) -> bool {
        self.inner.is_complete()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ChannelCoverage(epochs={}, spans={}, gaps={})",
            self.inner.epochs,
            self.inner.spans.len(),
            self.inner.gaps.len()
        )
    }
}

/// One satellite's position and clock coverage in a product.
#[pyclass(module = "sidereon._sidereon", name = "Sp3SatelliteCoverage")]
#[derive(Clone)]
pub struct PySp3SatelliteCoverage {
    inner: Sp3SatelliteCoverage,
}

#[pymethods]
impl PySp3SatelliteCoverage {
    #[getter]
    fn satellite(&self) -> String {
        self.inner.satellite.to_string()
    }

    /// Whether the header satellite list declares it.
    #[getter]
    fn declared(&self) -> bool {
        self.inner.declared
    }

    /// Epochs with a position record.
    #[getter]
    fn positions(&self) -> PySp3ChannelCoverage {
        PySp3ChannelCoverage {
            inner: self.inner.positions.clone(),
        }
    }

    /// Epochs with a clock: a position record's clock or a clock-only record.
    #[getter]
    fn clocks(&self) -> PySp3ChannelCoverage {
        PySp3ChannelCoverage {
            inner: self.inner.clocks.clone(),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3SatelliteCoverage(satellite={:?}, declared={}, positions={}, clocks={})",
            self.satellite(),
            self.inner.declared,
            self.inner.positions.epochs,
            self.inner.clocks.epochs
        )
    }
}

/// Position and clock coverage of every satellite in a product, with the grid
/// its epochs lie on. Returned by `Sp3.satellite_coverage()`.
#[pyclass(module = "sidereon._sidereon", name = "Sp3Coverage")]
#[derive(Clone)]
pub struct PySp3Coverage {
    inner: Sp3Coverage,
}

impl From<Sp3Coverage> for PySp3Coverage {
    fn from(inner: Sp3Coverage) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySp3Coverage {
    #[getter]
    fn grid(&self) -> PySp3EpochGrid {
        PySp3EpochGrid {
            inner: self.inner.grid.clone(),
        }
    }

    /// Each satellite's coverage, ascending by satellite, including every
    /// satellite the header declares.
    #[getter]
    fn satellites(&self) -> Vec<PySp3SatelliteCoverage> {
        self.inner
            .satellites
            .iter()
            .cloned()
            .map(|inner| PySp3SatelliteCoverage { inner })
            .collect()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3Coverage(interval_s={:?}, satellites={})",
            self.inner.grid.interval_s,
            self.inner.satellites.len()
        )
    }
}

/// Per-(epoch, satellite) agreement statistics for one accepted merge cell.
#[pyclass(module = "sidereon._sidereon", name = "Sp3AgreementMetric")]
#[derive(Clone)]
pub struct PySp3AgreementMetric {
    jd_whole: f64,
    jd_fraction: f64,
    satellite: String,
    position_members: usize,
    position_rms_m: Option<f64>,
    position_max_m: Option<f64>,
    clock_members: usize,
    clock_rms_s: Option<f64>,
    clock_max_s: Option<f64>,
}

#[pymethods]
impl PySp3AgreementMetric {
    #[getter]
    fn jd_whole(&self) -> f64 {
        self.jd_whole
    }

    #[getter]
    fn jd_fraction(&self) -> f64 {
        self.jd_fraction
    }

    #[getter]
    fn satellite(&self) -> String {
        self.satellite.clone()
    }

    /// Sources in the accepted position consensus; 0 for a clock-only cell,
    /// which carries no orbit.
    #[getter]
    fn position_members(&self) -> usize {
        self.position_members
    }

    /// RMS of the members' 3D distance from the combined position, metres.
    /// Zero for a single-source cell; `None` for a clock-only cell.
    #[getter]
    fn position_rms_m(&self) -> Option<f64> {
        self.position_rms_m
    }

    /// Largest member distance from the combined position, metres; `None` for
    /// a clock-only cell.
    #[getter]
    fn position_max_m(&self) -> Option<f64> {
        self.position_max_m
    }

    /// Sources in the accepted clock consensus; 0 when the cell carries no
    /// clock.
    #[getter]
    fn clock_members(&self) -> usize {
        self.clock_members
    }

    #[getter]
    fn clock_rms_s(&self) -> Option<f64> {
        self.clock_rms_s
    }

    #[getter]
    fn clock_max_s(&self) -> Option<f64> {
        self.clock_max_s
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3AgreementMetric(satellite={:?}, jd_whole={}, jd_fraction={}, position_members={}, clock_members={})",
            self.satellite,
            self.jd_whole,
            self.jd_fraction,
            self.position_members,
            self.clock_members,
        )
    }
}

/// Per-epoch aggregate of accepted multi-source agreement metrics.
#[pyclass(module = "sidereon._sidereon", name = "Sp3EpochAgreement")]
#[derive(Clone)]
pub struct PySp3EpochAgreement {
    epoch_j2000_seconds: f64,
    jd_whole: f64,
    jd_fraction: f64,
    satellites: usize,
    position_rms_m: Option<f64>,
    position_max_m: Option<f64>,
    clock_rms_s: Option<f64>,
    clock_max_s: Option<f64>,
}

#[pymethods]
impl PySp3EpochAgreement {
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        self.epoch_j2000_seconds
    }

    #[getter]
    fn jd_whole(&self) -> f64 {
        self.jd_whole
    }

    #[getter]
    fn jd_fraction(&self) -> f64 {
        self.jd_fraction
    }

    /// Satellites at this epoch with a multi-source position consensus.
    #[getter]
    fn satellites(&self) -> usize {
        self.satellites
    }

    /// Pooled RMS of the position dispersion over the multi-source cells at
    /// this epoch, metres; `None` when the epoch has none.
    #[getter]
    fn position_rms_m(&self) -> Option<f64> {
        self.position_rms_m
    }

    /// Largest position dispersion over the multi-source cells at this epoch,
    /// metres; `None` when the epoch has none.
    #[getter]
    fn position_max_m(&self) -> Option<f64> {
        self.position_max_m
    }

    #[getter]
    fn clock_rms_s(&self) -> Option<f64> {
        self.clock_rms_s
    }

    #[getter]
    fn clock_max_s(&self) -> Option<f64> {
        self.clock_max_s
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3EpochAgreement(jd_whole={}, jd_fraction={}, satellites={})",
            self.jd_whole, self.jd_fraction, self.satellites,
        )
    }
}

/// Flat per-epoch aggregate tuple. The position and clock spreads are `None`
/// for an epoch with no multi-source consensus in that channel.
type EpochAgreementTuple = (
    f64,
    usize,
    Option<f64>,
    Option<f64>,
    Option<f64>,
    Option<f64>,
);

/// Published Helmert parameters as `(translation_mm, scale_ppb, rotation_mas)`.
type HelmertParametersTuple = (Vec<f64>, f64, Vec<f64>);

/// Published Helmert rates as `(translation_mm_per_year, scale_ppb_per_year,
/// rotation_mas_per_year)`.
type HelmertRatesTuple = (Vec<f64>, f64, Vec<f64>);

/// One coordinate-label reconciliation applied before SP3 merge consensus.
#[pyclass(module = "sidereon._sidereon", name = "Sp3FrameReconciliation")]
#[derive(Clone)]
pub struct PySp3FrameReconciliation {
    source_index: usize,
    source_label: String,
    target_label: String,
    method: String,
    asserted_label_set: Option<Vec<String>>,
    source_frame: Option<String>,
    target_frame: Option<String>,
    catalog_source_frame: Option<String>,
    catalog_target_frame: Option<String>,
    catalog_inverse: bool,
    reference_epoch_year: Option<f64>,
    parameters: Option<HelmertParametersTuple>,
    rates: Option<HelmertRatesTuple>,
    provenance: Option<String>,
    epoch_year_span: Option<(f64, f64)>,
    records_affected: usize,
    identity: bool,
}

#[pymethods]
impl PySp3FrameReconciliation {
    /// Source index in the `merge_sp3` input sequence.
    #[getter]
    fn source_index(&self) -> usize {
        self.source_index
    }

    /// Original coordinate-system label on the reconciled source.
    #[getter]
    fn source_label(&self) -> String {
        self.source_label.clone()
    }

    /// Target coordinate-system label, taken from source 0.
    #[getter]
    fn target_label(&self) -> String {
        self.target_label.clone()
    }

    /// Reconciliation mechanism: `"asserted_equivalence"` or `"helmert"`.
    #[getter]
    fn method(&self) -> String {
        self.method.clone()
    }

    /// Caller-provided assertion set, when assertion reconciliation was used.
    #[getter]
    fn asserted_label_set(&self) -> Option<Vec<String>> {
        self.asserted_label_set.clone()
    }

    /// Resolved source terrestrial frame for Helmert reconciliation.
    #[getter]
    fn source_frame(&self) -> Option<String> {
        self.source_frame.clone()
    }

    /// Resolved target terrestrial frame for Helmert reconciliation.
    #[getter]
    fn target_frame(&self) -> Option<String> {
        self.target_frame.clone()
    }

    /// Source frame of the published catalog row used for Helmert reconciliation.
    #[getter]
    fn catalog_source_frame(&self) -> Option<String> {
        self.catalog_source_frame.clone()
    }

    /// Target frame of the published catalog row used for Helmert reconciliation.
    #[getter]
    fn catalog_target_frame(&self) -> Option<String> {
        self.catalog_target_frame.clone()
    }

    /// Whether the published catalog row was applied in reverse.
    #[getter]
    fn catalog_inverse(&self) -> bool {
        self.catalog_inverse
    }

    /// Published transform reference epoch, when a catalog entry was used.
    #[getter]
    fn reference_epoch_year(&self) -> Option<f64> {
        self.reference_epoch_year
    }

    /// Published parameters `(translation_mm, scale_ppb, rotation_mas)`.
    #[getter]
    fn parameters(&self) -> Option<HelmertParametersTuple> {
        self.parameters.clone()
    }

    /// Published rates `(translation_mm_per_year, scale_ppb_per_year,
    /// rotation_mas_per_year)`.
    #[getter]
    fn rates(&self) -> Option<HelmertRatesTuple> {
        self.rates.clone()
    }

    /// Published-table provenance for the catalog entry.
    #[getter]
    fn provenance(&self) -> Option<String> {
        self.provenance.clone()
    }

    /// Inclusive decimal-year span of affected records.
    #[getter]
    fn epoch_year_span(&self) -> Option<(f64, f64)> {
        self.epoch_year_span
    }

    /// Number of satellite position records covered by the reconciliation.
    #[getter]
    fn records_affected(&self) -> usize {
        self.records_affected
    }

    /// Whether coordinates were left bit-equal because both labels resolved to
    /// the same terrestrial realization.
    #[getter]
    fn identity(&self) -> bool {
        self.identity
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3FrameReconciliation(source_index={}, source_label={:?}, target_label={:?}, method={:?}, records_affected={}, identity={})",
            self.source_index,
            self.source_label,
            self.target_label,
            self.method,
            self.records_affected,
            self.identity
        )
    }
}

/// Audit report returned with a merged SP3 product.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeReport")]
#[derive(Clone)]
pub struct PySp3MergeReport {
    frame_reconciliations: Vec<PySp3FrameReconciliation>,
    quarantined: Vec<PySp3MergeFlag>,
    single_source: Vec<PySp3MergeFlag>,
    position_outliers: Vec<PySp3MergeFlag>,
    clock_outliers: Vec<PySp3MergeFlag>,
    agreement: Vec<PySp3AgreementMetric>,
    position_agreement_rms_m: Option<f64>,
    position_agreement_max_m: Option<f64>,
    clock_agreement_rms_s: Option<f64>,
    clock_agreement_max_s: Option<f64>,
    agreement_epochs: Vec<PySp3EpochAgreement>,
    continuity: Option<MergeContinuityReport>,
    single_source_fraction: Option<f64>,
    provenance: Option<MergeProvenance>,
    omitted_epochs: Vec<Instant>,
    arc_withheld: Vec<PySp3MergeFlag>,
    clock_omissions: Vec<ClockOmission>,
    dropped_input_epochs: Vec<DroppedInputEpoch>,
}

#[pymethods]
impl PySp3MergeReport {
    /// Coordinate-label reconciliations applied before consensus.
    #[getter]
    fn frame_reconciliations(&self) -> Vec<PySp3FrameReconciliation> {
        self.frame_reconciliations.clone()
    }

    /// Number of coordinate-label reconciliations applied before consensus.
    #[getter]
    fn frame_reconciliation_count(&self) -> usize {
        self.frame_reconciliations.len()
    }

    /// Cells omitted because sources disagreed beyond tolerance.
    #[getter]
    fn quarantined(&self) -> Vec<PySp3MergeFlag> {
        self.quarantined.clone()
    }

    /// Cells carried from one source because no cross-check was possible.
    #[getter]
    fn single_source(&self) -> Vec<PySp3MergeFlag> {
        self.single_source.clone()
    }

    /// Cells where an otherwise accepted consensus rejected source outliers.
    #[getter]
    fn position_outliers(&self) -> Vec<PySp3MergeFlag> {
        self.position_outliers.clone()
    }

    /// Clock contributors rejected from an accepted consensus or guard.
    #[getter]
    fn clock_outliers(&self) -> Vec<PySp3MergeFlag> {
        self.clock_outliers.clone()
    }

    #[getter]
    fn quarantined_count(&self) -> usize {
        self.quarantined.len()
    }

    #[getter]
    fn single_source_count(&self) -> usize {
        self.single_source.len()
    }

    #[getter]
    fn position_outlier_count(&self) -> usize {
        self.position_outliers.len()
    }

    #[getter]
    fn clock_outlier_count(&self) -> usize {
        self.clock_outliers.len()
    }

    /// Number of accepted cells with per-cell agreement statistics (one per
    /// (epoch, satellite) written to the merged product).
    #[getter]
    fn agreement_count(&self) -> usize {
        self.agreement.len()
    }

    /// Per-cell agreement records in canonical output (epoch, satellite) order,
    /// including the clock-only cells, whose position metrics are `None`.
    #[getter]
    fn agreement(&self) -> Vec<PySp3AgreementMetric> {
        self.agreement.clone()
    }

    /// Member-count-weighted pooled RMS of the per-cell position dispersion over
    /// every accepted multi-source cell, metres. `None` when no cell had two or
    /// more position-consensus members.
    #[getter]
    fn position_agreement_rms_m(&self) -> Option<f64> {
        self.position_agreement_rms_m
    }

    /// Largest single-cell position dispersion over the accepted cells that
    /// carry an orbit, metres. `None` when no accepted cell carries one.
    #[getter]
    fn position_agreement_max_m(&self) -> Option<f64> {
        self.position_agreement_max_m
    }

    /// Pooled RMS of the per-cell clock dispersion over multi-source cells,
    /// seconds. `None` when no multi-source clock consensus existed.
    #[getter]
    fn clock_agreement_rms_s(&self) -> Option<f64> {
        self.clock_agreement_rms_s
    }

    /// Largest single-cell clock dispersion over all accepted cells, seconds.
    #[getter]
    fn clock_agreement_max_s(&self) -> Option<f64> {
        self.clock_agreement_max_s
    }

    /// Per-epoch aggregate agreement, in output-epoch order, as tuples
    /// `(epoch_j2000_seconds, multi_source_satellites, position_rms_m,
    /// position_max_m, clock_rms_s, clock_max_s)`. A spread is `None` for an
    /// epoch with no multi-source consensus in that channel.
    #[getter]
    fn per_epoch_agreement(&self) -> Vec<EpochAgreementTuple> {
        self.agreement_epochs
            .iter()
            .map(|epoch| {
                (
                    epoch.epoch_j2000_seconds,
                    epoch.satellites,
                    epoch.position_rms_m,
                    epoch.position_max_m,
                    epoch.clock_rms_s,
                    epoch.clock_max_s,
                )
            })
            .collect()
    }

    /// Per-epoch aggregates with an exact split Julian-date epoch.
    #[getter]
    fn agreement_epochs(&self) -> Vec<PySp3EpochAgreement> {
        self.agreement_epochs.clone()
    }

    /// Fraction of accepted cells carried from a single source, with no
    /// second source to cross-check; `None` when no cell was accepted. Read it
    /// beside the agreement spreads, which cover multi-source cells only.
    #[getter]
    fn single_source_fraction(&self) -> Option<f64> {
        self.single_source_fraction
    }

    /// Per-epoch provenance, present only when `Sp3MergeOptions.provenance`
    /// requested it; `None` means it was not requested.
    #[getter]
    fn provenance(&self) -> Option<PySp3MergeProvenance> {
        self.provenance
            .clone()
            .map(|inner| PySp3MergeProvenance { inner })
    }

    /// Continuity verification of the merged product, present only when
    /// `Sp3MergeOptions.verify_continuity` requested it.
    #[getter]
    fn continuity(&self) -> Option<PySp3MergeContinuityReport> {
        self.continuity
            .clone()
            .map(|inner| PySp3MergeContinuityReport { inner })
    }

    /// Union-grid epochs at which the merge accepted no cell, in time order.
    /// None of them is written to the product; each position a source carried
    /// there is in `quarantined` or `arc_withheld`, and each source clock in
    /// `clock_omissions`.
    #[getter]
    fn omitted_epochs(&self) -> Vec<PyClockInstant> {
        self.omitted_epochs
            .iter()
            .copied()
            .map(PyClockInstant::from_core)
            .collect()
    }

    /// `omitted_epochs` as seconds since J2000 in the product time scale;
    /// NaN for an epoch held as integer nanoseconds.
    #[getter]
    fn omitted_epochs_j2000_seconds(&self) -> Vec<f64> {
        self.omitted_epochs.iter().map(epoch_seconds).collect()
    }

    /// Cells whose position some source carried but precedence did not write,
    /// because the preferred source (under satellite-arc precedence, the arc
    /// owner) carried none there.
    #[getter]
    fn arc_withheld(&self) -> Vec<PySp3MergeFlag> {
        self.arc_withheld.clone()
    }

    /// Each source clock the merge did not write, with the reason, in
    /// (epoch, satellite, source) order.
    #[getter]
    fn clock_omissions(&self) -> Vec<PySp3ClockOmission> {
        self.clock_omissions
            .iter()
            .cloned()
            .map(|inner| PySp3ClockOmission { inner })
            .collect()
    }

    /// Input epochs that took no part in the merge, with the reason, in
    /// (source, epoch) order.
    #[getter]
    fn dropped_input_epochs(&self) -> Vec<PySp3DroppedInputEpoch> {
        self.dropped_input_epochs
            .iter()
            .cloned()
            .map(|inner| PySp3DroppedInputEpoch { inner })
            .collect()
    }

    /// Contributor-changing violations that influence the inclusive window,
    /// when merge continuity verification was requested; `None` when it was
    /// not.
    fn splices_influencing(
        &self,
        from_j2000_s: f64,
        through_j2000_s: f64,
    ) -> PyResult<Option<Vec<PySp3MergeContinuityViolation>>> {
        let Some(report) = &self.continuity else {
            return Ok(None);
        };
        Ok(Some(violations_to_py(report.splices_influencing(window(
            from_j2000_s,
            through_j2000_s,
        )?))))
    }

    /// Decide whether the optional merge continuity post-condition can
    /// influence an inclusive evaluation window through `merged`'s derived
    /// interpolation stencil.
    ///
    /// `None` means continuity verification was not requested for the merge.
    /// A present verdict retains both the influencing findings and the complete
    /// defect and contributor-changing splice lists.
    fn continuity_verdict(
        &self,
        py: Python<'_>,
        merged: &PySp3,
        from_j2000_s: f64,
        through_j2000_s: f64,
    ) -> PyResult<Option<PyObject>> {
        let Some(report) = &self.continuity else {
            return Ok(None);
        };
        let _ = merged;
        let verdict = report.verdict_for_window(window(from_j2000_s, through_j2000_s)?);
        continuity_verdict_to_py(py, verdict).map(Some)
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeReport(frame_reconciliations={}, quarantined={}, single_source={}, position_outliers={}, clock_outliers={}, \
             agreement_count={}, omitted_epochs={}, arc_withheld={}, clock_omissions={}, dropped_input_epochs={}, provenance={}, continuity={})",
            self.frame_reconciliations.len(),
            self.quarantined.len(),
            self.single_source.len(),
            self.position_outliers.len(),
            self.clock_outliers.len(),
            self.agreement.len(),
            self.omitted_epochs.len(),
            self.arc_withheld.len(),
            self.clock_omissions.len(),
            self.dropped_input_epochs.len(),
            self.provenance.is_some(),
            self.continuity.is_some(),
        )
    }
}

impl From<MergeReport> for PySp3MergeReport {
    fn from(value: MergeReport) -> Self {
        let position_agreement_rms_m = value.position_agreement_rms_m();
        let position_agreement_max_m = value.position_agreement_max_m();
        let clock_agreement_rms_s = value.clock_agreement_rms_s();
        let clock_agreement_max_s = value.clock_agreement_max_s();
        let single_source_fraction = value.single_source_fraction();
        let agreement_epochs = value
            .per_epoch_agreement()
            .into_iter()
            .map(PySp3EpochAgreement::from)
            .collect();
        Self {
            frame_reconciliations: value
                .frame_reconciliations
                .into_iter()
                .map(PySp3FrameReconciliation::from)
                .collect(),
            quarantined: value
                .quarantined
                .into_iter()
                .map(PySp3MergeFlag::from)
                .collect(),
            single_source: value
                .single_source
                .into_iter()
                .map(PySp3MergeFlag::from)
                .collect(),
            position_outliers: value
                .position_outliers
                .into_iter()
                .map(PySp3MergeFlag::from)
                .collect(),
            clock_outliers: value
                .clock_outliers
                .into_iter()
                .map(PySp3MergeFlag::from)
                .collect(),
            agreement: value
                .agreement
                .into_iter()
                .map(PySp3AgreementMetric::from)
                .collect(),
            position_agreement_rms_m,
            position_agreement_max_m,
            clock_agreement_rms_s,
            clock_agreement_max_s,
            agreement_epochs,
            continuity: value.continuity,
            single_source_fraction,
            provenance: value.provenance,
            omitted_epochs: value.omitted_epochs,
            arc_withheld: value
                .arc_withheld
                .into_iter()
                .map(PySp3MergeFlag::from)
                .collect(),
            clock_omissions: value.clock_omissions,
            dropped_input_epochs: value.dropped_input_epochs,
        }
    }
}

impl From<Sp3FrameReconciliation> for PySp3FrameReconciliation {
    fn from(value: Sp3FrameReconciliation) -> Self {
        Self {
            source_index: value.source_index,
            source_label: value.source_label,
            target_label: value.target_label,
            method: match value.method {
                sidereon_core::ephemeris::Sp3FrameReconciliationMethod::AssertedEquivalence => {
                    "asserted_equivalence".to_string()
                }
                sidereon_core::ephemeris::Sp3FrameReconciliationMethod::Helmert => {
                    "helmert".to_string()
                }
            },
            asserted_label_set: value.asserted_label_set,
            source_frame: value.source_frame.map(|frame| frame.to_string()),
            target_frame: value.target_frame.map(|frame| frame.to_string()),
            catalog_source_frame: value.catalog_source_frame.map(|frame| frame.to_string()),
            catalog_target_frame: value.catalog_target_frame.map(|frame| frame.to_string()),
            catalog_inverse: value.catalog_inverse,
            reference_epoch_year: value.reference_epoch_year,
            parameters: value.parameters.map(|parameters| {
                (
                    parameters.translation_mm.to_vec(),
                    parameters.scale_ppb,
                    parameters.rotation_mas.to_vec(),
                )
            }),
            rates: value.rates.map(|rates| {
                (
                    rates.translation_mm_per_year.to_vec(),
                    rates.scale_ppb_per_year,
                    rates.rotation_mas_per_year.to_vec(),
                )
            }),
            provenance: value.provenance,
            epoch_year_span: value.epoch_year_span.map(|span| (span[0], span[1])),
            records_affected: value.records_affected,
            identity: value.identity,
        }
    }
}

impl From<MergeFlag> for PySp3MergeFlag {
    fn from(value: MergeFlag) -> Self {
        let (jd_whole, jd_fraction) = instant_split(&value.epoch);
        Self {
            epoch: value.epoch,
            epoch_j2000_seconds: instant_to_j2000_seconds(&value.epoch).unwrap_or(f64::NAN),
            jd_whole,
            jd_fraction,
            satellite: value.satellite.to_string(),
            sources: value.sources,
        }
    }
}

impl From<AgreementMetric> for PySp3AgreementMetric {
    fn from(value: AgreementMetric) -> Self {
        let (jd_whole, jd_fraction) = instant_split(&value.epoch);
        Self {
            jd_whole,
            jd_fraction,
            satellite: value.satellite.to_string(),
            position_members: value.position_members,
            position_rms_m: value.position_rms_m,
            position_max_m: value.position_max_m,
            clock_members: value.clock_members,
            clock_rms_s: value.clock_rms_s,
            clock_max_s: value.clock_max_s,
        }
    }
}

impl From<EpochAgreement> for PySp3EpochAgreement {
    fn from(value: EpochAgreement) -> Self {
        let (jd_whole, jd_fraction) = instant_split(&value.epoch);
        Self {
            epoch_j2000_seconds: instant_to_j2000_seconds(&value.epoch).unwrap_or(f64::NAN),
            jd_whole,
            jd_fraction,
            satellites: value.satellites,
            position_rms_m: value.position_rms_m,
            position_max_m: value.position_max_m,
            clock_rms_s: value.clock_rms_s,
            clock_max_s: value.clock_max_s,
        }
    }
}

/// Merge SP3 products with the core consensus merge path.
///
/// `sources` is ordered by source precedence. Returns `(sp3, report)`, where
/// `sp3` is a merged precise orbit and clock product and `report` records
/// quarantined, single-source, and position-outlier cells.
#[pyfunction]
#[pyo3(signature = (sources, options=None))]
fn merge_sp3(
    py: Python<'_>,
    sources: Vec<Py<PySp3>>,
    options: Option<Py<PySp3MergeOptions>>,
) -> PyResult<(PySp3, PySp3MergeReport)> {
    if sources.is_empty() {
        return Err(PyValueError::new_err(
            "merge_sp3 requires at least one SP3 product",
        ));
    }

    let core_sources: Vec<_> = sources
        .iter()
        .map(|source| source.borrow(py).inner.clone())
        .collect();
    let opts = option_py_or_default(
        py,
        options.as_ref(),
        PySp3MergeOptions::to_core,
        MergeOptions::default,
    );
    let (merged, report) = merge(&core_sources, &opts).map_err(|err| core_sp3_error(py, err))?;

    Ok((PySp3 { inner: merged }, report.into()))
}

/// Build the canonical identity of exact SP3 artifacts and merge controls.
///
/// Artifact JSON is an internal bridge format assembled by the typed Python
/// data API. Validation and canonicalization are performed by the shared core.
type Sp3MergeInputIdentityTuple = (u8, String, Vec<usize>, Option<Vec<usize>>);

#[pyfunction]
#[pyo3(signature = (artifacts_json, options=None))]
fn sp3_merge_input_identity(
    py: Python<'_>,
    artifacts_json: Vec<String>,
    options: Option<Py<PySp3MergeOptions>>,
) -> PyResult<Sp3MergeInputIdentityTuple> {
    let artifacts = artifacts_json
        .iter()
        .map(|value| artifact_identity(value))
        .collect::<PyResult<Vec<_>>>()?;
    let opts = option_py_or_default(
        py,
        options.as_ref(),
        PySp3MergeOptions::to_core,
        MergeOptions::default,
    );
    let identity = Sp3MergeInputIdentity::new(&artifacts, &opts)
        .map_err(|error| merge_identity_error(py, error))?;
    let indices = |contributors: &[Sp3ArtifactIdentity]| {
        contributors
            .iter()
            .map(|contributor| {
                artifacts
                    .iter()
                    .position(|artifact| artifact == contributor)
                    .expect("core canonical contributors originate in the input")
            })
            .collect::<Vec<_>>()
    };
    let canonical_indices = indices(&identity.contributors);
    let precedence_indices = identity.precedence_contributors.as_deref().map(indices);
    Ok((
        identity.schema_version,
        identity.stable_id,
        canonical_indices,
        precedence_indices,
    ))
}

fn validation_error(
    py: Python<'_>,
    kind: &str,
    field: &str,
    value: f64,
    reason: &str,
    message: String,
) -> PyErr {
    let error = PyValueError::new_err(message);
    let object = error.value(py);
    let _ = object.setattr("kind", kind);
    let _ = object.setattr("field", field);
    // Keep NaN and infinities as text so error payloads remain JSON-safe.
    let _ = object.setattr("value", format!("{value:?}"));
    let _ = object.setattr("reason", reason);
    error
}

fn interval_error(py: Python<'_>, error: Sp3EpochIntervalError) -> PyErr {
    validation_error(
        py,
        "sp3_epoch_interval",
        error.field,
        error.value,
        &format!("{:?}", error.reason),
        error.to_string(),
    )
}

fn continuity_validation_error(
    py: Python<'_>,
    error: sidereon_core::ephemeris::ContinuityOptionsError,
) -> PyErr {
    validation_error(
        py,
        "continuity_options",
        error.field,
        error.value,
        &format!("{:?}", error.reason),
        error.to_string(),
    )
}

fn tolerance_error(py: Python<'_>, error: sidereon_core::ephemeris::MergeToleranceError) -> PyErr {
    validation_error(
        py,
        "sp3_merge_tolerance",
        &format!("{:?}", error.field),
        error.value,
        "not_finite_or_negative",
        error.to_string(),
    )
}

fn core_sp3_error(py: Python<'_>, error: CoreError) -> PyErr {
    match error {
        CoreError::Sp3EpochInterval(error) => interval_error(py, error),
        CoreError::Sp3MergeTolerance(error) => tolerance_error(py, error),
        CoreError::ContinuityOptions(error) => continuity_validation_error(py, error),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn merge_identity_error(py: Python<'_>, error: Sp3MergeInputIdentityError) -> PyErr {
    match error {
        Sp3MergeInputIdentityError::InvalidTolerance(error) => tolerance_error(py, error),
        Sp3MergeInputIdentityError::TargetEpochInterval(error) => interval_error(py, error),
        Sp3MergeInputIdentityError::ContinuityOptions(error) => {
            continuity_validation_error(py, error)
        }
        other => {
            let error = PyValueError::new_err(other.to_string());
            let _ = error.value(py).setattr("kind", "sp3_merge_input_identity");
            error
        }
    }
}

/// Return the authoritative core speed bound for an SP3 continuity orbit class.
#[pyfunction]
fn sp3_orbit_class_speed_bound_m_s(orbit_class: &str) -> PyResult<f64> {
    let class = match orbit_class {
        "meo_gnss" => OrbitClass::MeoGnss,
        "geosynchronous" => OrbitClass::Geosynchronous,
        "leo" => OrbitClass::Leo,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown orbit class: {other}"
            )))
        }
    };
    Ok(class.max_earth_fixed_speed_m_s())
}

pub(crate) fn require_finite(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

fn require_nonnegative_finite(py: Python<'_>, field: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(validation_error(
            py,
            "sp3_merge_tolerance",
            field,
            value,
            "not_finite_or_negative",
            format!("SP3 merge tolerance {field} must be finite and nonnegative"),
        ))
    }
}

fn normalize_nonnegative_zero(value: f64) -> f64 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn parse_systems(values: Vec<String>) -> PyResult<BTreeSet<GnssSystem>> {
    if values.is_empty() {
        return Err(PyValueError::new_err("systems must not be empty"));
    }
    values
        .iter()
        .map(|value| parse_system(value))
        .collect::<PyResult<BTreeSet<_>>>()
}

fn parse_asserted_frame_label_sets(values: Option<Vec<Vec<String>>>) -> PyResult<Vec<Vec<String>>> {
    let Some(values) = values else {
        return Ok(Vec::new());
    };
    values
        .into_iter()
        .enumerate()
        .map(|(idx, labels)| {
            if labels.len() < 2 {
                return Err(PyValueError::new_err(format!(
                    "asserted_frame_label_sets[{idx}] must contain at least two labels"
                )));
            }
            labels
                .into_iter()
                .map(|label| {
                    let trimmed = label.trim().to_string();
                    if trimmed.is_empty() {
                        Err(PyValueError::new_err(format!(
                            "asserted_frame_label_sets[{idx}] contains an empty label"
                        )))
                    } else {
                        Ok(trimmed)
                    }
                })
                .collect::<PyResult<Vec<_>>>()
        })
        .collect()
}

fn parse_system(value: &str) -> PyResult<GnssSystem> {
    match value.trim().to_ascii_uppercase().as_str() {
        "G" | "GPS" => Ok(GnssSystem::Gps),
        "R" | "GLO" | "GLONASS" => Ok(GnssSystem::Glonass),
        "E" | "GAL" | "GALILEO" => Ok(GnssSystem::Galileo),
        "C" | "BDS" | "BEIDOU" => Ok(GnssSystem::BeiDou),
        "J" | "QZSS" => Ok(GnssSystem::Qzss),
        "I" | "IRNSS" | "NAVIC" => Ok(GnssSystem::Navic),
        "S" | "SBAS" => Ok(GnssSystem::Sbas),
        other => Err(PyValueError::new_err(format!(
            "unknown GNSS system {other:?}; expected one of G, R, E, C, J, I, S"
        ))),
    }
}

fn instant_to_j2000_seconds(epoch: &Instant) -> Option<f64> {
    match epoch.repr {
        InstantRepr::JulianDate(jd) => Some(seconds_between_splits(
            jd.jd_whole,
            jd.fraction,
            J2000_JD,
            0.0,
        )),
        InstantRepr::Nanos(_) => None,
    }
}

fn instant_split(epoch: &Instant) -> (f64, f64) {
    match epoch.repr {
        InstantRepr::JulianDate(jd) => (jd.jd_whole, jd.fraction),
        InstantRepr::Nanos(_) => (f64::NAN, f64::NAN),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySp3MergeCombine>()?;
    m.add_class::<PySp3MergePrecedenceScope>()?;
    m.add_class::<PySp3OutlierRejectOptions>()?;
    m.add_class::<PySp3MergeOptions>()?;
    m.add_class::<PySp3MergeFlag>()?;
    m.add_class::<PySp3AgreementMetric>()?;
    m.add_class::<PySp3EpochAgreement>()?;
    m.add_class::<PySp3FrameReconciliation>()?;
    m.add_class::<PySp3MergeReport>()?;
    m.add_class::<PySp3ProvenanceMode>()?;
    m.add_class::<PySp3ContinuityOptions>()?;
    m.add_class::<PySp3ClockOmission>()?;
    m.add_class::<PySp3DroppedInputEpoch>()?;
    m.add_class::<PySp3CellSelection>()?;
    m.add_class::<PySp3CellProvenance>()?;
    m.add_class::<PySp3PrecedenceTransition>()?;
    m.add_class::<PySp3ContributorCoverage>()?;
    m.add_class::<PySp3MergeProvenance>()?;
    m.add_class::<PySp3ContinuityDefect>()?;
    m.add_class::<PySp3MergeContinuityCell>()?;
    m.add_class::<PySp3MergeContinuityViolation>()?;
    m.add_class::<PySp3InterpolationNodes>()?;
    m.add_class::<PySp3MergeContinuityReport>()?;
    m.add_class::<PySp3EpochGrid>()?;
    m.add_class::<PySp3CoverageSpan>()?;
    m.add_class::<PySp3CoverageGap>()?;
    m.add_class::<PySp3ChannelCoverage>()?;
    m.add_class::<PySp3SatelliteCoverage>()?;
    m.add_class::<PySp3Coverage>()?;
    m.add_function(wrap_pyfunction!(merge_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(sp3_merge_input_identity, m)?)?;
    m.add_function(wrap_pyfunction!(sp3_orbit_class_speed_bound_m_s, m)?)?;
    Ok(())
}
