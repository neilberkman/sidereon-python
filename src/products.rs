//! Precise-product utilities beyond single-file SP3 parsing.
//!
//! This module marshals Python inputs into core precise-product APIs. The SP3
//! merge path returns the existing `Sp3` binding; ANTEX values are thin wrappers
//! over the parsed core antenna correction records.

use std::collections::BTreeSet;
use std::path::PathBuf;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyModule};
use serde::Deserialize;

use sidereon_core::antex::{Antenna, AntennaKind, Antex, AntexDateTime};
use sidereon_core::astro::time::civil::seconds_between_splits;
use sidereon_core::astro::time::{Instant, InstantRepr};
use sidereon_core::constants::J2000_JD;
use sidereon_core::data::ArchiveCompression;
use sidereon_core::ephemeris::{
    merge, AgreementMetric, EpochAgreement, MergeCombine, MergeFlag, MergeOptions,
    MergePrecedenceScope, MergeReport, OutlierRejectOptions, Sp3ArtifactIdentity, Sp3FrameLabelSet,
    Sp3FrameReconciliation, Sp3FrameReconciliationOptions, Sp3MergeInputIdentity,
};
use sidereon_core::GnssSystem;

use crate::marshal::option_py_or_default;
use crate::{np_array, to_antex_err, PySp3};

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

/// Receiver or satellite antenna block role.
#[pyclass(module = "sidereon._sidereon", name = "AntennaKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyAntennaKind {
    /// Receiver antenna calibration block.
    RECEIVER,
    /// Satellite antenna calibration block.
    SATELLITE,
}

impl From<AntennaKind> for PyAntennaKind {
    fn from(value: AntennaKind) -> Self {
        match value {
            AntennaKind::Receiver => Self::RECEIVER,
            AntennaKind::Satellite => Self::SATELLITE,
        }
    }
}

#[pymethods]
impl PyAntennaKind {
    /// Stable lowercase role label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::RECEIVER => "receiver",
            Self::SATELLITE => "satellite",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::RECEIVER => "AntennaKind.RECEIVER",
            Self::SATELLITE => "AntennaKind.SATELLITE",
        }
    }
}

/// Civil UTC-like ANTEX validity timestamp fields.
#[pyclass(module = "sidereon._sidereon", name = "AntexDateTime")]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyAntexDateTime {
    inner: AntexDateTime,
}

impl From<AntexDateTime> for PyAntexDateTime {
    fn from(inner: AntexDateTime) -> Self {
        Self { inner }
    }
}

impl PyAntexDateTime {
    fn to_core(self) -> AntexDateTime {
        self.inner
    }
}

#[pymethods]
impl PyAntexDateTime {
    /// Create an ANTEX validity timestamp.
    ///
    /// Fields are UTC-like calendar components from ANTEX `VALID FROM` /
    /// `VALID UNTIL` records. Leap seconds are not represented.
    #[new]
    #[pyo3(signature = (year, month, day, hour=0, minute=0, second=0))]
    fn new(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: u8) -> PyResult<Self> {
        AntexDateTime::new(year, month, day, hour, minute, second)
            .map(Self::from)
            .map_err(|err| PyValueError::new_err(err.to_string()))
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
    fn second(&self) -> u8 {
        self.inner.second
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexDateTime({:04}, {:02}, {:02}, {:02}, {:02}, {:02})",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second
        )
    }
}

/// Parsed ANTEX receiver and satellite antenna calibration product.
#[pyclass(module = "sidereon._sidereon", name = "Antex")]
#[derive(Clone)]
pub struct PyAntex {
    inner: Antex,
}

#[pymethods]
impl PyAntex {
    /// Number of antenna blocks parsed from the product.
    #[getter]
    fn antenna_count(&self) -> usize {
        self.inner.antennas.len()
    }

    /// ANTEX `TYPE / SERIAL` ids in deterministic order.
    #[getter]
    fn antenna_ids(&self) -> Vec<String> {
        self.inner.antennas.keys().cloned().collect()
    }

    /// Return an antenna by exact `TYPE / SERIAL` id, or `None`.
    fn antenna(&self, id: &str) -> Option<PyAntenna> {
        self.inner.antenna(id).cloned().map(PyAntenna::from)
    }

    /// Return the satellite antenna for `prn` valid at `epoch`, or `None`.
    fn satellite_antenna(&self, prn: &str, epoch: PyAntexDateTime) -> Option<PyAntenna> {
        self.inner
            .satellite_antenna(prn, epoch.to_core())
            .cloned()
            .map(PyAntenna::from)
    }

    /// Serialize this product to standard ANTEX text via the core writer
    /// ([`Antex::encode`](sidereon_core::antex::Antex::encode)). Re-parsing the
    /// output with [`load_antex`] yields an equal product.
    fn to_antex_string(&self) -> String {
        self.inner.encode()
    }

    fn __repr__(&self) -> String {
        format!("Antex(antenna_count={})", self.inner.antennas.len())
    }
}

/// Receiver or satellite ANTEX antenna calibration block.
#[pyclass(module = "sidereon._sidereon", name = "Antenna")]
#[derive(Clone)]
pub struct PyAntenna {
    inner: Antenna,
}

impl From<Antenna> for PyAntenna {
    fn from(inner: Antenna) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyAntenna {
    /// ANTEX `TYPE / SERIAL` id.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Receiver or satellite block role.
    #[getter]
    fn kind(&self) -> PyAntennaKind {
        self.inner.kind.into()
    }

    /// ANTEX antenna type field.
    #[getter]
    fn antenna_type(&self) -> String {
        self.inner.antenna_type.clone()
    }

    /// ANTEX serial, PRN, or radome field.
    #[getter]
    fn serial(&self) -> String {
        self.inner.serial.clone()
    }

    /// Azimuth grid spacing in degrees.
    #[getter]
    fn dazi_deg(&self) -> f64 {
        self.inner.dazi_deg
    }

    /// Zenith grid start angle in degrees.
    #[getter]
    fn zenith_start_deg(&self) -> f64 {
        self.inner.zenith_start_deg
    }

    /// Zenith grid end angle in degrees.
    #[getter]
    fn zenith_end_deg(&self) -> f64 {
        self.inner.zenith_end_deg
    }

    /// Zenith grid step in degrees.
    #[getter]
    fn zenith_step_deg(&self) -> f64 {
        self.inner.zenith_step_deg
    }

    /// Optional SINEX calibration code.
    #[getter]
    fn sinex_code(&self) -> Option<String> {
        self.inner.sinex_code.clone()
    }

    /// First valid timestamp, or `None` for receiver blocks without a window.
    #[getter]
    fn valid_from(&self) -> Option<PyAntexDateTime> {
        self.inner.valid_from.map(PyAntexDateTime::from)
    }

    /// Last valid timestamp, or `None` for open-ended blocks.
    #[getter]
    fn valid_until(&self) -> Option<PyAntexDateTime> {
        self.inner.valid_until.map(PyAntexDateTime::from)
    }

    /// Available frequency codes, for example `G01`.
    #[getter]
    fn frequencies(&self) -> Vec<String> {
        self.inner.frequencies.keys().cloned().collect()
    }

    /// Whether this antenna block is valid at `epoch`.
    fn valid_at(&self, epoch: PyAntexDateTime) -> bool {
        self.inner.valid_at(epoch.to_core())
    }

    /// Frequency-dependent phase-center offset, numpy `(3,)` north/east/up metres.
    fn pco<'py>(&self, py: Python<'py>, frequency: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pco = self
            .inner
            .pco(frequency)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(np_array(py, &pco))
    }

    /// Frequency-dependent phase-center variation in metres.
    ///
    /// `zenith_deg` and optional `azimuth_deg` are degrees. If the antenna has no
    /// azimuth grid, the core no-azimuth interpolation is used.
    #[pyo3(signature = (frequency, zenith_deg, azimuth_deg=None))]
    fn pcv(&self, frequency: &str, zenith_deg: f64, azimuth_deg: Option<f64>) -> PyResult<f64> {
        require_finite("zenith_deg", zenith_deg)?;
        if let Some(value) = azimuth_deg {
            require_finite("azimuth_deg", value)?;
        }
        self.inner
            .pcv(frequency, zenith_deg, azimuth_deg)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "Antenna(id={:?}, kind={})",
            self.inner.id,
            self.kind().label()
        )
    }
}

/// Parse an ANTEX 1.4 antenna product from in-memory bytes or a file path.
///
/// `source` may be bytes / bytearray containing the full ASCII text, or a path
/// (`str` / `os.PathLike`) to read. PCO/PCV values are exposed in metres.
#[pyfunction]
fn load_antex(source: &Bound<'_, PyAny>) -> PyResult<PyAntex> {
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        return parse_antex_bytes(bytes.as_bytes());
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        return parse_antex_bytes(unsafe { buf.as_bytes() });
    }
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_antex expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    parse_antex_bytes(&data)
}

fn parse_antex_bytes(bytes: &[u8]) -> PyResult<PyAntex> {
    let text = std::str::from_utf8(bytes).map_err(to_antex_err)?;
    let inner = Antex::parse(text).map_err(to_antex_err)?;
    Ok(PyAntex { inner })
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
    fn new(position_tolerance_m: f64, clock_tolerance_s: f64) -> PyResult<Self> {
        require_nonnegative_finite("position_tolerance_m", position_tolerance_m)?;
        require_nonnegative_finite("clock_tolerance_s", clock_tolerance_s)?;
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
}

#[pymethods]
impl PySp3MergeOptions {
    /// Create SP3 merge controls.
    ///
    /// `position_tolerance_m` is metres, `clock_tolerance_s` is seconds,
    /// `target_epoch_interval_s` is seconds or `None`, and `systems` is an
    /// optional sequence of RINEX system letters or names.
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
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
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
    ) -> PyResult<Self> {
        require_nonnegative_finite("position_tolerance_m", position_tolerance_m)?;
        require_nonnegative_finite("clock_tolerance_s", clock_tolerance_s)?;
        if min_agree == 0 {
            return Err(PyValueError::new_err("min_agree must be at least 1"));
        }
        if clock_min_common == 0 {
            return Err(PyValueError::new_err("clock_min_common must be at least 1"));
        }
        if let Some(value) = target_epoch_interval_s {
            require_positive_finite("target_epoch_interval_s", value)?;
            if (value - value.round()).abs() > 1.0e-6 {
                return Err(PyValueError::new_err(
                    "target_epoch_interval_s must be a whole number of seconds",
                ));
            }
        }

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

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeOptions(position_tolerance_m={}, clock_tolerance_s={}, min_agree={}, combine={:?}, precedence_scope={:?}, outlier_reject={}, helmert={})",
            self.position_tolerance_m,
            self.clock_tolerance_s,
            self.min_agree,
            self.combine.label(),
            self.precedence_scope.label(),
            self.outlier_reject.is_some(),
            self.helmert
        )
    }
}

impl PySp3MergeOptions {
    fn to_core(&self) -> MergeOptions {
        MergeOptions {
            position_tolerance_m: self.position_tolerance_m,
            clock_tolerance_s: self.clock_tolerance_s,
            min_agree: self.min_agree,
            clock_min_common: self.clock_min_common,
            combine: self.combine.into(),
            precedence_scope: self.precedence_scope.into(),
            outlier_reject: self
                .outlier_reject
                .as_ref()
                .map(|options| OutlierRejectOptions {
                    position_tolerance_m: options.position_tolerance_m,
                    clock_tolerance_s: options.clock_tolerance_s,
                }),
            target_epoch_interval_s: self.target_epoch_interval_s,
            systems: self.systems.clone(),
            frame_reconciliation: Sp3FrameReconciliationOptions {
                asserted_equivalent_label_sets: self
                    .asserted_frame_label_sets
                    .iter()
                    .map(|labels| Sp3FrameLabelSet::new(labels.iter().cloned()))
                    .collect(),
                helmert: self.helmert,
            },
        }
    }
}

/// One SP3 merge audit flag for an epoch and satellite.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeFlag")]
#[derive(Clone)]
pub struct PySp3MergeFlag {
    epoch_j2000_seconds: f64,
    jd_whole: f64,
    jd_fraction: f64,
    satellite: String,
    sources: Vec<usize>,
}

#[pymethods]
impl PySp3MergeFlag {
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

/// Per-(epoch, satellite) agreement statistics for one accepted merge cell.
#[pyclass(module = "sidereon._sidereon", name = "Sp3AgreementMetric")]
#[derive(Clone)]
pub struct PySp3AgreementMetric {
    jd_whole: f64,
    jd_fraction: f64,
    satellite: String,
    position_members: usize,
    position_rms_m: f64,
    position_max_m: f64,
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

    #[getter]
    fn position_members(&self) -> usize {
        self.position_members
    }

    #[getter]
    fn position_rms_m(&self) -> f64 {
        self.position_rms_m
    }

    #[getter]
    fn position_max_m(&self) -> f64 {
        self.position_max_m
    }

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
    position_rms_m: f64,
    position_max_m: f64,
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

    #[getter]
    fn satellites(&self) -> usize {
        self.satellites
    }

    #[getter]
    fn position_rms_m(&self) -> f64 {
        self.position_rms_m
    }

    #[getter]
    fn position_max_m(&self) -> f64 {
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

/// Backward-compatible flat per-epoch aggregate tuple.
type EpochAgreementTuple = (f64, usize, f64, f64, Option<f64>, Option<f64>);

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

    /// Per-cell agreement records in canonical output (epoch, satellite) order.
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

    /// Largest single-cell position dispersion over all accepted cells, metres.
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
    /// position_max_m, clock_rms_s, clock_max_s)`.
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

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeReport(frame_reconciliations={}, quarantined={}, single_source={}, position_outliers={}, clock_outliers={}, \
             agreement_count={})",
            self.frame_reconciliations.len(),
            self.quarantined.len(),
            self.single_source.len(),
            self.position_outliers.len(),
            self.clock_outliers.len(),
            self.agreement.len(),
        )
    }
}

impl From<MergeReport> for PySp3MergeReport {
    fn from(value: MergeReport) -> Self {
        let position_agreement_rms_m = value.position_agreement_rms_m();
        let position_agreement_max_m = value.position_agreement_max_m();
        let clock_agreement_rms_s = value.clock_agreement_rms_s();
        let clock_agreement_max_s = value.clock_agreement_max_s();
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
    let (merged, report) =
        merge(&core_sources, &opts).map_err(|err| PyValueError::new_err(err.to_string()))?;

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
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
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

fn require_finite(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

fn require_positive_finite(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be positive and finite"
        )))
    }
}

fn require_nonnegative_finite(name: &str, value: f64) -> PyResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} must be non-negative and finite"
        )))
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
    m.add_class::<PyAntennaKind>()?;
    m.add_class::<PyAntexDateTime>()?;
    m.add_class::<PyAntex>()?;
    m.add_class::<PyAntenna>()?;
    m.add_class::<PySp3MergeCombine>()?;
    m.add_class::<PySp3MergePrecedenceScope>()?;
    m.add_class::<PySp3OutlierRejectOptions>()?;
    m.add_class::<PySp3MergeOptions>()?;
    m.add_class::<PySp3MergeFlag>()?;
    m.add_class::<PySp3AgreementMetric>()?;
    m.add_class::<PySp3EpochAgreement>()?;
    m.add_class::<PySp3FrameReconciliation>()?;
    m.add_class::<PySp3MergeReport>()?;
    m.add_function(wrap_pyfunction!(load_antex, m)?)?;
    m.add_function(wrap_pyfunction!(merge_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(sp3_merge_input_identity, m)?)?;
    Ok(())
}
