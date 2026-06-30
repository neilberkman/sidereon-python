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

use sidereon_core::antex::{Antenna, AntennaKind, Antex, AntexDateTime};
use sidereon_core::astro::time::civil::seconds_between_splits;
use sidereon_core::astro::time::{Instant, InstantRepr};
use sidereon_core::constants::J2000_JD;
use sidereon_core::ephemeris::{merge, MergeCombine, MergeFlag, MergeOptions, MergeReport};
use sidereon_core::GnssSystem;

use crate::marshal::option_py_or_default;
use crate::{np_array, to_antex_err, PySp3};

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

/// Controls for merging SP3 precise orbit and clock products.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeOptions")]
#[derive(Clone)]
pub struct PySp3MergeOptions {
    position_tolerance_m: f64,
    clock_tolerance_s: f64,
    min_agree: usize,
    clock_min_common: usize,
    combine: PySp3MergeCombine,
    target_epoch_interval_s: Option<f64>,
    systems: Option<BTreeSet<GnssSystem>>,
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
        target_epoch_interval_s=None,
        systems=None,
    ))]
    fn new(
        position_tolerance_m: f64,
        clock_tolerance_s: f64,
        min_agree: usize,
        clock_min_common: usize,
        #[pyo3(from_py_with = extract_merge_combine)] combine: PySp3MergeCombine,
        target_epoch_interval_s: Option<f64>,
        systems: Option<Vec<String>>,
    ) -> PyResult<Self> {
        require_positive_finite("position_tolerance_m", position_tolerance_m)?;
        require_positive_finite("clock_tolerance_s", clock_tolerance_s)?;
        if min_agree == 0 {
            return Err(PyValueError::new_err("min_agree must be at least 1"));
        }
        if clock_min_common == 0 {
            return Err(PyValueError::new_err("clock_min_common must be at least 1"));
        }
        if let Some(value) = target_epoch_interval_s {
            require_positive_finite("target_epoch_interval_s", value)?;
        }

        let systems = systems.map(parse_systems).transpose()?;

        Ok(Self {
            position_tolerance_m,
            clock_tolerance_s,
            min_agree,
            clock_min_common,
            combine,
            target_epoch_interval_s,
            systems,
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

    /// Output epoch spacing in seconds, or `None` for the coarsest input grid.
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

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeOptions(position_tolerance_m={}, clock_tolerance_s={}, min_agree={}, combine={:?})",
            self.position_tolerance_m,
            self.clock_tolerance_s,
            self.min_agree,
            self.combine.label()
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
            target_epoch_interval_s: self.target_epoch_interval_s,
            systems: self.systems.clone(),
        }
    }
}

/// One SP3 merge audit flag for an epoch and satellite.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeFlag")]
#[derive(Clone)]
pub struct PySp3MergeFlag {
    epoch_j2000_seconds: f64,
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

/// One per-epoch agreement aggregate as a flat tuple:
/// `(epoch_j2000_seconds, multi_source_satellites, position_rms_m,
/// position_max_m, clock_rms_s, clock_max_s)`.
type EpochAgreementTuple = (f64, usize, f64, f64, Option<f64>, Option<f64>);

/// Audit report returned with a merged SP3 product.
#[pyclass(module = "sidereon._sidereon", name = "Sp3MergeReport")]
#[derive(Clone)]
pub struct PySp3MergeReport {
    quarantined: Vec<PySp3MergeFlag>,
    single_source: Vec<PySp3MergeFlag>,
    position_outliers: Vec<PySp3MergeFlag>,
    agreement_count: usize,
    position_agreement_rms_m: Option<f64>,
    position_agreement_max_m: Option<f64>,
    clock_agreement_rms_s: Option<f64>,
    clock_agreement_max_s: Option<f64>,
    per_epoch_agreement: Vec<EpochAgreementTuple>,
}

#[pymethods]
impl PySp3MergeReport {
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

    /// Number of accepted cells with per-cell agreement statistics (one per
    /// (epoch, satellite) written to the merged product).
    #[getter]
    fn agreement_count(&self) -> usize {
        self.agreement_count
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
        self.per_epoch_agreement.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3MergeReport(quarantined={}, single_source={}, position_outliers={}, \
             agreement_count={})",
            self.quarantined.len(),
            self.single_source.len(),
            self.position_outliers.len(),
            self.agreement_count,
        )
    }
}

impl From<MergeReport> for PySp3MergeReport {
    fn from(value: MergeReport) -> Self {
        let position_agreement_rms_m = value.position_agreement_rms_m();
        let position_agreement_max_m = value.position_agreement_max_m();
        let clock_agreement_rms_s = value.clock_agreement_rms_s();
        let clock_agreement_max_s = value.clock_agreement_max_s();
        let per_epoch_agreement = value
            .per_epoch_agreement()
            .into_iter()
            .map(|e| {
                (
                    instant_to_j2000_seconds(&e.epoch).unwrap_or(f64::NAN),
                    e.satellites,
                    e.position_rms_m,
                    e.position_max_m,
                    e.clock_rms_s,
                    e.clock_max_s,
                )
            })
            .collect();
        Self {
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
            agreement_count: value.agreement.len(),
            position_agreement_rms_m,
            position_agreement_max_m,
            clock_agreement_rms_s,
            clock_agreement_max_s,
            per_epoch_agreement,
        }
    }
}

impl From<MergeFlag> for PySp3MergeFlag {
    fn from(value: MergeFlag) -> Self {
        Self {
            epoch_j2000_seconds: instant_to_j2000_seconds(&value.epoch).unwrap_or(f64::NAN),
            satellite: value.satellite.to_string(),
            sources: value.sources,
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

fn parse_systems(values: Vec<String>) -> PyResult<BTreeSet<GnssSystem>> {
    if values.is_empty() {
        return Err(PyValueError::new_err("systems must not be empty"));
    }
    values
        .iter()
        .map(|value| parse_system(value))
        .collect::<PyResult<BTreeSet<_>>>()
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

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAntennaKind>()?;
    m.add_class::<PyAntexDateTime>()?;
    m.add_class::<PyAntex>()?;
    m.add_class::<PyAntenna>()?;
    m.add_class::<PySp3MergeCombine>()?;
    m.add_class::<PySp3MergeOptions>()?;
    m.add_class::<PySp3MergeFlag>()?;
    m.add_class::<PySp3MergeReport>()?;
    m.add_function(wrap_pyfunction!(load_antex, m)?)?;
    m.add_function(wrap_pyfunction!(merge_sp3, m)?)?;
    Ok(())
}
