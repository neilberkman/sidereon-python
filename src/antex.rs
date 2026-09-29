//! ANTEX 1.4 antenna calibration binding.
//!
//! A PyO3 surface over `sidereon_core::antex`. The core retains every record
//! the format defines - header version and PCV type, calibration methods,
//! comments in their places, validity bounds with their exact seconds,
//! frequency sections in file order with their RMS sections - and this module
//! copies those values into Python objects. Core refusals raise typed
//! exceptions carrying the core error as `detail`. No calibration arithmetic
//! lives here.

use std::path::PathBuf;

use numpy::PyArray1;
use pyo3::basic::CompareOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyModule, PyType};

use sidereon_core::antex::{
    Antenna, AntennaKind, Antex, AntexDateTime, AntexError, AntexHeader, AntexVersion, Calibration,
    Frequency, FrequencyRms, OuterComment, PcvGrid, PcvSample, PcvType, PcvTypeRecord,
    SecondFraction, ZenithGrid,
};

use crate::products::require_finite;
use crate::AntexParseError;
use crate::{antex_query_error_type, antex_write_error_type, np_array, to_antex_err};

/// Raise `ty` with the core error's message and its typed payload as `detail`.
fn antex_error(py: Python<'_>, ty: PyResult<Bound<'_, PyType>>, err: AntexError) -> PyErr {
    let ty = match ty {
        Ok(ty) => ty,
        Err(e) => return e,
    };
    let py_err = PyErr::from_type(ty, err.to_string());
    let detail = match (PyAntexErrorDetail { inner: err }).into_pyobject(py) {
        Ok(detail) => detail,
        Err(e) => return e,
    };
    if let Err(e) = py_err.value(py).setattr("detail", detail) {
        return e;
    }
    py_err
}

/// A parse refusal: `AntexParseError` with `detail`.
fn to_antex_parse_err(py: Python<'_>, err: AntexError) -> PyErr {
    antex_error(py, Ok(py.get_type::<AntexParseError>()), err)
}

/// A frequency, PCO or PCV lookup refusal: `AntexQueryError` with `detail`.
fn to_antex_query_err(py: Python<'_>, err: AntexError) -> PyErr {
    antex_error(py, antex_query_error_type(py), err)
}

/// A write refusal: `AntexWriteError` with `detail`, and the core's `field`
/// and `reason` for the variants that carry them.
fn to_antex_write_err(py: Python<'_>, err: AntexError) -> PyErr {
    let (field, reason): (Option<&'static str>, Option<String>) = match &err {
        AntexError::Unwritable { field, reason } => (Some(*field), Some(reason.clone())),
        AntexError::InvalidInput { field, reason } => (Some(*field), Some(reason.to_string())),
        _ => (None, None),
    };
    let py_err = antex_error(py, antex_write_error_type(py), err);
    let value = py_err.value(py);
    if let Err(e) = value.setattr("field", field) {
        return e;
    }
    if let Err(e) = value.setattr("reason", reason) {
        return e;
    }
    py_err
}

/// The typed payload an ANTEX refusal carries on its `detail` attribute.
///
/// `kind` is the core `AntexError` variant name and `details()` that
/// variant's fields under their own keys. `antenna_id`, `record`, `field`,
/// `value`, `frequency`, `reason` and `sections` are conveniences that return
/// None for a variant that names no such field. A failure with no core error
/// behind it (an unreadable file, bytes that are not UTF-8) and a hand-built
/// exception carry no detail.
#[pyclass(module = "sidereon._sidereon", name = "AntexErrorDetail")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexErrorDetail {
    inner: AntexError,
}

#[pymethods]
impl PyAntexErrorDetail {
    /// Error kind string matching the core `AntexError` variant name.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            AntexError::InvalidDateTime => "InvalidDateTime",
            AntexError::InvalidField { .. } => "InvalidField",
            AntexError::RepeatedRecord { .. } => "RepeatedRecord",
            AntexError::DegenerateGrid { .. } => "DegenerateGrid",
            AntexError::InvalidInput { .. } => "InvalidInput",
            AntexError::UnknownFrequency { .. } => "UnknownFrequency",
            AntexError::AmbiguousFrequency { .. } => "AmbiguousFrequency",
            AntexError::MissingPco { .. } => "MissingPco",
            AntexError::EmptyPcvGrid { .. } => "EmptyPcvGrid",
            AntexError::Unwritable { .. } => "Unwritable",
        }
    }

    /// Formatted core error message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// The antenna block the refusal names; None for a header record and for
    /// variants that name no block.
    #[getter]
    fn antenna_id(&self) -> Option<String> {
        match &self.inner {
            AntexError::InvalidField { antenna_id, .. }
            | AntexError::RepeatedRecord { antenna_id, .. } => antenna_id.clone(),
            AntexError::DegenerateGrid { antenna_id, .. }
            | AntexError::UnknownFrequency { antenna_id, .. }
            | AntexError::AmbiguousFrequency { antenna_id, .. }
            | AntexError::MissingPco { antenna_id, .. }
            | AntexError::EmptyPcvGrid { antenna_id, .. } => Some(antenna_id.clone()),
            _ => None,
        }
    }

    /// The record label the refusal names.
    #[getter]
    fn record(&self) -> Option<&'static str> {
        match &self.inner {
            AntexError::InvalidField { record, .. } | AntexError::RepeatedRecord { record, .. } => {
                Some(record)
            }
            _ => None,
        }
    }

    /// The field the refusal names.
    #[getter]
    fn field(&self) -> Option<&'static str> {
        match &self.inner {
            AntexError::InvalidField { field, .. }
            | AntexError::InvalidInput { field, .. }
            | AntexError::Unwritable { field, .. } => Some(field),
            _ => None,
        }
    }

    /// The trimmed field text, for `InvalidField`.
    #[getter]
    fn value(&self) -> Option<String> {
        match &self.inner {
            AntexError::InvalidField { value, .. } => Some(value.clone()),
            _ => None,
        }
    }

    /// The frequency label the refusal names.
    #[getter]
    fn frequency(&self) -> Option<String> {
        match &self.inner {
            AntexError::DegenerateGrid { frequency, .. }
            | AntexError::UnknownFrequency { frequency, .. }
            | AntexError::AmbiguousFrequency { frequency, .. }
            | AntexError::MissingPco { frequency, .. }
            | AntexError::EmptyPcvGrid { frequency, .. } => Some(frequency.clone()),
            _ => None,
        }
    }

    /// The reason text, for the variants that carry one.
    #[getter]
    fn reason(&self) -> Option<String> {
        match &self.inner {
            AntexError::DegenerateGrid { reason, .. } | AntexError::Unwritable { reason, .. } => {
                Some(reason.clone())
            }
            AntexError::InvalidInput { reason, .. } => Some((*reason).to_string()),
            _ => None,
        }
    }

    /// Number of sections carrying the label, for `AmbiguousFrequency`.
    #[getter]
    fn sections(&self) -> Option<usize> {
        match &self.inner {
            AntexError::AmbiguousFrequency { sections, .. } => Some(*sections),
            _ => None,
        }
    }

    /// Dictionary containing every field of this refusal variant.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            AntexError::InvalidDateTime => {}
            AntexError::InvalidField {
                antenna_id,
                record,
                field,
                value,
            } => {
                dict.set_item("antenna_id", antenna_id.clone())?;
                dict.set_item("record", *record)?;
                dict.set_item("field", *field)?;
                dict.set_item("value", value.as_str())?;
            }
            AntexError::RepeatedRecord { antenna_id, record } => {
                dict.set_item("antenna_id", antenna_id.clone())?;
                dict.set_item("record", *record)?;
            }
            AntexError::DegenerateGrid {
                antenna_id,
                frequency,
                reason,
            } => {
                dict.set_item("antenna_id", antenna_id.as_str())?;
                dict.set_item("frequency", frequency.as_str())?;
                dict.set_item("reason", reason.as_str())?;
            }
            AntexError::InvalidInput { field, reason } => {
                dict.set_item("field", *field)?;
                dict.set_item("reason", *reason)?;
            }
            AntexError::UnknownFrequency {
                antenna_id,
                frequency,
            }
            | AntexError::MissingPco {
                antenna_id,
                frequency,
            }
            | AntexError::EmptyPcvGrid {
                antenna_id,
                frequency,
            } => {
                dict.set_item("antenna_id", antenna_id.as_str())?;
                dict.set_item("frequency", frequency.as_str())?;
            }
            AntexError::AmbiguousFrequency {
                antenna_id,
                frequency,
                sections,
            } => {
                dict.set_item("antenna_id", antenna_id.as_str())?;
                dict.set_item("frequency", frequency.as_str())?;
                dict.set_item("sections", *sections)?;
            }
            AntexError::Unwritable { field, reason } => {
                dict.set_item("field", *field)?;
                dict.set_item("reason", reason.as_str())?;
            }
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexErrorDetail(kind=\"{}\", message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &PyAntexErrorDetail) -> bool {
        self.inner == other.inner
    }
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

/// Phase center variation type from `PCV TYPE / REFANT`: absolute values, or
/// values relative to a reference antenna.
#[pyclass(module = "sidereon._sidereon", name = "AntexPcvType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyAntexPcvType {
    /// `A`: absolute values.
    ABSOLUTE,
    /// `R`: values relative to a reference antenna.
    RELATIVE,
}

impl From<PcvType> for PyAntexPcvType {
    fn from(value: PcvType) -> Self {
        match value {
            PcvType::Absolute => Self::ABSOLUTE,
            PcvType::Relative => Self::RELATIVE,
        }
    }
}

/// Whether a PCV row is headed by `NOAZI` or by a numeric azimuth.
#[pyclass(module = "sidereon._sidereon", name = "AntexPcvGrid", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyAntexPcvGrid {
    /// A `NOAZI` row, used for zenith-only interpolation or fallback.
    NO_AZIMUTH,
    /// A numeric-azimuth row, grouped by azimuth for lookup.
    AZIMUTH,
}

impl From<PcvGrid> for PyAntexPcvGrid {
    fn from(value: PcvGrid) -> Self {
        match value {
            PcvGrid::NoAzimuth => Self::NO_AZIMUTH,
            PcvGrid::Azimuth => Self::AZIMUTH,
        }
    }
}

/// `ANTEX VERSION / SYST`: the format version (`F8.1`) and the satellite
/// system flag, None when that column is blank.
#[pyclass(module = "sidereon._sidereon", name = "AntexVersion")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyAntexVersion {
    inner: AntexVersion,
}

#[pymethods]
impl PyAntexVersion {
    /// Format version.
    #[getter]
    fn version(&self) -> f64 {
        self.inner.version
    }

    /// Satellite system flag (`G`, `R`, `E`, `C`, `J`, `S` or `M`), or None
    /// when blank.
    #[getter]
    fn system(&self) -> Option<String> {
        self.inner.system.map(String::from)
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexVersion(version={:?}, system={:?})",
            self.inner.version, self.inner.system
        )
    }

    fn __eq__(&self, other: &PyAntexVersion) -> bool {
        self.inner == other.inner
    }
}

/// `PCV TYPE / REFANT`: whether the values are absolute or relative, and the
/// reference antenna of relative values.
#[pyclass(module = "sidereon._sidereon", name = "AntexPcvTypeRecord")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexPcvTypeRecord {
    inner: PcvTypeRecord,
}

#[pymethods]
impl PyAntexPcvTypeRecord {
    /// Absolute or relative values.
    #[getter]
    fn pcv_type(&self) -> PyAntexPcvType {
        self.inner.pcv_type.into()
    }

    /// Reference antenna type from columns 21-40, trimmed; empty when blank.
    #[getter]
    fn reference_antenna_type(&self) -> &str {
        &self.inner.reference_antenna_type
    }

    /// Reference antenna serial number from columns 41-60, trimmed; empty
    /// when blank.
    #[getter]
    fn reference_antenna_serial(&self) -> &str {
        &self.inner.reference_antenna_serial
    }

    /// The antenna type relative values refer to: the stated type, or
    /// `AOAD/M_T`, which ANTEX 1.4 names for a relative file that leaves it
    /// blank. None for absolute values.
    #[getter]
    fn reference_antenna(&self) -> Option<&str> {
        self.inner.reference_antenna()
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexPcvTypeRecord(pcv_type={:?}, reference_antenna_type={:?})",
            self.inner.pcv_type, self.inner.reference_antenna_type
        )
    }

    fn __eq__(&self, other: &PyAntexPcvTypeRecord) -> bool {
        self.inner == other.inner
    }
}

/// ANTEX header records. `version` and `pcv_type` are None when the source
/// has no such record; `comments` holds the header `COMMENT` text in file
/// order, trailing blanks removed; `end_of_header` says whether the source
/// carries `END OF HEADER`.
#[pyclass(module = "sidereon._sidereon", name = "AntexHeader")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexHeader {
    inner: AntexHeader,
}

#[pymethods]
impl PyAntexHeader {
    /// `ANTEX VERSION / SYST`, or None.
    #[getter]
    fn version(&self) -> Option<PyAntexVersion> {
        self.inner.version.map(|inner| PyAntexVersion { inner })
    }

    /// `PCV TYPE / REFANT`, or None.
    #[getter]
    fn pcv_type(&self) -> Option<PyAntexPcvTypeRecord> {
        self.inner
            .pcv_type
            .clone()
            .map(|inner| PyAntexPcvTypeRecord { inner })
    }

    /// Header comment text in file order.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Whether the source carries `END OF HEADER`.
    #[getter]
    fn end_of_header(&self) -> bool {
        self.inner.end_of_header
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexHeader(version={:?}, comments={}, end_of_header={})",
            self.inner.version.map(|version| version.version),
            self.inner.comments.len(),
            self.inner.end_of_header
        )
    }

    fn __eq__(&self, other: &PyAntexHeader) -> bool {
        self.inner == other.inner
    }
}

/// A `COMMENT` record after `END OF HEADER` outside every antenna block,
/// placed by the number of blocks before it.
#[pyclass(module = "sidereon._sidereon", name = "AntexOuterComment")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexOuterComment {
    inner: OuterComment,
}

#[pymethods]
impl PyAntexOuterComment {
    /// Number of antenna blocks, in file order, before the comment.
    #[getter]
    fn blocks_before(&self) -> usize {
        self.inner.blocks_before
    }

    /// Comment text, trailing blanks removed.
    #[getter]
    fn text(&self) -> &str {
        &self.inner.text
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexOuterComment(blocks_before={}, text={:?})",
            self.inner.blocks_before, self.inner.text
        )
    }

    fn __eq__(&self, other: &PyAntexOuterComment) -> bool {
        self.inner == other.inner
    }
}

/// A `METH / BY / # / DATE` record. Text fields keep their text with trailing
/// blanks removed; `antennas_calibrated` is None when its `I6` field is blank.
#[pyclass(module = "sidereon._sidereon", name = "AntexCalibration")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexCalibration {
    inner: Calibration,
}

#[pymethods]
impl PyAntexCalibration {
    /// Calibration method from columns 1-20.
    #[getter]
    fn method(&self) -> &str {
        &self.inner.method
    }

    /// Agency from columns 21-40.
    #[getter]
    fn agency(&self) -> &str {
        &self.inner.agency
    }

    /// Number of individual antennas calibrated, or None when blank.
    #[getter]
    fn antennas_calibrated(&self) -> Option<u32> {
        self.inner.antennas_calibrated
    }

    /// Date text from columns 51-60.
    #[getter]
    fn date(&self) -> &str {
        &self.inner.date
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexCalibration(method={:?}, agency={:?}, antennas_calibrated={:?}, date={:?})",
            self.inner.method, self.inner.agency, self.inner.antennas_calibrated, self.inner.date
        )
    }

    fn __eq__(&self, other: &PyAntexCalibration) -> bool {
        self.inner == other.inner
    }
}

/// `ZEN1 / ZEN2 / DZEN`, degrees: the first and last zenith (receiver) or
/// nadir (satellite) angle of each row and the step between values.
#[pyclass(module = "sidereon._sidereon", name = "AntexZenithGrid")]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PyAntexZenithGrid {
    inner: ZenithGrid,
}

#[pymethods]
impl PyAntexZenithGrid {
    /// `ZEN1`, degrees.
    #[getter]
    fn start_deg(&self) -> f64 {
        self.inner.start_deg
    }

    /// `ZEN2`, degrees.
    #[getter]
    fn end_deg(&self) -> f64 {
        self.inner.end_deg
    }

    /// `DZEN`, degrees.
    #[getter]
    fn step_deg(&self) -> f64 {
        self.inner.step_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexZenithGrid(start_deg={:?}, end_deg={:?}, step_deg={:?})",
            self.inner.start_deg, self.inner.end_deg, self.inner.step_deg
        )
    }

    fn __eq__(&self, other: &PyAntexZenithGrid) -> bool {
        self.inner == other.inner
    }
}

/// One phase-center-variation grid value, metres, placed on the antenna's
/// grid: `azimuth_deg` is None for a `NOAZI` row.
#[pyclass(module = "sidereon._sidereon", name = "AntexPcvSample")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexPcvSample {
    inner: PcvSample,
}

#[pymethods]
impl PyAntexPcvSample {
    /// Row kind.
    #[getter]
    fn grid(&self) -> PyAntexPcvGrid {
        self.inner.grid.into()
    }

    /// Row azimuth in degrees as read; None for a `NOAZI` row.
    #[getter]
    fn azimuth_deg(&self) -> Option<f64> {
        self.inner.azimuth_deg
    }

    /// Zenith of the value, degrees.
    #[getter]
    fn zenith_deg(&self) -> f64 {
        self.inner.zenith_deg
    }

    /// The value in metres (`mm * 1e-3`, as RTKLIB `readantex` converts it).
    #[getter]
    fn value_m(&self) -> f64 {
        self.inner.value_m
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexPcvSample(azimuth_deg={:?}, zenith_deg={:?}, value_m={:?})",
            self.inner.azimuth_deg, self.inner.zenith_deg, self.inner.value_m
        )
    }

    fn __eq__(&self, other: &PyAntexPcvSample) -> bool {
        self.inner == other.inner
    }
}

fn samples(samples: &[PcvSample]) -> Vec<PyAntexPcvSample> {
    samples
        .iter()
        .cloned()
        .map(|inner| PyAntexPcvSample { inner })
        .collect()
}

/// The `START OF FREQ RMS` section of one frequency: RMS of the eccentricities
/// in metres (None when the section has no `NORTH / EAST / UP` record) and of
/// the pattern values, placed on the antenna's grid.
#[pyclass(module = "sidereon._sidereon", name = "AntexFrequencyRms")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexFrequencyRms {
    inner: FrequencyRms,
}

#[pymethods]
impl PyAntexFrequencyRms {
    /// RMS of the north/east/up eccentricities, numpy `(3,)` metres, or None.
    #[getter]
    fn pco_m<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner.pco_m.map(|pco| np_array(py, &pco))
    }

    /// RMS of the pattern values.
    #[getter]
    fn pcv_samples(&self) -> Vec<PyAntexPcvSample> {
        samples(&self.inner.pcv_samples)
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexFrequencyRms(pco_m={:?}, pcv_samples={})",
            self.inner.pco_m,
            self.inner.pcv_samples.len()
        )
    }

    fn __eq__(&self, other: &PyAntexFrequencyRms) -> bool {
        self.inner == other.inner
    }
}

/// One frequency section of an antenna block: its label, its north/east/up
/// phase-center offset in metres, its PCV values, and its RMS section when the
/// file has one.
#[pyclass(module = "sidereon._sidereon", name = "AntexFrequency")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyAntexFrequency {
    inner: Frequency,
}

#[pymethods]
impl PyAntexFrequency {
    /// Frequency label from `START OF FREQUENCY`, such as `G01`.
    #[getter]
    fn frequency(&self) -> &str {
        &self.inner.frequency
    }

    /// North/east/up phase-center offset, numpy `(3,)` metres.
    #[getter]
    fn pco_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.pco_m)
    }

    /// PCV values in row and column order.
    #[getter]
    fn pcv_samples(&self) -> Vec<PyAntexPcvSample> {
        samples(&self.inner.pcv_samples)
    }

    /// The RMS section for this frequency, or None.
    #[getter]
    fn rms(&self) -> Option<PyAntexFrequencyRms> {
        self.inner
            .rms
            .clone()
            .map(|inner| PyAntexFrequencyRms { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexFrequency(frequency={:?}, pcv_samples={}, rms={})",
            self.inner.frequency,
            self.inner.pcv_samples.len(),
            self.inner.rms.is_some()
        )
    }

    fn __eq__(&self, other: &PyAntexFrequency) -> bool {
        self.inner == other.inner
    }
}

/// Widest fraction written out digit by digit in a repr; a finer one is
/// written as `digits E -scale`, which states it exactly in bounded space.
const REPR_FRACTION_DIGITS: u64 = 40;

/// The exact text of a whole second and its fraction, for a repr.
fn seconds_text(second: u8, fraction: SecondFraction) -> String {
    let (digits, scale) = (fraction.digits(), fraction.scale());
    if digits == 0 {
        return second.to_string();
    }
    match usize::try_from(scale) {
        Ok(width) if scale <= REPR_FRACTION_DIGITS => format!("{second}.{digits:0>width$}"),
        _ => format!("{second}+{digits}E-{scale}"),
    }
}

/// A `decimal.Decimal` holding `text` exactly.
fn decimal<'py>(py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("decimal")?.getattr("Decimal")?.call1((text,))
}

/// A GPS-time ANTEX validity instant from `VALID FROM` / `VALID UNTIL`.
///
/// GPS time has no leap-second label, so `second` is `0..=59`. The seconds
/// field is kept exactly: its fraction is `fraction_digits / 10**fraction_scale`,
/// which holds every decimal the thirteen-column field can state, including
/// ones far below a nanosecond. `fraction` gives it as an exact
/// `decimal.Decimal`. Instants order by value, whatever the scale of their
/// fractions.
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

#[pymethods]
impl PyAntexDateTime {
    /// Create a GPS-time validity instant with an exact fraction of a second,
    /// `fraction_digits / 10**fraction_scale`.
    ///
    /// A component outside the calendar or GPS clock ranges, including a `60`
    /// second, raises `ValueError`, as does a fraction that is not below one
    /// second.
    #[new]
    #[pyo3(signature = (
        year,
        month,
        day,
        hour=0,
        minute=0,
        second=0,
        *,
        fraction_digits=0,
        fraction_scale=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: u8,
        fraction_digits: u64,
        fraction_scale: u64,
    ) -> PyResult<Self> {
        let fraction = SecondFraction::new(fraction_digits, fraction_scale).ok_or_else(|| {
            PyValueError::new_err("fraction_digits / 10**fraction_scale must be below one second")
        })?;
        AntexDateTime::new_with_fraction(year, month, day, hour, minute, second, fraction)
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

    /// Whole second, `0..=59`.
    #[getter]
    fn second(&self) -> u8 {
        self.inner.second
    }

    /// Significant digits of the fraction, normalized with no trailing zero.
    #[getter]
    fn fraction_digits(&self) -> u64 {
        self.inner.fraction.digits()
    }

    /// Power of ten dividing `fraction_digits`.
    #[getter]
    fn fraction_scale(&self) -> u64 {
        self.inner.fraction.scale()
    }

    /// The fraction of a second as an exact `decimal.Decimal` in `[0, 1)`,
    /// `fraction_digits` scaled by `10**-fraction_scale`.
    #[getter]
    fn fraction<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let fraction = self.inner.fraction;
        decimal(py, &format!("{}E-{}", fraction.digits(), fraction.scale()))
    }

    /// The fraction in whole nanoseconds, or None when it is not a whole
    /// number of them.
    #[getter]
    fn nanosecond(&self) -> Option<u32> {
        self.inner.fraction.nanoseconds()
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        op.matches(self.inner.cmp(&other.inner))
    }

    fn __repr__(&self) -> String {
        format!(
            "AntexDateTime({:04}, {:02}, {:02}, {:02}, {:02}, {})",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            seconds_text(self.inner.second, self.inner.fraction)
        )
    }
}

/// Parsed ANTEX receiver and satellite antenna calibration product.
///
/// Every record the format defines is retained, and `to_antex_string` writes
/// each from its retained value, writing no record the source did not carry
/// apart from the start and end records of blocks and sections. Equality
/// compares every retained value.
#[pyclass(module = "sidereon._sidereon", name = "Antex")]
#[derive(Clone)]
pub struct PyAntex {
    inner: Antex,
}

#[pymethods]
impl PyAntex {
    /// Number of distinct `TYPE / SERIAL NO` ids.
    #[getter]
    fn antenna_count(&self) -> usize {
        self.inner.antennas.len()
    }

    /// `TYPE / SERIAL NO` ids in sorted order.
    #[getter]
    fn antenna_ids(&self) -> Vec<String> {
        self.inner.antennas.keys().cloned().collect()
    }

    /// Header records.
    #[getter]
    fn header(&self) -> PyAntexHeader {
        PyAntexHeader {
            inner: self.inner.header.clone(),
        }
    }

    /// `COMMENT` records outside the header and every block, in file order.
    #[getter]
    fn outer_comments(&self) -> Vec<PyAntexOuterComment> {
        self.inner
            .outer_comments
            .iter()
            .cloned()
            .map(|inner| PyAntexOuterComment { inner })
            .collect()
    }

    /// Every antenna block in file order, including every validity block of
    /// an id.
    #[getter]
    fn antenna_blocks(&self) -> Vec<PyAntenna> {
        self.inner
            .antenna_blocks()
            .cloned()
            .map(PyAntenna::from)
            .collect()
    }

    /// Number of records skipped or found inconsistent while parsing: a line
    /// outside any record, a `# OF FREQUENCIES` count the sections do not
    /// match, a frequency end record naming another frequency, a header record
    /// after `END OF HEADER`, a block or section its own end record does not
    /// close. The block or section is kept.
    #[getter]
    fn skipped_records(&self) -> usize {
        self.inner.skipped_records()
    }

    /// The latest block for a `TYPE / SERIAL NO` id, or None.
    fn antenna(&self, id: &str) -> Option<PyAntenna> {
        self.inner.antenna(id).cloned().map(PyAntenna::from)
    }

    /// Every validity block of a `TYPE / SERIAL NO` id, in file order.
    fn antenna_intervals(&self, id: &str) -> Vec<PyAntenna> {
        self.inner
            .antenna_intervals(id)
            .cloned()
            .map(PyAntenna::from)
            .collect()
    }

    /// The block for a `TYPE / SERIAL NO` id valid at `epoch`, or None.
    fn antenna_at(&self, id: &str, epoch: PyAntexDateTime) -> Option<PyAntenna> {
        self.inner
            .antenna_at(id, epoch.inner)
            .cloned()
            .map(PyAntenna::from)
    }

    /// The satellite antenna for `prn` valid at `epoch`, or None.
    fn satellite_antenna(&self, prn: &str, epoch: PyAntexDateTime) -> Option<PyAntenna> {
        self.inner
            .satellite_antenna(prn, epoch.inner)
            .cloned()
            .map(PyAntenna::from)
    }

    /// Serialize this product to ANTEX text via the core writer.
    ///
    /// A value the fixed columns cannot state exactly, a validity second no
    /// form with a decimal point fits, a frequency label that is not a system
    /// flag and a two-column number, or public fields that disagree with the
    /// retained blocks are refused, raising `AntexWriteError` with the core
    /// error as `detail` and its `field` and `reason`.
    fn to_antex_string(&self, py: Python<'_>) -> PyResult<String> {
        self.inner
            .encode()
            .map_err(|err| to_antex_write_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "Antex(antenna_count={}, blocks={})",
            self.inner.antennas.len(),
            self.inner.antenna_blocks().count()
        )
    }

    fn __eq__(&self, other: &PyAntex) -> bool {
        self.inner == other.inner
    }
}

/// Receiver or satellite ANTEX antenna calibration block.
///
/// PCO values are numpy `(3,)` north/east/up metres. PCV values are metres;
/// zenith and azimuth inputs are degrees. A record the block does not carry is
/// None, never a default.
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
    /// `TYPE / SERIAL NO` id.
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Receiver or satellite block role.
    #[getter]
    fn kind(&self) -> PyAntennaKind {
        self.inner.kind.into()
    }

    /// Antenna type field.
    #[getter]
    fn antenna_type(&self) -> String {
        self.inner.antenna_type.clone()
    }

    /// Serial, PRN, or radome field.
    #[getter]
    fn serial(&self) -> String {
        self.inner.serial.clone()
    }

    /// `COMMENT` text between `START OF ANTENNA` and `TYPE / SERIAL NO`.
    #[getter]
    fn leading_comments(&self) -> Vec<String> {
        self.inner.leading_comments.clone()
    }

    /// Every `METH / BY / # / DATE` record, in file order.
    #[getter]
    fn calibrations(&self) -> Vec<PyAntexCalibration> {
        self.inner
            .calibrations
            .iter()
            .cloned()
            .map(|inner| PyAntexCalibration { inner })
            .collect()
    }

    /// `DAZI` in degrees, or None when the block has no such record.
    #[getter]
    fn dazi_deg(&self) -> Option<f64> {
        self.inner.dazi_deg
    }

    /// `ZEN1 / ZEN2 / DZEN`, or None when the block has no such record (it
    /// then has no PCV values).
    #[getter]
    fn zenith_grid(&self) -> Option<PyAntexZenithGrid> {
        self.inner
            .zenith_grid
            .map(|inner| PyAntexZenithGrid { inner })
    }

    /// `ZEN1` in degrees, or None when the block has no zenith grid.
    #[getter]
    fn zenith_start_deg(&self) -> Option<f64> {
        self.inner.zenith_grid.map(|grid| grid.start_deg)
    }

    /// `ZEN2` in degrees, or None when the block has no zenith grid.
    #[getter]
    fn zenith_end_deg(&self) -> Option<f64> {
        self.inner.zenith_grid.map(|grid| grid.end_deg)
    }

    /// `DZEN` in degrees, or None when the block has no zenith grid.
    #[getter]
    fn zenith_step_deg(&self) -> Option<f64> {
        self.inner.zenith_grid.map(|grid| grid.step_deg)
    }

    /// Whether the block carries `# OF FREQUENCIES`.
    #[getter]
    fn has_frequency_count(&self) -> bool {
        self.inner.has_frequency_count
    }

    /// Nonblank `SINEX CODE`, or None.
    #[getter]
    fn sinex_code(&self) -> Option<String> {
        self.inner.sinex_code.clone()
    }

    /// `VALID FROM`, the inclusive lower bound of `valid_at`, or None.
    #[getter]
    fn valid_from(&self) -> Option<PyAntexDateTime> {
        self.inner.valid_from.map(PyAntexDateTime::from)
    }

    /// `VALID UNTIL`, the inclusive upper bound of `valid_at`, or None.
    #[getter]
    fn valid_until(&self) -> Option<PyAntexDateTime> {
        self.inner.valid_until.map(PyAntexDateTime::from)
    }

    /// `COMMENT` text inside the block after `TYPE / SERIAL NO`.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Frequency labels of the sections in file order; a label repeated in the
    /// file is repeated here.
    #[getter]
    fn frequencies(&self) -> Vec<String> {
        self.inner
            .frequencies
            .iter()
            .map(|frequency| frequency.frequency.clone())
            .collect()
    }

    /// Frequency sections in file order.
    #[getter]
    fn frequency_sections(&self) -> Vec<PyAntexFrequency> {
        self.inner
            .frequencies
            .iter()
            .cloned()
            .map(|inner| PyAntexFrequency { inner })
            .collect()
    }

    /// Whether this block is valid at `epoch`.
    fn valid_at(&self, epoch: PyAntexDateTime) -> bool {
        self.inner.valid_at(epoch.inner)
    }

    /// The frequency section with the trimmed label. Several sections with the
    /// label must be identical; otherwise the lookup is refused as
    /// `AmbiguousFrequency`. Refusals raise `AntexQueryError`.
    fn frequency(&self, py: Python<'_>, frequency: &str) -> PyResult<PyAntexFrequency> {
        self.inner
            .frequency(frequency)
            .cloned()
            .map(|inner| PyAntexFrequency { inner })
            .map_err(|err| to_antex_query_err(py, err))
    }

    /// Frequency-dependent phase-center offset, numpy `(3,)` north/east/up
    /// metres. An unknown or ambiguous label raises `AntexQueryError`.
    fn pco<'py>(&self, py: Python<'py>, frequency: &str) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let pco = self
            .inner
            .pco(frequency)
            .map_err(|err| to_antex_query_err(py, err))?;
        Ok(np_array(py, &pco))
    }

    /// Frequency-dependent phase-center variation in metres.
    ///
    /// `zenith_deg` and optional `azimuth_deg` are degrees. If the antenna has
    /// no azimuth grid, the core no-azimuth interpolation is used. A zenith
    /// outside the grid, and an unknown or ambiguous label, raise
    /// `AntexQueryError`.
    #[pyo3(signature = (frequency, zenith_deg, azimuth_deg=None))]
    fn pcv(
        &self,
        py: Python<'_>,
        frequency: &str,
        zenith_deg: f64,
        azimuth_deg: Option<f64>,
    ) -> PyResult<f64> {
        require_finite("zenith_deg", zenith_deg)?;
        if let Some(value) = azimuth_deg {
            require_finite("azimuth_deg", value)?;
        }
        self.inner
            .pcv(frequency, zenith_deg, azimuth_deg)
            .map_err(|err| to_antex_query_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "Antenna(id={:?}, kind={})",
            self.inner.id,
            self.kind().label()
        )
    }

    fn __eq__(&self, other: &PyAntenna) -> bool {
        self.inner == other.inner
    }
}

/// Parse an ANTEX 1.4 antenna product from in-memory bytes or a file path.
///
/// `source` may be bytes / bytearray containing the full ASCII text, or a path
/// (`str` / `os.PathLike`) to read. PCO/PCV values are exposed in metres. A
/// refusal raises `AntexParseError` with the core error as `detail`.
#[pyfunction]
fn load_antex(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<PyAntex> {
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        return parse_antex_bytes(py, bytes.as_bytes());
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        return parse_antex_bytes(py, unsafe { buf.as_bytes() });
    }
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_antex expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    parse_antex_bytes(py, &data)
}

fn parse_antex_bytes(py: Python<'_>, bytes: &[u8]) -> PyResult<PyAntex> {
    let text = std::str::from_utf8(bytes).map_err(to_antex_err)?;
    let inner = Antex::parse(text).map_err(|err| to_antex_parse_err(py, err))?;
    Ok(PyAntex { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyAntexErrorDetail>()?;
    m.add_class::<PyAntennaKind>()?;
    m.add_class::<PyAntexPcvType>()?;
    m.add_class::<PyAntexPcvGrid>()?;
    m.add_class::<PyAntexVersion>()?;
    m.add_class::<PyAntexPcvTypeRecord>()?;
    m.add_class::<PyAntexHeader>()?;
    m.add_class::<PyAntexOuterComment>()?;
    m.add_class::<PyAntexCalibration>()?;
    m.add_class::<PyAntexZenithGrid>()?;
    m.add_class::<PyAntexPcvSample>()?;
    m.add_class::<PyAntexFrequencyRms>()?;
    m.add_class::<PyAntexFrequency>()?;
    m.add_class::<PyAntexDateTime>()?;
    m.add_class::<PyAntex>()?;
    m.add_class::<PyAntenna>()?;
    m.add_function(wrap_pyfunction!(load_antex, m)?)?;
    Ok(())
}
