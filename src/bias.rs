//! GNSS bias product binding.

use std::path::PathBuf;
use std::str::FromStr;

use pyo3::create_exception;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

use sidereon::bias::{
    bias_epoch_instant, BiasDeparture, BiasEpoch, BiasError as CoreBiasError, BiasKind, BiasLookup,
    BiasMode, BiasNotice, BiasObservableFamily, BiasReadPolicy, BiasRecord, BiasSet, BiasTarget,
    CodeDcbOptions, Diagnostics,
};
use sidereon_core::astro::time::model::Instant;
use sidereon_core::GnssSatelliteId;

use crate::frames::PyTimeScale;
use crate::marshal::{debug_variant_snake, PyGnssSystem};

create_exception!(
    _sidereon,
    BiasError,
    PyValueError,
    "A bias operation failed with a typed core error."
);

pub(crate) fn bias_error_kind(error: &CoreBiasError) -> &'static str {
    match error {
        CoreBiasError::InvalidInput { .. } => "invalid_input",
        CoreBiasError::InvalidEpoch => "invalid_epoch",
        CoreBiasError::UnknownObservable { .. } => "unknown_observable",
        CoreBiasError::UnsupportedVersion { .. } => "unsupported_version",
        CoreBiasError::MissingDcbMetadata => "missing_dcb_metadata",
        CoreBiasError::MissingClockReference => "missing_clock_reference",
        CoreBiasError::MissingWriterMetadata { .. } => "missing_writer_metadata",
        CoreBiasError::Utf8 => "utf8",
        CoreBiasError::Departure { .. } => "departure",
        CoreBiasError::InvalidUtf8Line { .. } => "invalid_utf8_line",
        CoreBiasError::UnsupportedTimeSystem { .. } => "unsupported_time_system",
        CoreBiasError::DcbRecordMismatch { .. } => "dcb_record_mismatch",
    }
}

pub(crate) fn bias_departure_details<'py>(
    py: Python<'py>,
    departure: &BiasDeparture,
) -> PyResult<Bound<'py, PyDict>> {
    let details = PyDict::new(py);
    let kind = match departure {
        BiasDeparture::HeaderLayout { reason } => {
            details.set_item("reason", reason)?;
            "header_layout"
        }
        BiasDeparture::OtherVersion { version } => {
            details.set_item("version", version)?;
            "other_version"
        }
        BiasDeparture::MissingFooter => "missing_footer",
        BiasDeparture::ContentAfterFooter { line } => {
            details.set_item("line", line)?;
            "content_after_footer"
        }
        BiasDeparture::UnexpectedControlLine { line } => {
            details.set_item("line", line)?;
            "unexpected_control_line"
        }
        BiasDeparture::UnclosedBlock { name, line } => {
            details.set_item("name", name)?;
            details.set_item("line", line)?;
            "unclosed_block"
        }
        BiasDeparture::UnopenedBlockEnd { name, line } => {
            details.set_item("name", name)?;
            details.set_item("line", line)?;
            "unopened_block_end"
        }
        BiasDeparture::MismatchedBlockEnd { open, close, line } => {
            details.set_item("open", open)?;
            details.set_item("close", close)?;
            details.set_item("line", line)?;
            "mismatched_block_end"
        }
        BiasDeparture::NestedBlock { open, inner, line } => {
            details.set_item("open", open)?;
            details.set_item("inner", inner)?;
            details.set_item("line", line)?;
            "nested_block"
        }
        BiasDeparture::MissingBlock { name } => {
            details.set_item("name", name)?;
            "missing_block"
        }
        BiasDeparture::UnknownBlock { name, line } => {
            details.set_item("name", name)?;
            details.set_item("line", line)?;
            "unknown_block"
        }
        BiasDeparture::BlockStartSuffix { line } => {
            details.set_item("line", line)?;
            "block_start_suffix"
        }
        BiasDeparture::DataOutsideBlock { line } => {
            details.set_item("line", line)?;
            "data_outside_block"
        }
        BiasDeparture::MissingDeclaration { keyword } => {
            details.set_item("keyword", keyword)?;
            "missing_declaration"
        }
        BiasDeparture::UnsupportedBiasMode { line, label } => {
            details.set_item("line", line)?;
            details.set_item("label", label)?;
            "unsupported_bias_mode"
        }
        BiasDeparture::NonStandardTimeSystem { line, label } => {
            details.set_item("line", line)?;
            details.set_item("label", label)?;
            "non_standard_time_system"
        }
        BiasDeparture::HeaderModeMismatch {
            header,
            description,
        } => {
            details.set_item("header", header)?;
            details.set_item("description", debug_variant_snake(description))?;
            "header_mode_mismatch"
        }
        BiasDeparture::UnknownDcbTimeSystem { line, label } => {
            details.set_item("line", line)?;
            details.set_item("label", label)?;
            "unknown_dcb_time_system"
        }
        BiasDeparture::EstimateCountMismatch {
            declared,
            solution_rows,
        } => {
            details.set_item("declared", declared)?;
            details.set_item("solution_rows", solution_rows)?;
            "estimate_count_mismatch"
        }
        _ => {
            details.set_item("variant", debug_variant_snake(departure))?;
            "unknown"
        }
    };
    details.set_item("kind", kind)?;
    Ok(details)
}

pub(crate) fn bias_error_details<'py>(
    py: Python<'py>,
    error: &CoreBiasError,
) -> PyResult<Bound<'py, PyDict>> {
    let details = PyDict::new(py);
    match error {
        CoreBiasError::InvalidEpoch
        | CoreBiasError::MissingDcbMetadata
        | CoreBiasError::MissingClockReference
        | CoreBiasError::Utf8 => {}
        CoreBiasError::InvalidInput { field, reason } => {
            details.set_item("field", field)?;
            details.set_item("reason", reason)?;
        }
        CoreBiasError::UnknownObservable { code } => details.set_item("code", code)?,
        CoreBiasError::UnsupportedVersion { version } => details.set_item("version", version)?,
        CoreBiasError::MissingWriterMetadata { field } => details.set_item("field", field)?,
        CoreBiasError::Departure { departure } => {
            details.set_item("departure", bias_departure_details(py, departure)?)?;
        }
        CoreBiasError::InvalidUtf8Line { line } => details.set_item("line", line)?,
        CoreBiasError::UnsupportedTimeSystem { scale } => {
            details.set_item("scale", scale.map(PyTimeScale::from))?;
        }
        CoreBiasError::DcbRecordMismatch { record, field } => {
            details.set_item("record", record)?;
            details.set_item("field", field)?;
        }
    }
    Ok(details)
}

pub(crate) fn bias_error_details_value(
    py: Python<'_>,
    error: &CoreBiasError,
) -> PyResult<Py<PyDict>> {
    Ok(bias_error_details(py, error)?.unbind())
}

fn bias_notice_details(py: Python<'_>, notice: &BiasNotice) -> PyResult<Py<PyDict>> {
    let details = PyDict::new(py);
    let kind = match notice {
        BiasNotice::Departure(departure) => {
            details.set_item("departure", bias_departure_details(py, departure)?)?;
            "departure"
        }
        BiasNotice::InvalidUtf8 { line } => {
            details.set_item("line", line)?;
            "invalid_utf8"
        }
        BiasNotice::RepeatedDeclaration { line, keyword } => {
            details.set_item("line", line)?;
            details.set_item("keyword", keyword)?;
            "repeated_declaration"
        }
        BiasNotice::ConflictingDeclaration { line, keyword } => {
            details.set_item("line", line)?;
            details.set_item("keyword", keyword)?;
            "conflicting_declaration"
        }
        BiasNotice::Overlap { first, second } => {
            details.set_item("first", first)?;
            details.set_item("second", second)?;
            "overlap"
        }
        BiasNotice::DcbTimeSystemAssumed => "dcb_time_system_assumed",
        BiasNotice::DcbTimeSystemAlias { line, label } => {
            details.set_item("line", line)?;
            details.set_item("label", label)?;
            "dcb_time_system_alias"
        }
        _ => {
            details.set_item("variant", debug_variant_snake(notice))?;
            "unknown"
        }
    };
    details.set_item("kind", kind)?;
    Ok(details.unbind())
}

fn to_bias_err(error: CoreBiasError) -> PyErr {
    Python::with_gil(|py| {
        let exception = PyErr::from_type(py.get_type::<BiasError>(), error.to_string());
        let value = exception.value(py);
        let _ = value.setattr("kind", bias_error_kind(&error));
        if let Ok(details) = bias_error_details_value(py, &error) {
            let _ = value.setattr("details", details);
        }
        exception
    })
}

fn to_bias_facade_err(error: sidereon::Error) -> PyErr {
    match error {
        sidereon::Error::Bias(source) => to_bias_err(source),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    GnssSatelliteId::from_str(token)
        .map_err(|_| PyValueError::new_err(format!("invalid satellite token: {token}")))
}

fn epoch_tuple(epoch: Option<BiasEpoch>) -> Option<(i32, u16, u32)> {
    epoch.map(|e| (e.year, e.day_of_year, e.second_of_day))
}

/// How the Bias-SINEX and CODE DCB readers treat a file that departs from the
/// format, mirroring the core `BiasReadPolicy`.
#[pyclass(module = "sidereon._sidereon", name = "BiasReadPolicy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyBiasReadPolicy {
    /// Refuse a file that departs from Bias-SINEX 1.00, naming the first
    /// departure, and a version other than `1.00`.
    STRICT,
    /// Read the file and report every departure as a notice.
    LENIENT,
}

impl From<PyBiasReadPolicy> for BiasReadPolicy {
    fn from(value: PyBiasReadPolicy) -> Self {
        match value {
            PyBiasReadPolicy::STRICT => BiasReadPolicy::Strict,
            PyBiasReadPolicy::LENIENT => BiasReadPolicy::Lenient,
        }
    }
}

/// Outcome of a bias lookup, mirroring the core `BiasLookup`.
///
/// `status` names the outcome: `available` (with `value`, `records` and
/// `overridden`), `absent`, `unsupported_scale` (with
/// `product_time_scale` and `query_time_scale`), `ambiguous` (with the
/// conflicting `records`), `carrier_frequency_required` (with `record`),
/// `invalid_carrier_frequency`, `carrier_frequency_unknown` (with
/// `observable`), `undefined_slope_reference` (with `record`) or
/// `invalid_epoch`.
#[pyclass(module = "sidereon._sidereon", name = "BiasLookup")]
#[derive(Clone)]
pub struct PyBiasLookup {
    inner: BiasLookup,
}

#[pymethods]
impl PyBiasLookup {
    /// Outcome label.
    #[getter]
    fn status(&self) -> String {
        let label = match &self.inner {
            BiasLookup::Available { .. } => "available",
            BiasLookup::Absent => "absent",
            BiasLookup::UnsupportedScale { .. } => "unsupported_scale",
            BiasLookup::Ambiguous { .. } => "ambiguous",
            BiasLookup::CarrierFrequencyRequired { .. } => "carrier_frequency_required",
            BiasLookup::InvalidCarrierFrequency => "invalid_carrier_frequency",
            BiasLookup::CarrierFrequencyUnknown { .. } => "carrier_frequency_unknown",
            BiasLookup::UndefinedSlopeReference { .. } => "undefined_slope_reference",
            BiasLookup::InvalidEpoch => "invalid_epoch",
            other => return debug_variant_snake(other),
        };
        label.to_string()
    }

    /// The value when it is available, in the unit the query names.
    #[getter]
    fn value(&self) -> Option<f64> {
        self.inner.value()
    }

    /// True when a value is available.
    #[getter]
    fn is_available(&self) -> bool {
        self.inner.is_available()
    }

    /// Indices into `BiasSet.records`: the records an available value comes
    /// from, or the conflicting records of an ambiguous lookup. Empty for
    /// every other outcome.
    #[getter]
    fn records(&self) -> Vec<usize> {
        match &self.inner {
            BiasLookup::Available { records, .. } | BiasLookup::Ambiguous { records } => {
                records.clone()
            }
            _ => Vec::new(),
        }
    }

    /// Records that also cover the query epoch but start earlier than a
    /// selected record, which overrides them. Empty unless available.
    #[getter]
    fn overridden(&self) -> Vec<usize> {
        match &self.inner {
            BiasLookup::Available { overridden, .. } => overridden.clone(),
            _ => Vec::new(),
        }
    }

    /// The record a `carrier_frequency_required` or
    /// `undefined_slope_reference` outcome names.
    #[getter]
    fn record(&self) -> Option<usize> {
        match &self.inner {
            BiasLookup::CarrierFrequencyRequired { record }
            | BiasLookup::UndefinedSlopeReference { record } => Some(*record),
            _ => None,
        }
    }

    /// The observable of a `carrier_frequency_unknown` outcome.
    #[getter]
    fn observable(&self) -> Option<String> {
        match &self.inner {
            BiasLookup::CarrierFrequencyUnknown { observable } => Some(observable.clone()),
            _ => None,
        }
    }

    /// The product time scale of an `unsupported_scale` outcome, `None` when
    /// the product declares none or for another outcome.
    #[getter]
    fn product_time_scale(&self) -> Option<PyTimeScale> {
        match &self.inner {
            BiasLookup::UnsupportedScale { product, .. } => product.map(Into::into),
            _ => None,
        }
    }

    /// The query time scale of an `unsupported_scale` outcome.
    #[getter]
    fn query_time_scale(&self) -> Option<PyTimeScale> {
        match &self.inner {
            BiasLookup::UnsupportedScale { query, .. } => Some((*query).into()),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner.value() {
            Some(value) => format!("BiasLookup(status='available', value={value:e})"),
            None => format!("BiasLookup(status={:?})", self.status()),
        }
    }
}

fn epoch(year: i32, day_of_year: u16, second_of_day: u32, scale: PyTimeScale) -> PyResult<Instant> {
    bias_epoch_instant(
        BiasEpoch::new(year, day_of_year, second_of_day).map_err(to_bias_err)?,
        scale.into(),
    )
    .map_err(to_bias_err)
}

#[pyclass(module = "sidereon._sidereon", name = "CodeDcbOptions")]
#[derive(Clone)]
/// Options for parsing legacy CODE DCB files.
pub struct PyCodeDcbOptions {
    inner: CodeDcbOptions,
}

impl PyCodeDcbOptions {
    fn inner(&self) -> CodeDcbOptions {
        self.inner.clone()
    }
}

#[pymethods]
impl PyCodeDcbOptions {
    /// Build CODE DCB parser options.
    #[new]
    #[pyo3(signature = (obs1, obs2, year, month, time_scale=PyTimeScale::GPST, receiver_system=None))]
    fn new(
        obs1: String,
        obs2: String,
        year: i32,
        month: u8,
        time_scale: PyTimeScale,
        receiver_system: Option<PyGnssSystem>,
    ) -> Self {
        let mut inner =
            CodeDcbOptions::new((obs1.clone(), obs2.clone()), year, month, time_scale.into());
        inner.pair = (obs1, obs2);
        inner.year = year;
        inner.month = month;
        inner.time_scale = time_scale.into();
        inner.receiver_system = receiver_system.map(Into::into);
        Self { inner }
    }

    fn __repr__(&self) -> String {
        format!(
            "CodeDcbOptions(obs1={:?}, obs2={:?}, year={}, month={})",
            self.inner.pair.0, self.inner.pair.1, self.inner.year, self.inner.month
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "BiasRecord")]
#[derive(Clone)]
/// One GNSS code or phase bias record.
pub struct PyBiasRecord {
    inner: BiasRecord,
}

#[pymethods]
impl PyBiasRecord {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            BiasKind::Osb => "OSB",
            BiasKind::Dsb => "DSB",
            BiasKind::Isb => "ISB",
        }
    }

    #[getter]
    fn target(&self) -> String {
        match &self.inner.target {
            BiasTarget::System(system) => system.as_str().to_string(),
            BiasTarget::Satellite(sat) => sat.to_string(),
            BiasTarget::Receiver { system, station } => format!("{}:{station}", system.as_str()),
            BiasTarget::SatelliteReceiver { sat, station } => format!("{sat}:{station}"),
        }
    }

    #[getter]
    fn obs1(&self) -> &str {
        &self.inner.obs1
    }

    #[getter]
    fn obs2(&self) -> Option<&str> {
        self.inner.obs2.as_deref()
    }

    #[getter]
    fn value(&self) -> f64 {
        self.inner.value
    }

    #[getter]
    fn sigma(&self) -> Option<f64> {
        self.inner.sigma
    }

    /// True for a phase bias, one whose observable is an `L` code.
    #[getter]
    fn is_phase(&self) -> bool {
        self.inner.is_phase()
    }

    /// Observable family from the observable code (Bias-SINEX 1.00 section
    /// 4.8): `code`, `phase`, or `mixed` for a DSB or ISB pairing a code with
    /// a phase observable, which code and phase lookups do not use.
    #[getter]
    fn family(&self) -> &'static str {
        match self.inner.family {
            BiasObservableFamily::Code => "code",
            BiasObservableFamily::Phase => "phase",
            BiasObservableFamily::Mixed => "mixed",
        }
    }

    /// Unit the source row states its values in: `ns` (stored in seconds) or
    /// `cyc` (stored in cycles).
    #[getter]
    fn unit(&self) -> &'static str {
        self.inner.unit.label()
    }

    /// Optional SVN text from the solution row.
    #[getter]
    fn svn(&self) -> Option<&str> {
        self.inner.svn.as_deref()
    }

    /// Inclusive validity start as `(year, day_of_year, second_of_day)`.
    #[getter]
    fn valid_from(&self) -> Option<(i32, u16, u32)> {
        epoch_tuple(self.inner.valid_from)
    }

    /// Exclusive validity end as `(year, day_of_year, second_of_day)`. An end
    /// written at second 86399 reads as the following midnight.
    #[getter]
    fn valid_until(&self) -> Option<(i32, u16, u32)> {
        epoch_tuple(self.inner.valid_until)
    }

    /// Start and end epoch tokens as the row writes them, trimmed.
    #[getter]
    fn raw_epochs(&self) -> (String, String) {
        self.inner.raw_epochs.clone()
    }

    /// Optional slope per second, in the units of `value`.
    #[getter]
    fn slope(&self) -> Option<f64> {
        self.inner.slope
    }

    /// Optional slope uncertainty.
    #[getter]
    fn slope_sigma(&self) -> Option<f64> {
        self.inner.slope_sigma
    }

    /// One-based source line of the row, when the record was read from text.
    #[getter]
    fn line(&self) -> Option<usize> {
        self.inner.line
    }

    fn __repr__(&self) -> String {
        format!(
            "BiasRecord(kind={:?}, target={:?}, obs1={:?}, obs2={:?}, value={:.6e})",
            self.kind(),
            self.target(),
            self.inner.obs1,
            self.inner.obs2,
            self.inner.value
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "BiasSet")]
#[derive(Clone)]
/// Parsed GNSS bias set with lookup helpers.
pub struct PyBiasSet {
    pub(crate) inner: BiasSet,
}

impl PyBiasSet {
    pub(crate) fn inner(&self) -> BiasSet {
        self.inner.clone()
    }

    /// The scale a query epoch is read on: the caller's, else the product's.
    /// A product that declares no usable time scale answers no query on any
    /// scale, so the caller has to name the query's scale.
    fn query_scale(&self, time_scale: Option<PyTimeScale>) -> PyResult<PyTimeScale> {
        match time_scale {
            Some(scale) => Ok(scale),
            None => self.inner.time_scale().map(Into::into).ok_or_else(|| {
                PyValueError::new_err(
                    "the bias product declares no usable time scale; pass time_scale",
                )
            }),
        }
    }
}

#[pymethods]
impl PyBiasSet {
    #[getter]
    fn record_count(&self) -> usize {
        self.inner.records().len()
    }

    #[getter]
    fn skipped_records(&self) -> usize {
        self.inner.skipped_records()
    }

    /// Time scale the product declares, `None` when it declares no usable
    /// one (a lenient read without `TIME_SYSTEM`, or a label naming no scale).
    #[getter]
    fn time_scale(&self) -> Option<PyTimeScale> {
        self.inner.time_scale().map(Into::into)
    }

    /// The `TIME_SYSTEM` label as written, or the DCB title label.
    #[getter]
    fn time_system_label(&self) -> Option<&str> {
        self.inner.time_system_label()
    }

    /// Declared `BIAS_MODE`: `absolute`, `relative`, or `unspecified` when
    /// it is missing, unrecognized or declared twice with different values.
    #[getter]
    fn mode(&self) -> &'static str {
        match self.inner.mode() {
            BiasMode::Absolute => "absolute",
            BiasMode::Relative => "relative",
            BiasMode::Unspecified => "unspecified",
        }
    }

    /// Satellite clock reference observables per system, as
    /// `(system, obs1, obs2)` in system order.
    #[getter]
    fn clock_reference(&self) -> Vec<(PyGnssSystem, String, String)> {
        self.inner
            .clock_reference()
            .per_system
            .iter()
            .map(|(system, (obs1, obs2))| ((*system).into(), obs1.clone(), obs2.clone()))
            .collect()
    }

    /// Non-fatal findings about the product (departures read under the
    /// lenient policy, invalid UTF-8 lines, repeated or conflicting
    /// declarations, overlapping records), each as the core states it.
    #[getter]
    fn notices(&self) -> Vec<String> {
        self.inner
            .notices()
            .iter()
            .map(|notice| format!("{notice:?}"))
            .collect()
    }

    /// Structured notices with stable `kind` values and variant fields.
    #[getter]
    fn notice_details(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        self.inner
            .notices()
            .iter()
            .map(|notice| bias_notice_details(py, notice))
            .collect()
    }

    #[getter]
    fn records(&self) -> Vec<PyBiasRecord> {
        self.inner
            .records()
            .iter()
            .cloned()
            .map(|inner| PyBiasRecord { inner })
            .collect()
    }

    #[pyo3(signature = (satellite, obs, year, day_of_year, second_of_day, time_scale=None))]
    fn code_osb_seconds(
        &self,
        satellite: &str,
        obs: &str,
        year: i32,
        day_of_year: u16,
        second_of_day: u32,
        time_scale: Option<PyTimeScale>,
    ) -> PyResult<PyBiasLookup> {
        let scale = self.query_scale(time_scale)?;
        let inner = self.inner.code_osb_seconds(
            parse_sat(satellite)?,
            obs,
            epoch(year, day_of_year, second_of_day, scale)?,
        );
        Ok(PyBiasLookup { inner })
    }

    /// Covering phase OSB in cycles. A record stated in nanoseconds is
    /// converted with `carrier_hz`, the carrier frequency of `obs` for this
    /// satellite (for a GLONASS FDMA signal, that of its channel); without it
    /// the lookup's status is `carrier_frequency_required`.
    #[pyo3(signature = (satellite, obs, year, day_of_year, second_of_day, time_scale=None, carrier_hz=None))]
    #[allow(clippy::too_many_arguments)]
    fn phase_osb_cycles(
        &self,
        satellite: &str,
        obs: &str,
        year: i32,
        day_of_year: u16,
        second_of_day: u32,
        time_scale: Option<PyTimeScale>,
        carrier_hz: Option<f64>,
    ) -> PyResult<PyBiasLookup> {
        let scale = self.query_scale(time_scale)?;
        let inner = self.inner.phase_osb_cycles(
            parse_sat(satellite)?,
            obs,
            epoch(year, day_of_year, second_of_day, scale)?,
            carrier_hz,
        );
        Ok(PyBiasLookup { inner })
    }

    #[pyo3(signature = (satellite, obs1, obs2, year, day_of_year, second_of_day, time_scale=None))]
    #[allow(clippy::too_many_arguments)]
    fn code_dsb_seconds(
        &self,
        satellite: &str,
        obs1: &str,
        obs2: &str,
        year: i32,
        day_of_year: u16,
        second_of_day: u32,
        time_scale: Option<PyTimeScale>,
    ) -> PyResult<PyBiasLookup> {
        let scale = self.query_scale(time_scale)?;
        let inner = self.inner.code_dsb_seconds(
            parse_sat(satellite)?,
            obs1,
            obs2,
            epoch(year, day_of_year, second_of_day, scale)?,
        );
        Ok(PyBiasLookup { inner })
    }

    #[pyo3(signature = (satellite, used_obs1, used_obs2, freq1_hz, freq2_hz, clock_ref_obs1, clock_ref_obs2, year, day_of_year, second_of_day, glonass_channel=None, time_scale=None))]
    #[allow(clippy::too_many_arguments)]
    fn code_bias_model_m(
        &self,
        satellite: &str,
        used_obs1: &str,
        used_obs2: &str,
        freq1_hz: f64,
        freq2_hz: f64,
        clock_ref_obs1: &str,
        clock_ref_obs2: &str,
        year: i32,
        day_of_year: u16,
        second_of_day: u32,
        glonass_channel: Option<i8>,
        time_scale: Option<PyTimeScale>,
    ) -> PyResult<PyBiasLookup> {
        let scale = self.query_scale(time_scale)?;
        let inner = self.inner.code_bias_model_m(
            parse_sat(satellite)?,
            (used_obs1, used_obs2),
            (freq1_hz, freq2_hz),
            glonass_channel,
            (clock_ref_obs1, clock_ref_obs2),
            epoch(year, day_of_year, second_of_day, scale)?,
        );
        Ok(PyBiasLookup { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "BiasSet(record_count={}, skipped_records={})",
            self.inner.records().len(),
            self.inner.skipped_records()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "BiasParsed")]
/// Lossy bias parse result containing a value and diagnostics.
pub struct PyBiasParsed {
    value: BiasSet,
    diagnostics: Diagnostics,
}

#[pymethods]
impl PyBiasParsed {
    /// The parsed bias set.
    #[getter]
    fn value(&self) -> PyBiasSet {
        PyBiasSet {
            inner: self.value.clone(),
        }
    }

    /// Non-fatal diagnostics collected while reading the bias set.
    #[getter]
    fn diagnostics(&self) -> crate::format_diagnostics::PyFormatDiagnostics {
        crate::format_diagnostics::PyFormatDiagnostics::from_inner(self.diagnostics.clone())
    }

    /// Number of records skipped during parsing.
    #[getter]
    fn skip_count(&self) -> usize {
        self.diagnostics.skips.len()
    }

    /// Number of advisory warnings collected during parsing.
    #[getter]
    fn warning_count(&self) -> usize {
        self.diagnostics.warnings.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "BiasParsed(record_count={}, skip_count={}, warning_count={})",
            self.value.records().len(),
            self.diagnostics.skips.len(),
            self.diagnostics.warnings.len()
        )
    }
}

fn lossy(parsed: sidereon::bias::Parsed<BiasSet>) -> PyBiasParsed {
    let (value, diagnostics) = parsed.into_parts();
    PyBiasParsed { value, diagnostics }
}

#[pyfunction]
#[pyo3(signature = (bytes, policy=PyBiasReadPolicy::STRICT))]
/// Parse Bias-SINEX bytes into a bias set.
///
/// Under `BiasReadPolicy.STRICT` a file that departs from Bias-SINEX 1.00,
/// or states another version, raises `ValueError`; `LENIENT` reads it and
/// reports each departure in `BiasSet.notices`.
fn parse_bias_sinex(bytes: Vec<u8>, policy: PyBiasReadPolicy) -> PyResult<PyBiasSet> {
    sidereon::parse_bias_sinex_with_policy(&bytes, policy.into())
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_facade_err)
}

#[pyfunction]
#[pyo3(signature = (bytes, policy=PyBiasReadPolicy::STRICT))]
/// Parse Bias-SINEX bytes and return diagnostics for skipped records.
///
/// The parsed value is available as `result.value`.
fn parse_bias_sinex_lossy(bytes: Vec<u8>, policy: PyBiasReadPolicy) -> PyResult<PyBiasParsed> {
    sidereon::parse_bias_sinex_lossy_with_policy(&bytes, policy.into())
        .map(lossy)
        .map_err(to_bias_facade_err)
}

#[pyfunction]
#[pyo3(signature = (path, policy=PyBiasReadPolicy::STRICT))]
/// Load a Bias-SINEX file from a filesystem path.
fn load_bias_sinex(path: PathBuf, policy: PyBiasReadPolicy) -> PyResult<PyBiasSet> {
    sidereon::load_bias_sinex_with_policy(path, policy.into())
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_facade_err)
}

#[pyfunction]
#[pyo3(signature = (path, policy=PyBiasReadPolicy::STRICT))]
/// Load a Bias-SINEX file and return diagnostics for skipped records.
fn load_bias_sinex_lossy(path: PathBuf, policy: PyBiasReadPolicy) -> PyResult<PyBiasParsed> {
    sidereon::load_bias_sinex_lossy_with_policy(path, policy.into())
        .map(lossy)
        .map_err(to_bias_facade_err)
}

#[pyfunction]
#[pyo3(signature = (bytes, options, policy=PyBiasReadPolicy::STRICT))]
/// Parse legacy CODE DCB bytes into a bias set.
///
/// Under `BiasReadPolicy.STRICT` a generated title whose time-system label
/// names no known scale, with no options, raises `ValueError`; `LENIENT`
/// reads the rows, leaves the set without a time scale and reports it.
fn parse_code_dcb(
    bytes: Vec<u8>,
    options: Option<&PyCodeDcbOptions>,
    policy: PyBiasReadPolicy,
) -> PyResult<PyBiasSet> {
    sidereon::parse_code_dcb_with_policy(
        &bytes,
        options.map(PyCodeDcbOptions::inner),
        policy.into(),
    )
    .map(|inner| PyBiasSet { inner })
    .map_err(to_bias_facade_err)
}

#[pyfunction]
#[pyo3(signature = (bytes, options, policy=PyBiasReadPolicy::STRICT))]
/// Parse legacy CODE DCB bytes and return diagnostics for skipped records.
fn parse_code_dcb_lossy(
    bytes: Vec<u8>,
    options: Option<&PyCodeDcbOptions>,
    policy: PyBiasReadPolicy,
) -> PyResult<PyBiasParsed> {
    sidereon::parse_code_dcb_lossy_with_policy(
        &bytes,
        options.map(PyCodeDcbOptions::inner),
        policy.into(),
    )
    .map(lossy)
    .map_err(to_bias_facade_err)
}

#[pyfunction]
#[pyo3(signature = (path, options, policy=PyBiasReadPolicy::STRICT))]
/// Load a legacy CODE DCB file from a filesystem path.
fn load_code_dcb(
    path: PathBuf,
    options: Option<&PyCodeDcbOptions>,
    policy: PyBiasReadPolicy,
) -> PyResult<PyBiasSet> {
    sidereon::load_code_dcb_with_policy(path, options.map(PyCodeDcbOptions::inner), policy.into())
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_facade_err)
}

#[pyfunction]
#[pyo3(signature = (path, options, policy=PyBiasReadPolicy::STRICT))]
/// Load a legacy CODE DCB file and return diagnostics for skipped records.
fn load_code_dcb_lossy(
    path: PathBuf,
    options: Option<&PyCodeDcbOptions>,
    policy: PyBiasReadPolicy,
) -> PyResult<PyBiasParsed> {
    sidereon::load_code_dcb_lossy_with_policy(
        path,
        options.map(PyCodeDcbOptions::inner),
        policy.into(),
    )
    .map(lossy)
    .map_err(to_bias_facade_err)
}

#[pyfunction]
/// Write a bias set as Bias-SINEX text without changing its values.
fn write_bias_sinex(bias_set: &PyBiasSet) -> PyResult<String> {
    sidereon_core::bias::write_bias_sinex(&bias_set.inner).map_err(to_bias_err)
}

#[pyfunction]
/// Write a bias set as Bias-SINEX bytes, preserving non-UTF-8 source lines.
fn write_bias_sinex_bytes(py: Python<'_>, bias_set: &PyBiasSet) -> PyResult<Py<PyBytes>> {
    let bytes =
        sidereon_core::bias::write_bias_sinex_bytes(&bias_set.inner).map_err(to_bias_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

#[pyfunction]
/// Write a bias set as CODE DCB text without changing its values.
fn write_code_dcb(bias_set: &PyBiasSet) -> PyResult<String> {
    sidereon_core::bias::write_code_dcb(&bias_set.inner).map_err(to_bias_err)
}

#[pyfunction]
/// Write a bias set as CODE DCB bytes, preserving non-UTF-8 source lines.
fn write_code_dcb_bytes(py: Python<'_>, bias_set: &PyBiasSet) -> PyResult<Py<PyBytes>> {
    let bytes = sidereon_core::bias::write_code_dcb_bytes(&bias_set.inner).map_err(to_bias_err)?;
    Ok(PyBytes::new(py, &bytes).unbind())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("BiasError", m.py().get_type::<BiasError>())?;
    m.add_class::<PyCodeDcbOptions>()?;
    m.add_class::<PyBiasRecord>()?;
    m.add_class::<PyBiasSet>()?;
    m.add_class::<PyBiasParsed>()?;
    m.add_class::<PyBiasReadPolicy>()?;
    m.add_class::<PyBiasLookup>()?;
    m.add_function(wrap_pyfunction!(parse_bias_sinex, m)?)?;
    m.add_function(wrap_pyfunction!(parse_bias_sinex_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(load_bias_sinex, m)?)?;
    m.add_function(wrap_pyfunction!(load_bias_sinex_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(parse_code_dcb, m)?)?;
    m.add_function(wrap_pyfunction!(parse_code_dcb_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(load_code_dcb, m)?)?;
    m.add_function(wrap_pyfunction!(load_code_dcb_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(write_bias_sinex, m)?)?;
    m.add_function(wrap_pyfunction!(write_bias_sinex_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(write_code_dcb, m)?)?;
    m.add_function(wrap_pyfunction!(write_code_dcb_bytes, m)?)?;
    Ok(())
}
