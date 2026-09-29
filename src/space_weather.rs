//! Space-weather table binding.
//!
//! The parser and lookup policy live in `sidereon-core`; this module only
//! accepts Python bytes or paths, delegates, and wraps the returned table.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyModule};

use sidereon_core::astro::forces::SpaceWeatherSource;
use sidereon_core::astro::space_weather::{
    encode_csv as core_encode_csv, encode_txt as core_encode_txt, parse as core_parse,
    ApHistorySample, ObservationClass, SpaceWeatherCoverage,
    SpaceWeatherError as CoreSpaceWeatherError, SpaceWeatherPolicy, SpaceWeatherSample,
    SpaceWeatherTable,
};
use sidereon_core::nmea::Diagnostics;

use crate::forces::PySpaceWeather;
use crate::SpaceWeatherError;

fn to_space_weather_err(py: Python<'_>, err: CoreSpaceWeatherError) -> PyErr {
    let py_err = SpaceWeatherError::new_err(err.to_string());
    let detail = match Py::new(py, PySpaceWeatherErrorDetail { inner: err }) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    if let Err(error) = py_err.value(py).setattr("detail", detail) {
        return error;
    }
    py_err
}

fn bytes_from_source(source: &Bound<'_, PyAny>, function_name: &str) -> PyResult<Vec<u8>> {
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        return Ok(bytes.as_bytes().to_vec());
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the bytearray is copied into owned Rust bytes immediately, and
        // no Python code runs before the copy completes.
        return Ok(unsafe { buf.as_bytes() }.to_vec());
    }
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err(format!(
            "{function_name} expects bytes, bytearray, or a path (str/os.PathLike)"
        ))
    })?;
    std::fs::read(&path).map_err(Into::into)
}

fn class_label(class: ObservationClass) -> &'static str {
    match class {
        ObservationClass::Observed => "observed",
        ObservationClass::Interpolated => "interpolated",
        ObservationClass::NotObserved => "not_observed",
        ObservationClass::DailyPredicted => "daily_predicted",
        ObservationClass::MonthlyPredicted => "monthly_predicted",
    }
}

/// Build a core lookup policy from the keyword arguments every policy-taking
/// method shares. The defaults mirror the core default policy.
fn policy(
    allow_interpolated: bool,
    allow_not_observed: bool,
    allow_daily_predicted: bool,
    allow_monthly_predicted: bool,
    require_geomagnetic: bool,
) -> SpaceWeatherPolicy {
    SpaceWeatherPolicy {
        allow_interpolated,
        allow_not_observed,
        allow_daily_predicted,
        allow_monthly_predicted,
        require_geomagnetic,
    }
}

#[pyclass(module = "sidereon._sidereon", name = "ObservationClass", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyObservationClass {
    OBSERVED,
    INTERPOLATED,
    /// A row whose F10.7 flux qualifier states the day had no observation
    /// (`Q` 3), refused by the default policy.
    NOT_OBSERVED,
    DAILY_PREDICTED,
    MONTHLY_PREDICTED,
}

impl From<ObservationClass> for PyObservationClass {
    fn from(value: ObservationClass) -> Self {
        match value {
            ObservationClass::Observed => Self::OBSERVED,
            ObservationClass::Interpolated => Self::INTERPOLATED,
            ObservationClass::NotObserved => Self::NOT_OBSERVED,
            ObservationClass::DailyPredicted => Self::DAILY_PREDICTED,
            ObservationClass::MonthlyPredicted => Self::MONTHLY_PREDICTED,
        }
    }
}

#[pymethods]
impl PyObservationClass {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::OBSERVED => "observed",
            Self::INTERPOLATED => "interpolated",
            Self::NOT_OBSERVED => "not_observed",
            Self::DAILY_PREDICTED => "daily_predicted",
            Self::MONTHLY_PREDICTED => "monthly_predicted",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::OBSERVED => "ObservationClass.OBSERVED",
            Self::INTERPOLATED => "ObservationClass.INTERPOLATED",
            Self::NOT_OBSERVED => "ObservationClass.NOT_OBSERVED",
            Self::DAILY_PREDICTED => "ObservationClass.DAILY_PREDICTED",
            Self::MONTHLY_PREDICTED => "ObservationClass.MONTHLY_PREDICTED",
        }
    }
}

/// The exact core variant and fields behind a space-weather refusal.
#[pyclass(module = "sidereon._sidereon", name = "SpaceWeatherErrorDetail")]
#[derive(Clone)]
pub struct PySpaceWeatherErrorDetail {
    inner: CoreSpaceWeatherError,
}

#[pymethods]
impl PySpaceWeatherErrorDetail {
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            CoreSpaceWeatherError::UnrecognizedFormat => "UnrecognizedFormat",
            CoreSpaceWeatherError::Malformed { .. } => "Malformed",
            CoreSpaceWeatherError::NotText => "NotText",
            CoreSpaceWeatherError::BeforeCoverage { .. } => "BeforeCoverage",
            CoreSpaceWeatherError::AfterCoverage { .. } => "AfterCoverage",
            CoreSpaceWeatherError::MissingData { .. } => "MissingData",
            CoreSpaceWeatherError::RejectedByPolicy { .. } => "RejectedByPolicy",
            CoreSpaceWeatherError::InvalidEpoch { .. } => "InvalidEpoch",
        }
    }

    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let details = PyDict::new(py);
        match &self.inner {
            CoreSpaceWeatherError::UnrecognizedFormat | CoreSpaceWeatherError::NotText => {}
            CoreSpaceWeatherError::Malformed { line, reason } => {
                details.set_item("line", *line)?;
                details.set_item("reason", reason)?;
            }
            CoreSpaceWeatherError::BeforeCoverage {
                requested_j2000_s,
                first_j2000_s,
            } => {
                details.set_item("requested_j2000_s", *requested_j2000_s)?;
                details.set_item("first_j2000_s", *first_j2000_s)?;
            }
            CoreSpaceWeatherError::AfterCoverage {
                requested_j2000_s,
                end_j2000_s,
            } => {
                details.set_item("requested_j2000_s", *requested_j2000_s)?;
                details.set_item("end_j2000_s", *end_j2000_s)?;
            }
            CoreSpaceWeatherError::MissingData {
                year,
                month,
                day,
                field,
            } => {
                details.set_item("year", *year)?;
                details.set_item("month", *month)?;
                details.set_item("day", *day)?;
                details.set_item("field", *field)?;
            }
            CoreSpaceWeatherError::RejectedByPolicy {
                class,
                year,
                month,
                day,
            } => {
                details.set_item("class", PyObservationClass::from(*class))?;
                details.set_item("year", *year)?;
                details.set_item("month", *month)?;
                details.set_item("day", *day)?;
            }
            CoreSpaceWeatherError::InvalidEpoch { epoch_j2000_s_bits } => {
                details.set_item("epoch_j2000_s_bits", *epoch_j2000_s_bits)?;
            }
        }
        Ok(details)
    }

    fn __repr__(&self) -> String {
        format!("SpaceWeatherErrorDetail(kind={:?})", self.kind())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SpaceWeatherCoverage")]
#[derive(Clone, Copy)]
pub struct PySpaceWeatherCoverage {
    inner: SpaceWeatherCoverage,
}

#[pymethods]
impl PySpaceWeatherCoverage {
    #[getter]
    fn first_j2000_s(&self) -> f64 {
        self.inner.first_j2000_s
    }

    #[getter]
    fn last_observed_j2000_s(&self) -> Option<f64> {
        self.inner.last_observed_j2000_s
    }

    #[getter]
    fn last_daily_predicted_j2000_s(&self) -> Option<f64> {
        self.inner.last_daily_predicted_j2000_s
    }

    #[getter]
    fn end_j2000_s(&self) -> f64 {
        self.inner.end_j2000_s
    }

    fn __repr__(&self) -> String {
        format!(
            "SpaceWeatherCoverage(first_j2000_s={}, end_j2000_s={})",
            self.inner.first_j2000_s, self.inner.end_j2000_s
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SpaceWeatherSample")]
#[derive(Clone, Copy)]
pub struct PySpaceWeatherSample {
    inner: SpaceWeatherSample,
}

#[pymethods]
impl PySpaceWeatherSample {
    #[getter]
    fn space_weather(&self) -> PySpaceWeather {
        self.inner.space_weather.into()
    }

    #[getter]
    fn class_(&self) -> &'static str {
        class_label(self.inner.class)
    }

    #[getter]
    fn class_kind(&self) -> PyObservationClass {
        self.inner.class.into()
    }

    #[getter]
    fn ap_defaulted(&self) -> bool {
        self.inner.ap_defaulted
    }

    fn __repr__(&self) -> String {
        format!(
            "SpaceWeatherSample(class_={:?}, ap_defaulted={})",
            self.class_(),
            self.inner.ap_defaulted
        )
    }
}

/// The NRLMSISE-00 Ap history at an epoch, with the least-trusted row class
/// consulted and the substitutions the policy allowed.
#[pyclass(module = "sidereon._sidereon", name = "ApHistorySample")]
#[derive(Clone, Copy)]
pub struct PyApHistorySample {
    inner: ApHistorySample,
}

#[pymethods]
impl PyApHistorySample {
    /// The seven-element NRLMSISE-00 Ap history array.
    #[getter]
    fn ap(&self) -> Vec<f64> {
        self.inner.ap.to_vec()
    }

    /// Least-trusted class label among the rows consulted.
    #[getter]
    fn class_(&self) -> &'static str {
        class_label(self.inner.class)
    }

    /// Least-trusted class among the rows consulted.
    #[getter]
    fn class_kind(&self) -> PyObservationClass {
        self.inner.class.into()
    }

    /// True when a row with a blank `AP_AVG` contributed the quiet default
    /// Ap.
    #[getter]
    fn ap_defaulted(&self) -> bool {
        self.inner.ap_defaulted
    }

    /// Number of three-hour slots whose blank `ap` bin was filled with the
    /// row's daily Ap.
    #[getter]
    fn bins_from_daily_ap(&self) -> u8 {
        self.inner.bins_from_daily_ap
    }

    fn __repr__(&self) -> String {
        format!(
            "ApHistorySample(class_={:?}, ap_defaulted={}, bins_from_daily_ap={})",
            self.class_(),
            self.inner.ap_defaulted,
            self.inner.bins_from_daily_ap
        )
    }
}

/// A parsed CelesTrak space-weather table.
///
/// Drag reads the table under `drag_policy`, the core default policy unless
/// `with_drag_policy` set another: every Ap it uses comes from the file, and a
/// day without an observation is refused.
#[pyclass(module = "sidereon._sidereon", name = "SpaceWeatherTable")]
#[derive(Clone)]
pub struct PySpaceWeatherTable {
    inner: Arc<SpaceWeatherTable>,
    drag_policy: SpaceWeatherPolicy,
    diagnostics: Diagnostics,
}

impl PySpaceWeatherTable {
    pub(crate) fn source(&self) -> SpaceWeatherSource {
        if self.drag_policy == SpaceWeatherPolicy::default() {
            SpaceWeatherSource::Table(Arc::clone(&self.inner))
        } else {
            SpaceWeatherSource::TableWithPolicy(Arc::clone(&self.inner), self.drag_policy)
        }
    }
}

#[pymethods]
impl PySpaceWeatherTable {
    fn space_weather_at(&self, py: Python<'_>, epoch_j2000_s: f64) -> PyResult<PySpaceWeather> {
        self.inner
            .space_weather_at(epoch_j2000_s)
            .map(Into::into)
            .map_err(|err| to_space_weather_err(py, err))
    }

    fn sample_at(&self, py: Python<'_>, epoch_j2000_s: f64) -> PyResult<PySpaceWeatherSample> {
        self.inner
            .sample_at(epoch_j2000_s)
            .map(|inner| PySpaceWeatherSample { inner })
            .map_err(|err| to_space_weather_err(py, err))
    }

    #[pyo3(signature = (
        epoch_j2000_s,
        *,
        allow_interpolated=true,
        allow_daily_predicted=true,
        allow_monthly_predicted=true,
        require_geomagnetic=true,
        allow_not_observed=false,
    ))]
    // The internal PyO3 interpreter handle adds one argument beyond the public options.
    #[allow(clippy::too_many_arguments)]
    fn sample_at_with_policy(
        &self,
        py: Python<'_>,
        epoch_j2000_s: f64,
        allow_interpolated: bool,
        allow_daily_predicted: bool,
        allow_monthly_predicted: bool,
        require_geomagnetic: bool,
        allow_not_observed: bool,
    ) -> PyResult<PySpaceWeatherSample> {
        let policy = policy(
            allow_interpolated,
            allow_not_observed,
            allow_daily_predicted,
            allow_monthly_predicted,
            require_geomagnetic,
        );
        self.inner
            .sample_at_with_policy(epoch_j2000_s, policy)
            .map(|inner| PySpaceWeatherSample { inner })
            .map_err(|err| to_space_weather_err(py, err))
    }

    /// The NRLMSISE-00 Ap history at an epoch under an explicit policy, with
    /// the least-trusted row class consulted and every substitution the
    /// policy allowed.
    #[pyo3(signature = (
        epoch_j2000_s,
        *,
        allow_interpolated=true,
        allow_daily_predicted=true,
        allow_monthly_predicted=true,
        require_geomagnetic=true,
        allow_not_observed=false,
    ))]
    // The internal PyO3 interpreter handle adds one argument beyond the public options.
    #[allow(clippy::too_many_arguments)]
    fn ap_history_at_with_policy(
        &self,
        py: Python<'_>,
        epoch_j2000_s: f64,
        allow_interpolated: bool,
        allow_daily_predicted: bool,
        allow_monthly_predicted: bool,
        require_geomagnetic: bool,
        allow_not_observed: bool,
    ) -> PyResult<PyApHistorySample> {
        let policy = policy(
            allow_interpolated,
            allow_not_observed,
            allow_daily_predicted,
            allow_monthly_predicted,
            require_geomagnetic,
        );
        self.inner
            .ap_history_at_with_policy(epoch_j2000_s, policy)
            .map(|inner| PyApHistorySample { inner })
            .map_err(|err| to_space_weather_err(py, err))
    }

    /// A copy of this table that drag reads under the given policy. The
    /// `lenient` core policy is `allow_not_observed=True,
    /// require_geomagnetic=False` with every row class allowed; each
    /// substitution it makes is then taken without a report.
    #[pyo3(signature = (
        *,
        allow_interpolated=true,
        allow_daily_predicted=true,
        allow_monthly_predicted=true,
        require_geomagnetic=true,
        allow_not_observed=false,
    ))]
    fn with_drag_policy(
        &self,
        allow_interpolated: bool,
        allow_daily_predicted: bool,
        allow_monthly_predicted: bool,
        require_geomagnetic: bool,
        allow_not_observed: bool,
    ) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            drag_policy: policy(
                allow_interpolated,
                allow_not_observed,
                allow_daily_predicted,
                allow_monthly_predicted,
                require_geomagnetic,
            ),
            diagnostics: self.diagnostics.clone(),
        }
    }

    /// Non-fatal diagnostics collected while reading the space-weather table.
    #[getter]
    fn diagnostics(&self) -> crate::format_diagnostics::PyFormatDiagnostics {
        crate::format_diagnostics::PyFormatDiagnostics::from_inner(self.diagnostics.clone())
    }

    /// The policy drag reads this table under, as a dict of the policy's
    /// keyword arguments.
    #[getter]
    fn drag_policy(&self) -> std::collections::BTreeMap<&'static str, bool> {
        std::collections::BTreeMap::from([
            ("allow_interpolated", self.drag_policy.allow_interpolated),
            ("allow_not_observed", self.drag_policy.allow_not_observed),
            (
                "allow_daily_predicted",
                self.drag_policy.allow_daily_predicted,
            ),
            (
                "allow_monthly_predicted",
                self.drag_policy.allow_monthly_predicted,
            ),
            ("require_geomagnetic", self.drag_policy.require_geomagnetic),
        ])
    }

    fn ap_array_at(&self, py: Python<'_>, epoch_j2000_s: f64) -> PyResult<Vec<f64>> {
        self.inner
            .ap_array_at(epoch_j2000_s)
            .map(|values| values.to_vec())
            .map_err(|err| to_space_weather_err(py, err))
    }

    fn coverage(&self) -> PySpaceWeatherCoverage {
        PySpaceWeatherCoverage {
            inner: self.inner.coverage(),
        }
    }

    fn to_csv_text(&self) -> String {
        core_encode_csv(&self.inner)
    }

    fn to_txt_text(&self) -> String {
        core_encode_txt(&self.inner)
    }

    fn __repr__(&self) -> String {
        let coverage = self.inner.coverage();
        format!(
            "SpaceWeatherTable(first_j2000_s={}, end_j2000_s={})",
            coverage.first_j2000_s, coverage.end_j2000_s
        )
    }
}

#[pyfunction]
fn load_space_weather(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<PySpaceWeatherTable> {
    let bytes = bytes_from_source(source, "load_space_weather")?;
    let parsed = core_parse(&bytes).map_err(|err| to_space_weather_err(py, err))?;
    Ok(PySpaceWeatherTable {
        inner: Arc::new(parsed.value),
        drag_policy: SpaceWeatherPolicy::default(),
        diagnostics: parsed.diagnostics,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    py.get_type::<SpaceWeatherError>()
        .setattr("detail", py.None())?;
    m.add_class::<PySpaceWeatherErrorDetail>()?;
    m.add_class::<PyObservationClass>()?;
    m.add_class::<PySpaceWeatherCoverage>()?;
    m.add_class::<PySpaceWeatherSample>()?;
    m.add_class::<PyApHistorySample>()?;
    m.add_class::<PySpaceWeatherTable>()?;
    m.add_function(wrap_pyfunction!(load_space_weather, m)?)?;
    Ok(())
}
