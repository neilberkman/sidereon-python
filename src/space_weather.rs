//! Space-weather table binding.
//!
//! The parser and lookup policy live in `sidereon-core`; this module only
//! accepts Python bytes or paths, delegates, and wraps the returned table.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyModule};

use sidereon_core::astro::forces::SpaceWeatherSource;
use sidereon_core::astro::space_weather::{
    encode_csv as core_encode_csv, encode_txt as core_encode_txt, parse as core_parse,
    ObservationClass, SpaceWeatherCoverage, SpaceWeatherPolicy, SpaceWeatherSample,
    SpaceWeatherTable,
};

use crate::forces::PySpaceWeather;
use crate::SpaceWeatherError;

fn to_space_weather_err<E: std::fmt::Display>(err: E) -> PyErr {
    SpaceWeatherError::new_err(err.to_string())
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
        ObservationClass::DailyPredicted => "daily_predicted",
        ObservationClass::MonthlyPredicted => "monthly_predicted",
    }
}

#[pyclass(module = "sidereon._sidereon", name = "ObservationClass", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyObservationClass {
    OBSERVED,
    INTERPOLATED,
    DAILY_PREDICTED,
    MONTHLY_PREDICTED,
}

impl From<ObservationClass> for PyObservationClass {
    fn from(value: ObservationClass) -> Self {
        match value {
            ObservationClass::Observed => Self::OBSERVED,
            ObservationClass::Interpolated => Self::INTERPOLATED,
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
            Self::DAILY_PREDICTED => "daily_predicted",
            Self::MONTHLY_PREDICTED => "monthly_predicted",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::OBSERVED => "ObservationClass.OBSERVED",
            Self::INTERPOLATED => "ObservationClass.INTERPOLATED",
            Self::DAILY_PREDICTED => "ObservationClass.DAILY_PREDICTED",
            Self::MONTHLY_PREDICTED => "ObservationClass.MONTHLY_PREDICTED",
        }
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

#[pyclass(module = "sidereon._sidereon", name = "SpaceWeatherTable")]
#[derive(Clone)]
pub struct PySpaceWeatherTable {
    inner: Arc<SpaceWeatherTable>,
}

impl PySpaceWeatherTable {
    pub(crate) fn source(&self) -> SpaceWeatherSource {
        SpaceWeatherSource::Table(Arc::clone(&self.inner))
    }
}

#[pymethods]
impl PySpaceWeatherTable {
    fn space_weather_at(&self, epoch_j2000_s: f64) -> PyResult<PySpaceWeather> {
        self.inner
            .space_weather_at(epoch_j2000_s)
            .map(Into::into)
            .map_err(to_space_weather_err)
    }

    fn sample_at(&self, epoch_j2000_s: f64) -> PyResult<PySpaceWeatherSample> {
        self.inner
            .sample_at(epoch_j2000_s)
            .map(|inner| PySpaceWeatherSample { inner })
            .map_err(to_space_weather_err)
    }

    #[pyo3(signature = (
        epoch_j2000_s,
        *,
        allow_interpolated=true,
        allow_daily_predicted=true,
        allow_monthly_predicted=true,
        require_geomagnetic=false,
    ))]
    fn sample_at_with_policy(
        &self,
        epoch_j2000_s: f64,
        allow_interpolated: bool,
        allow_daily_predicted: bool,
        allow_monthly_predicted: bool,
        require_geomagnetic: bool,
    ) -> PyResult<PySpaceWeatherSample> {
        let policy = SpaceWeatherPolicy {
            allow_interpolated,
            allow_daily_predicted,
            allow_monthly_predicted,
            require_geomagnetic,
        };
        self.inner
            .sample_at_with_policy(epoch_j2000_s, policy)
            .map(|inner| PySpaceWeatherSample { inner })
            .map_err(to_space_weather_err)
    }

    fn ap_array_at(&self, epoch_j2000_s: f64) -> PyResult<Vec<f64>> {
        self.inner
            .ap_array_at(epoch_j2000_s)
            .map(|values| values.to_vec())
            .map_err(to_space_weather_err)
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
fn load_space_weather(source: &Bound<'_, PyAny>) -> PyResult<PySpaceWeatherTable> {
    let bytes = bytes_from_source(source, "load_space_weather")?;
    let parsed = core_parse(&bytes).map_err(to_space_weather_err)?;
    Ok(PySpaceWeatherTable {
        inner: Arc::new(parsed.value),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObservationClass>()?;
    m.add_class::<PySpaceWeatherCoverage>()?;
    m.add_class::<PySpaceWeatherSample>()?;
    m.add_class::<PySpaceWeatherTable>()?;
    m.add_function(wrap_pyfunction!(load_space_weather, m)?)?;
    Ok(())
}
