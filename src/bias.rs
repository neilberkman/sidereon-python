//! GNSS bias product binding.

use std::path::PathBuf;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon::bias::{
    bias_epoch_instant, BiasEpoch, BiasKind, BiasRecord, BiasSet, BiasTarget, CodeDcbOptions,
    Diagnostics,
};
use sidereon_core::astro::time::model::Instant;
use sidereon_core::GnssSatelliteId;

use crate::frames::PyTimeScale;
use crate::marshal::PyGnssSystem;

fn to_bias_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    GnssSatelliteId::from_str(token)
        .map_err(|_| PyValueError::new_err(format!("invalid satellite token: {token}")))
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
        Self {
            inner: CodeDcbOptions {
                pair: (obs1, obs2),
                year,
                month,
                time_scale: time_scale.into(),
                receiver_system: receiver_system.map(Into::into),
            },
        }
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

    #[getter]
    fn is_phase(&self) -> bool {
        self.inner.is_phase
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

    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale.into()
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
    ) -> PyResult<Option<f64>> {
        let scale = time_scale.unwrap_or_else(|| self.inner.time_scale.into());
        Ok(self.inner.code_osb_seconds(
            parse_sat(satellite)?,
            obs,
            epoch(year, day_of_year, second_of_day, scale)?,
        ))
    }

    #[pyo3(signature = (satellite, obs, year, day_of_year, second_of_day, time_scale=None))]
    fn phase_osb_cycles(
        &self,
        satellite: &str,
        obs: &str,
        year: i32,
        day_of_year: u16,
        second_of_day: u32,
        time_scale: Option<PyTimeScale>,
    ) -> PyResult<Option<f64>> {
        let scale = time_scale.unwrap_or_else(|| self.inner.time_scale.into());
        Ok(self.inner.phase_osb_cycles(
            parse_sat(satellite)?,
            obs,
            epoch(year, day_of_year, second_of_day, scale)?,
        ))
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
    ) -> PyResult<Option<f64>> {
        let scale = time_scale.unwrap_or_else(|| self.inner.time_scale.into());
        Ok(self.inner.code_dsb_seconds(
            parse_sat(satellite)?,
            obs1,
            obs2,
            epoch(year, day_of_year, second_of_day, scale)?,
        ))
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
    ) -> PyResult<Option<f64>> {
        let scale = time_scale.unwrap_or_else(|| self.inner.time_scale.into());
        Ok(self.inner.code_bias_model_m(
            parse_sat(satellite)?,
            (used_obs1, used_obs2),
            (freq1_hz, freq2_hz),
            glonass_channel,
            (clock_ref_obs1, clock_ref_obs2),
            epoch(year, day_of_year, second_of_day, scale)?,
        ))
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
    #[getter]
    fn value(&self) -> PyBiasSet {
        PyBiasSet {
            inner: self.value.clone(),
        }
    }

    #[getter]
    fn skip_count(&self) -> usize {
        self.diagnostics.skips.len()
    }

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

#[pyfunction]
/// Parse Bias-SINEX bytes into a bias set.
///
/// Raises `ValueError` when the input cannot be parsed strictly.
fn parse_bias_sinex(bytes: Vec<u8>) -> PyResult<PyBiasSet> {
    sidereon::parse_bias_sinex(&bytes)
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
/// Parse Bias-SINEX bytes and return diagnostics for skipped records.
///
/// The parsed value is available as `result.value`.
fn parse_bias_sinex_lossy(bytes: Vec<u8>) -> PyResult<PyBiasParsed> {
    let parsed = sidereon::parse_bias_sinex_lossy(&bytes).map_err(to_bias_err)?;
    let (value, diagnostics) = parsed.into_parts();
    Ok(PyBiasParsed { value, diagnostics })
}

#[pyfunction]
/// Load a Bias-SINEX file from a filesystem path.
fn load_bias_sinex(path: PathBuf) -> PyResult<PyBiasSet> {
    sidereon::load_bias_sinex(path)
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
/// Load a Bias-SINEX file and return diagnostics for skipped records.
fn load_bias_sinex_lossy(path: PathBuf) -> PyResult<PyBiasParsed> {
    let parsed = sidereon::load_bias_sinex_lossy(path).map_err(to_bias_err)?;
    let (value, diagnostics) = parsed.into_parts();
    Ok(PyBiasParsed { value, diagnostics })
}

#[pyfunction]
/// Parse legacy CODE DCB bytes into a bias set.
fn parse_code_dcb(bytes: Vec<u8>, options: Option<&PyCodeDcbOptions>) -> PyResult<PyBiasSet> {
    sidereon::parse_code_dcb(&bytes, options.map(PyCodeDcbOptions::inner))
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
/// Parse legacy CODE DCB bytes and return diagnostics for skipped records.
fn parse_code_dcb_lossy(
    bytes: Vec<u8>,
    options: Option<&PyCodeDcbOptions>,
) -> PyResult<PyBiasParsed> {
    let parsed = sidereon::parse_code_dcb_lossy(&bytes, options.map(PyCodeDcbOptions::inner))
        .map_err(to_bias_err)?;
    let (value, diagnostics) = parsed.into_parts();
    Ok(PyBiasParsed { value, diagnostics })
}

#[pyfunction]
/// Load a legacy CODE DCB file from a filesystem path.
fn load_code_dcb(path: PathBuf, options: Option<&PyCodeDcbOptions>) -> PyResult<PyBiasSet> {
    sidereon::load_code_dcb(path, options.map(PyCodeDcbOptions::inner))
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
/// Load a legacy CODE DCB file and return diagnostics for skipped records.
fn load_code_dcb_lossy(
    path: PathBuf,
    options: Option<&PyCodeDcbOptions>,
) -> PyResult<PyBiasParsed> {
    let parsed = sidereon::load_code_dcb_lossy(path, options.map(PyCodeDcbOptions::inner))
        .map_err(to_bias_err)?;
    let (value, diagnostics) = parsed.into_parts();
    Ok(PyBiasParsed { value, diagnostics })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCodeDcbOptions>()?;
    m.add_class::<PyBiasRecord>()?;
    m.add_class::<PyBiasSet>()?;
    m.add_class::<PyBiasParsed>()?;
    m.add_function(wrap_pyfunction!(parse_bias_sinex, m)?)?;
    m.add_function(wrap_pyfunction!(parse_bias_sinex_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(load_bias_sinex, m)?)?;
    m.add_function(wrap_pyfunction!(load_bias_sinex_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(parse_code_dcb, m)?)?;
    m.add_function(wrap_pyfunction!(parse_code_dcb_lossy, m)?)?;
    m.add_function(wrap_pyfunction!(load_code_dcb, m)?)?;
    m.add_function(wrap_pyfunction!(load_code_dcb_lossy, m)?)?;
    Ok(())
}
