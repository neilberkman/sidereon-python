//! GNSS bias product binding.

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
}

#[pyclass(module = "sidereon._sidereon", name = "BiasRecord")]
#[derive(Clone)]
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
}

#[pyclass(module = "sidereon._sidereon", name = "BiasSet")]
#[derive(Clone)]
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
}

#[pyclass(module = "sidereon._sidereon", name = "BiasParsed")]
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
}

#[pyfunction]
fn parse_bias_sinex(bytes: Vec<u8>) -> PyResult<PyBiasSet> {
    sidereon::parse_bias_sinex(&bytes)
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
fn parse_bias_sinex_lossy(bytes: Vec<u8>) -> PyResult<PyBiasParsed> {
    let parsed = sidereon::parse_bias_sinex_lossy(&bytes).map_err(to_bias_err)?;
    let (value, diagnostics) = parsed.into_parts();
    Ok(PyBiasParsed { value, diagnostics })
}

#[pyfunction]
fn load_bias_sinex(path: String) -> PyResult<PyBiasSet> {
    sidereon::load_bias_sinex(path)
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
fn load_bias_sinex_lossy(path: String) -> PyResult<PyBiasParsed> {
    let parsed = sidereon::load_bias_sinex_lossy(path).map_err(to_bias_err)?;
    let (value, diagnostics) = parsed.into_parts();
    Ok(PyBiasParsed { value, diagnostics })
}

#[pyfunction]
fn parse_code_dcb(bytes: Vec<u8>, options: Option<&PyCodeDcbOptions>) -> PyResult<PyBiasSet> {
    sidereon::parse_code_dcb(&bytes, options.map(PyCodeDcbOptions::inner))
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
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
fn load_code_dcb(path: String, options: Option<&PyCodeDcbOptions>) -> PyResult<PyBiasSet> {
    sidereon::load_code_dcb(path, options.map(PyCodeDcbOptions::inner))
        .map(|inner| PyBiasSet { inner })
        .map_err(to_bias_err)
}

#[pyfunction]
fn load_code_dcb_lossy(path: String, options: Option<&PyCodeDcbOptions>) -> PyResult<PyBiasParsed> {
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
