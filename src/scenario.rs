//! Deterministic scenario simulator binding.
//!
//! Scenarios enter as Python mappings or JSON text and deserialize directly into
//! [`sidereon_core::scenario::Scenario`]. Outputs expose the core synthetic
//! observable arrays and the ground-truth term ledger without local modeling.

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyModule};

use sidereon_core::scenario::{
    simulate_scenario as core_simulate_scenario, Scenario, SyntheticObservableArrays,
    SyntheticObservationSet, SyntheticTermArrays, SCENARIO_ENGINE_VERSION, SCENARIO_SCHEMA_VERSION,
};

use crate::marshal::rows3_to_array;
use crate::np_array;

fn scenario_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn scenario_from_py(source: &Bound<'_, PyAny>) -> PyResult<Scenario> {
    if let Ok(text) = source.extract::<String>() {
        return serde_json::from_str(&text).map_err(scenario_err);
    }
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        return serde_json::from_slice(bytes.as_bytes()).map_err(scenario_err);
    }
    if let Ok(bytes) = source.downcast::<PyByteArray>() {
        // SAFETY: the bytearray is copied by serde before Python can mutate it.
        return serde_json::from_slice(unsafe { bytes.as_bytes() }).map_err(scenario_err);
    }
    pythonize::depythonize(source).map_err(scenario_err)
}

fn usize_array<'py>(py: Python<'py>, values: &[usize]) -> Bound<'py, PyArray1<usize>> {
    PyArray1::from_vec(py, values.to_vec())
}

fn string_satellites(values: &[sidereon_core::GnssSatelliteId]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

/// Contiguous synthetic observable arrays.
#[pyclass(module = "sidereon._sidereon", name = "SyntheticObservableArrays")]
#[derive(Clone)]
pub struct PySyntheticObservableArrays {
    inner: SyntheticObservableArrays,
}

impl From<SyntheticObservableArrays> for PySyntheticObservableArrays {
    fn from(inner: SyntheticObservableArrays) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySyntheticObservableArrays {
    /// Start index of each epoch in every per-observation array.
    #[getter]
    fn epoch_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        usize_array(py, &self.inner.epoch_offsets)
    }

    /// Epoch index for each observation.
    #[getter]
    fn epoch_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        usize_array(py, &self.inner.epoch_index)
    }

    /// Satellite id for each observation.
    #[getter]
    fn satellite_id(&self) -> Vec<String> {
        string_satellites(&self.inner.satellite_id)
    }

    /// Code observable label for each observation.
    #[getter]
    fn code_observable(&self) -> Vec<String> {
        self.inner.code_observable.clone()
    }

    /// Carrier phase observable label for each observation.
    #[getter]
    fn phase_observable(&self) -> Vec<String> {
        self.inner.phase_observable.clone()
    }

    /// Doppler observable label for each observation.
    #[getter]
    fn doppler_observable(&self) -> Vec<String> {
        self.inner.doppler_observable.clone()
    }

    /// Carrier frequency in hertz for each observation.
    #[getter]
    fn carrier_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_hz)
    }

    /// Synthetic code pseudorange in metres.
    #[getter]
    fn pseudorange_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.pseudorange_m)
    }

    /// Synthetic carrier phase in cycles.
    #[getter]
    fn carrier_phase_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_cycles)
    }

    /// Synthetic Doppler shift in hertz.
    #[getter]
    fn doppler_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_hz)
    }

    /// Number of observation rows.
    fn __len__(&self) -> usize {
        self.inner.pseudorange_m.len()
    }
}

/// Per-observation ground-truth term arrays.
#[pyclass(module = "sidereon._sidereon", name = "SyntheticTermArrays")]
#[derive(Clone)]
pub struct PySyntheticTermArrays {
    inner: SyntheticTermArrays,
}

impl From<SyntheticTermArrays> for PySyntheticTermArrays {
    fn from(inner: SyntheticTermArrays) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySyntheticTermArrays {
    /// Geometric range in metres.
    #[getter]
    fn geometric_range_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.geometric_range_m)
    }

    /// Nominal ephemeris satellite-clock contribution in metres.
    #[getter]
    fn satellite_clock_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.satellite_clock_m)
    }

    /// Receiver-clock contribution in metres.
    #[getter]
    fn receiver_clock_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.receiver_clock_m)
    }

    /// Injected satellite-clock contribution in metres.
    #[getter]
    fn satellite_clock_error_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.satellite_clock_error_m)
    }

    /// Ionospheric code delay in metres.
    #[getter]
    fn ionosphere_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.ionosphere_m)
    }

    /// Tropospheric delay in metres.
    #[getter]
    fn troposphere_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.troposphere_m)
    }

    /// Thermal code noise in metres.
    #[getter]
    fn thermal_noise_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.thermal_noise_m)
    }

    /// Specular code multipath in metres.
    #[getter]
    fn multipath_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.multipath_m)
    }

    /// Quantization contribution in metres.
    #[getter]
    fn quantization_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.quantization_m)
    }

    /// Carrier geometric range contribution in cycles.
    #[getter]
    fn carrier_phase_geometric_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_geometric_cycles)
    }

    /// Carrier receiver-clock contribution in cycles.
    #[getter]
    fn carrier_phase_receiver_clock_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_receiver_clock_cycles)
    }

    /// Carrier nominal satellite-clock contribution in cycles.
    #[getter]
    fn carrier_phase_satellite_clock_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_satellite_clock_cycles)
    }

    /// Carrier injected satellite-clock contribution in cycles.
    #[getter]
    fn carrier_phase_satellite_clock_error_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_satellite_clock_error_cycles)
    }

    /// Carrier ionosphere contribution in cycles.
    #[getter]
    fn carrier_phase_ionosphere_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_ionosphere_cycles)
    }

    /// Carrier troposphere contribution in cycles.
    #[getter]
    fn carrier_phase_troposphere_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_troposphere_cycles)
    }

    /// Carrier thermal-noise contribution in cycles.
    #[getter]
    fn carrier_phase_thermal_noise_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_thermal_noise_cycles)
    }

    /// Constant carrier-phase ambiguity contribution in cycles.
    #[getter]
    fn carrier_phase_bias_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_bias_cycles)
    }

    /// Carrier quantization contribution in cycles.
    #[getter]
    fn carrier_phase_quantization_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_quantization_cycles)
    }

    /// Doppler contribution from satellite line-of-sight motion in hertz.
    #[getter]
    fn doppler_satellite_motion_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_satellite_motion_hz)
    }

    /// Doppler contribution from receiver line-of-sight motion in hertz.
    #[getter]
    fn doppler_receiver_motion_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_receiver_motion_hz)
    }

    /// Doppler contribution from nominal ephemeris satellite-clock rate.
    #[getter]
    fn doppler_satellite_clock_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_satellite_clock_hz)
    }

    /// Doppler contribution from receiver-clock rate.
    #[getter]
    fn doppler_receiver_clock_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_receiver_clock_hz)
    }

    /// Doppler contribution from injected satellite-clock rate.
    #[getter]
    fn doppler_satellite_clock_error_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_satellite_clock_error_hz)
    }

    /// Doppler thermal-noise contribution in hertz.
    #[getter]
    fn doppler_thermal_noise_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_thermal_noise_hz)
    }

    /// Doppler quantization contribution in hertz.
    #[getter]
    fn doppler_quantization_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_quantization_hz)
    }

    /// Sum the pseudorange terms for one observation.
    fn pseudorange_sum_m(&self, index: usize) -> Option<f64> {
        self.inner.pseudorange_sum_m(index)
    }

    /// Sum the carrier-phase terms for one observation.
    fn carrier_phase_sum_cycles(&self, index: usize) -> Option<f64> {
        self.inner.carrier_phase_sum_cycles(index)
    }

    /// Sum the Doppler terms for one observation.
    fn doppler_sum_hz(&self, index: usize) -> Option<f64> {
        self.inner.doppler_sum_hz(index)
    }
}

/// Complete synthetic observation output.
#[pyclass(module = "sidereon._sidereon", name = "SyntheticObservationSet")]
#[derive(Clone)]
pub struct PySyntheticObservationSet {
    inner: SyntheticObservationSet,
}

impl From<SyntheticObservationSet> for PySyntheticObservationSet {
    fn from(inner: SyntheticObservationSet) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySyntheticObservationSet {
    /// Scenario schema version used to produce this output.
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }

    /// Engine version used to produce this output.
    #[getter]
    fn engine_version(&self) -> String {
        self.inner.engine_version.clone()
    }

    /// Seed used to produce this output.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Contiguous observable arrays.
    #[getter]
    fn observations(&self) -> PySyntheticObservableArrays {
        self.inner.observations.clone().into()
    }

    /// Per-observation term decomposition.
    #[getter]
    fn truth_terms(&self) -> PySyntheticTermArrays {
        self.inner.truth_terms.clone().into()
    }

    /// Receiver epoch seconds since J2000.
    #[getter]
    fn receiver_t_rx_j2000_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.t_rx_j2000_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Receiver ECEF position rows in metres.
    #[getter]
    fn receiver_position_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.position_ecef_m)
            .collect::<Vec<_>>();
        rows3_to_array(py, &rows)
    }

    /// Receiver ECEF velocity rows in metres per second.
    #[getter]
    fn receiver_velocity_ecef_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.velocity_ecef_m_s)
            .collect::<Vec<_>>();
        rows3_to_array(py, &rows)
    }

    /// Receiver clock contribution in metres.
    #[getter]
    fn receiver_clock_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.clock_m)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Receiver clock range-rate contribution in metres per second.
    #[getter]
    fn receiver_clock_rate_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.clock_rate_m_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Number of synthetic observations.
    fn observation_count(&self) -> usize {
        self.inner.observation_count()
    }

    /// Deterministic FNV-1a fingerprint over output bits.
    fn determinism_fingerprint(&self) -> u64 {
        self.inner.determinism_fingerprint()
    }

    /// Serialize this output to deterministic JSON bytes.
    fn as_json_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = serde_json::to_vec(&self.inner).map_err(scenario_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Serialize this output to deterministic JSON text.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(scenario_err)
    }

    /// Serialize the synthetic observations to RINEX OBS text.
    fn to_rinex_string(&self) -> String {
        self.inner.to_rinex_string()
    }

    /// Build SPP observations for one epoch from the pseudorange arrays.
    fn spp_observations_for_epoch(&self, epoch_index: usize) -> Vec<(String, f64)> {
        self.inner
            .spp_observations_for_epoch(epoch_index)
            .into_iter()
            .map(|row| (row.satellite_id.to_string(), row.pseudorange_m))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.observation_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "SyntheticObservationSet(observation_count={}, seed={})",
            self.inner.observation_count(),
            self.inner.seed
        )
    }
}

/// Accepted scenario schema version.
#[pyfunction]
fn scenario_schema_version() -> u32 {
    SCENARIO_SCHEMA_VERSION
}

/// Scenario engine version string used in determinism fingerprints.
#[pyfunction]
fn scenario_engine_version() -> &'static str {
    SCENARIO_ENGINE_VERSION
}

/// Validate and canonicalize a scenario mapping or JSON document to JSON text.
#[pyfunction]
fn scenario_to_json(scenario: &Bound<'_, PyAny>) -> PyResult<String> {
    let scenario = scenario_from_py(scenario)?;
    scenario.validate().map_err(scenario_err)?;
    serde_json::to_string(&scenario).map_err(scenario_err)
}

/// Simulate a deterministic synthetic-Keplerian scenario.
#[pyfunction]
fn simulate_scenario(scenario: &Bound<'_, PyAny>) -> PyResult<PySyntheticObservationSet> {
    let scenario = scenario_from_py(scenario)?;
    core_simulate_scenario(&scenario)
        .map(Into::into)
        .map_err(scenario_err)
}

/// Simulate a scenario and return deterministic JSON bytes for its output.
#[pyfunction]
fn simulate_scenario_bytes<'py>(
    py: Python<'py>,
    scenario: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let set = simulate_scenario(scenario)?;
    set.as_json_bytes(py)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySyntheticObservableArrays>()?;
    m.add_class::<PySyntheticTermArrays>()?;
    m.add_class::<PySyntheticObservationSet>()?;
    m.add_function(wrap_pyfunction!(scenario_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(scenario_engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(scenario_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_scenario_bytes, m)?)?;
    Ok(())
}
