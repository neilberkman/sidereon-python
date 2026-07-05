//! Closed-form navigation-signal analysis binding.
//!
//! This module exposes spectrum-domain metrics from
//! [`sidereon_core::signal::analysis`]. It only constructs modulation values,
//! forwards scalar or vector requests, and packages the core results.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::signal::analysis::{
    dll_lower_bound as core_dll_lower_bound,
    dll_thermal_noise_jitter as core_dll_thermal_noise_jitter,
    effective_cn0_degradation as core_effective_cn0_degradation,
    fraction_power_in_band as core_fraction_power_in_band,
    multipath_error_envelope as core_multipath_error_envelope, power_in_band as core_power_in_band,
    rms_bandwidth_hz as core_rms_bandwidth_hz,
    spectral_separation_coefficient_db_hz as core_spectral_separation_coefficient_db_hz,
    spectral_separation_coefficient_hz as core_spectral_separation_coefficient_hz,
    white_noise_spectral_separation_hz as core_white_noise_spectral_separation_hz, Cn0Degradation,
    DllJitter, DllProcessing, DllTrackingOptions, InterferenceTerm, MultipathEnvelopePoint,
    MultipathOptions, SignalModulation, BETZ_L1_RECEIVER_BANDWIDTH_HZ, REFERENCE_CHIP_RATE_HZ,
};

use crate::np_array;

fn analysis_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Navigation modulation model used by spectrum-domain metrics.
#[pyclass(module = "sidereon._sidereon", name = "SignalModulation")]
#[derive(Clone)]
pub struct PySignalModulation {
    inner: SignalModulation,
}

impl PySignalModulation {
    fn inner(&self) -> &SignalModulation {
        &self.inner
    }
}

#[pymethods]
impl PySignalModulation {
    /// Construct BPSK(n), with code rate `n * 1.023 MHz`.
    #[staticmethod]
    fn bpsk(order: f64) -> PyResult<Self> {
        SignalModulation::bpsk(order)
            .map(|inner| Self { inner })
            .map_err(analysis_err)
    }

    /// Construct BPSK(1), the GPS C/A spectral shape for long random codes.
    #[staticmethod]
    fn bpsk1() -> Self {
        Self {
            inner: SignalModulation::bpsk1(),
        }
    }

    /// Construct sine-phased BOC(m,n).
    #[staticmethod]
    fn boc_sine(m: f64, n: f64) -> PyResult<Self> {
        SignalModulation::boc_sine(m, n)
            .map(|inner| Self { inner })
            .map_err(analysis_err)
    }

    /// Construct cosine-phased BOC(m,n).
    #[staticmethod]
    fn boc_cosine(m: f64, n: f64) -> PyResult<Self> {
        SignalModulation::boc_cosine(m, n)
            .map(|inner| Self { inner })
            .map_err(analysis_err)
    }

    /// Construct the normalized MBOC(6,1,1/11) PSD.
    #[staticmethod]
    fn mboc_6_1_1_over_11() -> Self {
        Self {
            inner: SignalModulation::mboc_6_1_1_over_11(),
        }
    }

    /// Construct the GPS L1C pilot TMBOC(6,1,4/33) PSD.
    #[staticmethod]
    fn tmboc_6_1_4_over_33() -> Self {
        Self {
            inner: SignalModulation::tmboc_6_1_4_over_33(),
        }
    }

    /// Return the normalized PSD at offset frequency `offset_hz`.
    fn psd_hz(&self, offset_hz: f64) -> PyResult<f64> {
        self.inner.psd_hz(offset_hz).map_err(analysis_err)
    }

    /// Return the code rate in hertz when the modulation has one rate.
    fn code_rate_hz(&self) -> PyResult<f64> {
        self.inner.code_rate_hz().map_err(analysis_err)
    }

    /// Short stable label for the modulation.
    #[getter]
    fn label(&self) -> &'static str {
        self.inner.label()
    }

    fn __repr__(&self) -> String {
        format!("SignalModulation(label={:?})", self.inner.label())
    }
}

/// Interfering signal and received power used in C/N0 degradation metrics.
#[pyclass(module = "sidereon._sidereon", name = "InterferenceTerm")]
#[derive(Clone)]
pub struct PyInterferenceTerm {
    inner: InterferenceTerm,
}

impl PyInterferenceTerm {
    fn inner(&self) -> InterferenceTerm {
        self.inner.clone()
    }
}

#[pymethods]
impl PyInterferenceTerm {
    /// Build an interference term from a modulation and power ratio.
    #[new]
    fn new(modulation: &PySignalModulation, power_ratio_to_carrier: f64) -> Self {
        Self {
            inner: InterferenceTerm::new(modulation.inner.clone(), power_ratio_to_carrier),
        }
    }

    /// Interference received power divided by desired-signal received power.
    #[getter]
    fn power_ratio_to_carrier(&self) -> f64 {
        self.inner.power_ratio_to_carrier
    }
}

/// Effective C/N0 result with the corresponding degradation.
#[pyclass(module = "sidereon._sidereon", name = "Cn0Degradation")]
#[derive(Clone, Copy)]
pub struct PyCn0Degradation {
    inner: Cn0Degradation,
}

impl From<Cn0Degradation> for PyCn0Degradation {
    fn from(inner: Cn0Degradation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCn0Degradation {
    /// Effective carrier-to-noise-density ratio in hertz.
    #[getter]
    fn effective_cn0_hz(&self) -> f64 {
        self.inner.effective_cn0_hz
    }

    /// Effective carrier-to-noise-density ratio in decibel-hertz.
    #[getter]
    fn effective_cn0_db_hz(&self) -> f64 {
        self.inner.effective_cn0_db_hz
    }

    /// Loss from the input C/N0 to the effective C/N0, in decibels.
    #[getter]
    fn degradation_db(&self) -> f64 {
        self.inner.degradation_db
    }
}

/// Processing mode for early-late DLL thermal-noise jitter.
#[pyclass(module = "sidereon._sidereon", name = "DllProcessing", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyDllProcessing {
    /// Coherent early-minus-late processing.
    COHERENT,
    /// Non-coherent early-minus-late power processing.
    NON_COHERENT,
}

impl From<PyDllProcessing> for DllProcessing {
    fn from(value: PyDllProcessing) -> Self {
        match value {
            PyDllProcessing::COHERENT => Self::Coherent,
            PyDllProcessing::NON_COHERENT => Self::NonCoherent,
        }
    }
}

#[pymethods]
impl PyDllProcessing {
    /// Stable lowercase processing label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::COHERENT => "coherent",
            Self::NON_COHERENT => "non_coherent",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::COHERENT => "DllProcessing.COHERENT",
            Self::NON_COHERENT => "DllProcessing.NON_COHERENT",
        }
    }
}

/// Inputs for code-tracking thermal-noise figures.
#[pyclass(module = "sidereon._sidereon", name = "DllTrackingOptions")]
#[derive(Clone, Copy)]
pub struct PyDllTrackingOptions {
    inner: DllTrackingOptions,
}

impl PyDllTrackingOptions {
    fn inner(&self) -> DllTrackingOptions {
        self.inner
    }
}

#[pymethods]
impl PyDllTrackingOptions {
    /// Build DLL tracking options.
    #[new]
    fn new(
        cn0_db_hz: f64,
        loop_bandwidth_hz: f64,
        integration_time_s: f64,
        correlator_spacing_chips: f64,
        receiver_bandwidth_hz: f64,
    ) -> Self {
        Self {
            inner: DllTrackingOptions {
                cn0_db_hz,
                loop_bandwidth_hz,
                integration_time_s,
                correlator_spacing_chips,
                receiver_bandwidth_hz,
            },
        }
    }

    /// Carrier-to-noise-density ratio in decibel-hertz.
    #[getter]
    fn cn0_db_hz(&self) -> f64 {
        self.inner.cn0_db_hz
    }

    /// One-sided DLL loop bandwidth in hertz.
    #[getter]
    fn loop_bandwidth_hz(&self) -> f64 {
        self.inner.loop_bandwidth_hz
    }

    /// Predetection coherent integration time in seconds.
    #[getter]
    fn integration_time_s(&self) -> f64 {
        self.inner.integration_time_s
    }

    /// Early-late correlator spacing in code chips.
    #[getter]
    fn correlator_spacing_chips(&self) -> f64 {
        self.inner.correlator_spacing_chips
    }

    /// Two-sided receiver bandwidth in hertz.
    #[getter]
    fn receiver_bandwidth_hz(&self) -> f64 {
        self.inner.receiver_bandwidth_hz
    }
}

/// Code-tracking thermal-noise result.
#[pyclass(module = "sidereon._sidereon", name = "DllJitter")]
#[derive(Clone, Copy)]
pub struct PyDllJitter {
    inner: DllJitter,
}

impl From<DllJitter> for PyDllJitter {
    fn from(inner: DllJitter) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyDllJitter {
    /// One-sigma delay jitter in seconds.
    #[getter]
    fn seconds(&self) -> f64 {
        self.inner.seconds
    }

    /// One-sigma delay jitter in code chips.
    #[getter]
    fn chips(&self) -> f64 {
        self.inner.chips
    }

    /// One-sigma range jitter in metres.
    #[getter]
    fn meters(&self) -> f64 {
        self.inner.meters
    }

    /// Non-coherent squaring-loss multiplier.
    #[getter]
    fn squaring_loss(&self) -> f64 {
        self.inner.squaring_loss
    }
}

/// Inputs for one-path specular multipath envelope metrics.
#[pyclass(module = "sidereon._sidereon", name = "MultipathOptions")]
#[derive(Clone, Copy)]
pub struct PyMultipathOptions {
    inner: MultipathOptions,
}

impl PyMultipathOptions {
    fn inner(&self) -> MultipathOptions {
        self.inner
    }
}

#[pymethods]
impl PyMultipathOptions {
    /// Build one-path specular multipath options.
    #[new]
    fn new(
        multipath_to_direct_ratio: f64,
        correlator_spacing_chips: f64,
        receiver_bandwidth_hz: f64,
    ) -> Self {
        Self {
            inner: MultipathOptions {
                multipath_to_direct_ratio,
                correlator_spacing_chips,
                receiver_bandwidth_hz,
            },
        }
    }

    /// Reflected-path amplitude divided by direct-path amplitude.
    #[getter]
    fn multipath_to_direct_ratio(&self) -> f64 {
        self.inner.multipath_to_direct_ratio
    }

    /// Early-late correlator spacing in code chips.
    #[getter]
    fn correlator_spacing_chips(&self) -> f64 {
        self.inner.correlator_spacing_chips
    }

    /// Two-sided receiver bandwidth in hertz.
    #[getter]
    fn receiver_bandwidth_hz(&self) -> f64 {
        self.inner.receiver_bandwidth_hz
    }
}

/// Multipath envelope values on a reflected-delay grid.
#[pyclass(module = "sidereon._sidereon", name = "MultipathEnvelope")]
#[derive(Clone)]
pub struct PyMultipathEnvelope {
    points: Vec<MultipathEnvelopePoint>,
}

impl From<Vec<MultipathEnvelopePoint>> for PyMultipathEnvelope {
    fn from(points: Vec<MultipathEnvelopePoint>) -> Self {
        Self { points }
    }
}

#[pymethods]
impl PyMultipathEnvelope {
    /// Reflected-path delay relative to the direct path, in code chips.
    #[getter]
    fn delay_chips<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.delay_chips)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Reflected-path delay relative to the direct path, in seconds.
    #[getter]
    fn delay_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.delay_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// In-phase multipath tracking error in code chips.
    #[getter]
    fn in_phase_chips<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.in_phase_chips)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// In-phase multipath tracking error in seconds.
    #[getter]
    fn in_phase_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.in_phase_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// In-phase multipath tracking error in metres.
    #[getter]
    fn in_phase_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.in_phase_m)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Anti-phase multipath tracking error in code chips.
    #[getter]
    fn anti_phase_chips<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.anti_phase_chips)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Anti-phase multipath tracking error in seconds.
    #[getter]
    fn anti_phase_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.anti_phase_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Anti-phase multipath tracking error in metres.
    #[getter]
    fn anti_phase_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.anti_phase_m)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Running average of the absolute envelope in code chips.
    #[getter]
    fn running_average_chips<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.running_average_chips)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Running average of the absolute envelope in seconds.
    #[getter]
    fn running_average_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.running_average_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Running average of the absolute envelope in metres.
    #[getter]
    fn running_average_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .points
            .iter()
            .map(|point| point.running_average_m)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Number of delay-grid points.
    fn __len__(&self) -> usize {
        self.points.len()
    }
}

/// Reference chipping-rate unit used by BPSK(n) and BOC(m,n), in hertz.
#[pyfunction]
fn signal_reference_chip_rate_hz() -> f64 {
    REFERENCE_CHIP_RATE_HZ
}

/// Receiver bandwidth used by the Betz L1 SSC fixture, in hertz.
#[pyfunction]
fn signal_betz_l1_receiver_bandwidth_hz() -> f64 {
    BETZ_L1_RECEIVER_BANDWIDTH_HZ
}

/// Return the normalized PSD at one offset frequency.
#[pyfunction]
fn signal_psd_hz(modulation: &PySignalModulation, offset_hz: f64) -> PyResult<f64> {
    modulation.inner().psd_hz(offset_hz).map_err(analysis_err)
}

/// Return the normalized PSD over an offset-frequency array.
#[pyfunction]
fn signal_psd<'py>(
    py: Python<'py>,
    modulation: &PySignalModulation,
    offsets_hz: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let offsets = offsets_hz
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let values = offsets
        .iter()
        .copied()
        .map(|offset| modulation.inner().psd_hz(offset))
        .collect::<Result<Vec<_>, _>>()
        .map_err(analysis_err)?;
    Ok(np_array(py, &values))
}

/// Integrate normalized PSD over a two-sided receiver bandwidth.
#[pyfunction]
fn signal_power_in_band(
    modulation: &PySignalModulation,
    receiver_bandwidth_hz: f64,
) -> PyResult<f64> {
    core_power_in_band(modulation.inner(), receiver_bandwidth_hz).map_err(analysis_err)
}

/// Compute the fraction of total signal power inside a two-sided bandwidth.
#[pyfunction]
fn signal_fraction_power_in_band(
    modulation: &PySignalModulation,
    receiver_bandwidth_hz: f64,
) -> PyResult<f64> {
    core_fraction_power_in_band(modulation.inner(), receiver_bandwidth_hz).map_err(analysis_err)
}

/// Compute the RMS, or Gabor, bandwidth over a two-sided receiver bandwidth.
#[pyfunction]
fn signal_rms_bandwidth_hz(
    modulation: &PySignalModulation,
    receiver_bandwidth_hz: f64,
) -> PyResult<f64> {
    core_rms_bandwidth_hz(modulation.inner(), receiver_bandwidth_hz).map_err(analysis_err)
}

/// Compute the spectral separation coefficient between two modulations.
#[pyfunction]
fn signal_spectral_separation_coefficient_hz(
    desired: &PySignalModulation,
    interference: &PySignalModulation,
    receiver_bandwidth_hz: f64,
) -> PyResult<f64> {
    core_spectral_separation_coefficient_hz(
        desired.inner(),
        interference.inner(),
        receiver_bandwidth_hz,
    )
    .map_err(analysis_err)
}

/// Compute a spectral separation coefficient in decibel-hertz.
#[pyfunction]
fn signal_spectral_separation_coefficient_db_hz(
    desired: &PySignalModulation,
    interference: &PySignalModulation,
    receiver_bandwidth_hz: f64,
) -> PyResult<f64> {
    core_spectral_separation_coefficient_db_hz(
        desired.inner(),
        interference.inner(),
        receiver_bandwidth_hz,
    )
    .map_err(analysis_err)
}

/// Compute the SSC against white interference normalized over the band.
#[pyfunction]
fn signal_white_noise_spectral_separation_hz(
    desired: &PySignalModulation,
    receiver_bandwidth_hz: f64,
) -> PyResult<f64> {
    core_white_noise_spectral_separation_hz(desired.inner(), receiver_bandwidth_hz)
        .map_err(analysis_err)
}

/// Compute effective C/N0 and degradation for finite-band interference.
#[pyfunction]
fn signal_effective_cn0_degradation(
    py: Python<'_>,
    desired: &PySignalModulation,
    cn0_db_hz: f64,
    receiver_bandwidth_hz: f64,
    interferences: Vec<Py<PyInterferenceTerm>>,
) -> PyResult<PyCn0Degradation> {
    let interferences = interferences
        .iter()
        .map(|term| term.borrow(py).inner())
        .collect::<Vec<_>>();
    core_effective_cn0_degradation(
        desired.inner(),
        cn0_db_hz,
        receiver_bandwidth_hz,
        &interferences,
    )
    .map(Into::into)
    .map_err(analysis_err)
}

/// Compute early-late DLL thermal-noise jitter for a modulation.
#[pyfunction]
fn signal_dll_thermal_noise_jitter(
    modulation: &PySignalModulation,
    options: &PyDllTrackingOptions,
    processing: PyDllProcessing,
) -> PyResult<PyDllJitter> {
    core_dll_thermal_noise_jitter(modulation.inner(), options.inner(), processing.into())
        .map(Into::into)
        .map_err(analysis_err)
}

/// Compute the lower bound for code-delay tracking jitter.
#[pyfunction]
fn signal_dll_lower_bound(
    modulation: &PySignalModulation,
    options: &PyDllTrackingOptions,
) -> PyResult<PyDllJitter> {
    core_dll_lower_bound(modulation.inner(), options.inner())
        .map(Into::into)
        .map_err(analysis_err)
}

/// Compute one-path early-late multipath error envelopes on a delay grid.
#[pyfunction]
fn signal_multipath_error_envelope(
    modulation: &PySignalModulation,
    options: &PyMultipathOptions,
    delay_chips: PyReadonlyArray1<'_, f64>,
) -> PyResult<PyMultipathEnvelope> {
    let delay_chips = delay_chips
        .as_slice()
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    core_multipath_error_envelope(modulation.inner(), options.inner(), delay_chips)
        .map(Into::into)
        .map_err(analysis_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySignalModulation>()?;
    m.add_class::<PyInterferenceTerm>()?;
    m.add_class::<PyCn0Degradation>()?;
    m.add_class::<PyDllProcessing>()?;
    m.add_class::<PyDllTrackingOptions>()?;
    m.add_class::<PyDllJitter>()?;
    m.add_class::<PyMultipathOptions>()?;
    m.add_class::<PyMultipathEnvelope>()?;
    m.add_function(wrap_pyfunction!(signal_reference_chip_rate_hz, m)?)?;
    m.add_function(wrap_pyfunction!(signal_betz_l1_receiver_bandwidth_hz, m)?)?;
    m.add_function(wrap_pyfunction!(signal_psd_hz, m)?)?;
    m.add_function(wrap_pyfunction!(signal_psd, m)?)?;
    m.add_function(wrap_pyfunction!(signal_power_in_band, m)?)?;
    m.add_function(wrap_pyfunction!(signal_fraction_power_in_band, m)?)?;
    m.add_function(wrap_pyfunction!(signal_rms_bandwidth_hz, m)?)?;
    m.add_function(wrap_pyfunction!(
        signal_spectral_separation_coefficient_hz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        signal_spectral_separation_coefficient_db_hz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        signal_white_noise_spectral_separation_hz,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(signal_effective_cn0_degradation, m)?)?;
    m.add_function(wrap_pyfunction!(signal_dll_thermal_noise_jitter, m)?)?;
    m.add_function(wrap_pyfunction!(signal_dll_lower_bound, m)?)?;
    m.add_function(wrap_pyfunction!(signal_multipath_error_envelope, m)?)?;
    Ok(())
}
