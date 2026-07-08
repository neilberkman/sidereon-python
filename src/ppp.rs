//! PPP arc binding: static multi-epoch float PPP.
//!
//! Marshals the structured ionosphere-free epoch records, initial state, and
//! solve config from Python dicts into the `sidereon-core` PPP input types and
//! calls `sidereon::solve_ppp_float`. The SP3 product is the ephemeris source.
//! No modeling lives here.

use std::collections::BTreeMap;
use std::str::FromStr;

use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::atmosphere::troposphere::Met;
use sidereon_core::positioning::SurfaceMet;
use sidereon_core::ppp_corrections::CivilDateTime;
use sidereon_core::precise_positioning::defaults::{
    AMBIGUITY_TOLERANCE_M, CLOCK_TOLERANCE_M, MAX_ITERATIONS, POSITION_TOLERANCE_M,
    RATIO_THRESHOLD, ZTD_TOLERANCE_M,
};
use sidereon_core::precise_positioning::{
    solve_ppp_auto_init_fixed as core_solve_ppp_auto_init_fixed,
    solve_ppp_auto_init_float as core_solve_ppp_auto_init_float, FixedAmbiguityOptions,
    FixedSolution, FixedSolveConfig, FloatEpoch, FloatObservation, FloatSolution, FloatSolveConfig,
    FloatSolveOptions, FloatState, IntegerStatus as PppIntegerStatus, MeasurementWeights,
    PppAutoInitOptions, PppInitialGuess, RangeCorrections, TemporalCorrelationSummary,
    TropoMapping, TroposphereOptions, VmfSiteSample, VmfSiteSeries,
};
use sidereon_core::GnssSatelliteId;

use crate::marshal::{mat3_to_array, option_py_or_default};
use crate::rtk::PyIntegerStatus;
use crate::{np_array, to_solve_err, PySp3};

impl From<PppIntegerStatus> for PyIntegerStatus {
    fn from(status: PppIntegerStatus) -> Self {
        match status {
            PppIntegerStatus::Fixed => PyIntegerStatus::FIXED,
            PppIntegerStatus::NotFixed => PyIntegerStatus::NOT_FIXED,
        }
    }
}

fn mat2_to_array<'py>(py: Python<'py>, matrix: &[[f64; 2]; 2]) -> Bound<'py, PyArray2<f64>> {
    let mut array = Array2::<f64>::zeros((2, 2));
    for row in 0..2 {
        for col in 0..2 {
            array[[row, col]] = matrix[row][col];
        }
    }
    PyArray2::from_owned_array(py, array)
}

// --- input value/config objects -------------------------------------------

/// Civil epoch timestamp for a PPP epoch.
#[pyclass(module = "sidereon._sidereon", name = "PppCivilDateTime")]
#[derive(Clone, Copy)]
pub struct PyPppCivilDateTime {
    inner: CivilDateTime,
}

#[pymethods]
impl PyPppCivilDateTime {
    /// Create a civil timestamp used by the PPP model.
    #[new]
    fn new(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> Self {
        Self {
            inner: CivilDateTime {
                year,
                month,
                day,
                hour,
                minute,
                second,
            },
        }
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
    fn second(&self) -> f64 {
        self.inner.second
    }

    fn __repr__(&self) -> String {
        format!(
            "PppCivilDateTime({:04}-{:02}-{:02} {:02}:{:02}:{:06.3})",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second
        )
    }
}

/// One ionosphere-free code/phase observation in a PPP epoch.
#[pyclass(module = "sidereon._sidereon", name = "PppObservation")]
#[derive(Clone)]
pub struct PyPppObservation {
    inner: FloatObservation,
}

#[pymethods]
impl PyPppObservation {
    /// Create one PPP code/phase observation.
    #[new]
    #[pyo3(signature = (
        satellite_id,
        ambiguity_id,
        code_m,
        phase_m,
        freq1_hz=0.0,
        freq2_hz=0.0,
        glonass_channel=None,
    ))]
    fn new(
        satellite_id: String,
        ambiguity_id: String,
        code_m: f64,
        phase_m: f64,
        freq1_hz: f64,
        freq2_hz: f64,
        glonass_channel: Option<i8>,
    ) -> PyResult<Self> {
        let sat = GnssSatelliteId::from_str(&satellite_id).map_err(|_| {
            PyValueError::new_err(format!("invalid satellite token: {satellite_id}"))
        })?;
        Ok(Self {
            inner: FloatObservation {
                sat,
                satellite_id,
                ambiguity_id,
                code_m,
                phase_m,
                freq1_hz,
                freq2_hz,
                glonass_channel,
            },
        })
    }

    #[getter]
    fn satellite_id(&self) -> &str {
        &self.inner.satellite_id
    }

    #[getter]
    fn ambiguity_id(&self) -> &str {
        &self.inner.ambiguity_id
    }

    #[getter]
    fn code_m(&self) -> f64 {
        self.inner.code_m
    }

    #[getter]
    fn phase_m(&self) -> f64 {
        self.inner.phase_m
    }

    #[getter]
    fn freq1_hz(&self) -> f64 {
        self.inner.freq1_hz
    }

    #[getter]
    fn freq2_hz(&self) -> f64 {
        self.inner.freq2_hz
    }

    #[getter]
    fn glonass_channel(&self) -> Option<i8> {
        self.inner.glonass_channel
    }

    fn __repr__(&self) -> String {
        format!(
            "PppObservation(satellite_id={:?}, ambiguity_id={:?})",
            self.inner.satellite_id, self.inner.ambiguity_id
        )
    }
}

impl PyPppObservation {
    fn to_core(&self) -> FloatObservation {
        self.inner.clone()
    }
}

/// One static PPP epoch.
#[pyclass(module = "sidereon._sidereon", name = "PppEpoch")]
#[derive(Clone)]
pub struct PyPppEpoch {
    inner: FloatEpoch,
}

#[pymethods]
impl PyPppEpoch {
    /// Create a PPP epoch.
    #[new]
    #[pyo3(signature = (civil, jd_whole, jd_fraction, t_rx_j2000_s, observations))]
    fn new(
        py: Python<'_>,
        civil: &PyPppCivilDateTime,
        jd_whole: f64,
        jd_fraction: f64,
        t_rx_j2000_s: f64,
        observations: Vec<Py<PyPppObservation>>,
    ) -> Self {
        let observations = observations
            .iter()
            .map(|obs| obs.borrow(py).to_core())
            .collect();
        Self {
            inner: FloatEpoch {
                epoch: civil.inner,
                jd_whole,
                jd_fraction,
                t_rx_j2000_s,
                observations,
            },
        }
    }

    #[getter]
    fn jd_whole(&self) -> f64 {
        self.inner.jd_whole
    }

    #[getter]
    fn jd_fraction(&self) -> f64 {
        self.inner.jd_fraction
    }

    #[getter]
    fn t_rx_j2000_s(&self) -> f64 {
        self.inner.t_rx_j2000_s
    }

    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.observations.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppEpoch(t_rx_j2000_s={:.3}, observations={})",
            self.inner.t_rx_j2000_s,
            self.inner.observations.len()
        )
    }
}

impl PyPppEpoch {
    fn to_core(&self) -> FloatEpoch {
        self.inner.clone()
    }
}

/// Initial PPP state.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatState")]
#[derive(Clone)]
pub struct PyPppFloatState {
    inner: FloatState,
}

#[pymethods]
impl PyPppFloatState {
    /// Create an initial PPP state.
    #[new]
    #[pyo3(signature = (
        position_m,
        clocks_m,
        ambiguities_m,
        ztd_m=0.0,
        tropo_gradient_north_m=0.0,
        tropo_gradient_east_m=0.0,
        residual_ionosphere_m=None,
    ))]
    fn new(
        position_m: [f64; 3],
        clocks_m: Vec<f64>,
        ambiguities_m: BTreeMap<String, f64>,
        ztd_m: f64,
        tropo_gradient_north_m: f64,
        tropo_gradient_east_m: f64,
        residual_ionosphere_m: Option<BTreeMap<String, f64>>,
    ) -> Self {
        Self {
            inner: FloatState {
                position_m,
                clocks_m,
                ambiguities_m,
                ztd_m,
                tropo_gradient_north_m,
                tropo_gradient_east_m,
                residual_ionosphere_m: residual_ionosphere_m.unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }

    #[getter]
    fn clocks_m(&self) -> Vec<f64> {
        self.inner.clocks_m.clone()
    }

    #[getter]
    fn ambiguities_m(&self) -> BTreeMap<String, f64> {
        self.inner.ambiguities_m.clone()
    }

    #[getter]
    fn ztd_m(&self) -> f64 {
        self.inner.ztd_m
    }

    #[getter]
    fn tropo_gradient_north_m(&self) -> f64 {
        self.inner.tropo_gradient_north_m
    }

    #[getter]
    fn tropo_gradient_east_m(&self) -> f64 {
        self.inner.tropo_gradient_east_m
    }

    #[getter]
    fn residual_ionosphere_m(&self) -> BTreeMap<String, f64> {
        self.inner.residual_ionosphere_m.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatState(position_m=[{:.3}, {:.3}, {:.3}], clocks={})",
            self.inner.position_m[0],
            self.inner.position_m[1],
            self.inner.position_m[2],
            self.inner.clocks_m.len()
        )
    }
}

/// PPP measurement weights.
#[pyclass(module = "sidereon._sidereon", name = "PppMeasurementWeights")]
#[derive(Clone, Copy)]
pub struct PyPppMeasurementWeights {
    inner: MeasurementWeights,
}

#[pymethods]
impl PyPppMeasurementWeights {
    /// Create PPP measurement weights.
    #[new]
    #[pyo3(signature = (code=1.0, phase=100.0, elevation_weighting=false))]
    fn new(code: f64, phase: f64, elevation_weighting: bool) -> Self {
        Self {
            inner: MeasurementWeights {
                code,
                phase,
                elevation_weighting,
            },
        }
    }

    #[getter]
    fn code(&self) -> f64 {
        self.inner.code
    }

    #[getter]
    fn phase(&self) -> f64 {
        self.inner.phase
    }

    #[getter]
    fn elevation_weighting(&self) -> bool {
        self.inner.elevation_weighting
    }

    fn __repr__(&self) -> String {
        format!(
            "PppMeasurementWeights(code={:.3}, phase={:.3}, elevation_weighting={})",
            self.inner.code, self.inner.phase, self.inner.elevation_weighting
        )
    }
}

impl Default for PyPppMeasurementWeights {
    fn default() -> Self {
        Self::new(1.0, 100.0, false)
    }
}

/// PPP troposphere controls.
#[pyclass(module = "sidereon._sidereon", name = "PppTroposphereOptions")]
#[derive(Clone, Copy)]
pub struct PyPppTroposphereOptions {
    inner: TroposphereOptions,
}

#[pymethods]
impl PyPppTroposphereOptions {
    /// Create PPP troposphere controls.
    #[new]
    #[pyo3(signature = (
        enabled=false,
        estimate_ztd=false,
        estimate_tropo_gradients=false,
        pressure_hpa=SurfaceMet::default().pressure_hpa,
        temperature_k=SurfaceMet::default().temperature_k,
        relative_humidity=SurfaceMet::default().relative_humidity,
        vmf1_samples=None,
    ))]
    fn new(
        enabled: bool,
        estimate_ztd: bool,
        estimate_tropo_gradients: bool,
        pressure_hpa: f64,
        temperature_k: f64,
        relative_humidity: f64,
        vmf1_samples: Option<Vec<(f64, f64, f64)>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: if enabled {
                // Default to the climatological Niell mapping; if the caller
                // supplies a 6-hourly VMF1 site-wise `a`-coefficient series
                // (`(mjd, ah, aw)` rows) switch to the VMF1 mapping.
                let mapping = match vmf1_samples {
                    Some(rows) => {
                        let samples: Vec<VmfSiteSample> = rows
                            .into_iter()
                            .map(|(mjd, ah, aw)| VmfSiteSample { mjd, ah, aw })
                            .collect();
                        let series = VmfSiteSeries::new(&samples)
                            .map_err(|err| PyValueError::new_err(err.to_string()))?;
                        TropoMapping::Vmf1(series)
                    }
                    None => TropoMapping::Niell,
                };
                TroposphereOptions {
                    enabled: true,
                    estimate_ztd,
                    estimate_tropo_gradients,
                    met: Met::new(pressure_hpa, temperature_k, relative_humidity)
                        .map_err(|err| PyValueError::new_err(err.to_string()))?,
                    mapping,
                }
            } else {
                TroposphereOptions::disabled()
            },
        })
    }

    #[getter]
    fn enabled(&self) -> bool {
        self.inner.enabled
    }

    #[getter]
    fn estimate_ztd(&self) -> bool {
        self.inner.estimate_ztd
    }

    #[getter]
    fn estimate_tropo_gradients(&self) -> bool {
        self.inner.estimate_tropo_gradients
    }

    #[getter]
    fn pressure_hpa(&self) -> f64 {
        self.inner.met.pressure_hpa
    }

    #[getter]
    fn temperature_k(&self) -> f64 {
        self.inner.met.temperature_k
    }

    #[getter]
    fn relative_humidity(&self) -> f64 {
        self.inner.met.relative_humidity
    }

    /// The tropospheric mapping function in effect: `"niell"` or `"vmf1"`.
    #[getter]
    fn mapping(&self) -> &'static str {
        match self.inner.mapping {
            TropoMapping::Niell => "niell",
            TropoMapping::Vmf1(_) => "vmf1",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PppTroposphereOptions(enabled={}, estimate_ztd={}, mapping={:?})",
            self.inner.enabled,
            self.inner.estimate_ztd,
            self.mapping(),
        )
    }
}

impl Default for PyPppTroposphereOptions {
    fn default() -> Self {
        Self {
            inner: TroposphereOptions::disabled(),
        }
    }
}

/// Iteration and convergence controls for PPP.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatOptions")]
#[derive(Clone, Copy)]
pub struct PyPppFloatOptions {
    inner: FloatSolveOptions,
}

#[pymethods]
impl PyPppFloatOptions {
    /// Create PPP solve controls.
    #[new]
    #[pyo3(signature = (
        max_iterations=MAX_ITERATIONS,
        position_tolerance_m=POSITION_TOLERANCE_M,
        clock_tolerance_m=CLOCK_TOLERANCE_M,
        ambiguity_tolerance_m=AMBIGUITY_TOLERANCE_M,
        ztd_tolerance_m=ZTD_TOLERANCE_M,
    ))]
    fn new(
        max_iterations: usize,
        position_tolerance_m: f64,
        clock_tolerance_m: f64,
        ambiguity_tolerance_m: f64,
        ztd_tolerance_m: f64,
    ) -> Self {
        Self {
            inner: FloatSolveOptions {
                max_iterations,
                position_tolerance_m,
                clock_tolerance_m,
                ambiguity_tolerance_m,
                ztd_tolerance_m,
            },
        }
    }

    #[getter]
    fn max_iterations(&self) -> usize {
        self.inner.max_iterations
    }

    #[getter]
    fn position_tolerance_m(&self) -> f64 {
        self.inner.position_tolerance_m
    }

    #[getter]
    fn clock_tolerance_m(&self) -> f64 {
        self.inner.clock_tolerance_m
    }

    #[getter]
    fn ambiguity_tolerance_m(&self) -> f64 {
        self.inner.ambiguity_tolerance_m
    }

    #[getter]
    fn ztd_tolerance_m(&self) -> f64 {
        self.inner.ztd_tolerance_m
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatOptions(max_iterations={}, position_tolerance_m={:.3e})",
            self.inner.max_iterations, self.inner.position_tolerance_m
        )
    }
}

impl Default for PyPppFloatOptions {
    fn default() -> Self {
        Self::new(
            MAX_ITERATIONS,
            POSITION_TOLERANCE_M,
            CLOCK_TOLERANCE_M,
            AMBIGUITY_TOLERANCE_M,
            ZTD_TOLERANCE_M,
        )
    }
}

/// Complete typed configuration for a PPP float solve.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatConfig")]
pub struct PyPppFloatConfig {
    inner: FloatSolveConfig,
}

#[pymethods]
impl PyPppFloatConfig {
    /// Create a PPP float solve configuration.
    #[new]
    #[pyo3(signature = (
        weights=None,
        tropo=None,
        options=None,
        residual_screen=false,
        elevation_cutoff_deg=None,
        estimate_residual_ionosphere=false,
    ))]
    fn new(
        py: Python<'_>,
        weights: Option<Py<PyPppMeasurementWeights>>,
        tropo: Option<Py<PyPppTroposphereOptions>>,
        options: Option<Py<PyPppFloatOptions>>,
        residual_screen: bool,
        elevation_cutoff_deg: Option<f64>,
        estimate_residual_ionosphere: bool,
    ) -> Self {
        let weights = option_py_or_default(
            py,
            weights.as_ref(),
            |value| value.inner,
            || PyPppMeasurementWeights::default().inner,
        );
        let tropo = option_py_or_default(
            py,
            tropo.as_ref(),
            |value| value.inner,
            || PyPppTroposphereOptions::default().inner,
        );
        let opts = option_py_or_default(
            py,
            options.as_ref(),
            |value| value.inner,
            || PyPppFloatOptions::default().inner,
        );
        Self {
            inner: FloatSolveConfig {
                weights,
                tropo,
                corrections: RangeCorrections::disabled(),
                opts,
                elevation_cutoff_deg,
                residual_screen,
                estimate_residual_ionosphere,
            },
        }
    }

    #[getter]
    fn residual_screen(&self) -> bool {
        self.inner.residual_screen
    }

    #[getter]
    fn elevation_cutoff_deg(&self) -> Option<f64> {
        self.inner.elevation_cutoff_deg
    }

    #[getter]
    fn estimate_residual_ionosphere(&self) -> bool {
        self.inner.estimate_residual_ionosphere
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatConfig(residual_screen={}, max_iterations={})",
            self.inner.residual_screen, self.inner.opts.max_iterations
        )
    }
}

/// Integer ambiguity controls for PPP fixed solving.
#[pyclass(module = "sidereon._sidereon", name = "PppFixedAmbiguityOptions")]
pub struct PyPppFixedAmbiguityOptions {
    inner: FixedAmbiguityOptions,
}

#[pymethods]
impl PyPppFixedAmbiguityOptions {
    /// Create PPP fixed ambiguity-search controls.
    #[new]
    #[pyo3(signature = (wavelengths_m, offsets_m, ratio_threshold=RATIO_THRESHOLD))]
    fn new(
        wavelengths_m: BTreeMap<String, f64>,
        offsets_m: BTreeMap<String, f64>,
        ratio_threshold: f64,
    ) -> Self {
        Self {
            inner: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold,
            },
        }
    }

    #[getter]
    fn wavelengths_m(&self) -> BTreeMap<String, f64> {
        self.inner.wavelengths_m.clone()
    }

    #[getter]
    fn offsets_m(&self) -> BTreeMap<String, f64> {
        self.inner.offsets_m.clone()
    }

    #[getter]
    fn ratio_threshold(&self) -> f64 {
        self.inner.ratio_threshold
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFixedAmbiguityOptions(ambiguities={}, ratio_threshold={:.3})",
            self.inner.wavelengths_m.len(),
            self.inner.ratio_threshold
        )
    }
}

/// Complete typed configuration for a PPP fixed solve.
#[pyclass(module = "sidereon._sidereon", name = "PppFixedConfig")]
pub struct PyPppFixedConfig {
    inner: FixedSolveConfig,
}

#[pymethods]
impl PyPppFixedConfig {
    /// Create a PPP fixed solve configuration.
    #[new]
    #[pyo3(signature = (
        ambiguity,
        weights=None,
        tropo=None,
        options=None,
        elevation_cutoff_deg=None,
        estimate_residual_ionosphere=false,
    ))]
    fn new(
        py: Python<'_>,
        ambiguity: &PyPppFixedAmbiguityOptions,
        weights: Option<Py<PyPppMeasurementWeights>>,
        tropo: Option<Py<PyPppTroposphereOptions>>,
        options: Option<Py<PyPppFloatOptions>>,
        elevation_cutoff_deg: Option<f64>,
        estimate_residual_ionosphere: bool,
    ) -> Self {
        let weights = option_py_or_default(
            py,
            weights.as_ref(),
            |value| value.inner,
            || PyPppMeasurementWeights::default().inner,
        );
        let tropo = option_py_or_default(
            py,
            tropo.as_ref(),
            |value| value.inner,
            || PyPppTroposphereOptions::default().inner,
        );
        let opts = option_py_or_default(
            py,
            options.as_ref(),
            |value| value.inner,
            || PyPppFloatOptions::default().inner,
        );
        Self {
            inner: FixedSolveConfig {
                weights,
                tropo,
                corrections: RangeCorrections::disabled(),
                opts,
                elevation_cutoff_deg,
                ambiguity: ambiguity.inner.clone(),
                estimate_residual_ionosphere,
            },
        }
    }

    #[getter]
    fn elevation_cutoff_deg(&self) -> Option<f64> {
        self.inner.elevation_cutoff_deg
    }

    #[getter]
    fn estimate_residual_ionosphere(&self) -> bool {
        self.inner.estimate_residual_ionosphere
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFixedConfig(ambiguities={}, max_iterations={})",
            self.inner.ambiguity.wavelengths_m.len(),
            self.inner.opts.max_iterations
        )
    }
}

// --- result object ---------------------------------------------------------

/// Temporal-correlation summary for a static PPP residual sequence.
#[pyclass(module = "sidereon._sidereon", name = "PppTemporalCorrelationSummary")]
#[derive(Clone, Copy)]
pub struct PyPppTemporalCorrelationSummary {
    inner: TemporalCorrelationSummary,
}

impl From<TemporalCorrelationSummary> for PyPppTemporalCorrelationSummary {
    fn from(inner: TemporalCorrelationSummary) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyPppTemporalCorrelationSummary {
    #[getter]
    fn lag1_autocorrelation(&self) -> f64 {
        self.inner.lag1_autocorrelation
    }

    #[getter]
    fn decorrelation_time_epochs(&self) -> f64 {
        self.inner.decorrelation_time_epochs
    }

    #[getter]
    fn decorrelation_time_s(&self) -> Option<f64> {
        self.inner.decorrelation_time_s
    }

    #[getter]
    fn nominal_sample_count(&self) -> usize {
        self.inner.nominal_sample_count
    }

    #[getter]
    fn effective_sample_count(&self) -> f64 {
        self.inner.effective_sample_count
    }

    #[getter]
    fn variance_inflation_factor(&self) -> f64 {
        self.inner.variance_inflation_factor
    }

    #[getter]
    fn arcs_used(&self) -> usize {
        self.inner.arcs_used
    }

    fn __repr__(&self) -> String {
        format!(
            "PppTemporalCorrelationSummary(lag1_autocorrelation={:.3}, variance_inflation_factor={:.3})",
            self.inner.lag1_autocorrelation, self.inner.variance_inflation_factor
        )
    }
}

/// Static float PPP solution.
///
/// `position` is the receiver ECEF position as a numpy `float64` array of shape
/// `(3,)`, metres. Float ambiguities and residual RMS values are in metres.
#[pyclass(module = "sidereon._sidereon", name = "PppFloatSolution")]
pub struct PyPppFloatSolution {
    inner: FloatSolution,
}

#[pymethods]
impl PyPppFloatSolution {
    /// ECEF position as a numpy array `[x_m, y_m, z_m]`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_m)
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }

    /// Float ambiguities in metres, keyed by ambiguity id.
    #[getter]
    fn ambiguities_m(&self) -> BTreeMap<String, f64> {
        self.inner.ambiguities_m.clone()
    }

    #[getter]
    fn ztd_residual_m(&self) -> Option<f64> {
        self.inner.ztd_residual_m
    }

    #[getter]
    fn residual_ionosphere_m(&self) -> BTreeMap<String, f64> {
        self.inner.residual_ionosphere_m.clone()
    }

    #[getter]
    fn tropo_gradient_north_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_north_m
    }

    #[getter]
    fn tropo_gradient_east_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_east_m
    }

    #[getter]
    fn tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn formal_tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .formal_tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn position_covariance_ecef_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.ecef_m2)
    }

    #[getter]
    fn position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.enu_m2)
    }

    #[getter]
    fn formal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.ecef_m2)
    }

    #[getter]
    fn formal_position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.enu_m2)
    }

    #[getter]
    fn posterior_variance_factor(&self) -> f64 {
        self.inner.posterior_variance_factor
    }

    #[getter]
    fn position_covariance_scale_factor(&self) -> f64 {
        self.inner.position_covariance_scale_factor
    }

    #[getter]
    fn temporal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.ecef_m2)
    }

    #[getter]
    fn temporal_position_covariance_enu_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.enu_m2)
    }

    #[getter]
    fn temporal_position_covariance_scale_factor(&self) -> f64 {
        self.inner.temporal_position_covariance_scale_factor
    }

    #[getter]
    fn temporal_correlation(&self) -> PyPppTemporalCorrelationSummary {
        self.inner.temporal_correlation.into()
    }

    #[getter]
    fn code_rms_m(&self) -> f64 {
        self.inner.code_rms_m
    }

    #[getter]
    fn phase_rms_m(&self) -> f64 {
        self.inner.phase_rms_m
    }

    #[getter]
    fn weighted_rms_m(&self) -> f64 {
        self.inner.weighted_rms_m
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner.used_sats.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFloatSolution(position=[{:.3}, {:.3}, {:.3}], phase_rms_m={:.4}, converged={})",
            self.inner.position_m[0],
            self.inner.position_m[1],
            self.inner.position_m[2],
            self.inner.phase_rms_m,
            self.inner.converged
        )
    }
}

/// Static integer-fixed PPP solution.
///
/// `position` is the receiver ECEF position as a numpy `float64` array of shape
/// `(3,)`, metres. Fixed ambiguities are exposed in cycles and metres.
#[pyclass(module = "sidereon._sidereon", name = "PppFixedSolution")]
pub struct PyPppFixedSolution {
    inner: FixedSolution,
}

#[pymethods]
impl PyPppFixedSolution {
    /// ECEF position as a numpy array `[x_m, y_m, z_m]`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_m)
    }

    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.inner.position_m
    }

    /// Fixed ambiguities in cycles, keyed by ambiguity id.
    #[getter]
    fn fixed_ambiguities_cycles(&self) -> BTreeMap<String, i64> {
        self.inner.fixed_ambiguities_cycles.clone()
    }

    /// Fixed ambiguities in metres, keyed by ambiguity id.
    #[getter]
    fn fixed_ambiguities_m(&self) -> BTreeMap<String, f64> {
        self.inner.fixed_ambiguities_m.clone()
    }

    #[getter]
    fn ztd_residual_m(&self) -> Option<f64> {
        self.inner.ztd_residual_m
    }

    #[getter]
    fn residual_ionosphere_m(&self) -> BTreeMap<String, f64> {
        self.inner.residual_ionosphere_m.clone()
    }

    #[getter]
    fn tropo_gradient_north_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_north_m
    }

    #[getter]
    fn tropo_gradient_east_m(&self) -> Option<f64> {
        self.inner.tropo_gradient_east_m
    }

    #[getter]
    fn tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn formal_tropo_gradient_covariance_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<Bound<'py, PyArray2<f64>>> {
        self.inner
            .formal_tropo_gradient_covariance_m2
            .as_ref()
            .map(|matrix| mat2_to_array(py, matrix))
    }

    #[getter]
    fn position_covariance_ecef_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.ecef_m2)
    }

    #[getter]
    fn position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.enu_m2)
    }

    #[getter]
    fn formal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.ecef_m2)
    }

    #[getter]
    fn formal_position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.formal_position_covariance.enu_m2)
    }

    #[getter]
    fn posterior_variance_factor(&self) -> f64 {
        self.inner.posterior_variance_factor
    }

    #[getter]
    fn position_covariance_scale_factor(&self) -> f64 {
        self.inner.position_covariance_scale_factor
    }

    #[getter]
    fn temporal_position_covariance_ecef_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.ecef_m2)
    }

    #[getter]
    fn temporal_position_covariance_enu_m2<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.temporal_position_covariance.enu_m2)
    }

    #[getter]
    fn temporal_position_covariance_scale_factor(&self) -> f64 {
        self.inner.temporal_position_covariance_scale_factor
    }

    #[getter]
    fn temporal_correlation(&self) -> PyPppTemporalCorrelationSummary {
        self.inner.temporal_correlation.into()
    }

    /// The float solution that seeded integer search.
    #[getter]
    fn float_solution(&self) -> PyPppFloatSolution {
        PyPppFloatSolution {
            inner: self.inner.float_solution.clone(),
        }
    }

    #[getter]
    fn integer_status(&self) -> PyIntegerStatus {
        self.inner.integer.integer_status.into()
    }

    #[getter]
    fn integer_ratio(&self) -> f64 {
        self.inner.integer.integer_ratio
    }

    #[getter]
    fn integer_candidates(&self) -> usize {
        self.inner.integer.integer_candidates
    }

    #[getter]
    fn code_rms_m(&self) -> f64 {
        self.inner.code_rms_m
    }

    #[getter]
    fn phase_rms_m(&self) -> f64 {
        self.inner.phase_rms_m
    }

    #[getter]
    fn weighted_rms_m(&self) -> f64 {
        self.inner.weighted_rms_m
    }

    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    #[getter]
    fn iterations(&self) -> usize {
        self.inner.iterations
    }

    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner.used_sats.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppFixedSolution(position=[{:.3}, {:.3}, {:.3}], integer_status={:?}, converged={})",
            self.inner.position_m[0],
            self.inner.position_m[1],
            self.inner.position_m[2],
            self.inner.integer.integer_status,
            self.inner.converged
        )
    }
}

/// SPP-seeded auto-initialization policy for the raw-epochs PPP drivers.
///
/// With no explicit guess the driver seeds the static float state from a
/// per-epoch single-point-positioning solve (ionosphere off, optional
/// troposphere) using `spp_initial_guess` as the cold start. Supplying
/// `initial_guess_position_m` skips the SPP/mean stages and uses that position
/// (with `initial_guess_clock_m`, duplicated across epochs) directly.
#[pyclass(module = "sidereon._sidereon", name = "PppAutoInitOptions")]
#[derive(Clone, Copy, Default)]
pub struct PyPppAutoInitOptions {
    inner: PppAutoInitOptions,
}

#[pymethods]
impl PyPppAutoInitOptions {
    /// Create PPP auto-init controls.
    #[new]
    #[pyo3(signature = (
        spp_initial_guess=[0.0; 4],
        spp_troposphere=false,
        pressure_hpa=SurfaceMet::default().pressure_hpa,
        temperature_k=SurfaceMet::default().temperature_k,
        relative_humidity=SurfaceMet::default().relative_humidity,
        initial_guess_position_m=None,
        initial_guess_clock_m=0.0,
    ))]
    fn new(
        spp_initial_guess: [f64; 4],
        spp_troposphere: bool,
        pressure_hpa: f64,
        temperature_k: f64,
        relative_humidity: f64,
        initial_guess_position_m: Option<[f64; 3]>,
        initial_guess_clock_m: f64,
    ) -> Self {
        let initial_guess = initial_guess_position_m.map(|position_m| PppInitialGuess {
            position_m,
            clock_m: initial_guess_clock_m,
        });
        Self {
            inner: PppAutoInitOptions {
                initial_guess,
                spp_initial_guess,
                spp_troposphere,
                spp_met: SurfaceMet {
                    pressure_hpa,
                    temperature_k,
                    relative_humidity,
                },
            },
        }
    }

    #[getter]
    fn spp_initial_guess(&self) -> [f64; 4] {
        self.inner.spp_initial_guess
    }

    #[getter]
    fn spp_troposphere(&self) -> bool {
        self.inner.spp_troposphere
    }

    #[getter]
    fn initial_guess_position_m(&self) -> Option<[f64; 3]> {
        self.inner.initial_guess.map(|guess| guess.position_m)
    }

    #[getter]
    fn initial_guess_clock_m(&self) -> Option<f64> {
        self.inner.initial_guess.map(|guess| guess.clock_m)
    }

    fn __repr__(&self) -> String {
        format!(
            "PppAutoInitOptions(spp_troposphere={}, has_initial_guess={})",
            self.inner.spp_troposphere,
            self.inner.initial_guess.is_some()
        )
    }
}

// --- solve entry points ----------------------------------------------------

#[pyfunction]
#[pyo3(signature = (sp3, epochs, initial_state, config))]
fn solve_ppp_float(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    initial_state: &PyPppFloatState,
    config: &PyPppFloatConfig,
) -> PyResult<PyPppFloatSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let inner = sidereon::solve_ppp_float(
        &sp3.inner,
        &epochs,
        initial_state.inner.clone(),
        config.inner.clone(),
    )
    .map_err(to_solve_err)?;
    Ok(PyPppFloatSolution { inner })
}

#[pyfunction]
#[pyo3(signature = (sp3, epochs, float_solution, config))]
fn solve_ppp_fixed(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    float_solution: &PyPppFloatSolution,
    config: &PyPppFixedConfig,
) -> PyResult<PyPppFixedSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let inner = sidereon::solve_ppp_fixed(
        &sp3.inner,
        &epochs,
        float_solution.inner.clone(),
        config.inner.clone(),
    )
    .map_err(to_solve_err)?;
    Ok(PyPppFixedSolution { inner })
}

/// Solve a static multi-epoch float PPP arc from raw epochs, auto-initializing
/// the float state from the SPP seed described by `options`.
///
/// Unlike [`solve_ppp_float`], no explicit initial `FloatState` is supplied: the
/// driver seeds it (per-epoch SPP position/clock, phase-minus-code ambiguities,
/// zero ZTD) and then runs the same static float solve. The SP3 product is both
/// the SPP seed ephemeris and the PPP observable ephemeris. Raises `SolveError`
/// on a seed or float-solve failure.
#[pyfunction]
#[pyo3(signature = (sp3, epochs, config, options=None))]
fn solve_ppp_auto_init_float(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    config: &PyPppFloatConfig,
    options: Option<Py<PyPppAutoInitOptions>>,
) -> PyResult<PyPppFloatSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |value| value.inner,
        || PyPppAutoInitOptions::default().inner,
    );
    let inner = core_solve_ppp_auto_init_float(&sp3.inner, &epochs, options, config.inner.clone())
        .map_err(to_solve_err)?;
    Ok(PyPppFloatSolution { inner })
}

/// Solve a static integer-fixed PPP arc from raw epochs: auto-init seed, the
/// float solve, then the LAMBDA integer fix and ambiguity-conditioned re-solve.
///
/// This is the auto-initialized counterpart of [`solve_ppp_fixed`]: the float
/// arc is seeded from the SPP auto-init `options` (no explicit `FloatState` or
/// float solution is supplied), then the integer search and fixed re-solve run.
/// Raises `SolveError` on a seed, float-solve, or fixed-solve failure.
#[pyfunction]
#[pyo3(signature = (sp3, epochs, float_config, fixed_config, options=None))]
fn solve_ppp_auto_init_fixed(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<Py<PyPppEpoch>>,
    float_config: &PyPppFloatConfig,
    fixed_config: &PyPppFixedConfig,
    options: Option<Py<PyPppAutoInitOptions>>,
) -> PyResult<PyPppFixedSolution> {
    let epochs: Vec<FloatEpoch> = epochs
        .iter()
        .map(|epoch| epoch.borrow(py).to_core())
        .collect();
    let options = option_py_or_default(
        py,
        options.as_ref(),
        |value| value.inner,
        || PyPppAutoInitOptions::default().inner,
    );
    let inner = core_solve_ppp_auto_init_fixed(
        &sp3.inner,
        &epochs,
        options,
        float_config.inner.clone(),
        fixed_config.inner.clone(),
    )
    .map_err(to_solve_err)?;
    Ok(PyPppFixedSolution { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPppFloatSolution>()?;
    m.add_class::<PyPppTemporalCorrelationSummary>()?;
    m.add_function(wrap_pyfunction!(solve_ppp_float, m)?)?;
    m.add_class::<PyPppCivilDateTime>()?;
    m.add_class::<PyPppObservation>()?;
    m.add_class::<PyPppEpoch>()?;
    m.add_class::<PyPppFloatState>()?;
    m.add_class::<PyPppMeasurementWeights>()?;
    m.add_class::<PyPppTroposphereOptions>()?;
    m.add_class::<PyPppFloatOptions>()?;
    m.add_class::<PyPppFloatConfig>()?;
    m.add_class::<PyPppFixedAmbiguityOptions>()?;
    m.add_class::<PyPppFixedConfig>()?;
    m.add_class::<PyPppFixedSolution>()?;
    m.add_function(wrap_pyfunction!(solve_ppp_fixed, m)?)?;
    m.add_class::<PyPppAutoInitOptions>()?;
    m.add_function(wrap_pyfunction!(solve_ppp_auto_init_float, m)?)?;
    m.add_function(wrap_pyfunction!(solve_ppp_auto_init_fixed, m)?)?;
    Ok(())
}
