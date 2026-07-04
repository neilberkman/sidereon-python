//! Sidereal repeat-period and residual-filter bindings.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::time::Duration;
use sidereon_core::sidereal::{
    orbit_repeat_lag as core_orbit_repeat_lag, periodicity_strength as core_periodicity_strength,
    periodicity_strength_with_sample_interval as core_periodicity_strength_with_sample_interval,
    repeat_period as core_repeat_period, sidereal_filter as core_sidereal_filter,
    SiderealFilterOptions, SiderealFilterOutput, SiderealTemplateMethod,
};
use sidereon_core::GnssSatelliteId;

use crate::marshal::PyGnssSystem;
use crate::np_array;
use crate::rinex::PyBroadcastEphemeris;

fn duration_from_seconds(name: &str, seconds: f64) -> PyResult<Duration> {
    Duration::from_seconds(seconds).map_err(|err| PyValueError::new_err(format!("{name}: {err}")))
}

fn vec_from_array(name: &str, values: PyReadonlyArray1<'_, f64>) -> PyResult<Vec<f64>> {
    values
        .as_slice()
        .map(|slice| slice.to_vec())
        .map_err(|err| PyValueError::new_err(format!("{name} must be contiguous: {err}")))
}

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite token {token:?}: {err}")))
}

/// Template estimator used by the sidereal residual filter.
#[pyclass(module = "sidereon._sidereon", name = "SiderealTemplateMethod")]
#[derive(Clone, Copy)]
pub struct PySiderealTemplateMethod {
    inner: SiderealTemplateMethod,
}

impl PySiderealTemplateMethod {
    fn inner(&self) -> SiderealTemplateMethod {
        self.inner
    }
}

#[pymethods]
impl PySiderealTemplateMethod {
    /// Arithmetic mean of covered prior samples in the same phase bin.
    #[staticmethod]
    fn mean() -> Self {
        Self {
            inner: SiderealTemplateMethod::Mean,
        }
    }

    /// MAD-gated mean of covered prior samples in the same phase bin.
    #[staticmethod]
    fn robust_mad() -> Self {
        Self {
            inner: SiderealTemplateMethod::RobustMad,
        }
    }

    /// Exponentially weighted mean with gain `alpha`.
    #[staticmethod]
    fn ewma(alpha: f64) -> Self {
        Self {
            inner: SiderealTemplateMethod::Ewma { alpha },
        }
    }

    /// Stable lowercase method label.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            SiderealTemplateMethod::Mean => "mean",
            SiderealTemplateMethod::RobustMad => "robust_mad",
            SiderealTemplateMethod::Ewma { .. } => "ewma",
        }
    }

    /// EWMA gain, or `None` for non-EWMA methods.
    #[getter]
    fn alpha(&self) -> Option<f64> {
        match self.inner {
            SiderealTemplateMethod::Ewma { alpha } => Some(alpha),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            SiderealTemplateMethod::Mean => "SiderealTemplateMethod.mean()".to_string(),
            SiderealTemplateMethod::RobustMad => "SiderealTemplateMethod.robust_mad()".to_string(),
            SiderealTemplateMethod::Ewma { alpha } => {
                format!("SiderealTemplateMethod.ewma(alpha={alpha})")
            }
        }
    }
}

/// Options controlling sidereal residual template stacking.
#[pyclass(module = "sidereon._sidereon", name = "SiderealFilterOptions")]
#[derive(Clone, Copy)]
pub struct PySiderealFilterOptions {
    inner: SiderealFilterOptions,
}

impl PySiderealFilterOptions {
    fn inner(&self) -> SiderealFilterOptions {
        self.inner
    }
}

#[pymethods]
impl PySiderealFilterOptions {
    /// Build sidereal filter options.
    #[new]
    #[pyo3(signature = (
        sample_interval_s=1.0,
        prior_periods=1,
        min_coverage=1,
        template_method=None,
    ))]
    fn new(
        sample_interval_s: f64,
        prior_periods: usize,
        min_coverage: usize,
        template_method: Option<&PySiderealTemplateMethod>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SiderealFilterOptions {
                sample_interval: duration_from_seconds("sample_interval_s", sample_interval_s)?,
                prior_periods,
                min_coverage,
                template_method: template_method
                    .map(PySiderealTemplateMethod::inner)
                    .unwrap_or_default(),
            },
        })
    }

    /// Sampling interval of the residual series, seconds.
    #[getter]
    fn sample_interval_s(&self) -> f64 {
        self.inner.sample_interval.as_seconds()
    }

    /// Maximum number of prior repeats retained per phase bin.
    #[getter]
    fn prior_periods(&self) -> usize {
        self.inner.prior_periods
    }

    /// Minimum prior sample count required before subtracting a template value.
    #[getter]
    fn min_coverage(&self) -> usize {
        self.inner.min_coverage
    }

    /// Template estimator used for covered bins.
    #[getter]
    fn template_method(&self) -> PySiderealTemplateMethod {
        PySiderealTemplateMethod {
            inner: self.inner.template_method,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SiderealFilterOptions(sample_interval_s={}, prior_periods={}, min_coverage={}, template_method={})",
            self.sample_interval_s(),
            self.inner.prior_periods,
            self.inner.min_coverage,
            self.template_method().kind()
        )
    }
}

/// Output of sidereal residual filtering.
#[pyclass(module = "sidereon._sidereon", name = "SiderealFilterOutput")]
#[derive(Clone)]
pub struct PySiderealFilterOutput {
    inner: SiderealFilterOutput,
}

impl From<SiderealFilterOutput> for PySiderealFilterOutput {
    fn from(inner: SiderealFilterOutput) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySiderealFilterOutput {
    /// Filtered residuals after covered template subtraction.
    #[getter]
    fn filtered<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.filtered)
    }

    /// Last template value per phase bin, with `NaN` for bins without coverage.
    #[getter]
    fn template<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.template)
    }

    /// Prior sample count behind each phase-bin decision.
    #[getter]
    fn coverage(&self) -> Vec<usize> {
        self.inner.coverage.clone()
    }

    /// Whether each phase bin was under-covered for its last decision.
    #[getter]
    fn under_covered<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        PyArray1::from_slice(py, &self.inner.under_covered)
    }

    /// Number of residual samples returned.
    fn __len__(&self) -> usize {
        self.inner.filtered.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SiderealFilterOutput(samples={})",
            self.inner.filtered.len()
        )
    }
}

/// Default ground-track repeat period for a GNSS constellation, seconds.
#[pyfunction]
fn repeat_period(system: PyGnssSystem) -> f64 {
    core_repeat_period(system.into()).as_seconds()
}

/// Per-satellite broadcast-orbit repeat lag, seconds.
#[pyfunction]
fn orbit_repeat_lag(
    ephemeris: &PyBroadcastEphemeris,
    satellite: &str,
    near_epoch_j2000_s: f64,
) -> PyResult<f64> {
    let satellite = parse_satellite(satellite)?;
    core_orbit_repeat_lag(&ephemeris.inner, satellite, near_epoch_j2000_s)
        .map(Duration::as_seconds)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Apply sidereal residual template filtering.
#[pyfunction]
#[pyo3(signature = (series, period_s, options=None))]
fn sidereal_filter(
    series: PyReadonlyArray1<'_, f64>,
    period_s: f64,
    options: Option<&PySiderealFilterOptions>,
) -> PyResult<PySiderealFilterOutput> {
    let series = vec_from_array("series", series)?;
    let period = duration_from_seconds("period_s", period_s)?;
    let options = options
        .map(PySiderealFilterOptions::inner)
        .unwrap_or_default();
    core_sidereal_filter(&series, period, options)
        .map(Into::into)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Robust periodicity strength for 1 Hz samples.
#[pyfunction]
fn periodicity_strength(
    series: PyReadonlyArray1<'_, f64>,
    candidate_periods_s: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<(f64, f64)>> {
    let series = vec_from_array("series", series)?;
    let periods = vec_from_array("candidate_periods_s", candidate_periods_s)?
        .into_iter()
        .map(|period_s| duration_from_seconds("candidate_periods_s", period_s))
        .collect::<PyResult<Vec<_>>>()?;
    core_periodicity_strength(&series, &periods)
        .map(|values| {
            values
                .into_iter()
                .map(|(period, strength)| (period.as_seconds(), strength))
                .collect()
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Robust periodicity strength for an explicit sample interval.
#[pyfunction]
fn periodicity_strength_with_sample_interval(
    series: PyReadonlyArray1<'_, f64>,
    candidate_periods_s: PyReadonlyArray1<'_, f64>,
    sample_interval_s: f64,
) -> PyResult<Vec<(f64, f64)>> {
    let series = vec_from_array("series", series)?;
    let periods = vec_from_array("candidate_periods_s", candidate_periods_s)?
        .into_iter()
        .map(|period_s| duration_from_seconds("candidate_periods_s", period_s))
        .collect::<PyResult<Vec<_>>>()?;
    let sample_interval = duration_from_seconds("sample_interval_s", sample_interval_s)?;
    core_periodicity_strength_with_sample_interval(&series, &periods, sample_interval)
        .map(|values| {
            values
                .into_iter()
                .map(|(period, strength)| (period.as_seconds(), strength))
                .collect()
        })
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySiderealTemplateMethod>()?;
    m.add_class::<PySiderealFilterOptions>()?;
    m.add_class::<PySiderealFilterOutput>()?;
    m.add_function(wrap_pyfunction!(repeat_period, m)?)?;
    m.add_function(wrap_pyfunction!(orbit_repeat_lag, m)?)?;
    m.add_function(wrap_pyfunction!(sidereal_filter, m)?)?;
    m.add_function(wrap_pyfunction!(periodicity_strength, m)?)?;
    m.add_function(wrap_pyfunction!(
        periodicity_strength_with_sample_interval,
        m
    )?)?;
    Ok(())
}
