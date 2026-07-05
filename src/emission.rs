//! Emission-epoch media correction batch binding.
//!
//! Thin marshaling over [`sidereon_core::observables::emission_media_batch_at_j2000_s`].
//! The output keeps contiguous numpy arrays and a typed status per input row.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::constants::F_L1_HZ;
use sidereon_core::observables::{
    emission_media_batch_at_j2000_s as core_emission_media_batch_at_j2000_s, EmissionMediaBatch,
    EmissionMediaBatchOptions, EmissionMediaStatus, ObservableIonosphereCorrection,
    ObservableMediaOptions, ObservableTroposphereCorrection,
    OBSERVABLE_STATE_MISSING_POSITION_ECEF_M,
};

use crate::ephemeris::{observable_state_batch_error, parse_satellites, with_observable_source};
use crate::ionex::PyIonex;
use crate::marshal::rows3_to_array;
use crate::np_array;

fn option_scalar(value: Option<f64>) -> f64 {
    value.unwrap_or(f64::NAN)
}

/// Per-row status for an emission media batch.
#[pyclass(
    module = "sidereon._sidereon",
    name = "EmissionMediaStatus",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyEmissionMediaStatus {
    /// The row contains state, clock, ionosphere, and troposphere outputs.
    VALID,
    /// The ephemeris source has no usable state for this satellite and epoch.
    GAP,
    /// The row had a state, but its elevation was below the requested cutoff.
    BELOW_ELEVATION_CUTOFF,
    /// The scalar evaluator returned a non-gap error.
    ERROR,
}

impl From<EmissionMediaStatus> for PyEmissionMediaStatus {
    fn from(value: EmissionMediaStatus) -> Self {
        match value {
            EmissionMediaStatus::Valid => Self::VALID,
            EmissionMediaStatus::Gap => Self::GAP,
            EmissionMediaStatus::BelowElevationCutoff => Self::BELOW_ELEVATION_CUTOFF,
            EmissionMediaStatus::Error => Self::ERROR,
        }
    }
}

#[pymethods]
impl PyEmissionMediaStatus {
    /// Stable lowercase status label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::VALID => "valid",
            Self::GAP => "gap",
            Self::BELOW_ELEVATION_CUTOFF => "below_elevation_cutoff",
            Self::ERROR => "error",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::VALID => "EmissionMediaStatus.VALID",
            Self::GAP => "EmissionMediaStatus.GAP",
            Self::BELOW_ELEVATION_CUTOFF => "EmissionMediaStatus.BELOW_ELEVATION_CUTOFF",
            Self::ERROR => "EmissionMediaStatus.ERROR",
        }
    }
}

/// Contiguous per-satellite outputs for emission-epoch state and media lookup.
#[pyclass(module = "sidereon._sidereon", name = "EmissionMediaBatch")]
pub struct PyEmissionMediaBatch {
    inner: EmissionMediaBatch,
}

impl From<EmissionMediaBatch> for PyEmissionMediaBatch {
    fn from(inner: EmissionMediaBatch) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEmissionMediaBatch {
    /// Satellite ECEF positions as numpy `(n, 3)`, metres.
    ///
    /// Missing rows are filled with `OBSERVABLE_STATE_MISSING_POSITION_ECEF_M`.
    #[getter]
    fn positions_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows = self
            .inner
            .positions_ecef_m
            .iter()
            .map(|position| position.unwrap_or(OBSERVABLE_STATE_MISSING_POSITION_ECEF_M))
            .collect::<Vec<_>>();
        rows3_to_array(py, &rows)
    }

    /// Satellite clock offsets as numpy `(n,)`, seconds, with NaN for missing.
    #[getter]
    fn clocks_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .clocks_s
            .iter()
            .copied()
            .map(option_scalar)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Ionospheric slant group delays as numpy `(n,)`, metres, with NaN for missing.
    #[getter]
    fn ionosphere_slant_delays_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .ionosphere_slant_delays_m
            .iter()
            .copied()
            .map(option_scalar)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Tropospheric slant delays as numpy `(n,)`, metres, with NaN for missing.
    #[getter]
    fn troposphere_delays_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .troposphere_delays_m
            .iter()
            .copied()
            .map(option_scalar)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Per-row typed statuses.
    #[getter]
    fn statuses(&self) -> Vec<PyEmissionMediaStatus> {
        self.inner
            .statuses
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// Per-row non-success error text.
    #[getter]
    fn element_errors(&self) -> Vec<Option<String>> {
        self.inner
            .element_errors
            .iter()
            .map(|error| error.as_ref().map(ToString::to_string))
            .collect()
    }

    /// Number of batch elements.
    #[getter]
    fn element_count(&self) -> usize {
        self.inner.len()
    }

    /// Whether the batch has no elements.
    #[getter]
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the status for one element.
    fn element_status(&self, index: usize) -> PyResult<PyEmissionMediaStatus> {
        self.inner
            .element_status(index)
            .map(Into::into)
            .ok_or_else(|| PyValueError::new_err(format!("element index {index} out of range")))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("EmissionMediaBatch(element_count={})", self.inner.len())
    }
}

/// Evaluate emission-epoch satellite state and media corrections in one batch.
#[pyfunction]
#[pyo3(signature = (
    source,
    satellites,
    emission_epochs_j2000_s,
    receiver_ecef_m,
    *,
    carrier_hz=F_L1_HZ,
    troposphere=false,
    ionex=None,
    min_elevation_rad=None,
))]
#[allow(clippy::too_many_arguments)]
fn emission_media_batch_at_j2000_s(
    source: &Bound<'_, PyAny>,
    satellites: Vec<String>,
    emission_epochs_j2000_s: PyReadonlyArray1<'_, f64>,
    receiver_ecef_m: [f64; 3],
    carrier_hz: f64,
    troposphere: bool,
    ionex: Option<&PyIonex>,
    min_elevation_rad: Option<f64>,
) -> PyResult<PyEmissionMediaBatch> {
    let satellites = parse_satellites(&satellites)?;
    let epochs = emission_epochs_j2000_s
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let options = EmissionMediaBatchOptions {
        carrier_hz,
        media: ObservableMediaOptions {
            troposphere: troposphere.then(ObservableTroposphereCorrection::default),
            ionosphere: ionex.map(|ionex| ObservableIonosphereCorrection::Ionex(&ionex.inner)),
        },
        min_elevation_rad,
    };
    with_observable_source(source, |source| {
        core_emission_media_batch_at_j2000_s(source, &satellites, epochs, receiver_ecef_m, options)
            .map(Into::into)
            .map_err(observable_state_batch_error)
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEmissionMediaStatus>()?;
    m.add_class::<PyEmissionMediaBatch>()?;
    m.add_function(wrap_pyfunction!(emission_media_batch_at_j2000_s, m)?)?;
    Ok(())
}
