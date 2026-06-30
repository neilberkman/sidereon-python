//! Broadcast-vs-precise accuracy (SISRE) binding.
//!
//! Thin marshaling over [`sidereon_core::broadcast_comparison`]: difference a
//! broadcast navigation product against a precise SP3 product over a window,
//! decomposing the per-satellite-epoch orbit error into radial/along-track/
//! cross-track and summarizing the orbit and clock differences as RMS/max
//! statistics. The differencing, RAC projection, finite-difference velocity, and
//! all aggregation live in `sidereon-core`; the binding only converts the epoch
//! axis into the per-epoch evaluation keys the core consumes and rebuilds the
//! report.
//!
//! The epoch axis crosses the boundary as continuous seconds since J2000 (the
//! broadcast query convention, GPST-aligned). For each epoch the SP3 product is
//! queried at the split Julian date `2451545.0 + t / 86400` (and at the
//! `+/-` velocity-finite-difference neighbours), in the SP3's own header time
//! scale, exactly as the other bindings marshal it.

use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::broadcast_comparison::{
    compare, compare_window as core_compare_window, CompareReport, CompareStats, CompareWindow,
    EpochInputs,
};
use sidereon_core::constants::{J2000_JD, SECONDS_PER_DAY};
use sidereon_core::GnssSatelliteId;

use crate::rinex::PyBroadcastEphemeris;
use crate::PySp3;

fn invalid<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Split a continuous J2000 second into a day-anchored Julian date `(jd_whole,
/// fraction)`, with `jd_whole` the JD at the start of the UTC day (`.5`) and
/// `fraction` the within-day fraction. This matches the split the reference
/// (Elixir) interface marshals, so the SP3 query lands on the same instant.
fn split_jd(t_j2000_s: f64) -> (f64, f64) {
    let jd = J2000_JD + t_j2000_s / SECONDS_PER_DAY;
    let jd_whole = (jd - 0.5).floor() + 0.5;
    (jd_whole, jd - jd_whole)
}

fn epoch_inputs(t_j2000_s: f64, half_s: f64) -> PyResult<EpochInputs> {
    // Split each neighbour independently from its own J2000 second so the
    // day-anchored whole-JD is re-floored per instant, exactly as the reference
    // (Elixir) interface marshals (`epoch +/- half` then split). Folding the
    // half-step into a fixed day's fraction would mis-anchor (and overflow the
    // one-residual-day fraction bound) for epochs near a UTC day boundary.
    let (jd_whole, fraction) = split_jd(t_j2000_s);
    let (jd_whole_p, fraction_p) = split_jd(t_j2000_s + half_s);
    let (jd_whole_m, fraction_m) = split_jd(t_j2000_s - half_s);
    Ok(EpochInputs {
        broadcast_t_j2000_s: t_j2000_s,
        precise: JulianDateSplit::new(jd_whole, fraction).map_err(invalid)?,
        precise_plus: JulianDateSplit::new(jd_whole_p, fraction_p).map_err(invalid)?,
        precise_minus: JulianDateSplit::new(jd_whole_m, fraction_m).map_err(invalid)?,
    })
}

/// Orbit and clock difference statistics for one satellite (or the overall set).
///
/// All values are metres except `count` (the number of compared epochs). The
/// float fields are `None` when no compared epoch populated them. `orbit_3d_*`
/// are the Euclidean position-difference magnitudes; `radial_*`/`along_*`/
/// `cross_*` summarize the signed RAC components; `clock_*` are the raw
/// satellite-clock differences and `clock_datum_removed_*` the same after the
/// per-epoch common reference-clock offset (the SIS clock term) is removed.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastCompareStats")]
#[derive(Clone, Copy)]
pub struct PyCompareStats {
    inner: CompareStats,
}

#[pymethods]
impl PyCompareStats {
    #[getter]
    fn count(&self) -> usize {
        self.inner.count
    }
    #[getter]
    fn orbit_3d_rms_m(&self) -> Option<f64> {
        self.inner.orbit_3d_rms_m
    }
    #[getter]
    fn orbit_3d_max_m(&self) -> Option<f64> {
        self.inner.orbit_3d_max_m
    }
    #[getter]
    fn radial_rms_m(&self) -> Option<f64> {
        self.inner.radial_rms_m
    }
    #[getter]
    fn radial_max_m(&self) -> Option<f64> {
        self.inner.radial_max_m
    }
    #[getter]
    fn along_rms_m(&self) -> Option<f64> {
        self.inner.along_rms_m
    }
    #[getter]
    fn along_max_m(&self) -> Option<f64> {
        self.inner.along_max_m
    }
    #[getter]
    fn cross_rms_m(&self) -> Option<f64> {
        self.inner.cross_rms_m
    }
    #[getter]
    fn cross_max_m(&self) -> Option<f64> {
        self.inner.cross_max_m
    }
    #[getter]
    fn clock_rms_m(&self) -> Option<f64> {
        self.inner.clock_rms_m
    }
    #[getter]
    fn clock_max_m(&self) -> Option<f64> {
        self.inner.clock_max_m
    }
    #[getter]
    fn clock_datum_removed_rms_m(&self) -> Option<f64> {
        self.inner.clock_datum_removed_rms_m
    }
    #[getter]
    fn clock_datum_removed_max_m(&self) -> Option<f64> {
        self.inner.clock_datum_removed_max_m
    }

    fn __repr__(&self) -> String {
        format!(
            "BroadcastCompareStats(count={}, orbit_3d_rms_m={:?})",
            self.inner.count, self.inner.orbit_3d_rms_m
        )
    }
}

/// The result of a broadcast-vs-precise comparison.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastCompareReport")]
pub struct PyCompareReport {
    inner: CompareReport,
}

#[pymethods]
impl PyCompareReport {
    /// Statistics over every compared epoch across all satellites.
    #[getter]
    fn overall(&self) -> PyCompareStats {
        PyCompareStats {
            inner: self.inner.overall,
        }
    }

    /// Per-satellite statistics as a dict mapping each satellite token to its
    /// `BroadcastCompareStats`.
    #[getter]
    fn per_satellite(&self) -> Vec<(String, PyCompareStats)> {
        self.inner
            .per_satellite
            .iter()
            .map(|(sat, stats)| (sat.to_string(), PyCompareStats { inner: *stats }))
            .collect()
    }

    /// Satellites with one or more skipped epochs and their skip counts, as a
    /// list of `(token, count)`.
    #[getter]
    fn missing(&self) -> Vec<(String, usize)> {
        self.inner
            .missing
            .iter()
            .map(|(sat, count)| (sat.to_string(), *count))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "BroadcastCompareReport(satellites={}, overall_count={})",
            self.inner.per_satellite.len(),
            self.inner.overall.count
        )
    }
}

/// Compare a broadcast navigation product against a precise SP3 product.
///
/// `satellites` is a list of RINEX satellite tokens (e.g. `"G01"`);
/// `epochs_j2000_s` is the comparison-epoch axis as continuous seconds since
/// J2000 (GPST-aligned); `step_s` is the epoch spacing, used to size the
/// `round(step_s / 2)` velocity finite-difference half-step. Returns a
/// `BroadcastCompareReport`. Only epochs where both products return a valid
/// state contribute to the statistics. Raises `ValueError` on a malformed token
/// or epoch.
#[pyfunction]
#[pyo3(signature = (broadcast, sp3, satellites, epochs_j2000_s, step_s))]
fn broadcast_comparison(
    broadcast: &PyBroadcastEphemeris,
    sp3: &PySp3,
    satellites: Vec<String>,
    epochs_j2000_s: Vec<f64>,
    step_s: f64,
) -> PyResult<PyCompareReport> {
    let sats: Vec<GnssSatelliteId> = satellites
        .iter()
        .map(|token| {
            GnssSatelliteId::from_str(token)
                .map_err(|_| PyValueError::new_err(format!("invalid satellite token: {token}")))
        })
        .collect::<PyResult<_>>()?;

    let half_s = (step_s / 2.0).round();
    let epochs: Vec<EpochInputs> = epochs_j2000_s
        .iter()
        .map(|&t| epoch_inputs(t, half_s))
        .collect::<PyResult<_>>()?;

    let report = compare(&broadcast.inner, &sp3.inner, &sats, &epochs, half_s).map_err(invalid)?;
    Ok(PyCompareReport { inner: report })
}

/// Compare a broadcast navigation product against a precise SP3 product over a
/// regularly sampled window.
///
/// The window-form sibling of `broadcast_comparison`: instead of an explicit
/// per-epoch axis, the caller supplies an inclusive broadcast query window
/// `(t0_j2000_s, t1_j2000_s)` (continuous seconds since J2000, GPST-aligned), the
/// precise split Julian date for the window start `t0` as `(precise_start_jd_whole,
/// precise_start_fraction)` in the SP3's own header time scale, and the sampling
/// `step_s`. Epochs land at `t0, t0 + step, ...` up to and including `t1`, with a
/// final sample snapped to `t1` when the last step falls short. The precise
/// anchor advances in lockstep with the broadcast axis, so the broadcast-to-precise
/// time-scale offset stays fixed at the value baked into the two start anchors.
/// `velocity_half_s` sizes the velocity finite-difference neighbours; when omitted
/// it defaults to `round(step_s / 2)`, matching `broadcast_comparison`. Returns a
/// `BroadcastCompareReport`. Raises `ValueError` on a malformed token or window.
#[pyfunction]
#[pyo3(signature = (
    broadcast,
    sp3,
    satellites,
    t0_j2000_s,
    t1_j2000_s,
    precise_start_jd_whole,
    precise_start_fraction,
    step_s,
    *,
    velocity_half_s=None,
))]
#[allow(clippy::too_many_arguments)]
fn broadcast_comparison_window(
    broadcast: &PyBroadcastEphemeris,
    sp3: &PySp3,
    satellites: Vec<String>,
    t0_j2000_s: f64,
    t1_j2000_s: f64,
    precise_start_jd_whole: f64,
    precise_start_fraction: f64,
    step_s: f64,
    velocity_half_s: Option<f64>,
) -> PyResult<PyCompareReport> {
    let sats: Vec<GnssSatelliteId> = satellites
        .iter()
        .map(|token| {
            GnssSatelliteId::from_str(token)
                .map_err(|_| PyValueError::new_err(format!("invalid satellite token: {token}")))
        })
        .collect::<PyResult<_>>()?;

    let precise_start =
        JulianDateSplit::new(precise_start_jd_whole, precise_start_fraction).map_err(invalid)?;
    let velocity_half_s = velocity_half_s.unwrap_or_else(|| (step_s / 2.0).round());
    let window = CompareWindow {
        broadcast_window_j2000_s: (t0_j2000_s, t1_j2000_s),
        precise_start,
        step_s,
        velocity_half_s,
    };

    let report =
        core_compare_window(&broadcast.inner, &sp3.inner, &sats, &window).map_err(invalid)?;
    Ok(PyCompareReport { inner: report })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCompareStats>()?;
    m.add_class::<PyCompareReport>()?;
    m.add_function(wrap_pyfunction!(broadcast_comparison, m)?)?;
    m.add_function(wrap_pyfunction!(broadcast_comparison_window, m)?)?;
    Ok(())
}
