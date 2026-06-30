//! SP3-backed GNSS geometry: visibility, visibility series, and passes.
//!
//! This module is a thin INTERFACE over `sidereon_core::geometry`. It decodes
//! the loaded SP3 handle, receiver ECEF, epoch/window scalars, and the API-level
//! elevation mask and constellation filter, then calls the crate's
//! [`visible`](sidereon_core::geometry::visible) /
//! [`visibility_series`](sidereon_core::geometry::visibility_series) /
//! [`passes`](sidereon_core::geometry::passes) functions. Every number it
//! returns is produced by the core; no visibility, masking, or pass-segmentation
//! logic lives here.

use std::collections::BTreeSet;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use numpy::PyReadonlyArray1;

use sidereon_core::geometry::{
    passes as core_passes, visibility_series as core_visibility_series, visible as core_visible,
    VisibilityOptions,
};
use sidereon_core::GnssSystem;

use crate::marshal::{fixed_array, FinitePolicy};
use crate::{to_solve_err, PySp3};

/// One satellite visible from the receiver at an epoch.
#[pyclass(module = "sidereon._sidereon", name = "GnssVisibleSatellite")]
#[derive(Clone)]
pub struct PyVisibleSatellite {
    satellite: String,
    elevation_deg: f64,
    azimuth_deg: f64,
}

#[pymethods]
impl PyVisibleSatellite {
    /// Canonical satellite identifier (e.g. `"G05"`).
    #[getter]
    fn satellite(&self) -> &str {
        &self.satellite
    }

    /// Topocentric elevation, degrees.
    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.elevation_deg
    }

    /// Topocentric azimuth in `[0, 360)`, degrees.
    #[getter]
    fn azimuth_deg(&self) -> f64 {
        self.azimuth_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "VisibleSatellite(satellite={:?}, elevation_deg={:.3}, azimuth_deg={:.3})",
            self.satellite, self.elevation_deg, self.azimuth_deg
        )
    }
}

/// Visible-satellite count for one sampled epoch.
#[pyclass(module = "sidereon._sidereon", name = "VisibilitySeriesPoint")]
#[derive(Clone)]
pub struct PyVisibilitySeriesPoint {
    step_index: usize,
    n_visible: usize,
}

#[pymethods]
impl PyVisibilitySeriesPoint {
    /// Zero-based sample index from the series start.
    #[getter]
    fn step_index(&self) -> usize {
        self.step_index
    }

    /// Number of satellites visible at this sample.
    #[getter]
    fn n_visible(&self) -> usize {
        self.n_visible
    }

    fn __repr__(&self) -> String {
        format!(
            "VisibilitySeriesPoint(step_index={}, n_visible={})",
            self.step_index, self.n_visible
        )
    }
}

/// One sampled rise/set/peak visibility pass.
#[pyclass(module = "sidereon._sidereon", name = "VisibilityPass")]
#[derive(Clone)]
pub struct PyVisibilityPass {
    satellite: String,
    rise_step_index: usize,
    set_step_index: usize,
    peak_elevation_deg: f64,
    peak_step_index: usize,
}

#[pymethods]
impl PyVisibilityPass {
    /// Canonical satellite identifier.
    #[getter]
    fn satellite(&self) -> &str {
        &self.satellite
    }

    /// Zero-based sample index of the first above-mask sample.
    #[getter]
    fn rise_step_index(&self) -> usize {
        self.rise_step_index
    }

    /// Zero-based sample index of the last above-mask sample.
    #[getter]
    fn set_step_index(&self) -> usize {
        self.set_step_index
    }

    /// Maximum sampled elevation in the pass, degrees.
    #[getter]
    fn peak_elevation_deg(&self) -> f64 {
        self.peak_elevation_deg
    }

    /// Zero-based sample index of the maximum sampled elevation.
    #[getter]
    fn peak_step_index(&self) -> usize {
        self.peak_step_index
    }

    fn __repr__(&self) -> String {
        format!(
            "VisibilityPass(satellite={:?}, rise_step_index={}, set_step_index={}, \
             peak_elevation_deg={:.3}, peak_step_index={})",
            self.satellite,
            self.rise_step_index,
            self.set_step_index,
            self.peak_elevation_deg,
            self.peak_step_index
        )
    }
}

fn system_filter(systems: Option<Vec<String>>) -> PyResult<Option<BTreeSet<GnssSystem>>> {
    let Some(systems) = systems else {
        return Ok(None);
    };
    let mut set = BTreeSet::new();
    for system in &systems {
        let mut chars = system.chars();
        let parsed = chars
            .next()
            .filter(|_| chars.next().is_none())
            .and_then(GnssSystem::from_letter);
        match parsed {
            Some(system) => {
                set.insert(system);
            }
            None => {
                return Err(PyValueError::new_err(format!(
                    "invalid GNSS system letter {system:?}; expected one of G, R, E, C, J, I, S"
                )));
            }
        }
    }
    Ok(Some(set))
}

fn visibility_options(
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
) -> PyResult<VisibilityOptions> {
    Ok(VisibilityOptions {
        elevation_mask_deg,
        systems: system_filter(systems)?,
    })
}

/// Satellites visible from a static receiver at one epoch.
///
/// `receiver_ecef_m` is a 3-vector in ITRF meters; `t_rx_j2000_s` is the receive
/// time in seconds since J2000. Rows are sorted by descending elevation.
#[pyfunction]
#[pyo3(signature = (sp3, receiver_ecef_m, t_rx_j2000_s, elevation_mask_deg, systems=None))]
fn visible(
    sp3: &PySp3,
    receiver_ecef_m: PyReadonlyArray1<'_, f64>,
    t_rx_j2000_s: f64,
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
) -> PyResult<Vec<PyVisibleSatellite>> {
    let receiver_ecef_m = fixed_array::<3>(
        "receiver_ecef_m",
        &receiver_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let options = visibility_options(elevation_mask_deg, systems)?;
    let rows = core_visible(
        &sp3.inner,
        sp3.inner.satellites(),
        receiver_ecef_m,
        t_rx_j2000_s,
        &options,
    )
    .map_err(to_solve_err)?;
    Ok(rows
        .into_iter()
        .map(|row| PyVisibleSatellite {
            satellite: row.satellite.to_string(),
            elevation_deg: row.elevation_deg,
            azimuth_deg: row.azimuth_deg,
        })
        .collect())
}

/// Count visible satellites over an inclusive sampled time window.
#[pyfunction]
#[pyo3(signature = (
    sp3,
    receiver_ecef_m,
    start_j2000_s,
    end_j2000_s,
    step_seconds,
    elevation_mask_deg,
    systems=None,
))]
#[allow(clippy::too_many_arguments)]
fn visibility_series(
    sp3: &PySp3,
    receiver_ecef_m: PyReadonlyArray1<'_, f64>,
    start_j2000_s: f64,
    end_j2000_s: f64,
    step_seconds: u64,
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
) -> PyResult<Vec<PyVisibilitySeriesPoint>> {
    let receiver_ecef_m = fixed_array::<3>(
        "receiver_ecef_m",
        &receiver_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let options = visibility_options(elevation_mask_deg, systems)?;
    let points = core_visibility_series(
        &sp3.inner,
        sp3.inner.satellites(),
        receiver_ecef_m,
        (start_j2000_s, end_j2000_s),
        step_seconds,
        &options,
    )
    .map_err(to_solve_err)?;
    Ok(points
        .into_iter()
        .map(|point| PyVisibilitySeriesPoint {
            step_index: point.step_index,
            n_visible: point.n_visible,
        })
        .collect())
}

/// Build sampled rise/set/peak passes over an inclusive time window.
#[pyfunction]
#[pyo3(signature = (
    sp3,
    receiver_ecef_m,
    start_j2000_s,
    end_j2000_s,
    step_seconds,
    elevation_mask_deg,
    systems=None,
))]
#[allow(clippy::too_many_arguments)]
fn passes(
    sp3: &PySp3,
    receiver_ecef_m: PyReadonlyArray1<'_, f64>,
    start_j2000_s: f64,
    end_j2000_s: f64,
    step_seconds: u64,
    elevation_mask_deg: f64,
    systems: Option<Vec<String>>,
) -> PyResult<Vec<PyVisibilityPass>> {
    let receiver_ecef_m = fixed_array::<3>(
        "receiver_ecef_m",
        &receiver_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let options = visibility_options(elevation_mask_deg, systems)?;
    let found = core_passes(
        &sp3.inner,
        sp3.inner.satellites(),
        receiver_ecef_m,
        (start_j2000_s, end_j2000_s),
        step_seconds,
        &options,
    )
    .map_err(to_solve_err)?;
    Ok(found
        .into_iter()
        .map(|pass| PyVisibilityPass {
            satellite: pass.satellite.to_string(),
            rise_step_index: pass.rise_step_index,
            set_step_index: pass.set_step_index,
            peak_elevation_deg: pass.peak_elevation_deg,
            peak_step_index: pass.peak_step_index,
        })
        .collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVisibleSatellite>()?;
    m.add_class::<PyVisibilitySeriesPoint>()?;
    m.add_class::<PyVisibilityPass>()?;
    m.add_function(wrap_pyfunction!(visible, m)?)?;
    m.add_function(wrap_pyfunction!(visibility_series, m)?)?;
    m.add_function(wrap_pyfunction!(passes, m)?)?;
    Ok(())
}
