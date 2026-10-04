//! One-epoch satellite/station coverage grid binding.
//!
//! Thin INTERFACE over `sidereon_core::astro::coverage`. It collects the
//! initialized SGP4 satellites and ground stations, resolves the epoch, and
//! calls [`look_angles_batch`](sidereon_core::astro::coverage::look_angles_batch)
//! to build the `[satellite][station]` look-angle grid, then delegates the
//! reductions to
//! [`visible_mask`](sidereon_core::astro::coverage::visible_mask),
//! [`access_counts`](sidereon_core::astro::coverage::access_counts), and
//! [`max_elevation`](sidereon_core::astro::coverage::max_elevation). No look
//! angle, visibility, or reduction math lives here.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::Bound;

use sidereon::passes::UtcInstant;
use sidereon::sgp4::Satellite;
use sidereon_core::astro::coverage::{
    access_counts as core_access_counts, look_angles_batch as core_look_angles_batch,
    max_elevation as core_max_elevation, visible_mask as core_visible_mask, LookAngleGrid,
};

use crate::propagation::{PyGroundStation, PyTle};

/// A `[satellite][station]` look-angle grid for one epoch.
///
/// Each cell is the topocentric look angle of one satellite from one station, or
/// a failure (surfaced as `NaN` in the angle arrays). The visibility reductions
/// are delegated to the core.
#[pyclass(module = "sidereon._sidereon", name = "CoverageGrid")]
pub struct PyCoverageGrid {
    grid: LookAngleGrid,
    n_satellites: usize,
    n_stations: usize,
}

/// Build a `(n_satellites, n_stations)` float64 array, mapping a successful cell
/// through `value` and an error cell to `NaN`.
fn cell_array<'py>(
    py: Python<'py>,
    grid: &LookAngleGrid,
    shape: (usize, usize),
    value: impl Fn(&sidereon_core::astro::passes::LookAngle) -> f64,
) -> Bound<'py, PyArray2<f64>> {
    let mut array = Array2::<f64>::from_elem(shape, f64::NAN);
    for (sat_index, row) in grid.iter().enumerate() {
        for (station_index, cell) in row.iter().enumerate() {
            if let Ok(look) = cell {
                array[[sat_index, station_index]] = value(look);
            }
        }
    }
    array.into_pyarray(py)
}

#[pymethods]
impl PyCoverageGrid {
    /// Number of satellites (grid rows).
    #[getter]
    fn n_satellites(&self) -> usize {
        self.n_satellites
    }

    /// Number of stations (grid columns).
    #[getter]
    fn n_stations(&self) -> usize {
        self.n_stations
    }

    /// Azimuth per pair, numpy `(n_satellites, n_stations)` degrees, `NaN` where
    /// the look angle could not be evaluated.
    fn azimuth_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        cell_array(py, &self.grid, (self.n_satellites, self.n_stations), |l| {
            l.azimuth_deg
        })
    }

    /// Elevation per pair, numpy `(n_satellites, n_stations)` degrees, `NaN`
    /// where the look angle could not be evaluated.
    fn elevation_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        cell_array(py, &self.grid, (self.n_satellites, self.n_stations), |l| {
            l.elevation_deg
        })
    }

    /// Slant range per pair, numpy `(n_satellites, n_stations)` kilometres,
    /// `NaN` where the look angle could not be evaluated.
    fn range_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        cell_array(py, &self.grid, (self.n_satellites, self.n_stations), |l| {
            l.range_km
        })
    }

    /// Complete typed error payload for a failed cell, or `None` for a
    /// successful cell or out-of-range index. The existing NaN arrays remain
    /// unchanged; this accessor retains the original core error variants.
    fn cell_error<'py>(
        &self,
        py: Python<'py>,
        satellite_index: usize,
        station_index: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let Some(Err(error)) = self
            .grid
            .get(satellite_index)
            .and_then(|row| row.get(station_index))
        else {
            return Ok(None);
        };
        look_angle_error_value(py, error).map(Some)
    }

    /// Boolean `[satellite][station]` visibility at or above `min_elevation_deg`.
    /// Error cells are not visible.
    fn visible_mask(&self, min_elevation_deg: f64) -> Vec<Vec<bool>> {
        core_visible_mask(&self.grid, min_elevation_deg)
    }

    /// Visible-satellite count per station at or above `min_elevation_deg`.
    fn access_counts(&self, min_elevation_deg: f64) -> Vec<usize> {
        core_access_counts(&self.grid, min_elevation_deg)
    }

    /// Maximum successful elevation per station, degrees, or `None` for a station
    /// with no successful look angle.
    fn max_elevation(&self) -> Vec<Option<f64>> {
        core_max_elevation(&self.grid)
    }

    fn __repr__(&self) -> String {
        format!(
            "CoverageGrid(n_satellites={}, n_stations={})",
            self.n_satellites, self.n_stations
        )
    }
}

fn error_node<'py>(
    py: Python<'py>,
    kind: &str,
    fields: impl FnOnce(&Bound<'py, PyDict>) -> PyResult<()>,
) -> PyResult<Bound<'py, PyDict>> {
    let node = PyDict::new(py);
    node.set_item("kind", kind)?;
    let payload = PyDict::new(py);
    fields(&payload)?;
    node.set_item("fields", payload)?;
    Ok(node)
}

pub(crate) fn sgp4_error_value<'py>(
    py: Python<'py>,
    error: &sidereon::sgp4::Error,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon::sgp4::Error as E;
    match error {
        E::InvalidInput { field, kind } => error_node(py, "invalid_input", |fields| {
            fields.set_item("field", field)?;
            fields.set_item("kind", sgp4_input_kind(*kind))
        }),
        E::NonFiniteOutput { field } => error_node(py, "non_finite_output", |fields| {
            fields.set_item("field", field)
        }),
        E::InvalidTle(message) => error_node(py, "invalid_tle", |fields| {
            fields.set_item("message", message)
        }),
        E::Sgp4 { code } => error_node(py, "sgp4", |fields| fields.set_item("code", code)),
        E::ResonanceStepBudget { budget } => error_node(py, "resonance_step_budget", |fields| {
            fields.set_item("budget", budget)
        }),
    }
}

fn sgp4_input_kind(kind: sidereon::sgp4::Sgp4InputErrorKind) -> &'static str {
    use sidereon::sgp4::Sgp4InputErrorKind as K;
    match kind {
        K::NonFinite => "non_finite",
        K::NotPositive => "not_positive",
        K::Negative => "negative",
        K::OutOfRange => "out_of_range",
        K::Missing => "missing",
        K::FloatParse => "float_parse",
        K::IntParse => "int_parse",
        K::InvalidCivilDate => "invalid_civil_date",
        K::InvalidCivilTime => "invalid_civil_time",
    }
}

pub(crate) fn look_angle_error_value<'py>(
    py: Python<'py>,
    error: &sidereon_core::astro::passes::LookAngleError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::astro::passes::LookAngleError as E;
    match error {
        E::InvalidInput { field, reason } => error_node(py, "invalid_input", |fields| {
            fields.set_item("field", field)?;
            fields.set_item("reason", reason)
        }),
        E::Init(source) => error_node(py, "init", |fields| {
            fields.set_item("cause", sgp4_error_value(py, source)?)
        }),
        E::Propagate(source) => error_node(py, "propagate", |fields| {
            fields.set_item("cause", sgp4_error_value(py, source)?)
        }),
        E::FrameTransform(source) => {
            use sidereon_core::astro::frames::transforms::FrameTransformError as F;
            error_node(py, "frame_transform", |fields| {
                let cause = match source {
                    F::InvalidInput { field, reason } => {
                        error_node(py, "invalid_input", |details| {
                            details.set_item("field", field)?;
                            details.set_item("reason", reason)
                        })?
                    }
                    F::Ut1OutsideCoverage { reason } => {
                        error_node(py, "ut1_outside_coverage", |details| {
                            details.set_item(
                                "reason",
                                match reason {
                                    sidereon_core::astro::time::DegradeReason::BeforeCoverage => {
                                        "before_coverage"
                                    }
                                    sidereon_core::astro::time::DegradeReason::AfterCoverage => {
                                        "after_coverage"
                                    }
                                },
                            )
                        })?
                    }
                };
                fields.set_item("cause", cause)
            })
        }
    }
}

/// Compute the `[satellite][station]` look-angle grid at one epoch.
///
/// `tles` is a sequence of [`Tle`](crate::propagation::PyTle); `stations` a
/// sequence of [`GroundStation`](crate::propagation::PyGroundStation);
/// `epoch_unix_us` the UTC unix-microsecond epoch. Returns a [`CoverageGrid`].
#[pyfunction]
#[pyo3(signature = (tles, stations, epoch_unix_us))]
fn coverage_look_angles(
    tles: Vec<PyRef<'_, PyTle>>,
    stations: Vec<PyRef<'_, PyGroundStation>>,
    epoch_unix_us: i64,
) -> PyResult<PyCoverageGrid> {
    let satellites: Vec<Satellite> = tles.iter().map(|tle| tle.satellite().clone()).collect();
    let core_stations: Vec<_> = stations.iter().map(|station| station.core()).collect();
    let datetime = UtcInstant::from_unix_microseconds(epoch_unix_us);
    let grid = core_look_angles_batch(&satellites, &core_stations, datetime);
    Ok(PyCoverageGrid {
        grid,
        n_satellites: satellites.len(),
        n_stations: core_stations.len(),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCoverageGrid>()?;
    m.add_function(wrap_pyfunction!(coverage_look_angles, m)?)?;
    Ok(())
}
