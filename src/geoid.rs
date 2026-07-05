//! Geoid undulation lookup and orthometric/ellipsoidal height conversion.
//!
//! Thin INTERFACE over `sidereon_core::geoid`. It marshals the grid origin,
//! spacing, dimensions, and samples (or a grid text blob) into the core
//! [`GeoidGrid`](sidereon_core::geoid::GeoidGrid) and exposes the bilinear
//! undulation query, plus the zero-setup built-in-grid helpers
//! [`geoid_undulation`](sidereon_core::geoid::geoid_undulation) /
//! [`orthometric_height_m`](sidereon_core::geoid::orthometric_height_m) /
//! [`ellipsoidal_height_m`](sidereon_core::geoid::ellipsoidal_height_m). All
//! interpolation lives in the core; no geoid logic lives here.

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::geoid::{
    egm96_ellipsoidal_height_m as core_egm96_ellipsoidal_height_m,
    egm96_orthometric_height_m as core_egm96_orthometric_height_m,
    egm96_undulation as core_egm96_undulation, egm96_undulations_deg as core_egm96_undulations_deg,
    egm96_undulations_rad as core_egm96_undulations_rad,
    ellipsoidal_height_m as core_ellipsoidal_height_m, geoid_undulation as core_geoid_undulation,
    geoid_undulations_deg as core_geoid_undulations_deg,
    geoid_undulations_rad as core_geoid_undulations_rad,
    orthometric_height_m as core_orthometric_height_m, Egm2008GridSpacing, Egm2008RasterWindow,
    GeoidGrid,
};

fn to_geoid_err<E: std::fmt::Debug>(err: E) -> PyErr {
    PyValueError::new_err(format!("{err:?}"))
}

fn points2_from_array(name: &str, points: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<(f64, f64)>> {
    let view = points.as_array();
    if view.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 2), got (_, {})",
            view.ncols()
        )));
    }
    let mut out = Vec::with_capacity(view.nrows());
    for (row_index, row) in view.outer_iter().enumerate() {
        let value = (row[0], row[1]);
        if !value.0.is_finite() || !value.1.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name}[{row_index}] must contain only finite values"
            )));
        }
        out.push(value);
    }
    Ok(out)
}

#[pyclass(module = "sidereon._sidereon", name = "Egm2008GridSpacing", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyEgm2008GridSpacing {
    /// The 1-arcminute EGM2008 grid.
    ONE_MINUTE,
    /// The 2.5-arcminute EGM2008 grid.
    TWO_POINT_FIVE_MINUTE,
}

impl From<PyEgm2008GridSpacing> for Egm2008GridSpacing {
    fn from(value: PyEgm2008GridSpacing) -> Self {
        match value {
            PyEgm2008GridSpacing::ONE_MINUTE => Self::OneMinute,
            PyEgm2008GridSpacing::TWO_POINT_FIVE_MINUTE => Self::TwoPointFiveMinute,
        }
    }
}

impl From<Egm2008GridSpacing> for PyEgm2008GridSpacing {
    fn from(value: Egm2008GridSpacing) -> Self {
        match value {
            Egm2008GridSpacing::OneMinute => Self::ONE_MINUTE,
            Egm2008GridSpacing::TwoPointFiveMinute => Self::TWO_POINT_FIVE_MINUTE,
        }
    }
}

#[pymethods]
impl PyEgm2008GridSpacing {
    /// Grid spacing in arcminutes.
    #[getter]
    fn arc_minutes(&self) -> f64 {
        Egm2008GridSpacing::from(*self).arc_minutes()
    }

    /// Grid spacing in degrees.
    #[getter]
    fn degrees(&self) -> f64 {
        Egm2008GridSpacing::from(*self).degrees()
    }

    /// Official global row and column counts.
    #[getter]
    fn global_dimensions(&self) -> (usize, usize) {
        Egm2008GridSpacing::from(*self).global_dimensions()
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::ONE_MINUTE => "Egm2008GridSpacing.ONE_MINUTE",
            Self::TWO_POINT_FIVE_MINUTE => "Egm2008GridSpacing.TWO_POINT_FIVE_MINUTE",
        }
    }
}

/// Descriptor for a full or cropped EGM2008 row-framed raster window.
#[pyclass(module = "sidereon._sidereon", name = "Egm2008RasterWindow")]
#[derive(Clone, Copy)]
pub struct PyEgm2008RasterWindow {
    inner: Egm2008RasterWindow,
}

impl PyEgm2008RasterWindow {
    fn inner(&self) -> Egm2008RasterWindow {
        self.inner
    }
}

#[pymethods]
impl PyEgm2008RasterWindow {
    /// Build an EGM2008 raster-window descriptor.
    #[new]
    fn new(
        spacing: PyEgm2008GridSpacing,
        lat_min_deg: f64,
        lon_min_deg: f64,
        n_lat: usize,
        n_lon: usize,
    ) -> PyResult<Self> {
        let inner =
            Egm2008RasterWindow::new(spacing.into(), lat_min_deg, lon_min_deg, n_lat, n_lon)
                .map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Build the official full-global EGM2008 window for a spacing.
    #[staticmethod]
    fn global_window(spacing: PyEgm2008GridSpacing) -> Self {
        Self {
            inner: Egm2008RasterWindow::global(spacing.into()),
        }
    }

    /// Raster spacing for this window.
    #[getter]
    fn spacing(&self) -> PyEgm2008GridSpacing {
        self.inner.spacing().into()
    }

    /// Southwest latitude of this window in degrees.
    #[getter]
    fn lat_min_deg(&self) -> f64 {
        self.inner.lat_min_deg()
    }

    /// Western longitude of this window in degrees.
    #[getter]
    fn lon_min_deg(&self) -> f64 {
        self.inner.lon_min_deg()
    }

    /// Latitude node count in this window.
    #[getter]
    fn n_lat(&self) -> usize {
        self.inner.n_lat()
    }

    /// Longitude node count in this window.
    #[getter]
    fn n_lon(&self) -> usize {
        self.inner.n_lon()
    }

    fn __repr__(&self) -> String {
        let spacing = match self.inner.spacing() {
            Egm2008GridSpacing::OneMinute => "ONE_MINUTE",
            Egm2008GridSpacing::TwoPointFiveMinute => "TWO_POINT_FIVE_MINUTE",
        };
        format!(
            "Egm2008RasterWindow(spacing={}, lat_min_deg={}, lon_min_deg={}, n_lat={}, n_lon={})",
            spacing,
            self.inner.lat_min_deg(),
            self.inner.lon_min_deg(),
            self.inner.n_lat(),
            self.inner.n_lon()
        )
    }
}

/// A regular latitude/longitude grid of geoid undulation samples (metres) with
/// bilinear interpolation. Construct from explicit arrays or via
/// [`GeoidGrid.from_text`]; wraps [`sidereon_core::geoid::GeoidGrid`] unchanged.
#[pyclass(module = "sidereon._sidereon", name = "GeoidGrid")]
pub struct PyGeoidGrid {
    inner: GeoidGrid,
}

#[pymethods]
impl PyGeoidGrid {
    /// Build a geoid grid from its origin, spacing, dimensions, and row-major
    /// samples (metres). `values_m` must have exactly `n_lat * n_lon` entries in
    /// latitude-ascending-outer, longitude-ascending-inner order. Raises
    /// `ValueError` on a zero dimension, a sample-count mismatch, a non-finite or
    /// non-positive spacing, or a non-finite sample.
    #[new]
    #[pyo3(signature = (lat_min_deg, lon_min_deg, dlat_deg, dlon_deg, n_lat, n_lon, values_m))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lat_min_deg: f64,
        lon_min_deg: f64,
        dlat_deg: f64,
        dlon_deg: f64,
        n_lat: usize,
        n_lon: usize,
        values_m: Vec<f64>,
    ) -> PyResult<Self> {
        let inner = GeoidGrid::new(
            lat_min_deg,
            lon_min_deg,
            dlat_deg,
            dlon_deg,
            n_lat,
            n_lon,
            values_m,
        )
        .map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Parse a geoid grid from the documented whitespace-delimited text format:
    /// a six-field header `lat_min lon_min dlat dlon n_lat n_lon` followed by
    /// `n_lat * n_lon` undulation samples in metres. Raises `ValueError` on a
    /// malformed grid.
    #[staticmethod]
    fn from_text(text: &str) -> PyResult<Self> {
        let inner = GeoidGrid::from_text(text).map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Parse the NGA EGM96 `WW15MGH.DAC` binary grid from bytes.
    #[staticmethod]
    fn from_egm96_dac(data: &[u8]) -> PyResult<Self> {
        let inner = GeoidGrid::from_egm96_dac(data).map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Parse an official full-global NGA EGM2008 interpolation raster.
    #[staticmethod]
    fn from_egm2008_raster(data: &[u8], spacing: PyEgm2008GridSpacing) -> PyResult<Self> {
        let inner = GeoidGrid::from_egm2008_raster(data, spacing.into()).map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Parse a full or cropped NGA EGM2008 interpolation raster window.
    #[staticmethod]
    fn from_egm2008_raster_window(data: &[u8], window: &PyEgm2008RasterWindow) -> PyResult<Self> {
        let inner =
            GeoidGrid::from_egm2008_raster_window(data, window.inner()).map_err(to_geoid_err)?;
        Ok(Self { inner })
    }

    /// Bilinearly interpolated undulation `N` (metres) at a geodetic position in
    /// degrees (latitude positive north, longitude positive east).
    fn undulation_deg(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        self.inner.undulation_deg(lat_deg, lon_deg)
    }

    /// Bilinearly interpolated undulation `N` (metres) at a geodetic position in
    /// radians (latitude positive north, longitude positive east).
    fn undulation_rad(&self, lat_rad: f64, lon_rad: f64) -> f64 {
        self.inner.undulation_rad(lat_rad, lon_rad)
    }

    /// Batch bilinear undulations for `(lat_deg, lon_deg)` rows.
    fn undulations_deg<'py>(
        &self,
        py: Python<'py>,
        points_deg: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points = points2_from_array("points_deg", &points_deg)?;
        Ok(PyArray1::from_vec(py, self.inner.undulations_deg(&points)))
    }

    /// Batch bilinear undulations for `(lat_rad, lon_rad)` rows.
    fn undulations_rad<'py>(
        &self,
        py: Python<'py>,
        points_rad: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points = points2_from_array("points_rad", &points_rad)?;
        Ok(PyArray1::from_vec(py, self.inner.undulations_rad(&points)))
    }

    /// Orthometric height using this grid and degree geodetic input.
    fn orthometric_height_deg(&self, ellipsoidal_height_m: f64, lat_deg: f64, lon_deg: f64) -> f64 {
        self.inner
            .orthometric_height_deg(ellipsoidal_height_m, lat_deg, lon_deg)
    }

    /// Ellipsoidal height using this grid and degree geodetic input.
    fn ellipsoidal_height_deg(&self, orthometric_height_m: f64, lat_deg: f64, lon_deg: f64) -> f64 {
        self.inner
            .ellipsoidal_height_deg(orthometric_height_m, lat_deg, lon_deg)
    }

    /// Orthometric height using this grid and radian geodetic input.
    fn orthometric_height_rad(&self, ellipsoidal_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
        self.inner
            .orthometric_height_rad(ellipsoidal_height_m, lat_rad, lon_rad)
    }

    /// Ellipsoidal height using this grid and radian geodetic input.
    fn ellipsoidal_height_rad(&self, orthometric_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
        self.inner
            .ellipsoidal_height_rad(orthometric_height_m, lat_rad, lon_rad)
    }

    fn __repr__(&self) -> String {
        "GeoidGrid(...)".to_string()
    }
}

/// Geoid undulation `N` (metres above the WGS84 ellipsoid) at a geodetic
/// position in radians, from the coarse built-in global grid.
#[pyfunction]
#[pyo3(signature = (lat_rad, lon_rad))]
fn geoid_undulation(lat_rad: f64, lon_rad: f64) -> f64 {
    core_geoid_undulation(lat_rad, lon_rad)
}

/// Batch geoid undulations for `(lat_rad, lon_rad)` rows from the built-in grid.
#[pyfunction]
fn geoid_undulations_rad<'py>(
    py: Python<'py>,
    points_rad: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let points = points2_from_array("points_rad", &points_rad)?;
    Ok(PyArray1::from_vec(py, core_geoid_undulations_rad(&points)))
}

/// Batch geoid undulations for `(lat_deg, lon_deg)` rows from the built-in grid.
#[pyfunction]
fn geoid_undulations_deg<'py>(
    py: Python<'py>,
    points_deg: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let points = points2_from_array("points_deg", &points_deg)?;
    Ok(PyArray1::from_vec(py, core_geoid_undulations_deg(&points)))
}

/// Orthometric height `H = h - N` (metres above mean sea level) from an
/// ellipsoidal height and a geodetic position in radians, using the built-in
/// grid's undulation.
#[pyfunction]
#[pyo3(signature = (ellipsoidal_height_m, lat_rad, lon_rad))]
fn orthometric_height_m(ellipsoidal_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    core_orthometric_height_m(ellipsoidal_height_m, lat_rad, lon_rad)
}

/// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
/// orthometric height and a geodetic position in radians, using the built-in
/// grid's undulation.
#[pyfunction]
#[pyo3(signature = (orthometric_height_m, lat_rad, lon_rad))]
fn ellipsoidal_height_m(orthometric_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    core_ellipsoidal_height_m(orthometric_height_m, lat_rad, lon_rad)
}

/// Geoid undulation `N` (metres above the WGS84 ellipsoid) at a geodetic
/// position in radians, from the embedded GENUINE EGM96 1-degree global grid.
///
/// Latitude positive north, longitude positive east, both radians. This is the
/// recommended zero-setup default for metre-class datum work: the bilinear lookup
/// tracks the full 15-arcminute EGM96 grid to ~0.4 m RMS. The coarse 30-degree
/// [`geoid_undulation`] is only a sanity-check fallback.
#[pyfunction]
#[pyo3(signature = (lat_rad, lon_rad))]
fn egm96_undulation(lat_rad: f64, lon_rad: f64) -> f64 {
    core_egm96_undulation(lat_rad, lon_rad)
}

/// Batch EGM96 undulations for `(lat_rad, lon_rad)` rows.
#[pyfunction]
fn egm96_undulations_rad<'py>(
    py: Python<'py>,
    points_rad: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let points = points2_from_array("points_rad", &points_rad)?;
    Ok(PyArray1::from_vec(py, core_egm96_undulations_rad(&points)))
}

/// Batch EGM96 undulations for `(lat_deg, lon_deg)` rows.
#[pyfunction]
fn egm96_undulations_deg<'py>(
    py: Python<'py>,
    points_deg: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let points = points2_from_array("points_deg", &points_deg)?;
    Ok(PyArray1::from_vec(py, core_egm96_undulations_deg(&points)))
}

/// Orthometric height `H = h - N` (metres above mean sea level) from an
/// ellipsoidal height and a geodetic position in radians, using the embedded
/// genuine EGM96 1-degree model.
#[pyfunction]
#[pyo3(signature = (ellipsoidal_height_m, lat_rad, lon_rad))]
fn egm96_orthometric_height_m(ellipsoidal_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    core_egm96_orthometric_height_m(ellipsoidal_height_m, lat_rad, lon_rad)
}

/// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
/// orthometric height and a geodetic position in radians, using the embedded
/// genuine EGM96 1-degree model.
#[pyfunction]
#[pyo3(signature = (orthometric_height_m, lat_rad, lon_rad))]
fn egm96_ellipsoidal_height_m(orthometric_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    core_egm96_ellipsoidal_height_m(orthometric_height_m, lat_rad, lon_rad)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEgm2008GridSpacing>()?;
    m.add_class::<PyEgm2008RasterWindow>()?;
    m.add_class::<PyGeoidGrid>()?;
    m.add_function(wrap_pyfunction!(geoid_undulation, m)?)?;
    m.add_function(wrap_pyfunction!(geoid_undulations_rad, m)?)?;
    m.add_function(wrap_pyfunction!(geoid_undulations_deg, m)?)?;
    m.add_function(wrap_pyfunction!(orthometric_height_m, m)?)?;
    m.add_function(wrap_pyfunction!(ellipsoidal_height_m, m)?)?;
    m.add_function(wrap_pyfunction!(egm96_undulation, m)?)?;
    m.add_function(wrap_pyfunction!(egm96_undulations_rad, m)?)?;
    m.add_function(wrap_pyfunction!(egm96_undulations_deg, m)?)?;
    m.add_function(wrap_pyfunction!(egm96_orthometric_height_m, m)?)?;
    m.add_function(wrap_pyfunction!(egm96_ellipsoidal_height_m, m)?)?;
    Ok(())
}
