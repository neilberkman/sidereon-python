//! DTED terrain binding.

use std::path::PathBuf;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::terrain::{DtedInterpolation, DtedLookupOptions, DtedTerrain, DtedTile};

use crate::np_array;

fn to_terrain_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyclass(module = "sidereon._sidereon", name = "DtedInterpolation", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// DTED lookup interpolation mode.
pub enum PyDtedInterpolation {
    NEAREST_POSTING,
    BILINEAR,
}

impl From<PyDtedInterpolation> for DtedInterpolation {
    fn from(value: PyDtedInterpolation) -> Self {
        match value {
            PyDtedInterpolation::NEAREST_POSTING => Self::NearestPosting,
            PyDtedInterpolation::BILINEAR => Self::Bilinear,
        }
    }
}

impl From<DtedInterpolation> for PyDtedInterpolation {
    fn from(value: DtedInterpolation) -> Self {
        match value {
            DtedInterpolation::NearestPosting => Self::NEAREST_POSTING,
            DtedInterpolation::Bilinear => Self::BILINEAR,
        }
    }
}

#[pymethods]
impl PyDtedInterpolation {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::NEAREST_POSTING => "nearest_posting",
            Self::BILINEAR => "bilinear",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::NEAREST_POSTING => "DtedInterpolation.NEAREST_POSTING",
            Self::BILINEAR => "DtedInterpolation.BILINEAR",
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "DtedLookupOptions")]
#[derive(Clone, Copy)]
/// Options for DTED height lookup.
pub struct PyDtedLookupOptions {
    inner: DtedLookupOptions,
}

impl PyDtedLookupOptions {
    pub(crate) fn inner(&self) -> DtedLookupOptions {
        self.inner
    }
}

#[pymethods]
impl PyDtedLookupOptions {
    /// Build DTED lookup options.
    #[new]
    #[pyo3(signature = (interpolation=PyDtedInterpolation::BILINEAR))]
    fn new(interpolation: PyDtedInterpolation) -> Self {
        Self {
            inner: DtedLookupOptions {
                interpolation: interpolation.into(),
            },
        }
    }

    #[getter]
    fn interpolation(&self) -> PyDtedInterpolation {
        self.inner.interpolation.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "DtedLookupOptions(interpolation={})",
            self.interpolation().label()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "DtedTerrain")]
/// Lazy DTED terrain reader rooted at a directory of cached tiles.
pub struct PyDtedTerrain {
    inner: DtedTerrain,
}

#[pymethods]
impl PyDtedTerrain {
    /// Build a DTED terrain reader from a path.
    #[new]
    fn new(root: PathBuf) -> Self {
        Self {
            inner: DtedTerrain::new(root),
        }
    }

    /// Return ORTHOMETRIC terrain height in metres at latitude and longitude in degrees.
    ///
    /// Missing tiles use the core sea-level fallback.
    #[pyo3(signature = (latitude_deg, longitude_deg, options=None))]
    fn height_m(
        &mut self,
        latitude_deg: f64,
        longitude_deg: f64,
        options: Option<&PyDtedLookupOptions>,
    ) -> PyResult<f64> {
        match options {
            Some(options) => {
                self.inner
                    .height_m_with_options(longitude_deg, latitude_deg, options.inner())
            }
            None => self.inner.height_m(longitude_deg, latitude_deg),
        }
        .map_err(to_terrain_err)
    }

    /// Return ORTHOMETRIC terrain heights in metres for `(longitude, latitude)` rows.
    ///
    /// `points_lon_lat_deg` is a numpy `(n, 2)` array with longitude in column 0
    /// and latitude in column 1, both in degrees. Missing tiles use the core
    /// sea-level fallback. If any point is invalid, raises `ValueError` with the
    /// failing row index.
    #[pyo3(signature = (points_lon_lat_deg, options=None))]
    fn height_batch<'py>(
        &mut self,
        py: Python<'py>,
        points_lon_lat_deg: PyReadonlyArray2<'_, f64>,
        options: Option<&PyDtedLookupOptions>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let view = points_lon_lat_deg.as_array();
        if view.ncols() != 2 {
            return Err(PyValueError::new_err(
                "points_lon_lat_deg must have two columns",
            ));
        }
        let points = view
            .outer_iter()
            .map(|row| (row[0], row[1]))
            .collect::<Vec<_>>();
        let options = options.map(PyDtedLookupOptions::inner).unwrap_or_default();
        let mut heights = Vec::with_capacity(points.len());
        for (index, result) in self
            .inner
            .height_batch(&points, options)
            .into_iter()
            .enumerate()
        {
            heights.push(
                result.map_err(|err| {
                    PyValueError::new_err(format!("terrain point {index}: {err}"))
                })?,
            );
        }
        Ok(np_array(py, &heights))
    }

    fn __repr__(&self) -> &'static str {
        "DtedTerrain()"
    }
}

#[pyclass(module = "sidereon._sidereon", name = "DtedTile")]
/// Parsed single DTED tile.
pub struct PyDtedTile {
    inner: DtedTile,
}

#[pymethods]
impl PyDtedTile {
    /// Read a DTED tile from a path.
    #[staticmethod]
    fn from_path(path: PathBuf) -> PyResult<Self> {
        DtedTile::from_path(path)
            .map(|inner| Self { inner })
            .map_err(to_terrain_err)
    }

    /// Return nearest-posting ORTHOMETRIC height in metres at latitude and longitude in degrees.
    fn height_m(&self, latitude_deg: f64, longitude_deg: f64) -> PyResult<f64> {
        self.inner
            .get_elevation(longitude_deg, latitude_deg)
            .map(f64::from)
            .map_err(to_terrain_err)
    }

    fn __repr__(&self) -> &'static str {
        "DtedTile()"
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtedInterpolation>()?;
    m.add_class::<PyDtedLookupOptions>()?;
    m.add_class::<PyDtedTerrain>()?;
    m.add_class::<PyDtedTile>()?;
    Ok(())
}
