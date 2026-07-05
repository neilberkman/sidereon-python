//! Memory-mappable terrain store binding.

use std::path::PathBuf;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use sidereon_core::terrain::DtedLookupOptions;
use sidereon_core::terrain_store::{
    dted_tree_to_mmap_store as core_dted_tree_to_mmap_store,
    terrain_store_checksum64 as core_terrain_store_checksum64,
    write_dted_tree_to_mmap_store as core_write_dted_tree_to_mmap_store, Egm96FifteenMinuteGeoid,
    EllipsoidalHeightM, MmapTerrain, OrthometricHeightM, TerrainDatumError, TerrainGeoidModel,
    TerrainStoreError, TerrainStoreTileIndex, VerticalDatum,
};

use crate::np_array;
use crate::terrain::PyDtedLookupOptions;

fn points_lon_lat(name: &str, points: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<(f64, f64)>> {
    let view = points.as_array();
    if view.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have two columns, got {}",
            view.ncols()
        )));
    }
    Ok(view.outer_iter().map(|row| (row[0], row[1])).collect())
}

fn store_error_text(err: TerrainStoreError) -> String {
    let typed = PyTerrainStoreError::from(err);
    typed.error_text()
}

fn to_store_err(err: TerrainStoreError) -> PyErr {
    PyValueError::new_err(store_error_text(err))
}

fn to_lookup_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn datum_error_text(err: TerrainDatumError) -> String {
    let typed = PyTerrainDatumError::from(err);
    typed.error_text()
}

fn to_datum_err(err: TerrainDatumError) -> PyErr {
    PyValueError::new_err(datum_error_text(err))
}

fn options_or_default(options: Option<&PyDtedLookupOptions>) -> DtedLookupOptions {
    options.map(PyDtedLookupOptions::inner).unwrap_or_default()
}

/// Vertical datum carried by terrain store tile index records.
#[pyclass(module = "sidereon._sidereon", name = "VerticalDatum", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyVerticalDatum {
    /// Orthometric height above the EGM96 mean sea level geoid.
    EGM96_MSL_ORTHOMETRIC,
}

impl From<VerticalDatum> for PyVerticalDatum {
    fn from(value: VerticalDatum) -> Self {
        match value {
            VerticalDatum::Egm96MslOrthometric => Self::EGM96_MSL_ORTHOMETRIC,
        }
    }
}

impl From<PyVerticalDatum> for VerticalDatum {
    fn from(value: PyVerticalDatum) -> Self {
        match value {
            PyVerticalDatum::EGM96_MSL_ORTHOMETRIC => Self::Egm96MslOrthometric,
        }
    }
}

#[pymethods]
impl PyVerticalDatum {
    /// Lowercase label for the vertical datum.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::EGM96_MSL_ORTHOMETRIC => "egm96_msl_orthometric",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::EGM96_MSL_ORTHOMETRIC => "VerticalDatum.EGM96_MSL_ORTHOMETRIC",
        }
    }
}

/// Orthometric height `H` in metres above the EGM96 mean sea level geoid.
#[pyclass(module = "sidereon._sidereon", name = "OrthometricHeightM")]
#[derive(Clone)]
pub struct PyOrthometricHeightM {
    inner: OrthometricHeightM,
}

impl From<OrthometricHeightM> for PyOrthometricHeightM {
    fn from(inner: OrthometricHeightM) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOrthometricHeightM {
    /// Build an orthometric terrain height `H` in metres.
    #[new]
    fn new(value_m: f64) -> Self {
        Self {
            inner: OrthometricHeightM::new(value_m),
        }
    }

    /// Orthometric height `H` in metres.
    #[getter]
    fn value_m(&self) -> f64 {
        self.inner.value_m
    }

    /// Return the orthometric height `H` in metres.
    fn metres(&self) -> f64 {
        self.inner.metres()
    }

    /// Convert this orthometric height to ellipsoidal height in metres.
    ///
    /// Inputs are geodetic `(latitude_deg, longitude_deg)`. The geoid model is
    /// explicit, so the 15-arcminute EGM96 tier never falls back silently.
    fn to_ellipsoidal_height_deg(
        &self,
        py: Python<'_>,
        latitude_deg: f64,
        longitude_deg: f64,
        geoid_model: &PyTerrainGeoidModel,
    ) -> PyResult<PyEllipsoidalHeightM> {
        match &geoid_model.kind {
            PyTerrainGeoidModelKind::Egm96OneDegree => self
                .inner
                .to_ellipsoidal_height_deg(
                    latitude_deg,
                    longitude_deg,
                    TerrainGeoidModel::Egm96OneDegree,
                )
                .map(Into::into)
                .map_err(to_datum_err),
            PyTerrainGeoidModelKind::Egm96FifteenMinute(grid) => {
                let grid = grid.borrow(py);
                self.inner
                    .to_ellipsoidal_height_deg(
                        latitude_deg,
                        longitude_deg,
                        TerrainGeoidModel::Egm96FifteenMinute(&grid.inner),
                    )
                    .map(Into::into)
                    .map_err(to_datum_err)
            }
        }
    }

    /// Convert this orthometric height to ellipsoidal height in metres.
    ///
    /// Inputs are geodetic `(latitude_rad, longitude_rad)`. The geoid model is
    /// explicit, so the 15-arcminute EGM96 tier never falls back silently.
    fn to_ellipsoidal_height_rad(
        &self,
        py: Python<'_>,
        latitude_rad: f64,
        longitude_rad: f64,
        geoid_model: &PyTerrainGeoidModel,
    ) -> PyResult<PyEllipsoidalHeightM> {
        match &geoid_model.kind {
            PyTerrainGeoidModelKind::Egm96OneDegree => self
                .inner
                .to_ellipsoidal_height_rad(
                    latitude_rad,
                    longitude_rad,
                    TerrainGeoidModel::Egm96OneDegree,
                )
                .map(Into::into)
                .map_err(to_datum_err),
            PyTerrainGeoidModelKind::Egm96FifteenMinute(grid) => {
                let grid = grid.borrow(py);
                self.inner
                    .to_ellipsoidal_height_rad(
                        latitude_rad,
                        longitude_rad,
                        TerrainGeoidModel::Egm96FifteenMinute(&grid.inner),
                    )
                    .map(Into::into)
                    .map_err(to_datum_err)
            }
        }
    }

    fn __repr__(&self) -> String {
        format!("OrthometricHeightM(value_m={})", self.inner.value_m)
    }
}

/// Ellipsoidal height `h` in metres above the WGS84 reference ellipsoid.
#[pyclass(module = "sidereon._sidereon", name = "EllipsoidalHeightM")]
#[derive(Clone)]
pub struct PyEllipsoidalHeightM {
    inner: EllipsoidalHeightM,
}

impl From<EllipsoidalHeightM> for PyEllipsoidalHeightM {
    fn from(inner: EllipsoidalHeightM) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyEllipsoidalHeightM {
    /// Build an ellipsoidal height `h` in metres.
    #[new]
    fn new(value_m: f64) -> Self {
        Self {
            inner: EllipsoidalHeightM::new(value_m),
        }
    }

    /// Ellipsoidal height `h` in metres.
    #[getter]
    fn value_m(&self) -> f64 {
        self.inner.value_m
    }

    /// Return the ellipsoidal height `h` in metres.
    fn metres(&self) -> f64 {
        self.inner.metres()
    }

    fn __repr__(&self) -> String {
        format!("EllipsoidalHeightM(value_m={})", self.inner.value_m)
    }
}

/// Metadata for one tile index record in a memory-mappable terrain store.
#[pyclass(module = "sidereon._sidereon", name = "TerrainStoreTileIndex")]
#[derive(Clone, Copy)]
pub struct PyTerrainStoreTileIndex {
    inner: TerrainStoreTileIndex,
}

impl From<TerrainStoreTileIndex> for PyTerrainStoreTileIndex {
    fn from(inner: TerrainStoreTileIndex) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTerrainStoreTileIndex {
    /// Integer latitude tile id, e.g. `36` for `36..37` degrees.
    #[getter]
    fn lat_index(&self) -> i32 {
        self.inner.lat_index
    }

    /// Integer longitude tile id, e.g. `-107` for `-107..-106` degrees.
    #[getter]
    fn lon_index(&self) -> i32 {
        self.inner.lon_index
    }

    /// Western edge longitude in degrees.
    #[getter]
    fn min_longitude_deg(&self) -> f64 {
        self.inner.min_longitude_deg
    }

    /// Southern edge latitude in degrees.
    #[getter]
    fn min_latitude_deg(&self) -> f64 {
        self.inner.min_latitude_deg
    }

    /// Eastern edge longitude in degrees.
    #[getter]
    fn max_longitude_deg(&self) -> f64 {
        self.inner.max_longitude_deg
    }

    /// Northern edge latitude in degrees.
    #[getter]
    fn max_latitude_deg(&self) -> f64 {
        self.inner.max_latitude_deg
    }

    /// Number of longitude postings.
    #[getter]
    fn lon_count(&self) -> u32 {
        self.inner.lon_count
    }

    /// Number of latitude postings.
    #[getter]
    fn lat_count(&self) -> u32 {
        self.inner.lat_count
    }

    /// Byte offset of this tile's posting payload in the store.
    #[getter]
    fn data_offset(&self) -> u64 {
        self.inner.data_offset
    }

    /// Byte length of this tile's posting payload in the store.
    #[getter]
    fn data_len(&self) -> u64 {
        self.inner.data_len
    }

    /// FNV-1a checksum of this tile's posting payload bytes.
    #[getter]
    fn checksum64(&self) -> u64 {
        self.inner.checksum64
    }

    /// Vertical datum for the tile's posting payload.
    #[getter]
    fn vertical_datum(&self) -> PyVerticalDatum {
        self.inner.vertical_datum.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainStoreTileIndex(lat_index={}, lon_index={})",
            self.inner.lat_index, self.inner.lon_index
        )
    }
}

/// Loaded EGM96 15-arcminute geoid grid for explicit terrain datum conversion.
#[pyclass(module = "sidereon._sidereon", name = "Egm96FifteenMinuteGeoid")]
pub struct PyEgm96FifteenMinuteGeoid {
    inner: Egm96FifteenMinuteGeoid,
}

#[pymethods]
impl PyEgm96FifteenMinuteGeoid {
    /// Load `WW15MGH.DAC` bytes as an EGM96 15-arcminute geoid grid.
    #[staticmethod]
    fn from_ww15mgh_dac_bytes(data: &[u8]) -> PyResult<Self> {
        Egm96FifteenMinuteGeoid::from_ww15mgh_dac_bytes(data)
            .map(|inner| Self { inner })
            .map_err(to_datum_err)
    }

    /// Read and load `WW15MGH.DAC` from disk.
    ///
    /// A missing file raises `ValueError` whose message starts with
    /// `MissingEgm96Dac`; it does not fall back to the embedded 1-degree grid.
    #[staticmethod]
    fn from_ww15mgh_dac_path(path: PathBuf) -> PyResult<Self> {
        Egm96FifteenMinuteGeoid::from_ww15mgh_dac_path(path)
            .map(|inner| Self { inner })
            .map_err(to_datum_err)
    }

    fn __repr__(&self) -> &'static str {
        "Egm96FifteenMinuteGeoid()"
    }
}

enum PyTerrainGeoidModelKind {
    Egm96OneDegree,
    Egm96FifteenMinute(Py<PyEgm96FifteenMinuteGeoid>),
}

/// Geoid tier used to convert terrain orthometric height `H` to ellipsoidal
/// height `h`.
#[pyclass(module = "sidereon._sidereon", name = "TerrainGeoidModel")]
pub struct PyTerrainGeoidModel {
    kind: PyTerrainGeoidModelKind,
}

#[pymethods]
impl PyTerrainGeoidModel {
    /// Embedded EGM96 1-degree grid, always available in-process.
    #[staticmethod]
    fn egm96_one_degree() -> Self {
        Self {
            kind: PyTerrainGeoidModelKind::Egm96OneDegree,
        }
    }

    /// Caller-supplied EGM96 15-arcminute `WW15MGH.DAC` grid.
    #[staticmethod]
    fn egm96_fifteen_minute(geoid: Py<PyEgm96FifteenMinuteGeoid>) -> Self {
        Self {
            kind: PyTerrainGeoidModelKind::Egm96FifteenMinute(geoid),
        }
    }

    /// Lowercase label for the geoid tier.
    #[getter]
    fn label(&self) -> &'static str {
        match &self.kind {
            PyTerrainGeoidModelKind::Egm96OneDegree => "egm96_one_degree",
            PyTerrainGeoidModelKind::Egm96FifteenMinute(_) => "egm96_fifteen_minute",
        }
    }

    fn __repr__(&self) -> String {
        format!("TerrainGeoidModel.{}()", self.label())
    }
}

/// Terrain store conversion, serialization, and parsing error details.
#[pyclass(module = "sidereon._sidereon", name = "TerrainStoreError")]
#[derive(Clone)]
pub struct PyTerrainStoreError {
    kind: String,
    message: String,
    path: Option<String>,
    remediation: Option<String>,
}

impl PyTerrainStoreError {
    fn error_text(&self) -> String {
        format!("{}: {}", self.kind, self.message)
    }
}

impl From<TerrainStoreError> for PyTerrainStoreError {
    fn from(err: TerrainStoreError) -> Self {
        let message = err.to_string();
        match err {
            TerrainStoreError::Io { path, .. } => Self {
                kind: "Io".to_string(),
                message,
                path: Some(path.display().to_string()),
                remediation: None,
            },
            TerrainStoreError::Parse { .. } => Self {
                kind: "Parse".to_string(),
                message,
                path: None,
                remediation: None,
            },
            TerrainStoreError::UnsupportedVersion { .. } => Self {
                kind: "UnsupportedVersion".to_string(),
                message,
                path: None,
                remediation: None,
            },
            TerrainStoreError::UnsupportedDatum { .. } => Self {
                kind: "UnsupportedDatum".to_string(),
                message,
                path: None,
                remediation: None,
            },
            TerrainStoreError::DuplicateTile { .. } => Self {
                kind: "DuplicateTile".to_string(),
                message,
                path: None,
                remediation: None,
            },
            TerrainStoreError::TileIdMismatch { path, .. } => Self {
                kind: "TileIdMismatch".to_string(),
                message,
                path: Some(path.display().to_string()),
                remediation: None,
            },
            TerrainStoreError::Checksum { .. } => Self {
                kind: "Checksum".to_string(),
                message,
                path: None,
                remediation: None,
            },
        }
    }
}

#[pymethods]
impl PyTerrainStoreError {
    /// Variant name from the terrain store error enum.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// Human-readable error message.
    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    /// Path associated with the error, if any.
    #[getter]
    fn path(&self) -> Option<String> {
        self.path.clone()
    }

    /// Remediation text associated with the error, if any.
    #[getter]
    fn remediation(&self) -> Option<String> {
        self.remediation.clone()
    }

    fn __repr__(&self) -> String {
        format!("TerrainStoreError(kind={:?})", self.kind)
    }
}

/// Terrain datum conversion and optional geoid-grid loading error details.
#[pyclass(module = "sidereon._sidereon", name = "TerrainDatumError")]
#[derive(Clone)]
pub struct PyTerrainDatumError {
    kind: String,
    message: String,
    path: Option<String>,
    remediation: Option<String>,
}

impl PyTerrainDatumError {
    fn error_text(&self) -> String {
        format!("{}: {}", self.kind, self.message)
    }
}

impl From<TerrainDatumError> for PyTerrainDatumError {
    fn from(err: TerrainDatumError) -> Self {
        let message = err.to_string();
        match err {
            TerrainDatumError::Terrain(_) => Self {
                kind: "Terrain".to_string(),
                message,
                path: None,
                remediation: None,
            },
            TerrainDatumError::Geoid(_) => Self {
                kind: "Geoid".to_string(),
                message,
                path: None,
                remediation: None,
            },
            TerrainDatumError::Io { path, .. } => Self {
                kind: "Io".to_string(),
                message,
                path: Some(path.display().to_string()),
                remediation: None,
            },
            TerrainDatumError::MissingEgm96Dac { path, remediation } => Self {
                kind: "MissingEgm96Dac".to_string(),
                message,
                path: Some(path.display().to_string()),
                remediation: Some(remediation.to_string()),
            },
        }
    }
}

#[pymethods]
impl PyTerrainDatumError {
    /// Variant name from the terrain datum error enum.
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    /// Human-readable error message.
    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    /// Path associated with the error, if any.
    #[getter]
    fn path(&self) -> Option<String> {
        self.path.clone()
    }

    /// Remediation text associated with the error, if any.
    #[getter]
    fn remediation(&self) -> Option<String> {
        self.remediation.clone()
    }

    fn __repr__(&self) -> String {
        format!("TerrainDatumError(kind={:?})", self.kind)
    }
}

/// Memory-mappable terrain reader backed by terrain store bytes.
#[pyclass(module = "sidereon._sidereon", name = "MmapTerrain")]
pub struct PyMmapTerrain {
    inner: MmapTerrain<'static>,
}

impl PyMmapTerrain {
    fn ellipsoidal_height_with_model(
        &self,
        py: Python<'_>,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
        geoid_model: &PyTerrainGeoidModel,
    ) -> PyResult<PyEllipsoidalHeightM> {
        match &geoid_model.kind {
            PyTerrainGeoidModelKind::Egm96OneDegree => self
                .inner
                .ellipsoidal_height_m_with_model(
                    longitude_deg,
                    latitude_deg,
                    options,
                    TerrainGeoidModel::Egm96OneDegree,
                )
                .map(Into::into)
                .map_err(to_datum_err),
            PyTerrainGeoidModelKind::Egm96FifteenMinute(grid) => {
                let grid = grid.borrow(py);
                self.inner
                    .ellipsoidal_height_m_with_model(
                        longitude_deg,
                        latitude_deg,
                        options,
                        TerrainGeoidModel::Egm96FifteenMinute(&grid.inner),
                    )
                    .map(Into::into)
                    .map_err(to_datum_err)
            }
        }
    }
}

#[pymethods]
impl PyMmapTerrain {
    /// Parse terrain store bytes into an owned Python reader.
    ///
    /// The terrain store keeps orthometric postings `H` in metres. Inputs to
    /// lookup methods are `(longitude_deg, latitude_deg)`.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        MmapTerrain::from_vec(data.to_vec())
            .map(|inner| Self { inner })
            .map_err(to_store_err)
    }

    /// Parse an owned terrain store byte vector into a Python reader.
    ///
    /// This has the same Python behavior as [`MmapTerrain.from_bytes`].
    #[staticmethod]
    fn from_vec(data: &[u8]) -> PyResult<Self> {
        Self::from_bytes(data)
    }

    /// Read and parse a terrain store file from disk.
    #[staticmethod]
    fn from_path(path: PathBuf) -> PyResult<Self> {
        MmapTerrain::from_path(path)
            .map(|inner| Self { inner })
            .map_err(to_store_err)
    }

    /// Return the bilinearly interpolated orthometric height `H` in metres.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn height_m(&mut self, longitude_deg: f64, latitude_deg: f64) -> PyResult<f64> {
        self.inner
            .height_m(longitude_deg, latitude_deg)
            .map_err(to_lookup_err)
    }

    /// Return the orthometric height `H` in metres using explicit lookup
    /// options.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn height_m_with_options(
        &mut self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: &PyDtedLookupOptions,
    ) -> PyResult<f64> {
        self.inner
            .height_m_with_options(longitude_deg, latitude_deg, options.inner())
            .map_err(to_lookup_err)
    }

    /// Return the bilinearly interpolated orthometric height `H` as a typed
    /// value.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn orthometric_height_m(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> PyResult<PyOrthometricHeightM> {
        self.inner
            .orthometric_height_m(longitude_deg, latitude_deg)
            .map(Into::into)
            .map_err(to_lookup_err)
    }

    /// Return the orthometric height `H` as a typed value using explicit lookup
    /// options.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn orthometric_height_m_with_options(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: &PyDtedLookupOptions,
    ) -> PyResult<PyOrthometricHeightM> {
        self.inner
            .orthometric_height_m_with_options(longitude_deg, latitude_deg, options.inner())
            .map(Into::into)
            .map_err(to_lookup_err)
    }

    /// Evaluate `(longitude_deg, latitude_deg)` rows as orthometric heights
    /// `H` in metres.
    #[pyo3(signature = (points_lon_lat_deg, options=None))]
    fn height_batch<'py>(
        &mut self,
        py: Python<'py>,
        points_lon_lat_deg: PyReadonlyArray2<'_, f64>,
        options: Option<&PyDtedLookupOptions>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let points = points_lon_lat("points_lon_lat_deg", &points_lon_lat_deg)?;
        let options = options_or_default(options);
        let mut heights = Vec::with_capacity(points.len());
        for (index, result) in self
            .inner
            .height_batch(&points, options)
            .into_iter()
            .enumerate()
        {
            heights.push(result.map_err(|err| {
                PyValueError::new_err(format!("terrain store point {index}: {err}"))
            })?);
        }
        Ok(np_array(py, &heights))
    }

    /// Evaluate `(longitude_deg, latitude_deg)` rows as typed orthometric
    /// heights `H`.
    #[pyo3(signature = (points_lon_lat_deg, options=None))]
    fn orthometric_height_batch(
        &self,
        points_lon_lat_deg: PyReadonlyArray2<'_, f64>,
        options: Option<&PyDtedLookupOptions>,
    ) -> PyResult<Vec<PyOrthometricHeightM>> {
        let points = points_lon_lat("points_lon_lat_deg", &points_lon_lat_deg)?;
        let options = options_or_default(options);
        let mut heights = Vec::with_capacity(points.len());
        for (index, result) in self
            .inner
            .orthometric_height_batch(&points, options)
            .into_iter()
            .enumerate()
        {
            heights.push(result.map(Into::into).map_err(|err| {
                PyValueError::new_err(format!("terrain store point {index}: {err}"))
            })?);
        }
        Ok(heights)
    }

    /// Return ellipsoidal height `h` in metres using the embedded EGM96
    /// 1-degree grid.
    ///
    /// The input position is terrain order `(longitude_deg, latitude_deg)`.
    fn ellipsoidal_height_m(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> PyResult<PyEllipsoidalHeightM> {
        self.inner
            .ellipsoidal_height_m(longitude_deg, latitude_deg)
            .map(Into::into)
            .map_err(to_datum_err)
    }

    /// Return ellipsoidal height `h` in metres using explicit terrain lookup
    /// options and the embedded EGM96 1-degree grid.
    ///
    /// The input position is terrain order `(longitude_deg, latitude_deg)`.
    fn ellipsoidal_height_m_with_options(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: &PyDtedLookupOptions,
    ) -> PyResult<PyEllipsoidalHeightM> {
        self.inner
            .ellipsoidal_height_m_with_options(longitude_deg, latitude_deg, options.inner())
            .map(Into::into)
            .map_err(to_datum_err)
    }

    /// Return ellipsoidal height `h` in metres using an explicit geoid model.
    ///
    /// The input position is terrain order `(longitude_deg, latitude_deg)`.
    /// Choosing a 15-arcminute model requires a loaded `WW15MGH.DAC` grid and
    /// never falls back to the embedded 1-degree grid.
    fn ellipsoidal_height_m_with_model(
        &self,
        py: Python<'_>,
        longitude_deg: f64,
        latitude_deg: f64,
        options: &PyDtedLookupOptions,
        geoid_model: &PyTerrainGeoidModel,
    ) -> PyResult<PyEllipsoidalHeightM> {
        self.ellipsoidal_height_with_model(
            py,
            longitude_deg,
            latitude_deg,
            options.inner(),
            geoid_model,
        )
    }

    /// Parsed tile index records.
    #[getter]
    fn tile_index(&self) -> Vec<PyTerrainStoreTileIndex> {
        self.inner
            .tile_index()
            .iter()
            .copied()
            .map(Into::into)
            .collect()
    }

    /// File-level vertical datum.
    #[getter]
    fn vertical_datum(&self) -> PyVerticalDatum {
        self.inner.vertical_datum().into()
    }

    /// FNV-1a checksum of the full terrain store byte span.
    fn checksum64(&self) -> u64 {
        self.inner.checksum64()
    }

    /// Return the store bytes accepted by this reader.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    fn __repr__(&self) -> String {
        format!("MmapTerrain(tiles={})", self.inner.tile_index().len())
    }
}

/// Convert a DTED tile tree into memory-mappable terrain store bytes.
///
/// Input DTED postings are orthometric heights `H` in metres.
#[pyfunction]
fn dted_tree_to_mmap_store<'py>(py: Python<'py>, root: PathBuf) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = core_dted_tree_to_mmap_store(root).map_err(to_store_err)?;
    Ok(PyBytes::new(py, &bytes))
}

/// Convert a DTED tile tree and write terrain store bytes to `output_path`.
///
/// Input DTED postings are orthometric heights `H` in metres.
#[pyfunction]
fn write_dted_tree_to_mmap_store(root: PathBuf, output_path: PathBuf) -> PyResult<()> {
    core_write_dted_tree_to_mmap_store(root, output_path).map_err(to_store_err)
}

/// Return an FNV-1a checksum for terrain store bytes.
#[pyfunction]
fn terrain_store_checksum64(data: &[u8]) -> u64 {
    core_terrain_store_checksum64(data)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVerticalDatum>()?;
    m.add_class::<PyOrthometricHeightM>()?;
    m.add_class::<PyEllipsoidalHeightM>()?;
    m.add_class::<PyTerrainStoreTileIndex>()?;
    m.add_class::<PyEgm96FifteenMinuteGeoid>()?;
    m.add_class::<PyTerrainGeoidModel>()?;
    m.add_class::<PyTerrainStoreError>()?;
    m.add_class::<PyTerrainDatumError>()?;
    m.add_class::<PyMmapTerrain>()?;
    m.add_function(wrap_pyfunction!(dted_tree_to_mmap_store, m)?)?;
    m.add_function(wrap_pyfunction!(write_dted_tree_to_mmap_store, m)?)?;
    m.add_function(wrap_pyfunction!(terrain_store_checksum64, m)?)?;
    Ok(())
}
