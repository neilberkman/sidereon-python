//! DTED terrain binding.
//!
//! Terrain lookups and DTED tile reads that the core refuses raise
//! `TerrainError`, a `ValueError`, carrying the core error as a typed
//! `TerrainErrorDetail`: an unknown elevation names its tile and posting, a
//! tile on another horizontal datum names that datum.

use std::path::PathBuf;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon_core::terrain::{
    DtedHorizontalDatum, DtedInterpolation, DtedLookupOptions, DtedTerrain, DtedTile, DtedTileError,
};
use sidereon_core::Error as CoreError;

use crate::{np_array, terrain_error_type};

/// The variant name at the head of a derived `Debug` rendering, for the
/// `#[non_exhaustive]` core enums.
fn debug_variant_name<T: std::fmt::Debug>(value: &T) -> String {
    format!("{value:?}")
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// A refused terrain lookup or DTED tile read.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TerrainFailure {
    /// A lookup refusal, the crate error.
    Lookup(CoreError),
    /// A DTED tile read or single-tile lookup refusal.
    Tile(DtedTileError),
}

/// Raise `TerrainError` with `message`, the failure as `detail`, and the batch
/// row it happened at, if any, as `point_index`.
pub(crate) fn terrain_error(
    py: Python<'_>,
    failure: TerrainFailure,
    message: String,
    point_index: Option<usize>,
) -> PyErr {
    let ty = match terrain_error_type(py) {
        Ok(ty) => ty,
        Err(e) => return e,
    };
    let py_err = PyErr::from_type(ty, message);
    let detail = match (PyTerrainErrorDetail { inner: failure }).into_pyobject(py) {
        Ok(detail) => detail,
        Err(e) => return e,
    };
    let value = py_err.value(py);
    if let Err(e) = value.setattr("detail", detail) {
        return e;
    }
    if let Err(e) = value.setattr("point_index", point_index) {
        return e;
    }
    py_err
}

/// A lookup refusal with the core message.
pub(crate) fn to_lookup_err(py: Python<'_>, err: CoreError) -> PyErr {
    let message = err.to_string();
    terrain_error(py, TerrainFailure::Lookup(err), message, None)
}

/// Batch heights and their row-aligned validity mask.
pub(crate) type HeightsWithValidity<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<bool>>);

/// Batch heights with NaN and a False validity entry at each unknown
/// elevation (a lookup that weights a DTED null posting); any other refusal
/// raises `TerrainError` naming its row, prefixed `{row_label} {index}: `.
pub(crate) fn heights_with_validity<'py>(
    py: Python<'py>,
    results: Vec<sidereon_core::Result<f64>>,
    row_label: &str,
) -> PyResult<HeightsWithValidity<'py>> {
    let mut heights = Vec::with_capacity(results.len());
    let mut valid = Vec::with_capacity(results.len());
    for (index, result) in results.into_iter().enumerate() {
        match result {
            Ok(height) => {
                heights.push(height);
                valid.push(true);
            }
            Err(CoreError::UnknownTerrainElevation { .. }) => {
                heights.push(f64::NAN);
                valid.push(false);
            }
            Err(err) => {
                let message = format!("{row_label} {index}: {err}");
                return Err(terrain_error(
                    py,
                    TerrainFailure::Lookup(err),
                    message,
                    Some(index),
                ));
            }
        }
    }
    Ok((np_array(py, &heights), PyArray1::from_slice(py, &valid)))
}

/// A tile refusal with the core message.
fn to_tile_err(py: Python<'_>, err: DtedTileError) -> PyErr {
    let message = err.to_string();
    terrain_error(py, TerrainFailure::Tile(err), message, None)
}

/// A horizontal datum stated by a DTED tile's DSI record.
///
/// `kind` is `Wgs84`, `Wgs72`, `Unstated` (a blank or zero-filled field) or
/// `Other`, whose field text is `text`. Tiles on any datum read as tiles;
/// terrain lookups answer WGS84 positions only from a WGS84 or unstated tile.
#[pyclass(module = "sidereon._sidereon", name = "DtedHorizontalDatum")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyDtedHorizontalDatum {
    inner: DtedHorizontalDatum,
}

#[pymethods]
impl PyDtedHorizontalDatum {
    /// Datum kind, the core variant name.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            DtedHorizontalDatum::Wgs84 => "Wgs84".to_string(),
            DtedHorizontalDatum::Wgs72 => "Wgs72".to_string(),
            DtedHorizontalDatum::Unstated => "Unstated".to_string(),
            DtedHorizontalDatum::Other(_) => "Other".to_string(),
            other => debug_variant_name(other),
        }
    }

    /// The field text of an `Other` datum; None for every other kind.
    #[getter]
    fn text(&self) -> Option<String> {
        match &self.inner {
            DtedHorizontalDatum::Other(text) => Some(text.clone()),
            _ => None,
        }
    }

    /// Whether positions in the tile are WGS84 positions: the field states
    /// WGS84 or is blank.
    #[getter]
    fn is_wgs84_compatible(&self) -> bool {
        self.inner.is_wgs84_compatible()
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!("DtedHorizontalDatum({})", self.inner)
    }

    fn __eq__(&self, other: &PyDtedHorizontalDatum) -> bool {
        self.inner == other.inner
    }
}

/// The typed payload a `TerrainError` carries on its `detail` attribute.
///
/// `family` names the core error enum: `Error` for terrain lookups, or
/// `DtedTileError` for a tile read or single-tile lookup. `kind` is the core
/// variant name and `details()` its fields under their own keys. The
/// accessors return None for a variant that names no such field:
/// `UnknownTerrainElevation` names the tile (`lat_index`, `lon_index`) and the
/// null posting (`latitude_posting`, `longitude_posting`), `NullPosting` the
/// posting, `NonWgs84TerrainTile` the tile and its `datum`, and
/// `MissingTerrainTile` the tile.
#[pyclass(module = "sidereon._sidereon", name = "TerrainErrorDetail")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyTerrainErrorDetail {
    inner: TerrainFailure,
}

#[pymethods]
impl PyTerrainErrorDetail {
    /// The core error enum: `Error` or `DtedTileError`.
    #[getter]
    fn family(&self) -> &'static str {
        match &self.inner {
            TerrainFailure::Lookup(_) => "Error",
            TerrainFailure::Tile(_) => "DtedTileError",
        }
    }

    /// Error kind, the core variant name.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            TerrainFailure::Lookup(err) => match err {
                CoreError::MissingTerrainTile { .. } => "MissingTerrainTile".to_string(),
                CoreError::UnknownTerrainElevation { .. } => "UnknownTerrainElevation".to_string(),
                CoreError::NonWgs84TerrainTile { .. } => "NonWgs84TerrainTile".to_string(),
                other => debug_variant_name(other),
            },
            TerrainFailure::Tile(err) => debug_variant_name(err),
        }
    }

    /// Formatted core error message.
    #[getter]
    fn message(&self) -> String {
        match &self.inner {
            TerrainFailure::Lookup(err) => err.to_string(),
            TerrainFailure::Tile(err) => err.to_string(),
        }
    }

    /// Integer latitude tile id, for the variants that name a tile.
    #[getter]
    fn lat_index(&self) -> Option<i32> {
        match &self.inner {
            TerrainFailure::Lookup(
                CoreError::MissingTerrainTile { lat_index, .. }
                | CoreError::UnknownTerrainElevation { lat_index, .. }
                | CoreError::NonWgs84TerrainTile { lat_index, .. },
            ) => Some(*lat_index),
            _ => None,
        }
    }

    /// Integer longitude tile id, for the variants that name a tile.
    #[getter]
    fn lon_index(&self) -> Option<i32> {
        match &self.inner {
            TerrainFailure::Lookup(
                CoreError::MissingTerrainTile { lon_index, .. }
                | CoreError::UnknownTerrainElevation { lon_index, .. }
                | CoreError::NonWgs84TerrainTile { lon_index, .. },
            ) => Some(*lon_index),
            _ => None,
        }
    }

    /// Zero-based latitude posting index of a null posting.
    #[getter]
    fn latitude_posting(&self) -> Option<usize> {
        match &self.inner {
            TerrainFailure::Lookup(CoreError::UnknownTerrainElevation {
                latitude_posting, ..
            }) => Some(*latitude_posting),
            TerrainFailure::Tile(DtedTileError::NullPosting { latitude_index, .. }) => {
                Some(*latitude_index)
            }
            _ => None,
        }
    }

    /// Zero-based longitude posting (profile) index of a null posting.
    #[getter]
    fn longitude_posting(&self) -> Option<usize> {
        match &self.inner {
            TerrainFailure::Lookup(CoreError::UnknownTerrainElevation {
                longitude_posting,
                ..
            }) => Some(*longitude_posting),
            TerrainFailure::Tile(DtedTileError::NullPosting {
                longitude_index, ..
            }) => Some(*longitude_index),
            _ => None,
        }
    }

    /// The datum a `NonWgs84TerrainTile` tile states.
    #[getter]
    fn datum(&self) -> Option<PyDtedHorizontalDatum> {
        match &self.inner {
            TerrainFailure::Lookup(CoreError::NonWgs84TerrainTile { datum, .. }) => {
                Some(PyDtedHorizontalDatum {
                    inner: datum.clone(),
                })
            }
            _ => None,
        }
    }

    /// Dictionary containing every field of this refusal variant.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            TerrainFailure::Lookup(err) => lookup_details(&dict, err)?,
            TerrainFailure::Tile(err) => tile_details(&dict, err)?,
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainErrorDetail(family=\"{}\", kind=\"{}\", message={:?})",
            self.family(),
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &PyTerrainErrorDetail) -> bool {
        self.inner == other.inner
    }
}

fn lookup_details(dict: &Bound<'_, PyDict>, err: &CoreError) -> PyResult<()> {
    match err {
        CoreError::MissingTerrainTile {
            lat_index,
            lon_index,
        } => {
            dict.set_item("lat_index", *lat_index)?;
            dict.set_item("lon_index", *lon_index)?;
        }
        CoreError::UnknownTerrainElevation {
            lat_index,
            lon_index,
            latitude_posting,
            longitude_posting,
        } => {
            dict.set_item("lat_index", *lat_index)?;
            dict.set_item("lon_index", *lon_index)?;
            dict.set_item("latitude_posting", *latitude_posting)?;
            dict.set_item("longitude_posting", *longitude_posting)?;
        }
        CoreError::NonWgs84TerrainTile {
            lat_index,
            lon_index,
            datum,
        } => {
            dict.set_item("lat_index", *lat_index)?;
            dict.set_item("lon_index", *lon_index)?;
            dict.set_item(
                "datum",
                PyDtedHorizontalDatum {
                    inner: datum.clone(),
                },
            )?;
        }
        other => {
            dict.set_item("debug", format!("{other:?}"))?;
        }
    }
    Ok(())
}

pub(crate) fn tile_details(dict: &Bound<'_, PyDict>, err: &DtedTileError) -> PyResult<()> {
    match err {
        DtedTileError::Io { path, message } => {
            dict.set_item("path", path.as_str())?;
            dict.set_item("message", message.as_str())?;
        }
        DtedTileError::TooShort { path } | DtedTileError::MissingUhl1 { path } => {
            dict.set_item("path", path.as_str())?;
        }
        DtedTileError::InvalidEncoding(text) | DtedTileError::InvalidField(text) => {
            dict.set_item("text", text.as_str())?;
        }
        DtedTileError::InvalidDimensions {
            path,
            lon_count,
            lat_count,
        } => {
            dict.set_item("path", path.as_str())?;
            dict.set_item("lon_count", *lon_count)?;
            dict.set_item("lat_count", *lat_count)?;
        }
        DtedTileError::Truncated {
            path,
            actual,
            expected,
        } => {
            dict.set_item("path", path.as_str())?;
            dict.set_item("actual", *actual)?;
            dict.set_item("expected", *expected)?;
        }
        DtedTileError::Outside {
            longitude,
            latitude,
            origin_longitude,
            origin_latitude,
        } => {
            dict.set_item("longitude", *longitude)?;
            dict.set_item("latitude", *latitude)?;
            dict.set_item("origin_longitude", *origin_longitude)?;
            dict.set_item("origin_latitude", *origin_latitude)?;
        }
        DtedTileError::PostingIndexOutOfBounds {
            longitude_index,
            latitude_index,
        }
        | DtedTileError::NullPosting {
            longitude_index,
            latitude_index,
        } => {
            dict.set_item("longitude_index", *longitude_index)?;
            dict.set_item("latitude_index", *latitude_index)?;
        }
        DtedTileError::MissingDataSentinel { longitude_index } => {
            dict.set_item("longitude_index", *longitude_index)?;
        }
        DtedTileError::Checksum {
            longitude_index,
            checksum,
            sum,
        } => {
            dict.set_item("longitude_index", *longitude_index)?;
            dict.set_item("checksum", *checksum)?;
            dict.set_item("sum", *sum)?;
        }
        DtedTileError::EmptyCoordinate => {}
        DtedTileError::InvalidHemisphere { hemisphere } => {
            dict.set_item("hemisphere", *hemisphere)?;
        }
        DtedTileError::NegativePostingIndex { index } => {
            dict.set_item("index", *index)?;
        }
        DtedTileError::CoordinateOutOfRange { field, text }
        | DtedTileError::OriginNotWholeDegree { field, text } => {
            dict.set_item("field", *field)?;
            dict.set_item("text", text.as_str())?;
        }
        DtedTileError::WrongHemisphere {
            field,
            hemisphere,
            expected,
        } => {
            dict.set_item("field", *field)?;
            dict.set_item("hemisphere", *hemisphere)?;
            dict.set_item("expected", *expected)?;
        }
        DtedTileError::IntervalCountMismatch {
            field,
            interval_tenths_arcsec,
            count,
        } => {
            dict.set_item("field", *field)?;
            dict.set_item("interval_tenths_arcsec", *interval_tenths_arcsec)?;
            dict.set_item("count", *count)?;
        }
        DtedTileError::ProfileLongitudeCountMismatch {
            longitude_index,
            declared,
        } => {
            dict.set_item("longitude_index", *longitude_index)?;
            dict.set_item("declared", *declared)?;
        }
        DtedTileError::UnsupportedPartialProfile {
            longitude_index,
            first_latitude_index,
        } => {
            dict.set_item("longitude_index", *longitude_index)?;
            dict.set_item("first_latitude_index", *first_latitude_index)?;
        }
        other => {
            dict.set_item("debug", format!("{other:?}"))?;
        }
    }
    Ok(())
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
        let mut inner = DtedLookupOptions::default();
        inner.interpolation = interpolation.into();
        Self { inner }
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
    /// Missing tiles use the core sea-level fallback. A lookup that weights a
    /// DTED null posting (an unknown elevation) raises `TerrainError` naming
    /// the tile and posting, as does a tile on a horizontal datum other than
    /// WGS84.
    #[pyo3(signature = (latitude_deg, longitude_deg, options=None))]
    fn height_m(
        &mut self,
        py: Python<'_>,
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
        .map_err(|err| to_lookup_err(py, err))
    }

    /// Return ORTHOMETRIC terrain heights in metres for `(longitude, latitude)` rows.
    ///
    /// `points_lon_lat_deg` is a numpy `(n, 2)` array with longitude in column 0
    /// and latitude in column 1, both in degrees. Missing tiles use the core
    /// sea-level fallback. The first point the core refuses raises
    /// `TerrainError` with the failing row as `point_index`.
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
            heights.push(result.map_err(|err| {
                let message = format!("terrain point {index}: {err}");
                terrain_error(py, TerrainFailure::Lookup(err), message, Some(index))
            })?);
        }
        Ok(np_array(py, &heights))
    }

    /// Return ORTHOMETRIC terrain heights for `(longitude, latitude)` rows as
    /// `(heights, valid)`: a numpy `(n,)` float64 array, NaN at a row whose
    /// lookup weights a DTED null posting (an unknown elevation), and a
    /// row-aligned bool array, False at those rows. Any other refusal raises
    /// `TerrainError` with the failing row as `point_index`.
    #[pyo3(signature = (points_lon_lat_deg, options=None))]
    fn height_batch_with_validity<'py>(
        &mut self,
        py: Python<'py>,
        points_lon_lat_deg: PyReadonlyArray2<'_, f64>,
        options: Option<&PyDtedLookupOptions>,
    ) -> PyResult<HeightsWithValidity<'py>> {
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
        heights_with_validity(
            py,
            self.inner.height_batch(&points, options),
            "terrain point",
        )
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
    ///
    /// Metadata the reader places postings by is checked; an inconsistent
    /// header or data record raises `TerrainError` naming it.
    #[staticmethod]
    fn from_path(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        DtedTile::from_path(path)
            .map(|inner| Self { inner })
            .map_err(|err| to_tile_err(py, err))
    }

    /// The horizontal datum the tile's DSI record states.
    #[getter]
    fn horizontal_datum(&self) -> PyDtedHorizontalDatum {
        PyDtedHorizontalDatum {
            inner: self.inner.horizontal_datum().clone(),
        }
    }

    /// Return nearest-posting ORTHOMETRIC height in metres at latitude and longitude in degrees.
    ///
    /// A DTED null posting (an unknown elevation) raises `TerrainError` with
    /// kind `NullPosting`.
    fn height_m(&self, py: Python<'_>, latitude_deg: f64, longitude_deg: f64) -> PyResult<f64> {
        self.inner
            .get_elevation(longitude_deg, latitude_deg)
            .map(f64::from)
            .map_err(|err| to_tile_err(py, err))
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
    m.add_class::<PyDtedHorizontalDatum>()?;
    m.add_class::<PyTerrainErrorDetail>()?;
    Ok(())
}
