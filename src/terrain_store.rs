//! Memory-mappable terrain store binding.

use std::path::PathBuf;

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

use sidereon_core::geoid::GeoidError;
use sidereon_core::terrain::{DtedHorizontalDatum, DtedLookupOptions, DtedTileError};
use sidereon_core::terrain_store::{
    dted_tile_list_to_mmap_store as core_dted_tile_list_to_mmap_store,
    dted_tree_to_mmap_store as core_dted_tree_to_mmap_store,
    terrain_store_checksum64 as core_terrain_store_checksum64,
    write_dted_tile_list_to_mmap_store as core_write_dted_tile_list_to_mmap_store,
    write_dted_tree_to_mmap_store as core_write_dted_tree_to_mmap_store, DtedTileListEntry,
    Egm96FifteenMinuteGeoid, EllipsoidalHeightM, MmapTerrain, OrthometricHeightM,
    TerrainDatumError, TerrainGeoidModel, TerrainStoreError, TerrainStoreTileIndex, TerrainTileId,
    VerticalDatum, TERRAIN_STORE_NULL_POSTING,
};
use sidereon_core::DigestProvenance;
use sidereon_core::Error as CoreError;

use crate::terrain::{
    heights_with_validity, terrain_error, to_lookup_err, HeightsWithValidity, PyDtedLookupOptions,
    TerrainFailure,
};
use crate::{np_array, parse_claimed_checksum64};

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

fn set_tile_id(dict: &Bound<'_, PyDict>, key: &str, tile_id: TerrainTileId) -> PyResult<()> {
    dict.set_item(key, PyTerrainTileId::from(tile_id))
}

fn copy_details_dict<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let copy = PyDict::new(py);
    for (key, value) in source.iter() {
        if let Ok(nested) = value.downcast::<PyDict>() {
            copy.set_item(key, copy_details_dict(py, nested)?)?;
        } else {
            copy.set_item(key, value)?;
        }
    }
    Ok(copy)
}

fn dted_datum_fields(datum: &DtedHorizontalDatum) -> (&'static str, Option<String>) {
    match datum {
        DtedHorizontalDatum::Wgs84 => ("Wgs84", None),
        DtedHorizontalDatum::Wgs72 => ("Wgs72", None),
        DtedHorizontalDatum::Unstated => ("Unstated", None),
        DtedHorizontalDatum::Other(text) => ("Other", Some(text.clone())),
        other => ("Other", Some(format!("{other:?}"))),
    }
}

fn dted_tile_kind(err: &DtedTileError) -> &'static str {
    match err {
        DtedTileError::Io { .. } => "Io",
        DtedTileError::TooShort { .. } => "TooShort",
        DtedTileError::MissingUhl1 { .. } => "MissingUhl1",
        DtedTileError::InvalidEncoding(_) => "InvalidEncoding",
        DtedTileError::InvalidField(_) => "InvalidField",
        DtedTileError::InvalidDimensions { .. } => "InvalidDimensions",
        DtedTileError::Truncated { .. } => "Truncated",
        DtedTileError::Outside { .. } => "Outside",
        DtedTileError::PostingIndexOutOfBounds { .. } => "PostingIndexOutOfBounds",
        DtedTileError::MissingDataSentinel { .. } => "MissingDataSentinel",
        DtedTileError::Checksum { .. } => "Checksum",
        DtedTileError::EmptyCoordinate => "EmptyCoordinate",
        DtedTileError::InvalidHemisphere { .. } => "InvalidHemisphere",
        DtedTileError::NegativePostingIndex { .. } => "NegativePostingIndex",
        DtedTileError::CoordinateOutOfRange { .. } => "CoordinateOutOfRange",
        DtedTileError::WrongHemisphere { .. } => "WrongHemisphere",
        DtedTileError::OriginNotWholeDegree { .. } => "OriginNotWholeDegree",
        DtedTileError::IntervalCountMismatch { .. } => "IntervalCountMismatch",
        DtedTileError::ProfileLongitudeCountMismatch { .. } => "ProfileLongitudeCountMismatch",
        DtedTileError::UnsupportedPartialProfile { .. } => "UnsupportedPartialProfile",
        DtedTileError::NullPosting { .. } => "NullPosting",
        _ => "Other",
    }
}

fn dted_tile_details(dict: &Bound<'_, PyDict>, err: &DtedTileError) -> PyResult<()> {
    match err {
        DtedTileError::Io { path, message } => {
            dict.set_item("path", path)?;
            dict.set_item("message", message)?;
        }
        DtedTileError::TooShort { path } | DtedTileError::MissingUhl1 { path } => {
            dict.set_item("path", path)?;
        }
        DtedTileError::InvalidEncoding(text) | DtedTileError::InvalidField(text) => {
            dict.set_item("text", text)?;
        }
        DtedTileError::InvalidDimensions {
            path,
            lon_count,
            lat_count,
        } => {
            dict.set_item("path", path)?;
            dict.set_item("lon_count", *lon_count)?;
            dict.set_item("lat_count", *lat_count)?;
        }
        DtedTileError::Truncated {
            path,
            actual,
            expected,
        } => {
            dict.set_item("path", path)?;
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
            dict.set_item("text", text)?;
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

fn geoid_error_fields<'py>(
    py: Python<'py>,
    err: &GeoidError,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let details = PyDict::new(py);
    let kind = match err {
        GeoidError::InvalidDimensions { expected, found } => {
            details.set_item("expected", *expected)?;
            details.set_item("found", *found)?;
            "InvalidDimensions"
        }
        GeoidError::InvalidSpacing { field } => {
            details.set_item("field", *field)?;
            "InvalidSpacing"
        }
        GeoidError::NonFiniteValue { index } => {
            details.set_item("index", *index)?;
            "NonFiniteValue"
        }
        GeoidError::Parse { reason } => {
            details.set_item("reason", reason)?;
            "Parse"
        }
    };
    Ok((kind.to_string(), details))
}

fn core_terrain_error_fields<'py>(
    py: Python<'py>,
    err: &CoreError,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let details = PyDict::new(py);
    let kind = match err {
        CoreError::MissingTerrainTile {
            lat_index,
            lon_index,
        } => {
            details.set_item("lat_index", *lat_index)?;
            details.set_item("lon_index", *lon_index)?;
            "MissingTerrainTile"
        }
        CoreError::UnknownTerrainElevation {
            lat_index,
            lon_index,
            latitude_posting,
            longitude_posting,
        } => {
            details.set_item("lat_index", *lat_index)?;
            details.set_item("lon_index", *lon_index)?;
            details.set_item("latitude_posting", *latitude_posting)?;
            details.set_item("longitude_posting", *longitude_posting)?;
            "UnknownTerrainElevation"
        }
        CoreError::NonWgs84TerrainTile {
            lat_index,
            lon_index,
            datum,
        } => {
            details.set_item("lat_index", *lat_index)?;
            details.set_item("lon_index", *lon_index)?;
            let (kind, text) = dted_datum_fields(datum);
            let nested = PyDict::new(py);
            nested.set_item("kind", kind)?;
            nested.set_item("text", text)?;
            details.set_item("datum", nested)?;
            "NonWgs84TerrainTile"
        }
        CoreError::TerrainTile {
            lat_index,
            lon_index,
            error,
        } => {
            details.set_item("lat_index", *lat_index)?;
            details.set_item("lon_index", *lon_index)?;
            let nested = PyDict::new(py);
            nested.set_item("kind", dted_tile_kind(error))?;
            nested.set_item("message", error.to_string())?;
            dted_tile_details(&nested, error)?;
            details.set_item("error", nested)?;
            "TerrainTile"
        }
        CoreError::TerrainTileOrigin {
            path,
            lat_index,
            lon_index,
            origin_latitude,
            origin_longitude,
        } => {
            details.set_item("path", path.display().to_string())?;
            details.set_item("lat_index", *lat_index)?;
            details.set_item("lon_index", *lon_index)?;
            details.set_item("origin_latitude", *origin_latitude)?;
            details.set_item("origin_longitude", *origin_longitude)?;
            "TerrainTileOrigin"
        }
        CoreError::InvalidInput(reason) => {
            details.set_item("reason", reason)?;
            "InvalidInput"
        }
        other => {
            details.set_item("debug", format!("{other:?}"))?;
            "Other"
        }
    };
    Ok((kind.to_string(), details))
}

fn to_store_err(py: Python<'_>, err: TerrainStoreError) -> PyErr {
    let typed = match PyTerrainStoreError::from_core(py, err) {
        Ok(typed) => typed,
        Err(error) => return error,
    };
    let py_err = PyValueError::new_err(typed.error_text());
    let detail = match Py::new(py, typed) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    if let Err(error) = py_err.value(py).setattr("detail", detail) {
        return error;
    }
    py_err
}

/// A datum-conversion refusal. A terrain lookup refusal inside it raises
/// `TerrainError` with the lookup error as `detail`; every other refusal
/// raises `ValueError`. Both keep the `kind: message` text.
fn to_datum_err(py: Python<'_>, err: TerrainDatumError) -> PyErr {
    match err {
        TerrainDatumError::Terrain(lookup) => {
            let outer = match PyTerrainDatumError::from_core(
                py,
                TerrainDatumError::Terrain(lookup.clone()),
            ) {
                Ok(outer) => outer,
                Err(error) => return error,
            };
            let message = outer.error_text();
            let detail = match Py::new(py, outer) {
                Ok(detail) => detail,
                Err(error) => return error,
            };
            let py_err = terrain_error(py, TerrainFailure::Lookup(lookup), message, None);
            if let Err(error) = py_err.value(py).setattr("datum_error", detail) {
                return error;
            }
            py_err
        }
        other => {
            let typed = match PyTerrainDatumError::from_core(py, other) {
                Ok(typed) => typed,
                Err(error) => return error,
            };
            let py_err = PyValueError::new_err(typed.error_text());
            let detail = match Py::new(py, typed) {
                Ok(detail) => detail,
                Err(error) => return error,
            };
            if let Err(error) = py_err.value(py).setattr("detail", detail) {
                return error;
            }
            py_err
        }
    }
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
                .map_err(|err| to_datum_err(py, err)),
            PyTerrainGeoidModelKind::Egm96FifteenMinute(grid) => {
                let grid = grid.borrow(py);
                self.inner
                    .to_ellipsoidal_height_deg(
                        latitude_deg,
                        longitude_deg,
                        TerrainGeoidModel::Egm96FifteenMinute(&grid.inner),
                    )
                    .map(Into::into)
                    .map_err(|err| to_datum_err(py, err))
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
                .map_err(|err| to_datum_err(py, err)),
            PyTerrainGeoidModelKind::Egm96FifteenMinute(grid) => {
                let grid = grid.borrow(py);
                self.inner
                    .to_ellipsoidal_height_rad(
                        latitude_rad,
                        longitude_rad,
                        TerrainGeoidModel::Egm96FifteenMinute(&grid.inner),
                    )
                    .map(Into::into)
                    .map_err(|err| to_datum_err(py, err))
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

/// Integer terrain tile id used by DTED and terrain-store accessors.
#[pyclass(module = "sidereon._sidereon", name = "TerrainTileId")]
#[derive(Clone, Copy)]
pub struct PyTerrainTileId {
    inner: TerrainTileId,
}

impl From<TerrainTileId> for PyTerrainTileId {
    fn from(inner: TerrainTileId) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTerrainTileId {
    /// Build an integer terrain tile id.
    #[new]
    fn new(lat_index: i32, lon_index: i32) -> Self {
        Self {
            inner: TerrainTileId::new(lat_index, lon_index),
        }
    }

    /// Integer latitude tile id.
    #[getter]
    fn lat_index(&self) -> i32 {
        self.inner.lat_index
    }

    /// Integer longitude tile id.
    #[getter]
    fn lon_index(&self) -> i32 {
        self.inner.lon_index
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrainTileId(lat_index={}, lon_index={})",
            self.inner.lat_index, self.inner.lon_index
        )
    }
}

/// One explicit DTED tile source for list-based terrain-store conversion.
#[pyclass(module = "sidereon._sidereon", name = "DtedTileListEntry")]
#[derive(Clone)]
pub struct PyDtedTileListEntry {
    inner: DtedTileListEntry,
}

impl PyDtedTileListEntry {
    fn inner(&self) -> DtedTileListEntry {
        self.inner.clone()
    }
}

#[pymethods]
impl PyDtedTileListEntry {
    /// Build a tile-list entry from a tile id and DTED path.
    #[new]
    fn new(tile_id: &PyTerrainTileId, path: PathBuf) -> Self {
        Self {
            inner: DtedTileListEntry::new(tile_id.inner, path),
        }
    }

    /// Build a tile-list entry from integer tile indices and a DTED path.
    #[staticmethod]
    fn from_indices(lat_index: i32, lon_index: i32, path: PathBuf) -> Self {
        Self {
            inner: DtedTileListEntry::from_indices(lat_index, lon_index, path),
        }
    }

    /// Expected integer tile id for the path.
    #[getter]
    fn tile_id(&self) -> PyTerrainTileId {
        self.inner.tile_id.into()
    }

    /// Path to the DTED tile.
    #[getter]
    fn path(&self) -> String {
        self.inner.path.display().to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "DtedTileListEntry(tile_id=TerrainTileId(lat_index={}, lon_index={}))",
            self.inner.tile_id.lat_index, self.inner.tile_id.lon_index
        )
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
    fn from_ww15mgh_dac_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        Egm96FifteenMinuteGeoid::from_ww15mgh_dac_bytes(data)
            .map(|inner| Self { inner })
            .map_err(|err| to_datum_err(py, err))
    }

    /// Read and load `WW15MGH.DAC` from disk.
    ///
    /// A missing file raises `ValueError` whose message starts with
    /// `MissingEgm96Dac`; it does not fall back to the embedded 1-degree grid.
    #[staticmethod]
    fn from_ww15mgh_dac_path(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        Egm96FifteenMinuteGeoid::from_ww15mgh_dac_path(path)
            .map(|inner| Self { inner })
            .map_err(|err| to_datum_err(py, err))
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
pub struct PyTerrainStoreError {
    kind: String,
    message: String,
    path: Option<String>,
    remediation: Option<String>,
    details: Py<PyDict>,
}

impl PyTerrainStoreError {
    fn error_text(&self) -> String {
        format!("{}: {}", self.kind, self.message)
    }

    fn from_core(py: Python<'_>, err: TerrainStoreError) -> PyResult<Self> {
        let message = err.to_string();
        let details = PyDict::new(py);
        let (kind, path) = match &err {
            TerrainStoreError::Io { path, message } => {
                details.set_item("path", path.display().to_string())?;
                details.set_item("message", message)?;
                ("Io", Some(path.display().to_string()))
            }
            TerrainStoreError::Parse { reason } => {
                details.set_item("reason", reason)?;
                ("Parse", None)
            }
            TerrainStoreError::UnsupportedVersion { version } => {
                details.set_item("version", *version)?;
                ("UnsupportedVersion", None)
            }
            TerrainStoreError::UnsupportedDatum { tag } => {
                details.set_item("tag", *tag)?;
                ("UnsupportedDatum", None)
            }
            TerrainStoreError::DuplicateTile {
                lat_index,
                lon_index,
            } => {
                details.set_item("lat_index", *lat_index)?;
                details.set_item("lon_index", *lon_index)?;
                ("DuplicateTile", None)
            }
            TerrainStoreError::TileIdMismatch {
                path,
                expected,
                found,
            } => {
                details.set_item("path", path.display().to_string())?;
                set_tile_id(&details, "expected", *expected)?;
                set_tile_id(&details, "found", *found)?;
                ("TileIdMismatch", Some(path.display().to_string()))
            }
            TerrainStoreError::Checksum {
                lat_index,
                lon_index,
                expected,
                found,
            } => {
                details.set_item("lat_index", *lat_index)?;
                details.set_item("lon_index", *lon_index)?;
                details.set_item("expected", *expected)?;
                details.set_item("found", *found)?;
                ("Checksum", None)
            }
            TerrainStoreError::AttestedChecksumMismatch { expected, found } => {
                details.set_item("expected", *expected)?;
                details.set_item("found", *found)?;
                ("AttestedChecksumMismatch", None)
            }
            TerrainStoreError::TileIdOutOfRange {
                lat_index,
                lon_index,
            } => {
                details.set_item("lat_index", *lat_index)?;
                details.set_item("lon_index", *lon_index)?;
                ("TileIdOutOfRange", None)
            }
            TerrainStoreError::TileBoundsMismatch {
                lat_index,
                lon_index,
                field,
            } => {
                details.set_item("lat_index", *lat_index)?;
                details.set_item("lon_index", *lon_index)?;
                details.set_item("field", field)?;
                ("TileBoundsMismatch", None)
            }
            TerrainStoreError::NonWgs84Tile { path, datum } => {
                details.set_item("path", path.display().to_string())?;
                let (datum_kind, datum_text) = dted_datum_fields(datum);
                details.set_item("datum_kind", datum_kind)?;
                if let Some(text) = datum_text {
                    details.set_item("datum_text", text)?;
                }
                ("NonWgs84Tile", Some(path.display().to_string()))
            }
            TerrainStoreError::Tile { path, error } => {
                details.set_item("path", path.display().to_string())?;
                let nested = PyDict::new(py);
                nested.set_item("kind", dted_tile_kind(error))?;
                nested.set_item("message", error.to_string())?;
                dted_tile_details(&nested, error)?;
                details.set_item("error", nested)?;
                ("Tile", Some(path.display().to_string()))
            }
            other => {
                details.set_item("debug", format!("{other:?}"))?;
                ("Other", None)
            }
        };
        Ok(Self {
            kind: kind.to_string(),
            message,
            path,
            remediation: None,
            details: details.unbind(),
        })
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

    /// Complete payload fields from the core error variant.
    fn details(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        Ok(copy_details_dict(py, self.details.bind(py))?.unbind())
    }

    fn __repr__(&self) -> String {
        format!("TerrainStoreError(kind={:?})", self.kind)
    }
}

/// Terrain datum conversion and optional geoid-grid loading error details.
#[pyclass(module = "sidereon._sidereon", name = "TerrainDatumError")]
pub struct PyTerrainDatumError {
    kind: String,
    message: String,
    path: Option<String>,
    remediation: Option<String>,
    details: Py<PyDict>,
}

impl PyTerrainDatumError {
    fn error_text(&self) -> String {
        format!("{}: {}", self.kind, self.message)
    }
}

impl PyTerrainDatumError {
    fn from_core(py: Python<'_>, err: TerrainDatumError) -> PyResult<Self> {
        let message = err.to_string();
        let details = PyDict::new(py);
        let (kind, path, remediation) = match &err {
            TerrainDatumError::Terrain(core_err) => {
                let (error_kind, error_details) = core_terrain_error_fields(py, core_err)?;
                details.set_item("error_kind", error_kind)?;
                details.set_item("error", error_details)?;
                ("Terrain", None, None)
            }
            TerrainDatumError::Geoid(error) => {
                let (nested_kind, nested) = geoid_error_fields(py, error)?;
                details.set_item("error_kind", nested_kind)?;
                details.set_item("error", nested)?;
                ("Geoid", None, None)
            }
            TerrainDatumError::Io { path, message } => {
                details.set_item("path", path.display().to_string())?;
                details.set_item("message", message)?;
                ("Io", Some(path.display().to_string()), None)
            }
            TerrainDatumError::MissingEgm96Dac { path, remediation } => {
                details.set_item("path", path.display().to_string())?;
                details.set_item("remediation", remediation)?;
                (
                    "MissingEgm96Dac",
                    Some(path.display().to_string()),
                    Some(remediation.to_string()),
                )
            }
        };
        Ok(Self {
            kind: kind.to_string(),
            message,
            path,
            remediation,
            details: details.unbind(),
        })
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

    /// Complete payload fields from the core error variant.
    fn details(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        Ok(copy_details_dict(py, self.details.bind(py))?.unbind())
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
                .map_err(|err| to_datum_err(py, err)),
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
                    .map_err(|err| to_datum_err(py, err))
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
    fn from_bytes(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        MmapTerrain::from_vec(data.to_vec())
            .map(|inner| Self { inner })
            .map_err(|err| to_store_err(py, err))
    }

    /// Parse an owned terrain store byte vector into a Python reader.
    ///
    /// This has the same Python behavior as [`MmapTerrain.from_bytes`].
    #[staticmethod]
    fn from_vec(py: Python<'_>, data: &[u8]) -> PyResult<Self> {
        Self::from_bytes(py, data)
    }

    /// Read and parse a terrain store file from disk.
    #[staticmethod]
    fn from_path(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        MmapTerrain::from_path(path)
            .map(|inner| Self { inner })
            .map_err(|err| to_store_err(py, err))
    }

    /// Open a terrain store from disk using a caller-attested checksum.
    ///
    /// The claim must fit an unsigned 64-bit integer. This validates the store
    /// layout without hashing its payload; call [`MmapTerrain.verify`] to
    /// escalate the handle to verified provenance.
    #[staticmethod]
    fn from_path_attested(
        py: Python<'_>,
        path: PathBuf,
        claimed_checksum64: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let claimed_checksum64 = parse_claimed_checksum64(claimed_checksum64)?;
        MmapTerrain::from_path_attested(path, claimed_checksum64)
            .map(|inner| Self { inner })
            .map_err(|err| to_store_err(py, err))
    }

    /// Return the bilinearly interpolated orthometric height `H` in metres.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn height_m(&mut self, py: Python<'_>, longitude_deg: f64, latitude_deg: f64) -> PyResult<f64> {
        self.inner
            .height_m(longitude_deg, latitude_deg)
            .map_err(|err| to_lookup_err(py, err))
    }

    /// Return the orthometric height `H` in metres using explicit lookup
    /// options.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn height_m_with_options(
        &mut self,
        py: Python<'_>,
        longitude_deg: f64,
        latitude_deg: f64,
        options: &PyDtedLookupOptions,
    ) -> PyResult<f64> {
        self.inner
            .height_m_with_options(longitude_deg, latitude_deg, options.inner())
            .map_err(|err| to_lookup_err(py, err))
    }

    /// Return the bilinearly interpolated orthometric height `H` as a typed
    /// value.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn orthometric_height_m(
        &self,
        py: Python<'_>,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> PyResult<PyOrthometricHeightM> {
        self.inner
            .orthometric_height_m(longitude_deg, latitude_deg)
            .map(Into::into)
            .map_err(|err| to_lookup_err(py, err))
    }

    /// Return the orthometric height `H` as a typed value using explicit lookup
    /// options.
    ///
    /// The input position is `(longitude_deg, latitude_deg)`.
    fn orthometric_height_m_with_options(
        &self,
        py: Python<'_>,
        longitude_deg: f64,
        latitude_deg: f64,
        options: &PyDtedLookupOptions,
    ) -> PyResult<PyOrthometricHeightM> {
        self.inner
            .orthometric_height_m_with_options(longitude_deg, latitude_deg, options.inner())
            .map(Into::into)
            .map_err(|err| to_lookup_err(py, err))
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
                let message = format!("terrain store point {index}: {err}");
                terrain_error(py, TerrainFailure::Lookup(err), message, Some(index))
            })?);
        }
        Ok(np_array(py, &heights))
    }

    /// Evaluate `(longitude_deg, latitude_deg)` rows as `(heights, valid)`: a
    /// numpy `(n,)` float64 array of orthometric heights `H` in metres, NaN at
    /// a row whose lookup weights a null posting (an unknown elevation), and a
    /// row-aligned bool array, False at those rows. Any other refusal raises
    /// `TerrainError` with the failing row as `point_index`.
    #[pyo3(signature = (points_lon_lat_deg, options=None))]
    fn height_batch_with_validity<'py>(
        &mut self,
        py: Python<'py>,
        points_lon_lat_deg: PyReadonlyArray2<'_, f64>,
        options: Option<&PyDtedLookupOptions>,
    ) -> PyResult<HeightsWithValidity<'py>> {
        let points = points_lon_lat("points_lon_lat_deg", &points_lon_lat_deg)?;
        let options = options_or_default(options);
        heights_with_validity(
            py,
            self.inner.height_batch(&points, options),
            "terrain store point",
        )
    }

    /// Evaluate `(longitude_deg, latitude_deg)` rows as typed orthometric
    /// heights `H`.
    #[pyo3(signature = (points_lon_lat_deg, options=None))]
    fn orthometric_height_batch(
        &self,
        py: Python<'_>,
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
                let message = format!("terrain store point {index}: {err}");
                terrain_error(py, TerrainFailure::Lookup(err), message, Some(index))
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
        py: Python<'_>,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> PyResult<PyEllipsoidalHeightM> {
        self.inner
            .ellipsoidal_height_m(longitude_deg, latitude_deg)
            .map(Into::into)
            .map_err(|err| to_datum_err(py, err))
    }

    /// Return ellipsoidal height `h` in metres using explicit terrain lookup
    /// options and the embedded EGM96 1-degree grid.
    ///
    /// The input position is terrain order `(longitude_deg, latitude_deg)`.
    fn ellipsoidal_height_m_with_options(
        &self,
        py: Python<'_>,
        longitude_deg: f64,
        latitude_deg: f64,
        options: &PyDtedLookupOptions,
    ) -> PyResult<PyEllipsoidalHeightM> {
        self.inner
            .ellipsoidal_height_m_with_options(longitude_deg, latitude_deg, options.inner())
            .map(Into::into)
            .map_err(|err| to_datum_err(py, err))
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

    /// Number of tiles in this terrain store.
    #[getter]
    fn tile_count(&self) -> usize {
        self.inner.tile_count()
    }

    /// Sorted integer tile ids present in this terrain store.
    #[getter]
    fn tile_ids(&self) -> Vec<PyTerrainTileId> {
        self.inner
            .tile_ids()
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

    /// Return `"verified"` when the library measured the checksum, otherwise
    /// `"attested"` when the caller supplied it.
    fn digest_provenance(&self) -> &'static str {
        match self.inner.digest_provenance() {
            DigestProvenance::Verified => "verified",
            DigestProvenance::Attested => "attested",
        }
    }

    /// Verify payload and full-store checksums for this handle.
    ///
    /// Success changes [`MmapTerrain.digest_provenance`] to `"verified"`.
    fn verify(&mut self, py: Python<'_>) -> PyResult<()> {
        self.inner.verify().map_err(|err| to_store_err(py, err))
    }

    /// Return the store bytes accepted by this reader.
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_bytes())
    }

    /// Return the store bytes accepted by this reader.
    fn as_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.as_bytes())
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
    let bytes = core_dted_tree_to_mmap_store(root).map_err(|err| to_store_err(py, err))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Convert an explicit DTED tile list into memory-mappable terrain store bytes.
#[pyfunction]
fn dted_tile_list_to_mmap_store<'py>(
    py: Python<'py>,
    entries: Vec<PyDtedTileListEntry>,
) -> PyResult<Bound<'py, PyBytes>> {
    let entries = entries
        .iter()
        .map(PyDtedTileListEntry::inner)
        .collect::<Vec<_>>();
    let bytes = core_dted_tile_list_to_mmap_store(&entries).map_err(|err| to_store_err(py, err))?;
    Ok(PyBytes::new(py, &bytes))
}

/// Convert a DTED tile tree and write terrain store bytes to `output_path`.
///
/// Input DTED postings are orthometric heights `H` in metres.
#[pyfunction]
fn write_dted_tree_to_mmap_store(
    py: Python<'_>,
    root: PathBuf,
    output_path: PathBuf,
) -> PyResult<()> {
    core_write_dted_tree_to_mmap_store(root, output_path).map_err(|err| to_store_err(py, err))
}

/// Convert an explicit DTED tile list and write terrain store bytes.
#[pyfunction]
fn write_dted_tile_list_to_mmap_store(
    py: Python<'_>,
    entries: Vec<PyDtedTileListEntry>,
    output_path: PathBuf,
) -> PyResult<()> {
    let entries = entries
        .iter()
        .map(PyDtedTileListEntry::inner)
        .collect::<Vec<_>>();
    core_write_dted_tile_list_to_mmap_store(&entries, output_path)
        .map_err(|err| to_store_err(py, err))
}

/// Return an FNV-1a checksum for terrain store bytes.
#[pyfunction]
fn terrain_store_checksum64(data: &[u8]) -> u64 {
    core_terrain_store_checksum64(data)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::terrain_error_type(m.py())?.setattr("datum_error", m.py().None())?;
    m.add_class::<PyVerticalDatum>()?;
    m.add_class::<PyOrthometricHeightM>()?;
    m.add_class::<PyEllipsoidalHeightM>()?;
    m.add_class::<PyTerrainTileId>()?;
    m.add_class::<PyDtedTileListEntry>()?;
    m.add_class::<PyTerrainStoreTileIndex>()?;
    m.add_class::<PyEgm96FifteenMinuteGeoid>()?;
    m.add_class::<PyTerrainGeoidModel>()?;
    m.add_class::<PyTerrainStoreError>()?;
    // The stored posting value a DTED null (unknown elevation) converts to;
    // lookups that weight it raise `TerrainError`.
    m.add("TERRAIN_STORE_NULL_POSTING", TERRAIN_STORE_NULL_POSTING)?;
    m.add_class::<PyTerrainDatumError>()?;
    m.add_class::<PyMmapTerrain>()?;
    m.add_function(wrap_pyfunction!(dted_tree_to_mmap_store, m)?)?;
    m.add_function(wrap_pyfunction!(dted_tile_list_to_mmap_store, m)?)?;
    m.add_function(wrap_pyfunction!(write_dted_tree_to_mmap_store, m)?)?;
    m.add_function(wrap_pyfunction!(write_dted_tile_list_to_mmap_store, m)?)?;
    m.add_function(wrap_pyfunction!(terrain_store_checksum64, m)?)?;
    Ok(())
}
