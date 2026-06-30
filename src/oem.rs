//! CCSDS OEM binding.
//!
//! Provides typed Python value objects for the core `Oem` segment/state/
//! covariance structs and parse/encode entry points for KVN and XML. The grammar
//! and serialization stay entirely in `sidereon-core`; this module only marshals
//! strings, optional fields, and numpy vectors. It mirrors the sibling CDM and
//! OMM bindings: parsed messages round-trip through `to_kvn_string` /
//! `to_xml_string`, and the same value objects are constructible from Python.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::oem::{
    encode_kvn, encode_xml, parse_kvn, parse_xml, Oem, OemCovariance, OemMetadata, OemSegment,
    OemState,
};

use crate::marshal::{
    covariance6_from_array, covariance6_to_array, fixed_array, hash_debug, FinitePolicy,
};
use crate::{np_array, OemParseError};

fn to_oem_err<E: std::fmt::Display>(err: E) -> PyErr {
    OemParseError::new_err(err.to_string())
}

fn optional_vec3(
    name: &str,
    values: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Option<[f64; 3]>> {
    match values {
        Some(array) => Ok(Some(fixed_array::<3>(
            name,
            &array,
            FinitePolicy::AllowNonFinite,
        )?)),
        None => Ok(None),
    }
}

/// One OEM metadata/data segment's metadata block.
#[pyclass(module = "sidereon._sidereon", name = "OemMetadata")]
#[derive(Clone, PartialEq)]
pub struct PyOemMetadata {
    inner: OemMetadata,
}

impl PyOemMetadata {
    fn from_inner(inner: OemMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemMetadata {
    #[new]
    #[pyo3(signature = (
        object_name,
        object_id,
        center_name,
        ref_frame,
        time_system,
        start_time,
        stop_time,
        *,
        useable_start_time=None,
        useable_stop_time=None,
        interpolation=None,
        interpolation_degree=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: String,
        time_system: String,
        start_time: String,
        stop_time: String,
        useable_start_time: Option<String>,
        useable_stop_time: Option<String>,
        interpolation: Option<String>,
        interpolation_degree: Option<u32>,
    ) -> Self {
        Self {
            inner: OemMetadata {
                object_name,
                object_id,
                center_name,
                ref_frame,
                time_system,
                start_time,
                stop_time,
                useable_start_time,
                useable_stop_time,
                interpolation,
                interpolation_degree,
            },
        }
    }

    #[getter]
    fn object_name(&self) -> String {
        self.inner.object_name.clone()
    }

    #[getter]
    fn object_id(&self) -> String {
        self.inner.object_id.clone()
    }

    #[getter]
    fn center_name(&self) -> String {
        self.inner.center_name.clone()
    }

    #[getter]
    fn ref_frame(&self) -> String {
        self.inner.ref_frame.clone()
    }

    #[getter]
    fn time_system(&self) -> String {
        self.inner.time_system.clone()
    }

    #[getter]
    fn start_time(&self) -> String {
        self.inner.start_time.clone()
    }

    #[getter]
    fn stop_time(&self) -> String {
        self.inner.stop_time.clone()
    }

    #[getter]
    fn useable_start_time(&self) -> Option<String> {
        self.inner.useable_start_time.clone()
    }

    #[getter]
    fn useable_stop_time(&self) -> Option<String> {
        self.inner.useable_stop_time.clone()
    }

    #[getter]
    fn interpolation(&self) -> Option<String> {
        self.inner.interpolation.clone()
    }

    #[getter]
    fn interpolation_degree(&self) -> Option<u32> {
        self.inner.interpolation_degree
    }

    fn __repr__(&self) -> String {
        format!(
            "OemMetadata(object_name={:?}, ref_frame={:?}, time_system={:?})",
            self.inner.object_name, self.inner.ref_frame, self.inner.time_system
        )
    }

    fn __eq__(&self, other: &PyOemMetadata) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// One OEM Cartesian state sample.
#[pyclass(module = "sidereon._sidereon", name = "OemState")]
#[derive(Clone, PartialEq)]
pub struct PyOemState {
    inner: OemState,
}

impl PyOemState {
    fn from_inner(inner: OemState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemState {
    #[new]
    #[pyo3(signature = (epoch, position_km, velocity_km_s, *, acceleration_km_s2=None))]
    fn new(
        epoch: String,
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
        acceleration_km_s2: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OemState {
                epoch,
                position_km: fixed_array::<3>(
                    "position_km",
                    &position_km,
                    FinitePolicy::AllowNonFinite,
                )?,
                velocity_km_s: fixed_array::<3>(
                    "velocity_km_s",
                    &velocity_km_s,
                    FinitePolicy::AllowNonFinite,
                )?,
                acceleration_km_s2: optional_vec3("acceleration_km_s2", acceleration_km_s2)?,
            },
        })
    }

    /// Epoch text exactly as carried by the message.
    #[getter]
    fn epoch(&self) -> String {
        self.inner.epoch.clone()
    }

    /// Position vector as a numpy `(3,)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_km)
    }

    /// Velocity vector as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_km_s)
    }

    /// Acceleration as a numpy `(3,)` array in km/s^2, or `None` when absent.
    #[getter]
    fn acceleration_km_s2<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .acceleration_km_s2
            .map(|acceleration| np_array(py, &acceleration))
    }

    fn __repr__(&self) -> String {
        format!(
            "OemState(epoch={:?}, has_acceleration={})",
            self.inner.epoch,
            self.inner.acceleration_km_s2.is_some()
        )
    }

    fn __eq__(&self, other: &PyOemState) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// One OEM covariance block: an epoch, an optional reference frame, and a 6x6
/// state covariance.
#[pyclass(module = "sidereon._sidereon", name = "OemCovariance")]
#[derive(Clone, PartialEq)]
pub struct PyOemCovariance {
    inner: OemCovariance,
}

impl PyOemCovariance {
    fn from_inner(inner: OemCovariance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemCovariance {
    #[new]
    #[pyo3(signature = (epoch, matrix, *, cov_ref_frame=None))]
    fn new(
        epoch: String,
        matrix: PyReadonlyArray2<'_, f64>,
        cov_ref_frame: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OemCovariance {
                epoch,
                cov_ref_frame,
                matrix: covariance6_from_array(&matrix, "matrix")?,
            },
        })
    }

    /// Covariance epoch text exactly as carried by the message.
    #[getter]
    fn epoch(&self) -> String {
        self.inner.epoch.clone()
    }

    /// Covariance reference frame, or `None` when not stated.
    #[getter]
    fn cov_ref_frame(&self) -> Option<String> {
        self.inner.cov_ref_frame.clone()
    }

    /// State covariance as a numpy `(6, 6)` array for `[r, v]` in km and km/s.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        covariance6_to_array(py, &self.inner.matrix)
    }

    fn __repr__(&self) -> String {
        format!(
            "OemCovariance(epoch={:?}, cov_ref_frame={:?})",
            self.inner.epoch, self.inner.cov_ref_frame
        )
    }

    fn __eq__(&self, other: &PyOemCovariance) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// One OEM segment: metadata, ephemeris state samples, and covariance blocks.
#[pyclass(module = "sidereon._sidereon", name = "OemSegment")]
#[derive(Clone, PartialEq)]
pub struct PyOemSegment {
    inner: OemSegment,
}

impl PyOemSegment {
    fn from_inner(inner: OemSegment) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemSegment {
    #[new]
    #[pyo3(signature = (metadata, states, *, covariances=None))]
    fn new(
        metadata: PyOemMetadata,
        states: Vec<PyOemState>,
        covariances: Option<Vec<PyOemCovariance>>,
    ) -> Self {
        Self {
            inner: OemSegment {
                metadata: metadata.inner,
                states: states.into_iter().map(|state| state.inner).collect(),
                covariances: covariances
                    .unwrap_or_default()
                    .into_iter()
                    .map(|covariance| covariance.inner)
                    .collect(),
            },
        }
    }

    #[getter]
    fn metadata(&self) -> PyOemMetadata {
        PyOemMetadata::from_inner(self.inner.metadata.clone())
    }

    #[getter]
    fn states(&self) -> Vec<PyOemState> {
        self.inner
            .states
            .iter()
            .cloned()
            .map(PyOemState::from_inner)
            .collect()
    }

    #[getter]
    fn covariances(&self) -> Vec<PyOemCovariance> {
        self.inner
            .covariances
            .iter()
            .cloned()
            .map(PyOemCovariance::from_inner)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "OemSegment(object_name={:?}, states={}, covariances={})",
            self.inner.metadata.object_name,
            self.inner.states.len(),
            self.inner.covariances.len()
        )
    }

    fn __eq__(&self, other: &PyOemSegment) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// A CCSDS Orbit Ephemeris Message parsed from KVN or XML, or built directly.
#[pyclass(module = "sidereon._sidereon", name = "Oem")]
#[derive(Clone, PartialEq)]
pub struct PyOem {
    inner: Oem,
}

impl PyOem {
    fn from_inner(inner: Oem) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOem {
    #[new]
    #[pyo3(signature = (segments, *, ccsds_oem_vers=None, creation_date=None, originator=None))]
    fn new(
        segments: Vec<PyOemSegment>,
        ccsds_oem_vers: Option<String>,
        creation_date: Option<String>,
        originator: Option<String>,
    ) -> Self {
        Self {
            inner: Oem {
                ccsds_oem_vers: ccsds_oem_vers.unwrap_or_else(|| "2.0".to_string()),
                creation_date,
                originator,
                segments: segments.into_iter().map(|segment| segment.inner).collect(),
                skipped_states: 0,
            },
        }
    }

    #[getter]
    fn ccsds_oem_vers(&self) -> String {
        self.inner.ccsds_oem_vers.clone()
    }

    #[getter]
    fn creation_date(&self) -> Option<String> {
        self.inner.creation_date.clone()
    }

    #[getter]
    fn originator(&self) -> Option<String> {
        self.inner.originator.clone()
    }

    #[getter]
    fn segments(&self) -> Vec<PyOemSegment> {
        self.inner
            .segments
            .iter()
            .cloned()
            .map(PyOemSegment::from_inner)
            .collect()
    }

    /// Forgiving-parse count of ephemeris data lines skipped as malformed.
    #[getter]
    fn skipped_states(&self) -> usize {
        self.inner.skipped_states
    }

    /// Encode this OEM to CCSDS OEM KVN text via the core writer.
    fn to_kvn_string(&self) -> String {
        encode_kvn(&self.inner)
    }

    /// Encode this OEM to CCSDS OEM XML text via the core writer.
    fn to_xml_string(&self) -> String {
        encode_xml(&self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Oem(ccsds_oem_vers={:?}, segments={})",
            self.inner.ccsds_oem_vers,
            self.inner.segments.len()
        )
    }

    fn __eq__(&self, other: &PyOem) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// Parse CCSDS OEM KVN text.
#[pyfunction]
fn parse_oem_kvn(text: &str) -> PyResult<PyOem> {
    parse_kvn(text).map(PyOem::from_inner).map_err(to_oem_err)
}

/// Parse CCSDS OEM XML text.
#[pyfunction]
fn parse_oem_xml(text: &str) -> PyResult<PyOem> {
    parse_xml(text).map(PyOem::from_inner).map_err(to_oem_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOemMetadata>()?;
    m.add_class::<PyOemState>()?;
    m.add_class::<PyOemCovariance>()?;
    m.add_class::<PyOemSegment>()?;
    m.add_class::<PyOem>()?;
    m.add_function(wrap_pyfunction!(parse_oem_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_oem_xml, m)?)?;
    Ok(())
}
