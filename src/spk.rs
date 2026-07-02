//! SPK binding: the JPL/NAIF SPK (DAF `.bsp`) ephemeris kernel reader and its
//! body-to-center state query.
//!
//! Marshals SPK bytes (or a file path) into [`sidereon_core::astro::spk::Spk`]
//! and exposes its surface Pythonically: the parsed segment descriptors and a
//! single-call `state(target, center, et)` query returning position (km) and
//! velocity (km/s) at an ephemeris epoch (TDB seconds past J2000). No modeling
//! lives here: the parse is `Spk::from_bytes` and the query is `Spk::spk_state`,
//! so the numbers are exactly what `sidereon-core` produces, including segment
//! Type 21 (Extended Modified Difference Arrays).

use std::path::PathBuf;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyModule};

use sidereon_core::astro::spk::{Spk, SpkError, SpkSegmentDescriptor, SpkState};

use crate::{np_array, to_solve_err};

/// Map an SPK/DAF parse failure into [`SpkParseError`](crate::SpkParseError),
/// preserving the engine message. It derives from `ParseError`, so callers can
/// catch the product-specific type or the shared base.
fn to_spk_err<E: std::fmt::Display>(err: E) -> PyErr {
    crate::SpkParseError::new_err(err.to_string())
}

/// Translate a state-query [`SpkError`] into the right Python exception.
///
/// A request that names a body absent from the kernel, or two bodies with no
/// connecting segment chain, is bad input -> `ValueError`. Everything else
/// (no chain covers the epoch, an unsupported segment type on the path, a
/// malformed segment) is a query the loaded kernel cannot satisfy ->
/// `SolveError`, mirroring how the SP3 binding splits `UnknownSatellite` from
/// the rest.
fn state_query_err(err: SpkError) -> PyErr {
    match err {
        SpkError::UnknownBody { body } => {
            PyValueError::new_err(format!("body {body} is not present in any kernel segment"))
        }
        SpkError::NoSegmentPath { target, center } => PyValueError::new_err(format!(
            "no SPK segment path connects target {target} to center {center}"
        )),
        other => to_solve_err(other.to_string()),
    }
}

/// One SPK segment descriptor: the body pair, frame, data type, and coverage
/// window advertised by the DAF summary records. Read the list from
/// [`Spk.segments`].
#[pyclass(module = "sidereon._sidereon", name = "SpkSegment")]
pub struct PySpkSegment {
    name: String,
    target: i32,
    center: i32,
    frame: i32,
    data_type: i32,
    start_et: f64,
    stop_et: f64,
    start_address: i32,
    end_address: i32,
}

impl PySpkSegment {
    fn from_descriptor(descriptor: &SpkSegmentDescriptor) -> Self {
        Self {
            name: descriptor.name.clone(),
            target: descriptor.target,
            center: descriptor.center,
            frame: descriptor.frame,
            data_type: descriptor.data_type,
            start_et: descriptor.start_et,
            stop_et: descriptor.stop_et,
            start_address: descriptor.start_address,
            end_address: descriptor.end_address,
        }
    }
}

#[pymethods]
impl PySpkSegment {
    /// Segment name from the paired DAF name record.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// NAIF target body identifier.
    #[getter]
    fn target(&self) -> i32 {
        self.target
    }

    /// NAIF center body identifier.
    #[getter]
    fn center(&self) -> i32 {
        self.center
    }

    /// NAIF reference-frame identifier.
    #[getter]
    fn frame(&self) -> i32 {
        self.frame
    }

    /// SPK segment data type (2, 3, or 21 are evaluable here).
    #[getter]
    fn data_type(&self) -> i32 {
        self.data_type
    }

    /// Coverage start, ephemeris (TDB) seconds past J2000.
    #[getter]
    fn start_et(&self) -> f64 {
        self.start_et
    }

    /// Coverage stop, ephemeris (TDB) seconds past J2000.
    #[getter]
    fn stop_et(&self) -> f64 {
        self.stop_et
    }

    /// One-based DAF address of the first segment data word.
    #[getter]
    fn start_address(&self) -> i32 {
        self.start_address
    }

    /// One-based DAF address of the last segment data word.
    #[getter]
    fn end_address(&self) -> i32 {
        self.end_address
    }

    fn __repr__(&self) -> String {
        format!(
            "SpkSegment(target={}, center={}, frame={}, data_type={}, start_et={}, stop_et={})",
            self.target, self.center, self.frame, self.data_type, self.start_et, self.stop_et
        )
    }
}

/// The state of one body relative to another, evaluated from an SPK kernel.
///
/// `position_km` is the position of `target` relative to `center` (kilometres);
/// `velocity_km_s` is the relative velocity (km/s), or `None` when the resolved
/// segment path runs through a position-only Type 2 segment. `frame` is the NAIF
/// reference-frame id shared by the path. Returned by [`Spk.state`].
#[pyclass(module = "sidereon._sidereon", name = "SpkState")]
pub struct PySpkState {
    target: i32,
    center: i32,
    position_km: [f64; 3],
    velocity_km_s: Option<[f64; 3]>,
    frame: i32,
}

impl PySpkState {
    fn from_state(state: SpkState) -> Self {
        Self {
            target: state.target,
            center: state.center,
            position_km: state.position_km,
            velocity_km_s: state.velocity_km_s,
            frame: state.frame,
        }
    }
}

#[pymethods]
impl PySpkState {
    /// NAIF target body identifier for the returned relative state.
    #[getter]
    fn target(&self) -> i32 {
        self.target
    }

    /// NAIF center body identifier for the returned relative state.
    #[getter]
    fn center(&self) -> i32 {
        self.center
    }

    /// Position of `target` relative to `center` as a numpy `(3,)` array,
    /// kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.position_km)
    }

    /// Velocity of `target` relative to `center` as a numpy `(3,)` array in
    /// km/s, or `None` when the path runs through a position-only Type 2
    /// segment.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.velocity_km_s.map(|v| np_array(py, &v))
    }

    /// NAIF reference-frame identifier shared by the resolved segment path.
    #[getter]
    fn frame(&self) -> i32 {
        self.frame
    }

    fn __repr__(&self) -> String {
        format!(
            "SpkState(target={}, center={}, position_km=[{}, {}, {}], velocity_km_s={:?}, frame={})",
            self.target,
            self.center,
            self.position_km[0],
            self.position_km[1],
            self.position_km[2],
            self.velocity_km_s,
            self.frame
        )
    }
}

/// A parsed JPL/NAIF SPK (DAF `.bsp`) ephemeris kernel.
///
/// Construct with [`load_spk`]. Inspect the parsed segments with
/// [`Spk.segments`] and query a body's state relative to a center at an epoch
/// with [`Spk.state`]. Wraps [`sidereon_core::astro::spk::Spk`] unchanged; it
/// reads SPK segment Types 2, 3, and 21.
#[pyclass(module = "sidereon._sidereon", name = "Spk")]
pub struct PySpk {
    pub(crate) inner: Spk,
}

#[pymethods]
impl PySpk {
    /// The kernel's parsed segment descriptors, in DAF summary order.
    #[getter]
    fn segments(&self) -> Vec<PySpkSegment> {
        self.inner
            .segments()
            .iter()
            .map(PySpkSegment::from_descriptor)
            .collect()
    }

    /// DAF internal file name recorded in the kernel header.
    #[getter]
    fn internal_name(&self) -> String {
        self.inner.file_record().internal_name.clone()
    }

    /// Query the state of `target` relative to `center` at ephemeris epoch `et`
    /// (TDB seconds past J2000), resolving and chaining segments as needed.
    ///
    /// Returns an [`SpkState`]. Raises `ValueError` if either body is absent
    /// from the kernel or no segment chain connects them, and `SolveError` if a
    /// chain exists but none covers `et`, the path needs an unsupported segment
    /// type, or a segment is malformed.
    fn state(&self, target: i32, center: i32, et: f64) -> PyResult<PySpkState> {
        let state = self
            .inner
            .spk_state(target, center, et)
            .map_err(state_query_err)?;
        Ok(PySpkState::from_state(state))
    }

    fn __repr__(&self) -> String {
        format!(
            "Spk(internal_name={:?}, segments={})",
            self.inner.file_record().internal_name,
            self.inner.segments().len()
        )
    }
}

/// Parse a JPL/NAIF SPK (DAF `.bsp`) ephemeris kernel from in-memory bytes or a
/// file path.
///
/// `source` may be:
/// - `bytes` / `bytearray`: the full kernel content, parsed directly; or
/// - a path (`str` or `os.PathLike`): the file is read and parsed.
///
/// Raises [`SpkParseError`](crate::SpkParseError) on malformed content,
/// `OSError` if the path cannot be read, and `ValueError` if `source` is neither
/// bytes nor a path.
#[pyfunction]
fn load_spk(source: &Bound<'_, PyAny>) -> PyResult<PySpk> {
    // bytes-like first, so a `bytes` argument keeps the "content" meaning.
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        let inner = Spk::from_bytes(bytes.as_bytes()).map_err(to_spk_err)?;
        return Ok(PySpk { inner });
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        let inner = Spk::from_bytes(unsafe { buf.as_bytes() }).map_err(to_spk_err)?;
        return Ok(PySpk { inner });
    }
    // Otherwise treat it as a path (str / os.PathLike via PyO3's fspath support).
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_spk expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    let inner = Spk::from_bytes(&data).map_err(to_spk_err)?;
    Ok(PySpk { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySpk>()?;
    m.add_class::<PySpkState>()?;
    m.add_class::<PySpkSegment>()?;
    m.add_function(wrap_pyfunction!(load_spk, m)?)?;
    Ok(())
}
