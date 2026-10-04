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

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyModule};

use sidereon_core::astro::spk::{
    inertial_frame_name as core_inertial_frame_name,
    inertial_frame_rotation as core_inertial_frame_rotation, Spk, SpkError, SpkKernels,
    SpkSegmentDescriptor, SpkState,
};

use crate::{np_array, to_solve_err};

fn spk_detail<'py>(py: Python<'py>, err: &SpkError) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "spk")?;
    let kind = match err {
        SpkError::Io { path, message } => {
            detail.set_item("path", path)?;
            detail.set_item("message", message)?;
            "Io"
        }
        SpkError::Truncated {
            context,
            needed,
            actual,
        } => {
            detail.set_item("context", context)?;
            detail.set_item("needed", *needed)?;
            detail.set_item("actual", *actual)?;
            "Truncated"
        }
        SpkError::UnsupportedDafId { id_word } => {
            detail.set_item("id_word", id_word)?;
            "UnsupportedDafId"
        }
        SpkError::UnsupportedBinaryFormat { binary_format } => {
            detail.set_item("binary_format", binary_format)?;
            "UnsupportedBinaryFormat"
        }
        SpkError::UnsupportedSummaryShape { nd, ni } => {
            detail.set_item("nd", *nd)?;
            detail.set_item("ni", *ni)?;
            "UnsupportedSummaryShape"
        }
        SpkError::InvalidField { field, value } => {
            detail.set_item("field", field)?;
            detail.set_item("value", *value)?;
            "InvalidField"
        }
        SpkError::InvalidDoubleField { field, value } => {
            detail.set_item("field", field)?;
            detail.set_item("value", *value)?;
            detail.set_item("value_bits_hex", format!("{:016x}", value.to_bits()))?;
            "InvalidDoubleField"
        }
        SpkError::OutOfCoverage {
            et,
            start_et,
            stop_et,
        } => {
            for (name, value) in [("et", et), ("start_et", start_et), ("stop_et", stop_et)] {
                detail.set_item(name, *value)?;
                detail.set_item(
                    format!("{name}_bits_hex"),
                    format!("{:016x}", value.to_bits()),
                )?;
            }
            "OutOfCoverage"
        }
        SpkError::UnsupportedSegmentType { expected, actual } => {
            detail.set_item("expected", *expected)?;
            detail.set_item("actual", *actual)?;
            "UnsupportedSegmentType"
        }
        SpkError::InvalidSegmentLayout { context } => {
            detail.set_item("context", context)?;
            "InvalidSegmentLayout"
        }
        SpkError::UnknownBody { body } => {
            detail.set_item("body", *body)?;
            "UnknownBody"
        }
        SpkError::NoSegmentPath { target, center } => {
            detail.set_item("target", *target)?;
            detail.set_item("center", *center)?;
            "NoSegmentPath"
        }
        SpkError::CoverageGap { target, center, et } => {
            detail.set_item("target", *target)?;
            detail.set_item("center", *center)?;
            detail.set_item("et", *et)?;
            detail.set_item("et_bits_hex", format!("{:016x}", et.to_bits()))?;
            "CoverageGap"
        }
        SpkError::UnsupportedStateSegmentType { data_type } => {
            detail.set_item("data_type", *data_type)?;
            "UnsupportedStateSegmentType"
        }
        SpkError::NonInertialFrameRotation { from, to } => {
            detail.set_item("from", *from)?;
            detail.set_item("to", *to)?;
            "NonInertialFrameRotation"
        }
    };
    detail.set_item("kind", kind)?;
    Ok(detail)
}

fn attach_spk_detail(py: Python<'_>, py_err: &PyErr, err: &SpkError) -> PyResult<()> {
    py_err.value(py).setattr("detail", spk_detail(py, err)?)
}

/// Map an SPK/DAF parse failure into [`SpkParseError`](crate::SpkParseError),
/// preserving the engine message and exact variant payload.
fn to_spk_err(py: Python<'_>, err: SpkError) -> PyErr {
    let py_err = crate::SpkParseError::new_err(err.to_string());
    if let Err(error) = attach_spk_detail(py, &py_err, &err) {
        return error;
    }
    py_err
}

/// Translate a state-query [`SpkError`] into the right Python exception.
///
/// A request that names a body absent from the kernel, or two bodies with no
/// connecting segment chain, is bad input -> `ValueError`. Everything else
/// (no chain covers the epoch, an unsupported segment type on the path, a
/// malformed segment) is a query the loaded kernel cannot satisfy ->
/// `SolveError`, mirroring how the SP3 binding splits `UnknownSatellite` from
/// the rest.
fn state_query_err(py: Python<'_>, err: SpkError) -> PyErr {
    let py_err = match &err {
        SpkError::UnknownBody { body } => {
            PyValueError::new_err(format!("body {body} is not present in any kernel segment"))
        }
        SpkError::NoSegmentPath { target, center } => PyValueError::new_err(format!(
            "no SPK segment path connects target {target} to center {center}"
        )),
        other => to_solve_err(other.to_string()),
    };
    if let Err(error) = attach_spk_detail(py, &py_err, &err) {
        return error;
    }
    py_err
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
/// `velocity_km_s` is the relative velocity (km/s), which every supported
/// segment type yields (for Type 2, the derivative of the position Chebyshev
/// expansion, as CSPICE `SPKE02` forms it). `frame` is the NAIF frame of the
/// state: the one requested with `state_in_frame`, or for `state` the frame of
/// the first segment the query evaluated. Returned by [`Spk.state`].
#[pyclass(module = "sidereon._sidereon", name = "SpkState")]
pub struct PySpkState {
    target: i32,
    center: i32,
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
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
    /// km/s.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.velocity_km_s)
    }

    /// NAIF reference-frame identifier of the position and velocity.
    #[getter]
    fn frame(&self) -> i32 {
        self.frame
    }

    fn __repr__(&self) -> String {
        format!(
            "SpkState(target={}, center={}, position_km=[{}, {}, {}], velocity_km_s=[{}, {}, {}], frame={})",
            self.target,
            self.center,
            self.position_km[0],
            self.position_km[1],
            self.position_km[2],
            self.velocity_km_s[0],
            self.velocity_km_s[1],
            self.velocity_km_s[2],
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
#[derive(Clone)]
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
    fn state(&self, py: Python<'_>, target: i32, center: i32, et: f64) -> PyResult<PySpkState> {
        let state = self
            .inner
            .spk_state(target, center, et)
            .map_err(|err| state_query_err(py, err))?;
        Ok(PySpkState::from_state(state))
    }

    /// Query the state of `target` relative to `center` at `et` in the NAIF
    /// frame `frame`, rotating legs in other NAIF inertial frames (1-21) with
    /// the constant `IRFROT` rotations. A rotation that involves any other
    /// frame raises `SolveError`.
    fn state_in_frame(
        &self,
        py: Python<'_>,
        target: i32,
        center: i32,
        et: f64,
        frame: i32,
    ) -> PyResult<PySpkState> {
        let state = self
            .inner
            .spk_state_in_frame(target, center, et, frame)
            .map_err(|err| state_query_err(py, err))?;
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

/// SPK kernels in load order, queried with the NAIF segment-priority rules: a
/// segment of a later-loaded kernel takes precedence over every segment of an
/// earlier one, and within a kernel a later segment over an earlier one, as
/// CSPICE applies them across the files loaded with `FURNSH`.
#[pyclass(module = "sidereon._sidereon", name = "SpkKernels")]
pub struct PySpkKernels {
    inner: SpkKernels,
}

#[pymethods]
impl PySpkKernels {
    /// Build a kernel set from kernels in load order (lowest precedence
    /// first).
    #[new]
    #[pyo3(signature = (kernels=Vec::new()))]
    fn new(kernels: Vec<PySpk>) -> Self {
        let mut inner = SpkKernels::new();
        for kernel in kernels {
            inner.push(kernel.inner);
        }
        Self { inner }
    }

    /// Add a kernel; it takes precedence over every kernel added before it.
    fn push(&mut self, kernel: PySpk) {
        self.inner.push(kernel.inner);
    }

    /// The kernels in load order (lowest precedence first).
    #[getter]
    fn kernels(&self) -> Vec<PySpk> {
        self.inner
            .kernels()
            .iter()
            .cloned()
            .map(|inner| PySpk { inner })
            .collect()
    }

    /// Query the state of `target` relative to `center` at `et` across every
    /// kernel held, in the frame of the first segment evaluated.
    fn state(&self, py: Python<'_>, target: i32, center: i32, et: f64) -> PyResult<PySpkState> {
        let state = self
            .inner
            .spk_state(target, center, et)
            .map_err(|err| state_query_err(py, err))?;
        Ok(PySpkState::from_state(state))
    }

    /// Query the state of `target` relative to `center` at `et` across every
    /// kernel held, in the NAIF frame `frame`.
    fn state_in_frame(
        &self,
        py: Python<'_>,
        target: i32,
        center: i32,
        et: f64,
        frame: i32,
    ) -> PyResult<PySpkState> {
        let state = self
            .inner
            .spk_state_in_frame(target, center, et, frame)
            .map_err(|err| state_query_err(py, err))?;
        Ok(PySpkState::from_state(state))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("SpkKernels(kernels={})", self.inner.len())
    }
}

/// The name of a NAIF inertial frame (ids 1-21), or `None` for any other id.
#[pyfunction]
fn spk_inertial_frame_name(frame: i32) -> Option<&'static str> {
    core_inertial_frame_name(frame)
}

/// The constant rotation from NAIF inertial frame `from_frame` to `to_frame`
/// (ids 1-21), as SPICELIB `IRFROT` builds it, as a numpy `(3, 3)` array.
/// Raises `ValueError` for a frame outside 1-21.
#[pyfunction]
fn spk_inertial_frame_rotation<'py>(
    py: Python<'py>,
    from_frame: i32,
    to_frame: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rotation = core_inertial_frame_rotation(from_frame, to_frame)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let rows: Vec<Vec<f64>> = rotation.iter().map(|row| row.to_vec()).collect();
    PyArray2::from_vec2(py, &rows).map_err(|err| PyValueError::new_err(err.to_string()))
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
fn load_spk(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<PySpk> {
    // bytes-like first, so a `bytes` argument keeps the "content" meaning.
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        let inner = Spk::from_bytes(bytes.as_bytes()).map_err(|err| to_spk_err(py, err))?;
        return Ok(PySpk { inner });
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        let inner =
            Spk::from_bytes(unsafe { buf.as_bytes() }).map_err(|err| to_spk_err(py, err))?;
        return Ok(PySpk { inner });
    }
    // Otherwise treat it as a path (str / os.PathLike via PyO3's fspath support).
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_spk expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    let inner = Spk::from_bytes(&data).map_err(|err| to_spk_err(py, err))?;
    Ok(PySpk { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.getattr("SpkParseError")?
        .setattr("detail", m.py().None())?;
    m.add_class::<PySpk>()?;
    m.add_class::<PySpkKernels>()?;
    m.add_function(wrap_pyfunction!(spk_inertial_frame_name, m)?)?;
    m.add_function(wrap_pyfunction!(spk_inertial_frame_rotation, m)?)?;
    m.add_class::<PySpkState>()?;
    m.add_class::<PySpkSegment>()?;
    m.add_function(wrap_pyfunction!(load_spk, m)?)?;
    Ok(())
}
