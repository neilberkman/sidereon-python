//! Ephemeris binding: the parsed SP3 precise product, arbitrary-epoch
//! position/clock interpolation, per-record state access, and serialization.
//!
//! Marshals SP3 bytes (or a file path) into [`sidereon_core::ephemeris::Sp3`] and
//! exposes its query surface Pythonically: the node epoch axis as J2000 seconds,
//! batched interpolation to numpy `(n, 3)` / `(n,)` arrays, the exact per-record
//! state, and the deterministic SP3 text writer. No modeling lives here: the
//! interpolation is the engine's `position_at_j2000_seconds` recipe and the writer
//! is `to_sp3_string`, so the numbers and bytes are exactly what `sidereon-core`
//! produces. The per-query loop runs inside Rust, one FFI crossing per call.

use std::path::PathBuf;

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::{PyIndexError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyModule};

use sidereon_core::astro::time::civil::seconds_between_splits;
use sidereon_core::astro::time::{Instant, InstantRepr};
use sidereon_core::constants::J2000_JD;
use sidereon_core::ephemeris::{
    align_clock_reference as core_align_clock_reference,
    clock_reference_offset as core_clock_reference_offset, ClockReferenceOffset, Sp3,
};
use sidereon_core::Error as CoreError;
use sidereon_core::GnssSatelliteId;

use crate::marshal::rows3_to_array;
use crate::{np_array, to_solve_err, to_sp3_err};

/// A parsed SP3 precise-ephemeris product.
///
/// Construct with [`load_sp3`]. Query satellite states by epoch
/// ([`Sp3.interpolate`] for arbitrary epochs, [`Sp3.state`] for the exact parsed
/// records), read the node epoch grid with [`Sp3.epochs_j2000_seconds`], and
/// serialize back to SP3 text with [`Sp3.to_sp3_string`]. Also passed to the
/// solve functions as the ephemeris source. Wraps
/// [`sidereon_core::ephemeris::Sp3`] unchanged.
#[pyclass(module = "sidereon._sidereon", name = "Sp3")]
pub struct PySp3 {
    pub(crate) inner: Sp3,
}

impl PySp3 {
    /// Wrap an owned core product, for the staleness selection layer which hands
    /// back the selected (present or nearest-prior) product.
    pub(crate) fn from_sp3(inner: Sp3) -> Self {
        Self { inner }
    }
}

/// Parse a satellite token (e.g. `"G01"`) into a typed id, raising `ValueError`
/// on a malformed token (bad input, never a domain error).
fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|e| PyValueError::new_err(format!("invalid satellite token {token:?}: {e}")))
}

#[pymethods]
impl PySp3 {
    /// Number of epochs in the product.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.inner.epoch_count()
    }

    /// The satellite tokens (e.g. `"G01"`) present in the product, ascending.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner
            .satellites()
            .iter()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// The product's parsed epochs as seconds since J2000, in the file's own time
    /// scale, ascending, as a numpy `(n,)` `float64` array.
    ///
    /// This is the exact query axis [`Sp3.interpolate`] consumes; read it, form
    /// query times on it (e.g. midpoints, a finer grid), and pass them straight
    /// back without a Julian-date round-trip.
    #[getter]
    fn epochs_j2000_seconds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.epochs_j2000_seconds())
    }

    /// Interpolate `satellite`'s position and clock at each query epoch.
    ///
    /// `j2000_seconds` is a 1-D `float64` array of query times in seconds since
    /// J2000, in the product's own time scale (see
    /// [`epochs_j2000_seconds`](Self::epochs_j2000_seconds)). Returns a
    /// [`Sp3Interpolation`] whose `position_m` is a numpy `(n, 3)` ECEF array in
    /// metres and `clock_s` is a numpy `(n,)` array in seconds (NaN where the
    /// satellite has no clock estimate at that epoch).
    ///
    /// Raises `ValueError` if `satellite` is not present in the product or the
    /// query array is empty, and `SolveError` if a query lies outside the
    /// satellite's coverage (the engine refuses to interpolate across a gap rather
    /// than returning a diverging extrapolation).
    fn interpolate(
        &self,
        satellite: &str,
        j2000_seconds: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PySp3Interpolation> {
        let sat = parse_sat(satellite)?;
        let queries = j2000_seconds
            .as_slice()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        if queries.is_empty() {
            return Err(PyValueError::new_err("j2000_seconds array is empty"));
        }

        let mut positions = Vec::with_capacity(queries.len());
        let mut clocks = Vec::with_capacity(queries.len());
        for &q in queries {
            let state = self.inner.position_at_j2000_seconds(sat, q).map_err(|e| {
                match e {
                    // The satellite simply is not in the product: bad input.
                    CoreError::UnknownSatellite(id) => {
                        PyValueError::new_err(format!("satellite {id} is not in the product"))
                    }
                    // Out of coverage / too few nodes: a solve condition.
                    other => to_solve_err(format!("interpolation at j2000 second {q}: {other}")),
                }
            })?;
            positions.push(state.position.as_array());
            clocks.push(state.clock_s.unwrap_or(f64::NAN));
        }
        Ok(PySp3Interpolation { positions, clocks })
    }

    /// The exact parsed state of `satellite` at the record with index
    /// `epoch_index` (no interpolation).
    ///
    /// Returns an [`Sp3State`]. Raises `IndexError` if `epoch_index` is past the
    /// last epoch and `KeyError` if the satellite has no record at that epoch.
    fn state(&self, satellite: &str, epoch_index: usize) -> PyResult<PySp3State> {
        let sat = parse_sat(satellite)?;
        let state = self.inner.state(sat, epoch_index).map_err(|e| match e {
            CoreError::EpochOutOfRange => {
                PyIndexError::new_err(format!("epoch index {epoch_index} out of range"))
            }
            CoreError::UnknownSatellite(id) => PyKeyError::new_err(format!(
                "satellite {id} has no record at epoch {epoch_index}"
            )),
            other => to_solve_err(other.to_string()),
        })?;
        Ok(PySp3State {
            position: state.position.as_array(),
            clock_s: state.clock_s,
            velocity: state.velocity.map(|v| v.as_array()),
            clock_event: state.flags.clock_event,
            clock_predicted: state.flags.clock_predicted,
            maneuver: state.flags.maneuver,
            orbit_predicted: state.flags.orbit_predicted,
        })
    }

    /// Serialize this product to standard SP3 text (the format named by its header
    /// version, `c` or `d`). Pure and deterministic: the same product always
    /// produces byte-identical text, and re-parsing the output round-trips the
    /// epochs, satellites, positions, and clocks.
    fn to_sp3_string(&self) -> String {
        self.inner.to_sp3_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3(epoch_count={}, satellites={})",
            self.inner.epoch_count(),
            self.inner.satellites().len()
        )
    }
}

/// A batch of interpolated SP3 states: `position_m` as a numpy `(n, 3)` ECEF
/// array in metres and `clock_s` as a numpy `(n,)` array in seconds (NaN where no
/// clock estimate exists). Returned by [`Sp3.interpolate`].
#[pyclass(module = "sidereon._sidereon", name = "Sp3Interpolation")]
pub struct PySp3Interpolation {
    positions: Vec<[f64; 3]>,
    clocks: Vec<f64>,
}

#[pymethods]
impl PySp3Interpolation {
    /// Interpolated ECEF positions as a numpy `(n, 3)` array, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.positions)
    }

    /// Interpolated clock offsets as a numpy `(n,)` array, seconds (NaN where the
    /// satellite has no clock estimate at that epoch).
    #[getter]
    fn clock_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.clocks)
    }

    /// Number of query epochs in the batch.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.positions.len()
    }

    fn __len__(&self) -> usize {
        self.positions.len()
    }

    fn __repr__(&self) -> String {
        format!("Sp3Interpolation(epoch_count={})", self.positions.len())
    }
}

/// One epoch's clock-reference datum offset between two SP3 products.
#[pyclass(module = "sidereon._sidereon", name = "Sp3ClockReferenceOffset")]
#[derive(Clone)]
pub struct PySp3ClockReferenceOffset {
    epoch_j2000_seconds: f64,
    offset_s: f64,
    satellites: usize,
}

#[pymethods]
impl PySp3ClockReferenceOffset {
    /// Matched epoch as seconds since J2000 in the product time scale.
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        self.epoch_j2000_seconds
    }

    /// Clock datum offset, seconds, computed as `other - reference`.
    #[getter]
    fn offset_s(&self) -> f64 {
        self.offset_s
    }

    /// Number of common clocked satellites used in the median estimate.
    #[getter]
    fn satellites(&self) -> usize {
        self.satellites
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3ClockReferenceOffset(epoch_j2000_seconds={}, offset_s={}, satellites={})",
            self.epoch_j2000_seconds, self.offset_s, self.satellites
        )
    }
}

impl From<ClockReferenceOffset> for PySp3ClockReferenceOffset {
    fn from(value: ClockReferenceOffset) -> Self {
        Self {
            epoch_j2000_seconds: instant_to_j2000_seconds(&value.epoch).unwrap_or(f64::NAN),
            offset_s: value.offset_s,
            satellites: value.satellites,
        }
    }
}

/// The exact parsed state of one satellite at one SP3 epoch.
///
/// `position_m` is the ECEF position (metres); `clock_s` is the clock offset
/// (seconds) or `None` for the bad-clock sentinel; `velocity_m_s` is the ECEF
/// velocity (metres per second) or `None` for a position-only product. The four
/// status flags are surfaced verbatim from the record.
#[pyclass(module = "sidereon._sidereon", name = "Sp3State")]
pub struct PySp3State {
    position: [f64; 3],
    clock_s: Option<f64>,
    velocity: Option<[f64; 3]>,
    clock_event: bool,
    clock_predicted: bool,
    maneuver: bool,
    orbit_predicted: bool,
}

impl PySp3State {
    /// Build from a core interpolated/parsed state, for the staleness selection
    /// layer's `position_at_j2000_seconds` query.
    pub(crate) fn from_state(state: sidereon_core::ephemeris::Sp3State) -> Self {
        Self {
            position: state.position.as_array(),
            clock_s: state.clock_s,
            velocity: state.velocity.map(|v| v.as_array()),
            clock_event: state.flags.clock_event,
            clock_predicted: state.flags.clock_predicted,
            maneuver: state.flags.maneuver,
            orbit_predicted: state.flags.orbit_predicted,
        }
    }
}

#[pymethods]
impl PySp3State {
    /// ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.position)
    }

    /// Clock offset in seconds, or `None` for the bad-clock sentinel.
    #[getter]
    fn clock_s(&self) -> Option<f64> {
        self.clock_s
    }

    /// ECEF velocity as a numpy `(3,)` array in metres per second, or `None` for a
    /// position-only product.
    #[getter]
    fn velocity_m_s<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.velocity.map(|v| np_array(py, &v))
    }

    /// Clock discontinuity (`E`) flagged at this epoch (clock interpolation across
    /// it is unsafe).
    #[getter]
    fn clock_event(&self) -> bool {
        self.clock_event
    }

    /// The clock is predicted, not fitted.
    #[getter]
    fn clock_predicted(&self) -> bool {
        self.clock_predicted
    }

    /// The satellite was being maneuvered at this epoch.
    #[getter]
    fn maneuver(&self) -> bool {
        self.maneuver
    }

    /// The orbit is predicted, not fitted.
    #[getter]
    fn orbit_predicted(&self) -> bool {
        self.orbit_predicted
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3State(position_m=[{}, {}, {}], clock_s={:?})",
            self.position[0], self.position[1], self.position[2], self.clock_s
        )
    }
}

/// Parse an SP3-c or SP3-d product from in-memory bytes or a file path.
///
/// `source` may be:
/// - `bytes` / `bytearray`: the full, already-decompressed file content, parsed
///   directly; or
/// - a path (`str` or `os.PathLike`): the file is read and parsed.
///
/// Raises [`Sp3ParseError`](crate::Sp3ParseError) on malformed content, `OSError`
/// if the path cannot be read, and `TypeError` if `source` is neither bytes nor a
/// path.
#[pyfunction]
fn load_sp3(source: &Bound<'_, PyAny>) -> PyResult<PySp3> {
    // bytes-like first, so a `bytes` argument keeps the prior "content" meaning.
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        let inner = sidereon::load_sp3(bytes.as_bytes()).map_err(to_sp3_err)?;
        return Ok(PySp3 { inner });
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        let inner = sidereon::load_sp3(unsafe { buf.as_bytes() }).map_err(to_sp3_err)?;
        return Ok(PySp3 { inner });
    }
    // Otherwise treat it as a path (str / os.PathLike via PyO3's fspath support).
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_sp3 expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    let inner = sidereon::load_sp3(&data).map_err(to_sp3_err)?;
    Ok(PySp3 { inner })
}

/// Estimate per-epoch clock-reference offsets of `other` relative to
/// `reference`.
#[pyfunction]
#[pyo3(signature = (reference, other, min_common=3))]
fn sp3_clock_reference_offset(
    reference: &PySp3,
    other: &PySp3,
    min_common: usize,
) -> Vec<PySp3ClockReferenceOffset> {
    core_clock_reference_offset(&reference.inner, &other.inner, min_common)
        .into_iter()
        .map(PySp3ClockReferenceOffset::from)
        .collect()
}

/// Return a copy of `other` with clocks aligned to `reference` where possible.
#[pyfunction]
#[pyo3(signature = (reference, other, min_common=3))]
fn align_sp3_clock_reference(reference: &PySp3, other: &PySp3, min_common: usize) -> PySp3 {
    PySp3 {
        inner: core_align_clock_reference(&reference.inner, &other.inner, min_common),
    }
}

fn instant_to_j2000_seconds(epoch: &Instant) -> Option<f64> {
    match epoch.repr {
        InstantRepr::JulianDate(jd) => Some(seconds_between_splits(
            jd.jd_whole,
            jd.fraction,
            J2000_JD,
            0.0,
        )),
        InstantRepr::Nanos(_) => None,
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySp3>()?;
    m.add_class::<PySp3Interpolation>()?;
    m.add_class::<PySp3ClockReferenceOffset>()?;
    m.add_class::<PySp3State>()?;
    m.add_function(wrap_pyfunction!(load_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(sp3_clock_reference_offset, m)?)?;
    m.add_function(wrap_pyfunction!(align_sp3_clock_reference, m)?)?;
    Ok(())
}
