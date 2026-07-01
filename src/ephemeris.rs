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
use sidereon_core::astro::time::{Instant, InstantRepr, JulianDateSplit};
use sidereon_core::constants::J2000_JD;
use sidereon_core::ephemeris::{
    align_clock_reference as core_align_clock_reference,
    clock_reference_offset as core_clock_reference_offset, ClockReferenceOffset,
    PreciseEphemerisSample, PreciseEphemerisSamples, Sp3,
};
use sidereon_core::Error as CoreError;
use sidereon_core::GnssSatelliteId;

use crate::frames::PyTimeScale;
use crate::marshal::rows3_to_array;
use crate::{np_array, to_solve_err, to_sp3_err};

/// Seconds in one day, for the J2000-second <-> split-Julian-date reconstruction.
const SECONDS_PER_DAY: f64 = 86_400.0;

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

    /// Extract this product as the canonical precise-ephemeris samples, in SI
    /// units, one per parsed position record in ascending epoch order.
    ///
    /// Round-tripping the result through
    /// [`PreciseEphemerisSamples.from_samples`] rebuilds the same interpolatable
    /// source (byte-identical for samples whose metres are the faithful image of
    /// the fitted kilometres, sub-micron otherwise; see
    /// [`PreciseEphemerisSamples`]).
    fn precise_ephemeris_samples(&self) -> Vec<PyPreciseEphemerisSample> {
        self.inner
            .precise_ephemeris_samples()
            .into_iter()
            .map(PyPreciseEphemerisSample::from)
            .collect()
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

/// One precise-ephemeris sample: a satellite's ECEF position (and optional
/// clock) at one epoch, in SI units.
///
/// This is the canonical serialization-independent element behind
/// [`PreciseEphemerisSamples`]. `position_ecef_m` is the ITRF/IGS ECEF position
/// in metres; `clock_s` is the satellite clock offset in seconds, `None` when
/// the source carried no clock estimate. `clock_event` mirrors the SP3 `E`
/// clock-event flag: `True` marks a clock discontinuity, splitting the
/// interpolated clock arc there. The epoch is carried as seconds since J2000 in
/// the sample's own `time_scale`.
#[pyclass(module = "sidereon._sidereon", name = "PreciseEphemerisSample")]
#[derive(Clone)]
pub struct PyPreciseEphemerisSample {
    inner: PreciseEphemerisSample,
}

impl From<PreciseEphemerisSample> for PyPreciseEphemerisSample {
    fn from(inner: PreciseEphemerisSample) -> Self {
        Self { inner }
    }
}

impl PyPreciseEphemerisSample {
    fn to_core(&self) -> PreciseEphemerisSample {
        self.inner
    }
}

#[pymethods]
impl PyPreciseEphemerisSample {
    /// Build one precise-ephemeris sample.
    ///
    /// `satellite` is a canonical token such as `"G01"`, `epoch_j2000_seconds`
    /// is the sample epoch in `time_scale`, `position_ecef_m` is a length-3 ECEF
    /// position in metres, and `clock_s` is the optional clock offset in seconds.
    /// Set `clock_event=True` to reconstruct an epoch that carries the SP3 `E`
    /// clock reset.
    #[new]
    #[pyo3(signature = (
        satellite,
        epoch_j2000_seconds,
        position_ecef_m,
        clock_s=None,
        *,
        time_scale=PyTimeScale::GPST,
        clock_event=false,
    ))]
    fn new(
        satellite: &str,
        epoch_j2000_seconds: f64,
        position_ecef_m: [f64; 3],
        clock_s: Option<f64>,
        time_scale: PyTimeScale,
        clock_event: bool,
    ) -> PyResult<Self> {
        let sat = parse_sat(satellite)?;
        let epoch = instant_from_j2000_seconds(epoch_j2000_seconds, time_scale.into())?;
        let mut inner = PreciseEphemerisSample::new(sat, epoch, position_ecef_m, clock_s);
        inner.clock_event = clock_event;
        Ok(Self { inner })
    }

    /// Satellite token, e.g. `"G01"`.
    #[getter]
    fn satellite(&self) -> String {
        self.inner.sat.to_string()
    }

    /// Sample epoch as seconds since J2000, in `time_scale`.
    #[getter]
    fn epoch_j2000_seconds(&self) -> f64 {
        instant_to_j2000_seconds(&self.inner.epoch).unwrap_or(f64::NAN)
    }

    /// Time scale the epoch is expressed in.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.epoch.scale.into()
    }

    /// ECEF position as a numpy `(3,)` array, metres.
    #[getter]
    fn position_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_ecef_m)
    }

    /// Clock offset in seconds, or `None` when the source carried no estimate.
    #[getter]
    fn clock_s(&self) -> Option<f64> {
        self.inner.clock_s
    }

    /// Whether this epoch carries the SP3 `E` clock-event flag.
    #[getter]
    fn clock_event(&self) -> bool {
        self.inner.clock_event
    }

    fn __repr__(&self) -> String {
        format!(
            "PreciseEphemerisSample(satellite={:?}, epoch_j2000_seconds={}, clock_s={:?}, clock_event={})",
            self.inner.sat.to_string(),
            self.epoch_j2000_seconds(),
            self.inner.clock_s,
            self.inner.clock_event
        )
    }
}

/// A precise-ephemeris source built from samples rather than parsed SP3 text.
///
/// Construct with [`PreciseEphemerisSamples.from_samples`]. It drives the same
/// interpolation substrate the SP3-parsed product uses, so it can be passed to
/// [`predict_ranges`](crate) as an ephemeris source and yields interpolated
/// states / predicted ranges that match the SP3 path byte-for-byte for samples
/// that are the faithful image of the fitted nodes (the round-trip case), and to
/// within sub-micron precision otherwise.
#[pyclass(module = "sidereon._sidereon", name = "PreciseEphemerisSamples")]
pub struct PyPreciseEphemerisSamples {
    pub(crate) inner: PreciseEphemerisSamples,
}

#[pymethods]
impl PyPreciseEphemerisSamples {
    /// Build a source from a sequence of [`PreciseEphemerisSample`].
    ///
    /// Samples are grouped by satellite in supplied order and validated. Raises
    /// `ValueError` if the set is empty, a satellite has a single sample, a
    /// satellite's epochs are not strictly increasing, the samples mix time
    /// scales, a sample is non-finite, or an epoch is not representable as J2000
    /// seconds.
    #[staticmethod]
    fn from_samples(py: Python<'_>, samples: Vec<Py<PyPreciseEphemerisSample>>) -> PyResult<Self> {
        let samples = samples.iter().map(|s| s.borrow(py).to_core());
        let inner = PreciseEphemerisSamples::from_samples(samples)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Time scale every sample epoch is expressed in.
    #[getter]
    fn time_scale(&self) -> PyTimeScale {
        self.inner.time_scale().into()
    }

    /// Satellite tokens this source can interpolate, ascending.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner.satellites().map(|sat| sat.to_string()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "PreciseEphemerisSamples(satellites={})",
            self.inner.satellites().count()
        )
    }
}

/// Build a split-Julian-date [`Instant`] from J2000 seconds in a time scale.
///
/// The residual day fraction is split off `jd_whole` so it stays within one day,
/// matching the SP3 parser's node axis after the shared floor.
fn instant_from_j2000_seconds(
    seconds: f64,
    scale: sidereon_core::astro::time::TimeScale,
) -> PyResult<Instant> {
    if !seconds.is_finite() {
        return Err(PyValueError::new_err("epoch_j2000_seconds must be finite"));
    }
    let days = seconds / SECONDS_PER_DAY;
    let whole_days = days.floor();
    let split = JulianDateSplit::new(J2000_JD + whole_days, days - whole_days)
        .map_err(|e| PyValueError::new_err(format!("invalid epoch: {e}")))?;
    Ok(Instant {
        scale,
        repr: InstantRepr::JulianDate(split),
    })
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
    m.add_class::<PyPreciseEphemerisSample>()?;
    m.add_class::<PyPreciseEphemerisSamples>()?;
    m.add_function(wrap_pyfunction!(load_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(sp3_clock_reference_offset, m)?)?;
    m.add_function(wrap_pyfunction!(align_sp3_clock_reference, m)?)?;
    Ok(())
}
