//! Frames + time binding: scale-tagged instants, batched coordinate transforms,
//! sidereal time, precession/nutation, and the GNSS week/TOW + leap-second
//! value types.
//!
//! Marshals numpy state arrays plus a unix-microsecond epoch grid into the core's
//! [`sidereon_core::astro::frames`] compute functions and the precise
//! [`sidereon_core::astro::time`] scales. No modeling lives here: every instant's
//! scales come from the parity-critical [`UtcInstant::time_scales`] path and every
//! transform is the engine's own `*_compute`, so the numbers are bit-identical to
//! what `sidereon-core` produces. Each batched call crosses the FFI boundary once,
//! with the per-row loop inside Rust.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon::passes::UtcInstant;
use sidereon_core::astro::frames::nutation::{
    build_skyfield_nutation_matrix, skyfield_iau2000a_radians, skyfield_mean_obliquity_radians,
};
use sidereon_core::astro::frames::precession::compute_skyfield_precession_matrix;
use sidereon_core::astro::frames::transforms::{
    gcrs_to_itrs_compute, geodetic_to_itrs, greenwich_apparent_sidereal_time_radians,
    greenwich_mean_sidereal_time_radians, itrs_to_gcrs_compute, itrs_to_geodetic_compute,
    teme_to_gcrs_compute, TemeStateKm,
};
use sidereon_core::astro::time::civil::{
    day_of_year as core_day_of_year, j2000_seconds as core_j2000_seconds,
    second_of_day as core_second_of_day, split_julian_date as core_split_julian_date,
};
use sidereon_core::astro::time::scales::{
    find_leap_seconds, julian_day_number, leap_second_table, timescale_offset_at_s,
    timescale_offset_s, ut1_coverage, TimeOffsetError, TimeOffsetErrorCode,
};
use sidereon_core::astro::time::{GnssWeekTow, TimeScale, TimeScales};
use sidereon_core::constants::{MICROSECONDS_PER_SECOND, SECONDS_PER_DAY};

use crate::marshal::{
    mat3_to_array, rows3_from_array, rows3_to_array, time_scales_from_unix_micros, EmptyPolicy,
    FinitePolicy,
};
use crate::np_array;

/// A named time scale, mirroring [`sidereon_core::astro::time::TimeScale`].
#[pyclass(module = "sidereon._sidereon", name = "TimeScale", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTimeScale {
    /// Coordinated Universal Time.
    UTC,
    /// International Atomic Time.
    TAI,
    /// Terrestrial Time.
    TT,
    /// Barycentric Dynamical Time.
    TDB,
    /// GPS time.
    GPST,
    /// Galileo System Time.
    GST,
    /// BeiDou Time.
    BDT,
    /// GLONASS system time (UTC(SU)-based; offset is leap-second dependent).
    GLONASST,
    /// QZSS system time (steered synchronous with GPST).
    QZSST,
}

impl From<TimeScale> for PyTimeScale {
    fn from(scale: TimeScale) -> Self {
        match scale {
            TimeScale::Utc => PyTimeScale::UTC,
            TimeScale::Tai => PyTimeScale::TAI,
            TimeScale::Tt => PyTimeScale::TT,
            TimeScale::Tdb => PyTimeScale::TDB,
            TimeScale::Gpst => PyTimeScale::GPST,
            TimeScale::Gst => PyTimeScale::GST,
            TimeScale::Bdt => PyTimeScale::BDT,
            TimeScale::Glonasst => PyTimeScale::GLONASST,
            TimeScale::Qzsst => PyTimeScale::QZSST,
        }
    }
}

impl From<PyTimeScale> for TimeScale {
    fn from(scale: PyTimeScale) -> Self {
        match scale {
            PyTimeScale::UTC => TimeScale::Utc,
            PyTimeScale::TAI => TimeScale::Tai,
            PyTimeScale::TT => TimeScale::Tt,
            PyTimeScale::TDB => TimeScale::Tdb,
            PyTimeScale::GPST => TimeScale::Gpst,
            PyTimeScale::GST => TimeScale::Gst,
            PyTimeScale::BDT => TimeScale::Bdt,
            PyTimeScale::GLONASST => TimeScale::Glonasst,
            PyTimeScale::QZSST => TimeScale::Qzsst,
        }
    }
}

#[pymethods]
impl PyTimeScale {
    /// Short uppercase identifier (`"UTC"`, `"TAI"`, ...).
    #[getter]
    fn abbrev(&self) -> &'static str {
        TimeScale::from(*self).abbrev()
    }

    fn __repr__(&self) -> String {
        format!("TimeScale.{}", TimeScale::from(*self).abbrev())
    }
}

/// A two-part Julian date (whole-day boundary plus residual fraction), mirroring
/// [`sidereon_core::astro::time::JulianDateSplit`]. Carrying the integer day
/// separately preserves sub-microsecond precision across the Julian-date range.
#[pyclass(module = "sidereon._sidereon", name = "JulianDate")]
#[derive(Clone, Copy)]
pub struct PyJulianDate {
    whole: f64,
    fraction: f64,
}

#[pymethods]
impl PyJulianDate {
    /// Integer day boundary (typically `*.0` or `*.5`).
    #[getter]
    fn whole(&self) -> f64 {
        self.whole
    }

    /// Residual day fraction relative to `whole`.
    #[getter]
    fn fraction(&self) -> f64 {
        self.fraction
    }

    /// The recombined single-`float` Julian date (`whole + fraction`).
    #[getter]
    fn jd(&self) -> f64 {
        self.whole + self.fraction
    }

    fn __repr__(&self) -> String {
        format!(
            "JulianDate(whole={}, fraction={})",
            self.whole, self.fraction
        )
    }

    fn __eq__(&self, other: &PyJulianDate) -> bool {
        self.whole == other.whole && self.fraction == other.fraction
    }
}

/// A point in time, tagged UTC, with the precise time scales resolved.
///
/// Construct from a unix-microsecond UTC stamp (matching the epoch convention
/// used elsewhere in the binding) or from UTC calendar fields. The resolved
/// TT/UT1/TDB Julian dates, sidereal time, and precession/nutation are exposed as
/// read-only properties and methods, all from the engine's parity-critical
/// pipeline.
#[pyclass(module = "sidereon._sidereon", name = "Instant")]
#[derive(Clone, Copy)]
pub struct PyInstant {
    unix_micros: i64,
}

impl PyInstant {
    pub(crate) fn time_scales(&self) -> TimeScales {
        UtcInstant::from_unix_microseconds(self.unix_micros).time_scales()
    }
}

#[pymethods]
impl PyInstant {
    /// Build an instant from a unix-microsecond UTC stamp.
    #[staticmethod]
    fn from_unix_micros(unix_micros: i64) -> Self {
        Self { unix_micros }
    }

    /// Build an instant from UTC calendar fields. `second` may be fractional and
    /// is held to microsecond resolution. Raises `ValueError` on an out-of-range
    /// calendar field.
    #[staticmethod]
    #[pyo3(signature = (year, month, day, hour=0, minute=0, second=0.0))]
    fn from_utc(
        year: i32,
        month: i32,
        day: i32,
        hour: i32,
        minute: i32,
        second: f64,
    ) -> PyResult<Self> {
        if !second.is_finite() || second < 0.0 {
            return Err(PyValueError::new_err(
                "second must be finite and non-negative",
            ));
        }
        let whole_second = second.trunc() as i32;
        let microsecond = ((second.fract()) * MICROSECONDS_PER_SECOND).round() as i32;
        let instant =
            UtcInstant::from_utc(year, month, day, hour, minute, whole_second, microsecond)
                .ok_or_else(|| PyValueError::new_err("UTC calendar field out of range"))?;
        Ok(Self {
            unix_micros: instant.unix_microseconds(),
        })
    }

    /// The unix-microsecond UTC stamp backing this instant.
    #[getter]
    fn unix_micros(&self) -> i64 {
        self.unix_micros
    }

    /// The shared integer Julian-day boundary (TAI-aligned).
    #[getter]
    fn jd_whole(&self) -> f64 {
        self.time_scales().jd_whole
    }

    /// Full Terrestrial Time (TT) Julian date.
    #[getter]
    fn tt_jd(&self) -> f64 {
        self.time_scales().jd_tt
    }

    /// Full UT1 Julian date.
    #[getter]
    fn ut1_jd(&self) -> f64 {
        self.time_scales().jd_ut1
    }

    /// Full Barycentric Dynamical Time (TDB) Julian date.
    #[getter]
    fn tdb_jd(&self) -> f64 {
        self.time_scales().jd_tdb
    }

    /// TT day fraction relative to [`jd_whole`](Self::jd_whole).
    #[getter]
    fn tt_fraction(&self) -> f64 {
        self.time_scales().tt_fraction
    }

    /// UT1 day fraction relative to [`jd_whole`](Self::jd_whole).
    #[getter]
    fn ut1_fraction(&self) -> f64 {
        self.time_scales().ut1_fraction
    }

    /// TDB day fraction relative to [`jd_whole`](Self::jd_whole).
    #[getter]
    fn tdb_fraction(&self) -> f64 {
        self.time_scales().tdb_fraction
    }

    /// The two-part TT Julian date (`jd_whole`, `tt_fraction`).
    #[getter]
    fn tt_jd_split(&self) -> PyJulianDate {
        let ts = self.time_scales();
        PyJulianDate {
            whole: ts.jd_whole,
            fraction: ts.tt_fraction,
        }
    }

    /// The two-part UT1 Julian date (`jd_whole`, `ut1_fraction`).
    #[getter]
    fn ut1_jd_split(&self) -> PyJulianDate {
        let ts = self.time_scales();
        PyJulianDate {
            whole: ts.jd_whole,
            fraction: ts.ut1_fraction,
        }
    }

    /// The two-part TDB Julian date (`jd_whole`, `tdb_fraction`).
    #[getter]
    fn tdb_jd_split(&self) -> PyJulianDate {
        let ts = self.time_scales();
        PyJulianDate {
            whole: ts.jd_whole,
            fraction: ts.tdb_fraction,
        }
    }

    /// Delta-T (TT minus UT1) in seconds at this instant.
    #[getter]
    fn delta_t_seconds(&self) -> f64 {
        let ts = self.time_scales();
        (ts.tt_fraction - ts.ut1_fraction) * SECONDS_PER_DAY
    }

    /// IAU mean obliquity of the ecliptic, radians.
    #[getter]
    fn mean_obliquity_radians(&self) -> PyResult<f64> {
        skyfield_mean_obliquity_radians(self.time_scales().jd_tdb)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Greenwich Mean Sidereal Time, radians in `[0, 2pi)`.
    fn gmst_radians(&self) -> PyResult<f64> {
        greenwich_mean_sidereal_time_radians(&self.time_scales())
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Greenwich Apparent Sidereal Time, radians in `[0, 2pi)`.
    fn gast_radians(&self) -> PyResult<f64> {
        greenwich_apparent_sidereal_time_radians(&self.time_scales())
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// IAU 2000A nutation in longitude and obliquity `(dpsi, deps)`, radians.
    fn nutation_angles(&self) -> PyResult<(f64, f64)> {
        skyfield_iau2000a_radians(self.time_scales().jd_tt)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// IAU 2006 precession rotation matrix, as a numpy `(3, 3)` array.
    fn precession_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let matrix = compute_skyfield_precession_matrix(self.time_scales().jd_tdb)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(mat3_to_array(py, &matrix))
    }

    /// IAU 2000A nutation rotation matrix, as a numpy `(3, 3)` array.
    fn nutation_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let ts = self.time_scales();
        let (dpsi, deps) = skyfield_iau2000a_radians(ts.jd_tt)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let mean_ob = skyfield_mean_obliquity_radians(ts.jd_tdb)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let matrix = build_skyfield_nutation_matrix(mean_ob, mean_ob + deps, dpsi)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(mat3_to_array(py, &matrix))
    }

    fn __repr__(&self) -> String {
        format!(
            "Instant(unix_micros={}, tt_jd={})",
            self.unix_micros,
            self.time_scales().jd_tt
        )
    }

    fn __eq__(&self, other: &PyInstant) -> bool {
        self.unix_micros == other.unix_micros
    }
}

/// A GNSS week number plus time-of-week, tagged by constellation, mirroring
/// [`sidereon_core::astro::time::GnssWeekTow`].
#[pyclass(module = "sidereon._sidereon", name = "GnssWeekTow")]
#[derive(Clone, Copy)]
pub struct PyGnssWeekTow {
    inner: GnssWeekTow,
}

#[pymethods]
impl PyGnssWeekTow {
    #[new]
    fn new(system: PyTimeScale, week: u32, tow_s: f64) -> PyResult<Self> {
        Ok(Self {
            inner: GnssWeekTow::new(system.into(), week, tow_s)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        })
    }

    /// The constellation/system whose week/TOW convention this uses.
    #[getter]
    fn system(&self) -> PyTimeScale {
        self.inner.system.into()
    }

    /// Week number (constellation-native, may have rolled over).
    #[getter]
    fn week(&self) -> u32 {
        self.inner.week
    }

    /// Time of week in seconds, nominally `[0, 604800)`.
    #[getter]
    fn tow_s(&self) -> f64 {
        self.inner.tow_s
    }

    /// Normalize so `tow_s` lands in `[0, 604800)`, carrying whole weeks into
    /// `week`. Negative `tow_s` borrows from the week count.
    fn normalized(&self) -> PyResult<PyGnssWeekTow> {
        Ok(PyGnssWeekTow {
            inner: self
                .inner
                .normalized()
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        })
    }

    /// Apply a 1024-week rollover count to recover the continuous week number.
    fn unrolled_week(&self, rollovers: u32) -> PyResult<u32> {
        self.inner
            .unrolled_week(rollovers)
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "GnssWeekTow(system=TimeScale.{}, week={}, tow_s={})",
            self.inner.system.abbrev(),
            self.inner.week,
            self.inner.tow_s
        )
    }

    fn __eq__(&self, other: &PyGnssWeekTow) -> bool {
        self.inner.system == other.inner.system
            && self.inner.week == other.inner.week
            && self.inner.tow_s == other.inner.tow_s
    }
}

/// Provenance and coverage of the embedded IERS leap-second (TAI-UTC) table.
#[pyclass(module = "sidereon._sidereon", name = "LeapSecondTable")]
pub struct PyLeapSecondTable {
    source: &'static str,
    first_mjd: i32,
    last_mjd: i32,
    entries: usize,
}

#[pymethods]
impl PyLeapSecondTable {
    /// Human-readable provenance string for the table.
    #[getter]
    fn source(&self) -> &'static str {
        self.source
    }

    /// Modified Julian date of the first table entry.
    #[getter]
    fn first_mjd(&self) -> i32 {
        self.first_mjd
    }

    /// Modified Julian date of the last table entry.
    #[getter]
    fn last_mjd(&self) -> i32 {
        self.last_mjd
    }

    /// Number of entries in the table.
    #[getter]
    fn entries(&self) -> usize {
        self.entries
    }

    fn __repr__(&self) -> String {
        format!(
            "LeapSecondTable(entries={}, first_mjd={}, last_mjd={})",
            self.entries, self.first_mjd, self.last_mjd
        )
    }
}

/// Provenance and coverage of the embedded UT1-UTC / delta-T (EOP) table.
#[pyclass(module = "sidereon._sidereon", name = "Ut1Coverage")]
pub struct PyUt1Coverage {
    source: &'static str,
    first_mjd: i32,
    last_mjd: i32,
    first_jd_tt: f64,
    last_jd_tt: f64,
    entries: usize,
}

#[pymethods]
impl PyUt1Coverage {
    /// Human-readable provenance string for the table.
    #[getter]
    fn source(&self) -> &'static str {
        self.source
    }

    /// Modified Julian date of the first table entry.
    #[getter]
    fn first_mjd(&self) -> i32 {
        self.first_mjd
    }

    /// Modified Julian date of the last table entry.
    #[getter]
    fn last_mjd(&self) -> i32 {
        self.last_mjd
    }

    /// TT Julian date of the first table entry (coverage lower bound).
    #[getter]
    fn first_jd_tt(&self) -> f64 {
        self.first_jd_tt
    }

    /// TT Julian date of the last table entry (coverage upper bound).
    #[getter]
    fn last_jd_tt(&self) -> f64 {
        self.last_jd_tt
    }

    /// Number of entries in the table.
    #[getter]
    fn entries(&self) -> usize {
        self.entries
    }

    fn __repr__(&self) -> String {
        format!(
            "Ut1Coverage(entries={}, first_mjd={}, last_mjd={})",
            self.entries, self.first_mjd, self.last_mjd
        )
    }
}

/// TAI-UTC (cumulative leap seconds) in effect on a UTC calendar date.
///
/// Composes the engine's `julian_day_number` and `find_leap_seconds`; the lookup
/// uses the date only (leap seconds change at a day boundary). Returns the value
/// from the embedded IERS table, clamped at the table edges exactly as the
/// engine's own time-scale path does.
#[pyfunction]
fn leap_seconds(year: i32, month: i32, day: i32) -> f64 {
    // jd at UTC midnight of the given date: matches TimeScales::from_utc's jd1.
    let jd_utc_midnight = julian_day_number(year, month, day) as f64 - 0.5;
    find_leap_seconds(jd_utc_midnight)
}

#[pyfunction]
fn split_julian_date(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
) -> PyJulianDate {
    let (whole, fraction) = core_split_julian_date(year, month, day, hour, minute, second);
    PyJulianDate { whole, fraction }
}

#[pyfunction]
fn j2000_seconds(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
    core_j2000_seconds(year, month, day, hour, minute, second)
}

#[pyfunction]
fn second_of_day(hour: i32, minute: i32, second: f64) -> f64 {
    core_second_of_day(hour, minute, second)
}

#[pyfunction]
fn day_of_year(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
    core_day_of_year(year, month, day, hour, minute, second)
}

/// Provenance and coverage of the embedded leap-second (TAI-UTC) table.
#[pyfunction]
fn leap_second_table_info() -> PyLeapSecondTable {
    let table = leap_second_table();
    PyLeapSecondTable {
        source: table.source,
        first_mjd: table.first_mjd,
        last_mjd: table.last_mjd,
        entries: table.entries,
    }
}

/// Provenance and coverage of the embedded UT1-UTC / delta-T (EOP) table.
#[pyfunction]
fn ut1_coverage_info() -> PyUt1Coverage {
    let prov = ut1_coverage();
    PyUt1Coverage {
        source: prov.source,
        first_mjd: prov.first_mjd,
        last_mjd: prov.last_mjd,
        first_jd_tt: prov.first_jd_tt,
        last_jd_tt: prov.last_jd_tt,
        entries: prov.entries,
    }
}

/// Stable machine-readable discriminant for a time-scale offset failure,
/// mirroring [`sidereon_core::astro::time::scales::TimeOffsetErrorCode`].
///
/// Surfaced as the `.code` attribute on the `ValueError` raised by
/// [`timescale_offset`] / [`timescale_offset_at`], so a caller can branch on the
/// failure kind without parsing the message text.
#[pyclass(
    module = "sidereon._sidereon",
    name = "TimeOffsetErrorCode",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum PyTimeOffsetErrorCode {
    /// The scale is UTC-based; its offset is epoch-dependent (use
    /// `timescale_offset_at`).
    EPOCH_REQUIRED = 1,
    /// The scale (TDB) has no fixed/constant offset.
    UNSUPPORTED = 2,
    /// A leap-aware query received a non-finite UTC Julian date.
    NON_FINITE_EPOCH = 3,
}

impl From<TimeOffsetErrorCode> for PyTimeOffsetErrorCode {
    fn from(code: TimeOffsetErrorCode) -> Self {
        match code {
            TimeOffsetErrorCode::EpochRequired => Self::EPOCH_REQUIRED,
            TimeOffsetErrorCode::Unsupported => Self::UNSUPPORTED,
            TimeOffsetErrorCode::NonFiniteEpoch => Self::NON_FINITE_EPOCH,
        }
    }
}

#[pymethods]
impl PyTimeOffsetErrorCode {
    /// Stable lowercase selector.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::EPOCH_REQUIRED => "epoch_required",
            Self::UNSUPPORTED => "unsupported",
            Self::NON_FINITE_EPOCH => "non_finite_epoch",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::EPOCH_REQUIRED => "TimeOffsetErrorCode.EPOCH_REQUIRED",
            Self::UNSUPPORTED => "TimeOffsetErrorCode.UNSUPPORTED",
            Self::NON_FINITE_EPOCH => "TimeOffsetErrorCode.NON_FINITE_EPOCH",
        }
    }
}

/// Map a core [`TimeOffsetError`] onto a `ValueError` that also carries the
/// machine-readable [`PyTimeOffsetErrorCode`] as its `.code` attribute.
fn time_offset_err(py: Python<'_>, err: TimeOffsetError) -> PyErr {
    let exc = PyValueError::new_err(err.to_string());
    let code = PyTimeOffsetErrorCode::from(err.code());
    let _ = exc.value(py).setattr("code", code);
    exc
}

/// Fixed inter-system time-scale offset `to - from` in seconds.
///
/// Returns the value that, added to a `from`-scale reading, yields the
/// `to`-scale reading of the same instant. Defined for the atomic scales
/// (TAI/TT/GPST/GST/QZSST/BDT) whose mutual offsets are constants. Raises
/// `ValueError` for the UTC-based scales (UTC/GLONASST) whose offset is
/// leap-second dependent, use [`timescale_offset_at`] with an epoch, and for
/// TDB (epoch-dependent periodic term). The raised `ValueError` carries a
/// `.code` attribute (a `TimeOffsetErrorCode`).
#[pyfunction]
fn timescale_offset(py: Python<'_>, from_: PyTimeScale, to: PyTimeScale) -> PyResult<f64> {
    timescale_offset_s(from_.into(), to.into()).map_err(|e| time_offset_err(py, e))
}

/// Leap-aware inter-system time-scale offset `to - from` in seconds at a UTC
/// instant.
///
/// `utc_jd` is the UTC Julian date of the instant, used to resolve the
/// leap-second count when `from` or `to` is UTC-based (UTC/GLONASST); it is
/// ignored for purely atomic pairs. The result, added to a `from`-scale
/// reading, yields the `to`-scale reading of the same instant. Raises
/// `ValueError` for TDB or a non-finite `utc_jd` with a UTC-based scale.
#[pyfunction]
fn timescale_offset_at(
    py: Python<'_>,
    from_: PyTimeScale,
    to: PyTimeScale,
    utc_jd: f64,
) -> PyResult<f64> {
    timescale_offset_at_s(from_.into(), to.into(), utc_jd).map_err(|e| time_offset_err(py, e))
}

/// Require that a position array's row count matches the epoch count.
fn check_lengths(positions: &[[f64; 3]], epochs: usize) -> PyResult<()> {
    if positions.len() != epochs {
        return Err(PyValueError::new_err(format!(
            "positions ({}) and epochs ({epochs}) must have the same length",
            positions.len()
        )));
    }
    Ok(())
}

/// A batch of transformed states: `position_km` and `velocity_km_s` as numpy
/// `float64` arrays of shape `(n, 3)`. Returned by [`teme_to_gcrs`].
#[pyclass(module = "sidereon._sidereon", name = "FrameStates")]
pub struct PyFrameStates {
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
}

#[pymethods]
impl PyFrameStates {
    /// Transformed positions as a numpy `(n, 3)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.positions)
    }

    /// Transformed velocities as a numpy `(n, 3)` array, km/s.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.velocities)
    }

    /// Number of states in the batch.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.positions.len()
    }

    fn __len__(&self) -> usize {
        self.positions.len()
    }

    fn __repr__(&self) -> String {
        format!("FrameStates(epoch_count={})", self.positions.len())
    }
}

/// Transform a batch of TEME states to GCRS, each at its own epoch.
///
/// `position_km` and `velocity_km_s` are numpy `(n, 3)` arrays; `epochs_unix_us`
/// is a 1-D `int64` array of unix-microsecond UTC stamps. Returns a
/// [`FrameStates`] whose `position_km` / `velocity_km_s` are numpy `(n, 3)` arrays
/// in GCRS. When `skyfield_compat` is true (default) the FMA / AU-scaled
/// Skyfield-parity path is used; false uses the direct km path.
#[pyfunction]
#[pyo3(signature = (position_km, velocity_km_s, epochs_unix_us, *, skyfield_compat=true))]
fn teme_to_gcrs(
    position_km: PyReadonlyArray2<'_, f64>,
    velocity_km_s: PyReadonlyArray2<'_, f64>,
    epochs_unix_us: PyReadonlyArray1<'_, i64>,
    skyfield_compat: bool,
) -> PyResult<PyFrameStates> {
    let positions = rows3_from_array(
        "position_km",
        &position_km,
        EmptyPolicy::Allow,
        FinitePolicy::AllowNonFinite,
    )?;
    let velocities = rows3_from_array(
        "velocity_km_s",
        &velocity_km_s,
        EmptyPolicy::Allow,
        FinitePolicy::AllowNonFinite,
    )?;
    let scales = time_scales_from_unix_micros(&epochs_unix_us, EmptyPolicy::Reject)?;
    check_lengths(&positions, scales.len())?;
    check_lengths(&velocities, scales.len())?;

    let mut out_pos = Vec::with_capacity(scales.len());
    let mut out_vel = Vec::with_capacity(scales.len());
    for ((pos, vel), ts) in positions.iter().zip(velocities.iter()).zip(scales.iter()) {
        let (p, v) = teme_to_gcrs_compute(
            &TemeStateKm {
                position_km: *pos,
                velocity_km_s: *vel,
            },
            ts,
            skyfield_compat,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
        out_pos.push([p.0, p.1, p.2]);
        out_vel.push([v.0, v.1, v.2]);
    }
    Ok(PyFrameStates {
        positions: out_pos,
        velocities: out_vel,
    })
}

/// Transform a batch of GCRS positions to ITRS (Earth-fixed / ECEF), each at its
/// own epoch.
///
/// `position_km` is a numpy `(n, 3)` array; `epochs_unix_us` is a 1-D `int64`
/// array of unix-microsecond UTC stamps. Returns a numpy `(n, 3)` ITRS array.
/// `skyfield_compat` (default true) selects the AU-scaled Skyfield-parity path.
#[pyfunction]
#[pyo3(signature = (position_km, epochs_unix_us, *, skyfield_compat=true))]
fn gcrs_to_itrs<'py>(
    py: Python<'py>,
    position_km: PyReadonlyArray2<'_, f64>,
    epochs_unix_us: PyReadonlyArray1<'_, i64>,
    skyfield_compat: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let positions = rows3_from_array(
        "position_km",
        &position_km,
        EmptyPolicy::Allow,
        FinitePolicy::AllowNonFinite,
    )?;
    let scales = time_scales_from_unix_micros(&epochs_unix_us, EmptyPolicy::Reject)?;
    check_lengths(&positions, scales.len())?;

    let out: Vec<[f64; 3]> = positions
        .iter()
        .zip(scales.iter())
        .map(|(p, ts)| {
            gcrs_to_itrs_compute(p[0], p[1], p[2], ts, skyfield_compat)
                .map(|(x, y, z)| [x, y, z])
                .map_err(|err| PyValueError::new_err(err.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(rows3_to_array(py, &out))
}

/// Transform a batch of ITRS (ECEF) positions to GCRS, each at its own epoch.
///
/// `position_km` is a numpy `(n, 3)` array; `epochs_unix_us` is a 1-D `int64`
/// array of unix-microsecond UTC stamps. Returns a numpy `(n, 3)` GCRS array.
#[pyfunction]
fn itrs_to_gcrs<'py>(
    py: Python<'py>,
    position_km: PyReadonlyArray2<'_, f64>,
    epochs_unix_us: PyReadonlyArray1<'_, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let positions = rows3_from_array(
        "position_km",
        &position_km,
        EmptyPolicy::Allow,
        FinitePolicy::AllowNonFinite,
    )?;
    let scales = time_scales_from_unix_micros(&epochs_unix_us, EmptyPolicy::Reject)?;
    check_lengths(&positions, scales.len())?;

    let out: Vec<[f64; 3]> = positions
        .iter()
        .zip(scales.iter())
        .map(|(p, ts)| {
            itrs_to_gcrs_compute(p[0], p[1], p[2], ts)
                .map(|(x, y, z)| [x, y, z])
                .map_err(|err| PyValueError::new_err(err.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(rows3_to_array(py, &out))
}

/// Convert a batch of geodetic coordinates to ITRS (ECEF).
///
/// `geodetic` is a numpy `(n, 3)` array whose columns are
/// `[latitude_deg, longitude_deg, altitude_km]` (WGS84). Returns a numpy `(n, 3)`
/// ITRS array in kilometres. Time-independent.
#[pyfunction]
fn geodetic_to_ecef<'py>(
    py: Python<'py>,
    geodetic: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = rows3_from_array(
        "geodetic",
        &geodetic,
        EmptyPolicy::Allow,
        FinitePolicy::AllowNonFinite,
    )?;
    let out: Vec<[f64; 3]> = rows
        .iter()
        .map(|g| {
            geodetic_to_itrs(g[0], g[1], g[2])
                .map(|(x, y, z)| [x, y, z])
                .map_err(|err| PyValueError::new_err(err.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(rows3_to_array(py, &out))
}

/// Convert a batch of ITRS (ECEF) positions to geodetic coordinates.
///
/// `position_km` is a numpy `(n, 3)` array in kilometres. Returns a numpy
/// `(n, 3)` array whose columns are `[latitude_deg, longitude_deg, altitude_km]`
/// (WGS84). Time-independent.
#[pyfunction]
fn ecef_to_geodetic<'py>(
    py: Python<'py>,
    position_km: PyReadonlyArray2<'_, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let rows = rows3_from_array(
        "position_km",
        &position_km,
        EmptyPolicy::Allow,
        FinitePolicy::AllowNonFinite,
    )?;
    let out: Vec<[f64; 3]> = rows
        .iter()
        .map(|p| {
            itrs_to_geodetic_compute(p[0], p[1], p[2])
                .map(|(lat, lon, alt)| [lat, lon, alt])
                .map_err(|err| PyValueError::new_err(err.to_string()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    Ok(rows3_to_array(py, &out))
}

/// Leap seconds for a batch of UTC dates, as a numpy `(n,)` array. Convenience
/// over [`leap_seconds`] for many dates; columns of `dates` are
/// `[year, month, day]`.
#[pyfunction]
fn leap_seconds_batch<'py>(
    py: Python<'py>,
    dates: PyReadonlyArray2<'_, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let view = dates.as_array();
    if view.ncols() != 3 {
        return Err(PyValueError::new_err(
            "dates must have shape (n, 3) with columns [year, month, day]",
        ));
    }
    let values: Vec<f64> = view
        .outer_iter()
        .map(|r| leap_seconds(r[0] as i32, r[1] as i32, r[2] as i32))
        .collect();
    Ok(np_array(py, &values))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTimeScale>()?;
    m.add_class::<PyTimeOffsetErrorCode>()?;
    m.add_class::<PyInstant>()?;
    m.add_class::<PyJulianDate>()?;
    m.add_class::<PyGnssWeekTow>()?;
    m.add_class::<PyLeapSecondTable>()?;
    m.add_class::<PyUt1Coverage>()?;
    m.add_class::<PyFrameStates>()?;
    m.add_function(wrap_pyfunction!(leap_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(split_julian_date, m)?)?;
    m.add_function(wrap_pyfunction!(j2000_seconds, m)?)?;
    m.add_function(wrap_pyfunction!(second_of_day, m)?)?;
    m.add_function(wrap_pyfunction!(day_of_year, m)?)?;
    m.add_function(wrap_pyfunction!(leap_seconds_batch, m)?)?;
    m.add_function(wrap_pyfunction!(leap_second_table_info, m)?)?;
    m.add_function(wrap_pyfunction!(ut1_coverage_info, m)?)?;
    m.add_function(wrap_pyfunction!(timescale_offset, m)?)?;
    m.add_function(wrap_pyfunction!(timescale_offset_at, m)?)?;
    m.add_function(wrap_pyfunction!(teme_to_gcrs, m)?)?;
    m.add_function(wrap_pyfunction!(gcrs_to_itrs, m)?)?;
    m.add_function(wrap_pyfunction!(itrs_to_gcrs, m)?)?;
    m.add_function(wrap_pyfunction!(geodetic_to_ecef, m)?)?;
    m.add_function(wrap_pyfunction!(ecef_to_geodetic, m)?)?;
    Ok(())
}
