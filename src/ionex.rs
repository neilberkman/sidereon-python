//! IONEX binding: the parsed vertical-TEC grid product and its slant
//! ionospheric group-delay query.
//!
//! Marshals IONEX bytes (or a file path) into the core `Ionex` vertical-TEC grid
//! and exposes its surface Pythonically: the latitude/longitude node axes and
//! per-map TEC/RMS grids as numpy arrays, the map epoch axis as J2000 seconds,
//! and a single-call slant-delay query taking degrees in and metres out. No
//! modeling lives here: the parse is `Ionex::parse` and the delay is
//! `ionex_slant_delay`, so the numbers are exactly what `sidereon-core`
//! produces. The degree-to-radian boundary conversion mirrors the Elixir
//! wrapper so the two interfaces report the same value.

use std::f64::consts::PI;
use std::path::PathBuf;

/// Degrees to radians as a single rounded constant `pi/180`, so the boundary
/// conversion is `deg * DEG_TO_RAD` (one multiply, one rounding). This matches
/// the golden's `math.radians` exactly; `(deg * pi) / 180.0` rounds twice and
/// drifts by a ULP at some angles (for example -178 degrees).
const DEG_TO_RAD: f64 = PI / 180.0;

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyModule};

use sidereon_core::astro::time::model::Instant;
use sidereon_core::atmosphere::ionosphere::{
    galileo_nequick_g_native as core_galileo_nequick_g_native,
    ionosphere_delay as core_ionosphere_delay, klobuchar_native as core_klobuchar_native,
    nequick_g_delay_m as core_nequick_g_delay_m, nequick_g_stec_tecu as core_nequick_g_stec_tecu,
    GalileoNequickCoeffs, GalileoNequickEval, IonoModel, KlobucharParams, NequickGRayEval,
};
use sidereon_core::atmosphere::{ionex_slant_delay, Ionex};
use sidereon_core::Wgs84Geodetic;

use crate::{np_array, to_solve_err};

/// Map an IONEX parse failure into [`IonexParseError`](crate::IonexParseError),
/// preserving the engine message. Both it and the other product parse errors
/// derive from `ParseError`, so callers can catch the product-specific type or
/// the shared base.
fn to_ionex_err<E: std::fmt::Display>(err: E) -> PyErr {
    crate::IonexParseError::new_err(err.to_string())
}

/// Build a 3-D numpy `float64` array `(epoch, lat, lon)` from the core's nested
/// `[map][i_lat][i_lon]` grids. Returns a zero-extent array when there are no
/// maps, which the parser never produces for a TEC grid but is well defined.
fn maps_to_array3<'py>(py: Python<'py>, maps: &[Vec<Vec<f64>>]) -> Bound<'py, PyArray3<f64>> {
    let n_epoch = maps.len();
    let n_lat = maps.first().map_or(0, Vec::len);
    let n_lon = maps.first().and_then(|m| m.first()).map_or(0, Vec::len);
    let mut array = Array3::<f64>::zeros((n_epoch, n_lat, n_lon));
    for (epoch_index, grid) in maps.iter().enumerate() {
        for (lat_index, band) in grid.iter().enumerate() {
            for (lon_index, value) in band.iter().enumerate() {
                array[[epoch_index, lat_index, lon_index]] = *value;
            }
        }
    }
    array.into_pyarray(py)
}

/// A parsed IONEX vertical-TEC product.
///
/// Construct with [`load_ionex`]. Read the grid geometry with the
/// `lat_nodes_deg` / `lon_nodes_deg` axes (descending north-to-south,
/// ascending west-to-east), the per-map `tec_maps` / `rms_maps` cubes
/// (`(epoch, lat, lon)`, TECU), and the `map_epochs_j2000_s` epoch axis. Query
/// the line-of-sight delay with [`Ionex.slant_delay`]. Wraps
/// [`sidereon_core::atmosphere::Ionex`] unchanged.
#[pyclass(module = "sidereon._sidereon", name = "Ionex")]
pub struct PyIonex {
    pub(crate) inner: Ionex,
}

impl PyIonex {
    /// Wrap an owned core product, for the staleness selection layer which hands
    /// back the present (or diurnal-shifted) product.
    pub(crate) fn from_ionex(inner: Ionex) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIonex {
    /// Latitude node values in degrees as a numpy `(n_lat,)` `float64` array,
    /// descending (north-to-south).
    #[getter]
    fn lat_nodes_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, self.inner.lat_nodes_deg())
    }

    /// Longitude node values in degrees as a numpy `(n_lon,)` `float64` array,
    /// ascending (west-to-east).
    #[getter]
    fn lon_nodes_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, self.inner.lon_nodes_deg())
    }

    /// Signed latitude step in degrees (negative for the standard ordering).
    #[getter]
    fn dlat_deg(&self) -> f64 {
        self.inner.dlat_deg()
    }

    /// Signed longitude step in degrees (positive for the standard ordering).
    #[getter]
    fn dlon_deg(&self) -> f64 {
        self.inner.dlon_deg()
    }

    /// Single-layer shell height in kilometers.
    #[getter]
    fn shell_height_km(&self) -> f64 {
        self.inner.shell_height_km()
    }

    /// Mean earth radius used by the geometry, in kilometers.
    #[getter]
    fn base_radius_km(&self) -> f64 {
        self.inner.base_radius_km()
    }

    /// The IONEX `EXPONENT` header field; the TEC scale is `10^EXPONENT`.
    #[getter]
    fn exponent(&self) -> i32 {
        self.inner.exponent()
    }

    /// Map epochs as a numpy `(n_epoch,)` `int64` array of seconds since J2000,
    /// ascending. This is the exact axis [`Ionex.slant_delay`] brackets against.
    #[getter]
    fn map_epochs_j2000_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.inner.map_epochs_s())
    }

    /// Per-map vertical-TEC grids as a numpy `(epoch, lat, lon)` `float64` cube
    /// in TECU (after the `10^EXPONENT` scaling).
    #[getter]
    fn tec_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        maps_to_array3(py, self.inner.tec_maps())
    }

    /// Per-map RMS grids as a numpy `(epoch, lat, lon)` `float64` cube in TECU,
    /// or a `(0, 0, 0)` array when the product carries no RMS maps.
    #[getter]
    fn rms_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        maps_to_array3(py, self.inner.rms_maps())
    }

    /// IONEX vertical-TEC-grid slant ionospheric group delay, positive metres.
    ///
    /// Receiver geodetic latitude/longitude and the satellite azimuth/elevation
    /// are in degrees (latitude positive north, longitude positive east, azimuth
    /// clockwise from north); the pierce point rides on the IONEX shell, so no
    /// receiver height enters. `epoch_j2000_s` is an integer number of seconds
    /// since J2000, landing exactly on the product's own epoch axis.
    /// `frequency_hz` is the carrier the dispersive delay is reported on. Raises
    /// `ValueError` on out-of-range or non-finite input.
    #[pyo3(signature = (lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch_j2000_s, frequency_hz))]
    fn slant_delay(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch_j2000_s: i64,
        frequency_hz: f64,
    ) -> PyResult<f64> {
        let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, 0.0)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        ionex_slant_delay(
            &self.inner,
            receiver,
            elevation_deg * DEG_TO_RAD,
            azimuth_deg * DEG_TO_RAD,
            epoch_j2000_s,
            frequency_hz,
        )
        .map_err(|err| match err {
            sidereon_core::Error::InvalidInput(message) => {
                PyValueError::new_err(format!("invalid IONEX slant input: {message}"))
            }
            other => to_solve_err(other.to_string()),
        })
    }

    /// Serialize this product to standard IONEX text via the core writer.
    ///
    /// Re-parsing the output with [`load_ionex`] yields an equal product (node
    /// axes, geometry, exponent, map epochs, and every TEC / RMS value).
    fn to_ionex_string(&self) -> String {
        self.inner.to_ionex_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "Ionex(epochs={}, lat_nodes={}, lon_nodes={}, exponent={})",
            self.inner.map_epochs_s().len(),
            self.inner.lat_nodes_deg().len(),
            self.inner.lon_nodes_deg().len(),
            self.inner.exponent(),
        )
    }
}

/// Parse an IONEX vertical-TEC product from in-memory bytes or a file path.
///
/// `source` may be:
/// - `bytes` / `bytearray`: the full IONEX text content, parsed directly; or
/// - a path (`str` or `os.PathLike`): the file is read and parsed.
///
/// Raises [`IonexParseError`](crate::IonexParseError) on malformed content,
/// `OSError` if the path cannot be read, and `ValueError` if `source` is neither
/// bytes nor a path.
#[pyfunction]
fn load_ionex(source: &Bound<'_, PyAny>) -> PyResult<PyIonex> {
    // bytes-like first, so a `bytes` argument keeps the "content" meaning.
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        let inner = Ionex::parse(bytes.as_bytes()).map_err(to_ionex_err)?;
        return Ok(PyIonex { inner });
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        // SAFETY: the buffer is copied into the parser synchronously here; no
        // Python code runs in between to mutate or free it.
        let inner = Ionex::parse(unsafe { buf.as_bytes() }).map_err(to_ionex_err)?;
        return Ok(PyIonex { inner });
    }
    // Otherwise treat it as a path (str / os.PathLike via PyO3's fspath support).
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_ionex expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    let inner = Ionex::parse(&data).map_err(to_ionex_err)?;
    Ok(PyIonex { inner })
}

/// GPS broadcast Klobuchar ionospheric group delay in the model's native units
/// (positive metres). This is the bit-exact (0-ULP) entry: latitude/longitude
/// and azimuth/elevation are in **degrees**, `t_gps_s` is the GPS
/// **second-of-day** in `[0, 86400)`, and no angle or time conversion happens at
/// the boundary. `alpha` (a0..a3) and `beta` (b0..b3) are the eight transmitted
/// GPS Klobuchar coefficients. The L1 delay is scaled to `frequency_hz` by the
/// dispersive `(f_l1 / f)^2` factor. Raises `ValueError` on out-of-range or
/// non-finite input.
#[pyfunction]
#[pyo3(signature = (alpha, beta, lat_deg, lon_deg, az_deg, el_deg, t_gps_s, frequency_hz))]
#[allow(clippy::too_many_arguments)]
fn klobuchar_native(
    alpha: [f64; 4],
    beta: [f64; 4],
    lat_deg: f64,
    lon_deg: f64,
    az_deg: f64,
    el_deg: f64,
    t_gps_s: f64,
    frequency_hz: f64,
) -> PyResult<f64> {
    let params = KlobucharParams { alpha, beta };
    core_klobuchar_native(
        &params,
        lat_deg,
        lon_deg,
        az_deg,
        el_deg,
        t_gps_s,
        frequency_hz,
    )
    .map_err(|err| match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid Klobuchar input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    })
}

/// Galileo NeQuick-G single-frequency ionospheric group delay in the model's
/// native input units (positive metres).
///
/// `ai0` / `ai1` / `ai2` are the three broadcast NeQuick-G effective-ionisation
/// coefficients. Receiver latitude/longitude and the satellite elevation are in
/// **degrees**; `t_gal_s` is the Galileo-system **second of day** and
/// `day_of_year` the fractional day of year. `frequency_hz` is the carrier the
/// dispersive delay is reported on. This is the native (no Instant) entry
/// parallel to [`klobuchar_native`]; it never consumes GPS Klobuchar
/// coefficients. Raises `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (ai0, ai1, ai2, lat_deg, lon_deg, el_deg, t_gal_s, day_of_year, frequency_hz))]
#[allow(clippy::too_many_arguments)]
fn galileo_nequick_g_native(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    lat_deg: f64,
    lon_deg: f64,
    el_deg: f64,
    t_gal_s: f64,
    day_of_year: f64,
    frequency_hz: f64,
) -> PyResult<f64> {
    let coeffs = GalileoNequickCoeffs { ai0, ai1, ai2 };
    let eval = GalileoNequickEval {
        lat_deg,
        lon_deg,
        el_deg,
        t_gal_s,
        day_of_year,
        frequency_hz,
    };
    core_galileo_nequick_g_native(&coeffs, eval).map_err(|err| match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid NeQuick-G input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    })
}

/// Assemble the [`NequickGRayEval`] receiver-to-satellite ray the full NeQuick-G
/// model integrates over, from the boundary's degrees/metres/hours inputs. No
/// conversion happens: the reference algorithm consumes these units directly.
#[allow(clippy::too_many_arguments)]
fn nequick_g_ray(
    month: u8,
    utc_hours: f64,
    station_lon_deg: f64,
    station_lat_deg: f64,
    station_height_m: f64,
    satellite_lon_deg: f64,
    satellite_lat_deg: f64,
    satellite_height_m: f64,
) -> NequickGRayEval {
    NequickGRayEval {
        month,
        utc_hours,
        station_lon_deg,
        station_lat_deg,
        station_height_m,
        satellite_lon_deg,
        satellite_lat_deg,
        satellite_height_m,
    }
}

/// Map a NeQuick-G full-integration failure into a Pythonic error, preserving
/// the engine message.
fn nequick_g_error(err: sidereon_core::Error) -> PyErr {
    match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid NeQuick-G input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    }
}

/// Full Galileo NeQuick-G slant total electron content along a receiver-to-
/// satellite ray, in TECU.
///
/// This is the reference-grade three-dimensional NeQuick 2 profiler integrated
/// along the ray (the full model), distinct from the compact broadcast-driven
/// [`galileo_nequick_g_native`]. `ai0` / `ai1` / `ai2` are the three broadcast
/// effective-ionisation coefficients. `month` is `1..=12` and `utc_hours` the
/// UTC time of day in `[0, 24]`. Station and satellite longitudes/latitudes are
/// in degrees and heights in metres above the reference sphere. Raises
/// `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    ai0, ai1, ai2, month, utc_hours,
    station_lon_deg, station_lat_deg, station_height_m,
    satellite_lon_deg, satellite_lat_deg, satellite_height_m,
))]
#[allow(clippy::too_many_arguments)]
fn nequick_g_stec_tecu(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    month: u8,
    utc_hours: f64,
    station_lon_deg: f64,
    station_lat_deg: f64,
    station_height_m: f64,
    satellite_lon_deg: f64,
    satellite_lat_deg: f64,
    satellite_height_m: f64,
) -> PyResult<f64> {
    let coeffs = GalileoNequickCoeffs { ai0, ai1, ai2 };
    let ray = nequick_g_ray(
        month,
        utc_hours,
        station_lon_deg,
        station_lat_deg,
        station_height_m,
        satellite_lon_deg,
        satellite_lat_deg,
        satellite_height_m,
    );
    core_nequick_g_stec_tecu(&coeffs, &ray).map_err(nequick_g_error)
}

/// Full Galileo NeQuick-G slant ionospheric group delay (positive metres) on
/// `frequency_hz`.
///
/// The full three-dimensional slant TEC from [`nequick_g_stec_tecu`] is mapped to
/// a group delay with the dispersive `40.3e16 / f^2` relation. Inputs match
/// [`nequick_g_stec_tecu`]; `frequency_hz` is the carrier the delay is reported
/// on. Raises `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    ai0, ai1, ai2, month, utc_hours,
    station_lon_deg, station_lat_deg, station_height_m,
    satellite_lon_deg, satellite_lat_deg, satellite_height_m, frequency_hz,
))]
#[allow(clippy::too_many_arguments)]
fn nequick_g_delay_m(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    month: u8,
    utc_hours: f64,
    station_lon_deg: f64,
    station_lat_deg: f64,
    station_height_m: f64,
    satellite_lon_deg: f64,
    satellite_lat_deg: f64,
    satellite_height_m: f64,
    frequency_hz: f64,
) -> PyResult<f64> {
    let coeffs = GalileoNequickCoeffs { ai0, ai1, ai2 };
    let ray = nequick_g_ray(
        month,
        utc_hours,
        station_lon_deg,
        station_lat_deg,
        station_height_m,
        satellite_lon_deg,
        satellite_lat_deg,
        satellite_height_m,
    );
    core_nequick_g_delay_m(&coeffs, &ray, frequency_hz).map_err(nequick_g_error)
}

/// Build the core split-Julian-date UTC [`Instant`] the ionosphere dispatcher
/// consumes, from civil-calendar fields, via `Instant::from_utc_civil`.
fn instant_from_utc_civil(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
) -> PyResult<Instant> {
    Instant::from_utc_civil(year, month, day, hour, minute, second)
        .map_err(|err| PyValueError::new_err(format!("invalid epoch: {err}")))
}

#[allow(clippy::too_many_arguments)]
fn iono_delay(
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    epoch: Instant,
    frequency_hz: f64,
    model: &IonoModel,
) -> PyResult<f64> {
    let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, height_m)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    core_ionosphere_delay(
        receiver,
        elevation_deg * DEG_TO_RAD,
        azimuth_deg * DEG_TO_RAD,
        epoch,
        frequency_hz,
        model,
    )
    .map_err(|err| match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid ionosphere input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    })
}

/// GPS broadcast Klobuchar ionospheric group delay (positive metres) for a
/// civil UTC epoch.
///
/// Delegates to the core `ionosphere_delay` dispatcher with a Klobuchar model,
/// building the epoch `Instant` via `Instant::from_utc_civil`. Receiver
/// latitude/longitude and the satellite azimuth/elevation are in degrees;
/// `height_m` is the receiver ellipsoidal height. `alpha` (a0..a3) and `beta`
/// (b0..b3) are the eight transmitted Klobuchar coefficients. Raises
/// `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    alpha, beta, lat_deg, lon_deg, azimuth_deg, elevation_deg,
    year, month, day, hour, minute, second, frequency_hz, height_m=0.0
))]
#[allow(clippy::too_many_arguments)]
fn ionosphere_delay_klobuchar(
    alpha: [f64; 4],
    beta: [f64; 4],
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
    frequency_hz: f64,
    height_m: f64,
) -> PyResult<f64> {
    let epoch = instant_from_utc_civil(year, month, day, hour, minute, second)?;
    let model = IonoModel::Klobuchar(KlobucharParams { alpha, beta });
    iono_delay(
        lat_deg,
        lon_deg,
        height_m,
        azimuth_deg,
        elevation_deg,
        epoch,
        frequency_hz,
        &model,
    )
}

/// Galileo NeQuick-G ionospheric group delay (positive metres) for a civil UTC
/// epoch.
///
/// Delegates to the core `ionosphere_delay` dispatcher with a NeQuick-G model,
/// building the epoch `Instant` via `Instant::from_utc_civil`; the dispatcher
/// derives the Galileo second-of-day and day-of-year from that epoch. Receiver
/// latitude/longitude and the satellite azimuth/elevation are in degrees;
/// `height_m` is the receiver ellipsoidal height. `ai0` / `ai1` / `ai2` are the
/// three broadcast NeQuick-G coefficients. Raises `ValueError` on out-of-range
/// or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    ai0, ai1, ai2, lat_deg, lon_deg, azimuth_deg, elevation_deg,
    year, month, day, hour, minute, second, frequency_hz, height_m=0.0
))]
#[allow(clippy::too_many_arguments)]
fn ionosphere_delay_nequick(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
    frequency_hz: f64,
    height_m: f64,
) -> PyResult<f64> {
    let epoch = instant_from_utc_civil(year, month, day, hour, minute, second)?;
    let model = IonoModel::GalileoNequickG(GalileoNequickCoeffs { ai0, ai1, ai2 });
    iono_delay(
        lat_deg,
        lon_deg,
        height_m,
        azimuth_deg,
        elevation_deg,
        epoch,
        frequency_hz,
        &model,
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIonex>()?;
    m.add_function(wrap_pyfunction!(load_ionex, m)?)?;
    m.add_function(wrap_pyfunction!(klobuchar_native, m)?)?;
    m.add_function(wrap_pyfunction!(galileo_nequick_g_native, m)?)?;
    m.add_function(wrap_pyfunction!(nequick_g_stec_tecu, m)?)?;
    m.add_function(wrap_pyfunction!(nequick_g_delay_m, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_delay_klobuchar, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_delay_nequick, m)?)?;
    Ok(())
}
