//! Ground-observer Sun/Moon geometry binding.
//!
//! Thin INTERFACE over `sidereon_core::astro::bodies` ground-site helpers
//! ([`sun_az_el`](sidereon_core::astro::bodies::sun_az_el),
//! [`moon_az_el`](sidereon_core::astro::bodies::moon_az_el),
//! [`moon_illumination`](sidereon_core::astro::bodies::moon_illumination)) and the
//! rise/set finders
//! ([`moon_elevation_deg`](sidereon_core::astro::bodies::moon_elevation_deg),
//! [`find_moon_elevation_crossings`](sidereon_core::astro::bodies::find_moon_elevation_crossings),
//! [`find_moon_transits`](sidereon_core::astro::bodies::find_moon_transits)). The
//! binding marshals the station (geodetic degrees/km) and unix-microsecond UTC
//! instants; every angle is produced by the analytic core ephemeris.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon::passes::UtcInstant;
use sidereon_core::astro::bodies::{
    find_moon_elevation_crossings as core_find_moon_elevation_crossings,
    find_moon_transits as core_find_moon_transits, moon_az_el as core_moon_az_el,
    moon_illumination as core_moon_illumination, sun_az_el as core_sun_az_el, BodyObservationError,
    MoonElevationCrossingKind, MoonElevationOptions, MoonTransitKind,
};
use sidereon_core::astro::events::EventFinderError;
use sidereon_core::astro::frames::transforms::GeodeticStationKm;

/// Map a ground-site Sun/Moon observation failure to a Python exception. Every
/// [`BodyObservationError`] variant is a rejected caller input (a station/time
/// outside the ephemeris or frame-transform domain, or a degenerate phase-angle
/// geometry), so each maps to `ValueError`; the exhaustive match forces a
/// reclassification if the core ever grows a genuine solver-failure variant.
fn body_observation_error(err: BodyObservationError) -> PyErr {
    match err {
        BodyObservationError::Ephemeris(_)
        | BodyObservationError::FrameTransform(_)
        | BodyObservationError::Angle(_) => PyValueError::new_err(err.to_string()),
    }
}

/// Map an event-finder failure to a Python exception. The caller controls the
/// station, the time window, and the scan step/tolerance, so every
/// [`EventFinderError`] variant is an invalid input and maps to `ValueError`.
fn event_finder_error(err: EventFinderError) -> PyErr {
    match err {
        EventFinderError::InvalidInput { .. } => PyValueError::new_err(err.to_string()),
    }
}

fn station(latitude_deg: f64, longitude_deg: f64, altitude_km: f64) -> GeodeticStationKm {
    GeodeticStationKm {
        latitude_deg,
        longitude_deg,
        altitude_km,
    }
}

/// Topocentric look angle of a body from a ground site.
#[pyclass(module = "sidereon._sidereon", name = "BodyAzEl")]
#[derive(Clone)]
pub struct PyBodyAzEl {
    azimuth_deg: f64,
    elevation_deg: f64,
    range_km: f64,
}

#[pymethods]
impl PyBodyAzEl {
    /// Azimuth, degrees clockwise from north on `[0, 360)`.
    #[getter]
    fn azimuth_deg(&self) -> f64 {
        self.azimuth_deg
    }

    /// Elevation above the local horizon, degrees on `[-90, 90]`.
    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.elevation_deg
    }

    /// Slant range from the site to the body, kilometres.
    #[getter]
    fn range_km(&self) -> f64 {
        self.range_km
    }

    fn __repr__(&self) -> String {
        format!(
            "BodyAzEl(azimuth_deg={:.3}, elevation_deg={:.3}, range_km={:.3})",
            self.azimuth_deg, self.elevation_deg, self.range_km
        )
    }
}

/// The Moon's illuminated state as seen from a ground site.
#[pyclass(module = "sidereon._sidereon", name = "MoonIllumination")]
#[derive(Clone)]
pub struct PyMoonIllumination {
    illuminated_fraction: f64,
    phase_angle_deg: f64,
}

#[pymethods]
impl PyMoonIllumination {
    /// Sunlit fraction of the lunar disk on `[0, 1]` (0 = new, 1 = full).
    #[getter]
    fn illuminated_fraction(&self) -> f64 {
        self.illuminated_fraction
    }

    /// Sun-Moon-observer phase angle, degrees on `[0, 180]` (0 = full).
    #[getter]
    fn phase_angle_deg(&self) -> f64 {
        self.phase_angle_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "MoonIllumination(illuminated_fraction={:.4}, phase_angle_deg={:.3})",
            self.illuminated_fraction, self.phase_angle_deg
        )
    }
}

/// One Moon elevation threshold crossing (moonrise / moonset).
#[pyclass(module = "sidereon._sidereon", name = "MoonElevationCrossing")]
#[derive(Clone)]
pub struct PyMoonElevationCrossing {
    time_unix_us: i64,
    kind: &'static str,
    elevation_deg: f64,
}

#[pymethods]
impl PyMoonElevationCrossing {
    /// Refined crossing instant as a unix-microsecond UTC stamp.
    #[getter]
    fn time_unix_us(&self) -> i64 {
        self.time_unix_us
    }

    /// `"rising"` (moonrise) or `"setting"` (moonset).
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    /// Topocentric Moon elevation at the refined instant, degrees.
    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.elevation_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "MoonElevationCrossing(time_unix_us={}, kind={:?}, elevation_deg={:.3})",
            self.time_unix_us, self.kind, self.elevation_deg
        )
    }
}

/// One Moon meridian transit (culmination).
#[pyclass(module = "sidereon._sidereon", name = "MoonTransit")]
#[derive(Clone)]
pub struct PyMoonTransit {
    time_unix_us: i64,
    kind: &'static str,
    elevation_deg: f64,
}

#[pymethods]
impl PyMoonTransit {
    /// Refined culmination instant as a unix-microsecond UTC stamp.
    #[getter]
    fn time_unix_us(&self) -> i64 {
        self.time_unix_us
    }

    /// `"upper"` (due south, highest) or `"lower"` (due north, lowest).
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    /// Topocentric Moon elevation at the refined instant, degrees.
    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.elevation_deg
    }

    fn __repr__(&self) -> String {
        format!(
            "MoonTransit(time_unix_us={}, kind={:?}, elevation_deg={:.3})",
            self.time_unix_us, self.kind, self.elevation_deg
        )
    }
}

/// Topocentric azimuth/elevation/range of the Sun from a ground site at a
/// unix-microsecond UTC instant. The station is geodetic latitude/longitude in
/// degrees and altitude in kilometres.
#[pyfunction]
#[pyo3(signature = (latitude_deg, longitude_deg, altitude_km, epoch_unix_us))]
fn sun_az_el(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    epoch_unix_us: i64,
) -> PyResult<PyBodyAzEl> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let time = UtcInstant::from_unix_microseconds(epoch_unix_us);
    let look = core_sun_az_el(&station, time).map_err(body_observation_error)?;
    Ok(PyBodyAzEl {
        azimuth_deg: look.azimuth_deg,
        elevation_deg: look.elevation_deg,
        range_km: look.range_km,
    })
}

/// Topocentric azimuth/elevation/range of the Moon from a ground site at a
/// unix-microsecond UTC instant (parallax-corrected for the nearby Moon).
#[pyfunction]
#[pyo3(signature = (latitude_deg, longitude_deg, altitude_km, epoch_unix_us))]
fn moon_az_el(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    epoch_unix_us: i64,
) -> PyResult<PyBodyAzEl> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let time = UtcInstant::from_unix_microseconds(epoch_unix_us);
    let look = core_moon_az_el(&station, time).map_err(body_observation_error)?;
    Ok(PyBodyAzEl {
        azimuth_deg: look.azimuth_deg,
        elevation_deg: look.elevation_deg,
        range_km: look.range_km,
    })
}

/// Illuminated fraction and phase angle of the Moon as seen from a ground site at
/// a unix-microsecond UTC instant.
#[pyfunction]
#[pyo3(signature = (latitude_deg, longitude_deg, altitude_km, epoch_unix_us))]
fn moon_illumination(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    epoch_unix_us: i64,
) -> PyResult<PyMoonIllumination> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let time = UtcInstant::from_unix_microseconds(epoch_unix_us);
    let illum = core_moon_illumination(&station, time).map_err(body_observation_error)?;
    Ok(PyMoonIllumination {
        illuminated_fraction: illum.illuminated_fraction,
        phase_angle_deg: illum.phase_angle_deg,
    })
}

/// Topocentric geometric Moon (disk-center) elevation at a ground site and a
/// unix-microsecond UTC instant, degrees.
///
/// Delegates to [`moon_az_el`](sidereon_core::astro::bodies::moon_az_el) and
/// returns its elevation, so an invalid-but-finite station (for example a
/// latitude outside `[-90, 90]`) raises `ValueError` rather than panicking inside
/// the core `moon_elevation_deg` helper, which `expect`s a valid station.
#[pyfunction]
#[pyo3(signature = (latitude_deg, longitude_deg, altitude_km, epoch_unix_us))]
fn moon_elevation_deg(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    epoch_unix_us: i64,
) -> PyResult<f64> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let time = UtcInstant::from_unix_microseconds(epoch_unix_us);
    Ok(core_moon_az_el(&station, time)
        .map_err(body_observation_error)?
        .elevation_deg)
}

/// Find Moon elevation threshold crossings (moonrise / moonset) over a UTC
/// window.
///
/// `start_unix_us` / `end_unix_us` bound the window; `elevation_threshold_deg`
/// defaults to the standard `-0.833` upper-limb-on-horizon convention.
/// `step_seconds` is the scan step and `time_tolerance_seconds` the refinement
/// tolerance.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    latitude_deg,
    longitude_deg,
    altitude_km,
    start_unix_us,
    end_unix_us,
    elevation_threshold_deg=-0.833,
    step_seconds=600.0,
    time_tolerance_seconds=1.0,
))]
fn find_moon_elevation_crossings(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    start_unix_us: i64,
    end_unix_us: i64,
    elevation_threshold_deg: f64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyMoonElevationCrossing>> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let crossings = core_find_moon_elevation_crossings(
        &station,
        UtcInstant::from_unix_microseconds(start_unix_us),
        UtcInstant::from_unix_microseconds(end_unix_us),
        MoonElevationOptions {
            elevation_threshold_deg,
            step_seconds,
            time_tolerance_seconds,
        },
    )
    .map_err(event_finder_error)?;
    Ok(crossings
        .into_iter()
        .map(|crossing| PyMoonElevationCrossing {
            time_unix_us: crossing.time.unix_microseconds(),
            kind: match crossing.kind {
                MoonElevationCrossingKind::Rising => "rising",
                MoonElevationCrossingKind::Setting => "setting",
            },
            elevation_deg: crossing.elevation_deg,
        })
        .collect())
}

/// Find Moon meridian transits (upper and lower culminations) over a UTC window.
///
/// `step_seconds` is the scan step and `time_tolerance_seconds` the refinement
/// tolerance.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    latitude_deg,
    longitude_deg,
    altitude_km,
    start_unix_us,
    end_unix_us,
    step_seconds=600.0,
    time_tolerance_seconds=1.0,
))]
fn find_moon_transits(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    start_unix_us: i64,
    end_unix_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyMoonTransit>> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let transits = core_find_moon_transits(
        &station,
        UtcInstant::from_unix_microseconds(start_unix_us),
        UtcInstant::from_unix_microseconds(end_unix_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .map_err(event_finder_error)?;
    Ok(transits
        .into_iter()
        .map(|transit| PyMoonTransit {
            time_unix_us: transit.time.unix_microseconds(),
            kind: match transit.kind {
                MoonTransitKind::Upper => "upper",
                MoonTransitKind::Lower => "lower",
            },
            elevation_deg: transit.elevation_deg,
        })
        .collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBodyAzEl>()?;
    m.add_class::<PyMoonIllumination>()?;
    m.add_class::<PyMoonElevationCrossing>()?;
    m.add_class::<PyMoonTransit>()?;
    m.add_function(wrap_pyfunction!(sun_az_el, m)?)?;
    m.add_function(wrap_pyfunction!(moon_az_el, m)?)?;
    m.add_function(wrap_pyfunction!(moon_illumination, m)?)?;
    m.add_function(wrap_pyfunction!(moon_elevation_deg, m)?)?;
    m.add_function(wrap_pyfunction!(find_moon_elevation_crossings, m)?)?;
    m.add_function(wrap_pyfunction!(find_moon_transits, m)?)?;
    Ok(())
}
