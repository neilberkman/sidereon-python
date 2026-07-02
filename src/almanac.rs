//! Astronomical almanac event binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon::passes::UtcInstant;
use sidereon_core::astro::almanac as core;
use sidereon_core::astro::frames::transforms::GeodeticStationKm;

use crate::spk::PySpk;
use crate::SidereonError;

fn to_almanac_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn instant(us: i64) -> UtcInstant {
    UtcInstant::from_unix_microseconds(us)
}

fn station(latitude_deg: f64, longitude_deg: f64, altitude_km: f64) -> GeodeticStationKm {
    GeodeticStationKm {
        latitude_deg,
        longitude_deg,
        altitude_km,
    }
}

fn non_exhaustive_variant(enum_name: &str) -> PyErr {
    SidereonError::new_err(format!("core returned an unsupported {enum_name} variant"))
}

#[pyclass(module = "sidereon._sidereon", name = "SeasonKind", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Seasonal marker kind.
pub enum PySeasonKind {
    MARCH_EQUINOX,
    JUNE_SOLSTICE,
    SEPTEMBER_EQUINOX,
    DECEMBER_SOLSTICE,
}

impl TryFrom<core::SeasonKind> for PySeasonKind {
    type Error = PyErr;

    fn try_from(value: core::SeasonKind) -> Result<Self, Self::Error> {
        match value {
            core::SeasonKind::MarchEquinox => Ok(Self::MARCH_EQUINOX),
            core::SeasonKind::JuneSolstice => Ok(Self::JUNE_SOLSTICE),
            core::SeasonKind::SeptemberEquinox => Ok(Self::SEPTEMBER_EQUINOX),
            core::SeasonKind::DecemberSolstice => Ok(Self::DECEMBER_SOLSTICE),
            _ => Err(non_exhaustive_variant("SeasonKind")),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "MoonPhaseKind", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Principal lunar phase kind.
pub enum PyMoonPhaseKind {
    NEW,
    FIRST_QUARTER,
    FULL,
    LAST_QUARTER,
}

impl TryFrom<core::MoonPhaseKind> for PyMoonPhaseKind {
    type Error = PyErr;

    fn try_from(value: core::MoonPhaseKind) -> Result<Self, Self::Error> {
        match value {
            core::MoonPhaseKind::New => Ok(Self::NEW),
            core::MoonPhaseKind::FirstQuarter => Ok(Self::FIRST_QUARTER),
            core::MoonPhaseKind::Full => Ok(Self::FULL),
            core::MoonPhaseKind::LastQuarter => Ok(Self::LAST_QUARTER),
            _ => Err(non_exhaustive_variant("MoonPhaseKind")),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PlanetaryEventKind", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Planetary ecliptic-longitude event kind.
pub enum PyPlanetaryEventKind {
    CONJUNCTION,
    OPPOSITION,
}

impl PyPlanetaryEventKind {
    fn to_core(self) -> PyResult<core::PlanetaryEventKind> {
        match self {
            PyPlanetaryEventKind::CONJUNCTION => Ok(core::PlanetaryEventKind::Conjunction),
            PyPlanetaryEventKind::OPPOSITION => Ok(core::PlanetaryEventKind::Opposition),
        }
    }
}

impl TryFrom<core::PlanetaryEventKind> for PyPlanetaryEventKind {
    type Error = PyErr;

    fn try_from(value: core::PlanetaryEventKind) -> Result<Self, Self::Error> {
        match value {
            core::PlanetaryEventKind::Conjunction => Ok(Self::CONJUNCTION),
            core::PlanetaryEventKind::Opposition => Ok(Self::OPPOSITION),
            _ => Err(non_exhaustive_variant("PlanetaryEventKind")),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "CulminationKind", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Meridian transit kind.
pub enum PyCulminationKind {
    UPPER,
    LOWER,
}

impl TryFrom<core::CulminationKind> for PyCulminationKind {
    type Error = PyErr;

    fn try_from(value: core::CulminationKind) -> Result<Self, Self::Error> {
        match value {
            core::CulminationKind::Upper => Ok(Self::UPPER),
            core::CulminationKind::Lower => Ok(Self::LOWER),
            _ => Err(non_exhaustive_variant("CulminationKind")),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "EclipseKind", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(non_camel_case_types)]
/// Lunar or solar eclipse kind.
pub enum PyEclipseKind {
    LUNAR_PENUMBRAL,
    LUNAR_PARTIAL,
    LUNAR_TOTAL,
    SOLAR_PARTIAL,
    SOLAR_ANNULAR,
    SOLAR_TOTAL,
    SOLAR_HYBRID,
}

impl TryFrom<core::EclipseKind> for PyEclipseKind {
    type Error = PyErr;

    fn try_from(value: core::EclipseKind) -> Result<Self, Self::Error> {
        match value {
            core::EclipseKind::LunarPenumbral => Ok(Self::LUNAR_PENUMBRAL),
            core::EclipseKind::LunarPartial => Ok(Self::LUNAR_PARTIAL),
            core::EclipseKind::LunarTotal => Ok(Self::LUNAR_TOTAL),
            core::EclipseKind::SolarPartial => Ok(Self::SOLAR_PARTIAL),
            core::EclipseKind::SolarAnnular => Ok(Self::SOLAR_ANNULAR),
            core::EclipseKind::SolarTotal => Ok(Self::SOLAR_TOTAL),
            core::EclipseKind::SolarHybrid => Ok(Self::SOLAR_HYBRID),
            _ => Err(non_exhaustive_variant("EclipseKind")),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Planet", eq, eq_int)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Planet selector for almanac event finders.
pub enum PyPlanet {
    MERCURY,
    VENUS,
    MARS,
    JUPITER,
    SATURN,
    URANUS,
    NEPTUNE,
}

impl PyPlanet {
    fn to_core(self) -> PyResult<core::Planet> {
        match self {
            PyPlanet::MERCURY => Ok(core::Planet::Mercury),
            PyPlanet::VENUS => Ok(core::Planet::Venus),
            PyPlanet::MARS => Ok(core::Planet::Mars),
            PyPlanet::JUPITER => Ok(core::Planet::Jupiter),
            PyPlanet::SATURN => Ok(core::Planet::Saturn),
            PyPlanet::URANUS => Ok(core::Planet::Uranus),
            PyPlanet::NEPTUNE => Ok(core::Planet::Neptune),
        }
    }
}

impl TryFrom<core::Planet> for PyPlanet {
    type Error = PyErr;

    fn try_from(value: core::Planet) -> Result<Self, Self::Error> {
        match value {
            core::Planet::Mercury => Ok(Self::MERCURY),
            core::Planet::Venus => Ok(Self::VENUS),
            core::Planet::Mars => Ok(Self::MARS),
            core::Planet::Jupiter => Ok(Self::JUPITER),
            core::Planet::Saturn => Ok(Self::SATURN),
            core::Planet::Uranus => Ok(Self::URANUS),
            core::Planet::Neptune => Ok(Self::NEPTUNE),
            _ => Err(non_exhaustive_variant("Planet")),
        }
    }
}

#[derive(Clone, Copy)]
enum TransitBodyKind {
    Sun,
    Moon,
    Planet(PyPlanet),
}

#[pyclass(module = "sidereon._sidereon", name = "TransitBody")]
#[derive(Clone, Copy)]
/// Body selector for meridian transit searches.
pub struct PyTransitBody {
    kind: TransitBodyKind,
}

impl PyTransitBody {
    fn to_core(self) -> PyResult<core::TransitBody> {
        match self.kind {
            TransitBodyKind::Sun => Ok(core::TransitBody::Sun),
            TransitBodyKind::Moon => Ok(core::TransitBody::Moon),
            TransitBodyKind::Planet(planet) => Ok(core::TransitBody::Planet(planet.to_core()?)),
        }
    }
}

#[pymethods]
impl PyTransitBody {
    /// Select the Sun for meridian transit searches.
    #[staticmethod]
    fn sun() -> Self {
        Self {
            kind: TransitBodyKind::Sun,
        }
    }

    /// Select the Moon for meridian transit searches.
    #[staticmethod]
    fn moon() -> Self {
        Self {
            kind: TransitBodyKind::Moon,
        }
    }

    /// Select a planet for meridian transit searches.
    #[staticmethod]
    fn planet(planet: PyPlanet) -> Self {
        Self {
            kind: TransitBodyKind::Planet(planet),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SeasonEvent")]
#[derive(Clone, Copy)]
/// One seasonal marker event.
pub struct PySeasonEvent {
    inner: core::SeasonEvent,
}

#[pymethods]
impl PySeasonEvent {
    #[getter]
    fn time_unix_us(&self) -> i64 {
        self.inner.time.unix_microseconds()
    }

    #[getter]
    fn kind(&self) -> PyResult<PySeasonKind> {
        self.inner.kind.try_into()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "SeasonEvent(time_unix_us={}, kind={:?})",
            self.inner.time.unix_microseconds(),
            self.kind()?
        ))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "MoonPhaseEvent")]
#[derive(Clone, Copy)]
/// One principal lunar phase event.
pub struct PyMoonPhaseEvent {
    inner: core::MoonPhaseEvent,
}

#[pymethods]
impl PyMoonPhaseEvent {
    #[getter]
    fn time_unix_us(&self) -> i64 {
        self.inner.time.unix_microseconds()
    }

    #[getter]
    fn kind(&self) -> PyResult<PyMoonPhaseKind> {
        self.inner.kind.try_into()
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "MoonPhaseEvent(time_unix_us={}, kind={:?})",
            self.inner.time.unix_microseconds(),
            self.kind()?
        ))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PlanetaryEvent")]
#[derive(Clone, Copy)]
/// One planetary conjunction or opposition event.
pub struct PyPlanetaryEvent {
    inner: core::PlanetaryEvent,
}

#[pymethods]
impl PyPlanetaryEvent {
    #[getter]
    fn time_unix_us(&self) -> i64 {
        self.inner.time.unix_microseconds()
    }

    #[getter]
    fn planet(&self) -> PyResult<PyPlanet> {
        self.inner.planet.try_into()
    }

    #[getter]
    fn kind(&self) -> PyResult<PyPlanetaryEventKind> {
        self.inner.kind.try_into()
    }

    #[getter]
    fn elongation_deg(&self) -> f64 {
        self.inner.elongation_deg
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "PlanetaryEvent(time_unix_us={}, planet={:?}, kind={:?}, elongation_deg={:.6})",
            self.inner.time.unix_microseconds(),
            self.planet()?,
            self.kind()?,
            self.inner.elongation_deg
        ))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "CulminationEvent")]
#[derive(Clone, Copy)]
/// One upper or lower meridian transit event.
pub struct PyCulminationEvent {
    inner: core::CulminationEvent,
}

#[pymethods]
impl PyCulminationEvent {
    #[getter]
    fn time_unix_us(&self) -> i64 {
        self.inner.time.unix_microseconds()
    }

    #[getter]
    fn kind(&self) -> PyResult<PyCulminationKind> {
        self.inner.kind.try_into()
    }

    #[getter]
    fn altitude_deg(&self) -> f64 {
        self.inner.altitude_deg
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "CulminationEvent(time_unix_us={}, kind={:?}, altitude_deg={:.6})",
            self.inner.time.unix_microseconds(),
            self.kind()?,
            self.inner.altitude_deg
        ))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "EclipseEvent")]
#[derive(Clone, Copy)]
/// One lunar or solar eclipse event.
pub struct PyEclipseEvent {
    inner: core::EclipseEvent,
}

#[pymethods]
impl PyEclipseEvent {
    #[getter]
    fn time_maximum_unix_us(&self) -> i64 {
        self.inner.time_maximum.unix_microseconds()
    }

    #[getter]
    fn kind(&self) -> PyResult<PyEclipseKind> {
        self.inner.kind.try_into()
    }

    #[getter]
    fn magnitude(&self) -> f64 {
        self.inner.magnitude
    }

    #[getter]
    fn moon_latitude_deg(&self) -> f64 {
        self.inner.moon_latitude_deg
    }

    #[getter]
    fn gamma(&self) -> f64 {
        self.inner.gamma
    }

    #[getter]
    fn uncertain(&self) -> bool {
        self.inner.uncertain
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "EclipseEvent(time_maximum_unix_us={}, kind={:?}, magnitude={:.6}, uncertain={})",
            self.inner.time_maximum.unix_microseconds(),
            self.kind()?,
            self.inner.magnitude,
            self.inner.uncertain
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (start_unix_us, end_unix_us, step_seconds=86_400.0, time_tolerance_seconds=60.0, spk=None))]
/// Find seasonal marker events in a Unix-microsecond interval.
///
/// Pass an SPK kernel for ephemeris-backed evaluation, or omit it for the analytic source.
fn seasons(
    start_unix_us: i64,
    end_unix_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
    spk: Option<&PySpk>,
) -> PyResult<Vec<PySeasonEvent>> {
    let source = spk
        .map(|spk| core::EphemerisSource::Spk(&spk.inner))
        .unwrap_or(core::EphemerisSource::Analytic);
    core::seasons(
        source,
        instant(start_unix_us),
        instant(end_unix_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .map(|events| {
        events
            .into_iter()
            .map(|inner| PySeasonEvent { inner })
            .collect()
    })
    .map_err(to_almanac_err)
}

#[pyfunction]
#[pyo3(signature = (start_unix_us, end_unix_us, step_seconds=86_400.0, time_tolerance_seconds=60.0, spk=None))]
/// Find principal lunar phase events in a Unix-microsecond interval.
///
/// Pass an SPK kernel for ephemeris-backed evaluation, or omit it for the analytic source.
fn moon_phases(
    start_unix_us: i64,
    end_unix_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
    spk: Option<&PySpk>,
) -> PyResult<Vec<PyMoonPhaseEvent>> {
    let source = spk
        .map(|spk| core::EphemerisSource::Spk(&spk.inner))
        .unwrap_or(core::EphemerisSource::Analytic);
    core::moon_phases(
        source,
        instant(start_unix_us),
        instant(end_unix_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .map(|events| {
        events
            .into_iter()
            .map(|inner| PyMoonPhaseEvent { inner })
            .collect()
    })
    .map_err(to_almanac_err)
}

#[pyfunction]
#[pyo3(signature = (spk, planet, kind, start_unix_us, end_unix_us, step_seconds=86_400.0, time_tolerance_seconds=60.0))]
/// Find planetary conjunctions or oppositions using an SPK kernel.
///
/// The search interval is supplied as Unix microseconds.
fn planetary_events(
    spk: &PySpk,
    planet: PyPlanet,
    kind: PyPlanetaryEventKind,
    start_unix_us: i64,
    end_unix_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyPlanetaryEvent>> {
    core::planetary_events(
        core::EphemerisSource::Spk(&spk.inner),
        planet.to_core()?,
        kind.to_core()?,
        instant(start_unix_us),
        instant(end_unix_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .map(|events| {
        events
            .into_iter()
            .map(|inner| PyPlanetaryEvent { inner })
            .collect()
    })
    .map_err(to_almanac_err)
}

#[pyfunction]
#[pyo3(signature = (body, latitude_deg, longitude_deg, altitude_km, start_unix_us, end_unix_us, step_seconds=3_600.0, time_tolerance_seconds=30.0, spk=None))]
#[allow(clippy::too_many_arguments)]
/// Find upper and lower meridian transits for a body at a ground station.
///
/// Latitude and longitude are supplied in degrees.
fn meridian_transits(
    body: PyTransitBody,
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    start_unix_us: i64,
    end_unix_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
    spk: Option<&PySpk>,
) -> PyResult<Vec<PyCulminationEvent>> {
    let source = spk
        .map(|spk| core::EphemerisSource::Spk(&spk.inner))
        .unwrap_or(core::EphemerisSource::Analytic);
    let station = station(latitude_deg, longitude_deg, altitude_km);
    core::meridian_transits(
        source,
        body.to_core()?,
        &station,
        instant(start_unix_us),
        instant(end_unix_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .map(|events| {
        events
            .into_iter()
            .map(|inner| PyCulminationEvent { inner })
            .collect()
    })
    .map_err(to_almanac_err)
}

#[pyfunction]
#[pyo3(signature = (start_unix_us, end_unix_us, step_seconds=86_400.0, time_tolerance_seconds=60.0, spk=None))]
/// Find lunar and solar eclipse events in a Unix-microsecond interval.
///
/// Pass an SPK kernel for ephemeris-backed evaluation, or omit it for the analytic source.
fn lunar_solar_eclipses(
    start_unix_us: i64,
    end_unix_us: i64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
    spk: Option<&PySpk>,
) -> PyResult<Vec<PyEclipseEvent>> {
    let source = spk
        .map(|spk| core::EphemerisSource::Spk(&spk.inner))
        .unwrap_or(core::EphemerisSource::Analytic);
    core::lunar_solar_eclipses(
        source,
        instant(start_unix_us),
        instant(end_unix_us),
        step_seconds,
        time_tolerance_seconds,
    )
    .map(|events| {
        events
            .into_iter()
            .map(|inner| PyEclipseEvent { inner })
            .collect()
    })
    .map_err(to_almanac_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySeasonKind>()?;
    m.add_class::<PyMoonPhaseKind>()?;
    m.add_class::<PyPlanetaryEventKind>()?;
    m.add_class::<PyCulminationKind>()?;
    m.add_class::<PyEclipseKind>()?;
    m.add_class::<PyPlanet>()?;
    m.add_class::<PyTransitBody>()?;
    m.add_class::<PySeasonEvent>()?;
    m.add_class::<PyMoonPhaseEvent>()?;
    m.add_class::<PyPlanetaryEvent>()?;
    m.add_class::<PyCulminationEvent>()?;
    m.add_class::<PyEclipseEvent>()?;
    m.add_function(wrap_pyfunction!(seasons, m)?)?;
    m.add_function(wrap_pyfunction!(moon_phases, m)?)?;
    m.add_function(wrap_pyfunction!(planetary_events, m)?)?;
    m.add_function(wrap_pyfunction!(meridian_transits, m)?)?;
    m.add_function(wrap_pyfunction!(lunar_solar_eclipses, m)?)?;
    Ok(())
}
