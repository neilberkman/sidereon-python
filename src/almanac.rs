//! Astronomical almanac event binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon::passes::UtcInstant;
use sidereon_core::astro::almanac as core;
use sidereon_core::astro::frames::transforms::GeodeticStationKm;

use crate::spk::PySpk;

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

#[pyclass(module = "sidereon._sidereon", name = "SeasonKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PySeasonKind {
    MARCH_EQUINOX,
    JUNE_SOLSTICE,
    SEPTEMBER_EQUINOX,
    DECEMBER_SOLSTICE,
    UNKNOWN,
}

impl From<core::SeasonKind> for PySeasonKind {
    fn from(value: core::SeasonKind) -> Self {
        match value {
            core::SeasonKind::MarchEquinox => Self::MARCH_EQUINOX,
            core::SeasonKind::JuneSolstice => Self::JUNE_SOLSTICE,
            core::SeasonKind::SeptemberEquinox => Self::SEPTEMBER_EQUINOX,
            core::SeasonKind::DecemberSolstice => Self::DECEMBER_SOLSTICE,
            _ => Self::UNKNOWN,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "MoonPhaseKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyMoonPhaseKind {
    NEW,
    FIRST_QUARTER,
    FULL,
    LAST_QUARTER,
    UNKNOWN,
}

impl From<core::MoonPhaseKind> for PyMoonPhaseKind {
    fn from(value: core::MoonPhaseKind) -> Self {
        match value {
            core::MoonPhaseKind::New => Self::NEW,
            core::MoonPhaseKind::FirstQuarter => Self::FIRST_QUARTER,
            core::MoonPhaseKind::Full => Self::FULL,
            core::MoonPhaseKind::LastQuarter => Self::LAST_QUARTER,
            _ => Self::UNKNOWN,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PlanetaryEventKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPlanetaryEventKind {
    CONJUNCTION,
    OPPOSITION,
    UNKNOWN,
}

impl PyPlanetaryEventKind {
    fn to_core(self) -> PyResult<core::PlanetaryEventKind> {
        match self {
            PyPlanetaryEventKind::CONJUNCTION => Ok(core::PlanetaryEventKind::Conjunction),
            PyPlanetaryEventKind::OPPOSITION => Ok(core::PlanetaryEventKind::Opposition),
            PyPlanetaryEventKind::UNKNOWN => Err(PyValueError::new_err(
                "PlanetaryEventKind.UNKNOWN is not a valid input",
            )),
        }
    }
}

impl From<core::PlanetaryEventKind> for PyPlanetaryEventKind {
    fn from(value: core::PlanetaryEventKind) -> Self {
        match value {
            core::PlanetaryEventKind::Conjunction => Self::CONJUNCTION,
            core::PlanetaryEventKind::Opposition => Self::OPPOSITION,
            _ => Self::UNKNOWN,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "CulminationKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyCulminationKind {
    UPPER,
    LOWER,
    UNKNOWN,
}

impl From<core::CulminationKind> for PyCulminationKind {
    fn from(value: core::CulminationKind) -> Self {
        match value {
            core::CulminationKind::Upper => Self::UPPER,
            core::CulminationKind::Lower => Self::LOWER,
            _ => Self::UNKNOWN,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "EclipseKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyEclipseKind {
    LUNAR_PENUMBRAL,
    LUNAR_PARTIAL,
    LUNAR_TOTAL,
    SOLAR_PARTIAL,
    SOLAR_ANNULAR,
    SOLAR_TOTAL,
    SOLAR_HYBRID,
    UNKNOWN,
}

impl From<core::EclipseKind> for PyEclipseKind {
    fn from(value: core::EclipseKind) -> Self {
        match value {
            core::EclipseKind::LunarPenumbral => Self::LUNAR_PENUMBRAL,
            core::EclipseKind::LunarPartial => Self::LUNAR_PARTIAL,
            core::EclipseKind::LunarTotal => Self::LUNAR_TOTAL,
            core::EclipseKind::SolarPartial => Self::SOLAR_PARTIAL,
            core::EclipseKind::SolarAnnular => Self::SOLAR_ANNULAR,
            core::EclipseKind::SolarTotal => Self::SOLAR_TOTAL,
            core::EclipseKind::SolarHybrid => Self::SOLAR_HYBRID,
            _ => Self::UNKNOWN,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Planet", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyPlanet {
    MERCURY,
    VENUS,
    MARS,
    JUPITER,
    SATURN,
    URANUS,
    NEPTUNE,
    UNKNOWN,
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
            PyPlanet::UNKNOWN => Err(PyValueError::new_err("Planet.UNKNOWN is not a valid input")),
        }
    }
}

impl From<core::Planet> for PyPlanet {
    fn from(value: core::Planet) -> Self {
        match value {
            core::Planet::Mercury => Self::MERCURY,
            core::Planet::Venus => Self::VENUS,
            core::Planet::Mars => Self::MARS,
            core::Planet::Jupiter => Self::JUPITER,
            core::Planet::Saturn => Self::SATURN,
            core::Planet::Uranus => Self::URANUS,
            core::Planet::Neptune => Self::NEPTUNE,
            _ => Self::UNKNOWN,
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
    #[staticmethod]
    fn sun() -> Self {
        Self {
            kind: TransitBodyKind::Sun,
        }
    }

    #[staticmethod]
    fn moon() -> Self {
        Self {
            kind: TransitBodyKind::Moon,
        }
    }

    #[staticmethod]
    fn planet(planet: PyPlanet) -> Self {
        Self {
            kind: TransitBodyKind::Planet(planet),
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "SeasonEvent")]
#[derive(Clone, Copy)]
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
    fn kind(&self) -> PySeasonKind {
        self.inner.kind.into()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "MoonPhaseEvent")]
#[derive(Clone, Copy)]
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
    fn kind(&self) -> PyMoonPhaseKind {
        self.inner.kind.into()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PlanetaryEvent")]
#[derive(Clone, Copy)]
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
    fn planet(&self) -> PyPlanet {
        self.inner.planet.into()
    }

    #[getter]
    fn kind(&self) -> PyPlanetaryEventKind {
        self.inner.kind.into()
    }

    #[getter]
    fn elongation_deg(&self) -> f64 {
        self.inner.elongation_deg
    }
}

#[pyclass(module = "sidereon._sidereon", name = "CulminationEvent")]
#[derive(Clone, Copy)]
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
    fn kind(&self) -> PyCulminationKind {
        self.inner.kind.into()
    }

    #[getter]
    fn altitude_deg(&self) -> f64 {
        self.inner.altitude_deg
    }
}

#[pyclass(module = "sidereon._sidereon", name = "EclipseEvent")]
#[derive(Clone, Copy)]
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
    fn kind(&self) -> PyEclipseKind {
        self.inner.kind.into()
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
}

#[pyfunction]
#[pyo3(signature = (start_unix_us, end_unix_us, step_seconds=86_400.0, time_tolerance_seconds=60.0, spk=None))]
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
