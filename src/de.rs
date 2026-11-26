//! Geocentric Sun and Moon states from JPL DE binary ephemerides.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::de::{DeBody, DeEphemeris, DeError as CoreDeError, DeState, DeSunMoon};

use crate::{binding_error, SidereonError};

create_exception!(
    _sidereon,
    DeError,
    SidereonError,
    "A JPL DE binary open, parse, or evaluation failure."
);

fn de_error(error: CoreDeError) -> PyErr {
    let exception = DeError::new_err(error.to_string());
    match error {
        CoreDeError::Io(source) => binding_error(exception, "io", |detail| {
            detail.set_item("message", source.to_string())?;
            detail.set_item("raw_os_error", source.raw_os_error())
        }),
        CoreDeError::InvalidFile(reason) => binding_error(exception, "invalid_file", |detail| {
            detail.set_item("reason", reason)
        }),
        CoreDeError::NonFiniteEpoch => binding_error(exception, "non_finite_epoch", |_| Ok(())),
        CoreDeError::OutsideCoverage { epoch_tdb_seconds } => {
            binding_error(exception, "outside_coverage", |detail| {
                detail.set_item("epoch_tdb_seconds", epoch_tdb_seconds)
            })
        }
        CoreDeError::NonFiniteState => binding_error(exception, "non_finite_state", |_| Ok(())),
        _ => exception,
    }
}

/// A solar-system body with a mass parameter in a DE header.
#[pyclass(module = "sidereon._sidereon", name = "DeBody", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyDeBody {
    /// Sun.
    SUN,
    /// Mercury.
    MERCURY,
    /// Venus.
    VENUS,
    /// Earth–Moon barycentre.
    EARTH_MOON,
    /// Earth.
    EARTH,
    /// Moon.
    MOON,
    /// Mars system.
    MARS,
    /// Jupiter system.
    JUPITER,
    /// Saturn system.
    SATURN,
    /// Uranus system.
    URANUS,
    /// Neptune system.
    NEPTUNE,
    /// Pluto system.
    PLUTO,
    /// Entire solar system.
    SOLAR_SYSTEM,
}

impl From<PyDeBody> for DeBody {
    fn from(body: PyDeBody) -> Self {
        match body {
            PyDeBody::SUN => Self::Sun,
            PyDeBody::MERCURY => Self::Mercury,
            PyDeBody::VENUS => Self::Venus,
            PyDeBody::EARTH_MOON => Self::EarthMoon,
            PyDeBody::EARTH => Self::Earth,
            PyDeBody::MOON => Self::Moon,
            PyDeBody::MARS => Self::Mars,
            PyDeBody::JUPITER => Self::Jupiter,
            PyDeBody::SATURN => Self::Saturn,
            PyDeBody::URANUS => Self::Uranus,
            PyDeBody::NEPTUNE => Self::Neptune,
            PyDeBody::PLUTO => Self::Pluto,
            PyDeBody::SOLAR_SYSTEM => Self::SolarSystem,
        }
    }
}

/// Geocentric Cartesian position and velocity in GCRF.
#[pyclass(module = "sidereon._sidereon", name = "DeState", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyDeState(DeState);

#[pymethods]
impl PyDeState {
    /// Position in metres.
    #[getter]
    fn position_m(&self) -> [f64; 3] {
        self.0.position_m
    }

    /// Velocity in metres per second.
    #[getter]
    fn velocity_m_s(&self) -> [f64; 3] {
        self.0.velocity_m_s
    }
}

/// Geocentric Sun and Moon states evaluated at one epoch.
#[pyclass(module = "sidereon._sidereon", name = "DeSunMoon", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyDeSunMoon(DeSunMoon);

#[pymethods]
impl PyDeSunMoon {
    /// Sun state.
    #[getter]
    fn sun(&self) -> PyDeState {
        PyDeState(self.0.sun)
    }

    /// Moon state.
    #[getter]
    fn moon(&self) -> PyDeState {
        PyDeState(self.0.moon)
    }
}

/// Immutable JPL DE binary ephemeris.
#[pyclass(module = "sidereon._sidereon", name = "DeEphemeris")]
#[derive(Clone)]
pub struct PyDeEphemeris(pub(crate) Arc<DeEphemeris>);

#[pymethods]
impl PyDeEphemeris {
    /// Open and parse a DE binary.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        DeEphemeris::load(path)
            .map(Arc::new)
            .map(Self)
            .map_err(de_error)
    }

    /// Parse the contents of a DE binary.
    #[staticmethod]
    fn from_bytes(data: &[u8]) -> PyResult<Self> {
        DeEphemeris::from_bytes(data)
            .map(Arc::new)
            .map(Self)
            .map_err(de_error)
    }

    /// DE ephemeris version number.
    fn version(&self) -> u32 {
        self.0.version()
    }

    /// Number of bytes in each binary record.
    fn record_size(&self) -> usize {
        self.0.record_size()
    }

    /// Start and end epochs in TDB seconds from J2000.
    fn coverage_tdb_seconds(&self) -> (f64, f64) {
        self.0.coverage_tdb_seconds()
    }

    /// Named constants from the file header.
    fn constants(&self) -> BTreeMap<String, f64> {
        self.0.constants().clone()
    }

    /// Astronomical unit in metres.
    fn astronomical_unit_m(&self) -> f64 {
        self.0.astronomical_unit_m()
    }

    /// Earth-to-Moon mass ratio.
    fn earth_moon_mass_ratio(&self) -> f64 {
        self.0.earth_moon_mass_ratio()
    }

    /// Gravitational mass parameter in cubic metres per square second, if present.
    fn gm_m3_s2(&self, body: PyDeBody) -> Option<f64> {
        self.0.gm_m3_s2(body.into())
    }

    /// Geocentric GCRF positions in metres, Sun first and Moon second.
    fn sun_moon_positions_at_tdb_seconds(
        &self,
        epoch_tdb_seconds: f64,
    ) -> PyResult<([f64; 3], [f64; 3])> {
        let positions = self
            .0
            .sun_moon_positions_at_tdb_seconds(epoch_tdb_seconds)
            .map_err(de_error)?;
        Ok((positions.sun, positions.moon))
    }

    /// Geocentric GCRF Sun and Moon states at TDB seconds from J2000.
    fn sun_moon_at_tdb_seconds(&self, epoch_tdb_seconds: f64) -> PyResult<PyDeSunMoon> {
        self.0
            .sun_moon_at_tdb_seconds(epoch_tdb_seconds)
            .map(PyDeSunMoon)
            .map_err(de_error)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DeError", m.py().get_type::<DeError>())?;
    m.add_class::<PyDeBody>()?;
    m.add_class::<PyDeState>()?;
    m.add_class::<PyDeSunMoon>()?;
    m.add_class::<PyDeEphemeris>()?;
    Ok(())
}
