//! General ground-site body observation binding.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon::passes::UtcInstant;
use sidereon_core::astro::bodies::observe as core;
use sidereon_core::astro::frames::transforms::{GeodeticStationKm, PolarMotion};

use crate::marshal::{fixed_array_from_any, FinitePolicy};
use crate::spk::PySpk;

fn to_observe_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn station(latitude_deg: f64, longitude_deg: f64, altitude_km: f64) -> GeodeticStationKm {
    GeodeticStationKm {
        latitude_deg,
        longitude_deg,
        altitude_km,
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PolarMotion")]
#[derive(Clone, Copy)]
/// Polar motion offsets for apparent-place reductions.
pub struct PyPolarMotion {
    inner: PolarMotion,
}

impl PyPolarMotion {
    fn inner(&self) -> PolarMotion {
        self.inner
    }
}

#[pymethods]
impl PyPolarMotion {
    /// Build polar motion offsets in radians.
    #[new]
    fn new(xp_rad: f64, yp_rad: f64) -> PyResult<Self> {
        PolarMotion::from_radians(xp_rad, yp_rad)
            .map(|inner| Self { inner })
            .map_err(to_observe_err)
    }

    #[getter]
    fn xp_rad(&self) -> f64 {
        self.inner.xp_rad
    }

    #[getter]
    fn yp_rad(&self) -> f64 {
        self.inner.yp_rad
    }

    fn __repr__(&self) -> String {
        format!(
            "PolarMotion(xp_rad={:.6e}, yp_rad={:.6e})",
            self.inner.xp_rad, self.inner.yp_rad
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Refraction")]
#[derive(Clone, Copy)]
/// Simple atmospheric refraction settings for topocentric apparent places.
pub struct PyRefraction {
    inner: core::Refraction,
}

impl PyRefraction {
    fn inner(&self) -> core::Refraction {
        self.inner
    }
}

#[pymethods]
impl PyRefraction {
    /// Build refraction settings from pressure and temperature.
    #[new]
    fn new(pressure_mbar: f64, temperature_c: f64) -> Self {
        Self {
            inner: core::Refraction {
                pressure_mbar,
                temperature_c,
            },
        }
    }

    #[getter]
    fn pressure_mbar(&self) -> f64 {
        self.inner.pressure_mbar
    }

    #[getter]
    fn temperature_c(&self) -> f64 {
        self.inner.temperature_c
    }

    fn __repr__(&self) -> String {
        format!(
            "Refraction(pressure_mbar={:.3}, temperature_c={:.3})",
            self.inner.pressure_mbar, self.inner.temperature_c
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "ObserveOptions")]
#[derive(Clone, Copy)]
/// Options controlling apparent-place corrections for body observation.
pub struct PyObserveOptions {
    inner: core::ObserveOptions,
}

impl PyObserveOptions {
    fn inner(&self) -> core::ObserveOptions {
        self.inner
    }
}

#[pymethods]
impl PyObserveOptions {
    /// Build body-observation options.
    ///
    /// Omitted polar motion and refraction use the core defaults for those corrections.
    #[new]
    #[pyo3(signature = (polar_motion=None, refraction=None, deflection=true, aberration=true))]
    fn new(
        polar_motion: Option<&PyPolarMotion>,
        refraction: Option<&PyRefraction>,
        deflection: bool,
        aberration: bool,
    ) -> Self {
        Self {
            inner: core::ObserveOptions {
                polar_motion: polar_motion.map(PyPolarMotion::inner),
                refraction: refraction.map(PyRefraction::inner),
                deflection,
                aberration,
            },
        }
    }

    #[getter]
    fn deflection(&self) -> bool {
        self.inner.deflection
    }

    #[getter]
    fn aberration(&self) -> bool {
        self.inner.aberration
    }

    #[getter]
    fn polar_motion(&self) -> Option<PyPolarMotion> {
        self.inner.polar_motion.map(|inner| PyPolarMotion { inner })
    }

    #[getter]
    fn refraction(&self) -> Option<PyRefraction> {
        self.inner.refraction.map(|inner| PyRefraction { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ObserveOptions(polar_motion={}, refraction={}, deflection={}, aberration={})",
            self.inner.polar_motion.is_some(),
            self.inner.refraction.is_some(),
            self.inner.deflection,
            self.inner.aberration
        )
    }
}

enum TargetKind {
    Sun,
    Moon,
    Spk {
        kernel: Py<PySpk>,
        naif_id: i32,
    },
    BarycentricState {
        kernel: Py<PySpk>,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
    },
}

#[pyclass(module = "sidereon._sidereon", name = "Target")]
/// Target body selector for `observe_body`.
pub struct PyTarget {
    kind: TargetKind,
}

#[pymethods]
impl PyTarget {
    /// Select the Sun as the observation target.
    #[staticmethod]
    fn sun() -> Self {
        Self {
            kind: TargetKind::Sun,
        }
    }

    /// Select the Moon as the observation target.
    #[staticmethod]
    fn moon() -> Self {
        Self {
            kind: TargetKind::Moon,
        }
    }

    /// Select a body from an SPK kernel by NAIF id.
    #[staticmethod]
    fn spk(kernel: Py<PySpk>, naif_id: i32) -> Self {
        Self {
            kind: TargetKind::Spk { kernel, naif_id },
        }
    }

    /// Select a barycentric state evaluated against an SPK kernel.
    ///
    /// Position and velocity may be numpy arrays or ordinary Python sequences of three finite floats.
    #[staticmethod]
    fn barycentric_state(
        kernel: Py<PySpk>,
        position_km: &Bound<'_, PyAny>,
        velocity_km_s: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let position_km =
            fixed_array_from_any::<3>("position_km", position_km, FinitePolicy::RequireFinite)?;
        let velocity_km_s =
            fixed_array_from_any::<3>("velocity_km_s", velocity_km_s, FinitePolicy::RequireFinite)?;
        Ok(Self {
            kind: TargetKind::BarycentricState {
                kernel,
                position_km,
                velocity_km_s,
            },
        })
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Equatorial")]
#[derive(Clone, Copy)]
/// Equatorial coordinates for an observed body.
pub struct PyEquatorial {
    inner: core::Equatorial,
}

#[pymethods]
impl PyEquatorial {
    #[getter]
    fn right_ascension_deg(&self) -> f64 {
        self.inner.right_ascension_deg
    }

    #[getter]
    fn right_ascension_hours(&self) -> f64 {
        self.inner.right_ascension_hours
    }

    #[getter]
    fn declination_deg(&self) -> f64 {
        self.inner.declination_deg
    }

    #[getter]
    fn distance_km(&self) -> f64 {
        self.inner.distance_km
    }

    fn __repr__(&self) -> String {
        format!(
            "Equatorial(right_ascension_deg={:.6}, declination_deg={:.6}, distance_km={:.3})",
            self.inner.right_ascension_deg, self.inner.declination_deg, self.inner.distance_km
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Horizontal")]
#[derive(Clone, Copy)]
/// Topocentric horizontal coordinates for an observed body.
pub struct PyHorizontal {
    inner: core::Horizontal,
}

#[pymethods]
impl PyHorizontal {
    #[getter]
    fn azimuth_deg(&self) -> f64 {
        self.inner.azimuth_deg
    }

    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.inner.elevation_deg
    }

    #[getter]
    fn range_km(&self) -> f64 {
        self.inner.range_km
    }

    fn __repr__(&self) -> String {
        format!(
            "Horizontal(azimuth_deg={:.6}, elevation_deg={:.6}, range_km={:.3})",
            self.inner.azimuth_deg, self.inner.elevation_deg, self.inner.range_km
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Ecliptic")]
#[derive(Clone, Copy)]
/// Ecliptic coordinates for an observed body.
pub struct PyEcliptic {
    inner: core::Ecliptic,
}

#[pymethods]
impl PyEcliptic {
    #[getter]
    fn longitude_deg(&self) -> f64 {
        self.inner.longitude_deg
    }

    #[getter]
    fn latitude_deg(&self) -> f64 {
        self.inner.latitude_deg
    }

    #[getter]
    fn distance_km(&self) -> f64 {
        self.inner.distance_km
    }

    fn __repr__(&self) -> String {
        format!(
            "Ecliptic(longitude_deg={:.6}, latitude_deg={:.6}, distance_km={:.3})",
            self.inner.longitude_deg, self.inner.latitude_deg, self.inner.distance_km
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "Observation")]
#[derive(Clone, Copy)]
/// Complete apparent-place observation result for a target body.
pub struct PyObservation {
    inner: core::Observation,
}

impl From<core::Observation> for PyObservation {
    fn from(inner: core::Observation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyObservation {
    #[getter]
    fn astrometric(&self) -> PyEquatorial {
        PyEquatorial {
            inner: self.inner.astrometric,
        }
    }

    #[getter]
    fn apparent_icrs(&self) -> PyEquatorial {
        PyEquatorial {
            inner: self.inner.apparent_icrs,
        }
    }

    #[getter]
    fn apparent(&self) -> PyEquatorial {
        PyEquatorial {
            inner: self.inner.apparent,
        }
    }

    #[getter]
    fn horizontal(&self) -> PyHorizontal {
        PyHorizontal {
            inner: self.inner.horizontal,
        }
    }

    #[getter]
    fn hour_angle_deg(&self) -> f64 {
        self.inner.hour_angle_deg
    }

    #[getter]
    fn hour_angle_hours(&self) -> f64 {
        self.inner.hour_angle_hours
    }

    #[getter]
    fn ecliptic(&self) -> PyEcliptic {
        PyEcliptic {
            inner: self.inner.ecliptic,
        }
    }

    #[getter]
    fn reduced(&self) -> bool {
        self.inner.reduced
    }

    fn __repr__(&self) -> String {
        format!(
            "Observation(hour_angle_deg={:.6}, reduced={})",
            self.inner.hour_angle_deg, self.inner.reduced
        )
    }
}

#[pyfunction(name = "observe_body")]
#[pyo3(signature = (latitude_deg, longitude_deg, altitude_km, epoch_unix_us, target, options=None))]
/// Observe the Sun, Moon, SPK body, or barycentric target from a ground station.
///
/// Latitude and longitude are supplied in degrees and the epoch is Unix microseconds.
fn observe_body(
    py: Python<'_>,
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    epoch_unix_us: i64,
    target: &PyTarget,
    options: Option<&PyObserveOptions>,
) -> PyResult<PyObservation> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let time = UtcInstant::from_unix_microseconds(epoch_unix_us);
    let options = options.map(PyObserveOptions::inner).unwrap_or_default();
    match &target.kind {
        TargetKind::Sun => core::observe(&station, time, core::Target::Sun, options),
        TargetKind::Moon => core::observe(&station, time, core::Target::Moon, options),
        TargetKind::Spk { kernel, naif_id } => {
            let kernel = kernel.borrow(py);
            core::observe(
                &station,
                time,
                core::Target::Spk {
                    kernel: &kernel.inner,
                    naif_id: *naif_id,
                },
                options,
            )
        }
        TargetKind::BarycentricState {
            kernel,
            position_km,
            velocity_km_s,
        } => {
            let kernel = kernel.borrow(py);
            core::observe(
                &station,
                time,
                core::Target::BarycentricState {
                    kernel: &kernel.inner,
                    position_km: *position_km,
                    velocity_km_s: *velocity_km_s,
                },
                options,
            )
        }
    }
    .map(PyObservation::from)
    .map_err(to_observe_err)
}

#[pyfunction]
#[pyo3(signature = (latitude_deg, longitude_deg, altitude_km, epoch_unix_us, kernel, naif_id))]
/// Observe one SPK body from a ground station.
///
/// This convenience wrapper uses the default observation options.
fn observe_spk_body(
    latitude_deg: f64,
    longitude_deg: f64,
    altitude_km: f64,
    epoch_unix_us: i64,
    kernel: &PySpk,
    naif_id: i32,
) -> PyResult<PyObservation> {
    let station = station(latitude_deg, longitude_deg, altitude_km);
    let time = UtcInstant::from_unix_microseconds(epoch_unix_us);
    core::observe_spk_body(&station, time, &kernel.inner, naif_id)
        .map(PyObservation::from)
        .map_err(to_observe_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPolarMotion>()?;
    m.add_class::<PyRefraction>()?;
    m.add_class::<PyObserveOptions>()?;
    m.add_class::<PyTarget>()?;
    m.add_class::<PyEquatorial>()?;
    m.add_class::<PyHorizontal>()?;
    m.add_class::<PyEcliptic>()?;
    m.add_class::<PyObservation>()?;
    m.add_function(wrap_pyfunction!(observe_body, m)?)?;
    m.add_function(wrap_pyfunction!(observe_spk_body, m)?)?;
    Ok(())
}
