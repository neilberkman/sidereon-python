//! IERS 2010 orientation from daily Earth orientation parameters.

use std::sync::Arc;

use numpy::PyArray2;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::frames::transforms::FrameTransformError;
use sidereon_core::astro::frames::{
    EarthOrientation, EarthOrientationProvider, Iers2010EarthOrientationProvider, Iers2010EopSample,
};
use sidereon_core::astro::propagator::api::PropagationContext;

use crate::de::PyDeEphemeris;
use crate::marshal::rows3_to_array;
use crate::{binding_error, degrade_reason_label, PyValidityMode, SolveError};

pub(crate) fn frame_error(error: FrameTransformError) -> PyErr {
    let exception = SolveError::new_err(error.to_string());
    match error {
        FrameTransformError::InvalidInput { field, reason } => {
            binding_error(exception, "invalid_input", |detail| {
                detail.set_item("field", field)?;
                detail.set_item("reason", reason)
            })
        }
        FrameTransformError::Ut1OutsideCoverage { reason } => {
            binding_error(exception, "ut1_outside_coverage", |detail| {
                detail.set_item("reason", degrade_reason_label(reason))
            })
        }
    }
}

/// Daily Earth orientation sample for IERS 2010.
#[pyclass(module = "sidereon._sidereon", name = "Iers2010EopSample", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyIers2010EopSample(Iers2010EopSample);

#[pymethods]
impl PyIers2010EopSample {
    /// Specify daily Earth orientation parameters and optional pole rates.
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (mjd_utc, xp_arcsec, yp_arcsec, ut1_minus_utc_s, lod_s, dx_arcsec, dy_arcsec, pole_rates_rad_s=None))]
    fn new(
        mjd_utc: i32,
        xp_arcsec: f64,
        yp_arcsec: f64,
        ut1_minus_utc_s: f64,
        lod_s: f64,
        dx_arcsec: f64,
        dy_arcsec: f64,
        pole_rates_rad_s: Option<[f64; 2]>,
    ) -> Self {
        Self(Iers2010EopSample {
            mjd_utc,
            xp_arcsec,
            yp_arcsec,
            ut1_minus_utc_s,
            lod_s,
            dx_arcsec,
            dy_arcsec,
            pole_rates_rad_s,
        })
    }

    /// UTC Modified Julian Date at midnight.
    #[getter]
    fn mjd_utc(&self) -> i32 {
        self.0.mjd_utc
    }

    /// Polar motion X in arcseconds.
    #[getter]
    fn xp_arcsec(&self) -> f64 {
        self.0.xp_arcsec
    }

    /// Polar motion Y in arcseconds.
    #[getter]
    fn yp_arcsec(&self) -> f64 {
        self.0.yp_arcsec
    }

    /// UT1 minus UTC in seconds.
    #[getter]
    fn ut1_minus_utc_s(&self) -> f64 {
        self.0.ut1_minus_utc_s
    }

    /// Excess length of day in seconds.
    #[getter]
    fn lod_s(&self) -> f64 {
        self.0.lod_s
    }

    /// Celestial intermediate pole X correction in arcseconds.
    #[getter]
    fn dx_arcsec(&self) -> f64 {
        self.0.dx_arcsec
    }

    /// Celestial intermediate pole Y correction in arcseconds.
    #[getter]
    fn dy_arcsec(&self) -> f64 {
        self.0.dy_arcsec
    }

    /// Optional x and y pole rates in radians per second.
    #[getter]
    fn pole_rates_rad_s(&self) -> Option<[f64; 2]> {
        self.0.pole_rates_rad_s
    }
}

/// Orientation and rotation rates between GCRF and ITRF.
#[pyclass(module = "sidereon._sidereon", name = "EarthOrientation", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyEarthOrientation(EarthOrientation);

#[pymethods]
impl PyEarthOrientation {
    /// IERS 2010 rotation quaternion, scalar component first.
    fn iers2010_quaternion(&self) -> Option<[f64; 4]> {
        self.0.iers2010_quaternion()
    }

    /// Rate of Earth rotation in ITRF, radians per second.
    fn earth_rotation_vector_itrf_rad_s(&self) -> [f64; 3] {
        self.0.earth_rotation_vector_itrf_rad_s()
    }

    /// First UT1 table departure, if this orientation contains one.
    fn ut1_degraded(&self) -> Option<&'static str> {
        self.0.ut1_degraded().map(degrade_reason_label)
    }

    /// Rotation from GCRF into ITRF.
    fn gcrf_to_itrf_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.0.gcrf_to_itrf_matrix())
    }

    /// Rotation from ITRF into GCRF.
    fn itrf_to_gcrf_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.0.itrf_to_gcrf_matrix())
    }

    /// Time derivative of the GCRF to ITRF rotation, per second.
    fn gcrf_to_itrf_rotation_rate_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.0.gcrf_to_itrf_rotation_rate_matrix())
    }

    /// Time derivative of the ITRF to GCRF rotation, per second.
    fn itrf_to_gcrf_rotation_rate_matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.0.itrf_to_gcrf_rotation_rate_matrix())
    }
}

/// CIO-based IERS 2010 provider from daily Earth orientation samples.
#[pyclass(
    module = "sidereon._sidereon",
    name = "Iers2010EarthOrientationProvider",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyIers2010EarthOrientationProvider(pub(crate) Arc<Iers2010EarthOrientationProvider>);

#[pymethods]
impl PyIers2010EarthOrientationProvider {
    /// Build a strict provider from samples in increasing UTC MJD order.
    #[new]
    fn new(samples: Vec<PyIers2010EopSample>) -> PyResult<Self> {
        Iers2010EarthOrientationProvider::new(samples.into_iter().map(|sample| sample.0).collect())
            .map(Arc::new)
            .map(Self)
            .map_err(frame_error)
    }

    /// Select the policy for epochs outside available samples.
    fn with_validity(&self, validity: PyValidityMode) -> Self {
        Self(Arc::new(
            self.0.as_ref().clone().with_validity(validity.into()),
        ))
    }

    /// Evaluate at UTC MJD and seconds within that UTC day.
    fn orientation_at_utc_mjd(
        &self,
        mjd_utc: i32,
        seconds_in_day: f64,
    ) -> PyResult<PyEarthOrientation> {
        self.0
            .orientation_at_utc_mjd(mjd_utc, seconds_in_day)
            .map(PyEarthOrientation)
            .map_err(frame_error)
    }

    /// Evaluate at TDB seconds from J2000.
    fn orientation_at_tdb_seconds(&self, epoch_tdb_seconds: f64) -> PyResult<PyEarthOrientation> {
        self.0
            .orientation_at_tdb_seconds(epoch_tdb_seconds)
            .map(PyEarthOrientation)
            .map_err(frame_error)
    }
}

/// Optional external ephemeris and terrestrial orientation for force evaluation.
#[pyclass(module = "sidereon._sidereon", name = "PropagationContext")]
#[derive(Clone, Default)]
pub struct PyPropagationContext {
    pub(crate) inner: PropagationContext,
    de: Option<PyDeEphemeris>,
}

#[pymethods]
impl PyPropagationContext {
    /// Build an empty context using the core defaults.
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Use a DE ephemeris for Sun and Moon positions during propagation.
    fn with_de_ephemeris(&self, ephemeris: &PyDeEphemeris) -> Self {
        Self {
            inner: self.inner.clone().with_de_ephemeris(ephemeris.0.clone()),
            de: Some(ephemeris.clone()),
        }
    }

    /// Return the attached ephemeris if supplied.
    fn de_ephemeris(&self) -> Option<PyDeEphemeris> {
        self.de.clone()
    }

    /// Use the IERS 2010 provider for the terrestrial frame during propagation.
    fn with_body_fixed_frame_provider(
        &self,
        provider: &PyIers2010EarthOrientationProvider,
    ) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_body_fixed_frame_provider(provider.0.clone()),
            de: self.de.clone(),
        }
    }

    /// First recorded departure from UT1 coverage, if any.
    fn ut1_departure(&self) -> Option<&'static str> {
        self.inner.ut1_departure().map(degrade_reason_label)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIers2010EopSample>()?;
    m.add_class::<PyEarthOrientation>()?;
    m.add_class::<PyIers2010EarthOrientationProvider>()?;
    m.add_class::<PyPropagationContext>()?;
    Ok(())
}
