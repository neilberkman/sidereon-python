//! Bodies binding: analytic Sun and Moon positions over an epoch grid.
//!
//! Marshals a unix-microsecond UTC epoch array into the core's analytic
//! ephemerides ([`sidereon_core::astro::bodies`]) and packages the per-epoch
//! Sun/Moon vectors as numpy `(n, 3)` arrays. No modeling lives here: each epoch's
//! precise time scales come from the parity-critical [`UtcInstant::time_scales`]
//! path and the positions are exactly `sun_moon_eci_at` / `sun_moon_ecef`, so the
//! numbers are bit-identical to what `sidereon-core` produces. The per-epoch loop
//! runs inside Rust, so the call crosses the FFI boundary once.

use numpy::{PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::bodies::{self as core_bodies, SunMoon};

use crate::marshal::{rows3_to_array, time_scales_from_unix_micros, EmptyPolicy};

/// A batch of Sun and Moon positions, one per epoch: `sun` and `moon` as numpy
/// `float64` arrays of shape `(n, 3)`, in **metres**.
///
/// The frame (geocentric ECI, mean equator and equinox of date, vs Earth-fixed
/// ITRS/ECEF) is fixed by which function produced this object: [`sun_moon_eci`]
/// or [`sun_moon_ecef`].
#[pyclass(module = "sidereon._sidereon", name = "SunMoon")]
pub struct PySunMoon {
    sun: Vec<[f64; 3]>,
    moon: Vec<[f64; 3]>,
    frame: &'static str,
}

#[pymethods]
impl PySunMoon {
    /// Sun positions as a numpy `(n, 3)` array, metres, in this object's frame.
    #[getter]
    fn sun<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.sun)
    }

    /// Moon positions as a numpy `(n, 3)` array, metres, in this object's frame.
    #[getter]
    fn moon<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3_to_array(py, &self.moon)
    }

    /// Number of epochs in the batch.
    #[getter]
    fn epoch_count(&self) -> usize {
        self.sun.len()
    }

    fn __len__(&self) -> usize {
        self.sun.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SunMoon(frame='{}', epoch_count={})",
            self.frame,
            self.sun.len()
        )
    }
}

impl PySunMoon {
    fn collect<E: std::fmt::Display>(
        scales: impl Iterator<Item = Result<SunMoon, E>>,
        frame: &'static str,
    ) -> PyResult<Self> {
        let mut sun = Vec::new();
        let mut moon = Vec::new();
        for sm in scales {
            let sm = sm.map_err(|err| PyValueError::new_err(err.to_string()))?;
            sun.push(sm.sun);
            moon.push(sm.moon);
        }
        Ok(Self { sun, moon, frame })
    }
}

/// Analytic Sun and Moon positions in the geocentric ECI frame (mean equator and
/// equinox of date), metres, for a batch of epochs.
///
/// `epochs_unix_us` is a 1-D `int64` array of unix-microsecond UTC stamps. Returns
/// a [`SunMoon`] whose `sun` / `moon` are numpy `(n, 3)` arrays in metres. The
/// model is the low-precision analytic series (Montenbruck & Gill); precision is
/// at the few-centimetre / sub-degree level. Raises `ValueError` on an empty grid.
#[pyfunction]
fn sun_moon_eci(epochs_unix_us: PyReadonlyArray1<'_, i64>) -> PyResult<PySunMoon> {
    let scales = time_scales_from_unix_micros(&epochs_unix_us, EmptyPolicy::Reject)?;
    PySunMoon::collect(scales.iter().map(core_bodies::sun_moon_eci_at), "eci")
}

/// Analytic Sun and Moon geocentric positions in the Earth-fixed ITRS (ECEF)
/// frame, metres, for a batch of epochs.
///
/// `epochs_unix_us` is a 1-D `int64` array of unix-microsecond UTC stamps. Returns
/// a [`SunMoon`] whose `sun` / `moon` are numpy `(n, 3)` arrays in metres, rotated
/// to ITRS with the crate's nutation + GAST (precession is implicit in the
/// of-date series and not re-applied). Raises `ValueError` on an empty grid.
#[pyfunction]
fn sun_moon_ecef(epochs_unix_us: PyReadonlyArray1<'_, i64>) -> PyResult<PySunMoon> {
    let scales = time_scales_from_unix_micros(&epochs_unix_us, EmptyPolicy::Reject)?;
    PySunMoon::collect(scales.iter().map(core_bodies::sun_moon_ecef), "ecef")
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySunMoon>()?;
    m.add_function(wrap_pyfunction!(sun_moon_eci, m)?)?;
    m.add_function(wrap_pyfunction!(sun_moon_ecef, m)?)?;
    Ok(())
}
