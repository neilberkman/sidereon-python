//! Classical orbital element conversions: state vector <-> Keplerian elements.
//!
//! Thin INTERFACE over `sidereon_core::astro::elements`. It marshals the
//! position/velocity vectors and the gravitational parameter into the core
//! [`rv2coe`](sidereon_core::astro::elements::rv2coe) /
//! [`coe2rv`](sidereon_core::astro::elements::coe2rv) functions and packages the
//! result. Every number is produced by the core (Vallado Algorithms 9 and 10);
//! no element logic lives here.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::astro::elements::{
    coe2rv as core_coe2rv, rv2coe as core_rv2coe, ClassicalElements, OrbitType,
};

use crate::marshal::{fixed_array, FinitePolicy};
use crate::np_array;

fn to_elements_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// An inertial Cartesian state returned to Python as a `(position, velocity)`
/// pair of numpy `(3,)` arrays.
type StateArrays<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Geometric classification of a two-body orbit.
#[pyclass(module = "sidereon._sidereon", name = "OrbitType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyOrbitType {
    /// Eccentric and inclined: all six classical elements are defined.
    ELLIPTICAL_INCLINED,
    /// Eccentric but equatorial: the longitude of perigee replaces the node.
    ELLIPTICAL_EQUATORIAL,
    /// Circular but inclined: the argument of latitude replaces the perigee.
    CIRCULAR_INCLINED,
    /// Circular and equatorial: the true longitude replaces node and perigee.
    CIRCULAR_EQUATORIAL,
}

impl From<OrbitType> for PyOrbitType {
    fn from(value: OrbitType) -> Self {
        match value {
            OrbitType::EllipticalInclined => PyOrbitType::ELLIPTICAL_INCLINED,
            OrbitType::EllipticalEquatorial => PyOrbitType::ELLIPTICAL_EQUATORIAL,
            OrbitType::CircularInclined => PyOrbitType::CIRCULAR_INCLINED,
            OrbitType::CircularEquatorial => PyOrbitType::CIRCULAR_EQUATORIAL,
        }
    }
}

#[pymethods]
impl PyOrbitType {
    /// Stable lowercase identifier for this orbit type.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            PyOrbitType::ELLIPTICAL_INCLINED => "elliptical_inclined",
            PyOrbitType::ELLIPTICAL_EQUATORIAL => "elliptical_equatorial",
            PyOrbitType::CIRCULAR_INCLINED => "circular_inclined",
            PyOrbitType::CIRCULAR_EQUATORIAL => "circular_equatorial",
        }
    }

    fn __repr__(&self) -> String {
        format!("OrbitType.{}", self.label().to_uppercase())
    }
}

/// Classical (Keplerian) orbital elements in the Vallado convention.
///
/// Angles are radians. Auxiliary angles (`arglat`, `truelon`, `lonper`) carry
/// `nan` when they do not apply to the orbit type, exactly as the core returns
/// them. A value produced by [`rv2coe`] round-trips through [`coe2rv`].
#[pyclass(module = "sidereon._sidereon", name = "ClassicalElements")]
#[derive(Clone)]
pub struct PyClassicalElements {
    inner: ClassicalElements,
}

impl PyClassicalElements {
    fn from_inner(inner: ClassicalElements) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyClassicalElements {
    /// Build a non-degenerate (elliptical inclined) element set from the six
    /// primary elements, leaving the special-case auxiliary angles undefined.
    ///
    /// `p` is the semi-latus rectum (km); `ecc` the eccentricity; `incl`,
    /// `raan`, `argp`, and `nu` are radians. For circular or equatorial orbits,
    /// obtain the element set from [`rv2coe`] so the auxiliary angle and orbit
    /// type are populated.
    #[new]
    fn new(p: f64, ecc: f64, incl: f64, raan: f64, argp: f64, nu: f64) -> Self {
        Self {
            inner: ClassicalElements::new(p, ecc, incl, raan, argp, nu),
        }
    }

    /// Semi-latus rectum `p = h^2 / mu` (km).
    #[getter]
    fn p(&self) -> f64 {
        self.inner.p
    }

    /// Semi-major axis `a` (km). `inf` for a parabolic orbit.
    #[getter]
    fn a(&self) -> f64 {
        self.inner.a
    }

    /// Eccentricity (dimensionless).
    #[getter]
    fn ecc(&self) -> f64 {
        self.inner.ecc
    }

    /// Inclination in `[0, pi]` (rad).
    #[getter]
    fn incl(&self) -> f64 {
        self.inner.incl
    }

    /// Right ascension of the ascending node (rad); `nan` for equatorial orbits.
    #[getter]
    fn raan(&self) -> f64 {
        self.inner.raan
    }

    /// Argument of perigee (rad); `nan` for circular orbits.
    #[getter]
    fn argp(&self) -> f64 {
        self.inner.argp
    }

    /// True anomaly (rad); `nan` for circular orbits.
    #[getter]
    fn nu(&self) -> f64 {
        self.inner.nu
    }

    /// Argument of latitude `u = argp + nu` (rad); defined for circular inclined.
    #[getter]
    fn arglat(&self) -> f64 {
        self.inner.arglat
    }

    /// True longitude (rad); defined for circular equatorial orbits.
    #[getter]
    fn truelon(&self) -> f64 {
        self.inner.truelon
    }

    /// Longitude of perigee (rad); defined for elliptical equatorial orbits.
    #[getter]
    fn lonper(&self) -> f64 {
        self.inner.lonper
    }

    /// Geometric classification of the orbit.
    #[getter]
    fn orbit_type(&self) -> PyOrbitType {
        self.inner.orbit_type.into()
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassicalElements(p={:.3}, ecc={:.6}, incl={:.6}, raan={:.6}, argp={:.6}, nu={:.6})",
            self.inner.p,
            self.inner.ecc,
            self.inner.incl,
            self.inner.raan,
            self.inner.argp,
            self.inner.nu
        )
    }
}

/// Convert an inertial Cartesian state to classical orbital elements.
///
/// `r` is the ECI position `(3,)` (km), `v` the ECI velocity `(3,)` (km/s), and
/// `mu` the gravitational parameter (km^3/s^2). Raises `ValueError` on
/// non-finite or degenerate input.
#[pyfunction]
#[pyo3(signature = (r, v, mu))]
fn rv2coe(
    r: PyReadonlyArray1<'_, f64>,
    v: PyReadonlyArray1<'_, f64>,
    mu: f64,
) -> PyResult<PyClassicalElements> {
    let r = fixed_array::<3>("r", &r, FinitePolicy::RequireFinite)?;
    let v = fixed_array::<3>("v", &v, FinitePolicy::RequireFinite)?;
    core_rv2coe(r, v, mu)
        .map(PyClassicalElements::from_inner)
        .map_err(to_elements_err)
}

/// Convert classical orbital elements to an inertial Cartesian state.
///
/// Returns `(position_km, velocity_km_s)` as numpy `(3,)` arrays. `mu` is the
/// gravitational parameter (km^3/s^2). Raises `ValueError` on invalid input.
#[pyfunction]
#[pyo3(signature = (elements, mu))]
fn coe2rv<'py>(
    py: Python<'py>,
    elements: &PyClassicalElements,
    mu: f64,
) -> PyResult<StateArrays<'py>> {
    let (r, v) = core_coe2rv(&elements.inner, mu).map_err(to_elements_err)?;
    Ok((np_array(py, &r), np_array(py, &v)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOrbitType>()?;
    m.add_class::<PyClassicalElements>()?;
    m.add_function(wrap_pyfunction!(rv2coe, m)?)?;
    m.add_function(wrap_pyfunction!(coe2rv, m)?)?;
    Ok(())
}
