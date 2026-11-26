//! Closed-form zonal orbit propagation and mean element conversions.

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::propagator::analytic::{self, AnalyticPropagationError as CoreError};

use crate::relative::PyCartesianState;
use crate::{binding_error, SolveError};

create_exception!(
    _sidereon,
    AnalyticPropagationError,
    SolveError,
    "An analytic propagation or element conversion failure."
);

/// Supported zonal propagation theories.
#[pyclass(module = "sidereon._sidereon", name = "AnalyticTheory", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyAnalyticTheory {
    /// Eckstein-Hechler near-circular orbit theory.
    ECKSTEIN_HECHLER,
    /// Brouwer-Lyddane elliptic orbit theory.
    BROUWER_LYDDANE,
}

impl From<analytic::AnalyticTheory> for PyAnalyticTheory {
    fn from(value: analytic::AnalyticTheory) -> Self {
        match value {
            analytic::AnalyticTheory::EcksteinHechler => Self::ECKSTEIN_HECHLER,
            analytic::AnalyticTheory::BrouwerLyddane => Self::BROUWER_LYDDANE,
        }
    }
}

#[pymethods]
impl PyAnalyticTheory {
    fn __str__(&self) -> String {
        match self {
            Self::ECKSTEIN_HECHLER => analytic::AnalyticTheory::EcksteinHechler,
            Self::BROUWER_LYDDANE => analytic::AnalyticTheory::BrouwerLyddane,
        }
        .to_string()
    }
}

fn analytic_error(error: CoreError) -> PyErr {
    let exception = AnalyticPropagationError::new_err(error.to_string());
    match error {
        CoreError::NonFinite { field } => binding_error(exception, "non_finite", |detail| {
            detail.set_item("field", field)
        }),
        CoreError::InvalidParameter { field, reason } => {
            binding_error(exception, "invalid_parameter", |detail| {
                detail.set_item("field", field)?;
                detail.set_item("reason", reason)
            })
        }
        CoreError::NotElliptic { field } => binding_error(exception, "not_elliptic", |detail| {
            detail.set_item("field", field)
        }),
        CoreError::RetrogradeEquatorial { inclination_rad } => {
            binding_error(exception, "retrograde_equatorial", |detail| {
                detail.set_item("inclination_rad", inclination_rad)
            })
        }
        CoreError::InsideReferenceSphere {
            semi_major_axis_km,
            reference_radius_km,
        } => binding_error(exception, "inside_reference_sphere", |detail| {
            detail.set_item("semi_major_axis_km", semi_major_axis_km)?;
            detail.set_item("reference_radius_km", reference_radius_km)
        }),
        CoreError::EccentricityTooLarge {
            eccentricity,
            limit,
        } => binding_error(exception, "eccentricity_too_large", |detail| {
            detail.set_item("eccentricity", eccentricity)?;
            detail.set_item("limit", limit)
        }),
        CoreError::NearlyEquatorial { inclination_rad } => {
            binding_error(exception, "nearly_equatorial", |detail| {
                detail.set_item("inclination_rad", inclination_rad)
            })
        }
        CoreError::NearCriticalInclination { inclination_rad } => {
            binding_error(exception, "near_critical_inclination", |detail| {
                detail.set_item("inclination_rad", inclination_rad)
            })
        }
        CoreError::MeanElementsNotConverged { theory, iterations } => {
            binding_error(exception, "mean_elements_not_converged", |detail| {
                detail.set_item("theory", PyAnalyticTheory::from(theory))?;
                detail.set_item("iterations", iterations)
            })
        }
        CoreError::KeplerNotConverged { field } => {
            binding_error(exception, "kepler_not_converged", |detail| {
                detail.set_item("field", field)
            })
        }
        CoreError::NonFiniteResult { theory, field } => {
            binding_error(exception, "non_finite_result", |detail| {
                detail.set_item("theory", PyAnalyticTheory::from(theory))?;
                detail.set_item("field", field)
            })
        }
    }
}

/// Circular mean or osculating orbital elements.
#[pyclass(module = "sidereon._sidereon", name = "CircularElements", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyCircularElements(analytic::CircularElements);

#[pymethods]
impl PyCircularElements {
    /// Construct an element, gravity, or convergence value from its fields.
    #[new]
    fn new(
        a_km: f64,
        ex: f64,
        ey: f64,
        i_rad: f64,
        raan_rad: f64,
        mean_latitude_argument_rad: f64,
    ) -> Self {
        Self(analytic::CircularElements {
            a_km,
            ex,
            ey,
            i_rad,
            raan_rad,
            mean_latitude_argument_rad,
        })
    }

    /// Semi-major axis in kilometres.
    #[getter]
    fn a_km(&self) -> f64 {
        self.0.a_km
    }

    /// Eccentricity times cosine of argument of perigee.
    #[getter]
    fn ex(&self) -> f64 {
        self.0.ex
    }

    /// Eccentricity times sine of argument of perigee.
    #[getter]
    fn ey(&self) -> f64 {
        self.0.ey
    }

    /// Inclination in radians.
    #[getter]
    fn i_rad(&self) -> f64 {
        self.0.i_rad
    }

    /// Right ascension of the ascending node in radians.
    #[getter]
    fn raan_rad(&self) -> f64 {
        self.0.raan_rad
    }

    /// Mean argument of latitude in radians.
    #[getter]
    fn mean_latitude_argument_rad(&self) -> f64 {
        self.0.mean_latitude_argument_rad
    }

    /// Convert a Cartesian position and velocity to orbital elements.
    #[staticmethod]
    fn from_cartesian(
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        mu_km3_s2: f64,
    ) -> PyResult<Self> {
        analytic::CircularElements::from_cartesian(position_km, velocity_km_s, mu_km3_s2)
            .map(Self)
            .map_err(analytic_error)
    }

    /// Convert the orbital elements to position in km and velocity in km/s.
    fn to_cartesian(&self, mu_km3_s2: f64) -> PyResult<([f64; 3], [f64; 3])> {
        self.0.to_cartesian(mu_km3_s2).map_err(analytic_error)
    }

    /// Magnitude of the eccentricity vector.
    fn eccentricity(&self) -> f64 {
        self.0.eccentricity()
    }
}

/// Keplerian elements with mean anomaly for analytic propagation.
#[pyclass(
    module = "sidereon._sidereon",
    name = "AnalyticKeplerianElements",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyAnalyticKeplerianElements(analytic::KeplerianElements);

#[pymethods]
impl PyAnalyticKeplerianElements {
    /// Construct an element, gravity, or convergence value from its fields.
    #[new]
    fn new(
        a_km: f64,
        e: f64,
        i_rad: f64,
        arg_perigee_rad: f64,
        raan_rad: f64,
        mean_anomaly_rad: f64,
    ) -> Self {
        Self(analytic::KeplerianElements {
            a_km,
            e,
            i_rad,
            arg_perigee_rad,
            raan_rad,
            mean_anomaly_rad,
        })
    }

    /// Semi-major axis in kilometres.
    #[getter]
    fn a_km(&self) -> f64 {
        self.0.a_km
    }

    /// Eccentricity.
    #[getter]
    fn e(&self) -> f64 {
        self.0.e
    }

    /// Inclination in radians.
    #[getter]
    fn i_rad(&self) -> f64 {
        self.0.i_rad
    }

    /// Argument of perigee in radians.
    #[getter]
    fn arg_perigee_rad(&self) -> f64 {
        self.0.arg_perigee_rad
    }

    /// Right ascension of the ascending node in radians.
    #[getter]
    fn raan_rad(&self) -> f64 {
        self.0.raan_rad
    }

    /// Mean anomaly in radians.
    #[getter]
    fn mean_anomaly_rad(&self) -> f64 {
        self.0.mean_anomaly_rad
    }

    /// Convert a Cartesian position and velocity to orbital elements.
    #[staticmethod]
    fn from_cartesian(
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        mu_km3_s2: f64,
    ) -> PyResult<Self> {
        analytic::KeplerianElements::from_cartesian(position_km, velocity_km_s, mu_km3_s2)
            .map(Self)
            .map_err(analytic_error)
    }

    /// Convert the orbital elements to position in km and velocity in km/s.
    fn to_cartesian(&self, mu_km3_s2: f64) -> PyResult<([f64; 3], [f64; 3])> {
        self.0.to_cartesian(mu_km3_s2).map_err(analytic_error)
    }
}

/// Unnormalized zonal coefficients and associated gravity constants.
#[pyclass(module = "sidereon._sidereon", name = "ZonalHarmonics", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyZonalHarmonics(analytic::ZonalHarmonics);

#[pymethods]
impl PyZonalHarmonics {
    /// Construct an element, gravity, or convergence value from its fields.
    #[new]
    fn new(mu_km3_s2: f64, re_km: f64, c20: f64, c30: f64, c40: f64, c50: f64, c60: f64) -> Self {
        Self(analytic::ZonalHarmonics {
            mu_km3_s2,
            re_km,
            c20,
            c30,
            c40,
            c50,
            c60,
        })
    }

    /// Gravitational parameter in cubic kilometres per square second.
    #[getter]
    fn mu_km3_s2(&self) -> f64 {
        self.0.mu_km3_s2
    }

    /// Reference radius in kilometres.
    #[getter]
    fn re_km(&self) -> f64 {
        self.0.re_km
    }

    /// Unnormalized C20 coefficient.
    #[getter]
    fn c20(&self) -> f64 {
        self.0.c20
    }

    /// Unnormalized C30 coefficient.
    #[getter]
    fn c30(&self) -> f64 {
        self.0.c30
    }

    /// Unnormalized C40 coefficient.
    #[getter]
    fn c40(&self) -> f64 {
        self.0.c40
    }

    /// Unnormalized C50 coefficient.
    #[getter]
    fn c50(&self) -> f64 {
        self.0.c50
    }

    /// Unnormalized C60 coefficient.
    #[getter]
    fn c60(&self) -> f64 {
        self.0.c60
    }

    /// Default zonal gravity constants.
    #[staticmethod]
    fn default() -> Self {
        Self(analytic::ZonalHarmonics::default())
    }

    /// Standard Earth zonal gravity constants.
    #[staticmethod]
    fn earth() -> Self {
        Self(analytic::ZonalHarmonics::earth())
    }
}

/// Convergence controls for conversion to mean orbital elements.
#[pyclass(
    module = "sidereon._sidereon",
    name = "MeanConversionOptions",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyMeanConversionOptions(analytic::MeanConversionOptions);

#[pymethods]
impl PyMeanConversionOptions {
    /// Construct an element, gravity, or convergence value from its fields.
    #[new]
    fn new(tolerance: f64, max_iterations: u32) -> Self {
        Self(analytic::MeanConversionOptions::new(
            tolerance,
            max_iterations,
        ))
    }

    /// Relative convergence tolerance.
    #[getter]
    fn tolerance(&self) -> f64 {
        self.0.tolerance
    }

    /// Maximum iteration count.
    #[getter]
    fn max_iterations(&self) -> u32 {
        self.0.max_iterations
    }

    /// Default options for the eckstein-hechler theory.
    #[classattr]
    const ECKSTEIN_HECHLER_DEFAULT: Self =
        Self(analytic::MeanConversionOptions::ECKSTEIN_HECHLER_DEFAULT);

    /// Default options for the brouwer-lyddane theory.
    #[classattr]
    const BROUWER_LYDDANE_DEFAULT: Self =
        Self(analytic::MeanConversionOptions::BROUWER_LYDDANE_DEFAULT);
}

/// Converged mean elements and the iteration count.
#[pyclass(module = "sidereon._sidereon", name = "MeanElements", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyMeanElements(MeanElementsValue);

#[pymethods]
impl PyMeanElements {
    /// Mean orbital elements in the theory's parameterization.
    #[getter]
    fn elements(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.0 {
            MeanElementsValue::Circular(result) => {
                Ok(Py::new(py, PyCircularElements(result.elements))?.into_any())
            }
            MeanElementsValue::Keplerian(result) => {
                Ok(Py::new(py, PyAnalyticKeplerianElements(result.elements))?.into_any())
            }
        }
    }

    /// Number of fixed-point iterations.
    #[getter]
    fn iterations(&self) -> u32 {
        match &self.0 {
            MeanElementsValue::Circular(result) => result.iterations,
            MeanElementsValue::Keplerian(result) => result.iterations,
        }
    }
}

#[derive(Clone, PartialEq)]
enum MeanElementsValue {
    Circular(analytic::MeanElements<analytic::CircularElements>),
    Keplerian(analytic::MeanElements<analytic::KeplerianElements>),
}

/// Eckstein-Hechler analytic orbit propagator.
#[pyclass(module = "sidereon._sidereon", name = "EcksteinHechlerPropagator")]
pub struct PyEcksteinHechlerPropagator(pub(crate) analytic::EcksteinHechlerPropagator);

#[pymethods]
impl PyEcksteinHechlerPropagator {
    /// Build from an osculating Cartesian state and convert to mean elements.
    #[new]
    fn new(
        initial: &PyCartesianState,
        gravity: &PyZonalHarmonics,
        options: &PyMeanConversionOptions,
    ) -> PyResult<Self> {
        analytic::EcksteinHechlerPropagator::new(initial.inner(), &gravity.0, options.0)
            .map(Self)
            .map_err(analytic_error)
    }

    /// Build from mean orbital elements at a TDB epoch.
    #[staticmethod]
    fn from_mean_elements(
        epoch_tdb_seconds: f64,
        mean: &PyCircularElements,
        gravity: &PyZonalHarmonics,
    ) -> PyResult<Self> {
        analytic::EcksteinHechlerPropagator::from_mean_elements(
            epoch_tdb_seconds,
            mean.0,
            &gravity.0,
        )
        .map(Self)
        .map_err(analytic_error)
    }

    /// Epoch of the mean orbital elements in TDB seconds from J2000.
    fn epoch_tdb_seconds(&self) -> f64 {
        self.0.epoch_tdb_seconds()
    }

    /// Mean orbital elements at the initial epoch.
    fn mean_elements(&self) -> PyCircularElements {
        PyCircularElements(self.0.mean_elements())
    }

    /// Zonal gravity constants.
    fn gravity(&self) -> PyZonalHarmonics {
        PyZonalHarmonics(*self.0.gravity())
    }

    /// Evaluate osculating orbital elements at the given TDB epoch.
    fn osculating_elements(&self, epoch_tdb_seconds: f64) -> PyResult<PyCircularElements> {
        self.0
            .osculating_elements(epoch_tdb_seconds)
            .map(PyCircularElements)
            .map_err(analytic_error)
    }

    /// Evaluate the Cartesian state at the given TDB epoch.
    fn propagate(&self, epoch_tdb_seconds: f64) -> PyResult<PyCartesianState> {
        self.0
            .propagate(epoch_tdb_seconds)
            .map(PyCartesianState::from_inner)
            .map_err(analytic_error)
    }

    /// Evaluate Cartesian states at the given TDB epochs in input order.
    fn ephemeris(&self, epochs_tdb_seconds: Vec<f64>) -> PyResult<Vec<PyCartesianState>> {
        self.0
            .ephemeris(&epochs_tdb_seconds)
            .map(|states| {
                states
                    .into_iter()
                    .map(PyCartesianState::from_inner)
                    .collect()
            })
            .map_err(analytic_error)
    }
}

/// Convert osculating Cartesian state to Eckstein-Hechler mean elements.
#[pyfunction]
fn eckstein_hechler_mean_elements(
    osculating: &PyCartesianState,
    gravity: &PyZonalHarmonics,
    options: &PyMeanConversionOptions,
) -> PyResult<PyMeanElements> {
    analytic::eckstein_hechler_mean_elements(osculating.inner(), &gravity.0, options.0)
        .map(|result| PyMeanElements(MeanElementsValue::Circular(result)))
        .map_err(analytic_error)
}

/// Convert Eckstein-Hechler mean elements to osculating elements.
#[pyfunction]
fn eckstein_hechler_osculating_elements(
    mean: &PyCircularElements,
    gravity: &PyZonalHarmonics,
) -> PyResult<PyCircularElements> {
    analytic::eckstein_hechler_osculating_elements(&mean.0, &gravity.0)
        .map(PyCircularElements)
        .map_err(analytic_error)
}

/// Brouwer-Lyddane analytic orbit propagator.
#[pyclass(module = "sidereon._sidereon", name = "BrouwerLyddanePropagator")]
pub struct PyBrouwerLyddanePropagator(pub(crate) analytic::BrouwerLyddanePropagator);

#[pymethods]
impl PyBrouwerLyddanePropagator {
    /// Build from an osculating Cartesian state and convert to mean elements.
    #[new]
    fn new(
        initial: &PyCartesianState,
        gravity: &PyZonalHarmonics,
        m2_rad_s2: f64,
        options: &PyMeanConversionOptions,
    ) -> PyResult<Self> {
        analytic::BrouwerLyddanePropagator::new(initial.inner(), &gravity.0, m2_rad_s2, options.0)
            .map(Self)
            .map_err(analytic_error)
    }

    /// Build from mean orbital elements at a TDB epoch.
    #[staticmethod]
    fn from_mean_elements(
        epoch_tdb_seconds: f64,
        mean: &PyAnalyticKeplerianElements,
        gravity: &PyZonalHarmonics,
        m2_rad_s2: f64,
    ) -> PyResult<Self> {
        analytic::BrouwerLyddanePropagator::from_mean_elements(
            epoch_tdb_seconds,
            mean.0,
            &gravity.0,
            m2_rad_s2,
        )
        .map(Self)
        .map_err(analytic_error)
    }

    /// Epoch of the mean orbital elements in TDB seconds from J2000.
    fn epoch_tdb_seconds(&self) -> f64 {
        self.0.epoch_tdb_seconds()
    }

    /// Mean orbital elements at the initial epoch.
    fn mean_elements(&self) -> PyAnalyticKeplerianElements {
        PyAnalyticKeplerianElements(self.0.mean_elements())
    }

    /// Zonal gravity constants.
    fn gravity(&self) -> PyZonalHarmonics {
        PyZonalHarmonics(*self.0.gravity())
    }

    /// Evaluate osculating orbital elements at the given TDB epoch.
    fn osculating_elements(&self, epoch_tdb_seconds: f64) -> PyResult<PyAnalyticKeplerianElements> {
        self.0
            .osculating_elements(epoch_tdb_seconds)
            .map(PyAnalyticKeplerianElements)
            .map_err(analytic_error)
    }

    /// Evaluate the Cartesian state at the given TDB epoch.
    fn propagate(&self, epoch_tdb_seconds: f64) -> PyResult<PyCartesianState> {
        self.0
            .propagate(epoch_tdb_seconds)
            .map(PyCartesianState::from_inner)
            .map_err(analytic_error)
    }

    /// Evaluate Cartesian states at the given TDB epochs in input order.
    fn ephemeris(&self, epochs_tdb_seconds: Vec<f64>) -> PyResult<Vec<PyCartesianState>> {
        self.0
            .ephemeris(&epochs_tdb_seconds)
            .map(|states| {
                states
                    .into_iter()
                    .map(PyCartesianState::from_inner)
                    .collect()
            })
            .map_err(analytic_error)
    }

    /// Secular along-track coefficient in radians per square second.
    fn m2_rad_s2(&self) -> f64 {
        self.0.m2_rad_s2()
    }
}

/// Convert osculating Cartesian state to Brouwer-Lyddane mean elements.
#[pyfunction]
fn brouwer_lyddane_mean_elements(
    osculating: &PyCartesianState,
    gravity: &PyZonalHarmonics,
    options: &PyMeanConversionOptions,
) -> PyResult<PyMeanElements> {
    analytic::brouwer_lyddane_mean_elements(osculating.inner(), &gravity.0, options.0)
        .map(|result| PyMeanElements(MeanElementsValue::Keplerian(result)))
        .map_err(analytic_error)
}

/// Convert Brouwer-Lyddane mean elements to osculating elements.
#[pyfunction]
fn brouwer_lyddane_osculating_elements(
    mean: &PyAnalyticKeplerianElements,
    gravity: &PyZonalHarmonics,
) -> PyResult<PyAnalyticKeplerianElements> {
    analytic::brouwer_lyddane_osculating_elements(&mean.0, &gravity.0)
        .map(PyAnalyticKeplerianElements)
        .map_err(analytic_error)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add(
        "AnalyticPropagationError",
        m.py().get_type::<AnalyticPropagationError>(),
    )?;
    m.add_class::<PyAnalyticTheory>()?;
    m.add_class::<PyCircularElements>()?;
    m.add_class::<PyAnalyticKeplerianElements>()?;
    m.add_class::<PyZonalHarmonics>()?;
    m.add_class::<PyMeanConversionOptions>()?;
    m.add_class::<PyMeanElements>()?;
    m.add_class::<PyEcksteinHechlerPropagator>()?;
    m.add_function(wrap_pyfunction!(eckstein_hechler_mean_elements, m)?)?;
    m.add_function(wrap_pyfunction!(eckstein_hechler_osculating_elements, m)?)?;
    m.add_class::<PyBrouwerLyddanePropagator>()?;
    m.add_function(wrap_pyfunction!(brouwer_lyddane_mean_elements, m)?)?;
    m.add_function(wrap_pyfunction!(brouwer_lyddane_osculating_elements, m)?)?;
    Ok(())
}
