//! Time of closest approach (TCA) finding, conjunction Pc, and catalog screening.
//!
//! Thin INTERFACE over `sidereon_core::astro::tca`. It marshals TLE strings, the
//! two-part Julian-date search window, the finder tolerances, and the
//! collision-probability options into the core TCA finders and screeners and
//! packages the results. Every number (the refined TCA, miss distance, relative
//! state, and Pc) is produced by the core; no orbital logic lives here.

use numpy::{PyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::astro::propagator::api::IntegratorOptions;
use sidereon_core::astro::propagator::{
    ForceModelKind, IntegratorKind as CoreIntegratorKind, ProcessNoise,
};
use sidereon_core::astro::sgp4::JulianDate;
use sidereon_core::astro::tca::{
    find_tca_candidates_from_tles as core_find_tca_candidates_from_tles,
    find_tca_conjunctions_from_tles as core_find_tca_conjunctions_from_tles,
    find_tca_conjunctions_with_propagated_covariance_from_tles as core_find_tca_conjunctions_with_propagated_covariance_from_tles,
    screen_tca_candidates_from_tle_catalog_serial as core_screen_candidates,
    screen_tca_conjunctions_from_tle_catalog_serial as core_screen_conjunctions,
    screen_tca_conjunctions_with_propagated_covariance_from_tle_catalog_serial as core_screen_conjunctions_with_propagated_covariance,
    TcaCandidate, TcaConjunction, TcaFinderOptions, TcaPcCovariances, TcaPcOptions,
    TcaPropagatedCovarianceOptions, TcaPropagatedCovariancePcOptions, TcaScreeningConjunctionHit,
    TcaScreeningHit, TcaTle, TcaTleWithCovariance, TcaWindow, DEFAULT_TCA_POSITION_COVARIANCE_KM2,
};

use crate::conjunction::PyPcMethod;
use crate::covariance::PyProcessNoise;
use crate::marshal::{
    covariance6_from_array, matrix3_from_array, option_py_or_default, FinitePolicy,
};
use crate::propagation::{PyForceModel, PyIntegrator};
use crate::{np_array, to_solve_err};

fn julian_date(jd: (f64, f64)) -> JulianDate {
    JulianDate(jd.0, jd.1)
}

fn finder_options(coarse_step_seconds: f64, time_tolerance_seconds: f64) -> TcaFinderOptions {
    TcaFinderOptions {
        coarse_step_seconds,
        time_tolerance_seconds,
    }
}

fn force_model_kind(force_model: PyForceModel, mu_km3_s2: Option<f64>) -> ForceModelKind {
    match force_model {
        PyForceModel::TWO_BODY => match mu_km3_s2 {
            Some(mu_km3_s2) => ForceModelKind::TwoBody { mu_km3_s2 },
            None => ForceModelKind::two_body(),
        },
        PyForceModel::TWO_BODY_J2 => {
            let mut kind = ForceModelKind::two_body_j2();
            if let Some(mu) = mu_km3_s2 {
                if let ForceModelKind::TwoBodyJ2 { mu_km3_s2, .. } = &mut kind {
                    *mu_km3_s2 = mu;
                }
            }
            kind
        }
    }
}

fn integrator_kind(integrator: PyIntegrator) -> CoreIntegratorKind {
    match integrator {
        PyIntegrator::DP54 => CoreIntegratorKind::Dp54,
        PyIntegrator::RK4 => CoreIntegratorKind::Rk4,
    }
}

fn pc_options(
    hard_body_radius_km: f64,
    method: PyPcMethod,
    primary_covariance_km2: Option<PyReadonlyArray2<'_, f64>>,
    secondary_covariance_km2: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<TcaPcOptions> {
    let primary = match primary_covariance_km2 {
        Some(matrix) => matrix3_from_array(
            &matrix,
            "primary_covariance_km2",
            FinitePolicy::RequireFinite,
        )?,
        None => DEFAULT_TCA_POSITION_COVARIANCE_KM2,
    };
    let secondary = match secondary_covariance_km2 {
        Some(matrix) => matrix3_from_array(
            &matrix,
            "secondary_covariance_km2",
            FinitePolicy::RequireFinite,
        )?,
        None => DEFAULT_TCA_POSITION_COVARIANCE_KM2,
    };
    Ok(TcaPcOptions {
        hard_body_radius_km,
        method: method.into(),
        covariances: TcaPcCovariances {
            primary_covariance_km2: primary,
            secondary_covariance_km2: secondary,
        },
    })
}

/// One local time of closest approach candidate. The relative state is
/// `primary - secondary` in the TEME frame SGP4 returns.
#[pyclass(module = "sidereon._sidereon", name = "TcaCandidate")]
#[derive(Clone, Copy)]
pub struct PyTcaCandidate {
    inner: TcaCandidate,
}

#[pymethods]
impl PyTcaCandidate {
    /// Refined absolute TCA, recombined single-`float` Julian date.
    #[getter]
    fn tca_time_jd(&self) -> f64 {
        self.inner.tca_time.0 + self.inner.tca_time.1
    }

    /// Integer-day boundary of the refined TCA Julian date.
    #[getter]
    fn tca_time_jd_whole(&self) -> f64 {
        self.inner.tca_time.0
    }

    /// Residual day fraction of the refined TCA Julian date.
    #[getter]
    fn tca_time_jd_fraction(&self) -> f64 {
        self.inner.tca_time.1
    }

    /// Refined seconds since the search window start.
    #[getter]
    fn tca_seconds_since_window_start(&self) -> f64 {
        self.inner.tca_seconds_since_window_start
    }

    /// Miss distance (norm of the relative position), km.
    #[getter]
    fn miss_distance_km(&self) -> f64 {
        self.inner.miss_distance_km
    }

    /// Primary minus secondary TEME position as a numpy `(3,)` array, km.
    #[getter]
    fn relative_position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.relative_position_km)
    }

    /// Primary minus secondary TEME velocity as a numpy `(3,)` array, km/s.
    #[getter]
    fn relative_velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.relative_velocity_km_s)
    }

    fn __repr__(&self) -> String {
        format!(
            "TcaCandidate(miss_distance_km={:.6}, tca_seconds_since_window_start={:.6})",
            self.inner.miss_distance_km, self.inner.tca_seconds_since_window_start
        )
    }
}

/// A TCA candidate with the conjunction-module collision probability at that TCA.
#[pyclass(module = "sidereon._sidereon", name = "TcaConjunction")]
#[derive(Clone, Copy)]
pub struct PyTcaConjunction {
    inner: TcaConjunction,
}

#[pymethods]
impl PyTcaConjunction {
    /// Refined TCA candidate and miss-distance summary.
    #[getter]
    fn candidate(&self) -> PyTcaCandidate {
        PyTcaCandidate {
            inner: self.inner.candidate,
        }
    }

    /// Collision probability at the TCA.
    #[getter]
    fn pc(&self) -> f64 {
        self.inner.collision_probability.pc
    }

    /// Miss distance from the encounter-plane summary, km.
    #[getter]
    fn miss_km(&self) -> f64 {
        self.inner.collision_probability.miss_km
    }

    /// Relative speed at the encounter, km/s.
    #[getter]
    fn relative_speed_km_s(&self) -> f64 {
        self.inner.collision_probability.relative_speed_km_s
    }

    /// Encounter-plane standard deviation along x, km.
    #[getter]
    fn sigma_x_km(&self) -> f64 {
        self.inner.collision_probability.sigma_x_km
    }

    /// Encounter-plane standard deviation along z, km.
    #[getter]
    fn sigma_z_km(&self) -> f64 {
        self.inner.collision_probability.sigma_z_km
    }

    fn __repr__(&self) -> String {
        format!(
            "TcaConjunction(pc={:.6e}, miss_km={:.6})",
            self.inner.collision_probability.pc, self.inner.collision_probability.miss_km
        )
    }
}

/// One threshold-screening hit: the secondary catalog index and its TCA.
#[pyclass(module = "sidereon._sidereon", name = "TcaScreeningHit")]
#[derive(Clone, Copy)]
pub struct PyTcaScreeningHit {
    inner: TcaScreeningHit,
}

#[pymethods]
impl PyTcaScreeningHit {
    /// Index of the secondary satellite in the supplied catalog list.
    #[getter]
    fn secondary_index(&self) -> usize {
        self.inner.secondary_index
    }

    /// Refined TCA candidate at or below the miss-distance threshold.
    #[getter]
    fn candidate(&self) -> PyTcaCandidate {
        PyTcaCandidate {
            inner: self.inner.candidate,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TcaScreeningHit(secondary_index={}, miss_distance_km={:.6})",
            self.inner.secondary_index, self.inner.candidate.miss_distance_km
        )
    }
}

/// One threshold-screening hit with collision probability at the TCA.
#[pyclass(module = "sidereon._sidereon", name = "TcaScreeningConjunctionHit")]
#[derive(Clone, Copy)]
pub struct PyTcaScreeningConjunctionHit {
    inner: TcaScreeningConjunctionHit,
}

#[pymethods]
impl PyTcaScreeningConjunctionHit {
    /// Index of the secondary satellite in the supplied catalog list.
    #[getter]
    fn secondary_index(&self) -> usize {
        self.inner.secondary_index
    }

    /// TCA and Pc result for this threshold breach.
    #[getter]
    fn conjunction(&self) -> PyTcaConjunction {
        PyTcaConjunction {
            inner: self.inner.conjunction,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TcaScreeningConjunctionHit(secondary_index={}, pc={:.6e})",
            self.inner.secondary_index, self.inner.conjunction.collision_probability.pc
        )
    }
}

/// Find local TCA candidates between two TLEs over a two-part Julian-date window.
///
/// `window_start_jd` and `window_end_jd` are `(whole, fraction)` Julian dates.
/// `coarse_step_seconds` brackets local range minima; each is refined to
/// `time_tolerance_seconds`. Raises `SolveError` on TLE/SGP4 or finder failure.
#[pyfunction]
#[pyo3(signature = (
    primary_line1,
    primary_line2,
    secondary_line1,
    secondary_line2,
    window_start_jd,
    window_end_jd,
    coarse_step_seconds=60.0,
    time_tolerance_seconds=1.0e-3,
))]
#[allow(clippy::too_many_arguments)]
fn find_tca_candidates(
    primary_line1: &str,
    primary_line2: &str,
    secondary_line1: &str,
    secondary_line2: &str,
    window_start_jd: (f64, f64),
    window_end_jd: (f64, f64),
    coarse_step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyTcaCandidate>> {
    let candidates = core_find_tca_candidates_from_tles(
        primary_line1,
        primary_line2,
        secondary_line1,
        secondary_line2,
        julian_date(window_start_jd),
        julian_date(window_end_jd),
        finder_options(coarse_step_seconds, time_tolerance_seconds),
    )
    .map_err(to_solve_err)?;
    Ok(candidates
        .into_iter()
        .map(|inner| PyTcaCandidate { inner })
        .collect())
}

/// Find TCA candidates between two TLEs and compute Pc at each TCA.
///
/// `hard_body_radius_km` and `method` drive the conjunction Pc module;
/// `primary_covariance_km2` / `secondary_covariance_km2` are optional `(3, 3)`
/// GCRS position covariances (each defaulting to the 1 km^2 identity). Raises
/// `SolveError` on TLE/SGP4, finder, or Pc failure.
#[pyfunction]
#[pyo3(signature = (
    primary_line1,
    primary_line2,
    secondary_line1,
    secondary_line2,
    window_start_jd,
    window_end_jd,
    hard_body_radius_km,
    method=PyPcMethod::FOSTER_EQUAL_AREA,
    primary_covariance_km2=None,
    secondary_covariance_km2=None,
    coarse_step_seconds=60.0,
    time_tolerance_seconds=1.0e-3,
))]
#[allow(clippy::too_many_arguments)]
fn find_tca_conjunctions(
    primary_line1: &str,
    primary_line2: &str,
    secondary_line1: &str,
    secondary_line2: &str,
    window_start_jd: (f64, f64),
    window_end_jd: (f64, f64),
    hard_body_radius_km: f64,
    method: PyPcMethod,
    primary_covariance_km2: Option<PyReadonlyArray2<'_, f64>>,
    secondary_covariance_km2: Option<PyReadonlyArray2<'_, f64>>,
    coarse_step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyTcaConjunction>> {
    let pc = pc_options(
        hard_body_radius_km,
        method,
        primary_covariance_km2,
        secondary_covariance_km2,
    )?;
    let conjunctions = core_find_tca_conjunctions_from_tles(
        TcaTle::new(primary_line1, primary_line2),
        TcaTle::new(secondary_line1, secondary_line2),
        julian_date(window_start_jd),
        julian_date(window_end_jd),
        finder_options(coarse_step_seconds, time_tolerance_seconds),
        pc,
    )
    .map_err(to_solve_err)?;
    Ok(conjunctions
        .into_iter()
        .map(|inner| PyTcaConjunction { inner })
        .collect())
}

/// Find TCA conjunctions and propagate each object's initial 6x6 covariance to TCA.
#[pyfunction]
#[pyo3(signature = (
    primary_line1,
    primary_line2,
    secondary_line1,
    secondary_line2,
    primary_covariance0,
    secondary_covariance0,
    window_start_jd,
    window_end_jd,
    hard_body_radius_km,
    method=PyPcMethod::FOSTER_EQUAL_AREA,
    force_model=PyForceModel::TWO_BODY_J2,
    integrator=PyIntegrator::DP54,
    abs_tol=1.0e-9,
    rel_tol=1.0e-12,
    initial_step_s=60.0,
    min_step_s=1.0e-6,
    max_step_s=3600.0,
    max_steps=1_000_000,
    mu_km3_s2=None,
    process_noise=None,
    coarse_step_seconds=60.0,
    time_tolerance_seconds=1.0e-3,
))]
#[allow(clippy::too_many_arguments)]
fn find_tca_conjunctions_with_propagated_covariance(
    py: Python<'_>,
    primary_line1: &str,
    primary_line2: &str,
    secondary_line1: &str,
    secondary_line2: &str,
    primary_covariance0: PyReadonlyArray2<'_, f64>,
    secondary_covariance0: PyReadonlyArray2<'_, f64>,
    window_start_jd: (f64, f64),
    window_end_jd: (f64, f64),
    hard_body_radius_km: f64,
    method: PyPcMethod,
    force_model: PyForceModel,
    integrator: PyIntegrator,
    abs_tol: f64,
    rel_tol: f64,
    initial_step_s: f64,
    min_step_s: f64,
    max_step_s: f64,
    max_steps: u32,
    mu_km3_s2: Option<f64>,
    process_noise: Option<Py<PyProcessNoise>>,
    coarse_step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyTcaConjunction>> {
    let primary_covariance0 = covariance6_from_array(&primary_covariance0, "primary_covariance0")?;
    let secondary_covariance0 =
        covariance6_from_array(&secondary_covariance0, "secondary_covariance0")?;
    let process_noise = option_py_or_default(
        py,
        process_noise.as_ref(),
        PyProcessNoise::inner,
        ProcessNoise::default,
    );
    let pc_options = TcaPropagatedCovariancePcOptions::new(
        hard_body_radius_km,
        method.into(),
        primary_covariance0,
        secondary_covariance0,
    )
    .with_covariance_propagator(
        force_model_kind(force_model, mu_km3_s2),
        integrator_kind(integrator),
        IntegratorOptions {
            abs_tol,
            rel_tol,
            initial_step: initial_step_s,
            min_step: min_step_s,
            max_step: max_step_s,
            max_steps,
            dense_output: false,
        },
    )
    .with_process_noise(process_noise);

    let conjunctions = core_find_tca_conjunctions_with_propagated_covariance_from_tles(
        TcaTle::new(primary_line1, primary_line2),
        TcaTle::new(secondary_line1, secondary_line2),
        julian_date(window_start_jd),
        julian_date(window_end_jd),
        finder_options(coarse_step_seconds, time_tolerance_seconds),
        pc_options,
    )
    .map_err(to_solve_err)?;
    Ok(conjunctions
        .into_iter()
        .map(|inner| PyTcaConjunction { inner })
        .collect())
}

/// Screen a primary TLE against a catalog of secondary TLEs, keeping TCAs at or
/// below `miss_distance_threshold_km`.
///
/// `secondaries` is a list of `(line1, line2)` tuples; each returned hit carries
/// the secondary's index in that list. Raises `SolveError` on TLE/SGP4 or finder
/// failure.
#[pyfunction]
#[pyo3(signature = (
    primary_line1,
    primary_line2,
    secondaries,
    window_start_jd,
    window_end_jd,
    miss_distance_threshold_km,
    coarse_step_seconds=60.0,
    time_tolerance_seconds=1.0e-3,
))]
#[allow(clippy::too_many_arguments)]
fn screen_tca_candidates(
    primary_line1: &str,
    primary_line2: &str,
    secondaries: Vec<(String, String)>,
    window_start_jd: (f64, f64),
    window_end_jd: (f64, f64),
    miss_distance_threshold_km: f64,
    coarse_step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyTcaScreeningHit>> {
    let secondary_tles: Vec<TcaTle<'_>> = secondaries
        .iter()
        .map(|(line1, line2)| TcaTle::new(line1, line2))
        .collect();
    let hits = core_screen_candidates(
        TcaTle::new(primary_line1, primary_line2),
        &secondary_tles,
        TcaWindow::new(julian_date(window_start_jd), julian_date(window_end_jd)),
        miss_distance_threshold_km,
        finder_options(coarse_step_seconds, time_tolerance_seconds),
    )
    .map_err(to_solve_err)?;
    Ok(hits
        .into_iter()
        .map(|inner| PyTcaScreeningHit { inner })
        .collect())
}

/// Screen a primary TLE against a catalog of secondary TLEs and compute Pc for
/// each threshold breach.
///
/// `secondaries` is a list of `(line1, line2)` tuples. `hard_body_radius_km` and
/// `method` drive the Pc module; the optional `(3, 3)` covariances default to the
/// 1 km^2 identity. Raises `SolveError` on TLE/SGP4, finder, or Pc failure.
#[pyfunction]
#[pyo3(signature = (
    primary_line1,
    primary_line2,
    secondaries,
    window_start_jd,
    window_end_jd,
    miss_distance_threshold_km,
    hard_body_radius_km,
    method=PyPcMethod::FOSTER_EQUAL_AREA,
    primary_covariance_km2=None,
    secondary_covariance_km2=None,
    coarse_step_seconds=60.0,
    time_tolerance_seconds=1.0e-3,
))]
#[allow(clippy::too_many_arguments)]
fn screen_tca_conjunctions(
    primary_line1: &str,
    primary_line2: &str,
    secondaries: Vec<(String, String)>,
    window_start_jd: (f64, f64),
    window_end_jd: (f64, f64),
    miss_distance_threshold_km: f64,
    hard_body_radius_km: f64,
    method: PyPcMethod,
    primary_covariance_km2: Option<PyReadonlyArray2<'_, f64>>,
    secondary_covariance_km2: Option<PyReadonlyArray2<'_, f64>>,
    coarse_step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyTcaScreeningConjunctionHit>> {
    let pc = pc_options(
        hard_body_radius_km,
        method,
        primary_covariance_km2,
        secondary_covariance_km2,
    )?;
    let secondary_tles: Vec<TcaTle<'_>> = secondaries
        .iter()
        .map(|(line1, line2)| TcaTle::new(line1, line2))
        .collect();
    let hits = core_screen_conjunctions(
        TcaTle::new(primary_line1, primary_line2),
        &secondary_tles,
        TcaWindow::new(julian_date(window_start_jd), julian_date(window_end_jd)),
        miss_distance_threshold_km,
        finder_options(coarse_step_seconds, time_tolerance_seconds),
        pc,
    )
    .map_err(to_solve_err)?;
    Ok(hits
        .into_iter()
        .map(|inner| PyTcaScreeningConjunctionHit { inner })
        .collect())
}

/// Screen a TLE catalog and propagate each object's initial 6x6 covariance to TCA.
#[pyfunction]
#[pyo3(signature = (
    primary_line1,
    primary_line2,
    primary_covariance0,
    secondaries,
    window_start_jd,
    window_end_jd,
    miss_distance_threshold_km,
    hard_body_radius_km,
    method=PyPcMethod::FOSTER_EQUAL_AREA,
    force_model=PyForceModel::TWO_BODY_J2,
    integrator=PyIntegrator::DP54,
    abs_tol=1.0e-9,
    rel_tol=1.0e-12,
    initial_step_s=60.0,
    min_step_s=1.0e-6,
    max_step_s=3600.0,
    max_steps=1_000_000,
    mu_km3_s2=None,
    process_noise=None,
    coarse_step_seconds=60.0,
    time_tolerance_seconds=1.0e-3,
))]
#[allow(clippy::too_many_arguments)]
fn screen_tca_conjunctions_with_propagated_covariance(
    py: Python<'_>,
    primary_line1: &str,
    primary_line2: &str,
    primary_covariance0: PyReadonlyArray2<'_, f64>,
    secondaries: Vec<(String, String, PyReadonlyArray2<'_, f64>)>,
    window_start_jd: (f64, f64),
    window_end_jd: (f64, f64),
    miss_distance_threshold_km: f64,
    hard_body_radius_km: f64,
    method: PyPcMethod,
    force_model: PyForceModel,
    integrator: PyIntegrator,
    abs_tol: f64,
    rel_tol: f64,
    initial_step_s: f64,
    min_step_s: f64,
    max_step_s: f64,
    max_steps: u32,
    mu_km3_s2: Option<f64>,
    process_noise: Option<Py<PyProcessNoise>>,
    coarse_step_seconds: f64,
    time_tolerance_seconds: f64,
) -> PyResult<Vec<PyTcaScreeningConjunctionHit>> {
    let primary_covariance0 = covariance6_from_array(&primary_covariance0, "primary_covariance0")?;
    let mut secondary_tles = Vec::with_capacity(secondaries.len());
    for (index, (line1, line2, covariance0)) in secondaries.iter().enumerate() {
        let covariance0 = covariance6_from_array(covariance0, &format!("secondaries[{index}][2]"))?;
        secondary_tles.push(TcaTleWithCovariance::new(line1, line2, covariance0));
    }
    let process_noise = option_py_or_default(
        py,
        process_noise.as_ref(),
        PyProcessNoise::inner,
        ProcessNoise::default,
    );
    let pc_options = TcaPropagatedCovarianceOptions::new(hard_body_radius_km, method.into())
        .with_covariance_propagator(
            force_model_kind(force_model, mu_km3_s2),
            integrator_kind(integrator),
            IntegratorOptions {
                abs_tol,
                rel_tol,
                initial_step: initial_step_s,
                min_step: min_step_s,
                max_step: max_step_s,
                max_steps,
                dense_output: false,
            },
        )
        .with_process_noise(process_noise);
    let hits = core_screen_conjunctions_with_propagated_covariance(
        TcaTleWithCovariance::new(primary_line1, primary_line2, primary_covariance0),
        &secondary_tles,
        TcaWindow::new(julian_date(window_start_jd), julian_date(window_end_jd)),
        miss_distance_threshold_km,
        finder_options(coarse_step_seconds, time_tolerance_seconds),
        pc_options,
    )
    .map_err(to_solve_err)?;
    Ok(hits
        .into_iter()
        .map(|inner| PyTcaScreeningConjunctionHit { inner })
        .collect())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTcaCandidate>()?;
    m.add_class::<PyTcaConjunction>()?;
    m.add_class::<PyTcaScreeningHit>()?;
    m.add_class::<PyTcaScreeningConjunctionHit>()?;
    m.add_function(wrap_pyfunction!(find_tca_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(find_tca_conjunctions, m)?)?;
    m.add_function(wrap_pyfunction!(
        find_tca_conjunctions_with_propagated_covariance,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(screen_tca_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(screen_tca_conjunctions, m)?)?;
    m.add_function(wrap_pyfunction!(
        screen_tca_conjunctions_with_propagated_covariance,
        m
    )?)?;
    Ok(())
}
