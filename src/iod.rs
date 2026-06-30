//! Initial orbit determination (IOD) binding.
//!
//! Thin marshaling over [`sidereon_core::astro::iod`]: the Gibbs and
//! Herrick-Gibbs three-position velocity solvers and Gauss angles-only orbit
//! determination. numpy vectors in, numpy vectors plus scalar diagnostics out.
//! All numeric logic lives in the core engine; this layer only groups arguments
//! into the arrays the core entry points expect and maps the typed error onto
//! `SolveError`.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::iod::{
    gauss_angles as core_gauss, gibbs as core_gibbs, hgibbs as core_hgibbs,
};

use crate::marshal::{fixed_array, matrix3_from_array, ArrayPairF64, FinitePolicy};
use crate::{np_array, to_solve_err};

/// Gibbs three-position velocity solve.
///
/// Given three coplanar geocentric position vectors `r1`, `r2`, `r3` (numpy
/// `(3,)`, km), return `(v2, theta12_rad, theta23_rad, coplanarity_rad)`: the
/// velocity at `r2` (numpy `(3,)`, km/s), the angles between successive position
/// vectors, and the coplanarity angle. Algorithm 54 (Vallado). Raises
/// `ValueError` on a malformed shape and `SolveError` on degenerate geometry.
#[pyfunction]
#[pyo3(signature = (r1, r2, r3))]
fn gibbs<'py>(
    py: Python<'py>,
    r1: PyReadonlyArray1<'_, f64>,
    r2: PyReadonlyArray1<'_, f64>,
    r3: PyReadonlyArray1<'_, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64, f64)> {
    let r1 = fixed_array::<3>("r1", &r1, FinitePolicy::RequireFinite)?;
    let r2 = fixed_array::<3>("r2", &r2, FinitePolicy::RequireFinite)?;
    let r3 = fixed_array::<3>("r3", &r3, FinitePolicy::RequireFinite)?;
    let (v2, theta12, theta23, copa) = core_gibbs(&r1, &r2, &r3).map_err(to_solve_err)?;
    Ok((np_array(py, &v2), theta12, theta23, copa))
}

/// Herrick-Gibbs three-position velocity solve.
///
/// Given three closely-spaced geocentric position vectors `r1`, `r2`, `r3`
/// (numpy `(3,)`, km) and their Julian-date epochs `jd1`, `jd2`, `jd3` (days),
/// return `(v2, theta12_rad, theta23_rad, coplanarity_rad)`: the velocity at
/// `r2` (numpy `(3,)`, km/s), the angles between successive position vectors,
/// and the coplanarity angle. The Taylor-series companion to [`gibbs`] for
/// tightly-spaced observations. Raises `ValueError` on a malformed shape and
/// `SolveError` on degenerate geometry or time spacing.
#[pyfunction]
#[pyo3(signature = (r1, r2, r3, jd1, jd2, jd3))]
fn hgibbs<'py>(
    py: Python<'py>,
    r1: PyReadonlyArray1<'_, f64>,
    r2: PyReadonlyArray1<'_, f64>,
    r3: PyReadonlyArray1<'_, f64>,
    jd1: f64,
    jd2: f64,
    jd3: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64, f64, f64)> {
    let r1 = fixed_array::<3>("r1", &r1, FinitePolicy::RequireFinite)?;
    let r2 = fixed_array::<3>("r2", &r2, FinitePolicy::RequireFinite)?;
    let r3 = fixed_array::<3>("r3", &r3, FinitePolicy::RequireFinite)?;
    let (v2, theta12, theta23, copa) =
        core_hgibbs(&r1, &r2, &r3, jd1, jd2, jd3).map_err(to_solve_err)?;
    Ok((np_array(py, &v2), theta12, theta23, copa))
}

/// Gauss angles-only orbit determination.
///
/// Given three angular observations - declination `decl` and right ascension
/// `rtasc` (each numpy `(3,)`, radians) - with split Julian dates `jd` (whole
/// part) and `jdf` (fraction) (each numpy `(3,)`, days) and the observer site
/// ECI positions `rseci` (numpy `(3, 3)`, one row per epoch, km), determine the
/// orbit at the middle observation. Returns `(r2, v2)`: the position (numpy
/// `(3,)`, km) and velocity (numpy `(3,)`, km/s) at the middle epoch. Algorithm
/// 52 (Vallado). Raises `ValueError` on a malformed shape and `SolveError` on
/// degenerate geometry or a non-converging slant-range root solve.
#[pyfunction]
#[pyo3(signature = (decl, rtasc, jd, jdf, rseci))]
fn gauss_angles<'py>(
    py: Python<'py>,
    decl: PyReadonlyArray1<'_, f64>,
    rtasc: PyReadonlyArray1<'_, f64>,
    jd: PyReadonlyArray1<'_, f64>,
    jdf: PyReadonlyArray1<'_, f64>,
    rseci: PyReadonlyArray2<'_, f64>,
) -> PyResult<ArrayPairF64<'py>> {
    let decl = fixed_array::<3>("decl", &decl, FinitePolicy::RequireFinite)?;
    let rtasc = fixed_array::<3>("rtasc", &rtasc, FinitePolicy::RequireFinite)?;
    let jd = fixed_array::<3>("jd", &jd, FinitePolicy::RequireFinite)?;
    let jdf = fixed_array::<3>("jdf", &jdf, FinitePolicy::RequireFinite)?;
    let rseci = matrix3_from_array(&rseci, "rseci", FinitePolicy::RequireFinite)?;
    let (r2, v2) = core_gauss(&decl, &rtasc, &jd, &jdf, &rseci).map_err(to_solve_err)?;
    Ok((np_array(py, &r2), np_array(py, &v2)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gibbs, m)?)?;
    m.add_function(wrap_pyfunction!(hgibbs, m)?)?;
    m.add_function(wrap_pyfunction!(gauss_angles, m)?)?;
    Ok(())
}
