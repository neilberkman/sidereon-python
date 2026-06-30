//! Code-differential GNSS (DGPS) binding.
//!
//! Thin marshaling over [`sidereon_core::dgnss`]: a surveyed base station turns
//! its raw pseudoranges into per-satellite corrections (PRC), a rover applies
//! them, and the corrected rover pseudoranges feed a single-point solve with the
//! ionosphere/troposphere disabled (the differential already removed the common
//! path delays). No differencing or solve math lives here; the numbers are
//! exactly what `sidereon-core` produces.

use std::collections::BTreeMap;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::dgnss::{
    apply_corrections, pseudorange_corrections, solve_position, CodeObservation, DgnssError,
};

use crate::spp::PySppConfig;
use crate::{np_array, PySp3, SolveError};

/// Result of applying base corrections to rover observations: the corrected
/// `(token, pseudorange_m)` pairs in rover order, and the tokens dropped for want
/// of a matching correction.
type AppliedCorrections = (Vec<(String, f64)>, Vec<String>);

fn to_py_err(err: DgnssError) -> PyErr {
    match err {
        DgnssError::InvalidInput { field, reason } => {
            PyValueError::new_err(format!("invalid DGNSS input {field}: {reason}"))
        }
        DgnssError::Spp(spp) => SolveError::new_err(spp.to_string()),
    }
}

fn to_code_observations(observations: &[(String, f64)]) -> Vec<CodeObservation> {
    observations
        .iter()
        .map(|(sat, pr)| CodeObservation::new(sat.clone(), *pr))
        .collect()
}

/// Per-satellite pseudorange corrections (PRC) from a surveyed base station.
///
/// `base_position_m` is the known base ECEF position (metres); `base_observations`
/// is a list of `(satellite_token, pseudorange_m)` pairs; `t_rx_j2000_s` is the
/// receive time as continuous seconds since J2000. Returns a dict mapping each
/// correctable satellite token to its correction in metres. Satellites whose
/// ephemeris cannot be evaluated at this epoch are dropped (they cannot be
/// corrected). Raises `ValueError` on malformed input.
#[pyfunction]
fn dgnss_pseudorange_corrections(
    sp3: &PySp3,
    base_position_m: [f64; 3],
    base_observations: Vec<(String, f64)>,
    t_rx_j2000_s: f64,
) -> PyResult<BTreeMap<String, f64>> {
    pseudorange_corrections(
        &sp3.inner,
        base_position_m,
        &to_code_observations(&base_observations),
        t_rx_j2000_s,
    )
    .map_err(to_py_err)
}

/// Apply base pseudorange corrections to rover observations by satellite token.
///
/// `rover_observations` is a list of `(satellite_token, pseudorange_m)` pairs;
/// `corrections` is the PRC dict from [`dgnss_pseudorange_corrections`]. Returns
/// `(corrected, dropped)` where `corrected` is a list of `(token, pseudorange_m)`
/// in rover order and `dropped` lists rover tokens that had no matching
/// correction. Raises `ValueError` on malformed input.
#[pyfunction]
fn dgnss_apply_corrections(
    rover_observations: Vec<(String, f64)>,
    corrections: BTreeMap<String, f64>,
) -> PyResult<AppliedCorrections> {
    let applied = apply_corrections(&to_code_observations(&rover_observations), &corrections)
        .map_err(to_py_err)?;
    let corrected = applied
        .corrected
        .into_iter()
        .map(|obs| (obs.satellite_id, obs.pseudorange_m))
        .collect();
    Ok((corrected, applied.dropped))
}

/// A DGNSS rover solve result.
#[pyclass(module = "sidereon._sidereon", name = "DgnssSolution")]
pub struct PyDgnssSolution {
    inner: sidereon_core::dgnss::PositionSolution,
}

#[pymethods]
impl PyDgnssSolution {
    /// Corrected rover ECEF position as a numpy array `[x_m, y_m, z_m]`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let p = &self.inner.solution.position;
        np_array(py, &[p.x_m, p.y_m, p.z_m])
    }

    /// Receiver clock bias in seconds.
    #[getter]
    fn rx_clock_s(&self) -> f64 {
        self.inner.solution.rx_clock_s
    }

    /// `(lat_rad, lon_rad, height_m)` if the solve was asked for geodetic.
    #[getter]
    fn geodetic(&self) -> Option<(f64, f64, f64)> {
        self.inner
            .solution
            .geodetic
            .map(|g| (g.lat_rad, g.lon_rad, g.height_m))
    }

    /// Satellite tokens used in the accepted solution.
    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner
            .solution
            .used_sats
            .iter()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// Post-fit residuals in metres, index-aligned to `used_sats`.
    #[getter]
    fn residuals_m(&self) -> Vec<f64> {
        self.inner.solution.residuals_m.clone()
    }

    /// Rover-minus-base ECEF baseline vector as a numpy array `[dx, dy, dz]` (metres).
    #[getter]
    fn baseline_vector_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.baseline_vector_m)
    }

    /// Baseline length in metres.
    #[getter]
    fn baseline_m(&self) -> f64 {
        self.inner.baseline_m
    }

    /// Rover satellite tokens without a matching base correction.
    #[getter]
    fn dropped_sats(&self) -> Vec<String> {
        self.inner.dropped_sats.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DgnssSolution(baseline_m={:.3}, used_sats={}, dropped={})",
            self.inner.baseline_m,
            self.inner.solution.used_sats.len(),
            self.inner.dropped_sats.len()
        )
    }
}

/// Compute DGNSS corrections, apply them to rover observations, and solve.
///
/// `base_position_m` is the surveyed base ECEF position (metres);
/// `base_observations` and `rover_observations` are lists of
/// `(satellite_token, pseudorange_m)`; `config` is an `SppConfig` supplying the
/// receive-time scalars, initial guess, meteorology, and Klobuchar coefficients
/// (its own observations and correction switches are ignored: DGNSS solves the
/// corrected rover pseudoranges with the ionosphere and troposphere disabled).
/// Returns a `DgnssSolution`. Raises `ValueError` on malformed input or
/// `SolveError` if the corrected solve fails.
#[pyfunction]
fn dgnss_solve(
    sp3: &PySp3,
    base_position_m: [f64; 3],
    base_observations: Vec<(String, f64)>,
    rover_observations: Vec<(String, f64)>,
    config: &PySppConfig,
) -> PyResult<PyDgnssSolution> {
    let inner = solve_position(
        &sp3.inner,
        base_position_m,
        &to_code_observations(&base_observations),
        &to_code_observations(&rover_observations),
        config.to_inputs(),
        config.with_geodetic_flag(),
    )
    .map_err(to_py_err)?;
    Ok(PyDgnssSolution { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDgnssSolution>()?;
    m.add_function(wrap_pyfunction!(dgnss_pseudorange_corrections, m)?)?;
    m.add_function(wrap_pyfunction!(dgnss_apply_corrections, m)?)?;
    m.add_function(wrap_pyfunction!(dgnss_solve, m)?)?;
    Ok(())
}
