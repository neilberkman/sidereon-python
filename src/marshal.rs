//! Shared PyO3/numpy marshalling helpers for the Python binding.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pyclass::PyClass;

use sidereon::passes::UtcInstant;
use sidereon_core::astro::covariance::{Covariance6, Covariance6Error};
use sidereon_core::astro::time::TimeScales;
use sidereon_core::GnssSystem;

/// A pair of 1-D `f64` numpy arrays returned across the FFI boundary, e.g. the
/// two state vectors produced by a Lambert or Gauss IOD solve.
pub(crate) type ArrayPairF64<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// GNSS constellation identifier shared across the RINEX and observable bindings.
#[pyclass(module = "sidereon._sidereon", name = "GnssSystem", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyGnssSystem {
    /// GPS, RINEX letter `G`.
    GPS,
    /// GLONASS, RINEX letter `R`.
    GLONASS,
    /// Galileo, RINEX letter `E`.
    GALILEO,
    /// BeiDou, RINEX letter `C`.
    BEIDOU,
    /// QZSS, RINEX letter `J`.
    QZSS,
    /// NavIC, RINEX letter `I`.
    NAVIC,
    /// SBAS, RINEX letter `S`.
    SBAS,
}

impl From<GnssSystem> for PyGnssSystem {
    fn from(system: GnssSystem) -> Self {
        match system {
            GnssSystem::Gps => Self::GPS,
            GnssSystem::Glonass => Self::GLONASS,
            GnssSystem::Galileo => Self::GALILEO,
            GnssSystem::BeiDou => Self::BEIDOU,
            GnssSystem::Qzss => Self::QZSS,
            GnssSystem::Navic => Self::NAVIC,
            GnssSystem::Sbas => Self::SBAS,
        }
    }
}

impl From<PyGnssSystem> for GnssSystem {
    fn from(system: PyGnssSystem) -> Self {
        match system {
            PyGnssSystem::GPS => GnssSystem::Gps,
            PyGnssSystem::GLONASS => GnssSystem::Glonass,
            PyGnssSystem::GALILEO => GnssSystem::Galileo,
            PyGnssSystem::BEIDOU => GnssSystem::BeiDou,
            PyGnssSystem::QZSS => GnssSystem::Qzss,
            PyGnssSystem::NAVIC => GnssSystem::Navic,
            PyGnssSystem::SBAS => GnssSystem::Sbas,
        }
    }
}

#[pymethods]
impl PyGnssSystem {
    /// Canonical RINEX one-letter system identifier.
    #[getter]
    pub fn letter(&self) -> char {
        GnssSystem::from(*self).letter()
    }

    /// Stable display name for the constellation.
    #[getter]
    pub fn label(&self) -> &'static str {
        GnssSystem::from(*self).as_str()
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::GPS => "GnssSystem.GPS",
            Self::GLONASS => "GnssSystem.GLONASS",
            Self::GALILEO => "GnssSystem.GALILEO",
            Self::BEIDOU => "GnssSystem.BEIDOU",
            Self::QZSS => "GnssSystem.QZSS",
            Self::NAVIC => "GnssSystem.NAVIC",
            Self::SBAS => "GnssSystem.SBAS",
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum EmptyPolicy {
    /// Accept empty arrays for pure map/sample APIs where the core naturally
    /// returns an empty result with the same outer shape.
    Allow,
    /// Reject empty arrays for APIs that need at least one sample, query, or
    /// interval endpoint to define the requested operation.
    Reject,
}

impl EmptyPolicy {
    fn rejects(self) -> bool {
        matches!(self, Self::Reject)
    }
}

#[derive(Clone, Copy)]
pub(crate) enum FinitePolicy {
    AllowNonFinite,
    RequireFinite,
}

impl FinitePolicy {
    fn requires(self) -> bool {
        matches!(self, Self::RequireFinite)
    }
}

pub(crate) fn option_py_or_default<T, U>(
    py: Python<'_>,
    value: Option<&Py<T>>,
    inner: impl FnOnce(&T) -> U,
    default: impl FnOnce() -> U,
) -> U
where
    T: PyClass,
{
    value
        .map(|value| {
            let borrowed = value.borrow(py);
            inner(&*borrowed)
        })
        .unwrap_or_else(default)
}

pub(crate) fn fixed_array<const N: usize>(
    name: &str,
    values: &PyReadonlyArray1<'_, f64>,
    finite: FinitePolicy,
) -> PyResult<[f64; N]> {
    let view = values.as_array();
    if view.len() != N {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape ({N},)"
        )));
    }

    let mut out = [0.0; N];
    for (index, value) in view.iter().copied().enumerate() {
        if finite.requires() && !value.is_finite() {
            return Err(PyValueError::new_err(format!(
                "{name}[{index}] must be finite"
            )));
        }
        out[index] = value;
    }
    Ok(out)
}

pub(crate) fn rows3_from_array(
    name: &str,
    arr: &PyReadonlyArray2<'_, f64>,
    empty: EmptyPolicy,
    finite: FinitePolicy,
) -> PyResult<Vec<[f64; 3]>> {
    let view = arr.as_array();
    if empty.rejects() && view.nrows() == 0 {
        return Err(PyValueError::new_err(format!("{name} array is empty")));
    }
    if view.ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 3), got (_, {})",
            view.ncols()
        )));
    }

    let mut rows = Vec::with_capacity(view.nrows());
    for (row_index, row) in view.outer_iter().enumerate() {
        let value = [row[0], row[1], row[2]];
        if finite.requires() && !value.iter().all(|x| x.is_finite()) {
            return Err(PyValueError::new_err(format!(
                "{name}[{row_index}] must contain only finite values"
            )));
        }
        rows.push(value);
    }
    Ok(rows)
}

pub(crate) fn matrix3_from_array(
    values: &PyReadonlyArray2<'_, f64>,
    name: &str,
    finite: FinitePolicy,
) -> PyResult<[[f64; 3]; 3]> {
    let view = values.as_array();
    if view.shape() != [3, 3] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (3, 3)"
        )));
    }

    let matrix = [
        [view[[0, 0]], view[[0, 1]], view[[0, 2]]],
        [view[[1, 0]], view[[1, 1]], view[[1, 2]]],
        [view[[2, 0]], view[[2, 1]], view[[2, 2]]],
    ];
    if finite.requires() && matrix.iter().flatten().any(|value| !value.is_finite()) {
        return Err(PyValueError::new_err(format!(
            "{name} must contain only finite values"
        )));
    }
    Ok(matrix)
}

/// Stable hash of a value's canonical `Debug` form, so a value object that
/// defines structural `__eq__` can also define `__hash__` (equal values hash
/// equal) and stay usable as a set member or dict key.
pub(crate) fn hash_debug<T: std::fmt::Debug>(value: &T) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{value:?}").hash(&mut hasher);
    hasher.finish()
}

/// Read a `(6, 6)` numpy array into a row-major `6x6` matrix, the layout the
/// core's [`Covariance6`] uses for a state `[r_x, r_y, r_z, v_x, v_y, v_z]`.
pub(crate) fn matrix6_from_array(
    values: &PyReadonlyArray2<'_, f64>,
    name: &str,
    finite: FinitePolicy,
) -> PyResult<[[f64; 6]; 6]> {
    let view = values.as_array();
    if view.shape() != [6, 6] {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (6, 6)"
        )));
    }

    let mut matrix = [[0.0_f64; 6]; 6];
    for (row_index, row) in matrix.iter_mut().enumerate() {
        for (col_index, cell) in row.iter_mut().enumerate() {
            let value = view[[row_index, col_index]];
            if finite.requires() && !value.is_finite() {
                return Err(PyValueError::new_err(format!(
                    "{name} must contain only finite values"
                )));
            }
            *cell = value;
        }
    }
    Ok(matrix)
}

/// Validate a `(6, 6)` numpy array into a typed core [`Covariance6`], mapping the
/// core's symmetry / positive-semidefinite rejection to `ValueError`.
pub(crate) fn covariance6_from_array(
    values: &PyReadonlyArray2<'_, f64>,
    name: &str,
) -> PyResult<Covariance6> {
    let matrix = matrix6_from_array(values, name, FinitePolicy::RequireFinite)?;
    Covariance6::try_from_matrix(matrix).map_err(|err| {
        let reason = match err {
            Covariance6Error::NonFinite => "contains a non-finite entry",
            Covariance6Error::Asymmetric => "is not symmetric",
            Covariance6Error::NotPositiveSemidefinite => "is not positive semidefinite",
        };
        PyValueError::new_err(format!("{name} {reason}"))
    })
}

/// Pack a typed core [`Covariance6`] into a `(6, 6)` numpy `float64` array.
pub(crate) fn covariance6_to_array<'py>(
    py: Python<'py>,
    covariance: &Covariance6,
) -> Bound<'py, PyArray2<f64>> {
    rows_to_array(py, covariance.as_matrix())
}

pub(crate) fn unix_microseconds_slice<'a>(
    epochs_unix_us: &'a PyReadonlyArray1<'_, i64>,
    empty: EmptyPolicy,
) -> PyResult<&'a [i64]> {
    let micros = epochs_unix_us
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    if empty.rejects() && micros.is_empty() {
        return Err(PyValueError::new_err("epochs array is empty"));
    }
    Ok(micros)
}

pub(crate) fn time_scales_from_unix_micros(
    epochs_unix_us: &PyReadonlyArray1<'_, i64>,
    empty: EmptyPolicy,
) -> PyResult<Vec<TimeScales>> {
    let micros = unix_microseconds_slice(epochs_unix_us, empty)?;
    Ok(micros
        .iter()
        .map(|&us| UtcInstant::from_unix_microseconds(us).time_scales())
        .collect())
}

pub(crate) fn instants_from_unix_micros(
    epochs_unix_us: &PyReadonlyArray1<'_, i64>,
    empty: EmptyPolicy,
) -> PyResult<Vec<UtcInstant>> {
    let micros = unix_microseconds_slice(epochs_unix_us, empty)?;
    Ok(micros
        .iter()
        .map(|&us| UtcInstant::from_unix_microseconds(us))
        .collect())
}

pub(crate) fn rows_to_array<'py, const N: usize>(
    py: Python<'py>,
    rows: &[[f64; N]],
) -> Bound<'py, PyArray2<f64>> {
    let mut array = Array2::<f64>::zeros((rows.len(), N));
    for (row_index, row) in rows.iter().enumerate() {
        for (col_index, value) in row.iter().enumerate() {
            array[[row_index, col_index]] = *value;
        }
    }
    PyArray2::from_owned_array(py, array)
}

/// Pack per-row `[x, y, z]` triples into an `(n, 3)` numpy `float64` array.
pub(crate) fn rows3_to_array<'py>(py: Python<'py>, rows: &[[f64; 3]]) -> Bound<'py, PyArray2<f64>> {
    rows_to_array(py, rows)
}

/// Pack per-row six-state vectors into an `(n, 6)` numpy `float64` array.
pub(crate) fn rows6_to_array<'py>(py: Python<'py>, rows: &[[f64; 6]]) -> Bound<'py, PyArray2<f64>> {
    rows_to_array(py, rows)
}

pub(crate) fn mat3_to_array<'py>(
    py: Python<'py>,
    matrix: &[[f64; 3]; 3],
) -> Bound<'py, PyArray2<f64>> {
    rows_to_array(py, matrix)
}

pub(crate) fn vec3_rows_to_array3<'py>(
    py: Python<'py>,
    rows: &[Vec<[f64; 3]>],
    n_epochs: usize,
) -> Bound<'py, PyArray3<f64>> {
    let n_sats = rows.len();
    debug_assert!(rows.iter().all(|arc| arc.len() == n_epochs));
    let mut array = Array3::<f64>::zeros((n_sats, n_epochs, 3));
    for (sat_idx, arc) in rows.iter().enumerate() {
        for (epoch_idx, value) in arc.iter().enumerate() {
            array[[sat_idx, epoch_idx, 0]] = value[0];
            array[[sat_idx, epoch_idx, 1]] = value[1];
            array[[sat_idx, epoch_idx, 2]] = value[2];
        }
    }
    array.into_pyarray(py)
}

pub(crate) fn scalar_rows_to_array2<'py>(
    py: Python<'py>,
    rows: &[Vec<f64>],
    n_epochs: usize,
) -> Bound<'py, PyArray2<f64>> {
    let n_sats = rows.len();
    debug_assert!(rows.iter().all(|arc| arc.len() == n_epochs));
    let mut array = Array2::<f64>::zeros((n_sats, n_epochs));
    for (sat_idx, arc) in rows.iter().enumerate() {
        for (epoch_idx, value) in arc.iter().enumerate() {
            array[[sat_idx, epoch_idx]] = *value;
        }
    }
    array.into_pyarray(py)
}
