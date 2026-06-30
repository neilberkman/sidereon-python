//! Lambert two-point boundary-value (orbit transfer) binding.
//!
//! Thin marshaling over [`sidereon_core::astro::lambert`]: numpy 3-vectors and
//! scalars in, the two transfer velocity vectors out. All solver math (Battin's
//! method) lives in the core engine; this layer only converts arrays and enum
//! codes and maps the typed error onto `SolveError`.

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::astro::lambert::{
    battin as core_battin, DirectionOfEnergy as CoreDirectionOfEnergy,
    DirectionOfMotion as CoreDirectionOfMotion,
};

use crate::marshal::{fixed_array, ArrayPairF64, FinitePolicy};
use crate::{np_array, to_solve_err};

/// Direction of motion for a Lambert transfer.
#[pyclass(module = "sidereon._sidereon", name = "DirectionOfMotion", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyDirectionOfMotion {
    /// Short-way transfer (transfer angle < 180 degrees).
    SHORT,
    /// Long-way transfer (transfer angle > 180 degrees).
    LONG,
}

impl PyDirectionOfMotion {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "short" => Ok(Self::SHORT),
            "long" => Ok(Self::LONG),
            other => Err(PyValueError::new_err(format!(
                "unknown direction of motion {other:?}; expected \"short\" or \"long\""
            ))),
        }
    }
}

impl From<PyDirectionOfMotion> for CoreDirectionOfMotion {
    fn from(value: PyDirectionOfMotion) -> Self {
        match value {
            PyDirectionOfMotion::SHORT => CoreDirectionOfMotion::Short,
            PyDirectionOfMotion::LONG => CoreDirectionOfMotion::Long,
        }
    }
}

#[pymethods]
impl PyDirectionOfMotion {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::SHORT => "short",
            Self::LONG => "long",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::SHORT => "DirectionOfMotion.SHORT",
            Self::LONG => "DirectionOfMotion.LONG",
        }
    }
}

/// Energy branch for a Lambert transfer.
#[pyclass(module = "sidereon._sidereon", name = "DirectionOfEnergy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyDirectionOfEnergy {
    /// Low-energy branch.
    LOW,
    /// High-energy branch.
    HIGH,
}

impl PyDirectionOfEnergy {
    fn from_label(value: &str) -> PyResult<Self> {
        match value {
            "low" => Ok(Self::LOW),
            "high" => Ok(Self::HIGH),
            other => Err(PyValueError::new_err(format!(
                "unknown direction of energy {other:?}; expected \"low\" or \"high\""
            ))),
        }
    }
}

impl From<PyDirectionOfEnergy> for CoreDirectionOfEnergy {
    fn from(value: PyDirectionOfEnergy) -> Self {
        match value {
            PyDirectionOfEnergy::LOW => CoreDirectionOfEnergy::Low,
            PyDirectionOfEnergy::HIGH => CoreDirectionOfEnergy::High,
        }
    }
}

#[pymethods]
impl PyDirectionOfEnergy {
    /// Stable lowercase selector accepted as a string alias.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::LOW => "low",
            Self::HIGH => "high",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::LOW => "DirectionOfEnergy.LOW",
            Self::HIGH => "DirectionOfEnergy.HIGH",
        }
    }
}

fn extract_motion(obj: Option<&Bound<'_, PyAny>>) -> PyResult<PyDirectionOfMotion> {
    let Some(obj) = obj else {
        return Ok(PyDirectionOfMotion::SHORT);
    };
    if let Ok(value) = obj.extract::<PyDirectionOfMotion>() {
        return Ok(value);
    }
    PyDirectionOfMotion::from_label(&obj.extract::<String>()?)
}

fn extract_energy(obj: Option<&Bound<'_, PyAny>>) -> PyResult<PyDirectionOfEnergy> {
    let Some(obj) = obj else {
        return Ok(PyDirectionOfEnergy::LOW);
    };
    if let Ok(value) = obj.extract::<PyDirectionOfEnergy>() {
        return Ok(value);
    }
    PyDirectionOfEnergy::from_label(&obj.extract::<String>()?)
}

/// Solve Lambert's problem with Battin's method.
///
/// Given two position vectors `r1`, `r2` (numpy `(3,)`, km) and a time of flight
/// `dtsec` (seconds), return the transfer velocity vectors `(v1_transfer,
/// v2_transfer)` at `r1` and `r2` (each numpy `(3,)`, km/s). `v1` (km/s) is only
/// consulted for the degenerate 180-degree transfer where the transfer-plane
/// normal is otherwise undefined. `direction_of_motion` is `"short"`/`"long"`
/// (or a `DirectionOfMotion`), `direction_of_energy` is `"low"`/`"high"` (or a
/// `DirectionOfEnergy`), and `nrev` is the number of complete revolutions.
/// Raises `ValueError` on a malformed shape and `SolveError` on a degenerate
/// geometry or non-convergence.
#[pyfunction]
#[pyo3(signature = (r1, r2, v1, dtsec, direction_of_motion=None, direction_of_energy=None, nrev=0))]
#[allow(clippy::too_many_arguments)]
fn lambert_battin<'py>(
    py: Python<'py>,
    r1: PyReadonlyArray1<'_, f64>,
    r2: PyReadonlyArray1<'_, f64>,
    v1: PyReadonlyArray1<'_, f64>,
    dtsec: f64,
    direction_of_motion: Option<&Bound<'_, PyAny>>,
    direction_of_energy: Option<&Bound<'_, PyAny>>,
    nrev: i32,
) -> PyResult<ArrayPairF64<'py>> {
    let r1 = fixed_array::<3>("r1", &r1, FinitePolicy::RequireFinite)?;
    let r2 = fixed_array::<3>("r2", &r2, FinitePolicy::RequireFinite)?;
    let v1 = fixed_array::<3>("v1", &v1, FinitePolicy::RequireFinite)?;
    let dm = extract_motion(direction_of_motion)?;
    let de = extract_energy(direction_of_energy)?;
    let (v1t, v2t) =
        core_battin(&r1, &r2, &v1, dm.into(), de.into(), nrev, dtsec).map_err(to_solve_err)?;
    Ok((np_array(py, &v1t), np_array(py, &v2t)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDirectionOfMotion>()?;
    m.add_class::<PyDirectionOfEnergy>()?;
    m.add_function(wrap_pyfunction!(lambert_battin, m)?)?;
    Ok(())
}
