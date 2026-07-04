//! Geometry observability diagnostics shared by positioning solution wrappers.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::geometry_quality::{GeometryQuality as CoreGeometryQuality, ObservabilityTier};

/// Observability and covariance-validation tier for an estimated geometry.
#[pyclass(module = "sidereon._sidereon", name = "ObservabilityTier", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyObservabilityTier {
    /// At least one solved parameter was not observable.
    RANK_DEFICIENT,
    /// The design was full rank but had no residual degrees of freedom.
    ZERO_REDUNDANCY,
    /// The design had residual degrees of freedom but exceeded a cutoff.
    WEAK,
    /// The design was full rank and within the configured cutoffs.
    NOMINAL,
}

impl From<ObservabilityTier> for PyObservabilityTier {
    fn from(tier: ObservabilityTier) -> Self {
        match tier {
            ObservabilityTier::RankDeficient => Self::RANK_DEFICIENT,
            ObservabilityTier::ZeroRedundancy => Self::ZERO_REDUNDANCY,
            ObservabilityTier::Weak => Self::WEAK,
            ObservabilityTier::Nominal => Self::NOMINAL,
        }
    }
}

#[pymethods]
impl PyObservabilityTier {
    /// Stable lowercase label for this tier.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::RANK_DEFICIENT => "rank_deficient",
            Self::ZERO_REDUNDANCY => "zero_redundancy",
            Self::WEAK => "weak",
            Self::NOMINAL => "nominal",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::RANK_DEFICIENT => "ObservabilityTier.RANK_DEFICIENT",
            Self::ZERO_REDUNDANCY => "ObservabilityTier.ZERO_REDUNDANCY",
            Self::WEAK => "ObservabilityTier.WEAK",
            Self::NOMINAL => "ObservabilityTier.NOMINAL",
        }
    }
}

/// Geometry observability and covariance-validation diagnostics.
///
/// `ZeroRedundancy` and `Weak` report the core solver's raw validation tier:
/// zero-redundancy bounds are unvalidated unless a propagated prior validates
/// them, and weak-geometry bounds are reported without clamping.
#[pyclass(module = "sidereon._sidereon", name = "GeometryQuality")]
#[derive(Clone, Copy)]
pub struct PyGeometryQuality {
    inner: CoreGeometryQuality,
}

impl From<CoreGeometryQuality> for PyGeometryQuality {
    fn from(inner: CoreGeometryQuality) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyGeometryQuality {
    /// Observability and validation tier.
    #[getter]
    fn tier(&self) -> PyObservabilityTier {
        self.inner.tier.into()
    }

    /// Observation redundancy, defined as `n_obs - n_params`.
    #[getter]
    fn redundancy(&self) -> i32 {
        self.inner.redundancy
    }

    /// Rank of the design matrix used by the solve.
    #[getter]
    fn rank(&self) -> usize {
        self.inner.rank
    }

    /// Singular-value condition number of the design matrix.
    #[getter]
    fn condition_number(&self) -> f64 {
        self.inner.condition_number
    }

    /// Geometric dilution of precision for the solved state.
    #[getter]
    fn gdop(&self) -> f64 {
        self.inner.gdop
    }

    /// Whether residual-based RAIM can test the solve.
    #[getter]
    fn raim_checkable(&self) -> bool {
        self.inner.raim_checkable
    }

    /// Whether residuals or a propagated prior validated the covariance bound.
    #[getter]
    fn covariance_validated(&self) -> bool {
        self.inner.covariance_validated
    }

    fn __repr__(&self) -> String {
        format!(
            "GeometryQuality(tier={}, redundancy={}, rank={}, gdop={})",
            self.tier().label(),
            self.inner.redundancy,
            self.inner.rank,
            self.inner.gdop
        )
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyObservabilityTier>()?;
    m.add_class::<PyGeometryQuality>()?;
    Ok(())
}
