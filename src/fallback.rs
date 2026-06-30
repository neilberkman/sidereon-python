//! Broadcast-ephemeris SPP and precise-with-broadcast fallback binding (#2).
//!
//! Marshals the SPP input bundle into [`sidereon_core::positioning`]'s
//! broadcast-only solve and the unified precise-with-broadcast fallback, and
//! surfaces which source produced the fix and how stale it is. No modeling lives
//! here: the solves are the engine's, and the precise-present path is bit-for-bit
//! the same as solving the SP3 directly.
//!
//! This layer is pure and no-network: it solves in memory on products the caller
//! has already parsed. Collecting the navigation message or fetching SP3 products
//! is the binding's separate `data` surface.

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::ephemeris::Sp3;
use sidereon_core::positioning::{
    solve_broadcast as core_solve_broadcast, solve_with_fallback as core_solve_with_fallback,
    BroadcastReason, FixSource, SourcedSolution,
};

use crate::ephemeris::PySp3;
use crate::rinex::PyBroadcastEphemeris;
use crate::spp::{PySppConfig, PySppSolution};
use crate::staleness::{PyStalenessMetadata, PyStalenessPolicy};
use crate::{to_solve_err, FallbackError as PyFallbackError};

/// Which ephemeris source produced a [`SourcedSolution`].
///
/// Always present on the result; a fallback solve never substitutes a source
/// silently.
#[pyclass(module = "sidereon._sidereon", name = "FixSource", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum PyFixSource {
    /// A precise SP3 product produced the fix (exact or degraded; see
    /// `staleness`).
    PRECISE,
    /// The broadcast ephemeris path produced the fix (see `broadcast_reason`).
    BROADCAST,
}

#[pymethods]
impl PyFixSource {
    /// Stable lowercase selector for this source.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::PRECISE => "precise",
            Self::BROADCAST => "broadcast",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::PRECISE => "FixSource.PRECISE",
            Self::BROADCAST => "FixSource.BROADCAST",
        }
    }
}

/// Why [`solve_with_fallback`] produced a fix from broadcast ephemeris.
///
/// A broadcast fix is never substituted silently: this records whether the
/// precise selection was declined outright, or a stale-but-within-cap precise
/// product was selected and then turned out unusable for the requested epoch.
#[pyclass(module = "sidereon._sidereon", name = "BroadcastReason", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyBroadcastReason {
    /// The precise product staleness selection declined (no product set, none
    /// covering or preceding the epoch, or the nearest beyond the staleness cap).
    /// The selection reason is on `SourcedSolution.selection_error`.
    PRECISE_UNAVAILABLE,
    /// A stale (within-cap) precise product was selected but could not produce a
    /// fix for the requested epoch. The tried product's staleness is on
    /// `SourcedSolution.attempted_staleness`.
    PRECISE_DEGRADED_UNUSABLE,
}

#[pymethods]
impl PyBroadcastReason {
    /// Stable lowercase selector for this reason.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::PRECISE_UNAVAILABLE => "precise_unavailable",
            Self::PRECISE_DEGRADED_UNUSABLE => "precise_degraded_unusable",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::PRECISE_UNAVAILABLE => "BroadcastReason.PRECISE_UNAVAILABLE",
            Self::PRECISE_DEGRADED_UNUSABLE => "BroadcastReason.PRECISE_DEGRADED_UNUSABLE",
        }
    }
}

/// A receiver solution paired with the provenance of the ephemeris that produced
/// it.
///
/// Returned by [`solve_with_fallback`]. `solution` is the receiver fix; `source`
/// names which ephemeris produced it; `staleness` carries the precise product's
/// staleness for a precise fix (`None` for a broadcast fix); and for a broadcast
/// fix `broadcast_reason` / `attempted_staleness` / `selection_error` record why
/// precise was not used.
#[pyclass(module = "sidereon._sidereon", name = "SourcedSolution")]
pub struct PySourcedSolution {
    inner: SourcedSolution,
}

#[pymethods]
impl PySourcedSolution {
    /// The solved receiver position/clock with its geometry diagnostics.
    #[getter]
    fn solution(&self) -> PySppSolution {
        PySppSolution::from_solution(self.inner.solution.clone())
    }

    /// Which ephemeris source produced the fix.
    #[getter]
    fn source(&self) -> PyFixSource {
        match self.inner.source {
            FixSource::Precise(_) => PyFixSource::PRECISE,
            FixSource::Broadcast(_) => PyFixSource::BROADCAST,
        }
    }

    /// Whether a precise SP3 product produced the fix (exact or degraded).
    #[getter]
    fn is_precise(&self) -> bool {
        self.inner.source.is_precise()
    }

    /// Whether the broadcast path produced the fix.
    #[getter]
    fn is_broadcast(&self) -> bool {
        self.inner.source.is_broadcast()
    }

    /// Whether a precise product covering the exact epoch produced the fix (no
    /// degradation, zero staleness).
    #[getter]
    fn is_precise_exact(&self) -> bool {
        self.inner.source.is_precise_exact()
    }

    /// The staleness of the precise product that produced the fix, or `None` for a
    /// broadcast fix (which is not backed by a precise product). For the
    /// degraded-then-fell-back case use `attempted_staleness`.
    #[getter]
    fn staleness(&self) -> Option<PyStalenessMetadata> {
        self.inner.source.staleness().map(PyStalenessMetadata::from)
    }

    /// For a broadcast fix, why precise was not used; `None` for a precise fix.
    #[getter]
    fn broadcast_reason(&self) -> Option<PyBroadcastReason> {
        match &self.inner.source {
            FixSource::Broadcast(BroadcastReason::PreciseUnavailable(_)) => {
                Some(PyBroadcastReason::PRECISE_UNAVAILABLE)
            }
            FixSource::Broadcast(BroadcastReason::PreciseDegradedUnusable { .. }) => {
                Some(PyBroadcastReason::PRECISE_DEGRADED_UNUSABLE)
            }
            FixSource::Precise(_) => None,
        }
    }

    /// For the degraded-then-fell-back broadcast case, the staleness of the
    /// precise product that was tried (and could not serve the epoch); `None`
    /// otherwise.
    #[getter]
    fn attempted_staleness(&self) -> Option<PyStalenessMetadata> {
        match &self.inner.source {
            FixSource::Broadcast(reason) => {
                reason.attempted_staleness().map(PyStalenessMetadata::from)
            }
            FixSource::Precise(_) => None,
        }
    }

    /// For a broadcast fix where the precise selection was declined outright, the
    /// selection layer's typed reason as a message; `None` otherwise.
    #[getter]
    fn selection_error(&self) -> Option<String> {
        match &self.inner.source {
            FixSource::Broadcast(BroadcastReason::PreciseUnavailable(error)) => {
                Some(error.to_string())
            }
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "SourcedSolution(source={}, is_precise_exact={})",
            self.source().__repr__(),
            self.inner.source.is_precise_exact(),
        )
    }
}

/// Solve a receiver position from broadcast ephemeris alone: the supported
/// real-time / offline single-point-positioning mode.
///
/// `config` is the same [`SppConfig`](crate::spp) bundle that `solve_spp` takes;
/// `with_geodetic` is taken from it. Bit-for-bit identical to feeding the
/// broadcast store to the generic solve. Raises `SolveError` on a solve failure.
#[pyfunction]
#[pyo3(signature = (broadcast, config))]
fn solve_broadcast(
    broadcast: &PyBroadcastEphemeris,
    config: &PySppConfig,
) -> PyResult<PySppSolution> {
    let inputs = config.to_inputs();
    let inner = core_solve_broadcast(&broadcast.inner, &inputs, config.with_geodetic_flag())
        .map_err(to_solve_err)?;
    Ok(PySppSolution::from_solution(inner))
}

/// Solve a receiver position, preferring precise products and falling back to
/// broadcast ephemeris, reporting which source was used and how stale it is.
///
/// `precise` is the set of parsed SP3 products to try first through the
/// product-staleness selection layer; `broadcast` is the fallback source;
/// `config` is the SPP input bundle; `policy` bounds how stale a precise product
/// may be before broadcast is preferred (default: a three-day cap). Returns a
/// [`SourcedSolution`] whose `source` and metadata are never silently dropped.
/// Raises `FallbackError` when the selected precise product solve fails (a
/// genuine error, not masked by a silent broadcast re-solve) or the broadcast
/// fallback solve fails.
#[pyfunction]
#[pyo3(signature = (precise, broadcast, config, policy=None))]
fn solve_with_fallback(
    py: Python<'_>,
    precise: Vec<PyRef<'_, PySp3>>,
    broadcast: &PyBroadcastEphemeris,
    config: &PySppConfig,
    policy: Option<Py<PyStalenessPolicy>>,
) -> PyResult<PySourcedSolution> {
    let owned: Vec<Sp3> = precise.iter().map(|p| p.inner.clone()).collect();
    let inputs = config.to_inputs();
    let resolved = match policy {
        Some(policy) => policy.borrow(py).inner,
        None => sidereon_core::staleness::StalenessPolicy::default(),
    };
    let inner = core_solve_with_fallback(
        &owned,
        &broadcast.inner,
        &inputs,
        resolved,
        config.with_geodetic_flag(),
    )
    .map_err(|err| PyFallbackError::new_err(err.to_string()))?;
    Ok(PySourcedSolution { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFixSource>()?;
    m.add_class::<PyBroadcastReason>()?;
    m.add_class::<PySourcedSolution>()?;
    m.add_function(wrap_pyfunction!(solve_broadcast, m)?)?;
    m.add_function(wrap_pyfunction!(solve_with_fallback, m)?)?;
    Ok(())
}
