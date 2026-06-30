//! Product-staleness selection binding (#1).
//!
//! Marshals Python product lists into [`sidereon_core::staleness`]'s graceful
//! degradation layer and returns the selected product paired with its staleness
//! provenance. No modeling lives here: the selection, the diurnal shift, and the
//! delegated slant-delay / interpolation are exactly what `sidereon-core`
//! produces. A degraded result is never returned without its
//! [`StalenessMetadata`], and a request that cannot be satisfied raises the typed
//! [`SelectionError`](crate::SelectionError) carrying the core's reason.
//!
//! This layer is pure and no-network: it selects among products the caller has
//! already parsed. Fetching the products is the binding's separate `data` surface.

use std::f64::consts::PI;

use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::atmosphere::Ionex;
use sidereon_core::ephemeris::Sp3;
use sidereon_core::staleness::{
    DegradationKind, SelectionError, StalenessMetadata, StalenessPolicy,
};
use sidereon_core::{GnssSatelliteId, Wgs84Geodetic};

use crate::ephemeris::{PySp3, PySp3State};
use crate::ionex::PyIonex;
use crate::marshal::option_py_or_default;
use crate::{to_solve_err, SelectionError as PySelectionError};

/// Degrees to radians as a single rounded constant `pi/180`, matching the
/// `Ionex.slant_delay` boundary so a selected product's delay is bit-identical to
/// querying the same product directly.
const DEG_TO_RAD: f64 = PI / 180.0;

/// Map a core [`SelectionError`] into the typed Python
/// [`SelectionError`](crate::SelectionError), preserving the engine message.
pub(crate) fn to_selection_err(err: SelectionError) -> PyErr {
    PySelectionError::new_err(err.to_string())
}

/// How a selected product's source epoch relates to the requested epoch.
#[pyclass(module = "sidereon._sidereon", name = "DegradationKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum PyDegradationKind {
    /// A product covering the requested epoch was present; no degradation.
    EXACT,
    /// No product covered the requested epoch; the most-recent prior product was
    /// used as-is (SP3 path).
    NEAREST_PRIOR,
    /// No product covered the requested day; a prior day's IONEX grid was
    /// advanced by whole days onto the requested epoch (diurnal persistence).
    DIURNAL_SHIFT,
}

impl From<DegradationKind> for PyDegradationKind {
    fn from(kind: DegradationKind) -> Self {
        match kind {
            DegradationKind::Exact => Self::EXACT,
            DegradationKind::NearestPrior => Self::NEAREST_PRIOR,
            DegradationKind::DiurnalShift => Self::DIURNAL_SHIFT,
        }
    }
}

#[pymethods]
impl PyDegradationKind {
    /// Stable lowercase selector for this degradation path.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::EXACT => "exact",
            Self::NEAREST_PRIOR => "nearest_prior",
            Self::DIURNAL_SHIFT => "diurnal_shift",
        }
    }

    /// Whether this result used the exact present product (no degradation).
    #[getter]
    fn is_exact(&self) -> bool {
        matches!(self, Self::EXACT)
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::EXACT => "DegradationKind.EXACT",
            Self::NEAREST_PRIOR => "DegradationKind.NEAREST_PRIOR",
            Self::DIURNAL_SHIFT => "DegradationKind.DIURNAL_SHIFT",
        }
    }
}

/// Structured description of the product staleness behind a selection result.
///
/// Attached to every selection; a degraded result is never produced without it.
/// Epoch fields are seconds since the J2000 epoch (2000-01-01 12:00:00).
/// `staleness_s` is `requested - source` and is never negative.
#[pyclass(module = "sidereon._sidereon", name = "StalenessMetadata")]
#[derive(Clone, Copy)]
pub struct PyStalenessMetadata {
    pub(crate) inner: StalenessMetadata,
}

impl From<StalenessMetadata> for PyStalenessMetadata {
    fn from(inner: StalenessMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyStalenessMetadata {
    /// Which degradation path produced the result.
    #[getter]
    fn kind(&self) -> PyDegradationKind {
        self.inner.kind.into()
    }

    /// The requested epoch, J2000 seconds. For a range request this is the latest
    /// (most-stale) epoch of the range.
    #[getter]
    fn requested_epoch_j2000_s(&self) -> f64 {
        self.inner.requested_epoch_j2000_s
    }

    /// The source product epoch the result is backed by, J2000 seconds.
    #[getter]
    fn source_epoch_j2000_s(&self) -> f64 {
        self.inner.source_epoch_j2000_s
    }

    /// Staleness `requested - source`, seconds. Zero for an exact result; never
    /// negative.
    #[getter]
    fn staleness_s(&self) -> f64 {
        self.inner.staleness_s
    }

    /// Staleness in days (`staleness_s / 86400`). For a diurnal shift this is the
    /// integer day offset applied.
    #[getter]
    fn staleness_days(&self) -> f64 {
        self.inner.staleness_days
    }

    fn __repr__(&self) -> String {
        format!(
            "StalenessMetadata(kind={}, requested_epoch_j2000_s={}, \
             source_epoch_j2000_s={}, staleness_s={}, staleness_days={})",
            PyDegradationKind::from(self.inner.kind).__repr__(),
            self.inner.requested_epoch_j2000_s,
            self.inner.source_epoch_j2000_s,
            self.inner.staleness_s,
            self.inner.staleness_days,
        )
    }
}

/// Configurable staleness cap for product selection.
///
/// A selection that would rely on a product older than `max_staleness_s` raises
/// [`SelectionError`](crate::SelectionError) rather than returning data past the
/// cap. The default cap is three days.
#[pyclass(module = "sidereon._sidereon", name = "StalenessPolicy")]
#[derive(Clone, Copy)]
pub struct PyStalenessPolicy {
    pub(crate) inner: StalenessPolicy,
}

#[pymethods]
impl PyStalenessPolicy {
    /// Create a policy from a cap in seconds.
    #[new]
    fn new(max_staleness_s: f64) -> Self {
        Self {
            inner: StalenessPolicy { max_staleness_s },
        }
    }

    /// A policy with a cap expressed in days.
    #[staticmethod]
    fn days(days: f64) -> Self {
        Self {
            inner: StalenessPolicy::days(days),
        }
    }

    /// A policy with a cap expressed in seconds.
    #[staticmethod]
    fn seconds(seconds: f64) -> Self {
        Self {
            inner: StalenessPolicy::seconds(seconds),
        }
    }

    /// The default policy (a three-day cap).
    #[staticmethod]
    fn default_policy() -> Self {
        Self {
            inner: StalenessPolicy::default(),
        }
    }

    /// Maximum tolerated staleness, seconds.
    #[getter]
    fn max_staleness_s(&self) -> f64 {
        self.inner.max_staleness_s
    }

    fn __repr__(&self) -> String {
        format!(
            "StalenessPolicy(max_staleness_s={})",
            self.inner.max_staleness_s
        )
    }
}

impl PyStalenessPolicy {
    /// The core policy, defaulting to a three-day cap when the caller passed none.
    fn resolve(py: Python<'_>, policy: Option<&Py<PyStalenessPolicy>>) -> StalenessPolicy {
        option_py_or_default(py, policy, |value| value.inner, StalenessPolicy::default)
    }
}

/// A selected IONEX product plus its staleness metadata.
///
/// Returned by [`select_ionex`] / [`select_ionex_over_range`]. `ionex` is the
/// usable product (the present product for an exact result, or the
/// diurnal-shifted copy for a degraded one) and `slant_delay` runs the standard
/// query on it; for an exact selection both are bit-identical to querying the
/// caller's product directly.
#[pyclass(module = "sidereon._sidereon", name = "IonexSelection")]
pub struct PyIonexSelection {
    ionex: Ionex,
    metadata: StalenessMetadata,
}

#[pymethods]
impl PyIonexSelection {
    /// The staleness metadata for this selection.
    #[getter]
    fn metadata(&self) -> PyStalenessMetadata {
        self.metadata.into()
    }

    /// The usable IONEX product: the present product for an exact result, or the
    /// diurnal-shifted copy for a degraded one.
    #[getter]
    fn ionex(&self) -> PyIonex {
        PyIonex::from_ionex(self.ionex.clone())
    }

    /// IONEX slant ionospheric group delay (positive metres) from the selected
    /// product. Degrees in, metres out; mirrors `Ionex.slant_delay` exactly.
    #[pyo3(signature = (lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch_j2000_s, frequency_hz))]
    fn slant_delay(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch_j2000_s: i64,
        frequency_hz: f64,
    ) -> PyResult<f64> {
        let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, 0.0)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        sidereon_core::atmosphere::ionex_slant_delay(
            &self.ionex,
            receiver,
            elevation_deg * DEG_TO_RAD,
            azimuth_deg * DEG_TO_RAD,
            epoch_j2000_s,
            frequency_hz,
        )
        .map_err(|err| match err {
            sidereon_core::Error::InvalidInput(message) => pyo3::exceptions::PyValueError::new_err(
                format!("invalid IONEX slant input: {message}"),
            ),
            other => to_solve_err(other.to_string()),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexSelection(kind={}, staleness_s={})",
            PyDegradationKind::from(self.metadata.kind).__repr__(),
            self.metadata.staleness_s,
        )
    }
}

/// A selected SP3 product plus its staleness metadata.
///
/// Returned by [`select_sp3`] / [`select_sp3_over_range`]. `sp3` is the selected
/// product (the full query surface) and `position_at_j2000_seconds` interpolates
/// it; for an exact selection both are bit-identical to the caller's product.
#[pyclass(module = "sidereon._sidereon", name = "Sp3Selection")]
pub struct PySp3Selection {
    sp3: Sp3,
    metadata: StalenessMetadata,
}

#[pymethods]
impl PySp3Selection {
    /// The staleness metadata for this selection.
    #[getter]
    fn metadata(&self) -> PyStalenessMetadata {
        self.metadata.into()
    }

    /// The selected SP3 product, with its full interpolation/query surface.
    #[getter]
    fn sp3(&self) -> PySp3 {
        PySp3::from_sp3(self.sp3.clone())
    }

    /// Interpolate `satellite` at a J2000-second epoch on the selected product.
    ///
    /// Delegates to the engine's `position_at_j2000_seconds`, so an exact
    /// selection is bit-identical to interpolating the caller's product. Raises
    /// `ValueError` for an unknown satellite token and `SolveError` for a query
    /// outside the product's coverage.
    fn position_at_j2000_seconds(
        &self,
        satellite: &str,
        query_j2000_s: f64,
    ) -> PyResult<PySp3State> {
        let sat = satellite.parse::<GnssSatelliteId>().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "invalid satellite token {satellite:?}: {e}"
            ))
        })?;
        let state = self
            .sp3
            .position_at_j2000_seconds(sat, query_j2000_s)
            .map_err(|e| match e {
                sidereon_core::Error::UnknownSatellite(id) => {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "satellite {id} is not in the product"
                    ))
                }
                other => to_solve_err(format!(
                    "interpolation at j2000 second {query_j2000_s}: {other}"
                )),
            })?;
        Ok(PySp3State::from_state(state))
    }

    fn __repr__(&self) -> String {
        format!(
            "Sp3Selection(kind={}, staleness_s={})",
            PyDegradationKind::from(self.metadata.kind).__repr__(),
            self.metadata.staleness_s,
        )
    }
}

/// Select an IONEX product usable at `requested_epoch_j2000_s`, degrading to a
/// diurnal-shifted prior product within `policy` when the exact day is absent.
#[pyfunction]
#[pyo3(signature = (products, requested_epoch_j2000_s, policy=None))]
fn select_ionex(
    py: Python<'_>,
    products: Vec<PyRef<'_, PyIonex>>,
    requested_epoch_j2000_s: i64,
    policy: Option<Py<PyStalenessPolicy>>,
) -> PyResult<PyIonexSelection> {
    select_ionex_over_range(
        py,
        products,
        requested_epoch_j2000_s,
        requested_epoch_j2000_s,
        policy,
    )
}

/// Select an IONEX product usable across `[start, end]` (J2000 seconds).
#[pyfunction]
#[pyo3(name = "select_ionex_over_range", signature = (products, start_epoch_j2000_s, end_epoch_j2000_s, policy=None))]
fn select_ionex_over_range(
    py: Python<'_>,
    products: Vec<PyRef<'_, PyIonex>>,
    start_epoch_j2000_s: i64,
    end_epoch_j2000_s: i64,
    policy: Option<Py<PyStalenessPolicy>>,
) -> PyResult<PyIonexSelection> {
    let owned: Vec<Ionex> = products.iter().map(|p| p.inner.clone()).collect();
    let resolved = PyStalenessPolicy::resolve(py, policy.as_ref());
    let selection = sidereon_core::staleness::select_ionex_over_range(
        &owned,
        start_epoch_j2000_s,
        end_epoch_j2000_s,
        resolved,
    )
    .map_err(to_selection_err)?;
    Ok(PyIonexSelection {
        ionex: selection.ionex().clone(),
        metadata: selection.metadata(),
    })
}

/// Select an SP3 product usable at `requested_epoch_j2000_s`, degrading to the
/// most-recent prior product within `policy`.
#[pyfunction]
#[pyo3(signature = (products, requested_epoch_j2000_s, policy=None))]
fn select_sp3(
    py: Python<'_>,
    products: Vec<PyRef<'_, PySp3>>,
    requested_epoch_j2000_s: f64,
    policy: Option<Py<PyStalenessPolicy>>,
) -> PyResult<PySp3Selection> {
    select_sp3_over_range(
        py,
        products,
        requested_epoch_j2000_s,
        requested_epoch_j2000_s,
        policy,
    )
}

/// Select an SP3 product usable across `[start, end]` (J2000 seconds).
#[pyfunction]
#[pyo3(name = "select_sp3_over_range", signature = (products, start_epoch_j2000_s, end_epoch_j2000_s, policy=None))]
fn select_sp3_over_range(
    py: Python<'_>,
    products: Vec<PyRef<'_, PySp3>>,
    start_epoch_j2000_s: f64,
    end_epoch_j2000_s: f64,
    policy: Option<Py<PyStalenessPolicy>>,
) -> PyResult<PySp3Selection> {
    let owned: Vec<Sp3> = products.iter().map(|p| p.inner.clone()).collect();
    let resolved = PyStalenessPolicy::resolve(py, policy.as_ref());
    let selection = sidereon_core::staleness::select_sp3_over_range(
        &owned,
        start_epoch_j2000_s,
        end_epoch_j2000_s,
        resolved,
    )
    .map_err(to_selection_err)?;
    Ok(PySp3Selection {
        sp3: selection.sp3().clone(),
        metadata: selection.metadata(),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDegradationKind>()?;
    m.add_class::<PyStalenessMetadata>()?;
    m.add_class::<PyStalenessPolicy>()?;
    m.add_class::<PyIonexSelection>()?;
    m.add_class::<PySp3Selection>()?;
    m.add_function(wrap_pyfunction!(select_ionex, m)?)?;
    m.add_function(wrap_pyfunction!(select_ionex_over_range, m)?)?;
    m.add_function(wrap_pyfunction!(select_sp3, m)?)?;
    m.add_function(wrap_pyfunction!(select_sp3_over_range, m)?)?;
    Ok(())
}
