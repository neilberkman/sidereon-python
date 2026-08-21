//! Thin PyO3 bridge to the shared exact-product cache implementation.

use pyo3::exceptions::{PyOSError, PyTimeoutError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use serde::Deserialize;
use sidereon_core::data::{
    DistributionSource, ProductCampaign, ProductDate, ProductFormat, ProductIdentity,
    ProductPublisher, ProductType, SolutionClass,
};
use sidereon_core::exact_cache::{
    ExactCacheError, ExactCacheGuard, ExactCacheOpen, ExactCacheOwner,
    ExactCacheSingleFlightOptions, ExactProductCache, EXACT_CACHE_CONTROL_DIRECTORY,
};
use std::time::Duration;

#[derive(Deserialize)]
struct IdentityInput {
    family: String,
    analysis_center: String,
    publisher: String,
    solution_class: String,
    campaign: String,
    filename_version: u8,
    date: String,
    issue: String,
    span: String,
    sample: String,
    official_filename: String,
    format: String,
    format_version: Option<String>,
    prediction_horizon_days: Option<u8>,
}

fn value_error(message: impl ToString) -> PyErr {
    PyValueError::new_err(message.to_string())
}

fn cache_error(error: ExactCacheError) -> PyErr {
    match error {
        ExactCacheError::LockTimeout | ExactCacheError::SingleFlightTimeout => {
            PyTimeoutError::new_err(error.to_string())
        }
        ExactCacheError::InvalidSingleFlightOptions => value_error(error),
        _ => PyOSError::new_err(error.to_string()),
    }
}

fn positive_duration(name: &str, seconds: f64) -> PyResult<Duration> {
    let duration = Duration::try_from_secs_f64(seconds)
        .map_err(|_| value_error(format!("{name} must be finite and positive")))?;
    if duration.is_zero() {
        return Err(value_error(format!("{name} must be finite and positive")));
    }
    Ok(duration)
}

pub(crate) fn source(value: &str) -> PyResult<DistributionSource> {
    match value {
        "direct" => Ok(DistributionSource::Direct),
        "nasa_cddis" => Ok(DistributionSource::NasaCddis),
        "local_file" => Ok(DistributionSource::LocalFile),
        "in_memory" => Ok(DistributionSource::InMemory),
        _ => Err(value_error("unknown distribution source")),
    }
}

pub(crate) fn identity(json: &str) -> PyResult<ProductIdentity> {
    let value: IdentityInput = serde_json::from_str(json).map_err(value_error)?;
    let mut date_parts = value.date.split('-');
    let year = date_parts
        .next()
        .ok_or_else(|| value_error("invalid identity date"))?
        .parse::<i32>()
        .map_err(value_error)?;
    let month = date_parts
        .next()
        .ok_or_else(|| value_error("invalid identity date"))?
        .parse::<u8>()
        .map_err(value_error)?;
    let day = date_parts
        .next()
        .ok_or_else(|| value_error("invalid identity date"))?
        .parse::<u8>()
        .map_err(value_error)?;
    if date_parts.next().is_some() {
        return Err(value_error("invalid identity date"));
    }
    let identity = ProductIdentity {
        family: ProductType::from_code(&value.family)
            .ok_or_else(|| value_error("unknown product family"))?,
        analysis_center: value.analysis_center.parse().map_err(value_error)?,
        publisher: match value.publisher.as_str() {
            "IGS" => ProductPublisher::Igs,
            "COD" => ProductPublisher::Code,
            "ESA" => ProductPublisher::Esa,
            "GFZ" => ProductPublisher::Gfz,
            "WUM" => ProductPublisher::Whu,
            _ => return Err(value_error("unknown product publisher")),
        },
        solution: match value.solution_class.as_str() {
            "final" => SolutionClass::Final,
            "rapid" => SolutionClass::Rapid,
            "ultra_rapid" => SolutionClass::UltraRapid,
            "predicted" => SolutionClass::Predicted,
            "broadcast" => SolutionClass::Broadcast,
            "near_real_time" => SolutionClass::NearRealTime,
            _ => return Err(value_error("unknown solution class")),
        },
        campaign: match value.campaign.as_str() {
            "OPS" => ProductCampaign::Operational,
            "MGN" => ProductCampaign::MultiGnss,
            "MGX" => ProductCampaign::MultiGnssExperiment,
            "BRD" => ProductCampaign::Broadcast,
            _ => return Err(value_error("unknown product campaign")),
        },
        version: value.filename_version,
        date: ProductDate::new(year, month, day).map_err(value_error)?,
        issue: if value.issue.is_empty() {
            None
        } else {
            Some(value.issue)
        },
        span: value.span,
        sample: value.sample,
        official_filename: value.official_filename,
        format: match value.format.as_str() {
            "SP3" => ProductFormat::Sp3,
            "IONEX" => ProductFormat::Ionex,
            "RINEX_CLK" => ProductFormat::RinexClock,
            "RINEX_NAV" => ProductFormat::RinexNavigation,
            _ => return Err(value_error("unknown product format")),
        },
        format_version: value.format_version,
        prediction_horizon_days: value.prediction_horizon_days,
    };
    identity.validate().map_err(value_error)?;
    Ok(identity)
}

/// Lock-owning native cache transaction used by the Python acquisition layer.
#[pyclass(name = "_ExactProductCache")]
struct PyExactProductCache {
    cache: ExactProductCache,
    guard: Option<ExactCacheGuard>,
}

/// Native single-flight ownership retained until publish or close.
#[pyclass(name = "_ExactCacheOwner")]
struct PyExactCacheOwner {
    owner: Option<ExactCacheOwner>,
}

type CacheRead<'py> = (
    String,
    String,
    String,
    String,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
);

fn entry_to_python<'py>(
    py: Python<'py>,
    entry: sidereon_core::exact_cache::CommittedExactCacheEntry,
) -> CacheRead<'py> {
    (
        entry.product_path.to_string_lossy().into_owned(),
        entry.archive_path.to_string_lossy().into_owned(),
        entry.provenance_path.to_string_lossy().into_owned(),
        entry.entry_id,
        PyBytes::new(py, &entry.product),
        PyBytes::new(py, &entry.archive),
        PyBytes::new(py, &entry.provenance),
    )
}

#[pyfunction]
fn data_exact_cache_read<'py>(
    py: Python<'py>,
    stable_path: String,
    identity_json: &str,
    distribution_source: &str,
) -> PyResult<Option<CacheRead<'py>>> {
    ExactProductCache::new(
        stable_path,
        identity(identity_json)?,
        source(distribution_source)?,
    )
    .map_err(cache_error)?
    .read()
    .map(|entry| entry.map(|entry| entry_to_python(py, entry)))
    .map_err(cache_error)
}

#[pyfunction]
fn data_exact_cache_open_single_flight<'py>(
    py: Python<'py>,
    stable_path: String,
    identity_json: &str,
    distribution_source: &str,
    timing_s: (f64, f64, f64, f64),
) -> PyResult<(Option<CacheRead<'py>>, Option<Py<PyExactCacheOwner>>)> {
    let (poll_interval_s, heartbeat_interval_s, liveness_timeout_s, wait_timeout_s) = timing_s;
    let options = ExactCacheSingleFlightOptions {
        poll_interval: positive_duration("poll_interval_s", poll_interval_s)?,
        heartbeat_interval: positive_duration("heartbeat_interval_s", heartbeat_interval_s)?,
        liveness_timeout: positive_duration("liveness_timeout_s", liveness_timeout_s)?,
        wait_timeout: positive_duration("wait_timeout_s", wait_timeout_s)?,
    };
    let cache = ExactProductCache::new(
        stable_path,
        identity(identity_json)?,
        source(distribution_source)?,
    )
    .map_err(cache_error)?;
    match py
        .allow_threads(|| cache.open_single_flight(options))
        .map_err(cache_error)?
    {
        ExactCacheOpen::Hit(entry) => Ok((Some(entry_to_python(py, entry)), None)),
        ExactCacheOpen::Owner(owner) => Ok((
            None,
            Some(Py::new(py, PyExactCacheOwner { owner: Some(owner) })?),
        )),
    }
}

#[pyfunction]
fn data_validate_product_identity(identity_json: &str) -> PyResult<()> {
    identity(identity_json).map(|_| ())
}

#[pymethods]
impl PyExactProductCache {
    #[new]
    fn new(
        py: Python<'_>,
        stable_path: String,
        identity_json: &str,
        distribution_source: &str,
        timeout_s: f64,
    ) -> PyResult<Self> {
        let timeout = Duration::try_from_secs_f64(timeout_s)
            .map_err(|_| value_error("cache lock timeout must be finite and non-negative"))?;
        let cache = ExactProductCache::new(
            stable_path,
            identity(identity_json)?,
            source(distribution_source)?,
        )
        .map_err(cache_error)?;
        let guard = py
            .allow_threads(|| cache.lock(timeout))
            .map_err(cache_error)?;
        Ok(Self {
            cache,
            guard: Some(guard),
        })
    }

    fn read<'py>(&self, py: Python<'py>) -> PyResult<Option<CacheRead<'py>>> {
        self.require_open()?;
        self.cache
            .read()
            .map(|entry| entry.map(|entry| entry_to_python(py, entry)))
            .map_err(cache_error)
    }

    fn publish<'py>(
        &self,
        py: Python<'py>,
        product: &[u8],
        archive: &[u8],
        provenance: &[u8],
    ) -> PyResult<CacheRead<'py>> {
        let guard = self.require_open()?;
        let entry = self
            .cache
            .publish(guard, product, archive, provenance)
            .map_err(cache_error)?;
        Ok(entry_to_python(py, entry))
    }

    fn cleanup_abandoned(&self) -> PyResult<()> {
        let guard = self.require_open()?;
        self.cache.cleanup_abandoned(guard).map_err(cache_error)
    }

    fn close(&mut self) {
        self.guard.take();
    }
}

#[pymethods]
impl PyExactCacheOwner {
    fn heartbeat(&self, py: Python<'_>) -> PyResult<()> {
        let owner = self.require_open()?;
        py.allow_threads(|| owner.heartbeat()).map_err(cache_error)
    }

    fn publish<'py>(
        &mut self,
        py: Python<'py>,
        product: &[u8],
        archive: &[u8],
        provenance: &[u8],
    ) -> PyResult<CacheRead<'py>> {
        let owner = self
            .owner
            .take()
            .ok_or_else(|| PyOSError::new_err("exact-cache single-flight owner is closed"))?;
        let entry = py
            .allow_threads(|| owner.publish(product, archive, provenance))
            .map_err(cache_error)?;
        Ok(entry_to_python(py, entry))
    }

    fn close(&mut self) {
        self.owner.take();
    }
}

impl PyExactProductCache {
    fn require_open(&self) -> PyResult<&ExactCacheGuard> {
        self.guard
            .as_ref()
            .ok_or_else(|| PyOSError::new_err("exact-product cache lock is closed"))
    }
}

impl PyExactCacheOwner {
    fn require_open(&self) -> PyResult<&ExactCacheOwner> {
        self.owner
            .as_ref()
            .ok_or_else(|| PyOSError::new_err("exact-cache single-flight owner is closed"))
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyExactProductCache>()?;
    m.add_class::<PyExactCacheOwner>()?;
    m.add_function(wrap_pyfunction!(data_exact_cache_read, m)?)?;
    m.add_function(wrap_pyfunction!(data_exact_cache_open_single_flight, m)?)?;
    m.add_function(wrap_pyfunction!(data_validate_product_identity, m)?)?;
    m.add(
        "_EXACT_CACHE_CONTROL_DIRECTORY",
        EXACT_CACHE_CONTROL_DIRECTORY,
    )?;
    Ok(())
}
