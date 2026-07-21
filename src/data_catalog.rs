//! Core-backed data-product catalog bridge for the Python fetch layer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use sidereon_core::data as core;
use sidereon_core::data::{
    AnalysisCenter, ProductDate, ProductDateTime, ProductIdentity, ProductType,
    SpaceWeatherProduct, UltraIssue,
};

fn to_data_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn center(code: &str) -> PyResult<AnalysisCenter> {
    code.parse().map_err(to_data_err)
}

fn product_type(code: &str) -> PyResult<ProductType> {
    code.parse().map_err(to_data_err)
}

fn space_weather_product(code: &str) -> PyResult<SpaceWeatherProduct> {
    code.parse().map_err(to_data_err)
}

fn date(year: i32, month: u8, day: u8) -> PyResult<ProductDate> {
    ProductDate::new(year, month, day).map_err(to_data_err)
}

#[pyfunction]
fn data_centers() -> Vec<String> {
    core::centers()
        .iter()
        .map(|center| center.code().to_string())
        .collect()
}

#[pyfunction]
fn data_content_types() -> Vec<String> {
    core::product_types()
        .iter()
        .map(|descriptor| descriptor.product_type.code().to_string())
        .collect()
}

#[pyfunction]
fn data_allowed_hosts() -> Vec<String> {
    core::allowed_hosts()
        .iter()
        .map(|host| (*host).to_string())
        .collect()
}

#[pyfunction]
fn data_center_entry(code: &str) -> PyResult<(String, String, Vec<String>, Vec<String>)> {
    let center = center(code)?;
    let entry = core::center_catalog(center).expect("catalog entry exists for enum variant");
    Ok((
        entry.protocol.as_str().to_string(),
        entry.host.to_string(),
        entry
            .products
            .iter()
            .map(|product| product.product_type.code().to_string())
            .collect(),
        entry
            .issues
            .iter()
            .map(|issue| (*issue).to_string())
            .collect(),
    ))
}

#[pyfunction]
fn data_default_sample(center_code: &str, product_code: &str) -> PyResult<String> {
    core::default_sample(center(center_code)?, product_type(product_code)?)
        .map(ToOwned::to_owned)
        .map_err(to_data_err)
}

#[pyfunction]
fn data_default_sample_for_date(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
) -> PyResult<String> {
    core::default_sample_for_date(
        center(center_code)?,
        product_type(product_code)?,
        date(year, month, day)?,
    )
    .map(ToOwned::to_owned)
    .map_err(to_data_err)
}

#[pyfunction]
fn data_supported_samples(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
    issue: Option<&str>,
) -> PyResult<Vec<String>> {
    core::supported_samples(
        center(center_code)?,
        product_type(product_code)?,
        date(year, month, day)?,
        issue,
    )
    .map(|samples| samples.iter().map(|sample| (*sample).to_owned()).collect())
    .map_err(to_data_err)
}

#[pyfunction]
fn data_product_sample(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
    issue: Option<&str>,
) -> PyResult<String> {
    core::product(
        center(center_code)?,
        product_type(product_code)?,
        date(year, month, day)?,
        None,
        issue,
    )
    .map(|product| product.sample)
    .map_err(to_data_err)
}

#[pyfunction]
fn data_product_solution_class(center_code: &str, product_code: &str) -> PyResult<String> {
    core::product_solution_class(center(center_code)?, product_type(product_code)?)
        .map(|solution| solution.code().to_string())
        .map_err(to_data_err)
}

#[pyfunction]
fn data_sp3_content_start_convention(
    center_code: &str,
    year: i32,
    month: u8,
    day: u8,
    issue: Option<&str>,
) -> PyResult<(String, i64)> {
    core::sp3_content_start_convention(center(center_code)?, date(year, month, day)?, issue)
        .map(|convention| {
            (
                convention.code().to_owned(),
                convention.content_start_offset_s(),
            )
        })
        .map_err(to_data_err)
}

#[pyfunction]
fn data_gps_week(year: i32, month: u8, day: u8) -> PyResult<u32> {
    core::gps_week(date(year, month, day)?).map_err(to_data_err)
}

#[pyfunction]
fn data_day_of_year(year: i32, month: u8, day: u8) -> PyResult<u16> {
    Ok(core::day_of_year(date(year, month, day)?))
}

#[pyfunction]
fn data_predicted_day_offset(center_code: &str) -> PyResult<i64> {
    Ok(core::predicted_day_offset(center(center_code)?))
}

#[pyfunction]
fn data_canonical_filename(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
    sample: Option<&str>,
    issue: Option<&str>,
) -> PyResult<String> {
    core::canonical_filename(
        center(center_code)?,
        product_type(product_code)?,
        date(year, month, day)?,
        sample,
        issue,
    )
    .map_err(to_data_err)
}

#[pyfunction]
fn data_archive_url(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
    sample: Option<&str>,
    issue: Option<&str>,
) -> PyResult<String> {
    core::archive_url(
        center(center_code)?,
        product_type(product_code)?,
        date(year, month, day)?,
        sample,
        issue,
    )
    .map_err(to_data_err)
}

#[pyfunction]
fn data_archive_compression(center_code: &str, product_code: &str) -> PyResult<&'static str> {
    let convention = core::product_convention(center(center_code)?, product_type(product_code)?)
        .map_err(to_data_err)?;
    Ok(convention.compression.as_str())
}

fn identity_json(identity: &ProductIdentity) -> PyResult<String> {
    serde_json::to_string(&serde_json::json!({
        "family": identity.family.code(),
        "analysis_center": identity.analysis_center.code(),
        "publisher": identity.publisher.code(),
        "solution_class": identity.solution.code(),
        "campaign": identity.campaign.code(),
        "filename_version": identity.version,
        "date": format!(
            "{:04}-{:02}-{:02}",
            identity.date.year, identity.date.month, identity.date.day
        ),
        "issue": identity.issue.as_deref().unwrap_or(""),
        "span": identity.span,
        "sample": identity.sample,
        "official_filename": identity.official_filename,
        "format": identity.format.code(),
        "format_version": identity.format_version,
        "prediction_horizon_days": identity.prediction_horizon_days,
    }))
    .map_err(to_data_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn data_product_identity(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
    sample: Option<&str>,
    issue: Option<&str>,
    span: Option<&str>,
    official_filename: Option<&str>,
) -> PyResult<String> {
    let mut identity = core::product_identity(
        center(center_code)?,
        product_type(product_code)?,
        date(year, month, day)?,
        sample,
        issue,
    )
    .map_err(to_data_err)?;
    if let Some(span) = span {
        identity.span = span.to_owned();
    }
    if let Some(official_filename) = official_filename {
        identity.official_filename = official_filename.to_owned();
    }
    identity.validate().map_err(to_data_err)?;
    identity_json(&identity)
}

type DistributionLocationTuple = (String, Option<String>, String, String);

#[pyfunction]
fn data_distribution_location_for_identity(
    identity_json: &str,
    source_code: &str,
) -> PyResult<DistributionLocationTuple> {
    let location = core::distribution_location_for_identity(
        &crate::exact_cache::identity(identity_json)?,
        crate::exact_cache::source(source_code)?,
    )
    .map_err(to_data_err)?;
    Ok((
        location.source.code().to_string(),
        location.original_url,
        location.archive_filename,
        location.compression.as_str().to_string(),
    ))
}

#[pyfunction]
fn data_skadi_source_entry() -> (String, String, String, String) {
    let entry = core::skadi_source_entry();
    (
        entry.protocol.as_str().to_string(),
        entry.host.to_string(),
        entry.compression.as_str().to_string(),
        entry.root_url.to_string(),
    )
}

#[pyfunction]
fn data_space_weather_source_entry() -> (String, String, String, String) {
    let entry = core::space_weather_source_entry();
    (
        entry.protocol.as_str().to_string(),
        entry.host.to_string(),
        entry.compression.as_str().to_string(),
        entry.root_url.to_string(),
    )
}

#[pyfunction]
fn data_space_weather_filename(product_code: &str) -> PyResult<String> {
    Ok(core::space_weather_filename(space_weather_product(product_code)?).to_string())
}

#[pyfunction]
fn data_space_weather_archive_url(product_code: &str) -> PyResult<String> {
    Ok(core::space_weather_archive_url(space_weather_product(
        product_code,
    )?))
}

#[pyfunction]
fn data_space_weather_cache_relpath(product_code: &str) -> PyResult<String> {
    Ok(core::space_weather_cache_relpath(space_weather_product(
        product_code,
    )?))
}

#[pyfunction]
fn data_skadi_tile_id(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::skadi_tile_id(lat_index, lon_index).map_err(to_data_err)
}

#[pyfunction]
fn data_skadi_band(lat_index: i32) -> PyResult<String> {
    core::skadi_band(lat_index).map_err(to_data_err)
}

#[pyfunction]
fn data_skadi_archive_url(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::skadi_archive_url(lat_index, lon_index).map_err(to_data_err)
}

#[pyfunction]
fn data_terrain_tile_index(lat_deg: f64, lon_deg: f64) -> PyResult<(i32, i32)> {
    core::terrain_tile_index(lat_deg, lon_deg).map_err(to_data_err)
}

#[pyfunction]
fn data_dted_tile_filename(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::dted_tile_filename(lat_index, lon_index).map_err(to_data_err)
}

#[pyfunction]
fn data_dted_block_dir(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::dted_block_dir(lat_index, lon_index).map_err(to_data_err)
}

#[pyfunction]
fn data_dted_cache_relpath(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::dted_cache_relpath(lat_index, lon_index).map_err(to_data_err)
}

#[pyfunction]
fn data_parse_skadi_tile_id(id: &str) -> PyResult<(i32, i32)> {
    core::parse_skadi_tile_id(id).map_err(to_data_err)
}

#[pyfunction]
fn data_hgt_to_dted<'py>(
    py: Python<'py>,
    lat_index: i32,
    lon_index: i32,
    hgt: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let dted = core::hgt_to_dted(lat_index, lon_index, hgt).map_err(to_data_err)?;
    Ok(PyBytes::new(py, &dted))
}

#[pyfunction]
fn data_ultra_issue_candidates(
    center_code: &str,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
) -> PyResult<Vec<(i32, u8, u8, String)>> {
    let target =
        ProductDateTime::new(date(year, month, day)?, hour, minute, second).map_err(to_data_err)?;
    core::ultra_issue_candidates(center(center_code)?, target)
        .map(|candidates| {
            candidates
                .into_iter()
                .map(|candidate| {
                    (
                        candidate.date.year,
                        candidate.date.month,
                        candidate.date.day,
                        candidate.issue,
                    )
                })
                .collect()
        })
        .map_err(to_data_err)
}

type UltraSp3LocationTuple = (String, String, String, String, String, String);

#[pyfunction]
fn data_ultra_sp3_locations(
    center_code: &str,
    year: i32,
    month: u8,
    day: u8,
    issue: &str,
) -> PyResult<Vec<UltraSp3LocationTuple>> {
    core::ultra_sp3_locations(center(center_code)?, date(year, month, day)?, issue)
        .map(|locations| {
            locations
                .into_iter()
                .map(|location| {
                    (
                        location.pattern,
                        location.span,
                        location.sample,
                        location.filename,
                        location.url,
                        location.compression.as_str().to_string(),
                    )
                })
                .collect()
        })
        .map_err(to_data_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Mirrors the core timestamp signature used by the Python API.
fn data_latest_ultra_issue(
    center_code: &str,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
    available: Option<Vec<(i32, u8, u8, String)>>,
) -> PyResult<(i32, u8, u8, String)> {
    let target =
        ProductDateTime::new(date(year, month, day)?, hour, minute, second).map_err(to_data_err)?;
    let available = available
        .unwrap_or_default()
        .into_iter()
        .map(|(year, month, day, issue)| {
            UltraIssue::new(date(year, month, day)?, &issue).map_err(to_data_err)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let available_ref = if available.is_empty() {
        None
    } else {
        Some(available.as_slice())
    };
    core::latest_ultra_issue(center(center_code)?, target, available_ref)
        .map(|issue| {
            (
                issue.date.year,
                issue.date.month,
                issue.date.day,
                issue.issue,
            )
        })
        .map_err(to_data_err)
}

#[pyfunction]
fn data_gim_date_candidates(
    center_code: &str,
    year: i32,
    month: u8,
    day: u8,
    lookback: u32,
) -> PyResult<Vec<(i32, u8, u8)>> {
    core::gim_date_candidates(center(center_code)?, date(year, month, day)?, lookback)
        .map(|dates| {
            dates
                .into_iter()
                .map(|d| (d.year, d.month, d.day))
                .collect()
        })
        .map_err(to_data_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(data_centers, m)?)?;
    m.add_function(wrap_pyfunction!(data_content_types, m)?)?;
    m.add_function(wrap_pyfunction!(data_allowed_hosts, m)?)?;
    m.add_function(wrap_pyfunction!(data_center_entry, m)?)?;
    m.add_function(wrap_pyfunction!(data_default_sample, m)?)?;
    m.add_function(wrap_pyfunction!(data_default_sample_for_date, m)?)?;
    m.add_function(wrap_pyfunction!(data_supported_samples, m)?)?;
    m.add_function(wrap_pyfunction!(data_product_sample, m)?)?;
    m.add_function(wrap_pyfunction!(data_product_solution_class, m)?)?;
    m.add_function(wrap_pyfunction!(data_sp3_content_start_convention, m)?)?;
    m.add_function(wrap_pyfunction!(data_gps_week, m)?)?;
    m.add_function(wrap_pyfunction!(data_day_of_year, m)?)?;
    m.add_function(wrap_pyfunction!(data_predicted_day_offset, m)?)?;
    m.add_function(wrap_pyfunction!(data_canonical_filename, m)?)?;
    m.add_function(wrap_pyfunction!(data_archive_url, m)?)?;
    m.add_function(wrap_pyfunction!(data_archive_compression, m)?)?;
    m.add_function(wrap_pyfunction!(data_product_identity, m)?)?;
    m.add_function(wrap_pyfunction!(
        data_distribution_location_for_identity,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(data_skadi_source_entry, m)?)?;
    m.add_function(wrap_pyfunction!(data_space_weather_source_entry, m)?)?;
    m.add_function(wrap_pyfunction!(data_space_weather_filename, m)?)?;
    m.add_function(wrap_pyfunction!(data_space_weather_archive_url, m)?)?;
    m.add_function(wrap_pyfunction!(data_space_weather_cache_relpath, m)?)?;
    m.add_function(wrap_pyfunction!(data_skadi_tile_id, m)?)?;
    m.add_function(wrap_pyfunction!(data_skadi_band, m)?)?;
    m.add_function(wrap_pyfunction!(data_skadi_archive_url, m)?)?;
    m.add_function(wrap_pyfunction!(data_terrain_tile_index, m)?)?;
    m.add_function(wrap_pyfunction!(data_dted_tile_filename, m)?)?;
    m.add_function(wrap_pyfunction!(data_dted_block_dir, m)?)?;
    m.add_function(wrap_pyfunction!(data_dted_cache_relpath, m)?)?;
    m.add_function(wrap_pyfunction!(data_parse_skadi_tile_id, m)?)?;
    m.add_function(wrap_pyfunction!(data_hgt_to_dted, m)?)?;
    m.add_function(wrap_pyfunction!(data_ultra_issue_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(data_ultra_sp3_locations, m)?)?;
    m.add_function(wrap_pyfunction!(data_latest_ultra_issue, m)?)?;
    m.add_function(wrap_pyfunction!(data_gim_date_candidates, m)?)?;
    Ok(())
}
