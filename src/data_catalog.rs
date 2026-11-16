//! Core-backed data-product catalog bridge for the Python fetch layer.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyModule};

use crate::sgp4_elements::{PyElementSet, PyTleElements};

use sidereon_core::data as core;
use sidereon_core::data::{
    AnalysisCenter, ProductDate, ProductDateTime, ProductIdentity, ProductType,
    SpaceWeatherProduct, UltraIssue,
};

const GP_FORMATS: [core::CelestrakGpFormat; 8] = [
    core::CelestrakGpFormat::Tle,
    core::CelestrakGpFormat::ThreeLine,
    core::CelestrakGpFormat::TwoLine,
    core::CelestrakGpFormat::Xml,
    core::CelestrakGpFormat::Kvn,
    core::CelestrakGpFormat::Json,
    core::CelestrakGpFormat::JsonPretty,
    core::CelestrakGpFormat::Csv,
];
const GP_SPECIALS: [core::CelestrakGpSpecial; 3] = [
    core::CelestrakGpSpecial::Gpz,
    core::CelestrakGpSpecial::GpzPlus,
    core::CelestrakGpSpecial::Decaying,
];

fn gp_format(value: &str) -> PyResult<core::CelestrakGpFormat> {
    GP_FORMATS
        .into_iter()
        .find(|format| format.as_str() == value)
        .ok_or_else(|| gp_boundary_error("invalid_celestrak_gp_format", "FORMAT", value))
}

fn gp_special(value: &str) -> PyResult<core::CelestrakGpSpecial> {
    GP_SPECIALS
        .into_iter()
        .find(|special| special.as_str() == value)
        .ok_or_else(|| gp_boundary_error("invalid_celestrak_gp_special", "SPECIAL", value))
}

#[pyfunction]
fn data_celestrak_gp_format_values() -> Vec<&'static str> {
    GP_FORMATS.into_iter().map(|value| value.as_str()).collect()
}

#[pyfunction]
fn data_celestrak_gp_special_values() -> Vec<&'static str> {
    GP_SPECIALS
        .into_iter()
        .map(|value| value.as_str())
        .collect()
}

fn gp_boundary_error(kind: &str, selector: &str, value: &str) -> PyErr {
    Python::with_gil(|py| {
        let message = format!("invalid CelesTrak GP {selector}: {value}");
        let error = PyValueError::new_err(message.clone());
        let detail = PyDict::new(py);
        let result = (|| -> PyResult<()> {
            detail.set_item("family", "DataCatalogError")?;
            detail.set_item("kind", kind)?;
            detail.set_item("selector", selector)?;
            detail.set_item("value", value)?;
            detail.set_item("message", message)?;
            error.value(py).setattr("detail", detail)
        })();
        match result {
            Ok(()) => error,
            Err(error) => error,
        }
    })
}

#[pyfunction]
fn data_celestrak_gp_format_as_str(value: &str) -> PyResult<&'static str> {
    Ok(gp_format(value)?.as_str())
}

#[pyfunction]
fn data_celestrak_gp_special_as_str(value: &str) -> PyResult<&'static str> {
    Ok(gp_special(value)?.as_str())
}

#[pyfunction]
fn data_celestrak_gp_source_entry() -> (&'static str, &'static str, &'static str) {
    let e = core::celestrak_gp_source_entry();
    (e.protocol.as_str(), e.host, e.root_url)
}

#[pyfunction]
fn data_navcen_source_entry() -> (&'static str, &'static str, &'static str) {
    let e = core::navcen_source_entry();
    (e.protocol.as_str(), e.host, e.root_url)
}

#[pyfunction]
fn data_navcen_constellation_url() -> &'static str {
    core::navcen_constellation_url()
}

#[pyfunction]
fn data_celestrak_group(system: crate::marshal::PyGnssSystem) -> &'static str {
    sidereon_core::constellation::celestrak_group(system.into())
}

#[pyfunction]
fn data_celestrak_constellation_url(
    system: crate::marshal::PyGnssSystem,
    format: &str,
) -> PyResult<String> {
    core::celestrak_constellation_url(system.into(), gp_format(format)?).map_err(data_catalog_err)
}

#[pyfunction]
fn data_celestrak_gp_url(
    kind: &str,
    value: &Bound<'_, pyo3::types::PyAny>,
    format: &str,
) -> PyResult<String> {
    use core::CelestrakGpQuery as Q;
    let format = gp_format(format)?;
    let string_value;
    let query = match kind {
        "group" | "international_designator" | "name" => {
            string_value = value.extract::<String>().map_err(|_| {
                let selector = match kind {
                    "group" => "GROUP",
                    "name" => "NAME",
                    _ => "INTDES",
                };
                let display = value.str().map(|s| s.to_string()).unwrap_or_default();
                gp_boundary_error("invalid_celestrak_gp_selector", selector, &display)
            })?;
            match kind {
                "group" => Q::Group(&string_value),
                "name" => Q::Name(&string_value),
                _ => Q::InternationalDesignator(&string_value),
            }
        }
        "catalog_number" => {
            let invalid = || {
                let display = value
                    .str()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|_| "<unprintable>".to_string());
                gp_boundary_error("invalid_celestrak_gp_selector", "CATNR", &display)
            };
            if value.is_instance_of::<pyo3::types::PyBool>() {
                return Err(invalid());
            }
            Q::CatalogNumber(value.extract().map_err(|_| invalid())?)
        }
        "special" => Q::Special(gp_special(&value.extract::<String>().map_err(|_| {
            gp_boundary_error(
                "invalid_celestrak_gp_special",
                "SPECIAL",
                &value.str().map(|s| s.to_string()).unwrap_or_default(),
            )
        })?)?),
        _ => {
            return Err(gp_boundary_error(
                "invalid_celestrak_gp_query_kind",
                "QUERY",
                kind,
            ))
        }
    };
    core::celestrak_gp_url(query, format).map_err(data_catalog_err)
}

fn data_catalog_err(err: core::DataCatalogError) -> PyErr {
    use core::DataCatalogError as E;

    Python::with_gil(|py| {
        let result = (|| -> PyResult<PyErr> {
            let detail = PyDict::new(py);
            let kind = match &err {
                E::InvalidCelestrakGpSelector { selector, value } => {
                    detail.set_item("selector", selector)?;
                    detail.set_item("value", value)?;
                    "invalid_celestrak_gp_selector"
                }
                E::EmptyCelestrakGpResponse => "empty_celestrak_gp_response",
                E::CelestrakGpDataNotFound => "celestrak_gp_data_not_found",
                E::CelestrakGpNotModified {
                    query,
                    last_successful_download,
                } => {
                    detail.set_item("query", query)?;
                    detail.set_item("last_successful_download", last_successful_download)?;
                    "celestrak_gp_not_modified"
                }
                E::CelestrakGpHtmlResponse => "celestrak_gp_html_response",
                E::InvalidCelestrakGpResponse {
                    reason,
                    line,
                    column,
                } => {
                    detail.set_item("reason", reason)?;
                    detail.set_item("line", line)?;
                    detail.set_item("column", column)?;
                    "invalid_celestrak_gp_response"
                }
                E::UnknownCenter(value) => {
                    detail.set_item("value", value)?;
                    "unknown_center"
                }
                E::UnknownProductType(value) => {
                    detail.set_item("value", value)?;
                    "unknown_product_type"
                }
                E::UnsupportedProduct {
                    center,
                    product_type,
                } => {
                    detail.set_item("center", center.code())?;
                    detail.set_item("product_type", product_type.code())?;
                    "unsupported_product"
                }
                E::UnsupportedDistribution {
                    source,
                    product_type,
                } => {
                    detail.set_item("source", source.code())?;
                    detail.set_item("product_type", product_type.code())?;
                    "unsupported_distribution"
                }
                E::UnsupportedProductEra {
                    center,
                    product_type,
                    date,
                } => {
                    detail.set_item("center", center.code())?;
                    detail.set_item("product_type", product_type.code())?;
                    detail.set_item("date", date.to_string())?;
                    "unsupported_product_era"
                }
                E::UnsupportedDistributionEra {
                    source,
                    center,
                    product_type,
                    date,
                } => {
                    detail.set_item("source", source.code())?;
                    detail.set_item("center", center.code())?;
                    detail.set_item("product_type", product_type.code())?;
                    detail.set_item("date", date.to_string())?;
                    "unsupported_distribution_era"
                }
                E::NoDistributionSources => "no_distribution_sources",
                E::InvalidOfficialFilename(value) => {
                    detail.set_item("value", value)?;
                    "invalid_official_filename"
                }
                E::InconsistentProductIdentity { field } => {
                    detail.set_item("field", field)?;
                    "inconsistent_product_identity"
                }
                E::NoOpenMirror {
                    center,
                    product_type,
                } => {
                    detail.set_item("center", center)?;
                    detail.set_item("product_type", product_type)?;
                    "no_open_mirror"
                }
                E::InvalidDate { year, month, day } => {
                    detail.set_item("year", year)?;
                    detail.set_item("month", month)?;
                    detail.set_item("day", day)?;
                    "invalid_date"
                }
                E::DateOutOfRange => "date_out_of_range",
                E::DateBeforeGpsEpoch(date) => {
                    detail.set_item("date", date.to_string())?;
                    "date_before_gps_epoch"
                }
                E::InvalidGpsDayOfWeek(value) => {
                    detail.set_item("value", value)?;
                    "invalid_gps_day_of_week"
                }
                E::InvalidSample(value) => {
                    detail.set_item("value", value)?;
                    "invalid_sample"
                }
                E::UnsupportedSample {
                    center,
                    product_type,
                    sample,
                } => {
                    detail.set_item("center", center.code())?;
                    detail.set_item("product_type", product_type.code())?;
                    detail.set_item("sample", sample)?;
                    "unsupported_sample"
                }
                E::InvalidSpan(value) => {
                    detail.set_item("value", value)?;
                    "invalid_span"
                }
                E::InvalidIssue(value) => {
                    detail.set_item("value", value)?;
                    "invalid_issue"
                }
                E::MissingIssue { center } => {
                    detail.set_item("center", center.code())?;
                    "missing_issue"
                }
                E::UnexpectedIssue { center } => {
                    detail.set_item("center", center.code())?;
                    "unexpected_issue"
                }
                E::UnsupportedIssue { center, issue } => {
                    detail.set_item("center", center.code())?;
                    detail.set_item("issue", issue)?;
                    "unsupported_issue"
                }
                E::InvalidDateTime {
                    hour,
                    minute,
                    second,
                } => {
                    detail.set_item("hour", hour)?;
                    detail.set_item("minute", minute)?;
                    detail.set_item("second", second)?;
                    "invalid_date_time"
                }
                E::NoUltraIssue => "no_ultra_issue",
                E::NoAvailableUltraIssue => "no_available_ultra_issue",
                E::UnsupportedNominalSchedule {
                    center,
                    product_type,
                } => {
                    detail.set_item("center", center.code())?;
                    detail.set_item("product_type", product_type.code())?;
                    "unsupported_nominal_schedule"
                }
                E::UnrecognizedArchiveListing { reason } => {
                    detail.set_item("reason", reason)?;
                    "unrecognized_archive_listing"
                }
                E::InvalidStation(value) => {
                    detail.set_item("value", value)?;
                    "invalid_station"
                }
                E::InvalidCoordinate {
                    lat_deg_bits,
                    lon_deg_bits,
                } => {
                    detail.set_item("lat_deg_bits", lat_deg_bits)?;
                    detail.set_item("lon_deg_bits", lon_deg_bits)?;
                    "invalid_coordinate"
                }
                E::InvalidTileIndex {
                    lat_index,
                    lon_index,
                } => {
                    detail.set_item("lat_index", lat_index)?;
                    detail.set_item("lon_index", lon_index)?;
                    "invalid_tile_index"
                }
                E::InvalidTileId(value) => {
                    detail.set_item("value", value)?;
                    "invalid_tile_id"
                }
            };
            detail.set_item("family", "DataCatalogError")?;
            detail.set_item("kind", kind)?;
            detail.set_item("message", err.to_string())?;
            let error = PyValueError::new_err(err.to_string());
            error.value(py).setattr("detail", detail)?;
            Ok(error)
        })();
        match result {
            Ok(error) => error,
            Err(error) => error,
        }
    })
}

fn to_value_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn center(code: &str) -> PyResult<AnalysisCenter> {
    code.parse().map_err(data_catalog_err)
}

fn product_type(code: &str) -> PyResult<ProductType> {
    code.parse().map_err(data_catalog_err)
}

fn space_weather_product(code: &str) -> PyResult<SpaceWeatherProduct> {
    code.parse().map_err(data_catalog_err)
}

fn date(year: i32, month: u8, day: u8) -> PyResult<ProductDate> {
    ProductDate::new(year, month, day).map_err(data_catalog_err)
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
        .map_err(data_catalog_err)
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
    .map_err(data_catalog_err)
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
    .map_err(data_catalog_err)
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
    .map_err(data_catalog_err)
}

#[pyfunction]
fn data_product_solution_class(center_code: &str, product_code: &str) -> PyResult<String> {
    core::product_solution_class(center(center_code)?, product_type(product_code)?)
        .map(|solution| solution.code().to_string())
        .map_err(data_catalog_err)
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
        .map_err(data_catalog_err)
}

#[pyfunction]
fn data_gps_week(year: i32, month: u8, day: u8) -> PyResult<u32> {
    core::gps_week(date(year, month, day)?).map_err(data_catalog_err)
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
    .map_err(data_catalog_err)
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
    .map_err(data_catalog_err)
}

#[pyfunction]
fn data_archive_compression(center_code: &str, product_code: &str) -> PyResult<&'static str> {
    let convention = core::product_convention(center(center_code)?, product_type(product_code)?)
        .map_err(data_catalog_err)?;
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
    .map_err(to_value_err)
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
    .map_err(data_catalog_err)?;
    if let Some(span) = span {
        identity.span = span.to_owned();
    }
    if let Some(official_filename) = official_filename {
        identity.official_filename = official_filename.to_owned();
    }
    identity.validate().map_err(data_catalog_err)?;
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
    .map_err(data_catalog_err)?;
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
    core::skadi_tile_id(lat_index, lon_index).map_err(data_catalog_err)
}

#[pyfunction]
fn data_skadi_band(lat_index: i32) -> PyResult<String> {
    core::skadi_band(lat_index).map_err(data_catalog_err)
}

#[pyfunction]
fn data_skadi_archive_url(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::skadi_archive_url(lat_index, lon_index).map_err(data_catalog_err)
}

#[pyfunction]
fn data_terrain_tile_index(lat_deg: f64, lon_deg: f64) -> PyResult<(i32, i32)> {
    core::terrain_tile_index(lat_deg, lon_deg).map_err(data_catalog_err)
}

#[pyfunction]
fn data_dted_tile_filename(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::dted_tile_filename(lat_index, lon_index).map_err(data_catalog_err)
}

#[pyfunction]
fn data_dted_block_dir(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::dted_block_dir(lat_index, lon_index).map_err(data_catalog_err)
}

#[pyfunction]
fn data_dted_cache_relpath(lat_index: i32, lon_index: i32) -> PyResult<String> {
    core::dted_cache_relpath(lat_index, lon_index).map_err(data_catalog_err)
}

#[pyfunction]
fn data_parse_skadi_tile_id(id: &str) -> PyResult<(i32, i32)> {
    core::parse_skadi_tile_id(id).map_err(data_catalog_err)
}

#[pyfunction]
fn data_hgt_to_dted<'py>(
    py: Python<'py>,
    lat_index: i32,
    lon_index: i32,
    hgt: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let dted = core::hgt_to_dted(lat_index, lon_index, hgt).map_err(to_value_err)?;
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
    let target = ProductDateTime::new(date(year, month, day)?, hour, minute, second)
        .map_err(data_catalog_err)?;
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
        .map_err(data_catalog_err)
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
        .map_err(data_catalog_err)
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
    let target = ProductDateTime::new(date(year, month, day)?, hour, minute, second)
        .map_err(data_catalog_err)?;
    let available = available
        .unwrap_or_default()
        .into_iter()
        .map(|(year, month, day, issue)| {
            UltraIssue::new(date(year, month, day)?, &issue).map_err(data_catalog_err)
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
        .map_err(data_catalog_err)
}

type PredictedLineCandidateTuple = (String, i32, u8, u8, String, String, String, String);

#[pyfunction]
fn data_predicted_ionex_line_candidates(
    year: i32,
    month: u8,
    day: u8,
    sample: Option<&str>,
) -> PyResult<Vec<PredictedLineCandidateTuple>> {
    core::predicted_ionex_line_candidates(date(year, month, day)?, sample)
        .map_err(data_catalog_err)?
        .into_iter()
        .map(|candidate| {
            let filename = candidate.canonical_filename().map_err(data_catalog_err)?;
            let url = candidate.archive_url().map_err(data_catalog_err)?;
            Ok((
                candidate.center.code().to_string(),
                candidate.date.year,
                candidate.date.month,
                candidate.date.day,
                candidate.sample.clone(),
                candidate.issue.clone().unwrap_or_default(),
                filename,
                url,
            ))
        })
        .collect()
}

#[pyfunction]
fn data_publication_listing_urls(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
) -> PyResult<Vec<String>> {
    core::publication_listing_urls(
        center(center_code)?,
        product_type(product_code)?,
        date(year, month, day)?,
    )
    .map_err(data_catalog_err)
}

/// Archive-listing bodies are unbounded caller input: AIUB's public whole-tree
/// CSV is ~34 MiB over ~426k rows. The parse borrows only the caller's string
/// and returns plain Rust data, so it runs inside `Python::allow_threads` and
/// holds no interpreter lock while it works. Without that, a large listing
/// would stall every other thread in the process for the whole parse.
#[pyfunction]
fn data_parse_archive_listing(
    py: Python<'_>,
    body: &str,
) -> PyResult<Vec<(String, Option<String>)>> {
    py.allow_threads(|| {
        core::parse_archive_listing(body)
            .map(|objects| {
                objects
                    .into_iter()
                    .map(|object| (object.path, object.observed_at))
                    .collect()
            })
            .map_err(data_catalog_err)
    })
}

type PublishedProductTuple = (i32, u8, u8, String, String, Option<String>);

fn published_objects(rows: Vec<(String, Option<String>)>) -> Vec<core::PublishedObject> {
    rows.into_iter()
        .map(|(path, observed_at)| core::PublishedObject { path, observed_at })
        .collect()
}

#[pyfunction]
fn data_newest_published_product(
    center_code: &str,
    product_code: &str,
    objects: Vec<(String, Option<String>)>,
) -> PyResult<Option<PublishedProductTuple>> {
    core::newest_published_product(
        center(center_code)?,
        product_type(product_code)?,
        &published_objects(objects),
    )
    .map(|newest| {
        newest.map(|product| {
            (
                product.date.year,
                product.date.month,
                product.date.day,
                product.issue,
                product.filename,
                product.observed_at,
            )
        })
    })
    .map_err(data_catalog_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Mirrors the core timestamp signature used by the Python API.
fn data_published_issue_age_minutes(
    year: i32,
    month: u8,
    day: u8,
    issue: &str,
    filename: &str,
    now_year: i32,
    now_month: u8,
    now_day: u8,
    now_hour: u8,
    now_minute: u8,
    now_second: u8,
) -> PyResult<i64> {
    let published = core::PublishedProduct {
        date: date(year, month, day)?,
        issue: issue.to_string(),
        filename: filename.to_string(),
        observed_at: None,
    };
    let now = ProductDateTime::new(
        date(now_year, now_month, now_day)?,
        now_hour,
        now_minute,
        now_second,
    )
    .map_err(data_catalog_err)?;
    core::published_issue_age_minutes(&published, now).map_err(data_catalog_err)
}

type ProductDateTimeTuple = (i32, u8, u8, u8, u8, u8);
type NominalCoverageIntervalTuple = (ProductDateTimeTuple, ProductDateTimeTuple);
type NominalIssueTuple = (
    String,
    ProductDateTimeTuple,
    Option<NominalCoverageIntervalTuple>,
    Option<NominalCoverageIntervalTuple>,
);

fn product_datetime_tuple(value: ProductDateTime) -> ProductDateTimeTuple {
    (
        value.date.year,
        value.date.month,
        value.date.day,
        value.hour,
        value.minute,
        value.second,
    )
}

fn coverage_interval_tuple(value: core::NominalCoverageInterval) -> NominalCoverageIntervalTuple {
    (
        product_datetime_tuple(value.from),
        product_datetime_tuple(value.until),
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)] // Mirrors the core's second-resolution UTC timestamp.
fn data_next_issue_due(
    center_code: &str,
    product_code: &str,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: u8,
) -> PyResult<NominalIssueTuple> {
    let now = ProductDateTime::new(date(year, month, day)?, hour, minute, second)
        .map_err(data_catalog_err)?;
    let issue = core::next_issue_due(center(center_code)?, product_type(product_code)?, now)
        .map_err(data_catalog_err)?;
    Ok((
        identity_json(&issue.identity)?,
        product_datetime_tuple(issue.due_at),
        issue.covers.observed.map(coverage_interval_tuple),
        issue.covers.predicted.map(coverage_interval_tuple),
    ))
}

type CandidateSpecTuple = (String, String, i32, u8, u8, Option<String>, Option<String>);

#[pyfunction]
fn data_resolve_first_published(
    candidates: Vec<CandidateSpecTuple>,
    objects: Vec<(String, Option<String>)>,
) -> PyResult<Option<usize>> {
    let specs = candidates
        .into_iter()
        .map(
            |(center_code, product_code, year, month, day, sample, issue)| {
                core::product(
                    center(&center_code)?,
                    product_type(&product_code)?,
                    date(year, month, day)?,
                    sample.as_deref(),
                    issue.as_deref(),
                )
                .map_err(data_catalog_err)
            },
        )
        .collect::<PyResult<Vec<_>>>()?;
    core::resolve_first_published(&specs, &published_objects(objects)).map_err(data_catalog_err)
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
        .map_err(data_catalog_err)
}

/// Native TLE payload of a GP record.
#[pyclass(module = "sidereon._sidereon", name = "CelestrakTleRecord")]
#[derive(Clone)]
pub struct PyCelestrakTleRecord {
    inner: core::CelestrakTleRecord,
}
#[pymethods]
impl PyCelestrakTleRecord {
    /// Parsed fixed-column elements.
    #[getter]
    fn elements(&self) -> PyTleElements {
        PyTleElements {
            inner: self.inner.elements.clone(),
        }
    }
    /// Line 1 as received, without its line ending.
    #[getter]
    fn line1(&self) -> &str {
        &self.inner.line1
    }
    /// Line 2 as received, without its line ending.
    #[getter]
    fn line2(&self) -> &str {
        &self.inner.line2
    }
    /// Checksum findings accepted by the TLE reader.
    #[getter]
    fn checksum_warnings(&self) -> Vec<crate::propagation::PyChecksumWarning> {
        self.inner
            .checksum_warnings
            .iter()
            .map(Into::into)
            .collect()
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A GP record with exact source bytes.
#[pyclass(module = "sidereon._sidereon", name = "CelestrakGpRecord")]
#[derive(Clone)]
pub struct PyCelestrakGpRecord {
    inner: core::CelestrakGpRecord,
}
#[pymethods]
impl PyCelestrakGpRecord {
    /// One-based position among accepted and rejected records.
    #[getter]
    fn record_index(&self) -> usize {
        self.inner.record_index
    }
    /// One-based source line, when available.
    #[getter]
    fn line_number(&self) -> Option<usize> {
        self.inner.line_number
    }
    /// Exact source bytes assigned to this record.
    #[getter]
    fn raw<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.raw.as_bytes())
    }
    /// Object name carried by the native record, when present.
    #[getter]
    fn object_name(&self) -> Option<String> {
        self.inner.object_name.clone()
    }
    /// Native TLE record or OMM.
    #[getter]
    fn payload(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.inner.payload {
            core::CelestrakGpPayload::Tle(tle) => Ok(Py::new(
                py,
                PyCelestrakTleRecord {
                    inner: (**tle).clone(),
                },
            )?
            .into_any()),
            core::CelestrakGpPayload::Omm(omm) => {
                Ok(Py::new(py, crate::omm::PyOmm::from_inner((**omm).clone()))?.into_any())
            }
        }
    }
    /// Convert the native payload to an SGP4 element set.
    fn to_element_set(&self, py: Python<'_>) -> PyResult<PyElementSet> {
        self.inner
            .to_element_set()
            .map(|inner| PyElementSet { inner })
            .map_err(|e| gp_record_error(py, e))
    }
    /// Convert the native payload to an OMM.
    fn to_omm(&self, py: Python<'_>) -> PyResult<crate::omm::PyOmm> {
        self.inner
            .to_omm()
            .map(crate::omm::PyOmm::from_inner)
            .map_err(|e| gp_record_error(py, e))
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
fn gp_record_error(py: Python<'_>, error: core::CelestrakGpRecordError) -> PyErr {
    match error {
        core::CelestrakGpRecordError::Tle(e) => crate::tle_fit_error_detail::tle_parse_error(e),
        core::CelestrakGpRecordError::Omm(e) => crate::omm::to_omm_err(py, e),
    }
}

/// The reason an individual GP record was rejected.
#[pyclass(module = "sidereon._sidereon", name = "CelestrakGpRejectionReason")]
#[derive(Clone)]
pub struct PyCelestrakGpRejectionReason {
    inner: core::CelestrakGpRejectionReason,
}
#[pymethods]
impl PyCelestrakGpRejectionReason {
    /// Native error family: `tle`, `omm`, or `structural`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            core::CelestrakGpRejectionReason::Tle(_) => "tle",
            core::CelestrakGpRejectionReason::Omm(_) => "omm",
            core::CelestrakGpRejectionReason::Structural(_) => "structural",
        }
    }
    /// Structured variant fields, including nested causes.
    #[getter]
    fn detail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        use core::CelestrakGpStructuralError as S;
        match &self.inner {
            core::CelestrakGpRejectionReason::Tle(e) => {
                crate::tle_fit_error_detail::tle_detail(py, e)
            }
            core::CelestrakGpRejectionReason::Omm(e) => crate::omm::omm_detail(py, e),
            core::CelestrakGpRejectionReason::Structural(e) => {
                let d = PyDict::new(py);
                d.set_item("family", "CelestrakGpStructuralError")?;
                d.set_item("message", format!("{e:?}"))?;
                match e {
                    S::UnexpectedContent => d.set_item("kind", "unexpected_content")?,
                    S::IncompleteRecord => d.set_item("kind", "incomplete_record")?,
                    S::InvalidTleStructure => d.set_item("kind", "invalid_tle_structure")?,
                    S::InvalidElementSet(cause) => {
                        d.set_item("kind", "invalid_element_set")?;
                        d.set_item(
                            "cause",
                            crate::tle_fit_error_detail::sgp4_detail(py, cause)?,
                        )?;
                    }
                }
                Ok(d)
            }
        }
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// A rejected GP record with exact source bytes.
#[pyclass(module = "sidereon._sidereon", name = "CelestrakGpRejectedRecord")]
#[derive(Clone)]
pub struct PyCelestrakGpRejectedRecord {
    inner: core::CelestrakGpRejectedRecord,
}
#[pymethods]
impl PyCelestrakGpRejectedRecord {
    /// One-based position among accepted and rejected records.
    #[getter]
    fn record_index(&self) -> usize {
        self.inner.record_index
    }
    /// One-based source line, when available.
    #[getter]
    fn line_number(&self) -> Option<usize> {
        self.inner.line_number
    }
    /// Exact source bytes assigned to this rejection.
    #[getter]
    fn raw<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.raw.as_bytes())
    }
    /// Typed reason the source record was rejected.
    #[getter]
    fn reason(&self) -> PyCelestrakGpRejectionReason {
        PyCelestrakGpRejectionReason {
            inner: self.inner.reason.clone(),
        }
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// Accepted and rejected records from a GP response.
#[pyclass(module = "sidereon._sidereon", name = "CelestrakGpResponse")]
#[derive(Clone)]
pub struct PyCelestrakGpResponse {
    inner: core::CelestrakGpResponse,
}
#[pymethods]
impl PyCelestrakGpResponse {
    /// Accepted records in source order.
    #[getter]
    fn records(&self) -> Vec<PyCelestrakGpRecord> {
        self.inner
            .records
            .iter()
            .cloned()
            .map(|inner| PyCelestrakGpRecord { inner })
            .collect()
    }
    /// Rejected records in source order.
    #[getter]
    fn rejected(&self) -> Vec<PyCelestrakGpRejectedRecord> {
        self.inner
            .rejected
            .iter()
            .cloned()
            .map(|inner| PyCelestrakGpRejectedRecord { inner })
            .collect()
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

#[pyfunction]
fn data_parse_celestrak_gp_response(body: &[u8], format: &str) -> PyResult<PyCelestrakGpResponse> {
    core::parse_celestrak_gp_response(body, gp_format(format)?)
        .map(|inner| PyCelestrakGpResponse { inner })
        .map_err(data_catalog_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(data_celestrak_gp_format_values, m)?)?;
    m.add_function(wrap_pyfunction!(data_celestrak_gp_special_values, m)?)?;
    m.add_function(wrap_pyfunction!(data_celestrak_gp_format_as_str, m)?)?;
    m.add_function(wrap_pyfunction!(data_celestrak_gp_special_as_str, m)?)?;
    m.add_class::<PyCelestrakTleRecord>()?;
    m.add_class::<PyCelestrakGpRecord>()?;
    m.add_class::<PyCelestrakGpRejectionReason>()?;
    m.add_class::<PyCelestrakGpRejectedRecord>()?;
    m.add_class::<PyCelestrakGpResponse>()?;
    m.add_function(wrap_pyfunction!(data_parse_celestrak_gp_response, m)?)?;

    m.add_function(wrap_pyfunction!(data_celestrak_gp_source_entry, m)?)?;
    m.add_function(wrap_pyfunction!(data_navcen_source_entry, m)?)?;
    m.add_function(wrap_pyfunction!(data_navcen_constellation_url, m)?)?;
    m.add_function(wrap_pyfunction!(data_celestrak_group, m)?)?;
    m.add_function(wrap_pyfunction!(data_celestrak_constellation_url, m)?)?;
    m.add_function(wrap_pyfunction!(data_celestrak_gp_url, m)?)?;
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
    m.add_function(wrap_pyfunction!(data_predicted_ionex_line_candidates, m)?)?;
    m.add_function(wrap_pyfunction!(data_publication_listing_urls, m)?)?;
    m.add_function(wrap_pyfunction!(data_parse_archive_listing, m)?)?;
    m.add_function(wrap_pyfunction!(data_newest_published_product, m)?)?;
    m.add_function(wrap_pyfunction!(data_published_issue_age_minutes, m)?)?;
    m.add_function(wrap_pyfunction!(data_next_issue_due, m)?)?;
    m.add_function(wrap_pyfunction!(data_resolve_first_published, m)?)?;
    Ok(())
}

#[cfg(test)]
mod data_catalog_error_detail_tests {
    use super::*;
    use serde_json::{json, Value};

    #[test]
    fn every_catalog_error_returns_its_exact_python_detail_dictionary() {
        let center: AnalysisCenter = "cod".parse().expect("catalog center");
        let product_type: ProductType = "sp3".parse().expect("catalog product type");
        let product_date = ProductDate::new(2024, 1, 2).expect("valid date");
        let before_gps = ProductDate::new(1979, 1, 2).expect("valid pre-GPS date");
        let cases: Vec<(core::DataCatalogError, &str, Value)> = vec![
            (
                core::DataCatalogError::UnknownCenter("bad".into()),
                "unknown_center",
                json!({"value":"bad"}),
            ),
            (
                core::DataCatalogError::UnknownProductType("bad".into()),
                "unknown_product_type",
                json!({"value":"bad"}),
            ),
            (
                core::DataCatalogError::UnsupportedProduct {
                    center,
                    product_type,
                },
                "unsupported_product",
                json!({"center":"cod","product_type":"sp3"}),
            ),
            (
                core::DataCatalogError::UnsupportedDistribution {
                    source: core::DistributionSource::Direct,
                    product_type,
                },
                "unsupported_distribution",
                json!({"source":"direct","product_type":"sp3"}),
            ),
            (
                core::DataCatalogError::UnsupportedProductEra {
                    center,
                    product_type,
                    date: product_date,
                },
                "unsupported_product_era",
                json!({"center":"cod","product_type":"sp3","date":"2024-01-02"}),
            ),
            (
                core::DataCatalogError::UnsupportedDistributionEra {
                    source: core::DistributionSource::NasaCddis,
                    center,
                    product_type,
                    date: product_date,
                },
                "unsupported_distribution_era",
                json!({"source":"nasa_cddis","center":"cod","product_type":"sp3","date":"2024-01-02"}),
            ),
            (
                core::DataCatalogError::NoDistributionSources,
                "no_distribution_sources",
                json!({}),
            ),
            (
                core::DataCatalogError::InvalidOfficialFilename("../unsafe".into()),
                "invalid_official_filename",
                json!({"value":"../unsafe"}),
            ),
            (
                core::DataCatalogError::InconsistentProductIdentity { field: "sample" },
                "inconsistent_product_identity",
                json!({"field":"sample"}),
            ),
            (
                core::DataCatalogError::NoOpenMirror {
                    center: "grg".into(),
                    product_type: "sp3".into(),
                },
                "no_open_mirror",
                json!({"center":"grg","product_type":"sp3"}),
            ),
            (
                core::DataCatalogError::InvalidDate {
                    year: 2024,
                    month: 2,
                    day: 30,
                },
                "invalid_date",
                json!({"year":2024,"month":2,"day":30}),
            ),
            (
                core::DataCatalogError::DateOutOfRange,
                "date_out_of_range",
                json!({}),
            ),
            (
                core::DataCatalogError::DateBeforeGpsEpoch(before_gps),
                "date_before_gps_epoch",
                json!({"date":"1979-01-02"}),
            ),
            (
                core::DataCatalogError::InvalidGpsDayOfWeek(7),
                "invalid_gps_day_of_week",
                json!({"value":7}),
            ),
            (
                core::DataCatalogError::InvalidSample("00X".into()),
                "invalid_sample",
                json!({"value":"00X"}),
            ),
            (
                core::DataCatalogError::UnsupportedSample {
                    center,
                    product_type,
                    sample: "99Z".into(),
                },
                "unsupported_sample",
                json!({"center":"cod","product_type":"sp3","sample":"99Z"}),
            ),
            (
                core::DataCatalogError::InvalidSpan("bad".into()),
                "invalid_span",
                json!({"value":"bad"}),
            ),
            (
                core::DataCatalogError::InvalidIssue("bad".into()),
                "invalid_issue",
                json!({"value":"bad"}),
            ),
            (
                core::DataCatalogError::MissingIssue { center },
                "missing_issue",
                json!({"center":"cod"}),
            ),
            (
                core::DataCatalogError::UnexpectedIssue { center },
                "unexpected_issue",
                json!({"center":"cod"}),
            ),
            (
                core::DataCatalogError::UnsupportedIssue {
                    center,
                    issue: "2359".into(),
                },
                "unsupported_issue",
                json!({"center":"cod","issue":"2359"}),
            ),
            (
                core::DataCatalogError::InvalidDateTime {
                    hour: 24,
                    minute: 60,
                    second: 60,
                },
                "invalid_date_time",
                json!({"hour":24,"minute":60,"second":60}),
            ),
            (
                core::DataCatalogError::NoUltraIssue,
                "no_ultra_issue",
                json!({}),
            ),
            (
                core::DataCatalogError::NoAvailableUltraIssue,
                "no_available_ultra_issue",
                json!({}),
            ),
            (
                core::DataCatalogError::UnsupportedNominalSchedule {
                    center,
                    product_type,
                },
                "unsupported_nominal_schedule",
                json!({"center":"cod","product_type":"sp3"}),
            ),
            (
                core::DataCatalogError::UnrecognizedArchiveListing {
                    reason: "unknown format".into(),
                },
                "unrecognized_archive_listing",
                json!({"reason":"unknown format"}),
            ),
            (
                core::DataCatalogError::InvalidStation("bad".into()),
                "invalid_station",
                json!({"value":"bad"}),
            ),
            (
                core::DataCatalogError::InvalidCoordinate {
                    lat_deg_bits: 1.5f64.to_bits(),
                    lon_deg_bits: (-2.0f64).to_bits(),
                },
                "invalid_coordinate",
                json!({"lat_deg_bits":1.5f64.to_bits(),"lon_deg_bits":(-2.0f64).to_bits()}),
            ),
            (
                core::DataCatalogError::InvalidTileIndex {
                    lat_index: 91,
                    lon_index: -181,
                },
                "invalid_tile_index",
                json!({"lat_index":91,"lon_index":-181}),
            ),
            (
                core::DataCatalogError::InvalidTileId("bad".into()),
                "invalid_tile_id",
                json!({"value":"bad"}),
            ),
        ];
        const DISPLAYS: [&str; 30] = [
            "unknown analysis center \"bad\"",
            "unknown product type \"bad\"",
            "cod does not serve sp3",
            "distributor direct does not serve sp3",
            "cod/sp3 has no cataloged naming convention for 2024-01-02",
            "distributor nasa_cddis has no cataloged cod/sp3 layout for 2024-01-02",
            "exact product request has no distributors",
            "invalid official product filename \"../unsafe\"",
            "product identity field \"sample\" disagrees with its official filename",
            "grg/sp3 has no open mirror",
            "invalid product date 2024-02-30",
            "product date is out of range",
            "product date 1979-01-02 is before the GPS week epoch",
            "invalid GPS day-of-week 7",
            "invalid sample code \"00X\"",
            "cod/sp3 does not publish sample interval \"99Z\"",
            "invalid coverage span \"bad\"",
            "invalid issue time \"bad\"",
            "cod requires an issue time",
            "cod does not take an issue time",
            "cod does not publish issue \"2359\"",
            "invalid product time 24:60:60",
            "no ultra-rapid issue at or before target",
            "no available ultra-rapid issue at or before target",
            "cod/sp3 has no nominal due-time schedule",
            "unrecognized archive listing: unknown format",
            "invalid station code \"bad\"",
            "invalid terrain coordinate lat=1.5 lon=-2",
            "invalid terrain tile index lat=91 lon=-181",
            "invalid skadi tile id \"bad\"",
        ];
        let payload_field_count: usize = cases
            .iter()
            .map(|(_, _, fields)| fields.as_object().expect("field object").len())
            .sum();
        assert_eq!(cases.len(), 30, "DataCatalogError variants in discovery");
        assert_eq!(
            payload_field_count, 44,
            "DataCatalogError payload fields in discovery"
        );
        assert_eq!(
            1 + cases.len() + payload_field_count + 1,
            76,
            "Data discovery rows"
        );

        Python::with_gil(|py| {
            for ((error, kind, fields), display) in cases.into_iter().zip(DISPLAYS) {
                assert_eq!(error, error.clone(), "{kind} source equality");
                let message = error.to_string();
                assert_eq!(message, display, "{kind} literal Display");
                let python_error = data_catalog_err(error.clone());
                let error_text: String = python_error
                    .value(py)
                    .str()
                    .expect("catalog exception display")
                    .extract()
                    .expect("display text");
                assert_eq!(error_text, message, "{kind} display");
                let detail = python_error
                    .value(py)
                    .getattr("detail")
                    .expect("catalog error detail attribute");
                let actual: Value = pythonize::depythonize(&detail).expect("detail dictionary");
                let mut expected = json!({
                    "family": "DataCatalogError",
                    "kind": kind,
                    "message": message,
                });
                expected
                    .as_object_mut()
                    .expect("expected object")
                    .extend(fields.as_object().expect("field object").clone());
                let repeated_detail = data_catalog_err(error)
                    .value(py)
                    .getattr("detail")
                    .expect("repeated catalog error detail");
                let repeated: Value =
                    pythonize::depythonize(&repeated_detail).expect("repeated detail dictionary");
                assert_eq!(actual, expected, "{kind}");
                assert_eq!(repeated, expected, "{kind} repeated projection");
            }
        });
    }
}
