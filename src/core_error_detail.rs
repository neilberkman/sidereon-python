//! Owned Python dictionaries for structured core errors crossing observable APIs.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use sidereon_core::observables::{ObservablesError, ObservablesInputErrorKind};
use sidereon_core::Error as CoreError;

fn float_bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn set_float(dict: &Bound<'_, PyDict>, name: &str, value: f64) -> PyResult<()> {
    dict.set_item(name, value)?;
    dict.set_item(format!("{name}_bits"), float_bits(value))
}

fn kind_name(error: &CoreError) -> &'static str {
    match error {
        CoreError::Parse(_) => "parse",
        CoreError::UnknownSatellite(_) => "unknown_satellite",
        CoreError::MissingGlonassChannel => "missing_glonass_channel",
        CoreError::MissingTerrainTile { .. } => "missing_terrain_tile",
        CoreError::UnknownTerrainElevation { .. } => "unknown_terrain_elevation",
        CoreError::NonWgs84TerrainTile { .. } => "non_wgs84_terrain_tile",
        CoreError::TerrainTile { .. } => "terrain_tile",
        CoreError::TerrainTileOrigin { .. } => "terrain_tile_origin",
        CoreError::IonexOutOfCoverage(_) => "ionex_out_of_coverage",
        CoreError::IonexNodesNotAvailable(_) => "ionex_nodes_not_available",
        CoreError::IonexSlantUnavailable(_) => "ionex_slant_unavailable",
        CoreError::IonexEpoch(_) => "ionex_epoch",
        CoreError::EpochOutOfRange => "epoch_out_of_range",
        CoreError::InsufficientPreciseNodes { .. } => "insufficient_precise_nodes",
        CoreError::InvalidInput(_) => "invalid_input",
        CoreError::Sp3EpochInterval(_) => "sp3_epoch_interval",
        CoreError::Sp3MergeTolerance(_) => "sp3_merge_tolerance",
        CoreError::ContinuityOptions(_) => "continuity_options",
        CoreError::SbasEncode(_) => "sbas_encode",
        CoreError::RtcmEncode(_) => "rtcm_encode",
        CoreError::RtcmConversion(_) => "rtcm_conversion",
        CoreError::Ut1OutsideCoverage(_) => "ut1_outside_coverage",
        #[allow(unreachable_patterns)]
        _ => "unknown",
    }
}

fn ionex_coverage_kind(error: sidereon_core::atmosphere::IonexCoverageError) -> &'static str {
    use sidereon_core::atmosphere::IonexCoverageError as E;
    match error {
        E::EpochBeforeFirstMap => "epoch_before_first_map",
        E::EpochAfterLastMap => "epoch_after_last_map",
        E::LatitudeOutOfRange => "latitude_out_of_range",
        E::LongitudeOutOfRange => "longitude_out_of_range",
    }
}

fn ionex_epoch_kind(error: sidereon_core::atmosphere::IonexEpochError) -> &'static str {
    use sidereon_core::atmosphere::IonexEpochError as E;
    match error {
        E::NotWholeSecond { .. } => "not_whole_second",
        E::FractionalUtcSecond { .. } => "fractional_utc_second",
        E::NoExactUtcOffset { .. } => "no_exact_utc_offset",
        E::InsertedLeapSecond { .. } => "inserted_leap_second",
        E::BeforeIntegerLeapSeconds { .. } => "before_integer_leap_seconds",
        E::OutOfRange { .. } => "out_of_range",
        E::YearOutOfField { .. } => "year_out_of_field",
        _ => "unknown",
    }
}

fn degrade_reason(reason: sidereon_core::astro::time::DegradeReason) -> &'static str {
    match reason {
        sidereon_core::astro::time::DegradeReason::BeforeCoverage => "before_coverage",
        sidereon_core::astro::time::DegradeReason::AfterCoverage => "after_coverage",
    }
}

fn epoch_interval_reason(
    reason: sidereon_core::ephemeris::Sp3EpochIntervalRejection,
) -> &'static str {
    use sidereon_core::ephemeris::Sp3EpochIntervalRejection as R;
    match reason {
        R::NotFinite => "not_finite",
        R::NotPositive => "not_positive",
        R::NotWholeTicks => "not_whole_ticks",
        R::BeyondTickResolution => "beyond_tick_resolution",
        R::OutsideSpecificationRange => "outside_specification_range",
        _ => "unknown",
    }
}

fn continuity_reason(reason: sidereon_core::ephemeris::ContinuityOptionRejection) -> &'static str {
    use sidereon_core::ephemeris::ContinuityOptionRejection as R;
    match reason {
        R::NotFinite => "not_finite",
        R::Negative => "negative",
        _ => "unknown",
    }
}

fn merge_tolerance_field(field: sidereon_core::ephemeris::MergeToleranceField) -> &'static str {
    use sidereon_core::ephemeris::MergeToleranceField as F;
    match field {
        F::Position => "position",
        F::Clock => "clock",
        F::OutlierPosition => "outlier_position",
        F::OutlierClock => "outlier_clock",
        _ => "unknown",
    }
}

fn ionex_nodes(
    dict: &Bound<'_, PyDict>,
    nodes: sidereon_core::atmosphere::IonexMissingNodes,
) -> PyResult<()> {
    dict.set_item("map_number", nodes.map_number)?;
    dict.set_item("lat_index", nodes.lat_index)?;
    dict.set_item("lon_index", nodes.lon_index)?;
    dict.set_item("lon_index_next", nodes.lon_index_next)?;
    dict.set_item("missing", nodes.missing)?;
    Ok(())
}

fn dted_kind(error: &sidereon_core::terrain::DtedTileError) -> &'static str {
    use sidereon_core::terrain::DtedTileError as E;
    match error {
        E::Io { .. } => "io",
        E::TooShort { .. } => "too_short",
        E::MissingUhl1 { .. } => "missing_uhl1",
        E::InvalidEncoding(_) => "invalid_encoding",
        E::InvalidField(_) => "invalid_field",
        E::InvalidDimensions { .. } => "invalid_dimensions",
        E::Truncated { .. } => "truncated",
        E::Outside { .. } => "outside",
        E::PostingIndexOutOfBounds { .. } => "posting_index_out_of_bounds",
        E::MissingDataSentinel { .. } => "missing_data_sentinel",
        E::Checksum { .. } => "checksum",
        E::EmptyCoordinate => "empty_coordinate",
        E::InvalidHemisphere { .. } => "invalid_hemisphere",
        E::NegativePostingIndex { .. } => "negative_posting_index",
        E::CoordinateOutOfRange { .. } => "coordinate_out_of_range",
        E::WrongHemisphere { .. } => "wrong_hemisphere",
        E::OriginNotWholeDegree { .. } => "origin_not_whole_degree",
        E::IntervalCountMismatch { .. } => "interval_count_mismatch",
        E::ProfileLongitudeCountMismatch { .. } => "profile_longitude_count_mismatch",
        E::UnsupportedPartialProfile { .. } => "unsupported_partial_profile",
        E::NullPosting { .. } => "null_posting",
        #[allow(unreachable_patterns)]
        _ => "unknown",
    }
}

fn rtcm_nested_cause(py: Python<'_>, family: &str, error: &CoreError) -> PyResult<Py<PyDict>> {
    let (kind, fields) = crate::rtcm::core_error_nested_fields(py, error)?;
    let nested = PyDict::new(py);
    nested.set_item("family", family)?;
    nested.set_item("kind", kind)?;
    nested.set_item("message", error.to_string())?;
    for (key, value) in fields.iter() {
        nested.set_item(key, value)?;
    }
    Ok(nested.unbind())
}

pub(crate) fn core_error_detail(py: Python<'_>, error: &CoreError) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("family", "CoreError")?;
    dict.set_item("kind", kind_name(error))?;
    dict.set_item("message", error.to_string())?;
    match error {
        CoreError::Parse(message) => {
            dict.set_item("parse_message", message)?;
        }
        CoreError::InvalidInput(message) => {
            dict.set_item("input_message", message)?;
        }
        CoreError::UnknownSatellite(satellite) => {
            dict.set_item("satellite_id", satellite.to_string())?;
        }
        CoreError::MissingGlonassChannel | CoreError::EpochOutOfRange => {}
        CoreError::MissingTerrainTile {
            lat_index,
            lon_index,
        } => {
            dict.set_item("lat_index", lat_index)?;
            dict.set_item("lon_index", lon_index)?;
        }
        CoreError::UnknownTerrainElevation {
            lat_index,
            lon_index,
            latitude_posting,
            longitude_posting,
        } => {
            dict.set_item("lat_index", lat_index)?;
            dict.set_item("lon_index", lon_index)?;
            dict.set_item("latitude_posting", latitude_posting)?;
            dict.set_item("longitude_posting", longitude_posting)?;
        }
        CoreError::NonWgs84TerrainTile {
            lat_index,
            lon_index,
            datum,
        } => {
            dict.set_item("lat_index", lat_index)?;
            dict.set_item("lon_index", lon_index)?;
            dict.set_item("datum", datum.to_string())?;
        }
        CoreError::TerrainTile {
            lat_index,
            lon_index,
            error: cause,
        } => {
            dict.set_item("lat_index", lat_index)?;
            dict.set_item("lon_index", lon_index)?;
            let nested = PyDict::new(py);
            nested.set_item("family", "DtedTileError")?;
            nested.set_item("kind", dted_kind(cause))?;
            crate::terrain::tile_details(&nested, cause)?;
            if let sidereon_core::terrain::DtedTileError::Io { message, .. } = cause.as_ref() {
                nested.set_item("io_message", message)?;
            }
            nested.set_item("message", cause.to_string())?;
            dict.set_item("cause", nested)?;
        }
        CoreError::TerrainTileOrigin {
            path,
            lat_index,
            lon_index,
            origin_latitude,
            origin_longitude,
        } => {
            dict.set_item("path", path.display().to_string())?;
            dict.set_item("lat_index", lat_index)?;
            dict.set_item("lon_index", lon_index)?;
            dict.set_item("origin_latitude", origin_latitude)?;
            dict.set_item("origin_longitude", origin_longitude)?;
        }
        CoreError::IonexOutOfCoverage(cause) => {
            let nested = PyDict::new(py);
            nested.set_item("family", "IonexCoverageError")?;
            nested.set_item("kind", ionex_coverage_kind(*cause))?;
            nested.set_item("message", cause.to_string())?;
            dict.set_item("cause", nested)?;
        }
        CoreError::IonexNodesNotAvailable(cause) => {
            let nested = PyDict::new(py);
            nested.set_item("family", "IonexNodeGap")?;
            nested.set_item("kind", "missing_nodes")?;
            nested.set_item("message", cause.to_string())?;
            if let Some(earlier) = cause.earlier {
                let details = PyDict::new(py);
                ionex_nodes(&details, earlier)?;
                nested.set_item("earlier", details)?;
            } else {
                nested.set_item("earlier", py.None())?;
            }
            if let Some(later) = cause.later {
                let details = PyDict::new(py);
                ionex_nodes(&details, later)?;
                nested.set_item("later", details)?;
            } else {
                nested.set_item("later", py.None())?;
            }
            dict.set_item("cause", nested)?;
        }
        CoreError::IonexSlantUnavailable(cause) => {
            let nested = PyDict::new(py);
            nested.set_item("family", "IonexSlantRefusal")?;
            nested.set_item("message", cause.to_string())?;
            match cause {
                sidereon_core::atmosphere::IonexSlantRefusal::VaryingHeights {
                    map_number,
                    lat_index,
                    lon_index,
                }
                | sidereon_core::atmosphere::IonexSlantRefusal::HeightNotAvailable {
                    map_number,
                    lat_index,
                    lon_index,
                } => {
                    nested.set_item(
                        "kind",
                        match cause {
                            sidereon_core::atmosphere::IonexSlantRefusal::VaryingHeights {
                                ..
                            } => "varying_heights",
                            _ => "height_not_available",
                        },
                    )?;
                    nested.set_item("map_number", map_number)?;
                    nested.set_item("lat_index", lat_index)?;
                    nested.set_item("lon_index", lon_index)?;
                }
                sidereon_core::atmosphere::IonexSlantRefusal::MappingFunction(declaration) => {
                    nested.set_item("kind", "mapping_function")?;
                    match declaration {
                        sidereon_core::atmosphere::IonexMappingDeclaration::Declared(function) => {
                            nested.set_item("declaration", function.code())?;
                        }
                        sidereon_core::atmosphere::IonexMappingDeclaration::Absent => {
                            nested.set_item("declaration", py.None())?;
                        }
                    }
                }
                _ => nested.set_item("kind", "unknown")?,
            }
            dict.set_item("cause", nested)?;
        }
        CoreError::IonexEpoch(cause) => {
            let nested = PyDict::new(py);
            nested.set_item("family", "IonexEpochError")?;
            nested.set_item("message", cause.to_string())?;
            match cause {
                sidereon_core::atmosphere::IonexEpochError::NotWholeSecond { scale }
                | sidereon_core::atmosphere::IonexEpochError::FractionalUtcSecond { scale }
                | sidereon_core::atmosphere::IonexEpochError::NoExactUtcOffset { scale }
                | sidereon_core::atmosphere::IonexEpochError::InsertedLeapSecond { scale }
                | sidereon_core::atmosphere::IonexEpochError::BeforeIntegerLeapSeconds { scale }
                | sidereon_core::atmosphere::IonexEpochError::OutOfRange { scale } => {
                    nested.set_item("scale", scale.abbrev())?;
                    nested.set_item("kind", ionex_epoch_kind(*cause))?;
                }
                sidereon_core::atmosphere::IonexEpochError::YearOutOfField { utc_j2000_s } => {
                    nested.set_item("kind", "year_out_of_field")?;
                    nested.set_item("utc_j2000_s", utc_j2000_s)?;
                }
                _ => nested.set_item("kind", "unknown")?,
            }
            dict.set_item("cause", nested)?;
        }
        CoreError::InsufficientPreciseNodes {
            sat,
            nodes,
            required,
        } => {
            dict.set_item("satellite_id", sat.to_string())?;
            dict.set_item("nodes", nodes)?;
            dict.set_item("required", required)?;
        }
        CoreError::Sp3EpochInterval(error) => {
            dict.set_item("field", error.field)?;
            set_float(&dict, "value", error.value)?;
            dict.set_item("reason", epoch_interval_reason(error.reason))?;
        }
        CoreError::Sp3MergeTolerance(error) => {
            dict.set_item("field", merge_tolerance_field(error.field))?;
            set_float(&dict, "value", error.value)?;
        }
        CoreError::ContinuityOptions(error) => {
            dict.set_item("field", error.field)?;
            set_float(&dict, "value", error.value)?;
            dict.set_item("reason", continuity_reason(error.reason))?;
        }
        CoreError::SbasEncode(cause) => {
            dict.set_item(
                "cause",
                rtcm_nested_cause(py, "SbasEncodeError", &CoreError::SbasEncode(cause.clone()))?,
            )?;
        }
        CoreError::RtcmEncode(cause) => {
            dict.set_item(
                "cause",
                rtcm_nested_cause(py, "RtcmEncodeError", &CoreError::RtcmEncode(cause.clone()))?,
            )?;
        }
        CoreError::RtcmConversion(cause) => {
            dict.set_item(
                "cause",
                rtcm_nested_cause(
                    py,
                    "RtcmConversionError",
                    &CoreError::RtcmConversion(cause.clone()),
                )?,
            )?;
        }
        CoreError::Ut1OutsideCoverage(reason) => {
            dict.set_item("reason", degrade_reason(*reason))?;
        }
        #[allow(unreachable_patterns)]
        other => {
            dict.set_item("kind", "unknown")?;
            dict.set_item("message", other.to_string())?;
        }
    }
    Ok(dict.unbind())
}

fn input_kind(kind: ObservablesInputErrorKind) -> &'static str {
    match kind {
        ObservablesInputErrorKind::NonFinite => "non_finite",
        ObservablesInputErrorKind::NotPositive => "not_positive",
        ObservablesInputErrorKind::Negative => "negative",
        ObservablesInputErrorKind::OutOfRange => "out_of_range",
        ObservablesInputErrorKind::Missing => "missing",
        ObservablesInputErrorKind::FloatParse => "float_parse",
        ObservablesInputErrorKind::IntParse => "int_parse",
        ObservablesInputErrorKind::InvalidCivilDate => "invalid_civil_date",
        ObservablesInputErrorKind::InvalidCivilTime => "invalid_civil_time",
    }
}

pub(crate) fn observables_error_detail(
    py: Python<'_>,
    error: &ObservablesError,
) -> PyResult<Py<PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("family", "ObservablesError")?;
    dict.set_item("message", error.to_string())?;
    match error {
        ObservablesError::InvalidInput { field, kind } => {
            dict.set_item("kind", "invalid_input")?;
            dict.set_item("field", field)?;
            dict.set_item("reason", input_kind(*kind))?;
        }
        ObservablesError::NoEphemeris => dict.set_item("kind", "no_ephemeris")?,
        ObservablesError::Ephemeris(cause) => {
            dict.set_item("kind", "ephemeris")?;
            dict.set_item("cause", core_error_detail(py, cause)?)?;
        }
        ObservablesError::Media(cause) => {
            dict.set_item("kind", "media")?;
            dict.set_item("cause", core_error_detail(py, cause)?)?;
        }
    }
    Ok(dict.unbind())
}

pub(crate) fn attach_core_error_detail(error: PyErr, core: &CoreError) -> PyErr {
    Python::with_gil(|py| {
        let detail = match core_error_detail(py, core) {
            Ok(detail) => detail,
            Err(error) => return error,
        };
        if let Err(error) = error.value(py).setattr("detail", detail) {
            return error;
        }
        error
    })
}

pub(crate) fn attach_observables_error_detail(
    error: PyErr,
    observable: &ObservablesError,
) -> PyErr {
    Python::with_gil(|py| {
        let detail = match observables_error_detail(py, observable) {
            Ok(detail) => detail,
            Err(error) => return error,
        };
        if let Err(error) = error.value(py).setattr("detail", detail) {
            return error;
        }
        error
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidereon_core::rtcm::{RtcmEncodeError, RtcmFieldEncoding};
    use sidereon_core::sbas::SbasEncodeError;
    use sidereon_core::terrain::DtedTileError;

    fn nested_value<'py>(detail: &Bound<'py, PyDict>) -> Bound<'py, PyDict> {
        detail
            .get_item("cause")
            .expect("read nested cause")
            .expect("nested cause exists")
            .downcast_into::<PyDict>()
            .expect("nested cause is a dictionary")
    }

    fn field_string(detail: &Bound<'_, PyDict>, field: &str) -> String {
        detail
            .get_item(field)
            .expect("read detail field")
            .expect("detail field exists")
            .extract()
            .expect("extract detail string")
    }

    #[test]
    fn nested_codec_and_dted_causes_keep_variant_fields() {
        Python::with_gil(|py| {
            let dted_error = CoreError::TerrainTile {
                lat_index: 12,
                lon_index: 34,
                error: Box::new(DtedTileError::InvalidDimensions {
                    path: "N12E034.dt1".to_owned(),
                    lon_count: 1,
                    lat_count: 257,
                }),
            };
            let dted = core_error_detail(py, &dted_error).expect("map DTED error");
            let dted_cause = nested_value(dted.bind(py));
            assert_eq!(field_string(&dted_cause, "kind"), "invalid_dimensions");
            assert_eq!(field_string(&dted_cause, "path"), "N12E034.dt1");
            assert_eq!(
                dted_cause
                    .get_item("lon_count")
                    .expect("read longitude count")
                    .expect("longitude count exists")
                    .extract::<usize>()
                    .expect("extract longitude count"),
                1
            );
            assert_eq!(
                field_string(&dted_cause, "message"),
                DtedTileError::InvalidDimensions {
                    path: "N12E034.dt1".to_owned(),
                    lon_count: 1,
                    lat_count: 257,
                }
                .to_string()
            );

            let io_error = CoreError::TerrainTile {
                lat_index: 12,
                lon_index: 34,
                error: Box::new(DtedTileError::Io {
                    path: "N12E034.dt1".to_owned(),
                    message: "permission denied".to_owned(),
                }),
            };
            let io_detail = core_error_detail(py, &io_error).expect("map DTED I/O error");
            let io_cause = nested_value(io_detail.bind(py));
            assert_eq!(field_string(&io_cause, "kind"), "io");
            assert_eq!(field_string(&io_cause, "io_message"), "permission denied");
            assert_eq!(
                field_string(&io_cause, "message"),
                "N12E034.dt1: permission denied"
            );

            let sbas_error =
                CoreError::SbasEncode(Box::new(SbasEncodeError::PadBits { value: 64 }));
            let sbas = core_error_detail(py, &sbas_error).expect("map SBAS error");
            let sbas_cause = nested_value(sbas.bind(py));
            assert_eq!(field_string(&sbas_cause, "kind"), "sbas_pad_bits");
            assert_eq!(
                sbas_cause
                    .get_item("value")
                    .expect("read pad value")
                    .expect("pad value exists")
                    .extract::<u8>()
                    .expect("extract pad value"),
                64
            );

            let rtcm_error = CoreError::RtcmEncode(Box::new(RtcmEncodeError::FieldOutOfRange {
                message_number: 1005,
                field: "station_id".to_owned(),
                value: 4096,
                width: 12,
                encoding: RtcmFieldEncoding::Unsigned,
            }));
            let rtcm = core_error_detail(py, &rtcm_error).expect("map RTCM error");
            let rtcm_cause = nested_value(rtcm.bind(py));
            assert_eq!(field_string(&rtcm_cause, "kind"), "field_out_of_range");
            assert_eq!(field_string(&rtcm_cause, "field"), "station_id");
            assert_eq!(
                rtcm_cause
                    .get_item("value")
                    .expect("read RTCM value")
                    .expect("RTCM value exists")
                    .extract::<i128>()
                    .expect("extract RTCM value"),
                4096
            );

            let conversion_error = CoreError::RtcmConversion(Box::new(
                sidereon_core::rtcm::RtcmConversionError::WeekMismatch {
                    message_number: 1019,
                    full_week: 2048,
                    week: 0,
                },
            ));
            let conversion =
                core_error_detail(py, &conversion_error).expect("map RTCM conversion error");
            let conversion_cause = nested_value(conversion.bind(py));
            assert_eq!(field_string(&conversion_cause, "kind"), "week_mismatch");
            assert_eq!(
                conversion_cause
                    .get_item("full_week")
                    .expect("read full week")
                    .expect("full week exists")
                    .extract::<u32>()
                    .expect("extract full week"),
                2048
            );
            assert_eq!(
                conversion_cause
                    .get_item("week")
                    .expect("read reduced week")
                    .expect("reduced week exists")
                    .extract::<u16>()
                    .expect("extract reduced week"),
                0
            );
        });
    }
}
#[cfg(test)]
mod grouped1071_python_error_detail_contract {
    use super::*;
    use pyo3::types::PyModule;
    use sidereon_core::astro::time::{model::TimeScale, DegradeReason};
    use sidereon_core::atmosphere::ionosphere::{
        IonexCoverageError, IonexEpochError, IonexMissingNodes, IonexNodeGap, IonexSlantRefusal,
    };
    use sidereon_core::ephemeris::{
        ContinuityOptionRejection, ContinuityOptionsError, MergeToleranceError,
        MergeToleranceField, Sp3EpochIntervalError, Sp3EpochIntervalRejection,
    };
    use sidereon_core::rtcm::{RtcmConversionError, RtcmEncodeError, RtcmFieldEncoding};
    use sidereon_core::sbas::SbasEncodeError;
    use sidereon_core::terrain::{DtedHorizontalDatum, DtedTileError};
    use sidereon_core::{Error as CoreError, GnssSatelliteId};

    fn check(py: Python<'_>, error: &CoreError, expected_json: &str, display: &str) {
        assert_eq!(error.to_string(), display);
        assert_eq!(error, &error.clone());
        let attach = || {
            attach_core_error_detail(
                pyo3::exceptions::PyValueError::new_err(error.to_string()),
                error,
            )
            .value(py)
            .getattr("detail")
            .expect("exception detail")
        };
        let actual = attach();
        let repeated = attach();
        let expected = PyModule::import(py, "json")
            .unwrap()
            .call_method1("loads", (expected_json,))
            .unwrap();
        assert!(actual.eq(&expected).unwrap(), "detail for {error}");
        assert!(actual.eq(&repeated).unwrap(), "repeat detail for {error}");
        assert!(actual.downcast::<PyDict>().is_ok());
    }

    #[test]
    fn grouped1071_selected_error_variants_have_exact_attached_details() {
        const SELECTED_VARIANT_FIELD_DISPLAY_EQ_ROWS: usize = 48;
        const SELECTED_FROM_ROWS: usize = 6;
        const SELECTED_ROWS: usize = 54;
        assert_eq!(
            SELECTED_VARIANT_FIELD_DISPLAY_EQ_ROWS + SELECTED_FROM_ROWS,
            SELECTED_ROWS
        );
        Python::with_gil(|py| {
            let sat: GnssSatelliteId = "G09".parse().unwrap();
            let node = IonexMissingNodes {
                map_number: 2,
                lat_index: 3,
                lon_index: 4,
                lon_index_next: 5,
                missing: [true, false, false, true],
            };
            let dted = DtedTileError::InvalidDimensions {
                path: "N12E034.dt1".into(),
                lon_count: 1,
                lat_count: 257,
            };
            let cases: Vec<(CoreError, &str, &str)> = vec![
                (CoreError::InvalidInput("bad input".into()), r#"{"family":"CoreError","kind":"invalid_input","message":"invalid input: bad input","input_message":"bad input"}"#, "invalid input: bad input"),
                (CoreError::Parse("bad product".into()), r#"{"family":"CoreError","kind":"parse","message":"parse error: bad product","parse_message":"bad product"}"#, "parse error: bad product"),
                (CoreError::UnknownSatellite(sat), r#"{"family":"CoreError","kind":"unknown_satellite","message":"unknown satellite: G09","satellite_id":"G09"}"#, "unknown satellite: G09"),
                (CoreError::MissingGlonassChannel, r#"{"family":"CoreError","kind":"missing_glonass_channel","message":"missing GLONASS FDMA channel"}"#, "missing GLONASS FDMA channel"),
                (CoreError::MissingTerrainTile { lat_index: 12, lon_index: 34 }, r#"{"family":"CoreError","kind":"missing_terrain_tile","message":"missing terrain tile (12,34)","lat_index":12,"lon_index":34}"#, "missing terrain tile (12,34)"),
                (CoreError::UnknownTerrainElevation { lat_index: 12, lon_index: 34, latitude_posting: 7, longitude_posting: 8 }, r#"{"family":"CoreError","kind":"unknown_terrain_elevation","message":"unknown terrain elevation at posting lon=8 lat=7 of tile (12,34)","lat_index":12,"lon_index":34,"latitude_posting":7,"longitude_posting":8}"#, "unknown terrain elevation at posting lon=8 lat=7 of tile (12,34)"),
                (CoreError::NonWgs84TerrainTile { lat_index: 12, lon_index: 34, datum: DtedHorizontalDatum::Wgs72 }, r#"{"family":"CoreError","kind":"non_wgs84_terrain_tile","message":"terrain tile (12,34) states horizontal datum WGS72, not WGS84","lat_index":12,"lon_index":34,"datum":"WGS72"}"#, "terrain tile (12,34) states horizontal datum WGS72, not WGS84"),
                (CoreError::TerrainTile { lat_index: 12, lon_index: 34, error: Box::new(dted) }, r#"{"family":"CoreError","kind":"terrain_tile","message":"terrain tile (12,34): N12E034.dt1 has invalid DTED dimensions lon_count=1 lat_count=257; both must be at least 2","lat_index":12,"lon_index":34,"cause":{"family":"DtedTileError","kind":"invalid_dimensions","message":"N12E034.dt1 has invalid DTED dimensions lon_count=1 lat_count=257; both must be at least 2","path":"N12E034.dt1","lon_count":1,"lat_count":257}}"#, "terrain tile (12,34): N12E034.dt1 has invalid DTED dimensions lon_count=1 lat_count=257; both must be at least 2"),
                (CoreError::TerrainTileOrigin { path: "/tmp/N12E034.dt1".into(), lat_index: 12, lon_index: 34, origin_latitude: 11, origin_longitude: 34 }, r#"{"family":"CoreError","kind":"terrain_tile_origin","message":"/tmp/N12E034.dt1: DTED origin (11,34) does not match tile (12,34) named by the file","path":"/tmp/N12E034.dt1","lat_index":12,"lon_index":34,"origin_latitude":11,"origin_longitude":34}"#, "/tmp/N12E034.dt1: DTED origin (11,34) does not match tile (12,34) named by the file"),
                (CoreError::IonexOutOfCoverage(IonexCoverageError::EpochBeforeFirstMap), r#"{"family":"CoreError","kind":"ionex_out_of_coverage","message":"IONEX out of coverage: epoch precedes first map","cause":{"family":"IonexCoverageError","kind":"epoch_before_first_map","message":"epoch precedes first map"}}"#, "IONEX out of coverage: epoch precedes first map"),
                (CoreError::IonexNodesNotAvailable(Box::new(IonexNodeGap { earlier: Some(node), later: None })), r#"{"family":"CoreError","kind":"ionex_nodes_not_available","message":"IONEX nodes not available: map 2 cell [3][4] missing [3][4] [4][5]","cause":{"family":"IonexNodeGap","kind":"missing_nodes","message":"map 2 cell [3][4] missing [3][4] [4][5]","earlier":{"map_number":2,"lat_index":3,"lon_index":4,"lon_index_next":5,"missing":[true,false,false,true]},"later":null}}"#, "IONEX nodes not available: map 2 cell [3][4] missing [3][4] [4][5]"),
                (CoreError::IonexSlantUnavailable(IonexSlantRefusal::VaryingHeights { map_number: 2, lat_index: 3, lon_index: 4 }), r#"{"family":"CoreError","kind":"ionex_slant_unavailable","message":"IONEX slant delay unavailable: height map 2 gives node [3][4] another single-layer height than the first node; the slant delay uses one shell height","cause":{"family":"IonexSlantRefusal","kind":"varying_heights","message":"height map 2 gives node [3][4] another single-layer height than the first node; the slant delay uses one shell height","map_number":2,"lat_index":3,"lon_index":4}}"#, "IONEX slant delay unavailable: height map 2 gives node [3][4] another single-layer height than the first node; the slant delay uses one shell height"),
                (CoreError::IonexEpoch(IonexEpochError::NotWholeSecond { scale: TimeScale::Utc }), r#"{"family":"CoreError","kind":"ionex_epoch","message":"invalid input: IONEX map epoch in UTC is not a whole J2000 second","cause":{"family":"IonexEpochError","kind":"not_whole_second","message":"IONEX map epoch in UTC is not a whole J2000 second","scale":"UTC"}}"#, "invalid input: IONEX map epoch in UTC is not a whole J2000 second"),
                (CoreError::EpochOutOfRange, r#"{"family":"CoreError","kind":"epoch_out_of_range","message":"epoch out of range"}"#, "epoch out of range"),
                (CoreError::InsufficientPreciseNodes { sat, nodes: 2, required: 4 }, r#"{"family":"CoreError","kind":"insufficient_precise_nodes","message":"G09: 2 precise orbit nodes serve the query, 4 are needed","satellite_id":"G09","nodes":2,"required":4}"#, "G09: 2 precise orbit nodes serve the query, 4 are needed"),
                (CoreError::Sp3EpochInterval(Sp3EpochIntervalError { field: "interval_s", value: 0.5, reason: Sp3EpochIntervalRejection::NotWholeTicks }), r#"{"family":"CoreError","kind":"sp3_epoch_interval","message":"invalid input: interval_s 0.5 s is not an SP3 epoch interval: it is not a whole number of the 10-nanosecond ticks an SP3 epoch states","field":"interval_s","value":0.5,"value_bits":"3fe0000000000000","reason":"not_whole_ticks"}"#, "invalid input: interval_s 0.5 s is not an SP3 epoch interval: it is not a whole number of the 10-nanosecond ticks an SP3 epoch states"),
                (CoreError::Sp3MergeTolerance(MergeToleranceError { field: MergeToleranceField::Position, value: -1.0 }), r#"{"family":"CoreError","kind":"sp3_merge_tolerance","message":"invalid input: SP3 merge position tolerance (m) -1 must be finite and nonnegative","field":"position","value":-1.0,"value_bits":"bff0000000000000"}"#, "invalid input: SP3 merge position tolerance (m) -1 must be finite and nonnegative"),
                (CoreError::ContinuityOptions(ContinuityOptionsError { field: "residual_tolerance_m", value: -2.5, reason: ContinuityOptionRejection::Negative }), r#"{"family":"CoreError","kind":"continuity_options","message":"invalid input: continuity residual_tolerance_m -2.5 is refused: it is negative","field":"residual_tolerance_m","value":-2.5,"value_bits":"c004000000000000","reason":"negative"}"#, "invalid input: continuity residual_tolerance_m -2.5 is refused: it is negative"),
                (CoreError::SbasEncode(Box::new(SbasEncodeError::PadBits { value: 64 })), r#"{"family":"CoreError","kind":"sbas_encode","message":"SBAS encode error: SBAS pad bits value 64 does not fit six bits","cause":{"family":"SbasEncodeError","kind":"sbas_pad_bits","message":"SBAS encode error: SBAS pad bits value 64 does not fit six bits","value":64}}"#, "SBAS encode error: SBAS pad bits value 64 does not fit six bits"),
                (CoreError::RtcmEncode(Box::new(RtcmEncodeError::FieldOutOfRange { message_number: 1005, field: "station_id".into(), value: 4096, width: 12, encoding: RtcmFieldEncoding::Unsigned })), r#"{"family":"CoreError","kind":"rtcm_encode","message":"invalid input: RTCM 1005 station_id 4096 does not fit its 12-bit unsigned field (0..=4095)","cause":{"family":"RtcmEncodeError","kind":"field_out_of_range","message":"invalid input: RTCM 1005 station_id 4096 does not fit its 12-bit unsigned field (0..=4095)","message_number":1005,"field":"station_id","value":4096,"width":12,"encoding":"unsigned"}}"#, "invalid input: RTCM 1005 station_id 4096 does not fit its 12-bit unsigned field (0..=4095)"),
                (CoreError::RtcmConversion(Box::new(RtcmConversionError::WeekMismatch { message_number: 1019, full_week: 2048, week: 0 })), r#"{"family":"CoreError","kind":"rtcm_conversion","message":"invalid input: GPS full week 2048 disagrees with 10-bit RTCM week 0","cause":{"family":"RtcmConversionError","kind":"week_mismatch","message":"invalid input: GPS full week 2048 disagrees with 10-bit RTCM week 0","message_number":1019,"full_week":2048,"week":0}}"#, "invalid input: GPS full week 2048 disagrees with 10-bit RTCM week 0"),
                (CoreError::Ut1OutsideCoverage(DegradeReason::BeforeCoverage), r#"{"family":"CoreError","kind":"ut1_outside_coverage","message":"UT1 outside the table: instant precedes the UT1 table coverage","reason":"before_coverage"}"#, "UT1 outside the table: instant precedes the UT1 table coverage"),
            ];
            assert_eq!(cases.len(), 22);
            for (error, expected, display) in &cases {
                check(py, error, expected, display);
            }
        });
    }

    #[test]
    fn grouped1071_from_conversions_reach_the_attached_detail_route() {
        Python::with_gil(|py| {
            use sidereon_core::rinex::observations::RinexObsWriteError;
            // Message::decode exercises OutOfInput -> DecodeError -> Error, covering both From rows.
            let cases: Vec<(CoreError, &str, &str)> = vec![
                (RinexObsWriteError::NotVersionTwo { version: 3.0 }.into(), r#"{"family":"CoreError","kind":"invalid_input","message":"invalid input: RINEX OBS version 3 is not a version 2","input_message":"RINEX OBS version 3 is not a version 2"}"#, "invalid input: RINEX OBS version 3 is not a version 2"),
                (sidereon_core::rtcm::Message::decode(&[]).expect_err("truncated body"), r#"{"family":"CoreError","kind":"parse","message":"parse error: RTCM body truncated: need 12 more bits, 0 remain","parse_message":"RTCM body truncated: need 12 more bits, 0 remain"}"#, "parse error: RTCM body truncated: need 12 more bits, 0 remain"),
                (SbasEncodeError::PadBits { value: 64 }.into(), r#"{"family":"CoreError","kind":"sbas_encode","message":"SBAS encode error: SBAS pad bits value 64 does not fit six bits","cause":{"family":"SbasEncodeError","kind":"sbas_pad_bits","message":"SBAS encode error: SBAS pad bits value 64 does not fit six bits","value":64}}"#, "SBAS encode error: SBAS pad bits value 64 does not fit six bits"),
                (RtcmEncodeError::FieldOutOfRange { message_number: 1005, field: "station_id".into(), value: 4096, width: 12, encoding: RtcmFieldEncoding::Unsigned }.into(), r#"{"family":"CoreError","kind":"rtcm_encode","message":"invalid input: RTCM 1005 station_id 4096 does not fit its 12-bit unsigned field (0..=4095)","cause":{"family":"RtcmEncodeError","kind":"field_out_of_range","message":"invalid input: RTCM 1005 station_id 4096 does not fit its 12-bit unsigned field (0..=4095)","message_number":1005,"field":"station_id","value":4096,"width":12,"encoding":"unsigned"}}"#, "invalid input: RTCM 1005 station_id 4096 does not fit its 12-bit unsigned field (0..=4095)"),
                (RtcmConversionError::WeekMismatch { message_number: 1019, full_week: 2048, week: 0 }.into(), r#"{"family":"CoreError","kind":"rtcm_conversion","message":"invalid input: GPS full week 2048 disagrees with 10-bit RTCM week 0","cause":{"family":"RtcmConversionError","kind":"week_mismatch","message":"invalid input: GPS full week 2048 disagrees with 10-bit RTCM week 0","message_number":1019,"full_week":2048,"week":0}}"#, "invalid input: GPS full week 2048 disagrees with 10-bit RTCM week 0"),
            ];
            assert_eq!(
                cases.len() + 1,
                6,
                "decode accounts for both RTCM From rows"
            );
            for (error, expected, display) in &cases {
                check(py, error, expected, display);
            }
        });
    }
}
