//! Structured details for PPP, RTK, SBAS protection-level, and DGNSS failures.
//!
//! Each entry point is tied to the concrete core error type it receives. This
//! keeps typed payloads at the producer boundary without changing the shared
//! display-only mapper used by unrelated solve APIs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sidereon_core::dgnss::DgnssError;
use sidereon_core::precise_positioning::{
    FixedSolveError as PppFixedSolveError, FloatSolveError as PppFloatSolveError, PppAutoInitError,
};
use sidereon_core::rtk::{
    CodeSmoothingError, CycleSlipPrepError, DoubleDifferenceError, IonosphereFreeBaselineError,
    WideLaneError,
};
use sidereon_core::rtk_filter::{
    FilterStateValidationError, FilterStateValidationKind, FixedSolveError as RtkFixedSolveError,
    FloatSolveError as RtkFloatSolveError, InvalidStateKind, MovingBaselineError,
    MovingBaselineSequenceError, RtkArcError, RtkIonosphereFreeArcError, RtkRinexArcError,
    RtkStaticArcError, RtkWideLaneArcError, RtkWideLaneFixedArcError, UpdateError,
    ValidatedFixedSolveError,
};
use sidereon_core::sbas_pl::SbasPlError;

fn detail<'py>(py: Python<'py>, family: &str, message: &str) -> PyResult<Bound<'py, PyDict>> {
    let result = family_detail(py, family)?;
    result.set_item("message", message)?;
    Ok(result)
}

fn family_detail<'py>(py: Python<'py>, family: &str) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("family", family)?;
    Ok(result)
}

fn set_float(dict: &Bound<'_, PyDict>, field: &str, value: f64) -> PyResult<()> {
    dict.set_item(field, value)?;
    dict.set_item(format!("{field}_bits"), format!("{:016x}", value.to_bits()))
}

fn attach(py: Python<'_>, error: PyErr, payload: Bound<'_, PyDict>) -> PyErr {
    if let Err(error) = error.value(py).setattr("detail", payload) {
        return error;
    }
    error
}

fn solve_error(py: Python<'_>, message: String, payload: PyResult<Bound<'_, PyDict>>) -> PyErr {
    match payload {
        Ok(payload) => attach(py, crate::SolveError::new_err(message), payload),
        Err(error) => error,
    }
}

fn degrade_reason(reason: sidereon_core::astro::time::DegradeReason) -> &'static str {
    crate::degrade_reason_label(reason)
}

fn ppp_float_detail<'py>(
    py: Python<'py>,
    error: &PppFloatSolveError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::precise_positioning::FloatSolveError as E;
    let result = detail(py, "PppFloatSolveError", &error.to_string())?;
    match error {
        E::NoEphemeris {
            satellite_id,
            reason,
        } => {
            result.set_item("kind", "no_ephemeris")?;
            result.set_item("satellite_id", satellite_id)?;
            let cause = PyDict::new(py);
            match reason {
                sidereon_core::precise_positioning::NoEphemerisReason::NoEphemeris => {
                    cause.set_item("kind", "no_ephemeris")?;
                }
                sidereon_core::precise_positioning::NoEphemerisReason::MissingSatelliteClock => {
                    cause.set_item("kind", "missing_satellite_clock")?;
                }
                sidereon_core::precise_positioning::NoEphemerisReason::Reason(reason) => {
                    cause.set_item("kind", "reason")?;
                    cause.set_item("reason", reason)?;
                }
            }
            result.set_item("cause", cause)?;
        }
        E::SingularGeometry => result.set_item("kind", "singular_geometry")?,
        E::InvalidClockCount { expected, actual } => {
            result.set_item("kind", "invalid_clock_count")?;
            result.set_item("expected", expected)?;
            result.set_item("actual", actual)?;
        }
        E::InvalidSolveOption { field, reason } => {
            result.set_item("kind", "invalid_solve_option")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::InsufficientObservationsAfterElevationCutoff {
            cutoff_deg,
            retained_observations,
            required_observations,
        } => {
            result.set_item("kind", "insufficient_observations_after_elevation_cutoff")?;
            set_float(&result, "cutoff_deg", *cutoff_deg)?;
            result.set_item("retained_observations", retained_observations)?;
            result.set_item("required_observations", required_observations)?;
        }
        E::InsufficientObservationsAfterSsrBiasExclusion {
            excluded_observations,
            retained_observations,
            required_observations,
        } => {
            result.set_item("kind", "insufficient_observations_after_ssr_bias_exclusion")?;
            result.set_item("excluded_observations", excluded_observations)?;
            result.set_item("retained_observations", retained_observations)?;
            result.set_item("required_observations", required_observations)?;
        }
        E::MissingAmbiguity(id) => {
            result.set_item("kind", "missing_ambiguity")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::MissingCorrection {
            satellite_id,
            correction,
        } => {
            result.set_item("kind", "missing_correction")?;
            result.set_item("satellite_id", satellite_id)?;
            let cause = PyDict::new(py);
            use sidereon_core::precise_positioning::MissingCorrection as C;
            match correction {
                C::SolidEarthTide => cause.set_item("kind", "solid_earth_tide")?,
                C::PoleTide => cause.set_item("kind", "pole_tide")?,
                C::OceanLoading => cause.set_item("kind", "ocean_loading")?,
                C::PhaseWindup => cause.set_item("kind", "phase_windup")?,
                C::SatelliteAntennaPco => cause.set_item("kind", "satellite_antenna_pco")?,
                C::SatelliteAntennaPcv => cause.set_item("kind", "satellite_antenna_pcv")?,
                C::CodeBias => cause.set_item("kind", "code_bias")?,
                C::SsrCodeBias => cause.set_item("kind", "ssr_code_bias")?,
                C::PhaseBias => cause.set_item("kind", "phase_bias")?,
                C::ReceiverAntennaFrequency(label) => {
                    cause.set_item("kind", "receiver_antenna_frequency")?;
                    cause.set_item("label", label)?;
                }
                C::ReceiverAntennaPcv(label) => {
                    cause.set_item("kind", "receiver_antenna_pcv")?;
                    cause.set_item("label", label)?;
                }
                C::ReceiverAntennaGeometry => {
                    cause.set_item("kind", "receiver_antenna_geometry")?;
                }
            }
            result.set_item("cause", cause)?;
        }
        E::Ut1OutsideCoverage(reason) => {
            result.set_item("kind", "ut1_outside_coverage")?;
            result.set_item("reason", degrade_reason(*reason))?;
        }
    }
    Ok(result)
}

fn ils_detail<'py>(
    py: Python<'py>,
    error: &sidereon_core::ils::IlsError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::ils::IlsError as E;
    let result = detail(py, "IlsError", &error.to_string())?;
    match error {
        E::Singular => result.set_item("kind", "singular")?,
        E::NoCandidates(evaluated) => {
            result.set_item("kind", "no_candidates")?;
            result.set_item("evaluated", evaluated)?;
        }
        E::TooManyCandidates { evaluated, limit } => {
            result.set_item("kind", "too_many_candidates")?;
            result.set_item("evaluated", evaluated)?;
            result.set_item("limit", limit)?;
        }
        E::InvalidDimensions { n, rows } => {
            result.set_item("kind", "invalid_dimensions")?;
            result.set_item("n", n)?;
            result.set_item("rows", rows)?;
        }
        E::NonFinite => result.set_item("kind", "non_finite")?,
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::SearchLimitExceeded => result.set_item("kind", "search_limit_exceeded")?,
    }
    Ok(result)
}

fn ppp_fixed_detail<'py>(
    py: Python<'py>,
    error: &PppFixedSolveError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::precise_positioning::FixedSolveError as E;
    let result = detail(py, "PppFixedSolveError", &error.to_string())?;
    match error {
        E::Float(cause) => {
            result.set_item("kind", "float")?;
            result.set_item("cause", ppp_float_detail(py, cause)?)?;
        }
        E::Integer(cause) => {
            result.set_item("kind", "integer")?;
            result.set_item("cause", ils_detail(py, cause)?)?;
        }
        E::MissingWavelength(id) => {
            result.set_item("kind", "missing_wavelength")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::MissingOffset(id) => {
            result.set_item("kind", "missing_offset")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::MissingFixedAmbiguity(id) => {
            result.set_item("kind", "missing_fixed_ambiguity")?;
            result.set_item("ambiguity_id", id)?;
        }
    }
    Ok(result)
}

fn ppp_auto_init_detail<'py>(
    py: Python<'py>,
    error: &PppAutoInitError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::precise_positioning::PppAutoInitError as E;
    let result = detail(py, "PppAutoInitError", &error.to_string())?;
    match error {
        E::EmptyEpochs => result.set_item("kind", "empty_epochs")?,
        E::CodeSeedFailed {
            epoch_index,
            source,
        } => {
            result.set_item("kind", "code_seed_failed")?;
            result.set_item("epoch_index", epoch_index)?;
            result.set_item("cause", crate::spp_error_detail::spp_detail(py, source)?)?;
        }
        E::Float(cause) => {
            result.set_item("kind", "float")?;
            result.set_item("cause", ppp_float_detail(py, cause)?)?;
        }
        E::Fixed(cause) => {
            result.set_item("kind", "fixed")?;
            result.set_item("cause", ppp_fixed_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

pub(crate) fn ppp_float_error(py: Python<'_>, error: PppFloatSolveError) -> PyErr {
    let message = format!("PPP float solve failed: {error}");
    solve_error(py, message, ppp_float_detail(py, &error))
}

pub(crate) fn ppp_fixed_error(py: Python<'_>, error: PppFixedSolveError) -> PyErr {
    let message = format!("PPP fixed solve failed: {error}");
    solve_error(py, message, ppp_fixed_detail(py, &error))
}

pub(crate) fn ppp_auto_init_error(py: Python<'_>, error: PppAutoInitError) -> PyErr {
    let message = error.to_string();
    solve_error(py, message, ppp_auto_init_detail(py, &error))
}

fn dgnss_detail<'py>(py: Python<'py>, error: &DgnssError) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::dgnss::DgnssError as E;
    let result = detail(py, "DgnssError", &error.to_string())?;
    match error {
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::Spp(cause) => {
            result.set_item("kind", "spp")?;
            result.set_item("cause", crate::spp_error_detail::spp_detail(py, cause)?)?;
        }
        E::Ut1OutsideCoverage(reason) => {
            result.set_item("kind", "ut1_outside_coverage")?;
            result.set_item("reason", degrade_reason(*reason))?;
        }
    }
    Ok(result)
}

pub(crate) fn dgnss_error(py: Python<'_>, error: DgnssError) -> PyErr {
    let message = error.to_string();
    let detail = dgnss_detail(py, &error);
    let python_error = match error {
        DgnssError::Spp(spp) => crate::spp_error_detail::spp_error(py, spp, ""),
        DgnssError::Ut1OutsideCoverage(reason) => {
            crate::ut1_outside_coverage_err("DGNSS base satellite state", reason)
        }
        DgnssError::InvalidInput { .. } => PyValueError::new_err(message.clone()),
    };
    match detail {
        Ok(detail) => attach(py, python_error, detail),
        Err(error) => error,
    }
}

fn sbas_pl_detail<'py>(py: Python<'py>, error: &SbasPlError) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::sbas_pl::SbasPlError as E;
    let result = detail(py, "SbasPlError", &error.to_string())?;
    match error {
        E::InsufficientGeometry => result.set_item("kind", "insufficient_geometry")?,
        E::NumericalFailure => result.set_item("kind", "numerical_failure")?,
        E::InvalidErrorModel => result.set_item("kind", "invalid_error_model")?,
        E::Ut1OutsideCoverage(reason) => {
            result.set_item("kind", "ut1_outside_coverage")?;
            result.set_item("reason", degrade_reason(*reason))?;
        }
    }
    Ok(result)
}

pub(crate) fn sbas_pl_error(py: Python<'_>, error: SbasPlError) -> PyErr {
    let message = error.to_string();
    let detail = sbas_pl_detail(py, &error);
    let python_error = match error {
        SbasPlError::Ut1OutsideCoverage(reason) => {
            crate::ut1_outside_coverage_err("SBAS protection-level satellite position", reason)
        }
        _ => PyValueError::new_err(message),
    };
    match detail {
        Ok(detail) => attach(py, python_error, detail),
        Err(error) => error,
    }
}

// RTK mapper functions follow below; each uses a concrete core error type and
// recursively retains the stage that produced the failure.

fn rtk_input_kind(kind: sidereon_core::rtk_filter::RtkInputErrorKind) -> &'static str {
    use sidereon_core::rtk_filter::RtkInputErrorKind as K;
    match kind {
        K::NonFinite => "non_finite",
        K::NotPositive => "not_positive",
        K::Negative => "negative",
        K::OutOfRange => "out_of_range",
        K::Missing => "missing",
        K::FloatParse => "float_parse",
        K::IntParse => "int_parse",
        K::InvalidCivilDate => "invalid_civil_date",
        K::InvalidCivilTime => "invalid_civil_time",
    }
}

fn rtk_float_detail<'py>(
    py: Python<'py>,
    error: &RtkFloatSolveError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::FloatSolveError as E;
    let result = detail(py, "RtkFloatSolveError", &error.to_string())?;
    match error {
        E::MissingSystemReference(system) => {
            result.set_item("kind", "missing_system_reference")?;
            result.set_item("system", system)?;
        }
        E::MissingAmbiguityColumn(id) => {
            result.set_item("kind", "missing_ambiguity_column")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::InvalidInput { field, kind } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", rtk_input_kind(*kind))?;
        }
        E::SingularGeometry => result.set_item("kind", "singular_geometry")?,
        E::IncompleteResidualPair => result.set_item("kind", "incomplete_residual_pair")?,
        E::ReceiverAntenna(cause) => {
            result.set_item("kind", "receiver_antenna")?;
            result.set_item("cause", receiver_antenna_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn receiver_antenna_detail<'py>(
    py: Python<'py>,
    error: &sidereon_core::rtk_filter::ReceiverAntennaError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::ReceiverAntennaError as E;
    let result = detail(py, "ReceiverAntennaError", &error.to_string())?;
    match error {
        E::MissingPcv => result.set_item("kind", "missing_pcv")?,
        E::InvalidGeometry => result.set_item("kind", "invalid_geometry")?,
    }
    Ok(result)
}

fn rtk_fixed_detail<'py>(
    py: Python<'py>,
    error: &RtkFixedSolveError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::FixedSolveError as E;
    let result = detail(py, "RtkFixedSolveError", &error.to_string())?;
    match error {
        E::Float(cause) => {
            result.set_item("kind", "float")?;
            result.set_item("cause", rtk_float_detail(py, cause)?)?;
        }
        E::Ils(cause) => {
            result.set_item("kind", "ils")?;
            result.set_item("cause", ils_detail(py, cause)?)?;
        }
        E::MissingAmbiguity(id) => {
            result.set_item("kind", "missing_ambiguity")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::MissingWavelength(id) => {
            result.set_item("kind", "missing_wavelength")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::MissingOffset(id) => {
            result.set_item("kind", "missing_offset")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::InvalidCovarianceDimensions => {
            result.set_item("kind", "invalid_covariance_dimensions")?;
        }
        E::InvalidInput { field, kind } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", rtk_input_kind(*kind))?;
        }
        E::SingularGeometry => result.set_item("kind", "singular_geometry")?,
        E::IncompleteResidualPair => result.set_item("kind", "incomplete_residual_pair")?,
        E::ReceiverAntenna(cause) => {
            result.set_item("kind", "receiver_antenna")?;
            result.set_item("cause", receiver_antenna_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn residual_outlier_detail<'py>(
    py: Python<'py>,
    outlier: &sidereon_core::rtk_filter::ResidualValidationOutlier,
) -> PyResult<Bound<'py, PyDict>> {
    let result = detail(py, "ResidualValidationOutlier", "RTK residual outlier")?;
    result.set_item("epoch_index", outlier.epoch_index)?;
    result.set_item("satellite_id", &outlier.satellite_id)?;
    result.set_item("reference_satellite_id", &outlier.reference_satellite_id)?;
    result.set_item("ambiguity_id", &outlier.ambiguity_id)?;
    result.set_item(
        "residual_kind",
        match outlier.kind {
            sidereon_core::rtk_filter::ResidualComponentKind::Code => "code",
            sidereon_core::rtk_filter::ResidualComponentKind::Phase => "phase",
        },
    )?;
    set_float(&result, "residual_m", outlier.residual_m)?;
    set_float(&result, "sigma_m", outlier.sigma_m)?;
    set_float(&result, "normalized_residual", outlier.normalized_residual)?;
    set_float(&result, "threshold_sigma", outlier.threshold_sigma)?;
    Ok(result)
}

fn validated_fixed_detail<'py>(
    py: Python<'py>,
    error: &ValidatedFixedSolveError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::ValidatedFixedSolveError as E;
    let result = detail(py, "ValidatedFixedSolveError", &error.to_string())?;
    match error {
        E::Fixed(cause) => {
            result.set_item("kind", "fixed")?;
            result.set_item("cause", rtk_fixed_detail(py, cause)?)?;
        }
        E::ResidualValidationFailed {
            outlier,
            exclusions,
        } => {
            result.set_item("kind", "residual_validation_failed")?;
            result.set_item("outlier", residual_outlier_detail(py, outlier)?)?;
            result.set_item(
                "exclusions",
                exclusions
                    .iter()
                    .map(|row| residual_outlier_detail(py, row))
                    .collect::<PyResult<Vec<_>>>()?,
            )?;
        }
        E::DuplicateAmbiguityId {
            ambiguity_id,
            first_satellite_id,
            second_satellite_id,
        } => {
            result.set_item("kind", "duplicate_ambiguity_id")?;
            result.set_item("ambiguity_id", ambiguity_id)?;
            result.set_item("first_satellite_id", first_satellite_id)?;
            result.set_item("second_satellite_id", second_satellite_id)?;
        }
        E::Underdetermined {
            row_count,
            unknown_count,
        } => {
            result.set_item("kind", "underdetermined")?;
            result.set_item("row_count", row_count)?;
            result.set_item("unknown_count", unknown_count)?;
        }
    }
    Ok(result)
}

fn double_difference_detail<'py>(
    py: Python<'py>,
    error: &DoubleDifferenceError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk::DoubleDifferenceError as E;
    let result = family_detail(py, "DoubleDifferenceError")?;
    match error {
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::DuplicateObservation(id) => {
            result.set_item("kind", "duplicate_observation")?;
            result.set_item("satellite_id", id)?;
        }
        E::TooFewCommonSatellites { count, minimum } => {
            result.set_item("kind", "too_few_common_satellites")?;
            result.set_item("count", count)?;
            result.set_item("minimum", minimum)?;
        }
        E::NoCommonReferenceSatellite(system) => {
            result.set_item("kind", "no_common_reference_satellite")?;
            result.set_item("system", system)?;
        }
        E::MissingSatellitePosition(id) => {
            result.set_item("kind", "missing_satellite_position")?;
            result.set_item("satellite_id", id)?;
        }
        E::ReferenceSatelliteMissing(id) => {
            result.set_item("kind", "reference_satellite_missing")?;
            result.set_item("satellite_id", id)?;
        }
        E::ReferenceSatelliteSingleSystem(id) => {
            result.set_item("kind", "reference_satellite_single_system")?;
            result.set_item("satellite_id", id)?;
        }
        E::ReferenceSatelliteMissingSystem(system) => {
            result.set_item("kind", "reference_satellite_missing_system")?;
            result.set_item("system", system)?;
        }
        E::InvalidReferenceOption => result.set_item("kind", "invalid_reference_option")?,
    }
    Ok(result)
}

fn filter_state_validation_detail<'py>(
    py: Python<'py>,
    error: &FilterStateValidationError,
) -> PyResult<Bound<'py, PyDict>> {
    use FilterStateValidationKind as K;
    let result = detail(py, "FilterStateValidationError", &error.to_string())?;
    result.set_item("field", error.field)?;
    match &error.kind {
        K::Length { expected, actual } => {
            result.set_item("kind", "length")?;
            result.set_item("expected", expected)?;
            result.set_item("actual", actual)?;
        }
        K::NonFinite => result.set_item("kind", "non_finite")?,
        K::NotPositive => result.set_item("kind", "not_positive")?,
        K::NotSymmetric => result.set_item("kind", "not_symmetric")?,
        K::NotPositiveSemidefinite => result.set_item("kind", "not_positive_semidefinite")?,
        K::DimensionOverflow => result.set_item("kind", "dimension_overflow")?,
    }
    Ok(result)
}

fn invalid_state_kind_detail<'py>(
    py: Python<'py>,
    kind: &InvalidStateKind,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    match kind {
        InvalidStateKind::Length { expected, actual } => {
            result.set_item("kind", "length")?;
            result.set_item("expected", expected)?;
            result.set_item("actual", actual)?;
        }
        InvalidStateKind::DimensionOverflow => result.set_item("kind", "dimension_overflow")?,
        InvalidStateKind::NonFinite => result.set_item("kind", "non_finite")?,
        InvalidStateKind::NotPositive => result.set_item("kind", "not_positive")?,
        InvalidStateKind::NotSymmetric => result.set_item("kind", "not_symmetric")?,
        InvalidStateKind::NotPositiveSemidefinite => {
            result.set_item("kind", "not_positive_semidefinite")?;
        }
    }
    Ok(result)
}

fn update_detail<'py>(py: Python<'py>, error: &UpdateError) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::UpdateError as E;
    let result = detail(py, "UpdateError", &error.to_string())?;
    match error {
        E::InvalidState { field, kind } => {
            result.set_item("kind", "invalid_state")?;
            result.set_item("field", field)?;
            result.set_item("cause", invalid_state_kind_detail(py, kind)?)?;
        }
        E::ReferenceChanged {
            system,
            expected,
            actual,
        } => {
            result.set_item("kind", "reference_changed")?;
            result.set_item("system", system)?;
            result.set_item("expected", expected)?;
            result.set_item("actual", actual)?;
        }
        E::UnknownReferenceSystem(system) => {
            result.set_item("kind", "unknown_reference_system")?;
            result.set_item("system", system)?;
        }
        E::MissingSystemReference(system) => {
            result.set_item("kind", "missing_system_reference")?;
            result.set_item("system", system)?;
        }
        E::MissingAmbiguityColumn(id) => {
            result.set_item("kind", "missing_ambiguity_column")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::MissingWavelength(id) => {
            result.set_item("kind", "missing_wavelength")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::MissingOffset(id) => {
            result.set_item("kind", "missing_offset")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::InvalidInput { field, kind } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", rtk_input_kind(*kind))?;
        }
        E::SingularGeometry => result.set_item("kind", "singular_geometry")?,
        E::ReceiverAntenna(cause) => {
            result.set_item("kind", "receiver_antenna")?;
            result.set_item("cause", receiver_antenna_detail(py, cause)?)?;
        }
        E::Ils(cause) => {
            result.set_item("kind", "ils")?;
            result.set_item("cause", ils_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn cycle_slip_prep_detail<'py>(
    py: Python<'py>,
    error: &CycleSlipPrepError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk::CycleSlipPrepError as E;
    let result = family_detail(py, "CycleSlipPrepError")?;
    match error {
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::CycleSlipDetected {
            receiver,
            satellite_id,
            epoch_index,
            reasons,
        } => {
            result.set_item("kind", "cycle_slip_detected")?;
            result.set_item(
                "receiver",
                match receiver {
                    sidereon_core::rtk::CycleSlipReceiver::Base => "base",
                    sidereon_core::rtk::CycleSlipReceiver::Rover => "rover",
                },
            )?;
            result.set_item("satellite_id", satellite_id)?;
            result.set_item("epoch_index", epoch_index)?;
            result.set_item(
                "reasons",
                reasons
                    .iter()
                    .map(|reason| match reason {
                        sidereon_core::carrier_phase::SlipReason::Lli => "lli",
                        sidereon_core::carrier_phase::SlipReason::DataGap => "data_gap",
                        sidereon_core::carrier_phase::SlipReason::GeometryFree => "geometry_free",
                        sidereon_core::carrier_phase::SlipReason::MelbourneWubbena => {
                            "melbourne_wubbena"
                        }
                    })
                    .collect::<Vec<_>>(),
            )?;
        }
    }
    Ok(result)
}

fn code_smoothing_detail<'py>(
    py: Python<'py>,
    error: &CodeSmoothingError,
) -> PyResult<Bound<'py, PyDict>> {
    let result = family_detail(py, "CodeSmoothingError")?;
    match error {
        CodeSmoothingError::InvalidWindowCap => result.set_item("kind", "invalid_window_cap")?,
    }
    Ok(result)
}

fn rtk_arc_detail<'py>(py: Python<'py>, error: &RtkArcError) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::RtkArcError as E;
    let result = detail(py, "RtkArcError", &error.to_string())?;
    match error {
        E::EmptyEpochs => result.set_item("kind", "empty_epochs")?,
        E::TooFewSatellites { count, minimum } => {
            result.set_item("kind", "too_few_satellites")?;
            result.set_item("count", count)?;
            result.set_item("minimum", minimum)?;
        }
        E::Reference(cause) => {
            result.set_item("kind", "reference")?;
            result.set_item("cause", double_difference_detail(py, cause)?)?;
        }
        E::FilterState(cause) => {
            result.set_item("kind", "filter_state")?;
            result.set_item("cause", filter_state_validation_detail(py, cause)?)?;
        }
        E::Update {
            epoch_index,
            source,
        } => {
            result.set_item("kind", "update")?;
            result.set_item("epoch_index", epoch_index)?;
            result.set_item("cause", update_detail(py, source)?)?;
        }
        E::InvalidEpochTime { epoch_index } => {
            result.set_item("kind", "invalid_epoch_time")?;
            result.set_item("epoch_index", epoch_index)?;
        }
        E::MissingPosition {
            epoch_index,
            satellite_id,
        } => {
            result.set_item("kind", "missing_position")?;
            result.set_item("epoch_index", epoch_index)?;
            result.set_item("satellite_id", satellite_id)?;
        }
        E::CycleSlipPrep(cause) => {
            result.set_item("kind", "cycle_slip_prep")?;
            result.set_item("cause", cycle_slip_prep_detail(py, cause)?)?;
        }
        E::CodeSmoothing(cause) => {
            result.set_item("kind", "code_smoothing")?;
            result.set_item("cause", code_smoothing_detail(py, cause)?)?;
        }
        E::ElevationMask(cause) => {
            result.set_item("kind", "elevation_mask")?;
            result.set_item("cause", double_difference_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn rtk_static_arc_detail<'py>(
    py: Python<'py>,
    error: &RtkStaticArcError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::RtkStaticArcError as E;
    let result = detail(py, "RtkStaticArcError", &error.to_string())?;
    match error {
        E::Arc(cause) => {
            result.set_item("kind", "arc")?;
            result.set_item("cause", rtk_arc_detail(py, cause)?)?;
        }
        E::Float(cause) => {
            result.set_item("kind", "float")?;
            result.set_item("cause", rtk_float_detail(py, cause)?)?;
        }
        E::Fixed(cause) => {
            result.set_item("kind", "fixed")?;
            result.set_item("cause", validated_fixed_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn moving_baseline_detail<'py>(
    py: Python<'py>,
    error: &MovingBaselineError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::MovingBaselineError as E;
    let result = detail(py, "MovingBaselineError", &error.to_string())?;
    match error {
        E::Float(cause) => {
            result.set_item("kind", "float")?;
            result.set_item("cause", rtk_float_detail(py, cause)?)?;
        }
        E::Fixed(cause) => {
            result.set_item("kind", "fixed")?;
            result.set_item("cause", rtk_fixed_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn moving_sequence_detail<'py>(
    py: Python<'py>,
    error: &MovingBaselineSequenceError,
) -> PyResult<Bound<'py, PyDict>> {
    let result = detail(py, "MovingBaselineSequenceError", &error.to_string())?;
    result.set_item("kind", "epoch_failure")?;
    result.set_item("epoch_index", error.epoch_index)?;
    result.set_item("cause", moving_baseline_detail(py, &error.error)?)?;
    Ok(result)
}

fn rinex_arc_detail<'py>(
    py: Python<'py>,
    error: &RtkRinexArcError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::RtkRinexArcError as E;
    let result = detail(py, "RtkRinexArcError", &error.to_string())?;
    match error {
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::Observation(cause) => {
            result.set_item("kind", "observation")?;
            result.set_item(
                "cause",
                crate::core_error_detail::core_error_detail(py, cause)?,
            )?;
        }
        E::Ephemeris {
            satellite_id,
            epoch_j2000_s,
            reason,
        } => {
            result.set_item("kind", "ephemeris")?;
            result.set_item("satellite_id", satellite_id)?;
            set_float(&result, "epoch_j2000_s", *epoch_j2000_s)?;
            result.set_item("reason", reason)?;
        }
        E::NoSignalPairs => result.set_item("kind", "no_signal_pairs")?,
        E::NoUsableEpochs => result.set_item("kind", "no_usable_epochs")?,
        E::Ut1OutsideCoverage(reason) => {
            result.set_item("kind", "ut1_outside_coverage")?;
            result.set_item("reason", degrade_reason(*reason))?;
        }
    }
    Ok(result)
}

fn carrier_phase_detail<'py>(
    py: Python<'py>,
    error: &sidereon_core::carrier_phase::CarrierPhaseError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::carrier_phase::CarrierPhaseError as E;
    let result = detail(py, "CarrierPhaseError", &error.to_string())?;
    match error {
        E::EqualFrequencies => result.set_item("kind", "equal_frequencies")?,
        E::InvalidFrequency => result.set_item("kind", "invalid_frequency")?,
        E::InvalidObservation => result.set_item("kind", "invalid_observation")?,
        E::InvalidThreshold => result.set_item("kind", "invalid_threshold")?,
    }
    Ok(result)
}

fn wide_lane_detail<'py>(py: Python<'py>, error: &WideLaneError) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk::WideLaneError as E;
    let result = family_detail(py, "WideLaneError")?;
    match error {
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::ReferenceSatelliteMissing(id) => {
            result.set_item("kind", "reference_satellite_missing")?;
            result.set_item("satellite_id", id)?;
        }
        E::WideLaneFailed {
            satellite_id,
            reason,
        } => {
            result.set_item("kind", "wide_lane_failed")?;
            result.set_item("satellite_id", satellite_id)?;
            result.set_item("cause", carrier_phase_detail(py, reason)?)?;
        }
        E::TooFewWideLaneEpochs {
            ambiguity_id,
            count,
            minimum,
        } => {
            result.set_item("kind", "too_few_wide_lane_epochs")?;
            result.set_item("ambiguity_id", ambiguity_id)?;
            result.set_item("count", count)?;
            result.set_item("minimum", minimum)?;
        }
        E::WideLaneNotInteger {
            ambiguity_id,
            mean_cycles,
            fixed_cycles,
        } => {
            result.set_item("kind", "wide_lane_not_integer")?;
            result.set_item("ambiguity_id", ambiguity_id)?;
            set_float(&result, "mean_cycles", *mean_cycles)?;
            result.set_item("fixed_cycles", fixed_cycles)?;
        }
    }
    Ok(result)
}

fn ionosphere_free_detail<'py>(
    py: Python<'py>,
    error: &sidereon_core::combinations::IonosphereFreeError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::combinations::IonosphereFreeError as E;
    let result = detail(py, "IonosphereFreeError", &error.to_string())?;
    match error {
        E::UnknownSystem(system) => {
            result.set_item("kind", "unknown_system")?;
            result.set_item("system", system.to_string())?;
        }
        E::UnknownBand { system, band } => {
            result.set_item("kind", "unknown_band")?;
            result.set_item("system", system.to_string())?;
            result.set_item("band", band)?;
        }
        E::EqualFrequencies => result.set_item("kind", "equal_frequencies")?,
        E::InvalidFrequency => result.set_item("kind", "invalid_frequency")?,
        E::InvalidObservation => result.set_item("kind", "invalid_observation")?,
    }
    Ok(result)
}

fn ionosphere_free_baseline_detail<'py>(
    py: Python<'py>,
    error: &IonosphereFreeBaselineError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk::IonosphereFreeBaselineError as E;
    let result = family_detail(py, "IonosphereFreeBaselineError")?;
    match error {
        E::InvalidInput { field, reason } => {
            result.set_item("kind", "invalid_input")?;
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::NoEpochs => result.set_item("kind", "no_epochs")?,
        E::InconsistentFrequencies(id) => {
            result.set_item("kind", "inconsistent_frequencies")?;
            result.set_item("ambiguity_id", id)?;
        }
        E::NarrowLaneFailed(cause) => {
            result.set_item("kind", "narrow_lane_failed")?;
            result.set_item("cause", ionosphere_free_detail(py, cause)?)?;
        }
        E::IonosphereFreeFailed {
            satellite_id,
            reason,
        } => {
            result.set_item("kind", "ionosphere_free_failed")?;
            result.set_item("satellite_id", satellite_id)?;
            result.set_item("cause", ionosphere_free_detail(py, reason)?)?;
        }
    }
    Ok(result)
}

fn wide_lane_arc_detail<'py>(
    py: Python<'py>,
    error: &RtkWideLaneArcError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::RtkWideLaneArcError as E;
    let result = detail(py, "RtkWideLaneArcError", &error.to_string())?;
    match error {
        E::EmptyEpochs => result.set_item("kind", "empty_epochs")?,
        E::Reference(cause) => {
            result.set_item("kind", "reference")?;
            result.set_item("cause", double_difference_detail(py, cause)?)?;
        }
        E::CycleSlipPrep(cause) => {
            result.set_item("kind", "cycle_slip_prep")?;
            result.set_item("cause", cycle_slip_prep_detail(py, cause)?)?;
        }
        E::WideLane(cause) => {
            result.set_item("kind", "wide_lane")?;
            result.set_item("cause", wide_lane_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn ionosphere_free_arc_detail<'py>(
    py: Python<'py>,
    error: &RtkIonosphereFreeArcError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::RtkIonosphereFreeArcError as E;
    let result = detail(py, "RtkIonosphereFreeArcError", &error.to_string())?;
    match error {
        E::EmptyEpochs => result.set_item("kind", "empty_epochs")?,
        E::Reference(cause) => {
            result.set_item("kind", "reference")?;
            result.set_item("cause", double_difference_detail(py, cause)?)?;
        }
        E::IonosphereFree(cause) => {
            result.set_item("kind", "ionosphere_free")?;
            result.set_item("cause", ionosphere_free_baseline_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn wide_lane_fixed_arc_detail<'py>(
    py: Python<'py>,
    error: &RtkWideLaneFixedArcError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::rtk_filter::RtkWideLaneFixedArcError as E;
    let result = detail(py, "RtkWideLaneFixedArcError", &error.to_string())?;
    match error {
        E::UnsupportedMultiGnss => result.set_item("kind", "unsupported_multi_gnss")?,
        E::WideLane(cause) => {
            result.set_item("kind", "wide_lane")?;
            result.set_item("cause", wide_lane_arc_detail(py, cause)?)?;
        }
        E::IonosphereFree(cause) => {
            result.set_item("kind", "ionosphere_free")?;
            result.set_item("cause", ionosphere_free_arc_detail(py, cause)?)?;
        }
        E::Static(cause) => {
            result.set_item("kind", "static")?;
            result.set_item("cause", rtk_static_arc_detail(py, cause)?)?;
        }
        E::Sequential(cause) => {
            result.set_item("kind", "sequential")?;
            result.set_item("cause", rtk_arc_detail(py, cause)?)?;
        }
    }
    Ok(result)
}

fn solve_error_for_rtk<'py>(
    py: Python<'py>,
    message: String,
    payload: PyResult<Bound<'py, PyDict>>,
) -> PyErr {
    solve_error(py, message, payload)
}

pub(crate) fn rtk_float_error(py: Python<'_>, error: RtkFloatSolveError) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, rtk_float_detail(py, &error))
}

pub(crate) fn rtk_fixed_error(py: Python<'_>, error: ValidatedFixedSolveError) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, validated_fixed_detail(py, &error))
}

pub(crate) fn moving_sequence_error(py: Python<'_>, error: MovingBaselineSequenceError) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, moving_sequence_detail(py, &error))
}

pub(crate) fn rtk_arc_error(py: Python<'_>, error: RtkArcError) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, rtk_arc_detail(py, &error))
}

pub(crate) fn rtk_static_arc_error(py: Python<'_>, error: RtkStaticArcError) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, rtk_static_arc_detail(py, &error))
}

pub(crate) fn rtk_rinex_arc_error(py: Python<'_>, error: RtkRinexArcError) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, rinex_arc_detail(py, &error))
}

pub(crate) fn rtk_wide_lane_fixed_arc_error(
    py: Python<'_>,
    error: RtkWideLaneFixedArcError,
) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, wide_lane_fixed_arc_detail(py, &error))
}

pub(crate) fn rtk_wide_lane_arc_error(py: Python<'_>, error: RtkWideLaneArcError) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, wide_lane_arc_detail(py, &error))
}

pub(crate) fn rtk_ionosphere_free_arc_error(
    py: Python<'_>,
    error: RtkIonosphereFreeArcError,
) -> PyErr {
    let message = error.to_string();
    solve_error_for_rtk(py, message, ionosphere_free_arc_detail(py, &error))
}
