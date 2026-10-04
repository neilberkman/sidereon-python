//! Owned error details for SPP, static positioning, and RINEX SPP bindings.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use sidereon_core::positioning::{RinexSppError, SolvePolicyError, SppError, SppInputErrorKind};
use sidereon_core::quality::SolutionValidationError;
use sidereon_core::static_positioning::StaticSolveError;

fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}

fn set_float(dict: &Bound<'_, PyDict>, key: &str, value: f64) -> PyResult<()> {
    dict.set_item(key, value)?;
    dict.set_item(format!("{key}_bits"), bits(value))
}

fn input_kind(kind: SppInputErrorKind) -> &'static str {
    match kind {
        SppInputErrorKind::NonFinite => "non_finite",
        SppInputErrorKind::NotPositive => "not_positive",
        SppInputErrorKind::Negative => "negative",
        SppInputErrorKind::OutOfRange => "out_of_range",
        SppInputErrorKind::Missing => "missing",
        SppInputErrorKind::FloatParse => "float_parse",
        SppInputErrorKind::IntParse => "int_parse",
        SppInputErrorKind::InvalidCivilDate => "invalid_civil_date",
        SppInputErrorKind::InvalidCivilTime => "invalid_civil_time",
    }
}

fn least_squares_detail<'py>(
    py: Python<'py>,
    error: &sidereon_core::astro::math::least_squares::SolveError,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon_core::astro::math::least_squares::SolveError as E;
    let detail = PyDict::new(py);
    detail.set_item("family", "LeastSquaresSolveError")?;
    detail.set_item("message", error.to_string())?;
    match error {
        E::SingularJacobian => detail.set_item("kind", "singular_jacobian")?,
        E::InvalidInput { field, reason } => {
            detail.set_item("kind", "invalid_input")?;
            detail.set_item("field", field)?;
            detail.set_item("reason", reason)?;
        }
    }
    Ok(detail)
}

pub(crate) fn spp_detail<'py>(py: Python<'py>, error: &SppError) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "SppError")?;
    detail.set_item("message", error.to_string())?;
    match error {
        SppError::InvalidInput { field, kind } => {
            detail.set_item("kind", "invalid_input")?;
            detail.set_item("field", field)?;
            detail.set_item("reason", input_kind(*kind))?;
        }
        SppError::TooFewSatellites { used, required } => {
            detail.set_item("kind", "too_few_satellites")?;
            detail.set_item("used", used)?;
            detail.set_item("required", required)?;
        }
        SppError::Singular(cause) => {
            detail.set_item("kind", "singular")?;
            detail.set_item("cause", least_squares_detail(py, cause)?)?;
        }
        SppError::DuplicateObservation { satellite } => {
            detail.set_item("kind", "duplicate_observation")?;
            detail.set_item("satellite_id", satellite.to_string())?;
        }
        SppError::EphemerisLost { satellite } => {
            detail.set_item("kind", "ephemeris_lost")?;
            detail.set_item("satellite_id", satellite.to_string())?;
        }
        SppError::SelectionUnsettled { passes } => {
            detail.set_item("kind", "selection_unsettled")?;
            detail.set_item("passes", passes)?;
        }
        SppError::Ut1OutsideCoverage(reason) => {
            detail.set_item("kind", "ut1_outside_coverage")?;
            detail.set_item("reason", crate::degrade_reason_label(*reason))?;
        }
    }
    Ok(detail)
}

fn validation_detail<'py>(
    py: Python<'py>,
    error: &SolutionValidationError,
) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "SolutionValidationError")?;
    detail.set_item("message", error.to_string())?;
    match error {
        SolutionValidationError::InvalidOptions { field, reason } => {
            detail.set_item("kind", "invalid_options")?;
            detail.set_item("field", field)?;
            detail.set_item("reason", reason)?;
        }
        SolutionValidationError::DegenerateGeometryRankDeficient => {
            detail.set_item("kind", "degenerate_geometry_rank_deficient")?;
        }
        SolutionValidationError::DegenerateGeometryPdop(pdop) => {
            detail.set_item("kind", "degenerate_geometry_pdop")?;
            set_float(&detail, "pdop", *pdop)?;
        }
        SolutionValidationError::ImplausiblePosition(radius_m) => {
            detail.set_item("kind", "implausible_position")?;
            set_float(&detail, "radius_m", *radius_m)?;
        }
        SolutionValidationError::InvalidResiduals => {
            detail.set_item("kind", "invalid_residuals")?;
        }
        SolutionValidationError::NoConvergence(rms_m) => {
            detail.set_item("kind", "no_convergence")?;
            set_float(&detail, "residual_rms_m", *rms_m)?;
        }
    }
    Ok(detail)
}

pub(crate) fn policy_detail<'py>(
    py: Python<'py>,
    error: &SolvePolicyError,
    epoch_index: Option<usize>,
    prefix: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "SolvePolicyError")?;
    detail.set_item("message", format!("{prefix}{error}"))?;
    if let Some(index) = epoch_index {
        detail.set_item("epoch_index", index)?;
    }
    match error {
        SolvePolicyError::Solve(cause) => {
            detail.set_item("kind", "solve")?;
            detail.set_item("cause", spp_detail(py, cause)?)?;
        }
        SolvePolicyError::Validation(cause) => {
            detail.set_item("kind", "validation")?;
            detail.set_item("cause", validation_detail(py, cause)?)?;
        }
        SolvePolicyError::NoCoarseSolution => {
            detail.set_item("kind", "no_coarse_solution")?;
        }
    }
    Ok(detail)
}

pub(crate) fn static_detail<'py>(
    py: Python<'py>,
    error: &StaticSolveError,
) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "StaticSolveError")?;
    detail.set_item("message", error.to_string())?;
    match error {
        StaticSolveError::EmptyEpochs => detail.set_item("kind", "empty_epochs")?,
        StaticSolveError::InvalidInput { field, kind } => {
            detail.set_item("kind", "invalid_input")?;
            detail.set_item("field", field)?;
            detail.set_item("reason", input_kind(*kind))?;
        }
        StaticSolveError::EpochInput {
            epoch_index,
            source,
        } => {
            detail.set_item("kind", "epoch_input")?;
            detail.set_item("epoch_index", epoch_index)?;
            detail.set_item("cause", spp_detail(py, source)?)?;
        }
        StaticSolveError::DuplicateObservation {
            epoch_index,
            satellite,
        } => {
            detail.set_item("kind", "duplicate_observation")?;
            detail.set_item("epoch_index", epoch_index)?;
            detail.set_item("satellite_id", satellite.to_string())?;
        }
        StaticSolveError::TooFewMeasurements { used, required } => {
            detail.set_item("kind", "too_few_measurements")?;
            detail.set_item("used", used)?;
            detail.set_item("required", required)?;
        }
        StaticSolveError::EphemerisLost {
            epoch_index,
            satellite,
        } => {
            detail.set_item("kind", "ephemeris_lost")?;
            detail.set_item("epoch_index", epoch_index)?;
            detail.set_item("satellite_id", satellite.to_string())?;
        }
        StaticSolveError::Singular(cause) => {
            detail.set_item("kind", "singular")?;
            detail.set_item("cause", least_squares_detail(py, cause)?)?;
        }
        StaticSolveError::SelectionUnsettled { passes } => {
            detail.set_item("kind", "selection_unsettled")?;
            detail.set_item("passes", passes)?;
        }
        StaticSolveError::Ut1OutsideCoverage(reason) => {
            detail.set_item("kind", "ut1_outside_coverage")?;
            detail.set_item("reason", crate::degrade_reason_label(*reason))?;
        }
    }
    Ok(detail)
}

pub(crate) fn rinex_spp_detail<'py>(
    py: Python<'py>,
    error: &RinexSppError,
) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "RinexSppError")?;
    detail.set_item("message", error.to_string())?;
    match error {
        RinexSppError::Observation(cause) => {
            detail.set_item("kind", "observation")?;
            detail.set_item(
                "cause",
                crate::core_error_detail::core_error_detail(py, cause)?,
            )?;
        }
        RinexSppError::MissingApproxPosition => {
            detail.set_item("kind", "missing_approx_position")?;
        }
        _ => detail.set_item("kind", "unknown")?,
    }
    Ok(detail)
}

fn attach_detail<'py>(py: Python<'py>, error: PyErr, detail: Bound<'py, PyDict>) -> PyErr {
    if let Err(error) = error.value(py).setattr("detail", detail) {
        return error;
    }
    error
}

pub(crate) fn spp_error(py: Python<'_>, error: SppError, prefix: &str) -> PyErr {
    let message = format!("{prefix}{error}");
    let detail = match spp_detail(py, &error) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    let python_error = match &error {
        SppError::SelectionUnsettled { passes } => {
            let error_type = match crate::selection_unsettled_error_type(py) {
                Ok(error_type) => error_type,
                Err(error) => return error,
            };
            let error = PyErr::from_type(error_type, message);
            if let Err(error) = error.value(py).setattr("passes", *passes) {
                return error;
            }
            error
        }
        _ => crate::SolveError::new_err(message),
    };
    attach_detail(py, python_error, detail)
}

pub(crate) fn facade_spp_error(
    py: Python<'_>,
    error: sidereon::Error,
    prefix: &str,
    epoch_index: Option<usize>,
) -> PyErr {
    match error {
        sidereon::Error::Spp(error) => policy_error(py, error, prefix, epoch_index),
        other => crate::SolveError::new_err(format!("{prefix}{other}")),
    }
}

pub(crate) fn policy_error(
    py: Python<'_>,
    error: SolvePolicyError,
    prefix: &str,
    epoch_index: Option<usize>,
) -> PyErr {
    let message = format!("{prefix}{error}");
    let detail = match policy_detail(py, &error, epoch_index, prefix) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    let python_error = match &error {
        SolvePolicyError::Solve(SppError::SelectionUnsettled { passes }) => {
            let error_type = match crate::selection_unsettled_error_type(py) {
                Ok(error_type) => error_type,
                Err(error) => return error,
            };
            let error = PyErr::from_type(error_type, message);
            if let Err(error) = error.value(py).setattr("passes", *passes) {
                return error;
            }
            error
        }
        _ => crate::SolveError::new_err(message),
    };
    attach_detail(py, python_error, detail)
}

pub(crate) fn static_error(py: Python<'_>, error: StaticSolveError) -> PyErr {
    let detail = match static_detail(py, &error) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    let python_error = match &error {
        StaticSolveError::SelectionUnsettled { passes } => {
            let error_type = match crate::selection_unsettled_error_type(py) {
                Ok(error_type) => error_type,
                Err(error) => return error,
            };
            let error = PyErr::from_type(error_type, error.to_string());
            if let Err(error) = error.value(py).setattr("passes", *passes) {
                return error;
            }
            error
        }
        _ => crate::SolveError::new_err(error.to_string()),
    };
    attach_detail(py, python_error, detail)
}

pub(crate) fn rinex_spp_error(py: Python<'_>, error: RinexSppError) -> PyErr {
    let detail = match rinex_spp_detail(py, &error) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    attach_detail(py, PyValueError::new_err(error.to_string()), detail)
}
