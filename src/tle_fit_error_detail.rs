//! Structured details for inverse SGP4 fitting errors.
//!
//! This mapper consumes the concrete fitting error and keeps every nested
//! solver, SGP4, TLE-encoding, and best-effort result field.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use sidereon::sgp4::{TleFit, TleFitError};
use sidereon::tle::TleError;
use trust_region_least_squares::trf::{BackendError, TrfError};

fn detail<'py>(
    py: Python<'py>,
    family: &str,
    kind: &str,
    message: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("family", family)?;
    result.set_item("kind", kind)?;
    result.set_item("message", message)?;
    Ok(result)
}

fn float_field(dict: &Bound<'_, PyDict>, field: &str, value: f64) -> PyResult<()> {
    dict.set_item(field, value)?;
    dict.set_item(format!("{field}_bits"), format!("{:016x}", value.to_bits()))
}

fn sgp4_input_kind(kind: sidereon::sgp4::Sgp4InputErrorKind) -> &'static str {
    use sidereon::sgp4::Sgp4InputErrorKind as K;
    match kind {
        K::Missing => "missing",
        K::NonFinite => "non_finite",
        K::NotPositive => "not_positive",
        K::Negative => "negative",
        K::OutOfRange => "out_of_range",
        K::FloatParse => "float_parse",
        K::IntParse => "int_parse",
        K::InvalidCivilDate => "invalid_civil_date",
        K::InvalidCivilTime => "invalid_civil_time",
    }
}

fn sgp4_detail<'py>(
    py: Python<'py>,
    error: &sidereon::sgp4::Error,
) -> PyResult<Bound<'py, PyDict>> {
    use sidereon::sgp4::Error as E;
    let (kind, message) = match error {
        E::InvalidInput { .. } => ("invalid_input", error.to_string()),
        E::NonFiniteOutput { .. } => ("non_finite_output", error.to_string()),
        E::InvalidTle(_) => ("invalid_tle", error.to_string()),
        E::Sgp4 { .. } => ("sgp4", error.to_string()),
        E::ResonanceStepBudget { .. } => ("resonance_step_budget", error.to_string()),
    };
    let result = detail(py, "Sgp4Error", kind, &message)?;
    match error {
        E::InvalidInput { field, kind } => {
            result.set_item("field", field)?;
            result.set_item("reason", sgp4_input_kind(*kind))?;
        }
        E::NonFiniteOutput { field } => result.set_item("field", field)?,
        E::InvalidTle(reason) => result.set_item("reason", reason)?,
        E::Sgp4 { code } => result.set_item("code", code)?,
        E::ResonanceStepBudget { budget } => result.set_item("budget", budget)?,
    }
    Ok(result)
}

fn tle_detail<'py>(py: Python<'py>, error: &TleError) -> PyResult<Bound<'py, PyDict>> {
    use TleError as E;
    let (kind, message) = match error {
        E::NonAscii => ("non_ascii", error.to_string()),
        E::Format => ("format", error.to_string()),
        E::SatelliteMismatch => ("satellite_mismatch", error.to_string()),
        E::InvalidCatalogNumber { .. } => ("invalid_catalog_number", error.to_string()),
        E::CatalogNumberOutOfRange { .. } => ("catalog_number_out_of_range", error.to_string()),
        E::InvalidField { .. } => ("invalid_field", error.to_string()),
        E::Field(_) => ("field", error.to_string()),
        E::ChecksumMismatch { .. } => ("checksum_mismatch", error.to_string()),
        E::ChecksumNotDigit { .. } => ("checksum_not_digit", error.to_string()),
    };
    let result = detail(py, "TleError", kind, &message)?;
    match error {
        E::NonAscii | E::Format | E::SatelliteMismatch => {}
        E::InvalidCatalogNumber { value, reason } => {
            result.set_item("value", value)?;
            result.set_item("reason", reason)?;
        }
        E::CatalogNumberOutOfRange { catalog_number } => {
            result.set_item("catalog_number", catalog_number)?;
        }
        E::InvalidField { field, reason } => {
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::Field(field_message) => result.set_item("field_message", field_message)?,
        E::ChecksumMismatch {
            line_label,
            expected,
            computed,
        } => {
            result.set_item("line_label", line_label)?;
            result.set_item("expected", expected)?;
            result.set_item("computed", computed)?;
        }
        E::ChecksumNotDigit {
            line_label,
            found,
            computed,
        } => {
            result.set_item("line_label", line_label)?;
            result.set_item("found", found.to_string())?;
            result.set_item("computed", computed)?;
        }
    }
    Ok(result)
}

fn backend_detail<'py>(py: Python<'py>, error: &BackendError) -> PyResult<Bound<'py, PyDict>> {
    let (kind, message) = match error {
        BackendError::Failed(_) => ("failed", error.to_string()),
        BackendError::BadDimensions { .. } => ("bad_dimensions", error.to_string()),
    };
    let result = detail(py, "BackendError", kind, &message)?;
    match error {
        BackendError::Failed(message) => result.set_item("reason", message)?,
        BackendError::BadDimensions {
            expected_m,
            expected_n,
            got,
        } => {
            result.set_item("expected_m", expected_m)?;
            result.set_item("expected_n", expected_n)?;
            result.set_item("got", got)?;
        }
    }
    Ok(result)
}

fn trf_detail<'py>(py: Python<'py>, error: &TrfError) -> PyResult<Bound<'py, PyDict>> {
    use TrfError as E;
    let (kind, message) = match error {
        E::EmptyResidual => ("empty_residual", error.to_string()),
        E::EmptyParameters => ("empty_parameters", error.to_string()),
        E::NonFiniteParameters => ("non_finite_parameters", error.to_string()),
        E::NonFiniteInitialResidual => ("non_finite_initial_residual", error.to_string()),
        E::InsufficientRows { .. } => ("insufficient_rows", error.to_string()),
        E::SizeOverflow { .. } => ("size_overflow", error.to_string()),
        E::DegreeOverflow { .. } => ("degree_overflow", error.to_string()),
        E::InvalidMaxNfev => ("invalid_max_nfev", error.to_string()),
        E::InvalidFScale { .. } => ("invalid_f_scale", error.to_string()),
        E::InvalidXScaleLength { .. } => ("invalid_x_scale_length", error.to_string()),
        E::InvalidXScaleValue { .. } => ("invalid_x_scale_value", error.to_string()),
        E::InvalidJacobianLength { .. } => ("invalid_jacobian_length", error.to_string()),
        E::InvalidResidualLength { .. } => ("invalid_residual_length", error.to_string()),
        E::InvalidSliceLength { .. } => ("invalid_slice_length", error.to_string()),
        E::InvalidSvdOutput(_) => ("invalid_svd_output", error.to_string()),
        E::Backend(_) => ("backend", error.to_string()),
    };
    let result = detail(py, "TrfError", kind, &message)?;
    match error {
        E::EmptyResidual
        | E::EmptyParameters
        | E::NonFiniteParameters
        | E::NonFiniteInitialResidual
        | E::InvalidMaxNfev => {}
        E::InsufficientRows { m, n } | E::SizeOverflow { m, n } => {
            result.set_item("m", m)?;
            result.set_item("n", n)?;
        }
        E::DegreeOverflow { degree } => result.set_item("degree", degree)?,
        E::InvalidFScale { f_scale } => float_field(&result, "f_scale", *f_scale)?,
        E::InvalidXScaleLength { expected, got }
        | E::InvalidJacobianLength { expected, got }
        | E::InvalidResidualLength { expected, got } => {
            result.set_item("expected", expected)?;
            result.set_item("got", got)?;
        }
        E::InvalidXScaleValue { index, value } => {
            result.set_item("index", index)?;
            float_field(&result, "value", *value)?;
        }
        E::InvalidSliceLength {
            what,
            expected,
            got,
        } => {
            result.set_item("what", what)?;
            result.set_item("expected", expected)?;
            result.set_item("got", got)?;
        }
        E::InvalidSvdOutput(message) => result.set_item("reason", message)?,
        E::Backend(cause) => result.set_item("cause", backend_detail(py, cause)?)?,
    }
    Ok(result)
}

fn fit_result_detail<'py>(py: Python<'py>, fit: &TleFit) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item(
        "value",
        Py::new(py, crate::propagation::PyTleFit { inner: fit.clone() })?,
    )?;
    result.set_item(
        "omm_exact_sgp4_epoch",
        fit.omm.exact_sgp4_epoch.map(|epoch| (epoch.0, epoch.1)),
    )?;
    result.set_item(
        "omm_quantize_tle_derived_fields",
        fit.omm.quantize_tle_derived_fields,
    )?;
    result.set_item("element_set_omm_epoch_days", fit.elements.omm_epoch_days)?;
    Ok(result)
}

fn tle_fit_detail<'py>(py: Python<'py>, error: &TleFitError) -> PyResult<Bound<'py, PyDict>> {
    use TleFitError as E;
    let (kind, message) = match error {
        E::ArcTooShort { .. } => ("arc_too_short", error.to_string()),
        E::InvalidInput { .. } => ("invalid_input", error.to_string()),
        E::EpochsNotIncreasing { .. } => ("epochs_not_increasing", error.to_string()),
        E::EpochOutsideArc => ("epoch_outside_arc", error.to_string()),
        E::MixedVelocityPresence => ("mixed_velocity_presence", error.to_string()),
        E::NotElliptical => ("not_elliptical", error.to_string()),
        E::InclinationNearRetrograde { .. } => ("inclination_near_retrograde", error.to_string()),
        E::SeedPropagation { .. } => ("seed_propagation", error.to_string()),
        E::Solver(_) => ("solver", error.to_string()),
        E::SolutionInfeasible => ("solution_infeasible", error.to_string()),
        E::DidNotConverge { .. } => ("did_not_converge", error.to_string()),
        E::FinalElements(_) => ("final_elements", error.to_string()),
        E::TleEncode(_) => ("tle_encode", error.to_string()),
    };
    let result = detail(py, "TleFitError", kind, &message)?;
    match error {
        E::ArcTooShort { samples, needed } => {
            result.set_item("samples", samples)?;
            result.set_item("needed", needed)?;
        }
        E::InvalidInput { field, reason } => {
            result.set_item("field", field)?;
            result.set_item("reason", reason)?;
        }
        E::EpochsNotIncreasing { index } => result.set_item("index", index)?,
        E::EpochOutsideArc
        | E::MixedVelocityPresence
        | E::NotElliptical
        | E::SolutionInfeasible => {}
        E::InclinationNearRetrograde { inclination_deg } => {
            float_field(&result, "inclination_deg", *inclination_deg)?;
        }
        E::SeedPropagation {
            epoch_index,
            source,
        } => {
            result.set_item("epoch_index", epoch_index)?;
            result.set_item("cause", sgp4_detail(py, source)?)?;
        }
        E::Solver(cause) => result.set_item("cause", trf_detail(py, cause)?)?,
        E::DidNotConverge { result: fit } => {
            result.set_item("best_effort_fit", fit_result_detail(py, fit)?)?;
        }
        E::FinalElements(cause) => result.set_item("cause", sgp4_detail(py, cause)?)?,
        E::TleEncode(cause) => result.set_item("cause", tle_detail(py, cause)?)?,
    }
    Ok(result)
}

pub(crate) fn tle_fit_error(py: Python<'_>, error: TleFitError) -> PyErr {
    let message = error.to_string();
    match tle_fit_detail(py, &error) {
        Ok(payload) => {
            let python_error = crate::SolveError::new_err(message);
            if let Err(error) = python_error.value(py).setattr("detail", payload) {
                return error;
            }
            python_error
        }
        Err(error) => error,
    }
}
