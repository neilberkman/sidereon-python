//! ICGEM gravity coefficients and body-fixed geopotential evaluation.

use std::sync::Arc;

use pyo3::create_exception;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use sidereon_core::astro::forces;

use crate::earth_orientation::PyPropagationContext;
use crate::force_models::{
    propagation_error, with_context, PyAccelerationPartials, PyForceParameter,
};
use crate::propagation::PyTideSystem;
use crate::relative::PyCartesianState;
use crate::{binding_error, SolveError};
use sidereon_core::astro::forces::ForceModel;

create_exception!(
    _sidereon,
    IcgemError,
    SolveError,
    "An ICGEM gravity-field parse or evaluation failure."
);

fn icgem_error(error: forces::IcgemError) -> PyErr {
    let exception = IcgemError::new_err(error.to_string());
    match error {
        forces::IcgemError::MissingField {
            line,
            keyword,
            field,
        } => binding_error(exception, "missing_field", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("keyword", keyword)?;
            detail.set_item("field", field)
        }),
        forces::IcgemError::InvalidField {
            line,
            keyword,
            field,
            value,
        } => binding_error(exception, "invalid_field", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("keyword", keyword)?;
            detail.set_item("field", field)?;
            detail.set_item("value", value)
        }),
        forces::IcgemError::MissingHeaderKeyword { keyword } => {
            binding_error(exception, "missing_header_keyword", |detail| {
                detail.set_item("keyword", keyword)
            })
        }
        forces::IcgemError::RepeatedHeaderKeyword {
            line,
            first_line,
            keyword,
        } => binding_error(exception, "repeated_header_keyword", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("first_line", first_line)?;
            detail.set_item("keyword", keyword)
        }),
        forces::IcgemError::MissingEndOfHead => {
            binding_error(exception, "missing_end_of_head", |_| Ok(()))
        }
        forces::IcgemError::KeywordNotInFormat { line, keyword } => {
            binding_error(exception, "keyword_not_in_format", |detail| {
                detail.set_item("line", line)?;
                detail.set_item("keyword", keyword)
            })
        }
        forces::IcgemError::UnknownRecordKeyword { line, keyword } => {
            binding_error(exception, "unknown_record_keyword", |detail| {
                detail.set_item("line", line)?;
                detail.set_item("keyword", keyword)
            })
        }
        forces::IcgemError::MissingRecordKeyword { line } => {
            binding_error(exception, "missing_record_keyword", |detail| {
                detail.set_item("line", line)
            })
        }
        forces::IcgemError::DegreeAboveMaximum {
            line,
            degree,
            max_degree,
        } => binding_error(exception, "degree_above_maximum", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("degree", degree)?;
            detail.set_item("max_degree", max_degree)
        }),
        forces::IcgemError::OrderAboveDegree {
            line,
            degree,
            order,
        } => binding_error(exception, "order_above_degree", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)
        }),
        forces::IcgemError::RepeatedCoefficient {
            line,
            first_line,
            degree,
            order,
        } => binding_error(exception, "repeated_coefficient", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("first_line", first_line)?;
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)
        }),
        forces::IcgemError::OverlappingTerm {
            line,
            first_line,
            keyword,
            degree,
            order,
        } => binding_error(exception, "overlapping_term", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("first_line", first_line)?;
            detail.set_item("keyword", keyword)?;
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)
        }),
        forces::IcgemError::UncoveredTerm {
            line,
            keyword,
            degree,
            order,
        } => binding_error(exception, "uncovered_term", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("keyword", keyword)?;
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)
        }),
        forces::IcgemError::MissingCoefficient { degree, order } => {
            binding_error(exception, "missing_coefficient", |detail| {
                detail.set_item("degree", degree)?;
                detail.set_item("order", order)
            })
        }
        forces::IcgemError::TooFewRecords {
            max_degree,
            required,
            lines,
        } => binding_error(exception, "too_few_records", |detail| {
            detail.set_item("max_degree", max_degree)?;
            detail.set_item("required", required)?;
            detail.set_item("lines", lines)
        }),
        forces::IcgemError::NormalizationOutOfRange {
            line,
            degree,
            order,
        } => binding_error(exception, "normalization_out_of_range", |detail| {
            detail.set_item("line", line)?;
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)
        }),
        forces::IcgemError::TruncationOutOfRange {
            degree,
            order,
            max_degree,
        } => binding_error(exception, "truncation_out_of_range", |detail| {
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)?;
            detail.set_item("max_degree", max_degree)
        }),
        forces::IcgemError::UnknownTideSystem => {
            binding_error(exception, "unknown_tide_system", |_| Ok(()))
        }
        forces::IcgemError::TideSystemConflict { stated, assumed } => {
            binding_error(exception, "tide_system_conflict", |detail| {
                detail.set_item("stated", PyTideSystem::from(stated))?;
                detail.set_item("assumed", PyTideSystem::from(assumed))
            })
        }
        forces::IcgemError::EpochOutsideValidity {
            degree,
            order,
            epoch_tt_seconds,
        } => binding_error(exception, "epoch_outside_validity", |detail| {
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)?;
            detail.set_item("epoch_tt_seconds", epoch_tt_seconds)
        }),
        forces::IcgemError::NonFiniteEpoch => {
            binding_error(exception, "non_finite_epoch", |_| Ok(()))
        }
        forces::IcgemError::NonFiniteCoefficient {
            degree,
            order,
            epoch_tt_seconds,
        } => binding_error(exception, "non_finite_coefficient", |detail| {
            detail.set_item("degree", degree)?;
            detail.set_item("order", order)?;
            detail.set_item("epoch_tt_seconds", epoch_tt_seconds)
        }),
        _ => exception,
    }
}

/// Uncertainty columns declared by an ICGEM gravity field.
#[pyclass(
    module = "sidereon._sidereon",
    name = "IcgemErrors",
    eq,
    eq_int,
    frozen
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum PyIcgemErrors {
    /// `no`: no sigma columns.
    NO,
    /// `calibrated`: two calibrated sigma columns.
    CALIBRATED,
    /// `formal`: two formal sigma columns.
    FORMAL,
    /// `calibrated_and_formal`: two calibrated then two formal columns.
    CALIBRATED_AND_FORMAL,
}

#[pymethods]
impl PyIcgemErrors {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<forces::IcgemErrors> for PyIcgemErrors {
    fn from(value: forces::IcgemErrors) -> Self {
        match value {
            forces::IcgemErrors::No => Self::NO,
            forces::IcgemErrors::Calibrated => Self::CALIBRATED,
            forces::IcgemErrors::Formal => Self::FORMAL,
            forces::IcgemErrors::CalibratedAndFormal => Self::CALIBRATED_AND_FORMAL,
        }
    }
}

/// Coefficient normalization declared by an ICGEM gravity field.
#[pyclass(
    module = "sidereon._sidereon",
    name = "IcgemNormalization",
    eq,
    eq_int,
    frozen
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum PyIcgemNormalization {
    /// `fully_normalized`, the default.
    FULLY_NORMALIZED,
    /// `unnormalized`; converted to fully normalized on reading.
    UNNORMALIZED,
}

#[pymethods]
impl PyIcgemNormalization {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<forces::IcgemNormalization> for PyIcgemNormalization {
    fn from(value: forces::IcgemNormalization) -> Self {
        match value {
            forces::IcgemNormalization::FullyNormalized => Self::FULLY_NORMALIZED,
            forces::IcgemNormalization::Unnormalized => Self::UNNORMALIZED,
        }
    }
}

/// ICGEM file format version.
#[pyclass(
    module = "sidereon._sidereon",
    name = "IcgemFormat",
    eq,
    eq_int,
    frozen
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum PyIcgemFormat {
    /// intervals.
    ICGEM1_0,
    /// `icgem2.0`: time-variable records valid over `t0 <= t < t1`.
    ICGEM2_0,
}

#[pymethods]
impl PyIcgemFormat {
    /// Hash consistently with the integer value used in comparisons.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<forces::IcgemFormat> for PyIcgemFormat {
    fn from(value: forces::IcgemFormat) -> Self {
        match value {
            forces::IcgemFormat::Icgem1_0 => Self::ICGEM1_0,
            forces::IcgemFormat::Icgem2_0 => Self::ICGEM2_0,
        }
    }
}

/// Constant, trend, or periodic time-variable gravity term.
#[pyclass(module = "sidereon._sidereon", name = "IcgemTermKind", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyIcgemTermKind(forces::IcgemTermKind);

#[pymethods]
impl PyIcgemTermKind {
    /// Constant part of a time-variable coefficient.
    #[classattr]
    const CONSTANT: Self = Self(forces::IcgemTermKind::Constant);
    /// Linear trend per Julian year.
    #[classattr]
    const TREND: Self = Self(forces::IcgemTermKind::Trend);
    /// Cosine amplitude at a period in Julian years.
    #[staticmethod]
    fn cosine(period_years: f64) -> Self {
        Self(forces::IcgemTermKind::Cosine { period_years })
    }

    /// Sine amplitude at a period in Julian years.
    #[staticmethod]
    fn sine(period_years: f64) -> Self {
        Self(forces::IcgemTermKind::Sine { period_years })
    }

    /// Period of a sinusoidal term in Julian years.
    #[getter]
    fn period_years(&self) -> Option<f64> {
        match self.0 {
            forces::IcgemTermKind::Cosine { period_years }
            | forces::IcgemTermKind::Sine { period_years } => Some(period_years),
            _ => None,
        }
    }

    /// Stable name of the time-variable term.
    #[getter]
    fn label(&self) -> &'static str {
        match self.0 {
            forces::IcgemTermKind::Constant => "constant",
            forces::IcgemTermKind::Trend => "trend",
            forces::IcgemTermKind::Cosine { .. } => "cosine",
            forces::IcgemTermKind::Sine { .. } => "sine",
        }
    }
}

/// Metadata from the gravity field header.
#[pyclass(module = "sidereon._sidereon", name = "IcgemHeader", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyIcgemHeader(forces::IcgemHeader);

#[pymethods]
impl PyIcgemHeader {
    /// Model name.
    #[getter]
    fn model_name(&self) -> String {
        self.0.model_name.clone()
    }

    /// Gravitational parameter in cubic metres per square second.
    #[getter]
    fn gravity_constant_m3_s2(&self) -> f64 {
        self.0.gravity_constant_m3_s2
    }

    /// Reference radius in metres.
    #[getter]
    fn radius_m(&self) -> f64 {
        self.0.radius_m
    }

    /// Maximum degree stated in the file.
    #[getter]
    fn max_degree(&self) -> u16 {
        self.0.max_degree
    }

    /// Uncertainty column layout.
    #[getter]
    fn errors(&self) -> PyIcgemErrors {
        self.0.errors.into()
    }

    /// Normalization stated in the file.
    #[getter]
    fn normalization(&self) -> PyIcgemNormalization {
        self.0.normalization.into()
    }

    /// Tide system if stated.
    #[getter]
    fn tide_system(&self) -> Option<PyTideSystem> {
        self.0.tide_system.map(Into::into)
    }

    /// ICGEM format version.
    #[getter]
    fn format(&self) -> PyIcgemFormat {
        self.0.format.into()
    }
}

/// Fully normalized coefficient uncertainties.
#[pyclass(module = "sidereon._sidereon", name = "IcgemSigmas", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyIcgemSigmas(forces::IcgemSigmas);

#[pymethods]
impl PyIcgemSigmas {
    /// Build an uncertainty record with no sigmas.
    #[new]
    fn new() -> Self {
        Self(forces::IcgemSigmas::default())
    }

    /// Calibrated cosine and sine uncertainties, if present.
    #[getter]
    fn calibrated(&self) -> Option<[f64; 2]> {
        self.0.calibrated
    }

    /// Formal cosine and sine uncertainties, if present.
    #[getter]
    fn formal(&self) -> Option<[f64; 2]> {
        self.0.formal
    }
}

/// A time-variable coefficient record.
#[pyclass(
    module = "sidereon._sidereon",
    name = "IcgemTimeVariableTerm",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PyIcgemTimeVariableTerm(forces::IcgemTimeVariableTerm);

#[pymethods]
impl PyIcgemTimeVariableTerm {
    /// Line number in the gravity field text.
    #[getter]
    fn line(&self) -> usize {
        self.0.line
    }

    /// Type and optional period of the record.
    #[getter]
    fn kind(&self) -> PyIcgemTermKind {
        PyIcgemTermKind(self.0.kind)
    }

    /// Harmonic degree.
    #[getter]
    fn degree(&self) -> u16 {
        self.0.degree
    }

    /// Harmonic order.
    #[getter]
    fn order(&self) -> u16 {
        self.0.order
    }

    /// Fully normalized cosine value.
    #[getter]
    fn c(&self) -> f64 {
        self.0.c
    }

    /// Fully normalized sine value.
    #[getter]
    fn s(&self) -> f64 {
        self.0.s
    }

    /// Coefficient uncertainties.
    #[getter]
    fn sigmas(&self) -> PyIcgemSigmas {
        PyIcgemSigmas(self.0.sigmas)
    }

    /// Reference epoch in TT seconds from J2000.
    #[getter]
    fn reference_tt_seconds(&self) -> f64 {
        self.0.reference_tt_seconds
    }

    /// Inclusive validity start in TT seconds from J2000.
    #[getter]
    fn valid_from_tt_seconds(&self) -> f64 {
        self.0.valid_from_tt_seconds
    }

    /// Exclusive validity end in TT seconds from J2000.
    #[getter]
    fn valid_until_tt_seconds(&self) -> f64 {
        self.0.valid_until_tt_seconds
    }
}

/// Fully normalized real spherical harmonic coefficient.
#[pyclass(
    module = "sidereon._sidereon",
    name = "SphericalHarmonicCoefficient",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PySphericalHarmonicCoefficient(forces::SphericalHarmonicCoefficient);

#[pymethods]
impl PySphericalHarmonicCoefficient {
    /// Harmonic degree.
    #[getter]
    fn degree(&self) -> u16 {
        self.0.degree
    }

    /// Harmonic order.
    #[getter]
    fn order(&self) -> u16 {
        self.0.order
    }

    /// Fully normalized cosine coefficient.
    #[getter]
    fn c(&self) -> f64 {
        self.0.c
    }

    /// Fully normalized sine coefficient.
    #[getter]
    fn s(&self) -> f64 {
        self.0.s
    }
}

/// Static and time-variable coefficients from an ICGEM gravity field.
#[pyclass(module = "sidereon._sidereon", name = "IcgemGravityField", eq, frozen)]
#[derive(Clone, PartialEq)]
pub struct PyIcgemGravityField(forces::IcgemGravityField);

#[pymethods]
impl PyIcgemGravityField {
    /// Parse an ICGEM gravity field.
    #[staticmethod]
    fn parse(text: &str) -> PyResult<Self> {
        forces::IcgemGravityField::parse(text)
            .map(Self)
            .map_err(icgem_error)
    }

    /// Parse an ICGEM gravity field with a storage truncation.
    #[staticmethod]
    fn parse_truncated(text: &str, max_degree: u16, max_order: u16) -> PyResult<Self> {
        forces::IcgemGravityField::parse_truncated(text, max_degree, max_order)
            .map(Self)
            .map_err(icgem_error)
    }

    /// Gravity field metadata.
    fn header(&self) -> PyIcgemHeader {
        PyIcgemHeader(self.0.header().clone())
    }

    /// Highest stored degree.
    fn stored_max_degree(&self) -> u16 {
        self.0.stored_max_degree()
    }

    /// Highest stored order.
    fn stored_max_order(&self) -> u16 {
        self.0.stored_max_order()
    }

    /// Whether any stored coefficient varies with time.
    fn is_time_variable(&self) -> bool {
        self.0.is_time_variable()
    }

    /// Fully normalized static coefficient, if stored.
    fn static_coefficient(
        &self,
        degree: u16,
        order: u16,
    ) -> Option<PySphericalHarmonicCoefficient> {
        self.0
            .static_coefficient(degree, order)
            .map(PySphericalHarmonicCoefficient)
    }

    /// Uncertainties of a static coefficient, if stored.
    fn static_sigmas(&self, degree: u16, order: u16) -> Option<PyIcgemSigmas> {
        self.0.static_sigmas(degree, order).map(PyIcgemSigmas)
    }

    /// Time-variable records stored by the field.
    fn time_variable_terms(&self) -> Vec<PyIcgemTimeVariableTerm> {
        self.0
            .time_variable_terms()
            .iter()
            .copied()
            .map(PyIcgemTimeVariableTerm)
            .collect()
    }

    /// Fully normalized coefficient at a TT epoch, if stored.
    fn coefficient_at(
        &self,
        degree: u16,
        order: u16,
        epoch_tt_seconds: f64,
    ) -> PyResult<Option<PySphericalHarmonicCoefficient>> {
        self.0
            .coefficient_at(degree, order, epoch_tt_seconds)
            .map(|value| value.map(PySphericalHarmonicCoefficient))
            .map_err(icgem_error)
    }

    /// Apply a tide system when the field lacks one.
    fn with_assumed_tide_system(&self, tide_system: PyTideSystem) -> PyResult<Self> {
        self.0
            .clone()
            .with_assumed_tide_system(tide_system.into())
            .map(Self)
            .map_err(icgem_error)
    }

    /// Build a spherical-harmonic force model.
    fn gravity(&self, max_degree: u16, max_order: u16) -> PyResult<PySphericalHarmonicGravity> {
        self.0
            .gravity(max_degree, max_order)
            .map(Arc::new)
            .map(PySphericalHarmonicGravity)
            .map_err(icgem_error)
    }

    /// Build a spherical-harmonic force model converted to the requested tide system.
    fn gravity_in_tide_system(
        &self,
        max_degree: u16,
        max_order: u16,
        tide_system: PyTideSystem,
    ) -> PyResult<PySphericalHarmonicGravity> {
        self.0
            .gravity_in_tide_system(max_degree, max_order, tide_system.into())
            .map(Arc::new)
            .map(PySphericalHarmonicGravity)
            .map_err(icgem_error)
    }
}

/// Fully normalized gravity force in a body-fixed frame.
#[pyclass(
    module = "sidereon._sidereon",
    name = "SphericalHarmonicGravity",
    eq,
    frozen
)]
#[derive(Clone, PartialEq)]
pub struct PySphericalHarmonicGravity(pub(crate) Arc<forces::SphericalHarmonicGravity>);

#[pymethods]
impl PySphericalHarmonicGravity {
    /// Whether the gravity field has time-variable coefficients.
    fn is_time_variable(&self) -> bool {
        self.0.is_time_variable()
    }

    /// Evaluate perturbing acceleration in kilometres per square second.
    fn body_fixed_acceleration_km_s2(
        &self,
        position_body_fixed_km: [f64; 3],
    ) -> PyResult<[f64; 3]> {
        self.0
            .body_fixed_acceleration_km_s2(position_body_fixed_km)
            .map_err(propagation_error)
    }

    /// Evaluate perturbing acceleration in kilometres per square second at the given TT epoch.
    fn body_fixed_acceleration_at_km_s2(
        &self,
        position_body_fixed_km: [f64; 3],
        epoch_tt_seconds: f64,
    ) -> PyResult<[f64; 3]> {
        self.0
            .body_fixed_acceleration_at_km_s2(position_body_fixed_km, epoch_tt_seconds)
            .map_err(propagation_error)
    }

    /// Evaluate perturbing potential in square kilometres per square second.
    fn body_fixed_potential_km2_s2(&self, position_body_fixed_km: [f64; 3]) -> PyResult<f64> {
        self.0
            .body_fixed_potential_km2_s2(position_body_fixed_km)
            .map_err(propagation_error)
    }

    /// Evaluate perturbing potential in square kilometres per square second at the given TT epoch.
    fn body_fixed_potential_at_km2_s2(
        &self,
        position_body_fixed_km: [f64; 3],
        epoch_tt_seconds: f64,
    ) -> PyResult<f64> {
        self.0
            .body_fixed_potential_at_km2_s2(position_body_fixed_km, epoch_tt_seconds)
            .map_err(propagation_error)
    }

    /// Fully normalized coefficient at a TT epoch, if in the truncation.
    fn coefficient_at(
        &self,
        degree: u16,
        order: u16,
        epoch_tt_seconds: f64,
    ) -> PyResult<Option<PySphericalHarmonicCoefficient>> {
        self.0
            .coefficient_at(degree, order, epoch_tt_seconds)
            .map(|value| value.map(PySphericalHarmonicCoefficient))
            .map_err(propagation_error)
    }

    /// Return the model's estimable parameters.
    fn parameters(&self) -> Vec<PyForceParameter> {
        self.0
            .parameters()
            .into_iter()
            .map(PyForceParameter)
            .collect()
    }

    /// Number of estimable parameters.
    fn parameter_count(&self) -> usize {
        self.0.parameter_count()
    }

    /// Evaluate acceleration in the inertial frame.
    #[pyo3(signature = (state, context=None))]
    fn acceleration(
        &self,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| self.0.acceleration(state.inner(), ctx))
            .map(Into::into)
            .map_err(propagation_error)
    }

    /// Evaluate acceleration with parameters in the inertial frame.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_with_parameters(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<[f64; 3]> {
        with_context(context, |ctx| {
            self.0
                .acceleration_with_parameters(&parameters, state.inner(), ctx)
        })
        .map(Into::into)
        .map_err(propagation_error)
    }

    /// Evaluate acceleration partials in the inertial frame.
    #[pyo3(signature = (parameters, state, context=None))]
    fn acceleration_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<PyAccelerationPartials> {
        with_context(context, |ctx| {
            self.0
                .acceleration_partials(&parameters, state.inner(), ctx)
        })
        .map(PyAccelerationPartials)
        .map_err(propagation_error)
    }

    /// Evaluate parameter partials in the inertial frame.
    #[pyo3(signature = (parameters, state, context=None))]
    fn parameter_partials(
        &self,
        parameters: Vec<f64>,
        state: &PyCartesianState,
        context: Option<&PyPropagationContext>,
    ) -> PyResult<Vec<[f64; 3]>> {
        with_context(context, |ctx| {
            self.0.parameter_partials(&parameters, state.inner(), ctx)
        })
        .map(|values| values.into_iter().map(Into::into).collect())
        .map_err(propagation_error)
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("IcgemError", m.py().get_type::<IcgemError>())?;
    m.add_class::<PyIcgemErrors>()?;
    m.add_class::<PyIcgemNormalization>()?;
    m.add_class::<PyIcgemFormat>()?;
    m.add_class::<PyIcgemTermKind>()?;
    m.add_class::<PyIcgemHeader>()?;
    m.add_class::<PyIcgemSigmas>()?;
    m.add_class::<PyIcgemTimeVariableTerm>()?;
    m.add_class::<PySphericalHarmonicCoefficient>()?;
    m.add_class::<PyIcgemGravityField>()?;
    m.add_class::<PySphericalHarmonicGravity>()?;
    Ok(())
}
