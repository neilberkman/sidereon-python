//! CCSDS CDM binding.
//!
//! Provides typed Python value objects for the core `CdmKvn` / `CdmObject`
//! structs and parse/encode entry points for KVN and XML. The grammar and
//! serialization stay entirely in `sidereon-core`; this module only marshals
//! strings, optional fields, and numpy vectors.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon_core::astro::cdm::{
    encode_kvn, encode_xml, parse_kvn, parse_xml, CdmAdditionalParameters, CdmError,
    CdmInputErrorKind, CdmKvn, CdmObject, CdmOdParameters,
};
use sidereon_core::astro::ndm::TextIssue;

use crate::marshal::{fixed_array, FinitePolicy};
use crate::{np_array, CdmParseError};

fn cdm_kind(kind: CdmInputErrorKind) -> &'static str {
    match kind {
        CdmInputErrorKind::Missing => "Missing",
        CdmInputErrorKind::NonFinite => "NonFinite",
        CdmInputErrorKind::FloatParse => "FloatParse",
        CdmInputErrorKind::IntParse => "IntParse",
        CdmInputErrorKind::NotPositive => "NotPositive",
        CdmInputErrorKind::Negative => "Negative",
        CdmInputErrorKind::OutOfRange => "OutOfRange",
        CdmInputErrorKind::InvalidCivilDate => "InvalidCivilDate",
        CdmInputErrorKind::InvalidCivilTime => "InvalidCivilTime",
    }
}

fn text_issue(issue: TextIssue) -> &'static str {
    match issue {
        TextIssue::LineBreak => "LineBreak",
        TextIssue::SurroundingWhitespace => "SurroundingWhitespace",
        TextIssue::InteriorWhitespace => "InteriorWhitespace",
        TextIssue::KeywordSeparator => "KeywordSeparator",
        TextIssue::XmlIllegalCharacter => "XmlIllegalCharacter",
        TextIssue::Empty => "Empty",
        TextIssue::DetachedComment => "DetachedComment",
        TextIssue::RepeatedParameter => "RepeatedParameter",
        TextIssue::CommentNotCarried => "CommentNotCarried",
    }
}

fn cdm_detail<'py>(py: Python<'py>, err: &CdmError) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "cdm")?;
    let kind = match err {
        CdmError::IncompleteStateVector => "IncompleteStateVector",
        CdmError::InvalidField { field, kind } => {
            detail.set_item("field", field)?;
            detail.set_item("validation_kind", cdm_kind(*kind))?;
            "InvalidField"
        }
        CdmError::MalformedXml(message) => {
            detail.set_item("message", message)?;
            "MalformedXml"
        }
        CdmError::DuplicateField {
            field,
            first,
            second,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("first", first)?;
            detail.set_item("second", second)?;
            "DuplicateField"
        }
        CdmError::UnitMismatch {
            field,
            unit,
            expected,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("unit", unit)?;
            detail.set_item("expected", expected)?;
            "UnitMismatch"
        }
        CdmError::UnexpectedObjectCount(count) => {
            detail.set_item("count", *count)?;
            "UnexpectedObjectCount"
        }
        CdmError::MultipleMessages { count } => {
            detail.set_item("count", *count)?;
            "MultipleMessages"
        }
        CdmError::UnknownField(field) => {
            detail.set_item("field", field)?;
            "UnknownField"
        }
        CdmError::MalformedLine { line, text } => {
            detail.set_item("line", *line)?;
            detail.set_item("text", text)?;
            "MalformedLine"
        }
        CdmError::UnknownObject(value) => {
            detail.set_item("value", value)?;
            "UnknownObject"
        }
        CdmError::RepeatedObject(value) => {
            detail.set_item("value", value)?;
            "RepeatedObject"
        }
        CdmError::UnwritableText {
            field,
            value,
            issue,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("value", value)?;
            detail.set_item("issue", text_issue(*issue))?;
            "UnwritableText"
        }
        CdmError::HardBodyRadiusComment { comment } => {
            detail.set_item("comment", comment)?;
            "HardBodyRadiusComment"
        }
    };
    detail.set_item("kind", kind)?;
    Ok(detail)
}

fn to_cdm_err(py: Python<'_>, err: CdmError) -> PyErr {
    let py_err = CdmParseError::new_err(err.to_string());
    let detail = match cdm_detail(py, &err) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    if let Err(error) = py_err.value(py).setattr("detail", detail) {
        return error;
    }
    py_err
}

/// A three-component vector whose components are each optional, as the
/// relative state and screening volume of a CDM hold them.
type OptionalTriple = (Option<f64>, Option<f64>, Option<f64>);

fn triple(values: [Option<f64>; 3]) -> OptionalTriple {
    (values[0], values[1], values[2])
}

fn from_triple(values: OptionalTriple) -> [Option<f64>; 3] {
    [values.0, values.1, values.2]
}

/// The OD parameters of one CDM object (CCSDS 508.0-B-1 table 3-4). Every
/// item is optional.
#[pyclass(module = "sidereon._sidereon", name = "CdmOdParameters")]
#[derive(Clone, PartialEq, Default)]
pub struct PyCdmOdParameters {
    inner: CdmOdParameters,
}

#[pymethods]
impl PyCdmOdParameters {
    #[new]
    #[pyo3(signature = (
        *,
        comments=Vec::new(),
        time_lastob_start=None,
        time_lastob_end=None,
        recommended_od_span_d=None,
        actual_od_span_d=None,
        obs_available=None,
        obs_used=None,
        tracks_available=None,
        tracks_used=None,
        residuals_accepted_pct=None,
        weighted_rms=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        comments: Vec<String>,
        time_lastob_start: Option<String>,
        time_lastob_end: Option<String>,
        recommended_od_span_d: Option<f64>,
        actual_od_span_d: Option<f64>,
        obs_available: Option<u64>,
        obs_used: Option<u64>,
        tracks_available: Option<u64>,
        tracks_used: Option<u64>,
        residuals_accepted_pct: Option<f64>,
        weighted_rms: Option<f64>,
    ) -> Self {
        Self {
            inner: CdmOdParameters {
                comments,
                time_lastob_start,
                time_lastob_end,
                recommended_od_span_d,
                actual_od_span_d,
                obs_available,
                obs_used,
                tracks_available,
                tracks_used,
                residuals_accepted_pct,
                weighted_rms,
            },
        }
    }

    /// Comments at the start of the block.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Raw `TIME_LASTOB_START` text.
    #[getter]
    fn time_lastob_start(&self) -> Option<String> {
        self.inner.time_lastob_start.clone()
    }

    /// Raw `TIME_LASTOB_END` text.
    #[getter]
    fn time_lastob_end(&self) -> Option<String> {
        self.inner.time_lastob_end.clone()
    }

    /// `RECOMMENDED_OD_SPAN`, days.
    #[getter]
    fn recommended_od_span_d(&self) -> Option<f64> {
        self.inner.recommended_od_span_d
    }

    /// `ACTUAL_OD_SPAN`, days.
    #[getter]
    fn actual_od_span_d(&self) -> Option<f64> {
        self.inner.actual_od_span_d
    }

    /// `OBS_AVAILABLE`, a count.
    #[getter]
    fn obs_available(&self) -> Option<u64> {
        self.inner.obs_available
    }

    /// `OBS_USED`, a count.
    #[getter]
    fn obs_used(&self) -> Option<u64> {
        self.inner.obs_used
    }

    /// `TRACKS_AVAILABLE`, a count.
    #[getter]
    fn tracks_available(&self) -> Option<u64> {
        self.inner.tracks_available
    }

    /// `TRACKS_USED`, a count.
    #[getter]
    fn tracks_used(&self) -> Option<u64> {
        self.inner.tracks_used
    }

    /// `RESIDUALS_ACCEPTED`, percent.
    #[getter]
    fn residuals_accepted_pct(&self) -> Option<f64> {
        self.inner.residuals_accepted_pct
    }

    /// `WEIGHTED_RMS`, dimensionless.
    #[getter]
    fn weighted_rms(&self) -> Option<f64> {
        self.inner.weighted_rms
    }

    fn __repr__(&self) -> String {
        format!(
            "CdmOdParameters(obs_used={:?}, tracks_used={:?})",
            self.inner.obs_used, self.inner.tracks_used
        )
    }

    fn __eq__(&self, other: &PyCdmOdParameters) -> bool {
        self == other
    }
}

/// The additional parameters of one CDM object (CCSDS 508.0-B-1 table 3-4).
/// Every item is optional.
#[pyclass(module = "sidereon._sidereon", name = "CdmAdditionalParameters")]
#[derive(Clone, PartialEq, Default)]
pub struct PyCdmAdditionalParameters {
    inner: CdmAdditionalParameters,
}

#[pymethods]
impl PyCdmAdditionalParameters {
    #[new]
    #[pyo3(signature = (
        *,
        comments=Vec::new(),
        area_pc_m2=None,
        area_drg_m2=None,
        area_srp_m2=None,
        mass_kg=None,
        cd_area_over_mass_m2_kg=None,
        cr_area_over_mass_m2_kg=None,
        thrust_acceleration_m_s2=None,
        sedr_w_kg=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        comments: Vec<String>,
        area_pc_m2: Option<f64>,
        area_drg_m2: Option<f64>,
        area_srp_m2: Option<f64>,
        mass_kg: Option<f64>,
        cd_area_over_mass_m2_kg: Option<f64>,
        cr_area_over_mass_m2_kg: Option<f64>,
        thrust_acceleration_m_s2: Option<f64>,
        sedr_w_kg: Option<f64>,
    ) -> Self {
        Self {
            inner: CdmAdditionalParameters {
                comments,
                area_pc_m2,
                area_drg_m2,
                area_srp_m2,
                mass_kg,
                cd_area_over_mass_m2_kg,
                cr_area_over_mass_m2_kg,
                thrust_acceleration_m_s2,
                sedr_w_kg,
            },
        }
    }

    /// Comments at the start of the block.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// `AREA_PC`, m^2.
    #[getter]
    fn area_pc_m2(&self) -> Option<f64> {
        self.inner.area_pc_m2
    }

    /// `AREA_DRG`, m^2.
    #[getter]
    fn area_drg_m2(&self) -> Option<f64> {
        self.inner.area_drg_m2
    }

    /// `AREA_SRP`, m^2.
    #[getter]
    fn area_srp_m2(&self) -> Option<f64> {
        self.inner.area_srp_m2
    }

    /// `MASS`, kg.
    #[getter]
    fn mass_kg(&self) -> Option<f64> {
        self.inner.mass_kg
    }

    /// `CD_AREA_OVER_MASS`, m^2/kg.
    #[getter]
    fn cd_area_over_mass_m2_kg(&self) -> Option<f64> {
        self.inner.cd_area_over_mass_m2_kg
    }

    /// `CR_AREA_OVER_MASS`, m^2/kg.
    #[getter]
    fn cr_area_over_mass_m2_kg(&self) -> Option<f64> {
        self.inner.cr_area_over_mass_m2_kg
    }

    /// `THRUST_ACCELERATION`, m/s^2.
    #[getter]
    fn thrust_acceleration_m_s2(&self) -> Option<f64> {
        self.inner.thrust_acceleration_m_s2
    }

    /// `SEDR`, W/kg.
    #[getter]
    fn sedr_w_kg(&self) -> Option<f64> {
        self.inner.sedr_w_kg
    }

    fn __repr__(&self) -> String {
        format!(
            "CdmAdditionalParameters(mass_kg={:?}, area_pc_m2={:?})",
            self.inner.mass_kg, self.inner.area_pc_m2
        )
    }

    fn __eq__(&self, other: &PyCdmAdditionalParameters) -> bool {
        self == other
    }
}

/// One object's metadata, state vector, and RTN position covariance from a CDM.
#[pyclass(module = "sidereon._sidereon", name = "CdmObject")]
#[derive(Clone, PartialEq)]
pub struct PyCdmObject {
    inner: CdmObject,
}

impl PyCdmObject {
    fn from_inner(inner: CdmObject) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCdmObject {
    #[new]
    #[pyo3(signature = (
        position_km,
        velocity_km_s,
        covariance_rtn,
        *,
        object_designator=None,
        catalog_name=None,
        object_name=None,
        international_designator=None,
        object_type=None,
        operator_contact_position=None,
        operator_organization=None,
        operator_phone=None,
        operator_email=None,
        ephemeris_name=None,
        covariance_method=None,
        maneuverable=None,
        orbit_center=None,
        ref_frame=None,
        gravity_model=None,
        atmospheric_model=None,
        n_body_perturbations=None,
        solar_rad_pressure=None,
        earth_tides=None,
        intrack_thrust=None,
        velocity_covariance_rtn=None,
        metadata_comments=Vec::new(),
        od_parameters=None,
        additional_parameters=None,
        state_comments=Vec::new(),
        covariance_comments=Vec::new(),
        drag_covariance_rtn=None,
        srp_covariance_rtn=None,
        thrust_covariance_rtn=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
        covariance_rtn: PyReadonlyArray1<'_, f64>,
        object_designator: Option<String>,
        catalog_name: Option<String>,
        object_name: Option<String>,
        international_designator: Option<String>,
        object_type: Option<String>,
        operator_contact_position: Option<String>,
        operator_organization: Option<String>,
        operator_phone: Option<String>,
        operator_email: Option<String>,
        ephemeris_name: Option<String>,
        covariance_method: Option<String>,
        maneuverable: Option<String>,
        orbit_center: Option<String>,
        ref_frame: Option<String>,
        gravity_model: Option<String>,
        atmospheric_model: Option<String>,
        n_body_perturbations: Option<String>,
        solar_rad_pressure: Option<String>,
        earth_tides: Option<String>,
        intrack_thrust: Option<String>,
        velocity_covariance_rtn: Option<PyReadonlyArray1<'_, f64>>,
        metadata_comments: Vec<String>,
        od_parameters: Option<PyCdmOdParameters>,
        additional_parameters: Option<PyCdmAdditionalParameters>,
        state_comments: Vec<String>,
        covariance_comments: Vec<String>,
        drag_covariance_rtn: Option<PyReadonlyArray1<'_, f64>>,
        srp_covariance_rtn: Option<PyReadonlyArray1<'_, f64>>,
        thrust_covariance_rtn: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let position = fixed_array::<3>("position_km", &position_km, FinitePolicy::AllowNonFinite)?;
        let velocity = fixed_array::<3>(
            "velocity_km_s",
            &velocity_km_s,
            FinitePolicy::AllowNonFinite,
        )?;
        let velocity_covariance_rtn = velocity_covariance_rtn
            .map(|values| {
                fixed_array::<15>(
                    "velocity_covariance_rtn",
                    &values,
                    FinitePolicy::AllowNonFinite,
                )
            })
            .transpose()?;
        let drag_covariance_rtn = drag_covariance_rtn
            .map(|values| {
                fixed_array::<7>("drag_covariance_rtn", &values, FinitePolicy::AllowNonFinite)
            })
            .transpose()?;
        let srp_covariance_rtn = srp_covariance_rtn
            .map(|values| {
                fixed_array::<8>("srp_covariance_rtn", &values, FinitePolicy::AllowNonFinite)
            })
            .transpose()?;
        let thrust_covariance_rtn = thrust_covariance_rtn
            .map(|values| {
                fixed_array::<9>(
                    "thrust_covariance_rtn",
                    &values,
                    FinitePolicy::AllowNonFinite,
                )
            })
            .transpose()?;
        Ok(Self {
            inner: CdmObject {
                metadata_comments,
                object_designator,
                catalog_name,
                object_name,
                international_designator,
                object_type,
                operator_contact_position,
                operator_organization,
                operator_phone,
                operator_email,
                ephemeris_name,
                covariance_method,
                maneuverable,
                orbit_center,
                ref_frame,
                gravity_model,
                atmospheric_model,
                n_body_perturbations,
                solar_rad_pressure,
                earth_tides,
                intrack_thrust,
                od_parameters: od_parameters.map(|p| p.inner).unwrap_or_default(),
                additional_parameters: additional_parameters.map(|p| p.inner).unwrap_or_default(),
                state_comments,
                state: (
                    (position[0], position[1], position[2]),
                    (velocity[0], velocity[1], velocity[2]),
                ),
                covariance_rtn: fixed_array::<6>(
                    "covariance_rtn",
                    &covariance_rtn,
                    FinitePolicy::AllowNonFinite,
                )?,
                covariance_comments,
                velocity_covariance_rtn,
                drag_covariance_rtn,
                srp_covariance_rtn,
                thrust_covariance_rtn,
            },
        })
    }

    #[getter]
    fn object_designator(&self) -> Option<String> {
        self.inner.object_designator.clone()
    }

    #[getter]
    fn catalog_name(&self) -> Option<String> {
        self.inner.catalog_name.clone()
    }

    #[getter]
    fn object_name(&self) -> Option<String> {
        self.inner.object_name.clone()
    }

    #[getter]
    fn international_designator(&self) -> Option<String> {
        self.inner.international_designator.clone()
    }

    #[getter]
    fn object_type(&self) -> Option<String> {
        self.inner.object_type.clone()
    }

    #[getter]
    fn operator_contact_position(&self) -> Option<String> {
        self.inner.operator_contact_position.clone()
    }

    #[getter]
    fn operator_organization(&self) -> Option<String> {
        self.inner.operator_organization.clone()
    }

    #[getter]
    fn operator_phone(&self) -> Option<String> {
        self.inner.operator_phone.clone()
    }

    #[getter]
    fn operator_email(&self) -> Option<String> {
        self.inner.operator_email.clone()
    }

    #[getter]
    fn ephemeris_name(&self) -> Option<String> {
        self.inner.ephemeris_name.clone()
    }

    #[getter]
    fn covariance_method(&self) -> Option<String> {
        self.inner.covariance_method.clone()
    }

    #[getter]
    fn maneuverable(&self) -> Option<String> {
        self.inner.maneuverable.clone()
    }

    #[getter]
    fn orbit_center(&self) -> Option<String> {
        self.inner.orbit_center.clone()
    }

    #[getter]
    fn ref_frame(&self) -> Option<String> {
        self.inner.ref_frame.clone()
    }

    #[getter]
    fn gravity_model(&self) -> Option<String> {
        self.inner.gravity_model.clone()
    }

    #[getter]
    fn atmospheric_model(&self) -> Option<String> {
        self.inner.atmospheric_model.clone()
    }

    #[getter]
    fn n_body_perturbations(&self) -> Option<String> {
        self.inner.n_body_perturbations.clone()
    }

    #[getter]
    fn solar_rad_pressure(&self) -> Option<String> {
        self.inner.solar_rad_pressure.clone()
    }

    #[getter]
    fn earth_tides(&self) -> Option<String> {
        self.inner.earth_tides.clone()
    }

    #[getter]
    fn intrack_thrust(&self) -> Option<String> {
        self.inner.intrack_thrust.clone()
    }

    /// Position vector as a numpy `(3,)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ((x, y, z), _) = self.inner.state;
        np_array(py, &[x, y, z])
    }

    /// Velocity vector as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let (_, (x_dot, y_dot, z_dot)) = self.inner.state;
        np_array(py, &[x_dot, y_dot, z_dot])
    }

    /// RTN position-covariance lower triangle `(CR_R, CT_R, CT_T, CN_R, CN_T,
    /// CN_N)` as a numpy `(6,)` array.
    #[getter]
    fn covariance_rtn<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.covariance_rtn)
    }

    /// The RTN velocity-covariance lower-triangle rows completing the 6x6 matrix
    /// (`CRDOT_R, CRDOT_T, CRDOT_N, CRDOT_RDOT, CTDOT_R, ...`, 15 elements in
    /// CCSDS order) as a numpy `(15,)` array, or `None` when the producer carried
    /// only the position-covariance block.
    #[getter]
    fn velocity_covariance_rtn<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .velocity_covariance_rtn
            .map(|values| PyArray1::from_slice(py, &values))
    }

    /// Metadata comments of the object block.
    #[getter]
    fn metadata_comments(&self) -> Vec<String> {
        self.inner.metadata_comments.clone()
    }

    /// OD parameters of the object (table 3-4).
    #[getter]
    fn od_parameters(&self) -> PyCdmOdParameters {
        PyCdmOdParameters {
            inner: self.inner.od_parameters.clone(),
        }
    }

    /// Additional parameters of the object (table 3-4).
    #[getter]
    fn additional_parameters(&self) -> PyCdmAdditionalParameters {
        PyCdmAdditionalParameters {
            inner: self.inner.additional_parameters.clone(),
        }
    }

    /// Comments of the state vector block.
    #[getter]
    fn state_comments(&self) -> Vec<String> {
        self.inner.state_comments.clone()
    }

    /// Comments of the covariance block.
    #[getter]
    fn covariance_comments(&self) -> Vec<String> {
        self.inner.covariance_comments.clone()
    }

    /// Row 7 of the 9x9 RTN covariance lower triangle, `CDRG_R` to
    /// `CDRG_DRG`, as a numpy `(7,)` array, or `None` when not given.
    #[getter]
    fn drag_covariance_rtn<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .drag_covariance_rtn
            .map(|values| PyArray1::from_slice(py, &values))
    }

    /// Row 8, `CSRP_R` to `CSRP_SRP`, as a numpy `(8,)` array, or `None`.
    #[getter]
    fn srp_covariance_rtn<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .srp_covariance_rtn
            .map(|values| PyArray1::from_slice(py, &values))
    }

    /// Row 9, `CTHR_R` to `CTHR_THR`, as a numpy `(9,)` array, or `None`.
    #[getter]
    fn thrust_covariance_rtn<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .thrust_covariance_rtn
            .map(|values| PyArray1::from_slice(py, &values))
    }

    /// The RTN covariance as the symmetric matrix of the rows the object
    /// holds, a numpy array of shape `(3, 3)`, `(6, 6)` or `(7, 7)` to
    /// `(9, 9)`, validated positive semidefinite. The rows are held as the
    /// message states them; raises `CdmParseError` when the matrix is not
    /// positive semidefinite or a row is given while a row before it is
    /// absent.
    fn to_covariance_rtn<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let rows = self
            .inner
            .to_covariance_rtn()
            .map_err(|err| to_cdm_err(py, err))?;
        PyArray2::from_vec2(py, &rows).map_err(|err| CdmParseError::new_err(err.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "CdmObject(object_name={:?}, ref_frame={:?})",
            self.inner.object_name, self.inner.ref_frame
        )
    }

    fn __eq__(&self, other: &PyCdmObject) -> bool {
        self == other
    }
}

/// A two-object CCSDS Conjunction Data Message parsed from KVN or XML.
#[pyclass(module = "sidereon._sidereon", name = "Cdm")]
#[derive(Clone, PartialEq)]
pub struct PyCdm {
    inner: CdmKvn,
}

impl PyCdm {
    fn from_inner(inner: CdmKvn) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCdm {
    #[new]
    #[pyo3(signature = (
        object1,
        object2,
        *,
        creation_date=None,
        originator=None,
        message_id=None,
        tca=None,
        miss_distance_m=None,
        relative_speed_m_s=None,
        collision_probability=None,
        collision_probability_method=None,
        hard_body_radius_m=None,
        ccsds_cdm_vers=None,
        comments=Vec::new(),
        message_for=None,
        relative_comments=Vec::new(),
        relative_position_rtn_m=(None, None, None),
        relative_velocity_rtn_m_s=(None, None, None),
        start_screen_period=None,
        stop_screen_period=None,
        screen_volume_frame=None,
        screen_volume_shape=None,
        screen_volume_m=(None, None, None),
        screen_entry_time=None,
        screen_exit_time=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        object1: PyCdmObject,
        object2: PyCdmObject,
        creation_date: Option<String>,
        originator: Option<String>,
        message_id: Option<String>,
        tca: Option<String>,
        miss_distance_m: Option<f64>,
        relative_speed_m_s: Option<f64>,
        collision_probability: Option<f64>,
        collision_probability_method: Option<String>,
        hard_body_radius_m: Option<f64>,
        ccsds_cdm_vers: Option<String>,
        comments: Vec<String>,
        message_for: Option<String>,
        relative_comments: Vec<String>,
        relative_position_rtn_m: OptionalTriple,
        relative_velocity_rtn_m_s: OptionalTriple,
        start_screen_period: Option<String>,
        stop_screen_period: Option<String>,
        screen_volume_frame: Option<String>,
        screen_volume_shape: Option<String>,
        screen_volume_m: OptionalTriple,
        screen_entry_time: Option<String>,
        screen_exit_time: Option<String>,
    ) -> Self {
        Self {
            inner: CdmKvn {
                ccsds_cdm_vers,
                comments,
                creation_date,
                originator,
                message_for,
                message_id,
                relative_comments,
                tca,
                miss_distance_m,
                relative_speed_m_s,
                relative_position_rtn_m: from_triple(relative_position_rtn_m),
                relative_velocity_rtn_m_s: from_triple(relative_velocity_rtn_m_s),
                start_screen_period,
                stop_screen_period,
                screen_volume_frame,
                screen_volume_shape,
                screen_volume_m: from_triple(screen_volume_m),
                screen_entry_time,
                screen_exit_time,
                collision_probability,
                collision_probability_method,
                hard_body_radius_m,
                object1: object1.inner,
                object2: object2.inner,
            },
        }
    }

    #[getter]
    fn creation_date(&self) -> Option<String> {
        self.inner.creation_date.clone()
    }

    #[getter]
    fn originator(&self) -> Option<String> {
        self.inner.originator.clone()
    }

    #[getter]
    fn message_id(&self) -> Option<String> {
        self.inner.message_id.clone()
    }

    #[getter]
    fn tca(&self) -> Option<String> {
        self.inner.tca.clone()
    }

    #[getter]
    fn miss_distance_m(&self) -> Option<f64> {
        self.inner.miss_distance_m
    }

    #[getter]
    fn relative_speed_m_s(&self) -> Option<f64> {
        self.inner.relative_speed_m_s
    }

    #[getter]
    fn collision_probability(&self) -> Option<f64> {
        self.inner.collision_probability
    }

    #[getter]
    fn collision_probability_method(&self) -> Option<String> {
        self.inner.collision_probability_method.clone()
    }

    #[getter]
    fn hard_body_radius_m(&self) -> Option<f64> {
        self.inner.hard_body_radius_m
    }

    /// `CCSDS_CDM_VERS` as the message states it, `None` when it states
    /// none; the writers then state none.
    #[getter]
    fn ccsds_cdm_vers(&self) -> Option<String> {
        self.inner.ccsds_cdm_vers.clone()
    }

    /// Header comments.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// `MESSAGE_FOR` text.
    #[getter]
    fn message_for(&self) -> Option<String> {
        self.inner.message_for.clone()
    }

    /// Relative metadata/data comments.
    #[getter]
    fn relative_comments(&self) -> Vec<String> {
        self.inner.relative_comments.clone()
    }

    /// `RELATIVE_POSITION_R/T/N` in metres, each optional on its own.
    #[getter]
    fn relative_position_rtn_m(&self) -> OptionalTriple {
        triple(self.inner.relative_position_rtn_m)
    }

    /// `RELATIVE_VELOCITY_R/T/N` in metres per second, each optional.
    #[getter]
    fn relative_velocity_rtn_m_s(&self) -> OptionalTriple {
        triple(self.inner.relative_velocity_rtn_m_s)
    }

    /// Raw `START_SCREEN_PERIOD` text.
    #[getter]
    fn start_screen_period(&self) -> Option<String> {
        self.inner.start_screen_period.clone()
    }

    /// Raw `STOP_SCREEN_PERIOD` text.
    #[getter]
    fn stop_screen_period(&self) -> Option<String> {
        self.inner.stop_screen_period.clone()
    }

    /// `SCREEN_VOLUME_FRAME` text.
    #[getter]
    fn screen_volume_frame(&self) -> Option<String> {
        self.inner.screen_volume_frame.clone()
    }

    /// `SCREEN_VOLUME_SHAPE` text.
    #[getter]
    fn screen_volume_shape(&self) -> Option<String> {
        self.inner.screen_volume_shape.clone()
    }

    /// `SCREEN_VOLUME_X/Y/Z` in metres, each optional on its own.
    #[getter]
    fn screen_volume_m(&self) -> OptionalTriple {
        triple(self.inner.screen_volume_m)
    }

    /// Raw `SCREEN_ENTRY_TIME` text.
    #[getter]
    fn screen_entry_time(&self) -> Option<String> {
        self.inner.screen_entry_time.clone()
    }

    /// Raw `SCREEN_EXIT_TIME` text.
    #[getter]
    fn screen_exit_time(&self) -> Option<String> {
        self.inner.screen_exit_time.clone()
    }

    #[getter]
    fn object1(&self) -> PyCdmObject {
        PyCdmObject::from_inner(self.inner.object1.clone())
    }

    #[getter]
    fn object2(&self) -> PyCdmObject {
        PyCdmObject::from_inner(self.inner.object2.clone())
    }

    /// Encode this message to CCSDS CDM KVN text via the core writer.
    fn to_kvn_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_kvn(&self.inner).map_err(|err| to_cdm_err(py, err))
    }

    /// Encode this message to CCSDS CDM XML text via the core writer.
    fn to_xml_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_xml(&self.inner).map_err(|err| to_cdm_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "Cdm(message_id={:?}, object1={:?}, object2={:?})",
            self.inner.message_id, self.inner.object1.object_name, self.inner.object2.object_name
        )
    }

    fn __eq__(&self, other: &PyCdm) -> bool {
        self == other
    }
}

/// Parse CCSDS CDM KVN text.
#[pyfunction]
fn parse_cdm_kvn(py: Python<'_>, text: &str) -> PyResult<PyCdm> {
    parse_kvn(text)
        .map(PyCdm::from_inner)
        .map_err(|err| to_cdm_err(py, err))
}

/// Parse CCSDS CDM XML text.
#[pyfunction]
fn parse_cdm_xml(py: Python<'_>, text: &str) -> PyResult<PyCdm> {
    parse_xml(text)
        .map(PyCdm::from_inner)
        .map_err(|err| to_cdm_err(py, err))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.getattr("CdmParseError")?
        .setattr("detail", m.py().None())?;
    m.add_class::<PyCdmOdParameters>()?;
    m.add_class::<PyCdmAdditionalParameters>()?;
    m.add_class::<PyCdmObject>()?;
    m.add_class::<PyCdm>()?;
    m.add_function(wrap_pyfunction!(parse_cdm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cdm_xml, m)?)?;
    Ok(())
}
