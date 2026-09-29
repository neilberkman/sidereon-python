//! CCSDS OMM binding.
//!
//! Exposes the core canonical OMM container plus KVN/XML/JSON parse and encode
//! functions. The Python layer performs only structural validation and marshals
//! fields into `sidereon-core`; all format grammar and serialization lives in
//! the engine.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon_core::astro::ndm::TextIssue;
use sidereon_core::astro::omm::{
    encode_csv, encode_csv_discarding_comments, encode_json, encode_json_array,
    encode_json_array_discarding_comments, encode_json_discarding_comments, encode_kvn, encode_xml,
    parse_csv, parse_csv_array, parse_json, parse_json_array, parse_kvn, parse_xml, parse_xml_all,
    Omm, OmmArray, OmmComments, OmmCovariance, OmmEpoch, OmmError, OmmInputErrorKind,
    OmmSpacecraft, OmmUserDefined,
};

use crate::OmmParseError;

fn omm_input_kind(kind: OmmInputErrorKind) -> &'static str {
    match kind {
        OmmInputErrorKind::Missing => "Missing",
        OmmInputErrorKind::NonFinite => "NonFinite",
        OmmInputErrorKind::FloatParse => "FloatParse",
        OmmInputErrorKind::IntParse => "IntParse",
        OmmInputErrorKind::NotPositive => "NotPositive",
        OmmInputErrorKind::Negative => "Negative",
        OmmInputErrorKind::OutOfRange => "OutOfRange",
        OmmInputErrorKind::InvalidCivilDate => "InvalidCivilDate",
        OmmInputErrorKind::InvalidCivilTime => "InvalidCivilTime",
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

fn omm_detail_into<'py>(
    py: Python<'py>,
    detail: &Bound<'py, PyDict>,
    err: &OmmError,
) -> PyResult<&'static str> {
    let kind = match err {
        OmmError::MissingField(field) => {
            detail.set_item("field", field)?;
            "MissingField"
        }
        OmmError::InvalidField { field, kind } => {
            detail.set_item("field", field)?;
            detail.set_item("validation_kind", omm_input_kind(*kind))?;
            "InvalidField"
        }
        OmmError::Field(message) => {
            detail.set_item("message", message)?;
            "Field"
        }
        OmmError::Epoch(message) => {
            detail.set_item("message", message)?;
            "Epoch"
        }
        OmmError::DuplicateField {
            field,
            first,
            second,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("first", first)?;
            detail.set_item("second", second)?;
            "DuplicateField"
        }
        OmmError::UnknownField(field) => {
            detail.set_item("field", field)?;
            "UnknownField"
        }
        OmmError::CsvColumnCount { found, expected } => {
            detail.set_item("found", *found)?;
            detail.set_item("expected", *expected)?;
            "CsvColumnCount"
        }
        OmmError::CsvEmptyBlock(block) => {
            detail.set_item("block", block)?;
            "CsvEmptyBlock"
        }
        OmmError::MalformedLine { line, text } => {
            detail.set_item("line", *line)?;
            detail.set_item("text", text)?;
            "MalformedLine"
        }
        OmmError::UnitMismatch {
            field,
            unit,
            expected,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("unit", unit)?;
            detail.set_item("expected", expected)?;
            "UnitMismatch"
        }
        OmmError::MultipleMessages { count } => {
            detail.set_item("count", *count)?;
            "MultipleMessages"
        }
        OmmError::InRecord { index, source } => {
            detail.set_item("index", *index)?;
            let nested = PyDict::new(py);
            nested.set_item("family", "omm")?;
            let nested_kind = omm_detail_into(py, &nested, source)?;
            nested.set_item("kind", nested_kind)?;
            detail.set_item("source", nested)?;
            "InRecord"
        }
        OmmError::CsvColumnOrder { first, second } => {
            detail.set_item("first", first)?;
            detail.set_item("second", second)?;
            "CsvColumnOrder"
        }
        OmmError::IncompatibleMetadata { field, value } => {
            detail.set_item("field", field)?;
            detail.set_item("value", value)?;
            "IncompatibleMetadata"
        }
        OmmError::UnwritableText {
            field,
            value,
            issue,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("value", value)?;
            detail.set_item("issue", text_issue(*issue))?;
            "UnwritableText"
        }
    };
    Ok(kind)
}

fn omm_detail<'py>(py: Python<'py>, err: &OmmError) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "omm")?;
    let kind = omm_detail_into(py, &detail, err)?;
    detail.set_item("kind", kind)?;
    Ok(detail)
}

fn to_omm_err(py: Python<'_>, err: OmmError) -> PyErr {
    let py_err = OmmParseError::new_err(err.to_string());
    let detail = match omm_detail(py, &err) {
        Ok(value) => value,
        Err(error) => return error,
    };
    if let Err(error) = py_err.value(py).setattr("detail", detail) {
        return error;
    }
    py_err
}

fn finite(value: f64, name: &str) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

/// UTC calendar epoch carried by an OMM `EPOCH` field.
#[pyclass(module = "sidereon._sidereon", name = "OmmEpoch")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyOmmEpoch {
    inner: OmmEpoch,
}

impl PyOmmEpoch {
    fn from_inner(inner: OmmEpoch) -> Self {
        Self { inner }
    }

    fn iso8601_string(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second,
            self.inner.microsecond
        )
    }
}

#[pymethods]
impl PyOmmEpoch {
    #[new]
    #[pyo3(signature = (
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        femtosecond=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
        microsecond: u32,
        femtosecond: u32,
    ) -> PyResult<Self> {
        if !(1..=12).contains(&month) {
            return Err(PyValueError::new_err("month must be in 1..=12"));
        }
        if !(1..=31).contains(&day) {
            return Err(PyValueError::new_err("day must be in 1..=31"));
        }
        if hour > 23 {
            return Err(PyValueError::new_err("hour must be in 0..=23"));
        }
        if minute > 59 {
            return Err(PyValueError::new_err("minute must be in 0..=59"));
        }
        if second > 60 {
            return Err(PyValueError::new_err("second must be in 0..=60"));
        }
        if microsecond > 999_999 {
            return Err(PyValueError::new_err("microsecond must be in 0..=999999"));
        }
        if femtosecond > 999_999_999 {
            return Err(PyValueError::new_err(
                "femtosecond must be in 0..=999999999",
            ));
        }
        Ok(Self {
            inner: OmmEpoch {
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
                femtosecond,
            },
        })
    }

    #[getter]
    fn year(&self) -> i32 {
        self.inner.year
    }

    #[getter]
    fn month(&self) -> u32 {
        self.inner.month
    }

    #[getter]
    fn day(&self) -> u32 {
        self.inner.day
    }

    #[getter]
    fn hour(&self) -> u32 {
        self.inner.hour
    }

    #[getter]
    fn minute(&self) -> u32 {
        self.inner.minute
    }

    #[getter]
    fn second(&self) -> u32 {
        self.inner.second
    }

    #[getter]
    fn microsecond(&self) -> u32 {
        self.inner.microsecond
    }

    #[getter]
    fn femtosecond(&self) -> u32 {
        self.inner.femtosecond
    }

    /// ISO-8601 epoch text with microsecond precision.
    #[getter]
    fn iso8601(&self) -> String {
        self.iso8601_string()
    }

    fn __repr__(&self) -> String {
        format!("OmmEpoch({:?})", self.iso8601_string())
    }

    fn __eq__(&self, other: &PyOmmEpoch) -> bool {
        self == other
    }
}

fn finite_opt(value: Option<f64>, name: &str) -> PyResult<Option<f64>> {
    value.map(|v| finite(v, name)).transpose()
}

/// Comments of the OMM blocks whose presence does not depend on optional
/// keywords (CCSDS 502.0-B-3 7.8.8), each in source order.
#[pyclass(module = "sidereon._sidereon", name = "OmmComments")]
#[derive(Clone, PartialEq, Default)]
pub struct PyOmmComments {
    inner: OmmComments,
}

#[pymethods]
impl PyOmmComments {
    #[new]
    #[pyo3(signature = (
        *,
        header=Vec::new(),
        metadata=Vec::new(),
        mean_elements=Vec::new(),
        tle_parameters=Vec::new(),
        user_defined=Vec::new()
    ))]
    fn new(
        header: Vec<String>,
        metadata: Vec<String>,
        mean_elements: Vec<String>,
        tle_parameters: Vec<String>,
        user_defined: Vec<String>,
    ) -> Self {
        Self {
            inner: OmmComments {
                header,
                metadata,
                mean_elements,
                tle_parameters,
                user_defined,
            },
        }
    }

    /// Header comments, written after `CCSDS_OMM_VERS`.
    #[getter]
    fn header(&self) -> Vec<String> {
        self.inner.header.clone()
    }

    /// Metadata comments, written before `OBJECT_NAME`.
    #[getter]
    fn metadata(&self) -> Vec<String> {
        self.inner.metadata.clone()
    }

    /// Mean-elements comments, written before `EPOCH`.
    #[getter]
    fn mean_elements(&self) -> Vec<String> {
        self.inner.mean_elements.clone()
    }

    /// TLE-parameters comments, written before `EPHEMERIS_TYPE`.
    #[getter]
    fn tle_parameters(&self) -> Vec<String> {
        self.inner.tle_parameters.clone()
    }

    /// User-defined-parameters comments.
    #[getter]
    fn user_defined(&self) -> Vec<String> {
        self.inner.user_defined.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OmmComments(header={:?}, metadata={:?}, mean_elements={:?}, tle_parameters={:?}, user_defined={:?})",
            self.inner.header,
            self.inner.metadata,
            self.inner.mean_elements,
            self.inner.tle_parameters,
            self.inner.user_defined
        )
    }

    fn __eq__(&self, other: &PyOmmComments) -> bool {
        self == other
    }
}

/// Optional OMM spacecraft parameters (CCSDS 502.0-B-3 table 4-3).
#[pyclass(module = "sidereon._sidereon", name = "OmmSpacecraft")]
#[derive(Clone, PartialEq, Default)]
pub struct PyOmmSpacecraft {
    inner: OmmSpacecraft,
}

#[pymethods]
impl PyOmmSpacecraft {
    #[new]
    #[pyo3(signature = (
        *,
        comments=Vec::new(),
        mass_kg=None,
        solar_rad_area_m2=None,
        solar_rad_coeff=None,
        drag_area_m2=None,
        drag_coeff=None
    ))]
    fn new(
        comments: Vec<String>,
        mass_kg: Option<f64>,
        solar_rad_area_m2: Option<f64>,
        solar_rad_coeff: Option<f64>,
        drag_area_m2: Option<f64>,
        drag_coeff: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OmmSpacecraft {
                comments,
                mass_kg: finite_opt(mass_kg, "mass_kg")?,
                solar_rad_area_m2: finite_opt(solar_rad_area_m2, "solar_rad_area_m2")?,
                solar_rad_coeff: finite_opt(solar_rad_coeff, "solar_rad_coeff")?,
                drag_area_m2: finite_opt(drag_area_m2, "drag_area_m2")?,
                drag_coeff: finite_opt(drag_coeff, "drag_coeff")?,
            },
        })
    }

    /// Comments at the start of the block.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// `MASS`, kg.
    #[getter]
    fn mass_kg(&self) -> Option<f64> {
        self.inner.mass_kg
    }

    /// `SOLAR_RAD_AREA`, m^2.
    #[getter]
    fn solar_rad_area_m2(&self) -> Option<f64> {
        self.inner.solar_rad_area_m2
    }

    /// `SOLAR_RAD_COEFF`, dimensionless.
    #[getter]
    fn solar_rad_coeff(&self) -> Option<f64> {
        self.inner.solar_rad_coeff
    }

    /// `DRAG_AREA`, m^2.
    #[getter]
    fn drag_area_m2(&self) -> Option<f64> {
        self.inner.drag_area_m2
    }

    /// `DRAG_COEFF`, dimensionless.
    #[getter]
    fn drag_coeff(&self) -> Option<f64> {
        self.inner.drag_coeff
    }

    fn __repr__(&self) -> String {
        format!("OmmSpacecraft(mass_kg={:?})", self.inner.mass_kg)
    }

    fn __eq__(&self, other: &PyOmmSpacecraft) -> bool {
        self == other
    }
}

/// Optional OMM position/velocity covariance (CCSDS 502.0-B-3 table 4-3),
/// the 21 lower-triangle values kept exactly as read.
#[pyclass(module = "sidereon._sidereon", name = "OmmCovariance")]
#[derive(Clone, PartialEq)]
pub struct PyOmmCovariance {
    inner: OmmCovariance,
}

#[pymethods]
impl PyOmmCovariance {
    #[new]
    #[pyo3(signature = (lower_triangle, *, cov_ref_frame=None, comments=Vec::new()))]
    fn new(
        lower_triangle: [f64; 21],
        cov_ref_frame: Option<String>,
        comments: Vec<String>,
    ) -> Self {
        Self {
            inner: OmmCovariance {
                comments,
                cov_ref_frame,
                lower_triangle,
            },
        }
    }

    /// Comments at the start of the block.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// `COV_REF_FRAME`; when absent the matrix is in `REF_FRAME`.
    #[getter]
    fn cov_ref_frame(&self) -> Option<String> {
        self.inner.cov_ref_frame.clone()
    }

    /// The 21 lower-triangle values in keyword order `CX_X`, `CY_X`, `CY_Y`,
    /// ... `CZ_DOT_Z_DOT` (km^2, km^2/s, km^2/s^2).
    #[getter]
    fn lower_triangle(&self) -> Vec<f64> {
        self.inner.lower_triangle.to_vec()
    }

    fn __repr__(&self) -> String {
        format!(
            "OmmCovariance(cov_ref_frame={:?})",
            self.inner.cov_ref_frame
        )
    }

    fn __eq__(&self, other: &PyOmmCovariance) -> bool {
        self == other
    }
}

/// Canonical, format-agnostic CCSDS Orbit Mean-Elements Message.
///
/// Every TLE related parameter is `None` when the message does not state it;
/// CCSDS 502.0-B-3 table 4-3 requires them only of SGP/SGP4 element sets.
#[pyclass(module = "sidereon._sidereon", name = "Omm")]
#[derive(Clone, PartialEq)]
pub struct PyOmm {
    inner: Omm,
}

impl PyOmm {
    pub(crate) fn from_inner(inner: Omm) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOmm {
    #[new]
    #[pyo3(signature = (
        epoch,
        mean_motion,
        eccentricity,
        inclination_deg,
        ra_of_asc_node_deg,
        arg_of_pericenter_deg,
        mean_anomaly_deg,
        norad_cat_id,
        *,
        ccsds_omm_vers=None,
        creation_date=None,
        originator=None,
        object_name=None,
        object_id=None,
        center_name=None,
        ref_frame=None,
        time_system=None,
        mean_element_theory=None,
        ephemeris_type=None,
        classification_type=None,
        element_set_no=None,
        rev_at_epoch=None,
        bstar=None,
        mean_motion_dot=None,
        mean_motion_ddot=None,
        classification=None,
        message_id=None,
        ref_frame_epoch=None,
        semi_major_axis_km=None,
        gm_km3_s2=None,
        spacecraft=None,
        bterm_m2_kg=None,
        agom_m2_kg=None,
        covariance=None,
        user_defined=Vec::new(),
        comments=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        epoch: PyOmmEpoch,
        mean_motion: Option<f64>,
        eccentricity: f64,
        inclination_deg: f64,
        ra_of_asc_node_deg: f64,
        arg_of_pericenter_deg: f64,
        mean_anomaly_deg: f64,
        norad_cat_id: Option<u32>,
        ccsds_omm_vers: Option<String>,
        creation_date: Option<String>,
        originator: Option<String>,
        object_name: Option<String>,
        object_id: Option<String>,
        center_name: Option<String>,
        ref_frame: Option<String>,
        time_system: Option<String>,
        mean_element_theory: Option<String>,
        ephemeris_type: Option<i32>,
        classification_type: Option<String>,
        element_set_no: Option<i32>,
        rev_at_epoch: Option<i64>,
        bstar: Option<f64>,
        mean_motion_dot: Option<f64>,
        mean_motion_ddot: Option<f64>,
        classification: Option<String>,
        message_id: Option<String>,
        ref_frame_epoch: Option<String>,
        semi_major_axis_km: Option<f64>,
        gm_km3_s2: Option<f64>,
        spacecraft: Option<PyOmmSpacecraft>,
        bterm_m2_kg: Option<f64>,
        agom_m2_kg: Option<f64>,
        covariance: Option<PyOmmCovariance>,
        user_defined: Vec<(String, String)>,
        comments: Option<PyOmmComments>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Omm {
                ccsds_omm_vers,
                classification,
                creation_date,
                originator,
                message_id,
                object_name,
                object_id,
                center_name,
                ref_frame,
                ref_frame_epoch,
                time_system,
                mean_element_theory,
                epoch: epoch.inner,
                mean_motion: finite_opt(mean_motion, "mean_motion")?,
                semi_major_axis_km: finite_opt(semi_major_axis_km, "semi_major_axis_km")?,
                eccentricity: finite(eccentricity, "eccentricity")?,
                inclination_deg: finite(inclination_deg, "inclination_deg")?,
                ra_of_asc_node_deg: finite(ra_of_asc_node_deg, "ra_of_asc_node_deg")?,
                arg_of_pericenter_deg: finite(arg_of_pericenter_deg, "arg_of_pericenter_deg")?,
                mean_anomaly_deg: finite(mean_anomaly_deg, "mean_anomaly_deg")?,
                gm_km3_s2: finite_opt(gm_km3_s2, "gm_km3_s2")?,
                spacecraft: spacecraft.map(|s| s.inner),
                ephemeris_type,
                classification_type,
                norad_cat_id,
                element_set_no,
                rev_at_epoch,
                bstar: finite_opt(bstar, "bstar")?,
                bterm_m2_kg: finite_opt(bterm_m2_kg, "bterm_m2_kg")?,
                mean_motion_dot: finite_opt(mean_motion_dot, "mean_motion_dot")?,
                mean_motion_ddot: finite_opt(mean_motion_ddot, "mean_motion_ddot")?,
                agom_m2_kg: finite_opt(agom_m2_kg, "agom_m2_kg")?,
                covariance: covariance.map(|c| c.inner),
                user_defined: user_defined
                    .into_iter()
                    .map(|(parameter, value)| OmmUserDefined { parameter, value })
                    .collect(),
                comments: comments.map(|c| c.inner).unwrap_or_default(),
                exact_sgp4_epoch: None,
                quantize_tle_derived_fields: true,
            },
        })
    }

    /// `CCSDS_OMM_VERS` as the message states it, `None` when it states
    /// none, as CelesTrak GP JSON and CSV do.
    #[getter]
    fn ccsds_omm_vers(&self) -> Option<String> {
        self.inner.ccsds_omm_vers.clone()
    }

    /// Header `CLASSIFICATION` text, distinct from `classification_type`.
    #[getter]
    fn classification(&self) -> Option<String> {
        self.inner.classification.clone()
    }

    #[getter]
    fn creation_date(&self) -> Option<String> {
        self.inner.creation_date.clone()
    }

    #[getter]
    fn originator(&self) -> Option<String> {
        self.inner.originator.clone()
    }

    /// Header `MESSAGE_ID` text.
    #[getter]
    fn message_id(&self) -> Option<String> {
        self.inner.message_id.clone()
    }

    #[getter]
    fn object_name(&self) -> Option<String> {
        self.inner.object_name.clone()
    }

    #[getter]
    fn object_id(&self) -> Option<String> {
        self.inner.object_id.clone()
    }

    #[getter]
    fn center_name(&self) -> Option<String> {
        self.inner.center_name.clone()
    }

    #[getter]
    fn ref_frame(&self) -> Option<String> {
        self.inner.ref_frame.clone()
    }

    /// `REF_FRAME_EPOCH` text.
    #[getter]
    fn ref_frame_epoch(&self) -> Option<String> {
        self.inner.ref_frame_epoch.clone()
    }

    #[getter]
    fn time_system(&self) -> Option<String> {
        self.inner.time_system.clone()
    }

    #[getter]
    fn mean_element_theory(&self) -> Option<String> {
        self.inner.mean_element_theory.clone()
    }

    #[getter]
    fn epoch(&self) -> PyOmmEpoch {
        PyOmmEpoch::from_inner(self.inner.epoch.clone())
    }

    /// `MEAN_MOTION`, revolutions per day; `None` when the message gives
    /// `SEMI_MAJOR_AXIS` instead.
    #[getter]
    fn mean_motion(&self) -> Option<f64> {
        self.inner.mean_motion
    }

    /// `SEMI_MAJOR_AXIS`, km.
    #[getter]
    fn semi_major_axis_km(&self) -> Option<f64> {
        self.inner.semi_major_axis_km
    }

    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.eccentricity
    }

    #[getter]
    fn inclination_deg(&self) -> f64 {
        self.inner.inclination_deg
    }

    #[getter]
    fn ra_of_asc_node_deg(&self) -> f64 {
        self.inner.ra_of_asc_node_deg
    }

    #[getter]
    fn arg_of_pericenter_deg(&self) -> f64 {
        self.inner.arg_of_pericenter_deg
    }

    #[getter]
    fn mean_anomaly_deg(&self) -> f64 {
        self.inner.mean_anomaly_deg
    }

    /// `GM`, km^3/s^2.
    #[getter]
    fn gm_km3_s2(&self) -> Option<f64> {
        self.inner.gm_km3_s2
    }

    /// Spacecraft-parameters block.
    #[getter]
    fn spacecraft(&self) -> Option<PyOmmSpacecraft> {
        self.inner
            .spacecraft
            .clone()
            .map(|inner| PyOmmSpacecraft { inner })
    }

    #[getter]
    fn ephemeris_type(&self) -> Option<i32> {
        self.inner.ephemeris_type
    }

    #[getter]
    fn classification_type(&self) -> Option<String> {
        self.inner.classification_type.clone()
    }

    #[getter]
    fn norad_cat_id(&self) -> Option<u32> {
        self.inner.norad_cat_id
    }

    #[getter]
    fn element_set_no(&self) -> Option<i32> {
        self.inner.element_set_no
    }

    #[getter]
    fn rev_at_epoch(&self) -> Option<i64> {
        self.inner.rev_at_epoch
    }

    #[getter]
    fn bstar(&self) -> Option<f64> {
        self.inner.bstar
    }

    /// `BTERM`, the SGP4-XP ballistic coefficient, m^2/kg.
    #[getter]
    fn bterm_m2_kg(&self) -> Option<f64> {
        self.inner.bterm_m2_kg
    }

    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> {
        self.inner.mean_motion_dot
    }

    #[getter]
    fn mean_motion_ddot(&self) -> Option<f64> {
        self.inner.mean_motion_ddot
    }

    /// `AGOM`, the SGP4-XP solar radiation pressure coefficient, m^2/kg.
    #[getter]
    fn agom_m2_kg(&self) -> Option<f64> {
        self.inner.agom_m2_kg
    }

    /// Position/velocity covariance block.
    #[getter]
    fn covariance(&self) -> Option<PyOmmCovariance> {
        self.inner
            .covariance
            .clone()
            .map(|inner| PyOmmCovariance { inner })
    }

    /// `USER_DEFINED_*` parameters in source order as `(parameter, value)`,
    /// values verbatim.
    #[getter]
    fn user_defined(&self) -> Vec<(String, String)> {
        self.inner
            .user_defined
            .iter()
            .map(|entry| (entry.parameter.clone(), entry.value.clone()))
            .collect()
    }

    /// Comments of the header, metadata, mean-elements, TLE-parameters and
    /// user-defined blocks.
    #[getter]
    fn comments(&self) -> PyOmmComments {
        PyOmmComments {
            inner: self.inner.comments.clone(),
        }
    }

    /// Encode this OMM to CCSDS OMM KVN text. Raises `OmmParseError` for a
    /// value the KVN reader would not read back unchanged.
    fn to_kvn_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_kvn(&self.inner).map_err(|err| to_omm_err(py, err))
    }

    /// Encode this OMM to CCSDS OMM XML text. Raises `OmmParseError` for a
    /// value the XML reader would not return unchanged.
    fn to_xml_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_xml(&self.inner).map_err(|err| to_omm_err(py, err))
    }

    /// Encode this OMM to CCSDS/CelesTrak GP JSON text. GP JSON carries one
    /// header comment; any other comment raises `OmmParseError`.
    fn to_json_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_json(&self.inner).map_err(|err| to_omm_err(py, err))
    }

    /// Encode this OMM to GP JSON, leaving out the comments GP JSON cannot
    /// carry.
    fn to_json_string_discarding_comments(&self, py: Python<'_>) -> PyResult<String> {
        encode_json_discarding_comments(&self.inner).map_err(|err| to_omm_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "Omm(norad_cat_id={:?}, object_name={:?}, epoch={:?})",
            self.inner.norad_cat_id,
            self.inner.object_name,
            PyOmmEpoch::from_inner(self.inner.epoch.clone()).iso8601_string()
        )
    }

    fn __eq__(&self, other: &PyOmm) -> bool {
        self == other
    }
}

/// A GP JSON or CSV record, or an XML message, a multi-record reader did not
/// read.
#[pyclass(module = "sidereon._sidereon", name = "OmmSkippedRecord")]
#[derive(Clone)]
pub struct PyOmmSkippedRecord {
    index: usize,
    reason: String,
    error: OmmError,
}

#[pymethods]
impl PyOmmSkippedRecord {
    /// Zero-based position of the record among the records of its input.
    #[getter]
    fn index(&self) -> usize {
        self.index
    }

    /// Why the record was not read, as the core states it.
    #[getter]
    fn reason(&self) -> String {
        self.reason.clone()
    }

    /// Structured detail for the core error that caused this record to be skipped.
    #[getter]
    fn detail<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        omm_detail(py, &self.error)
    }

    fn __repr__(&self) -> String {
        format!(
            "OmmSkippedRecord(index={}, reason={:?})",
            self.index, self.reason
        )
    }
}

/// The OMMs a multi-record reader read, and the records it skipped.
#[pyclass(module = "sidereon._sidereon", name = "OmmArray")]
#[derive(Clone)]
pub struct PyOmmArray {
    omms: Vec<Omm>,
    skipped: Vec<PyOmmSkippedRecord>,
}

impl From<OmmArray> for PyOmmArray {
    fn from(array: OmmArray) -> Self {
        Self {
            omms: array.omms,
            skipped: array
                .skipped
                .into_iter()
                .map(|record| {
                    let reason = record.reason.to_string();
                    PyOmmSkippedRecord {
                        index: record.index,
                        reason,
                        error: record.reason,
                    }
                })
                .collect(),
        }
    }
}

#[pymethods]
impl PyOmmArray {
    /// The records read, in input order.
    #[getter]
    fn omms(&self) -> Vec<PyOmm> {
        self.omms.iter().cloned().map(PyOmm::from_inner).collect()
    }

    /// The records skipped, each with its position and reason.
    #[getter]
    fn skipped(&self) -> Vec<PyOmmSkippedRecord> {
        self.skipped.clone()
    }

    fn __len__(&self) -> usize {
        self.omms.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "OmmArray(omms={}, skipped={})",
            self.omms.len(),
            self.skipped.len()
        )
    }
}

fn inner_omms(omms: &[PyRef<'_, PyOmm>]) -> Vec<Omm> {
    omms.iter().map(|omm| omm.inner.clone()).collect()
}

/// Parse CCSDS OMM KVN text.
#[pyfunction]
fn parse_omm_kvn(py: Python<'_>, text: &str) -> PyResult<PyOmm> {
    parse_kvn(text)
        .map(PyOmm::from_inner)
        .map_err(|err| to_omm_err(py, err))
}

/// Parse CCSDS OMM XML text holding one OMM. A document holding several
/// raises `OmmParseError`; `parse_omm_xml_all` reads them all.
#[pyfunction]
fn parse_omm_xml(py: Python<'_>, text: &str) -> PyResult<PyOmm> {
    parse_xml(text)
        .map(PyOmm::from_inner)
        .map_err(|err| to_omm_err(py, err))
}

/// Parse every OMM of a single XML message or an NDM combined
/// instantiation, skipping and reporting a message that does not read.
#[pyfunction]
fn parse_omm_xml_all(py: Python<'_>, text: &str) -> PyResult<PyOmmArray> {
    parse_xml_all(text)
        .map(Into::into)
        .map_err(|err| to_omm_err(py, err))
}

/// Parse CCSDS/CelesTrak GP JSON text holding one record. A document holding
/// several raises `OmmParseError`; `parse_omm_json_array` reads them all.
#[pyfunction]
fn parse_omm_json(py: Python<'_>, text: &str) -> PyResult<PyOmm> {
    parse_json(text)
        .map(PyOmm::from_inner)
        .map_err(|err| to_omm_err(py, err))
}

/// Parse a GP JSON array, skipping and reporting a record that does not read.
#[pyfunction]
fn parse_omm_json_array(py: Python<'_>, text: &str) -> PyResult<PyOmmArray> {
    parse_json_array(text)
        .map(Into::into)
        .map_err(|err| to_omm_err(py, err))
}

/// Parse GP CSV text holding one record.
#[pyfunction]
fn parse_omm_csv(py: Python<'_>, text: &str) -> PyResult<PyOmm> {
    parse_csv(text)
        .map(PyOmm::from_inner)
        .map_err(|err| to_omm_err(py, err))
}

/// Parse GP CSV text, skipping and reporting a record that does not read.
#[pyfunction]
fn parse_omm_csv_array(py: Python<'_>, text: &str) -> PyResult<PyOmmArray> {
    parse_csv_array(text)
        .map(Into::into)
        .map_err(|err| to_omm_err(py, err))
}

/// Encode OMMs as a GP JSON array. A record holding a comment GP JSON cannot
/// carry raises `OmmParseError` naming the record.
#[pyfunction]
fn encode_omm_json_array(py: Python<'_>, omms: Vec<PyRef<'_, PyOmm>>) -> PyResult<String> {
    encode_json_array(&inner_omms(&omms)).map_err(|err| to_omm_err(py, err))
}

/// Encode OMMs as a GP JSON array, leaving out the comments GP JSON cannot
/// carry.
#[pyfunction]
fn encode_omm_json_array_discarding_comments(
    py: Python<'_>,
    omms: Vec<PyRef<'_, PyOmm>>,
) -> PyResult<String> {
    encode_json_array_discarding_comments(&inner_omms(&omms)).map_err(|err| to_omm_err(py, err))
}

/// Encode OMMs as GP CSV. A record `parse_omm_csv_array` would not return
/// unchanged raises `OmmParseError` naming the record.
#[pyfunction]
fn encode_omm_csv(py: Python<'_>, omms: Vec<PyRef<'_, PyOmm>>) -> PyResult<String> {
    encode_csv(&inner_omms(&omms)).map_err(|err| to_omm_err(py, err))
}

/// Encode OMMs as GP CSV, leaving out the comments GP CSV cannot carry.
#[pyfunction]
fn encode_omm_csv_discarding_comments(
    py: Python<'_>,
    omms: Vec<PyRef<'_, PyOmm>>,
) -> PyResult<String> {
    encode_csv_discarding_comments(&inner_omms(&omms)).map_err(|err| to_omm_err(py, err))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.getattr("OmmParseError")?
        .setattr("detail", m.py().None())?;
    m.add_class::<PyOmmEpoch>()?;
    m.add_class::<PyOmmComments>()?;
    m.add_class::<PyOmmSpacecraft>()?;
    m.add_class::<PyOmmCovariance>()?;
    m.add_class::<PyOmm>()?;
    m.add_class::<PyOmmSkippedRecord>()?;
    m.add_class::<PyOmmArray>()?;
    m.add_function(wrap_pyfunction!(parse_omm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_xml, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_xml_all, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_json, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_json_array, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_csv, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_csv_array, m)?)?;
    m.add_function(wrap_pyfunction!(encode_omm_json_array, m)?)?;
    m.add_function(wrap_pyfunction!(
        encode_omm_json_array_discarding_comments,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(encode_omm_csv, m)?)?;
    m.add_function(wrap_pyfunction!(encode_omm_csv_discarding_comments, m)?)?;
    Ok(())
}
