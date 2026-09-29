//! CCSDS OPM binding.
//!
//! Provides typed Python value objects for the core `Opm` metadata/state/
//! Keplerian/spacecraft/covariance/maneuver structs and parse/encode entry
//! points for KVN and XML. The grammar and serialization stay entirely in
//! `sidereon-core`; this module only marshals strings, optional fields, and
//! numpy vectors. It mirrors the sibling CDM and OMM bindings: parsed messages
//! round-trip through `to_kvn_string` / `to_xml_string`, and the same value
//! objects are constructible from Python.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon_core::astro::ndm::TextIssue;
use sidereon_core::astro::opm::{
    encode_kvn, encode_xml, parse_kvn, parse_xml, Opm, OpmAnomaly, OpmCovariance, OpmError,
    OpmInputErrorKind, OpmKeplerian, OpmManeuver, OpmMetadata, OpmSpacecraft, OpmState,
    OpmUserDefined,
};

use crate::marshal::{
    covariance6_error, covariance6_to_array, fixed_array, EqualityHasher, FinitePolicy,
};
use crate::{np_array, OpmParseError};

fn opm_input_kind(kind: OpmInputErrorKind) -> &'static str {
    match kind {
        OpmInputErrorKind::Missing => "Missing",
        OpmInputErrorKind::NonFinite => "NonFinite",
        OpmInputErrorKind::FloatParse => "FloatParse",
        OpmInputErrorKind::IntParse => "IntParse",
        OpmInputErrorKind::NotPositive => "NotPositive",
        OpmInputErrorKind::Negative => "Negative",
        OpmInputErrorKind::OutOfRange => "OutOfRange",
        OpmInputErrorKind::InvalidCivilDate => "InvalidCivilDate",
        OpmInputErrorKind::InvalidCivilTime => "InvalidCivilTime",
    }
}

fn opm_detail<'py>(py: Python<'py>, err: &OpmError) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "opm")?;
    let kind = match err {
        OpmError::MissingField(field) => {
            detail.set_item("field", field)?;
            "MissingField"
        }
        OpmError::InvalidField { field, kind } => {
            detail.set_item("field", field)?;
            detail.set_item("validation_kind", opm_input_kind(*kind))?;
            "InvalidField"
        }
        OpmError::Field(message) => {
            detail.set_item("message", message)?;
            "Field"
        }
        OpmError::DuplicateField {
            field,
            first,
            second,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("first", first)?;
            detail.set_item("second", second)?;
            "DuplicateField"
        }
        OpmError::UnitMismatch {
            field,
            unit,
            expected,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("unit", unit)?;
            detail.set_item("expected", expected)?;
            "UnitMismatch"
        }
        OpmError::MultipleMessages { count } => {
            detail.set_item("count", *count)?;
            "MultipleMessages"
        }
        OpmError::UnknownField(field) => {
            detail.set_item("field", field)?;
            "UnknownField"
        }
        OpmError::MalformedLine { line, text } => {
            detail.set_item("line", *line)?;
            detail.set_item("text", text)?;
            "MalformedLine"
        }
        OpmError::UnwritableText {
            field,
            value,
            issue,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("value", value)?;
            let issue = match issue {
                TextIssue::LineBreak => "LineBreak",
                TextIssue::SurroundingWhitespace => "SurroundingWhitespace",
                TextIssue::InteriorWhitespace => "InteriorWhitespace",
                TextIssue::KeywordSeparator => "KeywordSeparator",
                TextIssue::XmlIllegalCharacter => "XmlIllegalCharacter",
                TextIssue::Empty => "Empty",
                TextIssue::DetachedComment => "DetachedComment",
                TextIssue::RepeatedParameter => "RepeatedParameter",
                TextIssue::CommentNotCarried => "CommentNotCarried",
            };
            detail.set_item("issue", issue)?;
            "UnwritableText"
        }
    };
    detail.set_item("kind", kind)?;
    Ok(detail)
}

fn to_opm_err(py: Python<'_>, err: OpmError) -> PyErr {
    let py_err = OpmParseError::new_err(err.to_string());
    let detail = match opm_detail(py, &err) {
        Ok(value) => value,
        Err(error) => return error,
    };
    if let Err(error) = py_err.value(py).setattr("detail", detail) {
        return error;
    }
    py_err
}

fn write_opm_metadata(hasher: &mut EqualityHasher, value: &OpmMetadata) {
    hasher.begin("OpmMetadata");
    hasher.field("comments", &value.comments);
    hasher.field("object_name", &value.object_name);
    hasher.field("object_id", &value.object_id);
    hasher.field("center_name", &value.center_name);
    hasher.field("ref_frame", &value.ref_frame);
    hasher.field("ref_frame_epoch", &value.ref_frame_epoch);
    hasher.field("time_system", &value.time_system);
}

fn write_opm_state(hasher: &mut EqualityHasher, value: &OpmState) {
    hasher.begin("OpmState");
    hasher.field("comments", &value.comments);
    hasher.field("epoch", &value.epoch);
    hasher.float_array("position_km", &value.position_km);
    hasher.float_array("velocity_km_s", &value.velocity_km_s);
}

fn write_opm_keplerian(hasher: &mut EqualityHasher, value: &OpmKeplerian) {
    hasher.begin("OpmKeplerian");
    hasher.field("comments", &value.comments);
    hasher.float_field("semi_major_axis_km", value.semi_major_axis_km);
    hasher.float_field("eccentricity", value.eccentricity);
    hasher.float_field("inclination_deg", value.inclination_deg);
    hasher.float_field("ra_of_asc_node_deg", value.ra_of_asc_node_deg);
    hasher.float_field("arg_of_pericenter_deg", value.arg_of_pericenter_deg);
    match value.anomaly {
        OpmAnomaly::True(value) => {
            hasher.field("anomaly variant", &"True");
            hasher.float_field("true_anomaly_deg", value);
        }
        OpmAnomaly::Mean(value) => {
            hasher.field("anomaly variant", &"Mean");
            hasher.float_field("mean_anomaly_deg", value);
        }
    }
    hasher.float_field("gm_km3_s2", value.gm_km3_s2);
}

fn write_opm_spacecraft(hasher: &mut EqualityHasher, value: &OpmSpacecraft) {
    hasher.begin("OpmSpacecraft");
    hasher.field("comments", &value.comments);
    hasher.optional_float_field("mass_kg", value.mass_kg);
    hasher.optional_float_field("solar_rad_area_m2", value.solar_rad_area_m2);
    hasher.optional_float_field("solar_rad_coeff", value.solar_rad_coeff);
    hasher.optional_float_field("drag_area_m2", value.drag_area_m2);
    hasher.optional_float_field("drag_coeff", value.drag_coeff);
}

fn write_opm_covariance(hasher: &mut EqualityHasher, value: &OpmCovariance) {
    hasher.begin("OpmCovariance");
    hasher.field("comments", &value.comments);
    hasher.field("cov_ref_frame", &value.cov_ref_frame);
    hasher.float_array("lower_triangle", &value.lower_triangle);
}

fn write_opm_maneuver(hasher: &mut EqualityHasher, value: &OpmManeuver) {
    hasher.begin("OpmManeuver");
    hasher.field("comments", &value.comments);
    hasher.field("epoch_ignition", &value.epoch_ignition);
    hasher.float_field("duration_s", value.duration_s);
    hasher.float_field("delta_mass_kg", value.delta_mass_kg);
    hasher.field("ref_frame", &value.ref_frame);
    hasher.float_array("dv_km_s", &value.dv_km_s);
}

fn write_opm_user_defined(hasher: &mut EqualityHasher, value: &OpmUserDefined) {
    hasher.begin("OpmUserDefined");
    hasher.field("parameter", &value.parameter);
    hasher.field("value", &value.value);
}

fn write_opm(hasher: &mut EqualityHasher, value: &Opm) {
    hasher.begin("Opm");
    hasher.field("ccsds_opm_vers", &value.ccsds_opm_vers);
    hasher.field("comments", &value.comments);
    hasher.field("classification", &value.classification);
    hasher.field("creation_date", &value.creation_date);
    hasher.field("originator", &value.originator);
    hasher.field("message_id", &value.message_id);
    write_opm_metadata(hasher, &value.metadata);
    write_opm_state(hasher, &value.state);
    match &value.keplerian {
        Some(keplerian) => {
            hasher.field("keplerian present", &true);
            write_opm_keplerian(hasher, keplerian);
        }
        None => hasher.field("keplerian present", &false),
    }
    match &value.spacecraft {
        Some(spacecraft) => {
            hasher.field("spacecraft present", &true);
            write_opm_spacecraft(hasher, spacecraft);
        }
        None => hasher.field("spacecraft present", &false),
    }
    match &value.covariance {
        Some(covariance) => {
            hasher.field("covariance present", &true);
            write_opm_covariance(hasher, covariance);
        }
        None => hasher.field("covariance present", &false),
    }
    hasher.field("maneuvers count", &value.maneuvers.len());
    for maneuver in &value.maneuvers {
        write_opm_maneuver(hasher, maneuver);
    }
    hasher.field("user_defined count", &value.user_defined.len());
    for entry in &value.user_defined {
        write_opm_user_defined(hasher, entry);
    }
    hasher.field("user_defined_comments", &value.user_defined_comments);
}

fn hash_opm_metadata(value: &OpmMetadata) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_opm_metadata(&mut hasher, value);
    hasher.finish()
}

fn hash_opm_state(value: &OpmState) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_opm_state(&mut hasher, value);
    hasher.finish()
}

fn hash_opm_keplerian(value: &OpmKeplerian) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_opm_keplerian(&mut hasher, value);
    hasher.finish()
}

fn hash_opm_spacecraft(value: &OpmSpacecraft) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_opm_spacecraft(&mut hasher, value);
    hasher.finish()
}

fn hash_opm_covariance(value: &OpmCovariance) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_opm_covariance(&mut hasher, value);
    hasher.finish()
}

fn hash_opm_maneuver(value: &OpmManeuver) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_opm_maneuver(&mut hasher, value);
    hasher.finish()
}

fn hash_opm(value: &Opm) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_opm(&mut hasher, value);
    hasher.finish()
}

/// OPM metadata block.
#[pyclass(module = "sidereon._sidereon", name = "OpmMetadata")]
#[derive(Clone, PartialEq)]
pub struct PyOpmMetadata {
    inner: OpmMetadata,
}

impl PyOpmMetadata {
    fn from_inner(inner: OpmMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmMetadata {
    #[new]
    #[pyo3(signature = (
        object_name,
        object_id,
        center_name,
        ref_frame,
        time_system,
        *,
        ref_frame_epoch=None,
        comments=Vec::new()
    ))]
    fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: String,
        time_system: String,
        ref_frame_epoch: Option<String>,
        comments: Vec<String>,
    ) -> Self {
        Self {
            inner: OpmMetadata {
                comments,
                object_name,
                object_id,
                center_name,
                ref_frame,
                ref_frame_epoch,
                time_system,
            },
        }
    }

    /// Metadata comments, written before `OBJECT_NAME`.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// `REF_FRAME_EPOCH` text, written only when present.
    #[getter]
    fn ref_frame_epoch(&self) -> Option<String> {
        self.inner.ref_frame_epoch.clone()
    }

    #[getter]
    fn object_name(&self) -> String {
        self.inner.object_name.clone()
    }

    #[getter]
    fn object_id(&self) -> String {
        self.inner.object_id.clone()
    }

    #[getter]
    fn center_name(&self) -> String {
        self.inner.center_name.clone()
    }

    #[getter]
    fn ref_frame(&self) -> String {
        self.inner.ref_frame.clone()
    }

    #[getter]
    fn time_system(&self) -> String {
        self.inner.time_system.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmMetadata(object_name={:?}, ref_frame={:?}, time_system={:?})",
            self.inner.object_name, self.inner.ref_frame, self.inner.time_system
        )
    }

    fn __eq__(&self, other: &PyOpmMetadata) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_opm_metadata(&self.inner)
    }
}

/// OPM Cartesian state vector.
#[pyclass(module = "sidereon._sidereon", name = "OpmState")]
#[derive(Clone, PartialEq)]
pub struct PyOpmState {
    inner: OpmState,
}

impl PyOpmState {
    fn from_inner(inner: OpmState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmState {
    #[new]
    #[pyo3(signature = (epoch, position_km, velocity_km_s, *, comments=Vec::new()))]
    fn new(
        epoch: String,
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
        comments: Vec<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OpmState {
                comments,
                epoch,
                position_km: fixed_array::<3>(
                    "position_km",
                    &position_km,
                    FinitePolicy::AllowNonFinite,
                )?,
                velocity_km_s: fixed_array::<3>(
                    "velocity_km_s",
                    &velocity_km_s,
                    FinitePolicy::AllowNonFinite,
                )?,
            },
        })
    }

    /// State epoch text exactly as carried by the message.
    #[getter]
    fn epoch(&self) -> String {
        self.inner.epoch.clone()
    }

    /// Position vector as a numpy `(3,)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_km)
    }

    /// Velocity vector as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_km_s)
    }

    /// State vector comments, written before `EPOCH`.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    fn __repr__(&self) -> String {
        format!("OpmState(epoch={:?})", self.inner.epoch)
    }

    fn __eq__(&self, other: &PyOpmState) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_opm_state(&self.inner)
    }
}

/// Optional OPM Keplerian elements.
///
/// Exactly one of `true_anomaly_deg` / `mean_anomaly_deg` carries the orbit
/// position angle; the other reads back as `None`.
#[pyclass(module = "sidereon._sidereon", name = "OpmKeplerian")]
#[derive(Clone, PartialEq)]
pub struct PyOpmKeplerian {
    inner: OpmKeplerian,
}

impl PyOpmKeplerian {
    fn from_inner(inner: OpmKeplerian) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmKeplerian {
    #[new]
    #[pyo3(signature = (
        semi_major_axis_km,
        eccentricity,
        inclination_deg,
        ra_of_asc_node_deg,
        arg_of_pericenter_deg,
        gm_km3_s2,
        *,
        true_anomaly_deg=None,
        mean_anomaly_deg=None,
        comments=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        semi_major_axis_km: f64,
        eccentricity: f64,
        inclination_deg: f64,
        ra_of_asc_node_deg: f64,
        arg_of_pericenter_deg: f64,
        gm_km3_s2: f64,
        true_anomaly_deg: Option<f64>,
        mean_anomaly_deg: Option<f64>,
        comments: Vec<String>,
    ) -> PyResult<Self> {
        let anomaly = match (true_anomaly_deg, mean_anomaly_deg) {
            (Some(true_deg), None) => OpmAnomaly::True(true_deg),
            (None, Some(mean_deg)) => OpmAnomaly::Mean(mean_deg),
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "set exactly one of true_anomaly_deg or mean_anomaly_deg, not both",
                ))
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "set exactly one of true_anomaly_deg or mean_anomaly_deg",
                ))
            }
        };
        Ok(Self {
            inner: OpmKeplerian {
                comments,
                semi_major_axis_km,
                eccentricity,
                inclination_deg,
                ra_of_asc_node_deg,
                arg_of_pericenter_deg,
                anomaly,
                gm_km3_s2,
            },
        })
    }

    #[getter]
    fn semi_major_axis_km(&self) -> f64 {
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

    /// True anomaly in degrees, or `None` when the message carries mean anomaly.
    #[getter]
    fn true_anomaly_deg(&self) -> Option<f64> {
        match self.inner.anomaly {
            OpmAnomaly::True(value) => Some(value),
            OpmAnomaly::Mean(_) => None,
        }
    }

    /// Mean anomaly in degrees, or `None` when the message carries true anomaly.
    #[getter]
    fn mean_anomaly_deg(&self) -> Option<f64> {
        match self.inner.anomaly {
            OpmAnomaly::Mean(value) => Some(value),
            OpmAnomaly::True(_) => None,
        }
    }

    #[getter]
    fn gm_km3_s2(&self) -> f64 {
        self.inner.gm_km3_s2
    }

    /// Keplerian block comments, written before `SEMI_MAJOR_AXIS`.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmKeplerian(semi_major_axis_km={}, eccentricity={})",
            self.inner.semi_major_axis_km, self.inner.eccentricity
        )
    }

    fn __eq__(&self, other: &PyOpmKeplerian) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_opm_keplerian(&self.inner)
    }
}

/// Optional OPM spacecraft parameters. Every field is independently optional.
#[pyclass(module = "sidereon._sidereon", name = "OpmSpacecraft")]
#[derive(Clone, PartialEq)]
pub struct PyOpmSpacecraft {
    inner: OpmSpacecraft,
}

impl PyOpmSpacecraft {
    fn from_inner(inner: OpmSpacecraft) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmSpacecraft {
    #[new]
    #[pyo3(signature = (
        *,
        mass_kg=None,
        solar_rad_area_m2=None,
        solar_rad_coeff=None,
        drag_area_m2=None,
        drag_coeff=None,
        comments=Vec::new()
    ))]
    fn new(
        mass_kg: Option<f64>,
        solar_rad_area_m2: Option<f64>,
        solar_rad_coeff: Option<f64>,
        drag_area_m2: Option<f64>,
        drag_coeff: Option<f64>,
        comments: Vec<String>,
    ) -> Self {
        Self {
            inner: OpmSpacecraft {
                comments,
                mass_kg,
                solar_rad_area_m2,
                solar_rad_coeff,
                drag_area_m2,
                drag_coeff,
            },
        }
    }

    #[getter]
    fn mass_kg(&self) -> Option<f64> {
        self.inner.mass_kg
    }

    #[getter]
    fn solar_rad_area_m2(&self) -> Option<f64> {
        self.inner.solar_rad_area_m2
    }

    #[getter]
    fn solar_rad_coeff(&self) -> Option<f64> {
        self.inner.solar_rad_coeff
    }

    #[getter]
    fn drag_area_m2(&self) -> Option<f64> {
        self.inner.drag_area_m2
    }

    #[getter]
    fn drag_coeff(&self) -> Option<f64> {
        self.inner.drag_coeff
    }

    /// Spacecraft block comments, written before the first parameter.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    fn __repr__(&self) -> String {
        format!("OpmSpacecraft(mass_kg={:?})", self.inner.mass_kg)
    }

    fn __eq__(&self, other: &PyOpmSpacecraft) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_opm_spacecraft(&self.inner)
    }
}

/// Optional OPM 6x6 state covariance and its reference frame, the 21
/// lower-triangle values held exactly as read.
#[pyclass(module = "sidereon._sidereon", name = "OpmCovariance")]
#[derive(Clone, PartialEq)]
pub struct PyOpmCovariance {
    inner: OpmCovariance,
}

impl PyOpmCovariance {
    fn from_inner(inner: OpmCovariance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmCovariance {
    /// Build from the 21 lower-triangle values in keyword order `CX_X`,
    /// `CY_X`, `CY_Y` ... `CZ_DOT_Z_DOT` (km^2, km^2/s, km^2/s^2). Each value
    /// must be finite; the matrix is not checked for positive
    /// semidefiniteness, which `to_covariance6` checks.
    #[new]
    #[pyo3(signature = (lower_triangle, *, cov_ref_frame=None, comments=Vec::new()))]
    fn new(
        lower_triangle: PyReadonlyArray1<'_, f64>,
        cov_ref_frame: Option<String>,
        comments: Vec<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OpmCovariance {
                comments,
                cov_ref_frame,
                lower_triangle: fixed_array::<21>(
                    "lower_triangle",
                    &lower_triangle,
                    FinitePolicy::RequireFinite,
                )?,
            },
        })
    }

    /// The 21 lower-triangle values exactly as read, as a numpy `(21,)`
    /// array in keyword order `CX_X`, `CY_X`, `CY_Y` ... `CZ_DOT_Z_DOT`.
    #[getter]
    fn lower_triangle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.lower_triangle)
    }

    /// Covariance block comments.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// The symmetric matrix as a numpy `(6, 6)` array for `[r, v]` in km and
    /// km/s, validated symmetric positive semidefinite; raises `ValueError`
    /// when it is not.
    fn to_covariance6<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let covariance = self
            .inner
            .to_covariance6()
            .map_err(|err| covariance6_error("covariance", err))?;
        Ok(covariance6_to_array(py, &covariance))
    }

    /// Covariance reference frame, or `None` when not stated.
    #[getter]
    fn cov_ref_frame(&self) -> Option<String> {
        self.inner.cov_ref_frame.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmCovariance(cov_ref_frame={:?})",
            self.inner.cov_ref_frame
        )
    }

    fn __eq__(&self, other: &PyOpmCovariance) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_opm_covariance(&self.inner)
    }
}

/// One OPM maneuver block. Every field is mandatory in CCSDS 502.0-B when a
/// maneuver is present.
#[pyclass(module = "sidereon._sidereon", name = "OpmManeuver")]
#[derive(Clone, PartialEq)]
pub struct PyOpmManeuver {
    inner: OpmManeuver,
}

impl PyOpmManeuver {
    fn from_inner(inner: OpmManeuver) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmManeuver {
    #[new]
    #[pyo3(signature = (epoch_ignition, duration_s, delta_mass_kg, ref_frame, dv_km_s, *, comments=Vec::new()))]
    fn new(
        epoch_ignition: String,
        duration_s: f64,
        delta_mass_kg: f64,
        ref_frame: String,
        dv_km_s: PyReadonlyArray1<'_, f64>,
        comments: Vec<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OpmManeuver {
                comments,
                epoch_ignition,
                duration_s,
                delta_mass_kg,
                ref_frame,
                dv_km_s: fixed_array::<3>("dv_km_s", &dv_km_s, FinitePolicy::AllowNonFinite)?,
            },
        })
    }

    /// Maneuver ignition epoch text exactly as carried by the message.
    #[getter]
    fn epoch_ignition(&self) -> String {
        self.inner.epoch_ignition.clone()
    }

    #[getter]
    fn duration_s(&self) -> f64 {
        self.inner.duration_s
    }

    #[getter]
    fn delta_mass_kg(&self) -> f64 {
        self.inner.delta_mass_kg
    }

    #[getter]
    fn ref_frame(&self) -> String {
        self.inner.ref_frame.clone()
    }

    /// Maneuver delta-v as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn dv_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.dv_km_s)
    }

    /// Maneuver block comments, written before `MAN_EPOCH_IGNITION`.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmManeuver(epoch_ignition={:?}, ref_frame={:?})",
            self.inner.epoch_ignition, self.inner.ref_frame
        )
    }

    fn __eq__(&self, other: &PyOpmManeuver) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_opm_maneuver(&self.inner)
    }
}

/// A CCSDS Orbit Parameter Message parsed from KVN or XML, or built directly.
#[pyclass(module = "sidereon._sidereon", name = "Opm")]
#[derive(Clone, PartialEq)]
pub struct PyOpm {
    inner: Opm,
}

impl PyOpm {
    fn from_inner(inner: Opm) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpm {
    #[new]
    #[pyo3(signature = (
        metadata,
        state,
        *,
        ccsds_opm_vers=None,
        creation_date=None,
        originator=None,
        keplerian=None,
        spacecraft=None,
        covariance=None,
        maneuvers=None,
        comments=Vec::new(),
        classification=None,
        message_id=None,
        user_defined=Vec::new(),
        user_defined_comments=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        metadata: PyOpmMetadata,
        state: PyOpmState,
        ccsds_opm_vers: Option<String>,
        creation_date: Option<String>,
        originator: Option<String>,
        keplerian: Option<PyOpmKeplerian>,
        spacecraft: Option<PyOpmSpacecraft>,
        covariance: Option<PyOpmCovariance>,
        maneuvers: Option<Vec<PyOpmManeuver>>,
        comments: Vec<String>,
        classification: Option<String>,
        message_id: Option<String>,
        user_defined: Vec<(String, String)>,
        user_defined_comments: Vec<String>,
    ) -> Self {
        Self {
            inner: Opm {
                ccsds_opm_vers: ccsds_opm_vers.unwrap_or_else(|| "2.0".to_string()),
                comments,
                classification,
                creation_date,
                originator,
                message_id,
                metadata: metadata.inner,
                state: state.inner,
                keplerian: keplerian.map(|keplerian| keplerian.inner),
                spacecraft: spacecraft.map(|spacecraft| spacecraft.inner),
                covariance: covariance.map(|covariance| covariance.inner),
                maneuvers: maneuvers
                    .unwrap_or_default()
                    .into_iter()
                    .map(|maneuver| maneuver.inner)
                    .collect(),
                user_defined: user_defined
                    .into_iter()
                    .map(|(parameter, value)| OpmUserDefined { parameter, value })
                    .collect(),
                user_defined_comments,
            },
        }
    }

    /// Header comments, written after `CCSDS_OPM_VERS`.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    /// Header `CLASSIFICATION` text.
    #[getter]
    fn classification(&self) -> Option<String> {
        self.inner.classification.clone()
    }

    /// Header `MESSAGE_ID` text.
    #[getter]
    fn message_id(&self) -> Option<String> {
        self.inner.message_id.clone()
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

    /// Comments of the user-defined-parameters block.
    #[getter]
    fn user_defined_comments(&self) -> Vec<String> {
        self.inner.user_defined_comments.clone()
    }

    #[getter]
    fn ccsds_opm_vers(&self) -> String {
        self.inner.ccsds_opm_vers.clone()
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
    fn metadata(&self) -> PyOpmMetadata {
        PyOpmMetadata::from_inner(self.inner.metadata.clone())
    }

    #[getter]
    fn state(&self) -> PyOpmState {
        PyOpmState::from_inner(self.inner.state.clone())
    }

    #[getter]
    fn keplerian(&self) -> Option<PyOpmKeplerian> {
        self.inner.keplerian.clone().map(PyOpmKeplerian::from_inner)
    }

    #[getter]
    fn spacecraft(&self) -> Option<PyOpmSpacecraft> {
        self.inner
            .spacecraft
            .clone()
            .map(PyOpmSpacecraft::from_inner)
    }

    #[getter]
    fn covariance(&self) -> Option<PyOpmCovariance> {
        self.inner
            .covariance
            .clone()
            .map(PyOpmCovariance::from_inner)
    }

    #[getter]
    fn maneuvers(&self) -> Vec<PyOpmManeuver> {
        self.inner
            .maneuvers
            .iter()
            .cloned()
            .map(PyOpmManeuver::from_inner)
            .collect()
    }

    /// Encode this OPM to CCSDS OPM KVN text via the core writer. Raises
    /// `OpmParseError` for a value the KVN reader would not read back
    /// unchanged.
    fn to_kvn_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_kvn(&self.inner).map_err(|err| to_opm_err(py, err))
    }

    /// Encode this OPM to CCSDS OPM XML text via the core writer. Raises
    /// `OpmParseError` for a value the XML reader would not return
    /// unchanged.
    fn to_xml_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_xml(&self.inner).map_err(|err| to_opm_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "Opm(object_name={:?}, maneuvers={})",
            self.inner.metadata.object_name,
            self.inner.maneuvers.len()
        )
    }

    fn __eq__(&self, other: &PyOpm) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_opm(&self.inner)
    }
}

/// Parse CCSDS OPM KVN text.
#[pyfunction]
fn parse_opm_kvn(py: Python<'_>, text: &str) -> PyResult<PyOpm> {
    parse_kvn(text)
        .map(PyOpm::from_inner)
        .map_err(|err| to_opm_err(py, err))
}

/// Parse CCSDS OPM XML text.
#[pyfunction]
fn parse_opm_xml(py: Python<'_>, text: &str) -> PyResult<PyOpm> {
    parse_xml(text)
        .map(PyOpm::from_inner)
        .map_err(|err| to_opm_err(py, err))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.getattr("OpmParseError")?
        .setattr("detail", m.py().None())?;
    m.add_class::<PyOpmMetadata>()?;
    m.add_class::<PyOpmState>()?;
    m.add_class::<PyOpmKeplerian>()?;
    m.add_class::<PyOpmSpacecraft>()?;
    m.add_class::<PyOpmCovariance>()?;
    m.add_class::<PyOpmManeuver>()?;
    m.add_class::<PyOpm>()?;
    m.add_function(wrap_pyfunction!(parse_opm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_opm_xml, m)?)?;
    Ok(())
}
