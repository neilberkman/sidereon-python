//! CCSDS OEM binding.
//!
//! Provides typed Python value objects for the core `Oem` segment/state/
//! covariance structs and parse/encode entry points for KVN and XML. The grammar
//! and serialization stay entirely in `sidereon-core`; this module only marshals
//! strings, optional fields, and numpy vectors. It mirrors the sibling CDM and
//! OMM bindings: parsed messages round-trip through `to_kvn_string` /
//! `to_xml_string`, and the same value objects are constructible from Python.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon_core::astro::ndm::TextIssue;
use sidereon_core::astro::oem::{
    encode_kvn, encode_xml, parse_kvn, parse_xml, Oem, OemComment, OemCovariance, OemError,
    OemInputErrorKind, OemMetadata, OemSegment, OemSkippedState, OemState, OemStateLineError,
};

use crate::marshal::{
    covariance6_error, covariance6_to_array, fixed_array, EqualityHasher, FinitePolicy,
};
use crate::{np_array, OemParseError};

fn oem_input_kind(kind: OemInputErrorKind) -> &'static str {
    match kind {
        OemInputErrorKind::Missing => "Missing",
        OemInputErrorKind::NonFinite => "NonFinite",
        OemInputErrorKind::FloatParse => "FloatParse",
        OemInputErrorKind::IntParse => "IntParse",
        OemInputErrorKind::NotPositive => "NotPositive",
        OemInputErrorKind::Negative => "Negative",
        OemInputErrorKind::OutOfRange => "OutOfRange",
        OemInputErrorKind::InvalidCivilDate => "InvalidCivilDate",
        OemInputErrorKind::InvalidCivilTime => "InvalidCivilTime",
    }
}

fn oem_detail<'py>(py: Python<'py>, err: &OemError) -> PyResult<Bound<'py, PyDict>> {
    let detail = PyDict::new(py);
    detail.set_item("family", "oem")?;
    let kind = match err {
        OemError::MissingField(field) => {
            detail.set_item("field", field)?;
            "MissingField"
        }
        OemError::InvalidField { field, kind } => {
            detail.set_item("field", field)?;
            detail.set_item("validation_kind", oem_input_kind(*kind))?;
            "InvalidField"
        }
        OemError::Field(message) => {
            detail.set_item("message", message)?;
            "Field"
        }
        OemError::DuplicateField {
            field,
            first,
            second,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("first", first)?;
            detail.set_item("second", second)?;
            "DuplicateField"
        }
        OemError::UnitMismatch {
            field,
            unit,
            expected,
        } => {
            detail.set_item("field", field)?;
            detail.set_item("unit", unit)?;
            detail.set_item("expected", expected)?;
            "UnitMismatch"
        }
        OemError::MultipleMessages { count } => {
            detail.set_item("count", *count)?;
            "MultipleMessages"
        }
        OemError::UnknownField(field) => {
            detail.set_item("field", field)?;
            "UnknownField"
        }
        OemError::MalformedLine { line, text } => {
            detail.set_item("line", *line)?;
            detail.set_item("text", text)?;
            "MalformedLine"
        }
        OemError::UnwritableText {
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

fn to_oem_err(py: Python<'_>, err: OemError) -> PyErr {
    let py_err = OemParseError::new_err(err.to_string());
    let detail = match oem_detail(py, &err) {
        Ok(value) => value,
        Err(error) => return error,
    };
    if let Err(error) = py_err.value(py).setattr("detail", detail) {
        return error;
    }
    py_err
}

fn comments_to_tuples(comments: &[OemComment]) -> Vec<(usize, String)> {
    comments
        .iter()
        .map(|comment| (comment.position, comment.text.clone()))
        .collect()
}

fn comments_from_tuples(comments: Vec<(usize, String)>) -> Vec<OemComment> {
    comments
        .into_iter()
        .map(|(position, text)| OemComment { position, text })
        .collect()
}

fn optional_vec3(
    name: &str,
    values: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<Option<[f64; 3]>> {
    match values {
        Some(array) => Ok(Some(fixed_array::<3>(
            name,
            &array,
            FinitePolicy::AllowNonFinite,
        )?)),
        None => Ok(None),
    }
}

fn write_oem_comment(hasher: &mut EqualityHasher, comment: &OemComment) {
    hasher.begin("OemComment");
    hasher.field("position", &comment.position);
    hasher.field("text", &comment.text);
}

fn write_oem_comments(hasher: &mut EqualityHasher, name: &'static str, comments: &[OemComment]) {
    hasher.field(name, &"Vec<OemComment>");
    hasher.field("comment count", &comments.len());
    for comment in comments {
        write_oem_comment(hasher, comment);
    }
}

fn write_oem_metadata(hasher: &mut EqualityHasher, value: &OemMetadata) {
    hasher.begin("OemMetadata");
    hasher.field("comments", &value.comments);
    hasher.field("object_name", &value.object_name);
    hasher.field("object_id", &value.object_id);
    hasher.field("center_name", &value.center_name);
    hasher.field("ref_frame", &value.ref_frame);
    hasher.field("ref_frame_epoch", &value.ref_frame_epoch);
    hasher.field("time_system", &value.time_system);
    hasher.field("start_time", &value.start_time);
    hasher.field("stop_time", &value.stop_time);
    hasher.field("useable_start_time", &value.useable_start_time);
    hasher.field("useable_stop_time", &value.useable_stop_time);
    hasher.field("interpolation", &value.interpolation);
    hasher.field("interpolation_degree", &value.interpolation_degree);
}

fn write_oem_state(hasher: &mut EqualityHasher, value: &OemState) {
    hasher.begin("OemState");
    hasher.field("epoch", &value.epoch);
    hasher.float_array("position_km", &value.position_km);
    hasher.float_array("velocity_km_s", &value.velocity_km_s);
    match value.acceleration_km_s2 {
        Some(acceleration) => {
            hasher.field("acceleration_km_s2 present", &true);
            hasher.float_array("acceleration_km_s2", &acceleration);
        }
        None => hasher.field("acceleration_km_s2 present", &false),
    }
}

fn write_oem_covariance(hasher: &mut EqualityHasher, value: &OemCovariance) {
    hasher.begin("OemCovariance");
    hasher.field("epoch", &value.epoch);
    hasher.field("cov_ref_frame", &value.cov_ref_frame);
    hasher.float_array("lower_triangle", &value.lower_triangle);
}

fn write_oem_segment(hasher: &mut EqualityHasher, value: &OemSegment) {
    hasher.begin("OemSegment");
    write_oem_metadata(hasher, &value.metadata);
    write_oem_comments(hasher, "data_comments", &value.data_comments);
    hasher.field("states count", &value.states.len());
    for state in &value.states {
        write_oem_state(hasher, state);
    }
    write_oem_comments(hasher, "covariance_comments", &value.covariance_comments);
    hasher.field("covariances count", &value.covariances.len());
    for covariance in &value.covariances {
        write_oem_covariance(hasher, covariance);
    }
}

fn write_oem_skipped_state(hasher: &mut EqualityHasher, value: &OemSkippedState) {
    hasher.begin("OemSkippedState");
    hasher.field("line", &value.line);
    hasher.field("segment", &value.segment);
    hasher.field("text", &value.text);
    match &value.reason {
        OemStateLineError::ItemCount(count) => {
            hasher.field("reason", &"ItemCount");
            hasher.field("item_count", count);
        }
        OemStateLineError::InvalidField { field, kind } => {
            hasher.field("reason", &"InvalidField");
            hasher.field("field", field);
            let kind_tag = match kind {
                OemInputErrorKind::Missing => 0_u8,
                OemInputErrorKind::NonFinite => 1,
                OemInputErrorKind::FloatParse => 2,
                OemInputErrorKind::IntParse => 3,
                OemInputErrorKind::NotPositive => 4,
                OemInputErrorKind::Negative => 5,
                OemInputErrorKind::OutOfRange => 6,
                OemInputErrorKind::InvalidCivilDate => 7,
                OemInputErrorKind::InvalidCivilTime => 8,
            };
            hasher.field("kind", &kind_tag);
        }
    }
}

fn hash_oem(value: &Oem) -> u64 {
    let mut hasher = EqualityHasher::new();
    hasher.begin("Oem");
    hasher.field("ccsds_oem_vers", &value.ccsds_oem_vers);
    hasher.field("comments", &value.comments);
    hasher.field("classification", &value.classification);
    hasher.field("creation_date", &value.creation_date);
    hasher.field("originator", &value.originator);
    hasher.field("message_id", &value.message_id);
    hasher.field("segments count", &value.segments.len());
    for segment in &value.segments {
        write_oem_segment(&mut hasher, segment);
    }
    hasher.field("skipped_states count", &value.skipped_states.len());
    for skipped in &value.skipped_states {
        write_oem_skipped_state(&mut hasher, skipped);
    }
    hasher.finish()
}

fn hash_oem_metadata(value: &OemMetadata) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_oem_metadata(&mut hasher, value);
    hasher.finish()
}

fn hash_oem_state(value: &OemState) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_oem_state(&mut hasher, value);
    hasher.finish()
}

fn hash_oem_covariance(value: &OemCovariance) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_oem_covariance(&mut hasher, value);
    hasher.finish()
}

fn hash_oem_segment(value: &OemSegment) -> u64 {
    let mut hasher = EqualityHasher::new();
    write_oem_segment(&mut hasher, value);
    hasher.finish()
}

/// One OEM metadata/data segment's metadata block.
#[pyclass(module = "sidereon._sidereon", name = "OemMetadata")]
#[derive(Clone, PartialEq)]
pub struct PyOemMetadata {
    inner: OemMetadata,
}

impl PyOemMetadata {
    fn from_inner(inner: OemMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemMetadata {
    #[new]
    #[pyo3(signature = (
        object_name,
        object_id,
        center_name,
        ref_frame,
        time_system,
        start_time,
        stop_time,
        *,
        useable_start_time=None,
        useable_stop_time=None,
        interpolation=None,
        interpolation_degree=None,
        ref_frame_epoch=None,
        comments=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: String,
        time_system: String,
        start_time: String,
        stop_time: String,
        useable_start_time: Option<String>,
        useable_stop_time: Option<String>,
        interpolation: Option<String>,
        interpolation_degree: Option<u32>,
        ref_frame_epoch: Option<String>,
        comments: Vec<String>,
    ) -> Self {
        Self {
            inner: OemMetadata {
                comments,
                object_name,
                object_id,
                center_name,
                ref_frame,
                ref_frame_epoch,
                time_system,
                start_time,
                stop_time,
                useable_start_time,
                useable_stop_time,
                interpolation,
                interpolation_degree,
            },
        }
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

    #[getter]
    fn start_time(&self) -> String {
        self.inner.start_time.clone()
    }

    #[getter]
    fn stop_time(&self) -> String {
        self.inner.stop_time.clone()
    }

    #[getter]
    fn useable_start_time(&self) -> Option<String> {
        self.inner.useable_start_time.clone()
    }

    #[getter]
    fn useable_stop_time(&self) -> Option<String> {
        self.inner.useable_stop_time.clone()
    }

    #[getter]
    fn interpolation(&self) -> Option<String> {
        self.inner.interpolation.clone()
    }

    #[getter]
    fn interpolation_degree(&self) -> Option<u32> {
        self.inner.interpolation_degree
    }

    /// `REF_FRAME_EPOCH` text, written only when present.
    #[getter]
    fn ref_frame_epoch(&self) -> Option<String> {
        self.inner.ref_frame_epoch.clone()
    }

    /// Metadata comments.
    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OemMetadata(object_name={:?}, ref_frame={:?}, time_system={:?})",
            self.inner.object_name, self.inner.ref_frame, self.inner.time_system
        )
    }

    fn __eq__(&self, other: &PyOemMetadata) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_oem_metadata(&self.inner)
    }
}

/// One OEM Cartesian state sample.
#[pyclass(module = "sidereon._sidereon", name = "OemState")]
#[derive(Clone, PartialEq)]
pub struct PyOemState {
    inner: OemState,
}

impl PyOemState {
    fn from_inner(inner: OemState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemState {
    #[new]
    #[pyo3(signature = (epoch, position_km, velocity_km_s, *, acceleration_km_s2=None))]
    fn new(
        epoch: String,
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
        acceleration_km_s2: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OemState {
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
                acceleration_km_s2: optional_vec3("acceleration_km_s2", acceleration_km_s2)?,
            },
        })
    }

    /// Epoch text exactly as carried by the message.
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

    /// Acceleration as a numpy `(3,)` array in km/s^2, or `None` when absent.
    #[getter]
    fn acceleration_km_s2<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .acceleration_km_s2
            .map(|acceleration| np_array(py, &acceleration))
    }

    fn __repr__(&self) -> String {
        format!(
            "OemState(epoch={:?}, has_acceleration={})",
            self.inner.epoch,
            self.inner.acceleration_km_s2.is_some()
        )
    }

    fn __eq__(&self, other: &PyOemState) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_oem_state(&self.inner)
    }
}

/// One OEM covariance matrix: an epoch, an optional reference frame, and the
/// 21 lower-triangle values of a 6x6 state covariance held exactly as read.
#[pyclass(module = "sidereon._sidereon", name = "OemCovariance")]
#[derive(Clone, PartialEq)]
pub struct PyOemCovariance {
    inner: OemCovariance,
}

impl PyOemCovariance {
    fn from_inner(inner: OemCovariance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemCovariance {
    /// Build from the 21 lower-triangle values in keyword order `CX_X`,
    /// `CY_X`, `CY_Y` ... `CZ_DOT_Z_DOT`, row by row. Each value must be
    /// finite; the matrix is not checked for positive semidefiniteness, which
    /// `to_covariance6` checks.
    #[new]
    #[pyo3(signature = (epoch, lower_triangle, *, cov_ref_frame=None))]
    fn new(
        epoch: String,
        lower_triangle: PyReadonlyArray1<'_, f64>,
        cov_ref_frame: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OemCovariance {
                epoch,
                cov_ref_frame,
                lower_triangle: fixed_array::<21>(
                    "lower_triangle",
                    &lower_triangle,
                    FinitePolicy::RequireFinite,
                )?,
            },
        })
    }

    /// Covariance epoch text exactly as carried by the message.
    #[getter]
    fn epoch(&self) -> String {
        self.inner.epoch.clone()
    }

    /// Covariance reference frame, or `None` when not stated.
    #[getter]
    fn cov_ref_frame(&self) -> Option<String> {
        self.inner.cov_ref_frame.clone()
    }

    /// The 21 lower-triangle values exactly as read, as a numpy `(21,)`
    /// array.
    #[getter]
    fn lower_triangle<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.lower_triangle)
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

    fn __repr__(&self) -> String {
        format!(
            "OemCovariance(epoch={:?}, cov_ref_frame={:?})",
            self.inner.epoch, self.inner.cov_ref_frame
        )
    }

    fn __eq__(&self, other: &PyOemCovariance) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_oem_covariance(&self.inner)
    }
}

/// One OEM segment: metadata, ephemeris state samples, and covariance blocks.
#[pyclass(module = "sidereon._sidereon", name = "OemSegment")]
#[derive(Clone, PartialEq)]
pub struct PyOemSegment {
    inner: OemSegment,
}

impl PyOemSegment {
    fn from_inner(inner: OemSegment) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOemSegment {
    #[new]
    #[pyo3(signature = (
        metadata,
        states,
        *,
        covariances=None,
        data_comments=Vec::new(),
        covariance_comments=Vec::new()
    ))]
    fn new(
        metadata: PyOemMetadata,
        states: Vec<PyOemState>,
        covariances: Option<Vec<PyOemCovariance>>,
        data_comments: Vec<(usize, String)>,
        covariance_comments: Vec<(usize, String)>,
    ) -> Self {
        Self {
            inner: OemSegment {
                metadata: metadata.inner,
                data_comments: comments_from_tuples(data_comments),
                states: states.into_iter().map(|state| state.inner).collect(),
                covariance_comments: comments_from_tuples(covariance_comments),
                covariances: covariances
                    .unwrap_or_default()
                    .into_iter()
                    .map(|covariance| covariance.inner)
                    .collect(),
            },
        }
    }

    #[getter]
    fn metadata(&self) -> PyOemMetadata {
        PyOemMetadata::from_inner(self.inner.metadata.clone())
    }

    #[getter]
    fn states(&self) -> Vec<PyOemState> {
        self.inner
            .states
            .iter()
            .cloned()
            .map(PyOemState::from_inner)
            .collect()
    }

    /// Comments of the ephemeris data as `(position, text)`, `position`
    /// being the number of state lines before the comment.
    #[getter]
    fn data_comments(&self) -> Vec<(usize, String)> {
        comments_to_tuples(&self.inner.data_comments)
    }

    /// Comments of the covariance data as `(position, text)`, `position`
    /// being the number of covariance matrices before the comment.
    #[getter]
    fn covariance_comments(&self) -> Vec<(usize, String)> {
        comments_to_tuples(&self.inner.covariance_comments)
    }

    #[getter]
    fn covariances(&self) -> Vec<PyOemCovariance> {
        self.inner
            .covariances
            .iter()
            .cloned()
            .map(PyOemCovariance::from_inner)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "OemSegment(object_name={:?}, states={}, covariances={})",
            self.inner.metadata.object_name,
            self.inner.states.len(),
            self.inner.covariances.len()
        )
    }

    fn __eq__(&self, other: &PyOemSegment) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_oem_segment(&self.inner)
    }
}

/// A KVN ephemeris data line the forgiving OEM reader skipped.
#[pyclass(module = "sidereon._sidereon", name = "OemSkippedState")]
#[derive(Clone, PartialEq)]
pub struct PyOemSkippedState {
    inner: OemSkippedState,
}

#[pymethods]
impl PyOemSkippedState {
    /// One-based line number in the input.
    #[getter]
    fn line(&self) -> usize {
        self.inner.line
    }

    /// Zero-based index of the segment whose data section holds the line.
    #[getter]
    fn segment(&self) -> usize {
        self.inner.segment
    }

    /// The line text with surrounding whitespace removed.
    #[getter]
    fn text(&self) -> String {
        self.inner.text.clone()
    }

    /// Why the line was skipped: `item_count` or `invalid_field`.
    #[getter]
    fn reason(&self) -> &'static str {
        match self.inner.reason {
            OemStateLineError::ItemCount(_) => "item_count",
            OemStateLineError::InvalidField { .. } => "invalid_field",
        }
    }

    /// The number of items on the line, for an `item_count` reason.
    #[getter]
    fn item_count(&self) -> Option<usize> {
        match self.inner.reason {
            OemStateLineError::ItemCount(count) => Some(count),
            OemStateLineError::InvalidField { .. } => None,
        }
    }

    /// The item that failed (`X`, `Y_DOT`, ...), for an `invalid_field`
    /// reason.
    #[getter]
    fn field(&self) -> Option<&'static str> {
        match self.inner.reason {
            OemStateLineError::InvalidField { field, .. } => Some(field),
            OemStateLineError::ItemCount(_) => None,
        }
    }

    /// The validation failure of an `invalid_field` reason, as the core names
    /// it.
    #[getter]
    fn kind(&self) -> Option<String> {
        match &self.inner.reason {
            OemStateLineError::InvalidField { kind, .. } => Some(format!("{kind:?}")),
            OemStateLineError::ItemCount(_) => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OemSkippedState(line={}, segment={}, reason={:?})",
            self.inner.line,
            self.inner.segment,
            self.reason()
        )
    }

    fn __eq__(&self, other: &PyOemSkippedState) -> bool {
        self == other
    }
}

/// A CCSDS Orbit Ephemeris Message parsed from KVN or XML, or built directly.
#[pyclass(module = "sidereon._sidereon", name = "Oem")]
#[derive(Clone, PartialEq)]
pub struct PyOem {
    inner: Oem,
}

impl PyOem {
    fn from_inner(inner: Oem) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOem {
    #[new]
    #[pyo3(signature = (
        segments,
        *,
        ccsds_oem_vers=None,
        creation_date=None,
        originator=None,
        comments=Vec::new(),
        classification=None,
        message_id=None
    ))]
    fn new(
        segments: Vec<PyOemSegment>,
        ccsds_oem_vers: Option<String>,
        creation_date: Option<String>,
        originator: Option<String>,
        comments: Vec<String>,
        classification: Option<String>,
        message_id: Option<String>,
    ) -> Self {
        Self {
            inner: Oem {
                ccsds_oem_vers: ccsds_oem_vers.unwrap_or_else(|| "2.0".to_string()),
                comments,
                classification,
                creation_date,
                originator,
                message_id,
                segments: segments.into_iter().map(|segment| segment.inner).collect(),
                skipped_states: Vec::new(),
            },
        }
    }

    /// Header comments, written after `CCSDS_OEM_VERS`.
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

    #[getter]
    fn ccsds_oem_vers(&self) -> String {
        self.inner.ccsds_oem_vers.clone()
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
    fn segments(&self) -> Vec<PyOemSegment> {
        self.inner
            .segments
            .iter()
            .cloned()
            .map(PyOemSegment::from_inner)
            .collect()
    }

    /// KVN ephemeris data lines the forgiving reader skipped, in input
    /// order, each with its line number, segment, text and reason.
    #[getter]
    fn skipped_states(&self) -> Vec<PyOemSkippedState> {
        self.inner
            .skipped_states
            .iter()
            .cloned()
            .map(|inner| PyOemSkippedState { inner })
            .collect()
    }

    /// Encode this OEM to CCSDS OEM KVN text via the core writer. Raises
    /// `OemParseError` for a value the KVN reader would not read back
    /// unchanged.
    fn to_kvn_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_kvn(&self.inner).map_err(|err| to_oem_err(py, err))
    }

    /// Encode this OEM to CCSDS OEM XML text via the core writer. Raises
    /// `OemParseError` for a value the XML reader would not return
    /// unchanged.
    fn to_xml_string(&self, py: Python<'_>) -> PyResult<String> {
        encode_xml(&self.inner).map_err(|err| to_oem_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "Oem(ccsds_oem_vers={:?}, segments={})",
            self.inner.ccsds_oem_vers,
            self.inner.segments.len()
        )
    }

    fn __eq__(&self, other: &PyOem) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_oem(&self.inner)
    }
}

/// Parse CCSDS OEM KVN text.
#[pyfunction]
fn parse_oem_kvn(py: Python<'_>, text: &str) -> PyResult<PyOem> {
    parse_kvn(text)
        .map(PyOem::from_inner)
        .map_err(|err| to_oem_err(py, err))
}

/// Parse CCSDS OEM XML text.
#[pyfunction]
fn parse_oem_xml(py: Python<'_>, text: &str) -> PyResult<PyOem> {
    parse_xml(text)
        .map(PyOem::from_inner)
        .map_err(|err| to_oem_err(py, err))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.getattr("OemParseError")?
        .setattr("detail", m.py().None())?;
    m.add_class::<PyOemMetadata>()?;
    m.add_class::<PyOemState>()?;
    m.add_class::<PyOemCovariance>()?;
    m.add_class::<PyOemSegment>()?;
    m.add_class::<PyOemSkippedState>()?;
    m.add_class::<PyOem>()?;
    m.add_function(wrap_pyfunction!(parse_oem_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_oem_xml, m)?)?;
    Ok(())
}
