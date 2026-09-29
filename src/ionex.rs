//! IONEX binding: the parsed vertical-TEC grid product, sample reconstruction,
//! slant ionospheric group-delay evaluation with explicit policies, validity masks,
//! and structured diagnostic warnings.
//!
//! Marshals IONEX bytes (or a file path) into the core `Ionex` vertical-TEC grid
//! and exposes its surface Pythonically: the latitude/longitude node axes,
//! paired validity masks for missing cells, header metadata, composite slant
//! policies, batch evaluation results with typed refusal details, and
//! precision-preserving diagnostic warnings. No modeling lives here: the parse is
//! `Ionex::parse_with_warnings` and the delay is `ionex_slant_delay_with_policy`,
//! so the numbers and statuses are exactly what `sidereon-core` produces. The
//! degree-to-radian boundary conversion mirrors the reference order so all
//! interfaces report the exact same value.

use std::f64::consts::PI;
use std::path::PathBuf;

/// Degrees to radians as a single rounded constant `pi/180`, so the boundary
/// conversion is `deg * DEG_TO_RAD` (one multiply, one rounding). This matches
/// the golden's `math.radians` exactly; `(deg * pi) / 180.0` rounds twice and
/// drifts by a ULP at some angles (for example -178 degrees).
const DEG_TO_RAD: f64 = PI / 180.0;

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyArray1, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyModule};

use sidereon_core::astro::time::{
    j2000_seconds_from_split, split_julian_date_from_j2000_seconds, Instant, InstantRepr,
    JulianDateSplit, TimeScale,
};
use sidereon_core::atmosphere::ionosphere::{
    galileo_nequick_g_native as core_galileo_nequick_g_native,
    ionosphere_delay as core_ionosphere_delay, klobuchar_native as core_klobuchar_native,
    nequick_g_delay_m as core_nequick_g_delay_m, nequick_g_stec_tecu as core_nequick_g_stec_tecu,
    GalileoNequickCoeffs, GalileoNequickEval, IonoModel, KlobucharParams, NequickGRayEval,
};
use sidereon_core::atmosphere::{
    ionex_slant_delay_results as core_ionex_slant_delay_results,
    ionex_slant_delay_with_policy as core_ionex_slant_delay_with_policy, Ionex,
    IonexAssumedMapping, IonexCoverageError, IonexCoveragePolicy, IonexHeader,
    IonexMappingDeclaration, IonexMappingFunction, IonexMappingPolicy, IonexMissingNodePolicy,
    IonexMissingNodes, IonexNodeGap, IonexSlantDelayEvaluation, IonexSlantDelayStatus,
    IonexSlantPolicy, IonexSlantRefusal, IonexSlantRequest, IonexWarning, TecGridSamples,
    TecSample,
};
use sidereon_core::Wgs84Geodetic;

use crate::frames::PyTimeScale;
use crate::rinex_clock::PyClockInstant;
use crate::{np_array, to_solve_err};

/// Typed detail for an IONEX epoch conversion refusal.
#[pyclass(module = "sidereon._sidereon", name = "IonexEpochErrorDetail")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyIonexEpochErrorDetail {
    kind: &'static str,
    scale: Option<TimeScale>,
    utc_j2000_s: Option<i64>,
}

impl From<sidereon_core::atmosphere::IonexEpochError> for PyIonexEpochErrorDetail {
    fn from(error: sidereon_core::atmosphere::IonexEpochError) -> Self {
        use sidereon_core::atmosphere::IonexEpochError as E;
        let (kind, scale, utc_j2000_s) = match error {
            E::NotWholeSecond { scale } => ("not_whole_second", Some(scale), None),
            E::FractionalUtcSecond { scale } => ("fractional_utc_second", Some(scale), None),
            E::NoExactUtcOffset { scale } => ("no_exact_utc_offset", Some(scale), None),
            E::InsertedLeapSecond { scale } => ("inserted_leap_second", Some(scale), None),
            E::BeforeIntegerLeapSeconds { scale } => {
                ("before_integer_leap_seconds", Some(scale), None)
            }
            E::OutOfRange { scale } => ("out_of_range", Some(scale), None),
            E::YearOutOfField { utc_j2000_s } => ("year_out_of_field", None, Some(utc_j2000_s)),
            #[allow(unreachable_patterns)]
            _ => ("unknown", None, None),
        };
        Self {
            kind,
            scale,
            utc_j2000_s,
        }
    }
}

#[pymethods]
impl PyIonexEpochErrorDetail {
    #[getter]
    fn kind(&self) -> &'static str {
        self.kind
    }

    #[getter]
    fn scale(&self) -> Option<PyTimeScale> {
        self.scale.map(Into::into)
    }

    #[getter]
    fn utc_j2000_s(&self) -> Option<i64> {
        self.utc_j2000_s
    }
}

pub(crate) fn attach_ionex_epoch_detail(
    py: Python<'_>,
    error: PyErr,
    epoch_error: sidereon_core::atmosphere::IonexEpochError,
) -> PyErr {
    let detail = match PyIonexEpochErrorDetail::from(epoch_error).into_pyobject(py) {
        Ok(detail) => detail,
        Err(error) => return error,
    };
    if let Err(error) = error.value(py).setattr("detail", detail) {
        return error;
    }
    error
}

pub(crate) fn to_ionex_epoch_solve_err(
    py: Python<'_>,
    epoch_error: sidereon_core::atmosphere::IonexEpochError,
) -> PyErr {
    let message = format!("invalid input: {epoch_error}");
    attach_ionex_epoch_detail(py, to_solve_err(message), epoch_error)
}

/// Map an IONEX parse failure into [`IonexParseError`](crate::IonexParseError),
/// preserving the engine message. Both it and the other product parse errors
/// derive from `ParseError`, so callers can catch the product-specific type or
/// the shared base.
fn to_ionex_err<E: std::fmt::Display>(err: E) -> PyErr {
    crate::IonexParseError::new_err(err.to_string())
}

fn to_ionex_write_err(py: Python<'_>, err: sidereon_core::Error) -> PyErr {
    let error_type = match crate::ionex_write_error_type(py) {
        Ok(error_type) => error_type,
        Err(_) => return PyValueError::new_err(err.to_string()),
    };
    let detail = err.to_string();
    let py_error = PyErr::from_type(error_type, detail.clone());
    let _ = py_error.value(py).setattr("detail", detail);
    py_error
}

/// Build a 3-D numpy `float64` array `(epoch, lat, lon)` and a paired `bool`
/// validity mask from core nested `[map][i_lat][i_lon]` grids of `Option<f64>`.
///
/// Missing nodes render as `NaN` and `false` in the validity mask; present nodes
/// render as exact float values and `true` in the validity mask.
fn maps_to_arrays3<'py>(
    py: Python<'py>,
    maps: &[Vec<Vec<Option<f64>>>],
) -> (Bound<'py, PyArray3<f64>>, Bound<'py, PyArray3<bool>>) {
    let n_epoch = maps.len();
    let n_lat = maps.first().map_or(0, Vec::len);
    let n_lon = maps.first().and_then(|m| m.first()).map_or(0, Vec::len);
    let mut float_array = Array3::<f64>::zeros((n_epoch, n_lat, n_lon));
    let mut mask_array = Array3::<bool>::from_elem((n_epoch, n_lat, n_lon), false);
    for (epoch_index, grid) in maps.iter().enumerate() {
        for (lat_index, band) in grid.iter().enumerate() {
            for (lon_index, opt_val) in band.iter().enumerate() {
                match *opt_val {
                    Some(val) => {
                        float_array[[epoch_index, lat_index, lon_index]] = val;
                        mask_array[[epoch_index, lat_index, lon_index]] = true;
                    }
                    None => {
                        float_array[[epoch_index, lat_index, lon_index]] = f64::NAN;
                        mask_array[[epoch_index, lat_index, lon_index]] = false;
                    }
                }
            }
        }
    }
    (float_array.into_pyarray(py), mask_array.into_pyarray(py))
}

/// Convert a 3-D numpy `float64` array `(epoch, lat, lon)` and an optional `bool`
/// validity mask into core nested `[map][i_lat][i_lon]` grids of `Option<f64>`.
///
/// If a mask is supplied, mask dimensions must match value dimensions. The mask
/// decides absence (`false` -> `None`, `true` -> `Some(val)`). A true-masked cell
/// with `NaN` or `Inf` fails validation. If the mask is omitted for compatibility,
/// all values are assumed present and any `NaN` or `Inf` is rejected.
fn maps_from_array3_with_mask(
    values: PyReadonlyArray3<'_, f64>,
    mask: Option<PyReadonlyArray3<'_, bool>>,
) -> PyResult<Vec<Vec<Vec<Option<f64>>>>> {
    let val_view = values.as_array();
    let dims = val_view.dim();

    if let Some(mask_arr) = mask {
        let mask_view = mask_arr.as_array();
        let mask_dims = mask_view.dim();
        if dims != mask_dims {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} does not match values shape {:?}",
                mask_dims, dims
            )));
        }
        let mut maps = vec![vec![vec![None; dims.2]; dims.1]; dims.0];
        for e in 0..dims.0 {
            for i in 0..dims.1 {
                for j in 0..dims.2 {
                    if mask_view[[e, i, j]] {
                        let val = val_view[[e, i, j]];
                        if !val.is_finite() {
                            return Err(PyValueError::new_err(format!(
                                "sample value at [{e}, {i}, {j}] with true validity mask must be finite, got {val}"
                            )));
                        }
                        maps[e][i][j] = Some(val);
                    } else {
                        maps[e][i][j] = None;
                    }
                }
            }
        }
        Ok(maps)
    } else {
        let mut maps = vec![vec![vec![None; dims.2]; dims.1]; dims.0];
        for e in 0..dims.0 {
            for i in 0..dims.1 {
                for j in 0..dims.2 {
                    let val = val_view[[e, i, j]];
                    if !val.is_finite() {
                        return Err(PyValueError::new_err(format!(
                            "sample value at [{e}, {i}, {j}] is not finite ({val}); supply an explicit mask to indicate missing nodes"
                        )));
                    }
                    maps[e][i][j] = Some(val);
                }
            }
        }
        Ok(maps)
    }
}

/// Parse optional 3-D map arrays (`rms_maps` or `height_maps`) with paired validity masks.
///
/// Accepts canonical `(0, 0, 0)` arrays as absent, with a canonical matching empty bool mask
/// if supplied. Rejects partial-empty dimensions, mask shape mismatches, and mask without values.
/// Non-empty arrays must match `tec_dims`.
fn parse_optional_map_array(
    values: Option<PyReadonlyArray3<'_, f64>>,
    mask: Option<PyReadonlyArray3<'_, bool>>,
    tec_dims: (usize, usize, usize),
    name: &'static str,
) -> PyResult<Vec<Vec<Vec<Option<f64>>>>> {
    let Some(val_arr) = values else {
        if mask.is_some() {
            return Err(PyValueError::new_err(format!(
                "cannot supply {name}_mask or {name}_valid when {name}_maps is absent"
            )));
        }
        return Ok(Vec::new());
    };

    let dims = val_arr.as_array().dim();
    if (dims.0 == 0 || dims.1 == 0 || dims.2 == 0) && dims != (0, 0, 0) {
        return Err(PyValueError::new_err(format!(
            "{name}_maps shape {dims:?} has partial empty dimensions; only canonical (0, 0, 0) represents an absent map"
        )));
    }

    if dims == (0, 0, 0) {
        if let Some(ref mask_arr) = mask {
            let mask_dims = mask_arr.as_array().dim();
            if mask_dims != (0, 0, 0) {
                return Err(PyValueError::new_err(format!(
                    "{name}_mask shape {mask_dims:?} does not match {name}_maps shape (0, 0, 0)"
                )));
            }
        }
        return Ok(Vec::new());
    }

    if dims != tec_dims {
        return Err(PyValueError::new_err(format!(
            "{name}_maps shape {dims:?} does not match tec_maps shape {tec_dims:?}"
        )));
    }

    maps_from_array3_with_mask(val_arr, mask)
}

pub(crate) fn ionex_epoch_from_j2000_seconds(seconds: i64) -> PyResult<Instant> {
    let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(seconds);
    let split = JulianDateSplit::new(jd_whole, fraction).map_err(|err| {
        crate::time_model_error::py_error(err, format!("invalid epoch_j2000_s: {err}"))
    })?;
    Ok(Instant::from_julian_date(TimeScale::Utc, split))
}

fn ionex_epoch_to_j2000_seconds(epoch: Instant) -> Option<i64> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            let seconds = j2000_seconds_from_split(split.jd_whole, split.fraction);
            if seconds.is_finite() && seconds >= i64::MIN as f64 && seconds <= i64::MAX as f64 {
                Some(seconds.round() as i64)
            } else {
                None
            }
        }
        InstantRepr::Nanos(nanos) => {
            const NANOS_PER_SECOND: i128 = 1_000_000_000;
            if nanos % NANOS_PER_SECOND != 0 {
                return None;
            }
            i64::try_from(nanos / NANOS_PER_SECOND).ok()
        }
    }
}

fn mapping_function_from_py(obj: &Bound<'_, PyAny>) -> PyResult<IonexMappingFunction> {
    if let Ok(func) = obj.extract::<PyIonexMappingFunction>() {
        return Ok(func.inner);
    }
    if let Ok(code) = obj.extract::<String>() {
        return match code.as_str() {
            "NONE" => Ok(IonexMappingFunction::NoMapping),
            "COSZ" => Ok(IonexMappingFunction::CosZ),
            "QFAC" => Ok(IonexMappingFunction::QFactor),
            _ => Ok(IonexMappingFunction::Other(code)),
        };
    }
    Err(PyValueError::new_err(
        "expected IonexMappingFunction or str",
    ))
}

/// The mapping function an IONEX product declares for its vertical-to-slant TEC determination.
#[pyclass(module = "sidereon._sidereon", name = "IonexMappingFunction")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyIonexMappingFunction {
    pub(crate) inner: IonexMappingFunction,
}

#[pymethods]
impl PyIonexMappingFunction {
    /// Construct a mapping function from code text (`"NONE"`, `"COSZ"`, `"QFAC"`, or custom).
    #[new]
    fn new(code: &str) -> Self {
        let inner = match code {
            "NONE" => IonexMappingFunction::NoMapping,
            "COSZ" => IonexMappingFunction::CosZ,
            "QFAC" => IonexMappingFunction::QFactor,
            other => IonexMappingFunction::Other(other.to_string()),
        };
        Self { inner }
    }

    /// `NONE`: no mapping function was used.
    #[staticmethod]
    fn no_mapping() -> Self {
        Self {
            inner: IonexMappingFunction::NoMapping,
        }
    }

    /// `COSZ`: single-layer `1/cos(z')`.
    #[staticmethod]
    fn cos_z() -> Self {
        Self {
            inner: IonexMappingFunction::CosZ,
        }
    }

    /// `QFAC`: Q-factor.
    #[staticmethod]
    fn q_factor() -> Self {
        Self {
            inner: IonexMappingFunction::QFactor,
        }
    }

    /// Custom mapping function with arbitrary code text.
    #[staticmethod]
    fn other(code: String) -> Self {
        Self {
            inner: IonexMappingFunction::Other(code),
        }
    }

    /// Declared mapping code string (`"NONE"`, `"COSZ"`, `"QFAC"`, or other exact text).
    #[getter]
    fn code(&self) -> &str {
        self.inner.code()
    }

    /// Canonical mapping kind (`"NONE"`, `"COSZ"`, `"QFAC"`, or `"OTHER"`).
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            IonexMappingFunction::NoMapping => "NONE",
            IonexMappingFunction::CosZ => "COSZ",
            IonexMappingFunction::QFactor => "QFAC",
            IonexMappingFunction::Other(_) => "OTHER",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexMappingFunction(code='{}', kind='{}')",
            self.code(),
            self.kind()
        )
    }

    fn __str__(&self) -> &str {
        self.code()
    }

    fn __eq__(&self, other: &PyIonexMappingFunction) -> bool {
        self.inner == other.inner
    }
}

impl From<IonexMappingFunction> for PyIonexMappingFunction {
    fn from(inner: IonexMappingFunction) -> Self {
        Self { inner }
    }
}

/// Metadata describing whether an IONEX product explicitly declared a mapping function.
#[pyclass(module = "sidereon._sidereon", name = "IonexMappingDeclaration")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyIonexMappingDeclaration {
    pub(crate) inner: IonexMappingDeclaration,
}

#[pymethods]
impl PyIonexMappingDeclaration {
    /// True if the header contained a declared `MAPPING FUNCTION` record.
    #[getter]
    fn is_declared(&self) -> bool {
        matches!(self.inner, IonexMappingDeclaration::Declared(_))
    }

    /// True if the header carried no `MAPPING FUNCTION` record.
    #[getter]
    fn is_absent(&self) -> bool {
        matches!(self.inner, IonexMappingDeclaration::Absent)
    }

    /// The declared mapping function, or `None` if absent.
    #[getter]
    fn function(&self) -> Option<PyIonexMappingFunction> {
        match &self.inner {
            IonexMappingDeclaration::Declared(func) => Some(PyIonexMappingFunction {
                inner: func.clone(),
            }),
            IonexMappingDeclaration::Absent => None,
        }
    }

    /// Declared code text, or `None` if absent.
    #[getter]
    fn code(&self) -> Option<String> {
        match &self.inner {
            IonexMappingDeclaration::Declared(func) => Some(func.code().to_string()),
            IonexMappingDeclaration::Absent => None,
        }
    }

    /// `"DECLARED"` or `"ABSENT"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            IonexMappingDeclaration::Declared(_) => "DECLARED",
            IonexMappingDeclaration::Absent => "ABSENT",
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            IonexMappingDeclaration::Declared(func) => {
                format!("IonexMappingDeclaration.Declared(code='{}')", func.code())
            }
            IonexMappingDeclaration::Absent => "IonexMappingDeclaration.Absent".to_string(),
        }
    }

    fn __eq__(&self, other: &PyIonexMappingDeclaration) -> bool {
        self.inner == other.inner
    }
}

impl From<IonexMappingDeclaration> for PyIonexMappingDeclaration {
    fn from(inner: IonexMappingDeclaration) -> Self {
        Self { inner }
    }
}

/// Identifies which mapping function was assumed when mapping with single-layer `1/cos(z')`.
#[pyclass(
    module = "sidereon._sidereon",
    name = "IonexAssumedMapping",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum PyIonexAssumedMapping {
    NO_MAPPING,
    Q_FACTOR,
    OTHER,
    ABSENT,
}

#[pymethods]
impl PyIonexAssumedMapping {
    #[getter]
    fn kind(&self) -> &'static str {
        match self {
            Self::NO_MAPPING => "NO_MAPPING",
            Self::Q_FACTOR => "Q_FACTOR",
            Self::OTHER => "OTHER",
            Self::ABSENT => "ABSENT",
        }
    }

    #[getter]
    fn code(&self) -> &'static str {
        match self {
            Self::NO_MAPPING => "NONE",
            Self::Q_FACTOR => "QFAC",
            Self::OTHER => "OTHER",
            Self::ABSENT => "ABSENT",
        }
    }

    fn __repr__(&self) -> String {
        format!("IonexAssumedMapping.{}", self.kind())
    }
}

impl From<IonexAssumedMapping> for PyIonexAssumedMapping {
    fn from(inner: IonexAssumedMapping) -> Self {
        match inner {
            IonexAssumedMapping::NoMapping => Self::NO_MAPPING,
            IonexAssumedMapping::QFactor => Self::Q_FACTOR,
            IonexAssumedMapping::Other => Self::OTHER,
            IonexAssumedMapping::Absent => Self::ABSENT,
        }
    }
}

/// Coverage policy for IONEX queries falling outside the product boundary.
#[pyclass(
    module = "sidereon._sidereon",
    name = "IonexCoveragePolicy",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum PyIonexCoveragePolicy {
    STRICT,
    HOLD,
}

impl From<PyIonexCoveragePolicy> for IonexCoveragePolicy {
    fn from(p: PyIonexCoveragePolicy) -> Self {
        match p {
            PyIonexCoveragePolicy::STRICT => Self::Strict,
            PyIonexCoveragePolicy::HOLD => Self::Hold,
        }
    }
}

impl From<IonexCoveragePolicy> for PyIonexCoveragePolicy {
    fn from(p: IonexCoveragePolicy) -> Self {
        match p {
            IonexCoveragePolicy::Strict => Self::STRICT,
            IonexCoveragePolicy::Hold => Self::HOLD,
        }
    }
}

/// Policy applied when interpolation weights non-available grid nodes.
#[pyclass(
    module = "sidereon._sidereon",
    name = "IonexMissingNodePolicy",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum PyIonexMissingNodePolicy {
    STRICT,
    RENORMALIZE,
}

impl From<PyIonexMissingNodePolicy> for IonexMissingNodePolicy {
    fn from(p: PyIonexMissingNodePolicy) -> Self {
        match p {
            PyIonexMissingNodePolicy::STRICT => Self::Strict,
            PyIonexMissingNodePolicy::RENORMALIZE => Self::Renormalize,
        }
    }
}

impl From<IonexMissingNodePolicy> for PyIonexMissingNodePolicy {
    fn from(p: IonexMissingNodePolicy) -> Self {
        match p {
            IonexMissingNodePolicy::Strict => Self::STRICT,
            IonexMissingNodePolicy::Renormalize => Self::RENORMALIZE,
        }
    }
}

/// Mapping factor policy for IONEX slant-delay calculation.
#[pyclass(module = "sidereon._sidereon", name = "IonexMappingPolicy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum PyIonexMappingPolicy {
    DECLARED,
    SINGLE_LAYER,
}

impl From<PyIonexMappingPolicy> for IonexMappingPolicy {
    fn from(p: PyIonexMappingPolicy) -> Self {
        match p {
            PyIonexMappingPolicy::DECLARED => Self::Declared,
            PyIonexMappingPolicy::SINGLE_LAYER => Self::SingleLayer,
        }
    }
}

impl From<IonexMappingPolicy> for PyIonexMappingPolicy {
    fn from(p: IonexMappingPolicy) -> Self {
        match p {
            IonexMappingPolicy::Declared => Self::DECLARED,
            IonexMappingPolicy::SingleLayer => Self::SINGLE_LAYER,
        }
    }
}

/// Composite evaluation policy for IONEX slant-delay queries.
#[pyclass(module = "sidereon._sidereon", name = "IonexSlantPolicy")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PyIonexSlantPolicy {
    pub(crate) inner: IonexSlantPolicy,
}

#[pymethods]
impl PyIonexSlantPolicy {
    /// Construct policy with coverage, missing-node and mapping configurations.
    ///
    /// Defaults match core: Strict coverage, Strict missing nodes, SingleLayer mapping.
    #[new]
    #[pyo3(signature = (coverage=None, missing_nodes=None, mapping=None))]
    fn new(
        coverage: Option<PyIonexCoveragePolicy>,
        missing_nodes: Option<PyIonexMissingNodePolicy>,
        mapping: Option<PyIonexMappingPolicy>,
    ) -> Self {
        let mut policy = IonexSlantPolicy::default();
        if let Some(c) = coverage {
            policy = policy.with_coverage(c.into());
        }
        if let Some(m) = missing_nodes {
            policy = policy.with_missing_nodes(m.into());
        }
        if let Some(map) = mapping {
            policy = policy.with_mapping(map.into());
        }
        Self { inner: policy }
    }

    /// Default policy: Strict coverage, Strict missing nodes, SingleLayer mapping.
    #[staticmethod]
    fn default_policy() -> Self {
        Self {
            inner: IonexSlantPolicy::default(),
        }
    }

    #[getter]
    fn coverage(&self) -> PyIonexCoveragePolicy {
        self.inner.coverage.into()
    }

    #[getter]
    fn missing_nodes(&self) -> PyIonexMissingNodePolicy {
        self.inner.missing_nodes.into()
    }

    #[getter]
    fn mapping(&self) -> PyIonexMappingPolicy {
        self.inner.mapping.into()
    }

    /// Return a copy using the specified coverage policy.
    fn with_coverage(&self, coverage: PyIonexCoveragePolicy) -> Self {
        Self {
            inner: self.inner.with_coverage(coverage.into()),
        }
    }

    /// Return a copy using the specified missing-node policy.
    fn with_missing_nodes(&self, missing_nodes: PyIonexMissingNodePolicy) -> Self {
        Self {
            inner: self.inner.with_missing_nodes(missing_nodes.into()),
        }
    }

    /// Return a copy using the specified mapping policy.
    fn with_mapping(&self, mapping: PyIonexMappingPolicy) -> Self {
        Self {
            inner: self.inner.with_mapping(mapping.into()),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexSlantPolicy(coverage={:?}, missing_nodes={:?}, mapping={:?})",
            self.coverage(),
            self.missing_nodes(),
            self.mapping()
        )
    }

    fn __eq__(&self, other: &PyIonexSlantPolicy) -> bool {
        self.inner == other.inner
    }
}

/// IONEX coverage miss variants.
#[pyclass(module = "sidereon._sidereon", name = "IonexCoverageError", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[allow(non_camel_case_types)]
pub enum PyIonexCoverageError {
    EPOCH_BEFORE_FIRST_MAP,
    EPOCH_AFTER_LAST_MAP,
    LATITUDE_OUT_OF_RANGE,
    LONGITUDE_OUT_OF_RANGE,
}

#[pymethods]
impl PyIonexCoverageError {
    #[getter]
    fn message(&self) -> &'static str {
        match self {
            Self::EPOCH_BEFORE_FIRST_MAP => "epoch precedes first map",
            Self::EPOCH_AFTER_LAST_MAP => "epoch follows last map",
            Self::LATITUDE_OUT_OF_RANGE => "latitude outside grid",
            Self::LONGITUDE_OUT_OF_RANGE => "longitude outside grid",
        }
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self {
            Self::EPOCH_BEFORE_FIRST_MAP => "EPOCH_BEFORE_FIRST_MAP",
            Self::EPOCH_AFTER_LAST_MAP => "EPOCH_AFTER_LAST_MAP",
            Self::LATITUDE_OUT_OF_RANGE => "LATITUDE_OUT_OF_RANGE",
            Self::LONGITUDE_OUT_OF_RANGE => "LONGITUDE_OUT_OF_RANGE",
        }
    }

    fn __repr__(&self) -> String {
        format!("IonexCoverageError.{}", self.kind())
    }

    fn __str__(&self) -> &'static str {
        self.message()
    }
}

impl From<IonexCoverageError> for PyIonexCoverageError {
    fn from(err: IonexCoverageError) -> Self {
        match err {
            IonexCoverageError::EpochBeforeFirstMap => Self::EPOCH_BEFORE_FIRST_MAP,
            IonexCoverageError::EpochAfterLastMap => Self::EPOCH_AFTER_LAST_MAP,
            IonexCoverageError::LatitudeOutOfRange => Self::LATITUDE_OUT_OF_RANGE,
            IonexCoverageError::LongitudeOutOfRange => Self::LONGITUDE_OUT_OF_RANGE,
        }
    }
}

impl From<PyIonexCoverageError> for IonexCoverageError {
    fn from(err: PyIonexCoverageError) -> Self {
        match err {
            PyIonexCoverageError::EPOCH_BEFORE_FIRST_MAP => Self::EpochBeforeFirstMap,
            PyIonexCoverageError::EPOCH_AFTER_LAST_MAP => Self::EpochAfterLastMap,
            PyIonexCoverageError::LATITUDE_OUT_OF_RANGE => Self::LatitudeOutOfRange,
            PyIonexCoverageError::LONGITUDE_OUT_OF_RANGE => Self::LongitudeOutOfRange,
        }
    }
}

/// Nodes of one map's interpolation cell that carry non-zero weight and are missing.
#[pyclass(module = "sidereon._sidereon", name = "IonexMissingNodes")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PyIonexMissingNodes {
    pub(crate) inner: IonexMissingNodes,
}

#[pymethods]
impl PyIonexMissingNodes {
    #[getter]
    fn map_number(&self) -> usize {
        self.inner.map_number
    }

    #[getter]
    fn lat_index(&self) -> usize {
        self.inner.lat_index
    }

    #[getter]
    fn lon_index(&self) -> usize {
        self.inner.lon_index
    }

    #[getter]
    fn lon_index_next(&self) -> usize {
        self.inner.lon_index_next
    }

    /// Ordered missing flags: `[lat_index][lon_index]`, `[lat_index][lon_index_next]`,
    /// `[lat_index + 1][lon_index]`, `[lat_index + 1][lon_index_next]`.
    #[getter]
    fn missing(&self) -> (bool, bool, bool, bool) {
        (
            self.inner.missing[0],
            self.inner.missing[1],
            self.inner.missing[2],
            self.inner.missing[3],
        )
    }

    #[getter]
    fn missing_list(&self) -> Vec<bool> {
        self.inner.missing.to_vec()
    }

    #[getter]
    fn missing_00(&self) -> bool {
        self.inner.missing[0]
    }

    #[getter]
    fn missing_01(&self) -> bool {
        self.inner.missing[1]
    }

    #[getter]
    fn missing_10(&self) -> bool {
        self.inner.missing[2]
    }

    #[getter]
    fn missing_11(&self) -> bool {
        self.inner.missing[3]
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexMissingNodes(map_number={}, lat_index={}, lon_index={}, lon_index_next={}, missing={:?})",
            self.inner.map_number, self.inner.lat_index, self.inner.lon_index, self.inner.lon_index_next, self.inner.missing
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __eq__(&self, other: &PyIonexMissingNodes) -> bool {
        self.inner == other.inner
    }
}

impl From<IonexMissingNodes> for PyIonexMissingNodes {
    fn from(inner: IonexMissingNodes) -> Self {
        Self { inner }
    }
}

/// The non-available nodes a slant-delay query weights on earlier and later bracketing maps.
#[pyclass(module = "sidereon._sidereon", name = "IonexNodeGap")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PyIonexNodeGap {
    pub(crate) inner: IonexNodeGap,
}

#[pymethods]
impl PyIonexNodeGap {
    #[getter]
    fn earlier(&self) -> Option<PyIonexMissingNodes> {
        self.inner.earlier.map(Into::into)
    }

    #[getter]
    fn later(&self) -> Option<PyIonexMissingNodes> {
        self.inner.later.map(Into::into)
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexNodeGap(earlier={:?}, later={:?})",
            self.earlier(),
            self.later()
        )
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn __eq__(&self, other: &PyIonexNodeGap) -> bool {
        self.inner == other.inner
    }
}

impl From<IonexNodeGap> for PyIonexNodeGap {
    fn from(inner: IonexNodeGap) -> Self {
        Self { inner }
    }
}

/// Status of an evaluated IONEX slant delay.
#[pyclass(module = "sidereon._sidereon", name = "IonexSlantDelayStatus")]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PyIonexSlantDelayStatus {
    pub(crate) inner: IonexSlantDelayStatus,
}

#[pymethods]
impl PyIonexSlantDelayStatus {
    /// Nominal status: within coverage, all weighted nodes available, declared mapping.
    #[classattr]
    const VALID: Self = Self {
        inner: IonexSlantDelayStatus::VALID,
    };

    #[getter]
    fn held(&self) -> Option<PyIonexCoverageError> {
        self.inner.held.map(Into::into)
    }

    #[getter]
    fn degraded(&self) -> Option<PyIonexNodeGap> {
        self.inner.degraded.map(Into::into)
    }

    #[getter]
    fn assumed_mapping(&self) -> Option<PyIonexAssumedMapping> {
        self.inner.assumed_mapping.map(Into::into)
    }

    /// Whether the value was neither held through a coverage miss nor degraded
    /// by non-available nodes.
    ///
    /// Assumed mapping alone does NOT invalidate nominal coverage and node availability.
    #[getter]
    fn is_valid(&self) -> bool {
        self.inner.is_valid()
    }

    /// Whether the evaluation is fully nominal: within coverage, all weighted nodes
    /// available, and using the product's declared mapping function.
    #[getter]
    fn is_nominal(&self) -> bool {
        self.inner == IonexSlantDelayStatus::VALID
    }

    #[getter]
    fn is_held(&self) -> bool {
        self.inner.held.is_some()
    }

    #[getter]
    fn is_degraded(&self) -> bool {
        self.inner.degraded.is_some()
    }

    #[getter]
    fn is_assumed_mapping(&self) -> bool {
        self.inner.assumed_mapping.is_some()
    }

    #[getter]
    fn has_assumed_mapping(&self) -> bool {
        self.inner.assumed_mapping.is_some()
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexSlantDelayStatus(is_valid={}, held={:?}, degraded={:?}, assumed_mapping={:?})",
            self.is_valid(),
            self.held(),
            self.degraded(),
            self.assumed_mapping()
        )
    }

    fn __eq__(&self, other: &PyIonexSlantDelayStatus) -> bool {
        self.inner == other.inner
    }
}

impl From<IonexSlantDelayStatus> for PyIonexSlantDelayStatus {
    fn from(inner: IonexSlantDelayStatus) -> Self {
        Self { inner }
    }
}

/// Evaluated IONEX slant ionospheric group delay plus its complete status.
#[pyclass(module = "sidereon._sidereon", name = "IonexSlantDelayEvaluation")]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PyIonexSlantDelayEvaluation {
    pub(crate) inner: IonexSlantDelayEvaluation,
}

#[pymethods]
impl PyIonexSlantDelayEvaluation {
    /// Ionospheric group delay, metres.
    #[getter]
    fn delay_m(&self) -> f64 {
        self.inner.delay_m
    }

    /// Complete coverage, missing-node, and assumed-mapping status.
    #[getter]
    fn status(&self) -> PyIonexSlantDelayStatus {
        self.inner.status.into()
    }

    /// True if within coverage and evaluated from available nodes.
    #[getter]
    fn is_valid(&self) -> bool {
        self.inner.status.is_valid()
    }

    /// Whether the evaluation is fully nominal: within coverage, all weighted nodes
    /// available, and using the product's declared mapping function.
    #[getter]
    fn is_nominal(&self) -> bool {
        self.inner.status == IonexSlantDelayStatus::VALID
    }

    /// Coverage miss held through, or None.
    #[getter]
    fn held(&self) -> Option<PyIonexCoverageError> {
        self.inner.status.held.map(Into::into)
    }

    /// Non-available nodes degraded through, or None.
    #[getter]
    fn degraded(&self) -> Option<PyIonexNodeGap> {
        self.inner.status.degraded.map(Into::into)
    }

    /// Mapping function assumed when single-layer was applied, or None.
    #[getter]
    fn assumed_mapping(&self) -> Option<PyIonexAssumedMapping> {
        self.inner.status.assumed_mapping.map(Into::into)
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexSlantDelayEvaluation(delay_m={}, status={})",
            self.inner.delay_m,
            self.status().__repr__()
        )
    }

    fn __eq__(&self, other: &PyIonexSlantDelayEvaluation) -> bool {
        self.inner == other.inner
    }
}

impl From<IonexSlantDelayEvaluation> for PyIonexSlantDelayEvaluation {
    fn from(inner: IonexSlantDelayEvaluation) -> Self {
        Self { inner }
    }
}

/// One slant-delay query for batch evaluation.
#[pyclass(module = "sidereon._sidereon", name = "IonexSlantRequest")]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PyIonexSlantRequest {
    #[pyo3(get, set)]
    pub lat_deg: f64,
    #[pyo3(get, set)]
    pub lon_deg: f64,
    #[pyo3(get, set)]
    pub azimuth_deg: f64,
    #[pyo3(get, set)]
    pub elevation_deg: f64,
    #[pyo3(get, set)]
    pub epoch_j2000_s: i64,
    #[pyo3(get, set)]
    pub frequency_hz: f64,
}

#[pymethods]
impl PyIonexSlantRequest {
    #[new]
    fn new(
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch_j2000_s: i64,
        frequency_hz: f64,
    ) -> Self {
        Self {
            lat_deg,
            lon_deg,
            azimuth_deg,
            elevation_deg,
            epoch_j2000_s,
            frequency_hz,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexSlantRequest(lat_deg={}, lon_deg={}, azimuth_deg={}, elevation_deg={}, epoch_j2000_s={}, frequency_hz={})",
            self.lat_deg, self.lon_deg, self.azimuth_deg, self.elevation_deg, self.epoch_j2000_s, self.frequency_hz
        )
    }

    fn __eq__(&self, other: &PyIonexSlantRequest) -> bool {
        self.lat_deg == other.lat_deg
            && self.lon_deg == other.lon_deg
            && self.azimuth_deg == other.azimuth_deg
            && self.elevation_deg == other.elevation_deg
            && self.epoch_j2000_s == other.epoch_j2000_s
            && self.frequency_hz == other.frequency_hz
    }
}

/// One slant-delay query whose epoch retains its native time scale and precision.
#[pyclass(module = "sidereon._sidereon", name = "IonexInstantSlantRequest")]
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PyIonexInstantSlantRequest {
    pub(crate) lat_deg: f64,
    pub(crate) lon_deg: f64,
    pub(crate) azimuth_deg: f64,
    pub(crate) elevation_deg: f64,
    pub(crate) epoch: Instant,
    pub(crate) frequency_hz: f64,
}

#[pymethods]
impl PyIonexInstantSlantRequest {
    #[new]
    fn new(
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch: &PyClockInstant,
        frequency_hz: f64,
    ) -> Self {
        Self {
            lat_deg,
            lon_deg,
            azimuth_deg,
            elevation_deg,
            epoch: epoch.to_core(),
            frequency_hz,
        }
    }

    #[getter]
    fn lat_deg(&self) -> f64 {
        self.lat_deg
    }

    #[getter]
    fn lon_deg(&self) -> f64 {
        self.lon_deg
    }

    #[getter]
    fn azimuth_deg(&self) -> f64 {
        self.azimuth_deg
    }

    #[getter]
    fn elevation_deg(&self) -> f64 {
        self.elevation_deg
    }

    #[getter]
    fn epoch(&self) -> PyClockInstant {
        PyClockInstant::from_core(self.epoch)
    }

    #[getter]
    fn frequency_hz(&self) -> f64 {
        self.frequency_hz
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexInstantSlantRequest(lat_deg={}, lon_deg={}, azimuth_deg={}, elevation_deg={}, epoch={:?}, frequency_hz={})",
            self.lat_deg,
            self.lon_deg,
            self.azimuth_deg,
            self.elevation_deg,
            self.epoch,
            self.frequency_hz,
        )
    }

    fn __eq__(&self, other: &PyIonexInstantSlantRequest) -> bool {
        self.lat_deg == other.lat_deg
            && self.lon_deg == other.lon_deg
            && self.azimuth_deg == other.azimuth_deg
            && self.elevation_deg == other.elevation_deg
            && self.epoch == other.epoch
            && self.frequency_hz == other.frequency_hz
    }
}

/// Detailed reason why an IONEX slant query was refused under the requested policy.
#[pyclass(module = "sidereon._sidereon", name = "IonexSlantRefusal")]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PyIonexSlantRefusal {
    pub(crate) kind: String,
    pub(crate) coverage_error: Option<PyIonexCoverageError>,
    pub(crate) epoch_error: Option<PyIonexEpochErrorDetail>,
    pub(crate) node_gap: Option<PyIonexNodeGap>,
    pub(crate) map_number: Option<usize>,
    pub(crate) lat_index: Option<usize>,
    pub(crate) lon_index: Option<usize>,
    pub(crate) mapping_declaration: Option<PyIonexMappingDeclaration>,
    pub(crate) message: String,
}

#[pymethods]
impl PyIonexSlantRefusal {
    #[getter]
    fn kind(&self) -> &str {
        &self.kind
    }

    #[getter]
    fn coverage_error(&self) -> Option<PyIonexCoverageError> {
        self.coverage_error
    }

    #[getter]
    fn epoch_error(&self) -> Option<PyIonexEpochErrorDetail> {
        self.epoch_error.clone()
    }

    #[getter]
    fn node_gap(&self) -> Option<PyIonexNodeGap> {
        self.node_gap
    }

    #[getter]
    fn map_number(&self) -> Option<usize> {
        self.map_number
    }

    #[getter]
    fn lat_index(&self) -> Option<usize> {
        self.lat_index
    }

    #[getter]
    fn lon_index(&self) -> Option<usize> {
        self.lon_index
    }

    #[getter]
    fn mapping_declaration(&self) -> Option<PyIonexMappingDeclaration> {
        self.mapping_declaration.clone()
    }

    #[getter]
    fn message(&self) -> &str {
        &self.message
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexSlantRefusal(kind='{}', message='{}')",
            self.kind, self.message
        )
    }

    fn __str__(&self) -> &str {
        &self.message
    }
}

impl PyIonexSlantRefusal {
    pub(crate) fn from_core_error(err: sidereon_core::Error) -> Self {
        match err {
            sidereon_core::Error::IonexOutOfCoverage(cov) => Self {
                kind: "COVERAGE".to_string(),
                coverage_error: Some(cov.into()),
                epoch_error: None,
                node_gap: None,
                map_number: None,
                lat_index: None,
                lon_index: None,
                mapping_declaration: None,
                message: format!("IONEX out of coverage: {cov}"),
            },
            sidereon_core::Error::IonexNodesNotAvailable(gap) => Self {
                kind: "MISSING_NODES".to_string(),
                coverage_error: None,
                epoch_error: None,
                node_gap: Some((*gap).into()),
                map_number: None,
                lat_index: None,
                lon_index: None,
                mapping_declaration: None,
                message: format!("IONEX nodes not available: {gap}"),
            },
            sidereon_core::Error::IonexSlantUnavailable(refusal) => {
                let message = refusal.to_string();
                match refusal {
                    IonexSlantRefusal::VaryingHeights {
                        map_number,
                        lat_index,
                        lon_index,
                    } => Self {
                        kind: "VARYING_HEIGHTS".to_string(),
                        coverage_error: None,
                        epoch_error: None,
                        node_gap: None,
                        map_number: Some(map_number),
                        lat_index: Some(lat_index),
                        lon_index: Some(lon_index),
                        mapping_declaration: None,
                        message,
                    },
                    IonexSlantRefusal::HeightNotAvailable {
                        map_number,
                        lat_index,
                        lon_index,
                    } => Self {
                        kind: "HEIGHT_NOT_AVAILABLE".to_string(),
                        coverage_error: None,
                        epoch_error: None,
                        node_gap: None,
                        map_number: Some(map_number),
                        lat_index: Some(lat_index),
                        lon_index: Some(lon_index),
                        mapping_declaration: None,
                        message,
                    },
                    IonexSlantRefusal::MappingFunction(decl) => Self {
                        kind: "MAPPING_FUNCTION".to_string(),
                        coverage_error: None,
                        epoch_error: None,
                        node_gap: None,
                        map_number: None,
                        lat_index: None,
                        lon_index: None,
                        mapping_declaration: Some(decl.into()),
                        message,
                    },
                    #[allow(unreachable_patterns)]
                    _ => Self {
                        kind: "UNKNOWN".to_string(),
                        coverage_error: None,
                        epoch_error: None,
                        node_gap: None,
                        map_number: None,
                        lat_index: None,
                        lon_index: None,
                        mapping_declaration: None,
                        message,
                    },
                }
            }
            sidereon_core::Error::InvalidInput(msg) => Self {
                kind: "INVALID_INPUT".to_string(),
                coverage_error: None,
                epoch_error: None,
                node_gap: None,
                map_number: None,
                lat_index: None,
                lon_index: None,
                mapping_declaration: None,
                message: format!("invalid input: {msg}"),
            },
            sidereon_core::Error::IonexEpoch(epoch_error) => Self {
                kind: "EPOCH".to_string(),
                coverage_error: None,
                epoch_error: Some(epoch_error.into()),
                node_gap: None,
                map_number: None,
                lat_index: None,
                lon_index: None,
                mapping_declaration: None,
                message: sidereon_core::Error::IonexEpoch(epoch_error).to_string(),
            },
            #[allow(unreachable_patterns)]
            other => Self {
                kind: "OTHER".to_string(),
                coverage_error: None,
                epoch_error: None,
                node_gap: None,
                map_number: None,
                lat_index: None,
                lon_index: None,
                mapping_declaration: None,
                message: other.to_string(),
            },
        }
    }
}

/// One item in a batch slant-delay evaluation result.
#[pyclass(module = "sidereon._sidereon", name = "IonexSlantBatchResult")]
#[derive(Clone)]
pub struct PyIonexSlantBatchResult {
    pub(crate) is_ok: bool,
    pub(crate) evaluation: Option<PyIonexSlantDelayEvaluation>,
    pub(crate) refusal: Option<PyIonexSlantRefusal>,
}

#[pymethods]
impl PyIonexSlantBatchResult {
    #[getter]
    fn is_ok(&self) -> bool {
        self.is_ok
    }

    #[getter]
    fn evaluation(&self) -> Option<PyIonexSlantDelayEvaluation> {
        self.evaluation
    }

    #[getter]
    fn value(&self) -> Option<PyIonexSlantDelayEvaluation> {
        self.evaluation
    }

    #[getter]
    fn refusal(&self) -> Option<PyIonexSlantRefusal> {
        self.refusal.clone()
    }

    #[getter]
    fn error(&self) -> Option<PyIonexSlantRefusal> {
        self.refusal.clone()
    }

    #[getter]
    fn delay_m(&self) -> Option<f64> {
        self.evaluation.as_ref().map(|e| e.delay_m())
    }

    #[getter]
    fn status(&self) -> Option<PyIonexSlantDelayStatus> {
        self.evaluation.as_ref().map(|e| e.status())
    }

    fn unwrap(&self) -> PyResult<PyIonexSlantDelayEvaluation> {
        if self.is_ok {
            Ok(self.evaluation.expect("evaluation is present"))
        } else {
            let msg = self
                .refusal
                .as_ref()
                .map_or("slant evaluation failed", |r| &r.message);
            Err(PyValueError::new_err(msg.to_string()))
        }
    }

    fn __repr__(&self) -> String {
        if self.is_ok {
            format!(
                "IonexSlantBatchResult(ok=True, evaluation={})",
                self.evaluation.as_ref().expect("present").__repr__()
            )
        } else {
            format!(
                "IonexSlantBatchResult(ok=False, refusal={})",
                self.refusal.as_ref().expect("present").__repr__()
            )
        }
    }
}

/// Precision-preserving epoch for IONEX diagnostic findings.
#[pyclass(module = "sidereon._sidereon", name = "IonexDiagnosticEpoch")]
#[derive(Clone, Copy, PartialEq)]
pub struct PyIonexDiagnosticEpoch {
    pub(crate) inner: Instant,
}

#[pymethods]
impl PyIonexDiagnosticEpoch {
    #[getter]
    fn scale(&self) -> &'static str {
        self.inner.scale.abbrev()
    }

    #[getter]
    fn jd_whole(&self) -> Option<f64> {
        self.inner.julian_date().map(|jd| jd.jd_whole)
    }

    #[getter]
    fn fraction(&self) -> Option<f64> {
        self.inner.julian_date().map(|jd| jd.fraction)
    }

    #[getter]
    fn jd(&self) -> Option<f64> {
        self.inner.julian_date().map(|jd| jd.to_jd())
    }

    #[getter]
    fn nanos(&self) -> Option<i128> {
        match self.inner.repr {
            InstantRepr::Nanos(nanos) => Some(nanos),
            InstantRepr::JulianDate(_) => None,
        }
    }

    #[getter]
    fn j2000_seconds(&self) -> Option<i64> {
        ionex_epoch_to_j2000_seconds(self.inner)
    }

    #[getter]
    fn j2000_seconds_f64(&self) -> Option<f64> {
        match self.inner.repr {
            InstantRepr::JulianDate(split) => {
                Some(j2000_seconds_from_split(split.jd_whole, split.fraction))
            }
            InstantRepr::Nanos(nanos) => Some(nanos as f64 / 1.0e9),
        }
    }

    fn __repr__(&self) -> String {
        match self.inner.repr {
            InstantRepr::JulianDate(split) => format!(
                "IonexDiagnosticEpoch(scale='{}', jd_whole={}, fraction={})",
                self.scale(),
                split.jd_whole,
                split.fraction
            ),
            InstantRepr::Nanos(nanos) => format!(
                "IonexDiagnosticEpoch(scale='{}', nanos={})",
                self.scale(),
                nanos
            ),
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __eq__(&self, other: &PyIonexDiagnosticEpoch) -> bool {
        self.inner == other.inner
    }
}

/// A finding reported during IONEX parsing without refusing the file.
#[pyclass(module = "sidereon._sidereon", name = "IonexWarning")]
#[derive(Clone, PartialEq)]
pub struct PyIonexWarning {
    pub(crate) inner: IonexWarning,
}

#[pymethods]
impl PyIonexWarning {
    #[getter]
    fn kind(&self) -> String {
        let label = match &self.inner {
            IonexWarning::MissingRecord(_) => "MISSING_RECORD",
            IonexWarning::VersionRecordNotFirst { .. } => "VERSION_RECORD_NOT_FIRST",
            IonexWarning::EpochMismatch { .. } => "EPOCH_MISMATCH",
            IonexWarning::MapCountMismatch { .. } => "MAP_COUNT_MISMATCH",
            IonexWarning::NotANumberValue { .. } => "NOT_A_NUMBER_VALUE",
            IonexWarning::IntervalMismatch { .. } => "INTERVAL_MISMATCH",
            IonexWarning::ExponentCarriedIntoMap { .. } => "EXPONENT_CARRIED_INTO_MAP",
            #[allow(unreachable_patterns)]
            other => return crate::marshal::debug_variant_upper(other),
        };
        label.to_string()
    }

    #[getter]
    fn label(&self) -> Option<&'static str> {
        match &self.inner {
            IonexWarning::MissingRecord(label) => Some(*label),
            IonexWarning::EpochMismatch { label, .. } => Some(*label),
            _ => None,
        }
    }

    #[getter]
    fn line(&self) -> Option<usize> {
        match &self.inner {
            IonexWarning::VersionRecordNotFirst { line }
            | IonexWarning::EpochMismatch { line, .. }
            | IonexWarning::MapCountMismatch { line, .. }
            | IonexWarning::NotANumberValue { line, .. }
            | IonexWarning::IntervalMismatch { line, .. }
            | IonexWarning::ExponentCarriedIntoMap { line, .. } => Some(*line),
            _ => None,
        }
    }

    #[getter]
    fn declared_epoch(&self) -> Option<PyIonexDiagnosticEpoch> {
        match &self.inner {
            IonexWarning::EpochMismatch { declared, .. } => {
                Some(PyIonexDiagnosticEpoch { inner: *declared })
            }
            _ => None,
        }
    }

    #[getter]
    fn declared(&self) -> Option<PyIonexDiagnosticEpoch> {
        self.declared_epoch()
    }

    #[getter]
    fn maps_epoch(&self) -> Option<PyIonexDiagnosticEpoch> {
        match &self.inner {
            IonexWarning::EpochMismatch { maps, .. } => {
                Some(PyIonexDiagnosticEpoch { inner: *maps })
            }
            _ => None,
        }
    }

    #[getter]
    fn maps(&self) -> Option<PyIonexDiagnosticEpoch> {
        self.maps_epoch()
    }

    #[getter]
    fn declared_count(&self) -> Option<u64> {
        match &self.inner {
            IonexWarning::MapCountMismatch { declared, .. } => Some(*declared),
            _ => None,
        }
    }

    #[getter]
    fn tec_maps(&self) -> Option<usize> {
        match &self.inner {
            IonexWarning::MapCountMismatch { tec_maps, .. } => Some(*tec_maps),
            _ => None,
        }
    }

    #[getter]
    fn all_maps(&self) -> Option<usize> {
        match &self.inner {
            IonexWarning::MapCountMismatch { all_maps, .. } => Some(*all_maps),
            _ => None,
        }
    }

    #[getter]
    fn data_kind(&self) -> Option<&'static str> {
        match &self.inner {
            IonexWarning::NotANumberValue { kind, .. }
            | IonexWarning::ExponentCarriedIntoMap { kind, .. } => Some(*kind),
            _ => None,
        }
    }

    #[getter]
    fn map_number(&self) -> Option<usize> {
        match &self.inner {
            IonexWarning::NotANumberValue { map_number, .. }
            | IonexWarning::IntervalMismatch { map_number, .. }
            | IonexWarning::ExponentCarriedIntoMap { map_number, .. } => Some(*map_number),
            _ => None,
        }
    }

    #[getter]
    fn lat_deg(&self) -> Option<f64> {
        match &self.inner {
            IonexWarning::NotANumberValue { lat_deg, .. } => Some(*lat_deg),
            _ => None,
        }
    }

    #[getter]
    fn lon_deg(&self) -> Option<f64> {
        match &self.inner {
            IonexWarning::NotANumberValue { lon_deg, .. } => Some(*lon_deg),
            _ => None,
        }
    }

    #[getter]
    fn declared_s(&self) -> Option<u32> {
        match &self.inner {
            IonexWarning::IntervalMismatch { declared_s, .. } => Some(*declared_s),
            _ => None,
        }
    }

    #[getter]
    fn spacing_s(&self) -> Option<i64> {
        match &self.inner {
            IonexWarning::IntervalMismatch { spacing_s, .. } => Some(*spacing_s),
            _ => None,
        }
    }

    #[getter]
    fn exponent(&self) -> Option<i32> {
        match &self.inner {
            IonexWarning::ExponentCarriedIntoMap { exponent, .. } => Some(*exponent),
            _ => None,
        }
    }

    #[getter]
    fn set_by_line(&self) -> Option<usize> {
        match &self.inner {
            IonexWarning::ExponentCarriedIntoMap { set_by_line, .. } => Some(*set_by_line),
            _ => None,
        }
    }

    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexWarning(kind='{}', message='{}')",
            self.kind(),
            self.message()
        )
    }

    fn __str__(&self) -> String {
        self.message()
    }
}

/// Parsed IONEX product together with non-fatal parser warnings.
#[pyclass(module = "sidereon._sidereon", name = "IonexParseResult")]
#[derive(Clone)]
pub struct PyIonexParseResult {
    pub(crate) ionex: PyIonex,
    pub(crate) warnings: Vec<PyIonexWarning>,
}

#[pymethods]
impl PyIonexParseResult {
    #[getter]
    fn ionex(&self) -> PyIonex {
        self.ionex.clone()
    }

    #[getter]
    fn value(&self) -> PyIonex {
        self.ionex.clone()
    }

    #[getter]
    fn warnings(&self) -> Vec<PyIonexWarning> {
        self.warnings.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexParseResult(ionex={}, warnings_count={})",
            self.ionex.__repr__(),
            self.warnings.len()
        )
    }
}

/// Descriptive IONEX header records.
#[pyclass(module = "sidereon._sidereon", name = "IonexHeader")]
#[derive(Clone, PartialEq, Debug)]
pub struct PyIonexHeader {
    pub(crate) inner: IonexHeader,
}

#[pymethods]
impl PyIonexHeader {
    /// Construct descriptive IONEX header records with defaults for unstated fields.
    #[new]
    #[pyo3(signature = (
        mapping_function=None,
        version=1.0,
        satellite_system=None,
        program=None,
        run_by=None,
        date=None,
        descriptions=None,
        comments=None,
        interval_s=0,
        elevation_cutoff_deg=0.0,
        observables_used=None,
        station_count=None,
        satellite_count=None,
        maps_in_file=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mapping_function: Option<&Bound<'_, PyAny>>,
        version: f64,
        satellite_system: Option<String>,
        program: Option<String>,
        run_by: Option<String>,
        date: Option<String>,
        descriptions: Option<Vec<String>>,
        comments: Option<Vec<String>>,
        interval_s: u32,
        elevation_cutoff_deg: f64,
        observables_used: Option<String>,
        station_count: Option<u32>,
        satellite_count: Option<u32>,
        maps_in_file: Option<u32>,
    ) -> PyResult<Self> {
        let mf = match mapping_function {
            Some(obj) => Some(mapping_function_from_py(obj)?),
            None => None,
        };
        let mut inner = IonexHeader::new(IonexMappingFunction::CosZ);
        inner.mapping_function = mf;
        inner.version = version;
        inner.satellite_system = satellite_system.unwrap_or_default();
        inner.program = program.unwrap_or_default();
        inner.run_by = run_by.unwrap_or_default();
        inner.date = date.unwrap_or_default();
        inner.descriptions = descriptions.unwrap_or_default();
        inner.comments = comments.unwrap_or_default();
        inner.interval_s = interval_s;
        inner.elevation_cutoff_deg = elevation_cutoff_deg;
        inner.observables_used = observables_used.unwrap_or_default();
        inner.station_count = station_count;
        inner.satellite_count = satellite_count;
        inner.maps_in_file = maps_in_file;

        Ok(Self { inner })
    }

    #[getter]
    fn version(&self) -> f64 {
        self.inner.version
    }
    #[setter]
    fn set_version(&mut self, val: f64) {
        self.inner.version = val;
    }

    #[getter]
    fn satellite_system(&self) -> &str {
        &self.inner.satellite_system
    }
    #[setter]
    fn set_satellite_system(&mut self, val: String) {
        self.inner.satellite_system = val;
    }

    #[getter]
    fn program(&self) -> &str {
        &self.inner.program
    }
    #[setter]
    fn set_program(&mut self, val: String) {
        self.inner.program = val;
    }

    #[getter]
    fn run_by(&self) -> &str {
        &self.inner.run_by
    }
    #[setter]
    fn set_run_by(&mut self, val: String) {
        self.inner.run_by = val;
    }

    #[getter]
    fn date(&self) -> &str {
        &self.inner.date
    }
    #[setter]
    fn set_date(&mut self, val: String) {
        self.inner.date = val;
    }

    #[getter]
    fn descriptions(&self) -> Vec<String> {
        self.inner.descriptions.clone()
    }
    #[setter]
    fn set_descriptions(&mut self, val: Vec<String>) {
        self.inner.descriptions = val;
    }

    #[getter]
    fn comments(&self) -> Vec<String> {
        self.inner.comments.clone()
    }
    #[setter]
    fn set_comments(&mut self, val: Vec<String>) {
        self.inner.comments = val;
    }

    #[getter]
    fn interval_s(&self) -> u32 {
        self.inner.interval_s
    }
    #[setter]
    fn set_interval_s(&mut self, val: u32) {
        self.inner.interval_s = val;
    }

    #[getter]
    fn elevation_cutoff_deg(&self) -> f64 {
        self.inner.elevation_cutoff_deg
    }
    #[setter]
    fn set_elevation_cutoff_deg(&mut self, val: f64) {
        self.inner.elevation_cutoff_deg = val;
    }

    #[getter]
    fn observables_used(&self) -> &str {
        &self.inner.observables_used
    }
    #[setter]
    fn set_observables_used(&mut self, val: String) {
        self.inner.observables_used = val;
    }

    #[getter]
    fn station_count(&self) -> Option<u32> {
        self.inner.station_count
    }
    #[setter]
    fn set_station_count(&mut self, val: Option<u32>) {
        self.inner.station_count = val;
    }

    #[getter]
    fn satellite_count(&self) -> Option<u32> {
        self.inner.satellite_count
    }
    #[setter]
    fn set_satellite_count(&mut self, val: Option<u32>) {
        self.inner.satellite_count = val;
    }

    #[getter]
    fn maps_in_file(&self) -> Option<u32> {
        self.inner.maps_in_file
    }
    #[setter]
    fn set_maps_in_file(&mut self, val: Option<u32>) {
        self.inner.maps_in_file = val;
    }

    #[getter]
    fn mapping_function(&self) -> Option<PyIonexMappingFunction> {
        self.inner
            .mapping_function
            .as_ref()
            .map(|f| PyIonexMappingFunction { inner: f.clone() })
    }

    #[getter]
    fn mapping_function_code(&self) -> Option<String> {
        self.inner
            .mapping_function
            .as_ref()
            .map(|f| f.code().to_string())
    }

    #[setter]
    fn set_mapping_function(&mut self, val: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.mapping_function = match val {
            Some(obj) => Some(mapping_function_from_py(obj)?),
            None => None,
        };
        Ok(())
    }

    #[getter]
    fn mapping_declaration(&self) -> PyIonexMappingDeclaration {
        match &self.inner.mapping_function {
            Some(func) => IonexMappingDeclaration::Declared(func.clone()).into(),
            None => IonexMappingDeclaration::Absent.into(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "IonexHeader(version={}, satellite_system='{}', mapping_function={:?}, interval_s={})",
            self.inner.version,
            self.inner.satellite_system,
            self.inner.mapping_function.as_ref().map(|f| f.code()),
            self.inner.interval_s
        )
    }

    fn __eq__(&self, other: &PyIonexHeader) -> bool {
        self.inner == other.inner
    }
}

impl From<IonexHeader> for PyIonexHeader {
    fn from(inner: IonexHeader) -> Self {
        Self { inner }
    }
}

/// One IONEX vertical-TEC sample at one grid node.
#[pyclass(module = "sidereon._sidereon", name = "TecSample")]
#[derive(Clone, Copy)]
pub struct PyTecSample {
    pub(crate) inner: TecSample,
}

impl From<TecSample> for PyTecSample {
    fn from(inner: TecSample) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTecSample {
    /// Build one IONEX vertical-TEC sample.
    ///
    /// `epoch_j2000_s` is an integer UTC second on the IONEX map axis.
    /// Latitude and longitude are node coordinates in degrees. VTEC and RMS are
    /// in TECU; height offset is in kilometres.
    #[new]
    #[pyo3(signature = (epoch_j2000_s, lat_deg, lon_deg, vtec_tecu=None, rms_tecu=None, height_offset_km=None))]
    fn new(
        epoch_j2000_s: i64,
        lat_deg: f64,
        lon_deg: f64,
        vtec_tecu: Option<f64>,
        rms_tecu: Option<f64>,
        height_offset_km: Option<f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: TecSample {
                epoch: ionex_epoch_from_j2000_seconds(epoch_j2000_s)?,
                lat_deg,
                lon_deg,
                vtec_tecu,
                rms_tecu,
                height_offset_km,
            },
        })
    }

    /// Map epoch as integer UTC seconds since J2000.
    #[getter]
    fn epoch_j2000_s(&self) -> PyResult<i64> {
        ionex_epoch_to_j2000_seconds(self.inner.epoch)
            .ok_or_else(|| PyValueError::new_err("TEC sample epoch is not an integer J2000 second"))
    }

    /// Latitude node in degrees.
    #[getter]
    fn lat_deg(&self) -> f64 {
        self.inner.lat_deg
    }

    /// Longitude node in degrees.
    #[getter]
    fn lon_deg(&self) -> f64 {
        self.inner.lon_deg
    }

    /// Vertical TEC in TECU, or `None` if non-available.
    #[getter]
    fn vtec_tecu(&self) -> Option<f64> {
        self.inner.vtec_tecu
    }

    /// Optional RMS value in TECU.
    #[getter]
    fn rms_tecu(&self) -> Option<f64> {
        self.inner.rms_tecu
    }

    /// Optional height map offset from HGT1 in kilometres.
    #[getter]
    fn height_offset_km(&self) -> Option<f64> {
        self.inner.height_offset_km
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!(
            "TecSample(epoch_j2000_s={}, lat_deg={}, lon_deg={}, vtec_tecu={:?}, rms_tecu={:?}, height_offset_km={:?})",
            self.epoch_j2000_s()?,
            self.inner.lat_deg,
            self.inner.lon_deg,
            self.inner.vtec_tecu,
            self.inner.rms_tecu,
            self.inner.height_offset_km,
        ))
    }
}

fn resolve_mask_alias<'py>(
    mask: Option<PyReadonlyArray3<'py, bool>>,
    valid: Option<PyReadonlyArray3<'py, bool>>,
    name: &str,
) -> PyResult<Option<PyReadonlyArray3<'py, bool>>> {
    match (mask, valid) {
        (Some(_), Some(_)) => Err(PyValueError::new_err(format!(
            "cannot supply both {name}_mask and {name}_valid; choose one"
        ))),
        (Some(m), None) => Ok(Some(m)),
        (None, Some(v)) => Ok(Some(v)),
        (None, None) => Ok(None),
    }
}

/// Whole-grid IONEX vertical-TEC samples.
#[pyclass(module = "sidereon._sidereon", name = "TecGridSamples")]
#[derive(Clone)]
pub struct PyTecGridSamples {
    pub(crate) inner: TecGridSamples,
}

impl From<TecGridSamples> for PyTecGridSamples {
    fn from(inner: TecGridSamples) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTecGridSamples {
    /// Build whole-grid IONEX vertical-TEC samples.
    ///
    /// `map_epochs_j2000_s` are integer UTC seconds since J2000. Latitude nodes
    /// are degrees descending, longitude nodes are degrees ascending, shell and
    /// base radius are kilometres, and map values are TECU. Dimension matching,
    /// non-zero axes, and mask presence/conflicts are validated synchronously
    /// on construction. Full grid geometry (monotonicity, node counts >= 2,
    /// valid steps) is validated by core upon product instantiation (`Ionex.from_samples`).
    #[new]
    #[pyo3(signature = (
        map_epochs_j2000_s,
        lat_nodes_deg,
        lon_nodes_deg,
        dlat_deg,
        dlon_deg,
        shell_height_km,
        base_radius_km,
        exponent,
        tec_maps,
        rms_maps=None,
        height_maps=None,
        tec_mask=None,
        rms_mask=None,
        height_mask=None,
        header=None,
        tec_valid=None,
        rms_valid=None,
        height_valid=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        map_epochs_j2000_s: PyReadonlyArray1<'_, i64>,
        lat_nodes_deg: PyReadonlyArray1<'_, f64>,
        lon_nodes_deg: PyReadonlyArray1<'_, f64>,
        dlat_deg: f64,
        dlon_deg: f64,
        shell_height_km: f64,
        base_radius_km: f64,
        exponent: i32,
        tec_maps: PyReadonlyArray3<'_, f64>,
        rms_maps: Option<PyReadonlyArray3<'_, f64>>,
        height_maps: Option<PyReadonlyArray3<'_, f64>>,
        tec_mask: Option<PyReadonlyArray3<'_, bool>>,
        rms_mask: Option<PyReadonlyArray3<'_, bool>>,
        height_mask: Option<PyReadonlyArray3<'_, bool>>,
        header: Option<&PyIonexHeader>,
        tec_valid: Option<PyReadonlyArray3<'_, bool>>,
        rms_valid: Option<PyReadonlyArray3<'_, bool>>,
        height_valid: Option<PyReadonlyArray3<'_, bool>>,
    ) -> PyResult<Self> {
        let epochs = map_epochs_j2000_s
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .iter()
            .copied()
            .map(ionex_epoch_from_j2000_seconds)
            .collect::<PyResult<Vec<_>>>()?;
        let lat_nodes_deg = lat_nodes_deg
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .to_vec();
        let lon_nodes_deg = lon_nodes_deg
            .as_slice()
            .map_err(|err| PyValueError::new_err(err.to_string()))?
            .to_vec();

        let effective_tec_mask = resolve_mask_alias(tec_mask, tec_valid, "tec")?;
        let effective_rms_mask = resolve_mask_alias(rms_mask, rms_valid, "rms")?;
        let effective_height_mask = resolve_mask_alias(height_mask, height_valid, "height")?;

        let tec_dims = tec_maps.as_array().dim();
        if tec_dims.0 == 0 || tec_dims.1 == 0 || tec_dims.2 == 0 {
            return Err(PyValueError::new_err(
                "tec_maps dimensions must be non-zero along all axes",
            ));
        }
        if tec_dims.0 != epochs.len()
            || tec_dims.1 != lat_nodes_deg.len()
            || tec_dims.2 != lon_nodes_deg.len()
        {
            return Err(PyValueError::new_err(format!(
                "tec_maps shape {:?} does not match grid dimensions ({}, {}, {})",
                tec_dims,
                epochs.len(),
                lat_nodes_deg.len(),
                lon_nodes_deg.len()
            )));
        }

        let tec_maps_vec = maps_from_array3_with_mask(tec_maps, effective_tec_mask)?;
        let rms_maps_vec = parse_optional_map_array(rms_maps, effective_rms_mask, tec_dims, "rms")?;
        let height_maps_vec =
            parse_optional_map_array(height_maps, effective_height_mask, tec_dims, "height")?;

        let header_inner = match header {
            Some(h) => h.inner.clone(),
            None => {
                let mut h = IonexHeader::new(IonexMappingFunction::CosZ);
                h.mapping_function = None;
                h
            }
        };

        Ok(Self {
            inner: TecGridSamples {
                map_epochs: epochs,
                lat_nodes_deg,
                lon_nodes_deg,
                dlat_deg,
                dlon_deg,
                shell_height_km,
                base_radius_km,
                exponent,
                tec_maps: tec_maps_vec,
                rms_maps: rms_maps_vec,
                height_maps: height_maps_vec,
                header: header_inner,
            },
        })
    }

    /// Map epochs as a numpy `(n_epoch,)` `int64` array, UTC seconds since J2000.
    #[getter]
    fn map_epochs_j2000_s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let epochs = self
            .inner
            .map_epochs
            .iter()
            .copied()
            .map(|epoch| {
                ionex_epoch_to_j2000_seconds(epoch).ok_or_else(|| {
                    PyValueError::new_err("IONEX epoch is not an integer J2000 second")
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyArray1::from_vec(py, epochs))
    }

    /// Latitude node values in degrees, descending.
    #[getter]
    fn lat_nodes_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.lat_nodes_deg)
    }

    /// Longitude node values in degrees, ascending.
    #[getter]
    fn lon_nodes_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.lon_nodes_deg)
    }

    /// Signed latitude step in degrees.
    #[getter]
    fn dlat_deg(&self) -> f64 {
        self.inner.dlat_deg
    }

    /// Signed longitude step in degrees.
    #[getter]
    fn dlon_deg(&self) -> f64 {
        self.inner.dlon_deg
    }

    /// Single-layer shell height in kilometres.
    #[getter]
    fn shell_height_km(&self) -> f64 {
        self.inner.shell_height_km
    }

    /// Mean earth radius used by the geometry, in kilometres.
    #[getter]
    fn base_radius_km(&self) -> f64 {
        self.inner.base_radius_km
    }

    /// The IONEX `EXPONENT` header field.
    #[getter]
    fn exponent(&self) -> i32 {
        self.inner.exponent
    }

    /// Per-map vertical-TEC grids as a numpy `(epoch, lat, lon)` float64 cube, TECU.
    /// Missing cells render as NaN.
    #[getter]
    fn tec_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let (float_arr, _) = maps_to_arrays3(py, &self.inner.tec_maps);
        float_arr
    }

    /// Paired boolean validity mask for `tec_maps` (`True` for present, `False` for missing).
    #[getter]
    fn tec_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        let (_, mask_arr) = maps_to_arrays3(py, &self.inner.tec_maps);
        mask_arr
    }

    /// Alias for `tec_mask`.
    #[getter]
    fn tec_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        self.tec_mask(py)
    }

    /// Per-map RMS grids as a numpy `(epoch, lat, lon)` float64 cube, TECU, or `(0, 0, 0)` when absent.
    /// Missing cells render as NaN.
    #[getter]
    fn rms_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let (float_arr, _) = maps_to_arrays3(py, &self.inner.rms_maps);
        float_arr
    }

    /// Paired boolean validity mask for `rms_maps` (`True` for present, `False` for missing).
    #[getter]
    fn rms_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        let (_, mask_arr) = maps_to_arrays3(py, &self.inner.rms_maps);
        mask_arr
    }

    /// Alias for `rms_mask`.
    #[getter]
    fn rms_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        self.rms_mask(py)
    }

    /// Per-map single-layer height grids as a numpy `(epoch, lat, lon)` float64 cube, km, or `(0, 0, 0)` when absent.
    /// Missing cells render as NaN.
    #[getter]
    fn height_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let (float_arr, _) = maps_to_arrays3(py, &self.inner.height_maps);
        float_arr
    }

    /// Paired boolean validity mask for `height_maps` (`True` for present, `False` for missing).
    #[getter]
    fn height_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        let (_, mask_arr) = maps_to_arrays3(py, &self.inner.height_maps);
        mask_arr
    }

    /// Alias for `height_mask`.
    #[getter]
    fn height_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        self.height_mask(py)
    }

    /// True if the product carries RMS maps.
    #[getter]
    fn has_rms(&self) -> bool {
        !self.inner.rms_maps.is_empty()
    }

    /// True if the product carries per-map single-layer height grids.
    #[getter]
    fn has_height(&self) -> bool {
        !self.inner.height_maps.is_empty()
    }

    /// Complete descriptive IONEX header records.
    #[getter]
    fn header(&self) -> PyIonexHeader {
        PyIonexHeader {
            inner: self.inner.header.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "TecGridSamples(epochs={}, lat_nodes={}, lon_nodes={}, height_maps={})",
            self.inner.map_epochs.len(),
            self.inner.lat_nodes_deg.len(),
            self.inner.lon_nodes_deg.len(),
            self.inner.height_maps.len(),
        )
    }
}

/// A parsed IONEX vertical-TEC product.
///
/// Construct with [`load_ionex`] or [`load_ionex_with_warnings`]. Read the grid
/// geometry with the `lat_nodes_deg` / `lon_nodes_deg` axes (descending
/// north-to-south, ascending west-to-east), the per-map `tec_maps`, `rms_maps`
/// and `height_maps` cubes (`(epoch, lat, lon)`), paired boolean validity masks
/// (`tec_mask`, `rms_mask`, `height_mask`), and complete `header`. Query the
/// line-of-sight delay with [`Ionex.slant_delay`] or [`Ionex.slant_delay_with_policy`].
#[pyclass(module = "sidereon._sidereon", name = "Ionex")]
#[derive(Clone)]
pub struct PyIonex {
    pub(crate) inner: Ionex,
}

impl PyIonex {
    /// Wrap an owned core product, for the staleness selection layer which hands
    /// back the present (or diurnal-shifted) product.
    pub(crate) fn from_ionex(inner: Ionex) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyIonex {
    /// Build an IONEX product directly from whole-grid TEC samples.
    #[staticmethod]
    fn from_samples(samples: &PyTecGridSamples) -> PyResult<Self> {
        Ionex::from_samples(samples.inner.clone())
            .map(|inner| Self { inner })
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Build an IONEX product from a flat stream of node TEC samples.
    #[staticmethod]
    #[pyo3(signature = (samples, shell_height_km, base_radius_km, exponent, header=None))]
    fn from_node_samples(
        py: Python<'_>,
        samples: Vec<Py<PyTecSample>>,
        shell_height_km: f64,
        base_radius_km: f64,
        exponent: i32,
        header: Option<&PyIonexHeader>,
    ) -> PyResult<Self> {
        let samples = samples.iter().map(|sample| sample.borrow(py).inner);
        let header_inner = match header {
            Some(h) => h.inner.clone(),
            None => {
                let mut h = IonexHeader::new(IonexMappingFunction::CosZ);
                h.mapping_function = None;
                h
            }
        };
        Ionex::from_node_samples(
            samples,
            shell_height_km,
            base_radius_km,
            exponent,
            header_inner,
        )
        .map(|inner| Self { inner })
        .map_err(|err| PyValueError::new_err(err.to_string()))
    }

    /// Parse an IONEX product from bytes.
    #[staticmethod]
    fn parse(bytes: &[u8]) -> PyResult<Self> {
        let inner = Ionex::parse(bytes).map_err(to_ionex_err)?;
        Ok(Self { inner })
    }

    /// Parse an IONEX product from text.
    #[staticmethod]
    fn parse_str(text: &str) -> PyResult<Self> {
        let inner = Ionex::parse_str(text).map_err(to_ionex_err)?;
        Ok(Self { inner })
    }

    /// Parse an IONEX product from bytes and return product plus ordered warnings.
    #[staticmethod]
    fn parse_with_warnings(bytes: &[u8]) -> PyResult<PyIonexParseResult> {
        let (inner, warnings) = Ionex::parse_with_warnings(bytes).map_err(to_ionex_err)?;
        let py_warnings = warnings
            .into_iter()
            .map(|w| PyIonexWarning { inner: w })
            .collect();
        Ok(PyIonexParseResult {
            ionex: PyIonex { inner },
            warnings: py_warnings,
        })
    }

    /// Parse an IONEX product from text and return product plus ordered warnings.
    #[staticmethod]
    fn parse_str_with_warnings(text: &str) -> PyResult<PyIonexParseResult> {
        let (inner, warnings) = Ionex::parse_str_with_warnings(text).map_err(to_ionex_err)?;
        let py_warnings = warnings
            .into_iter()
            .map(|w| PyIonexWarning { inner: w })
            .collect();
        Ok(PyIonexParseResult {
            ionex: PyIonex { inner },
            warnings: py_warnings,
        })
    }

    /// Descriptive header records carried by this product.
    #[getter]
    fn header(&self) -> PyIonexHeader {
        PyIonexHeader {
            inner: self.inner.header().clone(),
        }
    }

    /// Latitude node values in degrees as a numpy `(n_lat,)` `float64` array,
    /// descending (north-to-south).
    #[getter]
    fn lat_nodes_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, self.inner.lat_nodes_deg())
    }

    /// Longitude node values in degrees as a numpy `(n_lon,)` `float64` array,
    /// ascending (west-to-east).
    #[getter]
    fn lon_nodes_deg<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, self.inner.lon_nodes_deg())
    }

    /// Signed latitude step in degrees (negative for the standard ordering).
    #[getter]
    fn dlat_deg(&self) -> f64 {
        self.inner.dlat_deg()
    }

    /// Signed longitude step in degrees (positive for the standard ordering).
    #[getter]
    fn dlon_deg(&self) -> f64 {
        self.inner.dlon_deg()
    }

    /// Single-layer shell height in kilometers.
    #[getter]
    fn shell_height_km(&self) -> f64 {
        self.inner.shell_height_km()
    }

    /// Mean earth radius used by the geometry, in kilometers.
    #[getter]
    fn base_radius_km(&self) -> f64 {
        self.inner.base_radius_km()
    }

    /// The IONEX `EXPONENT` header field; the TEC scale is `10^EXPONENT`.
    #[getter]
    fn exponent(&self) -> i32 {
        self.inner.exponent()
    }

    /// Map epochs as a numpy `(n_epoch,)` `int64` array of seconds since J2000,
    /// ascending. This is the exact axis [`Ionex.slant_delay`] brackets against.
    #[getter]
    fn map_epochs_j2000_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.inner.map_epochs_s())
    }

    /// Per-map vertical-TEC grids as a numpy `(epoch, lat, lon)` `float64` cube
    /// in TECU. Missing cells render as NaN.
    #[getter]
    fn tec_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let (float_arr, _) = maps_to_arrays3(py, self.inner.tec_maps());
        float_arr
    }

    /// Paired boolean validity mask for `tec_maps` (`True` for present, `False` for missing).
    #[getter]
    fn tec_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        let (_, mask_arr) = maps_to_arrays3(py, self.inner.tec_maps());
        mask_arr
    }

    /// Alias for `tec_mask`.
    #[getter]
    fn tec_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        self.tec_mask(py)
    }

    /// Per-map RMS grids as a numpy `(epoch, lat, lon)` `float64` cube in TECU,
    /// or a `(0, 0, 0)` array when the product carries no RMS maps.
    /// Missing cells render as NaN.
    #[getter]
    fn rms_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let (float_arr, _) = maps_to_arrays3(py, self.inner.rms_maps());
        float_arr
    }

    /// Paired boolean validity mask for `rms_maps` (`True` for present, `False` for missing).
    #[getter]
    fn rms_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        let (_, mask_arr) = maps_to_arrays3(py, self.inner.rms_maps());
        mask_arr
    }

    /// Alias for `rms_mask`.
    #[getter]
    fn rms_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        self.rms_mask(py)
    }

    /// Per-map single-layer height grids as a numpy `(epoch, lat, lon)` `float64` cube, km,
    /// or a `(0, 0, 0)` array when the product carries no height maps.
    /// Missing cells render as NaN.
    #[getter]
    fn height_maps<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f64>> {
        let (float_arr, _) = maps_to_arrays3(py, self.inner.height_maps());
        float_arr
    }

    /// Paired boolean validity mask for `height_maps` (`True` for present, `False` for missing).
    #[getter]
    fn height_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        let (_, mask_arr) = maps_to_arrays3(py, self.inner.height_maps());
        mask_arr
    }

    /// Alias for `height_mask`.
    #[getter]
    fn height_valid<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<bool>> {
        self.height_mask(py)
    }

    /// True if the product carries RMS maps.
    #[getter]
    fn has_rms(&self) -> bool {
        !self.inner.rms_maps().is_empty()
    }

    /// True if the product carries per-map single-layer height grids.
    #[getter]
    fn has_height(&self) -> bool {
        !self.inner.height_maps().is_empty()
    }

    /// The mapping function declared in the header, or `None` if unstated.
    #[getter]
    fn mapping_function(&self) -> Option<PyIonexMappingFunction> {
        self.inner
            .header()
            .mapping_function
            .as_ref()
            .map(|f| PyIonexMappingFunction { inner: f.clone() })
    }

    /// Full declared or absent mapping function declaration from the header.
    #[getter]
    fn mapping_declaration(&self) -> PyIonexMappingDeclaration {
        match &self.inner.header().mapping_function {
            Some(func) => IonexMappingDeclaration::Declared(func.clone()).into(),
            None => IonexMappingDeclaration::Absent.into(),
        }
    }

    /// Number of records skipped during a forgiving parse.
    #[getter]
    fn skipped_records(&self) -> usize {
        self.inner.skipped_records()
    }

    /// IONEX vertical-TEC-grid slant ionospheric group delay, positive metres.
    ///
    /// Preserves scalar strict convenience and exact DEG_TO_RAD conversion order.
    #[pyo3(signature = (lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch_j2000_s, frequency_hz, hold_out_of_coverage=false))]
    #[allow(clippy::too_many_arguments)]
    fn slant_delay(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch_j2000_s: i64,
        frequency_hz: f64,
        hold_out_of_coverage: bool,
    ) -> PyResult<f64> {
        let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, 0.0)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let policy = if hold_out_of_coverage {
            IonexSlantPolicy::default().with_coverage(IonexCoveragePolicy::Hold)
        } else {
            IonexSlantPolicy::default()
        };
        core_ionex_slant_delay_with_policy(
            &self.inner,
            receiver,
            elevation_deg * DEG_TO_RAD,
            azimuth_deg * DEG_TO_RAD,
            ionex_epoch_from_j2000_seconds(epoch_j2000_s)?,
            frequency_hz,
            policy,
        )
        .map(|evaluation| evaluation.delay_m)
        .map_err(|err| match err {
            sidereon_core::Error::InvalidInput(message) => {
                PyValueError::new_err(format!("invalid IONEX slant input: {message}"))
            }
            sidereon_core::Error::IonexOutOfCoverage(cov) => {
                PyValueError::new_err(format!("IONEX out of coverage: {cov}"))
            }
            sidereon_core::Error::IonexNodesNotAvailable(gap) => {
                to_solve_err(format!("IONEX nodes not available: {gap}"))
            }
            sidereon_core::Error::IonexSlantUnavailable(refusal) => {
                to_solve_err(format!("IONEX slant delay unavailable: {refusal}"))
            }
            other => to_solve_err(other.to_string()),
        })
    }

    /// Policy-aware IONEX slant-delay query returning delay and detailed status.
    #[pyo3(signature = (lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch_j2000_s, frequency_hz, policy=None))]
    #[allow(clippy::too_many_arguments)]
    fn slant_delay_with_policy(
        &self,
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch_j2000_s: i64,
        frequency_hz: f64,
        policy: Option<&PyIonexSlantPolicy>,
    ) -> PyResult<PyIonexSlantDelayEvaluation> {
        let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, 0.0)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let pol = policy.map_or_else(IonexSlantPolicy::default, |p| p.inner);
        core_ionex_slant_delay_with_policy(
            &self.inner,
            receiver,
            elevation_deg * DEG_TO_RAD,
            azimuth_deg * DEG_TO_RAD,
            ionex_epoch_from_j2000_seconds(epoch_j2000_s)?,
            frequency_hz,
            pol,
        )
        .map(Into::into)
        .map_err(|err| match err {
            sidereon_core::Error::InvalidInput(message) => {
                PyValueError::new_err(format!("invalid IONEX slant input: {message}"))
            }
            sidereon_core::Error::IonexOutOfCoverage(cov) => {
                PyValueError::new_err(format!("IONEX out of coverage: {cov}"))
            }
            sidereon_core::Error::IonexNodesNotAvailable(gap) => {
                to_solve_err(format!("IONEX nodes not available: {gap}"))
            }
            sidereon_core::Error::IonexSlantUnavailable(refusal) => {
                to_solve_err(format!("IONEX slant delay unavailable: {refusal}"))
            }
            other => to_solve_err(other.to_string()),
        })
    }

    /// Query using a scale-tagged `ClockInstant`, retaining fractional-second precision.
    #[pyo3(signature = (lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch, frequency_hz, hold_out_of_coverage=false))]
    #[allow(clippy::too_many_arguments)]
    fn slant_delay_at_instant(
        &self,
        py: Python<'_>,
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch: &PyClockInstant,
        frequency_hz: f64,
        hold_out_of_coverage: bool,
    ) -> PyResult<f64> {
        let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, 0.0)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let policy = if hold_out_of_coverage {
            IonexSlantPolicy::default().with_coverage(IonexCoveragePolicy::Hold)
        } else {
            IonexSlantPolicy::default()
        };
        core_ionex_slant_delay_with_policy(
            &self.inner,
            receiver,
            elevation_deg * DEG_TO_RAD,
            azimuth_deg * DEG_TO_RAD,
            epoch.to_core(),
            frequency_hz,
            policy,
        )
        .map(|evaluation| evaluation.delay_m)
        .map_err(|err| match err {
            sidereon_core::Error::InvalidInput(message) => {
                PyValueError::new_err(format!("invalid IONEX slant input: {message}"))
            }
            sidereon_core::Error::IonexOutOfCoverage(cov) => {
                PyValueError::new_err(format!("IONEX out of coverage: {cov}"))
            }
            sidereon_core::Error::IonexNodesNotAvailable(gap) => {
                to_solve_err(format!("IONEX nodes not available: {gap}"))
            }
            sidereon_core::Error::IonexSlantUnavailable(refusal) => {
                to_solve_err(format!("IONEX slant delay unavailable: {refusal}"))
            }
            sidereon_core::Error::IonexEpoch(epoch_error) => {
                to_ionex_epoch_solve_err(py, epoch_error)
            }
            other => to_solve_err(other.to_string()),
        })
    }

    /// Policy-aware query using a scale-tagged `ClockInstant`.
    #[pyo3(signature = (lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch, frequency_hz, policy=None))]
    #[allow(clippy::too_many_arguments)]
    fn slant_delay_at_instant_with_policy(
        &self,
        py: Python<'_>,
        lat_deg: f64,
        lon_deg: f64,
        azimuth_deg: f64,
        elevation_deg: f64,
        epoch: &PyClockInstant,
        frequency_hz: f64,
        policy: Option<&PyIonexSlantPolicy>,
    ) -> PyResult<PyIonexSlantDelayEvaluation> {
        let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, 0.0)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        core_ionex_slant_delay_with_policy(
            &self.inner,
            receiver,
            elevation_deg * DEG_TO_RAD,
            azimuth_deg * DEG_TO_RAD,
            epoch.to_core(),
            frequency_hz,
            policy.map_or_else(IonexSlantPolicy::default, |p| p.inner),
        )
        .map(Into::into)
        .map_err(|err| match err {
            sidereon_core::Error::InvalidInput(message) => {
                PyValueError::new_err(format!("invalid IONEX slant input: {message}"))
            }
            sidereon_core::Error::IonexOutOfCoverage(cov) => {
                PyValueError::new_err(format!("IONEX out of coverage: {cov}"))
            }
            sidereon_core::Error::IonexNodesNotAvailable(gap) => {
                to_solve_err(format!("IONEX nodes not available: {gap}"))
            }
            sidereon_core::Error::IonexSlantUnavailable(refusal) => {
                to_solve_err(format!("IONEX slant delay unavailable: {refusal}"))
            }
            sidereon_core::Error::IonexEpoch(epoch_error) => {
                to_ionex_epoch_solve_err(py, epoch_error)
            }
            other => to_solve_err(other.to_string()),
        })
    }

    /// Evaluate a batch of requests preserving 1-to-1 input order, every success and every refusal.
    #[pyo3(signature = (requests, policy=None))]
    fn slant_delays_batch_results(
        &self,
        requests: Vec<PyRef<'_, PyIonexSlantRequest>>,
        policy: Option<&PyIonexSlantPolicy>,
    ) -> PyResult<Vec<PyIonexSlantBatchResult>> {
        let core_policy = policy.map_or_else(IonexSlantPolicy::default, |p| p.inner);
        let n = requests.len();
        let mut results: Vec<Option<PyIonexSlantBatchResult>> = Vec::with_capacity(n);
        for _ in 0..n {
            results.push(None);
        }

        let mut valid_requests = Vec::with_capacity(n);
        let mut valid_indices = Vec::with_capacity(n);

        for (idx, req) in requests.iter().enumerate() {
            let lat_rad = req.lat_deg * DEG_TO_RAD;
            let lon_rad = req.lon_deg * DEG_TO_RAD;
            match Wgs84Geodetic::new(lat_rad, lon_rad, 0.0) {
                Ok(receiver) => {
                    valid_requests.push(IonexSlantRequest::new(
                        receiver,
                        req.elevation_deg * DEG_TO_RAD,
                        req.azimuth_deg * DEG_TO_RAD,
                        ionex_epoch_from_j2000_seconds(req.epoch_j2000_s)?,
                        req.frequency_hz,
                    ));
                    valid_indices.push(idx);
                }
                Err(err) => {
                    results[idx] = Some(PyIonexSlantBatchResult {
                        is_ok: false,
                        evaluation: None,
                        refusal: Some(PyIonexSlantRefusal {
                            kind: "INVALID_INPUT".to_string(),
                            coverage_error: None,
                            epoch_error: None,
                            node_gap: None,
                            map_number: None,
                            lat_index: None,
                            lon_index: None,
                            mapping_declaration: None,
                            message: format!("invalid receiver: {err}"),
                        }),
                    });
                }
            }
        }

        if !valid_requests.is_empty() {
            let core_results =
                core_ionex_slant_delay_results(&self.inner, &valid_requests, core_policy);
            for (idx, res) in valid_indices.into_iter().zip(core_results) {
                results[idx] = Some(match res {
                    Ok(eval) => PyIonexSlantBatchResult {
                        is_ok: true,
                        evaluation: Some(eval.into()),
                        refusal: None,
                    },
                    Err(err) => PyIonexSlantBatchResult {
                        is_ok: false,
                        evaluation: None,
                        refusal: Some(PyIonexSlantRefusal::from_core_error(err)),
                    },
                });
            }
        }

        let out = results
            .into_iter()
            .map(|opt| opt.expect("every slot populated"))
            .collect();
        Ok(out)
    }

    /// Evaluate instant-based requests one-to-one, preserving input order and typed refusals.
    #[pyo3(signature = (requests, policy=None))]
    fn slant_delays_at_instants_batch_results(
        &self,
        requests: Vec<PyRef<'_, PyIonexInstantSlantRequest>>,
        policy: Option<&PyIonexSlantPolicy>,
    ) -> PyResult<Vec<PyIonexSlantBatchResult>> {
        let core_policy = policy.map_or_else(IonexSlantPolicy::default, |p| p.inner);
        let mut results: Vec<Option<PyIonexSlantBatchResult>> =
            (0..requests.len()).map(|_| None).collect();
        let mut valid_requests = Vec::with_capacity(requests.len());
        let mut valid_indices = Vec::with_capacity(requests.len());

        for (index, request) in requests.iter().enumerate() {
            let receiver = Wgs84Geodetic::new(
                request.lat_deg * DEG_TO_RAD,
                request.lon_deg * DEG_TO_RAD,
                0.0,
            );
            match receiver {
                Ok(receiver) => {
                    valid_requests.push(IonexSlantRequest::new(
                        receiver,
                        request.elevation_deg * DEG_TO_RAD,
                        request.azimuth_deg * DEG_TO_RAD,
                        request.epoch,
                        request.frequency_hz,
                    ));
                    valid_indices.push(index);
                }
                Err(error) => {
                    results[index] = Some(PyIonexSlantBatchResult {
                        is_ok: false,
                        evaluation: None,
                        refusal: Some(PyIonexSlantRefusal {
                            kind: "INVALID_INPUT".to_string(),
                            coverage_error: None,
                            epoch_error: None,
                            node_gap: None,
                            map_number: None,
                            lat_index: None,
                            lon_index: None,
                            mapping_declaration: None,
                            message: format!("invalid receiver: {error}"),
                        }),
                    });
                }
            }
        }

        if !valid_requests.is_empty() {
            let core_results =
                core_ionex_slant_delay_results(&self.inner, &valid_requests, core_policy);
            for (index, result) in valid_indices.into_iter().zip(core_results) {
                results[index] = Some(match result {
                    Ok(evaluation) => PyIonexSlantBatchResult {
                        is_ok: true,
                        evaluation: Some(evaluation.into()),
                        refusal: None,
                    },
                    Err(error) => PyIonexSlantBatchResult {
                        is_ok: false,
                        evaluation: None,
                        refusal: Some(PyIonexSlantRefusal::from_core_error(error)),
                    },
                });
            }
        }

        Ok(results
            .into_iter()
            .map(|result| result.expect("every request slot has a result"))
            .collect())
    }

    /// Serialize this product to standard IONEX text via the core writer.
    ///
    /// Returns `PyResult` and propagates core named refusals without unwrap or lossy fallback.
    fn to_ionex_string(&self, py: Python<'_>) -> PyResult<String> {
        self.inner
            .to_ionex_string()
            .map_err(|err| to_ionex_write_err(py, err))
    }

    /// Extract this product as whole-grid IONEX samples.
    fn tec_grid_samples(&self) -> PyTecGridSamples {
        self.inner.tec_grid_samples().into()
    }

    /// Extract this product as one [`TecSample`] per grid node.
    fn tec_samples(&self) -> Vec<PyTecSample> {
        self.inner
            .tec_samples()
            .into_iter()
            .map(Into::into)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Ionex(epochs={}, lat_nodes={}, lon_nodes={}, exponent={})",
            self.inner.map_epochs_s().len(),
            self.inner.lat_nodes_deg().len(),
            self.inner.lon_nodes_deg().len(),
            self.inner.exponent(),
        )
    }
}

/// Parse an IONEX vertical-TEC product from in-memory bytes or a file path.
///
/// `source` may be:
/// - `bytes` / `bytearray`: the full IONEX text content, parsed directly; or
/// - a path (`str` or `os.PathLike`): the file is read and parsed.
///
/// Raises [`IonexParseError`](crate::IonexParseError) on malformed content,
/// `OSError` if the path cannot be read, and `ValueError` if `source` is neither
/// bytes nor a path.
#[pyfunction]
fn load_ionex(source: &Bound<'_, PyAny>) -> PyResult<PyIonex> {
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        let inner = Ionex::parse(bytes.as_bytes()).map_err(to_ionex_err)?;
        return Ok(PyIonex { inner });
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        let inner = Ionex::parse(unsafe { buf.as_bytes() }).map_err(to_ionex_err)?;
        return Ok(PyIonex { inner });
    }
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err("load_ionex expects bytes, bytearray, or a path (str/os.PathLike)")
    })?;
    let data = std::fs::read(&path)?;
    let inner = Ionex::parse(&data).map_err(to_ionex_err)?;
    Ok(PyIonex { inner })
}

/// Parse an IONEX product from bytes or a path and return product plus ordered warnings.
#[pyfunction]
fn load_ionex_with_warnings(source: &Bound<'_, PyAny>) -> PyResult<PyIonexParseResult> {
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        let (inner, warnings) =
            Ionex::parse_with_warnings(bytes.as_bytes()).map_err(to_ionex_err)?;
        let py_warnings = warnings
            .into_iter()
            .map(|w| PyIonexWarning { inner: w })
            .collect();
        return Ok(PyIonexParseResult {
            ionex: PyIonex { inner },
            warnings: py_warnings,
        });
    }
    if let Ok(buf) = source.downcast::<PyByteArray>() {
        let (inner, warnings) =
            Ionex::parse_with_warnings(unsafe { buf.as_bytes() }).map_err(to_ionex_err)?;
        let py_warnings = warnings
            .into_iter()
            .map(|w| PyIonexWarning { inner: w })
            .collect();
        return Ok(PyIonexParseResult {
            ionex: PyIonex { inner },
            warnings: py_warnings,
        });
    }
    let path: PathBuf = source.extract().map_err(|_| {
        PyValueError::new_err(
            "load_ionex_with_warnings expects bytes, bytearray, or a path (str/os.PathLike)",
        )
    })?;
    let data = std::fs::read(&path)?;
    let (inner, warnings) = Ionex::parse_with_warnings(&data).map_err(to_ionex_err)?;
    let py_warnings = warnings
        .into_iter()
        .map(|w| PyIonexWarning { inner: w })
        .collect();
    Ok(PyIonexParseResult {
        ionex: PyIonex { inner },
        warnings: py_warnings,
    })
}

/// Top-level scalar IONEX slant-delay calculation convenience.
#[pyfunction]
#[pyo3(signature = (ionex, lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch_j2000_s, frequency_hz, hold_out_of_coverage=false))]
#[allow(clippy::too_many_arguments)]
fn ionex_slant_delay(
    ionex: &PyIonex,
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    epoch_j2000_s: i64,
    frequency_hz: f64,
    hold_out_of_coverage: bool,
) -> PyResult<f64> {
    ionex.slant_delay(
        lat_deg,
        lon_deg,
        azimuth_deg,
        elevation_deg,
        epoch_j2000_s,
        frequency_hz,
        hold_out_of_coverage,
    )
}

/// Top-level policy-aware IONEX slant-delay calculation returning evaluation and status.
#[pyfunction]
#[pyo3(signature = (ionex, lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch_j2000_s, frequency_hz, policy=None))]
#[allow(clippy::too_many_arguments)]
fn ionex_slant_delay_with_policy(
    ionex: &PyIonex,
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    epoch_j2000_s: i64,
    frequency_hz: f64,
    policy: Option<&PyIonexSlantPolicy>,
) -> PyResult<PyIonexSlantDelayEvaluation> {
    ionex.slant_delay_with_policy(
        lat_deg,
        lon_deg,
        azimuth_deg,
        elevation_deg,
        epoch_j2000_s,
        frequency_hz,
        policy,
    )
}

/// Top-level batch IONEX slant-delay evaluation returning per-request results.
#[pyfunction]
#[pyo3(signature = (ionex, requests, policy=None))]
fn ionex_slant_delay_results(
    ionex: &PyIonex,
    requests: Vec<PyRef<'_, PyIonexSlantRequest>>,
    policy: Option<&PyIonexSlantPolicy>,
) -> PyResult<Vec<PyIonexSlantBatchResult>> {
    ionex.slant_delays_batch_results(requests, policy)
}

/// Top-level scalar IONEX query retaining the supplied `ClockInstant` precision.
#[pyfunction]
#[pyo3(signature = (ionex, lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch, frequency_hz, hold_out_of_coverage=false))]
#[allow(clippy::too_many_arguments)]
fn ionex_slant_delay_at_instant(
    py: Python<'_>,
    ionex: &PyIonex,
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    epoch: &PyClockInstant,
    frequency_hz: f64,
    hold_out_of_coverage: bool,
) -> PyResult<f64> {
    ionex.slant_delay_at_instant(
        py,
        lat_deg,
        lon_deg,
        azimuth_deg,
        elevation_deg,
        epoch,
        frequency_hz,
        hold_out_of_coverage,
    )
}

/// Top-level policy-aware IONEX query retaining the supplied instant.
#[pyfunction]
#[pyo3(signature = (ionex, lat_deg, lon_deg, azimuth_deg, elevation_deg, epoch, frequency_hz, policy=None))]
#[allow(clippy::too_many_arguments)]
fn ionex_slant_delay_at_instant_with_policy(
    py: Python<'_>,
    ionex: &PyIonex,
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    epoch: &PyClockInstant,
    frequency_hz: f64,
    policy: Option<&PyIonexSlantPolicy>,
) -> PyResult<PyIonexSlantDelayEvaluation> {
    ionex.slant_delay_at_instant_with_policy(
        py,
        lat_deg,
        lon_deg,
        azimuth_deg,
        elevation_deg,
        epoch,
        frequency_hz,
        policy,
    )
}

/// Top-level per-request instant-based IONEX batch evaluation.
#[pyfunction]
#[pyo3(signature = (ionex, requests, policy=None))]
fn ionex_slant_delay_results_at_instants(
    ionex: &PyIonex,
    requests: Vec<PyRef<'_, PyIonexInstantSlantRequest>>,
    policy: Option<&PyIonexSlantPolicy>,
) -> PyResult<Vec<PyIonexSlantBatchResult>> {
    ionex.slant_delays_at_instants_batch_results(requests, policy)
}

/// GPS broadcast Klobuchar ionospheric group delay in the model's native units
/// (positive metres). This is the bit-exact (0-ULP) entry: latitude/longitude
/// and azimuth/elevation are in **degrees**, `t_gps_s` is the GPS
/// **second-of-day** in `[0, 86400)`, and no angle or time conversion happens at
/// the boundary. `alpha` (a0..a3) and `beta` (b0..b3) are the eight transmitted
/// GPS Klobuchar coefficients. The L1 delay is scaled to `frequency_hz` by the
/// dispersive `(f_l1 / f)^2` factor. Raises `ValueError` on out-of-range or
/// non-finite input.
#[pyfunction]
#[pyo3(signature = (alpha, beta, lat_deg, lon_deg, az_deg, el_deg, t_gps_s, frequency_hz))]
#[allow(clippy::too_many_arguments)]
fn klobuchar_native(
    alpha: [f64; 4],
    beta: [f64; 4],
    lat_deg: f64,
    lon_deg: f64,
    az_deg: f64,
    el_deg: f64,
    t_gps_s: f64,
    frequency_hz: f64,
) -> PyResult<f64> {
    let params = KlobucharParams { alpha, beta };
    core_klobuchar_native(
        &params,
        lat_deg,
        lon_deg,
        az_deg,
        el_deg,
        t_gps_s,
        frequency_hz,
    )
    .map_err(|err| match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid Klobuchar input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    })
}

/// Galileo NeQuick-G single-frequency ionospheric group delay in the model's
/// native input units (positive metres).
///
/// `ai0` / `ai1` / `ai2` are the three broadcast NeQuick-G effective-ionisation
/// coefficients. Receiver latitude/longitude and the satellite elevation are in
/// **degrees**; `t_gal_s` is the Galileo-system **second of day** and
/// `day_of_year` the fractional day of year. `frequency_hz` is the carrier the
/// dispersive delay is reported on. This is the native (no Instant) entry
/// parallel to [`klobuchar_native`]; it never consumes GPS Klobuchar
/// coefficients. Raises `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (ai0, ai1, ai2, lat_deg, lon_deg, el_deg, t_gal_s, day_of_year, frequency_hz))]
#[allow(clippy::too_many_arguments)]
fn galileo_nequick_g_native(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    lat_deg: f64,
    lon_deg: f64,
    el_deg: f64,
    t_gal_s: f64,
    day_of_year: f64,
    frequency_hz: f64,
) -> PyResult<f64> {
    let coeffs = GalileoNequickCoeffs { ai0, ai1, ai2 };
    let eval = GalileoNequickEval {
        lat_deg,
        lon_deg,
        el_deg,
        t_gal_s,
        day_of_year,
        frequency_hz,
    };
    core_galileo_nequick_g_native(&coeffs, eval).map_err(|err| match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid NeQuick-G input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    })
}

/// Assemble the [`NequickGRayEval`] receiver-to-satellite ray the full NeQuick-G
/// model integrates over, from the boundary's degrees/metres/hours inputs. No
/// conversion happens: the reference algorithm consumes these units directly.
#[allow(clippy::too_many_arguments)]
fn nequick_g_ray(
    month: u8,
    utc_hours: f64,
    station_lon_deg: f64,
    station_lat_deg: f64,
    station_height_m: f64,
    satellite_lon_deg: f64,
    satellite_lat_deg: f64,
    satellite_height_m: f64,
) -> NequickGRayEval {
    NequickGRayEval {
        month,
        utc_hours,
        station_lon_deg,
        station_lat_deg,
        station_height_m,
        satellite_lon_deg,
        satellite_lat_deg,
        satellite_height_m,
    }
}

/// Map a NeQuick-G full-integration failure into a Pythonic error, preserving
/// the engine message.
fn nequick_g_error(err: sidereon_core::Error) -> PyErr {
    match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid NeQuick-G input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    }
}

/// Full Galileo NeQuick-G slant total electron content along a receiver-to-
/// satellite ray, in TECU.
///
/// This is the reference-grade three-dimensional NeQuick 2 profiler integrated
/// along the ray (the full model), distinct from the compact broadcast-driven
/// [`galileo_nequick_g_native`]. `ai0` / `ai1` / `ai2` are the three broadcast
/// effective-ionisation coefficients. `month` is `1..=12` and `utc_hours` the
/// UTC time of day in `[0, 24]`. Station and satellite longitudes/latitudes are
/// in degrees and heights in metres above the reference sphere. Raises
/// `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    ai0, ai1, ai2, month, utc_hours,
    station_lon_deg, station_lat_deg, station_height_m,
    satellite_lon_deg, satellite_lat_deg, satellite_height_m,
))]
#[allow(clippy::too_many_arguments)]
fn nequick_g_stec_tecu(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    month: u8,
    utc_hours: f64,
    station_lon_deg: f64,
    station_lat_deg: f64,
    station_height_m: f64,
    satellite_lon_deg: f64,
    satellite_lat_deg: f64,
    satellite_height_m: f64,
) -> PyResult<f64> {
    let coeffs = GalileoNequickCoeffs { ai0, ai1, ai2 };
    let ray = nequick_g_ray(
        month,
        utc_hours,
        station_lon_deg,
        station_lat_deg,
        station_height_m,
        satellite_lon_deg,
        satellite_lat_deg,
        satellite_height_m,
    );
    core_nequick_g_stec_tecu(&coeffs, &ray).map_err(nequick_g_error)
}

/// Full Galileo NeQuick-G slant ionospheric group delay (positive metres) on
/// `frequency_hz`.
///
/// The full three-dimensional slant TEC from [`nequick_g_stec_tecu`] is mapped to
/// a group delay with the dispersive `40.3e16 / f^2` relation. Inputs match
/// [`nequick_g_stec_tecu`]; `frequency_hz` is the carrier the delay is reported
/// on. Raises `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    ai0, ai1, ai2, month, utc_hours,
    station_lon_deg, station_lat_deg, station_height_m,
    satellite_lon_deg, satellite_lat_deg, satellite_height_m, frequency_hz,
))]
#[allow(clippy::too_many_arguments)]
fn nequick_g_delay_m(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    month: u8,
    utc_hours: f64,
    station_lon_deg: f64,
    station_lat_deg: f64,
    station_height_m: f64,
    satellite_lon_deg: f64,
    satellite_lat_deg: f64,
    satellite_height_m: f64,
    frequency_hz: f64,
) -> PyResult<f64> {
    let coeffs = GalileoNequickCoeffs { ai0, ai1, ai2 };
    let ray = nequick_g_ray(
        month,
        utc_hours,
        station_lon_deg,
        station_lat_deg,
        station_height_m,
        satellite_lon_deg,
        satellite_lat_deg,
        satellite_height_m,
    );
    core_nequick_g_delay_m(&coeffs, &ray, frequency_hz).map_err(nequick_g_error)
}

/// Build the core split-Julian-date UTC [`Instant`] the ionosphere dispatcher
/// consumes, from civil-calendar fields, via `Instant::from_utc_civil`.
fn instant_from_utc_civil(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
) -> PyResult<Instant> {
    Instant::from_utc_civil(year, month, day, hour, minute, second)
        .map_err(|err| crate::time_model_error::py_error(err, format!("invalid epoch: {err}")))
}

#[allow(clippy::too_many_arguments)]
fn iono_delay(
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    epoch: Instant,
    frequency_hz: f64,
    model: &IonoModel,
) -> PyResult<f64> {
    let receiver = Wgs84Geodetic::new(lat_deg * DEG_TO_RAD, lon_deg * DEG_TO_RAD, height_m)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    core_ionosphere_delay(
        receiver,
        elevation_deg * DEG_TO_RAD,
        azimuth_deg * DEG_TO_RAD,
        epoch,
        frequency_hz,
        model,
    )
    .map_err(|err| match err {
        sidereon_core::Error::InvalidInput(message) => {
            PyValueError::new_err(format!("invalid ionosphere input: {message}"))
        }
        other => to_solve_err(other.to_string()),
    })
}

/// GPS broadcast Klobuchar ionospheric group delay (positive metres) for a
/// civil UTC epoch.
///
/// Delegates to the core `ionosphere_delay` dispatcher with a Klobuchar model,
/// building the epoch `Instant` via `Instant::from_utc_civil`. Receiver
/// latitude/longitude and the satellite azimuth/elevation are in degrees;
/// `height_m` is the receiver ellipsoidal height. `alpha` (a0..a3) and `beta`
/// (b0..b3) are the eight transmitted Klobuchar coefficients. Raises
/// `ValueError` on out-of-range or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    alpha, beta, lat_deg, lon_deg, azimuth_deg, elevation_deg,
    year, month, day, hour, minute, second, frequency_hz, height_m=0.0
))]
#[allow(clippy::too_many_arguments)]
fn ionosphere_delay_klobuchar(
    alpha: [f64; 4],
    beta: [f64; 4],
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
    frequency_hz: f64,
    height_m: f64,
) -> PyResult<f64> {
    let epoch = instant_from_utc_civil(year, month, day, hour, minute, second)?;
    let model = IonoModel::Klobuchar(KlobucharParams { alpha, beta });
    iono_delay(
        lat_deg,
        lon_deg,
        height_m,
        azimuth_deg,
        elevation_deg,
        epoch,
        frequency_hz,
        &model,
    )
}

/// Galileo NeQuick-G ionospheric group delay (positive metres) for a civil UTC
/// epoch.
///
/// Delegates to the core `ionosphere_delay` dispatcher with a NeQuick-G model,
/// building the epoch `Instant` via `Instant::from_utc_civil`; the dispatcher
/// derives the Galileo second-of-day and day-of-year from that epoch. Receiver
/// latitude/longitude and the satellite azimuth/elevation are in degrees;
/// `height_m` is the receiver ellipsoidal height. `ai0` / `ai1` / `ai2` are the
/// three broadcast NeQuick-G coefficients. Raises `ValueError` on out-of-range
/// or non-finite input.
#[pyfunction]
#[pyo3(signature = (
    ai0, ai1, ai2, lat_deg, lon_deg, azimuth_deg, elevation_deg,
    year, month, day, hour, minute, second, frequency_hz, height_m=0.0
))]
#[allow(clippy::too_many_arguments)]
fn ionosphere_delay_nequick(
    ai0: f64,
    ai1: f64,
    ai2: f64,
    lat_deg: f64,
    lon_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: f64,
    frequency_hz: f64,
    height_m: f64,
) -> PyResult<f64> {
    let epoch = instant_from_utc_civil(year, month, day, hour, minute, second)?;
    let model = IonoModel::GalileoNequickG(GalileoNequickCoeffs { ai0, ai1, ai2 });
    iono_delay(
        lat_deg,
        lon_deg,
        height_m,
        azimuth_deg,
        elevation_deg,
        epoch,
        frequency_hz,
        &model,
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIonexMappingFunction>()?;
    m.add_class::<PyIonexMappingDeclaration>()?;
    m.add_class::<PyIonexAssumedMapping>()?;
    m.add_class::<PyIonexCoveragePolicy>()?;
    m.add_class::<PyIonexMissingNodePolicy>()?;
    m.add_class::<PyIonexMappingPolicy>()?;
    m.add_class::<PyIonexSlantPolicy>()?;
    m.add_class::<PyIonexCoverageError>()?;
    m.add_class::<PyIonexMissingNodes>()?;
    m.add_class::<PyIonexNodeGap>()?;
    m.add_class::<PyIonexSlantDelayStatus>()?;
    m.add_class::<PyIonexSlantDelayEvaluation>()?;
    m.add_class::<PyIonexSlantRequest>()?;
    m.add_class::<PyIonexInstantSlantRequest>()?;
    m.add_class::<PyIonexSlantRefusal>()?;
    m.add_class::<PyIonexEpochErrorDetail>()?;
    m.add_class::<PyIonexSlantBatchResult>()?;
    m.add_class::<PyIonexDiagnosticEpoch>()?;
    m.add_class::<PyIonexWarning>()?;
    m.add_class::<PyIonexParseResult>()?;
    m.add_class::<PyIonexHeader>()?;
    m.add_class::<PyTecSample>()?;
    m.add_class::<PyTecGridSamples>()?;
    m.add_class::<PyIonex>()?;
    m.add_function(wrap_pyfunction!(load_ionex, m)?)?;
    m.add_function(wrap_pyfunction!(load_ionex_with_warnings, m)?)?;
    m.add_function(wrap_pyfunction!(ionex_slant_delay, m)?)?;
    m.add_function(wrap_pyfunction!(ionex_slant_delay_with_policy, m)?)?;
    m.add_function(wrap_pyfunction!(ionex_slant_delay_results, m)?)?;
    m.add_function(wrap_pyfunction!(ionex_slant_delay_at_instant, m)?)?;
    m.add_function(wrap_pyfunction!(
        ionex_slant_delay_at_instant_with_policy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(ionex_slant_delay_results_at_instants, m)?)?;
    m.add_function(wrap_pyfunction!(klobuchar_native, m)?)?;
    m.add_function(wrap_pyfunction!(galileo_nequick_g_native, m)?)?;
    m.add_function(wrap_pyfunction!(nequick_g_stec_tecu, m)?)?;
    m.add_function(wrap_pyfunction!(nequick_g_delay_m, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_delay_klobuchar, m)?)?;
    m.add_function(wrap_pyfunction!(ionosphere_delay_nequick, m)?)?;
    Ok(())
}
