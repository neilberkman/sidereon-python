//! Station displacement tide models (solid-earth, pole, ocean loading).
//!
//! Thin INTERFACE over `sidereon_core::tides`. It marshals the ITRF station
//! vector, calendar date/hour, and the model's external drivers (Sun/Moon
//! vectors, pole coordinates, or per-station BLQ coefficients) into
//! [`solid_earth_tide`](sidereon_core::tides::solid_earth_tide),
//! [`solid_earth_pole_tide`](sidereon_core::tides::solid_earth_pole_tide), and
//! [`ocean_tide_loading`](sidereon_core::tides::ocean_tide_loading), returning
//! the displacement vectors as numpy arrays. No tide arithmetic lives here.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::Bound;

use sidereon_core::tides::{
    ocean_tide_loading as core_ocean_tide_loading,
    parse_ocean_loading_blq_block as core_parse_ocean_loading_blq_block,
    parse_ocean_loading_blq_blocks as core_parse_ocean_loading_blq_blocks,
    solid_earth_pole_tide as core_pole_tide, solid_earth_tide as core_solid_earth_tide,
    solid_earth_tide_with_constants as core_solid_earth_tide_with_constants,
    write_ocean_loading_blq_blocks as core_write_ocean_loading_blq_blocks, BlqParseErrorKind,
    BlqWriteErrorKind, OceanLoadingBlq, OceanLoadingBlqBlock, OceanLoadingBlqComment,
    OceanLoadingBlqCommentPlacement, OceanTideConstituent, StationDisplacement,
    StationDisplacementEpoch, StationDisplacementOptions, StationDisplacementPosition,
    StationPolarMotion, StationTideConstants, TideError, TideInputErrorKind,
    NUM_OCEAN_CONSTITUENTS, OCEAN_LOADING_CONSTITUENTS,
};

use crate::blq_error_type;
use crate::marshal::{fixed_array, FinitePolicy};
use crate::ppp_corrections::PyOceanLoadingBlq;

#[pyclass(
    name = "OceanTideConstituent",
    module = "sidereon._sidereon",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum PyOceanTideConstituent {
    M2,
    S2,
    N2,
    K2,
    K1,
    O1,
    P1,
    Q1,
    Mf,
    Mm,
    Ssa,
}

#[pyclass(name = "TideInputErrorKind", module = "sidereon._sidereon", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
enum PyTideInputErrorKind {
    MISSING,
    NON_FINITE,
    NOT_POSITIVE,
    NEGATIVE,
    OUT_OF_RANGE,
    FLOAT_PARSE,
    INT_PARSE,
    INVALID_CIVIL_DATE,
    INVALID_CIVIL_TIME,
}

impl From<TideInputErrorKind> for PyTideInputErrorKind {
    fn from(value: TideInputErrorKind) -> Self {
        match value {
            TideInputErrorKind::Missing => Self::MISSING,
            TideInputErrorKind::NonFinite => Self::NON_FINITE,
            TideInputErrorKind::NotPositive => Self::NOT_POSITIVE,
            TideInputErrorKind::Negative => Self::NEGATIVE,
            TideInputErrorKind::OutOfRange => Self::OUT_OF_RANGE,
            TideInputErrorKind::FloatParse => Self::FLOAT_PARSE,
            TideInputErrorKind::IntParse => Self::INT_PARSE,
            TideInputErrorKind::InvalidCivilDate => Self::INVALID_CIVIL_DATE,
            TideInputErrorKind::InvalidCivilTime => Self::INVALID_CIVIL_TIME,
        }
    }
}

impl From<OceanTideConstituent> for PyOceanTideConstituent {
    fn from(value: OceanTideConstituent) -> Self {
        match value {
            OceanTideConstituent::M2 => Self::M2,
            OceanTideConstituent::S2 => Self::S2,
            OceanTideConstituent::N2 => Self::N2,
            OceanTideConstituent::K2 => Self::K2,
            OceanTideConstituent::K1 => Self::K1,
            OceanTideConstituent::O1 => Self::O1,
            OceanTideConstituent::P1 => Self::P1,
            OceanTideConstituent::Q1 => Self::Q1,
            OceanTideConstituent::Mf => Self::Mf,
            OceanTideConstituent::Mm => Self::Mm,
            OceanTideConstituent::Ssa => Self::Ssa,
        }
    }
}

#[pymethods]
impl PyOceanTideConstituent {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::M2 => "M2",
            Self::S2 => "S2",
            Self::N2 => "N2",
            Self::K2 => "K2",
            Self::K1 => "K1",
            Self::O1 => "O1",
            Self::P1 => "P1",
            Self::Q1 => "Q1",
            Self::Mf => "Mf",
            Self::Mm => "Mm",
            Self::Ssa => "Ssa",
        }
    }
}

#[pyclass(name = "StationPolarMotion", module = "sidereon._sidereon")]
#[derive(Clone, Copy)]
struct PyStationPolarMotion {
    inner: StationPolarMotion,
}

#[pymethods]
impl PyStationPolarMotion {
    #[new]
    fn new(xp_arcsec: f64, yp_arcsec: f64) -> Self {
        Self {
            inner: StationPolarMotion::from_arcseconds(xp_arcsec, yp_arcsec),
        }
    }

    #[getter]
    fn xp_arcsec(&self) -> f64 {
        self.inner.xp_arcsec
    }

    #[getter]
    fn yp_arcsec(&self) -> f64 {
        self.inner.yp_arcsec
    }
}

#[pyclass(name = "StationDisplacementPosition", module = "sidereon._sidereon")]
#[derive(Clone, Copy)]
struct PyStationDisplacementPosition {
    inner: StationDisplacementPosition,
}

#[pymethods]
impl PyStationDisplacementPosition {
    #[staticmethod]
    fn from_ecef_m(py: Python<'_>, ecef_m: [f64; 3]) -> PyResult<Self> {
        StationDisplacementPosition::from_ecef_m(ecef_m)
            .map(|inner| Self { inner })
            .map_err(|error| tide_err(py, error))
    }

    #[staticmethod]
    fn from_geodetic(lat_rad: f64, lon_rad: f64, height_m: f64) -> PyResult<Self> {
        let position = sidereon_core::frame::Wgs84Geodetic::new(lat_rad, lon_rad, height_m)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self {
            inner: position.into(),
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner {
            StationDisplacementPosition::Ecef(_) => "ecef",
            StationDisplacementPosition::Geodetic(_) => "geodetic",
        }
    }

    #[getter]
    fn ecef_m(&self) -> Option<[f64; 3]> {
        match self.inner {
            StationDisplacementPosition::Ecef(position) => Some(position.as_array()),
            StationDisplacementPosition::Geodetic(_) => None,
        }
    }

    #[getter]
    fn geodetic(&self) -> Option<(f64, f64, f64)> {
        match self.inner {
            StationDisplacementPosition::Ecef(_) => None,
            StationDisplacementPosition::Geodetic(position) => {
                Some((position.lat_rad, position.lon_rad, position.height_m))
            }
        }
    }
}

#[pyclass(name = "StationDisplacementEpoch", module = "sidereon._sidereon")]
#[derive(Clone, Copy)]
struct PyStationDisplacementEpoch {
    inner: StationDisplacementEpoch,
}

#[pymethods]
impl PyStationDisplacementEpoch {
    #[staticmethod]
    fn from_utc(year: i32, month: u8, day: u8, hour: u8, minute: u8, second: f64) -> Self {
        Self {
            inner: StationDisplacementEpoch::from_utc(year, month, day, hour, minute, second),
        }
    }

    fn with_polar_motion_arcsec(&self, xp_arcsec: f64, yp_arcsec: f64) -> Self {
        Self {
            inner: self.inner.with_polar_motion_arcsec(xp_arcsec, yp_arcsec),
        }
    }

    #[getter]
    fn year(&self) -> i32 {
        self.inner.year
    }

    #[getter]
    fn month(&self) -> u8 {
        self.inner.month
    }

    #[getter]
    fn day(&self) -> u8 {
        self.inner.day
    }

    #[getter]
    fn hour(&self) -> u8 {
        self.inner.hour
    }

    #[getter]
    fn minute(&self) -> u8 {
        self.inner.minute
    }

    #[getter]
    fn second(&self) -> f64 {
        self.inner.second
    }

    #[getter]
    fn polar_motion(&self) -> Option<PyStationPolarMotion> {
        self.inner
            .polar_motion
            .map(|inner| PyStationPolarMotion { inner })
    }
}

#[pyclass(name = "StationDisplacementOptions", module = "sidereon._sidereon")]
#[derive(Clone)]
struct PyStationDisplacementOptions {
    solid_earth_tide: bool,
    pole_tide: bool,
    ocean_loading: Option<PyOceanLoadingBlq>,
    solid_earth_tide_constants: PyStationTideConstants,
}

#[pymethods]
impl PyStationDisplacementOptions {
    #[new]
    #[pyo3(signature = (solid_earth_tide=true, pole_tide=false, ocean_loading=None, solid_earth_tide_constants=PyStationTideConstants::CONVENTIONS))]
    fn new(
        solid_earth_tide: bool,
        pole_tide: bool,
        ocean_loading: Option<PyOceanLoadingBlq>,
        solid_earth_tide_constants: PyStationTideConstants,
    ) -> Self {
        Self {
            solid_earth_tide,
            pole_tide,
            ocean_loading,
            solid_earth_tide_constants,
        }
    }

    #[getter]
    fn solid_earth_tide(&self) -> bool {
        self.solid_earth_tide
    }

    #[getter]
    fn pole_tide(&self) -> bool {
        self.pole_tide
    }

    #[getter]
    fn ocean_loading(&self) -> Option<PyOceanLoadingBlq> {
        self.ocean_loading.clone()
    }

    #[getter]
    fn solid_earth_tide_constants(&self) -> PyStationTideConstants {
        self.solid_earth_tide_constants
    }
}

#[pyclass(name = "StationDisplacementBatchRow", module = "sidereon._sidereon")]
struct PyStationDisplacementBatchRow {
    value: Option<PyStationDisplacement>,
    error_kind: Option<String>,
    error_message: Option<String>,
    error_details: Vec<(String, String)>,
}

#[pymethods]
impl PyStationDisplacementBatchRow {
    #[getter]
    fn value(&self) -> Option<PyStationDisplacement> {
        self.value
    }

    #[getter]
    fn error_kind(&self) -> Option<&str> {
        self.error_kind.as_deref()
    }

    #[getter]
    fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    #[getter]
    fn error_details(&self) -> Vec<(String, String)> {
        self.error_details.clone()
    }
}

/// Map a core tide failure to a Python `ValueError`, preserving the engine
/// message.
fn tide_err(py: Python<'_>, err: TideError) -> PyErr {
    let (kind, details) = match &err {
        TideError::InvalidInput { field, kind } => {
            let details = PyDict::new(py);
            let _ = details.set_item("field", field);
            let _ = details.set_item("reason", format!("{kind:?}"));
            ("invalid_input", details)
        }
        TideError::TimeScale(source) => {
            let details = PyDict::new(py);
            let _ = set_coverage_details(&details, source);
            ("time_scale", details)
        }
        TideError::FrameTransform(source) => {
            let details = PyDict::new(py);
            let _ = set_frame_transform_details(&details, source);
            ("frame_transform", details)
        }
        TideError::SunMoon(source) => {
            let details = PyDict::new(py);
            let _ = set_sun_moon_details(&details, source);
            ("sun_moon", details)
        }
        TideError::MissingInput { field } => {
            let details = PyDict::new(py);
            let _ = details.set_item("field", field);
            ("missing_input", details)
        }
        TideError::BlqParse { line, kind } => {
            let details = PyDict::new(py);
            let _ = details.set_item("line", line);
            let _ = details.set_item("reason", format!("{kind:?}"));
            ("blq_parse", details)
        }
        TideError::BlqWrite { block, kind } => {
            let details = PyDict::new(py);
            let _ = details.set_item("block", block);
            let _ = details.set_item("reason", format!("{kind:?}"));
            ("blq_write", details)
        }
    };
    let error_type = match crate::tide_evaluation_error_type(py) {
        Ok(error_type) => error_type,
        Err(error) => return error,
    };
    let exception = PyErr::from_type(error_type, err.to_string());
    let value = exception.value(py);
    let _ = value.setattr("kind", kind);
    let _ = value.setattr("details", details);
    exception
}

fn set_coverage_details(
    details: &Bound<'_, PyDict>,
    error: &sidereon_core::astro::time::CoverageError,
) -> PyResult<()> {
    use sidereon_core::astro::time::{CoverageError, DegradeReason};
    match error {
        CoverageError::InvalidInput { field, kind } => {
            details.set_item("source_kind", "InvalidInput")?;
            details.set_item("field", field)?;
            details.set_item("input_kind", format!("{kind:?}"))?;
        }
        CoverageError::OutsideCoverage(reason) => {
            details.set_item("source_kind", "OutsideCoverage")?;
            details.set_item(
                "reason",
                match reason {
                    DegradeReason::BeforeCoverage => "before_coverage",
                    DegradeReason::AfterCoverage => "after_coverage",
                },
            )?;
        }
    }
    Ok(())
}

fn set_frame_transform_details(
    details: &Bound<'_, PyDict>,
    error: &sidereon_core::astro::frames::transforms::FrameTransformError,
) -> PyResult<()> {
    use sidereon_core::astro::frames::transforms::FrameTransformError;
    match error {
        FrameTransformError::InvalidInput { field, reason } => {
            details.set_item("source_kind", "InvalidInput")?;
            details.set_item("field", field)?;
            details.set_item("reason", reason)?;
        }
        FrameTransformError::Ut1OutsideCoverage { reason } => {
            details.set_item("source_kind", "Ut1OutsideCoverage")?;
            details.set_item(
                "reason",
                match reason {
                    sidereon_core::astro::time::DegradeReason::BeforeCoverage => "before_coverage",
                    sidereon_core::astro::time::DegradeReason::AfterCoverage => "after_coverage",
                },
            )?;
        }
    }
    Ok(())
}

fn set_sun_moon_details(
    details: &Bound<'_, PyDict>,
    error: &sidereon_core::astro::bodies::SunMoonError,
) -> PyResult<()> {
    use sidereon_core::astro::bodies::SunMoonError;
    match error {
        SunMoonError::InvalidInput { field, reason } => {
            details.set_item("source_kind", "InvalidInput")?;
            details.set_item("field", field)?;
            details.set_item("reason", reason)?;
        }
        SunMoonError::FrameTransform(source) => {
            details.set_item("source_kind", "FrameTransform")?;
            let nested = PyDict::new(details.py());
            set_frame_transform_details(&nested, source)?;
            details.set_item("frame_transform", nested)?;
        }
    }
    Ok(())
}

pub(crate) fn tide_error_kind(error: &TideError) -> &'static str {
    match error {
        TideError::InvalidInput { .. } => "invalid_input",
        TideError::TimeScale(_) => "time_scale",
        TideError::FrameTransform(_) => "frame_transform",
        TideError::SunMoon(_) => "sun_moon",
        TideError::MissingInput { .. } => "missing_input",
        TideError::BlqParse { .. } => "blq_parse",
        TideError::BlqWrite { .. } => "blq_write",
    }
}

pub(crate) fn tide_error_details(error: &TideError) -> Vec<(String, String)> {
    match error {
        TideError::InvalidInput { field, kind } => vec![
            ("field".to_owned(), (*field).to_owned()),
            ("reason".to_owned(), format!("{kind:?}")),
        ],
        TideError::TimeScale(sidereon_core::astro::time::CoverageError::InvalidInput {
            field,
            kind,
        }) => vec![
            ("source_kind".to_owned(), "InvalidInput".to_owned()),
            ("field".to_owned(), (*field).to_owned()),
            ("input_kind".to_owned(), format!("{kind:?}")),
        ],
        TideError::TimeScale(sidereon_core::astro::time::CoverageError::OutsideCoverage(reason)) => {
            vec![
                ("source_kind".to_owned(), "OutsideCoverage".to_owned()),
                (
                    "reason".to_owned(),
                    match reason {
                        sidereon_core::astro::time::DegradeReason::BeforeCoverage => {
                            "before_coverage"
                        }
                        sidereon_core::astro::time::DegradeReason::AfterCoverage => {
                            "after_coverage"
                        }
                    }
                    .to_owned(),
                ),
            ]
        }
        TideError::FrameTransform(sidereon_core::astro::frames::transforms::FrameTransformError::InvalidInput { field, reason }) => vec![
            ("source_kind".to_owned(), "InvalidInput".to_owned()),
            ("field".to_owned(), (*field).to_owned()),
            ("reason".to_owned(), (*reason).to_owned()),
        ],
        TideError::FrameTransform(sidereon_core::astro::frames::transforms::FrameTransformError::Ut1OutsideCoverage { reason }) => vec![
            ("source_kind".to_owned(), "Ut1OutsideCoverage".to_owned()),
            ("reason".to_owned(), match reason {
                sidereon_core::astro::time::DegradeReason::BeforeCoverage => "before_coverage",
                sidereon_core::astro::time::DegradeReason::AfterCoverage => "after_coverage",
            }.to_owned()),
        ],
        TideError::SunMoon(sidereon_core::astro::bodies::SunMoonError::InvalidInput { field, reason }) => vec![
            ("source_kind".to_owned(), "InvalidInput".to_owned()),
            ("field".to_owned(), (*field).to_owned()),
            ("reason".to_owned(), (*reason).to_owned()),
        ],
        TideError::SunMoon(sidereon_core::astro::bodies::SunMoonError::FrameTransform(source)) => {
            match source {
                sidereon_core::astro::frames::transforms::FrameTransformError::InvalidInput {
                    field,
                    reason,
                } => vec![
                    ("source_kind".to_owned(), "FrameTransform".to_owned()),
                    ("frame_transform_kind".to_owned(), "InvalidInput".to_owned()),
                    ("field".to_owned(), (*field).to_owned()),
                    ("reason".to_owned(), (*reason).to_owned()),
                ],
                sidereon_core::astro::frames::transforms::FrameTransformError::Ut1OutsideCoverage {
                    reason,
                } => vec![
                    ("source_kind".to_owned(), "FrameTransform".to_owned()),
                    ("frame_transform_kind".to_owned(), "Ut1OutsideCoverage".to_owned()),
                    (
                        "reason".to_owned(),
                        match reason {
                            sidereon_core::astro::time::DegradeReason::BeforeCoverage => {
                                "before_coverage"
                            }
                            sidereon_core::astro::time::DegradeReason::AfterCoverage => {
                                "after_coverage"
                            }
                        }
                        .to_owned(),
                    ),
                ],
            }
        }
        TideError::MissingInput { field } => vec![("field".to_owned(), (*field).to_owned())],
        TideError::BlqParse { line, kind } => vec![
            ("line".to_owned(), line.to_string()),
            ("reason".to_owned(), format!("{kind:?}")),
        ],
        TideError::BlqWrite { block, kind } => vec![
            ("block".to_owned(), block.to_string()),
            ("reason".to_owned(), format!("{kind:?}")),
        ],
    }
}

fn evaluate_station_displacement(
    position: StationDisplacementPosition,
    epoch: StationDisplacementEpoch,
    options: &PyStationDisplacementOptions,
    validity: crate::PyValidityMode,
) -> Result<sidereon_core::astro::time::Validated<StationDisplacement>, TideError> {
    let ocean_loading = options
        .ocean_loading
        .as_ref()
        .map(PyOceanLoadingBlq::to_core);
    let mut core_options = StationDisplacementOptions::default();
    core_options.solid_earth_tide = options.solid_earth_tide;
    core_options.pole_tide = options.pole_tide;
    core_options.ocean_loading = ocean_loading.as_ref();
    core_options.solid_earth_tide_constants = options.solid_earth_tide_constants.into();
    sidereon_core::tides::station_displacement_ecef_m_with_validity(
        position,
        epoch,
        core_options,
        validity.into(),
    )
}

/// Read a `[3][NUM_OCEAN_CONSTITUENTS]` BLQ block from a sequence of three rows.
fn blq_block(name: &str, rows: &[Vec<f64>]) -> PyResult<[[f64; NUM_OCEAN_CONSTITUENTS]; 3]> {
    if rows.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have exactly 3 rows (radial, west, south), got {}",
            rows.len()
        )));
    }
    let mut block = [[0.0; NUM_OCEAN_CONSTITUENTS]; 3];
    for (component, row) in rows.iter().enumerate() {
        if row.len() != NUM_OCEAN_CONSTITUENTS {
            return Err(PyValueError::new_err(format!(
                "{name}[{component}] must have exactly {NUM_OCEAN_CONSTITUENTS} constituents, got {}",
                row.len()
            )));
        }
        block[component].copy_from_slice(row);
    }
    Ok(block)
}

/// Solid-earth tide displacement of an ITRF station, numpy `(3,)` ECEF metres.
///
/// `station_ecef_m` is the geocentric station position (m, ITRF); `year`,
/// `month`, `day` and the UTC fractional hour `fhr` set the epoch; `sun_ecef_m`
/// and `moon_ecef_m` are the geocentric Sun and Moon positions (same units as
/// the station). Returns the displacement to project onto the line of sight.
#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, fhr, sun_ecef_m, moon_ecef_m))]
#[allow(clippy::too_many_arguments)]
fn solid_earth_tide<'py>(
    py: Python<'py>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    sun_ecef_m: PyReadonlyArray1<'_, f64>,
    moon_ecef_m: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xsta = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let xsun = fixed_array::<3>("sun_ecef_m", &sun_ecef_m, FinitePolicy::RequireFinite)?;
    let xmon = fixed_array::<3>("moon_ecef_m", &moon_ecef_m, FinitePolicy::RequireFinite)?;
    let d = core_solid_earth_tide(&xsta, year, month, day, fhr, &xsun, &xmon)
        .map_err(|error| tide_err(py, error))?;
    Ok(PyArray1::from_slice(py, &d))
}

#[pyclass(
    name = "StationTideConstants",
    module = "sidereon._sidereon",
    eq,
    eq_int
)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub(crate) enum PyStationTideConstants {
    CONVENTIONS,
    IERS_ROUTINE,
}

impl From<PyStationTideConstants> for StationTideConstants {
    fn from(value: PyStationTideConstants) -> Self {
        match value {
            PyStationTideConstants::CONVENTIONS => Self::Conventions,
            PyStationTideConstants::IERS_ROUTINE => Self::IersRoutine,
        }
    }
}

#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, fhr, sun_ecef_m, moon_ecef_m, constants=PyStationTideConstants::CONVENTIONS))]
#[allow(clippy::too_many_arguments)]
fn station_tide_displacement<'py>(
    py: Python<'py>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    sun_ecef_m: PyReadonlyArray1<'_, f64>,
    moon_ecef_m: PyReadonlyArray1<'_, f64>,
    constants: PyStationTideConstants,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let station = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let sun = fixed_array::<3>("sun_ecef_m", &sun_ecef_m, FinitePolicy::RequireFinite)?;
    let moon = fixed_array::<3>("moon_ecef_m", &moon_ecef_m, FinitePolicy::RequireFinite)?;
    let displacement = core_solid_earth_tide_with_constants(
        &station,
        year,
        month,
        day,
        fhr,
        &sun,
        &moon,
        constants.into(),
    )
    .map_err(|error| tide_err(py, error))?;
    Ok(PyArray1::from_slice(py, &displacement))
}

#[pyclass(name = "StationDisplacement", module = "sidereon._sidereon")]
#[derive(Clone, Copy)]
struct PyStationDisplacement {
    value: StationDisplacement,
    degraded_reason: Option<&'static str>,
}

#[pymethods]
impl PyStationDisplacement {
    #[getter]
    fn ecef_m(&self) -> [f64; 3] {
        self.value.ecef_m
    }

    #[getter]
    fn solid_earth_tide_ecef_m(&self) -> Option<[f64; 3]> {
        self.value.solid_earth_tide_ecef_m
    }

    #[getter]
    fn pole_tide_ecef_m(&self) -> Option<[f64; 3]> {
        self.value.pole_tide_ecef_m
    }

    #[getter]
    fn ocean_loading_ecef_m(&self) -> Option<[f64; 3]> {
        self.value.ocean_loading_ecef_m
    }

    #[getter]
    fn ut1_degraded(&self) -> bool {
        self.degraded_reason.is_some()
    }

    #[getter]
    fn degradation_reason(&self) -> Option<&'static str> {
        self.degraded_reason
    }
}

#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, hour, minute, second, solid_earth_tide=true, pole_tide=false, ocean_loading=None, constants=PyStationTideConstants::CONVENTIONS, validity=crate::PyValidityMode::STRICT, polar_motion_arcsec=None))]
#[allow(clippy::too_many_arguments)]
fn station_displacement_ecef_m(
    py: Python<'_>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: u8,
    day: u8,
    hour: u8,
    minute: u8,
    second: f64,
    solid_earth_tide: bool,
    pole_tide: bool,
    ocean_loading: Option<&PyOceanLoadingBlq>,
    constants: PyStationTideConstants,
    validity: crate::PyValidityMode,
    polar_motion_arcsec: Option<(f64, f64)>,
) -> PyResult<PyStationDisplacement> {
    let position = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let position =
        StationDisplacementPosition::from_ecef_m(position).map_err(|error| tide_err(py, error))?;
    let epoch = StationDisplacementEpoch::from_utc(year, month, day, hour, minute, second);
    let epoch = match polar_motion_arcsec {
        Some((xp_arcsec, yp_arcsec)) => epoch.with_polar_motion_arcsec(xp_arcsec, yp_arcsec),
        None => epoch,
    };
    let ocean_loading = ocean_loading.map(PyOceanLoadingBlq::to_core);
    let mut options = StationDisplacementOptions::default();
    options.solid_earth_tide = solid_earth_tide;
    options.pole_tide = pole_tide;
    options.ocean_loading = ocean_loading.as_ref();
    options.solid_earth_tide_constants = constants.into();
    let result = sidereon_core::tides::station_displacement_ecef_m_with_validity(
        position,
        epoch,
        options,
        validity.into(),
    )
    .map_err(|error| tide_err(py, error))?;
    let degraded_reason = result.degraded.map(|reason| match reason {
        sidereon_core::astro::time::DegradeReason::BeforeCoverage => "before_coverage",
        sidereon_core::astro::time::DegradeReason::AfterCoverage => "after_coverage",
    });
    Ok(PyStationDisplacement {
        value: result.value,
        degraded_reason,
    })
}

#[pyfunction]
#[pyo3(signature = (position, epoch, options=None, validity=crate::PyValidityMode::STRICT))]
fn station_displacement_ecef_m_at_epoch(
    py: Python<'_>,
    position: &PyStationDisplacementPosition,
    epoch: &PyStationDisplacementEpoch,
    options: Option<&PyStationDisplacementOptions>,
    validity: crate::PyValidityMode,
) -> PyResult<PyStationDisplacement> {
    let default_options = PyStationDisplacementOptions {
        solid_earth_tide: true,
        pole_tide: false,
        ocean_loading: None,
        solid_earth_tide_constants: PyStationTideConstants::CONVENTIONS,
    };
    let selected_options = options.unwrap_or(&default_options);
    let result =
        evaluate_station_displacement(position.inner, epoch.inner, selected_options, validity)
            .map_err(|error| tide_err(py, error))?;
    let degraded_reason = result.degraded.map(|reason| match reason {
        sidereon_core::astro::time::DegradeReason::BeforeCoverage => "before_coverage",
        sidereon_core::astro::time::DegradeReason::AfterCoverage => "after_coverage",
    });
    Ok(PyStationDisplacement {
        value: result.value,
        degraded_reason,
    })
}

#[pyfunction]
#[pyo3(signature = (position, epochs, options=None, validity=crate::PyValidityMode::STRICT))]
fn station_displacement_ecef_m_batch(
    py: Python<'_>,
    position: &PyStationDisplacementPosition,
    epochs: Vec<Py<PyStationDisplacementEpoch>>,
    options: Option<&PyStationDisplacementOptions>,
    validity: crate::PyValidityMode,
) -> PyResult<Vec<PyStationDisplacementBatchRow>> {
    let default_options = PyStationDisplacementOptions {
        solid_earth_tide: true,
        pole_tide: false,
        ocean_loading: None,
        solid_earth_tide_constants: PyStationTideConstants::CONVENTIONS,
    };
    let selected_options = options.unwrap_or(&default_options);
    Ok(epochs
        .iter()
        .map(|epoch| {
            let epoch = epoch.borrow(py);
            match evaluate_station_displacement(
                position.inner,
                epoch.inner,
                selected_options,
                validity,
            ) {
                Ok(result) => {
                    let degraded_reason = result.degraded.map(|reason| match reason {
                        sidereon_core::astro::time::DegradeReason::BeforeCoverage => {
                            "before_coverage"
                        }
                        sidereon_core::astro::time::DegradeReason::AfterCoverage => {
                            "after_coverage"
                        }
                    });
                    PyStationDisplacementBatchRow {
                        value: Some(PyStationDisplacement {
                            value: result.value,
                            degraded_reason,
                        }),
                        error_kind: None,
                        error_message: None,
                        error_details: Vec::new(),
                    }
                }
                Err(error) => PyStationDisplacementBatchRow {
                    value: None,
                    error_kind: Some(tide_error_kind(&error).to_owned()),
                    error_message: Some(error.to_string()),
                    error_details: tide_error_details(&error),
                },
            }
        })
        .collect())
}

/// Solid-earth pole tide displacement of an ITRF station, numpy `(3,)` ECEF
/// metres.
///
/// `xp_arcsec` / `yp_arcsec` are the IERS polar-motion coordinates in arcseconds
/// at the epoch.
#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, fhr, xp_arcsec, yp_arcsec))]
#[allow(clippy::too_many_arguments)]
fn solid_earth_pole_tide<'py>(
    py: Python<'py>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xsta = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let d = core_pole_tide(&xsta, year, month, day, fhr, xp_arcsec, yp_arcsec)
        .map_err(|error| tide_err(py, error))?;
    Ok(PyArray1::from_slice(py, &d))
}

/// Ocean tide loading displacement of an ITRF station, numpy `(3,)` ECEF metres.
///
/// `amplitude_m` and `phase_deg` are the station's BLQ coefficients, each three
/// rows (radial/up, west, south) of 11 constituents in BLQ column order
/// `M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa`. Amplitudes are metres, phases are
/// Greenwich phase lags in degrees.
#[pyfunction]
#[pyo3(signature = (station_ecef_m, year, month, day, fhr, amplitude_m, phase_deg))]
#[allow(clippy::too_many_arguments)]
fn ocean_tide_loading<'py>(
    py: Python<'py>,
    station_ecef_m: PyReadonlyArray1<'_, f64>,
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    amplitude_m: Vec<Vec<f64>>,
    phase_deg: Vec<Vec<f64>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xsta = fixed_array::<3>(
        "station_ecef_m",
        &station_ecef_m,
        FinitePolicy::RequireFinite,
    )?;
    let blq = OceanLoadingBlq {
        amplitude_m: blq_block("amplitude_m", &amplitude_m)?,
        phase_deg: blq_block("phase_deg", &phase_deg)?,
    };
    let d = core_ocean_tide_loading(&xsta, year, month, day, fhr, &blq)
        .map_err(|error| tide_err(py, error))?;
    Ok(PyArray1::from_slice(py, &d))
}

#[pyfunction]
fn ocean_loading_constituents() -> Vec<PyOceanTideConstituent> {
    OCEAN_LOADING_CONSTITUENTS
        .into_iter()
        .map(Into::into)
        .collect()
}

/// The variant name and fields of a BLQ parse refusal reason.
fn parse_kind_details(
    dict: &Bound<'_, PyDict>,
    kind: &BlqParseErrorKind,
) -> PyResult<&'static str> {
    Ok(match kind {
        BlqParseErrorKind::Empty => "Empty",
        BlqParseErrorKind::MissingStation => "MissingStation",
        BlqParseErrorKind::MissingCoefficientRows {
            station,
            expected,
            found,
        } => {
            dict.set_item("station", station.as_str())?;
            dict.set_item("expected", *expected)?;
            dict.set_item("found", *found)?;
            "MissingCoefficientRows"
        }
        BlqParseErrorKind::TooManyCoefficientRows { station } => {
            dict.set_item("station", station.as_str())?;
            "TooManyCoefficientRows"
        }
        BlqParseErrorKind::WrongColumnCount { expected, found } => {
            dict.set_item("expected", *expected)?;
            dict.set_item("found", *found)?;
            "WrongColumnCount"
        }
        BlqParseErrorKind::InvalidNumber { token } => {
            dict.set_item("token", token.as_str())?;
            "InvalidNumber"
        }
        BlqParseErrorKind::NonFiniteNumber { token } => {
            dict.set_item("token", token.as_str())?;
            "NonFiniteNumber"
        }
        BlqParseErrorKind::UnsupportedConstituent { constituent } => {
            dict.set_item("constituent", constituent.as_str())?;
            "UnsupportedConstituent"
        }
        BlqParseErrorKind::DuplicateConstituent { constituent } => {
            dict.set_item("constituent", constituent.as_str())?;
            "DuplicateConstituent"
        }
        BlqParseErrorKind::MultipleBlocks { found } => {
            dict.set_item("found", *found)?;
            "MultipleBlocks"
        }
    })
}

/// The variant name and fields of a BLQ write refusal reason.
fn write_kind_details(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    kind: &BlqWriteErrorKind,
) -> PyResult<&'static str> {
    Ok(match kind {
        BlqWriteErrorKind::EmptyStation => "EmptyStation",
        BlqWriteErrorKind::StationLineBreak => "StationLineBreak",
        BlqWriteErrorKind::StationSurroundingWhitespace => "StationSurroundingWhitespace",
        BlqWriteErrorKind::StationReadsAsComment => "StationReadsAsComment",
        BlqWriteErrorKind::StationReadsAsHeader => "StationReadsAsHeader",
        BlqWriteErrorKind::StationReadsAsCoefficientRow => "StationReadsAsCoefficientRow",
        BlqWriteErrorKind::NonFiniteCoefficient { row, constituent } => {
            dict.set_item("row", *row)?;
            dict.set_item("constituent", constituent.label())?;
            "NonFiniteCoefficient"
        }
        BlqWriteErrorKind::CommentLineBreak { index } => {
            dict.set_item("index", *index)?;
            "CommentLineBreak"
        }
        BlqWriteErrorKind::NotACommentLine { index } => {
            dict.set_item("index", *index)?;
            "NotACommentLine"
        }
        BlqWriteErrorKind::CommentPlacementOutOfRange { index } => {
            dict.set_item("index", *index)?;
            "CommentPlacementOutOfRange"
        }
        BlqWriteErrorKind::InvalidHeader { index, kind } => {
            dict.set_item("index", *index)?;
            let header = PyDict::new(py);
            let reason = parse_kind_details(&header, kind)?;
            header.set_item("reason", reason)?;
            dict.set_item("header", header)?;
            "InvalidHeader"
        }
        BlqWriteErrorKind::AfterRowsBeforeAnotherBlock { index } => {
            dict.set_item("index", *index)?;
            "AfterRowsBeforeAnotherBlock"
        }
        BlqWriteErrorKind::CommentsOutOfPlacementOrder { index } => {
            dict.set_item("index", *index)?;
            "CommentsOutOfPlacementOrder"
        }
    })
}

/// The typed payload a `BlqError` carries on its `detail` attribute.
///
/// `kind` is `BlqParse` (with the one-based `line`, zero for a whole-input
/// failure) or `BlqWrite` (with the zero-based `block` index). `reason` is the
/// core reason variant name - `UnsupportedConstituent`, `NonFiniteCoefficient`
/// and so on - and `details()` holds `line` or `block`, `reason`, and the
/// reason's fields under their own keys. An `InvalidHeader` write reason nests
/// the parse reason the header was refused with under `header`.
#[pyclass(module = "sidereon._sidereon", name = "BlqErrorDetail")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyBlqErrorDetail {
    inner: TideError,
}

#[pymethods]
impl PyBlqErrorDetail {
    /// `BlqParse` or `BlqWrite`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            TideError::BlqParse { .. } => "BlqParse",
            TideError::BlqWrite { .. } => "BlqWrite",
            TideError::InvalidInput { .. } => "InvalidInput",
            TideError::TimeScale(_) => "TimeScale",
            TideError::FrameTransform(_) => "FrameTransform",
            TideError::SunMoon(_) => "SunMoon",
            TideError::MissingInput { .. } => "MissingInput",
        }
    }

    /// The core reason variant name.
    #[getter]
    fn reason(&self, py: Python<'_>) -> PyResult<Option<&'static str>> {
        let scratch = PyDict::new(py);
        Ok(match &self.inner {
            TideError::BlqParse { kind, .. } => Some(parse_kind_details(&scratch, kind)?),
            TideError::BlqWrite { kind, .. } => Some(write_kind_details(py, &scratch, kind)?),
            TideError::TimeScale(sidereon_core::astro::time::CoverageError::InvalidInput {
                ..
            }) => Some("InvalidInput"),
            TideError::TimeScale(sidereon_core::astro::time::CoverageError::OutsideCoverage(_)) => {
                Some("OutsideCoverage")
            }
            TideError::FrameTransform(
                sidereon_core::astro::frames::transforms::FrameTransformError::InvalidInput {
                    ..
                },
            ) => Some("InvalidInput"),
            TideError::FrameTransform(
                sidereon_core::astro::frames::transforms::FrameTransformError::Ut1OutsideCoverage {
                    ..
                },
            ) => Some("Ut1OutsideCoverage"),
            TideError::SunMoon(sidereon_core::astro::bodies::SunMoonError::InvalidInput {
                ..
            }) => Some("InvalidInput"),
            TideError::SunMoon(sidereon_core::astro::bodies::SunMoonError::FrameTransform(_)) => {
                Some("FrameTransform")
            }
            _ => None,
        })
    }

    /// One-based line of a parse refusal; zero for a whole-input failure.
    #[getter]
    fn line(&self) -> Option<usize> {
        match &self.inner {
            TideError::BlqParse { line, .. } => Some(*line),
            _ => None,
        }
    }

    /// Zero-based index of the block a write refusal names.
    #[getter]
    fn block(&self) -> Option<usize> {
        match &self.inner {
            TideError::BlqWrite { block, .. } => Some(*block),
            _ => None,
        }
    }

    /// Formatted core error message.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    /// Dictionary containing every field of this refusal.
    fn details<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        match &self.inner {
            TideError::BlqParse { line, kind } => {
                dict.set_item("line", *line)?;
                let reason = parse_kind_details(&dict, kind)?;
                dict.set_item("reason", reason)?;
            }
            TideError::BlqWrite { block, kind } => {
                dict.set_item("block", *block)?;
                let reason = write_kind_details(py, &dict, kind)?;
                dict.set_item("reason", reason)?;
            }
            TideError::InvalidInput { field, kind } => {
                dict.set_item("field", field)?;
                dict.set_item("reason", format!("{kind:?}"))?;
            }
            TideError::TimeScale(source) => set_coverage_details(&dict, source)?,
            TideError::FrameTransform(source) => set_frame_transform_details(&dict, source)?,
            TideError::SunMoon(source) => set_sun_moon_details(&dict, source)?,
            TideError::MissingInput { field } => dict.set_item("field", field)?,
        }
        Ok(dict)
    }

    fn __repr__(&self) -> String {
        format!(
            "BlqErrorDetail(kind=\"{}\", message={:?})",
            self.kind(),
            self.message()
        )
    }

    fn __eq__(&self, other: &PyBlqErrorDetail) -> bool {
        self.inner == other.inner
    }
}

/// Map a BLQ parse or write refusal to `BlqError` with its typed payload; any
/// other tide failure raises `ValueError` as before.
fn to_blq_err(py: Python<'_>, err: TideError) -> PyErr {
    if !matches!(err, TideError::BlqParse { .. } | TideError::BlqWrite { .. }) {
        return tide_err(py, err);
    }
    let ty = match blq_error_type(py) {
        Ok(ty) => ty,
        Err(e) => return e,
    };
    let py_err = PyErr::from_type(ty, err.to_string());
    let detail = match (PyBlqErrorDetail { inner: err }).into_pyobject(py) {
        Ok(detail) => detail,
        Err(e) => return e,
    };
    if let Err(e) = py_err.value(py).setattr("detail", detail) {
        return e;
    }
    py_err
}

pub(crate) fn ocean_loading_blq_block_from_text(
    py: Python<'_>,
    text: &str,
) -> PyResult<PyOceanLoadingBlqBlock> {
    OceanLoadingBlq::from_blq_block(text)
        .map(|inner| PyOceanLoadingBlqBlock { inner })
        .map_err(|error| to_blq_err(py, error))
}

/// One comment or column-order header line retained from a BLQ block.
///
/// `placement` is `before_station`, `before_row` (before the zero-based
/// coefficient `row`, `0..=5`; `row` is None for the other placements) or
/// `after_rows`. `line` is the line exactly as read, without its terminator.
/// The writer refuses, as `BlqError`, a line or placement the parser would not
/// read back unchanged.
#[pyclass(module = "sidereon._sidereon", name = "OceanLoadingBlqComment")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyOceanLoadingBlqComment {
    inner: OceanLoadingBlqComment,
}

#[pymethods]
impl PyOceanLoadingBlqComment {
    #[new]
    #[pyo3(signature = (line, placement, row=None))]
    fn new(line: String, placement: &str, row: Option<usize>) -> PyResult<Self> {
        let placement = match (placement, row) {
            ("before_station", None) => OceanLoadingBlqCommentPlacement::BeforeStation,
            ("before_row", Some(row)) => OceanLoadingBlqCommentPlacement::BeforeRow(row),
            ("after_rows", None) => OceanLoadingBlqCommentPlacement::AfterRows,
            ("before_row", None) => {
                return Err(PyValueError::new_err("placement \"before_row\" needs a row"))
            }
            ("before_station" | "after_rows", Some(_)) => {
                return Err(PyValueError::new_err(format!(
                    "placement {placement:?} takes no row"
                )))
            }
            (other, _) => {
                return Err(PyValueError::new_err(format!(
                    "unknown BLQ comment placement {other:?}; expected \"before_station\", \"before_row\" or \"after_rows\""
                )))
            }
        };
        Ok(Self {
            inner: OceanLoadingBlqComment { placement, line },
        })
    }

    /// The line exactly as read, without its terminator.
    #[getter]
    fn line(&self) -> &str {
        &self.inner.line
    }

    /// `before_station`, `before_row` or `after_rows`.
    #[getter]
    fn placement(&self) -> &'static str {
        match self.inner.placement {
            OceanLoadingBlqCommentPlacement::BeforeStation => "before_station",
            OceanLoadingBlqCommentPlacement::BeforeRow(_) => "before_row",
            OceanLoadingBlqCommentPlacement::AfterRows => "after_rows",
        }
    }

    /// The zero-based coefficient row a `before_row` line precedes; None
    /// otherwise.
    #[getter]
    fn row(&self) -> Option<usize> {
        match self.inner.placement {
            OceanLoadingBlqCommentPlacement::BeforeRow(row) => Some(row),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OceanLoadingBlqComment(line={:?}, placement={:?}, row={:?})",
            self.inner.line,
            self.placement(),
            self.row()
        )
    }

    fn __eq__(&self, other: &PyOceanLoadingBlqComment) -> bool {
        self.inner == other.inner
    }
}

/// One BLQ station block: the station name, its coefficients in the standard
/// column order, and every comment and column-order header line of the block
/// with its place.
#[pyclass(module = "sidereon._sidereon", name = "OceanLoadingBlqBlock")]
#[derive(Clone, Debug, PartialEq)]
pub struct PyOceanLoadingBlqBlock {
    inner: OceanLoadingBlqBlock,
}

#[pymethods]
impl PyOceanLoadingBlqBlock {
    #[staticmethod]
    fn from_blq_block(py: Python<'_>, text: &str) -> PyResult<Self> {
        OceanLoadingBlq::from_blq_block(text)
            .map(|inner| Self { inner })
            .map_err(|error| to_blq_err(py, error))
    }

    #[new]
    #[pyo3(signature = (station, coefficients, comments=None))]
    fn new(
        station: String,
        coefficients: &PyOceanLoadingBlq,
        comments: Option<Vec<PyOceanLoadingBlqComment>>,
    ) -> Self {
        Self {
            inner: OceanLoadingBlqBlock {
                station,
                coefficients: OceanLoadingBlq::from(coefficients),
                comments: comments
                    .unwrap_or_default()
                    .into_iter()
                    .map(|comment| comment.inner)
                    .collect(),
            },
        }
    }

    /// Station identifier, trimmed.
    #[getter]
    fn station(&self) -> &str {
        &self.inner.station
    }

    /// Coefficients in the standard column order.
    #[getter]
    fn coefficients(&self) -> PyOceanLoadingBlq {
        PyOceanLoadingBlq::from_core(self.inner.coefficients)
    }

    /// Comment and column-order header lines, in input order.
    #[getter]
    fn comments(&self) -> Vec<PyOceanLoadingBlqComment> {
        self.inner
            .comments
            .iter()
            .cloned()
            .map(|inner| PyOceanLoadingBlqComment { inner })
            .collect()
    }

    /// Write the block as standard BLQ text that reads back to an equal
    /// block: the retained comments at their placements, the station from the
    /// third column, and each row in the column order its retained header
    /// declares. A station, comment or coefficient the parser would not read
    /// back unchanged raises `BlqError`.
    fn to_blq_block(&self, py: Python<'_>) -> PyResult<String> {
        self.inner.to_blq_block().map_err(|err| to_blq_err(py, err))
    }

    fn __repr__(&self) -> String {
        format!(
            "OceanLoadingBlqBlock(station={:?}, comments={})",
            self.inner.station,
            self.inner.comments.len()
        )
    }

    fn __eq__(&self, other: &PyOceanLoadingBlqBlock) -> bool {
        self.inner == other.inner
    }
}

/// Parse one standard BLQ station block, keeping its comment and header
/// lines. A refusal raises `BlqError`.
#[pyfunction]
fn parse_ocean_loading_blq_block(py: Python<'_>, text: &str) -> PyResult<PyOceanLoadingBlqBlock> {
    core_parse_ocean_loading_blq_block(text)
        .map(|inner| PyOceanLoadingBlqBlock { inner })
        .map_err(|err| to_blq_err(py, err))
}

/// Parse every station block of a BLQ file. A column-order header stays in
/// force for the rows and blocks after it. A refusal raises `BlqError`.
#[pyfunction]
fn parse_ocean_loading_blq_blocks(
    py: Python<'_>,
    text: &str,
) -> PyResult<Vec<PyOceanLoadingBlqBlock>> {
    core_parse_ocean_loading_blq_blocks(text)
        .map(|blocks| {
            blocks
                .into_iter()
                .map(|inner| PyOceanLoadingBlqBlock { inner })
                .collect()
        })
        .map_err(|err| to_blq_err(py, err))
}

/// Write station blocks as one BLQ file that parses back to equal blocks. A
/// column-order header retained on one block stays in force for the blocks
/// after it, as it does for the parser; a comment after the rows of any block
/// but the last is refused. A refusal raises `BlqError`.
#[pyfunction]
fn write_ocean_loading_blq_blocks(
    py: Python<'_>,
    blocks: Vec<PyOceanLoadingBlqBlock>,
) -> PyResult<String> {
    let blocks: Vec<OceanLoadingBlqBlock> = blocks.into_iter().map(|block| block.inner).collect();
    core_write_ocean_loading_blq_blocks(&blocks).map_err(|err| to_blq_err(py, err))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solid_earth_tide, m)?)?;
    m.add_class::<PyStationTideConstants>()?;
    m.add_class::<PyOceanTideConstituent>()?;
    m.add_class::<PyTideInputErrorKind>()?;
    m.add_class::<PyStationPolarMotion>()?;
    m.add_class::<PyStationDisplacementPosition>()?;
    m.add_class::<PyStationDisplacementEpoch>()?;
    m.add_class::<PyStationDisplacementOptions>()?;
    m.add_class::<PyStationDisplacement>()?;
    m.add_class::<PyStationDisplacementBatchRow>()?;
    m.add_function(wrap_pyfunction!(station_tide_displacement, m)?)?;
    m.add_function(wrap_pyfunction!(station_displacement_ecef_m, m)?)?;
    m.add_function(wrap_pyfunction!(station_displacement_ecef_m_at_epoch, m)?)?;
    m.add_function(wrap_pyfunction!(station_displacement_ecef_m_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solid_earth_pole_tide, m)?)?;
    m.add_function(wrap_pyfunction!(ocean_tide_loading, m)?)?;
    m.add_function(wrap_pyfunction!(ocean_loading_constituents, m)?)?;
    m.add_class::<PyBlqErrorDetail>()?;
    m.add_class::<PyOceanLoadingBlqComment>()?;
    m.add_class::<PyOceanLoadingBlqBlock>()?;
    m.add_function(wrap_pyfunction!(parse_ocean_loading_blq_block, m)?)?;
    m.add_function(wrap_pyfunction!(parse_ocean_loading_blq_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(write_ocean_loading_blq_blocks, m)?)?;
    Ok(())
}
