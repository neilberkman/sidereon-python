//! Python (PyO3) bindings over the `sidereon` ergonomic engine surface.
//!
//! This crate is a thin INTERFACE: it normalizes Python input, marshals it into
//! the `sidereon` / `sidereon-core` types, calls the reference solve, and
//! packages the result as a Pythonic object. It contains ZERO modeling logic of
//! its own; the numbers it returns are exactly what `sidereon-core` produces.
//!
//! The compiled module is imported as `sidereon._sidereon`; the human-facing
//! surface (keyword arguments, numpy arrays, dataclass-like repr) lives in
//! `python/sidereon/__init__.py`, which wraps the symbols defined here.

// Python enum members are UPPER_CASE by idiom (e.g. MoonPhase.NEW, SsrKind.URA);
// the Rust-centric acronym-casing lint does not apply to this binding surface.
#![allow(clippy::upper_case_acronyms)]

use numpy::PyArray1;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::GILOnceCell;
use pyo3::types::{PyAny, PyDict, PyModule, PyTuple, PyType};

mod almanac;
mod angles;
mod anomaly;
mod antex;
mod araim;
mod atmosphere;
mod bias;
mod bodies;
mod body_observe;
mod broadcast_comparison;
mod cdm;
mod clock_stability;
mod conjunction;
mod constellation;
mod core_error_detail;
mod covariance;
mod coverage;
mod data_catalog;
mod defaults;
mod dgnss;
mod doppler;
mod elements;
mod emission;
mod ephemeris;
mod equinoctial;
mod error_metrics;
mod estimation;
mod events;
mod exact_cache;
mod exact_time;
mod fallback;
mod forces;
mod format_diagnostics;
mod frame_catalog;
mod frames;
mod fusion;
mod geodesic;
mod geodetic_time_series;
mod geofence;
mod geoid;
mod geometry;
mod geometry_quality;
mod ils;
mod iod;
mod ionex;
mod lambert;
mod leap;
mod least_squares;
mod lnav;
mod marshal;
mod nmea;
mod normality;
mod ntrip;
mod observables;
mod observation;
mod oem;
mod omm;
mod opm;
mod orbit_determination;
mod ppp;
mod ppp_corrections;
mod products;
mod propagation;
mod qc;
mod reduced_orbit;
mod relative;
mod reliability;
mod rf;
mod rinex;
mod rinex_clock;
mod rtcm;
mod spp_error_detail;
pub(crate) use fallback::selection_error_detail;
pub(crate) use rtcm::to_rtcm_encode_err;
pub(crate) use spp_error_detail::spp_detail;
mod rtk;
mod sbas_pl;
mod sbas_ssr;
mod scenario;
mod sidereal;
mod signal_analysis;
mod sky;
mod solve_error_detail;
mod source_localization;
mod space_weather;
mod spk;
mod spp;
mod staleness;
mod static_positioning;
mod tca;
mod tdm;
mod terrain;
mod terrain_store;
mod tides;
mod time_model_error;
mod tle_fit_error_detail;
mod tropo;
mod unix_compress;

pub(crate) use ephemeris::{PyPreciseEphemerisSamples, PySp3};

create_exception!(
    _sidereon,
    SidereonError,
    PyException,
    "Base class for every Sidereon domain failure. Catch this to handle any\nparse or solve error from the engine."
);

create_exception!(
    _sidereon,
    ScenarioError,
    PyValueError,
    "Raised when core scenario validation, source identity, media, or simulation fails. `detail` retains the exact core error variant and payload."
);

create_exception!(
    _sidereon,
    ParseError,
    SidereonError,
    "Base class for input-format parse failures (SP3, TLE, ...)."
);

create_exception!(
    _sidereon,
    Sp3ParseError,
    ParseError,
    "Raised when an SP3 precise-ephemeris product fails to parse."
);

/// The class for SP3 serialization refusals, created once per interpreter.
///
/// `create_exception!` takes a single base, and this class has two: it is an
/// [`Sp3ParseError`], like the other format write errors, and a `ValueError`,
/// because the refusal is about the product's values. `type(name, bases,
/// namespace)` is what `PyErr_NewException` calls for a tuple of bases.
static SP3_WRITE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const SP3_WRITE_ERROR_DOC: &str = "Raised when Sp3.to_sp3_string cannot state a product in canonical SP3 text: a value finer or wider than its fixed columns, a value that would read back as one of the format's absence sentinels, text that would not survive the reader's trim, an epoch no record restates exactly, or a header that disagrees with the records. Subclasses Sp3ParseError and ValueError. detail carries the typed Sp3WriteErrorDetail and is None on a hand-built instance.";

/// The `Sp3WriteError` class, a subclass of both [`Sp3ParseError`] and
/// `ValueError`.
pub(crate) fn sp3_write_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    SP3_WRITE_ERROR
        .get_or_try_init(py, || -> PyResult<Py<PyType>> {
            let bases = PyTuple::new(
                py,
                [
                    py.get_type::<Sp3ParseError>(),
                    py.get_type::<PyValueError>(),
                ],
            )?;
            let namespace = PyDict::new(py);
            namespace.set_item("__module__", "_sidereon")?;
            namespace.set_item("__doc__", SP3_WRITE_ERROR_DOC)?;
            namespace.set_item("detail", py.None())?;
            let created = py
                .get_type::<PyType>()
                .call1(("Sp3WriteError", bases, namespace))?;
            Ok(created.downcast_into::<PyType>()?.unbind())
        })
        .map(|ty| ty.bind(py).clone())
}

create_exception!(
    _sidereon,
    AntexParseError,
    ParseError,
    "Raised when an ANTEX antenna product fails to parse. A refusal from the core parser sets detail to the typed AntexErrorDetail; an unreadable source, bytes that are not UTF-8 and a hand-built instance have none. AntexWriteError, a subclass, also sets field and reason from a core write refusal; both are None otherwise."
);

create_exception!(
    _sidereon,
    TleParseError,
    ParseError,
    "Raised when a two-line element set fails to parse or initialize SGP4."
);

create_exception!(
    _sidereon,
    GeodesicError,
    PyValueError,
    "Raised when a WGS84 geodesic direct or inverse input is outside its accepted domain."
);

create_exception!(
    _sidereon,
    ProjVgridshiftError,
    PyValueError,
    "Base class for invalid PROJ vertical-grid lookup coordinates."
);

create_exception!(
    _sidereon,
    ProjVgridshiftNonFiniteCoordinateError,
    ProjVgridshiftError,
    "Raised when a PROJ vertical-grid latitude or longitude is not finite."
);

create_exception!(
    _sidereon,
    ProjVgridshiftCoordinateOutsideGridError,
    ProjVgridshiftError,
    "Raised when a PROJ vertical-grid coordinate is outside the grid extent."
);

create_exception!(
    _sidereon,
    GeofenceError,
    SidereonError,
    "Raised when geofence construction, containment, probability, or crossing evaluation fails."
);

create_exception!(
    _sidereon,
    SolveError,
    SidereonError,
    "Raised when a solve or propagation fails: non-convergence, an SGP4 error\ncode, or an integration failure."
);

create_exception!(
    _sidereon,
    PrimitiveError,
    PyValueError,
    "Raised when an estimation or detection primitive rejects its scalar inputs."
);

create_exception!(
    _sidereon,
    SourceLocalizationError,
    PyValueError,
    "Raised when source-localization inputs or geometry cannot produce a solution."
);

create_exception!(
    _sidereon,
    CdmParseError,
    ParseError,
    "Raised when a CCSDS CDM KVN or XML message fails to parse."
);

create_exception!(
    _sidereon,
    TdmParseError,
    ParseError,
    "Raised when a CCSDS TDM KVN message fails to parse or validate. Manually constructed instances can lack core context (detail is None), while all domain failures populate a full typed payload."
);

create_exception!(
    _sidereon,
    TdmWriteError,
    TdmParseError,
    "Raised when a CCSDS TDM KVN message fails to encode. Manually constructed instances can lack core context (detail is None), while all domain failures populate a full typed payload."
);

create_exception!(
    _sidereon,
    TdmValidationError,
    TdmParseError,
    "Raised when CCSDS TDM metadata fields, comments, or structure fail validation. Manually constructed instances can lack core context (detail is None), while all domain failures populate a full typed payload."
);

create_exception!(
    _sidereon,
    OmmParseError,
    ParseError,
    "Raised when a CCSDS OMM KVN, XML, or JSON message fails to parse."
);

create_exception!(
    _sidereon,
    OemParseError,
    ParseError,
    "Raised when a CCSDS OEM KVN or XML message fails to parse."
);

create_exception!(
    _sidereon,
    OpmParseError,
    ParseError,
    "Raised when a CCSDS OPM KVN or XML message fails to parse."
);

create_exception!(
    _sidereon,
    RinexNavParseError,
    ParseError,
    "Raised when a RINEX navigation file fails to parse."
);

create_exception!(
    _sidereon,
    RinexObsParseError,
    ParseError,
    "Raised when a RINEX observation file fails to parse."
);

create_exception!(
    _sidereon,
    RinexObsWriteError,
    RinexObsParseError,
    "Raised when a RINEX observation file or scenario synthetic observations fail to write or downgrade. Manually constructed instances can lack core context (detail is None), while domain failures populate a typed RinexObsWriteErrorDetail."
);

create_exception!(
    _sidereon,
    RinexClockParseError,
    ParseError,
    "Raised when a RINEX clock file fails strict parsing. A refusal from the core parser populates a typed RinexClockErrorDetail; a hand-built instance, an unreadable source, and text that is not UTF-8 have no core context (detail is None)."
);

/// Whether an exception class carries a typed `detail` payload.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Detail {
    /// The class has a `detail` attribute defaulting to None.
    Typed,
    /// The core refusal behind the class carries only its message.
    MessageOnly,
}

/// Create, once per interpreter, an exception class with several bases.
///
/// `create_exception!` takes a single base; `type(name, bases, namespace)` is
/// what `PyErr_NewException` calls for a tuple of bases. A class with a typed
/// payload carries a `detail` attribute that defaults to None, so a hand-built
/// instance answers `e.detail` with None.
fn exception_type_with_bases<'py>(
    py: Python<'py>,
    cell: &'static GILOnceCell<Py<PyType>>,
    name: &'static str,
    doc: &'static str,
    bases: &[Bound<'py, PyType>],
    detail: Detail,
) -> PyResult<Bound<'py, PyType>> {
    cell.get_or_try_init(py, || -> PyResult<Py<PyType>> {
        let bases = PyTuple::new(py, bases)?;
        let namespace = PyDict::new(py);
        namespace.set_item("__module__", "_sidereon")?;
        namespace.set_item("__doc__", doc)?;
        if detail == Detail::Typed {
            namespace.set_item("detail", py.None())?;
        }
        let created = py.get_type::<PyType>().call1((name, bases, namespace))?;
        Ok(created.downcast_into::<PyType>()?.unbind())
    })
    .map(|ty| ty.bind(py).clone())
}

static ANTEX_WRITE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static ANTEX_QUERY_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static IONEX_WRITE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const ANTEX_WRITE_ERROR_DOC: &str = "Raised when Antex.to_antex_string refuses a product it cannot write without loss: a value its fixed columns cannot state exactly, a validity second no form with a decimal point fits, a frequency label that is not a system flag and a two-column number, or public fields that disagree with the retained blocks. Subclasses AntexParseError, which the writer raised before, and ValueError. detail carries the typed AntexErrorDetail, and field and reason the core's; all three are None on a hand-built instance.";

const ANTEX_QUERY_ERROR_DOC: &str = "Raised when the core refuses an ANTEX frequency, PCO or PCV lookup: an unknown label, a label whose sections differ, a zenith outside the grid, or an empty grid. Subclasses SidereonError and ValueError, which the lookups raised before. detail carries the typed AntexErrorDetail and is None on a hand-built instance.";

const IONEX_WRITE_ERROR_DOC: &str = "Raised when the IONEX writer cannot state a product without loss. Subclasses IonexParseError and ValueError. detail contains the core writer refusal text, and is None on a hand-built instance.";

/// The `AntexWriteError` class, a subclass of [`AntexParseError`] and
/// `ValueError`.
pub(crate) fn antex_write_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &ANTEX_WRITE_ERROR,
        "AntexWriteError",
        ANTEX_WRITE_ERROR_DOC,
        &[
            py.get_type::<AntexParseError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

/// The `AntexQueryError` class, a subclass of [`SidereonError`] and
/// `ValueError`.
pub(crate) fn antex_query_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &ANTEX_QUERY_ERROR,
        "AntexQueryError",
        ANTEX_QUERY_ERROR_DOC,
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

pub(crate) fn ionex_write_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &IONEX_WRITE_ERROR,
        "IonexWriteError",
        IONEX_WRITE_ERROR_DOC,
        &[
            py.get_type::<IonexParseError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

static TERRAIN_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const TERRAIN_ERROR_DOC: &str = "Raised when the core refuses a terrain lookup or a DTED tile: a lookup that weights a DTED null posting (an unknown elevation), a tile on a horizontal datum other than WGS84, a missing store tile, or tile metadata the reader cannot place postings by. Subclasses SidereonError and ValueError, which these calls raised before. detail carries the typed TerrainErrorDetail; point_index is the row of a batch lookup the refusal happened at, None otherwise. Both are None on a hand-built instance.";

/// The `TerrainError` class, a subclass of [`SidereonError`] and
/// `ValueError`.
pub(crate) fn terrain_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &TERRAIN_ERROR,
        "TerrainError",
        TERRAIN_ERROR_DOC,
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

static BLQ_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const BLQ_ERROR_DOC: &str = "Raised when the core refuses to read or write an ocean-loading BLQ block: text that is not a BLQ block, a column-order header naming a constituent it does not support, or a station, comment or coefficient the parser would not read back unchanged. Subclasses SidereonError and ValueError. detail carries the typed BlqErrorDetail and is None on a hand-built instance.";

/// The `BlqError` class, a subclass of [`SidereonError`] and `ValueError`.
pub(crate) fn blq_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &BLQ_ERROR,
        "BlqError",
        BLQ_ERROR_DOC,
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

static INERTIAL_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

pub(crate) fn inertial_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    let error_type = exception_type_with_bases(
        py,
        &INERTIAL_ERROR,
        "InertialError",
        "Raised when an inertial state, sample, model, or mechanization input is refused. Subclasses SidereonError and ValueError. kind, field, reason, and details preserve the core error variant.",
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )?;
    error_type.setattr("kind", py.None())?;
    error_type.setattr("field", py.None())?;
    error_type.setattr("reason", py.None())?;
    error_type.setattr("details", py.None())?;
    Ok(error_type)
}

static TIDE_EVALUATION_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

pub(crate) fn tide_evaluation_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    let error_type = exception_type_with_bases(
        py,
        &TIDE_EVALUATION_ERROR,
        "TideEvaluationError",
        "Raised when a station-tide evaluation or tide input is refused. Subclasses SidereonError and ValueError. kind and details preserve the core TideError variant.",
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )?;
    error_type.setattr("kind", py.None())?;
    error_type.setattr("details", py.None())?;
    Ok(error_type)
}

static PPP_CORRECTIONS_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

pub(crate) fn ppp_corrections_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    let error_type = exception_type_with_bases(
        py,
        &PPP_CORRECTIONS_ERROR,
        "PppCorrectionsError",
        "Raised when the core refuses PPP correction precomputation. kind, epoch_index, and details preserve the typed core refusal.",
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )?;
    error_type.setattr("kind", py.None())?;
    error_type.setattr("epoch_index", py.None())?;
    error_type.setattr("details", py.None())?;
    Ok(error_type)
}

static RTCM_ENCODE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const RTCM_ENCODE_ERROR_DOC: &str = "Raised when the core refuses to encode an RTCM or SBAS message without loss. Subclasses RtcmParseError and ValueError. kind names the typed core refusal (SBAS kinds start with sbas_), and details contains every payload field, including nested record, MSM, and departure data. Both are None on a hand-built instance.";

/// The `RtcmEncodeError` class, a subclass of [`RtcmParseError`] and
/// `ValueError`.
pub(crate) fn rtcm_encode_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    let error_type = exception_type_with_bases(
        py,
        &RTCM_ENCODE_ERROR,
        "RtcmEncodeError",
        RTCM_ENCODE_ERROR_DOC,
        &[
            py.get_type::<RtcmParseError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )?;
    error_type.setattr("kind", py.None())?;
    error_type.setattr("details", py.None())?;
    Ok(error_type)
}

static RINEX_NAV_WRITE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const RINEX_NAV_WRITE_ERROR_DOC: &str = "Raised when broadcast navigation records cannot be written as one RINEX navigation file: a set holding both a CNAV-family record, which only RINEX 4 holds, and an unclassified Galileo record, which only RINEX 3 holds. Subclasses RinexNavParseError and ValueError. The core refusal carries its line and reason in the message.";

/// The `RinexNavWriteError` class, a subclass of [`RinexNavParseError`] and
/// `ValueError`.
pub(crate) fn rinex_nav_write_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &RINEX_NAV_WRITE_ERROR,
        "RinexNavWriteError",
        RINEX_NAV_WRITE_ERROR_DOC,
        &[
            py.get_type::<RinexNavParseError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::MessageOnly,
    )
}

/// Map a core RINEX NAV write refusal into `RinexNavWriteError`.
pub(crate) fn to_rinex_nav_write_err<E: std::fmt::Display>(py: Python<'_>, err: E) -> PyErr {
    match rinex_nav_write_error_type(py) {
        Ok(ty) => PyErr::from_type(ty, err.to_string()),
        Err(e) => e,
    }
}

static UT1_OUTSIDE_COVERAGE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const UT1_OUTSIDE_COVERAGE_ERROR_DOC: &str = "Raised when an instant lies outside the UT1 table and the call reads UT1 under the default strict UT1 policy: a solve or an event search whose ephemeris source or frame transform needs UT1 there. It was previously an unrelated failure, a dropped satellite or a skipped base satellite. Subclasses SidereonError and ValueError. reason is 'before_coverage' or 'after_coverage', and None on a hand-built instance.";

/// The `Ut1OutsideCoverageError` class, a subclass of [`SidereonError`] and
/// `ValueError`.
pub(crate) fn ut1_outside_coverage_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &UT1_OUTSIDE_COVERAGE_ERROR,
        "Ut1OutsideCoverageError",
        UT1_OUTSIDE_COVERAGE_ERROR_DOC,
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::MessageOnly,
    )
}

static SELECTION_UNSETTLED_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const SELECTION_UNSETTLED_ERROR_DOC: &str = "Raised when an SPP or static solve's satellite selection does not settle: after 10 passes, each trust-region solve over a new selection and each least-squares step counting as one, no step at a selection that held fell below 1e-4 m. RTKLIB estpos fails the epoch the same way after MAXITR iterations. Subclasses SolveError. passes is the number of passes run, and None on a hand-built instance.";

/// The `SelectionUnsettledError` class, a subclass of `SolveError`.
pub(crate) fn selection_unsettled_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &SELECTION_UNSETTLED_ERROR,
        "SelectionUnsettledError",
        SELECTION_UNSETTLED_ERROR_DOC,
        &[py.get_type::<SolveError>()],
        Detail::MessageOnly,
    )
}

static QUALITY_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const QUALITY_ERROR_DOC: &str = "Raised when the core refuses a quality-control or RAIM input. Subclasses SidereonError and ValueError. `kind` is the stable core variant name; a hand-built instance has kind None.";

/// The `QualityError` class preserves the core variant while remaining a
/// `ValueError` for callers that already catch the old binding error.
pub(crate) fn quality_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &QUALITY_ERROR,
        "QualityError",
        QUALITY_ERROR_DOC,
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::MessageOnly,
    )
}

/// Preserve the core quality variant and message while keeping compatibility
/// with callers that catch `ValueError`.
pub(crate) fn quality_err(err: sidereon_core::quality::QualityError) -> PyErr {
    let message = err.to_string();
    let kind = match err {
        sidereon_core::quality::QualityError::InvalidElevation => "invalid_elevation",
        sidereon_core::quality::QualityError::MissingCn0 => "missing_cn0",
        sidereon_core::quality::QualityError::InvalidParameter => "invalid_parameter",
        sidereon_core::quality::QualityError::InvalidReliabilityParameter => {
            "invalid_reliability_parameter"
        }
        sidereon_core::quality::QualityError::InvalidProbability => "invalid_probability",
        sidereon_core::quality::QualityError::InvalidSystemCount => "invalid_system_count",
        sidereon_core::quality::QualityError::InvalidDof => "invalid_dof",
        sidereon_core::quality::QualityError::InvalidWeight => "invalid_weight",
        sidereon_core::quality::QualityError::InvalidResiduals => "invalid_residuals",
        sidereon_core::quality::QualityError::InvalidDesign => "invalid_design",
        sidereon_core::quality::QualityError::SingularGeometry => "singular_geometry",
        sidereon_core::quality::QualityError::MissingVariances => "missing_variances",
        sidereon_core::quality::QualityError::InvalidVariance => "invalid_variance",
    };
    Python::with_gil(|py| {
        let ty = match quality_error_type(py) {
            Ok(ty) => ty,
            Err(error) => return error,
        };
        let error = PyErr::from_type(ty, message);
        if let Err(error) = error.value(py).setattr("kind", kind) {
            return error;
        }
        error
    })
}

static FDE_FAULT_UNRESOLVED_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const FDE_FAULT_UNRESOLVED_ERROR_DOC: &str = "Raised when fault detection and exclusion stops with a fault still detected: the exclusion budget was spent (reason 'exclusion_budget_exhausted') or no leave-one-out re-solve was admissible, every candidate failing to solve, using fewer than five satellites or leaving a residual RMS above the cap (reason 'no_admissible_exclusion'). Subclasses SolveError, which FDE raised before. solution is the last SppSolution, which still fails the test; excluded the satellites excluded to reach it, in exclusion order; raim its RaimResult, with fault_detected True. All four are None on a hand-built instance.";

/// The `FdeFaultUnresolvedError` class, a subclass of `SolveError`.
pub(crate) fn fde_fault_unresolved_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &FDE_FAULT_UNRESOLVED_ERROR,
        "FdeFaultUnresolvedError",
        FDE_FAULT_UNRESOLVED_ERROR_DOC,
        &[py.get_type::<SolveError>()],
        Detail::MessageOnly,
    )
}

/// Map an SPP refusal: `SelectionUnsettledError` for a selection that did not
/// settle, `SolveError` for the rest, each with the core message.
pub(crate) fn spp_solve_err(err: sidereon_core::positioning::SppError) -> PyErr {
    Python::with_gil(|py| spp_error_detail::spp_error(py, err, ""))
}

/// Map a static-solve refusal as [`spp_solve_err`] maps an SPP one.
pub(crate) fn static_solve_err(err: sidereon_core::static_positioning::StaticSolveError) -> PyErr {
    Python::with_gil(|py| spp_error_detail::static_error(py, err))
}

/// How a call that reads UT1 treats an instant outside the UT1 table,
/// mirroring the core `ValidityMode`.
#[pyclass(module = "sidereon._sidereon", name = "ValidityMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyValidityMode {
    /// Refuse an instant outside the UT1 table with `Ut1OutsideCoverageError`.
    STRICT,
    /// Use the long-term delta-T curve outside the table and report the
    /// departure.
    PERMISSIVE,
}

impl From<PyValidityMode> for sidereon_core::astro::time::ValidityMode {
    fn from(mode: PyValidityMode) -> Self {
        match mode {
            PyValidityMode::STRICT => sidereon_core::astro::time::ValidityMode::Strict,
            PyValidityMode::PERMISSIVE => sidereon_core::astro::time::ValidityMode::Permissive,
        }
    }
}

/// The label of a core UT1 degrade reason.
pub(crate) fn degrade_reason_label(
    reason: sidereon_core::astro::time::DegradeReason,
) -> &'static str {
    match reason {
        sidereon_core::astro::time::DegradeReason::BeforeCoverage => "before_coverage",
        sidereon_core::astro::time::DegradeReason::AfterCoverage => "after_coverage",
    }
}

/// Map a core UT1 refusal into `Ut1OutsideCoverageError`, carrying the core
/// message and the reason label, prefixed with `context` when given.
pub(crate) fn ut1_outside_coverage_err(
    context: &str,
    reason: sidereon_core::astro::time::DegradeReason,
) -> PyErr {
    Python::with_gil(|py| {
        let message = if context.is_empty() {
            format!("UT1 outside coverage: {reason}")
        } else {
            format!("{context}: UT1 outside coverage: {reason}")
        };
        let ty = match ut1_outside_coverage_error_type(py) {
            Ok(ty) => ty,
            Err(e) => return e,
        };
        let err = PyErr::from_type(ty, message);
        if let Err(e) = err
            .value(py)
            .setattr("reason", degrade_reason_label(reason))
        {
            return e;
        }
        err
    })
}

static RINEX_CLOCK_QUERY_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static RINEX_CLOCK_WRITE_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();
static RINEX_CLOCK_EDIT_ERROR: GILOnceCell<Py<PyType>> = GILOnceCell::new();

const RINEX_CLOCK_QUERY_ERROR_DOC: &str = "Raised when the core refuses a RINEX clock query: civil fields that name no epoch in the product's time scale, a product whose time system resolves to no time scale, or GPS seconds that are not finite or lie outside the civil years 1 through 9999. Subclasses RinexClockParseError, which RinexClock.clock_s raised before, and ValueError, which the instant and GPS-seconds queries raised before. detail carries the typed RinexClockErrorDetail and is None on a hand-built instance.";

const RINEX_CLOCK_WRITE_ERROR_DOC: &str = "Raised when a RINEX clock product cannot be written: a value no 19-column field states exactly, an epoch no seconds field states exactly under the write policy, or a time scale no RINEX clock time system names. Subclasses SidereonError and ValueError. detail carries the typed RinexClockErrorDetail and is None on a hand-built instance.";

const RINEX_CLOCK_EDIT_ERROR_DOC: &str = "Raised when the core refuses a RINEX clock record, clock point, product construction or edit. A refused edit changes nothing. Subclasses SidereonError and ValueError. detail carries the typed RinexClockErrorDetail and is None on a hand-built instance.";

/// The `RinexClockQueryError` class, a subclass of [`RinexClockParseError`]
/// and `ValueError`.
pub(crate) fn rinex_clock_query_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &RINEX_CLOCK_QUERY_ERROR,
        "RinexClockQueryError",
        RINEX_CLOCK_QUERY_ERROR_DOC,
        &[
            py.get_type::<RinexClockParseError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

/// The `RinexClockWriteError` class, a subclass of [`SidereonError`] and
/// `ValueError`.
pub(crate) fn rinex_clock_write_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &RINEX_CLOCK_WRITE_ERROR,
        "RinexClockWriteError",
        RINEX_CLOCK_WRITE_ERROR_DOC,
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

/// The `RinexClockEditError` class, a subclass of [`SidereonError`] and
/// `ValueError`.
pub(crate) fn rinex_clock_edit_error_type(py: Python<'_>) -> PyResult<Bound<'_, PyType>> {
    exception_type_with_bases(
        py,
        &RINEX_CLOCK_EDIT_ERROR,
        "RinexClockEditError",
        RINEX_CLOCK_EDIT_ERROR_DOC,
        &[
            py.get_type::<SidereonError>(),
            py.get_type::<PyValueError>(),
        ],
        Detail::Typed,
    )
}

create_exception!(
    _sidereon,
    CrinexParseError,
    ParseError,
    "Raised when a Compact RINEX observation file fails to decode."
);

create_exception!(
    _sidereon,
    IonexParseError,
    ParseError,
    "Raised when an IONEX ionosphere-map product fails to parse."
);

create_exception!(
    _sidereon,
    SpkParseError,
    ParseError,
    "Raised when a JPL/NAIF SPK (DAF .bsp) ephemeris kernel fails to parse."
);

create_exception!(
    _sidereon,
    PreciseInterpolantArtifactError,
    ParseError,
    "Raised when a precise-interpolant artifact cannot be opened."
);

create_exception!(
    _sidereon,
    PreciseInterpolantArtifactCorruptError,
    PreciseInterpolantArtifactError,
    "Raised when a precise-interpolant artifact checksum indicates corrupt bytes."
);

create_exception!(
    _sidereon,
    PreciseInterpolantArtifactTruncatedError,
    PreciseInterpolantArtifactError,
    "Raised when a precise-interpolant artifact is shorter than its declared layout."
);

create_exception!(
    _sidereon,
    PreciseSamplesError,
    PyValueError,
    "Raised when precise-ephemeris samples or accuracy sidecars fail core validation. kind and satellite preserve the typed core refusal; satellite is None when the variant has no satellite payload."
);

create_exception!(
    _sidereon,
    AccuracySamplesMismatchError,
    PreciseSamplesError,
    "Raised when precise-ephemeris accuracy sidecars do not match their samples."
);

create_exception!(
    _sidereon,
    InvalidAccuracyValueError,
    PreciseSamplesError,
    "Raised when a precise-ephemeris accuracy sidecar contains an invalid value."
);

create_exception!(
    _sidereon,
    RtcmParseError,
    ParseError,
    "Raised when an RTCM 3 message body cannot be decoded or framed."
);

create_exception!(
    _sidereon,
    RtcmConversionError,
    RtcmParseError,
    "Raised when an RTCM message cannot be converted to a satellite record or its VTEC model cannot be evaluated. Core conversion refusals expose a stable kind string and a details dictionary of variant fields."
);

create_exception!(
    _sidereon,
    SpaceWeatherError,
    SidereonError,
    "Raised when a space-weather product cannot be parsed or queried."
);

create_exception!(
    _sidereon,
    ConstellationError,
    SidereonError,
    "Raised when the GNSS constellation catalog cannot be built or validated:\nan object name without a PRN, malformed NAVCEN status bytes, or an SP3\nvalidation finding."
);

create_exception!(
    _sidereon,
    SelectionError,
    SidereonError,
    "Raised when product-staleness selection cannot satisfy a request: an empty\nproduct set, no product at or before the epoch, the nearest product beyond\nthe staleness cap, or an invalid range/policy/product."
);

create_exception!(
    _sidereon,
    FallbackError,
    SidereonError,
    "Raised when a precise-with-broadcast fallback solve fails: either the\nselected precise product's solve failed (a genuine error, not masked by a\nsilent broadcast re-solve) or the broadcast fallback solve failed."
);

/// Build a `numpy.ndarray` of dtype float64 from a slice, so positions surface
/// to Python as numpy arrays rather than Rust through a keyhole.
pub(crate) fn np_array<'py>(py: Python<'py>, values: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, values)
}

/// Parse a Python checksum claim without allowing extraction failures to leak
/// implementation-specific `TypeError` or `OverflowError` exceptions.
pub(crate) fn parse_claimed_checksum64(value: &Bound<'_, PyAny>) -> PyResult<u64> {
    value.extract::<u64>().map_err(|_| {
        PyValueError::new_err("claimed_checksum64 must be an integer in range 0 <= value < 2**64")
    })
}

/// Map an SP3 parse failure into [`Sp3ParseError`], preserving the engine
/// message.
pub(crate) fn to_sp3_err<E: std::fmt::Display>(err: E) -> PyErr {
    Sp3ParseError::new_err(err.to_string())
}

/// Map an ANTEX parse failure into [`AntexParseError`], preserving the engine
/// message.
pub(crate) fn to_antex_err<E: std::fmt::Display>(err: E) -> PyErr {
    AntexParseError::new_err(err.to_string())
}

/// Map a TLE parse / SGP4-init failure into [`TleParseError`], preserving the
/// engine message.
pub(crate) fn to_tle_err<E: std::fmt::Display>(err: E) -> PyErr {
    TleParseError::new_err(err.to_string())
}

/// Map a solve / propagation failure into [`SolveError`], preserving the engine
/// message.
pub(crate) fn to_solve_err<E: std::fmt::Display>(err: E) -> PyErr {
    SolveError::new_err(err.to_string())
}

#[pymodule]
fn _sidereon(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SidereonError", py.get_type::<SidereonError>())?;
    m.add("ScenarioError", py.get_type::<ScenarioError>())?;
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add("Sp3ParseError", py.get_type::<Sp3ParseError>())?;
    m.add("Sp3WriteError", sp3_write_error_type(py)?)?;
    let antex_parse_error = py.get_type::<AntexParseError>();
    antex_parse_error.setattr("field", py.None())?;
    antex_parse_error.setattr("reason", py.None())?;
    antex_parse_error.setattr("detail", py.None())?;
    m.add("AntexParseError", &antex_parse_error)?;
    m.add("AntexWriteError", antex_write_error_type(py)?)?;
    m.add("AntexQueryError", antex_query_error_type(py)?)?;
    m.add("TleParseError", py.get_type::<TleParseError>())?;
    m.add("GeodesicError", py.get_type::<GeodesicError>())?;
    m.add("ProjVgridshiftError", py.get_type::<ProjVgridshiftError>())?;
    m.add(
        "ProjVgridshiftNonFiniteCoordinateError",
        py.get_type::<ProjVgridshiftNonFiniteCoordinateError>(),
    )?;
    m.add(
        "ProjVgridshiftCoordinateOutsideGridError",
        py.get_type::<ProjVgridshiftCoordinateOutsideGridError>(),
    )?;
    m.add("GeofenceError", py.get_type::<GeofenceError>())?;
    let solve_error = py.get_type::<SolveError>();
    solve_error.setattr("detail", py.None())?;
    m.add("SolveError", &solve_error)?;
    m.add("PrimitiveError", py.get_type::<PrimitiveError>())?;
    m.add(
        "SourceLocalizationError",
        py.get_type::<SourceLocalizationError>(),
    )?;
    m.add("CdmParseError", py.get_type::<CdmParseError>())?;
    m.add("OmmParseError", py.get_type::<OmmParseError>())?;
    m.add("OemParseError", py.get_type::<OemParseError>())?;
    m.add("OpmParseError", py.get_type::<OpmParseError>())?;
    m.add("RinexNavParseError", py.get_type::<RinexNavParseError>())?;
    m.add("RinexNavWriteError", rinex_nav_write_error_type(py)?)?;
    m.add("RinexObsParseError", py.get_type::<RinexObsParseError>())?;
    let rinex_obs_write_error = py.get_type::<RinexObsWriteError>();
    rinex_obs_write_error.setattr("detail", py.None())?;
    m.add("RinexObsWriteError", &rinex_obs_write_error)?;
    let rinex_clock_parse_error = py.get_type::<RinexClockParseError>();
    rinex_clock_parse_error.setattr("detail", py.None())?;
    m.add("RinexClockParseError", &rinex_clock_parse_error)?;
    m.add("RinexClockWriteError", rinex_clock_write_error_type(py)?)?;
    m.add("RinexClockQueryError", rinex_clock_query_error_type(py)?)?;
    m.add("RinexClockEditError", rinex_clock_edit_error_type(py)?)?;
    m.add("CrinexParseError", py.get_type::<CrinexParseError>())?;
    m.add("IonexParseError", py.get_type::<IonexParseError>())?;
    m.add("IonexWriteError", ionex_write_error_type(py)?)?;
    m.add("SpkParseError", py.get_type::<SpkParseError>())?;
    m.add(
        "PreciseInterpolantArtifactError",
        py.get_type::<PreciseInterpolantArtifactError>(),
    )?;
    m.add(
        "PreciseInterpolantArtifactCorruptError",
        py.get_type::<PreciseInterpolantArtifactCorruptError>(),
    )?;
    m.add(
        "PreciseInterpolantArtifactTruncatedError",
        py.get_type::<PreciseInterpolantArtifactTruncatedError>(),
    )?;
    let precise_samples_error = py.get_type::<PreciseSamplesError>();
    precise_samples_error.setattr("kind", py.None())?;
    precise_samples_error.setattr("satellite", py.None())?;
    m.add("PreciseSamplesError", &precise_samples_error)?;
    m.add(
        "AccuracySamplesMismatchError",
        py.get_type::<AccuracySamplesMismatchError>(),
    )?;
    let invalid_accuracy_value_error = py.get_type::<InvalidAccuracyValueError>();
    invalid_accuracy_value_error.setattr("satellite", py.None())?;
    m.add("InvalidAccuracyValueError", &invalid_accuracy_value_error)?;
    m.add("RtcmParseError", py.get_type::<RtcmParseError>())?;
    let rtcm_conversion_error = py.get_type::<RtcmConversionError>();
    rtcm_conversion_error.setattr("kind", py.None())?;
    rtcm_conversion_error.setattr("details", py.None())?;
    m.add("RtcmConversionError", &rtcm_conversion_error)?;
    m.add("RtcmEncodeError", rtcm_encode_error_type(py)?)?;
    let terrain_error = terrain_error_type(py)?;
    terrain_error.setattr("point_index", py.None())?;
    m.add("TerrainError", &terrain_error)?;
    m.add("BlqError", blq_error_type(py)?)?;
    m.add("InertialError", inertial_error_type(py)?)?;
    m.add("TideEvaluationError", tide_evaluation_error_type(py)?)?;
    m.add("PppCorrectionsError", ppp_corrections_error_type(py)?)?;
    let ut1_error = ut1_outside_coverage_error_type(py)?;
    ut1_error.setattr("reason", py.None())?;
    m.add("Ut1OutsideCoverageError", &ut1_error)?;
    let selection_error = selection_unsettled_error_type(py)?;
    selection_error.setattr("passes", py.None())?;
    m.add("SelectionUnsettledError", &selection_error)?;
    let quality_error = quality_error_type(py)?;
    quality_error.setattr("kind", py.None())?;
    m.add("QualityError", &quality_error)?;
    let fde_error = fde_fault_unresolved_error_type(py)?;
    for field in ["reason", "solution", "excluded", "raim"] {
        fde_error.setattr(field, py.None())?;
    }
    m.add("FdeFaultUnresolvedError", &fde_error)?;
    m.add_class::<PyValidityMode>()?;
    m.add("SpaceWeatherError", py.get_type::<SpaceWeatherError>())?;
    m.add("ConstellationError", py.get_type::<ConstellationError>())?;
    let selection_error = py.get_type::<SelectionError>();
    selection_error.setattr("detail", py.None())?;
    selection_error.setattr("selection_detail", py.None())?;
    m.add("SelectionError", &selection_error)?;
    let fallback_error = py.get_type::<FallbackError>();
    fallback_error.setattr("detail", py.None())?;
    m.add("FallbackError", &fallback_error)?;
    let tdm_parse_error = py.get_type::<TdmParseError>();
    tdm_parse_error.setattr("detail", py.None())?;
    m.add("TdmParseError", &tdm_parse_error)?;
    m.add("TdmWriteError", py.get_type::<TdmWriteError>())?;
    m.add("TdmValidationError", py.get_type::<TdmValidationError>())?;
    geodesic::register(m)?;
    geofence::register(m)?;
    frame_catalog::register(m)?;
    exact_time::register(m)?;
    ephemeris::register(m)?;
    orbit_determination::register(m)?;
    estimation::register(m)?;
    products::register(m)?;
    antex::register(m)?;
    bodies::register(m)?;
    geometry_quality::register(m)?;
    spp::register(m)?;
    spk::register(m)?;
    rtk::register(m)?;
    ppp::register(m)?;
    propagation::register(m)?;
    frames::register(m)?;
    ionex::register(m)?;
    rf::register(m)?;
    events::register(m)?;
    source_localization::register(m)?;
    conjunction::register(m)?;
    cdm::register(m)?;
    tdm::register(m)?;
    omm::register(m)?;
    oem::register(m)?;
    opm::register(m)?;
    rinex::register(m)?;
    rinex_clock::register(m)?;
    observables::register(m)?;
    forces::register(m)?;
    format_diagnostics::register(m)?;
    tropo::register(m)?;
    dgnss::register(m)?;
    broadcast_comparison::register(m)?;
    ppp_corrections::register(m)?;
    static_positioning::register(m)?;
    emission::register(m)?;
    fusion::register(m)?;
    scenario::register(m)?;
    signal_analysis::register(m)?;
    qc::register(m)?;
    constellation::register(m)?;
    staleness::register(m)?;
    fallback::register(m)?;
    ils::register(m)?;
    lambert::register(m)?;
    least_squares::register(m)?;
    covariance::register(m)?;
    error_metrics::register(m)?;
    normality::register(m)?;
    leap::register(m)?;
    clock_stability::register(m)?;
    sidereal::register(m)?;
    reliability::register(m)?;
    araim::register(m)?;
    sky::register(m)?;
    iod::register(m)?;
    geometry::register(m)?;
    geodetic_time_series::register(m)?;
    reduced_orbit::register(m)?;
    atmosphere::register(m)?;
    lnav::register(m)?;
    coverage::register(m)?;
    tides::register(m)?;
    doppler::register(m)?;
    defaults::register(m)?;
    data_catalog::register(m)?;
    unix_compress::register(m)?;
    exact_cache::register(m)?;
    elements::register(m)?;
    almanac::register(m)?;
    anomaly::register(m)?;
    bias::register(m)?;
    equinoctial::register(m)?;
    angles::register(m)?;
    relative::register(m)?;
    body_observe::register(m)?;
    observation::register(m)?;
    geoid::register(m)?;
    tca::register(m)?;
    terrain::register(m)?;
    terrain_store::register(m)?;
    sbas_pl::register(m)?;
    sbas_ssr::register(m)?;
    rtcm::register(m)?;
    space_weather::register(m)?;
    nmea::register(m)?;
    ntrip::register(m)?;
    Ok(())
}
