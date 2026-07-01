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

use numpy::PyArray1;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod atmosphere;
mod bodies;
mod broadcast_comparison;
mod cdm;
mod conjunction;
mod constellation;
mod covariance;
mod coverage;
mod defaults;
mod dgnss;
mod doppler;
mod elements;
mod ephemeris;
mod events;
mod fallback;
mod forces;
mod frames;
mod geoid;
mod geometry;
mod ils;
mod iod;
mod ionex;
mod lambert;
mod least_squares;
mod leap;
mod lnav;
mod marshal;
mod normality;
mod observables;
mod observation;
mod oem;
mod omm;
mod opm;
mod ppp;
mod ppp_corrections;
mod products;
mod propagation;
mod qc;
mod reduced_orbit;
mod rf;
mod rinex;
mod rtcm;
mod rtk;
mod sky;
mod spk;
mod spp;
mod staleness;
mod tca;
mod tides;
mod tropo;

pub(crate) use ephemeris::{PySp3, PyPreciseEphemerisSamples};

create_exception!(
    _sidereon,
    SidereonError,
    PyException,
    "Base class for every Sidereon domain failure. Catch this to handle any\nparse or solve error from the engine."
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

create_exception!(
    _sidereon,
    AntexParseError,
    ParseError,
    "Raised when an ANTEX antenna product fails to parse."
);

create_exception!(
    _sidereon,
    TleParseError,
    ParseError,
    "Raised when a two-line element set fails to parse or initialize SGP4."
);

create_exception!(
    _sidereon,
    SolveError,
    SidereonError,
    "Raised when a solve or propagation fails: non-convergence, an SGP4 error\ncode, or an integration failure."
);

create_exception!(
    _sidereon,
    CdmParseError,
    ParseError,
    "Raised when a CCSDS CDM KVN or XML message fails to parse."
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
    RinexClockParseError,
    ParseError,
    "Raised when a RINEX clock file fails strict parsing."
);

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
    RtcmParseError,
    ParseError,
    "Raised when an RTCM 3 message body cannot be decoded or framed."
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
    m.add("ParseError", py.get_type::<ParseError>())?;
    m.add("Sp3ParseError", py.get_type::<Sp3ParseError>())?;
    m.add("AntexParseError", py.get_type::<AntexParseError>())?;
    m.add("TleParseError", py.get_type::<TleParseError>())?;
    m.add("SolveError", py.get_type::<SolveError>())?;
    m.add("CdmParseError", py.get_type::<CdmParseError>())?;
    m.add("OmmParseError", py.get_type::<OmmParseError>())?;
    m.add("OemParseError", py.get_type::<OemParseError>())?;
    m.add("OpmParseError", py.get_type::<OpmParseError>())?;
    m.add("RinexNavParseError", py.get_type::<RinexNavParseError>())?;
    m.add("RinexObsParseError", py.get_type::<RinexObsParseError>())?;
    m.add(
        "RinexClockParseError",
        py.get_type::<RinexClockParseError>(),
    )?;
    m.add("CrinexParseError", py.get_type::<CrinexParseError>())?;
    m.add("IonexParseError", py.get_type::<IonexParseError>())?;
    m.add("SpkParseError", py.get_type::<SpkParseError>())?;
    m.add("RtcmParseError", py.get_type::<RtcmParseError>())?;
    m.add("ConstellationError", py.get_type::<ConstellationError>())?;
    m.add("SelectionError", py.get_type::<SelectionError>())?;
    m.add("FallbackError", py.get_type::<FallbackError>())?;
    ephemeris::register(m)?;
    products::register(m)?;
    bodies::register(m)?;
    spp::register(m)?;
    spk::register(m)?;
    rtk::register(m)?;
    ppp::register(m)?;
    propagation::register(m)?;
    frames::register(m)?;
    ionex::register(m)?;
    rf::register(m)?;
    events::register(m)?;
    conjunction::register(m)?;
    cdm::register(m)?;
    omm::register(m)?;
    oem::register(m)?;
    opm::register(m)?;
    rinex::register(m)?;
    observables::register(m)?;
    forces::register(m)?;
    tropo::register(m)?;
    dgnss::register(m)?;
    broadcast_comparison::register(m)?;
    ppp_corrections::register(m)?;
    qc::register(m)?;
    constellation::register(m)?;
    staleness::register(m)?;
    fallback::register(m)?;
    ils::register(m)?;
    lambert::register(m)?;
    least_squares::register(m)?;
    covariance::register(m)?;
    normality::register(m)?;
    leap::register(m)?;
    sky::register(m)?;
    iod::register(m)?;
    geometry::register(m)?;
    reduced_orbit::register(m)?;
    atmosphere::register(m)?;
    lnav::register(m)?;
    coverage::register(m)?;
    tides::register(m)?;
    doppler::register(m)?;
    defaults::register(m)?;
    elements::register(m)?;
    observation::register(m)?;
    geoid::register(m)?;
    tca::register(m)?;
    rtcm::register(m)?;
    Ok(())
}
