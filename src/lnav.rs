//! GPS LNAV (legacy navigation message) codec binding.
//!
//! Thin INTERFACE over `sidereon_core::navigation::lnav`. It marshals subframe
//! bit vectors and engineering-unit parameter maps into the core codec and
//! packages the decoded parameters back as a Pythonic object. Every value it
//! returns is produced by the core encoder/decoder; the binding holds no
//! IS-GPS-200 scaling, parity, or assembly logic of its own.

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyModule};
use pyo3::Bound;

use sidereon_core::navigation::lnav::{
    decode as core_decode, encode as core_encode, parity as core_parity,
    parity_valid as core_parity_valid, subframe_id as core_subframe_id, tow as core_tow,
    LnavDecoded, LnavError, LnavNumber, LnavOptions, LnavParams, PREAMBLE, SUBFRAME_LENGTH,
    WORD_LENGTH,
};

/// The 30 engineering-unit parameter field names, in `LnavParams` declaration
/// order. The binding reads each from the supplied mapping and tags missing
/// keys; the core owns every scaling and range rule.
const PARAM_FIELDS: [&str; 30] = [
    "week_number",
    "l2_code",
    "l2_p_data_flag",
    "ura_index",
    "sv_health",
    "iodc",
    "tgd",
    "toc",
    "af0",
    "af1",
    "af2",
    "iode",
    "crs",
    "delta_n",
    "m0",
    "cuc",
    "eccentricity",
    "cus",
    "sqrt_a",
    "toe",
    "fit_interval_flag",
    "aodo",
    "cic",
    "omega0",
    "cis",
    "i0",
    "crc",
    "omega",
    "omega_dot",
    "idot",
];

/// Map a core LNAV codec failure to a Python `ValueError`, preserving the field
/// and position the engine reported.
fn lnav_err(err: LnavError) -> PyErr {
    let message = match err {
        LnavError::OutOfRange { field, value } => {
            let value = match value {
                LnavNumber::Int(i) => i.to_string(),
                LnavNumber::Float(f) => f.to_string(),
            };
            format!("lnav field {} out of range: {value}", field.name())
        }
        LnavError::ParityFailed { subframe, word } => {
            format!("lnav parity failed at subframe {subframe}, word {word}")
        }
        LnavError::BadWordLength { expected, actual } => {
            format!("lnav parity source word has {actual} bits, expected {expected}")
        }
        LnavError::BadSubframeLength { subframe } => {
            format!("lnav subframe {subframe} is not exactly 300 bits")
        }
    };
    PyValueError::new_err(message)
}

/// Decode one Python integer/float into the core's type-preserving numeric.
fn to_lnav_number(obj: &Bound<'_, PyAny>) -> PyResult<LnavNumber> {
    // Order matters: a Python `bool` is an `int` subclass, so the int branch
    // also accepts it, matching the engine's integer-field expectations.
    if obj.is_instance_of::<PyInt>() {
        Ok(LnavNumber::Int(obj.extract::<i64>()?))
    } else if obj.is_instance_of::<PyFloat>() {
        Ok(LnavNumber::Float(obj.extract::<f64>()?))
    } else {
        Err(PyTypeError::new_err(
            "LNAV parameter values must be int or float",
        ))
    }
}

/// Read one required field from the parameter mapping as a `LnavNumber`.
fn field(params: &Bound<'_, PyDict>, key: &str) -> PyResult<LnavNumber> {
    let value = params
        .get_item(key)?
        .ok_or_else(|| PyKeyError::new_err(format!("missing LNAV parameter {key:?}")))?;
    to_lnav_number(&value)
}

fn params_from_dict(params: &Bound<'_, PyDict>) -> PyResult<LnavParams> {
    Ok(LnavParams {
        week_number: field(params, "week_number")?,
        l2_code: field(params, "l2_code")?,
        l2_p_data_flag: field(params, "l2_p_data_flag")?,
        ura_index: field(params, "ura_index")?,
        sv_health: field(params, "sv_health")?,
        iodc: field(params, "iodc")?,
        tgd: field(params, "tgd")?,
        toc: field(params, "toc")?,
        af0: field(params, "af0")?,
        af1: field(params, "af1")?,
        af2: field(params, "af2")?,
        iode: field(params, "iode")?,
        crs: field(params, "crs")?,
        delta_n: field(params, "delta_n")?,
        m0: field(params, "m0")?,
        cuc: field(params, "cuc")?,
        eccentricity: field(params, "eccentricity")?,
        cus: field(params, "cus")?,
        sqrt_a: field(params, "sqrt_a")?,
        toe: field(params, "toe")?,
        fit_interval_flag: field(params, "fit_interval_flag")?,
        aodo: field(params, "aodo")?,
        cic: field(params, "cic")?,
        omega0: field(params, "omega0")?,
        cis: field(params, "cis")?,
        i0: field(params, "i0")?,
        crc: field(params, "crc")?,
        omega: field(params, "omega")?,
        omega_dot: field(params, "omega_dot")?,
        idot: field(params, "idot")?,
    })
}

/// Decoded LNAV clock and ephemeris parameters (the typed output of `decode`).
///
/// Integer fields are recovered exactly; scaled fields are the transmitted
/// integer times the IS-GPS-200 LSB. `l2_p_data_flag` is an encode-only word-4
/// flag and is not recovered.
#[pyclass(module = "sidereon._sidereon", name = "LnavDecoded")]
#[derive(Clone)]
pub struct PyLnavDecoded {
    inner: LnavDecoded,
}

#[pymethods]
impl PyLnavDecoded {
    /// GPS week number (mod 1024).
    #[getter]
    fn week_number(&self) -> i64 {
        self.inner.week_number
    }
    /// L2 code flag.
    #[getter]
    fn l2_code(&self) -> i64 {
        self.inner.l2_code
    }
    /// SV accuracy (URA) index.
    #[getter]
    fn ura_index(&self) -> i64 {
        self.inner.ura_index
    }
    /// SV health bits.
    #[getter]
    fn sv_health(&self) -> i64 {
        self.inner.sv_health
    }
    /// Issue of data, clock.
    #[getter]
    fn iodc(&self) -> i64 {
        self.inner.iodc
    }
    /// Group delay differential, seconds.
    #[getter]
    fn tgd(&self) -> f64 {
        self.inner.tgd
    }
    /// Clock data reference time, seconds.
    #[getter]
    fn toc(&self) -> i64 {
        self.inner.toc
    }
    /// Clock bias, seconds.
    #[getter]
    fn af0(&self) -> f64 {
        self.inner.af0
    }
    /// Clock drift, seconds/second.
    #[getter]
    fn af1(&self) -> f64 {
        self.inner.af1
    }
    /// Clock drift rate, seconds/second^2.
    #[getter]
    fn af2(&self) -> f64 {
        self.inner.af2
    }
    /// Issue of data, ephemeris.
    #[getter]
    fn iode(&self) -> i64 {
        self.inner.iode
    }
    /// Sine harmonic correction to orbit radius, meters.
    #[getter]
    fn crs(&self) -> f64 {
        self.inner.crs
    }
    /// Mean motion difference, radians/second.
    #[getter]
    fn delta_n(&self) -> f64 {
        self.inner.delta_n
    }
    /// Mean anomaly at reference time, radians.
    #[getter]
    fn m0(&self) -> f64 {
        self.inner.m0
    }
    /// Cosine harmonic correction to argument of latitude, radians.
    #[getter]
    fn cuc(&self) -> f64 {
        self.inner.cuc
    }
    /// Orbital eccentricity.
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.eccentricity
    }
    /// Sine harmonic correction to argument of latitude, radians.
    #[getter]
    fn cus(&self) -> f64 {
        self.inner.cus
    }
    /// Square root of semi-major axis, sqrt(meters).
    #[getter]
    fn sqrt_a(&self) -> f64 {
        self.inner.sqrt_a
    }
    /// Ephemeris reference time, seconds.
    #[getter]
    fn toe(&self) -> i64 {
        self.inner.toe
    }
    /// Fit interval flag.
    #[getter]
    fn fit_interval_flag(&self) -> i64 {
        self.inner.fit_interval_flag
    }
    /// Age of data offset.
    #[getter]
    fn aodo(&self) -> i64 {
        self.inner.aodo
    }
    /// Cosine harmonic correction to inclination, radians.
    #[getter]
    fn cic(&self) -> f64 {
        self.inner.cic
    }
    /// Longitude of ascending node at reference time, radians.
    #[getter]
    fn omega0(&self) -> f64 {
        self.inner.omega0
    }
    /// Sine harmonic correction to inclination, radians.
    #[getter]
    fn cis(&self) -> f64 {
        self.inner.cis
    }
    /// Inclination at reference time, radians.
    #[getter]
    fn i0(&self) -> f64 {
        self.inner.i0
    }
    /// Cosine harmonic correction to orbit radius, meters.
    #[getter]
    fn crc(&self) -> f64 {
        self.inner.crc
    }
    /// Argument of perigee, radians.
    #[getter]
    fn omega(&self) -> f64 {
        self.inner.omega
    }
    /// Rate of right ascension, radians/second.
    #[getter]
    fn omega_dot(&self) -> f64 {
        self.inner.omega_dot
    }
    /// Rate of inclination, radians/second.
    #[getter]
    fn idot(&self) -> f64 {
        self.inner.idot
    }

    fn __repr__(&self) -> String {
        format!(
            "LnavDecoded(week_number={}, iode={}, toe={}, sqrt_a={}, eccentricity={})",
            self.inner.week_number,
            self.inner.iode,
            self.inner.toe,
            self.inner.sqrt_a,
            self.inner.eccentricity
        )
    }
}

/// Time-of-week count from a 30-bit hand-over word or a 300-bit subframe.
///
/// Returns `None` for any other bit-vector length.
#[pyfunction]
fn lnav_tow(bits: Vec<u8>) -> Option<u64> {
    core_tow(&bits)
}

/// Subframe id (1-5) from a 30-bit hand-over word or a 300-bit subframe.
///
/// Returns `None` for any other bit-vector length.
#[pyfunction]
fn lnav_subframe_id(bits: Vec<u8>) -> Option<u64> {
    core_subframe_id(&bits)
}

#[pyfunction]
fn lnav_word_length() -> usize {
    WORD_LENGTH
}

#[pyfunction]
fn lnav_subframe_length() -> usize {
    SUBFRAME_LENGTH
}

#[pyfunction]
fn lnav_preamble() -> u32 {
    PREAMBLE
}

/// The six parity bits `[D25..D30]` of a word from its 24 source data bits and
/// the previous word's trailing parity bits `d29_prev` / `d30_prev`.
#[pyfunction]
#[pyo3(signature = (data24, d29_prev, d30_prev))]
fn lnav_parity(data24: Vec<u8>, d29_prev: u8, d30_prev: u8) -> PyResult<Vec<u8>> {
    core_parity(&data24, d29_prev, d30_prev)
        .map(|bits| bits.to_vec())
        .map_err(lnav_err)
}

/// Whether a 30-bit transmitted word's parity is valid given the previous
/// word's trailing parity bits.
#[pyfunction]
#[pyo3(signature = (word30, d29_prev, d30_prev))]
fn lnav_parity_valid(word30: Vec<u8>, d29_prev: u8, d30_prev: u8) -> bool {
    core_parity_valid(&word30, d29_prev, d30_prev)
}

/// Encode clock and ephemeris parameters into LNAV subframes 1-3.
///
/// `params` is a mapping of the 30 engineering-unit parameter fields (each an
/// `int` or `float`); `tow`, `alert`, `anti_spoof`, `integrity`, and
/// `tlm_message` are the TLM/HOW header values. Returns the three 300-bit
/// subframes (most significant bit first) as `(sf1, sf2, sf3)` lists of `0`/`1`.
#[pyfunction]
#[pyo3(signature = (params, *, tow, alert=0, anti_spoof=0, integrity=0, tlm_message=0))]
fn lnav_encode(
    params: &Bound<'_, PyDict>,
    tow: i64,
    alert: i64,
    anti_spoof: i64,
    integrity: i64,
    tlm_message: i64,
) -> PyResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let parsed = params_from_dict(params)?;
    let opts = LnavOptions {
        tow: LnavNumber::Int(tow),
        alert: LnavNumber::Int(alert),
        anti_spoof: LnavNumber::Int(anti_spoof),
        integrity: LnavNumber::Int(integrity),
        tlm_message: LnavNumber::Int(tlm_message),
    };
    let [sf1, sf2, sf3] = core_encode(&parsed, &opts).map_err(lnav_err)?;
    Ok((sf1, sf2, sf3))
}

/// Decode LNAV subframes 1-3 back into engineering-unit parameters.
///
/// Each subframe is a 300-bit list of `0`/`1`. Parity is verified on all words
/// first; a failure raises `ValueError`.
#[pyfunction]
#[pyo3(signature = (sf1, sf2, sf3))]
fn lnav_decode(sf1: Vec<u8>, sf2: Vec<u8>, sf3: Vec<u8>) -> PyResult<PyLnavDecoded> {
    let inner = core_decode(&sf1, &sf2, &sf3).map_err(lnav_err)?;
    Ok(PyLnavDecoded { inner })
}

/// Names of the 30 engineering-unit parameter fields `lnav_encode` expects, in
/// the order the core declares them.
#[pyfunction]
fn lnav_param_fields() -> Vec<String> {
    PARAM_FIELDS.iter().map(|s| (*s).to_string()).collect()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLnavDecoded>()?;
    m.add_function(wrap_pyfunction!(lnav_tow, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_subframe_id, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_word_length, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_subframe_length, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_preamble, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_parity, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_parity_valid, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_encode, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_decode, m)?)?;
    m.add_function(wrap_pyfunction!(lnav_param_fields, m)?)?;
    Ok(())
}
