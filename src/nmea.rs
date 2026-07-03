//! NMEA parser and accumulator binding.
//!
//! This module wraps the core sans-I/O NMEA parser. It exposes the pieces that
//! callers need for rover feeds without duplicating sentence parsing in Python.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::nmea::{
    group_epochs as core_group_epochs, parse_nmea as core_parse_nmea,
    parse_sentence as core_parse_sentence, write_gga as core_write_gga, Diagnostics, EpochSnapshot,
    Gga, GgaQuality, NmeaAccumulator, NmeaBody, NmeaChunkOutput, NmeaSentence, NmeaTalker,
    NmeaTime,
};
use sidereon_core::Wgs84Geodetic;

fn to_nmea_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn talker_code(talker: NmeaTalker) -> String {
    match talker.code() {
        Ok(code) => String::from_utf8_lossy(&code).into_owned(),
        Err(_) => "??".to_string(),
    }
}

fn sentence_kind(sentence: &NmeaSentence) -> &'static str {
    match sentence.body {
        NmeaBody::Gga(_) => "GGA",
        NmeaBody::Rmc(_) => "RMC",
        NmeaBody::Gsa(_) => "GSA",
        NmeaBody::Gsv(_) => "GSV",
        NmeaBody::Gst(_) => "GST",
        NmeaBody::Vtg(_) => "VTG",
        NmeaBody::Gll(_) => "GLL",
        NmeaBody::Zda(_) => "ZDA",
    }
}

fn quality(value: u8) -> GgaQuality {
    match value {
        0 => GgaQuality::Invalid,
        1 => GgaQuality::GpsSps,
        2 => GgaQuality::Differential,
        3 => GgaQuality::Pps,
        4 => GgaQuality::RtkFixed,
        5 => GgaQuality::RtkFloat,
        6 => GgaQuality::Estimated,
        7 => GgaQuality::Manual,
        8 => GgaQuality::Simulator,
        other => GgaQuality::Other(other),
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NmeaDiagnostics")]
#[derive(Clone)]
pub struct PyNmeaDiagnostics {
    inner: Diagnostics,
}

#[pymethods]
impl PyNmeaDiagnostics {
    #[getter]
    fn skip_count(&self) -> usize {
        self.inner.skips.len()
    }

    #[getter]
    fn warning_count(&self) -> usize {
        self.inner.warnings.len()
    }

    #[getter]
    fn skips(&self) -> Vec<String> {
        self.inner
            .skips
            .iter()
            .map(|skip| format!("{:?}", skip.reason))
            .collect()
    }

    #[getter]
    fn warnings(&self) -> Vec<String> {
        self.inner
            .warnings
            .iter()
            .map(|warning| format!("{:?}", warning.kind))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "NmeaDiagnostics(skip_count={}, warning_count={})",
            self.inner.skips.len(),
            self.inner.warnings.len()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NmeaGga")]
#[derive(Clone)]
pub struct PyNmeaGga {
    inner: Gga,
}

#[pymethods]
impl PyNmeaGga {
    #[getter]
    fn time(&self) -> Option<(u8, u8, u8, u32, u8)> {
        self.inner
            .time
            .map(|t| (t.hour, t.minute, t.second, t.nanos, t.decimals))
    }

    #[getter]
    fn lat_deg(&self) -> Option<f64> {
        self.inner.latitude.map(|value| value.degrees_f64())
    }

    #[getter]
    fn lon_deg(&self) -> Option<f64> {
        self.inner.longitude.map(|value| value.degrees_f64())
    }

    #[getter]
    fn quality(&self) -> Option<u8> {
        self.inner.quality.map(GgaQuality::value)
    }

    #[getter]
    fn satellites_used(&self) -> Option<u8> {
        self.inner.satellites_used
    }

    #[getter]
    fn hdop(&self) -> Option<f64> {
        self.inner.hdop
    }

    #[getter]
    fn altitude_msl_m(&self) -> Option<f64> {
        self.inner.altitude_msl_m
    }

    #[getter]
    fn geoid_separation_m(&self) -> Option<f64> {
        self.inner.geoid_separation_m
    }

    #[getter]
    fn differential_age_s(&self) -> Option<f64> {
        self.inner.differential_age_s
    }

    #[getter]
    fn differential_station_id(&self) -> Option<u16> {
        self.inner.differential_station_id
    }

    fn __repr__(&self) -> String {
        format!(
            "NmeaGga(lat_deg={:?}, lon_deg={:?}, quality={:?})",
            self.lat_deg(),
            self.lon_deg(),
            self.quality()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NmeaSentence")]
#[derive(Clone)]
pub struct PyNmeaSentence {
    inner: NmeaSentence,
}

#[pymethods]
impl PyNmeaSentence {
    #[getter]
    fn talker(&self) -> String {
        talker_code(self.inner.talker)
    }

    #[getter]
    fn kind(&self) -> &'static str {
        sentence_kind(&self.inner)
    }

    #[getter]
    fn gga(&self) -> Option<PyNmeaGga> {
        match &self.inner.body {
            NmeaBody::Gga(gga) => Some(PyNmeaGga { inner: gga.clone() }),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NmeaSentence(talker={:?}, kind={:?})",
            self.talker(),
            self.kind()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NmeaEpochSnapshot")]
#[derive(Clone)]
pub struct PyNmeaEpochSnapshot {
    inner: EpochSnapshot,
}

#[pymethods]
impl PyNmeaEpochSnapshot {
    #[getter]
    fn time(&self) -> Option<(u8, u8, u8, u32, u8)> {
        self.inner
            .time_of_day
            .map(|t| (t.hour, t.minute, t.second, t.nanos, t.decimals))
    }

    #[getter]
    fn date(&self) -> Option<(u16, u8, u8)> {
        self.inner.date.map(|d| (d.year, d.month, d.day))
    }

    #[getter]
    fn gga(&self) -> Option<PyNmeaGga> {
        self.inner
            .gga
            .as_ref()
            .map(|gga| PyNmeaGga { inner: gga.clone() })
    }

    #[getter]
    fn position_geodetic(&self) -> Option<(f64, f64, f64)> {
        self.inner.position().map(|position| {
            (
                position.lat_rad.to_degrees(),
                position.lon_rad.to_degrees(),
                position.height_m,
            )
        })
    }

    #[getter]
    fn pdop(&self) -> Option<f64> {
        self.inner.pdop()
    }

    #[getter]
    fn hdop(&self) -> Option<f64> {
        self.inner.hdop()
    }

    #[getter]
    fn vdop(&self) -> Option<f64> {
        self.inner.vdop()
    }

    #[getter]
    fn satellites_in_view(&self) -> usize {
        self.inner.satellites_in_view()
    }

    #[getter]
    fn sentence_count(&self) -> usize {
        self.inner.sentence_count
    }

    #[getter]
    fn diagnostics(&self) -> PyNmeaDiagnostics {
        PyNmeaDiagnostics {
            inner: self.inner.diagnostics.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NmeaEpochSnapshot(sentence_count={})",
            self.inner.sentence_count
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NmeaLog")]
#[derive(Clone)]
pub struct PyNmeaLog {
    sentences: Vec<NmeaSentence>,
    diagnostics: Diagnostics,
}

#[pymethods]
impl PyNmeaLog {
    #[getter]
    fn sentences(&self) -> Vec<PyNmeaSentence> {
        self.sentences
            .iter()
            .cloned()
            .map(|inner| PyNmeaSentence { inner })
            .collect()
    }

    #[getter]
    fn diagnostics(&self) -> PyNmeaDiagnostics {
        PyNmeaDiagnostics {
            inner: self.diagnostics.clone(),
        }
    }

    fn group_epochs(&self) -> Vec<PyNmeaEpochSnapshot> {
        let log = sidereon_core::nmea::NmeaLog {
            sentences: self.sentences.clone(),
        };
        core_group_epochs(&log)
            .into_iter()
            .map(|inner| PyNmeaEpochSnapshot { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!("NmeaLog(sentences={})", self.sentences.len())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NmeaChunkOutput")]
#[derive(Clone)]
pub struct PyNmeaChunkOutput {
    inner: NmeaChunkOutput,
}

#[pymethods]
impl PyNmeaChunkOutput {
    #[getter]
    fn snapshots(&self) -> Vec<PyNmeaEpochSnapshot> {
        self.inner
            .snapshots
            .iter()
            .cloned()
            .map(|inner| PyNmeaEpochSnapshot { inner })
            .collect()
    }

    #[getter]
    fn sentences(&self) -> Vec<PyNmeaSentence> {
        self.inner
            .sentences
            .iter()
            .cloned()
            .map(|inner| PyNmeaSentence { inner })
            .collect()
    }

    #[getter]
    fn diagnostics(&self) -> PyNmeaDiagnostics {
        PyNmeaDiagnostics {
            inner: self.inner.diagnostics.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NmeaChunkOutput(sentences={}, snapshots={})",
            self.inner.sentences.len(),
            self.inner.snapshots.len()
        )
    }
}

#[pyclass(module = "sidereon._sidereon", name = "NmeaAccumulator")]
pub struct PyNmeaAccumulator {
    inner: NmeaAccumulator,
}

#[pymethods]
impl PyNmeaAccumulator {
    #[new]
    #[pyo3(signature = (max_sentences_per_epoch=None))]
    fn new(max_sentences_per_epoch: Option<usize>) -> Self {
        let mut inner = NmeaAccumulator::new();
        if let Some(max) = max_sentences_per_epoch {
            inner = inner.with_max_sentences_per_epoch(max);
        }
        Self { inner }
    }

    fn push_bytes(&mut self, chunk: &[u8]) -> PyNmeaChunkOutput {
        PyNmeaChunkOutput {
            inner: self.inner.push_bytes(chunk),
        }
    }

    fn finish(&mut self) -> Option<PyNmeaEpochSnapshot> {
        self.inner
            .finish()
            .map(|inner| PyNmeaEpochSnapshot { inner })
    }

    fn retained_len(&self) -> usize {
        self.inner.retained_len()
    }

    fn __repr__(&self) -> String {
        format!(
            "NmeaAccumulator(retained_len={})",
            self.inner.retained_len()
        )
    }
}

#[pyfunction]
fn parse_nmea_sentence(line: &str) -> PyResult<PyNmeaSentence> {
    core_parse_sentence(line)
        .map(|parsed| PyNmeaSentence {
            inner: parsed.value,
        })
        .map_err(to_nmea_err)
}

#[pyfunction]
fn parse_nmea(data: &[u8]) -> PyNmeaLog {
    let parsed = core_parse_nmea(data);
    PyNmeaLog {
        sentences: parsed.value.sentences,
        diagnostics: parsed.diagnostics,
    }
}

#[pyfunction]
#[pyo3(signature = (
    lat_deg,
    lon_deg,
    height_m,
    utc_seconds_of_day,
    *,
    fix_quality=1,
    num_satellites=10,
    hdop=1.0,
    talker="GP",
))]
#[allow(clippy::too_many_arguments)]
fn write_gga(
    lat_deg: f64,
    lon_deg: f64,
    height_m: f64,
    utc_seconds_of_day: f64,
    fix_quality: u8,
    num_satellites: u8,
    hdop: f64,
    talker: &str,
) -> PyResult<String> {
    let position = Wgs84Geodetic::new(lat_deg.to_radians(), lon_deg.to_radians(), height_m)
        .map_err(to_nmea_err)?;
    let time =
        NmeaTime::from_seconds_of_day_floor_centis(utc_seconds_of_day).map_err(to_nmea_err)?;
    let gga = Gga::vrs_position(
        position,
        time,
        quality(fix_quality),
        num_satellites,
        hdop,
        7,
    )
    .map_err(to_nmea_err)?;
    core_write_gga(NmeaTalker::parse(talker), &gga).map_err(to_nmea_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNmeaDiagnostics>()?;
    m.add_class::<PyNmeaGga>()?;
    m.add_class::<PyNmeaSentence>()?;
    m.add_class::<PyNmeaEpochSnapshot>()?;
    m.add_class::<PyNmeaLog>()?;
    m.add_class::<PyNmeaChunkOutput>()?;
    m.add_class::<PyNmeaAccumulator>()?;
    m.add_function(wrap_pyfunction!(parse_nmea_sentence, m)?)?;
    m.add_function(wrap_pyfunction!(parse_nmea, m)?)?;
    m.add_function(wrap_pyfunction!(write_gga, m)?)?;
    Ok(())
}
