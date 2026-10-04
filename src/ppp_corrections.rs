//! Static-arc PPP correction precompute binding.
//!
//! Thin marshaling over [`sidereon_core::ppp_corrections`]: for a precise-orbit
//! arc and a fixed receiver, precompute the per-epoch solid-earth tide
//! displacement, the per-satellite carrier-phase wind-up, and the satellite
//! antenna PCO/PCV projection. No tide, wind-up, or antenna algebra lives here;
//! the numbers are exactly what `sidereon-core` produces (0-ULP against the
//! core's own reference fixture).

use std::collections::BTreeMap;
use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use sidereon_core::bias::ClockReferenceObservables;
use sidereon_core::ppp_corrections::{
    build, build_with_validity_and_tide_constants, CivilDateTime, CodeBiasOptions, PoleTideOptions,
    PppCorrectionEpoch, PppCorrectionObservation, PppCorrectionsError, PppCorrectionsOptions,
    SatelliteAntenna, SatelliteAntennaFrequency, SatelliteAntennaOptions,
};
use sidereon_core::tides::OceanLoadingBlq;
use sidereon_core::{astro::time::ValidityMode, tides::StationTideConstants};
use sidereon_core::{GnssSatelliteId, GnssSystem};

/// Number of BLQ ocean-loading constituents (M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa).
const NUM_OCEAN_CONSTITUENTS: usize = 11;

use crate::bias::PyBiasSet;
use crate::marshal::PyGnssSystem;
use crate::tides::PyStationTideConstants;
use crate::PySp3;
use crate::PyValidityMode;

type CivilTuple = (i32, u8, u8, u8, u8, f64);

fn parse_sat(token: &str) -> PyResult<GnssSatelliteId> {
    GnssSatelliteId::from_str(token)
        .map_err(|_| PyValueError::new_err(format!("invalid satellite token: {token}")))
}

fn civil(t: CivilTuple) -> CivilDateTime {
    CivilDateTime {
        year: t.0,
        month: t.1,
        day: t.2,
        hour: t.3,
        minute: t.4,
        second: t.5,
    }
}

/// One satellite observation row (carrier frequencies) for the correction precompute.
#[pyclass(module = "sidereon._sidereon", name = "PppCorrectionObservation")]
#[derive(Clone)]
pub struct PyPppCorrectionObservation {
    sat: String,
    freq1_hz: f64,
    freq2_hz: f64,
    glonass_channel: Option<i8>,
}

#[pymethods]
impl PyPppCorrectionObservation {
    #[new]
    #[pyo3(signature = (sat, freq1_hz, freq2_hz, glonass_channel=None))]
    fn new(sat: String, freq1_hz: f64, freq2_hz: f64, glonass_channel: Option<i8>) -> Self {
        Self {
            sat,
            freq1_hz,
            freq2_hz,
            glonass_channel,
        }
    }
}

impl PyPppCorrectionObservation {
    fn to_core(&self) -> PyResult<PppCorrectionObservation> {
        Ok(PppCorrectionObservation {
            sat: parse_sat(&self.sat)?,
            freq1_hz: self.freq1_hz,
            freq2_hz: self.freq2_hz,
            glonass_channel: self.glonass_channel,
        })
    }
}

/// One receiver epoch: its civil date/time, the receive time as continuous
/// seconds since J2000, and the visible-satellite frequency rows.
#[pyclass(module = "sidereon._sidereon", name = "PppCorrectionEpoch")]
#[derive(Clone)]
pub struct PyPppCorrectionEpoch {
    epoch: CivilTuple,
    t_rx_j2000_s: f64,
    observations: Vec<PyPppCorrectionObservation>,
}

#[pymethods]
impl PyPppCorrectionEpoch {
    #[new]
    #[pyo3(signature = (year, month, day, hour, minute, second, t_rx_j2000_s, observations))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        year: i32,
        month: u8,
        day: u8,
        hour: u8,
        minute: u8,
        second: f64,
        t_rx_j2000_s: f64,
        observations: Vec<PyPppCorrectionObservation>,
    ) -> Self {
        Self {
            epoch: (year, month, day, hour, minute, second),
            t_rx_j2000_s,
            observations,
        }
    }
}

#[pyclass(module = "sidereon._sidereon", name = "PppCodeBiasOptions")]
#[derive(Clone)]
pub struct PyPppCodeBiasOptions {
    inner: CodeBiasOptions,
}

fn system_pairs(
    rows: Vec<(PyGnssSystem, String, String)>,
) -> BTreeMap<GnssSystem, (String, String)> {
    rows.into_iter()
        .map(|(system, obs1, obs2)| (system.into(), (obs1, obs2)))
        .collect()
}

#[pymethods]
impl PyPppCodeBiasOptions {
    #[new]
    #[pyo3(signature = (bias_set, used_observables_default, used_observables_per_sat=None, clock_reference=None))]
    fn new(
        bias_set: &PyBiasSet,
        used_observables_default: Vec<(PyGnssSystem, String, String)>,
        used_observables_per_sat: Option<Vec<(String, String, String)>>,
        clock_reference: Option<Vec<(PyGnssSystem, String, String)>>,
    ) -> PyResult<Self> {
        let used_observables_per_sat = used_observables_per_sat
            .unwrap_or_default()
            .into_iter()
            .map(|(sat, obs1, obs2)| Ok((parse_sat(&sat)?, (obs1, obs2))))
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        let clock_reference = clock_reference.map(|rows| ClockReferenceObservables {
            per_system: system_pairs(rows),
        });
        let mut inner = CodeBiasOptions::new(bias_set.inner());
        inner.bias_set = bias_set.inner();
        inner.used_observables_per_sat = used_observables_per_sat;
        inner.used_observables_default = system_pairs(used_observables_default);
        inner.clock_reference = clock_reference;
        Ok(Self { inner })
    }

    #[getter]
    fn default_system_count(&self) -> usize {
        self.inner.used_observables_default.len()
    }
}

impl PyPppCodeBiasOptions {
    fn to_core(&self) -> CodeBiasOptions {
        self.inner.clone()
    }
}

impl PyPppCorrectionEpoch {
    fn to_core(&self) -> PyResult<PppCorrectionEpoch> {
        Ok(PppCorrectionEpoch {
            epoch: civil(self.epoch),
            t_rx_j2000_s: self.t_rx_j2000_s,
            observations: self
                .observations
                .iter()
                .map(PyPppCorrectionObservation::to_core)
                .collect::<PyResult<_>>()?,
        })
    }
}

/// Frequency-dependent satellite antenna calibration: a label, a body-frame PCO
/// `[x, y, z]` (metres), and the nadir-angle no-azimuth PCV samples as
/// `(nadir_deg, pcv_m)` pairs.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteAntennaFrequency")]
#[derive(Clone)]
pub struct PySatelliteAntennaFrequency {
    label: String,
    pco_m: [f64; 3],
    noazi_pcv_m: Vec<(f64, f64)>,
}

#[pymethods]
impl PySatelliteAntennaFrequency {
    #[new]
    fn new(label: String, pco_m: [f64; 3], noazi_pcv_m: Vec<(f64, f64)>) -> Self {
        Self {
            label,
            pco_m,
            noazi_pcv_m,
        }
    }
}

impl From<&PySatelliteAntennaFrequency> for SatelliteAntennaFrequency {
    fn from(f: &PySatelliteAntennaFrequency) -> Self {
        SatelliteAntennaFrequency {
            label: f.label.clone(),
            pco_m: f.pco_m,
            noazi_pcv_m: f.noazi_pcv_m.clone(),
        }
    }
}

/// One satellite's antenna block, selected by PRN and an optional validity window
/// (`valid_from`/`valid_until` as `(year, month, day, hour, minute, second)`).
#[pyclass(module = "sidereon._sidereon", name = "SatelliteAntenna")]
#[derive(Clone)]
pub struct PySatelliteAntenna {
    sat: String,
    valid_from: Option<CivilTuple>,
    valid_until: Option<CivilTuple>,
    frequencies: Vec<PySatelliteAntennaFrequency>,
}

#[pymethods]
impl PySatelliteAntenna {
    #[new]
    #[pyo3(signature = (sat, frequencies, valid_from=None, valid_until=None))]
    fn new(
        sat: String,
        frequencies: Vec<PySatelliteAntennaFrequency>,
        valid_from: Option<CivilTuple>,
        valid_until: Option<CivilTuple>,
    ) -> Self {
        Self {
            sat,
            valid_from,
            valid_until,
            frequencies,
        }
    }
}

impl PySatelliteAntenna {
    fn to_core(&self) -> PyResult<SatelliteAntenna> {
        Ok(SatelliteAntenna {
            sat: parse_sat(&self.sat)?,
            valid_from: self.valid_from.map(civil),
            valid_until: self.valid_until.map(civil),
            frequencies: self.frequencies.iter().map(Into::into).collect(),
        })
    }
}

/// Satellite-antenna correction options: the two carrier labels/frequencies the
/// ionosphere-free combination uses, and the per-satellite antenna blocks.
#[pyclass(module = "sidereon._sidereon", name = "SatelliteAntennaOptions")]
#[derive(Clone)]
pub struct PySatelliteAntennaOptions {
    freq1_label: String,
    freq1_hz: f64,
    freq2_label: String,
    freq2_hz: f64,
    antennas: Vec<PySatelliteAntenna>,
}

#[pymethods]
impl PySatelliteAntennaOptions {
    #[new]
    fn new(
        freq1_label: String,
        freq1_hz: f64,
        freq2_label: String,
        freq2_hz: f64,
        antennas: Vec<PySatelliteAntenna>,
    ) -> Self {
        Self {
            freq1_label,
            freq1_hz,
            freq2_label,
            freq2_hz,
            antennas,
        }
    }
}

impl PySatelliteAntennaOptions {
    fn to_core(&self) -> PyResult<SatelliteAntennaOptions> {
        let antennas: Vec<SatelliteAntenna> = self
            .antennas
            .iter()
            .map(PySatelliteAntenna::to_core)
            .collect::<PyResult<_>>()?;
        let mut options = SatelliteAntennaOptions::new(
            self.freq1_label.clone(),
            self.freq1_hz,
            self.freq2_label.clone(),
            self.freq2_hz,
            antennas.clone(),
        );
        options.freq1_label = self.freq1_label.clone();
        options.freq1_hz = self.freq1_hz;
        options.freq2_label = self.freq2_label.clone();
        options.freq2_hz = self.freq2_hz;
        options.antennas = antennas;
        Ok(options)
    }
}

/// Solid-earth pole-tide options: the IERS polar motion of the date in
/// arcseconds. Polar motion is not in the engine's embedded EOP table, so the
/// caller supplies it (a single daily value is representative across a static arc).
#[pyclass(module = "sidereon._sidereon", name = "PoleTideOptions")]
#[derive(Clone, Copy)]
pub struct PyPoleTideOptions {
    xp_arcsec: f64,
    yp_arcsec: f64,
}

#[pymethods]
impl PyPoleTideOptions {
    #[new]
    fn new(xp_arcsec: f64, yp_arcsec: f64) -> Self {
        Self {
            xp_arcsec,
            yp_arcsec,
        }
    }
}

impl From<&PyPoleTideOptions> for PoleTideOptions {
    fn from(o: &PyPoleTideOptions) -> Self {
        let mut opts = PoleTideOptions::new(o.xp_arcsec, o.yp_arcsec);
        opts.xp_arcsec = o.xp_arcsec;
        opts.yp_arcsec = o.yp_arcsec;
        opts
    }
}

/// Per-station ocean-loading BLQ coefficients (Bos-Scherneck / HARDISP format).
///
/// `amplitude_m` (metres) and `phase_deg` (degrees, positive lag) are each a
/// `3 x 11` nested list indexed `[component][constituent]`: the component order
/// is radial/up (0), tangential EW/west (1), tangential NS/south (2); the
/// constituent order is the BLQ column order M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa.
#[pyclass(module = "sidereon._sidereon", name = "OceanLoadingBlq")]
#[derive(Clone)]
pub struct PyOceanLoadingBlq {
    amplitude_m: [[f64; NUM_OCEAN_CONSTITUENTS]; 3],
    phase_deg: [[f64; NUM_OCEAN_CONSTITUENTS]; 3],
}

fn blq_rows(name: &str, rows: Vec<Vec<f64>>) -> PyResult<[[f64; NUM_OCEAN_CONSTITUENTS]; 3]> {
    if rows.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have 3 component rows (radial, EW, NS), got {}",
            rows.len()
        )));
    }
    let mut out = [[0.0; NUM_OCEAN_CONSTITUENTS]; 3];
    for (i, row) in rows.into_iter().enumerate() {
        if row.len() != NUM_OCEAN_CONSTITUENTS {
            return Err(PyValueError::new_err(format!(
                "{name} row {i} must have {NUM_OCEAN_CONSTITUENTS} constituents, got {}",
                row.len()
            )));
        }
        out[i].copy_from_slice(&row);
    }
    Ok(out)
}

impl PyOceanLoadingBlq {
    /// Wrap core coefficients, for the BLQ block binding.
    pub(crate) fn from_core(blq: OceanLoadingBlq) -> Self {
        Self {
            amplitude_m: blq.amplitude_m,
            phase_deg: blq.phase_deg,
        }
    }

    pub(crate) fn to_core(&self) -> OceanLoadingBlq {
        OceanLoadingBlq {
            amplitude_m: self.amplitude_m,
            phase_deg: self.phase_deg,
        }
    }
}

#[pymethods]
impl PyOceanLoadingBlq {
    #[new]
    fn new(amplitude_m: Vec<Vec<f64>>, phase_deg: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            amplitude_m: blq_rows("amplitude_m", amplitude_m)?,
            phase_deg: blq_rows("phase_deg", phase_deg)?,
        })
    }

    #[staticmethod]
    fn from_blq_block(
        py: Python<'_>,
        text: &str,
    ) -> PyResult<crate::tides::PyOceanLoadingBlqBlock> {
        crate::tides::ocean_loading_blq_block_from_text(py, text)
    }

    /// Amplitudes in metres, `[component][constituent]`.
    #[getter]
    fn amplitude_m(&self) -> Vec<Vec<f64>> {
        self.amplitude_m.iter().map(|row| row.to_vec()).collect()
    }

    /// Greenwich phase lags in degrees, `[component][constituent]`.
    #[getter]
    fn phase_deg(&self) -> Vec<Vec<f64>> {
        self.phase_deg.iter().map(|row| row.to_vec()).collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "OceanLoadingBlq(amplitude_m={:?}, phase_deg={:?})",
            self.amplitude_m, self.phase_deg
        )
    }

    fn __eq__(&self, other: &PyOceanLoadingBlq) -> bool {
        OceanLoadingBlq::from(self) == OceanLoadingBlq::from(other)
    }
}

impl From<&PyOceanLoadingBlq> for OceanLoadingBlq {
    fn from(b: &PyOceanLoadingBlq) -> Self {
        OceanLoadingBlq {
            amplitude_m: b.amplitude_m,
            phase_deg: b.phase_deg,
        }
    }
}

/// Precomputed PPP correction tables.
///
/// Each field is a list keyed by the input epoch index. `tide` is
/// `(epoch_index, (dx, dy, dz))` solid-earth displacement (metres);
/// `windup_m`/`sat_pcv_m` are `(satellite_token, epoch_index, value_m)`;
/// `sat_pco_ecef` is `(satellite_token, epoch_index, (dx, dy, dz))` in metres.
#[pyclass(module = "sidereon._sidereon", name = "PppCorrections")]
pub struct PyPppCorrections {
    pub(crate) inner: sidereon_core::ppp_corrections::PppCorrections,
    degraded_reason: Option<&'static str>,
}

fn ppp_corrections_error(py: Python<'_>, error: PppCorrectionsError) -> PyErr {
    use sidereon_core::astro::time::{CoverageError, DegradeReason};

    let details = PyDict::new(py);
    let (kind, epoch_index) = match &error {
        PppCorrectionsError::InvalidInput { field, reason } => {
            let _ = details.set_item("field", field);
            let _ = details.set_item("reason", reason);
            ("invalid_input", None)
        }
        PppCorrectionsError::Epoch {
            epoch_index,
            source,
        } => {
            let _ = details.set_item("source_kind", "Epoch");
            let _ = details.set_item("epoch_index", *epoch_index);
            match source {
                CoverageError::InvalidInput { field, kind } => {
                    let _ = details.set_item("field", field);
                    let _ = details.set_item("input_kind", format!("{kind:?}"));
                }
                CoverageError::OutsideCoverage(reason) => {
                    let label = match reason {
                        DegradeReason::BeforeCoverage => "before_coverage",
                        DegradeReason::AfterCoverage => "after_coverage",
                    };
                    let _ = details.set_item("reason", label);
                }
            }
            ("epoch", Some(*epoch_index))
        }
        PppCorrectionsError::Tide {
            epoch_index,
            source,
        }
        | PppCorrectionsError::PoleTide {
            epoch_index,
            source,
        }
        | PppCorrectionsError::OceanLoading {
            epoch_index,
            source,
        } => {
            let source_kind = crate::tides::tide_error_kind(source);
            let _ = details.set_item("source_kind", source_kind);
            let _ = details.set_item("epoch_index", *epoch_index);
            for (key, value) in crate::tides::tide_error_details(source) {
                let _ = details.set_item(key, value);
            }
            let kind = match &error {
                PppCorrectionsError::Tide { .. } => "tide",
                PppCorrectionsError::PoleTide { .. } => "pole_tide",
                _ => "ocean_loading",
            };
            (kind, Some(*epoch_index))
        }
        PppCorrectionsError::WindupFrequency {
            epoch_index,
            sat,
            field,
            reason,
        }
        | PppCorrectionsError::CodeBiasObservable {
            epoch_index,
            sat,
            field,
            reason,
        } => {
            let kind = if matches!(&error, PppCorrectionsError::WindupFrequency { .. }) {
                "windup_frequency"
            } else {
                "code_bias_observable"
            };
            let _ = details.set_item("epoch_index", *epoch_index);
            let _ = details.set_item("satellite", sat.to_string());
            let _ = details.set_item("field", field);
            let _ = details.set_item("reason", reason);
            (kind, Some(*epoch_index))
        }
        PppCorrectionsError::SatelliteAntennaFrequency { field, reason } => {
            let _ = details.set_item("field", field);
            let _ = details.set_item("reason", reason);
            ("satellite_antenna_frequency", None)
        }
        PppCorrectionsError::Bias { source } => {
            let _ = details.set_item("source_kind", crate::bias::bias_error_kind(source));
            match crate::bias::bias_error_details(py, source) {
                Ok(source_details) => {
                    for (key, value) in source_details.iter() {
                        let _ = details.set_item(key, value);
                    }
                }
                Err(py_error) => return py_error,
            }
            ("bias", None)
        }
    };
    let error_type = match crate::ppp_corrections_error_type(py) {
        Ok(error_type) => error_type,
        Err(py_error) => return py_error,
    };
    let exception = PyErr::from_type(error_type, error.to_string());
    let value = exception.value(py);
    let _ = value.setattr("kind", kind);
    let _ = value.setattr("epoch_index", epoch_index);
    let _ = value.setattr("details", details);
    exception
}

#[pymethods]
impl PyPppCorrections {
    #[getter]
    fn ut1_degraded(&self) -> bool {
        self.degraded_reason.is_some()
    }

    #[getter]
    fn degradation_reason(&self) -> Option<&'static str> {
        self.degraded_reason
    }

    #[getter]
    fn tide(&self) -> Vec<(usize, (f64, f64, f64))> {
        self.inner
            .tide
            .iter()
            .map(|c| (c.epoch_index, (c.vector_m[0], c.vector_m[1], c.vector_m[2])))
            .collect()
    }

    #[getter]
    fn pole_tide(&self) -> Vec<(usize, (f64, f64, f64))> {
        self.inner
            .pole_tide
            .iter()
            .map(|c| (c.epoch_index, (c.vector_m[0], c.vector_m[1], c.vector_m[2])))
            .collect()
    }

    #[getter]
    fn ocean_loading(&self) -> Vec<(usize, (f64, f64, f64))> {
        self.inner
            .ocean_loading
            .iter()
            .map(|c| (c.epoch_index, (c.vector_m[0], c.vector_m[1], c.vector_m[2])))
            .collect()
    }

    #[getter]
    fn windup_m(&self) -> Vec<(String, usize, f64)> {
        self.inner
            .windup_m
            .iter()
            .map(|c| (c.sat.to_string(), c.epoch_index, c.value_m))
            .collect()
    }

    #[getter]
    fn sat_pco_ecef(&self) -> Vec<(String, usize, (f64, f64, f64))> {
        self.inner
            .sat_pco_ecef
            .iter()
            .map(|c| {
                (
                    c.sat.to_string(),
                    c.epoch_index,
                    (c.vector_m[0], c.vector_m[1], c.vector_m[2]),
                )
            })
            .collect()
    }

    #[getter]
    fn sat_pcv_m(&self) -> Vec<(String, usize, f64)> {
        self.inner
            .sat_pcv_m
            .iter()
            .map(|c| (c.sat.to_string(), c.epoch_index, c.value_m))
            .collect()
    }

    /// Clock-datum code-bias corrections in meters as (satellite_token, epoch_index, value_m).
    #[getter]
    fn code_bias_m(&self) -> Vec<(String, usize, f64)> {
        self.inner
            .code_bias_m
            .iter()
            .map(|c| (c.sat.to_string(), c.epoch_index, c.value_m))
            .collect()
    }

    /// Diagnostics containing missing-metadata warnings for observations without a code-bias used-observable mapping.
    #[getter]
    fn diagnostics(&self) -> crate::format_diagnostics::PyFormatDiagnostics {
        crate::format_diagnostics::PyFormatDiagnostics::from_inner(self.inner.diagnostics.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "PppCorrections(tide={}, windup_m={}, sat_pco_ecef={}, sat_pcv_m={}, code_bias_m={})",
            self.inner.tide.len(),
            self.inner.windup_m.len(),
            self.inner.sat_pco_ecef.len(),
            self.inner.sat_pcv_m.len(),
            self.inner.code_bias_m.len()
        )
    }
}

pub(crate) fn build_core_corrections_options(
    solid_earth_tide: bool,
    phase_windup: bool,
    satellite_antenna: Option<&PySatelliteAntennaOptions>,
    pole_tide: Option<&PyPoleTideOptions>,
    ocean_loading: Option<&PyOceanLoadingBlq>,
    code_bias: Option<&PyPppCodeBiasOptions>,
) -> PyResult<PppCorrectionsOptions> {
    let mut options = PppCorrectionsOptions::new();
    options.solid_earth_tide = solid_earth_tide;
    options.pole_tide = pole_tide.map(PoleTideOptions::from);
    options.ocean_loading = ocean_loading.map(OceanLoadingBlq::from);
    options.phase_windup = phase_windup;
    options.satellite_antenna = satellite_antenna
        .map(PySatelliteAntennaOptions::to_core)
        .transpose()?;
    options.code_bias = code_bias.map(PyPppCodeBiasOptions::to_core);
    Ok(options)
}

/// Precompute switches and options for static PPP range corrections.
#[pyclass(module = "sidereon._sidereon", name = "PppCorrectionsOptions")]
#[derive(Clone)]
pub struct PyPppCorrectionsOptions {
    pub(crate) inner: PppCorrectionsOptions,
    solid_earth_tide: bool,
    phase_windup: bool,
    satellite_antenna: Option<PySatelliteAntennaOptions>,
    pole_tide: Option<PyPoleTideOptions>,
    ocean_loading: Option<PyOceanLoadingBlq>,
    code_bias: Option<PyPppCodeBiasOptions>,
}

#[pymethods]
impl PyPppCorrectionsOptions {
    /// Create PPP correction precompute options.
    #[new]
    #[pyo3(signature = (
        solid_earth_tide=false,
        phase_windup=false,
        satellite_antenna=None,
        pole_tide=None,
        ocean_loading=None,
        code_bias=None,
    ))]
    fn new(
        solid_earth_tide: bool,
        phase_windup: bool,
        satellite_antenna: Option<PySatelliteAntennaOptions>,
        pole_tide: Option<PyPoleTideOptions>,
        ocean_loading: Option<PyOceanLoadingBlq>,
        code_bias: Option<PyPppCodeBiasOptions>,
    ) -> PyResult<Self> {
        let inner = build_core_corrections_options(
            solid_earth_tide,
            phase_windup,
            satellite_antenna.as_ref(),
            pole_tide.as_ref(),
            ocean_loading.as_ref(),
            code_bias.as_ref(),
        )?;
        Ok(Self {
            inner,
            solid_earth_tide,
            phase_windup,
            satellite_antenna,
            pole_tide,
            ocean_loading,
            code_bias,
        })
    }

    /// Solid-earth tide correction switch.
    #[getter]
    fn solid_earth_tide(&self) -> bool {
        self.solid_earth_tide
    }

    /// Carrier-phase wind-up correction switch.
    #[getter]
    fn phase_windup(&self) -> bool {
        self.phase_windup
    }

    /// Satellite antenna correction options.
    #[getter]
    fn satellite_antenna(&self) -> Option<PySatelliteAntennaOptions> {
        self.satellite_antenna.clone()
    }

    /// Solid-earth pole tide options.
    #[getter]
    fn pole_tide(&self) -> Option<PyPoleTideOptions> {
        self.pole_tide
    }

    /// Ocean tide loading coefficients.
    #[getter]
    fn ocean_loading(&self) -> Option<PyOceanLoadingBlq> {
        self.ocean_loading.clone()
    }

    /// Clock-datum code bias correction options.
    #[getter]
    fn code_bias(&self) -> Option<PyPppCodeBiasOptions> {
        self.code_bias.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "PppCorrectionsOptions(solid_earth_tide={}, phase_windup={}, satellite_antenna={}, pole_tide={}, ocean_loading={}, code_bias={})",
            self.solid_earth_tide,
            self.phase_windup,
            self.satellite_antenna.is_some(),
            self.pole_tide.is_some(),
            self.ocean_loading.is_some(),
            self.code_bias.is_some(),
        )
    }
}

/// Build static PPP correction tables for a precise-orbit arc.
///
/// `epochs` is a list of `PppCorrectionEpoch`; `receiver_ecef_m` is the fixed
/// receiver position (metres). The three switches select which corrections to
/// compute: `solid_earth_tide`, `phase_windup`, and `satellite_antenna` (a
/// `SatelliteAntennaOptions` or `None`); `pole_tide` (a `PoleTideOptions` or
/// `None`) adds the solid-earth pole tide; `ocean_loading` (an `OceanLoadingBlq`
/// or `None`) adds ocean tide loading. Returns a `PppCorrections`. Raises
/// `ValueError` on malformed input, an invalid epoch, or a tide/coverage failure.
#[pyfunction]
#[pyo3(signature = (sp3, epochs, receiver_ecef_m, solid_earth_tide=false, phase_windup=false, satellite_antenna=None, pole_tide=None, ocean_loading=None, code_bias=None))]
#[allow(clippy::too_many_arguments)]
fn ppp_corrections(
    sp3: &PySp3,
    epochs: Vec<PyPppCorrectionEpoch>,
    receiver_ecef_m: [f64; 3],
    solid_earth_tide: bool,
    phase_windup: bool,
    satellite_antenna: Option<PySatelliteAntennaOptions>,
    pole_tide: Option<PyPoleTideOptions>,
    ocean_loading: Option<PyOceanLoadingBlq>,
    code_bias: Option<PyPppCodeBiasOptions>,
) -> PyResult<PyPppCorrections> {
    let core_epochs: Vec<PppCorrectionEpoch> = epochs
        .iter()
        .map(PyPppCorrectionEpoch::to_core)
        .collect::<PyResult<_>>()?;
    let options = build_core_corrections_options(
        solid_earth_tide,
        phase_windup,
        satellite_antenna.as_ref(),
        pole_tide.as_ref(),
        ocean_loading.as_ref(),
        code_bias.as_ref(),
    )?;
    let inner = build(&sp3.inner, &core_epochs, receiver_ecef_m, &options)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyPppCorrections {
        inner,
        degraded_reason: None,
    })
}

#[pyfunction]
#[pyo3(signature = (
    sp3,
    epochs,
    receiver_ecef_m,
    solid_earth_tide=false,
    phase_windup=false,
    satellite_antenna=None,
    pole_tide=None,
    ocean_loading=None,
    code_bias=None,
    validity=PyValidityMode::STRICT,
    tide_constants=PyStationTideConstants::CONVENTIONS,
))]
#[allow(clippy::too_many_arguments)]
fn ppp_corrections_with_validity_and_tide_constants(
    py: Python<'_>,
    sp3: &PySp3,
    epochs: Vec<PyPppCorrectionEpoch>,
    receiver_ecef_m: [f64; 3],
    solid_earth_tide: bool,
    phase_windup: bool,
    satellite_antenna: Option<PySatelliteAntennaOptions>,
    pole_tide: Option<PyPoleTideOptions>,
    ocean_loading: Option<PyOceanLoadingBlq>,
    code_bias: Option<PyPppCodeBiasOptions>,
    validity: PyValidityMode,
    tide_constants: PyStationTideConstants,
) -> PyResult<PyPppCorrections> {
    let core_epochs: Vec<PppCorrectionEpoch> = epochs
        .iter()
        .map(PyPppCorrectionEpoch::to_core)
        .collect::<PyResult<_>>()?;
    let options = build_core_corrections_options(
        solid_earth_tide,
        phase_windup,
        satellite_antenna.as_ref(),
        pole_tide.as_ref(),
        ocean_loading.as_ref(),
        code_bias.as_ref(),
    )?;
    let validated = build_with_validity_and_tide_constants(
        &sp3.inner,
        &core_epochs,
        receiver_ecef_m,
        &options,
        ValidityMode::from(validity),
        StationTideConstants::from(tide_constants),
    )
    .map_err(|error| ppp_corrections_error(py, error))?;
    let degraded_reason = validated.degraded.map(|reason| match reason {
        sidereon_core::astro::time::DegradeReason::BeforeCoverage => "before_coverage",
        sidereon_core::astro::time::DegradeReason::AfterCoverage => "after_coverage",
    });
    Ok(PyPppCorrections {
        inner: validated.value,
        degraded_reason,
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPppCorrectionObservation>()?;
    m.add_class::<PyPppCorrectionEpoch>()?;
    m.add_class::<PyPppCodeBiasOptions>()?;
    m.add_class::<PySatelliteAntennaFrequency>()?;
    m.add_class::<PySatelliteAntenna>()?;
    m.add_class::<PySatelliteAntennaOptions>()?;
    m.add_class::<PyPoleTideOptions>()?;
    m.add_class::<PyOceanLoadingBlq>()?;
    m.add_class::<PyPppCorrections>()?;
    m.add_class::<PyPppCorrectionsOptions>()?;
    m.add_function(wrap_pyfunction!(ppp_corrections, m)?)?;
    m.add_function(wrap_pyfunction!(
        ppp_corrections_with_validity_and_tide_constants,
        m
    )?)?;
    Ok(())
}
