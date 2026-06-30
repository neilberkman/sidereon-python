//! Static-arc PPP correction precompute binding.
//!
//! Thin marshaling over [`sidereon_core::ppp_corrections`]: for a precise-orbit
//! arc and a fixed receiver, precompute the per-epoch solid-earth tide
//! displacement, the per-satellite carrier-phase wind-up, and the satellite
//! antenna PCO/PCV projection. No tide, wind-up, or antenna algebra lives here;
//! the numbers are exactly what `sidereon-core` produces (0-ULP against the
//! core's own reference fixture).

use std::str::FromStr;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::ppp_corrections::{
    build, CivilDateTime, PoleTideOptions, PppCorrectionEpoch, PppCorrectionObservation,
    PppCorrectionsOptions, SatelliteAntenna, SatelliteAntennaFrequency, SatelliteAntennaOptions,
};
use sidereon_core::tides::OceanLoadingBlq;
use sidereon_core::GnssSatelliteId;

/// Number of BLQ ocean-loading constituents (M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa).
const NUM_OCEAN_CONSTITUENTS: usize = 11;

use crate::PySp3;

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
}

#[pymethods]
impl PyPppCorrectionObservation {
    #[new]
    fn new(sat: String, freq1_hz: f64, freq2_hz: f64) -> Self {
        Self {
            sat,
            freq1_hz,
            freq2_hz,
        }
    }
}

impl PyPppCorrectionObservation {
    fn to_core(&self) -> PyResult<PppCorrectionObservation> {
        Ok(PppCorrectionObservation {
            sat: parse_sat(&self.sat)?,
            freq1_hz: self.freq1_hz,
            freq2_hz: self.freq2_hz,
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
        Ok(SatelliteAntennaOptions {
            freq1_label: self.freq1_label.clone(),
            freq1_hz: self.freq1_hz,
            freq2_label: self.freq2_label.clone(),
            freq2_hz: self.freq2_hz,
            antennas: self
                .antennas
                .iter()
                .map(PySatelliteAntenna::to_core)
                .collect::<PyResult<_>>()?,
        })
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
        PoleTideOptions {
            xp_arcsec: o.xp_arcsec,
            yp_arcsec: o.yp_arcsec,
        }
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

#[pymethods]
impl PyOceanLoadingBlq {
    #[new]
    fn new(amplitude_m: Vec<Vec<f64>>, phase_deg: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            amplitude_m: blq_rows("amplitude_m", amplitude_m)?,
            phase_deg: blq_rows("phase_deg", phase_deg)?,
        })
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
    inner: sidereon_core::ppp_corrections::PppCorrections,
}

#[pymethods]
impl PyPppCorrections {
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

    fn __repr__(&self) -> String {
        format!(
            "PppCorrections(tide={}, windup_m={}, sat_pco_ecef={}, sat_pcv_m={})",
            self.inner.tide.len(),
            self.inner.windup_m.len(),
            self.inner.sat_pco_ecef.len(),
            self.inner.sat_pcv_m.len()
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
#[pyo3(signature = (sp3, epochs, receiver_ecef_m, solid_earth_tide=false, phase_windup=false, satellite_antenna=None, pole_tide=None, ocean_loading=None))]
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
) -> PyResult<PyPppCorrections> {
    let core_epochs: Vec<PppCorrectionEpoch> = epochs
        .iter()
        .map(PyPppCorrectionEpoch::to_core)
        .collect::<PyResult<_>>()?;
    let options = PppCorrectionsOptions {
        solid_earth_tide,
        pole_tide: pole_tide.as_ref().map(PoleTideOptions::from),
        ocean_loading: ocean_loading.as_ref().map(OceanLoadingBlq::from),
        phase_windup,
        satellite_antenna: satellite_antenna
            .as_ref()
            .map(PySatelliteAntennaOptions::to_core)
            .transpose()?,
    };
    let inner = build(&sp3.inner, &core_epochs, receiver_ecef_m, &options)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyPppCorrections { inner })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPppCorrectionObservation>()?;
    m.add_class::<PyPppCorrectionEpoch>()?;
    m.add_class::<PySatelliteAntennaFrequency>()?;
    m.add_class::<PySatelliteAntenna>()?;
    m.add_class::<PySatelliteAntennaOptions>()?;
    m.add_class::<PyPoleTideOptions>()?;
    m.add_class::<PyOceanLoadingBlq>()?;
    m.add_class::<PyPppCorrections>()?;
    m.add_function(wrap_pyfunction!(ppp_corrections, m)?)?;
    Ok(())
}
