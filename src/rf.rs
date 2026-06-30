//! RF link-budget binding.
//!
//! Exposes the scalar RF helpers from `sidereon-core` and a typed `LinkBudget`
//! config object for link margin. No formula lives here: every public function
//! delegates to `sidereon_core::astro::rf` after Python-facing validation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::rf as core_rf;

fn ensure_finite(name: &str, value: f64) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

fn ensure_positive(name: &str, value: f64) -> PyResult<f64> {
    ensure_finite(name, value).and_then(|v| {
        if v > 0.0 {
            Ok(v)
        } else {
            Err(PyValueError::new_err(format!("{name} must be positive")))
        }
    })
}

/// Link-budget inputs for [`link_margin`].
#[pyclass(module = "sidereon._sidereon", name = "LinkBudget")]
#[derive(Clone, Copy)]
pub struct PyLinkBudget {
    eirp_dbw: f64,
    fspl_db: f64,
    receiver_gt_dbk: f64,
    other_losses_db: f64,
    required_cn0_dbhz: f64,
}

#[pymethods]
impl PyLinkBudget {
    /// Build a link-budget input object.
    #[new]
    #[pyo3(signature = (
        eirp_dbw,
        fspl_db,
        receiver_gt_dbk,
        required_cn0_dbhz,
        other_losses_db=0.0
    ))]
    fn new(
        eirp_dbw: f64,
        fspl_db: f64,
        receiver_gt_dbk: f64,
        required_cn0_dbhz: f64,
        other_losses_db: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            eirp_dbw: ensure_finite("eirp_dbw", eirp_dbw)?,
            fspl_db: ensure_finite("fspl_db", fspl_db)?,
            receiver_gt_dbk: ensure_finite("receiver_gt_dbk", receiver_gt_dbk)?,
            other_losses_db: ensure_finite("other_losses_db", other_losses_db)?,
            required_cn0_dbhz: ensure_finite("required_cn0_dbhz", required_cn0_dbhz)?,
        })
    }

    /// Transmitter EIRP, dBW.
    #[getter]
    fn eirp_dbw(&self) -> f64 {
        self.eirp_dbw
    }

    /// Free-space path loss, dB.
    #[getter]
    fn fspl_db(&self) -> f64 {
        self.fspl_db
    }

    /// Receiver figure of merit G/T, dB/K.
    #[getter]
    fn receiver_gt_dbk(&self) -> f64 {
        self.receiver_gt_dbk
    }

    /// Sum of miscellaneous losses, dB.
    #[getter]
    fn other_losses_db(&self) -> f64 {
        self.other_losses_db
    }

    /// Required C/N0 threshold, dB-Hz.
    #[getter]
    fn required_cn0_dbhz(&self) -> f64 {
        self.required_cn0_dbhz
    }

    fn __repr__(&self) -> String {
        format!(
            "LinkBudget(eirp_dbw={}, fspl_db={}, receiver_gt_dbk={}, required_cn0_dbhz={}, other_losses_db={})",
            self.eirp_dbw,
            self.fspl_db,
            self.receiver_gt_dbk,
            self.required_cn0_dbhz,
            self.other_losses_db
        )
    }

    fn __eq__(&self, other: &PyLinkBudget) -> bool {
        self.eirp_dbw == other.eirp_dbw
            && self.fspl_db == other.fspl_db
            && self.receiver_gt_dbk == other.receiver_gt_dbk
            && self.other_losses_db == other.other_losses_db
            && self.required_cn0_dbhz == other.required_cn0_dbhz
    }
}

impl From<&PyLinkBudget> for core_rf::LinkBudget {
    fn from(value: &PyLinkBudget) -> Self {
        Self {
            eirp_dbw: value.eirp_dbw,
            fspl_db: value.fspl_db,
            receiver_gt_dbk: value.receiver_gt_dbk,
            other_losses_db: value.other_losses_db,
            required_cn0_dbhz: value.required_cn0_dbhz,
        }
    }
}

/// Free-space path loss, dB, for range in km and frequency in MHz.
#[pyfunction]
fn fspl(distance_km: f64, frequency_mhz: f64) -> PyResult<f64> {
    core_rf::fspl(
        ensure_positive("distance_km", distance_km)?,
        ensure_positive("frequency_mhz", frequency_mhz)?,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Effective isotropic radiated power, dBW.
#[pyfunction]
fn eirp(tx_power_dbm: f64, tx_antenna_gain_dbi: f64) -> PyResult<f64> {
    core_rf::eirp(
        ensure_finite("tx_power_dbm", tx_power_dbm)?,
        ensure_finite("tx_antenna_gain_dbi", tx_antenna_gain_dbi)?,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Carrier-to-noise-density ratio, dB-Hz.
#[pyfunction]
#[pyo3(signature = (eirp_dbw, fspl_db, receiver_gt_dbk, other_losses_db=0.0))]
fn cn0(eirp_dbw: f64, fspl_db: f64, receiver_gt_dbk: f64, other_losses_db: f64) -> PyResult<f64> {
    core_rf::cn0(
        ensure_finite("eirp_dbw", eirp_dbw)?,
        ensure_finite("fspl_db", fspl_db)?,
        ensure_finite("receiver_gt_dbk", receiver_gt_dbk)?,
        ensure_finite("other_losses_db", other_losses_db)?,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Wavelength, metres, for frequency in Hz.
#[pyfunction]
fn wavelength(frequency_hz: f64) -> PyResult<f64> {
    core_rf::wavelength(ensure_positive("frequency_hz", frequency_hz)?)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Parabolic-dish antenna gain, dBi.
#[pyfunction]
fn dish_gain(diameter_m: f64, frequency_hz: f64, efficiency: f64) -> PyResult<f64> {
    core_rf::dish_gain(
        ensure_positive("diameter_m", diameter_m)?,
        ensure_positive("frequency_hz", frequency_hz)?,
        ensure_positive("efficiency", efficiency)?,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Link margin, dB, from a typed [`LinkBudget`] input.
#[pyfunction]
fn link_margin(budget: PyRef<'_, PyLinkBudget>) -> PyResult<f64> {
    let core_budget = core_rf::LinkBudget::from(&*budget);
    core_rf::link_margin(&core_budget).map_err(|err| PyValueError::new_err(err.to_string()))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyLinkBudget>()?;
    m.add_function(wrap_pyfunction!(fspl, m)?)?;
    m.add_function(wrap_pyfunction!(eirp, m)?)?;
    m.add_function(wrap_pyfunction!(cn0, m)?)?;
    m.add_function(wrap_pyfunction!(wavelength, m)?)?;
    m.add_function(wrap_pyfunction!(dish_gain, m)?)?;
    m.add_function(wrap_pyfunction!(link_margin, m)?)?;
    Ok(())
}
