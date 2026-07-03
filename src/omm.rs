//! CCSDS OMM binding.
//!
//! Exposes the core canonical OMM container plus KVN/XML/JSON parse and encode
//! functions. The Python layer performs only structural validation and marshals
//! fields into `sidereon-core`; all format grammar and serialization lives in
//! the engine.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::omm::{
    encode_json, encode_kvn, encode_xml, parse_json, parse_kvn, parse_xml, Omm, OmmEpoch,
};

use crate::OmmParseError;

fn to_omm_err<E: std::fmt::Display>(err: E) -> PyErr {
    OmmParseError::new_err(err.to_string())
}

fn finite(value: f64, name: &str) -> PyResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(PyValueError::new_err(format!("{name} must be finite")))
    }
}

/// UTC calendar epoch carried by an OMM `EPOCH` field.
#[pyclass(module = "sidereon._sidereon", name = "OmmEpoch")]
#[derive(Clone, PartialEq, Eq)]
pub struct PyOmmEpoch {
    inner: OmmEpoch,
}

impl PyOmmEpoch {
    fn from_inner(inner: OmmEpoch) -> Self {
        Self { inner }
    }

    fn iso8601_string(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second,
            self.inner.microsecond
        )
    }
}

#[pymethods]
impl PyOmmEpoch {
    #[new]
    #[pyo3(signature = (
        year,
        month,
        day,
        hour,
        minute,
        second,
        microsecond,
        femtosecond=0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
        microsecond: u32,
        femtosecond: u32,
    ) -> PyResult<Self> {
        if !(1..=12).contains(&month) {
            return Err(PyValueError::new_err("month must be in 1..=12"));
        }
        if !(1..=31).contains(&day) {
            return Err(PyValueError::new_err("day must be in 1..=31"));
        }
        if hour > 23 {
            return Err(PyValueError::new_err("hour must be in 0..=23"));
        }
        if minute > 59 {
            return Err(PyValueError::new_err("minute must be in 0..=59"));
        }
        if second > 60 {
            return Err(PyValueError::new_err("second must be in 0..=60"));
        }
        if microsecond > 999_999 {
            return Err(PyValueError::new_err("microsecond must be in 0..=999999"));
        }
        if femtosecond > 999_999_999 {
            return Err(PyValueError::new_err(
                "femtosecond must be in 0..=999999999",
            ));
        }
        Ok(Self {
            inner: OmmEpoch {
                year,
                month,
                day,
                hour,
                minute,
                second,
                microsecond,
                femtosecond,
            },
        })
    }

    #[getter]
    fn year(&self) -> i32 {
        self.inner.year
    }

    #[getter]
    fn month(&self) -> u32 {
        self.inner.month
    }

    #[getter]
    fn day(&self) -> u32 {
        self.inner.day
    }

    #[getter]
    fn hour(&self) -> u32 {
        self.inner.hour
    }

    #[getter]
    fn minute(&self) -> u32 {
        self.inner.minute
    }

    #[getter]
    fn second(&self) -> u32 {
        self.inner.second
    }

    #[getter]
    fn microsecond(&self) -> u32 {
        self.inner.microsecond
    }

    #[getter]
    fn femtosecond(&self) -> u32 {
        self.inner.femtosecond
    }

    /// ISO-8601 epoch text with microsecond precision.
    #[getter]
    fn iso8601(&self) -> String {
        self.iso8601_string()
    }

    fn __repr__(&self) -> String {
        format!("OmmEpoch({:?})", self.iso8601_string())
    }

    fn __eq__(&self, other: &PyOmmEpoch) -> bool {
        self == other
    }
}

/// Canonical, format-agnostic CCSDS Orbit Mean-Elements Message.
#[pyclass(module = "sidereon._sidereon", name = "Omm")]
#[derive(Clone, PartialEq)]
pub struct PyOmm {
    inner: Omm,
}

impl PyOmm {
    pub(crate) fn from_inner(inner: Omm) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOmm {
    #[new]
    #[pyo3(signature = (
        epoch,
        mean_motion,
        eccentricity,
        inclination_deg,
        ra_of_asc_node_deg,
        arg_of_pericenter_deg,
        mean_anomaly_deg,
        norad_cat_id,
        *,
        ccsds_omm_vers=None,
        creation_date=None,
        originator=None,
        object_name=None,
        object_id=None,
        center_name=None,
        ref_frame=None,
        time_system=None,
        mean_element_theory=None,
        ephemeris_type=0,
        classification_type=None,
        element_set_no=999,
        rev_at_epoch=0,
        bstar=0.0,
        mean_motion_dot=0.0,
        mean_motion_ddot=0.0
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        epoch: PyOmmEpoch,
        mean_motion: f64,
        eccentricity: f64,
        inclination_deg: f64,
        ra_of_asc_node_deg: f64,
        arg_of_pericenter_deg: f64,
        mean_anomaly_deg: f64,
        norad_cat_id: u32,
        ccsds_omm_vers: Option<String>,
        creation_date: Option<String>,
        originator: Option<String>,
        object_name: Option<String>,
        object_id: Option<String>,
        center_name: Option<String>,
        ref_frame: Option<String>,
        time_system: Option<String>,
        mean_element_theory: Option<String>,
        ephemeris_type: i32,
        classification_type: Option<String>,
        element_set_no: i32,
        rev_at_epoch: i64,
        bstar: f64,
        mean_motion_dot: f64,
        mean_motion_ddot: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Omm {
                ccsds_omm_vers: ccsds_omm_vers.unwrap_or_else(|| "2.0".to_string()),
                creation_date,
                originator,
                object_name,
                object_id,
                center_name,
                ref_frame,
                time_system,
                mean_element_theory,
                epoch: epoch.inner,
                mean_motion: finite(mean_motion, "mean_motion")?,
                eccentricity: finite(eccentricity, "eccentricity")?,
                inclination_deg: finite(inclination_deg, "inclination_deg")?,
                ra_of_asc_node_deg: finite(ra_of_asc_node_deg, "ra_of_asc_node_deg")?,
                arg_of_pericenter_deg: finite(arg_of_pericenter_deg, "arg_of_pericenter_deg")?,
                mean_anomaly_deg: finite(mean_anomaly_deg, "mean_anomaly_deg")?,
                ephemeris_type,
                classification_type: classification_type.unwrap_or_else(|| "U".to_string()),
                norad_cat_id,
                element_set_no,
                rev_at_epoch,
                bstar: finite(bstar, "bstar")?,
                mean_motion_dot: finite(mean_motion_dot, "mean_motion_dot")?,
                mean_motion_ddot: finite(mean_motion_ddot, "mean_motion_ddot")?,
                exact_sgp4_epoch: None,
                quantize_tle_derived_fields: true,
            },
        })
    }

    #[getter]
    fn ccsds_omm_vers(&self) -> String {
        self.inner.ccsds_omm_vers.clone()
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
    fn object_name(&self) -> Option<String> {
        self.inner.object_name.clone()
    }

    #[getter]
    fn object_id(&self) -> Option<String> {
        self.inner.object_id.clone()
    }

    #[getter]
    fn center_name(&self) -> Option<String> {
        self.inner.center_name.clone()
    }

    #[getter]
    fn ref_frame(&self) -> Option<String> {
        self.inner.ref_frame.clone()
    }

    #[getter]
    fn time_system(&self) -> Option<String> {
        self.inner.time_system.clone()
    }

    #[getter]
    fn mean_element_theory(&self) -> Option<String> {
        self.inner.mean_element_theory.clone()
    }

    #[getter]
    fn epoch(&self) -> PyOmmEpoch {
        PyOmmEpoch::from_inner(self.inner.epoch.clone())
    }

    #[getter]
    fn mean_motion(&self) -> f64 {
        self.inner.mean_motion
    }

    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.eccentricity
    }

    #[getter]
    fn inclination_deg(&self) -> f64 {
        self.inner.inclination_deg
    }

    #[getter]
    fn ra_of_asc_node_deg(&self) -> f64 {
        self.inner.ra_of_asc_node_deg
    }

    #[getter]
    fn arg_of_pericenter_deg(&self) -> f64 {
        self.inner.arg_of_pericenter_deg
    }

    #[getter]
    fn mean_anomaly_deg(&self) -> f64 {
        self.inner.mean_anomaly_deg
    }

    #[getter]
    fn ephemeris_type(&self) -> i32 {
        self.inner.ephemeris_type
    }

    #[getter]
    fn classification_type(&self) -> String {
        self.inner.classification_type.clone()
    }

    #[getter]
    fn norad_cat_id(&self) -> u32 {
        self.inner.norad_cat_id
    }

    #[getter]
    fn element_set_no(&self) -> i32 {
        self.inner.element_set_no
    }

    #[getter]
    fn rev_at_epoch(&self) -> i64 {
        self.inner.rev_at_epoch
    }

    #[getter]
    fn bstar(&self) -> f64 {
        self.inner.bstar
    }

    #[getter]
    fn mean_motion_dot(&self) -> f64 {
        self.inner.mean_motion_dot
    }

    #[getter]
    fn mean_motion_ddot(&self) -> f64 {
        self.inner.mean_motion_ddot
    }

    /// Encode this OMM to CCSDS OMM KVN text.
    fn to_kvn_string(&self) -> String {
        encode_kvn(&self.inner)
    }

    /// Encode this OMM to CCSDS OMM XML text.
    fn to_xml_string(&self) -> String {
        encode_xml(&self.inner)
    }

    /// Encode this OMM to CCSDS/CelesTrak JSON text.
    fn to_json_string(&self) -> String {
        encode_json(&self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Omm(norad_cat_id={}, object_name={:?}, epoch={:?})",
            self.inner.norad_cat_id,
            self.inner.object_name,
            PyOmmEpoch::from_inner(self.inner.epoch.clone()).iso8601_string()
        )
    }

    fn __eq__(&self, other: &PyOmm) -> bool {
        self == other
    }
}

/// Parse CCSDS OMM KVN text.
#[pyfunction]
fn parse_omm_kvn(text: &str) -> PyResult<PyOmm> {
    parse_kvn(text).map(PyOmm::from_inner).map_err(to_omm_err)
}

/// Parse CCSDS OMM XML text.
#[pyfunction]
fn parse_omm_xml(text: &str) -> PyResult<PyOmm> {
    parse_xml(text).map(PyOmm::from_inner).map_err(to_omm_err)
}

/// Parse CCSDS/CelesTrak OMM JSON text.
#[pyfunction]
fn parse_omm_json(text: &str) -> PyResult<PyOmm> {
    parse_json(text).map(PyOmm::from_inner).map_err(to_omm_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOmmEpoch>()?;
    m.add_class::<PyOmm>()?;
    m.add_function(wrap_pyfunction!(parse_omm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_xml, m)?)?;
    m.add_function(wrap_pyfunction!(parse_omm_json, m)?)?;
    Ok(())
}
