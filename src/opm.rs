//! CCSDS OPM binding.
//!
//! Provides typed Python value objects for the core `Opm` metadata/state/
//! Keplerian/spacecraft/covariance/maneuver structs and parse/encode entry
//! points for KVN and XML. The grammar and serialization stay entirely in
//! `sidereon-core`; this module only marshals strings, optional fields, and
//! numpy vectors. It mirrors the sibling CDM and OMM bindings: parsed messages
//! round-trip through `to_kvn_string` / `to_xml_string`, and the same value
//! objects are constructible from Python.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::opm::{
    encode_kvn, encode_xml, parse_kvn, parse_xml, Opm, OpmAnomaly, OpmCovariance, OpmKeplerian,
    OpmManeuver, OpmMetadata, OpmSpacecraft, OpmState,
};

use crate::marshal::{
    covariance6_from_array, covariance6_to_array, fixed_array, hash_debug, FinitePolicy,
};
use crate::{np_array, OpmParseError};

fn to_opm_err<E: std::fmt::Display>(err: E) -> PyErr {
    OpmParseError::new_err(err.to_string())
}

/// OPM metadata block.
#[pyclass(module = "sidereon._sidereon", name = "OpmMetadata")]
#[derive(Clone, PartialEq)]
pub struct PyOpmMetadata {
    inner: OpmMetadata,
}

impl PyOpmMetadata {
    fn from_inner(inner: OpmMetadata) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmMetadata {
    #[new]
    fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: String,
        time_system: String,
    ) -> Self {
        Self {
            inner: OpmMetadata {
                object_name,
                object_id,
                center_name,
                ref_frame,
                time_system,
            },
        }
    }

    #[getter]
    fn object_name(&self) -> String {
        self.inner.object_name.clone()
    }

    #[getter]
    fn object_id(&self) -> String {
        self.inner.object_id.clone()
    }

    #[getter]
    fn center_name(&self) -> String {
        self.inner.center_name.clone()
    }

    #[getter]
    fn ref_frame(&self) -> String {
        self.inner.ref_frame.clone()
    }

    #[getter]
    fn time_system(&self) -> String {
        self.inner.time_system.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmMetadata(object_name={:?}, ref_frame={:?}, time_system={:?})",
            self.inner.object_name, self.inner.ref_frame, self.inner.time_system
        )
    }

    fn __eq__(&self, other: &PyOpmMetadata) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// OPM Cartesian state vector.
#[pyclass(module = "sidereon._sidereon", name = "OpmState")]
#[derive(Clone, PartialEq)]
pub struct PyOpmState {
    inner: OpmState,
}

impl PyOpmState {
    fn from_inner(inner: OpmState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmState {
    #[new]
    fn new(
        epoch: String,
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OpmState {
                epoch,
                position_km: fixed_array::<3>(
                    "position_km",
                    &position_km,
                    FinitePolicy::AllowNonFinite,
                )?,
                velocity_km_s: fixed_array::<3>(
                    "velocity_km_s",
                    &velocity_km_s,
                    FinitePolicy::AllowNonFinite,
                )?,
            },
        })
    }

    /// State epoch text exactly as carried by the message.
    #[getter]
    fn epoch(&self) -> String {
        self.inner.epoch.clone()
    }

    /// Position vector as a numpy `(3,)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position_km)
    }

    /// Velocity vector as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.velocity_km_s)
    }

    fn __repr__(&self) -> String {
        format!("OpmState(epoch={:?})", self.inner.epoch)
    }

    fn __eq__(&self, other: &PyOpmState) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// Optional OPM Keplerian elements.
///
/// Exactly one of `true_anomaly_deg` / `mean_anomaly_deg` carries the orbit
/// position angle; the other reads back as `None`.
#[pyclass(module = "sidereon._sidereon", name = "OpmKeplerian")]
#[derive(Clone, PartialEq)]
pub struct PyOpmKeplerian {
    inner: OpmKeplerian,
}

impl PyOpmKeplerian {
    fn from_inner(inner: OpmKeplerian) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmKeplerian {
    #[new]
    #[pyo3(signature = (
        semi_major_axis_km,
        eccentricity,
        inclination_deg,
        ra_of_asc_node_deg,
        arg_of_pericenter_deg,
        gm_km3_s2,
        *,
        true_anomaly_deg=None,
        mean_anomaly_deg=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        semi_major_axis_km: f64,
        eccentricity: f64,
        inclination_deg: f64,
        ra_of_asc_node_deg: f64,
        arg_of_pericenter_deg: f64,
        gm_km3_s2: f64,
        true_anomaly_deg: Option<f64>,
        mean_anomaly_deg: Option<f64>,
    ) -> PyResult<Self> {
        let anomaly = match (true_anomaly_deg, mean_anomaly_deg) {
            (Some(true_deg), None) => OpmAnomaly::True(true_deg),
            (None, Some(mean_deg)) => OpmAnomaly::Mean(mean_deg),
            (Some(_), Some(_)) => {
                return Err(PyValueError::new_err(
                    "set exactly one of true_anomaly_deg or mean_anomaly_deg, not both",
                ))
            }
            (None, None) => {
                return Err(PyValueError::new_err(
                    "set exactly one of true_anomaly_deg or mean_anomaly_deg",
                ))
            }
        };
        Ok(Self {
            inner: OpmKeplerian {
                semi_major_axis_km,
                eccentricity,
                inclination_deg,
                ra_of_asc_node_deg,
                arg_of_pericenter_deg,
                anomaly,
                gm_km3_s2,
            },
        })
    }

    #[getter]
    fn semi_major_axis_km(&self) -> f64 {
        self.inner.semi_major_axis_km
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

    /// True anomaly in degrees, or `None` when the message carries mean anomaly.
    #[getter]
    fn true_anomaly_deg(&self) -> Option<f64> {
        match self.inner.anomaly {
            OpmAnomaly::True(value) => Some(value),
            OpmAnomaly::Mean(_) => None,
        }
    }

    /// Mean anomaly in degrees, or `None` when the message carries true anomaly.
    #[getter]
    fn mean_anomaly_deg(&self) -> Option<f64> {
        match self.inner.anomaly {
            OpmAnomaly::Mean(value) => Some(value),
            OpmAnomaly::True(_) => None,
        }
    }

    #[getter]
    fn gm_km3_s2(&self) -> f64 {
        self.inner.gm_km3_s2
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmKeplerian(semi_major_axis_km={}, eccentricity={})",
            self.inner.semi_major_axis_km, self.inner.eccentricity
        )
    }

    fn __eq__(&self, other: &PyOpmKeplerian) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// Optional OPM spacecraft parameters. Every field is independently optional.
#[pyclass(module = "sidereon._sidereon", name = "OpmSpacecraft")]
#[derive(Clone, PartialEq)]
pub struct PyOpmSpacecraft {
    inner: OpmSpacecraft,
}

impl PyOpmSpacecraft {
    fn from_inner(inner: OpmSpacecraft) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmSpacecraft {
    #[new]
    #[pyo3(signature = (
        *,
        mass_kg=None,
        solar_rad_area_m2=None,
        solar_rad_coeff=None,
        drag_area_m2=None,
        drag_coeff=None
    ))]
    fn new(
        mass_kg: Option<f64>,
        solar_rad_area_m2: Option<f64>,
        solar_rad_coeff: Option<f64>,
        drag_area_m2: Option<f64>,
        drag_coeff: Option<f64>,
    ) -> Self {
        Self {
            inner: OpmSpacecraft {
                mass_kg,
                solar_rad_area_m2,
                solar_rad_coeff,
                drag_area_m2,
                drag_coeff,
            },
        }
    }

    #[getter]
    fn mass_kg(&self) -> Option<f64> {
        self.inner.mass_kg
    }

    #[getter]
    fn solar_rad_area_m2(&self) -> Option<f64> {
        self.inner.solar_rad_area_m2
    }

    #[getter]
    fn solar_rad_coeff(&self) -> Option<f64> {
        self.inner.solar_rad_coeff
    }

    #[getter]
    fn drag_area_m2(&self) -> Option<f64> {
        self.inner.drag_area_m2
    }

    #[getter]
    fn drag_coeff(&self) -> Option<f64> {
        self.inner.drag_coeff
    }

    fn __repr__(&self) -> String {
        format!("OpmSpacecraft(mass_kg={:?})", self.inner.mass_kg)
    }

    fn __eq__(&self, other: &PyOpmSpacecraft) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// Optional OPM 6x6 state covariance and its reference frame.
#[pyclass(module = "sidereon._sidereon", name = "OpmCovariance")]
#[derive(Clone, PartialEq)]
pub struct PyOpmCovariance {
    inner: OpmCovariance,
}

impl PyOpmCovariance {
    fn from_inner(inner: OpmCovariance) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmCovariance {
    #[new]
    #[pyo3(signature = (matrix, *, cov_ref_frame=None))]
    fn new(matrix: PyReadonlyArray2<'_, f64>, cov_ref_frame: Option<String>) -> PyResult<Self> {
        Ok(Self {
            inner: OpmCovariance {
                cov_ref_frame,
                matrix: covariance6_from_array(&matrix, "matrix")?,
            },
        })
    }

    /// Covariance reference frame, or `None` when not stated.
    #[getter]
    fn cov_ref_frame(&self) -> Option<String> {
        self.inner.cov_ref_frame.clone()
    }

    /// State covariance as a numpy `(6, 6)` array for `[r, v]` in km and km/s.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        covariance6_to_array(py, &self.inner.matrix)
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmCovariance(cov_ref_frame={:?})",
            self.inner.cov_ref_frame
        )
    }

    fn __eq__(&self, other: &PyOpmCovariance) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// One OPM maneuver block. Every field is mandatory in CCSDS 502.0-B when a
/// maneuver is present.
#[pyclass(module = "sidereon._sidereon", name = "OpmManeuver")]
#[derive(Clone, PartialEq)]
pub struct PyOpmManeuver {
    inner: OpmManeuver,
}

impl PyOpmManeuver {
    fn from_inner(inner: OpmManeuver) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpmManeuver {
    #[new]
    fn new(
        epoch_ignition: String,
        duration_s: f64,
        delta_mass_kg: f64,
        ref_frame: String,
        dv_km_s: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OpmManeuver {
                epoch_ignition,
                duration_s,
                delta_mass_kg,
                ref_frame,
                dv_km_s: fixed_array::<3>("dv_km_s", &dv_km_s, FinitePolicy::AllowNonFinite)?,
            },
        })
    }

    /// Maneuver ignition epoch text exactly as carried by the message.
    #[getter]
    fn epoch_ignition(&self) -> String {
        self.inner.epoch_ignition.clone()
    }

    #[getter]
    fn duration_s(&self) -> f64 {
        self.inner.duration_s
    }

    #[getter]
    fn delta_mass_kg(&self) -> f64 {
        self.inner.delta_mass_kg
    }

    #[getter]
    fn ref_frame(&self) -> String {
        self.inner.ref_frame.clone()
    }

    /// Maneuver delta-v as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn dv_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.dv_km_s)
    }

    fn __repr__(&self) -> String {
        format!(
            "OpmManeuver(epoch_ignition={:?}, ref_frame={:?})",
            self.inner.epoch_ignition, self.inner.ref_frame
        )
    }

    fn __eq__(&self, other: &PyOpmManeuver) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// A CCSDS Orbit Parameter Message parsed from KVN or XML, or built directly.
#[pyclass(module = "sidereon._sidereon", name = "Opm")]
#[derive(Clone, PartialEq)]
pub struct PyOpm {
    inner: Opm,
}

impl PyOpm {
    fn from_inner(inner: Opm) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyOpm {
    #[new]
    #[pyo3(signature = (
        metadata,
        state,
        *,
        ccsds_opm_vers=None,
        creation_date=None,
        originator=None,
        keplerian=None,
        spacecraft=None,
        covariance=None,
        maneuvers=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        metadata: PyOpmMetadata,
        state: PyOpmState,
        ccsds_opm_vers: Option<String>,
        creation_date: Option<String>,
        originator: Option<String>,
        keplerian: Option<PyOpmKeplerian>,
        spacecraft: Option<PyOpmSpacecraft>,
        covariance: Option<PyOpmCovariance>,
        maneuvers: Option<Vec<PyOpmManeuver>>,
    ) -> Self {
        Self {
            inner: Opm {
                ccsds_opm_vers: ccsds_opm_vers.unwrap_or_else(|| "2.0".to_string()),
                creation_date,
                originator,
                metadata: metadata.inner,
                state: state.inner,
                keplerian: keplerian.map(|keplerian| keplerian.inner),
                spacecraft: spacecraft.map(|spacecraft| spacecraft.inner),
                covariance: covariance.map(|covariance| covariance.inner),
                maneuvers: maneuvers
                    .unwrap_or_default()
                    .into_iter()
                    .map(|maneuver| maneuver.inner)
                    .collect(),
            },
        }
    }

    #[getter]
    fn ccsds_opm_vers(&self) -> String {
        self.inner.ccsds_opm_vers.clone()
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
    fn metadata(&self) -> PyOpmMetadata {
        PyOpmMetadata::from_inner(self.inner.metadata.clone())
    }

    #[getter]
    fn state(&self) -> PyOpmState {
        PyOpmState::from_inner(self.inner.state.clone())
    }

    #[getter]
    fn keplerian(&self) -> Option<PyOpmKeplerian> {
        self.inner.keplerian.clone().map(PyOpmKeplerian::from_inner)
    }

    #[getter]
    fn spacecraft(&self) -> Option<PyOpmSpacecraft> {
        self.inner
            .spacecraft
            .clone()
            .map(PyOpmSpacecraft::from_inner)
    }

    #[getter]
    fn covariance(&self) -> Option<PyOpmCovariance> {
        self.inner
            .covariance
            .clone()
            .map(PyOpmCovariance::from_inner)
    }

    #[getter]
    fn maneuvers(&self) -> Vec<PyOpmManeuver> {
        self.inner
            .maneuvers
            .iter()
            .cloned()
            .map(PyOpmManeuver::from_inner)
            .collect()
    }

    /// Encode this OPM to CCSDS OPM KVN text via the core writer.
    fn to_kvn_string(&self) -> String {
        encode_kvn(&self.inner)
    }

    /// Encode this OPM to CCSDS OPM XML text via the core writer.
    fn to_xml_string(&self) -> String {
        encode_xml(&self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Opm(object_name={:?}, maneuvers={})",
            self.inner.metadata.object_name,
            self.inner.maneuvers.len()
        )
    }

    fn __eq__(&self, other: &PyOpm) -> bool {
        self == other
    }

    fn __hash__(&self) -> u64 {
        hash_debug(&self.inner)
    }
}

/// Parse CCSDS OPM KVN text.
#[pyfunction]
fn parse_opm_kvn(text: &str) -> PyResult<PyOpm> {
    parse_kvn(text).map(PyOpm::from_inner).map_err(to_opm_err)
}

/// Parse CCSDS OPM XML text.
#[pyfunction]
fn parse_opm_xml(text: &str) -> PyResult<PyOpm> {
    parse_xml(text).map(PyOpm::from_inner).map_err(to_opm_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOpmMetadata>()?;
    m.add_class::<PyOpmState>()?;
    m.add_class::<PyOpmKeplerian>()?;
    m.add_class::<PyOpmSpacecraft>()?;
    m.add_class::<PyOpmCovariance>()?;
    m.add_class::<PyOpmManeuver>()?;
    m.add_class::<PyOpm>()?;
    m.add_function(wrap_pyfunction!(parse_opm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_opm_xml, m)?)?;
    Ok(())
}
