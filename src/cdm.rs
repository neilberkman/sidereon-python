//! CCSDS CDM binding.
//!
//! Provides typed Python value objects for the core `CdmKvn` / `CdmObject`
//! structs and parse/encode entry points for KVN and XML. The grammar and
//! serialization stay entirely in `sidereon-core`; this module only marshals
//! strings, optional fields, and numpy vectors.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::astro::cdm::{encode_kvn, encode_xml, parse_kvn, parse_xml, CdmKvn, CdmObject};

use crate::marshal::{fixed_array, FinitePolicy};
use crate::{np_array, CdmParseError};

fn to_cdm_err<E: std::fmt::Display>(err: E) -> PyErr {
    CdmParseError::new_err(err.to_string())
}

/// One object's metadata, state vector, and RTN position covariance from a CDM.
#[pyclass(module = "sidereon._sidereon", name = "CdmObject")]
#[derive(Clone, PartialEq)]
pub struct PyCdmObject {
    inner: CdmObject,
}

impl PyCdmObject {
    fn from_inner(inner: CdmObject) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCdmObject {
    #[new]
    #[pyo3(signature = (
        position_km,
        velocity_km_s,
        covariance_rtn,
        *,
        object_designator=None,
        catalog_name=None,
        object_name=None,
        international_designator=None,
        object_type=None,
        operator_contact_position=None,
        operator_organization=None,
        operator_phone=None,
        operator_email=None,
        ephemeris_name=None,
        covariance_method=None,
        maneuverable=None,
        orbit_center=None,
        ref_frame=None,
        gravity_model=None,
        atmospheric_model=None,
        n_body_perturbations=None,
        solar_rad_pressure=None,
        earth_tides=None,
        intrack_thrust=None,
        velocity_covariance_rtn=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        position_km: PyReadonlyArray1<'_, f64>,
        velocity_km_s: PyReadonlyArray1<'_, f64>,
        covariance_rtn: PyReadonlyArray1<'_, f64>,
        object_designator: Option<String>,
        catalog_name: Option<String>,
        object_name: Option<String>,
        international_designator: Option<String>,
        object_type: Option<String>,
        operator_contact_position: Option<String>,
        operator_organization: Option<String>,
        operator_phone: Option<String>,
        operator_email: Option<String>,
        ephemeris_name: Option<String>,
        covariance_method: Option<String>,
        maneuverable: Option<String>,
        orbit_center: Option<String>,
        ref_frame: Option<String>,
        gravity_model: Option<String>,
        atmospheric_model: Option<String>,
        n_body_perturbations: Option<String>,
        solar_rad_pressure: Option<String>,
        earth_tides: Option<String>,
        intrack_thrust: Option<String>,
        velocity_covariance_rtn: Option<PyReadonlyArray1<'_, f64>>,
    ) -> PyResult<Self> {
        let position = fixed_array::<3>("position_km", &position_km, FinitePolicy::AllowNonFinite)?;
        let velocity = fixed_array::<3>(
            "velocity_km_s",
            &velocity_km_s,
            FinitePolicy::AllowNonFinite,
        )?;
        let velocity_covariance_rtn = velocity_covariance_rtn
            .map(|values| {
                fixed_array::<15>(
                    "velocity_covariance_rtn",
                    &values,
                    FinitePolicy::AllowNonFinite,
                )
            })
            .transpose()?;
        Ok(Self {
            inner: CdmObject {
                object_designator,
                catalog_name,
                object_name,
                international_designator,
                object_type,
                operator_contact_position,
                operator_organization,
                operator_phone,
                operator_email,
                ephemeris_name,
                covariance_method,
                maneuverable,
                orbit_center,
                ref_frame,
                gravity_model,
                atmospheric_model,
                n_body_perturbations,
                solar_rad_pressure,
                earth_tides,
                intrack_thrust,
                state: (
                    (position[0], position[1], position[2]),
                    (velocity[0], velocity[1], velocity[2]),
                ),
                covariance_rtn: fixed_array::<6>(
                    "covariance_rtn",
                    &covariance_rtn,
                    FinitePolicy::AllowNonFinite,
                )?,
                velocity_covariance_rtn,
            },
        })
    }

    #[getter]
    fn object_designator(&self) -> Option<String> {
        self.inner.object_designator.clone()
    }

    #[getter]
    fn catalog_name(&self) -> Option<String> {
        self.inner.catalog_name.clone()
    }

    #[getter]
    fn object_name(&self) -> Option<String> {
        self.inner.object_name.clone()
    }

    #[getter]
    fn international_designator(&self) -> Option<String> {
        self.inner.international_designator.clone()
    }

    #[getter]
    fn object_type(&self) -> Option<String> {
        self.inner.object_type.clone()
    }

    #[getter]
    fn operator_contact_position(&self) -> Option<String> {
        self.inner.operator_contact_position.clone()
    }

    #[getter]
    fn operator_organization(&self) -> Option<String> {
        self.inner.operator_organization.clone()
    }

    #[getter]
    fn operator_phone(&self) -> Option<String> {
        self.inner.operator_phone.clone()
    }

    #[getter]
    fn operator_email(&self) -> Option<String> {
        self.inner.operator_email.clone()
    }

    #[getter]
    fn ephemeris_name(&self) -> Option<String> {
        self.inner.ephemeris_name.clone()
    }

    #[getter]
    fn covariance_method(&self) -> Option<String> {
        self.inner.covariance_method.clone()
    }

    #[getter]
    fn maneuverable(&self) -> Option<String> {
        self.inner.maneuverable.clone()
    }

    #[getter]
    fn orbit_center(&self) -> Option<String> {
        self.inner.orbit_center.clone()
    }

    #[getter]
    fn ref_frame(&self) -> Option<String> {
        self.inner.ref_frame.clone()
    }

    #[getter]
    fn gravity_model(&self) -> Option<String> {
        self.inner.gravity_model.clone()
    }

    #[getter]
    fn atmospheric_model(&self) -> Option<String> {
        self.inner.atmospheric_model.clone()
    }

    #[getter]
    fn n_body_perturbations(&self) -> Option<String> {
        self.inner.n_body_perturbations.clone()
    }

    #[getter]
    fn solar_rad_pressure(&self) -> Option<String> {
        self.inner.solar_rad_pressure.clone()
    }

    #[getter]
    fn earth_tides(&self) -> Option<String> {
        self.inner.earth_tides.clone()
    }

    #[getter]
    fn intrack_thrust(&self) -> Option<String> {
        self.inner.intrack_thrust.clone()
    }

    /// Position vector as a numpy `(3,)` array, kilometres.
    #[getter]
    fn position_km<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ((x, y, z), _) = self.inner.state;
        np_array(py, &[x, y, z])
    }

    /// Velocity vector as a numpy `(3,)` array, kilometres per second.
    #[getter]
    fn velocity_km_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let (_, (x_dot, y_dot, z_dot)) = self.inner.state;
        np_array(py, &[x_dot, y_dot, z_dot])
    }

    /// RTN position-covariance lower triangle `(CR_R, CT_R, CT_T, CN_R, CN_T,
    /// CN_N)` as a numpy `(6,)` array.
    #[getter]
    fn covariance_rtn<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.inner.covariance_rtn)
    }

    /// The RTN velocity-covariance lower-triangle rows completing the 6x6 matrix
    /// (`CRDOT_R, CRDOT_T, CRDOT_N, CRDOT_RDOT, CTDOT_R, ...`, 15 elements in
    /// CCSDS order) as a numpy `(15,)` array, or `None` when the producer carried
    /// only the position-covariance block.
    #[getter]
    fn velocity_covariance_rtn<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .velocity_covariance_rtn
            .map(|values| PyArray1::from_slice(py, &values))
    }

    fn __repr__(&self) -> String {
        format!(
            "CdmObject(object_name={:?}, ref_frame={:?})",
            self.inner.object_name, self.inner.ref_frame
        )
    }

    fn __eq__(&self, other: &PyCdmObject) -> bool {
        self == other
    }
}

/// A two-object CCSDS Conjunction Data Message parsed from KVN or XML.
#[pyclass(module = "sidereon._sidereon", name = "Cdm")]
#[derive(Clone, PartialEq)]
pub struct PyCdm {
    inner: CdmKvn,
}

impl PyCdm {
    fn from_inner(inner: CdmKvn) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCdm {
    #[new]
    #[pyo3(signature = (
        object1,
        object2,
        *,
        creation_date=None,
        originator=None,
        message_id=None,
        tca=None,
        miss_distance_m=None,
        relative_speed_m_s=None,
        collision_probability=None,
        collision_probability_method=None,
        hard_body_radius_m=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        object1: PyCdmObject,
        object2: PyCdmObject,
        creation_date: Option<String>,
        originator: Option<String>,
        message_id: Option<String>,
        tca: Option<String>,
        miss_distance_m: Option<f64>,
        relative_speed_m_s: Option<f64>,
        collision_probability: Option<f64>,
        collision_probability_method: Option<String>,
        hard_body_radius_m: Option<f64>,
    ) -> Self {
        Self {
            inner: CdmKvn {
                creation_date,
                originator,
                message_id,
                tca,
                miss_distance_m,
                relative_speed_m_s,
                collision_probability,
                collision_probability_method,
                hard_body_radius_m,
                object1: object1.inner,
                object2: object2.inner,
            },
        }
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
    fn message_id(&self) -> Option<String> {
        self.inner.message_id.clone()
    }

    #[getter]
    fn tca(&self) -> Option<String> {
        self.inner.tca.clone()
    }

    #[getter]
    fn miss_distance_m(&self) -> Option<f64> {
        self.inner.miss_distance_m
    }

    #[getter]
    fn relative_speed_m_s(&self) -> Option<f64> {
        self.inner.relative_speed_m_s
    }

    #[getter]
    fn collision_probability(&self) -> Option<f64> {
        self.inner.collision_probability
    }

    #[getter]
    fn collision_probability_method(&self) -> Option<String> {
        self.inner.collision_probability_method.clone()
    }

    #[getter]
    fn hard_body_radius_m(&self) -> Option<f64> {
        self.inner.hard_body_radius_m
    }

    #[getter]
    fn object1(&self) -> PyCdmObject {
        PyCdmObject::from_inner(self.inner.object1.clone())
    }

    #[getter]
    fn object2(&self) -> PyCdmObject {
        PyCdmObject::from_inner(self.inner.object2.clone())
    }

    /// Encode this message to CCSDS CDM KVN text via the core writer.
    fn to_kvn_string(&self) -> PyResult<String> {
        encode_kvn(&self.inner).map_err(to_cdm_err)
    }

    /// Encode this message to CCSDS CDM XML text via the core writer.
    fn to_xml_string(&self) -> PyResult<String> {
        encode_xml(&self.inner).map_err(to_cdm_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "Cdm(message_id={:?}, object1={:?}, object2={:?})",
            self.inner.message_id, self.inner.object1.object_name, self.inner.object2.object_name
        )
    }

    fn __eq__(&self, other: &PyCdm) -> bool {
        self == other
    }
}

/// Parse CCSDS CDM KVN text.
#[pyfunction]
fn parse_cdm_kvn(text: &str) -> PyResult<PyCdm> {
    parse_kvn(text).map(PyCdm::from_inner).map_err(to_cdm_err)
}

/// Parse CCSDS CDM XML text.
#[pyfunction]
fn parse_cdm_xml(text: &str) -> PyResult<PyCdm> {
    parse_xml(text).map(PyCdm::from_inner).map_err(to_cdm_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyCdmObject>()?;
    m.add_class::<PyCdm>()?;
    m.add_function(wrap_pyfunction!(parse_cdm_kvn, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cdm_xml, m)?)?;
    Ok(())
}
