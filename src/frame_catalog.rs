//! Terrestrial frame catalog binding.
//!
//! Exposes the core ITRF/ETRF Helmert catalog and station position transforms
//! with only numpy vector marshaling at the Python boundary.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyModule};

use sidereon_core::frame_catalog as core_catalog;
use sidereon_core::{
    HelmertParameters, HelmertRates, HelmertTransform, TerrestrialFrame, TerrestrialPositionM,
    TerrestrialState, TerrestrialVelocityMPerYear,
};

use crate::marshal::{fixed_array, FinitePolicy};
use crate::np_array;

fn to_frame_catalog_err<E: std::fmt::Display>(err: E) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// Supported terrestrial reference-frame realization.
#[pyclass(module = "sidereon._sidereon", name = "TerrestrialFrame", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTerrestrialFrame {
    /// International Terrestrial Reference Frame 2020.
    ITRF2020,
    /// International Terrestrial Reference Frame 2014.
    ITRF2014,
    /// International Terrestrial Reference Frame 2008.
    ITRF2008,
    /// European Terrestrial Reference Frame 2020.
    ETRF2020,
}

impl PyTerrestrialFrame {
    fn from_label(value: &str) -> PyResult<Self> {
        match value.to_ascii_uppercase().as_str() {
            "ITRF2020" => Ok(Self::ITRF2020),
            "ITRF2014" => Ok(Self::ITRF2014),
            "ITRF2008" => Ok(Self::ITRF2008),
            "ETRF2020" => Ok(Self::ETRF2020),
            other => Err(PyValueError::new_err(format!(
                "unknown terrestrial frame {other:?}; expected ITRF2020, ITRF2014, ITRF2008, or ETRF2020"
            ))),
        }
    }
}

impl From<PyTerrestrialFrame> for TerrestrialFrame {
    fn from(value: PyTerrestrialFrame) -> Self {
        match value {
            PyTerrestrialFrame::ITRF2020 => Self::Itrf2020,
            PyTerrestrialFrame::ITRF2014 => Self::Itrf2014,
            PyTerrestrialFrame::ITRF2008 => Self::Itrf2008,
            PyTerrestrialFrame::ETRF2020 => Self::Etrf2020,
        }
    }
}

impl From<TerrestrialFrame> for PyTerrestrialFrame {
    fn from(value: TerrestrialFrame) -> Self {
        match value {
            TerrestrialFrame::Itrf2020 => Self::ITRF2020,
            TerrestrialFrame::Itrf2014 => Self::ITRF2014,
            TerrestrialFrame::Itrf2008 => Self::ITRF2008,
            TerrestrialFrame::Etrf2020 => Self::ETRF2020,
        }
    }
}

#[pymethods]
impl PyTerrestrialFrame {
    /// Stable catalog label.
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::ITRF2020 => "ITRF2020",
            Self::ITRF2014 => "ITRF2014",
            Self::ITRF2008 => "ITRF2008",
            Self::ETRF2020 => "ETRF2020",
        }
    }

    fn __repr__(&self) -> &'static str {
        match self {
            Self::ITRF2020 => "TerrestrialFrame.ITRF2020",
            Self::ITRF2014 => "TerrestrialFrame.ITRF2014",
            Self::ITRF2008 => "TerrestrialFrame.ITRF2008",
            Self::ETRF2020 => "TerrestrialFrame.ETRF2020",
        }
    }
}

fn extract_terrestrial_frame(obj: &Bound<'_, PyAny>) -> PyResult<PyTerrestrialFrame> {
    if let Ok(frame) = obj.extract::<PyTerrestrialFrame>() {
        return Ok(frame);
    }
    PyTerrestrialFrame::from_label(&obj.extract::<String>()?)
}

fn position_from_array(
    name: &str,
    values: &PyReadonlyArray1<'_, f64>,
) -> PyResult<TerrestrialPositionM> {
    let values = fixed_array::<3>(name, values, FinitePolicy::AllowNonFinite)?;
    TerrestrialPositionM::from_array(values).map_err(to_frame_catalog_err)
}

fn velocity_from_array(
    name: &str,
    values: &PyReadonlyArray1<'_, f64>,
) -> PyResult<TerrestrialVelocityMPerYear> {
    let values = fixed_array::<3>(name, values, FinitePolicy::AllowNonFinite)?;
    TerrestrialVelocityMPerYear::from_array(values).map_err(to_frame_catalog_err)
}

/// Helmert parameters in the units used by the published tables.
#[pyclass(module = "sidereon._sidereon", name = "HelmertParameters")]
#[derive(Clone, Copy)]
pub struct PyHelmertParameters {
    inner: HelmertParameters,
}

impl From<HelmertParameters> for PyHelmertParameters {
    fn from(inner: HelmertParameters) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyHelmertParameters {
    /// Build published-unit Helmert parameters.
    #[new]
    fn new(
        translation_mm: PyReadonlyArray1<'_, f64>,
        scale_ppb: f64,
        rotation_mas: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: HelmertParameters {
                translation_mm: fixed_array::<3>(
                    "translation_mm",
                    &translation_mm,
                    FinitePolicy::AllowNonFinite,
                )?,
                scale_ppb,
                rotation_mas: fixed_array::<3>(
                    "rotation_mas",
                    &rotation_mas,
                    FinitePolicy::AllowNonFinite,
                )?,
            },
        })
    }

    /// Translation components `[Tx, Ty, Tz]`, millimetres.
    #[getter]
    fn translation_mm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.translation_mm)
    }

    /// Scale difference `D`, parts per billion.
    #[getter]
    fn scale_ppb(&self) -> f64 {
        self.inner.scale_ppb
    }

    /// Rotation components `[Rx, Ry, Rz]`, milliarcseconds.
    #[getter]
    fn rotation_mas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.rotation_mas)
    }

    fn __repr__(&self) -> String {
        format!(
            "HelmertParameters(translation_mm={:?}, scale_ppb={}, rotation_mas={:?})",
            self.inner.translation_mm, self.inner.scale_ppb, self.inner.rotation_mas
        )
    }
}

/// Helmert parameter rates in the units used by the published tables.
#[pyclass(module = "sidereon._sidereon", name = "HelmertRates")]
#[derive(Clone, Copy)]
pub struct PyHelmertRates {
    inner: HelmertRates,
}

impl From<HelmertRates> for PyHelmertRates {
    fn from(inner: HelmertRates) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyHelmertRates {
    /// Build published-unit Helmert rates.
    #[new]
    fn new(
        translation_mm_per_year: PyReadonlyArray1<'_, f64>,
        scale_ppb_per_year: f64,
        rotation_mas_per_year: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: HelmertRates {
                translation_mm_per_year: fixed_array::<3>(
                    "translation_mm_per_year",
                    &translation_mm_per_year,
                    FinitePolicy::AllowNonFinite,
                )?,
                scale_ppb_per_year,
                rotation_mas_per_year: fixed_array::<3>(
                    "rotation_mas_per_year",
                    &rotation_mas_per_year,
                    FinitePolicy::AllowNonFinite,
                )?,
            },
        })
    }

    /// Translation rates `[Tx, Ty, Tz]`, millimetres per year.
    #[getter]
    fn translation_mm_per_year<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.translation_mm_per_year)
    }

    /// Scale rate `D`, parts per billion per year.
    #[getter]
    fn scale_ppb_per_year(&self) -> f64 {
        self.inner.scale_ppb_per_year
    }

    /// Rotation rates `[Rx, Ry, Rz]`, milliarcseconds per year.
    #[getter]
    fn rotation_mas_per_year<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.rotation_mas_per_year)
    }

    fn __repr__(&self) -> String {
        format!(
            "HelmertRates(translation_mm_per_year={:?}, scale_ppb_per_year={}, rotation_mas_per_year={:?})",
            self.inner.translation_mm_per_year,
            self.inner.scale_ppb_per_year,
            self.inner.rotation_mas_per_year
        )
    }
}

/// One published 14-parameter Helmert catalog entry.
#[pyclass(module = "sidereon._sidereon", name = "HelmertTransform")]
#[derive(Clone, Copy)]
pub struct PyHelmertTransform {
    inner: HelmertTransform,
}

impl From<HelmertTransform> for PyHelmertTransform {
    fn from(inner: HelmertTransform) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyHelmertTransform {
    /// Source frame for the published forward transform.
    #[getter]
    #[allow(clippy::wrong_self_convention)]
    fn from_frame(&self) -> PyTerrestrialFrame {
        self.inner.from.into()
    }

    /// Target frame for the published forward transform.
    #[getter]
    #[allow(clippy::wrong_self_convention)]
    fn to_frame(&self) -> PyTerrestrialFrame {
        self.inner.to.into()
    }

    /// Parameter reference epoch, decimal year.
    #[getter]
    fn reference_epoch_year(&self) -> f64 {
        self.inner.reference_epoch_year
    }

    /// Parameters at the reference epoch.
    #[getter]
    fn parameters(&self) -> PyHelmertParameters {
        self.inner.parameters.into()
    }

    /// Linear rates of the seven Helmert parameters.
    #[getter]
    fn rates(&self) -> PyHelmertRates {
        self.inner.rates.into()
    }

    /// Published source string for this catalog entry.
    #[getter]
    fn provenance(&self) -> &'static str {
        self.inner.provenance
    }

    /// Evaluate the Helmert parameters at a decimal year.
    fn parameters_at(&self, epoch_year: f64) -> PyResult<PyHelmertParameters> {
        self.inner
            .parameters_at(epoch_year)
            .map(Into::into)
            .map_err(to_frame_catalog_err)
    }

    fn __repr__(&self) -> String {
        format!(
            "HelmertTransform(from_frame={}, to_frame={}, reference_epoch_year={})",
            PyTerrestrialFrame::from(self.inner.from).label(),
            PyTerrestrialFrame::from(self.inner.to).label(),
            self.inner.reference_epoch_year
        )
    }
}

/// Transformed terrestrial position and optional station velocity.
#[pyclass(module = "sidereon._sidereon", name = "TerrestrialState")]
#[derive(Clone, Copy)]
pub struct PyTerrestrialState {
    inner: TerrestrialState,
}

impl From<TerrestrialState> for PyTerrestrialState {
    fn from(inner: TerrestrialState) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyTerrestrialState {
    /// Transformed Cartesian position, metres.
    #[getter]
    fn position_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.position.as_array())
    }

    /// Transformed station velocity, metres per year, or `None`.
    #[getter]
    fn velocity_m_per_year<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        self.inner
            .velocity
            .map(|velocity| np_array(py, &velocity.as_array()))
    }

    fn __repr__(&self) -> String {
        format!(
            "TerrestrialState(position_m={:?}, has_velocity={})",
            self.inner.position.as_array(),
            self.inner.velocity.is_some()
        )
    }
}

/// Return the built-in terrestrial frame Helmert catalog.
#[pyfunction]
fn frame_catalog() -> Vec<PyHelmertTransform> {
    core_catalog::catalog()
        .iter()
        .copied()
        .map(Into::into)
        .collect()
}

/// Return the built-in terrestrial frame Helmert catalog.
#[pyfunction]
fn terrestrial_frame_catalog() -> Vec<PyHelmertTransform> {
    frame_catalog()
}

/// Return the published forward catalog entry for two frames, or `None`.
#[pyfunction]
fn frame_catalog_entry(
    #[pyo3(from_py_with = extract_terrestrial_frame)] from_frame: PyTerrestrialFrame,
    #[pyo3(from_py_with = extract_terrestrial_frame)] to_frame: PyTerrestrialFrame,
) -> Option<PyHelmertTransform> {
    core_catalog::catalog_entry(from_frame.into(), to_frame.into())
        .copied()
        .map(Into::into)
}

/// Propagate a terrestrial station position between decimal-year epochs.
#[pyfunction]
fn frame_catalog_propagate_position<'py>(
    py: Python<'py>,
    position_m: PyReadonlyArray1<'_, f64>,
    velocity_m_per_year: PyReadonlyArray1<'_, f64>,
    from_epoch_year: f64,
    to_epoch_year: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let position = position_from_array("position_m", &position_m)?;
    let velocity = velocity_from_array("velocity_m_per_year", &velocity_m_per_year)?;
    let out = core_catalog::propagate_position(position, velocity, from_epoch_year, to_epoch_year)
        .map_err(to_frame_catalog_err)?;
    Ok(np_array(py, &out.as_array()))
}

/// Transform a Cartesian station position and optional velocity between frames.
#[pyfunction]
#[pyo3(signature = (
    position_m,
    from_frame,
    to_frame,
    epoch_year,
    *,
    velocity_m_per_year = None,
))]
fn frame_catalog_transform(
    position_m: PyReadonlyArray1<'_, f64>,
    #[pyo3(from_py_with = extract_terrestrial_frame)] from_frame: PyTerrestrialFrame,
    #[pyo3(from_py_with = extract_terrestrial_frame)] to_frame: PyTerrestrialFrame,
    epoch_year: f64,
    velocity_m_per_year: Option<PyReadonlyArray1<'_, f64>>,
) -> PyResult<PyTerrestrialState> {
    let position = position_from_array("position_m", &position_m)?;
    let velocity = velocity_m_per_year
        .as_ref()
        .map(|values| velocity_from_array("velocity_m_per_year", values))
        .transpose()?;
    core_catalog::transform(
        position,
        velocity,
        from_frame.into(),
        to_frame.into(),
        epoch_year,
    )
    .map(Into::into)
    .map_err(to_frame_catalog_err)
}

/// Propagate a station to a transform epoch, then transform it between frames.
#[pyfunction]
fn frame_catalog_transform_from_epoch(
    position_m: PyReadonlyArray1<'_, f64>,
    velocity_m_per_year: PyReadonlyArray1<'_, f64>,
    position_epoch_year: f64,
    #[pyo3(from_py_with = extract_terrestrial_frame)] from_frame: PyTerrestrialFrame,
    #[pyo3(from_py_with = extract_terrestrial_frame)] to_frame: PyTerrestrialFrame,
    transform_epoch_year: f64,
) -> PyResult<PyTerrestrialState> {
    let position = position_from_array("position_m", &position_m)?;
    let velocity = velocity_from_array("velocity_m_per_year", &velocity_m_per_year)?;
    core_catalog::transform_from_epoch(
        position,
        velocity,
        position_epoch_year,
        from_frame.into(),
        to_frame.into(),
        transform_epoch_year,
    )
    .map(Into::into)
    .map_err(to_frame_catalog_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTerrestrialFrame>()?;
    m.add_class::<PyHelmertParameters>()?;
    m.add_class::<PyHelmertRates>()?;
    m.add_class::<PyHelmertTransform>()?;
    m.add_class::<PyTerrestrialState>()?;
    m.add_function(wrap_pyfunction!(frame_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(terrestrial_frame_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(frame_catalog_entry, m)?)?;
    m.add_function(wrap_pyfunction!(frame_catalog_propagate_position, m)?)?;
    m.add_function(wrap_pyfunction!(frame_catalog_transform, m)?)?;
    m.add_function(wrap_pyfunction!(frame_catalog_transform_from_epoch, m)?)?;
    Ok(())
}
