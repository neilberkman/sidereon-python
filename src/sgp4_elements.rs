//! SGP4 and TLE element value bindings.

use pyo3::prelude::*;
use pyo3::types::PyModule;

/// Parsed fixed-column TLE elements.
#[pyclass(module = "sidereon._sidereon", name = "TleElements")]
#[derive(Clone)]
pub struct PyTleElements {
    pub(crate) inner: sidereon_core::astro::tle::TleElements,
}
#[pymethods]
impl PyTleElements {
    /// Construct fixed-column fields for conversion through the core.
    #[new]
    #[pyo3(signature = (catalog_number, classification, international_designator, epoch_year, epoch_day_of_year, mean_motion_dot, mean_motion_double_dot, mean_motion_double_dot_text, bstar, bstar_text, ephemeris_type, elset_number, inclination_deg, raan_deg, eccentricity, arg_perigee_deg, mean_anomaly_deg, mean_motion, rev_number))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        catalog_number: String,
        classification: String,
        international_designator: String,
        epoch_year: i32,
        epoch_day_of_year: f64,
        mean_motion_dot: f64,
        mean_motion_double_dot: f64,
        mean_motion_double_dot_text: Option<String>,
        bstar: f64,
        bstar_text: Option<String>,
        ephemeris_type: Option<i32>,
        elset_number: Option<i32>,
        inclination_deg: f64,
        raan_deg: f64,
        eccentricity: f64,
        arg_perigee_deg: f64,
        mean_anomaly_deg: f64,
        mean_motion: f64,
        rev_number: Option<i32>,
    ) -> Self {
        Self {
            inner: sidereon_core::astro::tle::TleElements {
                catalog_number,
                classification,
                international_designator,
                epoch_year,
                epoch_day_of_year,
                mean_motion_dot,
                mean_motion_double_dot,
                mean_motion_double_dot_text,
                bstar,
                bstar_text,
                ephemeris_type,
                elset_number,
                inclination_deg,
                raan_deg,
                eccentricity,
                arg_perigee_deg,
                mean_anomaly_deg,
                mean_motion,
                rev_number,
            },
        }
    }
    /// Catalog identifier as stated by the source.
    #[getter]
    fn catalog_number(&self) -> String {
        self.inner.catalog_number.clone()
    }
    /// TLE classification character.
    #[getter]
    fn classification(&self) -> String {
        self.inner.classification.clone()
    }
    /// Launch international designator from the TLE.
    #[getter]
    fn international_designator(&self) -> String {
        self.inner.international_designator.clone()
    }
    /// Full calendar year of the TLE epoch.
    #[getter]
    fn epoch_year(&self) -> i32 {
        self.inner.epoch_year
    }
    /// One-based fractional day of the epoch year.
    #[getter]
    fn epoch_day_of_year(&self) -> f64 {
        self.inner.epoch_day_of_year
    }
    /// TLE first mean-motion derivative field in revolutions per day squared.
    #[getter]
    fn mean_motion_dot(&self) -> f64 {
        self.inner.mean_motion_dot
    }
    /// TLE second mean-motion derivative field in revolutions per day cubed.
    #[getter]
    fn mean_motion_double_dot(&self) -> f64 {
        self.inner.mean_motion_double_dot
    }
    /// Exact eight-character second-derivative field, when parsed from text.
    #[getter]
    fn mean_motion_double_dot_text(&self) -> Option<String> {
        self.inner.mean_motion_double_dot_text.clone()
    }
    /// SGP4 drag coefficient in the dimensionless TLE convention.
    #[getter]
    fn bstar(&self) -> f64 {
        self.inner.bstar
    }
    /// Exact eight-character B-star field, when parsed from text.
    #[getter]
    fn bstar_text(&self) -> Option<String> {
        self.inner.bstar_text.clone()
    }
    /// TLE ephemeris type, or None if blank.
    #[getter]
    fn ephemeris_type(&self) -> Option<i32> {
        self.inner.ephemeris_type
    }
    /// TLE element-set number, or None if blank.
    #[getter]
    fn elset_number(&self) -> Option<i32> {
        self.inner.elset_number
    }
    /// Orbital inclination in degrees.
    #[getter]
    fn inclination_deg(&self) -> f64 {
        self.inner.inclination_deg
    }
    /// Right ascension of the ascending node in degrees.
    #[getter]
    fn raan_deg(&self) -> f64 {
        self.inner.raan_deg
    }
    /// Dimensionless orbital eccentricity.
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.eccentricity
    }
    /// Argument of perigee in degrees.
    #[getter]
    fn arg_perigee_deg(&self) -> f64 {
        self.inner.arg_perigee_deg
    }
    /// Mean anomaly in degrees.
    #[getter]
    fn mean_anomaly_deg(&self) -> f64 {
        self.inner.mean_anomaly_deg
    }
    /// Mean motion in revolutions per day.
    #[getter]
    fn mean_motion(&self) -> f64 {
        self.inner.mean_motion
    }
    /// Revolution number at the TLE epoch, or None if blank.
    #[getter]
    fn rev_number(&self) -> Option<i32> {
        self.inner.rev_number
    }
    /// Convert these fields to the core SGP4 element set.
    fn to_element_set(&self) -> PyResult<PyElementSet> {
        self.inner
            .to_element_set()
            .map(|inner| PyElementSet { inner })
            .map_err(crate::tle_fit_error_detail::tle_parse_error)
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

/// The format-independent elements supplied to SGP4.
#[pyclass(module = "sidereon._sidereon", name = "ElementSet")]
#[derive(Clone)]
pub struct PyElementSet {
    pub(crate) inner: sidereon_core::astro::sgp4::ElementSet,
}
#[pymethods]
impl PyElementSet {
    /// Epoch as a split Julian date in days.
    #[getter]
    fn epoch(&self) -> (f64, f64) {
        (self.inner.epoch.0, self.inner.epoch.1)
    }
    /// SGP4 drag coefficient in the dimensionless TLE convention.
    #[getter]
    fn bstar(&self) -> f64 {
        self.inner.bstar
    }
    /// TLE first mean-motion derivative field in revolutions per day squared.
    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> {
        self.inner.mean_motion_dot
    }
    /// TLE second mean-motion derivative field in revolutions per day cubed.
    #[getter]
    fn mean_motion_double_dot(&self) -> Option<f64> {
        self.inner.mean_motion_double_dot
    }
    /// Dimensionless orbital eccentricity.
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.eccentricity
    }
    /// Argument of perigee in degrees.
    #[getter]
    fn argument_of_perigee_deg(&self) -> f64 {
        self.inner.argument_of_perigee_deg
    }
    /// Orbital inclination in degrees.
    #[getter]
    fn inclination_deg(&self) -> f64 {
        self.inner.inclination_deg
    }
    /// Mean anomaly in degrees.
    #[getter]
    fn mean_anomaly_deg(&self) -> f64 {
        self.inner.mean_anomaly_deg
    }
    /// Mean motion in revolutions per day.
    #[getter]
    fn mean_motion_rev_per_day(&self) -> f64 {
        self.inner.mean_motion_rev_per_day
    }
    /// Right ascension of the ascending node in degrees.
    #[getter]
    fn right_ascension_deg(&self) -> f64 {
        self.inner.right_ascension_deg
    }
    /// Catalog identifier as stated by the source.
    #[getter]
    fn catalog_number(&self) -> Option<u32> {
        self.inner.catalog_number
    }
    /// OMM initialization epoch in days since 1949-12-31 UTC, when present.
    #[getter]
    fn omm_epoch_days(&self) -> Option<f64> {
        self.inner.omm_epoch_days
    }
    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTleElements>()?;
    m.add_class::<PyElementSet>()?;
    Ok(())
}
