//! Deterministic scenario simulator binding.
//!
//! Scenarios enter as Python mappings or JSON text and deserialize directly into
//! [`sidereon_core::scenario::Scenario`]. Outputs expose the core synthetic
//! observable arrays and the ground-truth term ledger without local modeling.

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyByteArray, PyBytes, PyDict, PyModule};

use sidereon_core::ephemeris::EphemerisSource;
use sidereon_core::observables::ObservableEphemerisSource;
use sidereon_core::scenario::{
    ionex_content_fingerprint as core_ionex_content_fingerprint,
    scenario_source_transcript_fingerprint as core_scenario_source_transcript_fingerprint,
    simulate_scenario as core_simulate_scenario,
    simulate_scenario_with_media as core_simulate_scenario_with_media,
    simulate_scenario_with_source_and_media as core_simulate_scenario_with_source_and_media,
    DeclaredIonexSource, DeclaredScenarioSource, Scenario, ScenarioExternalProduct,
    ScenarioExternalProductKind, ScenarioMediaSources, SyntheticObservableArrays,
    SyntheticObservationSet, SyntheticTermArrays, SCENARIO_ENGINE_VERSION, SCENARIO_SCHEMA_VERSION,
};

use crate::marshal::rows3_to_array;
use crate::np_array;

fn scenario_parse_err(err: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn scenario_core_err(py: Python<'_>, err: sidereon_core::scenario::ScenarioError) -> PyErr {
    let message = err.to_string();
    let py_err = crate::ScenarioError::new_err(message);
    let detail = (PyScenarioErrorDetail { inner: err }).into_pyobject(py);
    let detail = match detail {
        Ok(detail) => detail,
        Err(error) => return error.into(),
    };
    if let Err(error) = py_err.value(py).setattr("detail", detail) {
        return error;
    }
    py_err
}

fn scenario_from_py(source: &Bound<'_, PyAny>) -> PyResult<Scenario> {
    if let Ok(text) = source.extract::<String>() {
        return serde_json::from_str(&text).map_err(scenario_parse_err);
    }
    if let Ok(bytes) = source.downcast::<PyBytes>() {
        return serde_json::from_slice(bytes.as_bytes()).map_err(scenario_parse_err);
    }
    if let Ok(bytes) = source.downcast::<PyByteArray>() {
        // SAFETY: the bytearray is copied by serde before Python can mutate it.
        return serde_json::from_slice(unsafe { bytes.as_bytes() }).map_err(scenario_parse_err);
    }
    pythonize::depythonize(source).map_err(scenario_parse_err)
}

fn usize_array<'py>(py: Python<'py>, values: &[usize]) -> Bound<'py, PyArray1<usize>> {
    PyArray1::from_vec(py, values.to_vec())
}

fn string_satellites(values: &[sidereon_core::GnssSatelliteId]) -> Vec<String> {
    values.iter().map(ToString::to_string).collect()
}

fn scenario_degrade_reason(reason: sidereon_core::astro::time::DegradeReason) -> &'static str {
    use sidereon_core::astro::time::DegradeReason;
    match reason {
        DegradeReason::BeforeCoverage => "before_coverage",
        DegradeReason::AfterCoverage => "after_coverage",
    }
}

/// Structured payload preserving a core `ScenarioError` variant.
#[pyclass(module = "sidereon._sidereon", name = "ScenarioErrorDetail")]
pub struct PyScenarioErrorDetail {
    inner: sidereon_core::scenario::ScenarioError,
}

#[pymethods]
impl PyScenarioErrorDetail {
    #[getter]
    fn kind(&self) -> &'static str {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::InvalidInput { .. } => "invalid_input",
            E::ExternalSourceRequired => "external_source_required",
            E::ExternalSourceMismatch { .. } => "external_source_mismatch",
            E::ExternalIonosphereRequired => "external_ionosphere_required",
            E::Ionosphere(_) => "ionosphere",
            E::NoEphemeris { .. } => "no_ephemeris",
            E::Ut1OutsideCoverage { .. } => "ut1_outside_coverage",
            E::Observable(_) => "observable",
            E::Frame(_) => "frame",
        }
    }

    #[getter]
    fn field(&self) -> Option<&'static str> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::InvalidInput { field, .. } | E::ExternalSourceMismatch { field, .. } => Some(*field),
            _ => None,
        }
    }

    #[getter]
    fn reason(&self) -> Option<&'static str> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::InvalidInput { reason, .. } => Some(*reason),
            _ => None,
        }
    }

    #[getter]
    fn expected(&self) -> Option<&str> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::ExternalSourceMismatch { expected, .. } => Some(expected),
            _ => None,
        }
    }

    #[getter]
    fn actual(&self) -> Option<&str> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::ExternalSourceMismatch { actual, .. } => Some(actual),
            _ => None,
        }
    }

    #[getter]
    fn satellite(&self) -> Option<String> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::NoEphemeris { satellite } | E::Ut1OutsideCoverage { satellite, .. } => {
                Some(satellite.to_string())
            }
            _ => None,
        }
    }

    #[getter]
    fn ut1_reason(&self) -> Option<&'static str> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::Ut1OutsideCoverage { reason, .. } => Some(scenario_degrade_reason(*reason)),
            _ => None,
        }
    }

    #[getter]
    fn nested_kind(&self) -> Option<&'static str> {
        use sidereon_core::observables::ObservablesError as O;
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::Observable(O::InvalidInput { .. }) => Some("invalid_input"),
            E::Observable(O::NoEphemeris) => Some("no_ephemeris"),
            E::Observable(O::Ephemeris(_)) => Some("ephemeris"),
            E::Observable(O::Media(_)) => Some("media"),
            _ => None,
        }
    }

    #[getter]
    fn nested_field(&self) -> Option<&'static str> {
        use sidereon_core::observables::ObservablesError as O;
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::Observable(O::InvalidInput { field, .. }) => Some(*field),
            _ => None,
        }
    }

    #[getter]
    fn nested_reason(&self) -> Option<&'static str> {
        use sidereon_core::observables::{ObservablesError as O, ObservablesInputErrorKind as K};
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::Observable(O::InvalidInput { kind, .. }) => Some(match kind {
                K::NonFinite => "non_finite",
                K::NotPositive => "not_positive",
                K::Negative => "negative",
                K::OutOfRange => "out_of_range",
                K::Missing => "missing",
                K::FloatParse => "float_parse",
                K::IntParse => "int_parse",
                K::InvalidCivilDate => "invalid_civil_date",
                K::InvalidCivilTime => "invalid_civil_time",
            }),
            _ => None,
        }
    }

    /// Exact nested observable error, including the underlying core error payload.
    #[getter]
    fn nested_observable_error(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::Observable(error) => Ok(Some(crate::core_error_detail::observables_error_detail(
                py, error,
            )?)),
            _ => Ok(None),
        }
    }

    #[getter]
    fn nested_detail(&self) -> Option<String> {
        use sidereon_core::scenario::ScenarioError as E;
        match &self.inner {
            E::Ionosphere(value) | E::Frame(value) => Some(value.clone()),
            E::Observable(value) => Some(value.to_string()),
            _ => None,
        }
    }
}

/// Caller-declared identity for an external scenario product.
#[pyclass(module = "sidereon._sidereon", name = "ScenarioExternalProduct")]
#[derive(Clone)]
pub struct PyScenarioExternalProduct {
    inner: ScenarioExternalProduct,
}

#[pymethods]
impl PyScenarioExternalProduct {
    #[new]
    fn new(kind: &str, product_id: String, content_digest: String) -> PyResult<Self> {
        let kind = match kind {
            "sp3" => ScenarioExternalProductKind::Sp3,
            "broadcast" => ScenarioExternalProductKind::Broadcast,
            "tle" => ScenarioExternalProductKind::Tle,
            "ionex" => ScenarioExternalProductKind::Ionex,
            _ => return Err(PyValueError::new_err("unknown external product kind")),
        };
        Ok(Self {
            inner: ScenarioExternalProduct {
                kind,
                product_id,
                content_digest,
            },
        })
    }

    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            ScenarioExternalProductKind::Sp3 => "sp3",
            ScenarioExternalProductKind::Broadcast => "broadcast",
            ScenarioExternalProductKind::Tle => "tle",
            ScenarioExternalProductKind::Ionex => "ionex",
        }
    }

    #[getter]
    fn product_id(&self) -> &str {
        &self.inner.product_id
    }

    #[getter]
    fn content_digest(&self) -> &str {
        &self.inner.content_digest
    }
}

fn scenario_media<'a>(
    ionex: Option<&'a crate::ionex::PyIonex>,
    identity: Option<&'a PyScenarioExternalProduct>,
) -> PyResult<ScenarioMediaSources<'a>> {
    match (ionex, identity) {
        (None, None) => Ok(ScenarioMediaSources::default()),
        (Some(ionex), Some(identity)) => Ok(ScenarioMediaSources {
            ionex: Some(DeclaredIonexSource::new(&ionex.inner, &identity.inner)),
        }),
        (Some(_), None) => Err(PyValueError::new_err(
            "ionex_identity is required when an IONEX product is supplied",
        )),
        (None, Some(_)) => Err(PyValueError::new_err(
            "ionex_identity requires an IONEX product",
        )),
    }
}

enum ExternalScenarioOperation {
    Fingerprint,
    Simulate,
    SimulateWithMedia,
}

enum ExternalScenarioResult {
    Fingerprint(String),
    Simulation(PySyntheticObservationSet),
}

fn run_with_external_source<E>(
    py: Python<'_>,
    scenario: &Scenario,
    source: &E,
    identity: &PyScenarioExternalProduct,
    media: &ScenarioMediaSources<'_>,
    operation: ExternalScenarioOperation,
) -> PyResult<ExternalScenarioResult>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    let declared = DeclaredScenarioSource::new(source, identity.inner.clone());
    match operation {
        ExternalScenarioOperation::Fingerprint => {
            core_scenario_source_transcript_fingerprint(scenario, &declared, media)
                .map(ExternalScenarioResult::Fingerprint)
                .map_err(|err| scenario_core_err(py, err))
        }
        ExternalScenarioOperation::Simulate => {
            sidereon_core::scenario::simulate_scenario_with_source(scenario, &declared)
                .map(PySyntheticObservationSet::from)
                .map(ExternalScenarioResult::Simulation)
                .map_err(|err| scenario_core_err(py, err))
        }
        ExternalScenarioOperation::SimulateWithMedia => {
            core_simulate_scenario_with_source_and_media(scenario, &declared, media)
                .map(PySyntheticObservationSet::from)
                .map(ExternalScenarioResult::Simulation)
                .map_err(|err| scenario_core_err(py, err))
        }
    }
}

fn dispatch_external_source(
    scenario: &Scenario,
    source: &Bound<'_, PyAny>,
    identity: &PyScenarioExternalProduct,
    media: &ScenarioMediaSources<'_>,
    operation: ExternalScenarioOperation,
) -> PyResult<ExternalScenarioResult> {
    let py = source.py();
    if let Ok(source) = source.extract::<PyRef<'_, crate::ephemeris::PySp3>>() {
        return run_with_external_source(py, scenario, &source.inner, identity, media, operation);
    }
    if let Ok(source) = source.extract::<PyRef<'_, crate::rinex::PyBroadcastEphemeris>>() {
        return run_with_external_source(py, scenario, &source.inner, identity, media, operation);
    }
    if let Ok(source) =
        source.extract::<PyRef<'_, crate::ephemeris::PyPreciseEphemerisInterpolant>>()
    {
        return run_with_external_source(py, scenario, &source.inner, identity, media, operation);
    }
    if let Ok(source) = source.extract::<PyRef<'_, crate::sbas_ssr::PySsrCorrectedEphemeris>>() {
        let broadcast = source.broadcast.try_borrow(py)?;
        let store = source.store.try_borrow(py)?;
        let inner = source.source(&broadcast, &store);
        return run_with_external_source(py, scenario, &inner, identity, media, operation);
    }
    if let Ok(source) = source.extract::<PyRef<'_, crate::sbas_ssr::PySbasCorrectedEphemeris>>() {
        let broadcast = source.broadcast.try_borrow(py)?;
        let store = source.store.try_borrow(py)?;
        let inner = sidereon_core::sbas::SbasCorrectedEphemeris::new(
            &broadcast.inner,
            &store.inner,
            source.geo,
        )
        .with_mode(source.mode);
        return run_with_external_source(py, scenario, &inner, identity, media, operation);
    }
    Err(PyTypeError::new_err(
        "source must be Sp3, BroadcastEphemeris, PreciseEphemerisInterpolant, SsrCorrectedEphemeris, or SbasCorrectedEphemeris",
    ))
}

/// Contiguous synthetic observable arrays.
#[pyclass(module = "sidereon._sidereon", name = "SyntheticObservableArrays")]
#[derive(Clone)]
pub struct PySyntheticObservableArrays {
    inner: SyntheticObservableArrays,
}

impl From<SyntheticObservableArrays> for PySyntheticObservableArrays {
    fn from(inner: SyntheticObservableArrays) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySyntheticObservableArrays {
    /// Start index of each epoch in every per-observation array.
    #[getter]
    fn epoch_offsets<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        usize_array(py, &self.inner.epoch_offsets)
    }

    /// Epoch index for each observation.
    #[getter]
    fn epoch_index<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<usize>> {
        usize_array(py, &self.inner.epoch_index)
    }

    /// Satellite id for each observation.
    #[getter]
    fn satellite_id(&self) -> Vec<String> {
        string_satellites(&self.inner.satellite_id)
    }

    /// Code observable label for each observation.
    #[getter]
    fn code_observable(&self) -> Vec<String> {
        self.inner.code_observable.clone()
    }

    /// Carrier phase observable label for each observation.
    #[getter]
    fn phase_observable(&self) -> Vec<String> {
        self.inner.phase_observable.clone()
    }

    /// Doppler observable label for each observation.
    #[getter]
    fn doppler_observable(&self) -> Vec<String> {
        self.inner.doppler_observable.clone()
    }

    /// Carrier frequency in hertz for each observation.
    #[getter]
    fn carrier_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_hz)
    }

    /// Synthetic code pseudorange in metres.
    #[getter]
    fn pseudorange_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.pseudorange_m)
    }

    /// Synthetic carrier phase in cycles.
    #[getter]
    fn carrier_phase_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_cycles)
    }

    /// Synthetic Doppler shift in hertz.
    #[getter]
    fn doppler_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_hz)
    }

    /// Number of observation rows.
    fn __len__(&self) -> usize {
        self.inner.pseudorange_m.len()
    }
}

/// Per-observation ground-truth term arrays.
#[pyclass(module = "sidereon._sidereon", name = "SyntheticTermArrays")]
#[derive(Clone)]
pub struct PySyntheticTermArrays {
    inner: SyntheticTermArrays,
}

impl From<SyntheticTermArrays> for PySyntheticTermArrays {
    fn from(inner: SyntheticTermArrays) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySyntheticTermArrays {
    /// Geometric range in metres.
    #[getter]
    fn geometric_range_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.geometric_range_m)
    }

    /// Nominal ephemeris satellite-clock contribution in metres.
    #[getter]
    fn satellite_clock_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.satellite_clock_m)
    }

    /// Receiver-clock contribution in metres.
    #[getter]
    fn receiver_clock_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.receiver_clock_m)
    }

    /// Injected satellite-clock contribution in metres.
    #[getter]
    fn satellite_clock_error_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.satellite_clock_error_m)
    }

    /// Ionospheric code delay in metres.
    #[getter]
    fn ionosphere_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.ionosphere_m)
    }

    /// Tropospheric delay in metres.
    #[getter]
    fn troposphere_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.troposphere_m)
    }

    /// Thermal code noise in metres.
    #[getter]
    fn thermal_noise_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.thermal_noise_m)
    }

    /// Specular code multipath in metres.
    #[getter]
    fn multipath_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.multipath_m)
    }

    /// Quantization contribution in metres.
    #[getter]
    fn quantization_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.quantization_m)
    }

    /// Carrier geometric range contribution in cycles.
    #[getter]
    fn carrier_phase_geometric_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_geometric_cycles)
    }

    /// Carrier receiver-clock contribution in cycles.
    #[getter]
    fn carrier_phase_receiver_clock_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_receiver_clock_cycles)
    }

    /// Carrier nominal satellite-clock contribution in cycles.
    #[getter]
    fn carrier_phase_satellite_clock_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_satellite_clock_cycles)
    }

    /// Carrier injected satellite-clock contribution in cycles.
    #[getter]
    fn carrier_phase_satellite_clock_error_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_satellite_clock_error_cycles)
    }

    /// Carrier ionosphere contribution in cycles.
    #[getter]
    fn carrier_phase_ionosphere_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_ionosphere_cycles)
    }

    /// Carrier troposphere contribution in cycles.
    #[getter]
    fn carrier_phase_troposphere_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_troposphere_cycles)
    }

    /// Carrier thermal-noise contribution in cycles.
    #[getter]
    fn carrier_phase_thermal_noise_cycles<'py>(
        &self,
        py: Python<'py>,
    ) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_thermal_noise_cycles)
    }

    /// Constant carrier-phase ambiguity contribution in cycles.
    #[getter]
    fn carrier_phase_bias_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_bias_cycles)
    }

    /// Carrier quantization contribution in cycles.
    #[getter]
    fn carrier_phase_quantization_cycles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.carrier_phase_quantization_cycles)
    }

    /// Doppler contribution from satellite line-of-sight motion in hertz.
    #[getter]
    fn doppler_satellite_motion_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_satellite_motion_hz)
    }

    /// Doppler contribution from receiver line-of-sight motion in hertz.
    #[getter]
    fn doppler_receiver_motion_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_receiver_motion_hz)
    }

    /// Doppler contribution from nominal ephemeris satellite-clock rate.
    #[getter]
    fn doppler_satellite_clock_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_satellite_clock_hz)
    }

    /// Doppler contribution from receiver-clock rate.
    #[getter]
    fn doppler_receiver_clock_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_receiver_clock_hz)
    }

    /// Doppler contribution from injected satellite-clock rate.
    #[getter]
    fn doppler_satellite_clock_error_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_satellite_clock_error_hz)
    }

    /// Doppler thermal-noise contribution in hertz.
    #[getter]
    fn doppler_thermal_noise_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_thermal_noise_hz)
    }

    /// Doppler quantization contribution in hertz.
    #[getter]
    fn doppler_quantization_hz<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(py, &self.inner.doppler_quantization_hz)
    }

    /// Sum the pseudorange terms for one observation.
    fn pseudorange_sum_m(&self, index: usize) -> Option<f64> {
        self.inner.pseudorange_sum_m(index)
    }

    /// Sum the carrier-phase terms for one observation.
    fn carrier_phase_sum_cycles(&self, index: usize) -> Option<f64> {
        self.inner.carrier_phase_sum_cycles(index)
    }

    /// Sum the Doppler terms for one observation.
    fn doppler_sum_hz(&self, index: usize) -> Option<f64> {
        self.inner.doppler_sum_hz(index)
    }
}

/// Complete synthetic observation output.
#[pyclass(module = "sidereon._sidereon", name = "SyntheticObservationSet")]
#[derive(Clone)]
pub struct PySyntheticObservationSet {
    inner: SyntheticObservationSet,
}

impl From<SyntheticObservationSet> for PySyntheticObservationSet {
    fn from(inner: SyntheticObservationSet) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PySyntheticObservationSet {
    /// Scenario schema version used to produce this output.
    #[getter]
    fn schema_version(&self) -> u32 {
        self.inner.schema_version
    }

    /// Engine version used to produce this output.
    #[getter]
    fn engine_version(&self) -> String {
        self.inner.engine_version.clone()
    }

    /// Seed used to produce this output.
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }

    /// Contiguous observable arrays.
    #[getter]
    fn observations(&self) -> PySyntheticObservableArrays {
        self.inner.observations.clone().into()
    }

    /// Per-observation term decomposition.
    #[getter]
    fn truth_terms(&self) -> PySyntheticTermArrays {
        self.inner.truth_terms.clone().into()
    }

    /// Receiver epoch seconds since J2000.
    #[getter]
    fn receiver_t_rx_j2000_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.t_rx_j2000_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Receiver ECEF position rows in metres.
    #[getter]
    fn receiver_position_ecef_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.position_ecef_m)
            .collect::<Vec<_>>();
        rows3_to_array(py, &rows)
    }

    /// Receiver ECEF velocity rows in metres per second.
    #[getter]
    fn receiver_velocity_ecef_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let rows = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.velocity_ecef_m_s)
            .collect::<Vec<_>>();
        rows3_to_array(py, &rows)
    }

    /// Receiver clock contribution in metres.
    #[getter]
    fn receiver_clock_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.clock_m)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Receiver clock range-rate contribution in metres per second.
    #[getter]
    fn receiver_clock_rate_m_s<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values = self
            .inner
            .receiver_truth
            .iter()
            .map(|row| row.clock_rate_m_s)
            .collect::<Vec<_>>();
        np_array(py, &values)
    }

    /// Number of synthetic observations.
    fn observation_count(&self) -> usize {
        self.inner.observation_count()
    }

    /// Deterministic FNV-1a fingerprint over output bits.
    fn determinism_fingerprint(&self) -> u64 {
        self.inner.determinism_fingerprint()
    }

    /// Serialize this output to deterministic JSON bytes.
    fn as_json_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = serde_json::to_vec(&self.inner).map_err(scenario_parse_err)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Serialize this output to deterministic JSON text.
    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.inner).map_err(scenario_parse_err)
    }

    /// Serialize the synthetic observations to RINEX OBS text.
    fn to_rinex_string(&self, py: Python<'_>) -> PyResult<String> {
        self.inner
            .to_rinex_string()
            .map_err(|err| crate::rinex::to_obs_write_err(py, err))
    }

    /// Build SPP observations for one epoch from the pseudorange arrays.
    fn spp_observations_for_epoch(&self, epoch_index: usize) -> Vec<(String, f64)> {
        self.inner
            .spp_observations_for_epoch(epoch_index)
            .into_iter()
            .map(|row| (row.satellite_id.to_string(), row.pseudorange_m))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.observation_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "SyntheticObservationSet(observation_count={}, seed={})",
            self.inner.observation_count(),
            self.inner.seed
        )
    }
}

/// Accepted scenario schema version.
#[pyfunction]
fn scenario_schema_version() -> u32 {
    SCENARIO_SCHEMA_VERSION
}

/// Scenario engine version string used in determinism fingerprints.
#[pyfunction]
fn scenario_engine_version() -> &'static str {
    SCENARIO_ENGINE_VERSION
}

/// Validate and canonicalize a scenario mapping or JSON document to JSON text.
#[pyfunction]
fn scenario_to_json(scenario: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = scenario.py();
    let scenario = scenario_from_py(scenario)?;
    scenario
        .validate()
        .map_err(|err| scenario_core_err(py, err))?;
    serde_json::to_string(&scenario).map_err(scenario_parse_err)
}

/// Simulate a deterministic synthetic-Keplerian scenario.
#[pyfunction]
fn simulate_scenario(scenario: &Bound<'_, PyAny>) -> PyResult<PySyntheticObservationSet> {
    let py = scenario.py();
    let scenario = scenario_from_py(scenario)?;
    core_simulate_scenario(&scenario)
        .map(Into::into)
        .map_err(|err| scenario_core_err(py, err))
}

/// Simulate a synthetic scenario with explicitly declared external IONEX media.
#[pyfunction]
#[pyo3(signature = (scenario, ionex=None, ionex_identity=None))]
fn simulate_scenario_with_media(
    scenario: &Bound<'_, PyAny>,
    ionex: Option<&crate::ionex::PyIonex>,
    ionex_identity: Option<&PyScenarioExternalProduct>,
) -> PyResult<PySyntheticObservationSet> {
    let py = scenario.py();
    let scenario = scenario_from_py(scenario)?;
    let media = scenario_media(ionex, ionex_identity)?;
    core_simulate_scenario_with_media(&scenario, &media)
        .map(Into::into)
        .map_err(|err| scenario_core_err(py, err))
}

/// Return the source-transcript fingerprint for an external-product scenario.
/// Pass the same declared product ID and kind as the scenario; the digest may be
/// a placeholder while computing the transcript fingerprint.
#[pyfunction]
#[pyo3(signature = (scenario, source, source_identity, ionex=None, ionex_identity=None))]
fn scenario_source_transcript_fingerprint(
    scenario: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    source_identity: &PyScenarioExternalProduct,
    ionex: Option<&crate::ionex::PyIonex>,
    ionex_identity: Option<&PyScenarioExternalProduct>,
) -> PyResult<String> {
    let scenario = scenario_from_py(scenario)?;
    let media = scenario_media(ionex, ionex_identity)?;
    match dispatch_external_source(
        &scenario,
        source,
        source_identity,
        &media,
        ExternalScenarioOperation::Fingerprint,
    )? {
        ExternalScenarioResult::Fingerprint(value) => Ok(value),
        ExternalScenarioResult::Simulation(_) => unreachable!("fingerprint operation result"),
    }
}

/// Simulate an external-product scenario against a declared loaded source.
/// Supported sources are `Sp3`, `PreciseEphemerisInterpolant`,
/// `SsrCorrectedEphemeris`, and `SbasCorrectedEphemeris`.
#[pyfunction]
#[pyo3(signature = (scenario, source, source_identity, ionex=None, ionex_identity=None))]
fn simulate_scenario_with_source_and_media(
    scenario: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    source_identity: &PyScenarioExternalProduct,
    ionex: Option<&crate::ionex::PyIonex>,
    ionex_identity: Option<&PyScenarioExternalProduct>,
) -> PyResult<PySyntheticObservationSet> {
    let scenario = scenario_from_py(scenario)?;
    let media = scenario_media(ionex, ionex_identity)?;
    match dispatch_external_source(
        &scenario,
        source,
        source_identity,
        &media,
        ExternalScenarioOperation::SimulateWithMedia,
    )? {
        ExternalScenarioResult::Simulation(value) => Ok(value),
        ExternalScenarioResult::Fingerprint(_) => unreachable!("simulation operation result"),
    }
}

/// Simulate an external-product scenario with a declared ephemeris source and
/// no external media products.
#[pyfunction]
fn simulate_scenario_with_source(
    scenario: &Bound<'_, PyAny>,
    source: &Bound<'_, PyAny>,
    source_identity: &PyScenarioExternalProduct,
) -> PyResult<PySyntheticObservationSet> {
    let scenario = scenario_from_py(scenario)?;
    let media = ScenarioMediaSources::default();
    match dispatch_external_source(
        &scenario,
        source,
        source_identity,
        &media,
        ExternalScenarioOperation::Simulate,
    )? {
        ExternalScenarioResult::Simulation(value) => Ok(value),
        ExternalScenarioResult::Fingerprint(_) => unreachable!("simulation operation result"),
    }
}

/// Return the core content fingerprint of an IONEX product's serialized bytes.
#[pyfunction]
fn ionex_content_fingerprint(py: Python<'_>, ionex: &crate::ionex::PyIonex) -> PyResult<String> {
    core_ionex_content_fingerprint(&ionex.inner).map_err(|err| scenario_core_err(py, err))
}

/// Simulate a scenario and return deterministic JSON bytes for its output.
#[pyfunction]
fn simulate_scenario_bytes<'py>(
    py: Python<'py>,
    scenario: &Bound<'_, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let set = simulate_scenario(scenario)?;
    set.as_json_bytes(py)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyScenarioExternalProduct>()?;
    m.add_class::<PyScenarioErrorDetail>()?;
    m.add_class::<PySyntheticObservableArrays>()?;
    m.add_class::<PySyntheticTermArrays>()?;
    m.add_class::<PySyntheticObservationSet>()?;
    m.add_function(wrap_pyfunction!(scenario_schema_version, m)?)?;
    m.add_function(wrap_pyfunction!(scenario_engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(scenario_to_json, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_scenario, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_scenario_with_media, m)?)?;
    m.add_function(wrap_pyfunction!(scenario_source_transcript_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(
        simulate_scenario_with_source_and_media,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(simulate_scenario_with_source, m)?)?;
    m.add_function(wrap_pyfunction!(ionex_content_fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_scenario_bytes, m)?)?;
    Ok(())
}

#[cfg(test)]
mod scenario_error_detail_tests {
    use super::{scenario_degrade_reason, PyScenarioErrorDetail};
    use pyo3::prelude::*;
    use sidereon_core::observables::{ObservablesError, ObservablesInputErrorKind};
    use sidereon_core::scenario::ScenarioError;

    #[test]
    fn ut1_reason_uses_stable_python_tokens() {
        use sidereon_core::astro::time::DegradeReason;
        assert_eq!(
            scenario_degrade_reason(DegradeReason::BeforeCoverage),
            "before_coverage"
        );
        assert_eq!(
            scenario_degrade_reason(DegradeReason::AfterCoverage),
            "after_coverage"
        );
    }

    #[test]
    fn nested_observable_details_keep_reason_and_recursive_core_payload() {
        let invalid = PyScenarioErrorDetail {
            inner: ScenarioError::Observable(ObservablesError::InvalidInput {
                field: "carrier_hz",
                kind: ObservablesInputErrorKind::NotPositive,
            }),
        };
        assert_eq!(invalid.nested_field(), Some("carrier_hz"));
        assert_eq!(invalid.nested_reason(), Some("not_positive"));

        Python::with_gil(|py| {
            let nested = invalid.nested_observable_error(py).unwrap().unwrap();
            let nested = nested.bind(py);
            assert_eq!(
                nested
                    .get_item("kind")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "invalid_input"
            );
            assert_eq!(
                nested
                    .get_item("field")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "carrier_hz"
            );
            assert_eq!(
                nested
                    .get_item("reason")
                    .unwrap()
                    .unwrap()
                    .extract::<String>()
                    .unwrap(),
                "not_positive"
            );
            for (error, expected_family) in [
                (
                    ObservablesError::Ephemeris(sidereon_core::Error::EpochOutOfRange),
                    "ephemeris",
                ),
                (
                    ObservablesError::Media(sidereon_core::Error::EpochOutOfRange),
                    "media",
                ),
            ] {
                let detail = PyScenarioErrorDetail {
                    inner: ScenarioError::Observable(error),
                };
                let nested = detail.nested_observable_error(py).unwrap().unwrap();
                let nested = nested.bind(py);
                assert_eq!(
                    nested
                        .get_item("kind")
                        .unwrap()
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    expected_family
                );
                let cause = nested.get_item("cause").unwrap().unwrap();
                assert_eq!(
                    cause
                        .get_item("family")
                        .unwrap()
                        .extract::<String>()
                        .unwrap(),
                    "CoreError"
                );
                assert_eq!(
                    cause.get_item("kind").unwrap().extract::<String>().unwrap(),
                    "epoch_out_of_range"
                );
            }
        });
    }
}
