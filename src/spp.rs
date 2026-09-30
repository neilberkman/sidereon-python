//! Single-point positioning (SPP) binding.
//!
//! Marshals Pythonic scalars/lists into [`sidereon_core::positioning::SolveInputs`]
//! and returns the reference [`ReceiverSolution`] as a Pythonic object. No
//! modeling: the solve is `sidereon::solve_spp`, unchanged.

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyModule};

use sidereon_core::positioning::{
    residual_rms as core_residual_rms, Corrections, ExactSolveInputs, KlobucharCoeffs, Observation,
    PseudorangeCode, QzssClock, ReceiverSolution, RejectedSat, RejectionReason,
    RinexSppEpochInputs, RinexSppEpochSolution, RinexSppOptions, RinexSppSource, RobustConfig,
    SolveInputs, SolvePolicy, SurfaceMet, TroposphereModel,
};
use sidereon_core::quality::SolutionValidationOptions;
use sidereon_core::GnssSatelliteId;

use crate::events::PyDop;
use crate::exact_time::PyExactEpoch;
use crate::geometry_quality::PyGeometryQuality;
use crate::marshal::{mat3_to_array, option_py_or_default, PyGnssSystem};
use crate::rinex::{PyBroadcastEphemeris, PyObsEpochTime, PyRinexObs, PySignalPolicy};
use crate::{np_array, PySp3};

type PyRejectedSatelliteRow = (String, &'static str, Option<(f64, f64)>);

/// The core `RejectionReason` variant name a rejected satellite reports.
///
/// SPP selection tests a satellite in the order RTKLIB `rescode` does and
/// reports the first reason that applies: `NoEphemeris` (or strict SSR-size
/// refusal), `LowElevation`, `SbasIonoUncovered`, then `IonosphereCarrierUnresolved` (an
/// ionosphere-corrected solve with no carrier frequency for the satellite, a
/// GLONASS satellite with no channel or one outside `-7..=6`).
pub(crate) fn rejection_reason_label(reason: RejectionReason) -> &'static str {
    match reason {
        RejectionReason::NoEphemeris => "NoEphemeris",
        RejectionReason::SsrCorrectionExceedsLimit(_) => "SsrCorrectionExceedsLimit",
        RejectionReason::LowElevation => "LowElevation",
        RejectionReason::SbasWithdrawn => "SbasWithdrawn",
        RejectionReason::SbasIonoUncovered => "SbasIonoUncovered",
        RejectionReason::IonosphereCarrierUnresolved => "IonosphereCarrierUnresolved",
    }
}

/// `(satellite, reason)` rows for rejected satellites, in the core's order.
pub(crate) fn rejected_rows(rejected: &[RejectedSat]) -> Vec<(String, &'static str)> {
    rejected
        .iter()
        .map(|sat| {
            (
                sat.satellite_id.to_string(),
                rejection_reason_label(sat.reason),
            )
        })
        .collect()
}

pub(crate) fn rejected_rows_with_details(rejected: &[RejectedSat]) -> Vec<PyRejectedSatelliteRow> {
    rejected
        .iter()
        .map(|satellite| {
            let size = match satellite.reason {
                RejectionReason::SsrCorrectionExceedsLimit(size) => {
                    Some((size.orbit_m, size.clock_m))
                }
                _ => None,
            };
            (
                satellite.satellite_id.to_string(),
                rejection_reason_label(satellite.reason),
                size,
            )
        })
        .collect()
}

/// One pseudorange observation for an SPP solve.
#[pyclass(module = "sidereon._sidereon", name = "SppObservation")]
#[derive(Clone)]
pub struct PySppObservation {
    satellite_id: GnssSatelliteId,
    token: String,
    pseudorange_m: f64,
}

#[pymethods]
impl PySppObservation {
    /// Create a pseudorange observation keyed by a RINEX/IGS satellite token.
    #[new]
    fn new(satellite_id: String, pseudorange_m: f64) -> PyResult<Self> {
        let parsed = GnssSatelliteId::from_str(&satellite_id).map_err(|_| {
            PyValueError::new_err(format!("invalid satellite token: {satellite_id}"))
        })?;
        Ok(Self {
            satellite_id: parsed,
            token: satellite_id,
            pseudorange_m,
        })
    }

    /// Satellite token, for example `"G01"`.
    #[getter]
    fn satellite_id(&self) -> &str {
        &self.token
    }

    /// Pseudorange in metres.
    #[getter]
    fn pseudorange_m(&self) -> f64 {
        self.pseudorange_m
    }

    fn __repr__(&self) -> String {
        format!(
            "SppObservation(satellite_id={:?}, pseudorange_m={:.3})",
            self.token, self.pseudorange_m
        )
    }
}

impl PySppObservation {
    fn to_core(&self) -> Observation {
        Observation {
            satellite_id: self.satellite_id,
            pseudorange_m: self.pseudorange_m,
        }
    }
}

/// Boolean correction switches for an SPP solve.
#[pyclass(module = "sidereon._sidereon", name = "SppCorrections")]
#[derive(Clone, Copy)]
pub struct PySppCorrections {
    inner: Corrections,
}

#[pymethods]
impl PySppCorrections {
    /// Create SPP correction switches.
    #[new]
    #[pyo3(signature = (ionosphere=false, troposphere=false))]
    fn new(ionosphere: bool, troposphere: bool) -> Self {
        Self {
            inner: Corrections {
                ionosphere,
                troposphere,
            },
        }
    }

    /// Whether to apply the GPS Klobuchar ionosphere correction.
    #[getter]
    fn ionosphere(&self) -> bool {
        self.inner.ionosphere
    }

    /// Whether to apply the Saastamoinen/Niell troposphere correction.
    #[getter]
    fn troposphere(&self) -> bool {
        self.inner.troposphere
    }

    fn __repr__(&self) -> String {
        format!(
            "SppCorrections(ionosphere={}, troposphere={})",
            self.inner.ionosphere, self.inner.troposphere
        )
    }
}

impl Default for PySppCorrections {
    fn default() -> Self {
        Self {
            inner: Corrections {
                ionosphere: false,
                troposphere: false,
            },
        }
    }
}

/// GPS Klobuchar ionosphere coefficients.
#[pyclass(module = "sidereon._sidereon", name = "SppKlobucharCoeffs")]
#[derive(Clone, Copy)]
pub struct PySppKlobucharCoeffs {
    inner: KlobucharCoeffs,
}

#[pymethods]
impl PySppKlobucharCoeffs {
    /// Create Klobuchar alpha/beta coefficient vectors.
    #[new]
    #[pyo3(signature = (alpha=[0.0; 4], beta=[0.0; 4]))]
    fn new(alpha: [f64; 4], beta: [f64; 4]) -> Self {
        Self {
            inner: KlobucharCoeffs { alpha, beta },
        }
    }

    /// Alpha coefficients.
    #[getter]
    fn alpha(&self) -> [f64; 4] {
        self.inner.alpha
    }

    /// Beta coefficients.
    #[getter]
    fn beta(&self) -> [f64; 4] {
        self.inner.beta
    }

    fn __repr__(&self) -> String {
        format!(
            "SppKlobucharCoeffs(alpha={:?}, beta={:?})",
            self.inner.alpha, self.inner.beta
        )
    }
}

impl Default for PySppKlobucharCoeffs {
    fn default() -> Self {
        Self {
            inner: KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
        }
    }
}

/// Surface meteorology for troposphere-corrected SPP.
///
/// The default is the core standard atmosphere
/// ([`sidereon_core::positioning::SurfaceMet::default()`]); the binding holds no copy of
/// those values.
#[pyclass(module = "sidereon._sidereon", name = "SppSurfaceMet")]
#[derive(Clone, Copy, Default)]
pub struct PySppSurfaceMet {
    inner: SurfaceMet,
}

#[pymethods]
impl PySppSurfaceMet {
    /// Create surface meteorology values.
    #[new]
    #[pyo3(signature = (
        pressure_hpa=SurfaceMet::default().pressure_hpa,
        temperature_k=SurfaceMet::default().temperature_k,
        relative_humidity=SurfaceMet::default().relative_humidity,
    ))]
    fn new(pressure_hpa: f64, temperature_k: f64, relative_humidity: f64) -> Self {
        Self {
            inner: SurfaceMet {
                pressure_hpa,
                temperature_k,
                relative_humidity,
            },
        }
    }

    #[getter]
    fn pressure_hpa(&self) -> f64 {
        self.inner.pressure_hpa
    }

    #[getter]
    fn temperature_k(&self) -> f64 {
        self.inner.temperature_k
    }

    #[getter]
    fn relative_humidity(&self) -> f64 {
        self.inner.relative_humidity
    }

    fn __repr__(&self) -> String {
        format!(
            "SppSurfaceMet(pressure_hpa={:.2}, temperature_k={:.2}, relative_humidity={:.3})",
            self.inner.pressure_hpa, self.inner.temperature_k, self.inner.relative_humidity
        )
    }
}

/// Opt-in Huber/IRLS robust reweighting for an SPP solve.
///
/// Pass an instance as `SppConfig(robust=...)` to route the solve through the
/// outer iteratively-reweighted least-squares loop in the core
/// ([`sidereon_core::positioning::RobustConfig`]): a warm start at the static
/// variance weights (bit-identical to the non-robust solve), then resolves that
/// rebuild each weight as `inverse_variance * huber(residual / scale)`, with a
/// floored MAD scale. The loop settles when the position moves less than
/// `outer_tol_m` and the selection holds (status `selection_settled`); a loop
/// that returns to an earlier state without settling stops with
/// `outer_oscillation`, and one that reaches `max_outer` solves with
/// `outer_budget_exhausted`; neither has converged. With no `robust` config the
/// solve is byte-identical to the static path. The defaults match the core
/// `RobustConfig::default()` (textbook `huber_k = 1.345`; `max_outer = 100`, a
/// safety cap rather than a working budget, so the reweighting runs to its
/// settled fixed point).
#[pyclass(module = "sidereon._sidereon", name = "SppRobustConfig")]
#[derive(Clone, Copy, Default)]
pub struct PySppRobustConfig {
    pub(crate) inner: RobustConfig,
}

impl PySppRobustConfig {
    pub(crate) fn inner(&self) -> RobustConfig {
        self.inner
    }
}

#[pymethods]
impl PySppRobustConfig {
    /// Create a Huber/IRLS robust reweighting config. Omitted fields take the
    /// core `RobustConfig` defaults.
    #[new]
    #[pyo3(signature = (
        huber_k=RobustConfig::default().huber_k,
        scale_floor_m=RobustConfig::default().scale_floor_m,
        max_outer=RobustConfig::default().max_outer,
        outer_tol_m=RobustConfig::default().outer_tol_m,
    ))]
    fn new(huber_k: f64, scale_floor_m: f64, max_outer: usize, outer_tol_m: f64) -> Self {
        let mut inner = RobustConfig::default();
        inner.huber_k = huber_k;
        inner.scale_floor_m = scale_floor_m;
        inner.max_outer = max_outer;
        inner.outer_tol_m = outer_tol_m;
        Self { inner }
    }

    /// Huber tuning constant `k`; residuals scaled below this keep full weight.
    #[getter]
    fn huber_k(&self) -> f64 {
        self.inner.huber_k
    }

    /// Floor (metres) on the MAD robust scale.
    #[getter]
    fn scale_floor_m(&self) -> f64 {
        self.inner.scale_floor_m
    }

    /// Maximum total outer solves (the warm start plus reweighted resolves), a
    /// cap the reweighting reaches only when it does not settle.
    #[getter]
    fn max_outer(&self) -> usize {
        self.inner.max_outer
    }

    /// Outer-loop position L2 step tolerance (metres).
    #[getter]
    fn outer_tol_m(&self) -> f64 {
        self.inner.outer_tol_m
    }

    fn __repr__(&self) -> String {
        format!(
            "SppRobustConfig(huber_k={}, scale_floor_m={}, max_outer={}, outer_tol_m={})",
            self.inner.huber_k,
            self.inner.scale_floor_m,
            self.inner.max_outer,
            self.inner.outer_tol_m
        )
    }
}

/// Which pseudorange an SPP or static solve reads, mirroring the core
/// `PseudorangeCode`. The broadcast single-frequency group delay (GPS/QZSS
/// TGD, Galileo BGD, BeiDou TGD1) applies to single-frequency code only, as
/// RTKLIB `prange` applies it to `P1` and none under `IFLC`.
#[pyclass(module = "sidereon._sidereon", name = "PseudorangeCode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyPseudorangeCode {
    /// A single-frequency code (L1 C/A, E1, B1I): the group delay applies.
    SINGLE_FREQUENCY,
    /// The ionosphere-free combination: no group delay applies.
    IONOSPHERE_FREE,
}

#[pyclass(module = "sidereon._sidereon", name = "QzssClock", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyQzssClock {
    GPS,
    SEPARATE,
}

impl From<PyQzssClock> for QzssClock {
    fn from(value: PyQzssClock) -> Self {
        match value {
            PyQzssClock::GPS => Self::Gps,
            PyQzssClock::SEPARATE => Self::Separate,
        }
    }
}

impl From<QzssClock> for PyQzssClock {
    fn from(value: QzssClock) -> Self {
        match value {
            QzssClock::Gps => Self::GPS,
            QzssClock::Separate => Self::SEPARATE,
        }
    }
}

#[pymethods]
impl PyQzssClock {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::GPS => "gps",
            Self::SEPARATE => "separate",
        }
    }

    fn __repr__(&self) -> String {
        format!("QzssClock.{}", self.label().to_ascii_uppercase())
    }
}

#[pyclass(module = "sidereon._sidereon", name = "TroposphereModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyTroposphereModel {
    RTKLIB,
    SAASTAMOINEN_NIELL,
}

impl From<PyTroposphereModel> for TroposphereModel {
    fn from(value: PyTroposphereModel) -> Self {
        match value {
            PyTroposphereModel::RTKLIB => Self::Rtklib,
            PyTroposphereModel::SAASTAMOINEN_NIELL => Self::SaastamoinenNiell,
        }
    }
}

impl From<TroposphereModel> for PyTroposphereModel {
    fn from(value: TroposphereModel) -> Self {
        match value {
            TroposphereModel::Rtklib => Self::RTKLIB,
            TroposphereModel::SaastamoinenNiell => Self::SAASTAMOINEN_NIELL,
        }
    }
}

#[pymethods]
impl PyTroposphereModel {
    #[getter]
    fn label(&self) -> &'static str {
        match self {
            Self::RTKLIB => "rtklib",
            Self::SAASTAMOINEN_NIELL => "saastamoinen_niell",
        }
    }

    fn __repr__(&self) -> String {
        format!("TroposphereModel.{}", self.label().to_ascii_uppercase())
    }
}

impl From<PyPseudorangeCode> for PseudorangeCode {
    fn from(code: PyPseudorangeCode) -> Self {
        match code {
            PyPseudorangeCode::SINGLE_FREQUENCY => PseudorangeCode::SingleFrequency,
            PyPseudorangeCode::IONOSPHERE_FREE => PseudorangeCode::IonosphereFree,
        }
    }
}

impl From<PseudorangeCode> for PyPseudorangeCode {
    fn from(code: PseudorangeCode) -> Self {
        match code {
            PseudorangeCode::SingleFrequency => PyPseudorangeCode::SINGLE_FREQUENCY,
            PseudorangeCode::IonosphereFree => PyPseudorangeCode::IONOSPHERE_FREE,
        }
    }
}

/// Complete typed input bundle for an SPP solve.
#[pyclass(module = "sidereon._sidereon", name = "SppConfig")]
pub struct PySppConfig {
    observations: Vec<Observation>,
    t_rx_j2000_s: f64,
    t_rx_second_of_day_s: f64,
    day_of_year: f64,
    initial_guess: [f64; 4],
    corrections: Corrections,
    klobuchar: KlobucharCoeffs,
    glonass_channels: BTreeMap<u8, i8>,
    met: SurfaceMet,
    with_geodetic: bool,
    robust: Option<RobustConfig>,
    pseudorange_code: PseudorangeCode,
    qzss_clock: QzssClock,
    troposphere_model: TroposphereModel,
}

#[pymethods]
impl PySppConfig {
    /// Create a complete SPP solve configuration.
    #[new]
    #[pyo3(signature = (
        observations,
        t_rx_j2000_s,
        t_rx_second_of_day_s,
        day_of_year,
        initial_guess,
        corrections=None,
        klobuchar=None,
        glonass_channels=None,
        met=None,
        with_geodetic=true,
        robust=None,
        pseudorange_code=PyPseudorangeCode::SINGLE_FREQUENCY,
        qzss_clock=PyQzssClock::GPS,
        troposphere_model=PyTroposphereModel::RTKLIB,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        observations: Vec<Py<PySppObservation>>,
        t_rx_j2000_s: f64,
        t_rx_second_of_day_s: f64,
        day_of_year: f64,
        initial_guess: [f64; 4],
        corrections: Option<Py<PySppCorrections>>,
        klobuchar: Option<Py<PySppKlobucharCoeffs>>,
        glonass_channels: Option<BTreeMap<u8, i8>>,
        met: Option<Py<PySppSurfaceMet>>,
        with_geodetic: bool,
        robust: Option<Py<PySppRobustConfig>>,
        pseudorange_code: PyPseudorangeCode,
        qzss_clock: PyQzssClock,
        troposphere_model: PyTroposphereModel,
    ) -> Self {
        let observations = observations
            .iter()
            .map(|obs| obs.borrow(py).to_core())
            .collect();
        let corrections = option_py_or_default(
            py,
            corrections.as_ref(),
            |value| value.inner,
            || PySppCorrections::default().inner,
        );
        let klobuchar = option_py_or_default(
            py,
            klobuchar.as_ref(),
            |value| value.inner,
            || PySppKlobucharCoeffs::default().inner,
        );
        let met = option_py_or_default(
            py,
            met.as_ref(),
            |value| value.inner,
            || PySppSurfaceMet::default().inner,
        );
        let robust = robust.map(|cfg| cfg.borrow(py).inner);
        Self {
            observations,
            t_rx_j2000_s,
            t_rx_second_of_day_s,
            day_of_year,
            initial_guess,
            corrections,
            klobuchar,
            glonass_channels: glonass_channels.unwrap_or_default(),
            met,
            with_geodetic,
            robust,
            pseudorange_code: pseudorange_code.into(),
            qzss_clock: qzss_clock.into(),
            troposphere_model: troposphere_model.into(),
        }
    }

    /// Which pseudorange the observations carry; the broadcast group delay
    /// applies to single-frequency code only.
    #[getter]
    fn pseudorange_code(&self) -> PyPseudorangeCode {
        self.pseudorange_code.into()
    }

    #[getter]
    fn qzss_clock(&self) -> PyQzssClock {
        self.qzss_clock.into()
    }

    #[getter]
    fn troposphere_model(&self) -> PyTroposphereModel {
        self.troposphere_model.into()
    }

    /// Number of observations in this solve.
    #[getter]
    fn observation_count(&self) -> usize {
        self.observations.len()
    }

    #[getter]
    fn t_rx_j2000_s(&self) -> f64 {
        self.t_rx_j2000_s
    }

    #[getter]
    fn t_rx_second_of_day_s(&self) -> f64 {
        self.t_rx_second_of_day_s
    }

    #[getter]
    fn day_of_year(&self) -> f64 {
        self.day_of_year
    }

    #[getter]
    fn initial_guess(&self) -> [f64; 4] {
        self.initial_guess
    }

    #[getter]
    fn with_geodetic(&self) -> bool {
        self.with_geodetic
    }

    /// The opt-in Huber/IRLS robust reweighting config for this solve, or `None`
    /// when the solve runs the static elevation-weighted path.
    #[getter]
    fn robust(&self) -> Option<PySppRobustConfig> {
        self.robust.map(|inner| PySppRobustConfig { inner })
    }

    /// GLONASS FDMA frequency channels for this solve, as a dict mapping each
    /// GLONASS slot/PRN (int) to its channel number `k` (int). Empty unless
    /// `glonass_channels` was supplied at construction.
    #[getter]
    fn glonass_channels(&self) -> BTreeMap<u8, i8> {
        self.glonass_channels.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SppConfig(observations={}, t_rx_j2000_s={:.3}, with_geodetic={}, qzss_clock={}, troposphere_model={})",
            self.observations.len(),
            self.t_rx_j2000_s,
            self.with_geodetic,
            PyQzssClock::from(self.qzss_clock).label(),
            PyTroposphereModel::from(self.troposphere_model).label()
        )
    }
}

impl PySppConfig {
    /// The `with_geodetic` flag, for reuse by sibling solvers (DGNSS, FDE).
    pub(crate) fn with_geodetic_flag(&self) -> bool {
        self.with_geodetic
    }

    pub(crate) fn to_inputs(&self) -> SolveInputs {
        SolveInputs {
            observations: self.observations.clone(),
            t_rx_j2000_s: self.t_rx_j2000_s,
            t_rx_second_of_day_s: self.t_rx_second_of_day_s,
            day_of_year: self.day_of_year,
            initial_guess: self.initial_guess,
            corrections: self.corrections,
            klobuchar: self.klobuchar,
            beidou_klobuchar: None,
            galileo_nequick: None,
            sbas_iono: None,
            glonass_channels: self.glonass_channels.clone(),
            met: self.met,
            troposphere_model: self.troposphere_model,
            robust: self.robust,
            pseudorange_code: self.pseudorange_code,
            qzss_clock: self.qzss_clock,
        }
    }
}

/// Build the SPP solve policy from the boundary's optional gates: a PDOP ceiling
/// and a cold-start coarse-search seed count. A non-positive `max_pdop` or a
/// zero `coarse_search_seeds` is rejected here rather than silently ignored,
/// matching the Elixir `solve/4` option contract. All other validation gates
/// keep their core defaults.
fn build_policy(
    max_pdop: Option<f64>,
    coarse_search_seeds: Option<usize>,
) -> PyResult<SolvePolicy> {
    if let Some(pdop) = max_pdop {
        if !(pdop.is_finite() && pdop > 0.0) {
            return Err(PyValueError::new_err(format!(
                "max_pdop must be a positive finite number, got {pdop}"
            )));
        }
    }
    if let Some(seeds) = coarse_search_seeds {
        if seeds == 0 {
            return Err(PyValueError::new_err(
                "coarse_search_seeds must be a positive integer",
            ));
        }
    }
    let mut validation = SolutionValidationOptions::default();
    validation.max_pdop = max_pdop;
    Ok(SolvePolicy {
        validation,
        coarse_search_seeds,
    })
}

fn parse_satellite_token(token: &str) -> PyResult<GnssSatelliteId> {
    GnssSatelliteId::from_str(token)
        .map_err(|_| PyValueError::new_err(format!("invalid satellite token: {token}")))
}

#[allow(clippy::too_many_arguments)]
fn rinex_spp_options(
    py: Python<'_>,
    obs: &PyRinexObs,
    signal_policy: Option<&PySignalPolicy>,
    corrections: Option<&PySppCorrections>,
    initial_guess: Option<[f64; 4]>,
    satellites: Option<Vec<String>>,
    met: Option<&PySppSurfaceMet>,
    robust: Option<&PySppRobustConfig>,
    qzss_clock: PyQzssClock,
    troposphere_model: PyTroposphereModel,
) -> PyResult<RinexSppOptions> {
    let mut options = match signal_policy {
        Some(policy) => RinexSppOptions::new(policy.inner()),
        None => RinexSppOptions::default_for(obs.inner())
            .map_err(|err| crate::spp_error_detail::rinex_spp_error(py, err))?,
    };
    if let Some(corrections) = corrections {
        options = options.with_corrections(corrections.inner);
    }
    if let Some(initial_guess) = initial_guess {
        options = options.with_initial_guess(initial_guess);
    }
    if let Some(satellites) = satellites {
        let parsed = satellites
            .iter()
            .map(|sat| parse_satellite_token(sat))
            .collect::<PyResult<BTreeSet<_>>>()?;
        options = options.with_satellites(parsed);
    }
    if let Some(met) = met {
        options = options.with_surface_met(met.inner);
    }
    if let Some(robust) = robust {
        options = options.with_robust(Some(robust.inner()));
    }
    options = options
        .with_qzss_clock(qzss_clock.into())
        .with_troposphere_model(troposphere_model.into());
    Ok(options)
}

/// Options for assembling RINEX OBS epochs into SPP inputs.
#[pyclass(module = "sidereon._sidereon", name = "RinexSppOptions")]
pub struct PyRinexSppOptions {
    inner: RinexSppOptions,
}

#[pymethods]
impl PyRinexSppOptions {
    /// Build RINEX-to-SPP assembly options.
    ///
    /// If `signal_policy` is omitted, the default policy for the observation
    /// file's RINEX version is used. If `initial_guess` is omitted, the RINEX
    /// header `APPROX POSITION XYZ` value seeds the receiver position.
    #[new]
    #[pyo3(signature = (
        obs,
        signal_policy=None,
        corrections=None,
        initial_guess=None,
        satellites=None,
        met=None,
        robust=None,
        qzss_clock=PyQzssClock::GPS,
        troposphere_model=PyTroposphereModel::RTKLIB,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        obs: &PyRinexObs,
        signal_policy: Option<&PySignalPolicy>,
        corrections: Option<&PySppCorrections>,
        initial_guess: Option<[f64; 4]>,
        satellites: Option<Vec<String>>,
        met: Option<&PySppSurfaceMet>,
        robust: Option<&PySppRobustConfig>,
        qzss_clock: PyQzssClock,
        troposphere_model: PyTroposphereModel,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: rinex_spp_options(
                py,
                obs,
                signal_policy,
                corrections,
                initial_guess,
                satellites,
                met,
                robust,
                qzss_clock,
                troposphere_model,
            )?,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexSppOptions(satellites={}, qzss_clock={:?}, troposphere_model={:?})",
            self.inner.satellites.as_ref().map_or(0, BTreeSet::len),
            self.inner.qzss_clock,
            self.inner.troposphere_model
        )
    }

    #[getter]
    fn qzss_clock(&self) -> PyQzssClock {
        self.inner.qzss_clock.into()
    }

    #[getter]
    fn troposphere_model(&self) -> PyTroposphereModel {
        self.inner.troposphere_model.into()
    }
}

impl PyRinexSppOptions {
    fn inner(&self) -> RinexSppOptions {
        self.inner.clone()
    }
}

/// One assembled RINEX OBS epoch and its SPP input bundle.
#[pyclass(module = "sidereon._sidereon", name = "RinexSppEpochInputs")]
pub struct PyRinexSppEpochInputs {
    inner: RinexSppEpochInputs,
}

impl From<RinexSppEpochInputs> for PyRinexSppEpochInputs {
    fn from(inner: RinexSppEpochInputs) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexSppEpochInputs {
    /// Index in the source RINEX observation epoch list.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    /// Civil epoch exactly as it appears in the RINEX OBS file.
    #[getter]
    fn epoch(&self) -> PyObsEpochTime {
        self.inner.epoch.into()
    }

    /// Number of pseudorange observations assembled for this epoch.
    #[getter]
    fn observation_count(&self) -> usize {
        self.inner.inputs.observations.len()
    }

    /// Satellite tokens assembled for this epoch.
    #[getter]
    fn satellites(&self) -> Vec<String> {
        self.inner
            .inputs
            .observations
            .iter()
            .map(|obs| obs.satellite_id.to_string())
            .collect()
    }

    /// Pseudorange observations assembled for this epoch.
    #[getter]
    fn observations(&self) -> Vec<PySppObservation> {
        self.inner
            .inputs
            .observations
            .iter()
            .map(|obs| PySppObservation {
                satellite_id: obs.satellite_id,
                token: obs.satellite_id.to_string(),
                pseudorange_m: obs.pseudorange_m,
            })
            .collect()
    }

    #[getter]
    fn t_rx_j2000_s(&self) -> f64 {
        self.inner.inputs.t_rx_j2000_s
    }

    #[getter]
    fn t_rx_second_of_day_s(&self) -> f64 {
        self.inner.inputs.t_rx_second_of_day_s
    }

    #[getter]
    fn day_of_year(&self) -> f64 {
        self.inner.inputs.day_of_year
    }

    #[getter]
    fn qzss_clock(&self) -> PyQzssClock {
        self.inner.inputs.qzss_clock.into()
    }

    #[getter]
    fn troposphere_model(&self) -> PyTroposphereModel {
        self.inner.inputs.troposphere_model.into()
    }

    #[getter]
    fn exact_receive_epoch(&self) -> Option<PyExactEpoch> {
        self.inner.exact_solve_inputs().map(|inputs| PyExactEpoch {
            inner: inputs.receive_epoch,
        })
    }

    #[getter]
    fn initial_guess(&self) -> [f64; 4] {
        self.inner.inputs.initial_guess
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexSppEpochInputs(epoch_index={}, observations={})",
            self.inner.epoch_index,
            self.inner.inputs.observations.len()
        )
    }
}

/// One RINEX OBS epoch paired with its SPP solve result.
#[pyclass(module = "sidereon._sidereon", name = "RinexSppEpochSolution")]
pub struct PyRinexSppEpochSolution {
    inner: RinexSppEpochSolution,
}

impl From<RinexSppEpochSolution> for PyRinexSppEpochSolution {
    fn from(inner: RinexSppEpochSolution) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRinexSppEpochSolution {
    /// Index in the source RINEX observation epoch list.
    #[getter]
    fn epoch_index(&self) -> usize {
        self.inner.epoch_index
    }

    /// Civil epoch exactly as it appears in the RINEX OBS file.
    #[getter]
    fn epoch(&self) -> PyObsEpochTime {
        self.inner.epoch.into()
    }

    /// `True` when this epoch produced a receiver solution.
    #[getter]
    fn solved(&self) -> bool {
        self.inner.solution.is_ok()
    }

    /// Receiver solution for this epoch, or `None` if the epoch did not solve.
    #[getter]
    fn solution(&self) -> Option<PySppSolution> {
        self.inner
            .solution
            .as_ref()
            .ok()
            .cloned()
            .map(PySppSolution::from_solution)
    }

    /// Per-epoch solve error text, or `None` when the epoch solved.
    #[getter]
    fn error(&self) -> Option<String> {
        self.inner.solution.as_ref().err().map(ToString::to_string)
    }

    /// Owned typed detail for a failed solve, or `None` after a successful solve.
    #[getter]
    fn error_detail(&self, py: Python<'_>) -> PyResult<Option<Py<PyDict>>> {
        self.inner
            .solution
            .as_ref()
            .err()
            .map(|error| {
                crate::spp_error_detail::policy_detail(py, error, Some(self.inner.epoch_index), "")
                    .map(Bound::unbind)
            })
            .transpose()
    }

    fn __repr__(&self) -> String {
        format!(
            "RinexSppEpochSolution(epoch_index={}, solved={})",
            self.inner.epoch_index,
            self.inner.solution.is_ok()
        )
    }
}

/// The result of an SPP solve.
///
/// `position` is the ECEF receiver position as a numpy `float64` array of shape
/// `(3,)`, `[x, y, z]` metres; `geodetic` is `(lat_rad, lon_rad, height_m)` or
/// `None`.
#[pyclass(module = "sidereon._sidereon", name = "SppSolution")]
pub struct PySppSolution {
    inner: ReceiverSolution,
}

impl PySppSolution {
    /// Wrap a core receiver solution, for sibling entry points (broadcast SPP,
    /// precise-with-broadcast fallback) that return the same solution shape.
    pub(crate) fn from_solution(inner: ReceiverSolution) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &ReceiverSolution {
        &self.inner
    }
}

#[pymethods]
impl PySppSolution {
    /// ECEF position as a numpy array `[x_m, y_m, z_m]`.
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        np_array(
            py,
            &[
                self.inner.position.x_m,
                self.inner.position.y_m,
                self.inner.position.z_m,
            ],
        )
    }

    #[getter]
    fn x_m(&self) -> f64 {
        self.inner.position.x_m
    }

    #[getter]
    fn y_m(&self) -> f64 {
        self.inner.position.y_m
    }

    #[getter]
    fn z_m(&self) -> f64 {
        self.inner.position.z_m
    }

    /// Receiver clock bias in seconds.
    #[getter]
    fn rx_clock_s(&self) -> f64 {
        self.inner.rx_clock_s
    }

    /// Receiver clock drift in seconds per second, when solved from Doppler rows.
    #[getter]
    fn rx_clock_drift_s_s(&self) -> Option<f64> {
        self.inner.rx_clock_drift_s_s
    }

    /// ECEF position covariance in square metres.
    #[getter]
    fn position_covariance_ecef_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.ecef_m2)
    }

    /// Local ENU position covariance in square metres.
    #[getter]
    fn position_covariance_enu_m2<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        mat3_to_array(py, &self.inner.position_covariance.enu_m2)
    }

    /// `(lat_rad, lon_rad, height_m)` if the solve was asked for geodetic.
    #[getter]
    fn geodetic(&self) -> Option<(f64, f64, f64)> {
        self.inner
            .geodetic
            .map(|g| (g.lat_rad, g.lon_rad, g.height_m))
    }

    /// Satellite tokens used in the accepted solution, ascending.
    #[getter]
    fn used_sats(&self) -> Vec<String> {
        self.inner
            .used_sats
            .iter()
            .map(|sat| sat.to_string())
            .collect()
    }

    /// Post-fit residuals in metres, index-aligned to `used_sats`.
    #[getter]
    fn residuals_m(&self) -> Vec<f64> {
        self.inner.residuals_m.clone()
    }

    /// The pseudorange error variance of each used satellite, square metres,
    /// index-aligned to `used_sats`: the RTKLIB `rescode` variance the solve
    /// weighted the satellite by (ephemeris, code-bias, ionosphere, troposphere
    /// and code-error terms), taken at the selection the solution reports.
    /// Residual chi-square fault detection standardizes `residuals_m` by these,
    /// as RTKLIB `valsol` forms `sum (v / sigma)^2`.
    #[getter]
    fn pseudorange_variances_m2(&self) -> Vec<f64> {
        self.inner.pseudorange_variances_m2.clone()
    }

    /// The weight each used satellite carried in the reported solve, inverse
    /// square metres, index-aligned to `used_sats`: `1 /
    /// pseudorange_variances_m2` on the least-squares path, and that inverse
    /// variance times the final Huber factor on the robust path.
    /// `position_covariance_ecef_m2` is `(H^T W H)^-1` with these weights.
    #[getter]
    fn weights(&self) -> Vec<f64> {
        self.inner.weights.clone()
    }

    /// Satellites excluded from the solve as `(satellite, reason)` rows, the
    /// reason being the core `RejectionReason` variant name: `NoEphemeris`,
    /// `SsrCorrectionExceedsLimit`, `LowElevation`, `SbasWithdrawn`, `SbasIonoUncovered` or
    /// `IonosphereCarrierUnresolved`. An ionosphere-corrected solve excludes a
    /// satellite it has no carrier frequency for (a GLONASS satellite with no
    /// channel, or one outside `-7..=6`) with the last reason and solves the
    /// rest of the epoch.
    #[getter]
    fn rejected_sats(&self) -> Vec<(String, &'static str)> {
        rejected_rows(&self.inner.rejected_sats)
    }

    #[getter]
    fn rejected_sats_with_details(&self) -> Vec<PyRejectedSatelliteRow> {
        rejected_rows_with_details(&self.inner.rejected_sats)
    }

    /// Absolute per-constellation receiver clock as `(system, clock_s)` pairs,
    /// one entry per GNSS in the solve in ascending system order. The first
    /// entry's value equals `rx_clock_s` (the reference clock); an inter-system
    /// bias is any other system's clock minus that reference. One entry for a
    /// single-system solve.
    #[getter]
    fn system_clocks_s(&self) -> Vec<(PyGnssSystem, f64)> {
        self.inner
            .system_clocks_s
            .iter()
            .map(|&(sys, clk)| (sys.into(), clk))
            .collect()
    }

    /// Per-constellation time (clock) DOP as `(system, tdop)` pairs, one entry
    /// per GNSS in the solve in ascending system order. The first entry's value
    /// equals the geometry's scalar `tdop` (the reference clock). One entry for
    /// a single-system solve; empty only when the geometry is rank-deficient.
    #[getter]
    fn system_tdops(&self) -> Vec<(PyGnssSystem, f64)> {
        self.inner
            .system_tdops
            .iter()
            .map(|&(sys, tdop)| (sys.into(), tdop))
            .collect()
    }

    /// Dilution-of-precision diagnostics for the accepted geometry, or `None`
    /// when the geometry was rank-deficient (so no covariance was formed).
    #[getter]
    fn dop(&self) -> Option<PyDop> {
        self.inner.dop.clone().map(PyDop::from)
    }

    /// Geometry observability and covariance-validation diagnostics.
    #[getter]
    fn geometry_quality(&self) -> PyGeometryQuality {
        self.inner.geometry_quality.into()
    }

    /// How the whole solve ended: `selection_settled` for a solve whose last
    /// least-squares step at a held selection fell below 1e-4 m (or a robust
    /// solve whose position and selection settled), `outer_budget_exhausted`
    /// for a robust solve whose reweighting budget ran out first,
    /// `outer_oscillation` for a robust reweighting that returned to an earlier
    /// state without settling. A robust solve whose last trust-region solve
    /// spent its evaluations reports that solve's own status.
    #[getter]
    fn status(&self) -> &'static str {
        crate::static_positioning::status_label(self.inner.metadata.status)
    }

    /// Whether the whole solve converged.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.metadata.converged
    }

    /// Trust-region iterations of every solve plus one per least-squares
    /// step.
    #[getter]
    fn iterations(&self) -> usize {
        self.inner.metadata.iterations
    }

    /// Solution degrees of freedom, `used_count - (3 + systems)`.
    #[getter]
    fn redundancy(&self) -> isize {
        self.inner.metadata.redundancy
    }

    /// Whether residual-based RAIM can test the accepted solution.
    #[getter]
    fn raim_checkable(&self) -> bool {
        self.inner.metadata.raim_checkable
    }

    /// The UT1 departure a permissive UT1 policy accepted, `before_coverage`
    /// or `after_coverage`; `None` when every UT1 read was inside the table.
    #[getter]
    fn ut1_degraded(&self) -> Option<&'static str> {
        self.inner
            .metadata
            .ut1_degraded
            .map(crate::degrade_reason_label)
    }

    fn __repr__(&self) -> String {
        format!(
            "SppSolution(position=[{:.3}, {:.3}, {:.3}], rx_clock_s={:.6e}, used_sats={})",
            self.inner.position.x_m,
            self.inner.position.y_m,
            self.inner.position.z_m,
            self.inner.rx_clock_s,
            self.inner.used_sats.len()
        )
    }
}

#[pyfunction]
fn spp_residual_rms_m(residuals_m: Vec<f64>) -> f64 {
    core_residual_rms(&residuals_m)
}

/// Run single-point positioning.
///
/// `max_pdop` optionally caps the accepted geometry's PDOP (a fix that is
/// rank-deficient or exceeds the ceiling is refused). `coarse_search_seeds`
/// optionally widens the cold-start convergence basin: the solve runs once from
/// each of that many deterministic golden-spiral near-surface seeds (plus the
/// config's `initial_guess`) and selects the best redundant converged fix, so no
/// good position prior is needed. Both delegate to the core `SolvePolicy`; raise
/// `ValueError` on a non-positive value.
#[pyfunction]
#[pyo3(signature = (sp3, config, *, max_pdop=None, coarse_search_seeds=None))]
fn solve_spp(
    py: Python<'_>,
    sp3: &PySp3,
    config: &PySppConfig,
    max_pdop: Option<f64>,
    coarse_search_seeds: Option<usize>,
) -> PyResult<PySppSolution> {
    let inputs = config.to_inputs();
    let policy = build_policy(max_pdop, coarse_search_seeds)?;
    let inner = sidereon::solve_spp(&sp3.inner, &inputs, config.with_geodetic, policy)
        .map_err(|err| crate::spp_error_detail::facade_spp_error(py, err, "", None))?;
    Ok(PySppSolution { inner })
}

#[pyfunction]
#[pyo3(signature = (sp3, config, receive_epoch, *, max_pdop=None, coarse_search_seeds=None))]
fn solve_spp_exact(
    py: Python<'_>,
    sp3: &PySp3,
    config: &PySppConfig,
    receive_epoch: &PyExactEpoch,
    max_pdop: Option<f64>,
    coarse_search_seeds: Option<usize>,
) -> PyResult<PySppSolution> {
    let exact_inputs = ExactSolveInputs {
        inputs: config.to_inputs(),
        receive_epoch: receive_epoch.inner,
    };
    let policy = build_policy(max_pdop, coarse_search_seeds)?;
    let inner = sidereon_core::positioning::solve_with_exact_epoch_and_policy(
        &sp3.inner,
        &exact_inputs,
        config.with_geodetic,
        policy,
    )
    .map_err(|err| {
        crate::spp_error_detail::facade_spp_error(py, sidereon::Error::Spp(err), "", None)
    })?;
    Ok(PySppSolution { inner })
}

/// Solve a batch of independent SPP epochs against a shared SP3 ephemeris,
/// releasing the GIL for the whole compute.
///
/// `configs` is a sequence of `SppConfig`, each a self-contained receive epoch
/// (its own pseudoranges, receive time, and initial guess). The configs are
/// marshalled into core inputs with the GIL held, then the independent per-epoch
/// solves run inside `Python::allow_threads` -- by default across a rayon thread
/// pool (`parallel=True`), so a stream of fixes saturates all cores with no
/// interpreter lock held. Each epoch is solved by the same serial kernel as
/// `solve_spp` and rayon's indexed collect preserves order, so the result is
/// bit-identical to the serial path (`parallel=False`) element by element.
/// `with_geodetic` is shared by the batch (taken from the first config); an empty
/// batch returns an empty list. Raises `SidereonError` (naming the epoch index)
/// if an epoch fails to solve.
#[pyfunction]
#[pyo3(signature = (sp3, configs, *, parallel=true, max_pdop=None, coarse_search_seeds=None))]
fn solve_spp_batch(
    py: Python<'_>,
    sp3: &PySp3,
    configs: Vec<PyRef<'_, PySppConfig>>,
    parallel: bool,
    max_pdop: Option<f64>,
    coarse_search_seeds: Option<usize>,
) -> PyResult<Vec<PySppSolution>> {
    let with_geodetic = configs.first().map(|c| c.with_geodetic).unwrap_or(false);
    let epochs: Vec<SolveInputs> = configs.iter().map(|c| c.to_inputs()).collect();
    let policy = build_policy(max_pdop, coarse_search_seeds)?;
    let eph = &sp3.inner;

    // GIL released for the whole batch: the closure owns plain Rust data (the
    // marshalled epochs) and borrows only the immutable ephemeris.
    let results = py.allow_threads(move || {
        if parallel {
            sidereon::solve_spp_batch(eph, &epochs, with_geodetic, policy)
        } else {
            sidereon::solve_spp_batch_serial(eph, &epochs, with_geodetic, policy)
        }
    });

    results
        .into_iter()
        .enumerate()
        .map(|(idx, result)| {
            result
                .map(|inner| PySppSolution { inner })
                .map_err(|error| {
                    crate::spp_error_detail::facade_spp_error(
                        py,
                        error,
                        &format!("epoch {idx}: "),
                        Some(idx),
                    )
                })
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (sp3, configs, receive_epochs, *, max_pdop=None, coarse_search_seeds=None))]
fn solve_spp_batch_exact(
    py: Python<'_>,
    sp3: &PySp3,
    configs: Vec<PyRef<'_, PySppConfig>>,
    receive_epochs: Vec<PyRef<'_, PyExactEpoch>>,
    max_pdop: Option<f64>,
    coarse_search_seeds: Option<usize>,
) -> PyResult<Vec<PySppSolution>> {
    if configs.len() != receive_epochs.len() {
        return Err(PyValueError::new_err(
            "configs and receive_epochs must have equal lengths",
        ));
    }
    let with_geodetic = configs
        .first()
        .map(|config| config.with_geodetic)
        .unwrap_or(false);
    let exact_inputs: Vec<ExactSolveInputs> = configs
        .iter()
        .zip(receive_epochs.iter())
        .map(|(config, epoch)| ExactSolveInputs {
            inputs: config.to_inputs(),
            receive_epoch: epoch.inner,
        })
        .collect();
    let policy = build_policy(max_pdop, coarse_search_seeds)?;
    let ephemeris = &sp3.inner;
    let results = py.allow_threads(move || {
        exact_inputs
            .iter()
            .map(|inputs| {
                sidereon_core::positioning::solve_with_exact_epoch_and_policy(
                    ephemeris,
                    inputs,
                    with_geodetic,
                    policy,
                )
            })
            .collect::<Vec<_>>()
    });
    results
        .into_iter()
        .enumerate()
        .map(|(epoch_index, result)| {
            result
                .map(|inner| PySppSolution { inner })
                .map_err(|error| {
                    crate::spp_error_detail::facade_spp_error(
                        py,
                        sidereon::Error::Spp(error),
                        &format!("epoch {epoch_index}: "),
                        Some(epoch_index),
                    )
                })
        })
        .collect()
}

fn default_or_supplied_rinex_options(
    py: Python<'_>,
    obs: &PyRinexObs,
    options: Option<&PyRinexSppOptions>,
) -> PyResult<RinexSppOptions> {
    match options {
        Some(options) => Ok(options.inner()),
        None => RinexSppOptions::default_for(obs.inner())
            .map_err(|err| crate::spp_error_detail::rinex_spp_error(py, err)),
    }
}

fn with_source<T>(
    source: &Bound<'_, PyAny>,
    broadcast_context: Option<&PyBroadcastEphemeris>,
    precise: impl FnOnce(&RinexSppSource<'_, sidereon_core::ephemeris::Sp3>) -> PyResult<T>,
    broadcast: impl FnOnce(&sidereon_core::ephemeris::BroadcastEphemeris) -> PyResult<T>,
) -> PyResult<T> {
    if let Ok(sp3) = source.extract::<PyRef<'_, PySp3>>() {
        let delegated = match broadcast_context {
            Some(broadcast) => RinexSppSource::with_broadcast_context(&sp3.inner, &broadcast.inner),
            None => RinexSppSource::new(&sp3.inner),
        };
        return precise(&delegated);
    }
    if let Ok(broadcast_source) = source.extract::<PyRef<'_, PyBroadcastEphemeris>>() {
        return broadcast(&broadcast_source.inner);
    }
    Err(PyValueError::new_err(
        "source must be a Sp3 or BroadcastEphemeris",
    ))
}

/// Assemble RINEX OBS epochs into SPP solve inputs.
#[pyfunction]
#[pyo3(signature = (source, obs, options=None, *, broadcast_context=None))]
fn spp_inputs_from_rinex_obs(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    obs: &PyRinexObs,
    options: Option<&PyRinexSppOptions>,
    broadcast_context: Option<&PyBroadcastEphemeris>,
) -> PyResult<Vec<PyRinexSppEpochInputs>> {
    let options = default_or_supplied_rinex_options(py, obs, options)?;
    with_source(
        source,
        broadcast_context,
        |source| {
            sidereon_core::positioning::spp_inputs_from_rinex_obs(obs.inner(), source, &options)
                .map(|epochs| epochs.into_iter().map(Into::into).collect())
                .map_err(|err| crate::spp_error_detail::rinex_spp_error(py, err))
        },
        |source| {
            sidereon_core::positioning::spp_inputs_from_rinex_obs(obs.inner(), source, &options)
                .map(|epochs| epochs.into_iter().map(Into::into).collect())
                .map_err(|err| crate::spp_error_detail::rinex_spp_error(py, err))
        },
    )
}

/// Assemble RINEX OBS epochs and solve each epoch serially with SPP.
#[pyfunction]
#[pyo3(signature = (
    source,
    obs,
    options=None,
    *,
    with_geodetic=true,
    max_pdop=None,
    coarse_search_seeds=None,
    broadcast_context=None,
))]
#[allow(clippy::too_many_arguments)]
fn solve_spp_from_rinex_obs(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    obs: &PyRinexObs,
    options: Option<&PyRinexSppOptions>,
    with_geodetic: bool,
    max_pdop: Option<f64>,
    coarse_search_seeds: Option<usize>,
    broadcast_context: Option<&PyBroadcastEphemeris>,
) -> PyResult<Vec<PyRinexSppEpochSolution>> {
    let options = default_or_supplied_rinex_options(py, obs, options)?;
    let policy = build_policy(max_pdop, coarse_search_seeds)?;
    with_source(
        source,
        broadcast_context,
        |source| {
            sidereon_core::positioning::solve_spp_from_rinex_obs_exact_with_policy(
                source,
                obs.inner(),
                &options,
                with_geodetic,
                policy,
            )
            .map(|epochs| epochs.into_iter().map(Into::into).collect())
            .map_err(|err| crate::spp_error_detail::rinex_spp_error(py, err))
        },
        |source| {
            sidereon_core::positioning::solve_spp_from_rinex_obs_exact_with_policy(
                source,
                obs.inner(),
                &options,
                with_geodetic,
                policy,
            )
            .map(|epochs| epochs.into_iter().map(Into::into).collect())
            .map_err(|err| crate::spp_error_detail::rinex_spp_error(py, err))
        },
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySppSolution>()?;
    m.add_class::<PyPseudorangeCode>()?;
    m.add_class::<PyQzssClock>()?;
    m.add_class::<PyTroposphereModel>()?;
    m.add_class::<PyRinexSppOptions>()?;
    m.add_class::<PyRinexSppEpochInputs>()?;
    m.add_class::<PyRinexSppEpochSolution>()?;
    m.add_function(wrap_pyfunction!(spp_residual_rms_m, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp_exact, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp_batch, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp_batch_exact, m)?)?;
    m.add_function(wrap_pyfunction!(spp_inputs_from_rinex_obs, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp_from_rinex_obs, m)?)?;
    m.add_class::<PySppObservation>()?;
    m.add_class::<PySppCorrections>()?;
    m.add_class::<PySppKlobucharCoeffs>()?;
    m.add_class::<PySppSurfaceMet>()?;
    m.add_class::<PySppRobustConfig>()?;
    m.add_class::<PySppConfig>()?;
    Ok(())
}
