//! Single-point positioning (SPP) binding.
//!
//! Marshals Pythonic scalars/lists into [`sidereon_core::positioning::SolveInputs`]
//! and returns the reference [`ReceiverSolution`] as a Pythonic object. No
//! modeling: the solve is `sidereon::solve_spp`, unchanged.

use std::collections::BTreeMap;
use std::str::FromStr;

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use sidereon_core::positioning::{
    residual_rms as core_residual_rms, Corrections, KlobucharCoeffs, Observation, ReceiverSolution,
    RobustConfig, SolveInputs, SolvePolicy, SurfaceMet,
};
use sidereon_core::quality::SolutionValidationOptions;
use sidereon_core::GnssSatelliteId;

use crate::events::PyDop;
use crate::marshal::{option_py_or_default, PyGnssSystem};
use crate::{np_array, to_solve_err, PySp3, SolveError};

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
/// ([`sidereon_core::spp::SurfaceMet::default()`]); the binding holds no copy of
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
/// elevation weights (bit-identical to the non-robust solve), then resolves that
/// rebuild each weight as `base_weight * huber(residual / scale)`. With no
/// `robust` config the solve is byte-identical to the static path. The defaults
/// match the core `RobustConfig::default()` (textbook `huber_k = 1.345`).
#[pyclass(module = "sidereon._sidereon", name = "SppRobustConfig")]
#[derive(Clone, Copy, Default)]
pub struct PySppRobustConfig {
    inner: RobustConfig,
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
        Self {
            inner: RobustConfig {
                huber_k,
                scale_floor_m,
                max_outer,
                outer_tol_m,
            },
        }
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

    /// Maximum total outer solves (the warm start plus reweighted resolves).
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
        }
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
            "SppConfig(observations={}, t_rx_j2000_s={:.3}, with_geodetic={})",
            self.observations.len(),
            self.t_rx_j2000_s,
            self.with_geodetic
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
            robust: self.robust,
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
    Ok(SolvePolicy {
        validation: SolutionValidationOptions {
            max_pdop,
            ..Default::default()
        },
        coarse_search_seeds,
    })
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
    sp3: &PySp3,
    config: &PySppConfig,
    max_pdop: Option<f64>,
    coarse_search_seeds: Option<usize>,
) -> PyResult<PySppSolution> {
    let inputs = config.to_inputs();
    let policy = build_policy(max_pdop, coarse_search_seeds)?;
    let inner = sidereon::solve_spp(&sp3.inner, &inputs, config.with_geodetic, policy)
        .map_err(to_solve_err)?;
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
                .map_err(|e| SolveError::new_err(format!("epoch {idx}: {e}")))
        })
        .collect()
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySppSolution>()?;
    m.add_function(wrap_pyfunction!(spp_residual_rms_m, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp, m)?)?;
    m.add_function(wrap_pyfunction!(solve_spp_batch, m)?)?;
    m.add_class::<PySppObservation>()?;
    m.add_class::<PySppCorrections>()?;
    m.add_class::<PySppKlobucharCoeffs>()?;
    m.add_class::<PySppSurfaceMet>()?;
    m.add_class::<PySppRobustConfig>()?;
    m.add_class::<PySppConfig>()?;
    Ok(())
}
