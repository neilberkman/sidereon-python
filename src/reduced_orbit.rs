//! Compact mean-element (reduced-orbit) fitting and evaluation.
//!
//! Thin INTERFACE over the `sidereon_core::orbit` public API. It decodes
//! calendar epochs, ECEF samples, the time scale, and the model/frame selectors,
//! calls [`fit_with_model`](sidereon_core::orbit::fit_with_model),
//! [`position`](sidereon_core::orbit::position),
//! [`position_velocity`](sidereon_core::orbit::position_velocity), and
//! [`drift`](sidereon_core::orbit::drift), and encodes the results back. No
//! fitting, element, or frame math lives here.

use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::Bound;

use sidereon_core::astro::time::TimeScale;
use sidereon_core::orbit::{
    drift as core_drift, drift_reduced_orbit_source as core_source_drift,
    fit_piecewise as core_fit_piecewise, fit_reduced_orbit_source as core_source_fit,
    fit_with_model, piecewise_drift as core_piecewise_drift,
    piecewise_position as core_piecewise_position,
    piecewise_position_velocity as core_piecewise_position_velocity, position as core_position,
    position_velocity as core_position_velocity, select_piecewise_segment as core_select_segment,
    CalendarEpoch, EcefSample, Frame, Model, PiecewiseOrbit, PiecewiseOrbitError, PiecewiseSegment,
    ReducedOrbit, ReducedOrbitSource, ReducedOrbitSourceDriftOptions, ReducedOrbitSourceFitOptions,
    ReducedOrbitSourceSampling,
};
use sidereon_core::GnssSatelliteId;

use crate::frames::PyTimeScale;
use crate::{to_solve_err, PySp3};

/// Which mean-element model a reduced-orbit fit uses.
#[pyclass(module = "sidereon._sidereon", name = "ReducedOrbitModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyReducedOrbitModel {
    /// Circular orbit, eccentricity fixed at zero.
    CIRCULAR_SECULAR,
    /// Eccentric orbit via a nonsingular `(h, k)` parameterization.
    ECCENTRIC_SECULAR,
}

impl From<PyReducedOrbitModel> for Model {
    fn from(model: PyReducedOrbitModel) -> Self {
        match model {
            PyReducedOrbitModel::CIRCULAR_SECULAR => Model::CircularSecular,
            PyReducedOrbitModel::ECCENTRIC_SECULAR => Model::EccentricSecular,
        }
    }
}

impl From<Model> for PyReducedOrbitModel {
    fn from(model: Model) -> Self {
        match model {
            Model::CircularSecular => PyReducedOrbitModel::CIRCULAR_SECULAR,
            Model::EccentricSecular => PyReducedOrbitModel::ECCENTRIC_SECULAR,
        }
    }
}

/// Pair of `(3,)` numpy arrays returned for position/velocity queries.
type PyVec3Pair<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

/// Reference frame for a reduced-orbit position/velocity evaluation.
#[pyclass(module = "sidereon._sidereon", name = "ReducedOrbitFrame", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types, clippy::upper_case_acronyms)]
pub enum PyReducedOrbitFrame {
    /// Inertial GCRS (ECI).
    GCRS,
    /// Earth-fixed ITRF/IGS ECEF.
    ECEF,
}

impl From<PyReducedOrbitFrame> for Frame {
    fn from(frame: PyReducedOrbitFrame) -> Self {
        match frame {
            PyReducedOrbitFrame::GCRS => Frame::Gcrs,
            PyReducedOrbitFrame::ECEF => Frame::Ecef,
        }
    }
}

/// A calendar epoch (year, month, day, hour, minute, fractional second).
#[pyclass(module = "sidereon._sidereon", name = "CalendarEpoch")]
#[derive(Clone, Copy)]
pub struct PyCalendarEpoch {
    inner: CalendarEpoch,
}

#[pymethods]
impl PyCalendarEpoch {
    #[new]
    fn new(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> PyResult<Self> {
        if !second.is_finite() {
            return Err(PyValueError::new_err(
                "CalendarEpoch second must be finite (not NaN or infinite)",
            ));
        }
        Ok(Self {
            inner: CalendarEpoch::new(year, month, day, hour, minute, second),
        })
    }

    #[getter]
    fn year(&self) -> i32 {
        self.inner.year
    }

    #[getter]
    fn month(&self) -> i32 {
        self.inner.month
    }

    #[getter]
    fn day(&self) -> i32 {
        self.inner.day
    }

    #[getter]
    fn hour(&self) -> i32 {
        self.inner.hour
    }

    #[getter]
    fn minute(&self) -> i32 {
        self.inner.minute
    }

    #[getter]
    fn second(&self) -> f64 {
        self.inner.second
    }

    fn __repr__(&self) -> String {
        format!(
            "CalendarEpoch(year={}, month={}, day={}, hour={}, minute={}, second={})",
            self.inner.year,
            self.inner.month,
            self.inner.day,
            self.inner.hour,
            self.inner.minute,
            self.inner.second
        )
    }
}

/// A fitted reduced-orbit model: mean elements plus fit residual statistics.
///
/// Carries the time scale the fit was performed in; evaluation reinterprets
/// query epochs in that same scale.
#[pyclass(module = "sidereon._sidereon", name = "ReducedOrbit")]
#[derive(Clone, Copy)]
pub struct PyReducedOrbit {
    inner: ReducedOrbit,
    scale: TimeScale,
}

#[pymethods]
impl PyReducedOrbit {
    /// Which mean-element model these elements belong to.
    #[getter]
    fn model(&self) -> PyReducedOrbitModel {
        self.inner.elements.model.into()
    }

    /// Time scale the model was fitted in.
    #[getter]
    fn scale(&self) -> PyTimeScale {
        self.scale.into()
    }

    /// Reference epoch `t0`; all linear angle advances are measured from here.
    #[getter]
    fn epoch(&self) -> PyCalendarEpoch {
        PyCalendarEpoch {
            inner: self.inner.elements.epoch,
        }
    }

    /// Semi-major axis `a`, meters.
    #[getter]
    fn a_m(&self) -> f64 {
        self.inner.elements.a_m
    }

    /// Eccentricity.
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.elements.e
    }

    /// Inclination `i`, radians.
    #[getter]
    fn inclination_rad(&self) -> f64 {
        self.inner.elements.i_rad
    }

    /// Right ascension of the ascending node at `t0`, radians.
    #[getter]
    fn raan_rad(&self) -> f64 {
        self.inner.elements.raan_rad
    }

    /// Fitted nodal regression rate, radians per second.
    #[getter]
    fn raan_rate_rad_s(&self) -> f64 {
        self.inner.elements.raan_rate_rad_s
    }

    /// J2 nodal-regression seed for `raan_rate`, radians per second.
    #[getter]
    fn raan_rate_j2_rad_s(&self) -> f64 {
        self.inner.elements.raan_rate_j2_rad_s
    }

    /// Argument of latitude at `t0`, radians.
    #[getter]
    fn arg_lat_rad(&self) -> f64 {
        self.inner.elements.arg_lat_rad
    }

    /// Mean motion `n`, radians per second.
    #[getter]
    fn mean_motion_rad_s(&self) -> f64 {
        self.inner.elements.mean_motion_rad_s
    }

    /// Eccentricity vector component `h = e*sin(omega)`.
    #[getter]
    fn h(&self) -> f64 {
        self.inner.elements.h
    }

    /// Eccentricity vector component `k = e*cos(omega)`.
    #[getter]
    fn k(&self) -> f64 {
        self.inner.elements.k
    }

    /// Argument of perigee `omega = atan2(h, k)`, radians.
    #[getter]
    fn arg_perigee_rad(&self) -> f64 {
        self.inner.elements.arg_perigee_rad
    }

    /// Root-mean-square GCRS position residual over the fit samples, meters.
    #[getter]
    fn rms_m(&self) -> f64 {
        self.inner.stats.rms_m
    }

    /// Maximum GCRS position residual over the fit samples, meters.
    #[getter]
    fn max_m(&self) -> f64 {
        self.inner.stats.max_m
    }

    /// Number of samples used in the fit.
    #[getter]
    fn n_samples(&self) -> usize {
        self.inner.stats.n_samples
    }

    /// Evaluate the model position at `epoch` in `frame`, meters.
    fn position<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyCalendarEpoch,
        frame: PyReducedOrbitFrame,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let r = core_position(&self.inner.elements, epoch.inner, self.scale, frame.into())
            .map_err(to_solve_err)?;
        Ok(PyArray1::from_slice(py, &r))
    }

    /// Evaluate the model position and velocity at `epoch` in `frame`.
    ///
    /// Returns `(position_m, velocity_m_s)`, each a `(3,)` array.
    fn position_velocity<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyCalendarEpoch,
        frame: PyReducedOrbitFrame,
    ) -> PyResult<PyVec3Pair<'py>> {
        let (r, v) =
            core_position_velocity(&self.inner.elements, epoch.inner, self.scale, frame.into())
                .map_err(to_solve_err)?;
        Ok((PyArray1::from_slice(py, &r), PyArray1::from_slice(py, &v)))
    }

    /// Evaluate the model against truth ECEF samples.
    ///
    /// `truth` is a list of `(CalendarEpoch, x_m, y_m, z_m)`. Returns a
    /// [`DriftReport`].
    fn drift(
        &self,
        truth: Vec<(Py<PyCalendarEpoch>, f64, f64, f64)>,
        threshold_m: f64,
        py: Python<'_>,
    ) -> PyResult<PyDriftReport> {
        let samples = decode_samples(py, &truth)?;
        let report = core_drift(&self.inner.elements, &samples, self.scale, threshold_m)
            .map_err(to_solve_err)?;
        Ok(PyDriftReport {
            errors_m: report.per_epoch.iter().map(|entry| entry.error_m).collect(),
            max_m: report.max_m,
            rms_m: report.rms_m,
            threshold_horizon: report
                .threshold_horizon
                .map(|epoch| PyCalendarEpoch { inner: epoch }),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ReducedOrbit(model={:?}, a_m={:.3}, rms_m={:.3}, max_m={:.3}, n_samples={})",
            self.inner.elements.model,
            self.inner.elements.a_m,
            self.inner.stats.rms_m,
            self.inner.stats.max_m,
            self.inner.stats.n_samples
        )
    }
}

/// Model-vs-truth drift evaluation over a horizon of truth samples.
#[pyclass(module = "sidereon._sidereon", name = "DriftReport")]
#[derive(Clone)]
pub struct PyDriftReport {
    errors_m: Vec<f64>,
    max_m: f64,
    rms_m: f64,
    threshold_horizon: Option<PyCalendarEpoch>,
}

#[pymethods]
impl PyDriftReport {
    /// Per-epoch position error magnitudes, in input order, meters.
    #[getter]
    fn errors_m<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        PyArray1::from_slice(py, &self.errors_m)
    }

    /// Maximum error over the horizon, meters.
    #[getter]
    fn max_m(&self) -> f64 {
        self.max_m
    }

    /// Root-mean-square error over the horizon, meters.
    #[getter]
    fn rms_m(&self) -> f64 {
        self.rms_m
    }

    /// First epoch at which the error crosses the threshold, or `None`.
    #[getter]
    fn threshold_horizon(&self) -> Option<PyCalendarEpoch> {
        self.threshold_horizon
    }

    fn __repr__(&self) -> String {
        format!(
            "DriftReport(n={}, max_m={:.3}, rms_m={:.3}, crossed={})",
            self.errors_m.len(),
            self.max_m,
            self.rms_m,
            self.threshold_horizon.is_some()
        )
    }
}

/// A source-backed reduced-orbit fit plus sampling metadata.
#[pyclass(module = "sidereon._sidereon", name = "ReducedOrbitSourceFit")]
#[derive(Clone, Copy)]
pub struct PyReducedOrbitSourceFit {
    orbit: PyReducedOrbit,
    requested_samples: usize,
}

#[pymethods]
impl PyReducedOrbitSourceFit {
    /// The fitted reduced-orbit model.
    #[getter]
    fn orbit(&self) -> PyReducedOrbit {
        self.orbit
    }

    /// Number of source samples requested from the sampling window.
    #[getter]
    fn requested_samples(&self) -> usize {
        self.requested_samples
    }

    fn __repr__(&self) -> String {
        format!(
            "ReducedOrbitSourceFit(requested_samples={}, orbit={:?})",
            self.requested_samples, self.orbit.inner.elements.model
        )
    }
}

/// A source-backed reduced-orbit drift report plus sampling metadata.
#[pyclass(module = "sidereon._sidereon", name = "ReducedOrbitSourceDrift")]
#[derive(Clone)]
pub struct PyReducedOrbitSourceDrift {
    report: PyDriftReport,
    requested_samples: usize,
}

#[pymethods]
impl PyReducedOrbitSourceDrift {
    /// Model-vs-source drift report.
    #[getter]
    fn report(&self) -> PyDriftReport {
        self.report.clone()
    }

    /// Number of source samples requested from the sampling window.
    #[getter]
    fn requested_samples(&self) -> usize {
        self.requested_samples
    }

    fn __repr__(&self) -> String {
        format!(
            "ReducedOrbitSourceDrift(requested_samples={}, max_m={:.3})",
            self.requested_samples, self.report.max_m
        )
    }
}

fn decode_samples(
    py: Python<'_>,
    samples: &[(Py<PyCalendarEpoch>, f64, f64, f64)],
) -> PyResult<Vec<EcefSample>> {
    samples
        .iter()
        .map(|(epoch, x_m, y_m, z_m)| {
            Ok(EcefSample::new(
                epoch.try_borrow(py)?.inner,
                *x_m,
                *y_m,
                *z_m,
            ))
        })
        .collect()
}

/// Fit a reduced-orbit model to ECEF samples.
///
/// `samples` is a list of `(CalendarEpoch, x_m, y_m, z_m)` in ITRF meters,
/// interpreted in `scale`.
#[pyfunction]
#[pyo3(signature = (samples, scale, model))]
fn reduced_orbit_fit(
    py: Python<'_>,
    samples: Vec<(Py<PyCalendarEpoch>, f64, f64, f64)>,
    scale: PyTimeScale,
    model: PyReducedOrbitModel,
) -> PyResult<PyReducedOrbit> {
    let ecef = decode_samples(py, &samples)?;
    let scale: TimeScale = scale.into();
    let inner = fit_with_model(&ecef, scale, model.into()).map_err(to_solve_err)?;
    Ok(PyReducedOrbit { inner, scale })
}

fn parse_satellite(token: &str) -> PyResult<GnssSatelliteId> {
    token
        .parse::<GnssSatelliteId>()
        .map_err(|err| PyValueError::new_err(format!("invalid satellite token {token:?}: {err}")))
}

fn source_sampling(
    t0: &PyCalendarEpoch,
    t1: &PyCalendarEpoch,
    cadence_s: f64,
) -> ReducedOrbitSourceSampling {
    ReducedOrbitSourceSampling::new(t0.inner, t1.inner, cadence_s)
}

fn py_drift_report(report: sidereon_core::orbit::DriftReport) -> PyDriftReport {
    PyDriftReport {
        errors_m: report.per_epoch.iter().map(|entry| entry.error_m).collect(),
        max_m: report.max_m,
        rms_m: report.rms_m,
        threshold_horizon: report
            .threshold_horizon
            .map(|epoch| PyCalendarEpoch { inner: epoch }),
    }
}

/// Sample an SP3 product and fit a reduced-orbit model.
#[pyfunction]
#[pyo3(signature = (sp3, satellite, t0, t1, cadence_s, model))]
fn reduced_orbit_fit_sp3_source(
    sp3: &PySp3,
    satellite: &str,
    t0: &PyCalendarEpoch,
    t1: &PyCalendarEpoch,
    cadence_s: f64,
    model: PyReducedOrbitModel,
) -> PyResult<PyReducedOrbitSourceFit> {
    let satellite = parse_satellite(satellite)?;
    let scale = sp3.inner.header.time_scale;
    let source = ReducedOrbitSource::Sp3 {
        product: &sp3.inner,
        satellite,
    };
    let fit = core_source_fit(
        source,
        ReducedOrbitSourceFitOptions {
            sampling: source_sampling(t0, t1, cadence_s),
            model: model.into(),
        },
    )
    .map_err(to_solve_err)?;
    Ok(PyReducedOrbitSourceFit {
        orbit: PyReducedOrbit {
            inner: fit.orbit,
            scale,
        },
        requested_samples: fit.requested_samples,
    })
}

/// Sample an SP3 product and evaluate a reduced orbit against it.
#[pyfunction]
#[pyo3(signature = (orbit, sp3, satellite, t0, t1, cadence_s, threshold_m))]
fn reduced_orbit_drift_sp3_source(
    orbit: &PyReducedOrbit,
    sp3: &PySp3,
    satellite: &str,
    t0: &PyCalendarEpoch,
    t1: &PyCalendarEpoch,
    cadence_s: f64,
    threshold_m: f64,
) -> PyResult<PyReducedOrbitSourceDrift> {
    let satellite = parse_satellite(satellite)?;
    let source = ReducedOrbitSource::Sp3 {
        product: &sp3.inner,
        satellite,
    };
    let drift = core_source_drift(
        &orbit.inner.elements,
        source,
        ReducedOrbitSourceDriftOptions {
            sampling: source_sampling(t0, t1, cadence_s),
            threshold_m,
        },
    )
    .map_err(to_solve_err)?;
    Ok(PyReducedOrbitSourceDrift {
        report: py_drift_report(drift.report),
        requested_samples: drift.requested_samples,
    })
}

/// Map a piecewise reduced-orbit failure to a [`SolveError`](crate::SolveError),
/// preserving the engine's reason (the underlying single-segment error keeps its
/// own message).
fn piecewise_err(err: PiecewiseOrbitError) -> PyErr {
    let message = match err {
        PiecewiseOrbitError::InvalidSegment => {
            "piecewise segment length is missing, non-positive, or rounds below one second"
                .to_string()
        }
        PiecewiseOrbitError::OutOfRange => {
            "query epoch is outside the piecewise model coverage".to_string()
        }
        PiecewiseOrbitError::TooFewSamples { got, required } => {
            format!("piecewise fit needs at least {required} samples, got {got}")
        }
        PiecewiseOrbitError::Reduced(inner) => inner.to_string(),
    };
    to_solve_err(message)
}

/// One contiguous segment of a piecewise reduced orbit: its `[t0, t1)` coverage
/// (inclusive `t1` only for the final segment) and the fitted model.
#[pyclass(module = "sidereon._sidereon", name = "PiecewiseSegment")]
#[derive(Clone)]
pub struct PyPiecewiseSegment {
    inner: PiecewiseSegment,
    scale: TimeScale,
}

#[pymethods]
impl PyPiecewiseSegment {
    /// Inclusive segment start epoch.
    #[getter]
    fn t0(&self) -> PyCalendarEpoch {
        PyCalendarEpoch {
            inner: self.inner.t0,
        }
    }

    /// Segment end epoch (exclusive, except inclusive on the final segment).
    #[getter]
    fn t1(&self) -> PyCalendarEpoch {
        PyCalendarEpoch {
            inner: self.inner.t1,
        }
    }

    /// The fitted reduced-orbit model for this segment.
    #[getter]
    fn orbit(&self) -> PyReducedOrbit {
        PyReducedOrbit {
            inner: self.inner.orbit,
            scale: self.scale,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PiecewiseSegment(rms_m={:.3}, n_samples={})",
            self.inner.orbit.stats.rms_m, self.inner.orbit.stats.n_samples
        )
    }
}

/// A long span represented by contiguous independently-fitted reduced-orbit
/// segments. Evaluation selects the segment covering the query epoch.
#[pyclass(module = "sidereon._sidereon", name = "PiecewiseOrbit")]
#[derive(Clone)]
pub struct PyPiecewiseOrbit {
    inner: PiecewiseOrbit,
    scale: TimeScale,
}

#[pymethods]
impl PyPiecewiseOrbit {
    /// The mean-element model fitted in every segment.
    #[getter]
    fn model(&self) -> PyReducedOrbitModel {
        self.inner.model.into()
    }

    /// Time scale the model was fitted in.
    #[getter]
    fn scale(&self) -> PyTimeScale {
        self.scale.into()
    }

    /// Advertised coverage start epoch.
    #[getter]
    fn t0(&self) -> PyCalendarEpoch {
        PyCalendarEpoch {
            inner: self.inner.t0,
        }
    }

    /// Advertised coverage end epoch (inclusive on the final segment).
    #[getter]
    fn t1(&self) -> PyCalendarEpoch {
        PyCalendarEpoch {
            inner: self.inner.t1,
        }
    }

    /// Rounded segment length used to tile the requested window, seconds.
    #[getter]
    fn segment_s(&self) -> i64 {
        self.inner.segment_s
    }

    /// Number of contiguous fitted segments.
    #[getter]
    fn n_segments(&self) -> usize {
        self.inner.segments.len()
    }

    /// The contiguous fitted segments in coverage order.
    #[getter]
    fn segments(&self) -> Vec<PyPiecewiseSegment> {
        self.inner
            .segments
            .iter()
            .map(|segment| PyPiecewiseSegment {
                inner: segment.clone(),
                scale: self.scale,
            })
            .collect()
    }

    /// The segment covering `epoch`.
    fn select_segment(&self, epoch: &PyCalendarEpoch) -> PyResult<PyPiecewiseSegment> {
        let segment = core_select_segment(&self.inner, epoch.inner).map_err(piecewise_err)?;
        Ok(PyPiecewiseSegment {
            inner: segment.clone(),
            scale: self.scale,
        })
    }

    /// Evaluate the piecewise model position at `epoch` in `frame`, meters.
    fn position<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyCalendarEpoch,
        frame: PyReducedOrbitFrame,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let r = core_piecewise_position(&self.inner, epoch.inner, self.scale, frame.into())
            .map_err(piecewise_err)?;
        Ok(PyArray1::from_slice(py, &r))
    }

    /// Evaluate piecewise position and velocity at `epoch` in `frame`.
    ///
    /// Returns `(position_m, velocity_m_s)`, each a `(3,)` array.
    fn position_velocity<'py>(
        &self,
        py: Python<'py>,
        epoch: &PyCalendarEpoch,
        frame: PyReducedOrbitFrame,
    ) -> PyResult<PyVec3Pair<'py>> {
        let (r, v) =
            core_piecewise_position_velocity(&self.inner, epoch.inner, self.scale, frame.into())
                .map_err(piecewise_err)?;
        Ok((PyArray1::from_slice(py, &r), PyArray1::from_slice(py, &v)))
    }

    /// Evaluate the piecewise model against truth ECEF samples.
    ///
    /// `truth` is a list of `(CalendarEpoch, x_m, y_m, z_m)`; samples outside the
    /// model span are skipped. Returns a [`DriftReport`].
    fn drift(
        &self,
        truth: Vec<(Py<PyCalendarEpoch>, f64, f64, f64)>,
        threshold_m: f64,
        py: Python<'_>,
    ) -> PyResult<PyDriftReport> {
        let samples = decode_samples(py, &truth)?;
        let report = core_piecewise_drift(&self.inner, &samples, self.scale, threshold_m)
            .map_err(piecewise_err)?;
        Ok(PyDriftReport {
            errors_m: report.per_epoch.iter().map(|entry| entry.error_m).collect(),
            max_m: report.max_m,
            rms_m: report.rms_m,
            threshold_horizon: report
                .threshold_horizon
                .map(|epoch| PyCalendarEpoch { inner: epoch }),
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "PiecewiseOrbit(model={:?}, n_segments={}, segment_s={})",
            self.inner.model,
            self.inner.segments.len(),
            self.inner.segment_s
        )
    }
}

/// Fit a piecewise reduced-orbit model over `[t0, t1]` tiled into `segment_s`
/// segments.
///
/// `samples` is a list of `(CalendarEpoch, x_m, y_m, z_m)` in ITRF meters,
/// interpreted in `scale`. Each segment is fitted independently from the samples
/// that fall in it. Delegates to the core `fit_piecewise`.
#[pyfunction]
#[pyo3(signature = (samples, scale, model, t0, t1, segment_s))]
fn reduced_orbit_fit_piecewise(
    py: Python<'_>,
    samples: Vec<(Py<PyCalendarEpoch>, f64, f64, f64)>,
    scale: PyTimeScale,
    model: PyReducedOrbitModel,
    t0: &PyCalendarEpoch,
    t1: &PyCalendarEpoch,
    segment_s: i64,
) -> PyResult<PyPiecewiseOrbit> {
    let ecef = decode_samples(py, &samples)?;
    let scale: TimeScale = scale.into();
    let inner = core_fit_piecewise(&ecef, scale, model.into(), t0.inner, t1.inner, segment_s)
        .map_err(piecewise_err)?;
    Ok(PyPiecewiseOrbit { inner, scale })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyReducedOrbitModel>()?;
    m.add_class::<PyReducedOrbitFrame>()?;
    m.add_class::<PyCalendarEpoch>()?;
    m.add_class::<PyReducedOrbit>()?;
    m.add_class::<PyDriftReport>()?;
    m.add_class::<PyReducedOrbitSourceFit>()?;
    m.add_class::<PyReducedOrbitSourceDrift>()?;
    m.add_class::<PyPiecewiseSegment>()?;
    m.add_class::<PyPiecewiseOrbit>()?;
    m.add_function(wrap_pyfunction!(reduced_orbit_fit, m)?)?;
    m.add_function(wrap_pyfunction!(reduced_orbit_fit_sp3_source, m)?)?;
    m.add_function(wrap_pyfunction!(reduced_orbit_drift_sp3_source, m)?)?;
    m.add_function(wrap_pyfunction!(reduced_orbit_fit_piecewise, m)?)?;
    Ok(())
}
