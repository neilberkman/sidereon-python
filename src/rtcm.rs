//! RTCM 3.x differential-GNSS stream codec binding.
//!
//! Thin INTERFACE over `sidereon_core::rtcm`. It hands raw stream bytes to the
//! core frame scanner and message decoder, wraps each decoded
//! [`Message`](sidereon_core::rtcm::Message) in a Pythonic object that exposes
//! the typed field IR (station coordinates, antenna descriptor, GPS / GLONASS
//! ephemeris, MSM4 / MSM7 observations, and verbatim unsupported messages), and
//! re-encodes a decoded message back to bytes through the core encoder so a
//! decode/encode pair round-trips byte-for-byte. All framing, CRC, bit packing,
//! and message grammar live in the core; no codec logic lives here.
//!
//! The core `Message` enum is `#[non_exhaustive]` and its per-message encoders
//! are crate-private, so a message cannot be constructed field-by-field from
//! outside the engine: these objects originate from decoding and re-encode the
//! decoded value.

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use pyo3::Bound;

use sidereon_core::rtcm::{
    decode_messages as core_decode_messages, encode_frame as core_encode_frame,
    message_number as core_message_number, AntennaDescriptor, GlonassEphemeris, GpsEphemeris,
    Message, MsmHeader, MsmKind, MsmMessage, MsmSatellite, MsmSignal, StationCoordinates,
    UnsupportedMessage,
};

use pyo3::exceptions::PyValueError;
use sidereon_core::GnssSystem;

use crate::RtcmParseError;

fn to_rtcm_err<E: std::fmt::Display>(err: E) -> PyErr {
    RtcmParseError::new_err(err.to_string())
}

/// A 1005 / 1006 station antenna reference point. Each coordinate is the raw
/// transmitted integer (1/10000 m); the `*_m` helpers apply the scale.
#[pyclass(module = "sidereon._sidereon", name = "RtcmStationCoordinates")]
#[derive(Clone)]
pub struct PyRtcmStationCoordinates {
    inner: StationCoordinates,
}

#[pymethods]
impl PyRtcmStationCoordinates {
    /// Construct a 1005 / 1006 station antenna reference point from raw fields.
    ///
    /// Coordinates are the raw transmitted integers in 0.0001 m steps;
    /// `antenna_height` (raw 0.0001 m) is present only for message 1006.
    #[new]
    #[pyo3(signature = (
        message_number,
        reference_station_id,
        itrf_realization_year,
        gps_indicator,
        glonass_indicator,
        galileo_indicator,
        reference_station_indicator,
        ecef_x,
        ecef_y,
        ecef_z,
        single_receiver_oscillator,
        reserved,
        quarter_cycle_indicator,
        antenna_height=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        message_number: u16,
        reference_station_id: u16,
        itrf_realization_year: u8,
        gps_indicator: bool,
        glonass_indicator: bool,
        galileo_indicator: bool,
        reference_station_indicator: bool,
        ecef_x: i64,
        ecef_y: i64,
        ecef_z: i64,
        single_receiver_oscillator: bool,
        reserved: bool,
        quarter_cycle_indicator: u8,
        antenna_height: Option<u16>,
    ) -> Self {
        Self {
            inner: StationCoordinates {
                message_number,
                reference_station_id,
                itrf_realization_year,
                gps_indicator,
                glonass_indicator,
                galileo_indicator,
                reference_station_indicator,
                ecef_x,
                ecef_y,
                ecef_z,
                single_receiver_oscillator,
                reserved,
                quarter_cycle_indicator,
                antenna_height,
            },
        }
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn reference_station_id(&self) -> u16 {
        self.inner.reference_station_id
    }
    #[getter]
    fn itrf_realization_year(&self) -> u8 {
        self.inner.itrf_realization_year
    }
    #[getter]
    fn gps_indicator(&self) -> bool {
        self.inner.gps_indicator
    }
    #[getter]
    fn glonass_indicator(&self) -> bool {
        self.inner.glonass_indicator
    }
    #[getter]
    fn galileo_indicator(&self) -> bool {
        self.inner.galileo_indicator
    }
    #[getter]
    fn reference_station_indicator(&self) -> bool {
        self.inner.reference_station_indicator
    }
    #[getter]
    fn ecef_x(&self) -> i64 {
        self.inner.ecef_x
    }
    #[getter]
    fn ecef_y(&self) -> i64 {
        self.inner.ecef_y
    }
    #[getter]
    fn ecef_z(&self) -> i64 {
        self.inner.ecef_z
    }
    #[getter]
    fn single_receiver_oscillator(&self) -> bool {
        self.inner.single_receiver_oscillator
    }
    #[getter]
    fn reserved(&self) -> bool {
        self.inner.reserved
    }
    #[getter]
    fn quarter_cycle_indicator(&self) -> u8 {
        self.inner.quarter_cycle_indicator
    }
    #[getter]
    fn antenna_height(&self) -> Option<u16> {
        self.inner.antenna_height
    }
    /// ECEF X, metres.
    #[getter]
    fn x_m(&self) -> f64 {
        self.inner.x_m()
    }
    /// ECEF Y, metres.
    #[getter]
    fn y_m(&self) -> f64 {
        self.inner.y_m()
    }
    /// ECEF Z, metres.
    #[getter]
    fn z_m(&self) -> f64 {
        self.inner.z_m()
    }
    /// Antenna height, metres (1006 only).
    #[getter]
    fn antenna_height_m(&self) -> Option<f64> {
        self.inner.antenna_height_m()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmStationCoordinates(message_number={}, reference_station_id={}, x_m={:.4}, \
             y_m={:.4}, z_m={:.4})",
            self.inner.message_number,
            self.inner.reference_station_id,
            self.inner.x_m(),
            self.inner.y_m(),
            self.inner.z_m()
        )
    }
}

/// A 1007 / 1008 / 1033 antenna or receiver descriptor.
#[pyclass(module = "sidereon._sidereon", name = "RtcmAntennaDescriptor")]
#[derive(Clone)]
pub struct PyRtcmAntennaDescriptor {
    inner: AntennaDescriptor,
}

#[pymethods]
impl PyRtcmAntennaDescriptor {
    /// Construct a 1007 / 1008 / 1033 antenna or receiver descriptor from fields.
    ///
    /// The serial-number and receiver strings are present only for the richer
    /// message numbers (1008 adds the antenna serial; 1033 adds the receiver
    /// fields); pass `None` where the message omits them.
    #[new]
    #[pyo3(signature = (
        message_number,
        reference_station_id,
        antenna_descriptor,
        antenna_setup_id,
        antenna_serial_number=None,
        receiver_type=None,
        receiver_firmware_version=None,
        receiver_serial_number=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        message_number: u16,
        reference_station_id: u16,
        antenna_descriptor: String,
        antenna_setup_id: u8,
        antenna_serial_number: Option<String>,
        receiver_type: Option<String>,
        receiver_firmware_version: Option<String>,
        receiver_serial_number: Option<String>,
    ) -> Self {
        Self {
            inner: AntennaDescriptor {
                message_number,
                reference_station_id,
                antenna_descriptor,
                antenna_setup_id,
                antenna_serial_number,
                receiver_type,
                receiver_firmware_version,
                receiver_serial_number,
            },
        }
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn reference_station_id(&self) -> u16 {
        self.inner.reference_station_id
    }
    #[getter]
    fn antenna_descriptor(&self) -> String {
        self.inner.antenna_descriptor.clone()
    }
    #[getter]
    fn antenna_setup_id(&self) -> u8 {
        self.inner.antenna_setup_id
    }
    #[getter]
    fn antenna_serial_number(&self) -> Option<String> {
        self.inner.antenna_serial_number.clone()
    }
    #[getter]
    fn receiver_type(&self) -> Option<String> {
        self.inner.receiver_type.clone()
    }
    #[getter]
    fn receiver_firmware_version(&self) -> Option<String> {
        self.inner.receiver_firmware_version.clone()
    }
    #[getter]
    fn receiver_serial_number(&self) -> Option<String> {
        self.inner.receiver_serial_number.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmAntennaDescriptor(message_number={}, antenna_descriptor={:?})",
            self.inner.message_number, self.inner.antenna_descriptor
        )
    }
}

/// A 1019 GPS broadcast ephemeris. Every field is the raw transmitted integer.
#[pyclass(module = "sidereon._sidereon", name = "RtcmGpsEphemeris")]
#[derive(Clone)]
pub struct PyRtcmGpsEphemeris {
    inner: GpsEphemeris,
}

#[pymethods]
impl PyRtcmGpsEphemeris {
    /// Construct a 1019 GPS broadcast ephemeris from its raw transmitted
    /// integers. Each argument is the field as carried on the wire (the decoder
    /// applies no scaling), so a construct/encode pair round-trips byte-for-byte.
    #[new]
    #[pyo3(signature = (
        satellite_id, week_number, sv_accuracy, code_on_l2, idot, iode, t_oc, a_f2, a_f1, a_f0,
        iodc, c_rs, delta_n, m0, c_uc, eccentricity, c_us, sqrt_a, t_oe, c_ic, omega0, c_is, i0,
        c_rc, omega, omega_dot, t_gd, sv_health, l2_p_data_flag, fit_interval,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        satellite_id: u8,
        week_number: u16,
        sv_accuracy: u8,
        code_on_l2: u8,
        idot: i32,
        iode: u8,
        t_oc: u16,
        a_f2: i16,
        a_f1: i32,
        a_f0: i32,
        iodc: u16,
        c_rs: i32,
        delta_n: i32,
        m0: i64,
        c_uc: i32,
        eccentricity: u64,
        c_us: i32,
        sqrt_a: u64,
        t_oe: u16,
        c_ic: i32,
        omega0: i64,
        c_is: i32,
        i0: i64,
        c_rc: i32,
        omega: i64,
        omega_dot: i32,
        t_gd: i16,
        sv_health: u8,
        l2_p_data_flag: bool,
        fit_interval: bool,
    ) -> Self {
        Self {
            inner: GpsEphemeris {
                satellite_id,
                week_number,
                sv_accuracy,
                code_on_l2,
                idot,
                iode,
                t_oc,
                a_f2,
                a_f1,
                a_f0,
                iodc,
                c_rs,
                delta_n,
                m0,
                c_uc,
                eccentricity,
                c_us,
                sqrt_a,
                t_oe,
                c_ic,
                omega0,
                c_is,
                i0,
                c_rc,
                omega,
                omega_dot,
                t_gd,
                sv_health,
                l2_p_data_flag,
                fit_interval,
            },
        }
    }

    #[getter]
    fn satellite_id(&self) -> u8 {
        self.inner.satellite_id
    }
    #[getter]
    fn week_number(&self) -> u16 {
        self.inner.week_number
    }
    #[getter]
    fn sv_accuracy(&self) -> u8 {
        self.inner.sv_accuracy
    }
    #[getter]
    fn code_on_l2(&self) -> u8 {
        self.inner.code_on_l2
    }
    #[getter]
    fn idot(&self) -> i32 {
        self.inner.idot
    }
    #[getter]
    fn iode(&self) -> u8 {
        self.inner.iode
    }
    #[getter]
    fn t_oc(&self) -> u16 {
        self.inner.t_oc
    }
    #[getter]
    fn a_f2(&self) -> i16 {
        self.inner.a_f2
    }
    #[getter]
    fn a_f1(&self) -> i32 {
        self.inner.a_f1
    }
    #[getter]
    fn a_f0(&self) -> i32 {
        self.inner.a_f0
    }
    #[getter]
    fn iodc(&self) -> u16 {
        self.inner.iodc
    }
    #[getter]
    fn c_rs(&self) -> i32 {
        self.inner.c_rs
    }
    #[getter]
    fn delta_n(&self) -> i32 {
        self.inner.delta_n
    }
    #[getter]
    fn m0(&self) -> i64 {
        self.inner.m0
    }
    #[getter]
    fn c_uc(&self) -> i32 {
        self.inner.c_uc
    }
    #[getter]
    fn eccentricity(&self) -> u64 {
        self.inner.eccentricity
    }
    #[getter]
    fn c_us(&self) -> i32 {
        self.inner.c_us
    }
    #[getter]
    fn sqrt_a(&self) -> u64 {
        self.inner.sqrt_a
    }
    #[getter]
    fn t_oe(&self) -> u16 {
        self.inner.t_oe
    }
    #[getter]
    fn c_ic(&self) -> i32 {
        self.inner.c_ic
    }
    #[getter]
    fn omega0(&self) -> i64 {
        self.inner.omega0
    }
    #[getter]
    fn c_is(&self) -> i32 {
        self.inner.c_is
    }
    #[getter]
    fn i0(&self) -> i64 {
        self.inner.i0
    }
    #[getter]
    fn c_rc(&self) -> i32 {
        self.inner.c_rc
    }
    #[getter]
    fn omega(&self) -> i64 {
        self.inner.omega
    }
    #[getter]
    fn omega_dot(&self) -> i32 {
        self.inner.omega_dot
    }
    #[getter]
    fn t_gd(&self) -> i16 {
        self.inner.t_gd
    }
    #[getter]
    fn sv_health(&self) -> u8 {
        self.inner.sv_health
    }
    #[getter]
    fn l2_p_data_flag(&self) -> bool {
        self.inner.l2_p_data_flag
    }
    #[getter]
    fn fit_interval(&self) -> bool {
        self.inner.fit_interval
    }
    /// Canonical satellite identifier (e.g. `"G05"`), or `None` if the
    /// transmitted id is out of range.
    fn satellite(&self) -> Option<String> {
        self.inner.satellite().ok().map(|id| id.to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmGpsEphemeris(satellite_id={}, week_number={}, iode={})",
            self.inner.satellite_id, self.inner.week_number, self.inner.iode
        )
    }
}

/// A 1020 GLONASS broadcast ephemeris. Every field is the raw transmitted
/// integer.
#[pyclass(module = "sidereon._sidereon", name = "RtcmGlonassEphemeris")]
#[derive(Clone)]
pub struct PyRtcmGlonassEphemeris {
    inner: GlonassEphemeris,
}

#[pymethods]
impl PyRtcmGlonassEphemeris {
    /// Construct a 1020 GLONASS broadcast ephemeris from its raw transmitted
    /// integers. Each argument is the field as carried on the wire, so a
    /// construct/encode pair round-trips byte-for-byte.
    #[new]
    #[pyo3(signature = (
        satellite_id, frequency_channel, almanac_health, almanac_health_availability, p1, t_k,
        b_n_msb, p2, t_b, xn_dot, xn, xn_dot_dot, yn_dot, yn, yn_dot_dot, zn_dot, zn, zn_dot_dot,
        p3, gamma_n, m_p, m_l_n_third, tau_n, delta_tau_n, e_n, m_p4, m_f_t, m_n_t, m_m,
        additional_data_available, n_a, tau_c, m_n4, m_tau_gps, m_l_n_fifth, reserved,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        satellite_id: u8,
        frequency_channel: u8,
        almanac_health: bool,
        almanac_health_availability: bool,
        p1: u8,
        t_k: u16,
        b_n_msb: bool,
        p2: bool,
        t_b: u8,
        xn_dot: i32,
        xn: i32,
        xn_dot_dot: i8,
        yn_dot: i32,
        yn: i32,
        yn_dot_dot: i8,
        zn_dot: i32,
        zn: i32,
        zn_dot_dot: i8,
        p3: bool,
        gamma_n: i16,
        m_p: u8,
        m_l_n_third: bool,
        tau_n: i32,
        delta_tau_n: i8,
        e_n: u8,
        m_p4: bool,
        m_f_t: u8,
        m_n_t: u16,
        m_m: u8,
        additional_data_available: bool,
        n_a: u16,
        tau_c: i64,
        m_n4: u8,
        m_tau_gps: i32,
        m_l_n_fifth: bool,
        reserved: u8,
    ) -> Self {
        Self {
            inner: GlonassEphemeris {
                satellite_id,
                frequency_channel,
                almanac_health,
                almanac_health_availability,
                p1,
                t_k,
                b_n_msb,
                p2,
                t_b,
                xn_dot,
                xn,
                xn_dot_dot,
                yn_dot,
                yn,
                yn_dot_dot,
                zn_dot,
                zn,
                zn_dot_dot,
                p3,
                gamma_n,
                m_p,
                m_l_n_third,
                tau_n,
                delta_tau_n,
                e_n,
                m_p4,
                m_f_t,
                m_n_t,
                m_m,
                additional_data_available,
                n_a,
                tau_c,
                m_n4,
                m_tau_gps,
                m_l_n_fifth,
                reserved,
            },
        }
    }

    #[getter]
    fn satellite_id(&self) -> u8 {
        self.inner.satellite_id
    }
    #[getter]
    fn frequency_channel(&self) -> u8 {
        self.inner.frequency_channel
    }
    #[getter]
    fn almanac_health(&self) -> bool {
        self.inner.almanac_health
    }
    #[getter]
    fn almanac_health_availability(&self) -> bool {
        self.inner.almanac_health_availability
    }
    #[getter]
    fn p1(&self) -> u8 {
        self.inner.p1
    }
    #[getter]
    fn t_k(&self) -> u16 {
        self.inner.t_k
    }
    #[getter]
    fn b_n_msb(&self) -> bool {
        self.inner.b_n_msb
    }
    #[getter]
    fn p2(&self) -> bool {
        self.inner.p2
    }
    #[getter]
    fn t_b(&self) -> u8 {
        self.inner.t_b
    }
    #[getter]
    fn xn_dot(&self) -> i32 {
        self.inner.xn_dot
    }
    #[getter]
    fn xn(&self) -> i32 {
        self.inner.xn
    }
    #[getter]
    fn xn_dot_dot(&self) -> i8 {
        self.inner.xn_dot_dot
    }
    #[getter]
    fn yn_dot(&self) -> i32 {
        self.inner.yn_dot
    }
    #[getter]
    fn yn(&self) -> i32 {
        self.inner.yn
    }
    #[getter]
    fn yn_dot_dot(&self) -> i8 {
        self.inner.yn_dot_dot
    }
    #[getter]
    fn zn_dot(&self) -> i32 {
        self.inner.zn_dot
    }
    #[getter]
    fn zn(&self) -> i32 {
        self.inner.zn
    }
    #[getter]
    fn zn_dot_dot(&self) -> i8 {
        self.inner.zn_dot_dot
    }
    #[getter]
    fn p3(&self) -> bool {
        self.inner.p3
    }
    #[getter]
    fn gamma_n(&self) -> i16 {
        self.inner.gamma_n
    }
    #[getter]
    fn m_p(&self) -> u8 {
        self.inner.m_p
    }
    #[getter]
    fn m_l_n_third(&self) -> bool {
        self.inner.m_l_n_third
    }
    #[getter]
    fn tau_n(&self) -> i32 {
        self.inner.tau_n
    }
    #[getter]
    fn delta_tau_n(&self) -> i8 {
        self.inner.delta_tau_n
    }
    #[getter]
    fn e_n(&self) -> u8 {
        self.inner.e_n
    }
    #[getter]
    fn m_p4(&self) -> bool {
        self.inner.m_p4
    }
    #[getter]
    fn m_f_t(&self) -> u8 {
        self.inner.m_f_t
    }
    #[getter]
    fn m_n_t(&self) -> u16 {
        self.inner.m_n_t
    }
    #[getter]
    fn m_m(&self) -> u8 {
        self.inner.m_m
    }
    #[getter]
    fn additional_data_available(&self) -> bool {
        self.inner.additional_data_available
    }
    #[getter]
    fn n_a(&self) -> u16 {
        self.inner.n_a
    }
    #[getter]
    fn tau_c(&self) -> i64 {
        self.inner.tau_c
    }
    #[getter]
    fn m_n4(&self) -> u8 {
        self.inner.m_n4
    }
    #[getter]
    fn m_tau_gps(&self) -> i32 {
        self.inner.m_tau_gps
    }
    #[getter]
    fn m_l_n_fifth(&self) -> bool {
        self.inner.m_l_n_fifth
    }
    #[getter]
    fn reserved(&self) -> u8 {
        self.inner.reserved
    }
    /// Canonical satellite identifier (e.g. `"R07"`), or `None` if the
    /// transmitted id is out of range.
    fn satellite(&self) -> Option<String> {
        self.inner.satellite().ok().map(|id| id.to_string())
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmGlonassEphemeris(satellite_id={}, frequency_channel={})",
            self.inner.satellite_id, self.inner.frequency_channel
        )
    }
}

/// The MSM header common to every MSM type.
#[pyclass(module = "sidereon._sidereon", name = "RtcmMsmHeader")]
#[derive(Clone, Copy)]
pub struct PyRtcmMsmHeader {
    inner: MsmHeader,
}

#[pymethods]
impl PyRtcmMsmHeader {
    /// Construct the common MSM header from its raw transmitted fields.
    #[new]
    #[pyo3(signature = (
        reference_station_id,
        epoch_time,
        multiple_message,
        iods,
        reserved,
        clock_steering,
        external_clock,
        divergence_free_smoothing,
        smoothing_interval,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        reference_station_id: u16,
        epoch_time: u32,
        multiple_message: bool,
        iods: u8,
        reserved: u8,
        clock_steering: u8,
        external_clock: u8,
        divergence_free_smoothing: bool,
        smoothing_interval: u8,
    ) -> Self {
        Self {
            inner: MsmHeader {
                reference_station_id,
                epoch_time,
                multiple_message,
                iods,
                reserved,
                clock_steering,
                external_clock,
                divergence_free_smoothing,
                smoothing_interval,
            },
        }
    }

    #[getter]
    fn reference_station_id(&self) -> u16 {
        self.inner.reference_station_id
    }
    #[getter]
    fn epoch_time(&self) -> u32 {
        self.inner.epoch_time
    }
    #[getter]
    fn multiple_message(&self) -> bool {
        self.inner.multiple_message
    }
    #[getter]
    fn iods(&self) -> u8 {
        self.inner.iods
    }
    #[getter]
    fn reserved(&self) -> u8 {
        self.inner.reserved
    }
    #[getter]
    fn clock_steering(&self) -> u8 {
        self.inner.clock_steering
    }
    #[getter]
    fn external_clock(&self) -> u8 {
        self.inner.external_clock
    }
    #[getter]
    fn divergence_free_smoothing(&self) -> bool {
        self.inner.divergence_free_smoothing
    }
    #[getter]
    fn smoothing_interval(&self) -> u8 {
        self.inner.smoothing_interval
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmMsmHeader(reference_station_id={}, epoch_time={})",
            self.inner.reference_station_id, self.inner.epoch_time
        )
    }
}

/// Per-satellite data for one MSM satellite.
#[pyclass(module = "sidereon._sidereon", name = "RtcmMsmSatellite")]
#[derive(Clone, Copy)]
pub struct PyRtcmMsmSatellite {
    inner: MsmSatellite,
}

#[pymethods]
impl PyRtcmMsmSatellite {
    /// Construct one MSM satellite row. `extended_info` and
    /// `rough_phase_range_rate_m_s` are present only in MSM7; pass `None` for MSM4.
    #[new]
    #[pyo3(signature = (
        id,
        rough_range_ms,
        rough_range_mod1,
        extended_info=None,
        rough_phase_range_rate_m_s=None,
    ))]
    fn new(
        id: u8,
        rough_range_ms: u8,
        rough_range_mod1: u16,
        extended_info: Option<u8>,
        rough_phase_range_rate_m_s: Option<i16>,
    ) -> Self {
        Self {
            inner: MsmSatellite {
                id,
                rough_range_ms,
                rough_range_mod1,
                extended_info,
                rough_phase_range_rate_m_s,
            },
        }
    }

    #[getter]
    fn id(&self) -> u8 {
        self.inner.id
    }
    #[getter]
    fn rough_range_ms(&self) -> u8 {
        self.inner.rough_range_ms
    }
    #[getter]
    fn rough_range_mod1(&self) -> u16 {
        self.inner.rough_range_mod1
    }
    #[getter]
    fn extended_info(&self) -> Option<u8> {
        self.inner.extended_info
    }
    #[getter]
    fn rough_phase_range_rate_m_s(&self) -> Option<i16> {
        self.inner.rough_phase_range_rate_m_s
    }

    fn __repr__(&self) -> String {
        format!("RtcmMsmSatellite(id={})", self.inner.id)
    }
}

/// Per-cell signal data for one active (satellite, signal) pair.
#[pyclass(module = "sidereon._sidereon", name = "RtcmMsmSignal")]
#[derive(Clone, Copy)]
pub struct PyRtcmMsmSignal {
    inner: MsmSignal,
}

#[pymethods]
impl PyRtcmMsmSignal {
    /// Construct one MSM signal cell. `fine_phase_range_rate` is present only in
    /// MSM7; pass `None` for MSM4.
    #[new]
    #[pyo3(signature = (
        satellite_id,
        signal_id,
        fine_pseudorange,
        fine_phase_range,
        lock_time_indicator,
        half_cycle_ambiguity,
        cnr,
        fine_phase_range_rate=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        satellite_id: u8,
        signal_id: u8,
        fine_pseudorange: i32,
        fine_phase_range: i32,
        lock_time_indicator: u16,
        half_cycle_ambiguity: bool,
        cnr: u16,
        fine_phase_range_rate: Option<i16>,
    ) -> Self {
        Self {
            inner: MsmSignal {
                satellite_id,
                signal_id,
                fine_pseudorange,
                fine_phase_range,
                lock_time_indicator,
                half_cycle_ambiguity,
                cnr,
                fine_phase_range_rate,
            },
        }
    }

    #[getter]
    fn satellite_id(&self) -> u8 {
        self.inner.satellite_id
    }
    #[getter]
    fn signal_id(&self) -> u8 {
        self.inner.signal_id
    }
    #[getter]
    fn fine_pseudorange(&self) -> i32 {
        self.inner.fine_pseudorange
    }
    #[getter]
    fn fine_phase_range(&self) -> i32 {
        self.inner.fine_phase_range
    }
    #[getter]
    fn lock_time_indicator(&self) -> u16 {
        self.inner.lock_time_indicator
    }
    #[getter]
    fn half_cycle_ambiguity(&self) -> bool {
        self.inner.half_cycle_ambiguity
    }
    #[getter]
    fn cnr(&self) -> u16 {
        self.inner.cnr
    }
    #[getter]
    fn fine_phase_range_rate(&self) -> Option<i16> {
        self.inner.fine_phase_range_rate
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmMsmSignal(satellite_id={}, signal_id={})",
            self.inner.satellite_id, self.inner.signal_id
        )
    }
}

/// A decoded MSM4 / MSM7 multi-signal observation message.
#[pyclass(module = "sidereon._sidereon", name = "RtcmMsmMessage")]
#[derive(Clone)]
pub struct PyRtcmMsmMessage {
    inner: MsmMessage,
}

#[pymethods]
impl PyRtcmMsmMessage {
    /// Construct an MSM4 / MSM7 observation message from its parts.
    ///
    /// `system` is the constellation single-letter identifier (e.g. `"G"`,
    /// `"R"`, `"E"`); `kind` is `"msm4"` or `"msm7"`. Satellites must be in
    /// ascending id order and signals in satellite-major then signal order, as
    /// the encoder expects. Raises `ValueError` for an unknown system letter or
    /// kind.
    #[new]
    #[pyo3(signature = (message_number, system, kind, header, satellites, signals))]
    fn new(
        py: Python<'_>,
        message_number: u16,
        system: &str,
        kind: &str,
        header: &PyRtcmMsmHeader,
        satellites: Vec<Py<PyRtcmMsmSatellite>>,
        signals: Vec<Py<PyRtcmMsmSignal>>,
    ) -> PyResult<Self> {
        let mut letters = system.chars();
        let (Some(letter), None) = (letters.next(), letters.next()) else {
            return Err(PyValueError::new_err(format!(
                "invalid RTCM MSM system {system:?}; expected a single letter"
            )));
        };
        let system = GnssSystem::from_letter(letter).ok_or_else(|| {
            PyValueError::new_err(format!("unknown RTCM MSM system letter {letter:?}"))
        })?;
        let kind = match kind {
            "msm4" => MsmKind::Msm4,
            "msm7" => MsmKind::Msm7,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown RTCM MSM kind {other:?}; expected \"msm4\" or \"msm7\""
                )))
            }
        };
        let satellites = satellites.iter().map(|s| s.borrow(py).inner).collect();
        let signals = signals.iter().map(|s| s.borrow(py).inner).collect();
        Ok(Self {
            inner: MsmMessage {
                message_number,
                system,
                kind,
                header: header.inner,
                satellites,
                signals,
            },
        })
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    /// Constellation single-letter identifier (e.g. `"G"`, `"R"`, `"E"`).
    #[getter]
    fn system(&self) -> String {
        self.inner.system.letter().to_string()
    }
    /// MSM variant: `"msm4"` or `"msm7"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            MsmKind::Msm4 => "msm4",
            MsmKind::Msm7 => "msm7",
        }
    }
    #[getter]
    fn header(&self) -> PyRtcmMsmHeader {
        PyRtcmMsmHeader {
            inner: self.inner.header,
        }
    }
    #[getter]
    fn satellites(&self) -> Vec<PyRtcmMsmSatellite> {
        self.inner
            .satellites
            .iter()
            .map(|&inner| PyRtcmMsmSatellite { inner })
            .collect()
    }
    #[getter]
    fn signals(&self) -> Vec<PyRtcmMsmSignal> {
        self.inner
            .signals
            .iter()
            .map(|&inner| PyRtcmMsmSignal { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmMsmMessage(message_number={}, system={:?}, kind={:?}, satellites={}, signals={})",
            self.inner.message_number,
            self.inner.system.letter(),
            self.kind(),
            self.inner.satellites.len(),
            self.inner.signals.len()
        )
    }
}

/// A recognized-but-undecoded message, preserved verbatim so the frame still
/// round-trips.
#[pyclass(module = "sidereon._sidereon", name = "RtcmUnsupportedMessage")]
#[derive(Clone)]
pub struct PyRtcmUnsupportedMessage {
    inner: UnsupportedMessage,
}

#[pymethods]
impl PyRtcmUnsupportedMessage {
    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    /// The undecoded message body as bytes.
    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.body)
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmUnsupportedMessage(message_number={}, body_len={})",
            self.inner.message_number,
            self.inner.body.len()
        )
    }
}

/// One decoded RTCM 3 message.
///
/// Inspect [`kind`](Self::kind) to discover the variant, then read the typed
/// payload via the matching accessor (`station_coordinates`, `antenna_descriptor`,
/// `gps_ephemeris`, `glonass_ephemeris`, `msm`, `unsupported`), each of which
/// returns `None` for a different variant. Re-encode with [`encode`](Self::encode)
/// (body bytes) or [`to_frame`](Self::to_frame) (a full transport frame).
#[pyclass(module = "sidereon._sidereon", name = "RtcmMessage")]
#[derive(Clone)]
pub struct PyRtcmMessage {
    inner: Message,
}

#[pymethods]
impl PyRtcmMessage {
    /// The RTCM message number this IR encodes to.
    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number()
    }

    /// Stable variant tag: `"station_coordinates"`, `"antenna_descriptor"`,
    /// `"gps_ephemeris"`, `"glonass_ephemeris"`, `"msm"`, or `"unsupported"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match &self.inner {
            Message::StationCoordinates(_) => "station_coordinates",
            Message::AntennaDescriptor(_) => "antenna_descriptor",
            Message::GpsEphemeris(_) => "gps_ephemeris",
            Message::GlonassEphemeris(_) => "glonass_ephemeris",
            Message::Msm(_) => "msm",
            Message::Unsupported(_) => "unsupported",
        }
    }

    /// The station-coordinates payload, or `None` for another variant.
    #[getter]
    fn station_coordinates(&self) -> Option<PyRtcmStationCoordinates> {
        match &self.inner {
            Message::StationCoordinates(value) => Some(PyRtcmStationCoordinates { inner: *value }),
            _ => None,
        }
    }

    /// The antenna-descriptor payload, or `None` for another variant.
    #[getter]
    fn antenna_descriptor(&self) -> Option<PyRtcmAntennaDescriptor> {
        match &self.inner {
            Message::AntennaDescriptor(value) => Some(PyRtcmAntennaDescriptor {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The GPS ephemeris payload, or `None` for another variant.
    #[getter]
    fn gps_ephemeris(&self) -> Option<PyRtcmGpsEphemeris> {
        match &self.inner {
            Message::GpsEphemeris(value) => Some(PyRtcmGpsEphemeris { inner: *value }),
            _ => None,
        }
    }

    /// The GLONASS ephemeris payload, or `None` for another variant.
    #[getter]
    fn glonass_ephemeris(&self) -> Option<PyRtcmGlonassEphemeris> {
        match &self.inner {
            Message::GlonassEphemeris(value) => Some(PyRtcmGlonassEphemeris { inner: *value }),
            _ => None,
        }
    }

    /// The MSM observation payload, or `None` for another variant.
    #[getter]
    fn msm(&self) -> Option<PyRtcmMsmMessage> {
        match &self.inner {
            Message::Msm(value) => Some(PyRtcmMsmMessage {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The verbatim unsupported-message payload, or `None` for another variant.
    #[getter]
    fn unsupported(&self) -> Option<PyRtcmUnsupportedMessage> {
        match &self.inner {
            Message::Unsupported(value) => Some(PyRtcmUnsupportedMessage {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// Wrap a 1005 / 1006 station-coordinates payload as a message.
    #[staticmethod]
    fn from_station_coordinates(payload: &PyRtcmStationCoordinates) -> Self {
        Self {
            inner: Message::StationCoordinates(payload.inner),
        }
    }

    /// Wrap a 1007 / 1008 / 1033 antenna-descriptor payload as a message.
    #[staticmethod]
    fn from_antenna_descriptor(payload: &PyRtcmAntennaDescriptor) -> Self {
        Self {
            inner: Message::AntennaDescriptor(payload.inner.clone()),
        }
    }

    /// Wrap a 1019 GPS ephemeris payload as a message.
    #[staticmethod]
    fn from_gps_ephemeris(payload: &PyRtcmGpsEphemeris) -> Self {
        Self {
            inner: Message::GpsEphemeris(payload.inner),
        }
    }

    /// Wrap a 1020 GLONASS ephemeris payload as a message.
    #[staticmethod]
    fn from_glonass_ephemeris(payload: &PyRtcmGlonassEphemeris) -> Self {
        Self {
            inner: Message::GlonassEphemeris(payload.inner),
        }
    }

    /// Wrap an MSM4 / MSM7 observation payload as a message.
    #[staticmethod]
    fn from_msm(payload: &PyRtcmMsmMessage) -> Self {
        Self {
            inner: Message::Msm(payload.inner.clone()),
        }
    }

    /// Encode this message back into a body (without the transport frame).
    fn encode<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.encode())
    }

    /// Encode this message and wrap it in a fresh RTCM transport frame.
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let frame = self.inner.to_frame().map_err(to_rtcm_err)?;
        Ok(PyBytes::new(py, &frame))
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmMessage(message_number={}, kind={:?})",
            self.inner.message_number(),
            self.kind()
        )
    }
}

/// Decode every CRC-valid frame in a byte buffer into the message IR.
///
/// Frames whose CRC fails, or whose body cannot be decoded, are skipped and the
/// scan resynchronizes on the next preamble. This is the forgiving stream entry
/// point for a noisy serial feed.
#[pyfunction]
fn decode_rtcm(data: &[u8]) -> Vec<PyRtcmMessage> {
    core_decode_messages(data)
        .into_iter()
        .map(|inner| PyRtcmMessage { inner })
        .collect()
}

/// Decode a single RTCM 3 message body (the bytes between a frame's length word
/// and its CRC). Raises `RtcmParseError` on a truncated body of a recognized
/// type; an unrecognized message number decodes to the `"unsupported"` variant.
#[pyfunction]
fn decode_rtcm_message(body: &[u8]) -> PyResult<PyRtcmMessage> {
    Message::decode(body)
        .map(|inner| PyRtcmMessage { inner })
        .map_err(to_rtcm_err)
}

/// Wrap a message body in an RTCM transport frame (preamble, length, CRC-24Q).
/// Raises `RtcmParseError` if the body exceeds the frame length limit.
#[pyfunction]
fn encode_rtcm_frame<'py>(py: Python<'py>, body: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    let frame = core_encode_frame(body).map_err(to_rtcm_err)?;
    Ok(PyBytes::new(py, &frame))
}

/// Read the 12-bit RTCM message number from the start of a message body. Raises
/// `RtcmParseError` if the body is shorter than 12 bits.
#[pyfunction]
fn rtcm_message_number(body: &[u8]) -> PyResult<u16> {
    core_message_number(body).map_err(to_rtcm_err)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRtcmStationCoordinates>()?;
    m.add_class::<PyRtcmAntennaDescriptor>()?;
    m.add_class::<PyRtcmGpsEphemeris>()?;
    m.add_class::<PyRtcmGlonassEphemeris>()?;
    m.add_class::<PyRtcmMsmHeader>()?;
    m.add_class::<PyRtcmMsmSatellite>()?;
    m.add_class::<PyRtcmMsmSignal>()?;
    m.add_class::<PyRtcmMsmMessage>()?;
    m.add_class::<PyRtcmUnsupportedMessage>()?;
    m.add_class::<PyRtcmMessage>()?;
    m.add_function(wrap_pyfunction!(decode_rtcm, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rtcm_message, m)?)?;
    m.add_function(wrap_pyfunction!(encode_rtcm_frame, m)?)?;
    m.add_function(wrap_pyfunction!(rtcm_message_number, m)?)?;
    Ok(())
}
