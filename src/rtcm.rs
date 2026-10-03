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
use pyo3::types::{PyBytes, PyDict, PyModule};
use pyo3::Bound;

use sidereon_core::rtcm::{
    decode_messages as core_decode_messages, decode_stream_with_policy as core_decode_stream,
    derive_lli as core_derive_lli, encode_frame as core_encode_frame,
    message_number as core_message_number, minimum_lock_time_ms as core_minimum_lock_time_ms,
    msm_epoch_dt_ms as core_msm_epoch_dt_ms, msm_signal_mask as core_msm_signal_mask,
    msm_signal_rinex_code as core_msm_signal_rinex_code, AntennaDescriptor, BeidouEphemeris,
    CellLli, FkpGradient, FkpGradients, FrameSkip, FrameSkipReason, GalileoFnavEphemeris,
    GalileoInavEphemeris, GlonassCodePhaseBiases, GlonassEphemeris, GpsEphemeris, GridResidual,
    HelmertTransformation, LegacyL1, LegacyL2, LegacyObservations, LegacySatellite,
    LockTimeTracker, Message, MessageAnnouncement, MsmHeader, MsmKind, MsmMessage, MsmSatellite,
    MsmSignal, NavicEphemeris, NetworkAuxiliaryStation, NetworkCorrectionDifference,
    NetworkCorrectionDifferences, NetworkResidual, NetworkResiduals, PhysicalReferenceStation,
    PreviousLock, Projection, ProjectionParameters, QzssEphemeris, ResidualGrid, RotationPoint,
    RtcmDeparture, RtcmEncodeError as CoreRtcmEncodeError, RtcmFieldEncoding, RtcmPolicy,
    RtcmRecordKind, SsrMessage, SsrVtecEvaluation, SsrVtecLayer, SsrVtecLayerEvaluation,
    SsrVtecMessage, StationCoordinates, StreamDiagnostics, SystemParameters, TextMessage,
    UnsupportedMessage, LLI_HALF_CYCLE, LLI_LOSS_OF_LOCK, MSM4_FINE_PHASE_RANGE_INVALID,
    MSM4_FINE_PSEUDORANGE_INVALID, MSM7_FINE_PHASE_RANGE_INVALID, MSM7_FINE_PSEUDORANGE_INVALID,
    MSM_FINE_PHASE_RANGE_RATE_INVALID, MSM_ROUGH_PHASE_RANGE_RATE_INVALID, MSM_ROUGH_RANGE_INVALID,
};

use pyo3::exceptions::{PyTypeError, PyValueError};
use sidereon_core::GnssSystem;

use crate::RtcmParseError;

type PyNetworkCorrectionSatellite = (u8, u8, u8, Option<i32>, Option<u8>, Option<i32>);
type PyProjectionObliqueParameters = (bool, i64, i64, u64, i32, u32, u64, i64);
type PySsrHeader = (
    u32,
    u8,
    bool,
    u8,
    u16,
    u8,
    Option<bool>,
    Option<bool>,
    Option<bool>,
    u8,
);
type PySsrOrbitRow = (u8, u32, Option<u32>, i32, i32, i32, i32, i32, i32);
type PySsrPhaseBiasRow = (u8, u16, i8, Vec<(u8, u8, u8, u8, i32)>);

#[pyclass(module = "sidereon._sidereon", name = "RtcmNavicEphemeris")]
#[derive(Clone)]
pub struct PyRtcmNavicEphemeris {
    inner: NavicEphemeris,
}

#[pymethods]
impl PyRtcmNavicEphemeris {
    #[new]
    #[pyo3(signature = (
        satellite_id, week_number, a_f0, a_f1, a_f2, ura, t_oc, t_gd, delta_n, iodec,
        reserved, l5_flag, s_flag, c_uc, c_us, c_ic, c_is, c_rc, c_rs, idot, m0, t_oe,
        eccentricity, sqrt_a, omega0, omega, omega_dot, i0, spare_df544, spare_df545,
        trailing_bits=Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        satellite_id: u8,
        week_number: u16,
        a_f0: i32,
        a_f1: i32,
        a_f2: i16,
        ura: u8,
        t_oc: u16,
        t_gd: i16,
        delta_n: i32,
        iodec: u8,
        reserved: u16,
        l5_flag: bool,
        s_flag: bool,
        c_uc: i32,
        c_us: i32,
        c_ic: i32,
        c_is: i32,
        c_rc: i32,
        c_rs: i32,
        idot: i32,
        m0: i64,
        t_oe: u16,
        eccentricity: u64,
        sqrt_a: u64,
        omega0: i64,
        omega: i64,
        omega_dot: i32,
        i0: i64,
        spare_df544: u8,
        spare_df545: u8,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: NavicEphemeris {
                satellite_id,
                week_number,
                a_f0,
                a_f1,
                a_f2,
                ura,
                t_oc,
                t_gd,
                delta_n,
                iodec,
                reserved,
                l5_flag,
                s_flag,
                c_uc,
                c_us,
                c_ic,
                c_is,
                c_rc,
                c_rs,
                idot,
                m0,
                t_oe,
                eccentricity,
                sqrt_a,
                omega0,
                omega,
                omega_dot,
                i0,
                spare_df544,
                spare_df545,
                trailing_bits,
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
    fn a_f0(&self) -> i32 {
        self.inner.a_f0
    }
    #[getter]
    fn a_f1(&self) -> i32 {
        self.inner.a_f1
    }
    #[getter]
    fn a_f2(&self) -> i16 {
        self.inner.a_f2
    }
    #[getter]
    fn ura(&self) -> u8 {
        self.inner.ura
    }
    #[getter]
    fn t_oc(&self) -> u16 {
        self.inner.t_oc
    }
    #[getter]
    fn t_gd(&self) -> i16 {
        self.inner.t_gd
    }
    #[getter]
    fn delta_n(&self) -> i32 {
        self.inner.delta_n
    }
    #[getter]
    fn iodec(&self) -> u8 {
        self.inner.iodec
    }
    #[getter]
    fn reserved(&self) -> u16 {
        self.inner.reserved
    }
    #[getter]
    fn l5_flag(&self) -> bool {
        self.inner.l5_flag
    }
    #[getter]
    fn s_flag(&self) -> bool {
        self.inner.s_flag
    }
    #[getter]
    fn c_uc(&self) -> i32 {
        self.inner.c_uc
    }
    #[getter]
    fn c_us(&self) -> i32 {
        self.inner.c_us
    }
    #[getter]
    fn c_ic(&self) -> i32 {
        self.inner.c_ic
    }
    #[getter]
    fn c_is(&self) -> i32 {
        self.inner.c_is
    }
    #[getter]
    fn c_rc(&self) -> i32 {
        self.inner.c_rc
    }
    #[getter]
    fn c_rs(&self) -> i32 {
        self.inner.c_rs
    }
    #[getter]
    fn idot(&self) -> i32 {
        self.inner.idot
    }
    #[getter]
    fn m0(&self) -> i64 {
        self.inner.m0
    }
    #[getter]
    fn t_oe(&self) -> u16 {
        self.inner.t_oe
    }
    #[getter]
    fn eccentricity(&self) -> u64 {
        self.inner.eccentricity
    }
    #[getter]
    fn sqrt_a(&self) -> u64 {
        self.inner.sqrt_a
    }
    #[getter]
    fn omega0(&self) -> i64 {
        self.inner.omega0
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
    fn i0(&self) -> i64 {
        self.inner.i0
    }
    #[getter]
    fn spare_df544(&self) -> u8 {
        self.inner.spare_df544
    }
    #[getter]
    fn spare_df545(&self) -> u8 {
        self.inner.spare_df545
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
    fn satellite(&self, py: Python<'_>) -> PyResult<String> {
        self.inner
            .satellite()
            .map(|satellite| satellite.to_string())
            .map_err(|error| to_rtcm_conversion_err(py, error))
    }
    #[getter]
    fn health(&self) -> u8 {
        self.inner.health()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmGlonassCodePhaseBiases")]
#[derive(Clone)]
pub struct PyRtcmGlonassCodePhaseBiases {
    inner: GlonassCodePhaseBiases,
}

#[pymethods]
impl PyRtcmGlonassCodePhaseBiases {
    #[new]
    #[pyo3(signature = (
        reference_station_id, aligned, reserved, l1_ca=None, l1_p=None, l2_ca=None,
        l2_p=None, trailing_bits=Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        reference_station_id: u16,
        aligned: bool,
        reserved: u8,
        l1_ca: Option<i16>,
        l1_p: Option<i16>,
        l2_ca: Option<i16>,
        l2_p: Option<i16>,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: GlonassCodePhaseBiases {
                reference_station_id,
                aligned,
                reserved,
                l1_ca,
                l1_p,
                l2_ca,
                l2_p,
                trailing_bits,
            },
        }
    }

    #[getter]
    fn reference_station_id(&self) -> u16 {
        self.inner.reference_station_id
    }
    #[getter]
    fn aligned(&self) -> bool {
        self.inner.aligned
    }
    #[getter]
    fn reserved(&self) -> u8 {
        self.inner.reserved
    }
    #[getter]
    fn l1_ca(&self) -> Option<i16> {
        self.inner.l1_ca
    }
    #[getter]
    fn l1_p(&self) -> Option<i16> {
        self.inner.l1_p
    }
    #[getter]
    fn l2_ca(&self) -> Option<i16> {
        self.inner.l2_ca
    }
    #[getter]
    fn l2_p(&self) -> Option<i16> {
        self.inner.l2_p
    }
    #[getter]
    fn signal_mask(&self) -> u8 {
        self.inner.signal_mask()
    }
    #[getter]
    fn biases_m(&self) -> [Option<f64>; 4] {
        self.inner.biases_m()
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

fn to_rtcm_err<E: std::fmt::Display>(err: E) -> PyErr {
    RtcmParseError::new_err(err.to_string())
}

fn ssr_kind_name(kind: sidereon_core::rtcm::SsrKind) -> &'static str {
    use sidereon_core::rtcm::SsrKind as Kind;
    match kind {
        Kind::Orbit => "orbit",
        Kind::Clock => "clock",
        Kind::CombinedOrbitClock => "combined_orbit_clock",
        Kind::CodeBias => "code_bias",
        Kind::PhaseBias => "phase_bias",
        Kind::Ura => "ura",
        Kind::HighRateClock => "high_rate_clock",
    }
}

fn encode_error_kind(error: &CoreRtcmEncodeError) -> &'static str {
    use CoreRtcmEncodeError as Encode;
    match error {
        Encode::FieldOutOfRange { .. } => "field_out_of_range",
        Encode::NegativeZeroWithValue { .. } => "negative_zero_with_value",
        Encode::NegativeZeroMask { .. } => "negative_zero_mask",
        Encode::MessageNumber { .. } => "message_number",
        Encode::FieldPresence { .. } => "field_presence",
        Encode::SatelliteFieldPresence { .. } => "satellite_field_presence",
        Encode::CountMismatch { .. } => "count_mismatch",
        Encode::ValueOutOfRange { .. } => "value_out_of_range",
        Encode::NonLatin1Character { .. } => "non_latin1_character",
        Encode::SatelliteIdOutOfRange { .. } => "satellite_id_out_of_range",
        Encode::SsrSatelliteIdOutOfRange { .. } => "ssr_satellite_id_out_of_range",
        Encode::SsrRecordsNotCarried { .. } => "ssr_records_not_carried",
        Encode::SsrCombinedRecordCounts { .. } => "ssr_combined_record_counts",
        Encode::SsrCombinedSatelliteMismatch { .. } => "ssr_combined_satellite_mismatch",
        Encode::SsrHighRateClockTerms { .. } => "ssr_high_rate_clock_terms",
        Encode::SsrSatelliteCount { .. } => "ssr_satellite_count",
        Encode::MsmMask { .. } => "msm_mask",
        Encode::MsmOptional { .. } => "msm_optional",
        Encode::TrailingZeroBits { .. } => "trailing_zero_bits",
        Encode::StrictDeparture(_) => "strict_departure",
        Encode::UnsupportedBodyTooShort { .. } => "unsupported_body_too_short",
        Encode::UnsupportedBodyNumber { .. } => "unsupported_body_number",
        Encode::UnsupportedDecodedNumber { .. } => "unsupported_decoded_number",
        Encode::FrameBodyTooLong { .. } => "frame_body_too_long",
        Encode::FrameReservedOutOfRange { .. } => "frame_reserved_out_of_range",
        _ => "other",
    }
}

fn record_kind_details(py: Python<'_>, record: RtcmRecordKind) -> Bound<'_, PyDict> {
    let details = PyDict::new(py);
    match record {
        RtcmRecordKind::StationCoordinates => {
            let _ = details.set_item("kind", "station_coordinates");
        }
        RtcmRecordKind::AntennaDescriptor => {
            let _ = details.set_item("kind", "antenna_descriptor");
        }
        RtcmRecordKind::Msm { system, kind } => {
            let _ = details.set_item("kind", "msm");
            let _ = details.set_item("system", system.letter().to_string());
            let _ = details.set_item("message_kind", format!("msm{}", kind.number()));
        }
        RtcmRecordKind::Ssr { system, kind } => {
            let _ = details.set_item("kind", "ssr");
            let _ = details.set_item("system", system.letter().to_string());
            let _ = details.set_item("message_kind", ssr_kind_name(kind));
        }
        RtcmRecordKind::LegacyObservations => {
            let _ = details.set_item("kind", "legacy_observations");
        }
        RtcmRecordKind::SystemParameters => {
            let _ = details.set_item("kind", "system_parameters");
        }
        RtcmRecordKind::Text => {
            let _ = details.set_item("kind", "text");
        }
        RtcmRecordKind::Network { family } => {
            let _ = details.set_item("kind", "network");
            let _ = details.set_item("family", family);
        }
        RtcmRecordKind::Transformation { family } => {
            let _ = details.set_item("kind", "transformation");
            let _ = details.set_item("family", family);
        }
        RtcmRecordKind::GlonassCodePhaseBiases => {
            let _ = details.set_item("kind", "glonass_code_phase_biases");
        }
        RtcmRecordKind::SsrVtec { message_number } => {
            let _ = details.set_item("kind", "ssr_vtec");
            let _ = details.set_item("message_number", message_number);
        }
        _ => {
            let _ = details.set_item("kind", "other");
            let _ = details.set_item("debug", format!("{record:?}"));
        }
    }
    details
}

fn departure_details(py: Python<'_>, departure: RtcmDeparture) -> Bound<'_, PyDict> {
    let details = PyDict::new(py);
    match departure {
        RtcmDeparture::FrameReservedBits { reserved } => {
            let _ = details.set_item("kind", "frame_reserved_bits");
            let _ = details.set_item("reserved", reserved);
        }
        RtcmDeparture::TrailingBits {
            message_number,
            bits,
        } => {
            let _ = details.set_item("kind", "trailing_bits");
            let _ = details.set_item("message_number", message_number);
            let _ = details.set_item("bits", bits);
        }
        RtcmDeparture::MsmCellMaskOver64 {
            message_number,
            cells,
        } => {
            let _ = details.set_item("kind", "msm_cell_mask_over_64");
            let _ = details.set_item("message_number", message_number);
            let _ = details.set_item("cells", cells);
        }
        RtcmDeparture::OrderExceedsDegree {
            message_number,
            layer_index,
            degree,
            order,
        } => {
            let _ = details.set_item("kind", "order_exceeds_degree");
            let _ = details.set_item("message_number", message_number);
            let _ = details.set_item("layer_index", layer_index);
            let _ = details.set_item("degree", degree);
            let _ = details.set_item("order", order);
        }
        RtcmDeparture::SsrRecordsShort {
            message_number,
            declared,
            read,
        } => {
            let _ = details.set_item("kind", "ssr_records_short");
            let _ = details.set_item("message_number", message_number);
            let _ = details.set_item("declared", declared);
            let _ = details.set_item("read", read);
        }
        RtcmDeparture::RecordsShort {
            message_number,
            declared,
            read,
        } => {
            let _ = details.set_item("kind", "records_short");
            let _ = details.set_item("message_number", message_number);
            let _ = details.set_item("declared", declared);
            let _ = details.set_item("read", read);
        }
        _ => {
            let _ = details.set_item("kind", "other");
            let _ = details.set_item("debug", format!("{departure:?}"));
        }
    }
    details
}

fn msm_problem_details(
    py: Python<'_>,
    problem: sidereon_core::rtcm::MsmMaskProblem,
) -> Bound<'_, PyDict> {
    let details = PyDict::new(py);
    use sidereon_core::rtcm::MsmMaskProblem as Problem;
    match problem {
        Problem::SatelliteOutsideMask { satellite } => {
            let _ = details.set_item("kind", "satellite_outside_mask");
            let _ = details.set_item("satellite", satellite);
        }
        Problem::SatelliteListedTwice { satellite } => {
            let _ = details.set_item("kind", "satellite_listed_twice");
            let _ = details.set_item("satellite", satellite);
        }
        Problem::SignalOutsideMask { signal } => {
            let _ = details.set_item("kind", "signal_outside_mask");
            let _ = details.set_item("signal", signal);
        }
        Problem::SignalNotInMask { signal, mask } => {
            let _ = details.set_item("kind", "signal_not_in_mask");
            let _ = details.set_item("signal", signal);
            let _ = details.set_item("mask", mask);
        }
        Problem::SignalSatelliteNotListed { signal, satellite } => {
            let _ = details.set_item("kind", "signal_satellite_not_listed");
            let _ = details.set_item("signal", signal);
            let _ = details.set_item("satellite", satellite);
        }
        Problem::CellListedTwice { satellite, signal } => {
            let _ = details.set_item("kind", "cell_listed_twice");
            let _ = details.set_item("satellite", satellite);
            let _ = details.set_item("signal", signal);
        }
        _ => {
            let _ = details.set_item("kind", "other");
            let _ = details.set_item("debug", format!("{problem:?}"));
        }
    }
    details
}

fn to_rtcm_encode_detail(py: Python<'_>, error: CoreRtcmEncodeError) -> PyErr {
    let kind = encode_error_kind(&error);
    let fields = PyDict::new(py);
    use CoreRtcmEncodeError as Encode;
    match &error {
        Encode::FieldOutOfRange {
            message_number,
            field,
            value,
            width,
            encoding,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("width", width);
            let encoding_name = match encoding {
                RtcmFieldEncoding::Unsigned => "unsigned",
                RtcmFieldEncoding::TwosComplement => "twos_complement",
                RtcmFieldEncoding::SignMagnitude => "sign_magnitude",
                _ => "other",
            };
            let _ = fields.set_item("encoding", encoding_name);
        }
        Encode::NegativeZeroWithValue {
            message_number,
            field,
            value,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("value", value);
        }
        Encode::NegativeZeroMask {
            message_number,
            mask,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("mask", mask);
        }
        Encode::MessageNumber {
            message_number,
            record,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("record", record_kind_details(py, *record));
        }
        Encode::FieldPresence {
            message_number,
            record,
            field,
            carried,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("record", record_kind_details(py, *record));
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("carried", carried);
        }
        Encode::SatelliteFieldPresence {
            message_number,
            record,
            satellite,
            field,
            carried,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("record", record_kind_details(py, *record));
            let _ = fields.set_item("satellite", satellite);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("carried", carried);
        }
        Encode::CountMismatch {
            message_number,
            field,
            expected,
            actual,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("expected", expected);
            let _ = fields.set_item("actual", actual);
        }
        Encode::ValueOutOfRange {
            message_number,
            field,
            value,
            minimum,
            maximum,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("minimum", minimum);
            let _ = fields.set_item("maximum", maximum);
        }
        Encode::NonLatin1Character { field, character } => {
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("character", character.to_string());
            let _ = fields.set_item("codepoint", u32::from(*character));
        }
        Encode::SatelliteIdOutOfRange {
            message_number,
            field,
            value,
            width,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("width", width);
        }
        Encode::SsrSatelliteIdOutOfRange {
            message_number,
            value,
            width,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("width", width);
        }
        Encode::SsrRecordsNotCarried {
            message_number,
            kind,
            records,
            count,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("ssr_kind", ssr_kind_name(*kind));
            let _ = fields.set_item("records", records);
            let _ = fields.set_item("count", count);
        }
        Encode::SsrCombinedRecordCounts {
            message_number,
            orbit,
            clock,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("orbit", orbit);
            let _ = fields.set_item("clock", clock);
        }
        Encode::SsrCombinedSatelliteMismatch {
            message_number,
            index,
            orbit_satellite,
            clock_satellite,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("index", index);
            let _ = fields.set_item("orbit_satellite", orbit_satellite);
            let _ = fields.set_item("clock_satellite", clock_satellite);
        }
        Encode::SsrHighRateClockTerms {
            message_number,
            satellite,
            c1,
            c2,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("satellite", satellite);
            let _ = fields.set_item("c1", c1);
            let _ = fields.set_item("c2", c2);
        }
        Encode::SsrSatelliteCount {
            message_number,
            declared,
            records,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("declared", declared);
            let _ = fields.set_item("records", records);
        }
        Encode::MsmMask {
            message_number,
            problem,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("problem", msm_problem_details(py, *problem));
        }
        Encode::MsmOptional {
            message_number,
            kind,
            satellite,
            signal,
            field,
            problem,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("msm_kind", format!("msm{}", kind.number()));
            let _ = fields.set_item("satellite", satellite);
            let _ = fields.set_item("signal", signal);
            let field_name = match field {
                sidereon_core::rtcm::MsmOptionalField::ExtendedInfo => "extended_info",
                sidereon_core::rtcm::MsmOptionalField::RoughPhaseRangeRate => {
                    "rough_phase_range_rate"
                }
                sidereon_core::rtcm::MsmOptionalField::FinePhaseRangeRate => {
                    "fine_phase_range_rate"
                }
                _ => "other",
            };
            let _ = fields.set_item("field", field_name);
            let (problem_name, invalid_value) = match problem {
                sidereon_core::rtcm::MsmOptionalProblem::Missing => ("missing", None),
                sidereon_core::rtcm::MsmOptionalProblem::NotCarried => ("not_carried", None),
                sidereon_core::rtcm::MsmOptionalProblem::InvalidValue(value) => {
                    ("invalid_value", Some(*value))
                }
                _ => ("other", None),
            };
            let _ = fields.set_item("problem", problem_name);
            if let Some(value) = invalid_value {
                let _ = fields.set_item("invalid_value", value);
            }
        }
        Encode::TrailingZeroBits {
            message_number,
            bits,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("bits", bits);
        }
        Encode::StrictDeparture(departure) => {
            let _ = fields.set_item("departure", departure_details(py, departure.clone()));
        }
        Encode::UnsupportedBodyTooShort { message_number }
        | Encode::UnsupportedDecodedNumber { message_number } => {
            let _ = fields.set_item("message_number", message_number);
        }
        Encode::UnsupportedBodyNumber {
            message_number,
            carried,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("carried", carried);
        }
        Encode::FrameBodyTooLong { len } => {
            let _ = fields.set_item("len", len);
        }
        Encode::FrameReservedOutOfRange { value } => {
            let _ = fields.set_item("value", value);
        }
        _ => {
            let _ = fields.set_item("core_debug", format!("{error:?}"));
        }
    }
    let exception = match crate::rtcm_encode_error_type(py) {
        Ok(error_type) => PyErr::from_type(error_type, error.to_string()),
        Err(type_error) => return type_error,
    };
    let exception_value = exception.value(py);
    let _ = exception_value.setattr("kind", kind);
    let _ = exception_value.setattr("details", fields);
    exception
}

#[cfg(test)]
mod rtcm_encode_error_contract_tests {
    use super::to_rtcm_encode_detail;
    use pyo3::prelude::*;
    use sidereon_core::rtcm::{
        MsmKind, MsmMaskProblem, MsmOptionalField, MsmOptionalProblem, RtcmDeparture,
        RtcmEncodeError as Encode, RtcmFieldEncoding, RtcmRecordKind, SsrKind,
    };

    #[test]
    fn every_rtcm_encode_variant_and_field_has_the_documented_python_exception() {
        Python::with_gil(|py| {
            let cases = vec![
                (
                    Encode::FieldOutOfRange {
                        message_number: 1005,
                        field: "ecef_x".into(),
                        value: -2,
                        width: 38,
                        encoding: RtcmFieldEncoding::TwosComplement,
                    },
                    "field_out_of_range",
                    serde_json::json!({"message_number":1005,"field":"ecef_x","value":-2,"width":38,"encoding":"twos_complement"}),
                ),
                (
                    Encode::NegativeZeroWithValue {
                        message_number: 1020,
                        field: "tau_n".into(),
                        value: 7,
                    },
                    "negative_zero_with_value",
                    serde_json::json!({"message_number":1020,"field":"tau_n","value":7}),
                ),
                (
                    Encode::NegativeZeroMask {
                        message_number: 1020,
                        mask: 9,
                    },
                    "negative_zero_mask",
                    serde_json::json!({"message_number":1020,"mask":9}),
                ),
                (
                    Encode::MessageNumber {
                        message_number: 999,
                        record: RtcmRecordKind::StationCoordinates,
                    },
                    "message_number",
                    serde_json::json!({"message_number":999,"record":{"kind":"station_coordinates"}}),
                ),
                (
                    Encode::FieldPresence {
                        message_number: 1005,
                        record: RtcmRecordKind::StationCoordinates,
                        field: "antenna_height",
                        carried: false,
                    },
                    "field_presence",
                    serde_json::json!({"message_number":1005,"record":{"kind":"station_coordinates"},"field":"antenna_height","carried":false}),
                ),
                (
                    Encode::SatelliteFieldPresence {
                        message_number: 1074,
                        record: RtcmRecordKind::Msm {
                            system: sidereon_core::GnssSystem::Gps,
                            kind: MsmKind::Msm4,
                        },
                        satellite: 7,
                        field: "extended_info",
                        carried: false,
                    },
                    "satellite_field_presence",
                    serde_json::json!({"message_number":1074,"record":{"kind":"msm","system":"G","message_kind":"msm4"},"satellite":7,"field":"extended_info","carried":false}),
                ),
                (
                    Encode::CountMismatch {
                        message_number: 1015,
                        field: "satellites",
                        expected: 2,
                        actual: 1,
                    },
                    "count_mismatch",
                    serde_json::json!({"message_number":1015,"field":"satellites","expected":2,"actual":1}),
                ),
                (
                    Encode::ValueOutOfRange {
                        message_number: 1005,
                        field: "itrf".into(),
                        value: 64,
                        minimum: 0,
                        maximum: 63,
                    },
                    "value_out_of_range",
                    serde_json::json!({"message_number":1005,"field":"itrf","value":64,"minimum":0,"maximum":63}),
                ),
                (
                    Encode::NonLatin1Character {
                        field: "descriptor".into(),
                        character: 'λ',
                    },
                    "non_latin1_character",
                    serde_json::json!({"field":"descriptor","character":"λ","codepoint":955}),
                ),
                (
                    Encode::SatelliteIdOutOfRange {
                        message_number: 1019,
                        field: "GPS PRN",
                        value: 64,
                        width: 6,
                    },
                    "satellite_id_out_of_range",
                    serde_json::json!({"message_number":1019,"field":"GPS PRN","value":64,"width":6}),
                ),
                (
                    Encode::SsrSatelliteIdOutOfRange {
                        message_number: 1057,
                        value: 64,
                        width: 6,
                    },
                    "ssr_satellite_id_out_of_range",
                    serde_json::json!({"message_number":1057,"value":64,"width":6}),
                ),
                (
                    Encode::SsrRecordsNotCarried {
                        message_number: 1058,
                        kind: SsrKind::Clock,
                        records: "orbit",
                        count: 2,
                    },
                    "ssr_records_not_carried",
                    serde_json::json!({"message_number":1058,"ssr_kind":"clock","records":"orbit","count":2}),
                ),
                (
                    Encode::SsrCombinedRecordCounts {
                        message_number: 1060,
                        orbit: 2,
                        clock: 1,
                    },
                    "ssr_combined_record_counts",
                    serde_json::json!({"message_number":1060,"orbit":2,"clock":1}),
                ),
                (
                    Encode::SsrCombinedSatelliteMismatch {
                        message_number: 1060,
                        index: 1,
                        orbit_satellite: 4,
                        clock_satellite: 5,
                    },
                    "ssr_combined_satellite_mismatch",
                    serde_json::json!({"message_number":1060,"index":1,"orbit_satellite":4,"clock_satellite":5}),
                ),
                (
                    Encode::SsrHighRateClockTerms {
                        message_number: 1062,
                        satellite: 3,
                        c1: -4,
                        c2: 5,
                    },
                    "ssr_high_rate_clock_terms",
                    serde_json::json!({"message_number":1062,"satellite":3,"c1":-4,"c2":5}),
                ),
                (
                    Encode::SsrSatelliteCount {
                        message_number: 1057,
                        declared: 2,
                        records: 1,
                    },
                    "ssr_satellite_count",
                    serde_json::json!({"message_number":1057,"declared":2,"records":1}),
                ),
                (
                    Encode::MsmMask {
                        message_number: 1074,
                        problem: MsmMaskProblem::SignalNotInMask { signal: 3, mask: 5 },
                    },
                    "msm_mask",
                    serde_json::json!({"message_number":1074,"problem":{"kind":"signal_not_in_mask","signal":3,"mask":5}}),
                ),
                (
                    Encode::MsmOptional {
                        message_number: 1077,
                        kind: MsmKind::Msm7,
                        satellite: 4,
                        signal: Some(6),
                        field: MsmOptionalField::FinePhaseRangeRate,
                        problem: MsmOptionalProblem::InvalidValue(-16384),
                    },
                    "msm_optional",
                    serde_json::json!({"message_number":1077,"msm_kind":"msm7","satellite":4,"signal":6,"field":"fine_phase_range_rate","problem":"invalid_value","invalid_value":-16384}),
                ),
                (
                    Encode::TrailingZeroBits {
                        message_number: 1006,
                        bits: 3,
                    },
                    "trailing_zero_bits",
                    serde_json::json!({"message_number":1006,"bits":3}),
                ),
                (
                    Encode::StrictDeparture(RtcmDeparture::FrameReservedBits { reserved: 5 }),
                    "strict_departure",
                    serde_json::json!({"departure":{"kind":"frame_reserved_bits","reserved":5}}),
                ),
                (
                    Encode::UnsupportedBodyTooShort {
                        message_number: 4090,
                    },
                    "unsupported_body_too_short",
                    serde_json::json!({"message_number":4090}),
                ),
                (
                    Encode::UnsupportedBodyNumber {
                        message_number: 4090,
                        carried: 4089,
                    },
                    "unsupported_body_number",
                    serde_json::json!({"message_number":4090,"carried":4089}),
                ),
                (
                    Encode::UnsupportedDecodedNumber {
                        message_number: 4090,
                    },
                    "unsupported_decoded_number",
                    serde_json::json!({"message_number":4090}),
                ),
                (
                    Encode::FrameBodyTooLong { len: 1024 },
                    "frame_body_too_long",
                    serde_json::json!({"len":1024}),
                ),
                (
                    Encode::FrameReservedOutOfRange { value: 64 },
                    "frame_reserved_out_of_range",
                    serde_json::json!({"value":64}),
                ),
            ];

            assert_eq!(cases.len(), 25);
            let json = py.import("json").expect("Python json module");
            for (error, expected_kind, expected_details) in cases {
                let expected_message = error.to_string();
                let exception = to_rtcm_encode_detail(py, error);
                let value = exception.value(py);
                let kind: String = value
                    .getattr("kind")
                    .expect("kind attribute")
                    .extract()
                    .expect("kind string");
                assert_eq!(kind, expected_kind);
                let details = value.getattr("details").expect("details attribute");
                let encoded: String = json
                    .call_method1("dumps", (&details,))
                    .expect("JSON serializable details")
                    .extract()
                    .expect("JSON string");
                let actual_details: serde_json::Value =
                    serde_json::from_str(&encoded).expect("valid detail JSON");
                assert_eq!(actual_details, expected_details, "{expected_kind}");
                assert_eq!(
                    value.str().expect("exception text").to_string_lossy(),
                    expected_message
                );
            }
        });
    }
}

pub(crate) fn to_rtcm_encode_err(py: Python<'_>, error: sidereon_core::Error) -> PyErr {
    match error {
        sidereon_core::Error::RtcmEncode(encode_error) => to_rtcm_encode_detail(py, *encode_error),
        sidereon_core::Error::SbasEncode(encode_error) => to_sbas_encode_detail(py, *encode_error),
        other => to_rtcm_err(other),
    }
}

fn to_sbas_encode_detail(py: Python<'_>, error: sidereon_core::sbas::SbasEncodeError) -> PyErr {
    use sidereon_core::sbas::SbasEncodeError as Encode;
    let fields = PyDict::new(py);
    let kind = match &error {
        Encode::FieldOutOfRange {
            message_type,
            field,
            index,
            value,
            width,
            signed,
        } => {
            let _ = fields.set_item("message_type", message_type);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("index", index);
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("width", width);
            let _ = fields.set_item("signed", signed);
            "sbas_field_out_of_range"
        }
        Encode::UnrecognizedPreamble { preamble } => {
            let _ = fields.set_item("preamble", preamble);
            "sbas_unrecognized_preamble"
        }
        Encode::MessageType {
            message_type,
            reason,
        } => {
            let _ = fields.set_item("message_type", message_type);
            let _ = fields.set_item("reason", reason);
            "sbas_message_type"
        }
        Encode::RawPayload {
            message_type,
            bytes,
            bits_past_payload,
        } => {
            let _ = fields.set_item("message_type", message_type);
            let _ = fields.set_item("bytes", bytes);
            let _ = fields.set_item("bits_past_payload", bits_past_payload);
            "sbas_raw_payload"
        }
        Encode::ReservedLayout {
            message_type,
            part,
            expected,
            found,
        } => {
            let _ = fields.set_item("message_type", message_type);
            let _ = fields.set_item("part", part);
            let _ = fields.set_item("expected", expected);
            let _ = fields.set_item("found", found);
            "sbas_reserved_layout"
        }
        Encode::LongTermRecordCount {
            message_type,
            half,
            velocity_code,
            expected,
            found,
        } => {
            let _ = fields.set_item("message_type", message_type);
            let _ = fields.set_item("half", half);
            let _ = fields.set_item("velocity_code", velocity_code);
            let _ = fields.set_item("expected", expected);
            let _ = fields.set_item("found", found);
            "sbas_long_term_record_count"
        }
        Encode::LongTermFieldNotCarried {
            message_type,
            half,
            record,
            field,
        } => {
            let _ = fields.set_item("message_type", message_type);
            let _ = fields.set_item("half", half);
            let _ = fields.set_item("record", record);
            let _ = fields.set_item("field", field);
            "sbas_long_term_field_not_carried"
        }
        Encode::LongTermMissingTimeOfDay { message_type, half } => {
            let _ = fields.set_item("message_type", message_type);
            let _ = fields.set_item("half", half);
            "sbas_long_term_missing_time_of_day"
        }
        Encode::PadBits { value } => {
            let _ = fields.set_item("value", value);
            "sbas_pad_bits"
        }
        _ => "sbas_unknown",
    };
    let exception = match crate::rtcm_encode_error_type(py) {
        Ok(error_type) => PyErr::from_type(error_type, error.to_string()),
        Err(type_error) => return type_error,
    };
    let value = exception.value(py);
    if let Err(attribute_error) = value
        .setattr("kind", kind)
        .and_then(|()| value.setattr("details", fields))
    {
        return attribute_error;
    }
    exception
}

fn conversion_kind(error: &sidereon_core::rtcm::RtcmConversionError) -> &'static str {
    use sidereon_core::rtcm::RtcmConversionError as Conversion;
    match error {
        Conversion::SatelliteIdOutOfRange { .. } => "satellite_id_out_of_range",
        Conversion::InvalidSatellite { .. } => "invalid_satellite",
        Conversion::SbasPrnOutsideWindow { .. } => "sbas_prn_outside_window",
        Conversion::NoLnavRecord { .. } => "no_lnav_record",
        Conversion::WeekMismatch { .. } => "week_mismatch",
        Conversion::NavicWeekMismatch { .. } => "navic_week_mismatch",
        Conversion::TimeNotRepresentable { .. } => "time_not_representable",
        Conversion::GalileoWeekOverflow => "galileo_week_overflow",
        Conversion::SisaSpare { .. } => "sisa_spare",
        Conversion::SisaNoPrediction => "sisa_no_prediction",
        Conversion::UraOutOfRange { .. } => "ura_out_of_range",
        Conversion::UraNoPrediction { .. } => "ura_no_prediction",
        Conversion::FitInterval(_) => "fit_interval",
        Conversion::VtecEvaluation(_) => "vtec_evaluation",
        _ => "other",
    }
}

fn vtec_problem_kind(problem: &sidereon_core::rtcm::VtecEvaluationProblem) -> &'static str {
    use sidereon_core::rtcm::VtecEvaluationProblem as Problem;
    match problem {
        Problem::ComputationTime => "computation_time",
        Problem::Frequency => "frequency",
        Problem::NonFiniteCoordinates => "non_finite_coordinates",
        Problem::MessageIdentity { .. } => "message_identity",
        Problem::LayerCount { .. } => "layer_count",
        Problem::InvalidGeometry => "invalid_geometry",
        Problem::BelowHorizon => "below_horizon",
        Problem::LayerDegreeOrder { .. } => "layer_degree_order",
        Problem::CoefficientCounts { .. } => "coefficient_counts",
        Problem::UnavailableCoefficient { .. } => "unavailable_coefficient",
        Problem::ShellNotAboveReceiver { .. } => "shell_not_above_receiver",
        Problem::MissingCoefficient { .. } => "missing_coefficient",
        Problem::InvalidMappingFactor { .. } => "invalid_mapping_factor",
        Problem::PhysicalResultOutOfRange { .. } => "physical_result_out_of_range",
        _ => "other",
    }
}

fn satellite_id_error_kind(error: &sidereon_core::SatelliteIdError) -> &'static str {
    match error {
        sidereon_core::SatelliteIdError::InvalidInput { .. } => "invalid_input",
    }
}

fn lnav_record_error_kind(error: &sidereon_core::ephemeris::LnavRecordError) -> &'static str {
    use sidereon_core::ephemeris::LnavRecordError as LnavError;
    match error {
        LnavError::NotGps(_) => "not_gps",
        LnavError::InvalidEpoch(_) => "invalid_epoch",
        LnavError::WeekMismatch { .. } => "week_mismatch",
        LnavError::NoUraPrediction(_) => "no_ura_prediction",
        LnavError::FitIntervalUnsupported { .. } => "fit_interval_unsupported",
    }
}

fn to_rtcm_conversion_err(py: Python<'_>, error: sidereon_core::Error) -> PyErr {
    let sidereon_core::Error::RtcmConversion(detail) = error else {
        return to_rtcm_err(error);
    };
    let kind = conversion_kind(&detail);
    let py_error = crate::RtcmConversionError::new_err(detail.to_string());
    let fields = PyDict::new(py);
    use sidereon_core::rtcm::RtcmConversionError as Conversion;
    match *detail {
        Conversion::SatelliteIdOutOfRange {
            message_number,
            field,
            value,
            width,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("width", width);
        }
        Conversion::InvalidSatellite {
            message_number,
            field,
            value,
            error,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("field", field);
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("error", error.to_string());
            let _ = fields.set_item("error_kind", satellite_id_error_kind(&error));
            match error {
                sidereon_core::SatelliteIdError::InvalidInput { field, reason } => {
                    let _ = fields.set_item("error_field", field);
                    let _ = fields.set_item("reason", reason);
                }
            }
        }
        Conversion::SbasPrnOutsideWindow {
            value,
            broadcast_prn,
        } => {
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("broadcast_prn", broadcast_prn);
        }
        Conversion::NoLnavRecord { value, satellite } => {
            let _ = fields.set_item("value", value);
            let _ = fields.set_item("satellite", satellite.to_string());
        }
        Conversion::WeekMismatch {
            message_number,
            full_week,
            week,
        } => {
            let _ = fields.set_item("message_number", message_number);
            let _ = fields.set_item("full_week", full_week);
            let _ = fields.set_item("week", week);
        }
        Conversion::NavicWeekMismatch { full_week, week } => {
            let _ = fields.set_item("full_week", full_week);
            let _ = fields.set_item("week", week);
        }
        Conversion::TimeNotRepresentable { field } => {
            let _ = fields.set_item("field", field);
        }
        Conversion::GalileoWeekOverflow | Conversion::SisaNoPrediction => {}
        Conversion::SisaSpare { index } => {
            let _ = fields.set_item("index", index);
        }
        Conversion::UraOutOfRange { system, index }
        | Conversion::UraNoPrediction { system, index } => {
            let _ = fields.set_item("system", system.letter().to_string());
            let _ = fields.set_item("index", index);
        }
        Conversion::FitInterval(error) => {
            let _ = fields.set_item("error", error.to_string());
            let _ = fields.set_item("error_kind", lnav_record_error_kind(&error));
            use sidereon_core::ephemeris::LnavRecordError as LnavError;
            match error {
                LnavError::NotGps(satellite) => {
                    let _ = fields.set_item("satellite", satellite.to_string());
                }
                LnavError::InvalidEpoch(field) => {
                    let _ = fields.set_item("field", field);
                }
                LnavError::WeekMismatch {
                    full_week,
                    decoded_week,
                } => {
                    let _ = fields.set_item("full_week", full_week);
                    let _ = fields.set_item("decoded_week", decoded_week);
                }
                LnavError::NoUraPrediction(index) => {
                    let _ = fields.set_item("index", index);
                }
                LnavError::FitIntervalUnsupported {
                    fit_interval_flag,
                    iode,
                    iodc,
                } => {
                    let _ = fields.set_item("fit_interval_flag", fit_interval_flag);
                    let _ = fields.set_item("iode", iode);
                    let _ = fields.set_item("iodc", iodc);
                }
            }
        }
        Conversion::VtecEvaluation(problem) => {
            let _ = fields.set_item("problem", vtec_problem_kind(&problem));
            match problem {
                sidereon_core::rtcm::VtecEvaluationProblem::MessageIdentity { message_number } => {
                    let _ = fields.set_item("message_number", message_number);
                }
                sidereon_core::rtcm::VtecEvaluationProblem::LayerCount { layers } => {
                    let _ = fields.set_item("layers", layers);
                }
                sidereon_core::rtcm::VtecEvaluationProblem::LayerDegreeOrder {
                    layer_index,
                    degree,
                    order,
                } => {
                    let _ = fields.set_item("layer_index", layer_index);
                    let _ = fields.set_item("degree", degree);
                    let _ = fields.set_item("order", order);
                }
                sidereon_core::rtcm::VtecEvaluationProblem::CoefficientCounts {
                    layer_index,
                    cosine_expected,
                    cosine_actual,
                    sine_expected,
                    sine_actual,
                } => {
                    let _ = fields.set_item("layer_index", layer_index);
                    let _ = fields.set_item("cosine_expected", cosine_expected);
                    let _ = fields.set_item("cosine_actual", cosine_actual);
                    let _ = fields.set_item("sine_expected", sine_expected);
                    let _ = fields.set_item("sine_actual", sine_actual);
                }
                sidereon_core::rtcm::VtecEvaluationProblem::UnavailableCoefficient {
                    layer_index,
                }
                | sidereon_core::rtcm::VtecEvaluationProblem::ShellNotAboveReceiver {
                    layer_index,
                }
                | sidereon_core::rtcm::VtecEvaluationProblem::InvalidMappingFactor {
                    layer_index,
                } => {
                    let _ = fields.set_item("layer_index", layer_index);
                }
                sidereon_core::rtcm::VtecEvaluationProblem::MissingCoefficient {
                    layer_index,
                    field,
                    index,
                } => {
                    let _ = fields.set_item("layer_index", layer_index);
                    let _ = fields.set_item("field", field);
                    let _ = fields.set_item("index", index);
                }
                sidereon_core::rtcm::VtecEvaluationProblem::PhysicalResultOutOfRange { field } => {
                    let _ = fields.set_item("field", field);
                }
                _ => {}
            }
        }
        _ => {}
    }
    let exception = py_error.value(py);
    let _ = exception.setattr("kind", kind);
    let _ = exception.setattr("details", fields);
    py_error
}

/// Return the same typed kind and field dictionary used by the public RTCM,
/// SBAS, and conversion exceptions for nested core errors.
pub(crate) fn core_error_nested_fields<'py>(
    py: Python<'py>,
    error: &sidereon_core::Error,
) -> PyResult<(String, Bound<'py, PyDict>)> {
    let exception = match error {
        sidereon_core::Error::RtcmEncode(_) | sidereon_core::Error::SbasEncode(_) => {
            to_rtcm_encode_err(py, error.clone())
        }
        sidereon_core::Error::RtcmConversion(_) => to_rtcm_conversion_err(py, error.clone()),
        _ => {
            return Err(PyTypeError::new_err(
                "nested RTCM details require an RTCM or SBAS core error",
            ));
        }
    };
    let value = exception.value(py);
    let kind = value.getattr("kind")?.extract::<String>()?;
    let detail_value = value.getattr("details")?;
    let detail_fields = detail_value
        .downcast::<PyDict>()
        .map_err(|error| PyTypeError::new_err(error.to_string()))?;
    let fields = PyDict::new(py);
    for (key, item) in detail_fields.iter() {
        fields.set_item(key, item)?;
    }
    Ok((kind, fields))
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmLegacyL1")]
#[derive(Clone, Copy)]
pub struct PyRtcmLegacyL1 {
    inner: LegacyL1,
}

#[pymethods]
impl PyRtcmLegacyL1 {
    #[new]
    fn new(
        code_indicator: bool,
        pseudorange: u32,
        phase_range_minus_pseudorange: i32,
        lock_time_indicator: u8,
        pseudorange_modulus_ambiguity: Option<u8>,
        cnr: Option<u8>,
    ) -> Self {
        Self {
            inner: LegacyL1 {
                code_indicator,
                pseudorange,
                phase_range_minus_pseudorange,
                lock_time_indicator,
                pseudorange_modulus_ambiguity,
                cnr,
            },
        }
    }

    #[getter]
    fn code_indicator(&self) -> bool {
        self.inner.code_indicator
    }
    #[getter]
    fn pseudorange(&self) -> u32 {
        self.inner.pseudorange
    }
    #[getter]
    fn phase_range_minus_pseudorange(&self) -> i32 {
        self.inner.phase_range_minus_pseudorange
    }
    #[getter]
    fn lock_time_indicator(&self) -> u8 {
        self.inner.lock_time_indicator
    }
    #[getter]
    fn pseudorange_modulus_ambiguity(&self) -> Option<u8> {
        self.inner.pseudorange_modulus_ambiguity
    }
    #[getter]
    fn cnr(&self) -> Option<u8> {
        self.inner.cnr
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmLegacyL2")]
#[derive(Clone, Copy)]
pub struct PyRtcmLegacyL2 {
    inner: LegacyL2,
}

#[pymethods]
impl PyRtcmLegacyL2 {
    #[new]
    fn new(
        code_indicator: u8,
        pseudorange_difference: i16,
        phase_range_minus_l1_pseudorange: i32,
        lock_time_indicator: u8,
        cnr: Option<u8>,
    ) -> Self {
        Self {
            inner: LegacyL2 {
                code_indicator,
                pseudorange_difference,
                phase_range_minus_l1_pseudorange,
                lock_time_indicator,
                cnr,
            },
        }
    }

    #[getter]
    fn code_indicator(&self) -> u8 {
        self.inner.code_indicator
    }
    #[getter]
    fn pseudorange_difference(&self) -> i16 {
        self.inner.pseudorange_difference
    }
    #[getter]
    fn phase_range_minus_l1_pseudorange(&self) -> i32 {
        self.inner.phase_range_minus_l1_pseudorange
    }
    #[getter]
    fn lock_time_indicator(&self) -> u8 {
        self.inner.lock_time_indicator
    }
    #[getter]
    fn cnr(&self) -> Option<u8> {
        self.inner.cnr
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmLegacySatellite")]
#[derive(Clone, Copy)]
pub struct PyRtcmLegacySatellite {
    inner: LegacySatellite,
}

#[pymethods]
impl PyRtcmLegacySatellite {
    #[new]
    fn new(
        satellite_id: u8,
        frequency_channel: Option<u8>,
        l1: PyRef<'_, PyRtcmLegacyL1>,
        l2: Option<PyRef<'_, PyRtcmLegacyL2>>,
    ) -> Self {
        Self {
            inner: LegacySatellite {
                satellite_id,
                frequency_channel,
                l1: l1.inner,
                l2: l2.map(|payload| payload.inner),
            },
        }
    }

    #[getter]
    fn satellite_id(&self) -> u8 {
        self.inner.satellite_id
    }
    #[getter]
    fn frequency_channel(&self) -> Option<u8> {
        self.inner.frequency_channel
    }
    #[getter]
    fn l1(&self) -> PyRtcmLegacyL1 {
        PyRtcmLegacyL1 {
            inner: self.inner.l1,
        }
    }
    #[getter]
    fn l2(&self) -> Option<PyRtcmLegacyL2> {
        self.inner.l2.map(|inner| PyRtcmLegacyL2 { inner })
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmLegacyObservations")]
#[derive(Clone)]
pub struct PyRtcmLegacyObservations {
    inner: LegacyObservations,
}

#[pymethods]
impl PyRtcmLegacyObservations {
    #[new]
    #[pyo3(signature = (
        message_number, reference_station_id, epoch_time, synchronous_gnss,
        satellite_count, divergence_free_smoothing, smoothing_interval,
        satellites, trailing_bits=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        message_number: u16,
        reference_station_id: u16,
        epoch_time: u32,
        synchronous_gnss: bool,
        satellite_count: u8,
        divergence_free_smoothing: bool,
        smoothing_interval: u8,
        satellites: Vec<Py<PyRtcmLegacySatellite>>,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: LegacyObservations {
                message_number,
                reference_station_id,
                epoch_time,
                synchronous_gnss,
                satellite_count,
                divergence_free_smoothing,
                smoothing_interval,
                satellites: satellites
                    .iter()
                    .map(|payload| payload.borrow(py).inner)
                    .collect(),
                trailing_bits,
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
    fn epoch_time(&self) -> u32 {
        self.inner.epoch_time
    }
    #[getter]
    fn synchronous_gnss(&self) -> bool {
        self.inner.synchronous_gnss
    }
    #[getter]
    fn satellite_count(&self) -> u8 {
        self.inner.satellite_count
    }
    #[getter]
    fn divergence_free_smoothing(&self) -> bool {
        self.inner.divergence_free_smoothing
    }
    #[getter]
    fn smoothing_interval(&self) -> u8 {
        self.inner.smoothing_interval
    }
    #[getter]
    fn satellites(&self) -> Vec<PyRtcmLegacySatellite> {
        self.inner
            .satellites
            .iter()
            .copied()
            .map(|inner| PyRtcmLegacySatellite { inner })
            .collect()
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
    #[getter]
    fn system(&self) -> Option<crate::marshal::PyGnssSystem> {
        self.inner.system().map(Into::into)
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmMessageAnnouncement")]
#[derive(Clone, Copy)]
pub struct PyRtcmMessageAnnouncement {
    inner: MessageAnnouncement,
}

#[pymethods]
impl PyRtcmMessageAnnouncement {
    #[new]
    fn new(message_number: u16, synchronous: bool, interval: u16) -> Self {
        Self {
            inner: MessageAnnouncement {
                message_number,
                synchronous,
                interval,
            },
        }
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn synchronous(&self) -> bool {
        self.inner.synchronous
    }
    #[getter]
    fn interval(&self) -> u16 {
        self.inner.interval
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmSystemParameters")]
#[derive(Clone)]
pub struct PyRtcmSystemParameters {
    inner: SystemParameters,
}

#[pymethods]
impl PyRtcmSystemParameters {
    #[new]
    #[pyo3(signature = (
        reference_station_id, mjd, seconds_of_day, announcement_count,
        leap_seconds, announcements, trailing_bits=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        reference_station_id: u16,
        mjd: u16,
        seconds_of_day: u32,
        announcement_count: u8,
        leap_seconds: u8,
        announcements: Vec<Py<PyRtcmMessageAnnouncement>>,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: SystemParameters {
                reference_station_id,
                mjd,
                seconds_of_day,
                announcement_count,
                leap_seconds,
                announcements: announcements
                    .iter()
                    .map(|payload| payload.borrow(py).inner)
                    .collect(),
                trailing_bits,
            },
        }
    }

    #[getter]
    fn reference_station_id(&self) -> u16 {
        self.inner.reference_station_id
    }
    #[getter]
    fn mjd(&self) -> u16 {
        self.inner.mjd
    }
    #[getter]
    fn seconds_of_day(&self) -> u32 {
        self.inner.seconds_of_day
    }
    #[getter]
    fn announcement_count(&self) -> u8 {
        self.inner.announcement_count
    }
    #[getter]
    fn leap_seconds(&self) -> u8 {
        self.inner.leap_seconds
    }
    #[getter]
    fn announcements(&self) -> Vec<PyRtcmMessageAnnouncement> {
        self.inner
            .announcements
            .iter()
            .copied()
            .map(|inner| PyRtcmMessageAnnouncement { inner })
            .collect()
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmTextMessage")]
#[derive(Clone)]
pub struct PyRtcmTextMessage {
    inner: TextMessage,
}

#[pymethods]
impl PyRtcmTextMessage {
    #[new]
    #[pyo3(signature = (
        reference_station_id, mjd, seconds_of_day, character_count,
        code_units, trailing_bits=Vec::new()
    ))]
    fn new(
        reference_station_id: u16,
        mjd: u16,
        seconds_of_day: u32,
        character_count: u8,
        code_units: Vec<u8>,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: TextMessage {
                reference_station_id,
                mjd,
                seconds_of_day,
                character_count,
                code_units,
                trailing_bits,
            },
        }
    }

    #[getter]
    fn reference_station_id(&self) -> u16 {
        self.inner.reference_station_id
    }
    #[getter]
    fn mjd(&self) -> u16 {
        self.inner.mjd
    }
    #[getter]
    fn seconds_of_day(&self) -> u32 {
        self.inner.seconds_of_day
    }
    #[getter]
    fn character_count(&self) -> u8 {
        self.inner.character_count
    }
    #[getter]
    fn code_units(&self) -> Vec<u8> {
        self.inner.code_units.clone()
    }
    #[getter]
    fn text(&self) -> Option<String> {
        self.inner.text().ok().map(str::to_owned)
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmNetworkAuxiliaryStation")]
#[derive(Clone)]
pub struct PyRtcmNetworkAuxiliaryStation {
    inner: NetworkAuxiliaryStation,
}

#[pymethods]
impl PyRtcmNetworkAuxiliaryStation {
    #[new]
    #[pyo3(signature = (
        network_id, subnetwork_id, auxiliary_station_count, master_station_id,
        auxiliary_station_id, delta_latitude, delta_longitude, delta_height,
        trailing_bits=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        network_id: u8,
        subnetwork_id: u8,
        auxiliary_station_count: u8,
        master_station_id: u16,
        auxiliary_station_id: u16,
        delta_latitude: i32,
        delta_longitude: i32,
        delta_height: i32,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: NetworkAuxiliaryStation {
                network_id,
                subnetwork_id,
                auxiliary_station_count,
                master_station_id,
                auxiliary_station_id,
                delta_latitude,
                delta_longitude,
                delta_height,
                trailing_bits,
            },
        }
    }

    #[getter]
    fn network_id(&self) -> u8 {
        self.inner.network_id
    }
    #[getter]
    fn subnetwork_id(&self) -> u8 {
        self.inner.subnetwork_id
    }
    #[getter]
    fn auxiliary_station_count(&self) -> u8 {
        self.inner.auxiliary_station_count
    }
    #[getter]
    fn master_station_id(&self) -> u16 {
        self.inner.master_station_id
    }
    #[getter]
    fn auxiliary_station_id(&self) -> u16 {
        self.inner.auxiliary_station_id
    }
    #[getter]
    fn delta_latitude(&self) -> i32 {
        self.inner.delta_latitude
    }
    #[getter]
    fn delta_longitude(&self) -> i32 {
        self.inner.delta_longitude
    }
    #[getter]
    fn delta_height(&self) -> i32 {
        self.inner.delta_height
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(
    module = "sidereon._sidereon",
    name = "RtcmNetworkCorrectionDifferences"
)]
#[derive(Clone)]
pub struct PyRtcmNetworkCorrectionDifferences {
    inner: NetworkCorrectionDifferences,
}

#[pymethods]
impl PyRtcmNetworkCorrectionDifferences {
    #[new]
    #[pyo3(signature = (
        message_number, network_id, subnetwork_id, epoch_time, multiple_message,
        master_station_id, auxiliary_station_id, satellite_count, satellites,
        trailing_bits=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        message_number: u16,
        network_id: u8,
        subnetwork_id: u8,
        epoch_time: u32,
        multiple_message: bool,
        master_station_id: u16,
        auxiliary_station_id: u16,
        satellite_count: u8,
        satellites: Vec<PyNetworkCorrectionSatellite>,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: NetworkCorrectionDifferences {
                message_number,
                network_id,
                subnetwork_id,
                epoch_time,
                multiple_message,
                master_station_id,
                auxiliary_station_id,
                satellite_count,
                satellites: satellites
                    .into_iter()
                    .map(|record| NetworkCorrectionDifference {
                        satellite_id: record.0,
                        ambiguity_status: record.1,
                        non_sync_count: record.2,
                        geometric: record.3,
                        iod: record.4,
                        ionospheric: record.5,
                    })
                    .collect(),
                trailing_bits,
            },
        }
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn network_id(&self) -> u8 {
        self.inner.network_id
    }
    #[getter]
    fn subnetwork_id(&self) -> u8 {
        self.inner.subnetwork_id
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
    fn master_station_id(&self) -> u16 {
        self.inner.master_station_id
    }
    #[getter]
    fn auxiliary_station_id(&self) -> u16 {
        self.inner.auxiliary_station_id
    }
    #[getter]
    fn satellite_count(&self) -> u8 {
        self.inner.satellite_count
    }
    #[getter]
    fn satellites(&self) -> Vec<PyNetworkCorrectionSatellite> {
        self.inner
            .satellites
            .iter()
            .map(|record| {
                (
                    record.satellite_id,
                    record.ambiguity_status,
                    record.non_sync_count,
                    record.geometric,
                    record.iod,
                    record.ionospheric,
                )
            })
            .collect()
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmNetworkResiduals")]
#[derive(Clone)]
pub struct PyRtcmNetworkResiduals {
    inner: NetworkResiduals,
}

#[pymethods]
impl PyRtcmNetworkResiduals {
    #[new]
    #[pyo3(signature = (
        message_number, epoch_time, reference_station_id, reference_station_count,
        satellite_count, satellites, trailing_bits=Vec::new()
    ))]
    fn new(
        message_number: u16,
        epoch_time: u32,
        reference_station_id: u16,
        reference_station_count: u8,
        satellite_count: u8,
        satellites: Vec<(u8, u8, u16, u8, u16, u16)>,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: NetworkResiduals {
                message_number,
                epoch_time,
                reference_station_id,
                reference_station_count,
                satellite_count,
                satellites: satellites
                    .into_iter()
                    .map(|record| NetworkResidual {
                        satellite_id: record.0,
                        s_oc: record.1,
                        s_od: record.2,
                        s_oh: record.3,
                        s_lc: record.4,
                        s_ld: record.5,
                    })
                    .collect(),
                trailing_bits,
            },
        }
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn epoch_time(&self) -> u32 {
        self.inner.epoch_time
    }
    #[getter]
    fn reference_station_id(&self) -> u16 {
        self.inner.reference_station_id
    }
    #[getter]
    fn reference_station_count(&self) -> u8 {
        self.inner.reference_station_count
    }
    #[getter]
    fn satellite_count(&self) -> u8 {
        self.inner.satellite_count
    }
    #[getter]
    fn satellites(&self) -> Vec<(u8, u8, u16, u8, u16, u16)> {
        self.inner
            .satellites
            .iter()
            .map(|record| {
                (
                    record.satellite_id,
                    record.s_oc,
                    record.s_od,
                    record.s_oh,
                    record.s_lc,
                    record.s_ld,
                )
            })
            .collect()
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmPhysicalReferenceStation")]
#[derive(Clone)]
pub struct PyRtcmPhysicalReferenceStation {
    inner: PhysicalReferenceStation,
}

#[pymethods]
impl PyRtcmPhysicalReferenceStation {
    #[new]
    #[pyo3(signature = (
        non_physical_station_id, physical_station_id, itrf_realization_year,
        ecef_x, ecef_y, ecef_z, trailing_bits=Vec::new()
    ))]
    fn new(
        non_physical_station_id: u16,
        physical_station_id: u16,
        itrf_realization_year: u8,
        ecef_x: i64,
        ecef_y: i64,
        ecef_z: i64,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: PhysicalReferenceStation {
                non_physical_station_id,
                physical_station_id,
                itrf_realization_year,
                ecef_x,
                ecef_y,
                ecef_z,
                trailing_bits,
            },
        }
    }

    #[getter]
    fn non_physical_station_id(&self) -> u16 {
        self.inner.non_physical_station_id
    }
    #[getter]
    fn physical_station_id(&self) -> u16 {
        self.inner.physical_station_id
    }
    #[getter]
    fn itrf_realization_year(&self) -> u8 {
        self.inner.itrf_realization_year
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
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmFkpGradients")]
#[derive(Clone)]
pub struct PyRtcmFkpGradients {
    inner: FkpGradients,
}

#[pymethods]
impl PyRtcmFkpGradients {
    #[new]
    #[pyo3(signature = (
        message_number, reference_station_id, epoch_time, satellite_count,
        satellites, trailing_bits=Vec::new()
    ))]
    fn new(
        message_number: u16,
        reference_station_id: u16,
        epoch_time: u32,
        satellite_count: u8,
        satellites: Vec<(u8, u8, i16, i16, i16, i16)>,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: FkpGradients {
                message_number,
                reference_station_id,
                epoch_time,
                satellite_count,
                satellites: satellites
                    .into_iter()
                    .map(|record| FkpGradient {
                        satellite_id: record.0,
                        iod: record.1,
                        geometric_north: record.2,
                        geometric_east: record.3,
                        ionospheric_north: record.4,
                        ionospheric_east: record.5,
                    })
                    .collect(),
                trailing_bits,
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
    fn epoch_time(&self) -> u32 {
        self.inner.epoch_time
    }
    #[getter]
    fn satellite_count(&self) -> u8 {
        self.inner.satellite_count
    }
    #[getter]
    fn satellites(&self) -> Vec<(u8, u8, i16, i16, i16, i16)> {
        self.inner
            .satellites
            .iter()
            .map(|record| {
                (
                    record.satellite_id,
                    record.iod,
                    record.geometric_north,
                    record.geometric_east,
                    record.ionospheric_north,
                    record.ionospheric_east,
                )
            })
            .collect()
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmHelmertTransformation")]
#[derive(Clone)]
pub struct PyRtcmHelmertTransformation {
    inner: HelmertTransformation,
}

#[pymethods]
impl PyRtcmHelmertTransformation {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        message_number, source_name, target_name, system_id, utilized_messages,
        plate_number, computation_indicator, height_indicator, validity_latitude,
        validity_longitude, validity_extension_latitude, validity_extension_longitude,
        dx, dy, dz, r1, r2, r3, ds, rotation_point, add_as, add_bs, add_at, add_bt,
        horizontal_quality, vertical_quality, trailing_bits=Vec::new()
    ))]
    fn new(
        message_number: u16,
        source_name: String,
        target_name: String,
        system_id: u8,
        utilized_messages: u16,
        plate_number: u8,
        computation_indicator: u8,
        height_indicator: u8,
        validity_latitude: i32,
        validity_longitude: i32,
        validity_extension_latitude: u16,
        validity_extension_longitude: u16,
        dx: i32,
        dy: i32,
        dz: i32,
        r1: i32,
        r2: i32,
        r3: i32,
        ds: i32,
        rotation_point: Option<(i64, i64, i64)>,
        add_as: u32,
        add_bs: u32,
        add_at: u32,
        add_bt: u32,
        horizontal_quality: u8,
        vertical_quality: u8,
        trailing_bits: Vec<bool>,
    ) -> Self {
        Self {
            inner: HelmertTransformation {
                message_number,
                source_name,
                target_name,
                system_id,
                utilized_messages,
                plate_number,
                computation_indicator,
                height_indicator,
                validity_latitude,
                validity_longitude,
                validity_extension_latitude,
                validity_extension_longitude,
                dx,
                dy,
                dz,
                r1,
                r2,
                r3,
                ds,
                rotation_point: rotation_point.map(|point| RotationPoint {
                    x: point.0,
                    y: point.1,
                    z: point.2,
                }),
                add_as,
                add_bs,
                add_at,
                add_bt,
                horizontal_quality,
                vertical_quality,
                trailing_bits,
            },
        }
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn source_name(&self) -> String {
        self.inner.source_name.clone()
    }
    #[getter]
    fn target_name(&self) -> String {
        self.inner.target_name.clone()
    }
    #[getter]
    fn system_id(&self) -> u8 {
        self.inner.system_id
    }
    #[getter]
    fn utilized_messages(&self) -> u16 {
        self.inner.utilized_messages
    }
    #[getter]
    fn plate_number(&self) -> u8 {
        self.inner.plate_number
    }
    #[getter]
    fn computation_indicator(&self) -> u8 {
        self.inner.computation_indicator
    }
    #[getter]
    fn height_indicator(&self) -> u8 {
        self.inner.height_indicator
    }
    #[getter]
    fn validity_latitude(&self) -> i32 {
        self.inner.validity_latitude
    }
    #[getter]
    fn validity_longitude(&self) -> i32 {
        self.inner.validity_longitude
    }
    #[getter]
    fn validity_extension_latitude(&self) -> u16 {
        self.inner.validity_extension_latitude
    }
    #[getter]
    fn validity_extension_longitude(&self) -> u16 {
        self.inner.validity_extension_longitude
    }
    #[getter]
    fn dx(&self) -> i32 {
        self.inner.dx
    }
    #[getter]
    fn dy(&self) -> i32 {
        self.inner.dy
    }
    #[getter]
    fn dz(&self) -> i32 {
        self.inner.dz
    }
    #[getter]
    fn r1(&self) -> i32 {
        self.inner.r1
    }
    #[getter]
    fn r2(&self) -> i32 {
        self.inner.r2
    }
    #[getter]
    fn r3(&self) -> i32 {
        self.inner.r3
    }
    #[getter]
    fn ds(&self) -> i32 {
        self.inner.ds
    }
    #[getter]
    fn rotation_point(&self) -> Option<(i64, i64, i64)> {
        self.inner
            .rotation_point
            .map(|point| (point.x, point.y, point.z))
    }
    #[getter]
    fn add_as(&self) -> u32 {
        self.inner.add_as
    }
    #[getter]
    fn add_bs(&self) -> u32 {
        self.inner.add_bs
    }
    #[getter]
    fn add_at(&self) -> u32 {
        self.inner.add_at
    }
    #[getter]
    fn add_bt(&self) -> u32 {
        self.inner.add_bt
    }
    #[getter]
    fn horizontal_quality(&self) -> u8 {
        self.inner.horizontal_quality
    }
    #[getter]
    fn vertical_quality(&self) -> u8 {
        self.inner.vertical_quality
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmResidualGrid")]
#[derive(Clone)]
pub struct PyRtcmResidualGrid {
    inner: ResidualGrid,
}

#[pymethods]
impl PyRtcmResidualGrid {
    #[new]
    #[pyo3(signature = (
        message_number, system_id, horizontal_shift, vertical_shift, origin_1,
        origin_2, extension_1, extension_2, mean_offset_1, mean_offset_2,
        mean_height_offset, residuals, horizontal_interpolation,
        vertical_interpolation, horizontal_quality, vertical_quality, mjd,
        trailing_bits=Vec::new()
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        message_number: u16,
        system_id: u8,
        horizontal_shift: bool,
        vertical_shift: bool,
        origin_1: i32,
        origin_2: i32,
        extension_1: u16,
        extension_2: u16,
        mean_offset_1: i16,
        mean_offset_2: i16,
        mean_height_offset: i16,
        residuals: Vec<(i16, i16, i16)>,
        horizontal_interpolation: u8,
        vertical_interpolation: u8,
        horizontal_quality: u8,
        vertical_quality: u8,
        mjd: u16,
        trailing_bits: Vec<bool>,
    ) -> PyResult<Self> {
        let residuals: [GridResidual; 16] = residuals
            .into_iter()
            .map(|record| GridResidual {
                horizontal_1: record.0,
                horizontal_2: record.1,
                height: record.2,
            })
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| PyValueError::new_err("residuals must contain exactly 16 records"))?;
        Ok(Self {
            inner: ResidualGrid {
                message_number,
                system_id,
                horizontal_shift,
                vertical_shift,
                origin_1,
                origin_2,
                extension_1,
                extension_2,
                mean_offset_1,
                mean_offset_2,
                mean_height_offset,
                residuals,
                horizontal_interpolation,
                vertical_interpolation,
                horizontal_quality,
                vertical_quality,
                mjd,
                trailing_bits,
            },
        })
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn system_id(&self) -> u8 {
        self.inner.system_id
    }
    #[getter]
    fn horizontal_shift(&self) -> bool {
        self.inner.horizontal_shift
    }
    #[getter]
    fn vertical_shift(&self) -> bool {
        self.inner.vertical_shift
    }
    #[getter]
    fn origin_1(&self) -> i32 {
        self.inner.origin_1
    }
    #[getter]
    fn origin_2(&self) -> i32 {
        self.inner.origin_2
    }
    #[getter]
    fn extension_1(&self) -> u16 {
        self.inner.extension_1
    }
    #[getter]
    fn extension_2(&self) -> u16 {
        self.inner.extension_2
    }
    #[getter]
    fn mean_offset_1(&self) -> i16 {
        self.inner.mean_offset_1
    }
    #[getter]
    fn mean_offset_2(&self) -> i16 {
        self.inner.mean_offset_2
    }
    #[getter]
    fn mean_height_offset(&self) -> i16 {
        self.inner.mean_height_offset
    }
    #[getter]
    fn residuals(&self) -> Vec<(i16, i16, i16)> {
        self.inner
            .residuals
            .iter()
            .map(|residual| {
                (
                    residual.horizontal_1,
                    residual.horizontal_2,
                    residual.height,
                )
            })
            .collect()
    }
    #[getter]
    fn horizontal_interpolation(&self) -> u8 {
        self.inner.horizontal_interpolation
    }
    #[getter]
    fn vertical_interpolation(&self) -> u8 {
        self.inner.vertical_interpolation
    }
    #[getter]
    fn horizontal_quality(&self) -> u8 {
        self.inner.horizontal_quality
    }
    #[getter]
    fn vertical_quality(&self) -> u8 {
        self.inner.vertical_quality
    }
    #[getter]
    fn mjd(&self) -> u16 {
        self.inner.mjd
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmProjection")]
#[derive(Clone)]
pub struct PyRtcmProjection {
    inner: Projection,
}

#[pymethods]
impl PyRtcmProjection {
    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (system_id, projection_type, latitude, longitude, add_scale, false_easting, false_northing, trailing_bits=None))]
    fn natural_origin(
        system_id: u8,
        projection_type: u8,
        latitude: i64,
        longitude: i64,
        add_scale: u32,
        false_easting: u64,
        false_northing: i64,
        trailing_bits: Option<Vec<bool>>,
    ) -> Self {
        Self {
            inner: Projection {
                system_id,
                projection_type,
                parameters: ProjectionParameters::NaturalOrigin {
                    latitude,
                    longitude,
                    add_scale,
                    false_easting,
                    false_northing,
                },
                trailing_bits: trailing_bits.unwrap_or_default(),
            },
        }
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (system_id, projection_type, latitude, longitude, standard_parallel_1, standard_parallel_2, false_easting, false_northing, trailing_bits=None))]
    fn lambert_conic_conformal(
        system_id: u8,
        projection_type: u8,
        latitude: i64,
        longitude: i64,
        standard_parallel_1: i64,
        standard_parallel_2: i64,
        false_easting: u64,
        false_northing: i64,
        trailing_bits: Option<Vec<bool>>,
    ) -> Self {
        Self {
            inner: Projection {
                system_id,
                projection_type,
                parameters: ProjectionParameters::LambertConicConformal {
                    latitude,
                    longitude,
                    standard_parallel_1,
                    standard_parallel_2,
                    false_easting,
                    false_northing,
                },
                trailing_bits: trailing_bits.unwrap_or_default(),
            },
        }
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (system_id, projection_type, rectification, latitude, longitude, azimuth, rectified_to_skew, add_scale, easting, northing, trailing_bits=None))]
    fn oblique_mercator(
        system_id: u8,
        projection_type: u8,
        rectification: bool,
        latitude: i64,
        longitude: i64,
        azimuth: u64,
        rectified_to_skew: i32,
        add_scale: u32,
        easting: u64,
        northing: i64,
        trailing_bits: Option<Vec<bool>>,
    ) -> Self {
        Self {
            inner: Projection {
                system_id,
                projection_type,
                parameters: ProjectionParameters::ObliqueMercator {
                    rectification,
                    latitude,
                    longitude,
                    azimuth,
                    rectified_to_skew,
                    add_scale,
                    easting,
                    northing,
                },
                trailing_bits: trailing_bits.unwrap_or_default(),
            },
        }
    }

    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number()
    }
    #[getter]
    fn system_id(&self) -> u8 {
        self.inner.system_id
    }
    #[getter]
    fn projection_type(&self) -> u8 {
        self.inner.projection_type
    }
    #[getter]
    fn parameters_kind(&self) -> &'static str {
        match self.inner.parameters {
            ProjectionParameters::NaturalOrigin { .. } => "natural_origin",
            ProjectionParameters::LambertConicConformal { .. } => "lambert_conic_conformal",
            ProjectionParameters::ObliqueMercator { .. } => "oblique_mercator",
        }
    }
    #[getter]
    fn natural_origin_parameters(&self) -> Option<(i64, i64, u32, u64, i64)> {
        match self.inner.parameters {
            ProjectionParameters::NaturalOrigin {
                latitude,
                longitude,
                add_scale,
                false_easting,
                false_northing,
            } => Some((
                latitude,
                longitude,
                add_scale,
                false_easting,
                false_northing,
            )),
            _ => None,
        }
    }
    #[getter]
    fn lambert_conic_conformal_parameters(&self) -> Option<(i64, i64, i64, i64, u64, i64)> {
        match self.inner.parameters {
            ProjectionParameters::LambertConicConformal {
                latitude,
                longitude,
                standard_parallel_1,
                standard_parallel_2,
                false_easting,
                false_northing,
            } => Some((
                latitude,
                longitude,
                standard_parallel_1,
                standard_parallel_2,
                false_easting,
                false_northing,
            )),
            _ => None,
        }
    }
    #[getter]
    fn oblique_mercator_parameters(&self) -> Option<PyProjectionObliqueParameters> {
        match self.inner.parameters {
            ProjectionParameters::ObliqueMercator {
                rectification,
                latitude,
                longitude,
                azimuth,
                rectified_to_skew,
                add_scale,
                easting,
                northing,
            } => Some((
                rectification,
                latitude,
                longitude,
                azimuth,
                rectified_to_skew,
                add_scale,
                easting,
                northing,
            )),
            _ => None,
        }
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmSsrVtecLayer")]
#[derive(Clone)]
pub struct PyRtcmSsrVtecLayer {
    inner: SsrVtecLayer,
}

#[pymethods]
impl PyRtcmSsrVtecLayer {
    #[staticmethod]
    fn coefficient_counts(degree: u8, order: u8) -> (usize, usize) {
        SsrVtecLayer::coefficient_counts(degree, order)
    }

    #[getter]
    fn height(&self) -> u8 {
        self.inner.height
    }

    #[getter]
    fn degree(&self) -> u8 {
        self.inner.degree
    }

    #[getter]
    fn order(&self) -> u8 {
        self.inner.order
    }

    #[getter]
    fn cosine(&self) -> Vec<i16> {
        self.inner.cosine.clone()
    }

    #[getter]
    fn sine(&self) -> Vec<i16> {
        self.inner.sine.clone()
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmSsrVtecLayerEvaluation")]
#[derive(Clone, Copy)]
pub struct PyRtcmSsrVtecLayerEvaluation {
    inner: SsrVtecLayerEvaluation,
}

impl From<SsrVtecLayerEvaluation> for PyRtcmSsrVtecLayerEvaluation {
    fn from(inner: SsrVtecLayerEvaluation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRtcmSsrVtecLayerEvaluation {
    #[getter]
    fn pierce_latitude_rad(&self) -> f64 {
        self.inner.pierce_latitude_rad
    }
    #[getter]
    fn pierce_longitude_rad(&self) -> f64 {
        self.inner.pierce_longitude_rad
    }
    #[getter]
    fn sun_fixed_longitude_rad(&self) -> f64 {
        self.inner.sun_fixed_longitude_rad
    }
    #[getter]
    fn vtec_tecu(&self) -> f64 {
        self.inner.vtec_tecu
    }
    #[getter]
    fn mapping_factor(&self) -> f64 {
        self.inner.mapping_factor
    }
    #[getter]
    fn stec_tecu(&self) -> f64 {
        self.inner.stec_tecu
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmSsrVtecEvaluation")]
#[derive(Clone)]
pub struct PyRtcmSsrVtecEvaluation {
    inner: SsrVtecEvaluation,
}

impl From<SsrVtecEvaluation> for PyRtcmSsrVtecEvaluation {
    fn from(inner: SsrVtecEvaluation) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRtcmSsrVtecEvaluation {
    #[getter]
    fn layers(&self) -> Vec<PyRtcmSsrVtecLayerEvaluation> {
        self.inner.layers.iter().copied().map(Into::into).collect()
    }
    #[getter]
    fn stec_tecu(&self) -> f64 {
        self.inner.stec_tecu
    }
    #[getter]
    fn pseudorange_delay_m(&self) -> f64 {
        self.inner.pseudorange_delay_m
    }
    #[getter]
    fn phase_range_advance_m(&self) -> f64 {
        self.inner.phase_range_advance_m
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmSsrVtecMessage")]
#[derive(Clone)]
pub struct PyRtcmSsrVtecMessage {
    inner: SsrVtecMessage,
}

#[pymethods]
impl PyRtcmSsrVtecMessage {
    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn igs_ssr_version(&self) -> Option<u8> {
        self.inner.igs_ssr_version
    }
    #[getter]
    fn epoch_time_s(&self) -> u32 {
        self.inner.epoch_time_s
    }
    #[getter]
    fn update_interval(&self) -> u8 {
        self.inner.update_interval
    }
    #[getter]
    fn multiple_message(&self) -> bool {
        self.inner.multiple_message
    }
    #[getter]
    fn iod_ssr(&self) -> u8 {
        self.inner.iod_ssr
    }
    #[getter]
    fn provider_id(&self) -> u16 {
        self.inner.provider_id
    }
    #[getter]
    fn solution_id(&self) -> u8 {
        self.inner.solution_id
    }
    #[getter]
    fn quality_indicator(&self) -> u16 {
        self.inner.quality_indicator
    }
    #[getter]
    fn layers(&self) -> Vec<PyRtcmSsrVtecLayer> {
        self.inner
            .layers
            .iter()
            .cloned()
            .map(|inner| PyRtcmSsrVtecLayer { inner })
            .collect()
    }
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
    }
    fn evaluate(
        &self,
        py: Python<'_>,
        receiver_ecef_m: [f64; 3],
        satellite_transmit_ecef_m: [f64; 3],
        gps_seconds_of_day: f64,
        frequency_hz: f64,
    ) -> PyResult<PyRtcmSsrVtecEvaluation> {
        self.inner
            .evaluate(
                receiver_ecef_m,
                satellite_transmit_ecef_m,
                gps_seconds_of_day,
                frequency_hz,
            )
            .map(Into::into)
            .map_err(|error| to_rtcm_conversion_err(py, error))
    }
}

#[pyclass(module = "sidereon._sidereon", name = "RtcmSsrMessage")]
#[derive(Clone)]
pub struct PyRtcmSsrMessage {
    inner: SsrMessage,
}

#[pymethods]
impl PyRtcmSsrMessage {
    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number
    }
    #[getter]
    fn igs_ssr_version(&self) -> Option<u8> {
        self.inner.igs_ssr_version
    }
    #[getter]
    fn system(&self) -> crate::marshal::PyGnssSystem {
        self.inner.system.into()
    }
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            sidereon_core::rtcm::SsrKind::Orbit => "orbit",
            sidereon_core::rtcm::SsrKind::Clock => "clock",
            sidereon_core::rtcm::SsrKind::CombinedOrbitClock => "combined_orbit_clock",
            sidereon_core::rtcm::SsrKind::CodeBias => "code_bias",
            sidereon_core::rtcm::SsrKind::PhaseBias => "phase_bias",
            sidereon_core::rtcm::SsrKind::Ura => "ura",
            sidereon_core::rtcm::SsrKind::HighRateClock => "high_rate_clock",
        }
    }

    #[getter]
    fn header(&self) -> PySsrHeader {
        let header = &self.inner.header;
        (
            header.epoch_time_s,
            header.update_interval,
            header.multiple_message,
            header.iod_ssr,
            header.provider_id,
            header.solution_id,
            header.satellite_reference_datum,
            header.dispersive_bias_consistency,
            header.mw_consistency,
            header.satellite_count,
        )
    }
    #[getter]
    fn orbit(&self) -> Vec<PySsrOrbitRow> {
        self.inner
            .orbit
            .iter()
            .map(|record| {
                (
                    record.satellite_id,
                    record.iode,
                    record.iod_crc,
                    record.delta_radial,
                    record.delta_along,
                    record.delta_cross,
                    record.dot_delta_radial,
                    record.dot_delta_along,
                    record.dot_delta_cross,
                )
            })
            .collect()
    }
    #[getter]
    fn clock(&self) -> Vec<(u8, i32, i32, i32)> {
        self.inner
            .clock
            .iter()
            .map(|record| (record.satellite_id, record.c0, record.c1, record.c2))
            .collect()
    }
    #[getter]
    fn code_bias(&self) -> Vec<(u8, Vec<(u8, i16)>)> {
        self.inner
            .code_bias
            .iter()
            .map(|record| (record.satellite_id, record.biases.clone()))
            .collect()
    }
    #[getter]
    fn phase_bias(&self) -> Vec<PySsrPhaseBiasRow> {
        self.inner
            .phase_bias
            .iter()
            .map(|record| {
                (
                    record.satellite_id,
                    record.yaw_angle,
                    record.yaw_rate,
                    record
                        .biases
                        .iter()
                        .map(|signal| {
                            (
                                signal.signal_id,
                                signal.integer_indicator,
                                signal.wide_lane_integer_indicator,
                                signal.discontinuity_counter,
                                signal.bias,
                            )
                        })
                        .collect(),
                )
            })
            .collect()
    }
    #[getter]
    fn ura(&self) -> Vec<(u8, u8)> {
        self.inner.ura.clone()
    }
    #[getter]
    fn padding_bits(&self) -> Vec<bool> {
        self.inner.padding_bits.clone()
    }
    fn igs_ssr_subtype(&self) -> Option<u8> {
        self.inner.igs_ssr_subtype()
    }
}

/// How the RTCM decoders and encoders treat input that departs from the RTCM
/// 3 format while every field in it can still be read, mirroring the core
/// `RtcmPolicy`. A CRC-24Q mismatch and a body that ends inside a field are
/// refused under both.
#[pyclass(module = "sidereon._sidereon", name = "RtcmPolicy", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyRtcmPolicy {
    /// Refuse the first departure.
    STRICT,
    /// Read or write the input and report each departure.
    LENIENT,
}

impl From<PyRtcmPolicy> for RtcmPolicy {
    fn from(policy: PyRtcmPolicy) -> Self {
        match policy {
            PyRtcmPolicy::STRICT => RtcmPolicy::Strict,
            PyRtcmPolicy::LENIENT => RtcmPolicy::Lenient,
        }
    }
}

/// A departure from the RTCM 3 format, refused under `RtcmPolicy.STRICT` and
/// reported under `RtcmPolicy.LENIENT`.
///
/// `kind` is `frame_reserved_bits` (with `reserved`), `trailing_bits` (with
/// `message_number` and `bits`), `msm_cell_mask_over_64` (with
/// `message_number` and `cells`) or `ssr_records_short` (with
/// `message_number`, `declared` and `read`).
#[pyclass(module = "sidereon._sidereon", name = "RtcmDeparture")]
#[derive(Clone)]
pub struct PyRtcmDeparture {
    inner: RtcmDeparture,
    offset: Option<usize>,
}

impl PyRtcmDeparture {
    fn new(inner: RtcmDeparture) -> Self {
        Self {
            inner,
            offset: None,
        }
    }
}

#[pymethods]
impl PyRtcmDeparture {
    /// Departure kind label.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            RtcmDeparture::FrameReservedBits { .. } => "frame_reserved_bits".to_string(),
            RtcmDeparture::TrailingBits { .. } => "trailing_bits".to_string(),
            RtcmDeparture::MsmCellMaskOver64 { .. } => "msm_cell_mask_over_64".to_string(),
            RtcmDeparture::SsrRecordsShort { .. } => "ssr_records_short".to_string(),
            other => crate::marshal::debug_variant_snake(other),
        }
    }

    /// Byte offset of the frame preamble in the scanned stream, for a
    /// departure a stream read recorded.
    #[getter]
    fn offset(&self) -> Option<usize> {
        self.offset
    }

    /// The message number, when the departure is in a message body.
    #[getter]
    fn message_number(&self) -> Option<u16> {
        match &self.inner {
            RtcmDeparture::TrailingBits { message_number, .. }
            | RtcmDeparture::MsmCellMaskOver64 { message_number, .. }
            | RtcmDeparture::SsrRecordsShort { message_number, .. } => Some(*message_number),
            _ => None,
        }
    }

    /// The six frame reserved bits as read, for `frame_reserved_bits`.
    #[getter]
    fn reserved(&self) -> Option<u8> {
        match &self.inner {
            RtcmDeparture::FrameReservedBits { reserved } => Some(*reserved),
            _ => None,
        }
    }

    /// Every bit after the last field, for `trailing_bits`.
    #[getter]
    fn bits(&self) -> Option<Vec<bool>> {
        match &self.inner {
            RtcmDeparture::TrailingBits { bits, .. } => Some(bits.clone()),
            _ => None,
        }
    }

    /// Satellite count times signal count, for `msm_cell_mask_over_64`.
    #[getter]
    fn cells(&self) -> Option<usize> {
        match &self.inner {
            RtcmDeparture::MsmCellMaskOver64 { cells, .. } => Some(*cells),
            _ => None,
        }
    }

    /// The satellite count the SSR header states, for `ssr_records_short`.
    #[getter]
    fn declared(&self) -> Option<usize> {
        match &self.inner {
            RtcmDeparture::SsrRecordsShort { declared, .. } => Some(*declared),
            _ => None,
        }
    }

    /// The complete SSR records the body holds, for `ssr_records_short`.
    #[getter]
    fn read(&self) -> Option<usize> {
        match &self.inner {
            RtcmDeparture::SsrRecordsShort { read, .. } => Some(*read),
            _ => None,
        }
    }

    /// The departure as the core states it.
    #[getter]
    fn message(&self) -> String {
        self.inner.to_string()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmDeparture(kind={:?}, message={:?})",
            self.kind(),
            self.inner.to_string()
        )
    }
}

fn departures(list: Vec<RtcmDeparture>) -> Vec<PyRtcmDeparture> {
    list.into_iter().map(PyRtcmDeparture::new).collect()
}

fn parse_msm_kind(kind: &str) -> PyResult<MsmKind> {
    match kind {
        "msm1" => Ok(MsmKind::Msm1),
        "msm2" => Ok(MsmKind::Msm2),
        "msm3" => Ok(MsmKind::Msm3),
        "msm4" => Ok(MsmKind::Msm4),
        "msm5" => Ok(MsmKind::Msm5),
        "msm6" => Ok(MsmKind::Msm6),
        "msm7" => Ok(MsmKind::Msm7),
        other => Err(PyValueError::new_err(format!(
            "unknown RTCM MSM kind {other:?}; expected \"msm1\" through \"msm7\""
        ))),
    }
}

fn parse_system(system: &str) -> PyResult<GnssSystem> {
    let mut letters = system.chars();
    let (Some(letter), None) = (letters.next(), letters.next()) else {
        return Err(PyValueError::new_err(format!(
            "invalid RTCM MSM system {system:?}; expected a single letter"
        )));
    };
    GnssSystem::from_letter(letter)
        .ok_or_else(|| PyValueError::new_err(format!("unknown RTCM MSM system letter {letter:?}")))
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
        trailing_bits=Vec::new(),
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
        trailing_bits: Vec<bool>,
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
                trailing_bits,
            },
        }
    }

    /// Every body bit after the last field, when those bits are anything
    /// other than fewer than eight zeros (the byte alignment), as read under
    /// `RtcmPolicy.LENIENT`; empty otherwise. A strict `encode` refuses a
    /// nonempty value.
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
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
        trailing_bits=Vec::new(),
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
        trailing_bits: Vec<bool>,
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
                trailing_bits,
            },
        }
    }

    /// Every body bit after the last field, when those bits are anything
    /// other than fewer than eight zeros (the byte alignment), as read under
    /// `RtcmPolicy.LENIENT`; empty otherwise. A strict `encode` refuses a
    /// nonempty value.
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
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
        trailing_bits=Vec::new(),
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
        trailing_bits: Vec<bool>,
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
                trailing_bits,
            },
        }
    }

    /// Every body bit after the last field, when those bits are anything
    /// other than fewer than eight zeros (the byte alignment), as read under
    /// `RtcmPolicy.LENIENT`; empty otherwise. A strict `encode` refuses a
    /// nonempty value.
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
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

macro_rules! rtcm_keplerian_ephemeris_pyclass {
    (
        $(#[$meta:meta])*
        $py:ident, $core:ident, $py_name:literal, $repr_name:literal {
            $($field:ident : $ty:ty),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[pyclass(module = "sidereon._sidereon", name = $py_name)]
        #[derive(Clone)]
        pub struct $py {
            inner: $core,
        }

        #[pymethods]
        impl $py {
            /// Construct the ephemeris from raw transmitted integers.
            #[new]
            #[pyo3(signature = ($($field),+, trailing_bits=Vec::new()))]
            #[allow(clippy::too_many_arguments)]
            fn new($($field: $ty),+, trailing_bits: Vec<bool>) -> Self {
                Self {
                    inner: $core {
                        $($field),+,
                        trailing_bits,
                    },
                }
            }

            /// Every body bit after the last field, when those bits are anything
            /// other than fewer than eight zeros (the byte alignment), as read under
            /// `RtcmPolicy.LENIENT`; empty otherwise. A strict `encode` refuses a
            /// nonempty value.
            #[getter]
            fn trailing_bits(&self) -> Vec<bool> {
                self.inner.trailing_bits.clone()
            }


            $(
                #[getter]
                fn $field(&self) -> $ty {
                    self.inner.$field
                }
            )+

            /// Canonical satellite identifier, or `None` if the transmitted id
            /// is out of range.
            fn satellite(&self) -> Option<String> {
                self.inner.satellite().ok().map(|id| id.to_string())
            }

            fn __repr__(&self) -> String {
                format!(
                    "{}(satellite_id={}, week_number={})",
                    $repr_name, self.inner.satellite_id, self.inner.week_number
                )
            }
        }
    };
}

rtcm_keplerian_ephemeris_pyclass!(
    /// A 1045 Galileo F/NAV broadcast ephemeris. Every field is the raw transmitted integer.
    PyRtcmGalileoFnavEphemeris, GalileoFnavEphemeris, "RtcmGalileoFnavEphemeris", "RtcmGalileoFnavEphemeris" {
        satellite_id: u8,
        week_number: u16,
        iod_nav: u16,
        sisa: u8,
        idot: i32,
        t_oc: u16,
        a_f2: i16,
        a_f1: i32,
        a_f0: i64,
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
        bgd_e5a_e1: i16,
        e5a_signal_health: u8,
        e5a_data_validity: bool,
        reserved: u8,
    }
);

rtcm_keplerian_ephemeris_pyclass!(
    /// A 1046 Galileo I/NAV broadcast ephemeris. Every field is the raw transmitted integer.
    PyRtcmGalileoInavEphemeris, GalileoInavEphemeris, "RtcmGalileoInavEphemeris", "RtcmGalileoInavEphemeris" {
        satellite_id: u8,
        week_number: u16,
        iod_nav: u16,
        sisa_index: u8,
        idot: i32,
        t_oc: u16,
        a_f2: i16,
        a_f1: i32,
        a_f0: i64,
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
        bgd_e5a_e1: i16,
        bgd_e5b_e1: i16,
        e5b_signal_health: u8,
        e5b_data_validity: bool,
        e1b_signal_health: u8,
        e1b_data_validity: bool,
        reserved: u8,
    }
);

rtcm_keplerian_ephemeris_pyclass!(
    /// A 1042 BeiDou broadcast ephemeris. Every field is the raw transmitted integer.
    PyRtcmBeidouEphemeris, BeidouEphemeris, "RtcmBeidouEphemeris", "RtcmBeidouEphemeris" {
        satellite_id: u8,
        week_number: u16,
        sv_urai: u8,
        idot: i32,
        aode: u8,
        t_oc: u32,
        a_f2: i16,
        a_f1: i32,
        a_f0: i32,
        aodc: u8,
        c_rs: i32,
        delta_n: i32,
        m0: i64,
        c_uc: i32,
        eccentricity: u64,
        c_us: i32,
        sqrt_a: u64,
        t_oe: u32,
        c_ic: i32,
        omega0: i64,
        c_is: i32,
        i0: i64,
        c_rc: i32,
        omega: i64,
        omega_dot: i32,
        t_gd1: i16,
        t_gd2: i16,
        sv_health: bool,
    }
);

rtcm_keplerian_ephemeris_pyclass!(
    /// A 1044 QZSS broadcast ephemeris. Every field is the raw transmitted integer.
    PyRtcmQzssEphemeris, QzssEphemeris, "RtcmQzssEphemeris", "RtcmQzssEphemeris" {
        satellite_id: u8,
        t_oc: u16,
        a_f2: i16,
        a_f1: i32,
        a_f0: i32,
        iode: u8,
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
        idot: i32,
        codes_on_l2: u8,
        week_number: u16,
        ura: u8,
        sv_health: u8,
        t_gd: i16,
        iodc: u16,
        fit_interval: bool,
    }
);

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
        negative_zero=0, trailing_bits=Vec::new(),
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
        negative_zero: u16,
        trailing_bits: Vec<bool>,
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
                negative_zero,
                trailing_bits,
            },
        }
    }

    /// The sign-magnitude fields transmitted as negative zero, one bit per
    /// field (the `RTCM_GLONASS_NEGATIVE_ZERO_*` constants). Each reads as
    /// `0`, as RTKLIB `getbitg` reads it, and keeps its sign bit so the body
    /// re-encodes as transmitted.
    #[getter]
    fn negative_zero(&self) -> u16 {
        self.inner.negative_zero
    }

    /// Every body bit after the last field, when those bits are anything
    /// other than fewer than eight zeros (the byte alignment), as read under
    /// `RtcmPolicy.LENIENT`; empty otherwise. A strict `encode` refuses a
    /// nonempty value.
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
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
    /// Construct one MSM satellite row with fields absent from its message kind set to `None`.
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
        rough_range_ms: Option<u8>,
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
    fn rough_range_ms(&self) -> Option<u8> {
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
    /// Construct one MSM signal cell with fields absent from its message kind set to `None`.
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
        fine_pseudorange: Option<i32>,
        fine_phase_range: Option<i32>,
        lock_time_indicator: Option<u16>,
        half_cycle_ambiguity: Option<bool>,
        cnr: Option<u16>,
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
    fn fine_pseudorange(&self) -> Option<i32> {
        self.inner.fine_pseudorange
    }
    #[getter]
    fn fine_phase_range(&self) -> Option<i32> {
        self.inner.fine_phase_range
    }
    #[getter]
    fn lock_time_indicator(&self) -> Option<u16> {
        self.inner.lock_time_indicator
    }
    #[getter]
    fn half_cycle_ambiguity(&self) -> Option<bool> {
        self.inner.half_cycle_ambiguity
    }
    #[getter]
    fn cnr(&self) -> Option<u16> {
        self.inner.cnr
    }
    #[getter]
    fn fine_phase_range_rate(&self) -> Option<i16> {
        self.inner.fine_phase_range_rate
    }

    /// Minimum continuous-lock time encoded by this cell's raw lock indicator.
    ///
    /// The message kind selects the lock-time encoding.
    fn minimum_lock_time_ms(&self, kind: &str) -> PyResult<Option<u32>> {
        Ok(self.inner.minimum_lock_time_ms(parse_msm_kind(kind)?))
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmMsmSignal(satellite_id={}, signal_id={})",
            self.inner.satellite_id, self.inner.signal_id
        )
    }
}

/// A decoded MSM1 through MSM7 multi-signal observation message.
#[pyclass(module = "sidereon._sidereon", name = "RtcmMsmMessage")]
#[derive(Clone)]
pub struct PyRtcmMsmMessage {
    inner: MsmMessage,
}

#[pymethods]
impl PyRtcmMsmMessage {
    /// Construct an MSM1 through MSM7 observation message from its parts.
    ///
    /// `system` is the constellation single-letter identifier (e.g. `"G"`,
    /// `"R"`, `"E"`); `kind` is `"msm1"` through `"msm7"`. Satellites must be in
    /// ascending id order and signals in satellite-major then signal order, as
    /// the encoder expects. Raises `ValueError` for an unknown system letter or
    /// kind.
    #[new]
    ///
    /// `signal_mask` is the DF395 signal mask as transmitted (bit `32 - id`
    /// set for each listed signal id). A listed signal may have no cell, so
    /// the mask is kept as given; `None` builds it from the signal ids of
    /// `signals`.
    #[pyo3(signature = (
        message_number,
        system,
        kind,
        header,
        satellites,
        signals,
        signal_mask=None,
        trailing_bits=Vec::new(),
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        message_number: u16,
        system: &str,
        kind: &str,
        header: &PyRtcmMsmHeader,
        satellites: Vec<Py<PyRtcmMsmSatellite>>,
        signals: Vec<Py<PyRtcmMsmSignal>>,
        signal_mask: Option<u32>,
        trailing_bits: Vec<bool>,
    ) -> PyResult<Self> {
        let system = parse_system(system)?;
        let kind = parse_msm_kind(kind)?;
        let satellites = satellites.iter().map(|s| s.borrow(py).inner).collect();
        let signals: Vec<MsmSignal> = signals.iter().map(|s| s.borrow(py).inner).collect();
        let signal_mask = signal_mask.unwrap_or_else(|| core_msm_signal_mask(&signals));
        Ok(Self {
            inner: MsmMessage {
                message_number,
                system,
                kind,
                header: header.inner,
                satellites,
                signals,
                signal_mask,
                trailing_bits,
            },
        })
    }

    /// The DF395 signal mask as transmitted.
    #[getter]
    fn signal_mask(&self) -> u32 {
        self.inner.signal_mask
    }

    /// Every body bit after the last field, when those bits are anything
    /// other than fewer than eight zeros (the byte alignment), as read under
    /// `RtcmPolicy.LENIENT`; empty otherwise. A strict `encode` refuses a
    /// nonempty value.
    #[getter]
    fn trailing_bits(&self) -> Vec<bool> {
        self.inner.trailing_bits.clone()
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
    /// MSM variant, `"msm1"` through `"msm7"`.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            MsmKind::Msm1 => "msm1",
            MsmKind::Msm2 => "msm2",
            MsmKind::Msm3 => "msm3",
            MsmKind::Msm4 => "msm4",
            MsmKind::Msm5 => "msm5",
            MsmKind::Msm6 => "msm6",
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

/// Derived RINEX LLI for one RTCM MSM signal cell.
#[pyclass(module = "sidereon._sidereon", name = "RtcmCellLli")]
#[derive(Clone, Copy)]
pub struct PyRtcmCellLli {
    inner: CellLli,
}

#[pymethods]
impl PyRtcmCellLli {
    /// Satellite id, as carried by the MSM signal cell.
    #[getter]
    fn satellite_id(&self) -> u8 {
        self.inner.satellite_id
    }
    /// Signal id, as carried by the MSM signal cell.
    #[getter]
    fn signal_id(&self) -> u8 {
        self.inner.signal_id
    }
    /// Derived RINEX LLI digit. This binding exposes core bits 0 and 1 only.
    #[getter]
    fn lli(&self) -> u8 {
        self.inner.lli
    }
    /// Current normalized minimum lock time in milliseconds.
    #[getter]
    fn min_lock_time_ms(&self) -> Option<u32> {
        self.inner.min_lock_time_ms
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmCellLli(satellite_id={}, signal_id={}, lli={})",
            self.inner.satellite_id, self.inner.signal_id, self.inner.lli
        )
    }
}

/// Per-stream lock-time continuity tracker for RTCM MSM LLI derivation.
#[pyclass(module = "sidereon._sidereon", name = "RtcmLockTimeTracker")]
#[derive(Clone, Default)]
pub struct PyRtcmLockTimeTracker {
    inner: LockTimeTracker,
}

#[pymethods]
impl PyRtcmLockTimeTracker {
    #[new]
    fn new() -> Self {
        Self {
            inner: LockTimeTracker::new(),
        }
    }

    /// Derive LLI values for every signal cell in `message` and advance state.
    fn observe(&mut self, message: &PyRtcmMsmMessage) -> Vec<PyRtcmCellLli> {
        self.inner
            .observe(&message.inner)
            .into_iter()
            .map(|inner| PyRtcmCellLli { inner })
            .collect()
    }

    /// Drop all per-cell lock history.
    fn reset(&mut self) {
        self.inner.reset();
    }

    fn __repr__(&self) -> &'static str {
        "RtcmLockTimeTracker()"
    }
}

/// One CRC-valid RTCM frame whose body did not decode.
#[pyclass(module = "sidereon._sidereon", name = "RtcmFrameSkip")]
#[derive(Clone)]
pub struct PyRtcmFrameSkip {
    inner: FrameSkip,
}

#[pymethods]
impl PyRtcmFrameSkip {
    /// Byte offset of the skipped frame's preamble.
    #[getter]
    fn offset(&self) -> usize {
        self.inner.offset
    }
    /// RTCM message number when the frame body was long enough to carry one.
    #[getter]
    fn message_number(&self) -> Option<u16> {
        self.inner.message_number
    }
    /// Stable reason tag: `"truncated"`, `"malformed"` or `"departure"` (a
    /// departure from the format refused under the strict policy).
    #[getter]
    fn reason(&self) -> &'static str {
        match &self.inner.reason {
            FrameSkipReason::Truncated => "truncated",
            FrameSkipReason::Malformed(_) => "malformed",
            FrameSkipReason::Departure(_) => "departure",
        }
    }
    /// Error detail for malformed frames, or the departure's text.
    #[getter]
    fn detail(&self) -> Option<String> {
        match &self.inner.reason {
            FrameSkipReason::Truncated => None,
            FrameSkipReason::Malformed(detail) => Some(detail.clone()),
            FrameSkipReason::Departure(departure) => Some(departure.to_string()),
        }
    }
    /// The departure a strict read refused, for a `"departure"` reason.
    #[getter]
    fn departure(&self) -> Option<PyRtcmDeparture> {
        match &self.inner.reason {
            FrameSkipReason::Departure(departure) => Some(PyRtcmDeparture {
                inner: departure.clone(),
                offset: Some(self.inner.offset),
            }),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmFrameSkip(offset={}, message_number={:?}, reason={:?})",
            self.inner.offset,
            self.inner.message_number,
            self.reason()
        )
    }
}

/// Diagnostics collected while scanning an RTCM byte stream.
#[pyclass(module = "sidereon._sidereon", name = "RtcmStreamDiagnostics")]
#[derive(Clone)]
pub struct PyRtcmStreamDiagnostics {
    inner: StreamDiagnostics,
}

impl PyRtcmStreamDiagnostics {
    pub(crate) fn from_inner(inner: StreamDiagnostics) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRtcmStreamDiagnostics {
    /// Bytes skipped while resynchronizing on the next valid frame.
    #[getter]
    fn resync_bytes(&self) -> usize {
        self.inner.resync_bytes
    }
    /// Preambles whose declared frame lay in the buffer but failed its
    /// CRC-24Q. Each also counts one resync byte.
    #[getter]
    fn crc_failures(&self) -> usize {
        self.inner.crc_failures
    }
    /// Departures read under `RtcmPolicy.LENIENT`, in stream order, each with
    /// its frame offset.
    #[getter]
    fn departures(&self) -> Vec<PyRtcmDeparture> {
        self.inner
            .departures
            .iter()
            .map(|departure| PyRtcmDeparture {
                inner: departure.departure.clone(),
                offset: Some(departure.offset),
            })
            .collect()
    }
    /// True when nothing was skipped, no CRC failed and no departure was
    /// read.
    #[getter]
    fn is_clean(&self) -> bool {
        self.inner.is_clean()
    }
    /// CRC-valid frames whose bodies failed typed decode, or that departed
    /// from the format under the strict policy.
    #[getter]
    fn skipped_frames(&self) -> Vec<PyRtcmFrameSkip> {
        self.inner
            .skipped_frames
            .iter()
            .cloned()
            .map(|inner| PyRtcmFrameSkip { inner })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmStreamDiagnostics(resync_bytes={}, crc_failures={}, skipped_frames={}, departures={})",
            self.inner.resync_bytes,
            self.inner.crc_failures,
            self.inner.skipped_frames.len(),
            self.inner.departures.len()
        )
    }
}

/// A decoded RTCM byte stream plus forgiving-parse diagnostics.
#[pyclass(module = "sidereon._sidereon", name = "RtcmStream")]
#[derive(Clone)]
pub struct PyRtcmStream {
    messages: Vec<Message>,
    diagnostics: StreamDiagnostics,
}

#[pymethods]
impl PyRtcmStream {
    /// Every decoded message, in stream order.
    #[getter]
    fn messages(&self) -> Vec<PyRtcmMessage> {
        self.messages
            .iter()
            .cloned()
            .map(|inner| PyRtcmMessage { inner })
            .collect()
    }
    /// Stream diagnostics for resynchronization and skipped frames.
    #[getter]
    fn diagnostics(&self) -> PyRtcmStreamDiagnostics {
        PyRtcmStreamDiagnostics {
            inner: self.diagnostics.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RtcmStream(messages={}, resync_bytes={}, skipped_frames={})",
            self.messages.len(),
            self.diagnostics.resync_bytes,
            self.diagnostics.skipped_frames.len()
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
    pub(crate) inner: Message,
}

impl PyRtcmMessage {
    pub(crate) fn from_inner(inner: Message) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyRtcmMessage {
    /// The RTCM message number this IR encodes to.
    #[getter]
    fn message_number(&self) -> u16 {
        self.inner.message_number()
    }

    /// Stable variant tag for the decoded message family.
    #[getter]
    fn kind(&self) -> String {
        match &self.inner {
            Message::LegacyObservations(_) => "legacy_observations".to_owned(),
            Message::StationCoordinates(_) => "station_coordinates".to_owned(),
            Message::AntennaDescriptor(_) => "antenna_descriptor".to_owned(),
            Message::SystemParameters(_) => "system_parameters".to_owned(),
            Message::Text(_) => "text".to_owned(),
            Message::NetworkAuxiliaryStation(_) => "network_auxiliary_station".to_owned(),
            Message::NetworkCorrectionDifferences(_) => "network_correction_differences".to_owned(),
            Message::HelmertTransformation(_) => "helmert_transformation".to_owned(),
            Message::ResidualGrid(_) => "residual_grid".to_owned(),
            Message::Projection(_) => "projection".to_owned(),
            Message::NetworkResiduals(_) => "network_residuals".to_owned(),
            Message::PhysicalReferenceStation(_) => "physical_reference_station".to_owned(),
            Message::FkpGradients(_) => "fkp_gradients".to_owned(),
            Message::GpsEphemeris(_) => "gps_ephemeris".to_owned(),
            Message::GlonassEphemeris(_) => "glonass_ephemeris".to_owned(),
            Message::BeidouEphemeris(_) => "beidou_ephemeris".to_owned(),
            Message::QzssEphemeris(_) => "qzss_ephemeris".to_owned(),
            Message::GalileoFnavEphemeris(_) => "galileo_fnav_ephemeris".to_owned(),
            Message::GalileoInavEphemeris(_) => "galileo_inav_ephemeris".to_owned(),
            Message::NavicEphemeris(_) => "navic_ephemeris".to_owned(),
            Message::GlonassCodePhaseBiases(_) => "glonass_code_phase_biases".to_owned(),
            Message::Msm(_) => "msm".to_owned(),
            Message::Ssr(_) => "ssr".to_owned(),
            Message::SsrVtec(_) => "ssr_vtec".to_owned(),
            Message::Unsupported(_) => "unsupported".to_owned(),
        }
    }

    #[staticmethod]
    fn from_legacy_observations(payload: &PyRtcmLegacyObservations) -> Self {
        Self {
            inner: Message::LegacyObservations(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_system_parameters(payload: &PyRtcmSystemParameters) -> Self {
        Self {
            inner: Message::SystemParameters(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_text_message(payload: &PyRtcmTextMessage) -> Self {
        Self {
            inner: Message::Text(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_network_auxiliary_station(payload: &PyRtcmNetworkAuxiliaryStation) -> Self {
        Self {
            inner: Message::NetworkAuxiliaryStation(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_network_correction_differences(payload: &PyRtcmNetworkCorrectionDifferences) -> Self {
        Self {
            inner: Message::NetworkCorrectionDifferences(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_network_residuals(payload: &PyRtcmNetworkResiduals) -> Self {
        Self {
            inner: Message::NetworkResiduals(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_physical_reference_station(payload: &PyRtcmPhysicalReferenceStation) -> Self {
        Self {
            inner: Message::PhysicalReferenceStation(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_fkp_gradients(payload: &PyRtcmFkpGradients) -> Self {
        Self {
            inner: Message::FkpGradients(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_helmert_transformation(payload: &PyRtcmHelmertTransformation) -> Self {
        Self {
            inner: Message::HelmertTransformation(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_residual_grid(payload: &PyRtcmResidualGrid) -> Self {
        Self {
            inner: Message::ResidualGrid(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_projection(payload: &PyRtcmProjection) -> Self {
        Self {
            inner: Message::Projection(payload.inner.clone()),
        }
    }

    #[getter]
    fn legacy_observations(&self) -> Option<PyRtcmLegacyObservations> {
        match &self.inner {
            Message::LegacyObservations(value) => Some(PyRtcmLegacyObservations {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn system_parameters(&self) -> Option<PyRtcmSystemParameters> {
        match &self.inner {
            Message::SystemParameters(value) => Some(PyRtcmSystemParameters {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn text_message(&self) -> Option<PyRtcmTextMessage> {
        match &self.inner {
            Message::Text(value) => Some(PyRtcmTextMessage {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn network_auxiliary_station(&self) -> Option<PyRtcmNetworkAuxiliaryStation> {
        match &self.inner {
            Message::NetworkAuxiliaryStation(value) => Some(PyRtcmNetworkAuxiliaryStation {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn network_correction_differences(&self) -> Option<PyRtcmNetworkCorrectionDifferences> {
        match &self.inner {
            Message::NetworkCorrectionDifferences(value) => {
                Some(PyRtcmNetworkCorrectionDifferences {
                    inner: value.clone(),
                })
            }
            _ => None,
        }
    }

    #[getter]
    fn network_residuals(&self) -> Option<PyRtcmNetworkResiduals> {
        match &self.inner {
            Message::NetworkResiduals(value) => Some(PyRtcmNetworkResiduals {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn physical_reference_station(&self) -> Option<PyRtcmPhysicalReferenceStation> {
        match &self.inner {
            Message::PhysicalReferenceStation(value) => Some(PyRtcmPhysicalReferenceStation {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn fkp_gradients(&self) -> Option<PyRtcmFkpGradients> {
        match &self.inner {
            Message::FkpGradients(value) => Some(PyRtcmFkpGradients {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn helmert_transformation(&self) -> Option<PyRtcmHelmertTransformation> {
        match &self.inner {
            Message::HelmertTransformation(value) => Some(PyRtcmHelmertTransformation {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn residual_grid(&self) -> Option<PyRtcmResidualGrid> {
        match &self.inner {
            Message::ResidualGrid(value) => Some(PyRtcmResidualGrid {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn projection(&self) -> Option<PyRtcmProjection> {
        match &self.inner {
            Message::Projection(value) => Some(PyRtcmProjection {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The station-coordinates payload, or `None` for another variant.
    #[getter]
    fn station_coordinates(&self) -> Option<PyRtcmStationCoordinates> {
        match &self.inner {
            Message::StationCoordinates(value) => Some(PyRtcmStationCoordinates {
                inner: value.clone(),
            }),
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
            Message::GpsEphemeris(value) => Some(PyRtcmGpsEphemeris {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The GLONASS ephemeris payload, or `None` for another variant.
    #[getter]
    fn glonass_ephemeris(&self) -> Option<PyRtcmGlonassEphemeris> {
        match &self.inner {
            Message::GlonassEphemeris(value) => Some(PyRtcmGlonassEphemeris {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The BeiDou ephemeris payload, or `None` for another variant.
    #[getter]
    fn beidou_ephemeris(&self) -> Option<PyRtcmBeidouEphemeris> {
        match &self.inner {
            Message::BeidouEphemeris(value) => Some(PyRtcmBeidouEphemeris {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The QZSS ephemeris payload, or `None` for another variant.
    #[getter]
    fn qzss_ephemeris(&self) -> Option<PyRtcmQzssEphemeris> {
        match &self.inner {
            Message::QzssEphemeris(value) => Some(PyRtcmQzssEphemeris {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The Galileo F/NAV ephemeris payload, or `None` for another variant.
    #[getter]
    fn galileo_fnav_ephemeris(&self) -> Option<PyRtcmGalileoFnavEphemeris> {
        match &self.inner {
            Message::GalileoFnavEphemeris(value) => Some(PyRtcmGalileoFnavEphemeris {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    /// The Galileo I/NAV ephemeris payload, or `None` for another variant.
    #[getter]
    fn galileo_inav_ephemeris(&self) -> Option<PyRtcmGalileoInavEphemeris> {
        match &self.inner {
            Message::GalileoInavEphemeris(value) => Some(PyRtcmGalileoInavEphemeris {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn navic_ephemeris(&self) -> Option<PyRtcmNavicEphemeris> {
        match &self.inner {
            Message::NavicEphemeris(value) => Some(PyRtcmNavicEphemeris {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn glonass_code_phase_biases(&self) -> Option<PyRtcmGlonassCodePhaseBiases> {
        match &self.inner {
            Message::GlonassCodePhaseBiases(value) => Some(PyRtcmGlonassCodePhaseBiases {
                inner: value.clone(),
            }),
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

    #[getter]
    fn ssr_vtec(&self) -> Option<PyRtcmSsrVtecMessage> {
        match &self.inner {
            Message::SsrVtec(value) => Some(PyRtcmSsrVtecMessage {
                inner: value.clone(),
            }),
            _ => None,
        }
    }

    #[getter]
    fn ssr(&self) -> Option<PyRtcmSsrMessage> {
        match &self.inner {
            Message::Ssr(value) => Some(PyRtcmSsrMessage {
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
            inner: Message::StationCoordinates(payload.inner.clone()),
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
            inner: Message::GpsEphemeris(payload.inner.clone()),
        }
    }

    /// Wrap a 1020 GLONASS ephemeris payload as a message.
    #[staticmethod]
    fn from_glonass_ephemeris(payload: &PyRtcmGlonassEphemeris) -> Self {
        Self {
            inner: Message::GlonassEphemeris(payload.inner.clone()),
        }
    }

    /// Wrap a 1042 BeiDou ephemeris payload as a message.
    #[staticmethod]
    fn from_beidou_ephemeris(payload: &PyRtcmBeidouEphemeris) -> Self {
        Self {
            inner: Message::BeidouEphemeris(payload.inner.clone()),
        }
    }

    /// Wrap a 1044 QZSS ephemeris payload as a message.
    #[staticmethod]
    fn from_qzss_ephemeris(payload: &PyRtcmQzssEphemeris) -> Self {
        Self {
            inner: Message::QzssEphemeris(payload.inner.clone()),
        }
    }

    /// Wrap a 1045 Galileo F/NAV ephemeris payload as a message.
    #[staticmethod]
    fn from_galileo_fnav_ephemeris(payload: &PyRtcmGalileoFnavEphemeris) -> Self {
        Self {
            inner: Message::GalileoFnavEphemeris(payload.inner.clone()),
        }
    }

    /// Wrap a 1046 Galileo I/NAV ephemeris payload as a message.
    #[staticmethod]
    fn from_galileo_inav_ephemeris(payload: &PyRtcmGalileoInavEphemeris) -> Self {
        Self {
            inner: Message::GalileoInavEphemeris(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_navic_ephemeris(payload: &PyRtcmNavicEphemeris) -> Self {
        Self {
            inner: Message::NavicEphemeris(payload.inner.clone()),
        }
    }

    #[staticmethod]
    fn from_glonass_code_phase_biases(payload: &PyRtcmGlonassCodePhaseBiases) -> Self {
        Self {
            inner: Message::GlonassCodePhaseBiases(payload.inner.clone()),
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
    ///
    /// A message whose fields its bit layout cannot state - an MSM satellite
    /// id outside `1..=64` or signal id outside `1..=32`, a satellite or cell
    /// listed twice, a signal whose satellite is not listed, an ephemeris or
    /// SSR satellite field wider than the message's - raises
    /// `RtcmEncodeError` with the core's reason.
    ///
    /// Under `RtcmPolicy.LENIENT` a message holding a departure (trailing
    /// bits, an MSM cell mask over 64 bits, a short SSR body) is written as
    /// held; `encode_with_policy` also returns the departures.
    #[pyo3(signature = (policy=PyRtcmPolicy::STRICT))]
    fn encode<'py>(&self, py: Python<'py>, policy: PyRtcmPolicy) -> PyResult<Bound<'py, PyBytes>> {
        let (body, _) = self
            .inner
            .encode_with_policy(policy.into())
            .map_err(|err| to_rtcm_encode_err(py, err))?;
        Ok(PyBytes::new(py, &body))
    }

    /// Encode this message under `policy`, returning the body and the
    /// departures written (always empty under `RtcmPolicy.STRICT`, which
    /// refuses them).
    fn encode_with_policy<'py>(
        &self,
        py: Python<'py>,
        policy: PyRtcmPolicy,
    ) -> PyResult<(Bound<'py, PyBytes>, Vec<PyRtcmDeparture>)> {
        let (body, written) = self
            .inner
            .encode_with_policy(policy.into())
            .map_err(|err| to_rtcm_encode_err(py, err))?;
        Ok((PyBytes::new(py, &body), departures(written)))
    }

    /// Encode this message and wrap it in a fresh RTCM transport frame.
    ///
    /// A body the message cannot be encoded to raises `RtcmEncodeError`, as
    /// `encode` does; a body past the frame length limit raises
    /// `RtcmParseError`.
    fn to_frame<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        let body = self
            .inner
            .encode()
            .map_err(|error| to_rtcm_encode_err(py, error))?;
        let frame = core_encode_frame(&body).map_err(to_rtcm_err)?;
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

/// Decode a byte buffer of RTCM frames in full into the message IR.
///
/// Every byte has to belong to a CRC-valid frame whose body decodes under the
/// strict policy: a stray byte, a CRC-24Q failure, a trailing partial frame or
/// a frame that does not decode raises `RtcmParseError` naming it.
/// `decode_rtcm_stream` reads a noisy stream frame by frame and reports each
/// skip instead.
#[pyfunction]
fn decode_rtcm(data: &[u8]) -> PyResult<Vec<PyRtcmMessage>> {
    Ok(core_decode_messages(data)
        .map_err(to_rtcm_err)?
        .into_iter()
        .map(|inner| PyRtcmMessage { inner })
        .collect())
}

/// Decode every CRC-valid frame and return both messages and stream
/// diagnostics: resynchronized bytes, CRC-24Q failures, skipped frames with
/// their reasons, and under `RtcmPolicy.LENIENT` the departures read.
#[pyfunction]
#[pyo3(signature = (data, policy=PyRtcmPolicy::STRICT))]
fn decode_rtcm_stream(data: &[u8], policy: PyRtcmPolicy) -> PyRtcmStream {
    let stream = core_decode_stream(data, policy.into());
    PyRtcmStream {
        messages: stream.messages,
        diagnostics: stream.diagnostics,
    }
}

/// Decode a single RTCM 3 message body (the bytes between a frame's length word
/// and its CRC). Raises `RtcmParseError` on a truncated body of a recognized
/// type, and under `RtcmPolicy.STRICT` on a departure from the format; an
/// unrecognized message number decodes to the `"unsupported"` variant.
#[pyfunction]
#[pyo3(signature = (body, policy=PyRtcmPolicy::STRICT))]
fn decode_rtcm_message(body: &[u8], policy: PyRtcmPolicy) -> PyResult<PyRtcmMessage> {
    Message::decode_with_policy(body, policy.into())
        .map(|(inner, _)| PyRtcmMessage { inner })
        .map_err(to_rtcm_err)
}

/// Decode a single RTCM 3 message body under `policy`, returning the message
/// and the departures read (always empty under `RtcmPolicy.STRICT`, which
/// refuses them).
#[pyfunction]
fn decode_rtcm_message_with_policy(
    body: &[u8],
    policy: PyRtcmPolicy,
) -> PyResult<(PyRtcmMessage, Vec<PyRtcmDeparture>)> {
    let (inner, read) = Message::decode_with_policy(body, policy.into()).map_err(to_rtcm_err)?;
    Ok((PyRtcmMessage { inner }, departures(read)))
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

/// Minimum continuous-lock time for an MSM lock indicator, in milliseconds.
#[pyfunction]
fn rtcm_minimum_lock_time_ms(kind: &str, indicator: u16) -> PyResult<Option<u32>> {
    Ok(core_minimum_lock_time_ms(parse_msm_kind(kind)?, indicator))
}

/// Derive the RINEX LLI digit for one signal cell.
///
/// Pass `elapsed_ms=None` for the first observation. When `elapsed_ms` is set,
/// `previous_min_lock_time_ms=None` represents a previous reserved indicator.
#[pyfunction]
#[pyo3(signature = (
    previous_min_lock_time_ms,
    elapsed_ms,
    current_min_lock_time_ms,
    half_cycle_ambiguity,
))]
fn rtcm_derive_lli(
    previous_min_lock_time_ms: Option<u32>,
    elapsed_ms: Option<u64>,
    current_min_lock_time_ms: Option<u32>,
    half_cycle_ambiguity: bool,
) -> u8 {
    let previous = elapsed_ms.map(|elapsed_ms| PreviousLock {
        min_lock_time_ms: previous_min_lock_time_ms,
        elapsed_ms,
    });
    core_derive_lli(previous, current_min_lock_time_ms, half_cycle_ambiguity)
}

/// Elapsed milliseconds between two raw MSM epoch-time fields for one system.
#[pyfunction]
fn rtcm_msm_epoch_dt_ms(system: &str, previous: u32, current: u32) -> PyResult<u64> {
    Ok(core_msm_epoch_dt_ms(
        parse_system(system)?,
        previous,
        current,
    ))
}

/// RINEX observation-code suffix for an MSM signal id, or `None` if reserved.
#[pyfunction]
fn rtcm_msm_signal_rinex_code(system: &str, signal_id: u8) -> PyResult<Option<&'static str>> {
    Ok(core_msm_signal_rinex_code(parse_system(system)?, signal_id))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRtcmHelmertTransformation>()?;
    m.add_class::<PyRtcmResidualGrid>()?;
    m.add_class::<PyRtcmProjection>()?;
    m.add_class::<PyRtcmLegacyL1>()?;
    m.add_class::<PyRtcmLegacyL2>()?;
    m.add_class::<PyRtcmLegacySatellite>()?;
    m.add_class::<PyRtcmLegacyObservations>()?;
    m.add_class::<PyRtcmMessageAnnouncement>()?;
    m.add_class::<PyRtcmSystemParameters>()?;
    m.add_class::<PyRtcmTextMessage>()?;
    m.add_class::<PyRtcmNetworkAuxiliaryStation>()?;
    m.add_class::<PyRtcmNetworkCorrectionDifferences>()?;
    m.add_class::<PyRtcmNetworkResiduals>()?;
    m.add_class::<PyRtcmPhysicalReferenceStation>()?;
    m.add_class::<PyRtcmFkpGradients>()?;
    m.add_class::<PyRtcmStationCoordinates>()?;
    m.add_class::<PyRtcmAntennaDescriptor>()?;
    m.add_class::<PyRtcmGpsEphemeris>()?;
    m.add_class::<PyRtcmGlonassEphemeris>()?;
    m.add_class::<PyRtcmBeidouEphemeris>()?;
    m.add_class::<PyRtcmQzssEphemeris>()?;
    m.add_class::<PyRtcmGalileoFnavEphemeris>()?;
    m.add_class::<PyRtcmGalileoInavEphemeris>()?;
    m.add_class::<PyRtcmNavicEphemeris>()?;
    m.add_class::<PyRtcmGlonassCodePhaseBiases>()?;
    m.add_class::<PyRtcmMsmHeader>()?;
    m.add_class::<PyRtcmMsmSatellite>()?;
    m.add_class::<PyRtcmMsmSignal>()?;
    m.add_class::<PyRtcmMsmMessage>()?;
    m.add_class::<PyRtcmSsrVtecLayer>()?;
    m.add_class::<PyRtcmSsrVtecLayerEvaluation>()?;
    m.add_class::<PyRtcmSsrVtecEvaluation>()?;
    m.add_class::<PyRtcmSsrVtecMessage>()?;
    m.add_class::<PyRtcmSsrMessage>()?;
    m.add_class::<PyRtcmUnsupportedMessage>()?;
    m.add_class::<PyRtcmCellLli>()?;
    m.add_class::<PyRtcmLockTimeTracker>()?;
    m.add_class::<PyRtcmFrameSkip>()?;
    m.add_class::<PyRtcmStreamDiagnostics>()?;
    m.add_class::<PyRtcmStream>()?;
    m.add_class::<PyRtcmMessage>()?;
    m.add_class::<PyRtcmPolicy>()?;
    m.add_class::<PyRtcmDeparture>()?;
    m.add("RTCM_MSM_ROUGH_RANGE_INVALID", MSM_ROUGH_RANGE_INVALID)?;
    m.add(
        "RTCM_MSM4_FINE_PSEUDORANGE_INVALID",
        MSM4_FINE_PSEUDORANGE_INVALID,
    )?;
    m.add(
        "RTCM_MSM4_FINE_PHASE_RANGE_INVALID",
        MSM4_FINE_PHASE_RANGE_INVALID,
    )?;
    m.add(
        "RTCM_MSM7_FINE_PSEUDORANGE_INVALID",
        MSM7_FINE_PSEUDORANGE_INVALID,
    )?;
    m.add(
        "RTCM_MSM7_FINE_PHASE_RANGE_INVALID",
        MSM7_FINE_PHASE_RANGE_INVALID,
    )?;
    m.add(
        "RTCM_MSM_ROUGH_PHASE_RANGE_RATE_INVALID",
        MSM_ROUGH_PHASE_RANGE_RATE_INVALID,
    )?;
    m.add(
        "RTCM_MSM_FINE_PHASE_RANGE_RATE_INVALID",
        MSM_FINE_PHASE_RANGE_RATE_INVALID,
    )?;
    for (name, value) in [
        ("XN_DOT", GlonassEphemeris::NEGATIVE_ZERO_XN_DOT),
        ("XN", GlonassEphemeris::NEGATIVE_ZERO_XN),
        ("XN_DOT_DOT", GlonassEphemeris::NEGATIVE_ZERO_XN_DOT_DOT),
        ("YN_DOT", GlonassEphemeris::NEGATIVE_ZERO_YN_DOT),
        ("YN", GlonassEphemeris::NEGATIVE_ZERO_YN),
        ("YN_DOT_DOT", GlonassEphemeris::NEGATIVE_ZERO_YN_DOT_DOT),
        ("ZN_DOT", GlonassEphemeris::NEGATIVE_ZERO_ZN_DOT),
        ("ZN", GlonassEphemeris::NEGATIVE_ZERO_ZN),
        ("ZN_DOT_DOT", GlonassEphemeris::NEGATIVE_ZERO_ZN_DOT_DOT),
        ("GAMMA_N", GlonassEphemeris::NEGATIVE_ZERO_GAMMA_N),
        ("TAU_N", GlonassEphemeris::NEGATIVE_ZERO_TAU_N),
        ("DELTA_TAU_N", GlonassEphemeris::NEGATIVE_ZERO_DELTA_TAU_N),
        ("TAU_C", GlonassEphemeris::NEGATIVE_ZERO_TAU_C),
        ("M_TAU_GPS", GlonassEphemeris::NEGATIVE_ZERO_M_TAU_GPS),
    ] {
        m.add(format!("RTCM_GLONASS_NEGATIVE_ZERO_{name}"), value)?;
    }
    m.add("RTCM_LLI_LOSS_OF_LOCK", LLI_LOSS_OF_LOCK)?;
    m.add("RTCM_LLI_HALF_CYCLE", LLI_HALF_CYCLE)?;
    m.add_function(wrap_pyfunction!(decode_rtcm, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rtcm_stream, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rtcm_message, m)?)?;
    m.add_function(wrap_pyfunction!(decode_rtcm_message_with_policy, m)?)?;
    m.add_function(wrap_pyfunction!(encode_rtcm_frame, m)?)?;
    m.add_function(wrap_pyfunction!(rtcm_message_number, m)?)?;
    m.add_function(wrap_pyfunction!(rtcm_minimum_lock_time_ms, m)?)?;
    m.add_function(wrap_pyfunction!(rtcm_derive_lli, m)?)?;
    m.add_function(wrap_pyfunction!(rtcm_msm_epoch_dt_ms, m)?)?;
    m.add_function(wrap_pyfunction!(rtcm_msm_signal_rinex_code, m)?)?;
    Ok(())
}
