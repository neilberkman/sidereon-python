"""RTCM 3.x decode / encode / round-trip delegates to ``sidereon_core::rtcm``.

The frames below are emitted by the core encoder (a 1006 station-coordinates
message and a verbatim unsupported message), so decoding them through the binding
and re-encoding must reproduce the exact bytes.
"""

import pytest
import sidereon

# A 1006 station-coordinates frame (reference_station_id 2003) built by the core.
STATION_1006_FRAME = bytes.fromhex(
    "d300153ee7d30302aa3c6d183e4605ff0c02ef2b54843a98d8b487"
)
# A verbatim unsupported message (number 1230) framed by the core.
UNSUPPORTED_FRAME = bytes.fromhex("d300044ceabcdeb63c1b")
# The two frames concatenated, to exercise the stream decoder.
STREAM_TWO = STATION_1006_FRAME + UNSUPPORTED_FRAME


def test_decode_station_1006_fields():
    messages = sidereon.decode_rtcm(STATION_1006_FRAME)
    assert len(messages) == 1
    msg = messages[0]
    assert msg.kind == "station_coordinates"
    assert msg.message_number == 1006
    station = msg.station_coordinates
    assert station is not None
    assert station.reference_station_id == 2003
    assert station.gps_indicator is True
    assert station.galileo_indicator is False
    assert station.x_m == pytest.approx(1144602.14, abs=1e-2)
    assert station.y_m == pytest.approx(-741513.65, abs=1e-2)
    assert station.z_m == pytest.approx(1260252.89, abs=1e-2)
    assert station.antenna_height_m == pytest.approx(1.5, abs=1e-6)
    # The non-matching typed accessors return None.
    assert msg.gps_ephemeris is None
    assert msg.msm is None


def test_station_round_trips_to_exact_frame_bytes():
    msg = sidereon.decode_rtcm(STATION_1006_FRAME)[0]
    assert msg.to_frame() == STATION_1006_FRAME
    # The body re-decodes to the same message number.
    body = msg.encode()
    assert sidereon.rtcm_message_number(body) == 1006
    assert sidereon.decode_rtcm_message(body).message_number == 1006
    # Re-framing the body reproduces the original frame.
    assert sidereon.encode_rtcm_frame(body) == STATION_1006_FRAME


def test_stream_decodes_both_frames_in_order():
    messages = sidereon.decode_rtcm(STREAM_TWO)
    assert [m.kind for m in messages] == ["station_coordinates", "unsupported"]
    assert [m.message_number for m in messages] == [1006, 1230]


def test_unsupported_message_preserves_body():
    msg = sidereon.decode_rtcm(UNSUPPORTED_FRAME)[0]
    assert msg.kind == "unsupported"
    assert msg.message_number == 1230
    unsupported = msg.unsupported
    assert unsupported is not None
    assert isinstance(unsupported.body, bytes)
    # The whole unsupported frame round-trips verbatim.
    assert msg.to_frame() == UNSUPPORTED_FRAME


def test_decode_rtcm_message_rejects_truncated_body():
    # A body of fewer than 12 bits cannot yield a message number.
    with pytest.raises(sidereon.RtcmParseError):
        sidereon.rtcm_message_number(b"\x00")


# --- from-scratch construction -> encode -> decode round-trips --------------
#
# The core `rtcm::Message` is now constructible field-by-field (the per-type
# encoders are public), so the binding exposes a constructor for each payload and
# `RtcmMessage.from_*` wrappers. Each message built from fields must encode to a
# body that decodes back to the same fields, with a byte-identical re-encode.


def _assert_body_round_trips(msg, expected_number):
    body = msg.encode()
    decoded = sidereon.decode_rtcm_message(body)
    assert decoded.message_number == expected_number
    # The decoded message re-encodes to the exact same body bytes.
    assert decoded.encode() == body
    # The full transport frame also round-trips through the stream decoder.
    frame = msg.to_frame()
    assert sidereon.decode_rtcm(frame)[0].encode() == body
    return decoded


def test_construct_station_1006_round_trips():
    station = sidereon.RtcmStationCoordinates(
        message_number=1006,
        reference_station_id=2003,
        itrf_realization_year=21,
        gps_indicator=True,
        glonass_indicator=False,
        galileo_indicator=True,
        reference_station_indicator=False,
        ecef_x=1144602140,
        ecef_y=-741513650,
        ecef_z=1260252890,
        single_receiver_oscillator=True,
        reserved=False,
        quarter_cycle_indicator=0,
        antenna_height=15000,
    )
    msg = sidereon.RtcmMessage.from_station_coordinates(station)
    assert msg.kind == "station_coordinates"
    decoded = _assert_body_round_trips(msg, 1006)
    out = decoded.station_coordinates
    assert out.reference_station_id == 2003
    assert out.gps_indicator is True
    assert out.galileo_indicator is True
    assert out.ecef_x == 1144602140
    assert out.ecef_y == -741513650
    assert out.ecef_z == 1260252890
    assert out.antenna_height == 15000


def test_construct_station_1005_omits_antenna_height():
    station = sidereon.RtcmStationCoordinates(
        message_number=1005,
        reference_station_id=7,
        itrf_realization_year=0,
        gps_indicator=True,
        glonass_indicator=True,
        galileo_indicator=False,
        reference_station_indicator=False,
        ecef_x=10,
        ecef_y=20,
        ecef_z=30,
        single_receiver_oscillator=False,
        reserved=False,
        quarter_cycle_indicator=0,
    )
    decoded = _assert_body_round_trips(
        sidereon.RtcmMessage.from_station_coordinates(station), 1005
    )
    assert decoded.station_coordinates.antenna_height is None


def test_construct_antenna_descriptor_1033_round_trips():
    descriptor = sidereon.RtcmAntennaDescriptor(
        message_number=1033,
        reference_station_id=2003,
        antenna_descriptor="TRM59800.00",
        antenna_setup_id=1,
        antenna_serial_number="NONE",
        receiver_type="SEPT POLARX5",
        receiver_firmware_version="5.3.2",
        receiver_serial_number="3001234",
    )
    decoded = _assert_body_round_trips(
        sidereon.RtcmMessage.from_antenna_descriptor(descriptor), 1033
    )
    out = decoded.antenna_descriptor
    assert out.antenna_descriptor == "TRM59800.00"
    assert out.antenna_serial_number == "NONE"
    assert out.receiver_type == "SEPT POLARX5"
    assert out.receiver_firmware_version == "5.3.2"
    assert out.receiver_serial_number == "3001234"


def _zero_gps_ephemeris():
    # All numeric fields zero so the construct/encode/decode round-trip is a
    # clean bijection regardless of per-field bit widths; only the satellite id
    # and a couple of small fields carry non-zero values.
    return sidereon.RtcmGpsEphemeris(
        satellite_id=5,
        week_number=100,
        sv_accuracy=0,
        code_on_l2=0,
        idot=0,
        iode=50,
        t_oc=0,
        a_f2=0,
        a_f1=0,
        a_f0=0,
        iodc=0,
        c_rs=0,
        delta_n=0,
        m0=0,
        c_uc=0,
        eccentricity=0,
        c_us=0,
        sqrt_a=0,
        t_oe=0,
        c_ic=0,
        omega0=0,
        c_is=0,
        i0=0,
        c_rc=0,
        omega=0,
        omega_dot=0,
        t_gd=0,
        sv_health=0,
        l2_p_data_flag=False,
        fit_interval=False,
    )


def test_construct_gps_ephemeris_1019_round_trips():
    decoded = _assert_body_round_trips(
        sidereon.RtcmMessage.from_gps_ephemeris(_zero_gps_ephemeris()), 1019
    )
    out = decoded.gps_ephemeris
    assert out.satellite_id == 5
    assert out.week_number == 100
    assert out.iode == 50
    assert out.m0 == 0
    assert out.l2_p_data_flag is False


def _zero_glonass_ephemeris():
    return sidereon.RtcmGlonassEphemeris(
        satellite_id=7,
        frequency_channel=0,
        almanac_health=False,
        almanac_health_availability=False,
        p1=0,
        t_k=0,
        b_n_msb=False,
        p2=False,
        t_b=0,
        xn_dot=0,
        xn=0,
        xn_dot_dot=0,
        yn_dot=0,
        yn=0,
        yn_dot_dot=0,
        zn_dot=0,
        zn=0,
        zn_dot_dot=0,
        p3=False,
        gamma_n=0,
        m_p=0,
        m_l_n_third=False,
        tau_n=0,
        delta_tau_n=0,
        e_n=0,
        m_p4=False,
        m_f_t=0,
        m_n_t=0,
        m_m=0,
        additional_data_available=False,
        n_a=0,
        tau_c=0,
        m_n4=0,
        m_tau_gps=0,
        m_l_n_fifth=False,
        reserved=0,
    )


def test_construct_glonass_ephemeris_1020_round_trips():
    decoded = _assert_body_round_trips(
        sidereon.RtcmMessage.from_glonass_ephemeris(_zero_glonass_ephemeris()), 1020
    )
    out = decoded.glonass_ephemeris
    assert out.satellite_id == 7
    assert out.frequency_channel == 0


def _msm4_message():
    header = sidereon.RtcmMsmHeader(
        reference_station_id=2003,
        epoch_time=100,
        multiple_message=False,
        iods=0,
        reserved=0,
        clock_steering=0,
        external_clock=0,
        divergence_free_smoothing=False,
        smoothing_interval=0,
    )
    sat = sidereon.RtcmMsmSatellite(
        id=5,
        rough_range_ms=100,
        rough_range_mod1=200,
    )
    sig = sidereon.RtcmMsmSignal(
        satellite_id=5,
        signal_id=2,
        fine_pseudorange=100,
        fine_phase_range=200,
        lock_time_indicator=5,
        half_cycle_ambiguity=False,
        cnr=30,
    )
    return sidereon.RtcmMsmMessage(
        message_number=1074,
        system="G",
        kind="msm4",
        header=header,
        satellites=[sat],
        signals=[sig],
    )


def test_construct_msm4_round_trips():
    msg = sidereon.RtcmMessage.from_msm(_msm4_message())
    assert msg.kind == "msm"
    decoded = _assert_body_round_trips(msg, 1074)
    out = decoded.msm
    assert out.system == "G"
    assert out.kind == "msm4"
    assert out.header.epoch_time == 100
    assert [s.id for s in out.satellites] == [5]
    assert out.satellites[0].rough_range_ms == 100
    assert [(s.satellite_id, s.signal_id) for s in out.signals] == [(5, 2)]
    assert out.signals[0].cnr == 30


def test_construct_msm7_round_trips():
    header = sidereon.RtcmMsmHeader(
        reference_station_id=2003,
        epoch_time=100,
        multiple_message=False,
        iods=0,
        reserved=0,
        clock_steering=0,
        external_clock=0,
        divergence_free_smoothing=False,
        smoothing_interval=0,
    )
    # MSM7 carries the extended satellite info and the phase-range-rate cells.
    sat = sidereon.RtcmMsmSatellite(
        id=9,
        rough_range_ms=80,
        rough_range_mod1=300,
        extended_info=3,
        rough_phase_range_rate_m_s=-50,
    )
    sig = sidereon.RtcmMsmSignal(
        satellite_id=9,
        signal_id=4,
        fine_pseudorange=500,
        fine_phase_range=600,
        lock_time_indicator=20,
        half_cycle_ambiguity=True,
        cnr=400,
        fine_phase_range_rate=-7,
    )
    message = sidereon.RtcmMsmMessage(
        message_number=1077,
        system="G",
        kind="msm7",
        header=header,
        satellites=[sat],
        signals=[sig],
    )
    decoded = _assert_body_round_trips(sidereon.RtcmMessage.from_msm(message), 1077)
    out = decoded.msm
    assert out.kind == "msm7"
    assert out.satellites[0].extended_info == 3
    assert out.satellites[0].rough_phase_range_rate_m_s == -50
    assert out.signals[0].half_cycle_ambiguity is True
    assert out.signals[0].fine_phase_range_rate == -7


def test_construct_msm_rejects_bad_system_letter():
    header = sidereon.RtcmMsmHeader(
        reference_station_id=1,
        epoch_time=0,
        multiple_message=False,
        iods=0,
        reserved=0,
        clock_steering=0,
        external_clock=0,
        divergence_free_smoothing=False,
        smoothing_interval=0,
    )
    with pytest.raises(ValueError):
        sidereon.RtcmMsmMessage(
            message_number=1074,
            system="Z",
            kind="msm4",
            header=header,
            satellites=[],
            signals=[],
        )
