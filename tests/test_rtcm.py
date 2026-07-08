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


def test_stream_diagnostics_surface_resync_bytes():
    stream = sidereon.decode_rtcm_stream(b"junk" + STREAM_TWO)
    assert [m.message_number for m in stream.messages] == [1006, 1230]
    assert stream.diagnostics.resync_bytes >= 4
    assert stream.diagnostics.skipped_frames == []


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


def _galileo_fnav_ephemeris():
    return sidereon.RtcmGalileoFnavEphemeris(
        satellite_id=3,
        week_number=1402,
        iod_nav=7,
        sisa=0,
        idot=0,
        t_oc=0,
        a_f2=0,
        a_f1=0,
        a_f0=0,
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
        bgd_e5a_e1=0,
        e5a_signal_health=0,
        e5a_data_validity=False,
        reserved=0,
    )


def _galileo_inav_ephemeris():
    return sidereon.RtcmGalileoInavEphemeris(
        satellite_id=3,
        week_number=1402,
        iod_nav=7,
        sisa_index=0,
        idot=0,
        t_oc=0,
        a_f2=0,
        a_f1=0,
        a_f0=0,
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
        bgd_e5a_e1=0,
        bgd_e5b_e1=0,
        e5b_signal_health=0,
        e5b_data_validity=False,
        e1b_signal_health=0,
        e1b_data_validity=False,
        reserved=0,
    )


def _beidou_ephemeris():
    return sidereon.RtcmBeidouEphemeris(
        satellite_id=19,
        week_number=1100,
        sv_urai=0,
        idot=0,
        aode=0,
        t_oc=0,
        a_f2=0,
        a_f1=0,
        a_f0=0,
        aodc=0,
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
        t_gd1=0,
        t_gd2=0,
        sv_health=False,
    )


def _qzss_ephemeris():
    return sidereon.RtcmQzssEphemeris(
        satellite_id=2,
        t_oc=0,
        a_f2=0,
        a_f1=0,
        a_f0=0,
        iode=0,
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
        idot=0,
        codes_on_l2=0,
        week_number=1023,
        ura=0,
        sv_health=0,
        t_gd=0,
        iodc=0,
        fit_interval=False,
    )


def test_construct_new_broadcast_ephemerides_round_trip():
    cases = [
        (
            sidereon.RtcmMessage.from_galileo_fnav_ephemeris(_galileo_fnav_ephemeris()),
            1045,
        ),
        (
            sidereon.RtcmMessage.from_galileo_inav_ephemeris(_galileo_inav_ephemeris()),
            1046,
        ),
        (sidereon.RtcmMessage.from_beidou_ephemeris(_beidou_ephemeris()), 1042),
        (sidereon.RtcmMessage.from_qzss_ephemeris(_qzss_ephemeris()), 1044),
    ]
    for message, number in cases:
        decoded = _assert_body_round_trips(message, number)
        assert decoded.message_number == number
    assert cases[0][0].kind == "galileo_fnav_ephemeris"
    assert cases[1][0].kind == "galileo_inav_ephemeris"
    assert cases[2][0].kind == "beidou_ephemeris"
    assert cases[3][0].kind == "qzss_ephemeris"


def test_decode_real_galileo_1046_frame_exposes_payload():
    frame = bytes.fromhex(
        "d3003f4160d5e8076b06c941e03ffed3ffe33917f3a490e984d2089bf4f4011030b0343aa813ab5d41efffb7e44fe8cfff5277d0b011a2416397fffffc2280140700800a8e"
    )
    message = sidereon.decode_rtcm(frame)[0]
    assert message.kind == "galileo_inav_ephemeris"
    eph = message.galileo_inav_ephemeris
    assert eph.satellite_id == 3
    assert eph.week_number == 1402
    assert eph.iod_nav == 7
    assert eph.sqrt_a > 2_800_000_000
    assert eph.satellite() == "E03"
    assert message.to_frame() == frame


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


def test_rtcm_lli_helpers_delegate_to_core_tables():
    assert sidereon.RTCM_LLI_LOSS_OF_LOCK == 1
    assert sidereon.RTCM_LLI_HALF_CYCLE == 2
    assert sidereon.rtcm_minimum_lock_time_ms("msm4", 0) == 0
    assert sidereon.rtcm_minimum_lock_time_ms("msm4", 5) == 512
    assert sidereon.rtcm_minimum_lock_time_ms("msm4", 16) is None
    assert sidereon.rtcm_minimum_lock_time_ms("msm7", 704) == 67_108_864
    assert sidereon.rtcm_minimum_lock_time_ms("msm7", 705) is None
    assert sidereon.rtcm_derive_lli(None, None, 0, True) == sidereon.RTCM_LLI_HALF_CYCLE
    assert (
        sidereon.rtcm_derive_lli(1024, 500, 512, False)
        == sidereon.RTCM_LLI_LOSS_OF_LOCK
    )
    assert (
        sidereon.rtcm_derive_lli(512, 600, 512, False) == sidereon.RTCM_LLI_LOSS_OF_LOCK
    )
    assert sidereon.rtcm_derive_lli(512, 512, 512, False) == 0
    assert sidereon.rtcm_msm_epoch_dt_ms("G", 604_799_000, 1_000) == 2_000
    assert sidereon.rtcm_msm_signal_rinex_code("G", 2) == "1C"
    assert sidereon.rtcm_msm_signal_rinex_code("G", 1) is None


def test_msm_signal_lock_time_helper():
    signal = _msm4_message().signals[0]
    assert signal.minimum_lock_time_ms("msm4") == 512
    assert signal.minimum_lock_time_ms("msm7") == 5


def test_rtcm_lock_time_tracker_derives_per_cell_lli():
    first = _msm4_message()
    second = sidereon.RtcmMsmMessage(
        message_number=1074,
        system="G",
        kind="msm4",
        header=sidereon.RtcmMsmHeader(
            reference_station_id=2003,
            epoch_time=700,
            multiple_message=False,
            iods=0,
            reserved=0,
            clock_steering=0,
            external_clock=0,
            divergence_free_smoothing=False,
            smoothing_interval=0,
        ),
        satellites=first.satellites,
        signals=first.signals,
    )
    tracker = sidereon.RtcmLockTimeTracker()
    assert [cell.lli for cell in tracker.observe(first)] == [0]
    out = tracker.observe(second)
    assert [(cell.satellite_id, cell.signal_id) for cell in out] == [(5, 2)]
    assert out[0].min_lock_time_ms == 512
    assert out[0].lli == sidereon.RTCM_LLI_LOSS_OF_LOCK
    tracker.reset()
    assert [cell.lli for cell in tracker.observe(second)] == [0]


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
