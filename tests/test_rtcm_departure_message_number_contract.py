"""Regression for body-departure message-number projections."""

import sidereon


def _packed(fields):
    result = bytearray((sum(width for _value, width in fields) + 7) // 8)
    bit = 0
    for value, width in fields:
        for shift in range(width - 1, -1, -1):
            if value >> shift & 1:
                result[bit // 8] |= 1 << (7 - bit % 8)
            bit += 1
    return bytes(result)


def _order_above_degree():
    def twos(value):
        return value & 0xFFFF

    fields = [
        (1264, 12),
        (50_400, 20),
        (0, 4),
        (0, 1),
        (3, 4),
        (0x1234, 16),
        (2, 4),
        (7, 9),
        (0, 2),
        (45, 8),
        (0, 4),
        (1, 4),
        (twos(100), 16),
        (twos(-200), 16),
        (twos(300), 16),
        (twos(400), 16),
    ]
    _message, [departure] = sidereon.decode_rtcm_message_with_policy(
        _packed(fields), sidereon.RtcmPolicy.LENIENT
    )
    return departure


def _records_short():
    def satellite(satellite_id):
        return sidereon.RtcmLegacySatellite(
            satellite_id,
            None,
            sidereon.RtcmLegacyL1(False, 0, 0, 0, 0, 0),
            sidereon.RtcmLegacyL2(0, 0, 0, 0, 0),
        )

    payload = sidereon.RtcmLegacyObservations(
        1004,
        1,
        0,
        False,
        2,
        False,
        0,
        [satellite(1), satellite(2)],
    )
    body = sidereon.RtcmMessage.from_legacy_observations(payload).encode()[:30]
    _message, [departure] = sidereon.decode_rtcm_message_with_policy(
        body, sidereon.RtcmPolicy.LENIENT
    )
    return departure


def test_every_body_departure_projects_its_message_number():
    expected = [
        (_order_above_degree(), "order_exceeds_degree", 1264),
        (_records_short(), "records_short", 1004),
    ]
    for departure, kind, message_number in expected:
        assert departure.kind == kind
        assert departure.message_number == message_number
