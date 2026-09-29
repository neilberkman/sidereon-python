"""SBAS message fields that hold several four-bit or two-bit values come back
as a `list[int]`, as the stubs declare, not as `bytes`.

Provenance:
- The MT2 body is the captured RTKLIB `gw10_20110121.sbas` vector the core's
  `tests/sbas_real_vectors.rs` decodes, with the same expected values.
- The MT24 fields are the ones the core's `mt24_encode_uses_rtklib_offsets`
  test encodes.
- No MT6 or MT7 capture is committed, so those bodies are packed here from the
  field layout the core codec reads, with values chosen to differ slot by slot.
"""

import sidereon

PREAMBLE = 0x53
DATA_BITS = 212
BODY_BYTES = 29


def _body226(message_type, fields):
    """A 226-bit SBAS body, preamble and six-bit message type then the 212 data
    bits `fields` gives as `(value, width)` pairs, a negative value in two's
    complement, zero-padded to 29 bytes.
    """
    bits = f"{PREAMBLE:08b}{message_type:06b}"
    for value, width in fields:
        bits += format(value & ((1 << width) - 1), f"0{width}b")
    assert len(bits) == 8 + 6 + DATA_BITS
    bits += "0" * (BODY_BYTES * 8 - len(bits))
    return int(bits, 2).to_bytes(BODY_BYTES, "big")


def _decode(body):
    return sidereon.decode_sbas_block(body, sidereon.SbasWireForm.BODY226).message


def _assert_int_list(values, expected):
    assert type(values) is list
    assert all(type(value) is int for value in values)
    assert values == expected


def test_fast_corrections_udrei_is_a_list_of_ints():
    message = _decode(
        bytes.fromhex("5308DFFC010005FFC00DFFC009FFDFFC001FFDFFDFFFBABBBBBB9BBB80")
    )
    fast = message.fast_corrections
    assert fast is not None
    assert fast.message_type == 2
    assert fast.iodf == 0
    assert fast.iodp == 3
    assert fast.prc == [2047, 4, 1, 2047, 3, 2047, 2, 2047, 2047, 0, 2047, 2047, 2047]
    _assert_int_list(fast.udrei, [14, 14, 10, 14, 14, 14, 14, 14, 14, 6, 14, 14, 14])


def test_integrity_iodf_and_udrei_are_lists_of_ints():
    iodf = [0, 1, 2, 3]
    udrei = [(3 * slot + 1) % 16 for slot in range(51)]
    body = _body226(6, [(value, 2) for value in iodf] + [(v, 4) for v in udrei])

    integrity = _decode(body).integrity
    assert integrity is not None
    _assert_int_list(integrity.iodf, iodf)
    _assert_int_list(integrity.udrei, udrei)


def test_fast_degradation_ai_is_a_list_of_ints():
    ai = [(7 * slot + 2) % 16 for slot in range(51)]
    # System latency, IODP and the two reserved bits at message bit 20, then
    # the fifty-one indicators from bit 22 (RTKLIB `decode_sbstype7`).
    body = _body226(7, [(5, 4), (2, 2), (2, 2)] + [(value, 4) for value in ai])

    degradation = _decode(body).fast_degradation
    assert degradation is not None
    assert degradation.system_latency_s == 5
    assert degradation.iodp == 2
    assert degradation.reserved == [(2, 2)]
    _assert_int_list(degradation.ai, ai)


def test_mixed_fast_corrections_udrei_is_a_list_of_ints():
    prc = [-1, 2, -3, 4, -5, 6]
    udrei = [1, 2, 3, 4, 5, 6]
    fast_half = (
        [(value, 12) for value in prc]
        + [(value, 4) for value in udrei]
        # IODP, block id, IODF, then four spare bits.
        + [(2, 2), (3, 2), (1, 2), (0b1010, 4)]
    )
    # Velocity code 0: two records of index, IODE, dx, dy, dz and af0, then the
    # IODP and one spare bit.
    long_half = [
        (0, 1),
        (5, 6),
        (9, 8),
        (-10, 9),
        (11, 9),
        (-12, 9),
        (13, 10),
        (6, 6),
        (10, 8),
        (14, 9),
        (-15, 9),
        (16, 9),
        (-17, 10),
        (2, 2),
        (1, 1),
    ]

    mixed = _decode(_body226(24, fast_half + long_half)).mixed_corrections
    assert mixed is not None
    fast = mixed.fast
    assert fast.iodf == 1
    assert fast.iodp == 2
    assert fast.block_id == 3
    assert fast.prc == prc
    assert fast.reserved == [(0b1010, 4)]
    _assert_int_list(fast.udrei, udrei)
