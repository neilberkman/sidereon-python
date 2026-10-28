"""GPS LNAV (legacy navigation message) codec.

`lnav_encode` / `lnav_decode` and the parity / TOW / subframe-id helpers are thin
wrappers over `sidereon_core::navigation::lnav`. The representative MEO GPS SV
parameter set mirrors the core `example/0` fixture, so the round trip asserts the
same recovery contract the core test asserts: integer fields exactly, scaled
fields to within their IS-GPS-200 LSB.
"""

import ast
import math
from pathlib import Path

import pytest
import sidereon

# A representative MEO GPS SV, matching the core LNAV `example/0`. Integer fields
# are ints; scaled fields are floats.
EXAMPLE = {
    "week_number": 290,
    "l2_code": 1,
    "l2_p_data_flag": 0,
    "ura_index": 0,
    "sv_health": 0,
    "iodc": 0x2AB,
    "tgd": -5.587935447692871e-9,
    "toc": 504_000,
    "af0": -1.234e-4,
    "af1": -3.5e-12,
    "af2": 0.0,
    "iode": 0xAB,
    "crs": -55.625,
    "delta_n": 1.56e-9,
    "m0": -0.35,
    "cuc": -1.2e-6,
    "eccentricity": 0.012,
    "cus": 8.3e-6,
    "sqrt_a": 5153.65,
    "toe": 504_000,
    "fit_interval_flag": 0,
    "aodo": 0,
    "cic": 5.0e-8,
    "omega0": -0.78,
    "cis": -2.1e-7,
    "i0": 0.305,
    "crc": 250.625,
    "omega": 0.95,
    "omega_dot": -8.1e-9,
    "idot": 1.5e-10,
}

TWO_POW_M5 = 2.0**-5
TWO_POW_M19 = 2.0**-19
TWO_POW_M29 = 2.0**-29
TWO_POW_M31 = 2.0**-31
TWO_POW_M33 = 2.0**-33
TWO_POW_M43 = 2.0**-43
TWO_POW_M55 = 2.0**-55

PARITY_TAPS = (
    (1, 2, 3, 5, 6, 10, 11, 12, 13, 14, 17, 18, 20, 23),
    (2, 3, 4, 6, 7, 11, 12, 13, 14, 15, 18, 19, 21, 24),
    (1, 3, 4, 5, 7, 8, 12, 13, 14, 15, 16, 19, 20, 22),
    (2, 4, 5, 6, 8, 9, 13, 14, 15, 16, 17, 20, 21, 23),
    (1, 3, 5, 6, 7, 9, 10, 14, 15, 16, 17, 18, 21, 22, 24),
    (3, 5, 6, 8, 9, 10, 11, 13, 15, 19, 22, 23, 24),
)
PARITY_MASKS = tuple(
    sum(1 << (24 - position) for position in positions) for positions in PARITY_TAPS
)


def _transmitted_subframe(words):
    """Add LNAV parity and the chained D30* data complement to 24-bit words."""
    previous_d29 = 0
    previous_d30 = 0
    bits = []
    for source_word in words:
        parity_bits = 0
        for index, mask in enumerate(PARITY_MASKS):
            previous = previous_d29 if index in (0, 2, 5) else previous_d30
            check = (source_word & mask).bit_count() % 2
            parity_bits = parity_bits * 2 + (check ^ previous)
        transmitted_data = source_word ^ (0xFFFFFF if previous_d30 else 0)
        transmitted_word = transmitted_data * 64 + parity_bits
        bits.extend((transmitted_word >> index) & 1 for index in range(29, -1, -1))
        previous_d29 = parity_bits >> 1 & 1
        previous_d30 = parity_bits & 1
    return bytes(bits)


def _encode():
    return sidereon.lnav_encode(EXAMPLE, tow=12_345)


def test_subframes_are_300_bit_binary():
    sf1, sf2, sf3 = _encode()
    for sf in (sf1, sf2, sf3):
        assert len(sf) == 300
        assert set(sf) <= {0, 1}


def test_subframe_id_and_tow_helpers():
    sf1, sf2, sf3 = _encode()
    assert sidereon.lnav_subframe_id(sf1) == 1
    assert sidereon.lnav_subframe_id(sf2) == 2
    assert sidereon.lnav_subframe_id(sf3) == 3
    # The HOW carries a TOW count; a full subframe resolves it.
    assert isinstance(sidereon.lnav_tow(sf1), int)
    # Wrong-length inputs resolve to None (no panic).
    assert sidereon.lnav_tow([0, 1, 0]) is None
    assert sidereon.lnav_subframe_id([0, 1, 0]) is None


def test_full_subframe_helpers_restore_complemented_how():
    def set_bits(bits, start, width, value):
        for index in range(width):
            bits[start + index] = value >> (width - 1 - index) & 1

    source_how = [0] * 30
    set_bits(source_how, 0, 17, 54_321)
    set_bits(source_how, 19, 3, 1)
    subframe = [0] * 300
    subframe[29] = 1
    subframe[30:54] = [bit ^ 1 for bit in source_how[:24]]
    subframe[54:60] = source_how[24:30]

    assert sidereon.lnav_tow(subframe) == 54_321
    assert sidereon.lnav_subframe_id(subframe) == 1


def test_round_trip_recovers_integer_fields_exactly():
    sf1, sf2, sf3 = _encode()
    d = sidereon.lnav_decode(sf1, sf2, sf3)
    assert d.week_number == 290
    assert d.l2_code == 1
    assert d.ura_index == 0
    assert d.sv_health == 0
    assert d.iodc == 0x2AB
    assert d.iode == 0xAB
    assert d.toc == 504_000
    assert d.toe == 504_000
    assert d.fit_interval_flag == 0
    assert d.aodo == 0


def test_round_trip_recovers_scaled_fields_within_lsb():
    sf1, sf2, sf3 = _encode()
    d = sidereon.lnav_decode(sf1, sf2, sf3)
    cases = [
        (d.tgd, EXAMPLE["tgd"], TWO_POW_M31),
        (d.eccentricity, EXAMPLE["eccentricity"], TWO_POW_M33),
        (d.sqrt_a, EXAMPLE["sqrt_a"], TWO_POW_M19),
        (d.m0, EXAMPLE["m0"], TWO_POW_M31),
        (d.crs, EXAMPLE["crs"], TWO_POW_M5),
        (d.crc, EXAMPLE["crc"], TWO_POW_M5),
        (d.cuc, EXAMPLE["cuc"], TWO_POW_M29),
        (d.delta_n, EXAMPLE["delta_n"], TWO_POW_M43),
        (d.idot, EXAMPLE["idot"], TWO_POW_M43),
    ]
    for decoded, original, lsb in cases:
        quantized = round(original / lsb) * lsb
        assert abs(decoded - quantized) <= lsb / 2.0


def test_decode_preserves_raw_lnav_angular_fields_in_semicircles():
    raw = {
        "delta_n": 23_456,
        "m0": -123_456_789,
        "omega0": -345_678_901,
        "i0": 567_890_123,
        "omega": -678_901_234,
        "omega_dot": 3_456_789,
        "idot": -4_321,
    }
    encoded = {field: value & 0xFFFFFFFF for field, value in raw.items()}
    sf1_words = [(0x731FA9 + 0x13579B * index) & 0xFFFFFF for index in range(10)]
    sf2_words = [(0x2468AD + 0x13579B * index) & 0xFFFFFF for index in range(10)]
    sf3_words = [(0x1953BF + 0x13579B * index) & 0xFFFFFF for index in range(10)]
    sf2_words[3] = (raw["delta_n"] & 0xFFFF) << 8 | encoded["m0"] >> 24
    sf2_words[4] = encoded["m0"] & 0xFFFFFF
    sf3_words[2] = encoded["omega0"] >> 24
    sf3_words[3] = encoded["omega0"] & 0xFFFFFF
    sf3_words[4] = encoded["i0"] >> 24
    sf3_words[5] = encoded["i0"] & 0xFFFFFF
    sf3_words[6] = encoded["omega"] >> 24
    sf3_words[7] = encoded["omega"] & 0xFFFFFF
    sf3_words[8] = raw["omega_dot"] & 0xFFFFFF
    sf3_words[9] = (raw["idot"] & 0x3FFF) << 2

    decoded = sidereon.lnav_decode(
        _transmitted_subframe(sf1_words),
        _transmitted_subframe(sf2_words),
        _transmitted_subframe(sf3_words),
    )
    expected = {
        field: value
        * (TWO_POW_M43 if field in {"delta_n", "omega_dot", "idot"} else TWO_POW_M31)
        for field, value in raw.items()
    }
    for field, value in expected.items():
        assert getattr(decoded, field) == value
        assert getattr(decoded, field) != value * math.pi


LNAV_ANGULAR_PROPERTY_DOCS = {
    "delta_n": "Mean motion difference, semicircles/second.",
    "m0": "Mean anomaly at reference time, semicircles.",
    "omega0": "Longitude of ascending node at reference time, semicircles.",
    "i0": "Inclination at reference time, semicircles.",
    "omega": "Argument of perigee, semicircles.",
    "omega_dot": "Rate of right ascension, semicircles/second.",
    "idot": "Rate of inclination, semicircles/second.",
}


def test_lnav_angular_property_docs_state_raw_broadcast_units():
    for field, doc in LNAV_ANGULAR_PROPERTY_DOCS.items():
        assert getattr(sidereon.LnavDecoded, field).__doc__ == doc


def test_lnav_stub_docs_match_runtime_contract():
    stub = ast.parse(
        (Path(__file__).parents[1] / "python/sidereon/__init__.pyi").read_text()
    )
    lnav_decoded = next(
        node
        for node in stub.body
        if isinstance(node, ast.ClassDef) and node.name == "LnavDecoded"
    )
    stub_docs = {
        node.name: ast.get_docstring(node)
        for node in lnav_decoded.body
        if isinstance(node, ast.FunctionDef)
    }
    assert {
        field: stub_docs[field] for field in LNAV_ANGULAR_PROPERTY_DOCS
    } == LNAV_ANGULAR_PROPERTY_DOCS


def test_parity_helpers_are_self_consistent():
    # An arbitrary 24-bit data word, no D30* complement (prev parity 0, 0). Bit
    # vectors cross the boundary as bytes of 0x00 / 0x01.
    data24 = bytes(int(b) for b in "101100111000111000101011")
    parity = sidereon.lnav_parity(data24, 0, 0)
    assert len(parity) == 6
    word30 = data24 + parity
    assert sidereon.lnav_parity_valid(word30, 0, 0) is True

    # The first transmitted word (TLM) of subframe 1 is computed against a zero
    # previous-parity seed, so it validates against (0, 0).
    sf1, _sf2, _sf3 = _encode()
    assert sidereon.lnav_parity_valid(sf1[0:30], 0, 0) is True


def test_encode_rejects_out_of_range_field():
    bad = dict(EXAMPLE)
    bad["week_number"] = 99_999  # exceeds the 10-bit field
    with pytest.raises(ValueError):
        sidereon.lnav_encode(bad, tow=0)


def test_encode_rejects_missing_field():
    incomplete = dict(EXAMPLE)
    del incomplete["sqrt_a"]
    with pytest.raises(KeyError):
        sidereon.lnav_encode(incomplete, tow=0)


def test_decode_rejects_corrupted_parity():
    sf1, sf2, sf3 = _encode()
    corrupted = list(sf1)
    corrupted[29] ^= 1  # flip a parity bit of word 1
    with pytest.raises(ValueError):
        sidereon.lnav_decode(corrupted, sf2, sf3)


def test_param_fields_lists_thirty_names():
    fields = sidereon.lnav_param_fields()
    assert len(fields) == 30
    assert set(fields) == set(EXAMPLE.keys())
