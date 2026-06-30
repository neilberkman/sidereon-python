"""GPS LNAV (legacy navigation message) codec.

`lnav_encode` / `lnav_decode` and the parity / TOW / subframe-id helpers are thin
wrappers over `sidereon_core::navigation::lnav`. The representative MEO GPS SV
parameter set mirrors the core `example/0` fixture, so the round trip asserts the
same recovery contract the core test asserts: integer fields exactly, scaled
fields to within their IS-GPS-200 LSB.
"""

import math

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
        (d.delta_n, EXAMPLE["delta_n"], math.pi * TWO_POW_M43),
        (d.idot, EXAMPLE["idot"], math.pi * TWO_POW_M43),
    ]
    for decoded, original, lsb in cases:
        quantized = round(original / lsb) * lsb
        assert abs(decoded - quantized) <= lsb / 2.0


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
