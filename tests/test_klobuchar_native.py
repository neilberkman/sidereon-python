"""GPS broadcast Klobuchar ionospheric delay, native (bit-exact) entry.

`klobuchar_native` is a thin wrapper over
`sidereon_core::atmosphere::ionosphere::klobuchar_native`, the 0-ULP entry in the
model's published native units. The test replays the core's own hex-float golden
cases (`klobuchar_golden.json`) and asserts the returned L1 delay is bit-exact to
the recorded `delay_l1_m`.
"""

import json
import os
import struct

import sidereon
from _helpers import CORE_FIXTURES

GOLDEN = os.path.join(CORE_FIXTURES, "klobuchar_golden.json")


def _bits(x):
    return struct.unpack(">Q", struct.pack(">d", x))[0]


def _load():
    with open(GOLDEN) as fh:
        return json.load(fh)


def test_klobuchar_native_matches_golden_l1_delay_bit_exact():
    golden = _load()
    f_l1 = float.fromhex(golden["constants"]["f_l1"])
    cases = golden["cases"]
    assert len(cases) > 0
    for case in cases:
        inp = case["inputs"]
        alpha = [float.fromhex(v) for v in inp["alpha"]]
        beta = [float.fromhex(v) for v in inp["beta"]]
        delay = sidereon.klobuchar_native(
            alpha,
            beta,
            float.fromhex(inp["lat_deg"]),
            float.fromhex(inp["lon_deg"]),
            float.fromhex(inp["az_deg"]),
            float.fromhex(inp["el_deg"]),
            float.fromhex(inp["t_gps_s"]),
            f_l1,
        )
        expected = float.fromhex(case["expect"]["delay_l1_m"])
        assert _bits(delay) == _bits(expected), case["name"]


def test_klobuchar_native_is_dispersive_in_frequency():
    # At a lower frequency the group delay is larger by exactly (f_l1 / f)^2.
    golden = _load()
    f_l1 = float.fromhex(golden["constants"]["f_l1"])
    inp = golden["cases"][0]["inputs"]
    alpha = [float.fromhex(v) for v in inp["alpha"]]
    beta = [float.fromhex(v) for v in inp["beta"]]
    args = (
        float.fromhex(inp["lat_deg"]),
        float.fromhex(inp["lon_deg"]),
        float.fromhex(inp["az_deg"]),
        float.fromhex(inp["el_deg"]),
        float.fromhex(inp["t_gps_s"]),
    )
    d_l1 = sidereon.klobuchar_native(alpha, beta, *args, f_l1)
    f_l2 = 1227.60e6
    d_l2 = sidereon.klobuchar_native(alpha, beta, *args, f_l2)
    assert d_l2 > d_l1 > 0.0
    ratio = (f_l1 / f_l2) ** 2
    assert abs(d_l2 / d_l1 - ratio) < 1.0e-9
