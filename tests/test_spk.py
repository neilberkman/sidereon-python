"""SPK (JPL/NAIF DAF `.bsp`) ephemeris-kernel binding tests.

Proves the Python `Spk` surface end-to-end against the same Type 21 kernel the
core asserts on (`real_type21_kernel_matches_cspice_reference` in
`sidereon-core/src/astro/spk.rs`), so a query through Python reproduces the
CSPICE reference state. Also covers segment metadata, path vs bytes loading,
and the malformed-input error path.
"""

import os
import pathlib

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES

SPK_FIXTURES = os.path.join(FIXTURES, "spk")
EROS_TYPE21 = os.path.join(SPK_FIXTURES, "horizons_eros_type21.bsp")

# NAIF ids for the Type 21 Eros kernel.
EROS = 20000433
SUN = 10

# (et seconds past J2000 TDB, [x, y, z, vx, vy, vz]) km, km/s, taken verbatim
# from the core's CSPICE reference set. A subset is enough to prove agreement.
REFERENCE = [
    (
        757339200.0,
        [
            198083634.33689928,
            56306354.00566181,
            67761020.0290685,
            -14.136880898003753,
            18.729945253375007,
            8.080580941541488,
        ],
    ),
    (
        765244800.0,
        [
            14324682.473833444,
            151855494.96957216,
            88809564.6055465,
            -29.543141519840074,
            1.384349579197926,
            -4.552338928064369,
        ],
    ),
    (
        788961600.0,
        [
            -2423286.488811064,
            -220785626.12491044,
            -125794359.14041424,
            20.360009383792537,
            -4.508637229520069,
            1.1193915696949732,
        ],
    ),
]


def test_load_spk_from_path_segment_metadata():
    spk = sidereon.load_spk(pathlib.Path(EROS_TYPE21))

    segments = spk.segments
    assert len(segments) == 1
    seg = segments[0]
    assert seg.target == EROS
    assert seg.center == SUN
    assert seg.data_type == 21
    assert seg.start_et < seg.stop_et
    assert spk.internal_name  # DAF header internal name is populated.
    assert "Spk(" in repr(spk)
    assert "SpkSegment(" in repr(seg)


def test_type21_state_matches_cspice_reference():
    spk = sidereon.load_spk(EROS_TYPE21)

    max_position_error = 0.0
    max_velocity_error = 0.0
    for et, expected in REFERENCE:
        state = spk.state(EROS, SUN, et)
        assert state.target == EROS
        assert state.center == SUN
        pos = np.asarray(state.position_km, dtype=np.float64)
        vel = state.velocity_km_s
        assert vel is not None, "Type 21 segment must yield velocity"
        vel = np.asarray(vel, dtype=np.float64)
        assert pos.shape == (3,)
        assert vel.shape == (3,)
        max_position_error = max(
            max_position_error, float(np.max(np.abs(pos - expected[:3])))
        )
        max_velocity_error = max(
            max_velocity_error, float(np.max(np.abs(vel - expected[3:])))
        )

    # Same ~1-ULP bar the core asserts at these magnitudes (|pos|~2.2e8 km,
    # |vel|~20 km/s): position < 5e-8 km, velocity < 5e-15 km/s.
    assert max_position_error < 5e-8, max_position_error
    assert max_velocity_error < 5e-15, max_velocity_error


def test_load_spk_from_bytes_matches_path():
    with open(EROS_TYPE21, "rb") as handle:
        data = handle.read()

    spk = sidereon.load_spk(data)
    et, expected = REFERENCE[0]
    state = spk.state(EROS, SUN, et)
    np.testing.assert_allclose(
        np.asarray(state.position_km), expected[:3], rtol=0, atol=5e-8
    )


def test_unknown_body_raises_value_error():
    spk = sidereon.load_spk(EROS_TYPE21)
    with pytest.raises(ValueError):
        spk.state(999999, SUN, REFERENCE[0][0])


def test_bad_bytes_raise_parse_error_not_panic():
    with pytest.raises(sidereon.SpkParseError):
        sidereon.load_spk(b"this is not a DAF/SPK kernel")
