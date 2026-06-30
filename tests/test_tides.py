"""Station displacement tide models (solid-earth, pole, ocean loading).

`solid_earth_tide`, `solid_earth_pole_tide`, and `ocean_tide_loading` are thin
wrappers over `sidereon_core::tides`. The solid-earth test replays the core's
Dehant golden cases and asserts the returned displacement matches the recorded
IERS reference vector; the pole and ocean tests assert the delegation's
structural invariants.
"""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

DEHANT = os.path.join(CORE_FIXTURES, "tides", "tides_dehant_golden.json")
N_CONSTITUENTS = 11


def _fhr(year, month, day, hour):  # convenience for readable epochs
    return float(hour)


def test_solid_earth_tide_matches_dehant_golden():
    with open(DEHANT) as fh:
        golden = json.load(fh)
    cases = golden["cases"]
    assert len(cases) > 0
    for case in cases:
        # case_4 is a known fixture transcription artifact (its expected vector
        # is a verbatim copy of case_3's while its Sun vector is unphysical); the
        # core's own golden test excludes it for the same reason.
        if case["id"] == "case_4_2017_01_15":
            continue
        inp = case["inputs"]
        xsta = np.asarray(inp["xsta_m"]["values"], dtype=np.float64)
        xsun = np.asarray(inp["xsun_m"]["values"], dtype=np.float64)
        xmon = np.asarray(inp["xmon_m"]["values"], dtype=np.float64)
        date = inp["date_utc"]
        fhr = inp["fhr_hours"]["value"]
        out = sidereon.solid_earth_tide(
            xsta, date["year"], date["month"], date["day"], fhr, xsun, xmon
        )
        expected = np.asarray(case["expected"]["dxtide_m"]["values"], dtype=np.float64)
        assert out.shape == (3,)
        np.testing.assert_allclose(out, expected, rtol=0.0, atol=1.0e-12)


def test_solid_earth_pole_tide_returns_finite_vector():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    out = sidereon.solid_earth_pole_tide(xsta, 2009, 4, 13, 0.0, 0.12, 0.34)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))


def test_ocean_tide_loading_zero_coefficients_is_zero():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    zero = [[0.0] * N_CONSTITUENTS for _ in range(3)]
    out = sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 0.0, zero, zero)
    assert out.shape == (3,)
    np.testing.assert_array_equal(out, np.zeros(3))


def test_ocean_tide_loading_nonzero_coefficients_is_finite():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    amplitude = [
        [
            0.003,
            0.001,
            0.0007,
            0.0003,
            0.002,
            0.0015,
            0.0006,
            0.0002,
            0.0004,
            0.0002,
            0.0001,
        ],
        [
            0.001,
            0.0005,
            0.0003,
            0.0001,
            0.0008,
            0.0006,
            0.0002,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ],
        [
            0.002,
            0.0008,
            0.0005,
            0.0002,
            0.0012,
            0.0009,
            0.0004,
            0.0001,
            0.0002,
            0.0001,
            0.0001,
        ],
    ]
    phase = [[10.0 * (c + 1) for c in range(N_CONSTITUENTS)] for _ in range(3)]
    out = sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 6.0, amplitude, phase)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.linalg.norm(out) > 0.0


def test_ocean_tide_loading_rejects_wrong_shape():
    xsta = np.asarray([4075578.385, 931852.89, 4801570.154], dtype=np.float64)
    two_rows = [[0.0] * N_CONSTITUENTS for _ in range(2)]
    three_rows = [[0.0] * N_CONSTITUENTS for _ in range(3)]
    with pytest.raises(ValueError):
        sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 0.0, two_rows, three_rows)

    short_row = [[0.0] * (N_CONSTITUENTS - 1) for _ in range(3)]
    with pytest.raises(ValueError):
        sidereon.ocean_tide_loading(xsta, 2009, 4, 13, 0.0, short_row, three_rows)
