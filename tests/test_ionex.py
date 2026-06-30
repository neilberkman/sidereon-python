"""IONEX parse + slant-delay round-trip against the core golden, bit-exact.

The golden is the same JSON the Rust core asserts on. Numbers are stored as C99
hex-float literals and decoded with ``float.fromhex`` (NOT the big-endian bit
pattern helper in ``_helpers``). The golden certifies 0 ULP on
aarch64-apple-darwin, so every comparison is exact bit equality, not a tolerance.
"""

import json
import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _load_golden():
    with open(os.path.join(CORE_FIXTURES, "ionex_golden.json")) as handle:
        return json.load(handle)


GOLDEN = _load_golden()


def _ionex_path():
    name = GOLDEN["ionex_file"]["name"]
    return os.path.join(CORE_FIXTURES, "ionex", name)


def _bits(value):
    """Little-endian IEEE-754 float64 byte pattern of a Python float."""
    return struct.pack("<d", value)


def _assert_bit_exact(got, want_hex, where):
    want = float.fromhex(want_hex)
    assert got == want, f"{where}: got {got!r}, want {want!r}"
    assert _bits(got) == _bits(want), f"{where}: not bit-identical"


@pytest.fixture(scope="module")
def grid_from_bytes():
    with open(_ionex_path(), "rb") as handle:
        return sidereon.load_ionex(handle.read())


@pytest.fixture(scope="module")
def grid_from_path():
    return sidereon.load_ionex(_ionex_path())


def test_load_ionex_accepts_bytes_and_path(grid_from_bytes, grid_from_path):
    # Both input forms reach the same parser; the parsed surface must agree.
    assert grid_from_bytes.exponent == grid_from_path.exponent
    np.testing.assert_array_equal(
        grid_from_bytes.lat_nodes_deg, grid_from_path.lat_nodes_deg
    )
    np.testing.assert_array_equal(grid_from_bytes.tec_maps, grid_from_path.tec_maps)


def test_parsed_grid_matches_golden(grid_from_bytes):
    expected = GOLDEN["ionex_file"]

    assert grid_from_bytes.exponent == int(round(expected["exponent"]))

    lat = grid_from_bytes.lat_nodes_deg
    assert lat.dtype == np.float64
    assert lat.shape == (len(expected["lat_arr"]),)
    for index, want_hex in enumerate(expected["lat_arr"]):
        _assert_bit_exact(float(lat[index]), want_hex, f"lat_arr[{index}]")

    lon = grid_from_bytes.lon_nodes_deg
    assert lon.dtype == np.float64
    assert lon.shape == (len(expected["lon_arr"]),)
    for index, want_hex in enumerate(expected["lon_arr"]):
        _assert_bit_exact(float(lon[index]), want_hex, f"lon_arr[{index}]")

    epochs = grid_from_bytes.map_epochs_j2000_s
    assert epochs.dtype == np.int64
    assert list(epochs) == list(expected["map_epochs_s"])

    tec = grid_from_bytes.tec_maps
    assert tec.dtype == np.float64
    maps_vtec = expected["maps_vtec"]
    assert tec.shape == (
        len(maps_vtec),
        len(maps_vtec[0]),
        len(maps_vtec[0][0]),
    )
    for ei, grid in enumerate(maps_vtec):
        for li, band in enumerate(grid):
            for oi, want_hex in enumerate(band):
                _assert_bit_exact(
                    float(tec[ei, li, oi]), want_hex, f"maps_vtec[{ei}][{li}][{oi}]"
                )


def test_slant_delay_cases_bit_exact(grid_from_bytes):
    cases = GOLDEN["cases"]
    assert len(cases) == 12
    for case in cases:
        inputs = case["inputs"]
        got = grid_from_bytes.slant_delay(
            float.fromhex(inputs["lat_deg"]),
            float.fromhex(inputs["lon_deg"]),
            float.fromhex(inputs["az_deg"]),
            float.fromhex(inputs["el_deg"]),
            int(inputs["epoch_s"]),
            float.fromhex(inputs["frequency_hz"]),
        )
        _assert_bit_exact(got, case["expect"]["delay_m"], f"case {case['name']}")


def test_repr_and_dtypes(grid_from_bytes):
    assert "Ionex(" in repr(grid_from_bytes)
    assert grid_from_bytes.map_epochs_j2000_s.dtype == np.int64
    assert grid_from_bytes.tec_maps.dtype == np.float64
    assert grid_from_bytes.tec_maps.ndim == 3
    assert isinstance(grid_from_bytes.shell_height_km, float)
    assert isinstance(grid_from_bytes.exponent, int)


def test_slant_delay_rejects_invalid_elevation(grid_from_bytes):
    a_case = GOLDEN["cases"][0]["inputs"]
    with pytest.raises(ValueError):
        grid_from_bytes.slant_delay(
            float.fromhex(a_case["lat_deg"]),
            float.fromhex(a_case["lon_deg"]),
            float.fromhex(a_case["az_deg"]),
            999.0,
            int(a_case["epoch_s"]),
            float.fromhex(a_case["frequency_hz"]),
        )


def test_slant_delay_rejects_non_finite(grid_from_bytes):
    a_case = GOLDEN["cases"][0]["inputs"]
    with pytest.raises(ValueError):
        grid_from_bytes.slant_delay(
            float("nan"),
            float.fromhex(a_case["lon_deg"]),
            float.fromhex(a_case["az_deg"]),
            float.fromhex(a_case["el_deg"]),
            int(a_case["epoch_s"]),
            float.fromhex(a_case["frequency_hz"]),
        )


def test_load_ionex_malformed_raises_ionex_parse_error():
    # Malformed IONEX raises the product-specific IonexParseError (a ParseError),
    # not the SP3 parse error.
    assert issubclass(sidereon.IonexParseError, sidereon.ParseError)
    with pytest.raises(sidereon.IonexParseError):
        sidereon.load_ionex(b"this is not an IONEX file")


def test_to_ionex_string_round_trips_product(grid_from_bytes):
    reparsed = sidereon.load_ionex(grid_from_bytes.to_ionex_string().encode("utf-8"))

    assert reparsed.exponent == grid_from_bytes.exponent
    np.testing.assert_array_equal(reparsed.lat_nodes_deg, grid_from_bytes.lat_nodes_deg)
    np.testing.assert_array_equal(reparsed.lon_nodes_deg, grid_from_bytes.lon_nodes_deg)
    np.testing.assert_array_equal(
        reparsed.map_epochs_j2000_s, grid_from_bytes.map_epochs_j2000_s
    )
    np.testing.assert_array_equal(reparsed.tec_maps, grid_from_bytes.tec_maps)
    np.testing.assert_array_equal(reparsed.rms_maps, grid_from_bytes.rms_maps)
