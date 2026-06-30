"""Area 4 (bodies + SP3) binding reproduces the engine numbers bit-for-bit.

The fixture ``sp3_bodies.json`` is emitted by the crate's env-gated harness
(``SIDEREON_DUMP_FIXTURES=1 cargo test -p sidereon-core --test
sp3_bodies_python_fixture``); it carries the analytic Sun/Moon ECI and ECEF
vectors per real UTC epoch, plus the SP3 node axis, interpolated states, the
exact first record, and the serialized SP3 text, all from the same functions the
binding calls. The binding loads the SAME committed SP3 fixture and must return
identical bits -- a wrapper that diverges is a wrapper bug, not a new answer.
"""

import json
import os
import pathlib

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES, hex_to_f64


def _fixture():
    with open(os.path.join(FIXTURES, "sp3_bodies.json")) as fh:
        return json.load(fh)


FX = _fixture()

# The SP3 product the emitter used, read verbatim from the crate-side fixtures.
SP3_PATH = os.path.join(CORE_FIXTURES, "sp3", os.path.basename(FX["sp3_fixture"]))


def _bits(arr):
    """uint64 bit view of a float64 numpy array, for exact comparison."""
    return np.asarray(arr, dtype=np.float64).view(np.uint64)


def _expect_bits(hex_list):
    return np.asarray([int(h, 16) for h in hex_list], dtype=np.uint64)


def _load_sp3():
    with open(SP3_PATH, "rb") as fh:
        return sidereon.load_sp3(fh.read())


# --- Bodies: Sun/Moon ECI + ECEF ------------------------------------------


def _body_epochs():
    return np.asarray([b["unix_micros"] for b in FX["bodies"]], dtype=np.int64)


def test_sun_moon_eci_matches_reference_bits():
    epochs = _body_epochs()
    result = sidereon.sun_moon_eci(epochs)
    assert result.sun.shape == (len(FX["bodies"]), 3)
    assert result.moon.shape == (len(FX["bodies"]), 3)
    assert result.sun.dtype == np.float64
    assert result.epoch_count == len(FX["bodies"])
    for i, b in enumerate(FX["bodies"]):
        assert np.array_equal(_bits(result.sun[i]), _expect_bits(b["sun_eci_m_hex"]))
        assert np.array_equal(_bits(result.moon[i]), _expect_bits(b["moon_eci_m_hex"]))


def test_sun_moon_ecef_matches_reference_bits():
    epochs = _body_epochs()
    result = sidereon.sun_moon_ecef(epochs)
    assert result.sun.shape == (len(FX["bodies"]), 3)
    for i, b in enumerate(FX["bodies"]):
        assert np.array_equal(_bits(result.sun[i]), _expect_bits(b["sun_ecef_m_hex"]))
        assert np.array_equal(_bits(result.moon[i]), _expect_bits(b["moon_ecef_m_hex"]))


def test_sun_moon_single_epoch_matches_batch():
    """One epoch at a time reproduces the same row as the full-batch call."""
    epochs = _body_epochs()
    batch = sidereon.sun_moon_eci(epochs)
    for i, b in enumerate(FX["bodies"]):
        one = sidereon.sun_moon_eci(np.asarray([b["unix_micros"]], dtype=np.int64))
        assert np.array_equal(_bits(one.sun[0]), _bits(batch.sun[i]))


def test_sun_moon_empty_epochs_raises():
    with pytest.raises(ValueError):
        sidereon.sun_moon_eci(np.asarray([], dtype=np.int64))


def test_sun_moon_repr_names_frame():
    r = sidereon.sun_moon_ecef(_body_epochs())
    assert "ecef" in repr(r)
    assert "epoch_count" in repr(r)


# --- SP3 load (bytes + path) ----------------------------------------------


def test_load_sp3_from_bytes():
    sp3 = _load_sp3()
    assert sp3.epoch_count == FX["epoch_count"]
    assert sp3.satellites == FX["satellites"]


def test_load_sp3_from_path_str():
    sp3 = sidereon.load_sp3(SP3_PATH)
    assert sp3.epoch_count == FX["epoch_count"]
    assert sp3.satellites == FX["satellites"]


def test_load_sp3_from_pathlike():
    sp3 = sidereon.load_sp3(pathlib.Path(SP3_PATH))
    assert sp3.epoch_count == FX["epoch_count"]
    assert sp3.satellites == FX["satellites"]


def test_load_sp3_bad_bytes_raises_parse_error():
    with pytest.raises(sidereon.Sp3ParseError):
        sidereon.load_sp3(b"not an sp3 file")
    # And it is catchable via the base classes.
    with pytest.raises(sidereon.ParseError):
        sidereon.load_sp3(b"not an sp3 file")
    with pytest.raises(sidereon.SidereonError):
        sidereon.load_sp3(b"not an sp3 file")


def test_load_sp3_missing_path_raises_oserror():
    with pytest.raises(OSError):
        sidereon.load_sp3(os.path.join(CORE_FIXTURES, "sp3", "does_not_exist.sp3"))


def test_load_sp3_bad_type_raises():
    with pytest.raises((TypeError, ValueError)):
        sidereon.load_sp3(12345)


# --- SP3 node axis ---------------------------------------------------------


def test_epochs_j2000_seconds_matches_reference_bits():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    assert axis.shape == (FX["epoch_count"],)
    assert axis.dtype == np.float64
    assert np.array_equal(_bits(axis), _expect_bits(FX["epochs_j2000_seconds_hex"]))


# --- SP3 interpolation -----------------------------------------------------


def _query_array():
    return np.asarray(
        [hex_to_f64(h) for h in FX["query_j2000_seconds_hex"]], dtype=np.float64
    )


def test_interpolate_matches_reference_bits():
    sp3 = _load_sp3()
    queries = _query_array()
    for entry in FX["interpolation"]:
        result = sp3.interpolate(entry["satellite"], queries)
        assert result.position_m.shape == (len(queries), 3)
        assert result.clock_s.shape == (len(queries),)
        assert result.position_m.dtype == np.float64
        assert result.epoch_count == len(queries)
        for i, st in enumerate(entry["states"]):
            assert np.array_equal(
                _bits(result.position_m[i]), _expect_bits(st["position_m_hex"])
            )
            if st["clock_s_hex"] is None:
                assert np.isnan(result.clock_s[i])
            else:
                assert result.clock_s[i].view(np.uint64) == int(st["clock_s_hex"], 16)


def test_interpolate_unknown_token_raises_value_error():
    sp3 = _load_sp3()
    with pytest.raises(ValueError):
        sp3.interpolate("ZZ9", _query_array())


def test_interpolate_absent_satellite_raises_value_error():
    """A satellite parseable but not in this GPS-only product is bad input."""
    sp3 = _load_sp3()
    with pytest.raises(ValueError):
        sp3.interpolate("E01", _query_array())


def test_interpolate_out_of_coverage_raises_solve_error():
    sp3 = _load_sp3()
    axis = sp3.epochs_j2000_seconds
    far = np.asarray([float(axis[0]) - 1.0e6], dtype=np.float64)
    with pytest.raises(sidereon.SolveError):
        sp3.interpolate("G01", far)


def test_interpolate_empty_queries_raises():
    sp3 = _load_sp3()
    with pytest.raises(ValueError):
        sp3.interpolate("G01", np.asarray([], dtype=np.float64))


# --- SP3 exact per-record state -------------------------------------------


def test_state_first_record_matches_reference():
    sp3 = _load_sp3()
    st = sp3.state("G01", 0)
    ref = FX["state_g01_epoch0"]
    assert np.array_equal(_bits(st.position_m), _expect_bits(ref["position_m_hex"]))
    if ref["clock_s_hex"] is None:
        assert st.clock_s is None
    else:
        assert np.float64(st.clock_s).view(np.uint64) == int(ref["clock_s_hex"], 16)
    if ref["velocity_m_s_hex"] is None:
        assert st.velocity_m_s is None
    else:
        assert np.array_equal(
            _bits(st.velocity_m_s), _expect_bits(ref["velocity_m_s_hex"])
        )
    assert st.clock_event == ref["clock_event"]
    assert st.clock_predicted == ref["clock_predicted"]
    assert st.maneuver == ref["maneuver"]
    assert st.orbit_predicted == ref["orbit_predicted"]


def test_state_out_of_range_raises_index_error():
    sp3 = _load_sp3()
    with pytest.raises(IndexError):
        sp3.state("G01", FX["epoch_count"] + 100)


# --- SP3 write -------------------------------------------------------------


def test_to_sp3_string_matches_reference_exactly():
    sp3 = _load_sp3()
    assert sp3.to_sp3_string() == FX["to_sp3_string"]


def test_to_sp3_string_round_trips():
    sp3 = _load_sp3()
    text = sp3.to_sp3_string()
    reparsed = sidereon.load_sp3(text.encode("ascii"))
    assert reparsed.epoch_count == sp3.epoch_count
    assert reparsed.satellites == sp3.satellites
