"""CRINEX decoding through the Python binding uses real committed fixtures."""

import os

import pytest
import sidereon
from _helpers import FIXTURES

OBS_FIXTURES = os.path.join(FIXTURES, "obs")
ESBC_CRX = "ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx"
ESBC_RNX = "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
ALGO_V1_CRX = "algo0010_2015001_v1_trim.crx"
ALGO_V1_RNX = "algo0010_2015001_v1_trim.rnx"


def _read_obs(name):
    with open(os.path.join(OBS_FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


def test_decode_crinex_matches_plain_rinex_reference_and_parses_epochs():
    decoded = sidereon.decode_crinex(_read_obs(ESBC_CRX))
    reference = _read_obs(ESBC_RNX)

    assert decoded.splitlines() == reference.splitlines()

    decoded_obs = sidereon.parse_rinex_obs(decoded)
    reference_obs = sidereon.parse_rinex_obs(reference)
    assert decoded_obs.epoch_count == reference_obs.epoch_count == 2
    assert [epoch.satellites for epoch in decoded_obs.epochs] == [
        epoch.satellites for epoch in reference_obs.epochs
    ]
    assert decoded_obs.epochs[0].epoch.second == reference_obs.epochs[0].epoch.second
    assert decoded_obs.epochs[1].epoch.second == reference_obs.epochs[1].epoch.second


def test_decode_crinex_lines_matches_rinex_v1_reference():
    lines = sidereon.decode_crinex_lines(_read_obs(ALGO_V1_CRX))

    assert lines == _read_obs(ALGO_V1_RNX).splitlines()


def test_encode_crinex_round_trips_through_decode():
    reference = _read_obs(ESBC_RNX)

    encoded = sidereon.encode_crinex(reference)
    # CRINEX is a compact form: encoding shrinks the plain text.
    assert len(encoded) < len(reference)

    redecoded = sidereon.decode_crinex(encoded)
    assert redecoded.splitlines() == reference.splitlines()


def test_encode_crinex_round_trips_rinex_v1():
    reference = _read_obs(ALGO_V1_RNX)

    redecoded = sidereon.decode_crinex(sidereon.encode_crinex(reference))
    assert redecoded.splitlines() == reference.splitlines()


def test_encode_crinex_rejects_malformed_input():
    with pytest.raises(sidereon.CrinexParseError):
        sidereon.encode_crinex("not a rinex file\n")


def test_load_crinex_accepts_path_and_bytes_and_errors_are_typed():
    path = os.path.join(OBS_FIXTURES, ESBC_CRX)
    expected_lines = _read_obs(ESBC_RNX).splitlines()

    assert sidereon.load_crinex(path).splitlines() == expected_lines
    assert (
        sidereon.load_crinex(_read_obs(ESBC_CRX).encode("utf-8")).splitlines()
        == expected_lines
    )

    with pytest.raises(sidereon.CrinexParseError):
        sidereon.decode_crinex("not a crinex file\n")

    with pytest.raises(sidereon.CrinexParseError):
        sidereon.load_crinex(b"\xff")
