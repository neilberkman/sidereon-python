"""Standalone PPP correction precompute through the binding.

`ppp_corrections` is a pure wrapper over `sidereon_core::ppp_corrections`. The
bar is bit-exact against the core's own reference fixture test
(`ppp_corrections_match_elixir_reference_fixture`): the same SP3 arc, satellite,
epoch, receiver, and antenna options must reproduce the same solid-earth tide,
carrier-phase wind-up, and satellite-antenna PCO/PCV corrections to the bit.
"""

import os
import struct

import pytest
import sidereon
from _helpers import CORE_FIXTURES

SP3_FILE = "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
SAT = "G21"
# 2020-06-24 12:00:00. JDN(noon)=2459025; t_rx = (2459025.0 - 2451545.0)*86400.
T_RX_J2000_S = (2459025.0 - 2451545.0) * 86400.0  # = 646272000.0, exact
RECEIVER_M = [3512900.0, 780500.0, 5248700.0]

F_L1_HZ = 1575.42e6
F_L2_HZ = 1227.60e6

# Frozen IEEE-754 bits from the core reference test.
TIDE_BITS = (0x3FB8BC98E788ED00, 0x3FAA54D8C1097508, 0x3FB03498C46B3B50)
WINDUP_BITS = 0xBF808DE79DBD2C16
SAT_PCO_BITS = (0xBFE58ED947570048, 0x3FDEDBB280CEB1BE, 0xBFFE3BCA6A354E4A)
SAT_PCV_BITS = 0x3F77617E95BD232C


def _bits(u64):
    return struct.unpack("<d", struct.pack("<Q", u64))[0]


def _sp3():
    with open(os.path.join(CORE_FIXTURES, "sp3", SP3_FILE), "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _antenna_options():
    return sidereon.SatelliteAntennaOptions(
        "G01",
        F_L1_HZ,
        "G02",
        F_L2_HZ,
        [
            sidereon.SatelliteAntenna(
                SAT,
                [
                    sidereon.SatelliteAntennaFrequency(
                        "G01",
                        [0.1, -0.2, 1.0],
                        [(0.0, 0.001), (5.0, 0.002), (10.0, 0.004)],
                    ),
                    sidereon.SatelliteAntennaFrequency(
                        "G02",
                        [-0.1, 0.3, 0.5],
                        [(0.0, -0.001), (5.0, -0.002), (10.0, -0.003)],
                    ),
                ],
                valid_from=(2020, 1, 1, 0, 0, 0.0),
                valid_until=(2021, 1, 1, 0, 0, 0.0),
            )
        ],
    )


def _epoch():
    return sidereon.PppCorrectionEpoch(
        2020,
        6,
        24,
        12,
        0,
        0.0,
        T_RX_J2000_S,
        [sidereon.PppCorrectionObservation(SAT, F_L1_HZ, F_L2_HZ)],
    )


def test_ppp_corrections_are_bit_exact():
    corr = sidereon.ppp_corrections(
        _sp3(),
        [_epoch()],
        RECEIVER_M,
        solid_earth_tide=True,
        phase_windup=True,
        satellite_antenna=_antenna_options(),
    )

    assert len(corr.tide) == 1
    epoch_index, tide_vec = corr.tide[0]
    assert epoch_index == 0
    assert tide_vec == tuple(_bits(b) for b in TIDE_BITS)

    assert len(corr.windup_m) == 1
    sat, idx, windup = corr.windup_m[0]
    assert sat == SAT and idx == 0
    assert windup == _bits(WINDUP_BITS)

    assert len(corr.sat_pco_ecef) == 1
    sat, idx, pco = corr.sat_pco_ecef[0]
    assert sat == SAT and idx == 0
    assert pco == tuple(_bits(b) for b in SAT_PCO_BITS)

    assert len(corr.sat_pcv_m) == 1
    sat, idx, pcv = corr.sat_pcv_m[0]
    assert sat == SAT and idx == 0
    assert pcv == _bits(SAT_PCV_BITS)


def test_ppp_corrections_disabled_returns_empty():
    corr = sidereon.ppp_corrections(_sp3(), [_epoch()], RECEIVER_M)
    assert corr.tide == []
    assert corr.windup_m == []
    assert corr.sat_pco_ecef == []
    assert corr.sat_pcv_m == []


def test_ppp_corrections_rejects_degenerate_receiver():
    with pytest.raises(ValueError):
        sidereon.ppp_corrections(
            _sp3(), [_epoch()], [0.0, 0.0, 0.0], solid_earth_tide=True
        )
