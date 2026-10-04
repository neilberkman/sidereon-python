"""Standalone PPP correction precompute through the binding.

`ppp_corrections` is a pure wrapper over `sidereon_core::ppp_corrections`. The
bar is bit-exact against the core: `scripts/core_goldens` runs the core's
`build` on the same SP3 arc, satellite, epoch, receiver, and antenna options,
and the binding must reproduce its solid-earth tide, carrier-phase wind-up, and
satellite-antenna PCO/PCV corrections to the bit.
"""

import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, core_goldens

SP3_FILE = "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
SAT = "G21"
# 2020-06-24 12:00:00. JDN(noon)=2459025; t_rx = (2459025.0 - 2451545.0)*86400.
T_RX_J2000_S = (2459025.0 - 2451545.0) * 86400.0  # = 646272000.0, exact
RECEIVER_M = [3512900.0, 780500.0, 5248700.0]

F_L1_HZ = 1575.42e6
F_L2_HZ = 1227.60e6

# IEEE-754 bits of the core's own corrections for this setup.
_GOLDEN = core_goldens()["ppp_corrections"]
TIDE_BITS = tuple(int(bits, 16) for bits in _GOLDEN["tide"])
WINDUP_BITS = int(_GOLDEN["windup_m"], 16)
SAT_PCO_BITS = tuple(int(bits, 16) for bits in _GOLDEN["sat_pco_ecef"])
SAT_PCV_BITS = int(_GOLDEN["sat_pcv_m"], 16)


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


def test_ppp_corrections_tide_constants_and_ut1_validity_metadata():
    precise = _sp3()
    current_epoch = _epoch()
    default_constants = sidereon.ppp_corrections(
        precise, [current_epoch], RECEIVER_M, solid_earth_tide=True
    )
    explicit_conventions = sidereon.ppp_corrections_with_validity_and_tide_constants(
        precise,
        [current_epoch],
        RECEIVER_M,
        solid_earth_tide=True,
        tide_constants=sidereon.StationTideConstants.CONVENTIONS,
    )
    assert explicit_conventions.tide == default_constants.tide
    assert not explicit_conventions.ut1_degraded
    assert explicit_conventions.degradation_reason is None

    routine_constants = sidereon.ppp_corrections_with_validity_and_tide_constants(
        precise,
        [current_epoch],
        RECEIVER_M,
        solid_earth_tide=True,
        tide_constants=sidereon.StationTideConstants.IERS_ROUTINE,
    )
    assert len(routine_constants.tide) == 1
    assert all(np.isfinite(component) for component in routine_constants.tide[0][1])

    outside_epoch = sidereon.PppCorrectionEpoch(
        2500,
        1,
        1,
        0,
        0,
        0.0,
        T_RX_J2000_S,
        [],
    )
    with pytest.raises(sidereon.PppCorrectionsError) as refusal:
        sidereon.ppp_corrections_with_validity_and_tide_constants(
            precise, [outside_epoch], RECEIVER_M, solid_earth_tide=True
        )
    assert refusal.value.kind == "epoch"
    assert refusal.value.epoch_index == 0
    assert refusal.value.details["reason"] == "after_coverage"

    permissive = sidereon.ppp_corrections_with_validity_and_tide_constants(
        precise,
        [outside_epoch],
        RECEIVER_M,
        solid_earth_tide=True,
        validity=sidereon.ValidityMode.PERMISSIVE,
    )
    assert permissive.ut1_degraded
    assert permissive.degradation_reason == "after_coverage"


def test_ppp_bias_refusal_retains_nested_core_error_fields():
    bias_path = os.path.join(
        CORE_FIXTURES,
        "bias",
        "COD0OPSFIN_20261330000_01D_01D_OSB.BIA",
    )
    bias_set = sidereon.load_bias_sinex(bias_path)
    code_bias = sidereon.PppCodeBiasOptions(
        bias_set,
        [(sidereon.GnssSystem.GPS, "C1C", "C2W")],
        clock_reference=[],
    )
    with pytest.raises(sidereon.PppCorrectionsError) as refusal:
        sidereon.ppp_corrections_with_validity_and_tide_constants(
            _sp3(), [_epoch()], RECEIVER_M, code_bias=code_bias
        )
    assert refusal.value.kind == "bias"
    assert refusal.value.details["source_kind"] == "missing_clock_reference"
