import math

import numpy as np
import pytest
import sidereon
from test_constellation import _records


def test_civil_time_helpers_delegate_to_core():
    split = sidereon.split_julian_date(2000, 1, 1, 12, 0, 0.0)
    assert split.whole == 2451544.5
    assert split.fraction == 0.5
    assert split.jd == 2451545.0
    assert sidereon.j2000_seconds(2000, 1, 1, 12, 0, 0.0) == 0.0
    assert sidereon.second_of_day(1, 2, 3.5) == 3723.5
    assert sidereon.day_of_year(2020, 6, 24, 12, 0, 0.0) == 176.5


def test_lnav_constants_delegate_to_core():
    assert sidereon.lnav_word_length() == 30
    assert sidereon.lnav_subframe_length() == 300
    assert sidereon.lnav_preamble() == 0b10001011


def test_observation_frequency_and_carrier_helpers_delegate_to_core():
    assert sidereon.rinex_observation_frequency_hz(
        sidereon.GnssSystem.BEIDOU, "C1I", 3.02
    ) == sidereon.carrier_frequency_hz(
        sidereon.GnssSystem.BEIDOU, sidereon.CarrierBand.B1I
    )
    assert sidereon.rinex_observation_frequency_hz(
        sidereon.GnssSystem.BEIDOU, "C1I", 3.03
    ) == sidereon.carrier_frequency_hz(
        sidereon.GnssSystem.BEIDOU, sidereon.CarrierBand.B1C
    )
    assert sidereon.rinex_observation_wavelength_m(
        sidereon.GnssSystem.GPS, "C1C", 3.04
    ) == sidereon.wavelength_m(sidereon.GnssSystem.GPS, sidereon.CarrierBand.L1)

    f_l1 = sidereon.carrier_frequency_hz(
        sidereon.GnssSystem.GPS, sidereon.CarrierBand.L1
    )
    phase_m = sidereon.phase_meters(10.0, f_l1)
    assert sidereon.code_minus_carrier(100.0, 10.0, f_l1) == pytest.approx(
        100.0 - phase_m
    )


def test_signal_correlation_helpers_delegate_to_core():
    code = sidereon.ca_code(1).tolist()
    auto = sidereon.autocorrelation(code)
    assert auto.dtype == np.int32
    assert len(auto) == 1023
    assert int(auto[0]) == 1023
    assert sidereon.correlation_at(code, code, 0) == 1023

    other = sidereon.ca_code(2).tolist()
    cross = sidereon.cross_correlation(code, other)
    assert len(cross) == 1023
    assert int(cross[0]) == sidereon.correlation_at(code, other, 0)

    short_code = code[:32]
    iq = np.column_stack(
        [np.asarray(short_code, dtype=np.float64), np.zeros(len(short_code))]
    )
    corr = sidereon.correlate_against(iq, short_code, 1_023_000.0, 0.0)
    assert corr.i == pytest.approx(32.0)
    assert corr.q == pytest.approx(0.0)
    assert corr.power == pytest.approx(1024.0)


def test_quality_and_spp_scalar_helpers_delegate_to_core():
    assert sidereon.chi2_inv(0.999, 1) == pytest.approx(10.827566170662733)
    with pytest.raises(ValueError):
        sidereon.chi2_inv(0.95, 0)

    assert sidereon.spp_residual_rms_m([]) == 0.0
    assert sidereon.spp_residual_rms_m([3.0, 4.0]) == pytest.approx(math.sqrt(12.5))


def test_constellation_lookup_and_strict_validation_delegate_to_core():
    assert sidereon.galileo_prn_for_gsat(210) == 1
    assert sidereon.galileo_prn_for_gsat(999) is None
    assert sidereon.glonass_slot_for_number(730) == 1
    assert sidereon.glonass_slot_for_number(999) is None

    records = _records()
    ids = [record.sp3_id for record in records]
    assert sidereon.validate_against_sp3_ids_strict(records, ids) is None
    with pytest.raises(sidereon.ConstellationError):
        sidereon.validate_against_sp3_ids_strict(records, ids[:-1])
