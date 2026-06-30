"""Observable binding tests use canonical GNSS constants and Rust oracle bits."""

import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

SP3_PATH = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")

VELOCITY_OBS_BITS = [
    ("G07", 0xC0768A0B93C45F82),
    ("G08", 0xC081BBF2879835FD),
    ("G10", 0xC081C9B51570E844),
    ("G16", 0xC045EB58A1B7B54E),
    ("G18", 0x407EC07DD774B2F8),
    ("G20", 0xC0689F0E9E24FBC3),
    ("G21", 0x4063A9470C18C1A7),
    ("G26", 0x4079EF7D9618F6B0),
    ("G27", 0xC0775231A845D789),
]

CARRIER_ARC_ROWS = [
    (
        0,
        0x419AD7697CF35157,
        0x4194F2CAD78DD8CA,
        0x4174689C023D70A4,
        0x4174689BFD06A506,
        0,
        0,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        1,
        0x419AD771514355CD,
        0x4194F2D0F16095A4,
        0x417468A1F420C49C,
        0x417468A1F5C5F3EE,
        0,
        0,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        2,
        0x419AD779344A2977,
        0x4194F2D716AA7F95,
        0x417468A7F9374BC6,
        0x417468A7F499BDB8,
        0,
        0,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        3,
        0x419AD7812607CC59,
        0x4194F2DD476B96A2,
        0x417468ADFFE76C8B,
        0x417468AE036D8781,
        0,
        0,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        4,
        0x419AD7893E7C3E72,
        0x4194F2E383A3DAC9,
        0x417468B41A45A1CB,
        0x417468B41574847E,
        0,
        0,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        5,
        0x419AD7914DA77FC0,
        0x4194F2E9CB534C08,
        0x417468BA38A3D70A,
        0x417468BA3BA4773D,
        0,
        0,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        6,
        0x419AD7996B899045,
        0x4194F2F01E79EA62,
        0x417468C06AD91687,
        0x417468C066CA2C8C,
        0,
        1,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        7,
        0x419AD7A198226FFF,
        0x4194F2F67D17B5D4,
        0x417468C6A06A7EF9,
        0x417468C6A2BCAEA7,
        0,
        0,
        0x41D779C018000000,
        0x41D24AEC20000000,
    ),
    (
        8,
        0x419AD7A9D3721EF1,
        0x4194F2FCE72CAE62,
        0x417468CCE5D2F1A9,
        0x417468CCE471C01E,
        0,
        0,
        None,
        0x41D24AEC20000000,
    ),
]

SIGNAL_PRN1_CHIPS = [
    -1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
    1,
    1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    1,
    -1,
]

SIGNAL_REPLICA_SAMPLES = [
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    1,
    1,
    1,
    1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    1,
    -1,
    -1,
]


def _f64(bits):
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def _bits(value):
    return int.from_bytes(struct.pack(">d", value), "big")


def _maybe_bits(value):
    return None if value is None else _bits(value)


def _array_bits(values):
    return np.asarray(values, dtype=np.float64).view(np.uint64)


def _expect_bits(hex_values):
    return np.asarray([int(value, 16) for value in hex_values], dtype=np.uint64)


def _load_sp3():
    with open(SP3_PATH, "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _carrier_arc(rows=CARRIER_ARC_ROWS):
    return [
        sidereon.ArcEpoch(
            phi1_cycles=_f64(phi1),
            phi2_cycles=_f64(phi2),
            p1_m=_f64(p1),
            p2_m=_f64(p2),
            lli1=lli1,
            lli2=lli2,
            f1_hz=None if f1 is None else _f64(f1),
            f2_hz=None if f2 is None else _f64(f2),
            gap_time_s=float(epoch),
        )
        for epoch, phi1, phi2, p1, p2, lli1, lli2, f1, f2 in rows
    ]


def _clean_signal(prn, code_phase_chips, doppler_hz, n, sample_rate_hz):
    options = sidereon.ReplicaOptions(
        sample_rate_hz=sample_rate_hz,
        num_samples=n,
        code_phase_chips=code_phase_chips,
    )
    code = sidereon.replica(prn, options).astype(np.float64)
    theta = (2.0 * np.pi * doppler_hz / sample_rate_hz) * np.arange(n, dtype=np.float64)
    return np.column_stack((code * np.cos(theta), code * np.sin(theta))).astype(
        np.float64
    )


def test_carrier_frequency_constants_and_default_pair():
    gps = sidereon.GnssSystem.GPS
    assert gps.letter == "G"
    assert sidereon.CarrierBand.L1.name == "l1"
    assert (
        sidereon.carrier_frequency_hz(gps, sidereon.CarrierBand.L1) == 1_575_420_000.0
    )
    assert (
        sidereon.carrier_frequency_hz(gps, sidereon.CarrierBand.L2) == 1_227_600_000.0
    )
    assert sidereon.carrier_frequency_hz(gps, sidereon.CarrierBand.E1) is None

    pair = sidereon.default_pair(gps)
    assert pair == sidereon.CarrierPair(
        sidereon.CarrierBand.L1, sidereon.CarrierBand.L2
    )
    assert pair.band1 == sidereon.CarrierBand.L1
    assert pair.band2 == sidereon.CarrierBand.L2
    assert sidereon.default_pair(sidereon.GnssSystem.GLONASS) is None


def test_wavelength_is_c_over_frequency():
    expected = 299_792_458.0 / 1_575_420_000.0
    got = sidereon.wavelength_m(sidereon.GnssSystem.GPS, sidereon.CarrierBand.L1)
    assert got == expected


def test_rinex_band_frequency_lookup():
    assert (
        sidereon.rinex_band_frequency_hz(sidereon.GnssSystem.GPS, "1")
        == 1_575_420_000.0
    )
    assert (
        sidereon.rinex_band_frequency_hz(sidereon.GnssSystem.GALILEO, "5")
        == 1_176_450_000.0
    )
    assert (
        sidereon.rinex_band_frequency_hz(
            sidereon.GnssSystem.GLONASS, "1", glonass_channel=1
        )
        == 1_602_562_500.0
    )
    assert sidereon.rinex_band_frequency_hz(sidereon.GnssSystem.GLONASS, "1") is None

    wavelength = sidereon.rinex_band_wavelength_m(
        sidereon.GnssSystem.GLONASS, "1", glonass_channel=-7
    )
    assert wavelength == 299_792_458.0 / (1_602_000_000.0 - 7.0 * 562_500.0)


def test_rinex_band_requires_one_character():
    with pytest.raises(ValueError, match="single RINEX"):
        sidereon.rinex_band_frequency_hz(sidereon.GnssSystem.GPS, "12")
    with pytest.raises(ValueError, match="single RINEX"):
        sidereon.rinex_band_wavelength_m(sidereon.GnssSystem.GPS, "")


def test_linear_combination_scalars_match_rust_oracle_bits():
    f1 = _f64(0x41D779C018000000)
    f2 = _f64(0x41D24AEC20000000)

    assert _bits(sidereon.phase_meters(123_456_789.25, f1)) == 0x4176679B5DBB7FD0
    assert _bits(sidereon.gamma(f1, f2)) == 0x40045DA686C28E3C
    assert _bits(sidereon.noise_amplification(f1, f2)) == 0x4007D3777C503EBC
    assert (
        _bits(
            sidereon.ionosphere_free(
                _f64(0x4175EF3C40772A36), _f64(0x4175EF3C6A2BCBB5), f1, f2
            )
        )
        == 0x4175EF3C00000000
    )
    assert (
        _bits(
            sidereon.ionosphere_free_phase_m(
                _f64(0x4175F4F80DDD7ECD), _f64(0x4175FD37D057D184), f1, f2
            )
        )
        == 0x4175E837D93B3CBA
    )
    assert (
        _bits(
            sidereon.ionosphere_free_phase_cycles(
                _f64(0x419CD8990A6A993B), _f64(0x419682AD3BEA73B9), f1, f2
            )
        )
        == 0x4175E837D93B3CBA
    )

    assert _bits(sidereon.geometry_free(100.0, 60.0)) == 0x4044000000000000
    assert _bits(sidereon.wide_lane_wavelength(f1, f2)) == 0x3FEB94D5E5A6844D
    assert _bits(sidereon.narrow_lane_code(10.0, 12.0, f1, f2)) == 0x4025C077975B8FE2
    assert (
        _bits(sidereon.melbourne_wubbena(5.0, 3.0, 10.0, 12.0, f1, f2))
        == 0xC0224DDCDAA6BF58
    )
    wl_cycles = float.fromhex("-0x1.24ddcdaa6bf58p+3") / float.fromhex(
        "0x1.b94d5e5a6844dp-1"
    )
    assert sidereon.wide_lane_cycles(5.0, 3.0, 10.0, 12.0, f1, f2) == wl_cycles


def test_linear_combination_errors_are_value_errors():
    f1 = _f64(0x41D779C018000000)
    f2 = _f64(0x41D24AEC20000000)

    with pytest.raises(ValueError, match="equal carrier frequencies"):
        sidereon.gamma(f1, f1)
    with pytest.raises(ValueError, match="carrier frequencies must be positive"):
        sidereon.ionosphere_free_phase_cycles(1.0, 2.0, 0.0, f2)
    with pytest.raises(ValueError, match="carrier frequency must be positive"):
        sidereon.phase_meters(1.0, 0.0)
    with pytest.raises(ValueError, match="equal carrier frequencies"):
        sidereon.wide_lane_wavelength(f1, f1)


def test_cycle_slips_detect_clean_and_injected_arc_bits():
    opts = sidereon.CycleSlipOptions()
    assert opts.gf_threshold_m == 0.05
    assert opts.mw_threshold_cycles == 4.0
    assert opts.min_arc_gap_s == 300.0

    clean = sidereon.detect_cycle_slips(_carrier_arc(CARRIER_ARC_ROWS[:4]), opts)
    assert [result.slip for result in clean] == [False, False, False, False]
    assert [result.skipped for result in clean] == [False, False, False, False]

    actual = sidereon.detect_cycle_slips(_carrier_arc(), opts)
    expected = [
        (False, [], 0xC0E07FD931E60E00, 0xC0F7618A9FB55C00, False),
        (False, [], 0xC0E07FD93C7F8A00, 0xC0F76189F4FDF000, False),
        (False, [], 0xC0E07FD947190400, 0xC0F7618B8B4D9C00, False),
        (False, [], 0xC0E07FD951B28000, 0xC0F76189D67F0600, False),
        (
            True,
            [sidereon.SlipReason.GEOMETRY_FREE, sidereon.SlipReason.MELBOURNE_WUBBENA],
            0xC0E07FB4D2FB7200,
            0xC0F76136A660BE00,
            False,
        ),
        (False, [], 0xC0E07FB4DD94EC00, 0xC0F76136155F9C00, False),
        (
            True,
            [sidereon.SlipReason.LLI],
            0xC0E07FB4E82E6800,
            0xC0F76137A3E94C00,
            False,
        ),
        (False, [], 0xC0E07FB4F2C7E200, 0xC0F761373E423D00, False),
        (False, [], None, None, True),
    ]

    assert actual[4].reasons[0].label == "geometry_free"
    assert actual[6].reasons[0].label == "lli"
    assert len(actual) == len(expected)
    for got, (slip, reasons, gf, mw, skipped) in zip(actual, expected):
        assert got.slip is slip
        assert got.reasons == reasons
        assert _maybe_bits(got.gf_m) == gf
        assert _maybe_bits(got.mw_m) == mw
        assert got.skipped is skipped


def test_hatch_smoothing_matches_rust_oracle_bits():
    actual = sidereon.smooth_code(_carrier_arc(), hatch_window_cap=100)
    expected = [
        (0x4174689C023D70A4, 1, False),
        (0x417468A1F6000000, 2, False),
        (0x417468A7F7A06D39, 3, False),
        (0x417468AE02B851EB, 4, False),
        (0x417468B41A45A1CB, 1, True),
        (0x417468BA3AAC0831, 2, False),
        (0x417468C06AD91687, 1, True),
        (0x417468C6A20C49BA, 2, False),
        (None, 0, False),
    ]

    assert len(actual) == len(expected)
    for got, (p_smooth, window, reset) in zip(actual, expected):
        assert _maybe_bits(got.p_smooth_m) == p_smooth
        assert got.window == window
        assert got.reset is reset


def test_ionosphere_free_hatch_smoothing_matches_rust_oracle_bits():
    actual = sidereon.smooth_iono_free_code(_carrier_arc(), hatch_window_cap=100)
    expected = [
        (0x4174689C0A4CAB98, 0x4174689C0A4CAB98, 0x41746197D93B3CB8, 1, False),
        (0x417468A1F8BE0026, 0x417468A1F195BB1B, 0x4174619DCED4D652, 2, False),
        (0x417468A7FBCFC0CC, 0x417468A80059A882, 0x417461A3CFA1A31E, 3, False),
        (0x417468AE0479118A, 0x417468ADFA7503C6, 0x417461A9DBA1A31E, 4, False),
        (0x417468B421B7B0F7, 0x417468B421B7B0F7, 0x417461B021565566, 1, True),
        (0x417468BA3C0EEC2C, 0x417468BA33FFC0F9, 0x417461B643BCBBCE, 2, False),
        (0x417468C0711EF759, 0x417468C0711EF759, 0x417461BC71565568, 1, True),
        (0x417468C6A35FE7EF, 0x417468C69CD40BB8, 0x417461C2AA232235, 2, False),
        (None, None, None, 0, False),
    ]

    assert len(actual) == len(expected)
    for got, (p_smooth, p_if, l_if, window, reset) in zip(actual, expected):
        assert _maybe_bits(got.p_smooth_m) == p_smooth
        assert _maybe_bits(got.p_if_m) == p_if
        assert _maybe_bits(got.l_if_m) == l_if
        assert got.window == window
        assert got.reset is reset


def test_doppler_range_rate_conversions_match_formula():
    f_l1 = sidereon.carrier_frequency_hz(
        sidereon.GnssSystem.GPS, sidereon.CarrierBand.L1
    )

    assert (
        sidereon.doppler_to_range_rate(-1250.0, f_l1) == 1250.0 * 299_792_458.0 / f_l1
    )
    assert sidereon.range_rate_to_doppler(42.0, f_l1) == -42.0 * f_l1 / 299_792_458.0


def _velocity_observations():
    f_l1 = sidereon.carrier_frequency_hz(
        sidereon.GnssSystem.GPS, sidereon.CarrierBand.L1
    )
    return [
        sidereon.VelocityObservation(sat, _f64(bits), f_l1)
        for sat, bits in VELOCITY_OBS_BITS
    ]


def test_velocity_solve_matches_rust_oracle_bits():
    sp3 = _load_sp3()
    observations = _velocity_observations()
    receiver = np.asarray([4_500_000.0, 500_000.0, 4_500_000.0], dtype=np.float64)
    options = sidereon.VelocitySolveOptions()

    assert options.observable == sidereon.VelocityObservable.RANGE_RATE
    assert options.observable.label == "range_rate"
    solution = sidereon.solve_velocity(
        sp3, observations, receiver, 646_272_000.0, options
    )

    assert solution.used_sats == [sat for sat, _ in VELOCITY_OBS_BITS]
    assert solution.velocity_m_s.shape == (3,)
    assert solution.residuals_m_s.shape == (len(observations),)
    assert np.array_equal(
        _array_bits(solution.velocity_m_s),
        _expect_bits(
            ["0x4028000000000000", "0xc01c000000000016", "0x4007ffffffffff00"]
        ),
    )
    assert _bits(solution.speed_m_s) == 0x402C6CE322982A37
    assert _bits(solution.clock_drift_s_s) == 0x3E112E0BE826D2EE
    assert np.array_equal(
        _array_bits(solution.residuals_m_s),
        _expect_bits(
            [
                "0xbd01000000000000",
                "0xbd24000000000000",
                "0x3cfc000000000000",
                "0xbd16000000000000",
                "0xbd1a800000000000",
                "0x3cf0000000000000",
                "0xbd14000000000000",
                "0x3d31800000000000",
                "0x3d18000000000000",
            ]
        ),
    )


def test_velocity_doppler_solve_matches_rust_oracle_bits():
    sp3 = _load_sp3()
    range_rate_observations = _velocity_observations()
    doppler_observations = []
    for idx, obs in enumerate(range_rate_observations):
        channel = (idx % 14) - 7
        carrier = sidereon.rinex_band_frequency_hz(
            sidereon.GnssSystem.GLONASS, "1", glonass_channel=channel
        )
        doppler_observations.append(
            sidereon.VelocityObservation(
                obs.satellite_id,
                sidereon.range_rate_to_doppler(obs.value, carrier),
                carrier,
            )
        )

    solution = sidereon.solve_velocity(
        sp3,
        doppler_observations,
        np.asarray([4_500_000.0, 500_000.0, 4_500_000.0], dtype=np.float64),
        646_272_000.0,
        sidereon.VelocitySolveOptions(observable=sidereon.VelocityObservable.DOPPLER),
    )

    assert np.array_equal(
        _array_bits(solution.velocity_m_s),
        _expect_bits(
            ["0x402800000000000c", "0xc01c00000000000f", "0x4007ffffffffff60"]
        ),
    )
    assert _bits(solution.speed_m_s) == 0x402C6CE322982A44
    assert _bits(solution.clock_drift_s_s) == 0x3E112E0BE826D4B8


def test_ionosphere_free_pseudoranges_report_drop_reasons():
    band1 = [
        ("G01", 23_000_000.0),
        ("G01", 23_000_010.0),
        ("G02", 22_000_000.0),
        ("G03", 21_000_000.0),
        ("X01", 20_000_000.0),
    ]
    band2 = [
        ("G01", 23_000_000.0),
        ("G02", 22_000_000.0),
        ("G04", 24_000_000.0),
        ("X01", 20_000_000.0),
    ]

    combined, dropped = sidereon.ionosphere_free_pseudoranges(band1, band2)

    assert combined == [("G02", pytest.approx(22_000_000.0))]
    assert dropped == [
        ("G01", sidereon.PseudorangeDropReason.DUPLICATE_OBSERVATION),
        ("G03", sidereon.PseudorangeDropReason.MISSING_BAND2),
        ("G04", sidereon.PseudorangeDropReason.MISSING_BAND1),
        ("X01", sidereon.PseudorangeDropReason.UNKNOWN_SYSTEM),
    ]
    assert dropped[0][1].label == "duplicate_observation"


def test_pseudorange_override_system_requires_one_character():
    with pytest.raises(ValueError, match="single RINEX system"):
        sidereon.ionosphere_free_pseudoranges(
            [("G01", 23_000_000.0)], [("G01", 23_000_000.0)], [("GPS", "l1", "l2")]
        )


def test_quality_variance_and_weight_vectors_match_rust_oracle():
    opts = sidereon.PseudorangeVarianceOptions()
    assert opts.a_m == 0.3
    assert opts.b_m == 0.3
    assert opts.model == sidereon.PseudorangeVarianceModel.ELEVATION
    assert opts.model.label == "elevation"
    assert opts.cn0_dbhz is None
    assert opts.cn0_scale_m2 == 1.0
    assert sidereon.pseudorange_variance(30.0, opts) == pytest.approx(0.45, abs=1.0e-15)

    entries = [
        sidereon.WeightEntry("G02", 15.0),
        sidereon.WeightEntry("G01", 75.0),
        sidereon.WeightEntry("G03", 0.0),
    ]
    sigma_sats, sigma_values = sidereon.sigmas(entries, opts)
    weight_sats, weight_values = sidereon.weight_vector(entries, opts)

    assert sigma_sats == ["G01", "G02"]
    assert weight_sats == sigma_sats
    assert sigma_values.dtype == np.float64
    assert weight_values.dtype == np.float64
    assert sigma_values.shape == (2,)
    assert weight_values.shape == (2,)
    assert sigma_values[1] > sigma_values[0]
    assert weight_values[1] < weight_values[0]
    np.testing.assert_allclose(
        weight_values, 1.0 / (sigma_values * sigma_values), rtol=1.0e-15
    )


def test_quality_cn0_model_and_errors():
    missing = sidereon.PseudorangeVarianceOptions(
        model=sidereon.PseudorangeVarianceModel.ELEVATION_CN0
    )
    assert missing.model.label == "elevation_cn0"
    with pytest.raises(ValueError, match="missing C/N0"):
        sidereon.pseudorange_variance(30.0, missing)
    with pytest.raises(ValueError, match="invalid elevation"):
        sidereon.pseudorange_variance(0.0)

    weak = sidereon.pseudorange_variance(
        30.0,
        sidereon.PseudorangeVarianceOptions(
            model=sidereon.PseudorangeVarianceModel.ELEVATION_CN0,
            cn0_dbhz=30.0,
        ),
    )
    strong = sidereon.pseudorange_variance(
        30.0,
        sidereon.PseudorangeVarianceOptions(
            model=sidereon.PseudorangeVarianceModel.ELEVATION_CN0,
            cn0_dbhz=50.0,
        ),
    )
    assert strong < weak


def test_raim_weights_expose_sorted_numpy_vector():
    unit = sidereon.RaimWeights.unit()
    assert unit.is_unit
    assert unit.satellite_ids == []
    assert unit.weights.dtype == np.float64
    assert unit.weights.shape == (0,)

    weights = sidereon.RaimWeights.by_satellite(
        ["G02", "G01"], np.asarray([0.25, 4.0], dtype=np.float64)
    )
    assert not weights.is_unit
    assert weights.satellite_ids == ["G01", "G02"]
    assert np.array_equal(weights.weights, np.asarray([4.0, 0.25], dtype=np.float64))

    with pytest.raises(ValueError, match="same length"):
        sidereon.RaimWeights.by_satellite(
            ["G01"], np.asarray([1.0, 2.0], dtype=np.float64)
        )
    with pytest.raises(ValueError, match="positive finite"):
        sidereon.RaimWeights.by_satellite(["G01"], np.asarray([0.0], dtype=np.float64))


def test_signal_ca_code_and_replica_match_rust_oracle():
    code = sidereon.ca_code(1)
    assert code.dtype == np.int8
    assert code.shape == (1023,)
    assert code[: len(SIGNAL_PRN1_CHIPS)].tolist() == SIGNAL_PRN1_CHIPS
    assert sidereon.ca_chip(1, -1) == int(code[-1])

    one_period = sidereon.ReplicaOptions.one_code_period()
    assert one_period.sample_rate_hz == 2_046_000.0
    assert one_period.num_samples == 2046
    assert one_period.code_phase_chips == 0.0
    assert one_period.code_doppler_hz == 0.0

    options = sidereon.ReplicaOptions(
        sample_rate_hz=float.fromhex("0x1.f383000000000p+20"),
        num_samples=64,
        code_phase_chips=float.fromhex("0x1.ff00000000000p+8"),
    )
    replica = sidereon.replica(5, options)
    assert replica.dtype == np.int8
    assert replica.tolist() == SIGNAL_REPLICA_SAMPLES

    with pytest.raises(ValueError, match="unsupported GPS C/A PRN"):
        sidereon.ca_code(33)


def test_signal_correlate_and_acquire_match_rust_oracle_bits():
    prn = 5
    fs = float.fromhex("0x1.f383000000000p+20")
    doppler = float.fromhex("0x1.f400000000000p+9")
    code_phase = float.fromhex("0x1.ff00000000000p+8")

    iq = _clean_signal(prn, code_phase, doppler, 64, fs)
    correlation = sidereon.correlate(
        iq,
        prn,
        sidereon.CorrelateOptions(
            sample_rate_hz=fs,
            doppler_hz=doppler,
            code_phase_chips=code_phase,
        ),
    )
    assert _bits(correlation.i) == _bits(float.fromhex("0x1.0000000000000p+6"))
    assert _bits(correlation.q) == _bits(float.fromhex("0x0.0p+0"))
    assert _bits(correlation.power) == _bits(float.fromhex("0x1.0000000000000p+12"))

    samples = _clean_signal(prn, code_phase, doppler, 2046, fs)
    acquisition = sidereon.acquire(
        samples, prn, sidereon.AcquisitionOptions(sample_rate_hz=fs)
    )
    assert _bits(acquisition.code_phase_chips) == _bits(code_phase)
    assert _bits(acquisition.doppler_hz) == _bits(doppler)
    assert _bits(acquisition.peak_power) == _bits(
        float.fromhex("0x1.ff00200000000p+21")
    )
    assert _bits(acquisition.metric) == 0x409369E276358FF0
    assert _bits(acquisition.peak_metric) == _bits(acquisition.metric)
    assert acquisition.grid.code_phase_bins == 2046
    assert _bits(acquisition.grid.samples_per_chip) == _bits(float.fromhex("0x1.0p+1"))
    assert acquisition.grid.doppler_step_hz == 500.0
    assert np.array_equal(
        acquisition.grid.doppler_hz, np.arange(-2500.0, 2500.0 + 500.0, 500.0)
    )

    with pytest.raises(ValueError, match="shape"):
        sidereon.correlate(np.zeros((4, 3), dtype=np.float64), prn)
    with pytest.raises(ValueError, match="empty sample vector"):
        sidereon.correlate(np.empty((0, 2), dtype=np.float64), prn)


def test_signal_loss_helpers_match_rust_oracle_bits():
    cases = [
        (
            float.fromhex("0x0.0p+0"),
            float.fromhex("0x1.0624dd2f1a9fcp-10"),
            float.fromhex("0x1.0000000000000p+0"),
            float.fromhex("0x0.0p+0"),
        ),
        (
            float.fromhex("0x1.f400000000000p+7"),
            float.fromhex("0x1.0624dd2f1a9fcp-10"),
            float.fromhex("0x1.9f02f6222c71fp-1"),
            float.fromhex("-0x1.d2fe745bc3f62p-1"),
        ),
        (
            float.fromhex("0x1.f400000000000p+8"),
            float.fromhex("0x1.0624dd2f1a9fcp-9"),
            float.fromhex("0x1.f8f7171d21750p-110"),
            float.fromhex("-0x1.482ecab293e4ep+8"),
        ),
        (
            float.fromhex("0x1.ec00000000000p+6"),
            float.fromhex("0x1.0624dd2f1a9fcp-8"),
            float.fromhex("0x1.ac58d00563d6dp-2"),
            float.fromhex("-0x1.e47c4a524edbdp+1"),
        ),
    ]

    for freq_error_hz, integration_time_s, loss, loss_db in cases:
        assert _bits(
            sidereon.coherent_loss(freq_error_hz, integration_time_s)
        ) == _bits(loss)
        assert _bits(
            sidereon.coherent_loss_db(freq_error_hz, integration_time_s)
        ) == _bits(loss_db)
    assert _bits(sidereon.snr_post_db(40.0, 1.0e-3)) == _bits(10.0)
