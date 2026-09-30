"""PPP range corrections through the binding.

`scripts/ppp_esbc_expected` solves the ESBC float arc with the solid-earth tide,
the phase wind-up and the relativistic satellite range term, built with the core
`ppp_corrections::build` and `PppCorrectionLookup::from_options`, and writes the
solution to `ppp_esbc.json` as `expected["float_with_corrections"]`. The binding
builds the same corrections through `ppp_corrections`, `PppCorrectionsOptions`,
`PppCorrectionLookup.from_corrections` and `PppRangeCorrections`, and must return
that solution bit for bit.
"""

import pytest
import sidereon
from test_ppp import (
    _assert_outcome,
    _assert_ppp_metadata,
    _epochs,
    _load_fixture,
    _load_sp3,
    _options,
    _state,
    _tropo,
    _weights,
)

# The ESBC arc is GPS only and its observations state no carriers; the wind-up
# rows take the GPS L1 and L2 carriers, as the generator passes them.
GPS_L1_HZ = 1_575_420_000.0
GPS_L2_HZ = 1_227_600_000.0


def _correction_epochs(fx):
    out = []
    for epoch in fx["epochs"]:
        civil = epoch["civil"]
        observations = []
        for obs in epoch["observations"]:
            assert obs["satellite_id"].startswith("G")
            observations.append(
                sidereon.PppCorrectionObservation(
                    obs["satellite_id"], GPS_L1_HZ, GPS_L2_HZ
                )
            )
        out.append(
            sidereon.PppCorrectionEpoch(
                civil["year"],
                civil["month"],
                civil["day"],
                civil["hour"],
                civil["minute"],
                civil["second"],
                epoch["t_rx_j2000_s"],
                observations,
            )
        )
    return out


def test_ppp_float_solve_with_corrections_matches_core():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    receiver = fx["initial_state"]["position_m"]
    corrections = sidereon.ppp_corrections(
        sp3,
        _correction_epochs(fx),
        receiver,
        solid_earth_tide=True,
        phase_windup=True,
    )
    options = sidereon.PppCorrectionsOptions(solid_earth_tide=True, phase_windup=True)
    lookup = sidereon.PppCorrectionLookup.from_corrections(corrections, options)
    assert lookup.tide_enabled is True
    assert lookup.windup_enabled is True
    assert len(lookup.tide) == len(fx["epochs"])
    assert len(lookup.windup_m) == sum(len(e["observations"]) for e in fx["epochs"])

    raw = fx["config"]
    config = sidereon.PppFloatConfig(
        weights=_weights(raw["weights"]),
        tropo=_tropo(raw["tropo"]),
        options=_options(raw["opts"]),
        residual_screen=raw["residual_screen"],
        corrections=sidereon.PppRangeCorrections(sat_clock_relativity=True, ppp=lookup),
    )
    assert config.corrections.sat_clock_relativity is True
    assert config.corrections.ppp.tide == lookup.tide

    sol = sidereon.solve_ppp_float(
        sp3, epochs=_epochs(fx), initial_state=_state(fx), config=config
    )

    expected = fx["expected"]["float_with_corrections"]
    _assert_ppp_metadata(sol, expected)
    _assert_outcome(sol, expected)


def test_ppp_pcv_sample_round_trip():
    plain = sidereon.PppPcvSample(10.0, 0.005)
    assert (plain.zenith_deg, plain.value_m, plain.azimuth_deg) == (10.0, 0.005, None)
    placed = sidereon.PppPcvSample(25.0, -0.003, azimuth_deg=45.0)
    assert (placed.zenith_deg, placed.value_m, placed.azimuth_deg) == (
        25.0,
        -0.003,
        45.0,
    )
    assert "PppPcvSample(" in repr(placed)


def test_ppp_receiver_antenna_round_trip():
    l1 = sidereon.PppReceiverAntennaFrequency(
        "G01",
        [0.001, -0.002, 0.005],
        [sidereon.PppPcvSample(0.0, 0.001), sidereon.PppPcvSample(10.0, 0.002, 30.0)],
    )
    assert l1.label == "G01"
    assert l1.pco_m == [0.001, -0.002, 0.005]
    assert [(s.zenith_deg, s.value_m, s.azimuth_deg) for s in l1.pcv_samples] == [
        (0.0, 0.001, None),
        (10.0, 0.002, 30.0),
    ]
    l2 = sidereon.PppReceiverAntennaFrequency("G02", [0.0, 0.0, 0.1], [])
    antenna = sidereon.PppReceiverAntennaOptions(
        "G01", GPS_L1_HZ, "G02", GPS_L2_HZ, [l1, l2]
    )
    assert (antenna.freq1_label, antenna.freq1_hz) == ("G01", GPS_L1_HZ)
    assert (antenna.freq2_label, antenna.freq2_hz) == ("G02", GPS_L2_HZ)
    assert [f.label for f in antenna.frequencies] == ["G01", "G02"]
    assert antenna.frequencies[1].pco_m == [0.0, 0.0, 0.1]
    assert "PppReceiverAntennaOptions(" in repr(antenna)


def test_ppp_satellite_clock_corrections_round_trip():
    clock = sidereon.PppSatelliteClockCorrections(
        {"G01": [(0.0, 1.2e-6), (30.0, 1.25e-6)], "E02": [(0.0, -0.5e-6)]}
    )
    # Keys come back in the core map's order: GPS before Galileo.
    assert list(clock.series.items()) == [
        ("G01", [(0.0, 1.2e-6), (30.0, 1.25e-6)]),
        ("E02", [(0.0, -0.5e-6)]),
    ]
    with pytest.raises(ValueError, match="X99"):
        sidereon.PppSatelliteClockCorrections({"X99": [(0.0, 1e-6)]})


def test_ppp_correction_lookup_round_trip():
    empty = sidereon.PppCorrectionLookup()
    for table in (
        "tide",
        "pole_tide",
        "ocean_loading",
        "windup_m",
        "sat_pcv_m",
        "code_bias_m",
        "sat_pco_ecef",
        "ssr_code_bias_m",
        "phase_bias_m",
    ):
        assert getattr(empty, table) == {}
    flags = (
        "tide_enabled",
        "pole_tide_enabled",
        "ocean_loading_enabled",
        "windup_enabled",
        "satellite_antenna_enabled",
        "code_bias_enabled",
        "ssr_code_bias_enabled",
        "phase_bias_enabled",
    )
    for flag in flags:
        assert getattr(empty, flag) is False

    tables = {
        "tide": {1: (0.1, 0.2, 0.3)},
        "pole_tide": {2: (0.01, 0.02, 0.03)},
        "ocean_loading": {3: (0.001, 0.002, 0.003)},
        "windup_m": {("G01", 0): 0.05},
        "sat_pcv_m": {("G01", 0): -0.002},
        "code_bias_m": {("G01", 0): 0.12},
        "sat_pco_ecef": {("G01", 0): (0.1, 0.2, 0.3)},
        "ssr_code_bias_m": {("G01", 0, "G01"): 0.15},
        "phase_bias_m": {("G01", 0, "G01"): -0.03},
    }
    full = sidereon.PppCorrectionLookup(**tables, **{flag: True for flag in flags})
    for name, value in tables.items():
        assert getattr(full, name) == value
    for flag in flags:
        assert getattr(full, flag) is True


@pytest.mark.parametrize(
    "table,value",
    [
        ("windup_m", {("X99", 0): 0.1}),
        ("sat_pcv_m", {("X99", 0): 0.1}),
        ("code_bias_m", {("X99", 0): 0.1}),
        ("sat_pco_ecef", {("X99", 0): (0.0, 0.0, 0.0)}),
        ("ssr_code_bias_m", {("X99", 0, "X99"): 0.1}),
        ("phase_bias_m", {("X99", 0, "X99"): 0.1}),
    ],
)
def test_ppp_correction_lookup_refuses_a_bad_satellite_token(table, value):
    with pytest.raises(ValueError, match="X99"):
        sidereon.PppCorrectionLookup(**{table: value})


def test_ppp_corrections_options_round_trip():
    default = sidereon.PppCorrectionsOptions()
    assert default.solid_earth_tide is False
    assert default.phase_windup is False
    assert default.satellite_antenna is None
    assert default.pole_tide is None
    assert default.ocean_loading is None
    assert default.code_bias is None
    pole = sidereon.PoleTideOptions(0.1, 0.3)
    options = sidereon.PppCorrectionsOptions(
        solid_earth_tide=True, phase_windup=True, pole_tide=pole
    )
    assert options.solid_earth_tide is True
    assert options.phase_windup is True
    assert options.pole_tide is not None


def test_ppp_range_corrections_round_trip():
    disabled = sidereon.PppRangeCorrections.disabled()
    default = sidereon.PppRangeCorrections()
    for value in (disabled, default):
        assert value.receiver_antenna is None
        assert value.sat_clock_relativity is False
        assert value.satellite_clock is None
        assert value.ppp.tide == {}
        assert value.ppp.tide_enabled is False

    antenna = sidereon.PppReceiverAntennaOptions(
        "G01",
        GPS_L1_HZ,
        "G02",
        GPS_L2_HZ,
        [sidereon.PppReceiverAntennaFrequency("G01", [0.0, 0.0, 0.0], [])],
    )
    clock = sidereon.PppSatelliteClockCorrections({"G01": [(0.0, 1e-6)]})
    lookup = sidereon.PppCorrectionLookup(tide_enabled=True)
    full = sidereon.PppRangeCorrections(
        receiver_antenna=antenna,
        sat_clock_relativity=True,
        satellite_clock=clock,
        ppp=lookup,
    )
    assert full.receiver_antenna.freq1_label == "G01"
    assert full.sat_clock_relativity is True
    assert full.satellite_clock.series == {"G01": [(0.0, 1e-6)]}
    assert full.ppp.tide_enabled is True
    assert "PppRangeCorrections(" in repr(full)


def test_ppp_configs_default_to_disabled_corrections():
    disabled = sidereon.PppRangeCorrections.disabled()
    fixed = sidereon.PppFixedConfig(
        sidereon.PppFixedAmbiguityOptions({"G01": 0.19}, {"G01": 0.0})
    )
    for config in (sidereon.PppFloatConfig(), fixed):
        corrections = config.corrections
        assert corrections.receiver_antenna is None
        assert corrections.sat_clock_relativity is disabled.sat_clock_relativity
        assert corrections.satellite_clock is None
        for name in ("tide", "windup_m", "sat_pco_ecef", "ssr_code_bias_m"):
            assert getattr(corrections.ppp, name) == getattr(disabled.ppp, name)
        assert corrections.ppp.tide_enabled is disabled.ppp.tide_enabled
    assert fixed.ambiguity.wavelengths_m == {"G01": 0.19}


def test_ppp_corrections_code_bias_and_diagnostics():
    fx = _load_fixture()
    sp3 = _load_sp3(fx)
    corrections = sidereon.ppp_corrections(
        sp3,
        _correction_epochs(fx)[:1],
        fx["initial_state"]["position_m"],
        solid_earth_tide=True,
    )
    assert corrections.code_bias_m == []
    assert corrections.diagnostics.skip_count == 0
    assert corrections.diagnostics.warning_count == 0
