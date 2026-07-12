"""RINEX OBS parsing through the Python binding uses real committed fixtures."""

import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES

OBS_FIXTURES = os.path.join(FIXTURES, "obs")
NAV_FIXTURES = os.path.join(FIXTURES, "nav")
ESBC_TRIM = "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
ESBC_NAV = "ESBC00DNK_R_20201770000_01D_MN.rnx"


def _read_obs(name=ESBC_TRIM):
    with open(os.path.join(OBS_FIXTURES, name), encoding="utf-8") as fh:
        return fh.read()


def _row_index(series, satellite, code):
    return next(
        index
        for index, (sat, candidate) in enumerate(zip(series.satellites, series.codes))
        if sat == satellite and candidate == code
    )


def test_parse_rinex_obs_header_and_epochs_from_fixture():
    obs = sidereon.parse_rinex_obs(_read_obs())

    assert obs.epoch_count == 2
    assert len(obs.epochs) == 2
    assert "RinexObs(" in repr(obs)

    header = obs.header
    assert header.version == 3.05
    assert header.marker_name == "ESBC00DNK"
    assert header.interval_s == 30.0
    np.testing.assert_allclose(
        header.approx_position_m,
        np.array([3582105.2910, 532589.7313, 5232754.8054]),
        rtol=0.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        header.antenna_delta_hen_m,
        np.array([0.2160, 0.0, 0.0]),
        rtol=0.0,
        atol=1e-12,
    )
    assert sidereon.GnssSystem.GPS in header.systems
    assert header.obs_codes(sidereon.GnssSystem.GPS)[:5] == [
        "C1C",
        "C1W",
        "C2L",
        "C2W",
        "C5Q",
    ]
    assert header.obs_codes(sidereon.GnssSystem.BEIDOU)[0] == "C2I"
    assert len(header.phase_shifts) >= 20
    assert header.time_of_first_obs[0].year == 2020
    assert header.time_of_first_obs[1] == sidereon.TimeScale.GPST
    assert (1, 1) in header.glonass_slots

    epoch0 = obs.epoch(0)
    assert epoch0.flag == 0
    assert epoch0.satellite_count == 43
    assert epoch0.epoch.second == 0.0
    assert "G05" in epoch0.satellites
    assert obs.epoch(1).epoch.second == 30.0


def test_rinex_obs_pseudoranges_are_numpy_series():
    obs = sidereon.parse_rinex_obs(_read_obs())
    ranges = obs.pseudoranges(0)

    assert isinstance(ranges.ranges_m, np.ndarray)
    assert ranges.ranges_m.dtype == np.float64
    assert ranges.ranges_m.shape == (39,)
    by_sat = dict(zip(ranges.satellites, ranges.ranges_m))

    assert by_sat["C05"] == 40715949.461
    assert by_sat["E01"] == 27616185.992
    assert by_sat["G05"] == 20947300.931
    assert by_sat["R01"] == 19307563.721

    gps_policy = sidereon.SignalPolicy([(sidereon.GnssSystem.GPS, ["C1C"])])
    gps_ranges = obs.pseudoranges(0, gps_policy)
    assert all(sat.startswith("G") for sat in gps_ranges.satellites)
    assert len(gps_ranges) == 12


def test_rinex_obs_raw_values_and_carrier_rows_are_filtered_numpy_series():
    obs = sidereon.parse_rinex_obs(_read_obs())
    filt = sidereon.ObservationFilter([(sidereon.GnssSystem.GPS, ["C1C", "L1C"])])
    rows = obs.observation_values(0, filt)

    assert isinstance(rows.values, np.ndarray)
    assert rows.values.dtype == np.float64
    assert len(rows) == 24

    c_idx = _row_index(rows, "G05", "C1C")
    l_idx = _row_index(rows, "G05", "L1C")
    assert rows.kinds[c_idx] == sidereon.ObservationKind.PSEUDORANGE
    assert rows.kinds[l_idx] == sidereon.ObservationKind.CARRIER_PHASE
    assert rows.values[c_idx] == 20947300.931
    assert rows.values[l_idx] == 110078836.389
    assert rows.ssi[c_idx] == 8.0
    assert np.isnan(rows.lli[c_idx])

    phase = obs.carrier_phase_rows(
        0,
        sidereon.ObservationFilter([(sidereon.GnssSystem.GPS, ["L1C"])]),
    )
    p_idx = _row_index(phase, "G05", "L1C")
    assert isinstance(phase.value_cycles, np.ndarray)
    assert phase.value_cycles[p_idx] == 110078836.389
    assert phase.frequency_hz[p_idx] == 1575420000.0
    np.testing.assert_allclose(
        phase.value_m[p_idx],
        phase.value_cycles[p_idx] * phase.wavelength_m[p_idx],
        rtol=0.0,
        atol=1e-9,
    )
    assert phase.phase_shift_cycles[p_idx] == 0.0


def test_load_rinex_obs_accepts_path_and_bytes_and_errors_are_typed():
    path = os.path.join(OBS_FIXTURES, ESBC_TRIM)
    text = _read_obs()

    assert sidereon.load_rinex_obs(path).epoch_count == 2
    assert sidereon.load_rinex_obs(text.encode("utf-8")).epoch_count == 2

    nav_text = "     3.05           N: GNSS NAV DATA    M (MIXED)           RINEX VERSION / TYPE\n"  # noqa: E501
    with pytest.raises(sidereon.RinexObsParseError):
        sidereon.parse_rinex_obs(nav_text)

    with pytest.raises(ValueError):
        sidereon.parse_rinex_obs(text).epoch(99)


def test_rinex2_oversized_epoch_count_is_rejected_before_allocation():
    def header_line(body, label):
        return f"{body:<60}{label}"

    text = "\n".join(
        [
            header_line(
                "     2.11           OBSERVATION DATA    G (GPS)",
                "RINEX VERSION / TYPE",
            ),
            header_line(
                "     6    L1    L2    C1    P1    S1    S2",
                "# / TYPES OF OBSERV",
            ),
            header_line("", "END OF HEADER"),
            "5 10 5 5 0 5 0 155444444444444",
        ]
    )

    with pytest.raises(sidereon.RinexObsParseError, match="I3 field maximum of 999"):
        sidereon.parse_rinex_obs(text)


def test_to_rinex_string_round_trips_header_and_epochs():
    obs = sidereon.parse_rinex_obs(_read_obs())
    reparsed = sidereon.parse_rinex_obs(obs.to_rinex_string())

    assert reparsed.epoch_count == obs.epoch_count
    assert reparsed.header.version == obs.header.version
    assert reparsed.header.marker_name == obs.header.marker_name
    np.testing.assert_array_equal(
        reparsed.header.approx_position_m, obs.header.approx_position_m
    )
    for system in obs.header.systems:
        assert reparsed.obs_codes(system) == obs.obs_codes(system)
    # The pseudorange rows of the first epoch survive the re-encode bit-for-bit.
    original = obs.pseudoranges(0)
    again = reparsed.pseudoranges(0)
    assert again.satellites == original.satellites
    np.testing.assert_array_equal(again.ranges_m, original.ranges_m)


def test_rinex_obs_spp_inputs_and_solve_convenience_with_broadcast_nav():
    obs = sidereon.load_rinex_obs(os.path.join(OBS_FIXTURES, ESBC_TRIM))
    nav = sidereon.load_rinex_nav(os.path.join(NAV_FIXTURES, ESBC_NAV))
    options = sidereon.RinexSppOptions(
        obs,
        signal_policy=sidereon.SignalPolicy([(sidereon.GnssSystem.GPS, ["C1C"])]),
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=True),
    )

    inputs = sidereon.spp_inputs_from_rinex_obs(nav, obs, options)
    assert len(inputs) == obs.epoch_count
    assert inputs[0].epoch_index == 0
    assert inputs[0].epoch == obs.epochs[0].epoch
    assert inputs[0].observation_count >= 5
    assert all(sat.startswith("G") for sat in inputs[0].satellites)
    assert inputs[0].observations[0].satellite_id == inputs[0].satellites[0]

    solved = sidereon.solve_spp_from_rinex_obs(nav, obs, options)
    assert len(solved) == len(inputs)
    assert solved[0].solved
    assert solved[0].solution is not None
    assert solved[0].solution.geodetic is not None
    assert solved[0].solution.position.shape == (3,)
    assert np.all(np.isfinite(solved[0].solution.position))
