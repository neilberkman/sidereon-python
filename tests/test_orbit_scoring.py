from pathlib import Path

import pytest
import sidereon as s
from _helpers import CORE_FIXTURES, FIXTURES, core_goldens, hex_to_f64


def test_rtn_and_sisre():
    components = s.rtn_components([3.0, 4.0, 12.0], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0])
    assert components == s.RtnComponents(3.0, 4.0, 12.0)
    assert components != s.RtnComponents(3.0, 4.0, 0.0)
    assert [components.radial_m, components.transverse_m, components.normal_m] == [
        3.0,
        4.0,
        12.0,
    ]
    weights = s.SisreWeights(1.0, 1.0)
    assert weights.radial() == weights.transverse_normal_squared() == 1.0
    assert weights.orbit_only_m(components) == 13.0
    assert weights.with_clock_m(components, 6.0) == 13.0
    assert s.inertial_velocity_from_earth_fixed(
        [0.0, 0.0, 7_000_000.0], [1.0, 2.0, 3.0]
    ) == [1.0, 2.0, 3.0]
    assert (
        s.OrbitRegime.from_inertial_state([26_560_000.0, 0.0, 0.0], [0.0, 3874.0, 0.0])
        == s.OrbitRegime.MEO
    )
    gps = s.SisreWeights.montenbruck_2018(s.GnssSystem.GPS, s.OrbitRegime.MEO)
    assert gps == s.SisreWeights(0.98, 1.0 / 49.0)
    assert s.SisreWeights.montenbruck_2018(s.GnssSystem.GPS, s.OrbitRegime.GEO) is None
    frames = s.Sp3FrameReconciliationOptions([["IGS20", "ITRF2020"]], helmert=True)
    assert frames.asserted_equivalent_label_sets == [["IGS20", "ITRF2020"]]
    assert frames.helmert
    transform = s.FrameAgreement.helmert(
        s.TerrestrialFrame.ITRF2014, s.TerrestrialFrame.ITRF2020
    )
    assert transform.from_frame == s.TerrestrialFrame.ITRF2014
    assert transform.to_frame == s.TerrestrialFrame.ITRF2020
    assert transform != s.FrameAgreement.ASSERTED_EQUIVALENT
    assert s.FrameAgreement.BROADCAST_DATUMS.from_frame is None
    assert [
        str(regime)
        for regime in (s.OrbitRegime.MEO, s.OrbitRegime.IGSO, s.OrbitRegime.GEO)
    ] == ["MEO", "IGSO", "GEO"]
    assert str(s.ScoredProduct.SOURCE) == "orbit source"
    assert str(s.ScoredProduct.REFERENCE) == "reference product"
    assert s.TriadVelocity.default() == s.TriadVelocity.INERTIAL
    assert s.ClockDatum.default() == s.ClockDatum.EPOCH_SYSTEM_MEDIAN
    assert s.OrbitScoreGapPolicy.default() == s.OrbitScoreGapPolicy.COUNT
    assert s.ScoreEpochs() == s.ScoreEpochs.REFERENCE_RECORDS
    assert s.ScoreEpochs.REFERENCE_RECORDS.epoch_instants is None
    assert s.SisreWeightSelection() == s.SisreWeightSelection.MONTENBRUCK_2018
    assert s.SisreWeightSelection.fixed(weights).fixed_weights == s.SisreWeights(
        1.0, 1.0
    )
    assert s.SisreWeightSelection().fixed_weights is None
    assert s.DEFAULT_VELOCITY_HALF_STEP_S == 0.5
    assert s.MIN_VELOCITY_HALF_STEP_S == 1e-8
    with pytest.raises(s.OrbitScoreError) as caught:
        s.rtn_components([0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [float("nan"), 0, 0])
    assert caught.value.detail == {
        "kind": "invalid_input",
        "field": "velocity_m_s",
        "reason": "not finite",
    }


def test_sp3_and_sample_trajectory_scores():
    product = s.load_sp3(
        Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    )
    first, second = product.epoch_instants[:2]
    first_seconds = product.epochs_j2000_seconds[0]
    options = s.OrbitScoreOptions(
        [-1.0, 0.0, 1800.0],
        epochs=s.ScoreEpochs.instants([first, second]),
        triad=s.TriadVelocity.EARTH_FIXED,
        sisre_weights=s.SisreWeightSelection.fixed(s.SisreWeights(1.0, 1.0)),
        clock_datum=s.ClockDatum.UNADJUSTED,
        gaps=s.OrbitScoreGapPolicy.REFUSE,
    )
    assert options.epochs.epoch_instants == [first, second]
    assert options.horizon_bin_edges_s == [-1.0, 0.0, 1800.0]
    assert options.sisre_weights.fixed_weights == s.SisreWeights(1.0, 1.0)
    assert options != s.OrbitScoreOptions([-1.0, 0.0, 1800.0])
    assert s.OrbitScoreOptions([0.0, 1.0]) == s.OrbitScoreOptions([0.0, 1.0])
    empty = s.ScoreCoverage()
    assert [
        empty.epochs,
        empty.scored,
        empty.clocked,
        empty.no_prediction_origin,
        empty.outside_horizon_bins,
        empty.source_satellite_missing,
        empty.source_epoch_missing,
        empty.reference_satellite_missing,
        empty.reference_epoch_missing,
    ] == [0] * 9
    assert options.triad == s.TriadVelocity.EARTH_FIXED
    assert options.velocity_half_step_s == 0.5
    assert options.clock_datum == s.ClockDatum.UNADJUSTED
    assert options.gaps == s.OrbitScoreGapPolicy.REFUSE
    assert not options.frame_reconciliation.helmert
    assert options.frame_reconciliation.asserted_equivalent_label_sets == []
    source = s.PredictedSp3.with_cutoff(product, first)
    assert source.product() is product
    assert source.time_scale() == s.TimeScale.GPST
    assert source.sp3_time_system() == "GPS"
    assert source.frame() == s.OrbitFrame.label("IGc20")
    assert source.frame().frame_label == "IGc20"
    assert not source.clock_includes_relativity()
    assert source.reference_point() == s.OrbitScoreReferencePoint.CENTER_OF_MASS
    assert source.prediction_origin_j2000_s("G01") == first_seconds
    flagged = s.PredictedSp3.from_prediction_flags(product)
    assert flagged.prediction_origin_j2000_s("G01") == 830823300.0
    state = source.state_at_j2000_s("G01", first_seconds)
    assert first_seconds == 830814300.0
    assert state.position_ecef_m == [8322271.954, -18334373.649, -17289305.601]
    assert state.clock_s == 286.148987 * 1e-6
    samples = [
        s.TrajectorySample(
            "G01",
            epoch,
            product.state("G01", index).position_m,
            product.state("G01", index).clock_s,
        )
        for index, epoch in enumerate([first, second])
    ]
    assert samples[0].satellite == "G01"
    assert samples[0].epoch == first
    assert samples[0].position_ecef_m == state.position_ecef_m
    assert samples[0].clock_s == 286.148987 * 1e-6
    sample = s.TrajectorySample(
        "G01", first, [8322271.954, -18334373.649, -17289305.601], 286.148987 * 1e-6
    )
    assert samples[0] == sample
    assert samples[1] != sample
    trajectory = s.OrbitTrajectory("IGc20", first, samples)
    assert trajectory == s.OrbitTrajectory("IGc20", first, samples)
    assert trajectory != trajectory.with_clock_relativity_included(True)
    assert trajectory.with_clock_relativity_included(True).clock_includes_relativity()
    assert trajectory.epochs() == [first, second]
    assert trajectory.satellites() == ["G01"]
    assert trajectory.state_at_j2000_s("G01", first_seconds) == state
    assert trajectory.prediction_origin_j2000_s("G01") == first_seconds
    assert trajectory.sp3_time_system() is None
    assert trajectory.time_scale() == s.TimeScale.GPST
    assert trajectory.frame() == s.OrbitFrame.label("IGc20")
    assert trajectory.reference_point() == s.OrbitScoreReferencePoint.CENTER_OF_MASS
    reports = []
    for orbit in (source, trajectory):
        report = s.score_orbit(orbit, product, ["G01"], options)
        reports.append(report)
        assert report.horizon_bin_edges_s == [-1.0, 0.0, 1800.0]
        assert report.frame == s.FrameAgreement.SAME_LABEL
        assert (
            report.source_reference_point == s.OrbitScoreReferencePoint.CENTER_OF_MASS
        )
        assert (
            report.coverage.scored
            == report.coverage.epochs
            == report.coverage.clocked
            == 2
        )
        coverage = report.coverage
        assert [
            coverage.outside_horizon_bins,
            coverage.no_prediction_origin,
            coverage.source_satellite_missing,
            coverage.source_epoch_missing,
            coverage.reference_satellite_missing,
            coverage.reference_epoch_missing,
        ] == [0] * 6
        (satellite,) = report.per_satellite
        (system,) = report.per_system
        assert satellite.satellite == "G01"
        assert satellite.prediction_origin_j2000_s == first_seconds
        assert satellite.orbit_regime is None
        assert satellite.sisre_weights == s.SisreWeights(1.0, 1.0)
        assert system.system == s.GnssSystem.GPS
        assert system.coverage == satellite.coverage == report.coverage
        assert system.bins == satellite.bins
        for expected, bin_values in zip(
            [(-1.0, 0.0), (0.0, 1800.0)], satellite.bins, strict=True
        ):
            assert (bin_values.horizon_start_s, bin_values.horizon_end_s) == expected
            assert bin_values.count == bin_values.clock_count == 1
            for values in [
                bin_values.radial,
                bin_values.transverse,
                bin_values.normal,
                bin_values.orbit_3d,
                bin_values.sisre_orbit,
                bin_values.clock,
                bin_values.sisre,
            ]:
                assert [values.mean_m, values.rms_m, values.max_abs_m] == [0.0] * 3
    first_report, second_report = reports
    assert first_report == second_report
    assert first_report.per_satellite == second_report.per_satellite
    assert first_report.per_system == second_report.per_system
    assert first_report.coverage != empty
    first_bin, second_bin = first_report.per_satellite[0].bins
    assert first_bin != second_bin
    assert first_bin.radial == second_bin.radial
    assert second_report.per_satellite[0].bins[0] == first_bin
    alternate_frame = s.OrbitTrajectory("ITRF2020", first, samples)
    equivalent_options = s.OrbitScoreOptions(
        [-1.0, 0.0, 1800.0],
        epochs=s.ScoreEpochs.instants([first, second]),
        triad=s.TriadVelocity.EARTH_FIXED,
        sisre_weights=s.SisreWeightSelection.fixed(s.SisreWeights(1.0, 1.0)),
        clock_datum=s.ClockDatum.UNADJUSTED,
        frame_reconciliation=s.Sp3FrameReconciliationOptions([["ITRF2020", "IGc20"]]),
    )
    equivalent = s.score_orbit(alternate_frame, product, ["G01"], equivalent_options)
    assert equivalent.frame == s.FrameAgreement.ASSERTED_EQUIVALENT
    assert equivalent.frame != s.FrameAgreement.SAME_LABEL
    assert equivalent.per_satellite[0].bins[0].orbit_3d.rms_m == 0.0
    with pytest.raises(s.OrbitScoreError) as caught:
        s.score_orbit(trajectory, product, ["G02"], options)
    assert caught.value.detail == {
        "kind": "missing_satellite",
        "product": s.ScoredProduct.SOURCE,
        "satellite": "G02",
    }


def test_broadcast_source_metadata():
    epoch = s.ClockInstant.from_nanos(s.TimeScale.GPST, 0)
    ephemeris = s.load_rinex_nav(
        Path(FIXTURES) / "nav/BRDC00GOP_R_20210010000_01D_MN.rnx"
    )
    source = s.BroadcastOrbit(ephemeris, epoch)
    assert source.prediction_origin_j2000_s("G01") == 0.0
    assert source.time_scale() == s.TimeScale.GPST
    assert source.sp3_time_system() is None
    assert source.frame() == s.OrbitFrame.BROADCAST_DATUMS
    assert source.frame().frame_label is None
    assert source.clock_includes_relativity()
    assert source.reference_point() == s.OrbitScoreReferencePoint.ANTENNA_PHASE_CENTER
    with pytest.raises(s.OrbitScoreError) as caught:
        s.BroadcastOrbit(ephemeris, s.ClockInstant.from_nanos(s.TimeScale.UTC, 0))
    assert caught.value.detail == {
        "kind": "instant_time_system_mismatch",
        "field": "prediction_origin",
        "index": 0,
        "found": s.TimeScale.UTC,
        "expected": s.TimeScale.GPST,
    }


def test_broadcast_score_statistics():
    fx = core_goldens()["orbital_dynamics"]["broadcast_scores"]
    epochs = [
        s.ClockInstant.from_nanos(s.TimeScale.GPST, epoch * 10**9)
        for epoch in fx["epochs"]
    ]
    reference = s.load_sp3(
        Path(CORE_FIXTURES) / "sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    )
    nav = s.load_rinex_nav(Path(FIXTURES) / "nav/ESBC00DNK_R_20201770000_01D_MN.rnx")
    source = s.BroadcastOrbit(nav, epochs[0])
    options = s.OrbitScoreOptions(
        [-1.0, 600.0],
        epochs=s.ScoreEpochs.instants(epochs),
        clock_datum=s.ClockDatum.UNADJUSTED,
        triad=s.TriadVelocity.EARTH_FIXED,
        gaps=s.OrbitScoreGapPolicy.REFUSE,
    )
    report = s.score_orbit(source, reference, ["G01"], options)
    assert report.frame == s.FrameAgreement.BROADCAST_DATUMS
    for entry in (report.per_satellite[0], report.per_system[0]):
        (bin_values,) = entry.bins
        assert (bin_values.count, bin_values.clock_count) == (
            fx["count"],
            fx["clock_count"],
        )
        for stat, expected in zip(
            [
                bin_values.radial,
                bin_values.transverse,
                bin_values.normal,
                bin_values.orbit_3d,
                bin_values.sisre_orbit,
                bin_values.clock,
                bin_values.sisre,
            ],
            fx["statistics"],
            strict=True,
        ):
            assert stat is not None
            assert [stat.mean_m.hex(), stat.rms_m.hex(), stat.max_abs_m.hex()] == [
                hex_to_f64(value).hex() for value in expected
            ]


def test_score_source_refusals():
    path = Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    reference = s.load_sp3(path)
    first, second = reference.epoch_instants[:2]
    utc = s.ClockInstant.from_nanos(s.TimeScale.UTC, 0)
    position = [20e6, 10e6, 2e6]
    options = s.OrbitScoreOptions(
        [-1.0, 1800.0],
        epochs=s.ScoreEpochs.instants([first, second]),
        sisre_weights=s.SisreWeightSelection.fixed(s.SisreWeights(1.0, 1.0)),
        gaps=s.OrbitScoreGapPolicy.REFUSE,
    )
    cases = [
        (
            s.OrbitTrajectory("IGc20", utc, [s.TrajectorySample("G01", utc, position)]),
            {
                "kind": "time_system_mismatch",
                "orbit_source": s.TimeScale.UTC,
                "reference": s.TimeScale.GPST,
            },
        ),
        (
            s.OrbitTrajectory(
                "ITRF2020", first, [s.TrajectorySample("G01", first, position)]
            ),
            {
                "kind": "frame_mismatch",
                "orbit_source": "ITRF2020",
                "reference": "IGc20",
            },
        ),
        (
            s.PredictedSp3.with_cutoff(
                s.load_sp3(path.read_bytes().replace(b"%c G  cc GPS", b"%c G  cc IRN")),
                first,
            ),
            {
                "kind": "time_system_label_mismatch",
                "orbit_source": "IRN",
                "reference": "GPS",
            },
        ),
        (
            s.OrbitTrajectory(
                "IGc20", first, [s.TrajectorySample("G01", first, position)]
            ),
            {
                "kind": "missing_epoch",
                "product": s.ScoredProduct.SOURCE,
                "satellite": "G01",
                "epoch_index": 1,
                "error": {
                    "family": "CoreError",
                    "kind": "epoch_out_of_range",
                    "message": "epoch out of range",
                },
            },
        ),
        (
            s.OrbitTrajectory(
                "IGc20", first, [s.TrajectorySample("G01", first, [1.7e308] * 3)]
            ),
            {"kind": "non_finite_result", "satellite": "G01"},
        ),
    ]
    for source, detail in cases:
        with pytest.raises(s.OrbitScoreError) as caught:
            s.score_orbit(source, reference, ["G01"], options)
        assert caught.value.detail == detail


def test_score_prediction_origin_refusal():
    directory = Path(CORE_FIXTURES) / "sp3"
    predicted = s.load_sp3(
        directory / "IGS0OPSULT_20261311800_02D_15M_ORB_2245_0415.SP3"
    )
    reference = s.load_sp3(directory / "IGS0OPSFIN_20261330000_03H_15M_ORB.SP3")
    with pytest.raises(s.OrbitScoreError) as caught:
        s.score_orbit(
            s.PredictedSp3.from_prediction_flags(predicted),
            reference,
            ["G01"],
            s.OrbitScoreOptions(
                [18000.0, 25200.0, 32400.0], gaps=s.OrbitScoreGapPolicy.REFUSE
            ),
        )
    assert caught.value.detail == {"kind": "no_prediction_origin", "satellite": "G01"}


@pytest.mark.parametrize(
    "radius,triad,detail",
    [
        (
            100.0,
            s.TriadVelocity.EARTH_FIXED,
            {
                "kind": "degenerate_reference_state",
                "satellite": "G01",
                "epoch_index": 0,
            },
        ),
        (
            100.0,
            s.TriadVelocity.INERTIAL,
            {"kind": "undetermined_orbit_regime", "satellite": "G01"},
        ),
        (
            42164.0,
            s.TriadVelocity.INERTIAL,
            {
                "kind": "no_sisre_weights",
                "satellite": "G01",
                "regime": s.OrbitRegime.GEO,
            },
        ),
    ],
)
def test_score_reference_geometry_refusals(radius, triad, detail):
    path = Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    position = (
        (0.0, 0.0, radius)
        if triad == s.TriadVelocity.EARTH_FIXED
        else (radius, 0.0, 0.0)
    )
    coordinate_fields = "".join(f"{value:14.6f}" for value in position)
    text = (
        "\n".join(
            f"PG01{coordinate_fields}{line[46:]}" if line.startswith("PG01") else line
            for line in path.read_text().splitlines()
        )
        + "\n"
    )
    reference = s.load_sp3(text.encode())
    first = reference.epoch_instants[0]
    source = s.PredictedSp3.with_cutoff(reference, first)
    weights = (
        s.SisreWeightSelection.fixed(s.SisreWeights(1.0, 1.0))
        if triad == s.TriadVelocity.EARTH_FIXED
        else s.SisreWeightSelection.MONTENBRUCK_2018
    )
    options = s.OrbitScoreOptions(
        [-1.0, 1800.0],
        epochs=s.ScoreEpochs.instants([first]),
        triad=triad,
        sisre_weights=weights,
    )
    with pytest.raises(s.OrbitScoreError) as caught:
        s.score_orbit(source, reference, ["G01"], options)
    assert caught.value.detail == detail


def test_score_helmert_overflow_refusal():
    reference = s.load_sp3(
        Path(FIXTURES) / "sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    )
    first = reference.epoch_instants[0]
    max_finite = float.fromhex("0x1.fffffffffffffp+1023")
    source = s.OrbitTrajectory(
        "ITRF2014", first, [s.TrajectorySample("G01", first, [max_finite] * 3)]
    )
    options = s.OrbitScoreOptions(
        [-1.0, 1800.0],
        epochs=s.ScoreEpochs.instants([first]),
        frame_reconciliation=s.Sp3FrameReconciliationOptions(helmert=True),
        sisre_weights=s.SisreWeightSelection.fixed(s.SisreWeights(1.0, 1.0)),
    )
    with pytest.raises(s.OrbitScoreError) as caught:
        s.score_orbit(source, reference, ["G01"], options)
    assert caught.value.detail == {
        "kind": "frame_transform",
        "error": {"kind": "invalid_input", "field": "x_m", "reason": "must be finite"},
    }
