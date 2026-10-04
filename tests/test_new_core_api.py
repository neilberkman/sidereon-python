"""Smoke tests for the multi-system / PPP-correction core API surface.

These exercise the headline new core entry points the binding wraps as the first
real consumer of the post-campaign core: inter-system time-scale offsets
(A1), per-system TDOP (A3), the multi-system constellation catalog (A2), the PPP
correction options pole tide / ocean loading / VMF1 mapping (B1), and the SP3
merge agreement metrics (B2). The numbers themselves are the core's; these only
confirm the binding marshals the new shapes through faithfully.
"""

import datetime as dt
import json
import os
import pathlib
from fractions import Fraction

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, core_goldens, hex_to_f64

C_M_S = 299792458.0


def test_exact_epoch_integer_constructor_order_hash_and_query_offsets():
    atto = sidereon.ExactEpoch.ATTOSECONDS_PER_SECOND
    assert atto == 1_000_000_000_000_000_000
    assert sidereon.ExactEpoch.J2000 == sidereon.ExactEpoch.new(0, 0)

    before_j2000 = sidereon.ExactEpoch.new(-1, atto - 1)
    j2000 = sidereon.ExactEpoch.new(0, 0)
    same_before_j2000 = sidereon.ExactEpoch.new(-1, atto - 1)
    assert before_j2000 < j2000
    assert before_j2000 <= same_before_j2000
    assert before_j2000 == same_before_j2000
    assert hash(before_j2000) == hash(same_before_j2000)
    assert {before_j2000: "exact"}[same_before_j2000] == "exact"

    query = sidereon.ExactEpochQuery.at_epoch(j2000).checked_add_binary_seconds(0.125)
    origin = sidereon.ExactEpochQuery.at_epoch(j2000)
    assert query.seconds_since_epoch(j2000) == 0.125
    assert query.seconds_since_query(origin) == 0.125
    assert sidereon.ExactEpochQuery.from_binary_j2000_seconds(0.125) == query
    assert sidereon.ExactEpoch.from_binary_j2000_seconds(0.125) == query
    assert query.epoch() == j2000

    with pytest.raises(ValueError):
        sidereon.ExactEpoch.new(0, atto)


def test_sp3_raw_and_effective_accuracy_are_both_exposed():
    sp3 = _load_sp3("IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3")
    raw = sp3.record_accuracy_codes("G01", 0)
    decoded = sp3.record_accuracy("G01", 0)

    assert raw.p is not None
    assert tuple(raw.p.axis_exponents) == (3, 4, 5)
    assert raw.p.position_velocity_base == 1.25
    assert raw.p.position_velocity_base == sp3.header.pos_vel_base
    assert decoded.p is not None
    for exponent, sigma in zip(raw.p.axis_exponents, decoded.p.position_sigma_m):
        assert exponent is not None
        assert sigma.kind == "known"
        expected_sigma = float(Fraction(5, 4) ** exponent * Fraction.from_float(1.0e-3))
        assert sigma.value == expected_sigma

    assert sidereon.Sp3AccuracyValue.known(0.25).variance().value == 0.0625
    assert sidereon.Sp3AccuracyValue.unknown().variance().kind == "unknown"


def test_exact_ephemeris_source_hooks_preserve_state_and_selection_queries():
    sp3 = _load_sp3("IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3")
    epoch = sidereon.ExactEpochQuery.from_binary_j2000_seconds(
        float(sp3.epochs_j2000_seconds[5])
    )
    selection_epoch = epoch.checked_add_binary_seconds(0.25)
    state = sp3.selected_state_at_epoch_query("G01", epoch, selection_epoch)
    assert state is not None
    raw_state = sp3.position_at_epoch_query("G01", epoch)
    assert np.array_equal(state.position_ecef_m, raw_state.position_m)
    assert state.clock_s == raw_state.clock_s
    assert state.group_delay_s is None
    assert state.degraded_reason is None

    placed_clock = sp3.transmit_epoch_clock_at_epoch_query(
        "G01", epoch, selection_epoch
    )
    assert placed_clock is not None
    assert placed_clock[0] == raw_state.clock_s
    assert sp3.ephemeris_variance_at_epoch_query("G01", epoch, selection_epoch) >= 0.0
    relativity = sp3.clock_relativity_for_state_at_epoch_query(
        "G01", epoch, raw_state.position_m
    )
    assert relativity.kind in {"term", "unavailable"}
    if relativity.kind == "term":
        assert relativity.term_s is not None

    interpolant = sidereon.PreciseEphemerisInterpolant.from_sp3(sp3)
    assert (
        interpolant.selected_state_at_epoch_query("G01", epoch, selection_epoch).clock_s
        == raw_state.clock_s
    )
    artifact = sidereon.PreciseInterpolantArtifact.from_bytes(
        sp3.precise_interpolant_artifact_bytes()
    )
    assert (
        artifact.selected_state_at_epoch_query("G01", epoch, selection_epoch).clock_s
        == raw_state.clock_s
    )


def test_precise_samples_keep_native_epoch_when_float_epochs_collapse():
    first_epoch = sidereon.ClockInstant.from_nanos(
        sidereon.TimeScale.GPST, 900_000_000_000_000_000
    )
    second_epoch = sidereon.ClockInstant.from_nanos(
        sidereon.TimeScale.GPST, 900_000_000_000_000_001
    )
    assert first_epoch != second_epoch

    first_sample = sidereon.PreciseEphemerisSample.from_instant(
        "G01", first_epoch, [1.0, 2.0, 3.0]
    )
    second_sample = sidereon.PreciseEphemerisSample.from_instant(
        "G01", second_epoch, [1.0, 2.0, 3.0]
    )
    assert first_sample.epoch == first_epoch
    assert second_sample.epoch == second_epoch
    assert np.isnan(first_sample.epoch_j2000_seconds)
    assert np.isnan(second_sample.epoch_j2000_seconds)

    unknown = sidereon.Sp3AccuracyValue.unknown()
    first_accuracy = sidereon.PreciseEphemerisAccuracySample.from_instant(
        "G01", first_epoch, [unknown, unknown, unknown], unknown
    )
    second_accuracy = sidereon.PreciseEphemerisAccuracySample.from_instant(
        "G01", second_epoch, [unknown, unknown, unknown], unknown
    )
    assert first_accuracy.epoch == first_epoch
    assert second_accuracy.epoch == second_epoch
    assert first_accuracy.epoch != second_accuracy.epoch
    assert np.isnan(first_accuracy.epoch_j2000_seconds)
    assert np.isnan(second_accuracy.epoch_j2000_seconds)


# --- A1: inter-system time-scale offsets -----------------------------------


def test_timescale_offset_fixed_atomic_pairs():
    # BDT is 14 s behind GPST (BDT - GPST = -14 s).
    assert (
        sidereon.timescale_offset(sidereon.TimeScale.GPST, sidereon.TimeScale.BDT)
        == -14.0
    )
    # GST and QZSST are nominally synchronous with GPST.
    assert (
        sidereon.timescale_offset(sidereon.TimeScale.GPST, sidereon.TimeScale.GST)
        == 0.0
    )
    assert (
        sidereon.timescale_offset(sidereon.TimeScale.GPST, sidereon.TimeScale.QZSST)
        == 0.0
    )


def test_timescale_offset_rejects_utc_based_scales():
    # UTC-based scales (UTC, GLONASST) have a leap-dependent offset; the fixed
    # form must error and point the caller at the leap-aware variant.
    with pytest.raises(ValueError):
        sidereon.timescale_offset(sidereon.TimeScale.UTC, sidereon.TimeScale.GPST)
    with pytest.raises(ValueError):
        sidereon.timescale_offset(sidereon.TimeScale.GLONASST, sidereon.TimeScale.GPST)


def test_timescale_offset_at_is_leap_aware():
    # GLONASST = UTC + 3 h regardless of the leap count, so UTC->GLONASST is
    # exactly 10800 s at any epoch.
    utc_jd = 2461000.5
    assert (
        sidereon.timescale_offset_at(
            sidereon.TimeScale.UTC, sidereon.TimeScale.GLONASST, utc_jd
        )
        == 10800.0
    )
    # A purely atomic pair ignores the epoch and matches the fixed form.
    assert sidereon.timescale_offset_at(
        sidereon.TimeScale.GPST, sidereon.TimeScale.BDT, utc_jd
    ) == sidereon.timescale_offset(sidereon.TimeScale.GPST, sidereon.TimeScale.BDT)


# --- A3: per-system TDOP ---------------------------------------------------


def test_dop_exposes_system_tdops():
    receiver = sidereon.Wgs84Geodetic(0.0, 0.0, 0.0)
    az = np.array([0.0, 90.0, 180.0, 270.0, 45.0], dtype=np.float64)
    el = np.array([80.0, 30.0, 45.0, 20.0, 60.0], dtype=np.float64)
    dop = sidereon.Dop.from_az_el(az, el, receiver)
    # The standalone geometry DOP path carries no constellation identity, so
    # the system-tagged per-clock vector is empty; read the scalar `tdop` for
    # the lone clock column. System tagging only appears on the SPP solve path.
    assert dop.system_tdops == []
    assert np.isfinite(dop.tdop) and dop.tdop > 0.0


def test_spp_solution_exposes_system_tdops():
    sp3 = _load_sp3("GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")
    rx, observations, t_rx = _glonass_scenario(sp3)
    assert len(observations) >= 4
    cfg = sidereon.SppConfig(
        observations=observations,
        t_rx_j2000_s=t_rx,
        t_rx_second_of_day_s=0.0,
        day_of_year=176.0,
        initial_guess=[6378137.0, 0.0, 0.0, 0.0],
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
        with_geodetic=True,
    )
    sol = sidereon.solve_spp(sp3, cfg)
    # GLONASS-only solve -> one (system, tdop) entry, system-tagged.
    assert len(sol.system_tdops) == 1
    system, tdop = sol.system_tdops[0]
    assert system == sidereon.GnssSystem.GLONASS
    assert np.isfinite(tdop) and tdop > 0.0


# --- A2: multi-system constellation catalog --------------------------------


def test_from_celestrak_json_glonass_resolves_slots_and_fdma_channels():
    text = _read_const("glonass_ops_sample.json")
    records = sidereon.from_celestrak_json(text, sidereon.GnssSystem.GLONASS)
    assert records, "GLONASS sample produced records"
    for rec in records:
        assert rec.system == sidereon.GnssSystem.GLONASS
        assert rec.sp3_id.startswith("R")
        # Every resolved GLONASS slot carries an FDMA channel in -7..=6.
        assert rec.fdma_channel is not None
        assert -7 <= rec.fdma_channel <= 6


def test_glonass_fdma_channel_helper_matches_record():
    text = _read_const("glonass_ops_sample.json")
    records = sidereon.from_celestrak_json(text, sidereon.GnssSystem.GLONASS)
    rec = records[0]
    assert sidereon.glonass_fdma_channel(rec.prn) == rec.fdma_channel
    # A non-GLONASS / unknown slot has no published channel.
    assert sidereon.glonass_fdma_channel(99) is None


def test_gnss_sp3_id_renders_per_system_tokens():
    assert sidereon.gnss_sp3_id(sidereon.GnssSystem.GPS, 7) == "G07"
    assert sidereon.gnss_sp3_id(sidereon.GnssSystem.GLONASS, 13) == "R13"
    assert sidereon.gnss_sp3_id(sidereon.GnssSystem.GALILEO, 1) == "E01"


def test_validation_prn_findings_are_system_tagged():
    text = _read_const("gps_ops_sample.json")
    records = sidereon.from_celestrak_json(text)
    report = sidereon.validate(records)
    # New tuple shape: every PRN finding carries its system.
    for finding in report.duplicate_prns + report.inactive_unusable_prns:
        system, prn = finding
        assert isinstance(system, sidereon.GnssSystem)
        assert isinstance(prn, int)


# --- B1: PPP correction options (pole tide, ocean loading, VMF1) -----------

_SP3_FILE = "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
_SAT = "G21"
_T_RX_J2000_S = (2459025.0 - 2451545.0) * 86400.0
_RECEIVER_M = [3512900.0, 780500.0, 5248700.0]
_F_L1_HZ = 1575.42e6
_F_L2_HZ = 1227.60e6


def _ppp_epoch():
    return sidereon.PppCorrectionEpoch(
        2020,
        6,
        24,
        12,
        0,
        0.0,
        _T_RX_J2000_S,
        [sidereon.PppCorrectionObservation(_SAT, _F_L1_HZ, _F_L2_HZ)],
    )


def test_ppp_corrections_pole_tide_produces_a_displacement():
    sp3 = _load_sp3(_SP3_FILE)
    corr = sidereon.ppp_corrections(
        sp3,
        [_ppp_epoch()],
        _RECEIVER_M,
        pole_tide=sidereon.PoleTideOptions(0.2, 0.35),
    )
    assert len(corr.pole_tide) == 1
    idx, vec = corr.pole_tide[0]
    assert idx == 0
    assert all(np.isfinite(v) for v in vec)
    assert any(v != 0.0 for v in vec)


def test_ppp_corrections_ocean_loading_produces_a_displacement():
    sp3 = _load_sp3(_SP3_FILE)
    # A finite, real-valued BLQ block (3 components x 11 constituents).
    amplitude = [
        [
            0.0030,
            0.0010,
            0.0006,
            0.0003,
            0.0020,
            0.0012,
            0.0006,
            0.0002,
            0.0001,
            0.0001,
            0.0001,
        ],
        [
            0.0010,
            0.0004,
            0.0002,
            0.0001,
            0.0006,
            0.0004,
            0.0002,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ],
        [
            0.0008,
            0.0003,
            0.0002,
            0.0001,
            0.0005,
            0.0003,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ],
    ]
    phase = [[0.0] * 11 for _ in range(3)]
    blq = sidereon.OceanLoadingBlq(amplitude, phase)
    corr = sidereon.ppp_corrections(sp3, [_ppp_epoch()], _RECEIVER_M, ocean_loading=blq)
    assert len(corr.ocean_loading) == 1
    idx, vec = corr.ocean_loading[0]
    assert idx == 0
    assert all(np.isfinite(v) for v in vec)
    assert any(v != 0.0 for v in vec)


def test_ocean_loading_blq_rejects_wrong_shape():
    with pytest.raises(ValueError):
        sidereon.OceanLoadingBlq([[0.0] * 11, [0.0] * 11], [[0.0] * 11] * 3)
    with pytest.raises(ValueError):
        sidereon.OceanLoadingBlq([[0.0] * 5] * 3, [[0.0] * 11] * 3)


def test_ppp_troposphere_vmf1_mapping_selected_by_samples():
    niell = sidereon.PppTroposphereOptions(enabled=True)
    assert niell.mapping == "niell"
    vmf = sidereon.PppTroposphereOptions(
        enabled=True,
        vmf1_samples=[
            (58849.0, 0.00121738, 0.00058796),
            (58849.25, 0.00121800, 0.00058850),
        ],
    )
    assert vmf.mapping == "vmf1"


def test_ppp_troposphere_vmf1_rejects_non_ascending_samples():
    with pytest.raises(ValueError):
        sidereon.PppTroposphereOptions(
            enabled=True,
            vmf1_samples=[(58849.0, 0.0012, 0.0006), (58849.0, 0.0012, 0.0006)],
        )


# --- B2: SP3 merge agreement metrics ---------------------------------------


def test_merge_sp3_agreement_metrics_for_coincident_sources():
    sp3 = _load_sp3("degenerate_coincident_5sat.sp3")
    options = sidereon.Sp3MergeOptions(min_agree=1, clock_min_common=1)
    # Merge the product with an identical copy: every cell has a 2-source
    # consensus that agrees exactly, so the agreement dispersion is zero.
    _merged, report = sidereon.merge_sp3([sp3, sp3], options)
    assert report.agreement_count == sp3.epoch_count * len(sp3.satellites)
    assert report.position_agreement_rms_m == 0.0
    assert report.position_agreement_max_m == 0.0
    epochs = report.per_epoch_agreement
    assert len(epochs) == sp3.epoch_count
    for epoch_s, sats, pos_rms, pos_max, _clk_rms, _clk_max in epochs:
        assert np.isfinite(epoch_s)
        assert sats == len(sp3.satellites)
        assert pos_rms == 0.0 and pos_max == 0.0


# --- Phase B: astrodynamics and geometry -----------------------------------


def test_anomaly_conversions_round_trip_and_kepler_reports_iterations():
    mean = 0.8
    ecc = 0.2
    eccentric = sidereon.mean_to_eccentric(mean, ecc)
    solution = sidereon.solve_kepler(mean, ecc)
    assert solution.anomaly == pytest.approx(eccentric)
    assert solution.iterations > 0
    true = sidereon.eccentric_to_true(eccentric, ecc)
    assert sidereon.true_to_mean(true, ecc) == pytest.approx(mean, abs=1e-14)


def test_equinoctial_and_modified_equinoctial_round_trip_classical_elements():
    coe = sidereon.ClassicalElements(
        11067.79,
        0.1,
        np.radians(28.5),
        np.radians(40.0),
        np.radians(15.0),
        np.radians(80.0),
    )
    eq = sidereon.coe2eq(coe)
    back = sidereon.eq2coe(eq)
    assert back.p == pytest.approx(coe.p, rel=1e-12)
    assert back.ecc == pytest.approx(coe.ecc, rel=1e-12)

    mee = sidereon.coe2mee(coe)
    back2 = sidereon.mee2coe(mee)
    assert back2.p == pytest.approx(coe.p, rel=1e-12)
    assert back2.ecc == pytest.approx(coe.ecc, rel=1e-12)

    r, v = sidereon.coe2rv(coe, 398600.4418)
    eq_from_rv = sidereon.rv2eq(list(r), tuple(v), 398600.4418)
    r2, v2 = sidereon.eq2rv(eq_from_rv, 398600.4418)
    np.testing.assert_allclose(r2, r, rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(v2, v, rtol=1e-10, atol=1e-10)

    direct = sidereon.EquinoctialElements(
        a=7000.0,
        h=0.01,
        k=0.02,
        p=0.03,
        q=0.04,
        lambda_=0.5,
    )
    assert direct.lambda_ == pytest.approx(0.5)


def test_angles_beta_and_relative_frames_round_trip():
    assert sidereon.angular_separation(
        [1.0, 0.0, 0.0], (0.0, 1.0, 0.0)
    ) == pytest.approx(90.0)
    assert sidereon.position_angle(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0)
    beta = sidereon.beta_angle_from_state(
        [7000.0, 0.0, 0.0],
        (0.0, 7.5, 0.0),
        [0.0, 0.0, 1.0],
    )
    assert beta == pytest.approx(90.0)

    chief = sidereon.CartesianState(0.0, [7000.0, 0.0, 0.0], (0.0, 7.5, 0.0))
    deputy = sidereon.CartesianState(0.0, (7001.0, 0.0, 0.0), [0.0, 7.501, 0.0])
    rel = sidereon.relative_state(chief, deputy)
    rebuilt = sidereon.absolute_from_relative(chief, rel)
    np.testing.assert_allclose(rebuilt.position_km, deputy.position_km, atol=1e-12)
    assert sidereon.rtn_to_inertial_rotation(chief).shape == (3, 3)
    assert sidereon.cw_stm(sidereon.mean_motion_from_state(chief), 60.0).shape == (6, 6)
    with pytest.raises(ValueError, match="invalid input for n") as err:
        sidereon.cw_stm(float("nan"), 60.0)
    assert "InvalidInput" not in str(err.value)


def test_body_observe_and_almanac_events_are_exposed():
    epoch = _unix_us(2026, 3, 20)
    obs = sidereon.observe_body(40.0, -105.0, 1.6, epoch, sidereon.Target.sun())
    assert np.isfinite(obs.apparent.right_ascension_deg)
    assert np.isfinite(obs.horizontal.elevation_deg)

    start = _unix_us(2026, 1, 1)
    end = _unix_us(2026, 12, 31)
    events = sidereon.seasons(start, end)
    kinds = [event.kind for event in events]
    assert sidereon.SeasonKind.MARCH_EQUINOX in kinds
    assert sidereon.SeasonKind.JUNE_SOLSTICE in kinds
    phases = sidereon.moon_phases(_unix_us(2026, 1, 1), _unix_us(2026, 2, 1))
    assert phases


# --- Phase B: drag, sampling, terrain, bias, SBAS/SSR, robust FDE ----------


def test_drag_force_and_decay_estimate_are_callable():
    weather = sidereon.SpaceWeather()
    drag = sidereon.DragParameters.from_bc_factor_m2_kg(0.05, weather)
    accel = sidereon.force_drag_acceleration(
        drag.to_force(), 0.0, [6500.0, 0.0, 0.0], [0.0, 7.8, 0.0]
    )
    assert accel.shape == (3,)
    assert np.isfinite(accel).all()
    estimate = sidereon.estimate_decay(
        0.0,
        [6450.0, 0.0, 0.0],
        [0.0, 7.8, 0.0],
        drag,
        max_duration_s=3600.0,
        max_scan_samples=8,
    )
    assert estimate.time_to_decay_s >= 0.0


def test_ephemeris_sample_returns_grid_rows_with_status():
    sp3 = _load_sp3(_SP3_FILE)
    start = float(sp3.epochs_j2000_seconds[0])
    rows = sidereon.ephemeris_sample(sp3, ["G21"], start, start + 900.0, 900.0)
    assert [row.status for row in rows] == [
        sidereon.EphemerisSampleStatus.VALID,
        sidereon.EphemerisSampleStatus.VALID,
    ]
    assert rows[0].position_ecef_m.shape == (3,)


def test_dted_terrain_uses_core_tile_reader():
    root = pathlib.Path(CORE_FIXTURES) / "dted" / "tiles"
    terrain = sidereon.DtedTerrain(root)
    tile = sidereon.DtedTile.from_path(root / "n36_w107_1arc_v3.dt2")
    opts = sidereon.DtedLookupOptions(sidereon.DtedInterpolation.NEAREST_POSTING)
    lon = hex_to_f64("0xc05ac00000000000")
    lat = hex_to_f64("0x4042000000000000")
    assert terrain.height_m(lat, lon, opts) == pytest.approx(
        hex_to_f64("0xc034000000000000")
    )
    assert tile.height_m(lat, lon) == pytest.approx(hex_to_f64("0xc034000000000000"))


def test_bias_sinex_and_code_dcb_parsers_expose_bias_sets():
    bias_path = (
        pathlib.Path(CORE_FIXTURES) / "bias" / "COD0OPSFIN_20261330000_01D_01D_OSB.BIA"
    )
    with open(bias_path, "rb") as fh:
        bias = sidereon.parse_bias_sinex(fh.read())
    assert bias.record_count > 0
    assert bias.records[0].kind in {"OSB", "DSB", "ISB"}
    assert sidereon.load_bias_sinex(bias_path).record_count == bias.record_count
    assert (
        sidereon.load_bias_sinex_lossy(bias_path).value.record_count
        == bias.record_count
    )
    opts = sidereon.PppCodeBiasOptions(
        bias,
        [(sidereon.GnssSystem.GPS, "C1C", "C2W")],
    )
    assert opts is not None

    dcb_path = pathlib.Path(CORE_FIXTURES) / "bias" / "P1C1_RINEX.DCB"
    with open(dcb_path, "rb") as fh:
        dcb = sidereon.parse_code_dcb(fh.read(), None)
    assert dcb.record_count > 0
    assert sidereon.load_code_dcb(dcb_path, None).record_count == dcb.record_count
    assert (
        sidereon.load_code_dcb_lossy(dcb_path, None).value.record_count
        == dcb.record_count
    )


def test_sbas_decode_parse_store_and_mapping_helpers():
    body_hex = "5306000000000000000000000000000000000000000000000000000040"
    body = bytes.fromhex(body_hex)
    block = sidereon.decode_sbas_block(body, sidereon.SbasWireForm.BODY226)
    alias = sidereon.decode_sbas_message(body, sidereon.SbasWireForm.BODY226)
    assert block.message_type == 1
    assert block.kind == sidereon.SbasMessageKind.PRN_MASK
    assert alias.kind == block.kind
    assert block.kind_label == "prn_mask"
    assert block.encode() == body

    parsed = sidereon.parse_sbas_rtklib_lines(f"2360 259200 120 1 : {body_hex}\n")
    assert len(parsed) == 1
    assert parsed[0].satellite_id == "S20"
    store = sidereon.SbasCorrectionStore()
    store.ingest(block, "S20", 2360, 259200.0)
    assert sidereon.sbas_prn_to_satellite_id(120) == "S20"
    assert sidereon.satellite_id_to_sbas_prn("S20") == 120


def test_ssr_decode_store_and_correction_queries():
    assert sidereon.SsrSource.GALILEO_HAS.label == "galileo_has"
    with open(
        os.path.join(CORE_FIXTURES, "ssr", "SSRA02IGS0_2026181234930_1060.hex")
    ) as fh:
        frame = bytes.fromhex(fh.read())
    rtcm = sidereon.decode_rtcm(frame)[0]
    assert rtcm.kind == "ssr"
    ssr = sidereon.decode_ssr_message(rtcm.encode())
    ssr_alias = sidereon.decode_ssr(rtcm.encode())
    assert ssr.message_number == 1060
    assert ssr_alias.message_number == ssr.message_number
    assert ssr.kind == sidereon.SsrKind.COMBINED_ORBIT_CLOCK
    assert ssr.satellite_count == 2
    store = sidereon.SsrCorrectionStore()
    store.ingest_ssr(ssr, 2425, 344970.0)
    orbit = store.orbit("G30")
    clock = store.clock("G30")
    assert orbit is not None
    assert clock is not None
    assert orbit.solution.source == sidereon.SsrSource.RTCM_SSR
    assert clock.solution.source == sidereon.SsrSource.RTCM_SSR
    assert orbit.solution.provider_id == ssr.provider_id
    assert clock.solution.solution_id == ssr.solution_id

    ingest = sidereon.ssr_store_from_rtcm(frame, 2425, 344970.0)
    assert ingest.is_complete
    assert ingest.trailing_partial_frame_len == 0
    assert ingest.ingest_refusals == []
    assert ingest.diagnostics.resync_bytes == 0
    assert ingest.store.orbit("G30") is not None
    assert ingest.store.clock("G30") is not None
    strict = sidereon.ssr_store_from_rtcm_strict(frame, 2425, 344970.0)
    assert strict.orbit("G30") is not None
    assert strict.clock("G30") is not None
    with pytest.raises(sidereon.RtcmParseError):
        sidereon.ssr_store_from_rtcm_strict(b"\x00" + frame, 2425, 344970.0)
    noisy = sidereon.ssr_store_from_rtcm(b"\x00" + frame, 2425, 344970.0)
    assert not noisy.is_complete
    assert noisy.diagnostics.resync_bytes == 1
    assert noisy.store.orbit("G30") is not None


def test_ssr_com_orbit_requires_public_antenna_and_attitude_configuration():
    frame_path = (
        pathlib.Path(CORE_FIXTURES) / "ssr" / "SSRA02IGS0_2026181234930_1060.hex"
    )
    frame = bytes.fromhex(frame_path.read_text(encoding="ascii"))
    rtcm = sidereon.decode_rtcm(frame)[0]
    message = sidereon.decode_ssr_message(rtcm.encode())
    store = sidereon.SsrCorrectionStore(
        reference_point=sidereon.OrbitReferencePoint.CENTER_OF_MASS
    )
    store.ingest_ssr(message, 2425, 344_970.0)
    lenient_ingest = sidereon.ssr_store_from_rtcm(
        frame,
        2425,
        344_970.0,
        reference_point=sidereon.OrbitReferencePoint.CENTER_OF_MASS,
    )
    assert lenient_ingest.is_complete
    assert (
        lenient_ingest.store.orbit("G30").reference_point
        == sidereon.OrbitReferencePoint.CENTER_OF_MASS
    )
    strict_store = sidereon.ssr_store_from_rtcm_strict(
        frame,
        2425,
        344_970.0,
        reference_point=sidereon.OrbitReferencePoint.CENTER_OF_MASS,
    )
    assert (
        strict_store.orbit("G30").reference_point
        == sidereon.OrbitReferencePoint.CENTER_OF_MASS
    )
    broadcast = sidereon.load_rinex_nav(
        pathlib.Path(CORE_FIXTURES) / "ssr" / "BRDC00WRD_S_20261820000_G30_G31.rnx"
    )
    antex = sidereon.load_antex(
        pathlib.Path(__file__).parent / "fixtures" / "antex" / "igs20_wettzell_trim.atx"
    )
    epoch = 2425 * 604_800.0 + 344_970.0 - 630_763_200.0

    unavailable = sidereon.SsrCorrectedEphemeris(
        broadcast,
        store,
        max_staleness_s=60.0,
        satellite_antennas=antex,
        satellite_attitude=sidereon.SsrSatelliteAttitude.UNAVAILABLE,
    )
    assert unavailable.corrected_state_checked("G30", epoch)[0] is None

    nominal = sidereon.SsrCorrectedEphemeris(
        broadcast,
        store,
        max_staleness_s=60.0,
        satellite_antennas=antex,
        satellite_attitude=sidereon.SsrSatelliteAttitude.NOMINAL_SUN_FIXED,
    )
    checked, degraded = nominal.corrected_state_checked("G30", epoch)
    assert checked is not None
    assert degraded is None
    position, clock_s = checked
    assert position == pytest.approx(
        [-6_327_381.448161609, 15_802_128.916795386, -20_121_896.861226305],
        abs=1.0e-6,
    )
    assert clock_s == pytest.approx(0.0002800865527753679, abs=1.0e-15)
    assert nominal.position_clock_at_j2000_s("G30", epoch) == checked

    query = sidereon.ExactEpochQuery.from_binary_j2000_seconds(epoch)
    queried, query_degraded = nominal.corrected_state_at_epoch_query(
        "G30", query, query
    )
    assert queried == checked
    assert query_degraded is None

    apc_store = sidereon.ssr_store_from_rtcm(frame, 2425, 344_970.0)
    apc = sidereon.SsrCorrectedEphemeris(
        broadcast,
        apc_store.store,
        max_staleness_s=60.0,
        satellite_antennas=antex,
        satellite_attitude=sidereon.SsrSatelliteAttitude.NOMINAL_SUN_FIXED,
    )
    apc_checked, apc_degraded = apc.corrected_state_checked("G30", epoch)
    assert apc_checked is not None
    assert apc_degraded is None
    assert apc_checked[0] != position

    malformed_cases = [
        (b"\x00" + frame, 1, 0, 0, 0),
        (frame[:-1] + bytes([frame[-1] ^ 1]), len(frame), 1, 0, 0),
        (frame + b"\xd3\x00", 2, 0, 2, 0),
    ]
    for payload, resync, crc_failures, trailing_len, refusal_count in malformed_cases:
        apc_ingest = sidereon.ssr_store_from_rtcm(payload, 2425, 344_970.0)
        com_ingest = sidereon.ssr_store_from_rtcm(
            payload,
            2425,
            344_970.0,
            reference_point=sidereon.OrbitReferencePoint.CENTER_OF_MASS,
        )
        for ingest in (apc_ingest, com_ingest):
            assert ingest.diagnostics.resync_bytes == resync
            assert ingest.diagnostics.crc_failures == crc_failures
            assert ingest.trailing_partial_frame_len == trailing_len
            assert len(ingest.ingest_refusals) == refusal_count
        if resync or crc_failures or trailing_len:
            with pytest.raises(sidereon.RtcmParseError) as apc_error:
                sidereon.ssr_store_from_rtcm_strict(payload, 2425, 344_970.0)
            with pytest.raises(sidereon.RtcmParseError) as com_error:
                sidereon.ssr_store_from_rtcm_strict(
                    payload,
                    2425,
                    344_970.0,
                    reference_point=sidereon.OrbitReferencePoint.CENTER_OF_MASS,
                )
            assert str(apc_error.value) == str(com_error.value)

    invalid_satellite_body = bytearray(sidereon.decode_rtcm(frame)[0].encode())
    for bit in range(68, 74):
        index, offset = divmod(bit, 8)
        invalid_satellite_body[index] &= ~(1 << (7 - offset))
    refused_payload = sidereon.encode_rtcm_frame(bytes(invalid_satellite_body))
    apc_refused = sidereon.ssr_store_from_rtcm(refused_payload, 2425, 344_970.0)
    com_refused = sidereon.ssr_store_from_rtcm(
        refused_payload,
        2425,
        344_970.0,
        reference_point=sidereon.OrbitReferencePoint.CENTER_OF_MASS,
    )
    assert len(apc_refused.ingest_refusals) == len(com_refused.ingest_refusals) == 1
    assert apc_refused.ingest_refusals[0].message_number == 1060
    assert com_refused.ingest_refusals[0].message_number == 1060
    with pytest.raises(sidereon.RtcmParseError) as apc_error:
        sidereon.ssr_store_from_rtcm_strict(refused_payload, 2425, 344_970.0)
    with pytest.raises(sidereon.RtcmParseError) as com_error:
        sidereon.ssr_store_from_rtcm_strict(
            refused_payload,
            2425,
            344_970.0,
            reference_point=sidereon.OrbitReferencePoint.CENTER_OF_MASS,
        )
    assert str(apc_error.value) == str(com_error.value)


def test_solve_spp_robust_fde_returns_fde_result_and_alias_warns():
    sp3 = _load_sp3(_SP3_FILE)
    trace_path = os.path.join(CORE_FIXTURES, "spp_trace_L0_minimal.json")
    with open(trace_path) as handle:
        fixture = json.load(handle)["fixture"]
    inputs = fixture["inputs"]
    consistent = core_goldens()["spp_consistent"]
    observations = [
        sidereon.SppObservation(satellite, hex_to_f64(pseudorange))
        for satellite, pseudorange in zip(
            consistent["satellites"], consistent["pseudoranges_m"]
        )
    ]
    cfg = sidereon.SppConfig(
        observations=observations,
        t_rx_j2000_s=hex_to_f64(inputs["t_rx_j2000_s"]),
        t_rx_second_of_day_s=hex_to_f64(inputs["t_rx_sod_s"]),
        day_of_year=hex_to_f64(inputs["doy"]),
        initial_guess=[
            hex_to_f64(value) for value in fixture["frozen"]["initial_guess_x0"]
        ],
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
        klobuchar=sidereon.SppKlobucharCoeffs(
            alpha=[hex_to_f64(value) for value in inputs["klobuchar_alpha"]],
            beta=[hex_to_f64(value) for value in inputs["klobuchar_beta"]],
        ),
        met=sidereon.SppSurfaceMet(
            pressure_hpa=hex_to_f64(inputs["met"]["pressure_hpa"]),
            temperature_k=hex_to_f64(inputs["met"]["temperature_k"]),
            relative_humidity=hex_to_f64(inputs["met"]["relative_humidity"]),
        ),
        with_geodetic=True,
        qzss_clock=sidereon.QzssClock.GPS,
        troposphere_model=sidereon.TroposphereModel.RTKLIB,
    )
    result = sidereon.solve_spp_robust_fde(
        sp3, cfg, sidereon.SppRobustConfig(), 0.01, 2
    )
    assert result.iterations >= 0
    assert len(result.used_sats) >= 4
    with pytest.warns(DeprecationWarning):
        alias = sidereon.spp_robust_fde_driver(
            sp3, cfg, sidereon.SppRobustConfig(), 0.01, 2
        )
    assert alias.iterations == result.iterations


# --- shared helpers --------------------------------------------------------


def _unix_us(year, month, day):
    stamp = dt.datetime(year, month, day, tzinfo=dt.timezone.utc)
    return int(stamp.timestamp() * 1_000_000)


def _load_sp3(name):
    with open(os.path.join(CORE_FIXTURES, "sp3", name), "rb") as fh:
        return sidereon.load_sp3(fh.read())


def _read_const(name):
    with open(os.path.join(CORE_FIXTURES, "constellation", name)) as fh:
        return fh.read()


def _geodetic_to_ecef(lat_deg, lon_deg, h_m):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    return np.array(
        [
            (n + h_m) * np.cos(lat) * np.cos(lon),
            (n + h_m) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - e2) + h_m) * np.sin(lat),
        ]
    )


def _glonass_scenario(sp3):
    epoch_index = 48
    t_rx = float(sp3.epochs_j2000_seconds[epoch_index])
    rx = _geodetic_to_ecef(55.75, 37.62, 200.0)
    up = rx / np.linalg.norm(rx)
    observations = []
    for sat in sp3.satellites:
        if not sat.startswith("R"):
            continue
        interp = sp3.interpolate(sat, np.array([t_rx]))
        pos = interp.position_m[0]
        dt_sat = float(interp.clock_s[0])
        if not np.isfinite(pos).all() or not np.isfinite(dt_sat):
            continue
        los = pos - rx
        rng = float(np.linalg.norm(los))
        el_deg = np.degrees(np.arcsin(float(np.dot(los, up)) / rng))
        if el_deg < 10.0:
            continue
        observations.append(sidereon.SppObservation(sat, rng - C_M_S * dt_sat))
    return rx, observations, t_rx
