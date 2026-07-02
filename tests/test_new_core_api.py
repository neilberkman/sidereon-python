"""Smoke tests for the multi-system / PPP-correction core API surface.

These exercise the headline new core entry points the binding wraps as the first
real consumer of the post-campaign core: inter-system time-scale offsets
(A1), per-system TDOP (A3), the multi-system constellation catalog (A2), the PPP
correction options pole tide / ocean loading / VMF1 mapping (B1), and the SP3
merge agreement metrics (B2). The numbers themselves are the core's; these only
confirm the binding marshals the new shapes through faithfully.
"""

import datetime as dt
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, hex_to_f64

C_M_S = 299792458.0


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
    eq_from_rv = sidereon.rv2eq(r, v, 398600.4418)
    r2, v2 = sidereon.eq2rv(eq_from_rv, 398600.4418)
    np.testing.assert_allclose(r2, r, rtol=1e-10, atol=1e-7)
    np.testing.assert_allclose(v2, v, rtol=1e-10, atol=1e-10)


def test_angles_beta_and_relative_frames_round_trip():
    assert sidereon.angular_separation(
        np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    ) == pytest.approx(90.0)
    assert sidereon.position_angle(0.0, 0.0, np.pi / 2.0, 0.0) == pytest.approx(90.0)
    beta = sidereon.beta_angle_from_state(
        np.array([7000.0, 0.0, 0.0]),
        np.array([0.0, 7.5, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    )
    assert beta == pytest.approx(90.0)

    chief = sidereon.CartesianState(
        0.0, np.array([7000.0, 0.0, 0.0]), np.array([0.0, 7.5, 0.0])
    )
    deputy = sidereon.CartesianState(
        0.0, np.array([7001.0, 0.0, 0.0]), np.array([0.0, 7.501, 0.0])
    )
    rel = sidereon.relative_state(chief, deputy)
    rebuilt = sidereon.absolute_from_relative(chief, rel)
    np.testing.assert_allclose(rebuilt.position_km, deputy.position_km, atol=1e-12)
    assert sidereon.rtn_to_inertial_rotation(chief).shape == (3, 3)
    assert sidereon.cw_stm(sidereon.mean_motion_from_state(chief), 60.0).shape == (6, 6)


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
    root = os.path.join(CORE_FIXTURES, "dted", "tiles")
    terrain = sidereon.DtedTerrain(root)
    opts = sidereon.DtedLookupOptions(sidereon.DtedInterpolation.NEAREST_POSTING)
    lon = hex_to_f64("0xc05ac00000000000")
    lat = hex_to_f64("0x4042000000000000")
    assert terrain.height_m(lon, lat, opts) == pytest.approx(
        hex_to_f64("0xc034000000000000")
    )


def test_bias_sinex_and_code_dcb_parsers_expose_bias_sets():
    with open(
        os.path.join(CORE_FIXTURES, "bias", "COD0OPSFIN_20261330000_01D_01D_OSB.BIA"),
        "rb",
    ) as fh:
        bias = sidereon.parse_bias_sinex(fh.read())
    assert bias.record_count > 0
    assert bias.records[0].kind in {"OSB", "DSB", "ISB"}
    opts = sidereon.PppCodeBiasOptions(
        bias,
        [(sidereon.GnssSystem.GPS, "C1C", "C2W")],
    )
    assert opts is not None

    with open(os.path.join(CORE_FIXTURES, "bias", "P1C1_RINEX.DCB"), "rb") as fh:
        dcb = sidereon.parse_code_dcb(fh.read(), None)
    assert dcb.record_count > 0


def test_sbas_decode_parse_store_and_mapping_helpers():
    body_hex = "5306000000000000000000000000000000000000000000000000000040"
    body = bytes.fromhex(body_hex)
    block = sidereon.decode_sbas_block(body, sidereon.SbasWireForm.BODY226)
    assert block.message_type == 1
    assert block.kind == "prn_mask"
    assert block.encode() == body

    parsed = sidereon.parse_sbas_rtklib_lines(f"2360 259200 120 1 : {body_hex}\n")
    assert len(parsed) == 1
    assert parsed[0].satellite_id == "S20"
    store = sidereon.SbasCorrectionStore()
    store.ingest(block, "S20", 2360, 259200.0)
    assert sidereon.sbas_prn_to_satellite_id(120) == "S20"
    assert sidereon.satellite_id_to_sbas_prn("S20") == 120


def test_ssr_decode_store_and_correction_queries():
    with open(
        os.path.join(CORE_FIXTURES, "ssr", "SSRA02IGS0_2026181234930_1060.hex")
    ) as fh:
        frame = bytes.fromhex(fh.read())
    rtcm = sidereon.decode_rtcm(frame)[0]
    assert rtcm.kind == "ssr"
    ssr = sidereon.decode_ssr_message(rtcm.encode())
    assert ssr.message_number == 1060
    assert ssr.kind == sidereon.SsrKind.COMBINED_ORBIT_CLOCK
    assert ssr.satellite_count == 2
    store = sidereon.SsrCorrectionStore()
    store.ingest_ssr(ssr, 2425, 344970.0)
    assert store.orbit("G30") is not None
    assert store.clock("G30") is not None


def test_spp_robust_fde_driver_returns_fde_result():
    sp3 = _load_sp3(_SP3_FILE)
    _rx, observations, t_rx = _glonass_scenario(sp3)
    cfg = sidereon.SppConfig(
        observations=observations,
        t_rx_j2000_s=t_rx,
        t_rx_second_of_day_s=0.0,
        day_of_year=176.0,
        initial_guess=[6378137.0, 0.0, 0.0, 0.0],
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
        with_geodetic=True,
    )
    result = sidereon.spp_robust_fde_driver(
        sp3, cfg, sidereon.SppRobustConfig(), 0.01, 2
    )
    assert result.iterations >= 0
    assert len(result.used_sats) >= 4


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
