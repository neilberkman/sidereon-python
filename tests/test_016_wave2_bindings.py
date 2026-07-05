import calendar
import datetime as dt
import re
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import sidereon

FIXTURES = Path(__file__).with_name("fixtures")
REPO = Path(__file__).resolve().parents[1]


def test_016_geodesic_inverse_matches_vendored_geodtest_row():
    row = [
        float(token)
        for token in (FIXTURES / "geodesic" / "geodtest_row.dat").read_text().split()
    ]

    distance_m, azi1_deg, azi2_deg = sidereon.geodesic_inverse(
        row[0], row[1], row[3], row[4]
    )
    assert distance_m == pytest.approx(row[6], abs=2.0e-9)
    assert azi1_deg == pytest.approx(row[2], abs=1.0e-12)
    assert azi2_deg == pytest.approx(row[5], abs=1.0e-12)

    lat2_deg, lon2_deg, direct_azi2_deg = sidereon.geodesic_direct(
        row[0], row[1], row[2], row[6]
    )
    assert lat2_deg == pytest.approx(row[3], abs=3.0e-14)
    assert lon2_deg == pytest.approx(row[4], abs=3.0e-14)
    assert direct_azi2_deg == pytest.approx(row[5], abs=1.0e-12)


def test_016_frame_catalog_transform_matches_core_bits():
    position = np.asarray([4_000_000.0, 1_000_000.0, 4_800_000.0], dtype=np.float64)
    velocity = np.asarray([0.012, -0.005, 0.002], dtype=np.float64)

    state = sidereon.frame_catalog_transform(
        position,
        sidereon.TerrestrialFrame.ITRF2020,
        "ITRF2014",
        2026.25,
        velocity_m_per_year=velocity,
    )

    assert state.position_m.tobytes() == bytes.fromhex(
        "16139bff7f844e416987bffe7f842e417ec51a00804f5241"
    )
    assert state.velocity_m_per_year is not None
    assert state.velocity_m_per_year.tobytes() == bytes.fromhex(
        "fa7e6abc7493883f88855ad3bce374bf2f6ea301bc05623f"
    )

    entry = sidereon.frame_catalog_entry("ITRF2020", "ITRF2014")
    assert entry is not None
    assert entry.reference_epoch_year == 2015.0
    params = entry.parameters_at(2026.25)
    assert (
        params.translation_mm.tobytes()
        == np.asarray([-1.4, -2.025, 3.65], dtype=np.float64).tobytes()
    )


def test_016_egm2008_crop_undulations_match_core_bits():
    window = sidereon.Egm2008RasterWindow(
        sidereon.Egm2008GridSpacing.TWO_POINT_FIVE_MINUTE,
        37.0,
        -123.0,
        25,
        25,
    )
    grid = sidereon.GeoidGrid.from_egm2008_raster_window(
        (FIXTURES / "geoid" / "egm2008_25_norcal_crop.bin").read_bytes(),
        window,
    )
    points = np.asarray(
        [
            [37.7749, -122.4194],
            [37.5, -122.75],
            [37.875, -122.125],
            [38.0, -122.0],
            [37.0, -123.0],
        ],
        dtype=np.float64,
    )

    got = grid.undulations_deg(points)
    assert got.tobytes() == bytes.fromhex(
        "232a127bef1440c0000000c08ccd40c000000040edd83fc"
        "00000006091c43fc000000060eb3f42c0"
    )
    assert grid.undulation_deg(37.7749, -122.4194).hex() == ("-0x1.014ef7b122a23p+5")


def test_016_tdm_annex_round_trips_canonical_kvn():
    text = (FIXTURES / "tdm" / "annex_e_01.kvn").read_text()
    message = sidereon.parse_tdm_kvn(text)

    assert len(message.segments) == 1
    records = message.segments[0].data.records
    assert len(records) == 31
    assert records[0].keyword == "TRANSMIT_FREQ_2"
    assert records[0].value_text == "32023442781.733"
    assert records[0].unit.label == "Hz"

    encoded = message.to_kvn_string()
    reparsed = sidereon.parse_tdm_kvn(encoded)
    assert reparsed.to_kvn_string() == encoded


def test_016_ecef_sp3_precise_orbit_fit_variants_smoke():
    sp3 = sidereon.load_sp3(
        (FIXTURES / "sp3" / "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3").read_bytes()
    )
    options = sidereon.OrbitFitOptions(
        force_model=sidereon.ForceModelKind.two_body(),
        min_ledger_samples=3,
        max_steps=200_000,
    )

    report = sidereon.fit_sp3_ecef_precise_orbit(sp3, "G01", options)
    fit = report.fits[0]
    stats = report.ledger.per_sat[0][1]
    assert report.fit_count == 1
    assert fit.satellite == "G01"
    # Re-pinned after the core parsed-epoch-axis hardening.
    assert fit.fit_rms_3d_m == pytest.approx(139.667980698503, abs=1.0e-9)
    assert stats.rms_3d_m == pytest.approx(139.667980698503, abs=1.0e-9)
    assert fit.covariance.kind == "estimated"
    assert stats.n == 11
    assert stats.low_sample_count is False

    selected = sidereon.fit_sp3_ecef_precise_orbits(sp3, ["G01", "G02"], options)
    assert selected.fit_count == 2
    assert [entry[0] for entry in selected.ledger.per_sat] == ["G01", "G02"]

    all_fits = sidereon.fit_all_sp3_ecef_precise_orbits(sp3, options)
    assert all_fits.fit_count == 31
    assert len(all_fits.ledger.per_sat) == 31
    assert all_fits.ledger.per_sat[0][0] == "G01"


def test_016_spherical_harmonic_force_selection_matches_explicit_composite():
    position = np.asarray([7078.0, -30.0, 820.0], dtype=np.float64)
    velocity = np.asarray([0.2, 7.35, 1.05], dtype=np.float64)
    times = np.asarray([0.0, 120.0, 240.0], dtype=np.float64)
    kwargs = dict(
        integrator=sidereon.Integrator.RK4,
        initial_step_s=30.0,
        max_step_s=30.0,
        abs_tol=1.0e-9,
        rel_tol=1.0e-12,
    )

    phase_b = sidereon.propagate_state(
        0.0,
        position,
        velocity,
        times,
        force_model=sidereon.ForceModelKind.earth_phase_b(2, 0),
        **kwargs,
    )
    components = sidereon.ForceModelComponents(
        two_body_mu_km3_s2=398600.4418,
        spherical_harmonic_max_degree=2,
        spherical_harmonic_max_order=0,
        third_body=True,
        relativity=True,
    )
    explicit = sidereon.propagate_state(
        0.0,
        position,
        velocity,
        times,
        force_model=sidereon.ForceModelKind.composite(components),
        **kwargs,
    )

    assert components.spherical_harmonic_max_degree == 2
    assert components.spherical_harmonic_max_order == 0
    assert phase_b.states.tobytes() == explicit.states.tobytes()


def test_016_sgp4_decay_latch_preserves_first_decay_epoch():
    line1 = "1 28872U 05037B   05333.02012661  .25992681  00000-0  24476-3 0  1534"
    line2 = "2 28872  96.4736 157.9986 0303955 244.0492 110.6523 16.46015938 10708"
    tle = sidereon.Tle(line1, line2)
    offset_us = int((Decimal("333.02012661") - Decimal(1)) * Decimal(86_400_000_000))
    epoch = dt.datetime(2005, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        microseconds=offset_us
    )

    def unix_us(value):
        return calendar.timegm(value.utctimetuple()) * 1_000_000 + value.microsecond

    good = np.asarray([unix_us(epoch + dt.timedelta(minutes=120))], dtype=np.int64)
    decay = np.asarray([unix_us(epoch + dt.timedelta(minutes=1440))], dtype=np.int64)
    later = np.asarray([unix_us(epoch + dt.timedelta(minutes=1450))], dtype=np.int64)

    latched_good = tle.propagate_with_decay_latch(good, sidereon.DecayLatch())
    raw_good = tle.propagate(good)
    assert latched_good.position_km.tobytes() == raw_good.position_km.tobytes()
    assert latched_good.velocity_km_s.tobytes() == raw_good.velocity_km_s.tobytes()
    assert np.isfinite(tle.propagate(later).position_km).all()

    latch = sidereon.DecayLatch()
    with pytest.raises(sidereon.SolveError, match="first failed"):
        tle.propagate_with_decay_latch(decay, latch)
    assert latch.first_failing_minutes_since_epoch == 1440.0
    with pytest.raises(sidereon.SolveError, match="first failed"):
        tle.propagate_with_decay_latch(later, latch)


def test_016_troposphere_low_elevation_error_and_oblate_eclipse_bits():
    with pytest.raises(ValueError, match="below mapping validity"):
        sidereon.tropo_mapping_factors(np.deg2rad(1.0), 0.5, 100.0, 0)

    satellite = np.asarray([[-7000.0, 0.0, 6370.0]], dtype=np.float64)
    sun = np.asarray([[149597870.7, 0.0, 0.0]], dtype=np.float64)
    default = sidereon.shadow_fraction(satellite, sun)
    spherical = sidereon.shadow_fraction_with_model(
        satellite, sun, sidereon.EarthShadowModel.SPHERICAL
    )
    oblate = sidereon.shadow_fraction_with_model(
        satellite, sun, sidereon.EarthShadowModel.WGS84_OBLATE
    )

    assert spherical.tobytes() == default.tobytes()
    assert spherical.tobytes() == bytes.fromhex("1ffbe7082fa3e03f")
    assert oblate.tobytes() == bytes.fromhex("982b27487675c83f")


def test_016_wtest_noncentrality_uses_core_delta_and_manifest_has_no_path_deps():
    constants = sidereon.wtest_noncentrality(0.001, 0.80)
    assert constants.delta0.hex() == "0x1.08751cbd0bec7p+2"
    assert constants.lambda0.hex() == "0x1.1131c0d9309e7p+4"
    assert "lambda0.sqrt" not in (REPO / "src" / "reliability.rs").read_text()

    manifest = (REPO / "Cargo.toml").read_text()
    assert re.search(r'sidereon = "\d+\.\d+\.\d+"', manifest)
    assert re.search(r'sidereon-core = "\d+\.\d+\.\d+"', manifest)
    assert "path =" not in manifest
