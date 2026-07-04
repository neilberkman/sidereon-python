"""0.12 binding parity tests.

Provenance:
- Clock-stability checks use NIST SP 1065 section 12.4 Table 31 values.
- ARAIM checks use the WG-C Reference ADD v3.0 Appendix D public example.
- IONEX, DTED, and SBAS checks reuse the sidereon-core public fixtures and
  message examples bundled with the core test suite.
"""

import math
import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, hex_to_f64


def _bits_equal(a, b):
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_clock_stability_nist_table31_estimators_and_combined_driver():
    modulus = 2_147_483_647
    state = 1_234_567_890
    frequency = []
    for _ in range(1000):
        frequency.append(state / modulus)
        state = (16_807 * state) % modulus

    series = sidereon.AllanSeries.fractional_frequency(
        np.asarray(frequency, dtype=np.float64)
    )
    factors = [1, 10, 100]
    tau = np.asarray([1.0, 10.0, 100.0], dtype=np.float64)

    adev = sidereon.allan_deviation(series, 1.0, factors)
    oadev = sidereon.overlapping_adev(series, 1.0, factors)
    mdev = sidereon.modified_adev(series, 1.0, factors)
    tdev = sidereon.time_deviation(series, 1.0, factors)
    hdev = sidereon.hadamard_deviation(series, 1.0, factors)

    assert _bits_equal(adev.tau_s, tau)
    assert adev.n == [999, 99, 9]
    np.testing.assert_allclose(
        adev.deviation, [2.922319e-1, 9.965736e-2, 3.897804e-2], atol=5e-8
    )
    assert oadev.n == [999, 981, 801]
    np.testing.assert_allclose(
        oadev.deviation, [2.922319e-1, 9.159953e-2, 3.241343e-2], atol=5e-8
    )
    assert mdev.n == [999, 972, 702]
    np.testing.assert_allclose(
        mdev.deviation, [2.922319e-1, 6.172376e-2, 2.170921e-2], atol=5e-8
    )
    assert tdev.n == [999, 972, 702]
    np.testing.assert_allclose(
        tdev.deviation, [1.687202e-1, 3.563623e-1, 1.253382], atol=5e-7
    )
    assert hdev.n == [998, 971, 701]
    np.testing.assert_allclose(
        hdev.deviation, [2.943883e-1, 9.581083e-2, 3.237638e-2], atol=5e-8
    )

    options = sidereon.AllanOptions(
        sidereon.AllanEstimatorSet.all(),
        sidereon.TauGrid.explicit(factors),
        sidereon.GapPolicy.REJECT,
    )
    combined = sidereon.compute_allan_deviations(
        sidereon.AllanInput(series, 1.0, options)
    )
    assert _bits_equal(combined.overlapping_adev.deviation, oadev.deviation)


def test_dted_height_batch_matches_single_point_orthometric_lookup():
    root = os.path.join(CORE_FIXTURES, "dted", "tiles")
    terrain = sidereon.DtedTerrain(root)
    opts = sidereon.DtedLookupOptions(sidereon.DtedInterpolation.NEAREST_POSTING)
    lon = hex_to_f64("0xc05ac00000000000")
    lat = hex_to_f64("0x4042000000000000")
    points = np.asarray([[lon, lat], [lon + 1.0 / 3600.0, lat]], dtype=np.float64)

    batch = terrain.height_batch(points, opts)
    singles = np.asarray(
        [terrain.height_m(point[1], point[0], opts) for point in points],
        dtype=np.float64,
    )
    assert _bits_equal(batch, singles)
    assert batch[0] == pytest.approx(hex_to_f64("0xc034000000000000"))


def test_ionex_sample_sources_round_trip_grid_bytes_and_slant_delay():
    path = os.path.join(CORE_FIXTURES, "ionex", "synthetic_2map_7x7.20i")
    parsed = sidereon.load_ionex(path)

    grid_samples = parsed.tec_grid_samples()
    from_grid = sidereon.Ionex.from_samples(grid_samples)
    from_nodes = sidereon.Ionex.from_node_samples(
        parsed.tec_samples(),
        parsed.shell_height_km,
        parsed.base_radius_km,
        parsed.exponent,
    )

    assert _bits_equal(from_grid.tec_maps, parsed.tec_maps)
    assert _bits_equal(from_nodes.tec_maps, parsed.tec_maps)
    np.testing.assert_array_equal(
        from_grid.map_epochs_j2000_s, parsed.map_epochs_j2000_s
    )

    args = (
        -75.0,
        -35.0,
        40.0,
        35.0,
        int(parsed.map_epochs_j2000_s[0]),
        1_575_420_000.0,
    )
    assert from_grid.slant_delay(*args) == parsed.slant_delay(*args)
    assert from_nodes.slant_delay(*args) == parsed.slant_delay(*args)


def test_sbas_decoded_payload_and_store_accessors():
    body_hex = "5306000000000000000000000000000000000000000000000000000040"
    body = bytes.fromhex(body_hex)
    block = sidereon.decode_sbas_block(body, sidereon.SbasWireForm.BODY226)

    message = block.message
    assert message.kind == sidereon.SbasMessageKind.PRN_MASK
    assert message.prn_mask.iodp == 1
    assert message.prn_mask.mask[0] is True
    assert message.prn_mask.mask[1] is False
    assert block.encode() == body

    parsed = sidereon.parse_sbas_rtklib_lines(f"2360 259200 120 1 : {body_hex}\n")
    assert parsed[0].decode().message.prn_mask.mask == message.prn_mask.mask
    store = sidereon.SbasCorrectionStore()
    store.ingest(block, "S20", 2360, 259200.0)
    assert sidereon.sbas_prn_to_satellite_id(120) == "S20"
    assert sidereon.satellite_id_to_sbas_prn("S20") == 120


def test_araim_wgc_add_v3_public_example_protection_levels():
    def row(system, prn, design_enu, c_acc_m2):
        sigma_ure_m = 0.5
        variance_a_m = 0.3
        variance_b_m = 0.3
        local_variance_m2 = c_acc_m2 - sigma_ure_m * sigma_ure_m
        scaled = local_variance_m2 - variance_a_m * variance_a_m
        elevation_rad = math.asin(math.sqrt(variance_b_m * variance_b_m / scaled))
        los = [-design_enu[2], -design_enu[0], -design_enu[1]]
        return sidereon.AraimRow(
            f"{'G' if system is sidereon.GnssSystem.GPS else 'E'}{prn:02d}",
            np.asarray(los, dtype=np.float64),
            elevation_rad,
            system,
        )

    rows = [
        (sidereon.GnssSystem.GPS, 1, [0.0225, 0.9951, -0.0966], 3.5740),
        (sidereon.GnssSystem.GPS, 2, [0.6750, -0.6900, -0.2612], 1.1252),
        (sidereon.GnssSystem.GPS, 3, [0.0723, -0.6601, -0.7477], 0.5479),
        (sidereon.GnssSystem.GPS, 4, [-0.9398, 0.2553, -0.2269], 1.3258),
        (sidereon.GnssSystem.GPS, 5, [-0.5907, -0.7539, -0.2877], 1.0104),
        (sidereon.GnssSystem.GALILEO, 1, [-0.3236, -0.0354, -0.9455], 0.5309),
        (sidereon.GnssSystem.GALILEO, 2, [-0.6748, 0.4356, -0.5957], 0.5838),
        (sidereon.GnssSystem.GALILEO, 3, [0.0938, -0.7004, -0.7075], 0.5544),
        (sidereon.GnssSystem.GALILEO, 4, [0.5571, 0.3088, -0.7709], 0.5448),
        (sidereon.GnssSystem.GALILEO, 5, [0.6622, 0.6958, -0.2780], 1.0491),
    ]
    geometry = sidereon.AraimGeometry(
        [row(*args) for args in rows],
        sidereon.Wgs84Geodetic(0.0, 0.0, 0.0),
        [sidereon.GnssSystem.GPS, sidereon.GnssSystem.GALILEO],
    )
    model = sidereon.SatelliteIsmModel(0.75, 0.5, 0.5, 1.0e-5)
    ism = sidereon.Ism(
        [
            sidereon.ConstellationIsm(sidereon.GnssSystem.GPS, 1.0e-4, model),
            sidereon.ConstellationIsm(sidereon.GnssSystem.GALILEO, 1.0e-4, model),
        ],
        [],
    )

    result = sidereon.araim(geometry, ism, sidereon.IntegrityAllocation.lpv_200())
    assert result.availability
    assert result.vpl_m == pytest.approx(19.2, abs=1.0)
    assert result.hpl_m == pytest.approx(14.5, abs=1.0)
    assert result.emt_m == pytest.approx(7.8, abs=1.0)
    assert result.sigma_acc_v_m == pytest.approx(1.47, abs=0.02)
    modes = sidereon.enumerate_fault_modes(
        geometry, ism, sidereon.IntegrityAllocation.lpv_200()
    )
    assert modes[0].excluded == []


def test_angles_lon_lat_degree_order_matches_public_reference_values():
    assert sidereon.angular_separation_coords(0.0, 0.0, 90.0, 0.0) == pytest.approx(
        90.0
    )
    assert sidereon.position_angle(0.0, 0.0, 90.0, 0.0) == pytest.approx(90.0)
    assert sidereon.position_angle(0.0, 0.0, 0.0, 90.0) == pytest.approx(0.0)
