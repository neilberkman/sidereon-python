"""0.17 domain exposure parity tests against patched core outputs."""

import json
import math
import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES, hex_to_f64

SP3_2020 = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")

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


def _f64(bits):
    return struct.unpack(">d", bits.to_bytes(8, "big"))[0]


def _bits(value):
    return int.from_bytes(struct.pack(">d", float(value)), "big")


def _array_bits(values):
    return np.asarray(values, dtype=np.float64).ravel().view(np.uint64)


def _expect_bits(hex_values):
    return np.asarray([int(value, 16) for value in hex_values], dtype=np.uint64)


def _load_sp3(path):
    with open(path, "rb") as handle:
        return sidereon.load_sp3(handle.read())


def _spp_fixture():
    with open(os.path.join(CORE_FIXTURES, "spp_trace_L0_minimal.json")) as handle:
        return json.load(handle)["fixture"]


def _spp_sp3_config(perturb_m=0.0):
    fixture = _spp_fixture()
    inputs = fixture["inputs"]
    sp3 = _load_sp3(os.path.join(CORE_FIXTURES, "sp3", inputs["sp3_file"]))
    observations = [
        sidereon.SppObservation(
            row["sat_id"],
            hex_to_f64(row["p_meas_m"]) + perturb_m,
        )
        for row in inputs["observations"]
    ]
    config = sidereon.SppConfig(
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
    )
    return sp3, config


def _point(lat_deg, lon_deg):
    return sidereon.Wgs84Geodetic(
        math.radians(lat_deg),
        math.radians(lon_deg),
        0.0,
    )


def test_geofence_probability_and_crossing_bits():
    fence = sidereon.Geofence(
        [
            _point(-0.01, -0.01),
            _point(-0.01, 0.01),
            _point(0.01, 0.01),
            _point(0.01, -0.01),
        ]
    )
    inside = _point(0.0, 0.0)
    outside = _point(0.0, 0.03)
    near = _point(0.0, 0.0095)
    uncertainty = sidereon.GeofencePositionUncertainty.enu_covariance_m2(
        np.diag([100.0, 225.0, 9.0])
    )
    quadrature = sidereon.GeofenceProbabilityOptions(
        sidereon.GeofenceProbabilityMethod.PLANAR_QUADRATURE
    )

    assert fence.vertex_count == 4
    assert fence.edge_count == 4
    assert fence.planar_fast_path_applies(inside) is True
    assert fence.contains(inside) is True
    assert fence.contains(outside) is False
    assert _bits(fence.distance_to_boundary(inside)) == 0x409146F89A157D9C
    assert _bits(fence.distance_to_boundary(outside)) == 0xC0A164C795F1FD1A
    assert _bits(fence.distance_to_boundary(near)) == 0x404BD47289804B58
    assert _bits(fence.containment_probability(near, uncertainty)) == (
        0x3FEFFFFFF9008B00
    )
    assert _bits(fence.containment_probability(near, uncertainty, quadrature)) == (
        0x3FEFFFFFF9008ADB
    )

    events = fence.crossing([outside, inside, near, outside])
    assert [(event.sample_index, event.kind.label) for event in events] == [
        (1, "entered"),
        (3, "left"),
    ]

    hysteresis = sidereon.GeofenceProbabilityHysteresis(0.8, 0.8)
    estimates = [
        sidereon.GeofencePositionEstimate(outside, uncertainty),
        sidereon.GeofencePositionEstimate(inside, uncertainty),
        sidereon.GeofencePositionEstimate(near, uncertainty),
        sidereon.GeofencePositionEstimate(outside, uncertainty),
    ]
    probability_events = fence.crossing_probability(estimates, hysteresis, quadrature)
    assert [
        (event.sample_index, event.kind, _bits(event.inside_probability))
        for event in probability_events
    ] == [
        (1, sidereon.GeofenceCrossingKind.ENTERED, 0x3FF0000000000000),
        (3, sidereon.GeofenceCrossingKind.LEFT, 0x0000000000000000),
    ]


def test_static_positioning_solution_bits():
    sp3, config0 = _spp_sp3_config()
    _, config1 = _spp_sp3_config(0.2)
    _, config2 = _spp_sp3_config(-0.1)
    epochs = [
        sidereon.StaticEpoch(config0),
        sidereon.StaticEpoch(config1),
        sidereon.StaticEpoch(config2),
    ]
    options = sidereon.StaticSolveOptions(
        initial_position_m=config0.initial_guess[:3],
        with_geodetic=True,
    )

    solution = sidereon.solve_static(sp3, epochs, options)

    assert np.array_equal(
        _array_bits(solution.position),
        _expect_bits(
            [
                "0x41511b07ff824402",
                "0x4120cd6b5f861f31",
                "0x41511e62229e1c2c",
            ]
        ),
    )
    assert [
        (index, system, _bits(clock_s))
        for index, system, clock_s in solution.per_epoch_clock
    ] == [
        (0, sidereon.GnssSystem.GPS, 0x3F1A3B884188E523),
        (1, sidereon.GnssSystem.GPS, 0x3F1A3B93B79873AE),
        (2, sidereon.GnssSystem.GPS, 0x3F1A3B8286811900),
    ]
    assert np.array_equal(
        _array_bits(solution.covariance.position_ecef_m2),
        _expect_bits(
            [
                "0x4000deb4f6cc217c",
                "0x3fc9122ed5bf3530",
                "0x3ff531913bfceaa7",
                "0x3fc9122ed5bf3530",
                "0x3fdf2b2afe84fc1d",
                "0x3fd5ed92d1a77e9d",
                "0x3ff531913bfceaa7",
                "0x3fd5ed92d1a77e9d",
                "0x3ffd9eb64f996920",
            ]
        ),
    )
    assert _bits(solution.residual_rms_m) == 0x3E23988E1409212E
    assert solution.metadata.converged is True
    assert solution.metadata.status == "step_tolerance"
    assert solution.metadata.used_measurements == 24
    assert solution.metadata.n_parameters == 6
    assert solution.metadata.redundancy == 18
    assert [len(epoch) for epoch in solution.used_sats] == [8, 8, 8]
    assert len(solution.residuals_m) == 24
    assert len(solution.per_epoch_influence) == 3
    assert len(solution.per_satellite_influence) == 24
    assert len(solution.per_satellite_batch_influence) == 8


def test_velocity_covariance_and_spp_doppler_bits():
    sp3 = _load_sp3(SP3_2020)
    carrier_hz = sidereon.carrier_frequency_hz(
        sidereon.GnssSystem.GPS,
        sidereon.CarrierBand.L1,
    )
    observations = [
        sidereon.VelocityObservation(satellite, _f64(bits), carrier_hz)
        for satellite, bits in VELOCITY_OBS_BITS
    ]
    velocity = sidereon.solve_velocity(
        sp3,
        observations,
        np.asarray([4_500_000.0, 500_000.0, 4_500_000.0]),
        646_272_000.0,
        sidereon.VelocitySolveOptions(),
    )

    assert np.array_equal(
        _array_bits(velocity.state_covariance),
        _expect_bits(
            [
                "0x3ff0906b12ade753",
                "0xbfd3507feaeb34da",
                "0x3fe4b8aaad393152",
                "0x3e2653d2334473f0",
                "0xbfd3507feaeb34dc",
                "0x3fe06337a5bee55f",
                "0x3f9ceec75f8410a1",
                "0xbdfba852d0276899",
                "0x3fe4b8aaad39314d",
                "0x3f9ceec75f8410c1",
                "0x3ffc72af9d76e44d",
                "0x3e30eae3e3aecb8c",
                "0x3e2653d2334473f2",
                "0xbdfba852d027689e",
                "0x3e30eae3e3aecb8b",
                "0x3c6ae29fdfe7f6ff",
            ]
        ),
    )
    assert np.array_equal(
        _array_bits(velocity.velocity_covariance_ecef_m2_s2),
        _expect_bits(
            [
                "0x3ff0906b12ade753",
                "0xbfd3507feaeb34da",
                "0x3fe4b8aaad393152",
                "0xbfd3507feaeb34dc",
                "0x3fe06337a5bee55f",
                "0x3f9ceec75f8410a1",
                "0x3fe4b8aaad39314d",
                "0x3f9ceec75f8410c1",
                "0x3ffc72af9d76e44d",
            ]
        ),
    )

    sp3_spp, config = _spp_sp3_config()
    receiver = sidereon.solve_spp(sp3_spp, config)
    doppler_observations = []
    for satellite in receiver.used_sats:
        predicted = sidereon.observe(
            sp3_spp,
            satellite,
            receiver.position,
            config.t_rx_j2000_s,
            carrier_hz,
            True,
            True,
        )
        doppler_observations.append(
            sidereon.VelocityObservation(
                satellite,
                predicted.doppler_hz,
                carrier_hz,
            )
        )

    combined = sidereon.solve_spp_with_doppler_velocity(
        sp3_spp,
        config,
        doppler_observations,
    )
    assert combined.velocity_error is None
    assert _bits(combined.receiver.rx_clock_drift_s_s) == 0x3B29AEA08CA6BA9C
    assert np.array_equal(
        _array_bits(combined.velocity.velocity_m_s),
        _expect_bits(
            [
                "0xbd19c30b81d188ea",
                "0xbd12da0aa87eaef9",
                "0x3d1e8af83acabada",
            ]
        ),
    )
    assert _bits(combined.velocity.clock_drift_s_s) == 0x3B29AEA08CA6BA9C
    assert combined.velocity.used_sats == receiver.used_sats


def test_emission_media_batch_statuses_and_arrays():
    sp3 = _load_sp3(SP3_2020)
    epochs = np.asarray([sp3.epochs_j2000_seconds[20]] * 4, dtype=np.float64)
    receiver = np.asarray(
        [4_484_127.99232578, 550_581.68657014, 4_487_560.54090027],
        dtype=np.float64,
    )

    batch = sidereon.emission_media_batch_at_j2000_s(
        sp3,
        ["G08", "G10", "G16", "S20"],
        epochs,
        receiver,
        troposphere=True,
        min_elevation_rad=0.0,
    )

    assert len(batch) == 4
    assert batch.element_count == 4
    assert batch.is_empty is False
    assert [status.label for status in batch.statuses] == [
        "below_elevation_cutoff",
        "valid",
        "below_elevation_cutoff",
        "gap",
    ]
    assert batch.element_status(1) == sidereon.EmissionMediaStatus.VALID
    assert batch.element_errors[:3] == [None, None, None]
    assert batch.element_errors[3] == "unknown satellite: S20"
    assert np.array_equal(
        _array_bits(batch.positions_ecef_m[:3]),
        _expect_bits(
            [
                "0xc17862e2fb0e5605",
                "0xc1589c38cbb645a2",
                "0xc14fbbab1c8b4394",
                "0x41622a7299604188",
                "0xc175a31a4f8d4fdf",
                "0x4162a3283b8d4fdf",
                "0xc16428f5b8a3d70a",
                "0xc1641be8dd89374c",
                "0xc1752766ff0a3d70",
            ]
        ),
    )
    assert np.isnan(batch.positions_ecef_m[3]).all()
    assert np.array_equal(
        _array_bits(batch.clocks_s[:3]),
        _expect_bits(
            [
                "0xbf043ee565c458cc",
                "0xbf38ec32fa783b61",
                "0xbf26d7922860f9f7",
            ]
        ),
    )
    assert np.isnan(batch.clocks_s[3])
    assert np.isnan(batch.troposphere_delays_m[[0, 2, 3]]).all()
    assert _bits(batch.troposphere_delays_m[1]) == 0x40259A5E1E5E4264
    assert np.isnan(batch.ionosphere_slant_delays_m[[0, 2, 3]]).all()
    assert batch.ionosphere_slant_delays_m[1] == 0.0


def test_precise_interpolant_artifact_round_trip_and_typed_errors():
    sp3 = _load_sp3(
        os.path.join(
            FIXTURES,
            "sp3",
            "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3",
        )
    )
    artifact_bytes = sidereon.build_precise_interpolant_artifact_bytes(sp3)
    assert artifact_bytes == sp3.precise_interpolant_artifact_bytes()

    artifact = sidereon.PreciseInterpolantArtifact.from_bytes(artifact_bytes)
    assert artifact.byte_len == 128_064
    assert artifact.checksum64 == 0xA2C3B142602A6F56
    assert artifact.satellites[:5] == ["G01", "G02", "G03", "G04", "G05"]
    assert artifact.as_bytes() == artifact_bytes

    state = artifact.position_at_j2000_seconds("G01", 830_818_800.0)
    assert np.array_equal(
        _array_bits(state.position_m),
        _expect_bits(
            [
                "0x416b9b88594fdf3b",
                "0xc1747e3808b851eb",
                "0xc155928c6e872b02",
            ]
        ),
    )
    assert _bits(state.clock_s) == 0x3F32C01C0ACC1CF9

    with pytest.raises(sidereon.PreciseInterpolantArtifactTruncatedError):
        sidereon.PreciseInterpolantArtifact.from_bytes(artifact_bytes[:-1])

    corrupt = bytearray(artifact_bytes)
    corrupt[-1] ^= 0x01
    with pytest.raises(sidereon.PreciseInterpolantArtifactCorruptError):
        sidereon.PreciseInterpolantArtifact.from_bytes(corrupt)
