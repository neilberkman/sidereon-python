"""0.17 domain exposure parity tests against patched core outputs."""

import json
import math
import os
import struct

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES, STATUS_LABELS, core_goldens, hex_to_f64

SP3_2020 = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")

# The core `velocity` module scenario: range rates the core's `predict`
# synthesizes from a true velocity and clock drift, written with their
# solutions by `scripts/core_goldens`.
_VELOCITY = core_goldens()["velocity"]
VELOCITY_OBS_BITS = [
    (sat, int(bits, 16))
    for sat, bits in zip(_VELOCITY["satellites"], _VELOCITY["range_rate_m_s"])
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


def _spp_sp3_config(perturb_m=0.0, satellite_perturb_m=None):
    fixture = _spp_fixture()
    inputs = fixture["inputs"]
    sp3 = _load_sp3(os.path.join(CORE_FIXTURES, "sp3", inputs["sp3_file"]))
    satellite_perturb_m = satellite_perturb_m or {}
    observations = [
        sidereon.SppObservation(
            row["sat_id"],
            hex_to_f64(row["p_meas_m"])
            + perturb_m
            + satellite_perturb_m.get(row["sat_id"], 0.0),
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
    assert _bits(fence.distance_to_boundary(inside)) == 0x409146F89A157D9A
    assert _bits(fence.distance_to_boundary(outside)) == 0xC0A164C795F1FD1B
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

    golden = core_goldens()["static_three_epochs"]
    assert np.array_equal(
        _array_bits(solution.position), _expect_bits(golden["position_m"])
    )
    assert [
        (index, system.label, _bits(clock_s))
        for index, system, clock_s in solution.per_epoch_clock
    ] == [
        (index, system, int(bits, 16))
        for index, system, bits in golden["per_epoch_clock"]
    ]
    assert np.array_equal(
        _array_bits(solution.covariance.position_ecef_m2),
        _expect_bits(golden["position_covariance_ecef_m2"]),
    )
    assert _bits(solution.residual_rms_m) == int(golden["residual_rms_m"], 16)
    assert solution.metadata.converged is golden["converged"]
    assert solution.metadata.status == STATUS_LABELS[golden["status"]]
    assert solution.metadata.used_measurements == golden["used_measurements"]
    assert solution.metadata.n_parameters == golden["n_parameters"]
    assert solution.metadata.redundancy == golden["redundancy"]
    assert [len(epoch) for epoch in solution.used_sats] == golden["used_sat_counts"]
    assert len(solution.residuals_m) == golden["residual_count"]
    assert len(solution.per_epoch_influence) == 3
    assert len(solution.per_satellite_influence) == 24
    assert len(solution.per_satellite_batch_influence) == 8
    assert solution.per_epoch_influence[0][2] == "solved"
    assert solution.per_satellite_influence[0][2] == "solved"
    assert solution.per_satellite_batch_influence[0][2] == "solved"
    assert solution.covariance.state_m2.shape == (6, 6)
    assert solution.covariance.position_enu_m2.shape == (3, 3)
    assert solution.geometry_quality.redundancy == solution.metadata.redundancy


def test_static_positioning_covariance_redundancy_and_robust_contract():
    sp3, config0 = _spp_sp3_config()
    _, config1 = _spp_sp3_config(0.2, {"G08": 400.0})
    _, config2 = _spp_sp3_config(-0.1)
    epochs = [
        sidereon.StaticEpoch(config0),
        sidereon.StaticEpoch(config1),
        sidereon.StaticEpoch(config2),
    ]
    robust = sidereon.SppRobustConfig(max_outer=4, scale_floor_m=0.01)
    options = sidereon.StaticSolveOptions(
        initial_position_m=config0.initial_guess[:3],
        with_geodetic=True,
        robust=robust,
    )

    solution = sidereon.solve_static(sp3, epochs, options)

    assert solution.geodetic is not None
    assert solution.covariance.position_ecef_m2.shape == (3, 3)
    assert solution.covariance.position_enu_m2.shape == (3, 3)
    assert solution.covariance.state_m2.shape == (
        solution.metadata.n_parameters,
        solution.metadata.n_parameters,
    )
    assert solution.metadata.redundancy == (
        solution.metadata.used_measurements - solution.metadata.n_parameters
    )
    assert solution.metadata.outer_iterations > 0
    assert solution.metadata.final_robust_scale_m is not None
    assert any(row[4] < row[3] for row in solution.residuals_m)
    assert any(row[5] < 1.0 for row in solution.residuals_m)
    assert any(row[6] < 1.0 for row in solution.per_epoch_influence)
    assert any(row[9] < 1.0 for row in solution.per_satellite_influence)
    assert any(row[6] < 1.0 for row in solution.per_satellite_batch_influence)


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

    state = _VELOCITY["range_rate_solution"]["state_covariance"]
    assert np.array_equal(_array_bits(velocity.state_covariance), _expect_bits(state))
    # The ECEF velocity block is the upper-left 3x3 of the 4x4 state covariance.
    assert np.array_equal(
        _array_bits(velocity.velocity_covariance_ecef_m2_s2),
        _expect_bits([state[row * 4 + col] for row in range(3) for col in range(3)]),
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
    assert combined.velocity_error_detail is None

    partial = sidereon.solve_spp_with_doppler_velocity(
        sp3_spp,
        config,
        doppler_observations[:1],
    )
    assert partial.velocity is None
    assert partial.velocity_error == "too few satellites: 1, required 4"
    assert partial.velocity_error_detail == {
        "family": "VelocityError",
        "kind": "too_few_satellites",
        "message": partial.velocity_error,
        "used": 1,
        "required": 4,
    }
    # The core's own solve of the same inputs, written by `scripts/core_goldens`.
    # The receiver is stationary and no drift is injected, yet the recovered
    # speed is 6.1e-4 m/s. `spp_doppler_terms` in the same file separates the
    # causes. Solving these Dopplers with `solve_velocity`, which predicts each
    # row as `observe` does, gives 9e-15 m/s with Sagnac on both sides and
    # 5e-14 m/s with it off on both. Moving each satellite from the geometric
    # light-time epoch to its pseudorange transmission epoch, with Sagnac off on
    # both sides, moves each row by 1.5e-6 to 1.2e-5 m/s and the solution by
    # 1.1e-5 m/s. The rest comes from the Sagnac formulation: this solve ranges
    # the unrotated placed state and adds the rate of the first-order Sagnac
    # term, as RTKLIB `geodist` does, where `observe` predicts in the frame
    # rotated to the reception epoch.
    golden = core_goldens()["spp_doppler"]
    assert receiver.used_sats == golden["used_sats"]
    assert _bits(combined.receiver.rx_clock_drift_s_s) == int(
        golden["rx_clock_drift_s_s"], 16
    )
    assert np.array_equal(
        _array_bits(combined.velocity.velocity_m_s),
        _expect_bits(golden["velocity_m_s"]),
    )
    assert _bits(combined.velocity.clock_drift_s_s) == int(
        golden["clock_drift_s_s"], 16
    )
    assert combined.velocity.used_sats == golden["velocity_used_sats"]

    # The first two separating solves run through the binding as well.
    terms = core_goldens()["spp_doppler_terms"]
    for sagnac, key in ((True, "unplaced_sagnac_on"), (False, "unplaced_sagnac_off")):
        rows = [
            sidereon.VelocityObservation(
                satellite,
                sidereon.observe(
                    sp3_spp,
                    satellite,
                    receiver.position,
                    config.t_rx_j2000_s,
                    carrier_hz,
                    True,
                    sagnac,
                ).doppler_hz,
                carrier_hz,
            )
            for satellite in receiver.used_sats
        ]
        unplaced = sidereon.solve_velocity(
            sp3_spp,
            rows,
            receiver.position,
            config.t_rx_j2000_s,
            sidereon.VelocitySolveOptions(
                observable=sidereon.VelocityObservable.DOPPLER,
                light_time=True,
                sagnac=sagnac,
            ),
        )
        assert np.array_equal(
            _array_bits(unplaced.velocity_m_s),
            _expect_bits(terms[key]["velocity_m_s"]),
        )
        assert _bits(unplaced.clock_drift_s_s) == int(terms[key]["clock_drift_s_s"], 16)
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
    golden = core_goldens()["emission_media"]
    assert np.array_equal(
        _array_bits(batch.positions_ecef_m[:3]),
        _expect_bits(
            [
                component
                for position in golden["positions_ecef_m"][:3]
                for component in position
            ]
        ),
    )
    assert np.isnan(batch.positions_ecef_m[3]).all()
    assert np.array_equal(
        _array_bits(batch.clocks_s[:3]),
        _expect_bits(golden["clocks_s"][:3]),
    )
    assert np.isnan(batch.clocks_s[3])
    assert np.isnan(batch.troposphere_delays_m[[0, 2, 3]]).all()
    # The last bit of this value is build- and arch-sensitive (arm64 macOS and
    # x86_64 Linux libm land on adjacent ULPs on the mapping-function path, and
    # it has flipped between builds), so the pin is a one-ULP band around the
    # canonical value. Anything past one ULP is a real regression.
    assert (
        abs(
            _bits(batch.troposphere_delays_m[1])
            - int(golden["troposphere_delays_m"][1], 16)
        )
        <= 1
    ), _bits(batch.troposphere_delays_m[1])
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
    assert artifact_bytes[:8] == b"PEMAP001"
    assert struct.unpack_from("<H", artifact_bytes, 8)[0] == 2
    assert len(sp3.epochs_j2000_seconds) == 11
    assert len(artifact.satellites) == 31
    assert artifact.byte_len == 31 * 4096 + 11 * 160 + 32 == 128_768
    assert struct.unpack_from("<Q", artifact_bytes, 32)[0] == artifact.byte_len
    checksum = 0xCBF29CE484222325
    for index, byte in enumerate(artifact_bytes):
        value = 0 if 40 <= index < 48 else byte
        checksum = ((checksum ^ value) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    assert artifact.checksum64 == checksum
    assert struct.unpack_from("<Q", artifact_bytes, 40)[0] == checksum
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
