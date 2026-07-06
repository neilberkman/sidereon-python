"""0.18 domain exposure parity tests against patched core outputs."""

import math
import pathlib
import re
import struct

import numpy as np
import sidereon

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_SCENARIO_SEED = 0x515C_1E7E_0B5E_A11D


def _bits(value):
    return int.from_bytes(struct.pack(">d", float(value)), "big")


def _array_bits(values):
    return [_bits(value) for value in np.asarray(values, dtype=np.float64).ravel()]


def _gps_anchor_orbit(start_j2000_s, prn, raan_rad, inclination_rad, mean_anomaly_rad):
    return {
        "satellite_id": {"system": "Gps", "prn": prn},
        "semi_major_axis_m": 26_560_000.0,
        "eccentricity": 0.0,
        "inclination_rad": inclination_rad,
        "raan_rad": raan_rad,
        "arg_perigee_rad": 0.0,
        "mean_anomaly_rad": mean_anomaly_rad,
        "epoch_j2000_s": start_j2000_s,
        "clock_bias_s": 0.0,
        "clock_drift_s_s": 0.0,
    }


def _base_scenario():
    start_j2000_s = sidereon.j2000_seconds(2026, 1, 1, 0, 0, 0.0)
    u60 = math.pi / 3.0
    disabled_clock = {
        "enabled": False,
        "bias_s": 0.0,
        "drift_s_s": 0.0,
        "power_law_coefficients": [0.0] * 5,
    }
    return {
        "schema_version": sidereon.scenario_schema_version(),
        "seed": DEFAULT_SCENARIO_SEED,
        "epochs": {
            "start_j2000_s": start_j2000_s,
            "count": 2,
            "cadence_s": 30.0,
        },
        "receiver": {
            "kind": "static_geodetic",
            "position": {"lat_rad": 0.0, "lon_rad": 0.0, "height_m": 0.0},
        },
        "constellation": {
            "kind": "synthetic_keplerian",
            "satellites": [
                _gps_anchor_orbit(start_j2000_s, 1, 0.0, 0.0, 0.0),
                _gps_anchor_orbit(start_j2000_s, 2, 0.0, 0.0, u60),
                _gps_anchor_orbit(start_j2000_s, 3, 0.0, 0.0, -u60),
                _gps_anchor_orbit(start_j2000_s, 4, 0.0, math.pi / 2.0, u60),
                _gps_anchor_orbit(start_j2000_s, 5, 0.0, math.pi / 2.0, -u60),
            ],
        },
        "signals": [
            {
                "system": "Gps",
                "code_observable": "C1C",
                "phase_observable": "L1C",
                "doppler_observable": "D1C",
                "carrier_hz": 1_575_420_000.0,
                "carrier_phase_bias_cycles": 0.0,
            }
        ],
        "error_budget": {
            "receiver_clock": disabled_clock,
            "satellite_clock": disabled_clock,
            "ionosphere": {"kind": "off"},
            "troposphere": {"kind": "off"},
            "thermal_noise": {
                "enabled": False,
                "pseudorange_sigma_m": 0.0,
                "carrier_phase_sigma_m": 0.0,
                "doppler_sigma_hz": 0.0,
            },
            "multipath": {
                "enabled": False,
                "amplitude_m": 0.0,
                "reflector_height_m": 0.0,
                "phase_rad": 0.0,
            },
            "elevation_mask_deg": -5.0,
        },
    }


def _filter_state():
    nominal = sidereon.NavState(
        0.0,
        [6_378_137.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    return sidereon.InsFilterState.from_diagonal(
        nominal, sidereon.ErrorStateLayout.FIFTEEN, [1.0] * 15
    )


def _zero_fix(t_j2000_s, covariance_scale_m2):
    return sidereon.GnssFixMeasurement.position(
        t_j2000_s,
        [6_378_137.0, 0.0, 0.0],
        np.eye(3, dtype=np.float64) * covariance_scale_m2,
        5,
    )


def test_manifest_has_no_cargo_path_deps():
    # The dev-time [patch.crates-io] override lives in a git-excluded
    # .cargo/config.toml whose location and contents are environment plumbing;
    # the releasable invariant is that the manifest itself carries version-only
    # dependencies.
    cargo_toml = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert re.search(r'sidereon = "\d+\.\d+\.\d+"', cargo_toml)
    assert re.search(r'sidereon-core = "\d+\.\d+\.\d+"', cargo_toml)
    assert "sidereon = { path" not in cargo_toml
    assert "sidereon-core = { path" not in cargo_toml


def test_scenario_simulator_deterministic_bytes_and_core_term_bits():
    scenario = _base_scenario()

    bytes_a = sidereon.simulate_scenario_bytes(scenario)
    bytes_b = sidereon.simulate_scenario_bytes(scenario)
    output = sidereon.simulate_scenario(scenario)

    assert bytes_a == bytes_b
    assert output.as_json_bytes() == bytes_a
    assert len(bytes_a) == 3998
    assert bytes_a.startswith(b'{"schema_version":1,"engine_version":"')
    assert output.schema_version == 1
    assert re.fullmatch(r"\d+\.\d+\.\d+:scenario-observables-v1", output.engine_version)
    assert output.seed == DEFAULT_SCENARIO_SEED
    assert output.observation_count() == 10
    assert len(output) == 10
    # The fingerprint stamps the engine version by design, so it changes each
    # release; determinism across calls is the invariant.
    again = sidereon.simulate_scenario(_base_scenario())
    assert output.determinism_fingerprint() == again.determinism_fingerprint()
    assert output.observations.satellite_id == [
        "G01",
        "G02",
        "G03",
        "G04",
        "G05",
        "G01",
        "G02",
        "G03",
        "G04",
        "G05",
    ]
    assert _array_bits(output.observations.pseudorange_m[:3]) == [
        0x41733F367001A84B,
        0x4176E6F8EBDA917E,
        0x4176E701D7EB9C6A,
    ]
    assert _array_bits(output.observations.doppler_hz[:3]) == [
        0x3FA02C7DC468DEC0,
        0xC0A24AEDAD138034,
        0x40A24AF6B93D53C9,
    ]
    assert _bits(output.truth_terms.pseudorange_sum_m(0)) == _bits(
        output.observations.pseudorange_m[0]
    )
    assert output.spp_observations_for_epoch(0)[0][0] == "G01"


def test_signal_analysis_closed_forms_match_core_bits():
    bpsk = sidereon.SignalModulation.bpsk1()
    boc = sidereon.SignalModulation.boc_sine(1.0, 1.0)
    bandwidth_hz = sidereon.signal_betz_l1_receiver_bandwidth_hz()

    assert bpsk.label == "BPSK(1)"
    assert boc.label == "BOC(m,n)"
    assert bandwidth_hz == 24_000_000.0
    assert _array_bits(
        [sidereon.signal_psd_hz(bpsk, offset) for offset in [0.0, 1.023e6, 2.046e6]]
    ) == [
        0x3EB066676CCCDD33,
        0x3825A2CF64E71094,
        0x3825A2CF64E71094,
    ]
    assert _array_bits(
        [sidereon.signal_psd_hz(boc, offset) for offset in [0.0, 1.023e6, 2.046e6]]
    ) == [
        0x0000000000000000,
        0x3E9A96323AC6C962,
        0x0000000000000000,
    ]
    assert _bits(sidereon.signal_fraction_power_in_band(bpsk, bandwidth_hz)) == (
        0x3FEFBA305BC2217B
    )
    assert _bits(sidereon.signal_rms_bandwidth_hz(bpsk, bandwidth_hz)) == (
        0x4131348B6B5C9628
    )
    assert (
        _bits(
            sidereon.signal_spectral_separation_coefficient_hz(bpsk, boc, bandwidth_hz)
        )
        == 0x3E85DDCA164B7F18
    )
    assert (
        _bits(
            sidereon.signal_spectral_separation_coefficient_db_hz(
                bpsk, boc, bandwidth_hz
            )
        )
        == 0xC050F8575FE9E076
    )

    degradation = sidereon.signal_effective_cn0_degradation(
        bpsk, 45.0, bandwidth_hz, [sidereon.InterferenceTerm(boc, 0.5)]
    )
    assert _bits(degradation.effective_cn0_hz) == 0x40DECD352BDBBE97
    assert _bits(degradation.effective_cn0_db_hz) == 0x40467E8EBF2ECDD4
    assert _bits(degradation.degradation_db) == 0x3F87140D1322C000

    options = sidereon.DllTrackingOptions(45.0, 1.0, 0.02, 0.1, bandwidth_hz)
    coherent = sidereon.signal_dll_thermal_noise_jitter(
        bpsk, options, sidereon.DllProcessing.COHERENT
    )
    noncoherent = sidereon.signal_dll_thermal_noise_jitter(
        bpsk, options, sidereon.DllProcessing.NON_COHERENT
    )
    lower_bound = sidereon.signal_dll_lower_bound(bpsk, options)
    assert _array_bits(
        [coherent.seconds, coherent.chips, coherent.meters, coherent.squaring_loss]
    ) == [
        0x3E116ACA3A9BA2DA,
        0x3F50FE08EADCAE0A,
        0x3FD373A385AD2B5A,
        0x3FF0000000000000,
    ]
    assert _array_bits(
        [
            noncoherent.seconds,
            noncoherent.chips,
            noncoherent.meters,
            noncoherent.squaring_loss,
        ]
    ) == [
        0x3E116E7C96B098A5,
        0x3F5101A431BDAAF4,
        0x3FD377C46E0BDEDB,
        0x3FF006CB70574C26,
    ]
    assert _array_bits(
        [
            lower_bound.seconds,
            lower_bound.chips,
            lower_bound.meters,
            lower_bound.squaring_loss,
        ]
    ) == [
        0x3E0B409E692B7BA7,
        0x3F4A96736C07C9A3,
        0x3FCE6F95C8606022,
        0x3FF0000000000000,
    ]

    envelope = sidereon.signal_multipath_error_envelope(
        bpsk,
        sidereon.MultipathOptions(0.5, 0.1, bandwidth_hz),
        np.array([0.0, 0.1, 0.2], dtype=np.float64),
    )
    assert len(envelope) == 3
    assert _array_bits(envelope.in_phase_m) == [
        0x0000000000000000,
        0x401BA766FF476EBB,
        0x401AFC175D9AF909,
    ]
    assert _array_bits(envelope.anti_phase_m) == [
        0x0000000000000000,
        0xC01A62B42F7F4AC2,
        0xC01AB9D9FEBDE9CA,
    ]
    assert _array_bits(envelope.running_average_m) == [
        0x0000000000000000,
        0x400BA766FF476EBB,
        0x401492B9570A759E,
    ]


def test_fusion_filter_checkpoint_loose_ukf_tight_and_time_sync_bits():
    state = _filter_state()
    spec = sidereon.ImuSpec.mems()
    filter_ = sidereon.InertialFilter(state, spec)
    config = sidereon.InertialFilterConfig(spec)

    encoded = filter_.encode_state()
    restored = sidereon.InertialFilter.from_encoded_state(encoded, config)
    assert isinstance(encoded, bytes)
    assert encoded
    assert restored.encode_state() == encoded

    update = filter_.update_loose(_zero_fix(0.0, 4.0))
    assert update.applied is True
    assert _bits(update.nis) == 0x0000000000000000
    assert (update.rows, update.accepted_rows, update.rejected_rows) == (3, 3, 0)
    assert _bits(update.ekf.normalized_innovation_squared) == 0x0000000000000000
    assert update.ekf.dx.shape == (15,)

    ukf = sidereon.InertialFilter.with_config(
        state,
        sidereon.InertialFilterConfig(spec, filter_kind=sidereon.FusionFilterKind.UKF),
    )
    ukf_update = ukf.update_loose(_zero_fix(0.0, 4.0))
    assert ukf.config.filter_kind == sidereon.FusionFilterKind.UKF
    assert ukf.config.tight.initial_clock_bias_variance_m2 == 1.0e12
    assert ukf.config.tight.initial_clock_drift_variance_m2_s2 == 1.0e6
    assert ukf.config.tight.clock_bias_random_walk_m2_s == 1.0
    assert ukf.config.tight.clock_drift_random_walk_m2_s3 == 1.0e-2
    assert ukf.config.tight.update_options.innovation_gate is None
    assert ukf_update.applied is True
    assert _bits(ukf_update.nis) == 0x0000000000000000
    assert (ukf_update.rows, ukf_update.accepted_rows, ukf_update.rejected_rows) == (
        3,
        3,
        0,
    )

    range_rate = sidereon.TightRangeRateObservation(0.0, 0.1, 0.0)
    carrier_phase = sidereon.TightCarrierPhaseObservation(21_000_000.0, 0.01, 0.0)
    observation = sidereon.TightGnssObservation(
        "G01",
        21_000_000.0,
        1.0,
        range_rate=range_rate,
        carrier_phase=carrier_phase,
    )
    epoch = sidereon.TightGnssEpoch(0.0, [observation])
    assert epoch.t_j2000_s == 0.0
    assert epoch.observation_count == 1
    assert observation.satellite_id == "G01"
    assert observation.range_rate.measured_range_rate_m_s == 0.0
    assert observation.range_rate.sigma_m_s == 0.1
    assert observation.range_rate.satellite_clock_drift_m_s == 0.0
    assert observation.carrier_phase.phase_range_m == 21_000_000.0
    assert observation.carrier_phase.sigma_m == 0.01
    assert observation.carrier_phase.float_ambiguity_m == 0.0
    assert observation.ionosphere_delay_m == 0.0
    assert observation.troposphere_delay_m == 0.0

    replay_filter = sidereon.InertialFilter(state, spec)
    replay_filter.configure_time_sync_history(sidereon.TimeSyncHistoryConfig(8, 4))
    for t_j2000_s in [0.25, 0.5, 0.75, 1.0]:
        replay_filter.propagate(
            sidereon.ImuSample.rate(t_j2000_s, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        )
    status = replay_filter.time_sync_history_status()
    assert (status.imu_capacity, status.imu_len) == (8, 4)
    assert (status.checkpoint_capacity, status.checkpoint_len) == (4, 1)
    assert status.oldest_imu_epoch_j2000_s == 0.25
    assert status.newest_imu_epoch_j2000_s == 1.0

    replayed = replay_filter.update_loose_time_sync(_zero_fix(0.5, 25.0))
    assert replayed.late_measurement is True
    assert replayed.replayed_imu_segments == 4
    assert _bits(replayed.restored_checkpoint_epoch_j2000_s) == 0x0000000000000000
    assert _bits(replayed.current_epoch_j2000_s) == 0x3FF0000000000000
    assert replayed.update.applied is True
    assert _bits(replayed.update.nis) == 0x3FAD226CC390657C


def test_fusion_robust_loose_recorded_rts_bits():
    state = _filter_state()
    spec = sidereon.ImuSpec.mems()
    loose = sidereon.LooseCouplingConfig(
        update_options=sidereon.EkfUpdateOptions(sidereon.InnovationGate(4.0, 2)),
        measurement_reweighting=sidereon.IggIiiMeasurementReweighting.standard(),
        prediction_adaptation=sidereon.YangPredictionAdaptiveFactor.standard(),
    )
    assert loose.measurement_reweighting.k0_sigma == 2.0
    assert loose.measurement_reweighting.k1_sigma == 5.0
    assert loose.prediction_adaptation.threshold == 1.0
    assert loose.prediction_adaptation.outlier_gate_probability == 0.99

    filter_ = sidereon.InertialFilter.with_config(
        state,
        sidereon.InertialFilterConfig(spec, loose=loose),
    )
    history = sidereon.FusionRtsHistoryBuilder.from_filter(filter_)
    snapshot = filter_.snapshot()
    filter_.restore_snapshot(snapshot)
    filter_.propagate_recorded(
        sidereon.ImuSample.rate(1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        history,
    )
    update = filter_.update_loose_recorded(
        sidereon.GnssFixMeasurement.position(
            1.0,
            [6_378_137.35, 0.2, -0.1],
            np.eye(3, dtype=np.float64) * 0.5,
            7,
        ),
        history,
    )
    recorded = history.finish()
    smoothed = sidereon.smooth_fusion_rts(recorded)

    assert update.applied is True
    assert (update.rows, update.accepted_rows, update.rejected_rows) == (3, 3, 0)
    gate = update.ekf.innovation_gate
    assert gate.max_rejected_abs_normalized_innovation is None
    assert _bits(update.nis) == 0x400A42AD3B07976F
    assert _bits(gate.max_abs_normalized_innovation) == 0x3FFCF4BA7AE7BCC0
    assert _array_bits(filter_.state.nominal.position_ecef_m) == [
        0x415854A602757FB6,
        0x3FC7B6B11D7FA0D8,
        0xBFB7B6B11D5C2B22,
    ]
    assert len(recorded) == 2
    assert len(smoothed) == 2
    assert recorded.epochs[0].transition_from_previous is None
    assert recorded.epochs[1].transition_from_previous.shape == (15, 15)
    assert smoothed.epochs[0].rts_gain_to_next.shape == (17, 17)
    assert smoothed.epochs[1].rts_gain_to_next is None
    assert _array_bits(np.diag(recorded.epochs[1].transition_from_previous)[:3]) == [
        0x3FF000019D17A15A,
        0x3FEFFFFE650C7E2C,
        0x3FEFFFFE639F13D3,
    ]
    assert _array_bits(smoothed.epochs[0].snapshot.state.nominal.position_ecef_m) == [
        0x415854A6AFB47DAB,
        0x3FB5122C16E56642,
        0xBFA5122C1780E0A5,
    ]
    assert _array_bits(smoothed.epochs[1].snapshot.state.nominal.position_ecef_m) == [
        0x415854A602757FB6,
        0x3FC7B6B11D7FA0D8,
        0xBFB7B6B11D5C2B22,
    ]
    assert _array_bits(smoothed.epochs[0].error_state_correction[:6]) == [
        0xBFFBED1F6AC3E068,
        0xBFB5122C16E56642,
        0x3FA5122C1780E0A5,
        0xBFFBED164E925C0A,
        0xBFB51A847AAA1978,
        0x3FA5122D270AB803,
    ]
    assert _array_bits(np.diag(smoothed.epochs[0].covariance)[:5]) == [
        0x3FFDC64F219100F6,
        0x3FFA44D611536A90,
        0x3FFA44D6119F127C,
        0x3FFDBA1DE20184E2,
        0x3FFA389FFA3F4082,
    ]
