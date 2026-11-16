"""0.18 domain exposure parity tests against patched core outputs."""

import json
import math
import os
import pathlib
import re
import struct

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES, core_goldens


def _engine_dep_is_registry_versioned(manifest: str, name: str) -> bool:
    """Whether `name` is a registry dependency with a stated version.

    Accepts both `name = "X.Y.Z"` and `name = { version = "X.Y.Z", ... }`; the
    table form is required to enable a dependency feature. Rejects any path or
    git dependency, which is the invariant these tests exist to protect: the
    published wheel must build from the registry, not from a local checkout.
    """
    line = re.search(rf"^{re.escape(name)} = (.+)$", manifest, re.M)
    if line is None:
        return False
    value = line.group(1)
    if "path" in value or "git" in value:
        return False
    return (
        re.search(r'version = "=?\d+\.\d+\.\d+"', value) is not None
        or re.fullmatch(r'"=?\d+\.\d+\.\d+"', value.strip()) is not None
    )


def _assert_engine_manifest_sources_are_approved(manifest: str) -> None:
    candidate = re.search(
        r"^sidereon(?:-core)?\s*=\s*\{[^}\n]*\bgit\s*=",
        manifest,
        re.M,
    )
    if candidate is None:
        assert _engine_dep_is_registry_versioned(manifest, "sidereon")
        assert _engine_dep_is_registry_versioned(manifest, "sidereon-core")
        return

    import runpy

    import tomllib

    dependencies = tomllib.loads(manifest)["dependencies"]
    release_guard = runpy.run_path(str(ROOT / "scripts" / "check-release.py"))
    package_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    revision = release_guard["candidate_fixture_revision"](
        dependencies, package_version
    )
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


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


def _golden_bits(hex_values):
    return [int(value, 16) for value in hex_values]


def _assert_fusion_update_matches(update, golden):
    assert update.applied is golden["applied"]
    assert (update.rows, update.accepted_rows, update.rejected_rows) == (
        golden["rows"],
        golden["accepted_rows"],
        golden["rejected_rows"],
    )
    assert _bits(update.nis) == int(golden["nis"], 16)
    assert _bits(update.ekf.normalized_innovation_squared) == int(golden["ekf_nis"], 16)


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


def test_manifest_uses_registry_or_approved_candidate_dependencies():
    # The dev-time [patch.crates-io] override lives in a git-excluded
    # .cargo/config.toml whose location and contents are environment plumbing;
    # candidate Git dependencies are accepted only through the strict release
    # guard's full-revision validation.
    cargo_toml = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    _assert_engine_manifest_sources_are_approved(cargo_toml)


def test_scenario_simulator_deterministic_bytes_and_core_term_bits():
    scenario = _base_scenario()

    bytes_a = sidereon.simulate_scenario_bytes(scenario)
    bytes_b = sidereon.simulate_scenario_bytes(scenario)
    output = sidereon.simulate_scenario(scenario)

    assert bytes_a == bytes_b
    assert output.as_json_bytes() == bytes_a
    # `scripts/core_goldens` runs the core simulator on `scenario_base.json`,
    # which must be the scenario this test builds.
    with open(os.path.join(FIXTURES, "scenario_base.json")) as handle:
        assert _base_scenario() == json.load(handle)
    golden = core_goldens()["scenario"]
    # The output embeds the engine version string, so the total length moves
    # with the version's width; pin the version-independent remainder instead
    # of a literal that breaks on every release whose version changes width.
    engine_semver = output.engine_version.split(":")[0]
    assert len(bytes_a) == golden["bytes_without_version"] + len(engine_semver)
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
        int(bits, 16) for bits in golden["pseudorange_m"]
    ]
    assert _array_bits(output.observations.doppler_hz[:3]) == [
        int(bits, 16) for bits in golden["doppler_hz"]
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

    # The core's run of the same filter steps, written by
    # `scripts/core_goldens` to `fixtures/core_goldens.json`.
    golden = core_goldens()["fusion"]["checkpoint_ukf_time_sync"]
    update = filter_.update_loose(_zero_fix(0.0, 4.0))
    assert update.applied is True
    assert (update.rows, update.accepted_rows, update.rejected_rows) == (3, 3, 0)
    _assert_fusion_update_matches(update, golden["ekf_update"])
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
    assert (ukf_update.rows, ukf_update.accepted_rows, ukf_update.rejected_rows) == (
        3,
        3,
        0,
    )
    _assert_fusion_update_matches(ukf_update, golden["ukf_update"])

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
    time_sync = golden["time_sync"]
    assert replayed.late_measurement is True
    assert replayed.late_measurement is time_sync["late_measurement"]
    assert replayed.replayed_imu_segments == 4
    assert replayed.replayed_imu_segments == time_sync["replayed_imu_segments"]
    assert _bits(replayed.restored_checkpoint_epoch_j2000_s) == int(
        time_sync["restored_checkpoint_epoch_j2000_s"], 16
    )
    assert _bits(replayed.current_epoch_j2000_s) == int(
        time_sync["current_epoch_j2000_s"], 16
    )
    assert replayed.update.applied is True
    _assert_fusion_update_matches(replayed.update, time_sync["update"])


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

    # The core's run of the same filter steps, written by
    # `scripts/core_goldens` to `fixtures/core_goldens.json`.
    golden = core_goldens()["fusion"]["robust_loose_rts"]
    assert update.applied is True
    assert (update.rows, update.accepted_rows, update.rejected_rows) == (3, 3, 0)
    _assert_fusion_update_matches(update, golden["update"])
    gate = update.ekf.innovation_gate
    assert gate.max_rejected_abs_normalized_innovation is None
    assert _bits(gate.max_abs_normalized_innovation) == int(
        golden["gate_max_abs_normalized_innovation"], 16
    )
    assert _array_bits(filter_.state.nominal.position_ecef_m) == _golden_bits(
        golden["state"]["position_ecef_m"]
    )
    assert _array_bits(np.diag(filter_.state.covariance)) == _golden_bits(
        golden["state"]["covariance_diagonal"]
    )
    assert len(recorded) == 2
    assert len(recorded) == golden["recorded_epoch_count"]
    assert len(smoothed) == 2
    assert len(smoothed) == len(golden["smoothed"])
    assert recorded.epochs[0].transition_from_previous is None
    assert recorded.epochs[1].transition_from_previous.shape == (15, 15)
    assert smoothed.epochs[0].rts_gain_to_next.shape == (17, 17)
    assert smoothed.epochs[1].rts_gain_to_next is None
    assert _array_bits(np.diag(recorded.epochs[1].transition_from_previous)) == (
        _golden_bits(golden["recorded_transition_diagonal"])
    )
    for epoch, expected in zip(smoothed.epochs, golden["smoothed"]):
        assert _array_bits(epoch.snapshot.state.nominal.position_ecef_m) == (
            _golden_bits(expected["position_ecef_m"])
        )
        assert _array_bits(epoch.error_state_correction) == _golden_bits(
            expected["error_state_correction"]
        )
        assert _array_bits(np.diag(epoch.covariance)) == _golden_bits(
            expected["covariance_diagonal"]
        )


def test_joint_innovation_gate_report_from_loose_update():
    options = sidereon.EkfUpdateOptions(joint_innovation_gate_probability=0.01)
    config = sidereon.InertialFilterConfig(
        sidereon.ImuSpec.mems(),
        loose=sidereon.LooseCouplingConfig(update_options=options),
    )
    filter_ = sidereon.InertialFilter.with_config(_filter_state(), config)
    update = filter_.update_loose(_zero_fix(0.0, 1.0))
    gate = update.ekf.joint_innovation_gate
    assert gate.statistic == 0.0
    assert gate.degrees_of_freedom == 3
    assert gate.threshold == pytest.approx(11.344866730144373)
    assert gate.false_alarm_probability == 0.01
    assert gate.accepted is True
    assert update.applied is True


def test_ukf_joint_gate_uses_probability():
    config = sidereon.InertialFilterConfig(
        sidereon.ImuSpec.mems(),
        filter_kind=sidereon.FusionFilterKind.UKF,
        ukf_update_options=sidereon.UkfUpdateOptions(
            joint_innovation_gate_probability=0.01
        ),
    )
    update = sidereon.InertialFilter.with_config(_filter_state(), config).update_loose(
        _zero_fix(0.0, 1.0)
    )
    gate = update.ekf.joint_innovation_gate
    assert gate.degrees_of_freedom == 3
    assert gate.threshold == pytest.approx(11.344866730144373)
    assert gate.false_alarm_probability == 0.01
    assert gate.accepted is True
    assert gate.statistic == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize(
    "kind", [sidereon.FusionFilterKind.EKF, sidereon.FusionFilterKind.UKF]
)
def test_igg_joint_gate_uses_raw_covariance_core_oracle(kind):
    from _helpers import CORE_FIXTURES

    with open(os.path.join(CORE_FIXTURES, "fusion_gate_oracles.json")) as handle:
        oracle = json.load(handle)["raw_gate_after_igg_removal"]
    v = oracle["innovation"]
    measurement = sidereon.GnssFixMeasurement.position_velocity(
        0.0, [6_378_137.0 + v[0], v[1], v[2]], v[3:], np.eye(6), 8
    )
    config = sidereon.InertialFilterConfig(
        sidereon.ImuSpec.mems(),
        filter_kind=kind,
        loose=sidereon.LooseCouplingConfig(
            update_options=(
                sidereon.EkfUpdateOptions(joint_innovation_gate_probability=0.01)
                if kind == sidereon.FusionFilterKind.EKF
                else None
            ),
            measurement_reweighting=sidereon.IggIiiMeasurementReweighting.standard(),
        ),
        ukf_update_options=(
            sidereon.UkfUpdateOptions(joint_innovation_gate_probability=0.01)
            if kind == sidereon.FusionFilterKind.UKF
            else None
        ),
    )
    update = sidereon.InertialFilter.with_config(_filter_state(), config).update_loose(
        measurement
    )
    assert (update.applied, update.accepted_rows, update.rejected_rows) == (False, 0, 6)
    assert update.ekf.joint_innovation_gate.statistic == pytest.approx(16.0, abs=1e-12)
    assert update.ekf.joint_innovation_gate.degrees_of_freedom == 1
    assert update.ekf.joint_innovation_gate.accepted is False


def test_loose_innovation_covariance_and_nis_from_literal_matrices():
    # Position-only measurement: H selects the first three error states.
    h = np.zeros((3, 15))
    h[:, :3] = np.eye(3)
    p = np.eye(15)
    r = np.diag([2.0, 3.0, 4.0])
    innovation = np.array([1.0, 2.0, 3.0])
    expected_s = h @ p @ h.T + r
    expected_nis = innovation @ np.linalg.solve(expected_s, innovation)
    loose = sidereon.LooseCouplingConfig(
        update_options=sidereon.EkfUpdateOptions(sidereon.InnovationGate(100.0, 1))
    )
    filt = sidereon.InertialFilter.with_config(
        _filter_state(),
        sidereon.InertialFilterConfig(sidereon.ImuSpec.mems(), loose=loose),
    )
    report = filt.update_loose(
        sidereon.GnssFixMeasurement.position(0.0, [6_378_138.0, 2.0, 3.0], r, 8)
    )
    np.testing.assert_array_equal(report.ekf.innovation_covariance, expected_s)
    assert report.nis == pytest.approx(expected_nis, rel=1e-15)
    assert report.ekf.innovation_gate.max_abs_normalized_innovation == pytest.approx(
        np.max(np.abs(innovation) / np.sqrt(np.diag(expected_s))), rel=1e-15
    )


@pytest.mark.parametrize(
    "kind", [sidereon.FusionFilterKind.EKF, sidereon.FusionFilterKind.UKF]
)
@pytest.mark.parametrize("gate_kind", ["joint", "row"])
def test_igg_all_rows_removed_coasts(kind, gate_kind):
    row = sidereon.InnovationGate(3.0, 1) if gate_kind == "row" else None
    probability = 0.01 if gate_kind == "joint" else None
    ekf = sidereon.EkfUpdateOptions(row, probability)
    ukf = sidereon.UkfUpdateOptions(
        innovation_gate=row, joint_innovation_gate_probability=probability
    )
    config = sidereon.InertialFilterConfig(
        sidereon.ImuSpec.mems(),
        filter_kind=kind,
        loose=sidereon.LooseCouplingConfig(
            update_options=ekf if kind == sidereon.FusionFilterKind.EKF else None,
            measurement_reweighting=sidereon.IggIiiMeasurementReweighting.standard(),
        ),
        ukf_update_options=ukf if kind == sidereon.FusionFilterKind.UKF else None,
    )
    filt = sidereon.InertialFilter.with_config(_filter_state(), config)
    before = filt.state.covariance.copy()
    measurement = sidereon.GnssFixMeasurement.position_velocity(
        0.0, [6_378_187.0, 50.0, 50.0], [50.0] * 3, np.eye(6), 8
    )
    update = filt.update_loose(measurement)
    assert (update.applied, update.accepted_rows, update.rejected_rows) == (False, 0, 6)
    assert update.ekf.joint_innovation_gate is None
    if gate_kind == "row":
        gate = update.ekf.innovation_gate
        assert (
            gate.input_rows,
            gate.accepted_rows,
            gate.rejected_rows,
            gate.coasted,
        ) == (6, 0, 6, True)
        assert gate.max_abs_normalized_innovation is None
    np.testing.assert_array_equal(filt.state.covariance, before)


@pytest.mark.parametrize("field", ["ukf", "loose", "tight"])
@pytest.mark.parametrize("gate_kind", ["row", "joint"])
def test_options_for_other_filter_kind_are_refused(field, gate_kind):
    gate = sidereon.InnovationGate(3.0, 1) if gate_kind == "row" else None
    probability = 0.01 if gate_kind == "joint" else None
    kwargs = {}
    if field == "ukf":
        kwargs["ukf_update_options"] = sidereon.UkfUpdateOptions(
            innovation_gate=gate, joint_innovation_gate_probability=probability
        )
        expected = "ukf_update_options"
    else:
        kwargs["filter_kind"] = sidereon.FusionFilterKind.UKF
        options = sidereon.EkfUpdateOptions(gate, probability)
        kwargs[field] = (
            sidereon.LooseCouplingConfig
            if field == "loose"
            else sidereon.TightCouplingConfig
        )(update_options=options)
        expected = field + ".update_options"
    with pytest.raises(ValueError) as exc:
        sidereon.InertialFilterConfig(sidereon.ImuSpec.mems(), **kwargs)
    assert exc.value.detail["kind"] == "invalid_input"
    assert exc.value.detail["field"] == expected


@pytest.mark.parametrize("area", ["loose", "tight", "ukf"])
@pytest.mark.parametrize("probability", [0.0, 1.0, -0.1, float("inf"), float("nan")])
def test_invalid_joint_gate_probability_refused(area, probability):
    with pytest.raises(ValueError) as exc:
        if area != "ukf":
            options = sidereon.EkfUpdateOptions(
                joint_innovation_gate_probability=probability
            )
            config_type = (
                sidereon.LooseCouplingConfig
                if area == "loose"
                else sidereon.TightCouplingConfig
            )
            config = config_type(update_options=options)
            sidereon.InertialFilterConfig(sidereon.ImuSpec.mems(), **{area: config})
        else:
            sidereon.InertialFilterConfig(
                sidereon.ImuSpec.mems(),
                filter_kind=sidereon.FusionFilterKind.UKF,
                ukf_update_options=sidereon.UkfUpdateOptions(
                    joint_innovation_gate_probability=probability
                ),
            )
    assert exc.value.detail["kind"] == "invalid_input"
    assert exc.value.detail["field"] == "joint_innovation_gate_probability"


def _tight_source():
    return sidereon.PreciseEphemerisInterpolant.from_samples(
        [
            sidereon.PreciseEphemerisSample(
                "G01", float(t), [6_378_169.0, 0.0, 0.0], 0.0
            )
            for t in range(-8, 9)
        ]
    )


def _tight_epoch(code_innovation, rate_innovation, time=0.0):
    return sidereon.TightGnssEpoch(
        time,
        [
            sidereon.TightGnssObservation(
                "G01",
                32.0 + code_innovation,
                1.0,
                range_rate=sidereon.TightRangeRateObservation(
                    rate_innovation, 2.0, 0.0
                ),
            )
        ],
    )


@pytest.mark.parametrize(
    "kind", [sidereon.FusionFilterKind.EKF, sidereon.FusionFilterKind.UKF]
)
@pytest.mark.parametrize(
    "innovation,accepted", [([1.0, 2.0], True), ([8.0, 9.0], False)]
)
def test_tight_joint_gate_matches_literal_clock_model(kind, innovation, accepted):
    h = np.eye(2)
    p = np.diag([2.0, 3.0])
    r = np.diag([1.0, 4.0])
    expected_s = h @ p @ h.T + r
    expected_nis = np.array(innovation) @ np.linalg.solve(expected_s, innovation)
    state = sidereon.InsFilterState.from_diagonal(
        sidereon.NavState(0.0, [6_378_137.0, 0.0, 0.0], [0.0] * 3),
        sidereon.ErrorStateLayout.FIFTEEN,
        [0.0] * 15,
    )
    tight = sidereon.TightCouplingConfig(
        light_time=False,
        sagnac=False,
        initial_clock_bias_variance_m2=2.0,
        initial_clock_drift_variance_m2_s2=3.0,
        update_options=sidereon.EkfUpdateOptions(joint_innovation_gate_probability=0.01)
        if kind == sidereon.FusionFilterKind.EKF
        else None,
    )
    config = sidereon.InertialFilterConfig(
        sidereon.ImuSpec.mems(),
        filter_kind=kind,
        tight=tight,
        ukf_update_options=sidereon.UkfUpdateOptions(
            joint_innovation_gate_probability=0.01
        )
        if kind == sidereon.FusionFilterKind.UKF
        else None,
    )
    filt = sidereon.InertialFilter.with_config(state, config)
    before = filt.state.covariance.copy()
    before_position = filt.state.nominal.position_ecef_m.copy()
    before_velocity = filt.state.nominal.velocity_ecef_mps.copy()
    before_clock = filt.tight_clock_state().covariance.copy()
    update = filt.update_tight(_tight_source(), _tight_epoch(*innovation))
    gate = update.ekf.joint_innovation_gate
    assert gate.statistic == pytest.approx(expected_nis, rel=1e-7, abs=1e-7)
    assert gate.degrees_of_freedom == 2
    assert gate.threshold == pytest.approx(9.21034037197618, rel=1e-14)
    assert gate.false_alarm_probability == 0.01
    assert gate.accepted is accepted
    assert update.applied is accepted
    assert (update.accepted_rows, update.rejected_rows) == (
        (2, 0) if accepted else (0, 2)
    )
    if not accepted:
        np.testing.assert_array_equal(filt.state.covariance, before)
        np.testing.assert_array_equal(
            filt.state.nominal.position_ecef_m, before_position
        )
        np.testing.assert_array_equal(
            filt.state.nominal.velocity_ecef_mps, before_velocity
        )
        np.testing.assert_array_equal(filt.tight_clock_state().covariance, before_clock)


def test_rejected_loose_update_preserves_tight_cross_covariance():
    config = sidereon.InertialFilterConfig(
        sidereon.ImuSpec.mems(),
        loose=sidereon.LooseCouplingConfig(
            update_options=sidereon.EkfUpdateOptions(
                joint_innovation_gate_probability=0.01
            )
        ),
        tight=sidereon.TightCouplingConfig(
            light_time=False,
            sagnac=False,
            initial_clock_bias_variance_m2=2.0,
            initial_clock_drift_variance_m2_s2=3.0,
        ),
    )
    source = _tight_source()
    reference = sidereon.InertialFilter.with_config(_filter_state(), config)
    subject = sidereon.InertialFilter.with_config(_filter_state(), config)
    for filt in (reference, subject):
        assert filt.update_tight(source, _tight_epoch(1.0, 2.0)).applied
    for filt in (reference, subject):
        filt.propagate(sidereon.ImuSample.rate(0.5, [0.0] * 3, [0.0] * 3))
    rejected = subject.update_loose(
        sidereon.GnssFixMeasurement.position(
            0.5, [6_378_237.0, 100.0, 100.0], np.eye(3), 8
        )
    )
    assert not rejected.applied
    assert not rejected.ekf.joint_innovation_gate.accepted
    for filt in (reference, subject):
        filt.propagate(sidereon.ImuSample.rate(1.0, [0.0] * 3, [0.0] * 3))
    reference_update = reference.update_tight(source, _tight_epoch(1.0, 2.0, 1.0))
    subject_update = subject.update_tight(source, _tight_epoch(1.0, 2.0, 1.0))
    np.testing.assert_array_equal(subject_update.ekf.dx, reference_update.ekf.dx)
    np.testing.assert_array_equal(subject.state.covariance, reference.state.covariance)
    np.testing.assert_array_equal(
        subject.tight_clock_state().covariance, reference.tight_clock_state().covariance
    )


@pytest.mark.parametrize("area", ["loose", "tight", "ukf"])
def test_row_and_joint_gates_are_mutually_exclusive(area):
    row = sidereon.InnovationGate(3.0, 1)
    with pytest.raises(ValueError) as exc:
        if area == "ukf":
            options = sidereon.UkfUpdateOptions(
                innovation_gate=row, joint_innovation_gate_probability=0.01
            )
            sidereon.InertialFilterConfig(
                sidereon.ImuSpec.mems(),
                filter_kind=sidereon.FusionFilterKind.UKF,
                ukf_update_options=options,
            )
        else:
            options = sidereon.EkfUpdateOptions(row, 0.01)
            (
                sidereon.LooseCouplingConfig
                if area == "loose"
                else sidereon.TightCouplingConfig
            )(update_options=options)
    assert exc.value.detail["kind"] == "invalid_input"
    assert exc.value.detail["field"] == "innovation_gate"
