"""Structured core failure details on scalar and batched observable queries."""

import os

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

SP3_PATH = os.path.join(CORE_FIXTURES, "sp3", "GRG0MGXFIN_20201760000_01D_15M_ORB.SP3")


def test_observable_batch_keeps_fresh_owned_error_details():
    with open(SP3_PATH, "rb") as source:
        sp3 = sidereon.load_sp3(source.read())

    epoch = float(sp3.epochs_j2000_seconds[0])
    batch = sidereon.observable_states_at_shared_j2000_s(
        sp3, ["G01", "G99", "G01"], epoch
    )
    assert batch.statuses == [
        sidereon.ObservableStateElementStatus.VALID,
        sidereon.ObservableStateElementStatus.GAP,
        sidereon.ObservableStateElementStatus.VALID,
    ]
    assert batch.element_results[0] is None
    assert batch.element_results[1]
    assert batch.element_results[2] is None
    assert np.isnan(batch.positions_ecef_m[1]).all()
    assert batch.element_error_detail(0) is None
    assert batch.element_error_detail(2) is None
    source_nodes = sp3.interpolate("G01", np.asarray([epoch], dtype=np.float64))
    np.testing.assert_array_equal(batch.positions_ecef_m[0], source_nodes.position_m[0])
    assert batch.clocks_s[0] == source_nodes.clock_s[0]
    with pytest.raises(IndexError):
        batch.element_error_detail(batch.element_count)

    detail = batch.element_error_detail(1)
    assert detail["family"] == "ObservablesError"
    assert detail["kind"] == "ephemeris"
    assert detail["cause"]["family"] == "CoreError"
    assert detail["cause"]["kind"] == "unknown_satellite"
    assert detail["cause"]["satellite_id"] == "G99"
    assert detail["message"] == batch.element_results[1]
    assert batch.element_error_details[1] == detail

    detail["kind"] = "caller mutation"
    detail["cause"]["satellite_id"] = "G01"
    fresh = batch.element_error_detail(1)
    assert fresh["kind"] == "ephemeris"
    assert fresh["cause"]["satellite_id"] == "G99"

    exact_epoch = sidereon.ExactEpochQuery.from_binary_j2000_seconds(epoch)
    with pytest.raises(sidereon.SolveError) as query_refused:
        sp3.selected_state_at_epoch_query("G99", exact_epoch, exact_epoch)
    assert query_refused.value.detail["kind"] == "unknown_satellite"
    assert query_refused.value.detail["satellite_id"] == "G99"
    later_success = sidereon.observable_states_at_shared_j2000_s(sp3, ["G01"], epoch)
    assert later_success.element_error_details == [None]
    assert fresh["cause"]["satellite_id"] == "G99"

    listed = batch.element_error_details
    listed[1]["cause"]["satellite_id"] = "G02"
    assert batch.element_error_details[1]["cause"]["satellite_id"] == "G99"

    del batch
    del sp3
    assert fresh["cause"]["satellite_id"] == "G99"


def test_scalar_core_input_error_detail_keeps_value_error_compatibility():
    with open(SP3_PATH, "rb") as source:
        sp3 = sidereon.load_sp3(source.read())

    with pytest.raises(ValueError) as refused:
        sidereon.observe(
            sp3,
            "G01",
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            float("nan"),
            1_575_420_000.0,
        )

    detail = refused.value.detail
    assert detail["family"] == "ObservablesError"
    assert detail["kind"] == "invalid_input"
    assert detail["field"] == "t_rx_j2000_s"
    assert detail["reason"] == "non_finite"


def test_scalar_observable_domain_errors_keep_variant_payloads_and_classes():
    with pytest.raises(ValueError, match="equal carrier frequencies") as iono:
        sidereon.gamma(1.0, 1.0)
    assert iono.value.detail == {
        "family": "IonosphereFreeError",
        "kind": "equal_frequencies",
        "message": "equal carrier frequencies",
    }

    with pytest.raises(ValueError, match="carrier frequency must be positive") as phase:
        sidereon.phase_meters(1.0, 0.0)
    assert phase.value.detail["family"] == "CarrierPhaseError"
    assert phase.value.detail["kind"] == "invalid_frequency"
    assert phase.value.detail["message"] == str(phase.value)

    with pytest.raises(ValueError, match="unsupported GPS C/A PRN 0") as signal:
        sidereon.ca_code(0)
    assert signal.value.detail == {
        "family": "SignalError",
        "kind": "unsupported_prn",
        "message": str(signal.value),
        "prn": 0,
    }
    assert len(sidereon.ca_code(1)) == 1023
    huge_prn = 9_007_199_254_740_993
    with pytest.raises(
        ValueError, match=f"unsupported GPS C/A PRN {huge_prn}"
    ) as large_signal:
        sidereon.ca_code(huge_prn)
    assert large_signal.value.detail["prn"] == huge_prn

    bad_replica = sidereon.ReplicaOptions(
        sample_rate_hz=0.0,
        num_samples=4,
        code_phase_chips=0.0,
    )
    with pytest.raises(ValueError, match="sample_rate_hz") as signal_input:
        sidereon.replica(1, bad_replica)
    assert signal_input.value.detail["family"] == "SignalError"
    assert signal_input.value.detail["kind"] == "invalid_input"
    assert signal_input.value.detail["field"] == "sample_rate_hz"
    assert signal_input.value.detail["reason"] == "not positive"

    with pytest.raises(
        sidereon.SolveError, match="carrier_hz: not positive"
    ) as velocity:
        sidereon.doppler_to_range_rate(10.0, 0.0)
    assert velocity.value.detail == {
        "family": "VelocityError",
        "kind": "invalid_input",
        "message": str(velocity.value),
        "field": "carrier_hz",
        "reason": "not positive",
    }

    with open(SP3_PATH, "rb") as source:
        sp3 = sidereon.load_sp3(source.read())
    epoch = float(sp3.epochs_j2000_seconds[len(sp3.epochs_j2000_seconds) // 2])
    duplicate_rows = [
        sidereon.VelocityObservation("G01", 10.0, 1.57542e9),
        sidereon.VelocityObservation("G01", 11.0, 1.57542e9),
    ]
    with pytest.raises(
        sidereon.SolveError, match="duplicate observation for G01"
    ) as duplicate:
        sidereon.solve_velocity(
            sp3,
            duplicate_rows,
            np.asarray([3_582_105.291, 532_589.731, 5_232_754.805]),
            epoch,
        )
    assert duplicate.value.detail["family"] == "VelocityError"
    assert duplicate.value.detail["kind"] == "duplicate_observation"
    assert duplicate.value.detail["satellite_id"] == "G01"


def test_sp3_query_errors_keep_legacy_classes_messages_and_core_fields():
    with open(SP3_PATH, "rb") as source:
        sp3 = sidereon.load_sp3(source.read())

    epoch_seconds = float(sp3.epochs_j2000_seconds[0])
    epoch = sidereon.ExactEpochQuery.from_binary_j2000_seconds(epoch_seconds)

    with pytest.raises(
        ValueError, match="satellite G99 is not in the product"
    ) as interpolation:
        sp3.interpolate("G99", np.asarray([epoch_seconds], dtype=np.float64))
    assert interpolation.value.detail == {
        "family": "CoreError",
        "kind": "unknown_satellite",
        "message": "unknown satellite: G99",
        "satellite_id": "G99",
    }

    valid = sp3.position_at_epoch_query("G01", epoch)
    assert valid.position_m.shape == (3,)

    interpolant = sidereon.PreciseEphemerisInterpolant.from_sp3(sp3)
    artifact = sidereon.PreciseInterpolantArtifact.from_bytes(
        sp3.precise_interpolant_artifact_bytes()
    )
    core_backed_queries = (
        lambda: interpolant.position_at_j2000_seconds("G99", epoch_seconds),
        lambda: interpolant.position_at_epoch_query("G99", epoch),
        lambda: artifact.position_at_j2000_seconds("G99", epoch_seconds),
        lambda: artifact.position_at_epoch_query("G99", epoch),
    )
    for query in core_backed_queries:
        with pytest.raises(
            sidereon.SolveError, match="unknown satellite: G99"
        ) as query_refused:
            query()
        assert query_refused.value.detail == {
            "family": "CoreError",
            "kind": "unknown_satellite",
            "message": str(query_refused.value),
            "satellite_id": "G99",
        }

    with pytest.raises(sidereon.SolveError, match="unknown satellite: G99") as query:
        sp3.position_at_epoch_query("G99", epoch)
    assert query.value.detail["kind"] == "unknown_satellite"
    assert query.value.detail["satellite_id"] == "G99"

    gap_sp3 = sidereon.load_sp3(
        os.path.join(CORE_FIXTURES, "sp3", "GAP_G01_20201760000_15M.sp3")
    )
    gap_seconds = 646_260_300.0
    with pytest.raises(
        sidereon.SolveError, match="interpolation at j2000 second .*epoch out of range"
    ) as gap:
        gap_sp3.interpolate("G01", np.asarray([gap_seconds], dtype=np.float64))
    assert gap.value.detail["kind"] == "epoch_out_of_range"

    with pytest.raises(KeyError) as missing_record:
        sp3.state("G99", 0)
    assert missing_record.value.args[0] == "satellite G99 has no record at epoch 0"
    assert missing_record.value.detail["kind"] == "unknown_satellite"
    assert missing_record.value.detail["satellite_id"] == "G99"
    assert missing_record.value.detail["epoch_index"] == 0

    end = len(sp3.epochs_j2000_seconds)
    with pytest.raises(IndexError) as out_of_range:
        sp3.state("G01", end)
    assert out_of_range.value.args[0] == f"epoch index {end} out of range"
    assert out_of_range.value.detail["kind"] == "epoch_out_of_range"
    assert out_of_range.value.detail["epoch_index"] == end


def test_ephemeris_sample_preserves_solve_error_and_adds_observables_detail():
    with open(SP3_PATH, "rb") as source:
        sp3 = sidereon.load_sp3(source.read())

    start = float(sp3.epochs_j2000_seconds[0])
    with pytest.raises(
        sidereon.SolveError, match="invalid observable input step_s: not positive"
    ) as refused:
        sidereon.ephemeris_sample(sp3, ["G01"], start, start + 1.0, 0.0)

    assert refused.value.detail["family"] == "ObservablesError"
    assert refused.value.detail["kind"] == "invalid_input"
    assert refused.value.detail["field"] == "step_s"
    assert refused.value.detail["reason"] == "not_positive"

    rows = sidereon.ephemeris_sample(sp3, ["G01"], start, start, 1.0)
    assert len(rows) == 1
    assert rows[0].status == sidereon.EphemerisSampleStatus.VALID
