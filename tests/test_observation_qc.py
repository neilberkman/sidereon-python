"""RINEX observation QC and repair wrappers use core fixtures."""

import json
import os

import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _fixture_text(rel):
    marker = "tests/fixtures/"
    assert rel.startswith(marker)
    with open(os.path.join(CORE_FIXTURES, rel[len(marker) :]), encoding="utf-8") as fh:
        return fh.read()


def test_observation_qc_matches_real_oracle_summary():
    oracle_path = os.path.join(CORE_FIXTURES, "qc", "observation_qc_real_oracles.json")
    with open(oracle_path, encoding="utf-8") as fh:
        oracle = json.load(fh)["fixtures"][0]

    obs = sidereon.parse_rinex_obs(_fixture_text(oracle["path"]))
    report = sidereon.observation_qc(obs)

    assert report.total_epoch_records == oracle["total_epoch_records"]
    assert report.observation_epochs == oracle["observation_epochs"]
    assert report.event_records == oracle["event_records"]
    assert report.power_failure_epochs == oracle["power_failure_epochs"]
    assert report.skipped_records == oracle["skipped_records"]
    assert report.interval_s == pytest.approx(oracle["interval_s"])
    assert report.interval_source == sidereon.IntervalSource.HEADER
    assert report.missing_epochs == oracle["missing_epochs"]
    assert len(report.satellites) == len(oracle["satellites"])
    assert len(report.satellite_signals) == len(oracle["satellite_signals"])
    assert len(report.system_signals) == len(oracle["system_signals"])
    assert report.notes == []

    first_sat = report.satellites[0]
    assert (first_sat.satellite, first_sat.epochs_with_observations) == ("G02", 3)
    assert first_sat.value_observations == 9

    first_signal = report.satellite_signals[0]
    assert (first_signal.satellite, first_signal.code) == ("G02", "C1C")
    assert first_signal.value_observations == 3
    assert first_signal.ssi.counts == [0, 0, 0, 2, 1, 0, 0, 0, 0, 0]


def test_rinex_obs_lint_and_repair_are_exposed():
    text = _fixture_text(
        "tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx"
    )

    lint = sidereon.lint_rinex_obs(text)
    assert lint.is_clean
    assert lint.findings == []

    repair = sidereon.repair_rinex_obs(
        text,
        sidereon.RinexRepairOptions(
            set_interval=True,
            set_time_of_last_obs=True,
            set_obs_counts=True,
            drop_empty_records=True,
            drop_unsupported=True,
        ),
    )
    assert [(action.id, action.message) for action in repair.actions] == [
        ("A4", "recomputed TIME OF LAST OBS"),
        ("A5", "recomputed observation count headers"),
    ]
    assert repair.remaining.is_clean
    assert repair.repaired.epoch_count == 120
    assert (
        repair.to_crinex_string().splitlines()[0][60:].strip() == "CRINEX VERS   / TYPE"
    )
