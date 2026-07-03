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


def _row_for_system(rows, system):
    return next(row for row in rows if row.system == system)


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
    assert report.clock_jumps == []
    assert report.notes == []

    first_sat = report.satellites[0]
    assert (first_sat.satellite, first_sat.epochs_with_observations) == ("G02", 3)
    assert first_sat.value_observations == 9

    first_signal = report.satellite_signals[0]
    assert (first_signal.satellite, first_signal.code) == ("G02", "C1C")
    assert first_signal.value_observations == 3
    assert first_signal.ssi.counts == [0, 0, 0, 2, 1, 0, 0, 0, 0, 0]

    slips = report.cycle_slips
    assert slips.observations == 4135
    assert slips.total_slips == 27
    assert slips.observations_per_slip == pytest.approx(4135.0 / 27.0)
    gps_slips = _row_for_system(slips.by_system, sidereon.GnssSystem.GPS)
    assert gps_slips.observations == 1282
    assert gps_slips.slips == 4
    assert gps_slips.observations_per_slip == pytest.approx(1282.0 / 4.0)
    glonass_slips = _row_for_system(slips.by_system, sidereon.GnssSystem.GLONASS)
    assert glonass_slips.observations == 784
    assert glonass_slips.slips == 10
    galileo_slips = _row_for_system(slips.by_system, sidereon.GnssSystem.GALILEO)
    assert galileo_slips.observations == 1023
    assert galileo_slips.slips == 9
    beidou_slips = _row_for_system(slips.by_system, sidereon.GnssSystem.BEIDOU)
    assert beidou_slips.observations == 1046
    assert beidou_slips.slips == 4

    gps_mp = _row_for_system(report.multipath.systems, sidereon.GnssSystem.GPS)
    assert gps_mp.mp1.n == 1282
    assert gps_mp.mp1.rms_m == pytest.approx(0.29240479301672934)
    assert gps_mp.mp2.n == 1282
    assert gps_mp.mp2.rms_m == pytest.approx(0.28099636987578613)
    beidou_mp = _row_for_system(report.multipath.systems, sidereon.GnssSystem.BEIDOU)
    assert beidou_mp.mp1.n == 1046
    assert beidou_mp.mp1.rms_m == pytest.approx(1.0173872172139768)
    assert beidou_mp.mp2.n == 1046
    assert beidou_mp.mp2.rms_m == pytest.approx(1.1736185873490712)
    assert any(
        row.satellite.startswith("G") and row.mp1 is not None and row.mp2 is not None
        for row in report.multipath.satellites
    )

    rendered = report.render_text()
    assert "PER-CONSTELLATION" in rendered
    assert "G   GPS" in rendered
    assert "R   GLONASS" in rendered
    assert "E   Galileo" in rendered
    assert "C   BeiDou" in rendered
    assert "S   SBAS" in rendered
    assert "MP1 RMS" in rendered
    assert "MP2 RMS" in rendered
    assert "SLIPS" in rendered
    assert "<td class=\"text\">GPS</td>" in report.render_html()

    encoded = json.loads(report.to_json())
    assert encoded["cycle_slips"]["total_slips"] == 27
    assert encoded["multipath"]["systems"]


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
