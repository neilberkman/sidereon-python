"""Typed error details for public SPP and RINEX SPP entry points."""

from pathlib import Path

import pytest
import sidereon

ROOT = Path(__file__).parent
OBS_PATH = ROOT / "fixtures" / "obs" / "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
NAV_PATH = ROOT / "fixtures" / "nav" / "ESBC00DNK_R_20201770000_01D_MN.rnx"
MISMATCHED_SP3_PATH = (
    ROOT / "fixtures" / "sp3" / "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
)


def _invalid_spp_config():
    return sidereon.SppConfig(
        observations=[sidereon.SppObservation("G01", 22_000_000.0)],
        t_rx_j2000_s=float("nan"),
        t_rx_second_of_day_s=0.0,
        day_of_year=1.0,
        initial_guess=[6_378_137.0, 0.0, 0.0, 0.0],
    )


def _rinex_options(obs):
    return sidereon.RinexSppOptions(
        obs,
        signal_policy=sidereon.SignalPolicy([(sidereon.GnssSystem.GPS, ["C1C"])]),
        corrections=sidereon.SppCorrections(ionosphere=False, troposphere=False),
    )


def test_spp_and_static_errors_keep_core_fields_and_legacy_exceptions():
    sp3 = sidereon.load_sp3(MISMATCHED_SP3_PATH.read_bytes())
    config = _invalid_spp_config()

    with pytest.raises(sidereon.SolveError) as spp_error:
        sidereon.solve_spp(sp3, config)
    assert str(spp_error.value) == spp_error.value.detail["message"]
    assert spp_error.value.detail["family"] == "SolvePolicyError"
    assert spp_error.value.detail["kind"] == "solve"
    assert spp_error.value.detail["cause"] == {
        "family": "SppError",
        "kind": "invalid_input",
        "message": "invalid SPP input t_rx_j2000_s: not finite",
        "field": "t_rx_j2000_s",
        "reason": "non_finite",
    }

    with pytest.raises(sidereon.SolveError) as batch_error:
        sidereon.solve_spp_batch(sp3, [config, config])
    assert str(batch_error.value).startswith("epoch 0: ")
    assert batch_error.value.detail["epoch_index"] == 0
    assert batch_error.value.detail["message"] == str(batch_error.value)

    with pytest.raises(sidereon.SolveError) as static_error:
        sidereon.solve_static(sp3, [sidereon.StaticEpoch(config)])
    detail = static_error.value.detail
    assert str(static_error.value) == detail["message"]
    assert detail["family"] == "StaticSolveError"
    assert detail["kind"] == "epoch_input"
    assert detail["epoch_index"] == 0
    assert detail["cause"]["field"] == "t_rx_j2000_s"
    assert detail["cause"]["reason"] == "non_finite"

    with pytest.raises(sidereon.SolveError) as empty_static:
        sidereon.solve_static(sp3, [])
    assert empty_static.value.detail["kind"] == "empty_epochs"


def test_rinex_assembly_error_remains_value_error_with_owned_detail():
    source = NAV_PATH.read_bytes()
    text = OBS_PATH.read_text(encoding="utf-8")
    approx_line = next(
        line
        for line in text.splitlines()
        if line[60:80].strip() == "APPROX POSITION XYZ"
    )
    without_position = "\n".join(
        line for line in text.splitlines() if line != approx_line
    )
    obs = sidereon.parse_rinex_obs(without_position)

    with pytest.raises(ValueError) as captured:
        sidereon.spp_inputs_from_rinex_obs(
            sidereon.load_rinex_nav(source), obs, _rinex_options(obs)
        )

    detail = captured.value.detail
    assert str(captured.value) == detail["message"]
    assert detail["family"] == "RinexSppError"
    assert detail["kind"] == "missing_approx_position"

    # The options constructor selects a version-aware policy independently of
    # the approximate position; assembly reports the missing position later.
    assert isinstance(sidereon.RinexSppOptions(obs), sidereon.RinexSppOptions)


def test_rinex_epoch_details_survive_source_release_and_are_fresh():
    obs = sidereon.load_rinex_obs(OBS_PATH)
    options = _rinex_options(obs)
    mismatched = sidereon.load_sp3(MISMATCHED_SP3_PATH.read_bytes())
    failed = sidereon.solve_spp_from_rinex_obs(mismatched, obs, options)
    assert failed and all(not epoch.solved for epoch in failed)
    epoch = failed[0]
    legacy_message = epoch.error
    detail = epoch.error_detail
    assert detail is not None
    assert detail["family"] == "SolvePolicyError"
    assert detail["epoch_index"] == epoch.epoch_index
    assert detail["message"] == legacy_message
    detail["kind"] = "changed by caller"
    del mismatched, obs, options
    fresh_detail = epoch.error_detail
    assert fresh_detail["kind"] == "solve"
    fresh_snapshot = fresh_detail.copy()

    obs = sidereon.load_rinex_obs(OBS_PATH)
    nav = sidereon.load_rinex_nav(NAV_PATH)
    solved = sidereon.solve_spp_from_rinex_obs(nav, obs, _rinex_options(obs))
    assert solved and any(epoch.solved for epoch in solved)
    assert all(epoch.error_detail is None for epoch in solved if epoch.solved)
    assert fresh_detail == fresh_snapshot


def test_rinex_policy_validation_keeps_nested_core_error():
    obs = sidereon.load_rinex_obs(OBS_PATH)
    nav = sidereon.load_rinex_nav(NAV_PATH)
    options = _rinex_options(obs)

    baseline = sidereon.solve_spp_from_rinex_obs(nav, obs, options)
    baseline_solved = next(row for row in baseline if row.solved)
    rows = sidereon.solve_spp_from_rinex_obs(nav, obs, options, max_pdop=0.1)
    failed = next(row for row in rows if row.epoch_index == baseline_solved.epoch_index)
    assert not failed.solved
    detail = failed.error_detail
    assert detail is not None
    assert detail["family"] == "SolvePolicyError"
    assert detail["kind"] == "validation"
    assert detail["epoch_index"] == failed.epoch_index
    assert detail["cause"]["family"] == "SolutionValidationError"
    assert detail["cause"]["kind"] == "degenerate_geometry_pdop"
    assert detail["cause"]["pdop"] > 0.1
    assert len(detail["cause"]["pdop_bits"]) == 16
