"""RF link-budget binding reproduces the engine numbers bit-for-bit."""

import json
import os

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES, hex_to_f64


def _fixture():
    with open(os.path.join(FIXTURES, "rf_link_budget.json")) as fh:
        return json.load(fh)


FX = _fixture()


def _bits(value):
    return np.float64(value).view(np.uint64)


def _expect_bits(hex_value):
    return np.uint64(int(hex_value, 16))


def test_fspl_matches_reference_bits():
    case = FX["fspl"]
    got = sidereon.fspl(case["distance_km"], case["frequency_mhz"])
    assert _bits(got) == _expect_bits(case["value_hex"])


def test_eirp_matches_reference_bits():
    case = FX["eirp"]
    got = sidereon.eirp(case["tx_power_dbm"], case["tx_antenna_gain_dbi"])
    assert _bits(got) == _expect_bits(case["value_hex"])


def test_cn0_matches_reference_bits():
    case = FX["cn0"]
    got = sidereon.cn0(
        case["eirp_dbw"],
        case["fspl_db"],
        case["receiver_gt_dbk"],
        case["other_losses_db"],
    )
    assert _bits(got) == _expect_bits(case["value_hex"])


def test_wavelength_matches_reference_bits():
    case = FX["wavelength"]
    got = sidereon.wavelength(case["frequency_hz"])
    assert _bits(got) == _expect_bits(case["value_hex"])


def test_dish_gain_matches_reference_bits():
    case = FX["dish_gain"]
    got = sidereon.dish_gain(
        case["diameter_m"], case["frequency_hz"], case["efficiency"]
    )
    assert _bits(got) == _expect_bits(case["value_hex"])


def test_link_margin_matches_reference_bits():
    case = FX["link_margin"]
    b = case["budget"]
    budget = sidereon.LinkBudget(
        eirp_dbw=b["eirp_dbw"],
        fspl_db=b["fspl_db"],
        receiver_gt_dbk=b["receiver_gt_dbk"],
        other_losses_db=b["other_losses_db"],
        required_cn0_dbhz=b["required_cn0_dbhz"],
    )
    assert budget == sidereon.LinkBudget(
        b["eirp_dbw"],
        b["fspl_db"],
        b["receiver_gt_dbk"],
        b["required_cn0_dbhz"],
        b["other_losses_db"],
    )
    assert "LinkBudget" in repr(budget)
    assert _bits(sidereon.link_margin(budget)) == _expect_bits(case["value_hex"])
    assert sidereon.link_margin(budget) == hex_to_f64(case["value_hex"])


def test_rf_bad_inputs_raise_value_error():
    with pytest.raises(ValueError):
        sidereon.fspl(0.0, 1616.0)
    with pytest.raises(ValueError):
        sidereon.wavelength(-1.0)
    with pytest.raises(ValueError):
        sidereon.dish_gain(1.0, 1616.0e6, 0.0)
    with pytest.raises(ValueError):
        sidereon.LinkBudget(np.inf, 165.0, -12.0, 35.0)
