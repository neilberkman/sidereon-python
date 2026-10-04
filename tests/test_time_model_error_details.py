"""Public producer checks for owned TimeModelError detail."""

import os

import pytest
import sidereon
from _helpers import CORE_FIXTURES


def _load_comparison_products():
    broadcast = sidereon.load_rinex_nav(
        os.path.join(CORE_FIXTURES, "nav/ESBC00DNK_R_20201770000_01D_MN.rnx")
    )
    precise_path = os.path.join(
        CORE_FIXTURES, "sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    )
    with open(precise_path, "rb") as stream:
        precise = sidereon.load_sp3(stream.read())
    return broadcast, precise


def _assert_time_model_error(error, field, reason, legacy_text):
    assert type(error) is ValueError
    assert str(error) == legacy_text
    assert error.detail == {
        "family": "TimeModelError",
        "kind": "invalid_input",
        "message": legacy_text.removeprefix("sample_interval_s: ")
        if legacy_text.startswith("sample_interval_s: ")
        else legacy_text.removeprefix("invalid epoch_j2000_s: ").removeprefix(
            "invalid epoch: "
        ),
        "field": field,
        "reason": reason,
    }


def test_clock_instant_split_refusal_keeps_typed_time_model_fields():
    with pytest.raises(ValueError) as caught:
        sidereon.ClockInstant.from_split_julian_date(
            sidereon.TimeScale.GPST, 2_451_545.0, float("inf")
        )

    _assert_time_model_error(
        caught.value,
        "fraction",
        "must be finite",
        "invalid time model fraction: must be finite",
    )


def test_ionosphere_civil_epoch_refusal_keeps_typed_time_model_fields():
    with pytest.raises(ValueError) as caught:
        sidereon.ionosphere_delay_klobuchar(
            [0.0] * 4,
            [0.0] * 4,
            lat_deg=40.0,
            lon_deg=-3.0,
            azimuth_deg=120.0,
            elevation_deg=30.0,
            year=2020,
            month=6,
            day=24,
            hour=12,
            minute=0,
            second=86_401.0,
            frequency_hz=1_575.42e6,
        )

    _assert_time_model_error(
        caught.value,
        "fraction",
        "must be within one residual day",
        "invalid epoch: invalid time model fraction: must be within one residual day",
    )


def test_sidereal_duration_refusal_keeps_prefixed_legacy_text_and_fields():
    with pytest.raises(ValueError) as caught:
        sidereon.SiderealFilterOptions(float("nan"))

    _assert_time_model_error(
        caught.value,
        "seconds",
        "must be finite",
        "sample_interval_s: invalid time model seconds: must be finite",
    )


def test_broadcast_comparison_window_refusal_keeps_typed_split_fields():
    broadcast, precise = _load_comparison_products()
    with pytest.raises(ValueError) as caught:
        sidereon.broadcast_comparison_window(
            broadcast,
            precise,
            ["G01"],
            0.0,
            0.0,
            2_451_545.0,
            float("inf"),
            1.0,
        )

    _assert_time_model_error(
        caught.value,
        "fraction",
        "must be finite",
        "invalid time model fraction: must be finite",
    )


def test_broadcast_comparison_epoch_refusal_keeps_typed_split_fields():
    broadcast, precise = _load_comparison_products()
    with pytest.raises(ValueError) as caught:
        sidereon.broadcast_comparison(broadcast, precise, ["G01"], [float("inf")], 1.0)

    _assert_time_model_error(
        caught.value,
        "jd_whole",
        "must be finite",
        "invalid time model jd_whole: must be finite",
    )


def test_gnss_week_tow_constructor_normalize_and_rollover_keep_full_detail():
    with pytest.raises(ValueError) as caught:
        sidereon.GnssWeekTow(sidereon.TimeScale.GPST, 1, float("nan"))
    _assert_time_model_error(
        caught.value,
        "tow_s",
        "must be finite",
        "invalid time model tow_s: must be finite",
    )

    week = sidereon.GnssWeekTow(sidereon.TimeScale.GPST, 0xFFFFFFFF, 604_800.0)
    with pytest.raises(ValueError) as caught:
        week.normalized()
    _assert_time_model_error(
        caught.value,
        "tow_s",
        "normalized week is out of range",
        "invalid time model tow_s: normalized week is out of range",
    )

    week = sidereon.GnssWeekTow(sidereon.TimeScale.GPST, 0xFFFFFFFF, 0.0)
    with pytest.raises(ValueError) as caught:
        week.unrolled_week(1)
    _assert_time_model_error(
        caught.value,
        "rollovers",
        "unrolled week is out of range",
        "invalid time model rollovers: unrolled week is out of range",
    )
