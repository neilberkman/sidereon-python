"""Public producer checks for all three core TimeOffsetError variants."""

import pytest
import sidereon


def _assert_time_offset(error, kind, scale, message, code):
    assert type(error) is ValueError
    assert str(error) == message
    assert error.code == code
    assert error.detail == {
        "family": "TimeOffsetError",
        "kind": kind,
        "message": message,
        "scale": scale,
    }


def test_epoch_required_offset_retains_scale_and_legacy_error():
    with pytest.raises(ValueError) as caught:
        sidereon.timescale_offset(sidereon.TimeScale.GPST, sidereon.TimeScale.UTC)

    _assert_time_offset(
        caught.value,
        "epoch_required",
        "UTC",
        (
            "time-scale UTC is UTC-based; its offset is epoch-dependent, "
            "use timescale_offset_at_s"
        ),
        sidereon.TimeOffsetErrorCode.EPOCH_REQUIRED,
    )


def test_unsupported_offset_retains_scale_and_legacy_error():
    with pytest.raises(ValueError) as caught:
        sidereon.timescale_offset(sidereon.TimeScale.GPST, sidereon.TimeScale.TDB)

    _assert_time_offset(
        caught.value,
        "unsupported",
        "TDB",
        "time-scale TDB has no fixed/constant offset; resolve it through TimeScales",
        sidereon.TimeOffsetErrorCode.UNSUPPORTED,
    )


def test_non_finite_epoch_offset_retains_scale_and_legacy_error():
    with pytest.raises(ValueError) as caught:
        sidereon.timescale_offset_at(
            sidereon.TimeScale.UTC, sidereon.TimeScale.TAI, float("nan")
        )

    _assert_time_offset(
        caught.value,
        "non_finite_epoch",
        "UTC",
        "utc_jd must be finite to resolve leap seconds for scale UTC",
        sidereon.TimeOffsetErrorCode.NON_FINITE_EPOCH,
    )


def test_time_offset_details_retain_every_supported_scale_label():
    for scale_name in ("UTC", "GLONASST"):
        scale = getattr(sidereon.TimeScale, scale_name)
        with pytest.raises(ValueError) as caught:
            sidereon.timescale_offset(sidereon.TimeScale.GPST, scale)
        message = (
            f"time-scale {scale_name} is UTC-based; its offset is epoch-dependent, "
            "use timescale_offset_at_s"
        )
        _assert_time_offset(
            caught.value,
            "epoch_required",
            scale_name,
            message,
            sidereon.TimeOffsetErrorCode.EPOCH_REQUIRED,
        )

    for scale_name in ("TCG", "TDB", "TCB"):
        scale = getattr(sidereon.TimeScale, scale_name)
        with pytest.raises(ValueError) as caught:
            sidereon.timescale_offset(sidereon.TimeScale.GPST, scale)
        message = (
            f"time-scale {scale_name} has no fixed/constant offset; "
            "resolve it through TimeScales"
        )
        _assert_time_offset(
            caught.value,
            "unsupported",
            scale_name,
            message,
            sidereon.TimeOffsetErrorCode.UNSUPPORTED,
        )


def test_non_finite_epoch_offset_keeps_glonass_time_label():
    with pytest.raises(ValueError) as caught:
        sidereon.timescale_offset_at(
            sidereon.TimeScale.GLONASST, sidereon.TimeScale.TAI, float("inf")
        )

    _assert_time_offset(
        caught.value,
        "non_finite_epoch",
        "GLONASST",
        "utc_jd must be finite to resolve leap seconds for scale GLONASST",
        sidereon.TimeOffsetErrorCode.NON_FINITE_EPOCH,
    )
