"""Inverse SGP4 TLE fitting delegates to the core fitter."""

import datetime as dt
import math

import numpy as np
import pytest
import sidereon
from _helpers import core_goldens, hex_to_f64

LINE1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
LINE2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"


def _tle_epoch_datetime():
    return dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(
        days=184.80969102 - 1.0
    )


def _jd_pair(instant):
    year = instant.year
    month = instant.month
    day = instant.day
    seconds = (
        instant.hour * 3600
        + instant.minute * 60
        + instant.second
        + instant.microsecond / 1.0e6
    )
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    jd0 = (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + b
        - 1524.5
    )
    jd = jd0 + seconds / 86400.0
    whole = math.floor(jd)
    return whole, jd - whole


def _samples(offsets_s=None):
    if offsets_s is None:
        offsets_s = [-1800, 0, 1800, 3600, 5400, 7200]
    instants = [_tle_epoch_datetime() + dt.timedelta(seconds=s) for s in offsets_s]
    epochs_unix_us = np.array(
        [int(instant.timestamp() * 1_000_000) for instant in instants],
        dtype=np.int64,
    )
    propagation = sidereon.Tle(LINE1, LINE2, sidereon.OpsMode.IMPROVED).propagate(
        epochs_unix_us
    )
    out = []
    for instant, position, velocity in zip(
        instants, propagation.position_km, propagation.velocity_km_s
    ):
        jd_whole, jd_fraction = _jd_pair(instant)
        out.append(
            sidereon.FitSample(
                jd_whole,
                jd_fraction,
                position.tolist(),
                velocity.tolist(),
            )
        )
    return out


def test_fit_tle_round_trips_arc_and_omm_json():
    fit = sidereon.fit_tle(
        _samples(),
        epoch=sidereon.FitEpoch.sample(1),
        fit_bstar=False,
        use_velocity=True,
        metadata=sidereon.TleFitMetadata(
            catalog_number=25544,
            international_designator="98067A",
            object_name="ISS",
        ),
    )

    # A fit with `fit_bstar=False` holds a zero B* and second derivative, which
    # python-sgp4's `export_tle` spells `" 00000-0"` for the second derivative
    # and `" 00000+0"` for B*.
    assert fit.to_lines() == (
        "1 25544U 98067A   18184.80969102  .00000000  00000-0  00000+0 0  9996",
        "2 25544  51.6414 295.8524 0003435 262.6273 204.2861 15.54005769    06",
    )
    assert fit.catalog_number == 25544
    assert fit.omm_epoch_days is None
    assert fit.mean_motion_rev_per_day == pytest.approx(15.540057693132342)
    assert fit.eccentricity == pytest.approx(0.00034349193678013374)
    assert fit.inclination_deg == pytest.approx(51.64139999526309)

    stats = fit.stats
    expected = core_goldens()["tle_fit"]["stats"]
    assert stats.rms_position_km == hex_to_f64(expected["rms_position_km"])
    assert stats.max_position_km == hex_to_f64(expected["max_position_km"])
    assert stats.rms_velocity_km_s == hex_to_f64(expected["rms_velocity_km_s"])
    assert stats.tle_rms_position_km == hex_to_f64(expected["tle_rms_position_km"])
    assert stats.bstar_observable == expected["bstar_observable"]
    assert stats.status == expected["status"]
    assert stats.nfev == expected["nfev"]
    assert stats.njev == expected["njev"]
    assert stats.cost == hex_to_f64(expected["cost"])
    assert stats.optimality == hex_to_f64(expected["optimality"])
    assert stats.seed_refine_passes == expected["seed_refine_passes"]
    np.testing.assert_array_equal(
        stats.rms_position_axes_km,
        [hex_to_f64(value) for value in expected["rms_position_axes_km"]],
    )

    reparsed = sidereon.parse_omm_json(fit.omm.to_json_string())
    assert reparsed.epoch == fit.omm.epoch
    assert reparsed.norad_cat_id == 25544


def test_fit_tle_short_arc_preserves_variant_fields_and_message():
    with pytest.raises(sidereon.SolveError) as exc_info:
        sidereon.fit_tle(_samples()[:2])

    error = exc_info.value
    assert type(error) is sidereon.SolveError
    assert str(error) == "fit arc has 2 samples; need at least 3"
    assert error.detail == {
        "family": "TleFitError",
        "kind": "arc_too_short",
        "message": "fit arc has 2 samples; need at least 3",
        "samples": 2,
        "needed": 3,
    }


def test_fit_tle_did_not_converge_retains_complete_best_effort_fit():
    offsets_s = [index * 600 for index in range(-6, 7)]
    samples = _samples(offsets_s)

    with pytest.raises(sidereon.SolveError) as exc_info:
        sidereon.fit_tle(samples, max_nfev=2)

    error = exc_info.value
    assert type(error) is sidereon.SolveError
    assert str(error) == (
        "evaluation budget exhausted before convergence (best-effort result attached)"
    )
    assert error.detail["family"] == "TleFitError"
    assert error.detail["kind"] == "did_not_converge"
    assert error.detail["message"] == str(error)

    result_detail = error.detail["best_effort_fit"]
    fit = result_detail["value"]
    assert isinstance(fit, sidereon.TleFit)
    assert fit.stats.status == 0
    assert fit.line1 and fit.line2
    assert fit.omm_epoch_days is None
    assert result_detail["omm_exact_sgp4_epoch"] == (fit.jd_whole, fit.jd_fraction)
    assert result_detail["omm_quantize_tle_derived_fields"] is False
    assert result_detail["element_set_omm_epoch_days"] is None


def test_fit_tle_rejects_zero_evaluation_budget_with_typed_input_detail():
    with pytest.raises(sidereon.SolveError) as exc_info:
        sidereon.fit_tle(_samples(), max_nfev=0)

    error = exc_info.value
    assert str(error) == "fit input invalid: max_nfev: must be positive"
    assert error.detail == {
        "family": "TleFitError",
        "kind": "invalid_input",
        "message": "fit input invalid: max_nfev: must be positive",
        "field": "max_nfev",
        "reason": "must be positive",
    }


def test_fit_epoch_and_xscale_selectors_are_exposed():
    assert sidereon.FitEpoch.midpoint().kind == "midpoint"
    assert sidereon.FitEpoch.sample(2).sample_index == 2
    assert sidereon.FitEpoch.jd(2458303.0, 0.5).jd_pair == (2458303.0, 0.5)
    assert sidereon.XScale.unit().kind == "unit"
    assert sidereon.XScale.jac().kind == "jac"
    assert sidereon.XScale.values([1.0, 2.0]).scale_values == [1.0, 2.0]
    assert sidereon.Loss.SOFT_L1.label == "soft_l1"
