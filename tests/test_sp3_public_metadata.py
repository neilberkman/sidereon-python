"""Literal public projections for SP3 coverage, flags, and record accuracy."""

import json
import os
import pathlib

import pytest
import sidereon
from _helpers import CORE_FIXTURES, FIXTURES, hex_to_f64

with open(os.path.join(FIXTURES, "sp3_bodies.json")) as fixture_file:
    FX = json.load(fixture_file)


def _sp3():
    fixture_path = os.path.join(
        CORE_FIXTURES, "sp3", pathlib.Path(FX["sp3_fixture"]).name
    )
    with open(fixture_path, "rb") as source:
        return sidereon.load_sp3(source.read())


def _known(value):
    assert value.kind == "known"
    assert value.value is not None
    return value.value


def test_sp3_public_coverage_and_prediction_summary():
    sp3 = _sp3()
    expected_epochs = [hex_to_f64(value) for value in FX["epochs_j2000_seconds_hex"]]

    coverage = sp3.satellite_coverage()
    assert coverage.grid.interval_s == 900.0
    assert coverage.grid.agrees_with_header is True
    assert coverage.grid.out_of_order == []
    assert coverage.grid.unplaced == []
    assert len(coverage.satellites) == 31
    gps01 = next(row for row in coverage.satellites if row.satellite == "G01")
    assert gps01.declared is True
    assert gps01.positions.epochs == len(expected_epochs) == 11
    assert gps01.positions.is_complete is True
    assert gps01.clocks.epochs == 11
    assert gps01.clocks.is_complete is True
    assert len(gps01.positions.spans) == 1
    span = gps01.positions.spans[0]
    assert (span.first_index, span.last_index) == (
        0,
        10,
    )
    assert gps01.positions.gaps == []

    prediction = sp3.prediction_summary()
    assert len(prediction.epochs) == 11
    assert prediction.observed_through_j2000_seconds == expected_epochs[-1]
    assert [epoch.epoch_j2000_seconds for epoch in prediction.epochs] == expected_epochs
    assert all(epoch.observed for epoch in prediction.epochs)
    assert all(epoch.orbit_predicted_satellites == [] for epoch in prediction.epochs)
    assert all(epoch.clock_predicted_satellites == [] for epoch in prediction.epochs)


def test_sp3_public_record_accuracy_decodes_literal_record_codes():
    sp3 = _sp3()

    raw = sp3.record_accuracy_codes("G01", 0)
    assert raw.v is None
    assert raw.p.axis_exponents == [3, 4, 5]
    assert raw.p.clock_exponent == 63
    assert raw.p.position_velocity_base == 1.25
    assert raw.p.clock_rate_base == 1.025

    decoded = sp3.record_accuracy("G01", 0)
    assert decoded.v is None
    sigmas = [1.25**3 * 1e-3, 1.25**4 * 1e-3, 1.25**5 * 1e-3]
    assert [_known(value) for value in decoded.p.position_sigma_m] == pytest.approx(
        sigmas
    )
    assert _known(decoded.p.clock_sigma_m) == pytest.approx(
        1.025**63 * 299_792_458.0 * 1e-12, rel=1e-14
    )
    assert [
        _known(value) for value in decoded.p.position_variance_m2()
    ] == pytest.approx([value**2 for value in sigmas])
