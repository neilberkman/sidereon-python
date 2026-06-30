"""DOP local-ENU convention binding tests."""

import math

import numpy as np
import pytest
import sidereon


def _unit_rows():
    los = np.array(
        [[0.1, 0.1, 0.99], [0.9, 0.0, 0.4], [-0.5, 0.7, 0.5], [0.0, -0.9, 0.4]]
    )
    return los / np.linalg.norm(los, axis=1, keepdims=True)


def _receiver():
    return sidereon.Wgs84Geodetic(math.radians(45.0), math.radians(10.0), 100.0)


def test_convention_invariants():
    los = _unit_rows()
    rec = _receiver()
    geodetic = sidereon.gnss_dop_with_convention(los, rec, "geodetic_normal")
    geocentric = sidereon.gnss_dop_with_convention(
        los, rec, sidereon.EnuConvention.GEOCENTRIC_RADIAL
    )
    # GDOP/PDOP/TDOP are rotation-invariant; only the horizontal/vertical split
    # changes between conventions.
    assert geodetic.gdop == pytest.approx(geocentric.gdop)
    assert geodetic.pdop == pytest.approx(geocentric.pdop)
    assert geodetic.tdop == pytest.approx(geocentric.tdop)
    assert geodetic.hdop != geocentric.hdop


def test_geodetic_normal_matches_plain_gnss_dop():
    los = _unit_rows()
    rec = _receiver()
    weights = np.ones(len(los))
    plain = sidereon.gnss_dop(los, weights, rec)
    convention = sidereon.gnss_dop_with_convention(los, rec, "geodetic_normal")
    assert plain.hdop == pytest.approx(convention.hdop)
    assert plain.vdop == pytest.approx(convention.vdop)


def test_enum_and_string_alias_agree():
    los = _unit_rows()
    rec = _receiver()
    by_enum = sidereon.gnss_dop_with_convention(
        los, rec, sidereon.EnuConvention.GEODETIC_NORMAL
    )
    by_str = sidereon.gnss_dop_with_convention(los, rec, "geodetic_normal")
    assert by_enum.gdop == pytest.approx(by_str.gdop)
    assert sidereon.EnuConvention.GEOCENTRIC_RADIAL.label == "geocentric_radial"


def test_errors():
    rec = _receiver()
    with pytest.raises(ValueError, match="unknown ENU convention"):
        sidereon.gnss_dop_with_convention(_unit_rows(), rec, "spherical")
    with pytest.raises(ValueError, match="four"):
        sidereon.gnss_dop_with_convention(_unit_rows()[:3], rec, "geodetic_normal")
