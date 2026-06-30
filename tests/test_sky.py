"""Ground-observer Sun/Moon geometry binding tests.

Reference values come from the core's own Skyfield/de421-checked tolerances for
the low-precision analytic series (Royal Observatory, Greenwich)."""

import datetime

import pytest
import sidereon

# Royal Observatory, Greenwich (WGS84), altitude ~46 m.
LAT, LON, ALT_KM = 51.4769, 0.0, 0.046
# A finite but out-of-domain geodetic latitude (outside [-90, 90]).
BAD_LAT = 120.0


def _us(year, month, day, hour, minute, second):
    dt = datetime.datetime(
        year, month, day, hour, minute, second, tzinfo=datetime.timezone.utc
    )
    return int(dt.timestamp() * 1_000_000)


def test_sun_at_solar_transit_due_south_and_high():
    # Apparent solar upper transit at Greenwich on 2024-06-20 is 12:01:42 UTC.
    t = _us(2024, 6, 20, 12, 1, 42)
    look = sidereon.sun_az_el(LAT, LON, ALT_KM, t)
    assert abs(look.azimuth_deg - 180.0) < 0.5
    assert abs(look.elevation_deg - 61.96) < 0.5
    assert abs(look.range_km - 1.52011e8) < 5.0e5


def test_moon_at_transit_matches_reference():
    # Moon upper transit at Greenwich on 2024-04-23 is 23:55:59 UTC.
    t = _us(2024, 4, 23, 23, 55, 59)
    look = sidereon.moon_az_el(LAT, LON, ALT_KM, t)
    assert abs(look.azimuth_deg - 180.0) < 0.3
    assert abs(look.elevation_deg - 23.12) < 0.3
    assert abs(look.range_km - 397206.0) < 1000.0


def test_moon_illumination_full():
    t = _us(2024, 4, 23, 23, 49, 0)
    illum = sidereon.moon_illumination(LAT, LON, ALT_KM, t)
    assert abs(illum.illuminated_fraction - 0.998) < 0.02
    assert illum.phase_angle_deg < 11.0


def test_moon_elevation_deg_consistent_with_az_el():
    t = _us(2024, 4, 23, 23, 55, 59)
    el = sidereon.moon_elevation_deg(LAT, LON, ALT_KM, t)
    look = sidereon.moon_az_el(LAT, LON, ALT_KM, t)
    assert el == look.elevation_deg


def test_find_moon_elevation_crossings():
    start = _us(2024, 4, 23, 0, 0, 0)
    end = _us(2024, 4, 25, 0, 0, 0)
    crossings = sidereon.find_moon_elevation_crossings(LAT, LON, ALT_KM, start, end)
    assert crossings  # at least one moonrise/moonset in two days
    kinds = {c.kind for c in crossings}
    assert kinds <= {"rising", "setting"}
    for c in crossings:
        assert start <= c.time_unix_us <= end
        assert abs(c.elevation_deg - (-0.833)) < 0.5


def test_find_moon_transits():
    start = _us(2024, 4, 23, 0, 0, 0)
    end = _us(2024, 4, 25, 0, 0, 0)
    transits = sidereon.find_moon_transits(LAT, LON, ALT_KM, start, end)
    assert transits
    kinds = {t.kind for t in transits}
    assert kinds <= {"upper", "lower"}
    # An upper culmination should be the high point: positive elevation here.
    uppers = [t for t in transits if t.kind == "upper"]
    assert uppers
    assert max(t.elevation_deg for t in uppers) > 0.0


def test_az_el_and_illumination_reject_bad_station():
    """A finite-but-out-of-domain station latitude is a caller input error, so
    the look-angle and illumination helpers raise ``ValueError`` (a core
    invalid-input variant), not ``SolveError``."""
    t = _us(2024, 4, 23, 23, 55, 59)
    with pytest.raises(ValueError):
        sidereon.sun_az_el(BAD_LAT, LON, ALT_KM, t)
    with pytest.raises(ValueError):
        sidereon.moon_az_el(BAD_LAT, LON, ALT_KM, t)
    with pytest.raises(ValueError):
        sidereon.moon_illumination(BAD_LAT, LON, ALT_KM, t)


def test_moon_elevation_deg_bad_station_raises_not_panics():
    """An invalid-but-finite latitude used to panic inside the core
    ``moon_elevation_deg`` (`expect` on a rejected station); the binding now
    delegates to ``moon_az_el`` so it raises ``ValueError`` instead."""
    t = _us(2024, 4, 23, 23, 55, 59)
    with pytest.raises(ValueError):
        sidereon.moon_elevation_deg(BAD_LAT, LON, ALT_KM, t)


def test_finders_reject_nonpositive_step():
    """The finders take a caller-controlled scan step, so a non-positive step is
    a core invalid-input -> ``ValueError``, not a solver failure."""
    good_start = _us(2024, 4, 23, 0, 0, 0)
    good_end = _us(2024, 4, 25, 0, 0, 0)
    with pytest.raises(ValueError):
        sidereon.find_moon_elevation_crossings(
            LAT, LON, ALT_KM, good_start, good_end, step_seconds=0.0
        )
    with pytest.raises(ValueError):
        sidereon.find_moon_transits(
            LAT, LON, ALT_KM, good_start, good_end, step_seconds=-1.0
        )


def test_finders_reversed_window_returns_empty():
    """An end-before-start window is the documented empty-result contract, not an
    error: the finder returns no events."""
    good_start = _us(2024, 4, 23, 0, 0, 0)
    good_end = _us(2024, 4, 25, 0, 0, 0)
    assert (
        sidereon.find_moon_elevation_crossings(LAT, LON, ALT_KM, good_end, good_start)
        == []
    )
    assert sidereon.find_moon_transits(LAT, LON, ALT_KM, good_end, good_start) == []
