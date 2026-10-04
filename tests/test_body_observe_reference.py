"""Public body-observation checks against independent ephemeris references.

JPL values are reproducible with GET https://ssd.jpl.nasa.gov/api/horizons.api
and format=text, OBJ_DATA=NO, MAKE_EPHEM=YES, EPHEM_TYPE=OBSERVER,
CENTER=coord@399, COORD_TYPE=GEODETIC, SITE_COORD=0,51.4769,0.046,
STEP_SIZE=1 m, TIME_TYPE=UT, QUANTITIES=2,4,20, ANG_FORMAT=DEG,
APPARENT=AIRLESS, RANGE_UNITS=KM. For the Sun use COMMAND=10,
START_TIME=2024-06-20 12:01:42, STOP_TIME=2024-06-20 12:02:42; for the
Moon use COMMAND=301, START_TIME=2024-04-23 23:55:59,
STOP_TIME=2024-04-24 00:00:59. Both responses identify DE441. First data
rows (apparent RA/Dec, az/el, range km) are Sun 89.61786, 23.43665,
179.997296, 61.959749, 1.5201128641E+08; Moon 211.50898, -15.40328,
179.999082, 23.119822, 3.9720636319E+05. The JPL manual documents q2 apparent
coordinates as true-equator/equinox-of-date for Earth sites and q4 azimuth as
clockwise from north. Sidereon's analytic Sun/Moon series are intended for
planning and visualization, not almanac precision, so the angular tolerances
are deliberately sub-degree.
"""

from datetime import datetime, timezone

import pytest
import sidereon


@pytest.mark.parametrize(
    (
        "target",
        "epoch",
        "apparent_ra_deg",
        "apparent_dec_deg",
        "azimuth_deg",
        "elevation_deg",
        "range_km",
        "angle_tolerance_deg",
        "range_tolerance_km",
    ),
    [
        (
            sidereon.Target.sun(),
            datetime(2024, 6, 20, 12, 1, 42, tzinfo=timezone.utc),
            89.61786,
            23.43665,
            179.997296,
            61.959749,
            152_011_286.41,
            0.01,
            100_000.0,
        ),
        (
            sidereon.Target.moon(),
            datetime(2024, 4, 23, 23, 55, 59, tzinfo=timezone.utc),
            211.50898,
            -15.40328,
            179.999082,
            23.119822,
            397_206.36319,
            0.95,
            1_000.0,
        ),
    ],
    ids=("sun", "moon"),
)
def test_ground_observation_tracks_independent_horizons_reference(
    target,
    epoch,
    apparent_ra_deg,
    apparent_dec_deg,
    azimuth_deg,
    elevation_deg,
    range_km,
    angle_tolerance_deg,
    range_tolerance_km,
):
    epoch_unix_us = int(epoch.timestamp()) * 1_000_000

    observation = sidereon.observe_body(
        51.4769,
        0.0,
        0.046,
        epoch_unix_us,
        target,
    )

    assert observation.reduced is True
    assert 0.0 <= observation.apparent.right_ascension_deg < 360.0
    assert 0.0 <= observation.apparent.right_ascension_hours < 24.0
    assert -180.0 < observation.hour_angle_deg <= 180.0
    assert -12.0 < observation.hour_angle_hours <= 12.0
    assert 0.0 <= observation.horizontal.azimuth_deg < 360.0
    assert observation.apparent.right_ascension_deg == pytest.approx(
        apparent_ra_deg, abs=angle_tolerance_deg, rel=0.0
    )
    assert observation.apparent.declination_deg == pytest.approx(
        apparent_dec_deg, abs=angle_tolerance_deg, rel=0.0
    )
    assert observation.horizontal.azimuth_deg == pytest.approx(
        azimuth_deg, abs=angle_tolerance_deg, rel=0.0
    )
    assert observation.horizontal.elevation_deg == pytest.approx(
        elevation_deg, abs=angle_tolerance_deg, rel=0.0
    )
    assert observation.horizontal.range_km == pytest.approx(
        range_km, abs=range_tolerance_km, rel=0.0
    )

    # The API exposes both degree and hour forms of RA/hour angle.
    assert observation.apparent.right_ascension_hours * 15.0 == pytest.approx(
        observation.apparent.right_ascension_deg, abs=1e-12, rel=0.0
    )
    assert observation.hour_angle_hours * 15.0 == pytest.approx(
        observation.hour_angle_deg, abs=1e-12, rel=0.0
    )


# Full-target values are generated independently by the pinned Skyfield
# fixture generator also used by the core integration test. Its source is
# tests/fixtures/bodies/gen_observe_golden.py in sidereon-core; that generator
# uses observe_de.bsp and applies Sun-only gravitational deflection.
@pytest.mark.parametrize(
    (
        "naif_id",
        "latitude_deg",
        "longitude_deg",
        "altitude_km",
        "apparent_ra_deg",
        "apparent_dec_deg",
        "apparent_distance_km",
        "apparent_icrs_ra_deg",
        "apparent_icrs_dec_deg",
        "apparent_icrs_distance_km",
        "astrometric_ra_deg",
        "astrometric_dec_deg",
        "astrometric_distance_km",
        "azimuth_deg",
        "elevation_deg",
        "range_km",
        "ecliptic_longitude_deg",
        "ecliptic_latitude_deg",
        "ecliptic_distance_km",
    ),
    [
        (
            4,
            51.4779,
            -0.0015,
            0.046,
            267.0545293733468,
            -23.96185179890772,
            362_601_889.49888045,
            266.689454311102,
            -23.95235998065674,
            362_601_889.4988812,
            266.69569176937006,
            -23.95250704362053,
            362_601_889.498879,
            25.029803759785487,
            -60.701353351529,
            362_601_889.4988805,
            267.3084528018354,
            -0.550968260129117,
            362_601_889.49888045,
        ),
        (
            5,
            -33.8568,
            151.2153,
            0.040,
            33.685726736672464,
            12.26418101472206,
            670_428_149.8823149,
            33.36411888660434,
            12.152064228124974,
            670_428_149.8823142,
            33.36174062892731,
            12.151348694023607,
            670_428_149.8823142,
            113.07414946418533,
            -49.51460784852244,
            670_428_149.8823149,
            35.582668467646254,
            -1.1853565914672937,
            670_428_149.8823149,
        ),
    ],
    ids=("mars-greenwich", "jupiter-sydney"),
)
def test_spk_target_projects_complete_skyfield_golden(
    naif_id,
    latitude_deg,
    longitude_deg,
    altitude_km,
    apparent_ra_deg,
    apparent_dec_deg,
    apparent_distance_km,
    apparent_icrs_ra_deg,
    apparent_icrs_dec_deg,
    apparent_icrs_distance_km,
    astrometric_ra_deg,
    astrometric_dec_deg,
    astrometric_distance_km,
    azimuth_deg,
    elevation_deg,
    range_km,
    ecliptic_longitude_deg,
    ecliptic_latitude_deg,
    ecliptic_distance_km,
):
    import math
    from pathlib import Path

    kernel = sidereon.load_spk(
        Path(__file__).parent / "fixtures" / "bodies" / "observe_de.bsp"
    )
    observation = sidereon.observe_body(
        latitude_deg,
        longitude_deg,
        altitude_km,
        int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000,
        sidereon.Target.spk(kernel, naif_id),
    )

    def equatorial_separation_arcsec(coords, expected_ra_deg, expected_dec_deg):
        def vector(ra_deg, dec_deg):
            ra = math.radians(ra_deg)
            dec = math.radians(dec_deg)
            return (
                math.cos(dec) * math.cos(ra),
                math.cos(dec) * math.sin(ra),
                math.sin(dec),
            )

        actual = vector(coords.right_ascension_deg, coords.declination_deg)
        expected = vector(expected_ra_deg, expected_dec_deg)
        cross = (
            actual[1] * expected[2] - actual[2] * expected[1],
            actual[2] * expected[0] - actual[0] * expected[2],
            actual[0] * expected[1] - actual[1] * expected[0],
        )
        cross_norm = math.sqrt(sum(component * component for component in cross))
        dot = sum(a * b for a, b in zip(actual, expected))
        return math.degrees(math.atan2(cross_norm, dot)) * 3600.0

    def relative_distance(actual, expected):
        return abs(actual - expected) / max(abs(expected), 1.0)

    assert observation.reduced is False
    for coords, ra, dec, distance in (
        (
            observation.astrometric,
            astrometric_ra_deg,
            astrometric_dec_deg,
            astrometric_distance_km,
        ),
        (
            observation.apparent_icrs,
            apparent_icrs_ra_deg,
            apparent_icrs_dec_deg,
            apparent_icrs_distance_km,
        ),
        (
            observation.apparent,
            apparent_ra_deg,
            apparent_dec_deg,
            apparent_distance_km,
        ),
    ):
        assert 0.0 <= coords.right_ascension_deg < 360.0
        assert 0.0 <= coords.right_ascension_hours < 24.0
        assert equatorial_separation_arcsec(coords, ra, dec) < 1.0
        assert relative_distance(coords.distance_km, distance) < 1e-8
        assert coords.right_ascension_hours * 15.0 == pytest.approx(
            coords.right_ascension_deg, abs=1e-12, rel=0.0
        )

    assert -180.0 < observation.hour_angle_deg <= 180.0
    assert -12.0 < observation.hour_angle_hours <= 12.0
    assert 0.0 <= observation.horizontal.azimuth_deg < 360.0
    assert observation.horizontal.azimuth_deg == pytest.approx(
        azimuth_deg, abs=2.0 / 3600.0, rel=0.0
    )
    assert observation.horizontal.elevation_deg == pytest.approx(
        elevation_deg, abs=2.0 / 3600.0, rel=0.0
    )
    assert relative_distance(observation.horizontal.range_km, range_km) < 1e-8
    assert 0.0 <= observation.ecliptic.longitude_deg < 360.0
    assert observation.ecliptic.longitude_deg == pytest.approx(
        ecliptic_longitude_deg, abs=1.0 / 3600.0, rel=0.0
    )
    assert observation.ecliptic.latitude_deg == pytest.approx(
        ecliptic_latitude_deg, abs=1.0 / 3600.0, rel=0.0
    )
    assert (
        relative_distance(observation.ecliptic.distance_km, ecliptic_distance_km) < 1e-8
    )

    # Recover local hour angle from the independent Skyfield alt/az values.
    lat = math.radians(latitude_deg)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    hour_angle = math.degrees(
        math.atan2(
            -math.sin(az) * math.cos(el),
            math.sin(el) * math.cos(lat) - math.cos(el) * math.sin(lat) * math.cos(az),
        )
    )
    delta_hour_angle = (observation.hour_angle_deg - hour_angle + 180.0) % 360.0 - 180.0
    assert abs(delta_hour_angle) < 5.0 / 3600.0
    assert observation.hour_angle_hours * 15.0 == pytest.approx(
        observation.hour_angle_deg, abs=1e-12, rel=0.0
    )
