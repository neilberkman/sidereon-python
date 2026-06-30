"""Multi-record TLE file parsing (CelesTrak / Space-Track style).

`sidereon.parse_tle_file` wraps the core `parse_tle_file`: it returns a `TleFile`
whose `.satellites` are `NamedTle` (name + native `Tle`) and whose `.skipped`
counts complete records that failed SGP4 initialization. The returned `Tle`s are
the binding's native type, so they propagate / look-angle directly.
"""

import numpy as np
import sidereon

# Real ISS (ZARYA) TLE.
ISS_L1 = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993"
ISS_L2 = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106"

# A complete (line 1, line 2) record whose line 2 carries a non-numeric
# eccentricity field: it is found as a record but fails SGP4 init, so it is
# skipped and counted, not raised.
BAD_L1 = ISS_L1.replace("25544", "99999")
BAD_L2 = ISS_L2.replace("25544", "99999").replace("0003435", "XXXXXXX")


def _sample_file():
    # (a) a 3-line named record, (b) a malformed record, (c) a bare 2-line record.
    return "\n".join(
        [
            "ISS (ZARYA)",
            ISS_L1,
            ISS_L2,
            BAD_L1,
            BAD_L2,
            ISS_L1,
            ISS_L2,
        ]
    )


def test_parse_tle_file_shape_names_and_skipped():
    result = sidereon.parse_tle_file(_sample_file())

    assert isinstance(result, sidereon.TleFile)
    assert result.skipped == 1
    assert len(result) == 2
    assert len(result.satellites) == 2

    first, second = result.satellites
    assert isinstance(first, sidereon.NamedTle)
    # (a) the 3-line record carries its name.
    assert first.name == "ISS (ZARYA)"
    # (c) the bare 2-line record has an empty name.
    assert second.name == ""

    assert isinstance(first.tle, sidereon.Tle)
    assert first.tle.catalog_number == "25544"
    assert second.tle.catalog_number == "25544"


def test_parsed_tle_propagates_and_look_angles():
    result = sidereon.parse_tle_file(_sample_file())
    tle = result.satellites[0].tle

    # Epoch near the ISS TLE epoch (2018 day-of-year 184 ~ 2018-07-03).
    epochs = np.asarray(
        [np.datetime64("2018-07-03T19:25:00", "us")], dtype="datetime64[us]"
    ).astype("int64")

    prop = tle.propagate(epochs)
    assert prop.position_km.shape == (1, 3)
    assert prop.velocity_km_s.shape == (1, 3)
    assert np.all(np.isfinite(prop.position_km))
    # ISS is in LEO: radius ~6.7e3 km.
    radius = float(np.linalg.norm(prop.position_km[0]))
    assert 6.6e3 < radius < 7.0e3

    station = sidereon.GroundStation(latitude_deg=51.5, longitude_deg=-0.1)
    look = tle.look_angles(station, epochs)
    assert look.azimuth_deg.shape == (1,)
    assert np.all(np.isfinite(look.azimuth_deg))
    assert np.all(np.isfinite(look.elevation_deg))
    assert np.all(look.range_km > 0.0)


def test_skipped_distinguishes_empty_from_corrupt():
    empty = sidereon.parse_tle_file("\n\n   \n")
    assert len(empty) == 0
    assert empty.skipped == 0

    corrupt = sidereon.parse_tle_file("\n".join([BAD_L1, BAD_L2]))
    assert len(corrupt) == 0
    assert corrupt.skipped == 1
