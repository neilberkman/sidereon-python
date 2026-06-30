import os
import pathlib

import numpy as np
import pytest
import sidereon
from _helpers import FIXTURES

ANTEX_FIXTURES = os.path.join(FIXTURES, "antex")


def _antex_path(name):
    return os.path.join(ANTEX_FIXTURES, name)


def _mm(values):
    return np.asarray(values, dtype=np.float64) / 1000.0


def test_load_antex_from_path_and_lookup_satellite_pco_pcv():
    antex = sidereon.load_antex(pathlib.Path(_antex_path("igs20_wettzell_trim.atx")))
    epoch = sidereon.AntexDateTime(2020, 6, 25)

    g05 = antex.satellite_antenna("G05", epoch)

    assert antex.antenna_count == 10
    assert g05 is not None
    assert g05.kind == sidereon.AntennaKind.SATELLITE
    assert g05.kind.label == "satellite"
    assert g05.serial == "G05"
    assert g05.valid_at(epoch)
    assert g05.valid_from == sidereon.AntexDateTime(2009, 8, 17)
    assert g05.valid_until is None
    assert "G01" in g05.frequencies
    np.testing.assert_array_equal(g05.pco("G01"), _mm([-3.30, -0.30, 742.63]))
    assert g05.pcv("G01", 9.0) == -9.50 / 1000.0
    assert antex.satellite_antenna("G99", epoch) is None


def test_antex_receiver_lookup_from_bytes():
    with open(_antex_path("igs20_wettzell_trim.atx"), "rb") as fh:
        antex = sidereon.load_antex(fh.read())

    receiver = antex.antenna("LEIAR25.R3      LEIT")

    assert receiver is not None
    assert receiver.kind == sidereon.AntennaKind.RECEIVER
    assert receiver.antenna_type == "LEIAR25.R3      LEIT"
    assert receiver.serial == ""
    assert receiver.valid_from is None
    np.testing.assert_array_equal(receiver.pco("G01"), _mm([-0.05, 0.95, 160.96]))
    assert receiver.pcv("G01", 10.0) == 0.99 / 1000.0


def test_load_second_antex_fixture_and_receiver_pco():
    antex = sidereon.load_antex(_antex_path("igs20_pasa_scoa_gps.atx"))
    receiver_id = next(id for id in antex.antenna_ids if id.startswith("LEIAR20"))
    receiver = antex.antenna(receiver_id)

    assert receiver is not None
    assert receiver.kind == sidereon.AntennaKind.RECEIVER
    assert receiver.antenna_type == "LEIAR20         LEIM"
    assert receiver.serial == ""
    np.testing.assert_array_equal(receiver.pco("G01"), _mm([0.50, 0.13, 124.88]))
    assert receiver.pcv("G01", 20.0) == -0.99 / 1000.0


def test_antex_validation_and_lookup_errors():
    antex = sidereon.load_antex(_antex_path("igs20_wettzell_trim.atx"))
    receiver = antex.antenna("LEIAR25.R3      LEIT")

    assert receiver is not None
    with pytest.raises(ValueError, match="invalid ANTEX datetime"):
        sidereon.AntexDateTime(2020, 2, 30)
    with pytest.raises(ValueError, match="unknown frequency"):
        receiver.pco("UNKNOWN")
    with pytest.raises(ValueError, match="zenith_deg must be finite"):
        receiver.pcv("G01", float("nan"))


def test_to_antex_string_round_trips_through_load():
    antex = sidereon.load_antex(_antex_path("igs20_wettzell_trim.atx"))
    text = antex.to_antex_string()
    assert isinstance(text, str)
    assert "ANTEX VERSION" in text

    reparsed = sidereon.load_antex(text.encode("ascii"))
    assert reparsed.antenna_count == antex.antenna_count
    assert reparsed.antenna_ids == antex.antenna_ids

    epoch = sidereon.AntexDateTime(2020, 6, 25)
    g05 = antex.satellite_antenna("G05", epoch)
    g05_again = reparsed.satellite_antenna("G05", epoch)
    assert g05 is not None and g05_again is not None
    np.testing.assert_array_equal(g05_again.pco("G01"), g05.pco("G01"))
