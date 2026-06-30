"""The package imports and exposes its public surface."""

from importlib.metadata import version

import sidereon


def test_import_and_surface():
    # __version__ tracks the installed package metadata (single source of truth),
    # so this does not need editing on every release bump.
    assert sidereon.__version__ == version("sidereon")
    for name in (
        "load_sp3",
        "load_antex",
        "solve_spp",
        "Sp3",
        "Antex",
        "Antenna",
        "AntexDateTime",
        "AntennaKind",
        "SppSolution",
        "SidereonError",
        "Tle",
        "OpsMode",
        "ForceModel",
        "Integrator",
        "GroundStation",
        "TlePropagation",
        "LookAngles",
        "VisibilitySeries",
        "SppConfig",
        "IntegerStatus",
        "RtkStochasticModel",
        "RtkFloatConfig",
        "RtkFixedConfig",
        "PppFloatConfig",
        "PppFixedConfig",
        "PppFixedSolution",
        "solve_ppp_fixed",
        "load_rinex_nav",
        "parse_rinex_nav_records",
        "BroadcastEphemeris",
        "BroadcastEvaluation",
        "BroadcastRecord",
        "RinexNavParseError",
    ):
        assert hasattr(sidereon, name), name


def test_sidereon_error_is_exception():
    assert issubclass(sidereon.SidereonError, Exception)


def test_load_sp3_raises_on_garbage():
    import pytest

    with pytest.raises(sidereon.SidereonError):
        sidereon.load_sp3(b"not a valid sp3 file")
