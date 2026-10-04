"""Direct equality and inequality checks for remaining public value wrappers."""

from pathlib import Path

import pytest
import sidereon
from _helpers import CORE_FIXTURES
from test_rtk_unresolved_carriers import (
    GLONASS,
    _record,
)
from test_rtk_unresolved_carriers import (
    _obs as _rtk_obs,
)
from test_rtk_unresolved_carriers import (
    _sp3 as _rtk_sp3,
)
from test_sp3_write_and_records import (
    _G01,
    _G02_CLOCK_ONLY,
    _write_refusal,
)
from test_sp3_write_and_records import (
    _load as _load_sp3,
)
from test_tides import ZIM2_BLQ_BLOCK


def _assert_equal_and_different(left, equal, different):
    assert left == equal
    assert not (left != equal)
    assert left != different
    assert not (left == different)


def test_exact_epoch_instant_and_carrier_pair_equality():
    atto = sidereon.ExactEpoch.ATTOSECONDS_PER_SECOND
    epoch = sidereon.ExactEpoch.new(-1, atto - 1)
    later_epoch = sidereon.ExactEpoch.new(-1, atto - 2)
    _assert_equal_and_different(
        epoch, sidereon.ExactEpoch.new(-1, atto - 1), later_epoch
    )

    instant = sidereon.Instant.from_utc(2026, 10, 4, 12, 30, 1.25)
    same_instant = sidereon.Instant.from_unix_micros(instant.unix_micros)
    next_instant = sidereon.Instant.from_unix_micros(instant.unix_micros + 1)
    _assert_equal_and_different(instant, same_instant, next_instant)

    pair = sidereon.default_pair(sidereon.GnssSystem.GPS)
    assert pair is not None
    _assert_equal_and_different(
        pair,
        sidereon.CarrierPair(sidereon.CarrierBand.L1, sidereon.CarrierBand.L2),
        sidereon.CarrierPair(sidereon.CarrierBand.L1, sidereon.CarrierBand.L5),
    )


def _obs_at(second):
    header = [
        f"{'     3.05           OBSERVATION DATA    M (MIXED)':<60}"
        "RINEX VERSION / TYPE",
        f"{'G    1 C1C':<60}SYS / # / OBS TYPES",
        f"{'':<60}END OF HEADER",
    ]
    epoch = f"> 2020 01 01 00 00 {second:11.7f}  0  1"
    measurement = "G01" + f"{20_000_000.0:14.3f}" + "  "
    return sidereon.parse_rinex_obs("\n".join([*header, epoch, measurement, ""]))


def test_observation_epoch_value_equality_comes_from_public_parser():
    left = _obs_at(0.0).epochs[0].epoch
    same = _obs_at(0.0).epochs[0].epoch
    changed = _obs_at(1.0).epochs[0].epoch
    _assert_equal_and_different(left, same, changed)


def test_rtk_unresolved_carrier_equality_comes_from_independent_arc_builds():
    obs = _rtk_obs(
        "R    2 C1C L1C",
        _record("R01", 20_000_000.0, 100.0),
        _record("R28", 20_000_000.0, 100.0),
    )
    options = sidereon.RtkRinexArcOptions(
        signal_pairs=[sidereon.RtkRinexSignalPair(GLONASS, "C1C", "L1C")],
        min_common_satellites=1,
        include_prediction_time=False,
    )
    left = sidereon.build_rinex_rtk_arc(
        _rtk_sp3(), obs, obs, options
    ).unresolved_carriers[0]
    same = sidereon.build_rinex_rtk_arc(
        _rtk_sp3(), obs, obs, options
    ).unresolved_carriers[0]
    other_receiver = sidereon.build_rinex_rtk_arc(
        _rtk_sp3(), obs, obs, options
    ).unresolved_carriers[1]
    _assert_equal_and_different(left, same, other_receiver)


def test_sp3_clock_record_and_writer_error_detail_equality():
    flags = {"G02": " " * 14 + "E"}
    first = _load_sp3([_G01, _G02_CLOCK_ONLY], flags=flags)
    same = _load_sp3([_G01, _G02_CLOCK_ONLY], flags=flags)
    changed = _load_sp3([_G01, ("G02", (0.0, 0.0, 0.0), 251.0)], flags=flags)
    _assert_equal_and_different(
        first.clock_record("G02", 0),
        same.clock_record("G02", 0),
        changed.clock_record("G02", 0),
    )

    wide = _load_sp3([_G01], agency="TSTAB")
    same_wide = _load_sp3([_G01], agency="TSTAB")
    finer = _load_sp3(
        [_G01],
        pf_line="%f 1.25000001  1.025000000  0.00000000000  0.000000000000000",
    )
    _assert_equal_and_different(
        _write_refusal(wide),
        _write_refusal(same_wide),
        _write_refusal(finer),
    )


def test_dted_datum_and_tide_error_details_compare_public_values(tmp_path):
    tile_path = Path(CORE_FIXTURES) / "dted" / "tiles" / "n36_w107_1arc_v3.dt2"
    wgs84 = bytearray(tile_path.read_bytes())
    wgs72 = bytearray(wgs84)
    wgs72[224:229] = b"WGS72"
    path_a = tmp_path / "a.dt2"
    path_b = tmp_path / "b.dt2"
    path_a.write_bytes(wgs84)
    path_b.write_bytes(wgs72)
    datum = sidereon.DtedTile.from_path(path_a).horizontal_datum
    same_datum = sidereon.DtedTile.from_path(path_a).horizontal_datum
    changed_datum = sidereon.DtedTile.from_path(path_b).horizontal_datum
    _assert_equal_and_different(datum, same_datum, changed_datum)

    invalid_a = ZIM2_BLQ_BLOCK.replace("Ssa", "M4")
    invalid_b = ZIM2_BLQ_BLOCK.replace("Ssa", "M3")
    with pytest.raises(sidereon.BlqError) as error_a:
        sidereon.parse_ocean_loading_blq_block(invalid_a)
    with pytest.raises(sidereon.BlqError) as error_again:
        sidereon.parse_ocean_loading_blq_block(invalid_a)
    with pytest.raises(sidereon.BlqError) as error_b:
        sidereon.parse_ocean_loading_blq_block(invalid_b)
    _assert_equal_and_different(
        error_a.value.detail, error_again.value.detail, error_b.value.detail
    )


def test_ocean_loading_blq_comment_and_block_equality():
    _assert_equal_and_different(
        sidereon.OceanLoadingBlqComment("$$ Station note", "before_station"),
        sidereon.OceanLoadingBlqComment("$$ Station note", "before_station"),
        sidereon.OceanLoadingBlqComment("$$ Different note", "before_station"),
    )
    block = sidereon.parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK)
    same_block = sidereon.parse_ocean_loading_blq_block(ZIM2_BLQ_BLOCK)
    changed_block = sidereon.parse_ocean_loading_blq_block(
        ZIM2_BLQ_BLOCK.replace("\nZIM2\n", "\nABCD\n", 1)
    )
    _assert_equal_and_different(block, same_block, changed_block)
