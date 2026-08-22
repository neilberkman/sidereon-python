from pathlib import Path

import pytest
import sidereon
from _helpers import CORE_FIXTURES

_FIXTURE = "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"


def _daily_product():
    return sidereon.load_sp3(Path(CORE_FIXTURES) / "sp3" / _FIXTURE)


def _product_with_seam_jump(first_shifted_epoch: int = 145):
    """Translate G01 from one node onward, leaving one speed-bound defect."""
    lines = _daily_product().to_sp3_string().splitlines()
    epoch_index = -1
    shifted = 0
    for index, line in enumerate(lines):
        if line.startswith("* "):
            epoch_index += 1
        elif epoch_index >= first_shifted_epoch and line.startswith("PG01"):
            x_km = float(line[4:18]) + 3_000.0
            lines[index] = f"{line[:4]}{x_km:14.6f}{line[18:]}"
            shifted += 1
    assert shifted > 0
    product = sidereon.load_sp3(("\n".join(lines) + "\n").encode("ascii"))
    seam = float(product.epochs_j2000_seconds[first_shifted_epoch - 1])
    return product, seam


def _speed_only_verdict(product, start: float, stop: float) -> dict:
    return product.continuity_verdict(
        start,
        stop,
        orbit_class="meo_gnss",
        residual_tolerance_m=None,
    )


def test_stencil_extent_and_window_verdict_map_the_three_core_cases():
    product, seam = _product_with_seam_jump()
    axis = product.epochs_j2000_seconds

    before_s, after_s = product.stencil_extent()
    assert (before_s, after_s) == (1_500.0, 1_500.0)

    inside_one_day = _speed_only_verdict(product, float(axis[24]), float(axis[72]))
    assert inside_one_day == {
        "decision": "accept",
        "accepted": True,
        "influencing_defects": [],
        "influencing_splices": [],
        "all_defects": inside_one_day["all_defects"],
        "all_splices": [],
    }
    assert len(inside_one_day["all_defects"]) == 1
    defect = inside_one_day["all_defects"][0]
    assert defect["kind"] == "speed_bound"
    assert defect["satellite"] == "G01"
    assert defect["from_j2000_s"] == seam
    assert defect["to_j2000_s"] == seam + 300.0

    straddling = _speed_only_verdict(product, seam - 600.0, seam + 600.0)
    assert straddling["decision"] == "refuse"
    assert straddling["accepted"] is False
    assert straddling["influencing_defects"] == straddling["all_defects"]
    assert straddling["influencing_splices"] == []
    assert straddling["all_splices"] == []

    reaches_seam = _speed_only_verdict(product, seam - 7_200.0, seam - after_s)
    assert reaches_seam["decision"] == "refuse"
    assert len(reaches_seam["influencing_defects"]) == 1

    misses_seam = _speed_only_verdict(product, seam - 7_200.0, seam - after_s - 0.001)
    assert misses_seam["decision"] == "accept"
    assert misses_seam["influencing_defects"] == []
    assert len(misses_seam["all_defects"]) == 1


def test_window_validation_and_unrequested_merge_continuity_are_preserved():
    product = _daily_product()
    start = float(product.epochs_j2000_seconds[0])

    with pytest.raises(ValueError, match="endpoints must be finite"):
        product.continuity_verdict(float("nan"), start)
    with pytest.raises(ValueError, match="start must not follow its end"):
        product.continuity_verdict(start + 1.0, start)

    merged, report = sidereon.merge_sp3(
        [product], sidereon.Sp3MergeOptions(min_agree=1, clock_min_common=1)
    )
    assert report.continuity_verdict(merged, start, start + 300.0) is None
