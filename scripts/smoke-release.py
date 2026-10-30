#!/usr/bin/env python3
"""Smoke-test public APIs from the installed release artifact."""

from __future__ import annotations

import gzip
import math
import os
import struct
import tempfile
from pathlib import Path

import sidereon
import sidereon.data as sidereon_data


def _sp3(text: str):
    return sidereon.load_sp3(text.encode("ascii"))


def _check_sp3_grid_merge() -> None:
    first_text = """\
#cP2020  6 25  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59025 0.0000000000000
+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* RELEASE CONSUMER SP3-C FIXTURE
*  2020  6 25  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    100.000000
EOF
"""
    second_text = first_text.replace("15000.000000", "15000.000001")
    merged, _report = sidereon.merge_sp3(
        [_sp3(first_text), _sp3(second_text)],
        sidereon.Sp3MergeOptions(min_agree=1, clock_min_common=1),
    )

    strict_text = merged.to_sp3_string()
    reread = sidereon.load_sp3(strict_text.encode("ascii"))
    first_x = _sp3(first_text).state("G01", 0).position_m[0]
    second_x = _sp3(second_text).state("G01", 0).position_m[0]
    x_m = reread.state("G01", 0).position_m[0]
    if x_m != merged.state("G01", 0).position_m[0] or x_m not in (first_x, second_x):
        raise SystemExit(
            "strict SP3 merge round-trip did not preserve a parser-scaled grid value"
        )

    rounded_text, rounding = merged.to_sp3_string_with_rounding()
    if rounded_text != strict_text or rounding.position_x_m != 0.0:
        raise SystemExit(
            "SP3 rounding report disagrees with the writable merged product"
        )


def _clock_sp3(clocks_us: list[float]):
    satellites = ["G01", "G02", "G03"]
    satellite_header = "".join(satellites) + "  0" * 14
    records = "\n".join(
        f"P{sat}{15000.0:14.6f}{-20000.0:14.6f}{5000.0:14.6f}{clock:14.6f}"
        for sat, clock in zip(satellites, clocks_us)
    )
    text = f"""\
#cP2020  6 25  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59025 0.0000000000000
+    3   {satellite_header}
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* RELEASE CONSUMER SP3 CLOCK FIXTURE
*  2020  6 25  0  0  0.00000000
{records}
EOF
"""
    return _sp3(text)


def _check_sp3_half_grid_clock_merge() -> None:
    first = _clock_sp3([-598.410235, 138.983111, 0.0])
    second = _clock_sp3([-598.410234, 138.983110, 0.0])
    first.to_sp3_string()
    second.to_sp3_string()
    merged, _report = sidereon.merge_sp3(
        [first, second], sidereon.Sp3MergeOptions(min_agree=1, clock_min_common=3)
    )

    strict_text = merged.to_sp3_string()
    reread = sidereon.load_sp3(strict_text.encode("ascii"))
    for satellite in ("G01", "G02", "G03"):
        left = first.state(satellite, 0).clock_s
        right = second.state(satellite, 0).clock_s
        clock = reread.state(satellite, 0).clock_s
        if clock != merged.state(satellite, 0).clock_s or clock not in (left, right):
            raise SystemExit(f"strict SP3 merge round-trip changed {satellite} clock")

    rounded_text, report = merged.to_sp3_string_with_rounding()
    if (
        rounded_text != strict_text
        or report.clock_s != 0.0
        or report.clock_rate_s_s != 0.0
    ):
        raise SystemExit(
            "SP3 half-grid clock merge did not return a zero rounding report"
        )


def _accuracy_product(satellite: str, base: str, x_exponent: int | None) -> str:
    code = "" if x_exponent is None else f" {x_exponent:2d} {'':2} {'':2} {'':3} "
    records = "\n".join(
        f"P{satellite}{x:14.6f}{y:14.6f}{z:14.6f}{-50.0 + index:14.6f}{code}"
        for index, (x, y, z) in enumerate(
            ((15000.0, -20000.0, 5000.0), (15001.0, -20001.0, 5001.0))
        )
    )
    return f"""\
#cP2020  6 25  0  0  0.00000000       2 ORBIT IGS14 FIT  TST
## 2111 345600.00000000   300.00000000 59025 0.0000000000000
+    1   {satellite}{"  0" * 16}
+         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  {base}  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* RELEASE CONSUMER ACCURACY ROUNDING FIXTURE
*  2020  6 25  0  0  0.00000000
{records.splitlines()[0]}
*  2020  6 25  0  5  0.00000000
{records.splitlines()[1]}
EOF
"""


def _accuracy_group_product(
    satellites: list[str], position_base: float, clock_base: float, exponent: int
):
    satellite_header = "".join(satellites) + "  0" * (17 - len(satellites))
    code = f" {exponent:2d} {exponent:2d} {exponent:2d} {exponent:3d}  "
    lines = [
        "#cV2020  6 25  0  0  0.00000000       2 ORBIT IGS14 FIT  TST",
        "## 2111 345600.00000000   300.00000000 59025 0.0000000000000",
        f"+   {len(satellites):2}   {satellite_header}",
        f"++       {'  0' * 17}",
        "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        (
            f"%f {position_base:10.7f} {clock_base:12.9f} "
            "0.00000000000  0.000000000000000"
        ),
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
        "/* RELEASE CONSUMER SP3 ACCURACY FIXTURE",
    ]
    positions = {
        "G01": (15000.0, -20000.0, 5000.0),
        "G02": (13000.0, -18000.0, 7000.0),
        "G12": (16000.0, -19000.0, 4000.0),
    }
    for epoch in range(2):
        lines.append(f"*  2020  6 25  0 {epoch * 5:2d}  0.00000000")
        for satellite in satellites:
            x, y, z = positions[satellite]
            lines.append(
                f"P{satellite}{x + epoch:14.6f}{y - epoch:14.6f}{z + epoch:14.6f}"
                f"{-50.0 + epoch:14.6f}{code}"
            )
            lines.append(
                f"V{satellite}{10000.0:14.6f}{-20000.0:14.6f}{30000.0:14.6f}"
                f"{1.0:14.6f}{code}"
            )
    lines.append("EOF")
    return "\n".join(lines) + "\n"


def _check_sp3_accuracy_rounding() -> None:
    a = _sp3(_accuracy_product("G01", "0.0000000", None))
    b = _sp3(_accuracy_product("G12", "1.2500000", 13))
    a.to_sp3_string()
    b.to_sp3_string()
    options = sidereon.Sp3MergeOptions(
        combine="precedence",
        precedence_scope="cell",
        min_agree=1,
        clock_min_common=1,
        position_tolerance_m=5.0,
        clock_tolerance_s=50e-9,
        target_epoch_interval_s=300.0,
        systems=["G"],
    )
    merged, _ = sidereon.merge_sp3([a, b], options)
    text, report = merged.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    if not isinstance(report.record_values, sidereon.Sp3RoundingReport):
        raise SystemExit("SP3 numeric rounding report is not publicly exported")
    reread = _sp3(text)
    source_sigma = merged.record_accuracy("G12", 0).p.position_sigma_m[0].value
    result_sigma = reread.record_accuracy("G12", 0).p.position_sigma_m[0].value
    if source_sigma != 0.018189894035458565 or result_sigma != source_sigma:
        raise SystemExit("accuracy-aware SP3 writer changed a grid-representable sigma")
    if reread.record_accuracy("G01", 0).p.position_sigma_m[0].kind != "unknown":
        raise SystemExit("accuracy-aware SP3 writer changed an unknown sigma")
    if any(
        getattr(report.accuracy_sigmas, name) != 0.0
        for name in (
            "position_sigma_x_m",
            "position_sigma_y_m",
            "position_sigma_z_m",
            "velocity_sigma_x_m_s",
            "velocity_sigma_y_m_s",
            "velocity_sigma_z_m_s",
            "clock_sigma_m",
            "clock_rate_sigma_m_s",
        )
    ):
        raise SystemExit("accuracy-aware SP3 writer returned a nonzero grid delta")
    try:
        merged.to_sp3_string_with_accuracy_rounding("unknown")
    except ValueError:
        pass
    else:
        raise SystemExit("unknown accuracy rounding policy did not raise")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "rounded.sp3.gz"
        written, data_report = sidereon_data.write_sp3_with_accuracy_rounding(
            merged,
            str(path),
            policy=sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE,
            gzip=True,
        )
        if written != str(path) or not path.is_file():
            raise SystemExit("accuracy-aware SP3 data writer did not write its path")
        if gzip.decompress(path.read_bytes()).decode("ascii") != text:
            raise SystemExit("accuracy-aware SP3 data writer changed serialized text")
        if data_report.accuracy_sigmas.position_sigma_x_m != 0.0:
            raise SystemExit("accuracy-aware SP3 data writer report differs")

    all_sigma_source = _sp3(_accuracy_group_product(["G12"], 2.0, 2.0, 1))
    all_sigma_text, all_sigma_report = (
        all_sigma_source.to_sp3_string_with_accuracy_rounding(
            sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
        )
    )
    all_sigma_roundtrip = _sp3(all_sigma_text)
    all_sigma_source_accuracy = all_sigma_source.record_accuracy("G12", 0)
    all_sigma_output_accuracy = all_sigma_roundtrip.record_accuracy("G12", 0)
    if any(
        getattr(all_sigma_report.accuracy_sigmas, name) != 0.0
        for name in (
            "position_sigma_x_m",
            "position_sigma_y_m",
            "position_sigma_z_m",
            "velocity_sigma_x_m_s",
            "velocity_sigma_y_m_s",
            "velocity_sigma_z_m_s",
            "clock_sigma_m",
            "clock_rate_sigma_m_s",
        )
    ):
        raise SystemExit("direct SP3 writer changed a grid-representable sigma")
    if all_sigma_source_accuracy.v is None or all_sigma_output_accuracy.v is None:
        raise SystemExit("direct SP3 writer lost velocity accuracy records")
    if tuple(
        value.value for value in all_sigma_source_accuracy.v.velocity_sigma_m_s
    ) != tuple(value.value for value in all_sigma_output_accuracy.v.velocity_sigma_m_s):
        raise SystemExit("direct SP3 writer changed velocity sigmas")

    first = _sp3(_accuracy_group_product(["G01", "G02"], 2.0, 2.0, 1))
    second = _sp3(_accuracy_group_product(["G02", "G12"], 1.25, 1.025, 13))
    changed, _merge_report = sidereon.merge_sp3(
        [first, second],
        sidereon.Sp3MergeOptions(
            combine="precedence",
            precedence_scope="cell",
            min_agree=1,
            clock_min_common=1,
            position_tolerance_m=5.0,
            clock_tolerance_s=50e-9,
            target_epoch_interval_s=300.0,
            systems=["G"],
        ),
    )
    changed_text, detailed = changed.to_sp3_string_with_accuracy_rounding(
        sidereon.Sp3AccuracyRoundingPolicy.CONSERVATIVE
    )
    changed_roundtrip = _sp3(changed_text)
    source = changed.record_accuracy("G12", 0)
    output = changed_roundtrip.record_accuracy("G12", 0)
    expected_changes = (
        *(
            value.value - source.p.position_sigma_m[index].value
            for index, value in enumerate(output.p.position_sigma_m)
        ),
        output.p.clock_sigma_m.value - source.p.clock_sigma_m.value,
    )
    reported_changes = tuple(
        getattr(detailed.accuracy_sigmas, name)
        for name in (
            "position_sigma_x_m",
            "position_sigma_y_m",
            "position_sigma_z_m",
            "clock_sigma_m",
        )
    )
    if reported_changes != expected_changes or not all(
        value > 0.0 for value in reported_changes
    ):
        raise SystemExit(
            "SP3 accuracy report does not map merged position and clock changes"
        )
    if (
        any(
            getattr(detailed.accuracy_sigmas, name) != 0.0
            for name in (
                "velocity_sigma_x_m_s",
                "velocity_sigma_y_m_s",
                "velocity_sigma_z_m_s",
                "clock_rate_sigma_m_s",
            )
        )
        or output.v is not None
    ):
        raise SystemExit("merged SP3 unexpectedly retained velocity accuracy")
    if tuple(value.value for value in output.p.position_sigma_m) != (0.032,) * 3:
        raise SystemExit(
            "SP3 position sigmas were not conservatively raised to 0.032 m"
        )
    if output.p.clock_sigma_m.value != 2.0 * (299_792_458.0 * 1e-12):
        raise SystemExit("SP3 clock sigma was not conservatively raised")


def main() -> None:
    expected_version = os.environ.get("SIDEREON_EXPECTED_VERSION")
    if expected_version is not None and sidereon.__version__ != expected_version:
        raise SystemExit(
            f"installed sidereon {sidereon.__version__}, expected {expected_version}"
        )

    header = struct.pack(">ddddii", -90.0, -180.0, 0.25, 0.25, 721, 1440)
    data = header + struct.pack(">f", 2.5) * (721 * 1440)
    grid = sidereon.GeoidGrid.from_proj_egm96_gtx(data)

    for arithmetic in (
        sidereon.ProjVgridshiftArithmetic.SEPARATE_MULTIPLY_ADD,
        sidereon.ProjVgridshiftArithmetic.FUSED_MULTIPLY_ADD,
    ):
        got = grid.undulation_proj_rad(
            math.radians(40.0), math.radians(-105.0), arithmetic
        )
        if got != 2.5:
            raise SystemExit(f"PROJ EGM96 GTX smoke returned {got}, expected 2.5")

    try:
        grid.undulation_proj_rad(
            math.nan,
            0.0,
            sidereon.ProjVgridshiftArithmetic.FUSED_MULTIPLY_ADD,
        )
    except sidereon.ProjVgridshiftNonFiniteCoordinateError as error:
        if not isinstance(error, (sidereon.ProjVgridshiftError, ValueError)):
            raise SystemExit(
                "typed PROJ coordinate exception hierarchy is broken"
            ) from None
    else:
        raise SystemExit("non-finite PROJ coordinate did not raise")

    _check_sp3_grid_merge()
    _check_sp3_half_grid_clock_merge()
    _check_sp3_accuracy_rounding()

    print(f"sidereon {sidereon.__version__} release smoke passed")


if __name__ == "__main__":
    main()
