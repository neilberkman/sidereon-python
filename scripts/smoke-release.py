#!/usr/bin/env python3
"""Smoke-test public APIs from the installed release artifact."""

from __future__ import annotations

import math
import os
import struct

import sidereon


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


def _check_nonzero_rounding_report() -> None:
    lines = _clock_sp3([-598.410235, 138.983111, 0.0]).to_sp3_string().splitlines()
    record_index = next(i for i, line in enumerate(lines) if line.startswith("PG01"))
    record = lines[record_index]
    lines[record_index] = record[:46] + f"{-598.4102354:14.7f}"
    source = _sp3("\n".join(lines) + "\n")
    original_clock_s = source.state("G01", 0).clock_s

    try:
        source.to_sp3_string()
    except sidereon.Sp3WriteError as error:
        if error.detail.kind != "RecordValueNotRepresentable":
            raise SystemExit(
                f"strict SP3 writer refused for {error.detail.kind!r}"
            ) from None
    else:
        raise SystemExit("strict SP3 writer accepted a clock finer than its SP3 column")

    text, report = source.to_sp3_string_with_rounding()
    reread = sidereon.load_sp3(text.encode("ascii"))
    delta = abs(reread.state("G01", 0).clock_s - original_clock_s)
    if delta <= 0.0 or report.clock_s != delta:
        raise SystemExit("SP3 clock rounding report did not expose the nonzero change")
    if source.state("G01", 0).clock_s != original_clock_s:
        raise SystemExit("SP3 rounding writer changed its source product")
    other_fields = (
        "position_x_m",
        "position_y_m",
        "position_z_m",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_z_m_s",
        "clock_rate_s_s",
    )
    if any(getattr(report, field) != 0.0 for field in other_fields):
        raise SystemExit("SP3 rounding report changed fields other than the clock")


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
    _check_nonzero_rounding_report()

    print(f"sidereon {sidereon.__version__} release smoke passed")


if __name__ == "__main__":
    main()
