#!/usr/bin/env python3
"""Smoke-test the installed release artifact's PROJ geoid surface."""

from __future__ import annotations

import math
import os
import struct

import sidereon


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

    print(f"sidereon {sidereon.__version__} release smoke passed")


if __name__ == "__main__":
    main()
