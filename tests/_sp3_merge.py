"""Shared builders and golden converters for the SP3 merge audit tests.

The SP3 sources are the ones the core tests
`crates/sidereon-core/tests/sp3_merge_per_epoch_provenance.rs` and
`tests/sp3_merge_coverage.rs` build as text. The coverage fixture's circular
trajectories come as exact bits from the `sp3_merge` section of
`core_goldens.json`, so the text built here is byte for byte the text the core
generator parsed; each test checks that with the digest the generator wrote.

The converters render binding objects in the generator's JSON shape (floats as
their bit patterns, epochs as their split Julian dates) so a whole merge outcome
compares with one equality.
"""

import struct

import sidereon
from _helpers import core_goldens, hex_to_f64

GOLDEN = core_goldens()["sp3_merge"]
COVERAGE = GOLDEN["coverage_fixture"]
PROVENANCE = GOLDEN["provenance_fixture"]
STEP_S = COVERAGE["step_s"]
TRAJECTORY_KM = [
    [[hex_to_f64(value) for value in position] for position in row]
    for row in COVERAGE["trajectory_km"]
]

FULL = "full"
NO_CLOCK = "no_clock"
CLOCK_ONLY = "clock_only"
ABSENT = "absent"

_COVERAGE_HEADER_TAIL = (
    "+    6   G01G02G03G04G05G06  0  0  0  0  0  0  0  0  0  0  0\n"
    "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
    "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
    "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n"
    "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n"
    "%i    0    0    0    0      0      0      0      0         0\n"
    "%i    0    0    0    0      0      0      0      0         0\n"
    "/* SYNTHETIC SP3 COVERAGE FIXTURE\n"
)

_PROVENANCE_HEADER_TAIL = (
    "+    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
    "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
    "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n"
    "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n"
    "%i    0    0    0    0      0      0      0      0         0\n"
    "%i    0    0    0    0      0      0      0      0         0\n"
    "/* TEST SP3-c FIXTURE\n"
)


def fnv1a64(data: bytes) -> str:
    value = 0xCBF29CE484222325
    for byte in data:
        value ^= byte
        value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{value:016x}"


def f64_hex(value: float) -> str:
    return "0x%016x" % struct.unpack(">Q", struct.pack(">d", value))[0]


def optional_hex(value):
    return None if value is None else f64_hex(value)


def _load(fixture, name, text):
    assert fnv1a64(text.encode("ascii")) == fixture["texts_fnv1a64"][name], name
    return sidereon.load_sp3(text.encode("ascii"))


# --- the coverage fixture ---------------------------------------------------


def _epoch_fields(index: int) -> str:
    return f"2020  6 25 {index // 12:2d}{(index % 12) * 5:3d}  0.00000000"


def product_text(indices, offset_m=0.0, interval_s=300.0, record=None) -> str:
    """The product `sp3_merge_coverage.rs` builds, its header interval written
    as `interval_s`; `record(prn, index)` picks what each satellite carries."""
    first = indices[0]
    text = (
        f"#cP{_epoch_fields(first)}     {len(indices):3d} ORBIT IGS14 FIT  TST\n"
        f"## 2111 {345_600.0 + float(first * STEP_S):14.8f}{interval_s:15.8f}"
        f" 59025 {float(first * STEP_S) / 86_400.0:.13f}\n" + _COVERAGE_HEADER_TAIL
    )
    for index in indices:
        text += f"*  {_epoch_fields(index)}\n"
        for prn in range(1, 7):
            x0, y, z = TRAJECTORY_KM[index][prn - 1]
            x = x0 + offset_m / 1000.0
            clock = 10.0 + float(prn)
            kind = FULL if record is None else record(prn, index)
            if kind == FULL:
                text += f"PG{prn:02d}{x:14.6f}{y:14.6f}{z:14.6f}{clock:14.6f}\n"
            elif kind == NO_CLOCK:
                text += (
                    f"PG{prn:02d}{x:14.6f}{y:14.6f}{z:14.6f}{999_999.999_999:14.6f}\n"
                )
            elif kind == CLOCK_ONLY:
                text += f"PG{prn:02d}{0.0:14.6f}{0.0:14.6f}{0.0:14.6f}{clock:14.6f}\n"
            else:
                assert kind == ABSENT
    return text + "EOF\n"


def coverage_product(name, indices, offset_m=0.0, interval_s=300.0, record=None):
    """Build, digest-check against the golden, and parse a coverage product."""
    return _load(COVERAGE, name, product_text(indices, offset_m, interval_s, record))


def coverage_case(name):
    return COVERAGE["cases"][name]


# --- the per-epoch provenance fixture ----------------------------------------


def provenance_source(name, positions_km):
    text = (
        "#cP2020  6 25  0  0  0.00000000       2 ORBIT IGS14 FIT  TST\n"
        "## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n"
        + _PROVENANCE_HEADER_TAIL
        + "*  2020  6 25  0  0  0.00000000\n"
        + f"PG01 {positions_km[0]:13.6f} -20000.000000   5000.000000    100.000000\n"
        + "*  2020  6 25  0 15  0.00000000\n"
        + f"PG01 {positions_km[1]:13.6f} -20000.000000   5000.000000    100.000000\n"
        + "EOF\n"
    )
    return _load(PROVENANCE, name, text)


def provenance_late(name, position_km):
    text = (
        "#cP2020  6 25  0 15  0.00000000       1 ORBIT IGS14 FIT  TST\n"
        "## 2111 432900.00000000   900.00000000 59025 0.0000000000000\n"
        + _PROVENANCE_HEADER_TAIL
        + "*  2020  6 25  0 15  0.00000000\n"
        + f"PG01 {position_km:13.6f} -20000.000000   5000.000000    100.000000\n"
        + "EOF\n"
    )
    return _load(PROVENANCE, name, text)


def provenance_early(name):
    text = (
        "#cP2020  6 25  0  0  0.00000000       1 ORBIT IGS14 FIT  TST\n"
        "## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n"
        + _PROVENANCE_HEADER_TAIL
        + "*  2020  6 25  0  0  0.00000000\n"
        + "PG01  15000.000000 -20000.000000   5000.000000    100.000000\n"
        + "EOF\n"
    )
    return _load(PROVENANCE, name, text)


def provenance_case(name):
    return PROVENANCE["cases"][name]


# --- binding objects in the golden's JSON shape -------------------------------


def epoch(instant):
    if instant.repr_kind == "JulianDate":
        return [f64_hex(instant.jd_whole), f64_hex(instant.fraction)]
    return {"nanos": str(instant.nanos)}


def _optional_epoch(instant):
    return None if instant is None else epoch(instant)


def flags(values):
    return [
        {
            "epoch": epoch(flag.epoch),
            "satellite": flag.satellite,
            "sources": list(flag.sources),
        }
        for flag in values
    ]


def selection(value):
    if value is None:
        return None
    return {
        "kind": value.kind,
        "source": value.selected_source,
        "members": list(value.members),
        "rule": None if value.rule is None else value.rule.label,
    }


def provenance(record):
    if record is None:
        return None
    return {
        "mode": record.mode.label,
        "cells": [
            {
                "epoch": epoch(cell.epoch),
                "satellite": cell.satellite,
                "position": selection(cell.position),
                "clock": selection(cell.clock),
            }
            for cell in record.cells
        ],
        "transitions": [
            {
                "satellite": transition.satellite,
                "epoch": epoch(transition.epoch),
                "from_source": transition.from_source,
                "to_source": transition.to_source,
                "reason": transition.reason,
            }
            for transition in record.transitions
        ],
        "coverage": [
            {
                "source": coverage.source,
                "cells_contributed": coverage.cells_contributed,
                "cells_selected": coverage.cells_selected,
                "first_epoch": _optional_epoch(coverage.first_epoch),
                "last_epoch": _optional_epoch(coverage.last_epoch),
                "cells_absent": coverage.cells_absent,
            }
            for coverage in record.coverage
        ],
    }


_DEFECT_FIELDS = {
    "duplicate_epoch": (("epoch_j2000_s", True), ("occurrences", False)),
    "single_sample_series": (),
    "speed_bound": (
        ("from_j2000_s", True),
        ("to_j2000_s", True),
        ("interval_s", True),
        ("displacement_m", True),
        ("implied_speed_m_s", True),
        ("bound_m_s", True),
    ),
    "hold_out_residual": (
        ("epoch_j2000_s", True),
        ("preceding_j2000_s", True),
        ("residual_m", True),
        ("tolerance_m", True),
    ),
}


def defect(value):
    out = {"kind": value.kind, "satellite": value.satellite}
    for field, is_float in _DEFECT_FIELDS[value.kind]:
        raw = getattr(value, field)
        out[field] = f64_hex(raw) if is_float else raw
    if value.kind == "hold_out_residual":
        out["node_epochs_j2000_s"] = [f64_hex(v) for v in value.node_epochs_j2000_s]
    return out


def continuity(report):
    if report is None:
        return None
    return {
        "attested": report.attested,
        "pairs_checked": report.pairs_checked,
        "residuals_checked": report.residuals_checked,
        "residuals_skipped": report.residuals_skipped,
        "defects": [defect(value) for value in report.defects],
        "violations": [
            {
                "defect": defect(violation.defect),
                "from_sources": list(violation.from_sources),
                "to_sources": list(violation.to_sources),
                "cells": [
                    {
                        "epoch_j2000_s": f64_hex(cell.epoch_j2000_s),
                        "role": cell.role,
                        "selection": selection(cell.selection),
                    }
                    for cell in violation.cells
                ],
                "sources": list(violation.sources),
                "crosses_contributors": violation.crosses_contributors,
            }
            for violation in report.violations
        ],
    }


def report(value):
    return {
        "quarantined": flags(value.quarantined),
        "single_source": flags(value.single_source),
        "position_outliers": flags(value.position_outliers),
        "clock_outliers": flags(value.clock_outliers),
        "arc_withheld": flags(value.arc_withheld),
        "omitted_epochs": [epoch(instant) for instant in value.omitted_epochs],
        "clock_omissions": [
            {
                "epoch": epoch(omission.epoch),
                "satellite": omission.satellite,
                "source": omission.source,
                "reason": omission.reason,
                "preferred_source": omission.preferred_source,
                "cell_has_clock": omission.cell_has_clock,
            }
            for omission in value.clock_omissions
        ],
        "dropped_input_epochs": [
            {
                "source": dropped.source,
                "epoch_index": dropped.epoch_index,
                "epoch": epoch(dropped.epoch),
                "reason": dropped.reason,
            }
            for dropped in value.dropped_input_epochs
        ],
        "provenance": provenance(value.provenance),
        "continuity": continuity(value.continuity),
        "single_source_fraction": optional_hex(value.single_source_fraction),
        "position_agreement_rms_m": optional_hex(value.position_agreement_rms_m),
        "position_agreement_max_m": optional_hex(value.position_agreement_max_m),
        "clock_agreement_rms_s": optional_hex(value.clock_agreement_rms_s),
        "clock_agreement_max_s": optional_hex(value.clock_agreement_max_s),
        "per_epoch_agreement": [
            {
                "epoch": [f64_hex(entry.jd_whole), f64_hex(entry.jd_fraction)],
                "satellites": entry.satellites,
                "position_rms_m": optional_hex(entry.position_rms_m),
                "position_max_m": optional_hex(entry.position_max_m),
                "clock_rms_s": optional_hex(entry.clock_rms_s),
                "clock_max_s": optional_hex(entry.clock_max_s),
            }
            for entry in value.agreement_epochs
        ],
    }


def text_or_none(sp3):
    try:
        return sp3.to_sp3_string()
    except sidereon.Sp3WriteError:
        return None


def product(sp3):
    text = text_or_none(sp3)
    header = sp3.header
    return {
        "epochs": [epoch(instant) for instant in sp3.epoch_instants],
        "num_epochs": header.num_epochs,
        "epoch_interval_s": f64_hex(header.epoch_interval_s),
        "mjd": header.mjd,
        "mjd_fraction": f64_hex(header.mjd_fraction),
        "seconds_of_week": f64_hex(header.seconds_of_week),
        "declared_start_j2000_s": optional_hex(sp3.declared_start_j2000_s),
        "text_fnv1a64": None if text is None else fnv1a64(text.encode("ascii")),
    }


def merged(sources, options):
    """A merge outcome in the golden's shape, and the binding's result."""
    try:
        merged_product, merged_report = sidereon.merge_sp3(sources, options)
    except ValueError as error:
        return {"error": str(error)}, None
    return (
        {"product": product(merged_product), "report": report(merged_report)},
        (merged_product, merged_report),
    )


def _channel(value):
    return {
        "epochs": value.epochs,
        "spans": [
            {
                "first_index": span.first_index,
                "last_index": span.last_index,
                "first_epoch": epoch(span.first_epoch),
                "last_epoch": epoch(span.last_epoch),
            }
            for span in value.spans
        ],
        "gaps": [
            {
                "after_index": gap.after_index,
                "before_index": gap.before_index,
                "missing_epochs": gap.missing_epochs,
            }
            for gap in value.gaps
        ],
    }


def coverage(value):
    return {
        "grid": {
            "interval_s": optional_hex(value.grid.interval_s),
            "agrees_with_header": value.grid.agrees_with_header,
            "out_of_order": list(value.grid.out_of_order),
            "unplaced": list(value.grid.unplaced),
        },
        "satellites": [
            {
                "satellite": satellite.satellite,
                "declared": satellite.declared,
                "positions": _channel(satellite.positions),
                "clocks": _channel(satellite.clocks),
            }
            for satellite in value.satellites
        ],
    }


def node_selection(nodes, sp3, windows):
    axis = sp3.epochs_j2000_seconds
    return [
        {
            "from_index": first,
            "through_index": last,
            "selected_nodes": {
                satellite: [
                    f64_hex(value)
                    for value in nodes.selected_nodes(
                        satellite, float(axis[first]), float(axis[last])
                    )
                ]
                for satellite in sp3.satellites
            },
        }
        for first, last in windows
    ]
