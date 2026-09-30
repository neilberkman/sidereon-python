"""SP3 header descriptors, clock-only records, clock rates and writer refusals.

Every product here is SP3-c text built in the test, in the fixed columns the
core parser reads, so each case states exactly which field carries the value
under test.
"""

import numpy as np
import pytest
import sidereon

_PF_LINE = "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000"
_G01 = ("G01", (15000.0, -20000.0, 5000.0), 100.0)
_G02_CLOCK_ONLY = ("G02", (0.0, 0.0, 0.0), 250.0)


def _sp3_text(
    records,
    *,
    kind="P",
    agency="TST",
    data_used="ORBIT",
    file_type="G",
    pf_line=_PF_LINE,
    velocities=None,
    flags=None,
):
    """One-epoch SP3-c product.

    ``records`` is ``(satellite, position_km, clock_us)``; ``velocities`` maps a
    satellite to ``(velocity_dm_s, clock_rate)`` for a paired ``V`` record, and
    ``flags`` maps a satellite to the text written from column 61 onward.
    """
    velocities = velocities or {}
    flags = flags or {}
    sats = [sat for sat, _, _ in records]
    sat_field = "".join(sats) + "  0" * (17 - len(sats))
    line1 = f"#c{kind}2020  6 25  0  0  0.00000000       1 {data_used:<5} IGS14 FIT"
    lines = [
        f"{line1}  {agency}",
        "## 2111 432000.00000000   900.00000000 59025 0.0000000000000",
        f"+   {len(sats):2}   {sat_field}",
        "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0",
        f"%c {file_type:<2} cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        pf_line,
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
        "/* TEST SP3-c FIXTURE",
        "*  2020  6 25  0  0  0.00000000",
    ]
    for sat, (x, y, z), clock_us in records:
        lines.append(
            f"P{sat}{x:14.6f}{y:14.6f}{z:14.6f}{clock_us:14.6f}{flags.get(sat, '')}"
        )
        if sat in velocities:
            (vx, vy, vz), rate = velocities[sat]
            lines.append(f"V{sat}{vx:14.6f}{vy:14.6f}{vz:14.6f}{rate:14.6f}")
    lines.append("EOF")
    return ("\n".join(lines) + "\n").encode("ascii")


def _load(records, **kwargs):
    return sidereon.load_sp3(_sp3_text(records, **kwargs))


def _write_refusal(sp3):
    with pytest.raises(sidereon.Sp3WriteError) as excinfo:
        sp3.to_sp3_string()
    error = excinfo.value
    detail = error.detail
    assert isinstance(detail, sidereon.Sp3WriteErrorDetail)
    assert detail.message == str(error)
    assert detail == detail
    assert repr(detail).startswith(f'Sp3WriteErrorDetail(kind="{detail.kind}"')
    return detail


# --- header ------------------------------------------------------------------


def test_header_retains_descriptors_and_the_writer_states_them_back():
    sp3 = _load([_G01])
    header = sp3.header

    assert header.version == "c"
    assert header.data_type == "P"
    assert header.num_epochs == 1
    assert header.data_used == "ORBIT"
    assert header.coordinate_system == "IGS14"
    assert header.orbit_type == "FIT"
    assert header.agency == "TST"
    assert header.gnss_week == 2111
    assert header.seconds_of_week == 432000.0
    assert header.epoch_interval_s == 900.0
    assert header.mjd == 59025
    assert header.mjd_fraction == 0.0
    assert header.file_type == "G"
    assert header.time_system == "GPS"
    assert header.time_scale == sidereon.TimeScale.GPST
    assert header.pos_vel_base == 1.25
    assert header.clock_rate_base == 1.025
    assert header.satellites == ["G01"]
    assert header.satellite_accuracy_codes == [0]
    assert sp3.comments == ["TEST SP3-c FIXTURE"]
    assert sp3.skipped_records == 0

    text = sp3.to_sp3_string()
    lines = text.splitlines()
    assert lines[0].endswith(" ORBIT IGS14 FIT  TST")
    assert "%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc" in lines
    assert _PF_LINE in lines
    assert sidereon.load_sp3(text.encode("ascii")).header == header


def test_blank_header_descriptors_stay_absent_through_a_write():
    # Columns 4-13 and 15-26 blank: no base declared.
    blank_pf = "%f" + " " * 24 + "  0.00000000000  0.000000000000000"
    assert blank_pf[28:] == _PF_LINE[28:]
    sp3 = _load([_G01], data_used="", file_type="", pf_line=blank_pf)
    header = sp3.header

    assert header.data_used is None
    assert header.file_type is None
    assert header.pos_vel_base is None
    assert header.clock_rate_base is None
    assert header.time_system == "GPS"

    again = sidereon.load_sp3(sp3.to_sp3_string().encode("ascii"))
    assert again.header == header


# --- clock-only records ------------------------------------------------------


def test_clock_beside_a_missing_orbit_is_retained_and_written_back():
    flags = {"G02": " " * 14 + "E"}
    sp3 = _load([_G01, _G02_CLOCK_ONLY], flags=flags)

    assert sp3.satellites == ["G01", "G02"]
    with pytest.raises(KeyError):
        sp3.state("G02", 0)
    with pytest.raises(KeyError):
        sp3.clock_record("G01", 0)
    with pytest.raises(IndexError):
        sp3.clock_record("G02", 1)
    with pytest.raises(IndexError):
        sp3.clock_records_at(1)

    record = sp3.clock_record("G02", 0)
    assert record.clock_us == 250.0
    assert record.clock_s == pytest.approx(250.0e-6, rel=1.0e-15)
    assert record.velocity_m_s is None
    assert record.clock_rate_s_s is None
    assert record.clock_rate_raw is None
    assert record.clock_event is True
    assert record.clock_predicted is False
    assert record.maneuver is False
    assert record.orbit_predicted is False

    records = sp3.clock_records_at(0)
    assert list(records) == ["G02"]
    assert records["G02"] == record

    text = sp3.to_sp3_string()
    g02_line = (
        "PG02      0.000000      0.000000      0.000000    250.000000" + flags["G02"]
    )
    assert g02_line in text.splitlines()
    again = sidereon.load_sp3(text.encode("ascii"))
    assert again.clock_record("G02", 0) == record
    with pytest.raises(KeyError):
        again.state("G02", 0)


def test_clock_rate_reaches_states_and_clock_only_records():
    sp3 = _load(
        [_G01, _G02_CLOCK_ONLY],
        kind="V",
        velocities={
            "G01": ((10000.0, 20000.0, -5000.0), 12.5),
            "G02": ((0.0, 0.0, 0.0), 3.0),
        },
    )
    assert sp3.header.data_type == "V"

    state = sp3.state("G01", 0)
    np.testing.assert_allclose(
        state.velocity_m_s, [1000.0, 2000.0, -500.0], rtol=0.0, atol=1.0e-9
    )
    assert state.clock_rate_s_s == pytest.approx(12.5e-10, rel=1.0e-15)

    # The all-zero velocity is the format's "no velocity"; the rate beside it
    # is still a rate.
    record = sp3.clock_record("G02", 0)
    assert record.velocity_m_s is None
    assert record.clock_rate_raw == 3.0
    assert record.clock_rate_s_s == pytest.approx(3.0e-10, rel=1.0e-15)

    again = sidereon.load_sp3(sp3.to_sp3_string().encode("ascii"))
    assert again.state("G01", 0).clock_rate_s_s == state.clock_rate_s_s
    assert again.clock_record("G02", 0) == record


# --- writer refusals ---------------------------------------------------------


def test_write_error_is_an_sp3_error_and_a_value_error():
    assert issubclass(sidereon.Sp3WriteError, sidereon.Sp3ParseError)
    assert issubclass(sidereon.Sp3WriteError, sidereon.SidereonError)
    assert issubclass(sidereon.Sp3WriteError, ValueError)
    assert sidereon.Sp3WriteError.detail is None
    assert sidereon.Sp3WriteError("hand built").detail is None


def test_writer_refuses_a_header_base_finer_than_its_field():
    # Columns 4-13 hold eight decimals; the canonical field is F10.7.
    finer = "%f 1.25000001  1.025000000  0.00000000000  0.000000000000000"
    sp3 = _load([_G01], pf_line=finer)
    assert sp3.header.pos_vel_base == 1.25000001

    detail = _write_refusal(sp3)
    assert detail.kind == "PrecisionNotRepresentable"
    assert detail.field == "pos/vel base"
    assert detail.columns == 10
    assert detail.decimals == 7
    assert detail.epoch_index is None
    assert detail.satellite is None
    assert detail.details() == {
        "field": "pos/vel base",
        "columns": 10,
        "decimals": 7,
        "value": 1.25000001,
    }


def test_writer_refuses_text_wider_than_its_columns():
    sp3 = _load([_G01], agency="TSTAB")
    assert sp3.header.agency == "TSTAB"

    detail = _write_refusal(sp3)
    assert detail.kind == "TextTooWide"
    assert detail.field == "agency"
    assert detail.columns == 4
    assert detail.decimals is None
    assert detail.details() == {"field": "agency", "columns": 4, "value": "TSTAB"}


def test_writer_refuses_velocity_accuracy_in_a_position_product():
    sp3 = _load(
        [_G01], kind="P", velocities={"G01": ((10000.0, 20000.0, -5000.0), 12.5)}
    )
    assert sp3.state("G01", 0).velocity_m_s is not None

    detail = _write_refusal(sp3)
    assert detail.kind == "AccuracyRecordMismatch"
    assert detail.field is None
    assert detail.satellite == "G01"
    assert detail.epoch_index == 0
    assert detail.columns is None
    assert detail.details() == {
        "satellite": "G01",
        "epoch_index": 0,
    }


def test_writer_refuses_a_mean_merged_position_finer_than_its_column():
    first = _load([("G01", (15000.000001, -20000.0, 5000.0), 100.0)])
    second = _load([("G01", (15000.000002, -20000.0, 5000.0), 100.0)])
    merged, report = sidereon.merge_sp3([first, second])
    assert report.agreement[0].position_members == 2
    x_m = float(merged.state("G01", 0).position_m[0])

    detail = _write_refusal(merged)
    assert detail.kind == "RecordValueNotRepresentable"
    assert detail.field == "position x"
    assert detail.satellite == "G01"
    assert detail.epoch_index == 0
    assert detail.columns == 14
    assert detail.decimals == 6
    details = detail.details()
    assert details["stored"] == x_m
    assert details["column_value"] == x_m / 1000.0
    assert set(details) == {
        "field",
        "satellite",
        "epoch_index",
        "columns",
        "decimals",
        "stored",
        "column_value",
    }
