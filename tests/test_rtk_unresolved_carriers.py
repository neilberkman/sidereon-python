"""RTK RINEX arcs leave out a satellite with no carrier frequency, not the arc.

R01 is on channel -7 and R28 on the channel 7 real IGS headers give that
extended slot, outside the `-7..=6` FDMA allocation, so R28's GLONASS L1 and L2
phases have no carrier frequency. The arc builder leaves R28 out of the epoch
and reports it for each receiver; a configured pair whose carrier resolves -
the CDMA G3 `L3Q`, which needs no channel - is used instead where the epoch
holds it.
"""

import datetime as dt

import pytest
import sidereon

C_M_S = 299792458.0
G3_HZ = 1202.025e6
GLONASS = sidereon.GnssSystem.GLONASS


def _header_line(content, label):
    return f"{content:<60}{label}"


def _obs(types, r01, r28):
    text = "\n".join(
        [
            _header_line(
                "     3.05           OBSERVATION DATA    R (GLONASS)",
                "RINEX VERSION / TYPE",
            ),
            _header_line(types, "SYS / # / OBS TYPES"),
            _header_line("  2 R01 -7 R28  7", "GLONASS SLOT / FRQ #"),
            _header_line("", "END OF HEADER"),
            "> 2020 01 01 00 00  0.0000000  0  2",
            r01,
            r28,
        ]
    )
    return sidereon.parse_rinex_obs(text)


def _record(sat, *values):
    return sat + "  ".join(f"{value:14.3f}" for value in values)


def _sp3():
    # Twelve 15-minute epochs around 2020-01-01 00:00 GPS time, both slots held
    # still, so the query epoch sits mid-window.
    start = dt.datetime(2019, 12, 31, 22, 30)
    epochs = [start + dt.timedelta(minutes=15 * k) for k in range(12)]
    gps_seconds = (start - dt.datetime(1980, 1, 6)).total_seconds()
    week, sow = divmod(gps_seconds, 7 * 86400)
    mjd = start - dt.datetime(1858, 11, 17)
    lines = [
        f"#cP{start.year:4d}{start.month:3d}{start.day:3d}{start.hour:3d}"
        f"{start.minute:3d}{0.0:12.8f}{len(epochs):8d} ORBIT IGS14 FIT  TST",
        f"## {int(week):4d} {sow:15.8f} {900.0:14.8f} {mjd.days:5d} "
        f"{mjd.seconds / 86400:15.13f}",
        "+    2   R01R28" + "  0" * 15,
        "++       " + "  0" * 17,
        "%c R  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc",
        "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000",
        "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000",
        "%i    0    0    0    0      0      0      0      0         0",
        "%i    0    0    0    0      0      0      0      0         0",
        "/* SYNTHETIC GLONASS FIXTURE",
    ]
    for epoch in epochs:
        lines.append(
            f"*  {epoch.year:4d}{epoch.month:3d}{epoch.day:3d}{epoch.hour:3d}"
            f"{epoch.minute:3d}{0.0:12.8f}"
        )
        for sat in ("R01", "R28"):
            lines.append(
                f"P{sat}{20000.0:14.6f}{10000.0:14.6f}{10000.0:14.6f}{100.0:14.6f}"
            )
    lines.append("EOF")
    return sidereon.load_sp3(("\n".join(lines) + "\n").encode("ascii"))


def test_an_unresolved_carrier_leaves_out_one_satellite_not_the_arc():
    obs = _obs(
        "R    2 C1C L1C",
        _record("R01", 20_000_000.0, 100.0),
        _record("R28", 20_000_000.0, 100.0),
    )
    options = sidereon.RtkRinexArcOptions(
        signal_pairs=[sidereon.RtkRinexSignalPair(GLONASS, "C1C", "L1C")],
        min_common_satellites=1,
        include_prediction_time=False,
    )
    arc = sidereon.build_rinex_rtk_arc(_sp3(), obs, obs, options)

    assert len(arc.epochs) == 1
    assert arc.epochs[0].base_count == 1
    assert list(arc.wavelengths_m) == ["R01"]
    unresolved = arc.unresolved_carriers
    assert [
        (item.receiver, item.epoch_index, item.satellite_id, item.observable_code)
        for item in unresolved
    ] == [
        (sidereon.RtkRinexReceiver.BASE, 0, "R28", "L1C"),
        (sidereon.RtkRinexReceiver.ROVER, 0, "R28", "L1C"),
    ]
    assert unresolved[0] == unresolved[0]
    assert unresolved[0] != unresolved[1]
    assert "RtkRinexUnresolvedCarrier(" in repr(unresolved[0])


def test_dual_frequency_arc_reports_each_unresolved_phase():
    obs = _obs(
        "R    4 C1C L1C C2C L2C",
        _record("R01", 20_000_000.0, 100.0, 20_000_001.0, 90.0),
        _record("R28", 20_000_000.0, 100.0, 20_000_001.0, 90.0),
    )
    options = sidereon.RtkRinexDualArcOptions(
        signal_pairs=[
            sidereon.RtkRinexDualSignalPair(GLONASS, "C1C", "L1C", "C2C", "L2C")
        ],
        min_common_satellites=1,
        include_prediction_time=False,
    )
    arc = sidereon.build_dual_frequency_rinex_rtk_arc(_sp3(), obs, obs, options)

    assert len(arc.epochs) == 1
    assert arc.epochs[0].observation_count == 1
    assert [
        (item.receiver, item.satellite_id, item.observable_code)
        for item in arc.unresolved_carriers
    ] == [
        (sidereon.RtkRinexReceiver.BASE, "R28", "L1C"),
        (sidereon.RtkRinexReceiver.BASE, "R28", "L2C"),
        (sidereon.RtkRinexReceiver.ROVER, "R28", "L1C"),
        (sidereon.RtkRinexReceiver.ROVER, "R28", "L2C"),
    ]


def test_a_later_pair_with_a_resolvable_carrier_is_used_before_reporting():
    obs = _obs(
        "R    4 C1C L1C C3Q L3Q",
        _record("R01", 20_000_000.0, 100.0, 20_000_002.0, 80.0),
        _record("R28", 20_000_000.0, 100.0, 20_000_002.0, 80.0),
    )
    options = sidereon.RtkRinexArcOptions(
        signal_pairs=[
            sidereon.RtkRinexSignalPair(GLONASS, "C1C", "L1C"),
            sidereon.RtkRinexSignalPair(GLONASS, "C3Q", "L3Q"),
        ],
        min_common_satellites=1,
        include_prediction_time=False,
    )
    arc = sidereon.build_rinex_rtk_arc(_sp3(), obs, obs, options)

    assert arc.unresolved_carriers == []
    assert arc.epochs[0].base_count == 2
    assert len(arc.wavelengths_m) == 2
    assert any(
        wavelength == pytest.approx(C_M_S / G3_HZ, abs=1.0e-12)
        for wavelength in arc.wavelengths_m.values()
    )
