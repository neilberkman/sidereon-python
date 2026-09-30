"""SBAS PRN-mask bits that name no satellite keep their place.

The PRN mask follows the RTCA DO-229 layout: mask numbers 1..37 are GPS,
38..61 GLONASS slots 1..24 and 120..158 SBAS. An active bit that names no
satellite held here - mask number 71 below - keeps its place among the active
bits, so the correction after it still reaches its own satellite, and the
correction addressed to it is counted instead of being dropped silently.
"""

import sidereon

BODY226 = sidereon.SbasWireForm.BODY226


def _body(bits):
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def _mask_body(active, iodp=1):
    mask = ["0"] * 210
    for index in active:
        mask[index] = "1"
    return _body("01010011" + "000001" + "".join(mask) + format(iodp, "02b"))


def _fast_body(prc_counts, iodf=1, iodp=1):
    prc = list(prc_counts) + [0] * (13 - len(prc_counts))
    return _body(
        "01010011"
        + "000010"
        + format(iodf, "02b")
        + format(iodp, "02b")
        + "".join(format(value & 0xFFF, "012b") for value in prc)
        + "0000" * 13
    )


def test_unassigned_mask_bits_keep_later_corrections_on_their_own_satellites():
    # G01, mask number 71 (no satellite), and SBAS PRN 120 (S20).
    mask = sidereon.decode_sbas_block(_mask_body([0, 70, 119]), BODY226)
    assert mask.kind == sidereon.SbasMessageKind.PRN_MASK
    fast = sidereon.decode_sbas_block(_fast_body([8, 16, 24]), BODY226)

    store = sidereon.SbasCorrectionStore()
    assert store.unassigned_mask_corrections("S20") is None
    store.ingest(mask, "S20", 2360, 259200.0)
    assert store.unassigned_mask_corrections("S20") == {}
    store.ingest(fast, "S20", 2360, 259210.0)

    # 0.125 m per count: the third active bit's correction reaches S20.
    assert store.fast("S20", "G01").prc_m == 1.0
    assert store.fast("S20", "S20").prc_m == 3.0
    assert store.unassigned_mask_corrections("S20") == {71: 1}
    assert store.unassigned_mask_corrections("S21") is None


def test_sbas_prn_mapping_is_bounded_by_the_broadcast_window():
    assert sidereon.sbas_prn_to_satellite_id(120) == "S20"
    assert sidereon.sbas_prn_to_satellite_id(158) == "S58"
    assert sidereon.satellite_id_to_sbas_prn("S58") == 158
    # S59..S99 are satellite tokens but no SBAS broadcast PRN names them.
    assert sidereon.satellite_id_to_sbas_prn("S59") is None
    assert sidereon.satellite_id_to_sbas_prn("S99") is None
