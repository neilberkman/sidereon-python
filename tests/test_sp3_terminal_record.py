"""Python parity for the shared exact-SP3 terminal-record contract."""

import datetime as dt
import gzip
import json
from pathlib import Path

import pytest
import sidereon
from sidereon import data, distribution

_GOLDEN = Path(__file__).with_name("golden") / "sp3-terminal-record-v1.json"
_CORPUS = json.loads(_GOLDEN.read_text(encoding="utf-8"))
_START = dt.date(2020, 1, 1)
_P_G01 = "PG01  15000.000000 -20000.000000   5000.000000    123.456789\n"
_P_G02 = "PG02  16000.000000 -21000.000000   6000.000000    124.456789\n"


def _exact_fixture() -> bytes:
    text = (
        "#dP2020  1  1  0  0  0.00000000      12 ORBIT IGS20  FIT TST\n"
        "## 2086 259200.00000000   300.00000000 58849 0.0000000000000\n"
        "+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n"
    )
    text += "+          0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n" * 4
    text += "++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n" * 5
    text += "%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
    text += "%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n"
    text += "%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n"
    text += "%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n"
    text += "%i    0    0    0    0      0      0      0      0         0\n"
    text += "%i    0    0    0    0      0      0      0      0         0\n"
    text += "/* PYTHON TERMINAL RECORD FIXTURE\n" * 4
    for epoch in range(12):
        text += f"*  2020  1  1  0 {epoch * 5:>2}  0.00000000\n"
        text += _P_G01
        text += _P_G02
    return (text + "EOF\n").encode("ascii")


def _case_bytes(base: bytes, case: dict) -> bytes:
    assert base.endswith(b"EOF\n")
    content = bytearray(base[:-4])
    content.extend(bytes.fromhex(case["leading_hex"]))
    if case["marker"] is not None:
        content.extend(case["marker"].encode("ascii"))
    content.extend(b" " * case["padding_spaces"])
    for field in ("suffix_hex", "separator_hex", "trailing_hex"):
        content.extend(bytes.fromhex(case[field]))
    return bytes(content)


def _result_class(content: bytes, request: sidereon.ExactSp3Request) -> str:
    try:
        sidereon.parse_exact_sp3(content, request)
    except sidereon.ExactSp3ValidationError as error:
        message = str(error)
        if "malformed EOF record" in message:
            return "malformed_eof_record"
        if "missing its EOF record" in message:
            return "missing_eof"
        if "nonblank records after EOF" in message:
            return "trailing_content_after_eof"
        raise AssertionError(
            f"terminal corpus reached unrelated error: {message}"
        ) from error
    return "accept"


def _padded_terminal(content: bytes, *, crlf: bool = False) -> bytes:
    lines = content.splitlines()
    assert lines[-1] == b"EOF"
    separator = b"\r\n" if crlf else b"\n"
    return separator.join(lines[:-1]) + separator + b"EOF" + b" " * 77 + separator


def test_shared_terminal_record_corpus_identity():
    assert _CORPUS["schema"] == "sidereon-sp3-terminal-record-v1"
    assert _CORPUS["record_width"] == 80
    assert _CORPUS["record_width_authority"] == "sidereon-interoperability-policy"


@pytest.mark.parametrize(
    "case",
    _CORPUS["cases"],
    ids=[case["name"] for case in _CORPUS["cases"]],
)
def test_parse_exact_sp3_obeys_shared_terminal_record_contract(case):
    request = sidereon.ExactSp3Request(_START, "01H", "05M", issue="0000")

    assert _result_class(_case_bytes(_exact_fixture(), case), request) == case["expect"]


@pytest.mark.parametrize("crlf", [False, True], ids=["lf", "crlf"])
def test_exact_product_acquisition_accepts_record_width_padded_eof(tmp_path, crlf):
    product = data.mgex_sp3("cod", dt.date(2026, 6, 25))
    request = distribution.request(
        product,
        [
            distribution.Distribution.in_memory(
                gzip.compress(
                    _padded_terminal(_catalog_fixture(product.date), crlf=crlf),
                    mtime=0,
                )
            )
        ],
    )

    acquired = distribution.acquire(request, cache_dir=tmp_path)

    assert acquired.provenance.archive_compression == "gzip"
    assert acquired.provenance.resolved_identity.format_version == "SP3-d"
    assert (
        Path(acquired.path)
        .read_bytes()
        .endswith(b"EOF" + b" " * 77 + (b"\r\n" if crlf else b"\n"))
    )


def test_exact_product_acquisition_rejects_truncated_gzip(tmp_path):
    product = data.mgex_sp3("cod", dt.date(2026, 6, 25))
    archive = gzip.compress(
        _padded_terminal(_catalog_fixture(product.date)),
        mtime=0,
    )
    request = distribution.request(
        product,
        [distribution.Distribution.in_memory(archive[:-8])],
    )

    with pytest.raises(distribution.DecompressionFailure, match="truncated"):
        distribution.acquire(request, cache_dir=tmp_path)


def _catalog_fixture(date: dt.date) -> bytes:
    """Reuse the committed full-day SP3 fixture with a catalog-valid date."""
    from _helpers import CORE_FIXTURES, sp3_bytes_for_date

    fixture = (
        Path(CORE_FIXTURES) / "sp3" / "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    ).read_bytes()
    return sp3_bytes_for_date(fixture, date)
