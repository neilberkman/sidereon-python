"""Comprehensive tests for CCSDS TDM 3.0 bindings, policies, and mutation semantics.

Covers:
1. Strict default refusals and explicit policy-aware reading with positioned comments.
2. Table-driven coverage of all 9 warning variants and all 9 departure variants with
   exact ordered payloads, coexistence, and multiple same-variant/record retention.
3. Representative error cases covering all exposed payload accessors and named strict
   refusals.
4. Exception class attribute defaults and instance context verification.
5. Raw field construction, derived properties, atomic mutations, and duplicate
   retention, distinguishing sole participant removal from dangling PATH with a
   surviving participant.
6. Deterministic comment offset shifting during field insertion, deletion, and trailing
   append.
7. Lenient construction and editing, strict serialization refusal, and matching policy
   roundtrip.
8. UTC vs. GLONASS distinct leap second syntax (second 60), 38-digit fractional seconds,
   and continuous-scale refusals.
9. Unit mismatch writer refusals, getter copy semantics, and annex_e_01 integration.
"""

from pathlib import Path

import pytest
import sidereon

FIXTURES = Path(__file__).with_name("fixtures")


def _valid_header_lines():
    return (
        "CCSDS_TDM_VERS = 2.0\nCREATION_DATE = 2026-160T20:15:00Z\nORIGINATOR = NASA\n"
    )


def _valid_tdm():
    meta = sidereon.TdmMetadata(
        [
            sidereon.TdmField("TIME_SYSTEM", "UTC"),
            sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
        ]
    )
    data = sidereon.TdmDataSection(
        [
            sidereon.TdmDataRecord(
                sidereon.TdmObservable.range(),
                "RANGE",
                "2026-160T20:15:00",
                sidereon.TdmScalar("1000.0", 1000.0),
                sidereon.TdmUnit("km"),
            )
        ]
    )
    seg = sidereon.TdmSegment(meta, data)
    return sidereon.Tdm(
        "2.0",
        [seg],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )


# ---------------------------------------------------------------------------
# Section 1: Strict default refusal & policy-aware read
# ---------------------------------------------------------------------------


def test_tdm_strict_default_refuses_missing_mandatory_keyword():
    raw_missing_time_system = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_info:
        sidereon.parse_tdm_kvn(raw_missing_time_system)

    err = exc_info.value
    assert hasattr(err, "detail")
    assert err.detail is not None
    assert err.detail.kind == "missing_keyword"
    assert err.detail.keyword == "TIME_SYSTEM"
    assert err.detail.segment == 1


def test_tdm_policy_read_retains_positioned_comments_and_diagnostics():
    raw_lenient = (
        "CCSDS_TDM_VERS = 2.0\n"
        "COMMENT Leading header comment\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "COMMENT Leading metadata comment\n"
        "TIME_SYSTEM = UTC\n"
        "COMMENT Interleaved metadata comment\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "RANGE = 2026-160T20:14:00 2000.0\n"
        "RANGE = 2026-160T20:14:00 2000.0\n"
        "DATA_STOP\n"
    )

    # CCSDS 503.0-B-2 4.5.2 puts a metadata comment between META_START and
    # the first metadata keyword, and table 3-3 ranks COMMENT first, so the
    # comment at line 8, after TIME_SYSTEM, is a keyword out of order. The
    # strict reader refuses it by name; keeping it where it sits takes a
    # policy that forgives keyword order.
    strict_order = (
        sidereon.TdmPolicy.strict()
        .with_record_order(sidereon.TdmLeniency.FORGIVE)
        .with_duplicate_records(sidereon.TdmLeniency.FORGIVE)
    )
    with pytest.raises(sidereon.TdmParseError) as refused:
        sidereon.parse_tdm_kvn_with_policy(raw_lenient, strict_order)
    assert refused.value.detail.kind == "keyword_out_of_order"
    assert refused.value.detail.keyword == "COMMENT"
    assert refused.value.detail.section == "metadata"
    assert refused.value.detail.line == 8

    policy = strict_order.with_keyword_order(sidereon.TdmLeniency.FORGIVE)

    result = sidereon.parse_tdm_kvn_with_policy(raw_lenient, policy)
    assert isinstance(result, sidereon.TdmParseResult)
    tdm = result.message
    warnings = result.warnings

    records = tdm.segments[0].data.records
    assert len(records) == 3
    assert records[0].epoch == "2026-160T20:15:00"
    assert records[1].epoch == "2026-160T20:14:00"
    assert records[2].epoch == "2026-160T20:14:00"

    header_comments = tdm.comments
    assert len(header_comments) == 1
    assert header_comments[0].text == "Leading header comment"
    assert header_comments[0].before_record == 1

    meta_comments = tdm.segments[0].metadata.comments
    assert len(meta_comments) == 2
    assert meta_comments[0].text == "Leading metadata comment"
    assert meta_comments[0].before_record == 0
    assert meta_comments[1].text == "Interleaved metadata comment"
    assert meta_comments[1].before_record == 1

    warning_kinds = [w.kind for w in warnings]
    assert "records_out_of_order" in warning_kinds
    assert "duplicate_record" in warning_kinds
    assert "keyword_out_of_order" in warning_kinds

    order_warn = next(w for w in warnings if w.kind == "keyword_out_of_order")
    assert order_warn.line == 8
    assert order_warn.keyword == "COMMENT"
    assert order_warn.section == "metadata"

    out_of_order_warn = next(w for w in warnings if w.kind == "records_out_of_order")
    assert out_of_order_warn.segment == 1
    assert out_of_order_warn.keyword == "RANGE"
    assert out_of_order_warn.epoch == "2026-160T20:14:00"

    dup_warn = next(w for w in warnings if w.kind == "duplicate_record")
    assert dup_warn.segment == 1
    assert dup_warn.keyword == "RANGE"
    assert dup_warn.epoch == "2026-160T20:14:00"


# ---------------------------------------------------------------------------
# Section 2: Table-driven warning and departure variant coverage & coexistence
# ---------------------------------------------------------------------------


def test_tdm_table_driven_all_9_warning_variants():
    base_header = _valid_header_lines()
    cases = [
        (
            "non_printable_character",
            (
                "CCSDS_TDM_VERS = 2.0\n"
                "CREATION_DATE = 2026-160T20:15:00Z\n"
                "ORIGINATOR = NASA\x07\n"
                "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict().with_non_printable(
                sidereon.TdmLeniency.FORGIVE
            ),
            {
                "kind": "non_printable_character",
                "line": 3,
                "column": 18,
                "character": "\x07",
                "keyword": "ORIGINATOR",
            },
        ),
        (
            "line_too_long",
            (
                "CCSDS_TDM_VERS = 2.0\n"
                "COMMENT " + ("A" * 250) + "\n"
                "CREATION_DATE = 2026-160T20:15:00Z\n"
                "ORIGINATOR = NASA\n"
                "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict().with_long_lines(sidereon.TdmLeniency.FORGIVE),
            {
                "kind": "line_too_long",
                "line": 2,
                "length": 258,
                "keyword": "COMMENT",
            },
        ),
        (
            "repeated_keyword",
            (
                base_header + "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "PARTICIPANT_1 = DSS-25\n"
                "MODE = REAL_TIME\n"
                "MODE = REAL_TIME\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict(),
            {
                "kind": "repeated_keyword",
                "line": 8,
                "keyword": "MODE",
                "section": "metadata",
            },
        ),
        (
            "missing_keyword",
            (
                base_header + "META_START\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict().with_missing_keywords(
                sidereon.TdmLeniency.FORGIVE
            ),
            {
                "kind": "missing_keyword",
                "keyword": "TIME_SYSTEM",
                "segment": 1,
            },
        ),
        (
            "empty_data_section",
            (
                base_header + "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict().with_empty_data_sections(
                sidereon.TdmLeniency.FORGIVE
            ),
            {
                "kind": "empty_data_section",
                "segment": 1,
            },
        ),
        (
            "records_out_of_order",
            (
                base_header + "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "RANGE = 2026-160T20:14:00 2000.0\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict().with_record_order(sidereon.TdmLeniency.FORGIVE),
            {
                "kind": "records_out_of_order",
                "segment": 1,
                "keyword": "RANGE",
                "epoch": "2026-160T20:14:00",
            },
        ),
        (
            "duplicate_record",
            (
                base_header + "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict().with_duplicate_records(
                sidereon.TdmLeniency.FORGIVE
            ),
            {
                "kind": "duplicate_record",
                "segment": 1,
                "keyword": "RANGE",
                "epoch": "2026-160T20:15:00",
            },
        ),
        (
            "keyword_out_of_order",
            (
                base_header + "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "TRACK_ID = TRK_01\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "DATA_STOP\n"
            ),
            sidereon.TdmPolicy.strict().with_keyword_order(
                sidereon.TdmLeniency.FORGIVE
            ),
            {
                "kind": "keyword_out_of_order",
                "line": 6,
                "keyword": "TRACK_ID",
                "section": "metadata",
            },
        ),
        (
            "unterminated_final_line",
            (
                base_header + "META_START\n"
                "TIME_SYSTEM = UTC\n"
                "PARTICIPANT_1 = DSS-25\n"
                "META_STOP\n"
                "DATA_START\n"
                "RANGE = 2026-160T20:15:00 1000.0\n"
                "DATA_STOP"
            ),
            sidereon.TdmPolicy.strict().with_final_terminator(
                sidereon.TdmLeniency.FORGIVE
            ),
            {
                "kind": "unterminated_final_line",
                "line": 10,
            },
        ),
    ]

    for name, raw_kvn, policy, expected in cases:
        res = sidereon.parse_tdm_kvn_with_policy(raw_kvn, policy)
        assert len(res.warnings) == 1, f"Failed on {name}"
        w = res.warnings[0]
        assert w.kind == expected["kind"], f"Kind mismatch on {name}"
        for attr, val in expected.items():
            assert getattr(w, attr) == val, f"Attr {attr} mismatch on {name}"
        assert str(w) == w.message
        assert repr(w).startswith("TdmWarning")


def test_tdm_warning_coexistence_and_multiple_record_retention():
    raw = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "MODE = REAL_TIME\n"
        "MODE = REAL_TIME\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "RANGE = 2026-160T20:14:00 2000.0\n"
        "RANGE = 2026-160T20:13:00 3000.0\n"
        "RANGE = 2026-160T20:13:00 3000.0\n"
        "DATA_STOP"
    )
    policy = (
        sidereon.TdmPolicy.strict()
        .with_record_order(sidereon.TdmLeniency.FORGIVE)
        .with_duplicate_records(sidereon.TdmLeniency.FORGIVE)
        .with_final_terminator(sidereon.TdmLeniency.FORGIVE)
    )
    res = sidereon.parse_tdm_kvn_with_policy(raw, policy)

    records = res.message.segments[0].data.records
    assert len(records) == 4
    assert [r.epoch for r in records] == [
        "2026-160T20:15:00",
        "2026-160T20:14:00",
        "2026-160T20:13:00",
        "2026-160T20:13:00",
    ]

    expected_kinds = [
        "repeated_keyword",
        "records_out_of_order",
        "records_out_of_order",
        "duplicate_record",
        "records_out_of_order",
        "unterminated_final_line",
    ]
    assert [w.kind for w in res.warnings] == expected_kinds
    assert res.warnings[0].keyword == "MODE"
    assert res.warnings[1].epoch == "2026-160T20:14:00"
    assert res.warnings[2].epoch == "2026-160T20:13:00"
    assert res.warnings[3].epoch == "2026-160T20:13:00"
    assert res.warnings[4].epoch == "2026-160T20:13:00"
    assert res.warnings[5].line == 15


def test_tdm_table_driven_all_9_departure_variants():
    # 1. non_printable_character
    tdm_bad_char = sidereon.Tdm(
        "2.0",
        [_valid_tdm().segments[0]],
        creation_date="2026-160T20:15:00Z",
        originator="NASA\x07",
    )
    wp_char = sidereon.TdmWritePolicy.strict().with_non_printable(
        sidereon.TdmLeniency.FORGIVE
    )
    res_char = tdm_bad_char.to_kvn_string_with_policy(wp_char)
    assert len(res_char.departures) == 1
    d_char = res_char.departures[0]
    assert d_char.kind == "non_printable_character"
    assert d_char.keyword == "ORIGINATOR"
    assert d_char.character == "\x07"

    # 2. line_too_long
    tdm_long = sidereon.Tdm(
        "2.0",
        [_valid_tdm().segments[0]],
        comments=[sidereon.TdmComment("A" * 250, before_record=1)],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    wp_long = sidereon.TdmWritePolicy.strict().with_long_lines(
        sidereon.TdmLeniency.FORGIVE
    )
    res_long = tdm_long.to_kvn_string_with_policy(wp_long)
    assert len(res_long.departures) == 1
    d_long = res_long.departures[0]
    assert d_long.kind == "line_too_long"
    assert d_long.keyword == "COMMENT"
    assert d_long.length == 258

    # 3. repeated_keyword
    fields_repeat = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
        sidereon.TdmField("MODE", "REAL_TIME"),
        sidereon.TdmField("MODE", "REAL_TIME"),
    ]
    wp_rep = sidereon.TdmWritePolicy.strict().with_repeated_keywords(
        sidereon.TdmLeniency.FORGIVE
    )
    meta_rep = sidereon.TdmMetadata.from_raw_with_policy(
        fields_repeat, [], wp_rep
    ).metadata
    seg_rep = sidereon.TdmSegment(meta_rep, _valid_tdm().segments[0].data)
    tdm_rep = sidereon.Tdm(
        "2.0",
        [seg_rep],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    res_rep = tdm_rep.to_kvn_string_with_policy(wp_rep)
    assert len(res_rep.departures) == 1
    d_rep = res_rep.departures[0]
    assert d_rep.kind == "repeated_keyword"
    assert d_rep.keyword == "MODE"
    assert d_rep.section == "metadata"

    # 4. missing_keyword
    fields_miss = [sidereon.TdmField("PARTICIPANT_1", "DSS-25")]
    wp_miss = sidereon.TdmWritePolicy.strict().with_missing_keywords(
        sidereon.TdmLeniency.FORGIVE
    )
    meta_miss = sidereon.TdmMetadata.from_raw_with_policy(
        fields_miss, [], wp_miss
    ).metadata
    seg_miss = sidereon.TdmSegment(meta_miss, _valid_tdm().segments[0].data)
    tdm_miss = sidereon.Tdm(
        "2.0",
        [seg_miss],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    res_miss = tdm_miss.to_kvn_string_with_policy(wp_miss)
    assert len(res_miss.departures) == 1
    d_miss = res_miss.departures[0]
    assert d_miss.kind == "missing_keyword"
    assert d_miss.keyword == "TIME_SYSTEM"
    assert d_miss.segment == 1

    # 5. empty_data_section
    seg_empty = sidereon.TdmSegment(
        _valid_tdm().segments[0].metadata,
        sidereon.TdmDataSection([]),
    )
    tdm_empty = sidereon.Tdm(
        "2.0",
        [seg_empty],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    wp_empty = sidereon.TdmWritePolicy.strict().with_empty_data_sections(
        sidereon.TdmLeniency.FORGIVE
    )
    res_empty = tdm_empty.to_kvn_string_with_policy(wp_empty)
    assert len(res_empty.departures) == 1
    d_empty = res_empty.departures[0]
    assert d_empty.kind == "empty_data_section"
    assert d_empty.segment == 1

    # 6. records_out_of_order
    data_order = sidereon.TdmDataSection(
        [
            sidereon.TdmDataRecord(
                sidereon.TdmObservable.range(),
                "RANGE",
                "2026-160T20:15:00",
                sidereon.TdmScalar("1000.0", 1000.0),
                sidereon.TdmUnit("km"),
            ),
            sidereon.TdmDataRecord(
                sidereon.TdmObservable.range(),
                "RANGE",
                "2026-160T20:14:00",
                sidereon.TdmScalar("2000.0", 2000.0),
                sidereon.TdmUnit("km"),
            ),
        ]
    )
    seg_order = sidereon.TdmSegment(_valid_tdm().segments[0].metadata, data_order)
    tdm_order = sidereon.Tdm(
        "2.0",
        [seg_order],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    wp_order = sidereon.TdmWritePolicy.strict().with_record_order(
        sidereon.TdmLeniency.FORGIVE
    )
    res_order = tdm_order.to_kvn_string_with_policy(wp_order)
    assert len(res_order.departures) == 1
    d_order = res_order.departures[0]
    assert d_order.kind == "records_out_of_order"
    assert d_order.segment == 1
    assert d_order.keyword == "RANGE"
    assert d_order.epoch == "2026-160T20:14:00"

    # 7. duplicate_record
    rec = sidereon.TdmDataRecord(
        sidereon.TdmObservable.range(),
        "RANGE",
        "2026-160T20:15:00",
        sidereon.TdmScalar("1000.0", 1000.0),
        sidereon.TdmUnit("km"),
    )
    seg_dup = sidereon.TdmSegment(
        _valid_tdm().segments[0].metadata,
        sidereon.TdmDataSection([rec, rec]),
    )
    tdm_dup = sidereon.Tdm(
        "2.0",
        [seg_dup],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    wp_dup = sidereon.TdmWritePolicy.strict().with_duplicate_records(
        sidereon.TdmLeniency.FORGIVE
    )
    res_dup = tdm_dup.to_kvn_string_with_policy(wp_dup)
    assert len(res_dup.departures) == 1
    d_dup = res_dup.departures[0]
    assert d_dup.kind == "duplicate_record"
    assert d_dup.segment == 1
    assert d_dup.keyword == "RANGE"
    assert d_dup.epoch == "2026-160T20:15:00"

    # 8. keyword_out_of_order
    fields_ooo = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("TRACK_ID", "TRK_01"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
    ]
    wp_ooo = sidereon.TdmWritePolicy.strict().with_keyword_order(
        sidereon.TdmLeniency.FORGIVE
    )
    meta_ooo = sidereon.TdmMetadata.from_raw_with_policy(
        fields_ooo, [], wp_ooo
    ).metadata
    seg_ooo = sidereon.TdmSegment(meta_ooo, _valid_tdm().segments[0].data)
    tdm_ooo = sidereon.Tdm(
        "2.0",
        [seg_ooo],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    res_ooo = tdm_ooo.to_kvn_string_with_policy(wp_ooo)
    assert len(res_ooo.departures) == 1
    d_ooo = res_ooo.departures[0]
    assert d_ooo.kind == "keyword_out_of_order"
    assert d_ooo.keyword == "TRACK_ID"
    assert d_ooo.section == "metadata"

    # 9. unterminated_final_line
    wp_term = sidereon.TdmWritePolicy.strict().with_final_terminator(
        sidereon.TdmLeniency.FORGIVE
    )
    res_term = _valid_tdm().to_kvn_string_with_policy(wp_term)
    assert len(res_term.departures) == 1
    d_term = res_term.departures[0]
    assert d_term.kind == "unterminated_final_line"
    assert not res_term.text.endswith("\n")


def test_tdm_departure_coexistence_and_payload_ordering():
    fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
        sidereon.TdmField("MODE", "REAL_TIME"),
        sidereon.TdmField("MODE", "REAL_TIME"),
    ]
    wp = (
        sidereon.TdmWritePolicy.strict()
        .with_repeated_keywords(sidereon.TdmLeniency.FORGIVE)
        .with_record_order(sidereon.TdmLeniency.FORGIVE)
        .with_duplicate_records(sidereon.TdmLeniency.FORGIVE)
        .with_final_terminator(sidereon.TdmLeniency.FORGIVE)
    )
    meta = sidereon.TdmMetadata.from_raw_with_policy(fields, [], wp).metadata
    data = sidereon.TdmDataSection(
        [
            sidereon.TdmDataRecord(
                sidereon.TdmObservable.range(),
                "RANGE",
                "2026-160T20:15:00",
                sidereon.TdmScalar("1000.0", 1000.0),
                sidereon.TdmUnit("km"),
            ),
            sidereon.TdmDataRecord(
                sidereon.TdmObservable.range(),
                "RANGE",
                "2026-160T20:14:00",
                sidereon.TdmScalar("2000.0", 2000.0),
                sidereon.TdmUnit("km"),
            ),
            sidereon.TdmDataRecord(
                sidereon.TdmObservable.range(),
                "RANGE",
                "2026-160T20:14:00",
                sidereon.TdmScalar("2000.0", 2000.0),
                sidereon.TdmUnit("km"),
            ),
        ]
    )
    seg = sidereon.TdmSegment(meta, data)
    tdm = sidereon.Tdm(
        "2.0",
        [seg],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    res = tdm.to_kvn_string_with_policy(wp)

    assert [d.kind for d in res.departures] == [
        "repeated_keyword",
        "records_out_of_order",
        "duplicate_record",
        "records_out_of_order",
        "unterminated_final_line",
    ]
    assert res.departures[0].keyword == "MODE"
    assert res.departures[1].epoch == "2026-160T20:14:00"
    assert res.departures[2].epoch == "2026-160T20:14:00"
    assert res.departures[3].epoch == "2026-160T20:14:00"
    assert not res.text.endswith("\n")


# ---------------------------------------------------------------------------
# Section 3: Representative errors & all payload accessors
# ---------------------------------------------------------------------------


def test_tdm_representative_errors_and_all_payload_accessors():
    # 1. unwritable: reason, keyword
    with pytest.raises(sidereon.TdmValidationError) as exc_unwritable:
        sidereon.TdmMetadata(
            [
                sidereon.TdmField("TIME_SYSTEM", "UTC"),
                sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
            ],
            [sidereon.TdmComment("out of bounds comment", 99)],
        )
    detail = exc_unwritable.value.detail
    assert detail.kind == "unwritable"
    assert detail.keyword == "COMMENT"
    assert detail.reason == "comment position is out of bounds"
    assert "out of bounds" in detail.message
    assert str(detail) == detail.message
    assert repr(detail).startswith("TdmErrorDetail")

    # 2. malformed_line: text, line
    raw_bad_line = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "MALFORMED_LINE_NO_EQUALS\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_line:
        sidereon.parse_tdm_kvn(raw_bad_line)
    detail = exc_line.value.detail
    assert detail.kind == "malformed_line"
    assert detail.text == "MALFORMED_LINE_NO_EQUALS"
    assert detail.line == 3

    # 3. invalid_version: value, line
    with pytest.raises(sidereon.TdmParseError) as exc_ver:
        sidereon.parse_tdm_kvn("CCSDS_TDM_VERS = 2.X\n")
    detail = exc_ver.value.detail
    assert detail.kind == "invalid_version"
    assert detail.value == "2.X"
    assert detail.line == 1

    # 4. section: detail, line
    raw_nested_meta = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_sec:
        sidereon.parse_tdm_kvn(raw_nested_meta)
    detail = exc_sec.value.detail
    assert detail.kind == "section"
    assert detail.detail == "nested metadata block"
    assert detail.line == 5

    # 5. conflicting_keyword: first, second, keyword, section, line
    raw_conflict = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "MODE = REAL_TIME\n"
        "MODE = NEAR_REAL_TIME\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_conf:
        sidereon.parse_tdm_kvn(raw_conflict)
    detail = exc_conf.value.detail
    assert detail.kind == "conflicting_keyword"
    assert detail.first == "REAL_TIME"
    assert detail.second == "NEAR_REAL_TIME"
    assert detail.keyword == "MODE"
    assert detail.section == "metadata"
    assert detail.line == 8

    # 6. invalid_field: input_error_kind, keyword
    bad_record = sidereon.TdmDataRecord(
        sidereon.TdmObservable.range(),
        "RANGE",
        "2026-160T10:00:00",
        sidereon.TdmScalar("100.0", 100.0),
        sidereon.TdmUnit("Hz"),
    )
    seg_mismatch = sidereon.TdmSegment(
        _valid_tdm().segments[0].metadata,
        sidereon.TdmDataSection([bad_record]),
    )
    tdm_mismatch = sidereon.Tdm(
        "2.0",
        [seg_mismatch],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )
    with pytest.raises(sidereon.TdmWriteError) as exc_unit:
        tdm_mismatch.to_kvn_string()
    detail = exc_unit.value.detail
    assert detail.kind == "invalid_field"
    assert detail.keyword == "RANGE"
    assert detail.input_error_kind == sidereon.TdmInputErrorKind.UNIT_MISMATCH

    # 7. non_printable_character: character, column, line, keyword
    raw_bad_char = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\x07\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_char:
        sidereon.parse_tdm_kvn(raw_bad_char)
    detail = exc_char.value.detail
    assert detail.kind == "non_printable_character"
    assert detail.character == "\x07"
    assert detail.column == 18
    assert detail.line == 3
    assert detail.keyword == "ORIGINATOR"

    # 8. line_too_long: length, line, keyword
    raw_long = (
        "CCSDS_TDM_VERS = 2.0\n"
        "COMMENT " + ("A" * 250) + "\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_long:
        sidereon.parse_tdm_kvn(raw_long)
    detail = exc_long.value.detail
    assert detail.kind == "line_too_long"
    assert detail.length == 258
    assert detail.line == 2
    assert detail.keyword == "COMMENT"

    # 9. undefined_participant: index, keyword, segment
    raw_bad_part = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "PATH = 1,3\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_part:
        sidereon.parse_tdm_kvn(raw_bad_part)
    detail = exc_part.value.detail
    assert detail.kind == "undefined_participant"
    assert detail.index == 3
    assert detail.keyword == "PATH"
    assert detail.segment == 1

    # 10. records_out_of_order & duplicate_record: epoch, segment, keyword
    raw_order = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "RANGE = 2026-160T20:14:00 2000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_ord:
        sidereon.parse_tdm_kvn(raw_order)
    detail = exc_ord.value.detail
    assert detail.kind == "records_out_of_order"
    assert detail.epoch == "2026-160T20:14:00"
    assert detail.segment == 1
    assert detail.keyword == "RANGE"

    # 11. Named strict refusals: missing_keyword, keyword_out_of_order
    raw_missing = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_miss:
        sidereon.parse_tdm_kvn(raw_missing)
    detail = exc_miss.value.detail
    assert detail.kind == "missing_keyword"
    assert detail.keyword == "TIME_SYSTEM"
    assert detail.segment == 1

    raw_order_meta = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "TRACK_ID = TRK_01\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    with pytest.raises(sidereon.TdmParseError) as exc_order_meta:
        sidereon.parse_tdm_kvn(raw_order_meta)
    detail = exc_order_meta.value.detail
    assert detail.kind == "keyword_out_of_order"
    assert detail.keyword == "TRACK_ID"
    assert detail.section == "metadata"
    assert detail.line == 6


# ---------------------------------------------------------------------------
# Section 4: Exception class attributes and instance context
# ---------------------------------------------------------------------------


def test_tdm_exception_class_attributes_and_instance_context():
    # Base exception class default None, inherited by subclasses
    assert hasattr(sidereon.TdmParseError, "detail")
    assert sidereon.TdmParseError.detail is None
    assert hasattr(sidereon.TdmWriteError, "detail")
    assert sidereon.TdmWriteError.detail is None
    assert hasattr(sidereon.TdmValidationError, "detail")
    assert sidereon.TdmValidationError.detail is None

    # Manually constructed exceptions lack core context (detail is None)
    manual_parse_err = sidereon.TdmParseError("manual parse error")
    assert manual_parse_err.detail is None
    manual_write_err = sidereon.TdmWriteError("manual write error")
    assert manual_write_err.detail is None
    manual_val_err = sidereon.TdmValidationError("manual validation error")
    assert manual_val_err.detail is None

    # Domain failures populate full typed payload
    with pytest.raises(sidereon.TdmParseError) as exc_info:
        sidereon.parse_tdm_kvn("CCSDS_TDM_VERS = invalid\n")
    err = exc_info.value
    assert err.detail is not None
    assert isinstance(err.detail, sidereon.TdmErrorDetail)
    assert err.detail.kind == "invalid_version"
    assert err.detail.value == "invalid"


# ---------------------------------------------------------------------------
# Section 5: Raw field construction, derived properties, & atomic mutations
# ---------------------------------------------------------------------------


def test_tdm_metadata_raw_construction_and_derived_properties():
    fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("START_TIME", "2026-160T10:00:00Z"),
        sidereon.TdmField("PARTICIPANT_1", "GROUND_STATION"),
        sidereon.TdmField("PARTICIPANT_2", "SATELLITE_A"),
        sidereon.TdmField("MODE", "REAL_TIME"),
        sidereon.TdmField("PATH", "1,2"),
        sidereon.TdmField("RANGE_UNITS", "RU"),
    ]
    comments = [sidereon.TdmComment("Derived metadata test", before_record=0)]

    meta = sidereon.TdmMetadata(fields, comments)
    assert meta.time_system == "UTC"
    assert meta.mode == "REAL_TIME"
    assert meta.range_units.label == "RU"
    assert len(meta.participants) == 2
    assert meta.participants[0].index == 1
    assert meta.participants[0].name == "GROUND_STATION"
    assert len(meta.paths) == 1
    assert meta.paths[0].participants == [1, 2]
    assert meta.get_last("START_TIME") == "2026-160T10:00:00Z"


def test_tdm_metadata_failed_mutation_is_fully_atomic():
    fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
    ]
    meta = sidereon.TdmMetadata(fields)
    assert meta.time_system == "UTC"
    assert len(meta.fields) == 2

    # Attempt an invalid raw replacement: missing required PARTICIPANT_n
    bad_fields = [sidereon.TdmField("TIME_SYSTEM", "GPS")]
    with pytest.raises(sidereon.TdmParseError):
        meta.replace_raw(bad_fields)

    # Object remains completely unchanged
    assert meta.time_system == "UTC"
    assert len(meta.fields) == 2
    assert meta.fields[0].value == "UTC"
    assert meta.fields[1].key == "PARTICIPANT_1"


def test_tdm_scalar_edit_updates_raw_fields_and_preserves_duplicates():
    fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
        sidereon.TdmField("MODE", "REAL_TIME"),
        sidereon.TdmField("MODE", "REAL_TIME"),
    ]
    write_policy = sidereon.TdmWritePolicy.strict().with_repeated_keywords(
        sidereon.TdmLeniency.FORGIVE
    )
    result = sidereon.TdmMetadata.from_raw_with_policy(fields, [], write_policy)
    meta = result.metadata

    departures = meta.set_mode_with_policy("NEAR_REAL_TIME", write_policy)
    assert len(departures) == 1
    assert departures[0].kind == "repeated_keyword"

    mode_fields = [f for f in meta.fields if f.key == "MODE"]
    assert len(mode_fields) == 2
    assert mode_fields[0].value == "NEAR_REAL_TIME"
    assert mode_fields[1].value == "NEAR_REAL_TIME"


# ---------------------------------------------------------------------------
# Section 6: Comment offset shifting & mandatory field atomic refusal
# ---------------------------------------------------------------------------


def test_tdm_comment_offset_preservation_on_insert_and_delete():
    fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("START_TIME", "2026-160T10:00:00Z"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
    ]
    comments = [
        sidereon.TdmComment("Leading comment", 0),
        sidereon.TdmComment("Interleaved comment", 2),
        sidereon.TdmComment("Trailing comment", 3),
    ]
    # CCSDS 503.0-B-2 4.5.2 puts every metadata comment before the first
    # metadata keyword, and table 3-3 ranks COMMENT first, so strict
    # construction refuses the comments at 2 and 3 by name. Comments kept at
    # later positions are what a read that forgives keyword order returns, and
    # editing around them takes the write policy that forgives the same.
    with pytest.raises(sidereon.TdmValidationError) as refused:
        sidereon.TdmMetadata(fields, comments)
    assert refused.value.detail.kind == "keyword_out_of_order"
    assert refused.value.detail.keyword == "COMMENT"
    assert refused.value.detail.section == "metadata"

    policy = sidereon.TdmWritePolicy.strict().with_keyword_order(
        sidereon.TdmLeniency.FORGIVE
    )

    def comment_departures(departures):
        """The count of `departures`, each a COMMENT written after a metadata
        keyword that outranks it."""
        assert all(
            (d.kind, d.keyword, d.section)
            == ("keyword_out_of_order", "COMMENT", "metadata")
            for d in departures
        )
        return len(departures)

    result = sidereon.TdmMetadata.from_raw_with_policy(fields, comments, policy)
    meta = result.metadata
    assert len(meta.comments) == 3
    # The comments at 2 and 3 are written after START_TIME and PARTICIPANT_1.
    assert comment_departures(result.departures) == 2

    # Insert TRACK_ID at index 0 (before TIME_SYSTEM, rank-valid under Table 3-3)
    track_field = sidereon.TdmField("TRACK_ID", "TRK_01")
    assert (
        comment_departures(meta.insert_field_with_policy(0, track_field, policy)) == 2
    )

    assert meta.fields[0].key == "TRACK_ID"
    assert meta.fields[1].key == "TIME_SYSTEM"
    # Equal offset semantics: comment at index 0 stays at 0 (precedes TRACK_ID)
    assert meta.comments[0].before_record == 0
    # Comments beyond insert index 0 shift by +1
    assert meta.comments[1].before_record == 3
    assert meta.comments[2].before_record == 4

    # Remove the inserted field at index 0
    removed, departures = meta.remove_field_at_with_policy(0, policy)
    assert removed.key == "TRACK_ID"
    assert comment_departures(departures) == 2

    # Offsets shift back precisely
    assert meta.comments[0].before_record == 0
    assert meta.comments[1].before_record == 2
    assert meta.comments[2].before_record == 3

    # Trailing append semantics: append MODE (rank-valid after PARTICIPANT)
    mode_field = sidereon.TdmField("MODE", "REAL_TIME")
    assert comment_departures(meta.append_field_with_policy(mode_field, policy)) == 2
    assert meta.fields[3].key == "MODE"
    # Existing trailing comment was at 3; it now precedes appended MODE
    assert meta.comments[2].before_record == 3
    # Add new trailing comment at index 4 (after MODE)
    assert (
        comment_departures(
            meta.add_comment_with_policy("New trailing comment", 4, policy)
        )
        == 3
    )
    assert meta.comments[3].before_record == 4
    assert meta.comments[3].text == "New trailing comment"

    # Remove appended MODE at index 3
    removed_mode, departures = meta.remove_field_at_with_policy(3, policy)
    assert removed_mode.key == "MODE"
    assert comment_departures(departures) == 3
    # New trailing comment shifts back to 3
    assert meta.comments[3].before_record == 3


def test_tdm_mandatory_field_removal_refuses_atomically():
    # Case 1: Attempt to remove required TIME_SYSTEM strictly
    fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
    ]
    comments = [sidereon.TdmComment("Mandatory test comment", 0)]
    meta = sidereon.TdmMetadata(fields, comments)

    with pytest.raises(sidereon.TdmValidationError) as exc_info:
        meta.remove_field("TIME_SYSTEM")

    assert exc_info.value.detail.kind == "missing_keyword"
    assert exc_info.value.detail.keyword == "TIME_SYSTEM"
    assert exc_info.value.detail.segment == 1
    # Full atomic check: fields, comments, and derived state unchanged
    assert meta.time_system == "UTC"
    assert len(meta.fields) == 2
    assert meta.fields[0].key == "TIME_SYSTEM"
    assert meta.fields[1].key == "PARTICIPANT_1"
    assert len(meta.comments) == 1
    assert meta.comments[0].text == "Mandatory test comment"
    assert meta.comments[0].before_record == 0

    # Cases 2 and 3 keep a comment after the first metadata keyword, which
    # 4.5.2 places before it, so the metadata is built and edited under a
    # policy that forgives keyword order. That leaves the refusal each case
    # names as the only one the removal meets, and shows the comment's offset
    # survives the refused edit.
    order_policy = sidereon.TdmWritePolicy.strict().with_keyword_order(
        sidereon.TdmLeniency.FORGIVE
    )

    # Case 2: Sole participant removal raises MissingKeyword for PARTICIPANT_n
    sole_fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
    ]
    sole_comments = [sidereon.TdmComment("Sole participant comment", 1)]
    meta_sole = sidereon.TdmMetadata.from_raw_with_policy(
        sole_fields, sole_comments, order_policy
    ).metadata

    with pytest.raises(sidereon.TdmValidationError) as exc_info_sole:
        meta_sole.remove_field_with_policy("PARTICIPANT_1", order_policy)

    assert exc_info_sole.value.detail.kind == "missing_keyword"
    assert exc_info_sole.value.detail.keyword == "PARTICIPANT_n"
    assert exc_info_sole.value.detail.segment == 1
    # Full atomic check: fields, comments, derived state preserved
    assert len(meta_sole.fields) == 2
    assert meta_sole.fields[1].key == "PARTICIPANT_1"
    assert len(meta_sole.participants) == 1
    assert meta_sole.participants[0].name == "DSS-25"
    assert len(meta_sole.comments) == 1
    assert meta_sole.comments[0].before_record == 1

    # Case 3: Dangling PATH with surviving participant raises UndefinedParticipant
    path_fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
        sidereon.TdmField("PARTICIPANT_2", "SATELLITE_A"),
        sidereon.TdmField("PATH", "1,2"),
    ]
    path_comments = [sidereon.TdmComment("Path test comment", 2)]
    meta_path = sidereon.TdmMetadata.from_raw_with_policy(
        path_fields, path_comments, order_policy
    ).metadata

    with pytest.raises(sidereon.TdmValidationError) as exc_info_path:
        meta_path.remove_field_with_policy("PARTICIPANT_1", order_policy)

    assert exc_info_path.value.detail.kind == "undefined_participant"
    assert exc_info_path.value.detail.keyword == "PATH"
    assert exc_info_path.value.detail.index == 1
    assert exc_info_path.value.detail.segment == 1
    # Full atomic check: fields, comments, derived state preserved
    assert len(meta_path.fields) == 4
    assert meta_path.fields[1].key == "PARTICIPANT_1"
    assert len(meta_path.participants) == 2
    assert len(meta_path.paths) == 1
    assert meta_path.paths[0].participants == [1, 2]
    assert len(meta_path.comments) == 1
    assert meta_path.comments[0].before_record == 2


# ---------------------------------------------------------------------------
# Section 7: Lenient construction, strict serialization refusal
# ---------------------------------------------------------------------------


def test_tdm_lenient_source_refused_by_strict_writer():
    fields = [sidereon.TdmField("PARTICIPANT_1", "DSS-25")]
    write_policy = sidereon.TdmWritePolicy.strict().with_missing_keywords(
        sidereon.TdmLeniency.FORGIVE
    )
    meta_res = sidereon.TdmMetadata.from_raw_with_policy(fields, [], write_policy)
    meta = meta_res.metadata
    assert len(meta_res.departures) == 1
    assert meta_res.departures[0].kind == "missing_keyword"
    assert meta_res.departures[0].keyword == "TIME_SYSTEM"
    assert meta_res.departures[0].segment == 1

    data = sidereon.TdmDataSection(
        [
            sidereon.TdmDataRecord(
                sidereon.TdmObservable.range(),
                "RANGE",
                "2026-160T10:00:00",
                sidereon.TdmScalar("100.0", 100.0),
                sidereon.TdmUnit("km"),
            )
        ]
    )
    seg = sidereon.TdmSegment(meta, data)
    tdm = sidereon.Tdm(
        "2.0",
        [seg],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )

    with pytest.raises(sidereon.TdmWriteError) as exc_info:
        tdm.to_kvn_string()

    assert exc_info.value.detail.kind == "missing_keyword"
    assert exc_info.value.detail.keyword == "TIME_SYSTEM"
    assert exc_info.value.detail.segment == 1

    write_res = tdm.to_kvn_string_with_policy(write_policy)
    assert "TIME_SYSTEM" not in write_res.text
    assert len(write_res.departures) == 1
    dep = write_res.departures[0]
    assert dep.kind == "missing_keyword"
    assert dep.keyword == "TIME_SYSTEM"
    assert dep.segment == 1


# ---------------------------------------------------------------------------
# Section 8: Leap seconds (UTC vs. GLONASS) & 38-digit fractions
# ---------------------------------------------------------------------------


def test_tdm_utc_and_glonass_leap_seconds_and_fraction38():
    utc_kvn = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T23:59:60 1000.0\n"
        "DATA_STOP\n"
    )
    utc_tdm = sidereon.parse_tdm_kvn(utc_kvn)
    assert utc_tdm.segments[0].data.records[0].epoch == "2026-160T23:59:60"
    assert utc_tdm.to_kvn_string() == utc_kvn

    glonass_kvn = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = GLONASS\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T02:59:60 1000.0\n"
        "DATA_STOP\n"
    )
    glonass_tdm = sidereon.parse_tdm_kvn(glonass_kvn)
    assert glonass_tdm.segments[0].data.records[0].epoch == "2026-160T02:59:60"
    assert glonass_tdm.to_kvn_string() == glonass_kvn

    glonass_bad_kvn = glonass_kvn.replace("02:59:60", "23:59:60")
    with pytest.raises(sidereon.TdmParseError) as exc:
        sidereon.parse_tdm_kvn(glonass_bad_kvn)
    assert exc.value.detail.kind == "malformed_epoch"

    gps_bad_kvn = utc_kvn.replace("TIME_SYSTEM = UTC", "TIME_SYSTEM = GPS")
    with pytest.raises(sidereon.TdmParseError) as exc2:
        sidereon.parse_tdm_kvn(gps_bad_kvn)
    assert exc2.value.detail.kind == "malformed_epoch"

    epoch_38 = "2026-160T12:00:00.12345678901234567890123456789012345678"
    frac38_kvn = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        f"RANGE = {epoch_38} 1000.0\n"
        "DATA_STOP\n"
    )
    frac_tdm = sidereon.parse_tdm_kvn(frac38_kvn)
    assert frac_tdm.segments[0].data.records[0].epoch == epoch_38
    assert frac_tdm.to_kvn_string() == frac38_kvn

    epoch_39 = epoch_38 + "9"
    frac39_kvn = frac38_kvn.replace(epoch_38, epoch_39)
    with pytest.raises(sidereon.TdmParseError) as exc3:
        sidereon.parse_tdm_kvn(frac39_kvn)
    assert exc3.value.detail.kind == "malformed_epoch"


# ---------------------------------------------------------------------------
# Section 9: Unit mismatch writer refusal, getter copy semantics & annex
# ---------------------------------------------------------------------------


def test_tdm_unit_mismatch_writer_refusal_details():
    fields = [
        sidereon.TdmField("TIME_SYSTEM", "UTC"),
        sidereon.TdmField("PARTICIPANT_1", "DSS-25"),
    ]
    meta = sidereon.TdmMetadata(fields)

    bad_record = sidereon.TdmDataRecord(
        sidereon.TdmObservable.range(),
        "RANGE",
        "2026-160T10:00:00",
        sidereon.TdmScalar("100.0", 100.0),
        sidereon.TdmUnit("Hz"),
    )
    data = sidereon.TdmDataSection([bad_record])
    seg = sidereon.TdmSegment(meta, data)
    tdm = sidereon.Tdm(
        "2.0",
        [seg],
        creation_date="2026-160T20:15:00Z",
        originator="NASA",
    )

    with pytest.raises(sidereon.TdmWriteError) as exc_info:
        tdm.to_kvn_string()

    detail = exc_info.value.detail
    assert detail.kind == "invalid_field"
    assert detail.keyword == "RANGE"
    assert detail.input_error_kind == sidereon.TdmInputErrorKind.UNIT_MISMATCH


def test_tdm_getter_copies_and_deliberate_reassignment():
    kvn = (
        "CCSDS_TDM_VERS = 2.0\n"
        "CREATION_DATE = 2026-160T20:15:00Z\n"
        "ORIGINATOR = NASA\n"
        "META_START\n"
        "TIME_SYSTEM = UTC\n"
        "PARTICIPANT_1 = DSS-25\n"
        "META_STOP\n"
        "DATA_START\n"
        "RANGE = 2026-160T20:15:00 1000.0\n"
        "DATA_STOP\n"
    )
    tdm = sidereon.parse_tdm_kvn(kvn)
    seg = tdm.segments[0]
    meta_copy = seg.metadata

    meta_copy.time_system = "TAI"
    assert meta_copy.time_system == "TAI"

    # Snapshot semantics: segment metadata is not mutated in place
    assert seg.metadata.time_system == "UTC"

    # Deliberate reassignment updates the segment
    seg.metadata = meta_copy
    assert seg.metadata.time_system == "TAI"

    # Reassign segment back to message
    tdm.set_segment(0, seg)
    assert "TIME_SYSTEM = TAI" in tdm.to_kvn_string()


def test_tdm_annex_e_01_integration_passes():
    text = (FIXTURES / "tdm" / "annex_e_01.kvn").read_text()
    message = sidereon.parse_tdm_kvn(text)

    assert len(message.segments) == 1
    records = message.segments[0].data.records
    assert len(records) == 31
    assert records[0].keyword == "TRANSMIT_FREQ_2"
    assert records[0].value_text == "32023442781.733"
    assert records[0].unit.label == "Hz"

    encoded = message.to_kvn_string()
    reparsed = sidereon.parse_tdm_kvn(encoded)
    assert reparsed.to_kvn_string() == encoded


def test_tdm_input_error_kind_unknown_variant_contract():
    unknown = sidereon.TdmInputErrorKind.UNKNOWN
    missing = sidereon.TdmInputErrorKind.MISSING
    assert unknown != missing
    assert unknown.kind == "unknown"
    assert getattr(unknown, "name", "UNKNOWN") == "UNKNOWN"
    assert repr(unknown) == "TdmInputErrorKind.UNKNOWN"


def test_tdm_error_detail_values_are_preserved_on_public_exceptions():
    header = _valid_header_lines()
    metadata = "META_START\nTIME_SYSTEM = UTC\nPARTICIPANT_1 = DSS-25\nMETA_STOP\n"
    record = "RANGE = 2026-160T20:15:00 1000.0\n"
    valid = header + metadata + "DATA_START\n" + record + "DATA_STOP\n"
    strict = sidereon.TdmPolicy.strict()
    cases = [
        (
            "no_segments",
            header,
            {"kind": "no_segments", "message": "missing TDM segment"},
        ),
        (
            "malformed_epoch",
            header + metadata + "DATA_START\nRANGE = bad-epoch 1000.0\nDATA_STOP\n",
            {
                "kind": "malformed_epoch",
                "line": 9,
                "keyword": "RANGE",
                "text": "bad-epoch",
                "message": (
                    "TDM record RANGE at line 9 has the timetag bad-epoch, "
                    "which is not a form 4.3.9 defines"
                ),
            },
        ),
        (
            "duplicate_record",
            header + metadata + "DATA_START\n" + record + record + "DATA_STOP\n",
            {
                "kind": "duplicate_record",
                "keyword": "RANGE",
                "segment": 1,
                "epoch": "2026-160T20:15:00",
                "message": "TDM segment 1 repeats RANGE at 2026-160T20:15:00",
            },
        ),
        (
            "unterminated_final_line",
            valid.rstrip("\n"),
            {
                "kind": "unterminated_final_line",
                "line": 10,
                "message": "TDM line 10 carries no terminator",
            },
        ),
        (
            "conflicting_keyword",
            header
            + (
                "META_START\nTIME_SYSTEM = UTC\nPARTICIPANT_1 = DSS-25\n"
                "MODE = A\nMODE = B\nMETA_STOP\nDATA_START\n"
            )
            + record
            + "DATA_STOP\n",
            {
                "kind": "conflicting_keyword",
                "line": 8,
                "keyword": "MODE",
                "section": "metadata",
                "first": "A",
                "second": "B",
                "message": (
                    'TDM metadata keyword MODE at line 8 repeats with "B" after "A"'
                ),
            },
        ),
        (
            "undefined_keyword",
            header + "FOO = BAR\n" + metadata + "DATA_START\n" + record + "DATA_STOP\n",
            {
                "kind": "undefined_keyword",
                "line": 4,
                "keyword": "FOO",
                "section": "header",
                "message": (
                    "TDM header keyword FOO at line 4 is not one the standard defines"
                ),
            },
        ),
        (
            "empty_data_section",
            header + metadata + "DATA_START\nDATA_STOP\n",
            {
                "kind": "empty_data_section",
                "segment": 1,
                "message": "TDM segment 1 holds no tracking data record",
            },
        ),
        (
            "empty_value",
            header.replace("ORIGINATOR = NASA", "ORIGINATOR =")
            + metadata
            + "DATA_START\n"
            + record
            + "DATA_STOP\n",
            {
                "kind": "empty_value",
                "line": 3,
                "keyword": "ORIGINATOR",
                "message": "TDM keyword ORIGINATOR has no value at line 3",
            },
        ),
        (
            "undefined_keyword_in_metadata",
            header
            + (
                "META_START\nTIME_SYSTEM = UTC\nPARTICIPANT_1 = DSS-25\n"
                "RANGE = 2026-160T20:15:00 1000.0\nMETA_STOP\nDATA_START\n"
            )
            + record
            + "DATA_STOP\n",
            {
                "kind": "undefined_keyword",
                "line": 7,
                "keyword": "RANGE",
                "section": "metadata",
                "message": (
                    "TDM metadata keyword RANGE at line 7 is not one the "
                    "standard defines"
                ),
            },
        ),
        (
            "malformed_record",
            header + metadata + "DATA_START\nRANGE = 2026-160T20:15:00\nDATA_STOP\n",
            {
                "kind": "malformed_record",
                "line": 9,
                "keyword": "RANGE",
                "message": "malformed TDM data record RANGE at line 9",
            },
        ),
    ]
    attributes = (
        "kind",
        "line",
        "keyword",
        "column",
        "character",
        "length",
        "text",
        "detail",
        "reason",
        "section",
        "segment",
        "epoch",
        "index",
        "first",
        "second",
        "value",
        "input_error_kind_name",
        "message",
    )

    for name, raw, expected in cases:
        try:
            sidereon.parse_tdm_kvn_with_policy(raw, strict)
        except sidereon.TdmParseError as error:
            detail = error.detail
            try:
                sidereon.parse_tdm_kvn_with_policy(raw, strict)
            except sidereon.TdmParseError as repeated_error:
                equal_detail = repeated_error.detail
            else:
                pytest.fail(f"{name} did not reproduce its error")
        else:
            pytest.fail(f"{name} did not raise TdmParseError")
        assert isinstance(detail, sidereon.TdmErrorDetail), name
        for attribute in attributes:
            assert getattr(detail, attribute) == expected.get(attribute), (
                name,
                attribute,
            )
        assert str(detail) == expected["message"]
        assert repr(detail).startswith("TdmErrorDetail")
        assert detail == equal_detail

    repeated = (
        header
        + "META_START\nTIME_SYSTEM = UTC\nPARTICIPANT_1 = DSS-25\n"
        + "MODE = A\nMODE = A\nMETA_STOP\nDATA_START\n"
        + record
        + "DATA_STOP\n"
    )
    tdm = sidereon.parse_tdm_kvn_with_policy(repeated, strict).message
    with pytest.raises(sidereon.TdmWriteError) as write_error:
        tdm.to_kvn_string()
    detail = write_error.value.detail
    assert isinstance(detail, sidereon.TdmErrorDetail)
    assert detail.kind == "repeated_keyword"
    assert detail.line is None
    assert detail.keyword == "MODE"
    assert detail.section == "metadata"
    assert detail.message == "TDM metadata writes MODE twice with the same value"
    assert str(write_error.value) == detail.message

    unassignable = sidereon.parse_tdm_kvn(valid)
    unassignable.header_fields = [sidereon.TdmField("COMMENT", "")]
    with pytest.raises(sidereon.TdmWriteError) as unassignable_error:
        unassignable.to_kvn_string()
    detail = unassignable_error.value.detail
    assert detail.kind == "keyword_not_assignable"
    assert detail.keyword == "COMMENT"
    assert detail.message == "TDM keyword COMMENT cannot be given a value"
    assert str(unassignable_error.value) == detail.message
