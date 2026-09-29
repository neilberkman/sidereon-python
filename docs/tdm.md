# CCSDS Tracking Data Message (TDM) Python API

This document provides a comprehensive reference and usage guide for the CCSDS Tracking Data Message (TDM) 3.0 Python bindings in `sidereon`. The binding adheres strictly to CCSDS 503.0-B-2 (KVN format) while offering policy-driven leniency, structured diagnostic reporting, positioned comments, and atomic metadata mutation semantics.

---

## Architecture & Public Contract

In Sidereon 3.0, the TDM binding enforces a clear separation of authority:
1. **Raw Authority**: The ordered list of raw `TdmField` entries (`fields`) and positioned `TdmComment` entries (`comments`) within `TdmMetadata` are the sole authority.
2. **Derived Properties**: All convenience properties on `TdmMetadata`—including `participants`, `paths`, `mode`, `timetag_ref`, `time_system`, and `range_units`—are synchronously derived by the Rust core upon construction or mutation. No duplicate parsers or synthetic fallback values reside in the Python layer.
3. **Strict Validation by Default**: Default construction and serialization enforce existing core validation checks against CCSDS 503.0-B-2 (not proof that all conceivable CCSDS semantics are validated). Invalid inputs, missing mandatory fields (e.g. `CREATION_DATE` or `ORIGINATOR` in headers, `TIME_SYSTEM` or `PARTICIPANT_n` in metadata), or contradictory entries are refused with typed exceptions. Header and data section direct setters modify snapshots directly and are validated at write time (`to_kvn_string`), while the atomic edits contract applies to `TdmMetadata` helpers which validate candidates synchronously through the core builder.
4. **Policy-Driven Leniency**: Explicit policy objects (`TdmPolicy` for reading, `TdmWritePolicy` for writing) enable controlled forgiveness of structural or syntactical departures. Every forgiven departure is returned as a typed warning or departure object.
5. **Atomic Mutations**: Failed mutations (via property setters, field manipulation, or bulk replacement) leave the metadata object completely unmodified. Raw field order, duplicates, and comments remain the sole authority, and full typed diagnostics are returned from core validation without independent binding parser duplicate logic.

---

## Data Model & Types

### 1. Positioned Comments (`TdmComment`)
Comments in CCSDS 503.0-B-2 are positioned within their enclosing section. Rather than flattening comments into raw string arrays, `sidereon` models each comment with:
- `text`: The string content of the comment (excluding the `COMMENT` keyword and required delimiter space).
- `before_record`: Zero-based index indicating which field or data record the comment precedes.

#### Comment Positioning Rules
- **Header Comments**: Conforming header comments sit between `CCSDS_TDM_VERS` and the first header metadata field, having `before_record = 1`.
- **Metadata Leading Comments**: Positioned at the top of the block before any metadata field (`before_record = 0`).
- **Metadata Interleaved Comments**: Positioned immediately before field $k$ (`before_record = k`).
- **Metadata Trailing Comments**: Positioned after all $N$ metadata fields (`before_record = N`).
- **Data Block Comments**: Positioned before the record at index $k$ (`before_record = k`), or leading (`before_record = 0`).

CCSDS 503.0-B-2 4.5.2 places metadata comments before the first metadata field and data comments before the first data record, so `before_record = 0` is the only conforming position in those sections. A comment at any later position is a keyword-order departure: strict reading, strict `TdmMetadata` construction and strict writing refuse it as `keyword_out_of_order`, and a policy that forgives `keyword_order` keeps the comment where it sits and reports it as a warning or departure.

### 2. Policies and Leniency

#### `TdmLeniency`
An enumeration indicating how a departure axis is treated:
- `TdmLeniency.STRICT` (`TdmLeniency.Strict`): Refuse non-conforming messages or values, raising an error.
- `TdmLeniency.FORGIVE` (`TdmLeniency.Forgive`): Forgive the departure, recording a typed diagnostic.

#### `TdmPolicy` (Reader, 8 Axes)
Controls parser leniency:
1. `non_printable`: Characters outside printable ASCII (CCSDS 503.0-B-2 §4.2.1).
2. `missing_keywords`: Mandatory keywords omitted from header or metadata (§3.2, §3.3).
3. `long_lines`: Lines exceeding 254 characters (§4.2.1).
4. `empty_data_sections`: Data blocks containing no tracking data records (§3.1.3).
5. `record_order`: Observable records not in chronological order (§3.4.10).
6. `duplicate_records`: Identical timetag and keyword repeats within a segment (§3.4.11).
7. `keyword_order`: Keywords out of the strict sequence defined in Table 3-2 or Table 3-3.
8. `final_terminator`: Missing line terminators on the final message line (§4.2.11).

Builders and constructors:
```python
policy = TdmPolicy.strict()
lenient = TdmPolicy.lenient()
custom = (
    TdmPolicy.strict()
    .with_missing_keywords(TdmLeniency.FORGIVE)
    .with_long_lines(TdmLeniency.FORGIVE)
)
```

#### `TdmWritePolicy` (Writer, 9 Axes)
Controls serializer leniency when emitting KVN text:
- Mirrors all 8 reader axes above.
- Adds `repeated_keywords`: Controls whether duplicate identical keyword assignments within a section are permitted in output.
- `as_read()`: Converts the write policy into its corresponding 8-axis `TdmPolicy`.

```python
write_policy = TdmWritePolicy.strict()
lenient_writer = TdmWritePolicy.lenient()
custom_writer = write_policy.with_repeated_keywords(TdmLeniency.FORGIVE)
```

---

## Parsing and Writing

### Basic Strict Parsing and Serialization
```python
import sidereon

# Strict parsing
tdm = sidereon.parse_tdm_kvn(kvn_text)

# Strict serialization
output_text = tdm.to_kvn_string()  # or tdm.to_kvn()
```

### Policy-Aware Parsing (`parse_tdm_kvn_with_policy`)
Returns a `TdmParseResult` containing the parsed `message` and ordered `warnings`:
```python
result = sidereon.parse_tdm_kvn_with_policy(lenient_text, policy)
tdm = result.message
for warning in result.warnings:
    print(f"[{warning.kind}] Line {warning.line}: {warning.message}")

# Supports tuple unpacking
tdm, warnings = sidereon.parse_tdm_kvn_with_policy(lenient_text, policy)
```

### Policy-Aware Serialization (`to_kvn_string_with_policy`)
Returns a `TdmWriteResult` containing the serialized `text` and ordered `departures`:
```python
write_res = tdm.to_kvn_string_with_policy(write_policy)
text = write_res.text
for departure in write_res.departures:
    print(f"[{departure.kind}] Emitted departure: {departure.message}")

# Supports tuple unpacking
text, departures = tdm.to_kvn_string_with_policy(write_policy)
```

---

## Metadata Construction and Mutation

### 1. Construction
Constructing `TdmMetadata` requires raw ordered fields and optional positioned comments. The core validates the candidate and synchronously derives all properties:
```python
fields = [
    sidereon.TdmField("TIME_SYSTEM", "UTC"),
    sidereon.TdmField("START_TIME", "2026-09-22T00:00:00Z"),
    sidereon.TdmField("PARTICIPANT_1", "GROUND_ANTENNA"),
    sidereon.TdmField("PARTICIPANT_2", "SATELLITE_A"),
    sidereon.TdmField("PATH", "1,2"),
]
comments = [sidereon.TdmComment("Track session", before_record=0)]

# Strict construction
metadata = sidereon.TdmMetadata(fields, comments)
assert metadata.time_system == "UTC"
assert len(metadata.participants) == 2

# Policy-aware construction returning TdmMetadataResult
result = sidereon.TdmMetadata.from_raw_with_policy(fields, comments, write_policy)
metadata = result.metadata
departures = result.departures
```

### 2. Scalar Property Setters
Ordinary scalar property setters (`time_system`, `mode`, `timetag_ref`, `range_units`) update raw matching occurrences in place:
- **Duplicate Preservation**: If multiple identical keys exist (forgiven under policy), all matching occurrences are updated consistently; they are not collapsed.
- **Order Preservation**: Existing key order is maintained; values are updated in place without disturbing comments.
- **Deterministic Insertion**: If the key is not present, it is appended to the raw field list. If the resulting placement violates Table 3-3 keyword order, the core refuses the update under strict policy.
- **Removal**: Setting a supported scalar property (such as `mode` or `timetag_ref`) to `None` deletes all matching occurrences, shifting comment offsets accordingly. Note that `range_units` does not accept `None` (it requires a valid `TdmUnit` or unit string). If a required field (such as `TIME_SYSTEM`) is deleted, the core refuses the mutation under strict policy.

```python
# Update time system strictly
metadata.time_system = "GPS"

# Explicit policy update returning departures
departures = metadata.set_time_system_with_policy("GPS", write_policy)
```

### 3. Field Manipulation & Comment Offset Shifting
When fields are inserted or removed, comment offsets (`before_record`) are deterministically shifted to preserve their association with surviving fields:

#### Removal Semantics
When removing a field at index $d$:
- Comments with `before_record < d`: Unchanged (preceding fields are unaffected).
- Comments with `before_record == d`: Unchanged (they now precede the field that shifted into index $d$).
- Comments with `before_record > d`: Decremented by 1 (`before_record -= 1`).
- Trailing comments at the end of the block remain trailing; no comments are ever clamped or dropped.

#### Insertion Semantics
When inserting a field at index $i$:
- Comments with `before_record <= i`: Unchanged (comments before or at index $i$ precede the newly inserted field).
- Comments with `before_record > i`: Incremented by 1 (`before_record += 1`).

```python
# Insert TRACK_ID at index 0 (before TIME_SYSTEM to maintain Table 3-3 order)
metadata.insert_field(0, sidereon.TdmField("TRACK_ID", "TRACK_001"))

# Remove the inserted field at index 0
removed_field = metadata.remove_field_at(0)
assert removed_field.key == "TRACK_ID"

# Or remove by keyword
# metadata.remove_field("TRACK_ID")

# Note: Attempting to remove required fields (like TIME_SYSTEM) is refused under strict policy:
# try:
#     metadata.remove_field("TIME_SYSTEM")
# except sidereon.TdmValidationError as err:
#     print(f"Refused mandatory field removal: {err.detail.keyword}")
```

### 4. Bulk Replacement (`replace_raw`)
For batch updates, `replace_raw` and `replace_raw_with_policy` provide complete atomic replacement of fields and comments:
```python
metadata.replace_raw(new_fields, new_comments)

res = metadata.replace_raw_with_policy(new_fields, new_comments, write_policy)
departures = res.departures
```

---

## Getter Copy Semantics & Reassignment

In Sidereon's Python bindings, nested properties such as `segment.metadata`, `segment.data`, and `tdm.segments` return **value snapshots** (getter copies) rather than live mutable references.

To ensure modifications take effect in the message hierarchy:
```python
# Read segment copy from the message
seg = tdm.segments[0]
meta = seg.metadata

# Mutate metadata copy atomically
meta.time_system = "UTC"

# Reassign metadata to the segment copy, then update the message
seg.metadata = meta
tdm.set_segment(0, seg)
```

---

## High-Precision Epochs & Leap Seconds

### Raw String Preservation
TDM observable timetags are retained as lossless raw strings. They are never narrowed to standard floating-point seconds or 64-bit datetime structures:
- Supports decimal fractions up to 38 digits (`MAX_FRACTIONAL_DIGITS = 38`).
- Exact decimal source tokens are preserved across parse and write round-trips.

### UTC vs. GLONASS Leap Second Syntax
Second `60` is valid only under explicit leap-second-carrying time systems and only at specific canonical hours:
- **`UTC`**: Legal reading is `23:59:60`.
- **`GLONASS`**: Legal reading is `02:59:60` (reflecting the constant +3h Moscow offset from UTC(SU)).
- **Continuous Scales** (`GPS`, `TAI`, `TT`, etc.): Second `60` is illegal and refused with `TdmParseError` / `TdmErrorDetail(kind="malformed_epoch")`.

---

## Exception Handling and Structured Error Details

All TDM errors raise exceptions inheriting from `TdmParseError`:
- `TdmParseError`: Base class for KVN parsing or validation failures.
- `TdmWriteError`: Subclass raised when serialization fails.
- `TdmValidationError`: Subclass raised when metadata or structure validation fails.

Every exception carries a `detail` attribute of type `Optional[TdmErrorDetail]`. Manually constructed exception instances can lack core context (`detail` is `None`), while all domain failures populate a full typed payload:
```python
try:
    sidereon.parse_tdm_kvn(invalid_kvn)
except sidereon.TdmParseError as err:
    detail = err.detail
    print(f"Error kind: {detail.kind}")
    print(f"Keyword: {detail.keyword}")
    print(f"Line: {detail.line}, Column: {detail.column}")
    if detail.kind == "conflicting_keyword":
        print(f"Conflict: first={detail.first}, second={detail.second}")
```
