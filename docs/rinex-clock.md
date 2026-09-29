# RINEX clock (`.CLK`) Python API

Reference for the RINEX clock bindings in `sidereon`: what a product keeps, how
it is read and written, how epochs are represented, which queries answer in
which time scale, how a product is edited, and what each refusal carries.

The binding is an interface over the `sidereon-core` reader and writer. Every
number it returns is the number the core holds; it runs no clock parsing,
arithmetic or validation of its own apart from reading Python arguments.

---

## Entry points

| Function | Behaviour |
| --- | --- |
| `parse_rinex_clock(text)` | Strict parse. The first line that does not read refuses the file with `RinexClockParseError`. |
| `load_rinex_clock(source)` | Strict parse of bytes, a bytearray, or a path. |
| `parse_rinex_clock_lossy(text)` | Keeps a line that does not read verbatim, with a typed diagnostic. |
| `load_rinex_clock_lossy(source)` | Lossy parse of bytes, a bytearray, or a path. |
| `RinexClock.from_series_rows(rows)` | A GPST product from `(satellite, [(gps_seconds, bias_s), ...])` rows. |
| `RinexClock.from_instant_series_rows(time_scale, rows)` | A product from `(satellite, [(ClockInstant, bias_s), ...])` rows. |
| `RinexClock.from_clock_points(time_scale, rows)` | A product from `(satellite, [ClockPoint, ...])` rows, every declared value kept. |

A file read from bytes or a path reaches the parser with its line terminators
unchanged. Text read in Python's text mode has its CRLF terminators converted
before the parser sees them; read the file as bytes to keep them.

---

## The product keeps its text

A product read from text keeps that text as its authority: every header line
with its exact label and payload, and every body line in order, including
blank lines, records of every type (`AR`, `AS`, `CR`, `DR`, `MS`), continuation
lines and, after a lossy read, the lines that did not read. `to_rinex_string`
on an unedited product restates its input byte for byte, line terminators
included. Every other view is derived from the retained lines:

| View | Contents |
| --- | --- |
| `version`, `layout`, `satellite_system` | The `RINEX VERSION / TYPE` fields; `layout` is `ClockLayout.V300` (80 columns, before 3.04) or `ClockLayout.V304` (85 columns). |
| `time_system`, `time_system_status`, `time_scale` | The `TIME SYSTEM ID` reading; see below. |
| `header_records` | Every header line as a `ClockHeaderRecord`. |
| `records`, `record_count` | Every data record as a `ClockRecord`, in file order. |
| `series`, `series_for`, `satellites`, `satellite_count`, `sample_count` | The per-satellite `AS` sample series. |
| `skipped_records`, `skipped_record_count` | Records outside the satellite series, one per logical record at its parent line. |
| `diagnostics`, `diagnostic_count` | Lines a lossy read kept without reading them, and header time-system errors, with their typed errors. |
| `notices` | Findings that do not stop the product being read. |
| `source_line(line)` | One source line by one-based number, without its terminator. |

Equality compares the retained text and records, so a product rebuilt from its
own series rows does not equal the parsed product; compare `series` or
`series_rows()` for that.

### Header records

`ClockHeaderRecord` holds `line` (one-based, None for a line an edit wrote),
the exact `text`, the `label`, its `label_column` (60 or 65), the `payload`
before the label, `reading` and `field`.

`reading` is `Columns` (the file's version columns), `OtherVersionColumns`,
`Whitespace`, `Uninterpreted` (a known label whose fields do not read) or
`UnknownLabel`. `field` is a `ClockHeaderField` when the fields read: `kind` is
the Table A15 record (`VersionType`, `ProgramRunByDate`, `Comment`,
`ObservationTypes`, `TimeSystem`, `LeapSeconds`, `LeapSecondsGnss`,
`DcbsApplied`, `PcvsApplied`, `TypesOfData`, `StationNameNum`,
`StationClockRef`, `AnalysisCenter`, `ClockRefCount`, `AnalysisClockRef`,
`SolutionStationCount`, `SolutionStation`, `SolutionSatelliteCount`,
`PrnList`, `EndOfHeader`) and `details()` its fields. Blank optional fields are
None: a `ClockRefCount` start or stop epoch, an `AnalysisClockRef` constraint,
an `ObservationTypes` system and count on a continuation line.

### Data records

`ClockRecord` holds the `record_type` (`ClockRecordType.AR`, `AS`, `CR`, `DR`
or `MS`), the `name`, the canonical `satellite` of an `AS` record, the
`civil_epoch` (`ClockEpoch`), the `epoch` (`ClockInstant`, None when the time
system resolves to no scale), the declared `values` (bias first,
`declared_count` of them), `bias_s`, `additional_values`, `surplus_values`,
`line`, `line_count`, and how the first line and the continuation line were
read (`reading`, `continuation_reading`: `Columns` with the `layout` read,
`Whitespace`, or `Edited`).

A value present beyond the declared count - the bias sigma every `AS` record of
some analysis-centre products carries while declaring one value - is kept as a
`ClockSurplusValue` with its `position` in the value sequence (0 bias, 1 bias
sigma, 2 rate, 3 rate sigma, 4 acceleration, 5 acceleration sigma), rather than
refusing the record. Records repeated for one name and epoch all remain,
including the two `AR` records RINEX clock section 4 uses for a discontinuity;
the satellite series keeps the last in file order.

`ClockRecord(record_type, name, epoch, values)` builds a record to insert. An
`AS` name must be a satellite identifier and is stored in its canonical
spelling; other names must be a single ASCII token of at most nine characters.

### Notices

`ClockNotice.kind` is `TimeSystemDefaulted` and `TimeSystemWithoutScale` (with
`system`), `TimeSystemMissing`, `HeaderRecordNonconforming`,
`HeaderRecordUninterpreted` and `HeaderRecordUnknownLabel` (with `line`), and
`SurplusValues`, `OtherLayoutRecords` and `WhitespaceRecords` (with `records`
and `first_line`).

---

## Time systems

`ClockTimeSystem` has the eight `TIME SYSTEM ID` labels: `GPS`, `GLO`, `GAL`,
`QZS`, `BDS` (the spelling `BDT` is also read), `IRN`, `UTC`, `TAI`. `GLO` is
UTC in every version, as RINEX clock 3.00, RINEX 3.05 section 4.1.2 and RTKLIB
read it, so a `23:59:60` label on a leap-second day is an epoch in a `GLO`
product. `IRN` has no core time scale: its records keep their civil epochs
with no instant, and `time_scale` is None.

`time_system_status.kind` says how the system was established: `Declared`,
`Defaulted` (no record; the 3.00 default applies: `GLO` for a pure GLONASS
file, `GAL` for a pure Galileo file, otherwise `GPS`, with a notice, and a
further notice for a 3.04 file, which requires the record), `Unrecognized`
(the label is in `label`; a lossy read keeps the epochs civil), `Conflicting`
(the labels in `labels`) or `Constructed` (built from rows).

`time_scale` is None when the time system is missing, unrecognized,
conflicting or has no core scale.

---

## Samples

`ClockSeries` keeps every sample of one satellite, in time order, whatever its
time scale. These views are row-aligned with each other and with
`len(series)`:

| View | Contents |
| --- | --- |
| `points` | `ClockPoint`, the complete typed sample |
| `epochs` | `ClockInstant`, the scale-tagged epoch |
| `bias_s` | numpy `(n,)` float64, seconds |
| `additional_values` | per-row list of the declared values following the bias |
| `gps_seconds` | numpy `(n,)` float64, seconds since the GPS epoch, NaN where unavailable |
| `gps_seconds_valid` | numpy `(n,)` bool, False where `gps_seconds` is NaN |

GPST and QZSST samples project to GPS seconds (QZSST shares the GPST alignment
to TAI); NaN means the sample has no GPS-seconds projection, not that it is
missing. `ClockPoint` names its declared values `bias_sigma_s`, `rate`,
`rate_sigma`, `acceleration_per_s` and `acceleration_sigma_per_s`, each None
when not declared; a declared zero, including a signed zero, is data.
`ClockPoint(epoch, bias_s, additional_values=None)` builds a point, and
`validate()` checks it as the core does.

---

## Epochs

`ClockInstant` is the core `Instant` as stored: a `TimeScale` plus a split
Julian date (`jd_whole`, `fraction`, never summed here) or integer `nanos` from
J2000 in its own scale. `ClockInstant.from_civil(scale, ...)` builds one; the
second is read as the shortest decimal of the float given, with every digit
kept, as the reader reads a seconds field.

`ClockEpoch` holds civil fields read in the product's own time system. The
constructor refuses only fields that name no civil epoch in any RINEX clock
time system; it accepts `23:59:60` on a day that ends with a positive leap
second, which a UTC or `GLO` product answers and a GPS product refuses at
query time. `ClockEpoch.gps_seconds` is the fields read as GPS time and is None
for fields that name no GPS-time epoch.

---

## Queries

| Method | Epoch | Refusal |
| --- | --- | --- |
| `clock_s(satellite_id, epoch)` | `ClockEpoch`, read in the product's scale | `RinexClockQueryError` |
| `clock_s_at_instant(satellite_id, epoch)` | `ClockInstant`, explicitly tagged | `RinexClockQueryError` |
| `clock_s_at_gps_seconds(satellite_id, gps_seconds)` | GPS seconds | `RinexClockQueryError` |

All three return None for an unknown satellite and for an epoch outside the
stored bracket. An instant on a different timeline from the stored samples
returns None rather than being converted. Interpolation across a UTC leap
second uses elapsed time. GPS seconds outside the civil years 1 through 9999
are refused rather than converted. A product whose time system resolves to no
scale refuses `clock_s`.

---

## Writing

`to_rinex_string()` restates every retained line of a product read from text.
Records held as typed values - edited, inserted, or built from rows - are
written in the product's layout; values go in 19-column fields only when they
read back to the same bits, and an epoch only when a seconds field states it
exactly. A product built from rows is written with a header stating its
version, satellite system, time system and data types; GPST, GST, UTC and TAI
are written as 3.00 and QZSST and BDT as 3.04, and a scale no RINEX clock time
system names, GLONASS system time among them, is refused.

`to_rinex_string_with_policy(policy)` returns a `ClockWriteResult` (`text`,
`departures`; it unpacks as `(text, departures)`). `ClockWritePolicy` defaults
to strict; with `nearest_microsecond_epochs=ClockWriteLeniency.ALLOW` an epoch
no microsecond text states exactly is written as the nearest one and reported
as a `ClockWriteDeparture` of kind `EpochAtNearestMicrosecond`, naming the
`record` index, the `name`, the `epoch` the product holds and the fields
`written`. Values are never approximated under any policy.

---

## Editing

| Method | Change |
| --- | --- |
| `set_time_system(system)` | Writes one `TIME SYSTEM ID` record at the layout's columns, replacing any others or inserting it where Table A15 orders it. |
| `set_record_values(index, values)` | Replaces a record's declared values, bias first; the record keeps its type, name and exact seconds text. |
| `insert_record(index, record)` | Inserts a `ClockRecord` before `index`, or after the last record at `record_count`. |
| `remove_record(index)` | Removes a record with every line it spans and returns it. |
| `retain_records(keep)` | Keeps the records `keep(record)` accepts; returns the number removed. |
| `edit_records(edit)` | Replaces the values of every record `edit(record)` returns a sequence for; returns the number edited. |

The core validates the whole change before applying it. A refusal raises
`RinexClockEditError` and changes nothing: a value no 19-column field states
exactly, a name or year the layout cannot hold, an epoch that names no instant
in the product's time system (a `23:59:60` label in a continuous scale), a
value list that drops values the source record carries beyond its declared
count, an index with no record. `edit_records` applies all of its edits or
none. `retain_records` and `edit_records` call their Python function once per
record, in `records` order, before anything changes, so an exception it raises
leaves the product unchanged; the product cannot be read from inside it.

---

## Failures

| Exception | Bases | Raised by |
| --- | --- | --- |
| `RinexClockParseError` | `ParseError` | strict parsing |
| `RinexClockQueryError` | `RinexClockParseError`, `ValueError` | the three queries |
| `RinexClockWriteError` | `SidereonError`, `ValueError` | `to_rinex_string`, `to_rinex_string_with_policy` |
| `RinexClockEditError` | `SidereonError`, `ValueError` | `ClockRecord(...)`, `ClockPoint.validate`, the `from_*` constructors, the edits |

`clock_s` raised `RinexClockParseError` and the instant and GPS-seconds
queries raised `ValueError`; `RinexClockQueryError` is both, so either catch
still holds.

Each carries the core refusal as `detail`, a `RinexClockErrorDetail`: `kind`
is the core variant name (`MalformedAsRecord`, `MissingContinuation`,
`MalformedContinuation`, `BadField`, `InvalidInput`, `UnsupportedTimeScale`),
`details()` that variant's payload under its own keys, and `message`, `line`,
`reason`, `record`, `record_type`, `field`, `value` and `time_scale` are
conveniences, each None for a variant that names no such field. A failure with
no core error behind it carries no detail: an unreadable file, text that is not
UTF-8, an argument Python itself rejects, and a hand-built exception.

`ClockEpoch(...)` and `ClockInstant.from_civil(...)` sit on core helpers that
return an optional value with no error payload, so invalid fields raise a
plain `ValueError` with no `detail`.
