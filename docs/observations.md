# RINEX observations (OBS) Python API

A reference for the RINEX observation bindings in `sidereon`: what a parsed
product holds, how an epoch's rows are read, how the header can change part-way
through a file, and how writing refuses what it cannot state exactly.

The binding contains no observation parsing or modelling of its own. Every value
below is the one `sidereon-core` produced; this document describes the shapes it
arrives in and the distinctions the API keeps apart.

---

## What a parsed product holds

`parse_rinex_obs(text)` and `load_rinex_obs(source)` return a `RinexObs` for a
major version 2, 3 or 4 file.

```python
import sidereon

obs = sidereon.load_rinex_obs("ESBC00DNK_R_20201770000_01D_30S_MO.rnx")

obs.header  # ObsHeader parsed from the file header
obs.epochs  # every epoch record, in file order
obs.epoch_count  # how many
obs.epoch(3)  # one by zero-based index
obs.skipped_records  # records the reader could not represent, counted
```

Epochs stay in file order with event records and cycle slip records among them,
so an epoch index means the same thing everywhere in the API. `skipped_records`
counts what the reader kept out rather than aborting on: a satellite designator
the engine's id does not hold (one numbered `00`; the id takes `01`..`99` under
every system letter, so an extended GLONASS slot such as `R28` is held), and each
header contradiction that reads as ambiguous below.

Every property on `ObsHeader` and `ObsEpoch` returns a detached copy. Changing a
returned list, array or record changes nothing in the product; there is no
header-editing API, so the product stays the one authority for its own values.

---

## The three kinds of epoch record

RINEX writes three different things in the epoch stream, and `ObsEpoch` keeps
them apart rather than flattening them into one satellite map.

| `flag` | Record | Carries |
| --- | --- | --- |
| `0`, `1` | Observation epoch (`1` is a power failure) | `sats` |
| `6` (`sidereon.CYCLE_SLIP_FLAG`) | Cycle slip records | `cycle_slips` |
| any other above `1` | Event | `special_records` |

```python
epoch = obs.epoch(1)
epoch.flag  # 6
epoch.epoch  # ObsEpochTime, or None for an untimed event
epoch.declared_record_count  # the count the epoch line declared
epoch.rcv_clock_offset_s  # optional, seconds; None where the field is blank
epoch.epoch_picoseconds  # optional RINEX 4.02 extension; None where absent
epoch.sats  # {"G01": [ObsValue, ...]} — empty here
epoch.cycle_slips  # {"G01": [ObsValue, ...]} — the slips
epoch.special_records  # [] here
```

Three distinctions this keeps:

- **Slips are not measurements.** RINEX writes a detected or repaired cycle slip
  in the observation record layout, with the slip in place of the observation.
  They are in `cycle_slips`, never in `sats`, and never reach
  `observation_values`, `carrier_phase_rows` or `pseudoranges`.
- **An event carries text, not values.** A flag 3 epoch is followed by the
  header records for a new site occupation, and a flag 4 epoch by header records
  or comments. `special_records` holds them exactly as the file wrote them,
  label column and all, because what they mean depends on the labels they carry.
- **An untimed event is not an epoch at time zero.** RINEX lets an event without
  a significant epoch leave its epoch fields blank, and `epoch` is then `None`.
  An observation epoch and a cycle slip epoch always carry a time.

An event epoch and a cycle slip epoch produce no observation rows at all:

```python
assert len(obs.observation_values(1)) == 0
assert len(obs.carrier_phase_rows(1)) == 0
assert len(obs.pseudoranges(1)) == 0
```

A blank observation field stays blank. `ObsValue.value`, `.lli` and `.ssi` are
`None` where the file left the field empty, and the row is still there:

```python
[value.value for value in obs.epoch(0).sats["G01"]]  # [20000000.0, None, ...]
```

---

## Row extraction

Three methods flatten one epoch into row-aligned series. Each takes the epoch
index; the first two take an optional `ObservationFilter`, the third an optional
`SignalPolicy`.

```python
rows = obs.observation_values(0)
rows.satellites, rows.codes, rows.kinds  # lists, one entry per row
rows.values, rows.lli, rows.ssi  # numpy (n,) float64

phase = obs.carrier_phase_rows(0)
ranges = obs.pseudoranges(0)
```

A blank RINEX field and unknown carrier metadata read as `NaN` in the numeric
arrays. Rows are never dropped to avoid a `NaN`: `len(rows)` is the number of
(satellite, code) pairs the epoch's lists declare, and every list and array has
exactly that many entries.

`pseudoranges` is the exception by design: it takes, for each satellite, the
first code in that system's preference list whose value is *present*, so a
satellite with no usable code is left out rather than given a range of zero.

### Filters and policies

```python
gps_l1 = sidereon.ObservationFilter([(sidereon.GnssSystem.GPS, ["L1C"])])
obs.carrier_phase_rows(0, gps_l1)

policy = sidereon.SignalPolicy.default_for(obs.header.version)
policy = policy.with_override(sidereon.GnssSystem.GPS, ["C1W", "C1C"])
obs.pseudoranges(0, policy)
```

An empty filter (`ObservationFilter.all()`, or the default `None`) keeps every
parsed system and code.

---

## Union lists versus declared lists

`ObsHeader` carries two code lists per constellation, and they are not
interchangeable.

- `obs_codes(system)` is the product's **union**: the file header's codes first,
  then each code a later event list declares, in the order first declared. Epoch
  value vectors are index-aligned to this list, so it is what reads an epoch's
  values.
- `declared_obs_codes(system)` is what **that header** declares. On the file
  header it is the file header's own `SYS / # / OBS TYPES` records; on a header
  from `header_at` or `header_timeline` it is the list in effect at that epoch.

```python
obs.header.obs_codes(sidereon.GnssSystem.GPS)  # ["C1C", "L1C", "S1C"]
obs.header.declared_obs_codes(sidereon.GnssSystem.GPS)  # ["C1C", "L1C"]
obs.header_at(2).declared_obs_codes(sidereon.GnssSystem.GPS)
```

A value sits at its code's place in the union, blank at every position the list
in effect does not declare. A reordered event list therefore does not move a
value's meaning.

---

## The header in effect at an epoch

RINEX 3.05 section 6.5: "Each value remains valid until changed by an additional
header record." A flag 3 or 4 event's records lay over the header in effect
before it, and apply from that event's own epoch onward.

```python
obs.header_at(0)  # the file header
obs.header_at(9)  # with every event at or before epoch 9 laid over it

timeline = obs.header_timeline()
timeline.at(9)  # the same header, looked up rather than rebuilt
timeline.segment_index(9)  # which segment that is
timeline.segments()  # [(first_epoch_index, ObsHeader), ...]
timeline.segment_count
```

`header_at` reads the event records again on each call; `header_timeline` reads
them once, so a loop over many epochs uses the timeline. Both raise
`RinexObsParseError` if an event's header record does not read, which a product
read from text never holds; `header_at` also raises it for an index past the
last epoch.

`carrier_phase_rows` already uses the header in effect at the epoch it is given,
so a phase shift or GLONASS channel an event declares applies to the rows after
it and not to the rows before it. Nothing needs to be passed in.

---

## Corrections the header does not state

Two header records give a value that may be stated, absent, declared unknown, or
contradicted. The API keeps those four answers apart rather than collapsing them
into a number.

### `SYS / PHASE SHIFT`

RINEX 3 stores phase observations **already aligned**. The correction is
metadata describing what was applied when the file was written, for
reconstructing the original values. It is not a correction to apply again to
`value_cycles`.

`CarrierPhaseSeries` gives it three ways:

```python
phase = obs.carrier_phase_rows(0)
phase.phase_shift_results  # list[PhaseShiftResult], authoritative
phase.phase_shift_statuses  # the same rows' status strings
phase.phase_shift_cycles  # numpy (n,), NaN where not one value
phase.phase_shift_valid  # numpy (n,) bool
```

`PhaseShiftResult.status` is one of:

| `status` | Means | `cycles` | `corrections` |
| --- | --- | --- | --- |
| `"available"` | The header states one correction | the value | `[]` |
| `"unknown"` | The only covering record names just the constellation | `None` | `[]` |
| `"ambiguous"` | Records in one block give the code different corrections | `None` | every one, in header order |

Three things follow, and each is a distinction a wrapper could lose:

- **A stated `0.0` is not an unknown correction.** No record covering the row,
  or a covering record whose correction field is blank ("Correction applied
  (cycles) or blank if none", Table A2), states that none was applied: status
  `"available"`, `cycles == 0.0`. Only a record with no observation code says
  the alignment is unknown (section 5.2.12), and that is status `"unknown"`.
- **An ambiguous row keeps every conflicting value.** RINEX gives a header
  block's records no order to choose one by, so none is chosen.
  `corrections` holds them as written, a blank one as `None`, and the
  contradiction is counted in `skipped_records`.
- **`NaN` in `phase_shift_cycles` says only "not one value".** It does not say
  which of the two reasons applies. Read `phase_shift_valid` or the typed
  results before using a row.

```python
for code, result in zip(phase.codes, phase.phase_shift_results):
    if result.status == "ambiguous":
        print(code, "header gives", result.corrections)
    elif result.status == "unknown":
        print(code, "alignment declared unknown")
    else:
        print(code, "correction applied when written:", result.cycles)
```

The records themselves stay on `ObsHeader.phase_shifts`, as written:

```python
shift = obs.header.phase_shifts[0]
shift.system, shift.code, shift.correction_cycles
shift.satellites  # representable ids, as RINEX designators
shift.unrepresentable_satellites  # tokens the engine's id does not hold
shift.covers_every_satellite()  # names no satellite of either kind
shift.satellite_count()  # representable + unrepresentable
```

`code` is `None` for the constellation-only record, and `correction_cycles` is
`None` for a blank field. A record naming a satellite the engine cannot hold
keeps the token so the record writes back whole; no observation of that
satellite is retained, so the correction applies to none of them.

### `GLONASS COD/PHS/BIS`

`ObsHeader.glonass_code_phase_bias(code)` returns a `GlonassBiasResult` with
four statuses:

| `status` | Means |
| --- | --- |
| `"available"` | `value` is the bias in metres |
| `"none"` | The header gives this signal no bias |
| `"unknown"` | A blank record, or a blank bias for this signal (section 5.2.16) |
| `"ambiguous"` | One block gives it different biases, kept in `corrections` |

No record at all is `"none"`; a blank record is `"unknown"`. The raw entries stay
on `ObsHeader.glonass_cod_phs_bis`, which is `None` where the file has no record
and a list of `(code, bias_or_None)` pairs otherwise, so a blank bias is visible
as itself.

### RINEX 4 deprecation

RINEX 4.00, 4.01 and 4.02 Table A2 say of both records that "the lines should be
ignored by RINEX decoders and encoders". A product of that version therefore
reports `"available"` with `0.0` cycles for every phase-shift row and `"none"`
for every GLONASS bias, whatever its records say. The records are still parsed
and kept on the header so the file writes back whole.

---

## `LEAP SECONDS`

```python
leap = obs.header.leap_seconds  # None where the file has no record
leap.current, leap.delta_future, leap.week, leap.day
leap.time_system  # "GPS", "BDS", "BDT", or None
```

`time_system` is the A3 identifier from columns 25 to 27, returned exactly as the
record wrote it. Two distinctions hold:

- **An absent identifier is not an implicit `GPS`.** The field is optional;
  `None` means the record left it out.
- **`BDS` and `BDT` are not normalised to one another.** RINEX 3.03 introduced
  `BDS` and `GPS`, 3.05 renamed BeiDou's token to `BDT`, and RINEX 4 permits only
  `GPS`. As an accepted extension `GPS` is also read in pre-3.03 versions.

The version rule is enforced by the core, in one place, on both sides. The
parser refuses a token the file's own version does not permit, so a parsed
header always carries a token that version allows; the writer refuses one the
target version does not permit (see the two leap-second refusals below). The
binding adds no version table of its own.

---

## Writing, and what it refuses

```python
text = obs.to_rinex_string()
```

The version on the header decides the records written: below 3.0 the file is
version 2 throughout, otherwise version 3. The text is returned only when
reading it back gives the same product, compared field by field. Nothing is
dropped, rounded, wrapped or truncated to make a product fit — what the text
could not carry is refused.

### `RinexObsWriteError` and its typed detail

Both `RinexObs.to_rinex_string()` and
`SyntheticObservationSet.to_rinex_string()` raise `RinexObsWriteError`, so a
caller handles one type for both. It subclasses `RinexObsParseError`, so code
that caught that already still catches these.

```python
try:
    text = obs.to_rinex_string()
except sidereon.RinexObsWriteError as err:
    detail = err.detail
    print(detail.kind)  # the core variant name
    print(detail.details())  # that variant's own payload, by its own keys
```

`detail.kind` is the variant name and `detail.details()` its payload as native
values — a `GnssSystem` for a constellation, a RINEX designator for a satellite,
numbers as numbers. There is no need to parse `detail.message`. The conveniences
`epoch_index`, `system`, `satellite`, `version`, `code` and `time_system` return
`None` for a variant that names no such field.

A hand-built `RinexObsWriteError` has no core context, so `detail` defaults to
`None` and reading it never raises.

Variants, with their payload keys:

| `kind` | Payload keys |
| --- | --- |
| `CodeListsNotVersionTwo` | `system`, `position`, `code` |
| `NotVersionTwo` | `version` |
| `ScaleFactorsInVersionTwo` | `count` |
| `ValuesWithoutCodes` | `epoch_index`, `satellite`, `codes`, `values` |
| `CountsWithoutCodes` | `satellite`, `codes`, `counts` |
| `CodeListNotStated` | `system` |
| `EpochFlagTooWide` | `epoch_index`, `flag` |
| `EpochTimeMissing` | `epoch_index`, `flag` |
| `EpochPicosecondsNotInVersion` | `epoch_index`, `version` |
| `TooManyObservationTypes` | `count` |
| `CodeListsNotUnion` | `system` |
| `ValueOutsideDeclaredList` | `epoch_index`, `satellite`, `code` |
| `DeclaredListNotStated` | `system` |
| `EventRecordsUnreadable` | `message` |
| `ObservableNotRepresentable` | `system`, `code`, `version` |
| `LeapSecondsTimeSystemNotInVersion` | `time_system`, `version` |
| `InvalidLeapSecondsTimeSystem` | `time_system` |
| `ReadBackMismatch` | `what` |

### Reading a payload under a type checker

`details()` is typed as the union of that class's per-variant payloads, one
`TypedDict` per `kind` with the keys and value types the tables here list. The
same holds for `ObsDowngradeChange.details()` and `RinexLintFinding.details()`.

Nothing changes at run time: `details()` returns a plain `dict` and
`payload["count"]` works whatever the checker knows. To read one variant's key
under a checker, test `kind` — typed as the literal set of variant names, so a
misspelt one is an error — and cast to that variant's payload type:

```python
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:  # payload types live in the stub, not in the extension
    from sidereon import _ObsWriteObservableNotRepresentable

detail = err.detail
if detail.kind == "ObservableNotRepresentable":
    payload = cast("_ObsWriteObservableNotRepresentable", detail.details())
    print(payload["system"], payload["code"], payload["version"])
```

The payload types are stub-only helpers, named `_ObsWrite*` for write refusals,
`_ObsDowngrade*` for downgrade changes and `_Finding*` for lint findings. No
class under those names exists in the extension module, which is why the import
goes under `TYPE_CHECKING`.

A key test (`if "satellite" in payload:`) reads fine at run time, but mypy does
not narrow a union of `TypedDict`s that way and reports the subscript against
every member that lacks the key. Where a field is wanted without a cast, the
flat properties are the simpler route: `RinexObsWriteErrorDetail` answers
`epoch_index`, `system`, `satellite`, `version`, `code` and `time_system`
directly, `None` where the variant names no such field.

### What the parser accepts is not what the writer can write

The reader takes some header records without asking the version, so a parsed
product can hold a record its own version has no field for. `SYS / SCALE FACTOR`
is the case to know: the reader keeps it in a version 2.11 header, the version 2
writer has no record to write it as, and `to_rinex_string()` refuses with
`ScaleFactorsInVersionTwo`, whose `count` is every scale-factor record the file
header and its events declare.

```python
obs = sidereon.parse_rinex_obs(v211_text_with_a_scale_factor)  # accepted
obs.header.scale_factors  # the record, kept
obs.to_rinex_string()  # RinexObsWriteError
```

`downgrade_to_rinex2` is the path that gets such a product written: it removes
the scale-factor records and reports the removal as a `ScaleFactorsRemoved`
change, with the values it rounded alongside.

---

## Downgrading to RINEX 2

A product that has to lose something to become a version 2 file goes through the
explicit path, which returns every change it made:

```python
downgraded, changes = obs.downgrade_to_rinex2(2.11)
# or the free function, which is the same call
downgraded, changes = sidereon.downgrade_to_rinex2(obs, 2.11)

for change in changes:
    print(change.kind, change.details())
```

The source product is not modified. `ObsDowngradeChange.kind` is the variant
name and `details()` its payload:

| `kind` | Payload keys |
| --- | --- |
| `CodeRenamed` | `system`, `from_code`, `to_code` |
| `CodeMoved` | `system`, `code`, `from_position`, `to_position` |
| `CodeAdded` | `system`, `code` |
| `CodeListRemoved` | `system`, `codes` |
| `ValueRounded` | `epoch_index`, `satellite`, `code`, `from_value`, `to_value` |
| `CycleSlipRounded` | `epoch_index`, `satellite`, `code`, `from_value`, `to_value` |
| `ScaleFactorsRemoved` | `count` |
| `EpochPicosecondsRemoved` | `epoch_index`, `picoseconds` |
| `ClockOffsetRounded` | `epoch_index`, `from_offset_s`, `to_offset_s` |
| `InEventLists` | `epoch_index`, `change` (a nested `ObsDowngradeChange`) |
| `DeprecatedRecordsRemoved` | `label`, `epoch_index`, `records` |
| `EventRecordsRewritten` | `epoch_index`, `from_records`, `to_records` |

What version 2 still cannot state is refused rather than lost. Two refusals are
worth naming, because both are places where guessing would change what the file
says:

- **`ObservableNotRepresentable`.** A code on a physical carrier version 2
  cannot represent — BeiDou B1C against B1I, say — is refused with its
  constellation, its original code and the target version. Writing it under a
  version 2 name on another carrier would keep the number and move the
  measurement to a different signal, so no alias is invented.
- **`LeapSecondsTimeSystemNotInVersion`.** A `LEAP SECONDS` identifier the
  target version does not permit — `BDT` going to version 2 — is refused rather
  than dropped or rewritten as `GPS`.

```python
try:
    obs.downgrade_to_rinex2(2.11)
except sidereon.RinexObsWriteError as err:
    if err.detail.kind == "ObservableNotRepresentable":
        print(err.detail.system, err.detail.code, "cannot go to", err.detail.version)
```

---

## Quality control

`observation_qc` runs the rollups over a parsed product, and `lint_rinex_obs`
lints the text.

```python
report = sidereon.observation_qc(obs)
report.total_epoch_records  # every retained record, events among them
report.observation_epochs  # flag 0 and flag 1
report.event_records  # every flag above 1, cycle slip epochs included
report.power_failure_epochs  # flag 1
report.skipped_records
report.notes
```

`ObservationQcNote.kind` is one of:

- `non_monotonic_epoch` — adjacent observation epochs were duplicate or out of
  order; `epoch_index` names the later one.
- `interval_unresolved` — no interval could be resolved.
- `event_header_records_unread` — an event's header records did not read, so
  every epoch was taken with the file header. A product read from text never
  holds one.

Only `non_monotonic_epoch` carries an `epoch_index`; the other two return `None`.

Lint findings keep their native payload:

```python
for finding in sidereon.lint_rinex_obs(text).findings:
    print(finding.code, finding.kind, finding.severity.label, finding.details())
    print(finding.at.epoch_index, finding.at.satellite, finding.at.field)
```

`details()` returns that variant's own fields under their own keys. `detail` (no
`s`) is a debug rendering of the whole finding, fit for a log line; read
`details()` to get values.

Two OBS findings name the event stream: `ObsEventEpoch` (`OBS-B07`) reports a
retained event epoch with its flag, and `ObsEventHeaderUnreadable` (`OBS-B10`)
reports an event whose header records did not read, with the reader's message.

Every variant and its payload keys:

| `kind` | Payload keys |
| --- | --- |
| `ObsFatalParse` | `message` |
| `ObsUnpublishedVersion` | `version` |
| `ObsMissingHeader` | `label` |
| `ObsMissingObsTypes` | — |
| `ObsInvalidObsCode` | `system`, `code` |
| `ObsDuplicateObsCode` | `system`, `code` |
| `ObsTimeOfFirstMismatch` | `declared`, `declared_scale`, `observed`, `observed_scale` |
| `ObsTimeOfLastMismatch` | `declared`, `declared_scale`, `observed`, `observed_scale` |
| `ObsIntervalMismatch` | `declared_s`, `observed_s` |
| `ObsIntervalUnavailable` | — |
| `ObsInvalidInterval` | `declared_s` |
| `ObsSatelliteCountMismatch` | `declared`, `observed` |
| `ObsPrnObsCountMismatch` | `satellite`, `code`, `declared`, `observed` |
| `ObsGlonassSlotIssue` | `satellite`, `issue` |
| `ObsPhaseShiftUndeclaredCode` | `system`, `code` |
| `ObsScaleFactorIssue` | `system`, `code` |
| `ObsMarkerTypeIssue` | `marker_type` |
| `ObsIdentityFieldIssue` | `label`, `value` |
| `ObsImplausibleApproxPosition` | `radius_m` |
| `ObsImplausibleAntennaDelta` | `component`, `value_m` |
| `ObsEpochOrder` | `previous`, `current` |
| `ObsDuplicateEpoch` | `epoch` |
| `ObsSkippedRecords` | `count` |
| `ObsEpochSatCountMismatch` | `declared`, `retained` |
| `ObsEventHeaderUnreadable` | `message` |
| `ObsUnretainedHeader` | `label` |
| `ObsPseudorangeOutOfRange` | `code`, `value_m` |
| `ObsLossOfLockOutOfRange` | `code`, `lli` |
| `ObsEventEpoch` | `flag` |
| `ObsEmptySatelliteRecord` | — |
| `ObsEpochGap` | `gap_s`, `interval_s` |
| `NavFatalParse` | `message` |
| `NavLeapSecondsAbsent` | — |
| `NavIonoMalformed` | `message` |
| `NavDroppedBlock` | `satellite`, `message` |
| `NavDuplicateRecord` | `satellite`, `same_payload` |
| `NavUnsortedRecords` | — |
| `NavImplausibleRecord` | `satellite`, `field`, `value` |
| `NavUnhealthyRecords` | `system`, `count` |
| `NavOutOfScopeRecords` | `class`, `count` |

`declared` on `ObsPrnObsCountMismatch` and `code` on `ObsScaleFactorIssue` are
`None` where the record states no value. `declared` and `observed` are
`ObsEpochTime` on the two time-of-observation findings and counts elsewhere,
which is what the per-variant payload types keep apart. The core enum is
`#[non_exhaustive]`: a variant a newer core adds reads as `kind == "Unknown"`
with an empty payload until this binding maps it.

### Repairing, and writing the repair as CRINEX

`repair_rinex_obs(text, options)` returns a `RinexObsRepair`, and
`to_crinex_string()` writes the repaired product as RINEX and encodes that text:

```python
repair = sidereon.repair_rinex_obs(text, sidereon.RinexRepairOptions(...))
crinex = repair.to_crinex_string()
assert sidereon.parse_rinex_obs(sidereon.decode_crinex(crinex))
```

A writer refusal on that path raises `RinexObsWriteError` carrying its `detail`,
the same exception `repair.repaired.to_rinex_string()` raises; a CRINEX encoding
failure raises `CrinexParseError`.

`repair_rinex_obs` serializes nothing itself: it parses, applies the mechanical
edits the options name, and hands back the product. Writing happens only when
you call `to_crinex_string()`, which composes the RINEX writer and the CRINEX
encoder, so the writer refuses first and the encoder is never reached. Under the
default options no scale-factor record is removed, so a repair of accepted
version 2 text carrying one reaches `ScaleFactorsInVersionTwo` — see the section
above on parser acceptance versus writer representability.

---

## Known limits

- **No construction or editing API.** A `RinexObs` comes from parsing text or
  from `downgrade_to_rinex2`. Header and epoch properties are read-only detached
  copies, so refusals that only a hand-built product can reach —
  `InvalidLeapSecondsTimeSystem`, `EpochFlagTooWide`, `EpochTimeMissing` among
  them — are mapped and typed but not reachable from Python today.
- **A repair can hold a product the writer refuses.** Text the parser accepts is
  not the same set as products the writer can write, so a repair of accepted
  text can still refuse at `to_crinex_string()`. Version 2.11 text carrying a
  `SYS / SCALE FACTOR` record is the worked case: the reader keeps the record,
  the default repair options remove no scale factor, and the version 2 writer
  refuses with `ScaleFactorsInVersionTwo`. Put the product through
  `downgrade_to_rinex2` to get a version 2 file out of it.
- **The CRINEX intermediate representation is not exposed.** `decode_crinex`,
  `encode_crinex` and friends are text codecs; the typed `ObsStream` /
  `EpochRecord` / `SatRecord` values the core builds are not bound.
