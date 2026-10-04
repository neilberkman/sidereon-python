# SP3 precise ephemeris Python API

Reference for what a parsed SP3 product keeps, how a clock without an orbit is
represented, what the writer refuses and what each refusal carries, and how SP3
merge reports state absence.

The binding is an interface over the `sidereon-core` parser, writer and merge.
Every number it returns is the number the core stored; it does no SP3
parsing, rounding or repair of its own.

---

## Product

`load_sp3(source)` parses bytes, a bytearray or a path into an `Sp3`.

| Member | Contents |
| --- | --- |
| `header` | `Sp3Header`, below |
| `comments` | the text of every `/*` record that carries text, in file order, trailing blanks removed |
| `skipped_records` | entries the parser skipped: records and `+` declarations whose satellite token names no representable satellite, and `EP`/`EV` correlation records |
| `satellites` | the header satellite list |
| `epochs_j2000_seconds` | the parsed epochs, numpy `(n,)` float64 |
| `state(satellite, epoch_index)` | `Sp3State`: a record that carries an orbit |
| `clock_record(satellite, epoch_index)` | `Sp3ClockRecord`: a clock beside a missing orbit |
| `clock_records_at(epoch_index)` | dict from satellite token to `Sp3ClockRecord` at one epoch, ascending |
| `to_sp3_string()` | the SP3 text, or `Sp3WriteError` |

`state` and `clock_record` raise `IndexError` for an epoch index past the end
and `KeyError` when the satellite has no such record at that epoch.
`clock_records_at` returns an empty dict for an epoch with no clock-only
records.

## Header

`Sp3Header` carries every header field the core retains, including the
descriptors the writer states back instead of writing constants of its own.

| Property | Source | Absent |
| --- | --- | --- |
| `version` | line 1, `"a"` to `"d"` | |
| `data_type` | line 1, `"P"` or `"V"` | |
| `num_epochs` | epoch records parsed | |
| `data_used` | line 1, columns 41-45 | None when blank |
| `coordinate_system`, `orbit_type`, `agency` | line 1 | |
| `gnss_week`, `seconds_of_week`, `epoch_interval_s`, `mjd`, `mjd_fraction` | line 2 | |
| `file_type` | first `%c` line, columns 4-5 | None when blank |
| `time_system` | first `%c` line: `GPS`, `GLO`, `GAL`, `TAI`, `UTC`, `QZS`, `BDT` or `IRN` | |
| `time_scale` | the core `TimeScale` the epochs are tagged with | |
| `pos_vel_base` | first `%f` line, columns 4-13 | None when blank |
| `clock_rate_base` | first `%f` line, columns 15-26 | None when blank |
| `satellites`, `satellite_accuracy_codes` | `+` and `++` lines, index-aligned | |

An explicit zero base is kept with its sign. A base is kept as the source
field stated it, including one finer than the canonical `F10.7` or `F12.9`
field; the writer reports that case (below) rather than the reader refusing
the file. The count line 1 declared is `Sp3.declared_epoch_count`.

## States and clock-only records

A `P` record whose position is the missing-orbit sentinel (`0.0 0.0 0.0`) beside
a valid clock is a clock-only record. It is never a state: `state` raises
`KeyError` for it, it contributes no interpolation node, and a position consumer
never receives a geocentre that was never observed.

| | `Sp3State` | `Sp3ClockRecord` |
| --- | --- | --- |
| position | `position_m`, numpy `(3,)` metres | none |
| clock | `clock_s`, None for the bad-clock sentinel | `clock_s` (seconds) and `clock_us` (the file's microseconds, as read) |
| velocity | `velocity_m_s`, None without a `V` record or for the missing-velocity sentinel | same |
| clock rate | `clock_rate_s_s`, None without a `V` record or for the bad-rate sentinel | `clock_rate_s_s` and `clock_rate_raw` (the file's 1e-4 microseconds per second, as read) |
| flags | `clock_event`, `clock_predicted`, `maneuver`, `orbit_predicted` | same |

`Sp3State` reaches Python through one constructor, so `state`, the interpolant
queries and the staleness selection carry the same fields, including
`clock_rate_s_s`.

## Writing

`to_sp3_string()` writes the format its header names. Every numeric field is
checked by reading the column back the way the parser reads it, and must give
the stored value bit for bit. Nothing is rounded, shifted, defaulted or dropped
to make a write succeed; a product that holds a value its columns cannot state
raises `Sp3WriteError`. A product read from a file writes its record values
back unchanged; it can still be refused on a header base finer than its field,
on header text wider than its columns, or on velocity records under a `P`
header.

A mean or median merge usually holds positions finer than the millimetre
`F14.6` column, and a clock shifted onto the reference clock datum can hold a
clock finer than its column, so writing such a merge raises
`RecordValueNotRepresentable` naming the value. `data.write_sp3` and
`data.fetch_merged_sp3_file` write nothing in that case.

### Sp3WriteError

`Sp3WriteError` subclasses `Sp3ParseError` (and so `ParseError` and
`SidereonError`) and `ValueError`. `detail` is an `Sp3WriteErrorDetail`, or None
on a hand-built instance.

`Sp3WriteErrorDetail` has `kind` (the core variant name), `message` (the core
message, equal to `str(error)`), `details()` (every field of the variant), and
the conveniences `field`, `epoch_index`, `satellite`, `columns` and `decimals`,
which are None for a variant that names no such field. In `details()` a
satellite is its token under `"satellite"`, a time scale is a `TimeScale`, and
the SP3 time system is its label.

| `kind` | `details()` keys |
| --- | --- |
| `NoEpochs` | none |
| `TextNotColumnSafe`, `TextNotColumnStable`, `BlankDescriptor` | `field`, `value` |
| `EmptyComment` | `index` (into `Sp3.comments`), `value` |
| `TextTooWide` | `field`, `columns`, `value` |
| `IntegerTooWide` | `field`, `columns`, `value` |
| `NonFinite` | `field` |
| `NumberTooWide`, `PrecisionNotRepresentable` | `field`, `columns`, `decimals`, `value` |
| `YearNotRepresentable` | `epoch_index`, `year` |
| `EpochRepresentationUnsupported` | `epoch_index` |
| `EpochNotRestatable` | `epoch_index`, `field_seconds`, `residual_s` |
| `EpochTimeScaleMismatch` | `epoch_index`, `epoch_scale`, `header_scale` |
| `HeaderTimeScaleMismatch` | `time_system`, `time_scale` |
| `EpochCountMismatch` | `declared`, `epochs` |
| `AccuracyCodeCountMismatch` | `satellites`, `codes` |
| `DuplicateSatellite` | `satellite` |
| `SatelliteNotRepresentable` | `satellite`, `system`, `prn` |
| `EpochArrayLengthMismatch` | `field`, `epochs`, `entries` |
| `UndeclaredSatelliteRecord`, `ConflictingRecords` | `satellite`, `epoch_index` |
| `VelocityStateInPositionProduct`, `RecordValueNonFinite` | `field`, `satellite`, `epoch_index` |
| `RecordValueTooWide` | `field`, `satellite`, `epoch_index`, `columns`, `decimals`, `column_value` |
| `RecordValueNotRepresentable` | `field`, `satellite`, `epoch_index`, `columns`, `decimals`, `stored`, `column_value` |
| `RecordReadsAsAbsent` | `field`, `satellite`, `epoch_index`, `column_value` |
| `RecordFieldsDisagree` | `field`, `satellite`, `epoch_index`, `stored`, `native` |

`stored` is the value in the product's own units (metres, seconds, metres per
second, seconds per second) and `column_value` the number the column would
carry, in the format's units. `EpochNotRestatable`'s `residual_s` is the stored
epoch minus the instant the record would state, and None where no candidate
record could be read back at all (the core holds NaN there).
`RecordFieldsDisagree`'s `stored` and `native` are each None where the product
holds no such value.

A refusal variant this build does not name, such as one a later core adds,
still maps: `kind` is its core name and `details()` holds its core `Debug`
text under `"debug"`. `SatelliteNotRepresentable` is a header satellite with no
`01`..`99` token that reads back as itself; `system` and `prn` give it apart
from its rendered `satellite` text. A product parsed or merged from SP3 text
cannot hold such a satellite.

## Merge agreement

`merge_sp3` retains clock-only records: a satellite whose orbit is missing or
quarantined but whose clock reached a consensus is written as the missing-orbit
sentinel beside that clock. Merged products carry no velocities or clock rates.

A clock-only cell has an `Sp3AgreementMetric` with `position_members == 0` and
`position_rms_m` and `position_max_m` None. A single-source orbit has a zero
spread, not None: it is present and has no dispersion.

`Sp3EpochAgreement.position_rms_m` and `position_max_m` cover only the
multi-source position cells of the epoch and are None for an epoch with none
(`satellites == 0`); `per_epoch_agreement` carries the same None in its tuples.
`Sp3MergeReport.position_agreement_max_m` is the largest spread over the cells
that carry an orbit, None when no cell does.

`data.MergeReport.to_dict()` writes merged-SP3 report schema 3 with these
values and the merge audit trail below. `data.verify_merge_report` also reads
schema 2, written before the audit trail, and schema 1, written while every
merge cell carried an orbit, by the rules it was written under: there an epoch
without a multi-source position consensus holds a 0.0 spread and the maximum
over every cell of the epoch.

## Merge audit trail

`Sp3MergeReport` lists every cell and epoch a merge did not write, with its
reason. `omitted_epochs` are union-grid epochs at which no cell was accepted;
they are not written as blocks of missing records, so the product ends where
its data ends. `arc_withheld` lists the positions a source carried that
precedence did not write because the preferred source (under
`SATELLITE_ARC`, the arc owner) carried none there. `clock_omissions` lists
each source clock left out, one `Sp3ClockOmission` per source and cell, with
`reason` `datum_not_observable` (the source's datum offset to source 0 is not
estimable at that epoch and is never extrapolated), `preferred_source_without_clock`
or `no_consensus`, and whether the cell got a clock from other sources.
`dropped_input_epochs` lists each input epoch that took no part, with reason
`off_target_grid` or `not_on_tick_axis`. `single_source_fraction` is the share
of accepted cells no second source cross-checked.

`Sp3MergeOptions(provenance=...)` records per-epoch provenance
(`Sp3MergeProvenance`): under `FULL` one `Sp3CellProvenance` per accepted cell
with an `Sp3CellSelection` for its position and clock, and under both modes
every `Sp3PrecedenceTransition` and one `Sp3ContributorCoverage` per source.
A mean- or median-combined value is `kind == "combined"` with no
`selected_source`: no single contributor supplied it. `provenance` is None
when it was not requested.

`Sp3MergeOptions(verify_continuity=Sp3ContinuityOptions(...))` checks the
merged product's continuity after the merge and attributes each
`Sp3MergeContinuityViolation` to the contributors whose records it rests on:
`cells` names every merged cell the finding rests on (for a hold-out residual,
the held-out sample and each node its prediction used) with the selection the
merge recorded there, and `crosses_contributors` marks a splice. The report's
`nodes` (`Sp3InterpolationNodes`) give, per satellite, exactly the nodes an
evaluation window's interpolations select, and `violations_influencing`,
`splices_influencing` and `verdict` scope the findings to a window by them.
Neither option changes the merged product.

## Coverage

`Sp3.satellite_coverage()` states, for every satellite, where its positions and
where its clocks are: an `Sp3ChannelCoverage` of `Sp3CoverageSpan`s (inclusive
epoch-index runs) and `Sp3CoverageGap`s, and the `Sp3EpochGrid` the epochs lie
on. Its `interval_s` is None when the steps are not whole multiples of the
declared interval or the epochs are out of order. A clock-only record is a clock and not
a position; a position with the missing-clock sentinel is a position and not a
clock. Coverage describes the records a product holds, not where an
interpolation is served.
