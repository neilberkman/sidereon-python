# ANTEX Python API

Reference for the ANTEX 1.4 bindings in `sidereon`: what a parsed product
retains, how validity seconds are held, how lookups and writing refuse, and
what each refusal carries.

The binding is an interface over the `sidereon-core` reader and writer; the
values it returns are the values the core holds.

---

## Reading

`load_antex(source)` reads bytes, a bytearray, or a path. Millimetre fields
become metres as `mm * 1e-3`, the arithmetic of RTKLIB `readantex`, so stored
PCO and PCV values match that reader bit for bit.

Every record the format defines is retained, and a record the source does not
carry is None, never a default:

| Object | Retained |
| --- | --- |
| `Antex.header` | `AntexHeader`: `version` (`AntexVersion`: format `version`, satellite `system`), `pcv_type` (`AntexPcvTypeRecord`), header `comments`, `end_of_header`. |
| `Antex.outer_comments` | `AntexOuterComment`: comments between and after the blocks, each with the number of blocks before it. |
| `Antex.antenna_blocks` | Every antenna block in file order. |
| `Antenna` | `leading_comments` (before `TYPE / SERIAL NO`), `calibrations` (every `METH / BY / # / DATE` as `AntexCalibration`), `dazi_deg`, `zenith_grid` (`AntexZenithGrid`), `has_frequency_count`, `sinex_code`, `valid_from`, `valid_until`, `comments`, and the frequency sections. |
| `Antenna.frequency_sections` | `AntexFrequency` in file order: `frequency` label, `pco_m` (numpy `(3,)` north/east/up metres), `pcv_samples` (`AntexPcvSample`), and the `rms` section (`AntexFrequencyRms`) when the file has one. |

`AntexPcvTypeRecord.pcv_type` is `AntexPcvType.ABSOLUTE` or `RELATIVE`;
`reference_antenna` is the antenna relative values refer to - the stated type,
or `AOAD/M_T` when a relative file leaves it blank - and None for absolute
values.

`Antenna.frequencies` lists the section labels in file order; a label the file
repeats is listed again. `Antenna.dazi_deg`, `zenith_start_deg`,
`zenith_end_deg` and `zenith_step_deg` are None for a block without `DAZI` or
`ZEN1 / ZEN2 / DZEN`.

`Antex.skipped_records` counts records the parser found inconsistent and kept
the rest around: a line outside any record, a `# OF FREQUENCIES` count the
sections do not match, a frequency end record naming another frequency, a
header record after `END OF HEADER`, a block or section its own end record does
not close.

`Antex.antenna(id)` returns the latest block of an id, `antenna_intervals(id)`
every validity block of it in file order, `antenna_at(id, epoch)` the block
valid at an epoch, and `satellite_antenna(prn, epoch)` the satellite block for
a PRN.

## Validity instants

`AntexDateTime` is a GPS-time instant; GPS time has no leap-second label, so
`second` is `0..=59`. The `F13.7` seconds field is kept exactly: the fraction
is `fraction_digits / 10**fraction_scale`, normalized with no trailing zero,
and `fraction` gives it as an exact `decimal.Decimal`. A field such as
`59.9999999`, `.123456789012` or `1.2345678E-9` keeps every digit it states;
`nanosecond` is None for a fraction that is not a whole number of
nanoseconds. Instants compare and order by value, whatever the scale of their
fractions.

`AntexDateTime(year, month, day, hour=0, minute=0, second=0, *,
fraction_digits=0, fraction_scale=0)` builds one; a component outside the
calendar or GPS clock ranges, including a second of 60, raises `ValueError`.

## Lookups

`Antenna.frequency(label)`, `pco(label)` and `pcv(label, zenith_deg,
azimuth_deg=None)` raise `AntexQueryError` for an unknown label, a label whose
sections differ (`AmbiguousFrequency`, with the number of `sections`), a zenith
outside the grid, or an empty grid. Sections that repeat a label with
identical content answer as one.

## Writing

`Antex.to_antex_string()` writes every record from its retained value and
writes no record the source did not carry, apart from the start and end
records of blocks and sections. It raises `AntexWriteError` for a value the
fixed columns cannot state exactly, a validity second no form with a decimal
point fits in 13 columns, a frequency label that is not a system flag and a
two-column number, or public fields that disagree with the retained blocks.

## Failures

| Exception | Bases | Raised by |
| --- | --- | --- |
| `AntexParseError` | `ParseError` | `load_antex` |
| `AntexWriteError` | `AntexParseError`, `ValueError` | `Antex.to_antex_string` |
| `AntexQueryError` | `SidereonError`, `ValueError` | `Antenna.frequency`, `pco`, `pcv` |

Each carries the core refusal as `detail`, an `AntexErrorDetail`: `kind` is the
core variant name (`InvalidField`, `RepeatedRecord`, `DegenerateGrid`,
`InvalidInput`, `UnknownFrequency`, `AmbiguousFrequency`, `MissingPco`,
`EmptyPcvGrid`, `Unwritable`, `InvalidDateTime`) and `details()` its fields;
`antenna_id`, `record`, `field`, `value`, `frequency`, `reason` and `sections`
are None for a variant that names no such field. `AntexWriteError` also sets
`field` and `reason` from an `Unwritable` or `InvalidInput` refusal. An
unreadable source, bytes that are not UTF-8 and a hand-built exception carry no
detail.
