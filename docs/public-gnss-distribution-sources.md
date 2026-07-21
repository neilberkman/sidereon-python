# Public GNSS distribution sources

Sidereon treats a GNSS product and the place used to obtain it as separate
public concepts. An exact request fixes the publisher, analysis-center product
line, solution class, issue, cadence, coverage date, official filename, and
format. Selecting another distributor never changes those fields.

The exact-acquisition API is available from `sidereon.distribution` and is
also lazily re-exported by `sidereon.data`.

```python
from datetime import date
import os

from sidereon import data

product = data.mgex_sp3("cod", date(2026, 6, 25))
request = data.request(
    product,
    [
        data.Distribution.nasa_cddis(),
        data.Distribution.direct(),
    ],
)

result = data.acquire(
    request,
    earthdata_auth=data.EarthdataAuth.bearer(os.environ["EARTHDATA_TOKEN"]),
)
print(result.path)
print(result.provenance.distribution_source)
print(result.provenance.sha256)
```

Only the listed distributors are attempted, in order. A 404 from CDDIS may
therefore lead to the direct archive in this example, but both attempts use the
same `ProductIdentity`. Sidereon does not change center, tier, issue, cadence,
date, product family, or official filename. Earlier public failures are retained
in `result.provenance.attempts`.

For a workflow that requires several exact products, declare the complete set
before acquisition and gate dependent processing on the resolved identities:

```python
expected = [request_a.identity, request_b.identity]
available = [
    result_a.provenance.resolved_identity,
    result_b.provenance.resolved_identity,
]
data.validate_exact_product_set(expected, available)
```

The gate rejects empty declarations, duplicates, missing products, undeclared
products, and same-filename identities with different prediction metadata. It
returns only by completing successfully; otherwise it raises
`ExactProductSetError`. A format version resolved from validated bytes may be
present only on the available identity.

For SP3 observed/predicted timing, use `sp3.prediction_summary()`. It reads the
record flags in the product; issue times and catalog fields are not substitutes
for that metadata.

Exact SP3 content can be validated independently of acquisition:

```python
from pathlib import Path

import sidereon

exact = sidereon.ExactSp3Request.from_identity(request.identity)
sp3, coverage = sidereon.parse_exact_sp3(Path(result.path).read_bytes(), exact)
assert coverage in {
    sidereon.ExactSp3Coverage.HALF_OPEN,
    sidereon.ExactSp3Coverage.INCLUSIVE,
}
```

The exact gate requires a finite, positive, supported requested cadence; a
matching finite header cadence; matching line-1 and line-2 start metadata; a
complete SP3 header and terminal `EOF`; a declared epoch count equal to the
parsed count; the expected producer agency; and a strictly regular parsed epoch
grid. For a one-day five-minute request, both 288 half-open epochs through 23:55
and 289 inclusive epochs through the next midnight are valid. A 287-epoch,
longer, irregular, cadence-mismatched, or metadata-inconsistent product raises
`ExactSp3ValidationError`. Parsed `Sp3` values expose
`declared_epoch_count` and `declared_start_j2000_s` for audit use.

## Sources and caller input

The supported source descriptors are:

- `Distribution.direct()` for the cataloged analysis-center or IGS URL;
- `Distribution.nasa_cddis()` for the official CDDIS HTTPS archive;
- `Distribution.local_file(path, compression="auto")`;
- `Distribution.in_memory(content, compression="auto")`.

SP3 CDDIS paths use the GNSS-products GPS-week directory. Current long-name
IONEX paths use `ionex/<year>/<day-of-year>`. Both retain the exact official
filename. CDDIS packages current long-name products with gzip. IGS combined
final SP3 before GPS week 2238 uses the official `igs<week><day>.sp3.Z` name and
Unix-compress packaging; Python accepts `compression="unix_compress"` and
detects the `.Z` magic bytes in `compression="auto"` input. Unix-compress
decoding validates terminal code completeness and padding before product
parsing, so a structurally partial archive fails decompression.

IGS combined final SP3 begins at GPS week 0730. Before week 2238, Sidereon
derives the verified short filename and CDDIS location; at week 2238 it switches
to `IGS0OPSFIN_<epoch>_01D_15M_ORB.SP3.gz`. Current direct BKG locations use the
long filename. The reviewed BKG listings did not establish one uniform legacy
direct layout, so a historical direct request is rejected instead of guessed.
IGS broadcast navigation retains its existing `BRDC00WRD` catalog behavior.
Use `data.product_solution_class("igs", "sp3")` for `"final"` and
`data.product_solution_class("igs", "nav")` for `"broadcast"`; classification
is intentionally product-aware rather than inferred from the center alone.

CODE product families are routed independently through AIUB's HTTPS service:

```text
SP3 and clock:  https://www.aiub.unibe.ch/download/CODE_MGEX/CODE/<year>/...
final IONEX:    https://www.aiub.unibe.ch/download/CODE/<year>/...
rapid IONEX:    https://www.aiub.unibe.ch/download/CODE/...
predicted P1:   https://www.aiub.unibe.ch/download/CODE/IONO/P1/<year>/...
predicted P2:   https://www.aiub.unibe.ch/download/CODE/IONO/P2/<year>/...
```

The current `cod` catalog line is not applied to dates before GPS week 2238;
those legacy CODE products use different names and directories that are not yet
modeled by this API.

Other long-name SP3 families are likewise bounded by their verified public
archive eras: ESA final starts on 2014-01-05, GFZ rapid on 2020-05-13, ESA
ultra-rapid on 2022-10-04, and GFZ ultra-rapid on 2020-10-06. IGS ultra-rapid
is modeled only from GPS week 2238. ESA ultra-rapid defaults to `15M` through
the 2025-02-02 0600 issue and `05M` from the 1200 issue; the date-only query
uses the start-of-day (`0000`) convention. GFZ ultra-rapid defaults to `15M`
through 2021-05-15 and `05M` from 2021-05-16.

CDDIS distribution is rejected for every pre-week-2238 long-name SP3 or IONEX
identity. The one supported historical CDDIS SP3 exception is the IGS
combined-final family, whose exact identity uses the documented legacy short
name and `.Z` packaging; it is not a long-name alias.

CDDIS support is exact-product specific, not publisher-wide. Sidereon does not
map ESA's `ESA0MGNFIN` final SP3 identity to CDDIS in any era: the official ESA
archive serves that line directly, while the public CDDIS evidence reviewed for
this release does not establish an exact `ESA0MGNFIN` object. Current long-name
IONEX identities use CDDIS's documented
`ionex/<year>/<day-of-year>/<official-filename>.gz` layout; historical long-name
identities are not projected backward across the week-2238 transition.

NASA documents CDDIS HTTPS access and Earthdata Login at:

- <https://www.earthdata.nasa.gov/centers/cddis-daac/archive-access>
- <https://urs.earthdata.nasa.gov/documentation/for_users/data_access/curl_and_wget>
- <https://urs.earthdata.nasa.gov/documentation/for_users/data_access/python_user_token_script>

The product paths and long filenames follow NASA's GNSS product pages and the
IGS long-product-filename guidelines:

- <https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product>
- <https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/atmospheric-products>
- <https://files.igs.org/pub/resource/guidelines/Guidelines_for_Long_Product_Filenames_in_the_IGS.pdf>

## Earthdata credentials

Credentials are always caller supplied. A bearer token can be passed with
`EarthdataAuth.bearer(token)`. The value is excluded from object repr, errors,
source-attempt records, and provenance. Authorization headers are sent only to
the allowed CDDIS and Earthdata Login hosts.

Earthdata's documented netrc mechanism is also supported:

```text
machine urs.earthdata.nasa.gov login EARTHDATA_USERNAME password EARTHDATA_PASSWORD
```

```python
result = data.acquire(
    request,
    earthdata_auth=data.EarthdataAuth.from_netrc(),
)
```

The HTTP client retains cookies during the deliberate CDDIS-to-URS redirect
flow. Redirect targets are restricted to the documented NASA hosts; direct
analysis-center redirects retain the existing Sidereon host policy. Query
strings, user information, cookies, bearer tokens, and passwords are removed
from recorded URLs.

## Provenance

`AcquisitionProvenance` records:

- requested and parsed/resolved identity;
- publisher, distributor, and official filename;
- sanitized original and final public URLs;
- UTC retrieval time, byte counts, SHA-256 hashes, compression, ETag, and
  Last-Modified when present;
- cache-hit versus network retrieval;
- each failed explicitly allowed source attempted before success.

The decompressed standard product and exact downloaded archive bytes are kept
separately. The provenance sidecar contains no credentials.

### Merged SP3 provenance

`data.fetch_merged_sp3` acquires every successful candidate through this same
exact path. Each `report.contributors` entry carries an `artifact_identity`
with the requested and parsed identities, distributor, official filename,
decompressed and archive hashes and lengths, and compression. Its separate
`acquisition_facts` contains the sanitized URLs, retrieval time, cache status,
HTTP metadata, and prior failed attempts.

The report's `stable_input_identity` is computed by the Rust core from the
complete artifact identities and effective `Sp3MergeOptions`. It is unchanged
by cache hits, retrieval timestamps, mapping order, or mean/median contributor
enumeration. Precedence contributor order is input priority and remains
identity-bearing. Changing an artifact, resolved identity, contributor set, or
merge policy changes the identity. `input_identity_schema_version` identifies
the canonical encoding. Calling `data.sp3_merge_input_identity` directly returns
the stable ID together with the canonical contributor order and, for precedence
merges, the distinct ordered priority contributors. Two-value unpacking remains
available for callers that only need `(schema_version, stable_id)`.
An incomplete or inconsistent artifact record is rejected rather than hashed.
The JSON-safe `merge_policy` records every effective option. When precedence
combining is selected, `precedence_artifact_sha256` records the ordered artifact
priority; mean and median policies leave that list empty.

```python
merged, report = data.fetch_merged_sp3(day, ["cod", "esa"])
persisted_report_json = json.dumps(report.to_dict(), sort_keys=True)

path, same_report = data.fetch_merged_sp3_file(
    day, ["cod", "esa"], "merged.sp3", return_report=True
)

assert data.verify_merge_report(json.loads(persisted_report_json))
```

Persisted-report verification is exact-schema and non-coercive at every nested
level. It rechecks contributor filename, catalog pattern, center, date, and issue
against the authenticated artifact identity; contributors and absent centers
must exactly partition `requested_centers`; count and metric domains must be
internally consistent. Unknown authorization, cookie, secret, or local-path
fields are rejected. Alias candidates are accepted only after the parsed epoch
grid proves both the catalog cadence and complete declared coverage span.

The default file-helper return remains the written path for compatibility.
Neither `report.to_dict()` nor the stable identity includes credentials,
cookies, authorization headers, cache directories, or temporary paths.

## Cache behavior

The cache key includes source plus all exact identity fields. Products from
different publishers, solution classes, issues, dates, cadences, or families
cannot collide. A cache hit is accepted only after checking the decompressed
and archive hashes, lengths, stored identity, provenance, and a fresh SP3 or
IONEX parse with coverage-start and cadence checks.

Each accepted entry is one immutable transaction containing the decompressed
product, original archive, and provenance. A SHA-256-bound commit record names
that transaction. The commit record is replaced atomically only after all three
files and their directories have been synchronized. Readers follow only the
commit record and then repeat the content-hash, length, identity, source,
caller-checksum, and parsed-product checks.

On Linux and macOS, acquisition delegates to the shared Rust transaction
implementation. Concurrent processes and threads use its per-entry advisory
lock across cache validation, acquisition, and commit, so a waiter rechecks and
reuses the completed entry instead of downloading it again.
`cache_lock_timeout_s` bounds the wait (30 seconds by default); a timeout is a
terminal `CacheWriteFailure` and does not authorize trying another source. The
operating system releases the lock if its owner exits or is killed. A later lock
owner may remove uncommitted transaction directories without guessing whether
a writer is still alive.

Publication relies on same-filesystem atomic `rename`/`replace`, `fsync` of each
file, and `fsync` of the entry, entries, and commit-record directories. A process
death or power loss at a publication boundary therefore leaves the previous
complete commit or no acceptable commit. Valid cache triples from the
0.29.0-0.29.2 layout are fully revalidated and migrated without a remote
request. The legacy files are then ignored. `result.path` retains the official
filename inside the immutable transaction directory.

These crash and cross-process guarantees are for local filesystems on Linux and
macOS. Filesystems that do not honor POSIX advisory locks, atomic same-directory
rename, or directory synchronization are outside the guarantee. Existing
verified entries are returned without a remote request, including during a
remote outage.

Low-level consumers can use `sidereon.exact_cache.ExactProductCache`,
`entry_lock`, and `read` directly. Those calls authenticate bytes and identity;
the caller remains responsible for transport and product-format validation.

## Typed failures

Exact acquisition distinguishes authentication required, authentication
failed, authorization denied, not published (404), retired endpoint (410),
redirect policy, malformed URL, timeout/connection/HTTP transport, HTML or
invalid content, content-length mismatch, gzip/Unix-compress failure, caller
checksum, product parse/semantic validation, and cache read/write failures.

An ordinary 404/not-published response or retired endpoint may advance to the
next explicitly allowed distributor for the same exact identity. An exhausted
timeout, connection, 408, 429, or server failure may try another distributor,
but is preserved if no source succeeds. Malformed data, parsing, checksum,
cadence, span, identity, authentication/authorization, redirect-policy, cache,
and caller-configuration failures are terminal and do not authorize fallback.
No official source reviewed here documented a moving-alias race that justified
a validation-error exception, so none is implemented.

Retries are bounded and limited to connection/timeouts, 408, 429, and server
errors. NASA publishes no universal numeric CDDIS request limit; callers should
keep concurrency modest, honor service responses, and use a verified local
cache. NASA's public data-use guidance is at
<https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy>.

## Evidence for the 0.33 catalog and validation changes

All sources in this table were accessed on 2026-07-20.

| Change | Official public evidence | Accessed |
|---|---|---|
| IGS combined final orbits begin at GPS week 0730. | [1994 IGS Annual Report](https://files.igs.org/pub/resource/pubs/94an_repta.pdf) | 2026-07-20 |
| IGS final SP3 changes from the short `.sp3.Z` convention to the long `.SP3.gz` convention at GPS week 2238. | [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [IGSMAIL-8256](https://lists.igs.org/pipermail/igsmail/2022/008252.html), [week 2237 CDDIS object](https://cddis.nasa.gov/archive/gnss/products/2237/igs22370.sp3.Z), [week 2238 CDDIS object](https://cddis.nasa.gov/archive/gnss/products/2238/IGS0OPSFIN_20223310000_01D_15M_ORB.SP3.gz) | 2026-07-20 |
| CDDIS's documented legacy orbit convention and the IGS week-2238 transition support one modeled pre-transition mapping: the IGS combined-final short name with `.Z`. Other centers' pre-transition long names are not projected into CDDIS. | [NASA precise-orbit convention](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product), [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [week 2237 legacy object](https://cddis.nasa.gov/archive/gnss/products/2237/igs22370.sp3.Z), [week 2238 long-name object](https://cddis.nasa.gov/archive/gnss/products/2238/IGS0OPSFIN_20223310000_01D_15M_ORB.SP3.gz) | 2026-07-20 |
| ESA publishes the `ESA0MGNFIN` final-SP3 line from its official archive; the reviewed public CDDIS convention does not establish an exact CDDIS mapping for that identity, so Sidereon rejects the distributor instead of substituting another ESA line. | [ESA week-2320 listing](https://navigation-office.esa.int/products/gnss-products/2320/), [ESA final-SP3 object](https://navigation-office.esa.int/products/gnss-products/2320/ESA0MGNFIN_20241760000_01D_05M_ORB.SP3.gz), [NASA precise-orbit convention](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product) | 2026-07-20 |
| BKG's current path is `IGS/products/<week>`, while transition-era listings do not establish one universal historical direct path. | [BKG week 2238](https://igs.bkg.bund.de/root_ftp/IGS/products/2238/), [BKG legacy week 2235](https://igs.bkg.bund.de/root_ftp/IGS/products/orbits/2235/), [BKG week 2236](https://igs.bkg.bund.de/root_ftp/IGS/products/2236/) | 2026-07-20 |
| SP3 declares start/count on line 1 and start/cadence metadata on line 2; its mandatory records and terminal `EOF` are integrity inputs. | [SP3-d specification](https://files.igs.org/pub/data/format/sp3d.pdf) | 2026-07-20 |
| AIUB's current CODE families use product-specific `CODE_MGEX/CODE/<year>`, `CODE/<year>`, `CODE`, and predicted IONEX tier directories. | [AIUB product documentation](https://www.aiub.unibe.ch/download/AIUB_AFTP.TXT), [MGEX 2026 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE_MGEX%2FCODE%2F2026), [CODE 2026 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE%2F2026), [P1 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE%2FIONO%2FP1%2F2026), [P2 listing](https://code.aiub.unibe.ch/s3_script/aiub_s3_bucket_listing.php?path=CODE%2FIONO%2FP2%2F2026) | 2026-07-20 |
| GFZ rapid SP3 used `15M` through 2021 day 137 and `05M` from day 138. | [GFZ week 2158 listing](https://isdc-data.gfz.de/gnss/products/rapid/w2158/) | 2026-07-20 |
| ESA's MGEX final-SP3 and clock archive begins on 2014-01-05. | [preceding week 1773](https://navigation-office.esa.int/products/gnss-products/1773/), [first week 1774 listing](https://navigation-office.esa.int/products/gnss-products/1774/), [first SP3 object](https://navigation-office.esa.int/products/gnss-products/1774/ESA0MGNFIN_20140050000_01D_05M_ORB.SP3.gz), [first clock object](https://navigation-office.esa.int/products/gnss-products/1774/ESA0MGNFIN_20140050000_01D_30S_CLK.CLK.gz) | 2026-07-20 |
| GFZ's rapid-SP3 and clock listing begins on 2020-05-13. | [GFZ week-2105 listing](https://isdc-data.gfz.de/gnss/products/rapid/w2105/), [first rapid SP3 object](https://isdc-data.gfz.de/gnss/products/rapid/w2105/GFZ0OPSRAP_20201340000_01D_15M_ORB.SP3.gz), [first rapid clock object](https://isdc-data.gfz.de/gnss/products/rapid/w2105/GFZ0OPSRAP_20201340000_01D_30S_CLK.CLK.gz) | 2026-07-20 |
| IGS operational ultra-rapid long names start with GPS week 2238. | [IGS transition guideline](https://files.igs.org/pub/resource/guidelines/Guideline_for_the_transition_of_the_IGS_products_to_IGS20_and_long_filenames_v2.0.pdf), [BKG week-2238 listing](https://igs.bkg.bund.de/root_ftp/IGS/products/2238/), [first long-name ultra SP3 object](https://igs.bkg.bund.de/root_ftp/IGS/products/2238/IGS0OPSULT_20223310000_02D_15M_ORB.SP3.gz) | 2026-07-20 |
| ESA's operational ultra-rapid SP3 line begins on 2022-10-04. | [preceding week 2229](https://navigation-office.esa.int/products/gnss-products/2229/), [week-2230 listing](https://navigation-office.esa.int/products/gnss-products/2230/), [first ultra SP3 object](https://navigation-office.esa.int/products/gnss-products/2230/ESA0OPSULT_20222770000_02D_15M_ORB.SP3.gz) | 2026-07-20 |
| GFZ's operational ultra-rapid SP3 listing begins on 2020-10-06. | [GFZ week-2126 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2126/), [first ultra SP3 object](https://isdc-data.gfz.de/gnss/products/ultra/w2126/GFZ0OPSULT_20202800000_02D_15M_ORB.SP3.gz) | 2026-07-20 |
| ESA ultra-rapid SP3 changes from `15M` at the 2025-02-02 0600 issue to `05M` at 1200. | [ESA week-2352 listing](https://navigation-office.esa.int/products/gnss-products/2352/), [0600 15M object](https://navigation-office.esa.int/products/gnss-products/2352/ESA0OPSULT_20250330600_02D_15M_ORB.SP3.gz), [1200 05M object](https://navigation-office.esa.int/products/gnss-products/2352/ESA0OPSULT_20250331200_02D_05M_ORB.SP3.gz) | 2026-07-20 |
| GFZ ultra-rapid SP3 defaults to `15M` through 2021-05-15 and `05M` from 2021-05-16; one 0000 `05M` object overlaps the otherwise-`15M` final day. | [GFZ week-2157 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2157/), [last 15M issue](https://isdc-data.gfz.de/gnss/products/ultra/w2157/GFZ0OPSULT_20211352100_02D_15M_ORB.SP3.gz), [overlapping 05M object](https://isdc-data.gfz.de/gnss/products/ultra/w2157/GFZ0OPSULT_20211350000_02D_05M_ORB.SP3.gz), [GFZ week-2158 listing](https://isdc-data.gfz.de/gnss/products/ultra/w2158/), [first next-day 05M object](https://isdc-data.gfz.de/gnss/products/ultra/w2158/GFZ0OPSULT_20211360000_02D_05M_ORB.SP3.gz) | 2026-07-20 |
| CDDIS IONEX filenames transitioned from historical short names toward long names beginning at week 2238, with center-specific timing. Sidereon therefore does not derive a pre-transition CDDIS URL for a caller's long-name IONEX identity. | [IGS ionospheric products](https://igs.org/products/#ionosphere), [NASA Earthdata support clarification](https://forum.earthdata.nasa.gov/viewtopic.php?t=4779) | 2026-07-20 |
| CDDIS HTTPS access uses Earthdata Login and keeps product archive paths distinct from product identity. | [NASA CDDIS archive access](https://www.earthdata.nasa.gov/centers/cddis-daac/archive-access), [NASA precise-orbit product](https://www.earthdata.nasa.gov/data/space-geodesy-techniques/gnss/precise-orbits-product) | 2026-07-20 |

Compatibility assessment: the Python additions are source-compatible. Catalog
derivation is deliberately stricter for unsupported historical CODE dates and
dates before the verified SP3-family publication floors, known center/product
mismatches now fail before HTTP, and integrity failures no longer fall through
to another distributor. Those observable corrections plus the additive
exact-SP3 API put these changes in the `0.33` minor line. Python `0.33.1`
aligns with the core source-package compliance patch on that line.

## Adding another distributor

A new distributor supplies transport URL and compression metadata for an
existing `ProductIdentity`. It must not rewrite the identity. Its transport
must apply the same redirect, size, content, decompression, parse, provenance,
and atomic-cache requirements before returning bytes.

The legacy `data.fetch(product)` API is unchanged and continues to use the
cataloged direct archive and legacy cache location. Callers migrate only when
they want explicit distributor selection and returned provenance.
