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

product = data.mgex_sp3("cod", date(2020, 6, 25))
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

## Sources and caller input

The supported source descriptors are:

- `Distribution.direct()` for the cataloged analysis-center or IGS URL;
- `Distribution.nasa_cddis()` for the official CDDIS HTTPS archive;
- `Distribution.local_file(path, compression="auto")`;
- `Distribution.in_memory(content, compression="auto")`.

SP3 CDDIS paths use the GNSS-products GPS-week directory. Current long-name
IONEX paths use `ionex/<year>/<day-of-year>`. Both retain the exact official
filename and use the distributor's gzip transport packaging.

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
the canonical encoding.
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
invalid content, content-length mismatch, gzip failure, caller checksum,
product parse/semantic validation, and cache read/write failures.

When one distributor was requested, its typed exception is raised directly.
When several were requested and all fail, `AllDistributorsFailed.attempts`
contains each sanitized structured reason.

Retries are bounded and limited to connection/timeouts, 408, 429, and server
errors. NASA publishes no universal numeric CDDIS request limit; callers should
keep concurrency modest, honor service responses, and use a verified local
cache. NASA's public data-use guidance is at
<https://www.earthdata.nasa.gov/engage/open-data-services-software/data-use-policy>.

## Adding another distributor

A new distributor supplies transport URL and compression metadata for an
existing `ProductIdentity`. It must not rewrite the identity. Its transport
must apply the same redirect, size, content, decompression, parse, provenance,
and atomic-cache requirements before returning bytes.

The legacy `data.fetch(product)` API is unchanged and continues to use the
cataloged direct archive and legacy cache location. Callers migrate only when
they want explicit distributor selection and returned provenance.
