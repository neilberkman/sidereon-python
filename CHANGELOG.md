# Changelog

All notable changes to the Sidereon Python interface are documented here.

## [Unreleased]

## [0.35.0] - 2026-07-24

### Added

- Observation QC reports now expose their compact core lint findings through
  `ObservationQcReport.lint_findings` and `ObservationQcFinding`, matching the
  lint information already present in the serialized report.

### Fixed

- Default observation QC now treats a RINEX `INTERVAL` of zero as the
  standards-defined unavailable value, reports `OBS-H19` at informational
  severity, and infers cadence from body epochs when possible. Negative parsed
  values, and non-finite values in programmatically constructed core headers,
  report `OBS-H20` as invalid metadata. Neither kind is used for calculations;
  an unresolved cadence remains explicit when the body cannot supply one.
  Explicit caller interval overrides must still be finite and positive.
- Opt-in interval repair replaces unavailable or invalid source metadata when
  cadence can be inferred and removes it when cadence is unresolved. Default
  repair continues to preserve source `INTERVAL` metadata.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.35.0.

### Compatibility

- This is an additive report surface and a source-metadata compatibility fix.
  Existing Python call signatures and explicit option validation are unchanged.
  Positioning, orbit, and other solver numerical kernels are unaffected.

## [0.34.0] - 2026-07-21

### Added

- Added `data.sp3_content_start_convention`, returning the core-backed
  `Sp3ContentStartConvention` value and its exact offset in seconds. The query
  enforces each center's issue schedule and exposes the historical GFZ
  ultra-rapid transition without duplicating catalog policy in Python.
- Added `data.supported_samples`, a product-, date-, and issue-aware query for
  officially cataloged sampling tokens. Explicit product construction enforces
  the same result before deriving a filename, URL, identity, or cache key.

### Fixed

- Exact SP3 parsing and acquisition now accept a complete `EOF` logical record
  padded with ASCII spaces through the 80-column interoperability limit,
  including LF and CRLF line endings. Malformed EOF-like records, nonblank
  trailing data, and missing terminal records continue to fail closed in the
  shared Rust core. Python acquisition also continues to reject truncated
  compressed products before exact parsing or cache publication.
- Python's bounded gzip decoder now accepts complete RFC 1952 multi-member
  archives, applies one cumulative decompressed-byte cap, and verifies every
  member's end marker, CRC32, and ISIZE. Truncated or corrupt later members and
  non-member trailing bytes remain terminal integrity failures. Exact-product
  acquisition and the legacy fetch path now use this same decoder.
- Built-in exact and legacy HTTP download paths now retain at most the
  compressed-input limit plus one probe byte, even when a transport supplies
  one oversized chunk. Network reads also request bounded chunks explicitly;
  local-file reads retain the same limit-plus-one policy.
- Exact requests derived from historical GFZ ultra-rapid identities now apply
  the core's cataloged filename-epoch/content-start relationship while keeping
  declared-start, header-metadata, first-epoch, cadence, grid, and span checks
  strict.
- Ultra-rapid exact candidates now contain only dated span/cadence variants
  evidenced for the exact center, date, and issue. CODE's moving latest-product
  snapshot is excluded because it is not the dated one-day product; the
  documented GFZ `2021-05-15 0000` dual-cadence overlap remains the only
  two-candidate issue. Caller-built identities must use the cataloged span.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.34.0.

### Compatibility

- The new catalog query is additive; existing Python call signatures and all
  numerical calculations are unchanged. Ultra-SP3 candidate lists can be
  shorter because unsupported alternate spans/cadences and the non-exact CODE
  moving snapshot are no longer returned. Reports or cache entries that claimed
  that snapshot as a dated identity no longer verify. This is a minor release
  because the catalog API is public and identity-derived historical GFZ and
  span validation are newly enforced. Valid concatenated gzip members are newly
  accepted; incomplete or corrupt archives remain rejected. Terminal-record
  behavior is inherited from the same core used by every Sidereon interface.

## [0.33.1] - 2026-07-20

### Added

- Added product-aware `data.product_solution_class` and date-aware
  `data.default_sample_for_date` catalog queries.
- Added `ExactSp3Request`, `ExactSp3Coverage`, `parse_exact_sp3`, and
  `validate_exact_sp3`. Parsed `Sp3` values now expose the epoch count and start
  declared by header line 1.
- Added historical IGS final-orbit identity and CDDIS `.Z` routing, including
  bounded Unix-compress decoding for local, in-memory, and CDDIS acquisition.

### Changed

- Catalog identity and distributor locations now come from the Rust core, so
  IGS final SP3 naming switches at GPS week 2238, GFZ rapid defaults retain the
  historical `15M`/`05M` cadence boundary, and CODE SP3, clock, final IONEX,
  rapid IONEX, and predicted tiers keep product-specific AIUB HTTPS routes.
- Catalog derivation now enforces the verified publication floors for ESA
  final, GFZ rapid, and IGS/ESA/GFZ ultra-rapid SP3. ESA and GFZ ultra-rapid
  defaults follow their historical cadence eras, including ESA's intraday
  transition between the 2025-02-02 0600 and 1200 issues. CDDIS rejects
  pre-week-2238 long-name SP3 and IONEX identities instead of inventing
  archive paths. ESA `ESA0MGNFIN` final SP3 stays on its verified direct archive
  rather than being assigned an unverified CDDIS mapping.
- Exact SP3 acquisition now enforces the full core structural, start, agency,
  cadence, regular-grid, count, span, and format-version contract. Both the
  288-epoch half-open and 289-epoch inclusive representations of a one-day
  five-minute product are accepted.
- Multi-distributor acquisition advances after ordinary publication absence,
  retired endpoints, and exhausted source-local transport availability only.
  Parsing, digest, cadence, span, identity, policy, cache, and caller errors are
  terminal and preserve the first integrity failure.
- Unix-compress acquisition now rejects partial terminal codes and invalid
  terminal padding before product parsing. The wheel and source distribution
  include the corresponding third-party attribution notice and the full
  Apache-2.0 and ISC terms required by compiled Rust dependencies. Release
  artifacts also include the exact SciPy 1.18.0 and ERFA 2.0.1 licenses, the
  full IERS license, and the public 0.33.1 tide sources corresponding to the
  IERS-derived routines in the extension.
- Warm-cache and cold-acquisition caller checksum mismatches now expose the
  same typed integrity failure. Local-file archives are read only through the
  configured compressed-byte bound, and caller/configuration HTTP failures no
  longer authorize another distributor while explicit transient failures do.
- Exact product-set comparison now enforces a caller-declared format version
  while retaining the normal unresolved-request to validated-result workflow.
  The CODE DCB stubs now correctly require the native API's explicit options
  argument (which may be `None`).
- Known unsupported center/product pairs now fail before HTTP instead of being
  reported as an absent merge contributor.
- Updated `sidereon` and `sidereon-core` to 0.33.1 and
  `trust-region-least-squares` to 0.9.2.

### Compatibility

- Existing public call signatures remain available. The new APIs are additive.
  Historical CODE dates before GPS week 2238 and dates before each verified
  ESA/GFZ/IGS ultra-rapid publication floor are now rejected instead of being
  assigned unsupported current long filenames. Callers that depended on
  fallback after corrupt bytes must handle the integrity failure. These
  Exact-set callers that declare a format version must now provide a matching
  resolved version. These observable corrections belong in the 0.33 minor
  line; 0.33.1 also incorporates the core source-package compliance patch.

## [0.32.0] - 2026-07-18

### Added

- Added `parse_navcen_at` and `merge_navcen_at`, plus `NavcenAssessment`, for
  deterministic NAVCEN usability decisions at explicit UTC Unix microseconds.
  Assessments expose the raw NANU notice, Outage Start cell, evaluation instant,
  parsed interval, and explicit unparseable/not-applicable timing state.

### Compatibility

- Existing `parse_navcen` and `merge_navcen` behavior is unchanged. Operational
  callers should use the explicit-time API so a future or completed forecast is
  not treated as a current outage. The time-aware path additionally recognizes
  active `UNUSUFN` notices as immediately unusable; the legacy parser's
  pre-existing behavior remains unchanged.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.32.0.

## [0.31.2] - 2026-07-16

### Fixed

- Alias acquisition now proves the parsed SP3 epoch grid and complete coverage
  duration match the exact catalog span before publishing artifact provenance.
- Persisted merged-SP3 verification now enforces the complete versioned schema
  without coercions, validates every nested record, checks catalog-facing
  contributor fields, and requires contributors and absent centers to exactly
  partition the requested center set.
- The merged-SP3 identity API now returns the canonical contributor order and
  the distinct ordered precedence contributors alongside the stable ID, while
  preserving two-value unpacking compatibility.
- Merge-policy validation now matches the core's executable domain, including
  whole-second target spacing and canonical equivalence of negative zero.
- Updated `sidereon` and `sidereon-core` to 0.31.2.

## [0.31.0] - 2026-07-16

### Added

- Merged-SP3 reports now retain each contributor's exact artifact identity and
  separate observational acquisition facts. Reports carry the core-backed,
  versioned `stable_input_identity`; `fetch_merged_sp3_file(...,
  return_report=True)` preserves the same report while writing the product.
- Added `data.sp3_merge_input_identity` for deterministic identity of a
  complete verified artifact set and merge policy. Mean and median contributor
  order is canonicalized; precedence order remains identity-bearing priority.
- Added `data.verify_merge_report` to restore and core-verify the exact
  artifacts, effective policy, precedence order, and stable identity from a
  persisted `MergeReport.to_dict()` value.

### Changed

- Merged-SP3 acquisition now requires complete exact-cache provenance for every
  contributor. A legacy path plus the older digest-only sidecar is no longer an
  acceptable merged input; acquire it once through the exact path to populate
  the transactional cache and the parsed SP3 format revision.
- SP3 merge position and clock tolerances now accept zero, matching the shared
  core's finite, non-negative policy contract.
- Updated `sidereon` and `sidereon-core` to 0.31.0.

## [0.30.0] - 2026-07-16

### Fixed

- Publishes exact-product cache entries as immutable payload/archive/provenance
  transactions selected by one atomic digest-bound commit record. Cache hits
  cannot observe a mixed three-file update after concurrent processes or a
  process death at a write boundary.
- Delegates cache-first acquisition to the shared Rust transaction
  implementation. Its bounded advisory lock coordinates Linux and macOS
  processes; dead owners release automatically, duplicate downloads are
  avoided, and abandoned transactions are removed only while the entry lock is
  held.
- Revalidates and atomically migrates valid 0.29.0-0.29.2 cache triples without
  downloading them again. Cache lock/write failures are terminal and never
  authorize distributor substitution.

### Added

- Added the optional `cache_lock_timeout_s` acquisition argument, defaulting to
  30 seconds.
- Added the public `sidereon.exact_cache` module for native locked publication,
  locked and unlocked verified reads, and abandoned-entry cleanup.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.30.0. Full identity hashing now
  uses the same golden canonical key in all five interfaces.

## [0.29.2]

### Added

- Added `validate_exact_product_set` and structured
  `ExactProductSetError` diagnostics. Multi-product workflows can now require
  the complete declared identity inventory before dependent processing starts;
  empty declarations, duplicates, missing products, and undeclared products
  fail closed.
- Exact-set comparison preserves prediction-tier identity even when official
  filenames match. SP3 observed/predicted timing remains sourced from
  `Sp3.prediction_summary()` record flags rather than inferred metadata.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.29.2.

## [0.29.1]

### Fixed

- Fetches CODE predicted IONEX P1 and P2 products from their current official
  tier-specific HTTPS directories, retaining the requested identity year and
  exact filename across validated AIUB redirects.
- Routes the legacy IONEX helper through exact acquisition so downloaded and
  cached bytes receive the same date, issue, and cadence validation. Explicit
  legacy lookback continues only after typed not-published or offline-miss
  results; validation and transport failures remain terminal.
- Keeps P1 and P2 cache identities isolated even when their filenames match.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.29.1.

## [0.29.0]

### Added

- Added an exact GNSS acquisition API that separates product identity from its
  ordered, caller-selected distributors: direct archives, NASA CDDIS/Earthdata,
  local files, and in-memory bytes.
- Added caller-supplied Earthdata bearer-token and netrc authentication,
  structured source failures, secret-free provenance, validated source-specific
  caches, original archive retention, and parsed SP3/IONEX semantic checks.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.29.0.

## [0.28.1]

### Fixed

- Updated CODE ultra-rapid SP3 retrieval to AIUB's official HTTPS endpoint,
  including narrowly validated redirects to AIUB's public object store.
- Candidate-URL 404 results now retain the attempted URL, HTTP status,
  filename, center, and candidate pattern without claiming authoritative
  publication status. Access and transport failures remain typed errors.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.28.1. Sequential RTK updates
  inherit the core's exact information-matrix symmetry enforcement when
  process noise is enabled; the zero-process-noise path and Python API are
  unchanged.

## [0.28.0]

### Added

- Added per-cell SP3 precedence, optional deterministic outlier rejection,
  clock-outlier provenance, and observed/predicted epoch summaries.
- Added current/alternate ultra-rapid product probing and complete merge-policy
  forwarding through `data.fetch_merged_sp3`.

### Fixed

- Fixed sibling Rust fixture discovery in the standard multi-repository
  development layout.

## [0.27.1]

### Fixed

- Updated `sidereon` and `sidereon-core` to 0.27.1 so LAMBDA integer
  ambiguity searches reject finite values outside the `i64` result domain
  instead of returning saturated integers with non-finite scores.

## [0.27.0]

### Added

- Added `GeoidGrid.from_proj_egm96_gtx` and
  `GeoidGrid.undulation_proj_rad` for PROJ 9.3.0-compatible interpolation of
  the public EGM96 15-arcminute GTX grid.
- Added the required `ProjVgridshiftArithmetic` policy so callers explicitly
  select fused or separately rounded multiply-add evaluation to match their
  reference PROJ build.
- Added typed `ProjVgridshiftError` subclasses for non-finite and outside-grid
  lookup coordinates.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.27.0.

## [0.26.1]

### Security

- Updated `sidereon` and `sidereon-core` to 0.26.1, which rejects RINEX 2
  observation epoch headers that declare an oversized satellite count before
  processing continuation records. Malicious input could otherwise request an
  enormous allocation and terminate the Python process. Sidereon Python
  versions 0.11.1 through 0.26.0 are affected; upgrade to 0.26.1.

## [0.26.0]

### Breaking

- Removed the unsound sequential-RTK innovation-screen surface in step with
  `sidereon-core` 0.26.0: the `innovation_threshold_sigma` and
  `innovation_min_rows` arguments on `RtkArcUpdateOptions`, the
  `RtkArcInnovationScreen` class, and the `RtkArcEpochSolution.innovation_screen`
  property.

### Changed

- Updated the `sidereon` and `sidereon-core` engine dependencies to 0.26.0.

### Fixed

- Near-polar ionospheric pierce-point calculations now inherit the core 0.26.0
  finiteness correction.
