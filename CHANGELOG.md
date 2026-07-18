# Changelog

All notable changes to the Sidereon Python interface are documented here.

## [Unreleased]

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
