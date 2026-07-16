# Changelog

All notable changes to the Sidereon Python interface are documented here.

## [Unreleased]

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
