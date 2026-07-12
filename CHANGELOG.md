# Changelog

All notable changes to the Sidereon Python interface are documented here.

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
