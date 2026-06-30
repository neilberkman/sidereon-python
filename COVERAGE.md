# Python binding coverage map (`sidereon-core` -> `sidereon` Python package)

Tracks how much of the `sidereon-core` public API is reachable from the Python
binding (`bindings/python`, PyO3), and whether each exposed capability meets the
idiomatic bar in `PYTHON_IMPROVE_BRIEF.md` (real exception hierarchy, numpy
in/out, typed config objects with `.pyi`, `@property`/`__repr__`/`__eq__`,
`enum.Enum` status fields, no `dict[str, Any]` blobs).

The binding is a thin INTERFACE: marshal + idiom only, ZERO modeling logic. Every
exposed item must reproduce the engine numbers exactly and is cross-checked in
pytest against an env-gated fixture dumped from the validated Rust harness.

## Status legend

- `exposed-idiomatic` - reachable from Python and meets the full bar above.
- `not-idiomatic` - reachable but violates the bar (dict blobs, no typed inputs,
  collapses every failure into one `SidereonError`, no enums, etc.).
- `not-exposed` - present in the Rust public API, unreachable from Python.
- `deferred` - intentionally not exposed; reason stated.

Source of truth for the API surface: `grep '^pub'` over
`crates/sidereon-core/src/**` on branch `sidereon/rename`, verified
2026-06-17. Current Python surface: `python/sidereon/__init__.py(.pyi)` plus
`src/{lib,spp,rtk,ppp,propagation,frames,bodies,ephemeris}.rs`.

---

## Cross-cutting (applies to every area)

| Capability                                                                                               | Status            | Notes                                                                                                                                                                                                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Exception hierarchy (`SidereonError` base + `Sp3ParseError`, `TleParseError`, `SolveError`, convergence) | exposed-idiomatic | Tree built: `SidereonError` base; `ParseError` -> {`Sp3ParseError`, `TleParseError`}; `SolveError`. `load_sp3` raises `Sp3ParseError`, `Tle(..)` raises `TleParseError`, SGP4/numerical propagation raise `SolveError`, and GNSS solver failures raise `SolveError`. Bad input still raises `TypeError`/`ValueError`. |
| numpy in/out for batched calls                                                                           | partial           | Propagation/look-angles/frame transforms, Sun/Moon, and SP3 interpolation return numpy `(n,3)`/`(n,)`. SPP/RTK/PPP return scalar/list/dict, take Python sequences.                                                                                                                                                                                        |
| Typed `#[pyclass]` config objects (no `dict[str, Any]`)                                                  | partial           | Area 9 GNSS solvers now use typed pyclasses (`SppConfig`, `Rtk*Config`, `Ppp*Config`, typed epoch/observation/state records) instead of dict blobs. Remaining future areas must follow the same rule as they are exposed.                                                                                                                                  |
| `enum.Enum` for status/discrete fields                                                                   | partial           | `TimeScale` exposed as a typed PyO3 enum (Area 3). RTK/PPP `integer_status` now returns `IntegerStatus`, not a raw string. Remaining future discrete fields must use enums as they are exposed.                                                                                                                                                            |
| `os.PathLike` file-load helpers                                                                          | exposed-idiomatic | `load_sp3` accepts `bytes`/`bytearray` OR a path (`str`/`os.PathLike`); a missing path raises `OSError`, non-path/non-bytes raises `ValueError` (Area 4).                                                                                                                                                                                                 |

---

## Area 1 - SGP4 + TLE (astro)

Core: `astro::sgp4` (`Satellite`, `from_tle*`, `propagate`/`propagate_jd`,
`propagate_elements*`, `ElementSet`, `OpsMode`, `GravConstType`, raw
`sgp4`/`sgp4init`), `astro::tle` (`parse`, `encode`, `TleElements`,
`ParsedTle`, `TleError`, `ChecksumWarning`).

| Capability                                          | Status            | Notes                                                                                        |
| --------------------------------------------------- | ----------------- | -------------------------------------------------------------------------------------------- |
| Parse TLE (two lines) -> elements                   | exposed-idiomatic | `Tle(line1, line2, opsmode=...)`; element fields as `@property`; `__repr__`.                 |
| SGP4 propagate over epoch grid (TEME)               | exposed-idiomatic | `Tle.propagate(epochs_unix_us)` -> `TlePropagation` numpy `(n,3)` pos/vel km.                |
| Topocentric look angles (az/el/range)               | exposed-idiomatic | `Tle.look_angles(station, epochs)` -> `LookAngles` numpy. (Also covers Area 5 look-angle.)   |
| Format/encode elements -> TLE lines (`tle::encode`) | exposed-idiomatic | `Tle.to_lines()` -> `(str, str)`; character-exact round-trip cross-checked vs `tle::encode`. |
| Checksum warnings (`ChecksumWarning`) surfaced      | exposed-idiomatic | `Tle.checksum_warnings` -> `list[ChecksumWarning]` (typed, `@property`/`__repr__`/`__eq__`). |
| `TleParseError` vs generic error                    | exposed-idiomatic | `Tle(..)` raises `TleParseError` (a `ParseError`/`SidereonError`).                           |
| Raw `ElementSet` / `sgp4init` kernels               | deferred          | Low-level `pub` kernels; not the intended user API. Idiomatic surface is `Tle`.              |

## Area 2 - Numerical propagation (astro)

Core: `astro::forces` (`TwoBodyGravity`, `J2Gravity`, `CompositeForceModel`,
`ForceModel`), `astro::integrators` (`RK4`, `DP54`, `Integrator`,
`DynamicsModel`), `astro::propagator` (`PropagationContext`,
`PropagationResult`, `PropagationPoint`, `PropagationStats`,
`OrbitalDynamics`, `IntegratorOptions`, `DenseOutput`, `DenseSegment`,
`PIController`), `astro::state` (`CartesianState`, `StateDerivative`).

| Capability                                  | Status            | Notes                                                                                                                                                                       |
| ------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cartesian state value type                  | exposed-idiomatic | Crossed as numpy `position_km` / `velocity_km_s` (length 3) into `propagate_state` and back out via `Ephemeris`.                                                            |
| Force models (two-body, J2, composite)      | exposed-idiomatic | `propagate_state(force_model="two_body" \| "two_body_j2", mu_km3_s2=...)`; backed by core `ForceModelKind`.                                                                 |
| Integrators (RK4 fixed-step, DP54 adaptive) | exposed-idiomatic | `propagate_state(integrator="dp54" \| "rk4", abs_tol=, rel_tol=, initial_step_s=, min_step_s=, max_step_s=, max_steps=)`.                                                   |
| Propagate a state over a time grid          | exposed-idiomatic | `propagate_state(epoch_s, position_km, velocity_km_s, times_s, ...)` -> `Ephemeris` numpy `times_s` `(n,)`, pos/vel `(n,3)`, `states` `(n,6)`. Backed by `StatePropagator`. |
| Dense output / interpolation                | not-exposed       | `DenseOutput.eval(t)`; the grid-sampling `Ephemeris` path covers the common need.                                                                                           |

## Area 3 - Frames + time (astro)

Core: `astro::frames` (`gcrs_to_itrs_*`, `itrs_to_gcrs_*`, `teme_to_gcrs_*`,
`mean_of_date_to_itrs_matrix`, geodetic<->ITRS, precession/nutation/GAST
helpers, `mat3_vec3_mul`, `TemeStateKm`, `GeodeticStationKm`), `astro::time`
(`Instant`, `Time`, `TimeScale`, `TimeScales`, `Duration`, `JulianDateSplit`,
`GnssWeekTow`, leap seconds, UT1 coverage/provenance, `Validated<T>`).

| Capability                                     | Status            | Notes                                                                                                                                                                                                                                                               |
| ---------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ITRS/GCRS/TEME transforms (matrices + apply)   | exposed-idiomatic | Batched numpy `(n,3)`: `teme_to_gcrs` -> `FrameStates`, `gcrs_to_itrs`, `itrs_to_gcrs`; `skyfield_compat` flag. Per-row `TimeScales` from the parity path. Cross-checked bit-exact (both compat modes).                                                             |
| Geodetic <-> ECEF/ITRS                         | exposed-idiomatic | `geodetic_to_ecef` / `ecef_to_geodetic`, batched `(n,3)`. PROJ-parity variant `geodetic_from_ecef_proj` deferred (see below).                                                                                                                                       |
| Precession/nutation/GAST/obliquity             | exposed-idiomatic | `Instant.precession_matrix()`/`nutation_matrix()` (numpy `(3,3)`), `nutation_angles()`, `mean_obliquity_radians`, `gmst_radians()`/`gast_radians()`. GMST/GAST added as core public wrappers; cross-checked bit-exact.                                              |
| Time scales (TT/TAI/UT1/UTC/GPS) + conversions | exposed-idiomatic | `Instant.from_utc`/`from_unix_micros`; `tt_jd`/`ut1_jd`/`tdb_jd` + fractions + `delta_t_seconds`; `TimeScale` enum. Cross-checked bit-exact.                                                                                                                        |
| Julian date / GNSS week-TOW / leap seconds     | exposed-idiomatic | `JulianDate` (split), `Instant.{tt,ut1,tdb}_jd_split`; `GnssWeekTow` (`normalized`/`unrolled_week`); `leap_seconds(y,m,d)` + `leap_seconds_batch`, `leap_second_table_info()`, `ut1_coverage_info()`. Cross-checked bit-exact.                                      |
| PROJ-parity ECEF->geodetic variant             | deferred          | `geodetic_from_ecef_proj` returns `[lon_deg, lat_deg, alt_m]` (pyproj convention, distinct column order/units) for pyproj 3.6.1 exactness. The primary `ecef_to_geodetic` (Skyfield WGS84, lat/lon/alt km) is exposed; revisit if pyproj-exact ECEF->LLA is needed. |

## Area 4 - Ephemeris / bodies (astro + GNSS)

Core: `astro::bodies` (`sun_moon_eci`, `sun_moon_ecef`, `SunMoon`),
`ephemeris`/`sp3` (`Sp3::parse`, `epoch_count`, `satellites`, position/clock
interpolation, `sp3::write`, `sp3::combine`), broadcast (`rinex_nav`
`BroadcastStore`/`BroadcastRecord`).

| Capability                                | Status            | Notes                                                                                                                                                                                                                                                                        |
| ----------------------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sun/Moon position (ECI, ECEF)             | exposed-idiomatic | `sun_moon_eci(epochs_unix_us)` / `sun_moon_ecef(...)` -> `SunMoon` with `sun`/`moon` numpy `(n,3)` metres; batched (one FFI crossing). Backed by core `sun_moon_eci_at`/`sun_moon_ecef`; cross-checked bit-exact.                                                            |
| Parse SP3 product                         | exposed-idiomatic | `load_sp3(bytes)` -> `Sp3`; `epoch_count`, `satellites`, `__repr__`; raises `Sp3ParseError`.                                                                                                                                                                                 |
| SP3 position/clock query at epoch         | exposed-idiomatic | `Sp3.epochs_j2000_seconds` (node axis), `Sp3.interpolate(sat, j2000_seconds)` -> `Sp3Interpolation` (`position_m` `(n,3)`, `clock_s` `(n,)` NaN-where-missing), `Sp3.state(sat, idx)` -> `Sp3State`. Backed by `position_at_j2000_seconds`/`state`; cross-checked bit-exact. |
| SP3 path-accepting loader (`os.PathLike`) | exposed-idiomatic | `load_sp3` accepts `bytes`/`bytearray` or a path (`str`/`os.PathLike`); missing path -> `OSError`.                                                                                                                                                                           |
| SP3 write                                 | exposed-idiomatic | `Sp3.to_sp3_string()` -> deterministic SP3 text; byte-exact vs `to_sp3_string` and round-trips via `load_sp3`.                                                                                                                                                               |
| SP3 combine (multi-product)               | not-exposed       | `sp3::merge` + `MergeOptions`/`MergeReport`; scheduled as the next Area 4 step (typed merge config + audit-trail result).                                                                                                                                                    |
| Broadcast ephemeris access                | not-exposed       | `ephemeris::BroadcastEphemeris` (`rinex_nav::BroadcastStore`); bound together with RINEX-nav parsing in Area 10.                                                                                                                                                             |

## Area 5 - Geometry + events (astro)

Core: `astro::passes` (`predict_passes`, `visible_from_constellation`,
`PassPredictionOptions`, `PredictedPass`, `ConstellationMember`,
`VisibleSatellite`), `astro::events` (`status`, `shadow_fraction`,
`EclipseStatus`, `DetectedEvent`), `astro::angles` (`sun_angle`, `phase_angle`,
`sun_elevation`, `moon_angle`, `earth_angular_radius`), GNSS `geometry`
(`dop_series`, `visibility_series`, `passes`, DOP/visibility types).

| Capability                                | Status            | Notes                                                                                                                                                                                 |
| ----------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Look angle az/el/range (single + arc)     | exposed-idiomatic | Via `Tle.look_angles` (Area 1).                                                                                                                                                       |
| Pass finding (AOS/LOS/culmination + mask) | exposed-idiomatic | `Tle.find_passes(station, start_unix_us, end_unix_us, elevation_mask_deg=, step_seconds=, time_tolerance_s=)` -> `list[SatellitePass]`. Dense-sample finder; backed by `find_passes`. |
| Pass prediction (rise/set windows)        | not-exposed       | Legacy `predict_passes` / `PredictedPass` / `PassPredictionOptions`; the `find_passes` path above supersedes it for the binding.                                                      |
| Constellation visibility                  | not-exposed       | `visible_from_constellation`, GNSS `visibility_series`.                                                                                                                               |
| Eclipse / shadow status + fraction        | exposed-idiomatic | `shadow_fraction` -> numpy `(n,)`; `eclipse_status` -> `list[EclipseStatus]` enum over batched `(n,3)` satellite/Sun km vectors. Cross-checked bit-exact against Rust fixture. |
| Sun/phase/moon angles                     | exposed-idiomatic | `sun_angle`, `moon_angle`, `sun_elevation`, `phase_angle`, `earth_angular_radius` over batched `(n,3)` km vectors, returning numpy `(n,)` degrees. Cross-checked bit-exact. |
| GNSS DOP from line-of-sight geometry      | exposed-idiomatic | `gnss_dop(line_of_sight, weights, Wgs84Geodetic)` -> typed `Dop` (`gdop/pdop/hdop/vdop/tdop`, `__repr__`, `__eq__`); bad geometry uses `ValueError`/`SolveError`. Cross-checked bit-exact. |
| GNSS DOP series / visibility planning     | not-exposed       | SP3-backed `geometry::dop_series`, `visibility_series`, `passes`, `DopWeighting`; outside this branch's assigned direct-DOP scope. |

## Area 6 - Conjunction (astro)

Core: `astro::conjunction` (`collision_probability`, `encounter_frame`,
`encounter_plane_covariance`, `CollisionPc`, `ConjunctionState`,
`EncounterFrame`, `PcMethod`), `astro::covariance` (`rtn_to_eci`, `symmetric`,
`positive_semidefinite`, `RtnFrameError`).

| Capability                                  | Status      | Notes                                                    |
| ------------------------------------------- | ----------- | -------------------------------------------------------- |
| Close-approach / collision probability (Pc) | exposed-idiomatic | `ConjunctionState`, `PcMethod`, `collision_probability` -> `CollisionProbability`; numpy states/covariances, typed enum, `SolveError` for undefined relative motion. |
| Encounter frame + B-plane covariance        | exposed-idiomatic | `encounter_frame` -> `EncounterFrame`; `encounter_plane_covariance` returns numpy `(2, 2)` B-plane covariance. Cross-checked against core fixture. |
| RTN-frame covariance transform              | exposed-idiomatic | `rtn_to_eci_covariance`, `covariance_is_symmetric`, `covariance_is_positive_semidefinite`; numpy `(3, 3)` in/out, core validation errors mapped to `ValueError`. |

## Area 7 - CCSDS CDM (astro)

Core: `astro::cdm` (`parse_kvn`, `parse_xml`, `encode_kvn`, `encode_xml`,
`CdmKvn`, `CdmObject`, `CdmError`).

| Capability           | Status      | Notes                                                   |
| -------------------- | ----------- | ------------------------------------------------------- |
| Parse CDM (KVN, XML) | exposed-idiomatic | `parse_cdm_kvn` / `parse_cdm_xml` -> `Cdm`; `CdmParseError` under `ParseError`; typed `CdmObject` with numpy state/covariance properties. |
| Write CDM (KVN, XML) | exposed-idiomatic | `Cdm.to_kvn_string()` / `Cdm.to_xml_string()` delegate to core `encode_kvn` / `encode_xml`; constructor supports typed value creation. |

## Area 7b - CCSDS OMM (astro)

Core: `astro::omm` (`parse_kvn`, `parse_xml`, `parse_json`, `encode_kvn`,
`encode_xml`, `encode_json`, `Omm`, `OmmEpoch`, `OmmError`).

| Capability                 | Status            | Notes                                                                                                                                          |
| -------------------------- | ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Parse OMM (KVN, XML, JSON) | exposed-idiomatic | `parse_omm_kvn` / `parse_omm_xml` / `parse_omm_json` -> `Omm`; `OmmParseError` under `ParseError`; typed `OmmEpoch` value.                    |
| Write OMM (KVN, XML, JSON) | exposed-idiomatic | `Omm.to_kvn_string()` / `to_xml_string()` / `to_json_string()` delegate to core encoders; constructor supports typed value creation.           |

## Area 8 - RF link budget (astro)

Core: `astro::rf` (`fspl`, `eirp`, `cn0`, `dish_gain`, `link_margin`,
`wavelength`, `LinkBudget`).

| Capability                                                | Status      | Notes                                        |
| --------------------------------------------------------- | ----------- | -------------------------------------------- |
| Link budget helpers (FSPL, EIRP, C/N0, dish gain, margin) | exposed-idiomatic | `fspl`, `eirp`, `cn0`, `wavelength`, `dish_gain`, `link_margin(LinkBudget)`; scalar floats plus typed `LinkBudget` (`@property`, `__repr__`, `__eq__`). Cross-checked bit-exact against Rust fixture. |

## Area 9 - GNSS positioning solvers (already exposed; bring to bar)

Core: `positioning` (`solve_with_policy`, `SolveInputs`, `ReceiverSolution`,
`SolvePolicy`), `rtk_filter` (`solve_float_baseline`,
`solve_fixed_baseline_validated`, `MeasModel`, `Epoch`, opts/result structs),
`precise_positioning` (`solve_float_epochs`, `solve_fixed_from_float`,
`FloatEpoch`/`FloatState`/`*Config`, solutions).

| Capability                    | Status        | Notes                                                                                                                               |
| ----------------------------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| SPP single-point solve        | exposed-idiomatic | `solve_spp(sp3, SppConfig)` uses typed `SppObservation`, correction, Klobuchar, and meteorology pyclasses; failures map to `SolveError`/`ValueError`; pytest reuses the Rust SPP trace fixture. |
| RTK float baseline            | exposed-idiomatic | `solve_rtk_float(RtkFloatConfig)` uses typed epoch/satellite records, measurement model, and float options; pytest cross-checks the Rust-dumped WTZR fixture bit-exact.                       |
| RTK fixed baseline            | exposed-idiomatic | `solve_rtk_fixed(RtkFixedConfig)` uses typed config/options and returns `IntegerStatus`; pytest cross-checks the Rust-dumped WTZR fixed fixture bit-exact.                                      |
| PPP float arc                 | exposed-idiomatic | `solve_ppp_float(sp3, epochs, PppFloatState, PppFloatConfig)` uses typed epoch/observation/state/config pyclasses; pytest cross-checks the Rust-dumped ESBC fixture bit-exact.                  |
| PPP fixed (search + re-solve) | exposed-idiomatic | `solve_ppp_fixed(sp3, epochs, PppFloatSolution, PppFixedConfig)` exposes the ergonomic crate's fixed solver; the ESBC fixture emitter dumps fixed config/result metadata for pytest.          |

## Area 10 - GNSS data parsing + products (not exposed)

Core modules: `rinex` (nav/obs parse, CRINEX/Hatanaka decode, clock,
iono corrections), `antex` (PCO/PCV), `frequencies` (carrier table),
`navigation` (LNAV codecs), `combinations` (iono-free), `observables`
(forward predict), `velocity` (Doppler/range-rate solve), `dgnss`,
`quality` (RAIM/FDE/weighting), `signal` (C/A code, correlation, acquisition),
`carrier_phase` (geometry-free, Melbourne-Wubbena, cycle slip, smoothing),
`ppp_corrections`, `broadcast_comparison` (SISRE), `orbit` (reduced orbit),
`tides`, `terrain` (DTED), `atmosphere` (IONEX/Klobuchar).

| Capability                                    | Status      | Notes                                                                              |
| --------------------------------------------- | ----------- | ---------------------------------------------------------------------------------- |
| RINEX navigation parse                        | not-exposed | `rinex::parse_nav`, `BroadcastStore`.                                              |
| RINEX observation parse + CRINEX decode       | not-exposed | `rinex` obs, Hatanaka.                                                             |
| RINEX clock parse/interp                      | not-exposed | `rinex_clock` (private; via `rinex`).                                              |
| ANTEX PCO/PCV                                 | not-exposed | `antex::Antex`, `pco`/`pcv`.                                                       |
| Carrier-frequency table                       | not-exposed | `frequencies::CarrierBand`, `wavelength_m`.                                        |
| LNAV navigation-message codec                 | not-exposed | `navigation::{decode,encode,parity}`.                                              |
| Observable linear combinations (iono-free)    | not-exposed | `combinations::ionosphere_free*`.                                                  |
| Forward observable prediction                 | not-exposed | `observables::predict`.                                                            |
| Receiver velocity / clock-drift solve         | not-exposed | `velocity::solve`, Doppler<->range-rate.                                           |
| DGNSS code-differential corrections           | not-exposed | `dgnss`.                                                                           |
| Quality: RAIM / FDE / weighting               | not-exposed | `quality::{raim,fde,weight_vector,validate_receiver_solution}`.                    |
| Signal: C/A code, correlation, acquisition    | not-exposed | `signal::{ca_code,correlate,acquire}`.                                             |
| Carrier-phase combos + cycle-slip + smoothing | not-exposed | `carrier_phase::{geometry_free,melbourne_wubbena,detect_cycle_slips,smooth_code}`. |
| PPP correction tables                         | not-exposed | `ppp_corrections::build`.                                                          |
| Broadcast-vs-precise SISRE                    | not-exposed | `broadcast_comparison::compare`.                                                   |
| Reduced-orbit fit/eval                        | not-exposed | `orbit::ReducedOrbitModel`.                                                        |
| Solid-earth tides                             | not-exposed | `tides::solid_earth_tide`.                                                         |
| DTED terrain lookup                           | not-exposed | `terrain::DtedTerrain`.                                                            |
| Atmosphere (IONEX/Klobuchar)                  | not-exposed | `atmosphere` (used internally by SPP).                                             |

## Area 11 - Estimation substrate / low-level kernels

Core: `estimation` (recipe/strategy substrate), `ils` (integer least squares),
`astro::math` (`vec3`/`mat3`/`linear`/`least_squares`/`polynomial`/`special`),
`astro::constants`, `astro::tolerances`, `constants`/`tolerances`.

| Capability                                                    | Status   | Notes                                                                                                                            |
| ------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Estimation recipe substrate (`estimate`, recipes, strategies) | deferred | Internal operation-order substrate behind the public solvers; not a user-facing API. Re-evaluate if a user needs custom recipes. |
| ILS ambiguity-search kernels                                  | deferred | Low-level kernel consumed by RTK/PPP; the user-facing path is the fixed solvers.                                                 |
| math/constants/tolerances                                     | deferred | Engine internals; constants can be surfaced as module-level Python constants if a need appears.                                  |

---

## Iteration order (proposed; one area per iteration)

1. [DONE] Exception hierarchy (cross-cutting) + brought SGP4/TLE area fully onto
   it (`TleParseError`, `Sp3ParseError`, `SolveError`), added `tle::encode`
   round-trip (`Tle.to_lines`) and `Tle.checksum_warnings`.
2. [DONE] Frames + time (Area 3): scale-tagged `Instant` (TT/UT1/TDB + sidereal
   time + precession/nutation), batched `teme_to_gcrs`/`gcrs_to_itrs`/
   `itrs_to_gcrs` + geodetic<->ECEF, `TimeScale`/`JulianDate`/`GnssWeekTow`,
   leap seconds + UT1/leap-second provenance. Added core public GMST/GAST
   wrappers and made `UtcInstant::time_scales` public (additive). The PROJ
   ECEF->geodetic variant is deferred (distinct convention, stated reason).
3. Numerical propagation (Area 2): already exposed-idiomatic (`propagate_state`)
   from a prior increment; only the `DenseOutput` interpolation row remains
   not-exposed. Revisit that row in a later pass.
4. [DONE, partial] Bodies + SP3 (Area 4): Sun/Moon ECI+ECEF batched numpy,
   SP3 `epochs_j2000_seconds`/`interpolate`/`state`, path-accepting `load_sp3`,
   and `to_sp3_string`. Added core `sun_moon_eci_at` + `Sp3::epochs_j2000_seconds`
   (additive). Remaining Area 4 rows: SP3 combine (`merge`) - next step; broadcast
   ephemeris - folded into Area 10 (RINEX nav).
5. Geometry + events (Area 5): pass prediction, eclipse, angles, DOP.
6. Conjunction (Area 6).
7. CCSDS CDM (Area 7).
8. RF link budget (Area 8).
9. [DONE] GNSS solvers to the bar (Area 9): typed config objects replacing
   dict blobs, `solve_ppp_fixed`, status enums.
10. GNSS data parsing + products (Area 10), split across iterations as needed.
11. Late iteration: calibrated propagation and topocentric benchmark on the propagation /
6. [DONE] Conjunction (Area 6): Pc methods, encounter frame + B-plane
   covariance, RTN->ECI covariance, symmetry/PSD checks. Python pytest
   cross-checks the core env-gated fixture bit-for-bit.
7. [DONE] CCSDS CDM (Area 7): KVN/XML parse and encode with typed `Cdm` /
   `CdmObject`, CDM parse exception, and pytest cross-check against a core
   env-gated fixture.
8. [DONE] CCSDS OMM (Area 7b): KVN/XML/JSON parse and encode with typed
   `Omm` / `OmmEpoch`, OMM parse exception, and pytest cross-check against a
   core env-gated fixture.
9. RF link budget (Area 8).
10. GNSS solvers to the bar (Area 9): typed config objects replacing dict blobs,
   `solve_ppp_fixed`, status enums.
11. GNSS data parsing + products (Area 10), split across iterations as needed.
12. Late iteration: calibrated propagation and topocentric benchmark on the propagation /
    topocentric overlap (`bench/`), real measured numbers, recorded measured numbers.

When every row above is `exposed-idiomatic` or `deferred` with a stated reason
and the benchmark is committed: append "PYTHON IMPROVE DONE" to PROGRESS.md and
touch the completion sentinel.
