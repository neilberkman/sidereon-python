# Binding test fixtures

These JSON fixtures carry the fully built RTK / PPP solve inputs plus the
engine's own reference outputs, so the Python binding's pytest can run a real
solve per technique and assert it returns the identical numbers (bit-exact).

They are not hand-authored. They are emitted by the `sidereon-core` validated
integration tests under an env gate, reusing the exact harness those tests use:

    cd ../../../../crates/sidereon-core
    SIDEREON_DUMP_FIXTURES=1 cargo test --test rtk_real_arc \
        wettzell_static_gps_rtk_real_arc_self_validates_batch_paths
    SIDEREON_DUMP_FIXTURES=1 cargo test --test ppp_real_arc \
        esbc_real_float_ppp_arc_improves_with_troposphere_correction
    SIDEREON_DUMP_FIXTURES=1 cargo test --test sgp4_topocentric_arc \
        iss_arc_matches_frozen_bits
    SIDEREON_DUMP_FIXTURES=1 cargo test --test tle_python_fixture \
        iss_round_trip_fixture_self_validates
    SIDEREON_DUMP_FIXTURES=1 cargo test --test frames_time_python_fixture \
        frames_time_reference_self_validates
    SIDEREON_DUMP_FIXTURES=1 cargo test --test sp3_bodies_python_fixture \
        sp3_bodies_reference_self_validates

- `rtk_wtzr.json` -- WTZR/WTZZ static GPS L1 short-baseline arc (120 epochs):
  built epochs, ambiguity ids/scale, measurement model, and the engine's float
  and validated-fixed reference baselines.
- `ppp_esbc.json` -- ESBC troposphere-corrected static float-PPP arc (120
  epochs): built epochs, initial state, config, and the engine's reference
  position. References the committed SP3 product by filename.
- `sgp4_topocentric.json` -- committed ISS TLE propagated over a 10-epoch grid:
  the two TLE lines, a London ground station, epoch unix microseconds, and the
  engine's reference TEME states and topocentric az/el/range (raw f64 plus
  IEEE-754 hex bits). Cross-checks the batched `Tle.propagate` / `look_angles`.
- `tle_roundtrip.json` -- committed ISS TLE: the two lines, the engine's parsed
  element fields, the lines `tle::encode` reproduces, and the advisory
  checksum-warning case (a flipped column-69 digit). Cross-checks `Tle.to_lines`
  and `Tle.checksum_warnings`.
- `frames_time.json` -- four real UTC epochs: the resolved TT/UT1/TDB Julian
  dates + fractions, delta-T, GMST/GAST, mean obliquity, IAU 2000A nutation
  angles + matrix, IAU 2006 precession matrix, and the engine's TEME->GCRS /
  GCRS->ITRS / ITRS->GCRS / geodetic<->ECEF transforms on a shared sample state
  (all IEEE-754 hex bits), plus leap-second/UT1 provenance and a GnssWeekTow
  rollover case. Cross-checks `Instant`, the batched frame transforms,
  `leap_seconds`, `GnssWeekTow`, and the provenance accessors.

- `sp3_bodies.json` -- Area 4 (bodies + SP3): analytic Sun/Moon ECI and ECEF
  vectors (metres) at three real UTC epochs, plus, for the committed
  `IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3` product (read verbatim from the
  crate fixtures, named by relative path), the node J2000-second axis,
  per-satellite interpolated position/clock at shared interior query epochs, the
  exact first G01 record, and the full serialized SP3 text (all IEEE-754 hex bits
  where numeric). Cross-checks `sun_moon_eci`/`sun_moon_ecef`,
  `Sp3.epochs_j2000_seconds`/`interpolate`/`state`/`to_sp3_string`, and the
  path-accepting `load_sp3`.

The SPP test (`test_spp.py`) needs no generated fixture: it reuses the crate's
own `spp_trace_L0_minimal.json` directly.
