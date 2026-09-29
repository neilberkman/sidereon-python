# Binding test fixtures

These JSON fixtures carry the fully built RTK / PPP solve inputs plus the
engine's own reference outputs, so the Python binding's pytest can run a real
solve per technique and assert it returns the identical numbers (bit-exact).

They are not hand-authored. Every generated fixture names the exact core
revision that produced it in `core_revision`. `tests/test_core_revision.py`
checks the Python extension and both fixture generators against that value. For
a Git candidate, it verifies the canonical repository and full commit in each
manifest and resolved lockfile. For a crates.io build, it verifies the locked
package checksums against the downloaded crate archives, reads the published
VCS commit from those checksum-matched archives, and requires the facade and
core packages to name the same source commit. The generators perform the same
source check before writing fixture provenance. When the binding moves to
another core, update all three dependency graphs, then regenerate the fixtures
from that resolved source.

The core fixtures are emitted by the `sidereon-core` integration tests under an
environment gate, reusing the harness those tests use. In a checkout of
`https://github.com/neilberkman/sidereon` at the pinned revision, from
`crates/sidereon-core`:

    SIDEREON_DUMP_FIXTURES=1 cargo test --test rtk_real_arc \
        wettzell_static_gps_rtk_real_arc_self_validates_batch_paths
    SIDEREON_DUMP_FIXTURES=1 cargo test --test ppp_real_arc \
        esbc_real_float_ppp_arc_improves_with_troposphere_correction
    SIDEREON_DUMP_FIXTURES=1 cargo test --test cdm_python_fixture
    SIDEREON_DUMP_FIXTURES=1 cargo test --test omm_python_fixture
    SIDEREON_DUMP_FIXTURES=1 cargo test --test pass_finder_arc
    SIDEREON_DUMP_FIXTURES=1 cargo test --test sgp4_topocentric_arc
    SIDEREON_DUMP_FIXTURES=1 cargo test --test tle_python_fixture \
        iss_round_trip_fixture_self_validates
    SIDEREON_DUMP_FIXTURES=1 cargo test --test frames_time_python_fixture \
        frames_time_reference_self_validates
    SIDEREON_DUMP_FIXTURES=1 cargo test --test sp3_bodies_python_fixture \
        sp3_bodies_reference_self_validates
    SIDEREON_DUMP_FIXTURES=1 cargo test --test conjunction_python_fixture
    SIDEREON_DUMP_FIXTURES=1 cargo test --test events_bodies_dop_python_fixture
    SIDEREON_DUMP_FIXTURES=1 cargo test --test rf_python_fixture
    SIDEREON_DUMP_FIXTURES=1 cargo test --test numerical_propagation \
        reference_arc_is_deterministic_and_frozen

Each test writes its fixture to `bindings/python/tests/fixtures/` of that
checkout; copy it into this directory. `rtk_wtzr.json` and the other dumps come
from the core's own tests and carry no `core_revision`; the revision they come
from is the pinned one above.

- `rtk_wtzr.json` -- WTZR/WTZZ static GPS L1 short-baseline arc (120 epochs):
  built epochs, ambiguity ids/scale, measurement model, and the engine's float
  and validated-fixed reference baselines.
- `ppp_esbc.json` -- ESBC troposphere-corrected static float-PPP arc (120
  epochs): built epochs, initial state, config, and the engine's reference
  position. References the committed SP3 product by filename. The core dump
  writes the inputs and the core-level `expected` values; the float, fixed and
  fixed-float solution metadata, the 10 degree elevation-cutoff solve, the
  troposphere-gradient solve, the auto-init solves and a solve with codes that
  place no transmission epoch are then added by
  `scripts/ppp_esbc_expected`, which builds the solve inputs as the Python
  constructors build them and solves with the entry points the binding calls:

      cd scripts/ppp_esbc_expected
      cargo run --release --locked -- ../../tests/fixtures/ppp_esbc.json \
          <sidereon-core>/tests/fixtures/sp3
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

- `core_goldens.json` -- results of core computations that the SPP, QC, SPP
  robustness, static, velocity, PPP-correction and scenario tests compare with
  bit for bit: the default-model SPP solve of the crate's
  `spp_trace_L0_minimal.json`, a pseudorange set consistent with its own
  solution, the per-satellite redundancy of the robust-solve scenario and its
  default robust solve with the bias on G08, the SP3 merge audit trail
  (provenance, omitted epochs, withheld arc cells, clock omissions, dropped
  input epochs, continuity attribution, node selection, agreement aggregates
  and per-satellite coverage) of the scenarios the core
  `sp3_merge_per_epoch_provenance.rs` and `sp3_merge_coverage.rs` tests build,
  with the coverage fixture's trajectories as exact bits and a digest of every
  SP3 text built, the three-epoch static solve, the range-rate and Doppler velocity
  solves, the combined SPP and Doppler velocity solve and the solves that
  separate its terms, the static reference-station RTK solve, the fusion
  updates, the DGNSS solve, the standalone PPP corrections, and the scenario
  simulator output for
  `scenario_base.json`. Each section builds its inputs as the Python test
  builds them, through the core entry points the binding calls. Written by
  `scripts/core_goldens`:

      cd scripts/core_goldens
      cargo run --release --locked -- <sidereon-core>/tests/fixtures ../../tests/fixtures
- `scenario_base.json` -- the scenario `test_018_domain_exposure.py` builds,
  read by `scripts/core_goldens`. The test asserts that the scenario it builds
  equals this file.
