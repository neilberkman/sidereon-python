# Lane PP Parity Closure Report

## #4 RINEX OBS -> SPP Convenience

Closed. Python now exposes `RinexSppOptions`, `RinexSppEpochInputs`,
`RinexSppEpochSolution`, `spp_inputs_from_rinex_obs`, and
`solve_spp_from_rinex_obs`. The source may be `BroadcastEphemeris` or `Sp3`;
the `Sp3` path also accepts a `broadcast_context` NAV product for RINEX assembly
metadata.

Proof: `tests/test_rinex_obs.py::test_rinex_obs_spp_inputs_and_solve_convenience_with_broadcast_nav`.

## #21 `raim_for_solution`

Closed. Python now exposes `raim_for_solution(solution, ...)` over an existing
`SppSolution`, delegating to the core solution RAIM path.

Proof: `tests/test_qc.py::test_raim_typed_input_and_solution_path_use_real_spp_residuals`.

## #24 Broadcast FDE

Closed. Python now exposes `qc_fde_broadcast` and `fde_broadcast` for
broadcast-ephemeris SPP FDE using the same core `fde_spp` driver as the SP3
path.

Proof: `tests/test_broadcast_fallback.py::test_broadcast_fde_runs_real_broadcast_spp_path`.

## #46/#47 SSR and SBAS Decode Visibility

Closed. Existing names remain available. Python now also exposes canonical
top-level aliases `decode_ssr`, `decode_sbas_message`, and
`ssr_store_from_rtcm`.

Proof:
`tests/test_new_core_api.py::test_sbas_decode_parse_store_and_mapping_helpers`;
`tests/test_new_core_api.py::test_ssr_decode_store_and_correction_queries`.

## #93 Terrain Store Parity

Closed for the terrain-store contract drift identified in this lane. Python now
exposes `TerrainTileId`, `DtedTileListEntry`,
`dted_tile_list_to_mmap_store`, `write_dted_tile_list_to_mmap_store`,
`MmapTerrain.as_bytes`, `MmapTerrain.tile_count`, and `MmapTerrain.tile_ids`.

Proof: `tests/test_012_bindings.py::test_mmap_terrain_store_matches_dted_reader_and_missing_dac_is_typed`.

## #20/#93 Checker Follow-up: `raim_standalone` Partial

Closed. The remaining Python drift was the typed RAIM input path and canonical
`raim` visibility. Python keeps `qc_raim`, keeps canonical `raim`, and adds
`RaimInput`; `raim` accepts either `RaimInput` or the existing
`(used_sats, residuals_m)` call form.

Proof: `tests/test_qc.py::test_raim_typed_input_and_solution_path_use_real_spp_residuals`;
`tests/test_stub_accuracy.py::test_stub_function_parameter_names_match_runtime`.

## Gates

Passed:

- `maturin build --release`
- `.venv-lane/bin/pip install --force-reinstall target/wheels/sidereon-0.24.0-cp39-abi3-macosx_11_0_arm64.whl`
- `SIDEREON_CORE_FIXTURES=/Users/neil/xuku/sidereon/crates/sidereon-core/tests/fixtures .venv-lane/bin/pytest tests/ -q`
- `.venv-lane/bin/ruff format --check .`
- `.venv-lane/bin/ruff check .`
- `cargo clippy --all-targets -- -D warnings`
- `cargo fmt --check`
