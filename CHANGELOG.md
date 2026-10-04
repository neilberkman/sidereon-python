# Changelog

All notable changes to the Sidereon Python interface are documented here.

## [Unreleased]

### Added

- Opt-in `acquire(..., http_client_exception_diagnostics=callback)` reports
  the class and bounded module/function metadata of unexpected exceptions
  from caller-supplied HTTP transports. Diagnostics omit messages, arguments,
  URLs and request/response data; observer failures leave the original terminal
  exception unchanged.

## [3.0.0] - 2026-10-04

### Added

- Exact `ExactEpoch` and `ExactEpochQuery` entry points now reach precise, broadcast, SBAS-corrected and SSR-corrected ephemeris source hooks for selected state, transmit-clock placement, variance and state-aware clock relativity. `EphemerisQueryState` preserves group delay and accepted UT1 degradation; strict UT1 and SSR correction-size refusals remain typed.
- `ClockInstant` can be constructed from integer nanoseconds or split Julian-date components. Precise position and accuracy samples accept and return this native instant alongside their compatibility J2000-seconds properties.
- `solve_spp_batch_exact` solves aligned configuration and exact-epoch batches. RTCM legacy, system, text, network, transformation, NavIC and GLONASS code/phase-bias payloads support construction and message encoding.
- IONEX serialization refusals raise `IonexWriteError` with the writer detail. SP3 accuracy conversions are checked against exact rational expectations.
- The inertial surface includes IMU error-model correction and validation, per-spec Gauss-Markov helpers, Rodrigues/DCM/quaternion conversions, and both simulation constants. Station tide bindings expose typed position/epoch/polar-motion/options inputs, scalar and row-preserving batch evaluations, ocean constituent labels/order, and nested typed tide failure details. Composite propagation accepts `TideSystem` for solid-Earth gravity and zonal coefficients; embedded EGM96 spherical harmonics remain tide-free.
- SPP, RINEX SPP assembly, and static positioning expose `QzssClock` and `TroposphereModel` selectors, including exact and batch solve forwarding. PPP correction precompute accepts explicit station-tide constants and UT1 validity, reports degradation metadata, and raises structured `PppCorrectionsError` refusals.

- `FieldErrorKind`, `FieldError`, `FormatRecordRef`, `FormatSkipReasonKind`, `FormatSkipReason`, `FormatSkip`, `FormatWarningKind`, `FormatWarning` and `FormatDiagnostics` expose typed parser diagnostics. `BiasParsed` and `SpaceWeatherTable` gain `diagnostics` returning `FormatDiagnostics`.

- `PppPcvSample`, `PppReceiverAntennaFrequency`, `PppReceiverAntennaOptions`, `PppSatelliteClockCorrections`, `PppCorrectionLookup`, `PppCorrectionsOptions` and `PppRangeCorrections`. `PppCorrections` gains `code_bias_m` and `diagnostics`. `PppFloatConfig` and `PppFixedConfig` take a `corrections` argument and gain a `corrections` getter.
- `Ut1OutsideCoverageError`, a subclass of `SidereonError` and `ValueError` with a `reason` of `before_coverage` or `after_coverage`, and `ValidityMode` (`STRICT`, `PERMISSIVE`). `SppSolution`, `StaticSolutionMetadata`, `AraimGeometry`, `ProtectionGeometry`, `FusionUpdate` and the static reference-station solutions gain `ut1_degraded`; `StaticReferenceModeError` gains the kind `ut1_outside_coverage` with `ut1_reason`; `SbasPlError` gains `UT1_OUTSIDE_COVERAGE`. `SsrCorrectedEphemeris` takes `ut1_validity` and gains `corrected_state_checked`, which raises `Ut1OutsideCoverageError` where `position_clock_at_j2000_s` declines the satellite.
- `BiasReadPolicy` and a `policy` argument on every Bias-SINEX and CODE DCB reader; `BiasLookup`; `BiasRecord.family`, `unit`, `svn`, `valid_from`, `valid_until`, `raw_epochs`, `slope`, `slope_sigma` and `line`; `BiasSet.mode`, `time_system_label`, `clock_reference` and `notices`. `BiasSet.phase_osb_cycles` takes `carrier_hz`.
- `TlePolicy` and a `policy` argument on `Tle`, `parse_tle_file`, `propagate_batch` and `look_angles_batch`. `TleFile.rejected` lists every rejected record, stray line and orphan name line as a `RejectedTleRecord`; `NamedTle.line_number`; `Tle.elset_number`, `ephemeris_type`, `bstar_text` and `mean_motion_double_dot_text`. `ChecksumWarning.kind` and `found`.
- OMM: `OmmComments`, `OmmSpacecraft`, `OmmCovariance`, and on `Omm` the `classification`, `message_id`, `ref_frame_epoch`, `semi_major_axis_km`, `gm_km3_s2`, `spacecraft`, `bterm_m2_kg`, `agom_m2_kg`, `covariance`, `user_defined` and `comments` items. `parse_omm_xml_all`, `parse_omm_json_array`, `parse_omm_csv`, `parse_omm_csv_array`, `encode_omm_json_array`, `encode_omm_csv`, their `_discarding_comments` forms and `Omm.to_json_string_discarding_comments`; `OmmArray` and `OmmSkippedRecord`.
- `Omm.to_satellite()` creates a nonconstructible `OmmSatellite` through the core OMM-to-SGP4 bridge; `OmmSatellite.propagate()` returns `TlePropagation` TEME states for UTC unix-microsecond epochs.
- `parse_omm_epoch()` parses an OMM epoch directly, preserving 15-digit fractional seconds and UTC-like leap seconds; `OmmEpoch.iso8601` retains a present femtosecond remainder.
- OPM and OEM: the header `comments`, `classification` and `message_id`, `ref_frame_epoch` and block comments, OPM `user_defined` parameters and their comments, OEM `data_comments` and `covariance_comments` as `(position, text)`, and `OemSkippedState`. `OpmCovariance.to_covariance6` and `OemCovariance.to_covariance6` validate the matrix on request.
- CDM: `CdmOdParameters`, `CdmAdditionalParameters`, and every other item of CCSDS 508.0-B-1 tables 3-1 to 3-4 with its comments: `ccsds_cdm_vers`, `message_for`, the relative state vector, the screening period, volume and times, and the drag, SRP and thrust covariance rows. `CdmObject.to_covariance_rtn` returns the validated matrix of the rows the object holds.
- RTCM: `RtcmPolicy` and `RtcmDeparture`; a `policy` argument on `decode_rtcm_stream`, `decode_rtcm_message`, `RtcmMessage.encode`, `SsrMessage.encode`, `decode_ssr_message` and `decode_ssr`; `decode_rtcm_message_with_policy` and `RtcmMessage.encode_with_policy`, which return the departures. `trailing_bits` on the station, antenna, ephemeris and MSM payloads; `RtcmMsmMessage.signal_mask`; `RtcmGlonassEphemeris.negative_zero` with the `RTCM_GLONASS_NEGATIVE_ZERO_*` constants; the `RTCM_MSM*_INVALID` constants; `RtcmStreamDiagnostics.crc_failures`, `departures` and `is_clean`; `RtcmFrameSkip.departure`; `SsrMessage.padding_bits`.
- SSR: `ssr_store_from_rtcm_strict`, `SsrRtcmIngest` and `SsrIngestRefusal`; `transmitted_epoch_j2000_s` and `nav_message` on the orbit and clock corrections.
- SBAS: `SbasPolicy`, `SbasDeparture`, `SbasBlock.pad_bits` and `encode_with_policy`, `decode_sbas_block_with_policy`, and a `policy` argument on `decode_sbas_block`, `decode_sbas_message`, `SbasBlock.encode` and `SbasLogBlock.decode`. `parse_sbas_ems_log` and `parse_sbas_rtklib_log` return an `SbasLog` of blocks, skipped lines, `SbasRefusedLine`s and departures. `SbasLogBlock.declared_message_type` and `message_type`; `SbasIonoGrid.unavailable_igps` as `SbasUnavailableIgp`.
- RINEX NAV: `BroadcastRecord.iodc`, `l2_codes`, `l2p_data_flag`, `galileo_data_sources`, `beidou_aodc`, `transmission_time_sow` and `stated_fields`; `GlonassRecord.epoch_utc_j2000_s`, `toe_gpst_j2000_s`, `stated_freq_channel`, `message_frame_time_s`, `age_days`, the fourth-orbit-line fields with their word readings, `is_healthy`, `single_frequency_group_delay_s` and `clock_bias_at`; `SkippedNavBlock.line`; `RinexNavParse.departures` and `other` (`NavDiagnostic`, `OtherNavBlock`); `RinexNavWriteError`.
- SPK: `Spk.state_in_frame`, `SpkKernels`, `spk_inertial_frame_name` and `spk_inertial_frame_rotation`.
- Space weather: `ObservationClass.NOT_OBSERVED`, `allow_not_observed` on the policy arguments, `SpaceWeatherTable.ap_history_at_with_policy` returning `ApHistorySample`, and `with_drag_policy` and `drag_policy`, the policy drag reads the table under.
- `PseudorangeCode` and `SppConfig(pseudorange_code=...)`, also read by `StaticEpoch.pseudorange_code`. `PppObservation(signals=...)` carries the tracking codes of its two pseudoranges and two carrier phases. `PppFloatSolution` and `PppFixedSolution` gain `epoch_clocks_m`, `solved_epoch_indices` and `unplaced_observations` (`PppUnplacedObservation`); `PppFloatSolution` gains `residual_screen` and `residual_screen_removals`.
- `SppSolution.status`, `converged` and `iterations` report how the whole SPP solve ended. `SelectionUnsettledError`, a subclass of `SolveError` with `passes`, is raised when an SPP or static solve's satellite selection does not settle within 10 passes, as RTKLIB `estpos` fails an epoch after `MAXITR` iterations.
- `SppSolution.rejected_sats_with_details` and `StaticSolution.rejected_sats_with_details` retain per-epoch SSR orbit and clock correction sizes alongside the rejection reason; the label-only properties remain available.
- `PppFloatSolution` and `PppFixedSolution` gain `residuals_m` (`PppFloatResidual` rows with the epoch index, satellite, ambiguity id, code and phase residuals and both row weights), `status` (`PppFloatStatus`: `STATE_TOLERANCE` or `MAX_ITERATIONS`) and `ssr_bias_exclusions` (`PppSsrBiasExclusion`, with its `PppSsrTransmitTimeFailure` and its `PppSsrObservationApplication` and `PppSsrSignalReport` rows); `PppFloatSolution` gains `solve_options`, the `PppFloatOptions` the solve ran with. `PppUnplacedObservation.message` states the reason in words.
- Labels read from a core enumeration that a newer core may extend (`BiasLookup.status`, `SbasDeparture.kind`, `SbasRefusedLine.reason`, `RtcmDeparture.kind`, `TdmWarning.kind`, `TdmDeparture.kind`, `TdmErrorDetail.kind`, `IonexWarning.kind`, `PppUnplacedObservation.reason`) give a variant this binding does not name its own name in the same case, where they gave `other` or `unknown`. `TdmErrorDetail.input_error_kind_name` names a `TdmInputErrorKind` the same way.
- IONEX: `Ionex.parse`, `Ionex.parse_str`, `Ionex.parse_with_warnings`, `Ionex.parse_str_with_warnings` and `load_ionex_with_warnings`. The `_with_warnings` forms return an `IonexParseResult` holding the product and an `IonexWarning` for each header record that summarizes the maps and disagrees with them or is absent (`EPOCH OF FIRST MAP`, `EPOCH OF LAST MAP`, `# OF MAPS IN FILE`, `INTERVAL`, an `IONEX VERSION / TYPE` that is not the first record, a missing mandatory record), for an `EXPONENT` carried from an earlier map into a map that does not restate it, and for a value field reading `nan`, which reads as non-available. `IonexDiagnosticEpoch` carries a warning's epochs without rounding. `Ionex.skipped_records` counts the records the reader passed over, `AUX DATA` blocks and unrecognized header records included.
- IONEX: `IonexHeader`, from `Ionex.header` and `TecGridSamples.header`, holds the descriptive header records (version, satellite system, `PGM / RUN BY / DATE`, `DESCRIPTION`, `COMMENT`, `INTERVAL`, `ELEVATION CUTOFF`, `OBSERVABLES USED`, `# OF STATIONS`, `# OF SATELLITES`, `# OF MAPS IN FILE` and `MAPPING FUNCTION` as an `IonexMappingFunction`), and the writer writes them back. `IonexHeader.mapping_declaration` and `Ionex.mapping_declaration` return an `IonexMappingDeclaration` separating a declared function from an absent record. `TecGridSamples(header=...)` and `Ionex.from_node_samples(..., header=...)` take a header.
- IONEX: `START OF HEIGHT MAP` blocks are read into `Ionex.height_maps`, `TecGridSamples.height_maps` and `TecSample.height_offset_km`, in kilometres above `HGT1`. `tec_mask`, `rms_mask` and `height_mask` on `Ionex` and `TecGridSamples` (with the aliases `tec_valid`, `rms_valid` and `height_valid`) mark the nodes that hold a value, and `has_rms` and `has_height` say whether the product carries those maps. `TecGridSamples` takes the masks as constructor arguments.
- IONEX slant delays: `IonexSlantPolicy`, made of an `IonexCoveragePolicy`, an `IonexMissingNodePolicy` (`STRICT`, or `RENORMALIZE`, which interpolates around non-available nodes with the weights of the nodes and maps that hold values renormalized and marks the value degraded) and an `IonexMappingPolicy` (`SINGLE_LAYER`, or `DECLARED`, which maps only with the factor the product declares). `Ionex.slant_delay_with_policy` and `ionex_slant_delay_with_policy` return an `IonexSlantDelayEvaluation` with its `IonexSlantDelayStatus` (`held`, `degraded` as an `IonexNodeGap` of `IonexMissingNodes`, `assumed_mapping` as an `IonexAssumedMapping`). `Ionex.slant_delays_batch_results` and `ionex_slant_delay_results` evaluate a list of `IonexSlantRequest`s and return one `IonexSlantBatchResult` per request, holding the evaluation or an `IonexSlantRefusal`. `ionex_slant_delay` is the module-level form of `Ionex.slant_delay`.
- TDM: `parse_tdm_kvn_with_policy` reads a message under a `TdmPolicy` and returns a `TdmParseResult` with the `TdmWarning` for each departure from CCSDS 503.0-B-2 the policy forgave; `Tdm.to_kvn_string_with_policy` and `encode_tdm_kvn_with_policy` write under a `TdmWritePolicy` and return a `TdmWriteResult` with each `TdmDeparture` written. Each policy axis is a `TdmLeniency` (`STRICT` or `FORGIVE`); `TdmWritePolicy.as_read` gives the matching reader policy. `encode_tdm_kvn` and `Tdm.to_kvn` are the strict writer.
- TDM: `TdmComment` keeps a comment's text with its place among the fields or records of its block. `TdmMetadata.from_raw`, `from_raw_with_policy`, `replace_raw` and `replace_raw_with_policy` build or replace metadata from ordered raw fields and positioned comments, the policy forms returning a `TdmMetadataResult` with every departure; `set_field`, `insert_field`, `append_field`, `remove_field`, `remove_field_at`, `add_comment` and the property setters edit metadata, each with a `_with_policy` form returning its departures. `TdmDataSection` gains `add_record`, `insert_record`, `remove_record` and record and comment setters; `Tdm` gains setters for its header items and segments, `set_segment` and `replace_segment`; `TdmSegment` and `TdmField` gain setters.
- TDM: `TdmWriteError` and `TdmValidationError`, subclasses of `TdmParseError`, and `TdmParseError.detail`, a `TdmErrorDetail` carrying the core error's kind, line, keyword, column, character, section, segment, epoch and the other fields it names, with `input_error_kind` as a `TdmInputErrorKind`. `TdmError` is an alias of `TdmErrorDetail`.
- RINEX OBS: `ObsHeader` exposes the rest of the header: `marker_number`, `marker_type`, `observer`, `agency`, the receiver and antenna numbers, types and version, `program`, `run_by`, `program_date`, `comments`, `n_satellites`, `declared_systems`, `rinex2_types`, `rinex2_system`, `signal_strength_unit`, `time_of_last_obs`, `prn_obs_counts`, `unretained_header_labels`, `scale_factors` (`ObsScaleFactor`), `leap_seconds` (`ObsLeapSeconds`, whose `time_system` is the identifier as written, `None` where the record leaves it out, with `BDS` and `BDT` kept apart) and `glonass_cod_phs_bis`, with a blank bias as `None`. `ObsHeader.declared_obs_codes` gives the lists that header declares, and `ObsHeader.glonass_code_phase_bias` returns a `GlonassBiasResult` whose `status` is `available`, `none`, `unknown` or `ambiguous`, with every conflicting bias in `corrections`.
- RINEX OBS: `RinexObs.header_at` and `RinexObs.header_timeline` (`ObsHeaderTimeline`) give the header in effect at an epoch, with the header records of every event at or before it laid over the file header. `RinexObs.skipped_records` counts the records and header contradictions the reader kept out. `ObsEpoch` gains `sats` and `cycle_slips` as `ObsValue`s with their LLI and SSI, `special_records`, `declared_record_count`, `rcv_clock_offset_s`, `epoch_picoseconds`, `cycle_slip_satellites` and `cycle_slip_count`; `CYCLE_SLIP_FLAG` names flag 6.
- RINEX OBS: `CarrierPhaseSeries.phase_shift_results` (`PhaseShiftResult`, with `status` `available`, `unknown` or `ambiguous`), `phase_shift_statuses` and `phase_shift_valid`. `ObsPhaseShift` gains `unrepresentable_satellites`, `covers_every_satellite` and `satellite_count`.
- RINEX OBS: `RinexObs.downgrade_to_rinex2` and `downgrade_to_rinex2` return a product a version 2 file states exactly with every `ObsDowngradeChange` made to reach it. `RinexObsWriteError`, a subclass of `RinexObsParseError`, carries the core refusal as a `RinexObsWriteErrorDetail`. See `docs/observations.md`.
- `Sp3.header` returns an `Sp3Header` with every header field the core retains, including the line-1 data-used descriptor, the `%c` file type and the first `%f` line's position/velocity and clock/rate bases, each `None` when its columns are blank. `Sp3.comments` and `Sp3.skipped_records` expose the retained comment text and the count of skipped entries.
- `Sp3.clock_record(satellite, epoch_index)` and `Sp3.clock_records_at(epoch_index)` return `Sp3ClockRecord` for a valid clock beside the missing-orbit sentinel, with its clock in seconds and in the file's microseconds, the paired `V` record's velocity and clock rate, and the record flags. Such a record is not a state and is never an interpolation node.
- `Sp3State.clock_rate_s_s` carries the clock rate of a velocity product through `Sp3.state`, the interpolant queries and the staleness selection.
- `Sp3WriteErrorDetail` carries the core `Sp3WriteError` variant name and every field of the refusal; see `docs/sp3.md`.
- `RinexClock` exposes the lossless clock model: `header_records` (`ClockHeaderRecord` with its exact text, label, reading and typed `ClockHeaderField`), `records` (`ClockRecord` of every type with its declared values, `surplus_values`, source line and reading), `version`, `layout`, `satellite_system`, `time_system`, `time_system_status`, `notices`, `record_count`, `source_line`, `series_rows` and `instant_series_rows`. `RinexClock.from_series_rows`, `from_instant_series_rows` and `from_clock_points` build products, and `ClockPoint` and `ClockRecord` can be constructed. `set_time_system`, `set_record_values`, `insert_record`, `remove_record`, `retain_records` and `edit_records` edit a product after the core validates the whole change. `to_rinex_string_with_policy` takes a `ClockWritePolicy` and returns a `ClockWriteResult` with each `ClockWriteDeparture`. See `docs/rinex-clock.md`.
- ANTEX products expose every retained record: `Antex.header` (`AntexHeader` with `AntexVersion` and the `AntexPcvTypeRecord` calibration type and reference antenna), `outer_comments`, `antenna_blocks`, `skipped_records`, `antenna_intervals` and `antenna_at`; `Antenna.leading_comments`, `calibrations`, `zenith_grid`, `has_frequency_count`, `comments`, `frequency_sections` and `frequency`; `AntexFrequency` with its PCV samples and RMS section. `AntexDateTime` keeps the exact validity fraction as `fraction_digits` and `fraction_scale` and as a `decimal.Decimal`. See `docs/antex.md`.
- `SppSolution.rejected_sats` lists the satellites an SPP solve excluded, with the core `RejectionReason` name.
- `RinexRtkArc.unresolved_carriers` and `RinexDualFrequencyRtkArc.unresolved_carriers` report, as `RtkRinexUnresolvedCarrier` with its `RtkRinexReceiver`, each measurement left out of an epoch because its phase observable has no carrier frequency; `RtkStaticArcSolution` and `RinexWideLaneFixedRtkSolution` carry the list for a solve made directly from RINEX.
- `SbasCorrectionStore.unassigned_mask_corrections` counts the corrections a GEO addressed to PRN-mask bits that name no satellite.
- `DtedTile.horizontal_datum` returns the DSI datum as a `DtedHorizontalDatum`. `TERRAIN_STORE_NULL_POSTING` is the stored value of a DTED null posting. `DtedTerrain.height_batch_with_validity` and `MmapTerrain.height_batch_with_validity` return heights with NaN, and a False validity mask entry, at each unknown elevation.
- `parse_ocean_loading_blq_block`, `parse_ocean_loading_blq_blocks`, `write_ocean_loading_blq_blocks`, `OceanLoadingBlqBlock` and `OceanLoadingBlqComment` read and write BLQ station blocks with their comment and column-order header lines in place. `OceanLoadingBlq` gains `amplitude_m` and `phase_deg`.
- `SppSolution` and `DgnssSolution` gain `pseudorange_variances_m2`, the RTKLIB `rescode` variance the solve weighted each used satellite by, and `weights`, the weight each carried in the reported solve. `RaimInput` takes and returns `variances_m2`, and `raim` and `qc_raim` take `variances_m2`. `RaimWeights.solution()` and `RaimWeights.is_solution`; the RAIM and FDE `weights` argument also takes a `RaimWeights` or the label `"solution"` or `"unit"`. `FdeResult` gains `raim`, the accepted solution's detection test, and `solution`. `FdeFaultUnresolvedError`, a subclass of `SolveError`, with `reason` (`exclusion_budget_exhausted` or `no_admissible_exclusion`), the last `solution`, the `excluded` satellites and its `raim`. The FDE entry points take `max_exclusions` and `max_exclusion_rms_m`, and `qc_raim_fde_design` takes `max_exclusion_rms_m`. `RAIM_DEFAULT_P_FA`, `FDE_DEFAULT_MAX_EXCLUSION_RMS_M`, `FDE_MIN_OBSERVATIONS` and `FDE_MIN_CANDIDATE_SATELLITES`.
- SP3 merge audit trail: `Sp3MergeReport` gains `omitted_epochs` (as `ClockInstant`s) and `omitted_epochs_j2000_seconds`, `arc_withheld`, `clock_omissions` (`Sp3ClockOmission`, with `reason` and `preferred_source`), `dropped_input_epochs` (`Sp3DroppedInputEpoch`), `single_source_fraction`, `provenance` and `continuity`, and `splices_influencing`. `Sp3MergeOptions` takes `provenance` (`Sp3ProvenanceMode` `SUMMARY` or `FULL`) and `verify_continuity` (`Sp3ContinuityOptions`, whose residual tolerance must be finite and non-negative); neither changes the merged product. Provenance is an `Sp3MergeProvenance` of `Sp3CellProvenance` cells with their `Sp3CellSelection`, `Sp3PrecedenceTransition`s and `Sp3ContributorCoverage`. The continuity post-condition is an `Sp3MergeContinuityReport` with its `Sp3ContinuityDefect`s (every field of each kind), `Sp3MergeContinuityViolation`s with their `Sp3MergeContinuityCell`s and sources, the merged product's `Sp3InterpolationNodes` and window-scoped `violations_influencing`, `splices_influencing` and `verdict`. `Sp3InterpolationNodes.for_sp3` and `selected_nodes` give the nodes an evaluation window's interpolations select. `Sp3MergeFlag.epoch` is the exact epoch. The continuity defect and violation dicts `Sp3.check_continuity` and the verdicts return carry each defect kind's own fields, and a violation's `cells` and `sources`.
- `Sp3.satellite_coverage()` returns an `Sp3Coverage`: the `Sp3EpochGrid` the product's epochs lie on and, for every satellite, an `Sp3SatelliteCoverage` with the `Sp3ChannelCoverage` of its positions and of its clocks as `Sp3CoverageSpan`s and `Sp3CoverageGap`s. `Sp3.epoch_instants` lists the product's epochs exactly.

### Changed

- **Breaking.** FDE (`qc_fde`, `qc_fde_broadcast`, `fde_broadcast`, `solve_spp_robust_fde`) chooses each exclusion by RTKLIB demo5 `raim_fde`'s leave-one-out rule, as the core does: each satellite of the flagged solution is left out in turn and the rest re-solved, and the re-solve with the smallest unweighted residual RMS, no larger than `max_exclusion_rms_m` (default 100 m) and using at least five satellites, is kept. It no longer removes the satellite with the largest residual, which removed healthy satellites when the fault sat on a satellite the geometry leans on. The budget is `max_exclusions`, default 1, RTKLIB's single exclusion, the core's name for it; the `max_iterations` argument is removed. `p_fa` defaults to `1e-3`, RTKLIB's `chisqr` alpha. A fault still detected when the loop stops raises `FdeFaultUnresolvedError`, where a `SolveError` carried only the statistic in its message. `qc_raim_fde_design` applies the same rule, and its `max_exclusions` defaults to 1 (`None` is unbounded).
- **Breaking.** RAIM tests residuals against the variances the solve weighted them by by default (`RaimWeights.solution()`), RTKLIB `valsol`'s `sum (v / sigma)^2`, where it assumed sigma = 1 m. `raim_for_solution` and FDE read the solution's `pseudorange_variances_m2` and clock count; `raim` and `qc_raim` refuse residuals without `variances_m2` with `ValueError` unless `weights` selects `"unit"` or per-satellite weights. `RaimWeights()` is the solution weighting. `RaimResult.reduced_chi_square` is `test_statistic / dof` in every mode.
- **Breaking.** Merged-SP3 report schema 3 records acquisition failures in one vocabulary shared with the Elixir interface, and `data.verify_merge_report` refuses any other spelling there; schemas 1 and 2 are still read with their earlier spellings. `data.AbsentCenter.reason` is `offline_cache_miss`, `product_not_published`, `checksum_mismatch` or `http_status` for a failed candidate (it was `offline_miss`, `candidate_not_found`, `checksum` or `http_status:<status>`), with the status in `http_status`, and `no_candidate` or `catalog_unavailable` when none was tried. A candidate failure no canonical reason describes is `unclassified`, with its raw text in the new `AbsentCenter.detail`, and a source failure no canonical type describes is `unclassified_failure`, with its raw type and text in the new `SourceFailure.detail`; the record carries `detail` exactly for those. An HTTP status (100-599) with no more specific failure raises `distribution.HttpStatusFailure`, a `TransportFailure` subclass whose code is `http_status`, where it raised a `TransportFailure` recorded as `transport_failure`. A response status outside 100-599 is no HTTP status (RFC 9110 section 15): it raises a `TransportFailure` of kind `http_<value>` with no `status`, and a recorded `status` or `http_status` is always 100-599, which `data.verify_merge_report` requires in every schema.
- `data.verify_merge_report` accepts a UTC `23:59:60.x` epoch in the spelling the core writes since 3.0.0, on the next day's boundary with a fraction in [-1/86400, 0) after a day that ends with a positive leap second, and a single-product schema 2 or 3 report holding a clock-only record, which it refused. It also refuses a continuity record with more findings than the pairs or residuals it checked, and a continuity gap threshold factor that is not greater than 1.
- `Sp3MergeOptions` no longer refuses a `target_epoch_interval_s` that is not a whole number of seconds. The merge takes any interval a whole number of 10-nanosecond ticks states and the core refuses the rest; the persisted merge-input identity still binds whole seconds only, and the core refuses a fractional interval there.
- **Breaking.** The robust (Huber) reweighting runs until it settles: `SPP_DEFAULT_ROBUST_MAX_OUTER`, the default `SppRobustConfig.max_outer`, is 100 instead of 5, a safeguard rather than a working budget. A reweighting that cycles stops with the new status `outer_oscillation`.

- Engine update: sidereon and sidereon-core 3.0.0, at core revision `e2fb3dfdc392d23087ed8aa1ee028a0056b4021b`. Every core breaking change below reaches Python as the core states it; the core CHANGELOG gives the reason for each.
- **Breaking.** `NmeaDiagnostics.skips` and `warnings` are lists of `FormatSkip` and `FormatWarning` instead of strings, and `NmeaDiagnostics` is `FormatDiagnostics`.
- **Breaking.** `BiasSet.code_osb_seconds`, `phase_osb_cycles`, `code_dsb_seconds` and `code_bias_model_m` return a `BiasLookup` whose `status` names the outcome (`available`, `absent`, `unsupported_scale`, `ambiguous`, `carrier_frequency_required`, `invalid_carrier_frequency`, `carrier_frequency_unknown`, `undefined_slope_reference`, `invalid_epoch`) instead of a float or `None`. A phase bias stated in nanoseconds needs `carrier_hz`. `BiasSet.time_scale` is `None` for a product with no usable time scale, and a lookup on such a product needs `time_scale`. `BiasRecord.is_phase` follows the observable code, so a phase bias stated in nanoseconds is a phase bias.
- **Breaking.** The Bias-SINEX readers are strict by default and raise `ValueError` for a file that departs from Bias-SINEX 1.00 or states another version; `BiasReadPolicy.LENIENT` reads it and reports each departure in `BiasSet.notices`. A CODE DCB title whose time-system label names no known scale is refused likewise.
- **Breaking.** TLE reading is strict by default: a column-69 digit that disagrees with the checksum, or a column 69 that is not a digit, raises `TleParseError` from `Tle`, `propagate_batch` and `look_angles_batch`, and `parse_tle_file` lists the record in `TleFile.rejected`. `TlePolicy.LENIENT` reads it and reports it in `checksum_warnings`. `ChecksumWarning.expected` is `None` except for a mismatching digit. `TleFile.skipped` counts every rejected entry, stray lines and orphan name lines included. `Tle.rev_number` is `None` for a blank field. `TleFit.catalog_number`, `mean_motion_dot` and `mean_motion_double_dot` are `None` when the fitted element set states none.
- **Breaking.** `Omm.mean_motion`, `norad_cat_id`, `ephemeris_type`, `classification_type`, `element_set_no`, `rev_at_epoch`, `bstar`, `mean_motion_dot`, `mean_motion_ddot` and `ccsds_omm_vers` are `None` when the message does not state them; the constructor no longer fills in `0`, `U`, `999`, `2.0` or zero derivatives. `parse_omm_json` and `parse_omm_xml` read one message and raise `OmmParseError` for a document holding several. `Omm.to_kvn_string`, `to_xml_string` and `to_json_string` raise `OmmParseError` for text the reader would not return unchanged; GP JSON carries one header comment, so any other comment raises. `SkippedOmm.norad_id` is `None` for a record without `NORAD_CAT_ID`.
- **Breaking.** `OpmCovariance` and `OemCovariance` hold the 21 lower-triangle values as read: they are built from `lower_triangle` and expose it, and `to_covariance6()` replaces the `matrix` property, since a matrix printed to a few digits can fall short of positive semidefinite while every value reads correctly. `Oem.skipped_states` is a list of `OemSkippedState` instead of a count. The OPM, OEM and CDM writers raise their parse error for text their reader would not return unchanged.
- **Breaking.** `decode_rtcm` reads a byte stream in full or raises `RtcmParseError` naming the first stray byte, CRC-24Q failure, trailing partial frame or undecodable frame; `decode_rtcm_stream` reads a noisy stream and reports each skip. The strict decoders refuse frame reserved bits that are not zero, trailing bits after a message's last field and an MSM cell mask over 64 bits. The RTCM encoders refuse every value they would otherwise truncate, fill or drop. `RtcmFrameSkip.reason` is also `departure`.
- **Breaking.** `ssr_store_from_rtcm` reads every readable frame under the lenient policy and returns an `SsrRtcmIngest` holding the store and an account of what was not read or not applied; `ssr_store_from_rtcm_strict` returns the store and refuses anything it cannot read and apply in full.
- **Breaking.** `SsrCorrectionStore.code_bias` and `phase_bias` take the signal as a RINEX 3 band and attribute (`"1C"`) or observation code (`"C1C"`) of the satellite's system, or a raw index with its `source`, instead of a bare index: biases are keyed by physical signal, so a Galileo HAS and an RTCM SSR record of different signals at one index no longer share an entry.
- **Breaking.** `SbasBlock.encode` raises `RtcmEncodeError` for a value the wire form cannot carry as held, and under the strict policy for a preamble other than 0x53, 0x9A and 0xC6, where it truncated or filled it. `decode_sbas_block` refuses such a preamble by default. `SbasCorrectionStore.ingest` refuses a message the wire form cannot carry. The SBAS log readers refuse a record whose declared message type differs from the type its message carries.
- **Breaking.** `BroadcastRecord.sv_accuracy_m` is `None` for a blank, unreadable or no-prediction accuracy, and `issue` and `issue_message` are `None` for a CNAV-family record, which states no issue of data. `encode_rinex_nav` writes `PGM / RUN BY / DATE` and raises `RinexNavWriteError` for a set holding both a CNAV-family record and an unclassified Galileo record. `NavMessage` gains `NAVIC_LNAV` and `GALILEO_UNCLASSIFIED`.
- **Breaking.** `SpkState.velocity_km_s` is always an array: Type 2 segments return the derivative of the position Chebyshev expansion, as CSPICE `SPKE02` does. `Spk.state` selects segments and joins chains as CSPICE `SPKSFS` and `SPKGEO` do, and a query whose target equals its center returns the zero state.
- **Breaking.** `SpaceWeatherTable.sample_at_with_policy` defaults to `require_geomagnetic=True`, the core default: an Ap value the file does not state is refused unless the caller allows it. A row whose F10.7 flux qualifier states no observation reads as `ObservationClass.NOT_OBSERVED` and is refused unless `allow_not_observed` is set. Drag reads a table under the same default policy.
- **Breaking.** Event searches and DGNSS raise `Ut1OutsideCoverageError` for an instant outside the UT1 table; SBAS protection levels raise it for a satellite position refused that way.
- **Breaking.** `PppFloatSolution.epoch_clocks_m` and `PppFixedSolution.epoch_clocks_m` hold one clock per solved epoch, in `solved_epoch_indices` order. A PPP observation whose code is zero or negative is left out before the solve and listed in `unplaced_observations`.
- **Breaking.** SPP, the static solve, DGNSS, the tightly coupled fusion code, carrier-phase and range-rate rows, PPP and the RINEX RTK arc builders place each satellite at the transmission epoch its pseudorange gives, `t_tx = (t_rx - P / c) - dts`, as RTKLIB `satposs` places it, and range the unrotated state there with the first-order Sagnac term, as RTKLIB `geodist` does. Solutions move by the receiver clock times each satellite's range rate: nothing measurable on a steered receiver clock of about 10 ns, and decimetres of range on an epoch whose receiver clock is 0.48 ms. The broadcast clock a positioning model reads no longer carries the single-frequency group delay, which the SPP, DGNSS and fusion code models subtract from single-frequency code, as RTKLIB `prange` does, so a single-frequency broadcast SPP solution is unchanged; SPP on ionosphere-free code applies none.
- **Breaking.** SPP, DGNSS, the static solve and the tightly coupled fusion code rows add the relativistic clock term `-2 r·v / c²` to a precise (SP3 or interpolated) satellite clock, as RTKLIB `peph2pos` does; they used the clock as written. Solutions on precise sources move by metres to tens of metres: the SPP trace solve in `tests/test_spp.py` moves by 19.75 m, and the three-epoch static solve in `tests/test_017_domain_exposure.py` by 13.2, -2.5 and 14.5 m in ECEF with its residual RMS going from 0 to 3.01 m, because the trace inputs came from a model without the term. A satellite within 1 ms of the end of the product's coverage, where the term cannot be formed, is declined.
- **Breaking.** `build_rinex_rtk_arc` and `build_dual_frequency_rinex_rtk_arc` place each receiver's satellites from that receiver's own pseudoranges, as RTKLIB `rtkpos` calls `satposs` per receiver, where they took the reception epoch less the pseudorange over `c` rounded to whole microseconds with no satellite clock. A satellite is left out of an epoch where the source has no clock for it or its pseudorange is not a positive distance. The elevation mask is taken at the base, from the satellite placed from the base's pseudorange, as RTKLIB `selsat` masks.
- **Breaking.** `dgnss_solve` places each rover satellite from the rover's raw pseudorange, as RTKLIB `rtkpos` calls `satposs` with the rover's own observations, and forms the residual from the corrected pseudorange.
- **Breaking.** The fusion range-rate rows add the rate of the first-order Sagnac term the code rows' ranges carry, from the placed satellite position and velocity and the receiver position and velocity.
- **Breaking.** `solve_spp_with_doppler_velocity` reads each Doppler satellite at the transmission epoch of its pseudorange, the state the position solve used, with the same Sagnac rate term, as RTKLIB `estvel` forms its rows from the `satposs` states. A Doppler satellite without a pseudorange has no row. `solve_velocity`, which has no pseudoranges, keeps the geometric light-time prediction at the reception epoch.
- **Breaking.** `broadcast_comparison` and `broadcast_comparison_window` add the relativistic term to the precise clock they compare a broadcast clock with, and compare against a broadcast clock without the group delay, so their clock statistics move by both.
- **Breaking.** SPP and the static solve select, mask and weight the satellites at every iterate, as RTKLIB `estpos` re-runs `rescode` at every iteration, where they selected once at the initial guess. A pass whose selection is the one the last pass used takes RTKLIB's least-squares step, and the solve ends with the first such step below 1e-4 m, so the position is a fixed point of the RTKLIB iteration; a solve that has not ended after 10 passes raises `SelectionUnsettledError`. `SppSolution.status` and `StaticSolutionMetadata.status` read `selection_settled` for such a solve and `outer_budget_exhausted` for a robust solve whose reweighting budget ran out, with `converged` false; they describe how the whole solve ended, not its last trust-region solve, and `iterations` counts the trust-region iterations of every solve plus one per step. `used_sats`, `rejected_sats`, the residuals, DOP, covariance and geometry diagnostics are those of the last selection at the reported position. A cold start from the geocentre keeps every satellite with an ephemeris on its first pass and masks from the next, so the coarse search lands on the warm-start solution. Solves whose initial guess was not their solution move: the SPP goldens by about 1e-5 m, a coarse-search solve by up to 0.88 m. The broadcast ionosphere delay is not applied for a receiver more than 1 km below the ellipsoid or a satellite at or below its horizon, and the augmentation-grid delay not more than 100 m below it.
- The SPP seed of `solve_ppp_auto_init_float` and `solve_ppp_auto_init_fixed` takes each GLONASS observation's frequency channel, where it took none.
- **Breaking.** `Tle.to_lines` and the TLE writers spell B* and the second mean-motion derivative as python-sgp4's `export_tle` does: the five significant digits correctly rounded with ties to even (3.21675e-9 is `" 32168-8"`, where it was `" 32167-8"`), a zero B* as `" 00000+0"` and a negative-zero B* as `"-00000+0"`, a second derivative at exponent zero with a `-0` exponent, and below 1e-10 the digits at exponent -9 rounded from the exact value. A negative value whose digits round to zero, and a negative-zero first derivative, keep their sign. An OMM's B* and second derivative are quantized with the same digits; a value no TLE field holds passes through unquantized, and only the TLE writer refuses it.
- **Breaking.** An OMM whose epoch is a whole number of microseconds is initialised as python-sgp4 2.22's `sgp4.omm.initialize` initialises it, which Skyfield's `EarthSatellite.from_omm` uses: the epoch as python-sgp4 gives it (NAVSTAR 43 and GALAXY 15 move by about 0.3 microseconds), the mean motion converted as `n / 720 * pi`, and B* and the second derivative as stated. An OMM with a finer epoch, or with an exact SGP4 epoch, is bridged as a TLE as before.
- **Breaking.** Pass prediction and look angles propagate SGP4 at the split Julian date Skyfield 1.54 uses for the same UTC instant, bit for bit, where they split at civil midnight; the fixtures `pass_finder.json` and `sgp4_topocentric.json` move with it. TLE propagation accepts that split, whose fraction is negative for the seconds before each UTC noon.
- **Breaking.** The 1997 and 1999 leap seconds are dated 1997-07-01 and 1999-01-01, as IERS Bulletin C dates them; they were dated 1997-01-01 and 1998-01-01, so every UTC conversion from 1997-01-01 to 1997-06-30 and from 1998-01-01 to 1998-12-31 was one second off, and `23:59:60` was accepted on the wrong days.
- RINEX clock GPS seconds (`ClockPoint.gps_seconds`, `RinexClock.series_rows`, `civil_to_gps_seconds`) are the `f64` nearest the GPS second count a tag states, rounded once; they missed it by one unit in the last place for about one microsecond epoch in seven from 2014 to 2048. `ClockRecord` civil epochs are the `f64` nearest the stated second, and a writer allowed to round a sub-microsecond second rounds to the nearest microsecond. NMEA epoch instants take the `f64` nearest the stated seconds.
- The RTK fixture `tests/fixtures/rtk_wtzr.json`, the CDM fixture `cdm.json`, the OMM fixture `omm.json`, the pass-finder fixture `pass_finder.json` and the PPP fixture `ppp_esbc.json` are regenerated from the core. The SPP, QC, static, velocity, SPP-Doppler, RTK reference-station, fusion, DGNSS, PPP, PPP-correction and scenario tests compare with `tests/fixtures/core_goldens.json` and the `expected` blocks of `ppp_esbc.json`, which the new `scripts/core_goldens` and `scripts/ppp_esbc_expected` programs write from the core, in place of bit patterns and tolerances written into the tests. Both programs pin the core by git revision and write it into their output as `core_revision`.
- **Breaking.** An IONEX value a file gives as `9999`, which IONEX 1 defines as non-available, is `NaN` in `Ionex.tec_maps`, `rms_maps` and `height_maps` and in the matching `TecGridSamples` arrays, with `False` in the mask, and `TecSample.vtec_tecu` and `rms_tecu` are `None` for it; it read as 999.9 TECU at the default exponent and the slant delay interpolated it. `TecSample(vtec_tecu=...)` is optional. `TecGridSamples` refuses a `NaN` or infinite value that no mask marks as absent. A slant delay whose interpolation weights a non-available node raises `SolveError` under the default `IonexMissingNodePolicy.STRICT`.
- **Breaking.** An IONEX slant delay maps vertical TEC with the single-layer `1/cos(z')` whatever the product's `MAPPING FUNCTION` declares, and reports a product declaring anything but `COSZ` in `IonexSlantDelayStatus.assumed_mapping`; the CODE, ESA, JPL, UPC and EMR final GIMs declare `NONE` or `MOD`. Under `IonexMappingPolicy.DECLARED`, a product whose code defines no factor raises `SolveError`, and so does a product whose height maps give different heights or a non-available height, under either mapping policy. A product whose height maps give every node one height uses `HGT1` plus that height.
- **Breaking.** An IONEX axis is read in the direction its step gives, so `Ionex.lat_nodes_deg`, `lon_nodes_deg` and the `TecGridSamples` axes hold an ascending latitude axis as the file gives it, where such a file was refused. A slant delay interpolates across the longitude seam of a grid that closes the circle, where a query past the last longitude was refused or held.
- **Breaking.** `Ionex.to_ionex_string` raises `ValueError` for a value, axis or header field its IONEX field cannot hold exactly, where it rounded values to the header exponent and wrote fields past their columns. It writes the header records the product holds, `EPOCH OF FIRST MAP`, `EPOCH OF LAST MAP`, `# OF MAPS IN FILE` (as the file stated it, for a product read from one), `MAP DIMENSION` and `END OF FILE`, and lays records out in the IONEX 1 columns.
- **Breaking.** The IONEX reader reads values in their `16I5` columns and places each band by its own `LAT/LON1/LON2/DLON/H` record, and refuses with `IonexParseError` a band off the header grid, a node given twice or left without a value, a map numbered out of order, an RMS or height map naming no TEC map, a `MAP DIMENSION 3` product, and a non-ASCII character in a data record. An epoch at hour 24 with a zero minute and second reads as 00:00 of the next day.
- **Breaking.** `parse_tdm_kvn` reads the lines and characters CCSDS 503.0-B-2 defines and refuses, with `TdmParseError` and its `detail`, a character outside printable ASCII, a line over 254 characters, a last line without a terminator, a missing mandatory keyword (`CCSDS_TDM_VERS`, `CREATION_DATE`, `ORIGINATOR`, `TIME_SYSTEM`, a `PARTICIPANT_n`), an empty data section, an empty value, a version outside `x.y`, a keyword outside its section's table or out of table order, an indexed keyword outside its range, a keyword repeated with different values, a `PATH` naming an undefined participant, records out of time order, a repeated record, and a timetag outside the 4.3.9 forms. `TdmPolicy` forgives the departures that change no value; nothing that changes what the message says is forgiven.
- **Breaking.** `Tdm.comments`, `TdmMetadata.comments` and `TdmDataSection.comments` are lists of `TdmComment` instead of `str`, and a comment is written back where it was read instead of at the top of its block. The constructors take a `TdmComment`, a `str`, or a `(text, before_record)` pair.
- **Breaking.** `TdmMetadata(fields, comments=None)` builds metadata from its ordered raw fields and raises `TdmValidationError` for fields the writer would refuse; `participants`, `mode`, `paths`, `timetag_ref`, `time_system` and `range_units` are read from those fields and are no longer constructor arguments.
- **Breaking.** `Tdm.to_kvn_string` raises `TdmWriteError`, a subclass of `TdmParseError`, for a message the reader would refuse or read back as another message, including a field or comment the KVN form cannot carry, where it wrote it. The writer terminates the last line.
- **Breaking.** `RinexObs.to_rinex_string` and `SyntheticObservationSet.to_rinex_string` raise `RinexObsWriteError` and return text only when reading it back gives the product, field by field. They dropped, rounded, wrapped or truncated what a file could not carry and returned the text anyway. A version 2 product whose code lists no version 2 names read as, or which carries `SYS / SCALE FACTOR` records, is refused; `downgrade_to_rinex2` writes it.
- **Breaking.** `ObsEpoch.epoch` is `None` for an event whose epoch fields are blank, which RINEX 2.11 and 3.05 allow and the reader refused. A flag 6 epoch's cycle slips are read into `ObsEpoch.cycle_slips` and never into the observation rows.
- **Breaking.** The header records an event epoch carries take effect for the epochs after it. `ObsHeader.obs_codes` is the union of every list the file declares for a constellation, and epoch values are index-aligned to it; `carrier_phase_rows` uses the header in effect at its epoch.
- **Breaking.** `ObsPhaseShift.code` is `None` for a record naming only its constellation, which RINEX 3.05 section 5.2.12 gives where the alignment is unknown, and `correction_cycles` is `None` for a blank correction. `CarrierPhaseSeries.phase_shift_cycles` is `NaN` where the header states no one correction: a constellation-only record, or one header block giving the code different corrections. From RINEX 4.00, which says the records should be ignored, every row reads 0 cycles and every GLONASS bias `none`.
- **Breaking.** A header block giving one value two different readings is refused, naming the label, for `INTERVAL`, the marker, position, antenna, receiver, observer and signal-strength records, two scale factors for one code, and two channels for one GLONASS slot; phase shifts and GLONASS biases that disagree are kept and read as `ambiguous`. A `LEAP SECONDS` time-system identifier the file's version does not permit is refused.
- **Breaking.** `Sp3.to_sp3_string` raises `Sp3WriteError`, a subclass of `Sp3ParseError` and `ValueError`, when a value cannot be stated in its fixed columns without changing it, instead of rounding it. A mean or median merge usually holds positions finer than the millimetre record column. `data.write_sp3` and `data.fetch_merged_sp3_file` write nothing in that case.
- **Breaking.** SP3 merges keep a clock-only record where a satellite's orbit is missing or quarantined but its clock reached a consensus. Its `Sp3AgreementMetric` has `position_members` 0 and `position_rms_m` and `position_max_m` `None`. `Sp3EpochAgreement.position_rms_m`, `position_max_m` and the matching `per_epoch_agreement` tuple entries cover only the epoch's multi-source position cells and are `None` for an epoch with none, which previously reported 0.0. `Sp3MergeReport.position_agreement_max_m` covers the cells that carry an orbit.
- **Breaking.** `data.MergeReport.to_dict()` writes merged-SP3 report schema 3. It keeps schema 2's agreement, with those `None` values, and adds what the merge did not write (`dropped_input_epochs`, `omitted_epochs`, `arc_withheld`, `clock_omissions`) and the `continuity` and `provenance` records, each `None` when the merge was not asked for it; its merge policy is policy schema 2, which records the `verify_continuity` and `provenance` options (they change neither the product nor the stable input identity). `data.verify_merge_report` verifies each new field against the agreement and the policy (for example an omitted epoch holds no accepted cell, a withheld cell only arises under satellite-arc precedence, a clock omission states whether its cell has a clock as the agreement does, a continuity record holds one violation per defect with the cells the defect rests on, and full provenance accounts for every accepted cell, each source's coverage and every transition), and still verifies schema 2 and schema 1 records, each with exactly its own fields, by the rules they were written under.
- **Breaking.** `RinexClock.to_rinex_string` restates a product read from text byte for byte, line terminators included, where it wrote a three-line header and the `AS` series only; `AR`, `CR`, `DR` and `MS` records, every header record, and after a lossy read the lines that did not read are kept and written. `RinexClock.time_scale` is `None` when the time system resolves to no scale, and a `GLO` product is UTC. Record epochs keep every digit of their seconds field, and QZSST samples project to GPS seconds.
- **Breaking.** `RinexClock.clock_s` raises `RinexClockQueryError`, now a subclass of `RinexClockParseError` and `ValueError`, so the catch it had still holds. `RinexClockWriteError` also subclasses `SidereonError`. Record, point, construction and edit refusals raise the new `RinexClockEditError`. GPS seconds outside the civil years 1 through 9999 raise `RinexClockQueryError` instead of `PanicException`.
- **Breaking.** `ClockEpoch` accepts a `23:59:60` label on a day that ends with a positive leap second, which a UTC or `GLO` product answers; it refused every label that names no GPS-time epoch. `ClockEpoch.gps_seconds` is `None` for such a label.
- **Breaking.** `Antex.to_antex_string` raises `AntexWriteError`, a subclass of `AntexParseError` and `ValueError`, when the core refuses to write the product without loss, with the core error as `detail` and its `field` and `reason` on the exception. Parse refusals set `detail` on `AntexParseError`. `Antenna.pco`, `pcv` and `frequency` raise `AntexQueryError`, a `ValueError`, with `detail`; a label with differing sections is refused as `AmbiguousFrequency`.
- **Breaking.** `Antenna.dazi_deg`, `zenith_start_deg`, `zenith_end_deg` and `zenith_step_deg` are `None` for a block without the record, where they were `0.0`. `Antenna.frequencies` lists section labels in file order, repeats included. ANTEX millimetres become metres as `mm * 1e-3`, as RTKLIB reads them, which moves about one value in seven by one unit in the last place. A `VALID FROM` or `VALID UNTIL` second of 60 is refused.
- **Breaking.** An ionosphere-corrected SPP or static solve excludes a satellite it has no carrier frequency for - a GLONASS satellite with no channel or a channel outside `-7..=6` - and reports it as `IonosphereCarrierUnresolved` instead of failing the solve with `SolveError`. RINEX RTK arcs leave such a satellite out of its epoch instead of failing the arc.
- **Breaking.** `RtcmMessage.encode`, `RtcmMessage.to_frame` and `SsrMessage.encode` raise `RtcmEncodeError`, a subclass of `RtcmParseError` and `ValueError`, for fields the message's bit layout cannot state, such as an MSM satellite id outside the 64-bit mask; these were written as a different mask or satellite.
- **Breaking.** Terrain lookups that weight a DTED null posting raise `TerrainError`, a subclass of `SidereonError` and `ValueError`, naming the tile and posting in a `TerrainErrorDetail`, and so does a lookup from a tile on a horizontal datum other than WGS84. Other terrain lookup and DTED tile refusals raise `TerrainError` with the same messages as before; a batch lookup sets `point_index`. `data.hgt_to_dted` writes SRTM voids as the DTED null instead of 0 m, and `data.fetch_dted` fetches and converts again a cached tile whose provenance names the converter that wrote voids as 0 m.
- **Breaking.** Satellite tokens take the shared `01`..`99` range for every constellation, so `satellite_id_to_sbas_prn` returns `None` for `S59`..`S99` instead of raising.

### Fixed

- `TdmPath.participants`, `SbasFastCorrections.udrei`, `SbasIntegrity.iodf`, `SbasIntegrity.udrei`, `SbasFastDegradation.ai` and `SbasMixedFastCorrections.udrei` return a `list[int]`, as their stubs declare. They returned `bytes`.

## [2.1.1] - 2026-09-22

### Fixed

- CODE predicted ionosphere maps resolve to the archive AIUB now serves them
  from. `cod_prd1` is `CODE/IONO/PRD/COD0OPSP0D_<date>0000_01D_01H_GIM.INX.gz`
  and `cod_prd2` is `CODE/IONO/PRD/COD0OPSP1D_<date>0000_01D_01H_GIM.INX.gz`;
  the `CODE/IONO/P1/<year>` and `CODE/IONO/P2/<year>` `COD0OPSPRD` trees they
  were read from stopped receiving issues after 2026-09-21 and are now empty,
  so every predicted-IONEX request (`predicted_ionex`, `fetch_ionex` with
  `cod_prd1`/`cod_prd2`, the cross-line walk) returned not-published. For the
  dates both layouts carried the objects decompress to the same bytes. The two
  lines now carry distinct official filenames, so their identities and cache
  paths differ by name as well as by prediction horizon. Publication status
  counts only objects under `CODE/IONO/PRD/`, not the rolling copies CODE keeps
  at the tree root.

### Changed

- Engine update: sidereon 2.1.1 / sidereon-core 2.1.1.

## [2.1.0] - 2026-09-05

### Added

- Configurable SP3 coverage-gap interpolation policy (`gap_threshold_factor`, default 1.5):
  - `load_sp3(..., gap_threshold_factor=None)` loads SP3 products with an explicit gap threshold factor.
  - `Sp3.gap_threshold_factor` reads the active factor; `Sp3.with_interpolation_options(factor)` returns a copy with the updated policy.
  - `Sp3.check_continuity(..., gap_threshold_factor=None)` and `Sp3.continuity_verdict(..., gap_threshold_factor=None)` accept an optional gap threshold factor override.
  - `PreciseEphemerisSamples.from_samples(..., gap_threshold_factor=None)` preserves or overrides the factor; `PreciseEphemerisSamples.gap_threshold_factor` and `with_interpolation_options(factor)` inspect and update it.
  - `PreciseEphemerisInterpolant.from_sp3(..., gap_threshold_factor=None)`, `from_samples(..., gap_threshold_factor=None)`, and `from_precise_ephemeris_samples(..., gap_threshold_factor=None)` accept an optional factor; `PreciseEphemerisInterpolant.gap_threshold_factor` and `with_interpolation_options(factor)` inspect and update it.
  - `Sp3.precise_interpolant_artifact_bytes(gap_threshold_factor=None)`, `build_precise_interpolant_artifact_bytes(sp3, gap_threshold_factor=None)`, and `PreciseInterpolantArtifact.gap_threshold_factor` serialize, build, and inspect precomputed artifacts with the interpolation policy.

### Changed

- Engine update: sidereon 2.1.0 / sidereon-core 2.1.0. Additive upstream release: the SP3 coverage-gap threshold is now a validated, product-carried policy (`Sp3InterpolationOptions`, default 1.5 and bit-identical to before), the SP3 window-scoped continuity reach is derived from the interpolator's actual selectable node spans, and RINEX 4 CNAV week/TOW round trips are stable at the week boundary.

## [2.0.0] - 2026-09-03

### Changed

- Engine update: sidereon 2.0.0 / sidereon-core 2.0.0.
  - Core options and configuration structs are now marked `#[non_exhaustive]`;
    all struct initializations in the Python binding now use explicit
    constructor functions (`::new(...)` or `::default()`) followed by field
    assignments to ensure forward compatibility against future field additions.
  - Core error handling transitioned to typed error enums (including
    `terrain::DtedTileError`, `ionex::tec_grid::TecGridError`, and
    `astro::propagator::dense_output::DenseOutputError`), with error strings
    preserved across the Python interface boundary.

### Changed

- Engine update: sidereon 1.4.1 / sidereon-core 1.4.1 uses portable numerical
  kernels for bit-identical results across x86_64 and arm64. It preserves the
  1.3.3 optimality diagnostics and evaluation counts while iterative-fit
  outputs move only within the documented last-bit differences.

## [1.3.3] - 2026-08-30

### Fixed

- `parse_archive_listing` no longer holds the interpreter lock while it works,
  so other threads keep running during a large parse. Against the previous
  engine a 400k-row body held the lock for ~183 s.

### Changed

- Engine update: sidereon 1.3.3 / sidereon-core 1.3.3. Archive-listing parsing
  is no longer quadratic (154 s to 0.23 s on AIUB's ~426k-row listing), and
  transcendental math is bit-identical across x86_64 and arm64.

## [1.3.1] - 2026-08-29

### Changed

- Engine update: sidereon 1.3.1 / sidereon-core 1.3.1. This release keeps the
  shared release number across the language interfaces, which ships the Go
  interface relicense from Apache-2.0 to MIT. No interface API changes.

## [1.3.0] - 2026-08-29

### Changed

- Engine update: sidereon 1.3.0 / sidereon-core 1.3.0. This release keeps the
  shared release number across the language interfaces, which now include a Go
  interface. No interface API changes.

## [1.2.0] - 2026-08-28

### Changed

- Engine update: sidereon 1.2.0 / sidereon-core 1.2.0, which corrects lenient
  RINEX 4 CNAV decoding and RTKLIB SBAS wire-form preservation. No interface
  API changes.

## [1.1.1] - 2026-08-26

### Changed

- Engine update: sidereon 1.1.1 / sidereon-core 1.1.1, the coordination
  release restoring the shared release number across the language interfaces.
  No numerical, algorithmic, or API changes.

## [1.1.0] - 2026-08-24

### Added

- Source localization now exposes `SourceLocateOptions(include_influence=...)`;
  setting it to `False` skips per-sensor leave-one-out solves and returns an
  empty influence list while preserving the primary solution exactly.
- `closed_form_initial_guess()` names the source-localization seed accurately.
  `chan_ho_initial_guess()` remains available as a deprecated Python alias.

### Changed

- `SourceSensorInfluence.score` is now the larger absolute value of the full
  and leave-one-out residuals divided by the timing sigma; robust downweighting
  is reported separately by `loss_weight`.

## [1.0.1] - 2026-08-22

### Changed

- engine update: sidereon-core 1.0.1 with trust-region-least-squares 0.10.0 (unified fail-closed HostNumerics backend seam; host power dispatch reproduces NumPy's stride-0 scalar-exponent fast paths bit-for-bit). No interface API changes.

## [1.0.0] - 2026-08-21

Sidereon 1.0.0 across every interface; additions arrive without breaking
existing callers from here.

### Added

- Exact-cache single-flight coalescing on the cache surface: Hit/Owner
  outcomes, owner heartbeat and publish, typed timeout errors.
- Window-scoped continuity verdicts: `Sp3.stencil_extent()`,
  `Sp3.continuity_verdict(...)`, and the merge-report equivalent, with
  the stencil reach derived from the interpolator itself.
- `sidereon.data.next_issue_due()`: network-free next nominal issue for
  cataloged product lines.

### Changed

- Engine pinned to `sidereon-core` 1.0.0.

## [0.39.1] - 2026-08-11

### Fixed

- DTED terrain lookups compute the grid cell and intra-cell fraction in
  exact integer arithmetic (engine fix): the binary64 scaling product
  rounded away up to 4096 ULP of the fraction and could flip a
  coordinate strictly below a posting into the next cell's stencil.
  No API change; heights at dyadic-exact coordinates are byte-identical.

### Changed

- Engine pinned to `sidereon-core` 0.39.1.

## [0.39.0] - 2026-08-10

### Added

- `MmapTerrain.from_path_attested()` and
  `PreciseInterpolantArtifact.from_path_attested()`: open a large mapped
  artifact with a caller-attested content checksum instead of the
  O(payload) hash pass, for callers who already hold a trustworthy
  measurement (fs-verity, signed manifest, content-addressed store).
  `digest_provenance()` reports `"verified"` or `"attested"` and
  `verify()` escalates to the full hash on demand. A malformed claim
  raises `ValueError`; it never silently falls back to hashing.

### Changed

- Engine pinned to `sidereon-core` 0.39.0.

## [0.38.0] - 2026-08-09

### Changed

- `MmapTerrain.from_path()` and `PreciseInterpolantArtifact.from_path()`
  now memory-map the file read-only instead of reading it into memory, so
  opening a 30+ GB terrain store no longer costs its size in process
  memory. No API change: existing callers get this by upgrading. Verified
  to return identical heights to the byte-based constructors.

  `from_bytes()` still copies, and deliberately so: a `#[pyclass]` cannot
  borrow from an object whose lifetime Python controls. For a large
  artifact, use `from_path()`.

- Engine pinned to `sidereon-core` 0.38.0 with its `mmap` feature enabled.

## [0.37.0] - 2026-08-09

### Added

- `Sp3.check_continuity()` attests that a parsed or merged product is
  physically continuous, or reports each violation with its epochs and
  magnitude. Two checks with different jobs: a physical earth-fixed speed
  gate whose bound is a true upper bound for the orbit class, so it cannot
  false-positive and catches gross corruption; and a hold-out
  interpolation residual, which supplies the sensitivity a speed gate
  structurally cannot - adjacent GNSS MEO epochs are hundreds of
  kilometres apart, so a metre-scale splice moves the implied speed by a
  fraction of a percent. Reports rather than refuses.

### Changed

- Engine pinned to `sidereon-core` 0.37.0.

## [0.36.3] - 2026-08-04

### Fixed

- Updated `sidereon` and `sidereon-core` to 0.36.3: `parse_archive_listing`
  accepts AIUB whole-tree CSV rows with spaces in unrelated object paths
  instead of rejecting the entire live 426k-row listing over one such row.
  Found by downstream 0.36.1 verification.

## [0.36.2] - 2026-08-04

### Added

- Anonymous-FTP transport (stdlib `ftplib`, no new dependency) for cataloged
  `ftp://` archives, at parity with the Elixir interface: the same bounded
  semantics as HTTP (connect timeout, streamed byte cap surfacing as the
  same typed validation failure, FTP 550 mapped to archive absence like an
  HTTP 404), directory URLs fetching `LIST` text for the core's
  closed-dialect listing parser, and the same retry policy. This makes the
  Wuhan `wum_nrt` hourly line acquirable from Python through the exact
  cache-first acquisition path, with full provenance.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.36.2 (version-alignment engine
  release; no engine changes).

## [0.36.1] - 2026-08-04

### Fixed

- Caller-built product identities now accept the `WUM` publisher token and
  the `near_real_time` solution class introduced by core 0.36.0, so
  `wum_nrt` artifacts round-trip through exact acquisition and merge
  provenance. 0.36.0 rejected them at identity parsing.

## [0.36.0] - 2026-08-04

### Added

- Publication-lag resilience surface over core 0.36.0:
  `data.predicted_ionex_line_candidates` (the opt-in CODE `P1`/`P2`
  cross-line walk for one map date, never a neighboring day's map, each
  candidate keeping its own line identity), `data.parse_archive_listing`
  (closed dialect detection - an unrecognizable listing body raises rather
  than reading as "nothing published"), `data.newest_published_product`,
  `data.publication_listing_urls`, `data.published_issue_age`,
  `data.resolve_first_published`, and the `PublishedObject` /
  `PublishedProduct` dataclasses. `observed_at` is the archive-reported
  modification text, verbatim.
- `data.fetch_ionex(..., cross_line=True)` walks the sibling predicted line
  for the same map date before falling back a day, cache-first, with
  provenance naming the line actually served. Off by default; the
  single-line request stays fail-closed.
- The Wuhan MGEX near-real-time orbit line (`wum_nrt`, hourly
  `WUM0MGXNRT` 02D/05M over anonymous FTP, archive-verified from
  2024-07-03) flows through the catalog surface: `centers()`,
  `ops_ultra_sp3`, ultra-issue selection, and the `near_real_time`
  solution class.

### Changed

- Updated `sidereon` and `sidereon-core` to 0.36.0.

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
