"""External scenario source identity and transcript binding."""

import copy
import json
import pathlib

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES

CORE_FIXTURE_ROOT = pathlib.Path(CORE_FIXTURES)


def _external_fixture_scenario(source, identity, *, satellite=None, epoch=None):
    scenario_path = pathlib.Path(__file__).parent / "fixtures" / "scenario_base.json"
    scenario = json.loads(scenario_path.read_text(encoding="utf-8"))
    start_j2000_s = (
        float(source.epochs_j2000_seconds[1]) if epoch is None else float(epoch)
    )
    scenario["epochs"].update(start_j2000_s=start_j2000_s, count=1, cadence_s=1.0)
    scenario["error_budget"]["elevation_mask_deg"] = -90.0
    satellite = source.satellites[0] if satellite is None else satellite
    system = {
        "G": "Gps",
        "R": "Glonass",
        "E": "Galileo",
        "C": "BeiDou",
        "J": "Qzss",
        "I": "Navic",
        "S": "Sbas",
    }[satellite[0]]
    scenario["constellation"] = {
        "kind": "external_products",
        "source": {
            "kind": identity.kind,
            "product_id": identity.product_id,
            "content_digest": identity.content_digest,
        },
        "satellites": [{"system": system, "prn": int(satellite[1:])}],
    }
    return scenario


def test_external_sp3_scenario_fingerprint_and_typed_refusals():
    path = CORE_FIXTURE_ROOT / "sp3" / "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    source = sidereon.load_sp3(path)
    pending = sidereon.ScenarioExternalProduct("sp3", "fixture/sp3", "pending")
    scenario = _external_fixture_scenario(source, pending)

    fingerprint = sidereon.scenario_source_transcript_fingerprint(
        scenario, source, pending
    )
    identity = sidereon.ScenarioExternalProduct("sp3", "fixture/sp3", fingerprint)
    scenario["constellation"]["source"]["content_digest"] = fingerprint
    result = sidereon.simulate_scenario_with_source(scenario, source, identity)
    assert result.observations.satellite_id == [source.satellites[0]]
    assert result.observations.epoch_index.tolist() == [0]
    # Independent geometric reference: iterate light time over the checked-in
    # SP3 positions and compute Euclidean range to the equatorial WGS-84 origin.
    receive_time = float(source.epochs_j2000_seconds[1])
    transmit_time = receive_time - 0.075
    receiver_ecef_m = np.array([6_378_137.0, 0.0, 0.0])
    for _ in range(10):
        satellite_ecef_m = source.interpolate(
            source.satellites[0], np.array([transmit_time], dtype=np.float64)
        ).position_m[0]
        reference_range_m = float(np.linalg.norm(satellite_ecef_m - receiver_ecef_m))
        reference_range_m += (
            7.292_115_146_7e-5
            * (
                satellite_ecef_m[0] * receiver_ecef_m[1]
                - satellite_ecef_m[1] * receiver_ecef_m[0]
            )
            / 299_792_458.0
        )
        transmit_time = receive_time - reference_range_m / 299_792_458.0
    assert abs(result.truth_terms.geometric_range_m[0] - reference_range_m) < 5e-5

    with pytest.raises(
        sidereon.ScenarioError,
        match="external source identity mismatch.*constellation.source",
    ) as caught:
        sidereon.simulate_scenario_with_source(
            scenario,
            source,
            sidereon.ScenarioExternalProduct("broadcast", "fixture/sp3", fingerprint),
        )
    assert caught.value.detail.kind == "external_source_mismatch"
    assert caught.value.detail.field == "constellation.source"
    assert caught.value.detail.expected.startswith("Sp3:fixture/sp3:")
    assert caught.value.detail.actual.startswith("Broadcast:fixture/sp3:")

    changed = copy.deepcopy(scenario)
    changed["constellation"]["source"]["content_digest"] = "sha256:wrong-transcript"
    wrong_digest = sidereon.ScenarioExternalProduct(
        "sp3", "fixture/sp3", "sha256:wrong-transcript"
    )
    with pytest.raises(
        sidereon.ScenarioError,
        match="external source identity mismatch.*content_digest",
    ) as caught:
        sidereon.simulate_scenario_with_source(changed, source, wrong_digest)
    assert caught.value.detail.kind == "external_source_mismatch"
    assert caught.value.detail.field == "constellation.source.content_digest"

    with pytest.raises(
        sidereon.ScenarioError,
        match="scenario requires an external ephemeris source",
    ) as caught:
        sidereon.simulate_scenario(scenario)
    assert caught.value.detail.kind == "external_source_required"


def test_precise_interpolant_dispatches_as_an_external_scenario_source():
    sp3_path = CORE_FIXTURE_ROOT / "sp3" / "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    sp3 = sidereon.load_sp3(sp3_path)
    source = sidereon.PreciseEphemerisInterpolant.from_sp3(sp3)
    identity = sidereon.ScenarioExternalProduct("sp3", "fixture/precise", "pending")
    scenario = _external_fixture_scenario(
        source, identity, epoch=sp3.epochs_j2000_seconds[1]
    )
    fingerprint = sidereon.scenario_source_transcript_fingerprint(
        scenario, source, identity
    )
    identity = sidereon.ScenarioExternalProduct("sp3", "fixture/precise", fingerprint)
    scenario["constellation"]["source"]["content_digest"] = fingerprint
    result = sidereon.simulate_scenario_with_source(scenario, source, identity)
    assert result.observations.satellite_id == [sp3.satellites[0]]
    assert result.observations.epoch_index.tolist() == [0]
    with pytest.raises(sidereon.ScenarioError) as caught:
        sidereon.simulate_scenario_with_source(
            scenario,
            source,
            sidereon.ScenarioExternalProduct(
                "broadcast", "fixture/precise", fingerprint
            ),
        )
    assert caught.value.detail.field == "constellation.source"


def test_ssr_and_sbas_sources_dispatch_with_broadcast_fallback():
    nav_path = CORE_FIXTURE_ROOT / "nav" / "KMS300DNK_R_20221591000_01H_MN.rnx"
    broadcast = sidereon.parse_rinex_nav(nav_path.read_text(encoding="utf-8"))
    record = next(
        record for record in broadcast.records if record.satellite.startswith("G")
    )
    assert record.time_scale == sidereon.TimeScale.GPST
    epoch = record.toe_week * 604_800.0 + record.toe_tow_s - 630_763_200.0

    ssr = sidereon.SsrCorrectedEphemeris(
        broadcast,
        sidereon.SsrCorrectionStore(),
        fallback=sidereon.SsrFallbackPolicy(
            on_missing_correction=sidereon.MissingCorrectionAction.FALL_BACK_TO_BROADCAST
        ),
    )
    sbas = sidereon.SbasCorrectedEphemeris(
        broadcast,
        sidereon.SbasCorrectionStore(),
        "S20",
        mode=sidereon.SbasSolveMode.MIXED_AUGMENTATION,
    )
    for source in (ssr, sbas):
        identity = sidereon.ScenarioExternalProduct(
            "broadcast", "fixture/broadcast", "pending"
        )
        scenario = _external_fixture_scenario(
            source, identity, satellite=record.satellite, epoch=epoch
        )
        fingerprint = sidereon.scenario_source_transcript_fingerprint(
            scenario, source, identity
        )
        identity = sidereon.ScenarioExternalProduct(
            "broadcast", "fixture/broadcast", fingerprint
        )
        scenario["constellation"]["source"]["content_digest"] = fingerprint
        result = sidereon.simulate_scenario_with_source(scenario, source, identity)
        assert result.observations.satellite_id == [record.satellite]
        assert result.observations.epoch_index.tolist() == [0]
        with pytest.raises(sidereon.ScenarioError) as caught:
            sidereon.simulate_scenario_with_source(
                scenario,
                source,
                sidereon.ScenarioExternalProduct(
                    "sp3", "fixture/broadcast", fingerprint
                ),
            )
        assert caught.value.detail.field == "constellation.source"


def test_broadcast_source_dispatches_with_declared_identity():
    nav_path = CORE_FIXTURE_ROOT / "nav" / "KMS300DNK_R_20221591000_01H_MN.rnx"
    source = sidereon.parse_rinex_nav(nav_path.read_text(encoding="utf-8"))
    record = next(
        record for record in source.records if record.satellite.startswith("G")
    )
    assert record.time_scale == sidereon.TimeScale.GPST
    epoch = record.toe_week * 604_800.0 + record.toe_tow_s - 630_763_200.0
    identity = sidereon.ScenarioExternalProduct("broadcast", "fixture/nav", "pending")
    scenario = _external_fixture_scenario(
        source, identity, satellite=record.satellite, epoch=epoch
    )
    fingerprint = sidereon.scenario_source_transcript_fingerprint(
        scenario, source, identity
    )
    identity = sidereon.ScenarioExternalProduct("broadcast", "fixture/nav", fingerprint)
    scenario["constellation"]["source"]["content_digest"] = fingerprint
    result = sidereon.simulate_scenario_with_source(scenario, source, identity)
    assert result.observations.satellite_id == [record.satellite]
    assert result.observations.epoch_index.tolist() == [0]

    with pytest.raises(sidereon.ScenarioError) as caught:
        sidereon.simulate_scenario_with_source(
            scenario,
            source,
            sidereon.ScenarioExternalProduct("sp3", "fixture/nav", fingerprint),
        )
    assert caught.value.detail.field == "constellation.source"


def test_ionex_fingerprint_is_available_for_declared_media_identity():
    ionex_path = CORE_FIXTURE_ROOT / "ionex" / "synthetic_2map_7x7.20i"
    ionex = sidereon.load_ionex(ionex_path)
    fingerprint = sidereon.ionex_content_fingerprint(ionex)
    assert fingerprint.startswith("sidereon-fnv64:")
    assert sidereon.ionex_content_fingerprint(ionex) == fingerprint
    identity = sidereon.ScenarioExternalProduct("ionex", "fixture/ionex", fingerprint)
    assert identity.kind == "ionex"
    assert identity.content_digest == fingerprint


def test_external_sp3_scenario_accepts_declared_ionex_and_checks_its_digest():
    sp3_path = CORE_FIXTURE_ROOT / "sp3" / "IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3"
    source = sidereon.load_sp3(sp3_path)
    source_identity = sidereon.ScenarioExternalProduct("sp3", "fixture/sp3", "pending")
    scenario = _external_fixture_scenario(source, source_identity)

    ionex_path = CORE_FIXTURE_ROOT / "ionex" / "synthetic_2map_7x7.20i"
    original_ionex = sidereon.load_ionex(ionex_path)
    epoch = float(source.epochs_j2000_seconds[1])
    samples = sidereon.TecGridSamples(
        map_epochs_j2000_s=np.array(
            [int(round(epoch)) - 3600, int(round(epoch)) + 3600], dtype=np.int64
        ),
        lat_nodes_deg=original_ionex.lat_nodes_deg,
        lon_nodes_deg=original_ionex.lon_nodes_deg,
        dlat_deg=original_ionex.dlat_deg,
        dlon_deg=original_ionex.dlon_deg,
        shell_height_km=original_ionex.shell_height_km,
        base_radius_km=original_ionex.base_radius_km,
        exponent=original_ionex.exponent,
        tec_maps=original_ionex.tec_maps,
        rms_maps=original_ionex.rms_maps,
        height_maps=original_ionex.height_maps,
        tec_mask=original_ionex.tec_mask,
        rms_mask=original_ionex.rms_mask,
        height_mask=original_ionex.height_mask,
        header=original_ionex.header,
    )
    ionex = sidereon.Ionex.from_samples(samples)
    ionex_digest = sidereon.ionex_content_fingerprint(ionex)
    ionex_identity = sidereon.ScenarioExternalProduct(
        "ionex", "fixture/ionex", ionex_digest
    )
    scenario["error_budget"]["ionosphere"] = {
        "kind": "supplied_ionex",
        "source": {
            "kind": "ionex",
            "product_id": ionex_identity.product_id,
            "content_digest": ionex_identity.content_digest,
        },
    }

    transcript = sidereon.scenario_source_transcript_fingerprint(
        scenario, source, source_identity, ionex, ionex_identity
    )
    source_identity = sidereon.ScenarioExternalProduct("sp3", "fixture/sp3", transcript)
    scenario["constellation"]["source"]["content_digest"] = transcript
    result = sidereon.simulate_scenario_with_source_and_media(
        scenario, source, source_identity, ionex, ionex_identity
    )
    assert result.observations.satellite_id == [source.satellites[0]]
    assert result.truth_terms.ionosphere_m[0] > 0.0

    corrupt = sidereon.ScenarioExternalProduct(
        "ionex", "fixture/ionex", "sidereon-fnv64:0000000000000000"
    )
    with pytest.raises(sidereon.ScenarioError) as caught:
        sidereon.simulate_scenario_with_source_and_media(
            scenario, source, source_identity, ionex, corrupt
        )
    assert caught.value.detail.kind == "external_source_mismatch"
    assert caught.value.detail.field == "error_budget.ionosphere.source"
