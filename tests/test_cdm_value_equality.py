"""Public equality follows value fields and Rust float PartialEq semantics."""

import re

import numpy as np
import sidereon
from test_cdm import KVN_PATH, _load


def test_public_cdm_dto_equality_and_inequality():
    parsed = sidereon.parse_cdm_kvn(_load(KVN_PATH))
    other_kvn = re.sub(
        r"(?m)^(\s*MESSAGE_ID\s*=\s*).*$",
        r"\1DIFFERENT",
        _load(KVN_PATH),
        count=1,
    )
    assert parsed != sidereon.parse_cdm_kvn(other_kvn)

    obj = parsed.object1
    rebuilt = sidereon.CdmObject(
        obj.position_km,
        obj.velocity_km_s,
        obj.covariance_rtn,
        object_name=obj.object_name,
    )
    assert rebuilt == sidereon.CdmObject(
        obj.position_km,
        obj.velocity_km_s,
        obj.covariance_rtn,
        object_name=obj.object_name,
    )
    changed = sidereon.CdmObject(
        obj.position_km,
        obj.velocity_km_s,
        obj.covariance_rtn,
        object_name="DIFFERENT",
    )
    assert rebuilt != changed

    od = sidereon.CdmOdParameters(obs_used=7, weighted_rms=0.125)
    assert od == sidereon.CdmOdParameters(obs_used=7, weighted_rms=0.125)
    assert od != sidereon.CdmOdParameters(obs_used=8, weighted_rms=0.125)

    additional = sidereon.CdmAdditionalParameters(mass_kg=12.5, sedr_w_kg=0.25)
    assert additional == sidereon.CdmAdditionalParameters(mass_kg=12.5, sedr_w_kg=0.25)
    assert additional != sidereon.CdmAdditionalParameters(mass_kg=13.5, sedr_w_kg=0.25)


def test_public_cdm_dto_equality_matches_float_partial_eq_edges():
    nan = float("nan")

    nan_object = sidereon.CdmObject(
        np.array([nan, 2.0, 3.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0]),
    )
    assert nan_object != nan_object

    positive_zero = sidereon.CdmObject(
        np.array([0.0, 2.0, 3.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0]),
    )
    negative_zero = sidereon.CdmObject(
        np.array([-0.0, 2.0, 3.0]),
        np.array([0.1, 0.2, 0.3]),
        np.array([1.0, 0.0, 2.0, 0.0, 0.0, 3.0]),
    )
    assert positive_zero == negative_zero

    nan_od = sidereon.CdmOdParameters(weighted_rms=nan)
    assert nan_od != nan_od
    nan_additional = sidereon.CdmAdditionalParameters(mass_kg=nan)
    assert nan_additional != nan_additional

    parsed = sidereon.parse_cdm_kvn(_load(KVN_PATH))
    nan_cdm = sidereon.Cdm(
        parsed.object1,
        parsed.object2,
        collision_probability=nan,
    )
    assert nan_cdm != nan_cdm
