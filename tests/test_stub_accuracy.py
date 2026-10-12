"""Runtime-vs-stub drift checks for the exported package surface."""

import ast
import datetime as dt
import enum
import inspect
import pathlib

import numpy as np
import pytest
import sidereon
from _helpers import CORE_FIXTURES, sp3_bytes_for_date

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUB = ROOT / "python" / "sidereon" / "__init__.pyi"
INSTANCE_ONLY_STUB_ATTRIBUTES = {
    "BiasError": {"kind", "details"},
    "ExactSp3ValidationError": {"kind", "detail"},
    "ScenarioError": {"detail"},
}


def _stub_module():
    return ast.parse(STUB.read_text(encoding="utf-8"), filename=str(STUB))


def _stub_defs():
    functions = {}
    classes = {}
    variables = set()
    for node in _stub_module().body:
        if isinstance(node, ast.FunctionDef):
            functions[node.name] = _param_names(node)
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = _class_attrs(node)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    variables.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            variables.add(node.target.id)
    return functions, classes, variables


def _param_names(node):
    params = [arg.arg for arg in [*node.args.posonlyargs, *node.args.args]]
    params.extend(arg.arg for arg in node.args.kwonlyargs)
    return [name for name in params if name not in {"self", "cls"}]


def _class_attrs(node):
    attrs = set()
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            if item.name != "__init__":
                attrs.add(item.name)
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    attrs.add(target.id)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            attrs.add(item.target.id)
    return attrs


def _runtime_param_names(obj):
    signature = inspect.signature(obj)
    names = []
    for param in signature.parameters.values():
        if param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }:
            names.append(param.name)
    return [name for name in names if name not in {"self", "cls"}]


def test_exported_names_are_stubbed():
    functions, classes, variables = _stub_defs()
    missing = [
        name
        for name in sidereon.__all__
        if name not in functions and name not in classes and name not in variables
    ]
    assert missing == []


def test_stub_function_parameter_names_match_runtime():
    functions, _classes, _variables = _stub_defs()
    mismatches = []
    for name in sidereon.__all__:
        if name not in functions:
            continue
        obj = getattr(sidereon, name)
        if inspect.isclass(obj):
            continue
        try:
            runtime = _runtime_param_names(obj)
        except (TypeError, ValueError):
            mismatches.append((name, "uninspectable", functions[name]))
            continue
        if runtime != functions[name]:
            mismatches.append((name, runtime, functions[name]))
    assert mismatches == []


def test_stubbed_class_attributes_exist_at_runtime():
    _functions, classes, _variables = _stub_defs()
    missing = []
    for name in sidereon.__all__:
        if name not in classes:
            continue
        cls = getattr(sidereon, name)
        for attr in classes[name]:
            if attr in INSTANCE_ONLY_STUB_ATTRIBUTES.get(name, set()):
                continue
            if not hasattr(cls, attr):
                missing.append(f"{name}.{attr}")
    assert missing == []


def test_time_scale_stub_members_match_runtime():
    _functions, classes, _variables = _stub_defs()
    declared = classes["TimeScale"] - {"abbrev", "__repr__"}
    runtime = {
        name
        for name, value in vars(sidereon.TimeScale).items()
        if isinstance(value, sidereon.TimeScale)
    }
    assert declared == runtime


def test_instance_only_stub_attribute_allowlist_is_exact():
    assert INSTANCE_ONLY_STUB_ATTRIBUTES == {
        "BiasError": {"kind", "details"},
        "ExactSp3ValidationError": {"kind", "detail"},
        "ScenarioError": {"detail"},
    }
    _functions, classes, _variables = _stub_defs()
    assert INSTANCE_ONLY_STUB_ATTRIBUTES["BiasError"] <= classes["BiasError"]
    assert (
        INSTANCE_ONLY_STUB_ATTRIBUTES["ExactSp3ValidationError"]
        <= classes["ExactSp3ValidationError"]
    )
    assert INSTANCE_ONLY_STUB_ATTRIBUTES["ScenarioError"] <= classes["ScenarioError"]


def test_bias_error_instance_fields_match_stub():
    with pytest.raises(sidereon.BiasError) as captured:
        sidereon.parse_bias_sinex(b"not a SINEX bias file\n")

    assert type(captured.value.kind) is str
    assert type(captured.value.details) is dict


def _sp3_start_mismatch_inputs():
    fixture_path = (
        pathlib.Path(CORE_FIXTURES) / "sp3" / "COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    )
    sp3_date = dt.date(2026, 6, 25)
    shifted = sp3_bytes_for_date(fixture_path.read_bytes(), sp3_date)
    lines = shifted.decode("ascii").splitlines()
    lines[0] = lines[0].replace("0.00000000", "0.00000001", 1)
    content = ("\n".join(lines) + "\n").encode("ascii")
    request = sidereon.ExactSp3Request(sp3_date, "01D", "05M")
    return content, request


def test_exact_sp3_validation_error_instance_fields_match_stub():
    assert not hasattr(sidereon.ExactSp3ValidationError, "kind")
    assert not hasattr(sidereon.ExactSp3ValidationError, "detail")

    direct = sidereon.ExactSp3ValidationError("untyped validation failure")
    assert direct.kind is None
    assert direct.detail is None
    assert "kind" in direct.__dict__
    assert "detail" in direct.__dict__

    with pytest.raises(sidereon.ExactSp3ValidationError) as default_captured:
        sidereon.ExactSp3Request(
            "not-a-date",
            "01D",
            "05M",  # type: ignore[arg-type]
        )

    default_err = default_captured.value
    assert default_err.kind is None
    assert default_err.detail is None
    assert "kind" in default_err.__dict__
    assert "detail" in default_err.__dict__

    content, request = _sp3_start_mismatch_inputs()
    with pytest.raises(sidereon.ExactSp3ValidationError) as mismatch_captured:
        sidereon.parse_exact_sp3(content, request)

    err = mismatch_captured.value
    assert "kind" in err.__dict__
    assert "detail" in err.__dict__
    assert type(err.kind) is str
    assert err.kind == "declared_start_mismatch"
    assert type(err.detail) is dict
    assert err.detail["kind"] == "declared_start_mismatch"
    assert type(err.detail["requested_j2000_s"]) is float
    assert type(err.detail["declared_j2000_s"]) is float
    assert type(err.detail["requested_tick"]) is str
    assert type(err.detail["declared_tick"]) is str
    assert err.detail["requested_tick"] != err.detail["declared_tick"]


def test_code_dcb_option_requiredness_matches_runtime():
    functions = {
        node.name: node
        for node in _stub_module().body
        if isinstance(node, ast.FunctionDef)
    }
    for name in (
        "parse_code_dcb",
        "parse_code_dcb_lossy",
        "load_code_dcb",
        "load_code_dcb_lossy",
    ):
        runtime_parameter = inspect.signature(getattr(sidereon, name)).parameters[
            "options"
        ]
        assert runtime_parameter.default is inspect.Parameter.empty

        stub = functions[name]
        positional = [*stub.args.posonlyargs, *stub.args.args]
        first_default = len(positional) - len(stub.args.defaults)
        option_index = next(
            index
            for index, parameter in enumerate(positional)
            if parameter.arg == "options"
        )
        assert option_index < first_default


def test_selector_declarations_match_runtime_classes():
    names = (
        "TlePolicy",
        "SlipReason",
        "VelocityObservable",
        "ReducedOrbitModel",
        "ReducedOrbitFrame",
        "OrbitType",
        "ProjVgridshiftArithmetic",
        "SeasonKind",
        "MoonPhaseKind",
        "PlanetaryEventKind",
        "CulminationKind",
        "EclipseKind",
        "Planet",
        "SbasSolveMode",
        "CovarianceFrame",
        "FusionFilterKind",
        "ErrorStateLayout",
        "TimeScale",
    )
    classes = {
        node.name: node
        for node in _stub_module().body
        if isinstance(node, ast.ClassDef)
    }
    for name in names:
        assert not issubclass(getattr(sidereon, name), enum.Enum)
        assert classes[name].bases == []


def test_enum_declarations_match_runtime_classes():
    enum_claims = (
        node.name
        for node in _stub_module().body
        if isinstance(node, ast.ClassDef)
        and any(ast.unparse(base) == "enum.Enum" for base in node.bases)
    )
    for name in enum_claims:
        assert issubclass(getattr(sidereon, name), enum.Enum)


def test_fixed_vector_declarations_match_list_runtime_contract():
    expected = {
        "CorrectedImuIncrement": {
            "delta_theta_rad": "list[float]",
            "delta_velocity_mps": "list[float]",
        },
        "EphemerisQueryState": {
            "position_ecef_m": "list[float]",
        },
        "GnssFixMeasurement": {
            "velocity_ecef_mps": "list[float] | None",
        },
        "ImuBias": {
            "accel_mps2": "list[float]",
            "gyro_rps": "list[float]",
        },
        "ImuSample": {
            "angular_rate_rps": "list[float] | None",
            "delta_theta_rad": "list[float] | None",
            "delta_velocity_mps": "list[float] | None",
            "specific_force_mps2": "list[float] | None",
        },
        "InertialFilter": {
            "last_body_rate_wrt_ecef_rps": "list[float]",
        },
        "InertialFilterSnapshot": {
            "last_body_rate_wrt_ecef_rps": "list[float]",
        },
        "LooseCouplingConfig": {
            "lever_arm_body_m": "list[float]",
        },
        "MovingBaselineEpoch": {
            "base_position_m": "list[float]",
        },
        "MovingBaselineEpochSolution": {
            "base_position_m": "list[float]",
            "baseline_m": "list[float]",
        },
        "PppFixedSolution": {
            "position_m": "list[float]",
        },
        "PppFloatSolution": {
            "position_m": "list[float]",
        },
        "PppFloatState": {
            "position_m": "list[float]",
        },
        "PreciseEphemerisAccuracySample": {
            "position_variance_m2": "list[Sp3AccuracyValue]",
        },
        "RinexSppEpochInputs": {
            "initial_guess": "list[float]",
        },
        "RinexWideLaneFixedRtkSolution": {
            "fixed_baseline_m": "list[float]",
            "float_baseline_m": "list[float]",
        },
        "RtcmGlonassCodePhaseBiases": {
            "biases_m": "list[float | None]",
        },
        "RtkArcConfig": {
            "base": "list[float]",
            "initial_baseline_m": "list[float]",
        },
        "RtkArcEpoch": {
            "velocity_mps": "list[float] | None",
        },
        "RtkArcEpochSolution": {
            "float_baseline_m": "list[float]",
            "reported_baseline_m": "list[float]",
        },
        "RtkDualFrequencyArcEpoch": {
            "velocity_mps": "list[float] | None",
        },
        "RtkEpoch": {
            "velocity_mps": "list[float] | None",
        },
        "RtkFilterState": {
            "baseline_m": "list[float]",
        },
        "RtkFixedConfig": {
            "base": "list[float]",
            "initial_baseline_m": "list[float]",
        },
        "RtkFixedSolution": {
            "fixed_baseline_m": "list[float]",
            "float_baseline_m": "list[float]",
        },
        "RtkFloatConfig": {
            "base": "list[float]",
            "initial_baseline_m": "list[float]",
        },
        "RtkFloatSolution": {
            "baseline_m": "list[float]",
        },
        "RtkIonosphereFreeArcConfig": {
            "base": "list[float]",
            "initial_baseline_m": "list[float]",
        },
        "RtkSatMeasurement": {
            "base_tx_pos": "list[float]",
            "pos": "list[float]",
            "rover_tx_pos": "list[float]",
        },
        "RtkStaticArcConfig": {
            "base": "list[float]",
        },
        "RtkWideLaneArcConfig": {
            "base": "list[float]",
        },
        "Sp3AccuracyCodeGroup": {
            "axis_exponents": "list[int | None]",
        },
        "Sp3PositionClockAccuracy": {
            "position_sigma_m": "list[Sp3AccuracyValue]",
        },
        "Sp3VelocityAccuracy": {
            "velocity_sigma_m_s": "list[Sp3AccuracyValue]",
        },
        "SppConfig": {
            "initial_guess": "list[float]",
        },
        "SppKlobucharCoeffs": {
            "alpha": "list[float]",
            "beta": "list[float]",
        },
        "StaticReferenceCarrierSolution": {
            "baseline_vector_m": "list[float]",
            "position_m": "list[float]",
        },
        "StaticReferenceCodeSolution": {
            "baseline_vector_m": "list[float]",
            "position_m": "list[float]",
        },
        "StaticReferenceStationSolution": {
            "baseline_vector_m": "list[float]",
            "position_m": "list[float]",
        },
        "StaticSolveOptions": {
            "initial_position_m": "list[float]",
        },
        "StationDisplacement": {
            "ecef_m": "list[float]",
            "ocean_loading_ecef_m": "list[float] | None",
            "pole_tide_ecef_m": "list[float] | None",
            "solid_earth_tide_ecef_m": "list[float] | None",
        },
        "StationDisplacementPosition": {
            "ecef_m": "list[float] | None",
        },
        "TightCouplingConfig": {
            "lever_arm_body_m": "list[float]",
        },
        "VelocityMatchedTrajectory": {
            "endpoint_position_correction_ecef_m": "list[float]",
            "endpoint_velocity_correction_ecef_mps": "list[float]",
        },
    }
    classes = {
        node.name: node
        for node in _stub_module().body
        if isinstance(node, ast.ClassDef)
    }
    for class_name, properties in expected.items():
        methods = {
            node.name: node
            for node in classes[class_name].body
            if isinstance(node, ast.FunctionDef)
        }
        for property_name, annotation in properties.items():
            assert ast.unparse(methods[property_name].returns) == annotation

    klobuchar = sidereon.SppKlobucharCoeffs()
    bias = sidereon.ImuBias()
    sample = sidereon.ImuSample.rate(1.0, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    corrected = sidereon.CorrectedImuIncrement(
        1.0, [1.0, 2.0, 3.0], [0.1, 0.2, 0.3], 1.0
    )
    measurement = sidereon.GnssFixMeasurement.position_velocity(
        1.0,
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
        np.eye(6, dtype=np.float64),
        6,
    )
    position = sidereon.StationDisplacementPosition.from_ecef_m([1.0, 2.0, 3.0])
    ppp_state = sidereon.PppFloatState(
        position_m=[1.0, 2.0, 3.0], clocks_m=[], ambiguities_m={}
    )
    for value in (
        ppp_state.position_m,
        klobuchar.alpha,
        klobuchar.beta,
        bias.accel_mps2,
        bias.gyro_rps,
        sample.specific_force_mps2,
        sample.angular_rate_rps,
        corrected.delta_velocity_mps,
        corrected.delta_theta_rad,
        measurement.velocity_ecef_mps,
        position.ecef_m,
    ):
        assert type(value) is list
