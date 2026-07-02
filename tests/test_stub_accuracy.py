"""Runtime-vs-stub drift checks for the exported package surface."""

import ast
import inspect
import pathlib

import sidereon

ROOT = pathlib.Path(__file__).resolve().parents[1]
STUB = ROOT / "python" / "sidereon" / "__init__.pyi"


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
            if not hasattr(cls, attr):
                missing.append(f"{name}.{attr}")
    assert missing == []
