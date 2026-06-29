import ast
import importlib
import sys
import types
from pathlib import Path

import pytest
import torch
from torch._library.fake_class_registry import FakeScriptObject

from torch_tensorrt.dynamo._exporter import _resolve_lifted_custom_obj


@pytest.mark.unit
def test_lazy_import_error_when_executorch_missing(monkeypatch):
    original_module = sys.modules.pop("torch_tensorrt.executorch", None)
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "executorch.exir":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    module = importlib.import_module("torch_tensorrt.executorch")

    with pytest.raises(ImportError, match=r"torch_tensorrt\[executorch\]"):
        _ = module.TensorRTBackend

    sys.modules.pop("torch_tensorrt.executorch", None)
    if original_module is not None:
        sys.modules["torch_tensorrt.executorch"] = original_module


@pytest.mark.unit
def test_save_executorch_error_when_executorch_missing(monkeypatch, tmp_path):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "executorch.exir":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    from torch_tensorrt._compile import save

    with pytest.raises(ImportError, match=r"torch_tensorrt\[executorch\]"):
        save(
            torch.nn.Linear(1, 1),
            str(tmp_path / "model.pte"),
            output_format="executorch",
        )


@pytest.mark.unit
def test_public_api_symbols_present():
    module = importlib.import_module("torch_tensorrt.executorch")
    assert "get_edge_compile_config" in module.__all__
    assert "TensorRTPartitioner" in module.__all__
    assert "TensorRTBackend" in module.__all__


_REPO_ROOT = Path(__file__).resolve().parents[4]
_SETUP_PY = _REPO_ROOT / "setup.py"


def _setup_tree():
    return ast.parse(_SETUP_PY.read_text(encoding="utf-8"))


def _assignment_value(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return node.value
    raise AssertionError(f"Could not find assignment for {name}")


def _function_def(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Could not find function {name}")


@pytest.mark.unit
def test_packaging_declares_executorch_extra():
    tree = _setup_tree()
    extras = _assignment_value(tree, "EXTRAS_REQUIRE")
    assert isinstance(extras, ast.Dict)

    extras_by_name = {
        key.value: value
        for key, value in zip(extras.keys, extras.values)
        if isinstance(key, ast.Constant)
    }
    for extra_name in ("executorch", "all"):
        assert extra_name in extras_by_name
        requirements = extras_by_name[extra_name]
        assert isinstance(requirements, ast.List)
        assert any(
            isinstance(requirement, ast.Name)
            and requirement.id == "EXECUTORCH_REQUIREMENT"
            for requirement in requirements.elts
        )

    setup_call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "setup"
    )
    extras_keyword = next(
        (keyword for keyword in setup_call.keywords if keyword.arg == "extras_require"),
        None,
    )
    assert extras_keyword is not None
    assert isinstance(extras_keyword.value, ast.Name)
    assert extras_keyword.value.id == "EXTRAS_REQUIRE"


@pytest.mark.unit
def test_executorch_is_not_base_install_requirement():
    tree = _setup_tree()
    for function_name in (
        "get_jetpack_requirements",
        "get_sbsa_requirements",
        "get_x86_64_requirements",
        "get_requirements",
    ):
        function = _function_def(tree, function_name)
        assert not any(
            isinstance(node, ast.Name) and node.id == "EXECUTORCH_REQUIREMENT"
            for node in ast.walk(function)
        )


@pytest.mark.unit
def test_executorch_headers_are_not_dlfw_gated():
    tree = _setup_tree()
    header_package_data = _assignment_value(tree, "executorch_header_package_data")
    assert isinstance(header_package_data, ast.List)
    assert not any(
        isinstance(node, ast.Name) and node.id == "IS_DLFW_CI"
        for node in ast.walk(header_package_data)
    )


def _stub_node(name, target=None):
    return types.SimpleNamespace(name=name, target=name if target is None else target)


def _stub_exported_program(constants, name_to_fqn=None):
    sig = (
        None
        if name_to_fqn is None
        else types.SimpleNamespace(inputs_to_lifted_custom_objs=name_to_fqn)
    )
    return types.SimpleNamespace(constants=constants, graph_signature=sig)


@pytest.mark.unit
def test_resolve_lifted_custom_obj_via_signature_fqn():
    # Modern torch.export: placeholder name differs from the constants FQN key.
    sentinel = object()
    ep = _stub_exported_program({"engine_fqn": sentinel}, {"obj_engine": "engine_fqn"})
    assert _resolve_lifted_custom_obj(ep, _stub_node("obj_engine")) is sentinel


@pytest.mark.unit
def test_resolve_lifted_custom_obj_legacy_fallback():
    # No signature mapping: fall back to a direct name/target lookup.
    sentinel = object()
    ep = _stub_exported_program({"engine": sentinel}, name_to_fqn=None)
    assert _resolve_lifted_custom_obj(ep, _stub_node("engine")) is sentinel


@pytest.mark.unit
def test_resolve_lifted_custom_obj_signature_present_name_absent_is_none():
    # A present-but-incomplete mapping must not bind a different object by name.
    ep = _stub_exported_program({"engine": object()}, {"some_other_obj": "x"})
    assert _resolve_lifted_custom_obj(ep, _stub_node("engine")) is None


@pytest.mark.unit
def test_resolve_lifted_custom_obj_missing_is_none():
    ep = _stub_exported_program({}, name_to_fqn=None)
    assert _resolve_lifted_custom_obj(ep, _stub_node("missing")) is None


@pytest.mark.unit
def test_resolve_lifted_custom_obj_unwraps_fake_script_object():
    class _Real:
        pass

    fake = FakeScriptObject(object(), "Engine", _Real())
    ep = _stub_exported_program({"engine_fqn": fake}, {"obj_engine": "engine_fqn"})
    resolved = _resolve_lifted_custom_obj(ep, _stub_node("obj_engine"))
    assert not isinstance(resolved, FakeScriptObject)
    assert isinstance(resolved, _Real)
