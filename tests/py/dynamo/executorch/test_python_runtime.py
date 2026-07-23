import importlib.util
import sys
import types
from pathlib import Path

import pytest

RUNTIME_PATH = Path(__file__).parents[4] / "py/torch_tensorrt/executorch/runtime.py"
DELEGATE_PATH = (
    Path(__file__).parents[4]
    / "py/torch-tensorrt-executorch-delegate/torch_tensorrt_executorch_delegate/__init__.py"
)


def load_runtime_module():
    spec = importlib.util.spec_from_file_location(
        "torchtrt_et_runtime_test", RUNTIME_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_delegate_module():
    spec = importlib.util.spec_from_file_location(
        "torchtrt_et_delegate_test", DELEGATE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeMethod:
    def execute(self, inputs):
        return [inputs[0] + 1]


class FakeProgram:
    method_names = {"forward"}

    def load_method(self, name):
        return FakeMethod() if name == "forward" else None


class FakeRuntime:
    def load_program(self, data):
        self.data = data
        return FakeProgram()


def test_load_and_forward(monkeypatch, tmp_path):
    delegate = types.ModuleType("torch_tensorrt_executorch_delegate")
    fake_runtime = FakeRuntime()
    delegate.runtime = lambda: fake_runtime
    monkeypatch.setitem(sys.modules, delegate.__name__, delegate)
    model = tmp_path / "model.pte"
    model.write_bytes(b"pte")

    program = load_runtime_module().load(model)

    assert program.forward(2) == [3]
    assert program._data is fake_runtime.data


def test_unknown_method(monkeypatch, tmp_path):
    delegate = types.ModuleType("torch_tensorrt_executorch_delegate")
    delegate.runtime = FakeRuntime
    monkeypatch.setitem(sys.modules, delegate.__name__, delegate)
    model = tmp_path / "model.pte"
    model.write_bytes(b"pte")

    with pytest.raises(ValueError, match="Unknown method"):
        load_runtime_module().load(model).run([], "missing")


def test_missing_model():
    with pytest.raises(FileNotFoundError):
        load_runtime_module().load("does-not-exist.pte")


def test_activate_twice_is_safe(monkeypatch):
    delegate = load_delegate_module()
    data_loader = types.ModuleType(delegate.__name__ + ".data_loader")
    native = types.ModuleType(delegate.__name__ + "._portable_lib")
    imported = []

    def fake_import(name):
        imported.append(name)
        return {
            data_loader.__name__: data_loader,
            native.__name__: native,
        }[name]

    monkeypatch.setattr(
        delegate, "importlib", types.SimpleNamespace(import_module=fake_import)
    )
    assert delegate.activate() is native
    wrapper = types.ModuleType(delegate._WRAPPER_NAME)
    monkeypatch.setitem(sys.modules, delegate._WRAPPER_NAME, wrapper)
    assert delegate.activate() is native
    assert imported == [data_loader.__name__, native.__name__]
    assert sys.modules[delegate._NATIVE_NAME] is native
    assert sys.modules[delegate._DATA_LOADER_NAME] is data_loader
    assert sys.modules[delegate._WRAPPER_NAME] is wrapper


def test_activate_rejects_preloaded_stock_runtime(monkeypatch):
    delegate = load_delegate_module()
    stock_runtime = types.ModuleType(delegate._NATIVE_NAME)
    monkeypatch.setitem(sys.modules, delegate._NATIVE_NAME, stock_runtime)

    with pytest.raises(delegate.DelegateCompatibilityError, match="stock runtime"):
        delegate.activate()


def test_activate_rejects_preloaded_stock_wrapper(monkeypatch):
    delegate = load_delegate_module()
    stock_wrapper = types.ModuleType(delegate._WRAPPER_NAME)
    monkeypatch.delitem(sys.modules, delegate._NATIVE_NAME, raising=False)
    monkeypatch.setitem(sys.modules, delegate._WRAPPER_NAME, stock_wrapper)

    with pytest.raises(
        delegate.DelegateCompatibilityError,
        match=r"Call torch_tensorrt\.executorch\.load",
    ):
        delegate.activate()


def test_activate_cleans_up_data_loader_when_native_import_fails(monkeypatch):
    delegate = load_delegate_module()
    data_loader = types.ModuleType(delegate.__name__ + ".data_loader")

    def fake_import(name):
        if name == data_loader.__name__:
            return data_loader
        assert delegate._DATA_LOADER_NAME not in sys.modules
        raise ImportError("native module failed to load")

    monkeypatch.setattr(
        delegate, "importlib", types.SimpleNamespace(import_module=fake_import)
    )
    monkeypatch.delitem(sys.modules, delegate._NATIVE_NAME, raising=False)
    monkeypatch.delitem(sys.modules, delegate._DATA_LOADER_NAME, raising=False)

    with pytest.raises(delegate.DelegateCompatibilityError):
        delegate.activate()

    assert delegate._DATA_LOADER_NAME not in sys.modules
