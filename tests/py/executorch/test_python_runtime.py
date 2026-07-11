import importlib.util
import sys
import types
from pathlib import Path

import pytest

RUNTIME_PATH = Path(__file__).parents[3] / "py/torch_tensorrt/executorch/runtime.py"


def load_runtime_module():
    spec = importlib.util.spec_from_file_location(
        "torchtrt_et_runtime_test", RUNTIME_PATH
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
    def load_program(self, path):
        self.path = path
        return FakeProgram()


def test_load_and_forward(monkeypatch, tmp_path):
    delegate = types.ModuleType("torch_tensorrt_executorch_delegate")
    delegate.runtime = FakeRuntime
    monkeypatch.setitem(sys.modules, delegate.__name__, delegate)
    model = tmp_path / "model.pte"
    model.write_bytes(b"pte")

    program = load_runtime_module().load(model)

    assert program.forward(2) == [3]


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
