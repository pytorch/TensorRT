import importlib
import sys

import pytest


@pytest.mark.unit
def test_lazy_import_error_when_executorch_missing(monkeypatch):
    original_module = sys.modules.pop("torch_tensorrt.executorch", None)
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, package=None):
        if name == "executorch":
            return None
        return original_find_spec(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    module = importlib.import_module("torch_tensorrt.executorch")

    with pytest.raises(ImportError, match="ExecuTorch is required"):
        _ = module.TensorRTBackend

    sys.modules.pop("torch_tensorrt.executorch", None)
    if original_module is not None:
        sys.modules["torch_tensorrt.executorch"] = original_module


@pytest.mark.unit
def test_public_api_symbols_present():
    module = importlib.import_module("torch_tensorrt.executorch")
    assert "get_edge_compile_config" in module.__all__
    assert "TensorRTPartitioner" in module.__all__
    assert "TensorRTBackend" in module.__all__
