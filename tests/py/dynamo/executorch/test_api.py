import importlib
import sys
from types import SimpleNamespace

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

    with pytest.raises(ImportError, match="ExecuTorch is required"):
        _ = module.to_trt

    sys.modules.pop("torch_tensorrt.executorch", None)
    if original_module is not None:
        sys.modules["torch_tensorrt.executorch"] = original_module


@pytest.mark.unit
def test_public_api_symbols_present():
    module = importlib.import_module("torch_tensorrt.executorch")
    assert "to_trt" in module.__all__
    assert "get_edge_compile_config" in module.__all__
    assert "TensorRTPartitioner" in module.__all__
    assert "TensorRTBackend" in module.__all__


@pytest.mark.unit
def test_to_trt_uses_export_and_placeholder_rewrite(monkeypatch):
    pytest.importorskip("executorch")
    module = importlib.import_module("torch_tensorrt.executorch")

    compiled_gm = object()
    exported_program = SimpleNamespace(graph_module=SimpleNamespace(graph=None))
    rewritten_program = object()

    monkeypatch.setattr(
        "torch_tensorrt.dynamo.compile",
        lambda *args, **kwargs: compiled_gm,
    )
    monkeypatch.setattr(
        "torch_tensorrt.dynamo._exporter.export",
        lambda *args, **kwargs: exported_program,
    )
    monkeypatch.setattr(
        "torch_tensorrt._compile._count_executorch_engine_nodes",
        lambda ep: 1,
    )
    monkeypatch.setattr(
        "torch_tensorrt._compile._replace_execute_engine_for_executorch",
        lambda ep: rewritten_program,
    )

    result = module.to_trt(object(), inputs=[])

    assert result is rewritten_program
