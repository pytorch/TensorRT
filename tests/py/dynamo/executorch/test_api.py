import importlib

import executorch.exir  # noqa: F401
import pytest


@pytest.mark.unit
def test_public_api_symbols_present():
    module = importlib.import_module("torch_tensorrt.executorch")
    assert "get_edge_compile_config" in module.__all__
    assert "TensorRTPartitioner" in module.__all__
    assert "TensorRTBackend" in module.__all__
    assert callable(module.get_edge_compile_config)
    assert module.TensorRTPartitioner is not None
    assert module.TensorRTBackend is not None


@pytest.mark.unit
def test_save_accepts_executorch_output_format(monkeypatch, tmp_path):
    import torch_tensorrt._compile as compile_api

    saved = {}
    model = object()
    partitioner = object()
    file_path = tmp_path / "model.pte"

    monkeypatch.setattr(
        compile_api,
        "_parse_module_type",
        lambda module: compile_api._ModuleType.ep,
    )

    def fake_save_as_executorch(module, path, **kwargs):
        saved["module"] = module
        saved["path"] = path
        saved["kwargs"] = kwargs

    monkeypatch.setattr(compile_api, "_save_as_executorch", fake_save_as_executorch)

    compile_api.save(
        model,
        str(file_path),
        output_format="executorch",
        partitioners=[partitioner],
    )

    assert saved == {
        "module": model,
        "path": str(file_path),
        "kwargs": {"partitioners": [partitioner]},
    }
