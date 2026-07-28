import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from torch_tensorrt._compile import (
    _save_as_executorch,
    _write_external_tensor_data,
)
from torch_tensorrt.dynamo.runtime._TorchTensorRTModule import (
    REQUIRES_OUTPUT_ALLOCATOR_IDX,
    SERIALIZATION_LEN,
)
from torch_tensorrt.executorch._export_utils import validate_engine_info


@pytest.mark.unit
def test_validate_executorch_engine_info_rejects_output_allocator():
    engine_info = [""] * SERIALIZATION_LEN
    engine_info[REQUIRES_OUTPUT_ALLOCATOR_IDX] = "1"

    with pytest.raises(RuntimeError, match="output allocator"):
        validate_engine_info(engine_info, node_name="trt")


@pytest.mark.unit
def test_save_as_executorch_uses_public_lowering_and_persists_data(
    monkeypatch, tmp_path
):
    import torch_tensorrt.executorch as executorch_api

    program = SimpleNamespace(
        _tensor_data={"forward": b"weights"},
        write_to_file=MagicMock(),
        write_tensor_data_to_file=MagicMock(),
    )
    edge = SimpleNamespace(to_executorch=MagicMock(return_value=program))
    export = MagicMock(return_value=edge)
    monkeypatch.setattr(executorch_api, "export", export)

    pte = tmp_path / "model.pte"
    source = object()
    partitioners = [object()]
    compile_specs = [object()]
    backend_config = object()
    _save_as_executorch(
        source,
        str(pte),
        partitioners=partitioners,
        compile_specs=compile_specs,
        backend_config=backend_config,
    )

    export.assert_called_once_with(
        source,
        partitioners=partitioners,
        compile_specs=compile_specs,
    )
    edge.to_executorch.assert_called_once_with(config=backend_config)
    program.write_to_file.assert_called_once()
    program.write_tensor_data_to_file.assert_called_once_with(str(tmp_path))


@pytest.mark.unit
@pytest.mark.parametrize("option", ["partitioners", "compile_specs"])
def test_save_as_executorch_rejects_per_method_mapping(monkeypatch, tmp_path, option):
    import torch_tensorrt.executorch as executorch_api

    export = MagicMock()
    monkeypatch.setattr(executorch_api, "export", export)

    with pytest.raises(TypeError, match="must be a list or tuple"):
        _save_as_executorch(
            object(), str(tmp_path / "model.pte"), **{option: {"forward": []}}
        )
    export.assert_not_called()


@pytest.mark.unit
def test_write_external_tensor_data_writes_when_present(tmp_path):
    prog = SimpleNamespace(
        _tensor_data={"forward": b"weights"},
        write_tensor_data_to_file=MagicMock(),
    )
    pte = tmp_path / "model.pte"
    _write_external_tensor_data(prog, str(pte))
    prog.write_tensor_data_to_file.assert_called_once_with(
        os.path.dirname(os.path.abspath(str(pte)))
    )


@pytest.mark.unit
def test_write_external_tensor_data_noop_when_empty(tmp_path):
    prog = SimpleNamespace(
        _tensor_data={},
        write_tensor_data_to_file=MagicMock(),
    )
    _write_external_tensor_data(prog, str(tmp_path / "model.pte"))
    prog.write_tensor_data_to_file.assert_not_called()


@pytest.mark.unit
def test_write_external_tensor_data_fails_loud_without_attr(tmp_path):
    prog = SimpleNamespace(write_tensor_data_to_file=MagicMock())
    with pytest.raises(AttributeError):
        _write_external_tensor_data(prog, str(tmp_path / "model.pte"))
