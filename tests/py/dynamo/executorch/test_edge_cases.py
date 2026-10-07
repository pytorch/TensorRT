# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch_tensorrt
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
def test_save_runs_a_view_left_outside_the_engines(tmp_path):
    """A view TensorRT cannot take stays in PyTorch as aten._reshape_copy, which
    ExecuTorch has no kernel for; the save must still produce a program that runs."""
    pytest.importorskip("executorch.exir")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("requires CUDA + TensorRT for a real engine")
    pytest.importorskip("torch_tensorrt_executorch_runtime")
    from executorch.runtime import Runtime

    class Patchify(torch.nn.Module):
        # Nine dimensions, one more than TensorRT allows, as in Qwen-VL's patch packing.
        def forward(self, x):
            patches = (x * 2.0).view(1, 2, 3, 2, 1, 2, 2, 1, 4)
            patches = patches.permute(0, 1, 3, 2, 4, 5, 6, 7, 8)
            return (patches.reshape(12, 16) + 1.0).relu()

    model = Patchify().eval().cuda()
    x = torch.randn(6, 32, device="cuda")
    with torch.no_grad():
        expected = model(x)
        exported = torch.export.export(model, (x,))
    trt_module = torch_tensorrt.dynamo.compile(
        exported, arg_inputs=[x], min_block_size=1
    )
    pte = tmp_path / "model.pte"
    torch_tensorrt.save(
        trt_module, str(pte), output_format="executorch", arg_inputs=[x], retrace=False
    )

    method = Runtime.get().load_program(pte).load_method("forward")
    torch.testing.assert_close(method.execute([x.cpu()])[0].cpu(), expected.cpu())


@pytest.mark.unit
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)
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

    # The complete set of lowering options _save_as_executorch forwards. The six this
    # test does not pass are still forwarded explicitly, as None or False rather than
    # left out. backend_config is absent by design -- it is not a lowering option and is
    # routed to to_executorch() below.
    export.assert_called_once_with(
        source,
        partitioners=partitioners,
        compile_specs=compile_specs,
        transform_passes=None,
        compile_config=None,
        constant_methods=None,
        generate_etrecord=False,
        weight_streaming_budget_per_engine=None,
        zero_copy_kv=False,
    )
    # With zero_copy_kv off the caller's backend_config reaches to_executorch()
    # exactly as given: save() wraps it only to install the un-staging pass.
    edge.to_executorch.assert_called_once_with(config=backend_config)
    program.write_to_file.assert_called_once()
    program.write_tensor_data_to_file.assert_called_once_with(str(tmp_path))


@pytest.mark.unit
@pytest.mark.skipif(
    not torch_tensorrt.ENABLED_FEATURES.torch_tensorrt_runtime,
    reason="Torch-TensorRT runtime operators are not available",
)
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
