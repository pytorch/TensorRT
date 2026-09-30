from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch import nn
from torch_tensorrt_edge_llm import ops
from torch_tensorrt_edge_llm.artifact import EdgeExecuTorchArtifact
from torch_tensorrt_edge_llm.serialization import EdgeComponentMetadata, EdgeOutputSpec
from torch_tensorrt_edge_llm.vision import export_vision


def test_opset_import_is_independent_of_exporters_and_executorch():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch, torch_tensorrt_edge_llm; "
            "assert 'exporters' not in sys.modules; "
            "assert 'executorch.exir' not in sys.modules; "
            "assert torch.ops.tensorrt_edge_llm.vision_tower.default._schema.name "
            "== 'tensorrt_edge_llm::vision_tower'",
        ],
        check=True,
    )


class _PackingModule(nn.Module):
    def forward(self, vision, language, compact_index, mask):
        return (
            ops.fuse_prefix(vision, language, compact_index),
            ops.scatter_image_tokens(vision, language, mask),
        )


def test_packing_program_executes_after_save_and_load(tmp_path):
    inputs = (
        torch.tensor([[[10.0, 20.0], [30.0, 40.0]]]),
        torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]]),
        torch.tensor([[0, 2, 4]]),
        torch.tensor([[True, False, True]]),
    )
    program = torch.export.export(_PackingModule(), inputs)
    path = tmp_path / "packing.pt2"
    torch.export.save(program, path)
    restored = torch.export.load(path)

    targets = {
        node.target for node in restored.graph.nodes if node.op == "call_function"
    }
    assert torch.ops.tensorrt_edge_llm.fuse_prefix.default in targets
    assert torch.ops.tensorrt_edge_llm.scatter_image_tokens.default in targets
    compact, scattered = restored.module()(*inputs)
    torch.testing.assert_close(
        compact, torch.tensor([[[10.0, 20.0], [1.0, 2.0], [5.0, 6.0]]])
    )
    torch.testing.assert_close(
        scattered, torch.tensor([[[10.0, 20.0], [3.0, 4.0], [30.0, 40.0]]])
    )
    torch.testing.assert_close(
        inputs[1], torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    )


def test_embedded_vision_program_has_runnable_python_implementation(
    tmp_path, monkeypatch
):
    pixels = torch.arange(24, dtype=torch.float16).reshape(1, 2, 4, 3)
    metadata = EdgeComponentMetadata(
        component="vision",
        runner="vit",
        outputs=(EdgeOutputSpec(shape=(1, 2, 3), dtype="float16"),),
    )
    blob = b"an-embedded-engine"
    artifact = EdgeExecuTorchArtifact(blob, metadata.to_json())
    program = export_vision(artifact, pixels)
    path = tmp_path / "vision.pt2"
    torch.export.save(program, path)
    restored = torch.export.load(path)
    ops._EMBEDDED_MODULES.clear()

    class PythonVision(nn.Module):
        def forward(self, value):
            return value.mean(dim=2)

    def load_engine(payload, restored_metadata):
        assert ops._tensor_bytes(payload) == blob
        assert restored_metadata == metadata
        return PythonVision()

    monkeypatch.setattr(ops, "_get_embedded_engine", load_engine)
    torch.testing.assert_close(restored.module()(pixels), pixels.mean(dim=2))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="TensorRT execution requires CUDA"
)
def test_embedded_vision_loads_real_engine_in_fresh_process(tmp_path):
    import numpy as np
    import torch_tensorrt.executorch.serialization as serialization

    trt = pytest.importorskip("tensorrt")
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    tensor = network.add_input("pixels", trt.float16, (1, 2, 4, 3))
    output = network.add_identity(tensor).get_output(0)
    output.name = "features"
    network.mark_output(output)
    serialized = builder.build_serialized_network(
        network, builder.create_builder_config()
    )
    assert serialized is not None
    blob = serialization.serialize_engine(
        bytes(serialized),
        serialization.TensorRTBlobMetadata(
            io_bindings=[
                serialization.TensorRTIOBinding("pixels", is_input=True),
                serialization.TensorRTIOBinding("features", is_input=False),
            ]
        ),
    )
    metadata = EdgeComponentMetadata(
        component="vision",
        runner="vit",
        outputs=(EdgeOutputSpec(shape=(1, 2, 4, 3), dtype="float16"),),
    )
    pixels = (
        torch.from_numpy(np.arange(24, dtype=np.float16)).reshape(1, 2, 4, 3).cuda()
    )
    program = export_vision(EdgeExecuTorchArtifact(blob, metadata.to_json()), pixels)
    path = tmp_path / "real_vision.pt2"
    torch.export.save(program, path)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch, torch_tensorrt_edge_llm; "
            "program = torch.export.load(sys.argv[1]); "
            "pixels = torch.arange(24, device='cuda', dtype=torch.float16).reshape(1,2,4,3); "
            "torch.testing.assert_close(program.module()(pixels), pixels); "
            "assert 'exporters' not in sys.modules",
            str(path),
        ],
        check=True,
    )


def test_embedded_loader_honors_blob_device(monkeypatch):
    import torch_tensorrt.dynamo.runtime as runtime
    from torch_tensorrt.executorch.serialization import (
        TensorRTBlobMetadata,
        TensorRTIOBinding,
        serialize_engine,
    )

    captured = {}

    def module(**kwargs):
        captured.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(runtime, "TorchTensorRTModule", module)
    blob = serialize_engine(
        b"device-test",
        TensorRTBlobMetadata(
            device_id=2,
            io_bindings=[
                TensorRTIOBinding("input", is_input=True),
                TensorRTIOBinding("output", is_input=False),
            ],
        ),
    )
    metadata = EdgeComponentMetadata(
        component="vision",
        runner="vit",
        outputs=(EdgeOutputSpec(shape=(1, 2, 3), dtype="float32"),),
    )
    payload = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
    ops._get_embedded_engine(payload, metadata)
    assert captured["settings"].device.gpu_id == 2
