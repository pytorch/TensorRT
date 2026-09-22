from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

pytest.importorskip("executorch.exir")

# The regular exporters package eagerly registers optional model families. These
# focused lowering tests need only the package-local ops and must not require
# LeRobot or other model stacks.
if "exporters" not in sys.modules:
    exporters_package = types.ModuleType("exporters")
    exporters_package.__path__ = [
        str(Path(__file__).resolve().parents[4] / "tools/hf/exporters")
    ]
    sys.modules["exporters"] = exporters_package

from exporters import ops as edge_ops
from exporters.executorch.artifact import build_vision_artifact
from exporters.executorch.backend import EdgeLLMBackend
from exporters.executorch.partitioner import EdgeLLMPartitioner
from exporters.executorch.serialization import (
    EdgeComponentMetadata,
    EdgeOutputSpec,
    deserialize_edge_component,
    serialize_edge_component,
)
from exporters.executorch.vision import save_vision_pte
from torch_tensorrt.executorch.operator_support import TensorRTOperatorSupport
from torch_tensorrt.executorch.serialization import deserialize_engine


def _vision_metadata() -> EdgeComponentMetadata:
    return EdgeComponentMetadata(
        component="vision",
        runner="vit",
        outputs=(EdgeOutputSpec(shape=(1, 4, 8), dtype="float16"),),
        runner_config={"model_type": "vit"},
    )


class _VisionModule(nn.Module):
    def __init__(self, metadata_json: str, trt_blob: bytes) -> None:
        super().__init__()
        self.metadata_json = metadata_json
        self.register_buffer(
            "trt_blob", torch.tensor(list(trt_blob), dtype=torch.uint8)
        )

    def forward(self, pixel_values):
        return edge_ops.call_vision_tower(
            self.trt_blob, self.metadata_json, pixel_values
        )[0]


@pytest.mark.unit
def test_edge_component_payload_round_trip():
    metadata = _vision_metadata()
    trt_blob = b"TR01-test-engine"

    payload = serialize_edge_component(trt_blob, metadata)
    restored_blob, restored_metadata = deserialize_edge_component(payload)

    assert restored_blob == trt_blob
    assert restored_metadata == metadata


@pytest.mark.unit
def test_vision_tower_fake_uses_embedded_output_spec():
    metadata = _vision_metadata()
    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
    with fake_mode:
        outputs = torch.ops.edge_llm.vision_tower.default(
            [torch.empty(1, 16, 16, 3, device="cuda")],
            torch.empty(16, dtype=torch.uint8),
            metadata.to_json(),
        )

    assert len(outputs) == 1
    assert tuple(outputs[0].shape) == (1, 4, 8)
    assert outputs[0].dtype == torch.float16
    assert outputs[0].device.type == "cuda"


@pytest.mark.unit
def test_edge_llm_partitioner_tags_only_vision_operator():
    metadata = _vision_metadata()
    module = _VisionModule(metadata.to_json(), b"TR01-test-engine").eval()
    exported = torch.export.export(module, (torch.randn(1, 16, 16, 3, device="cuda"),))

    result = EdgeLLMPartitioner().partition(exported)
    assert len(result.partition_tags) == 1
    spec = next(iter(result.partition_tags.values()))
    assert spec.backend_id == EdgeLLMBackend.__name__

    vision_node = next(
        node
        for node in result.tagged_exported_program.graph_module.graph.nodes
        if node.op == "call_function"
        and hasattr(node.target, "_schema")
        and node.target._schema.name == "edge_llm::vision_tower"
    )
    assert vision_node.meta["delegation_tag"] in result.partition_tags
    assert not TensorRTOperatorSupport().is_node_supported({}, vision_node)


@pytest.mark.unit
def test_edge_llm_backend_wraps_vision_component():
    metadata = _vision_metadata()
    trt_blob = b"TR01-test-engine"
    module = _VisionModule(metadata.to_json(), trt_blob).eval()
    exported = torch.export.export(module, (torch.randn(1, 16, 16, 3, device="cuda"),))

    result = EdgeLLMBackend.preprocess(exported, [])
    restored_blob, restored_metadata = deserialize_edge_component(
        result.processed_bytes
    )

    assert restored_blob == trt_blob
    assert restored_metadata == metadata


@pytest.mark.unit
def test_vision_metadata_rejects_wrong_runner():
    metadata = EdgeComponentMetadata(
        component="vision",
        runner="qwen_vit",
        outputs=(EdgeOutputSpec(shape=(1, 4, 8), dtype="float16"),),
    )
    with pytest.raises(ValueError, match="runner='vit'"):
        edge_ops._vision_metadata(metadata.to_json())


@pytest.mark.unit
def test_build_vision_artifact_wraps_saved_engine(tmp_path):
    engine_dir = tmp_path / "vision"
    engine_dir.mkdir()
    (engine_dir / "visual.engine").write_bytes(b"serialized-vision-engine")
    (engine_dir / "config.json").write_text("""{
  "model_type": "vit",
  "component": "vision",
  "engine_file": "visual.engine",
  "input_names": ["pixel_values"],
  "output_names": ["visual_embeds"],
  "outputs": [{"shape": [4, 8], "dtype": "torch.float16"}],
  "input_layout": "hwc",
  "input_dtype": "float16"
}
""")

    artifact = build_vision_artifact(engine_dir, device_id=2)
    nested_blob, edge_metadata = deserialize_edge_component(
        serialize_edge_component(
            artifact.trt_blob,
            EdgeComponentMetadata.from_json(artifact.edge_metadata_json),
        )
    )
    engine, trt_metadata = deserialize_engine(nested_blob)

    assert engine == b"serialized-vision-engine"
    assert edge_metadata.runner_config["input_layout"] == "hwc"
    assert trt_metadata.device_id == 2
    assert [binding.name for binding in trt_metadata.io_bindings] == [
        "pixel_values",
        "visual_embeds",
    ]


@pytest.mark.unit
def test_save_vision_pte_contains_edge_backend(tmp_path):
    engine_dir = tmp_path / "vision"
    engine_dir.mkdir()
    (engine_dir / "visual.engine").write_bytes(b"serialized-vision-engine")
    (engine_dir / "config.json").write_text("""{
  "model_type": "vit",
  "component": "vision",
  "engine_file": "visual.engine",
  "input_names": ["pixel_values"],
  "output_names": ["visual_embeds"],
  "outputs": [{"shape": [4, 8], "dtype": "torch.float16"}],
  "input_layout": "hwc",
  "input_dtype": "float16"
}
""")
    artifact = build_vision_artifact(engine_dir)
    pte = tmp_path / "vision.pte"

    save_vision_pte(
        artifact,
        torch.randn(1, 16, 16, 3, device="cuda", dtype=torch.float16),
        pte,
    )

    from executorch.exir._serialize._program import deserialize_pte_binary

    program = deserialize_pte_binary(pte.read_bytes()).program
    delegate_ids = [
        delegate.id for plan in program.execution_plan for delegate in plan.delegates
    ]
    assert delegate_ids == ["EdgeLLMBackend"]
