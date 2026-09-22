"""ExecuTorch backend preprocessing for Edge-LLM component delegates."""

from __future__ import annotations

from typing import Any, final

import torch
import torch.fx
from executorch.exir.backend.backend_details import (
    BackendDetails,
    CompileSpec,
    PreprocessResult,
)
from torch.export import ExportedProgram

from .serialization import EdgeComponentMetadata, serialize_edge_component

_VISION_SCHEMA = "edge_llm::vision_tower"


def _schema_name(target: Any) -> str:
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    return ""


def _vision_nodes(program: ExportedProgram) -> list[torch.fx.Node]:
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function" and _schema_name(node.target) == _VISION_SCHEMA
    ]


def _resolve_tensor_constant(
    program: ExportedProgram, node: torch.fx.Node
) -> torch.Tensor:
    if node.op == "get_attr":
        value = getattr(program.graph_module, node.target, None)
    elif node.op == "placeholder":
        target = node.target
        for spec in program.graph_signature.input_specs:
            arg = getattr(spec, "arg", None)
            if arg is not None and getattr(arg, "name", None) == node.name:
                target = spec.target or target
                break
        value = (program.state_dict or {}).get(target)
        if value is None:
            value = (program.constants or {}).get(target)
    else:
        raise ValueError(
            f"Edge-LLM payload must be a constant tensor, got node op {node.op!r}"
        )
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Edge-LLM payload {node.name!r} did not resolve to a tensor")
    return value


def _tensor_bytes(value: torch.Tensor) -> bytes:
    data = value.detach().cpu().contiguous().view(torch.uint8)
    return bytes(memoryview(data.numpy()))


@final
class EdgeLLMBackend(BackendDetails):  # type: ignore[misc]
    """Packs one named Edge component for the native EdgeLLMBackend."""

    @staticmethod
    def preprocess(
        edge_program: ExportedProgram,
        compile_specs: list[CompileSpec],
    ) -> PreprocessResult:
        del compile_specs
        nodes = _vision_nodes(edge_program)
        if len(nodes) != 1:
            raise RuntimeError(
                "EdgeLLMBackend expects exactly one vision_tower node per "
                f"partition, found {len(nodes)}"
            )

        node = nodes[0]
        if len(node.args) != 3:
            raise RuntimeError(
                "edge_llm::vision_tower must receive tensors, trt_blob, and "
                f"metadata_json; found {len(node.args)} arguments"
            )
        trt_blob_node = node.args[1]
        metadata_json = node.args[2]
        if not isinstance(trt_blob_node, torch.fx.Node):
            raise ValueError("vision_tower trt_blob argument is not a graph value")
        if not isinstance(metadata_json, str):
            raise ValueError("vision_tower metadata_json argument must be a string")

        metadata = EdgeComponentMetadata.from_json(metadata_json)
        if metadata.component != "vision" or metadata.runner != "vit":
            raise ValueError(
                "vision_tower payload must select component='vision' and "
                f"runner='vit', got {metadata.component!r}/{metadata.runner!r}"
            )

        trt_blob = _resolve_tensor_constant(edge_program, trt_blob_node)
        payload = serialize_edge_component(_tensor_bytes(trt_blob), metadata)
        return PreprocessResult(processed_bytes=payload)
