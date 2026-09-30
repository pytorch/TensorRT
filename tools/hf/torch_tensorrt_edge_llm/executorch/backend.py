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
from torch_tensorrt_edge_llm.serialization import (
    EdgeComponentMetadata,
    serialize_edge_component,
)

_COMPONENT_SCHEMAS = {
    "tensorrt_edge_llm::vision_tower": (1, 2, "vision", "vit", 1),
    "tensorrt_edge_llm::llm_prefill": (3, 4, "language", "pi05_prefill", 3),
    "tensorrt_edge_llm::action_expert": (6, 7, "action", "pi05_action", 1),
}


def _schema_name(target: Any) -> str:
    if hasattr(target, "_schema"):
        return str(target._schema.name)
    return ""


def _component_nodes(program: ExportedProgram) -> list[torch.fx.Node]:
    return [
        node
        for node in program.graph_module.graph.nodes
        if node.op == "call_function"
        and _schema_name(node.target) in _COMPONENT_SCHEMAS
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
        nodes = _component_nodes(edge_program)
        if len(nodes) != 1:
            raise RuntimeError(
                "EdgeLLMBackend expects exactly one component node per "
                f"partition, found {len(nodes)}"
            )

        node = nodes[0]
        payload_index, metadata_index, component, runner, num_outputs = (
            _COMPONENT_SCHEMAS[_schema_name(node.target)]
        )
        if len(node.args) != metadata_index + 1:
            raise RuntimeError(
                f"{_schema_name(node.target)} must receive its inputs, trt_blob, and "
                f"metadata_json; found {len(node.args)} arguments"
            )
        trt_blob_node = node.args[payload_index]
        metadata_json = node.args[metadata_index]
        if not isinstance(trt_blob_node, torch.fx.Node):
            raise ValueError("Component trt_blob argument is not a graph value")
        if not isinstance(metadata_json, str):
            raise ValueError("Component metadata_json argument must be a string")

        metadata = EdgeComponentMetadata.from_json(metadata_json)
        if (
            metadata.component != component
            or metadata.runner != runner
            or len(metadata.outputs) != num_outputs
        ):
            raise ValueError(
                f"Operator payload must select component={component!r}, runner={runner!r}, "
                f"and {num_outputs} outputs; got {metadata.component!r}/{metadata.runner!r} "
                f"with {len(metadata.outputs)} outputs"
            )

        trt_blob = _resolve_tensor_constant(edge_program, trt_blob_node)
        payload = serialize_edge_component(_tensor_bytes(trt_blob), metadata)
        return PreprocessResult(processed_bytes=payload)
