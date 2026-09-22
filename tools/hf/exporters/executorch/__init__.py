"""ExecuTorch lowering for named Edge-LLM component operators.

Imports stay lazy so the core Edge exporter does not require ExecuTorch unless
the caller explicitly requests ExecuTorch lowering.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "EDGE_LLM_ABI_VERSION",
    "EDGE_LLM_MAGIC",
    "EdgeComponentMetadata",
    "EdgeExecuTorchArtifact",
    "EdgeLLMBackend",
    "EdgeLLMPartitioner",
    "EdgeOutputSpec",
    "build_vision_artifact",
    "deserialize_edge_component",
    "lower_vision_to_executorch",
    "save_vision_pte",
    "serialize_edge_component",
]


def __getattr__(name: str) -> Any:
    if name == "EdgeLLMBackend":
        from .backend import EdgeLLMBackend

        return EdgeLLMBackend
    if name == "EdgeLLMPartitioner":
        from .partitioner import EdgeLLMPartitioner

        return EdgeLLMPartitioner
    if name in {"EdgeExecuTorchArtifact", "build_vision_artifact"}:
        from . import artifact

        return getattr(artifact, name)
    if name in {"lower_vision_to_executorch", "save_vision_pte"}:
        from . import vision

        return getattr(vision, name)
    if name in {
        "EDGE_LLM_ABI_VERSION",
        "EDGE_LLM_MAGIC",
        "EdgeComponentMetadata",
        "EdgeOutputSpec",
        "deserialize_edge_component",
        "serialize_edge_component",
    }:
        from . import serialization

        return getattr(serialization, name)
    raise AttributeError(name)
