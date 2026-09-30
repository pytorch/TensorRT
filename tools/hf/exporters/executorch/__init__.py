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
    "build_action_artifact",
    "build_language_artifact",
    "build_vision_artifact",
    "deserialize_edge_component",
    "lower_action_to_executorch",
    "lower_language_decode_to_executorch",
    "lower_language_prefill_to_executorch",
    "lower_vision_to_executorch",
    "save_language_decode_pte",
    "save_language_prefill_pte",
    "save_action_pte",
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
    if name in {
        "EdgeExecuTorchArtifact",
        "build_action_artifact",
        "build_language_artifact",
        "build_vision_artifact",
    }:
        from . import artifact

        return getattr(artifact, name)
    if name in {"lower_action_to_executorch", "save_action_pte"}:
        from . import action

        return getattr(action, name)
    if name in {
        "lower_language_decode_to_executorch",
        "save_language_decode_pte",
    }:
        from . import decode

        return getattr(decode, name)
    if name in {
        "lower_language_prefill_to_executorch",
        "save_language_prefill_pte",
    }:
        from . import language

        return getattr(language, name)
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
