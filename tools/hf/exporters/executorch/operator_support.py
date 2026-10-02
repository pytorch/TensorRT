"""Operator support for the Edge-LLM ExecuTorch partitioner."""

from __future__ import annotations

from typing import Dict

import torch
from torch.fx.passes.operator_support import OperatorSupportBase


class EdgeLLMOperatorSupport(OperatorSupportBase):  # type: ignore[misc]
    """Recognizes named Edge component operators, starting with vision."""

    _SUPPORTED_OPS = frozenset({"edge_llm::vision_tower"})

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        del submodules
        if node.op != "call_function" or not hasattr(node.target, "_schema"):
            return False
        return node.target._schema.name in self._SUPPORTED_OPS
