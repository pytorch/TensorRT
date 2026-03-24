# Operator support for ExecuTorch TensorRT partitioner: only execute_engine is supported.

from typing import Dict

import torch
from torch.fx.passes.operator_support import OperatorSupportBase


class TensorRTOperatorSupport(OperatorSupportBase):  # type: ignore[misc]
    """Supports torch.ops.tensorrt.no_op_placeholder_for_execute_engine for partitioning.

    Prior to calling to_edge_transform_and_lower, _save_as_executorch replaces
    execute_engine nodes with no_op_placeholder_for_execute_engine so that
    ExecuTorch's edge-lowering passes (which symbolically execute every node) do
    not trip on the Engine custom-class schema check.  The partitioner therefore
    targets the no-op placeholder instead of execute_engine directly.
    """

    def __init__(self) -> None:
        super().__init__()
        self._no_op = torch.ops.tensorrt.no_op_placeholder_for_execute_engine.default

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.op != "call_function":
            return False
        return node.target is self._no_op
