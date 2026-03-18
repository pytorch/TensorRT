# Operator support for ExecuTorch TensorRT partitioner: only execute_engine is supported.

from typing import Dict

import torch
from torch.fx.passes.operator_support import OperatorSupportBase


class TensorRTOperatorSupport(OperatorSupportBase):  # type: ignore[misc]
    """Supports only torch.ops.tensorrt.execute_engine for partitioning.

    Used so that TRT-compiled graphs (which already contain execute_engine nodes)
    are partitioned per engine; each partition is then lowered to TensorRTBackend
    which serializes the engine to the same blob format as the TRT runtime.
    """

    def __init__(self) -> None:
        super().__init__()
        self._execute_engine_op = torch.ops.tensorrt.execute_engine.default

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.op != "call_function":
            return False
        return node.target is self._execute_engine_op
