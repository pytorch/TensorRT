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

    _SUPPORTED_OPS = frozenset(
        [
            "tensorrt::execute_engine",
            "tensorrt::no_op_placeholder_for_execute_engine",
        ]
    )

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.op != "call_function":
            return False
        # After ExecuTorch edge lowering the op is wrapped as EdgeOpOverload, so
        # compare by schema name rather than by op object identity.
        target = node.target
        if hasattr(target, "_schema"):
            return target._schema.name in self._SUPPORTED_OPS
        return False
