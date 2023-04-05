from typing import Dict
import logging

import torch

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport

from torch_tensorrt.fx.conversion.converter_registry import CONVERTERS

log = logging.getLogger(__name__)


class TorchTensorRTOperatorSupport(OperatorSupport):
    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.target in CONVERTERS.keys():
            print(f"{node.target} is supported")
            return True
        else:
            print(f"{node.target} is not supported")
            return False


def partition(gm: torch.fx.GraphModule):
    supported_ops = TorchTensorRTOperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, supported_ops)
    fused_graph = partitioner.partition_and_fuse()
    return fused_graph
