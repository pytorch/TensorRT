from typing import Dict

import torch

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport

from torch_tensorrt.fx.converter_registry import CONVERTERS


MAX_NUM_TRT_ENGINES = 10


class TorchTensorRTOperatorSupport(OperatorSupport):
    """Class to determine whether the aten operators have converters"""

    def __init__(self, support_dict=None):
        super().__init__(support_dict)

        # Initialize sets of supported/unsupported operators
        self.supported_operators = set()
        self.unsupported_operators = set()

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        if node.target in CONVERTERS.keys():
            # If node is a proper computational node, store the operator
            if not node.is_impure():
                node_name = node._pretty_print_target(node.target)
                self.supported_operators.add(node_name)

            return True
        else:
            if not node.is_impure():
                node_name = node._pretty_print_target(node.target)
                self.unsupported_operators.add(node_name)

            return False

    def print_support_overview(self, num_trt_blocks=None):
        if num_trt_blocks is not None:
            print(f"Number of TensorRT-Accelerated Subgraphs: {num_trt_blocks}\n")

        print("Supported Nodes:")
        for node_name in self.supported_operators:
            print(node_name)

        print("\nUnsupported Nodes:")
        for node_name in self.unsupported_operators:
            print(node_name)


def partition(gm: torch.fx.GraphModule, verbose=True):
    """Partition an FX GraphModule with aten ops into TRT engines
    Partitioning is based on operator support
    """
    supported_ops = TorchTensorRTOperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, supported_ops)

    # Determine partitions, and raise error if the degree of partitioning
    # exceeds a specified threshold
    partitions = partitioner.propose_partitions()
    num_blocks = len(partitions)
    if num_blocks > MAX_NUM_TRT_ENGINES:
        raise AssertionError(
            f"The graph module has {num_blocks} TRT Engines which is larger than the "
            + f"threshold={MAX_NUM_TRT_ENGINES}. Falling back to non-TRT module."
        )

    # Fuse partitions and display overview of supported/unsupported operators
    fused_graph = partitioner.fuse_partitions(partitions)
    num_blocks = len(partitions)

    if verbose:
        supported_ops.print_support_overview(num_blocks)

    return fused_graph
