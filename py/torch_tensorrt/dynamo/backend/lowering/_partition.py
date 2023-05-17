from typing import Dict, Optional, Sequence

import torch

from torch_tensorrt.dynamo.backend._defaults import MAX_NUM_TRT_ENGINES
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport

from torch_tensorrt.fx.converter_registry import CONVERTERS


class TorchTensorRTOperatorSupport(OperatorSupport):
    """Class to determine whether operators within a module are supported"""

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

    def print_support_overview(self, num_trt_blocks: Optional[int] = None):
        if num_trt_blocks is not None:
            print(f"\nNumber of TensorRT-Accelerated Subgraphs: {num_trt_blocks}")

        print("\nSupported Nodes:")
        for node_name in self.supported_operators:
            print("-", node_name)

        if len(self.unsupported_operators) != 0:
            print("\nUnsupported Nodes:")
            for node_name in self.unsupported_operators:
                print("-", node_name)
            print("\n")
        else:
            print("\nAll Nodes Supported\n")


def partition(
    gm: torch.fx.GraphModule,
    verbose: bool = True,
    max_num_trt_engines: int = MAX_NUM_TRT_ENGINES,
) -> torch.fx.GraphModule:
    """Partition an FX GraphModule with aten ops into TRT engines
    Partitioning is based on converter operator support

    Args:
        gm: FX GraphModule to partition
        verbose: Bool representing whether to print operator support
        max_num_trt_engines: Maximum number of allowed TRT engines in partitioning
    Returns:
        torch.fx.GraphModule
    """
    supported_ops = TorchTensorRTOperatorSupport()
    partitioner = CapabilityBasedPartitioner(gm, supported_ops)

    # Determine partitions, and raise error if the degree of partitioning
    # exceeds a specified threshold
    partitions = partitioner.propose_partitions()
    num_blocks = len(partitions)
    if num_blocks > max_num_trt_engines:
        raise AssertionError(
            f"The graph module has {num_blocks} TRT Engines which is larger than the "
            + f"threshold={max_num_trt_engines}. Falling back to non-TRT module."
        )

    # Fuse partitions and display overview of supported/unsupported operators
    fused_graph = partitioner.fuse_partitions(partitions)
    num_blocks = len(partitions)

    if verbose:
        supported_ops.print_support_overview(num_blocks)

    return fused_graph


def get_submod_inputs(
    mod: torch.fx.GraphModule,
    submod: torch.fx.GraphModule,
    inputs: Sequence[torch.Tensor],
) -> Sequence[torch.Tensor]:
    """Helper function to get inputs to a Torch submodule

    Args:
        mod: Parent FX GraphModule
        submod: Child FX GraphModule
        inputs: Sample inputs to parent module
    Returns:
        Sequence of Tensors representing inputs to child module
    """
    acc_inputs = None

    def get_input(self, inputs):
        nonlocal acc_inputs
        acc_inputs = inputs

    handle = submod.register_forward_pre_hook(get_input)
    mod(*inputs)
    handle.remove()
    return acc_inputs
