import logging
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from torch.fx.passes.splitter_base import (
    Subgraph,
    _SplitterBase,
    _SplitterSettingBase,
    FxNetAccNodesFinder,
    FxNetAccFusionsFinder,
)
import torch.fx.passes.operator_support as ops
from torch.fx.passes.tools_common import NodeSet, CALLABLE_NODE_OPS
from torch.fx.node import Target

from torch_tensorrt.dynamo.conversion.converter_registry import ConverterRegistry
from .common import DEFAULT_SINGLE_NODE_PARTITIONS
from torch_tensorrt.dynamo._defaults import MIN_BLOCK_SIZE

from torch_tensorrt.dynamo import DYNAMO_CONVERTERS as CONVERTERS


logger = logging.getLogger(__name__)


class OpSupportTester(ops.OperatorSupportBase):
    """Class to determine whether operators within a module are supported"""

    def __init__(self, torch_executed_ops: Sequence[Target] = set()) -> None:
        super().__init__()

        # Initialize sets of supported/unsupported operators
        self.supported_operators = {}
        self.unsupported_operators = {}
        self.torch_executed_ops = torch_executed_ops

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        node_name = ConverterRegistry.qualified_name_or_str(node.target)

        if node in CONVERTERS and node_name not in self.torch_executed_ops:
            # If node is a proper, supported computational node, store the operator
            if not node.is_impure():
                if node_name not in self.supported_operators:
                    self.supported_operators[node_name] = 1
                else:
                    self.supported_operators[node_name] += 1

            return True
        else:
            if not node.is_impure():
                if node_name not in self.unsupported_operators:
                    self.unsupported_operators[node_name] = 1
                else:
                    self.unsupported_operators[node_name] += 1

            return False

    def print_support_overview(self, num_trt_blocks: Optional[int] = None):
        if num_trt_blocks is not None:
            logger.debug(
                f"\nNumber of TensorRT-Accelerated Engines Generated: {num_trt_blocks}"
            )

        # Reformat support messages for debugger to print node overview as a single string
        supported_nodes_str = "\nSupported Nodes:\n"
        for node_name, count in self.supported_operators.items():
            supported_nodes_str += f"- {node_name} + Operator Count: {count}\n"

        logger.debug(supported_nodes_str)

        if self.unsupported_operators:
            unsupported_nodes_str = "\nUnsupported or Excluded Nodes:\n"
            for node_name, count in self.unsupported_operators.items():
                unsupported_nodes_str += f"- {node_name} + Operator Count: {count}\n"

            logger.debug(unsupported_nodes_str)
        else:
            logger.debug("\nAll Nodes Supported\n")


class TRTPartitioner(_SplitterBase):
    """Partitioner to split an FX graph into subgraphs based on operator support

    Adapted from, and modified for the Torch-TensorRT Dynamo case:
    https://github.com/pytorch/pytorch/blob/93f538db355ea10c684a57f7a632ed03292ef98f/torch/fx/passes/splitter_base.py#L256C9-L871

    Args:
        module: FX GraphModule to partition
        operator_support: OperatorSupport class describing allowed operators
        allowed_single_node_partition_ops: Nodes which can be included in single-node partitons.
            Generally useful for module-level exclusion ops which are intensive despite being single functions
        min_block_size: Minimum number of computational operators per block
    Returns:
        torch.fx.GraphModule
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: ops.OperatorSupportBase,
        allowed_single_node_partition_ops: Optional[
            Sequence[str]
        ] = DEFAULT_SINGLE_NODE_PARTITIONS,
        min_block_size: int = MIN_BLOCK_SIZE,
    ):
        """
        Preprocesses graph before splitting:
        - finds nodes supported by ACC,
        - finds fusion groups for ACC nodes having non-tensor IO,
        - builds a graph of direct dependencies,
        - builds a map of fused nodes to their fusions.
        As a result we get self.acc_nodes, self.deps and self.fusions.
        """
        assert isinstance(module, torch.fx.GraphModule)

        self.module = module

        self.settings = _SplitterSettingBase(
            min_acc_module_size=min_block_size,
            allow_non_tensor=True,
        )
        self.operator_support = operator_support

        # Get all accelerated nodes based on operator support conditions
        self.acc_nodes = FxNetAccNodesFinder(
            self.module, self.operator_support, self.settings.allow_non_tensor
        )()

        if self.settings.skip_fusion:
            self.fusions = {}
        else:
            self.fusions = FxNetAccFusionsFinder(module, set(self.acc_nodes))()

        # Modify deps to add more deps for fused nodes
        self.deps = self.find_deps()
        self.update_deps_for_fusions()

        self.non_acc_submodule_name = "_run_on_gpu_"
        self._node_submodule_map: Dict[str, str] = {}

        self.num_trt_accelerated_subgraphs = None
        self.allowed_single_node_partition_ops = allowed_single_node_partition_ops

    def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent GPU submodules.
        """
        result: List[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                if len(subgraph.nodes) >= self.settings.min_acc_module_size or any(
                    ConverterRegistry.qualified_name_or_str(node.target)
                    in self.allowed_single_node_partition_ops
                    for node in subgraph.nodes
                ):
                    result.append(subgraph)
                else:
                    logger.debug(
                        "Eliminating acc subgraph because it's smaller than the threshold: "
                        f"{len(subgraph.nodes)} < {self.settings.min_acc_module_size}"
                    )
                    if result:
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        subgraph.is_acc = False
                        result.append(subgraph)
            else:
                if result and not result[-1].is_acc:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    result.append(subgraph)
        return result

    def partition_graph(self) -> torch.fx.GraphModule:
        """Partitions the GraphModule into subgraphs based on operator support

        Returns a GraphModule with submodules for each segment
        """
        # Delegate nodes based on operator coverage
        subgraphs = self.put_nodes_into_subgraphs()

        # Remove segments smaller than the block size (with exceptions)
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)

        # Set the number of TRT engines to be generated
        self.num_trt_accelerated_subgraphs = len([s for s in subgraphs if s.is_acc])

        # Tag the accelerated nodes and split the graph accordingly
        self.tag(subgraphs)
        return self.split()

    def starter_nodes(self) -> Tuple[NodeSet, NodeSet]:
        """Generates starter nodes for partitioning + segmentation"""
        # Starter accelerated nodes are all callable accelerated ops
        starter_acc_nodes = {
            node for node in self.acc_nodes if node.op in CALLABLE_NODE_OPS
        }

        # Started non-accelerated nodes are the rest of the callable nodes
        starter_non_acc_nodes = {
            node
            for node in self.module.graph.nodes
            if (node not in starter_acc_nodes and node.op in CALLABLE_NODE_OPS)
        }

        return starter_non_acc_nodes, starter_acc_nodes


def partition(
    gm: torch.fx.GraphModule,
    verbose: bool = True,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Sequence[Target] = set(),
) -> torch.fx.GraphModule:
    """Partition an FX GraphModule with aten ops into TRT engines
    Partitioning is based on converter operator support

    Args:
        gm: FX GraphModule to partition
        verbose: Bool representing whether to print operator support
        min_block_size: Minimum number of operators per TRT-Engine Block
        torch_executed_ops: Sequence of operations to run in Torch, regardless of converter coverage
    Returns:
        torch.fx.GraphModule
    """
    # Ensure graph is clean prior to partitioning
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    # Construct
    supported_ops = OpSupportTester(torch_executed_ops=torch_executed_ops)
    partitioner = TRTPartitioner(gm, supported_ops, min_block_size=min_block_size)

    partitioned_graph = partitioner.partition_graph()

    if verbose:
        supported_ops.print_support_overview(partitioner.num_trt_accelerated_subgraphs)

    return partitioned_graph
