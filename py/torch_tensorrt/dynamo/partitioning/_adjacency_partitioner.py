from ast import Assert
import logging
from typing import Collection, Dict, List, Optional, Tuple

import psutil
import torch
import torch.fx.passes.operator_support as ops
from torch.fx.node import Target
from torch.fx.passes.splitter_base import (
    FxNetAccFusionsFinder,
    FxNetAccNodesFinder,
    Subgraph,
    _SplitterBase,
    _SplitterSettingBase,
)
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS, NodeSet
from torch_tensorrt.dynamo._defaults import (
    MIN_BLOCK_SIZE,
    REQUIRE_FULL_COMPILATION,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS as CONVERTERS,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    ConverterRegistry,
)

logger = logging.getLogger(__name__)


class OpSupportTester(ops.OperatorSupportBase):  # type: ignore
    """Class to determine whether operators within a module are supported"""

    def __init__(self, torch_executed_ops: Collection[Target] = set()) -> None:
        super().__init__()

        # Initialize sets of supported/unsupported operators
        self.supported_operators: Dict[str, int] = {}
        self.unsupported_operators: Dict[str, int] = {}
        self.torch_executed_ops = torch_executed_ops

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        node_name = ConverterRegistry.qualified_name_or_str(node.target)

        if (
            (node in CONVERTERS or node.op == "get_attr")
            and node_name not in self.torch_executed_ops
            and node.target not in self.torch_executed_ops
        ):
            # If node is a proper, supported computational node, store the operator
            if not node.is_impure() and node.op != "get_attr":
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

    def print_support_overview(self, num_trt_blocks: Optional[int] = None) -> None:
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


class TRTPartitioner(_SplitterBase):  # type: ignore
    """Partitioner to split an FX graph into subgraphs based on operator support

    Adapted from, and modified for the Torch-TensorRT Dynamo case:
    https://github.com/pytorch/pytorch/blob/93f538db355ea10c684a57f7a632ed03292ef98f/torch/fx/passes/splitter_base.py#L256C9-L871

    Args:
        module: FX GraphModule to partition
        operator_support: OperatorSupport class describing allowed operators
        allowed_single_node_partition_ops: Nodes which can be included in single-node partitions.
            Generally useful for module-level exclusion ops which are intensive despite being single functions
        min_block_size: Minimum number of computational operators per block
        require_full_compilation: Require that all computational operators be run in TRT
    Returns:
        torch.fx.GraphModule
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: ops.OperatorSupportBase,
        allowed_single_node_partition_ops: Optional[Collection[str]] = None,
        min_block_size: int = MIN_BLOCK_SIZE,
        require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
        return_tuple: bool = False,
        skip_fusion: bool = False,
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
            skip_fusion=skip_fusion,
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

        self.num_trt_accelerated_subgraphs: Optional[int] = None
        self.allowed_single_node_partition_ops = allowed_single_node_partition_ops
        self.require_full_compilation = require_full_compilation
        self._return_tuple = return_tuple

    def remove_small_acc_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """
        This pass finds ACC submodules with less than specified size and merges
        them with adjacent GPU submodules.
        """
        result: List[Subgraph] = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                if (
                    len(subgraph.nodes) >= self.settings.min_acc_module_size
                    or self.require_full_compilation
                    or (
                        self.allowed_single_node_partition_ops is not None
                        and any(
                            ConverterRegistry.qualified_name_or_str(node.target)
                            in self.allowed_single_node_partition_ops
                            for node in subgraph.nodes
                        )
                    )
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

        # A graph is fully supported if there is a single partition and all operators are supported/convertible
        full_support = len([s for s in subgraphs if s.is_acc]) == 1 and not getattr(
            self.operator_support, "unsupported_operators", True
        )

        if not full_support and self.require_full_compilation:
            raise AssertionError(
                "require_full_compilation=True was specified, but model is not fully supported"
            )

        if (
            full_support
            and self.require_full_compilation
            and self.settings.min_acc_module_size != MIN_BLOCK_SIZE
        ):
            logger.warning(
                "Detected both require_full_compilation and min_block_size compilation "
                "arguments were specified. Disregarding min_block_size argument for "
                "fully supported model."
            )

        # Remove segments smaller than the block size (with exceptions)
        subgraphs = self.remove_small_acc_subgraphs(subgraphs)

        # num_of_break = self.calculate_num_of_break(subgraphs)
        subgraphs = self.break_subgraphs_by_node(subgraphs, num_of_break=5)

        # Set the number of TRT engines to be generated
        self.num_trt_accelerated_subgraphs = len([s for s in subgraphs if s.is_acc])

        # Tag the accelerated nodes and split the graph accordingly
        print([len(s.nodes) for s in subgraphs])
        self.tag(subgraphs)

        for s in subgraphs:
            print(s.nodes)

        gm = self.split()
        self.weight_visited_nodes = set()
        [self.size_of_subgraph(s) for s in subgraphs]
        

        return gm
    
    def calculate_num_of_break(self, subgraphs: List[Subgraph]) -> int:
        """
        This function calculates the break period based on the number of subgraphs.
        """
        rss = psutil.Process().memory_info().rss
        available_rss = psutil.virtual_memory().available
        num_of_graphs = len(subgraphs)
        if rss < available_rss * 0.3:
            num_of_graphs = 1
        elif rss < available_rss * 0.5:
            num_of_graphs = 2
        elif rss < available_rss:
            num_of_graphs = 4
        elif rss < available_rss * 1.5:
            num_of_graphs = 8
        elif rss < available_rss * 2:
            num_of_graphs = 16
        else:
            num_of_graphs = 32

        return max(
            1, num_of_graphs // ((len(subgraphs) + 1) // 2)
        )  # If there are already graph breaks, for each TRT subgraph, we break for a few times.


    def break_subgraphs_by_node(
        self, subgraphs: List[Subgraph], num_of_break: int = 1
    ) -> List[Subgraph]:
        """
        This function breaks the subgraphs into smaller subgraphs at the specified frequency to save CPU memory.
        """
        op_to_break = "add."
        num_of_sdpa_node = len(
            [node for node in self.acc_nodes if op_to_break in str(node.target)]
        )
        break_period = num_of_sdpa_node // num_of_break + 1
        current_break_idx = 0
        current_num_break = 0
        new_subgraphs = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                for i, node in enumerate(subgraph.nodes):
                    if op_to_break in str(node.target):
                        current_num_break += 1
                        if current_num_break % break_period != 0:
                            continue
                        new_subgraphs.append(
                            Subgraph(
                                is_acc=True,
                                nodes=subgraph.nodes[current_break_idx : i + 1],
                                device_ordinal=subgraph.device_ordinal,
                            )
                        )
                        current_break_idx = i + 1
                new_subgraphs.append(
                    Subgraph(
                        is_acc=True,
                        nodes=subgraph.nodes[current_break_idx:],
                        device_ordinal=subgraph.device_ordinal,
                    )
                )
            else:
                new_subgraphs.append(subgraph)

        new_subgraphs = self.validate_and_correct_subgraphs(new_subgraphs)
        
        return new_subgraphs

    def break_subgraphs(
        self, subgraphs: List[Subgraph], num_of_break: int = 1
    ) -> List[Subgraph]:
        """
        This function breaks the subgraphs into smaller subgraphs at the specified frequency to save CPU memory.
        """
        break_pos = [0, 100, 200, 300, 400]
        current_break_idx = 0
        new_subgraphs = []
        for subgraph in subgraphs:
            if subgraph.is_acc:
                for i, node in enumerate(subgraph.nodes):
                    if i in break_pos:

                        new_subgraphs.append(
                            Subgraph(
                                is_acc=True,
                                nodes=subgraph.nodes[current_break_idx : i + 1],
                                device_ordinal=subgraph.device_ordinal,
                            )
                        )
                        current_break_idx = i + 1
                new_subgraphs.append(
                    Subgraph(
                        is_acc=True,
                        nodes=subgraph.nodes[current_break_idx:],
                        device_ordinal=subgraph.device_ordinal,
                    )
                )
            else:
                new_subgraphs.append(subgraph)

        new_subgraphs = self.validate_and_correct_subgraphs(new_subgraphs)
        return new_subgraphs

    def size_of_subgraph(self, subgraph: Subgraph) -> int:
        """
        This function calculates the size of the subgraph.
        """
        stack = subgraph.nodes.copy()
        size = 0
        while stack:
            node = stack.pop()
            if node in self.weight_visited_nodes:
                continue
            self.weight_visited_nodes.add(node)
            if node.op == "get_attr":
                weight = self.module.state_dict()[node.target]
                size += weight.numel() * weight.element_size()
                self.weight_visited_nodes.add(node)
                continue
            for input_node in node._input_nodes: 
                if input_node not in self.weight_visited_nodes:
                    stack.append(input_node)
        print(size)
        return size

    def validate_and_correct_subgraphs(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """
        This function validates the subgraphs by checking if the subgraphs are valid, and corrects the subgraphs if they are not valid.
        """
        visited_nodes = {}
        print([len(s.nodes) for s in subgraphs])
        for i, subgraph in enumerate(subgraphs):
            if i == 0:
                for node in subgraph.nodes:
                    visited_nodes[node] = i
                visited_nodes[subgraph.nodes[-1]] = i + 1
                continue


            elif not subgraph.is_acc:
                for node in subgraph.nodes:
                    visited_nodes[subgraph.nodes[-1]] = i + 1
                continue

            else:
                to_remove_nodes = []
                for j, node in enumerate(subgraph.nodes):
                    if j == len(subgraph.nodes) - 1:
                        visited_nodes[node] = i + 1
                        continue
                    subgraph_idx = 0
                    for dep in self.deps[node]:
                        if dep in visited_nodes:
                            subgraph_idx = max(subgraph_idx, visited_nodes[dep])
                        else:
                            raise ValueError(f"Node {node} have a dependency that is not covered in the previous subgraphs. This is caused by a invalid subgraph segmentation.")
                    if subgraph_idx != i:
                        subgraphs[subgraph_idx].nodes.append(node)
                        to_remove_nodes.append(node)
                    visited_nodes[node] = subgraph_idx
                for node in to_remove_nodes:
                    subgraph.nodes.remove(node)
        
        return subgraphs
                    
                    

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
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Collection[Target] = set(),
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
    skip_fusion: bool = False,
) -> Tuple[torch.fx.GraphModule, OpSupportTester]:
    """Partition an FX GraphModule with aten ops into TRT engines
    Partitioning is based on converter operator support

    Args:
        gm: FX GraphModule to partition
        min_block_size: Minimum number of operators per TRT-Engine Block
        torch_executed_ops: Collection of operations to run in Torch, regardless of converter coverage
        require_full_compilation: Require that all computational operators be run in TRT
        skip_fusion: Skip fusions found by FxNetAccFusionsFinder
    Returns:
        torch.fx.GraphModule, OpSupportTester
    """
    # Ensure graph is clean prior to partitioning
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    # Construct
    supported_ops = OpSupportTester(torch_executed_ops=torch_executed_ops)
    partitioner = TRTPartitioner(
        gm,
        supported_ops,
        min_block_size=min_block_size,
        require_full_compilation=require_full_compilation,
        skip_fusion=skip_fusion,
    )

    partitioned_graph = partitioner.partition_graph()

    supported_ops.print_support_overview(partitioner.num_trt_accelerated_subgraphs)

    return partitioned_graph, supported_ops
