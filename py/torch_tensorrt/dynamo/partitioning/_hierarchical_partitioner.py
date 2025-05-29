import logging
from typing import Collection, Dict, List, Optional, Set, Tuple

import torch
import torch.fx.passes.operator_support as ops
from torch._ops import OpOverload
from torch.fx.node import Target, _get_qualified_name
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS, NodeList, NodeSet
from torch_tensorrt.dynamo._defaults import (
    DEBUG,
    MIN_BLOCK_SIZE,
    REQUIRE_FULL_COMPILATION,
)
from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_ATEN_CONVERTERS,
    ConverterRegistry,
)
from torch_tensorrt.dynamo.partitioning.splitter_base import (
    FxNetAccFusionsFinder,
    FxNetAccNodesFinder,
    Subgraph,
    _SplitterBase,
    _SplitterSettingBase,
)

logger = logging.getLogger(__name__)


class BackendOpSupportTester(ops.OperatorSupportBase):  # type: ignore
    """Class to determine whether operators are supported by specific backends"""

    def __init__(
        self,
        backend_support_map: Dict[str, Set[OpOverload]],
        backend_priority: List[str],
        torch_executed_ops: Collection[Target] = set(),
    ) -> None:
        super().__init__()

        # Initialize sets of supported/unsupported operators
        self.supported_operators: Dict[str, int] = {}
        self.unsupported_operators: Dict[str, int] = {}
        self.torch_executed_ops = torch_executed_ops
        # Map of backend names to sets of supported operators
        self.backend_support_map = backend_support_map
        # Ordered list of backend names, from highest to lowest priority
        self.backend_priority = backend_priority

    def is_node_supported(
        self, submodules: Dict[str, torch.nn.Module], node: torch.fx.Node
    ) -> Tuple[bool, Optional[str]]:
        node_name = ConverterRegistry.qualified_name_or_str(node.target)

        for i, backend_name in enumerate(self.backend_priority):
            supported_ops = self.backend_support_map.get(backend_name, set())
            supported_ops = {_get_qualified_name(op) for op in supported_ops}

            if (
                (node_name in supported_ops or node.op == "get_attr")
                and node_name not in self.torch_executed_ops
                and node.target not in self.torch_executed_ops
            ):
                # If node is a proper, supported computational node, store the operator
                if not node.is_impure() and node.op != "get_attr":
                    if node_name not in self.supported_operators:
                        self.supported_operators[f"{backend_name}_{node_name}"] = 1
                    else:
                        self.supported_operators[f"{backend_name}_{node_name}"] += 1

                return True, backend_name
            else:
                if i == len(self.backend_priority) - 1 and not node.is_impure():
                    if node_name not in self.unsupported_operators:
                        self.unsupported_operators[node_name] = 1
                    else:
                        self.unsupported_operators[node_name] += 1

        return False, None

    def print_support_overview(self, num_acc_subgraphs: Optional[int] = None) -> None:
        if num_acc_subgraphs is not None:
            logger.debug(
                f"\nNumber of Accelerated Subgraphs Generated: {num_acc_subgraphs}"
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


class HierarchicalTRTPartitioner(_SplitterBase):
    """Hierarchical partitioner to split an FX graph into subgraphs based on backend priority

    This partitioner extends the TRTPartitioner of adjacency_partitioner with backend priority awareness,
    allowing different parts of the model to be executed on different backends based on
    operator support and priority ordering.

    Args:
        module: FX GraphModule to partition
        operator_support: OperatorSupport class describing allowed operators
        backend_support_map: Dictionary mapping backend names to sets of supported operators
        backend_priority: Ordered list of backend names, from highest to lowest priority
        allowed_single_node_partition_ops: Nodes which can be included in single-node partitions
        min_block_size: Minimum number of computational operators per block
        require_full_compilation: Require that all computational operators be run in TRT
    Returns:
        torch.fx.GraphModule
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        operator_support: ops.OperatorSupportBase,
        backend_support_map: Dict[str, Set[Target]],
        backend_priority: List[str],
        allowed_single_node_partition_ops: Optional[Collection[str]] = None,
        min_block_size: int = MIN_BLOCK_SIZE,
        require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
        return_tuple: bool = False,
        skip_fusion: bool = False,
    ):
        """
        Preprocesses graph before splitting with backend priority awareness
        """
        assert isinstance(module, torch.fx.GraphModule)

        self.module = module
        self.backend_support_map = backend_support_map
        self.backend_priority = backend_priority

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

        self.num_accelerated_subgraphs: Optional[int] = None
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
                    # if the last subgraph result[-1] is non-acc or has the same backend, merge the current subgraph into it
                    if result and (
                        not result[-1].is_acc or result[-1].backend == subgraph.backend
                    ):
                        result[-1].nodes.extend(subgraph.nodes)
                    else:
                        # if the last subgraph result[-1] has different backends, then append the current subgraph as non-acc
                        subgraph.is_acc = False
                        subgraph.backend = "None"
                        result.append(subgraph)
            else:
                if result and not result[-1].is_acc:
                    result[-1].nodes.extend(subgraph.nodes)
                else:
                    if result:
                        if result[-1].backend == subgraph.backend:
                            result[-1].nodes.extend(subgraph.nodes)
                        else:
                            result.append(subgraph)
                    else:
                        result.append(subgraph)
        return result

    def partition_graph(self) -> torch.fx.GraphModule:
        """Partitions the GraphModule into subgraphs based on backend priority

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
                "require_full_compilation=True was specified, but model is not fully supported or multiple partitions are found"
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

        # Set the number of accelerated subgraphs to be generated
        self.num_accelerated_subgraphs = len([s for s in subgraphs if s.is_acc])

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

    def put_nodes_into_subgraphs(self) -> list[Subgraph]:
        # We start graph traversal from leaf nodes
        current_cpu_nodes, current_acc_nodes = self.starter_nodes()
        visited_nodes: NodeSet = set()

        # Determine which subgraph to start from based on which subgraph has
        # 0-dep node
        acc_subgraph: bool = not any(len(self.deps[n]) == 0 for n in current_cpu_nodes)

        current_subgraph_nodes: NodeList = []

        # Result accumulator
        subgraphs: list[Subgraph] = []
        while current_cpu_nodes or current_acc_nodes:
            # Find the first node that should belong to the current subgraph and has all dependencies resolved
            current_nodes = current_acc_nodes if acc_subgraph else current_cpu_nodes
            node = next(
                (n for n in current_nodes if self.deps[n] <= visited_nodes),
                None,
            )

            # If no node was found, then it's time to flip the mode and start a new subgraph
            if node is None:
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                subgraphs.append(
                    Subgraph(
                        is_acc=acc_subgraph,
                        nodes=current_subgraph_nodes,
                        backend=(
                            current_subgraph_nodes[-1].backend
                            if acc_subgraph
                            else "None"
                        ),
                    )
                )
                acc_subgraph = not acc_subgraph
                current_subgraph_nodes = []
                continue

            # If the backend changed, then it's time to start a new subgraph
            if (
                current_subgraph_nodes
                and current_subgraph_nodes[-1].backend != node.backend
            ):
                if not current_subgraph_nodes:
                    raise FxNetSplitterInternalError("Subgraph can't be empty")

                subgraphs.append(
                    Subgraph(
                        is_acc=acc_subgraph,
                        nodes=current_subgraph_nodes,
                        backend=current_subgraph_nodes[-1].backend,
                    )
                )
                current_subgraph_nodes = []
                continue

            current_nodes.remove(node)
            visited_nodes.add(node)
            current_subgraph_nodes.append(node)

            # Add fusion buddies
            if node in self.fusions:
                if node in self.acc_nodes:
                    current_acc_nodes.update(self.fusions[node] - visited_nodes)
                else:
                    current_cpu_nodes.update(self.fusions[node] - visited_nodes)

            # Put depending nodes into the queue
            for user in node.users:
                if user.op not in CALLABLE_NODE_OPS:
                    continue

                # Add downstream nodes
                if user in self.acc_nodes:
                    current_acc_nodes.add(user)
                else:
                    current_cpu_nodes.add(user)

        # Check if the last subgraph was not created
        if current_subgraph_nodes:
            subgraphs.append(
                Subgraph(
                    is_acc=acc_subgraph,
                    nodes=current_subgraph_nodes,
                    backend=(
                        current_subgraph_nodes[-1].backend if acc_subgraph else "None"
                    ),
                )
            )

        if not subgraphs:
            raise FxNetSplitterInternalError("Couldn't create subgraphs")

        return subgraphs


class FxNetSplitterInternalError(Exception):
    pass


def hierarchical_partition(
    gm: torch.fx.GraphModule,
    verbose: bool = DEBUG,
    min_block_size: int = MIN_BLOCK_SIZE,
    torch_executed_ops: Collection[Target] = set(),
    backend_support_map: Optional[Dict[str, Set[OpOverload]]] = None,
    backend_priority: Optional[List[str]] = None,
    require_full_compilation: bool = REQUIRE_FULL_COMPILATION,
    skip_fusion: bool = False,
) -> Tuple[torch.fx.GraphModule, BackendOpSupportTester]:
    """Partition an FX GraphModule with aten ops into submodules using hierarchical partitioning
    based on backend priority and operator support

    Args:
        gm: FX GraphModule to partition
        verbose: Bool representing whether to print operator support
        min_block_size: Minimum number of operators per TRT-Engine Block
        backend_support_map: Dictionary mapping backend names to sets of supported operators
        backend_priority: Ordered list of backend names, from highest to lowest priority
        require_full_compilation: Require that all computational operators be run in TRT
        skip_fusion: Skip fusions found by FxNetAccFusionsFinder
    Returns:
        torch.fx.GraphModule, BackendOpSupportTester
    """
    # Ensure graph is clean prior to partitioning
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

    # Default backend support map if none provided
    if backend_support_map is None:
        backend_support_map = {
            "tensorrt": set(DYNAMO_ATEN_CONVERTERS.keys()),
            "inductor": set(),
        }

    # Default backend priority if none provided
    if backend_priority is None:
        backend_priority = ["tensorrt", "inductor"]

    # Construct BackendOpSupportTester
    supported_ops = BackendOpSupportTester(
        backend_support_map=backend_support_map,
        backend_priority=backend_priority,
        torch_executed_ops=torch_executed_ops,
    )
    partitioner = HierarchicalTRTPartitioner(
        gm,
        supported_ops,
        backend_support_map=backend_support_map,
        backend_priority=backend_priority,
        min_block_size=min_block_size,
        require_full_compilation=require_full_compilation,
        skip_fusion=skip_fusion,
    )

    partitioned_graph = partitioner.partition_graph()

    if verbose:
        supported_ops.print_support_overview(partitioner.num_accelerated_subgraphs)

    return partitioned_graph, supported_ops
