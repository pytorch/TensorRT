"""Resource-aware graph partitioner for TensorRT compilation.

This module refines an existing capability-based partitioning (accelerated vs
non-accelerated subgraphs) by further splitting accelerated subgraphs to meet
host CPU memory constraints during TensorRT engine building.

High-level algorithm
--------------------
Given an original `torch.fx.GraphModule` and a capability-partitioned
`GraphModule` (produced earlier in the pipeline), we:

1) Reconstruct subgraphs on the original graph
   - Iterate over the capability-partitioned module to determine which original
     nodes belong to which subgraph (accelerated or not).
   - Preserve fusion groups discovered in each subgraph so that all nodes in a fusion
     group remain in the same subgraph and not be split across subgraphs.
   - Verify subgraphs respect topological order. This is to ensure the validity of the subgraphs.
   - Reconstruting subgraphs from partitioned module is easier than building nasted partitioned graph modules and flattening them later.

2) Estimate memory cost of each possible subgraphs
   - Compute a per-subgraph "size" by traversing the graph to find weights
     (get_attr) reachable from its nodes and summing tensor bytes.
   - Use a set to record the visited nodes and avoid double counting shared parameters across subgraphs.


4) Split large accelerated subgraphs
   - While a subgraph exceeds the per-engine budget, split it into two or more subgraphs.
   - Move nodes incrementally from the front of the original subgraph into a
     new left subgraph, repeatedly validating/correcting topological, partitioning, and
     dependency constraints.
   - Ensure we never split across a fusion group; when a split would break a
     fusion, we backtrack dependencies and move the entire fusion and related nodes into the left
     side.
   - Continue until the left subgraph fits the budget
   - Repeat the process for the right subgraph until all subgraphs fit the budget.

5) Finalize
   - After splitting, assert all fusion groups reside in a single subgraph.
   - Tag nodes and produce a `GraphModule` where each subgraph becomes either a
     TRT engine (accelerated) or runs in Torch (non-accelerated).

Notes
-----
- The budget is a heuristic bound. If the total model size exceeds 40x the
  per-engine budget, we fail early with a clear error suggesting remedies.
"""

import logging
from typing import Dict, List, Tuple

import psutil
import torch
from torch.fx.passes.splitter_base import Subgraph, _SplitterBase
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch_tensorrt.dynamo.partitioning.fusion_patterns import (
    get_node_in_fusion_pattern,
)

logger = logging.getLogger(__name__)


class ResourcePartitioner(_SplitterBase):  # type: ignore
    """Refine capability-based subgraphs to meet host CPU memory constraints.

    This partitioner takes:
    - an original `torch.fx.GraphModule` (`module`)
    - a capability-partitioned `GraphModule` (`partitioned_module`) containing
      submodules that delineate accelerated vs non-accelerated regions
    - a CPU memory budget in bytes (`cpu_memory_budget`)

    It maps nodes from `module` into subgraphs according to `partitioned_module`
    and then splits oversized accelerated subgraphs so that each resulting TRT
    engine's estimated size fits within a conservative budget derived from
    available CPU memory or predefined CPU budget.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        partitioned_module: torch.fx.GraphModule,
        cpu_memory_budget: int,
    ):

        assert isinstance(module, torch.fx.GraphModule)
        assert isinstance(partitioned_module, torch.fx.GraphModule)

        self.module = module
        self.partitioned_module = partitioned_module
        self.cpu_memory_budget = cpu_memory_budget

        self.deps = self.find_deps()

        self.non_acc_submodule_name = "_run_on_gpu_"
        self._node_submodule_map: Dict[str, str] = {}
        self._return_tuple = False
        self.fusion_patterns: Dict[torch.fx.Node, List[torch.fx.Node]] = {}

    def partition_graph(self) -> torch.fx.GraphModule:
        """Build the final partitioned `GraphModule` honoring memory constraints.

        Steps:
        - Build subgraph assignments from the capability-partitioned module
        - Split oversized accelerated subgraphs based on memory budget
        - Tag nodes and construct the final split graph

        Returns:
            torch.fx.GraphModule: A graph split into subgraphs based on capability partitioning and memory constraints.
        """
        # Delegate nodes based on operator coverage
        subgraphs = self.put_nodes_into_subgraphs()

        subgraphs = self.break_subgraphs(
            subgraphs, subgraph_size_budget=self.calculate_size_budget()
        )

        # Set the number of TRT engines to be generated
        self.num_trt_accelerated_subgraphs = len([s for s in subgraphs if s.is_acc])

        # Tag the accelerated nodes and split the graph accordingly
        self.tag(subgraphs)

        gm = self.split()

        return gm

    def put_nodes_into_subgraphs(self) -> list[Subgraph]:
        """Map original graph nodes into capability-based subgraphs.

        - Iterates `partitioned_module` submodules to establish which node names
          belong to which subgraph (accelerated or not).
        - Builds a fusion pattern map for each subgraph so that known fusion groups remain intact.
          Note that since fusion map is built for each subgraph, the capability partitioning can still break the fusion groups.
        - Put the nodes into the subgraphs based on the capability partitioning.
        - Verifies the resulting list of subgraphs is topologically ordered.

        Returns:
            list[Subgraph]: Ordered subgraphs consisting of nodes in `module` based on capability partitioning.
        """
        subgraphs_map = {}
        subgraphs = []
        name_to_node_map = (
            {}
        )  # We use this map to help map the nodes in partitioned module to the nodes in original module.
        for name, _ in self.partitioned_module.named_children():
            # We first iterate over the partitioned module to find the subgraphs based on capability partitioning.
            submodule = getattr(self.partitioned_module, name)
            if not isinstance(submodule, torch.fx.graph_module.GraphModule):
                continue
            subgraph = Subgraph(is_acc="acc" in name, nodes=[])
            subgraphs.append(subgraph)
            self.fusion_patterns.update(get_node_in_fusion_pattern(submodule.graph))

            for node in submodule.graph.nodes:
                # Erase the tag from previous partitioner if it exists
                if hasattr(node, "tag"):
                    delattr(node, "tag")

                if node.op in CALLABLE_NODE_OPS:
                    # Store which subgraph the node should be put in
                    subgraphs_map[node.name] = subgraph

        # We then iterate over the original module to put the nodes into the subgraphs.
        for node in self.module.graph.nodes:
            if hasattr(node, "tag"):
                # Erase the tag from previous partitioner
                delattr(node, "tag")
            if node.op in CALLABLE_NODE_OPS:
                name_to_node_map[node.name] = node
                subgraphs_map[node.name].nodes.append(node)

        assert self.check_topological_order(
            subgraphs
        ), "The subgraphs are not topologically ordered"
        self.fusion_patterns = {
            name_to_node_map[node.name]: [
                name_to_node_map[n.name] for n in fusion_nodes
            ]
            for node, fusion_nodes in self.fusion_patterns.items()
        }

        return subgraphs

    def check_topological_order(self, subgraphs: List[Subgraph]) -> bool:
        """Return True if subgraphs are in a valid topological order.

        Each node's dependencies must appear in earlier subgraphs or earlier
        positions within the same subgraph. Subgraphs should be topologically ordered to ensure the validity of the subgraphs.
        """
        visited_nodes: set[torch.fx.Node] = set()
        for subgraph in subgraphs:
            for node in subgraph.nodes:
                if self.deps[node] > visited_nodes:
                    return False
                visited_nodes.add(node)
        return True

    def calculate_size_budget(
        self, engine_compilation_memory_usage_multiplier: int = 4
    ) -> int:
        """Compute the per-engine size budget in bytes.

        Uses explicit `cpu_memory_budget` minus used RSS
        divided by a safety multiplier.

        Args:
            engine_compilation_memory_usage_multiplier: Safety divisor applied to
                available memory to approximate a per-engine budget. By default we assume TensorRT
                compilation requires up to 4x the model's size.

        Returns:
            int: Budget in bytes for a single accelerated subgraph.
        """

        used_rss: int = psutil.virtual_memory().used
        available_rss = self.cpu_memory_budget - used_rss
        return available_rss // engine_compilation_memory_usage_multiplier

    def break_subgraphs(
        self, subgraphs: List[Subgraph], subgraph_size_budget: int
    ) -> List[Subgraph]:
        """Split oversized accelerated subgraphs until they fit within budget.

        - Compute sizes for each subgraph (in bytes of parameters reachable from
          that subgraph).
        - If the sum of all sizes is catastrophically larger than budget
          (threshold 40x), raise a ValueError with guidance.
        - For any subgraph whose size exceeds `subgraph_size_budget`, iteratively
          split it using `break_subgraph_by_size` and append resulting segments.
        - Validate that fusion groups remain intact post splitting.

        Args:
            subgraphs: Ordered list of subgraphs from capability partitioning.
            subgraph_size_budget: Target maximum size per accelerated subgraph.

        Returns:
            List[Subgraph]: New list of subgraphs after resource-aware splitting.
        """

        new_subgraphs = []
        # We throw an error if the remaining memory is almost empty compared to the model size.
        # i.e. if the remaining memory is 4G (budget is 1G) the model size is greater than 40G, we stop the compilation.
        sizes = self.size_of_subgraphs(subgraphs)
        if sum(sizes) > subgraph_size_budget * 40:
            raise ValueError(
                f"CPU memory budget or available memory is too small to compile the model. CPU memory budget: {self.cpu_memory_budget // (1024 * 1024) if self.cpu_memory_budget != -1 else "All available memory"} MB, Model size: {sum(sizes) // (1024 * 1024)} MB. "
                + "Consider setting cpu_memory_budget to a larger value or disable offload_module_to_cpu to save more CPU memory."
            )
        for subgraph, size in zip(subgraphs, sizes):

            while size > subgraph_size_budget:
                broken_subgraphs, size_0, size_1 = self.break_subgraph_by_size(
                    subgraph, subgraph_size_budget
                )
                size = size_1
                new_subgraphs.append(broken_subgraphs[0])
                subgraph = broken_subgraphs[1]
            new_subgraphs.append(subgraph)

        self._varify_all_fusion_nodes_in_same_subgraph(new_subgraphs)

        return new_subgraphs

    def _varify_all_fusion_nodes_in_same_subgraph(
        self, subgraphs: List[Subgraph]
    ) -> None:
        """Assert that every fusion group is contained in exactly one subgraph."""
        node_to_subgraph = {}
        for i, s in enumerate(subgraphs):
            for n in s.nodes:
                node_to_subgraph[n] = i

        fusion_nodes_map_list = [
            len({node_to_subgraph[n] for n in ns}) == 1
            for ns in self.fusion_patterns.values()
        ]  # fusion nodes must be in the same subgraph

        assert all(
            fusion_nodes_map_list
        ), "All fusion nodes must be in the same subgraph"
        logger.info("All fusion nodes are in the same subgraph.")

    def break_subgraph_by_size(
        self, subgraph: Subgraph, size_to_break: int
    ) -> Tuple[List[Subgraph], int, int]:
        """Split a single oversized subgraph into two valid subgraphs.

        Moves nodes from the head of `subgraph` into a new left segment until
        the left segment's estimated size exceeds `size_to_break`. During the
        process we:
        - Repeatedly validate/correct topological placement
        - Detect and avoid splitting fusion groups by moving all fused nodes
          (and their producer chain) into the left segment

        Returns:
            (segments, size_left, size_right):
                segments[0] is the new left subgraph, segments[1] is the residual
                right subgraph. Sizes are estimated parameter bytes of each.
        """
        all_nodes = subgraph.nodes
        device_ordinal = subgraph.device_ordinal
        new_subgraphs = [
            Subgraph(
                is_acc=True,
                nodes=[],
                device_ordinal=device_ordinal,
            ),
            Subgraph(
                is_acc=True,
                nodes=all_nodes,
                device_ordinal=device_ordinal,
            ),
        ]

        # We break the subgraph until the left subgraph fits the budget.
        while True:
            # Set a step size proportional to the size of the subgraph to make the algorithm more efficient.
            # This reduce the time complexity from O(N**2) to O(N). The max number of steps is 50.
            # Note: we want the first step size to be 1.
            step_size = (
                1 if not new_subgraphs[0].nodes else max(1, len(all_nodes) // 50)
            )
            new_subgraphs = self.step_and_validate(new_subgraphs, step_size)
            size_0, size_1 = self.size_of_subgraphs(new_subgraphs)
            if size_0 > size_to_break:
                break

        if len(new_subgraphs[1].nodes) == 0:
            new_subgraphs.pop(1)
        return new_subgraphs, size_0, size_1

    def step_and_validate(
        self, new_subgraphs: List[Subgraph], step_size: int = 1
    ) -> List[Subgraph]:
        """Advance the split by `step_size` nodes, then add more nodes to the left subgraph if rules are broken.
           There are two rules to check:
            1. The subgraphs should be ordered in a way that is safely to partition.
               This is checked by validate_and_correct_subgraphs. Check that function for more details.
            2. The subgraphs should not break any fusion groups.
        - Move `step_size` nodes from the right to the left subgraph.
        - Run validation/correction to ensure a legal partitioning placement.
        - Get all leaf nodes in the left subgraph and check whether any of them are in a fusion group.
        - If the move splits a fusion group, migrate the entire fusion into the left subgraph.

        Returns:
            List[Subgraph]: Updated pair of subgraphs after stabilization.
        """

        for _ in range(step_size):
            new_subgraphs[0].nodes.append(new_subgraphs[1].nodes.pop(0))

        while True:
            new_subgraphs = self.validate_and_correct_subgraphs(new_subgraphs)
            nodes_in_first_subgraph = set(new_subgraphs[0].nodes)
            nodes_in_second_subgraph = set(new_subgraphs[1].nodes)
            leaf_node = self.get_leaf_node(nodes_in_first_subgraph)
            broken_fusion = self.step_if_break_fusion(
                new_subgraphs,
                leaf_node,
                nodes_in_first_subgraph,
                nodes_in_second_subgraph,
            )
            if not broken_fusion or len(new_subgraphs[1].nodes) == 0:
                break

        return new_subgraphs

    def step_if_break_fusion(
        self,
        subgraphs: List[Subgraph],
        leaf_nodes: set[torch.fx.Node],
        nodes_in_first_subgraph: set[torch.fx.Node],
        nodes_in_second_subgraph: set[torch.fx.Node],
    ) -> bool:
        """Detect a fusion split and migrate fused nodes to the left subgraph.

        Given the current split boundary (captured by `leaf_nodes` of the left
        subgraph), check all recorded fusion groups. If any fused node remains
        on the right while its peer is on the left, pull the node and all of its
        producer chain into the left subgraph to keep fusions intact.

        Returns:
            bool: True if any fusion was migrated (i.e., a split would have
                  broken a fusion), otherwise False.
        """

        def add_nodes(node: torch.fx.Node) -> None:
            """
            This function adds a node and all its previous nodes to the first subgraph and removes it from the second subgraph in post order.
            """
            if (
                node.op in CALLABLE_NODE_OPS
                and node not in nodes_in_first_subgraph
                and node in nodes_in_second_subgraph
            ):
                # Exclude all nodes already in the first subgraph
                nodes_in_first_subgraph.add(node)
                nodes_in_second_subgraph.remove(node)
                for input_node in node._input_nodes:
                    add_nodes(input_node)
                subgraphs[0].nodes.append(node)
                subgraphs[1].nodes.remove(node)

        fusion_broken = False
        for leaf in leaf_nodes:
            for node in self.fusion_patterns.get(leaf, []):
                if (
                    node not in nodes_in_first_subgraph
                    and node in nodes_in_second_subgraph
                ):
                    fusion_broken = True
                    add_nodes(node)

        return fusion_broken

    def get_leaf_node(
        self, nodes_in_first_subgraph: set[torch.fx.Node]
    ) -> set[torch.fx.Node]:
        """Return nodes in the left subgraph that feed any node on the right.

        A node is considered a leaf if at least one of its users is not in the
        left subgraph.
        """
        leaf_node = set()

        for node in nodes_in_first_subgraph:
            for user in node.users:
                if user not in nodes_in_first_subgraph:
                    leaf_node.add(node)
                    break
        return leaf_node

    def size_of_subgraphs(self, subgraphs: List[Subgraph]) -> List[int]:
        """Estimate parameter footprint (bytes) for each subgraph.

        Traverses each subgraph's nodes and their producer chains to find
        parameters referenced via `get_attr`, summing tensor bytes. Shared
        parameters are counted only once globally.

        Returns:
            List[int]: Size per subgraph in bytes.
        """
        state_dict = self.module.state_dict(keep_vars=True)
        sizes = []
        weight_visited_nodes = set()
        for subgraph in subgraphs:
            nodes_in_subgraph = set(subgraph.nodes)
            stack = subgraph.nodes.copy()
            size = 0
            while stack:
                node = stack.pop()
                if node in weight_visited_nodes:
                    continue
                weight_visited_nodes.add(node)
                if node.op == "get_attr":
                    weight = state_dict.get(node.target, None)
                    if weight is None:
                        logger.warning(f"Weight {node.target} not found in state_dict")
                        continue
                    size += weight.numel() * weight.element_size()
                    continue
                if node not in nodes_in_subgraph:
                    # Trace to other subgraphs
                    continue
                for input_node in node._input_nodes:
                    if input_node not in weight_visited_nodes:
                        stack.append(input_node)
            sizes.append(size)
        return sizes

    def validate_and_correct_subgraphs(
        self, subgraphs: List[Subgraph]
    ) -> List[Subgraph]:
        """This is very important for the correctness of the partitioning. Torch gives undefined behavior if the subgraphs are not ordered correctly.

        The principle is: nodes that have all dependencies resolved in previous subgraphs should also be moved to the previous subgraph.
        For example, given a breakpoint node n resulting in two subgraphs S1 [..., n] and S2 [n+1, ...], all nodes in S2 that is not directly or indirectly depend on n should be moved to S1.

        We use a map to record the index of the subgraph that a node's users should belong to. If the node N is in subgraph S1 and is not the breakpoint node (subgraph.nodes[-1]),
        then the users that only depend on N should also be moved to S1. However, N is a breakpoint node, then the users that only depend on N should also be moved to S2.

        With the map, we can determine with subgraph a later node should be moved to according to all its inputs. We take max indices of all inputs nodes to determine the subgraph index.

        Returns:
            List[Subgraph]: Corrected subgraphs.
        """
        # a map from a node to the index of the subgraph it's user should belong to
        visited_nodes = {}

        for i, subgraph in enumerate(subgraphs):
            if i == 0:
                for node in subgraph.nodes:
                    visited_nodes[node] = i
                # breakpoint node's users should belong to the next subgraph
                visited_nodes[subgraph.nodes[-1]] = i + 1
                continue

            elif not subgraph.is_acc:
                # non-accelerated subgraphs should be put in the next subgraph
                for node in subgraph.nodes:
                    visited_nodes[subgraph.nodes[-1]] = i + 1
                continue

            else:
                to_remove_nodes = []
                for j, node in enumerate(subgraph.nodes):
                    if j == len(subgraph.nodes) - 1:
                        # breakpoint node's users should belong to the next subgraph
                        visited_nodes[node] = i + 1
                        continue
                    subgraph_idx = 0
                    for dep in self.deps[node]:
                        if dep in visited_nodes:
                            # We take max indices of all inputs nodes to determine the subgraph index.
                            subgraph_idx = max(subgraph_idx, visited_nodes[dep])

                    if subgraph_idx != i:
                        # If the node should be moved to a different subgraph, we move it and remove it from the current subgraph.
                        subgraphs[subgraph_idx].nodes.append(node)
                        to_remove_nodes.append(node)
                    # Record the the subgraph that the users of this node should belong to
                    visited_nodes[node] = subgraph_idx

                # Remove the nodes that are moved to other subgraphs
                for node in to_remove_nodes:
                    subgraph.nodes.remove(node)

        return subgraphs


def resource_partition(
    gm: torch.fx.GraphModule,
    partitioned_module: torch.fx.GraphModule,
    cpu_memory_budget: int,
) -> torch.fx.GraphModule:
    """Resource-aware partitioning entry point.

    Takes an original FX graph (`gm`) and a capability-partitioned module
    (`partitioned_module`) and returns a final graph where accelerated segments
    are split further, if necessary, to satisfy CPU memory limits for TRT
    engine compilation.

    Args:
        gm: Original FX `GraphModule`.
        partitioned_module: Capability-partitioned `GraphModule` indicating
            accelerated vs non-accelerated regions.
        cpu_memory_budget: CPU memory budget in bytes for engine compilation.
            Use -1 to base the budget on currently available system memory.

    Returns:
        torch.fx.GraphModule: Final graph with resource-constrained subgraphs.
    """

    # Construct
    partitioner = ResourcePartitioner(
        gm,
        partitioned_module,
        cpu_memory_budget=cpu_memory_budget,
    )

    partitioned_graph = partitioner.partition_graph()

    return partitioned_graph
