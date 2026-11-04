# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

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
   - Reconstruting subgraphs from partitioned module is easier than building nasted partitioned graph modules and flattening them later.

2) Estimate memory cost of each possible subgraphs
   - Compute a per-subgraph "size" by traversing the graph to find weights
     (get_attr) reachable from its nodes and summing tensor bytes.
   - A weight read by several subgraphs counts in each of them, because every TensorRT engine
     embeds its own copy.


4) Split large accelerated subgraphs
   - While a subgraph exceeds the per-engine budget, split it into two or more subgraphs.
   - Move nodes incrementally from the front of the original subgraph into a
     new left subgraph. The nodes are in topological order, so the left subgraph
     never depends on the right one.
   - Ensure we never split across a fusion group; when a split would break a
     fusion, we backtrack dependencies and move the entire fusion and related nodes into the left
     side.
   - Stop before a step that adds weights the left subgraph has no room for, or once
     both sides fit and the left one is larger. A node larger than the budget is kept
     while the left subgraph has no weights, together with the nodes after it that add
     no weights.
   - Repeat the process for the right subgraph until all subgraphs fit the budget.

5) Recompute input-only nodes
   - Weightless nodes computed from the graph inputs alone (attention masks, rotary tables) end up in
     the first subgraph, where the model computes them; each later subgraph that reads one gets its own
     copy of the chain instead of taking it as an input.

6) Finalize
   - After splitting, assert all fusion groups reside in a single subgraph.
   - Tag nodes and produce a `GraphModule` where each subgraph becomes either a
     TRT engine (accelerated) or runs in Torch (non-accelerated).

Notes
-----
- The budget is a heuristic bound. If the total model size exceeds 40x the
  per-engine budget, we fail early with a clear error suggesting remedies.
"""

import logging
import operator
from typing import Dict, List, Optional, Set, Tuple

import psutil
import torch
from torch.fx.experimental.const_fold import _inline_module
from torch.fx.passes.splitter_base import Subgraph, _SplitterBase
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch_tensorrt.dynamo.partitioning._atomic_subgraphs import (
    get_node_in_fusion_pattern,
)

logger = logging.getLogger(__name__)

MAX_NUM_OF_ENGINES = 50
ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER = 4
# Constants a recomputed input-only node may read: mask fill values and rotary tables, not weights.
RECOMPUTE_MAX_CONSTANT_BYTES = 64 << 10
# Constant folding turns the fixed part of an attention mask into one constant (968x968 bools for pi0.5).
RECOMPUTE_MAX_FOLDED_BYTES = 4 << 20
# A value up to this size crosses between engines as it is; only larger ones are recomputed from it.
RECOMPUTE_MIN_OUTPUT_BYTES = 64 << 10


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
        cpu_memory_budget: Optional[int],
        submodule_name: str,
    ):

        assert isinstance(module, torch.fx.GraphModule)

        self.module = module
        used_rss: int = psutil.Process().memory_info().rss
        self.remaining_memory_budget = (
            cpu_memory_budget - used_rss
            if cpu_memory_budget is not None
            else psutil.virtual_memory().available
        )
        self.not_set_limit = cpu_memory_budget is None
        self.resource_split_count = 0
        self.submodule_name = submodule_name

        self._node_submodule_map: Dict[str, str] = {}
        self._return_tuple = False
        self.fusion_patterns: Dict[torch.fx.Node, Set[torch.fx.Node]] = {}

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
        sizes = self.size_of_subgraphs(subgraphs)
        if (
            sum(sizes) * ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER
            < self.remaining_memory_budget
        ):
            return self.module

        subgraphs = self.break_subgraphs(
            subgraphs, subgraph_size_budget=self.calculate_size_budget()
        )
        subgraphs = self.recompute_input_only_nodes(subgraphs)

        if len(subgraphs) == 1:
            return self.module

        # Set the number of TRT engines to be generated
        self.num_trt_accelerated_subgraphs = len([s for s in subgraphs if s.is_acc])

        # Tag the accelerated nodes and split the graph accordingly
        self.tag(subgraphs)

        gm = self.split(remove_tag=True)

        return gm

    def tag(self, subgraphs: list[Subgraph]) -> None:
        self.tags = []
        for subgraph in subgraphs:
            tag = f"{self.submodule_name}_resource_split_{self.resource_split_count}"
            self.resource_split_count += 1
            self.tags.append(tag)
            for node in subgraph.nodes:
                node.tag = tag
                self._node_submodule_map[node.name] = tag

    def put_nodes_into_subgraphs(self) -> list[Subgraph]:
        """
        Put the nodes into the subgraphs and erase the tag from previous partitioner if it exists.
        Returns:
            list[Subgraph]: Ordered subgraphs consisting of nodes in `module`.
        """

        nodes = []
        for node in self.module.graph.nodes:
            if hasattr(node, "tag"):
                del node.tag
            if node.op in CALLABLE_NODE_OPS:
                nodes.append(node)
        subgraphs = [Subgraph(is_acc=True, nodes=nodes)]
        self.fusion_patterns = get_node_in_fusion_pattern(self.module.graph)
        # A TensorRT engine cannot output a tuple, so a node that returns several values
        # stays in one subgraph with the getitems that unpack them, like a fusion group.
        # Storing the group on the node alone is enough: a getitem left on the other side of a split
        # makes the node a leaf, and the split checks the groups of leaves.
        for node in nodes:
            getitems = {u for u in node.users if u.target == operator.getitem}
            if getitems:
                self.fusion_patterns.setdefault(node, set()).update(getitems | {node})

        return subgraphs

    def calculate_size_budget(
        self,
        engine_compilation_memory_usage_multiplier: int = ENGINE_COMPILATION_MEMORY_USAGE_MULTIPLIER,
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

        return (
            self.remaining_memory_budget // engine_compilation_memory_usage_multiplier
        )

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
        # This bounds the model size, not the engine count. There can be more engines, because a split
        # engine can stop well below the budget and a weight counts in every engine that reads it.
        sizes = self.size_of_subgraphs(subgraphs)
        if sum(sizes) > subgraph_size_budget * MAX_NUM_OF_ENGINES:
            if self.not_set_limit:
                raise ValueError(
                    "The system memory is too constrained to compile the model without severe perf degradation. Consider setting offload_module_to_cpu=False to save more CPU memory."
                )
            else:
                raise ValueError(
                    "CPU memory budget is too small to compile the model. "
                    + f"CPU memory budget: {self.remaining_memory_budget // (1024 * 1024)} MB, Model size: {sum(sizes) // (1024 * 1024)} MB. "
                    + "Consider setting cpu_memory_budget to a larger value."
                )
        for subgraph, size in zip(subgraphs, sizes):

            while size > subgraph_size_budget:
                broken_subgraphs, size_0, size_1 = self.break_subgraph_by_size(
                    subgraph, subgraph_size_budget
                )
                size = size_1
                new_subgraphs.append(broken_subgraphs[0])
                subgraph = broken_subgraphs[1]

            if len(subgraph.nodes) != 0:
                new_subgraphs.append(subgraph)

        self._verify_all_fusion_nodes_in_same_subgraph(new_subgraphs)

        return new_subgraphs

    def _verify_all_fusion_nodes_in_same_subgraph(
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

        Moves nodes from the head of `subgraph` into a new left segment. It stops
        before a step that adds weights the left segment has no room for in
        `size_to_break`, or once both segments fit and the left one is larger.
        A node larger than `size_to_break` is kept while the left segment has
        no weights, together with the nodes after it that add no weights.
        During the process we detect and avoid splitting fusion groups by
        moving all fused nodes (and their producer chain) into the left segment.

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

        step_size = 1
        size_0 = 0
        while new_subgraphs[1].nodes:
            nodes_before_step = [s.nodes.copy() for s in new_subgraphs]
            size_before_step = size_0
            new_subgraphs = self.step_and_validate(new_subgraphs, step_size)
            size_0, size_1 = self.size_of_subgraphs(new_subgraphs)
            # Comparing with size_before_step too keeps the steps after an oversized node while
            # they add no weights.
            if size_0 > max(size_to_break, size_before_step):
                # The step added weights the left subgraph has no room for: undo it and retry with half
                # as many nodes, or stop if it was a single node. A single node is kept instead while
                # the left subgraph has no weights, because a subgraph without weights saves no memory.
                if step_size > 1 or size_before_step > 0:
                    for s, nodes in zip(new_subgraphs, nodes_before_step):
                        s.nodes[:] = nodes
                    size_0 = size_before_step
                    if step_size == 1:
                        break
                    step_size //= 2
                    continue
            elif size_1 < size_0 <= size_to_break:
                # Both sides fit, and stopping here keeps the two engines about the same size.
                break
            # Steps of 2% of the remaining nodes keep the number of size estimates low.
            step_size = max(1, len(new_subgraphs[1].nodes) // 50)

        size_0, size_1 = self.size_of_subgraphs(new_subgraphs)
        return new_subgraphs, size_0, size_1

    def step_and_validate(
        self, new_subgraphs: List[Subgraph], step_size: int = 1
    ) -> List[Subgraph]:
        """Advance the split by `step_size` nodes, then add more nodes to the left subgraph if a fusion group is broken.
        - Move `step_size` nodes from the right to the left subgraph.
        - Get all leaf nodes in the left subgraph and check whether any of them are in a fusion group.
        - If the move splits a fusion group, migrate the entire fusion into the left subgraph.

        Returns:
            List[Subgraph]: Updated pair of subgraphs after stabilization.
        """

        for _ in range(step_size):
            new_subgraphs[0].nodes.append(new_subgraphs[1].nodes.pop(0))

        while True:
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
            This function moves a node and all its previous nodes in the second subgraph to the first subgraph, keeping their order.
            """
            # An explicit stack, not recursion: a chain of producers can be longer than Python's recursion limit.
            nodes_to_move, stack = set(), [node]
            while stack:
                n = stack.pop()
                if (
                    n.op in CALLABLE_NODE_OPS
                    and n not in nodes_to_move
                    and n in nodes_in_second_subgraph
                ):
                    nodes_to_move.add(n)
                    stack.extend(n._input_nodes)
            nodes_in_first_subgraph.update(nodes_to_move)
            nodes_in_second_subgraph.difference_update(nodes_to_move)
            # The second subgraph is in graph order, so producers move before their users.
            for n in [n for n in subgraphs[1].nodes if n in nodes_to_move]:
                subgraphs[0].nodes.append(n)
                subgraphs[1].nodes.remove(n)

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
        parameters referenced via `get_attr`, summing tensor bytes. A parameter
        read by several subgraphs counts in each of them, because every TensorRT
        engine embeds its own copy.

        Returns:
            List[int]: Size per subgraph in bytes.
        """
        state_dict = self.module.state_dict(keep_vars=True)
        sizes = []
        for subgraph in subgraphs:
            weight_visited_nodes = set()
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

    def recompute_input_only_nodes(self, subgraphs: List[Subgraph]) -> List[Subgraph]:
        """Recompute, in each subgraph, the weightless nodes it reads that depend only on graph inputs.

        Splitting keeps graph order, and a model computes the values that depend only on its inputs
        (an attention mask, rotary cos/sin) before its first layer, so they land in the first engine and
        reach every later engine as inputs. Besides the extra
        engine I/O, TensorRT's fused attention on sm_110 (11.0 to 11.4) cannot build a layer fed
        that way: its kernel needs 0x39174 bytes of shared memory, 0x38c00 max. Each subgraph
        that reads such a value gets its own copy of the chain instead.

        Only large values are recomputed, from the small ones they are built from: a value of at most
        `RECOMPUTE_MIN_OUTPUT_BYTES` (a padding mask, position ids, prompt tokens) crosses engines as it
        is, and so does any node that reads a graph input directly, which keeps each input a single
        reader, as GPU-resident ExecuTorch programs (`skip_h2d_for_method_inputs`) require. A whole
        chain recomputed into an attention engine lets TensorRT fuse it into the attention kernel,
        which fails the same way.
        """
        graph = self.module.graph
        params = self.module.state_dict(keep_vars=True)
        # Whether each node is weightless and computed from the graph inputs alone. One pass in
        # graph order, which puts a node's inputs before it: chains can be thousands of nodes
        # deep, too deep to walk recursively.
        cheap: Dict[torch.fx.Node, bool] = {}
        for node in graph.nodes:
            if node.op == "placeholder":
                cheap[node] = True
            elif node.op == "get_attr":
                weight = params.get(node.target)
                limit = (
                    RECOMPUTE_MAX_FOLDED_BYTES
                    if str(node.target).startswith("_frozen_param")
                    else RECOMPUTE_MAX_CONSTANT_BYTES
                )
                cheap[node] = (
                    weight is None or weight.numel() * weight.element_size() <= limit
                )
            else:
                cheap[node] = (
                    node.op == "call_function"
                    and torch.Tag.nondeterministic_seeded
                    not in getattr(node.target, "tags", ())
                    and all(cheap[i] for i in node._input_nodes)
                )

        def crosses(node: torch.fx.Node) -> bool:
            """Whether the node's value goes to later subgraphs as it is rather than being recomputed."""
            if any(i.op == "placeholder" for i in node._input_nodes):
                return True
            value = node.meta.get("val")
            return (
                isinstance(value, torch.Tensor)
                and value.numel() * value.element_size() <= RECOMPUTE_MIN_OUTPUT_BYTES
            )

        where = {n: i for i, s in enumerate(subgraphs) for n in s.nodes}
        order = {
            n: k for k, n in enumerate(graph.nodes)
        }  # subgraph lists aren't in graph order
        recomputed = 0
        for j, subgraph in enumerate(subgraphs):
            copies: Dict[torch.fx.Node, torch.fx.Node] = {}

            def kept(node: torch.fx.Node) -> bool:
                """Whether subgraph j reads the node itself rather than a copy."""
                return (
                    node.op in ("placeholder", "get_attr")
                    or where.get(node) == j
                    or crosses(node)
                )

            def copy(node: torch.fx.Node, before: torch.fx.Node) -> torch.fx.Node:
                """Copy the chain that computes `node` into subgraph j, ahead of `before`."""
                chain, stack = set(), [node]
                while stack:  # not recursive, for the same reason as `cheap`
                    n = stack.pop()
                    if n not in chain and n not in copies and not kept(n):
                        chain.add(n)
                        stack.extend(n._input_nodes)
                chain_in_order = sorted(chain, key=order.__getitem__)
                for n in chain_in_order:
                    with graph.inserting_before(before):
                        copies[n] = graph.node_copy(n, lambda i: copies.get(i, i))
                position = subgraph.nodes.index(before)
                subgraph.nodes[position:position] = [copies[n] for n in chain_in_order]
                return copies.get(node, node)

            for user in sorted(subgraph.nodes, key=order.__getitem__):
                for node in list(user._input_nodes):
                    if where.get(node, j) < j and cheap[node] and not crosses(node):
                        user.replace_input_with(node, copy(node, user))
                        recomputed += 1
        for subgraph in subgraphs:  # the originals nothing reads any more
            for node in reversed(list(subgraph.nodes)):
                if not node.users and node.op == "call_function" and cheap.get(node):
                    subgraph.nodes.remove(node)
                    graph.erase_node(node)
        if recomputed:
            logger.info(
                f"Recomputed {recomputed} input-only inputs in the subgraphs that read them."
            )
        return [s for s in subgraphs if s.nodes]


def resource_partition(
    gm: torch.fx.GraphModule,
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
    for name, _ in gm.named_children():
        submodule = getattr(gm, name)
        if (
            not isinstance(submodule, torch.fx.graph_module.GraphModule)
            or "_run_on_acc" not in name
        ):
            continue
        partitioner = ResourcePartitioner(
            submodule,
            submodule_name=name,
            cpu_memory_budget=cpu_memory_budget,
        )

        partitioned_graph = partitioner.partition_graph()
        setattr(gm, name, partitioned_graph)

    for name, module in list(gm.named_children()):
        split = False
        if "_run_on_acc" in name:
            for subname, submodule in module.named_children():
                if "resource_split" in subname:
                    split = True
                    setattr(gm, subname, submodule)
            if split:
                _inline_module(gm, name)
                delattr(gm, name)

    gm.recompile()
    return gm
