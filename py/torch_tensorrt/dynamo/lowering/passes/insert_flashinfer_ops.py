import logging
import operator
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch._export.utils import _detect_fake_mode_from_gm
from torch._ops import OpOverload, OpOverloadPacket
from torch._subclasses import FakeTensor, FakeTensorMode
from torch.fx import Graph, GraphModule, Node
from torch.fx.passes.shape_prop import _extract_tensor_metadata
from torch.utils._pytree import _LEAF_SPEC
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

from ..attention_interface import AttentionInfo, SequenceInfo

logger = logging.getLogger(__name__)


def add_graph_input(
    gm: GraphModule, name: str, val: Optional[torch.Tensor] = None, dynamic_shape=None
) -> Node:
    """Add a graph input to the given GraphModule and return the newly created node.

    NOTE: function does NOT do any graph canonicalization. This is left to the user!

    Args:
        gm (GraphModule): The GraphModule to add the input to.
        name (str): The name of the input.
        val (torch.Tensor): An example tensor to use for the input.
        dynamic_shape: The dynamic shape of the input tensor [NOT SUPPORTED YET]
    """
    # check that no dynamic shape is provided...
    if dynamic_shape:
        raise NotImplementedError("Dynamic shape not supported for adding graph inputs")

    # extract graph and input spec
    graph: Graph = gm.graph

    in_spec = graph._codegen.pytree_info.in_spec
    in_spec_for_args = in_spec.children_specs[0]
    orig_args = graph._codegen.pytree_info.orig_args
    assert in_spec_for_args.type is tuple

    # insert input node after currently last input node
    node_last_input = graph.find_nodes(op="placeholder", sort=True)[-1]
    with graph.inserting_after(node_last_input):
        in_node = graph.placeholder(name)
        in_spec_for_args.children_specs.append(_LEAF_SPEC)
        orig_args.append(f"arg_{name}")

    # update pytree info recursively with __post_init__ starting at leaves
    def call_post_init(spec):
        for child_spec in spec.children_specs:
            call_post_init(child_spec)
        spec.__post_init__()

    call_post_init(in_spec)

    # set fake tensor information if all required information is available
    fake_mode: Optional[FakeTensorMode] = _detect_fake_mode_from_gm(gm)
    if fake_mode and val:
        fake_tensor: FakeTensor = fake_mode.from_tensor(val, static_shapes=True)
        in_node.meta["val"] = fake_tensor
        in_node.meta["tensor_meta"] = _extract_tensor_metadata(fake_tensor)

    # return new node...
    return in_node


def is_dist_op(node: Node) -> bool:
    """Check if the node is a distributed op."""
    dist_ops = {
        torch.ops.dist.all_gather,
        torch.ops.dist.all_reduce,
    }
    return is_op(node, dist_ops)


def is_linear_op(node: Node, include_quantization: bool = False) -> bool:
    """Check if the node is a linear op.

    Using this function is preferred over `is_op` for linear ops to ensure all variants are covered.
    """
    lin_ops = {
        torch.ops.aten.linear,
    }

    return is_op(node, lin_ops)


def _is_dist_lin_op(node: Node, exclude: Optional[List[Node]] = None) -> bool:
    return node not in (exclude or []) and (
        is_linear_op(node, include_quantization=True)
    )


def _bfs(node: Node, target: Callable, attr_next: str = "users") -> Node:
    queue = [node]
    while queue:
        cur_node = queue.pop(0)
        if target(cur_node):
            return cur_node
        queue.extend(getattr(cur_node, attr_next))
    raise RuntimeError(f"Could not find node with target condition {target}.")


def is_op(node: Node, ops: Union[OpOverloadPacket, Iterable[OpOverloadPacket]]) -> bool:
    """Check if the node is a call to one of the ops."""
    if node.op != "call_function":
        return False
    # check if it's a single op that's provided
    if isinstance(ops, OpOverloadPacket):
        ops = [ops]

    # check if it's the op itself instead of an overload
    if any(node.target == op for op in ops):
        return True

    return False


def get_all_input_output_nodes(graph: Graph) -> Tuple[List[Node], List[Node]]:
    input_nodes: List[Node] = graph.find_nodes(op="placeholder")
    output_nodes: List[Node] = graph.find_nodes(op="output")
    return (input_nodes, output_nodes)


def insert_flashinfer_attn_with_cache(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Insert FlashInfer MHA + KV cache ops in the graph"""
    """Perform insertion of kv-caches and attention kernel."""
    graph = gm.graph
    cm = settings.cached_seq_interface
    # list of MHA kernels we would want to detect and replace
    mha_ops = {
        torch._C._nn.scaled_dot_product_attention,
    }

    # loop through nodes to get input, output, and get_attr nodes
    input_nodes, output_nodes = get_all_input_output_nodes(graph)

    # we only expect one input node
    assert len(input_nodes) == 1, "Expected exactly one input node."

    # NOTE: for now, we wanna make sure we *only* return the final output and no hidden states.
    # Later on, we can revisit how to support returning hidden states.
    # assert len(output_nodes) == 1, "Expected exactly one output node!"
    # assert (
    #     len(output_nodes[0].all_input_nodes) == 1
    # ), "Expected to only return final tensor output!"

    # get all mha nodes and their GEMMs as well as sanity checks and shape information
    mha_gemms = defaultdict(list)
    mha_info = {}
    for mha_node in graph.nodes:
        if mha_node.op != "call_function" or not is_op(mha_node, mha_ops):
            continue
        # do some sanity checks on the args of the node
        assert mha_node.kwargs == {}, "We don't handle kwargs for mha nodes right now."
        assert (
            len(mha_node.args) >= 3
        ), "MHA nodes should have at least 3 args: q, k, v."
        args_other = mha_node.args[3:]
        args_other_expected = (None, 0.0, True)[
            : len(args_other)
        ]  # other args expected
        if args_other != args_other_expected:
            logger.debug(f"Unexpected args for MHA node: {args_other}.")

        # get fake q tensor that is an MHA input node to retrieve head_dim, num_heads, and dtype
        # also retrieve fake tensor corresponding to output of k GEMM to infer number of kv heads
        q_fake = mha_node.args[0].meta["val"]
        kv_gemm_fake = mha_node.args[1].meta[
            "val"
        ]  # mha_gemms[mha_node][1].meta["val"]

        mha_info[mha_node] = AttentionInfo(
            num_heads=q_fake.shape[1],
            num_kv_heads=kv_gemm_fake.shape[1],  # // q_fake.shape[3],
            head_dim=q_fake.shape[3],
            dtype=q_fake.dtype,
            cache_dtype=None,
            rope_theta=None,
        )

    # insert metadata computation and extract each argument as a node
    mha_0 = next(iter(mha_info.keys()))
    get_metadata, num_metadata = cm.attention_op.get_prepare_metadata_op()
    with graph.inserting_before(mha_0):
        ret_node = graph.call_function(
            get_metadata,
            args=(
                input_nodes[0],
                *(add_graph_input(gm, name) for name in cm.info.extra_arg_names),
                cm.info.page_size,
            ),
        )
        metadata_nodes = [
            graph.call_function(operator.getitem, args=(ret_node, idx))
            for idx in range(num_metadata)
        ]

    buffer_in_lookup: Dict[str, Node] = {}

    for idx, (mha_node, mha_node_info) in enumerate(mha_info.items()):
        # setup + store cache initializers and caches as input nodes
        cache_in_nodes = []
        for k, fn in cm.attention_op.get_cache_initializers(mha_info[mha_node]).items():
            k_indexed = f"{k}_{idx}"
            cm.add_cache(k_indexed, fn)
            cache_in_nodes.append(add_graph_input(gm, k_indexed))

        # setup + store global buffer initializers and buffers as input nodes
        # NOTE: we have to check against existing keys to make sure nothing is registered twice...
        buffer_in_nodes = []
        for k, fn in cm.attention_op.get_global_buffer_initializers(
            mha_info[mha_node]
        ).items():
            if k not in buffer_in_lookup:
                cm.add_cache(k, fn)
                buffer_in_lookup[k] = add_graph_input(gm, k)
            buffer_in_nodes.append(
                buffer_in_lookup[k]
            )  # store buffer nodes for this op

        # retrieve constants for attention_op
        constants = cm.attention_op.get_constants(mha_info[mha_node])

        # insert fused replacement op
        with graph.inserting_before(mha_node):
            mha_node_with_cache = graph.call_function(
                cm.attention_op.get_attention_op(),
                args=(
                    *mha_node.all_input_nodes,
                    *metadata_nodes,
                    *cache_in_nodes,
                    *buffer_in_nodes,
                    *constants,
                ),
            )
        mha_node.replace_all_uses_with(mha_node_with_cache)
        graph.erase_node(mha_node)

    gm = clean_up_graph_after_modifications(gm)

    cm.initialize_caches()
    logger.debug("After inserting MHA with KV cache: " + str(gm.graph))
    return gm
