import logging
from typing import List, Tuple

import torch
import torch.utils._pytree as pytree
from cache_utils import _add_graph_input, create_random_output_tensors, get_kv_nodes
from torch.fx import Node
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.utils import extract_var_range_info

logger = logging.getLogger(__name__)

SDPA_OP = torch._C._nn.scaled_dot_product_attention


def add_kv_as_outputs(gm, kv_cache_for_graph: List[Tuple[torch.Tensor, torch.Tensor]]):
    """
    Modifies the graph to add query, key, and value tensors as outputs.

    This function identifies all scaled dot-product attention (SDPA) operations
    in the graph, creates copies of their query, key, and value inputs, and adds
    these copies to the graph's outputs. This allows for accessing these tensors
    externally, which is useful for operations like key-value caching.

    Args:
        graph: The torch.fx.Graph to modify

    Returns:
        None. The graph is modified in-place.
    """
    output_node = next(node for node in gm.graph.nodes if node.op == "output")

    # Get the current output args (typically a tuple)
    current_outputs = output_node.args[0]

    # If the current output is a tuple, extend it with our new outputs
    if isinstance(current_outputs, tuple):
        new_outputs = current_outputs + tuple(kv_cache_for_graph)
    else:
        # If there's only one output or it's not a tuple, create a new tuple
        new_outputs = (current_outputs,) + tuple(kv_cache_for_graph)

    gm.graph.output(new_outputs)
    gm.graph.erase_node(output_node)

    return new_outputs


def add_kv_cache_inputs(gm, fixed_kv: bool = True):
    """
    Add key-value tensors, index parameters as inputs to the graph.

    Args:
        gm: The GraphModule to modify
        fixed_kv: Boolean indicating whether to use static tensors for KV cache. Default is True.

    Returns:
        A tuple containing:
        - List of (k_input, v_input) node pairs for each SDPA operation
        - start_idx input node for slicing operations
        - end_idx input node for slicing operations
    """

    def get_static_tensor(tensor: torch.Tensor):
        key_shape = []
        for dim in tensor.shape:
            if isinstance(dim, torch.SymInt):
                min_max_opt = extract_var_range_info(dim)
                key_shape.append(min_max_opt["max"])
            else:
                key_shape.append(dim)

        static_tensor = torch.randn(key_shape, dtype=tensor.dtype, device=tensor.device)
        return static_tensor

    keys_values = get_kv_nodes(gm)

    kv_inputs = []
    for idx, key_value in enumerate(keys_values):
        k_val = key_value[0].meta["val"]
        v_val = key_value[1].meta["val"]
        if fixed_kv:
            k_val = get_static_tensor(k_val)
            v_val = get_static_tensor(v_val)

        # Add new inputs using _add_graph_input
        k_input = _add_graph_input(gm, key_value[0].name + "_k_input", k_val)
        v_input = _add_graph_input(gm, key_value[1].name + "_v_input", v_val)
        kv_inputs.append((k_input, v_input))

    # Add start_idx and end_idx as inputs
    start_idx_input = _add_graph_input(gm, "start_idx", torch.tensor(0))
    end_idx_input = _add_graph_input(gm, "end_idx", torch.tensor(1))

    # Get the max sequence length from the first key_cache node. The order of nodes is: input_ids, is_causal, key_cache1, value_cache1, key_cache2, value_cache2, ..
    input_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    input_ids_meta = input_nodes[0].meta["val"]
    seq_len = input_ids_meta.shape[1]
    min_max_opt = extract_var_range_info(seq_len)
    max_seq_len = min_max_opt["max"]

    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    # Create symbolic ints for start_idx and end_idx with range [0, seq_len] inclusive
    start_idx_unbacked_symint = shape_env.create_unbacked_symint()
    torch._check(start_idx_unbacked_symint >= 0)
    torch._check(start_idx_unbacked_symint <= max_seq_len)

    end_idx_unbacked_symint = shape_env.create_unbacked_symint()
    torch._check(end_idx_unbacked_symint >= 0)
    torch._check(end_idx_unbacked_symint <= max_seq_len)
    # Set the symbolic ints as the metadata for start_idx and end_idx inputs
    start_idx_input.meta["val"] = start_idx_unbacked_symint
    end_idx_input.meta["val"] = end_idx_unbacked_symint

    return kv_inputs, start_idx_input, end_idx_input


def insert_kv_slicing_before_sdpa(
    gm,
    incoming_keys_values: List[Tuple[torch.Tensor, torch.Tensor]],
    start_idx_input: Node,
    end_idx_input: Node,
):
    """
    Insert slicing operations before each scaled_dot_product_attention operation.
    """
    # Find all nodes with scaled_dot_product_attention
    sdpa_nodes = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == SDPA_OP:
            sdpa_nodes.append(node)
    kv_cache_for_graph = []
    for idx, sdpa_node in enumerate(sdpa_nodes):
        assert (
            len(sdpa_node.args) == 6
        ), f"SDPA node should have 6 arguments but got {len(sdpa_node.args)} arguments"
        q_node, k_node, v_node, attn_mask, dropout_p, is_causal = sdpa_node.args
        incoming_key, incoming_value = incoming_keys_values[idx]
        kv_cache_for_sdpa_node = []
        new_keys_values = []
        for key_or_value, current_key_or_value_node in zip(
            [incoming_key, incoming_value], [k_node, v_node]
        ):
            # Create a slice node for key_cache[:,:,:start_idx,:]. The shape of key_cache is batch_size x num_heads x seq_len x head_dim
            with gm.graph.inserting_before(sdpa_node):
                slice_1 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(key_or_value,),
                    kwargs={},
                )
                slice_2 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_1, 1),
                    kwargs={},
                )
                slice_3 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_2, 2, None, start_idx_input),
                    kwargs={},
                )
                slice_4 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_3, 3),
                    kwargs={},
                )
                # =============================================== #
                # Create a slice node for key_cache[:,:, end_idx:,:]. The shape of key_cache is batch_size x num_heads x seq_len x head_dim
                slice_5 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(key_or_value,),
                    kwargs={},
                )
                slice_6 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_5, 1),
                    kwargs={},
                )
                slice_7 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_6, 2, end_idx_input),
                    kwargs={},
                )
                slice_8 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_7, 3),
                    kwargs={},
                )
                # =============================================== #
                # Concatenate the sliced tensors to build KV cache
                cat = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.cat.default,
                    args=([slice_4, current_key_or_value_node, slice_8], 2),
                    kwargs={},
                )
                # Update the metadata of the newly built KV cache node with the metadata of the input KV cache node to the graph
                cat.meta.update(key_or_value.meta)
                kv_cache_for_sdpa_node.append(cat)
                # =============================================== #
                # Get the current key and value by indexing the KV cache
                slice_9 = gm.graph.create_node(
                    "call_function", torch.ops.aten.slice.Tensor, args=(cat,), kwargs={}
                )
                slice_10 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_9, 1),
                    kwargs={},
                )
                slice_11 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_10, 2, None, end_idx_input),
                    kwargs={},
                )
                slice_12 = gm.graph.create_node(
                    "call_function",
                    torch.ops.aten.slice.Tensor,
                    args=(slice_11, 3),
                    kwargs={},
                )
                new_keys_values.append(slice_12)

        kv_cache_for_graph.extend(kv_cache_for_sdpa_node)

        sdpa_node.args = (q_node, new_keys_values[0], new_keys_values[1]) + (
            attn_mask,
            dropout_p,
            True,
        )

    return gm, kv_cache_for_graph


@_aten_lowering_pass
def insert_static_cache_v1(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Insert KV cache ops in the graph"""
    """Perform insertion of kv-caches and attention kernel."""
    # Add static key and value as inputs to the graph
    kv_inputs, start_idx_input, end_idx_input = add_kv_cache_inputs(gm, fixed_kv=True)

    # Build and update the KV cache using computed KV inputs for current token and
    # incoming keys and values from previous tokens (which were added as inputs)
    gm, kv_cache_for_graph = insert_kv_slicing_before_sdpa(
        gm, kv_inputs, start_idx_input, end_idx_input
    )

    # Call the function to add KV as outputs
    logits_keys_values = add_kv_as_outputs(gm, kv_cache_for_graph)

    gm = clean_up_graph_after_modifications(gm)

    new_output_tensors = create_random_output_tensors(logits_keys_values)

    new_out_spec = pytree.tree_flatten(new_output_tensors)[1]
    gm._out_spec = new_out_spec
    logger.debug("After inserting KV cache into the graph: " + str(gm.graph))

    return gm
