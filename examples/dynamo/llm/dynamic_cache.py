import logging
from typing import List, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.utils import extract_var_range_info
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

from cache_utils import add_graph_input, create_random_output_tensors, get_kv_nodes, is_op
import torch.utils._pytree as pytree
logger = logging.getLogger(__name__)


def add_kv_as_outputs(gm):
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
    # list of MHA kernels we would want to detect and replace
    mha_ops = {
        torch._C._nn.scaled_dot_product_attention,
    }
    
    # Find all SDPA nodes in the graph
    mha_nodes = []
    for node in gm.graph.nodes:
        if is_op(node, mha_ops):
            mha_nodes.append(node)

    # Iterate through each MHA node to extract shape information
    for mha_node in mha_nodes:
        if "val" in mha_node.meta and len(mha_node.args) >= 3:
            # Get the input nodes (query, key, value)
            q_node, k_node, v_node = mha_node.args[:3]
            
            # Add the copy nodes as outputs to the graph
            output_node = next(node for node in gm.graph.nodes if node.op == "output")            

            # Get the current output args (typically a tuple)
            current_outputs = output_node.args[0]
            
            # If the current output is a tuple, extend it with our new outputs
            if isinstance(current_outputs, tuple):
                new_outputs = current_outputs + ((k_node, v_node),)
            else:
                # If there's only one output or it's not a tuple, create a new tuple
                new_outputs = (current_outputs, (k_node, v_node))
            
            gm.graph.output(new_outputs)
            gm.graph.erase_node(output_node)
        
    return new_outputs




def add_kv_and_indices_as_inputs(gm, fixed_kv: bool = True):
        """
        Add key-value tensors and index parameters as inputs to the graph.
        
        Args:
            gm: The GraphModule to modify
            fixed_kv: Boolean indicating whether to use static tensors for KV cache
            
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

            # Add new inputs using add_graph_input
            k_input = add_graph_input(gm, key_value[0].name+"_k_input", k_val)
            v_input = add_graph_input(gm, key_value[1].name+"_v_input", v_val)
            kv_inputs.append((k_input, v_input))


        # Add is_generate as input
        is_generate_input = add_graph_input(gm, "is_generate", True)
        is_generate_input.meta["val"] = torch.tensor(True)

        return kv_inputs, is_generate_input


def insert_torch_cond_before_sdpa(gm, incoming_keys_values: List[Tuple[torch.Tensor, torch.Tensor]], is_generate_input: torch.Tensor):
    """
    Insert a torch.cond operation before each scaled_dot_product_attention operation.
    
    Args:
        gm: The FX GraphModule to modify
        
    Returns:
        The modified GraphModule
    """
    # Find all nodes with scaled_dot_product_attention
    sdpa_nodes = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch._C._nn.scaled_dot_product_attention:
            sdpa_nodes.append(node)

    # For each SDPA node, insert a torch.cond operation before it
    for idx, sdpa_node in enumerate(sdpa_nodes):
 
        with gm.graph.inserting_before(sdpa_node):
            # pred_node = add_graph_input(gm, "is_generate", torch.tensor(False, dtype=torch.bool))
            q_node, k_node, v_node = sdpa_node.args[:3]
            incoming_key, incoming_value = incoming_keys_values[idx]
            # Create nodes for concatenating k with incoming_key and v with incoming_value
            concatenated_k_node = gm.graph.create_node(
                "call_function",
                torch.ops.aten.cat.default,
                args=([incoming_key, k_node], 2),  # Concatenate along sequence length dimension
                kwargs={}
            )
            concatenated_v_node = gm.graph.create_node(
                "call_function",
                torch.ops.aten.cat.default,
                args=([incoming_value, v_node], 2),  #  Concatenate along sequence length dimension
                kwargs={}
            )
            
            # Create the torch.cond node
            cond_k_node = gm.graph.create_node(
                "call_function",
                torch.ops.higher_order.cond,
                args=(is_generate_input, concatenated_k_node, k_node),
            )
 
            cond_v_node = gm.graph.create_node(
                "call_function",
                torch.ops.higher_order.cond,
                args=(is_generate_input, concatenated_v_node, v_node),
            )

            sdpa_node.args = (q_node, cond_k_node, cond_v_node) + sdpa_node.args[3:]
    
    return gm



@_aten_lowering_pass
def insert_dynamic_kv_cache(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Insert FlashInfer MHA + KV cache ops in the graph"""
    """Perform insertion of kv-caches and attention kernel."""

    # Add static key and value as inputs to the graph
    kv_inputs, is_generate_input = add_kv_and_indices_as_inputs(gm, fixed_kv=True)

    # Call the function to add KV as outputs
    logits_keys_values = add_kv_as_outputs(gm)

    # Insert torch.cond before each SDPA node which acts toggles between prefill and generate phases
    gm = insert_torch_cond_before_sdpa(gm, kv_inputs, is_generate_input)

    gm = clean_up_graph_after_modifications(gm)
    
    new_output_tensors = create_random_output_tensors(logits_keys_values)
    new_out_spec = pytree.tree_flatten(new_output_tensors)[1]
    gm._out_spec = new_out_spec
    
    logger.debug("After inserting KV cache into the graph: " + str(gm.graph))
    return gm


