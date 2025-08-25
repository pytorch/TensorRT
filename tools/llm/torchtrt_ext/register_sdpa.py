import copy
import logging
import operator
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import args_bounds_check
from torch_tensorrt.dynamo.lowering import TORCH_TRT_DECOMPOSITIONS
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

from .sdpa_converter import *

logger = logging.getLogger(__name__)

# Remove decompositions for aten.scaled_dot_product_attention, aten._scaled_dot_product_efficient_attention, aten._scaled_dot_product_flash_attention
# This is because we want to have SDPA as a standalone operator in the graph and invoke the custom converter for it.
TORCH_TRT_DECOMPOSITIONS.pop(torch.ops.aten.scaled_dot_product_attention.default, None)
TORCH_TRT_DECOMPOSITIONS.pop(
    torch.ops.aten._scaled_dot_product_efficient_attention.default, None
)
TORCH_TRT_DECOMPOSITIONS.pop(
    torch.ops.aten._scaled_dot_product_flash_attention.default, None
)

REPLACEABLE_ATEN_OPS = {
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
}


@_aten_lowering_pass
def replace_variants_of_sdpa(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace scaled_dot_product_attention with an equivalent
    implementation which can be accurately converted to TRT
    """

    for node in gm.graph.nodes:
        attn_mask = None
        is_causal = False
        if node.op == "call_function" and node.target in REPLACEABLE_ATEN_OPS:
            if (
                node.target
                == torch.ops.aten._scaled_dot_product_efficient_attention.default
            ):
                if len(node.args) == 7:
                    (
                        query,
                        key,
                        value,
                        attn_mask,
                        compute_log_sumexp,
                        dropout_p,
                        is_causal,
                    ) = node.args
                elif len(node.args) == 5:
                    query, key, value, attn_mask, is_causal = node.args
                    dropout_p = 0.0

                else:
                    raise ValueError(
                        f"Unexpected number of arguments for {node.target} in the graph"
                    )
            elif (
                node.target
                == torch.ops.aten._scaled_dot_product_flash_attention.default
            ):
                if len(node.args) == 6:
                    (
                        query,
                        key,
                        value,
                        dropout_p,
                        is_causal,
                        return_debug_mask,
                    ) = node.args
                if len(node.args) == 5:
                    query, key, value, dropout_p, is_causal = node.args
                elif len(node.args) == 3:
                    query, key, value = node.args
                    dropout_p = 0.0
                    is_causal = True
                else:
                    raise ValueError(
                        f"Unexpected number of arguments for {node.target} in the graph"
                    )

            logger.warning(
                f"This current version of SDPA converter only supports attn_mask = {attn_mask}, dropout_p = {dropout_p} and is_causal = {is_causal} configuration. This could cause issues with accuracy for models with different configurations."
            )
            modified_input_args = (query, key, value, attn_mask, dropout_p, is_causal)
            # Create a new node with torch.nn.functional.scaled_dot_product_attention
            # The input args is (query, key, value, attn_mask, dropout_p, is_causal). kwargs has scale
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    torch.nn.functional.scaled_dot_product_attention,
                    args=modified_input_args,
                    kwargs={
                        "scale": node.kwargs.get("scale", None),
                        "use_fp32_acc": settings.use_fp32_acc,
                    },
                )

                # Deep copy encounters RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor). So we use copy instead.
                new_node.meta = copy.copy(node.meta)
                # Check if there's a getitem node following this attention node
                for user in list(node.users):
                    if user.op == "call_function" and user.target == operator.getitem:
                        # If the getitem is extracting the first element (the output tensor)
                        if user.args[1] == 0:
                            # Replace all uses of the getitem with the new attention node
                            user.replace_all_uses_with(new_node)
                            new_node.meta["val"] = new_node.meta["val"][0]
                # Replace all uses of the original node with the new node
                node.replace_all_uses_with(new_node)

            gm.graph.erase_node(node)

    # Clean up the graph
    clean_up_graph_after_modifications(gm)

    logger.debug(
        "Replaced variants of scaled_dot_product_attention with torch.nn.functional.scaled_dot_product_attention"
    )
    add_attn_mask_as_output = False
    if add_attn_mask_as_output:
        add_one_attn_mask_as_output(gm)
    return gm


# try to add one of the attn_mask as output, so that I can actually see the shape and value in the generation phase.
def add_one_attn_mask_as_output(gm: torch.fx.GraphModule):
    import torch.utils._pytree as pytree
    from cache_utils import create_random_output_tensors

    attn_mask_node = None
    for node in gm.graph.nodes:
        if (
            node.op == "call_function"
            and node.target == torch.nn.functional.scaled_dot_product_attention
        ):
            attn_mask_node = node.args[3]
            break

    output_node = next(node for node in gm.graph.nodes if node.op == "output")

    current_outputs = output_node.args[0]
    if isinstance(current_outputs, tuple):
        new_outputs = current_outputs + (attn_mask_node,)
    else:
        new_outputs = (current_outputs, attn_mask_node)
    output_node.args = new_outputs
    gm.graph.output(new_outputs)
    gm.graph.erase_node(output_node)

    gm = clean_up_graph_after_modifications(gm)
    new_output_tensors = create_random_output_tensors(new_outputs)
    new_out_spec = pytree.tree_flatten(new_output_tensors)[1]
    gm._out_spec = new_out_spec
    return gm