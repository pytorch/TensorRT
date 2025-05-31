import copy
import logging
import operator
from typing import Callable, Sequence, Tuple

import torch
from sdpa_converter import *
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import args_bounds_check
from torch_tensorrt.dynamo.lowering import TORCH_TRT_DECOMPOSITIONS
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

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
    attn_mask = None
    is_causal = True
    for node in gm.graph.nodes:
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
                        attn_bias,
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
                    query, key, value, dropout_p, is_causal, return_debug_mask = (
                        node.args
                    )
                if len(node.args) == 5:
                    query, key, value, dropout_p, is_causal = (
                        node.args
                    )
                elif len(node.args) == 3:
                    query, key, value = node.args
                    dropout_p = 0.0
                    is_causal = True
                else:
                    raise ValueError(
                        f"Unexpected number of arguments for {node.target} in the graph"
                    )
            if attn_mask is not None:
                logger.warning(
                    f"This current version of SDPA converter does not support attn_mask for {node.target} in the graph. Ignoring it and using is_causal=True configuration."
                )

            modified_input_args = (query, key, value, None, dropout_p, is_causal)
            # Create a new node with torch.nn.functional.scaled_dot_product_attention
            # The input args is (query, key, value, is_causal). kwargs has scale
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    torch.nn.functional.scaled_dot_product_attention,
                    args=modified_input_args,
                    kwargs={"scale": node.kwargs.get("scale", None), "use_fp32_acc": settings.use_fp32_acc},
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
    return gm
