import copy
import logging
import operator
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo.conversion.aten_ops_converters import args_bounds_check
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)
REPLACEABLE_ATEN_OPS = {
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
}


def lower_scaled_dot_product_attention(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Replace specific versions of scaled_dot_product_attention with an equivalent
    implementation which can be easily converted to TRT
    """
    original_fns, replacement = scaled_dot_product_attention_replacement()
    replaced_nodes = []
    # For each original function, search for it in the graph and replace
    for original in original_fns:
        replaced_nodes += torch.fx.subgraph_rewriter.replace_pattern_with_filters(
            gm,
            original,
            replacement,
            ignore_literals=True,
        )

    if replaced_nodes:
        # Repair instances which use the kwargs field (specifically the "scale" kwarg)
        # Also repair instances which specified the is_causal or attn_bias fields
        for match in replaced_nodes:
            attention_node_replaced = None
            # Seek the attention operator being replaced
            for node in match.nodes_map:
                if node.target in REPLACEABLE_ATEN_OPS:
                    attention_node_replaced = match.nodes_map[node]
                    break

            assert attention_node_replaced is not None
            assert len(match.replacements) == 1

            new_attention_node = match.replacements[0]

            assert (
                new_attention_node.target
                == torch.nn.functional.scaled_dot_product_attention
            )

            # Copy the metadata of the replaced attention node to the new node
            # TODO: Investigate why there are multiple FakeTensors in the metadata.
            # We only use the first one as it contains the output shape information for this node.
            if "val" in attention_node_replaced.meta:
                new_attention_node.meta["val"] = copy.copy(
                    attention_node_replaced.meta["val"][0]
                )

            # If the attention operator had keyword-args, copy them to the new node
            if attention_node_replaced.kwargs:
                new_attention_node.kwargs = {**attention_node_replaced.kwargs}

            # Set default args in new node:
            # Tensor? attn_mask=None, float dropout_p=0.0, bool is_causal=False
            new_attention_node.args = new_attention_node.args + (None, 0.0, False)

            # The `is_causal` argument was specified
            if (
                (
                    attention_node_replaced.target
                    == torch.ops.aten._scaled_dot_product_flash_attention.default
                )
                and args_bounds_check(attention_node_replaced.args, 4, False)
            ) or (
                (
                    attention_node_replaced.target
                    == torch.ops.aten._scaled_dot_product_efficient_attention.default
                )
                and args_bounds_check(attention_node_replaced.args, 6, False)
            ):
                new_attention_node.args = (
                    new_attention_node.args[:5] + (True,) + new_attention_node.args[6:]
                )

            # The `attn_bias` argument was specified
            if (
                attention_node_replaced.target
                == torch.ops.aten._scaled_dot_product_efficient_attention.default
            ) and args_bounds_check(attention_node_replaced.args, 3) is not None:
                new_attention_node.args = (
                    new_attention_node.args[:3]
                    + attention_node_replaced.args[3]
                    + new_attention_node.args[4:]
                )

        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after lowering scaled dot product attention:\n{gm.graph}")

    return gm


def scaled_dot_product_attention_replacement() -> Tuple[
    Sequence[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]],
    Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
]:
    """Constructs the original and replacement functions for efficient attention"""

    # Efficient Attention original graph
    def efficient(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        outputs = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            q,
            k,
            v,
            None,
            False,
        )
        out = operator.getitem(outputs, 0)
        return out

    # Flash Attention original graph
    def flash(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        outputs = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q,
            k,
            v,
        )
        out = operator.getitem(outputs, 0)
        return out

    # Efficient Attention w/Scale original graph
    def efficient_scale(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        outputs = torch.ops.aten._scaled_dot_product_efficient_attention.default(
            q,
            k,
            v,
            None,
            False,
            scale=1.0,
        )
        out = operator.getitem(outputs, 0)
        return out

    # Flash Attention w/Scale original graph
    def flash_scale(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        outputs = torch.ops.aten._scaled_dot_product_flash_attention.default(
            q,
            k,
            v,
            scale=1.0,
        )
        out = operator.getitem(outputs, 0)
        return out

    # Replacement graph consists of the functional version of scaled_dot_product_attention
    def replacement(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(query, key, value)

    return (efficient, flash, efficient_scale, flash_scale), replacement
