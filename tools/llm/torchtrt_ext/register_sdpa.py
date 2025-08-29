import copy
import logging
import operator
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Type

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
from transformers import AutoConfig, Gemma3TextConfig

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

from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    get_lowering_pass_config,
)


def _process_sdpa_node(
    gm: torch.fx.GraphModule,
    node: torch.fx.Node,
    settings: CompilationSettings,
    sliding_window_size: Optional[int] = None,
    use_gqa: bool = False,
) -> torch.fx.GraphModule:
    """Helper function to process SDPA nodes with common logic."""

    if node.target == torch.ops.aten._scaled_dot_product_efficient_attention.default:
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
    elif node.target == torch.ops.aten._scaled_dot_product_flash_attention.default:
        if len(node.args) == 6:
            (
                query,
                key,
                value,
                dropout_p,
                is_causal,
                return_debug_mask,
            ) = node.args
        elif len(node.args) == 5:
            query, key, value, dropout_p, is_causal = node.args
        elif len(node.args) == 3:
            query, key, value = node.args
            dropout_p = 0.0
            is_causal = True
        else:
            raise ValueError(
                f"Unexpected number of arguments for {node.target} in the graph"
            )
    else:
        return gm

    # Always set causal to True and generate attn_mask inside the sdpa operator
    attn_mask = None
    is_causal = True
    dropout_p = 0.0

    logger.warning(
        f"SDPA converter configuration: attn_mask={attn_mask}, dropout_p={dropout_p}, "
        f"is_causal={is_causal}, sliding_window_size={sliding_window_size}, use_gqa={use_gqa}"
    )

    modified_input_args = (
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
    )

    # Create a new node with torch.nn.functional.scaled_dot_product_attention
    with gm.graph.inserting_after(node):
        new_node = gm.graph.call_function(
            torch.nn.functional.scaled_dot_product_attention,
            args=modified_input_args,
            kwargs={
                "scale": node.kwargs.get("scale", None),
                "use_fp32_acc": settings.use_fp32_acc,
                "sliding_window_size": sliding_window_size,
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
    return gm


def register_gemma3_sdpa_pass(index: int = 0, model_config: Any = None) -> None:
    @_aten_lowering_pass(index=index, model_config=model_config)
    def gemma3_sdpa_pass(
        gm: torch.fx.GraphModule, settings: CompilationSettings
    ) -> torch.fx.GraphModule:
        """SDPA pass specifically for Gemma3 models with sliding window attention."""
        config = get_lowering_pass_config(gemma3_sdpa_pass)
        sliding_window = None
        layer_types = None
        model_config = config.get("model_config", None)
        if not isinstance(model_config, Gemma3TextConfig):
            logger.warning(
                f"Expected Gemma3TextConfig, got {type(model_config)}, will use default SDPA replacement instead"
            )
        else:
            sliding_window = getattr(model_config, "sliding_window", None)
            layer_types = getattr(model_config, "layer_types", None)
            logger.debug(
                f"got Gemma3 config: sliding_window={sliding_window}, layer_types={layer_types}"
            )

        index = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in REPLACEABLE_ATEN_OPS:
                sliding_window_size = None
                if (
                    sliding_window is not None
                    and sliding_window > 0
                    and layer_types is not None
                    and index < len(layer_types)
                ):
                    if layer_types[index] == "sliding_attention":
                        sliding_window_size = sliding_window
                index += 1

                # Process the node
                logger.debug(
                    f"Applying Gemma3-specific SDPA replacement with {node.name=}, {node.target=}, {sliding_window_size=}"
                )
                gm = _process_sdpa_node(gm, node, settings, sliding_window_size)

        clean_up_graph_after_modifications(gm)
        logger.debug("Applied Gemma3-specific SDPA replacement")
        return gm


def register_default_sdpa_pass(index: int = 0, model_config: Any = None) -> None:
    @_aten_lowering_pass(index=index, model_config=model_config)
    def default_sdpa_pass(
        gm: torch.fx.GraphModule,
        settings: CompilationSettings,
    ) -> torch.fx.GraphModule:
        """Default SDPA pass for models without specific implementations."""

        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target in REPLACEABLE_ATEN_OPS:
                # Process the node with default logic
                gm = _process_sdpa_node(gm, node, settings)

        clean_up_graph_after_modifications(gm)
        logger.debug("Applied default SDPA replacement")
        return gm


# Global registry for SDPA passes
_SDPA_MAPPING: Dict[str, Callable] = {
    "google/gemma-3-1b-it": register_gemma3_sdpa_pass,
    "default": register_default_sdpa_pass,
}
