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
from transformers import Gemma3TextConfig

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


def register_sdpa_pass_with_model_config(index: int = 0, model_config=None):
    """
    Register the SDPA replacement pass with a specific model configuration.

    Args:
        model_config: The model configuration object (e.g., from transformers.AutoConfig)
        index: Position in the lowering pass list (default: 0)

    Example:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("microsoft/DialoGPT-medium")
        register_sdpa_pass_with_model_config(config)
    """
    from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
        _aten_lowering_pass,
        _remove_lowering_pass,
    )

    # Create a new pass with the model configuration
    @_aten_lowering_pass(index=index, model_config=model_config)
    def replace_variants_of_sdpa_with_config(
        gm: torch.fx.GraphModule, settings: CompilationSettings
    ) -> torch.fx.GraphModule:
        """Replace scaled_dot_product_attention with model-specific configuration"""

        # Access the model configuration from the decorator parameters
        from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
            get_lowering_pass_config,
        )

        config = get_lowering_pass_config(replace_variants_of_sdpa_with_config)

        model_config = config.get("model_config", None)
        layer_types = []
        sliding_window = None
        # Extract model-specific parameters
        if model_config is not None:
            if isinstance(model_config, Gemma3TextConfig):
                sliding_window = getattr(model_config, "sliding_window", None)
                layer_types = getattr(model_config, "layer_types", None)
                logger.info(f"Model config: {sliding_window=} {layer_types=}")
        else:
            logger.warning(
                "No model configuration provided, using default SDPA replacement behavior"
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

                # always set_causal to True and generate attn_mask inside the sdpa operator, do not use the attn_mask from the transformers.
                attn_mask = None
                is_causal = True
                dropout_p = 0.0

                logger.warning(
                    f"This current version of SDPA converter only supports {attn_mask=}, {dropout_p=} and {is_causal=} and {sliding_window_size=}  configuration. This could cause issues with accuracy for models with different configurations."
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
                # The input args is (query, key, value, attn_mask, dropout_p, is_causal). kwargs has scale
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
                        if (
                            user.op == "call_function"
                            and user.target == operator.getitem
                        ):
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

        if model_config:
            logger.debug(
                f"Replaced variants of scaled_dot_product_attention for {getattr(model_config, 'model_type', 'unknown')} model"
            )
        else:
            logger.debug(
                "Replaced variants of scaled_dot_product_attention with torch.nn.functional.scaled_dot_product_attention"
            )
        add_attn_mask_as_output = False
        if add_attn_mask_as_output:
            add_one_attn_mask_as_output(gm)
        return gm

    logger.info(
        f"Registered SDPA pass with model config: {getattr(model_config, 'model_type', 'unknown')}"
    )
    return replace_variants_of_sdpa_with_config
