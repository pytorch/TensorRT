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

from .trt_sdpa_converter import *

logger = logging.getLogger(__name__)

_SDPA_OPS_TO_REMOVE = (
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
)


def _remove_decompositions():
    """
    Remove decompositions for SDPA operators.

    This function is idempotent. It ensures that the SDPA operators are removed
    from the decomposition table, allowing a custom converter to be used.
    """
    # Check if any of the decompositions still exist before proceeding
    if any(op in TORCH_TRT_DECOMPOSITIONS for op in _SDPA_OPS_TO_REMOVE):
        logger.debug("Removing SDPA decompositions to enable custom converter.")
        for op in _SDPA_OPS_TO_REMOVE:
            TORCH_TRT_DECOMPOSITIONS.pop(op, None)


REPLACEABLE_ATEN_OPS = {
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
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
    """
    Helper function to process SDPA nodes with common logic.

    This function handles the replacement of various scaled dot product attention operations
    with the standard torch.nn.functional.scaled_dot_product_attention function. It supports
    both efficient attention and flash attention variants, and can handle sliding window
    attention for models like Gemma3.

    Args:
        gm: The graph module containing the SDPA nodes
        node: The specific node to process (must be an SDPA operation)
        settings: TensorRT compilation settings
        sliding_window_size: Optional sliding window size for models with sliding attention
        use_gqa: Whether the model uses Grouped Query Attention

    Returns:
        The modified graph module with SDPA nodes replaced

    Raises:
        ValueError: If the SDPA node has an unexpected number of arguments
    """

    if node.target in [
        torch.ops.aten._scaled_dot_product_efficient_attention.default,
        torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    ]:
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

    logger.debug(
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
    """
    Register SDPA pass for Gemma3 models with sliding window attention.

    This function creates and registers a specialized SDPA replacement pass for Gemma3 models.
    The pass handles sliding window attention by extracting the sliding_window and layer_types
    configuration from the model config and applying appropriate transformations.

    Args:
        index: Position in the lowering pass list where this pass should be inserted
        model_config: The model configuration object (should be Gemma3TextConfig)

    Example:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("google/gemma-3-1b-it")
        register_gemma3_sdpa_pass(index=0, model_config=config)

    Note:
        This pass is specifically designed for Gemma3 models and will fall back to
        default behavior if the model_config is not a Gemma3TextConfig.
    """

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
    """
    Register default SDPA pass for models without specific implementations.

    This function creates and registers a default SDPA replacement pass that can be used
    for any model type. It provides basic SDPA replacement functionality without
    model-specific optimizations.

    Args:
        index: Position in the lowering pass list where this pass should be inserted
        model_config: The model configuration object (optional, for consistency)

    Example:
        # Register default pass at index 0
        register_default_sdpa_pass(index=0)

        # Or with model config for consistency
        config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
        register_default_sdpa_pass(index=0, model_config=config)

    Note:
        This is a fallback pass that should be used when no model-specific
        SDPA pass is available or when you want generic SDPA replacement behavior.
    """

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


def enable_sdpa_converter(model_name: str, model_config: Any) -> None:
    """
    Enables the custom SDPA converter for a given model.

    This function performs two main actions:
    1. Removes the default PyTorch SDPA decompositions from Torch-TensorRT's
       lowering registry. This is necessary to prevent them from being used
       instead of our custom converter.
    2. Registers a model-specific or default lowering pass that replaces the
       standard SDPA operators with a version optimized for TensorRT conversion.

    Args:
        model_name (str): The name of the model (e.g., from Hugging Face).
        model_config (Any): The model's configuration object. This is used to
                            extract parameters for model-specific optimizations,
                            like sliding window attention.
    """
    _remove_decompositions()

    pass_registrator = _SDPA_MAPPING.get(model_name)

    if pass_registrator:
        logger.info(f"Registering specific SDPA lowering pass for model: {model_name}")
        pass_registrator(model_config=model_config)
    else:
        logger.info(
            f"No specific SDPA lowering pass for model '{model_name}'. "
            "Using default SDPA pass."
        )
        _SDPA_MAPPING["default"](model_config=model_config)


# Global registry for SDPA passes
_SDPA_MAPPING: Dict[str, Callable] = {
    "google/gemma-3-1b-it": register_gemma3_sdpa_pass,
    "default": register_default_sdpa_pass,
}
