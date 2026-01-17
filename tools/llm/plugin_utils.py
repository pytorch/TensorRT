"""
Plugin utilities for TensorRT LLM inference with custom attention plugins.

This module provides model-agnostic utilities for using TensorRT attention plugins
with various LLM architectures (Qwen, Llama, etc.).
"""

import ctypes
import inspect
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt
from torch_tensorrt.dynamo.conversion import (
    ConversionContext,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.converter_utils import get_trt_tensor

# Default plugin path - can be overridden
DEFAULT_PLUGIN_PATH = (
    "/develop/TensorRT/TensorRT-Edge-LLM-release/build/libNvInfer_edgellm_plugin.so"
)

# Global configuration for plugin converter
_PLUGIN_CONFIG: Dict[str, Any] = {}


def load_plugin(plugin_path: Optional[str] = None) -> bool:
    """
    Load the TensorRT attention plugin library.

    Args:
        plugin_path: Path to the plugin .so file. If None, uses DEFAULT_PLUGIN_PATH.

    Returns:
        True if plugin was loaded successfully, False otherwise.

    Raises:
        RuntimeError: If plugin file does not exist.
    """
    path = plugin_path or DEFAULT_PLUGIN_PATH
    if not os.path.exists(path):
        raise RuntimeError(f"Plugin not found at {path}")
    ctypes.CDLL(path)
    return True


def set_plugin_config(
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
) -> None:
    """
    Set global configuration for the plugin converter.

    Args:
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value attention heads (for GQA).
        head_dim: Dimension of each attention head.
        max_seq_len: Maximum sequence length for KV cache.
        max_batch_size: Maximum batch size.
    """
    global _PLUGIN_CONFIG
    _PLUGIN_CONFIG = {
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "max_seq_len": max_seq_len,
        "max_batch_size": max_batch_size,
    }


def get_plugin_config() -> Dict[str, Any]:
    """Get the current plugin configuration."""
    return _PLUGIN_CONFIG.copy()


def set_plugin_config_from_model(model_config: Any, max_seq_len: int = 2048) -> None:
    """
    Set plugin configuration from a HuggingFace model config.

    Args:
        model_config: HuggingFace model configuration object.
        max_seq_len: Maximum sequence length for KV cache.
    """
    # Qwen3 has explicit head_dim in config that differs from hidden_size // num_attention_heads
    if hasattr(model_config, "head_dim") and model_config.head_dim is not None:
        head_dim = model_config.head_dim
    else:
        head_dim = model_config.hidden_size // model_config.num_attention_heads

    set_plugin_config(
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
    )


# -----------------------------------------------------------------------------
# Plugin Op Registration
# -----------------------------------------------------------------------------


def _register_plugin_op_impl() -> None:
    """
    Internal implementation to register the xqa::attn custom op for PyTorch.

    Note: The release version of TensorRT-Edge-LLM requires 5 inputs:
    - qkv: [B, S, (Hq+Hk+Hv)*D] fused QKV tensor
    - kv: [B, 2, Hkv, Capacity, D] KV cache tensor
    - ctx_len: [B] context length per batch
    - rope: [S, D] rotary position encoding
    - kv_cache_start_idx: [B] starting index in KV cache (required for release version)
    """

    @torch.library.custom_op("xqa::attn", mutates_args=())
    def attn(
        qkv: torch.Tensor,
        kv: torch.Tensor,
        ctx_len: torch.Tensor,
        rope: torch.Tensor,
        kv_cache_start_idx: torch.Tensor,  # Required 5th input for release plugin
        nq: int,
        nkv: int,
        d: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        attn_out = torch.zeros(
            batch_size, seq_len, nq, d, dtype=qkv.dtype, device=qkv.device
        )
        updated_kv = kv.clone()
        return attn_out, updated_kv

    @torch.library.register_fake("xqa::attn")
    def _(qkv, kv, ctx_len, rope, kv_cache_start_idx, nq, nkv, d):
        batch_size = qkv.shape[0]
        seq_len = qkv.shape[1]
        attn_out = torch.empty(
            batch_size, seq_len, nq, d, dtype=qkv.dtype, device=qkv.device
        )
        updated_kv = kv.clone()
        return attn_out, updated_kv


def register_plugin_op() -> None:
    """
    Register the xqa::attn custom op for PyTorch.

    This function is idempotent - safe to call multiple times.
    """
    if hasattr(torch.ops, "xqa") and hasattr(torch.ops.xqa, "attn"):
        return
    _register_plugin_op_impl()


# Register the op at module import time so the converter decorator works
# This is safe because the op registration is idempotent
if not (hasattr(torch.ops, "xqa") and hasattr(torch.ops.xqa, "attn")):
    _register_plugin_op_impl()


# -----------------------------------------------------------------------------
# TensorRT Converter
# -----------------------------------------------------------------------------


@dynamo_tensorrt_converter(torch.ops.xqa.attn.default, supports_dynamic_shapes=True)
def convert_attn(ctx: ConversionContext, target, args, kwargs, name):
    """
    Convert xqa::attn op to TensorRT AttentionPlugin.

    Release version of TensorRT-Edge-LLM requires 5 inputs:
    - qkv, kv, ctx_len, rope, kv_cache_start_idx

    Plugin fields for release version:
    - num_q_heads, num_kv_heads, head_size, enable_tree_attention, enable_delta_kv_output
    """
    # args: qkv, kv, ctx_len, rope, kv_cache_start_idx, nq, nkv, d
    qkv, kv, ctx_len, rope, kv_cache_start_idx, nq, nkv, d = args[:8]

    creator = trt.get_plugin_registry().get_plugin_creator("AttentionPlugin", "1", "")
    if creator is None:
        raise RuntimeError("AttentionPlugin not found in TensorRT plugin registry!")

    # Get config from global settings
    config = get_plugin_config()
    if config:
        nq_val = config["num_attention_heads"]
        nkv_val = config["num_key_value_heads"]
        d_val = config["head_dim"]
    else:
        # Fallback to values from args (may not work correctly)
        nq_val = nq if isinstance(nq, int) else 14
        nkv_val = nkv if isinstance(nkv, int) else 2
        d_val = d if isinstance(d, int) else 64

    # Plugin fields for release version of TensorRT-Edge-LLM
    field_list = [
        trt.PluginField(
            "num_q_heads", np.array([nq_val], dtype=np.int32), trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            "num_kv_heads",
            np.array([nkv_val], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "head_size", np.array([d_val], dtype=np.int32), trt.PluginFieldType.INT32
        ),
        trt.PluginField(
            "enable_tree_attention",
            np.array([0], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
        trt.PluginField(
            "enable_delta_kv_output",
            np.array([1], dtype=np.int32),
            trt.PluginFieldType.INT32,
        ),
    ]

    fields = trt.PluginFieldCollection(field_list)
    plugin = creator.create_plugin(name, fields)

    # 5 inputs for release version: qkv, kv, ctx_len, rope, kv_cache_start_idx
    inputs = [
        (
            get_trt_tensor(ctx, i, f"{name}_i{idx}")
            if not isinstance(i, trt.ITensor)
            else i
        )
        for idx, i in enumerate([qkv, kv, ctx_len, rope, kv_cache_start_idx])
    ]

    # Handle ctx_len shape if needed (squeeze if [B, 1] -> [B])
    if len(inputs[2].shape) == 2 and inputs[2].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[2])
        shuffle_layer.reshape_dims = (inputs[2].shape[0],)
        inputs[2] = shuffle_layer.get_output(0)

    # Handle kv_cache_start_idx shape if needed (squeeze if [B, 1] -> [B])
    if len(inputs[4].shape) == 2 and inputs[4].shape[1] == 1:
        shuffle_layer = ctx.net.add_shuffle(inputs[4])
        shuffle_layer.reshape_dims = (inputs[4].shape[0],)
        inputs[4] = shuffle_layer.get_output(0)

    layer = ctx.net.add_plugin_v2(inputs, plugin)
    return layer.get_output(0), layer.get_output(1)


# -----------------------------------------------------------------------------
# RoPE Cache Generation
# -----------------------------------------------------------------------------


def get_plugin_rope_cache(
    rotary_emb: nn.Module,
    max_seq_len: int,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate RoPE cache tensor for the plugin from a rotary embedding module.

    Args:
        rotary_emb: The rotary embedding module from the model.
        max_seq_len: Maximum sequence length.
        head_dim: Dimension of each attention head.
        device: Device to create the cache on.

    Returns:
        RoPE cache tensor of shape [1, max_seq_len, head_dim].
    """
    inv_freq = rotary_emb.inv_freq.to(device).float()
    attention_scaling = getattr(rotary_emb, "attention_scaling", 1.0)
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos_half = freqs.cos() * attention_scaling
    sin_half = freqs.sin() * attention_scaling
    rope = torch.cat([cos_half, sin_half], dim=-1)
    return rope.unsqueeze(0)


# -----------------------------------------------------------------------------
# Plugin Attention Module
# -----------------------------------------------------------------------------


class PluginAttention(nn.Module):
    """
    Model-agnostic Plugin Attention module that replaces standard attention.

    This module wraps the projection layers from the original attention module
    and uses the xqa::attn plugin op for the attention computation.

    Supports:
    - Qwen2.5, Llama: Standard attention
    - Qwen3: Attention with QK Normalization (q_norm, k_norm)
    """

    def __init__(
        self,
        original_attn: nn.Module,
        config: Any,
        layer_idx: int,
        rope_cache: torch.Tensor,
    ):
        """
        Initialize PluginAttention.

        Args:
            original_attn: The original attention module to wrap.
            config: Model configuration.
            layer_idx: Index of this layer in the model.
            rope_cache: Pre-computed RoPE cache tensor.
        """
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # Qwen3 has QK Normalization
        self.q_norm = getattr(original_attn, "q_norm", None)
        self.k_norm = getattr(original_attn, "k_norm", None)

        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads

        # Qwen3 has explicit head_dim that may differ from hidden_size // num_attention_heads
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // config.num_attention_heads

        # For Qwen3, attention output size is num_heads * head_dim, not hidden_size
        self.attn_hidden_size = self.num_heads * self.head_dim
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.register_buffer("rope_cache", rope_cache)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        ctx_len: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using the plugin attention.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Unused (plugin handles masking internally).
            position_ids: Position IDs (unused, plugin uses RoPE cache).
            past_key_value: KV cache tensor of shape [batch, 2, num_kv_heads, capacity, head_dim].
            ctx_len: Context length tensor for each batch item.

        Returns:
            Tuple of (output tensor, updated KV cache).
        """
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Qwen3: Apply QK Normalization if available
        if self.q_norm is not None:
            # Reshape for per-head normalization: [B, S, num_heads, head_dim]
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            q = self.q_norm(q)
            q = q.view(batch_size, seq_len, -1)

        if self.k_norm is not None:
            # Reshape for per-head normalization: [B, S, num_kv_heads, head_dim]
            k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
            k = self.k_norm(k)
            k = k.view(batch_size, seq_len, -1)

        qkv = torch.cat([q, k, v], dim=-1)

        if ctx_len is None:
            ctx_len = torch.tensor(
                [seq_len], dtype=torch.int32, device=hidden_states.device
            ).expand(batch_size)

        rope_fp32 = self.rope_cache.float()

        if past_key_value is None:
            raise ValueError("past_key_value (KV cache tensor) must be provided")

        # kv_cache_start_idx: starting position in KV cache for each batch
        # For normal inference, this is 0 (start from beginning)
        kv_cache_start_idx = torch.zeros(
            batch_size, dtype=torch.int32, device=hidden_states.device
        )

        attn_out, updated_kv = torch.ops.xqa.attn.default(
            qkv,
            past_key_value,
            ctx_len,
            rope_fp32,
            kv_cache_start_idx,
            self.num_heads,
            self.num_key_value_heads,
            self.head_dim,
        )

        # Use attn_hidden_size for reshape (may differ from hidden_size in Qwen3)
        attn_out = attn_out.reshape(batch_size, seq_len, self.attn_hidden_size)
        output = self.o_proj(attn_out)
        return output, updated_kv


# -----------------------------------------------------------------------------
# Model Wrappers
# -----------------------------------------------------------------------------


class LLMPluginWrapper(nn.Module):
    """
    Generic wrapper for LLM models with plugin attention.

    This wrapper handles the forward pass for models with replaced attention modules,
    managing KV caches and context lengths appropriately.
    """

    def __init__(self, model: nn.Module, model_type: str = "auto"):
        """
        Initialize the wrapper.

        Args:
            model: The model with replaced attention modules.
            model_type: Type of model ("qwen", "llama", or "auto" for auto-detection).
        """
        super().__init__()
        self.model = model
        self.model_type = (
            self._detect_model_type(model) if model_type == "auto" else model_type
        )

    def _detect_model_type(self, model: nn.Module) -> str:
        """Auto-detect model type from model structure."""
        model_class = model.__class__.__name__.lower()
        if "qwen" in model_class:
            return "qwen"
        elif "llama" in model_class or "mistral" in model_class:
            return "llama"
        else:
            # Default to generic transformer structure
            return "generic"

    def _get_transformer(self) -> nn.Module:
        """Get the transformer backbone based on model type."""
        if self.model_type == "qwen":
            return self.model.model
        elif self.model_type == "llama":
            return self.model.model
        else:
            # Try common attribute names
            for attr in ["model", "transformer", "backbone"]:
                if hasattr(self.model, attr):
                    return getattr(self.model, attr)
            raise ValueError(
                f"Cannot find transformer backbone for model type: {self.model_type}"
            )

    def _get_layers(self, transformer: nn.Module) -> nn.ModuleList:
        """Get the list of transformer layers."""
        for attr in ["layers", "h", "blocks"]:
            if hasattr(transformer, attr):
                return getattr(transformer, attr)
        raise ValueError("Cannot find transformer layers")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        ctx_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass with plugin attention.

        Args:
            input_ids: Input token IDs [batch, seq_len].
            position_ids: Position IDs [batch, seq_len].
            kv_caches: List of KV cache tensors, one per layer.
            ctx_len: Context length tensor [batch].

        Returns:
            Tuple of (logits, list of updated KV caches).
        """
        transformer = self._get_transformer()
        hidden_states = transformer.embed_tokens(input_ids)

        layers = self._get_layers(transformer)
        new_kv_caches = []

        for i, layer in enumerate(layers):
            past_key_value = kv_caches[i]
            residual = hidden_states

            # Input layer norm
            if hasattr(layer, "input_layernorm"):
                hidden_states = layer.input_layernorm(hidden_states)
            elif hasattr(layer, "ln_1"):
                hidden_states = layer.ln_1(hidden_states)

            # Self attention
            hidden_states, updated_kv = layer.self_attn(
                hidden_states=hidden_states,
                attention_mask=None,
                position_ids=position_ids,
                past_key_value=past_key_value,
                ctx_len=ctx_len,
            )
            hidden_states = residual + hidden_states

            # Post attention layer norm + MLP
            residual = hidden_states
            if hasattr(layer, "post_attention_layernorm"):
                hidden_states = layer.post_attention_layernorm(hidden_states)
            elif hasattr(layer, "ln_2"):
                hidden_states = layer.ln_2(hidden_states)
            hidden_states = layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            new_kv_caches.append(updated_kv)

        # Final layer norm
        if hasattr(transformer, "norm"):
            hidden_states = transformer.norm(hidden_states)
        elif hasattr(transformer, "ln_f"):
            hidden_states = transformer.ln_f(hidden_states)

        # LM head
        logits = self.model.lm_head(hidden_states)

        return logits, new_kv_caches


# -----------------------------------------------------------------------------
# Model Modification Functions
# -----------------------------------------------------------------------------


def replace_attention_with_plugin(
    model: nn.Module,
    config: Any,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Replace all attention modules in a model with PluginAttention.

    Args:
        model: The HuggingFace model to modify.
        config: Model configuration.
        max_seq_len: Maximum sequence length for RoPE cache.
        device: Device for the model.
        dtype: Data type for the model.

    Returns:
        The modified model with plugin attention.
    """
    # Get rotary embedding from model
    transformer = model.model if hasattr(model, "model") else model

    # Try to find rotary embedding
    rotary_emb = None
    if hasattr(transformer, "rotary_emb"):
        rotary_emb = transformer.rotary_emb
    elif hasattr(transformer, "layers") and len(transformer.layers) > 0:
        first_layer = transformer.layers[0]
        if hasattr(first_layer, "self_attn") and hasattr(
            first_layer.self_attn, "rotary_emb"
        ):
            rotary_emb = first_layer.self_attn.rotary_emb

    if rotary_emb is None:
        raise ValueError("Cannot find rotary embedding in model")

    # Qwen3 has explicit head_dim that may differ from hidden_size // num_attention_heads
    if hasattr(config, "head_dim") and config.head_dim is not None:
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads
    rope_cache = get_plugin_rope_cache(rotary_emb, max_seq_len, head_dim, device)

    # Get layers
    if hasattr(transformer, "layers"):
        layers = transformer.layers
    elif hasattr(transformer, "h"):
        layers = transformer.h
    else:
        raise ValueError("Cannot find transformer layers")

    # Replace attention modules
    for i, layer in enumerate(layers):
        layer.self_attn = PluginAttention(layer.self_attn, config, i, rope_cache)

    return model


# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------


def compile_plugin_model(
    model: nn.Module,
    config: Any,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    debug: bool = False,
) -> Callable:
    """
    Compile a model with plugin attention for TensorRT inference.

    Args:
        model: The wrapped model (should be LLMPluginWrapper or similar).
        config: Model configuration.
        max_seq_len: Maximum sequence length.
        device: Device for compilation.
        dtype: Data type.
        debug: Whether to enable debug logging.

    Returns:
        Compiled TensorRT model function.
    """
    # Prepare dummy inputs
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    # Qwen3 has explicit head_dim that may differ from hidden_size // num_attention_heads
    if hasattr(config, "head_dim") and config.head_dim is not None:
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    dummy_input_ids = torch.tensor([[1, 2, 3]], device=device)
    dummy_pos_ids = torch.tensor([[0, 1, 2]], device=device)
    dummy_ctx_len = torch.tensor([3], dtype=torch.int32, device=device)
    dummy_kvs = [
        torch.zeros(
            1, 2, num_kv_heads, max_seq_len, head_dim, dtype=dtype, device=device
        )
        for _ in range(num_layers)
    ]

    # Dynamic shapes
    seq_len_dim = torch.export.Dim("seq_len", min=1, max=max_seq_len)
    kv_cache_dynamics = [{}] * num_layers
    dynamic_shapes = {
        "input_ids": {1: seq_len_dim},
        "position_ids": {1: seq_len_dim},
        "kv_caches": kv_cache_dynamics,
        "ctx_len": {},
    }

    # Export
    ep = torch.export.export(
        model,
        args=(dummy_input_ids, dummy_pos_ids, dummy_kvs, dummy_ctx_len),
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # Compile
    trt_model = torch_tensorrt.dynamo.compile(
        ep,
        inputs=[dummy_input_ids, dummy_pos_ids, dummy_kvs, dummy_ctx_len],
        enabled_precisions={torch.float32},
        use_explicit_typing=True,
        use_fp32_acc=True,
        device=device,
        disable_tf32=True,
        min_block_size=1,
        debug=debug,
    )

    return trt_model


# -----------------------------------------------------------------------------
# KV Cache Utilities
# -----------------------------------------------------------------------------


def create_kv_caches(
    config: Any,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """
    Create empty KV cache tensors for all layers.

    Args:
        config: Model configuration.
        max_seq_len: Maximum sequence length (capacity).
        batch_size: Batch size.
        device: Device to create tensors on.
        dtype: Data type for the tensors.

    Returns:
        List of KV cache tensors, one per layer.
    """
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    # Qwen3 has explicit head_dim that may differ from hidden_size // num_attention_heads
    if hasattr(config, "head_dim") and config.head_dim is not None:
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads

    return [
        torch.zeros(
            batch_size,
            2,
            num_kv_heads,
            max_seq_len,
            head_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]


# -----------------------------------------------------------------------------
# Generation Utilities
# -----------------------------------------------------------------------------


def generate_with_plugin(
    model_func: Callable,
    input_ids: torch.Tensor,
    kv_caches: List[torch.Tensor],
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Generate tokens using the plugin model.

    Args:
        model_func: The compiled model function.
        input_ids: Input token IDs [batch, seq_len].
        kv_caches: List of KV cache tensors.
        max_new_tokens: Maximum number of new tokens to generate.
        eos_token_id: EOS token ID for early stopping (optional).
        device: Device for computation.

    Returns:
        Tuple of (generated token IDs, updated KV caches).
    """
    generated_ids = input_ids.clone()
    seq_len = input_ids.shape[1]

    # Prefill
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    ctx_len = torch.tensor([seq_len], dtype=torch.int32, device=device)

    output = model_func(input_ids, position_ids, kv_caches, ctx_len)

    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            logits, delta_kvs = output
        else:
            logits = output[0]
            delta_kvs = output[1:]
    else:
        logits = output
        delta_kvs = []

    # Update KV caches
    if len(delta_kvs) > 0:
        for i, delta in enumerate(delta_kvs):
            seq_len_out = delta.shape[3]
            kv_caches[i][:, :, :, :seq_len_out, :] = delta

    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Check for EOS
    if eos_token_id is not None and next_token.item() == eos_token_id:
        return generated_ids, kv_caches

    # Decode
    cur_pos = seq_len

    for _ in range(max_new_tokens - 1):
        input_ids_step = next_token
        position_ids_step = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        ctx_len_step = torch.tensor([cur_pos + 1], dtype=torch.int32, device=device)

        output = model_func(input_ids_step, position_ids_step, kv_caches, ctx_len_step)

        if isinstance(output, (tuple, list)):
            if len(output) == 2:
                logits, delta_kvs = output
            else:
                logits = output[0]
                delta_kvs = output[1:]

        # Update KV caches
        if len(delta_kvs) > 0:
            for i, delta in enumerate(delta_kvs):
                kv_caches[i][:, :, :, cur_pos : cur_pos + 1, :] = delta

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        cur_pos += 1

        # Check for EOS
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

    return generated_ids, kv_caches


def benchmark_plugin_generation(
    model_func: Callable,
    config: Any,
    isl: int,
    osl: int,
    max_seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
    run_name: str = "Plugin",
) -> float:
    """
    Benchmark plugin model generation.

    Args:
        model_func: The compiled model function.
        config: Model configuration.
        isl: Input sequence length.
        osl: Output sequence length (number of tokens to generate).
        max_seq_len: Maximum sequence length for KV cache.
        device: Device for computation.
        dtype: Data type.
        run_name: Name for logging.

    Returns:
        Elapsed time in milliseconds.
    """
    # Check for extra kwargs the model might need
    extra_kwargs = {}
    if hasattr(model_func, "forward"):
        sig = inspect.signature(model_func.forward)
        if "arg_start_idx" in sig.parameters:
            extra_kwargs["arg_start_idx"] = 0
        if "arg_end_idx" in sig.parameters:
            extra_kwargs["arg_end_idx"] = 0

    # Prepare inputs
    input_ids = torch.randint(0, config.vocab_size, (1, isl), device=device)
    kv_caches = create_kv_caches(config, max_seq_len, 1, device, dtype)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    # Prefill
    seq_len = isl
    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    ctx_len = torch.tensor([seq_len], dtype=torch.int32, device=device)

    output = model_func(input_ids, position_ids, kv_caches, ctx_len, **extra_kwargs)

    if isinstance(output, (tuple, list)):
        if len(output) == 2:
            logits, delta_kvs = output
        else:
            logits = output[0]
            delta_kvs = output[1:]
    else:
        logits = output
        delta_kvs = []

    # Update KV caches
    if len(delta_kvs) > 0:
        for i, delta in enumerate(delta_kvs):
            seq_len_out = delta.shape[3]
            kv_caches[i][:, :, :, :seq_len_out, :] = delta

    next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)

    # Decode
    cur_pos = seq_len

    for _ in range(osl - 1):
        input_ids_step = next_token
        position_ids_step = torch.tensor([[cur_pos]], dtype=torch.long, device=device)
        ctx_len_step = torch.tensor([cur_pos + 1], dtype=torch.int32, device=device)

        output = model_func(
            input_ids_step, position_ids_step, kv_caches, ctx_len_step, **extra_kwargs
        )

        if isinstance(output, (tuple, list)):
            if len(output) == 2:
                logits, delta_kvs = output
            else:
                logits = output[0]
                delta_kvs = output[1:]

        # Update KV caches
        if len(delta_kvs) > 0:
            for i, delta in enumerate(delta_kvs):
                kv_caches[i][:, :, :, cur_pos : cur_pos + 1, :] = delta

        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
        cur_pos += 1

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print(
        f"{run_name} | ISL: {isl}, OSL: {osl} | Total Time: {elapsed_ms:.2f} ms | Tokens/sec: {osl / (elapsed_ms / 1000.0):.2f}"
    )
    return elapsed_ms
