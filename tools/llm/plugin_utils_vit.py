"""
Plugin utilities for TensorRT ViT inference with custom attention plugins.

This module provides Vision Transformer-specific utilities for using TensorRT
attention plugins with ViT models. Unlike LLMs, ViT models:
- Do not use KV caching (full bidirectional attention)
- Do not use RoPE (learnable/absolute position embeddings)
- Process fixed-size image patches at once
"""

import ctypes
import os
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorrt as trt
import torch
import torch.nn as nn
import torch_tensorrt

_TENSORRT_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_WORKSPACE_ROOT = os.path.dirname(_TENSORRT_REPO_ROOT)

# Default plugin path for ViT attention plugin. TensorRT-Edge-LLM is checked out
# next to TensorRT in this workspace, not inside the TensorRT repository.
DEFAULT_PLUGIN_PATH = os.path.join(
    _WORKSPACE_ROOT,
    "TensorRT-Edge-LLM",
    "build",
    "libNvInfer_edgellm_plugin.so",
)

# Global configuration for ViT plugin converter
_VIT_PLUGIN_CONFIG: Dict[str, Any] = {}

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
    path = plugin_path or os.environ.get("TRT_EDGE_LLM_PLUGIN_PATH") or DEFAULT_PLUGIN_PATH
    if not os.path.exists(path):
        raise RuntimeError(f"Plugin not found at {path}")
    ctypes.CDLL(path)
    print(f"Loaded plugin: {path}")
    return True


def set_vit_plugin_config(
    num_attention_heads: int,
    head_dim: int,
    num_patches: int,
    max_batch_size: int = 4,
    mask_type: int = 0,
    max_seq_len: int = 0,
) -> None:
    """
    Set global configuration for the ViT plugin converter.

    Args:
        num_attention_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        num_patches: Number of image patches (including [CLS] token).
        max_batch_size: Maximum batch size.
        mask_type: Plugin mask mode. 0=dense additive mask, 1=packed cu_seqlens.
        max_seq_len: Maximum packed segment length for cu_seqlens FMHA.
    """
    global _VIT_PLUGIN_CONFIG
    _VIT_PLUGIN_CONFIG = {
        "num_attention_heads": num_attention_heads,
        "head_dim": head_dim,
        "num_patches": num_patches,
        "max_batch_size": max_batch_size,
        "mask_type": mask_type,
        "max_seq_len": max_seq_len,
    }

def get_vit_plugin_config() -> Dict[str, Any]:
    """Get the current ViT plugin configuration."""
    return _VIT_PLUGIN_CONFIG.copy()

def set_vit_plugin_config_from_model(model_config: Any) -> None:
    """
    Set ViT plugin configuration from a HuggingFace vision model config.

    Args:
        model_config: HuggingFace model configuration object.
    """
    # HuggingFace vision configs use slightly different field names across
    # families. Plain ViT uses num_attention_heads; Mllama/Llama Vision uses
    # attention_heads.
    num_heads = getattr(model_config, "num_attention_heads", None) or getattr(
        model_config, "attention_heads"
    )
    head_dim = model_config.hidden_size // num_heads
    
    # Calculate number of patches from image size
    image_size = model_config.image_size
    patch_size = model_config.patch_size
    if isinstance(image_size, (tuple, list)):
        image_h, image_w = image_size
    else:
        image_h = image_w = image_size
    if isinstance(patch_size, (tuple, list)):
        patch_h, patch_w = patch_size
    else:
        patch_h = patch_w = patch_size
    num_patches = (image_h // patch_h) * (image_w // patch_w) + 1  # +1 for [CLS]

    set_vit_plugin_config(
        num_attention_heads=num_heads,
        head_dim=head_dim,
        num_patches=num_patches,
    )


# -----------------------------------------------------------------------------
# Plugin Op Registration
# -----------------------------------------------------------------------------

def _register_vit_plugin_op_impl() -> None:
    """
    Internal implementation to register the tensorrt_vit::attention custom op for PyTorch.

    ViT attention differs from LLM attention:
    - No KV cache - full bidirectional attention
    - Simple fused QKV input
    - Single output - no separate KV output
    """

    @torch.library.custom_op("tensorrt_vit::attention", mutates_args=())
    def attn(
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask_or_cu_seqlens: torch.Tensor,
        num_heads: int,
        head_dim: int,
        qkv_fused: int = 1,
        mask_type: int = 0,
        max_seq_len: int = 0,
    ) -> torch.Tensor:
        """
        ViT attention operation.

        Args:
            qkv: Fused [Q, K, V] tensor of shape [B, S, (H*D*3)].
            cos: RoPE cosine tensor of shape [S, D].
            sin: RoPE sine tensor of shape [S, D].
            mask_or_cu_seqlens: Dense additive mask or INT32 cu_seqlens.
            num_heads: Number of attention heads.
            head_dim: Dimension per head.
            qkv_fused: Whether QKV is fused (1=yes, 0=no).
            mask_type: 0 for dense additive mask, 1 for packed cu_seqlens.
            max_seq_len: Max segment length when mask_type=1.

        Returns:
            Attention output of shape [B, S, H*D].
        """
        batch_size, seq_len, _ = qkv.shape
        output_dim = num_heads * head_dim
        attn_out = torch.zeros(
            batch_size, seq_len, output_dim, dtype=qkv.dtype, device=qkv.device
        )
        return attn_out

    @torch.library.register_fake("tensorrt_vit::attention")
    def _(qkv, cos, sin, mask_or_cu_seqlens, num_heads, head_dim, qkv_fused=1, mask_type=0, max_seq_len=0):
        batch_size, seq_len, _ = qkv.shape
        output_dim = num_heads * head_dim
        attn_out = torch.empty(
            batch_size, seq_len, output_dim, dtype=qkv.dtype, device=qkv.device
        )
        return attn_out


def register_vit_plugin_op() -> None:
    """
    Register the tensorrt_vit::attention custom op for PyTorch.

    This function is idempotent - safe to call multiple times.
    """
    if hasattr(torch.ops, "tensorrt_vit") and hasattr(
        torch.ops.tensorrt_vit, "attention"
    ):
        return
    _register_vit_plugin_op_impl()


# Register the op at module import time so the converter decorator works
if not (
    hasattr(torch.ops, "tensorrt_vit")
    and hasattr(torch.ops.tensorrt_vit, "attention")
):
    _register_vit_plugin_op_impl()

# Importing plugin_converter_vit at the bottom of this file registers the
# Torch-TensorRT converter for tensorrt_vit::attention.

from plugin_converter_vit import (  # noqa: E402 (must be after op registration)
    convert_vit_attention,
    get_vit_plugin_conversion_count,
    reset_vit_plugin_conversion_count,
)

# -----------------------------------------------------------------------------
# Plugin Attention Module
# -----------------------------------------------------------------------------

class ViTPluginAttention(nn.Module):
    """
    Model-agnostic ViT attention wrapper using the TensorRT ViT attention plugin.

    The wrapper follows the same idea as the LLM plugin path: infer the attention
    module layout from the original module instead of requiring a separate hand
    written implementation for every model family. It supports common vision
    attention layouts:
    - fused QKV projection: qkv + proj (Qwen-VL style)
    - separate Q/K/V: q_proj/k_proj/v_proj + o_proj (Mllama/Llama Vision style)
    - HuggingFace ViT: query/key/value + output.dense

    RoPE is also inferred from the forward inputs. Models that pass
    position_embeddings=(cos, sin) use those tensors; models without visual RoPE
    get identity cos/sin tensors.
    """

    def __init__(
        self,
        original_attn: nn.Module,
        config: Any,
        layer_idx: int,
        return_tuple: bool = False,
        use_plugin_op: bool = True,
    ):
        super().__init__()
        self.original_attn = original_attn
        self.layer_idx = layer_idx
        self.return_tuple = return_tuple
        self.use_plugin_op = use_plugin_op

        self.projection_layout = self._detect_projection_layout(original_attn)
        self.output_proj = self._detect_output_projection(original_attn)
        self.num_heads = self._detect_num_heads(original_attn, config)
        self.head_dim = self._detect_head_dim(original_attn, config, self.num_heads)

    def _detect_projection_layout(self, original_attn: nn.Module) -> str:
        if hasattr(original_attn, "qkv"):
            return "fused_qkv"
        if all(hasattr(original_attn, name) for name in ("q_proj", "k_proj", "v_proj")):
            return "separate_qkv"
        if all(hasattr(original_attn, name) for name in ("query", "key", "value")):
            return "hf_vit_qkv"
        raise ValueError(
            "Unsupported ViT attention projection layout. Expected qkv, "
            "q_proj/k_proj/v_proj, or query/key/value projections."
        )

    def _detect_output_projection(self, original_attn: nn.Module) -> nn.Module:
        for name in ("proj", "o_proj", "out_proj", "out"):
            if hasattr(original_attn, name):
                return getattr(original_attn, name)
        if hasattr(original_attn, "output"):
            output = original_attn.output
            return output.dense if hasattr(output, "dense") else output
        raise ValueError(
            "Unsupported ViT attention output projection layout. Expected proj, "
            "o_proj, out_proj, out, or output(.dense)."
        )

    def _detect_num_heads(self, original_attn: nn.Module, config: Any) -> int:
        for source in (original_attn, config):
            for name in ("num_heads", "attention_heads", "num_attention_heads"):
                value = getattr(source, name, None)
                if value is not None:
                    return int(value)
        raise ValueError("Could not infer number of attention heads for ViT plugin.")

    def _detect_head_dim(
        self, original_attn: nn.Module, config: Any, num_heads: int
    ) -> int:
        for source in (original_attn, config):
            value = getattr(source, "head_dim", None)
            if value is not None:
                return int(value)

        hidden_size = None
        for source in (config, original_attn):
            for name in ("hidden_size", "embed_dim", "dim"):
                value = getattr(source, name, None)
                if value is not None:
                    hidden_size = int(value)
                    break
            if hidden_size is not None:
                break
        if hidden_size is None:
            raise ValueError("Could not infer hidden size for ViT plugin head_dim.")
        return hidden_size // num_heads

    def _project_qkv(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.projection_layout == "fused_qkv":
            return self.original_attn.qkv(hidden_states)
        if self.projection_layout == "separate_qkv":
            q = self.original_attn.q_proj(hidden_states)
            k = self.original_attn.k_proj(hidden_states)
            v = self.original_attn.v_proj(hidden_states)
            return torch.cat([q, k, v], dim=-1)

        q = self.original_attn.query(hidden_states)
        k = self.original_attn.key(hidden_states)
        v = self.original_attn.value(hidden_states)
        return torch.cat([q, k, v], dim=-1)

    def _get_rope_tensors(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = hidden_states.shape[-2]
        if position_embeddings is None:
            cos = torch.ones(
                seq_len,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            sin = torch.zeros_like(cos)
            return cos, sin

        cos, sin = position_embeddings
        if not self.use_plugin_op:
            return cos.to(device=hidden_states.device), sin.to(device=hidden_states.device)
        return (
            cos.to(device=hidden_states.device, dtype=hidden_states.dtype),
            sin.to(device=hidden_states.device, dtype=hidden_states.dtype),
        )

    def _normalize_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.zeros(batch_size, seq_len, seq_len, dtype=dtype, device=device)

        if attention_mask.dim() == 1 and attention_mask.dtype == torch.int32:
            return attention_mask

        attention_mask = attention_mask.to(dtype=dtype)
        if attention_mask.dim() == 4:
            if attention_mask.shape[1] == 1:
                attention_mask = attention_mask[:, 0, :, :]
            else:
                attention_mask = attention_mask.reshape(
                    attention_mask.shape[0] * attention_mask.shape[1],
                    attention_mask.shape[2],
                    attention_mask.shape[3],
                )
        return attention_mask

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _torch_attention(
        self,
        qkv: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)
        if attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(v.dtype)
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        return attn_out

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        max_seq_len: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        squeeze_batch = False
        if hidden_states.dim() == 2:
            hidden_states = hidden_states.unsqueeze(0)
            squeeze_batch = True

        batch_size, seq_len, _ = hidden_states.shape
        qkv = self._project_qkv(hidden_states)
        cos, sin = self._get_rope_tensors(hidden_states, position_embeddings)
        attention_mask = self._normalize_attention_mask(
            attention_mask,
            batch_size,
            seq_len,
            hidden_states.dtype,
            hidden_states.device,
        )
        mask_type = 0
        if attention_mask.dim() == 1 and attention_mask.dtype == torch.int32:
            mask_type = 1
            if max_seq_len <= 0:
                max_seq_len = seq_len

        if self.use_plugin_op:
            attn_out = torch.ops.tensorrt_vit.attention.default(
                qkv,
                cos,
                sin,
                attention_mask,
                self.num_heads,
                self.head_dim,
                1,
                mask_type,
                max_seq_len,
            )
        else:
            if mask_type == 1:
                raise ValueError("PyTorch reference attention requires a dense mask.")
            attn_out = self._torch_attention(qkv, cos, sin, attention_mask)
        output = self.output_proj(attn_out)
        output = output.squeeze(0) if squeeze_batch else output
        if self.return_tuple:
            return output, None
        return output


# -----------------------------------------------------------------------------
# Model Wrappers
# -----------------------------------------------------------------------------

VIT_INPUT_CONTRACT_NATIVE = "native"
VIT_INPUT_CONTRACT_WINDOWED_ROPE = "windowed_rope"
VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO = "tiled_aspect_ratio"

def _require_tensor(value: Optional[torch.Tensor], name: str) -> torch.Tensor:
    if value is None:
        raise ValueError(f"ViT plugin forward requires {name}.")
    return value


def _get_windowed_rope_visual_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "visual"):
        return model.visual
    if hasattr(model, "patch_embed") and hasattr(model, "blocks"):
        return model
    raise ValueError("Cannot find a windowed-RoPE visual backbone.")


def _get_windowed_rope_blocks(visual_model: nn.Module) -> nn.ModuleList:
    if hasattr(visual_model, "blocks"):
        return visual_model.blocks
    raise ValueError("Cannot find windowed-RoPE visual blocks.")


def _forward_windowed_rope_vision(
    model: nn.Module,
    pixel_values: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    window_attention_mask: Optional[torch.Tensor] = None,
    cu_window_seqlens: Optional[torch.Tensor] = None,
    window_index: Optional[torch.Tensor] = None,
    reverse_window_index: Optional[torch.Tensor] = None,
    max_window_seq_len: int = 0,
    **kwargs,
) -> torch.Tensor:
    visual = _get_windowed_rope_visual_model(model)
    rotary_pos_emb = _require_tensor(rotary_pos_emb, "rotary_pos_emb")
    attention_mask = _require_tensor(attention_mask, "attention_mask")
    window_attention_mask = _require_tensor(
        window_attention_mask, "window_attention_mask"
    )
    cu_window_seqlens = _require_tensor(cu_window_seqlens, "cu_window_seqlens")
    window_index = _require_tensor(window_index, "window_index")
    reverse_window_index = _require_tensor(reverse_window_index, "reverse_window_index")

    hidden_states = visual.patch_embed(pixel_values)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)

    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // visual.spatial_merge_unit,
        visual.spatial_merge_unit,
        -1,
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    blocks = _get_windowed_rope_blocks(visual)
    for layer_idx, block in enumerate(blocks):
        full_attention = layer_idx in visual.fullatt_block_indexes
        attention_mask_now = attention_mask if full_attention else window_attention_mask

        residual = hidden_states
        hidden_states = block.norm1(hidden_states)
        hidden_states = block.attn(
            hidden_states,
            attention_mask=attention_mask_now,
            position_embeddings=position_embeddings,
            max_seq_len=0,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = block.norm2(hidden_states)
        hidden_states = block.mlp(hidden_states)
        hidden_states = residual + hidden_states

    hidden_states = visual.merger(hidden_states)
    hidden_states = hidden_states[reverse_window_index, :]
    return hidden_states


def _forward_tiled_aspect_ratio_vision(
    model: nn.Module,
    pixel_values: torch.Tensor,
    aspect_ratio_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    aspect_ratio_ids = _require_tensor(aspect_ratio_ids, "aspect_ratio_ids")
    attention_mask = _require_tensor(attention_mask, "attention_mask")
    vision = model.vision_model if hasattr(model, "vision_model") else model
    batch_size, num_concurrent_media, num_tiles, num_channels, height, width = (
        pixel_values.shape
    )
    pixel_values = pixel_values.reshape(
        batch_size * num_concurrent_media * num_tiles,
        num_channels,
        height,
        width,
    )
    aspect_ratio_ids = aspect_ratio_ids.reshape(
        batch_size * num_concurrent_media, -1
    )

    target_dtype = vision.patch_embedding.weight.dtype
    target_device = vision.patch_embedding.weight.device
    patch_embeds = vision.patch_embedding(
        pixel_values.to(target_device, target_dtype)
    )
    hidden_state = patch_embeds.flatten(2).transpose(1, 2)

    _, num_patches, dim = hidden_state.shape
    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        -1,
        dim,
    )
    hidden_state = vision.pre_tile_positional_embedding(
        hidden_state,
        aspect_ratio_ids,
    )

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media * num_tiles,
        num_patches,
        dim,
    )
    hidden_state = vision.apply_class_embedding(hidden_state)
    num_patches += 1

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches,
        dim,
    )
    hidden_state = vision.gated_positional_embedding(
        hidden_state,
        aspect_ratio_ids,
    )
    hidden_state = vision.layernorm_pre(hidden_state)

    num_padding_patches = (8 - (hidden_state.shape[-2] % 8)) % 8
    hidden_state = torch.nn.functional.pad(
        hidden_state,
        (0, 0, 0, num_padding_patches),
        mode="constant",
        value=0,
    )
    slice_index = -num_padding_patches if num_padding_patches > 0 else None

    hidden_state = hidden_state.view(batch_size * num_concurrent_media, -1, dim)
    output = vision.transformer(
        hidden_state,
        attention_mask=attention_mask,
    )
    hidden_state = output.last_hidden_state
    hidden_state = vision.layernorm_post(hidden_state)

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches + num_padding_patches,
        dim,
    )
    hidden_state = vision.post_tile_positional_embedding(
        hidden_state,
        aspect_ratio_ids,
    )
    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles * (num_patches + num_padding_patches),
        dim,
    )
    global_output = vision.global_transformer(
        hidden_state,
        attention_mask=attention_mask,
    )
    hidden_state = global_output.last_hidden_state

    hidden_state = hidden_state.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches + num_padding_patches,
        dim,
    )
    hidden_state = hidden_state[:, :, :slice_index]
    hidden_state = hidden_state.reshape(
        batch_size,
        num_concurrent_media,
        num_tiles,
        num_patches,
        dim,
    )

    all_intermediate_hidden_states = [
        output.hidden_states[i] for i in vision.intermediate_layers_indices
    ]
    intermediate_hidden_states = torch.stack(
        all_intermediate_hidden_states,
        dim=-1,
    )
    intermediate_hidden_states = intermediate_hidden_states.reshape(
        batch_size * num_concurrent_media,
        num_tiles,
        num_patches + num_padding_patches,
        -1,
    )
    intermediate_hidden_states = intermediate_hidden_states[:, :, :slice_index]
    intermediate_hidden_states = intermediate_hidden_states.reshape(
        batch_size,
        num_concurrent_media,
        num_tiles,
        num_patches,
        -1,
    )

    return torch.cat([hidden_state, intermediate_hidden_states], dim=-1)


def _forward_native_vision(
    model: nn.Module,
    pixel_values: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    output = model(pixel_values)
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


class ViTPluginWrapper(nn.Module):
    """
    Generic wrapper for vision models with plugin attention.

    The caller chooses the tensor input contract and provides the corresponding
    tensors during export/runtime. The contract describes the vision tower's
    tensor interface, not a concrete model name.
    """

    def __init__(
        self,
        model: nn.Module,
        input_contract: str = VIT_INPUT_CONTRACT_NATIVE,
        max_window_seq_len: int = 0,
    ):
        super().__init__()
        self.model = model
        self.input_contract = input_contract
        self.max_window_seq_len = max_window_seq_len

    def forward(
        self,
        pixel_values: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        window_attention_mask: Optional[torch.Tensor] = None,
        cu_window_seqlens: Optional[torch.Tensor] = None,
        window_index: Optional[torch.Tensor] = None,
        reverse_window_index: Optional[torch.Tensor] = None,
        aspect_ratio_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.input_contract == VIT_INPUT_CONTRACT_WINDOWED_ROPE:
            return _forward_windowed_rope_vision(
                self.model,
                pixel_values,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                window_attention_mask=window_attention_mask,
                cu_window_seqlens=cu_window_seqlens,
                window_index=window_index,
                reverse_window_index=reverse_window_index,
                max_window_seq_len=self.max_window_seq_len,
            )
        if self.input_contract == VIT_INPUT_CONTRACT_TILED_ASPECT_RATIO:
            return _forward_tiled_aspect_ratio_vision(
                self.model,
                pixel_values,
                aspect_ratio_ids=aspect_ratio_ids,
                attention_mask=attention_mask,
            )
        if self.input_contract == VIT_INPUT_CONTRACT_NATIVE:
            return _forward_native_vision(self.model, pixel_values)
        raise ValueError(f"Unsupported ViT plugin input contract: {self.input_contract}")


# -----------------------------------------------------------------------------
# Model Modification Functions
# -----------------------------------------------------------------------------


def replace_vit_attention_with_plugin(
    model: nn.Module,
    config: Any,
    use_plugin_op: bool = True,
) -> nn.Module:
    """
    Replace all supported vision attention modules with plugin attention.

    This is the vision-side equivalent of the LLM helper: callers use one
    replacement entry point, and the function detects the model structure:
    - Qwen-VL visual blocks: ``blocks[*].attn``
    - Mllama/Llama Vision stacks: ``transformer/global_transformer.layers[*].self_attn``
    - HF ViT-style encoders: ``encoder.layer[*].attention.self``

    Args:
        model: The HuggingFace vision model or visual tower to modify.
        config: Model configuration.

    Returns:
        The modified model with plugin attention.
    """
    replacement_count = 0

    # Qwen2.5-VL visual tower: model.visual.blocks or visual.blocks.
    visual_model = model.visual if hasattr(model, "visual") else model
    if hasattr(visual_model, "blocks"):
        for i, block in enumerate(visual_model.blocks):
            if hasattr(block, "attn"):
                block.attn = ViTPluginAttention(
                    block.attn, config, i, use_plugin_op=use_plugin_op
                )
                replacement_count += 1
        if replacement_count:
            return model

    # Mllama is HuggingFace's architecture name for official Meta Llama 3.2
    # Vision models. Its self_attn returns (hidden_state, attn_weights), so
    # self_attn replacements ask the generic plugin wrapper to return a tuple.
    vision_model = model.vision_model if hasattr(model, "vision_model") else model
    layer_idx = 0
    for encoder_name in ("transformer", "global_transformer"):
        encoder = getattr(vision_model, encoder_name, None)
        if encoder is None or not hasattr(encoder, "layers"):
            continue

        for layer in encoder.layers:
            if hasattr(layer, "self_attn"):
                layer.self_attn = ViTPluginAttention(
                    layer.self_attn,
                    config,
                    layer_idx,
                    return_tuple=True,
                )
                layer_idx += 1
                replacement_count += 1

    if layer_idx:
        return model

    # HF SigLIP/SigLIP2-style tower: model.vision_model.encoder.layers or
    # model.encoder.layers. These attention modules return
    # (hidden_state, attn_weights), so the replacement returns a tuple.
    if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layers"):
        for i, layer in enumerate(vision_model.encoder.layers):
            if hasattr(layer, "self_attn"):
                layer.self_attn = ViTPluginAttention(
                    layer.self_attn,
                    config,
                    i,
                    return_tuple=True,
                    use_plugin_op=use_plugin_op,
                )
                replacement_count += 1

    if replacement_count:
        return model

    # HF ViT-style tower: model.vision_model.encoder.layer or model.encoder.layer.
    if hasattr(vision_model, "encoder") and hasattr(vision_model.encoder, "layer"):
        for i, layer in enumerate(vision_model.encoder.layer):
            if hasattr(layer, "attention"):
                layer.attention.self = ViTPluginAttention(
                    layer.attention.self, config, i, use_plugin_op=use_plugin_op
                )
                replacement_count += 1

    if replacement_count == 0:
        raise ValueError("Cannot find supported ViT attention modules")

    return model


def count_vit_plugin_attention_modules(model: nn.Module) -> int:
    """Count ViT attention modules replaced with the plugin wrapper."""
    return sum(1 for module in model.modules() if isinstance(module, ViTPluginAttention))

# -----------------------------------------------------------------------------
# Compilation
# -----------------------------------------------------------------------------

def compile_vit_plugin_model(
    model: nn.Module,
    example_inputs: Optional[Tuple[torch.Tensor, ...]],
    device: torch.device,
    example_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    dynamic_shapes: Optional[Dict[str, Any]] = None,
    debug: bool = False,
) -> Callable:
    """
    Compile a ViT/VLM visual wrapper with plugin attention.

    Model-specific wrappers own input preparation and forward signatures. This
    helper owns the shared torch.export -> Torch-TensorRT compile path.

    Args:
        model: The vision wrapper or model to export.
        example_inputs: Example tensor inputs matching ``model.forward``.
        example_kwargs: Optional named tensor inputs matching ``model.forward``.
        dynamic_shapes: Optional torch.export dynamic shape spec.
        device: Device for compilation.
        debug: Whether to enable debug logging.

    Returns:
        Compiled TensorRT model function.
    """
    if dynamic_shapes is None:
        dynamic_shapes = {}
    if example_inputs is None:
        example_inputs = ()
    if example_kwargs is None:
        example_kwargs = {}
    if dynamic_shapes:
        dynamic_shapes = {
            name: shape
            for name, shape in dynamic_shapes.items()
            if isinstance(example_kwargs.get(name), torch.Tensor)
        }

    ep = torch.export.export(
        model,
        args=example_inputs,
        kwargs=example_kwargs,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    compile_inputs = list(example_inputs) + [
        value for value in example_kwargs.values() if isinstance(value, torch.Tensor)
    ]
    with torch_tensorrt.dynamo.Debugger() if debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=compile_inputs,
            use_explicit_typing=True,
            use_fp32_acc=True,
            device=device,
            disable_tf32=True,
            min_block_size=1,
        )

    return trt_model


# -----------------------------------------------------------------------------
# Inference Utilities
# -----------------------------------------------------------------------------


def inference_vit_plugin(
    model_func: Callable,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    """
    Run inference on a compiled ViT plugin model.

    Args:
        model_func: The compiled TensorRT model function.
        pixel_values: Input images [batch, channels, height, width].

    Returns:
        Model output (logits or embeddings depending on model).
    """
    return model_func(pixel_values)


# Benchmark utilities

def measure_vit_latency(
    fn: Callable,
    num_warmup: int = 5,
    num_runs: int = 10,
) -> Tuple[float, float, float]:
    """
    Measure function latency with GPU synchronization.

    Args:
        fn: Function to benchmark.
        num_warmup: Number of warmup runs.
        num_runs: Number of timing runs.

    Returns:
        Tuple of (mean_latency_ms, std_latency_ms, median_latency_ms).
    """
    import statistics

    # Warmup
    for _ in range(num_warmup):
        fn()

    torch.cuda.synchronize()
    times = []

    for _ in range(num_runs):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        fn()
        end_event.record()
        torch.cuda.synchronize()

        times.append(start_event.elapsed_time(end_event))

    mean_time = statistics.mean(times)
    stdev_time = statistics.stdev(times) if len(times) > 1 else 0.0
    median_time = statistics.median(times)

    return mean_time, stdev_time, median_time


def measure_vit_memory(
    model: nn.Module,
    pixel_values: torch.Tensor,
) -> Tuple[float, float]:
    """
    Measure model memory usage.

    Args:
        model: The model.
        pixel_values: Sample input.

    Returns:
        Tuple of (peak_memory_mb, reserved_memory_mb).
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(pixel_values)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1e6
    reserved_memory = torch.cuda.memory_reserved() / 1e6

    return peak_memory, reserved_memory


# Importing this module registers the Torch-TensorRT converter for
# tensorrt_vit::attention, matching the LLM plugin_utils/plugin_converter split.
from plugin_converter_vit import convert_vit_attention  # noqa: F401,E402
