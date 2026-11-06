import ctypes
import os
from typing import Any, List, Optional, Sequence, Tuple

import tensorrt as trt
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import (
    ContextAttentionMaskType,
    MolmoPluginAttention,
    MolmoViTPluginAttention,
    PluginAttention,
    PluginNemotronAttention,
    SiglipReferenceAttention,
    ViTPluginAttention,
)

_PLUGIN_CONFIG: dict[str, Any] = {}


def get_plugin_config() -> dict[str, Any]:
    return _PLUGIN_CONFIG.copy()


def set_plugin_config(
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    max_seq_len: int = 2048,
    max_batch_size: int = 4,
) -> None:
    """Store LM plugin metadata used by Alpamayo plugin compile helpers."""
    global _PLUGIN_CONFIG
    _PLUGIN_CONFIG = {
        "num_attention_heads": int(num_attention_heads),
        "num_key_value_heads": int(num_key_value_heads),
        "head_dim": int(head_dim),
        "max_seq_len": int(max_seq_len),
        "max_batch_size": int(max_batch_size),
    }


def set_plugin_config_from_model(model_config: Any, max_seq_len: int = 2048) -> None:
    """Populate plugin config from a HuggingFace-style model config."""
    if getattr(model_config, "head_dim", None) is not None:
        head_dim = int(model_config.head_dim)
    else:
        head_dim = int(model_config.hidden_size) // int(
            model_config.num_attention_heads
        )
    set_plugin_config(
        num_attention_heads=int(model_config.num_attention_heads),
        num_key_value_heads=int(model_config.num_key_value_heads),
        head_dim=head_dim,
        max_seq_len=int(max_seq_len),
    )


def create_kv_caches(
    config: Any,
    max_seq_len: int,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> List[torch.Tensor]:
    """Allocate empty per-layer KV caches ``[B, 2, n_kv, capacity, head_dim]``."""
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(config.num_key_value_heads)
    if getattr(config, "head_dim", None) is not None:
        head_dim = int(config.head_dim)
    else:
        head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    return [
        torch.zeros(
            int(batch_size),
            2,
            num_kv_heads,
            int(max_seq_len),
            head_dim,
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]


def _has_torch_op(namespace: str, name: str) -> bool:
    return hasattr(torch.ops, namespace) and hasattr(
        getattr(torch.ops, namespace), name
    )


def _apply_plugin_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply AttentionPlugin cos/sin layout: first half cos, second half sin."""
    half = head_size // 2
    seq_len = q.shape[1]
    cos = rope_rotary_cos_sin[:, :seq_len, :half]
    sin = rope_rotary_cos_sin[:, :seq_len, half:]
    cos = torch.cat([cos, cos], dim=-1).unsqueeze(2)
    sin = torch.cat([sin, sin], dim=-1).unsqueeze(2)

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def _attention_plugin_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    context_attention_mask_type: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eager SDPA stand-in for ``AttentionPlugin`` (prefill, linear KV)."""
    del kvcache_start_index
    orig_dtype = q.dtype
    batch, seq_len, _ = q.shape
    q = q.view(batch, seq_len, num_q_heads, head_size)
    k = k.view(batch, seq_len, num_kv_heads, head_size)
    v = v.view(batch, seq_len, num_kv_heads, head_size)
    q, k = _apply_plugin_rope(q, k, rope_rotary_cos_sin.float(), head_size)
    q = q.to(dtype=orig_dtype)
    k = k.to(dtype=orig_dtype)
    v = v.to(dtype=orig_dtype)

    present = past_key_value.clone()
    present[:, 0, :, :seq_len, :] = k.permute(0, 2, 1, 3).to(dtype=present.dtype)
    present[:, 1, :, :seq_len, :] = v.permute(0, 2, 1, 3).to(dtype=present.dtype)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    if num_q_heads != num_kv_heads:
        repeats = num_q_heads // num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    is_causal = int(context_attention_mask_type) == int(ContextAttentionMaskType.CAUSAL)
    attn = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal and seq_len > 1)
    attn = attn.permute(0, 2, 1, 3).contiguous()
    if context_lengths is not None:
        lengths = context_lengths.to(device=attn.device, dtype=torch.long)
        token = torch.arange(seq_len, device=attn.device)
        valid = token.unsqueeze(0) < lengths.unsqueeze(1)
        attn = attn.masked_fill(~valid[:, :, None, None], 0)
    return attn.to(dtype=q.dtype), present


# These registrations follow the same custom-op pattern as TensorRT-Edge-LLM's
# ONNX exporter: define a torch.ops.trt operator and register a fake
# implementation for Dynamo shape propagation. The pipelines diverge only
# after capture: Edge-LLM translates the op into a custom ONNX node, while
# Torch-TensorRT lowers it directly through a Dynamo converter.
def _register_attention_plugin_op() -> None:
    """Register the LLM attention op using Edge-LLM's ONNX-export pattern."""
    # TODO: Reuse TensorRT-Edge-LLM's canonical attention custom-op registration
    # once this wrapper and its converter use the same argument order and
    # present-KV output shape. Preserve context_attention_mask_type, which is
    # required by VLA models but is not currently exposed by Edge-LLM's schema.
    if _has_torch_op("trt", "attention_plugin"):
        return

    @torch.library.custom_op("trt::attention_plugin", mutates_args=())
    def attention_plugin(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        enable_tree_attention: bool,
        head_size: int,
        enable_fp8_kv_cache: bool,
        sliding_window_size: int = -1,
        context_attention_mask_type: int = ContextAttentionMaskType.CAUSAL,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        qkv_scales: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del enable_tree_attention, enable_fp8_kv_cache, sliding_window_size
        del attention_mask, position_ids, qkv_scales
        return _attention_plugin_eager(
            q,
            k,
            v,
            past_key_value,
            context_lengths,
            rope_rotary_cos_sin,
            kvcache_start_index,
            int(num_q_heads),
            int(num_kv_heads),
            int(head_size),
            int(context_attention_mask_type),
        )

    @attention_plugin.register_fake
    def _attention_plugin_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        past_key_value: torch.Tensor,
        context_lengths: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        kvcache_start_index: torch.Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        enable_tree_attention: bool,
        head_size: int,
        enable_fp8_kv_cache: bool,
        sliding_window_size: int = -1,
        context_attention_mask_type: int = ContextAttentionMaskType.CAUSAL,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        qkv_scales: Optional[Sequence[float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        del k, v, context_lengths, rope_rotary_cos_sin, kvcache_start_index
        del num_kv_heads, enable_tree_attention, enable_fp8_kv_cache
        del (
            sliding_window_size,
            context_attention_mask_type,
            attention_mask,
            position_ids,
            qkv_scales,
        )
        batch_size, seq_len, _ = q.shape
        attn_output = torch.empty(
            batch_size,
            seq_len,
            num_q_heads,
            head_size,
            dtype=q.dtype,
            device=q.device,
        )
        return attn_output, torch.empty_like(past_key_value)


def _register_vit_attention_plugin_op() -> None:
    """Register the same ViT attention op and fake used by Edge-LLM ONNX export."""
    if _has_torch_op("trt", "vit_attention_plugin"):
        return

    @torch.library.custom_op("trt::vit_attention_plugin", mutates_args=())
    def vit_attention_plugin(
        query_states: torch.Tensor,  # [T, num_heads, head_size]
        key_states: torch.Tensor,  # [T, num_heads, head_size]
        value_states: torch.Tensor,  # [T, num_heads, head_size]
        cu_seqlens: torch.Tensor,  # [batch+1] int32
        max_seqlen_carrier: torch.Tensor,  # [] or [1] int32 (scalar)
        num_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        """ViT ragged self-attention.

        In eager mode, implements varlen SDPA using cu_seqlens to process each
        sequence segment independently.  During dynamo/ONNX tracing the
        register_fake shape propagation is used and this body is not executed.

        Unlike AttentionPlugin, ViT attention has no KV cache and takes ragged
        input with cu_seqlens instead of context_lengths.  RoPE is applied before
        this call.
        """
        import torch.nn.functional as F

        out = torch.empty_like(query_states)
        seqlens = cu_seqlens.tolist()
        for i in range(len(seqlens) - 1):
            start, end = int(seqlens[i]), int(seqlens[i + 1])
            if start >= end:
                continue
            # q/k/v: [S, H, D] -> [1, H, S, D] for SDPA
            q = query_states[start:end].permute(1, 0, 2).unsqueeze(0)
            k = key_states[start:end].permute(1, 0, 2).unsqueeze(0)
            v = value_states[start:end].permute(1, 0, 2).unsqueeze(0)
            attn = F.scaled_dot_product_attention(q, k, v)  # [1, H, S, D]
            out[start:end] = attn.squeeze(0).permute(1, 0, 2)
        return out

    @vit_attention_plugin.register_fake
    def _(
        query_states,
        key_states,
        value_states,
        cu_seqlens,
        max_seqlen_carrier,
        num_heads,
        head_size,
    ):
        return torch.empty_like(query_states)


def get_trt_plugin_creator(
    plugin_name: str,
    version: str = "1",
    namespace: str = "",
):
    """Return a TensorRT plugin creator, preferring the TRT 10.14+ V3 API."""
    registry = trt.get_plugin_registry()
    if hasattr(registry, "get_creator"):
        creator = registry.get_creator(plugin_name, version, namespace)
        if creator is not None:
            return creator
    if hasattr(registry, "get_plugin_creator"):
        return registry.get_plugin_creator(plugin_name, version, namespace)
    return None


def load_plugin():
    plugin_so = (
        os.environ.get("EDGE_LLM_PLUGIN_SO")
        or os.environ.get("EDGELLM_TRT_PLUGIN_SO")
        or os.environ.get("EDGELLM_PLUGIN_PATH")
    )
    if not plugin_so:
        raise RuntimeError(
            "Set EDGE_LLM_PLUGIN_SO (or EDGELLM_PLUGIN_PATH) to libNvInfer_edgellm_plugin.so"
        )

    ctypes.CDLL(plugin_so)
    trt.init_libnvinfer_plugins(None, "")
    return plugin_so


def load_plugins_for_trt():
    from .mamba import register_mamba_plugin_ops
    from .moe import register_moe_plugin_ops

    _register_attention_plugin_op()
    _register_vit_attention_plugin_op()
    register_mamba_plugin_ops()
    register_moe_plugin_ops()
    load_plugin()

    from . import plugin_converter as _plugin_converter  # noqa: F401,E402


def restore_attention(patched):
    for item in patched:
        if len(item) == 2:
            layer, original_attn = item
            layer.self_attn = original_attn
        else:
            module, attr_name, original_attn = item
            setattr(module, attr_name, original_attn)


def patch_vision_attention(
    vision_model,
    *,
    batch_size: int,
    seq_len: int,
    name: str,
    allow_attention_mask: bool = False,
):
    patched = []

    for layer in vision_model.encoder.layers:
        patched.append((layer, layer.self_attn))
        layer.self_attn = ViTPluginAttention(
            layer.self_attn,
            batch_size=batch_size,
            seq_len=seq_len,
            name=name,
            allow_attention_mask=allow_attention_mask,
        ).eval()

    print(f"patched {name} attention modules: {len(patched)}")
    return patched


def patch_molmo_vision_attention(
    vision_backbone,
    *,
    batch_size: int,
    seq_len: int,
    name: str = "molmo-vision",
):
    patched = []
    resblocks = vision_backbone.image_vit.transformer.resblocks
    for i, block in enumerate(resblocks):
        patched.append((block, "attention", block.attention))
        block.attention = MolmoViTPluginAttention(
            block.attention,
            batch_size=batch_size,
            seq_len=seq_len,
            name=f"{name}.image_vit.block{i}",
        ).eval()

    print(f"patched {name} image_vit attention modules: {len(patched)}")
    return patched


def patch_molmo_language_attention(
    transformer: nn.Module,
    *,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    context_attention_mask_type: int = ContextAttentionMaskType.PADDING,
    name: str = "molmo-language",
):
    patched = []
    for i, block in enumerate(transformer.blocks):
        patched.append((block, block.self_attn))
        block.self_attn = MolmoPluginAttention(
            block.self_attn,
            num_attention_heads=int(num_attention_heads),
            num_key_value_heads=int(num_key_value_heads),
            head_dim=int(head_dim),
            hidden_size=int(hidden_size),
            layer_idx=i,
            context_attention_mask_type=context_attention_mask_type,
        ).eval()
    print(f"patched {name} attention modules: {len(patched)}")
    return patched


def patch_vision_attention_reference(vision_model):
    """
    Swap every SigLIP encoder-layer self_attn for SiglipReferenceAttention.
    `vision_model` must be the INNER transformer (SiglipVisionTransformer),
    i.e. eagle_model.vision_model.vision_model, matching patch_vision_attention.
    Returns a list of (layer, original_attn) so it can be undone.
    """
    patched = []
    for layer in vision_model.encoder.layers:
        patched.append((layer, layer.self_attn))
        layer.self_attn = SiglipReferenceAttention(layer.self_attn).eval()
    print(f"patched SigLIP reference attention modules: {len(patched)}")
    return patched


def patch_language_attention(
    language_model,
    *,
    hidden_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    context_attention_mask_type: int = ContextAttentionMaskType.PADDING,
    name: str = "language",
):
    patched = []

    for i, layer in enumerate(language_model.layers):
        patched.append((layer, layer.self_attn))
        layer.self_attn = PluginAttention(
            layer.self_attn,
            num_attention_heads=int(num_attention_heads),
            num_key_value_heads=int(num_key_value_heads),
            head_dim=int(head_dim),
            hidden_size=int(hidden_size),
            layer_idx=i,
            context_attention_mask_type=context_attention_mask_type,
        ).eval()

    print(f"patched {name} attention modules: {len(patched)}")
    return patched


def patch_nemotron_mixers(model, config):
    """Replace Nemotron hybrid mixers with plugin wrappers. MLP stays native."""
    from .mamba import PluginNemotronMamba
    from .moe import PluginNemotronMoE

    wrappers = {
        "NemotronHAttention": lambda mixer, idx: PluginNemotronAttention(
            mixer, config, idx
        ),
        "NemotronHMamba2Mixer": lambda mixer, idx: PluginNemotronMamba(mixer),
        "NemotronHMoE": lambda mixer, idx: PluginNemotronMoE(mixer, config),
    }
    layers = (getattr(model, "backbone", None) or model.model).layers
    patched = []
    for i, block in enumerate(layers):
        wrap = wrappers.get(type(block.mixer).__name__)
        if wrap is None:
            continue
        original = block.mixer
        block.mixer = wrap(original, i).eval()
        patched.append((block, "mixer", original))
    print(f"patched nemotron mixers: {len(patched)}")
    return patched


@torch.no_grad()
def infer_smolvlm_seq_len(vision_model, image):
    patch_size = vision_model.patch_size
    patch_attention_mask = torch.ones(
        image.shape[0],
        image.shape[2] // patch_size,
        image.shape[3] // patch_size,
        dtype=torch.bool,
        device=image.device,
    )
    hidden_states = vision_model.embeddings(
        pixel_values=image,
        patch_attention_mask=patch_attention_mask,
    )
    return int(hidden_states.shape[0]), int(hidden_states.shape[1])
