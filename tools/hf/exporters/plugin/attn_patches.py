"""HF-style class ``setattr`` patches that swap attention ``forward`` for Edge plugins.

Same contract as ``transformers.exporters.utils.register_patch``: one factory per
backend, listed against every attention class that shares that layout. Patches are
installed only while ``apply_patches`` is active (or left installed on dryrun so
``execute_engine`` still hits the plugin).

Language dispatch: Edge prefill calls ``self_attn(..., rope_rotary_cos_sin=...)``.
The PI05 action expert is often the same class (GemmaAttention / PiGemmaModel)
but uses HF ``past_key_values``. If ``rope_rotary_cos_sin`` is absent, the
original forward runs.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
from typing import Any, Callable

import torch
import torch.nn as nn

from .attention import ContextAttentionMaskType

_PATCHES: dict[str, list[tuple[str, Callable]]] = {}
_LANGUAGE_MASK_TYPE = int(ContextAttentionMaskType.PADDING)

VISION_BACKEND = "edge_vision"
LANGUAGE_BACKEND = "edge_language"


def set_language_mask_type(mask_type: int) -> None:
    """Prefill mask for ``trt::attention_plugin`` (PADDING vs CAUSAL)."""
    global _LANGUAGE_MASK_TYPE
    _LANGUAGE_MASK_TYPE = int(mask_type)


def language_mask_type() -> int:
    return _LANGUAGE_MASK_TYPE


@contextlib.contextmanager
def patch_attribute(obj: Any, attribute: str, factory: Callable):
    original = getattr(obj, attribute)
    setattr(obj, attribute, factory(original))
    try:
        yield
    finally:
        setattr(obj, attribute, original)


@contextlib.contextmanager
def apply_patches(backend: str | None):
    """Resolve dotted paths and install those class attributes for the block.

    Paths are resolved here so the policy's modeling modules
    are already loaded and we do not import every HF family up front.
    """
    if not backend:
        yield
        return
    with contextlib.ExitStack() as stack:
        for path, factory in _PATCHES.get(backend, []):
            obj_path, _, attribute = path.rpartition(".")
            obj = _resolve_dotted_path(obj_path)
            if obj is None:
                continue
            stack.enter_context(patch_attribute(obj, attribute, factory))
        yield


def register_patch(backend: str, *paths: str):
    """Record ``factory(original)`` for each dotted ``Class.attribute`` path."""

    def decorator(fn: Callable) -> Callable:
        for path in paths:
            _PATCHES.setdefault(backend, []).append((path, fn))
        return fn

    return decorator


def _resolve_dotted_path(path: str) -> Any | None:
    parts = path.split(".")
    try:
        obj: Any = importlib.import_module(parts[0])
        for part in parts[1:]:
            try:
                obj = importlib.import_module(f"{obj.__name__}.{part}")
            except (ImportError, AttributeError):
                obj = getattr(obj, part)
        return obj
    except Exception:
        # Missing family, or a modeling import that fails for unrelated reasons
        # (e.g. torchaudio CUDA mismatch). Skip that class path.
        return None


def _returns_tuple(original: Callable) -> bool:
    try:
        src = inspect.getsource(original)
    except (OSError, TypeError):
        return True
    return (
        "return attn_output, attn_weights" in src
        or "return attn_output, attn_weight" in src
        or "return attn_output, None" in src
        or "return output, attn_weights" in src
    )


def _out_proj(module: nn.Module) -> nn.Module:
    proj = getattr(module, "out_proj", None) or getattr(
        module, "projection_layer", None
    )
    if proj is None:
        raise AttributeError(
            f"{type(module).__name__} has no out_proj/projection_layer"
        )
    return proj


def _language_dims(module: nn.Module) -> tuple[int, int, int]:
    cfg = getattr(module, "config", None)
    num_heads = int(
        getattr(module, "num_heads", None)
        or getattr(module, "num_attention_heads", None)
        or cfg.num_attention_heads
    )
    num_kv = int(
        getattr(module, "num_key_value_heads", None)
        or getattr(cfg, "num_key_value_heads", num_heads)
    )
    head_dim = int(
        getattr(module, "head_dim", None)
        or (cfg.hidden_size // cfg.num_attention_heads)
    )
    return num_heads, num_kv, head_dim


@register_patch(
    VISION_BACKEND,
    "transformers.models.internvl.modeling_internvl.InternVLVisionAttention.forward",
)
def _patch_vision_attention(original: Callable) -> Callable:
    returns_tuple = _returns_tuple(original)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        del kwargs
        if attention_mask is not None:
            raise RuntimeError(
                f"{type(self).__name__} Edge vision plugin expects no attention_mask"
            )

        batch_size, seq_len, _ = hidden_states.shape
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_norm = getattr(self, "q_norm", None)
        k_norm = getattr(self, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q)
        if k_norm is not None:
            k = k_norm(k)

        num_heads = int(self.num_heads)
        head_dim = int(self.head_dim)
        q = (
            q.reshape(batch_size * seq_len, num_heads, head_dim)
            .to(torch.float16)
            .contiguous()
        )
        k = (
            k.reshape(batch_size * seq_len, num_heads, head_dim)
            .to(torch.float16)
            .contiguous()
        )
        v = (
            v.reshape(batch_size * seq_len, num_heads, head_dim)
            .to(torch.float16)
            .contiguous()
        )

        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seq_len,
            seq_len,
            device=q.device,
            dtype=torch.int32,
        )
        max_seqlen_carrier = torch.zeros(seq_len, device=q.device, dtype=torch.int32)

        attn_output = torch.ops.trt.vit_attention_plugin.default(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen_carrier,
            num_heads,
            head_dim,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, num_heads * head_dim)
        out_proj = _out_proj(self)
        attn_output = attn_output.to(dtype=out_proj.weight.dtype)
        attn_output = out_proj(attn_output)
        drop = getattr(self, "projection_dropout", None)
        if drop is not None:
            attn_output = drop(attn_output)
        if returns_tuple:
            return attn_output, None
        return attn_output

    return forward


@register_patch(
    LANGUAGE_BACKEND,
    "transformers.models.gemma.modeling_gemma.GemmaAttention.forward",
    "transformers.models.gemma2.modeling_gemma2.Gemma2Attention.forward",
    "transformers.models.llama.modeling_llama.LlamaAttention.forward",
    "transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward",
    "transformers.models.qwen3.modeling_qwen3.Qwen3Attention.forward",
)
def _patch_language_attention(original: Callable) -> Callable:
    def forward(self, hidden_states, *args, **kwargs):
        rope_rotary_cos_sin = kwargs.get("rope_rotary_cos_sin")
        if rope_rotary_cos_sin is None:
            return original(self, hidden_states, *args, **kwargs)

        past_key_value = kwargs.get("past_key_value")
        ctx_len = kwargs.get("ctx_len")
        kvcache_start_index = kwargs.get("kvcache_start_index")
        if rope_rotary_cos_sin.dtype != torch.float32:
            raise ValueError("rope_rotary_cos_sin must be FP32")
        if past_key_value is None:
            raise ValueError("past_key_value (KV cache tensor) must be provided")
        if kvcache_start_index is None:
            raise ValueError("kvcache_start_index must be provided")

        batch_size, seq_len, _ = hidden_states.shape
        num_heads, num_kv, head_dim = _language_dims(self)
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q_norm = getattr(self, "q_norm", None)
        k_norm = getattr(self, "k_norm", None)
        if q_norm is not None:
            q = q_norm(q.view(batch_size, seq_len, num_heads, head_dim)).view(
                batch_size, seq_len, -1
            )
        if k_norm is not None:
            k = k_norm(k.view(batch_size, seq_len, num_kv, head_dim)).view(
                batch_size, seq_len, -1
            )

        dtype = q.dtype
        attn_out, updated_kv = torch.ops.trt.attention_plugin.default(
            q.to(torch.float16),
            k.to(torch.float16),
            v.to(torch.float16),
            past_key_value,
            ctx_len,
            rope_rotary_cos_sin,
            kvcache_start_index,
            num_heads,
            num_kv,
            False,
            head_dim,
            False,
            -1,
            language_mask_type(),
        )
        attn_hidden = num_heads * head_dim
        attn_out = attn_out.reshape(batch_size, seq_len, attn_hidden).to(dtype)
        o_proj = getattr(self, "o_proj", None) or getattr(self, "out_proj", None)
        if o_proj is None:
            raise AttributeError(f"{type(self).__name__} has no o_proj/out_proj")
        return o_proj(attn_out), updated_kv

    return forward


def registered_backends() -> dict[str, int]:
    """Test helper: how many class paths are registered per backend."""
    return {name: len(items) for name, items in _PATCHES.items()}
