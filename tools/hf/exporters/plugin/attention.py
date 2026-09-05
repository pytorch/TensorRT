from enum import IntEnum
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ContextAttentionMaskType(IntEnum):
    """Context prefill mask type, mirroring the C++ ``ContextAttentionMaskType`` enum.

    The integer values must stay in sync with
    ``cpp/kernels/contextAttentionKernels/fmhaParams_v2.h`` since they are passed
    directly to the ``AttentionPlugin`` ``context_attention_mask_type`` field.
    """

    PADDING = 0  # Bidirectional full-prefix (attend to all valid tokens)
    CAUSAL = 1
    SLIDING_OR_CHUNKED_CAUSAL = 2
    CUSTOM_MASK = 3


class PluginAttention(nn.Module):
    """
    Model-agnostic Plugin Attention module that replaces standard attention.

    This module wraps the projection layers from the original attention module
    and uses ``trt.attention_plugin`` with separate Q/K/V tensors for the
    attention computation.

    Supports:
    - Qwen2.5, Llama: Standard attention
    - Qwen3: Attention with QK Normalization (q_norm, k_norm)
    """

    def __init__(
        self,
        original_attn: nn.Module,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        hidden_size: int,
        layer_idx: int,
        context_attention_mask_type: int = ContextAttentionMaskType.PADDING,
    ):
        """
        Initialize PluginAttention.

        Args:
            original_attn: The original attention module to wrap.
            num_attention_heads: Number of query attention heads.
            num_key_value_heads: Number of key/value attention heads.
            head_dim: Per-head dimension.
            hidden_size: Model hidden size.
            layer_idx: Index of this layer in the model.
            context_attention_mask_type: Context prefill mask type
                (``ContextAttentionMaskType`` enum value).
        """
        super().__init__()
        self.q_proj = original_attn.q_proj
        self.k_proj = original_attn.k_proj
        self.v_proj = original_attn.v_proj
        self.o_proj = original_attn.o_proj

        # Qwen3 has QK Normalization
        self.q_norm = getattr(original_attn, "q_norm", None)
        self.k_norm = getattr(original_attn, "k_norm", None)

        self.num_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.attn_hidden_size = self.num_heads * self.head_dim
        self.hidden_size = int(hidden_size)
        self.layer_idx = layer_idx
        self.context_attention_mask_type = int(context_attention_mask_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        ctx_len: Optional[torch.Tensor] = None,
        kvcache_start_index: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass using the plugin attention.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            rope_rotary_cos_sin: External RoPE cache of shape
                [rope_batch, max_seq_len, rotary_dim] (float32). For standard
                RoPE, rope_batch is typically 1 and the plugin broadcasts over
                batch. Layout: cos in [:, :, :rotary_dim // 2], sin in
                [:, :, rotary_dim // 2:]. Supplied at export as a graph input;
                at runtime filled by LLMEngineRunner (not computed here).
            attention_mask: Unused (plugin handles masking internally).
            position_ids: Unused; RoPE lookup uses rope_rotary_cos_sin and ctx_len.
            past_key_value: KV cache tensor of shape [batch, 2, num_kv_heads, capacity, head_dim].
            ctx_len: Context length tensor for each batch item.
            kvcache_start_index: External KV cache start indices. Empty tensor
                ``[0]`` for fresh prefill; ``[batch]`` for decode/chunked prefill.

        Returns:
            Tuple of (output tensor, updated KV cache).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Ensure rope embeddings are FP32
        assert (
            rope_rotary_cos_sin.dtype == torch.float32
        ), "rope_rotary_cos_sin must be FP32"

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

        if ctx_len is None:
            ctx_len = torch.tensor(
                [seq_len], dtype=torch.int32, device=hidden_states.device
            ).expand(batch_size)

        if past_key_value is None:
            raise ValueError("past_key_value (KV cache tensor) must be provided")

        if kvcache_start_index is None:
            raise ValueError("kvcache_start_index must be provided")

        dtype = q.dtype
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        attn_out, updated_kv = torch.ops.trt.attention_plugin.default(
            q,
            k,
            v,
            past_key_value,
            ctx_len,
            rope_rotary_cos_sin,
            kvcache_start_index,
            self.num_heads,
            self.num_key_value_heads,
            False,
            self.head_dim,
            False,
            -1,
            self.context_attention_mask_type,
        )

        # Use attn_hidden_size for reshape (may differ from hidden_size in Qwen3)
        attn_out = attn_out.reshape(batch_size, seq_len, self.attn_hidden_size).to(
            dtype
        )
        output = self.o_proj(attn_out)
        return output, updated_kv


class ViTPluginAttention(nn.Module):
    def __init__(
        self,
        attn,
        *,
        batch_size: int,
        seq_len: int,
        name: str,
        allow_attention_mask: bool = False,
    ):
        super().__init__()
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj
        self.num_heads = int(attn.num_heads)
        self.head_dim = int(attn.head_dim)
        self.name = name
        self.allow_attention_mask = bool(allow_attention_mask)

        device = self.q_proj.weight.device

        cu_seqlens = torch.arange(
            0,
            (int(batch_size) + 1) * int(seq_len),
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )
        max_seqlen_carrier = torch.zeros(
            int(seq_len),
            device=device,
            dtype=torch.int32,
        )

        self.register_buffer("cu_seqlens", cu_seqlens, persistent=False)
        self.register_buffer("max_seqlen_carrier", max_seqlen_carrier, persistent=False)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        if attention_mask is not None and not self.allow_attention_mask:
            raise RuntimeError(
                f"{self.name} ViT plugin path expects no vision attention_mask"
            )

        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = (
            q.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            .to(torch.float16)
            .contiguous()
        )
        k = (
            k.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            .to(torch.float16)
            .contiguous()
        )
        v = (
            v.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            .to(torch.float16)
            .contiguous()
        )

        attn_output = torch.ops.trt.vit_attention_plugin.default(
            q,
            k,
            v,
            self.cu_seqlens,
            self.max_seqlen_carrier,
            self.num_heads,
            self.head_dim,
        )

        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = attn_output.to(dtype=self.out_proj.weight.dtype)
        attn_output = self.out_proj(attn_output)
        return attn_output, None


class MolmoViTPluginAttention(nn.Module):
    """TRT ViT attention wrapper for MolmoAct2 vision self-attention.

    Molmo vision uses ``wq/wk/wv/wo`` and returns a tensor directly, unlike the
    SigLIP ``q_proj/k_proj/v_proj/out_proj`` modules wrapped by ViTPluginAttention.
    """

    def __init__(self, attn, *, batch_size: int, seq_len: int, name: str):
        super().__init__()
        self.wq = attn.wq
        self.wk = attn.wk
        self.wv = attn.wv
        self.wo = attn.wo
        self.residual_dropout = attn.residual_dropout

        self.num_heads = int(attn.num_heads)
        self.num_key_value_heads = int(attn.num_key_value_heads)
        self.num_key_value_groups = int(attn.num_key_value_groups)
        self.head_dim = int(attn.head_dim)
        self.name = name

        device = self.wq.weight.device
        self.register_buffer(
            "cu_seqlens",
            torch.arange(
                0,
                (int(batch_size) + 1) * int(seq_len),
                int(seq_len),
                device=device,
                dtype=torch.int32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "max_seqlen_carrier",
            torch.zeros(int(seq_len), device=device, dtype=torch.int32),
            persistent=False,
        )

    def forward(
        self,
        inputs_q: torch.Tensor,
        inputs_kv: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_kv is not None:
            raise RuntimeError(
                f"{self.name} Molmo ViT plugin path only supports self-attention"
            )
        if attn_mask is not None:
            raise RuntimeError(
                f"{self.name} Molmo ViT plugin path expects no attn_mask"
            )

        batch_size, seq_len, _ = inputs_q.shape
        q = self.wq(inputs_q).reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        k = self.wk(inputs_q).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )
        v = self.wv(inputs_q).reshape(
            batch_size, seq_len, self.num_key_value_heads, self.head_dim
        )

        if self.num_heads != self.num_key_value_heads:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)

        q = (
            q.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            .to(torch.float16)
            .contiguous()
        )
        k = (
            k.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            .to(torch.float16)
            .contiguous()
        )
        v = (
            v.reshape(batch_size * seq_len, self.num_heads, self.head_dim)
            .to(torch.float16)
            .contiguous()
        )

        attn_output = torch.ops.trt.vit_attention_plugin.default(
            q,
            k,
            v,
            self.cu_seqlens,
            self.max_seqlen_carrier,
            self.num_heads,
            self.head_dim,
        )

        attn_output = attn_output.reshape(
            batch_size, seq_len, self.num_heads * self.head_dim
        )
        attn_output = attn_output.to(dtype=self.wo.weight.dtype)
        attn_output = self.wo(attn_output)
        return self.residual_dropout(attn_output)


class MolmoPluginAttention(nn.Module):
    """Plugin wrapper for MolmoAct2 fused attention (att_proj + attn_out).

    Exposes the same runtime ABI as PluginAttention:
      (hidden_states, rope_rotary_cos_sin, past_key_value, ctx_len, kvcache_start_index)
      -> (attn_output, updated_kv)
    """

    def __init__(
        self,
        original_attn: nn.Module,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        hidden_size: int,
        layer_idx: int,
        context_attention_mask_type: int = ContextAttentionMaskType.PADDING,
    ):
        super().__init__()
        self.att_proj = original_attn.att_proj
        self.attn_out = original_attn.attn_out
        self.q_norm = getattr(original_attn, "q_norm", None)
        self.k_norm = getattr(original_attn, "k_norm", None)
        self.qk_norm_type = getattr(original_attn, "qk_norm_type", None)

        self.num_heads = int(num_attention_heads)
        self.num_kv_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.attn_hidden_size = self.num_heads * self.head_dim
        self.hidden_size = int(hidden_size)
        self.layer_idx = int(layer_idx)
        self.context_attention_mask_type = int(context_attention_mask_type)

        self.q_dim = self.num_heads * self.head_dim
        self.k_dim = self.num_kv_heads * self.head_dim
        self.v_dim = self.num_kv_heads * self.head_dim

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_rotary_cos_sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: torch.Tensor | None = None,
        ctx_len: torch.Tensor | None = None,
        kvcache_start_index: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        del attention_mask, position_ids, kwargs
        assert rope_rotary_cos_sin.dtype == torch.float32

        qkv = self.att_proj(hidden_states)
        q, k, v = qkv.split([self.q_dim, self.k_dim, self.v_dim], dim=-1)

        # Match MolmoAct2Attention norm ordering.
        if (
            self.q_norm is not None
            and self.k_norm is not None
            and self.qk_norm_type != "qwen3"
        ):
            q = self.q_norm(q)
            k = self.k_norm(k)
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        else:
            q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        if (
            self.q_norm is not None
            and self.k_norm is not None
            and self.qk_norm_type == "qwen3"
        ):
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.reshape(batch_size, seq_len, -1)
        k = k.reshape(batch_size, seq_len, -1)
        v = v.reshape(batch_size, seq_len, -1)

        if ctx_len is None:
            ctx_len = torch.full(
                (batch_size,), seq_len, device=hidden_states.device, dtype=torch.int32
            )
        if past_key_value is None:
            raise ValueError("past_key_value must be provided")
        if kvcache_start_index is None:
            raise ValueError("kvcache_start_index must be provided")

        dtype = q.dtype
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

        attn_out, updated_kv = torch.ops.trt.attention_plugin.default(
            q,
            k,
            v,
            past_key_value,
            ctx_len,
            rope_rotary_cos_sin,
            kvcache_start_index,
            self.num_heads,
            self.num_kv_heads,
            False,  # is_cross_attention
            self.head_dim,
            False,  # do_rotary_embedding (RoPE supplied externally)
            -1,
            self.context_attention_mask_type,
        )

        attn_out = attn_out.reshape(batch_size, seq_len, self.attn_hidden_size).to(
            dtype
        )
        return self.attn_out(attn_out), updated_kv


"""
Below are reference attention implementations to compare math
with plugin stack and kernel implementations
"""


class SiglipReferenceAttention(nn.Module):
    """
    Hand-written re-implementation of SigLIP multi-head attention.

    Reuses the original module's projection weights and shape params, but computes
    QK^T -> softmax -> (attn @ V) explicitly instead of dispatching through HF's
    attention_interface. Used to validate that our understanding of the math
    matches the stock eager path bit-for-bit (up to fp accumulation).

    Drop-in replacement for SiglipAttention: same forward signature and same
    (attn_output, attn_weights) return contract that SiglipEncoderLayer expects.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        # Reuse the trained projection layers directly (no copy).
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.out_proj = attn.out_proj

        self.num_heads = int(attn.num_heads)
        self.head_dim = int(attn.head_dim)
        self.embed_dim = self.num_heads * self.head_dim
        # SiglipAttention.scale == head_dim ** -0.5
        self.scale = float(getattr(attn, "scale", self.head_dim**-0.5))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # hidden_states: [B, S, embed_dim]
        input_shape = hidden_states.shape[:-1]  # (B, S)
        hidden_shape = (*input_shape, self.num_heads, self.head_dim)

        # Project and split heads: [B, S, E] -> [B, num_heads, S, head_dim]
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # Scores: [B, num_heads, S, S]
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Match HF eager: softmax in fp32 then cast back to input dtype.
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(q.dtype)

        # Weighted sum of values: [B, num_heads, S, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Merge heads back: [B, S, E]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class PluginNemotronAttention(PluginAttention):
    """``PluginAttention`` constructed from a Nemotron HF config."""

    def __init__(self, original_attn: nn.Module, config, layer_idx: int):
        head_dim = int(getattr(config, "head_dim", 0) or original_attn.head_dim)
        super().__init__(
            original_attn,
            num_attention_heads=int(config.num_attention_heads),
            num_key_value_heads=int(config.num_key_value_heads),
            head_dim=head_dim,
            hidden_size=int(config.hidden_size),
            layer_idx=layer_idx,
            context_attention_mask_type=ContextAttentionMaskType.CAUSAL,
        )
