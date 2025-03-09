# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import flashinfer
import torch
import torch.nn as nn
import triton
import triton.language as tl
from torch import Tensor
from torch.nn import functional as F

# from ...utils.logger import logger


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Rms norm kernel."""
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w * out).to(x.dtype)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


def rms_norm_func(hidden_states: Tensor, weight: Tensor, eps: float = 1e-5):
    """Rms norm."""
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    out = torch.empty_like(hidden_states)

    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        out,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return out


@torch.library.custom_op("llama::rms_norm", mutates_args=())
def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return rms_norm_func(x, w, eps)


@rms_norm.register_fake
def rms_norm_fake(x, w, eps):
    return torch.empty_like(x)


@torch.library.custom_op("llama::apply_rotary_emb", mutates_args=())
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    freqs_cis = freqs_cis[
        None, : x.shape[1], None
    ]  #             --> [1, s,   1, h_d//2, 2]
    xshaped = x.float().unflatten(
        -1, (-1, 2)
    )  # [b, s, n_h, h_d] --> [b, s, n_h, h_d//2, 2]
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )  # [b, s, n_h, h_d//2, 2]

    return x_out2.flatten(-2).type_as(x)  # [b, s, n_h, h_d//2, 2] --> [b, s, n_h, h_d]


@apply_rotary_emb.register_fake
def apply_rotary_emb_fake(x, freqs_cis):
    return torch.empty_like(x)


@torch.library.custom_op("llama::repeat_kv", mutates_args=())
def repeat_kv(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep).

    This version avoid any memcopy.
    """
    # q, k, v is [b,s,n,d]
    n_heads = q.shape[2]
    bs, slen, n_kv_heads, head_dim = kv.shape
    n_rep = n_heads // n_kv_heads
    return (
        kv[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        .contiguous()
    )


@repeat_kv.register_fake
def repeat_kv_fake(q, kv):
    return torch.empty_like(q)


@dataclass
class ModelArgs:
    vocab_size: int = 32000
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_key_value_heads: int = 32
    rope_theta: float = 10000
    norm_eps: float = 1e-5
    max_position_embeddings: int = 2048
    torch_dtype: torch.dtype = torch.float16

    def __post_init__(self):
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = getattr(torch, self.torch_dtype)
        assert isinstance(
            self.torch_dtype, torch.dtype
        ), f"Invalid dtype: {self.torch_dtype}"

    @classmethod
    def from_hf_config(cls, path: str):
        keys_to_check = [
            "num_hidden_layers",
            "num_attention_heads",
            "hidden_size",
            "vocab_size",
            "num_key_value_heads",
            "intermediate_size",
        ]

        with open(path) as json_map:
            hf_config = json.load(json_map)

        kwargs = {k: hf_config[k] for k in keys_to_check if k in hf_config}

        # logger.info(str(kwargs))

        return cls(**kwargs)

    def save_to_file(self, path: str):
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "ModelArgs":
        """Load config from pre-defined string or from file path."""
        # load from pre-defined config or path
        if name_or_path in TRANSFORMER_CONFIGS:
            conf = TRANSFORMER_CONFIGS[name_or_path]
        elif os.path.isfile(name_or_path) and name_or_path.endswith(".json"):
            with open(name_or_path) as json_map:
                conf = json.load(json_map)
        else:
            raise FileNotFoundError(f"Model {name_or_path} not found")

        # check for customization options
        k_to_update = kwargs.keys() & {f.name for f in fields(cls)}
        conf.update({k: kwargs[k] for k in k_to_update})

        # initialize class
        model_args = cls(**conf)

        # logger.info(str(model_args))
        return model_args


TRANSFORMER_CONFIGS = {
    "llama2-7B": dict(
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
        intermediate_size=11008,
    ),
    "llama2-13B": dict(
        num_hidden_layers=40,
        num_attention_heads=40,
        hidden_size=5120,
        intermediate_size=13824,
    ),
    "llama2-30B": dict(
        num_hidden_layers=60,
        num_attention_heads=52,
        hidden_size=6656,
        intermediate_size=17920,
    ),
    "llama2-34B": dict(  # CodeLlama-34B-Python-hf
        num_hidden_layers=48,
        num_attention_heads=64,
        hidden_size=8192,
        vocab_size=32000,
        num_key_value_heads=8,
        intermediate_size=22016,
        rope_theta=1000000,
    ),
    "llama2-70B": dict(
        num_hidden_layers=80,
        num_attention_heads=64,
        hidden_size=8192,
        num_key_value_heads=8,
        intermediate_size=28672,
    ),
    "llama3-8B": dict(
        num_hidden_layers=32,
        num_attention_heads=32,
        hidden_size=4096,
        vocab_size=128256,
        num_key_value_heads=8,
        intermediate_size=14336,
        rope_theta=500000,
        torch_dtype="bfloat16",
    ),
    "llama-3.1-70b": dict(
        num_hidden_layers=80,
        num_attention_heads=64,
        hidden_size=8192,
        num_key_value_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
        rope_base=500000,
        torch_dtype="bfloat16",
    ),
    "llama-3.1-405b": dict(
        num_hidden_layers=126,
        num_attention_heads=128,
        hidden_size=8192,
        num_key_value_heads=8,
        intermediate_size=53248,
        vocab_size=128256,
        rope_base=500000,
        torch_dtype="bfloat16",
    ),
}


class LlamaModel(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(
                config.max_position_embeddings * 2,
                config.hidden_size // config.num_attention_heads,
                config.rope_theta,
            ),  # hidden_size / n_head = d_head
        )

    @classmethod
    def from_config(cls, config: ModelArgs, **kwargs) -> "LLamaTransformer":
        return cls(config)

    @torch.no_grad()
    def forward(self, idx) -> Tuple[Tensor]:
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x, self.freqs_cis)
        x = self.norm(x)
        return x


class LLamaTransformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_config(cls, config: ModelArgs, **kwargs) -> "LLamaTransformer":
        return cls(config)

    @torch.no_grad()
    def forward(self, idx) -> Tuple[Tensor]:
        x = self.model(idx)
        logits = self.lm_head(x)
        return (logits,)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.self_attn = GQA(config)
        self.mlp = FeedForward(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.input_layernorm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x, freqs_cis) -> Tensor:
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis)

        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class GQA(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.is_gqa = config.num_key_value_heads < config.num_attention_heads
        assert self.hidden_size == self.num_attention_heads * self.head_dim

        # key, query, value, out projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: Tensor, freqs_cis: Optional[Tensor] = None) -> Tensor:
        b, s, _ = x.shape
        q = self.q_proj(x).view(b, s, self.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(b, s, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(b, s, self.num_key_value_heads, self.head_dim)

        if freqs_cis is not None:
            # q shape is [b,s,n,d]
            q = torch.ops.llama.apply_rotary_emb(q, freqs_cis)
            k = torch.ops.llama.apply_rotary_emb(k, freqs_cis)

        if self.is_gqa:
            k = torch.ops.llama.repeat_kv(q, k)
            v = torch.ops.llama.repeat_kv(q, v)

        q, k, v = map(
            lambda x: x.transpose(1, 2).contiguous(), (q, k, v)
        )  # [b,n,s,h_d]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [b,n,s,h_d]
        y = y.transpose(1, 2).contiguous().view(b, s, self.hidden_size)  # [b,s,n*h_d]

        return self.o_proj(y)


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def silu(self, x):
        """Custom Silu that doesn't insert a cast to float32."""
        y = F.sigmoid(x)
        return y * x

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

        # return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return self._norm(x)  # torch.ops.llama.rms_norm(x, self.weight, self.eps) #


def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000) -> Tensor:
    freqs = 1.0 / (
        base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem)
    )
    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)
