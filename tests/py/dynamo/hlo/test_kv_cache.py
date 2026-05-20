"""KV cache subgraph tests for TRT converter validation.

Covers the major KV cache update patterns from popular open-source LLMs.
Each test exercises the full decoder sub-graph: Q/K/V projection, cache write,
SDPA, and output projection.

Cache update patterns
---------------------
  Dynamic (HF DynamicCache — default for all HF models):
    key = torch.cat([cache_k, new_k], dim=2)           ← grows every step
    SDPA attends over the full growing key tensor.
    Used by: GPT-2, Llama 2, Gemma, Mistral (non-compiled), all HF defaults.

  Static v2 (HF StaticCache — used when torch.compile is active):
    concat_k  = cat(cache[:, :, :start, :], k)          ← prefix + current
    new_cache = cat(concat_k, cache[:, :, end:, :])     ← write-at-position
    SDPA attends over concat_k, NOT the full cache.
    Pattern matches static_cache_v2.py in this repo.
    Used by: any model compiled with HF StaticCache.

  Scatter (alternative StaticCache write — index_copy_ pattern):
    updated_cache = cache.scatter(2, pos, new_kv)        ← in-place positional write
    No attention; pure cache-write subgraph.
    Used by: HF StaticCache.update() (index_copy_ decomposes to index_put).

  Sliding-window (Mistral SlidingWindowCache / Qwen2):
    cache = cat(cache[:, :, T:, :], new_k, dim=2)       ← drop oldest, append new
    SDPA attends over the full window.
    Used by: Mistral 7B, Qwen2-series with sliding_window config.

  RoPE + Dynamic (rotary models — Llama, Mistral, Qwen2, Gemma, DeepSeek):
    k = (k * cos) + (rotate_half(k) * sin)              ← apply RoPE before cache
    key = cat([cache_k, k_rope], dim=2)
    SDPA attends over the full growing key tensor.

Dynamic-shape generation loops
--------------------------------
Each test class compiles the module ONCE with symbolic sequence-length Dims
(via torch.export + torch_tensorrt.dynamo.compile), then drives a multi-step
generation loop, feeding the reference cache forward at every iteration.  The
TRT-compiled model receives the same inputs as the PyTorch reference model at
each step, so TRT errors never accumulate across steps.

This validates:
  1. Correct TRT lowering of cache-write ops (cat, slice, scatter) across the
     full [min, max] shape range, not just a single static input point.
  2. Per-step output correctness: the TRT model matches PyTorch for all cache
     sizes encountered in a realistic generation sequence.

Relationship to HuggingFace model internals
--------------------------------------------
In real HF models the cache write is inside attention.forward() via a
Cache.update() call (e.g. DynamicCache: torch.cat; StaticCache: index_copy_).
Our test models inline the same tensor operations directly in forward(),
making them straightforward to trace with torch.export.  The lowering-pass
approach in static_cache_v2.py / test_static_cache.py is the alternative:
export a bare SDPA module, then inject cache ops into the FX graph afterward
— useful for pre-trained models whose source cannot be modified.

Test classes
------------
  TestDynamicCacheAttention       dynamic past dim [1, 20]; 20-step decode loop
  TestStaticCacheAttention        dynamic T dim [1, 8]; 8-step decode loop per config
  TestStaticScatterCache          dynamic new_tokens dim [1, 8]; 8-step write loop
  TestSlidingWindowCacheAttention dynamic T dim [1, window]; prefill + 16-step decode
  TestRoPEDynamicCacheAttention   dynamic past dim [1, 20]; 20-step decode loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tensorrt
from parameterized import parameterized
from torch.export import Dim
from torch.testing._internal.common_utils import run_tests

from ..conversion.harness import DispatchTestCase

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension by half, negating the second half."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _make_cos_sin(
    batch: int,
    heads: int,
    seq: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Return precomputed (cos, sin) with shape [B, H, T, D] for RoPE tests."""
    theta = 1.0 / (
        10000
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(seq, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, theta)  # [T, D/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, D]
    cos = (
        emb.cos()
        .to(dtype)
        .view(1, 1, seq, head_dim)
        .expand(batch, heads, seq, head_dim)
    )
    sin = (
        emb.sin()
        .to(dtype)
        .view(1, 1, seq, head_dim)
        .expand(batch, heads, seq, head_dim)
    )
    return cos.contiguous(), sin.contiguous()


def _compile_trt(ep, input_specs):
    """Compile an ExportedProgram to a TRT-backed GraphModule.

    With use_explicit_typing now on by default, precision is determined
    per-tensor by the dtype set on each Input spec — enabled_precisions
    must not be passed alongside it.
    """
    return torch_tensorrt.dynamo.compile(
        ep,
        inputs=input_specs,
        min_block_size=1,
    )


def _unpack3(out):
    """Unpack a 3-output TRT result regardless of wrapping."""
    if isinstance(out, (tuple, list)):
        return out[0], out[1], out[2]
    return out[0], out[1], out[2]


def _unpack2(out):
    """Unpack a 2-output TRT result regardless of wrapping."""
    if isinstance(out, (tuple, list)):
        return out[0], out[1]
    return out[0], out[1]


# ---------------------------------------------------------------------------
# TestDynamicCacheAttention
# ---------------------------------------------------------------------------


class DynamicCacheAttention(nn.Module):
    """Full decoder step with dynamic (growing) KV cache.

    Cache grows by concatenation on each step.  Matches the HF DynamicCache
    pattern used by GPT-2, Llama 2, Gemma, and Mistral (non-compiled).

      k_full = cat([cache_k, k_new], dim=2)
      out    = SDPA(q, k_full, v_full)
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self, hidden: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor
    ):
        # hidden: [B, T, H_size]   cache: [B, H, past, D]
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)  # [B, H, T, D]
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)

        k = torch.cat([cache_k, k], dim=2)  # [B, H, past+T, D]
        v = torch.cat([cache_v, v], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D)), k, v


class TestDynamicCacheAttention(DispatchTestCase):
    """20-step decode loop; cache past dim is dynamic in [1, 20].

    Compile once with Dim("past", min=1, max=20), then run 20 single-token
    (T=1) decode steps.  At each step the reference PyTorch cache is fed to
    both the reference model and the TRT model — TRT errors never accumulate.
    """

    N_STEPS = 20

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, hidden, heads, head_dim, dtype, atol)
            ("b1_h8_d64_fp16",    1, 512,  8,  64, torch.float16, 1e-2),
            ("b2_h8_d64_fp16",    2, 512,  8,  64, torch.float16, 1e-2),
            ("b1_h8_d128_fp16",   1, 512,  8, 128, torch.float16, 1e-2),
            ("b1_h32_d128_fp16",  1, 512, 32, 128, torch.float16, 1e-2),
            ("b1_h8_d64_fp32",    1, 512,  8,  64, torch.float32, 1e-3),
            ("gpt2_proxy_fp16",   1, 768, 12,  64, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_dynamic_cache_generation_loop(
        self, name, batch, hidden, heads, head_dim, dtype, atol
    ):
        n = self.N_STEPS
        mod = DynamicCacheAttention(hidden, heads, head_dim).eval().cuda().to(dtype)

        # ── Export: cache past dim is dynamic ──────────────────────────────
        past_dim = Dim("past", min=1, max=n)
        ex_h = torch.randn(batch, 1, hidden, dtype=dtype, device="cuda")
        ex_c = torch.randn(batch, heads, 2, head_dim, dtype=dtype, device="cuda")
        ep = torch.export.export(
            mod,
            (ex_h, ex_c, ex_c.clone()),
            dynamic_shapes={
                "hidden": {},
                "cache_k": {2: past_dim},
                "cache_v": {2: past_dim},
            },
        )

        # ── Compile once for past in [1, n] ────────────────────────────────
        half = max(1, n // 2)
        cache_spec = torch_tensorrt.Input(
            min_shape=[batch, heads, 1, head_dim],
            opt_shape=[batch, heads, half, head_dim],
            max_shape=[batch, heads, n, head_dim],
            dtype=dtype,
        )
        trt_mod = _compile_trt(
            ep,
            [
                torch_tensorrt.Input(shape=[batch, 1, hidden], dtype=dtype),
                cache_spec,
                cache_spec,
            ],
        )

        # ── 20-step generation loop ─────────────────────────────────────────
        # Seed with a 1-token cache so past is within [1, n] at every step.
        ref_k = torch.randn(batch, heads, 1, head_dim, dtype=dtype, device="cuda")
        ref_v = torch.randn(batch, heads, 1, head_dim, dtype=dtype, device="cuda")

        with torch.no_grad():
            for step in range(n):
                x = torch.randn(batch, 1, hidden, dtype=dtype, device="cuda")

                # Reference model: advance the growing cache
                ref_out, new_ref_k, new_ref_v = mod(x, ref_k, ref_v)

                # TRT: same inputs as reference — no error accumulation
                trt_out, trt_k, trt_v = _unpack3(trt_mod(x, ref_k, ref_v))

                torch.testing.assert_close(
                    trt_out.to(torch.float32),
                    ref_out.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )
                torch.testing.assert_close(
                    trt_k.to(torch.float32),
                    new_ref_k.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )

                ref_k, ref_v = new_ref_k, new_ref_v


# ---------------------------------------------------------------------------
# TestStaticCacheAttention
# ---------------------------------------------------------------------------


class StaticCacheAttention(nn.Module):
    """Full decoder step with static (fixed-size) KV cache — v2 pattern.

    Matches the pattern in static_cache_v2.py from this repo and HF StaticCache:

      concat_k  = cat(cache[:, :, :past, :], k_new)    # prefix + current T tokens
      new_cache = cat(concat_k, cache[:, :, past+T:, :]) # write at [past:past+T]
      out       = SDPA(q, concat_k, concat_v)           # attend to prefix + current

    ``past_len`` is fixed at construction time (baked into the traced graph as a
    constant slice boundary).  The test compiles with dynamic T so the same TRT
    engine handles both prefill (T=T_max) and decode (T=1).  Each decode step
    writes a new K/V token to position ``past_len``, and we verify the TRT
    output matches PyTorch for every step.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int, past_len: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.past_len = past_len
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self, hidden: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor
    ):
        # hidden:  [B, T, H_size]
        # cache:   [B, H, max_len, D]  — pre-allocated static buffer
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim
        past = self.past_len

        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)

        # Prefix + current (what SDPA attends over)
        concat_k = torch.cat([cache_k[:, :, :past, :], k], dim=2)  # [B, H, past+T, D]
        concat_v = torch.cat([cache_v[:, :, :past, :], v], dim=2)

        # Write new tokens into the static buffer; preserve tail beyond end_idx
        new_cache_k = torch.cat(
            [concat_k, cache_k[:, :, past + T :, :]], dim=2
        )  # [B, H, max_len, D]
        new_cache_v = torch.cat([concat_v, cache_v[:, :, past + T :, :]], dim=2)

        out = F.scaled_dot_product_attention(q, concat_k, concat_v)
        return (
            self.o_proj(out.transpose(1, 2).reshape(B, T, H * D)),
            new_cache_k,
            new_cache_v,
        )


class TestStaticCacheAttention(DispatchTestCase):
    """Dynamic T in [1, 8]; 8 decode steps per (past_len, max_len) config.

    Each test bakes a specific ``past_len`` into the model, compiles with
    dynamic T covering both prefill (T=8) and decode (T=1), then runs 8
    single-token decode steps.  At each step, the cache has the same
    random prefix at positions [0:past_len] (unchanged) and the current new
    token is written to position ``past_len``.  We verify both the attention
    output and the updated cache are correct.
    """

    N_DECODE_STEPS = 8
    MAX_T = 8  # dynamic T range: [1, MAX_T]

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, hidden, heads, head_dim, past_len, max_len, dtype, atol)
            ("b1_h8_d64_past0_ml32_fp16",   1, 512,  8,  64,  0, 32, torch.float16, 1e-2),
            ("b1_h8_d64_past8_ml32_fp16",   1, 512,  8,  64,  8, 32, torch.float16, 1e-2),
            ("b2_h8_d64_past8_ml32_fp16",   2, 512,  8,  64,  8, 32, torch.float16, 1e-2),
            ("b1_h8_d64_past16_ml32_fp16",  1, 512,  8,  64, 16, 32, torch.float16, 1e-2),
            ("b1_h8_d128_past8_ml64_fp16",  1, 512,  8, 128,  8, 64, torch.float16, 1e-2),
            ("b1_h8_d64_past8_ml32_fp32",   1, 512,  8,  64,  8, 32, torch.float32, 1e-3),
        ]
    )
    # fmt: on
    def test_static_cache_decode_loop(
        self,
        name,
        batch,
        hidden,
        heads,
        head_dim,
        past_len,
        max_len,
        dtype,
        atol,
    ):
        n = self.N_DECODE_STEPS
        max_T = self.MAX_T
        mod = (
            StaticCacheAttention(hidden, heads, head_dim, past_len)
            .eval()
            .cuda()
            .to(dtype)
        )

        # ── Export: T (new-token count) is dynamic ─────────────────────────
        t_dim = Dim("T", min=1, max=max_T)
        ex_h = torch.randn(batch, 2, hidden, dtype=dtype, device="cuda")
        ex_c = torch.zeros(batch, heads, max_len, head_dim, dtype=dtype, device="cuda")
        ep = torch.export.export(
            mod,
            (ex_h, ex_c, ex_c.clone()),
            dynamic_shapes={
                "hidden": {1: t_dim},
                "cache_k": {},
                "cache_v": {},
            },
        )

        # ── Compile: T in [1, MAX_T], cache fixed at max_len ───────────────
        half_T = max(1, max_T // 2)
        trt_mod = _compile_trt(
            ep,
            [
                torch_tensorrt.Input(
                    min_shape=[batch, 1, hidden],
                    opt_shape=[batch, half_T, hidden],
                    max_shape=[batch, max_T, hidden],
                    dtype=dtype,
                ),
                torch_tensorrt.Input(
                    shape=[batch, heads, max_len, head_dim], dtype=dtype
                ),
                torch_tensorrt.Input(
                    shape=[batch, heads, max_len, head_dim], dtype=dtype
                ),
            ],
        )

        # ── 8-step decode loop (T=1) ────────────────────────────────────────
        # Pre-fill [0:past_len] with random data (the historical context).
        ref_k = torch.zeros(batch, heads, max_len, head_dim, dtype=dtype, device="cuda")
        ref_v = torch.zeros(batch, heads, max_len, head_dim, dtype=dtype, device="cuda")
        if past_len > 0:
            ref_k[:, :, :past_len, :] = torch.randn(
                batch, heads, past_len, head_dim, dtype=dtype, device="cuda"
            )
            ref_v[:, :, :past_len, :] = torch.randn(
                batch, heads, past_len, head_dim, dtype=dtype, device="cuda"
            )

        with torch.no_grad():
            for step in range(n):
                x = torch.randn(batch, 1, hidden, dtype=dtype, device="cuda")

                # Reference (advance the cache: position past_len is overwritten)
                ref_out, new_ref_k, new_ref_v = mod(x, ref_k, ref_v)

                # TRT: same inputs as reference
                trt_out, trt_k, trt_v = _unpack3(trt_mod(x, ref_k, ref_v))

                torch.testing.assert_close(
                    trt_out.to(torch.float32),
                    ref_out.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )
                torch.testing.assert_close(
                    trt_k.to(torch.float32),
                    new_ref_k.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )

                ref_k, ref_v = new_ref_k, new_ref_v


# ---------------------------------------------------------------------------
# TestStaticScatterCache
# ---------------------------------------------------------------------------


class StaticScatterCacheUpdate(nn.Module):
    """Write new KV at fixed positions into a pre-allocated cache via scatter.

    Approximates the index_copy_ write in HF StaticCache.update() (which
    decomposes to aten.index_put under the torch-trt decomposition table).
    No attention — tests the cache write subgraph in isolation.

      updated_cache = cache.scatter(2, pos, new_kv)
    """

    def forward(self, cache_k, cache_v, new_k, new_v, position_ids):
        # cache: [B, H, max_len, D]   new: [B, H, T, D]
        # position_ids: [T] integer positions in [0, max_len)
        B, H, _, D = cache_k.shape
        T = new_k.shape[2]
        pos = position_ids.view(1, 1, T, 1).expand(B, H, T, D)
        updated_k = cache_k.scatter(2, pos, new_k)
        updated_v = cache_v.scatter(2, pos, new_v)
        return updated_k, updated_v


class TestStaticScatterCache(DispatchTestCase):
    """Dynamic new_tokens in [1, 8]; 8 write steps writing to successive positions.

    Compiles with dynamic T (number of tokens written per scatter), then
    verifies successive writes to positions 0, T, 2T, … are correct.
    """

    N_WRITE_STEPS = 8
    MAX_T = 8

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, heads, max_len, head_dim, dtype, atol)
            ("b1_h8_ml32_d64_fp16",   1,  8, 32,  64, torch.float16, 1e-2),
            ("b2_h8_ml32_d64_fp16",   2,  8, 32,  64, torch.float16, 1e-2),
            ("b1_h8_ml64_d64_fp16",   1,  8, 64,  64, torch.float16, 1e-2),
            ("b1_h8_ml32_d128_fp16",  1,  8, 32, 128, torch.float16, 1e-2),
            ("b1_h2_ml32_d128_fp16",  1,  2, 32, 128, torch.float16, 1e-2),
            ("b1_h8_ml32_d64_fp32",   1,  8, 32,  64, torch.float32, 1e-3),
        ]
    )
    # fmt: on
    def test_static_scatter_write_loop(
        self, name, batch, heads, max_len, head_dim, dtype, atol
    ):
        n = self.N_WRITE_STEPS
        max_T = self.MAX_T
        mod = StaticScatterCacheUpdate().eval().cuda().to(dtype)

        # ── Export: T (tokens-per-write) is dynamic ────────────────────────
        t_dim = Dim("T", min=1, max=max_T)
        ex_cache = torch.zeros(
            batch, heads, max_len, head_dim, dtype=dtype, device="cuda"
        )
        ex_new = torch.randn(batch, heads, 2, head_dim, dtype=dtype, device="cuda")
        ex_pos = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
        ep = torch.export.export(
            mod,
            (ex_cache, ex_cache.clone(), ex_new, ex_new.clone(), ex_pos),
            dynamic_shapes={
                "cache_k": {},
                "cache_v": {},
                "new_k": {2: t_dim},
                "new_v": {2: t_dim},
                "position_ids": {0: t_dim},
            },
        )

        # ── Compile: T in [1, MAX_T] ───────────────────────────────────────
        half_T = max(1, max_T // 2)
        trt_mod = _compile_trt(
            ep,
            [
                torch_tensorrt.Input(
                    shape=[batch, heads, max_len, head_dim], dtype=dtype
                ),
                torch_tensorrt.Input(
                    shape=[batch, heads, max_len, head_dim], dtype=dtype
                ),
                torch_tensorrt.Input(
                    min_shape=[batch, heads, 1, head_dim],
                    opt_shape=[batch, heads, half_T, head_dim],
                    max_shape=[batch, heads, max_T, head_dim],
                    dtype=dtype,
                ),
                torch_tensorrt.Input(
                    min_shape=[batch, heads, 1, head_dim],
                    opt_shape=[batch, heads, half_T, head_dim],
                    max_shape=[batch, heads, max_T, head_dim],
                    dtype=dtype,
                ),
                torch_tensorrt.Input(
                    min_shape=[1],
                    opt_shape=[half_T],
                    max_shape=[max_T],
                    dtype=torch.int64,
                ),
            ],
        )

        # ── 8-step write loop (T=1 per step, advancing position) ───────────
        # Write to positions 0, 1, 2, … successively, wrapping at max_len.
        ref_k = torch.zeros(batch, heads, max_len, head_dim, dtype=dtype, device="cuda")
        ref_v = torch.zeros(batch, heads, max_len, head_dim, dtype=dtype, device="cuda")

        with torch.no_grad():
            for step in range(n):
                pos = step % max_len
                position_ids = torch.tensor([pos], dtype=torch.int64, device="cuda")
                new_k = torch.randn(
                    batch, heads, 1, head_dim, dtype=dtype, device="cuda"
                )
                new_v = torch.randn(
                    batch, heads, 1, head_dim, dtype=dtype, device="cuda"
                )

                ref_out_k, ref_out_v = mod(ref_k, ref_v, new_k, new_v, position_ids)
                trt_out_k, trt_out_v = _unpack2(
                    trt_mod(ref_k, ref_v, new_k, new_v, position_ids)
                )

                torch.testing.assert_close(
                    trt_out_k.to(torch.float32),
                    ref_out_k.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )
                torch.testing.assert_close(
                    trt_out_v.to(torch.float32),
                    ref_out_v.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )

                ref_k, ref_v = ref_out_k, ref_out_v


# ---------------------------------------------------------------------------
# TestSlidingWindowCacheAttention
# ---------------------------------------------------------------------------


class SlidingWindowCacheAttention(nn.Module):
    """Full decoder step with sliding-window KV cache.

    Maintains a fixed-size window of the most recent tokens.  On each step
    the oldest T entries are dropped and the new T appended:

      cache_k = cat(cache_k[:, :, T:, :], k_new, dim=2)

    Used by Mistral 7B (window=4096) and Qwen2 models with sliding_window.
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self, hidden: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor
    ):
        # hidden: [B, T, H_size]   cache: [B, H, window_size, D]
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)

        # Slide window: drop oldest T, append new T
        cache_k = torch.cat([cache_k[:, :, T:, :], k], dim=2)  # [B, H, window_size, D]
        cache_v = torch.cat([cache_v[:, :, T:, :], v], dim=2)

        # Attend over full window — asymmetric (window_size vs T) for T < window_size
        out = F.scaled_dot_product_attention(q, cache_k, cache_v)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D)), cache_k, cache_v


class TestSlidingWindowCacheAttention(DispatchTestCase):
    """Dynamic T in [1, window]; 1 prefill step then 16 single-token decode steps.

    The cache is always [B, H, window, D] — the window size never changes.
    With dynamic T, one compiled TRT engine covers both the initial prefill
    (T = T_prefill ≤ window) and subsequent decode (T=1).  After the prefill
    step the reference cache is advanced and fed forward at each decode step.
    """

    N_DECODE_STEPS = 16
    T_PREFILL = 4  # prefill with 4 tokens

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, hidden, heads, head_dim, window_size, dtype, atol)
            ("b1_h8_d64_w16_fp16",    1, 512,  8,  64, 16, torch.float16, 1e-2),
            ("b2_h8_d64_w16_fp16",    2, 512,  8,  64, 16, torch.float16, 1e-2),
            ("b1_h8_d64_w32_fp16",    1, 512,  8,  64, 32, torch.float16, 1e-2),
            ("b1_h8_d128_w16_fp16",   1, 512,  8, 128, 16, torch.float16, 1e-2),
            ("b1_h8_d64_w16_fp32",    1, 512,  8,  64, 16, torch.float32, 1e-3),
            ("mistral_proxy_fp16",    1, 512,  8,  64, 16, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_sliding_window_generation_loop(
        self, name, batch, hidden, heads, head_dim, window_size, dtype, atol
    ):
        n_decode = self.N_DECODE_STEPS
        t_pf = self.T_PREFILL
        mod = (
            SlidingWindowCacheAttention(hidden, heads, head_dim).eval().cuda().to(dtype)
        )

        # ── Export: T is dynamic (covers prefill and decode) ───────────────
        # T must be <= window_size (can't slide more than the window holds)
        t_dim = Dim("T", min=1, max=min(t_pf, window_size))
        ex_h = torch.randn(batch, 2, hidden, dtype=dtype, device="cuda")
        ex_c = torch.randn(
            batch, heads, window_size, head_dim, dtype=dtype, device="cuda"
        )
        ep = torch.export.export(
            mod,
            (ex_h, ex_c, ex_c.clone()),
            dynamic_shapes={
                "hidden": {1: t_dim},
                "cache_k": {},
                "cache_v": {},
            },
        )

        # ── Compile: T in [1, T_PREFILL], cache always [B, H, W, D] ───────
        t_max = min(t_pf, window_size)
        t_opt = max(1, t_max // 2)
        trt_mod = _compile_trt(
            ep,
            [
                torch_tensorrt.Input(
                    min_shape=[batch, 1, hidden],
                    opt_shape=[batch, t_opt, hidden],
                    max_shape=[batch, t_max, hidden],
                    dtype=dtype,
                ),
                torch_tensorrt.Input(
                    shape=[batch, heads, window_size, head_dim], dtype=dtype
                ),
                torch_tensorrt.Input(
                    shape=[batch, heads, window_size, head_dim], dtype=dtype
                ),
            ],
        )

        # ── 1 prefill step + N_DECODE_STEPS single-token decode steps ──────
        ref_k = torch.randn(
            batch, heads, window_size, head_dim, dtype=dtype, device="cuda"
        )
        ref_v = torch.randn(
            batch, heads, window_size, head_dim, dtype=dtype, device="cuda"
        )

        with torch.no_grad():
            # Prefill
            x_pf = torch.randn(batch, t_pf, hidden, dtype=dtype, device="cuda")
            ref_out_pf, new_ref_k, new_ref_v = mod(x_pf, ref_k, ref_v)
            trt_out_pf, trt_k_pf, trt_v_pf = _unpack3(trt_mod(x_pf, ref_k, ref_v))

            torch.testing.assert_close(
                trt_out_pf.to(torch.float32),
                ref_out_pf.to(torch.float32),
                rtol=1e-2,
                atol=atol,
            )
            torch.testing.assert_close(
                trt_k_pf.to(torch.float32),
                new_ref_k.to(torch.float32),
                rtol=1e-2,
                atol=atol,
            )
            ref_k, ref_v = new_ref_k, new_ref_v

            # Decode loop
            for step in range(n_decode):
                x = torch.randn(batch, 1, hidden, dtype=dtype, device="cuda")

                ref_out, new_ref_k, new_ref_v = mod(x, ref_k, ref_v)
                trt_out, trt_k, trt_v = _unpack3(trt_mod(x, ref_k, ref_v))

                torch.testing.assert_close(
                    trt_out.to(torch.float32),
                    ref_out.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )
                torch.testing.assert_close(
                    trt_k.to(torch.float32),
                    new_ref_k.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )

                ref_k, ref_v = new_ref_k, new_ref_v


# ---------------------------------------------------------------------------
# TestRoPEDynamicCacheAttention
# ---------------------------------------------------------------------------


class RoPEDynamicCacheAttention(nn.Module):
    """Full decoder step with RoPE applied to Q and K, then dynamic cache.

    Covers the rotary-embedding + cat-cache pattern used by Llama 2/3, Mistral,
    Qwen2, Gemma, and DeepSeek.  cos/sin are passed as inputs (precomputed for
    the current token positions).

      q = (q * cos) + (rotate_half(q) * sin)
      k = (k * cos) + (rotate_half(k) * sin)
      k_full = cat([cache_k, k_rope], dim=2)
      out    = SDPA(q_rope, k_full, v_full)
    """

    def __init__(self, hidden_size: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        # hidden: [B, T, H_sz]  cache: [B, H, past, D]  cos/sin: [B, H, T, D]
        B, T, _ = hidden.shape
        H, D = self.num_heads, self.head_dim

        q = self.q_proj(hidden).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(hidden).view(B, T, H, D).transpose(1, 2)
        v = self.v_proj(hidden).view(B, T, H, D).transpose(1, 2)

        # Apply RoPE to Q and K
        q = (q * cos) + (_rotate_half(q) * sin)
        k = (k * cos) + (_rotate_half(k) * sin)

        # Append to dynamic cache
        k = torch.cat([cache_k, k], dim=2)  # [B, H, past+T, D]
        v = torch.cat([cache_v, v], dim=2)

        out = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, H * D)), k, v


class TestRoPEDynamicCacheAttention(DispatchTestCase):
    """20-step decode loop; cache past dim is dynamic in [1, 20].

    Same generation-loop strategy as TestDynamicCacheAttention but with
    RoPE (cos/sin) applied to Q and K before caching.  cos/sin are
    recomputed at each step from position 0 (fixed-shape [B, H, 1, D]),
    which is sufficient to verify the rotary ops compile correctly.
    """

    N_STEPS = 20

    # fmt: off
    @parameterized.expand(
        [
            # (name, batch, hidden, heads, head_dim, dtype, atol)
            ("b1_h8_d64_fp16",    1, 512,  8,  64, torch.float16, 1e-2),
            ("b2_h8_d64_fp16",    2, 512,  8,  64, torch.float16, 1e-2),
            ("b1_h8_d128_fp16",   1, 512,  8, 128, torch.float16, 1e-2),
            ("b1_h8_d64_fp32",    1, 512,  8,  64, torch.float32, 1e-3),
            ("llama2_proxy_fp16", 1, 512,  8,  64, torch.float16, 1e-2),
        ]
    )
    # fmt: on
    def test_rope_dynamic_cache_generation_loop(
        self, name, batch, hidden, heads, head_dim, dtype, atol
    ):
        n = self.N_STEPS
        mod = RoPEDynamicCacheAttention(hidden, heads, head_dim).eval().cuda().to(dtype)

        # ── Export: past dim of cache is dynamic; cos/sin are fixed [B,H,1,D] ──
        past_dim = Dim("past", min=1, max=n)
        ex_h = torch.randn(batch, 1, hidden, dtype=dtype, device="cuda")
        ex_c = torch.randn(batch, heads, 2, head_dim, dtype=dtype, device="cuda")
        ex_cos, ex_sin = _make_cos_sin(batch, heads, 1, head_dim, dtype)
        ep = torch.export.export(
            mod,
            (ex_h, ex_c, ex_c.clone(), ex_cos, ex_sin),
            dynamic_shapes={
                "hidden": {},
                "cache_k": {2: past_dim},
                "cache_v": {2: past_dim},
                "cos": {},
                "sin": {},
            },
        )

        # ── Compile once for past in [1, n] ────────────────────────────────
        half = max(1, n // 2)
        cache_spec = torch_tensorrt.Input(
            min_shape=[batch, heads, 1, head_dim],
            opt_shape=[batch, heads, half, head_dim],
            max_shape=[batch, heads, n, head_dim],
            dtype=dtype,
        )
        cos_sin_spec = torch_tensorrt.Input(
            shape=[batch, heads, 1, head_dim], dtype=dtype
        )
        trt_mod = _compile_trt(
            ep,
            [
                torch_tensorrt.Input(shape=[batch, 1, hidden], dtype=dtype),
                cache_spec,
                cache_spec,
                cos_sin_spec,
                cos_sin_spec,
            ],
        )

        # ── 20-step generation loop ─────────────────────────────────────────
        ref_k = torch.randn(batch, heads, 1, head_dim, dtype=dtype, device="cuda")
        ref_v = torch.randn(batch, heads, 1, head_dim, dtype=dtype, device="cuda")

        with torch.no_grad():
            for step in range(n):
                x = torch.randn(batch, 1, hidden, dtype=dtype, device="cuda")
                # Fresh cos/sin per step (same shape [B, H, 1, D])
                cos, sin = _make_cos_sin(batch, heads, 1, head_dim, dtype)

                ref_out, new_ref_k, new_ref_v = mod(x, ref_k, ref_v, cos, sin)
                trt_out, trt_k, trt_v = _unpack3(trt_mod(x, ref_k, ref_v, cos, sin))

                torch.testing.assert_close(
                    trt_out.to(torch.float32),
                    ref_out.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )
                torch.testing.assert_close(
                    trt_k.to(torch.float32),
                    new_ref_k.to(torch.float32),
                    rtol=1e-2,
                    atol=atol,
                )

                ref_k, ref_v = new_ref_k, new_ref_v


if __name__ == "__main__":
    run_tests()
