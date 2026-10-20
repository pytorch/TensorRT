# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Static KV cache v2 with grouped-query attention (GQA).

HF models call ``scaled_dot_product_attention(..., enable_gqa=True)`` with ``num_key_value_heads``
K/V heads (or expand K/V with ``repeat_kv``). The static cache must hold the *un-expanded* K/V and the
SDPA converter must broadcast them, otherwise the cache is ``num_heads / num_kv_heads`` times larger
than necessary (or the engine fails to build).

Run with: ``pytest tools/llm/tests/test_static_cache_gqa.py``
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

torch_tensorrt = pytest.importorskip("torch_tensorrt")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HIDDEN, HEADS, HEAD_DIM, LAYERS = 256, 8, 32, 2
# 2 KV heads -> GQA (groups of 4); 8 -> plain multi-head attention (regression for groups == 1)
KV_HEADS_VARIANTS = [2, 8]
MAX_SEQ = 64


class GQALayer(nn.Module):
    def __init__(self, kv_heads):
        super().__init__()
        self.kv_heads = kv_heads
        self.q = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k = nn.Linear(HIDDEN, kv_heads * HEAD_DIM, bias=False)
        self.v = nn.Linear(HIDDEN, kv_heads * HEAD_DIM, bias=False)
        self.o = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q(x).view(B, L, HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k(x).view(B, L, self.kv_heads, HEAD_DIM).transpose(1, 2)
        v = self.v(x).view(B, L, self.kv_heads, HEAD_DIM).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.kv_heads != HEADS
        )
        return x + self.o(out.transpose(1, 2).reshape(B, L, HEADS * HEAD_DIM))


class GQADecoder(nn.Module):
    def __init__(self, kv_heads):
        super().__init__()
        self.layers = nn.ModuleList(GQALayer(kv_heads) for _ in range(LAYERS))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.fixture(scope="module", params=KV_HEADS_VARIANTS, ids=lambda h: f"kv{h}")
def compiled(request):
    kv_heads = request.param
    # Both passes register themselves globally on import / call.
    from torchtrt_ext import register_sdpa

    register_sdpa.enable_sdpa_converter("default", None)
    import static_cache_v2  # noqa: F401

    torch.manual_seed(0)
    model = GQADecoder(kv_heads).cuda().half().eval()
    x = torch.randn(1, 8, HIDDEN, device="cuda", dtype=torch.half)
    seq = torch.export.Dim("seq", min=1, max=MAX_SEQ)
    with torch.no_grad():
        ep = torch.export.export(model, (x,), dynamic_shapes=({1: seq},), strict=False)
        trt_mod = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[x],
            enabled_precisions={torch.half},
            use_explicit_typing=False,
            min_block_size=1,
            disable_tf32=True,
            use_python_runtime=True,  # as in run_llm.py
        )
    return model, trt_mod, kv_heads


def _cache_inputs(trt_mod):
    from utils import get_zeroed_static_cache_inputs

    return get_zeroed_static_cache_inputs(trt_mod, has_position_ids=False)


def test_cache_holds_unexpanded_kv_heads(compiled):
    _, trt_mod, kv_heads = compiled
    caches = _cache_inputs(trt_mod)
    assert len(caches) == 2 * LAYERS
    for c in caches:
        assert c.shape[1] == kv_heads, (
            f"static cache stores {c.shape[1]} heads, expected the {kv_heads} KV heads "
            f"(not the {HEADS} query heads)"
        )


def test_prefill_and_decode_match_full_forward(compiled):
    model, trt_mod, _ = compiled
    torch.manual_seed(1)
    total, prefill = 12, 7
    x = torch.randn(1, total, HIDDEN, device="cuda", dtype=torch.half)
    with torch.no_grad():
        ref = model(x)

    caches = _cache_inputs(trt_mod)
    outs, start = [], 0
    with torch.no_grad():
        for n in [prefill] + [1] * (total - prefill):
            end = start + n
            res = trt_mod(x[:, start:end], *caches, start, end)
            out, caches = res[0], res[1:]
            outs.append(out)
            start = end
    got = torch.cat(outs, dim=1)
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)
