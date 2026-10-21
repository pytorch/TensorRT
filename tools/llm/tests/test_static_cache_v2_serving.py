# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Static KV cache v2 as a serving engine: chunked prefill, dynamic batch, decoupled cache length.

A VLA/LLM server prefills a long prompt in chunks at batch 1, then decodes N samples at batch N
from the same cache. That needs, on top of the basic GQA support (``test_static_cache_gqa.py``):

* a causal mask aligned to the *end* of the cache for chunks that are neither the whole sequence
  nor a single token (``L != S`` and ``L != 1``),
* a KV cache length (``static_cache_v2.KV_CACHE_LEN``) independent of the largest per-call length,
* a dynamic batch dimension on the cache,
* query heads whose count TensorRT does not know statically (RoPE-style broadcast), and
* ``torch_tensorrt.save``/``torch.export.load`` of an engine with scalar ``start_idx``/``end_idx``.

Run with: ``pytest tools/llm/tests/test_static_cache_v2_serving.py``
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

HIDDEN, HEADS, KV_HEADS, HEAD_DIM, LAYERS = 256, 8, 2, 32, 2
CACHE_LEN, MAX_CHUNK, MAX_BATCH = 48, 16, 3


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.o = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)

    def forward(self, x, cos):
        B, L, _ = x.shape
        # ``view(..., -1, D)`` + a broadcast against the dynamic-shaped RoPE table is what makes
        # TensorRT lose the static head count of the query (as in the HF Qwen3 attention).
        q = self.q(x).view(B, L, -1, HEAD_DIM).transpose(1, 2) * cos
        k = self.k(x).view(B, L, -1, HEAD_DIM).transpose(1, 2) * cos
        v = self.v(x).view(B, L, -1, HEAD_DIM).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        return x + self.o(out.transpose(1, 2).reshape(B, L, HEADS * HEAD_DIM))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(LAYERS))

    def forward(self, x, position_ids):  # position_ids [B, L]
        cos = torch.cos(
            position_ids[:, None, :, None].to(x.dtype) * 0.1
        )  # [B, 1, L, 1]
        for layer in self.layers:
            x = layer(x, cos)
        return x


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    from torchtrt_ext import register_sdpa

    register_sdpa.enable_sdpa_converter("default", None)
    import static_cache_v2

    static_cache_v2.KV_CACHE_LEN = CACHE_LEN
    torch.manual_seed(0)
    model = Decoder().cuda().half().eval()
    x = torch.randn(2, MAX_CHUNK, HIDDEN, device="cuda", dtype=torch.half)
    pos = (
        torch.arange(MAX_CHUNK, device="cuda", dtype=torch.int32)
        .expand(2, -1)
        .contiguous()
    )
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    seq = torch.export.Dim("seq", min=1, max=MAX_CHUNK)
    try:
        with torch.no_grad():
            ep = torch.export.export(
                model,
                (x, pos),
                dynamic_shapes=({0: batch, 1: seq}, {0: batch, 1: seq}),
                strict=False,
            )
            trt_mod = torch_tensorrt.dynamo.compile(
                ep,
                inputs=[
                    torch_tensorrt.Input(
                        min_shape=(1, 1, HIDDEN),
                        opt_shape=(MAX_BATCH, 1, HIDDEN),
                        max_shape=(MAX_BATCH, MAX_CHUNK, HIDDEN),
                        dtype=torch.half,
                    ),
                    torch_tensorrt.Input(
                        min_shape=(1, 1),
                        opt_shape=(MAX_BATCH, 1),
                        max_shape=(MAX_BATCH, MAX_CHUNK),
                        dtype=torch.int32,
                    ),
                ],
                enabled_precisions={torch.half},
                use_explicit_typing=False,
                min_block_size=1,
                disable_tf32=True,
            )
    finally:
        static_cache_v2.KV_CACHE_LEN = None
    path = str(tmp_path_factory.mktemp("engine") / "decoder.ep")
    torch_tensorrt.save(trt_mod, path, retrace=False)
    return model, trt_mod, torch.export.load(path).module()


def _zero_caches(batch):
    return [
        torch.zeros(
            batch, KV_HEADS, CACHE_LEN, HEAD_DIM, device="cuda", dtype=torch.half
        )
        for _ in range(2 * LAYERS)
    ]


def _step(engine, x, pos, caches, start, end):
    out = engine(x[:, start:end], pos[:, start:end], *caches, start, end)
    return out[0], list(out[1:])


@pytest.mark.parametrize("which", ["compiled", "reloaded"])
def test_chunked_prefill_then_batched_decode(compiled, which):
    model, trt_mod, reloaded = compiled
    engine = trt_mod if which == "compiled" else reloaded
    total, chunks = 21, [8, 8, 5]  # chunked prefill: the 2nd/3rd chunk have 1 < L < S
    torch.manual_seed(1)
    x = torch.randn(1, total + 2, HIDDEN, device="cuda", dtype=torch.half)
    pos = torch.arange(total + 2, device="cuda", dtype=torch.int32)[None]
    with torch.no_grad():
        ref = model(x, pos)

        caches, start, outs = _zero_caches(1), 0, []
        for n in chunks:
            out, caches = _step(engine, x, pos, caches, start, start + n)
            outs.append(out)
            start += n
        torch.testing.assert_close(
            torch.cat(outs, 1), ref[:, :total], atol=2e-2, rtol=2e-2
        )

        # decode at batch MAX_BATCH from the batch-1 cache
        caches = [c.expand(MAX_BATCH, -1, -1, -1).contiguous() for c in caches]
        for t in range(total, total + 2):
            xb, pb = x[:, t : t + 1].expand(MAX_BATCH, -1, -1), pos[
                :, t : t + 1
            ].expand(MAX_BATCH, -1)
            out = engine(xb, pb, *caches, t, t + 1)
            out, caches = out[0], list(out[1:])
            for b in range(MAX_BATCH):
                torch.testing.assert_close(out[b, 0], ref[0, t], atol=2e-2, rtol=2e-2)


def test_cache_length_is_independent_of_chunk_length(compiled):
    _, trt_mod, _ = compiled
    for node in trt_mod.graph.nodes:
        if node.op == "placeholder" and node.name.endswith(("_k_input", "_v_input")):
            assert node.meta["val"].shape[2] == CACHE_LEN
