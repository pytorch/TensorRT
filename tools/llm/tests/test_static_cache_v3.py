# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""static_cache_v3: a shape-static decode engine matches the eager model.

Run with ``python -m pytest tools/llm/tests/test_static_cache_v3.py``.
"""

import importlib
import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch_tensorrt  # noqa: E402
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (  # noqa: E402
    ATEN_POST_LOWERING_PASSES,
)
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
TOTAL = 24


@pytest.fixture
def static_cache_v3():
    # Importing the module registers its lowering pass for the whole process;
    # remove it afterwards so other tests compile without it.
    sys.modules.pop("static_cache_v3", None)
    module = importlib.import_module("static_cache_v3")
    yield module
    passes = ATEN_POST_LOWERING_PASSES.passes
    for index in reversed(range(len(passes))):
        if getattr(passes[index], "__name__", "") == "insert_static_cache_v3":
            del passes[index]
    sys.modules.pop("static_cache_v3", None)


def tiny_qwen3():
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return Qwen3ForCausalLM(config).eval().cuda()


@pytest.mark.parametrize("cudagraphs", [False, True])
def test_single_token_decode_matches_the_full_forward(static_cache_v3, cudagraphs):
    model = tiny_qwen3()
    tokens = torch.randint(0, 128, (TOTAL,), device="cuda")
    with torch.inference_mode():
        expected = model(tokens[None]).logits[0].float()
        static_cache_v3.configure(TOTAL)
        ids = tokens[:1].view(1, 1)
        positions = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        program = torch.export.export(
            static_cache_v3.DecodeStep(model), (ids, positions), strict=False
        )
        compiled = torch_tensorrt.dynamo.compile(
            program, arg_inputs=[ids, positions], min_block_size=1, disable_tf32=True
        )
        placeholders = [n for n in compiled.graph.nodes if n.op == "placeholder"]
        assert [n.name for n in placeholders[:2]] == ["input_ids", "position_ids"]
        caches = [
            torch.zeros(n.meta["val"].shape, dtype=n.meta["val"].dtype, device="cuda")
            for n in placeholders[2:]
        ]
        # Two layers with key and value caches holding the 2 grouped KV heads.
        assert [tuple(cache.shape) for cache in caches] == [(1, 2, TOTAL, 16)] * 4

        torch_tensorrt.runtime.set_cudagraphs_mode(cudagraphs)
        try:
            logits = []
            for position in range(TOTAL):
                # The engine writes the caches in place: keep passing the same
                # tensors (a captured CUDA graph keeps their addresses).
                out = compiled(
                    tokens[position].view(1, 1),
                    torch.full((1, 1), position, dtype=torch.int64, device="cuda"),
                    *caches,
                )
                logits.append(out[0][0, -1].float().clone())
        finally:
            torch_tensorrt.runtime.set_cudagraphs_mode(False)
    torch.testing.assert_close(torch.stack(logits), expected, atol=2e-3, rtol=2e-3)


def test_decode_steps_do_not_synchronize_with_the_host(static_cache_v3):
    model = tiny_qwen3()
    with torch.inference_mode():
        static_cache_v3.configure(TOTAL)
        ids = torch.ones((1, 1), dtype=torch.int64, device="cuda")
        positions = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        program = torch.export.export(
            static_cache_v3.DecodeStep(model), (ids, positions), strict=False
        )
        compiled = torch_tensorrt.dynamo.compile(
            program, arg_inputs=[ids, positions], min_block_size=1
        )
        placeholders = [n for n in compiled.graph.nodes if n.op == "placeholder"]
        caches = [
            torch.zeros(n.meta["val"].shape, dtype=n.meta["val"].dtype, device="cuda")
            for n in placeholders[2:]
        ]
        step = torch.full((1, 1), 5, dtype=torch.int64, device="cuda")
        compiled(ids, step, *caches)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            compiled(ids, step, *caches)
            torch.cuda.synchronize()
    copies = [event.name for event in profile.events() if "DtoH" in event.name]
    assert copies == []


def test_a_saved_decode_engine_keeps_the_calling_convention(static_cache_v3, tmp_path):
    model = tiny_qwen3()
    tokens = torch.randint(0, 128, (TOTAL,), device="cuda")
    with torch.inference_mode():
        expected = model(tokens[None]).logits[0].float()
        static_cache_v3.configure(TOTAL)
        ids = tokens[:1].view(1, 1)
        positions = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        program = torch.export.export(
            static_cache_v3.DecodeStep(model), (ids, positions), strict=False
        )
        compiled = torch_tensorrt.dynamo.compile(
            program, arg_inputs=[ids, positions], min_block_size=1, disable_tf32=True
        )
        placeholders = [n for n in compiled.graph.nodes if n.op == "placeholder"]
        shapes = [
            (tuple(n.meta["val"].shape), n.meta["val"].dtype) for n in placeholders[2:]
        ]
        torch_tensorrt.save(compiled, str(tmp_path / "decode.ep"), retrace=False)
        loaded = torch_tensorrt.load(str(tmp_path / "decode.ep")).module()
        caches = [
            torch.zeros(shape, dtype=dtype, device="cuda") for shape, dtype in shapes
        ]
        logits = []
        for position in range(TOTAL):
            out = loaded(
                tokens[position].view(1, 1),
                torch.full((1, 1), position, dtype=torch.int64, device="cuda"),
                *caches,
            )
            logits.append(out[0][0, -1].float().clone())
    torch.testing.assert_close(torch.stack(logits), expected, atol=2e-3, rtol=2e-3)
