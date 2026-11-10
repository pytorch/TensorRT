# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""qwen3_regions: Qwen3 decoder blocks as fused Triton kernels inside TensorRT.

Run with ``python -m pytest tools/llm/tests/test_qwen3_regions.py``.
"""

import importlib
import os
import sys

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import qwen3_regions  # noqa: E402
import torch_tensorrt  # noqa: E402
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (  # noqa: E402
    ATEN_POST_LOWERING_PASSES,
)
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
TOTAL = 16
# BLOCK_K must divide K (o_proj K = 4 heads x 32 = 128, down K = 256); half of
# K makes the gemv_add kernels loop over two chunks.
LAUNCH = {"attn_out": {"BLOCK_K": 64}, "mlp_out": {"BLOCK_K": 128}}


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
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).eval()
    # Random norm weights, so a kernel that drops or misplaces one shows.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("norm.weight"):
                parameter.uniform_(0.5, 1.5)
    return model.cuda().half()


@pytest.mark.parametrize("preset", ["mlp", "layers", "all"])
def test_the_patched_forward_computes_what_hugging_face_computes(preset):
    model = tiny_qwen3()
    ids = torch.tensor([[7]], device="cuda")
    positions = torch.tensor([[5]], device="cuda")
    with torch.no_grad():
        expected = model(input_ids=ids, position_ids=positions).logits.float()
        qwen3_regions.patch_qwen3(model, qwen3_regions.PRESETS[preset], LAUNCH)
        actual = model(input_ids=ids, position_ids=positions).logits.float()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("fold_scale", [0, 1])
def test_the_q_and_k_block_computes_what_hugging_face_computes(fold_scale):
    # With one token and no cache attention returns v whatever q and k are, so
    # the forward test above cannot see this block: compare it directly.
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    model = tiny_qwen3()
    qwen3_regions.patch_qwen3(
        model, ("qk_rope",), {"qk_rope": {"FOLD_SCALE": fold_scale}}
    )
    blocks, layer = model._qwen3_blocks, model.model.layers[1]
    attn = layer.self_attn
    torch.manual_seed(1)
    q = torch.randn(1, 1, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 1, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(1, 1, 64, device="cuda", dtype=torch.float16)
    position = torch.tensor([[11]], device="cuda")
    with torch.no_grad():
        cos, sin = model.model.rotary_emb(q, position)
        actual_q, actual_k, _ = blocks.qk_rope(layer, q, k, v, cos, sin)
        expected_q, expected_k = apply_rotary_pos_emb(
            attn.q_norm(q.view(1, 1, -1, 32)).transpose(1, 2),
            attn.k_norm(k.view(1, 1, -1, 32)).transpose(1, 2),
            cos,
            sin,
        )
    if fold_scale:
        expected_q = expected_q * attn.scaling
    torch.testing.assert_close(actual_q, expected_q, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual_k, expected_k, atol=2e-3, rtol=2e-3)


def test_unknown_blocks_and_untileable_launches_are_refused():
    model = tiny_qwen3()
    with pytest.raises(ValueError, match="Unknown blocks"):
        qwen3_regions.patch_qwen3(model, ("mlp_in", "attention"))
    ids = torch.tensor([[1]], device="cuda")
    positions = torch.tensor([[0]], device="cuda")
    for launch, match in (
        ({"attn_out": {"BLOCK_N": 3}}, "BLOCK_N=3 is not a power of two"),
        ({"attn_out": {"BLOCK_K": 96}}, "BLOCK_K=96 is not a power of two"),
        ({"mlp_out": {"BLOCK_K": 512}}, "256 is not a multiple of BLOCK_K=512"),
        # Programs must not straddle q and k (q has 128 rows, k and v 64).
        ({"attn_in": {"BLOCK_N": 128}}, "64 is not a multiple of BLOCK_N=128"),
    ):
        block = next(iter(launch))
        qwen3_regions.patch_qwen3(model, (block,), launch)
        with pytest.raises(ValueError, match=match):
            with torch.no_grad():
                model(input_ids=ids, position_ids=positions)
    config = model.config
    config.hidden_size = 96
    with pytest.raises(ValueError, match="hidden_size=96 is not a power of two"):
        qwen3_regions.patch_qwen3(model, ("mlp_in",))


@pytest.mark.parametrize(
    "preset, plugins",
    # Two layers: 'mlp' fuses 2 blocks per layer (mlp_in adds o_proj's output
    # as a pending residual), 'layers' 5, and 'all' adds the final block.
    [("mlp", 4), ("layers", 10), ("all", 11)],
)
def test_a_decode_engine_with_fused_blocks_matches_the_eager_model(
    static_cache_v3, preset, plugins
):
    from torch_tensorrt.autotune import describe_module

    model = tiny_qwen3()
    tokens = torch.randint(0, 256, (TOTAL,), device="cuda")
    with torch.no_grad():
        expected = model(tokens[None]).logits[0].float()
    qwen3_regions.patch_qwen3(model, qwen3_regions.PRESETS[preset], LAUNCH)
    compiled = qwen3_regions.compile_decode(
        model, max_seq_len=TOTAL, device=torch.device("cuda:0")
    )
    structure = describe_module(compiled)
    assert structure["trt_engines"] == 1 and structure["torch_partitions"] == 0
    assert structure["plugin_regions"] == plugins
    placeholders = [n for n in compiled.graph.nodes if n.op == "placeholder"]
    caches = [
        torch.zeros(n.meta["val"].shape, dtype=n.meta["val"].dtype, device="cuda")
        for n in placeholders[2:]
    ]
    # The plugins reload their module on every call outside CUDA graphs.
    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    try:
        logits = []
        with torch.no_grad():
            for position in range(TOTAL):
                out = compiled(
                    tokens[position].view(1, 1),
                    torch.full((1, 1), position, dtype=torch.int64, device="cuda"),
                    *caches,
                )
                logits.append(out[0][0, -1].float().clone())
    finally:
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
    actual = torch.stack(logits)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    assert (actual.argmax(-1) == expected.argmax(-1)).float().mean() >= 0.9


@pytest.mark.parametrize("preset, attention", [("", "lengths"), ("layers", None)])
def test_a_long_cache_engine_attends_only_to_the_written_slots(
    static_cache_v3, preset, attention
):
    # For long caches TensorRT 11.4 picks a decode-attention kernel that
    # ignores attention masks; attention="lengths" (the default "auto" for
    # caches over 192 slots) limits attention with IAttention
    # key_value_lengths instead. Garbage in every unwritten slot must not
    # change the logits.
    model = tiny_qwen3()
    tokens = torch.randint(0, 256, (TOTAL,), device="cuda")
    with torch.no_grad():
        expected = model(tokens[None]).logits[0].float()
    qwen3_regions.patch_qwen3(model, qwen3_regions.PRESETS[preset], LAUNCH)
    options = {} if attention is None else {"attention": attention}
    compiled = qwen3_regions.compile_decode(
        model, max_seq_len=1088, device=torch.device("cuda:0"), **options
    )
    placeholders = [n for n in compiled.graph.nodes if n.op == "placeholder"]
    torch.manual_seed(1)
    caches = [
        torch.randn(n.meta["val"].shape, device="cuda").to(n.meta["val"].dtype) * 3
        for n in placeholders[2:]
    ]
    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    try:
        logits = []
        with torch.no_grad():
            for position in range(TOTAL):
                out = compiled(
                    tokens[position].view(1, 1),
                    torch.full((1, 1), position, dtype=torch.int64, device="cuda"),
                    *caches,
                )
                logits.append(out[0][0, -1].float().clone())
    finally:
        torch_tensorrt.runtime.set_cudagraphs_mode(False)
    actual = torch.stack(logits)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def _kernel_names(run):
    """Names of the CUDA kernels that ``run()`` launches (CUDA graph replays too)."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    return {e.name for e in prof.events() if e.device_type == DeviceType.CUDA}


@pytest.mark.parametrize(
    "variant, preset, engines, regions, plugins, capture",
    # Two layers. lm-head: one engine, then the Inductor region. attention and
    # attention-math: a region per layer between three engines. attention-flex:
    # three Inductor flex-decoding plugins per layer, in one engine, under the
    # step graph or per-engine graphs.
    [
        ("lm-head", "layers", 1, 1, 10, "step"),
        ("attention", "layers", 3, 2, 10, "step"),
        ("attention-math", "layers", 3, 2, 10, "step"),
        ("attention-math", "", 3, 2, 0, "step"),
        ("attention-flex", "layers", 1, 0, 16, "step"),
        ("attention-flex", "layers", 1, 0, 16, "engine"),
    ],
)
def test_a_hybrid_decode_step_matches_the_eager_model(
    static_cache_v3, variant, preset, engines, regions, plugins, capture
):
    import torch._inductor.config as inductor_config
    from torch_tensorrt.autotune import describe_module
    from torch_tensorrt.dynamo.region_compilation import get_region_records

    model = tiny_qwen3()
    if variant == "attention-flex":
        # Inductor's 3-kernel flex-decoding layout needs 8 KV heads here
        # (with 2 it splits the combine into more kernels, which is refused).
        config = model.config
        config.num_attention_heads, config.num_key_value_heads = 16, 8
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(config).eval().cuda().half()
    tokens = torch.randint(0, 256, (TOTAL,), device="cuda")
    with torch.no_grad():
        expected = model(tokens[None]).logits[0].float()
    qwen3_regions.patch_qwen3(
        model, qwen3_regions.PRESETS[preset], LAUNCH, inductor=variant
    )
    with inductor_config.patch(max_autotune=False):
        compiled = qwen3_regions.compile_decode(
            model, max_seq_len=1088, device=torch.device("cuda:0")
        )
        structure = describe_module(compiled)
        assert structure["trt_engines"] == engines
        assert structure["plugin_regions"] == plugins
        records = get_region_records(compiled)
        torch_regions = [r for r in records if r.route == "torch"]
        assert len(torch_regions) == regions
        assert all(r.backend == "inductor" for r in torch_regions)
        placeholders = [n for n in compiled.graph.nodes if n.op == "placeholder"]
        torch.manual_seed(1)
        # Garbage in every unwritten slot must not change the logits.
        caches = [
            torch.randn(n.meta["val"].shape, device="cuda").to(n.meta["val"].dtype) * 3
            for n in placeholders[2:]
        ]
        token = tokens[:1].view(1, 1).clone()
        position = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        logits = []
        with torch.no_grad():
            if capture == "step":
                # One CUDA graph for the whole step (engines, plugins and
                # regions), the capture runtime.step_graph uses; warm-up
                # compiles the regions.
                for _ in range(2):
                    compiled(token, position, *caches)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = compiled(token, position, *caches)

                def step(index):
                    token.copy_(tokens[index].view(1, 1))
                    position.fill_(index)
                    graph.replay()
                    return out

            else:
                # The session's capture: each engine in its own CUDA graph.
                torch_tensorrt.runtime.set_cudagraphs_mode(True)

                def step(index):
                    index_tensor = torch.full(
                        (1, 1), index, dtype=torch.int64, device="cuda"
                    )
                    return compiled(tokens[index].view(1, 1), index_tensor, *caches)

            try:
                for index in range(TOTAL):
                    logits.append(step(index)[0][0, -1].float().clone())
                # Inductor's kernels run: its flex-decoding template (hosted
                # or in a region), or the kernels it generated for a region.
                names = _kernel_names(lambda: step(TOTAL - 1))
            finally:
                torch_tensorrt.runtime.set_cudagraphs_mode(False)
    actual = torch.stack(logits)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)
    assert (actual.argmax(-1) == expected.argmax(-1)).float().mean() >= 0.9
    if variant == "attention-flex":
        assert {"inductor_flex_split", "inductor_flex_out"} <= names
    elif variant == "attention":
        assert any(
            name.startswith("triton_") and "flex_attention" in name for name in names
        ), sorted(names)
    else:
        assert any(name.startswith("triton_") for name in names), sorted(names)
