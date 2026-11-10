# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""agentic_encoder: the encoder workload's inputs, attention rewrite and knobs.

Run with ``python -m pytest tools/llm/tests/test_agentic_encoder.py``; no GPU
or model download is needed.
"""

import math
import os
import re
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import agentic_encoder  # noqa: E402
from torch_tensorrt.autotune._knobs import knobs_by_name, resolve  # noqa: E402
from torch_tensorrt.autotune._settings import violations  # noqa: E402
from transformers import BertConfig, ModernBertConfig, ModernBertModel  # noqa: E402


def modernbert_config(**overrides):
    """A tiny ModernBERT: global layers 0 and 3, local window 4 (half width)."""
    settings = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=2,
        local_attention=8,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=1,
        sep_token_id=2,
    )
    settings.update(overrides)
    return ModernBertConfig(**settings)


def dense_band_attention(query, key, value, window):
    positions = torch.arange(query.shape[2])
    mask = (positions[:, None] - positions[None, :]).abs() <= window
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)


def test_sequence_ids_frame_repeated_text():
    assert agentic_encoder.sequence_ids([5, 6, 7], 8, 1, 2) == [1, 5, 6, 7, 5, 6, 7, 2]
    assert agentic_encoder.sequence_ids([5, 6, 7], 4, None, 2) == [5, 6, 7, 2]
    assert agentic_encoder.sequence_ids([5], 3, None, None) == [5, 5, 5]
    with pytest.raises(ValueError, match="no room"):
        agentic_encoder.sequence_ids([5], 2, 1, 2)
    with pytest.raises(ValueError, match="no tokens"):
        agentic_encoder.sequence_ids([], 8, 1, 2)


def test_band_block_mask_marks_the_window_inside_the_sequence():
    mask = agentic_encoder.band_block_mask(length=10, window=2, block=4)
    # Three blocks of four queries, each over a span of 4 + 2 * 2 keys.
    assert mask.shape == (3, 1, 4, 8)
    for block in range(3):
        for row in range(4):
            query = block * 4 + row
            for col in range(8):
                key = block * 4 + col - 2
                expected = abs(query - key) <= 2 and 0 <= key < 10
                assert bool(mask[block, 0, row, col]) == expected


@pytest.mark.parametrize(
    "length, window, block",
    [(64, 4, 16), (50, 4, 16), (37, 3, 64), (40, 0, 8), (33, 64, 128), (96, 8, 8)],
)
def test_banded_attention_matches_dense_masked_attention(length, window, block):
    torch.manual_seed(0)
    query, key, value = (
        torch.randn(2, 3, length, 8, dtype=torch.float64) for _ in range(3)
    )
    expected = dense_band_attention(query, key, value, window).transpose(1, 2)
    actual = agentic_encoder.banded_attention(
        query, key, value, window=window, block=block, scale=None
    )
    assert actual.shape == (2, length, 3, 8)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_banded_attention_passes_the_scale():
    torch.manual_seed(1)
    query, key, value = (
        torch.randn(1, 2, 24, 4, dtype=torch.float64) for _ in range(3)
    )
    positions = torch.arange(24)
    mask = (positions[:, None] - positions[None, :]).abs() <= 3
    expected = F.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=0.3
    ).transpose(1, 2)
    actual = agentic_encoder.banded_attention(
        query, key, value, window=3, block=8, scale=0.3
    )
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("local_attention", ["dense", "banded64", "banded128"])
def test_encoder_wrapper_reproduces_the_model(local_attention):
    """Explicit masks and the banded rewrite give Hugging Face's own output."""
    torch.manual_seed(0)
    config = modernbert_config()
    model = ModernBertModel(config).eval()
    ids = torch.randint(3, config.vocab_size, (1, 70))
    with torch.no_grad():
        expected = model(input_ids=ids).last_hidden_state
        window = config.sliding_window
        assert window == 4
        wrapper = agentic_encoder._Encoder(
            model, window=window, local_attention=local_attention
        )
        actual = wrapper(ids)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert torch.equal(
        agentic_encoder.band_mask(70, window)[0, 0],
        (torch.arange(70)[:, None] - torch.arange(70)[None, :]).abs() <= window,
    )


def test_banded_attention_needs_sliding_window_layers():
    model = ModernBertModel(modernbert_config()).eval()
    with pytest.raises(ValueError, match="sliding-window"):
        agentic_encoder._Encoder(model, window=None, local_attention="banded128")


def test_hidden_state_metrics():
    torch.manual_seed(0)
    reference = torch.randn(1, 6, 8)
    same = agentic_encoder.hidden_state_metrics(reference.half().float(), reference)
    assert same["token_cosine_min"] > 0.9999
    assert same["rel_l2_error"] < 1e-3
    noisy = reference.clone()
    noisy[0, 2] = -noisy[0, 2]
    metrics = agentic_encoder.hidden_state_metrics(noisy, reference)
    assert metrics["token_cosine_min"] == pytest.approx(-1.0)
    assert metrics["token_cosine_mean"] == pytest.approx(4 / 6)
    assert metrics["outlier_token_fraction"] == pytest.approx(1 / 6)
    assert metrics["max_abs_error"] == pytest.approx(
        2 * reference[0, 2].abs().max().item()
    )
    # The bulk check drops the worst token (1% of 6 rounds up to one).
    assert metrics["rel_l2_error_trimmed"] == pytest.approx(0.0, abs=1e-12)
    broken = reference.clone()
    broken[0, 0, 0] = math.nan
    failed = agentic_encoder.hidden_state_metrics(broken, reference)
    assert failed["token_cosine_min"] == -1.0 and math.isinf(failed["rel_l2_error"])
    assert math.isinf(failed["rel_l2_error_trimmed"])
    assert failed["block_cosine_median_min"] == -1.0
    assert failed["outlier_token_fraction"] == 1.0
    with pytest.raises(ValueError, match="reference"):
        agentic_encoder.hidden_state_metrics(reference[:, :5], reference)
    # Shapes are compared before flattening: same element count is not enough.
    with pytest.raises(ValueError, match="reference"):
        agentic_encoder.hidden_state_metrics(reference[0], reference)
    with pytest.raises(ValueError, match="reference"):
        agentic_encoder.hidden_state_metrics(reference.view(2, 3, 8), reference)


def test_block_median_sees_errors_the_trim_hides():
    torch.manual_seed(0)
    reference = torch.randn(1, 8192, 16, dtype=torch.float64)
    # 64 consecutive wrong tokens: under 1% of 8192, so the trim drops them.
    local = reference.clone()
    local[0, 4096:4160] += 0.3 * torch.randn(64, 16, dtype=torch.float64)
    metrics = agentic_encoder.hidden_state_metrics(local, reference)
    assert metrics["rel_l2_error_trimmed"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["block_cosine_median_min"] < 0.99
    # A remainder under half a block joins the last block.
    blocks = agentic_encoder._blocks(torch.arange(70.0), 64)
    assert [len(part) for part in blocks] == [70]
    assert [len(part) for part in agentic_encoder._blocks(torch.arange(100.0), 64)] == [
        64,
        36,
    ]


def perturbed(reference, bulk, outliers=(), outlier_scale=3.0, seed=1):
    """``reference`` plus relative noise: ``bulk`` per token, more on ``outliers``."""
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(reference.shape, generator=generator, dtype=reference.dtype)
    noise *= reference.norm(dim=-1, keepdim=True) / noise.norm(dim=-1, keepdim=True)
    out = reference + bulk * noise
    for token in outliers:
        out[0, token] = reference[0, token] + outlier_scale * noise[0, token]
    return out


@pytest.mark.parametrize("length", [512, 8192])
def test_accuracy_bounds_separate_rounding_from_semantic_errors(length):
    torch.manual_seed(0)
    bounds = agentic_encoder.accuracy_bounds(length)
    assert set(bounds) == {
        "rel_l2_error_trimmed",
        "block_cosine_median_min",
        "outlier_token_fraction",
        "token_cosine_min",
        "rel_l2_error",
    }
    reference = torch.randn(1, length, 64, dtype=torch.float64)

    def misses(output):
        metrics = agentic_encoder.hidden_state_metrics(output, reference)
        return violations(metrics, bounds)

    # FP16-like: small error on every token plus two ill-conditioned tokens.
    fp16_like = perturbed(reference, 0.003, outliers=(271, 287), outlier_scale=1.2)
    assert misses(fp16_like) == []
    # A systematic error on every token (a shifted window is ~0.009-0.012).
    assert any("trimmed" in miss for miss in misses(perturbed(reference, 0.012)))
    # Two anti-correlated tokens.
    flipped = reference.clone()
    flipped[0, 10] = -flipped[0, 10]
    assert any("token_cosine_min" in miss for miss in misses(flipped))
    # One block of 64 tokens wrong.
    local = reference.clone()
    local[0, 128:192] = perturbed(reference, 0.2)[0, 128:192]
    assert any("block_cosine_median_min" in miss for miss in misses(local))
    # More scattered outliers than the bulk check leaves out (2% of tokens).
    scattered = perturbed(reference, 0.003, outliers=range(0, length, 50))
    assert any("outlier_token_fraction" in miss for miss in misses(scattered))
    # Right direction, wrong magnitude on a few tokens.
    scaled = reference.clone()
    scaled[0, :3] *= 10
    assert any("rel_l2_error=" in miss for miss in misses(scaled))


def test_accuracy_bounds_interpolate_between_the_calibrated_lengths():
    limit = {
        n: agentic_encoder.accuracy_bounds(n)["rel_l2_error_trimmed"][1]
        for n in (128, 512, 2048, 8192, 32768)
    }
    assert limit[128] == limit[512] == agentic_encoder._BULK_LIMIT[512]
    assert limit[8192] == limit[32768] == agentic_encoder._BULK_LIMIT[8192]
    assert limit[512] < limit[2048] < limit[8192]


def test_attention_gflop_counts_dense_and_banded_work():
    types = ["full_attention", "sliding_attention", "sliding_attention"]
    dense = agentic_encoder.attention_gflop(1024, 64, types, 64, block=None)
    banded = agentic_encoder.attention_gflop(1024, 64, types, 64, block=128)
    assert dense["attention_gflop"] == pytest.approx(3 * 4 * 1024 * 1024 * 64 / 1e9)
    assert banded["attention_gflop"] == pytest.approx(
        4 * 1024 * 64 * (1024 + 2 * 256) / 1e9
    )
    needed = 4 * 1024 * 64 * (1024 + 2 * 129) / 1e9
    assert dense["attention_gflop_needed"] == banded["attention_gflop_needed"]
    assert banded["attention_gflop_needed"] == pytest.approx(needed)


def test_modernbert_knobs_and_baseline(tmp_path):
    modernbert_config().save_pretrained(tmp_path / "modernbert")
    workload = agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=512)
    knobs = knobs_by_name(workload.knobs())
    assert knobs["workload.local_attention"].values == (
        "dense",
        "banded128",
        "banded64",
        "banded256",
    )
    assert "4 positions away" in knobs["workload.local_attention"].description
    assert knobs["engine.precision"].values == ("fp16", "fp32")
    assert "engine.tiling_optimization_level" not in knobs
    if agentic_encoder.build_routes_supported():
        assert knobs["engine.build_route"].values == (
            "",
            "-cask:fp16_acc=off",
            "-cask:fp16_acc=off -cask_fusion:fp16_acc=off",
            "-tactic:splitk=off",
        )
        assert knobs["engine.build_route"].default == ""
    assert {"graph.fold_attention_masks", "engine.decompose_attention"} <= set(knobs)
    assert {"runtime.cudagraphs", "engine.optimization_level"} <= set(knobs)
    values = resolve(workload.baseline(), knobs)
    assert values["workload.local_attention"] == "dense"
    assert values["runtime.cudagraphs"] is False
    assert workload.timing().calls == 10
    assert (
        agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=8192)
        .timing()
        .calls
        == 2
    )
    assert workload.accuracy_defaults() == agentic_encoder.accuracy_bounds(512)


def test_other_encoders_get_no_attention_knobs(tmp_path):
    BertConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=1, num_attention_heads=2
    ).save_pretrained(tmp_path / "bert")
    workload = agentic_encoder.EncoderWorkload(str(tmp_path / "bert"), length=64)
    names = {knob.name for knob in workload.knobs()}
    assert "workload.local_attention" not in names
    assert "graph.fold_attention_masks" not in names
    assert "engine.precision" in names


def test_unknown_backend_is_refused(tmp_path):
    modernbert_config().save_pretrained(tmp_path / "modernbert")
    workload = agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=64)
    with pytest.raises(ValueError, match="bench.backend"):
        workload.build({"bench.backend": "onnx"})
    with pytest.raises(ValueError, match="needs length"):
        agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=4)


def test_bench_values_are_checked():
    check = agentic_encoder.check_bench_values
    assert check({"engine.precision": "fp16"}) == "tensorrt"
    assert (
        check(
            {
                "engine.precision": "fp16",
                "bench.backend": "inductor",
                "bench.mode": "max-autotune",
                "bench.attn": "flex_attention",
                "bench.inductor": {"coordinate_descent_tuning": True},
                "bench.fp16_accumulation": True,
            }
        )
        == "inductor"
    )
    eager = {"bench.backend": "eager", "bench.fp16_accumulation": True}
    assert check({**eager, "engine.precision": "fp16"}) == "eager"
    with pytest.raises(ValueError, match="Unknown bench keys"):
        check({"bench.fp16_acc": True})
    with pytest.raises(ValueError, match="do not apply"):
        check({"bench.mode": "max-autotune"})
    with pytest.raises(ValueError, match="do not apply"):
        check({"bench.backend": "eager", "bench.inductor": {}})
    with pytest.raises(ValueError, match="needs engine.precision 'fp16'"):
        check({**eager, "engine.precision": "fp32"})


def test_fp16_accumulation_is_restored():
    matmul = torch.backends.cuda.matmul
    before = matmul.allow_fp16_accumulation
    with agentic_encoder._fp16_accumulation(True):
        assert matmul.allow_fp16_accumulation is True
    assert matmul.allow_fp16_accumulation == before


def test_build_route_probe_needs_no_gpu():
    agentic_encoder.build_routes_supported.cache_clear()
    first = agentic_encoder.build_routes_supported()
    assert agentic_encoder.build_routes_supported() is first
    assert agentic_encoder.build_routes_supported.cache_info().hits >= 1


# ------------------------------------------------------- kernels.inductor
import modernbert_kernels  # noqa: E402


def test_kernels_inductor_knob_needs_cuda_graphs(tmp_path):
    modernbert_config().save_pretrained(tmp_path / "modernbert")
    workload = agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=512)
    knobs = knobs_by_name(workload.knobs())
    knob = knobs["kernels.inductor"]
    assert knob.values == (
        "none",
        "rope-kernel",
        "inductor-regions",
        "rope-kernel-eager-numerics",
    )
    assert knob.values == modernbert_kernels.VARIANTS
    assert set(modernbert_kernels.KERNEL_VARIANTS) < set(knob.values)
    assert knob.default == "none"
    assert dict(knob.requires) == {"runtime.cudagraphs": True}
    # A planner must not read measured results as facts about the knob.
    assert not re.search(r"\d\s*(ms|us|%|x faster)", knob.description)
    # Counts follow the model: 4 layers here (ModernBERT-base has 22).
    assert "transformer layer (4)" in knob.description
    assert "4 regions, 8 TensorRT/PyTorch boundaries and 5 engines" in knob.description
    assert "22" not in knob.description
    # Off by default: existing configs resolve as before.
    assert resolve({}, knobs)["kernels.inductor"] == "none"
    session = {
        "engine.decompose_attention": True,
        "graph.fold_attention_masks": True,
        "workload.local_attention": "banded128",
    }
    assert resolve(session, knobs)["kernels.inductor"] == "none"
    for variant in knob.values[1:]:
        with pytest.raises(ValueError, match="requires runtime.cudagraphs=True"):
            resolve({**session, "kernels.inductor": variant}, knobs)
        values = resolve(
            {**session, "kernels.inductor": variant, "runtime.cudagraphs": True},
            knobs,
        )
        assert agentic_encoder.check_hybrid_values(values) == variant


def test_check_hybrid_values():
    check = agentic_encoder.check_hybrid_values
    assert check({}) == "none"
    assert check({"kernels.inductor": "none"}) == "none"
    with pytest.raises(ValueError, match="needs runtime.cudagraphs"):
        check({"kernels.inductor": "rope-kernel", "runtime.cudagraphs": False})
    with pytest.raises(ValueError, match="Unknown kernels.inductor"):
        check({"kernels.inductor": "all", "runtime.cudagraphs": True})


def _encoder_candidate(module, hybrid=None):
    details = {"inductor_kernels": hybrid} if hybrid else {}
    return agentic_encoder._EncoderCandidate(
        module,
        torch.zeros(1, 4, dtype=torch.int64),
        backend="tensorrt",
        details=details,
    )


def test_region_builds_report_that_they_cannot_be_saved(tmp_path):
    # The exporter itself rejects a program with an Inductor region record
    # (before writing anything); the candidate names the value in the error.
    from torch_tensorrt import region
    from torch_tensorrt.autotune._measure import export_unsupported
    from torch_tensorrt.dynamo.region_compilation._types import (
        RegionRecord,
        attach_region_records,
    )

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    program = torch.fx.symbolic_trace(AddOne())
    record = RegionRecord(
        id="rope.0#0", name="rope.0", child_target="x", backend="inductor"
    )
    attach_region_records(program, [record])
    assert agentic_encoder.REGION_VARIANTS == ("inductor-regions",)
    candidate = _encoder_candidate(program, "inductor-regions")
    with pytest.raises(NotImplementedError, match="cannot be saved") as caught:
        candidate.export(tmp_path)
    assert "kernels.inductor='inductor-regions'" in str(caught.value)
    assert isinstance(caught.value.__cause__, region.RegionError)
    assert export_unsupported(caught.value)
    assert list(tmp_path.iterdir()) == []
    # The hosted-kernel values and plain builds keep the exporter's error.
    for hybrid in (None, "rope-kernel", "rope-kernel-eager-numerics"):
        with pytest.raises(region.RegionError, match="is not supported"):
            _encoder_candidate(program, hybrid).export(tmp_path)


def test_builds_without_regions_save_their_program(tmp_path, monkeypatch):
    import torch_tensorrt

    saved = []
    monkeypatch.setattr(
        torch_tensorrt,
        "save",
        lambda module, path, **kwargs: saved.append((module, path, kwargs)),
    )
    module = torch.nn.Identity()
    candidate = _encoder_candidate(module, "rope-kernel")
    assert candidate.export(tmp_path) == ["encoder.ep"]
    ((saved_module, path, kwargs),) = saved
    assert saved_module is module and path == str(tmp_path / "encoder.ep")
    assert kwargs["retrace"] is False
    assert kwargs["arg_inputs"][0] is candidate.input_ids


def test_hybrid_builds_are_refused_before_loading(tmp_path):
    """Both refusals happen before the model is loaded: no GPU needed."""
    modernbert_config().save_pretrained(tmp_path / "modernbert")
    workload = agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=64)
    knobs = knobs_by_name(workload.knobs())
    values = resolve({}, knobs)
    values["kernels.inductor"] = "rope-kernel"
    with pytest.raises(ValueError, match="needs runtime.cudagraphs"):
        workload.build(values)
    eager = {**values, "runtime.cudagraphs": True, "bench.backend": "eager"}
    with pytest.raises(ValueError, match="bench.backend='tensorrt'"):
        workload.build(eager)


def test_split_rope_is_hugging_face_rope():
    """The region body equals Hugging Face's rotary embedding and RoPE, bit for bit."""
    from transformers.models.modernbert.modeling_modernbert import (
        apply_rotary_pos_emb,
    )

    torch.manual_seed(0)
    config = modernbert_config()
    model = ModernBertModel(config).eval()
    heads, dim = (
        config.num_attention_heads,
        config.hidden_size // config.num_attention_heads,
    )
    qkv = torch.randn(1, 37, 3 * config.hidden_size)
    hidden = torch.randn(1, 37, config.hidden_size)
    positions = torch.arange(37).unsqueeze(0)
    for layer_type in ("full_attention", "sliding_attention"):
        cos, sin = model.rotary_emb(hidden, positions, layer_type)
        query, key, _ = qkv.view(1, 37, 3, heads, dim).unbind(dim=-3)
        expected = apply_rotary_pos_emb(
            query.transpose(1, 2), key.transpose(1, 2), cos, sin
        )
        inv_freq = getattr(model.rotary_emb, f"{layer_type}_inv_freq")
        actual = modernbert_kernels.split_rope(qkv, inv_freq, heads, dim)
        for a, e in zip(actual, expected):
            assert a.is_contiguous() and a.shape == (1, 37, heads, dim)
            assert torch.equal(a.transpose(1, 2), e)


def fake_kernel(**overrides):
    """An InductorKernel record around a stand-in @triton.jit kernel."""
    settings = dict(
        name="triton_poi_fused_fake_0",
        fn=_fake_inductor_kernel(),
        source="def triton_poi_fused_fake_0(): pass",
        path="",
        parameters=("in_ptr0", "in_ptr1", "out_ptr1", "out_ptr3", "xnumel", "XBLOCK"),
        inputs=(("in_ptr0", 0), ("in_ptr1", 1)),
        outputs=(("out_ptr1", 0), ("out_ptr3", 1)),
        numels=(("xnumel", 4096),),
        config={"XBLOCK": 512, "num_warps": 4, "num_stages": 1},
        mode=modernbert_kernels.GENERATION_MODE,
        length=64,
        dtypes=("fp16", "fp32", "fp16", "fp16"),
    )
    settings.update(overrides)
    return modernbert_kernels.InductorKernel(**settings)


def _fake_inductor_kernel():
    triton = pytest.importorskip("triton")
    import triton.language as tl

    @triton.jit
    def triton_poi_fused_fake_0(
        in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, XBLOCK: tl.constexpr
    ):
        xindex = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)[:]
        x = tl.load(in_ptr0 + xindex).to(tl.float32)
        tl.store(out_ptr1 + xindex, x)
        tl.store(out_ptr3 + xindex, x * tl.load(in_ptr1 + (xindex % 32)))

    return triton_poi_fused_fake_0


@pytest.mark.parametrize(
    "variant", ["inductor-regions", "rope-kernel", "rope-kernel-eager-numerics"]
)
def test_patched_model_is_hugging_face_in_eager(variant):
    """Outside torch_tensorrt.compile the regions run their bodies: same output."""
    torch.manual_seed(0)
    config = modernbert_config()
    model = ModernBertModel(config).eval()
    ids = torch.randint(3, config.vocab_size, (1, 70))
    with torch.no_grad():
        expected = model(input_ids=ids).last_hidden_state
        kernel = None
        if variant in modernbert_kernels.KERNEL_VARIANTS:
            kernel = fake_kernel(
                eager_numerics=modernbert_kernels.KERNEL_VARIANTS[variant]
            )
        with modernbert_kernels.no_hosted_kernels():
            assert modernbert_kernels.install(model, variant, kernel) == 4
            if kernel is not None:
                assert (
                    modernbert_kernels.hosted_kernel(kernel.key).__name__ == kernel.name
                )
            for local_attention in ("dense", "banded64"):
                wrapper = agentic_encoder._Encoder(
                    model, window=config.sliding_window, local_attention=local_attention
                )
                torch.testing.assert_close(wrapper(ids), expected, rtol=1e-5, atol=1e-5)
    attention = model.layers[1].attn
    assert torch.equal(
        attention.hybrid_inv_freq, model.rotary_emb.sliding_attention_inv_freq
    )
    assert "hybrid_inv_freq" not in model.state_dict().keys()


def _unpatched(model):
    return all(
        "hybrid_inv_freq" not in dict(layer.attn.named_buffers())
        and not hasattr(layer.attn, "_modernbert_hybrid")
        for layer in model.layers
    )


def test_install_checks_its_arguments():
    model = ModernBertModel(modernbert_config()).eval()
    assert modernbert_kernels.install(model, "none") == 0
    with pytest.raises(ValueError, match="Unknown variant"):
        modernbert_kernels.install(model, "everything")
    with modernbert_kernels.no_hosted_kernels():
        with pytest.raises(ValueError, match="rope-kernel needs a kernel"):
            modernbert_kernels.install(model, "rope-kernel")
        with pytest.raises(ValueError, match="inductor-regions takes no kernel"):
            modernbert_kernels.install(model, "inductor-regions", fake_kernel())
        # The kernel must have been generated for the variant's numerics.
        with pytest.raises(ValueError, match="eager_numerics=True"):
            modernbert_kernels.install(
                model, "rope-kernel-eager-numerics", fake_kernel()
            )
        with pytest.raises(ValueError, match="eager_numerics=False"):
            modernbert_kernels.install(
                model, "rope-kernel", fake_kernel(eager_numerics=True)
            )
        # ... and for the model's RoPE: its attention scaling and inv_freq dtype.
        with pytest.raises(ValueError, match="scale RoPE by 1.0"):
            modernbert_kernels.install(model, "rope-kernel", fake_kernel(scaling=0.5))
        with pytest.raises(ValueError, match="inv_freq; the hosted kernel reads fp16"):
            modernbert_kernels.install(
                model, "rope-kernel", fake_kernel(dtypes=("fp16",) * 4)
            )
    assert _unpatched(model)
    with pytest.raises(RuntimeError, match="No hosted Inductor kernel"):
        modernbert_kernels.hosted_kernel("triton_poi_fused_missing_0")


def test_install_checks_every_layer_before_patching_any():
    model = ModernBertModel(modernbert_config()).eval()
    rotary = model.rotary_emb
    assert set(modernbert_kernels.layer_rope(model)) == {
        "full_attention",
        "sliding_attention",
    }
    # Layer 0 is a global layer; a local layer's RoPE type fails after it.
    rotary.rope_type = {**rotary.rope_type, "sliding_attention": "linear"}
    with pytest.raises(
        ValueError, match="sliding_attention layers use RoPE type 'linear'"
    ):
        modernbert_kernels.install(model, "inductor-regions")
    assert _unpatched(model)


def test_hybrid_build_generates_the_kernel_for_the_models_rope(tmp_path, monkeypatch):
    """_install_hybrid passes the RoPE scaling, dtypes and numerics; no GPU needed."""
    modernbert_config().save_pretrained(tmp_path / "modernbert")
    workload = agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=64)
    calls = []

    def generate(length, heads, head_dim, **kwargs):
        calls.append(dict(length=length, heads=heads, head_dim=head_dim, **kwargs))
        return fake_kernel(
            scaling=kwargs["scaling"], eager_numerics=kwargs["eager_numerics"]
        )

    monkeypatch.setattr(modernbert_kernels, "generate_rope_kernel", generate)
    monkeypatch.setattr(modernbert_kernels, "check_resources", lambda kernel: {})
    with modernbert_kernels.no_hosted_kernels():
        for variant, eager in modernbert_kernels.KERNEL_VARIANTS.items():
            model = ModernBertModel(modernbert_config()).eval()
            details = workload._install_hybrid(model, variant)
            assert calls[-1] == dict(
                length=64,
                heads=2,
                head_dim=16,
                scaling=1.0,
                dtype=torch.float32,
                inv_freq_dtype=torch.float32,
                eager_numerics=eager,
            )
            assert details["hybrid_layers"] == 4
            assert details["inductor_emulate_precision_casts"] is eager
        # One kernel serves every layer: layer types with different RoPE
        # scalings are refused before anything is generated.
        model = ModernBertModel(modernbert_config()).eval()
        model.rotary_emb.sliding_attention_attention_scaling = 0.5
        with pytest.raises(ValueError, match="RoPE scaling or inv_freq dtype differ"):
            workload._install_hybrid(model, "rope-kernel")
        assert len(calls) == 2 and _unpatched(model)
        regions = workload._install_hybrid(
            ModernBertModel(modernbert_config()), "inductor-regions"
        )
        assert regions == {"inductor_kernels": "inductor-regions", "hybrid_layers": 4}
    assert len(calls) == 2


# Inductor's output code for split_rope (trimmed): one kernel, the inputs
# arg0_1 (qkv) and arg1_1 (inv_freq), the outputs buf1 (query), buf3 (key).
INDUCTOR_CODE = """
# kernel path: /cache/mv/cmvaufcm.py
triton_poi_fused_rope_0 = async_compile.triton('triton_poi_fused_rope_0', '''
import triton

@triton.jit
def triton_poi_fused_rope_0(in_ptr0, in_ptr1, out_ptr1, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
''', device_str='cuda')

class Runner:
    def call(self, args):
        arg0_1, arg1_1 = args
        args.clear()
        with torch.cuda._DeviceGuard(0):
            buf1 = empty_strided_cuda((1, 8192, 12, 64), (6291456, 768, 64, 1), torch.float16)
            buf3 = empty_strided_cuda((1, 8192, 12, 64), (6291456, 768, 64, 1), torch.float16)
            triton_poi_fused_rope_0.run(arg0_1, arg1_1, buf1, buf3, 6291456, stream=raw_stream0)
            del arg0_1
        return (buf1, buf3, )
"""
PARAMETERS = ["in_ptr0", "in_ptr1", "out_ptr1", "out_ptr3", "xnumel", "XBLOCK"]


def test_parse_inductor_wrapper_and_bind_the_kernel():
    parsed = modernbert_kernels.parse_wrapper(INDUCTOR_CODE)
    assert parsed == {
        "name": "triton_poi_fused_rope_0",
        "path": "/cache/mv/cmvaufcm.py",
        "run_args": ["arg0_1", "arg1_1", "buf1", "buf3", "6291456"],
        "graph_inputs": ["arg0_1", "arg1_1"],
        "graph_outputs": ["buf1", "buf3"],
    }
    inputs, outputs, numels = modernbert_kernels.bind_parameters(
        PARAMETERS, ["XBLOCK"], parsed
    )
    assert inputs == (("in_ptr0", 0), ("in_ptr1", 1))
    assert outputs == (("out_ptr1", 0), ("out_ptr3", 1))
    assert numels == (("xnumel", 6291456),)
    # A 2-D tiled kernel: (..., ynumel, xnumel, YBLOCK, XBLOCK).
    tiled = {**parsed, "run_args": ["arg0_1", "arg1_1", "buf1", "buf3", "8192", "768"]}
    _, _, numels = modernbert_kernels.bind_parameters(
        PARAMETERS[:4] + ["ynumel", "xnumel", "YBLOCK", "XBLOCK"],
        ["YBLOCK", "XBLOCK"],
        tiled,
    )
    assert numels == (("ynumel", 8192), ("xnumel", 768))
    # Outputs returned in the other order bind the other way round.
    swapped = {**parsed, "graph_outputs": ["buf3", "buf1"]}
    _, outputs, _ = modernbert_kernels.bind_parameters(PARAMETERS, ["XBLOCK"], swapped)
    assert outputs == (("out_ptr1", 1), ("out_ptr3", 0))


def test_parse_refuses_what_one_plugin_cannot_host():
    two = INDUCTOR_CODE.replace(
        "class Runner:",
        "triton_poi_fused_copy_1 = async_compile.triton('triton_poi_fused_copy_1', "
        "'''\\n''', device_str='cuda')\n\nclass Runner:",
    )
    with pytest.raises(ValueError, match="2 Triton kernels"):
        modernbert_kernels.parse_wrapper(two)
    extern = INDUCTOR_CODE.replace(
        "del arg0_1", "extern_kernels.bmm(arg1_1, arg1_1, out=buf1)"
    )
    with pytest.raises(ValueError, match="extern kernel"):
        modernbert_kernels.parse_wrapper(extern)
    parsed = modernbert_kernels.parse_wrapper(INDUCTOR_CODE)
    # A kernel writing an intermediate buffer is part of a multi-kernel region.
    intermediate = {
        **parsed,
        "run_args": ["arg0_1", "arg1_1", "buf1", "buf2", "6291456"],
    }
    with pytest.raises(ValueError, match="more than one kernel"):
        modernbert_kernels.bind_parameters(PARAMETERS, ["XBLOCK"], intermediate)
    dynamic = {**parsed, "run_args": ["arg0_1", "arg1_1", "buf1", "buf3", "s0"]}
    with pytest.raises(ValueError, match="not static"):
        modernbert_kernels.bind_parameters(PARAMETERS, ["XBLOCK"], dynamic)
    with pytest.raises(ValueError, match="Expected a pointwise kernel"):
        modernbert_kernels.bind_parameters(
            PARAMETERS[:-1] + ["r0_numel", "XBLOCK", "R0_BLOCK"],
            ["XBLOCK", "R0_BLOCK"],
            parsed,
        )


def test_launcher_calls_inductors_kernel_in_plugin_order():
    # Inductor's parameters receive (key, qkv, query, inv_freq): the launcher
    # takes the plugin's order (inputs, then outputs) and permutes.
    kernel = fake_kernel(
        parameters=("in_ptr0", "out_ptr0", "in_ptr1", "out_ptr1", "xnumel", "XBLOCK"),
        inputs=(("in_ptr0", 1), ("in_ptr1", 0)),
        outputs=(("out_ptr0", 1), ("out_ptr1", 0)),
    )
    source = kernel.launcher_source()
    assert f"def {kernel.name}(in0, in1, out0, out1, XNUMEL: tl.constexpr" in source
    assert "_inductor_kernel(in1, out1, in0, out0, XNUMEL, XBLOCK)" in source
    for param in ("in0", "in1", "out0", "out1"):
        assert f"{param} = tl.multiple_of({param} + tl.program_id(2), 16)" in source
    assert kernel.constexprs() == {"XNUMEL": 4096, "XBLOCK": 512}
    assert kernel.grid == (8,)
    assert kernel.describe()["inductor_kernel"] == kernel.name
    tiled = fake_kernel(
        parameters=("in_ptr0", "in_ptr1", "out_ptr0", "out_ptr1")
        + ("ynumel", "xnumel", "YBLOCK", "XBLOCK"),
        inputs=(("in_ptr0", 0), ("in_ptr1", 1)),
        outputs=(("out_ptr0", 0), ("out_ptr1", 1)),
        numels=(("ynumel", 512), ("xnumel", 768)),
        config={"YBLOCK": 64, "XBLOCK": 32, "num_warps": 4, "num_stages": 1},
    )
    source = tiled.launcher_source()
    assert (
        "(in0, in1, out0, out1, YNUMEL: tl.constexpr, XNUMEL: tl.constexpr, "
        "YBLOCK: tl.constexpr, XBLOCK: tl.constexpr)"
    ) in source
    assert (
        "_inductor_kernel(in0, in1, out0, out1, YNUMEL, XNUMEL, YBLOCK, XBLOCK)"
        in source
    )
    assert tiled.constexprs() == {
        "YNUMEL": 512,
        "XNUMEL": 768,
        "YBLOCK": 64,
        "XBLOCK": 32,
    }
    # Inductor's 2-D grid: x blocks on program id 0, y blocks on program id 1.
    assert tiled.grid == (24, 8)


def test_launcher_compiles_with_vectorized_access_as_the_plugin_does():
    """The plugin's compile path has no alignment specialization; the hints restore it."""
    pytest.importorskip("triton")
    from triton.backends.compiler import GPUTarget

    kernel = fake_kernel()
    with modernbert_kernels.no_hosted_kernels():
        resources = modernbert_kernels.check_resources(
            kernel, target=GPUTarget("cuda", 120, 32)
        )
    assert resources["entry"] == kernel.name
    assert resources["shared"] <= modernbert_kernels.SHARED_MEMORY_LIMIT
    # Without the alignment hints every FP16 access would be 16 bits wide.
    widths = resources["global_access_bits"]
    assert widths["ld"] and widths["st"]
    assert min(int(bits) for bits in widths["ld"]) >= 32
    assert min(int(bits) for bits in widths["st"]) >= 32


def test_access_widths_reads_ptx():
    ptx = """
    @%p1 ld.global.L1::evict_last.v2.b32 { %r1, %r2 }, [ %rd1 + 0 ];
    ld.global.b16 { %rs1 }, [ %rd2 + 0 ];
    ld.global.b32 { %r3 }, [ %rd3 + 0 ];
    st.global.v4.b32 [ %rd4 + 0 ], { %r4, %r5, %r6, %r7 };
    """
    assert modernbert_kernels.access_widths(ptx) == {
        "ld": {"64": 1, "16": 1, "32": 1},
        "st": {"128": 1},
    }


# ------------------------------------------------------ kernels.generated
import region_generation  # noqa: E402


def test_kernels_generated_knob_is_prepared_and_needs_cuda_graphs(tmp_path):
    from torch_tensorrt.autotune._knobs import apply_changes, with_requirements

    modernbert_config().save_pretrained(tmp_path / "modernbert")
    workload = agentic_encoder.EncoderWorkload(str(tmp_path / "modernbert"), length=512)
    knobs = knobs_by_name(workload.knobs())
    knob = knobs["kernels.generated"]
    assert knob.values == ("none", "split-rope") == agentic_encoder.GENERATED_KERNELS
    assert knob.values[1] == modernbert_kernels.GENERATED
    assert knob.default == "none"
    # The hosted variant's requirement, and no Inductor variant: both replace
    # each layer's QKV split and RoPE.
    assert dict(knob.requires) == {
        "runtime.cudagraphs": True,
        "kernels.inductor": "none",
    }
    # A session generates the kernel once, before its search.
    assert knob.prepare == ("split-rope",)
    assert not re.search(r"\d\s*(ms|us|%|x faster)", knob.description)
    assert "one plugin per transformer layer (4)" in knob.description
    # Off by default: existing configs resolve as before.
    assert resolve({}, knobs)["kernels.generated"] == "none"
    base = {"engine.decompose_attention": True}
    with pytest.raises(ValueError, match="requires runtime.cudagraphs=True"):
        apply_changes(base, {"kernels.generated": "split-rope"}, knobs)
    with pytest.raises(ValueError, match="requires kernels.inductor='none'"):
        apply_changes(
            {**base, "runtime.cudagraphs": True, "kernels.inductor": "rope-kernel"},
            {"kernels.generated": "split-rope"},
            knobs,
        )
    values = resolve(
        {**base, "runtime.cudagraphs": True, "kernels.generated": "split-rope"}, knobs
    )
    assert agentic_encoder.check_generated_values(values) == "split-rope"
    assert agentic_encoder.check_hybrid_values(values) == "none"
    # A planner's change brings CUDA graphs along, and drops a hosted kernel.
    hosted = {**base, "runtime.cudagraphs": True, "kernels.inductor": "rope-kernel"}
    bundled = with_requirements(hosted, {"kernels.generated": "split-rope"}, knobs)
    assert bundled == {"kernels.generated": "split-rope", "kernels.inductor": "none"}
    assert with_requirements(base, {"kernels.generated": "split-rope"}, knobs) == {
        "kernels.generated": "split-rope",
        "runtime.cudagraphs": True,
    }
    # Only ModernBERT has the subgraph.
    BertConfig(
        num_hidden_layers=1, hidden_size=32, num_attention_heads=2
    ).save_pretrained(tmp_path / "bert")
    bert = agentic_encoder.EncoderWorkload(str(tmp_path / "bert"), length=64)
    assert "kernels.generated" not in {knob.name for knob in bert.knobs()}


def test_check_generated_values():
    check = agentic_encoder.check_generated_values
    assert check({}) == "none"
    assert check({"kernels.generated": "none"}) == "none"
    with pytest.raises(ValueError, match="needs runtime.cudagraphs"):
        check({"kernels.generated": "split-rope", "runtime.cudagraphs": False})
    with pytest.raises(ValueError, match="needs kernels.inductor='none'"):
        check(
            {
                "kernels.generated": "split-rope",
                "runtime.cudagraphs": True,
                "kernels.inductor": "inductor-regions",
            }
        )
    with pytest.raises(ValueError, match="Unknown kernels.generated"):
        check({"kernels.generated": "mlp", "runtime.cudagraphs": True})


def test_generated_split_rope_is_hugging_face_in_eager(tmp_path):
    """Outside torch_tensorrt.compile the generated regions run their bodies."""
    torch.manual_seed(0)
    config = modernbert_config()
    model = ModernBertModel(config).eval()
    ids = torch.randint(3, config.vocab_size, (1, 70))
    generator = region_generation.make_generator(tmp_path)
    with torch.no_grad():
        expected = model(input_ids=ids).last_hidden_state
        installed = modernbert_kernels.install(
            model, modernbert_kernels.GENERATED, generator=generator
        )
        assert installed == 4
        wrapper = agentic_encoder._Encoder(model, window=config.sliding_window)
        torch.testing.assert_close(wrapper(ids), expected, rtol=1e-5, atol=1e-5)
    for layer in model.layers:
        hybrid = layer.attn._modernbert_hybrid
        assert hybrid.variant == "split-rope" and hybrid.generator is generator
        assert not hybrid.kernel_key


def test_install_needs_a_generator_for_the_generated_variant(tmp_path):
    model = ModernBertModel(modernbert_config()).eval()
    generator = region_generation.make_generator(tmp_path)
    with pytest.raises(ValueError, match="split-rope needs a generator"):
        modernbert_kernels.install(model, modernbert_kernels.GENERATED)
    with pytest.raises(ValueError, match="inductor-regions takes no generator"):
        modernbert_kernels.install(model, "inductor-regions", generator=generator)
    assert _unpatched(model)


def test_generated_split_rope_scopes_are_the_hosted_regions(tmp_path, monkeypatch):
    """The generated variant opens the hosted kernel's scope: same name and inputs."""
    from torch_tensorrt import region

    calls = []

    def fused_kernel(**kwargs):
        calls.append(kwargs)

    torch.manual_seed(0)
    model = ModernBertModel(modernbert_config()).eval()
    generator = region_generation.make_generator(tmp_path)
    modernbert_kernels.install(model, modernbert_kernels.GENERATED, generator=generator)
    monkeypatch.setattr(region, "fused_kernel", fused_kernel)
    hybrid = model.layers[2].attn._modernbert_hybrid
    qkv, inv_freq = torch.randn(1, 8, 96), model.layers[2].attn.hybrid_inv_freq
    hybrid.scope(2, qkv, inv_freq)
    (call,) = calls
    assert call["generator"] is generator and call["name"] == "split_rope_2"
    assert call["inputs"][0] is qkv and call["inputs"][1] is inv_freq
    assert set(call) == {"generator", "inputs", "name"}


def test_generated_build_installs_the_workloads_generator(tmp_path, monkeypatch):
    """build() wires kernel_generator() into every layer; nothing compiles here."""
    modernbert_config().save_pretrained(tmp_path / "modernbert")
    workload = agentic_encoder.create_case(
        str(tmp_path / "modernbert"), length=64, kernel_cache=str(tmp_path / "kcache")
    )
    assert workload.kernel_cache == str(tmp_path / "kcache")
    generator = workload.kernel_generator()
    assert generator == region_generation.make_generator(
        tmp_path / "kcache", "modernbert_rope_kernel_notes.md"
    )
    assert generator == workload.kernel_generator()
    model = ModernBertModel(modernbert_config()).eval()
    monkeypatch.setattr(workload, "_load", lambda *args: model)

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    # The build reads the input ids after patching the model, right before it
    # compiles: stop there.
    monkeypatch.setattr(workload, "input_ids", stop)
    values = resolve(
        {"kernels.generated": "split-rope", "runtime.cudagraphs": True},
        knobs_by_name(workload.knobs()),
    )
    with pytest.raises(Stop):
        workload.build(values)
    for layer in model.layers:
        hybrid = layer.attn._modernbert_hybrid
        assert hybrid.variant == "split-rope" and hybrid.generator == generator
    details = workload._install_generated(ModernBertModel(modernbert_config()))
    assert details == {
        "kernels_generated": "split-rope",
        "generated_layers": 4,
        "kernel_cache": str(tmp_path / "kcache"),
        "generator_notes": ["modernbert_rope_kernel_notes.md"],
        "generator_fp32_matmul_precision": "highest",
    }
    # PyTorch backends refuse it before loading anything.
    with pytest.raises(ValueError, match="bench.backend='tensorrt'"):
        workload.build({**values, "bench.backend": "eager"})
