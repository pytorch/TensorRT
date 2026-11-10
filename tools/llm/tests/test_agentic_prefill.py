# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""agentic_prefill: the prefill workload's knobs, attention lowerings and metrics.

Run with ``python -m pytest tools/llm/tests/test_agentic_prefill.py``; no GPU
or model download is needed (tiny random models on the CPU). One test checks
PyTorch's flash-attention kernel behind the custom operator on the GPU and is
skipped without one.
"""

import os
import re
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import agentic_prefill  # noqa: E402
from torch_tensorrt.autotune._knobs import (  # noqa: E402
    apply_changes,
    knobs_by_name,
    resolve,
)
from transformers import (  # noqa: E402
    Gemma3ForCausalLM,
    Gemma3TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
)


def gemma3_config(window=8):
    return Gemma3TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=window,
        # Layers 0-2 local, layer 3 global, as in Gemma 3's 5:1 pattern.
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        query_pre_attn_scalar=16,
    )


def qwen3_config():
    return Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )


def reference_attention(q, k, v, window, scale):
    """Plain causal (sliding-window) attention, keys repeated for GQA."""
    k = k.repeat_interleave(q.shape[1] // k.shape[1], 1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], 1)
    mask = agentic_prefill.position_mask(q.shape[2], k.shape[2], window, q.device)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)


def test_logit_positions_are_evenly_spaced_and_end_at_the_last_token():
    positions = agentic_prefill.logit_positions(2048, 64)
    assert positions.tolist()[:3] == [31, 63, 95]
    assert positions[-1].item() == 2047
    assert len(set(positions.tolist())) == 64
    assert agentic_prefill.logit_positions(10, 3).tolist() == [2, 5, 9]
    with pytest.raises(ValueError):
        agentic_prefill.logit_positions(8, 9)


def test_python_docs_prompt_is_long_natural_text():
    text = agentic_prefill.prompt_text("python-docs", 40000)
    assert len(text) == 40000
    assert text == agentic_prefill.prompt_text("python-docs", 40000)
    # Long enough for 8k tokens without repeating a paragraph.
    paragraphs = [p for p in text.split("\n\n") if len(p) > 200]
    assert len(paragraphs) == len(set(paragraphs))
    with pytest.raises(ValueError):
        agentic_prefill.prompt_text("lorem", 10)


def test_layer_windows_follow_the_layer_types():
    assert agentic_prefill.layer_windows(gemma3_config(8)) == [8, 8, 8, None]
    assert agentic_prefill.layer_windows(qwen3_config()) == [None, None]


def test_position_mask_matches_hugging_face_sliding_window_semantics():
    from transformers.masking_utils import (
        sdpa_mask,
        sliding_window_causal_mask_function,
    )

    for window in (None, 3):
        kwargs = (
            {"mask_function": sliding_window_causal_mask_function(window)}
            if window
            else {}
        )
        expected = sdpa_mask(
            batch_size=1,
            q_length=7,
            kv_length=7,
            allow_is_causal_skip=False,
            **kwargs,
        )
        actual = agentic_prefill.position_mask(7, 7, window, torch.device("cpu"))
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("chunk,window", [(4, 8), (8, 8), (2, 8), (4, 4)])
def test_banded_attention_matches_masked_attention(chunk, window):
    torch.manual_seed(0)
    q = torch.randn(1, 4, 32, 16, dtype=torch.float64)
    k = torch.randn(1, 1, 32, 16, dtype=torch.float64)
    v = torch.randn(1, 1, 32, 16, dtype=torch.float64)
    expected = reference_attention(q, k, v, window, 0.25)
    actual = agentic_prefill.banded_attention(q, k, v, window, chunk, 0.25)
    torch.testing.assert_close(actual, expected)


def test_banded_attention_rejects_chunks_that_do_not_divide():
    q = torch.zeros(1, 2, 12, 4)
    with pytest.raises(ValueError, match="chunk"):
        agentic_prefill.banded_attention(q, q[:, :1], q[:, :1], 8, 8, None)


def test_banded_masks_never_leave_a_query_without_keys():
    mask = agentic_prefill.banded_mask(4, 4, 8, torch.device("cpu"))
    assert mask.shape == (4, 1, 4, 12)
    assert mask.any(-1).all()
    # The first chunk sees no padding; each query sees itself and at most
    # window - 1 earlier positions.
    assert mask[0, 0, 0].tolist() == [False] * 8 + [True, False, False, False]
    assert mask.sum(-1).max().item() == 8


@pytest.mark.parametrize(
    "masks,global_mode,local_mode",
    [
        ("positions", "mask", "mask"),
        ("positions", "causal", "mask"),
        ("hf", "causal", "banded-4"),
        ("hf", "torch-flash", "mask"),
        ("positions", "causal", "torch-flash"),
        ("positions", "causal", "banded-8"),
    ],
)
def test_attention_lowerings_reproduce_the_hugging_face_model(
    masks, global_mode, local_mode
):
    torch.manual_seed(0)
    config = gemma3_config(8)
    config._attn_implementation = "sdpa"
    model = Gemma3ForCausalLM(config).eval().double()
    ids = torch.randint(0, 128, (1, 32))
    positions = agentic_prefill.logit_positions(32, 4)
    with torch.no_grad():
        expected = model(ids, logits_to_keep=positions).logits
        name = agentic_prefill.attention_implementation(masks, global_mode, local_mode)
        model.set_attn_implementation(name)
        actual = model(ids, logits_to_keep=positions).logits
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("window", [0, 5])
def test_flash_attention_op_matches_its_reference_on_the_cpu(window):
    torch.manual_seed(0)
    q = torch.randn(1, 4, 12, 8)
    k = torch.randn(1, 2, 12, 8)
    v = torch.randn(1, 2, 12, 8)
    op = agentic_prefill._flash_attention_op()
    out = op(q, k, v, 0.5, window)
    assert out.shape == (1, 12, 4, 8)
    expected = reference_attention(q, k, v, window or None, 0.5).transpose(1, 2)
    torch.testing.assert_close(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
@pytest.mark.parametrize("window", [0, 128])
def test_flash_attention_op_on_the_gpu_uses_exactly_the_window(window):
    # PyTorch's flash kernel counts window_size_left keys before the query;
    # Hugging Face's window counts the query too. An off-by-one mapping is
    # invisible end to end in BF16, so check the operator itself: it must
    # match the window-W reference far better than the W - 1 and W + 1 ones.
    torch.manual_seed(0)
    q = torch.randn(1, 4, 1024, 256, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, 1, 1024, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, 1, 1024, 256, device="cuda", dtype=torch.bfloat16)
    out = agentic_prefill._flash_attention_op()(q, k, v, 0.0625, window).float()

    def error(reference_window):
        expected = agentic_prefill.flash_attention_reference(
            q.float(), k.float(), v.float(), 0.0625, reference_window
        )
        return (out - expected).abs().mean().item()

    assert error(window) < 2e-3
    if window:
        assert 5 * error(window) < min(error(window - 1), error(window + 1))
    else:
        # Window 0 is full causal attention, not a window.
        assert 5 * error(0) < error(512)


def test_torch_flash_lowering_compiles_with_fullgraph():
    # The custom operator is registered when the implementation is named,
    # outside tracing, so torch.compile can trace the forward in one graph.
    torch.manual_seed(0)
    config = gemma3_config(8)
    config._attn_implementation = "sdpa"
    model = Gemma3ForCausalLM(config).eval()
    ids = torch.randint(0, 128, (1, 32))
    positions = agentic_prefill.logit_positions(32, 4)
    with torch.no_grad():
        expected = model(ids, logits_to_keep=positions).logits
        model.set_attn_implementation(
            agentic_prefill.attention_implementation(
                "positions", "torch-flash", "torch-flash"
            )
        )
        compiled = torch.compile(
            agentic_prefill._Prefill(model, positions),
            backend="aot_eager",
            fullgraph=True,
            dynamic=False,
        )
        torch.testing.assert_close(compiled(ids), expected)


def test_default_modes_keep_hugging_face_sdpa():
    assert agentic_prefill.attention_implementation("hf", "mask", "mask") == "sdpa"


def make_workload(tmp_path, config, length=64):
    config.save_pretrained(tmp_path / "model")
    return agentic_prefill.PrefillWorkload(str(tmp_path / "model"), length=length)


def test_gemma3_knobs_offer_local_and_global_attention(tmp_path):
    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    knobs = knobs_by_name(workload.knobs())
    assert knobs["workload.masks"].values == ("hf", "positions")
    assert knobs["workload.global_attention"].values == (
        "mask",
        "causal",
        "torch-flash",
    )
    assert knobs["workload.local_attention"].values == (
        "mask",
        "banded-128",
        "banded-512",
        "torch-flash",
    )
    assert knobs["engine.precision"].values == ("bf16", "fp16")
    assert "engine.optimization_level" in knobs
    assert workload.name == "model-prefill-2048"
    values = resolve(workload.baseline(), knobs)
    assert values["workload.masks"] == "hf"
    assert apply_changes({}, {"workload.local_attention": "banded-512"}, knobs) == {
        "workload.local_attention": "banded-512"
    }


def test_build_route_offers_only_fp16_accumulation_off_with_fp16(tmp_path, monkeypatch):
    import torch_tensorrt.dynamo

    # TensorRT reports Global Performance Tuning only with a GPU.
    monkeypatch.setattr(
        torch_tensorrt.dynamo, "is_global_perf_tuning_available", lambda: True
    )
    knobs = knobs_by_name(
        make_workload(tmp_path, gemma3_config(512), length=2048).knobs()
    )
    assert knobs["engine.build_route"].values == ("", "-cask:fp16_acc=off")
    # The route changes FP16 engines only, so BF16 configs cannot ask for it.
    with pytest.raises(ValueError, match="requires engine.precision='fp16'"):
        apply_changes({}, {"engine.build_route": "-cask:fp16_acc=off"}, knobs)
    changes = {"engine.build_route": "-cask:fp16_acc=off", "engine.precision": "fp16"}
    assert apply_changes({}, changes, knobs) == changes


def test_unused_masks_value_builds_the_same_model(tmp_path):
    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    knobs = knobs_by_name(workload.knobs())

    def attention(**modes):
        values = resolve({f"workload.{k}": v for k, v in modes.items()}, knobs)
        return workload.attention(values)

    assert attention() == "sdpa"
    assert attention(masks="positions") != "sdpa"
    # Masks matter while some layer still uses them...
    assert attention(global_attention="causal") != attention(
        global_attention="causal", masks="positions"
    )
    # ...and not once no layer does.
    for local in ("banded-128", "torch-flash"):
        assert attention(global_attention="causal", local_attention=local) == (
            attention(
                global_attention="causal", local_attention=local, masks="positions"
            )
        )
    # Without sliding-window layers only the global mode counts.
    qwen = make_workload(tmp_path / "qwen", qwen3_config(), length=64)
    values = resolve(
        {"workload.global_attention": "causal"}, knobs_by_name(qwen.knobs())
    )
    assert qwen.attention(values) == qwen.attention(
        {**values, "workload.masks": "positions"}
    )


def test_local_attention_needs_a_prompt_longer_than_the_window(tmp_path):
    workload = make_workload(tmp_path, gemma3_config(512), length=512)
    assert "workload.local_attention" not in knobs_by_name(workload.knobs())
    # Chunks must divide the length.
    workload = make_workload(tmp_path, gemma3_config(512), length=1536 + 128)
    assert workload.local_modes() == ("mask", "banded-128", "torch-flash")


def test_models_without_sliding_windows_have_no_local_tool(tmp_path):
    workload = make_workload(tmp_path, qwen3_config(), length=64)
    knobs = knobs_by_name(workload.knobs())
    assert "workload.local_attention" not in knobs
    assert "workload.global_attention" in knobs


def test_compare_reports_agreement_and_divergence(tmp_path):
    workload = make_workload(tmp_path, qwen3_config(), length=64)
    torch.manual_seed(0)
    reference = torch.randn(4, 32)
    same = workload.compare(reference.clone(), reference)
    assert same["top1_agreement"] == 1.0
    assert same["mean_kl"] == pytest.approx(0.0, abs=1e-12)
    worse = workload.compare(reference.flip(-1), reference)
    assert worse["top1_agreement"] < 1.0 and worse["mean_kl"] > 0.1
    broken = workload.compare(torch.full_like(reference, float("nan")), reference)
    assert broken["mean_kl"] == float("inf")
    assert workload.accuracy_defaults() == {
        "top1_agreement": (">=", 0.9),
        "mean_kl": ("<=", 0.005),
        "max_kl": ("<=", 0.1),
    }


def test_an_error_at_one_position_fails_the_largest_kl_limit(tmp_path):
    workload = make_workload(tmp_path, qwen3_config(), length=64)
    torch.manual_seed(0)
    reference = torch.randn(64, 32) * 4
    wrong = reference.clone()
    # Only the last (time-to-first-token) row is wrong, by a KL of about 0.2:
    # the mean over 64 rows hides up to 64 x 0.005 = 0.32.
    wrong[-1] += 0.7 * torch.randn(32)
    metrics = workload.compare(wrong, reference)
    assert 0.1 < 64 * metrics["mean_kl"] < 0.32
    limits = workload.accuracy_defaults()
    # The mean over 64 positions and the top-1 agreement hide it...
    assert metrics["top1_agreement"] >= limits["top1_agreement"][1]
    assert metrics["mean_kl"] <= limits["mean_kl"][1]
    # ...the largest KL does not.
    assert metrics["max_kl"] > limits["max_kl"][1]


def bench_values(**bench):
    return {f"bench.{key}": value for key, value in bench.items()}


def test_bench_settings_default_to_tensorrt():
    settings = agentic_prefill.bench_settings({"engine.precision": "bf16"})
    assert settings == agentic_prefill.BenchSettings()
    assert not settings.cudagraphs


def test_bench_settings_collect_the_torch_backends():
    settings = agentic_prefill.bench_settings(
        {
            "engine.precision": "fp16",
            **bench_values(
                backend="inductor",
                mode="max-autotune",
                attn="flex_attention",
                inductor={"coordinate_descent_tuning": True},
                fp16_accumulation=True,
            ),
        }
    )
    assert settings.backend == "inductor" and settings.cudagraphs
    assert settings.inductor == {"coordinate_descent_tuning": True}
    assert settings.attn == "flex_attention" and settings.fp16_accumulation
    no_graphs = agentic_prefill.bench_settings(
        bench_values(backend="inductor", mode="max-autotune-no-cudagraphs")
    )
    assert not no_graphs.cudagraphs
    eager = agentic_prefill.bench_settings(
        bench_values(backend="eager", mode="default")
    )
    assert eager.backend == "eager" and not eager.cudagraphs


@pytest.mark.parametrize(
    "values,match",
    [
        (bench_values(backend="onnx"), "bench.backend"),
        (bench_values(backend="inductor", mode="fast"), "bench.mode"),
        (bench_values(attention="sdpa"), "Unknown bench values"),
        (bench_values(backend="inductor", inductor=["x"]), "mapping"),
        (bench_values(attn="flex_attention"), "do not apply"),
        (bench_values(backend="eager", mode="max-autotune"), "do not apply"),
        (bench_values(backend="eager", inductor={"a": 1}), "do not apply"),
        (bench_values(backend="eager", fp16_accumulation=True), "fp16"),
    ],
)
def test_bench_settings_reject_what_they_would_ignore(values, match):
    with pytest.raises(ValueError, match=match):
        agentic_prefill.bench_settings({"engine.precision": "bf16", **values})


def test_flex_block_masks_are_patched_for_inductor_only():
    import torch._inductor.config as inductor_config
    import transformers.masking_utils as masking_utils

    original = masking_utils.create_block_mask
    matmul = torch.backends.cuda.matmul
    fp16_accumulation = matmul.allow_fp16_accumulation
    tuning = inductor_config.coordinate_descent_tuning

    eager = agentic_prefill.BenchSettings(backend="eager", fp16_accumulation=True)
    with agentic_prefill.torch_backend_context(eager, "flex_attention"):
        # Eager keeps Hugging Face's compiled block-mask builder.
        assert masking_utils.create_block_mask is original
        assert matmul.allow_fp16_accumulation
    assert matmul.allow_fp16_accumulation == fp16_accumulation

    inductor = agentic_prefill.BenchSettings(
        backend="inductor", inductor={"coordinate_descent_tuning": not tuning}
    )
    stack = agentic_prefill.torch_backend_context(inductor, "flex_attention")
    assert masking_utils.create_block_mask is not original
    assert inductor_config.coordinate_descent_tuning == (not tuning)
    stack.close()
    assert masking_utils.create_block_mask is original
    assert inductor_config.coordinate_descent_tuning == tuning

    with agentic_prefill.torch_backend_context(inductor, "sdpa"):
        assert masking_utils.create_block_mask is original
    unknown = agentic_prefill.BenchSettings(backend="inductor", inductor={"nope": 1})
    with pytest.raises(ValueError, match="nope"):
        agentic_prefill.torch_backend_context(unknown, "flex_attention")
    assert masking_utils.create_block_mask is original


def test_prefill_flops_count_linear_head_and_unmasked_attention():
    config = gemma3_config(8)
    model = Gemma3ForCausalLM(config)
    flops = agentic_prefill.prefill_flops(model, 32, 4)
    linear = 4 * (64 * 64 + 2 * 64 * 16 + 64 * 64 + 3 * 64 * 128)
    assert flops["linear_gflop"] == pytest.approx(2 * 32 * linear / 1e9)
    assert flops["lm_head_gflop"] == pytest.approx(2 * 4 * 128 * 64 / 1e9)
    # Three windowed layers (36 + 24 * 8 pairs) and one causal (528 pairs).
    pairs = 3 * (36 + 24 * 8) + 32 * 33 // 2
    assert flops["attention_gflop"] == pytest.approx(4 * 4 * 16 * pairs / 1e9)
    dense = 4 * 4 * 16 * 32 * 32 * 4
    assert flops["attention_gflop_dense"] == pytest.approx(dense / 1e9)


def test_qwen3_prefill_wrapper_returns_the_kept_logits():
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(qwen3_config()).eval()
    positions = agentic_prefill.logit_positions(16, 4)
    ids = torch.randint(0, 128, (1, 16))
    with torch.no_grad():
        kept = agentic_prefill._Prefill(model, positions)(ids)
        full = model(ids).logits
    torch.testing.assert_close(kept, full[:, positions])


# ---------------------------------------------------------- kernels.inductor
ALL_INDUCTOR_KERNELS = (
    "none",
    "attention-fa2-global",
    "attention-fa2-all",
    "attention-inductor-global",
    "attention-inductor",
    "attention-rope-inductor-global",
    "attention-rope-inductor",
    "rope-plugin",
)


def test_inductor_knob_is_off_by_default_and_needs_cuda_graphs(tmp_path):
    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    knobs = knobs_by_name(workload.knobs())
    knob = knobs["kernels.inductor"]
    assert knob.values == ALL_INDUCTOR_KERNELS
    assert knob.default == "none"
    assert dict(knob.requires) == {"runtime.cudagraphs": True}
    assert resolve({}, knobs)["kernels.inductor"] == "none"
    # Boundaries of the tiny model: 1 global layer (2), all 4 layers (8).
    assert "(2 for the global layers, 8 for all)" in knob.description
    with pytest.raises(ValueError, match="requires runtime.cudagraphs=True"):
        apply_changes({}, {"kernels.inductor": "attention-fa2-all"}, knobs)
    changes = {"kernels.inductor": "attention-inductor", "runtime.cudagraphs": True}
    assert apply_changes({}, changes, knobs) == changes


def test_inductor_knob_offers_only_values_that_fit_the_model(tmp_path):
    # Without sliding windows (Qwen 3) or with windows covering the prompt,
    # the global-only values would equal the all-layer ones.
    qwen = make_workload(tmp_path / "qwen", qwen3_config(), length=64)
    expected = ("none", "attention-fa2-all", "attention-inductor")
    rope = ("attention-rope-inductor", "rope-plugin")
    assert qwen.inductor_kernels() == expected + rope
    short = make_workload(tmp_path / "short", gemma3_config(512), length=512)
    assert short.inductor_kernels() == expected + rope
    # The rope regions and plugins rewrite only Gemma 3 / Qwen 3 attention.
    from transformers import LlamaConfig

    llama = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    assert make_workload(tmp_path / "llama", llama).inductor_kernels() == expected


def test_inductor_placements_override_the_moved_layers_only(tmp_path):
    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    knobs = knobs_by_name(workload.knobs())
    session = {
        "engine.precision": "fp16",
        "workload.global_attention": "causal",
        "workload.masks": "positions",
        "runtime.cudagraphs": True,
    }

    def modes(kernels, **changes):
        values = resolve({**session, **changes, "kernels.inductor": kernels}, knobs)
        return workload.attention_modes(values)

    # Existing configs keep their implementation.
    values = resolve({**session}, knobs)
    assert workload.attention_modes(values) == ("positions", "causal", "mask")
    assert workload.attention(resolve({}, knobs)) == "sdpa"
    assert modes("none") == ("positions", "causal", "mask")
    assert modes("attention-fa2-global") == ("positions", "torch-flash", "mask")
    assert modes("attention-fa2-all") == ("positions", "torch-flash", "torch-flash")
    assert modes("attention-inductor-global") == ("positions", "inductor", "mask")
    assert modes("attention-inductor") == ("positions", "inductor", "inductor")
    # Rope regions call torch-flash inside the region the patched forward opens.
    assert modes("attention-rope-inductor-global") == (
        "positions",
        "torch-flash",
        "mask",
    )
    assert modes("attention-rope-inductor") == (
        "positions",
        "torch-flash",
        "torch-flash",
    )
    # The rope plugins keep the attention in TensorRT under the config's modes.
    assert modes("rope-plugin") == ("positions", "causal", "mask")
    assert modes("rope-plugin", **{"workload.local_attention": "banded-128"}) == (
        "positions",
        "causal",
        "banded-128",
    )
    # The global-only values keep the config's local layers, masks included...
    assert modes("attention-fa2-global", **{"workload.masks": "hf"})[0] == "hf"
    banded = {"workload.local_attention": "banded-128"}
    assert modes("attention-inductor-global", **banded) == (
        "positions",
        "inductor",
        "banded-128",
    )
    # ...and with every layer moved, no mask is left to choose.
    assert modes("attention-inductor", **{"workload.masks": "hf"})[0] == "positions"
    # The FA2 values build exactly what the workload's torch-flash modes build
    # (aliases of those configs, as the description says).
    for kernels, attention_modes in (
        ("attention-fa2-global", {"workload.global_attention": "torch-flash"}),
        (
            "attention-fa2-all",
            {
                "workload.global_attention": "torch-flash",
                "workload.local_attention": "torch-flash",
            },
        ),
    ):
        torch_flash = resolve({**session, **attention_modes}, knobs)
        fa2 = resolve({**session, "kernels.inductor": kernels}, knobs)
        assert workload.attention(fa2) == workload.attention(torch_flash)
    # The attention knobs are ignored for the layers a value moves.
    for kernels in ("attention-fa2-global", "attention-inductor-global"):
        assert modes(kernels, **{"workload.global_attention": "mask"}) == modes(kernels)
    for kernels in ("attention-fa2-all", "attention-inductor"):
        assert modes(kernels, **banded) == modes(kernels)
    description = knobs["kernels.inductor"].description
    assert "aliases of those configs" in description
    assert "cannot be saved" in description


def test_inductor_placements_name_their_layers():
    windows = agentic_prefill.layer_windows(gemma3_config(8))
    placement = agentic_prefill.inductor_placement
    assert placement("none").moved(windows) == []
    assert not placement("none").regions
    assert placement("attention-fa2-global").moved(windows) == [3]
    assert placement("attention-inductor").moved(windows) == [0, 1, 2, 3]
    assert not placement("attention-fa2-all").regions
    assert placement("attention-inductor-global").regions
    assert placement("attention-rope-inductor").regions
    assert placement("attention-rope-inductor").moves_attention
    plugin = placement("rope-plugin")
    assert plugin.moved(windows) == [0, 1, 2, 3]
    assert plugin.regions and not plugin.moves_attention
    assert not placement("none").moves_attention
    with pytest.raises(ValueError, match="kernels.inductor"):
        placement("attention-tensorrt")


def test_inductor_attention_mode_reproduces_the_hugging_face_model():
    # Outside strict Torch-TensorRT compilation the region is ordinary code.
    torch.manual_seed(0)
    config = gemma3_config(8)
    config._attn_implementation = "sdpa"
    model = Gemma3ForCausalLM(config).eval().double()
    ids = torch.randint(0, 128, (1, 32))
    positions = agentic_prefill.logit_positions(32, 4)
    with torch.no_grad():
        expected = model(ids, logits_to_keep=positions).logits
        for local in ("inductor", "mask"):
            name = agentic_prefill.attention_implementation(
                "positions", "inductor", local
            )
            model.set_attn_implementation(name)
            actual = model(ids, logits_to_keep=positions).logits
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("family", ["gemma3", "qwen3"])
def test_attention_regions_reproduce_the_hugging_face_model(family):
    torch.manual_seed(0)
    if family == "gemma3":
        config, model_class = gemma3_config(8), Gemma3ForCausalLM
    else:
        config, model_class = qwen3_config(), Qwen3ForCausalLM
    config._attn_implementation = "sdpa"
    model = model_class(config).eval().double()
    ids = torch.randint(0, 128, (1, 32))
    positions = agentic_prefill.logit_positions(32, 4)
    with torch.no_grad():
        expected = model(ids, logits_to_keep=positions).logits
    name = agentic_prefill.attention_implementation(
        "positions", "torch-flash", "torch-flash"
    )
    model.set_attn_implementation(name)
    layers = range(config.num_hidden_layers)
    assert agentic_prefill.install_attention_regions(model, layers) == len(layers)
    # Installed once: a second call changes nothing.
    assert agentic_prefill.install_attention_regions(model, layers) == 0
    attention = model.model.layers[0].self_attn
    # The projections view the fused weight the region's GEMM uses.
    q, k, v = attention.q_proj.weight, attention.k_proj.weight, attention.v_proj.weight
    assert attention.qkv_sizes == (q.shape[0], k.shape[0], v.shape[0])
    assert q.untyped_storage().data_ptr() == attention.qkv_weight.data_ptr()
    torch.testing.assert_close(attention.qkv_weight, torch.cat([q, k, v]))
    with torch.no_grad():
        actual = agentic_prefill._Prefill(model, positions)(ids)
    torch.testing.assert_close(actual, expected)


def test_attention_regions_need_q_and_k_norms():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    with pytest.raises(ValueError, match="q/k-norm"):
        agentic_prefill.install_attention_regions(LlamaForCausalLM(config), [0])


def _captured_regions(model, ids, positions):
    """The region records strict Torch-TensorRT capture finds (no compilation)."""
    from torch_tensorrt.dynamo.region_compilation import (
        normalize_region_scopes,
        region_hop_records,
    )
    from torch_tensorrt.region._session import RegionCompilationSession

    wrapper = agentic_prefill._Prefill(model, positions).eval()
    with RegionCompilationSession(strict=True), torch.no_grad():
        program = torch.export.export(wrapper, (ids,), strict=True)
        program = normalize_region_scopes(program)
        return region_hop_records(program.graph_module), program


@pytest.mark.parametrize(
    "kernels,regions",
    [
        ("attention-inductor-global", ["attention.3"]),
        ("attention-inductor", [f"attention.{i}" for i in range(4)]),
        ("attention-rope-inductor-global", ["attention.3"]),
        ("attention-rope-inductor", [f"attention.{i}" for i in range(4)]),
    ],
)
def test_strict_capture_finds_one_inductor_region_per_moved_layer(
    tmp_path, kernels, regions
):
    workload = make_workload(tmp_path, gemma3_config(8), length=64)
    knobs = knobs_by_name(workload.knobs())
    values = resolve(
        {
            "workload.masks": "positions",
            "workload.global_attention": "causal",
            "kernels.inductor": kernels,
            "runtime.cudagraphs": True,
        },
        knobs,
    )
    torch.manual_seed(0)
    config = gemma3_config(8)
    config._attn_implementation = workload.attention(values)
    model = Gemma3ForCausalLM(config).eval()
    placement = agentic_prefill.inductor_placement(kernels)
    if placement.route == "region-rope":
        moved = placement.moved(agentic_prefill.layer_windows(config))
        agentic_prefill.install_attention_regions(model, moved)
    ids = torch.randint(0, 128, (1, 32))
    records, program = _captured_regions(
        model, ids, agentic_prefill.logit_positions(32, 4)
    )
    assert sorted(record.name for record in records) == regions
    assert {record.backend for record in records} == {"inductor"}
    targets = {
        str(node.target)
        for graph in program.graph_module.modules()
        if isinstance(graph, torch.fx.GraphModule)
        for node in graph.graph.nodes
    }
    assert "agentic_prefill.flash_attention.default" in targets


def test_strict_export_traces_are_released():
    import gc
    import weakref

    torch.manual_seed(0)
    config = gemma3_config(8)
    config._attn_implementation = "sdpa"
    ids = torch.randint(0, 128, (1, 32))
    positions = agentic_prefill.logit_positions(32, 4)
    for _ in range(2):  # a second strict export still works after a release
        model = Gemma3ForCausalLM(config).eval()
        weight = weakref.ref(model.model.embed_tokens.weight)
        wrapper = agentic_prefill._Prefill(model, positions).eval()
        with torch.no_grad():
            program = torch.export.export(wrapper, (ids,), strict=True)
        del program, wrapper, model
        agentic_prefill.release_strict_export_traces()
        gc.collect()
        # Nothing else keeps the exported model's weights alive.
        assert weight() is None
    assert agentic_prefill.release_strict_export_traces() == 0


def test_inductor_values_need_cuda_graphs_in_a_tensorrt_build(tmp_path):
    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    values = resolve({}, knobs_by_name(workload.knobs()))
    # A TensorRT build checks the CUDA-graph requirement before loading.
    with pytest.raises(ValueError, match="runtime.cudagraphs"):
        workload.build({**values, "kernels.inductor": "attention-inductor"})


@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_inductor_values_are_refused_by_pytorch_backends(tmp_path, backend):
    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    values = resolve({}, knobs_by_name(workload.knobs()))
    # PyTorch backends have no engines to place attention around; they refuse
    # before loading the model.
    with pytest.raises(ValueError, match=f"bench.backend '{backend}' cannot honor"):
        workload.build(
            {
                **values,
                "kernels.inductor": "attention-fa2-all",
                "runtime.cudagraphs": True,
                "bench.backend": backend,
            }
        )


def _tensorrt_candidate(module, **structure):
    return agentic_prefill._TensorRTPrefill(
        module,
        torch.zeros(1, 4, dtype=torch.long),
        cudagraphs=False,
        pre_allocated_outputs=False,
        structure=structure,
    )


def test_region_builds_report_that_they_cannot_be_saved(tmp_path):
    # The exporter itself rejects a program with an Inductor region record
    # (before writing anything); the candidate names the value in the error.
    from torch_tensorrt import region
    from torch_tensorrt.dynamo.region_compilation._types import (
        RegionRecord,
        attach_region_records,
    )

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1

    program = torch.fx.symbolic_trace(AddOne())
    record = RegionRecord(
        id="attention.3#0", name="attention.3", child_target="x", backend="inductor"
    )
    attach_region_records(program, [record])
    candidate = _tensorrt_candidate(
        program, kernels_inductor="attention-inductor-global", inductor_regions=1
    )
    with pytest.raises(NotImplementedError, match="cannot be saved") as caught:
        candidate.export(tmp_path)
    assert "attention-inductor-global" in str(caught.value)
    assert isinstance(caught.value.__cause__, region.RegionError)
    assert list(tmp_path.iterdir()) == []
    # Without Inductor regions the exporter's own error is not rewritten.
    with pytest.raises(region.RegionError):
        _tensorrt_candidate(program).export(tmp_path)


def test_builds_without_regions_save_their_program(tmp_path, monkeypatch):
    import torch_tensorrt

    saved = []
    monkeypatch.setattr(
        torch_tensorrt,
        "save",
        lambda module, path, **kwargs: saved.append((module, path, kwargs)),
    )
    module = torch.nn.Identity()
    candidate = _tensorrt_candidate(module, kernels_inductor="attention-fa2-all")
    assert candidate.export(tmp_path) == ["prefill.ep"]
    assert saved == [(module, str(tmp_path / "prefill.ep"), {"retrace": False})]


# ------------------------------------------------------------- rope-plugin
def _fake_rope_kernels(monkeypatch, key):
    """A registry entry for the plugin forward without a GPU (CPU tests).

    Outside strict Torch-TensorRT compilation the fused_kernel scopes run
    their PyTorch body; capture only resolves the kernel (a ``@triton.jit``
    function), and nothing here launches or compiles it.
    """
    triton = pytest.importorskip("triton")

    @triton.jit
    def fake_norm_rope(QKV, WEIGHT, COS, SIN, OUT):
        pass

    hosted = agentic_prefill.HostedKernel(
        kernel=fake_norm_rope,
        grid=(1, 1, 1),
        num_warps=1,
        num_stages=1,
        constexprs={},
        name="fake",
        sha256="",
        source_path="",
        matches_inductor="",
    )
    monkeypatch.setitem(agentic_prefill._ROPE_KERNELS, key, {"q": hosted, "k": hosted})


@pytest.mark.parametrize("family", ["gemma3", "qwen3"])
def test_rope_plugin_forward_reproduces_the_hugging_face_model(monkeypatch, family):
    torch.manual_seed(0)
    if family == "gemma3":
        config, model_class = gemma3_config(8), Gemma3ForCausalLM
    else:
        config, model_class = qwen3_config(), Qwen3ForCausalLM
    config._attn_implementation = "sdpa"
    model = model_class(config).eval().double()
    ids = torch.randint(0, 128, (1, 32))
    positions = agentic_prefill.logit_positions(32, 4)
    with torch.no_grad():
        expected = model(ids, logits_to_keep=positions).logits
    # The config's attention modes stay (here: causal SDPA and position masks).
    model.set_attn_implementation(
        agentic_prefill.attention_implementation("positions", "causal", "mask")
    )
    _fake_rope_kernels(monkeypatch, ("test",))
    layers = range(config.num_hidden_layers)
    changed = agentic_prefill.install_attention_regions(model, layers, plugin=("test",))
    assert changed == len(layers)
    attention = model.model.layers[0].self_attn
    assert type(attention).__name__.startswith("Plugin")
    assert attention.rope_plugin == ("test",)
    with torch.no_grad():
        actual = agentic_prefill._Prefill(model, positions)(ids)
    torch.testing.assert_close(actual, expected)


def test_norm_rope_matches_the_attention_forward():
    # The plugin body (and the function Inductor compiles) equals Hugging
    # Face's q/k path on the q and k columns of the fused GEMM output.
    from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb

    torch.manual_seed(0)
    config = gemma3_config(8)
    model = Gemma3ForCausalLM(config).eval().double()
    attention = model.model.layers[3].self_attn
    hidden = torch.randn(1, 16, 64, dtype=torch.float64)
    angles = torch.rand(1, 16, 16, dtype=torch.float64)
    cos, sin = angles.cos(), angles.sin()
    shape = (1, 16, -1, 16)
    q = attention.q_norm(attention.q_proj(hidden).view(shape).transpose(1, 2))
    k = attention.k_norm(attention.k_proj(hidden).view(shape).transpose(1, 2))
    q, k = apply_rotary_pos_emb(q, k, cos, sin)
    qkv = torch.cat(
        [attention.q_proj(hidden), attention.k_proj(hidden), attention.v_proj(hidden)],
        -1,
    )
    sizes = (64, 16, 16)
    roles = (("q", attention.q_norm, q), ("k", attention.k_norm, k))
    for role, norm, expected in roles:
        actual = agentic_prefill._norm_rope(
            norm, apply_rotary_pos_emb, sizes, 16, role, qkv, cos, sin
        )
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected)


def test_strict_capture_finds_two_rope_plugins_per_layer(tmp_path, monkeypatch):
    workload = make_workload(tmp_path, gemma3_config(8), length=64)
    knobs = knobs_by_name(workload.knobs())
    values = resolve(
        {
            "workload.masks": "positions",
            "workload.global_attention": "causal",
            "kernels.inductor": "rope-plugin",
            "runtime.cudagraphs": True,
        },
        knobs,
    )
    torch.manual_seed(0)
    config = gemma3_config(8)
    config._attn_implementation = workload.attention(values)
    model = Gemma3ForCausalLM(config).eval()
    _fake_rope_kernels(monkeypatch, ("capture",))
    agentic_prefill.install_attention_regions(model, range(4), plugin=("capture",))
    ids = torch.randint(0, 128, (1, 32))
    records, program = _captured_regions(
        model, ids, agentic_prefill.logit_positions(32, 4)
    )
    names = sorted(record.name for record in records)
    assert names == sorted(
        f"{role}_norm_rope.{layer}" for role in ("q", "k") for layer in range(4)
    )
    assert {record.route for record in records} == {"plugin"}
    # The attention stays in the graph TensorRT compiles.
    targets = {
        str(node.target)
        for graph in program.graph_module.modules()
        if isinstance(graph, torch.fx.GraphModule)
        for node in graph.graph.nodes
    }
    assert "agentic_prefill.flash_attention.default" not in targets
    assert any("scaled_dot_product" in target for target in targets)


INDUCTOR_CODE = """
triton_per_fused_mean_0 = async_compile.triton('triton_per_fused_mean_0', '''
import triton

@triton.jit
def triton_per_fused_mean_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8192
''', device_str='cuda')

def call(args):
    triton_per_fused_mean_0.run(arg0_1, arg2_1, arg1_1, arg3_1, buf0, 8192, 256, stream=stream0)
"""


def test_inductor_kernel_launch_becomes_the_plugin_call():
    name, source, params = agentic_prefill.parse_inductor_kernel(INDUCTOR_CODE)
    assert name == "triton_per_fused_mean_0"
    assert "xnumel = 8192" in source
    assert params == [
        "in_ptr0",
        "in_ptr1",
        "in_ptr2",
        "in_ptr3",
        "out_ptr0",
        "xnumel",
        "r0_numel",
        "XBLOCK",
    ]
    qkv, weight, cos, sin, out = (torch.zeros(4) for _ in range(5))
    roles = {
        qkv.data_ptr(): "QKV",
        weight.data_ptr(): "WEIGHT",
        cos.data_ptr(): "COS",
        sin.data_ptr(): "SIN",
    }
    # Inductor's order of first use, not the wrapper's, decides the mapping.
    launch = (qkv, cos, weight, sin, out, 8192, 256)
    arguments = agentic_prefill.plugin_call_arguments(
        params, launch, roles, out.data_ptr(), {"XBLOCK": 8}
    )
    assert arguments == [
        "_aligned(QKV)",
        "_aligned(COS)",
        "_aligned(WEIGHT)",
        "_aligned(SIN)",
        "_aligned(OUT)",
        "8192",
        "256",
        "8",
    ]
    stranger = torch.zeros(4)
    with pytest.raises(RuntimeError, match="none of the plugin's inputs"):
        agentic_prefill.plugin_call_arguments(
            params, (stranger,) + launch[1:], roles, out.data_ptr(), {"XBLOCK": 8}
        )
    with pytest.raises(RuntimeError, match="no value for XBLOCK"):
        agentic_prefill.plugin_call_arguments(params, launch, roles, out.data_ptr(), {})
    two = INDUCTOR_CODE + "    triton_per_fused_mean_0.run(buf0, stream=stream0)\n"
    with pytest.raises(RuntimeError, match="launches one Triton kernel"):
        agentic_prefill.parse_inductor_kernel(two)


# ------------------------------------------------------ kernels.generated
def test_generated_knob_is_prepared_and_needs_cuda_graphs(tmp_path):
    from torch_tensorrt.autotune._knobs import with_requirements

    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    knobs = knobs_by_name(workload.knobs())
    knob = knobs["kernels.generated"]
    assert knob.values == ("none", "qk-norm-rope") == agentic_prefill.GENERATED_KERNELS
    assert knob.default == "none"
    # rope-plugin's requirement, and no kernels.inductor value: they rewrite
    # the same attention.
    assert dict(knob.requires) == {
        "runtime.cudagraphs": True,
        "kernels.inductor": "none",
    }
    # A session generates the kernels once, before its search.
    assert knob.prepare == ("qk-norm-rope",)
    assert not re.search(r"\d\s*(ms|us|%|x faster)", knob.description)
    assert "two plugins per layer (8 in all)" in knob.description
    assert resolve({}, knobs)["kernels.generated"] == "none"
    with pytest.raises(ValueError, match="requires runtime.cudagraphs=True"):
        apply_changes({}, {"kernels.generated": "qk-norm-rope"}, knobs)
    hosted = {"kernels.inductor": "rope-plugin", "runtime.cudagraphs": True}
    with pytest.raises(ValueError, match="requires kernels.inductor='none'"):
        apply_changes(hosted, {"kernels.generated": "qk-norm-rope"}, knobs)
    assert with_requirements(hosted, {"kernels.generated": "qk-norm-rope"}, knobs) == {
        "kernels.generated": "qk-norm-rope",
        "kernels.inductor": "none",
    }
    values = resolve(
        {"kernels.generated": "qk-norm-rope", "runtime.cudagraphs": True}, knobs
    )
    assert workload.generated_kernels(values) == "qk-norm-rope"
    # Offered for the models whose attention the regions rewrite.
    qwen3 = make_workload(tmp_path / "qwen3", qwen3_config(), length=64)
    assert "kernels.generated" in {knob.name for knob in qwen3.knobs()}
    from transformers import LlamaConfig

    llama = make_workload(
        tmp_path / "llama",
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
        ),
        length=64,
    )
    assert "kernels.generated" not in {knob.name for knob in llama.knobs()}
    with pytest.raises(ValueError, match="needs a model whose attention"):
        llama.generated_kernels(
            {"kernels.generated": "qk-norm-rope", "runtime.cudagraphs": True}
        )


def test_generated_values_are_checked_before_loading(tmp_path):
    workload = make_workload(tmp_path, gemma3_config(512), length=2048)
    values = resolve({}, knobs_by_name(workload.knobs()))
    generated = {**values, "kernels.generated": "qk-norm-rope"}
    with pytest.raises(ValueError, match="needs runtime.cudagraphs=True"):
        workload.build(generated)
    with pytest.raises(ValueError, match="needs kernels.inductor='none'"):
        workload.build(
            {
                **generated,
                "runtime.cudagraphs": True,
                "kernels.inductor": "attention-fa2-global",
            }
        )
    with pytest.raises(ValueError, match="must be one of"):
        workload.build({**values, "kernels.generated": "mlp"})
    with pytest.raises(ValueError, match="bench.backend 'eager' cannot honor"):
        workload.build(
            {**generated, "runtime.cudagraphs": True, "bench.backend": "eager"}
        )


@pytest.mark.parametrize("family", ["gemma3", "qwen3"])
def test_generated_rope_forward_reproduces_the_hugging_face_model(tmp_path, family):
    import region_generation

    torch.manual_seed(0)
    if family == "gemma3":
        config, model_class = gemma3_config(8), Gemma3ForCausalLM
    else:
        config, model_class = qwen3_config(), Qwen3ForCausalLM
    config._attn_implementation = "sdpa"
    model = model_class(config).eval().double()
    ids = torch.randint(0, 128, (1, 32))
    positions = agentic_prefill.logit_positions(32, 4)
    with torch.no_grad():
        expected = model(ids, logits_to_keep=positions).logits
    model.set_attn_implementation(
        agentic_prefill.attention_implementation("positions", "causal", "mask")
    )
    generator = region_generation.make_generator(tmp_path)
    layers = range(config.num_hidden_layers)
    changed = agentic_prefill.install_attention_regions(
        model, layers, generator=generator
    )
    assert changed == len(layers)
    attention = model.model.layers[0].self_attn
    assert type(attention).__name__.startswith("Plugin")
    assert attention.rope_generator is generator
    assert not hasattr(attention, "rope_plugin")
    with torch.no_grad():
        actual = agentic_prefill._Prefill(model, positions)(ids)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="not both"):
        agentic_prefill.install_attention_regions(
            model_class(config), [0], plugin=("test",), generator=generator
        )


def test_strict_capture_finds_two_generated_rope_regions_per_layer(
    tmp_path, monkeypatch
):
    """The generated regions are rope-plugin's: the same names and inputs.

    Capture only (no compilation): nothing generates a kernel, and generation
    is switched off besides.
    """
    import region_generation
    from torch_tensorrt.dynamo.region_compilation import (
        normalize_region_scopes,
        region_hop_records,
    )
    from torch_tensorrt.region._session import RegionCompilationSession

    monkeypatch.setenv("TORCH_TENSORRT_REGION_GENERATION", "0")
    workload = make_workload(tmp_path, gemma3_config(8), length=64)
    values = resolve(
        {
            "workload.masks": "positions",
            "workload.global_attention": "causal",
            "kernels.generated": "qk-norm-rope",
            "runtime.cudagraphs": True,
        },
        knobs_by_name(workload.knobs()),
    )
    torch.manual_seed(0)
    config = gemma3_config(8)
    config._attn_implementation = workload.attention(values)
    model = Gemma3ForCausalLM(config).eval()
    generator = region_generation.make_generator(tmp_path / "kcache")
    agentic_prefill.install_attention_regions(model, range(4), generator=generator)
    ids = torch.randint(0, 128, (1, 32))
    wrapper = agentic_prefill._Prefill(
        model, agentic_prefill.logit_positions(32, 4)
    ).eval()
    with RegionCompilationSession(strict=True) as session, torch.no_grad():
        program = torch.export.export(wrapper, (ids,), strict=True)
        program = normalize_region_scopes(program)
        records = region_hop_records(program.graph_module)
        registrations = [session.plugins[record.plugin_key] for record in records]
    names = sorted(record.name for record in records)
    assert names == sorted(
        f"{role}_norm_rope.{layer}" for role in ("q", "k") for layer in range(4)
    )
    assert {record.route for record in records} == {"plugin"}
    assert all(registration == generator for registration in registrations)


def test_generated_build_installs_the_workloads_generator(tmp_path, monkeypatch):
    """build() wires kernel_generator() into every attention; nothing compiles."""
    import region_generation

    gemma3_config(8).save_pretrained(tmp_path / "model")
    workload = agentic_prefill.create_case(
        str(tmp_path / "model"), length=64, kernel_cache=str(tmp_path / "kcache")
    )
    assert workload.kernel_cache == str(tmp_path / "kcache")
    generator = workload.kernel_generator()
    # The shared settings, without notes.
    assert generator == region_generation.make_generator(tmp_path / "kcache")
    torch.manual_seed(0)
    model = Gemma3ForCausalLM(gemma3_config(8)).eval()
    monkeypatch.setattr(workload, "_load", lambda *args: model)
    monkeypatch.setattr(workload, "_structure", lambda model: {})

    class Stop(Exception):
        pass

    def stop():
        raise Stop

    # The build reads the prompt after patching the model, right before it
    # compiles: stop there.
    monkeypatch.setattr(workload, "tokens", stop)
    values = resolve(
        {"kernels.generated": "qk-norm-rope", "runtime.cudagraphs": True},
        knobs_by_name(workload.knobs()),
    )
    with pytest.raises(Stop):
        workload.build(values)
    for layer in model.model.layers:
        assert layer.self_attn.rope_generator == generator
    details = workload._install_generated(Gemma3ForCausalLM(gemma3_config(8)))
    assert details == {
        "kernels_generated": "qk-norm-rope",
        "generated_regions": 8,
        "kernel_cache": str(tmp_path / "kcache"),
        "generator_notes": [],
        "generator_fp32_matmul_precision": "highest",
    }
