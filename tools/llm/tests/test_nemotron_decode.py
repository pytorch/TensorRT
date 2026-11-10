# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""nemotron_decode: the functional decode step, the FP32 formulation gate, the
prefill module and SSD scans, the FlexAttention mask fix, the knobs and the
estimates, and the kernels.inductor hybrid variants' CPU-side logic.

Run with ``python -m pytest tools/llm/tests/test_nemotron_decode.py``; no GPU
or model download is needed (a tiny random Nemotron-H runs on the CPU). Its
Mamba input projections and convolutions are scaled up so that the SSM states
are of order one and every state update moves the logits (at the default
initialization the states are ~1e-5 and a frozen state hides below 1e-5).
"""

import functools
import inspect
import os
import sys
import types

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import nemotron_decode  # noqa: E402
from torch_tensorrt.autotune import load_module  # noqa: E402
from torch_tensorrt.autotune._code_edits import (  # noqa: E402
    apply_edits,
    editable_blocks,
)
from torch_tensorrt.autotune._knobs import (  # noqa: E402
    apply_changes,
    knobs_by_name,
    resolve,
)
from transformers import NemotronHConfig, NemotronHForCausalLM  # noqa: E402

ISL, OSL = 11, 6
# The edited line of a step that never updates its SSM state.
FROZEN_SSM = {
    "search": "new_ssm = state.to(ssm_state.dtype).reshape(ssm_state.shape)",
    "replace": "new_ssm = ssm_state.clone()",
}


def tiny_config(**overrides):
    settings = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_num_heads=4,
        mamba_head_dim=8,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=8,
        hybrid_override_pattern="M-M*-M",
        mlp_hidden_act="relu2",
    )
    settings.update(overrides)
    return NemotronHConfig(**settings)


def tiny_model(attn="sdpa"):
    torch.manual_seed(0)
    model = NemotronHForCausalLM._from_config(
        tiny_config(), attn_implementation=attn
    ).eval()
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "A_log"):
                module.A_log.uniform_(-1, 1)
                module.D.uniform_(0.5, 1.5)
                module.dt_bias.uniform_(-2, 1)
                module.in_proj.weight.mul_(10)
                module.conv1d.weight.mul_(10)
    return model


@pytest.fixture(scope="module")
def tiny():
    model = tiny_model()
    tokens = torch.randint(
        0, 128, (ISL + OSL,), generator=torch.Generator().manual_seed(1)
    )
    reference = nemotron_decode._hf_decode_logits(model, tokens, ISL, OSL)
    return model, tokens, reference


def mamba_states(cache, kinds):
    """Hugging Face's conv and SSM states of every Mamba layer."""
    layers = [
        cache.layers[i] for i, kind in enumerate(kinds) if kind == "linear_attention"
    ]
    return [layer.conv_states[0] for layer in layers], [
        layer.recurrent_states[0] for layer in layers
    ]


def run_step(model, tokens, inplace, step_class=None):
    """Decode OSL tokens with the decode step run eagerly, threading the states.

    Returns the logits ``[OSL, vocab]`` and each step's state outputs.
    """
    step_class = step_class or nemotron_decode.NemotronDecodeStep
    kinds = nemotron_decode.layer_kinds(model.config)
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)
    states = nemotron_decode.static_states(cache, kinds, inplace=inplace)
    step = step_class(model, ISL + OSL, inplace=inplace)
    logits, history = [], []
    with torch.inference_mode():
        for j in range(OSL):
            position = torch.full((1, 1), ISL + j)
            out = step(tokens[ISL + j].view(1, 1), position, *states)
            logits.append(out[0][0])
            # Eagerly, index_copy returns new tensors: thread every output.
            states = list(out[1:])
            history.append(states)
    return torch.stack(logits), history


def test_hugging_face_decode_matches_its_full_forward(tiny):
    model, tokens, reference = tiny
    with torch.inference_mode():
        full = model(tokens.view(1, -1)).logits[0, ISL : ISL + OSL]
    torch.testing.assert_close(reference, full, atol=1e-5, rtol=1e-5)


def test_tiny_model_states_are_of_order_one_and_move_the_logits(tiny):
    model, tokens, reference = tiny
    kinds = nemotron_decode.layer_kinds(model.config)
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)
    _, ssm = mamba_states(cache, kinds)
    assert min(state.abs().max().item() for state in ssm) > 0.1
    # A step that never updates the SSM state is far outside the tolerance.
    frozen = nemotron_decode.NemotronDecodeStep._mamba

    def no_update(self, mixer, hidden, conv_state, ssm_state, slot):
        out, new_conv, _ = frozen(self, mixer, hidden, conv_state, ssm_state, slot)
        return out, new_conv, ssm_state.clone()

    class FrozenStep(nemotron_decode.NemotronDecodeStep):
        _mamba = no_update

    logits, _ = run_step(model, tokens, False, FrozenStep)
    assert (logits - reference).abs().max() > 1e-2


@pytest.mark.parametrize("inplace", [False, True])
def test_functional_step_matches_hugging_face_decode(tiny, inplace):
    """Logits and every new conv and SSM state, step by step, against Hugging Face."""
    model, tokens, reference = tiny
    logits, history = run_step(model, tokens, inplace)
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)
    kinds = nemotron_decode.layer_kinds(model.config)
    n_kv = 2 * kinds.count("full_attention")
    n_mamba = kinds.count("linear_attention")
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)
    with torch.inference_mode():
        for j, states in enumerate(history):
            model(
                input_ids=tokens[ISL + j].view(1, 1),
                position_ids=torch.full((1, 1), ISL + j),
                past_key_values=cache,
                use_cache=True,
            )
            conv, ssm = mamba_states(cache, kinds)
            ours_conv = states[n_kv : n_kv + n_mamba]
            ours_ssm = states[n_kv + n_mamba :]
            for got, want in zip(ours_conv, conv):
                torch.testing.assert_close(got.reshape(want.shape), want)
            for got, want in zip(ours_ssm, ssm):
                torch.testing.assert_close(
                    got.reshape(want.shape), want, atol=1e-5, rtol=1e-5
                )
            # The new key and value are written at the step's position.
            layer = cache.layers[kinds.index("full_attention")]
            torch.testing.assert_close(states[0], layer.keys, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(states[1], layer.values, atol=1e-6, rtol=1e-6)


def test_step_inputs_follow_state_shapes(tiny):
    model, tokens, _ = tiny
    config = model.config
    kinds = nemotron_decode.layer_kinds(config)
    assert kinds.count("linear_attention") == 3
    assert kinds.count("full_attention") == 1
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)
    for inplace in (False, True):
        shapes = nemotron_decode.state_shapes(config, ISL + OSL, inplace=inplace)
        states = nemotron_decode.static_states(
            cache, kinds, ssm_dtype=torch.bfloat16, inplace=inplace
        )
        assert [tuple(s.shape) for s in states] == [shapes["kv"]] * 2 + [
            shapes["conv"]
        ] * 3 + [shapes["ssm"]] * 3
        assert all(s.is_contiguous() for s in states)
        assert states[-1].dtype == torch.bfloat16
        assert shapes["conv"][-1] == config.conv_kernel
    # The one-slot layouts hold the same elements as the functional ones.
    functional = nemotron_decode.state_shapes(config, ISL + OSL)
    one_slot = nemotron_decode.state_shapes(config, ISL + OSL, inplace=True)
    for name in ("conv", "ssm"):
        assert len(one_slot[name]) == 4 and one_slot[name][2] == 1
        assert (
            torch.Size(one_slot[name]).numel() == torch.Size(functional[name]).numel()
        )


def hugging_face_candidate(model, tokens):
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)

    def step(input_ids, position_ids):
        return model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        ).logits

    return nemotron_decode._TorchCandidate(
        step,
        cache,
        kinds=nemotron_decode.layer_kinds(model.config),
        tokens=tokens,
        isl=ISL,
        osl=OSL,
        cudagraphs=False,
    )


def test_candidate_rewinds_the_hugging_face_cache(tiny):
    model, tokens, reference = tiny
    candidate = hugging_face_candidate(model, tokens)
    assert (candidate.units, candidate.items) == (OSL, 1)
    first, second = candidate(), candidate()
    torch.testing.assert_close(first, reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(second, first, atol=0, rtol=0)


def test_flex_attention_decode_mask_fix(tiny, monkeypatch):
    """transformers' FlexAttention decode sees the next cache slot until fixed."""
    import transformers.masking_utils as masking_utils

    for name in ("add_offsets_to_mask_function", "create_block_mask"):
        monkeypatch.setattr(masking_utils, name, getattr(masking_utils, name))
    monkeypatch.setattr(masking_utils, "_nemotron_masks_fixed", False, raising=False)
    model, tokens, reference = tiny
    flex = tiny_model("flex_attention")
    flex.load_state_dict(model.state_dict())
    kinds = nemotron_decode.layer_kinds(model.config)
    layer = kinds.index("full_attention")

    def decode_with_garbage_ahead():
        candidate = hugging_face_candidate(flex, tokens)
        # Large values in every slot after the prompt, as a previous call with
        # other tokens would leave; a correct mask never reads them early.
        with torch.inference_mode():
            candidate.cache.layers[layer].keys[:, :, ISL:] = 50.0
            candidate.cache.layers[layer].values[:, :, ISL:] = -50.0
        return candidate()

    broken = decode_with_garbage_ahead()
    assert (broken - reference).abs().max() > 1e-2
    nemotron_decode.fix_flex_attention_masks()
    fixed = decode_with_garbage_ahead()
    torch.testing.assert_close(fixed, reference, atol=1e-5, rtol=1e-5)


def test_tensorrt_candidate_threads_functional_states(tiny):
    model, tokens, reference = tiny
    kinds = nemotron_decode.layer_kinds(model.config)
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)
    start = nemotron_decode.static_states(cache, kinds)
    step = nemotron_decode.NemotronDecodeStep(model, ISL + OSL)

    class InPlaceCaches(torch.nn.Module):
        """Eager stand-in for the engine: KV caches written in place, as TensorRT does."""

        def forward(self, input_ids, position_ids, *states):
            out = step(input_ids, position_ids, *states)
            for cache_input, updated in zip(states[:2], out[1:3]):
                cache_input.copy_(updated)
            return out

    candidate = nemotron_decode._TensorRTCandidate(
        InPlaceCaches(),
        tokens=tokens,
        isl=ISL,
        osl=OSL,
        start=start,
        n_kv=2,
        inplace=False,
        cudagraphs=False,
    )
    first, second = candidate(), candidate()
    torch.testing.assert_close(first, reference, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(second, first, atol=0, rtol=0)


def test_knobs_and_baseline():
    workload = nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=8)
    knobs = knobs_by_name(workload.knobs())
    values = resolve(workload.baseline(), knobs)
    assert values["engine.precision"] == "bf16"
    assert values["workload.ssm_state"] == "fp32"
    assert values["workload.state_update"] == "functional"
    assert values["runtime.cudagraphs"] is False
    # Measured no-ops (optimization levels 4 and 5), a knob engines with
    # aliased KV outputs ignore and one TensorRT 11.4 cannot build them with
    # are not offered.
    assert "engine.optimization_level" not in knobs
    assert "runtime.pre_allocated_outputs" not in knobs
    assert "engine.tiling_optimization_level" not in knobs
    if "engine.build_route" in knobs:
        routes = knobs["engine.build_route"].values
        assert routes == nemotron_decode.DECODE_ROUTES
        assert "-kgen:codegen:cuda_tile=2" not in routes
    with pytest.raises(ValueError, match="engine.precision='fp16'"):
        apply_changes({}, {"engine.use_fp32_acc": True}, knobs)
    config = apply_changes(
        {}, {"engine.precision": "fp16", "engine.use_fp32_acc": True}, knobs
    )
    assert config == {"engine.precision": "fp16", "engine.use_fp32_acc": True}
    assert workload.accuracy_defaults() == {
        "top1_agreement": (">=", 0.97),
        "mean_kl": ("<=", 2e-3),
    }
    with pytest.raises(ValueError, match="isl >= 1 and osl >= 2"):
        nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=1)


def test_unknown_backend_is_refused():
    workload = nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=8)
    with pytest.raises(ValueError, match="Unknown bench.backend"):
        workload.build({"bench.backend": "onnx"})


def test_bytes_per_token_of_nemotron_3_nano_4b():
    config = tiny_config(
        vocab_size=131072,
        hidden_size=3136,
        intermediate_size=12544,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        mamba_num_heads=96,
        mamba_head_dim=80,
        n_groups=8,
        ssm_state_size=128,
        hybrid_override_pattern="M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-",
    )
    kinds = nemotron_decode.layer_kinds(config)
    assert (kinds.count("linear_attention"), kinds.count("mlp")) == (21, 17)
    embedding = 131072 * 3136 * 2
    weights = 7_947_000_000
    fp32 = nemotron_decode.decode_bytes_per_token(
        config,
        weight_bytes=weights,
        embedding_bytes=embedding,
        max_seq_len=192,
        dtype_size=2,
        ssm_dtype_size=4,
    )
    bf16 = nemotron_decode.decode_bytes_per_token(
        config,
        weight_bytes=weights,
        embedding_bytes=embedding,
        max_seq_len=192,
        dtype_size=2,
        ssm_dtype_size=2,
    )
    # The states are read and written once each: 2 x 4 bytes per FP32 element.
    ssm_elements = 21 * 96 * 80 * 128
    assert fp32 - bf16 == pytest.approx(2 * 2 * ssm_elements)
    kv = 2 * 4 * 8 * 192 * 128 * 2
    conv = 2 * 21 * (96 * 80 + 2 * 8 * 128) * 4 * 2
    assert fp32 == pytest.approx(weights - embedding + kv + conv + 2 * 4 * ssm_elements)
    # About 7.3 GB: a 4.5 ms floor at 1.61 TB/s.
    assert 7.2e9 < fp32 < 7.4e9


def test_compare_checks_the_reference_inputs():
    workload = nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=8)
    workload._tokens = torch.arange(16)
    logits = torch.randn(8, 32)
    reference = {"logits": logits, "tokens": torch.arange(16)}
    same = workload.compare(logits.clone(), reference)
    assert same["top1_agreement"] == 1.0
    assert same["mean_kl"] == pytest.approx(0.0, abs=1e-12)
    other = workload.compare(-logits, reference)
    assert other["top1_agreement"] < 0.5
    assert other["mean_kl"] > 0.1
    with pytest.raises(ValueError, match="Logits shape"):
        workload.compare(logits[:4], reference)
    with pytest.raises(ValueError, match="other tokens"):
        workload.compare(logits, {"logits": logits, "tokens": torch.arange(1, 17)})
    with pytest.raises(ValueError, match="recompute it"):
        workload.compare(logits, logits)
    prefill = nemotron_decode.create_prefill_case(
        "unused/model", length=64, positions=4
    )
    prefill._tokens = torch.arange(64)
    recorded = {"logits": logits[:4], "tokens": torch.arange(64)}
    with pytest.raises(ValueError, match="other positions"):
        prefill.compare(logits[:4], {**recorded, "positions": torch.arange(4)})
    result = prefill.compare(logits[:4], {**recorded, "positions": prefill.positions})
    assert result["top1_agreement"] == 1.0


def test_decode_step_editable_file_and_formulation_gate(tiny, tmp_path):
    model, tokens, reference = tiny
    workload = nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=8)
    (editable,) = workload.editable_files()
    assert editable.name == "decode_step"
    assert os.path.isabs(editable.path)
    with open(editable.path) as source:
        text = source.read()
    blocks = editable_blocks(text)
    assert len(blocks) == 2
    assert "def _rms_norm" in blocks[0] and "def _mamba" in blocks[1]
    # An edited copy (here: unchanged) is imported as its own module, its step
    # reproduces Hugging Face's decode, and it passes the gate exactly.
    copy = tmp_path / "edited.py"
    copy.write_text(text)
    edited = load_module(copy, "nemotron_decode_test_edited")
    assert edited.NemotronDecodeStep is not nemotron_decode.NemotronDecodeStep
    logits, _ = run_step(model, tokens, False, edited.NemotronDecodeStep)
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)
    for inplace in (False, True):
        report = nemotron_decode.formulation_gate(
            model, tokens, ISL, ISL + OSL, edited.NemotronDecodeStep, inplace=inplace
        )
        assert report["max_abs_logit_error"] == 0.0
        assert report["max_rel_state_error"] == 0.0
    # A step whose SSM state never updates fails it.
    broken = tmp_path / "broken.py"
    broken.write_text(apply_edits(text, [FROZEN_SSM]))
    frozen = load_module(broken, "nemotron_decode_test_frozen").NemotronDecodeStep
    report = nemotron_decode.check_formulation(model, tokens, ISL, ISL + OSL, frozen)
    assert report["max_rel_state_error"] > 1e-2
    assert nemotron_decode.formulation_failures(report)
    with pytest.raises(ValueError, match="max_rel_state_error"):
        nemotron_decode.formulation_gate(model, tokens, ISL, ISL + OSL, frozen)


def test_formulation_gate_runs_in_fp32_and_restores_dtypes(tiny):
    model, tokens, _ = tiny
    half = tiny_model()
    half.load_state_dict(model.state_dict())
    half.to(torch.bfloat16)
    half.model.layers[0].mixer.A_log.data = half.model.layers[0].mixer.A_log.float()
    before = {name: p.clone() for name, p in half.named_parameters()}
    seen = []

    class Spy(nemotron_decode.NemotronDecodeStep):
        def forward(self, input_ids, position_ids, *states):
            seen.append(self.model.lm_head.weight.dtype)
            return super().forward(input_ids, position_ids, *states)

    nemotron_decode.formulation_gate(half, tokens, ISL, ISL + OSL, Spy)
    assert seen and set(seen) == {torch.float32}
    for name, parameter in half.named_parameters():
        assert parameter.dtype == before[name].dtype
        assert torch.equal(parameter, before[name])


def test_prefill_logits_match_the_full_forward_for_any_chunk_size(tiny):
    model, tokens, _ = tiny
    length = ISL + OSL
    positions = nemotron_decode.prefill_positions(length, 4)
    assert positions.tolist() == [3, 7, 11, 16]
    ids = tokens.view(1, -1)
    with torch.inference_mode():
        full = model(ids).logits[0, positions]
        logits = []
        for chunk in (8, 4, 2):
            nemotron_decode.set_chunk_size(model, chunk)
            logits.append(nemotron_decode.PrefillLogits(model, positions)(ids))
    nemotron_decode.set_chunk_size(model, 8)
    for out in logits:
        assert out.dtype == torch.float32
        torch.testing.assert_close(out, full, atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError):
        nemotron_decode.prefill_positions(8, 9)


def test_prefill_knobs_baseline_and_text():
    workload = nemotron_decode.create_prefill_case("unused/model", length=2048)
    assert workload.positions.tolist()[:2] == [31, 63]
    assert workload.positions[-1] == 2047 and len(workload.positions) == 64
    knobs = knobs_by_name(workload.knobs())
    values = resolve(workload.baseline(), knobs)
    assert values["engine.precision"] == "bf16"
    assert values["engine.tf32"] is False
    assert values["workload.chunk_size"] == 256
    assert values["workload.ssd"] == "hf"
    # CUDA graphs and optimization levels measured as no-ops: not offered.
    assert set(knobs) == {
        "engine.precision",
        "engine.use_fp32_acc",
        "engine.tf32",
        "workload.ssd",
        "workload.chunk_size",
        "kernels.inductor",
    }
    assert values["kernels.inductor"] == "none"
    assert workload.timing().calls == 1
    assert workload.accuracy_defaults() == {
        "top1_agreement": (">=", 0.85),
        "mean_kl": ("<=", 2e-3),
    }
    with pytest.raises(ValueError, match="Unknown bench.backend"):
        workload.build({"bench.backend": "onnx"})
    with pytest.raises(ValueError, match="bench.fp16_accumulation"):
        workload.build({"bench.backend": "inductor", "bench.fp16_accumulation": True})
    # The prompt does not repeat: no sentence occurs twice, and it is long
    # enough for 2048 tokens (about 1.3 tokens per word).
    text = nemotron_decode.PREFILL_TEXT
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    assert len(sentences) == len(set(sentences))
    assert len(text.split()) > 2048 / 1.3


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("chunk", [8, 16])
def test_ssd_chunk_scan_matches_hugging_face(grouped, chunk):
    from transformers.models.nemotron_h import modeling_nemotron_h

    hf_scan = getattr(
        modeling_nemotron_h,
        "_hf_mamba2_chunk_scan",
        modeling_nemotron_h.mamba2_chunk_scan,
    )
    torch.manual_seed(0)
    # 37 tokens: the last chunk is padded.
    x, dt = torch.randn(1, 37, 8, 4), torch.randn(1, 37, 8)
    a, b, c = -torch.rand(8) * 2, torch.randn(1, 37, 2, 6), torch.randn(1, 37, 2, 6)
    kwargs = dict(
        D=torch.rand(8),
        dt_bias=torch.randn(8),
        initial_states=torch.randn(1, 8, 4, 6),
        dt_softplus=True,
        dt_limit=(0.001, float("inf")),
        return_final_states=True,
    )
    expected = hf_scan(x, dt, a, b, c, chunk, **kwargs)
    actual = nemotron_decode.ssd_chunk_scan(
        x, dt, a, b, c, chunk, grouped=grouped, **kwargs
    )
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("variant", ["segsum", "grouped"])
def test_use_ssd_switches_the_model_scan(tiny, monkeypatch, variant):
    model, tokens, _ = tiny
    positions = nemotron_decode.prefill_positions(ISL + OSL, 4)
    ids = tokens.view(1, -1)
    calls = []
    scan = nemotron_decode.ssd_chunk_scan

    def counting_scan(*args, **kwargs):
        calls.append(kwargs["grouped"])
        return scan(*args, **kwargs)

    monkeypatch.setattr(nemotron_decode, "ssd_chunk_scan", counting_scan)
    try:
        with torch.inference_mode():
            nemotron_decode.use_ssd("hf")
            expected = nemotron_decode.PrefillLogits(model, positions)(ids)
            assert calls == []
            nemotron_decode.use_ssd(variant)
            actual = nemotron_decode.PrefillLogits(model, positions)(ids)
    finally:
        nemotron_decode.use_ssd("hf")
    # Every Mamba layer ran the selected scan, with the same result.
    assert calls == [variant == "grouped"] * 3
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    with torch.inference_mode():
        nemotron_decode.PrefillLogits(model, positions)(ids)
    assert len(calls) == 3
    with pytest.raises(ValueError, match="Unknown SSD scan"):
        nemotron_decode.use_ssd("triton")


# ------------------------------------------------- kernels.inductor (hybrids)


def test_decode_inductor_knob_requires_inplace_states_and_cudagraphs():
    workload = nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=8)
    knobs = knobs_by_name(workload.knobs())
    knob = knobs["kernels.inductor"]
    assert knob.values == nemotron_decode.DECODE_INDUCTOR
    assert knob.default == "none"
    assert dict(knob.requires) == {
        "runtime.cudagraphs": True,
        "workload.state_update": "inplace",
    }
    assert resolve(workload.baseline(), knobs)["kernels.inductor"] == "none"
    # No measured results in the description, only the mechanism.
    assert "%" not in knob.description and " ms" not in knob.description
    for variant in ("lm-head", "mlp-inductor"):
        with pytest.raises(ValueError, match="requires"):
            apply_changes({}, {"kernels.inductor": variant}, knobs)
        config = apply_changes(
            {"workload.state_update": "inplace", "runtime.cudagraphs": True},
            {"kernels.inductor": variant},
            knobs,
        )
        assert config["kernels.inductor"] == variant
    # The plain config is unchanged: "none" is the default and drops out.
    assert apply_changes({}, {"kernels.inductor": "none"}, knobs) == {}


def test_decode_hybrid_builds_check_their_requirements_before_loading():
    workload = nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=8)
    values = {
        "engine.precision": "bf16",
        "engine.use_fp32_acc": False,
        "workload.ssm_state": "bf16",
        "workload.state_update": "functional",
        "runtime.cudagraphs": True,
        "kernels.inductor": "lm-head",
    }
    with pytest.raises(ValueError, match="state_update='inplace'"):
        workload.build(values)
    with pytest.raises(ValueError, match="runtime.cudagraphs=True"):
        workload.build(
            {**values, "workload.state_update": "inplace", "runtime.cudagraphs": False}
        )
    # The whole-step capture control has the same requirements.
    with pytest.raises(ValueError, match="bench.whole_graph"):
        workload.build(
            {**values, "kernels.inductor": "none", "bench.whole_graph": True}
        )
    with pytest.raises(ValueError, match="Unknown kernels.inductor"):
        workload.build({**values, "kernels.inductor": "attention"})
    with pytest.raises(ValueError, match="Unknown kernels.inductor"):
        nemotron_decode.NemotronDecodeStep(tiny_model(), 8, inductor="attention")


@pytest.mark.parametrize("variant", ["lm-head", "mlp-inductor"])
@pytest.mark.parametrize("inplace", [False, True])
def test_decode_regions_change_nothing_eagerly(tiny, variant, inplace):
    """Outside strict torch_tensorrt compilation the region scopes are no-ops."""
    model, tokens, reference = tiny
    plain, plain_states = run_step(model, tokens, inplace)
    step = functools.partial(nemotron_decode.NemotronDecodeStep, inductor=variant)
    logits, states = run_step(model, tokens, inplace, step)
    torch.testing.assert_close(logits, plain, atol=0, rtol=0)
    for got, want in zip(states[-1], plain_states[-1]):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)


def test_decode_regions_are_the_scopes_strict_export_captures(tiny, monkeypatch):
    """Which operations each variant scopes: counted by name at trace time."""
    model, tokens, _ = tiny
    names = []

    def scope(name):
        names.append(name)
        return nemotron_decode.contextlib.nullcontext()

    monkeypatch.setattr(nemotron_decode, "_inductor_scope", scope)
    kinds = nemotron_decode.layer_kinds(model.config)
    for variant, expected in (
        ("none", []),
        ("lm-head", ["lm_head"]),
        ("mlp-inductor", [f"mlp_{i}" for i, k in enumerate(kinds) if k == "mlp"]),
    ):
        names.clear()
        step = functools.partial(nemotron_decode.NemotronDecodeStep, inductor=variant)
        run_step(model, tokens, True, step)
        assert names == expected * OSL
    assert kinds.count("mlp") == 2


def test_named_inputs_names_every_input():
    """torch_tensorrt's tracer keys dynamic shapes by forward parameter names."""

    class Varargs(torch.nn.Module):
        def forward(self, first, *rest):
            return first + sum(rest)

    inputs = [torch.ones(2) * i for i in range(4)]
    wrapper = nemotron_decode.named_inputs(Varargs(), len(inputs))
    names = list(inspect.signature(wrapper.forward).parameters)
    assert names == ["input_0", "input_1", "input_2", "input_3"]
    torch.testing.assert_close(wrapper(*inputs), Varargs()(*inputs))
    from torch_tensorrt.dynamo._tracer import get_dynamic_shapes_args

    assert list(get_dynamic_shapes_args(wrapper, inputs)) == names
    # The tracer's names for the bare module put four inputs under three keys.
    assert len(get_dynamic_shapes_args(Varargs(), inputs)) == 2
    program = torch.export.export(
        wrapper, tuple(inputs), dynamic_shapes={name: {} for name in names}
    )
    torch.testing.assert_close(program.module()(*inputs), wrapper(*inputs))


def test_prefill_inductor_knob_and_build_checks():
    workload = nemotron_decode.create_prefill_case(
        "unused/model", length=64, positions=4
    )
    knobs = knobs_by_name(workload.knobs())
    knob = knobs["kernels.inductor"]
    assert knob.values == nemotron_decode.PREFILL_INDUCTOR
    assert knob.default == "none" and dict(knob.requires) == {}
    assert "%" not in knob.description and " ms" not in knob.description
    assert apply_changes({}, {"kernels.inductor": "mamba-inductor"}, knobs) == {
        "kernels.inductor": "mamba-inductor"
    }
    with pytest.raises(ValueError, match="cannot honor it"):
        workload.build(
            {"bench.backend": "inductor", "kernels.inductor": "mamba-inductor"}
        )
    with pytest.raises(ValueError, match="Unknown kernels.inductor"):
        workload.build({"kernels.inductor": "attention"})


@pytest.mark.parametrize("ssd", ["hf", "grouped"])
def test_mamba_regions_reproduce_the_mixer(tiny, ssd):
    """The patched mixers compute exactly what Hugging Face's do, eagerly."""
    model, tokens, reference = tiny
    patched = tiny_model()
    patched.load_state_dict(model.state_dict())
    assert nemotron_decode.install_mamba_regions(patched) == 3
    mixers = nemotron_decode.mamba_mixers(patched)
    assert [m.layer_idx for m in mixers] == [0, 2, 5]
    assert all(isinstance(m.forward, types.MethodType) for m in mixers)
    positions = nemotron_decode.prefill_positions(ISL + OSL, 4)
    ids = tokens.view(1, -1)
    try:
        nemotron_decode.use_ssd(ssd)
        with torch.inference_mode():
            expected = nemotron_decode.PrefillLogits(model, positions)(ids)
            actual = nemotron_decode.PrefillLogits(patched, positions)(ids)
    finally:
        nemotron_decode.use_ssd("hf")
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Calls with a cache (prefill into it, then decode) take Hugging Face's path.
    decoded = nemotron_decode._hf_decode_logits(patched, tokens, ISL, OSL)
    torch.testing.assert_close(decoded, reference, atol=0, rtol=0)


def test_mamba_region_scope_covers_the_interior_only(tiny, monkeypatch):
    model, tokens, _ = tiny
    patched = tiny_model()
    patched.load_state_dict(model.state_dict())
    nemotron_decode.install_mamba_regions(patched)
    inside = []
    calls = []
    scope_open = [False]

    @nemotron_decode.contextlib.contextmanager
    def scope(name):
        calls.append(name)
        scope_open[0] = True
        try:
            yield
        finally:
            scope_open[0] = False

    def watch(module, name):
        module.register_forward_hook(lambda *args: inside.append((name, scope_open[0])))

    for mixer in nemotron_decode.mamba_mixers(patched):
        watch(mixer.in_proj, "in_proj")
        watch(mixer.out_proj, "out_proj")
        watch(mixer.norm, "norm")
    monkeypatch.setattr(nemotron_decode, "_inductor_scope", scope)
    with torch.inference_mode():
        nemotron_decode.PrefillLogits(patched, torch.tensor([3]))(tokens.view(1, -1))
    assert calls == ["mamba_scan"] * 3
    assert inside == [("in_proj", False), ("norm", True), ("out_proj", False)] * 3


def test_release_unused_weights_keeps_what_the_compiled_module_reads():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    model.register_buffer("scale", torch.ones(4))
    kept = model[1].weight
    compiled = torch.fx.symbolic_trace(lambda x: x)
    compiled.register_parameter("w", kept)
    compiled.register_buffer("scale_view", model.scale[:2])
    released = nemotron_decode.release_unused_weights(model, compiled)
    # model[0].weight and both biases are freed; the kept weight and the
    # buffer that a view shares storage with are not.
    assert released == (16 + 4 + 4) * 4
    assert model[0].weight.numel() == 0 and model[0].bias.numel() == 0
    assert model[1].bias.numel() == 0
    assert model[1].weight.numel() == 16 and model.scale.numel() == 4


def test_region_compilation_sets_and_restores_inductor_and_tf32():
    import torch._inductor.config as inductor_config

    before = (
        inductor_config.max_autotune,
        inductor_config.coordinate_descent_tuning,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    options = torch._inductor.list_mode_options(nemotron_decode.REGION_INDUCTOR_MODE)
    assert options.get("max_autotune") and options.get("coordinate_descent_tuning")
    assert not options.get("triton.cudagraphs")
    stack = nemotron_decode.region_compilation(tf32=not before[2])
    try:
        assert inductor_config.max_autotune is True
        assert inductor_config.coordinate_descent_tuning is True
        assert torch.backends.cuda.matmul.allow_tf32 is (not before[2])
        assert torch.backends.cudnn.allow_tf32 is (not before[2])
    finally:
        stack.close()
    after = (
        inductor_config.max_autotune,
        inductor_config.coordinate_descent_tuning,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    assert after == before
    # Without tf32 the flags are left alone.
    nemotron_decode.region_compilation().close()
    assert torch.backends.cuda.matmul.allow_tf32 is before[2]


def test_inductor_kernels_reads_the_generated_wrapper_code(tmp_path, monkeypatch):
    from torch._inductor.codecache import PyCodeCache

    wrapper = tmp_path / "wrapper.py"
    wrapper.write_text(
        "async_compile = AsyncCompile()\n"
        "triton_red_fused_add_mean_mm_0 = async_compile.triton(\n"
        "    'triton_red_fused_add_mean_mm_0', '''...''')\n"
        "triton_tem_fused_bmm_1 = async_compile.triton('triton_tem_fused_bmm_1', '')\n"
        "def call(args):\n"
        "    extern_kernels.bmm(a, b, out=c)\n"
    )
    other = tmp_path / "benchmark.py"
    other.write_text("triton_poi_fused_0 = async_compile.triton('x', '')\n")
    modules = [
        types.SimpleNamespace(__file__=str(wrapper)),
        types.SimpleNamespace(__file__=str(other)),
        types.SimpleNamespace(__file__=None),
    ]
    monkeypatch.setattr(PyCodeCache, "modules", modules)
    assert nemotron_decode.inductor_kernels() == [
        "extern_kernels.bmm",
        "triton_red_fused_add_mean_mm_0",
        "triton_tem_fused_bmm_1",
    ]


def test_whole_graph_candidate_needs_inplace_states(tiny):
    model, tokens, _ = tiny
    kinds = nemotron_decode.layer_kinds(model.config)
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)
    start = nemotron_decode.static_states(cache, kinds)
    with pytest.raises(ValueError, match="in-place state updates"):
        nemotron_decode._TensorRTCandidate(
            torch.nn.Identity(),
            tokens=tokens,
            isl=ISL,
            osl=OSL,
            start=start,
            n_kv=2,
            inplace=False,
            cudagraphs=True,
            whole_graph=True,
        )


def test_prefill_graph_variant_runs_through_the_whole_graph_wrapper(monkeypatch):
    """mamba-inductor-graph replays the module through enable_cudagraphs."""
    import contextlib

    import torch_tensorrt

    events = []

    @contextlib.contextmanager
    def enable_cudagraphs(module):
        events.append("enter")
        yield lambda ids: module(ids) * 2
        events.append("exit")

    monkeypatch.setattr(torch_tensorrt.runtime, "enable_cudagraphs", enable_cudagraphs)
    ids = torch.ones(1, 3)
    plain = nemotron_decode._PrefillCandidate(lambda x: x + 1, ids, backend="tensorrt")
    plain.setup()
    torch.testing.assert_close(plain(), ids + 1)
    plain.close()
    graph = nemotron_decode._PrefillCandidate(
        lambda x: x + 1, ids, backend="tensorrt", whole_graph=True
    )
    graph.setup()
    torch.testing.assert_close(graph(), (ids + 1) * 2)
    graph.close()
    torch.testing.assert_close(graph(), ids + 1)
    assert events == ["enter", "exit"]


def _with_inductor_region(module):
    """``module`` carrying the record of one execute_in_torch(backend="inductor")
    region, which the exporter refuses before it writes anything."""
    from torch_tensorrt.dynamo.region_compilation._types import (
        RegionRecord,
        attach_region_records,
    )

    record = RegionRecord(
        id="mlp_1#0", name="mlp_1", child_target="x", backend="inductor"
    )
    attach_region_records(module, [record])
    return module


class _AddOne(torch.nn.Module):
    def forward(self, x):
        return x + 1


@pytest.mark.parametrize(
    "inductor",
    nemotron_decode.DECODE_INDUCTOR[1:] + nemotron_decode.PREFILL_INDUCTOR[1:],
)
def test_inductor_builds_report_that_they_cannot_be_saved(tmp_path, inductor):
    from torch_tensorrt import region
    from torch_tensorrt.autotune._measure import export_unsupported

    program = _with_inductor_region(torch.fx.symbolic_trace(_AddOne()))
    with pytest.raises(NotImplementedError, match="cannot be saved") as caught:
        nemotron_decode.save_program(program, tmp_path / "decode.ep", inductor)
    assert f"kernels.inductor={inductor!r}" in str(caught.value)
    assert isinstance(caught.value.__cause__, region.RegionError)
    # The session records it as an export the candidate cannot make.
    assert export_unsupported(caught.value)
    assert list(tmp_path.iterdir()) == []
    # Without a hybrid value the exporter's own error is not rewritten.
    for plain in (None, "none"):
        with pytest.raises(region.RegionError, match="is not supported"):
            nemotron_decode.save_program(program, tmp_path / "decode.ep", plain)


def test_candidates_save_with_their_hybrid_value(tiny, tmp_path, monkeypatch):
    import torch_tensorrt

    saved = []
    monkeypatch.setattr(
        torch_tensorrt,
        "save",
        lambda module, path, **kwargs: saved.append((module, path, kwargs)),
    )
    module = torch.nn.Identity()
    ids = torch.zeros(1, 4, dtype=torch.int64)
    prefill = nemotron_decode._PrefillCandidate(module, ids, backend="tensorrt")
    assert prefill.export(tmp_path) == ["prefill.ep"]
    # Protocol 4: the engine's weights are one pickled constant above 4 GiB.
    assert saved == [
        (module, str(tmp_path / "prefill.ep"), {"retrace": False, "pickle_protocol": 4})
    ]
    eager = nemotron_decode._PrefillCandidate(module, ids, backend="eager")
    assert eager.export(tmp_path) == []
    # Each candidate passes its kernels.inductor value on to save_program.
    model, tokens, _ = tiny
    kinds = nemotron_decode.layer_kinds(model.config)
    cache = nemotron_decode.prefill(model, tokens[:ISL].view(1, -1), ISL + OSL)
    decode = nemotron_decode._TensorRTCandidate(
        module,
        tokens=tokens,
        isl=ISL,
        osl=OSL,
        start=nemotron_decode.static_states(cache, kinds),
        n_kv=2,
        inplace=True,
        cudagraphs=False,
    )
    calls = []

    def save_program(module, path, inductor=None):
        calls.append((path.name, inductor))
        return [path.name]

    monkeypatch.setattr(nemotron_decode, "save_program", save_program)
    decode.details = {"inductor": "lm-head"}
    prefill.details = {"inductor": "mamba-inductor"}
    assert decode.export(tmp_path) == ["decode.ep"]
    assert prefill.export(tmp_path) == ["prefill.ep"]
    decode.details = {}
    decode.export(tmp_path)
    assert calls == [
        ("decode.ep", "lm-head"),
        ("prefill.ep", "mamba-inductor"),
        ("decode.ep", None),
    ]


# ------------------------------------------------ kernels.generated (decode)
import region_generation  # noqa: E402


def test_decode_generated_knob_is_prepared_and_needs_cuda_graphs():
    from torch_tensorrt.autotune._knobs import with_requirements

    workload = nemotron_decode.NemotronDecodeWorkload("unused/model", isl=8, osl=8)
    knobs = knobs_by_name(workload.knobs())
    knob = knobs["kernels.generated"]
    assert knob.values == ("none", "mlp") == nemotron_decode.DECODE_GENERATED
    assert knob.default == "none"
    assert dict(knob.requires) == {
        "runtime.cudagraphs": True,
        "kernels.inductor": "none",
    }
    # A session generates the MLP kernels once, before its search.
    assert knob.prepare == ("mlp",)
    assert knobs["kernels.inductor"].prepare == ()
    assert "%" not in knob.description and " ms" not in knob.description
    assert resolve(workload.baseline(), knobs)["kernels.generated"] == "none"
    with pytest.raises(ValueError, match="requires runtime.cudagraphs=True"):
        apply_changes({}, {"kernels.generated": "mlp"}, knobs)
    # Per-engine CUDA graphs suffice: no in-place state requirement.
    assert apply_changes(
        {"runtime.cudagraphs": True}, {"kernels.generated": "mlp"}, knobs
    ) == {"kernels.generated": "mlp", "runtime.cudagraphs": True}
    hybrid = {
        "kernels.inductor": "lm-head",
        "runtime.cudagraphs": True,
        "workload.state_update": "inplace",
    }
    with pytest.raises(ValueError, match="requires kernels.inductor='none'"):
        apply_changes(hybrid, {"kernels.generated": "mlp"}, knobs)
    assert with_requirements(hybrid, {"kernels.generated": "mlp"}, knobs) == {
        "kernels.generated": "mlp",
        "kernels.inductor": "none",
    }
    # The prefill workload has no generated regions.
    prefill = nemotron_decode.NemotronPrefillWorkload("unused/model")
    assert "kernels.generated" not in {knob.name for knob in prefill.knobs()}


def test_decode_generated_values_are_checked_before_loading(monkeypatch, tmp_path):
    workload = nemotron_decode.create_case(
        "unused/model", isl=8, osl=8, kernel_cache=str(tmp_path / "kcache")
    )
    values = {
        "engine.precision": "bf16",
        "engine.use_fp32_acc": False,
        "workload.ssm_state": "bf16",
        "workload.state_update": "functional",
        "runtime.cudagraphs": False,
        "kernels.inductor": "none",
        "kernels.generated": "mlp",
    }
    with pytest.raises(ValueError, match="needs runtime.cudagraphs=True"):
        workload.build(values)
    with pytest.raises(ValueError, match="needs kernels.inductor='none'"):
        workload.build(
            {
                **values,
                "runtime.cudagraphs": True,
                "workload.state_update": "inplace",
                "kernels.inductor": "lm-head",
            }
        )
    with pytest.raises(ValueError, match="Unknown kernels.generated"):
        workload.build({**values, "kernels.generated": "attention"})
    with pytest.raises(ValueError, match="bench.backend='tensorrt'"):
        workload.build({**values, "runtime.cudagraphs": True, "bench.backend": "eager"})
    # A valid config makes the workload's generator before it loads the model.
    made = []

    class Stop(Exception):
        pass

    def load(*args, **kwargs):
        raise Stop

    original = workload.kernel_generator
    monkeypatch.setattr(
        workload, "kernel_generator", lambda: made.append(original()) or made[-1]
    )
    monkeypatch.setattr(nemotron_decode, "_load", load)
    with pytest.raises(Stop):
        workload.build({**values, "runtime.cudagraphs": True})
    (generator,) = made
    assert generator == region_generation.make_generator(
        tmp_path / "kcache", "nemotron_decode_kernel_notes.md"
    )


def test_generated_mlp_regions_change_nothing_eagerly(tiny, tmp_path):
    """Outside strict torch_tensorrt compilation the regions run their bodies:
    the step computes, bit for bit, what the plain step computes."""
    model, tokens, reference = tiny
    generator = region_generation.make_generator(tmp_path)
    for inplace in (False, True):
        plain, plain_states = run_step(model, tokens, inplace)
        step = functools.partial(
            nemotron_decode.NemotronDecodeStep, generator=generator
        )
        logits, states = run_step(model, tokens, inplace, step)
        torch.testing.assert_close(logits, plain, atol=0, rtol=0)
        for got, want in zip(states[-1], plain_states[-1]):
            torch.testing.assert_close(got, want, atol=0, rtol=0)
        torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="cannot be combined"):
        nemotron_decode.NemotronDecodeStep(
            model, 8, inductor="mlp-inductor", generator=generator
        )


def test_generated_mlp_regions_are_two_per_block(tiny, tmp_path, monkeypatch):
    """mlp_in (norm, up GEMV, ReLU^2) and mlp_out (down GEMV, residual add) of
    every MLP block, with their inputs and the step's generator."""
    import contextlib

    from torch_tensorrt import region

    model, tokens, _ = tiny
    generator = region_generation.make_generator(tmp_path)
    calls = []

    def fused_kernel(**kwargs):
        calls.append(kwargs)
        return contextlib.nullcontext()

    monkeypatch.setattr(region, "fused_kernel", fused_kernel)
    step = functools.partial(nemotron_decode.NemotronDecodeStep, generator=generator)
    run_step(model, tokens, True, step)
    kinds = nemotron_decode.layer_kinds(model.config)
    mlp = [i for i, kind in enumerate(kinds) if kind == "mlp"]
    names = [name for i in mlp for name in (f"mlp_in_{i}", f"mlp_out_{i}")]
    assert [call["name"] for call in calls] == names * OSL
    assert all(set(call) == {"generator", "inputs", "name"} for call in calls)
    assert all(call["generator"] is generator for call in calls)
    hidden, inter = model.config.hidden_size, model.config.intermediate_size
    for index, (first, second) in zip(mlp, zip(calls[::2], calls[1::2])):
        block = model.model.layers[index]
        residual, weight, up = first["inputs"]
        act, down, residual_again = second["inputs"]
        assert residual_again is residual and residual.shape == (1, 1, hidden)
        assert weight is block.norm.weight
        assert up is block.mixer.up_proj.weight
        assert down is block.mixer.down_proj.weight
        assert act.shape == (1, 1, inter) and act.is_contiguous()
