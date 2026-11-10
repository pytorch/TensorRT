# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""agentic_decode: the decode workload's knobs, editable files and estimates.

Run with ``python -m pytest tools/llm/tests/test_agentic_decode.py``; no GPU
or model download is needed.
"""

import os
import re
import sys
import warnings
from types import SimpleNamespace

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import agentic_decode  # noqa: E402
from torch_tensorrt.autotune._code_edits import editable_blocks  # noqa: E402
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (  # noqa: E402
    ATEN_POST_LOWERING_PASSES,
)
from transformers import (  # noqa: E402
    GPT2Config,
    GPT2LMHeadModel,
    Qwen3Config,
    Qwen3ForCausalLM,
)


def qwen3_config():
    return Qwen3Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )


def test_qwen3_models_get_fused_blocks_and_their_editable_files(tmp_path):
    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    knobs = {knob.name: knob for knob in workload.knobs()}
    assert knobs["kernels.decode_blocks"].values == (
        "",
        "mlp",
        "mlp+attn",
        "layers",
        "all",
        "generated",
    )
    assert knobs["kernels.decode_blocks"].requires == (
        ("engine.precision", "fp16"),
        ("runtime.cudagraphs", True),
        ("workload.cache", "static_v3"),
    )
    # A session prepares the generated blocks once, before its search.
    assert knobs["kernels.decode_blocks"].prepare == ("generated",)
    assert knobs["kernels.decode_blocks"].to_json()["prepare"] == ["generated"]
    assert knobs["kernels.inductor"].prepare == ()
    files = {editable.name: editable for editable in workload.editable_files()}
    assert set(files) == {"decode_launch", "decode_kernels"}
    # The editable file quotes this model's GEMV shapes.
    assert "attn_out N=128 K=128" in files["decode_launch"].description
    for editable in files.values():
        assert os.path.isabs(editable.path)
        with open(editable.path) as source:
            assert editable_blocks(source.read())
    GPT2Config(n_layer=1, n_embd=32, n_head=2).save_pretrained(tmp_path / "gpt2")
    # The kernels keep a whole hidden row in registers: no fused blocks for a
    # hidden size that is not a power of two (Qwen3-4B has 2560).
    odd = qwen3_config()
    odd.hidden_size = 96
    odd.save_pretrained(tmp_path / "qwen3-odd")
    for name in ("gpt2", "qwen3-odd"):
        other = agentic_decode.DecodeWorkload(str(tmp_path / name), isl=8, osl=8)
        assert "kernels.decode_blocks" not in {knob.name for knob in other.knobs()}
        assert other.editable_files() == ()


@pytest.mark.parametrize("blocks", ["layers", "generated"])
@pytest.mark.parametrize(
    "changes, match",
    [
        ({"workload.cache": "static_v2"}, "workload.cache=static_v3"),
        ({"engine.precision": "bf16"}, "engine.precision=fp16"),
        ({"runtime.cudagraphs": False}, "runtime.cudagraphs=True"),
    ],
)
def test_fused_blocks_refuse_settings_they_cannot_run_with(
    tmp_path, changes, match, blocks
):
    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    values = {
        "kernels.decode_blocks": blocks,
        "workload.cache": "static_v3",
        "engine.precision": "fp16",
        "runtime.cudagraphs": True,
        **changes,
    }
    with pytest.raises(ValueError, match=match):
        workload.build(values)


def test_generated_blocks_need_the_settings_of_the_hand_written_ones(tmp_path):
    from torch_tensorrt.autotune._knobs import (
        apply_changes,
        knobs_by_name,
        with_requirements,
    )

    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    knobs = knobs_by_name(workload.knobs())
    blocks = knobs["kernels.decode_blocks"]
    assert blocks.values[-1] == "generated"
    # Generated kernels run in the engine in place of the hand-written ones,
    # so "generated" requires what the presets require, and says so.
    settings = {
        "engine.precision": "fp16",
        "runtime.cudagraphs": True,
        "workload.cache": "static_v3",
    }
    assert blocks.requirements("generated") == settings
    assert blocks.requirements("layers") == settings
    assert blocks.requirements("") == {}
    assert "Every value but '' requires static_v3, fp16 and CUDA" in (
        blocks.description
    )
    assert "'generated' fuses the 'layers' blocks" in blocks.description
    assert "same settings (static_v3, fp16, CUDA graphs)" in blocks.description
    assert not re.search(r"\d\s*(ms|us|%)", blocks.description)
    # A config without them is refused before anything is built; a planner's
    # change brings along what the baseline lacks (it is fp16 already).
    with pytest.raises(
        ValueError, match="kernels.decode_blocks='generated' requires runtime"
    ):
        apply_changes({}, {"kernels.decode_blocks": "generated"}, knobs)
    bundled = with_requirements(
        workload.baseline(), {"kernels.decode_blocks": "generated"}, knobs
    )
    assert bundled == {
        "kernels.decode_blocks": "generated",
        "workload.cache": "static_v3",
        "runtime.cudagraphs": True,
    }
    assert apply_changes(workload.baseline(), bundled, knobs) == bundled


def test_generated_blocks_take_their_kernels_from_the_kernel_cache(
    tmp_path, monkeypatch
):
    import qwen3_regions
    from torch_tensorrt.region import KernelGenerator

    qwen3_config().save_pretrained(tmp_path / "qwen3")
    model = str(tmp_path / "qwen3")
    # The cache: the workload's kernel_cache, else $TORCHTRT_KERNEL_CACHE,
    # else the default directory.
    monkeypatch.delenv("TORCHTRT_KERNEL_CACHE", raising=False)
    workload = agentic_decode.DecodeWorkload(model, isl=8, osl=8)
    import region_generation

    assert workload.kernel_cache == region_generation.DEFAULT_KERNEL_CACHE
    monkeypatch.setenv("TORCHTRT_KERNEL_CACHE", str(tmp_path / "from-env"))
    workload = agentic_decode.DecodeWorkload(model, isl=8, osl=8)
    assert workload.kernel_cache == str(tmp_path / "from-env")
    workload = agentic_decode.create_case(
        model, isl=8, osl=8, kernel_cache=str(tmp_path / "kcache")
    )
    assert workload.kernel_cache == str(tmp_path / "kcache")
    # A preset launches the hand-written kernels; "generated" fuses the
    # "layers" blocks with the shared generator and the decode blocks' notes,
    # so every build finds the kernels cached there.
    assert workload._blocks(qwen3_regions, "mlp") == (qwen3_regions.PRESETS["mlp"], {})
    blocks, options = workload._blocks(qwen3_regions, "generated")
    assert blocks == qwen3_regions.PRESETS["layers"]
    assert list(options) == ["generator"]
    assert isinstance(options["generator"], KernelGenerator)
    assert str(options["generator"].cache_dir) == str(tmp_path / "kcache")
    # The shared generator settings, with the decode blocks' notes.
    import region_generation

    assert agentic_decode.GENERATION_NOTES == "qwen3_decode_kernel_notes.md"
    assert options["generator"] == region_generation.make_generator(
        str(tmp_path / "kcache"), "qwen3_decode_kernel_notes.md"
    )
    # Its checks pass, and the build goes on to load the model, which has no
    # weights here.
    values = {
        "kernels.decode_blocks": "generated",
        "workload.cache": "static_v3",
        "engine.precision": "fp16",
        "runtime.cudagraphs": True,
    }
    with pytest.raises(OSError, match="no file named"):
        workload.build(values)
    # A generated build says which cache its kernels came from.
    candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=False)
    assert "kernel_cache" not in candidate.describe()
    candidate.kernel_cache = workload.kernel_cache
    assert candidate.describe()["kernel_cache"] == str(tmp_path / "kcache")


def test_bytes_per_token_count_weights_once_and_the_kv_cache():
    torch.manual_seed(0)
    qwen3 = Qwen3ForCausalLM(qwen3_config())
    weights = sum(p.numel() * p.element_size() for p in qwen3.parameters())
    if qwen3.lm_head.weight.data_ptr() != qwen3.model.embed_tokens.weight.data_ptr():
        weights -= qwen3.model.embed_tokens.weight.numel() * 4
    # 2 (K and V) x 2 layers x 2 KV heads x 32 dims x 16 positions x 4 bytes.
    kv = 2 * 2 * 2 * 32 * 16 * 4
    estimate = agentic_decode._decode_bytes_per_token(qwen3, "static_v3", 8, 8)
    assert estimate == weights + kv
    # A tied LM head reads the embedding table, which counts once.
    tied_config = qwen3_config()
    tied_config.tie_word_embeddings = True
    tied = Qwen3ForCausalLM(tied_config)
    assert tied.lm_head.weight.data_ptr() == tied.model.embed_tokens.weight.data_ptr()
    tied_weights = sum(p.numel() * p.element_size() for p in tied.parameters())
    assert agentic_decode._decode_bytes_per_token(tied, "static_v3", 8, 8) == (
        tied_weights + kv
    )
    # With key_value_lengths, static_v3 reads only the written slots, like the
    # other caches: on average isl + osl / 2 = 12 of the 16 positions.
    lengths = agentic_decode._decode_bytes_per_token(
        qwen3, "static_v3", 8, 8, attention="lengths"
    )
    assert lengths == weights + kv * 12 // 16
    assert agentic_decode._decode_bytes_per_token(qwen3, "static_v2", 8, 8) == lengths
    # GPT-2 configs have no num_key_value_heads or num_hidden_layers.
    gpt2 = GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=32, n_head=2))
    assert agentic_decode._decode_bytes_per_token(gpt2, "static_v2", 8, 8) > 0


# Stand-ins for the token ids of TEXT (50 tokens), PASSAGE (60 tokens) and
# CODA (20 tokens).
TEXT_IDS = torch.arange(100, 150)
PASSAGE_IDS = torch.arange(0, 60)
CODA_IDS = torch.arange(200, 220)


def layout(isl, osl, prompt):
    return agentic_decode.layout_tokens(
        TEXT_IDS, PASSAGE_IDS, CODA_IDS, isl, osl, prompt
    )


def test_the_text_layout_is_the_text_repeated():
    # The 128-token workload's layout: no repeat needed, so auto keeps it.
    for prompt in ("auto", "text"):
        assert torch.equal(layout(30, 20, prompt), TEXT_IDS)
    assert torch.equal(layout(100, 20, "text"), TEXT_IDS.repeat(3)[:120])


def test_the_recall_layout_recalls_the_opening_passage_then_continues_unseen():
    isl, cue, recalled = 200, agentic_decode.RECALL_CUE, agentic_decode.RECALL_DECODE
    osl = recalled + 8
    for prompt in ("auto", "recall"):
        tokens = layout(isl, osl, prompt)
        assert tokens.numel() == isl + osl
        assert torch.equal(tokens[:60], PASSAGE_IDS)
        # Between the two passages only the filler text: nothing to copy from
        # nearby, so decode must attend back to the start of the prompt.
        filler = tokens[60 : isl - cue]
        assert bool(((filler >= 100) & (filler < 150)).all())
        assert torch.equal(tokens[isl - cue : isl], PASSAGE_IDS[:cue])
        # Decode copies the passage, then goes on with the coda, which is in
        # no prompt: its tokens keep a natural uncertainty.
        assert torch.equal(
            tokens[isl : isl + recalled], PASSAGE_IDS[cue : cue + recalled]
        )
        assert torch.equal(tokens[isl + recalled :], CODA_IDS[:8])
        assert not bool(torch.isin(tokens[:isl], CODA_IDS).any())
    # Decode that outlasts the coda continues with the text; one that ends
    # early is all recall.
    tokens = layout(isl, recalled + 30, "recall")
    assert torch.equal(tokens[isl + recalled :], torch.cat([CODA_IDS, TEXT_IDS[:10]]))
    assert torch.equal(layout(isl, 4, "recall")[isl:], PASSAGE_IDS[cue : cue + 4])


def test_a_prompt_too_short_for_recall_falls_back_or_is_refused():
    isl = 60 + agentic_decode.RECALL_CUE - 1
    assert torch.equal(layout(isl, 20, "auto"), TEXT_IDS.repeat(2)[: isl + 20])
    with pytest.raises(ValueError, match="recall layout needs isl >= 76"):
        layout(isl, 20, "recall")


def test_the_workload_texts_keep_coda_out_of_the_prompt():
    # Word-level stand-in for the tokenizer: no three-word run of the coda
    # occurs in the passage or the filler text, so decode cannot copy it.
    def trigrams(text):
        words = text.split()
        return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}

    prompt = trigrams(agentic_decode.PASSAGE) | trigrams(agentic_decode.TEXT * 2)
    assert not trigrams(agentic_decode.CODA) & prompt
    # The coda continues the recalled passage mid-sentence.
    assert agentic_decode.CODA.startswith(" and,")


def test_long_contexts_get_an_eager_prefill_and_length_limited_attention(
    tmp_path, monkeypatch
):
    import torch_tensorrt.dynamo

    # The build-route knob probes TensorRT for Global Performance Tuning.
    monkeypatch.setattr(
        torch_tensorrt.dynamo, "is_global_perf_tuning_available", lambda: True
    )
    qwen3_config().save_pretrained(tmp_path / "qwen3")
    model = str(tmp_path / "qwen3")
    short = agentic_decode.DecodeWorkload(model, isl=128, osl=64)
    assert not short.eager_prefill and short.v3_attention == "mask"
    long = agentic_decode.DecodeWorkload(model, isl=8192, osl=64)
    assert long.eager_prefill and long.v3_attention == "lengths"
    engine = agentic_decode.DecodeWorkload(model, isl=8192, osl=64, prefill="engine")
    assert not engine.eager_prefill
    eager = agentic_decode.DecodeWorkload(model, isl=128, osl=64, prefill="eager")
    assert eager.eager_prefill
    # Only long caches say how static_v3 attends; the short knobs are as published.
    cache = {knob.name: knob for knob in short.knobs()}["workload.cache"]
    assert "key_value_lengths" not in cache.description
    cache = {knob.name: knob for knob in long.knobs()}["workload.cache"]
    assert "key_value_lengths" in cache.description
    # Same routes at both lengths; only the long knob says which one
    # builds the same static_v3 engine as no route.
    short_routes = {knob.name: knob for knob in short.knobs()}["engine.build_route"]
    long_routes = {knob.name: knob for knob in long.knobs()}["engine.build_route"]
    assert short_routes.values == long_routes.values == agentic_decode.BUILD_ROUTES
    assert "same static_v3 engine" not in short_routes.description
    assert "same static_v3 engine" in long_routes.description
    for kwargs, match in (
        ({"prompt": "lorem"}, "prompt"),
        ({"prefill": "x"}, "prefill"),
    ):
        with pytest.raises(ValueError, match=match):
            agentic_decode.DecodeWorkload(model, isl=8, osl=8, **kwargs)


@pytest.mark.parametrize(
    "values, match",
    [
        ({"bench.backend": "onnx"}, "bench.backend must be one of"),
        (
            {"bench.backend": "inductor", "bench.inductor": {"no_such_option": 1}},
            "Unknown torch._inductor.config option 'no_such_option'",
        ),
        (
            {"bench.backend": "inductor", "bench.inductor": {"triton.no_such": 1}},
            "Unknown torch._inductor.config option 'triton.no_such'",
        ),
    ],
)
def test_bench_values_are_checked_before_anything_is_loaded(tmp_path, values, match):
    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    with pytest.raises(ValueError, match=match):
        workload.build({"engine.precision": "fp16", **values})


def test_nested_inductor_options_are_accepted(tmp_path):
    import torch._inductor.config as inductor_config

    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    # A dotted option is applied (to its current value, so other tests see no
    # change), then the build goes on to load the model, which has no weights.
    current = inductor_config.triton.cudagraph_trees
    with pytest.raises(OSError, match="no file named"):
        workload.build(
            {
                "engine.precision": "fp16",
                "bench.backend": "inductor",
                "bench.inductor": {"triton.cudagraph_trees": current},
            }
        )
    assert inductor_config.triton.cudagraph_trees == current


def test_torch_candidates_rewind_the_static_cache_on_every_call():
    isl, osl, vocab = 6, 3, 5
    layers = [
        SimpleNamespace(
            cumulative_length=torch.tensor(0),
            keys=torch.ones(1, 2, isl + osl, 4),
            values=torch.ones(1, 2, isl + osl, 4),
        )
        for _ in range(2)
    ]
    cache = SimpleNamespace(layers=layers)
    seen = []

    def step(input_ids, position_ids):
        layer = cache.layers[0]
        seen.append(
            (
                int(input_ids),
                int(position_ids),
                int(layer.cumulative_length),
                float(layer.keys[0, 0, isl:].sum()),
            )
        )
        for layer in cache.layers:
            layer.cumulative_length += 1
            layer.keys[:, :, int(position_ids)] = 7.0
        return torch.full((1, 1, vocab), float(position_ids), dtype=torch.float16)

    tokens = torch.arange(isl + osl) + 10
    candidate = agentic_decode._TorchDecodeCandidate(
        step,
        cache,
        tokens=tokens,
        isl=isl,
        osl=osl,
        cudagraphs=False,
        clear_future=True,
    )
    for _ in range(2):
        seen.clear()
        logits = candidate()
        assert logits.dtype == torch.float32 and logits.shape == (osl, vocab)
        assert logits[:, 0].tolist() == [6.0, 7.0, 8.0]
        # Teacher-forced inputs at positions isl.., from a rewound cache whose
        # unwritten slots are clear (the second call too).
        assert [s[:3] for s in seen] == [(16, 6, 6), (17, 7, 7), (18, 8, 8)]
        assert seen[0][3] == 0.0
    assert candidate.units == osl and candidate.items == 1


@pytest.fixture
def static_cache_v3():
    # Importing the module registers its lowering pass for the whole process;
    # remove it afterwards so other tests compile without it.
    import importlib

    sys.modules.pop("static_cache_v3", None)
    module = importlib.import_module("static_cache_v3")
    yield module
    passes = ATEN_POST_LOWERING_PASSES.passes
    for index in reversed(range(len(passes))):
        if getattr(passes[index], "__name__", "") == "insert_static_cache_v3":
            del passes[index]
    sys.modules.pop("static_cache_v3", None)


def test_length_limited_attention_ignores_the_unwritten_slots(static_cache_v3):
    torch.manual_seed(0)
    query = torch.randn(1, 4, 1, 16)
    keys, values = torch.randn(1, 2, 32, 16), torch.randn(1, 2, 32, 16)
    out = torch.ops.static_cache_v3.attention(
        query, keys, values, torch.tensor([5]), 0.25, True
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, keys[:, :, :5], values[:, :, :5], scale=0.25, enable_gqa=True
    )
    torch.testing.assert_close(out, expected)
    keys[:, :, 5:], values[:, :, 5:] = 1e4, -1e4
    again = torch.ops.static_cache_v3.attention(
        query, keys, values, torch.tensor([5]), 0.25, True
    )
    torch.testing.assert_close(again, expected)
    with pytest.raises(ValueError, match="attention must be one of"):
        static_cache_v3.configure(16, attention="window")


def test_long_caches_default_to_length_limited_attention(static_cache_v3):
    assert static_cache_v3.MASK_ATTENTION_MAX == agentic_decode.MASK_ATTENTION_MAX
    assert static_cache_v3.resolve_attention(192) == "mask"
    assert static_cache_v3.resolve_attention(193) == "lengths"
    assert static_cache_v3.resolve_attention(8256, "mask") == "mask"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        static_cache_v3.configure(192)
        assert static_cache_v3._attention == "mask"
        static_cache_v3.configure(8256)
        assert static_cache_v3._attention == "lengths"
    # TensorRT 11.4 ignores the mask of long caches: asking for it warns.
    with pytest.warns(UserWarning, match="ignores the mask"):
        static_cache_v3.configure(1088, attention="mask")
    assert static_cache_v3._attention == "mask"


class _FakeDecodeEngine(torch.nn.Module):
    """A static_v3-like step over one layer's caches ``[1, 2, slots, 4]``.

    It writes the token id as its key and value at ``position_ids`` and
    returns, as all five logits, the sum of the keys it attends to: the slots
    up to the position, or every slot when ``reads_every_slot``.
    """

    def __init__(self, reads_every_slot: bool) -> None:
        super().__init__()
        self.reads_every_slot = reads_every_slot

    def forward(self, input_ids, position_ids, key_cache, value_cache):
        index = position_ids.reshape(-1)
        new = input_ids.reshape(1, 1, 1, 1).to(key_cache.dtype).expand(1, 2, 1, 4)
        key_cache.index_copy_(2, index, new)
        value_cache.index_copy_(2, index, new)
        keys = key_cache
        if not self.reads_every_slot:
            slots = torch.arange(key_cache.shape[2])
            keys = key_cache * (slots <= index).reshape(1, 1, -1, 1)
        logits = keys.sum(dim=(1, 2, 3)).reshape(1, 1, 1).expand(1, 1, 5)
        return logits, key_cache, value_cache


def _fake_candidate(reads_every_slot, fill_unwritten, isl=6, osl=3):
    module = torch.fx.symbolic_trace(_FakeDecodeEngine(reads_every_slot))
    shapes = {"key_cache": (1, 2, isl + osl, 4), "value_cache": (1, 2, isl + osl, 4)}
    for node in module.graph.nodes:
        if node.op == "placeholder" and node.name in shapes:
            node.meta["val"] = torch.empty(shapes[node.name])
    prompt = torch.arange(1.0, isl + 1).reshape(1, 1, isl, 1).expand(1, 2, isl, 4)
    return agentic_decode._DecodeCandidate(
        module,
        cache="static_v3",
        tokens=torch.arange(isl + osl) + 10,
        isl=isl,
        osl=osl,
        cudagraphs=False,
        pre_allocated_outputs=False,
        prompt_cache=[prompt, prompt],
        fill_unwritten=fill_unwritten,
    )


def test_static_v3_candidates_fill_the_unwritten_slots_with_noise():
    isl = 6
    candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=True)
    candidate.setup()
    keys = candidate._start[0]
    # The prompt's keys from the eager prefill, then noise at their scale,
    # the same for every candidate.
    prompt = torch.arange(1.0, isl + 1).expand(1, 2, isl)
    assert torch.equal(keys[:, :, :isl, 0], prompt)
    noise = keys[:, :, isl:]
    assert bool(noise.isfinite().all()) and bool((noise != 0).all())
    again = _fake_candidate(reads_every_slot=False, fill_unwritten=True)
    again.setup()
    assert torch.equal(again._start[0], keys)
    # An engine that attends only to the written slots never sees the noise:
    # its first call matches its second (which finds the first call's keys).
    first, second = candidate(), candidate()
    assert torch.equal(first, second)
    # Prompt keys 1..6 = 21, plus the decode tokens 16, 17, 18 as they come,
    # times 8 elements per slot.
    assert first[:, 0].tolist() == [8.0 * (21 + 16), 8.0 * (37 + 17), 8.0 * (54 + 18)]
    # One that reads every slot sees the noise on its first call.
    reader = _fake_candidate(reads_every_slot=True, fill_unwritten=True)
    reader.setup()
    assert not torch.equal(reader(), reader())
    # Without the fill (as at 128 tokens), the unwritten slots stay zero.
    plain = _fake_candidate(reads_every_slot=True, fill_unwritten=False)
    plain.setup()
    assert not bool(plain._start[0][:, :, isl:].any())
    assert plain.describe()["cache"] == "static_v3"
    assert candidate.describe()["unwritten_slots"] == "noise"


# ------------------------------------------------- kernels.inductor (hybrid)


def test_qwen3_models_get_the_hybrid_knob_and_every_model_the_step_graph(tmp_path):
    from torch_tensorrt.autotune._knobs import knobs_by_name, unmet_requirements

    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    knobs = knobs_by_name(workload.knobs())
    hybrid = knobs["kernels.inductor"]
    assert hybrid.values == (
        "none",
        "attention-flex",
        "lm-head",
        "attention",
        "attention-math",
    )
    assert hybrid.default == "none"
    # Requirements per value: every value needs the static_v3 cache, FP16 and
    # CUDA graphs; the region values also need the step graph, and
    # attention-flex, which adds no region, runs under either capture.
    assert hybrid.requires_by_value
    every = {
        "engine.precision": "fp16",
        "runtime.cudagraphs": True,
        "workload.cache": "static_v3",
    }
    assert hybrid.requirements("none") == {}
    assert hybrid.requirements("attention-flex") == every
    for value in ("lm-head", "attention", "attention-math"):
        assert hybrid.requirements(value) == {**every, "runtime.step_graph": True}
    # The description counts this model's boundaries (2 layers) and states
    # mechanisms, not measurements.
    assert "2 regions, 3 engines, 4 engine-region boundaries" in hybrid.description
    assert "also require runtime.step_graph" in hybrid.description
    assert "cannot be saved" in hybrid.description
    assert "16-byte aligned" in hybrid.description
    assert not re.search(r"\d\s*(ms|us|%)", hybrid.description)
    step = knobs["runtime.step_graph"]
    assert step.values == (False, True) and step.default is False
    assert dict(step.requires) == {
        "runtime.cudagraphs": True,
        "workload.cache": "static_v3",
    }
    # A session that proposes a hybrid value is told what that value needs
    # (the workload's precision already defaults to fp16).
    assert unmet_requirements(
        {"kernels.inductor": "attention", "workload.cache": "static_v3"},
        "kernels.inductor",
        knobs,
    ) == {"runtime.cudagraphs": True, "runtime.step_graph": True}
    assert unmet_requirements(
        {"kernels.inductor": "attention-flex", "workload.cache": "static_v3"},
        "kernels.inductor",
        knobs,
    ) == {"runtime.cudagraphs": True}
    # The step graph applies to any static_v3 model; the hybrid needs Qwen3.
    GPT2Config(n_layer=1, n_embd=32, n_head=2).save_pretrained(tmp_path / "gpt2")
    other = agentic_decode.DecodeWorkload(str(tmp_path / "gpt2"), isl=8, osl=8)
    names = {knob.name for knob in other.knobs()}
    assert "runtime.step_graph" in names and "kernels.inductor" not in names


def test_configs_get_the_requirements_of_their_own_hybrid_value(tmp_path):
    from torch_tensorrt.autotune._knobs import (
        Knob,
        apply_changes,
        chosen_changes,
        knobs_by_name,
        with_requirements,
    )

    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    knobs = knobs_by_name(workload.knobs())
    engine = {"workload.cache": "static_v3", "runtime.cudagraphs": True}
    # A region value without the step graph is refused before anything is
    # built, so no trial is spent on a build bound to fail; attention-flex
    # is not refused.
    for value in ("lm-head", "attention", "attention-math"):
        with pytest.raises(
            ValueError,
            match=f"kernels.inductor='{value}' requires runtime.step_graph=True",
        ):
            apply_changes(engine, {"kernels.inductor": value}, knobs)
        config = apply_changes(
            engine, {"kernels.inductor": value, "runtime.step_graph": True}, knobs
        )
        assert config["kernels.inductor"] == value
    assert apply_changes(engine, {"kernels.inductor": "attention-flex"}, knobs) == {
        **engine,
        "kernels.inductor": "attention-flex",
    }
    # What a planner's change brings along from the baseline (static_v2, no
    # CUDA graphs), followed transitively: the step graph only for regions.
    baseline = apply_changes(workload.baseline(), {}, knobs)
    bundled = with_requirements(baseline, {"kernels.inductor": "attention"}, knobs)
    assert bundled == {
        "kernels.inductor": "attention",
        "workload.cache": "static_v3",
        "runtime.cudagraphs": True,
        "runtime.step_graph": True,
    }
    apply_changes(baseline, bundled, knobs)
    flex = with_requirements(baseline, {"kernels.inductor": "attention-flex"}, knobs)
    assert flex == {
        "kernels.inductor": "attention-flex",
        "workload.cache": "static_v3",
        "runtime.cudagraphs": True,
    }
    apply_changes(baseline, flex, knobs)
    # The settings that came along do not count as tried.
    assert chosen_changes(bundled, knobs) == {"kernels.inductor": "attention"}
    # A session records the knobs as JSON and rebuilds them when it resumes.
    hybrid = knobs["kernels.inductor"]
    assert Knob.from_json(hybrid.to_json()) == hybrid


def _with_inductor_region(module):
    """``module`` carrying the record of one execute_in_torch(backend="inductor")
    region, which the exporter refuses before it writes anything."""
    from torch_tensorrt.dynamo.region_compilation._types import (
        RegionRecord,
        attach_region_records,
    )

    record = RegionRecord(
        id="attention_0#0", name="attention_0", child_target="x", backend="inductor"
    )
    attach_region_records(module, [record])
    return module


@pytest.mark.parametrize("variant", ["lm-head", "attention", "attention-math"])
def test_region_builds_report_that_they_cannot_be_saved(tmp_path, variant):
    from torch_tensorrt import region
    from torch_tensorrt.autotune._measure import export_unsupported

    candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=False)
    _with_inductor_region(candidate.module)
    candidate.hybrid = {"inductor": variant}
    with pytest.raises(NotImplementedError, match="cannot be saved") as caught:
        candidate.export(tmp_path)
    assert f"kernels.inductor='{variant}'" in str(caught.value)
    assert isinstance(caught.value.__cause__, region.RegionError)
    # The session records it as an export the candidate cannot make.
    assert export_unsupported(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_other_builds_keep_the_exporters_own_errors(tmp_path, monkeypatch):
    import torch_tensorrt
    from torch_tensorrt import region

    # Without a region value the exporter's error is not rewritten.
    for hybrid in ({}, {"inductor": "attention-flex"}):
        candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=False)
        _with_inductor_region(candidate.module)
        candidate.hybrid = dict(hybrid)
        with pytest.raises(region.RegionError, match="is not supported"):
            candidate.export(tmp_path)
    saved = []
    monkeypatch.setattr(
        torch_tensorrt,
        "save",
        lambda module, path, **kwargs: saved.append((module, path, kwargs)),
    )
    candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=False)
    candidate.hybrid = {"inductor": "attention-flex"}
    assert candidate.export(tmp_path) == ["decode.ep"]
    assert saved == [
        (candidate.module, str(tmp_path / "decode.ep"), {"retrace": False})
    ]


HYBRID = {
    "kernels.inductor": "attention",
    "kernels.decode_blocks": "",
    "workload.cache": "static_v3",
    "engine.precision": "fp16",
    "runtime.cudagraphs": True,
    "runtime.step_graph": True,
}


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"runtime.step_graph": False}, "inductor=attention needs runtime.step_graph"),
        (
            {"runtime.step_graph": False, "kernels.inductor": "lm-head"},
            "inductor=lm-head needs runtime.step_graph",
        ),
        (
            {"runtime.step_graph": False, "kernels.inductor": "attention-math"},
            "inductor=attention-math needs runtime.step_graph",
        ),
        ({"workload.cache": "static_v2"}, "runtime.step_graph needs"),
        ({"runtime.cudagraphs": False}, "runtime.step_graph needs"),
        (
            {"runtime.cudagraphs": False, "runtime.step_graph": False},
            "runtime.cudagraphs=True",
        ),
        ({"engine.precision": "bf16"}, "engine.precision=fp16"),
        ({"kernels.inductor": "mlp"}, "kernels.inductor must be one of"),
    ],
)
def test_hybrid_configs_refuse_settings_they_cannot_run_with(tmp_path, changes, match):
    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    with pytest.raises(ValueError, match=match):
        workload.build({**HYBRID, **changes})


@pytest.mark.parametrize("step_graph", [False, True])
def test_hosted_flex_decoding_runs_under_either_capture(tmp_path, step_graph):
    # attention-flex adds no region: the session's per-engine graphs suffice.
    # Its checks pass, and the build goes on to load the model, which has no
    # weights here.
    qwen3_config().save_pretrained(tmp_path / "qwen3")
    workload = agentic_decode.DecodeWorkload(str(tmp_path / "qwen3"), isl=8, osl=8)
    values = {
        **HYBRID,
        "kernels.inductor": "attention-flex",
        "kernels.decode_blocks": "all",
        "runtime.step_graph": step_graph,
    }
    with pytest.raises(OSError, match="no file named"):
        workload.build(values)


def test_a_step_graph_needs_the_static_v3_cache_and_cuda_graphs():
    for cache, cudagraphs in (("static_v2", True), ("static_v3", False)):
        with pytest.raises(ValueError, match="step graph needs"):
            agentic_decode._DecodeCandidate(
                torch.nn.Identity(),
                cache=cache,
                tokens=torch.arange(8),
                isl=4,
                osl=4,
                cudagraphs=cudagraphs,
                pre_allocated_outputs=False,
                step_graph=True,
            )


class _FakeStepGraph:
    """Stands in for _StepGraph on the CPU: warms up, then 'replays' eagerly."""

    instances: list = []

    def __init__(self, module, caches, token, position, warmup=2):
        self.module, self.caches = module, tuple(caches)
        self.token, self.position = token.clone(), position.clone()
        self.captured_at = (int(token.reshape(-1)[0]), int(position.reshape(-1)[0]))
        self.replays = 0
        for _ in range(warmup):
            module(self.token, self.position, *self.caches)
        _FakeStepGraph.instances.append(self)

    def __call__(self, token, position):
        self.token.copy_(token)
        self.position.copy_(position)
        self.replays += 1
        return self.module(self.token, self.position, *self.caches)[0]


def test_a_step_graph_candidate_replays_one_graph_bound_to_the_caches(monkeypatch):
    monkeypatch.setattr(agentic_decode, "_StepGraph", _FakeStepGraph)
    _FakeStepGraph.instances.clear()
    plain = _fake_candidate(reads_every_slot=False, fill_unwritten=True)
    plain.setup()
    expected = plain()
    candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=True)
    candidate.cudagraphs = candidate.step_graph = True
    candidate.setup()
    (graph,) = _FakeStepGraph.instances
    # The graph is bound to the very cache tensors every call starts from.
    assert graph.caches is candidate._start
    # Warm-up wrote the first decode slot, then the noise filled the rest.
    noise = candidate._start[0][:, :, 6:]
    assert bool((noise != 0).all())
    for _ in range(2):
        assert torch.equal(candidate(), expected)
    assert graph.replays == 2 * candidate.osl
    assert candidate.describe()["step_graph"] is True


def test_a_step_graph_prefills_through_its_replays(monkeypatch):
    # With an engine prefill the graph is captured first (at position 0, as
    # the prefill's first step) and the prompt runs through its replays:
    # outside CUDA graphs TensorRT 11.4 reloads plugin modules every enqueue.
    monkeypatch.setattr(agentic_decode, "_StepGraph", _FakeStepGraph)
    _FakeStepGraph.instances.clear()

    def engine_prefill(step_graph):
        candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=True)
        candidate.prompt_cache = None
        candidate.cudagraphs = candidate.step_graph = step_graph
        candidate.setup()
        return candidate

    plain = engine_prefill(False)
    expected = plain()
    candidate = engine_prefill(True)
    (graph,) = _FakeStepGraph.instances
    assert graph.captured_at == (10, 0)
    assert graph.replays == candidate.isl
    assert graph.caches is candidate._start
    # The prompt tokens 10..15 in slots 0..5, then noise.
    keys = candidate._start[0]
    assert keys[0, 0, : candidate.isl, 0].tolist() == [10.0, 11, 12, 13, 14, 15]
    assert torch.equal(keys[:, :, : candidate.isl], plain._start[0][:, :, :6])
    assert bool((keys[:, :, candidate.isl :] != 0).all())
    assert torch.equal(candidate(), expected)
    assert graph.replays == candidate.isl + candidate.osl


def test_hosted_flex_decoding_needs_aligned_caches(monkeypatch):
    candidate = _fake_candidate(reads_every_slot=False, fill_unwritten=False)
    candidate.hybrid = {"inductor": "attention-flex"}
    aligned = candidate._zero_caches()
    # A view one FP32 element in: 4 bytes past a 16-byte boundary.
    storage = torch.zeros(aligned[0].numel() + 1)
    shifted = storage[1:].view(aligned[0].shape)
    assert shifted.data_ptr() % 16
    monkeypatch.setattr(candidate, "_zero_caches", lambda: [shifted, aligned[1]])
    with pytest.raises(ValueError, match="KV cache input 0 is not 16-byte aligned"):
        candidate.setup()
    # Other variants bind what they get; aligned caches pass.
    monkeypatch.setattr(candidate, "_zero_caches", lambda: aligned)
    candidate.setup()


def test_hybrid_attention_bodies_ignore_the_unwritten_slots():
    import qwen3_regions

    torch.manual_seed(0)
    heads, kv_heads, dim, slots, position = 4, 2, 16, 300, 200
    q = torch.randn(1, heads, 1, dim)
    keys = torch.randn(1, kv_heads, slots, dim)
    values = torch.randn(1, kv_heads, slots, dim)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q,
        keys[:, :, : position + 1],
        values[:, :, : position + 1],
        scale=0.25,
        enable_gqa=True,
    )
    keys[:, :, position + 1 :], values[:, :, position + 1 :] = 1e4, -1e4
    pos = torch.tensor([[position]])
    out = qwen3_regions.decode_attention(q, keys, values, pos, 0.25)
    torch.testing.assert_close(out, expected)
    # The hosted flex-decoding kernels' reference chain: split-KV parts, their
    # max and sum, then the output.
    mask = qwen3_regions.flex_block_mask(pos, slots)
    big_m, big_l, big_acc = qwen3_regions.flex_split_reference(
        q, keys, values, *mask, scale=0.25, splits=7
    )
    assert big_m.shape == big_l.shape == (1, 7, heads, 1)
    assert big_acc.shape == (1, 7, heads, 1, dim)
    global_m, global_l = qwen3_regions.flex_lse_reference(big_m, big_l)
    flex = qwen3_regions.flex_out_reference(big_acc, global_m, big_m, global_l, q.dtype)
    torch.testing.assert_close(flex, expected)
    # The attention regions' operator (FlexAttention's eager reference off
    # CUDA) over the same block mask.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        region = torch.ops.qwen3_regions.flex_decode(q, keys, values, *mask, 0.25)
    assert region.is_contiguous()
    torch.testing.assert_close(region, expected)


def test_flex_decoding_reaches_inductor_regions_as_an_opaque_operator():
    # Region capture rejects higher-order operators, as export captures
    # flex_attention, inside a scope; the custom operator around it is one
    # ordinary node, which the region keeps.
    import importlib.util

    import qwen3_regions
    from torch.nn.attention.flex_attention import flex_attention
    from torch_tensorrt import region
    from torch_tensorrt.dynamo.region_compilation import normalize_region_scopes
    from torch_tensorrt.region._errors import RegionCaptureError
    from torch_tensorrt.region._session import RegionCompilationSession

    class Step(torch.nn.Module):
        def __init__(self, opaque):
            super().__init__()
            self.opaque = opaque

        def forward(self, q, k, v, position):
            mask = qwen3_regions.flex_block_mask(position, k.shape[2])
            with region.execute_in_torch(name="attention_0", backend="inductor"):
                if self.opaque:
                    out = torch.ops.qwen3_regions.flex_decode.default(
                        q, k, v, *mask, 0.25
                    )
                else:
                    out = flex_attention(q, k, v, scale=0.25, enable_gqa=True)
            return out + 1

    inputs = (
        torch.randn(1, 4, 1, 16),
        torch.randn(1, 2, 160, 16),
        torch.randn(1, 2, 160, 16),
        torch.tensor([[150]]),
    )
    with warnings.catch_warnings(), RegionCompilationSession(strict=True):
        warnings.simplefilter("ignore")
        program = torch.export.export(Step(True), inputs, strict=True)
        captured = normalize_region_scopes(program)
        rejected = torch.export.export(Step(False), inputs, strict=True)
        with pytest.raises(RegionCaptureError, match="flex_attention"):
            normalize_region_scopes(rejected)
    (body,) = [
        module
        for name, module in captured.graph_module.named_children()
        if name.startswith("_ttrt_region_body")
    ]
    targets = [n.target for n in body.graph.nodes if n.op == "call_function"]
    assert torch.ops.qwen3_regions.flex_decode.default in targets
    # An edited decode_launch file is a copy of qwen3_regions: it finds the operator
    # registered and reuses it.
    spec = importlib.util.spec_from_file_location(
        "qwen3_regions_copy", qwen3_regions.__file__
    )
    copy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(copy)
    assert copy.INDUCTOR_VARIANTS == qwen3_regions.INDUCTOR_VARIANTS


@pytest.mark.parametrize(
    "position, full, partial",
    [(0, 0, 1), (126, 0, 1), (127, 1, 0), (200, 1, 1), (255, 2, 0), (299, 2, 1)],
)
def test_the_flex_block_mask_has_full_blocks_then_the_positions_block(
    position, full, partial
):
    import qwen3_regions

    kv_num, kv_idx, full_num, full_idx, last = qwen3_regions.flex_block_mask(
        torch.tensor([[position]]), slots=300, block=128
    )
    assert kv_idx.shape == full_idx.shape == (1, 1, 1, 3)
    mask = (kv_num, kv_idx, full_num, full_idx, last)
    assert all(t.dtype == torch.int32 for t in mask)
    assert int(full_num) == full and int(kv_num) == partial
    assert full_idx.flatten().tolist() == [0, 1, 2]
    if partial:
        # The one partial block is the position's own block.
        assert int(kv_idx.flatten()[0]) == position // 128
    assert int(last) == position


# A miniature of Inductor's flex-decoding output code, in the shape the
# parser reads: three kernel sources and their launches.
_FLEX_CODE = """
triton_tem_fused_flex_0 = async_compile.triton('triton_tem_fused_flex_0', '''
@triton.jit
def triton_tem_fused_flex_0(arg_Q, arg_K, arg_V, arg_M, arg_L, arg_KV_NUM_BLKS, \
arg_KV_IDX, arg_FULL_KV_NUM_BLKS, arg_FULL_KV_IDX, in_ptr9, out_ptr0):
    pass
''', device_str='cuda')
triton_per_fused_flex_1 = async_compile.triton('triton_per_fused_flex_1', '''
def triton_per_fused_flex_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, \
XBLOCK):
    pass
''', device_str='cuda')
triton_per_fused_flex_2 = async_compile.triton('triton_per_fused_flex_2', '''
def triton_per_fused_flex_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, \
r0_numel, XBLOCK):
    pass
''', device_str='cuda')
def call(args):
    triton_tem_fused_flex_0.run(arg4_1, arg5_1, arg6_1, buf10, buf11, arg3_1, arg0_1, \
arg1_1, arg2_1, arg7_1, buf12, 1, 8, 42, stream=raw_stream0)
    triton_per_fused_flex_1.run(buf10, buf11, buf15, buf17, 16, 42, stream=raw_stream0)
    triton_per_fused_flex_2.run(buf12, buf15, buf10, buf17, buf19, 2048, 42, \
stream=raw_stream0)
"""


def test_inductor_flex_decoding_code_is_parsed_or_refused():
    import qwen3_regions

    parsed = qwen3_regions.parse_flex_decoding_code(_FLEX_CODE)
    assert parsed["names"] == {
        "split": "triton_tem_fused_flex_0",
        "lse": "triton_per_fused_flex_1",
        "out": "triton_per_fused_flex_2",
    }
    assert parsed["grid"] == (1, 8, 42)
    assert parsed["numel"] == {"lse": (16, 42), "out": (2048, 42)}
    lse_source = parsed["sources"]["triton_per_fused_flex_1"]
    assert "def triton_per_fused_flex_1(" in lse_source
    # Anything else Inductor might emit (more kernels, another dataflow) is
    # refused rather than hosted wrongly.
    for broken in (
        _FLEX_CODE.replace("buf10, buf11, buf15", "buf11, buf10, buf15"),
        _FLEX_CODE.replace("buf12, buf15, buf10, buf17", "buf12, buf17, buf10, buf15"),
        _FLEX_CODE.replace("triton_per_fused_flex_2.run(", "triton_red_fused_x_3.run("),
        _FLEX_CODE.replace("in_ptr9, out_ptr0", "in_ptr9, in_ptr10, out_ptr0"),
    ):
        with pytest.raises(RuntimeError, match="Unexpected|lacks"):
            qwen3_regions.parse_flex_decoding_code(broken)


@pytest.mark.parametrize(
    "variant", ["attention", "attention-math", "attention-flex", "lm-head"]
)
def test_the_hybrid_forward_computes_what_hugging_face_computes(variant):
    # Eagerly, every region runs its body: the explicit-cache step must write
    # the caches and attend only to the written slots, garbage after them.
    import qwen3_regions

    config = qwen3_config()
    config.use_cache = False
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config).eval()
    tokens = torch.randint(0, 256, (12,))
    with torch.no_grad():
        if variant == "lm-head":
            # No explicit caches: a cache-less step (static_cache_v3 adds the
            # cache in compilation), so each token is compared alone.
            expected = torch.cat(
                [
                    model(t.view(1, 1), position_ids=torch.tensor([[i]])).logits[0]
                    for i, t in enumerate(tokens)
                ]
            )
        else:
            expected = model(tokens[None]).logits[0]
    qwen3_regions.patch_qwen3(model, (), inductor=variant)
    blocks = model._qwen3_blocks
    if variant == "attention-flex":
        # What prepare() would generate on a GPU: the eager reference bodies
        # use the split count; the launch settings only reach the plugins.
        launch = {"grid": (1, 1, 1), "num_warps": 1, "num_stages": 1}
        blocks.flex = qwen3_regions.FlexDecoding(
            (), {}, dict.fromkeys(("split", "lse", "out"), launch), 5, {}
        )
    caches = (
        [torch.randn(1, 2, 300, 32) * 3 for _ in range(4)]
        if (blocks.explicit_caches)
        else None
    )
    logits = []
    with torch.no_grad():
        for position in range(tokens.numel()):
            kwargs = {"caches": caches} if caches is not None else {}
            out = model(
                tokens[position].view(1, 1),
                position_ids=torch.tensor([[position]]),
                **kwargs,
            )
            if caches is not None:
                written = list(out.past_key_values)
                assert len(written) == 4
                # Functional writes: the inputs are unchanged, the outputs
                # carry this position.
                new, old = written[0][:, :, position], caches[0][:, :, position]
                assert not torch.equal(new, old)
                caches = written
            logits.append(out.logits[0, -1])
    torch.testing.assert_close(torch.stack(logits), expected, atol=1e-4, rtol=1e-4)
    with pytest.raises(ValueError, match="explicit caches"):
        model(
            tokens[:1].view(1, 1),
            position_ids=torch.tensor([[0]]),
            **({} if caches is not None else {"caches": [torch.zeros(1)]}),
        )


def test_cache_decode_steps_name_every_cache(static_cache_v3):
    import inspect

    class Model(torch.nn.Module):
        def forward(self, input_ids, position_ids=None, caches=None):
            written = [
                static_cache_v3.write(
                    cache, position_ids, input_ids.float().expand(1, 2, 1, 4)
                )
                for cache in caches
            ]
            return SimpleNamespace(logits=input_ids.float(), past_key_values=written)

    step = static_cache_v3.cache_decode_step(Model(), layers=2)
    assert list(inspect.signature(step.forward).parameters) == [
        "input_ids",
        "position_ids",
        "key_cache_0",
        "value_cache_0",
        "key_cache_1",
        "value_cache_1",
    ]
    caches = [torch.zeros(1, 2, 6, 4) for _ in range(4)]
    out = step(torch.tensor([[7]]), torch.tensor([[3]]), *caches)
    assert len(out) == 5 and torch.equal(out[0], torch.tensor([[7.0]]))
    assert bool((out[1][:, :, 3] == 7).all()) and not bool(out[1][:, :, :3].any())
    # write() is functional: the caller's cache is untouched.
    assert not bool(caches[0].any())
    shapes = static_cache_v3.cache_examples(
        qwen3_config(), 16, torch.device("cpu"), torch.float16
    )
    assert len(shapes) == 4 and shapes[0].shape == (1, 2, 16, 32)
