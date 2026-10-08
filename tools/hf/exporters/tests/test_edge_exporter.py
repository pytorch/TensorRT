from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch_tensorrt
import torch_tensorrt.dynamo.runtime as trt_runtime
from exporters import EdgeConfig, EdgeExporter
from exporters import ops as exporter_ops
from exporters import register_edge_spec
from exporters.ops import call_engine
from exporters.spec import ComponentBundle, EdgeSpec, registered_specs
from torch.export import ExportedProgram


def _install_fake_trt(monkeypatch) -> None:
    monkeypatch.setattr(torch_tensorrt.dynamo, "compile", _fake_trt_compile)
    monkeypatch.setattr(
        torch_tensorrt.dynamo,
        "convert_exported_program_to_serialized_trt_engine",
        lambda *args, **kwargs: b"fake-engine",
    )


def _fake_trt_compile(exported, arg_inputs=None, **kwargs):
    del arg_inputs, kwargs
    return exported.module()


@register_edge_spec("dummy_edge")
class DummySpec(EdgeSpec):
    def prepare_sample_inputs(self, model, raw, config):
        return {"x": raw["x"]}

    def capture_eager_outputs(self, model, sample, config, bench=None):
        del config
        from exporters.measure import cuda_ms

        with torch.no_grad():
            y = model(sample["x"])
        if bench is not None:
            bench["language"] = cuda_ms(lambda: model(sample["x"]))
        return {"language": y}

    def prepare(self, model, sample, config) -> dict[str, ComponentBundle]:
        x = sample["x"]
        return {
            "language": ComponentBundle(
                module=model.eval(),
                trace_args=(x,),
                save_args=(x,),
                input_names=["x"],
                output_names=["y"],
                model_type="dummy",
                engine_file="language.engine",
            )
        }

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]):
        return call_engine(engines["language"], "language", sample["x"])[0]


@pytest.mark.unit
def test_builtin_specs_are_registered():
    keys = registered_specs()
    assert "pi05" in keys
    assert "groot" in keys
    assert "nemotron_h" in keys
    assert "dummy_edge" in keys


@pytest.mark.unit
def test_edge_exporter_exported_program(tmp_path, monkeypatch):
    _install_fake_trt(monkeypatch)
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    sample = {"x": torch.randn(2, 4) + 1}
    exporter = EdgeExporter()
    program = exporter.export(
        model,
        sample,
        EdgeConfig(
            model_type="dummy_edge",
            engine_dir=tmp_path,
        ),
    )
    assert isinstance(program, ExportedProgram)
    assert "language" in exporter.engines
    assert (tmp_path / "language" / "config.json").is_file()
    with torch.no_grad():
        out = program.module()(x=sample["x"])
        expected = model(sample["x"])
    torch.testing.assert_close(out, expected)


@pytest.mark.unit
def test_execute_engine_prefers_in_process_module(tmp_path, monkeypatch):
    engine_path = str(tmp_path / "language")
    module = nn.Identity()
    exporter_ops._COMPILED_MODULES[engine_path] = module
    monkeypatch.setattr(
        exporter_ops,
        "_load_serialized_engine",
        lambda *args: pytest.fail("serialized engine should not be loaded"),
    )

    try:
        assert exporter_ops._get_engine(engine_path, "language") is module
    finally:
        exporter_ops._COMPILED_MODULES.pop(engine_path, None)


@pytest.mark.unit
def test_execute_engine_loads_and_caches_serialized_engine(tmp_path, monkeypatch):
    engine_dir = tmp_path / "language"
    engine_dir.mkdir()
    (engine_dir / "language.engine").write_bytes(b"serialized-engine")
    (engine_dir / "config.json").write_text("""{
  "engine_file": "language.engine",
  "input_names": ["x"],
  "output_names": ["y"]
}
""")
    engine_path = str(engine_dir)
    constructor_calls = []
    module = nn.Identity()

    def fake_runtime_module(**kwargs):
        constructor_calls.append(kwargs)
        return module

    monkeypatch.setattr(
        trt_runtime,
        "TorchTensorRTModule",
        fake_runtime_module,
    )

    try:
        first = exporter_ops._get_engine(engine_path, "language")
        second = exporter_ops._get_engine(engine_path, "language")
    finally:
        exporter_ops._COMPILED_MODULES.pop(engine_path, None)

    assert first is module
    assert second is module
    assert constructor_calls == [
        {
            "serialized_engine": b"serialized-engine",
            "input_binding_names": ["x"],
            "output_binding_names": ["y"],
            "name": "language",
        }
    ]


@pytest.mark.unit
def test_attn_patch_attribute_restores():
    from exporters.plugin.attn_patches import patch_attribute

    class Owner:
        def go(self):
            return 1

    def factory(original):
        def go(self):
            return original(self) + 1

        return go

    with patch_attribute(Owner, "go", factory):
        assert Owner().go() == 2
    assert Owner().go() == 1


@pytest.mark.unit
def test_language_attn_keeps_hf_forward_without_rope():
    from exporters.plugin.attn_patches import (
        _patch_language_attention,
    )

    class Dummy(nn.Module):
        def forward(self, hidden_states, past_key_values=None, **kwargs):
            del past_key_values, kwargs
            return hidden_states * 2, None

    Dummy.forward = _patch_language_attention(Dummy.forward)
    hidden = torch.ones(1, 2, 4)
    out, extra = Dummy()(hidden, past_key_values="cache")
    torch.testing.assert_close(out, hidden * 2)
    assert extra is None


@pytest.mark.unit
def test_pi05_backend_registers_vision_and_language():
    from exporters.models.pi05.patches import PI05
    from exporters.plugin.attn_patches import _PATCHES

    paths = [p for p, _ in _PATCHES[PI05]]
    assert any("SiglipAttention.forward" in p for p in paths)
    assert any("PaliGemmaModel.forward" in p for p in paths)
    assert any("GemmaAttention.forward" in p for p in paths)
    assert any("PiGemmaModel.forward" in p for p in paths)
    assert any("PI05Pytorch.forward" in p for p in paths)


@pytest.mark.unit
def test_paligemma_image_features_patch_returns_tensor():
    from exporters.models.pi05.patches import (
        _patch_paligemma_image_features,
    )

    class _Out:
        def __init__(self, last_hidden_state):
            self.last_hidden_state = last_hidden_state

    class Tower(nn.Module):
        def forward(self, pixel_values, **kwargs):
            del kwargs
            return _Out(pixel_values.new_ones(pixel_values.shape[0], 4, 8))

    class Proj(nn.Module):
        def forward(self, hidden):
            return hidden

    class DummyPaliGemma(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_tower = Tower()
            self.multi_modal_projector = Proj()

        def forward(self, *args, **kwargs):
            raise AssertionError("original PaliGemmaModel.forward should not run")

    DummyPaliGemma.forward = _patch_paligemma_image_features(DummyPaliGemma.forward)
    pixel_values = torch.randn(2, 3, 8, 8, dtype=torch.float16)
    out = DummyPaliGemma()(pixel_values)
    assert out.shape == (2, 4, 8)
    assert out.dtype == torch.float16


@pytest.mark.unit
def test_pi05_language_model_keeps_hf_forward_without_rope():
    from exporters.models.pi05.patches import (
        _patch_pi05_language_model,
    )

    class Dummy(nn.Module):
        def forward(self, inputs_embeds=None, past_key_values=None, **kwargs):
            del past_key_values, kwargs
            return inputs_embeds * 2

    Dummy.forward = _patch_pi05_language_model(Dummy.forward)
    hidden = torch.ones(1, 2, 4)
    out = Dummy()(inputs_embeds=hidden, past_key_values="cache")
    torch.testing.assert_close(out, hidden * 2)


@pytest.mark.unit
def test_pi05_action_keeps_training_forward_without_prefix_kv():
    from exporters.models.pi05.patches import (
        _patch_pi05_action_step_forward,
    )

    class Dummy(nn.Module):
        def forward(self, images, img_masks, tokens, masks, actions, noise, time):
            del img_masks, tokens, masks, actions, noise, time
            return images

    Dummy.forward = _patch_pi05_action_step_forward(Dummy.forward)
    assert Dummy()(7, 0, 0, 0, 0, 0, 0) == 7


@pytest.mark.unit
def test_language_attn_plugin_when_rope_present():
    from exporters.plugin.attn_patches import (
        _patch_language_attention,
    )
    from exporters.plugin.plugin_utils import (
        _register_attention_plugin_op,
    )

    _register_attention_plugin_op()

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.num_key_value_heads = 2
            self.head_dim = 4
            self.q_proj = nn.Linear(8, 8)
            self.k_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.o_proj = nn.Linear(8, 8)

        def forward(self, hidden_states, **kwargs):
            raise AssertionError("HF forward should not run for plugin kwargs")

    Dummy.forward = _patch_language_attention(Dummy.forward)
    hidden = torch.randn(1, 3, 8)
    rope = torch.randn(1, 3, 4, dtype=torch.float32)
    kv = torch.zeros(1, 2, 2, 8, 4)
    ctx = torch.tensor([3], dtype=torch.int32)
    start = torch.empty(0, dtype=torch.int32)
    out, present = Dummy()(
        hidden,
        rope_rotary_cos_sin=rope,
        past_key_value=kv,
        ctx_len=ctx,
        kvcache_start_index=start,
    )
    assert out.shape == hidden.shape
    assert present.shape == kv.shape


@pytest.mark.unit
def test_groot_backend_registers_components():
    from exporters.models.groot.patches import GROOT
    from exporters.plugin.attn_patches import _PATCHES

    paths = [p for p, _ in _PATCHES[GROOT]]
    assert any("Qwen3VLTextAttention.forward" in p for p in paths)
    assert any("Qwen3VLTextModel.forward" in p for p in paths)
    assert any("GR00TN17ActionHead.forward" in p for p in paths)
    assert any("groot_n1_7.CategorySpecificLinear.forward" in p for p in paths)
    assert not any("eagle" in p.lower() for p in paths)


@pytest.mark.unit
def test_nemotron_backend_registers_causal_lm():
    from exporters.models.nemotron.patches import NEMOTRON
    from exporters.plugin.attn_patches import _PATCHES

    paths = [p for p, _ in _PATCHES[NEMOTRON]]
    assert any("NemotronHForCausalLM.forward" in p for p in paths)


@pytest.mark.unit
def test_groot_action_keeps_training_forward_without_context():
    from exporters.models.groot.patches import (
        _patch_groot_action_step_forward,
    )

    class Dummy(nn.Module):
        def forward(self, backbone_output, action_input):
            return backbone_output

    Dummy.forward = _patch_groot_action_step_forward(Dummy.forward)
    assert Dummy()("backbone", "action") == "backbone"


@pytest.mark.unit
def test_groot_vision_rope_matches_hf():
    pytest.importorskip("transformers.models.qwen3_vl")
    from exporters.models.groot.vision import vision_rope
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionRotaryEmbedding,
    )

    rotary = Qwen3VLVisionRotaryEmbedding(
        Qwen3VLVisionConfig(hidden_size=64, num_heads=2)
    )
    position_ids = torch.stack(
        torch.meshgrid(torch.arange(4), torch.arange(6), indexing="ij"), dim=-1
    ).reshape(-1, 2)
    ref_cos, ref_sin = rotary(torch.zeros(1), position_ids)
    cos, sin = vision_rope(rotary, position_ids)
    torch.testing.assert_close(cos, ref_cos)
    torch.testing.assert_close(sin, ref_sin)


@pytest.mark.unit
def test_groot_mrope_cache_matches_text_rope_and_extends():
    pytest.importorskip("transformers.models.qwen3_vl")
    from exporters.models.groot.helpers import mrope_rotary_cos_sin
    from exporters.rope import make_normal_rope_rotary_cos_sin
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding,
    )

    cfg = Qwen3VLTextConfig(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [6, 5, 5],
            "mrope_interleaved": True,
        },
    )
    language = nn.Module()
    language.config = cfg
    language.rotary_emb = Qwen3VLTextRotaryEmbedding(cfg)

    # Text-only prompt: all three M-RoPE rows equal -> plain RoPE, incl. the tail.
    prompt = torch.arange(5).view(1, 1, -1).expand(3, 1, -1)
    cache = mrope_rotary_cos_sin(language, prompt, max_seq_len=9)
    ref = make_normal_rope_rotary_cos_sin(
        9, 32, rope_theta=10000.0, device=torch.device("cpu")
    )
    assert cache.shape == (1, 9, 32)
    torch.testing.assert_close(cache, ref)

    # Image grid rows differ; decode rows continue from max(position) + 1.
    grid = torch.tensor([[[0, 1, 1, 1, 3]], [[0, 1, 1, 2, 3]], [[0, 1, 2, 1, 3]]])
    cache = mrope_rotary_cos_sin(language, grid, max_seq_len=7)
    torch.testing.assert_close(cache[:, 5:], ref[:, 4:6])
    assert not torch.allclose(cache[:, 2], cache[:, 3])


@pytest.mark.unit
def test_nemotron_keeps_hf_forward_without_rope():
    from exporters.models.nemotron.patches import (
        _patch_nemotron_causal_lm,
    )

    class Dummy(nn.Module):
        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            del input_ids, kwargs
            return inputs_embeds * 2

    Dummy.forward = _patch_nemotron_causal_lm(Dummy.forward)
    hidden = torch.ones(1, 2, 4)
    out = Dummy()(inputs_embeds=hidden)
    torch.testing.assert_close(out, hidden * 2)


@pytest.mark.unit
def test_category_specific_linear_uses_index_select():
    from exporters.models.groot.patches import (
        _patch_category_specific_linear,
    )

    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.W = nn.Parameter(
                torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
            )
            self.b = nn.Parameter(
                torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
            )

        def forward(self, x, cat_ids):
            raise AssertionError(
                "original CategorySpecificLinear.forward should not run"
            )

    Dummy.forward = _patch_category_specific_linear(Dummy.forward)
    layer = Dummy()
    x = torch.ones(2, 5, 3)
    cat_ids = torch.tensor([1, 0])
    out = layer(x, cat_ids)
    expected = torch.bmm(x, layer.W[cat_ids]) + layer.b[cat_ids].unsqueeze(1)
    torch.testing.assert_close(out, expected)


@pytest.mark.unit
def test_measure_parity_and_bench(capsys):
    from exporters.measure import cuda_ms, parity, print_bench, speedup

    a = torch.ones(2, 2)
    parity("dummy A vs C (TRT)", a, a)
    log = capsys.readouterr().out
    assert "dummy A vs C (TRT)" in log
    assert "close%=100.0" in log
    assert speedup(10.0, 5.0) == "2.000x"
    assert speedup(0.0, 5.0) == "n/a"

    elapsed = cuda_ms(lambda: torch.ones(2, 2).sum(), warmup=1, iters=3)
    assert elapsed >= 0.0

    print_bench({"vision": (10.0, 5.0), "language": (4.0, 2.0)})
    log = capsys.readouterr().out
    assert "vision eager execute: 10.000 ms" in log
    assert "vision trt execute: 5.000 ms" in log
    assert "total speedup: 2.000x" in log
    print_bench({})
    assert capsys.readouterr().out == ""


@pytest.mark.unit
def test_mamba_ragged_ops_match_batch_major():
    from exporters.plugin.mamba import register_mamba_plugin_ops
    from exporters.plugin.plugin_utils import ragged_prefill_metadata

    register_mamba_plugin_ops()
    torch.manual_seed(0)
    batch, seq, conv_dim, kernel = 2, 5, 6, 4
    heads, head_dim, groups, dstate = 4, 3, 2, 5
    lengths = torch.full((batch,), seq, dtype=torch.int32)
    offsets, phase, ctx = ragged_prefill_metadata(batch, seq, torch.device("cpu"))
    rows = torch.arange(batch, dtype=torch.int32)
    meta = (lengths, offsets, rows, phase, ctx)

    x = torch.randn(batch, seq, conv_dim)
    weight = torch.randn(conv_dim, 1, kernel)
    bias = torch.randn(conv_dim)
    conv_state = torch.randn(batch, conv_dim, kernel)
    ref = torch.ops.trt.causal_conv1d(
        x, weight, bias, conv_state, lengths, 1, kernel - 1, 1, conv_dim
    )
    out = torch.ops.trt.causal_conv1d_ragged(
        x.reshape(-1, conv_dim), weight, bias, conv_state, *meta, 1, kernel - 1, 1, conv_dim
    )
    torch.testing.assert_close(out[0], ref[0].reshape(-1, conv_dim))
    torch.testing.assert_close(out[1], ref[1])

    u = torch.randn(batch, seq, heads, head_dim)
    a = -torch.rand(heads)
    b = torch.randn(batch, seq, groups, dstate)
    c = torch.randn(batch, seq, groups, dstate)
    d = torch.randn(heads)
    dt = torch.randn(batch, seq, heads)
    dt_bias = torch.randn(heads)
    state = torch.randn(batch, heads, head_dim, dstate)
    fields = (1, groups, heads, head_dim, dstate)
    ref = torch.ops.trt.update_ssm_state(
        u, a, b, c, d, dt, dt_bias, state, lengths, *fields
    )
    out = torch.ops.trt.update_ssm_state_ragged(
        u.reshape(-1, heads, head_dim),
        a,
        b.reshape(-1, groups, dstate),
        c.reshape(-1, groups, dstate),
        d,
        dt.reshape(-1, heads),
        dt_bias,
        state,
        *meta,
        *fields,
    )
    torch.testing.assert_close(out[0], ref[0].reshape(-1, heads, head_dim))
    torch.testing.assert_close(out[1], ref[1])


@pytest.mark.unit
def test_nemotron_patches_remote_code_class():
    from exporters.models.nemotron.patches import apply_nemotron_patches

    class RemoteNemotronHForCausalLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = None
            self.backbone = nn.Module()
            self.backbone.layers = nn.ModuleList()

        def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
            return inputs_embeds

    model = RemoteNemotronHForCausalLM()
    original = RemoteNemotronHForCausalLM.forward
    with apply_nemotron_patches(model):
        assert RemoteNemotronHForCausalLM.forward is not original
        hidden = torch.ones(1, 2, 4)
        torch.testing.assert_close(model(inputs_embeds=hidden), hidden)
    assert RemoteNemotronHForCausalLM.forward is original
