from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
import torch.nn as nn
from exporters import EdgeConfig, EdgeExporter, register_edge_spec
from exporters.ops import call_engine
from exporters.spec import ComponentBundle, EdgeSpec, registered_specs


@register_edge_spec("dummy_edge")
class DummySpec(EdgeSpec):
    components = ("language",)

    def prepare_sample_inputs(self, model, raw, config):
        return {"x": raw["x"]}

    def prepare(self, name, model, sample, upstream, config) -> ComponentBundle:
        x = sample["x"]
        return ComponentBundle(
            module=model.eval(),
            trace_args=(x,),
            save_args=(x,),
            input_names=["x"],
            output_names=["y"],
            model_type="dummy",
            engine_file="language.engine",
        )

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
def test_edge_exporter_dryrun_runtime(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    sample = {"x": torch.randn(2, 4)}
    exporter = EdgeExporter()
    runtime = exporter.export(
        model,
        sample,
        EdgeConfig(
            dryrun=True,
            skip_runtime_export=True,
            model_type="dummy_edge",
            engine_dir=tmp_path,
        ),
    )
    assert "language" in exporter.engines
    assert (tmp_path / "language" / "config.json").is_file()
    with torch.no_grad():
        got = runtime(x=sample["x"])
        expected = model(sample["x"])
    torch.testing.assert_close(got, expected)


@pytest.mark.unit
def test_edge_exporter_dryrun_exported_program(tmp_path):
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    # Packing tensors are intermediates, not graph leaves.
    sample = {"x": torch.randn(2, 4) + 1}
    exporter = EdgeExporter()
    program = exporter.export(
        model,
        sample,
        EdgeConfig(
            dryrun=True,
            model_type="dummy_edge",
            engine_dir=tmp_path,
        ),
    )
    assert program is not None
    with torch.no_grad():
        out = program.module()(x=sample["x"])
        expected = model(sample["x"])
    torch.testing.assert_close(out, expected)


class _NativeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, hidden_states, **kwargs):
        raise TypeError("cannot unpack non-iterable NoneType object")


class _PluginAttn(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.linear = inner.linear

    def forward(self, hidden_states, **kwargs):
        return self.linear(hidden_states)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _NativeAttn()


class _PatchedWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = _Layer()

    def forward(self, x):
        return self.layer.self_attn(x, rope_rotary_cos_sin=x)


@register_edge_spec("patch_edge")
class _PatchSpec(EdgeSpec):
    components = ("language",)

    def prepare_sample_inputs(self, model, raw, config):
        return {"x": raw["x"]}

    def prepare(self, name, model, sample, upstream, config) -> ComponentBundle:
        x = sample["x"]

        def _patch(mod):
            orig = mod.layer.self_attn
            mod.layer.self_attn = _PluginAttn(orig).eval()
            return [(mod.layer, orig)]

        return ComponentBundle(
            module=model.eval(),
            trace_args=(x,),
            save_args=(x,),
            input_names=["x"],
            output_names=["y"],
            patch_fn=_patch,
            model_type="dummy",
            engine_file="language.engine",
        )

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]):
        return call_engine(engines["language"], "language", sample["x"])[0]


@pytest.mark.unit
def test_edge_exporter_dryrun_keeps_attention_patch(tmp_path):
    """Language wrappers pass plugin kwargs; native HF attention cannot run them."""
    torch.manual_seed(0)
    model = _PatchedWrapper()
    sample = {"x": torch.randn(2, 4)}
    exporter = EdgeExporter()
    runtime = exporter.export(
        model,
        sample,
        EdgeConfig(
            dryrun=True,
            skip_runtime_export=True,
            model_type="patch_edge",
            engine_dir=tmp_path,
        ),
    )
    with torch.no_grad():
        got = runtime(x=sample["x"])
    assert got.shape == (2, 4)


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
    assert any("SiglipAttention.forward" in p for p in paths)
    assert any("Qwen3Attention.forward" in p for p in paths)
    assert any("Eagle25VLForConditionalGeneration.forward" in p for p in paths)
    assert any("Qwen3ForCausalLM.forward" in p for p in paths)
    assert any("GR00TN15.forward" in p for p in paths)
    assert any("FlowmatchingActionHead.forward" in p for p in paths)
    assert any("CategorySpecificLinear.forward" in p for p in paths)


@pytest.mark.unit
def test_nemotron_backend_registers_causal_lm():
    from exporters.models.nemotron.patches import NEMOTRON
    from exporters.plugin.attn_patches import _PATCHES

    paths = [p for p, _ in _PATCHES[NEMOTRON]]
    assert any("NemotronHForCausalLM.forward" in p for p in paths)


@pytest.mark.unit
def test_eagle_vision_patch_extracts_features():
    from exporters.models.groot.patches import (
        _patch_eagle_image_features,
    )

    class Dummy:
        def extract_feature(self, pixel_values):
            return pixel_values + 1

        def forward(self, *args, **kwargs):
            raise AssertionError("full VLM forward should not run")

    Dummy.forward = _patch_eagle_image_features(Dummy.forward)
    pixel_values = torch.zeros(1, 3, 4, 4)
    torch.testing.assert_close(Dummy()(pixel_values), pixel_values + 1)


@pytest.mark.unit
def test_groot_patches_live_eagle_class():
    from exporters.models.groot.patches import apply_groot_patches

    class Eagle:
        def extract_feature(self, pixel_values):
            return pixel_values + 1

        def forward(self, pixel_values, input_ids=None, **kwargs):
            raise AssertionError("unpatched Eagle.forward should not run")

    class Groot:
        def __init__(self):
            self.backbone = type("Backbone", (), {})()
            self.backbone.eagle_model = Eagle()

    class Policy:
        def __init__(self):
            self._groot_model = Groot()

    policy = Policy()
    eagle = policy._groot_model.backbone.eagle_model
    pixel_values = torch.zeros(1, 3, 4, 4)
    with apply_groot_patches(policy):
        torch.testing.assert_close(eagle(pixel_values), pixel_values + 1)


@pytest.mark.unit
def test_eagle_vision_keeps_vlm_forward_with_input_ids():
    from exporters.models.groot.patches import (
        _patch_eagle_image_features,
    )

    class Dummy:
        def extract_feature(self, pixel_values):
            raise AssertionError("extract_feature should not run")

        def forward(self, pixel_values, input_ids=None, **kwargs):
            del kwargs
            return (pixel_values, input_ids)

    Dummy.forward = _patch_eagle_image_features(Dummy.forward)
    pixel_values = torch.zeros(1, 3, 4, 4)
    input_ids = torch.ones(1, 2, dtype=torch.long)
    out = Dummy()(pixel_values, input_ids)
    assert out[1] is input_ids


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
def test_groot_context_keeps_training_forward_without_hidden():
    from exporters.models.groot.patches import (
        _patch_groot_context_projection,
    )

    class Dummy(nn.Module):
        def forward(self, backbone_inputs, action_inputs):
            return backbone_inputs

    Dummy.forward = _patch_groot_context_projection(Dummy.forward)
    assert Dummy()("backbone", "action") == "backbone"


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
