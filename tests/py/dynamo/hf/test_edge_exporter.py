from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch_tensorrt.hf.exporters import EdgeConfig, EdgeExporter, register_edge_spec
from torch_tensorrt.hf.exporters.ops import call_engine
from torch_tensorrt.hf.exporters.spec import ComponentBundle, EdgeSpec, registered_specs


@register_edge_spec("dummy_edge")
class DummySpec(EdgeSpec):
    components = ("language",)

    def prepare_sample_inputs(self, model, raw, config):
        return {"x": raw["x"]}

    def wrap(self, name, model, sample, config) -> nn.Module:
        return model.eval()

    def prepare(self, name, model, sample, upstream, config, module) -> ComponentBundle:
        x = sample["x"]
        return ComponentBundle(
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

    def wrap(self, name, model, sample, config) -> nn.Module:
        return model.eval()

    def prepare(self, name, model, sample, upstream, config, module) -> ComponentBundle:
        x = sample["x"]

        def _patch(mod):
            orig = mod.layer.self_attn
            mod.layer.self_attn = _PluginAttn(orig).eval()
            return [(mod.layer, orig)]

        return ComponentBundle(
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
