from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from exporters import EdgeConfig
from exporters.executorch import exporter as executorch_exporter
from exporters.executorch import packager as executorch_packager
from exporters.executorch.exporter import EdgeExecuTorchExporter
from exporters.executorch.packager import (
    ActionPackager,
    LanguagePackager,
    VisionPackager,
)
from exporters.spec import ComponentBundle, EdgeSpec


def _bundle(model_type: str) -> ComponentBundle:
    value = torch.ones(1)
    return ComponentBundle(
        module=nn.Identity(),
        trace_args=(value,),
        save_args=(value,),
        input_names=["x"],
        output_names=["y"],
        model_type=model_type,
    )


class _ExporterSpec(EdgeSpec):
    def __init__(self, model_types: tuple[str, ...]) -> None:
        self.model_types = model_types

    def prepare_sample_inputs(self, model, raw, config):
        del model, config
        return dict(raw)

    def capture_eager_outputs(self, model, sample, config, bench=None):
        del model, sample, config, bench
        return {}

    def prepare(self, model, sample, config):
        del model, sample, config
        return {
            f"component_{index}": _bundle(model_type)
            for index, model_type in enumerate(self.model_types)
        }

    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]):
        del engines, sample
        return None

    def apply_patches(self, model=None):
        del model
        return nullcontext()


class _RecordingPackager:
    def __init__(self, *program_names: str) -> None:
        self.program_names = program_names
        self.prepared: list[ComponentBundle] = []
        self.packaged: list[Path] = []

    def prepare_bundle(self, bundle, spec, config):
        del spec, config
        self.prepared.append(bundle)
        return bundle

    def package(self, engine_path, bundle, output_dir, *, device_id):
        del bundle, device_id
        self.packaged.append(engine_path)
        programs = {}
        for name in self.program_names:
            path = output_dir / f"{name}.pte"
            path.write_bytes(b"pte")
            programs[name] = path
        return programs


@pytest.mark.unit
def test_executorch_exporter_compiles_once_and_writes_manifest(tmp_path, monkeypatch):
    spec = _ExporterSpec(("vit", "language", "action"))
    monkeypatch.setattr(executorch_exporter, "get_edge_spec", lambda *args: spec)

    compile_calls = []

    def fake_compile(bundle, *, name, engine_dir, trt_settings):
        del bundle, trt_settings
        compile_calls.append(name)
        path = engine_dir / name
        path.mkdir(parents=True)
        return str(path), (torch.ones(1),), 1.0

    monkeypatch.setattr(executorch_exporter, "compile_component", fake_compile)

    vision = _RecordingPackager("vision")
    language = _RecordingPackager("language_prefill", "language_decode")
    action = _RecordingPackager("action")
    exporter = EdgeExecuTorchExporter(
        {
            "vit": vision,
            "language": language,
            "action": action,
        }
    )

    programs = exporter.export(
        nn.Identity(),
        {"x": torch.ones(1)},
        EdgeConfig(model_type="test"),
        output_dir=tmp_path,
    )

    assert compile_calls == ["component_0", "component_1", "component_2"]
    assert len(vision.prepared) == len(vision.packaged) == 1
    assert len(language.prepared) == len(language.packaged) == 1
    assert len(action.prepared) == len(action.packaged) == 1
    assert set(programs) == {
        "vision",
        "language_prefill",
        "language_decode",
        "action",
    }

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert set(manifest["engines"]) == {
        "component_0",
        "component_1",
        "component_2",
    }
    assert set(manifest["programs"]) == set(programs)
    assert exporter.manifest_path == tmp_path / "manifest.json"


@pytest.mark.unit
def test_executorch_exporter_rejects_unregistered_packager(tmp_path, monkeypatch):
    spec = _ExporterSpec(("unknown",))
    monkeypatch.setattr(executorch_exporter, "get_edge_spec", lambda *args: spec)

    with pytest.raises(ValueError, match="No ExecuTorch packager registered"):
        EdgeExecuTorchExporter({}).export(
            nn.Identity(),
            {"x": torch.ones(1)},
            EdgeConfig(model_type="test"),
            output_dir=tmp_path,
        )


class _DynamicSpec(_ExporterSpec):
    def __init__(self) -> None:
        super().__init__(())
        self.dynamic_args: tuple[Any, ...] | None = None

    def create_dynamic_shapes(
        self,
        input_names,
        trace_args,
        *,
        max_seq_len,
    ):
        assert input_names[3] == "kvcache_start_index"
        assert max_seq_len == 16
        self.dynamic_args = trace_args
        return ("dynamic-specs",)


def _language_bundle() -> ComponentBundle:
    batch_size = 2
    sequence_length = 4
    hidden_size = 8
    values = (
        torch.randn(batch_size, sequence_length, hidden_size),
        torch.randn(16, 2, 1, 4),
        torch.full((batch_size,), sequence_length, dtype=torch.int32),
        torch.empty(0, dtype=torch.int32),
        torch.full((batch_size, 1), sequence_length - 1, dtype=torch.int64),
        torch.zeros(3, batch_size, sequence_length, hidden_size),
        torch.zeros(batch_size, 2, 1, 16, 4),
    )
    return ComponentBundle(
        module=nn.Identity(),
        trace_args=values,
        save_args=values,
        execute_args=values,
        input_names=[
            "inputs_embeds",
            "rope_rotary_cos_sin",
            "context_lengths",
            "kvcache_start_index",
            "last_token_ids",
            "ds_stack",
            "past_key_values_0",
        ],
        output_names=["logits", "lm_hidden_states", "prefix_k", "prefix_v"],
        model_type="language",
    )


@pytest.mark.unit
def test_language_packager_normalizes_kv_start_for_both_profiles():
    spec = _DynamicSpec()
    prepared = LanguagePackager().prepare_bundle(
        _language_bundle(),
        spec,
        EdgeConfig(max_seq_len=8),
    )

    for values in (
        prepared.trace_args,
        prepared.save_args,
        prepared.execute_args,
    ):
        assert values is not None
        assert values[3].shape == (2,)
        assert values[3].dtype == torch.int32
        assert torch.count_nonzero(values[3]) == 0
    assert prepared.input_specs == ("dynamic-specs",)
    assert spec.dynamic_args is prepared.trace_args


@pytest.mark.unit
def test_language_packager_creates_prefill_and_decode_programs(tmp_path, monkeypatch):
    bundle = LanguagePackager().prepare_bundle(
        _language_bundle(),
        _DynamicSpec(),
        EdgeConfig(max_seq_len=16),
    )
    artifacts = []
    saved_prefill = []
    saved_decode = []

    def fake_build(engine_path, **kwargs):
        artifacts.append((engine_path, kwargs))
        return object()

    monkeypatch.setattr(executorch_packager, "build_language_artifact", fake_build)
    monkeypatch.setattr(
        executorch_packager,
        "save_language_prefill_pte",
        lambda artifact, inputs, path: saved_prefill.append((inputs, path)),
    )
    monkeypatch.setattr(
        executorch_packager,
        "call_engine",
        lambda *args: (
            torch.zeros(2, 1, 8),
            torch.zeros(2, 1, 8),
            torch.zeros(3, 2, 1, 4, 4),
            torch.zeros(3, 2, 1, 4, 4),
        ),
    )
    monkeypatch.setattr(
        executorch_packager,
        "save_language_decode_pte",
        lambda artifact, inputs, path: saved_decode.append((inputs, path)),
    )

    programs = LanguagePackager().package(
        tmp_path / "language",
        bundle,
        tmp_path,
        device_id=0,
    )

    assert set(programs) == {"language_prefill", "language_decode"}
    assert len(saved_prefill) == len(saved_decode) == 1
    decode_inputs = saved_decode[0][0]
    assert decode_inputs[0].shape == (2, 1, 8)
    assert decode_inputs[2].tolist() == [5, 5]
    assert decode_inputs[3].tolist() == [4, 4]
    assert decode_inputs[5].shape == (3, 2, 1, 8)
    assert artifacts[0][1]["runner"] == "llm_prefill"
    assert artifacts[1][1]["runner"] == "llm_decode"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("packager", "build_name", "save_name", "program_name", "input_count"),
    [
        (VisionPackager(), "build_vision_artifact", "save_vision_pte", "vision", 1),
        (ActionPackager(), "build_action_artifact", "save_action_pte", "action", 6),
    ],
)
def test_single_program_packagers(
    tmp_path,
    monkeypatch,
    packager,
    build_name,
    save_name,
    program_name,
    input_count,
):
    saved = []
    monkeypatch.setattr(
        executorch_packager, build_name, lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        executorch_packager,
        save_name,
        lambda artifact, inputs, path: saved.append((inputs, path)),
    )
    values = tuple(torch.ones(1) for _ in range(input_count))
    bundle = ComponentBundle(
        module=nn.Identity(),
        trace_args=values,
        save_args=values,
        input_names=[f"input_{index}" for index in range(input_count)],
        output_names=["output"],
        model_type=program_name,
    )

    programs = packager.package(
        tmp_path / program_name,
        bundle,
        tmp_path,
        device_id=0,
    )

    assert programs == {program_name: tmp_path / f"{program_name}.pte"}
    assert len(saved) == 1
