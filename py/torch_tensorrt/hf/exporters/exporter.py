from __future__ import annotations

import copy
import inspect
import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.export import ExportedProgram
from torch_tensorrt.hf.exporters import ops as _ops  # noqa: F401
from torch_tensorrt.hf.exporters.compile import compile_component
from torch_tensorrt.hf.exporters.config import EdgeConfig
from torch_tensorrt.hf.exporters.runtime import EdgeRuntimeModule
from torch_tensorrt.hf.exporters.spec import get_edge_spec
from transformers.exporters.exporter_dynamo import DynamoExporter

logger = logging.getLogger(__name__)


def _clone_export_kwargs(sample_inputs: MutableMapping[str, Any]) -> dict[str, Any]:
    """Copy example kwargs into graph leaves.

    Vision/language packing produces intermediate tensors. ``copy.deepcopy``
    refuses those (``Only Tensors created explicitly by the user...``).
    """
    cloned: dict[str, Any] = {}
    for key, value in dict(sample_inputs).items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.detach().contiguous().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


class EdgeExporter(DynamoExporter):  # type: ignore[misc]
    def __init__(self) -> None:
        super().__init__()
        self.engines: dict[str, str] = {}
        self.runtime: EdgeRuntimeModule | None = None
        self.sample: dict[str, Any] = {}

    def export(
        self,
        model: nn.Module,
        sample_inputs: MutableMapping[str, Any],
        config: EdgeConfig | dict[str, Any],
    ) -> ExportedProgram | EdgeRuntimeModule:
        if isinstance(config, dict):
            config = EdgeConfig(**config)
        elif not isinstance(config, EdgeConfig):
            raise TypeError(f"Expected EdgeConfig or dict, got {type(config)}")

        spec = get_edge_spec(model, config.model_type)
        names = config.components or spec.components
        if not names:
            raise ValueError(f"{type(spec).__name__} has empty components")

        sample = spec.prepare_sample_inputs(model, sample_inputs, config)

        engine_dir = Path(config.engine_dir or "edge_engines")
        engine_dir.mkdir(parents=True, exist_ok=True)

        engines: dict[str, str] = {}
        upstream: dict[str, Any] = {}

        for name in names:
            module = spec.wrap(name, model, sample, config)
            bundle = spec.prepare(name, model, sample, upstream, config, module)
            engines[name], outs = compile_component(
                module,
                bundle,
                name=name,
                engine_dir=engine_dir,
                dryrun=config.dryrun,
                trt_settings=config.trt_settings,
            )
            upstream.update(spec.capture_upstream(name, outs, sample, bundle))

        runtime = EdgeRuntimeModule(spec, engines)
        runtime_kwargs = _clone_export_kwargs(spec.runtime_kwargs(sample))

        self.engines = engines
        self.runtime = runtime
        self.sample = dict(runtime_kwargs)

        if config.skip_runtime_export:
            return runtime
        return self._export_runtime(runtime, runtime_kwargs, config)

    def _export_runtime(
        self,
        model: nn.Module,
        sample_inputs: MutableMapping[str, Any],
        config: EdgeConfig,
    ) -> ExportedProgram:
        try:
            from transformers.exporters.exporter_dynamo import (
                get_auto_dynamic_shapes,
                patch_forward_signature,
                register_cache_pytrees_for_model,
                reset_model_state,
            )
            from transformers.exporters.utils import prepare_for_export
        except ImportError:
            return torch.export.export(
                model,
                args=(),
                kwargs=_clone_export_kwargs(sample_inputs),
                strict=config.strict,
                dynamic_shapes=config.dynamic_shapes,
            )

        sample_inputs = _clone_export_kwargs(sample_inputs)
        model, sample_inputs, _output_flags = prepare_for_export(model, sample_inputs)
        dynamic_shapes = config.dynamic_shapes

        if config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(sample_inputs)

        if inspect.getmodule(model) is not None:
            try:
                register_cache_pytrees_for_model(model)
            except Exception:
                logger.debug("register_cache_pytrees_for_model skipped", exc_info=True)

        with (
            reset_model_state(model),
            patch_forward_signature(model, sample_inputs),
        ):
            return torch.export.export(
                model,
                args=(),
                kwargs=_clone_export_kwargs(sample_inputs),
                strict=config.strict,
                dynamic_shapes=dynamic_shapes,
                prefer_deferred_runtime_asserts_over_guards=(
                    config.prefer_deferred_runtime_asserts_over_guards
                ),
            )

    def save_engines(self, out_dir: str | Path | None = None) -> dict[str, Path]:
        if not self.engines:
            raise RuntimeError("save_engines() requires export() first")
        if out_dir is None:
            return {name: Path(path) for name, path in self.engines.items()}
        import shutil

        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for name, path in self.engines.items():
            target = dest / name
            if Path(path).resolve() != target.resolve():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(path, target)
            written[name] = target
        return written
