from __future__ import annotations

import copy
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.export import ExportedProgram
from transformers.exporters.exporter_dynamo import (
    DynamoExporter,
    get_auto_dynamic_shapes,
    patch_forward_signature,
)

from . import ops as _ops  # noqa: F401
from .compile import compile_component
from .config import EdgeConfig
from .measure import parity
from .runtime import EdgeRuntimeModule
from .spec import get_edge_spec


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
        self.sample: dict[str, Any] = {}
        self.bench: dict[str, tuple[float, float]] = {}

    def export(
        self,
        model: nn.Module,
        sample_inputs: MutableMapping[str, Any],
        config: EdgeConfig | dict[str, Any],
    ) -> ExportedProgram:
        if isinstance(config, dict):
            config = EdgeConfig(**config)
        elif not isinstance(config, EdgeConfig):
            raise TypeError(f"Expected EdgeConfig or dict, got {type(config)}")

        spec = get_edge_spec(model, config.model_type)
        sample = spec.prepare_sample_inputs(model, sample_inputs, config)
        bundles = spec.prepare(model, sample, config)
        engine_dir = Path(config.engine_dir or "edge_engines")
        engine_dir.mkdir(parents=True, exist_ok=True)

        eager_ms: dict[str, float] = {}
        eager = spec.capture_eager_outputs(model, sample, config, bench=eager_ms)

        engines: dict[str, str] = {}
        self.bench = {}
        with spec.apply_patches(model):
            for name, bundle in bundles.items():
                engines[name], trt_out, trt_ms = compile_component(
                    bundle,
                    name=name,
                    engine_dir=engine_dir,
                    trt_settings=config.trt_settings,
                )
                self.bench[name] = (eager_ms.get(name, 0.0), trt_ms)
                out_name = bundle.parity_output or bundle.output_names[0]
                trt = trt_out[bundle.output_names.index(out_name)]
                ref = eager.get(name)
                if isinstance(ref, torch.Tensor) and isinstance(trt, torch.Tensor):
                    parity(f"{name} eager vs TRT", ref, trt)

        runtime = EdgeRuntimeModule(spec, engines)
        runtime_kwargs = _clone_export_kwargs(spec.runtime_kwargs(sample))
        self.engines = engines
        self.sample = dict(runtime_kwargs)

        dynamic_shapes = config.dynamic_shapes
        if config.dynamic and dynamic_shapes is None:
            dynamic_shapes = get_auto_dynamic_shapes(runtime_kwargs)

        with patch_forward_signature(runtime, runtime_kwargs):
            return torch.export.export(
                runtime,
                args=(),
                kwargs=_clone_export_kwargs(runtime_kwargs),
                strict=config.strict,
                dynamic_shapes=dynamic_shapes,
                prefer_deferred_runtime_asserts_over_guards=(
                    config.prefer_deferred_runtime_asserts_over_guards
                ),
            )
