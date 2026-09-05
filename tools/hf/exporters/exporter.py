from __future__ import annotations

import contextlib
import copy
import inspect
import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.export import ExportedProgram
from transformers.exporters.exporter_dynamo import DynamoExporter

from . import ops as _ops  # noqa: F401
from .compile import compile_component
from .config import EdgeConfig
from .runtime import EdgeRuntimeModule
from .spec import get_edge_spec

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
        self.bench: dict[str, tuple[float, float]] = {}
        self._dryrun_patches: contextlib.ExitStack | None = None

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

        # Family spec owns flatten / stitch. The exporter only loops
        # over component names (vision, language, action, ...).
        spec = get_edge_spec(model, config.model_type)
        names = config.components or spec.components
        if not names:
            raise ValueError(f"{type(spec).__name__} has empty components")

        # Caller payload (policy batch, tokenizer ids, ...) -> shared sample
        # dict used by prepare and the stitched runtime.
        sample = spec.prepare_sample_inputs(model, sample_inputs, config)

        engine_dir = Path(config.engine_dir or "edge_engines")
        engine_dir.mkdir(parents=True, exist_ok=True)

        engines: dict[str, str] = {}
        upstream: dict[str, Any] = {}
        self.bench = {}

        def _compile_components() -> None:
            for name in names:
                bundle = spec.prepare(name, model, sample, upstream, config)
                engines[name], outs = compile_component(
                    bundle,
                    name=name,
                    engine_dir=engine_dir,
                    dryrun=config.dryrun,
                    trt_settings=config.trt_settings,
                    bench=self.bench,
                )
                upstream.update(spec.capture_upstream(name, outs, sample, bundle))

        # Family setattr. Dryrun leaves them installed so execute_engine still
        # hits the patched original module after export() returns.
        if config.dryrun:
            if self._dryrun_patches is not None:
                self._dryrun_patches.close()
            self._dryrun_patches = contextlib.ExitStack()
            self._dryrun_patches.enter_context(spec.apply_patches(model))
            _compile_components()
        else:
            with spec.apply_patches(model):
                _compile_components()

        # One module whose forward is spec.run() over execute_engine calls.
        runtime = EdgeRuntimeModule(spec, engines)
        runtime_kwargs = _clone_export_kwargs(spec.runtime_kwargs(sample))

        self.engines = engines
        self.runtime = runtime
        self.sample = dict(runtime_kwargs)

        if config.skip_runtime_export:
            return runtime
        # torch.export the stitched graph so the product is one ExportedProgram.
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
