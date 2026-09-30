from __future__ import annotations

import copy
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch.export import ExportedProgram

from . import ops as _ops  # noqa: F401
from .compile import compile_component
from .config import EdgeConfig
from .measure import parity
from .runtime import EdgeRuntimeModule
from .spec import get_edge_spec

if TYPE_CHECKING:
    from torch_tensorrt_edge_llm.pi05 import PI05Export


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


class EdgeExporter:
    def __init__(self) -> None:
        self.engines: dict[str, str] = {}
        self.sample: dict[str, Any] = {}
        self.bench: dict[str, tuple[float, float]] = {}

    def export_pi05(
        self,
        model: nn.Module,
        sample_inputs: MutableMapping[str, Any],
        config: EdgeConfig | dict[str, Any],
        *,
        num_steps: int = 10,
    ) -> PI05Export:
        """Export a full PI0.5 rollout with embedded, reusable runtime methods.

        Unlike the legacy single-velocity export(), this boundary takes
        preprocessed images/image_masks/tokens/token_mask/noise and returns a
        PI05Export with a runnable program and multi-method ExecuTorch save.
        """
        from torch_tensorrt_edge_llm.pi05 import export_pi05, prepare_pi05_sample

        if isinstance(config, dict):
            config = EdgeConfig(**config)
        elif not isinstance(config, EdgeConfig):
            raise TypeError(f"Expected EdgeConfig or dict, got {type(config)}")
        if config.dynamic or config.dynamic_shapes is not None:
            raise ValueError("Embedded PI0.5 currently requires fixed shapes")
        core = model if hasattr(model, "paligemma_with_expert") else model.model
        sample = prepare_pi05_sample(
            core,
            sample_inputs["images"],
            sample_inputs["image_masks"],
            sample_inputs["tokens"],
            sample_inputs["token_mask"],
            sample_inputs["noise"],
        )
        directory = Path(config.engine_dir or "edge_engines")
        result = export_pi05(
            core,
            sample,
            engine_dir=directory,
            cameras=len(sample_inputs["images"]),
            num_steps=num_steps,
            settings=config.trt_settings,
        )
        self.engines = {
            name: str(directory / name) for name in ("vision", "language", "action")
        }
        return result

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
            dynamic_shapes = {
                name: {axis: torch.export.Dim.AUTO for axis in range(value.ndim)}
                for name, value in runtime_kwargs.items()
            }

        return torch.export.export(
            runtime,
            args=(),
            kwargs=_clone_export_kwargs(runtime_kwargs),
            strict=config.strict,
            dynamic_shapes={"sample": dynamic_shapes}
            if dynamic_shapes is not None
            else None,
            prefer_deferred_runtime_asserts_over_guards=(
                config.prefer_deferred_runtime_asserts_over_guards
            ),
        )
