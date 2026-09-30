from __future__ import annotations

import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..compile import compile_component
from ..config import EdgeConfig
from ..measure import parity
from ..spec import ComponentBundle, get_edge_spec
from .packager import DEFAULT_PACKAGERS, ComponentPackager


class EdgeExecuTorchExporter:
    """Compile Edge components once and package their ExecuTorch programs."""

    def __init__(
        self,
        packagers: Mapping[str, ComponentPackager] | None = None,
    ) -> None:
        self.packagers = dict(packagers or DEFAULT_PACKAGERS)
        self.engines: dict[str, Path] = {}
        self.programs: dict[str, Path] = {}
        self.sample: dict[str, Any] = {}
        self.bench: dict[str, tuple[float, float]] = {}
        self.manifest_path: Path | None = None

    def export(
        self,
        model: nn.Module,
        sample_inputs: MutableMapping[str, Any],
        config: EdgeConfig | dict[str, Any],
        *,
        output_dir: str | Path,
        device_id: int = 0,
    ) -> dict[str, Path]:
        if isinstance(config, dict):
            config = EdgeConfig(**config)
        elif not isinstance(config, EdgeConfig):
            raise TypeError(f"Expected EdgeConfig or dict, got {type(config)}")

        output_dir = Path(output_dir)
        engine_dir = output_dir / "engines"
        engine_dir.mkdir(parents=True, exist_ok=True)

        spec = get_edge_spec(model, config.model_type)
        sample = spec.prepare_sample_inputs(model, sample_inputs, config)
        bundles = spec.prepare(model, sample, config)

        prepared: dict[str, tuple[ComponentBundle, ComponentPackager]] = {}
        for name, bundle in bundles.items():
            try:
                packager = self.packagers[bundle.model_type]
            except KeyError as exc:
                raise ValueError(
                    f"No ExecuTorch packager registered for component {name!r} "
                    f"with model type {bundle.model_type!r}"
                ) from exc
            prepared[name] = (
                packager.prepare_bundle(bundle, spec, config),
                packager,
            )

        eager_ms: dict[str, float] = {}
        eager = spec.capture_eager_outputs(model, sample, config, bench=eager_ms)

        engines: dict[str, Path] = {}
        self.bench = {}
        with spec.apply_patches(model):
            for name, (bundle, _) in prepared.items():
                engine, trt_out, trt_ms = compile_component(
                    bundle,
                    name=name,
                    engine_dir=engine_dir,
                    trt_settings=config.trt_settings,
                )
                engines[name] = Path(engine)
                self.bench[name] = (eager_ms.get(name, 0.0), trt_ms)

                output_name = bundle.parity_output or bundle.output_names[0]
                trt = trt_out[bundle.output_names.index(output_name)]
                reference = eager.get(name)
                if isinstance(reference, torch.Tensor) and isinstance(
                    trt, torch.Tensor
                ):
                    parity(f"{name} eager vs TRT", reference, trt)

        programs: dict[str, Path] = {}
        for name, (bundle, packager) in prepared.items():
            programs.update(
                packager.package(
                    engines[name],
                    bundle,
                    output_dir,
                    device_id=device_id,
                )
            )

        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "engines": {
                        name: str(path.relative_to(output_dir))
                        for name, path in engines.items()
                    },
                    "programs": {
                        name: str(path.relative_to(output_dir))
                        for name, path in programs.items()
                    },
                },
                indent=2,
            )
            + "\n"
        )

        self.engines = engines
        self.programs = programs
        self.sample = dict(sample)
        self.manifest_path = manifest_path
        return programs
