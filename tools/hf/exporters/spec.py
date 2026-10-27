from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, MutableMapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

_SPECS: dict[str, type[EdgeSpec]] = {}


def register_edge_spec(*model_types: str) -> Callable[[type[EdgeSpec]], type[EdgeSpec]]:
    """Register an :class:`EdgeSpec` for one or more ``config.model_type`` keys."""

    def decorator(cls: type[EdgeSpec]) -> type[EdgeSpec]:
        for model_type in model_types:
            _SPECS[model_type] = cls
        return cls

    return decorator


def registered_specs() -> dict[str, type[EdgeSpec]]:
    return dict(_SPECS)


@dataclass
class ComponentBundle:
    """Everything :func:`compile_component` needs to build one engine."""

    module: nn.Module
    trace_args: tuple[Any, ...]
    save_args: tuple[Any, ...]
    input_names: list[str]
    output_names: list[str]
    extra_config: dict[str, Any] = field(default_factory=dict)
    trt_settings: dict[str, Any] = field(default_factory=dict)
    patch_fn: Callable[[nn.Module], Any] | None = None
    context_attention_mask_type: int | None = None
    execute_args: tuple[Any, ...] | None = None
    model_type: str = "edge"
    engine_file: str = "engine.engine"


class EdgeSpec(ABC):
    """Per-family flatten / runtime wiring.

    ``EdgeExporter.export`` never branches on PI05 vs Nemotron. It only loops
    ``spec.components``.
    """

    components: tuple[str, ...] = ()

    def apply_patches(
        self, model: nn.Module | None = None
    ) -> AbstractContextManager[None]:
        """Install this family's setattr replacements for the whole ``export()``.

        Default is a no-op. Families register factories on their own backend
        and return ``apply_patches(backend)``. ``model`` is the export root;
        Nemotron uses it to wrap hybrid mixers.
        """
        del model
        return nullcontext()

    @abstractmethod
    def prepare_sample_inputs(
        self,
        model: nn.Module,
        raw: Mapping[str, Any],
        config: Any,
    ) -> MutableMapping[str, Any]:
        """Caller payload → stem dict used by prepare/run."""

    @abstractmethod
    def prepare(
        self,
        name: str,
        model: nn.Module,
        sample: MutableMapping[str, Any],
        upstream: Mapping[str, Any],
        config: Any,
    ) -> ComponentBundle:
        """Select the original submodule and build its trace/save tuple."""

    def capture_upstream(
        self,
        name: str,
        outputs: Any,
        sample: Mapping[str, Any],
        bundle: ComponentBundle,
    ) -> dict[str, Any]:
        """Map this engine's outputs into keys the next ``prepare`` needs."""
        return {}

    @abstractmethod
    def run(self, engines: Mapping[str, str], sample: Mapping[str, Any]) -> Any:
        """Packing + ``execute_engine`` calls. This is the dumped graph."""

    def runtime_kwargs(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Tensor kwargs for ``torch.export`` of :class:`EdgeRuntimeModule`."""
        return {
            key: value
            for key, value in sample.items()
            if hasattr(value, "dtype") and hasattr(value, "device")
        }


def infer_model_type(model: nn.Module, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str) and model_type in _SPECS:
        return model_type
    if hasattr(model, "paligemma_with_expert") or hasattr(
        getattr(model, "model", None), "paligemma_with_expert"
    ):
        return "pi05"
    if hasattr(model, "_groot_model") or (
        getattr(getattr(model, "backbone", None), "eagle_model", None) is not None
    ):
        return "groot"
    name = type(model).__name__.lower()
    if "nemotron" in name:
        return "nemotron_h"
    if isinstance(model_type, str):
        return model_type
    raise KeyError(
        f"No EdgeSpec for {type(model).__name__}. "
        f"Pass EdgeConfig(model_type=...) or register one. "
        f"Known: {sorted(_SPECS)}"
    )


def get_edge_spec(model: nn.Module, model_type: str | None = None) -> EdgeSpec:
    key = infer_model_type(model, model_type)
    if key not in _SPECS:
        raise KeyError(f"No EdgeSpec registered for {key!r}. Known: {sorted(_SPECS)}")
    return _SPECS[key]()
