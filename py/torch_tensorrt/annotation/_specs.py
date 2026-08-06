"""
TTA Spec Types and Annotation Metadata
=======================================

This module defines the kernel spec type hierarchy used by the TTA annotation
layer to describe how a custom QDP plugin should be compiled and registered.

Spec hierarchy
--------------

::

    CustomPluginSpec — AOT QDP plugin descriptor built by ``tta.custom_plugin()``
      ├── TritonSpec    — kernel implemented with Triton
      ├── CuTileSpec    — kernel implemented with NVIDIA CuTile (cuda-tile)
      ├── CuTeDSLSpec   — kernel implemented with NVIDIA CuTe DSL
      └── TvmFfiSpec    — kernel compiled via TVM FFI (planned; blocked on QDP TVM FFI support)

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# ── Shared helpers ────────────────────────────────────────────────────────────


def _dict_to_stable_tuple(d: Dict[str, Any]) -> tuple:
    """Return a deterministic, hashable tuple representation of a dict by sorting keys."""
    return tuple(sorted(d.items()))


def _validate_kernel_spec_fields(spec_name: str, launch_fn: Any, configs: Any) -> None:
    """Validate the common fields shared by TritonSpec, CuTileSpec, and CuTeDSLSpec."""
    if not callable(launch_fn):
        raise TypeError(
            f"{spec_name}: launch_fn must be callable, got {type(launch_fn).__name__!r}"
        )
    if configs is not None and not isinstance(configs, list):
        raise TypeError(
            f"{spec_name}: configs must be a list or None, got {type(configs).__name__!r}"
        )


# ── Custom kernel specs (Triton / CuTile / CuTeDSL / TvmFfi) ─────────────────
# TvmFfiSpec (tta.tvmffi) is planned as a fourth backend on par with the three
# below, but is blocked on TVM FFI support landing in QDP so that compiled
# kernels can be handed off to TRT's plugin runtime without a live Python
# interpreter.


@dataclass(frozen=True)
class TritonSpec:
    """Specification for a custom plugin implemented with a Triton kernel.

    Attributes:
        launch_fn: The Triton kernel entry-point.
        configs: List of launch-parameter dicts for autotuning candidates.
            ``None`` means no explicit configs.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.
        kwargs: Additional Triton-specific parameters.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    input_formats: Optional[Sequence[int]] = None
    output_formats: Optional[Sequence[int]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        configs_tuple = tuple(
            tuple(sorted(cfg.items())) for cfg in (self.configs or [])
        )
        in_fmts = tuple(int(f) for f in self.input_formats) if self.input_formats else ()
        out_fmts = tuple(int(f) for f in self.output_formats) if self.output_formats else ()
        return (
            "triton",
            id(self.launch_fn),
            configs_tuple,
            _dict_to_stable_tuple(self.kwargs),
            in_fmts,
            out_fmts,
        )

    def __post_init__(self) -> None:
        _validate_kernel_spec_fields("TritonSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class CuTileSpec:
    """Specification for a custom plugin implemented with an NVIDIA CuTile kernel.

    Attributes:
        launch_fn: The CuTile kernel entry-point.
        configs: List of launch-parameter dicts for autotuning candidates.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.
        kwargs: Additional CuTile-specific parameters.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    input_formats: Optional[Sequence[int]] = None
    output_formats: Optional[Sequence[int]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        configs_tuple = tuple(
            tuple(sorted(cfg.items())) for cfg in (self.configs or [])
        )
        in_fmts = tuple(int(f) for f in self.input_formats) if self.input_formats else ()
        out_fmts = tuple(int(f) for f in self.output_formats) if self.output_formats else ()
        return (
            "cutile",
            id(self.launch_fn),
            configs_tuple,
            _dict_to_stable_tuple(self.kwargs),
            in_fmts,
            out_fmts,
        )

    def __post_init__(self) -> None:
        _validate_kernel_spec_fields("CuTileSpec", self.launch_fn, self.configs)


@dataclass(frozen=True)
class CuTeDSLSpec:
    """Specification for a custom plugin implemented with the NVIDIA CuTe DSL.

    Attributes:
        launch_fn: The CuTe DSL kernel entry-point.
        configs: List of launch-parameter dicts for autotuning candidates.
        arch: Optional target GPU architecture string (e.g. ``"sm_80"``).
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.
        kwargs: Additional CuTe DSL-specific parameters.
    """

    launch_fn: Callable
    configs: Optional[List[Dict[str, Any]]] = None
    arch: Optional[str] = None
    input_formats: Optional[Sequence[int]] = None
    output_formats: Optional[Sequence[int]] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_cache_key(self) -> tuple:
        configs_tuple = tuple(
            tuple(sorted(cfg.items())) for cfg in (self.configs or [])
        )
        in_fmts = tuple(int(f) for f in self.input_formats) if self.input_formats else ()
        out_fmts = tuple(int(f) for f in self.output_formats) if self.output_formats else ()
        return (
            "cutedsl",
            id(self.launch_fn),
            self.arch,
            configs_tuple,
            _dict_to_stable_tuple(self.kwargs),
            in_fmts,
            out_fmts,
        )

    def __post_init__(self) -> None:
        _validate_kernel_spec_fields("CuTeDSLSpec", self.launch_fn, self.configs)



# ── Factory functions ─────────────────────────────────────────────────────────


def triton(
    launch_fn: Callable,
    configs: Optional[List[Dict[str, Any]]] = None,
    input_formats: Optional[Sequence[int]] = None,
    output_formats: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> TritonSpec:
    """Create a :class:`TritonSpec` for a Triton kernel custom plugin.

    Args:
        launch_fn: Triton kernel function.
        configs: List of autotuning config dicts.  Pass ``None`` for no configs.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.

    Returns:
        A :class:`TritonSpec` instance.
    """
    return TritonSpec(
        launch_fn=launch_fn,
        configs=configs,
        input_formats=input_formats,
        output_formats=output_formats,
        kwargs=kwargs,
    )


def cutile(
    launch_fn: Callable,
    configs: Optional[List[Dict[str, Any]]] = None,
    input_formats: Optional[Sequence[int]] = None,
    output_formats: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> CuTileSpec:
    """Create a :class:`CuTileSpec` for a CuTile kernel custom plugin.

    Args:
        launch_fn: CuTile kernel function.
        configs: List of autotuning config dicts.  Pass ``None`` for no configs.
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.

    Returns:
        A :class:`CuTileSpec` instance.
    """
    return CuTileSpec(
        launch_fn=launch_fn,
        configs=configs,
        input_formats=input_formats,
        output_formats=output_formats,
        kwargs=kwargs,
    )


def cutedsl(
    launch_fn: Callable,
    configs: Optional[List[Dict[str, Any]]] = None,
    arch: Optional[str] = None,
    input_formats: Optional[Sequence[int]] = None,
    output_formats: Optional[Sequence[int]] = None,
    **kwargs: Any,
) -> CuTeDSLSpec:
    """Create a :class:`CuTeDSLSpec` for a CuTe DSL kernel custom plugin.

    Args:
        launch_fn: CuTe DSL kernel function.
        configs: List of autotuning config dicts.  Pass ``None`` for no configs.
        arch: Target GPU architecture string (e.g. ``"sm_80"``).
        input_formats: Optional tensor layout descriptors for input tensors.
        output_formats: Optional tensor layout descriptors for output tensors.

    Returns:
        A :class:`CuTeDSLSpec` instance.
    """
    return CuTeDSLSpec(
        launch_fn=launch_fn,
        configs=configs,
        arch=arch,
        input_formats=input_formats,
        output_formats=output_formats,
        kwargs=kwargs,
    )



