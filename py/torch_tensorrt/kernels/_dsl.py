"""Declarative kernel descriptor used by :func:`torch_tensorrt.kernels.cuda_kernel_op`.

This module intentionally contains no runtime logic beyond dataclass
construction. Derivation of meta / eager / aot / schema happens in
``_kernel_plugin.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple, Union

import torch

# ---------- ShapeRel: how to derive output shape from inputs ----------


@dataclass(frozen=True)
class SameAs:
    """Output has the same shape as the referenced tensor input.

    ``input_idx`` may be either the integer position into the *tensor*-only
    input list (``ScalarInput`` entries are skipped) or the ``name`` of a
    tensor input declared via :class:`InputDecl`. The name form is preferred
    because it stays correct when the input list is reordered.
    """

    input_idx: Union[int, str] = 0


@dataclass(frozen=True)
class ReduceDims:
    """Output = the referenced tensor input with ``dims`` removed.

    If ``keepdim=True`` those axes are kept with size 1 instead of removed.
    Negative axes are allowed. ``input_idx`` accepts either the integer
    position into the tensor-only input list or the input ``name``.
    """

    input_idx: Union[int, str]
    dims: Tuple[int, ...]
    keepdim: bool = False


ShapeRel = Union[SameAs, ReduceDims]


# ---------- Extra scalar args (between input ptrs and output ptrs) ----------


@dataclass(frozen=True)
class Numel:
    """Pass ``inputs[input_name].numel()`` as an ``int`` extra."""

    input_name: str


@dataclass(frozen=True)
class DimSize:
    """Pass ``inputs[input_name].shape[axis]`` as an ``int`` extra.

    Negative ``axis`` allowed.
    """

    input_name: str
    axis: int


ExtraArg = Union[Numel, DimSize]


# ---------- Launch geometry ----------


@dataclass(frozen=True)
class Elementwise:
    """One thread per output element.

    ``layout="flat"``: 1D launch over the flattened output ``numel``.
        ``block = (bx,)`` → ``grid = (cdiv(numel(out), bx),)``.
    ``layout="nd"``: the trailing ``len(block)`` axes of the output are
        block-parallelized; any leading axes are folded into ``grid_z``.
        ``block[0]`` maps to the last (innermost) axis, matching CUDA's
        convention that ``grid_x`` / ``block_x`` varies fastest.
    """

    block: Tuple[int, ...] = (256,)
    layout: Literal["flat", "nd"] = "flat"


@dataclass(frozen=True)
class Reduction:
    """One block per output element; block threads cooperate across the
    reduction axes. ``reduce_dims`` are axes of the **input** (not output)
    that are collapsed. Grid = ``numel(output)``, block = ``block_size``.
    """

    reduce_dims: Tuple[int, ...]
    block_size: int = 256


@dataclass(frozen=True)
class Custom:
    """Escape hatch. ``fn(inputs, outputs, tactic)`` returns the same shape
    as today's hand-written aot_fn: ``(KernelLaunchParams, SymExprs)``.
    """

    fn: Callable[..., Any]


Geometry = Union[Elementwise, Reduction, Custom]


# ---------- Input / output decls ----------


@dataclass(frozen=True)
class InputDecl:
    """Tensor kernel input.

    The corresponding kernel argument is a ``T*`` (data pointer) at the input
    pointer position in the calling convention.
    """

    name: str
    dtype: Optional[torch.dtype] = None


@dataclass(frozen=True)
class ScalarInput:
    """Scalar (non-tensor) kernel input — e.g. ``float alpha`` or ``int k``.

    Scalars are forwarded by value to the kernel at the input position
    (after all preceding tensor/scalar inputs, before extras and output
    pointers).  ``py_type`` must be ``float``, ``int``, or ``bool``.
    """

    name: str
    py_type: type  # float, int, bool


InputSpec = Union[InputDecl, ScalarInput]


@dataclass(frozen=True)
class OutputDecl:
    name: str
    shape: ShapeRel
    dtype_from: Optional[str] = None


# ---------- Top-level spec ----------


@dataclass
class KernelSpec:
    """Declarative description of a CUDA kernel.

    Kernel signature convention: the ``__global__`` function receives
    arguments in this fixed order — input pointers, then extras in
    ``extras`` order, then output pointers.

    All DSL fields beyond ``kernel_source`` / ``kernel_name`` are optional.
    Whichever fields are populated drive auto-derivation of the matching
    meta / eager / aot / schema artifacts; whatever is missing must be
    supplied as an override keyword argument to :func:`cuda_kernel_op`.
    """

    kernel_source: str
    kernel_name: str
    inputs: Optional[Sequence[InputSpec]] = None
    outputs: Optional[Sequence[OutputDecl]] = None
    extras: Sequence[ExtraArg] = field(default_factory=list)
    geometry: Optional[Geometry] = None
    include_paths: Optional[List[str]] = None
    compile_std: str = "c++17"
    arch_override: Optional[str] = None
