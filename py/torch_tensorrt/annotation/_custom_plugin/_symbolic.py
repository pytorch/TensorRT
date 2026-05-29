"""Symbolic tensor proxies used during AOT kernel compilation (shape expression bindings).

During the AOT compilation pipeline each backend (Triton, cuTILE, CuTe DSL) needs
to run the user's *launch function* without a real GPU so that it can:

  1. Intercept the kernel call and record which tensors / scalars are passed.
  2. Capture the symbolic launch grid (grid_x/y/z) as trtp.SymInt32 expressions
     so that TRT can evaluate them at engine-run time with actual input shapes.

``SymbolicTensor`` is the proxy object injected in place of real ``torch.Tensor``
arguments.  It wraps a QDP ``TensorDesc`` and exposes:

* ``shape`` / ``shape_expr`` — per-dimension *SymInt32 expressions* that bind to
  the input dimension at runtime (e.g. ``shape[0]`` evaluates to the batch size
  that TRT resolves during engine execution).
* ``stride`` — row-major symbolic strides, or format-specific strides for packed
  channel layouts (HWC, DHWC, etc.).
* ``numel()`` — symbolic product of all dimensions.

The ``TensorRole`` enum distinguishes input tensors from output tensors so that
``analyze_launch_args`` in ``_qdp_utils.py`` can reconstruct the correct
``param_binding_indices`` that TRT uses to identify which runtime buffer to pass
for each kernel pointer parameter.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Tuple

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
    trt = None  # type: ignore[assignment]
    trtp = None  # type: ignore[assignment]


class TensorRole(Enum):
    """Identifies whether a SymbolicTensor represents an input or output tensor.

    Used by ``analyze_launch_args`` to map each kernel pointer parameter back to
    the correct TRT binding index (inputs occupy indices 0..num_inputs-1, outputs
    occupy indices num_inputs..num_inputs+num_outputs-1).
    """

    INPUT = auto()
    OUTPUT = auto()


def _to_sym_int(x: Any) -> Any:
    """Convert a shape/stride element to SymInt32; accept int or SymInt32-like."""
    if _TRT_AVAILABLE and isinstance(x, trtp.SymInt32):
        return x
    if isinstance(x, int):
        return trtp.SymInt32(x) if _TRT_AVAILABLE else x
    if _TRT_AVAILABLE:
        return trtp.SymInt32(x)
    return x


def _sym_to_int_if_const(x: Any) -> Any:
    """Return Python int if x is concrete (Python int or constant SymInt32), else return x unchanged.

    In TRT's aot_impl context, shape_expr elements may be SymInt32 objects backed by
    a trt.IDimensionExpr constant (e.g. for a static-shape tensor).  We can extract
    the concrete value via _int_expr.is_constant() / .get_constant_value().
    This lets downstream code use isinstance(v, int) checks correctly so that grid
    computation in user launch_fn works even with SymbolicTensor arguments.
    """
    if isinstance(x, int):
        return x
    # Try via TRT IDimensionExpr (available when _exprBuilder was set during build).
    # Broad catch is necessary: IDimensionExpr attribute access raises different
    # exception types across TRT versions (AttributeError, RuntimeError, TypeError).
    try:
        ie = getattr(x, "_int_expr", None)
        if ie is not None and ie.is_constant():
            return int(ie.get_constant_value())
    except (AttributeError, RuntimeError, TypeError, ValueError) as _e:
        logger.debug("_sym_to_int_if_const: failed to extract constant from %r: %s", x, _e)
    return x


class _ShapeDim:
    """Wrapper for a symbolic shape dimension supporting arithmetic in launch fns.

    math.prod(iterable) starts with 1 (int) and multiplies left-to-right.
    The first step is ``1 * element``, which Python evaluates as:
      1. int.__mul__(element) → NotImplemented
      2. element.__rmul__(1)  ← this method

    Standard trtp.SymInt32 has no __rmul__, so math.prod fails for dynamic
    dims. _ShapeDim wraps a SymInt32 (or int) and delegates all arithmetic to
    self._v so that expressions like math.prod(x.shape) and ceiling division
    (M + BX - 1) // BX work transparently in @cute.jit launch functions.

    _expr is exposed so that SymIntExpr._op and _as_symint32 can extract the
    underlying IDimensionExpr without knowing the concrete wrapper type.
    """

    def __init__(self, v: Any) -> None:
        self._v = v  # int or trtp.SymInt32
        # Cache the underlying IDimensionExpr so SymIntExpr._op and _as_symint32
        # can extract it without knowing the concrete wrapper type.
        self._expr = getattr(v, "_expr", getattr(v, "_int_expr", None))

    def __add__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v + v

    def __radd__(self, other: Any) -> Any:
        return self._v + other

    def __sub__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v - v

    def __rsub__(self, other: Any) -> Any:
        return other - self._v

    def __mul__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v * v

    def __rmul__(self, other: Any) -> Any:
        """Called as: other * self (e.g., int(1) * _ShapeDim from math.prod)."""
        return self._v * other  # commutative: SymInt32.__mul__(int) handles it

    def __floordiv__(self, other: Any) -> Any:
        v = other._v if isinstance(other, _ShapeDim) else other
        return self._v // v

    def __rfloordiv__(self, other: Any) -> Any:
        return other // self._v

    def __int__(self) -> int:
        return int(self._v)

    def __repr__(self) -> str:
        return f"_ShapeDim({self._v!r})"


def _strides_from_td(td: Any) -> Tuple[Any, ...]:
    """Return strides for *td*, sourced from TRT runtime where possible.

    Two sources, probed in priority order:

    1. ``td.strides`` — physical strides on a ``trtp.Tensor`` object (the
       runtime tensor type passed to JIT ``enqueue``).  When a
       ``SymbolicTensor`` wraps a ``trtp.Tensor``, this gives the actual
       strides TRT has assigned to the buffer, correctly handling any layout
       (e.g. row-padded LINEAR from Myelin matmul fusion).

    2. Row-major fallback from ``shape_expr`` — correct for contiguous tensors.
       Used when wrapping a ``TensorDesc`` in the AOT path (which carries only
       logical shape, not physical strides).

    Analytical per-format stride reconstruction is intentionally absent: it
    replicates TRT's internal layout logic and is fragile.  Correct physical
    strides must come from TRT itself.
    """
    strides = getattr(td, "strides", None)
    if strides is not None:
        return tuple(_to_sym_int(s) for s in strides)

    # Fallback: logical row-major strides from shape_expr.
    shape_expr = td.shape_expr
    prod = trtp.SymInt32(1) if _TRT_AVAILABLE else 1
    strides_list: list = []
    for i in range(len(shape_expr) - 1, -1, -1):
        strides_list.insert(0, prod)
        prod = prod * _to_sym_int(shape_expr[i])
    return tuple(_to_sym_int(s) for s in strides_list)


@dataclass
class SymbolicTensor:
    """Symbolic view over a QDP TensorDesc with role metadata.

    Attributes:
      td: TensorDesc from QDP.
      role: TensorRole.INPUT or TensorRole.OUTPUT.
      index: role-local index (0-based, within inputs or outputs).
    """

    td: Any  # trtp.TensorDesc
    role: TensorRole
    index: int

    def __post_init__(self) -> None:
        # Pre-compute shape dims once: Python int for static, _ShapeDim for dynamic.
        # _ShapeDim adds __rmul__ so math.prod(x.shape) works in @cute.jit kernels.
        # Guard against mock/test TDs that lack shape_expr.
        shape_expr = getattr(self.td, "shape_expr", None)
        _shape = []
        if shape_expr is not None:
            for d in shape_expr:
                concrete = _sym_to_int_if_const(d)
                _shape.append(concrete if isinstance(concrete, int) else _ShapeDim(concrete))
        self._shape: Tuple[Any, ...] = tuple(_shape)

        self._stride: Tuple[Any, ...] = _strides_from_td(self.td) if shape_expr is not None else ()

        # Pre-compute numel: Python int for fully-static shapes, SymInt32 for dynamic.
        if shape_expr is None:
            self._numel: Any = 0
        else:
            concrete_dims = [_sym_to_int_if_const(d) for d in shape_expr]
            if all(isinstance(v, int) for v in concrete_dims):
                result = 1
                for v in concrete_dims:
                    result *= v
                self._numel = result
            else:
                n = trtp.SymInt32(1) if _TRT_AVAILABLE else 1
                for d in shape_expr:
                    n = n * _to_sym_int(d)
                self._numel = n

    @property
    def shape(self) -> Tuple[Any, ...]:
        return self._shape

    @property
    def shape_expr(self) -> Tuple[Any, ...]:
        """Symbolic shape dimensions (same as .shape); use for grid or extra_args."""
        return self.shape

    def size(self, dim: int | None = None):
        """PyTorch-style alias for shape / shape[dim]."""
        if dim is None:
            return self.shape
        return self.shape[dim]

    def shape_dim(self, dim: int) -> Any:
        """Return the raw SymInt32 for dimension *dim* from the underlying TensorDesc."""
        return _to_sym_int(self.td.shape_expr[dim])

    def stride(self, dim: int | None = None):
        """PyTorch-style stride API: stride() or stride(dim)."""
        if dim is None:
            return self._stride
        return self._stride[dim]

    def numel(self) -> Any:
        """Total element count: Python int for static shapes, SymInt32 for dynamic."""
        return self._numel

    @property
    def is_cuda(self) -> bool:
        return True
