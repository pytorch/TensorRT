"""Shared utilities for QDP integration.

Provides sandboxing, fingerprinting, dtype/format conversion, meta tensor
helpers, and launch-arg analysis used by all backend AOT implementations.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import tempfile
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
    trt = None  # type: ignore[assignment]
    trtp = None  # type: ignore[assignment]


# ---- SymInt patching ----

# LIMITATION (experimental shim): Monkey-patches a private TRT binding class
# (tensorrt_bindings.plugin._tensor.SymInt32) to add a missing _is_dummy
# property required by TRT's get_launch_params() on Blackwell sm_120 builds.
# This patches an internal, undocumented attribute of TRT's Python bindings.
# It will break silently if TRT renames or removes _tensor.SymInt32, or if a
# future TRT release adds _is_dummy natively (in which case the guard
# ``isinstance(..., property)`` will skip the patch automatically).
# Remove once TRT exposes _is_dummy on SymInt32 in its public plugin API.
#
# TRT's get_launch_params() reads _is_dummy on SymInt32 to determine whether
# a grid dimension is symbolic.  SymInt32 doesn't expose this attribute by
# default, causing AttributeError on some TRT builds (e.g. Blackwell sm_120).
# We add it as a property with a no-op setter so that:
#   (a) TRT can read _is_dummy without AttributeError,
#   (b) ShapeExpr.__init__ (which sets self._is_dummy = False) doesn't crash.
try:
    import tensorrt_bindings.plugin._tensor as _trtp_tensor  # type: ignore[import]
    _SymInt32 = _trtp_tensor.SymInt32
    if not isinstance(getattr(_SymInt32, "_is_dummy", None), property):
        _SymInt32._is_dummy = property(
            lambda self: getattr(self, "_int_expr", None) is None,
            lambda self, v: None,  # no-op setter: lets ShapeExpr.__init__ assign _is_dummy = False
        )
    del _SymInt32, _trtp_tensor
except (ImportError, AttributeError):
    # Broad catch is intentional: this is an optional compatibility shim.
    # If tensorrt_bindings is unavailable, or the internal _tensor module has
    # changed its layout, we skip the patch silently.  The missing _is_dummy
    # attribute will only manifest at runtime on affected TRT builds, at which
    # point the QDP call will raise a clear AttributeError from TRT itself.
    pass


from ._symbolic import SymbolicTensor, TensorRole


# ---- Error type ----


class QDPRuntimeError(RuntimeError):
    """Structured error for QDP plugin runtime failures.

    Distinct from TTAPluginError (in _plugin_lowering.py), which is used for
    FX-graph-level lowering failures. QDPRuntimeError covers TRT-side errors
    in AOT impl, kernel dispatch, and plugin compilation.

    Args:
        op: QDP op name or annotated function name.
        stage: Processing stage (e.g. "aot_impl", "compile", "sandbox").
        backend: Backend name ("triton", "cutile", "cutedsl", "custom_plugin").
        msg: Human-readable error message.
    """

    def __init__(self, op: str, stage: str, backend: str, msg: str) -> None:
        super().__init__(f"{op}: [{stage}] [{backend}] {msg}")
        self.op = op
        self.stage = stage
        self.backend = backend
        self.msg = msg


# Backward-compatible alias: other modules in this package import TTAPluginError
# from _qdp_utils; keep the name available until those files are updated.
TTAPluginError = QDPRuntimeError


# ---- AOT metadata ----


@dataclass
class AOTMetadata:
    """Unified AOT compilation result, shared by all backends.

    Attributes:
        binary:        PTX or CUBIN bytes ready for TRT to JIT-compile.
        kernel_name:   Entry point name matching the .entry directive in PTX.
        launch_params: SimpleNamespace from _launch_params_from_trt (grid, block,
                       shared_mem, param_binding_indices, sym_int_exprs).
        backend:       One of "triton", "cutile", or "cutedsl".
    """

    binary: bytes
    kernel_name: str
    launch_params: Any  # types.SimpleNamespace from _launch_params_from_trt
    backend: str  # "triton", "cutile", or "cutedsl"


# ---- Tactic table ----


@dataclass
class TacticEntry:
    """A single tactic entry referencing a kernel spec and config by index.

    Attributes:
        spec_idx:   Index into the kernel-spec list passed to build_tactic_table.
        config_idx: Index into that spec's configs list (0 when configs is empty).
    """

    spec_idx: int
    config_idx: int


def build_tactic_table(specs: Sequence[Any]) -> List[TacticEntry]:
    """Build a flat tactic table from a list of kernel specs.

    Each (spec_idx, config_idx) pair becomes one tactic entry.
    """
    table: List[TacticEntry] = []
    for s_idx, spec in enumerate(specs):
        configs = spec.configs if spec.configs else [{}]
        for c_idx, _ in enumerate(configs):
            table.append(TacticEntry(spec_idx=s_idx, config_idx=c_idx))
    return table


# ---- Sandboxing ----


def make_sandboxed_host(
    launch_fn: Callable[..., Any],
    overrides: Dict[str, Any],
) -> Callable[..., Any]:
    """Create a copy of launch_fn with cloned globals plus symbol overrides.

    This lets us swap real @triton.jit kernels (or cuTILE programs) for
    recorder proxies without mutating the original module.
    """
    mod = inspect.getmodule(launch_fn)
    if mod is None:
        raise RuntimeError(f"cannot find module for launch_fn {launch_fn}")

    new_globals = dict(mod.__dict__)
    new_globals.update(overrides)

    return types.FunctionType(
        launch_fn.__code__,
        new_globals,
        name=launch_fn.__name__,
        argdefs=launch_fn.__defaults__,
        closure=launch_fn.__closure__,
    )


# ---- Fingerprinting ----


def fingerprint_fn(fn: Callable[..., Any]) -> bytes:
    """Return a stable byte fingerprint for a Python function.

    The fingerprint combines the function's module path, qualified name, and
    a SHA-1 hash of its raw bytecode.  This makes the fingerprint stable
    across interpreter restarts (assuming the source has not changed) while
    still changing whenever the function body is edited.
    """
    mod = getattr(fn, "__module__", "<unknown>")
    qn = getattr(fn, "__qualname__", getattr(fn, "__name__", "<lambda>"))
    code = getattr(fn, "__code__", None)
    code_bytes = getattr(code, "co_code", b"") if code is not None else b""
    blob = {
        "module": mod,
        "qualname": qn,
        "code_sha1": hashlib.sha1(code_bytes).hexdigest(),
    }
    return str(blob).encode("utf-8")


def derive_impl_id(
    specs: Sequence[Any],
    attrs: Optional[Dict[str, Any]] = None,
    num_weights: int = 0,
) -> str:
    """Derive a deterministic ID string from a list of kernel specs, plugin attrs, and weight count.

    ``num_weights`` is included in the hash so that the same kernel spec used
    with different numbers of declared weight inputs produces a distinct op_name
    and QDP registration.  Without this, a plugin registered with num_inputs=1
    (activation only) would be incorrectly reused when called with num_inputs=2
    (activation + weight), causing silent stale-registration bugs.
    """
    h = hashlib.sha1()
    for spec in specs:
        h.update(fingerprint_fn(spec.launch_fn))
        configs = spec.configs if spec.configs else [{}]
        h.update(str(configs).encode("utf-8"))
        in_fmts = getattr(spec, "input_formats", None)
        out_fmts = getattr(spec, "output_formats", None)
        h.update(str([int(f) for f in in_fmts] if in_fmts else []).encode("utf-8"))
        h.update(str([int(f) for f in out_fmts] if out_fmts else []).encode("utf-8"))
    if attrs:
        h.update(str(sorted(attrs.items())).encode("utf-8"))
    if num_weights:
        h.update(f"num_weights:{num_weights}".encode("utf-8"))
    return h.hexdigest()


def make_qdp_symbol(impl_id: str) -> str:
    """Derive a unique QDP namespace::name from an impl_id."""
    return f"tta_custom::host_kernel_{impl_id[:8]}"


# ---- Backend-object detection heuristics ----


def is_triton_kernel(obj: Any) -> bool:
    """Heuristic: Triton kernels implement __getitem__ for [grid] indexing.

    Check both obj.__module__ (may be the user module) and
    type(obj).__module__ (always "triton.runtime.jit" for JITFunction)
    so that kernels defined in user modules are detected correctly.
    """
    if not hasattr(obj, "__getitem__"):
        return False
    return (
        "triton" in getattr(obj, "__module__", "")
        or "triton" in getattr(type(obj), "__module__", "")
    )


def is_cutile_program(obj: Any) -> bool:
    """Heuristic: cuTILE programs are callables from cuda.tile.* modules."""
    modname = getattr(obj, "__module__", "")
    return callable(obj) and ("cuda.tile" in modname or "cutile" in modname)


def is_cute_kernel(obj: Any) -> bool:
    """Heuristic: @cute.kernel objects from CUTLASS CuTe DSL.

    @cute.kernel objects have type(obj).__module__ == 'builtins', so we check
    the _dsl_cls attribute which points to the CuTe DSL kernel class.
    """
    if not callable(obj) or isinstance(obj, type):
        return False
    dsl_cls = getattr(obj, "_dsl_cls", None)
    if dsl_cls is None:
        return False
    cls_mod = getattr(dsl_cls, "__module__", "") or ""
    return "cutlass" in cls_mod or "cute" in cls_mod


# ---- SymInt helpers ----


def _sym_dim(d: Any) -> Any:
    """Convert a shape element (int or SymInt32) to SymInt32 for symbolic arithmetic."""
    if _TRT_AVAILABLE and isinstance(d, trtp.SymInt32):
        return d
    return trtp.SymInt32(int(d))


def _as_symint32(v: Any) -> Any:
    """Ensure v is a trtp.SymInt32 (not SymIntExpr).

    SymInt32 arithmetic returns SymIntExpr objects, but TRT's KernelLaunchParams.grid_*
    and extra_args slots require SymInt32. SymIntExpr wraps IDimensionExpr in ._expr;
    SymInt32 accepts IDimensionExpr directly.
    """
    if _TRT_AVAILABLE and isinstance(v, trtp.SymInt32):
        return v
    expr = getattr(v, "_expr", None)
    if expr is not None:
        return trtp.SymInt32(expr)
    try:
        return trtp.SymInt32(int(v))
    except (TypeError, ValueError):
        return trtp.SymInt32(1)


def _assign_recorded_grid(launch: Any, recorded_grid: Any) -> None:
    """Assign ``grid_x``/``grid_y``/``grid_z`` on *launch* from a recorded grid value.

    Centralises the repeated pattern used by all three AOT backends (Triton,
    cuTILE, CuTe DSL) that capture a grid tuple during sandbox execution and
    must populate a ``trtp.KernelLaunchParams`` object.

    Args:
        launch:        A ``trtp.KernelLaunchParams`` instance to modify in-place.
        recorded_grid: The grid value captured during sandbox execution.  May be:

            * ``None``  — all three dimensions are set to ``SymInt32(1)``.
            * A non-tuple scalar (int / SymInt32) — ``grid_x`` is set to that
              value; ``grid_y`` and ``grid_z`` default to 1.  This covers the
              Triton path where the launch_fn passes a plain integer grid.
            * A tuple of 1–3 elements (int or SymInt32/SymIntExpr) — each
              element is converted with :func:`_as_symint32`; missing dimensions
              default to 1.  This covers the cuTILE and CuTe DSL paths.
    """
    if recorded_grid is None:
        launch.grid_x = trtp.SymInt32(1)
        launch.grid_y = trtp.SymInt32(1)
        launch.grid_z = trtp.SymInt32(1)
        return
    if not isinstance(recorded_grid, tuple):
        launch.grid_x = _as_symint32(recorded_grid)
        launch.grid_y = trtp.SymInt32(1)
        launch.grid_z = trtp.SymInt32(1)
        return
    launch.grid_x = _as_symint32(recorded_grid[0]) if len(recorded_grid) >= 1 else trtp.SymInt32(1)
    launch.grid_y = _as_symint32(recorded_grid[1]) if len(recorded_grid) >= 2 else trtp.SymInt32(1)
    launch.grid_z = _as_symint32(recorded_grid[2]) if len(recorded_grid) >= 3 else trtp.SymInt32(1)


def _safe_dim(d: Any, default: int = 1) -> int:
    """Extract a concrete int from a shape element safely.

    For TRT SymInt32 elements (dynamic shapes), calling int() directly does NOT raise
    but returns a garbage pointer-like value (~470 TB), causing OOM.  Check
    _int_expr.is_constant() first; return *default* for dynamic dims.

    The default is 1 (minimum valid tensor dimension) rather than a larger value
    so that dummy tensors constructed for kernel compilation use the most compact
    shape.  CuTeDSL kernels bake the tensor layout into their type; using 1 for
    dynamic dims gives a [1, static_dim, ...] dummy whose row-major offset formula
    (offset = idx) is valid for any larger batch size at runtime.
    """
    if isinstance(d, int):
        return d
    ie = getattr(d, "_int_expr", None)
    if ie is not None:
        try:
            if ie.is_constant():
                return int(ie.get_constant_value())
        except (AttributeError, RuntimeError):
            # is_constant() or get_constant_value() may not be available on all
            # TRT builds; fall through to return the safe default.
            pass
        return default
    try:
        return int(d)
    except (TypeError, ValueError):
        return default


# ---- Dtype / format conversion ----


def td_dtype_to_torch(td_dtype: Any) -> torch.dtype:
    """Convert a TensorRT ``DataType`` to a ``torch.dtype``.

    Thin wrapper around ``torch_tensorrt.dtype`` so the annotation module
    stays aligned with the project-wide dtype bridge instead of carrying its
    own dispatch table.
    """
    from torch_tensorrt import dtype as _ttdtype

    return _ttdtype._from(td_dtype).to(torch.dtype)


_MAX_DIM = 2**31 - 1
_MIN_VALID_DIM = 1


def dtype_token(td: Any) -> str:
    """Return the AutoTuneCombination string token for a TensorDesc dtype.

    Used by ``_build_autotune_fn`` to construct the dtype string passed to
    ``trtp.AutoTuneCombination``.  Falls back to ``"FP32"`` for unrecognised
    dtypes so that autotune registration does not fail silently.
    """
    if td.dtype == trt.float16:
        return "FP16"
    if td.dtype == trt.bfloat16:
        return "BF16"
    if td.dtype == trt.float32:
        return "FP32"
    if td.dtype == trt.int32:
        return "INT32"
    if td.dtype == trt.int8:
        return "INT8"
    return "FP32"


def format_token(tf: Any) -> str:
    """Return a short string token for a trt.TensorFormat."""
    if tf == trt.TensorFormat.LINEAR:
        return "LINEAR"
    if tf == trt.TensorFormat.CHW2:
        return "CHW2"
    if tf == trt.TensorFormat.HWC8:
        return "HWC8"
    if tf == trt.TensorFormat.CHW4:
        return "CHW4"
    if tf == trt.TensorFormat.CHW16:
        return "CHW16"
    if tf == trt.TensorFormat.CHW32:
        return "CHW32"
    if tf == trt.TensorFormat.DHWC8:
        return "DHWC8"
    if tf == trt.TensorFormat.CDHW32:
        return "CDHW32"
    if tf == trt.TensorFormat.HWC:
        return "HWC"
    if tf == trt.TensorFormat.DLA_LINEAR:
        return "DLA_LINEAR"
    if tf == trt.TensorFormat.DLA_HWC4:
        return "DLA_HWC4"
    if tf == trt.TensorFormat.HWC16:
        return "HWC16"
    if tf == trt.TensorFormat.DHWC:
        return "DHWC"
    return str(tf)


# ---- Shape expression utilities ----


def _collect_shape_var_bindings(shape_expr: Any, bindings: Dict[int, int]) -> None:
    """Recursively find free/fake/dynamic shape vars and assign them the minimum valid value.

    Walks *shape_expr* recursively (handling nested tensors with a `.shape_expr`
    attribute and plain list/tuple containers) and populates *bindings* with a
    mapping from ``id(var)`` → 1 for every element that is not a plain ``int``
    and is either:
      - marked as fake (``is_fake == True``), or
      - a non-constant symbolic expression (``is_constant == False``), or
      - not directly convertible to int via ``int()``.

    The value 1 is the minimum positive integer accepted as a tensor dimension.
    Both ``is_fake`` fakes (from TRT's shape-inference placeholder pass) and
    true dynamic ``ShapeExpr`` dims (from dynamic-shape engines) are bound so
    that ``_shape_expr_to_ints`` can produce a concrete fallback shape for
    ``meta_impl`` even in dynamic-shape contexts.

    Mutates *bindings* in place.
    """
    for d in shape_expr:
        if hasattr(d, "shape_expr"):
            _collect_shape_var_bindings(d.shape_expr, bindings)
        elif isinstance(d, (list, tuple)):
            _collect_shape_var_bindings(d, bindings)
        elif not isinstance(d, int):
            is_symbolic = (
                getattr(d, "is_fake", False)
                or (hasattr(d, "is_constant") and not d.is_constant)
            )
            if not is_symbolic:
                # Last resort: try converting to int; if it fails, treat as symbolic.
                try:
                    int(d)
                except (TypeError, ValueError):
                    is_symbolic = True
            if is_symbolic:
                vid = id(d)
                if vid not in bindings:
                    bindings[vid] = _MIN_VALID_DIM


def _shape_elem_to_int(d: Any, bindings: Dict[int, int]) -> int:
    """Convert a single shape dimension to a concrete int.

    Resolution order:
    1. Check *bindings* (fake/symbolic vars already assigned a substitute value).
    2. Plain ``int`` — returned directly.
    3. ``int(d)`` — works for SymInt-like objects that support __int__.
    4. ``d.max()`` — upper bound from a symbolic range.
    5. ``d.value`` — direct value attribute (callable or plain).
    6. ``d.__index__()`` — integer protocol.
    7. ``d.constant_value`` — static constant (skipped if ``is_fake`` is True).
    8. ``d.eval(bindings)`` — explicit evaluation with the binding map.

    Raises:
        RuntimeError: If no resolution path succeeds, including the dimension
            type, so callers can identify which shape caused the failure.
    """
    vid = id(d)
    if vid in bindings:
        v = bindings[vid]
    elif isinstance(d, int):
        v = d
    else:
        v = None
        try:
            v = int(d)
        except (TypeError, ValueError):
            pass
        if v is None and hasattr(d, "max"):
            try:
                v = int(d.max())
            except (TypeError, ValueError):
                pass
        if v is None and hasattr(d, "value"):
            val = getattr(d, "value")
            try:
                v = int(val()) if callable(val) else int(val)
            except (TypeError, ValueError):
                pass
        if v is None and hasattr(d, "__index__"):
            try:
                v = d.__index__()
            except (TypeError, ValueError):
                pass
        if v is None and hasattr(d, "constant_value") and not getattr(d, "is_fake", True):
            try:
                cv = d.constant_value
                raw = cv() if callable(cv) else cv
                v = int(raw)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass
        if v is None and hasattr(d, "eval"):
            try:
                ev = d.eval(bindings)
                v = int(ev)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                pass
        if v is None:
            raise RuntimeError(
                f"Cannot convert shape_expr element (type {type(d).__name__!r}) to int"
            )
    if v < 0 or v > _MAX_DIM:
        raise RuntimeError(
            f"shape_expr dimension {v} out of range [0, {_MAX_DIM}]"
            f" (element type {type(d).__name__!r})"
        )
    return v


def _shape_elem_concrete_without_fake(d: Any) -> bool:
    """Return True if this dimension can be resolved to int without a fake-variable substitution.

    "Concrete without fake" means the dimension has a knowable integer value
    that does not rely on an artificial placeholder assigned to a free symbolic
    variable (i.e. a variable whose ``is_fake`` attribute is True).  Such fake
    variables arise from TRT's dynamic-shape mechanism: during the first-call
    meta_impl pass, shape variables that are not yet bound to any concrete size
    appear as fake SymInt32 objects.

    Concreteness is tested in the following order:
    1. Plain ``int`` — always concrete.
    2. ``int(d)`` succeeds — the object can convert itself to an integer.
    3. ``d.is_fake`` is True — explicitly marked as fake, so **not** concrete.
    4. ``d.constant_value`` resolves without error — a static symbolic constant.
    5. ``d.max()`` resolves without error — a bounded symbolic dim's upper bound.

    If none of those paths succeeds the dimension is considered not concrete.
    """
    if isinstance(d, int):
        return True
    try:
        int(d)
        return True
    except (TypeError, ValueError):
        pass
    if hasattr(d, "is_fake") and getattr(d, "is_fake"):
        return False
    if hasattr(d, "constant_value"):
        try:
            cv = d.constant_value
            raw = cv() if callable(cv) else cv
            int(raw)
            return True
        except (TypeError, ValueError, AttributeError, RuntimeError):
            pass
    if hasattr(d, "max"):
        try:
            int(d.max())
            return True
        except (TypeError, ValueError):
            pass
    return False


def _shape_expr_is_concrete(shape_expr: Any) -> bool:
    """Return True if every element can be resolved to int without fake/symbolic substitution.

    Recursively descends into nested tensors (via `.shape_expr`) and
    list/tuple containers.
    """
    for d in shape_expr:
        if hasattr(d, "shape_expr"):
            if not _shape_expr_is_concrete(d.shape_expr):
                return False
        elif isinstance(d, (list, tuple)):
            if not _shape_expr_is_concrete(d):
                return False
        elif not _shape_elem_concrete_without_fake(d):
            return False
    return True


def _shape_expr_to_ints(shape_expr: Any) -> List[int]:
    """Convert a TRT shape_expr to a list of concrete ints.

    Free/fake shape variables are assigned the minimum valid value (1) via
    ``_collect_shape_var_bindings`` before evaluation so that the result is
    always a valid concrete shape even in dynamic-shape contexts.
    """
    bindings: Dict[int, int] = {}
    _collect_shape_var_bindings(shape_expr, bindings)

    def eval_rec(expr: Any) -> List[int]:
        result: List[int] = []
        for d in expr:
            if hasattr(d, "shape_expr"):
                result.extend(eval_rec(d.shape_expr))
            elif isinstance(d, (list, tuple)):
                result.extend(eval_rec(d))
            else:
                result.append(_shape_elem_to_int(d, bindings))
        return result

    return eval_rec(shape_expr)


def _shape_expr_to_meta_shape(shape_expr: Any) -> Tuple[int, ...]:
    """Convert a TRT shape_expr to a shape tuple of ints for meta tensors.

    Fake/symbolic dims get unique placeholder ints (1, 2, 3, ...).  The counter
    is local to each call so values are always 1, 2, 3, ... regardless of prior
    calls — making the output fully deterministic across repeated runs of the
    same program.

    This differs from ``_shape_expr_to_ints`` in that distinct fake/symbolic
    dims receive distinct placeholder values, which preserves shape-broadcasting
    semantics when creating meta tensors for first-call shape inference.
    """
    bindings: Dict[int, int] = {}
    _collect_shape_var_bindings(shape_expr, bindings)
    placeholder_cache: Dict[int, int] = {}
    _counter = [1]  # mutable cell; local to this call

    def eval_rec(expr: Any) -> List[int]:
        result: List[int] = []
        for d in expr:
            if hasattr(d, "shape_expr"):
                result.extend(eval_rec(d.shape_expr))
            elif isinstance(d, (list, tuple)):
                result.extend(eval_rec(d))
            elif _shape_elem_concrete_without_fake(d):
                result.append(_shape_elem_to_int(d, bindings))
            else:
                key = id(d)
                if key not in placeholder_cache:
                    placeholder_cache[key] = _counter[0]
                    _counter[0] += 1
                result.append(placeholder_cache[key])
        return result

    return tuple(eval_rec(shape_expr))


# ---- TensorDesc helpers ----


# ---- Artifact dump helper ----


def dump_code_artifact(
    env_key: str,
    filename: str,
    content: Union[str, bytes],
    default_dir: str = "",
) -> None:
    """Write content to $env_key/<filename>, silently ignoring all errors.

    Justified as shared: Triton and CuTile both have near-identical
    try: makedirs; write to env-dir; except: pass blocks. CuTeDSL uses
    cute.compile(options="--dump-dir=...") so it does not need this helper.

    Args:
        env_key:     Environment variable naming the target directory.
        filename:    File name (not path) to write inside that directory.
        content:     str or bytes content to write.
        default_dir: Fallback directory if the env var is unset or empty.
    """
    import os as _os

    dump_dir = _os.environ.get(env_key) or default_dir
    if not dump_dir:
        return
    try:
        _os.makedirs(dump_dir, exist_ok=True)
        path = _os.path.join(dump_dir, filename)
        mode = "wb" if isinstance(content, bytes) else "w"
        with open(path, mode) as f:
            f.write(content)
    except (OSError, IOError):
        # Broad OS/IO catch is intentional: this is a best-effort debug dump.
        # Failures (e.g. read-only filesystem, missing env var dir, permission
        # denied) must never propagate to the caller and abort compilation.
        pass


# ---- Generic sandbox runner ----


def run_kernel_sandbox(
    launch_fn: Any,
    host_args: List[Any],
    is_kernel_fn: Callable[[Any], bool],
    recorder_factory: Callable[[Any], Any],
    raw_fn: Any = None,
    host_kwargs: Optional[Dict[str, Any]] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
    strict: bool = False,
    op: str = "<run_kernel_sandbox>",
    backend: str = "unknown",
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """Run launch_fn in a sandbox, replacing kernel objects with recorder proxies.

    All three backends share the same module-discovery + proxy-injection +
    sandbox-run pattern.  Each backend provides:
    - is_kernel_fn: predicate to detect backend kernel objects in module globals.
    - recorder_factory: callable(kernel_obj) → recorder proxy.
    - raw_fn: (CuTeDSL only) the unwrapped Python function to sandbox instead of
      launch_fn (which may be a @cute.jit wrapper without __code__).
    - host_kwargs: extra keyword args passed to the sandboxed function call
      (Triton passes cfg kwargs; CuTile passes cfg kwargs; CuTeDSL passes none).
    - extra_overrides: additional name→value overrides injected into the sandbox
      module globals alongside recorder proxies (CuTile uses this for its
      patched `ct` module).
    - strict: if True, raise QDPRuntimeError when module cannot be found or no
      kernel objects exist; if False (default), return (None, {}) instead.

    Returns:
        (used_recorder_or_None, all_recorders_dict)
        used_recorder_or_None is the first recorder whose grid/args was set.
        all_recorders_dict is keyed by module-global name.

    Note on silent-exception policy:
        For CuTeDSL the @cute.jit wrapper may not expose __code__, so raw_fn
        could be None.  When strict=False and module discovery fails, or when
        the sandbox run raises, we return (None, {}) — the caller falls back
        to a (1,1,1) grid.  Triton and CuTile use strict=True so that
        missing-module and no-kernel errors are surfaced clearly.
    """
    fn_to_sandbox = raw_fn if raw_fn is not None else launch_fn
    host_mod_name = getattr(fn_to_sandbox, "__module__", None)
    module_obj = None
    if host_mod_name:
        try:
            module_obj = importlib.import_module(host_mod_name)
        except ImportError:
            pass

    if module_obj is None:
        if strict:
            raise QDPRuntimeError(
                op=op,
                stage="aot_impl",
                backend=backend,
                msg=f"cannot import module for launch_fn {launch_fn}",
            )
        return None, {}

    recorders: Dict[str, Any] = {
        name: recorder_factory(obj)
        for name, obj in vars(module_obj).items()
        if is_kernel_fn(obj)
    }

    if not recorders:
        if strict:
            raise QDPRuntimeError(
                op=op,
                stage="aot_impl",
                backend=backend,
                msg=f"no kernel objects matching predicate found in module {host_mod_name}",
            )
        return None, {}

    overrides: Dict[str, Any] = dict(recorders)
    if extra_overrides:
        overrides.update(extra_overrides)

    try:
        sandboxed = make_sandboxed_host(fn_to_sandbox, overrides)
        sandboxed(*host_args, **(host_kwargs or {}))
    except Exception:
        # Broad catch is intentional: the sandbox runs user-supplied launch
        # functions with fake/symbolic tensor proxies.  Any exception — from
        # wrong kernel argument types to missing attributes on proxy objects —
        # is a sandbox-execution failure, not a programming error in this
        # module.  When strict=True the original exception propagates so the
        # caller can surface it; otherwise we return (None, {}) and let the
        # caller fall back to a default (1,1,1) grid.
        if strict:
            raise
        return None, {}

    used = [
        r for r in recorders.values()
        if getattr(r, "grid", None) is not None or getattr(r, "args", None) is not None
    ]
    used_recorder = used[0] if used else None
    return used_recorder, recorders


# ---- Launch params helper ----


def _launch_params_from_trt(
    launch: Any,
    extra_args: Any,
    num_inputs: int = 1,
    num_outputs: int = 1,
) -> types.SimpleNamespace:
    """Build a SimpleNamespace from a trtp.KernelLaunchParams and SymIntExprs.

    Justified as shared: all three backends produce a KernelLaunchParams object
    with identical grid_x/y/z and block_x/y/z attributes, and the conversion
    to SimpleNamespace is identical in all three AOT files (only the default
    block_x fallback differs, but block_x is always set explicitly at the call
    site before this function is called, so the default is never exercised).

    Args:
        launch:      A ``trtp.KernelLaunchParams`` instance with grid_x/y/z and
                     block_x/y/z attributes (missing attributes default to 1).
        extra_args:  SymIntExprs list to forward as ``sym_int_exprs``.
        num_inputs:  Number of input tensor bindings; used to build
                     ``param_binding_indices`` as ``[0..num_inputs+num_outputs)``.
        num_outputs: Number of output tensor bindings.

    Returns:
        A ``types.SimpleNamespace`` with fields: grid, block, shared_mem,
        param_binding_indices, sym_int_exprs.
    """
    grid = (
        getattr(launch, "grid_x", 1),
        getattr(launch, "grid_y", 1),
        getattr(launch, "grid_z", 1),
    )
    block = (
        getattr(launch, "block_x", 1),
        getattr(launch, "block_y", 1),
        getattr(launch, "block_z", 1),
    )
    return types.SimpleNamespace(
        grid=grid,
        block=block,
        shared_mem=getattr(launch, "shared_mem", 0),
        param_binding_indices=list(range(num_inputs))
        + list(range(num_inputs, num_inputs + num_outputs)),
        sym_int_exprs=extra_args,
    )


# ---- Launch-arg analysis ----


def analyze_launch_args(
    *,
    args: Sequence[Any],
    num_inputs: int,
    num_outputs: int,
    op: str,
    backend: str,
) -> Tuple[List[int], List[Any]]:
    """Split recorded kernel call arguments into pointer bindings and scalar SymInts.

    Enforces the contract: all tensor pointer arguments must appear before
    all scalar arguments in the backend kernel call.

    Args:
        args:        Positional arguments recorded from the kernel call, each
                     either a ``SymbolicTensor`` (pointer binding) or a scalar
                     (``trtp.SymInt32``, ``int``, or SymIntExpr with ``._expr``).
        num_inputs:  Number of input tensor bindings declared for this op.
        num_outputs: Number of output tensor bindings declared for this op.
        op:          Op name included in error messages for diagnostics.
        backend:     Backend name included in error messages for diagnostics.

    Returns:
        (param_binding_indices, scalar_symints)
        - param_binding_indices[k] = b means kernel pointer parameter k
          maps to exported tensor binding b in [inputs..., outputs...].
        - scalar_symints is a list of trtp.SymInt32 for scalar kernel args.

    Raises:
        QDPRuntimeError: If a tensor argument follows a scalar argument, if an
            input/output index is out of range, if a tensor has an unknown role,
            or if an argument type is not supported.
    """

    def compute_binding_index(st: SymbolicTensor) -> int:
        if st.role is TensorRole.INPUT:
            if st.index < 0 or st.index >= num_inputs:
                raise QDPRuntimeError(
                    op=op,
                    stage="aot_impl",
                    backend=backend,
                    msg=f"invalid input index {st.index} for {num_inputs} inputs",
                )
            return st.index
        if st.role is TensorRole.OUTPUT:
            if st.index < 0 or st.index >= num_outputs:
                raise QDPRuntimeError(
                    op=op,
                    stage="aot_impl",
                    backend=backend,
                    msg=f"invalid output index {st.index} for {num_outputs} outputs",
                )
            return num_inputs + st.index
        raise QDPRuntimeError(
            op=op,
            stage="aot_impl",
            backend=backend,
            msg=f"unexpected tensor role {st.role}",
        )

    param_binding_indices: List[int] = []
    scalar_symints: List[Any] = []
    seen_scalar = False

    for a in args:
        if isinstance(a, SymbolicTensor):
            if seen_scalar:
                raise QDPRuntimeError(
                    op=op,
                    stage="aot_impl",
                    backend=backend,
                    msg=(
                        "backend kernel arguments must be ordered as "
                        "[all tensor pointers..., then all scalars...]; "
                        "found a tensor argument after scalar arguments"
                    ),
                )
            param_binding_indices.append(compute_binding_index(a))
        elif _TRT_AVAILABLE and isinstance(a, trtp.SymInt32):
            scalar_symints.append(a)
            seen_scalar = True
        elif isinstance(a, int):
            scalar_symints.append(a)
            seen_scalar = True
        elif _TRT_AVAILABLE and hasattr(a, "_expr"):
            scalar_symints.append(a)
            seen_scalar = True
        else:
            raise QDPRuntimeError(
                op=op,
                stage="aot_impl",
                backend=backend,
                msg=(
                    f"unsupported launch argument type {type(a)!r}; "
                    "only SymbolicTensor, SymInt32, and int scalars are allowed"
                ),
            )

    return param_binding_indices, scalar_symints
