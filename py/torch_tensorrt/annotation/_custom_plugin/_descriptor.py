"""tta.custom_plugin: CustomPluginSpec, factory, and QDP registration.

This module provides:
- CustomPluginSpec: the descriptor returned by tta.custom_plugin(...)
- custom_plugin(): factory that computes a deterministic QDP op_name
- register_custom_plugin(): registers @trtp.register / @trtp.autotune / @trtp.aot_impl
- lower_custom_plugin_descriptor(): lowers a CustomPluginSpec to a
  TRT plugin layer via trtp.op

Backend-specific AOT logic lives in _triton_aot / _cutile_aot /
_cutedsl_aot.
"""
from __future__ import annotations

import inspect
import logging
import threading
import typing

import torch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

try:
    import tensorrt as trt
    import tensorrt.plugin as trtp

    _TRT_AVAILABLE = True
except ImportError:
    _TRT_AVAILABLE = False
    trt = None  # type: ignore[assignment]
    trtp = None  # type: ignore[assignment]

from ._qdp_utils import (
    QDPRuntimeError,
    TacticEntry,
    build_tactic_table,
    derive_impl_id,
    dtype_token,
    format_token,
    make_qdp_symbol,
)
from .._specs import CuTeDSLSpec, CuTileSpec, TritonSpec
from ._symbolic import SymbolicTensor, TensorRole

KernelSpec = Union[TritonSpec, CuTileSpec, CuTeDSLSpec]

# ---------------------------------------------------------------------------
# Process-level QDP registration registry
# ---------------------------------------------------------------------------
# TRT's internal QDP registry is process-global, so we track registered op_names
# at the process level rather than using thread-local storage.  Thread-local storage
# would allow two concurrent threads (e.g. pytest-xdist workers sharing a process)
# to each attempt QDP registration for the same op_name, causing TRT to raise an
# "already registered" error on the second attempt.
#
# Threading contract for _qdp_registered_ops:
#   WRITE: Always done under _qdp_registration_lock.  The set is only grown, never
#          shrunk, so writes are monotonic.  The final add() and the return from
#          register_custom_plugin() are both inside the lock.
#   READ (under lock): Always safe; used inside register_custom_plugin() after
#          acquiring _qdp_registration_lock for the definitive TOCTOU-safe check.
#   READ (without lock, fast path): Safe because the set only grows.  A thread
#          that observes op_name IN the set can safely skip registration — the
#          worst outcome of a race is a redundant lock acquisition on the slow
#          path, which is also safe.  A thread that observes op_name NOT IN the
#          set proceeds to acquire the lock and re-checks inside (double-checked
#          locking pattern).  This avoids lock contention on the common post-
#          registration path without risking double-registration.
_qdp_registered_ops: set = set()
_qdp_registration_lock = threading.Lock()

# Thread-local cache for _aot_fn builders.  Building the closure is cheap,
# but we avoid redundant work within a single thread.
_tls = threading.local()


def _get_aot_fn_cache() -> Dict[Tuple[str, int], Callable[..., Any]]:
    """Return the thread-local cache mapping (op_name, num_inputs) to built _aot_fn.

    The cache is never cleared because TRT's QDP registry is also process-persistent,
    so the two remain in sync without any explicit eviction.
    """
    if not hasattr(_tls, "aot_fn_cache"):
        _tls.aot_fn_cache = {}
    return _tls.aot_fn_cache


# ---------------------------------------------------------------------------
# Public descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CustomPluginSpec:
    """Descriptor returned by ``tta.custom_plugin(...)``.

    Lifecycle
    ---------
    1. **Creation** — ``custom_plugin()`` constructs a ``CustomPluginSpec``
       by splitting kwargs into weights/attrs, hashing the kernel specs
       to derive a deterministic ``op_name``, and probing ``meta_impl`` with dummy
       meta tensors to count outputs.  No TRT objects are touched at this stage.

    2. **QDP registration** — ``register_custom_plugin()`` (called lazily from
       ``lower_custom_plugin_descriptor()``) registers three QDP callbacks with
       TRT's process-global plugin registry under ``op_name``:

       * ``@trtp.register``   — shape/dtype descriptor (uses ``meta_impl`` or identity)
       * ``@trtp.autotune``   — enumerates (dtype, format, tactic) combinations
       * ``@trtp.aot_impl``   — AOT kernel dispatch (Triton / CuTile / CuTeDSL)

       Registration is idempotent: a process-level set (``_qdp_registered_ops``)
       guards against double-registration across threads.

    3. **Use in lowering** — ``lower_custom_plugin_descriptor()`` calls
       ``ctx.net.add_plugin(trtp.op.<ns>.<name>(*trt_inputs), aot=True)`` to add
       the plugin layer to the TRT network.  Weight tensors declared in kwargs are
       injected as ``trt.add_constant`` layers and appended to ``trt_inputs`` before
       the plugin call so the launch_fn receives
       ``(*activations, *weights_in_order, *outputs, ...)``.

    Attributes:
        op_name:     Unique QDP symbol, e.g. ``"tta_custom::host_kernel_a1b2c3d4"``.
                     Deterministically derived from the kernel specs and attrs so that
                     re-creating the same descriptor in a different process produces
                     the same name.
        specs:       Non-empty list of kernel specs (TritonSpec, CuTileSpec,
                     CuTeDSLSpec).  Multiple specs provide alternative tactics; TRT's
                     autotune selects the fastest one at engine-build time.
        meta_impl:   PyTorch meta function for shape/dtype inference.  Receives meta
                     tensors (one per plugin input) and must return a single
                     ``torch.Tensor`` or a tuple/list of ``torch.Tensor`` s.
        num_outputs: *Deprecated internal field — do not rely on this value.*
                     Retained for backward compatibility only.  The QDP
                     registration path re-infers the output count from real
                     ``trt.ITensor`` input ranks at lowering time.
                     :meth:`auto_register_torch_op` infers it independently
                     at schema-registration time.
        attrs:       Scalar kwargs baked into kernel PTX at AOT time
                     (e.g. ``addend=1.0``).  NOT forwarded as TRT plugin fields.
        weights:     Tensor kwargs bound at plugin creation time.  At lowering, each
                     weight is added to the TRT network as a constant tensor and
                     appended to the dynamic activation inputs.  The launch_fn must
                     accept dynamic inputs first, then weights (in declaration order),
                     then outputs.  Named so debug messages can identify each weight.
    """

    op_name: str
    specs: List[KernelSpec]
    meta_impl: Callable[..., Any]
    num_outputs: int = 1
    attrs: Dict[str, Any] = field(default_factory=dict)
    # Tensor weights are excluded from __hash__ and __eq__ because torch.Tensor
    # is not hashable.  Identity / equality of a descriptor is captured by
    # op_name (which already encodes num_weights via derive_impl_id).
    weights: Dict[str, "torch.Tensor"] = field(
        default_factory=dict, hash=False, compare=False
    )

    def lower_to_trt(
        self,
        ctx: Any,
        trt_inputs: List[Any],
        name: str,
        qdp_name: Optional[str] = None,
    ) -> Any:
        """Lower this spec to a TRT ``IPluginV3`` layer via :func:`lower_custom_plugin_descriptor`.

        Shared entry-point used by both the native TTA lowering pass and the
        Dynamo integration path (``trt_plugins.custom_op(impl=...)``).  Using
        this method ensures both paths go through the same code: weight
        injection, TTA layer metadata, and ``aot=True`` semantics.

        When ``qdp_name`` is supplied the plugin is looked up under that name
        (the torch op name, e.g. ``"ns::my_op"``) rather than the auto-derived
        TTA fingerprint in ``self.op_name``.  This is required when the plugin
        was registered under the torch op name rather than the TTA fingerprint
        (e.g. via :func:`register_custom_plugin` with an explicit ``qdp_name``).

        Args:
            ctx:        Torch-TRT ``ConversionContext`` (carries ``ctx.net``).
            trt_inputs: Ordered ``trt.ITensor`` activation inputs.
            name:       Layer name for TRT debugging/profiling.
            qdp_name:   Optional QDP name override (e.g. torch op name).

        Returns:
            A single ``trt.ITensor`` or a tuple of ``trt.ITensor`` s.
        """
        import dataclasses
        desc = dataclasses.replace(self, op_name=qdp_name) if qdp_name is not None else self
        return lower_custom_plugin_descriptor(ctx, desc, trt_inputs, name)

    def auto_register_torch_op(self, op_name: str) -> None:
        """Auto-register ``torch.library.custom_op`` and ``register_fake`` for ``op_name``.

        Eliminates the boilerplate of writing ``@torch.library.custom_op`` and
        ``@torch.library.register_fake`` by hand when a :class:`CustomPluginSpec`
        already carries ``meta_impl`` and kernel specs.

        The eager implementation calls the first spec's ``launch_fn`` with the
        first available config.  The fake implementation delegates to
        ``meta_impl`` for shape/dtype inference.

        If the op is already registered in ``torch.ops``, the call is a no-op.

        Args:
            op_name: torch op name in ``"namespace::name"`` form.

        Raises:
            ValueError: If ``meta_impl`` is ``None`` (required for shape inference).
        """
        if self.meta_impl is None:
            raise ValueError(
                f"auto_register_torch_op: meta_impl is required to auto-register "
                f"'{op_name}'; set meta_impl in tta.custom_plugin(..., meta_impl=...)"
            )

        namespace, name = op_name.split("::", 1)

        # Skip if the op is already registered.
        ns_obj = getattr(torch.ops, namespace, None)
        if ns_obj is not None and hasattr(ns_obj, name):
            return

        meta_sig = inspect.signature(self.meta_impl)
        param_names = list(meta_sig.parameters.keys())

        first_spec = self.specs[0]
        first_config = first_spec.configs[0] if getattr(first_spec, "configs", None) else {}
        _launch = first_spec.launch_fn
        _meta = self.meta_impl

        # Infer num_outputs for the torch.library schema.
        from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
            _probe_num_outputs_from_callable,
        )
        _n_args = len(inspect.signature(_meta).parameters)
        _num_outputs = _probe_num_outputs_from_callable(_meta, _n_args)

        # CuTeDSL launch functions are @cute.jit decorated, which means they only
        # accept cute.Tensor arguments (not torch.Tensor).  The eager path receives
        # plain torch.Tensor inputs/outputs from PyTorch dispatch, so we wrap the
        # launch function with a DLPack bridge that converts torch.Tensor → cute.Tensor
        # before forwarding to the jit-compiled kernel.
        #
        # Why not put this conversion inside the user's launch_fn?  Because
        # cute.compile() (called in the AOT path) requires its argument to be
        # @cute.jit decorated.  A regular Python wrapper around a @cute.jit function
        # is NOT itself @cute.jit, so cute.compile() would raise:
        #   DSLRuntimeError: Function <...> is not decorated with jit decorator.
        # Therefore TTA must own the eager-path conversion — the user's launch_fn
        # must remain a pure @cute.jit function.
        if isinstance(first_spec, CuTeDSLSpec):
            _jit_launch = _launch

            def _launch(*args: Any, **kwargs: Any) -> Any:
                from cutlass.cute.runtime import from_dlpack as _from_dlpack

                converted = tuple(
                    _from_dlpack(a.contiguous()) if isinstance(a, torch.Tensor) else a
                    for a in args
                )
                return _jit_launch(*converted, **kwargs)

        def _eager_body(*args: torch.Tensor) -> Any:
            meta_outs = _meta(*args)
            if not isinstance(meta_outs, (tuple, list)):
                meta_outs = (meta_outs,)
            outs = [
                torch.empty(o.shape, dtype=o.dtype, device=args[0].device)
                for o in meta_outs
            ]
            _launch(*args, *outs, **first_config)
            # Count dynamically from meta_impl result — no _num_outputs needed here.
            return outs[0] if len(outs) == 1 else list(outs)

        # Reuse the same __signature__ trick as _build_desc_fn: attach a custom
        # inspect.Signature so torch.library's schema inference sees real named
        # Tensor parameters without exec()-generated source code.
        # Multi-output ops use List[torch.Tensor] (→ schema "Tensor[]") so that
        # torch.library registers the correct return type.
        sig_params = [
            inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=torch.Tensor)
            for p in param_names
        ]
        ret_annotation = torch.Tensor if _num_outputs == 1 else List[torch.Tensor]
        sig = inspect.Signature(sig_params, return_annotation=ret_annotation)
        _eager_body.__signature__ = sig
        torch.library.custom_op(op_name, mutates_args=())(_eager_body)

        # Fake impl: forward to meta_impl for shape/dtype inference during tracing.
        # For multi-output ops, return a list to match the "Tensor[]" schema.
        # Count dynamically from result — real-ranked tensors are available here.
        def _fake_body(*args: torch.Tensor) -> Any:
            result = _meta(*args)
            if isinstance(result, tuple):
                return list(result)
            return result

        _fake_body.__signature__ = sig
        torch.library.register_fake(op_name)(_fake_body)




def custom_plugin(
    kernel: Union[KernelSpec, List[KernelSpec]],
    meta_impl: Callable[..., Any],
    **kwargs: Any,
) -> CustomPluginSpec:
    """Create a :class:`CustomPluginSpec` for one or more kernel specs.

    This is the primary entry point for the ``tta.custom_plugin`` API.  The
    returned descriptor is used as the ``impl=`` argument of ``@tta.export_as``.

    Args:
        kernel:    Single kernel spec or a non-empty list of specs.  Multiple specs
                   provide alternative tactics; TRT's autotune benchmarks all of them
                   at engine-build time and selects the fastest.
        meta_impl: Required.  PyTorch meta function used by QDP for shape/dtype
                   inference.  Receives meta tensors (one per plugin input) and
                   must return a single ``torch.Tensor`` or a tuple/list of
                   ``torch.Tensor`` s.  Each tensor's ``.shape`` and ``.dtype``
                   define the output ``TensorDesc`` s.  For the descriptor we use
                   only the first returned tensor; its shape and dtype must match
                   the plugin's actual output so TRT gets correct types and shapes.
        **kwargs:  Plugin-level keyword arguments, split by type at creation time:

                   * ``torch.Tensor`` values → **weights**: frozen tensors bound
                     to this plugin.  At TRT lowering each weight is added to the
                     network as a ``trt.add_constant`` layer and appended to the
                     dynamic activation inputs before calling the plugin.  The
                     launch_fn must accept ``(*activations, *weights,
                     *outputs, ...)``.  The count of weights is included in the
                     ``op_name`` fingerprint so the same kernel spec can be
                     registered with different input arities without stale-
                     registration bugs.

                   * All other values → **attrs**: scalar compile-time constants
                     (e.g. ``addend=1.0``, ``scale=2``).  These are baked into
                     the kernel PTX at AOT time and are NOT forwarded as TRT
                     plugin fields.

    Returns:
        A :class:`CustomPluginSpec` with an auto-computed ``op_name``.

    Raises:
        ValueError: If ``kernel`` is an empty list, or if ``meta_impl`` is ``None``.
        TypeError:  If any element of ``kernel`` is not a valid ``KernelSpec``,
                    or if ``meta_impl`` is not callable.
    """
    specs: List[KernelSpec] = kernel if isinstance(kernel, list) else [kernel]
    if not specs:
        raise ValueError("custom_plugin: kernel list cannot be empty")
    for s in specs:
        if not isinstance(s, (TritonSpec, CuTileSpec, CuTeDSLSpec)):
            raise TypeError(
                f"custom_plugin: kernel must be TritonSpec, CuTileSpec, or "
                f"CuTeDSLSpec, got {type(s).__name__!r}"
            )
    if meta_impl is None:
        raise ValueError(
            "custom_plugin: meta_impl is required and cannot be None; "
            "provide a PyTorch meta function for shape/dtype inference"
        )
    if not callable(meta_impl):
        raise TypeError(
            f"custom_plugin: meta_impl must be callable, "
            f"got {type(meta_impl).__name__!r}"
        )

    # Split kwargs by value type:
    #   torch.Tensor    → weights (TRT constant layers injected at lowering)
    #   everything else → attrs (scalar compile-time constants)
    weights: Dict[str, torch.Tensor] = {}
    attrs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            weights[k] = v
        else:
            attrs[k] = v

    impl_id = derive_impl_id(specs, attrs=attrs, num_weights=len(weights))
    op_name = make_qdp_symbol(impl_id)
    return CustomPluginSpec(
        op_name=op_name, specs=specs, meta_impl=meta_impl,
        attrs=attrs, weights=weights,
    )


# ---------------------------------------------------------------------------
# Shared parameter-list helpers
# ---------------------------------------------------------------------------


def _build_input_params(
    num_inputs: int,
    annotation: Any,
    names: Optional[List[str]] = None,
) -> List[inspect.Parameter]:
    """Build a positional ``inspect.Parameter`` list for TRT descriptor functions.

    Args:
        num_inputs:  Number of input parameters to generate.
        annotation:  Type annotation attached to each parameter (typically
                     ``trtp.TensorDesc``).
        names:       Optional explicit parameter names — required when the
                     desc/autotune functions must match a peer registration's
                     parameter names (TRT's ``@trtp.autotune`` validator
                     rejects mismatched names against the previously registered
                     ``@trtp.register`` desc).  When ``None`` (default) the
                     generic ``inp0`` … ``inpN`` names are used.

    Returns:
        List of ``inspect.Parameter`` objects suitable for use with
        ``inspect.Signature``.
    """
    if names is None:
        names = [f"inp{i}" for i in range(num_inputs)]
    assert len(names) == num_inputs, (
        f"expected {num_inputs} names, got {len(names)}"
    )
    return [
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation)
        for n in names
    ]


# ---------------------------------------------------------------------------
# Descriptor function builder (shape / dtype via @trtp.register)
# ---------------------------------------------------------------------------


def _build_desc_fn(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    num_outputs: int = 1,
) -> Callable[..., Any]:
    """Build the ``@trtp.register`` meta handler for a :class:`CustomPluginSpec`.

    Uses ``meta_impl`` (if provided) to infer output ``TensorDesc`` s; falls back
    to mirroring ``inp0`` (same shape/dtype) when ``meta_impl`` is ``None``.

    We use a ``*args`` closure and attach a custom ``inspect.Signature`` so that
    TRT's ``@trtp.register`` validation (``issubclass(param.annotation, TensorDesc)``)
    sees real ``TensorDesc`` class objects — not strings that would result from
    exec()-ing code in a module with ``from __future__ import annotations``.

    For ``num_outputs > 1``, the function returns a tuple of ``TensorDesc`` s (one
    per output), and ``meta_impl`` must return a tuple/list of that many tensors.

    Note: attrs are intentionally excluded from the descriptor signature.
    LIMITATION (workaround for NumPy 1.25+ incompatibility): TRT stores plugin
    field values as numpy arrays and calls ``attr_type_annot(f.data)`` to
    convert them back.  In NumPy 1.25+ this raises
    ``"only 0-dimensional arrays can be converted to Python scalars"`` for 1-D
    arrays.  Until TRT's plugin API is updated to handle NumPy 1.25+, we avoid
    the problem entirely by not registering any attrs.  Plugins that need to
    pass scalar hyperparameters (e.g. epsilon, axis) currently have no way to
    do so via the attrs mechanism and must bake constants into the kernel or
    pass them as additional tensor inputs.

    Args:
        descriptor:  The :class:`CustomPluginSpec` being registered.
        num_inputs:  Number of input ``TensorDesc`` positional parameters.
        num_outputs: Number of output ``TensorDesc`` s to produce (default 1).

    Returns:
        A callable with the correct ``inspect.Signature`` for ``@trtp.register``.
    """
    tensor_desc_cls = trtp.TensorDesc

    # meta_impl expects only the dynamic activation inputs, not the weight
    # inputs (which are TRT constant layers appended after activations).
    num_dynamic = num_inputs - len(descriptor.weights)
    from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
        _build_symbolic_desc_fn,
    )
    _desc = _build_symbolic_desc_fn(descriptor.meta_impl, num_dynamic, num_outputs)

    sig_params = _build_input_params(num_inputs, tensor_desc_cls)
    if num_outputs == 1:
        return_annotation = tensor_desc_cls
    else:
        # Build Tuple[TensorDesc, TensorDesc, ...] (n elements).
        # Subscripting typing.Tuple with a plain Python tuple of type args
        # unpacks them as separate type parameters (identical to writing
        # typing.Tuple[X, X] for n=2).  Do NOT use a list — that would
        # produce Tuple[List[X]] instead of Tuple[X, X].
        return_annotation = typing.Tuple[tuple([tensor_desc_cls] * num_outputs)]
    _desc.__signature__ = inspect.Signature(
        sig_params, return_annotation=return_annotation
    )
    return _desc


# ---------------------------------------------------------------------------
# Autotune function builder (format / tactic combinations via @trtp.autotune)
# ---------------------------------------------------------------------------


def _format_token_for_tactic(
    entry: TacticEntry,
    specs: List[KernelSpec],
) -> str:
    """Return the ``AutoTuneCombination`` format string for a single tactic entry.

    Looks up the first ``input_formats`` entry on the kernel spec selected by
    ``entry.spec_idx``.  Falls back to ``"LINEAR"`` if ``input_formats`` is
    absent or empty, or if the format is not in the known token map.

    Args:
        entry: The tactic table entry identifying the spec and config indices.
        specs: The full list of kernel specs from the descriptor.

    Returns:
        A format token string such as ``"LINEAR"`` or ``"CHW32"``.
    """
    spec = specs[entry.spec_idx]
    input_fmts = getattr(spec, "input_formats", None)
    if input_fmts:
        return format_token(input_fmts[0])
    return "LINEAR"


def _build_autotune_fn(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    num_outputs: int,
    tactic_table: List[TacticEntry],
    tensor_arg_names: Optional[List[str]] = None,
) -> Optional[Callable[..., Any]]:
    """Build a ``@trtp.autotune`` function that registers dtype/format/tactic combinations.

    Returns ``None`` if the QDP autotune API (``trtp.autotune`` /
    ``trtp.AutoTuneCombination``) is not available in this TRT build.

    TRT calls this callback during engine build to enumerate valid
    (dtype, format, tactic) combinations.  It then benchmarks all valid
    combinations and passes the winning 1-based tactic ID to ``aot_impl``.

    ``AutoTuneCombination`` uses the string constructor::

        AutoTuneCombination(dtype_str, format_str, tactic_ids)

    where:

    * ``dtype_str`` — comma-separated per-I/O dtype options,
      e.g. ``"FP32|FP16, FP32|FP16"``
    * ``format_str`` — memory format, e.g. ``"LINEAR"``
    * ``tactic_ids`` — list of 1-based integer tactic IDs (0 is reserved by TRT)

    The autotune function signature must match::

        (inp0: TensorDesc, ..., inpN: TensorDesc,
         outputs: Tuple[TensorDesc]) -> List[AutoTuneCombination]

    Args:
        descriptor:    The :class:`CustomPluginSpec` being registered.
        num_inputs:    Number of input ``TensorDesc`` positional parameters.
        num_outputs:   Number of output tensors (used to iterate ``outputs``).
        tactic_table:  Pre-built list of :class:`TacticEntry` objects.

    Returns:
        A callable with the correct ``inspect.Signature`` for ``@trtp.autotune``,
        or ``None`` if the autotune API is unavailable.
    """
    if not (
        _TRT_AVAILABLE
        and hasattr(trtp, "autotune")
        and hasattr(trtp, "AutoTuneCombination")
    ):
        return None

    n_tactics = len(tactic_table)
    # Tactic IDs are 1-based: TRT reserves 0 as the "no-autotune" default.
    tactic_ids = list(range(1, n_tactics + 1))

    # Pre-compute format string per tactic so the closure doesn't need to
    # re-evaluate on every call to _autotune_fn.
    _tactic_formats = [
        _format_token_for_tactic(entry, descriptor.specs)
        for entry in tactic_table
    ]

    tensor_desc_cls = trtp.TensorDesc

    def _autotune_fn(*args: Any) -> List[Any]:
        inp_descs = list(args[:num_inputs])
        out_descs_raw = args[num_inputs]
        out_descs = list(out_descs_raw) if hasattr(out_descs_raw, "__iter__") else [out_descs_raw]
        parts = [
            dtype_token(td)
            for td in inp_descs + out_descs
        ]
        dtype_str = ", ".join(parts)
        return [
            trtp.AutoTuneCombination(dtype_str, fmt, [tid])
            for tid, fmt in zip(tactic_ids, _tactic_formats)
        ]

    sig_params = (
        _build_input_params(num_inputs, tensor_desc_cls, names=tensor_arg_names)
        + [inspect.Parameter("outputs", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    _autotune_fn.__signature__ = inspect.Signature(sig_params)
    return _autotune_fn


# ---------------------------------------------------------------------------
# AOT impl builder (dispatches to backend)
# ---------------------------------------------------------------------------

# LIMITATION (TensorRT 10.14 bug — Blackwell only): Mixed Triton+CuTile or
# Triton+CuTeDSL plugins that produce SymIntExprs of different lengths crash
# with CUDA error 700 (illegal memory access) at execute_async_v3 on Blackwell
# (Myelin QUICKAOT path).  TRT picks a single SymIntExprs length for the whole
# plugin and passes that many extra int32s to every tactic's kernel; if the
# selected kernel has fewer .param declarations, the extras corrupt unrelated
# memory.  Workaround: do not mix Triton and CuTeDSL tactics in the same
# CustomPluginSpec on Blackwell.  Fix expected in a future TRT release.
def _build_aot_fn(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    tactic_table: List[TacticEntry],
    tensor_arg_names: Optional[List[str]] = None,
) -> Callable[..., Any]:
    """Build a ``@trtp.aot_impl`` function that dispatches to the right backend.

    TRT calls this once with the winning tactic index (chosen by autotune).
    The first ``num_inputs`` positional args are ``TensorDesc`` inputs; then
    ``outputs`` (sequence of ``TensorDesc``); then ``tactic`` (int-like).

    The built function is cached in the thread-local ``_aot_fn_cache`` keyed by
    ``(op_name, num_inputs)`` to avoid redundant closure construction within a
    single thread.

    .. note:: **TensorRT 10.14 bug on Blackwell (Myelin QUICKAOT)**

        When tactics return ``SymIntExprs`` of different lengths, TRT/Myelin
        crashes with CUDA error 700 (illegal memory access) at
        ``execute_async_v3``.  Mixed Triton+CuTeDSL plugins trigger this because:

        - Triton: returns ``SymIntExprs(N)`` with N scalar kernel args (e.g.
          ``n_elements``)
        - CuTeDSL: returns ``SymIntExprs(0)`` — no scalar args, shape is in CuTe
          descriptors

        On Blackwell, TRT picks one ``SymIntExprs`` length for the entire plugin
        and passes that many extra ``int32`` s to every tactic's kernel at launch.
        If the selected kernel's PTX has fewer ``.param`` declarations than
        expected, the extras land in unrelated memory → crash.  On pre-Blackwell
        (non-Myelin), the mismatch is silently tolerated.

        No workaround is applied here; the annotation layer routes mixed-backend
        tests to pre-Blackwell GPUs (see ``tests/py/annotation/conftest.py``).
        Filed as a TensorRT bug; repro:
        ``tests/py/annotation/repro_blackwell_extra_len_mismatch.py``

    Args:
        descriptor:    The :class:`CustomPluginSpec` being registered.
        num_inputs:    Number of input ``TensorDesc`` positional arguments.
        tactic_table:  Pre-built list of :class:`TacticEntry` objects.

    Returns:
        A callable with the correct ``inspect.Signature`` for ``@trtp.aot_impl``.

    Raises:
        :class:`QDPRuntimeError`: If the tactic index is out of range or the
            kernel spec type is not supported.
    """
    cache = _get_aot_fn_cache()
    # Include tensor_arg_names in the cache key — two registrations of the
    # same op_name with different arg-name conventions (e.g. ``inp0`` vs
    # the torch op's actual schema names) need distinct __signature__s.
    cache_key = (
        descriptor.op_name,
        num_inputs,
        tuple(tensor_arg_names) if tensor_arg_names is not None else None,
    )
    if cache_key in cache:
        return cache[cache_key]

    op_name = descriptor.op_name
    tensor_desc_cls = trtp.TensorDesc

    def _aot_fn(*args: Any) -> Any:
        inp_descs = list(args[:num_inputs])
        if len(args) <= num_inputs:
            raise IndexError(
                f"AOT impl for {op_name!r} called with {len(args)} args but "
                f"expected at least {num_inputs + 1} "
                f"(inputs + outputs). TRT may not have passed the outputs "
                f"argument."
            )
        out_descs = list(args[num_inputs])        # outputs is a sequence
        tactic_id = int(args[num_inputs + 1]) if len(args) > num_inputs + 1 else 1
        tactic_idx = tactic_id - 1

        if tactic_idx < 0 or tactic_idx >= len(tactic_table):
            raise QDPRuntimeError(
                op=op_name,
                stage="aot_impl",
                backend="custom_plugin",
                msg=(
                    f"tactic ID {tactic_id} (index {tactic_idx}) out of range "
                    f"[0, {len(tactic_table)}) for op {op_name!r}"
                ),
            )
        entry = tactic_table[tactic_idx]
        spec = descriptor.specs[entry.spec_idx]
        configs = spec.configs if spec.configs else [{}]
        cfg = configs[entry.config_idx]
        launch_fn = spec.launch_fn

        sym_inputs = [
            SymbolicTensor(td=td, role=TensorRole.INPUT, index=i)
            for i, td in enumerate(inp_descs)
        ]
        sym_outputs = [
            SymbolicTensor(td=td, role=TensorRole.OUTPUT, index=j)
            for j, td in enumerate(out_descs)
        ]
        host_args = sym_inputs + sym_outputs

        plugin_attrs = descriptor.attrs

        if isinstance(spec, TritonSpec):
            from ._aot._triton import aot_impl_triton
            return aot_impl_triton(
                qdp_symbol=op_name, spec=spec, cfg=cfg, launch_fn=launch_fn,
                host_args=host_args, inp_descs=inp_descs, out_descs=out_descs,
                attrs=plugin_attrs,
            )
        elif isinstance(spec, CuTileSpec):
            from ._aot._cutile import aot_impl_cutile
            return aot_impl_cutile(
                qdp_symbol=op_name, spec=spec, cfg=cfg, launch_fn=launch_fn,
                host_args=host_args, inp_descs=inp_descs, out_descs=out_descs,
                attrs=plugin_attrs,
            )
        elif isinstance(spec, CuTeDSLSpec):
            from ._aot._cutedsl import aot_impl_cutedsl
            return aot_impl_cutedsl(
                qdp_symbol=op_name, spec=spec, cfg=cfg, launch_fn=launch_fn,
                host_args=host_args, inp_descs=inp_descs, out_descs=out_descs,
                attrs=plugin_attrs,
            )
        else:
            raise QDPRuntimeError(
                op=op_name, stage="aot_impl", backend="custom_plugin",
                msg=f"unsupported kernel spec type {type(spec).__name__!r} for op {op_name!r}",
            )

    sig_params = (
        _build_input_params(num_inputs, tensor_desc_cls, names=tensor_arg_names)
        + [inspect.Parameter("outputs", inspect.Parameter.POSITIONAL_OR_KEYWORD),
           inspect.Parameter("tactic",  inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    )
    _aot_fn.__signature__ = inspect.Signature(
        sig_params,
        return_annotation=typing.Tuple[
            typing.Union[str, bytes],
            typing.Union[str, bytes],
            trtp.KernelLaunchParams,
            trtp.SymIntExprs,
        ],
    )

    cache[cache_key] = _aot_fn
    return _aot_fn


# ---------------------------------------------------------------------------
# QDP registration
# ---------------------------------------------------------------------------


def register_custom_plugin(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    num_outputs: int = 1,
    qdp_name: Optional[str] = None,
) -> None:
    """Register a :class:`CustomPluginSpec` with TRT's QDP plugin registry.

    Registers three QDP callbacks under ``qdp_name`` (if provided) or
    ``descriptor.op_name``:

    * ``@trtp.register``  — shape/dtype descriptor (uses ``meta_impl`` or identity)
    * ``@trtp.autotune``  — format/tactic combinations per I/O position
    * ``@trtp.aot_impl``  — AOT kernel dispatch (Triton/CuTile/CuTeDSL backend)

    **Idempotency**: repeated calls for the same ``op_name`` are no-ops.  The
    function uses a double-checked locking pattern against
    ``_qdp_registered_ops`` (a process-global set) to guard against concurrent
    registration from multiple threads.  See the module-level comment on
    ``_qdp_registered_ops`` for the full threading contract.

    If TRT's own registry reports that an op is "already registered" (e.g.
    because a prior in-process call registered it before ``_qdp_registered_ops``
    was updated), the function catches the error, logs a debug message, and
    marks the op as registered so future calls are no-ops.

    Args:
        descriptor:  The :class:`CustomPluginSpec` to register.
        num_inputs:  Number of TRT input tensors for this op.
        num_outputs: Number of TRT output tensors (default 1).
        qdp_name:    Optional override for the QDP registration name.  When
                     provided the plugin is registered under this name instead
                     of ``descriptor.op_name``.  Use this when wiring a
                     :class:`CustomPluginSpec` to a ``torch.library`` op whose
                     name differs from the TTA fingerprint name (e.g. when
                     called from ``trt_plugins.custom_op``).

    Raises:
        Exception: Any exception raised by TRT's ``trtp.register`` that is
            **not** an "already registered" error is re-raised verbatim.
        Exception: Any exception raised by TRT's ``trtp.aot_impl`` that is
            **not** an "already registered" error is re-raised verbatim.
    """
    op_name = qdp_name if qdp_name is not None else descriptor.op_name

    # Fast path: check without the lock first (common case after first registration).
    # Safe because _qdp_registered_ops only grows — see module-level threading contract.
    if op_name in _qdp_registered_ops:
        return

    with _qdp_registration_lock:
        # Re-check inside the lock to handle the race between the fast-path
        # check above and acquiring the lock (TOCTOU).
        if op_name in _qdp_registered_ops:
            return

        from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
            register_plugin,
        )

        tactic_table = build_tactic_table(descriptor.specs)

        # Build the three QDP callbacks, then delegate the raw trtp calls to
        # register_plugin so that all plugin registration (JIT and AOT)
        # converges on the same path in the existing plugin system.
        desc_fn = _build_desc_fn(descriptor, num_inputs, num_outputs)
        autotune_fn = _build_autotune_fn(descriptor, num_inputs, num_outputs, tactic_table)
        aot_fn = _build_aot_fn(descriptor, num_inputs, tactic_table)

        # TRT's internal QDP registry is process-global; if TRT already knows the
        # op (from a prior call in this process), catch "already has a definition"
        # and mark it registered without re-registering.
        try:
            register_plugin(op_name, desc_fn, autotune_fn, aot_fn)
        except Exception as exc:  # noqa: BLE001
            if "already" in str(exc).lower():
                logger.debug(
                    "QDP op %r already registered in TRT registry, skipping: %s",
                    op_name, exc,
                )
                _qdp_registered_ops.add(op_name)
                return
            raise

        _qdp_registered_ops.add(op_name)
        logger.debug(
            "Registered QDP plugin %r (num_inputs=%d, num_outputs=%d, tactics=%d)",
            op_name,
            num_inputs,
            num_outputs,
            len(tactic_table),
        )


def register_autotune_and_aot(
    descriptor: CustomPluginSpec,
    num_inputs: int,
    num_outputs: int,
    qdp_name: str,
    tensor_arg_names: Optional[List[str]] = None,
) -> None:
    """Layer ``@trtp.autotune`` + ``@trtp.aot_impl`` on top of a desc already
    registered by ``generate_plugin(qdp_name)``.

    Used by the public ``custom_op(..., _aot_register=...)`` hook (see
    :mod:`torch_tensorrt.dynamo.conversion.plugins._custom_op`) for the
    no-weights annotation path. ``generate_plugin`` provides the
    schema-derived ``@trtp.register`` descriptor and a JIT ``@trtp.impl``;
    the autotune + AOT callbacks built here select tactics and dispatch
    PTX/cubin to TRT at build time so the runtime never needs Python.

    Idempotent against :data:`_qdp_registered_ops` (same lock + set as
    :func:`register_custom_plugin`).

    Args:
        descriptor:  The :class:`CustomPluginSpec` carrying backend specs.
        num_inputs:  Number of activation TRT inputs (weights are not
            supported on this path — use :func:`register_custom_plugin` for
            weighted plugins).
        num_outputs: Number of TRT outputs.
        qdp_name:    QDP op name in ``"namespace::name"`` form. Must match
            the op already registered by ``generate_plugin``.
    """
    if qdp_name in _qdp_registered_ops:
        return

    with _qdp_registration_lock:
        if qdp_name in _qdp_registered_ops:
            return

        import tensorrt.plugin as trtp

        tactic_table = build_tactic_table(descriptor.specs)
        autotune_fn = _build_autotune_fn(
            descriptor, num_inputs, num_outputs, tactic_table,
            tensor_arg_names=tensor_arg_names,
        )
        aot_fn = _build_aot_fn(
            descriptor, num_inputs, tactic_table,
            tensor_arg_names=tensor_arg_names,
        )

        try:
            trtp.autotune(qdp_name)(autotune_fn)
            trtp.aot_impl(qdp_name)(aot_fn)
        except Exception as exc:  # noqa: BLE001
            if "already" in str(exc).lower():
                logger.debug(
                    "QDP op %r autotune/aot already registered, skipping: %s",
                    qdp_name, exc,
                )
                _qdp_registered_ops.add(qdp_name)
                return
            raise

        _qdp_registered_ops.add(qdp_name)
        logger.debug(
            "Registered autotune + aot_impl for %r (num_inputs=%d, num_outputs=%d, tactics=%d)",
            qdp_name,
            num_inputs,
            num_outputs,
            len(tactic_table),
        )


# ---------------------------------------------------------------------------
# TRT network lowering
# ---------------------------------------------------------------------------


def lower_custom_plugin_descriptor(
    ctx: Any,
    descriptor: CustomPluginSpec,
    trt_inputs: List[Any],
    name: str,
) -> Union[Any, Sequence[Any]]:
    """Lower a :class:`CustomPluginSpec` to a TRT ``IPluginV3`` layer.

    This is the final step in the :class:`CustomPluginSpec` lifecycle (see
    class docstring).  It:

    1. Injects weight tensors as ``trt.add_constant`` layers, appending them to
       ``trt_inputs`` so the plugin sees ``(*activations, *weights)`` as inputs.
    2. Calls :func:`register_custom_plugin` (idempotent) to ensure the QDP
       callbacks are registered before the network refers to the op.
    3. Resolves ``trtp.op.<namespace>.<plugin_name>`` and calls it with
       ``trt_inputs`` to obtain the plugin handle.
    4. Adds the plugin layer via ``ctx.net.add_plugin(..., aot=True)`` and
       attaches TTA layer metadata for debugging.
    5. Returns a single ``trt.ITensor`` (for single-output plugins) or a tuple
       of ``trt.ITensor`` s.

    Args:
        ctx:        Torch-TRT ``ConversionContext`` providing ``ctx.net``.
        descriptor: :class:`CustomPluginSpec` for this op.
        trt_inputs: List of ``trt.ITensor`` dynamic activation inputs.
        name:       Layer name used for TRT network debugging.

    Returns:
        A single ``trt.ITensor`` if the plugin has one output, or a ``tuple``
        of ``trt.ITensor`` s for multi-output plugins.

    Raises:
        :class:`QDPRuntimeError`: Wraps any non-QDP exception raised during
            lowering (registration, plugin construction, or layer addition).
    """
    op_name = descriptor.op_name

    try:
        # Re-infer num_outputs from real trt_inputs ranks (most accurate path).
        # trt_inputs at this point are the activation inputs only (weights are
        # appended below), so their ranks exactly match what meta_impl expects.
        from torch_tensorrt.dynamo.conversion.plugins._generate_plugin import (
            _probe_num_outputs_from_callable,
        )
        _hint = len(trt_inputs[0].shape) if trt_inputs else None
        num_outputs = _probe_num_outputs_from_callable(
            descriptor.meta_impl, len(trt_inputs), preferred_rank=_hint
        )

        # Weight binding: tensor-valued kwargs declared in custom_plugin() are injected
        # here as TRT constant layers, appended after the dynamic activation inputs.
        # The annotated function only receives the activations (no weight args), so the
        # eager body is unchanged.  The launch_fn contract is:
        #   (*activations, *weights_in_declaration_order, *outputs, ...)
        if descriptor.weights:
            from torch_tensorrt.dynamo.conversion.converter_utils import (
                create_constant,
            )

            weight_trt_tensors = [
                create_constant(
                    ctx,
                    wtensor.contiguous(),
                    name=f"{name}_weight_{wname}",
                    dtype=wtensor.dtype,
                )
                for wname, wtensor in descriptor.weights.items()
            ]
            trt_inputs = list(trt_inputs) + weight_trt_tensors

        num_inputs = len(trt_inputs)
        register_custom_plugin(descriptor, num_inputs, num_outputs)

        # Parse namespace and plugin name from op_name (format: "ns::op").
        if "::" in op_name:
            namespace, plugin_name = op_name.split("::", 1)
        else:
            namespace = "tta_custom"
            plugin_name = op_name

        ns_module = getattr(trtp.op, namespace)
        plugin_fn = getattr(ns_module, plugin_name)

        # Attrs are baked into the kernel PTX at AOT time; do not pass as TRT
        # plugin fields (TRT stores them as numpy arrays and the round-trip
        # float(np.array([v])) fails in NumPy 1.25+).
        plugin_layer = ctx.net.add_plugin(plugin_fn(*trt_inputs), aot=True)
        plugin_layer.name = name
        if plugin_layer.num_outputs == 1:
            return plugin_layer.get_output(0)
        return tuple(
            plugin_layer.get_output(i) for i in range(plugin_layer.num_outputs)
        )

    except Exception as exc:
        if isinstance(exc, QDPRuntimeError):
            raise
        raise QDPRuntimeError(
            op=op_name,
            stage="compile",
            backend="custom_plugin",
            msg=f"lowering failed for op {op_name!r} (layer {name!r}): {exc}",
        ) from exc
