from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, get_type_hints

import torch
from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.plugins import custom_op
from torch_tensorrt.kernels._cuda_python_spec import CudaPythonSpec

_LOGGER = logging.getLogger(__name__)


def _patch_trt_shape_expr_reflected_ops() -> None:
    # TODO(upstream-trt): trtp.ShapeExpr defines forward __mul__ / __add__
    # but not the reflected __rmul__ / __radd__. torch_tensorrt lowers meta-fn
    # shape expressions via sympy.lambdify(..., "math"), which emits canonical
    # forms like ``lambda N: 2*N`` — at runtime Python does ``int * ShapeExpr``,
    # falls back to ``ShapeExpr.__rmul__``, and crashes with TypeError.
    # Reflected forms are commutative so aliasing fwd -> rev is safe.
    #
    # File against NVIDIA/TensorRT (the trtp Python plugin module). Once
    # trtp.ShapeExpr ships __rmul__ / __radd__ natively, this whole function
    # becomes a no-op (the ``not hasattr(cls, rev)`` guard self-disables) and
    # can be deleted along with the unconditional call below.
    # Tracking issue: <ADD TENSORRT ISSUE URL>.
    try:
        import tensorrt.plugin as trtp
    except ImportError:
        return
    cls = getattr(trtp, "ShapeExpr", None)
    if cls is None:
        return
    for fwd, rev in (("__mul__", "__rmul__"), ("__add__", "__radd__")):
        if hasattr(cls, fwd) and not hasattr(cls, rev):
            try:
                setattr(cls, rev, getattr(cls, fwd))
            except (AttributeError, TypeError):
                pass


_patch_trt_shape_expr_reflected_ops()

# Keep Library instances alive – torch frees op registrations when a Library is GC'd.
_LIVE_LIBS: List[torch.library.Library] = []

_TORCH_TYPE_TO_SCHEMA = {
    torch.Tensor: "Tensor",
    int: "int",
    float: "float",
    bool: "bool",
    str: "str",
}


def _infer_schema(fn: Callable[..., Any]) -> str:
    """Derive a TorchScript schema like '(Tensor x, int n) -> Tensor' from type hints."""
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}

    params = list(inspect.signature(fn).parameters.keys())
    args_str = ", ".join(
        "{} {}".format(
            _TORCH_TYPE_TO_SCHEMA.get(hints.get(p, torch.Tensor), "Tensor"), p
        )
        for p in params
    )

    ret = hints.get("return", torch.Tensor)
    origin = getattr(ret, "__origin__", None)
    if origin is tuple:
        ret_str = "({})".format(
            ", ".join(_TORCH_TYPE_TO_SCHEMA.get(t, "Tensor") for t in ret.__args__)
        )
    else:
        ret_str = _TORCH_TYPE_TO_SCHEMA.get(ret, "Tensor")

    return f"({args_str}) -> {ret_str}"


def _torch_op_already_registered(op_name: str) -> bool:
    """Return True if ``op_name`` is already known to the torch dispatcher."""
    ns, name = op_name.split("::", 1)
    try:
        op_packet = getattr(getattr(torch.ops, ns), name)
    except AttributeError:
        return False
    return bool(getattr(op_packet, "_schemas", {}))


def _register_pytorch_op(
    op_name: str,
    meta_fn: Callable[..., Any],
    eager_fn: Optional[Callable[..., Any]],
    schema: Optional[str] = None,
) -> None:
    """Register a new PyTorch custom op using torch.library.Library.

    Idempotent: if ``op_name`` is already registered (e.g. a prior call in the
    same process), this is a no-op rather than re-defining and raising from
    the dispatcher.

    Atomic with respect to failure: registration touches three slots
    (``lib.define`` → optional ``lib.impl("CUDA")`` → ``register_fake``). If
    any later step raises, the partial state is torn down via ``lib._destroy``
    so a retry sees the op as un-registered. Without this, a failure in step 2
    or 3 would leave ``torch.ops.<ns>.<name>`` populated by step 1, causing
    ``_torch_op_already_registered`` to short-circuit subsequent attempts and
    permanently hide the missing CUDA / fake impl.
    """
    if _torch_op_already_registered(op_name):
        _LOGGER.debug("PyTorch op %s already registered; skipping re-register", op_name)
        return

    ns, name = op_name.split("::")
    schema_str = schema if schema is not None else _infer_schema(meta_fn)

    lib = torch.library.Library(ns, "FRAGMENT")
    try:
        lib.define(f"{name}{schema_str}")
        if eager_fn is not None:
            lib.impl(name, eager_fn, "CUDA")
        torch.library.register_fake(op_name)(meta_fn)
    except Exception:
        # Tear down whatever made it onto the dispatcher before re-raising.
        # ``lib._destroy`` resets the underlying C++ Library, deregisters the
        # impls / fake impls we appended, and removes ``torch.ops.<ns>.<name>``
        # from the cached namespace so the next attempt restarts clean.
        try:
            lib._destroy()
        except Exception:
            pass
        raise

    # Only keep the Library alive after every step succeeded. Appending earlier
    # would also retain a partial registration on the failure path.
    _LIVE_LIBS.append(lib)
    _LOGGER.debug("Registered PyTorch op %s  schema: %s%s", op_name, name, schema_str)


def _register_aot_impl(op_name: str, ptx: bytes, spec: CudaPythonSpec) -> None:
    """Dynamically build a correctly-typed aot_impl and register it with trtp."""
    from typing import Tuple, Union  # noqa: F401 – used in annotations dict

    import numpy as np
    import tensorrt.plugin as trtp

    ns, name = op_name.split("::")
    torch_op = getattr(getattr(torch.ops, ns), name)
    schema = torch_op._schemas[""]

    tensor_arg_names = []
    attr_arg_names = []
    attr_annotations = {}
    for arg in schema.arguments:
        if arg.type.isSubtypeOf(torch._C.TensorType.get()):
            tensor_arg_names.append(arg.name)
        else:
            attr_arg_names.append(arg.name)
            if arg.type.isSubtypeOf(torch._C.FloatType.get()):
                attr_annotations[arg.name] = np.ndarray[Any, np.dtype[np.float64]]
            elif arg.type.isSubtypeOf(torch._C.IntType.get()):
                attr_annotations[arg.name] = np.ndarray[Any, np.dtype[np.int64]]
            elif arg.type.isSubtypeOf(torch._C.BoolType.get()):
                attr_annotations[arg.name] = np.ndarray[Any, np.dtype[np.bool_]]
            elif arg.type.isSubtypeOf(torch._C.StringType.get()):
                attr_annotations[arg.name] = str
            else:
                raise ValueError(f"arg type is not handled for {arg.name}")

    ptx_str: str = ptx.decode("utf-8") if isinstance(ptx, bytes) else ptx
    kernel_name = spec.kernel_name
    user_aot_fn = spec.aot_fn

    # Build the aot_impl function body with the correct positional arg names so
    # trtp.aot_impl can match them to the registered descriptor.
    sig = ", ".join(tensor_arg_names + attr_arg_names + ["outputs", "tactic"])
    fn_body = f"""\
def _aot_impl({sig}):
    inputs = [{", ".join(tensor_arg_names)}]
    result = _user_aot_fn(inputs, outputs, tactic)
    if isinstance(result, tuple) and len(result) == 2:
        launch_params, extra_args = result
    else:
        launch_params, extra_args = result, None
    if extra_args is None:
        extra_args = _trtp.SymIntExprs(0)
    return (_kernel_name, _ptx_str, launch_params, extra_args)
"""

    fn_globals = {
        "_user_aot_fn": user_aot_fn,
        "_kernel_name": kernel_name,
        "_ptx_str": ptx_str,
        "_trtp": trtp,
    }
    local_ns: Dict[str, Any] = {}
    exec(compile(fn_body, "<aot_impl>", "exec"), fn_globals, local_ns)
    aot_fn = local_ns["_aot_impl"]

    aot_fn.__annotations__ = dict.fromkeys(tensor_arg_names, trtp.TensorDesc)
    aot_fn.__annotations__.update(attr_annotations)
    # Exact-arity tuple — trtp validates this via issubclass on each arg, which
    # rejects Ellipsis (so ``Tuple[TensorDesc, ...]`` wouldn't work).
    num_outputs = len(schema.returns)
    aot_fn.__annotations__["outputs"] = Tuple[(trtp.TensorDesc,) * num_outputs]
    aot_fn.__annotations__["tactic"] = int
    aot_fn.__annotations__["return"] = Tuple[
        Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
    ]

    trtp.aot_impl(op_name)(aot_fn)
    _LOGGER.debug("Registered AOT impl for %s", op_name)


def register_cuda_python_plugin(
    op_name: str,
    spec: CudaPythonSpec,
    meta_fn: Optional[Callable[..., Any]],
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
    register_torch_op: bool = True,
    schema: Optional[str] = None,
    precompiled_ptx: Optional[bytes] = None,
    use_aot_if_available: bool = True,
) -> None:
    """Register a NVRTC-compiled CUDA kernel as a TensorRT QDP plugin end-to-end.

    Steps performed:
    1. Compile kernel source to PTX via NVRTC (skipped if ``precompiled_ptx`` is passed).
    2. Optionally register the PyTorch custom op (define + fake impl).
    3. Register the TRT plugin descriptor + JIT impl via generate_plugin().
    4. Register the AOT impl with the compiled PTX.
    5. Register the Torch-TensorRT converter via generate_plugin_converter().

    ``precompiled_ptx`` lets higher-level entry points (e.g. ``cuda_kernel_op``)
    avoid a redundant second NVRTC pass when they already compiled the source
    to build an eager kernel handle.
    """
    if spec.aot_fn is None:
        raise ValueError(
            f"CudaPythonSpec.aot_fn must be set before registering plugin '{op_name}'. "
            "Pass aot_fn= to cuda_python() or assign spec.aot_fn directly."
        )

    if precompiled_ptx is not None:
        ptx = precompiled_ptx
    else:
        from torch_tensorrt.kernels._nvrtc import compile_to_ptx

        ptx, _device, _kernel = compile_to_ptx(
            spec.kernel_source,
            spec.kernel_name,
            spec.include_paths,
            spec.compile_std,
            spec.arch_override,
        )

    if register_torch_op:
        if meta_fn is None:
            raise ValueError(
                "meta_fn is required when register_torch_op=True. "
                "Provide the fake/meta implementation as the decorated function."
            )
        _register_pytorch_op(op_name, meta_fn, spec.eager_fn, schema=schema)

    # Delegate the TRT-side wiring (plugin desc + converter) to ``custom_op``
    # so there's exactly one place that owns it. ``_aot_register`` slots our
    # cuda-python AOT impl in between, preserving the original ordering of
    # generate_plugin -> _register_aot_impl -> generate_plugin_converter.
    custom_op(
        op_name,
        capability_validator=capability_validator,
        priority=priority,
        supports_dynamic_shapes=supports_dynamic_shapes,
        requires_output_allocator=requires_output_allocator,
        use_aot_if_available=use_aot_if_available,
        _aot_register=lambda: _register_aot_impl(op_name, ptx, spec),
    )

    _LOGGER.info("cuda-python QDP plugin '%s' registered successfully", op_name)
