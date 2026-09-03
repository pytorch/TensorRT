from __future__ import annotations

import inspect
import keyword
import logging
import re
import threading
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
    cast,
    get_type_hints,
)

import torch

from torch_tensorrt.dynamo.conversion._ConverterRegistry import ConverterPriority
from torch_tensorrt.dynamo.conversion.plugins import custom_op

_LOGGER = logging.getLogger(__name__)

_PYTHON_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PTX_ENTRY_IDENTIFIER = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$.]*$")
_GENERATED_PARAMETER_NAMES = frozenset({"outputs", "stream", "tactic"})
_MISSING_REGISTRATION = object()
# Registration spans several process-global Torch, TensorRT QDP, native plugin,
# and converter registries. Keep the availability check, mutations, and any
# rollback in one critical section so a losing same-name registration cannot
# mistake another thread's successful state for its own and remove it.
_REGISTRATION_LOCK = threading.RLock()


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

_TENSOR_SCHEMA_TYPE = torch._C.TensorType.get()
_SUPPORTED_ATTRIBUTE_SCHEMA_TYPES = (
    torch._C.FloatType.get(),
    torch._C.IntType.get(),
    torch._C.BoolType.get(),
    torch._C.StringType.get(),
)


class OpSchemaInfo(NamedTuple):
    """One parsed, validated view of the Torch/QDP operator schema."""

    schema: str
    tensor_arg_names: Tuple[str, ...]
    attr_arg_names: Tuple[str, ...]
    num_outputs: int


def _validate_op_name(op_name: str) -> Tuple[str, str]:
    """Validate and split a QDP/Torch qualified operator name."""
    if not isinstance(op_name, str) or op_name.count("::") != 1:
        raise ValueError(
            f"op_name must have the form 'namespace::name'; got {op_name!r}."
        )
    namespace, name = op_name.split("::")
    if (
        not _PYTHON_IDENTIFIER.fullmatch(namespace)
        or not _PYTHON_IDENTIFIER.fullmatch(name)
        or keyword.iskeyword(namespace)
        or keyword.iskeyword(name)
    ):
        raise ValueError(
            "QDP operator namespaces and names must be valid Python-style "
            f"identifiers; got {op_name!r}."
        )
    return namespace, name


def _infer_schema(
    fn: Callable[..., Any], *, require_complete_hints: bool = False
) -> str:
    """Derive a TorchScript schema like '(Tensor x, int n) -> Tensor' from type hints."""
    if not callable(fn):
        raise ValueError(f"meta_fn must be callable; got {type(fn).__name__}.")
    try:
        hints = get_type_hints(fn)
    except Exception as exc:
        raise ValueError(
            "could not resolve meta_fn type hints; pass an explicit schema if "
            "the annotations cannot be resolved."
        ) from exc

    params = list(inspect.signature(fn).parameters.keys())
    if require_complete_hints:
        missing = [name for name in params if name not in hints]
        if "return" not in hints:
            missing.append("return")
        if missing:
            raise ValueError(
                "triton_op requires complete meta_fn type hints to determine its "
                f"pointer ABI; missing annotations for {missing}. Pass schema= "
                "explicitly if the function cannot be annotated."
            )
    unsupported = [
        name
        for name in params
        if name in hints and hints[name] not in _TORCH_TYPE_TO_SCHEMA
    ]
    if unsupported:
        raise ValueError(
            "meta_fn has unsupported input annotation(s) for "
            f"{unsupported}; pass an explicit schema."
        )

    args_str = ", ".join(
        "{} {}".format(
            _TORCH_TYPE_TO_SCHEMA.get(hints.get(param, torch.Tensor), "Tensor"),
            param,
        )
        for param in params
    )

    ret = hints.get("return", torch.Tensor)
    if getattr(ret, "__origin__", None) is tuple:
        if not ret.__args__ or any(t is not torch.Tensor for t in ret.__args__):
            raise ValueError("meta_fn outputs must all be annotated as torch.Tensor.")
        ret_str = "({})".format(
            ", ".join(_TORCH_TYPE_TO_SCHEMA.get(t, "Tensor") for t in ret.__args__)
        )
    else:
        if "return" in hints and ret is not torch.Tensor:
            raise ValueError("meta_fn outputs must all be annotated as torch.Tensor.")
        ret_str = _TORCH_TYPE_TO_SCHEMA.get(ret, "Tensor")

    return f"({args_str}) -> {ret_str}"


def analyze_op_schema(
    meta_fn: Callable[..., Any],
    schema: Optional[str] = None,
    *,
    require_complete_hints: bool = False,
    tensor_inputs_only: bool = False,
) -> OpSchemaInfo:
    """Resolve and validate the one schema used by every registration stage."""
    if not callable(meta_fn):
        raise ValueError(f"meta_fn must be callable; got {type(meta_fn).__name__}.")

    schema_str = (
        schema
        if schema is not None
        else _infer_schema(meta_fn, require_complete_hints=require_complete_hints)
    )
    if not isinstance(schema_str, str):
        raise ValueError(f"schema must be a string; got {type(schema_str).__name__}.")
    try:
        parsed = torch._C.parse_schema(f"_ttk::_probe{schema_str}")
    except Exception as exc:
        raise ValueError(
            f"could not parse the explicit schema {schema_str!r}."
        ) from exc

    argument_names = [arg.name for arg in parsed.arguments]
    invalid_names = [
        name
        for name in argument_names
        if not name.isidentifier()
        or keyword.iskeyword(name)
        or name in _GENERATED_PARAMETER_NAMES
    ]
    duplicate_names = sorted(
        {name for name in argument_names if argument_names.count(name) > 1}
    )
    if invalid_names or duplicate_names:
        details = []
        if invalid_names:
            details.append(
                "names incompatible with generated QDP callbacks: "
                f"{sorted(set(invalid_names))}"
            )
        if duplicate_names:
            details.append(f"duplicate names: {duplicate_names}")
        raise ValueError(
            "QDP schema has invalid argument names (" + "; ".join(details) + ")."
        )

    keyword_only_args = [arg.name for arg in parsed.arguments if arg.kwarg_only]
    if keyword_only_args:
        raise ValueError(
            "QDP schemas do not support keyword-only inputs; declare "
            f"{keyword_only_args} as positional arguments."
        )

    try:
        inspect.signature(meta_fn).bind(*([None] * len(parsed.arguments)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "meta_fn must accept the schema's arguments positionally; schema "
            f"declares {len(parsed.arguments)} argument(s)."
        ) from exc

    tensor_arg_names: List[str] = []
    attr_arg_names: List[str] = []
    seen_attr = False
    unsupported_args: List[str] = []
    for arg in parsed.arguments:
        if arg.type.isSubtypeOf(_TENSOR_SCHEMA_TYPE):
            if seen_attr:
                raise ValueError(
                    "QDP schemas must declare every Tensor input before scalar "
                    f"attributes; Tensor argument {arg.name!r} follows an attribute."
                )
            tensor_arg_names.append(arg.name)
            continue

        seen_attr = True
        if not any(
            arg.type.isSubtypeOf(supported)
            for supported in _SUPPORTED_ATTRIBUTE_SCHEMA_TYPES
        ):
            unsupported_args.append(f"{arg.name}: {arg.type}")
        attr_arg_names.append(arg.name)

    if unsupported_args:
        raise ValueError(
            "QDP schema contains unsupported input type(s): "
            + ", ".join(unsupported_args)
            + "."
        )
    if not tensor_arg_names:
        raise ValueError("QDP operators must have at least one Tensor input.")
    if tensor_inputs_only and attr_arg_names:
        raise ValueError(
            "triton_op currently supports Tensor-only Torch schemas; scalar Torch "
            f"attributes {list(attr_arg_names)} cannot be forwarded to its AOT "
            "grid or kernel arguments. Derive i32 kernel extras from TensorDesc "
            "shapes, or use the lower-level QDP JIT API for user attributes."
        )

    non_tensor_returns = [
        str(ret.type)
        for ret in parsed.returns
        if not ret.type.isSubtypeOf(_TENSOR_SCHEMA_TYPE)
    ]
    if non_tensor_returns:
        raise ValueError(
            "QDP outputs must all be Tensor values; schema returns unsupported "
            f"type(s) {non_tensor_returns}."
        )
    if not parsed.returns:
        raise ValueError("QDP operators must return at least one Tensor.")

    return OpSchemaInfo(
        schema_str,
        tuple(tensor_arg_names),
        tuple(attr_arg_names),
        len(parsed.returns),
    )


def _torch_op_already_registered(op_name: str) -> bool:
    """Return True if ``op_name`` is already known to the torch dispatcher."""
    ns, name = _validate_op_name(op_name)
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
    *,
    schema_info: Optional[OpSchemaInfo] = None,
) -> Optional[torch.library.Library]:
    """Register a new PyTorch custom op using torch.library.Library.

    An exact existing schema is treated as an idempotent direct call. A
    conflicting schema is rejected rather than silently retaining a different
    dispatcher ABI.

    Atomic with respect to failure: registration touches three slots
    (``lib.define`` → optional ``lib.impl("CUDA")`` → ``register_fake``). If
    any later step raises, the partial state is torn down via ``lib._destroy``
    so a retry sees the op as un-registered. Without this, a failure in step 2
    or 3 would leave ``torch.ops.<ns>.<name>`` populated by step 1, causing
    ``_torch_op_already_registered`` to short-circuit subsequent attempts and
    permanently hide the missing CUDA / fake impl.
    """
    ns, name = _validate_op_name(op_name)
    info = schema_info or analyze_op_schema(meta_fn, schema)
    expected_schema = str(torch._C.parse_schema(f"{op_name}{info.schema}"))
    if _torch_op_already_registered(op_name):
        op_packet = getattr(getattr(torch.ops, ns), name)
        actual_schema = op_packet._schemas.get("")
        actual_schema_str = str(actual_schema) if actual_schema is not None else None
        if actual_schema_str != expected_schema:
            raise ValueError(
                f"PyTorch op '{op_name}' is already registered with schema "
                f"{actual_schema_str!r}; requested {expected_schema!r}. Reuse a "
                "different qualified operator name."
            )
        _LOGGER.debug("PyTorch op %s already has the requested schema", op_name)
        return None

    lib = torch.library.Library(ns, "FRAGMENT")
    try:
        lib.define(f"{name}{info.schema}")
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
    _LOGGER.debug("Registered PyTorch op %s with schema %s", op_name, expected_schema)
    return lib


def _make_aot_impl(
    schema_info: OpSchemaInfo,
    ptx: str,
    kernel_name: str,
    user_aot_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Build the exact-signature QDP AOT callback without mutating registries."""
    import tensorrt.plugin as trtp

    # Build the aot_impl function body with the correct positional arg names so
    # trtp.aot_impl can match them to the registered descriptor.
    tensor_arg_names = list(schema_info.tensor_arg_names)
    sig = ", ".join(tensor_arg_names + ["outputs", "tactic"])
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
        "_ptx_str": ptx,
        "_trtp": trtp,
    }
    local_ns: Dict[str, Any] = {}
    exec(compile(fn_body, "<aot_impl>", "exec"), fn_globals, local_ns)
    aot_fn = cast(Callable[..., Any], local_ns["_aot_impl"])

    aot_fn.__annotations__ = dict.fromkeys(tensor_arg_names, trtp.TensorDesc)
    # Exact-arity tuple — trtp validates this via issubclass on each arg, which
    # rejects Ellipsis (so ``Tuple[TensorDesc, ...]`` wouldn't work).
    aot_fn.__annotations__["outputs"] = Tuple[
        (trtp.TensorDesc,) * schema_info.num_outputs
    ]
    aot_fn.__annotations__["tactic"] = int
    aot_fn.__annotations__["return"] = Tuple[
        Union[str, bytes], Union[str, bytes], trtp.KernelLaunchParams, trtp.SymExprs
    ]

    return aot_fn


def _register_aot_impl(op_name: str, aot_fn: Callable[..., Any]) -> None:
    """Register one already-built AOT callback with TensorRT QDP."""
    import tensorrt.plugin as trtp

    trtp.aot_impl(op_name)(aot_fn)
    _LOGGER.debug("Registered AOT impl for %s", op_name)


def _decode_and_validate_ptx(op_name: str, ptx: bytes, kernel_name: str) -> str:
    """Validate a PTX artifact before touching any process-global registry."""
    if not isinstance(ptx, bytes) or not ptx:
        raise ValueError(
            f"precompiled PTX for plugin '{op_name}' must be non-empty bytes."
        )
    try:
        ptx_text = ptx.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"precompiled PTX for plugin '{op_name}' must be valid UTF-8 text."
        ) from exc
    if not isinstance(kernel_name, str) or not _PTX_ENTRY_IDENTIFIER.fullmatch(
        kernel_name
    ):
        raise ValueError(
            f"kernel_name for plugin '{op_name}' must be a valid PTX entry "
            f"identifier; got {kernel_name!r}."
        )

    entry = re.compile(
        rf"^\s*(?:\.visible\s+)?\.entry\s+{re.escape(kernel_name)}(?=\s|\()",
        re.MULTILINE,
    )
    if entry.search(ptx_text) is None:
        raise ValueError(
            f"precompiled PTX for plugin '{op_name}' does not define the requested "
            f"entry symbol {kernel_name!r}."
        )
    return ptx_text


def _assert_registration_name_available(op_name: str) -> None:
    """Reject Torch or QDP collisions before starting registration."""
    if _torch_op_already_registered(op_name):
        raise ValueError(
            f"operator '{op_name}' is already registered with PyTorch; choose a "
            "unique qualified name instead of reusing an unchecked ABI."
        )

    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    namespace, name = _validate_op_name(op_name)
    op_namespace = getattr(trtp.op, namespace, None)
    native_creator = trt.get_plugin_registry().get_creator(name, "1", namespace)
    if (
        op_name in QDP_REGISTRY
        or op_name in QDP_CREATORS
        or (op_namespace is not None and hasattr(op_namespace, name))
        or native_creator is not None
    ):
        raise ValueError(
            f"plugin '{op_name}' is already registered with TensorRT QDP; choose "
            "a unique qualified name."
        )


class _QDPRegistrationState(NamedTuple):
    """Process-global state that must survive a failed registration unchanged."""

    qdp_definition: Any
    qdp_creator: Any
    native_creator: Any
    op_namespace: Any
    op_definition: Any
    converter_target: Any
    converter_entry: Any


def _snapshot_qdp_registration(op_name: str) -> _QDPRegistrationState:
    """Capture the exact slots ``custom_op`` may mutate for ``op_name``."""
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        DYNAMO_ATEN_CONVERTERS,
    )

    namespace, name = _validate_op_name(op_name)
    op_namespace = getattr(trtp.op, namespace, _MISSING_REGISTRATION)
    op_definition = (
        getattr(op_namespace, name, _MISSING_REGISTRATION)
        if op_namespace is not _MISSING_REGISTRATION
        else _MISSING_REGISTRATION
    )
    try:
        converter_target = getattr(getattr(torch.ops, namespace), name).default
    except AttributeError:
        converter_target = _MISSING_REGISTRATION
    converter_entry = (
        DYNAMO_ATEN_CONVERTERS.get(converter_target, _MISSING_REGISTRATION)
        if converter_target is not _MISSING_REGISTRATION
        else _MISSING_REGISTRATION
    )
    # Converter registration appends to an existing list in place. Preserve a
    # shallow copy rather than the mutable list itself so rollback can restore
    # the pre-call contents.
    if isinstance(converter_entry, list):
        converter_entry = list(converter_entry)

    return _QDPRegistrationState(
        QDP_REGISTRY.get(op_name, _MISSING_REGISTRATION),
        QDP_CREATORS.get(op_name, _MISSING_REGISTRATION),
        trt.get_plugin_registry().get_creator(name, "1", namespace),
        op_namespace,
        op_definition,
        converter_target,
        converter_entry,
    )


def _rollback_qdp_registration(op_name: str, state: _QDPRegistrationState) -> None:
    """Best-effort rollback of QDP and converter state created after ``state``.

    TensorRT exposes no transaction around ``register`` / ``impl`` /
    ``aot_impl``. Restore each Python registry slot and use its public native
    creator deregistration API. Every removal is conditioned on the slot being
    absent in the snapshot, so a failed attempt never deletes pre-existing
    state.
    """
    import tensorrt as trt
    import tensorrt.plugin as trtp
    from tensorrt.plugin._lib import QDP_CREATORS, QDP_REGISTRY

    from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
        DYNAMO_ATEN_CONVERTERS,
    )

    namespace, name = _validate_op_name(op_name)

    try:
        if state.converter_target is not _MISSING_REGISTRATION:
            if state.converter_entry is _MISSING_REGISTRATION:
                DYNAMO_ATEN_CONVERTERS.pop(state.converter_target, None)
            else:
                DYNAMO_ATEN_CONVERTERS[state.converter_target] = state.converter_entry
    except Exception:
        _LOGGER.warning("Could not roll back the converter for %s", op_name)

    current_definition = QDP_REGISTRY.get(op_name, _MISSING_REGISTRATION)
    try:
        current_namespace = getattr(trtp.op, namespace, _MISSING_REGISTRATION)
        if state.op_definition is _MISSING_REGISTRATION:
            if current_namespace is not _MISSING_REGISTRATION and hasattr(
                current_namespace, name
            ):
                current_op_definition = getattr(current_namespace, name)
                # A namespace can be shared by many plugins. Only remove the
                # attribute installed by this attempt, identified by the same
                # PluginDef object stored in QDP_REGISTRY.
                if (
                    current_definition is _MISSING_REGISTRATION
                    or current_op_definition is current_definition
                ):
                    delattr(current_namespace, name)
        elif current_namespace is _MISSING_REGISTRATION:
            setattr(trtp.op, namespace, state.op_namespace)
            setattr(state.op_namespace, name, state.op_definition)
            current_namespace = state.op_namespace
        else:
            setattr(current_namespace, name, state.op_definition)

        if (
            state.op_namespace is _MISSING_REGISTRATION
            and current_namespace is not _MISSING_REGISTRATION
            and not any(
                key != "_namespace" and not key.startswith("__")
                for key in vars(current_namespace)
            )
        ):
            delattr(trtp.op, namespace)
    except Exception:
        _LOGGER.warning("Could not roll back trtp.op for %s", op_name)

    try:
        if state.qdp_definition is _MISSING_REGISTRATION:
            QDP_REGISTRY.pop(op_name, None)
        else:
            QDP_REGISTRY[op_name] = state.qdp_definition
    except Exception:
        _LOGGER.warning("Could not roll back the QDP descriptor for %s", op_name)

    try:
        plugin_registry = trt.get_plugin_registry()
        current_native_creator = plugin_registry.get_creator(name, "1", namespace)
        # An exact-name native creator cannot coexist with another creator. If
        # there was one before the attempt, it is necessarily pre-existing and
        # must be retained even if QDP_CREATORS temporarily pointed at it.
        if state.native_creator is None and current_native_creator is not None:
            if not plugin_registry.deregister_creator(current_native_creator):
                _LOGGER.warning(
                    "TensorRT refused to deregister the native QDP creator for %s; "
                    "future registration attempts will remain fail-closed",
                    op_name,
                )
    except Exception:
        _LOGGER.warning("Could not deregister the native QDP creator for %s", op_name)

    try:
        if state.qdp_creator is _MISSING_REGISTRATION:
            QDP_CREATORS.pop(op_name, None)
        else:
            QDP_CREATORS[op_name] = state.qdp_creator
    except Exception:
        _LOGGER.warning("Could not restore the QDP creator map for %s", op_name)


def _destroy_torch_registration(lib: Optional[torch.library.Library]) -> None:
    """Best-effort cleanup if a later QDP registration stage fails."""
    if lib is None:
        return
    try:
        _LIVE_LIBS.remove(lib)
    except ValueError:
        pass
    try:
        lib._destroy()
    except Exception:
        pass


def register_precompiled_qdp_plugin(
    op_name: str,
    ptx: bytes,
    kernel_name: str,
    aot_fn: Callable[..., Any],
    eager_fn: Optional[Callable[..., Any]],
    meta_fn: Callable[..., Any],
    *,
    supports_dynamic_shapes: bool = False,
    requires_output_allocator: bool = False,
    priority: ConverterPriority = ConverterPriority.STANDARD,
    capability_validator: Optional[Callable[..., Any]] = None,
    schema: Optional[str] = None,
    use_aot_if_available: bool = True,
) -> None:
    """Register an already-compiled PTX kernel as a TensorRT AOT QDP plugin.

    This is the single backend-independent funnel shared by ``cuda_kernel_op``,
    ``ptx_op``, and ``triton_op`` after their respective compilation steps.
    It owns PyTorch custom-op, QDP AOT, and converter registration; it does not
    retain compiler-specific source, signatures, or constexpr metadata.
    """
    _validate_op_name(op_name)
    if not callable(aot_fn):
        raise ValueError(
            f"aot_fn must be callable before registering plugin '{op_name}'."
        )
    if eager_fn is not None and not callable(eager_fn):
        raise ValueError(f"eager_fn for plugin '{op_name}' must be callable or None.")
    if capability_validator is not None and not callable(capability_validator):
        raise ValueError(
            f"capability_validator for plugin '{op_name}' must be callable or None."
        )

    schema_info = analyze_op_schema(meta_fn, schema)
    if use_aot_if_available and schema_info.attr_arg_names:
        raise ValueError(
            f"AOT plugin '{op_name}' cannot safely forward Torch scalar attributes "
            f"{list(schema_info.attr_arg_names)} to its launch callback. Use the "
            "QDP JIT path or a Tensor-only schema."
        )
    ptx_text = _decode_and_validate_ptx(op_name, ptx, kernel_name)
    prepared_aot = (
        _make_aot_impl(schema_info, ptx_text, kernel_name, aot_fn)
        if use_aot_if_available
        else None
    )

    with _REGISTRATION_LOCK:
        _assert_registration_name_available(op_name)
        lib = _register_pytorch_op(
            op_name,
            meta_fn,
            eager_fn,
            schema_info=schema_info,
        )

        # Delegate the TRT-side wiring (plugin desc + converter) to ``custom_op``
        # so there's exactly one place that owns it. ``_aot_register`` slots our
        # precompiled-kernel AOT impl in between, preserving the original ordering
        # of generate_plugin -> _register_aot_impl -> generate_plugin_converter.
        registration_state: Optional[_QDPRegistrationState] = None
        try:
            registration_state = _snapshot_qdp_registration(op_name)
            custom_op(
                op_name,
                capability_validator=capability_validator,
                priority=priority,
                supports_dynamic_shapes=supports_dynamic_shapes,
                requires_output_allocator=requires_output_allocator,
                use_aot_if_available=use_aot_if_available,
                _aot_register=(
                    (lambda: _register_aot_impl(op_name, prepared_aot))
                    if prepared_aot is not None
                    else None
                ),
            )
        except Exception:
            try:
                if registration_state is not None:
                    _rollback_qdp_registration(op_name, registration_state)
            except Exception:
                # Rollback is intentionally best-effort because TensorRT provides
                # no registration transaction. Never let cleanup hide the original
                # registration error or prevent teardown of the Torch dispatcher.
                _LOGGER.warning("Could not fully roll back QDP plugin %s", op_name)
            finally:
                _destroy_torch_registration(lib)
            raise

    _LOGGER.info("QDP plugin '%s' registered successfully", op_name)
