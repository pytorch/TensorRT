import itertools
import logging
import re
from types import FunctionType
from typing import Any, Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import sympy
import torch
from sympy import lambdify
from torch._dynamo.source import LocalSource
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv

from torch_tensorrt._features import needs_qdp_plugin

_LOGGER: logging.Logger = logging.getLogger(__name__)


_TORCH_SCHEMA_TYPE_TO_PLUGIN_ATTR_TYPE = {
    "float": npt.NDArray[np.float64],
    "int": npt.NDArray[np.int64],
    "bool": npt.NDArray[np.bool_],
}


def _identifier_from_plugin_name(plugin_name: str) -> str:
    """Return a valid Python identifier fragment for generated plugin functions."""
    return re.sub(r"\W|^(?=\d)", "_", plugin_name)


def _scalar_attr_to_python(value: Any) -> Any:
    """Convert QDP scalar-attribute arrays back to Python scalars."""
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(
                "Expected scalar plugin attribute, got ndarray with shape"
                f" {value.shape}"
            )
        return value.reshape(()).item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def mksym(
    shape_env: ShapeEnv, value: int, source: LocalSource, dynamic_dim: DimDynamic
) -> torch.SymInt:
    return shape_env.create_symintnode(
        shape_env.create_symbol(
            value,
            source=source,
            dynamic_dim=dynamic_dim,
        ),
        hint=value,
        source=source,
    )


def _probe_num_outputs_from_callable(
    meta_impl: Callable[[Any], Any],
    n_args: int,
    preferred_rank: Optional[int] = None,
) -> int:
    """Probe a meta callable (not a registered torch_op) to count its outputs.

    Used by :meth:`CustomPluginSpec.auto_register_torch_op` before the
    ``torch.library`` op exists, so we cannot rely on a registered schema.
    Tries ranks 1–4 with size-2 FakeTensors; returns 1 if all probes fail.

    Args:
        meta_impl:      The shape-inference callable (typically
                        ``CustomPluginSpec.meta_impl``).
        n_args:         Number of tensor arguments to pass.
        preferred_rank: If provided, this rank is tried first before the
                        default sweep (1–4).  Pass the actual input rank when
                        it is known (e.g. from live ``trt.ITensor`` inputs in
                        the lowering path) for a more accurate probe.
    """
    ranks = [preferred_rank] if preferred_rank is not None else []
    ranks += [r for r in range(1, 5) if r != preferred_rank]
    for rank in ranks:
        try:
            with FakeTensorMode():
                dummy = torch.randn([2] * rank)
                result = meta_impl(*([dummy] * max(n_args, 1)))
            return len(result) if isinstance(result, (tuple, list)) else 1
        except Exception:
            continue
    return 1


def _probe_num_outputs(torch_op: Callable[[Any], Any], schema: Any) -> int:
    """Probe ``torch_op`` with rank-2 FakeTensors to count its outputs.

    Non-tensor scalar arguments (int, float, bool, str) receive neutral
    defaults (0, 0.0, False, "") which are sufficient for shape inference.
    Tries ranks 1–4 with size-2 extents; returns 1 if all probes fail.
    """
    # torch._C type singletons are not hashable, so use a list of (type, default) pairs.
    _scalar_defaults = [
        (torch._C.IntType.get(),    0),
        (torch._C.FloatType.get(),  0.0),
        (torch._C.BoolType.get(),   False),
        (torch._C.StringType.get(), ""),
    ]

    for rank in range(1, 5):
        try:
            probe_args = []
            with FakeTensorMode():
                for arg in schema.arguments:
                    if arg.type.isSubtypeOf(torch._C.TensorType.get()):
                        probe_args.append(torch.randn([2] * rank))
                    else:
                        for scalar_type, default in _scalar_defaults:
                            if arg.type.isSubtypeOf(scalar_type):
                                probe_args.append(default)
                                break
                result = torch_op(*probe_args)
            if isinstance(result, torch.Tensor):
                return 1
            return len(result)
        except Exception:
            continue
    return 1


def _compute_out_descs_symbolic(
    tensor_args: Any,
    callable_impl: Callable[[Any], Any],
) -> list:
    """Core ShapeEnv + lambdify computation shared by the JIT and TTA AOT paths.

    Builds symbolic FakeTensors (one per TensorDesc in ``tensor_args``), runs
    ``callable_impl`` to obtain output shapes, lambdifies those shapes into
    sympy-backed formulas, and returns a list of ``TensorDesc`` s with the
    formulas assigned to ``shape_expr``.

    Args:
        tensor_args:   Sequence of input ``TensorDesc`` objects (TRT native type).
        callable_impl: Shape-inference callable.  Receives one ``FakeTensor``
                       per element of ``tensor_args``; must return a
                       ``torch.Tensor`` or a sequence of ``torch.Tensor`` s.

    Returns:
        List of output ``TensorDesc`` objects (one per output tensor).
    """
    shape_env = ShapeEnv()
    syms_args = []
    for tensor_arg in tensor_args:
        sample = {f"{i}": 5 for i in range(tensor_arg.ndim)}
        syms_arg = [
            mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
            for k, v in sample.items()
        ]
        syms_args.append(syms_arg)

    with FakeTensorMode(shape_env=shape_env):
        fake_args = [torch.randn(syms_arg) for syms_arg in syms_args]
        raw_output = callable_impl(*fake_args)

    outputs = (
        [raw_output] if isinstance(raw_output, torch.Tensor) else list(raw_output)
    )

    input_node_expr = list(
        itertools.chain.from_iterable(
            [sym.node.expr for sym in syms_arg] for syms_arg in syms_args
        )
    )

    out_descs = []
    for output in outputs:
        shape_calc_fns = [
            lambdify(tuple(input_node_expr), output.shape[i].node.expr, "math")
            for i in range(output.ndim)
        ]
        out_desc = tensor_args[0].like()
        input_shape_expr = list(
            itertools.chain.from_iterable(td.shape_expr for td in tensor_args)
        )
        for i in range(output.ndim):
            if output.shape[i].node.expr is None:
                raise ValueError(f"output.shape[{i}].node.expr cannot be None")
            out_desc.shape_expr[i] = shape_calc_fns[i](*input_shape_expr)  # type: ignore[misc]
        out_descs.append(out_desc)

    return out_descs


def _build_symbolic_desc_fn(
    callable_impl: Callable[[Any], Any],
    num_inputs: int,
    num_outputs: int = 1,
) -> Callable[..., Any]:
    """Build a TRT descriptor callable backed by FakeTensorMode + ShapeEnv + lambdify.

    Used by the TTA AOT path (``_build_desc_fn`` in
    ``annotation/_custom_plugin/_descriptor.py``) where ``callable_impl`` is
    the user-supplied ``meta_impl``.  The JIT path (``_generic_plugin_desc``
    inside ``_generate_plugin``) uses the same ``_compute_out_descs_symbolic``
    core but wraps it in a closure that filters mixed tensor/scalar ``*args``.

    Args:
        callable_impl: Shape-inference callable.  Receives one ``FakeTensor``
                       per input TensorDesc; must return a ``torch.Tensor`` or
                       a sequence of ``torch.Tensor`` s.
        num_inputs:    Number of leading ``TensorDesc`` positional arguments.
        num_outputs:   Expected output count.  Returns a single ``TensorDesc``
                       when 1, a ``tuple`` otherwise.
    """
    _fn = callable_impl
    _n_in = num_inputs
    _n_out = num_outputs

    def _desc(*args: Any) -> Any:
        out_descs = _compute_out_descs_symbolic(args[:_n_in], _fn)
        return out_descs[0] if _n_out == 1 else tuple(out_descs)

    return _desc


def _generate_plugin(plugin_name: str) -> None:
    try:
        import tensorrt.plugin as trtp
    except ImportError as e:
        raise RuntimeError(
            "Unable to import TensorRT plugin. TensorRT version must be 10.7.0 or"
            " higher to support for Triton based TensorRT plugins"
        )

    namespace, name = plugin_name.split("::")

    # retrieve the corresponding torch operation using the passed in string
    torch_op = getattr(getattr(torch.ops, namespace), name)

    schema = torch_op._schemas[""]
    num_outputs = _probe_num_outputs(torch_op, schema)

    # helper function that generates the required signature based on the torch operation
    def generate_signature(
        torch_op: Callable[[Any], Any],
        num_outputs: int,
    ) -> Tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        arg_list = []

        register_func_annotation = {}
        impl_func_annotation = {}

        for arg in schema.arguments:
            arg_list.append(arg.name)

            # TODO: Torch types need to be converted to python primitive types here
            # Some other types are not handled:
            # - torch._C.ListType.ofT(<type>)
            # - torch._C.TupleType.get()
            # - torch._C.DictType.get(<key_type>, <value_type>)
            # - torch._C.OptionalType.ofT(<type>)
            # - torch._C.DeviceObjType.get()
            # - torch._C.FunctionType.get()
            # - torch._C.ClassType

            if arg.type.isSubtypeOf(torch._C.TensorType.get()):
                register_func_annotation[arg.name] = trtp.TensorDesc
                impl_func_annotation[arg.name] = trtp.Tensor
            elif arg.type.isSubtypeOf(torch._C.FloatType.get()):
                register_func_annotation[arg.name] = (
                    _TORCH_SCHEMA_TYPE_TO_PLUGIN_ATTR_TYPE["float"]
                )
                impl_func_annotation[arg.name] = register_func_annotation[arg.name]
            elif arg.type.isSubtypeOf(torch._C.IntType.get()):
                register_func_annotation[arg.name] = (
                    _TORCH_SCHEMA_TYPE_TO_PLUGIN_ATTR_TYPE["int"]
                )
                impl_func_annotation[arg.name] = register_func_annotation[arg.name]
            elif arg.type.isSubtypeOf(torch._C.BoolType.get()):
                register_func_annotation[arg.name] = (
                    _TORCH_SCHEMA_TYPE_TO_PLUGIN_ATTR_TYPE["bool"]
                )
                impl_func_annotation[arg.name] = register_func_annotation[arg.name]
            elif arg.type.isSubtypeOf(torch._C.StringType.get()):
                register_func_annotation[arg.name] = str
                impl_func_annotation[arg.name] = str
            else:
                raise ValueError("arg type is not handled")

        input_signature = ", ".join(arg_list)

        plugin_signature = f"def add_plugin_desc({input_signature}):"

        plugin_impl_arg_list = arg_list
        plugin_impl_arg_list.append("outputs")
        plugin_impl_arg_list.append("stream")
        plugin_impl_input = ", ".join(plugin_impl_arg_list)
        plugin_impl_signature = f"def add_plugin_impl({plugin_impl_input}):"

        # trtp validates these annotations by calling ``typing.get_args`` and
        # then ``issubclass`` on each element. Variadic ``Tuple[X, ...]`` would
        # surface ``Ellipsis`` (not a class) and raise. Build an exact-arity
        # tuple instead, one TensorDesc/Tensor per schema return.
        register_func_annotation["return"] = Tuple[(trtp.TensorDesc,) * num_outputs]
        impl_func_annotation["outputs"] = Tuple[(trtp.Tensor,) * num_outputs]
        impl_func_annotation["stream"] = int

        return (
            input_signature,
            plugin_signature,
            plugin_impl_signature,
            register_func_annotation,
            impl_func_annotation,
        )

    # Use the helper function to get the required signatures
    (
        input_signature,
        plugin_signature,
        plugin_impl_signature,
        register_func_annotation,
        impl_func_annotation,
    ) = generate_signature(torch_op, num_outputs)

    def _generic_plugin_desc(*args: Any, **kwargs: Any) -> Tuple[trtp.TensorDesc, ...]:
        shape_env = ShapeEnv()
        syms_args = []
        tensor_args = [elem for elem in args if isinstance(elem, trtp.TensorDesc)]
        non_tensor_args = [
            _scalar_attr_to_python(elem)
            for elem in args
            if not isinstance(elem, trtp.TensorDesc)
        ]
        torch_kwargs = {k: _scalar_attr_to_python(v) for k, v in kwargs.items()}

        for tensor_arg in tensor_args:
            sample = {f"{i}": 5 for i in range(tensor_arg.ndim)}
            syms_arg = [
                mksym(shape_env, v, LocalSource(k), DimDynamic.DYNAMIC)
                for k, v in sample.items()
            ]
            syms_args.append(syms_arg)

        with FakeTensorMode(shape_env=shape_env) as fake_mode:
            fake_args = []
            for syms_arg in syms_args:
                fake_arg = torch.randn(syms_arg)
                fake_args.append(fake_arg)

            output = torch_op(*fake_args, *non_tensor_args, **torch_kwargs)

        # Normalize to a list of fake outputs. Multi-output torch ops return
        # a tuple; single-output ops return a bare Tensor.
        outputs_list = list(output) if isinstance(output, (tuple, list)) else [output]

        input_node_expr = list(
            itertools.chain.from_iterable(
                [sym.node.expr for sym in syms_arg] for syms_arg in syms_args
            )
        )
        # TODO(upstream-torch): torch's ShapeEnv.create_symbol(source=...)
        # produces sympy.Symbol names derived from the source (e.g. ``L['0']``
        # for ``LocalSource('0')``), which are not valid Python identifiers.
        # ``sympy.lambdify`` pastes arg names verbatim into the generated
        # ``def`` body, so without this substitution we'd emit
        # ``lambda L['0']: ...`` — a SyntaxError. Renaming to ``_a0, _a1, ...``
        # before lambdifying sidesteps the issue.
        clean_args = [sympy.Symbol(f"_a{j}") for j in range(len(input_node_expr))]
        subs_map = dict(zip(input_node_expr, clean_args))

        input_shape_expr = list(
            itertools.chain.from_iterable(arg.shape_expr for arg in tensor_args)
        )

        out_descs = []
        for out_idx, fake_out in enumerate(outputs_list):
            shape_calc_fns: list[Any] = [None] * fake_out.ndim
            for i in range(fake_out.ndim):
                out_dim = fake_out.shape[i]
                if hasattr(out_dim, "node"):
                    if out_dim.node.expr is None:
                        raise ValueError(
                            f"output[{out_idx}].shape[{i}].node.expr cannot be None"
                        )
                    out_expr = out_dim.node.expr.subs(subs_map)
                else:
                    out_expr = sympy.Integer(int(out_dim))
                shape_calc_fns[i] = lambdify(tuple(clean_args), out_expr, "math")

            # ``TensorDesc.like()`` keeps the input rank, which is wrong for
            # shape-changing ops such as reductions. Use the meta output rank.
            out_desc = tensor_args[0].like()
            new_shape_expr = trtp.ShapeExprs(fake_out.ndim)
            for i in range(fake_out.ndim):
                new_shape_expr[i] = shape_calc_fns[i](*input_shape_expr)
            out_desc.shape_expr = new_shape_expr
            out_descs.append(out_desc)

        return tuple(out_descs)

    plugin_name_fragment = _identifier_from_plugin_name(plugin_name)
    desc_func_name = f"add_plugin_desc_{plugin_name_fragment}"
    impl_func_name = f"add_plugin_impl_{plugin_name_fragment}"
    plugin_signature = plugin_signature.replace("add_plugin_desc", desc_func_name, 1)
    plugin_impl_signature = plugin_impl_signature.replace(
        "add_plugin_impl", impl_func_name, 1
    )

    codegen_plugin = f"""
{plugin_signature}
    return _generic_plugin_desc({input_signature})
    """

    plugin_code = compile(codegen_plugin, "<string>", "exec")

    # Keep each generated descriptor bound to its own closure. A shared module
    # global lets later plugin registrations overwrite earlier descriptors.
    plugin_desc_globals = {**globals(), "_generic_plugin_desc": _generic_plugin_desc}

    plugin = FunctionType(
        plugin_code.co_consts[0],
        plugin_desc_globals,
        desc_func_name,
    )
    plugin.__qualname__ = desc_func_name

    # Function annotation is required for dynamic function to work in TensorRT.Plugin
    plugin.__annotations__ = register_func_annotation

    trtp.register(plugin_name)(plugin)

    def _generic_plugin_impl(
        outputs: Tuple[trtp.Tensor], stream: int, *args: Any, **kwargs: Any
    ) -> None:
        tensor_args = [elem for elem in args if isinstance(elem, trtp.Tensor)]
        non_tensor_args = [elem for elem in args if not isinstance(elem, trtp.Tensor)]
        non_tensor_args = [_scalar_attr_to_python(v) for v in non_tensor_args]
        torch_kwargs = {k: _scalar_attr_to_python(v) for k, v in kwargs.items()}
        in_tensors = [torch.as_tensor(i, device="cuda") for i in tensor_args]

        dest_tensors = [torch.as_tensor(o, device="cuda") for o in outputs]

        stream = torch.cuda.ExternalStream(stream)
        with torch.cuda.stream(stream):
            out_tensors = torch_op(*in_tensors, *non_tensor_args, **torch_kwargs)
            if isinstance(out_tensors, torch.Tensor):
                out_tensors = (out_tensors,)
            [d.copy_(o) for (d, o) in zip(dest_tensors, out_tensors)]

    plugin_impl_func = f"""
{plugin_impl_signature}
    _generic_plugin_impl(outputs, stream, {input_signature})
    """

    plugin_impl_code = compile(plugin_impl_func, "<string>", "exec")

    # Use a per-plugin globals dict so each plugin gets its own isolated
    # _generic_plugin_impl binding. Writing to the shared module globals()
    # causes cross-plugin contamination when multiple plugins are registered
    # in the same process.
    plugin_globals = {**globals(), "_generic_plugin_impl": _generic_plugin_impl}

    plugin_impl = FunctionType(
        plugin_impl_code.co_consts[0], plugin_globals, impl_func_name
    )
    plugin_impl.__qualname__ = impl_func_name

    plugin_impl.__annotations__ = impl_func_annotation

    trtp.impl(plugin_name)(plugin_impl)


@needs_qdp_plugin  # type: ignore
def generate_plugin(plugin_name: str) -> None:
    """
    Generate the Plugin using external kernels and TensorRT Quick Deployable Plugin APIs.

    Args:
        plugin_name: the plugin name that is used to generate the plugin automatically.
            There should be existing kernels and pytorch custom operation for this plugin name.
    """
    _generate_plugin(plugin_name)


def register_plugin_with_aot(
    plugin_name: str,
    desc_fn: Any,
    autotune_fn: Optional[Any] = None,
    aot_fn: Optional[Any] = None,
) -> None:
    """Register a QDP plugin's descriptor, autotune, and AOT-impl callbacks with TRT.

    Centralises the three ``trtp`` registration calls so that both the
    automatic JIT path (``generate_plugin``) and the TTA AOT path
    (``register_custom_plugin``) converge on the same registration surface.

    Callers are responsible for idempotency and error handling — this function
    makes the raw ``trtp`` calls and raises whatever TRT raises.

    Args:
        plugin_name:  TRT QDP op name in ``"namespace::name"`` form.
        desc_fn:      Callable registered via ``@trtp.register``.  Must carry a
                      correct ``inspect.Signature`` with ``TensorDesc``-annotated
                      parameters.
        autotune_fn:  Optional callable registered via ``@trtp.autotune``.
                      Pass ``None`` to skip autotune registration (TRT will use a
                      single default tactic).
        aot_fn:       Optional callable registered via ``@trtp.aot_impl``.
                      Pass ``None`` to skip AOT registration (TRT will use the
                      JIT ``@trtp.impl`` path if one was registered separately).
    """
    try:
        import tensorrt.plugin as trtp
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT with plugin support is required for AOT plugin registration."
        ) from exc

    trtp.register(plugin_name)(desc_fn)

    if autotune_fn is not None:
        trtp.autotune(plugin_name)(autotune_fn)

    if aot_fn is not None:
        trtp.aot_impl(plugin_name)(aot_fn)
