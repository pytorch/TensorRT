import itertools
import logging
import re
from types import FunctionType
from typing import Any, Callable, Tuple

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

    # helper function that generates the required signature based on the torch operation
    def generate_signature(
        torch_op: Callable[[Any], Any],
    ) -> Tuple[str, str, str, dict[str, Any], dict[str, Any]]:
        schema = torch_op._schemas[""]

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
        num_outputs = len(schema.returns)
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
    ) = generate_signature(torch_op)

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
