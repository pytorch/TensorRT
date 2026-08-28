"""Convert ``auto_functionalized_v2`` wrappers for QDP in-place plugins.

``run_decompositions`` functionalizes a mutating custom op into::

    %af = auto_functionalized_v2(my_inplace, _x_base_index=N, _all_bases=[%x], ...)
    %g0 = af[0]                 # the op's declared return
    %gk = af[k]                 # post-mutation value of base k-1
    %c  = aten.copy_(%x, %gk)   # write-back

Rather than rewriting the graph back into the mutating call, we convert the
wrapper itself. A capability validator claims only wrappers whose inner op has
a compatible plugin converter, delegates to that converter, and returns the
promised ``[*op_returns, *post_mutation_bases]`` tuple for downstream
``getitem`` nodes. A scoped no-op converter absorbs only the matching
write-back because the plugin already performed it in place. Non-plugin
``copy_`` nodes are untouched.
"""

import logging
import operator
from typing import Any, Dict, List, Mapping, NamedTuple, Optional

import torch
from torch.fx.node import Node, map_arg

from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    CallingConvention,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.plugins._alias_utils import (
    detect_and_validate_output_aliases,
    is_tensor_arg,
    tensor_positions,
)

logger = logging.getLogger(__name__)

_auto_functionalized_v2 = getattr(
    getattr(torch.ops, "higher_order", None), "auto_functionalized_v2", None
)
_WRAPPER_TARGETS = (
    (_auto_functionalized_v2,) if _auto_functionalized_v2 is not None else ()
)

# Alias identity is an operator contract and does not vary with shape.
_ALIAS_MAP_CACHE: Dict[Any, Dict[int, int]] = {}

_VIEW_METADATA_SUFFIXES = (
    "alias",
    "size",
    "stride",
    "storage_offset",
    "slice_dim",
    "slice_start",
    "slice_end",
)


class _UnsupportedMutation(RuntimeError):
    pass


class _ResolvedInnerConverter(NamedTuple):
    converter: Any
    calling_convention: CallingConvention
    info: Dict[str, bool]
    alias_map: Dict[int, int]


def _inner_op(node: Node) -> Any:
    return node.args[0] if node.args else None


def _reconstruct_op_args(op: Any, kwargs: Mapping[str, Any]) -> List[Any]:
    """Rebuild the inner call from v2 functionalization metadata."""
    bases = kwargs.get("_all_bases", [])
    op_args: List[Any] = []

    for arg in op._schema.arguments:
        prefix = f"_{arg.name}"
        if f"{prefix}_length" in kwargs:
            raise _UnsupportedMutation(
                f"In-place QDP plugin {op} cannot mutate tensor-list argument "
                f"'{arg.name}'."
            )

        base_key = f"{prefix}_base_index"
        if base_key not in kwargs:
            if arg.name in kwargs:
                op_args.append(kwargs[arg.name])
            elif arg.has_default_value():
                op_args.append(arg.default_value)
            else:
                raise RuntimeError(
                    f"auto_functionalized_v2 is missing argument '{arg.name}' for {op}"
                )
            continue

        if any(f"{prefix}_{suffix}" in kwargs for suffix in _VIEW_METADATA_SUFFIXES):
            raise _UnsupportedMutation(
                f"In-place QDP plugin {op} can mutate only a base tensor; "
                f"argument '{arg.name}' is a view."
            )

        base_index = kwargs[base_key]
        if base_index is None:
            op_args.append(None)
        elif (
            not isinstance(base_index, int)
            or isinstance(base_index, bool)
            or not 0 <= base_index < len(bases)
        ):
            raise _UnsupportedMutation(
                f"Invalid {base_key}={base_index!r} for {op}; expected an index "
                f"into {len(bases)} functionalized base(s)."
            )
        else:
            op_args.append(bases[base_index])

    return op_args


def _node_value(value: Any) -> Any:
    if not isinstance(value, Node):
        return value
    if "val" not in value.meta:
        raise RuntimeError(f"Node {value.name!r} has no meta['val']")
    return value.meta["val"]


def _concrete_dim(dim: Any) -> int:
    """Get a representative size without specializing a symbolic dimension."""
    if isinstance(dim, int) and not isinstance(dim, bool):
        value = dim
    else:
        value = getattr(getattr(dim, "node", None), "hint", 8)
    return value if isinstance(value, int) and value > 0 else 8


def _output_alias_map(op: Any, op_args: List[Any]) -> Dict[int, int]:
    """Run the fake kernel and validate its output-to-input alias signal."""
    if op in _ALIAS_MAP_CACHE:
        return _ALIAS_MAP_CACHE[op]

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            call_args: List[Any] = []
            fake_tensors: List[Any] = []
            for schema_arg, value in zip(op._schema.arguments, op_args):
                value = map_arg(value, _node_value)
                if not is_tensor_arg(schema_arg):
                    call_args.append(value)
                    continue

                shape = getattr(value, "shape", None)
                dims = (
                    [_concrete_dim(dim) for dim in shape] if shape is not None else [8]
                )
                dtype = getattr(value, "dtype", torch.float32)
                if not isinstance(dtype, torch.dtype):
                    dtype = torch.float32
                fake = torch.empty(dims, dtype=dtype, device="cuda")
                call_args.append(fake)
                fake_tensors.append(fake)

            output = op(*call_args)
    except Exception as exc:
        raise RuntimeError(
            f"Could not run the fake implementation for in-place QDP plugin {op}. "
            "It must accept CUDA fake tensors and return every mutated tensor "
            "by identity."
        ) from exc

    outputs = list(output) if isinstance(output, (tuple, list)) else [output]
    alias_map = detect_and_validate_output_aliases(op, outputs, fake_tensors)
    _ALIAS_MAP_CACHE[op] = alias_map
    return alias_map


def _clone_inner_node(wrapper: Node, op: Any, op_args: List[Any]) -> Node:
    """Build an inner-op node so the registry can apply normal validation."""
    graph = torch.fx.Graph()
    env: Dict[Node, Node] = {}

    def clone(node: Node) -> Node:
        if node not in env:
            env[node] = graph.node_copy(node, clone)
        return env[node]

    inner = graph.call_function(op, args=map_arg(tuple(op_args), clone), kwargs={})
    inner.meta = dict(wrapper.meta)

    wrapper_value = wrapper.meta.get("val")
    num_returns = len(op._schema.returns)
    if isinstance(wrapper_value, (tuple, list)):
        values = wrapper_value[:num_returns]
        inner.meta["val"] = values[0] if num_returns == 1 else tuple(values)
    return inner


def _base_to_tensor_arg(op: Any, kwargs: Mapping[str, Any]) -> Dict[int, int]:
    positions = {
        position: index for index, position in enumerate(tensor_positions(op._schema))
    }
    return {
        base_index: positions[position]
        for position, arg in enumerate(op._schema.arguments)
        if (base_index := kwargs.get(f"_{arg.name}_base_index")) is not None
        and position in positions
    }


def _matching_writeback_wrapper(node: Node) -> Optional[Node]:
    if node.op != "call_function" or node.target is not torch.ops.aten.copy_.default:
        return None
    if len(node.args) < 2:
        return None

    source = node.args[1]
    if not isinstance(source, Node):
        return None
    if source.op != "call_function" or source.target is not operator.getitem:
        return None

    wrapper = source.args[0]
    if not isinstance(wrapper, Node) or wrapper.target not in _WRAPPER_TARGETS:
        return None
    op = _inner_op(wrapper)
    if op is None or not hasattr(op, "_schema"):
        return None

    index = source.args[1]
    num_returns = len(op._schema.returns)
    if not isinstance(index, int) or isinstance(index, bool) or index < num_returns:
        return None
    bases = wrapper.kwargs.get("_all_bases", [])
    base_index = index - num_returns
    if base_index >= len(bases) or node.args[0] is not bases[base_index]:
        return None
    return wrapper


def _is_matching_writeback(node: Node) -> bool:
    return _matching_writeback_wrapper(node) is not None


def _has_unsafe_multi_output_use(
    wrapper: Node, op: Any, alias_map: Dict[int, int]
) -> bool:
    if len(op._schema.returns) <= 1:
        return False

    base_to_tensor = _base_to_tensor_arg(op, wrapper.kwargs)
    aliased_tensor_args = set(alias_map.values())
    aliased_slots = set(alias_map)
    aliased_slots.update(
        len(op._schema.returns) + base_index
        for base_index, tensor_arg in base_to_tensor.items()
        if tensor_arg in aliased_tensor_args
    )

    for getitem in wrapper.users:
        if (
            getitem.op != "call_function"
            or getitem.target is not operator.getitem
            or len(getitem.args) < 2
            or getitem.args[1] not in aliased_slots
        ):
            continue
        if any(
            user.op != "output" and not _is_matching_writeback(user)
            for user in getitem.users
        ):
            logger.warning(
                "Leaving %s in PyTorch because TensorRT cannot safely consume "
                "an aliased output from a multi-output QDP plugin.",
                op,
            )
            return True
    return False


def _resolve_inner_converter(wrapper: Node) -> Optional[_ResolvedInnerConverter]:
    op = _inner_op(wrapper)
    if op is None or not hasattr(op, "_schema"):
        return None

    try:
        op_args = _reconstruct_op_args(op, wrapper.kwargs)
    except _UnsupportedMutation as exc:
        logger.warning("Leaving %s in PyTorch: %s", op, exc)
        return None

    converter_entry = DYNAMO_CONVERTERS.get(_clone_inner_node(wrapper, op, op_args))
    if converter_entry is None:
        return None

    converter, calling_convention, info = converter_entry
    if not info.get("requires_aliased_plugin_io", False):
        return None

    alias_map = _output_alias_map(op, op_args)
    if _has_unsafe_multi_output_use(wrapper, op, alias_map):
        return None
    return _ResolvedInnerConverter(converter, calling_convention, info, alias_map)


def wrapper_wraps_plugin_op(node: Node, settings: Any) -> bool:
    return _resolve_inner_converter(node) is not None


def _convert_auto_functionalized(
    ctx: Any, target: Any, args: Any, kwargs: Any, name: str
) -> Any:
    wrapper = ctx.current_node
    if not isinstance(wrapper, Node):
        raise RuntimeError("Missing current FX node for auto_functionalized_v2")
    resolved = _resolve_inner_converter(wrapper)
    if resolved is None:
        raise RuntimeError(f"No validated in-place QDP converter for {wrapper}")

    op = args[0]
    op_args = _reconstruct_op_args(op, kwargs)
    ctx.requires_output_allocator |= resolved.info.get(
        "requires_output_allocator", False
    )
    ctx.requires_native_multidevice |= resolved.info.get(
        "requires_native_multidevice", False
    )
    if resolved.calling_convention is CallingConvention.LEGACY:
        result = resolved.converter(ctx.net, op, tuple(op_args), {}, name)
    else:
        result = resolved.converter(ctx, op, tuple(op_args), {}, name)

    op_returns = list(result) if isinstance(result, (tuple, list)) else [result]
    expected_returns = len(op._schema.returns)
    if len(op_returns) != expected_returns:
        raise RuntimeError(
            f"Converter for {op} returned {len(op_returns)} value(s), expected "
            f"{expected_returns}."
        )

    base_to_tensor = _base_to_tensor_arg(op, kwargs)
    tensor_to_output = {
        tensor_arg: output_index
        for output_index, tensor_arg in resolved.alias_map.items()
    }
    base_outputs = [
        (
            op_returns[tensor_to_output[base_to_tensor[index]]]
            if index in base_to_tensor and base_to_tensor[index] in tensor_to_output
            else base
        )
        for index, base in enumerate(kwargs.get("_all_bases", []))
    ]

    outputs = op_returns + base_outputs
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def _is_aliased_writeback(node: Node, settings: Any) -> bool:
    wrapper = _matching_writeback_wrapper(node)
    return wrapper is not None and wrapper_wraps_plugin_op(wrapper, settings)


def _convert_aliased_writeback_copy(
    ctx: Any, target: Any, args: Any, kwargs: Any, name: str
) -> Any:
    return args[1]


for _target in _WRAPPER_TARGETS:
    dynamo_tensorrt_converter(
        _target,
        capability_validator=wrapper_wraps_plugin_op,
        # The validator delegates dynamic-shape support to the inner converter.
        supports_dynamic_shapes=True,
        requires_aliased_plugin_io=True,
    )(_convert_auto_functionalized)

dynamo_tensorrt_converter(
    torch.ops.aten.copy_.default,
    capability_validator=_is_aliased_writeback,
    supports_dynamic_shapes=True,
)(_convert_aliased_writeback_copy)
