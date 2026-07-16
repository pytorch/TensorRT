"""Convert ``auto_functionalized(_v2)`` wrappers for QDP in-place plugins.

``run_decompositions`` rewrites a mutating custom op ``my_inplace(x, ...)`` into::

    %af = auto_functionalized_v2(my_inplace, _x_base_index=N, _all_bases=[%x], ...)
    %g0 = af[0]     # the op's declared return
    %gk = af[k]     # post-mutation value of base (k = 1..len(_all_bases))
    %c  = aten.copy_(%x, %gk)

The op's plugin converter is registered against the original mutating overload,
so the wrapper (a higher-order op) has no direct converter. Rather than rewrite
the graph back into the mutating call (which reverses functionalization and
requires deleting the ``copy_`` write-backs), we register a converter on the
wrapper itself, gated by a ``capability_validator`` so it only claims wrappers
whose inner op *does* have a plugin converter. The graph stays functional; the
scoping is per-node.

The wrapper converter reconstructs the inner op's positional args in schema
order and delegates to the inner op's existing (auto-generated) converter, then
returns the ``[*op_returns, *post_mutation_bases]`` tuple the wrapper promises so
the downstream ``getitem`` converter unpacks it. A companion no-op converter
absorbs the aliased write-back ``copy_`` (the plugin already wrote in place).

When a mutated tensor is returned directly, functionalization leaves the
write-back ``copy_`` as a graph output with no ``meta["val"]``. Symbolic-shape
extraction handles this itself by resolving a ``copy_`` output to its
destination's value (see ``_symbolic_shape_capture._resolve_meta_val``), so no
lowering pass is needed.
"""

import logging
import operator
from typing import Any, Dict, List

import torch

from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    dynamo_tensorrt_converter,
)

logger = logging.getLogger(__name__)


def auto_functionalized_targets() -> List[Any]:
    """The higher-order wrappers functionalization may emit, newest first."""
    higher_order = getattr(torch.ops, "higher_order", None)
    out: List[Any] = []
    if higher_order is None:
        return out
    for name in ("auto_functionalized_v2", "auto_functionalized"):
        op = getattr(higher_order, name, None)
        if op is not None:
            out.append(op)
    return out


def _inner_op(node: Any) -> Any:
    return node.args[0] if getattr(node, "args", None) else None


def wrapper_wraps_plugin_op(node: Any, settings: Any) -> bool:
    """capability_validator: true only when the wrapped op has a converter."""
    op = _inner_op(node)
    return op is not None and hasattr(op, "_schema") and op in DYNAMO_CONVERTERS


def _is_tensor_arg(arg: Any) -> bool:
    return bool(arg.type.isSubtypeOf(torch._C.TensorType.get()))


def _reconstruct_op_args(op: Any, kwargs: Dict[str, Any]) -> List[Any]:
    """Rebuild the inner op's positional args in schema order.

    ``auto_functionalized_v2`` packs mutated tensors as ``_all_bases=[...]`` plus
    ``_<arg>_base_index: N``; other args are passed by name (or defaulted).
    """
    bases = kwargs.get("_all_bases", [])
    out: List[Any] = []
    for arg in op._schema.arguments:
        base_key = f"_{arg.name}_base_index"
        if base_key in kwargs:
            out.append(bases[kwargs[base_key]])
        elif arg.name in kwargs:
            out.append(kwargs[arg.name])
        elif arg.has_default_value():
            out.append(arg.default_value)
        else:
            raise RuntimeError(
                f"auto_functionalized missing argument '{arg.name}' for {op}"
            )
    return out


def _output_to_tensor_arg_alias(op: Any, op_args: List[Any]) -> Dict[int, int]:
    """``{out_idx: tensor_arg_idx}`` for the inner op.

    Mirrors ``_generate_plugin``'s alias detection: an output aliases a mutated
    input iff the schema marks that input ``is_write`` AND the fake/meta run
    returns that input tensor by identity. ``tensor_arg_idx`` is the position
    among tensor-typed schema args (== plugin input order). Returns ``{}`` on any
    failure so the caller falls back to routing base slots to the base itensor.
    """
    schema = op._schema
    tensor_positions = [i for i, a in enumerate(schema.arguments) if _is_tensor_arg(a)]
    mutated_tensor_idx = {
        tensor_positions.index(i)
        for i, a in enumerate(schema.arguments)
        if _is_tensor_arg(a) and a.alias_info is not None and a.alias_info.is_write
    }
    if not mutated_tensor_idx:
        return {}

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            call_args: List[Any] = []
            fake_tensors: List[Any] = []
            for i, a in enumerate(schema.arguments):
                if _is_tensor_arg(a):
                    itensor = op_args[i]
                    shp = getattr(itensor, "shape", None)
                    if shp is not None and len(shp) > 0:
                        dims = [int(d) if int(d) > 0 else 8 for d in shp]
                    else:
                        dims = [8]
                    fake = torch.empty(dims)
                    call_args.append(fake)
                    fake_tensors.append(fake)
                else:
                    call_args.append(op_args[i])
            output = op(*call_args)
            outs = list(output) if isinstance(output, (tuple, list)) else [output]
    except Exception as e:  # pragma: no cover - defensive
        logger.debug(f"alias detection fake-run failed for {op}: {e}")
        return {}

    alias_map: Dict[int, int] = {}
    for out_idx, fake_out in enumerate(outs):
        for tensor_arg_idx in mutated_tensor_idx:
            if (
                tensor_arg_idx < len(fake_tensors)
                and fake_out is fake_tensors[tensor_arg_idx]
            ):
                alias_map[out_idx] = tensor_arg_idx
                break
    return alias_map


def _convert_auto_functionalized(
    ctx: Any, target: Any, args: Any, kwargs: Any, name: str
) -> Any:
    op = args[0]
    kw = dict(kwargs)
    op_args = _reconstruct_op_args(op, kw)

    inner_converter, _calling_convention = (
        DYNAMO_CONVERTERS.__getitem_without_validation__(op)
    )
    result = inner_converter(ctx, op, tuple(op_args), {}, name)
    op_returns = list(result) if isinstance(result, (tuple, list)) else [result]

    # Wrapper output layout is [*op_returns, *post_mutation_bases]. Each base's
    # post-mutation value is the op-return that aliases it (the plugin wrote it
    # in place); if nothing aliases it, fall back to the base itensor.
    bases = kw.get("_all_bases", [])
    schema = op._schema
    tensor_positions = [i for i, a in enumerate(schema.arguments) if _is_tensor_arg(a)]
    base_to_tensor_arg: Dict[int, int] = {}
    for i, a in enumerate(schema.arguments):
        base_key = f"_{a.name}_base_index"
        if base_key in kw:
            base_to_tensor_arg[kw[base_key]] = tensor_positions.index(i)

    alias_map = _output_to_tensor_arg_alias(op, op_args)
    tensor_arg_to_out = {ti: oi for oi, ti in alias_map.items()}

    base_outs: List[Any] = []
    for b in range(len(bases)):
        tensor_arg_idx = base_to_tensor_arg.get(b)
        out_idx = (
            tensor_arg_to_out.get(tensor_arg_idx)
            if tensor_arg_idx is not None
            else None
        )
        if out_idx is not None and out_idx < len(op_returns):
            base_outs.append(op_returns[out_idx])
        else:
            base_outs.append(bases[b])

    outs = op_returns + base_outs
    return outs[0] if len(outs) == 1 else tuple(outs)


def _is_aliased_writeback(node: Any, settings: Any) -> bool:
    """capability_validator: ``copy_(placeholder, getitem(af_with_plugin, k))``
    — the redundant write-back the aliased plugin already performed in place."""
    if len(node.args) < 2:
        return False
    src = node.args[1]
    if not (
        getattr(src, "op", None) == "call_function" and src.target is operator.getitem
    ):
        return False
    af = src.args[0]
    return (
        getattr(af, "op", None) == "call_function"
        and af.target in auto_functionalized_targets()
        and wrapper_wraps_plugin_op(af, settings)
    )


def _convert_aliased_writeback_copy(
    ctx: Any, target: Any, args: Any, kwargs: Any, name: str
) -> Any:
    # dest already aliases src (plugin wrote in place); pass src through.
    return args[1]


for _target in auto_functionalized_targets():
    dynamo_tensorrt_converter(
        _target,
        capability_validator=wrapper_wraps_plugin_op,
        supports_dynamic_shapes=True,
    )(_convert_auto_functionalized)

dynamo_tensorrt_converter(
    torch.ops.aten.copy_.default,
    capability_validator=_is_aliased_writeback,
    supports_dynamic_shapes=True,
)(_convert_aliased_writeback_copy)
