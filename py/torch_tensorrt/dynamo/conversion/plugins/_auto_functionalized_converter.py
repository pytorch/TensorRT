"""Convert ``auto_functionalized(_v2)`` wrappers for QDP in-place plugins.

``run_decompositions`` functionalizes a mutating custom op into::

    %af = auto_functionalized_v2(my_inplace, _x_base_index=N, _all_bases=[%x], ...)
    %g0 = af[0]                 # the op's declared return
    %gk = af[k]                 # post-mutation value of base k-1
    %c  = aten.copy_(%x, %gk)   # write-back

Rather than rewriting the graph back into the mutating call, we convert the
wrapper itself: a converter gated by a ``capability_validator`` (claims only
wrappers whose inner op has a plugin converter) reconstructs the inner op's
args, delegates to its plugin converter, and returns the promised
``[*op_returns, *post_mutation_bases]`` tuple for the downstream ``getitem``
nodes. A scoped no-op converter absorbs the redundant write-back ``copy_``
(the plugin already wrote in place). The graph stays functional; non-plugin
``copy_`` nodes are untouched. A write-back's missing ``meta['val']`` is filled
generically by ``FakeTensorUpdater`` during ``post_lowering``.
"""

import logging
import operator
from typing import Any, Dict, List

import torch

from torch_tensorrt.dynamo.conversion._ConverterRegistry import (
    DYNAMO_CONVERTERS,
    dynamo_tensorrt_converter,
)
from torch_tensorrt.dynamo.conversion.plugins._alias_utils import (
    detect_output_aliases,
    is_tensor_arg,
    mutated_tensor_indices,
    tensor_positions,
)

logger = logging.getLogger(__name__)

# The higher-order wrappers functionalization may emit, newest first.
_WRAPPER_TARGETS = tuple(
    op
    for name in ("auto_functionalized_v2", "auto_functionalized")
    if (op := getattr(getattr(torch.ops, "higher_order", None), name, None)) is not None
)

# op -> {out_idx: tensor_arg_idx}; alias identity is shape-invariant, so one
# fake run per op suffices.
_ALIAS_MAP_CACHE: Dict[Any, Dict[int, int]] = {}


def _inner_op(node: Any) -> Any:
    return node.args[0] if getattr(node, "args", None) else None


def wrapper_wraps_plugin_op(node: Any, settings: Any) -> bool:
    """capability_validator: true only when the wrapped op has a converter."""
    op = _inner_op(node)
    return op is not None and hasattr(op, "_schema") and op in DYNAMO_CONVERTERS


def _reconstruct_op_args(op: Any, kwargs: Dict[str, Any]) -> List[Any]:
    """Rebuild the inner op's positional args in schema order.

    ``auto_functionalized_v2`` packs mutated tensors as ``_all_bases=[...]``
    plus ``_<arg>_base_index: N``; other args are passed by name or defaulted.
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


def _output_alias_map(op: Any, op_args: List[Any]) -> Dict[int, int]:
    """``{out_idx: tensor_arg_idx}`` for the inner op (see ``_alias_utils``).

    Returns ``{}`` on failure (caller falls back to routing base slots to the
    base itensor).
    """
    if op in _ALIAS_MAP_CACHE:
        return _ALIAS_MAP_CACHE[op]

    schema = op._schema
    alias_map: Dict[int, int] = {}
    mutated = mutated_tensor_indices(schema)
    if mutated:
        try:
            from torch._subclasses.fake_tensor import FakeTensorMode

            with FakeTensorMode():
                call_args: List[Any] = []
                fake_tensors: List[Any] = []
                for i, a in enumerate(schema.arguments):
                    if is_tensor_arg(a):
                        # Symbolic/unknown dims are replaced with a dummy size;
                        # alias identity does not depend on shape.
                        shp = getattr(op_args[i], "shape", None)
                        dims = (
                            [int(d) if int(d) > 0 else 8 for d in shp] if shp else [8]
                        )
                        fake = torch.empty(dims)
                        call_args.append(fake)
                        fake_tensors.append(fake)
                    else:
                        call_args.append(op_args[i])
                output = op(*call_args)
                outs = list(output) if isinstance(output, (tuple, list)) else [output]
            alias_map = detect_output_aliases(outs, fake_tensors, mutated)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"alias detection fake-run failed for {op}: {e}")
            alias_map = {}

    _ALIAS_MAP_CACHE[op] = alias_map
    return alias_map


def _convert_auto_functionalized(
    ctx: Any, target: Any, args: Any, kwargs: Any, name: str
) -> Any:
    op = args[0]
    op_args = _reconstruct_op_args(op, kwargs)

    inner_converter, _calling_convention = (
        DYNAMO_CONVERTERS.__getitem_without_validation__(op)
    )
    result = inner_converter(ctx, op, tuple(op_args), {}, name)
    op_returns = list(result) if isinstance(result, (tuple, list)) else [result]

    # Wrapper output layout is [*op_returns, *post_mutation_bases]: a base's
    # post-mutation value is the op-return aliasing it (the plugin wrote it in
    # place); if nothing aliases it, fall back to the base itensor.
    bases = kwargs.get("_all_bases", [])
    tensor_pos = tensor_positions(op._schema)
    base_to_tensor_arg = {
        kwargs[key]: tensor_pos.index(i)
        for i, a in enumerate(op._schema.arguments)
        if (key := f"_{a.name}_base_index") in kwargs
    }
    tensor_arg_to_out = {t: o for o, t in _output_alias_map(op, op_args).items()}

    base_outs: List[Any] = []
    for b, base in enumerate(bases):
        t_idx = base_to_tensor_arg.get(b)
        out_idx = tensor_arg_to_out.get(t_idx) if t_idx is not None else None
        if out_idx is not None and out_idx < len(op_returns):
            base_outs.append(op_returns[out_idx])
        else:
            base_outs.append(base)

    outs = op_returns + base_outs
    return outs[0] if len(outs) == 1 else tuple(outs)


def _is_aliased_writeback(node: Any, settings: Any) -> bool:
    """capability_validator: ``copy_(dest, getitem(af_with_plugin, k))`` — the
    redundant write-back the aliased plugin already performed in place."""
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
        and af.target in _WRAPPER_TARGETS
        and wrapper_wraps_plugin_op(af, settings)
    )


def _convert_aliased_writeback_copy(
    ctx: Any, target: Any, args: Any, kwargs: Any, name: str
) -> Any:
    # dest already aliases src (plugin wrote in place); pass src through.
    return args[1]


for _target in _WRAPPER_TARGETS:
    dynamo_tensorrt_converter(
        _target,
        capability_validator=wrapper_wraps_plugin_op,
        supports_dynamic_shapes=True,
        # auto_functionalized only wraps mutating ops, and the validator only
        # claims wrappers whose inner op has a plugin converter — so a claimed
        # wrapper always implies a plugin with aliased I/O. The flag must live
        # here because the wrapper delegates to the inner converter directly,
        # bypassing the registry's per-node flag propagation.
        requires_aliased_plugin_io=True,
    )(_convert_auto_functionalized)

dynamo_tensorrt_converter(
    torch.ops.aten.copy_.default,
    capability_validator=_is_aliased_writeback,
    supports_dynamic_shapes=True,
)(_convert_aliased_writeback_copy)
