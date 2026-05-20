"""Reverse ``run_decompositions``' functionalization of mutating custom ops
that have a registered Dynamo converter (QDP in-place plugins).

``run_decompositions`` rewrites ``my_inplace_op(x, ...)`` into::

    %af = auto_functionalized_v2(my_inplace_op, _x_base_index=N, _all_bases=[%x], ...)
    %g0 = af[0]    # the op's actual return
    %gk = af[k]    # post-mutation base (k = 1..len(_all_bases))
    %copy_ = aten.copy_.default(%x, %gk)

Correct in eager, but our converter is registered against the original
mutating overload — the partitioner sees the HOP wrapper as unsupported and
bails. This pass restores the direct mutating call when a converter exists
and drops the synthesized copy_ nodes.
"""

import logging
import operator
from typing import Any, Dict, List

import torch

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

logger = logging.getLogger(__name__)


def _auto_functionalized_targets() -> List[Any]:
    targets: List[Any] = []
    higher_order = getattr(torch.ops, "higher_order", None)
    if higher_order is None:
        return targets
    for name in ("auto_functionalized_v2", "auto_functionalized"):
        op = getattr(higher_order, name, None)
        if op is not None:
            targets.append(op)
    return targets


def _reconstruct_op_args(op_overload: Any, node_kwargs: Dict[str, Any]) -> List[Any]:
    """Rebuild positional args for a direct call to ``op_overload``.

    The QDP-generated converter reads tensor inputs positionally
    (``args[0 : len(tensor_inputs)]``), so args must be in schema order.
    ``auto_functionalized_v2`` packs mutated tensors via
    ``_<arg>_base_index: N`` + ``_all_bases: [t0, ...]``; non-mutated args
    are passed by name as-is.
    """
    bases = node_kwargs.get("_all_bases", [])
    out: List[Any] = []
    schema = op_overload._schema
    for arg in schema.arguments:
        base_key = f"_{arg.name}_base_index"
        if base_key in node_kwargs:
            out.append(bases[node_kwargs[base_key]])
        elif arg.name in node_kwargs:
            out.append(node_kwargs[arg.name])
        elif arg.has_default_value():
            out.append(arg.default_value)
        else:
            raise RuntimeError(
                f"auto_functionalized_v2 missing argument '{arg.name}' for"
                f" {op_overload} (no value and no default)"
            )
    return out


def unfunctionalize_qdp_inplace(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    af_targets = _auto_functionalized_targets()
    if not af_targets:
        return gm

    converter_check_cache: Dict[Any, bool] = {}
    aten_copy_ = torch.ops.aten.copy_.default
    hops_to_rewrite: List[Any] = []
    copy_candidates: List[Any] = []

    for node in list(gm.graph.nodes):
        if node.op != "call_function":
            continue
        if node.target in af_targets:
            if not node.args or not hasattr(node.args[0], "_schema"):
                continue
            op_overload = node.args[0]
            has_converter = converter_check_cache.get(op_overload)
            if has_converter is None:
                has_converter = op_overload in DYNAMO_CONVERTERS
                converter_check_cache[op_overload] = has_converter
            if has_converter:
                hops_to_rewrite.append(node)
        elif (
            node.target is aten_copy_
            and len(node.args) >= 2
            and getattr(node.args[0], "op", None) == "placeholder"
        ):
            copy_candidates.append(node)

    if not hops_to_rewrite:
        return gm

    for node in hops_to_rewrite:
        op_overload = node.args[0]
        op_args = _reconstruct_op_args(op_overload, dict(node.kwargs))
        n_outputs = len(op_overload._schema.returns)
        hop_val = node.meta.get("val")
        bases = node.kwargs.get("_all_bases", [])

        with gm.graph.inserting_before(node):
            new_call = gm.graph.call_function(op_overload, args=tuple(op_args))
            if isinstance(hop_val, tuple) and len(hop_val) >= 1:
                if n_outputs == 1:
                    new_call.meta["val"] = hop_val[0]
                else:
                    new_call.meta["val"] = tuple(hop_val[:n_outputs])
            if "tensor_meta" in node.meta:
                new_call.meta["tensor_meta"] = node.meta["tensor_meta"]

        # For single-output ops the op's return and every post-mutation base
        # are the same tensor (the in-place result), so routing all getitem
        # users to ``new_call`` is correct and keeps it alive. For
        # multi-output ops we materialize one getitem per return and route
        # base-slot users to the corresponding base placeholder — the
        # mutation has already been applied in place by ``new_call``.
        getitem_users = [u for u in list(node.users) if u.target is operator.getitem]
        if n_outputs == 1:
            for user in getitem_users:
                user.replace_all_uses_with(new_call)
                gm.graph.erase_node(user)
        else:
            return_getitems: List[Any] = []
            with gm.graph.inserting_after(new_call):
                for i in range(n_outputs):
                    g = gm.graph.call_function(
                        operator.getitem, args=(new_call, i)
                    )
                    if isinstance(hop_val, tuple) and i < len(hop_val):
                        g.meta["val"] = hop_val[i]
                    return_getitems.append(g)
            for user in getitem_users:
                idx = user.args[1]
                if idx < n_outputs:
                    user.replace_all_uses_with(return_getitems[idx])
                else:
                    base_idx = idx - n_outputs
                    user.replace_all_uses_with(bases[base_idx])
                gm.graph.erase_node(user)

        if list(node.users):
            raise RuntimeError(
                f"auto_functionalized_v2 node {node.name} has non-getitem users"
                f" {list(node.users)}; cannot un-functionalize safely."
            )
        gm.graph.erase_node(node)

    # Functionalization adds copy_(base, op_return) to write the mutation
    # back through the placeholder. The direct call already mutates the
    # buffer, so the copy_ is redundant and would block partitioning.
    for node in copy_candidates:
        node.replace_all_uses_with(node.args[1])
        gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Un-functionalized QDP in-place ops:\n{gm.graph}")
    return gm
