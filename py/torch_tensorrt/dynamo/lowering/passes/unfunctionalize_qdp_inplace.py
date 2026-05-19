"""Reverse ``run_decompositions``' functionalization of mutating custom ops
that have a registered Dynamo converter (QDP in-place plugins).

``run_decompositions`` rewrites every call ``my_inplace_op(x, ...)`` (declared
with ``mutates_args=("x",)``) into::

    %af = auto_functionalized_v2(my_inplace_op, _x_base_index=N, _all_bases=[%x], ...)
    %getitem_0 = af[0]      # the op's actual return
    %getitem_k = af[k]      # for k in 1..len(_all_bases): the post-mutation base
    %copy_ = aten.copy_.default(%x, %getitem_k)   # propagate the mutation back

The result is correct for PyTorch eager, but our converter is registered against
the original mutating overload and the partitioner sees the HOP wrapper as
unsupported — so the whole subgraph is bailed out. This pass restores the
direct mutating call when the underlying op has a converter, lets the QDP
``.aliased()`` descriptor declared in `_generate_plugin.py` actually reach the
engine builder, and drops the now-redundant copy_/getitem nodes.
"""

import logging
from typing import Any, Dict, List

import torch

from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion._ConverterRegistry import DYNAMO_CONVERTERS

logger = logging.getLogger(__name__)


def _auto_functionalized_targets() -> List[Any]:
    # Both HOPs exist across torch versions; v2 is what current export emits.
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
    """Rebuild positional args for a direct call to ``op_overload`` from the
    HOP kwargs. The QDP-generated converter reads tensor inputs positionally
    (``args[0 : len(tensor_inputs)]``), so we must place args in schema order
    rather than passing them by name.

    ``auto_functionalized_v2`` packs mutated tensor arguments via
    ``_<arg>_base_index: N`` plus ``_all_bases: [t0, t1, ...]`` instead of
    inlining them. Non-mutated args are passed by name as-is.
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

    modified = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target not in af_targets:
            continue
        if not node.args or not hasattr(node.args[0], "_schema"):
            continue
        op_overload = node.args[0]
        if op_overload not in DYNAMO_CONVERTERS:
            # No converter for the underlying op — leave it functionalized.
            continue

        op_args = _reconstruct_op_args(op_overload, dict(node.kwargs))
        bases = node.kwargs.get("_all_bases", [])

        with gm.graph.inserting_before(node):
            new_call = gm.graph.call_function(op_overload, args=tuple(op_args))
            # Propagate the op's own meta["val"] (first element of the HOP's
            # tuple) so downstream shape extraction has what it needs.
            hop_val = node.meta.get("val")
            if isinstance(hop_val, tuple) and len(hop_val) >= 1:
                new_call.meta["val"] = hop_val[0]
            if "tensor_meta" in node.meta:
                new_call.meta["tensor_meta"] = node.meta["tensor_meta"]

        # Rewrite getitem users of the HOP. Both tuple slots — index 0 (the
        # op's actual return) and indices >= 1 (post-mutation bases) — get
        # routed to the direct op call. Routing index-k users to the original
        # base placeholder would point downstream nodes at the pre-mutation
        # FX value (and would also strand `new_call` as dead code), so we
        # anchor everything to the new call instead.
        getitem_users = [u for u in list(node.users) if u.target is _operator_getitem()]
        for user in getitem_users:
            user.replace_all_uses_with(new_call)
            gm.graph.erase_node(user)

        # If the HOP has any remaining users (unusual — would mean someone
        # consumed the tuple directly), bail rather than leave a dangling
        # reference to the now-stale HOP node.
        if list(node.users):
            raise RuntimeError(
                f"auto_functionalized_v2 node {node.name} has non-getitem users"
                f" {list(node.users)}; cannot un-functionalize safely."
            )
        gm.graph.erase_node(node)
        modified = True

    if not modified:
        return gm

    # `copy_(base, op_return)` was synthesized by functionalization to write
    # the mutation back through the base placeholder. With the direct mutating
    # call restored, the buffer is already mutated by the kernel — keeping the
    # copy_ would leave an unsupported mutating op blocking partitioning. Drop
    # it and route its users straight to the op return.
    for node in list(gm.graph.nodes):
        if (
            node.op == "call_function"
            and node.target is torch.ops.aten.copy_.default
            and len(node.args) >= 2
            and node.args[0].op == "placeholder"
        ):
            node.replace_all_uses_with(node.args[1])
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Un-functionalized QDP in-place ops:\n{gm.graph}")
    return gm


def _operator_getitem() -> Any:
    import operator

    return operator.getitem
