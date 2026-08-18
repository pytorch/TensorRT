"""Lower ``higher_order.associative_scan`` for Mamba's affine recurrence.

``combine_mode="pointwise"`` keeps the HOP intact through export. The only
combine pattern this pass accepts is Mamba's first-order linear recurrence::

    combine((a_l, b_l), (a_r, b_r)) = (a_l * a_r, a_r * b_l + b_r)

which is ``h_t = a_t * h_{t-1} + b_t``. Anything else is left alone.

For a static scan length the HOP is replaced with a Hillis–Steele inclusive
scan built from ``slice`` / ``cat`` / ``mul`` / ``add`` / ``ones_like`` /
``zeros_like`` — all of which already have converters. Dynamic sequence
lengths are declined so today's PyTorch fallback behaviour is preserved.
"""

from __future__ import annotations

import logging
import math
import operator
from typing import Optional, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def _is_associative_scan(target: object) -> bool:
    if target is torch.ops.higher_order.associative_scan:
        return True
    name = str(target)
    return "associative_scan" in name and "higher_order" in name


def _is_mul(n: torch.fx.Node, x: torch.fx.Node, y: torch.fx.Node) -> bool:
    return (
        n.op == "call_function"
        and n.target
        in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.default, operator.mul)
        and set(n.args[:2]) == {x, y}
    )


def _is_add(n: torch.fx.Node, x: torch.fx.Node, y: torch.fx.Node) -> bool:
    return (
        n.op == "call_function"
        and n.target
        in (torch.ops.aten.add.Tensor, torch.ops.aten.add.default, operator.add)
        and set(n.args[:2]) == {x, y}
    )


def _unwrap_output_pair(output_node: torch.fx.Node) -> Optional[Tuple[torch.fx.Node, torch.fx.Node]]:
    out_args = output_node.args[0]
    # wrap_combine_fn_flat may nest the pair as a list/tuple of length 1
    while isinstance(out_args, (list, tuple)) and len(out_args) == 1:
        out_args = out_args[0]
    if not isinstance(out_args, (list, tuple)) or len(out_args) != 2:
        return None
    o0, o1 = out_args
    if not isinstance(o0, torch.fx.Node) or not isinstance(o1, torch.fx.Node):
        return None
    return o0, o1


def _is_mamba_affine_combine(combine_gm: torch.fx.GraphModule) -> bool:
    """True iff the combine subgraph is ``(a_l*a_r, a_r*b_l + b_r)``."""
    placeholders = [n for n in combine_gm.graph.nodes if n.op == "placeholder"]
    output = next(n for n in combine_gm.graph.nodes if n.op == "output")
    if len(placeholders) != 4:
        return False

    pair = _unwrap_output_pair(output)
    if pair is None:
        return False
    o0, o1 = pair

    a_l, b_l, a_r, b_r = placeholders
    if not _is_mul(o0, a_l, a_r):
        return False

    # o1 = (a_r * b_l) + b_r
    mul_nodes = [
        n
        for n in combine_gm.graph.nodes
        if n.op == "call_function"
        and n.target
        in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.default, operator.mul)
        and set(n.args[:2]) == {a_r, b_l}
    ]
    if len(mul_nodes) != 1:
        return False
    return _is_add(o1, mul_nodes[0], b_r)


def _static_scan_length(xs: Tuple[torch.fx.Node, ...]) -> Optional[int]:
    """Return the concrete length along dim 0, or None if dynamic/unknown."""
    lengths = set()
    for x in xs:
        val = x.meta.get("val")
        if val is None or not hasattr(val, "shape") or len(val.shape) < 1:
            return None
        length = val.shape[0]
        if isinstance(length, torch.SymInt):
            return None
        if not isinstance(length, int) or length < 0:
            return None
        lengths.add(int(length))
    if len(lengths) != 1:
        return None
    return lengths.pop()


def _hillis_steele_scan(
    gm: torch.fx.GraphModule,
    a: torch.fx.Node,
    b: torch.fx.Node,
    scan_len: int,
    before: torch.fx.Node,
) -> Tuple[torch.fx.Node, torch.fx.Node]:
    """Inclusive Hillis–Steele scan of ``(a, b)`` along dim 0.

    Identity for the affine combine is ``(1, 0)``. Positions ``i < step`` keep
    their value because they are combined with that identity.
    """
    with gm.graph.inserting_before(before):
        for d in range(math.ceil(math.log2(scan_len)) if scan_len > 1 else 0):
            step = 1 << d
            a_prefix = gm.graph.call_function(
                torch.ops.aten.slice.Tensor, (a, 0, 0, scan_len - step, 1)
            )
            b_prefix = gm.graph.call_function(
                torch.ops.aten.slice.Tensor, (b, 0, 0, scan_len - step, 1)
            )
            a_head = gm.graph.call_function(
                torch.ops.aten.slice.Tensor, (a, 0, 0, step, 1)
            )
            b_head = gm.graph.call_function(
                torch.ops.aten.slice.Tensor, (b, 0, 0, step, 1)
            )
            ones = gm.graph.call_function(torch.ops.aten.ones_like.default, (a_head,))
            zeros = gm.graph.call_function(torch.ops.aten.zeros_like.default, (b_head,))
            a_left = gm.graph.call_function(
                torch.ops.aten.cat.default, ([ones, a_prefix], 0)
            )
            b_left = gm.graph.call_function(
                torch.ops.aten.cat.default, ([zeros, b_prefix], 0)
            )
            a_new = gm.graph.call_function(torch.ops.aten.mul.Tensor, (a_left, a))
            tmp = gm.graph.call_function(torch.ops.aten.mul.Tensor, (a, b_left))
            b_new = gm.graph.call_function(torch.ops.aten.add.Tensor, (tmp, b))
            a, b = a_new, b_new
    return a, b


def _getitem_index(user: torch.fx.Node) -> Optional[int]:
    if user.op != "call_function":
        return None
    if user.target not in (operator.getitem, torch.ops.aten.select.int):
        return None
    idx = user.args[1]
    if not isinstance(idx, int):
        return None
    return idx


def _rewrite_associative_scan(gm: torch.fx.GraphModule, node: torch.fx.Node) -> bool:
    if len(node.args) < 2:
        return False

    combine_node = node.args[0]
    xs = node.args[1]
    if combine_node.op != "get_attr":
        return False
    if not isinstance(xs, (list, tuple)) or len(xs) != 2:
        return False
    if any(not isinstance(x, torch.fx.Node) for x in xs):
        return False

    # additional_inputs must be empty for the narrow Mamba pattern
    if len(node.args) > 2 and node.args[2] not in ((), [], None):
        return False

    try:
        combine_gm = gm.get_submodule(combine_node.target)
    except AttributeError:
        combine_gm = getattr(gm, combine_node.target, None)
    if not isinstance(combine_gm, torch.fx.GraphModule):
        return False
    if not _is_mamba_affine_combine(combine_gm):
        logger.debug(
            "associative_scan %s: combine subgraph is not the Mamba affine pattern",
            node.name,
        )
        return False

    scan_len = _static_scan_length(tuple(xs))
    if scan_len is None:
        logger.debug(
            "associative_scan %s: dynamic scan length; leaving HOP for PyTorch fallback",
            node.name,
        )
        return False

    # Validate users before mutating the graph.
    replacements = {}
    for user in list(node.users):
        idx = _getitem_index(user)
        if idx not in (0, 1):
            logger.debug(
                "associative_scan %s: unexpected user %s; leaving HOP alone",
                node.name,
                user,
            )
            return False
        replacements[user] = idx

    a_out, b_out = _hillis_steele_scan(gm, xs[0], xs[1], scan_len, node)
    outs = (a_out, b_out)
    for user, idx in replacements.items():
        user.replace_all_uses_with(outs[idx])
        gm.graph.erase_node(user)
    gm.graph.erase_node(node)
    logger.debug(
        "associative_scan %s: lowered Mamba affine scan (S=%d) to Hillis-Steele aten ops",
        node.name,
        scan_len,
    )
    return True


def lower_associative_scan(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace Mamba-style ``associative_scan`` HOPs with a parallel aten scan."""
    changed = False
    for node in list(gm.graph.nodes):
        if node.op != "call_function" or not _is_associative_scan(node.target):
            continue
        if _rewrite_associative_scan(gm, node):
            changed = True

    if changed:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("After lower_associative_scan:\n%s", gm.graph)

    return gm
