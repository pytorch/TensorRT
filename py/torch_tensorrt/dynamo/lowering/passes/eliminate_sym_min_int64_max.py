import sys

import torch
from torch.fx import GraphModule, Node

from .pass_utils import clean_up_graph_after_modifications


_INT64_MAX = 2**63 - 1
_SYM_MIN = getattr(torch, "sym_min", None)


def _is_int64_max(x: object) -> bool:
    return isinstance(x, int) and x in (sys.maxsize, _INT64_MAX)


def eliminate_sym_min_int64_max(
    gm: GraphModule, settings: object = None
) -> GraphModule:
    """Remove no-op sym_min nodes where one operand is INT64_MAX.

    torch.export may emit sym_min(sym, INT64_MAX) for an effectively unbounded
    symbolic value. That expression is equivalent to sym, and leaving it in the
    graph can produce runtime calls to torch.sym_min with Tensor inputs.
    """
    if _SYM_MIN is None:
        return gm

    modified = False
    for node in list(gm.graph.nodes):
        if (
            node.op != "call_function"
            or node.target is not _SYM_MIN
            or len(node.args) < 2
        ):
            continue

        lhs, rhs = node.args[:2]
        if _is_int64_max(rhs) and isinstance(lhs, Node):
            passthrough = lhs
        elif _is_int64_max(lhs) and isinstance(rhs, Node):
            passthrough = rhs
        else:
            continue

        node.replace_all_uses_with(passthrough)
        gm.graph.erase_node(node)
        modified = True

    return clean_up_graph_after_modifications(gm) if modified else gm
