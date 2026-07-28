import operator
from typing import Optional, cast

from torch_tensorrt.dynamo.lowering._SubgraphBuilder import SubgraphBuilder

import torch
from torch.fx import GraphModule, Node

from .pass_utils import clean_up_graph_after_modifications


def _negative_symint_operand(x: object) -> Optional[object]:
    # Return n for symbolic bounds represented as -n. The caller rewrites
    # that bound to dim_size - n, matching Python's negative indexing rules.
    if (
        isinstance(x, Node)
        and x.op == "call_function"
        and x.target in (operator.neg, torch.ops.aten.neg.default)
        and len(x.args) == 1
    ):
        return cast(object, x.args[0])
    return None


def _rank(x: Node) -> Optional[int]:
    val = x.meta.get("val")
    if isinstance(val, torch.Tensor):
        return cast(int, val.dim())
    if hasattr(val, "shape"):
        return len(val.shape)
    return None


def normalize_negative_slice_stop(
    gm: GraphModule, settings: object = None
) -> GraphModule:
    """Normalize negative symbolic slice bounds to positive dim-relative bounds.

    Python slicing accepts negative bounds such as x[-n:] or x[:-n]. TensorRT
    shape expressions need the equivalent positive bound, dim_size - n.
    """
    modified = False

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.slice.Tensor:
            continue

        args = list(node.args)
        if len(args) < 3:
            continue

        input_node, dim = args[:2]
        if not isinstance(input_node, Node) or not isinstance(dim, int):
            continue

        rank = _rank(input_node)
        if rank is not None:
            # Match PyTorch dim normalization for negative dims.
            dim = dim % rank

        rewritten = False
        # aten.slice.Tensor can appear as (input, dim, start) or
        # (input, dim, start, stop, ...). Normalize either symbolic bound.
        for bound_index in (2, 3):
            if len(args) <= bound_index:
                continue

            bound = args[bound_index]
            positive_offset = _negative_symint_operand(bound)
            if positive_offset is None:
                continue

            with SubgraphBuilder(gm.graph, node.prev) as b:
                dim_size = b(torch.ops.aten.sym_size.int, input_node, dim)
                # A negative symbolic bound -n becomes dim_size - n.
                normalized_bound = b(operator.sub, dim_size, positive_offset)

            args[bound_index] = normalized_bound
            rewritten = True

        if rewritten:
            args[1] = dim
            node.args = tuple(args)
            modified = True

    return clean_up_graph_after_modifications(gm) if modified else gm
