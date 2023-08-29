import logging
import operator

import torch

logger = logging.getLogger(__name__)

BUILTIN_TRANSLATION = {
    operator.ipow: operator.pow,
    operator.imul: operator.mul,
    operator.imatmul: operator.matmul,
    operator.ifloordiv: operator.floordiv,
    operator.itruediv: operator.truediv,
    operator.imod: operator.mod,
    operator.iadd: operator.add,
    operator.isub: operator.sub,
    operator.ilshift: operator.lshift,
    operator.irshift: operator.rshift,
    operator.iand: operator.and_,
    operator.ixor: operator.xor,
    operator.ior: operator.or_,
}


def replace_builtin_inplace_ops(gm: torch.fx.GraphModule) -> None:
    """Replaces inplace builtins from Python's operator class

    Replaces inplace builtins with out-of-place equivalent ops
    """
    for node in gm.graph.nodes:
        # If a node uses one of the inplace builtins
        # Replace it with its out-of-place equivalent
        if node.target in BUILTIN_TRANSLATION:
            out_of_place_op = BUILTIN_TRANSLATION[node.target]

            # Replace inplace operator node and delete
            with gm.graph.inserting_before(node):
                out_of_place = gm.graph.call_function(
                    out_of_place_op,
                    args=node.args,
                    kwargs=node.kwargs,
                )

            logger.debug(f"Replacing {node.target} with {out_of_place.target}")

            node.replace_all_uses_with(out_of_place)
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()
