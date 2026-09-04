from __future__ import annotations

from typing import Any, Callable, Dict

import torch
from torch._ops import OpOverload
from torch_tensorrt.dynamo.conversion.converter_utils import get_positive_dim


def _has_symbolic_scatter_add_extent(graph_module: torch.fx.GraphModule) -> bool:
    """Return whether scatter_add would require unrolling a symbolic extent."""
    for node in graph_module.graph.nodes:
        if node.target != torch.ops.aten.scatter_add.default or len(node.args) < 4:
            continue
        dim = node.args[1]
        src_node = node.args[3]
        if not isinstance(dim, int) or not isinstance(src_node, torch.fx.Node):
            continue
        src_val = src_node.meta.get("val", src_node.meta.get("example_value"))
        if not isinstance(src_val, torch.Tensor) or not src_val.ndim:
            continue
        if isinstance(src_val.shape[get_positive_dim(dim, src_val.ndim)], torch.SymInt):
            return True

    return False


def filter_decomposition_table(
    decompositions: Dict[OpOverload, Callable[[Any], Any]],
    graph_module: torch.fx.GraphModule,
) -> Dict[OpOverload, Callable[[Any], Any]]:
    """Filter graph-incompatible entries from a decomposition table.

    Decomposition tables are keyed by operator rather than individual FX node,
    so encountering one dynamic ``scatter_add`` conservatively keeps every
    ``scatter_add`` in this graph intact for partitioning to handle.
    """
    filtered_decompositions = dict(decompositions)
    if _has_symbolic_scatter_add_extent(graph_module):
        # The custom decomposition uses a Python range over this extent.
        # Keeping the op lets partitioning fall back to Torch without trying
        # to specialize an unbacked or otherwise dynamic SymInt.
        filtered_decompositions.pop(torch.ops.aten.scatter_add.default, None)

    return filtered_decompositions
