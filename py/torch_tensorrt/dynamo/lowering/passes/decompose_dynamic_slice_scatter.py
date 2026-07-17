import logging
from typing import Any, Optional

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def decompose_dynamic_slice_scatter(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Lower ``aten.slice_scatter`` to ``arange + view + expand + scatter`` when
    any of ``start``/``end``/``step`` is dynamic (an fx.Node holding a SymInt).

    The slice_scatter converter requires Python ``int`` for start/end/step
    (both the KV-cache fast path and the static-indices fallback). When the
    exported graph passes a SymInt (e.g. KV write position derived from the
    current sequence length), this pass rewrites the op into an equivalent
    aten subgraph whose components all have dynamic-shape converters.

    Static cases (start/end/step are all Python ints) are left untouched so
    the converter can still pick the KV fast path or its static fallback.
    """
    target = torch.ops.aten.slice_scatter.default
    changed = False

    for node in list(gm.graph.nodes):
        if node.op != "call_function" or node.target != target:
            continue

        args = node.args
        input_node = args[0]
        src_node = args[1]
        dim = args[2]
        start: Optional[Any] = args[3] if len(args) > 3 else None
        end: Optional[Any] = args[4] if len(args) > 4 else None
        step: Optional[Any] = args[5] if len(args) > 5 else None

        is_dynamic = any(isinstance(x, torch.fx.Node) for x in (start, end, step))
        if not is_dynamic:
            continue

        input_val = input_node.meta.get("val")
        if input_val is None:
            logger.debug(
                "decompose_dynamic_slice_scatter: %s has no meta['val']; skipping",
                node,
            )
            continue
        rank = input_val.dim()

        if start is None:
            start = 0
        if step is None:
            step = 1

        with gm.graph.inserting_before(node):

            src_val = src_node.meta.get("val")

            def src_size(i: int) -> Any:
                size = src_val.shape[i] if src_val is not None else None
                if isinstance(size, int):
                    return size
                return gm.graph.call_function(
                    torch.ops.aten.sym_size.int, (src_node, i)
                )

            offsets_node = gm.graph.call_function(
                torch.ops.aten.arange.start_step,
                (0, src_size(dim), 1),
                {
                    "dtype": torch.int64,
                    "device": input_val.device,
                },
            )
            scaled_offsets_node = gm.graph.call_function(
                torch.ops.aten.mul.Tensor, (offsets_node, step)
            )
            indices_node = gm.graph.call_function(
                torch.ops.aten.add.Tensor, (scaled_offsets_node, start)
            )

            view_shape = [-1 if i == dim else 1 for i in range(rank)]
            view_node = gm.graph.call_function(
                torch.ops.aten.view.default,
                (indices_node, view_shape),
            )

            expand_size = [src_size(i) for i in range(rank)]
            expand_node = gm.graph.call_function(
                torch.ops.aten.expand.default,
                (view_node, expand_size),
            )

            scatter_node = gm.graph.call_function(
                torch.ops.aten.scatter.src,
                (input_node, dim, expand_node, src_node),
            )

        node.replace_all_uses_with(scatter_node)
        gm.graph.erase_node(node)
        changed = True
        logger.debug(
            "decompose_dynamic_slice_scatter: rewrote %s to arange+scatter (dim=%s)",
            node,
            dim,
        )

    if changed:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug("After decompose_dynamic_slice_scatter:\n%s", gm.graph)

    return gm
