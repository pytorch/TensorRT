from typing import Any, Dict, List, Sequence

import torch
from torch_tensorrt.dynamo.utils import COMPLEX_DTYPES

# When True, clean_up_graph_after_modifications only marks the graph dirty and
# skips DCE/lint/recompile. post_lowering enables this around the pass list so
# multiple cheap FX repairs share one cleanup cycle (Naren: "one iteration").
_defer_graph_cleanup: bool = False
_graph_cleanup_pending: bool = False


def set_defer_graph_cleanup(enabled: bool) -> None:
    """Enable/disable deferred FX cleanup for the current post_lowering run."""
    global _defer_graph_cleanup, _graph_cleanup_pending
    _defer_graph_cleanup = enabled
    if not enabled:
        _graph_cleanup_pending = False


def flush_deferred_graph_cleanup(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Run a real cleanup if any deferred clean_up call happened."""
    global _graph_cleanup_pending
    if _graph_cleanup_pending:
        _graph_cleanup_pending = False
        return _run_graph_cleanup(gm)
    return gm


def _run_graph_cleanup(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Runs dead-code elimination, linting, and recompilation for graph, in-place"""
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


def clean_up_graph_after_modifications(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Runs dead-code elimination, linting, and recompilation for graph, in-place.

    If deferred cleanup is enabled (see ``set_defer_graph_cleanup``), records that
    a cleanup is needed and returns immediately so callers can batch mutations.
    """
    global _graph_cleanup_pending
    if _defer_graph_cleanup:
        _graph_cleanup_pending = True
        return gm
    return _run_graph_cleanup(gm)


def get_tensor_placeholders(
    gm: torch.fx.GraphModule,
) -> List[torch.fx.Node]:
    """Returns placeholder nodes of GraphModule which are torch.Tensor types"""
    # Tensor placeholders must be subclasses of torch.Tensor
    placeholders = [
        node
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.Tensor)
        )
    ]

    return placeholders


def find_complex_nodes(gm: torch.fx.GraphModule) -> List[torch.fx.Node]:
    complex_nodes: List[torch.fx.Node] = []
    complexNodes: Dict[str, bool] = {}
    for node in gm.graph.nodes:
        if is_node_complex(node, complexNodes):
            complex_nodes.append(node)
    return complex_nodes


def is_node_complex(node: Any, complexNodes: Dict[str, bool]) -> bool:
    if not isinstance(node, torch.fx.Node):
        return False
    if node.name in complexNodes:
        return True
    if node.op == "call_function" and node.args is not None:
        for arg in node.args:
            if isinstance(arg, int):
                continue
            elif isinstance(arg, (list, tuple)):
                for eachNode in arg:
                    if is_node_complex(eachNode, complexNodes):
                        complexNodes[node.name] = True
                        return True

            elif hasattr(arg, "meta") and "val" in arg.meta:
                if isinstance(arg.meta["val"], (list, tuple)):
                    for eachFakeTensorMeta in arg.meta["val"]:
                        if eachFakeTensorMeta.dtype in COMPLEX_DTYPES:
                            complexNodes[node.name] = True
                            return True
                elif arg.meta["val"].dtype in COMPLEX_DTYPES:
                    complexNodes[node.name] = True
                    return True
    return False


def trace_intermediate_node_outputs(
    gm: torch.fx.GraphModule,
    calibration_dataloader: torch.utils.data.DataLoader,
    excluded_ops: Sequence[torch.fx.node.Target] = [],
) -> Dict[str, torch.Tensor]:
    """Trace the intermediate node outputs of a graph module.

    Args:
        gm (torch.fx.GraphModule): The graph module to trace the intermediate node outputs of.
        calibration_dataloader (torch.utils.data.DataLoader): The dataloader to use for tracing.
        excluded_ops (Set[torch.fx.node.Target]): The set of ATen ops that should be excluded from the trace. For example, `{torch.ops.higher_order.wrap_with_autocast, operator.getitem}`. Default is an empty set.

    Returns:
        Dict[str, torch.Tensor]: A dictionary of intermediate node outputs. The key is the node name and the value is the tensor.
    """

    intermediate_node_outputs: Dict[str, torch.Tensor] = {}

    class IntermediateNodeTracer(torch.fx.Interpreter):  # type: ignore[misc]
        def run_node(self, n: torch.fx.Node) -> Any:
            out = super().run_node(n)
            if n.op == "call_function" and n.target not in excluded_ops:
                if not isinstance(out, torch.Tensor):
                    return out
                if n.name in intermediate_node_outputs:
                    intermediate_node_outputs[n.name] = torch.cat(
                        [intermediate_node_outputs[n.name], out], dim=0
                    )
                else:
                    intermediate_node_outputs[n.name] = out
            return out

    if calibration_dataloader is not None:
        tracer = IntermediateNodeTracer(gm)
        for batch in calibration_dataloader:
            tracer.run(tuple(batch))
    return intermediate_node_outputs
