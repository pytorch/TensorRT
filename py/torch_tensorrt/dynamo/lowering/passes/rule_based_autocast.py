import logging
import operator
from typing import Any

import torch
from torch._export.passes.replace_autocast_with_hop_pass import (
    replace_autocast_with_hop_pass,
)
from torch_tensorrt._enums import dtype
from torch_tensorrt.dynamo._settings import CompilationSettings

from .nodeclassifier import NodeClassifier
from .pass_utils import clean_up_graph_after_modifications

logger = logging.getLogger(__name__)


def is_tensor_node(n: torch.fx.Node) -> bool:
    val = n.meta.get("val", None)
    if hasattr(val, "dtype"):
        return True
    return False


def rule_based_autocast(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Rule-based autocast"""
    if settings.use_explicit_typing:
        logger.debug("Strong typing is enabled, skipping rule-based autocast.")
        return gm

    # nodes = list(gm.graph.nodes)
    # # insert enter autocast node in the beginning of the graph
    # with gm.graph.inserting_before(nodes[0]):
    #     enter_autocast_node = gm.graph.call_function(torch.amp.autocast_mode._enter_autocast, args=("cuda", torch.float16, True, True))
    #     enter_autocast_node.meta.update(getattr(nodes[0], "meta", {}))

    # # insert exit autocast node before the return node, assuming the return node is the last node
    # with gm.graph.inserting_before(nodes[-1]):
    #     exit_autocast_node = gm.graph.call_function(torch.amp.autocast_mode._exit_autocast, args=(enter_autocast_node,))
    #     exit_autocast_node.meta.update(getattr(nodes[-1], "meta", {}))

    # gm = clean_up_graph_after_modifications(gm)
    # gm, new_signature = replace_autocast_with_hop_pass(gm, None)
    # logger.debug("Graph after replace_autocast_with_hop_pass:\n%s", gm.graph)

    # get config from settings
    low_precision_type = settings.low_precision_type
    if low_precision_type is None:
        return gm
    if isinstance(low_precision_type, dtype):
        low_precision_type = low_precision_type.to(torch.dtype)
    high_precision_type = torch.float32
    nodes_to_exclude = settings.nodes_to_exclude
    targets_to_exclude = settings.targets_to_exclude
    data_max = settings.data_max
    max_depth_of_reduction = settings.max_depth_of_reduction
    reference_data: dict[str, torch.Tensor] = settings.intermediate_node_outputs

    node_classifier = NodeClassifier(
        gm.graph.nodes,
        nodes_to_exclude=nodes_to_exclude,
        targets_to_exclude=targets_to_exclude,
        data_max=data_max,
        max_depth_of_reduction=max_depth_of_reduction,
    )
    low_precision_nodes, high_precision_nodes = node_classifier.run(reference_data)

    for node in list(gm.graph.nodes):
        if node.op == "call_function":
            if (
                node.target == torch.ops.higher_order.wrap_with_autocast
                or node.target == operator.getitem
            ):
                continue

            def _cast_all_tensor_args_to_dtype(arg: Any, dtype: torch.dtype) -> Any:
                """Cast all tensor args to the given dtype

                Args:
                    arg: The argument to cast
                    dtype: The dtype to cast to

                Returns:
                    The casted argument
                """
                if isinstance(arg, torch.fx.Node) and is_tensor_node(arg):
                    val = arg.meta.get("val", None)
                    with gm.graph.inserting_before(node):
                        cast = gm.graph.call_function(
                            torch.ops.aten.to.dtype, args=(arg, dtype)
                        )

                    if isinstance(val, torch.Tensor):
                        arg.meta["val"] = val.to(dtype)
                    cast.meta.update(arg.meta)
                    return cast
                elif isinstance(arg, (tuple, list)):
                    return type(arg)(
                        _cast_all_tensor_args_to_dtype(a, dtype) for a in arg
                    )
                elif isinstance(arg, dict):
                    return {
                        k: _cast_all_tensor_args_to_dtype(v, dtype)
                        for k, v in arg.items()
                    }
                else:
                    return arg

            if node.name in low_precision_nodes:
                node.args = _cast_all_tensor_args_to_dtype(
                    node.args, low_precision_type
                )
                node.kwargs = _cast_all_tensor_args_to_dtype(
                    node.kwargs, low_precision_type
                )
            elif node.name in high_precision_nodes:
                node.args = _cast_all_tensor_args_to_dtype(
                    node.args, high_precision_type
                )
                node.kwargs = _cast_all_tensor_args_to_dtype(
                    node.kwargs, high_precision_type
                )

    gm = clean_up_graph_after_modifications(gm)
    logger.debug("Graph after Autocast based on the rules:\n%s", gm.graph)

    return gm
