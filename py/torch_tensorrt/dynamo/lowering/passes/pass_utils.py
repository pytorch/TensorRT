from typing import List

import torch


def clean_up_graph_after_modifications(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Runs dead-code elimination, linting, and recompilation for graph, in-place"""
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return gm


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


def find_complex_nodes(gm: torch.fx.GraphModule):
    complex_nodes = []
    complexNodes = {}
    for node in gm.graph.nodes:
        if is_node_complex(node, complexNodes):
            complex_nodes.append(node)
    return complex_nodes


def is_node_complex(node: torch.fx.Node, complexNodes):
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
                        if eachFakeTensorMeta.dtype in (
                            torch.complex64,
                            torch.complex128,
                        ):
                            complexNodes[node.name] = True
                            return True
                elif arg.meta["val"].dtype in (torch.complex64, torch.complex128):
                    complexNodes[node.name] = True
                    return True
    return False
