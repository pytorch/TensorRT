import torch


def dynamic_unsupported(node: torch.fx.Node) -> bool:
    # Validate that none of the inputs to the node have Dynamic shapes
    assert isinstance(
        node, torch.fx.Node
    ), "Inputs to validator functions must be FX Nodes"

    # Check node value itself
    if getattr(node.meta["val"], "_has_symbolic_sizes_strides", False):
        return False

    # Check node arguments individually
    if any(
        getattr(arg.meta["val"], "_has_symbolic_sizes_strides", False)
        for arg in node.args
        if isinstance(arg, torch.fx.Node)
    ):
        return False

    # Check node keyword arguments individually
    if any(
        getattr(kwarg.meta["val"], "_has_symbolic_sizes_strides", False)
        for kwarg in node.kwargs.values()
        if isinstance(kwarg, torch.fx.Node)
    ):
        return False

    return True
