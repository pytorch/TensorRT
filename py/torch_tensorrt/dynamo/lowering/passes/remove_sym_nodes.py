import logging
from typing import Any, Sequence

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings

logger = logging.getLogger(__name__)


def remove_sym_nodes(
    gm: torch.fx.GraphModule,
    sample_inputs: Sequence[Any],
    settings: CompilationSettings,
) -> torch.fx.GraphModule:
    """Remove sym_int placeholders which get inserted due to torch.compile's
    dynamic=True behavior
    """
    gm = replace_symint_with_sym_size(gm)

    # Extract SymInt placeholder Tensors
    placeholder_idx_sym_ints = [
        (idx, node)
        for idx, node in enumerate(gm.graph.nodes)
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.SymInt)
            and not node.users
        )
    ]

    for idx, node in placeholder_idx_sym_ints:
        gm.graph.erase_node(node)
        sample_inputs.pop(idx)

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Removed SymInt placeholders:\n{gm.graph}")

    return gm


def replace_symint_with_sym_size(
    gm: torch.fx.GraphModule,
) -> torch.fx.GraphModule:
    """Replace SymInt placeholders with sym_size nodes"""
    # Find all SymInt placeholders and their args
    symint_node_arg_dict = {}
    for node in gm.graph.nodes:
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.SymInt)
        ):
            ga = node.meta.get("grapharg", None)
            if ga is not None:
                src = ga.source  # TensorPropertySource
                symint_node_arg_dict[node] = (src.base.local_name, src.idx)

    # Replace SymInt placeholders with sym_size nodes
    for node in gm.graph.nodes:
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.Tensor)
        ):
            ga = node.meta.get("grapharg", None)
            if ga is not None:
                src = ga.source
                if hasattr(src, "local_name") and getattr(src, "is_input", False):
                    node_local_name = src.local_name
                    for symint_node, (
                        symint_local_name,
                        idx,
                    ) in symint_node_arg_dict.items():
                        if node_local_name == symint_local_name:
                            with gm.graph.inserting_after(node):
                                size_node = gm.graph.call_function(
                                    torch.ops.aten.sym_size, args=(node, idx)
                                )
                            symint_node.replace_all_uses_with(size_node)
                            logger.debug(
                                f"The SymInt node {symint_node} is replaced with the sym_size node {size_node}"
                            )
                            # the symint_node is not used anymore, but it cannot be directly erased here
                            # because it will cause the number of positional arguments mismatch error.
                            # The node will be removed in the outside of the function

    gm.graph.lint()
    gm.recompile()
    logger.debug(f"Added sym_size nodes for SymInt placeholders:\n{gm.graph}")

    return gm
