import logging

import torch

logger = logging.getLogger(__name__)


def repair_input_as_output(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    """Repair scenarios where inputs are also outputs of the graph

    TRT does not allow such cases, so we insert a clone (identity) layer
    """
    modified_graph = False

    # Extract graph placeholder Tensors
    placeholders = [
        node
        for node in gm.graph.nodes
        if (
            node.op == "placeholder"
            and isinstance(node.type, type)
            and issubclass(node.type, torch.Tensor)
        )
    ]

    for placeholder in placeholders:
        # If any placeholder has any users which are direct graph outputs
        if len(placeholder.users) >= 1 and any(
            user.op == "output" for user in placeholder.users
        ):
            modified_graph = True

            # Get direct graph outputs which are direct uses of placeholders
            direct_outputs = [user for user in placeholder.users if user.op == "output"]

            # Insert clone node for placeholder to ensure placeholder is not a direct output
            with gm.graph.inserting_after(placeholder):
                cloned_placeholder = gm.graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(placeholder,),
                )

            # Replace placeholder as output with cloned version
            for output in direct_outputs:
                output.replace_input_with(placeholder, cloned_placeholder)

    if modified_graph:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        logger.debug(f"Graph after repair_input_as_output:\n{gm.graph}")

    return gm
