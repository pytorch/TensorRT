"""
.. _dynamo_aten_lowering_passes:

Dynamo ATen Lowering Passes
======================================================

This interactive script is intended as an overview of the process by which ATen lowering passes are written and used."""

# %%
# 1. Lowering Pass Function
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# An ATen lowering pass function in Torch-TRT must satisfy two requirements:
# - The function must take as input a single `torch.fx.GraphModule` and return the lowered
# `torch.fx.GraphModule`
# - The function must leave the graph in a valid and invoke-able state, including performing any
# necessary linting and recompilation
#
# See below for an example of a lowering pass which repairs graphs that have inputs which are
# also outputs, a disallowed configuration for TRT Engines.

# %%
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

    # If the graph was modified, clean up the graph and ensure it is up-to-date
    if modified_graph:
        gm.graph.eliminate_dead_code()
        gm.graph.lint()
        gm.recompile()
        logger.debug(f"Graph after repair_input_as_output:\n{gm.graph}")

    return gm


# %%
# 2. Lowering Pass Registration
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To add a lowering pass, use the convenience function `add_lowering_pass` in the module
# `torch_tensorrt.dynamo.lowering.passes`. See below for an example:

# %%
from torch_tensorrt.dynamo.lowering.passes import add_lowering_pass

# Adds the lowering pass at the end of the pass list
add_lowering_pass(repair_input_as_output)

# Alternatively, specify an index to insert the lowering pass at a specific location
add_lowering_pass(repair_input_as_output, 1)

# To remove a lowering pass, specify the index of the pass to remove:
from torch_tensorrt.dynamo.lowering.passes import remove_lowering_pass

# Removes the lowering pass at index 1
remove_lowering_pass(1)


# To view all lowering passes, in the order they will be run, use the following
from torch_tensorrt.dynamo.lowering.passes import dump_lowering_passes

print(dump_lowering_passes())

# %%
# 3. Apply Available Lowering Passes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# To apply all lowering passes to a graph, the convenience function `apply_lowering_passes` in the module
# `torch_tensorrt.dynamo.lowering.passes` can be used. This function is automatically invoked in the Torch-TRT Dynamo
# paths. Additionally, the graph after each modifying pass is logged in the debug logs for Torch-TRT runs.
