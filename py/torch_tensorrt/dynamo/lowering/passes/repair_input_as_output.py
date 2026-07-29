import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
    get_tensor_placeholders,
)

logger = logging.getLogger(__name__)


def repair_input_as_output(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Repair scenarios where inputs are also outputs of the graph

    TRT does not allow such cases, so we insert a clone (identity) layer
    """
    modified_graph = False

    # Extract graph placeholder Tensors. Constant folding also turns
    # input-independent tensor factories (for example torch.zeros) into
    # registered `_frozen_param*` attributes. When such a value is returned
    # across a graph break, eager code may legally mutate it in-place. Return
    # a clone so the registered constant itself remains immutable.
    placeholders = get_tensor_placeholders(gm)
    folded_constants = [
        node
        for node in gm.graph.nodes
        if node.op == "get_attr" and str(node.target).startswith("_frozen_param")
    ]

    for source in [*placeholders, *folded_constants]:
        # If any source has any users which are direct graph outputs
        if len(source.users) >= 1 and any(
            user.op == "output" for user in source.users
        ):
            modified_graph = True

            # Get graph outputs which directly use the source
            direct_outputs = [
                user for user in source.users if user.op == "output"
            ]

            # Insert a clone so the source is not returned directly
            insertion_point = (
                placeholders[-1] if source.op == "placeholder" else source
            )
            with gm.graph.inserting_after(insertion_point):
                cloned_source = gm.graph.call_function(
                    torch.ops.aten.clone.default,
                    args=(source,),
                )

            # Replace the direct output with the cloned version
            for output in direct_outputs:
                output.replace_input_with(source, cloned_source)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(f"Graph after repair_input_as_output:\n{gm.graph}")

    return gm
