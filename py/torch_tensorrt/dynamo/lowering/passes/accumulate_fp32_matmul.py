import logging

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


def accumulate_fp32_matmul(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace a matmul layer with fp32 accumulation nodes"""
    if settings.use_fp32_acc:
        matmul_targets = [
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
        ]

        matmul_nodes = [
            node for node in gm.graph.nodes if node.target in matmul_targets
        ]
        for matmul_node in matmul_nodes:
            # Prior to the matmul node, insert a cast to the 32-bit float32 node
            node_inputs = matmul_node.all_input_nodes

            for node_input in node_inputs:
                with gm.graph.inserting_before(matmul_node):
                    node_32bit = gm.graph.call_function(
                        torch.ops.aten._to_copy.default,
                        args=(node_input,),
                        kwargs={"dtype": torch.float32},
                    )

                # Replace the input to matmul node with new 32-bit cast node
                matmul_node.replace_input_with(node_input, node_32bit)

            # Add a cast back to original precision
            with gm.graph.inserting_after(matmul_node):
                node_orig_precision = gm.graph.call_function(
                    torch.ops.aten._to_copy.default,
                    args=(matmul_node,),
                    kwargs={"dtype": torch.float16},
                )
                matmul_node.replace_all_uses_with(
                    node_orig_precision, propagate_meta=False
                )
                # This is a hack. replace_all_uses_with isn't working here. It complains node_orig_precision is already being used before created.
                node_orig_precision.replace_input_with(
                    node_orig_precision.all_input_nodes[0], matmul_node
                )

        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after enabling matmul layers to use FP32 accumulation:\n{gm.graph}"
        )
    else:
        logger.debug(
            "Skipping FP32 accumulation for matmul layers as use_fp32_acc is not enabled in the compilation settings"
        )

    return gm
