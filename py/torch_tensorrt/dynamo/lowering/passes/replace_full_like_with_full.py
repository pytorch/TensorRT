import logging

import torch
import torch.fx
from torch_tensorrt.dynamo._defaults import default_device
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from torch_tensorrt.dynamo.utils import to_torch_device

logger = logging.getLogger(__name__)


def replace_full_like_with_full(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace full_like nodes with equivalent full nodes"""
    modified_graph = False

    for node in gm.graph.nodes:
        if node.target == torch.ops.aten.full_like.default:
            modified_graph = True

            # Extract arguments from full_like
            input_tensor = node.args[0]
            fill_value = node.args[1]
            input_dtype = None
            input_shape = None
            input_device = to_torch_device(default_device())
            if "val" in input_tensor.meta:
                input_dtype = input_tensor.meta["val"].dtype
                input_device = input_tensor.meta["val"].device
                input_shape = list(input_tensor.meta["val"].shape)
            elif "tensor_meta" in input_tensor.meta:
                input_dtype = input_tensor.meta["tensor_meta"].dtype
                input_shape = list(input_tensor.meta["tensor_meta"].shape)

            # There's no memory format argument for torch.full.
            # Set the input_device and dtype correspondingly.
            new_kwargs = {}
            for key, val in node.kwargs.items():
                if key != "memory_format":
                    new_kwargs[key] = val
            new_kwargs["device"] = input_device
            new_kwargs["dtype"] = input_dtype
            # Replace full_like with full, using the shape as a list
            input_nodes = (input_shape, fill_value)
            with gm.graph.inserting_after(node):
                full_node = gm.graph.call_function(
                    torch.ops.aten.full.default,
                    args=input_nodes,
                    kwargs=new_kwargs,
                )
                full_node.meta = node.meta

            node.replace_all_uses_with(full_node)
            gm.graph.erase_node(node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)

    return gm
