import logging
import operator

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)

# Replace RMSNorm based on the implementation here: https://github.com/huggingface/transformers/blob/51ed61e2f05176f81fa7c9decba10cc28e138f61/src/transformers/models/llama/modeling_llama.py#L70-L74
def replace_rmsnorm(gm: torch.fx.GraphModule, settings: CompilationSettings) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if (
            node.target == torch.ops.aten._to_copy.default
            and node.kwargs.get("dtype") is torch.float32
            and len(node.users) == 2
        ):
            if (list(node.users)[0].target == torch.ops.aten.pow.Tensor_Scalar and list(node.users)[1].target == torch.ops.aten.mul.Tensor): 
                pow_node = list(node.users)[0]
                if (len(pow_node.users) == 1 and list(pow_node.users)[0].target == torch.ops.aten.mean.dim):
                    mean_node = list(pow_node.users)[0]
                    if (len(mean_node.users) == 1 and list(mean_node.users)[0].target == torch.ops.aten.add.Tensor):
                        add_node = list(mean_node.users)[0]
                        if (len(add_node.users) == 1 and list(add_node.users)[0].target == torch.ops.aten.sqrt.default):
                            sqrt_node = list(add_node.users)[0]
                            if (len(sqrt_node.users) == 1 and list(sqrt_node.users)[0].target == torch.ops.aten.div.Tensor):
                                div_node = list(sqrt_node.users)[0]
                                if (list(div_node.users)[0] == list(node.users)[1]):
                                    print(" good match!")

        

    return gm
        