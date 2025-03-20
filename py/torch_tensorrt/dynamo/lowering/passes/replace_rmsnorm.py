import logging
import operator

import flashinfer
import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

logger = logging.getLogger(__name__)


@torch.library.custom_op("flashinfer::rmsnorm", mutates_args=())  # type: ignore[misc]
def flashinfer_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return flashinfer.norm.rmsnorm(input, weight)


@torch.library.register_fake("flashinfer::rmsnorm")
def _(input: torch.Tensor, weight: torch.Tensor, b: float = 1e-6) -> torch.Tensor:
    return input


# Replace RMSNorm based on the implementation here: https://github.com/huggingface/transformers/blob/51ed61e2f05176f81fa7c9decba10cc28e138f61/src/transformers/models/llama/modeling_llama.py#L70-L74
def replace_rmsnorm(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if (
            node.target == torch.ops.aten._to_copy.default
            and node.kwargs.get("dtype") is torch.float32
            and len(node.users) == 2
        ):
            if (
                list(node.users)[0].target == torch.ops.aten.pow.Tensor_Scalar
                and list(node.users)[1].target == torch.ops.aten.mul.Tensor
            ):
                pow_node = list(node.users)[0]
                if (
                    len(pow_node.users) == 1
                    and list(pow_node.users)[0].target == torch.ops.aten.mean.dim
                ):
                    mean_node = list(pow_node.users)[0]
                    if (
                        len(mean_node.users) == 1
                        and list(mean_node.users)[0].target == torch.ops.aten.add.Tensor
                    ):
                        add_node = list(mean_node.users)[0]
                        if (
                            len(add_node.users) == 1
                            and list(add_node.users)[0].target
                            == torch.ops.aten.sqrt.default
                        ):
                            sqrt_node = list(add_node.users)[0]
                            if (
                                len(sqrt_node.users) == 1
                                and list(sqrt_node.users)[0].target
                                == torch.ops.aten.div.Tensor
                            ):
                                div_node = list(sqrt_node.users)[0]
                                if list(div_node.users)[0] == list(node.users)[1]:
                                    mul_node = list(div_node.users)[0]
                                    copy_node = list(mul_node.users)[0]
                                    weight_mul_node = list(copy_node.users)[0]

                                    weight = weight_mul_node.args[0]

                                    with gm.graph.inserting_after(weight_mul_node):
                                        flashinfer_rmsnorm_node = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.flashinfer.rmsnorm.default,
                                            args=(
                                                node.args[0],
                                                weight,
                                                add_node.args[1],
                                            ),
                                        )

                                    weight_mul_node.replace_all_uses_with(
                                        flashinfer_rmsnorm_node
                                    )
                                    flashinfer_rmsnorm_node.meta.update(
                                        weight_mul_node.meta
                                    )

                                    modified_graph = True

                                    gm.graph.erase_node(weight_mul_node)
                                    gm.graph.erase_node(copy_node)
                                    gm.graph.erase_node(mul_node)
                                    gm.graph.erase_node(div_node)
                                    gm.graph.erase_node(sqrt_node)
                                    gm.graph.erase_node(add_node)
                                    gm.graph.erase_node(mean_node)
                                    gm.graph.erase_node(pow_node)
                                    gm.graph.erase_node(node)

    if modified_graph:
        gm = clean_up_graph_after_modifications(gm)
        logger.debug(
            f"Graph after replacing rmsnorm with flashinfer::rmsnorm:\n{gm.graph}"
        )

    return gm
