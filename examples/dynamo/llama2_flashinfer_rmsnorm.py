from typing import Callable, Optional, Sequence, Union

import flashinfer
import torch
import torch_tensorrt
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)
from transformers import LlamaConfig, LlamaForCausalLM


@torch.library.custom_op("flashinfer::rmsnorm", mutates_args=())  # type: ignore[misc]
def flashinfer_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return flashinfer.norm.rmsnorm(input, weight)


@torch.library.register_fake("flashinfer::rmsnorm")
def _(input: torch.Tensor, weight: torch.Tensor, b: float = 1e-6) -> torch.Tensor:
    return input


torch_tensorrt.dynamo.conversion.plugins.custom_op(
    "flashinfer::rmsnorm", supports_dynamic_shapes=True
)


@_aten_lowering_pass
def replace_rmsnorm(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
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
                                        reshape_node = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.reshape.default,
                                            args=(node.args[0], [64, 4096]),
                                        )

                                    with gm.graph.inserting_after(reshape_node):
                                        flashinfer_rmsnorm_node = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.flashinfer.rmsnorm.default,
                                            args=(
                                                reshape_node,
                                                weight,
                                                add_node.args[1],
                                            ),
                                        )

                                    with gm.graph.inserting_after(
                                        flashinfer_rmsnorm_node
                                    ):
                                        reshapback_node = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.reshape.default,
                                            args=(
                                                flashinfer_rmsnorm_node,
                                                [1, 64, 4096],
                                            ),
                                        )

                                    weight_mul_node.replace_all_uses_with(
                                        reshapback_node
                                    )
                                    reshapback_node.meta.update(weight_mul_node.meta)

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

    return gm


# 1. Create a custom config with 1 layer
config = LlamaConfig(
    vocab_size=32000,
    hidden_size=4096,  # LLaMA2-7B dimensions
    intermediate_size=11008,  # FFN hidden_dim = 4 * 4096 * 0.7 (SwiGLU scaling)
    num_hidden_layers=1,  # Only 1 decoder layer
    num_attention_heads=32,
    max_position_embeddings=4096,
    use_cache=False,  # Disable KV caching for export
)

# 2. Initialize model (random weights)
with torch.no_grad():
    model = LlamaForCausalLM(config).eval().half()

# 3. Export with static shapes
input_ids = torch.randint(0, 32000, (1, 64))  # Static [batch=1, seq=64]
exported = torch.export.export(
    model,
    (input_ids,),
    dynamic_shapes=None,  # Fully static
)

# Test forward pass
input_ids = torch.randint(0, 32000, (1, 64))
output = model(input_ids)
print(output)  # Should be [1, 64, 32000]

# Export validation
# print(exported.graph_module.code)

DEVICE = torch.device("cuda:0")

with torch_tensorrt.logging.debug():
    trt_model = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[input_ids],
        enabled_precisions={torch.float32, torch.float16},
        truncate_double=True,
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=False,
        use_fp32_acc=True,
        debug=True,
    )

input_ids = input_ids.to(DEVICE)

res = trt_model.forward(input_ids)
print(res)
