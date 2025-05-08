"""
.._llama2_flashinfer_rmsnorm:

Automatically generate a TensorRT Plugin for RMSNorm module and apply it in Llama2
===================================================================

This example showcases how to optimize inference for a LLaMA2 model by replacing its RMSNorm layers with FlashInfer's high-performance implementation. It demonstrates the use of Torch-TensorRT's automatic plugin feature, which dynamically generates and integrates custom TensorRT plugins during compilation.

Key features:
- Leverages automatic plugin registration for FlashInfer RMSNorm ops.
- Applies a custom TorchDynamo lowering pass to replace standard RMSNorm ops.
- Compiles the modified model using Torch-TensorRT's Dynamo path.
- Benchmarks inference performance with and without FlashInfer RMSNorm.

This example illustrates advanced extensibility in Torch-TensorRT through automatic plugin generation and operator lowering customization.
"""

from typing import Callable, Optional, Sequence, Union

import flashinfer
import torch
import torch_tensorrt
from torch.fx.passes.shape_prop import TensorMetadata
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

                                    original_meta = weight_mul_node.meta.get(
                                        "tensor_meta", {}
                                    )
                                    memory_format = original_meta.memory_format

                                    with gm.graph.inserting_after(weight_mul_node):
                                        b = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.sym_size.int,
                                            args=(node.args[0], 0),
                                        )
                                        b.meta["tensor_meta"] = TensorMetadata(
                                            shape=torch.Size([1]),
                                            dtype=torch.int64,
                                            requires_grad=False,
                                            stride=None,
                                            memory_format=memory_format,
                                            is_quantized=False,
                                            qparams={},
                                        )
                                        s = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.sym_size.int,
                                            args=(node.args[0], 1),
                                        )
                                        s.meta.update(b.meta)

                                        d = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.sym_size.int,
                                            args=(node.args[0], 2),
                                        )
                                        d.meta.update(b.meta)

                                    with gm.graph.inserting_after(b):
                                        new_first_dim = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.mul.Scalar,
                                            args=(b, s),
                                        )
                                        new_first_dim.meta.update(b.meta)

                                    with gm.graph.inserting_after(new_first_dim):
                                        # with gm.graph.inserting_after(weight_mul_node):
                                        reshape_node = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.reshape.default,
                                            args=(node.args[0], [new_first_dim, d]),
                                        )
                                        b_val = original_meta.shape[0]
                                        s_val = original_meta.shape[1]
                                        d_val = original_meta.shape[2]

                                        reshape_node.meta["tensor_meta"] = (
                                            TensorMetadata(
                                                shape=torch.Size(
                                                    [b_val * s_val, d_val]
                                                ),
                                                dtype=original_meta.dtype,
                                                requires_grad=True,
                                                stride=None,
                                                memory_format=memory_format,
                                                is_quantized=False,
                                                qparams={},
                                            )
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
                                        flashinfer_rmsnorm_node.meta.update(
                                            reshape_node.meta
                                        )

                                    with gm.graph.inserting_after(
                                        flashinfer_rmsnorm_node
                                    ):
                                        reshapback_node = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.reshape.default,
                                            args=(
                                                flashinfer_rmsnorm_node,
                                                [b, s, d],
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
print(output)

# Export validation

DEVICE = torch.device("cuda:0")

with torch_tensorrt.logging.errors():
    trt_model = torch_tensorrt.dynamo.compile(
        exported,
        inputs=[input_ids],
        enabled_precisions={torch.float32, torch.float16},
        truncate_double=True,
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=False,
        use_fp32_acc=True,
        # debug=True,
    )

input_ids = input_ids.to(DEVICE)

res = trt_model.forward(input_ids)
print(res)
