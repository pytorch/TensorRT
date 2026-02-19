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

from typing import Any, Callable, Optional, Sequence, Union

import flashinfer
import torch
import torch_tensorrt
from torch._subclasses import FakeTensor
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
                                    hidden_states_node = node.args[0]

                                    original_meta = hidden_states_node.meta.get(
                                        "tensor_meta", {}
                                    )
                                    memory_format = original_meta.memory_format
                                    from torch.fx.experimental.symbolic_shapes import (
                                        ShapeEnv,
                                    )

                                    shape_env = ShapeEnv()

                                    with gm.graph.inserting_after(weight_mul_node):
                                        input_meta = node.args[0].meta["val"]
                                        batch_size = input_meta.shape[0]
                                        seq_len = input_meta.shape[1]
                                        head_dim = input_meta.shape[2]

                                        # Create symbolic ints for batch_size
                                        if isinstance(batch_size, int):
                                            batch_size_unbacked_symint = (
                                                shape_env.create_unbacked_symint()
                                            )
                                            torch._check(
                                                batch_size_unbacked_symint >= batch_size
                                            )
                                            torch._check(
                                                batch_size_unbacked_symint <= batch_size
                                            )
                                        elif isinstance(batch_size, torch.SymInt):
                                            pass
                                        else:
                                            raise ValueError(
                                                "Batch size must be a sym int"
                                            )

                                        # Create symbolic ints for head_dim
                                        if isinstance(head_dim, int):
                                            head_dim_unbacked_symint = (
                                                shape_env.create_unbacked_symint()
                                            )
                                            torch._check(
                                                head_dim_unbacked_symint >= head_dim
                                            )
                                            torch._check(
                                                head_dim_unbacked_symint <= head_dim
                                            )
                                        elif isinstance(head_dim, torch.SymInt):
                                            pass
                                        else:
                                            raise ValueError(
                                                "head_dim must be a sym int"
                                            )

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

                                        batch_size = node.args[0].meta["val"].shape[0]
                                        b.meta["val"] = batch_size_unbacked_symint

                                        s = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.sym_size.int,
                                            args=(node.args[0], 1),
                                        )
                                        s.meta.update(b.meta)
                                        s.meta["val"] = seq_len
                                        d = gm.graph.create_node(
                                            op="call_function",
                                            target=torch.ops.aten.sym_size.int,
                                            args=(node.args[0], 2),
                                        )
                                        d.meta.update(b.meta)
                                        d.meta["val"] = head_dim_unbacked_symint

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
                                                stride=None,
                                                memory_format=memory_format,
                                                is_quantized=False,
                                                qparams={},
                                                requires_grad=False,
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
                                    reshapback_node.meta["tensor_meta"] = (
                                        TensorMetadata(
                                            shape=torch.Size([b_val, s_val, d_val]),
                                            dtype=original_meta.dtype,
                                            stride=None,
                                            memory_format=memory_format,
                                            is_quantized=False,
                                            qparams={},
                                            requires_grad=False,
                                        )
                                    )

                                    # reshapback_node.meta.update(weight_mul_node.meta)
                                    weight_mul_node.replace_all_uses_with(
                                        reshapback_node
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

    return gm


@_aten_lowering_pass
def set_copy_node_meta_data(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    for node in gm.graph.nodes:
        if node.target == torch.ops.aten._to_copy.default and (
            "tensor_meta" not in node.meta
        ):
            input_node = node.args[0]

            # Check if input has metadata
            if "tensor_meta" in input_node.meta:
                # Copy input metadata and update dtype to float32
                output_meta = input_node.meta["tensor_meta"]
                # output_meta.dtype = node.kwargs.get("dtype")

                # # Assign to the _to_copy node
                # node.meta["tensor_meta"] = output_meta
                node.meta["tensor_meta"] = TensorMetadata(
                    shape=output_meta.shape,
                    dtype=node.kwargs.get("dtype"),
                    requires_grad=True,
                    stride=None,
                    memory_format=input_node.meta["tensor_meta"].memory_format,
                    is_quantized=False,
                    qparams={},
                )

            else:
                # Handle missing metadata (optional warning/logging)
                print(f"Warning: Input node {input_node} has no tensor_meta")

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
    model = LlamaForCausalLM(config).cuda().half().eval()

MAX_TOKENS = 64
seq_len = torch.export.Dim("seq_len", min=2, max=MAX_TOKENS)
# 3. Export with static shapes
input_ids = torch.randint(0, 32000, (1, 64))  # Static [batch=1, seq=64]
exported = torch.export.export(
    model,
    (input_ids,),
    dynamic_shapes=({1: seq_len},),
)

# Test forward pass
input_ids = torch.randint(0, 32000, (1, 64))
output = model(input_ids)
print(output)

# Export validation

DEVICE = torch.device("cuda:0")
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    with torch_tensorrt.dynamo.Debugger(
        log_level="info",
        # profile_format="trex",
        # save_engine_profile=True,
        capture_fx_graph_before=["remove_detach"],
        capture_fx_graph_after=["replace_rmsnorm"],
        logging_dir="/home/profile/logging/torchtrt",
        engine_builder_monitor=False,
    ):
        trt_model = torch_tensorrt.dynamo.compile(
            exported,
            inputs=[input_ids],
            enabled_precisions={torch.float32, torch.float16},
            truncate_double=True,
            device=DEVICE,
            disable_tf32=True,
            use_explicit_typing=False,
            use_fp32_acc=True,
            use_python_runtime=True,
        )

    input_ids = input_ids.to(DEVICE)

    res = trt_model.forward(input_ids)

    # Benchmark TensorRT models

    import time

    def benchmark_model(model, input_ids, label, n_runs=100):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                out = model(input_ids)
        torch.cuda.synchronize()
        end = time.time()
        print(f"{label}: {n_runs} runs, total {(end - start):.4f} s")
        return out

    # Warmup
    with torch.no_grad():
        _ = trt_model(input_ids)
        # Benchmark
        trt_out = benchmark_model(trt_model, input_ids, "TensorRT model")

# Compare outputs

pytorch_logits = output.logits
trt_logits = trt_out.logits

pytorch_logits = pytorch_logits.to(DEVICE)
trt_logits = trt_logits.to(DEVICE)
print("Max abs diff:", (pytorch_logits - trt_logits).abs().max().item())
print("Mean abs diff:", (pytorch_logits - trt_logits).abs().mean().item())
