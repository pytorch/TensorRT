"""
.. _torch_export_llama2:

Compiling Llama2 using the dynamo backend
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular Llama2 model.
"""

import copy
import random
import timeit
from typing import Callable, Optional, Sequence, Union, cast

import flashinfer
import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch.fx.passes.shape_prop import TensorMetadata

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)
from utils import export_llm, generate

import torch_tensorrt
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)


@torch.library.custom_op("flashinfer::rmsnorm", mutates_args=())  # type: ignore[misc]
def flashinfer_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    return flashinfer.norm.rmsnorm(input, weight)


@torch.library.register_fake("flashinfer::rmsnorm")
def _(input: torch.Tensor, weight: torch.Tensor, b: float = 1e-6) -> torch.Tensor:
    return input


# torch_tensorrt.dynamo.conversion.plugins.custom_op(
#     "flashinfer::rmsnorm", supports_dynamic_shapes=True
# )


@_aten_lowering_pass
def replace_rmsnorm(
    gm: torch.fx.GraphModule, sample_inputs: Sequence[torch.Tensor]
) -> torch.fx.GraphModule:
    print("before2\n")
    print(gm.graph)
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


def recordStats(backend, timings, precision, batch_size=1, compile_time_s=None):
    """
    Records different timing stats and adds it to the result
    """
    times = np.array(timings)
    speeds = batch_size / times
    time_mean = np.mean(times)
    time_med = np.median(times)
    time_99th = np.percentile(times, 99)
    time_std = np.std(times, ddof=0)
    speed_mean = np.mean(speeds)
    speed_med = np.median(speeds)

    stats = {
        "Backend": backend,
        "Precision": precision,
        "Batch size": batch_size,
        "Median(FPS)": speed_med,
        "Mean(FPS)": speed_mean,
        "Median-Latency(ms)": time_med * 1000,
        "Mean-Latency(ms)": time_mean * 1000,
        "Latency-StdDev(ms)": time_std * 1000,
        "Compile Time(s)": compile_time_s,
    }
    return stats


# %%
# Define the parameters and initialize the model
MAX_TOKENS = 100
DEVICE = torch.device("cuda:0")

# Define the Llama2 model from hugging face
# kv_cache is not supported in Torch-TRT currently.
# CPU is used here so that GPU memory is reserved for TRT compilation.
# llama_path = "meta-llama/Llama-2-7b-chat-hf"
# llama_path = "meta-llama/Llama-3.1-8B-Instruct"
llama_path = "meta-llama/Llama-3.2-1B"
with torch.no_grad():
    model = (
        AutoModelForCausalLM.from_pretrained(
            llama_path,
            use_cache=False,
            attn_implementation="eager",
            num_hidden_layers=2,
        )
        .eval()
        .half()
        .cuda()
    )

tokenizer = AutoTokenizer.from_pretrained(llama_path)


# %%
# Tokenize a sample input prompt and get pytorch model outputs
prompt = "What is dynamic programming?"
model_inputs = tokenizer(prompt, return_tensors="pt")


input_ids = model_inputs.input_ids.to(DEVICE)

# Auto-regressive generation loop for greedy decoding using PyTorch model
# We use a custom generate function which is very similar to the huggingface one.
# pyt_gen_tokens = generate(model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# %%
# Compilation with `Torch-TensorRT` using dynamo backend and generate TensorRT outputs
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# Export the llama2 model into an ExportedProgram which is input of TRT compilation
# To compile the model in FP16, we do the following
# 1) Cast the model to FP16 via model.half()
# 2) Enable use_explicit_typing=True. Certain layers are explicitly casted to FP32 within the pytorch model and this flag respects this behavior during TRT compilation
# 3) Enable use_fp32_acc=True. This ensures all the matmuls are accumulated in FP32 precision (similar to PyTorch)
# llama2_ep = export_llm(model, input_ids, max_seq_len=64)
# trt_model = torch_tensorrt.dynamo.compile(
#     llama2_ep,
#     inputs=[input_ids],
#     enabled_precisions={torch.float16},
#     truncate_double=True,
#     device=DEVICE,
#     disable_tf32=True,
#     use_explicit_typing=False,
#     use_fp32_acc=True
#     # debug=True,
# )

# # Auto-regressive generation loop for greedy decoding using TensorRT model
# # We use a custom generate function which is very similar to the huggingface one.
# # Move inputs to GPU
# input_ids = input_ids.to(DEVICE)
# trt_gen_tokens = generate(trt_model, input_ids, MAX_TOKENS, tokenizer.eos_token_id)

# %%
# Decode the output sentences of PyTorch and TensorRT
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# print("=============================")
# print(
#     "Pytorch model generated text: ",
#     tokenizer.batch_decode(
#         pyt_gen_tokens[0], skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )[0],
# )
# print("=============================")
# print(
#     "TensorRT model generated text: ",
#     tokenizer.batch_decode(
#         trt_gen_tokens,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )[0],
# )


# Prompt : What is dynamic programming?

# =============================
# Pytorch model generated text: Dynamic programming is an algorithmic technique used to solve complex problems by breaking them down into smaller subproblems, solving each subproblem only once, and

# =============================
# TensorRT model generated text: Dynamic programming is an algorithmic technique used to solve complex problems by breaking them down into smaller subproblems, solving each subproblem only once, and


def time_generate(
    generate_fn, model, inputs, output_seq_length, eos_token_id, csi=None, iterations=10
):
    """
    Measure the time for generating a sentence over certain number of iterations
    """

    timings = []
    for _ in range(iterations):
        start_time = timeit.default_timer()
        inputs_copy = copy.copy(inputs)
        _, iter_time = generate_fn(
            model, inputs_copy, output_seq_length, eos_token_id, csi=csi
        )
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)
    # breakpoint()

    return timings


# pyt_gen_tokens, pyt_iter_time = generate(
#     model, input_ids.clone(), MAX_TOKENS, tokenizer.eos_token_id
# )
# pyt_timings = time_generate(
#     generate,
#     model,
#     input_ids.clone(),
#     MAX_TOKENS,
#     tokenizer.eos_token_id,
#     csi=None,
#     iterations=10,
# )
# pyt_stats = recordStats(
#     "PyTorch", pyt_timings, "fp16", batch_size=1, compile_time_s=None
# )

# print("=============================")
# print(
#     "Pytorch model generated text: ",
#     tokenizer.decode(pyt_gen_tokens[0], skip_special_tokens=True),
# )

# print("===================== \n")
# print("=========PyTorch PERFORMANCE============ \n")
# print(pyt_stats)


llama2_ep = export_llm(model, input_ids, max_seq_len=256)

# DEVICE = torch.device("cuda:0")

with torch_tensorrt.logging.debug():
    trt_model = torch_tensorrt.dynamo.compile(
        llama2_ep,
        inputs=[input_ids],
        enabled_precisions={torch.float32, torch.float16},
        truncate_double=True,
        device=DEVICE,
        disable_tf32=True,
        use_explicit_typing=False,
        use_fp32_acc=True,
        min_block_size=5,
        # min_block_size=1e7,
        debug=True,
    )


# Auto-regressive generation loop for greedy decoding using TensorRT model
# We use a custom generate function which is very similar to the huggingface one.
# Move inputs to GPU
input_ids = input_ids.to(DEVICE)
trt_gen_tokens, trt_iter_time = generate(
    trt_model, input_ids, MAX_TOKENS, tokenizer.eos_token_id
)

# print("=============================")
# print(
#     "TensorRT model generated text: ",
#     tokenizer.batch_decode(
#         trt_gen_tokens,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False,
#     )[0],
# )


print("=============================")
print(
    "TensorRT model generated text: ",
    tokenizer.decode(trt_gen_tokens[0], skip_special_tokens=True),
)

trt_timings = time_generate(
    generate,
    trt_model,
    input_ids.clone(),
    MAX_TOKENS,
    tokenizer.eos_token_id,
    iterations=10,
)
trt_stats = recordStats(
    "TensorRT", trt_timings, "fp16", batch_size=1, compile_time_s=None
)

print("===================== \n")
print("=========TRT PERFORMANCE============ \n")
print(trt_stats)
