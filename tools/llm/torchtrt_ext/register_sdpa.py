import copy
import logging
import operator
from typing import Callable, Sequence, Tuple

import torch
from torch_tensorrt.dynamo._settings import CompilationSettings
from torch_tensorrt.dynamo.conversion.aten_ops_converters import args_bounds_check
from torch_tensorrt.dynamo.lowering import TORCH_TRT_DECOMPOSITIONS
from torch_tensorrt.dynamo.lowering.passes._aten_lowering_pass import (
    _aten_lowering_pass,
)
from torch_tensorrt.dynamo.lowering.passes.pass_utils import (
    clean_up_graph_after_modifications,
)

from .sdpa_converter import *

logger = logging.getLogger(__name__)

# Remove decompositions for aten.scaled_dot_product_attention, aten._scaled_dot_product_efficient_attention, aten._scaled_dot_product_flash_attention
# This is because we want to have SDPA as a standalone operator in the graph and invoke the custom converter for it.
TORCH_TRT_DECOMPOSITIONS.pop(torch.ops.aten.scaled_dot_product_attention.default, None)
TORCH_TRT_DECOMPOSITIONS.pop(
    torch.ops.aten._scaled_dot_product_efficient_attention.default, None
)
TORCH_TRT_DECOMPOSITIONS.pop(
    torch.ops.aten._scaled_dot_product_flash_attention.default, None
)

REPLACEABLE_ATEN_OPS = {
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
}


@_aten_lowering_pass
def replace_variants_of_sdpa(
    gm: torch.fx.GraphModule, settings: CompilationSettings
) -> torch.fx.GraphModule:
    """Replace scaled_dot_product_attention with an equivalent
    implementation which can be accurately converted to TRT
    """
    attn_mask = None
    is_causal = True
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target in REPLACEABLE_ATEN_OPS:
            if (
                node.target
                == torch.ops.aten._scaled_dot_product_efficient_attention.default
            ):
                if len(node.args) == 7:
                    (
                        query,
                        key,
                        value,
                        attn_bias,
                        compute_log_sumexp,
                        dropout_p,
                        is_causal,
                    ) = node.args
                elif len(node.args) == 5:
                    query, key, value, attn_mask, is_causal = node.args
                    dropout_p = 0.0

                else:
                    raise ValueError(
                        f"Unexpected number of arguments for {node.target} in the graph"
                    )
            elif (
                node.target
                == torch.ops.aten._scaled_dot_product_flash_attention.default
            ):
                if len(node.args) == 6:
                    query, key, value, dropout_p, is_causal, return_debug_mask = (
                        node.args
                    )
                if len(node.args) == 5:
                    query, key, value, dropout_p, is_causal = node.args
                elif len(node.args) == 3:
                    query, key, value = node.args
                    dropout_p = 0.0
                    is_causal = True
                else:
                    raise ValueError(
                        f"Unexpected number of arguments for {node.target} in the graph"
                    )

            logger.warning(
                f"This current version of SDPA converter only supports attn_mask = None, dropout_p = 0.0 and is_causal = True configuration. This could cause issues with accuracy for models with different configurations."
            )
            modified_input_args = (query, key, value, None, dropout_p, True)
            # Create a new node with torch.nn.functional.scaled_dot_product_attention
            # The input args is (query, key, value, is_causal). kwargs has scale
            with gm.graph.inserting_after(node):
                new_node = gm.graph.call_function(
                    torch.nn.functional.scaled_dot_product_attention,
                    args=modified_input_args,
                    kwargs={
                        "scale": node.kwargs.get("scale", None),
                        "use_fp32_acc": settings.use_fp32_acc,
                    },
                )

                # Deep copy encounters RuntimeError: Cannot access data pointer of Tensor (e.g. FakeTensor, FunctionalTensor). So we use copy instead.
                new_node.meta = copy.copy(node.meta)
                # Check if there's a getitem node following this attention node
                for user in list(node.users):
                    if user.op == "call_function" and user.target == operator.getitem:
                        # If the getitem is extracting the first element (the output tensor)
                        if user.args[1] == 0:
                            # Replace all uses of the getitem with the new attention node
                            user.replace_all_uses_with(new_node)
                            new_node.meta["val"] = new_node.meta["val"][0]
                # Replace all uses of the original node with the new node
                node.replace_all_uses_with(new_node)

            gm.graph.erase_node(node)

    # Clean up the graph
    clean_up_graph_after_modifications(gm)

    logger.debug(
        "Replaced variants of scaled_dot_product_attention with torch.nn.functional.scaled_dot_product_attention"
    )
    return gm


"""
.. _run_llm:

Running LLM inference with Torch-TensorRT
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular LLM models.
"""

import argparse
import copy
import os
import timeit
from contextlib import nullcontext

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from torchtrt_ext import register_sdpa
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import (
    export_llm,
    generate,
    generate_with_static_cache,
    record_stats,
    time_generate,
)

DEVICE = torch.device("cuda:0")


def get_model(args):
    """
    Load and configure the language model for inference.

    This function loads a pre-trained causal language model using the specified
    model name and configures it with the appropriate precision and settings
    for inference.

    Args:
        args: Parsed command line arguments containing:
            - model (str): Name or path of the model to load
            - precision (str): Precision to use ("FP16", "BF16", or "FP32")

    Returns:
        torch.nn.Module: The loaded and configured model ready for inference,
            moved to CUDA device with the specified precision
    """
    with torch.no_grad():
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                attn_implementation="sdpa",
            )
            .eval()
            .cuda()
        )
        if register_sdpa._SDPA_MAPPING.get(args.model, None) is not None:
            register_sdpa._SDPA_MAPPING[args.model](model_config=model.config)
        else:
            register_sdpa._SDPA_MAPPING["default"](model_config=model.config)

    if args.precision == "FP16":
        model = model.to(torch.float16)
    elif args.precision == "BF16":
        model = model.to(torch.bfloat16)
    else:
        model = model.to(torch.float32)

    return model


def compile_torchtrt(model, input_ids, args):
    """
    Compile a PyTorch model to TensorRT using torch_tensorrt.dynamo.compile.

    This function exports the given model to a TorchScript representation and then
    compiles it to TensorRT for optimized inference. The compilation process includes
    precision-specific optimizations and various performance tuning parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to compile
        input_ids (torch.Tensor): Input token IDs tensor used for model export
        args: Parsed command line arguments containing:
            - num_tokens (int): Number of tokens to generate (used for max sequence length)
            - precision (str): Precision to use ("FP16", "BF16", or "FP32")
            - debug (bool): Whether to enable debug logging
            - min_block_size (int): Minimum block size for TensorRT compilation

    Returns:
        torch_tensorrt.dynamo.TorchTensorRTModule: The compiled TensorRT model ready
            for optimized inference
    """
    max_seq_len = input_ids.shape[1] + args.num_tokens
    ep = export_llm(model, input_ids, max_seq_len=max_seq_len)
    position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
    # Set precision specific flags
    use_fp32_acc = False
    use_explicit_typing = False
    if args.precision == "FP16":
        enabled_precisions = {torch.float32}
        use_fp32_acc = True
        use_explicit_typing = True
    elif args.precision == "BF16":
        enabled_precisions = {torch.bfloat16}
        use_fp32_acc = False
    else:
        enabled_precisions = {torch.float32}

    with torch_tensorrt.logging.debug() if args.debug else nullcontext():
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[input_ids, position_ids],
            enabled_precisions=enabled_precisions,
            # truncate_double=True,
            use_explicit_typing=use_explicit_typing,
            use_fp32_acc=use_fp32_acc,
            device=DEVICE,
            disable_tf32=True,
            use_python_runtime=True,
            debug=args.debug,
            offload_module_to_cpu=True,
            min_block_size=args.min_block_size,
        )

    return trt_model


def print_outputs(backend_name, gen_tokens, tokenizer):
    """
    Print the generated tokens from the model.
    """
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")


def measure_perf(trt_model, input_signature, backend_name):
    """
    Measure the performance of a TensorRT model by running it multiple times and
    calculating the average time per iteration.
    """
    total_time = 0
    iterations = 10

    print("Running warmup iteration...")
    # Warmup run
    _ = trt_model(*input_signature)
    torch.cuda.synchronize()

    print(f"Measuring performance over {iterations} iterations...")
    for i in range(iterations):
        start_time = timeit.default_timer()
        _ = trt_model(*input_signature)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        iter_time = end_time - start_time
        total_time += iter_time

    avg_time = total_time / iterations
    print(
        f"Backend: {backend_name} Average time per iteration: {avg_time*1000:.4f} milliseconds"
    )
    print(
        f"Backend: {backend_name} Average throughput: {1.0/avg_time:.2f} iterations/second"
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    arg_parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of LLM model",
    )
    arg_parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Name of LLM model tokenizer",
    )
    arg_parser.add_argument(
        "--prompt", type=str, default="What is parallel programming ?", help="Prompt"
    )
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        help="Precision to use in the model. Options: FP16, BF16, FP32",
    )
    arg_parser.add_argument(
        "--iterations", type=int, default=5, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--min_block_size", type=int, default=1, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
        help="no. of output tokens to be generated",
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size used for benchmarking"
    )
    arg_parser.add_argument(
        "--isl",
        type=int,
        default=2048,
        help="Input sequence length used for benchmarking",
    )
    arg_parser.add_argument(
        "--enable_pytorch_run",
        action="store_true",
        help="Enable pytorch run (default: False)",
    )
    arg_parser.add_argument(
        "--cache",
        type=str,
        default="",
        help="Type of KV cache to use. Options: static_v1, static_v2",
    )
    arg_parser.add_argument(
        "--cudagraph", action="store_true", help="Enable cudagraphs (default: False)"
    )
    arg_parser.add_argument(
        "--debug", action="store_true", help="Enable debug (default: False)"
    )
    arg_parser.add_argument(
        "--benchmark", action="store_true", help="Enable benchmark (default: False)"
    )

    args = arg_parser.parse_args()
    with torch.inference_mode():
        model = get_model(args)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

        # Prepare input for benchmarking or evaluation
        if args.benchmark:
            input_ids = torch.randint(
                1, 10000, (args.batch_size, args.isl), dtype=torch.int64
            ).to(model.device)
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)
        else:
            model_inputs = tokenizer(args.prompt, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(DEVICE)
            position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0).to(DEVICE)

        MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + args.num_tokens
        # Pyt
        pyt_gen_tokens = None
        pyt_timings = None
        pyt_stats = None

        if args.enable_pytorch_run:
            pyt_gen_tokens = generate(
                model, input_ids.clone(), MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id
            )
            if args.benchmark:
                pyt_timings = time_generate(
                    generate,
                    model,
                    input_ids.clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    tokenizer.eos_token_id,
                    iterations=args.iterations,
                )
                pyt_stats = record_stats(
                    "PyTorch",
                    pyt_timings,
                    args.precision,
                    batch_size=args.batch_size,
                    compile_time_s=None,
                )

        if args.cache == "static_v1":
            # This import is required to register static v1 KV cache transformations as lowering passes
            import static_cache_v1
        if args.cache == "static_v2":
            # This import is required to register static v2 KV cache transformations as lowering passes
            import static_cache_v2

        # Compile the model with Torch-TensorRT
        trt_model = compile_torchtrt(model, input_ids, args)

        if args.cache == "static_v1" or args.cache == "static_v2":
            if args.cudagraph:
                # Run a decoding loop with prefill and generate phases so that the CUDAGraph is recorded for both of these phases.
                # trt_input_signature = (input_ids.clone(),) + get_zeroed_kv_cache_inputs(trt_model)
                torch_tensorrt.runtime.set_cudagraphs_mode(True)

            trt_gen_tokens = generate_with_static_cache(
                trt_model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
            )

            if args.benchmark:
                trt_timings = time_generate(
                    generate_with_static_cache,
                    trt_model,
                    input_ids.clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    tokenizer.eos_token_id,
                    iterations=args.iterations,
                )
        else:
            trt_gen_tokens = generate(
                trt_model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
            )
            if args.benchmark:
                trt_timings = time_generate(
                    generate,
                    trt_model,
                    input_ids.clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    tokenizer.eos_token_id,
                    iterations=args.iterations,
                )

        if args.benchmark:
            trt_stats = record_stats(
                "TensorRT",
                trt_timings,
                args.precision,
                batch_size=args.batch_size,
                compile_time_s=None,
            )

        if not args.benchmark:
            if args.enable_pytorch_run:
                print_outputs("PyTorch", pyt_gen_tokens, tokenizer)

            print_outputs("TensorRT", trt_gen_tokens, tokenizer)

            if args.enable_pytorch_run:
                print(
                    f"PyTorch and TensorRT outputs match: {torch.equal(pyt_gen_tokens, trt_gen_tokens)}"
                )

        if args.benchmark:
            if args.enable_pytorch_run:
                print("=========PyTorch PERFORMANCE============ \n")
                print(pyt_stats)
            print("===================== \n")
            print("=========TensorRT PERFORMANCE============ \n")
            print(trt_stats)
