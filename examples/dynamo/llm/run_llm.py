"""
.. _torch_export_gpt2:

Compiling GPT2 using the dynamo backend
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular GPT2 model.
"""

import argparse
import copy
import os
import timeit

# %%
# Imports and Model Definition
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import nullcontext
from utils import export_llm, generate, recordStats, time_generate, generate_with_kv_cache
import sys
import os

# Register SDPA as a standalone operator. Converter and lowering pass are defined in register_sdpa.py
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from register_sdpa import *

DEVICE = torch.device("cuda:0")

def get_model(args):
    with torch.no_grad():
        # Supported list of models:
        # - meta-llama/Llama-3.2-1B-Instruct
        # - meta-llama/Llama-3.2-3B-Instruct
        # - meta-llama/Llama-3.1-8B-Instruct
        # - Qwen/Qwen2.5-1.5B-Instruct
        model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                    # num_hidden_layers=1
                )
                .eval()
                .cuda()
            )
    if args.precision == "FP16":
        model = model.to(torch.float16)
    elif args.precision == "BF16":
        model = model.to(torch.bfloat16)
    else:
        model = model.to(torch.float32)

    return model


def compile_torchtrt(model, input_ids, args):
    max_seq_len = input_ids.shape[1] + args.num_tokens
    ep = export_llm(model, input_ids, max_seq_len=max_seq_len)

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

    with (torch_tensorrt.logging.debug() if args.debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[input_ids],
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
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")



def measure_perf(trt_model, input_signature, backend_name):
    # Measure average time for 10 iterations
    import timeit
    import numpy as np
    
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
        # print(f"Iteration {i+1}: {iter_time:.4f} seconds")
    
    avg_time = total_time / iterations
    print(f"Backend: {backend_name} Average time per iteration: {avg_time*1000:.4f} milliseconds")
    print(f"Backend: {backend_name} Average throughput: {1.0/avg_time:.2f} iterations/second")

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(
        description="Run inference on a model with random input values"
    )
    arg_parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Name of LLM model"
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
    arg_parser.add_argument("--precision", type=str, default="FP16", help="Precision to use in the model. Options: FP16, BF16, FP32")
    arg_parser.add_argument(
        "--iterations", type=int, default=5, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--min_block_size", type=int, default=1, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--num_tokens", type=int, default=128, help="no. of output tokens to be generated"
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size used for benchmarking"
    )
    arg_parser.add_argument(
        "--isl", type=int, default=2048, help="Input sequence length used for benchmarking"
    )
    arg_parser.add_argument(
        "--enable_pytorch_run", 
        action="store_true", 
        help="Enable pytorch run (default: False)"
    )
    arg_parser.add_argument(
        "--cache",
        type=str,
        default="",
        help="Type of KV cache to use. Options: static_v1, static_v2, dynamic",
    )
    arg_parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="Enable cudagraphs (default: False)"
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug (default: False)"
    )
    arg_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark (default: False)"
    )
    
    args = arg_parser.parse_args()
    with torch.inference_mode():
        model = get_model(args)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

        # Prepare input for benchmarking or evaluation
        if args.benchmark:
            input_ids = torch.randint(1, 10000, (args.batch_size, args.isl), dtype=torch.int64).to(model.device)
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
                pyt_stats = recordStats(
                    "PyTorch", pyt_timings, args.precision, batch_size=args.batch_size, compile_time_s=None
                )

        if args.cache == "static_v1":
            # This import is required to register static v1 KV cache transformations as lowering passes
            import static_cache_v1
        if args.cache == "static_v2":
            # This import is required to register static v2 KV cache transformations as lowering passes
            import static_cache_v2
        elif args.cache == "dynamic":
            import dynamic_cache


        trt_model = compile_torchtrt(model, input_ids, args) 
            
        if args.cache == "static_v1" or args.cache == "static_v2" or args.cache == "dynamic":
            if args.cudagraph:
                # Run a decoding loop with prefill and generate phases so that the CUDAGraph is recorded for both of these phases.
                # trt_input_signature = (input_ids.clone(),) + get_zeroed_kv_cache_inputs(trt_model)
                torch_tensorrt.runtime.set_cudagraphs_mode(True)
             
            trt_gen_tokens = generate_with_kv_cache(
                trt_model, input_ids.clone(), MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id,
                )

            if args.benchmark:
                trt_timings = time_generate(
                    generate_with_kv_cache,
                    trt_model,
                    input_ids.clone(),
                    MAX_OUTPUT_SEQ_LENGTH,
                    tokenizer.eos_token_id,
                    iterations=args.iterations,
                )
        else:
            trt_gen_tokens = generate(
                trt_model, input_ids.clone(), MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id,
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
            trt_stats = recordStats(
                "TensorRT", trt_timings, args.precision, batch_size=args.batch_size, compile_time_s=None
            )

        
        if not args.benchmark:
            if args.enable_pytorch_run: 
                print_outputs("PyTorch", pyt_gen_tokens, tokenizer)
            
            print_outputs("TensorRT", trt_gen_tokens, tokenizer)

            if args.enable_pytorch_run: 
                print(f"PyTorch and TensorRT outputs match: {torch.equal(pyt_gen_tokens, trt_gen_tokens)}")

        if args.benchmark:
            if args.enable_pytorch_run:
                print("=========PyTorch PERFORMANCE============ \n")
                print(pyt_stats)
            print("===================== \n")
            print("=========TensorRT PERFORMANCE============ \n")
            print(trt_stats)
