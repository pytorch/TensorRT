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
from utils import export_llm, generate, recordStats, time_generate, generate_with_kv_cache, get_zeroed_kv_cache_inputs


DEVICE = torch.device("cuda:0")

def get_model(args):
    with torch.no_grad():
        if args.model == "meta-llama/Llama-2-7b-chat-hf":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                    num_hidden_layers=1
                )
                .eval()
                .cuda()
            )
        elif args.model == "meta-llama/Llama-3.2-1B-Instruct":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                    num_hidden_layers=1
                )
                .eval()
                .cuda()
            )
            
        elif args.model == "meta-llama/Llama-3.2-3B-Instruct":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                    # num_hidden_layers=2
                )
                .eval()
                .cuda()
            )
        elif args.model == "meta-llama/Llama-3.1-8B-Instruct":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",  # num_hidden_layers=1
                )
                .eval()
                .cuda()
            )
        elif args.model == "google/gemma-3-1b-it":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    "google/gemma-3-1b-it", 
                    use_cache=False, 
                    attn_implementation="sdpa"
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
    max_seq_len = input_ids.shape[1] + args.max_tokens
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
        "--tokenizer_path",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of LLM model tokenizer",
    )
    arg_parser.add_argument(
        "--prompt", type=str, default="What is parallel programming ?", help="Prompt"
    )
    arg_parser.add_argument("--precision", type=str, default="FP16", help="Prompt")
    arg_parser.add_argument(
        "--iterations", type=int, default=5, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--min_block_size", type=int, default=1, help="no. of iterations to run"
    )
    arg_parser.add_argument(
        "--max_tokens", type=int, default=128, help="no. of max tokens to be generated"
    )
    arg_parser.add_argument(
        "--enable_pytorch_run", 
        action="store_true", 
        help="Enable pytorch run (default: False)"
    )
    arg_parser.add_argument(
        "--cache",
        type=str,
        default="static",
        help="Type of KV cache to use",
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

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

        prompt = "What is parallel programming ?"
        # prompt = "What is the capital of France ?"
        model_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(DEVICE)
        # Prepare input prompt
        # word = "What"
        # word_ids = tokenizer(word, return_tensors="pt").input_ids[0]  # Get the first (and only) sequence
        # input_ids = word_ids.repeat(1024).unsqueeze(0).to(model.device)  # Add batch dimension and move to device

        MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + args.max_tokens
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
                    "PyTorch", pyt_timings, args.precision, batch_size=1, compile_time_s=None
                )

        # TRT
        pyt_logits_tok1 = model.cuda()(input_ids)
        next_tokens = torch.argmax(pyt_logits_tok1.logits[:, -1, :], dim=-1)
        input_seq = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        pyt_logits_tok2 = model.cuda()(input_seq)
        from lower_sdpa import *
        if args.cache == "static":
            # This import is required to register static KV cache transformations as lowering passes
            from static_cache2 import *
            trt_model = compile_torchtrt(model, input_ids, args) 
            kv_cache = get_zeroed_kv_cache_inputs(trt_model)

            # First token generation
            pyt_keys = torch.load("key.pt"); pyt_values = torch.load("value.pt")
            trt_logits, key_cache, value_cache, trt_keys_1, trt_values_1 = trt_model(input_ids.clone(), True, *kv_cache, 0, input_ids.shape[1])
            print(f"Diff between pyt and trt logits: {torch.mean(torch.abs(pyt_logits_tok1.logits - trt_logits))}")
            print(f"Diff between pyt and trt keys: {torch.mean(torch.abs(pyt_keys - trt_keys_1))}")
            print(f"Diff between pyt and trt keys in cache: {torch.mean(torch.abs(pyt_keys - key_cache[:, :, :-2, :]))}")
            print(f"Diff between pyt and trt values: {torch.mean(torch.abs(pyt_values - trt_values_1))}")
            print(f"Diff between pyt and trt values in cache: {torch.mean(torch.abs(pyt_values - value_cache[:, :, :-2, :]))}")
            next_tokens = torch.argmax(trt_logits[:, -1, :], dim=-1)

            # Second token generation
            trt_logits_2, key_cache2, value_cache2, trt_keys_2, trt_values_2 = trt_model(next_tokens[:, None], False, key_cache.clone(), value_cache.clone(), input_ids.shape[1], input_ids.shape[1]+1)
            pyt_keys2 = torch.load("key2.pt"); pyt_values2 = torch.load("value2.pt")
            print(f"Diff between pyt and trt logits: {torch.mean(torch.abs(pyt_logits_tok2.logits[:, -1:, :] - trt_logits_2))}")
            print(f"Diff between pyt and trt keys: {torch.mean(torch.abs(pyt_keys2[:, :, -2:-1, :] - trt_keys_2))}")
            print(f"Diff between pyt and trt keys in cache: {torch.mean(torch.abs(pyt_keys2 - key_cache2[:, :, :-1, :]))}")
            print(f"Diff between pyt and trt values: {torch.mean(torch.abs(pyt_values2[:, :, -2:-1, :] - trt_values_2))}")
            print(f"Diff between pyt and trt values in cache: {torch.mean(torch.abs(pyt_values2 - value_cache2[:, :, :-1, :]))}")
            breakpoint()
        elif args.cache == "dynamic":
            from dynamic_cache import *
            trt_model = compile_torchtrt(model, input_ids, args) 
            breakpoint()
            kv_cache = get_zeroed_kv_cache_inputs(trt_model)
        else:
            # pyt_logits = model.cuda()(input_ids.clone())
            trt_model = compile_torchtrt(model, input_ids, args) 
            # trt_logits = trt_model(input_ids.clone(), True)
            # print(f"Diff between pyt and trt: {torch.mean(torch.abs(pyt_logits - trt_logits))}")
            # print(f"Diff between pyt and trt logits: {torch.mean(torch.abs(pyt_logits.logits - trt_logits.logits))}")
        if args.cache == "static":
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
        elif args.cache == "dynamic":
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
                "TensorRT", trt_timings, args.precision, batch_size=1, compile_time_s=None
            )

        if args.enable_pytorch_run: 
            print_outputs("PyTorch", pyt_gen_tokens, tokenizer)
        print_outputs("TensorRT", trt_gen_tokens, tokenizer)

        if  args.benchmark:
            if args.enable_pytorch_run:
                print("=========PyTorch PERFORMANCE============ \n")
                print(pyt_stats)
            print("===================== \n")
            print("=========TensorRT PERFORMANCE============ \n")
            print(trt_stats)
