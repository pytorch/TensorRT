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
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList
from contextlib import nullcontext
from utils import export_llm, generate, recordStats, time_generate, generate_with_kv_cache

MAX_TOKENS = 128
DEVICE = torch.device("cuda:0")

def get_model(args):
    with torch.no_grad():
        if args.model == "meta-llama/Llama-3.2-1B-Instruct":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    args.model,
                    use_cache=False,
                    attn_implementation="sdpa",
                )
                .eval()
                .half()
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
                .half()
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
                .half()
                .cuda()
            )
        elif args.model == "google/gemma-3-1b-it":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    "google/gemma-3-1b-it", use_cache=False, attn_implementation="sdpa"
                )
                .eval()
                .half()
                .cuda()
            )
    model = model.to(torch.float16)
    return model


def compile_torchtrt(model, input_ids, min_block_size=1, debug=False):
    max_seq_len = input_ids.shape[1] + MAX_TOKENS
    ep = export_llm(model, input_ids, max_seq_len=max_seq_len)

    with (torch_tensorrt.logging.debug() if debug else nullcontext()):
        trt_model = torch_tensorrt.dynamo.compile(
            ep,
            inputs=[input_ids],
            enabled_precisions={torch.float16},
            # truncate_double=True,
            device=DEVICE,
            disable_tf32=True,
            use_python_runtime=True,
            debug=debug,
            min_block_size=min_block_size,
        )

    return trt_model


def print_outputs(backend_name, gen_tokens, tokenizer):


    print(f"============================= {backend_name} ==============================")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("=============================")

def get_zeroed_kv_cache_inputs(model: torch.fx.GraphModule):
    """
    Extracts and returns zeroed KV cache tensors from a torch.fx.GraphModule.
    
    This function identifies placeholder nodes in the graph that represent KV cache tensors,
    and creates zeroed tensors with the same shape, dtype, and device as the original placeholders.
    
    Args:
        model (torch.fx.GraphModule): The exported model graph containing KV cache placeholders
        
    Returns:
        tuple: A tuple of zeroed tensors corresponding to the KV cache placeholders in the graph
    """
    # placeholder nodes are expected to be in the following order:
    # input_ids, kv_cache_key, kv_cache_value, start_idx, end_idx
    placeholder_nodes = [node for node in model.graph.nodes if node.op == "placeholder"]
    kv_cache_inputs = placeholder_nodes[1:-2]
    zeroed_kv_cache_inputs = []
    for input in kv_cache_inputs:
        zeroed_kv_cache_inputs.append(torch.zeros(input.meta["val"].shape, dtype=input.meta["val"].dtype, device=DEVICE))

    return tuple(zeroed_kv_cache_inputs)

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
        "--disable_pytorch_run", 
        action="store_false", 
        help="Disable pytorch run (default: True)"
    )
    arg_parser.add_argument(
        "--kv_cache",
        action="store_true",
        help="Enable kv_cache (default: False)"
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug (default: False)"
    )
    args = arg_parser.parse_args()
    with torch.inference_mode():
        model = get_model(args)

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

        prompt = "What is parallel programming ?"
        model_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(DEVICE)

        # Prepare input prompt
        # word = "What"
        # word_ids = tokenizer(word, return_tensors="pt").input_ids[0]  # Get the first (and only) sequence
        # input_ids = word_ids.repeat(1024).unsqueeze(0).to(model.device)  # Add batch dimension and move to device

        MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + MAX_TOKENS
        # Pyt
        pytorch_input_signature = (input_ids.clone(),)
        if args.disable_pytorch_run:
            pyt_gen_tokens = None
        else:
            pyt_gen_tokens = generate(
                model, pytorch_input_signature, MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id
            )

            pyt_timings = time_generate(
                generate,
                model,
                pytorch_input_signature,
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
                iterations=args.iterations,
            )
            pyt_stats = recordStats(
                "PyTorch", pyt_timings, args.precision, batch_size=1, compile_time_s=None
            )

        # TRT
        if args.kv_cache:
            # This import is required to register static/dynamic KV cache transformations as lowering passes
            import torch_tensorrt.extensions

        trt_model = compile_torchtrt(model, input_ids, min_block_size=args.min_block_size, debug=args.debug)
        
        if args.kv_cache:
            trt_input_signature = (input_ids.clone(),) + get_zeroed_kv_cache_inputs(trt_model)
            trt_gen_tokens = generate_with_kv_cache(
                trt_model, trt_input_signature, MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id,
            )
            trt_timings = time_generate(
                generate_with_kv_cache,
                trt_model,
                trt_input_signature,
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
                iterations=args.iterations,
            )

        else:
            trt_gen_tokens = generate(
                trt_model, input_ids.clone(), MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id,
            )
            trt_timings = time_generate(
                generate,
                trt_model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
                iterations=args.iterations,
            )
        trt_stats = recordStats(
            "TensorRT", trt_timings, args.precision, batch_size=1, compile_time_s=None
        )


        print_outputs("TensorRT", trt_gen_tokens, tokenizer)
        print("===================== \n")
        if not args.disable_pytorch_run:
            print("=========PyTorch PERFORMANCE============ \n")
            print(pyt_stats)
            print("===================== \n")
        print("=========TensorRT PERFORMANCE============ \n")
        print(trt_stats)
