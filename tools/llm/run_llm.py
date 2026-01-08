"""
.. _run_llm:

Running LLM inference with Torch-TensorRT
==========================================================

This script illustrates Torch-TensorRT workflow with dynamo backend on popular LLM models.

Backend Options:
    - sdpa: Uses SDPA lowering for attention operations (default)
    - plugin: Uses TensorRT attention plugin with KV cache support

Static Cache:
    - Only applicable with 'sdpa' backend
    - Options: static_v1, static_v2
"""

import argparse
import gc
import os
import sys
import timeit
from contextlib import nullcontext

# Add examples/dynamo to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../../examples/dynamo"))

import torch
import torch_tensorrt
from transformers import AutoModelForCausalLM, AutoTokenizer

from torchtrt_ext import register_sdpa
from utils import (
    export_llm,
    generate,
    generate_with_static_cache,
    record_stats,
    time_generate,
)

# Import plugin utilities
try:
    from plugin_utils import (
        load_plugin,
        register_plugin_op,
        set_plugin_config_from_model,
        replace_attention_with_plugin,
        LLMPluginWrapper,
        compile_plugin_model,
        create_kv_caches,
        generate_with_plugin,
        benchmark_plugin_generation,
    )
    PLUGIN_AVAILABLE = True
except ImportError as e:
    PLUGIN_AVAILABLE = False
    print(f"Warning: Plugin utilities not available: {e}")

DEVICE = torch.device("cuda:0")


def get_model(args):
    """
    Load and configure the language model for inference.

    Args:
        args: Parsed command line arguments containing:
            - model (str): Name or path of the model to load
            - precision (str): Precision to use ("FP16", "BF16", or "FP32")
            - backend (str): Backend to use ("sdpa" or "plugin")

    Returns:
        torch.nn.Module: The loaded and configured model ready for inference
    """
    with torch.no_grad():
        # For plugin backend, we don't set attn_implementation
        attn_impl_kwargs = {}
        if args.backend == "sdpa":
            attn_impl_kwargs["attn_implementation"] = "sdpa"
        
        model = (
            AutoModelForCausalLM.from_pretrained(
                args.model,
                use_cache=False,
                **attn_impl_kwargs,
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


def get_dtype_from_precision(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    if precision == "FP16":
        return torch.float16
    elif precision == "BF16":
        return torch.bfloat16
    else:
        return torch.float32


def compile_sdpa_model(model, input_ids, args):
    """
    Compile a PyTorch model to TensorRT using SDPA lowering.

    Args:
        model (torch.nn.Module): The PyTorch model to compile
        input_ids (torch.Tensor): Input token IDs tensor used for model export
        args: Parsed command line arguments

    Returns:
        torch_tensorrt.dynamo.TorchTensorRTModule: The compiled TensorRT model
    """
    # Register SDPA converter for the model
    register_sdpa.enable_sdpa_converter(args.model, model.config)
    
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


def compile_plugin_model_wrapper(model, args, max_seq_len):
    """
    Compile a model with plugin attention for TensorRT inference.
    
    Args:
        model: The PyTorch model (will be modified)
        args: Parsed command line arguments
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (compiled model function, model config)
    """
    if not PLUGIN_AVAILABLE:
        raise RuntimeError("Plugin utilities not available. Cannot compile plugin model.")
    
    dtype = get_dtype_from_precision(args.precision)
    config = model.config
    
    # Load plugin and register op
    load_plugin()
    register_plugin_op()
    
    # Set plugin config
    set_plugin_config_from_model(config, max_seq_len)
    
    # Replace attention with plugin
    model = replace_attention_with_plugin(model, config, max_seq_len, DEVICE, dtype)
    
    # Wrap model
    wrapper = LLMPluginWrapper(model)
    
    # Compile
    print("Compiling TensorRT Plugin model...")
    trt_model = compile_plugin_model(wrapper, config, max_seq_len, DEVICE, dtype, args.debug)
    
    return trt_model, config


def print_outputs(backend_name, gen_tokens, tokenizer):
    """Print the generated tokens from the model."""
    print(f"========= {backend_name} =========")
    print(
        f"{backend_name} model generated text: ",
        tokenizer.decode(gen_tokens[0], skip_special_tokens=True),
    )
    print("===================================")


def run_sdpa_generation(model, input_ids, args, tokenizer, use_cache=False):
    """
    Run generation using SDPA backend.
    
    Args:
        model: Compiled TensorRT model
        input_ids: Input token IDs
        args: Parsed arguments
        tokenizer: Tokenizer
        use_cache: Whether to use static cache
        
    Returns:
        Tuple of (generated tokens, timings if benchmark else None)
    """
    MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + args.num_tokens
    
    if use_cache:
        gen_tokens = generate_with_static_cache(
            model,
            input_ids.clone(),
            MAX_OUTPUT_SEQ_LENGTH,
            tokenizer.eos_token_id,
        )
        
        if args.benchmark:
            timings = time_generate(
                generate_with_static_cache,
                model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
                iterations=args.iterations,
            )
        else:
            timings = None
    else:
        gen_tokens = generate(
            model,
            input_ids.clone(),
            MAX_OUTPUT_SEQ_LENGTH,
            tokenizer.eos_token_id,
        )
        
        if args.benchmark:
            timings = time_generate(
                generate,
                model,
                input_ids.clone(),
                MAX_OUTPUT_SEQ_LENGTH,
                tokenizer.eos_token_id,
                iterations=args.iterations,
            )
        else:
            timings = None
    
    return gen_tokens, timings


def run_plugin_generation(model_func, config, input_ids, args, tokenizer, max_seq_len):
    """
    Run generation using plugin backend.
    
    Args:
        model_func: Compiled plugin model function
        config: Model configuration
        input_ids: Input token IDs
        args: Parsed arguments
        tokenizer: Tokenizer
        max_seq_len: Maximum sequence length
        
    Returns:
        Tuple of (generated tokens, timings if benchmark else None)
    """
    dtype = get_dtype_from_precision(args.precision)
    kv_caches = create_kv_caches(config, max_seq_len, 1, DEVICE, dtype)
    
    gen_tokens, _ = generate_with_plugin(
        model_func,
        input_ids.clone(),
        kv_caches,
        args.num_tokens,
        tokenizer.eos_token_id,
        DEVICE,
    )
    
    if args.benchmark:
        timings = []
        for i in range(args.iterations):
            elapsed_ms = benchmark_plugin_generation(
                model_func,
                config,
                input_ids.shape[1],
                args.num_tokens,
                max_seq_len,
                DEVICE,
                dtype,
                run_name=f"TensorRT-Plugin (Iter {i+1})",
            )
            timings.append(elapsed_ms / 1000.0)
    else:
        timings = None
    
    return gen_tokens, timings


def run_pytorch_baseline(model, input_ids, args, tokenizer):
    """
    Run PyTorch baseline generation.
    
    Args:
        model: PyTorch model
        input_ids: Input token IDs
        args: Parsed arguments
        tokenizer: Tokenizer
        
    Returns:
        Tuple of (generated tokens, timings if benchmark else None)
    """
    MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + args.num_tokens
    
    gen_tokens = generate(
        model, input_ids.clone(), MAX_OUTPUT_SEQ_LENGTH, tokenizer.eos_token_id
    )
    
    if args.benchmark:
        timings = time_generate(
            generate,
            model,
            input_ids.clone(),
            MAX_OUTPUT_SEQ_LENGTH,
            tokenizer.eos_token_id,
            iterations=args.iterations,
        )
    else:
        timings = None
    
    return gen_tokens, timings


def main():
    arg_parser = argparse.ArgumentParser(
        description="Run LLM inference with Torch-TensorRT"
    )
    
    # Model configuration
    arg_parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name or path of the LLM model",
    )
    arg_parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help="Name of LLM model tokenizer (defaults to model name)",
    )
    arg_parser.add_argument(
        "--prompt",
        type=str,
        default="What is parallel programming?",
        help="Input prompt for generation",
    )
    arg_parser.add_argument(
        "--precision",
        type=str,
        default="FP16",
        choices=["FP16", "BF16", "FP32"],
        help="Precision to use in the model",
    )
    
    # Backend selection
    arg_parser.add_argument(
        "--backend",
        type=str,
        default="sdpa",
        choices=["sdpa", "plugin"],
        help="Backend for TensorRT compilation: 'sdpa' (SDPA lowering) or 'plugin' (attention plugin with KV cache)",
    )
    
    # Static cache (only for SDPA backend)
    arg_parser.add_argument(
        "--cache",
        type=str,
        default="",
        choices=["", "static_v1", "static_v2"],
        help="Type of KV cache to use (only applicable with 'sdpa' backend)",
    )
    
    # Generation settings
    arg_parser.add_argument(
        "--num_tokens",
        type=int,
        default=128,
        help="Number of output tokens to generate",
    )
    
    # Benchmark settings
    arg_parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable benchmark mode (runs all backends: pytorch, sdpa, plugin)",
    )
    arg_parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations for benchmarking",
    )
    arg_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for benchmarking",
    )
    arg_parser.add_argument(
        "--isl",
        type=int,
        default=2048,
        help="Input sequence length for benchmarking",
    )
    
    # Other settings
    arg_parser.add_argument(
        "--min_block_size",
        type=int,
        default=1,
        help="Minimum block size for TensorRT compilation",
    )
    arg_parser.add_argument(
        "--cudagraph",
        action="store_true",
        help="Enable CUDA graphs",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    arg_parser.add_argument(
        "--enable_pytorch_run",
        action="store_true",
        help="Enable PyTorch baseline run (for non-benchmark mode)",
    )

    args = arg_parser.parse_args()
    
    # Validate arguments
    if args.cache and args.backend == "plugin":
        print("Warning: --cache is only applicable with 'sdpa' backend. Ignoring.")
        args.cache = ""
    
    if args.backend == "plugin" and not PLUGIN_AVAILABLE:
        raise RuntimeError("Plugin backend requested but plugin utilities are not available.")
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Precision: {args.precision}")
    print(f"  Cache: {args.cache if args.cache else 'none'}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"{'='*60}\n")
    
    with torch.inference_mode():
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
        
        # Prepare input
        if args.benchmark:
            input_ids = torch.randint(
                1, 10000, (args.batch_size, args.isl), dtype=torch.int64
            ).to(DEVICE)
        else:
            model_inputs = tokenizer(args.prompt, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(DEVICE)
        
        MAX_OUTPUT_SEQ_LENGTH = input_ids.shape[1] + args.num_tokens
        
        # =====================================================================
        # BENCHMARK MODE: Run all backends
        # =====================================================================
        if args.benchmark:
            results = {}
            
            # -----------------------------------------------------------------
            # 1. PyTorch Baseline
            # -----------------------------------------------------------------
            print("\n" + "="*60)
            print("Running PyTorch Baseline...")
            print("="*60)
            
            pyt_model = get_model(args)
            pyt_gen_tokens, pyt_timings = run_pytorch_baseline(
                pyt_model, input_ids, args, tokenizer
            )
            results["PyTorch"] = record_stats(
                "PyTorch",
                pyt_timings,
                args.precision,
                batch_size=args.batch_size,
                compile_time_s=None,
            )
            del pyt_model
            torch.cuda.empty_cache()
            gc.collect()
            
            # -----------------------------------------------------------------
            # 2. SDPA Backend
            # -----------------------------------------------------------------
            print("\n" + "="*60)
            print("Running SDPA Backend...")
            print("="*60)
            
            sdpa_model = get_model(args)
            
            # Import static cache if needed
            use_static_cache = args.cache in ["static_v1", "static_v2"]
            if args.cache == "static_v1":
                import static_cache_v1
            elif args.cache == "static_v2":
                import static_cache_v2
            
            sdpa_trt_model = compile_sdpa_model(sdpa_model, input_ids, args)
            
            if args.cudagraph and use_static_cache:
                torch_tensorrt.runtime.set_cudagraphs_mode(True)
            
            sdpa_gen_tokens, sdpa_timings = run_sdpa_generation(
                sdpa_trt_model, input_ids, args, tokenizer, use_cache=use_static_cache
            )
            
            cache_label = f" ({args.cache})" if args.cache else ""
            results[f"TensorRT-SDPA{cache_label}"] = record_stats(
                f"TensorRT-SDPA{cache_label}",
                sdpa_timings,
                args.precision,
                batch_size=args.batch_size,
                compile_time_s=None,
            )
            
            del sdpa_model, sdpa_trt_model
            torch.cuda.empty_cache()
            gc.collect()
            
            # -----------------------------------------------------------------
            # 3. Plugin Backend (if available)
            # -----------------------------------------------------------------
            if PLUGIN_AVAILABLE:
                print("\n" + "="*60)
                print("Running Plugin Backend...")
                print("="*60)
                
                try:
                    # Load fresh model for plugin
                    plugin_model = get_model(args)
                    plugin_config = plugin_model.config
                    
                    plugin_trt_model, _ = compile_plugin_model_wrapper(
                        plugin_model, args, max(2048, MAX_OUTPUT_SEQ_LENGTH)
                    )
                    
                    plugin_gen_tokens, plugin_timings = run_plugin_generation(
                        plugin_trt_model,
                        plugin_config,
                        input_ids,
                        args,
                        tokenizer,
                        max(2048, MAX_OUTPUT_SEQ_LENGTH),
                    )
                    
                    results["TensorRT-Plugin"] = record_stats(
                        "TensorRT-Plugin",
                        plugin_timings,
                        args.precision,
                        batch_size=args.batch_size,
                        compile_time_s=None,
                    )
                except Exception as e:
                    print(f"Plugin benchmark failed: {e}")
            
            # -----------------------------------------------------------------
            # Print Results
            # -----------------------------------------------------------------
            print("\n" + "="*60)
            print("BENCHMARK RESULTS")
            print("="*60)
            for name, stats in results.items():
                print(f"\n========= {name} =========")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
        
        # =====================================================================
        # GENERATION MODE: Run selected backend only
        # =====================================================================
        else:
            pyt_gen_tokens = None
            
            # PyTorch baseline if requested
            if args.enable_pytorch_run:
                print("\nRunning PyTorch Baseline...")
                pyt_model = get_model(args)
                pyt_gen_tokens, _ = run_pytorch_baseline(
                    pyt_model, input_ids, args, tokenizer
                )
                print_outputs("PyTorch", pyt_gen_tokens, tokenizer)
                del pyt_model
                torch.cuda.empty_cache()
                gc.collect()
            
            # Run selected backend
            if args.backend == "sdpa":
                print("\nCompiling SDPA model...")
                model = get_model(args)
                
                # Import static cache if needed
                use_static_cache = args.cache in ["static_v1", "static_v2"]
                if args.cache == "static_v1":
                    import static_cache_v1
                elif args.cache == "static_v2":
                    import static_cache_v2
                
                trt_model = compile_sdpa_model(model, input_ids, args)
                
                if args.cudagraph and use_static_cache:
                    torch_tensorrt.runtime.set_cudagraphs_mode(True)
                
                gen_tokens, _ = run_sdpa_generation(
                    trt_model, input_ids, args, tokenizer, use_cache=use_static_cache
                )
                
                cache_label = f" ({args.cache})" if args.cache else ""
                print_outputs(f"TensorRT-SDPA{cache_label}", gen_tokens, tokenizer)
                
                if pyt_gen_tokens is not None:
                    print(f"PyTorch and TensorRT outputs match: {torch.equal(pyt_gen_tokens, gen_tokens)}")
                
            elif args.backend == "plugin":
                print("\nCompiling Plugin model...")
                model = get_model(args)
                config = model.config
                
                trt_model, _ = compile_plugin_model_wrapper(
                    model, args, max(2048, MAX_OUTPUT_SEQ_LENGTH)
                )
                
                gen_tokens, _ = run_plugin_generation(
                    trt_model,
                    config,
                    input_ids,
                    args,
                    tokenizer,
                    max(2048, MAX_OUTPUT_SEQ_LENGTH),
                )
                
                print_outputs("TensorRT-Plugin", gen_tokens, tokenizer)
                
                if pyt_gen_tokens is not None:
                    print(f"PyTorch and TensorRT outputs match: {torch.equal(pyt_gen_tokens, gen_tokens)}")


if __name__ == "__main__":
    main()
