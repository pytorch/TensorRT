"""
End-to-End LLM Generation Example with TensorRT Attention Plugin

This example demonstrates how to use the TensorRT attention plugin for
efficient LLM inference with KV caching.

The plugin utilities are shared with tools/llm/run_llm.py for consistency.
"""

import os
import sys
import time

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add tools/llm to path for shared utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../tools/llm"))

from plugin_utils import (
    LLMPluginWrapper,
    PluginAttention,
    benchmark_plugin_generation,
    compile_plugin_model,
    create_kv_caches,
    generate_with_plugin,
    get_plugin_config,
    get_plugin_rope_cache,
    load_plugin,
    register_plugin_op,
    replace_attention_with_plugin,
    set_plugin_config_from_model,
)

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_SEQ_LEN = 2048
DTYPE = torch.float16
DEVICE = torch.device("cuda:0")

# Load the plugin
load_plugin()
register_plugin_op()


# -----------------------------------------------------------------------------
# Backward Compatibility Exports
# -----------------------------------------------------------------------------

# These are exported for backward compatibility with any code that imports
# from this module directly.

# Re-export Qwen2Wrapper as an alias for LLMPluginWrapper
Qwen2Wrapper = LLMPluginWrapper


# Re-export replace_attention for backward compatibility
def replace_attention(model, config):
    """
    Replace attention modules with plugin attention.

    This is a backward-compatible wrapper around replace_attention_with_plugin.
    """
    return replace_attention_with_plugin(model, config, MAX_SEQ_LEN, DEVICE, DTYPE)


def compile_model(model, input_ids, position_ids, kv_caches, ctx_len):
    """
    Compile a model for TensorRT inference.

    This is a backward-compatible wrapper that extracts config from the model.
    """
    # Get config from the wrapped model
    if hasattr(model, "model"):
        inner_model = model.model
        if hasattr(inner_model, "config"):
            config = inner_model.config
        else:
            config = inner_model.model.config
    else:
        config = model.config

    return compile_plugin_model(model, config, MAX_SEQ_LEN, DEVICE, DTYPE)


# Global config for backward compatibility with converter
TARGET_CONFIG = None


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def apply_repetition_penalty(logits, generated_ids, penalty):
    """Apply repetition penalty to logits."""
    if penalty == 1.0:
        return logits

    score = torch.gather(logits, 1, generated_ids)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, generated_ids, score)
    return logits


# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------


def benchmark_generation(model_func, isl, osl, config, run_name="Model"):
    """
    Benchmark generation with the plugin model.

    This wraps benchmark_plugin_generation for backward compatibility.
    """
    return benchmark_plugin_generation(
        model_func, config, isl, osl, MAX_SEQ_LEN, DEVICE, DTYPE, run_name
    )


def run_pytorch_benchmark_manual(model, config, isl, osl):
    """Run PyTorch benchmark with manual loop (no KV cache)."""
    input_ids = torch.randint(0, config.vocab_size, (1, isl), device=DEVICE)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        generated_ids = input_ids

        for _ in range(osl):
            outputs = model(generated_ids, use_cache=False)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print(
        f"PyTorch (Manual - No Cache) | ISL: {isl}, OSL: {osl} | Total Time: {elapsed_ms:.2f} ms | Tokens/sec: {osl / (elapsed_ms / 1000.0):.2f}"
    )
    return elapsed_ms


def run_pytorch_benchmark_generate(model, config, isl, osl):
    """Run PyTorch benchmark with model.generate() API."""
    input_ids = torch.randint(0, config.vocab_size, (1, isl), device=DEVICE)

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()

    with torch.no_grad():
        _ = model.generate(
            input_ids,
            max_new_tokens=osl,
            min_new_tokens=osl,
            do_sample=False,
            use_cache=True,
            pad_token_id=config.eos_token_id,
        )

    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)
    print(
        f"PyTorch (Generate) | ISL: {isl}, OSL: {osl} | Total Time: {elapsed_ms:.2f} ms | Tokens/sec: {osl / (elapsed_ms / 1000.0):.2f}"
    )
    return elapsed_ms


def generate_reference(model, tokenizer, prompt, max_new_tokens=20):
    """
    Generate reference output with PyTorch (greedy, no cache).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
    generated_ids = input_ids

    repetition_penalty = getattr(model.generation_config, "repetition_penalty", 1.0)
    print(
        f"DEBUG: Using repetition_penalty={repetition_penalty} for Reference Generation"
    )

    for _ in range(max_new_tokens):
        current_seq_len = generated_ids.shape[1]
        position_ids = torch.arange(
            current_seq_len, dtype=torch.long, device=DEVICE
        ).unsqueeze(0)

        outputs = model(generated_ids, position_ids=position_ids, use_cache=False)
        next_token_logits = outputs.logits[:, -1, :]

        next_token_logits = apply_repetition_penalty(
            next_token_logits, generated_ids, repetition_penalty
        )
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

        if next_token.item() == tokenizer.eos_token_id:
            break

        generated_ids = torch.cat([generated_ids, next_token], dim=1)

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def verify_output(trt_model_func, model_pytorch, tokenizer, prompt, max_new_tokens=20):
    """Verify TensorRT output matches PyTorch reference."""
    print(f"\nPrompt: '{prompt}'")

    # 1. PyTorch Reference Generation
    print("\n=== PyTorch Reference Generation ===")
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids

    with torch.no_grad():
        pyt_outputs = generate_reference(
            model_pytorch, tokenizer, prompt, max_new_tokens=30
        )
    print(f"PyTorch Reference Text Output: {pyt_outputs}")

    with torch.no_grad():
        pyt_outputs_generate_ids = model_pytorch.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    pyt_outputs_generate_text = tokenizer.decode(
        pyt_outputs_generate_ids[0], skip_special_tokens=True
    )
    print(f"PyTorch Generate Text Output: {pyt_outputs_generate_text}")

    pyt_text = pyt_outputs
    print(f"PyTorch Output: {pyt_text}")

    # 2. TensorRT Plugin Generation
    print("\n=== TensorRT Plugin Generation ===")

    repetition_penalty = getattr(
        model_pytorch.generation_config, "repetition_penalty", 1.0
    )
    print(
        f"DEBUG: Using repetition_penalty={repetition_penalty} for TensorRT Generation"
    )

    seq_len = input_ids.shape[1]
    position_ids = torch.arange(seq_len, dtype=torch.long, device=DEVICE).unsqueeze(0)

    config = model_pytorch.config
    kv_caches = create_kv_caches(config, MAX_SEQ_LEN, 1, DEVICE, DTYPE)

    generated_ids = input_ids

    # Prefill
    ctx_len = torch.tensor([seq_len], dtype=torch.int32, device=DEVICE)
    logits, kv_caches_delta = trt_model_func(
        input_ids, position_ids, kv_caches, ctx_len
    )

    for i, delta in enumerate(kv_caches_delta):
        seq_len_out = delta.shape[3]
        kv_caches[i][:, :, :, :seq_len_out, :] = delta

    next_token_logits = logits[:, -1, :]
    next_token_logits = apply_repetition_penalty(
        next_token_logits, generated_ids, repetition_penalty
    )
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

    generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # Decode
    cur_pos = seq_len

    if next_token.item() != tokenizer.eos_token_id:
        for _ in range(max_new_tokens - 1):
            input_ids_step = next_token
            position_ids_step = torch.tensor(
                [[cur_pos]], dtype=torch.long, device=DEVICE
            )
            ctx_len_step = torch.tensor([cur_pos + 1], dtype=torch.int32, device=DEVICE)

            logits, kv_caches_delta = trt_model_func(
                input_ids_step, position_ids_step, kv_caches, ctx_len_step
            )

            for i, delta in enumerate(kv_caches_delta):
                kv_caches[i][:, :, :, cur_pos : cur_pos + 1, :] = delta

            next_token_logits = logits[:, -1, :]
            next_token_logits = apply_repetition_penalty(
                next_token_logits, generated_ids, repetition_penalty
            )
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)

            if next_token.item() == tokenizer.eos_token_id:
                break

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            cur_pos += 1

    trt_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"TensorRT Output: {trt_text}")

    # 3. Comparison
    print("\n=== Comparison ===")
    if pyt_text == trt_text:
        print("SUCCESS: Outputs match exactly!")
    else:
        print("FAILURE: Outputs differ.")
        print(f"PyTorch:  {pyt_text}")
        print(f"TensorRT: {trt_text}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    print(f"Loading {MODEL_NAME}...")
    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Set global config for backward compatibility
    # Note: TARGET_CONFIG is defined at module level for backward compatibility
    globals()["TARGET_CONFIG"] = config

    # Set plugin config
    set_plugin_config_from_model(config, MAX_SEQ_LEN)

    # 1. PyTorch Model
    model_pytorch = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model_pytorch.eval()

    # 2. TensorRT Plugin Model
    model_trt = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=DTYPE).to(
        DEVICE
    )
    model_trt.eval()

    model_trt = replace_attention(model_trt, config)
    wrapper = LLMPluginWrapper(model_trt)

    # Compilation
    print("Compiling TensorRT model...")

    dummy_input_ids = torch.tensor([[1, 2, 3]], device=DEVICE)
    dummy_pos_ids = torch.tensor([[0, 1, 2]], device=DEVICE)
    dummy_ctx_len = torch.tensor([3], dtype=torch.int32, device=DEVICE)
    dummy_kvs = create_kv_caches(config, MAX_SEQ_LEN, 1, DEVICE, DTYPE)

    trt_model_func = compile_model(
        wrapper, dummy_input_ids, dummy_pos_ids, dummy_kvs, dummy_ctx_len
    )

    # 3. Verification
    print("\n=== Verifying Output Accuracy ===")
    verify_output(
        trt_model_func,
        model_pytorch,
        tokenizer,
        "What is parallel programming?",
        max_new_tokens=30,
    )

    # 4. Benchmarks
    benchmarks = [
        (128, 128),
        (256, 128),
        (512, 256),
    ]

    print("\n=== Starting Benchmarks ===")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    for isl, osl in benchmarks:
        print("-" * 60)
        # PyTorch Manual Loop
        run_pytorch_benchmark_manual(model_pytorch, config, isl, osl)

        # PyTorch Generate API
        run_pytorch_benchmark_generate(model_pytorch, config, isl, osl)

        # TensorRT
        benchmark_generation(trt_model_func, isl, osl, config, run_name="TensorRT")
