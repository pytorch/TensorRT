import time
import torch
import torch_tensorrt
from transformers import AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM

# Re-use helper utilities shipped with the other dynamo examples
from utils import export_llm, generate

"""
Torch-TensorRT compilation example for Qwen2 language model
----------------------------------------------------------
This script mirrors ``torch_export_gpt2.py`` but focuses **only** on the
language model component (``Qwen2ForCausalLM``).  We measure simple greedy
text-generation latency *before* and *after* compiling the model with
Torch-TensorRT.

Notes
-----
* KV-cache is disabled because current Torch-TensorRT does not yet support
  ``past_key_values`` graphs.
* Flash-Attention kernels are turned off via ``attn_implementation="eager"``
  to avoid unsupported ops during export/compilation.
* The example uses the 1.5-B parameter checkpoint to keep GPU memory
  requirements modest.  Feel free to swap ``model_id`` with any other
  ``Qwen/Qwen2-*`` variant that fits your hardware.
"""

MAX_NEW_TOKENS = 32
DEVICE = torch.device("cuda:0")
model_id = "Qwen/Qwen2-1.5B"  # change as you like — e.g. "Qwen/Qwen2-0.5B"

torch.manual_seed(0)

def load_baseline_model():
    """Load the FP16 Qwen2 causal-LM on GPU with KV-cache disabled."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # ------------------------------------------------------------------
    #   * use_cache=False → KV-cache disabled
    #   * attn_implementation="eager" → avoid Flash-Attention kernels
    # ------------------------------------------------------------------
    model = (
        Qwen2ForCausalLM.from_pretrained(
            model_id,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
            attn_implementation="eager",
            # attn_implementation = "sdpa",
            # attn_implementation = "flash_attention_2",
            torch_dtype=torch.float16,
        )
        .eval()
        .to(DEVICE)
    )
    return tokenizer, model


def compile_with_trt(model, example_input_ids):
    """Compile the *language model only* with Torch-TensorRT using dynamo backend."""
    # Export a torch.export graph with dynamic sequence length
    exported_program = export_llm(model, example_input_ids, min_seq_len=2, max_seq_len=1023)

    trt_model = torch_tensorrt.dynamo.compile(
        exported_program,
        inputs=[example_input_ids],
        enabled_precisions={torch.float16},  # FP16 inference
        device=DEVICE,
        truncate_double=True,
        disable_tf32=True,
        # use_explicit_typing=True,
        use_fp32_acc=True,  # keep matmul accumulations in FP32 (matches PyTorch)
    )
    return trt_model


def timed_generate(model, input_ids, eos_id, max_new_tokens):
    """Helper that returns runtime (seconds) + generated ids using greedy decode."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        out = generate(model, input_ids, max_new_tokens, eos_id)
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


if __name__ == "__main__":
    tokenizer, torch_model = load_baseline_model()

    prompt = "I enjoy walking with my cute dog"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    # ────────────────────────────────────────────────────────────
    # Baseline (Pure PyTorch)
    # ────────────────────────────────────────────────────────────
    torch_out, torch_time = timed_generate(
        torch_model, input_ids.clone(), tokenizer.eos_token_id, MAX_NEW_TOKENS
    )
    print("PyTorch generated text   :", tokenizer.decode(torch_out[0], skip_special_tokens=True))

    # ────────────────────────────────────────────────────────────
    # TensorRT-compiled model
    # ────────────────────────────────────────────────────────────
    print("\nCompiling Qwen2 language model with Torch-TensorRT …")
    trt_model = compile_with_trt(torch_model, input_ids)

    trt_out, trt_time = timed_generate(
        trt_model, input_ids.clone(), tokenizer.eos_token_id, MAX_NEW_TOKENS
    )

    # ────────────────────────────────────────────────────────────
    # Results
    # ────────────────────────────────────────────────────────────
    print("\n================  RESULTS  ================")
    print("PyTorch generated text   :", tokenizer.decode(torch_out[0], skip_special_tokens=True))
    print("TensorRT generated text  :", tokenizer.decode(trt_out[0], skip_special_tokens=True))
    print("Tokens identical         :", torch.equal(torch_out, trt_out))
    print()
    print(f"PyTorch time   : {torch_time:.3f} s")
    print(f"TensorRT time  : {trt_time:.3f} s")
    if trt_time > 0:
        print(f"Speed-up (×)   : {torch_time / trt_time:.2f}")
    print("===========================================") 