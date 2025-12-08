import fcntl
import logging
import os
import time

import torch
import torch_tensorrt
from accelerate import PartialState
from torchtrt_ext import register_sdpa
from transformers import AutoTokenizer, GPT2LMHeadModel
from utils import export_llm, generate

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Two different prompts for data parallel processing
prompt1 = "GPT2 is a model developed by."
prompt2 = "Llama is a model developed by."

input_id1 = tokenizer(prompt1, return_tensors="pt").input_ids
input_id2 = tokenizer(prompt2, return_tensors="pt").input_ids

distributed_state = PartialState()

logger.info("=" * 80)
logger.info(f"APPROACH: Built-in TensorRT Cache")
logger.info(f"GPU: {distributed_state.device}")
logger.info(f"Process Index: {distributed_state.process_index}")
logger.info("=" * 80)

# Path to cache TRT engine
engine_cache_dir = "./trt_builtin_cache"
os.makedirs(engine_cache_dir, exist_ok=True)
lock_file = os.path.join(engine_cache_dir, "compile.lock")

# Enable switching cuda context from one GPU to another
torch_tensorrt.runtime.set_multi_device_safe_mode(True)

with torch.inference_mode():
    logger.info(f"[{distributed_state.device}] Loading GPT-2 model...")
    model = (
        GPT2LMHeadModel.from_pretrained(
            "gpt2", use_cache=False, attn_implementation="sdpa"
        )
        .eval()
        .to(distributed_state.device)
    )

    model = model.to(torch.float16)
    register_sdpa.enable_sdpa_converter("gpt2", model.config)
    logger.info(f"[{distributed_state.device}] Model loaded successfully")

    # Export model
    max_seq_len = input_id1.shape[1] + 128
    logger.info(
        f"[{distributed_state.device}] Exporting model with max_seq_len={max_seq_len}..."
    )
    export_start = time.time()
    ep1 = export_llm(
        model, input_id1.to(distributed_state.device), max_seq_len=max_seq_len
    )
    export_time = time.time() - export_start
    logger.info(f"[{distributed_state.device}] Export completed in {export_time:.2f}s")

    # --- Using TensorRT's Built-in Engine Cache with Multi-GPU Synchronization ---
    logger.info(f"[{distributed_state.device}] Waiting for compilation lock...")
    lock_acquired = False

    try:
        with open(lock_file, "w") as f_lock:
            # Acquire exclusive lock (blocks until available)
            fcntl.flock(f_lock, fcntl.LOCK_EX)
            lock_acquired = True
            logger.info(
                f"[{distributed_state.device}] Lock acquired, starting TensorRT compilation (BUILT-IN CACHE)..."
            )
            compile_start = time.time()
            trt_model = torch_tensorrt.dynamo.compile(
                ep1,
                inputs=[
                    input_id1.to(distributed_state.device),
                    torch.arange(input_id1.shape[1])
                    .unsqueeze(0)
                    .to(distributed_state.device),
                ],
                enabled_precisions={torch.float32},
                use_explicit_typing=True,
                use_fp32_acc=True,
                device=distributed_state.device,
                disable_tf32=True,
                use_python_runtime=False,
                offload_module_to_cpu=False,
                min_block_size=5,
                debug=False,
                # Built-in caching configuration
                cache_built_engines=True,  # Enable automatic engine caching
                reuse_cached_engines=True,  # Reuse cached engines if available
                engine_cache_dir=engine_cache_dir,  # Directory to store cached engines
                engine_cache_size=1073741824,  # Max cache size: 1GB
                immutable_weights=False,
            )

            compile_time = time.time() - compile_start
            logger.info(
                f"[{distributed_state.device}] Compilation/loading completed in {compile_time:.2f}s"
            )

            # Release lock
            fcntl.flock(f_lock, fcntl.LOCK_UN)
            lock_acquired = False
            logger.info(f"[{distributed_state.device}] Lock released")

    except Exception as e:
        logger.error(f"[{distributed_state.device}] Error during compilation: {e}")
        if lock_acquired:
            try:
                fcntl.flock(f_lock, fcntl.LOCK_UN)
                logger.info(f"[{distributed_state.device}] Lock released after error")
            except:
                pass
        raise

# --- Data Parallel Inference: Each GPU processes different prompt ---
logger.info(f"[{distributed_state.device}] Starting data parallel inference...")

with distributed_state.split_between_processes([input_id1, input_id2]) as prompts:
    cur_input = torch.clone(prompts[0]).to(distributed_state.device)

    # Determine which prompt this GPU is processing
    prompt_text = prompt1 if distributed_state.process_index == 0 else prompt2
    logger.info(f"[{distributed_state.device}] Processing prompt: '{prompt_text}'")

    # Generate text using autoregressive generation
    generation_start = time.time()
    gen_tokens = generate(trt_model, cur_input, max_seq_len, tokenizer.eos_token_id)
    generation_time = time.time() - generation_start

    gen_text = tokenizer.batch_decode(gen_tokens)[0]

    logger.info(
        f"[{distributed_state.device}] Generation completed in {generation_time:.4f}s"
    )
    logger.info(f"[{distributed_state.device}] Generated text: {gen_text}")

# Summary
logger.info("")
logger.info("=" * 80)
logger.info(f"[{distributed_state.device}] SUMMARY - Built-in Cache Approach")
logger.info("=" * 80)
logger.info(f"[{distributed_state.device}] Export time:      {export_time:.2f}s")
logger.info(f"[{distributed_state.device}] Compilation time: {compile_time:.2f}s")
logger.info(f"[{distributed_state.device}] Generation time:  {generation_time:.4f}s")
logger.info(
    f"[{distributed_state.device}] Total time:       {export_time + compile_time + generation_time:.2f}s"
)
logger.info("=" * 80)
