# Optimizing LLMs in Torch-TensorRT

This directory provides utilities and scripts for compiling, optimizing, and benchmarking Large Language Models (LLMs) and Visual Language Models (VLMs) using Torch-TensorRT, with a focus on efficient inference on NVIDIA GPUs. The main entry points are `run_llm.py` for text-only LLMs and `run_vlm.py` for vision-language models. Note that this is an **experimental release** and APIs may change in future versions.

### Key Features

- **Model Support:** Works with popular LLMs such as Llama-3, Qwen2.5, etc.
- **VLM Support:** Supports Visual Language Models like Qwen2.5-VL and Eagle2.
- **Precision Modes:** Supports FP16, BF16, and FP32.
- **Quantization:** Supports FP8 and NVFP4 quantization formats for reduced memory usage and improved inference speed.
- **KV Cache:** Supports static and dynamic KV cache for efficient autoregressive decoding.
- **Benchmarking:** Measures and compares throughput and latency for PyTorch and TensorRT backends.
- **Custom Attention:** Registers and converts custom scaled dot-product attention (SDPA) for compatibility with TensorRT.


### Supported Models

We have officially verified support for the following models:

| Model Series | HF Model Card | Precision | KV Cache Supported ? |
|--------------|---------------|-----------|-------------------|
| GPT-2 | gpt2<br>gpt2-medium | FP16, FP32 | Yes |
| LLaMA 2 | meta-llama/Llama-2-7b-chat-hf | FP16, FP32 | Yes |
| LLaMA 3.1 | meta-llama/Llama-3.1-8B-Instruct | FP16, FP32 | Yes |
| LLaMA 3.2 | meta-llama/Llama-3.2-1B-Instruct<br>meta-llama/Llama-3.2-3B-Instruct | FP16, FP32 | Yes |
| Qwen 2.5 | Qwen/Qwen2.5-0.5B-Instruct<br>Qwen/Qwen2.5-1.5B-Instruct<br>Qwen/Qwen2.5-4B-Instruct<br>Qwen/Qwen2.5-7B-Instruct | FP16, FP32 | Yes |
| Qwen 3 | Qwen/Qwen3-0.6B<br>Qwen/Qwen3-1.7B<br>Qwen/Qwen3-4B<br>Qwen/Qwen3-8B | FP16, FP32 | Yes |
| Gemma 3 | google/gemma-3-1b-it | FP16, FP32 | Yes |

### Supported VLM Models

| Model Series | HF Model Card | Precision | KV Cache Supported ? |
|--------------|---------------|-----------|-------------------|
| Qwen 2.5 VL | Qwen/Qwen2.5-VL-3B-Instruct | FP16, FP32 | Yes |
| Eagle2 | nvidia/Eagle2-2B | FP16, FP32 | Yes |

### Usage

#### Text-only LLMs: `run_llm.py`

```bash
python run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128 --cache static_v2 --benchmark
```

#### Vision Language Models: `run_vlm.py`

```bash
python run_vlm.py --model nvidia/Eagle2-2B --precision FP16 --num_tokens 128 --cache static_v1 --enable_pytorch_run --benchmark
```

#### Key Arguments

- `--model`: Name or path of the HuggingFace LLM/VLM.
- `--tokenizer`: (Optional) Tokenizer name; defaults to model.
- `--prompt`: Input prompt for generation.
- `--image_path`: (Optional) Path to input image file for VLM models. If not provided, will use a sample image.
- `--model_precision`: Precision of model weight/buffer (`FP16`, `BF16`, `FP32`).
- `--quant_format`: (Optional) Quantization format (`int8`,  `fp8`, `nvfp4`) to apply.
- `--quant_algo`: (Optional) Quantization algorithm (`max`, `smoothquant`), by default it is `max`.
- `--weight_only`: (Optional) weight only quantization flag, by default it False.
- `--num_tokens`: Number of output tokens to generate.
- `--cache`: KV cache type (`static_v1`, `static_v2`, or empty for no KV caching).
- `--benchmark`: Enable benchmarking mode.
- `--enable_pytorch_run`: Also run and compare PyTorch baseline.

### Tensor-parallel export: FP16 and FP32

[`tensor_parallel_llm_export.py`](tensor_parallel_llm_export.py) exports
Llama/Qwen models into per-rank TensorRT engines. `--precision fp16` is the default
and uses FP16 autocast. `--precision fp32` converts floating model weights and
buffers to FP32 before tracing and disables autocast. TensorRT compilation disables
TF32, and the FP32 eager reference uses PyTorch's highest float32 matmul precision.

Run these commands from the repository root with two GPUs. Export a cached FP32
Qwen model, then reload it in a separate process:

```bash
torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llm_export.py \
    --mode export --model Qwen/Qwen2.5-0.5B-Instruct \
    --precision fp32 --cache static_v2 --save_dir /tmp/qwen_tp_fp32 \
    --prompt "What is tensor parallelism?" --num_tokens 128

torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llm_export.py \
    --mode load --model Qwen/Qwen2.5-0.5B-Instruct \
    --precision fp32 --cache static_v2 --save_dir /tmp/qwen_tp_fp32 \
    --prompt "What is tensor parallelism?" --num_tokens 128
```

Both `static_v1` and `static_v2` are supported. To run v1, replace
`--cache static_v2` with `--cache static_v1` in both commands and use a separate
save directory such as `/tmp/qwen_tp_fp32_v1`.

Use `--precision fp16` and a different save directory for FP16. Changing the flag
in load mode does not convert an existing engine: re-export when changing
precision, and keep the load flag consistent with the saved engine. The saved
sequence/cache capacity must cover the inference prompt and requested tokens.
FP32 uses more memory; its performance was not measured in this investigation.

#### Accuracy check

Compare the saved cached engines with a full, unsharded eager reference using the
same token history at every step. Each GPU must also fit the reference model.
Save this snippet as `/tmp/check_tp_accuracy.py`; its settings match the FP32
export command above. The same snippet works with `static_v1` engines by setting
`engine_dir` to their save directory. `start` and `end` select the cache positions
to update.

```python
import gc

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer

import tensor_parallel_llm_export as example  # Initializes NCCL and the device.
import torch_tensorrt
from torch_tensorrt.distributed._nccl_utils import initialize_nccl_comm
from utils import get_zeroed_static_cache_inputs

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
engine_dir = "/tmp/qwen_tp_fp32"
precision = "fp32"
prompt = "What is tensor parallelism?"
steps = 128
atol, rtol = 0.001, 0.0001

assert precision in ("fp16", "fp32") and steps > 0
rank, world = dist.get_rank(), dist.get_world_size()
device = example.DEVICE
initialize_nccl_comm()
program = torch_tensorrt.load(example._rank_path(engine_dir, rank, world))
loaded = program.module()
reference = (
    AutoModelForCausalLM.from_pretrained(
        model_id, use_cache=False, attn_implementation="sdpa"
    )
    .to(device)
    .eval()
)
if precision == "fp32":
    reference = reference.float()
    torch.set_float32_matmul_precision("highest")

tokenizer = AutoTokenizer.from_pretrained(model_id)
sequence = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
dist.broadcast(sequence, src=0)
current = sequence
kv = get_zeroed_static_cache_inputs(loaded, device=device)
if precision == "fp32":
    assert all(t.dtype == torch.float32 for t in kv), "Re-export in FP32"
for node in program.graph.nodes:
    value = node.meta.get("val")
    if node.op == "placeholder" and isinstance(value, torch.SymInt):
        capacity = program.range_constraints[value.node.expr].upper
        assert sequence.shape[1] + steps - 1 <= capacity, "Re-export with more capacity"

max_difference, token_matches = 0.0, 0
with torch.inference_mode(), torch.autocast(
    "cuda", dtype=torch.float16, enabled=precision == "fp16"
):
    for step in range(steps):
        end = sequence.shape[1]
        start = 0 if step == 0 else end - 1
        positions = torch.arange(end, device=device).unsqueeze(0)
        outputs = loaded(current, positions[:, start:], *kv, start, end)
        actual, kv = outputs[0][:, -1, :], outputs[1:]
        expected = reference(sequence, position_ids=positions).logits[:, -1, :]
        difference = (actual.float() - expected.float()).abs().max().item()
        max_difference = max(max_difference, difference)
        token_matches += int(torch.equal(actual.argmax(-1), expected.argmax(-1)))
        torch.testing.assert_close(
            actual.float(), expected.float(), atol=atol, rtol=rtol
        )
        # Feed the reference's token to both paths for the next comparison.
        current = expected.argmax(-1, keepdim=True)
        dist.broadcast(current, src=0)
        sequence = torch.cat([sequence, current], dim=1)

print(
    f"rank={rank}: max logit difference={max_difference:.7f}; "
    f"matching tokens={token_matches}/{steps}",
    flush=True,
)
# Release the engine before destroying its communicator.
del loaded, program, reference, outputs, actual, expected, kv
gc.collect()
dist.barrier()
dist.destroy_process_group()
```

Run it from the repository root:

```bash
PYTHONPATH=tools/llm torchtrtrun --nproc_per_node=2 /tmp/check_tp_accuracy.py
```

For FP16 engines, set `precision = "fp16"` and the matching `engine_dir`.
Use `atol=rtol=0.02` as the default FP16 check; the measured
Qwen2.5-0.5B-Instruct / `static_v2` / TP=2 / batch=1 configuration needs
`atol=0.08, rtol=0.02`. The FP32 snippet uses `atol=0.001, rtol=0.0001` to allow
small floating-point differences between TensorRT and PyTorch eager.
The numerical tolerance does not require identical next-token choices, which
are reported separately. Export and load do not take tolerance arguments.

#### Precision findings

Settings: pretrained Qwen2.5-0.5B-Instruct, B300 GPUs, TensorRT 11.3.0.99,
PyTorch 2.15.0.dev20261005+cu130, Transformers 5.14.1, TP=2, batch size 1,
with both `static_v1` and `static_v2`. Each cache/precision combination was checked
on the same 904 contexts: six prompts x 64 steps, four x 128 steps, and eight
steps from synthetic tokens `[4, 5, 6, 7]`. All runs followed the original FP16
reference's token histories; matching results across the two ranks are counted once.

**TRT versus full eager reference**

Each TRT engine was compared with eager PyTorch at the same precision. The logit
checks below use `atol=0.08, rtol=0.02` for FP16 and `atol=0.001, rtol=0.0001`
for FP32.

| Cache | Precision | Largest absolute logit difference | Matching next-token choices | Passing logit checks |
|---|---|---:|---:|---:|
| `static_v1` | FP16 autocast | 0.0957031 | 902 / 904 | 903 / 904 |
| `static_v1` | FP32 | 0.0006673 | 904 / 904 | 904 / 904 |
| `static_v2` | FP16 autocast | 0.0805664 | 902 / 904 | 904 / 904 |
| `static_v2` | FP32 | 0.0006111 | 904 / 904 | 904 / 904 |

For `static_v1` FP16, one logit at the eighth synthetic-input step exceeded the
numerical tolerance (required `atol` approximately 0.08412 at `rtol=0.02`), while
the next-token choice still matched. The tolerance remains unchanged. FP32
matched all token choices for both cache variants, with small logit differences.

**Prompt and generated text**

Both cache variants had the same two FP16 token mismatches, where the full eager
reference's top scores tied:

| Prompt and generated text before the differing token | FP16 eager | FP16 TRT | FP32 eager and TRT |
|---|---|---|---|
| Implement an LRU cache: "This implementation uses a ..." | `list` | `dictionary` | `dictionary` |
| Plan a library schedule: "a reading event for children from 3 PM to ..." | `5` | `6` | `6` |

FP16 TRT matched both FP32 implementations at those positions. These comparisons
measure numerical agreement; the library prompt did not specify the event's
ending time, so matching `6` does not establish factual correctness.

### Quantization

Torch-TensorRT supports quantization to reduce model memory footprint and improve inference performance:

#### Using Pre-quantized Models

To use pre-quantized models from HuggingFace:
If a model contains quantization configuration (detected automatically), the model's linear layers are converted to TensorRT quantized versions using the specified quantization algorithm (e.g., FP8, NVFP4). The quantization algorithm type is displayed during conversion.

**Note:** The `--quant_format` option will raise an error if it's used with pre-quantized models, as quantization cannot be applied to models that are already quantized.

```bash
python run_llm.py --model nvidia/Llama-3.1-8B-Instruct-FP8 --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128

python run_llm.py --model google/gemma-3-1b-it  --prompt "What is parallel programming?" --model_precision FP16 --quant_format int8 --quant_algo smoothquant --num_tokens 128

python run_llm.py --model google/gemma-3-1b-it  --prompt "What is parallel programming?" --model_precision FP16 --quant_format int8 --weight-only --num_tokens 128
```

**Expected output:**
```
Model is FP8 pre-quantized hf model. Quantized linear layers are applied
```

#### Applying quantization by ModelOpt

To apply quantization to non-quantized models using ModelOpt:
The `--quant_format` option calls `mtq.quantize()` to apply ModelOpt post-training quantization to the model.

```bash
python run_llm.py --model meta-llama/Llama-3.1-8B --quant_format fp8 --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128
```

#### Quantization Requirements

- **ModelOpt Library**: Required for quantization operations
- **FP8**: Supported on Hopper and Blackwell-generation GPUs.
- **NVFP4**: Supported on Blackwell-generation GPUs.

### Caching Strategies

- **Static Cache v1/v2:** Adds static KV cache tensors as model inputs/outputs for efficient reuse.
- **No Cache:** Standard autoregressive decoding.

Please read our tutorial on how static cache is implemented.

## Extension

This codebase can be extended to
- Add new models by specifying their HuggingFace name.
- Implement new cache strategies by adding FX graph passes.
- Customize SDPA conversion for new attention mechanisms.

## Limitations
- We do not currently support sliding window attention (used in Gemma3 and Qwen 3 models) yet.
- **Flash Attention Limitation**: Some models (e.g., Eagle2-2B) internally use flash attention operations (`torch.ops.flash_attn._flash_attn_forward.default`) which require the `flash-attn` package to be installed. Without flash-attn, these models will fail to load or run properly.
- **Qwen2.5‑VL vision is not compiled (LLM-only)**: We only compile the language model for Qwen2.5‑VL. The vision encoder is skipped because its `get_window_index` relies on dynamic Python operations.

## Requirements

- Torch-TensorRT 2.8.0
- Transformers v4.52.3
- For VLM models (run_vlm.py):
  - `pip install qwen-vl-utils` (for Qwen2.5-VL-3B-Instruct model)
  - **Flash Attention**: For models using flash attention operations (e.g., Eagle2-2B), install one of the following:
    - **Fast installation (recommended)**: `pip install flash-attn==2.8.1` (pre-built wheel, should work)
    - **Source build (slow)**: `pip install flash-attn --no-build-isolation -v` (fallback if pre-built wheels fail)