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

[`tensor_parallel_llama_export.py`](tensor_parallel_llama_export.py) exports
Llama/Qwen models into per-rank TensorRT engines. `--precision fp16` is the default
and uses FP16 autocast. `--precision fp32` converts floating model weights and
buffers to FP32 before tracing and disables autocast. TensorRT compilation disables
TF32, and the FP32 eager reference uses PyTorch's highest float32 matmul precision.

Run these commands from the repository root with two GPUs. Export a cached FP32
Qwen model, then reload it in a separate process:

```bash
torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llama_export.py \
    --mode export --model Qwen/Qwen2.5-0.5B-Instruct \
    --precision fp32 --cache static_v2 --save_dir /tmp/qwen_tp_fp32 \
    --prompt "What is tensor parallelism?" --num_tokens 128

torchtrtrun --nproc_per_node=2 tools/llm/tensor_parallel_llama_export.py \
    --mode load --model Qwen/Qwen2.5-0.5B-Instruct \
    --precision fp32 --cache static_v2 --save_dir /tmp/qwen_tp_fp32 \
    --prompt "What is tensor parallelism?" --num_tokens 128
```

Use `--precision fp16` and a different save directory for FP16. Changing the flag
in load mode does not convert an existing engine: re-export when changing
precision, and keep the load flag consistent with the saved engine. The saved
sequence/cache capacity must cover the inference prompt and requested tokens.
FP32 uses more memory; its performance was not measured in this investigation.

#### Optional accuracy check

Export and load do not require a tolerance profile. Without a cache, export logs
the maximum logit difference against the sharded eager model; with a cache, use
[`check_tensor_parallel_llama_export.py`](check_tensor_parallel_llama_export.py)
to compare the saved model against a full, unsharded eager reference. Each GPU
must also have enough memory for that reference.

```bash
torchtrtrun --nproc_per_node=2 tools/llm/check_tensor_parallel_llama_export.py \
    --model Qwen/Qwen2.5-0.5B-Instruct --save-dir /tmp/qwen_tp_fp32 \
    --precision fp32 --cache static_v2 --steps 128 \
    --prompt "What is tensor parallelism?"
```

No `--tolerance-profile` argument is needed for FP32. The checker uses
`atol=0.001, rtol=0.0001` and reports next-token agreement separately. FP32 still
has small numerical differences, so exact equality is not required. The initial
`atol=rtol=0.0001` check passed all 896 natural-prompt contexts below but failed
the eight synthetic-token contexts; those needed `atol` up to `0.000465` with
`rtol=0.0001`.

For FP16 the checker defaults to `atol=rtol=0.02`. Checking the measured
Qwen2.5-0.5B-Instruct configuration can explicitly select
`--tolerance-profile qwen2.5-0.5b-fp16`, which uses `atol=0.08, rtol=0.02`.
That profile requires `static_v2`, TP=2 and the matching model configuration;
the checker uses batch size 1 and rejects this profile with FP32.

#### Precision findings (2026-10-08)

The pretrained Qwen2.5-0.5B-Instruct experiment used B300 GPUs, TensorRT 11.3.0.99,
PyTorch 2.15.0.dev20261005+cu130, Transformers 5.14.1, TP=2, batch size 1 and
`static_v2`. Six prompts were checked for 64 decoding steps and four for 128
steps (896 contexts), plus eight steps from the original synthetic token input
`[4, 5, 6, 7]`. FP32 replay used the same token histories as the FP16 experiment,
so the input context at every comparison was identical. Each precision was
compared with a full eager reference at that precision; results agreed across
both TP ranks, which are counted once in this table.

| TRT versus full eager reference | Largest absolute logit difference | Matching next-token choices |
|---|---:|---:|
| FP16 autocast | 0.0805664 | 902 / 904 |
| FP32 | 0.0006111 | 904 / 904 |

Matching next-token choices means the same token had the highest score; it does
not mean the logit tensors were identical. For example, in the code prompt the
FP32 eager score for `dictionary` was 18.762611, while FP32 TRT scored it
18.762600. Both selected `dictionary`, despite the small numerical difference.

The two FP16 mismatches occurred when the full FP16 reference's highest scores
were tied. Replaying those contexts in FP32 gave the following choices:

| Prompt and generated text before the differing token | FP16 eager | FP16 TRT | FP32 eager and TRT |
|---|---|---|---|
| Implement an LRU cache: "This implementation uses a ..." | `list` | `dictionary` | `dictionary` |
| Plan a library schedule: "a reading event for children from 3 PM to ..." | `5` | `6` | `6` |

Thus FP16 TRT matched the FP32 reference at both disputed positions. This does
not establish factual answer correctness: for example, the library prompt gave
the event's start time but did not specify its ending time. Logit closeness and
next-token agreement measure numerical behavior, not free-running answer quality.

Fresh FP32 export, separate-process reload and accuracy checks passed for the
pretrained Qwen model. Small randomly initialized Llama and Qwen3 fixtures also
passed FP32 export/load/accuracy checks; these do not establish accuracy for
pretrained Llama or larger Qwen checkpoints. The default FP16 path passed the
same checks on the small Qwen3 fixture.

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