# Optimizing LLMs in Torch-TensorRT

This directory provides utilities and scripts for compiling, optimizing, and benchmarking Large Language Models (LLMs) and Visual Language Models (VLMs) using Torch-TensorRT, with a focus on efficient inference on NVIDIA GPUs. The main entry points are `run_llm.py` for text-only LLMs, `run_vlm.py` for vision-language models and `run_vla.py` for vision-language-action models (Alpamayo). Note that this is an **experimental release** and APIs may change in future versions.

### Key Features

- **Model Support:** Works with popular LLMs such as Llama-3, Qwen2.5, etc.
- **VLM Support:** Supports Visual Language Models like Qwen2.5-VL and Eagle2.
- **VLA Support:** Runs the Alpamayo 1.5 vision-language-action model (Qwen3-VL backbone + action expert) end to end.
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

### Supported VLA Models

| Model Series | HF Model Card | Precision | KV Cache Supported ? |
|--------------|---------------|-----------|-------------------|
| Alpamayo 1.5 | nvidia/Alpamayo-1.5-10B | FP16, FP8 (ModelOpt checkpoint) | Yes (`static_cache_v2`) |

### Usage

#### Text-only LLMs: `run_llm.py`

```bash
python run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128 --cache static_v2 --benchmark
```

#### Vision Language Models: `run_vlm.py`

```bash
python run_vlm.py --model nvidia/Eagle2-2B --precision FP16 --num_tokens 128 --cache static_v1 --enable_pytorch_run --benchmark
```

#### Vision-Language-Action models: `run_vla.py`

`run_vla.py` runs [Alpamayo 1.5](https://huggingface.co/nvidia/Alpamayo-1.5-10B) (Qwen3-VL backbone + flow-matching
action expert) as separate Torch-TensorRT engines for the vision tower, the text decoder (`static_cache_v2` on
the unmodified Hugging Face layers, split into segments, chunked prefill) and the expert. It needs the
`alpamayo1_5` package for the model and data.

```bash
python run_vla.py --mode build --component all --clip_id <clip uuid>      # one process per engine
python run_vla.py --mode reference --clip_id <clip uuid> --save ref.pt    # PyTorch rollout
python run_vla.py --mode run --clip_id <clip uuid> --ref ref.pt           # TensorRT rollout + comparison
```

By default `--mode run` loads one engine group at a time, so peak GPU memory stays low; the engines are then
reloaded on every call. Pass `--resident` to keep them loaded (~14 GB for FP8, ~28 GB for FP16). The script then
runs a warmup call and a timed call, and only the timed `[trt:run]` line gives steady-state latency. Engine load
time is reported apart from the stage timings.

`--model` can also be a ModelOpt FP8 checkpoint directory (one with a `modelopt_state.pth`; needs `nvidia-modelopt`).
The FP8 path needs two workarounds for TensorRT problems before TensorRT 11.6, described in `maybe_separate_sibling_input_scales` and `maybe_disable_attention_output_quantizers`.

Key arguments of `run_vla.py`:

- `--mode`: `build` (compile and save the engines), `reference` (PyTorch rollout) or `run` (TensorRT rollout).
- `--component`: Engine(s) to build (`vision`, `llm`, `expert`, `all`); `all` builds each in its own process.
- `--model`: Alpamayo checkpoint, `nvidia/Alpamayo-1.5-10B` by default, or a ModelOpt FP8 checkpoint directory.
- `--clip_id`: Dataset clip to run on (or `--parquet` with `--clip_index`).
- `--engine_dir`: Where the engines are saved and loaded (`vla_engines` by default).
- `--llm_segments`: Number of engines the text decoder is split into (2 by default) to keep build memory down.
- `--resident`: Keep all engines loaded and time a second, steady-state call (needs more GPU memory).
- `--save` / `--ref`: Write the reference results / compare the TensorRT rollout against them.

#### Key Arguments

Key arguments of `run_llm.py` (`run_vlm.py` takes a similar set):

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
- For the VLA model (run_vla.py), developed with TensorRT 11.4, Transformers 4.57.1 and PyTorch 2.15:
  - The `alpamayo1_5` package (model and dataset loading) and access to its dataset
  - `nvidia-modelopt` for FP8 checkpoints
  - A Transformers version with Qwen3-VL support
- For VLM models (run_vlm.py):
  - `pip install qwen-vl-utils` (for Qwen2.5-VL-3B-Instruct model)
  - **Flash Attention**: For models using flash attention operations (e.g., Eagle2-2B), install one of the following:
    - **Fast installation (recommended)**: `pip install flash-attn==2.8.1` (pre-built wheel, should work)
    - **Source build (slow)**: `pip install flash-attn --no-build-isolation -v` (fallback if pre-built wheels fail)

## Steps for Alpamayo 1.5 FP8 Reproduction

Alpamayo PTQ and TensorRT compilation need different PyTorch versions, so the steps below install them into
one venv in a specific order: the PTQ step runs first with the recipe's PyTorch, then PyTorch is upgraded for
Torch-TensorRT.

Measured with the FP8 checkpoint on an RTX 5090 (one clip, 6 trajectory samples, 17 generated tokens): eager PyTorch
takes about 5.3 s and TensorRT about 1.2 s (`--resident`, steady state), roughly 4.5x faster.

The Alpamayo 1.5 model and the PhysicalAI-AV dataset are gated on Hugging Face: request access to both and run
`hf auth login` first (see the [Alpamayo Recipes README](https://github.com/NVlabs/alpamayo-recipes)).

### Run the PTQ Script from Alpamayo Recipes

```bash
cd ~
# Setup Alpamayo venv with Python 3.12
uv venv --python 3.12 venv_am15
source venv_am15/bin/activate

# Clone Alpamayo Recipes and perform PTQ
git clone https://github.com/NVlabs/alpamayo-recipes.git
cd ~/alpamayo-recipes/recipes/alpamayo1_5_quant
uv sync --active --no-install-package flash-attn  # all dependencies except flash-attn
uv sync --active  # builds flash-attn against the installed torch; set MAX_JOBS=N if the build runs out of memory
# writes ./alpamayo1.5_fp8_calib100
python3 quantize.py --quant_format=fp8 --num_of_calib_clips=100 --save_model_dir=.
```

### Export and Run the TensorRT Engines

```bash
# From the root of this Torch-TensorRT checkout, with venv_am15 still active
# Torch-TensorRT needs a newer torch than the recipe: install 2.15 first so the editable install builds against it...
uv pip install torch==2.15 torchvision --upgrade
PYTHON_ONLY=1 uv pip install -e . --no-build-isolation
# ...and again in case the install pulled in a later torch (2.16 drops TorchScript, which breaks the import)
uv pip install torch==2.15 torchvision --upgrade
cd tools/llm

FP8=~/alpamayo-recipes/recipes/alpamayo1_5_quant/alpamayo1.5_fp8_calib100
PARQUET=~/alpamayo-recipes/data/1005_7cam_gold_eval_metadb_public.parquet

# Build the engines (one process per engine)
python3 run_vla.py --mode build --component all --model $FP8 --parquet $PARQUET
# Eager PyTorch reference: latency, trajectories and first-token logits
python3 run_vla.py --mode reference --model $FP8 --parquet $PARQUET --save ref.pt
# TensorRT rollout compared against the reference (--resident: keep the engines loaded, report steady-state latency)
python3 run_vla.py --mode run --resident --model $FP8 --parquet $PARQUET --ref ref.pt
```
