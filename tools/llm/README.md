# Optimizing LLMs in Torch-TensorRT

This directory provides utilities and scripts for compiling, optimizing, and benchmarking Large Language Models (LLMs) and Visual Language Models (VLMs) using Torch-TensorRT, with a focus on efficient inference on NVIDIA GPUs. The main entry points are `run_llm.py` for text-only LLMs and `run_vlm.py` for vision-language models. Note that this is an **experimental release** and APIs may change in future versions.

### Key Features

- **Model Support:** Works with popular LLMs such as Llama-3, Qwen2.5, etc.
- **VLM Support:** Supports Visual Language Models like Qwen2.5-VL and Eagle2.
- **Precision Modes:** Supports FP16, BF16, and FP32.
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
python run_llm.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "What is parallel programming?" --precision FP16 --num_tokens 128 --cache static_v2 --benchmark
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
- `--precision`: Precision mode (`FP16`, `FP32`).
- `--num_tokens`: Number of output tokens to generate.
- `--cache`: KV cache type (`static_v1`, `static_v2`, or empty for no KV caching).
- `--benchmark`: Enable benchmarking mode.
- `--enable_pytorch_run`: Also run and compare PyTorch baseline.

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