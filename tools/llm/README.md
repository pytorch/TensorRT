# Optimizing LLMs in Torch-TensorRT

This directory provides utilities and scripts for compiling, optimizing, and benchmarking Large Language Models (LLMs) using Torch-TensorRT, with a focus on efficient inference on NVIDIA GPUs. The main entry point is `run_llm.py`, which demonstrates how to export, compile, and run LLMs with various caching strategies and precision modes. Note that this is an **experimental release** and APIs may change in future versions.

### Key Features

- **Model Support:** Works with popular LLMs such as Llama-3, Qwen2.5, etc.
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


### Usage

The main entry point is : `run_llm.py`

```bash
python run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is parallel programming?" --precision FP16 --num_tokens 128 --cache static_v2 --benchmark
```

#### Key Arguments

- `--model`: Name or path of the HuggingFace LLM.
- `--tokenizer`: (Optional) Tokenizer name; defaults to model.
- `--prompt`: Input prompt for generation.
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

## Requirements

- Torch-TensorRT 2.8.0
- Transformers v4.52.3