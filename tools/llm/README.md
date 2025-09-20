# Optimizing LLMs in Torch-TensorRT

This directory provides utilities and scripts for compiling, optimizing, and benchmarking Large Language Models (LLMs) using Torch-TensorRT, with a focus on efficient inference on NVIDIA GPUs. The main entry point is `run_llm.py`, which demonstrates how to export, compile, and run LLMs with various caching strategies and precision modes. Note that this is an **experimental release** and APIs may change in future versions.

### Key Features

- **Model Support:** Works with popular LLMs such as Llama-3, Qwen2.5, etc.
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


### Usage

The main entry point is : `run_llm.py`

```bash
python run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --prompt "What is parallel programming?" --model_precision FP16 --num_tokens 128 --cache static_v2 --benchmark
```

#### Key Arguments

- `--model`: Name or path of the HuggingFace LLM.
- `--tokenizer`: (Optional) Tokenizer name; defaults to model.
- `--prompt`: Input prompt for generation.
- `--model_precision`: Precision of model weight/buffer (`FP16`, `BF16`, `FP32`).
- `--quant_format`: (Optional) Quantization format (`fp8`, `nvfp4`) to apply.
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

## Requirements

- Torch-TensorRT 2.8.0
- Transformers v4.52.3