# Optimizing LLMs in Torch-TensorRT

This directory provides utilities and scripts for compiling, optimizing, and benchmarking Large Language Models (LLMs) and Visual Language Models (VLMs) using Torch-TensorRT, with a focus on efficient inference on NVIDIA GPUs. The main entry points are `run_llm.py` for text-only LLMs and `run_vlm.py` for vision-language models. Note that this is an **experimental release** and APIs may change in future versions.

### Key Features

- **Model Support:** Works with popular LLMs such as Llama-3, Qwen2.5, etc.
- **VLM Support:** Supports Visual Language Models like Qwen2.5-VL and Eagle2.
- **Precision Modes:** Supports FP16, BF16, and FP32.
- **Multiple Backends:** 
  - **SDPA Backend** (default): Registers custom lowering pass for SDPA operations, enabling TensorRT conversion with optional static KV cache support
  - **Plugin Backend**: Uses TensorRT Edge-LLM attention plugin for optimized inference with built-in KV cache management
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

**1. Generation with Output Verification**

Compare PyTorch and TensorRT outputs to verify correctness:

*SDPA Backend:*
```bash
python run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --backend sdpa \
  --prompt "What is parallel programming?" --precision FP16 --num_tokens 30 --enable_pytorch_run
```
<details>
<summary>Expected Output</summary>

```
========= PyTorch =========
PyTorch model generated text:  What is parallel programming? Parallel programming is a technique used to improve the performance of a program by dividing the work into smaller tasks and executing them simultaneously on multiple processors or cores.
===================================
========= TensorRT =========
TensorRT model generated text:  What is parallel programming? Parallel programming is a technique used to improve the performance of a program by dividing the work into smaller tasks and executing them simultaneously on multiple processors or cores.
===================================
PyTorch and TensorRT outputs match: True
```
</details>

*Plugin Backend:*
```bash
python run_llm.py --model Qwen/Qwen2.5-0.5B-Instruct --backend plugin \
  --prompt "What is parallel programming?" --precision FP16 --num_tokens 30 --enable_pytorch_run
```
<details>
<summary>Expected Output</summary>

```
========= PyTorch =========
PyTorch model generated text:  What is parallel programming? What are the benefits of parallel programming? What are the challenges of parallel programming? What are the different types of parallel programming? What are the advantages of
===================================
========= TensorRT =========
TensorRT model generated text:  What is parallel programming? What are the benefits of parallel programming? What are the challenges of parallel programming? What are the different types of parallel programming? What are the advantages of
===================================
PyTorch and TensorRT outputs match: True
```
</details>

**2. Benchmarking for Performance Comparison**

*Plugin Backend (compares TensorRT-Plugin vs PyTorch):*
```bash
python run_llm.py --model Qwen/Qwen2.5-0.5B-Instruct --backend plugin --precision FP16 \
  --benchmark --iterations 5 --isl 128 --num_tokens 20 --batch_size 1 --enable_pytorch_run
```

*SDPA with Static Cache (compares TensorRT-SDPA-StaticCache vs PyTorch):*
```bash
python run_llm.py --model meta-llama/Llama-3.2-1B-Instruct --backend sdpa --cache static_v2 \
  --precision FP16 --benchmark --iterations 5 --isl 128 --num_tokens 20 --batch_size 1 --enable_pytorch_run
```

> **Note**: In benchmark mode, `--prompt` is not used. Random input tokens are generated based on `--isl` (input sequence length).

#### Vision Language Models: `run_vlm.py`

*Generation with Output Verification:*
```bash
python run_vlm.py --model nvidia/Eagle2-2B --precision FP16 --num_tokens 64 --cache static_v1 --enable_pytorch_run
```

*Benchmarking:*
```bash
python run_vlm.py --model nvidia/Eagle2-2B --precision FP16 --cache static_v1 --benchmark --iterations 5 --num_tokens 128
```

#### Key Arguments

**Model Configuration:**
- `--model`: Name or path of the HuggingFace LLM/VLM.
- `--tokenizer`: (Optional) Tokenizer name; defaults to model.
- `--backend`: Backend to use (`sdpa` or `plugin`). Default is `sdpa`. Only applicable for LLM models.

**Generation Settings:**
- `--prompt`: Input prompt for generation (generation mode only, ignored in benchmark mode).
- `--image_path`: (Optional) Path to input image file for VLM models. If not provided, will use a sample image.
- `--precision`: Precision mode (`FP16`, `FP32`).
- `--num_tokens`: Number of output tokens to generate.

**Cache and Optimization:**
- `--cache`: KV cache type for SDPA backend (`static_v1`, `static_v2`, or empty for no KV caching). 
  - Note: Not applicable for plugin backend (manages cache internally).

**Benchmarking:**
- `--benchmark`: Enable benchmarking mode (uses random inputs instead of prompt).
- `--iterations`: Number of benchmark iterations. Default is 5.
- `--isl`: Input sequence length for benchmarking. Default is 2048.
- `--batch_size`: Batch size for benchmarking. Default is 1.
- `--enable_pytorch_run`: Also run and compare PyTorch baseline.

### Caching Strategies

#### SDPA Backend
- **Static Cache v1/v2:** Adds static KV cache tensors as model inputs/outputs for efficient reuse.
- **No Cache:** Standard autoregressive decoding.

Please read our tutorial on how static cache is implemented.

#### Plugin Backend
The plugin backend uses the TensorRT Edge-LLM AttentionPlugin which manages KV cache internally. The `--cache` option is not applicable and will be ignored if specified with `--backend plugin`.

## Plugin Backend Setup

To use the plugin backend (`--backend plugin`), you need to build the TensorRT Edge-LLM AttentionPlugin library.

### Building the AttentionPlugin

Currently, the plugin support requires a custom build from a feature branch:

```bash
# Clone the repository with the torch-tensorrt-python-runtime feature
git clone -b feature/torch-tensorrt-python-runtime https://github.com/chohk88/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM

# Build the plugin library
mkdir build && cd build

# Configure with CMake (adjust paths based on your environment)
# Example for typical Ubuntu setup with CUDA 12.9 and TensorRT in /usr:
cmake .. -DTRT_PACKAGE_DIR=/usr -DCUDA_VERSION=12.9

# Build
make -j$(nproc)

# The plugin library will be at: build/libNvInfer_edgellm_plugin.so
```

> **Note**: CMake configuration may vary depending on your system setup. Common options include:
> - `-DTRT_PACKAGE_DIR`: TensorRT installation directory (e.g., `/usr`, `/usr/local`)
> - `-DCUDA_VERSION`: CUDA version (e.g., `12.9`, `12.6`)
> 
> Refer to the [TensorRT-Edge-LLM build documentation](https://github.com/chohk88/TensorRT-Edge-LLM/tree/feature/torch-tensorrt-python-runtime) for complete build instructions and dependencies.

After building, update the plugin path in `plugin_utils.py`:
```python
DEFAULT_PLUGIN_PATH = "/path/to/your/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so"
```

### Additional Examples

Two comprehensive examples are provided in `examples/dynamo/` to demonstrate plugin usage:

- **`attention_plugin_example.py`**: Standalone example showing how to use the AttentionPlugin with custom models
- **`end_to_end_llm_generation_example.py`**: End-to-end LLM generation example with plugin integration

These examples can serve as references for integrating the plugin into your own applications.

## Extension

This codebase can be extended to
- Add new models by specifying their HuggingFace name.
- Implement new cache strategies by adding FX graph passes.
- Customize SDPA conversion for new attention mechanisms.
- Add new backend implementations (see `plugin_utils.py` for plugin backend reference).

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