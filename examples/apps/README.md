# Flux Demo with Torch-TensorRT

This demo showcases the Flux image generation model accelerated using Torch-TensorRT, with support for different precision modes (FP8, INT8, FP16) and dynamic shapes.


## Installation

1. Install the required dependencies:

```bash
pip install gradio==5.29.0 nvidia-modelopt==0.27.1 diffusers==0.40.0 accelerate==1.3.0
```

## Usage

The demo can be run with different configurations:

### Basic Usage (FP16)

```bash
python flux_demo.py
```

### Using Different Precision Modes

- FP4 mode:
```bash
python flux_demo.py --dtype fp4
```

- FP8 mode:
```bash
python flux_demo.py --dtype fp8
```

- INT8 mode:
```bash
python flux_demo.py --dtype int8
```

- FP16 mode (default):
```bash
python flux_demo.py --dtype fp16
```

### Additional Options

- Enable dynamic shapes (allows variable batch sizes):
```bash
python flux_demo.py --dynamic_shapes
```

- Low VRAM mode (for GPUs with ≤32GB VRAM):
```bash
python flux_demo.py --low_vram_mode
```

You can combine these options as needed. For example:
```bash
python flux_demo.py --dtype fp8 --dynamic_shapes --low_vram_mode
```

## SDXL image generation

`sdxl_demo.py` loads the pretrained SDXL base pipeline and saves one generated
image. It runs the text encoders, denoising loop, and VAE decoder end to end.
The default mode accelerates the UNet with Torch-TensorRT; `--pytorch` runs the
pipeline with PyTorch only. Both modes use FP16 weights and require a CUDA GPU.
ModelOpt, xformers, Gradio, LoRA weights, and the SDXL refiner are not required.

On Windows ARM64, install native ARM64 CUDA PyTorch and a matching
[Torch-TensorRT-RTX build](https://docs.pytorch.org/TensorRT/getting_started/windows_arm64.html)
before running the TensorRT mode. The PyTorch mode does not import Torch-TensorRT.
Install the pipeline dependencies into that same environment:

```bash
python -m pip install --upgrade diffusers transformers "peft>=0.21.2" accelerate safetensors
```

From the repository root, first generate a PyTorch baseline:

```bash
python examples/apps/sdxl_demo.py --pytorch --output sdxl_pytorch.png
```

Then generate an image with the TensorRT UNet:

```bash
python examples/apps/sdxl_demo.py --output sdxl_trt.png
```

The first run downloads the model weights. The first TensorRT generation also
compiles the UNet and takes longer than subsequent denoising steps. Outputs
default to 1024x1024 with 30 inference steps; for a smaller initial run, add
`--height 512 --width 512 --num_inference_steps 10`. Change the scene with
`--prompt "a red panda sitting in a bamboo forest"`. `--model` also accepts a
local Diffusers SDXL base checkpoint directory containing the FP16 variant.

### Timing and comparing with PyTorch

Run both backends with the same prompt, seed, resolution, and inference steps:

```bash
python examples/apps/sdxl_demo.py --compare --height 512 --width 512 --num_inference_steps 10 --runs 3 --timings_json sdxl_timings.json
```

This loads the pipeline once, benchmarks native PyTorch first, then wraps the
same UNet with Torch-TensorRT. It saves `sdxl_pytorch.png` and
`sdxl_tensorrt.png`. In comparison mode, `--output scene.png` changes those names
to `scene_pytorch.png` and `scene_tensorrt.png`.

Each backend reports its first generation separately, followed by `--runs`
measured generations (default: 1). The first generation is excluded from the
warmed mean, median, and speedup. `--warmup N` adds N more unmeasured generations
after the first. Normal single-backend invocations also report these timings.

The stage timings are synchronized wall-clock seconds:

- Pipeline imports, model download/loading, GPU transfer, TensorRT import, and
  wrapper construction are setup costs, reported separately.
- `compile/export` measures the mutable module's compilation method, including
  graph export, lowering, engine building, and source-model offloading.
- Text encoding, UNet calls excluding compilation, VAE decoding, and other
  pipeline work add up to generation time. Other work includes the scheduler,
  latent preparation, and PIL conversion. UNet timing includes dispatch and
  mutable-module checks as well as inference. Any first-execution engine
  initialization or TensorRT-RTX runtime JIT is also included in UNet time,
  rather than the separate mutable-module `compile/export` measurement.
- Image saving is reported separately and excluded from generation time.

The comparison uses the median warmed end-to-end generation time. If compilation
occurs again in measured runs, the script flags it. Stage measurements synchronize
CUDA at each boundary, adding some diagnostic overhead to both backends. The
TensorRT first generation follows the PyTorch runs and shares their initialized
CUDA context; its compile timing remains measured directly.

The earlier SDXL example is in
[`examples/dynamo/mutable_torchtrt_module_example.py`](../dynamo/mutable_torchtrt_module_example.py),
under "Stable Diffusion with Huggingface". That tutorial additionally runs
ResNet, dynamic shapes, LoRA refitting, and engine caching examples.
