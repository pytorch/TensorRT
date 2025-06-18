# Flux Demo with Torch-TensorRT

This demo showcases the Flux image generation model accelerated using Torch-TensorRT, with support for different precision modes (FP8, INT8, FP16) and dynamic shapes.


## Installation

1. Install the required dependencies:

```bash
pip install gradio==5.29.0 nvidia-modelopt==0.27.1 diffusers==0.33.1 accelerate==1.3.0
```

## Usage

The demo can be run with different configurations:

### Basic Usage (FP16)

```bash
python flux_demo.py
```

### Using Different Precision Modes

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