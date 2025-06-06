## Introduction
We use the TRT Engine Explorer (TREX) to visualize the engine graph structure. TREX is a diagnostic and profiling tool for TensorRT engine files. It allows you to inspect, benchmark, and debug TensorRT engines with ease.

## Installation
```bash
pip install git+https://github.com/NVIDIA/TensorRT.git#subdirectory=tools/experimental/trt-engine-explorer
sudo apt --yes install graphviz
```

## Usage
The example usage can be found in `draw_engine_graph_example.py`. We use `torch_tensorrt.dynamo.debugger` to first output the engine profile info that required by TREX. Note that only when the compilation settings `use_python_runtime=False` can it produce TREX profiling. When it is saved to a folder, we call `draw_engine` on the same directory where the profile files are saved, which is in the subdirectory `engine_visualization_profile`.