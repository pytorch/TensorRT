## Introduction

This toolchain captures TensorRT network creation and build parameters at runtime via a shim, then deterministically replays them to reproduce an engine build. Use it to debug or reproduce builds independent of the originating framework.

### Prerequisites
- TensorRT installed (ensure you know the absolute path to its `lib` and `bin` directories)
- `libtensorrt_shim.so` available in your TensorRT `lib` directory
- `tensorrt_player` available in your TensorRT `bin` directory

### Quick start: Capture
1) Export environment for the shim and paths (adjust paths for your system):

```bash
export TENSORRT_DIR=/path/to/TensorRT-<version>
export LD_LIBRARY_PATH=$TENSORRT_DIR/lib:$TENSORRT_DIR/bin:$LD_LIBRARY_PATH
export PATH=$TENSORRT_DIR/bin:$PATH

# Tell the shim which libnvinfer to interpose
export TRT_SHIM_NVINFER_LIB_NAME=$TENSORRT_DIR/lib/libnvinfer.so

# Preload the shim so it intercepts TensorRT API calls
export LD_PRELOAD=$TENSORRT_DIR/lib/libtensorrt_shim.so

# Where to write the capture (JSON metadata); the .bin payload will be co-located
export TRT_SHIM_OUTPUT_JSON_FILE=/absolute/path/to/shim_output.json
```

2) Run your program that builds TensorRT engines. For Torch-TensorRT Dynamo flows, wrap compilation with the debugger to trigger capture:

```python
import torch
import torch_tensorrt as torchtrt

model = ...  # your model on CUDA, in eval() mode
compile_spec = {
    "inputs": [torchtrt.Input(min_shape=(1, 3, 3), opt_shape=(2, 3, 3), max_shape=(3, 3, 3), dtype=torch.float32)],
}

with torchtrt.dynamo.Debugger(capture_shim=True):
    trt_mod = torchtrt.compile(model, **compile_spec)
```

3) After the run completes, verify the capture artifacts exist:
- JSON metadata: the path you set in `TRT_SHIM_OUTPUT_JSON_FILE`
- BIN payload: same directory as the JSON (e.g., `shim_output.bin`)

### Replay: Build the engine from the capture
Use `tensorrt_player` to replay the captured build without the original framework:

```bash
tensorrt_player -j /absolute/path/to/shim_output.json -o /absolute/path/to/output_engine
```

This produces a serialized TensorRT engine at `output_engine`.

### Validate the engine
Run the engine with `trtexec`:

```bash
trtexec --loadEngine=/absolute/path/to/output_engine
```

### Notes
- Ensure the `libnvinfer.so` used by the shim matches the TensorRT version in your environment.
- If multiple TensorRT versions are installed, prefer absolute paths as shown above.
- The capture is best-effort; if your program builds multiple engines, multiple captures may be produced.
