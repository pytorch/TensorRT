# Torch-TensorRT ExecuTorch Delegate Wheel

This directory builds `torch-tensorrt-executorch-delegate`. The Linux wheel
contains an ExecuTorch `_portable_lib` Python runtime with `TensorRTBackend`
force-linked into the same native module that owns the backend registry.

The wheel must use the same Python, PyTorch, ExecuTorch, CUDA, TensorRT, and
C++ ABI as its matching Torch-TensorRT wheel.

## Build

```bash
executorch_cmake_location="$(bazel query \
  @executorch//:executorch/CMakeLists.txt --output=location)"
export EXECUTORCH_SOURCE_DIR="$(dirname "${executorch_cmake_location%%:*}")"
export TensorRT_ROOT=/path/to/TensorRT

python -m pip install executorch==1.3.1
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir dist packaging/executorch_delegate
```

The static ExecuTorch and delegate archives are intermediate build inputs;
users receive the final native Python module and do not compile anything.

## Use

```bash
pip install "torch-tensorrt[executorch]"
```

```python
import torch
from torch_tensorrt.executorch.runtime import load

program = load("model.pte")
outputs = program.forward(torch.ones((2, 3, 4, 4)))
```
