# Torch-TensorRT ExecuTorch Runtime Wheel

This directory builds `torch-tensorrt-executorch-runtime`. The Linux wheel
contains an ExecuTorch `_portable_lib` Python runtime with `TensorRTBackend`
force-linked into the same native module that owns the backend registry.

The wheel must use the same Python, PyTorch, ExecuTorch, CUDA, TensorRT, and
C++ ABI as its matching Torch-TensorRT wheel.

## Runtime libraries

The wheel does not bundle PyTorch, c10, TensorRT, or CUDA shared libraries.
Its `_portable_lib.so` has origin-relative runtime search paths for the
TensorRT and CUDA library locations installed by their Python packages:

- `tensorrt_libs`
- `nvidia/cuda_runtime/lib` (CUDA 12)
- `nvidia/cu13/lib` (CUDA 13)

These packages are installed transitively with the matching `torch-tensorrt`
wheel. For a system TensorRT or CUDA installation outside these standard
locations, its `lib` directory must be available through the system dynamic
loader configuration or `LD_LIBRARY_PATH`.

The CI manylinux repair step changes the wheel platform tag; it does not
bundle these external libraries. The origin-relative paths are therefore part
of the wheel runtime contract.

## Build

> [!IMPORTANT]
> Build this wheel with `--no-build-isolation`. Its native extension must use
> the exact PyTorch installation that the matching Torch-TensorRT artifacts
> were built against. An isolated build may download a newer, ABI-incompatible
> PyTorch version.

```bash
executorch_cmake_location="$(bazel query \
  @executorch//:executorch/CMakeLists.txt --output=location)"
export EXECUTORCH_SOURCE_DIR="$(dirname "${executorch_cmake_location%%:*}")"
export TensorRT_ROOT=/path/to/TensorRT

python -m pip install executorch==1.3.1
python -m pip wheel --no-build-isolation --no-deps \
  --wheel-dir dist py/torch-tensorrt-executorch-runtime
```

The ExecuTorch source commit pinned in `MODULE.bazel` is the revision recorded
by the `executorch==1.3.1` wheel.

The static ExecuTorch and delegate archives are intermediate build inputs;
users receive the final native Python module and do not compile anything.

## Use

```bash
pip install "torch-tensorrt[executorch]"
```

```python
import torch
import torch_tensorrt

program = torch_tensorrt.load("model.pte", format="executorch")
outputs = program.forward(torch.ones((2, 3, 4, 4)))
```
