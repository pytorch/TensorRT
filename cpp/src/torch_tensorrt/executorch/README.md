# Torch-TensorRT ExecuTorch Backend

This package is included in `libtorchtrt.tar.gz` as
`torch_tensorrt/src/torch_tensorrt/executorch/`. It builds the TensorRT
backend delegate for ExecuTorch from source.

```text
user_runner_project/
  executorch/
  torch_tensorrt/
```

This backend requires ExecuTorch 1.4 or a source commit containing
`pytorch/executorch#20158` and `pytorch/executorch#20498`. The normal integration
path is to add both ExecuTorch (with `EXECUTORCH_BUILD_CUDA=ON`) and this package
from your runner CMake, so this backend links ExecuTorch's shared `extension_cuda`
target directly. If that target is not available, set
`EXECUTORCH_EXTENSION_CUDA_LIBRARY` to a prebuilt `libextension_cuda`, or the CMake
target builds the minimal shared `extension_cuda` caller-stream library from the
ExecuTorch source checkout as a fallback. Linking `torchtrt::executorch_backend`
makes the backend archive a dependency of your runner target, so you do not need a
separate backend build step.

```cmake
add_subdirectory("executorch")
add_subdirectory("torch_tensorrt/src/torch_tensorrt/executorch")

target_link_libraries(
  my_runner
  PRIVATE
    executorch
    executorch::backends
    executorch::extensions
    executorch::kernels
    torchtrt::executorch_backend)
```

The backend archive is available as the `executorch_trt_backend` CMake target
and is written to `${CMAKE_BINARY_DIR}/lib/libexecutorch_trt_backend.a`.
`libextension_cuda` remains a shared runtime dependency so every CUDA-capable
delegate in the process observes the same caller-stream TLS instance.

## Caller Stream API Migration

`torch_tensorrt::executorch_backend::CudaStreamGuard` has been removed. Use
ExecuTorch's backend-neutral guard instead:

```cpp
#include <executorch/extension/cuda/caller_stream.h>

executorch::extension::cuda::CallerStreamGuard guard(stream);
module.forward(inputs);
```

The old class is intentionally not kept as a deprecated alias: the goal is one
backend-neutral primitive and one shared TLS definition, so all CUDA-capable
delegates read the same caller-stream selection. (A deprecated `using` alias to
`executorch::extension::cuda::CallerStreamGuard` would have shared that same TLS,
so this removal is an API-simplification choice, not a correctness requirement.)
This is a source-breaking C++ change; downstream callers must switch to the new
type.

### Caller-stream contract for the TensorRT backend

The upstream `CallerStreamGuard` documents the generic contract (per-thread,
nested scoping; green-context confinement rides the stream; the caller owns the
stream for the guard's lifetime; the caller manages host-data lifetime for async
work). The TensorRT backend adds these requirements, which previously lived on
the removed `CudaStreamGuard`:

- The selected stream must be on the TensorRT engine's device.
- Calls using one delegate handle must not overlap, and must not overlap with
  its destruction; the backend serializes `execute()` calls with an internal
  mutex, but destruction is not mutex-guarded.
- With a guard active and when no host staging is required (all inputs and
  outputs are directly bindable — device, managed, or unified memory),
  `execute()` may return with the TensorRT enqueue still in flight on the
  stream (no end-of-execute sync). The backend orders the next `execute()` and
  the handle's destruction after that work via an internal completion event, but
  that event only protects backend-owned state. The caller must therefore keep
  all directly bound input/output storage alive and unmodified until the work is
  complete, order any cross-stream producers/consumers with their own events,
  and synchronize the stream before reading outputs on the host.
- With no guard active, the backend falls back to `cudaStreamPerThread`. That
  fallback is invalid while a CUDA green context is current — scope a
  `CallerStreamGuard` with a green-context stream in that case. An explicit null
  stream is not a substitute for a green-context stream.

## Standalone Backend Archive

Use this path only when you need `libexecutorch_trt_backend.a` without building
a runner that adds ExecuTorch with `add_subdirectory`. In that standalone mode,
build the ExecuTorch core runtime first:

```bash
export EXECUTORCH_ROOT="${PWD}/executorch"
export TensorRT_ROOT=/path/to/extracted/TensorRT
export LD_LIBRARY_PATH="${TensorRT_ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cmake -S "${EXECUTORCH_ROOT}" -B "${EXECUTORCH_ROOT}/cmake-out" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DEXECUTORCH_BUILD_PYBIND=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON

cmake --build "${EXECUTORCH_ROOT}/cmake-out" --target executorch_core -j
```

Then build the TensorRT backend archive from this package:

```bash
cmake -S torch_tensorrt/src/torch_tensorrt/executorch -B build-torchtrt-executorch \
  -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
  -DTensorRT_ROOT="${TensorRT_ROOT}"

cmake --build build-torchtrt-executorch \
  --target executorch_trt_backend extension_cuda -j
```
