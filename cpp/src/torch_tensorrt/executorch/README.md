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
path for a runner that already enables ExecuTorch's CUDA backend is to add both
ExecuTorch (with `EXECUTORCH_BUILD_CUDA=ON`) and this package, so the TensorRT
backend links ExecuTorch's shared `extension_cuda` target directly. A libtorch-free
runner should leave the full CUDA/AOTI backend disabled; this package then builds
only the minimal shared `extension_cuda` caller-stream library from the ExecuTorch
source checkout. Consumers on ELF platforms may instead set
`EXECUTORCH_EXTENSION_CUDA_LIBRARY` to a prebuilt shared library; the value is
checked to be a shared object, because a static copy would give each delegate its
own caller-stream state. Linking `torchtrt::executorch_backend`
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
nested scoping; the caller owns the stream for the guard's lifetime; the caller
manages host-data lifetime for async work). The TensorRT backend adds these
requirements, which previously lived on
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
- With no guard active, the backend falls back to `cudaStreamPerThread`.
- With the `use_shared_activation_scratch` backend option enabled, one buffer
  per device backs the activation scratch of every execution context created
  while it was on, so no two enqueues against it may overlap. The backend
  enforces this itself: it holds a per-device lock from the claim on the buffer
  through the enqueue and the completion event recorded on it, so two
  `execute()` calls on one device are serialized at submission and the second's
  stream waits on the first's enqueue. They may run on one stream or on two, and
  they may be submitted concurrently from two threads — but they will not run
  concurrently on the device, so the pool costs the parallelism between them.
  Contexts created while the option was off keep their own scratch and are
  unaffected.
- The reference-runner smoke test runs inference inside a caller-stream guard on
  the discrete-GPU CI configuration, where all inputs and outputs are host-backed
  and therefore take the synchronized staging path. CI separately asserts that the
  runner resolves one shared `libextension_cuda.so`. Device-resident asynchronous
  return is not covered end to end.
- CUDA green-context streams work, and are the case this shared primitive exists
  for: one `cuGreenCtxStreamCreate` stream drives both the TensorRT delegate and
  ExecuTorch's CUDA/AOTI delegate, so both are confined to the same SM partition.
  Verified by hand on an A100 with 108 SMs, using a `.pte` whose graph splits
  across both delegates and a green context holding 8 of them: the program runs
  and matches its eager reference. To reproduce, build the reference runner with
  `-DEXECUTORCH_BUILD_CUDA=ON` and run it with `--green_context_sms=8`.

  Two limits on that result. It is not in CI, because the CI configuration builds
  the runner without the CUDA delegate. And it took the synchronized path, since
  the method inputs and outputs are host-backed, so the device-resident
  asynchronous return described above is still uncovered and the interaction
  between a green context and the internal completion event remains untested.

## Shared activation scratch

A TensorRT execution context allocates its own activation scratch and holds it
for as long as the context lives, so a model lowered to N single-layer engines
pays N copies and can run out of device memory on the layer count alone. The
`use_shared_activation_scratch` backend option — a boolean, off by default —
instead backs every context on a device from one buffer, grown to the largest
engine's requirement:

```cpp
#include <executorch/runtime/backend/interface.h>

executorch::runtime::BackendOptions<1> options;
options.set_option("use_shared_activation_scratch", true);
executorch::runtime::set_option("TensorRTBackend", options.view());
```

Check what `executorch::runtime::set_option` returns: `Error::NotFound` means no
backend is registered under that name, which is what a binary that has not linked
the backend archive gets.

N per-engine copies collapse to one, so the reclaimed memory is the sum of the N
requirements less the largest of them. Set the option before loading the methods
whose engines should use the pool, and read the `use_shared_activation_scratch`
bullet of the caller-stream contract above: engines sharing a buffer do not run
concurrently on the device. The pool never returns memory to the
device, so the largest scratch it was ever asked for stays allocated until the
process exits.

The buffer grows when an engine asks for more than every engine before it did,
and a growth is not free. It frees the buffer it replaces, and `cudaFree` waits
for everything queued on the device, not only for the enqueues that used that
buffer, so it can stall for far longer than the event wait that precedes it. The
backend keeps that stall out from under the per-device lock, so it does not hold
up another engine on the device, but it does fall after the growing call's own
enqueue, so that one `execute()` waits for its own engine work too. A growth
happens only on an engine's first run and only for an engine larger than every
engine before it, so loading the largest engine first reduces the pool to a
single allocation.

How much any one engine asks for is fixed when it is built, not when it runs.
The builder's `kRUNTIME_ACTIVATION_RESIZE_10_10` preview feature makes an engine
report what the shapes just bound need; without it, whether an engine does that
or reports its profile maximum depends on how TensorRT planned it. Either way the
pool can settle well above the live data, and nothing the runtime does changes it.

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
  --target executorch_trt_backend -j
```
