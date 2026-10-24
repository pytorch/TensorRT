# Torch-TensorRT ExecuTorch Backend

This directory is the source of the TensorRT delegate for ExecuTorch: the plugin that runs the
TensorRT engines stored in a `.pte` file. It ships in `libtorchtrt.tar.gz` as
`torch_tensorrt/src/torch_tensorrt/executorch/`, except in the JetPack and TensorRT-RTX builds.

Most applications do not need to build it. The `torch-tensorrt-executorch-runtime` wheel ships it
prebuilt, with a CMake package. The
[ExecuTorch deployment guide](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html)
covers installing, exporting and running a model. This README covers building the delegate from
source, and what a C++ application has to know when it runs one.

## Build it with your runner

You need:

- A CUDA Toolkit, 12.5 or newer. The prebuilt pieces need CUDA 13: the wheels, and the
  `libextension_cuda.so` in `libtorchtrt.tar.gz` that the standalone archive below uses.
- A TensorRT SDK. Point `TensorRT_ROOT` at it.
- An ExecuTorch source checkout with its submodules (`git submodule update --init --recursive`),
  at the commit your Torch-TensorRT release pins in its `MODULE.bazel`. `libtorchtrt.tar.gz` does
  not carry that file, so look the commit up in the release's source on GitHub.
- PyTorch, installed in the Python environment that CMake finds. ExecuTorch's CUDA backend looks
  for it when it is configured.

Put ExecuTorch and Torch-TensorRT side by side. The paths below are the ones in
`libtorchtrt.tar.gz`. In a source checkout of Torch-TensorRT, the delegate is in
`cpp/src/torch_tensorrt/executorch` instead.

```text
user_runner_project/
  executorch/
  torch_tensorrt/
```

Then add both to your runner's CMake build and link the delegate's target:

```cmake
set(EXECUTORCH_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/executorch")
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_PYBIND OFF CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_CUDA ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_DATA_LOADER ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_MODULE ON CACHE BOOL "" FORCE)
set(EXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP ON CACHE BOOL "" FORCE)

add_subdirectory("executorch")
add_subdirectory("torch_tensorrt/src/torch_tensorrt/executorch")

target_link_libraries(my_runner PRIVATE
  executorch
  executorch::backends
  executorch::extensions
  executorch::kernels
  executorch::backend_tensorrt
)
```

- **Turn on ExecuTorch's CUDA backend, even if your programs use only TensorRT.** It builds the
  shared `extension_cuda` library that the delegate links, and it registers the GPU memory
  allocator that the program's GPU buffers need. Configuring it needs PyTorch, but the runner it
  builds does not link libtorch.
- **Keep `extension_cuda` a shared library.** Every CUDA delegate in the process must read the same
  caller stream, so the build refuses a static copy.
- `EXECUTORCH_SOURCE_DIR` (or `EXECUTORCH_ROOT`) tells the delegate where the ExecuTorch checkout
  is. It reads the headers from there, and the build in the next paragraph needs it too.
- The extension options are what the CUDA backend and the `Module` class need.

If you cannot turn on the CUDA backend, for example because PyTorch is not available where you
build, add ExecuTorch without it. The delegate then builds the shared `extension_cuda` library
itself from the checkout that `EXECUTORCH_SOURCE_DIR` names, or uses a prebuilt one that you name
with `EXECUTORCH_EXTENSION_CUDA_LIBRARY` (Linux only), such as
`torch_tensorrt/lib/libextension_cuda.so` from `libtorchtrt.tar.gz`. Nothing then registers the GPU
memory allocator, so your runner has to register ExecuTorch's `CudaAllocator` itself before it loads
a program, as `cpp/src/torch_tensorrt/executorch/RegisterCudaDeviceAllocator.cpp` in the
Torch-TensorRT repository does. Without that, loading fails with `No device allocator registered`.

Linking `executorch::backend_tensorrt` builds the delegate's static archive,
`${CMAKE_BINARY_DIR}/lib/libexecutorch_trt_backend.a` (target `executorch_trt_backend`), as a
dependency of your runner. There is no separate step.

`examples/executorch_reference_runner/` in the Torch-TensorRT repository is a complete runner
built this way.

### Standalone archive

Use this only when you need `libexecutorch_trt_backend.a` without a runner that adds ExecuTorch
with `add_subdirectory`. First build the ExecuTorch core runtime:

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

Then build the archive, from the folder that holds `executorch/` and `torch_tensorrt/`:

```bash
cmake -S torch_tensorrt/src/torch_tensorrt/executorch -B build-torchtrt-executorch \
  -DEXECUTORCH_ROOT="${EXECUTORCH_ROOT}" \
  -DEXECUTORCH_EXTENSION_CUDA_LIBRARY="${PWD}/torch_tensorrt/lib/libextension_cuda.so" \
  -DTensorRT_ROOT="${TensorRT_ROOT}"

cmake --build build-torchtrt-executorch --target executorch_trt_backend -j
```

- `EXECUTORCH_EXTENSION_CUDA_LIBRARY` names the prebuilt `extension_cuda` library that
  `libtorchtrt.tar.gz` ships. Without it, the delegate builds that library from the checkout,
  without ExecuTorch's include directories, and the build stops on `c10/util/safe_numerics.h`.
  The prebuilt library links CUDA 13, so use a CUDA 13 toolkit here.
- Link the archive into your program with `-Wl,--whole-archive`, together with
  `libextension_cuda.so`, the ExecuTorch core (`libexecutorch_core.a`), TensorRT's `nvinfer`, the
  CUDA runtime and pthreads. The delegate registers itself from a static initializer, and a plain
  static link drops it.
- Nothing in this setup registers the GPU memory allocator, so your program has to register
  ExecuTorch's `CudaAllocator` itself, as described above.

## Running programs from C++

The delegate registers itself as `TensorRTBackend` when its library loads. After that you use the
ordinary ExecuTorch C++ API, for example `executorch::extension::Module`. The rules below are the
ones a caller has to keep.

### Threads

- **Use one `Module` per thread.** Many threads, each with its own `Module`, work, including
  threads that share one input buffer or one caller stream. ExecuTorch's `Module` is not thread
  safe, so never call one `Module` from two threads at once, not even two different methods of it.
  Mostly ExecuTorch then fails with `Error::InvalidState`, but sometimes a thread gets the other
  thread's answer.
- **With the Python bindings, load programs from one thread.** Loading from several threads at
  once can crash the process, because ExecuTorch's logging and the bindings' output redirection
  race. Load every program on one thread first, then hand them out. In C++, concurrent loads work.

### Buffers

Every buffer you pass in must stay alive and unchanged until `execute()` returns. If another
thread frees an input during the call, the call can return a wrong answer with `Error::Ok`. Nothing
below your code can detect that. Python callers cannot hit this, because a tensor owns its memory.

### CUDA streams

- With no stream chosen, the delegate runs on `cudaStreamPerThread`, the default stream of the
  calling thread. ExecuTorch's CUDA backend does the same, so a program split between the two runs
  in the right order on one thread.
- To choose a stream, scope `executorch::extension::cuda::CallerStreamGuard` from
  `<executorch/extension/cuda/caller_stream.h>` around the call. The guard applies to the calling
  thread, and there it reaches every CUDA delegate. The stream can come from a CUDA green context,
  which keeps the program inside a fixed set of the GPU's streaming multiprocessors (SMs). TensorRT
  then warns that the engine was built without green context constraints, which it says may cause
  crashes or undefined behavior. Torch-TensorRT has no option to build engines with them.
- The stream must be on the same device as the engine.
- `execute()` always waits for the engine to finish before it returns, on any stream.
- Calls on one delegate handle are serialized by the delegate. Destroying a handle while a call on
  it runs is not allowed.

`torch_tensorrt::executorch_backend::CudaStreamGuard` was removed in Torch-TensorRT 2.14. Use
`CallerStreamGuard` instead; it is the same idea, shared by every CUDA delegate:

```cpp
#include <executorch/extension/cuda/caller_stream.h>

executorch::extension::cuda::CallerStreamGuard guard(stream);
module.forward(inputs);
```

### Trusted input

Treat every `.pte` and `.ptd` file as trusted input, like code. The engines in them run on the GPU,
and nothing checks them against a digest: the file format has none. A corrupted header is refused,
but corrupted engine bytes can load and return wrong answers.

### Load errors

- Loading returns `Error::MemoryAllocationFailed` when TensorRT runs out of memory while it
  deserializes the engine, applies the weight streaming budget, or creates the execution context.
  The log then shows the free and total GPU memory. Other TensorRT load failures return
  `Error::InvalidProgram`, with TensorRT's reason in the log.
- **Nightly only.** An engine saved without its weights, for example compiled with
  `immutable_weights=False, strip_engine_weights=True`, is refused with `Error::InvalidProgram`,
  because this delegate cannot refit it. To check, the delegate asks TensorRT for the engine's
  stripped size, and TensorRT logs one harmless warning per engine that it was not built with
  `kSTRIP_PLAN`.

### Known issues

- **Coalesced programs with weights need one directory each.** A coalesced program is one split
  between TensorRT and ExecuTorch's CUDA backend. The CUDA backend always names its weight file
  `aoti_cuda_blob.ptd`, so a second export into the same directory overwrites the first one's
  weights. The first program can then return wrong answers with no error, or fail to load.
- **Loading a coalesced program again and again grows host memory**, by about 400 KB per load,
  run and drop. ExecuTorch's CUDA backend opens a compiled library on every load and never closes
  it. GPU memory does not grow, and a program with only TensorRT engines does not grow. Load once
  and keep the program if you can.
- **A coalesced program may fail on any GPU but the first**, with an illegal memory access. This
  was seen only with a program whose recorded device was edited by hand, not one exported for that
  device, so it is not confirmed. On a machine with several GPUs, prefer the first one for a
  coalesced program until it is.

## Runtime options

The delegate reads these options. A load option applies to one `Module::load` call. A process
option applies to every program loaded after it is set with `executorch::runtime::set_option`.
The [deployment guide](https://docs.pytorch.org/TensorRT/user_guide/runtime_performance/executorch.html#runtime-options)
shows how to set both: from C++, or from Python with a nightly build.

| Option | Kind | Default | What it does |
| --- | --- | --- | --- |
| `use_cuda_graphs` | load and process, bool | the choice saved at export, else off | CUDA graph replay, below |
| `use_shared_activation_scratch` | process, bool | off | One scratch buffer per device, below |
| `use_shared_engines` | load, bool | on | Loads of the same engine share it, below |
| `weight_streaming_budget` | load, bytes as a decimal string or an int | the budget saved at export, else TensorRT's automatic one | GPU bytes for each engine's weights |

A load option of the wrong type fails the load, and a process option of the wrong type fails
`set_option`. An int budget must be smaller than 2^31 bytes (2 GiB), so pass a larger one as a
string. `set_option` returns `Error::NotFound` when the delegate is not linked into the binary, and
`Error::InvalidArgument` for `use_shared_engines`, which is a load option only. It ignores keys it
does not read, including `weight_streaming_budget`.

### CUDA graph replay

With replay on, the delegate records an engine's kernel launches once as a CUDA graph, and then
replays the whole engine with one launch. It helps engines with fixed shapes that launch many
short kernels.

Each engine reads the setting when it loads, so a later change does not affect engines already
loaded. The first match wins:

1. The `use_cuda_graphs` load option.
2. A process option set to `false`.
3. The choice saved at export with `use_cuda_graphs=` in `torch_tensorrt.save()` or
   `torch_tensorrt.executorch.export()`.
4. The process option, which is off if it was never set.

How it runs: the first call with a set of input shapes runs normally, the second records the
graph, and later calls replay it. Each engine keeps one GPU buffer per input and output, and
every call through the graph copies the inputs in and the outputs out. For an engine with few
kernels, or with large inputs and outputs, these copies can cost more than they save.

Some calls do not replay:

- A change of input shapes, or of the optimization profile, records again. So does a growth of the
  shared activation scratch below. A loop that changes shape or profile on every call never
  replays, and still pays the copies.
- An engine with aliased outputs, a GPU without stream-ordered memory, or a driver older than CUDA
  12.5 never replays. The delegate logs the reason.
- A call on a green context stream, or on a stream from another CUDA context, takes the ordinary
  path. The recorded graph is kept for the next call that can replay it.
- After three failed recordings for one set of shapes, the engine logs an error and stays on the
  ordinary path until its shapes or its profile change.

Engines on the shared activation scratch replay too. **Nightly only:** in Torch-TensorRT 2.15 they
never replay.

**When recording is unsafe.** The delegate keeps its own engine loads and destroys from overlapping
a recording. It cannot protect against the rest of the process. While a recording runs, these are
unsafe on any thread:

- creating or destroying a TensorRT engine or execution context outside this delegate,
- a whole device sync, such as `cudaDeviceSynchronize()` or `torch.cuda.synchronize()`. Loading a
  program with ExecuTorch's CUDA backend can make such a sync too.

Recording can happen at any time in an engine's life, for example on its first larger shape. Leave
replay off when other threads may do any of these things. Also do not record a call to `execute()`
into a CUDA graph of your own.

### Shared activation scratch

Every TensorRT execution context keeps its own scratch memory for intermediate results. A model
lowered to many small engines pays for that many times, and can run out of GPU memory on the
number of engines alone. With the `use_shared_activation_scratch` process option on, the contexts
created after it share one buffer per device. The buffer grows to the largest size any call has
needed, so you save the sum of the separate sizes minus the largest one. An engine that needs no
scratch at all stays out of the pool.

```cpp
#include <executorch/runtime/backend/interface.h>

using namespace executorch::runtime;

BackendOptions<1> options;
options.set_option("use_shared_activation_scratch", true);
const Error error = set_option("TensorRTBackend", options.view());
```

Set it before you load the methods that should use the pool. What it costs:

- **No parallelism between pooled calls on one device.** The delegate holds a per-device lock from
  claiming the buffer until the work is queued and marked. Two pooled calls can be submitted from
  two threads, on one stream or two, but they never run at the same time on the device.
- **A pooled call waits for the pooled call before it.** Do not hold back the work of one pooled
  call, for example with a host function on its stream, until a later pooled call returns. The
  later call then never returns.
- **The buffer never shrinks.** The largest size ever needed stays allocated until the process
  exits. Whether an engine asks for the size its current shapes need, or for the most its profile
  allows, is decided when TensorRT builds it.
- **No CUDA graph capture.** See the capture section below.
- **No `cudaDeviceReset()`.** The pool keeps its buffer, an event and a stream for the life of the
  process. Do not reset a device that ran a pooled engine.

With `use_cuda_graphs` on as well, each replaying engine still keeps its own buffers for its inputs
and outputs.

#### Capturing a call into a CUDA graph is not supported

A pooled call whose stream is being captured fails with `Error::NotSupported`. The handoff from one
pooled call to the next waits on an event recorded outside the capture, and that wait fails with
`cudaErrorStreamCaptureIsolation` and invalidates the capture in every capture mode. A growth of
the buffer adds a `cudaMalloc` and a host wait, which invalidate a global or thread-local capture.
Without the refusal, you would learn of it only when `cudaStreamEndCapture` returns
`cudaErrorStreamCaptureInvalidated`.

- The check runs before anything in the call that could break the capture.
- If the query fails for another reason, such as an earlier fault on the device, the call fails
  with `Error::InvalidProgram` and that fault's message instead.
- Only pooled calls check, and only their own stream. A capture on another stream is not caught.
  So do not overlap `execute()` with a global capture on any thread, or with a thread-local capture
  on the calling thread. This holds for every stream, and also with the option off or for an engine
  that needs no scratch: the call's final `cudaStreamSynchronize`, or the `cudaFree(nullptr)` on a
  default stream, can invalidate the capture too.

None of this makes capturing the delegate itself supported.

#### How the pool grows

The delegate asks each engine for its scratch size on every `execute()`, after the input shapes are
bound. An engine whose answer does not change with the shapes grows the pool at most once, on its
first run, so running the largest such engine first leaves one allocation. An engine whose answer
does change can grow it on any call that needs more than every call before it. Which kind an engine
is gets decided when TensorRT builds it, for example by the `kRUNTIME_ACTIVATION_RESIZE_10_10`
preview feature.

A growth frees the old buffer before `execute()` returns, without holding the per-device lock: it
waits on the host for the last call that used the buffer, queues `cudaFreeAsync` on a stream the
pool owns, and synchronizes that stream. The host wait is a whole inference with no time limit, so
if work queued ahead of that call is released only after this thread returns, the call never
returns. Measured on an A100 with CUDA 13.0, with 512 MiB behind a 10 ms call and a host function
parked for 1500 ms on an unrelated stream:

- A device-wide `cudaFree` waits for the whole device: 1500.8 to 1501.4 ms.
- The pool's own stream waits only for the buffer's work: 10.6 to 11.7 ms.
- The caller's stream does not work. A `cudaFreeAsync` of `cudaMalloc` memory gives the bytes back
  only at the next synchronize of that stream, and a caller that keeps its data on the GPU may
  never synchronize it. Measured, 4 GiB queued that way stayed lost for the life of the process,
  and a growth sequence of 8, 16, 24 and 32 GiB ran out of memory on the fourth step.

Where the device has no stream-ordered allocator, or the pool's stream could not be created, the
growth falls back to a device-wide `cudaFree`.

A call whose shapes need no scratch still needs a buffer installed, because `enqueueV3` refuses a
context with none when its engine needs scratch for any shape. Against an empty pool such a call
allocates the one-byte minimum, not the engine's figure for its whole profile, because the pool
never shrinks: on one dynamic test engine that figure is 4 MiB, against 128 KiB for the next real
call. Each call also installs its own size rather than the pool's capacity. A size query that
failed then shows up, because TensorRT refuses a buffer smaller than the bound shapes need, and
the delegate fails the call with `Error::InvalidState` instead of running on a buffer it may have
freed.

### Shared engines

Loading the same program twice in one process, for example once per robot arm, keeps one
TensorRT engine in GPU memory instead of two. Loads share an engine when they have the same engine
bytes, the same device, and the same requested weight streaming budget. Each load still creates its
own execution context, buffers and lock, so two loads of one engine can run at the same time from
two threads, unless both use the shared activation scratch.

Measured on an 8 GB Jetson Orin Nano, loading one robot policy a second time took 806 MiB more GPU
memory without sharing, and 122 MiB more with it. Inference speed did not change. To find a match,
every load hashes the engine bytes, which takes about 0.15 seconds per GB on a Jetson AGX Thor and
0.25 seconds per GB on the Orin Nano.

- The hash does not resist deliberate collisions, so every program in the process must be trusted.
- The weight streaming budget belongs to the engine, so a load that asks for a different budget
  gets its own engine. With no budget requested, the first load's automatic budget is kept.
- The engine is freed when the last load that uses it is destroyed.

Set the `use_shared_engines` load option to `false` to keep one program's engines private, for
example to compare memory, or when loads run under different CUDA driver contexts on one device,
which sharing does not tell apart. A private load does not hash the bytes.

### Engines in a .ptd file

**Nightly only.** A program exported with `external_engine_data=` keeps its engines in a `.ptd` file
next to the `.pte`. Load it with that file, for example
`Module module("model.pte", "engines.ptd");`. Without the file, or with one that lacks the engine,
loading returns `Error::InvalidExternalData`. A program whose engines share `aoti_cuda_blob.ptd`
with the CUDA backend can fail first in that backend, with `Error::NotFound`. Each engine is
stored under a key made from its SHA-256 hash, but the key is only a name: nothing checks the
engine against it.

### Optimization profiles

**Nightly only.** An engine exported with several optimization profiles, for example one for
decoding one token at a time and one for a long prompt, runs one profile at a time. A C++ host
picks it for each call with `OptimizationProfileGuard` from
`torch_tensorrt/executorch/TensorRTBackend.h`:

```cpp
#include <torch_tensorrt/executorch/TensorRTBackend.h>

using torch_tensorrt::executorch_backend::OptimizationProfileGuard;

constexpr int32_t kDecodeProfile = 0;  // same order as the profiles at export
constexpr int32_t kPrefillProfile = 1;

{
  OptimizationProfileGuard guard(kPrefillProfile);
  module.forward(prefill_inputs);
}
{
  auto guard = OptimizationProfileGuard::automatic();  // chosen from the input shapes
  module.forward(inputs);
}
```

- Without a guard, every engine runs profile 0. An input outside profile 0's range fails with
  `Error::InvalidArgument`, and when the engine has more profiles the message says so.
- A pinned index runs that profile, and an index the engine does not have fails the call. The one
  exception is an engine with a single profile and a pin above 0: it runs profile 0 and logs that
  once. A negative index always fails.
- `automatic()` keeps the current profile while the inputs fit it, and otherwise takes the lowest
  index that fits. Where ranges overlap, the choice depends on earlier calls, so pin when it
  matters.
- The guard belongs to the thread that made it, and a nested guard restores the outer one when it
  ends. Every TensorRT engine in the method reads the same index, so pin by index only when the
  engines were built from one list of profiles.
- Each load keeps its own active profile, so loads that share an engine switch on their own. A
  switch drops the load's recorded CUDA graph.
- The header is `cpp/include/torch_tensorrt/executorch/TensorRTBackend.h` in the source tree, and
  `torch_tensorrt/include/torch_tensorrt/executorch/TensorRTBackend.h` in `libtorchtrt.tar.gz`. The
  wheel ships no header. On Linux, an application can compile against the header, which also needs
  the TensorRT headers, and link the wheel's delegate. On Windows, the wheel's library does not
  export the guard. Python cannot choose a profile.
