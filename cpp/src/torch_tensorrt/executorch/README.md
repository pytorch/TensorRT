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
own caller-stream state. Linking `executorch::backend_tensorrt`
makes the backend archive a dependency of your runner target, so you do not need a
separate backend build step.

```cmake
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

### Coalesced programs with weights need one directory each

A coalesced program's CUDA partition keeps its weights in a separate file written beside the
program,
under a fixed name. Exporting a second program into the same directory overwrites the first one's
weights, and the keys inside that file describe the graph's shape rather than its values, so the
first
program still loads, still reports finding weights, and returns a wrong answer with no error.

Measured on two GPUs: wrong by 0.85, and bit identical across five runs, which is what makes it
read as
a working model rather than a broken one. Retraining and re-exporting the same architecture into one
directory is the ordinary way to hit this.

Give each export its own directory.

### A device-resident output reaches Python tagged for the device but backed by host memory

Measured on two GPUs. A program exported device resident runs, and its values are right, but the
tensor
the Python bindings hand back reports the device while its pointer is ordinary host memory. Asked
through the driver, a genuine device tensor answers device memory on device zero, and this one
answers
unregistered host memory on device minus two.

So from Python, `is_cuda` on that output proves nothing about residency, and neither does
anything built
on it. The same programs run correctly through a C++ consumer, where the caller supplies the output
address, so the runtime and this backend are doing their part. The structural reason is that the
type
carrying output metadata has no device field, so a generic runner cannot learn where a non-planned
output belongs, and a fix belongs upstream.

Two consequences worth knowing. The default output clone in the bindings copies from that
pointer, which
is why an asynchronous run can return zeros. And ExecuTorch's own CUDA backend refuses such a buffer
while this backend accepts it, so a coalesced program's behaviour depends on which backend owns
the last
partition. Measured with the same script, the same machine and the same three operators: with
TensorRT
last it passes five times out of five, and with the CUDA backend last it fails five times out of
five.

There is a workaround, and it is one export flag. Asking for the graph output to be planned, so the
program's own CUDA arena owns it instead of the bindings, turns both of those into five passes
out of
five. That is also the evidence that the two backends do not really disagree: hand either of
them real
device memory and both accept it. Only one of them checks.

Which is also why this backend's leniency is not something to rely on. It never reads the device
tag at
all, and the devices differ: of the four tested, one reads pageable host memory through shared
page tables, two read it by faulting pages in, and one cannot read it at all, so the same program
that works on a discrete card can fail on an integrated one.

### A split program runs only on the first GPU

A program whose graph is split between this delegate and ExecuTorch's CUDA
backend appears to fail on any GPU but the first. On a four-GPU machine, with the
engine moved to the second device and every buffer also there, it failed ten runs
of ten with an illegal memory access, from Python and from C++ alike, while the
same program on the first device worked and this delegate alone on the second
device worked. The allocator was seen freeing on the first device while the engine
ran on the second, so only one half of the program moved.

Read that as a warning rather than a settled result. The engine was moved by
rewriting the device recorded in the program, not by exporting for that device, so
it has not been reproduced the way a user would reach it. If you have more than one
GPU and a split program, prefer the first one until this is confirmed properly.

### Loading the same program many times grows host memory

A program whose graph is split across this delegate and ExecuTorch's CUDA backend grows the
process by roughly 390 to 440 kilobytes for every load, run and drop cycle. Measured on three
machines and two architectures as a slope fitted over hundreds of cycles, and the clearest
evidence is a count rather than a slope: one extra mapped shared object and four extra mapped
regions per cycle, with no variance at all, on the split program and none on a program carrying
only this delegate.

The cause is on the CUDA backend's side: it opens a compiled library per load and never unmaps
it. A program carrying only this delegate does not, and the same measurement puts that at a
couple of hundred bytes per cycle, which is noise. Device memory does not grow at all, with a
slope of zero over hundreds of cycles on a probe proven to be reporting rather than dead.

So a long-lived process that loads programs repeatedly will grow. Load once and keep the
program if you can.

### A corrupted engine can return a wrong answer instead of an error

Nothing checks the serialized engine against a digest, because the blob format carries none. The
header's
own fields are validated, and a corruption that breaks them is refused, but a corruption inside
the engine
bytes themselves is not seen.

Measured with 24 probes that change one value inside the engine: 12 were refused, 12 were
accepted, and 8
of those returned a wrong answer, one of them negative infinity. Each wrong value repeated
identically
across five runs, so this is deterministic rather than flaky. Corruption inside the delegate's own
metadata was ignored in all four probes.

That is a property of the format rather than of this backend, and closing it means adding a
digest to the
format and to everything that writes one, which would not be readable by programs exported
before it. Until
then, treat a program file as trusted input: check it in transit, and do not run one from a
source you would
not run code from.

### Running from several threads at once

One program per thread works. Sharing one program across threads does not, and is refused
rather than allowed, because the runtime this plugs into does not permit it.

Measured on a two-card A100 host, 100 inference calls per thread per attempt, with the threads
released into `execute()` by a barrier so they really do overlap (confirmed by an in-flight
counter reaching the thread count, and by the calls' own start and finish times overlapping for
about half of the busy time):

| shape | attempts with a failure |
| --- | --- |
| one `Module` per thread, 2 threads | 0 of 20 |
| one `Module` per thread, 4 threads | 0 of 20 |
| one `Module` per thread, staggered starts instead of a barrier | 0 of 20 |
| one `Module` per thread, both threads reading one input buffer | 0 of 20 |
| one `Module` per thread, both inside a `CallerStreamGuard` on one shared stream | 0 of 20 |
| one `Module` per thread, loaded concurrently with nothing warmed up first | 0 of 10 |
| one `Module` shared by 2 threads | 20 of 20 |
| one `Module` shared by 4 threads | 19 of 20 |

That is 26,000 answers compared against the expected value with no mismatch on the per-thread
shape. The backend's own `execute()` mutex is what makes it hold: two threads on one delegate
handle are serialized, and two threads on two handles run at the same time without touching each
other's state.

Sharing one `Module` is not a backend problem and no backend change can fix it. ExecuTorch says so
itself, in `extension/module/module.h`: "This class is not thread-safe and performs no internal
synchronization. Calling execute concurrently on the same Module instance from multiple threads is
unsafe, regardless of whether share_memory_arenas is true or false." Note "regardless": two threads
on two different methods of one `Module` is unsafe too, not only two threads on the same method.
Mostly the runtime catches it and returns `Error::InvalidState` with "Inputs can not be set mid
execution", but not always: in the measurements above it also returned between 16 and 107 answers
per configuration that belonged to the other thread.

### A crash when two threads load at the same time under the Python bindings

Loading from more than one thread through the Python bindings crashes, roughly one attempt in six.
The same shape in C++ is clean (the sixth row of the table above), so this is specific to the
bindings and not to loading.

The crash is a jump to a null address inside a stream flush, and the stack names every layer:

    TensorRTBackend::init
      -> ET_LOG -> executorch::runtime::internal::logf -> vlogf   (in libexecutorch.so)
        -> std::cerr flush -> the sentry also flushes std::cout
          -> std::cout's stream buffer -> address 0

ExecuTorch's logging writes to the process-wide `std::cerr`, and `std::cerr` is tied to
`std::cout`, so every log line flushes `std::cout` as well. The bindings swap what those two
streams point at, per call, so that output reaches a Python stream. When a second thread is
inside a log call while that swap happens, it flushes a stream buffer that is being replaced and
calls through a pointer that is no longer there. An earlier reading of this as a deadlock was
looking at the same stack with a different ending: whether it hangs on the interpreter lock or
crashes on the swapped buffer depends on where the second thread happens to be.

Nothing in this backend can fix that, because a backend cannot log without going through
ExecuTorch's logging. The workaround is the same either way: load every program once on one
thread, then hand them out. The fix belongs in ExecuTorch, which should serialize its own logging
and should not redirect a process-wide stream while another thread may be writing through it.

A standalone program using TensorRT the same way from several threads, with no ExecuTorch and no
Python, runs eighty thousand cycles cleanly.

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
- `execute()` always returns with the work finished, whatever the memory it was
  given. The runtime this plugs into has no asynchronous execute: its own
  `execute()` returns to mean the work is done, and callers read the outputs
  straight afterwards. There was an opt-in that returned early, and it could not
  be honoured safely, so nothing returns in flight any more.
- With no guard active, the backend falls back to `cudaStreamPerThread`.
- The backend always waits for the engine before returning, whatever stream it ran on.
  ExecuTorch's runtime has no asynchronous execute: its `execute()` returns success to mean the
  work is finished, every caller reads the outputs straight after it returns, and the API hands
  back no event or future to wait on. There is therefore nobody an early return could be honest
  with, and no option to ask for one. An earlier revision of this backend had such an option and
  it is gone: on an idle GPU it returned zeros forty times out of forty, with the stream still
  unready for a median of 142 microseconds, growing to 38 milliseconds on a larger graph.
- The reference-runner smoke test runs inference inside a caller-stream guard on
  the discrete-GPU CI configuration, and it takes the staging path there. The backend
  binds a host pointer straight through only where the device both reads pageable host
  memory and does so through shared host page tables. A discrete card reports the first
  and not the second, because it serves pageable memory by faulting pages in one at a
  time, which measured 33 times slower than one bulk copy on a loop that rewrites its
  input each call. Either way that
  configuration returns with the work already finished, because the backend always
  waits.
  waits. CI separately asserts that the runner resolves one shared
  `libextension_cuda.so`.
- CUDA green-context streams work, and are the case this shared primitive exists
  for: one `cuGreenCtxStreamCreate` stream drives both the TensorRT delegate and
  ExecuTorch's CUDA/AOTI delegate, so both are confined to the same SM partition.
  That both honour it is a property of the code, not of a passing run: the CUDA
  delegate takes the per-thread stream when it initialises, then reads the caller's
  selection again on every execute and routes every kernel and boundary copy through
  it. Reading only the initialisation suggests the opposite, and a program split over
  two separate streams still returns the right numbers, so a matching result settles
  nothing either way.
  Exercised by hand on an A100 with 108 SMs, using a `.pte` whose graph splits across
  both delegates and a green context holding 8 of them. To reproduce, build the
  reference runner, whose CMake configuration always enables the CUDA delegate, and
  run it with `--green_context_sms=8`.

  CI runs the same option on the coalesced program, so this rests on more than the
  hand measurement. A device that cannot provide the partition makes the runner exit
  with its distinct status, and that case is skipped rather than failed, so a green
  context is only exercised where the device has the SMs for one.


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
