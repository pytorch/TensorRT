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
all, and the devices differ: three of the four tested report that they can read pageable host
memory and
one does not, so the same program that works on a discrete card can fail on an integrated one.

### Loading the same program many times grows host memory

A program whose graph is split across this delegate and ExecuTorch's CUDA backend grows the
process by roughly 390 to 415 kilobytes for every load, run and drop cycle. Measured on three
machines and two architectures, each time as a slope fitted over hundreds of cycles rather than
as the difference between two readings, and the three agree within about six percent.

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

### Running from several threads at once does not work today

Measured on three GPUs, two architectures. Loading and running from more than one thread fails
consistently: on one discrete card two threads failed twenty one times out of twenty one, and on an
integrated part every variant failed five times out of five, including the variant where all threads
share a single loaded program. A separate case hangs rather than crashing.

The cause is a deadlock, and it has been narrowed to a stack. TensorRT calls this backend's
logger from
whichever thread it is initialising on, that logger writes through ExecuTorch's logging, and
under the
Python bindings that logging is redirected into a Python text stream, whose flush needs the
interpreter
lock. A thread that does not hold the lock waits for it there and never returns:

    TRTLogger::log -> ET_LOG -> std::ostream -> the bindings' stream redirect
                   -> TextIOWrapper flush -> acquire the interpreter lock

Two measurements pin it down. Loading each program under a lock, so only one is ever
initialising, fixes
it: zero failures in five. Running a program once before any thread starts, so initialisation is
already
done, also fixes it: zero failures in five. Silencing standard error does not, in any of three ways,
because the redirect is inside the process rather than at the file descriptor.

So the practical workaround is to load and run each program once on one thread, and only then
hand it to
several. Sharing one program does not help on its own, which is why an earlier reading of this as a
problem with several programs at once was wrong.

A standalone program using TensorRT the same way from several threads, with no ExecuTorch and no
Python,
runs eighty thousand cycles cleanly. The fix belongs in the bindings, which should not take the
interpreter lock on a thread the runtime owns.

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
  outputs are directly bindable, so device, managed, or unified memory),
  `execute()` may return with the TensorRT enqueue still in flight on the
  stream (no end-of-execute sync). The backend orders the next `execute()` and
  the handle's destruction after that work via an internal completion event, but
  that event only protects backend-owned state. The caller must therefore keep
  all directly bound input/output storage alive and unmodified until the work is
  complete, order any cross-stream producers/consumers with their own events,
  and synchronize the stream before reading outputs on the host.
- With no guard active, the backend falls back to `cudaStreamPerThread`.
- The backend always waits for the engine before returning, whatever stream it ran on.
  ExecuTorch's runtime has no asynchronous execute: its `execute()` returns success to mean the
  work is finished, every caller reads the outputs straight after it returns, and the API hands
  back no event or future to wait on. There is therefore nobody an early return could be honest
  with, and no option to ask for one. An earlier revision of this backend had such an option and
  it is gone: on an idle GPU it returned zeros forty times out of forty, with the stream still
  unready for a median of 142 microseconds, growing to 38 milliseconds on a larger graph.
- The reference-runner smoke test runs inference inside a caller-stream guard on
  the discrete-GPU CI configuration. Host-backed input and output do not imply the
  staging path there: the backend binds a host pointer straight through whenever the
  device reports that it can read pageable host memory, which a discrete H100 does,
  and stages it only on a device that reports it cannot. Either way that
  configuration returns with the work already finished, because the backend always
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
