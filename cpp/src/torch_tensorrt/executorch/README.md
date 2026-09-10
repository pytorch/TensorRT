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
requirements, which previously lived on the removed `CudaStreamGuard`:

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
  they may be submitted concurrently from two threads, but they will not run
  concurrently on the device, so the pool costs the parallelism between them.
  Contexts created while the option was off keep their own scratch and are
  unaffected, and so is a context whose engine needs no activation scratch under
  any shape: that one is left out of the pool and serializes against nothing.
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
`use_shared_activation_scratch` backend option (a boolean, off by default)
instead backs a device's contexts from one buffer, grown to the largest
requirement any call on that device has asked for. Which contexts: those created
while the option was on, less any whose engine needs no scratch under any shape,
which are left out of the pool entirely. The two exceptions are spelled out
below.

```cpp
#include <executorch/runtime/backend/interface.h>

executorch::runtime::Error enable_shared_activation_scratch() {
  executorch::runtime::BackendOptions<1> options;
  // Returns Error::InvalidArgument when the object has no room left. One key in a
  // BackendOptions<1> always fits, so this cannot fail as written; the check is
  // what keeps it honest when a second key is added beside it.
  const executorch::runtime::Error stored =
      options.set_option("use_shared_activation_scratch", true);
  if (stored != executorch::runtime::Error::Ok) {
    return stored;
  }
  // Nothing was set if this fails: every context still allocates its own scratch.
  return executorch::runtime::set_option("TensorRTBackend", options.view());
}
```

`Error::NotFound` means no backend is registered under that name, which is what a
binary that has not linked the backend archive gets. Nothing forces the check: the
free `executorch::runtime::set_option` is not `ET_NODISCARD`, so dropping its
return compiles.

N per-engine copies collapse to one, so the reclaimed memory is the sum of the N
requirements less the largest of them. Set the option before loading the methods
whose engines should use the pool, and read the `use_shared_activation_scratch`
bullet of the caller-stream contract above: engines sharing a buffer do not run
concurrently on the device. The pool never shrinks, so the largest scratch it was
ever asked for stays allocated until the process exits.

A call that asks for nothing never grows a pool that already holds a buffer: it is
handed that buffer, whatever size it is. Against an empty pool it does allocate,
which the paragraph after next is about. It is not handed the pool's size with it
either: what the context is told it owns is the figure this call asked for, so a
zero becomes the one-byte minimum below and TensorRT is told the buffer is one
byte, whatever the pool's capacity is. Two of the things a zero from the per-shape
query can mean reach this point, and are indistinguishable where it is read: the
shapes bound to this call need none -- an empty batch inside a profile that admits
one -- or the query failed. Neither wants a buffer of its own.

Such a call cannot be left with no buffer at all. `enqueueV3` refuses a
`kUSER_MANAGED` context with no device memory installed as soon as the engine
reports needing some under *any* shape, however little the shapes actually bound
need. So where the pool holds nothing yet, a zero allocates the smallest buffer
that satisfies that check, and the first call with a real requirement grows it.
Standing the engine's own figure in for the zero would satisfy the check too, but
that figure covers the whole profile: on the dynamic engine the tests use it is
4 MiB against the 128 KiB the next call needs, and the pool never shrinks.

Only one of the two causes is safe to hand a buffer that small, and the install is
what separates them. `setDeviceMemoryV2` refuses a buffer smaller than the bound
shapes need, and it returns `void`: where the shapes genuinely need nothing the
expected size is zero and the one-byte buffer is accepted, but where the query
failed the engine expects what it always did, the install is refused, and the
context keeps the buffer it was last given -- which a growth may already have
freed, so the enqueue would read and write memory the pool no longer owns while
`enqueueV3` reports success. The backend reads the refusal back through a TensorRT
`IErrorRecorder` scoped to that one call and fails with `Error::InvalidState`
rather than enqueueing. Nothing reaches `enqueueV3` on an install TensorRT
rejected.

What each call installs is its own requirement, not the capacity the pool holds,
which is larger whenever an earlier call asked for more. Handing a context the
larger figure would say it owns bytes that belong to whatever ran before it, and
the pool never clears the buffer between users -- activation scratch is written
before it is read, so a context that stays inside its own requirement never sees
those bytes, and one that runs past it sees another engine's activations rather
than zeros. The smaller figure also keeps the refusal above sharp: after a growth
the capacity may well cover what a failed query concealed, and an install sized
from it would be accepted.

The third thing a zero can mean is an engine that needs no scratch under *any*
shape, and that one is settled before a call is ever made: such an engine is left
out of the pool entirely, so it takes no per-device lock and does not serialize
against the engines that do.

The buffer grows when a call asks for more than every call before it did, and the
growth has to get rid of the buffer it replaces *before `execute()` returns*.
Not disposing of it inside the call is not an option: the pool grows monotonically,
so a run that let each retired buffer outlive its growth would end up holding the
sum of every size the pool was ever grown to rather than the largest of them.

Every growth gets rid of it the same way, and none of the three steps reads
anything about the call the growth was made from: wait on the host for the enqueue
the handoff event names, queue a `cudaFreeAsync` on a stream the pool owns for that
device, and synchronize that stream, which is what returns the bytes. All three are
made with the per-device lock dropped, so none of them is serialized behind the
pool's own lock and none holds the next claimant off while it runs: two pooled calls
serialize at submission, as the caller-stream contract above says, and a growth is
not an exception to it.

**The host wait is the thing to know about**, because it is what a growth pays
whenever the buffer it replaces has an enqueue recorded against it. It is a whole
inference, and it has no upper bound: whatever the enqueue is waiting for holds it
too, so if a host function ahead of that enqueue will be released only by this
thread once `execute()` returns, the growing call does not return and the thread
that would release it is the one inside the wait.

The event it waits on is the handoff, and a call that reached its own enqueue has
recorded that enqueue there before it releases the claim, so the wait covers this
call's inference as well as the one before it. A growth therefore returns with its
engine work finished rather than in flight, however asynchronous the caller-stream
contract above lets a call that does not grow the pool be, and anything `execute()`
queues after the release -- an aliased reflect, a D2H copy -- goes behind a
completed enqueue rather than a running one. Only a growth pays that; a call with
nothing retired makes no CUDA call in the release at all. A growth whose retired
buffer never had an enqueue recorded against it does not pay it either: the
handoff carries no recording, so there is nothing to wait for and the disposal
goes straight to the free. A pooled call that returned between taking the buffer
and recording its enqueue -- a refused install, a refused `enqueueV3` -- is what
leaves the pool in that state.

Sharing one buffer has an unbounded wait of its own, growth or no growth. A call
that synchronizes waits, through the handoff, for every pooled enqueue submitted
before it on that device. Park a host function ahead of one pooled call's enqueue
and release it only once a later pooled call has returned, and the later call is
the one that never returns.

What a growth does *not* wait for is the rest of the device, and that is what the
pool's own stream buys. A device-wide `cudaFree` waits for everything queued on the
device rather than for the work that touched the buffer. Measured on an A100 with
CUDA 13.0 and driver 13.0, freeing a 512 MiB buffer behind a 10 ms enqueue with a
host function parked for 1500 ms on a stream nothing else in the measurement
submitted to: 1500.8-1501.4 ms device-wide against 10.6-11.7 ms on the pool's
stream, both with every byte back by the time the disposal returned. That wait is
not hypothetical: the backend's own tests deadlocked on the device-wide free once,
when a change of test order turned a case that parks a host function on its own
stream into the one that grew the pool.

The stream has to be the pool's rather than the calling engine's. A `cudaFreeAsync`
of a `cudaMalloc`'d pointer -- which is what this pool holds -- hands the bytes
back at the next `cudaStreamSynchronize` of the stream it was queued on and at no
point before it. Measured on the same machine: it returns `cudaSuccess` and defers
the free, a stream `cudaStreamQuery` reports as drained still holds the bytes,
`cudaMemGetInfo` does not move, and a `cudaMalloc` of the same size fails with out
of memory until the synchronize. So whoever queues the free has to be able to
promise that synchronize, and only the pool can. Queuing it on the caller's stream
costs the disposal nothing, 0.0 ms in the same measurement, because all it does is
queue; the
whole 512 MiB was still resident when it returned, and the caller's own synchronize
is what gives it back. Where the caller makes one, that is no faster overall:
10.4-11.8 ms to the same point. Where it does not -- the caller-stream,
device-resident case, which is the one this option exists for -- the bytes never
come back at all: nothing this backend does will ever synchronize that stream, and
the stream is the caller's to destroy. Measured, a 4 GiB block queued
that way and then abandoned with the stream is gone for the life of the process, on
a device with tens of gigabytes still free: not at `cudaStreamDestroy`, not at
`cudaDeviceSynchronize`, not from another stream, and not at `cudaMemPoolTrimTo`.
Replaying the pool's own growth sequence on one such stream -- 8, 16, 24 and 32 GiB,
no synchronize between them -- four growths that all succeed when the disposal
returns the bytes fail on the fourth with out of memory, free stuck at 30.83 GiB
on a device that started with 78.83 GiB of it.

That last is a property of the pointer rather than of stream-ordered frees in
general, so it is worth saying which one this pool holds. Replayed with
`cudaMallocAsync` on both sides of the same never-synchronized stream, all four
growths succeed: the free moves from 54.83 to 38.83 GiB for the 24 GiB request,
the retired 16 GiB reused with nobody having synchronized anything. Those bytes go
back to the device's memory pool rather than to the driver, and the next
allocation can take them from there. The pool allocates with `cudaMalloc`, so the
shape above is the one it is in, and the disposal stream is what that shape costs.

Which calls do synchronize their stream is knowable, and the disposal does not ask.
It is a claim about what the caller has still to do rather than about anything the
pool holds, and three defects on this path in a row were a disposal getting that
claim wrong -- the last of them a call that failed after the growth and so had no
synchronization left to make, having recorded at the claim that it would make one.
So the disposal takes no argument: there is nothing to record when the buffer is
retired and nothing a later return can leave stale.

A device-wide `cudaFree`, and the wait for the whole device that comes with it, is
still the fallback. It is what a growth makes where the device has no
stream-ordered allocator (`cudaDevAttrMemoryPoolsSupported`) and `cudaFreeAsync`
reports `cudaErrorNotSupported` instead of freeing -- and on any other code it
reports, since a queued free that did not happen is not a free -- and where the
pool's stream could not be created at all. A program that parks work across an
`execute()` has to keep such growths away from it, and the paragraph below on how
often the pool grows is what says whether a run order can do that.

What an engine answers when asked how much it needs is decided when it is built,
not when it runs. The builder's `kRUNTIME_ACTIVATION_RESIZE_10_10` preview feature
makes an engine report what the shapes just bound need; without it, whether an
engine does that or reports its profile maximum depends on how TensorRT planned
it. Either way the pool can settle well above the live data, and nothing the
runtime does changes it.

How often the pool grows follows from that. The backend asks afresh on every
`execute()`, after the input shapes are bound, so an engine whose answer does not
vary with the bound shapes grows the pool at most once, on its first run. For a
program built only from such engines, running the largest one first leaves the
pool with a single allocation. An engine whose answer does vary can grow it on any
call whose shapes need more than every call before them, so with one of those in
the program no run order bounds the number of allocations. One engine over a
`[1..4, 512, 512]` profile is either, depending on how it was built.

### CUDA graph capture is not supported

A pooled call whose selected stream is capturing is refused with
`Error::NotSupported`. The handoff between one enqueue and the next waits on an
event recorded outside the capture, and `cudaStreamWaitEvent` on such an event
fails with `cudaErrorStreamCaptureIsolation` and invalidates the capture under
every capture mode, `cudaStreamCaptureModeRelaxed` included. A growth adds more of
the same: its `cudaMalloc`, and the host wait its disposal of the replaced buffer
begins with, invalidate the capture under `Global` and `ThreadLocal`, and under
`Relaxed` are permitted but run uncaptured. Left to run, the caller learns of any
of it only when `cudaStreamEndCapture` hands back
`cudaErrorStreamCaptureInvalidated` and a null graph, long after the call that
caused it.

The check sits ahead of everything `execute()` does that a capture cannot take,
not merely ahead of the pool's own calls: the wait on a previous enqueue and the
`cudaMalloc` that grows a host-input staging buffer come before those and, outside
`Relaxed`, would invalidate the capture first. Only the device query and the
device switch run before the check, and a capture takes both.

Where the query fails for a reason that is not a capture, the call is not told it
is capturing. `cudaStreamIsCapturing` also hands back a sticky fault left by
earlier work on the device -- measured on an A100 with CUDA 12.8, after an illegal
memory access it returns that fault with the capture status still
`cudaStreamCaptureStatusNone` -- and a call that meets one fails with
`Error::InvalidProgram` under that fault's own message instead. The one query
failure that does still mean a capture, `cudaErrorStreamCaptureImplicit`, is
refused with the rest.

Only the selected stream is checked. A capture running on another stream under
`Global`, or under `ThreadLocal` from the calling thread, is invalidated by the
same calls and is not refused, because CUDA has no query for "is a capture live in
this process". So that much is the caller's to keep: do not run a pooled engine
while any capture is open anywhere in the process. Turning the option off removes
the refusal, since the check lives inside the pooled path, but this delegate does
not support capture with the option off either. No call shape here is built or
tested for it, and the completion event `execute()` records so that the next call
can wait for an enqueue that outlived the return would become a node of the graph
rather than an event the host can wait on.

### cudaDeviceReset() is not survivable

The pool is not guarded against it: it holds its buffer, its handoff event and the
stream its growths dispose on for the process lifetime, and a reset destroys the
primary context under all three. The next pooled `execute()` on that device then
waits on a destroyed event and hands the engine a pointer that is no longer a
device allocation, and the next growth after that queues its free on a stream that
no longer exists. Guarding this would mean revalidating all three on every call,
and the check that would catch it is as expensive as the work it protects. Treat a
device the backend has run a pooled engine on as one that must not be reset.

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
