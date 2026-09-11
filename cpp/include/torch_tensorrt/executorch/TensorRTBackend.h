/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt. The processed blob uses the standalone wire format from
 * py/torch_tensorrt/executorch/serialization.py and is parsed directly here.
 * This runtime path intentionally does not depend on the legacy
 * Torch-TensorRT C++ runtime or libtorch.
 */
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/runtime/backend/interface.h>

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

struct TRTDeleter {
  template <typename T>
  void operator()(T* p) const {
    delete p;
  }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

class TRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

struct InputProfileBounds {
  nvinfer1::Dims min{};
  nvinfer1::Dims max{};
};

struct EngineHandle {
  TRTLogger logger;
  TRTUniquePtr<nvinfer1::IRuntime> runtime;
  TRTUniquePtr<nvinfer1::ICudaEngine> engine;
  TRTUniquePtr<nvinfer1::IExecutionContext> exec_ctx;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  std::vector<InputProfileBounds> input_profile_bounds;
  std::vector<void*> cached_input_ptrs;
  std::vector<size_t> cached_input_sizes;
  std::vector<void*> cached_output_ptrs;
  std::vector<size_t> cached_output_sizes;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  // Per output binding [0..num_outputs): index into input_binding_names of the
  // input it aliases (in-place KV-cache / user alias), or -1 for a normal output.
  // Built at init from the blob's aliased_io. The KV buffers are threaded by
  // ExecuTorch as caller-owned mutable-buffer delegate args (input AND aliased
  // output): execute() binds each aliased TRT output binding to its aliased
  // input's caller-provided pointer (in-place) and reflects the result into the
  // delegate output EValue, which ExecuTorch's write-back copy_ then reads.
  std::vector<int> output_aliased_input_idx;
  // Per input binding [0..num_inputs): true if any output aliases this input, so
  // its in-place (KV/user) update must land in the caller-owned storage. Built at
  // init from aliased_io; execute() uses it to reject a non-device-resident
  // aliased input instead of silently staging its update into delegate scratch.
  std::vector<bool> input_is_alias_target;
  size_t num_aliased_outputs = 0;
  int device_id = 0;
  bool unified_memory = false;
  // Whether exec_ctx was created kUSER_MANAGED, because
  // kSharedActivationScratchKey was on at this engine's load. Such a context takes
  // its activation scratch from the shared per-device pool on every call, unless
  // claims_pooled_scratch below is false, in which case the engine needs none and
  // never claims from the pool at all.
  bool shared_scratch = false;
  // Whether this handle takes its activation scratch from the pool. Set at init
  // only where shared_scratch above is on, and then only where the engine reports
  // needing activation scratch under some shape; a handle loaded with the option
  // off keeps this false and says nothing about what its engine needs, because it
  // has its own kSTATIC scratch either way. It is a predicate and not a size
  // because the size is never the right amount to install: enqueueV3 refuses a
  // kUSER_MANAGED context with no device memory once the engine needs any, however
  // little the shapes bound to a given call need, so execute() must hand such a
  // call some buffer -- but the engine's own figure covers every shape in the
  // profile, and installing it would size the pool for the largest of them.
  bool claims_pooled_scratch = false;
  std::mutex mu;
  // Makes the skip-sync fast path safe to reuse: TensorRT forbids reconfiguring or
  // destroying an execution context while one of its enqueues is in flight, so when
  // execute() returns without an end sync it records this event; the next execute()
  // and the destructor wait on it before touching exec_ctx. One event/flag pair
  // suffices because a handle runs on a single thread at a time.
  cudaEvent_t inflight_event = nullptr;
  bool inflight_pending = false;

  ~EngineHandle();
};

// Runtime backend option that backs execution-context activation scratch with a
// shared per-device pool instead of giving every context its own. Boolean,
// default false. Read by TensorRTBackend::set_option below, and delivered as
//   executorch::runtime::set_option("TensorRTBackend", options.view())
// A context's allocation strategy is fixed when the context is created, so a
// later call governs only the engines loaded after it, and a pooled context and
// a private-scratch one coexist in one process.
inline constexpr char kSharedActivationScratchKey[] = "use_shared_activation_scratch";

class TensorRTBackend final : public ::executorch::runtime::BackendInterface {
 public:
  bool is_available() const override;

  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec> compile_specs) const override;

  // Runs the engine. With an executorch::extension::cuda::CallerStreamGuard active and
  // no host staging required, this may return while the enqueue is still in flight on
  // the selected stream, so the caller must keep device buffers alive and unmodified
  // past return, order any other stream against this one, and synchronize the stream
  // before reading device-resident outputs. The selected stream must be on the engine's
  // device, and calls on one handle must not overlap each other or its destruction.
  // Other CUDA delegates sharing the same guard may instead synchronize before
  // returning, so do not assume results are ready on return from this one.
  // With the shared activation scratch pool (kSharedActivationScratchKey) one
  // buffer per device backs every context created while the option was on. Calls
  // on two such handles on one device may overlap: a per-device lock held across
  // the enqueue serializes them, so they do not run concurrently on the device. A
  // handle whose context was created while the option was off keeps its own
  // scratch and is not subject to this, nor is one whose engine reports needing no
  // activation scratch under any shape, which is left out of the pool. For the
  // handles that do draw on it, five further consequences:
  //   - A call needing more scratch than the pool holds grows it, and the growth
  //     has to get rid of the buffer it replaces before it returns, or the bytes
  //     of every size the pool ever grew to stay resident at once. Where that
  //     buffer has an enqueue recorded against it, the disposal begins with a host
  //     wait for that enqueue -- a whole inference, with no upper bound, paid
  //     whatever kind of call the growth is and whether or not it goes on to fail.
  //     Unbounded is meant literally: if that enqueue is itself waiting on
  //     something only this thread supplies once execute() returns -- a host
  //     function it will release, a copy it will enqueue next -- the call does not
  //     return, and the thread that would unblock it is the one inside the wait. A
  //     growth that reached its own enqueue therefore comes back with that engine
  //     work finished rather than in flight, since the event it waits on names that
  //     enqueue too. What the growth does not wait for is the rest of the device:
  //     the free is queued on a stream the pool owns and synchronizes itself, so
  //     work on streams this call never submitted to is not in the wait, and the
  //     bytes are back before it returns. Three things make the free a device-wide
  //     cudaFree instead, and then the call does block until the device is idle: a
  //     device with no stream-ordered allocator, where cudaFreeAsync reports
  //     cudaErrorNotSupported rather than freeing; any other code it reports, since
  //     a queued free that did not happen is not a free; and a disposal stream the
  //     pool could not create at all. Which calls grow the pool is not knowable
  //     from here; see the README.
  //   - A call that synchronizes the stream waits, through the handoff, for every
  //     pooled enqueue submitted before it on that device. That is what one buffer
  //     costs, and it is another unbounded wait: park a host function ahead of a
  //     pooled enqueue and release it only after a later execute() returns, and
  //     the later call is the one that does not return.
  //   - Capturing a CUDA graph around this delegate is not supported, with the
  //     option on or off. With it on, a call whose selected stream is capturing is
  //     refused with Error::NotSupported, ahead of every CUDA call it makes that a
  //     capture cannot take -- only the device query and the device switch run
  //     first. The pool's event handoff waits on an event recorded outside the
  //     capture, which invalidates it under every capture mode, and a growth's
  //     allocation and its disposal of the buffer it replaces invalidate it under
  //     every mode but cudaStreamCaptureModeRelaxed. The alternative to refusing is
  //     a capture that silently comes back invalidated. Where the query cannot be
  //     answered the call is not told it is capturing: cudaStreamIsCapturing also
  //     hands back a sticky fault left by earlier work on the device, and a call
  //     that meets one fails with that fault's own message and
  //     Error::InvalidProgram instead. Only a capture on the selected stream is
  //     caught; one running on any other stream under cudaStreamCaptureModeGlobal,
  //     or under cudaStreamCaptureModeThreadLocal from this thread, is invalidated
  //     by the same calls and is not refused, because CUDA offers no query for it.
  //     Do not run a pooled engine while capturing anywhere in the process. With
  //     the option off nothing refuses, because the check is inside the pooled
  //     path, but capture is unsupported there too: no call shape is built or
  //     tested for it. The README states this in full.
  //   - cudaDeviceReset() invalidates the pool without emptying it. The buffer,
  //     the handoff event and the disposal stream it still holds are destroyed
  //     with the primary context, and the next call on that device uses all three.
  //     There is no guard: do not reset a device this backend has run a pooled
  //     engine on.
  //   - The pool adds failure points to a call, on top of Error::NotSupported for
  //     the capture above. Only Error::Internal is its own: a pooled call returns
  //     it when the device's handoff event cannot be created, and an unpooled call
  //     never returns it at all. The rest reuse codes an unpooled call already
  //     returns from elsewhere in execute(), so the code alone does not say the
  //     pool was involved -- the log line does, and every one of them logs at Error
  //     first. Error::MemoryAllocationFailed if the shared buffer cannot be
  //     allocated or grown; Error::InvalidState if the wait ordering this call
  //     behind the previous enqueue fails, if TensorRT refuses the installed
  //     buffer, or if this call's enqueue cannot be recorded for the next claimant;
  //     Error::InvalidProgram if the capture query fails for a reason that is not a
  //     capture, or if the disposal's host wait fails -- and that last one leaks
  //     the retired buffer rather than freeing it under an enqueue that may still
  //     be reading it. The record and the disposal are the only two reached after
  //     this call's enqueue is submitted, and both synchronize the stream before
  //     returning, so neither leaves engine work in flight. Separately, a pooled
  //     call that staged a host-resident input synchronizes the stream on any error
  //     return it makes after that copy: the copy reads the caller's own memory and
  //     the caller owns that memory again the moment execute() returns. That too is
  //     a whole-stream wait, so it covers work the caller queued before calling. An
  //     unpooled call makes no such wait -- its error returns can leave that copy
  //     running, which is what they have always done.
  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  // Applies the runtime backend options a caller passes to
  // executorch::runtime::set_option("TensorRTBackend", ...). The only key read is
  // kSharedActivationScratchKey, a boolean.
  ::executorch::runtime::Error set_option(
      ET_UNUSED ::executorch::runtime::BackendOptionContext& context,
      const ::executorch::runtime::Span<::executorch::runtime::BackendOption>& backend_options) override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
