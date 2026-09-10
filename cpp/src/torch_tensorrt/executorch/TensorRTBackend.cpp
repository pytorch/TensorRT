/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_tensorrt/executorch/TensorRTBackend.h"
#include "torch_tensorrt/executorch/PooledScratchInstall.h"
#include "torch_tensorrt/executorch/SharedScratchPool.h"
#include "torch_tensorrt/executorch/TensorRTBindingNames.h"
#include "torch_tensorrt/executorch/TensorRTBlobHeader.h"
#include "torch_tensorrt/executorch/WeightStreamingBudget.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/extension/cuda/caller_stream.h>
#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/exec_aten/util/tensor_util.h>
#include <executorch/runtime/platform/log.h>

namespace torch_tensorrt {
namespace executorch_backend {

using ::executorch::aten::SizesType;
using ::executorch::runtime::ArrayRef;
using ::executorch::runtime::BackendExecutionContext;
using ::executorch::runtime::BackendInitContext;
using ::executorch::runtime::BackendOption;
using ::executorch::runtime::BackendOptionContext;
using ::executorch::runtime::CompileSpec;
using ::executorch::runtime::DelegateHandle;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::FreeableBuffer;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

#define TORCHTRT_ET_CHECK_NOT_NULL(VALUE, ERROR_CODE, ...) \
  do {                                                     \
    if ((VALUE) == nullptr) {                              \
      ET_LOG(Error, __VA_ARGS__);                          \
      return ERROR_CODE;                                   \
    }                                                      \
  } while (false)

namespace {

extern const Error kRegistrationResult;

Error check_registration() {
  if (kRegistrationResult != Error::Ok) {
    ET_LOG(Error, "TensorRTBackend registration failed: %s", ::executorch::runtime::to_string(kRegistrationResult));
  }
  return kRegistrationResult;
}

} // namespace

void TRTLogger::log(Severity severity, const char* msg) noexcept {
  if (severity <= Severity::kERROR) {
    ET_LOG(Error, "TensorRT: %s", msg);
  } else if (severity == Severity::kWARNING) {
    ET_LOG(Info, "TensorRT warning: %s", msg);
  }
}

EngineHandle::~EngineHandle() {
  cudaSetDevice(device_id);
  // No wait here: execute already waited, and a device-wide one blocks unrelated work.
  for (void* p : cached_input_ptrs) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
  for (void* p : cached_output_ptrs) {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
  exec_ctx.reset();
  engine.reset();
  // The runtime is shared and outlives this handle, so there is nothing to release for it.
}

namespace {

// The process-wide TensorRT runtime and its logger, built once on first use and then never
// destroyed. TensorRT requires the runtime to outlive every engine deserialized from it, and never
// destroying it is the only way to promise that here. Destroying it at exit would not: statics are
// torn down in reverse order of when their construction FINISHED, and an application holding a
// program in a global finishes that global's constructor during static initialization, before
// anything has called this. The runtime would then be destroyed first and the engine second, which
// is the wrong way round. The cost is one runtime and one logger still allocated at exit.
nvinfer1::IRuntime* shared_runtime() {
  static std::mutex mutex;
  static TRTLogger* logger = nullptr;
  static nvinfer1::IRuntime* runtime = nullptr;
  const std::lock_guard<std::mutex> lock(mutex);
  // Retried while null, so one transient failure does not disable the backend for the process.
  if (runtime == nullptr) {
    if (logger == nullptr) {
      logger = new TRTLogger();
    }
    runtime = nvinfer1::createInferRuntime(*logger);
  }
  return runtime;
}

// Drains on an early exit, so a retry cannot land in a staging buffer still being written.
class StreamDrainOnEarlyReturn {
 public:
  explicit StreamDrainOnEarlyReturn(cudaStream_t stream) : stream_(stream) {}
  StreamDrainOnEarlyReturn(const StreamDrainOnEarlyReturn&) = delete;
  StreamDrainOnEarlyReturn& operator=(const StreamDrainOnEarlyReturn&) = delete;
  ~StreamDrainOnEarlyReturn() {
    if (armed_) {
      (void)cudaStreamSynchronize(stream_);
      cudaGetLastError();
    }
  }
  void arm() {
    armed_ = true;
  }
  void disarm() {
    armed_ = false;
  }

 private:
  cudaStream_t stream_;
  bool armed_ = false;
};

struct EngineHandleDeleter {
  void operator()(EngineHandle* handle) const {
    if (handle != nullptr) {
      handle->~EngineHandle();
    }
  }
};

nvinfer1::Dims to_trt_dims(const exec_aten::Tensor& t) {
  nvinfer1::Dims dims{};
  dims.nbDims = t.dim();
  if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
    return dims;
  }
  for (int d = 0; d < t.dim(); ++d) {
    dims.d[d] = static_cast<int64_t>(t.size(d));
  }
  return dims;
}

bool infer_binding_names(
    nvinfer1::ICudaEngine* engine,
    std::vector<std::string>& inputs,
    std::vector<std::string>& outputs) {
  if (engine == nullptr) {
    return false;
  }

  detail::TensorRTBindingNames binding_names;
  if (!detail::infer_engine_binding_names(*engine, binding_names)) {
    return false;
  }

  inputs = std::move(binding_names.input_names);
  outputs = std::move(binding_names.output_names);
  return true;
}

// The setting behind kSharedActivationScratchKey: whether an execution context
// created subsequently draws its activation scratch from the shared per-device
// pool rather than allocating its own.
//
// execute() must read EngineHandle::claims_pooled_scratch, never this. This can
// have moved since the engine was loaded, and what the engine can do about it
// cannot: its context's allocation strategy was fixed when the context was
// created. initialize_engine_io below is the one place this is read.
std::atomic<bool> scratch_enabled{false};

Error initialize_engine_io(EngineHandle& handle) {
  if (handle.input_binding_names.empty() && handle.output_binding_names.empty() &&
      !infer_binding_names(handle.engine.get(), handle.input_binding_names, handle.output_binding_names)) {
    ET_LOG(Error, "TensorRTBackend::init: failed to infer TensorRT binding names");
    return Error::InvalidProgram;
  }

  handle.num_inputs = handle.input_binding_names.size();
  handle.num_outputs = handle.output_binding_names.size();

  // Against the engine's own count, since a metadata list that lost an entry still parses.
  const int32_t engine_io_count = handle.engine->getNbIOTensors();
  const size_t named_io_count = handle.num_inputs + handle.num_outputs;
  if (engine_io_count < 0 || named_io_count != static_cast<size_t>(engine_io_count)) {
    ET_LOG(
        Error,
        "TensorRTBackend::init: the program names %zu bindings (%zu in, %zu out) but the engine "
        "has %d. The metadata and the engine disagree, so the program is damaged rather than "
        "merely unsupported.",
        named_io_count,
        handle.num_inputs,
        handle.num_outputs,
        engine_io_count);
    return Error::InvalidProgram;
  }

  handle.exec_ctx.reset(handle.engine->createExecutionContext());
  // kSTATIC gives the context its own activation scratch; kUSER_MANAGED makes it
  // allocate none and take a buffer from execute() instead. The strategy is fixed
  // at creation, so it is captured on the handle here rather than read per call.
  handle.shared_scratch = scratch_enabled.load(std::memory_order_relaxed);
  const auto strategy = handle.shared_scratch ? nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED
                                              : nvinfer1::ExecutionContextAllocationStrategy::kSTATIC;
  handle.exec_ctx.reset(handle.engine->createExecutionContext(strategy));
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle.exec_ctx, Error::InvalidProgram, "TensorRTBackend::init: failed to create TensorRT execution context");

  if (handle.shared_scratch) {
    // Gated so an engine loaded with the option off pays no TensorRT call for a
    // pool it will never claim from. Read after the weight streaming budget is
    // applied, which the caller does before this runs because TensorRT forbids
    // moving the budget once a context exists -- and the budget is the one thing
    // that moves this figure.
    handle.claims_pooled_scratch = handle.engine->getDeviceMemorySizeV2() > 0;
  }

  return Error::Ok;
}

Error initialize_input_profiles(EngineHandle& handle) {
  for (const auto& name : handle.input_binding_names) {
    if (handle.engine->isShapeInferenceIO(name.c_str())) {
      ET_LOG(Error, "TensorRTBackend::init: shape tensor input '%s' is not supported", name.c_str());
      return Error::InvalidProgram;
    }
  }

  handle.input_profile_bounds.reserve(handle.num_inputs);
  for (const auto& name : handle.input_binding_names) {
    InputProfileBounds bounds;
    bounds.min = handle.engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
    bounds.max = handle.engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    if (bounds.min.nbDims < 0 || bounds.max.nbDims < 0) {
      ET_LOG(Error, "TensorRTBackend::init: getProfileShape failed for input '%s'", name.c_str());
      return Error::InvalidProgram;
    }
    handle.input_profile_bounds.push_back(bounds);
  }

  return Error::Ok;
}

bool is_cuda_accessible_ptr(const void* ptr) {
  if (ptr == nullptr) {
    return false;
  }
  cudaPointerAttributes attrs{};
  const cudaError_t err = cudaPointerGetAttributes(&attrs, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attrs.type == cudaMemoryTypeDevice || attrs.type == cudaMemoryTypeManaged;
}

// The device a pointer is tied to, or -1 for host, managed, and this engine's own.
int cuda_foreign_device_of_ptr(const void* ptr, int engine_device) {
  if (ptr == nullptr) {
    return -1;
  }
  cudaPointerAttributes attrs{};
  // Let through: an older runtime reports plain host memory as an outright failure.
  if (cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess) {
    cudaGetLastError();
    return -1;
  }
  if (attrs.type != cudaMemoryTypeDevice || attrs.device == engine_device) {
    return -1;
  }
  // A non-null devicePointer is an address usable from here, which is the question. Asking instead
  // whether the two cards CAN reach each other answered yes on every pair measured and still faulted,
  // because a mapping has to be enabled per pair; refusing outright then rejected buffers that work
  // once it is.
  if (attrs.devicePointer != nullptr) {
    return -1;
  }
  return attrs.device;
}

// A caller's hold on one device's shared scratch: the device lock, plus the
// buffer a growth displaced, freed once that lock is dropped.
//
// The lock spans the enqueue, not just the choice of buffer. A claimant that
// released it as soon as it had a buffer would leave its enqueue live for a
// window the marker's event does not yet cover, and a second claimant entering
// that window is handed the same buffer and told to wait for the enqueue before
// it -- so nothing orders the two and both write the same scratch. The failure
// is silent: wrong output, no CUDA error, no TensorRT error.
//
// This lock nests inside the per-handle EngineHandle::mu, which already spans
// the enqueue, and is never taken in the other order.
class SharedScratchClaim {
 public:
  SharedScratchClaim() = default;
  SharedScratchClaim(const SharedScratchClaim&) = delete;
  SharedScratchClaim& operator=(const SharedScratchClaim&) = delete;
  ~SharedScratchClaim() {
    // A claim that still holds a retired buffer here was never released, so
    // execute() returned between the claim and the release below -- a refused
    // install, a refused enqueue, or a failed record of one already submitted.
    // The disposal it makes is the one every other path makes, and it has the
    // bytes back before it returns, so a bail-out after a growth costs the
    // retired buffer and not the whole pool.
    //
    // The return is the caller's to report, and on this path there is no caller
    // left to report it to: an execute() that returned early has already failed.
    (void)release();
  }

  SharedScratchDevice& hold(int device_id) {
    dev_ = &scratch_pool().get(device_id);
    device_id_ = device_id;
    lock_ = std::unique_lock<std::mutex>(dev_->mu);
    return *dev_;
  }

  // Null until hold() runs and null again after release(): non-null exactly while
  // this claim holds the device's lock.
  SharedScratchDevice* device() const {
    return dev_;
  }

  // Takes ownership of a buffer a growth displaced, to be freed by release().
  // `wait_for` is the marker event the enqueues that used it were recorded on, or
  // null if none were; `disposal_stream` is the pool's own stream for this device,
  // or null if one could not be created.
  void retire(void* buffer, cudaEvent_t wait_for, cudaStream_t disposal_stream) {
    retired_ = buffer;
    retired_wait_ = wait_for;
    disposal_stream_ = disposal_stream;
  }

  // Drops the lock, and the device pointer with it so device() cannot hand out a
  // pointer this claim no longer holds the lock for. Then disposes of whatever a
  // growth displaced, after the unlock, because that disposal waits for an enqueue
  // another pooled engine on this device has nothing to do with.
  //
  // The disposal reads nothing the caller supplies and asks nothing about it: wait
  // on the host for the enqueue the marker names, queue the free on the pool's own
  // stream for this device, and synchronize that stream, which is what has the
  // bytes back before this returns. It is the same three steps on every path,
  // including the destructor's, so a return added between the claim and the
  // release cannot change what the disposal does. Deciding instead from what the
  // calling execute() has still to do is a claim about the caller, and three
  // defects on this path in a row were a disposal getting that claim wrong.
  //
  // The free is stream-ordered, and on the pool's own stream, for two reasons. A
  // device-wide cudaFree waits for everything queued on the device rather than for
  // the work that touched the buffer, and that wait has no upper bound. And a
  // cudaFreeAsync of a cudaMalloc'd pointer, which is what the pool holds, hands
  // the bytes back at the next synchronize of the stream it was queued on and at
  // no point before it, so whoever queues the free has to be able to promise that
  // synchronize -- which a caller on the path this pool exists for cannot, since it
  // never synchronizes its stream and is free to destroy it the moment execute()
  // returns. The README's shared activation scratch section has the measurements
  // behind both.
  //
  // cudaFreeAsync needs the device's stream-ordered allocator, which not every
  // platform has. Where it is missing, or the pool's stream could not be created,
  // the free is device-wide instead and blocks until the device is idle.
  //
  // Returns false when the wait for the enqueue failed, which leaves the buffer
  // leaked rather than freed under a live enqueue; the caller reports it. Frees on
  // the current device, which must still be the buffer's.
  bool release() {
    void* const retired = retired_;
    const cudaEvent_t wait_for = retired_wait_;
    const cudaStream_t disposal_stream = disposal_stream_;
    // Clearing the pointer is what stops the destructor's release from disposing
    // of the same buffer twice.
    retired_ = nullptr;
    if (lock_.owns_lock()) {
      lock_.unlock();
    }
    dev_ = nullptr;
    if (retired == nullptr) {
      return true;
    }
    if (!wait_for_the_enqueue_that_used_it(wait_for)) {
      return false;
    }
    if (disposal_stream == nullptr || !free_on_the_pools_stream(retired, disposal_stream)) {
      free_device_wide(retired);
    }
    return true;
  }

 private:
  // Waits for the enqueue that last used the retired buffer, so the free below is
  // not made under one still reading it. `wait_for` is the device's handoff marker,
  // so where the calling execute() has already recorded its own enqueue there, the
  // wait covers that one as well. Unbounded either way -- it is a whole inference,
  // and the execute() contract says as much.
  bool wait_for_the_enqueue_that_used_it(cudaEvent_t wait_for) {
    if (wait_for == nullptr) {
      return true;
    }
    const cudaError_t wait_err = cudaEventSynchronize(wait_for);
    if (wait_err != cudaSuccess) {
      // This wait is the only thing keeping the free off a buffer an enqueue may
      // still be reading, so a failed wait leaks it instead. What it reports is
      // usually an asynchronous fault raised by earlier work on this device.
      ET_LOG(
          Error,
          "TensorRTBackend::execute: waiting for the enqueue on the replaced shared activation scratch on device %d failed (%s), which for a wait on device work is usually an earlier asynchronous fault on this device surfacing here; leaking that buffer rather than freeing it under a live enqueue",
          device_id_,
          cudaGetErrorString(wait_err));
      cudaGetLastError();
      return false;
    }
    return true;
  }

  // Queues the free on the pool's stream for this device and synchronizes it,
  // which is what returns the bytes. Reports whether the buffer was freed, so a
  // refusal falls back to the device-wide free; a synchronize that fails after the
  // free was accepted is not one, and freeing again would be freeing twice.
  bool free_on_the_pools_stream(void* retired, cudaStream_t disposal_stream) {
    const cudaError_t free_err = cudaFreeAsync(retired, disposal_stream);
    if (free_err != cudaSuccess) {
      // The error is this call's own, so it is cleared here rather than left for
      // the next CUDA call in execute() to report under its own name.
      cudaGetLastError();
      if (free_err == cudaErrorNotSupported) {
        ET_LOG(
            Info,
            "TensorRTBackend::execute: device %d has no stream-ordered allocator, so the free of the shared activation scratch buffer a pool growth replaced falls back to a device-wide free, which blocks this call until the device is idle",
            device_id_);
      } else {
        // Any other code is a fault on a device that does have the allocator, so
        // this does not say the platform lacks one.
        ET_LOG(
            Info,
            "TensorRTBackend::execute: the stream-ordered free of the shared activation scratch buffer a pool growth replaced on device %d returned %s, so it falls back to a device-wide free, which blocks this call until the device is idle",
            device_id_,
            cudaGetErrorString(free_err));
      }
      return false;
    }

    const cudaError_t sync_err = cudaStreamSynchronize(disposal_stream);
    if (sync_err != cudaSuccess) {
      // Nothing was on this stream but the free, and the enqueue it had to follow
      // was already waited for, so a failure here is a fault this device was
      // already in. Whether the bytes came back is not knowable from it.
      ET_LOG(
          Error,
          "TensorRTBackend::execute: synchronizing the shared activation scratch pool's disposal stream on device %d reported %s, so the free of the buffer a growth replaced may not have returned its bytes",
          device_id_,
          cudaGetErrorString(sync_err));
      cudaGetLastError();
    }
    return true;
  }

  void free_device_wide(void* retired) {
    const cudaError_t err = cudaFree(retired);
    if (err != cudaSuccess) {
      // cudaFree synchronizes, so what it reports is more often an earlier
      // asynchronous fault on this device than a fault in the free -- which is
      // why the message does not call it one.
      ET_LOG(
          Error,
          "TensorRTBackend::execute: freeing the shared activation scratch buffer that a pool growth replaced on device %d reported %s; a device-wide free reports whatever fault this device is already in, so this need not be the pool's",
          device_id_,
          cudaGetErrorString(err));
      // Clears a non-sticky error so it does not resurface under the name of the
      // next CUDA call in execute(). A sticky one survives the clear and will
      // resurface anyway; the caller learns of it from that call.
      cudaGetLastError();
    }
  }

  SharedScratchDevice* dev_ = nullptr;
  int device_id_ = -1;
  std::unique_lock<std::mutex> lock_;
  void* retired_ = nullptr;
  cudaEvent_t retired_wait_ = nullptr;
  cudaStream_t disposal_stream_ = nullptr;
};

// What a call needing no activation scratch is given when the pool holds nothing
// yet. It cannot be given nothing: enqueueV3 refuses a kUSER_MANAGED context with
// no device memory installed as soon as the engine reports needing any under some
// shape, whatever the shapes actually bound need. So the pool starts at the
// smallest allocation that satisfies that check and the first call with a real
// requirement grows it -- as against standing the engine's profile-wide figure in
// for the zero, which would pin the pool at the largest shape the engine admits
// on the strength of a call that uses none of it.
constexpr size_t kMinPooledScratchBytes = 1;

// Refuses a pooled call whose stream is capturing a CUDA graph.
//
// The handoff's event wait is on an event recorded outside the capture:
// cudaStreamWaitEvent fails it with cudaErrorStreamCaptureIsolation and
// invalidates the capture under every capture mode. A growth's cudaMalloc and its
// disposal of the buffer it replaces invalidate it under every mode but
// cudaStreamCaptureModeRelaxed, and under Relaxed run uncaptured, leaving a replay
// pointed at a buffer the pool may since have freed. None of that fails cleanly:
// the caller learns of it only when cudaStreamEndCapture hands back an error and a
// null graph. Refusing names the cause instead.
//
// execute() calls this ahead of every CUDA call it makes that a capture cannot
// take, not just the pool's own: the cudaEventSynchronize on a previous enqueue,
// the cudaMalloc that grows a host-input staging buffer, and the
// cudaStreamSynchronize that ends a call this backend does not let return early.
// Any of them leaves a refusal made after it nothing to save. Only the device
// query and the device switch run earlier, and a capture takes both.
//
// It sees only `stream`, and there is no query for "is any capture live in this
// process": a capture on another stream under Global, or under ThreadLocal from
// this thread, is invalidated by the pool's calls just the same and is not caught.
// The header and the README say so.
//
// The query fails in two ways, and they are not the same answer.
// cudaErrorStreamCaptureImplicit -- `stream` is the legacy stream and some other
// stream is capturing -- is a capture, so it is refused with the rest. Any other
// code is not a capture at all: cudaStreamIsCapturing also hands back a sticky
// fault left by earlier work on this device, measured on an A100 with CUDA 12.8
// after an illegal memory access, returning that fault with the status still
// cudaStreamCaptureStatusNone. Answering that with the capture refusal names a
// cause that is not there and buries the one that is, so it is reported as itself.
Error refuse_pooled_call_on_a_capturing_stream(cudaStream_t stream, int device_id) {
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  const cudaError_t capture_err = cudaStreamIsCapturing(stream, &capture);
  if (capture_err != cudaSuccess && capture_err != cudaErrorStreamCaptureImplicit) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: could not tell whether the selected stream is capturing a CUDA graph, because cudaStreamIsCapturing on device %d reported %s. That code is not a capture, and it need not be this call's doing: cudaStreamIsCapturing also hands back a sticky fault left by earlier work on this device. The shared activation scratch pool reports it because it is the first thing here to ask the device a question.",
        device_id,
        cudaGetErrorString(capture_err));
    // The query's own failure is this call's, not the next one's; a sticky fault
    // survives the clear and resurfaces on whatever this thread calls next, which
    // is where an unpooled call would have reported it.
    cudaGetLastError();
    return Error::InvalidProgram;
  }
  if (capture_err == cudaSuccess && capture == cudaStreamCaptureStatusNone) {
    return Error::Ok;
  }
  ET_LOG(
      Error,
      "TensorRTBackend::execute: the selected stream is capturing a CUDA graph (%s), which the shared activation scratch pool on device %d does not support. Loading the engine with '%s' off removes this refusal, but this delegate does not support capture that way either. The capture section of the backend README has the detail.",
      capture_err == cudaSuccess ? "capture in progress" : cudaGetErrorString(capture_err),
      device_id,
      kSharedActivationScratchKey);
  if (capture_err != cudaSuccess) {
    // cudaErrorStreamCaptureImplicit, the one failure that still reports a
    // capture. Its error is the query's own, so it is cleared here rather than
    // left for the next CUDA call to report under its own name. Not on the
    // ordinary refusal: there the query succeeded, so anything pending on this
    // thread was left by earlier work and belongs to whoever calls CUDA next.
    cudaGetLastError();
  }
  return Error::NotSupported;
}

// Sets out_ptr to a buffer of at least `need` bytes on `device_id`, with `stream`
// ordered after the enqueue that last used the buffer. Returns with `claim`
// holding the device's lock: the caller must submit its enqueue, call
// record_shared_scratch_enqueue, and only then release the claim.
//
// The buffer's capacity is not reported, because no caller has any use for it:
// what a call installs on its context is its own requirement, not whatever the
// pool grew to for someone else.
//
// `need` is a real request and never zero -- execute() substitutes
// kMinPooledScratchBytes for a zero before calling, so that the size the pool
// guarantees and the size the context is told it owns are one figure and not two.
//
// The caller must already have refused a capturing `stream`: the handoff's wait
// invalidates a capture under every mode, a growth's allocation under every mode
// but Relaxed, and a growth's disposal of the buffer it replaces under every mode
// but Relaxed as well.
//
// Must be called with `device_id` already current: cudaEventCreateWithFlags and
// cudaMalloc both act on the *current* device and nothing in here sets it.
Error claim_shared_scratch(SharedScratchClaim& claim, int device_id, size_t need, cudaStream_t stream, void*& out_ptr) {
  SharedScratchDevice& dev = claim.hold(device_id);

  const SharedScratchHandoff handoff = shared_scratch_claim_event(dev, []() -> cudaEvent_t {
    cudaEvent_t event = nullptr;
    // Blocking-sync so the host yields instead of busy-spinning. The only host
    // wait ever made on this event is the one a growth's disposal makes in
    // SharedScratchClaim::release(), and it waits for a whole inference; spinning
    // would burn a core for that time and be no faster. Nothing on a call that
    // does not grow the pool waits on it from the host, and the flag costs
    // nothing there.
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming | cudaEventBlockingSync) != cudaSuccess) {
      // The pool's own failure, cleared where it is made: the caller is told by
      // the return, and leaving it pending would surface it under the name of
      // whatever this thread calls next.
      cudaGetLastError();
      return nullptr;
    }
    return event;
  });
  if (handoff.event == nullptr) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: failed to create the shared activation scratch handoff event on device %d",
        device_id);
    return Error::Internal;
  }
  if (handoff.needs_wait) {
    const cudaError_t err = cudaStreamWaitEvent(stream, handoff.event, 0);
    if (err != cudaSuccess) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: waiting for the enqueue that last used the shared activation scratch failed: %s",
          cudaGetErrorString(err));
      cudaGetLastError();
      return Error::InvalidState;
    }
  }

  const bool first_buffer = dev.buffer == nullptr;
  RetiredScratch retired;
  void* const buffer = shared_scratch_get_or_grow(
      dev,
      need,
      [device_id, first_buffer](size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
          cudaGetLastError();
          return nullptr;
        }
        ET_LOG(
            Info,
            "TensorRTBackend::execute: shared scratch pool (device %d) %s %zu bytes",
            device_id,
            first_buffer ? "allocated" : "grew to",
            bytes);
        return p;
      },
      retired);
  if (buffer == nullptr) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: failed to allocate %zu bytes of shared activation scratch on device %d",
        need,
        device_id);
    return Error::MemoryAllocationFailed;
  }

  // The retired buffer is disposed of at release(), with the device's lock
  // dropped; see SharedScratchClaim::release() for what that costs. Nothing here
  // makes a CUDA call that blocks on device work under that lock -- the stream is
  // created and not waited on, and only a growth that displaced a buffer creates
  // one at all.
  if (retired.buffer != nullptr) {
    const cudaStream_t disposal_stream = shared_scratch_disposal_stream(dev, [device_id]() -> cudaStream_t {
      cudaStream_t stream_for_disposals = nullptr;
      // Non-blocking, so a free queued here is not ordered against the legacy
      // default stream: the disposal waits for the enqueue that used the buffer
      // and for nothing else, and the legacy stream would add whatever any other
      // library on this device happens to have queued on it.
      if (cudaStreamCreateWithFlags(&stream_for_disposals, cudaStreamNonBlocking) != cudaSuccess) {
        // The pool's own failure, cleared where it is made: the disposal falls
        // back to a device-wide free and says so, and leaving this pending would
        // surface it under the name of whatever this thread calls next.
        cudaGetLastError();
        ET_LOG(
            Info,
            "TensorRTBackend::execute: could not create the shared activation scratch pool's disposal stream on device %d, so this growth's free of the buffer it replaced is device-wide",
            device_id);
        return nullptr;
      }
      return stream_for_disposals;
    });
    claim.retire(retired.buffer, retired.wait_for, disposal_stream);
  }

  out_ptr = buffer;
  return Error::Ok;
}

// Records the enqueue now in flight on `stream` against the claimed device's
// shared scratch, so the next call to claim_shared_scratch waits for it.
//
// Call on a `claim` that claim_shared_scratch returned Error::Ok on and that
// still holds the device's lock. Both halves matter, and together they are why
// neither the device nor the event below is checked: a claim's device is non-null
// exactly while it holds the lock, and claim_shared_scratch has already failed the
// call with Error::Internal if the marker had no event, which only the test-only
// reset clears again and that needs the lock this claim is holding.
Error record_shared_scratch_enqueue(SharedScratchClaim& claim, cudaStream_t stream) {
  const cudaEvent_t event = shared_scratch_mark_in_flight(*claim.device());
  const cudaError_t err = cudaEventRecord(event, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: recording the completion event for the shared activation scratch enqueue failed: %s",
        cudaGetErrorString(err));
    cudaGetLastError();
    return Error::InvalidState;
  }
  return Error::Ok;
}

} // namespace

// ---------------------------------------------------------------------------
// is_available
// ---------------------------------------------------------------------------
bool TensorRTBackend::is_available() const {
  if (check_registration() != Error::Ok) {
    return false;
  }

  // Building it is the check, so a process that asks and loads nothing still carries one.
  return shared_runtime() != nullptr;
}

// ---------------------------------------------------------------------------
// init
//
// Deserializes the processed blob into a TensorRT engine handle. The handle is
// placement-new'd into memory provided by the ExecuTorch MemoryAllocator so
// that ExecuTorch owns the arena lifetime; destroy() calls the destructor.
// ---------------------------------------------------------------------------
Result<DelegateHandle*> TensorRTBackend::init(
    BackendInitContext& context,
    FreeableBuffer* processed,
    ArrayRef<CompileSpec> compile_specs) const {
  (void)compile_specs;

  const Error registration_result = check_registration();
  if (registration_result != Error::Ok) {
    return registration_result;
  }

  TORCHTRT_ET_CHECK_NOT_NULL(processed, Error::InvalidArgument, "TensorRTBackend::init: null processed buffer");
  TORCHTRT_ET_CHECK_NOT_NULL(processed->data(), Error::InvalidArgument, "TensorRTBackend::init: null processed buffer");

  TensorRTBlobHeader header;
  if (!TensorRTBlobHeader::parse(processed->data(), processed->size(), header)) {
    ET_LOG(Error, "TensorRTBackend::init: failed to parse TensorRT blob");
    return Error::InvalidProgram;
  }

  MemoryAllocator* allocator = context.get_runtime_allocator();
  TORCHTRT_ET_CHECK_NOT_NULL(allocator, Error::InvalidState, "TensorRTBackend::init: null runtime allocator");

  EngineHandle* handle = allocator->allocateInstance<EngineHandle>();
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle, Error::MemoryAllocationFailed, "TensorRTBackend::init: EngineHandle allocation failed");
  new (handle) EngineHandle();
  std::unique_ptr<EngineHandle, EngineHandleDeleter> handle_guard(handle);

  handle->input_binding_names = std::move(header.input_binding_names);
  handle->output_binding_names = std::move(header.output_binding_names);
  handle->device_id = header.device_id;

  cudaError_t cuda_err = cudaSetDevice(handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Error, "TensorRTBackend::init: cudaSetDevice(%d) failed: %s", handle->device_id, cudaGetErrorString(cuda_err));
    return Error::InvalidProgram;
  }

  // Whether this device can reach pageable host memory at all. Speed is the query below.
  int pageable_access = 0;
  cuda_err = cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Info,
        "TensorRTBackend::init: cudaDeviceGetAttribute(cudaDevAttrPageableMemoryAccess) failed: %s",
        cudaGetErrorString(cuda_err));
  }
  // Both, because reachable and reachable-without-per-page-faults are different questions.
  int pageable_via_page_tables = 0;
  cuda_err = cudaDeviceGetAttribute(
      &pageable_via_page_tables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Info,
        "TensorRTBackend::init: cudaDeviceGetAttribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables) "
        "failed: %s",
        cudaGetErrorString(cuda_err));
  }
  handle->pageable_host_access = pageable_access != 0 && pageable_via_page_tables != 0;

  // One per process: TensorRT wants one runtime outliving every engine from it.
  nvinfer1::IRuntime* runtime = shared_runtime();
  TORCHTRT_ET_CHECK_NOT_NULL(
      runtime, Error::InvalidProgram, "TensorRTBackend::init: failed to create TensorRT runtime");

  const void* engine_data = TensorRTBlobHeader::engine_data(processed->data(), header);
  handle->engine.reset(runtime->deserializeCudaEngine(engine_data, header.engine_size));
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle->engine, Error::InvalidProgram, "TensorRTBackend::init: failed to deserialize TensorRT engine");

  // Apply the weight streaming budget before the execution context is created
  // below: TensorRT forbids changing the budget while a context is active. The
  // budget is a non-negative decimal byte count and may come from two places, in
  // order of precedence:
  //   1. A load-time backend option ("weight_streaming_budget" runtime spec) that
  //      the caller passes to Module::load(LoadBackendOptionsMap). This lets a
  //      deployment size the budget for its own GPU without re-exporting.
  //   2. The same key baked into the .pte as a compile spec at export, used as a
  //      default when no load-time option is given (and the only channel for
  //      loaders that cannot pass backend options yet, e.g. Python/Android).
  // When neither is present and the engine supports streaming, we apply
  // TensorRT's automatic budget, mirroring what the PyTorch runtimes do on
  // deserialize. Negative or malformed values are rejected as InvalidProgram.
  WsBudget ws_request;
  bool is_explicit = false;

  // (1) A load-time runtime spec takes precedence over the baked compile spec.
  // The value is a decimal byte string; a non-negative int is also accepted for
  // small budgets. A present-but-wrong-type or empty option is handled explicitly
  // so a runtime option is never silently dropped. The const char* returned by
  // get_runtime_spec points into the caller's LoadBackendOptionsMap storage, which
  // outlives init(); we parse it immediately and keep only the int64 result.

  const auto ws_runtime = context.get_runtime_spec<const char*>(kWeightStreamingBudgetKey);
  if (ws_runtime.ok()) {
    const char* const value = ws_runtime.get();
    // The option array need not be NUL terminated (the struct is public), so
    // bound the scan. An empty value means "unset", so fall through to (2).
    constexpr std::size_t kRuntimeBudgetMaxScan = 256;
    std::size_t len = 0;
    if (value != nullptr) {
      while (len < kRuntimeBudgetMaxScan && value[len] != '\0') {
        ++len;
      }
    }
    if (len > 0) {
      ws_request = parse_weight_streaming_budget(value, len);
      if (!ws_request.valid) {
        ET_LOG(Error, "TensorRTBackend::init: malformed weight_streaming_budget runtime option");
        return Error::InvalidProgram;
      }
      is_explicit = true;
    } else {
      // The option was supplied but carries no characters, so nothing was set. Say so
      // rather than falling through silently, since a caller who passed the option
      // expects it to take effect. The actual fallback is resolved below: either a
      // budget compile spec if the program carries one, or TensorRT's automatic
      // budget, so do not name one here.
      ET_LOG(Error, "TensorRTBackend::init: weight_streaming_budget runtime option is empty and was ignored");
    }
  } else if (ws_runtime.error() != Error::NotFound) {
    // The key is present but stored as a non-string type. Accept a non-negative
    // int for convenience (its 32-bit range only covers budgets under 2 GB);
    // otherwise reject it so a wrong-typed option is never silently ignored.
    const auto ws_runtime_int = context.get_runtime_spec<int>(kWeightStreamingBudgetKey);
    if (ws_runtime_int.ok() && ws_runtime_int.get() >= 0) {
      ws_request.valid = true;
      ws_request.bytes = ws_runtime_int.get();
      is_explicit = true;
    } else {
      ET_LOG(
          Error,
          "TensorRTBackend::init: weight_streaming_budget runtime option must be a "
          "non-negative int or a decimal byte string");
      return Error::InvalidProgram;
    }
  }

  // (2) Otherwise fall back to the compile spec baked into the .pte at export.
  if (!is_explicit) {
    const CompileSpec* ws_spec = nullptr;
    for (const auto& spec : compile_specs) {
      if (spec.key != nullptr && std::strcmp(spec.key, kWeightStreamingBudgetKey) == 0) {
        if (ws_spec != nullptr) {
          // The budget must appear at most once; a second match means the spec
          // list is inconsistent, so reject the program instead of guessing.
          ET_LOG(Error, "TensorRTBackend::init: duplicate weight_streaming_budget compile spec");
          return Error::InvalidProgram;
        }
        ws_spec = &spec;
      }
    }
    if (ws_spec != nullptr) {
      ws_request = parse_weight_streaming_budget(ws_spec->value.buffer, ws_spec->value.nbytes);
      if (!ws_request.valid) {
        ET_LOG(Error, "TensorRTBackend::init: malformed weight_streaming_budget compile spec");
        return Error::InvalidProgram;
      }
      is_explicit = true;
    }
  }

  const int64_t streamable = handle->engine->getStreamableWeightsSize();
  if (streamable > 0) {
    // getStreamableWeightsSize is > 0 only when the engine was built with
    // BuilderFlag::kWEIGHT_STREAMING.
    int64_t budget;
    if (is_explicit) {
      // An explicit budget is a non-negative byte count, clamped to the
      // streamable size (TensorRT also caps it, but clamp for a clear log).
      budget = ws_request.bytes > streamable ? streamable : ws_request.bytes;
    } else {
      budget = handle->engine->getWeightStreamingAutomaticBudget();
    }
    if (!handle->engine->setWeightStreamingBudgetV2(budget)) {
      if (!is_explicit && handle->engine->setWeightStreamingBudgetV2(0)) {
        // The automatic budget could not be applied; fall back to budget 0, which
        // streams all weights (minimum resident memory) and always fits.
        ET_LOG(
            Info,
            "TensorRTBackend::init: automatic weight streaming budget failed; falling back to budget 0 (stream all weights)");
      } else {
        ET_LOG(
            Error,
            "TensorRTBackend::init: setWeightStreamingBudgetV2 failed (requested=%lld%s)",
            (long long)budget,
            is_explicit ? "" : ", and fallback to 0 also failed");
        return Error::InvalidProgram;
      }
    }
    ET_LOG(
        Info,
        "TensorRTBackend::init: weight streaming budget=%lld streamable=%lld scratch=%lld",
        (long long)handle->engine->getWeightStreamingBudgetV2(),
        (long long)streamable,
        (long long)handle->engine->getWeightStreamingScratchMemorySize());
  } else if (is_explicit) {
    // A budget was requested but the engine has no streamable weights (it was not
    // built with enable_weight_streaming=True, or nothing is streamable). The
    // engine is still valid and runs fully resident, so log and continue rather
    // than fail; failing here would break mixed multi-engine programs where only
    // some engines were built for streaming. Logged at Error because the caller
    // asked for a memory setting that will not take effect, and ExecuTorch has no
    // Warning level.
    ET_LOG(
        Error,
        "TensorRTBackend::init: weight_streaming_budget ignored; engine has no streamable weights (it was not built with enable_weight_streaming=True, or none of its weights are streamable). The engine runs with all weights resident.");
  }

  Error err = initialize_engine_io(*handle);
  if (err != Error::Ok) {
    return err;
  }

  // Map each aliased output binding to the index of the input it aliases so
  // execute() can bind it to that input's device pointer (in-place).
  // Non-aliased models have an empty header.aliased_io -> all -1, unchanged path.
  handle->output_aliased_input_idx.assign(handle->num_outputs, -1);
  handle->input_is_alias_target.assign(handle->num_inputs, false);
  for (const auto& ab : header.aliased_io) {
    int oi = -1;
    for (size_t k = 0; k < handle->output_binding_names.size(); ++k) {
      if (handle->output_binding_names[k] == ab.output) {
        oi = static_cast<int>(k);
        break;
      }
    }
    int ii = -1;
    for (size_t k = 0; k < handle->input_binding_names.size(); ++k) {
      if (handle->input_binding_names[k] == ab.input) {
        ii = static_cast<int>(k);
        break;
      }
    }
    if (oi < 0 || ii < 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::init: aliased_io names not found (output='%s', input='%s')",
          ab.output.c_str(),
          ab.input.c_str());
      return Error::InvalidProgram;
    }
    // Validate the alias kind against the two we understand. The blob parser
    // defaults a missing "kind" to "kv_cache_update"; any other value is a
    // corrupt or newer-than-us wire format we can't safely bind, so fail loudly
    // rather than fall through and treat it as a KV alias (which would bind two
    // tensors to the same storage). Mirrors the Python _reconcile_aliased_io.
    if (ab.kind != "kv_cache_update" && ab.kind != "user") {
      ET_LOG(
          Error,
          "TensorRTBackend::init: aliased_io entry (output='%s') has unknown kind '%s'",
          ab.output.c_str(),
          ab.kind.c_str());
      return Error::InvalidProgram;
    }
    if (ab.kind == "kv_cache_update") {
      // TensorRT's IKVCacheUpdateLayer aliasing is the source of truth for
      // kv_cache_update; the persisted map must agree with what the engine
      // reports (via ICudaEngine::getAliasedInputTensor), else the blob is
      // inconsistent with its own engine.
      const char* trt_alias = handle->engine->getAliasedInputTensor(ab.output.c_str());
      if (trt_alias == nullptr || ab.input != trt_alias) {
        ET_LOG(
            Error,
            "TensorRTBackend::init: kv_cache_update alias for output '%s' disagrees with the "
            "engine (persisted input='%s', engine input='%s')",
            ab.output.c_str(),
            ab.input.c_str(),
            trt_alias == nullptr ? "<none>" : trt_alias);
        return Error::InvalidProgram;
      }
    } else {
      // AliasKind::USER aliases are declared by Torch-TensorRT and not tracked
      // by TensorRT, so it can't validate them; confirm the aliased output and
      // input share a shape before binding them to the same storage.
      const nvinfer1::Dims od = handle->engine->getTensorShape(ab.output.c_str());
      const nvinfer1::Dims id = handle->engine->getTensorShape(ab.input.c_str());
      bool compatible = od.nbDims == id.nbDims;
      for (int d = 0; compatible && d < od.nbDims; ++d) {
        compatible = od.d[d] == id.d[d];
      }
      if (!compatible) {
        ET_LOG(
            Error,
            "TensorRTBackend::init: user alias output '%s' shape is incompatible with input '%s'",
            ab.output.c_str(),
            ab.input.c_str());
        return Error::InvalidProgram;
      }
    }
    handle->output_aliased_input_idx[static_cast<size_t>(oi)] = ii;
    handle->input_is_alias_target[static_cast<size_t>(ii)] = true;
    ++handle->num_aliased_outputs;
  }

  if (handle->num_aliased_outputs > 0) {
    ET_LOG(
        Info,
        "TensorRTBackend::init: %zu aliased output(s) bound in-place to caller-owned inputs",
        handle->num_aliased_outputs);
  }

  err = initialize_input_profiles(*handle);
  if (err != Error::Ok) {
    return err;
  }

  processed->Free();

  ET_LOG(
      Info,
      "TensorRTBackend::init: TensorRT engine ready (%zu inputs, %zu outputs)",
      handle->num_inputs,
      handle->num_outputs);

  handle_guard.release();
  return static_cast<DelegateHandle*>(handle);
}

// ---------------------------------------------------------------------------
// execute
//
// Binds the ExecuTorch input/output tensor data pointers directly to the
// TRT IExecutionContext and calls enqueueV3().  ExecuTorch pre-allocates
// all output tensors before calling execute(), so we only need to register
// their addresses; no separate output allocation is required.
//
// Args layout (mirroring the Python exporter):
//   args[0 .. num_inputs-1]             – input EValues
//   args[num_inputs .. num_inputs+num_outputs-1] – output EValues
// ---------------------------------------------------------------------------
Error TensorRTBackend::execute(BackendExecutionContext& context, DelegateHandle* handle, Span<EValue*> args) const {
  (void)context;
  TORCHTRT_ET_CHECK_NOT_NULL(handle, Error::InvalidArgument, "TensorRTBackend::execute: null delegate handle");
  auto* engine = static_cast<EngineHandle*>(handle);

  const size_t num_inputs = engine->num_inputs;
  const size_t num_outputs = engine->num_outputs;
  // Caller-owned KV: every input is a delegate arg, and each aliased output is
  // threaded as a delegate output arg (the caller-owned mutable buffer's mutation
  // slot), so all engine bindings map 1:1 to delegate args.
  const size_t num_delegate_outputs = num_outputs;
  const size_t num_delegate_inputs = num_inputs;
  if (args.size() < num_delegate_inputs + num_delegate_outputs) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: expected at least %zu args, got %zu",
        num_delegate_inputs + num_delegate_outputs,
        args.size());
    return Error::InvalidArgument;
  }

  int entry_device = -1;
  cudaError_t cuda_err = cudaGetDevice(&entry_device);
  if (cuda_err != cudaSuccess) {
    ET_LOG(Error, "TensorRTBackend::execute: cudaGetDevice failed: %s", cudaGetErrorString(cuda_err));
    return Error::InvalidProgram;
  }
  // Put the engine on its own device for multi-GPU correctness, restoring the
  // caller's device on exit; green-context confinement rides the selected stream,
  // independent of the current device/context.
  const bool switch_device = (entry_device != engine->device_id);
  if (switch_device) {
    cuda_err = cudaSetDevice(engine->device_id);
    if (cuda_err != cudaSuccess) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: cudaSetDevice(%d) failed: %s",
          engine->device_id,
          cudaGetErrorString(cuda_err));
      return Error::InvalidProgram;
    }
  }
  struct DeviceRestore {
    int device;
    bool active;
    ~DeviceRestore() {
      if (active) {
        cudaSetDevice(device);
      }
    }
  } device_restore{entry_device, switch_device};

  std::unique_lock<std::mutex> lock(engine->mu);

  nvinfer1::IExecutionContext* ctx = engine->exec_ctx.get();
  TORCHTRT_ET_CHECK_NOT_NULL(ctx, Error::InvalidState, "TensorRTBackend::execute: backend is not initialized");

  const auto caller_stream = ::executorch::extension::cuda::getCallerStream();
  cudaStream_t stream = caller_stream.value_or(cudaStreamPerThread);
  StreamDrainOnEarlyReturn drain_on_early_return(stream);

  // Settled at init: the option was on at this engine's load and the engine needs
  // scratch under some shape. An engine that needs none is left out of the pool
  // entirely -- enqueueV3 accepts it with no device memory installed, so it need
  // not claim the device and does not serialize against the engines that do.
  const bool pooled_scratch = engine->claims_pooled_scratch;

  // Refused here, ahead of every call this function makes that a capture cannot
  // take. The pool's own are not the only ones: the cudaMalloc that grows a
  // host-input staging buffer further down is prohibited under every mode but
  // cudaStreamCaptureModeRelaxed, so outside that mode it would invalidate the
  // capture before the pooled path was ever reached and a refusal any later would
  // arrive after the thing it exists to protect was gone.
  if (pooled_scratch) {
    const Error capture_err = refuse_pooled_call_on_a_capturing_stream(stream, engine->device_id);
    if (capture_err != Error::Ok) {
      return capture_err;
    }
  }

  bool input_staged_from_host = false;

  if (engine->cached_input_ptrs.empty()) {
    engine->cached_input_ptrs.resize(num_inputs, nullptr);
    engine->cached_input_sizes.resize(num_inputs, 0);
  }
  if (engine->cached_output_ptrs.empty()) {
    engine->cached_output_ptrs.resize(num_outputs, nullptr);
    engine->cached_output_sizes.resize(num_outputs, 0);
  }

  // ------------------------------------------------------------------
  // 1. Bind input shapes and addresses
  // ------------------------------------------------------------------
  // Device pointer each input binding was bound to; aliased outputs reuse the
  // pointer of the input they alias so their update lands in-place.
  std::vector<void*> input_bind_ptrs(num_inputs, nullptr);
  size_t arg_idx = 0; // running index into delegate args
  for (size_t i = 0; i < num_inputs; ++i) {
    const std::string& name = engine->input_binding_names[i];

    EValue* arg = args[arg_idx++];
    TORCHTRT_ET_CHECK_NOT_NULL(
        arg, Error::InvalidArgument, "TensorRTBackend::execute: input arg %zu is not a tensor", i);
    if (!arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: input %zu is not a tensor", i);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_in = arg->toTensor();
    // Caught here rather than at submission. Device memory on the wrong GPU binds without complaint
    // and then fails inside TensorRT as an invalid program, which sends the reader to re-export a
    // model that was never the problem.
    const int input_device = cuda_foreign_device_of_ptr(et_in.const_data_ptr(), engine->device_id);
    if (input_device >= 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: input '%s' is on CUDA device %d, this engine on %d. Move it, or "
          "load the program on that device. Cards able to reach each other is not enough: the "
          "mapping must be on, and nothing here turns it on.",
          name.c_str(),
          input_device,
          engine->device_id);
      return Error::InvalidArgument;
    }
    nvinfer1::Dims dims = to_trt_dims(et_in);
    if (dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
      ET_LOG(Error, "TensorRTBackend::execute: input '%s' rank exceeds TensorRT limit", name.c_str());
      return Error::InvalidArgument;
    }

    const auto& bounds = engine->input_profile_bounds[i];
    if (dims.nbDims != bounds.min.nbDims) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: input '%s' rank %d does not match profile rank %d",
          name.c_str(),
          dims.nbDims,
          bounds.min.nbDims);
      return Error::InvalidArgument;
    }
    for (int d = 0; d < dims.nbDims; ++d) {
      if (dims.d[d] < bounds.min.d[d] || dims.d[d] > bounds.max.d[d]) {
        ET_LOG(Error, "TensorRTBackend::execute: input '%s' dim %d is outside profile bounds", name.c_str(), d);
        return Error::InvalidArgument;
      }
    }

    if (!ctx->setInputShape(name.c_str(), dims)) {
      ET_LOG(Error, "TensorRTBackend::execute: setInputShape failed for '%s'", name.c_str());
      return Error::InvalidState;
    }

    // Caller-owned aliased input: an aliased output binds in-place to this
    // input's device pointer, so its update must land in the caller's storage.
    // If it isn't device-resident the branches below would stage it through
    // delegate scratch, and the in-place update (bound to that scratch) would be
    // silently lost on the next execute() when the staging copy re-reads the
    // caller's unchanged buffer. Fail loudly instead.
    if (engine->input_is_alias_target[i]) {
      const bool device_resident =
          et_in.nbytes() > 0 && (engine->pageable_host_access || is_cuda_accessible_ptr(et_in.const_data_ptr()));
      if (!device_resident) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: aliased input '%s' must be device-reachable without staging, "
            "because its caller-owned in-place update cannot go through host scratch",
            name.c_str());
        return Error::InvalidArgument;
      }
    }

    void* bind_ptr = nullptr;
    if (et_in.nbytes() == 0) {
      if (engine->cached_input_sizes[i] == 0) {
        cuda_err = cudaMalloc(&engine->cached_input_ptrs[i], 1);
        if (cuda_err != cudaSuccess) {
          return Error::MemoryAllocationFailed;
        }
        engine->cached_input_sizes[i] = 1;
      }
      bind_ptr = engine->cached_input_ptrs[i];
    } else if (engine->pageable_host_access || is_cuda_accessible_ptr(et_in.const_data_ptr())) {
      bind_ptr = et_in.mutable_data_ptr();
    } else {
      const size_t needed = et_in.nbytes();
      if (needed > engine->cached_input_sizes[i]) {
        if (engine->cached_input_ptrs[i] != nullptr) {
          cudaFree(engine->cached_input_ptrs[i]);
        }
        cuda_err = cudaMalloc(&engine->cached_input_ptrs[i], needed);
        if (cuda_err != cudaSuccess) {
          engine->cached_input_ptrs[i] = nullptr;
          engine->cached_input_sizes[i] = 0;
          return Error::MemoryAllocationFailed;
        }
        engine->cached_input_sizes[i] = needed;
      }
      bind_ptr = engine->cached_input_ptrs[i];
      cuda_err = cudaMemcpyAsync(bind_ptr, et_in.const_data_ptr(), needed, cudaMemcpyHostToDevice, stream);
      drain_on_early_return.arm();
      if (cuda_err != cudaSuccess) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: H2D copy failed for input '%s': %s",
            name.c_str(),
            cudaGetErrorString(cuda_err));
        return Error::InvalidProgram;
      }
    }

    input_bind_ptrs[i] = bind_ptr;
    if (!ctx->setTensorAddress(name.c_str(), bind_ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for input '%s'", name.c_str());
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 2. Infer output shapes (requires all input shapes to be set first)
  // ------------------------------------------------------------------
  {
    const int32_t io_size = engine->engine->getNbIOTensors();
    std::vector<const char*> unresolved(static_cast<size_t>(io_size), nullptr);
    const int32_t n_unresolved = ctx->inferShapes(io_size, unresolved.data());
    if (n_unresolved != 0) {
      ET_LOG(Error, "TensorRTBackend::execute: inferShapes could not resolve %d tensor(s)", n_unresolved);
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 3. Bind output addresses
  // ExecuTorch pre-allocates output tensors at the maximum shape for
  // dynamic models.  After inferShapes() TRT knows the actual output
  // dims, so update the ExecuTorch TensorImpl's sizes before computing
  // nbytes() and before the Python binding reads back the shape.
  // If the buffer is CPU, stage through a temporary CUDA allocation.
  // ------------------------------------------------------------------
  // (arg index, device_src ptr) for outputs staged through a device buffer.
  std::vector<std::pair<size_t, void*>> outputs_needing_copy;
  // Caller-owned KV: (dst = delegate output EValue ptr, src = aliased input ptr,
  // nbytes). The engine updates the aliased input in place; reflect that into the
  // delegate output EValue after enqueue so ExecuTorch's write-back copy_ sees the
  // updated cache.
  std::vector<std::tuple<void*, void*, size_t>> aliased_reflects;
  for (size_t o = 0; o < num_outputs; ++o) {
    const std::string& name = engine->output_binding_names[o];

    // Aliased output (KV-cache / user): the engine updates the aliased input in
    // place, so bind this output binding to the aliased input's device pointer.
    const int alias_in = engine->output_aliased_input_idx[o];
    if (alias_in >= 0) {
      void* bind_ptr = input_bind_ptrs[static_cast<size_t>(alias_in)];
      if (bind_ptr == nullptr) {
        ET_LOG(Error, "TensorRTBackend::execute: aliased output '%s' has no bound input pointer", name.c_str());
        return Error::InvalidState;
      }
      if (!ctx->setTensorAddress(name.c_str(), bind_ptr)) {
        ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for aliased output '%s'", name.c_str());
        return Error::InvalidState;
      }
      // The aliased output IS a delegate output arg (the caller-owned mutable
      // buffer's mutation slot). Consume it and record a reflect so ExecuTorch's
      // write-back copy_ sees the engine's in-place update.
      const size_t arg_i = arg_idx++;
      EValue* out_arg = args[arg_i];
      TORCHTRT_ET_CHECK_NOT_NULL(
          out_arg, Error::InvalidArgument, "TensorRTBackend::execute: aliased output %zu is not a tensor", o);
      if (!out_arg->isTensor()) {
        ET_LOG(Error, "TensorRTBackend::execute: aliased output %zu is not a tensor", o);
        return Error::InvalidArgument;
      }
      exec_aten::Tensor et_alias_out = out_arg->toTensor();
      // nbytes() below sizes both the reflect copy and ExecuTorch's write-back, so
      // the tensor has to carry the shape TRT inferred before either of them reads it.
      nvinfer1::Dims a_dims = ctx->getTensorShape(name.c_str());
      if (a_dims.nbDims < 0 || a_dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
        ET_LOG(Error, "TensorRTBackend::execute: invalid rank for aliased output '%s'", name.c_str());
        return Error::InvalidState;
      }
      SizesType a_sizes[nvinfer1::Dims::MAX_DIMS];
      for (int d = 0; d < a_dims.nbDims; ++d) {
        a_sizes[d] = static_cast<SizesType>(a_dims.d[d]);
      }
      Error a_resize_err =
          executorch::runtime::resize_tensor(et_alias_out, {a_sizes, static_cast<size_t>(a_dims.nbDims)});
      if (a_resize_err != Error::Ok) {
        ET_LOG(Error, "TensorRTBackend::execute: resize_tensor failed for aliased output '%s'", name.c_str());
        return a_resize_err;
      }
      // Nothing to copy when the slot holds no bytes, which is a shape TensorRT inferred as empty
      // rather than a mistake, so it is skipped rather than refused.
      if (et_alias_out.nbytes() == 0) {
        continue;
      }
      void* dst = et_alias_out.mutable_data_ptr();
      // Bytes to write and no address: refuse, or the engine's update goes nowhere silently.
      if (dst == nullptr) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: aliased output '%s' needs %zu bytes but carries no address. A "
            "program built without runtime-allocated outputs needs the caller to supply each output "
            "buffer through set_output_data_ptr before running.",
            name.c_str(),
            size_t(et_alias_out.nbytes()));
        return Error::InvalidArgument;
      }
      // The ordinary output path asks this and the aliased one did not, so a buffer on a card
      // this engine cannot reach was copied into rather than refused.
      const int foreign_alias_device = cuda_foreign_device_of_ptr(dst, engine->device_id);
      if (foreign_alias_device >= 0) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: aliased output '%s' is on CUDA device %d, this engine on "
            "%d. Move it, or load the program on that device.",
            name.c_str(),
            foreign_alias_device,
            engine->device_id);
        return Error::InvalidArgument;
      }
      // dst != bind_ptr guards against issuing a self-copy. The memory planner does
      // not currently place the delegate's output slot on the aliased input -- the
      // two are live at the same time -- so this holds for every aliased output.
      if (dst != bind_ptr) {
        aliased_reflects.emplace_back(dst, bind_ptr, et_alias_out.nbytes());
      }
      continue;
    }

    const size_t arg_i = arg_idx++; // continue the shared running arg index after the inputs
    EValue* arg = args[arg_i];
    TORCHTRT_ET_CHECK_NOT_NULL(arg, Error::InvalidArgument, "TensorRTBackend::execute: output %zu is not a tensor", o);
    if (!arg->isTensor()) {
      ET_LOG(Error, "TensorRTBackend::execute: output %zu is not a tensor", o);
      return Error::InvalidArgument;
    }

    exec_aten::Tensor et_out = arg->toTensor();
    // Asked separately, because an output address comes from the caller rather than the engine.
    const int output_device = cuda_foreign_device_of_ptr(et_out.const_data_ptr(), engine->device_id);
    if (output_device >= 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: the buffer supplied for output '%s' is on CUDA device %d but "
          "this engine runs on device %d. Supply a buffer on the engine's device. The program "
          "itself is fine.",
          name.c_str(),
          output_device,
          engine->device_id);
      return Error::InvalidArgument;
    }

    // Update the ExecuTorch tensor shape to the actual TRT output shape.
    // getTensorShape() is valid after inferShapes() has been called.
    nvinfer1::Dims actual_dims = ctx->getTensorShape(name.c_str());
    if (actual_dims.nbDims < 0 || actual_dims.nbDims > nvinfer1::Dims::MAX_DIMS) {
      ET_LOG(Error, "TensorRTBackend::execute: invalid output rank for '%s'", name.c_str());
      return Error::InvalidState;
    }
    SizesType new_sizes[nvinfer1::Dims::MAX_DIMS];
    for (int d = 0; d < actual_dims.nbDims; ++d) {
      new_sizes[d] = static_cast<SizesType>(actual_dims.d[d]);
    }
    // A 0-d output has an immutable rank of zero, and TensorRT reports it as a
    // 1-element 1-D shape, so resizing would be rejected. Skip it when the
    // element count already agrees.
    const bool scalar_output = et_out.dim() == 0 && actual_dims.nbDims == 1 && actual_dims.d[0] == 1;
    if (!scalar_output) {
      Error resize_err =
          executorch::runtime::resize_tensor(et_out, {new_sizes, static_cast<size_t>(actual_dims.nbDims)});
      if (resize_err != Error::Ok) {
        ET_LOG(Error, "TensorRTBackend::execute: resize_tensor failed for output '%s'", name.c_str());
        return resize_err;
      }
    }

    void* bind_ptr = nullptr;
    if (et_out.nbytes() == 0) {
      if (engine->cached_output_sizes[o] == 0) {
        cuda_err = cudaMalloc(&engine->cached_output_ptrs[o], 1);
        if (cuda_err != cudaSuccess) {
          return Error::MemoryAllocationFailed;
        }
        engine->cached_output_sizes[o] = 1;
      }
      bind_ptr = engine->cached_output_ptrs[o];
    } else if (engine->pageable_host_access || is_cuda_accessible_ptr(et_out.const_data_ptr())) {
      bind_ptr = et_out.mutable_data_ptr();
    } else {
      const size_t needed = et_out.nbytes();
      if (needed > engine->cached_output_sizes[o]) {
        if (engine->cached_output_ptrs[o] != nullptr) {
          cudaFree(engine->cached_output_ptrs[o]);
        }
        cuda_err = cudaMalloc(&engine->cached_output_ptrs[o], needed);
        if (cuda_err != cudaSuccess) {
          engine->cached_output_ptrs[o] = nullptr;
          engine->cached_output_sizes[o] = 0;
          return Error::MemoryAllocationFailed;
        }
        engine->cached_output_sizes[o] = needed;
      }
      bind_ptr = engine->cached_output_ptrs[o];
      outputs_needing_copy.push_back({arg_i, bind_ptr});
    }

    if (!ctx->setTensorAddress(name.c_str(), bind_ptr)) {
      ET_LOG(Error, "TensorRTBackend::execute: setTensorAddress failed for output '%s'", name.c_str());
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 4. Back activation scratch with the shared per-device pool
  // ------------------------------------------------------------------
  // The query requires every input shape to be bound, which they are by here, and
  // whatever it answers is binding rather than advisory: an engine backed by less
  // than it asked for writes past the end. The buffer is installed on every call,
  // not once, because any call needing more than the pool holds -- this engine on
  // other shapes, or another one -- grows it and moves it. A kSTATIC context owns
  // its private scratch, so the install must not be made on one.
  //
  // A reported zero has two causes that reach here and nothing distinguishes them:
  // bound shapes that genuinely need none -- an empty input inside a profile that
  // admits one -- and a query that failed. Neither sizes the pool: a zero asks for
  // kMinPooledScratchBytes, which any live buffer already covers. The install is
  // what tells the two apart, because setDeviceMemoryV2 refuses a buffer smaller
  // than the bound shapes need, so a call whose query failed ends here rather than
  // enqueueing against whatever pointer the context was last given, which a growth
  // may since have freed. The README's shared activation scratch section says why
  // the minimum rather than the engine's profile-wide figure.
  //
  // What is installed is this call's own requirement and not the capacity the pool
  // holds, which is larger whenever an earlier call asked for more. The larger
  // figure would tell TensorRT it owns bytes holding another engine's activations
  // -- the pool never clears the buffer -- and would blunt the refusal above,
  // since after a growth the capacity may well cover what a failed query
  // concealed.
  //
  // The claim holds the device's pool lock from here through the record of the
  // enqueue below; see SharedScratchClaim for why it spans that far. Every return
  // in between drops it through the destructor, which runs ahead of the device
  // restore above, so its free lands on the right device.
  SharedScratchClaim scratch_claim;
  if (pooled_scratch) {
    const size_t need = ctx->updateDeviceMemorySizeForShapes();
    // The substitution for a zero is made here and once: the same figure has to
    // be the size the pool guarantees and the size the context is told it owns,
    // and nothing downstream checks that two copies of it still agree --
    // setDeviceMemoryV2 refuses an install smaller than the bound shapes need and
    // says nothing about one that is larger. A context whose engine needs scratch
    // under some shape has to be given a buffer whatever this call's shapes need,
    // or enqueueV3 refuses it.
    const size_t scratch_bytes = need == 0 ? kMinPooledScratchBytes : need;
    void* pool = nullptr;
    const Error scratch_err = claim_shared_scratch(scratch_claim, engine->device_id, scratch_bytes, stream, pool);
    if (scratch_err != Error::Ok) {
      return scratch_err;
    }
    if (!install_pooled_scratch(*ctx, pool, scratch_bytes, engine->device_id)) {
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 5. Enqueue inference on the current CUDA stream
  // ------------------------------------------------------------------
  // Armed before the launch, so a partial launch still drains what it submitted.
  drain_on_early_return.arm();
  if (!ctx->enqueueV3(stream)) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: enqueueV3 failed. Likely an output with no address: supply each "
        "with set_output_data_ptr. Else the guard's stream is on another device, or is the "
        "per-thread stream inside a green context, which is invalid there.");
    return Error::InvalidState;
  }

  // Pairs with claim_shared_scratch: the next claimant waits on this event.
  if (pooled_scratch) {
    const Error mark_err = record_shared_scratch_enqueue(scratch_claim, stream);
    if (mark_err != Error::Ok) {
      // Nothing will wait for this enqueue otherwise: the record that would have
      // put it on the marker is the call that just failed.
      //
      // Waited for here rather than left to drain_on_early_return, because this
      // has to happen with the device's pool lock still held and the guard's
      // destructor runs after the claim's. Releasing first would hand the buffer
      // to a claimant with nothing ordering it against the enqueue this call just
      // submitted, which is the silent-corruption case the lock exists for. It is
      // the one place a pooled call holds the lock across a host wait, so another
      // pooled engine on this device waits out this inference.
      (void)cudaStreamSynchronize(stream);
      return mark_err;
    }
  }
  // The enqueue is now on the marker's event, so the device's pool is safe to
  // hand to the next claimant. Released here rather than at the end of the
  // function so the rest of execute() -- the aliased reflects, the D2H copies and
  // their synchronizations -- does not hold up another engine on this device.
  // The release is also where a growth disposes of the buffer it replaced, with
  // the lock dropped; this is the only place that release is made rather than
  // left to the claim's destructor.
  if (!scratch_claim.release()) {
    // The only failure it reports is its wait for the enqueue on the buffer a
    // growth retired, which for a wait on device work means this device is
    // already in a faulted state. drain_on_early_return, armed before the launch,
    // waits for this call's own enqueue on the way out.
    return Error::InvalidProgram;
  }

  // Caller-owned KV: reflect each engine in-place update into its delegate output
  // EValue (D2D on the same stream, after the engine work).
  for (const auto& r : aliased_reflects) {
    cuda_err = cudaMemcpyAsync(std::get<0>(r), std::get<1>(r), std::get<2>(r), cudaMemcpyDefault, stream);
    drain_on_early_return.arm();
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "TensorRTBackend::execute: aliased-output reflect copy failed: %s", cudaGetErrorString(cuda_err));
      // Drain first: the engine work is on this stream and the buffers outlive the call.
      (void)cudaStreamSynchronize(stream);
      return Error::InvalidProgram;
    }
  }

  // The engine work is in flight on `stream`, and we always wait for it. ExecuTorch's runtime
  // has no asynchronous execute: its execute() returns Error::Ok to mean the work is finished,
  // every caller reads the outputs straight after it returns, and the API hands back no event or
  // future to wait on. So there is nobody an early return could be honest with.
  Error copy_err = Error::Ok;
  for (auto& output : outputs_needing_copy) {
    exec_aten::Tensor et_out = args[output.first]->toTensor();
    cuda_err =
        cudaMemcpyAsync(et_out.mutable_data_ptr(), output.second, et_out.nbytes(), cudaMemcpyDeviceToHost, stream);
    if (cuda_err != cudaSuccess) {
      // Name the output and number it the way the caller does. output.first indexes the whole
      // argument list, so on a one-input engine the first output read as "output 1", and the
      // index a caller passes to set_output_data_ptr counts outputs from zero.
      const size_t output_index = output.first - engine->num_inputs;
      const char* output_name = output_index < engine->output_binding_names.size()
          ? engine->output_binding_names[output_index].c_str()
          : "unknown";
      ET_LOG(
          Error,
          "TensorRTBackend::execute: copy out of output %zu ('%s') failed: %s. Either no buffer "
          "was supplied for it with set_output_data_ptr, or an earlier failure left the device "
          "unusable. Read the first error, not this one.",
          output_index,
          output_name,
          cudaGetErrorString(cuda_err));
      // The engine is still running on the stream, so drain before returning.
      copy_err = Error::InvalidArgument;
      break;
    }
  }
  cuda_err = cudaStreamSynchronize(stream);
  if (cuda_err != cudaSuccess) {
    // Returning with the guard still armed, so its destructor waits: a failed drain says nothing
    // about whether the work finished, and the staging buffers it may still be writing outlive
    // this call.
    ET_LOG(
        Error,
        "TensorRTBackend::execute: the device reported '%s' while finishing this call. Usual causes, "
        "in order: an input buffer shorter than its shape, storage freed while the call was running, "
        "and an output address that is not writable.",
        cudaGetErrorString(cuda_err));
    return Error::Internal;
  }
  if (copy_err != Error::Ok) {
    return copy_err;
  }
  drain_on_early_return.disarm();
  return Error::Ok;
}

// ---------------------------------------------------------------------------
// set_option
// ---------------------------------------------------------------------------
Error TensorRTBackend::set_option(ET_UNUSED BackendOptionContext& context, const Span<BackendOption>& backend_options) {
  // The whole span is read before anything is stored. A span is one request, so a
  // caller told it was refused must not find part of it applied -- and the part
  // that would be applied here is process-wide and governs every engine loaded
  // after it. Where a span names this key more than once the last one wins, which
  // is what applying each in turn did.
  bool requested = false;
  bool have_request = false;
  for (const auto& option : backend_options) {
    // A caller may address one option span to several backends, so a key this
    // backend does not read is skipped rather than refused.
    if (std::strcmp(option.key, kSharedActivationScratchKey) == 0) {
      const bool* const val = std::get_if<bool>(&option.value);
      if (val == nullptr) {
        ET_LOG(Error, "TensorRTBackend::set_option: option '%s' must be a boolean", kSharedActivationScratchKey);
        return Error::InvalidArgument;
      }
      requested = *val;
      have_request = true;
    }
  }

  if (have_request) {
    scratch_enabled.store(requested, std::memory_order_relaxed);
  }
  return Error::Ok;
}

// ---------------------------------------------------------------------------
// destroy
//
// Explicitly destructs the EngineHandle. The underlying memory was allocated
// by ExecuTorch's MemoryAllocator and will be reclaimed by the arena.
// ---------------------------------------------------------------------------
void TensorRTBackend::destroy(DelegateHandle* handle) const {
  if (handle != nullptr) {
    static_cast<EngineHandle*>(handle)->~EngineHandle();
  }
}

} // namespace executorch_backend
} // namespace torch_tensorrt

// ---------------------------------------------------------------------------
// Static registration – links the name "TensorRTBackend" used in the .pte
// file to this implementation at program startup.
// ---------------------------------------------------------------------------
namespace torch_tensorrt {
namespace executorch_backend {
namespace {

TensorRTBackend& get_backend() {
  static torch_tensorrt::executorch_backend::TensorRTBackend backend;
  return backend;
}

const ::executorch::runtime::Backend kBackendId{"TensorRTBackend", &get_backend()};
const Error kRegistrationResult = ::executorch::runtime::register_backend(kBackendId);

} // namespace

// Compiled everywhere, not only in the wheel build. The Python package refuses to import unless it
// can ask this, and a C++ consumer linking two delegates is the case most likely to need the answer,
// yet the in-tree build that serves those consumers used to leave it out entirely.
extern "C" bool torch_tensorrt_owns_executorch_registration() {
  return ::executorch::runtime::get_backend_class(kBackendId.name) == &get_backend();
}

} // namespace executorch_backend
} // namespace torch_tensorrt
