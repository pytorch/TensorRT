/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "torch_tensorrt/executorch/TensorRTBackend.h"
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
  // A fast-path execute() may have returned with its enqueue still in flight on the
  // caller's stream, still using exec_ctx and the cached staging buffers. Wait on
  // the recorded completion event before destroying the context or freeing the
  // buffers. We wait on the event, not the stream, so this stays valid even if the
  // caller already destroyed the stream. Non-skip executes synchronized inline, so
  // inflight_pending is false there. Fall back to a device sync if no event exists.
  if (inflight_event != nullptr) {
    if (inflight_pending) {
      cudaError_t err = cudaEventSynchronize(inflight_event);
      if (err != cudaSuccess) {
        ET_LOG(Error, "EngineHandle::~EngineHandle: cudaEventSynchronize failed: %s", cudaGetErrorString(err));
        cudaGetLastError(); // clear sticky error; tear down regardless
      }
      inflight_pending = false;
    }
  } else {
    cudaDeviceSynchronize();
  }
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
  runtime.reset();
  if (inflight_event != nullptr) {
    cudaEventDestroy(inflight_event);
    inflight_event = nullptr;
  }
}

namespace {

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
// execute() must read EngineHandle::shared_scratch, never this.
std::atomic<bool> scratch_enabled{false};

Error initialize_engine_io(EngineHandle& handle) {
  if (handle.input_binding_names.empty() && handle.output_binding_names.empty() &&
      !infer_binding_names(handle.engine.get(), handle.input_binding_names, handle.output_binding_names)) {
    ET_LOG(Error, "TensorRTBackend::init: failed to infer TensorRT binding names");
    return Error::InvalidProgram;
  }

  handle.num_inputs = handle.input_binding_names.size();
  handle.num_outputs = handle.output_binding_names.size();

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
    // Read after the weight streaming budget is applied, which the caller does
    // before this runs because TensorRT forbids moving the budget once a context
    // exists -- and the budget is the one thing that moves this figure.
    handle.engine_scratch_bytes = static_cast<size_t>(handle.engine->getDeviceMemorySizeV2());
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

// Process-wide per-device pool for TensorRT execution-context activation scratch.
// One buffer, grown to the largest requirement any call on that device has asked
// for, serves every kUSER_MANAGED context on a device, instead of each of N
// layer-engines pinning its own scratch, which makes device memory scale with the
// layer count and OOMs multi-layer models.
//
// ORDERING: a context reads and writes its scratch for the whole enqueue, which
// can still be in flight when execute() returns, so two enqueues must never hold
// this buffer at the same time. A device's lock is what enforces that -- see
// SharedScratchClaim -- and it is held from the claim through the enqueue and the
// record of it, so two execute() calls on one device are serialized at
// submission. The lock does not couple two
// devices: each carries its own, and no CUDA call is made under the one lock the
// registry itself holds.
//
// The buffers and the events are intentionally never freed at teardown. Nothing
// here runs a CUDA call at process exit, which keeps the pool clear of
// teardown-order hazards against anything else holding device memory.
SharedScratchPool scratch_pool;

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
    release();
  }

  SharedScratchDevice& hold(int device_id) {
    dev_ = &scratch_pool.get(device_id);
    device_id_ = device_id;
    lock_ = std::unique_lock<std::mutex>(dev_->mu);
    return *dev_;
  }

  // Null until hold() runs and null again after release(): non-null exactly while
  // this claim holds the device's lock.
  SharedScratchDevice* device() const {
    return dev_;
  }

  int device_id() const {
    return device_id_;
  }

  // Takes ownership of a buffer a growth displaced, to be freed by release().
  void retire(void* buffer) {
    retired_ = buffer;
  }

  // Drops the lock, and the device pointer with it so device() cannot hand out a
  // pointer this claim no longer holds the lock for. Then frees whatever a growth
  // displaced -- after the unlock, because cudaFree waits for every stream on the
  // device, which under the lock would stall the next claim on work unrelated to
  // the pool. Outside it the stall is this caller's alone and falls after its own
  // enqueue, so a growth makes that one execute() wait for its own engine work.
  //
  // Frees on the current device, which must still be the buffer's.
  void release() {
    if (lock_.owns_lock()) {
      lock_.unlock();
    }
    dev_ = nullptr;
    if (retired_ != nullptr) {
      // cudaFree synchronizes, so an earlier asynchronous fault on this device
      // often surfaces here. Report and clear it, or it resurfaces under the
      // name of the next CUDA call in execute().
      const cudaError_t err = cudaFree(retired_);
      if (err != cudaSuccess) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: freeing the shared activation scratch buffer that a pool growth replaced on device %d failed: %s",
            device_id_,
            cudaGetErrorString(err));
        cudaGetLastError(); // clear sticky error; the free is cleanup, so execute() continues
      }
      retired_ = nullptr;
    }
  }

 private:
  SharedScratchDevice* dev_ = nullptr;
  int device_id_ = -1;
  std::unique_lock<std::mutex> lock_;
  void* retired_ = nullptr;
};

// Sets out_ptr to a buffer of at least `need` bytes on `device_id` and out_size to
// its capacity, with `stream` ordered after the enqueue that last used the buffer.
// Returns with `claim` holding the device's lock: the caller must submit its
// enqueue, call mark_shared_scratch_in_flight, and only then release the claim.
//
// Must be called with `device_id` already current: cudaEventCreateWithFlags,
// cudaMalloc and cudaFree all act on the *current* device and nothing in here
// sets it.
Error get_or_grow_shared_scratch(
    SharedScratchClaim& claim,
    int device_id,
    size_t need,
    cudaStream_t stream,
    void*& out_ptr,
    size_t& out_size) {
  SharedScratchDevice& dev = claim.hold(device_id);

  const SharedScratchHandoff handoff = shared_scratch_claim_event(dev, []() -> cudaEvent_t {
    cudaEvent_t event = nullptr;
    if (cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
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
      return Error::InvalidState;
    }
  }

  const bool first_buffer = dev.buffer == nullptr;
  RetiredScratch retired;
  void* const buffer = shared_scratch_get_or_grow(
      dev,
      need,
      out_size,
      [device_id, first_buffer](size_t bytes) -> void* {
        void* p = nullptr;
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
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

  if (retired.buffer != nullptr) {
    // The wait runs here and the free runs at release(), because this caller
    // records its own enqueue on the same event before it drops the lock: a wait
    // deferred to sit beside the free would block on that enqueue too.
    const cudaError_t err = retired.wait_for != nullptr ? cudaEventSynchronize(retired.wait_for) : cudaSuccess;
    if (err == cudaSuccess) {
      claim.retire(retired.buffer);
    } else {
      // This wait is the only thing keeping the free off a buffer an enqueue may
      // still be reading, so a failed wait leaks it instead -- one buffer for each
      // growth whose wait fails. How many growths a run sees is not bounded by the
      // engine count: the requirement is re-queried every execute() below, so an
      // engine that answers with what the bound shapes need can grow the pool on
      // any call.
      ET_LOG(
          Error,
          "TensorRTBackend::execute: waiting for the enqueue on the replaced shared activation scratch on device %d failed (%s); leaking that buffer rather than freeing it under a live enqueue",
          device_id,
          cudaGetErrorString(err));
      cudaGetLastError(); // clear sticky error; execute() continues regardless
    }
  }

  out_ptr = buffer;
  return Error::Ok;
}

// Records the enqueue now in flight on `stream` against the claimed device's
// shared scratch, so the next call to get_or_grow_shared_scratch waits for it.
// Call with `claim` still holding the device's lock.
Error mark_shared_scratch_in_flight(SharedScratchClaim& claim, cudaStream_t stream) {
  SharedScratchDevice* const dev = claim.device();
  if (dev == nullptr) {
    ET_LOG(Error, "TensorRTBackend::execute: no shared activation scratch claim to record an enqueue against");
    return Error::Internal;
  }

  const cudaEvent_t event = shared_scratch_mark_in_flight(*dev);
  if (event == nullptr) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: shared activation scratch on device %d has no handoff event",
        claim.device_id());
    return Error::Internal;
  }
  const cudaError_t err = cudaEventRecord(event, stream);
  if (err != cudaSuccess) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: recording the completion event for the shared activation scratch enqueue failed: %s",
        cudaGetErrorString(err));
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

  TRTLogger logger;
  TRTUniquePtr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
  return runtime != nullptr;
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

  // Created while device_id is current so the event belongs to the engine's device.
  // It orders a later execute()/teardown after a skip-sync enqueue (see execute()
  // and ~EngineHandle). Blocking-sync so the host yields instead of busy-spinning.
  cuda_err = cudaEventCreateWithFlags(&handle->inflight_event, cudaEventDisableTiming | cudaEventBlockingSync);
  if (cuda_err != cudaSuccess) {
    ET_LOG(Error, "TensorRTBackend::init: cudaEventCreateWithFlags failed: %s", cudaGetErrorString(cuda_err));
    return Error::InvalidProgram;
  }

  int is_integrated = 0;
  cuda_err = cudaDeviceGetAttribute(&is_integrated, cudaDevAttrIntegrated, handle->device_id);
  if (cuda_err != cudaSuccess) {
    ET_LOG(
        Info,
        "TensorRTBackend::init: cudaDeviceGetAttribute(cudaDevAttrIntegrated) failed: %s",
        cudaGetErrorString(cuda_err));
  }
  handle->unified_memory = is_integrated != 0;

  handle->runtime.reset(nvinfer1::createInferRuntime(handle->logger));
  TORCHTRT_ET_CHECK_NOT_NULL(
      handle->runtime, Error::InvalidProgram, "TensorRTBackend::init: failed to create TensorRT runtime");

  const void* engine_data = TensorRTBlobHeader::engine_data(processed->data(), header);
  handle->engine.reset(handle->runtime->deserializeCudaEngine(engine_data, header.engine_size));
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

  // A prior fast-path execute() may have returned with its enqueue still in flight
  // on the shared exec_ctx. Wait for it before reconfiguring the context below:
  // TensorRT forbids mutating a context while one of its enqueues is in flight, and
  // setInputShape/setTensorAddress run on the host, so this must be a host-side wait.
  if (engine->inflight_pending) {
    cuda_err = cudaEventSynchronize(engine->inflight_event);
    engine->inflight_pending = false;
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "TensorRTBackend::execute: cudaEventSynchronize failed: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidProgram;
    }
  }
  const auto caller_stream = ::executorch::extension::cuda::getCallerStream();
  const bool caller_stream_set = caller_stream.has_value();
  cudaStream_t stream = caller_stream.value_or(cudaStreamPerThread);
  bool output_staged_to_host = false;
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
          et_in.nbytes() > 0 && (engine->unified_memory || is_cuda_accessible_ptr(et_in.const_data_ptr()));
      if (!device_resident) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: aliased input '%s' must be device-resident (non-empty and "
            "CUDA-accessible or unified memory); its caller-owned in-place update cannot be staged "
            "through host scratch",
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
    } else if (engine->unified_memory || is_cuda_accessible_ptr(et_in.const_data_ptr())) {
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
      input_staged_from_host = true;
      cuda_err = cudaMemcpyAsync(bind_ptr, et_in.const_data_ptr(), needed, cudaMemcpyHostToDevice, stream);
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
      void* dst = et_alias_out.nbytes() > 0 ? et_alias_out.mutable_data_ptr() : nullptr;
      // dst != bind_ptr guards against issuing a self-copy. The memory planner does
      // not currently place the delegate's output slot on the aliased input -- the
      // two are live at the same time -- so this holds for every aliased output.
      if (dst != nullptr && dst != bind_ptr) {
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
    } else if (engine->unified_memory || is_cuda_accessible_ptr(et_out.const_data_ptr())) {
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
      output_staged_to_host = true;
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
  // The query requires every input shape to be bound, which they are by here.
  // Whatever it answers is binding rather than advisory: setDeviceMemoryV2 refuses
  // a smaller buffer, and an engine backed by less than it asked for writes past
  // the end.
  //
  // The buffer is installed on every call, not once, because any call needing more
  // than the pool holds -- this engine on other shapes, or another one -- grows it
  // and moves it. A kSTATIC context owns its private scratch, so setDeviceMemoryV2
  // must not be called on one.
  //
  // A reported zero is ambiguous: TensorRT answers a failed query and an engine
  // that genuinely needs no scratch the same way, and the engine's own
  // requirement is what separates them. An engine that needs none is given no
  // buffer, so it has nothing to claim and nothing for the next claimant to order
  // against. A failed query carried on would instead leave the context enqueueing
  // against whatever buffer it last held, because setDeviceMemoryV2(nullptr, 0)
  // is rejected and returns nothing to test.
  //
  // The claim holds the device's pool lock from here through the record of the
  // enqueue below; see SharedScratchClaim for why it spans that far. Every return
  // in between drops it through the destructor, which runs ahead of the device
  // restore above, so its free lands on the right device.
  SharedScratchClaim scratch_claim;
  bool scratch_from_pool = false;
  if (engine->shared_scratch) {
    const size_t need = ctx->updateDeviceMemorySizeForShapes();
    if (need > 0) {
      void* pool = nullptr;
      size_t pool_size = 0;
      const Error scratch_err =
          get_or_grow_shared_scratch(scratch_claim, engine->device_id, need, stream, pool, pool_size);
      if (scratch_err != Error::Ok) {
        return scratch_err;
      }
      scratch_from_pool = true;
      ctx->setDeviceMemoryV2(pool, static_cast<int64_t>(pool_size));
    } else if (engine->engine_scratch_bytes > 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: updateDeviceMemorySizeForShapes returned 0, but the engine needs %zu bytes of activation scratch",
          engine->engine_scratch_bytes);
      return Error::InvalidState;
    }
  }

  // ------------------------------------------------------------------
  // 5. Enqueue inference on the current CUDA stream
  // ------------------------------------------------------------------
  if (!ctx->enqueueV3(stream)) {
    ET_LOG(
        Error,
        "TensorRTBackend::execute: enqueueV3 failed. Verify that the selected "
        "CallerStreamGuard stream belongs to the TensorRT engine device. If a CUDA "
        "green context is current, scope a CallerStreamGuard with a green-context "
        "stream: cudaStreamPerThread is invalid while a green context is current.");
    return Error::InvalidState;
  }

  // Pairs with get_or_grow_shared_scratch: the next claimant waits on this event.
  if (scratch_from_pool) {
    const Error mark_err = mark_shared_scratch_in_flight(scratch_claim, stream);
    if (mark_err != Error::Ok) {
      // Nothing will wait for this enqueue, so wait for it here instead of
      // leaving the next user of the buffer to overwrite live scratch.
      (void)cudaStreamSynchronize(stream);
      engine->inflight_pending = false;
      return mark_err;
    }
  }
  // The enqueue is now on the marker's event, so the device's pool is safe to
  // hand to the next claimant. Released here rather than at the end of the
  // function so the rest of execute() -- the aliased reflects, the D2H copies and
  // their synchronizations -- does not hold up another engine on this device.
  scratch_claim.release();

  // Caller-owned KV: reflect each engine in-place update into its delegate output
  // EValue (D2D on the same stream, after the engine work).
  for (const auto& r : aliased_reflects) {
    cuda_err = cudaMemcpyAsync(std::get<0>(r), std::get<1>(r), std::get<2>(r), cudaMemcpyDeviceToDevice, stream);
    if (cuda_err != cudaSuccess) {
      ET_LOG(
          Error, "TensorRTBackend::execute: aliased-output reflect D2D copy failed: %s", cudaGetErrorString(cuda_err));
      // enqueueV3 already submitted engine work to `stream`, and inflight_pending
      // is not armed until the end of the happy path -- drain now so a later
      // execute() or the destructor never reconfigures/frees exec_ctx while this
      // enqueue is still running.
      (void)cudaStreamSynchronize(stream);
      engine->inflight_pending = false;
      return Error::InvalidProgram;
    }
  }

  // The engine work is now in flight on `stream`. Decide whether to wait for it:
  //   must_sync = an output is staged to host (the caller reads the D2H result on
  //   return), an input was staged from host (its async H2D read the caller's host
  //   buffer, which the caller may reuse once we return), or no caller stream is
  //   active (preserve the historical "results ready on return" behavior).
  // Otherwise (caller stream + all I/O device-resident) leave the work enqueued so
  // it composes with the caller's later GPU work, and record inflight_event so the
  // next execute() and the destructor wait before reusing/freeing exec_ctx. The D2H
  // copies live in the must_sync branch: an output staged to host always sets
  // output_staged_to_host, so outputs_needing_copy is empty on the skip path.
  // An aliased reflect enqueues the engine's in-place update into the delegate
  // output EValue on `stream`; ExecuTorch's buffer-mutation copy_ reads that EValue
  // after execute() returns, so the reflect must complete first. A model with
  // aliased outputs therefore always syncs here.
  const bool aliased_reflect_pending = !aliased_reflects.empty();
  const bool must_sync =
      output_staged_to_host || input_staged_from_host || aliased_reflect_pending || !caller_stream_set;
  if (must_sync) {
    Error copy_err = Error::Ok;
    for (auto& output : outputs_needing_copy) {
      exec_aten::Tensor et_out = args[output.first]->toTensor();
      cuda_err =
          cudaMemcpyAsync(et_out.mutable_data_ptr(), output.second, et_out.nbytes(), cudaMemcpyDeviceToHost, stream);
      if (cuda_err != cudaSuccess) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: D2H copy failed for output %zu: %s",
            output.first,
            cudaGetErrorString(cuda_err));
        // The enqueue already succeeded, so the engine is still running on the
        // stream. Drain below before returning, or the next call mutates a live
        // execution context, which TensorRT forbids.
        copy_err = Error::InvalidProgram;
        break;
      }
    }
    cuda_err = cudaStreamSynchronize(stream);
    engine->inflight_pending = false;
    if (cuda_err != cudaSuccess) {
      ET_LOG(Error, "TensorRTBackend::execute: cudaStreamSynchronize failed: %s", cudaGetErrorString(cuda_err));
      return Error::InvalidProgram;
    }
    if (copy_err != Error::Ok) {
      return copy_err;
    }
  } else {
    cuda_err = cudaEventRecord(engine->inflight_event, stream);
    if (cuda_err != cudaSuccess) {
      // Could not arm the completion marker; drain now so a later execute() or the
      // destructor never reconfigures or frees exec_ctx while this enqueue runs.
      ET_LOG(Error, "TensorRTBackend::execute: cudaEventRecord failed: %s", cudaGetErrorString(cuda_err));
      (void)cudaStreamSynchronize(stream);
      engine->inflight_pending = false;
      return Error::InvalidProgram;
    }
    engine->inflight_pending = true;
  }
  return Error::Ok;
}

// ---------------------------------------------------------------------------
// set_option
// ---------------------------------------------------------------------------
Error TensorRTBackend::set_option(ET_UNUSED BackendOptionContext& context, const Span<BackendOption>& backend_options) {
  for (const auto& option : backend_options) {
    // A caller may address one option span to several backends, so a key this
    // backend does not read is skipped rather than refused.
    if (std::strcmp(option.key, kSharedActivationScratchKey) == 0) {
      if (const bool* const val = std::get_if<bool>(&option.value)) {
        scratch_enabled.store(*val, std::memory_order_relaxed);
      } else {
        ET_LOG(Error, "TensorRTBackend::set_option: option '%s' must be a boolean", kSharedActivationScratchKey);
        return Error::InvalidArgument;
      }
    }
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
} // namespace executorch_backend
} // namespace torch_tensorrt
