/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_tensorrt/executorch/ExecutionGraph.h"
#include "EngineHandle.h"

#include <cudaTypedefs.h>
#include <executorch/runtime/platform/log.h>

#include <algorithm>
#include <atomic>

namespace torch_tensorrt {
namespace executorch_backend {

using namespace ::executorch::runtime;

std::shared_mutex& cuda_graph_capture_mutex() {
  // Handles can outlive other static objects during process teardown.
  static auto* mutex = new std::shared_mutex();
  return *mutex;
}

namespace {
constexpr int kMaxCaptureAttempts = 3;
// Green contexts with a graph of their own per engine. An engine is rarely shared by more SM
// partitions than this, and each graph may hold buffers of its own.
constexpr size_t kMaxGreenGraphs = 4;

template <typename Function>
Function driver_entry(const char* name) {
  void* entry = nullptr;
  if (cudaGetDriverEntryPointByVersion(name, &entry, 12050, cudaEnableDefault) != cudaSuccess || entry == nullptr) {
    cudaGetLastError();
    return nullptr;
  }
  return reinterpret_cast<Function>(entry);
}

// Finds the context `stream` runs its work in, when a graph can be recorded there: the stream's green
// context, or else the context current on this thread. Graph nodes keep the context they were
// recorded in, not that of the stream that launches them, so a graph recorded anywhere else would run
// a green context's work outside its SMs.
bool recording_context(cudaStream_t stream, CUcontext& context, CUgreenCtx& green, unsigned long long& green_id) {
  static const auto get_context = driver_entry<PFN_cuStreamGetCtx_v12050>("cuStreamGetCtx");
  if (get_context == nullptr) {
    // Logging can wait for a lock held by another caller, so it must not hold the initialization guard.
    static std::atomic_flag logged = ATOMIC_FLAG_INIT;
    if (!logged.test_and_set(std::memory_order_relaxed)) {
      ET_LOG(Info, "TensorRTBackend::execute: CUDA graph replay needs a CUDA 12.5 or newer driver; using enqueueV3");
    }
    return false;
  }
  if (get_context(stream, &context, &green) != CUDA_SUCCESS) {
    return false;
  }
  // A green stream reports the primary context of its device, so this also refuses one on another device.
  CUcontext current_context = nullptr;
  CUgreenCtx current_green = nullptr;
  if (get_context(nullptr, &current_context, &current_green) != CUDA_SUCCESS || current_context != context) {
    return false;
  }
  if (green == nullptr) {
    green_id = 0;
    // A plain capture stream is created in the current context, so that one must not be green either.
    return current_green == nullptr;
  }
  // A destroyed green context's handle can be handed out again, but its id never is.
  static const auto from_green = driver_entry<PFN_cuCtxFromGreenCtx_v12040>("cuCtxFromGreenCtx");
  static const auto context_id = driver_entry<PFN_cuCtxGetId_v12000>("cuCtxGetId");
  CUcontext green_context = nullptr;
  return from_green != nullptr && context_id != nullptr && from_green(&green_context, green) == CUDA_SUCCESS &&
      context_id(green_context, &green_id) == CUDA_SUCCESS;
}

cudaError_t create_capture_stream(CUgreenCtx green, cudaStream_t& stream) {
  if (green == nullptr) {
    return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  static const auto create_stream = driver_entry<PFN_cuGreenCtxStreamCreate_v12050>("cuGreenCtxStreamCreate");
  if (create_stream == nullptr) {
    return cudaErrorNotSupported;
  }
  CUstream created = nullptr;
  const CUresult result = create_stream(&created, green, CU_STREAM_NON_BLOCKING, 0);
  if (result != CUDA_SUCCESS) {
    return result == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation : cudaErrorUnknown;
  }
  stream = created;
  return cudaSuccess;
}

// Makes `context` current on this thread until the end of the scope, when `needed`.
class ContextScope {
 public:
  ContextScope(CUcontext context, bool needed) {
    if (!needed) {
      ok_ = true;
      return;
    }
    static const auto push = driver_entry<PFN_cuCtxPushCurrent_v4000>("cuCtxPushCurrent");
    pushed_ = push != nullptr && push(context) == CUDA_SUCCESS;
    ok_ = pushed_;
  }
  ~ContextScope() {
    static const auto pop = driver_entry<PFN_cuCtxPopCurrent_v4000>("cuCtxPopCurrent");
    CUcontext popped = nullptr;
    if (pushed_ && pop != nullptr) {
      pop(&popped);
    }
  }
  ContextScope(const ContextScope&) = delete;
  ContextScope& operator=(const ContextScope&) = delete;
  bool ok() const {
    return ok_;
  }

 private:
  bool ok_ = false;
  bool pushed_ = false;
};

const std::string& binding_name(const EngineHandle& handle, size_t index) {
  return index < handle.num_inputs ? handle.input_binding_names[index]
                                   : handle.output_binding_names[index - handle.num_inputs];
}

Error cuda_error(cudaError_t error, const char* operation) {
  if (error == cudaSuccess) {
    return Error::Ok;
  }
  ET_LOG(Error, "TensorRTBackend::execute: CUDA graph %s failed: %s", operation, cudaGetErrorString(error));
  cudaGetLastError();
  return error == cudaErrorMemoryAllocation ? Error::MemoryAllocationFailed : Error::Internal;
}

void fall_back(const char* reason, cudaError_t error) {
  ET_LOG(Info, "TensorRTBackend::execute: CUDA graph %s (%s); using enqueueV3", reason, cudaGetErrorString(error));
  cudaGetLastError();
}

// TensorRT requires a green context to outlive the graphs recorded in it, but a caller that re-partitions its SMs
// may destroy one first. Releasing that graph then fails, and the error must not reach the caller's next CUDA call,
// which may be another context's execute().
void released(cudaError_t error) {
  if (error != cudaSuccess) {
    ET_LOG(
        Info,
        "TensorRTBackend: releasing a CUDA graph failed (%s); its green context was likely destroyed before the engine",
        cudaGetErrorString(error));
    cudaGetLastError();
  }
}

// A stream-ordered free that fails still leaves a buffer this call is done with.
void retire(void* buffer, cudaStream_t stream) {
  const auto error = cudaFreeAsync(buffer, stream);
  if (error != cudaSuccess) {
    fall_back("stream-ordered free failed, freeing after a device wait", error);
    cudaFree(buffer);
    cudaGetLastError();
  }
}
} // namespace

Error enqueue_plain(nvinfer1::IExecutionContext& context, cudaStream_t stream) {
  if (context.enqueueV3(stream)) {
    return Error::Ok;
  }
  ET_LOG(
      Error,
      "TensorRTBackend::execute: enqueueV3 failed. Likely an output with no address: supply each "
      "with set_output_data_ptr. Else the guard's stream is on another device, belongs to another "
      "regular CUDA context, where only a CUDA graph the engine recorded in its own context can run, "
      "or is the per-thread stream inside a green context, which is invalid there.");
  return Error::InvalidState;
}

ExecutionGraph::~ExecutionGraph() {
  reset_graph();
  for (void* buffer : buffers_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
  }
}

void ExecutionGraph::drop_graph() {
  if (graph_exec_ != nullptr) {
    released(cudaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }
}

void ExecutionGraph::reset_graph() {
  drop_graph();
  warmed_up_ = false;
}

void ExecutionGraph::release_buffers(cudaStream_t stream) {
  for (size_t i = 0; i < buffers_.size(); ++i) {
    if (buffers_[i] != nullptr) {
      retire(buffers_[i], stream);
      buffers_[i] = nullptr;
      capacities_[i] = 0;
    }
  }
}

Error ExecutionGraph::enqueue(
    EngineHandle& handle,
    cudaStream_t stream,
    const std::vector<void*>& addresses,
    const std::vector<size_t>& binding_bytes,
    const void* scratch) {
  const size_t count = handle.num_inputs + handle.num_outputs;
  if (addresses.size() != count || binding_bytes.size() != count) {
    return Error::InvalidArgument;
  }
  CUcontext stream_context = nullptr;
  CUgreenCtx stream_green = nullptr;
  unsigned long long stream_green_id = 0;
  // A call that cannot replay keeps the graphs and their buffers for the next call that can.
  if (!recording_context(stream, stream_context, stream_green, stream_green_id)) {
    return enqueue_plain(*handle.exec_ctx, stream);
  }
  if (stream_green == nullptr) {
    last_graph_ = this;
    return enqueue_in(handle, stream, addresses, binding_bytes, scratch, stream_context, nullptr, 0);
  }
  // Each green context keeps a graph of its own, so a caller that also calls from a plain stream, or
  // from a few green contexts in turn, replays on each instead of recording at every switch.
  auto found =
      std::find_if(greens_.begin(), greens_.end(), [&](const auto& green) { return green.first == stream_green_id; });
  if (found == greens_.end()) {
    if (greens_.size() == kMaxGreenGraphs) {
      greens_.back().second->release_buffers(stream);
      greens_.pop_back();
    }
    greens_.emplace_back(stream_green_id, std::make_unique<ExecutionGraph>());
    found = greens_.end() - 1;
  }
  // Most recent first, so the one dropped for a new green context is the one called longest ago.
  std::rotate(greens_.begin(), found, found + 1);
  ExecutionGraph& green = *greens_.front().second;
  last_graph_ = &green;
  return green.enqueue_in(
      handle, stream, addresses, binding_bytes, scratch, stream_context, stream_green, stream_green_id);
}

Error ExecutionGraph::enqueue_in(
    EngineHandle& handle,
    cudaStream_t stream,
    const std::vector<void*>& addresses,
    const std::vector<size_t>& binding_bytes,
    const void* scratch,
    CUcontext stream_context,
    CUgreenCtx stream_green,
    unsigned long long stream_green_id) {
  auto& context = *handle.exec_ctx;
  const size_t count = handle.num_inputs + handle.num_outputs;
  // A recorded graph keeps the scratch address it was recorded with. The shared pool frees that
  // buffer when it grows, so a graph recorded on another buffer must not be replayed.
  if (scratch != scratch_) {
    reset_graph();
    scratch_ = scratch;
  }
  // A graph replays the kernels of the shapes and the profile it was captured under, so a change of
  // either must record again.
  // ponytail: one graph per handle and kind of context, so alternating profiles every call never
  // replays; a graph per profile is the upgrade if that pattern matters.
  if (handle.shape_epoch != shape_epoch_) {
    reset_graph();
    capture_failed_ = false;
    failed_captures_ = 0;
    moves_ = 0;
    staged_ = false;
    shape_epoch_ = handle.shape_epoch;
  }
  if (capture_failed_) {
    return enqueue_plain(context, stream);
  }
  // Each green context has a graph of its own, which learns its context on its first call.
  if (stream_green_id != capture_green_id_) {
    capture_green_ = stream_green;
    capture_green_id_ = stream_green_id;
  }
  // A plain graph and its buffers live in the context TensorRT ran the engine in, so a stream of
  // another regular context, where TensorRT cannot run, still replays it.
  const CUcontext record_context =
      capture_green_ == nullptr && engine_context_ != nullptr ? engine_context_ : stream_context;
  const bool other_context = record_context != stream_context;
  // A graph reads and writes the addresses it was recorded on. A caller that moves its buffers once
  // gets a new recording on the new ones; one that keeps moving them, such as one alternating two
  // sets, would record on every call, so the second move switches to buffers this graph owns.
  if (!other_context && !staged_ && addresses != recorded_ && graph_exec_ != nullptr) {
    drop_graph();
    staged_ = ++moves_ > 1;
  }
  // TensorRT cannot run on a stream of another regular context, so a call from one that no graph
  // serves records one in the engine's context, on buffers this graph owns. Until that recording can
  // go ahead the call fails, and the graph and the failure count of the engine's own calls stay.
  const bool records_for_other_context =
      other_context && (graph_exec_ == nullptr || (!staged_ && addresses != recorded_));
  const bool own_buffers = staged_ || records_for_other_context;
  // A pooled caller holds its device's pool lock, so waiting here for a load, a destroy, or another
  // engine's recording on any device would stall every pooled engine on this device. It records on a
  // later call instead.
  std::unique_lock<std::shared_mutex> capture_lock(cuda_graph_capture_mutex(), std::defer_lock);
  if (records_for_other_context) {
    const char* reason = nullptr;
    if (!warmed_up_) {
      reason = "the engine has not run these shapes in its own context yet";
    } else if (scratch == nullptr) {
      capture_lock.lock();
    } else if (!capture_lock.try_lock()) {
      reason = "another engine is being loaded, destroyed or recorded";
    }
    if (reason != nullptr) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: a call on a stream of another regular CUDA context needs a CUDA graph "
          "recorded in the engine's context, and none can be recorded now: %s",
          reason);
      return Error::InvalidState;
    }
  }
  if (own_buffers) {
    buffers_.resize(count, nullptr);
    capacities_.resize(count, 0);
    for (size_t i = 0; i < count; ++i) {
      // TensorRT refuses a null address even for an empty tensor.
      const size_t bytes = binding_bytes[i] == 0 ? 1 : binding_bytes[i];
      if (capacities_[i] < bytes) {
        // A caller in another context cannot run a warm-up there, and warmed_up_ already says whether
        // the engine ran these shapes in its own context. A graph on the caller's buffers stays valid.
        if (!other_context) {
          reset_graph();
        } else if (staged_) {
          drop_graph();
        }
        // Doubling keeps shapes that grow a little at a time from allocating at every growth.
        const size_t capacity = std::max(bytes, capacities_[i] * 2);
        void* buffer = nullptr;
        cudaError_t allocation = cudaErrorDeviceUninitialized;
        {
          ContextScope scope(record_context, other_context);
          if (scope.ok()) {
            allocation = cudaMalloc(&buffer, capacity);
            if (allocation == cudaSuccess && buffers_[i] != nullptr) {
              retire(buffers_[i], stream);
            }
          }
        }
        if (allocation != cudaSuccess) {
          if (records_for_other_context) {
            ET_LOG(
                Error,
                "TensorRTBackend::execute: allocating buffers for a call on a stream of another regular CUDA "
                "context failed (%s)",
                cudaGetErrorString(allocation));
            cudaGetLastError();
            return Error::MemoryAllocationFailed;
          }
          fall_back("buffer allocation failed", allocation);
          capture_failed_ = true;
          release_buffers(stream);
          return enqueue_plain(context, stream);
        }
        buffers_[i] = buffer;
        capacities_[i] = capacity;
      }
    }
    for (size_t i = 0; i < count; ++i) {
      if (!handle.bind_address(i, buffers_[i])) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: CUDA graph setTensorAddress failed for '%s'",
            binding_name(handle, i).c_str());
        return Error::InvalidState;
      }
      if (i < handle.num_inputs && binding_bytes[i] != 0) {
        const auto error = cuda_error(
            cudaMemcpyAsync(buffers_[i], addresses[i], binding_bytes[i], cudaMemcpyDefault, stream), "input copy");
        if (error != Error::Ok) {
          return error;
        }
      }
    }
  }

  if (!capture_lock.owns_lock() && warmed_up_ && graph_exec_ == nullptr) {
    if (scratch == nullptr) {
      capture_lock.lock();
    } else {
      (void)capture_lock.try_lock();
    }
  }
  if (capture_lock.owns_lock()) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaError_t error = cudaErrorDeviceUninitialized;
    bool enqueued = true;
    {
      // The launch below runs on the caller's stream, in whichever context it belongs to.
      ContextScope scope(record_context, other_context);
      cudaStream_t capture_stream = nullptr;
      if (scope.ok()) {
        error = create_capture_stream(capture_green_, capture_stream);
      }
      // Only this handle can submit to this stream, so capture cannot absorb another caller's work.
      if (error == cudaSuccess) {
        error = cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeThreadLocal);
        if (error == cudaSuccess) {
          enqueued = context.enqueueV3(capture_stream);
          error = cudaStreamEndCapture(capture_stream, &graph);
        }
        // The graph does not need its stream, and a stream kept in a green context could not be freed
        // once that context is destroyed.
        (void)cuda_error(cudaStreamDestroy(capture_stream), "capture stream destroy");
      }
      capture_lock.unlock();
      if (error == cudaSuccess && enqueued) {
        error = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
        if (error != cudaSuccess) {
          exec = nullptr;
        }
      }
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
      }
    }
    if (error != cudaSuccess || !enqueued) {
      if (records_for_other_context) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: recording a CUDA graph for a call on a stream of another regular CUDA "
            "context failed (%s); the engine keeps the graph of its own calls",
            enqueued ? cudaGetErrorString(error) : "TensorRT refused the enqueue");
        cudaGetLastError();
      } else if (enqueued) {
        fall_back("capture failed", error);
      } else {
        ET_LOG(
            Info, "TensorRTBackend::execute: TensorRT refused the enqueue during CUDA graph capture; using enqueueV3");
        cudaGetLastError();
      }
      // Rebound unconditionally: whatever made TensorRT refuse may have changed what the context holds.
      handle.forget_bound_addresses();
      for (size_t i = 0; i < count; ++i) {
        if (!handle.bind_address(i, addresses[i])) {
          ET_LOG(Error, "TensorRTBackend::execute: restoring binding '%s' failed", binding_name(handle, i).c_str());
          return Error::InvalidState;
        }
      }
      if (records_for_other_context) {
        return Error::InvalidState;
      }
      if (++failed_captures_ == kMaxCaptureAttempts) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: CUDA graph recording failed %d times; until its input shapes or profile "
            "change, this engine runs its calls from %s with enqueueV3%s",
            kMaxCaptureAttempts,
            capture_green_ != nullptr ? "this green context" : "plain streams",
            capture_green_ != nullptr ? "" : ", and calls from another regular CUDA context fail");
        capture_failed_ = true;
        release_buffers(stream);
      }
      return enqueue_plain(context, stream);
    }
    drop_graph();
    graph_exec_ = exec;
    staged_ = own_buffers;
    recorded_ = own_buffers ? buffers_ : addresses;
  }
  if (graph_exec_ != nullptr) {
    const auto error = cuda_error(cudaGraphLaunch(graph_exec_, stream), "launch");
    if (error != Error::Ok) {
      return error;
    }
  } else {
    const auto error = enqueue_plain(context, stream);
    if (error != Error::Ok) {
      return error;
    }
    warmed_up_ = true;
    // The graph's buffers stay in the engine's context, so a call from another one must not move it.
    if (capture_green_ == nullptr && !other_context) {
      engine_context_ = stream_context;
    }
  }
  if (staged_) {
    for (size_t i = handle.num_inputs; i < count; ++i) {
      if (binding_bytes[i] != 0) {
        const auto error = cuda_error(
            cudaMemcpyAsync(addresses[i], buffers_[i], binding_bytes[i], cudaMemcpyDefault, stream), "output copy");
        if (error != Error::Ok) {
          return error;
        }
      }
    }
  }
  return Error::Ok;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
