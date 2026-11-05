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
  if (create_stream(&created, green, CU_STREAM_NON_BLOCKING, 0) != CUDA_SUCCESS) {
    return cudaErrorUnknown;
  }
  stream = created;
  return cudaSuccess;
}

// A stream from a green context the caller has destroyed can still be destroyed, but that reports an
// error, which must not be left behind for the caller's next CUDA call to find.
void destroy_stream(cudaStream_t stream) {
  if (cudaStreamDestroy(stream) != cudaSuccess) {
    cudaGetLastError();
  }
}

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
      "with set_output_data_ptr. Else the guard's stream is on another device, or is the "
      "per-thread stream inside a green context, which is invalid there.");
  return Error::InvalidState;
}

ExecutionGraph::~ExecutionGraph() {
  reset_graph();
  for (void* buffer : buffers_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
  }
  if (capture_stream_ != nullptr) {
    destroy_stream(capture_stream_);
  }
}

void ExecutionGraph::drop_graph() {
  if (graph_exec_ != nullptr) {
    cudaGraphExecDestroy(graph_exec_);
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
  // Green context calls keep a graph of their own, so a caller that also calls from a plain stream
  // replays on both instead of recording at every switch.
  last_call_green_ = stream_green != nullptr;
  if (last_call_green_) {
    if (green_ == nullptr) {
      green_ = std::make_unique<ExecutionGraph>();
    }
    return green_->enqueue_in(
        handle, stream, addresses, binding_bytes, scratch, stream_context, stream_green, stream_green_id);
  }
  return enqueue_in(handle, stream, addresses, binding_bytes, scratch, stream_context, nullptr, 0);
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
  // A graph runs in the context it was recorded in, so a stream from another one records its own.
  if (stream_context != capture_context_ || stream_green_id != capture_green_id_) {
    reset_graph();
    if (capture_stream_ != nullptr) {
      destroy_stream(capture_stream_);
      capture_stream_ = nullptr;
    }
    capture_context_ = stream_context;
    capture_green_ = stream_green;
    capture_green_id_ = stream_green_id;
  }
  // A graph reads and writes the addresses it was recorded on. A caller that moves its buffers once
  // gets a new recording on the new ones; one that keeps moving them, such as one alternating two
  // sets, would record on every call, so the second move switches to buffers this graph owns.
  if (!staged_ && graph_exec_ != nullptr && addresses != recorded_) {
    drop_graph();
    staged_ = ++moves_ > 1;
  }
  if (staged_) {
    buffers_.resize(count, nullptr);
    capacities_.resize(count, 0);
    for (size_t i = 0; i < count; ++i) {
      // TensorRT refuses a null address even for an empty tensor.
      const size_t bytes = binding_bytes[i] == 0 ? 1 : binding_bytes[i];
      if (capacities_[i] < bytes) {
        reset_graph();
        // Doubling keeps a shape that grows a little on every call from allocating on every call.
        const size_t capacity = std::max(bytes, capacities_[i] * 2);
        void* buffer = nullptr;
        const auto allocation = cudaMalloc(&buffer, capacity);
        if (allocation != cudaSuccess) {
          fall_back("buffer allocation failed", allocation);
          capture_failed_ = true;
          release_buffers(stream);
          return enqueue_plain(context, stream);
        }
        if (buffers_[i] != nullptr) {
          retire(buffers_[i], stream);
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

  // A pooled caller holds its device's pool lock, so waiting here for a load, a destroy, or another
  // engine's recording on any device would stall every pooled engine on this device. It records on a
  // later call instead.
  std::unique_lock<std::shared_mutex> capture_lock(cuda_graph_capture_mutex(), std::defer_lock);
  if (warmed_up_ && graph_exec_ == nullptr) {
    if (scratch == nullptr) {
      capture_lock.lock();
    } else {
      (void)capture_lock.try_lock();
    }
  }
  if (capture_lock.owns_lock()) {
    cudaGraph_t graph = nullptr;
    cudaError_t error = cudaSuccess;
    if (capture_stream_ == nullptr) {
      error = create_capture_stream(capture_green_, capture_stream_);
      if (error != cudaSuccess) {
        capture_stream_ = nullptr;
      }
    }
    bool enqueued = true;
    // Only this handle can submit to this stream, so capture cannot absorb another caller's work.
    if (error == cudaSuccess) {
      error = cudaStreamBeginCapture(capture_stream_, cudaStreamCaptureModeThreadLocal);
      if (error == cudaSuccess) {
        enqueued = context.enqueueV3(capture_stream_);
        error = cudaStreamEndCapture(capture_stream_, &graph);
      }
    }
    capture_lock.unlock();
    if (error == cudaSuccess && enqueued) {
      error = cudaGraphInstantiate(&graph_exec_, graph, nullptr, nullptr, 0);
      if (error != cudaSuccess) {
        graph_exec_ = nullptr;
      }
    }
    if (graph != nullptr) {
      cudaGraphDestroy(graph);
    }
    if (error != cudaSuccess || !enqueued) {
      if (enqueued) {
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
      if (++failed_captures_ == kMaxCaptureAttempts) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: CUDA graph recording failed %d times; this engine uses enqueueV3 until an "
            "input shape changes. Its outputs are not affected.",
            kMaxCaptureAttempts);
        capture_failed_ = true;
        release_buffers(stream);
      }
      return enqueue_plain(context, stream);
    }
    recorded_ = staged_ ? buffers_ : addresses;
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
