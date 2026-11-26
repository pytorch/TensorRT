/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "torch_tensorrt/executorch/ExecutionGraph.h"
#include "EngineHandle.h"

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

// CUDA allows a green context handle on one CPU thread at a time, and engines that share a green context can run
// on several threads. Never freed, as cuda_graph_capture_mutex is, since handles can outlive other static objects.
std::mutex& green_context_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

// Finds the context `stream` runs its work in, and whether a graph can serve it: the stream's green
// context, or else the context current on this thread. Graph nodes keep the context they were
// recorded in, not that of the stream that launches them, so a green context's graph is recorded in
// that green context. enqueue_in picks the context a plain graph is recorded in.
bool recording_context(cudaStream_t stream, CUcontext& context, CUgreenCtx& green, unsigned long long& green_id) {
  static const auto get_context = driver_entry_point<PFN_cuStreamGetCtx_v12050>("cuStreamGetCtx", 12050);
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
  static const auto from_green = driver_entry_point<PFN_cuCtxFromGreenCtx_v12040>("cuCtxFromGreenCtx", 12040);
  static const auto context_id = driver_entry_point<PFN_cuCtxGetId_v12000>("cuCtxGetId", 12000);
  if (from_green == nullptr || context_id == nullptr) {
    return false;
  }
  CUcontext green_context = nullptr;
  const std::lock_guard<std::mutex> lock(green_context_mutex());
  return from_green(&green_context, green) == CUDA_SUCCESS && context_id(green_context, &green_id) == CUDA_SUCCESS;
}

cudaError_t create_capture_stream(CUgreenCtx green, cudaStream_t& stream) {
  if (green == nullptr) {
    return cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  }
  static const auto create_stream =
      driver_entry_point<PFN_cuGreenCtxStreamCreate_v12050>("cuGreenCtxStreamCreate", 12050);
  if (create_stream == nullptr) {
    return cudaErrorNotSupported;
  }
  CUstream created = nullptr;
  CUresult result = CUDA_SUCCESS;
  {
    const std::lock_guard<std::mutex> lock(green_context_mutex());
    result = create_stream(&created, green, CU_STREAM_NON_BLOCKING, 0);
  }
  if (result != CUDA_SUCCESS) {
    return result == CUDA_ERROR_OUT_OF_MEMORY ? cudaErrorMemoryAllocation : cudaErrorUnknown;
  }
  stream = created;
  return cudaSuccess;
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
    ET_LOG(
        Info,
        "TensorRTBackend::execute: freeing a CUDA graph buffer on the stream failed (%s); freeing it after a device wait",
        cudaGetErrorString(error));
    cudaGetLastError();
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
      "TensorRTBackend::execute: enqueueV3 failed. Likely an output with no address (see "
      "set_output_data_ptr), a stream of another device or of another regular CUDA context, "
      "or the per-thread stream in a green context.");
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
    // A failed release must not leave its error for the caller's next CUDA call, which may be another context's
    // execute().
    const cudaError_t error = cudaGraphExecDestroy(graph_exec_);
    if (error != cudaSuccess) {
      ET_LOG(Error, "TensorRTBackend: releasing a CUDA graph failed (%s)", cudaGetErrorString(error));
      cudaGetLastError();
    }
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
#ifdef TRT_MAJOR_RTX
  if (!handle.exec_ctx->isStreamCapturable(stream)) {
    reset_graph();
    return enqueue_plain(*handle.exec_ctx, stream);
  }
#endif
  CUcontext stream_context = nullptr;
  CUgreenCtx stream_green = nullptr;
  unsigned long long stream_green_id = 0;
  // A call that cannot replay keeps the graphs and their buffers for the next call that can.
  if (!recording_context(stream, stream_context, stream_green, stream_green_id)) {
    return enqueue_plain(*handle.exec_ctx, stream);
  }
  if (stream_green == nullptr) {
    last_graph_ = this;
    return enqueue_in(handle, stream, addresses, binding_bytes, scratch, stream_context, nullptr);
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
  return green.enqueue_in(handle, stream, addresses, binding_bytes, scratch, stream_context, stream_green);
}

Error ExecutionGraph::enqueue_in(
    EngineHandle& handle,
    cudaStream_t stream,
    const std::vector<void*>& addresses,
    const std::vector<size_t>& binding_bytes,
    const void* scratch,
    CUcontext stream_context,
    CUgreenCtx stream_green) {
  auto& context = *handle.exec_ctx;
  const size_t count = handle.num_inputs + handle.num_outputs;
  moves_.resize(count, 0);
  staged_.resize(count, false);
  // A plain graph and its buffers live in the context TensorRT ran the engine in, so a stream of
  // another regular context, where TensorRT cannot run, still replays it.
  const CUcontext record_context =
      stream_green == nullptr && handle.context != nullptr ? handle.context : stream_context;
  const bool other_context = record_context != stream_context;
  // An aliased output has to share its input's buffer, and a buffer of this graph for it would mean copying the
  // whole buffer back after every call, such as a whole KV cache for one new token. So such a binding stays on
  // the caller's address: while it moves the calls run plainly, and at its second counted move the graph stops
  // recording until kSettledCalls calls pass without another.
  const auto in_place_only = [&](size_t i) {
    return i < handle.num_inputs ? handle.input_is_alias_target[i]
                                 : handle.output_aliased_input_idx[i - handle.num_inputs] >= 0;
  };
  const auto stopped = [&] {
    for (size_t i = 0; i < count; ++i) {
      if (in_place_only(i) && moves_[i] > 1) {
        return true;
      }
    }
    return false;
  };
  bool in_place_counted = false;
  const auto count_move = [&](size_t i) {
    in_place_counted = in_place_counted || in_place_only(i);
    const bool was_stopped = stopped();
    // Every rule tells apart only no move, one, and more, and a cache that keeps moving while stopped would
    // otherwise overflow the count.
    moves_[i] = std::min(moves_[i] + 1, 2);
    if (!was_stopped && stopped()) {
      ET_LOG(
          Info,
          "TensorRTBackend::execute: the buffer of an aliased output moved away from its CUDA graph twice, so the "
          "engine runs with enqueueV3 until %d calls pass without another such move",
          kSettledCalls);
    }
  };
  // A recorded graph keeps the scratch address it was recorded with. The shared pool frees that
  // buffer when it grows, so a graph recorded on another buffer must not be replayed. A move of an aliased output
  // or its input away from a recording at the same shapes still counts, also between calls that ran without a graph
  // since the growth drops the warm-up that counts those, or each growth would let it record once more.
  if (scratch != scratch_) {
    if (!other_context && !capture_failed_ && !recorded_.empty() && handle.shape_epoch == shape_epoch_) {
      for (size_t i = 0; i < count; ++i) {
        if (in_place_only(i) && addresses[i] != recorded_[i] && addresses[i] != previous_[i]) {
          count_move(i);
        }
      }
    }
    reset_graph();
    scratch_ = scratch;
  }
  // A graph replays the kernels of the shapes and the profile it was captured under, so a change of
  // either must record again. Buffers that change along with the shapes are not counted as a move: new shapes
  // record again.
  // Each graph holds one recording, so alternating profiles every call never replays; a
  // recording per profile is the upgrade if that pattern matters.
  if (handle.shape_epoch != shape_epoch_) {
    reset_graph();
    capture_failed_ = false;
    own_buffers_failed_ = false;
    failed_captures_ = 0;
    // A buffer the caller moved away from recordings on it twice keeps moving under new shapes too, so it
    // stays on a buffer this graph owns instead of paying more recordings for every set of shapes.
    for (size_t i = 0; i < count; ++i) {
      staged_[i] = !in_place_only(i) && moves_[i] > 1;
    }
    staged_until_repeat_ = false;
    shape_epoch_ = handle.shape_epoch;
    // The last recording was under other shapes, so a buffer that leaves its addresses has not moved away from a
    // recording of these shapes, such as a decode cache after a prefill whose last chunk had the decode shapes.
    recorded_.clear();
  }
  if (capture_failed_) {
    return enqueue_plain(context, stream);
  }
  // A graph reads and writes the addresses it was recorded on. A caller that hands in new buffers on
  // every call, or alternates two sets, would record on every call, so a first recording goes on the
  // caller's addresses only when they repeat the previous call's, and on buffers this graph owns until
  // they do. A caller that moves a buffer away from a recording on it once gets a new recording on the
  // new one; one that keeps moving it switches to a buffer this graph owns at the second move. Each
  // buffer counts on its own, so the ones that stay put are read and written in place.
  const auto replays_on_addresses = [&] {
    for (size_t i = 0; i < count; ++i) {
      if (!staged_[i] && addresses[i] != recorded_[i]) {
        return false;
      }
    }
    return true;
  };
  const auto moved_since_previous = [&] {
    for (size_t i = 0; i < count; ++i) {
      if (!staged_[i] && addresses[i] != previous_[i]) {
        return true;
      }
    }
    return false;
  };
  if (!other_context) {
    bool waits_for_repeat = false;
    if (graph_exec_ != nullptr && !replays_on_addresses()) {
      drop_graph();
      for (size_t i = 0; i < count; ++i) {
        if (!staged_[i] && addresses[i] != recorded_[i]) {
          count_move(i);
          staged_[i] = !in_place_only(i) && moves_[i] > 1;
          waits_for_repeat = waits_for_repeat || in_place_only(i);
        }
      }
    } else if (graph_exec_ == nullptr && warmed_up_ && moved_since_previous()) {
      // Without buffers of its own, the graph runs these calls plainly.
      waits_for_repeat = own_buffers_failed_;
      for (size_t i = 0; i < count; ++i) {
        if (addresses[i] == previous_[i]) {
          continue;
        }
        if (in_place_only(i)) {
          // Counted here too, or a cache that alternates between two buffers after a recording would never stop.
          if (!recorded_.empty() && addresses[i] != recorded_[i]) {
            count_move(i);
          }
          waits_for_repeat = true;
        } else if (!own_buffers_failed_) {
          staged_[i] = true;
        }
      }
      if (!own_buffers_failed_) {
        staged_until_repeat_ = true;
      }
    }
    // This runs after either branch above, so a buffer that repeats goes back in place also on a call where another
    // moves. One that moved twice stays staged for good.
    if (staged_until_repeat_) {
      bool settled = false;
      staged_until_repeat_ = false;
      for (size_t i = 0; i < count; ++i) {
        if (staged_[i] && moves_[i] < 2 && addresses[i] == previous_[i]) {
          staged_[i] = false;
          settled = true;
        }
        // Recomputed only here. A recording for another context clears it, so the bindings it staged never settle.
        staged_until_repeat_ = staged_until_repeat_ || (staged_[i] && moves_[i] < 2);
      }
      if (settled) {
        drop_graph();
        if (std::find(staged_.begin(), staged_.end(), true) == staged_.end()) {
          release_buffers(stream);
        }
      }
    }
    // Moves that are rare next to the calls between them, such as a new cache for each conversation, never stop
    // replay, while a buffer that keeps moving stays without a graph.
    bool in_place_moved = previous_.empty();
    for (size_t i = 0; i < count && !in_place_moved; ++i) {
      in_place_moved = in_place_only(i) && addresses[i] != previous_[i];
    }
    settled_calls_ = in_place_counted ? 0 : std::min(settled_calls_ + 1, kSettledCalls);
    moving_calls_ = in_place_moved && waits_for_repeat ? std::min(moving_calls_ + 1, kSettledCalls + 1) : 0;
    if (settled_calls_ >= kSettledCalls) {
      for (size_t i = 0; i < count; ++i) {
        if (in_place_only(i)) {
          moves_[i] = 0;
        }
      }
    }
    previous_ = addresses;
    if (stopped()) {
      // Nothing records until the buffer settles, so buffers of this graph's own, such as a staged token's, would only
      // hold memory.
      release_buffers(stream);
      const auto error = enqueue_plain(context, stream);
      // A call that ran these shapes is their warm-up, so moves still count after a scratch growth reset it.
      warmed_up_ = warmed_up_ || error == Error::Ok;
      return error;
    }
    if (waits_for_repeat) {
      // A move away from no recording is not counted, so that separate prefill and decode caches keep replaying.
      // A cache that moves on every call then never records and never stops, so say why.
      if (moving_calls_ == kSettledCalls) {
        ET_LOG(
            Info,
            "TensorRTBackend::execute: the buffer of an aliased output moved on each of the last %d calls, so they "
            "ran with enqueueV3; a CUDA graph records on that buffer once a call repeats it",
            kSettledCalls);
      }
      return enqueue_plain(context, stream);
    }
  }
  // TensorRT cannot run on a stream of another regular context, so a call from one that no graph
  // serves records one in the engine's context, on buffers this graph owns. Until that recording can
  // go ahead the call fails. The graph and the failure count of the engine's own calls stay, unless the
  // call's shapes were new, since the shape change above has then reset both.
  const bool records_for_other_context = other_context && (graph_exec_ == nullptr || !replays_on_addresses());
  const bool own_buffers =
      records_for_other_context || std::find(staged_.begin(), staged_.end(), true) != staged_.end();
  const auto copies = [&](size_t i) { return records_for_other_context || staged_[i]; };
  std::unique_lock<std::shared_mutex> capture_lock(cuda_graph_capture_mutex(), std::defer_lock);
  if (records_for_other_context) {
    // That recording would put an aliased output on a buffer of this graph.
    if (handle.num_aliased_outputs > 0) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: an engine with aliased outputs replays a call on a stream of another regular "
          "CUDA context only on the buffers of a CUDA graph its own calls recorded, and %s",
          graph_exec_ != nullptr ? "this call brings other ones"
              : stopped()        ? "it records none while the buffer of an aliased output keeps moving"
                                 : "there is none");
      return Error::InvalidState;
    }
    // Pooled calls never get here: execute() refuses a pooled call from another regular context first.
    if (!warmed_up_) {
      ET_LOG(
          Error,
          "TensorRTBackend::execute: a call on a stream of another regular CUDA context needs a CUDA graph "
          "recorded in the engine's context, and the engine has not run these shapes there yet");
      return Error::InvalidState;
    }
    capture_lock.lock();
  }
  if (own_buffers) {
    buffers_.resize(count, nullptr);
    capacities_.resize(count, 0);
    for (size_t i = 0; i < count; ++i) {
      if (!copies(i)) {
        continue;
      }
      // TensorRT refuses a null address even for an empty tensor.
      const size_t bytes = binding_bytes[i] == 0 ? 1 : binding_bytes[i];
      if (capacities_[i] < bytes) {
        // New buffers need no warm-up, since warmed_up_ already says whether the engine ran these shapes
        // in its own context, and a graph that does not read this buffer stays valid.
        if (staged_[i]) {
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
          release_buffers(stream);
          // Refused by a global capture on another thread, like the cudaGraphInstantiate below, so a later call
          // still tries again.
          if (allocation == cudaErrorStreamCaptureUnsupported) {
            return enqueue_plain(context, stream);
          }
          if (staged_until_repeat_) {
            // Only the bindings that waited for a repeat go back to the caller's addresses, and calls run plainly
            // until one repeats the previous call's address for every binding read in place. A binding that moved
            // twice stays staged, and the next recording allocates a buffer for it again.
            staged_until_repeat_ = false;
            for (size_t j = 0; j < count; ++j) {
              staged_[j] = staged_[j] && moves_[j] > 1;
            }
            own_buffers_failed_ = true;
          } else {
            capture_failed_ = true;
          }
          return enqueue_plain(context, stream);
        }
        buffers_[i] = buffer;
        capacities_[i] = capacity;
      }
    }
    for (size_t i = 0; i < count; ++i) {
      if (!handle.bind_address(i, copies(i) ? buffers_[i] : addresses[i])) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: CUDA graph setTensorAddress failed for '%s'",
            binding_name(handle, i).c_str());
        return Error::InvalidState;
      }
      if (i < handle.num_inputs && binding_bytes[i] != 0 && copies(i)) {
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
      // A pooled caller holds its device's pool lock, so waiting here for a load, a destroy, or another
      // engine's recording on any device would stall every pooled engine on this device. It records on
      // a later call instead.
      (void)capture_lock.try_lock();
    }
  }
  if (capture_lock.owns_lock()) {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaError_t error = cudaErrorDeviceUninitialized;
    bool enqueued = true;
    bool refused_by_a_capture = false;
    {
      // The engine is captured on a stream of its own in record_context. The scope ends before the launch
      // below, which runs on the caller's stream, in whichever context it belongs to.
      ContextScope scope(record_context, other_context);
      cudaStream_t capture_stream = nullptr;
      if (scope.ok()) {
        error = create_capture_stream(stream_green, capture_stream);
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
        // A global capture on another thread refuses this, which says nothing about whether the engine can be
        // recorded, so it does not count toward turning replay off.
        refused_by_a_capture = error == cudaErrorStreamCaptureUnsupported;
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
      for (size_t i = 0; i < handle.num_inputs; ++i) {
        const nvinfer1::Dims dims = handle.bound_input_dims[i];
        if (dims.nbDims >= 0 && !handle.bind_input_shape(i, dims, true)) {
          ET_LOG(
              Error, "TensorRTBackend::execute: restoring the shape of '%s' failed", binding_name(handle, i).c_str());
          return Error::InvalidState;
        }
      }
      if (records_for_other_context) {
        return Error::InvalidState;
      }
      if (!refused_by_a_capture && ++failed_captures_ == kMaxCaptureAttempts) {
        ET_LOG(
            Error,
            "TensorRTBackend::execute: CUDA graph recording failed %d times; until its input shapes or profile "
            "change, this engine runs its calls from %s with enqueueV3%s",
            kMaxCaptureAttempts,
            stream_green != nullptr ? "this green context" : "plain streams",
            stream_green != nullptr ? "" : ", and calls from another regular CUDA context fail");
        capture_failed_ = true;
        release_buffers(stream);
      }
      return enqueue_plain(context, stream);
    }
    drop_graph();
    graph_exec_ = exec;
    if (records_for_other_context) {
      // Every call this graph serves copies every binding from now on until the shapes or the profile change, so a
      // repeat does not record again.
      staged_until_repeat_ = false;
      std::fill(staged_.begin(), staged_.end(), true);
    }
    recorded_ = addresses;
    for (size_t i = 0; i < count; ++i) {
      if (copies(i)) {
        recorded_[i] = buffers_[i];
      }
    }
    // A buffer this recording does not read is unused from now on.
    for (size_t i = 0; i < buffers_.size(); ++i) {
      if (buffers_[i] != nullptr && !copies(i)) {
        retire(buffers_[i], stream);
        buffers_[i] = nullptr;
        capacities_[i] = 0;
      }
    }
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
  for (size_t i = handle.num_inputs; i < count; ++i) {
    if (binding_bytes[i] != 0 && copies(i)) {
      const auto error = cuda_error(
          cudaMemcpyAsync(addresses[i], buffers_[i], binding_bytes[i], cudaMemcpyDefault, stream), "output copy");
      if (error != Error::Ok) {
        return error;
      }
    }
  }
  return Error::Ok;
}

} // namespace executorch_backend
} // namespace torch_tensorrt
