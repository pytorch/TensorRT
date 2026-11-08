/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "execution_graph_cuda_wrappers.h"

#include <cudaTypedefs.h>

#include <chrono>
#include <cstring>
#include <thread>

namespace torch_tensorrt {
namespace executorch_backend {
namespace testing {

thread_local CudaCalls* active_calls = nullptr;
std::atomic<int> primary_context_retains{0};
std::atomic<int> primary_context_releases{0};

namespace {
PFN_cuCtxGetId_v12000 real_context_id = nullptr;
PFN_cuCtxFromGreenCtx_v12040 real_from_green = nullptr;
std::atomic<bool> slow_green_calls{false};
std::atomic<int> green_calls_inside{0};
std::atomic<int> most_green_calls_inside{0};

CUresult slow_from_green(CUcontext* context, CUgreenCtx green) {
  const int inside = ++green_calls_inside;
  int most = most_green_calls_inside.load();
  while (inside > most && !most_green_calls_inside.compare_exchange_weak(most, inside)) {
  }
  if (slow_green_calls) {
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  const CUresult result = real_from_green(context, green);
  --green_calls_inside;
  return result;
}
std::atomic<unsigned long long> context_id_offset{0};
PFN_cuDevicePrimaryCtxRetain_v7000 real_retain = nullptr;
PFN_cuDevicePrimaryCtxRelease_v11000 real_release = nullptr;

CUresult renewable_context_id(CUcontext context, unsigned long long* id) {
  const CUresult result = real_context_id(context, id);
  if (result == CUDA_SUCCESS) {
    *id += context_id_offset.load();
  }
  return result;
}

CUresult counting_retain(CUcontext* context, CUdevice device) {
  const CUresult result = real_retain(context, device);
  if (result == CUDA_SUCCESS) {
    ++primary_context_retains;
  }
  return result;
}

CUresult counting_release(CUdevice device) {
  const CUresult result = real_release(device);
  if (result == CUDA_SUCCESS) {
    ++primary_context_releases;
  }
  return result;
}
} // namespace

// Far above any id the driver hands out in one process, since it counts them up from 1.
RenewedContextIds::RenewedContextIds() {
  context_id_offset = 1ull << 40;
}

RenewedContextIds::~RenewedContextIds() {
  context_id_offset = 0;
}

SlowGreenContextCalls::SlowGreenContextCalls() {
  most_green_calls_inside = 0;
  slow_green_calls = true;
}

SlowGreenContextCalls::~SlowGreenContextCalls() {
  slow_green_calls = false;
}

int SlowGreenContextCalls::most_at_once() const {
  return most_green_calls_inside.load();
}

CudaCalls::CudaCalls() : previous_(active_calls) {
  active_calls = this;
}

CudaCalls::~CudaCalls() {
  active_calls = previous_;
}

} // namespace testing
} // namespace executorch_backend
} // namespace torch_tensorrt

using namespace torch_tensorrt::executorch_backend::testing;

extern "C" {
cudaError_t __real_cudaMalloc(void**, size_t);
cudaError_t __real_cudaFree(void*);
cudaError_t __real_cudaFreeAsync(void*, cudaStream_t);
cudaError_t __real_cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t __real_cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
cudaError_t __real_cudaStreamEndCapture(cudaStream_t, cudaGraph_t*);
cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t __real_cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
cudaError_t __real_cudaStreamDestroy(cudaStream_t);
cudaError_t __real_cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t __real_cudaEventCreateWithFlags(cudaEvent_t*, unsigned int);
cudaError_t __real_cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t __real_cudaGetDriverEntryPointByVersion(
    const char*,
    void**,
    unsigned int,
    unsigned long long,
    cudaDriverEntryPointQueryResult*);

cudaError_t __wrap_cudaGetDriverEntryPointByVersion(
    const char* name,
    void** entry,
    unsigned int version,
    unsigned long long flags,
    cudaDriverEntryPointQueryResult* status) {
  if (active_calls && std::strcmp(name, "cuStreamGetCtx") == 0) {
    ++active_calls->stream_context_queries;
    if (active_calls->disable_stream_context) {
      *entry = nullptr;
      return cudaErrorNotSupported;
    }
  }
  const auto result = __real_cudaGetDriverEntryPointByVersion(name, entry, version, flags, status);
  if (result != cudaSuccess || *entry == nullptr) {
    return result;
  }
  if (std::strcmp(name, "cuCtxFromGreenCtx") == 0) {
    real_from_green = reinterpret_cast<PFN_cuCtxFromGreenCtx_v12040>(*entry);
    *entry = reinterpret_cast<void*>(&slow_from_green);
  } else if (std::strcmp(name, "cuCtxGetId") == 0) {
    real_context_id = reinterpret_cast<PFN_cuCtxGetId_v12000>(*entry);
    *entry = reinterpret_cast<void*>(&renewable_context_id);
  } else if (std::strcmp(name, "cuDevicePrimaryCtxRetain") == 0) {
    real_retain = reinterpret_cast<PFN_cuDevicePrimaryCtxRetain_v7000>(*entry);
    *entry = reinterpret_cast<void*>(&counting_retain);
  } else if (std::strcmp(name, "cuDevicePrimaryCtxRelease") == 0) {
    real_release = reinterpret_cast<PFN_cuDevicePrimaryCtxRelease_v11000>(*entry);
    *entry = reinterpret_cast<void*>(&counting_release);
  }
  return result;
}

cudaError_t __wrap_cudaDeviceGetAttribute(int* value, cudaDeviceAttr attribute, int device) {
  if (active_calls && attribute == cudaDevAttrMemoryPoolsSupported) {
    ++active_calls->memory_pool_queries;
    if (active_calls->disable_memory_pools) {
      *value = 0;
      return cudaSuccess;
    }
  }
  return __real_cudaDeviceGetAttribute(value, attribute, device);
}

cudaError_t __wrap_cudaMalloc(void** pointer, size_t bytes) {
  if (active_calls && ++active_calls->malloc_calls == active_calls->fail_malloc_call) {
    *pointer = nullptr;
    return cudaErrorMemoryAllocation;
  }
  return __real_cudaMalloc(pointer, bytes);
}

cudaError_t __wrap_cudaFree(void* pointer) {
  if (active_calls && pointer) {
    ++active_calls->free_calls;
  }
  return __real_cudaFree(pointer);
}

cudaError_t __wrap_cudaFreeAsync(void* pointer, cudaStream_t stream) {
  if (active_calls && pointer) {
    active_calls->retirements.push_back(stream);
  }
  return __real_cudaFreeAsync(pointer, stream);
}

cudaError_t __wrap_cudaMemcpyAsync(
    void* destination,
    const void* source,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream) {
  if (active_calls) {
    active_calls->operations.push_back({CudaCallKind::Copy, stream});
    if (++active_calls->memcpy_calls == active_calls->fail_memcpy_call) {
      return cudaErrorInvalidValue;
    }
  }
  return __real_cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

cudaError_t __wrap_cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
  if (active_calls) {
    active_calls->captures.push_back(stream);
    active_calls->operations.push_back({CudaCallKind::Capture, stream});
    if (active_calls->fail_captures > 0) {
      --active_calls->fail_captures;
      return cudaErrorStreamCaptureUnsupported;
    }
  }
  const auto result = __real_cudaStreamBeginCapture(stream, mode);
  if (result == cudaSuccess && active_calls && active_calls->during_capture) {
    active_calls->during_capture(stream);
  }
  return result;
}

cudaError_t __wrap_cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
  const auto result = __real_cudaStreamEndCapture(stream, graph);
  if (result == cudaSuccess && active_calls && active_calls->after_capture) {
    active_calls->after_capture(*graph);
  }
  return result;
}

cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t graph, cudaStream_t stream) {
  if (active_calls) {
    active_calls->launches.push_back(stream);
    active_calls->operations.push_back({CudaCallKind::Launch, stream});
    if (active_calls->launch_error != cudaSuccess) {
      return active_calls->launch_error;
    }
  }
  const auto result = __real_cudaGraphLaunch(graph, stream);
  if (result == cudaSuccess && active_calls && active_calls->after_launch) {
    active_calls->after_launch(stream);
  }
  return result;
}

cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  if (active_calls) {
    active_calls->operations.push_back({CudaCallKind::Synchronize, stream});
    if (active_calls->before_synchronize) {
      active_calls->before_synchronize(stream);
    }
  }
  return __real_cudaStreamSynchronize(stream);
}

cudaError_t __wrap_cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
  if (active_calls && active_calls->fail_event_creates > 0) {
    --active_calls->fail_event_creates;
    return cudaErrorMemoryAllocation;
  }
  return __real_cudaEventCreateWithFlags(event, flags);
}

cudaError_t __wrap_cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  if (active_calls && ++active_calls->event_record_calls == active_calls->fail_event_record_call) {
    return cudaErrorInvalidResourceHandle;
  }
  return __real_cudaEventRecord(event, stream);
}

cudaError_t __wrap_cudaStreamDestroy(cudaStream_t stream) {
  if (active_calls) {
    active_calls->destroyed_streams.push_back(stream);
  }
  return __real_cudaStreamDestroy(stream);
}

cudaError_t __wrap_cudaGraphExecDestroy(cudaGraphExec_t graph) {
  auto result = __real_cudaGraphExecDestroy(graph);
  if (active_calls && active_calls->fail_graph_exec_destroys) {
    // An invalid device fails without changing the current one, and leaves its error for cudaGetLastError.
    result = cudaSetDevice(-1);
  }
  if (active_calls) {
    active_calls->graph_exec_destroys.push_back(result);
  }
  return result;
}
}
