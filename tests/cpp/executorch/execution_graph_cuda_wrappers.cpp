/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "execution_graph_cuda_wrappers.h"

#include <cstring>

namespace torch_tensorrt {
namespace executorch_backend {
namespace testing {

thread_local CudaCalls* active_calls = nullptr;

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
cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t __real_cudaDeviceGetAttribute(int*, cudaDeviceAttr, int);
cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
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
  return __real_cudaGetDriverEntryPointByVersion(name, entry, version, flags, status);
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
}
