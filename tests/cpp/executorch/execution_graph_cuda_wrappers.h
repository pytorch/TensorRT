/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {
namespace testing {

enum class CudaCallKind { Copy, Capture, Launch, Synchronize };

struct CudaCall {
  CudaCallKind kind;
  cudaStream_t stream;
};

// Linker wrappers observe only this thread and only while this scope is alive.
struct CudaCalls {
  CudaCalls();
  ~CudaCalls();
  CudaCalls(const CudaCalls&) = delete;
  CudaCalls& operator=(const CudaCalls&) = delete;

  size_t malloc_calls = 0;
  size_t fail_malloc_call = 0;
  size_t free_calls = 0;
  size_t memcpy_calls = 0;
  // The copy with this 1-based count fails without being queued.
  size_t fail_memcpy_call = 0;
  // The next this many captures fail.
  size_t fail_captures = 0;
  bool disable_memory_pools = false;
  bool disable_stream_context = false;
  size_t stream_context_queries = 0;
  size_t memory_pool_queries = 0;
  cudaError_t launch_error = cudaSuccess;
  std::vector<cudaStream_t> captures;
  std::vector<cudaStream_t> launches;
  std::vector<cudaStream_t> retirements;
  std::function<void(cudaStream_t)> during_capture;
  std::function<void(cudaStream_t)> after_launch;
  std::function<void(cudaStream_t)> before_synchronize;
  std::vector<CudaCall> operations;

 private:
  CudaCalls* previous_;
};

} // namespace testing
} // namespace executorch_backend
} // namespace torch_tensorrt
