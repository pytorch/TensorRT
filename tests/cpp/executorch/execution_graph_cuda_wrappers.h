/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <cuda_runtime.h>

#include <atomic>
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

// The delegate looks up each driver entry point once per process, so these count and fail calls on every thread.
extern std::atomic<int> primary_context_retains;
extern std::atomic<int> primary_context_releases;
// The next this many cuStreamWriteValue64 calls fail without being queued.
extern std::atomic<int> fail_flag_writes;

// While alive, cuCtxGetId reports a new id for every context under its old handle, as for a new context the
// driver gives a destroyed one's handle.
struct RenewedContextIds {
  RenewedContextIds();
  ~RenewedContextIds();
  RenewedContextIds(const RenewedContextIds&) = delete;
  RenewedContextIds& operator=(const RenewedContextIds&) = delete;
};

// While alive, each cuCtxFromGreenCtx takes 200 microseconds, so calls from 2 threads that are not serialized
// overlap. most_at_once() is the most calls that were inside it at the same time.
struct SlowGreenContextCalls {
  SlowGreenContextCalls();
  ~SlowGreenContextCalls();
  SlowGreenContextCalls(const SlowGreenContextCalls&) = delete;
  SlowGreenContextCalls& operator=(const SlowGreenContextCalls&) = delete;
  int most_at_once() const;
};

// Linker wrappers observe only this thread and only while this scope is alive.
struct CudaCalls {
  CudaCalls();
  ~CudaCalls();
  CudaCalls(const CudaCalls&) = delete;
  CudaCalls& operator=(const CudaCalls&) = delete;

  size_t malloc_calls = 0;
  size_t fail_malloc_call = 0;
  // Each allocation that succeeds is filled with this byte before it is handed back, unless negative.
  int fill_allocations_with = -1;
  size_t free_calls = 0;
  size_t memcpy_calls = 0;
  // The copy with this 1-based count fails without being queued.
  size_t fail_memcpy_call = 0;
  // The next this many captures fail.
  size_t fail_captures = 0;
  size_t event_record_calls = 0;
  // The event record with this 1-based count fails without being queued.
  size_t fail_event_record_call = 0;
  // The next this many cudaEventCreateWithFlags calls fail without creating an event.
  size_t fail_event_creates = 0;
  // Each cudaStreamWaitEvent returns success without being queued, as the driver bug the handoff flag guards against.
  bool drop_stream_waits = false;
  size_t dropped_stream_waits = 0;
  bool disable_memory_pools = false;
  bool disable_stream_context = false;
  size_t stream_context_queries = 0;
  size_t memory_pool_queries = 0;
  cudaError_t launch_error = cudaSuccess;
  std::vector<cudaStream_t> captures;
  std::vector<cudaStream_t> launches;
  std::vector<cudaStream_t> retirements;
  std::vector<cudaStream_t> destroyed_streams;
  // What each cudaGraphExecDestroy returned, since the delegate clears a failed release.
  std::vector<cudaError_t> graph_exec_destroys;
  // Each cudaGraphExecDestroy releases the graph, then returns a CUDA error that it also leaves pending.
  bool fail_graph_exec_destroys = false;
  std::function<void(cudaStream_t)> during_capture;
  std::function<void(cudaGraph_t)> after_capture;
  std::function<void(cudaStream_t)> after_launch;
  std::function<void(cudaStream_t)> before_synchronize;
  std::vector<CudaCall> operations;

 private:
  CudaCalls* previous_;
};

} // namespace testing
} // namespace executorch_backend
} // namespace torch_tensorrt
