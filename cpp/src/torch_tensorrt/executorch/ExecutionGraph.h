/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <executorch/runtime/core/error.h>

#include <shared_mutex>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

struct EngineHandle;

// Loads and destruction share this lock; only the capture window takes it exclusively.
std::shared_mutex& cuda_graph_capture_mutex();

// enqueueV3, logging the error the delegate reports when TensorRT refuses the enqueue.
::executorch::runtime::Error enqueue_plain(nvinfer1::IExecutionContext& context, cudaStream_t stream);

// Owns the stable binding storage required by a captured TensorRT enqueue.
class ExecutionGraph {
 public:
  ExecutionGraph() = default;
  ExecutionGraph(const ExecutionGraph&) = delete;
  ExecutionGraph& operator=(const ExecutionGraph&) = delete;
  ~ExecutionGraph();

  // Requires the engine's device and a context to be current and the handle lock to be held. The
  // caller drains the stream before it releases that lock. binding_bytes holds one byte count per
  // binding, the inputs and then the outputs. scratch is the pooled activation scratch this call
  // installed on the context, or null when it installed none. The caller holds the pool's claim
  // until the stream is drained or the enqueue is recorded on the pool's handoff event, so the
  // buffer stays live for the work submitted here. Rebind every address before each call. Returns
  // Ok after submission, before the stream is drained.
  ::executorch::runtime::Error enqueue(
      EngineHandle& handle,
      cudaStream_t stream,
      const std::vector<size_t>& binding_bytes,
      const void* scratch);

  bool is_captured() const {
    return graph_exec_ != nullptr;
  }

 private:
  void reset_graph();
  void release_buffers(cudaStream_t stream);

  std::vector<void*> buffers_;
  std::vector<size_t> capacities_;
  std::vector<nvinfer1::Dims> shapes_;
  // Borrowed: the pool frees it on another engine's call. Compared, never dereferenced.
  const void* scratch_ = nullptr;
  cudaGraphExec_t graph_exec_ = nullptr;
  cudaStream_t capture_stream_ = nullptr;
  int failed_captures_ = 0;
  bool warmed_up_ = false;
  bool capture_failed_ = false;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
