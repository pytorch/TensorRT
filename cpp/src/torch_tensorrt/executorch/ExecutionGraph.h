/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <executorch/runtime/core/error.h>

#include <cstdint>
#include <shared_mutex>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

struct EngineHandle;

// Loads and destruction share this lock; only the capture window takes it exclusively.
std::shared_mutex& cuda_graph_capture_mutex();

// enqueueV3, logging the error the delegate reports when TensorRT refuses the enqueue.
::executorch::runtime::Error enqueue_plain(nvinfer1::IExecutionContext& context, cudaStream_t stream);

// Records a TensorRT enqueue into a CUDA graph and replays it: on the caller's buffers while they
// stay put, and on buffers of its own once they keep moving.
class ExecutionGraph {
 public:
  ExecutionGraph() = default;
  ExecutionGraph(const ExecutionGraph&) = delete;
  ExecutionGraph& operator=(const ExecutionGraph&) = delete;
  ~ExecutionGraph();

  // Requires the engine's device and a context to be current and the handle lock to be held. The
  // caller drains the stream before it releases that lock. addresses and binding_bytes hold one entry
  // per binding, the inputs and then the outputs, and the context must hold those addresses on entry.
  // scratch is the pooled activation scratch this call installed on the context, or null when it
  // installed none. The caller holds the pool's claim until the stream is drained or the enqueue is
  // recorded on the pool's handoff event, so the buffer stays live for the work submitted here. With
  // scratch set, a call never waits for cuda_graph_capture_mutex(): if it is taken, recording moves
  // to a later call. The context may hold other addresses on return. A change of input shapes or of
  // the optimization profile is seen only through handle.shape_epoch. Returns Ok after submission,
  // before the stream is drained.
  ::executorch::runtime::Error enqueue(
      EngineHandle& handle,
      cudaStream_t stream,
      const std::vector<void*>& addresses,
      const std::vector<size_t>& binding_bytes,
      const void* scratch);

  bool is_captured() const {
    return graph_exec_ != nullptr;
  }

 private:
  void drop_graph();
  void reset_graph();
  void release_buffers(cudaStream_t stream);

  std::vector<void*> buffers_;
  std::vector<size_t> capacities_;
  // The binding addresses the graph was recorded on.
  std::vector<void*> recorded_;
  // Borrowed: any growth of the pool frees it, this engine's own included. Compared, never
  // dereferenced. A later buffer at the same address keeps the graph safely, since the pool always
  // hands over at least what these shapes need.
  const void* scratch_ = nullptr;
  uint64_t shape_epoch_ = 0;
  cudaGraphExec_t graph_exec_ = nullptr;
  cudaStream_t capture_stream_ = nullptr;
  int failed_captures_ = 0;
  // Times the caller's addresses moved away from a recording on them, across shapes and profiles.
  int moves_ = 0;
  bool warmed_up_ = false;
  bool capture_failed_ = false;
  // A graph recorded on the caller's addresses replays with no copies while they stay put. Once they
  // have moved twice in total, the graph is recorded on buffers_ instead, and every
  // call copies its inputs in and its outputs out. That lasts across changes of shapes and profile.
  bool staged_ = false;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
