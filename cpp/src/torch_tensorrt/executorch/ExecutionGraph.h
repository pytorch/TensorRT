/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <NvInfer.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <executorch/runtime/core/error.h>

#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <utility>
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

  // Whether `stream` belongs to a regular CUDA context other than the one the engine ran in, while
  // that context is current. False until the engine has run in its own context.
  bool serves_another_regular_context(cudaStream_t stream) const;

  // Whether the graph for the context of the last call that could replay is recorded.
  bool is_captured() const {
    return last_graph_->graph_exec_ != nullptr;
  }

 private:
  // enqueue() in the context it found for the stream.
  ::executorch::runtime::Error enqueue_in(
      EngineHandle& handle,
      cudaStream_t stream,
      const std::vector<void*>& addresses,
      const std::vector<size_t>& binding_bytes,
      const void* scratch,
      CUcontext stream_context,
      CUgreenCtx stream_green,
      unsigned long long stream_green_id);
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
  // The regular context a plain warm-up ran in, where TensorRT can run this engine.
  CUcontext engine_context_ = nullptr;
  // The green context of this graph, or none. It is told apart by its id, since a destroyed one's
  // handle can be handed out again; the handle only creates the capture stream.
  CUgreenCtx capture_green_ = nullptr;
  unsigned long long capture_green_id_ = 0;
  int failed_captures_ = 0;
  // Times the caller's addresses moved away from a recording since the shapes or the profile last changed.
  int moves_ = 0;
  bool warmed_up_ = false;
  bool capture_failed_ = false;
  // A graph recorded on the caller's addresses replays with no copies while they stay put. Once they
  // have moved twice for one set of shapes, or a call from another regular context brings other ones,
  // the graph is recorded on buffers_ instead, and every call copies its inputs in and its outputs out.
  bool staged_ = false;
  // The graphs of green context calls, by green context id, most recently called first. This one
  // serves plain contexts.
  std::vector<std::pair<unsigned long long, std::unique_ptr<ExecutionGraph>>> greens_;
  const ExecutionGraph* last_graph_ = this;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
