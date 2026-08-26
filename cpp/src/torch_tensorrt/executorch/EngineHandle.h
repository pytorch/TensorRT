/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Private state of a TensorRT ExecuTorch delegate.
 *
 * This header is deliberately not installed. EngineHandle grows fields as the
 * backend gains features, so keeping it out of the public API means a new
 * header can never disagree about its layout with an already-built backend
 * archive.
 */
#pragma once

#include "torch_tensorrt/executorch/OptimizationProfileSelection.h"
#include "torch_tensorrt/executorch/TensorRTBackend.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

struct EngineHandle {
  TRTLogger logger;
  TRTUniquePtr<nvinfer1::IRuntime> runtime;
  TRTUniquePtr<nvinfer1::ICudaEngine> engine;
  TRTUniquePtr<nvinfer1::IExecutionContext> exec_ctx;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  ProfileTable profiles;
  std::vector<void*> cached_input_ptrs;
  std::vector<size_t> cached_input_sizes;
  std::vector<void*> cached_output_ptrs;
  std::vector<size_t> cached_output_sizes;
  size_t num_inputs = 0;
  size_t num_outputs = 0;
  // Per output binding [0..num_outputs): index into input_binding_names of the
  // input it aliases (in-place KV-cache / user alias), or -1 for a normal output.
  // Built at init from the blob's aliased_io. The KV buffers are threaded by
  // ExecuTorch as caller-owned mutable-buffer delegate args (input AND aliased
  // output): execute() binds each aliased TRT output binding to its aliased
  // input's caller-provided pointer (in-place) and reflects the result into the
  // delegate output EValue, which ExecuTorch's write-back copy_ then reads.
  std::vector<int> output_aliased_input_idx;
  // Per input binding [0..num_inputs): true if any output aliases this input, so
  // its in-place (KV/user) update must land in the caller-owned storage. Built at
  // init from aliased_io; execute() uses it to reject a non-device-resident
  // aliased input instead of silently staging its update into delegate scratch.
  std::vector<bool> input_is_alias_target;
  size_t num_aliased_outputs = 0;
  int device_id = 0;
  bool unified_memory = false;
  std::mutex mu;
  // Makes the skip-sync fast path safe to reuse: TensorRT forbids reconfiguring or
  // destroying an execution context while one of its enqueues is in flight, so when
  // execute() returns without an end sync it records this event; the next execute()
  // and the destructor wait on it before touching exec_ctx. One event/flag pair
  // suffices because a handle runs on a single thread at a time.
  cudaEvent_t inflight_event = nullptr;
  bool inflight_pending = false;
  // A pin this engine cannot honor is a property of the caller's guard, not of the
  // call, so it would otherwise be reported identically on every execute(). One
  // engine, one report: a decode loop must not turn it into a log flood.
  bool pin_ignored_reported = false;

  ~EngineHandle();
};

} // namespace executorch_backend
} // namespace torch_tensorrt
