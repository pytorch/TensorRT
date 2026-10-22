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

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

class ExecutionGraph;

struct EngineHandle {
  EngineHandle();
  std::unique_ptr<ExecutionGraph> execution_graph;
  // Shared with every other handle loaded with kSharedEnginesKey on from the same engine bytes, for
  // the same device and weight streaming request, so the weights are on the device once. The context
  // and everything below it stay this handle's own.
  std::shared_ptr<nvinfer1::ICudaEngine> engine;
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
  // Built at init from the blob's aliased_io. Either way execute() binds the
  // aliased TRT output binding to its aliased input's caller-provided pointer,
  // so the engine's write lands in the caller's buffer; what differs is how the
  // .pte carries the buffer. Threaded: the buffer is both a delegate input arg
  // and a delegate output arg (the caller-owned mutable buffer's mutation slot),
  // and execute() reflects the result into that output EValue for ExecuTorch's
  // write-back copy_ to read. Elided -- zero-copy KV -- the buffer is an input
  // arg only, the delegate has no output for it, and execute() skips the
  // reflect: the in-place write already is the update.
  std::vector<int> output_aliased_input_idx;
  // Per input binding [0..num_inputs): true if any output aliases this input, so
  // its in-place (KV/user) update must land in the caller-owned storage. Built at
  // init from aliased_io; execute() uses it to reject a non-device-resident
  // aliased input instead of silently staging its update into delegate scratch.
  std::vector<bool> input_is_alias_target;
  size_t num_aliased_outputs = 0;
  int device_id = 0;
  // Whether this device can reach pageable host memory through shared host page tables, which is the
  // question the uses of this flag ask. A discrete card answers no: it can reach pageable memory, but
  // only by faulting pages in one at a time, so it takes the staged-copy path instead.
  bool pageable_host_access = false;
  // Whether exec_ctx was created kUSER_MANAGED, because
  // kSharedActivationScratchKey was on at this handle's load. Such a context takes
  // its activation scratch from the shared per-device pool on every call, unless
  // claims_pooled_scratch below is false, in which case the engine needs none and
  // never claims from the pool at all.
  bool shared_scratch = false;
  // Whether this handle takes its activation scratch from the pool. Set at init
  // only where shared_scratch above is on, and then only where the engine reports
  // needing activation scratch under some shape; a handle loaded with the option
  // off keeps this false and says nothing about what its engine needs, because it
  // has its own kSTATIC scratch either way. It is a predicate and not a size
  // because the size is never the right amount to install: enqueueV3 refuses a
  // kUSER_MANAGED context with no device memory once the engine needs any, however
  // little the shapes bound to a given call need, so execute() must hand such a
  // call some buffer -- but the engine's own figure covers every shape in the
  // profile, and installing it would size the pool for the largest of them.
  bool claims_pooled_scratch = false;
  std::mutex mu;
  // A pin this engine cannot honor is a property of the caller's guard, not of the
  // call, so it would otherwise be reported identically on every execute(). One
  // engine, one report: a decode loop must not turn it into a log flood.
  bool pin_ignored_reported = false;

  ~EngineHandle();
};

} // namespace executorch_backend
} // namespace torch_tensorrt
