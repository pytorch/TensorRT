/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt. The processed blob uses the standalone wire format from
 * py/torch_tensorrt/executorch/serialization.py and is parsed directly here.
 * This runtime path intentionally does not depend on the legacy
 * Torch-TensorRT C++ runtime or libtorch.
 */
#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <executorch/runtime/backend/interface.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace torch_tensorrt {
namespace executorch_backend {

struct TRTDeleter {
  template <typename T>
  void operator()(T* p) const {
    delete p;
  }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

class TRTLogger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override;
};

struct InputProfileBounds {
  nvinfer1::Dims min{};
  nvinfer1::Dims max{};
};

struct EngineHandle {
  TRTLogger logger;
  TRTUniquePtr<nvinfer1::IRuntime> runtime;
  TRTUniquePtr<nvinfer1::ICudaEngine> engine;
  TRTUniquePtr<nvinfer1::IExecutionContext> exec_ctx;
  std::vector<std::string> input_binding_names;
  std::vector<std::string> output_binding_names;
  std::vector<InputProfileBounds> input_profile_bounds;
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

  ~EngineHandle();
};

class TensorRTBackend final : public ::executorch::runtime::BackendInterface {
 public:
  bool is_available() const override;

  ::executorch::runtime::Result<::executorch::runtime::DelegateHandle*> init(
      ::executorch::runtime::BackendInitContext& context,
      ::executorch::runtime::FreeableBuffer* processed,
      ::executorch::runtime::ArrayRef<::executorch::runtime::CompileSpec> compile_specs) const override;

  // Runs the engine. With an executorch::extension::cuda::CallerStreamGuard active and
  // no host staging required, this may return while the enqueue is still in flight on
  // the selected stream, so the caller must keep device buffers alive and unmodified
  // past return, order any other stream against this one, and synchronize the stream
  // before reading device-resident outputs. The selected stream must be on the engine's
  // device, and calls on one handle must not overlap each other or its destruction.
  // Note that other CUDA delegates sharing the same guard may instead synchronize before
  // returning, so do not assume results are ready on return from this one.
  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
