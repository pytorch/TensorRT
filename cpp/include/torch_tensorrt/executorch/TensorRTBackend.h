/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt. The processed blob uses the standalone TR01 wire format from
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

  ::executorch::runtime::Error execute(
      ::executorch::runtime::BackendExecutionContext& context,
      ::executorch::runtime::DelegateHandle* handle,
      ::executorch::runtime::Span<::executorch::runtime::EValue*> args) const override;

  void destroy(::executorch::runtime::DelegateHandle* handle) const override;
};

} // namespace executorch_backend
} // namespace torch_tensorrt
