/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * ExecuTorch backend delegate that runs TensorRT engines serialized by
 * torch_tensorrt.  The processed blob must be in the vector-of-strings wire
 * format produced by
 *   py/torch_tensorrt/executorch/serialization.py::serialize_engine_info()
 * which maps 1-to-1 to the std::vector<std::string> accepted by
 *   core/runtime/TRTEngine::TRTEngine(std::vector<std::string>).
 */
#pragma once

#include <executorch/runtime/backend/interface.h>

namespace torch_tensorrt {
namespace executorch_backend {

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
