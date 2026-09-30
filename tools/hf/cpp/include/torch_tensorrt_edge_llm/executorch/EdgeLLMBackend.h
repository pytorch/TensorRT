#pragma once

#include <executorch/runtime/backend/interface.h>
#include <torch_tensorrt/executorch/TensorRTBackend.h>

namespace torch_tensorrt_edge_llm {
namespace executorch_backend {

class EdgeLLMBackend final : public ::executorch::runtime::BackendInterface {
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

 private:
  ::torch_tensorrt::executorch_backend::TensorRTBackend engine_backend_;
};

} // namespace executorch_backend
} // namespace torch_tensorrt_edge_llm
