#pragma once

#include <cuda_runtime.h>
#include <executorch/runtime/backend/interface.h>

#include <executorch/vitExecutorchAdapter.h>

#include <memory>
#include <mutex>

namespace torch_tensorrt {
namespace executorch_backend {

struct EdgeLLMHandle {
  int device_id = 0;
  std::unique_ptr<trt_edgellm::executorch::VitExecutorchAdapter> vision_runner;
  std::mutex mu;
  cudaEvent_t inflight_event = nullptr;
  bool inflight_pending = false;

  ~EdgeLLMHandle();
};

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
};

} // namespace executorch_backend
} // namespace torch_tensorrt
