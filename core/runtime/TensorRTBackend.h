#pragma once

#include <executorch/runtime/backend/interface.h>
#include <vector>
#include "ATen/core/TensorBody.h"
#include "core/runtime/TRTEngine.h"
#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

/**
 * Thin backend that holds a TRT engine and runs inference by calling
 * execute_engine. Implements executorch::runtime::BackendInterface
 * (init sets up the engine, execute runs the engine).
 */
class TensorRTBackend final : public executorch::runtime::BackendInterface {
 public:
  TensorRTBackend() = default;

  explicit TensorRTBackend(c10::intrusive_ptr<TRTEngine> engine) : engine_(std::move(engine)) {}

  void set_engine(c10::intrusive_ptr<TRTEngine> engine) {
    engine_ = std::move(engine);
  }

  c10::intrusive_ptr<TRTEngine> get_engine() const {
    return engine_;
  }

  bool is_initialized() const {
    return engine_ != nullptr;
  }

  /**
   * Run inference: forwards to execute_engine(inputs, engine_).
   * Returns output tensors from the TRT engine.
   */
  std::vector<at::Tensor> execute(std::vector<at::Tensor> inputs) {
    TORCHTRT_CHECK(engine_ != nullptr, "TensorRTBackend: engine is null");
    return execute_engine(std::move(inputs), engine_);
  }

  // executorch::runtime::BackendInterface
  bool is_available() const override;
  executorch::runtime::Result<executorch::runtime::DelegateHandle*> init(
      executorch::runtime::BackendInitContext& context,
      executorch::runtime::FreeableBuffer* processed,
      executorch::runtime::ArrayRef<executorch::runtime::CompileSpec> compile_specs) const override;
  executorch::runtime::Error execute(
      executorch::runtime::BackendExecutionContext& context,
      executorch::runtime::DelegateHandle* handle,
      executorch::runtime::Span<executorch::runtime::EValue*> args) const override;
  void destroy(executorch::runtime::DelegateHandle* handle) const override;

 private:
  c10::intrusive_ptr<TRTEngine> engine_;
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
