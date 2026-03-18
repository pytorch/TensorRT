#pragma once

#include <vector>
#include "ATen/core/TensorBody.h"
#include "core/runtime/TRTEngine.h"
#include "core/runtime/runtime.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace runtime {

/**
 * Minimal backend interface: initialized state and execution.
 * Mirrors the ExecuTorch BackendInterface pattern (init/execute/destroy).
 */
class BackendInterface {
 public:
  virtual ~BackendInterface() = default;
  virtual bool is_initialized() const = 0;
  virtual std::vector<at::Tensor> execute(std::vector<at::Tensor> inputs) = 0;
};

/**
 * Thin backend that holds a TRT engine and runs inference by calling
 * execute_engine. Mirrors the ExecuTorch TensorRTBackend pattern:
 * init sets up the engine, execute runs the engine.
 */
class TensorRTBackend final : public BackendInterface {
 public:
  TensorRTBackend() = default;

  explicit TensorRTBackend(c10::intrusive_ptr<TRTEngine> engine) : engine_(std::move(engine)) {}

  void set_engine(c10::intrusive_ptr<TRTEngine> engine) {
    engine_ = std::move(engine);
  }

  c10::intrusive_ptr<TRTEngine> get_engine() const {
    return engine_;
  }

  bool is_initialized() const override {
    return engine_ != nullptr;
  }

  /**
   * Run inference: forwards to execute_engine(inputs, engine_).
   * Returns output tensors from the TRT engine.
   */
  std::vector<at::Tensor> execute(std::vector<at::Tensor> inputs) override {
    TORCHTRT_CHECK(engine_ != nullptr, "TensorRTBackend: engine is null");
    return execute_engine(std::move(inputs), engine_);
  }

 private:
  c10::intrusive_ptr<TRTEngine> engine_;
};

} // namespace runtime
} // namespace core
} // namespace torch_tensorrt
