#pragma once
#include "torch/csrc/jit/api/module.h"
#include "torch/csrc/jit/backends/backend.h"
#include "torch/csrc/jit/backends/backend_debug_handler.h"
#include "torch/csrc/jit/backends/backend_preprocess.h"

namespace trtorch {
namespace backend {

class TensorRTBackend : public torch::jit::PyTorchBackendInterface {
 public:
  explicit TensorRTBackend() {}
  virtual ~TensorRTBackend() = default;

  bool is_available() override {
    return true;
  }

  c10::impl::GenericDict compile(c10::IValue processed_mod, c10::impl::GenericDict method_compile_spec) override;
  c10::impl::GenericList execute(c10::IValue handle, c10::impl::GenericList inputs) override;
};

} // namespace backend
} // namespace trtorch