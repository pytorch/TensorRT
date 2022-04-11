#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace evaluators {

c10::optional<torch::jit::IValue> toIValue(const torch::jit::Value* v);
at::Tensor createTensorFromList(
    const torch::jit::IValue& data,
    const torch::jit::IValue& dtype,
    const torch::jit::IValue& device);

at::Tensor scalar_to_tensor_util(const at::Scalar& s, const at::Device device = at::kCPU);

} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt