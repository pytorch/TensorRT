#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace evaluators {

c10::optional<torch::jit::IValue> toIValue(const torch::jit::Value* v);
at::Tensor createTensorFromList(
    const torch::jit::IValue& data,
    const torch::jit::IValue& dtype,
    const torch::jit::IValue& device);

} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch