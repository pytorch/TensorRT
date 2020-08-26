#pragma once

#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace evaluators {

c10::optional<torch::jit::IValue> toIValue(const torch::jit::Value* v);

} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace trtorch