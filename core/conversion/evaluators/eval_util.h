#pragma once

#include "core/conversion/evaluators/evaluators.h"
#include "torch/csrc/jit/ir/ir.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace evaluators {

nvinfer1::ITensor* index_layer(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    nvinfer1::ITensor* input_tensor,
    int64_t index);

c10::IValue dynamic_size_layer(ConversionCtx* ctx, const torch::jit::Node* n, kwargs& args);

c10::optional<torch::jit::IValue> toIValue(const torch::jit::Value* v);
at::Tensor createTensorFromList(
    const torch::jit::IValue& data,
    const torch::jit::IValue& dtype,
    const torch::jit::IValue& device);

int64_t normalizeIndex(int64_t idx, int64_t list_size);

at::Tensor scalar_to_tensor(const at::Scalar& s, const at::Device device = at::kCPU);

} // namespace evaluators
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
