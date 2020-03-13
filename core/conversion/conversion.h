#pragma once

#include <map>

#include "NvInfer.h"
#include "torch/csrc/jit/ir.h"
#include "core/conversion/conversionctx/ConversionCtx.h"

namespace torch {
namespace jit {
namespace script {
const std::unordered_map<std::string, c10::TypePtr>& string_to_type_lut();
}
}
}

namespace trtorch {
namespace core {
namespace conversion {

struct InputRange {
    nvinfer1::Dims min;
    nvinfer1::Dims max;
    nvinfer1::Dims opt;
    nvinfer1::Dims input_shape;
    // Should we restrict to unsigned?
    InputRange(std::vector<int64_t> d);
    InputRange(std::vector<int64_t> min_shape,
               std::vector<int64_t> opt_shape,
               std::vector<int64_t> max_shape);
};

struct ExtraInfo {
    std::vector<InputRange> input_ranges;
    BuilderSettings engine_settings;
    ExtraInfo(std::vector<InputRange> input_ranges)
        : input_ranges(std::move(input_ranges)), engine_settings(BuilderSettings()) {}
};

using GraphParams = std::map<torch::jit::Value*, at::Tensor>;

GraphParams get_named_params(c10::ArrayRef<torch::jit::Value*> inputs, std::vector<at::Tensor> params);

// Converts a already lowered block (blocks with no sub blocks) to
// a serialized TensorRT engine that can be deserialized and run
std::string ConvertBlockToEngine(torch::jit::Block* b, ExtraInfo build_info, GraphParams& static_params);

bool OpSupported(const torch::jit::Node* n);

} // namespace conversion
} // namespace core
} // namespace trtorch
