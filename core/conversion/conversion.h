#pragma once

#include <map>

#include "NvInfer.h"
#include "core/conversion/conversionctx/ConversionCtx.h"
#include "core/ir/ir.h"
#include "torch/csrc/jit/ir/ir.h"

namespace trtorch {
namespace core {
namespace conversion {

struct ConversionInfo {
  std::vector<ir::Input> inputs;
  BuilderSettings engine_settings;
  ConversionInfo(std::vector<ir::Input> inputs) : inputs(std::move(inputs)), engine_settings(BuilderSettings()) {}
};

// TODO: REMOVE GRAPH AND PARAMS AND MOVE FULLY TO INLINED CONSTANTS

using GraphParams = std::map<torch::jit::Value*, torch::jit::IValue>;

GraphParams get_named_params(c10::ArrayRef<torch::jit::Value*> inputs, std::vector<torch::jit::IValue> params);

// Converts a already lowered block (blocks with no sub blocks) to
// a serialized TensorRT engine that can be deserialized and run
std::string ConvertBlockToEngine(const torch::jit::Block* b, ConversionInfo build_info, GraphParams& static_params);

bool OpSupported(const torch::jit::Node* n);

bool VerifyConverterSupportForBlock(const torch::jit::Block* b);

c10::optional<torch::jit::IValue> EvaluateNode(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    int level = 0,
    int limit = 10);

} // namespace conversion
} // namespace core
} // namespace trtorch
