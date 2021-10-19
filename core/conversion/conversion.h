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
  ir::InputSpecMap inputs;
  BuilderSettings engine_settings;
};

// Converts a already lowered block (blocks with no sub blocks) to
// a serialized TensorRT engine that can be deserialized and run
std::string ConvertBlockToEngine(
    const torch::jit::Block* b,
    ConversionInfo build_info,
    ir::StaticParams& static_params);

bool OpSupported(const torch::jit::Node* n);

bool VerifyConverterSupportForBlock(const torch::jit::Block* b, bool suppress_errors=false);

c10::optional<torch::jit::IValue> EvaluateNode(
    ConversionCtx* ctx,
    const torch::jit::Node* n,
    int level = 0,
    int limit = 10);

} // namespace conversion
} // namespace core
} // namespace trtorch
