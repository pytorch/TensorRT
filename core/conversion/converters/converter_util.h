#pragma once

#include <map>
#include <string>

#include "core/conversion/conversionctx/ConversionCtx.h"
#include "core/conversion/converters/Weights.h"
#include "core/conversion/var/Var.h"
#include "core/util/prelude.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
nvinfer1::ILayer* addPaddingLayer(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int nDim, bool trailing=true, bool use_zeros=true);
nvinfer1::ILayer* addUnpaddingLayer(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int nDim, bool trailing=true, bool use_zeros=true);
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
