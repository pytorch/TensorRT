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
namespace util {
    using namespace core::util;
    nvinfer1::ILayer* padTensorDim(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int dim);
    nvinfer1::ILayer* unpadTensorDim(ConversionCtx* ctx, const torch::jit::Node* n, nvinfer1::ITensor* tensor, int dim);
} // namespace util
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch
