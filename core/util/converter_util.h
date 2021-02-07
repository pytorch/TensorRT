#pragma once

#include "NvInfer.h"
#include "core/conversion/conversionctx/ConversionCtx.h"

namespace trtorch {
namespace core {
namespace util {

nvinfer1::ITensor* arrToTensor(int32_t* dim, int rank, trtorch::core::conversion::ConversionCtx* ctx);

} // namespace util
} // namespace core
} // namespace trtorch