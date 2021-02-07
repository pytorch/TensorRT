#include "core/util/converter_util.h"

namespace trtorch {
namespace core {
namespace util {

nvinfer1::ITensor* arrToTensor(int32_t* dim, int rank, trtorch::core::conversion::ConversionCtx* ctx) {
  const nvinfer1::Dims d{1, {static_cast<int32_t>(rank)}};
  const nvinfer1::Weights w{nvinfer1::DataType::kINT32, dim, rank};
  return ctx->net->addConstant(d, w)->getOutput(0);
}

} // namespace util
} // namespace core
} // namespace trtorch