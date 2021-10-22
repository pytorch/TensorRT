#pragma once

#include "core/conversion/conversionctx/ConversionCtx.h"
#include "core/util/prelude.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {

struct Weights {
  nvinfer1::Weights data;
  nvinfer1::Dims kernel_shape;
  nvinfer1::Dims shape;
  int64_t num_input_maps;
  int64_t num_output_maps;

  Weights();
  Weights(ConversionCtx* ctx, at::Tensor t);
  Weights(ConversionCtx* ctx, float val);
  Weights(ConversionCtx* ctx, int32_t val);
  friend std::ostream& operator<<(std::ostream& os, const Weights& w);
};

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
