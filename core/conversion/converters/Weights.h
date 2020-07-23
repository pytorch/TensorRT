#pragma once

#include "core/util/prelude.h"
#include "core/conversion/conversionctx/ConversionCtx.h"

namespace trtorch {
namespace core {    
namespace conversion {
namespace converters {

struct Weights {
    //TODO: Rebuild this in a way that makes sense for more than just conv2/3D and linear
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

inline nvinfer1::ITensor* tensor_to_const(ConversionCtx* ctx, at::Tensor t) {
    auto t_weights = Weights(ctx, t);
    return ctx->net->addConstant(t_weights.shape, t_weights.data)->getOutput(0);
}

} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch