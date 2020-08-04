#pragma once

#include "core/util/prelude.h"
#include "core/conversion/conversionctx/ConversionCtx.h"

namespace trtorch {
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

inline nvinfer1::ITensor* tensor_to_const(ConversionCtx* ctx, at::Tensor t) {
    auto t_weights = Weights(ctx, t);
    auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
    TRTORCH_CHECK(const_layer, "Unable to freeze tensor");

    auto out = const_layer->getOutput(0);

    std::ostringstream tensor_id;
    tensor_id << reinterpret_cast<int*>(out);

    LOG_DEBUG(ctx->logger, "Freezing tensor " << tensor_id.str() << " as an IConstantLayer");
    const_layer->setName(("[Freeze Tensor " + tensor_id.str() + " ]").c_str());

    return out;
}


} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch