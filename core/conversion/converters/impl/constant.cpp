#pragma once
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {
auto constant_registrations = RegisterNodeConversionPatterns()
    .pattern({
        "trt::const(Tensor self) -> Tensor",
        [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
            // This converter may be abusing what the registry is supposed to be used for
            // Fundimentally this is because of the differing philosophies between
            // TensorRT and PyTorch, i.e. Variables contain Tensors vs. just Tensors

            auto t = args[0].unwrapToTensor();
            auto t_weights = Weights(ctx, t);
            auto const_layer = ctx->net->addConstant(t_weights.shape, t_weights.data);
            const_layer->setName(util::node_info(n).c_str());
            auto out_value = n->outputs()[0];
            auto out_tensor = const_layer->getOutput(0);
            out_tensor->setName(out_value->debugName().c_str());
            ctx->value_tensor_map[out_value] = out_tensor;
            LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
            
            return true;
        }
    });
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch 

