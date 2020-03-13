#pragma once
#include "core/util/prelude.h"
#include "core/conversion/converters/converters.h"

namespace trtorch {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

bool ConvertConvBatchNorm(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {
    auto input = args[0].ITensor();
    auto shape = util::toVec(input->getDimensions());
    LOG_WARNING("Assuming channel dimension is 3 because input is from a conv layer, please verify");
    auto gamma = args[1].unwrapToTensor(at::full({shape[shape.size() - 3]}, 1));
    auto beta = args[2].unwrapToTensor(at::full({shape[shape.size() - 3]}, 1));
    auto mean = args[3].unwrapToTensor(at::full({shape[shape.size() - 3]}, 0));
    auto var = args[4].unwrapToTensor(at::full({shape[shape.size() - 3]}, 0));
    LOG_WARNING("Momentum argument is disregarded");
    //auto momentum = args[6].unwrapToDouble(0);
    auto eps = args[7].unwrapToDouble(0);

    auto w = at::diag(gamma / at::sqrt(var + eps));
    auto w_shape = w.sizes().vec();
    w_shape.push_back(1);
    w_shape.push_back(1);
    w = w.reshape(w_shape);
    auto b = beta - gamma * (mean / at::sqrt(var + eps));

    auto weights = Weights(ctx, w);
    auto bias = Weights(ctx, b);

    auto bn_as_conv = ctx->net->addConvolutionNd(*input, weights.num_output_maps, weights.kernel_shape, weights.data, bias.data);
    
    bn_as_conv->setName(util::node_info(n).c_str());
    auto out_value = n->outputs()[0];
    auto out_tensor = bn_as_conv->getOutput(0);
    out_tensor->setName(out_value->debugName().c_str());
    ctx->value_tensor_map[out_value] = out_tensor;
    LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
    return true;
}

bool ConvertLinearBatchNorm(ConversionCtx* ctx, const torch::jit::Node* n, args& args) {
    auto input = args[0].ITensor();
    auto shape = util::toVec(input->getDimensions());
    auto gamma = args[1].unwrapToTensor(at::full({shape},1));
    auto beta = args[2].unwrapToTensor(at::full({shape},1));
    auto mean = args[3].unwrapToTensor(at::full({shape},0));
    auto var = args[4].unwrapToTensor(at::full({shape},0));
    LOG_WARNING("Momentum argument is disregarded");
    //auto momentum = args[6].unwrapToDouble(0);
    auto eps = args[7].unwrapToDouble(0);

    auto mean_ = tensor_to_const(ctx, mean);
    auto bot_half = at::sqrt(var + eps);
    auto bot_half_ = tensor_to_const(ctx, bot_half);
    auto gamma_ = tensor_to_const(ctx, gamma);
    auto beta_ = tensor_to_const(ctx, beta);

    auto top_half = ctx->net->addElementWise(*input, *mean_, nvinfer1::ElementWiseOperation::kSUB);
    auto top_half_out = top_half->getOutput(0);
    auto x_hat = ctx->net->addElementWise(*top_half_out, *bot_half_, nvinfer1::ElementWiseOperation::kDIV);
    auto x_hat_out = x_hat->getOutput(0);
    auto bn_scaled = ctx->net->addElementWise(*gamma_, *x_hat_out, nvinfer1::ElementWiseOperation::kPROD);
    auto bn_scaled_out = bn_scaled->getOutput(0);
    auto bn_biased = ctx->net->addElementWise(*beta_, *bn_scaled_out, nvinfer1::ElementWiseOperation::kSUM);
    auto bn_biased_out = bn_biased->getOutput(0);

    bn_biased->setName(util::node_info(n).c_str());
    auto out_value = n->outputs()[0];
    bn_biased_out->setName(out_value->debugName().c_str());
    ctx->value_tensor_map[out_value] = bn_biased_out;
    return true;
}

volatile auto batch_norm_registrations = RegisterNodeConversionPatterns()
    .pattern({
            R"SIG(aten::batch_norm(Tensor input, Tensor? gamma, Tensor? beta, 
                               Tensor? mean, Tensor? var, 
                               bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor))SIG",
            [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
                auto input = args[0].ITensor();
                auto shape = input->getDimensions();
                auto gamma = args[1].unwrapToTensor();
                
                if (/*training*/ args[5].unwrapToBool()) {
                    LOG_WARNING("TensorRT only converts forward pass of graphs, but saw training = True, may see undefined behavior, consider placing module in eval mode");
                }
                
                // If gamma is None this fails
                if (util::volume(shape) == gamma.numel()) {
                    return ConvertLinearBatchNorm(ctx, n, args);
                } else {
                    return ConvertConvBatchNorm(ctx, n, args);
                }
            }
        });


} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace trtorch 
