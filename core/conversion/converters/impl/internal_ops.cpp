#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"
#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto linear_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"trt::attn_bias_from_attn_mask(Tensor attn_mask) -> Tensor",
     [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       // Converter for internal op used in unpack_scaled_dot_product_attention
       // We don't have visibility to check types during lowering and can't introduce conditionals so do type specific
       // specialization here
       auto in = args[0].ITensorOrFreeze(ctx);
       auto out = in;
       if (in->getType() == nvinfer1::DataType::kBOOL) {
         auto not_layer = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kNOT);
         TORCHTRT_CHECK(not_layer, "Unable to create not layer for attn_bias_from_attn_mask");
         not_layer->setName((util::node_info(n) + "_not").c_str());
         auto neg_inf = torch::tensor(-std::numeric_limits<float>::infinity());
         auto neg_inf_itensor = tensor_to_const(ctx, neg_inf);
         auto prod_layer = add_elementwise(
             ctx,
             nvinfer1::ElementWiseOperation::kPROD,
             not_layer->getOutput(0),
             neg_inf_itensor,
             util::node_info(n) + "_mul");
         auto add_layer = add_elementwise(
             ctx, nvinfer1::ElementWiseOperation::kSUM, prod_layer->getOutput(0), in, util::node_info(n) + "_add");
         out = add_layer->getOutput(0);
       }
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], out);
       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       LOG_DEBUG("Output tensor type: " << out_tensor->getType());
       return true;
     }});
} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
