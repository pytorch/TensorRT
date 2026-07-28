#include <limits>
#include <vector>
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
         std::vector<int64_t> singleton_dims(in->getDimensions().nbDims, 1);
         auto options = torch::TensorOptions().dtype(torch::kFloat32);
         auto zero = tensor_to_const(ctx, torch::full(singleton_dims, 0.0f, options), util::node_info(n) + "_zero");
         auto neg_inf = tensor_to_const(
             ctx,
             torch::full(singleton_dims, -std::numeric_limits<float>::infinity(), options),
             util::node_info(n) + "_neg_inf");
         auto select_layer = ctx->net->addSelect(*in, *zero, *neg_inf);
         TORCHTRT_CHECK(select_layer, "Unable to create select layer for attn_bias_from_attn_mask");
         select_layer->setName(util::node_info(n).c_str());
         out = select_layer->getOutput(0);
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
