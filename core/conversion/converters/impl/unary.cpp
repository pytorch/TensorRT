#include "core/conversion/converters/converters.h"
#include "core/util/prelude.h"

#include "torch/torch.h"

namespace torch_tensorrt {
namespace core {
namespace conversion {
namespace converters {
namespace impl {
namespace {

auto abs_registration TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::abs(Tensor self) -> Tensor", [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       bool unary_supported_input = in->getType() == nvinfer1::DataType::kFLOAT ||
           in->getType() == nvinfer1::DataType::kHALF || in->getType() == nvinfer1::DataType::kINT8;
       if (unary_supported_input) {
         auto unary_layer = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kABS);
         TORCHTRT_CHECK(unary_layer, "Unable to create abs layer from node: " << *n);
         unary_layer->setName(util::node_info(n).c_str());
         auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], unary_layer->getOutput(0));
         LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
         return true;
       } else {
         LOG_GRAPH(
             "Tensor is of unsupported type "
             << in->getType() << " for IUnaryLayer::kABS. Using backup implementation via IElementWise (max(x, -x)");
         // For types not supported by kABS, use an elementwise implementation abs(x) = max(x, -1 * x)
         at::Tensor neg_one = torch::full({1}, -1).to(util::TRTDataTypeToScalarType(in->getType()));
         auto neg_one_const = tensor_to_const(ctx, neg_one);
         auto neg_layer = add_elementwise(
             ctx,
             nvinfer1::ElementWiseOperation::kPROD,
             in,
             neg_one_const,
             util::node_info(n) + std::string("_Negation"));
         TORCHTRT_CHECK(neg_layer, "Unable to create prod layer from node: " << *n);
         auto max_layer = add_elementwise(
             ctx,
             nvinfer1::ElementWiseOperation::kMAX,
             in,
             neg_layer->getOutput(0),
             util::node_info(n) + std::string("_Max"));
         TORCHTRT_CHECK(max_layer, "Unable to create max layer from node: " << *n);
         auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], max_layer->getOutput(0));
         LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
         return true;
       }
     }});

#define convert(unary, trt_type)                                                               \
  auto unary##_registrations TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(       \
      {"aten::" #unary "(Tensor self) -> Tensor",                                              \
       [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {                 \
         auto in = args[0].ITensorOrFreeze(ctx);                                               \
         auto unary = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::trt_type);             \
                                                                                               \
         TORCHTRT_CHECK(unary, "Unable to create " #unary " layer from node: " << *n);         \
                                                                                               \
         unary->setName(util::node_info(n).c_str());                                           \
         auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], unary->getOutput(0)); \
         LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());                    \
                                                                                               \
         return true;                                                                          \
       }});

convert(cos, kCOS);
convert(acos, kACOS);
convert(cosh, kCOSH);
convert(sin, kSIN);
convert(asin, kASIN);
convert(sinh, kSINH);
convert(tan, kTAN);
convert(atan, kATAN);
convert(floor, kFLOOR);
convert(reciprocal, kRECIP);
convert(log, kLOG);
convert(ceil, kCEIL);
convert(sqrt, kSQRT);
convert(exp, kEXP);
convert(neg, kNEG);
convert(erf, kERF);
convert(asinh, kASINH);
convert(acosh, kACOSH);
convert(atanh, kATANH);

#undef convert

} // namespace
} // namespace impl
} // namespace converters
} // namespace conversion
} // namespace core
} // namespace torch_tensorrt
