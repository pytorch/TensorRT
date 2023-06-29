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
       auto abs_tensor = add_abs(ctx, n, in, util::node_info(n));
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], abs_tensor);
       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       return true;
     }});

auto reciprocal_registration TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::reciprocal(Tensor self) -> Tensor", [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       if (in->getType() == nvinfer1::DataType::kINT32) {
         // pytorch implicitly casts to float for aten::reciprocal(int)
         in = castITensor(ctx, in, nvinfer1::DataType::kFLOAT);
       }
       auto unary_layer = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kRECIP);
       TORCHTRT_CHECK(unary_layer, "Unable to create recip layer from node: " << *n);
       unary_layer->setName(util::node_info(n).c_str());
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], unary_layer->getOutput(0));
       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       return true;
     }});

auto log2_registration TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::log2(Tensor self) -> Tensor", [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       const static float ln2 = 0.693147180559945309; // same constant onnx uses
       auto in = args[0].ITensorOrFreeze(ctx);
       auto tensor_type = util::TRTDataTypeToScalarType(in->getType());
       if (in->getType() == nvinfer1::DataType::kINT32) {
         // pytorch implicitly casts to float for aten::log2(int)
         in = castITensor(ctx, in, nvinfer1::DataType::kFLOAT);
         tensor_type = at::kFloat;
       }

       auto log_layer = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kLOG);
       TORCHTRT_CHECK(log_layer, "Unable to create log layer from node: " << *n);
       log_layer->setName((util::node_info(n) + "_log").c_str());

       std::vector<int64_t> ln2_dims(in->getDimensions().nbDims, 1);
       auto ln2_tensor = at::full(ln2_dims, ln2, at::TensorOptions().dtype(tensor_type));
       auto ln2_itensor = converters::tensor_to_const(ctx, ln2_tensor);

       auto div_layer = add_elementwise(
           ctx,
           nvinfer1::ElementWiseOperation::kDIV,
           log_layer->getOutput(0),
           ln2_itensor,
           (util::node_info(n) + "_div").c_str());
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], div_layer->getOutput(0));
       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       return true;
     }});

auto logical_not_registration TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::logical_not(Tensor self) -> Tensor", [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       if (in->getType() != nvinfer1::DataType::kBOOL) {
         // unary not layer only supports bool inputs
         in = castITensor(ctx, in, nvinfer1::DataType::kBOOL, util::node_info(n).c_str());
       }
       auto unary_layer = ctx->net->addUnary(*in, nvinfer1::UnaryOperation::kNOT);
       TORCHTRT_CHECK(unary_layer, "Unable to create logical_not layer from node: " << *n);
       unary_layer->setName(util::node_info(n).c_str());
       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], unary_layer->getOutput(0));
       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       return true;
     }});

auto isfinite_registration TORCHTRT_UNUSED = RegisterNodeConversionPatterns().pattern(
    {"aten::isfinite(Tensor self) -> Tensor", [](ConversionCtx* ctx, const torch::jit::Node* n, args& args) -> bool {
       auto in = args[0].ITensorOrFreeze(ctx);
       // assuming x-x = 0 for all values other than nan/inf/-inf where x-x = nan
       // x==x for all non-nan values
       auto inf_test_layer = ctx->net->addElementWise(*in, *in, nvinfer1::ElementWiseOperation::kSUB);
       TORCHTRT_CHECK(inf_test_layer, "Unable to create sub layer from node: " << *n);
       inf_test_layer->setName((util::node_info(n) + "_inf_test").c_str());
       auto inf_test_tensor = inf_test_layer->getOutput(0);

       auto nan_test_layer =
           ctx->net->addElementWise(*inf_test_tensor, *inf_test_tensor, nvinfer1::ElementWiseOperation::kEQUAL);
       TORCHTRT_CHECK(nan_test_layer, "Unable to create eq layer from node: " << *n);
       nan_test_layer->setName((util::node_info(n) + "_nan_test").c_str());

       auto out_tensor = ctx->AssociateValueAndTensor(n->outputs()[0], nan_test_layer->getOutput(0));
       LOG_DEBUG("Output tensor shape: " << out_tensor->getDimensions());
       return true;
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
convert(log, kLOG);
convert(ceil, kCEIL);
convert(sqrt, kSQRT);
convert(exp, kEXP);
convert(neg, kNEG);
convert(erf, kERF);
convert(sign, kSIGN);
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
